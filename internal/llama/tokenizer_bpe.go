package llama

import (
	"bytes"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"

	"go_llama/internal/gguf"
)

// Tokenizer implements GPT-2 BPE (as indicated by gguf key tokenizer.ggml.model == "gpt2").
// It uses the merges table stored in tokenizer.ggml.merges and a byte<->unicode reversible mapping.

type Tokenizer struct {
	TokenStrings []string
	Vocab        map[string]int32
	BpeRanks     map[string]uint32 // key: a + "\x00" + b
	Cache        map[string][]string

	byteToRune [256]rune
	runeToByte map[rune]byte

	// pretokenization regex (approx for llama-bpe)
	re *regexp.Regexp

	Specials []string // tokens starting with <|

	BOS int32
	EOS int32
	EOT int32
}

func LoadTokenizerFromGGUF(g *gguf.File) (*Tokenizer, error) {
	model, _ := g.GetString("tokenizer.ggml.model")
	if model != "gpt2" {
		return nil, fmt.Errorf("unsupported tokenizer model %q", model)
	}

	toks, ok := g.GetStrings("tokenizer.ggml.tokens")
	if !ok || len(toks) == 0 {
		return nil, fmt.Errorf("gguf missing tokenizer.ggml.tokens")
	}
	merges, ok := g.GetStrings("tokenizer.ggml.merges")
	if !ok || len(merges) == 0 {
		return nil, fmt.Errorf("gguf missing tokenizer.ggml.merges")
	}

	bos := int32(-1)
	eos := int32(-1)
	eot := int32(-1)
	if v, ok := g.GetU32("tokenizer.ggml.bos_token_id"); ok {
		bos = int32(v)
	}
	if v, ok := g.GetU32("tokenizer.ggml.eos_token_id"); ok {
		eos = int32(v)
	}
	if v, ok := g.GetU32("tokenizer.ggml.eot_token_id"); ok {
		eot = int32(v)
	}

	vocab := make(map[string]int32, len(toks))
	specials := make([]string, 0, 128)
	for i, s := range toks {
		vocab[s] = int32(i)
		if strings.HasPrefix(s, "<|") {
			specials = append(specials, s)
		}
	}
	// longest-first to make matching deterministic
	sort.Slice(specials, func(i, j int) bool { return len(specials[i]) > len(specials[j]) })

	ranks := make(map[string]uint32, len(merges))
	for i, m := range merges {
		// merges are of form: "a b"
		sp := strings.SplitN(m, " ", 2)
		if len(sp) != 2 {
			continue
		}
		key := sp[0] + "\x00" + sp[1]
		ranks[key] = uint32(i)
	}

	t := &Tokenizer{
		TokenStrings: toks,
		Vocab:        vocab,
		BpeRanks:     ranks,
		Cache:        make(map[string][]string, 1<<14),
		runeToByte:   make(map[rune]byte, 256),
		Specials:     specials,
		BOS:          bos,
		EOS:          eos,
		EOT:          eot,
	}
	t.initByteUnicode()

	// "llama-bpe" uses a GPT-2-like pretokenization. Go regexp has no lookarounds,
	// so we use a simpler RE2-compatible pattern that still works well.
	//
	// Pieces are sequences with optional leading space.
	t.re = regexp.MustCompile(`(?i:'s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`)
	return t, nil
}

func (t *Tokenizer) initByteUnicode() {
	// bytes_to_unicode from GPT-2 BPE
	bs := make([]int, 0, 256)
	for i := 33; i <= 126; i++ {
		bs = append(bs, i)
	}
	for i := 161; i <= 172; i++ {
		bs = append(bs, i)
	}
	for i := 174; i <= 255; i++ {
		bs = append(bs, i)
	}
	present := make([]bool, 256)
	for _, b := range bs {
		present[b] = true
	}

	cs := make([]int, len(bs))
	copy(cs, bs)
	n := 0
	for b := 0; b < 256; b++ {
		if !present[b] {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}
	for i := 0; i < 256; i++ {
		r := rune(cs[i])
		b := byte(bs[i])
		t.byteToRune[b] = r
		t.runeToByte[r] = b
	}
}

func (t *Tokenizer) Encode(text string, addBOS bool) ([]int32, error) {
	out := make([]int32, 0, len(text)/4)
	if addBOS && t.BOS >= 0 {
		out = append(out, t.BOS)
	}

	for len(text) > 0 {
		// Special token fast-path
		matched := false
		for _, sp := range t.Specials {
			if strings.HasPrefix(text, sp) {
				id, ok := t.Vocab[sp]
				if !ok {
					return nil, fmt.Errorf("special token not in vocab: %q", sp)
				}
				out = append(out, id)
				text = text[len(sp):]
				matched = true
				break
			}
		}
		if matched {
			continue
		}

		// Take a chunk until next special token occurrence.
		next := len(text)
		for _, sp := range t.Specials {
			if idx := strings.Index(text, sp); idx >= 0 && idx < next {
				next = idx
			}
		}
		chunk := text[:next]
		text = text[next:]

		pieces := t.re.FindAllString(chunk, -1)
		for _, p := range pieces {
			enc := t.byteEncode(p)
			subs := t.bpe(enc)
			for _, s := range subs {
				id, ok := t.Vocab[s]
				if !ok {
					return nil, fmt.Errorf("token not in vocab: %q", s)
				}
				out = append(out, id)
			}
		}
	}
	return out, nil
}

func (t *Tokenizer) byteEncode(s string) string {
	b := []byte(s)
	var sb strings.Builder
	sb.Grow(len(b) * 2)
	for _, x := range b {
		sb.WriteRune(t.byteToRune[x])
	}
	return sb.String()
}

func (t *Tokenizer) bpe(token string) []string {
	if v, ok := t.Cache[token]; ok {
		return v
	}

	// split into runes
	word := make([]string, 0, len(token))
	for _, r := range token {
		word = append(word, string(r))
	}
	if len(word) == 1 {
		res := []string{word[0]}
		t.Cache[token] = res
		return res
	}

	pairs := getPairs(word)
	for {
		minRank := uint32(^uint32(0))
		bestA := ""
		bestB := ""
		for k := range pairs {
			r, ok := t.BpeRanks[k]
			if ok && r < minRank {
				minRank = r
				ab := strings.SplitN(k, "\x00", 2)
				bestA, bestB = ab[0], ab[1]
			}
		}
		if bestA == "" {
			break
		}

		newWord := make([]string, 0, len(word))
		i := 0
		for i < len(word) {
			j := indexOfPair(word, bestA, bestB, i)
			if j == -1 {
				newWord = append(newWord, word[i:]...)
				break
			}
			newWord = append(newWord, word[i:j]...)
			newWord = append(newWord, bestA+bestB)
			i = j + 2
		}
		word = newWord
		if len(word) == 1 {
			break
		}
		pairs = getPairs(word)
	}

	res := make([]string, len(word))
	copy(res, word)
	// cache with a cap to avoid unbounded growth
	if len(t.Cache) < 200000 {
		t.Cache[token] = res
	}
	return res
}

func getPairs(word []string) map[string]struct{} {
	pairs := make(map[string]struct{}, len(word))
	for i := 0; i < len(word)-1; i++ {
		pairs[word[i]+"\x00"+word[i+1]] = struct{}{}
	}
	return pairs
}

func indexOfPair(word []string, a, b string, start int) int {
	for i := start; i < len(word)-1; i++ {
		if word[i] == a && word[i+1] == b {
			return i
		}
	}
	return -1
}

func (t *Tokenizer) Decode(ids []int32) string {
	var buf bytes.Buffer
	for _, id := range ids {
		buf.Write(t.DecodeToken(id))
	}
	return buf.String()
}

func (t *Tokenizer) DecodeToken(id int32) []byte {
	if id < 0 || int(id) >= len(t.TokenStrings) {
		return nil
	}
	s := t.TokenStrings[id]
	// Special tokens are already plain text.
	if strings.HasPrefix(s, "<|") {
		return []byte(s)
	}

	// Convert byte-unicode runes back to raw bytes.
	out := make([]byte, 0, len(s))
	for len(s) > 0 {
		r, size := utf8.DecodeRuneInString(s)
		if r == utf8.RuneError && size == 1 {
			// invalid rune, treat as byte
			out = append(out, s[0])
			s = s[1:]
			continue
		}
		if b, ok := t.runeToByte[r]; ok {
			out = append(out, b)
		} else {
			var tmp [utf8.UTFMax]byte
			n := utf8.EncodeRune(tmp[:], r)
			out = append(out, tmp[:n]...)
		}
		s = s[size:]
	}
	return out
}
