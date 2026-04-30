package llama

import (
	"fmt"
	"strconv"

	"go_llama/internal/gguf"
)

type HyperParams struct {
	VocabSize int
	NLayers   int
	NEmb      int
	NHeads    int
	NHeadsKV  int
	NFF       int
	RmsEps    float32
	RopeBase  float32
	RopeDim   int
	CtxLen    int
}

type LayerWeights struct {
	AttnNorm Tensor
	FFNNorm  Tensor

	Wq Tensor
	Wk Tensor
	Wv Tensor
	Wo Tensor

	Wup   Tensor
	Wgate Tensor
	Wdown Tensor
}

type Model struct {
	File *GGUFModelFile
	HP   HyperParams
	Tok  *Tokenizer

	TokenEmb Tensor
	OutNorm  Tensor
	OutProj  Tensor

	Layers []LayerWeights

	AttnNormW [][]float32
	FFNNormW  [][]float32
	OutNormW  []float32

	Pool *WorkerPool
}

func LoadModel(path string, threads int) (*Model, error) {
	f, err := OpenGGUF(path)
	if err != nil {
		return nil, err
	}

	hp, err := readHyperParams(f.GGUF)
	if err != nil {
		f.Close()
		return nil, err
	}

	tok, err := readTokenizer(f.GGUF)
	if err != nil {
		f.Close()
		return nil, err
	}

	get := func(name string) (Tensor, error) {
		t, ok := f.ByName[name]
		if !ok {
			return Tensor{}, fmt.Errorf("missing tensor %q", name)
		}
		return t, nil
	}

	m := &Model{File: f, HP: hp, Tok: tok, Pool: NewWorkerPool(threads)}

	if m.TokenEmb, err = get("token_embd.weight"); err != nil {
		f.Close()
		return nil, err
	}
	if m.OutNorm, err = get("output_norm.weight"); err != nil {
		f.Close()
		return nil, err
	}
	if m.OutProj, err = get("output.weight"); err != nil {
		// Some GGUFs omit output.weight and use tied weights.
		m.OutProj = m.TokenEmb
	}

	m.Layers = make([]LayerWeights, m.HP.NLayers)
	m.AttnNormW = make([][]float32, m.HP.NLayers)
	m.FFNNormW = make([][]float32, m.HP.NLayers)
	for i := 0; i < m.HP.NLayers; i++ {
		prefix := "blk." + strconv.Itoa(i) + "."
		lw := LayerWeights{}
		if lw.AttnNorm, err = get(prefix + "attn_norm.weight"); err != nil {
			f.Close()
			return nil, err
		}
		if lw.FFNNorm, err = get(prefix + "ffn_norm.weight"); err != nil {
			f.Close()
			return nil, err
		}
		if lw.Wq, err = get(prefix + "attn_q.weight"); err != nil {
			f.Close()
			return nil, err
		}
		if lw.Wk, err = get(prefix + "attn_k.weight"); err != nil {
			f.Close()
			return nil, err
		}
		if lw.Wv, err = get(prefix + "attn_v.weight"); err != nil {
			f.Close()
			return nil, err
		}
		if lw.Wo, err = get(prefix + "attn_output.weight"); err != nil {
			f.Close()
			return nil, err
		}
		if lw.Wup, err = get(prefix + "ffn_up.weight"); err != nil {
			f.Close()
			return nil, err
		}
		if lw.Wgate, err = get(prefix + "ffn_gate.weight"); err != nil {
			f.Close()
			return nil, err
		}
		if lw.Wdown, err = get(prefix + "ffn_down.weight"); err != nil {
			f.Close()
			return nil, err
		}
		m.Layers[i] = lw

		if m.AttnNormW[i], err = readVectorAnyToF32(lw.AttnNorm, m.HP.NEmb); err != nil {
			f.Close()
			return nil, err
		}
		if m.FFNNormW[i], err = readVectorAnyToF32(lw.FFNNorm, m.HP.NEmb); err != nil {
			f.Close()
			return nil, err
		}
	}

	if m.OutNormW, err = readVectorAnyToF32(m.OutNorm, m.HP.NEmb); err != nil {
		f.Close()
		return nil, err
	}

	return m, nil
}

func (m *Model) Close() error {
	if m.Pool != nil {
		m.Pool.Close()
	}
	if m.File != nil {
		return m.File.Close()
	}
	return nil
}

func readHyperParams(g *gguf.File) (HyperParams, error) {
	var hp HyperParams

	// Required
	if v, ok := g.GetU32("llama.vocab_size"); ok {
		hp.VocabSize = int(v)
	}
	if v, ok := g.GetU32("llama.block_count"); ok {
		hp.NLayers = int(v)
	}
	if v, ok := g.GetU32("llama.embedding_length"); ok {
		hp.NEmb = int(v)
	}
	if v, ok := g.GetU32("llama.feed_forward_length"); ok {
		hp.NFF = int(v)
	}
	if v, ok := g.GetU32("llama.attention.head_count"); ok {
		hp.NHeads = int(v)
	}
	if v, ok := g.GetU32("llama.attention.head_count_kv"); ok {
		hp.NHeadsKV = int(v)
	}
	if v, ok := g.GetF32("llama.norm_rms_eps"); ok {
		hp.RmsEps = v
	} else {
		hp.RmsEps = 1e-5
	}
	if v, ok := g.GetF32("llama.rope.freq_base"); ok {
		hp.RopeBase = v
	} else {
		hp.RopeBase = 10000
	}
	if v, ok := g.GetU32("llama.rope.dimension_count"); ok {
		hp.RopeDim = int(v)
	}
	if v, ok := g.GetU32("llama.context_length"); ok {
		hp.CtxLen = int(v)
	} else {
		hp.CtxLen = 2048
	}

	if hp.VocabSize == 0 || hp.NLayers == 0 || hp.NEmb == 0 || hp.NHeads == 0 || hp.NFF == 0 {
		return HyperParams{}, fmt.Errorf("missing required llama.* hyperparams in gguf")
	}
	if hp.NHeadsKV == 0 {
		hp.NHeadsKV = hp.NHeads
	}
	if hp.RopeDim == 0 {
		hp.RopeDim = hp.NEmb / hp.NHeads
	}
	return hp, nil
}

func readTokenizer(g *gguf.File) (*Tokenizer, error) {
	return LoadTokenizerFromGGUF(g)
}
