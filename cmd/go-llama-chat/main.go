package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"go_llama/internal/downloader"
	"go_llama/internal/llama"
)

const defaultModelURL = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf"

func main() {
	// Fixed for this machine: 12 hardware threads.
	// Set this before flag defaults that depend on GOMAXPROCS.
	runtime.GOMAXPROCS(12)

	var (
		modelURL     = flag.String("model-url", defaultModelURL, "URL do arquivo .gguf")
		modelPath    = flag.String("model-path", "", "Caminho local do .gguf (default: cache do usuário)")
		threads      = flag.Int("threads", 12, "Threads de CPU (goroutines para matvec)")
		ctxLen       = flag.Int("ctx", 2048, "Context length (máx suportado pelo modelo)")
		temp         = flag.Float64("temp", 0.8, "Temperatura")
		topk         = flag.Int("topk", 40, "Top-K")
		maxTokens    = flag.Int("max-tokens", 256, "Máx tokens por resposta")
		systemPrompt = flag.String("system", llama.DefaultSystemPrompt(), "System prompt")
	)
	flag.Parse()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	if *modelPath == "" {
		cacheDir, err := os.UserCacheDir()
		if err != nil {
			fmt.Fprintln(os.Stderr, "erro cache dir:", err)
			os.Exit(1)
		}
		*modelPath = filepath.Join(cacheDir, "go-llama", "Llama-3.2-1B-Instruct-Q8_0.gguf")
	}

	if err := ensureModel(ctx, *modelURL, *modelPath); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	fmt.Println("Carregando modelo... (isso pode levar alguns segundos)")
	m, err := llama.LoadModel(*modelPath, *threads)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load model:", err)
		os.Exit(1)
	}
	defer m.Close()

	fmt.Printf("OK. vocab=%d layers=%d emb=%d heads=%d heads_kv=%d ffn=%d ctx=%d\n",
		m.HP.VocabSize, m.HP.NLayers, m.HP.NEmb, m.HP.NHeads, m.HP.NHeadsKV, m.HP.NFF, m.HP.CtxLen)

	sampler := llama.NewSampler(float32(*temp), *topk, 0)
	ws := m.NewWorkspace(*ctxLen, *topk)

	reset := func() *llama.State {
		s := m.NewState(*ctxLen)
		// ingest system prompt
		sys := llama.BuildSystemPrompt(*systemPrompt)
		ids, err := m.Tok.Encode(sys, false)
		if err != nil {
			fmt.Fprintln(os.Stderr, "tokenize system:", err)
			os.Exit(1)
		}
		if err := ingest(ctx, m, s, ids, ws); err != nil {
			fmt.Fprintln(os.Stderr, "ingest system:", err)
			os.Exit(1)
		}
		return s
	}

	state := reset()
	in := bufio.NewScanner(os.Stdin)
	fmt.Println("Chat pronto. Comandos: /exit /reset")

	for {
		fmt.Print("\nVocê> ")
		if !in.Scan() {
			break
		}
		line := strings.TrimSpace(in.Text())
		if line == "" {
			continue
		}
		if line == "/exit" {
			break
		}
		if line == "/reset" {
			state = reset()
			continue
		}

		turn := llama.BuildUserTurn(line) + llama.BuildAssistantHeader()
		ids, err := m.Tok.Encode(turn, false)
		if err != nil {
			fmt.Fprintln(os.Stderr, "tokenize:", err)
			continue
		}

		items, err := ingestWithTopK(ctx, m, state, ids, *topk, ws)
		if err != nil {
			fmt.Fprintln(os.Stderr, "ingest:", err)
			continue
		}

		fmt.Print("Assistente> ")
		genStart := time.Now()
		tokensGenerated := 0
		for tokensGenerated < *maxTokens {
			if err := ctx.Err(); err != nil {
				return
			}
			next := sampler.SampleTopK(items)
			// stop tokens
			if (m.Tok.EOT >= 0 && next == m.Tok.EOT) || (m.Tok.EOS >= 0 && next == m.Tok.EOS) {
				// still ingest stop token to keep state consistent
				_ = m.ForwardIntoWorkspace(state, next, nil, ws)
				break
			}

			piece := m.Tok.DecodeToken(next)
			if len(piece) > 0 {
				_, _ = os.Stdout.Write(piece)
				_ = os.Stdout.Sync()
			}

			items, err = m.ForwardTopKIntoWorkspace(state, next, *topk, ws)
			if err != nil {
				fmt.Fprintln(os.Stderr, "\nforward:", err)
				break
			}
			tokensGenerated++
		}

		// ensure end-of-turn is in context
		if m.Tok.EOT >= 0 {
			_ = m.ForwardIntoWorkspace(state, m.Tok.EOT, nil, ws)
		}
		elapsed := time.Since(genStart)
		tokPerSec := 0.0
		if elapsed > 0 {
			tokPerSec = float64(tokensGenerated) / elapsed.Seconds()
		}
		fmt.Printf("\n[%d tokens, %.2f tokens/s]\n", tokensGenerated, tokPerSec)
	}
}

func ensureModel(ctx context.Context, url, dest string) error {
	if st, err := os.Stat(dest); err == nil && st.Size() > 0 {
		return nil
	}

	fmt.Println("Baixando modelo... (1.3GB)")
	progressCh := make(chan downloader.Progress, 16)
	errCh := make(chan error, 1)

	hc := &http.Client{Timeout: 0}
	token := os.Getenv("HF_TOKEN")
	go func() {
		err := downloader.Download(ctx, downloader.Options{
			URL:         url,
			DestPath:    dest,
			HTTPClient:  hc,
			BearerToken: token,
		}, func(p downloader.Progress) {
			select {
			case progressCh <- p:
			default:
			}
		})
		errCh <- err
	}()

	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()
	var last downloader.Progress
	for {
		select {
		case p := <-progressCh:
			last = p
		case <-ticker.C:
			printProgress(last)
		case err := <-errCh:
			printProgress(last)
			fmt.Println()
			if err != nil {
				return fmt.Errorf("download falhou: %w", err)
			}
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

func printProgress(p downloader.Progress) {
	if p.Total > 0 {
		pct := float64(p.Downloaded) / float64(p.Total) * 100
		fmt.Printf("\r%.1f%% (%s/%s) %.1f MB/s",
			pct,
			humanBytes(p.Downloaded),
			humanBytes(p.Total),
			p.SpeedBps/1024/1024,
		)
	} else {
		fmt.Printf("\r%s baixado", humanBytes(p.Downloaded))
	}
}

func humanBytes(n int64) string {
	const (
		KB = 1024
		MB = 1024 * KB
		GB = 1024 * MB
	)
	if n >= GB {
		return fmt.Sprintf("%.2fGB", float64(n)/GB)
	}
	if n >= MB {
		return fmt.Sprintf("%.2fMB", float64(n)/MB)
	}
	if n >= KB {
		return fmt.Sprintf("%.2fKB", float64(n)/KB)
	}
	return fmt.Sprintf("%dB", n)
}

func ingest(ctx context.Context, m *llama.Model, s *llama.State, ids []int32, ws *llama.Workspace) error {
	for _, id := range ids {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := m.ForwardIntoWorkspace(s, id, nil, ws); err != nil {
			return err
		}
	}
	return nil
}

func ingestWithTopK(ctx context.Context, m *llama.Model, s *llama.State, ids []int32, topK int, ws *llama.Workspace) ([]llama.TopKItem, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	for i := 0; i < len(ids); i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		id := ids[i]
		if i == len(ids)-1 {
			items, err := m.ForwardTopKIntoWorkspace(s, id, topK, ws)
			if err != nil {
				return nil, err
			}
			return items, nil
		} else {
			if err := m.ForwardIntoWorkspace(s, id, nil, ws); err != nil {
				return nil, err
			}
		}
	}
	return nil, nil
}
