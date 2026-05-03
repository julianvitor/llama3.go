# go_llama (runtime em Go + Cgo com fallback 100% em go)

Chat no terminal **baixa automaticamente** um GGUF do **Llama 3.2 1B Instruct** e roda inferência **in-process** em Go (sem servidor externo).

## Rodar
Com otimizações para x86(AVX2) ou ARMv8 (neon)
```bash
CGO_ENABLED=1 go run ./cmd/go-llama-chat
```

Para rodar em go puro(cpus sem SIMD compativel).
```bash
CGO_ENABLED=0 go run ./cmd/go-llama-chat
```

Na primeira execução, ele baixa por padrão:

- `Llama-3.2-1B-Instruct-Q8_0.gguf` (≈ 1.3GB)
- URL default: `bartowski/Llama-3.2-1B-Instruct-GGUF` (Hugging Face)

O arquivo fica em cache em `~/.cache/go-llama/` (Linux).



## Flags úteis

```bash
go run ./cmd/go-llama-chat \
  --threads 8 \
  --ctx 2048 \
  --temp 0.8 \
  --topk 40 \
  --max-tokens 256
```

Para apontar pra outro arquivo GGUF:

```bash
go run ./cmd/go-llama-chat --model-path /caminho/model.gguf
```

Ou outro URL:

```bash
go run ./cmd/go-llama-chat --model-url 'https://.../model.gguf'
```

Se o download do Hugging Face exigir autenticação, defina:

```bash
export HF_TOKEN=...  # token do Hugging Face
```

### Profiling e Análise de Desempenho

**CPU Profiling** (identifica hotspots):
```bash
go run ./cmd/go-llama-chat -cpuprofile=cpu.prof --model-path model.gguf
# Depois: go tool pprof cpu.prof
```

**Memory Profiling** (analisa alocações):
```bash
go run ./cmd/go-llama-chat -memprofile=mem.prof --model-path model.gguf
# Depois: go tool pprof mem.prof
```

**Benchmarks** (medir taxa de operações paralelas):
```bash
# Benchmark completo de Pool.Run (3 tarefas paralelas)
go test -bench=TaskPoolRun -benchmem ./internal/llama/

# Benchmark de paralelização de trabalho
go test -bench=ParallelFor -benchmem ./internal/llama/

# Todos os benchmarks
go test -bench=. -benchmem ./internal/llama/
```

**Script de Validação** (testa compilação, races e benchmarks):
```bash
bash test_perf.sh
```

## Observações

- Este runtime suporta **GGUF com pesos `Q8_0`** para as matrizes (o default). Outros quant types (ex: `Q4_K_M`) não estão implementados.
- É um runtime **CPU-only** e com poucos kernels SIMD otimizados; então será **bem mais lento** que `llama.cpp`.
- O tokenizer implementado é **GPT-2 BPE** usando `tokenizer.ggml.tokens` + `tokenizer.ggml.merges` do próprio GGUF (o arquivo deste modelo declara `tokenizer.ggml.model = gpt2`).
