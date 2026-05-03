# go_llama (Go + Cgo runtime with 100% Go fallback)

The terminal chat **automatically downloads** a GGUF for **Llama 3.2 1B Instruct** and runs **in-process** inference in Go, with no external server.

<img src="docs/llama_go.jpg" width="600"/>

## Run

With optimizations for x86 (AVX2) or ARMv8 (NEON):
```bash
CGO_ENABLED=1 go run ./cmd/go-llama-chat
```

To run in pure Go on CPUs without compatible SIMD support:
```bash
CGO_ENABLED=0 go run ./cmd/go-llama-chat
```

On first run, it downloads by default:

- `Llama-3.2-1B-Instruct-Q8_0.gguf` (about 1.3 GB)
- Default URL: `bartowski/Llama-3.2-1B-Instruct-GGUF` (Hugging Face)

The file is cached in `~/.cache/go-llama/` on Linux.

## Useful Flags

```bash
go run ./cmd/go-llama-chat \
  --threads 8 \
  --ctx 2048 \
  --temp 0.8 \
  --topk 40 \
  --max-tokens 256
```

To point to another GGUF file:

```bash
go run ./cmd/go-llama-chat --model-path /path/to/model.gguf
```

Or another URL:

```bash
go run ./cmd/go-llama-chat --model-url 'https://.../model.gguf'
```

If the Hugging Face download requires authentication, set:

```bash
export HF_TOKEN=...  # Hugging Face token
```

## Tested Models

| Model | Parameters | Quantization | Download Link |
| :--- | :---: | :---: | :--- |
| **Llama 3.2 Instruct** | 1B | Q8_0 | [hf.co/Llama-3.2-1B](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf) |
| **Llama 3.2 Instruct** | 3B | Q8_0 | [hf.co/Llama-3.2-3B](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q8_0.gguf) |
| **Llama 3.1 Instruct** | 8B | Q8_0 | [hf.co/Meta-Llama-3.1-8B](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf) |

---

## Checks

| Operating System | CPU | Compilation Version | Status | Throughput |
| :--- | :--- | :--- | :---: | :---: |
| Windows | AMD Zen 2 | CGO_ENABLED=1 | ✅ OK | 10.49 tokens/s |
| Windows | AMD Zen 2 | CGO_ENABLED=0 | ✅ OK | 3.27 tokens/s |
| Linux (WSL2) | AMD Zen 2 | CGO_ENABLED=0 | ✅ OK | 2.86 tokens/s |
| Linux (WSL2) | AMD Zen 2 | CGO_ENABLED=1 | ✅ OK | 8.89 tokens/s |

🧪 Prompt used: `Quem é você?`

---

### Profiling and Performance Analysis

**CPU Profiling** (finds hotspots):
```bash
go run ./cmd/go-llama-chat -cpuprofile=cpu.prof --model-path model.gguf
# Then: go tool pprof cpu.prof
```

**Memory Profiling** (analyzes allocations):
```bash
go run ./cmd/go-llama-chat -memprofile=mem.prof --model-path model.gguf
# Then: go tool pprof mem.prof
```

**Benchmarks** (measure parallel work throughput):
```bash
# Full Pool.Run benchmark (3 parallel tasks)
go test -bench=TaskPoolRun -benchmem ./internal/llama/

# Work parallelization benchmark
go test -bench=ParallelFor -benchmem ./internal/llama/

# All benchmarks
go test -bench=. -benchmem ./internal/llama/
```

**Validation Script** (checks build, races, and benchmarks):
```bash
bash test_perf.sh
```

## Notes

- This runtime supports **GGUF with `Q8_0` weights** for matrices by default. Other quant types, such as `Q4_K_M`, are not implemented.
- It is a **CPU-only** runtime with only a few optimized SIMD kernels, so it will be **significantly slower** than `llama.cpp`.
- The tokenizer is **GPT-2 BPE**, using `tokenizer.ggml.tokens` plus `tokenizer.ggml.merges` from the GGUF itself. This model declares `tokenizer.ggml.model = gpt2`.

---

# go_llama (runtime em Go + Cgo com fallback 100% em Go)

O chat no terminal **baixa automaticamente** um GGUF do **Llama 3.2 1B Instruct** e roda inferência **in-process** em Go, sem servidor externo.

<img src="docs/llama_go.jpg" width="600"/>

## Rodar

Com otimizações para x86 (AVX2) ou ARMv8 (NEON):
```bash
CGO_ENABLED=1 go run ./cmd/go-llama-chat
```

Para rodar em Go puro em CPUs sem SIMD compatível:
```bash
CGO_ENABLED=0 go run ./cmd/go-llama-chat
```

Na primeira execução, ele baixa por padrão:

- `Llama-3.2-1B-Instruct-Q8_0.gguf` (cerca de 1,3 GB)
- URL padrão: `bartowski/Llama-3.2-1B-Instruct-GGUF` (Hugging Face)

O arquivo fica em cache em `~/.cache/go-llama/` no Linux.

## Flags úteis

```bash
go run ./cmd/go-llama-chat \
  --threads 8 \
  --ctx 2048 \
  --temp 0.8 \
  --topk 40 \
  --max-tokens 256
```

Para apontar para outro arquivo GGUF:

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

## Modelos testados

| Modelo | Parâmetros | Quantização | Link de Download |
| :--- | :---: | :---: | :--- |
| **Llama 3.2 Instruct** | 1B | Q8_0 | [hf.co/Llama-3.2-1B](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf) |
| **Llama 3.2 Instruct** | 3B | Q8_0 | [hf.co/Llama-3.2-3B](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q8_0.gguf) |
| **Llama 3.1 Instruct** | 8B | Q8_0 | [hf.co/Meta-Llama-3.1-8B](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf) |

---

## Checks

| Sistema operacional | CPU | Versão da compilação | Status | Vazão |
| :--- | :--- | :--- | :---: | :---: |
| Windows | AMD Zen 2 | CGO_ENABLED=1 | ✅ OK | 10.49 tokens/s |
| Windows | AMD Zen 2 | CGO_ENABLED=0 | ✅ OK | 3.27 tokens/s |
| Linux (WSL2) | AMD Zen 2 | CGO_ENABLED=0 | ✅ OK | 2.86 tokens/s |
| Linux (WSL2) | AMD Zen 2 | CGO_ENABLED=1 | ✅ OK | 8.89 tokens/s |

🧪 Prompt usado: `Quem é você?`

---

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

- Este runtime suporta **GGUF com pesos `Q8_0`** para as matrizes por padrão. Outros tipos de quantização, como `Q4_K_M`, não estão implementados.
- É um runtime **somente CPU** e com poucos kernels SIMD otimizados; então será **bem mais lento** que `llama.cpp`.
- O tokenizer implementado é **GPT-2 BPE** usando `tokenizer.ggml.tokens` + `tokenizer.ggml.merges` do próprio GGUF. Este modelo declara `tokenizer.ggml.model = gpt2`.
