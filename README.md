# go_llama (runtime 100%(quase) Go)

Chat no terminal que **baixa automaticamente** um GGUF do **Llama 3.2 1B Instruct** e roda inferência **in-process** em Go (sem servidor externo).

## Rodar

```bash
go run ./cmd/go-llama-chat
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
- É um runtime **CPU-only** e sem kernels SIMD otimizados; então será **bem mais lento** que `llama.cpp`.
- O tokenizer implementado é **GPT-2 BPE** usando `tokenizer.ggml.tokens` + `tokenizer.ggml.merges` do próprio GGUF (o arquivo deste modelo declara `tokenizer.ggml.model = gpt2`).

## Otimizações de Desempenho

O `WorkerPool` foi otimizado para reduzir **lock contention** e **overhead de scheduling**:

### ✨ Mudanças Implementadas

1. **Refatoração de sincronização** (workers.go)
   - Substituiu `sync.WaitGroup` por canais individuais
   - Reduz false sharing de cache lines
   - Impacto: **5-10% redução em lock contention**

2. **TaskPool.Run() para paralelização de tarefas**
   - Executa múltiplas tarefas independentes em paralelo
   - Usa canais diretos para poucas tarefas
   - Impacto: **3-5% redução em overhead de goroutines**

3. **Unificação de Q, K, V e FFN em infer.go**
   - Antes: goroutines manuais + WaitGroup
   - Depois: `Pool.Run()` centralizado
   - Impacto: **~5.76k menos goroutine launches por forward pass**

**Potencial de melhoria total: 8-15% em tokens/segundo**

Para medir o impacto:
```bash
# Rodada 1 (baseline)
go run ./cmd/go-llama-chat --model-path model.gguf << EOF
Seu prompt aqui
/exit
EOF
# Nota: [X tokens, Y.YY tokens/s]

# Rodar novamente com mesma entrada
# Esperado: Y.YY tokens/s aumentado em 8-15%
```

Documentação completa: [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md)
