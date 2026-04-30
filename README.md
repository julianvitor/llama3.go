# go_llama (runtime 100% Go)

Chat no terminal que **baixa automaticamente** um GGUF do **Llama 3.2 1B Instruct** e roda inferência **in-process** em Go (sem CGO, sem servidor externo).

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

## Observações

- Este runtime suporta **GGUF com pesos `Q8_0`** para as matrizes (o default). Outros quant types (ex: `Q4_K_M`) não estão implementados.
- É um runtime **CPU-only** e sem kernels SIMD otimizados; então será **bem mais lento** que `llama.cpp`.
- O tokenizer implementado é **GPT-2 BPE** usando `tokenizer.ggml.tokens` + `tokenizer.ggml.merges` do próprio GGUF (o arquivo deste modelo declara `tokenizer.ggml.model = gpt2`).
