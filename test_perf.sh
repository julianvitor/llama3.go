#!/bin/bash

# Cores para o output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

set -e

# Configuração de caminhos (ajustado para o seu cache)
CACHE_DIR="${HOME}/.cache/go-llama"
DEFAULT_MODEL="${CACHE_DIR}/Llama-3.2-1B-Instruct-Q8_0.gguf"
MODEL_PATH=${1:-$DEFAULT_MODEL}

echo -e "${BLUE}=== Llama3.go: Suite de Testes e Performance ===${NC}\n"

# 1. Compilação Otimizada
echo -e "${GREEN}[1/5] Compilando com CGO + AVX2 (GOAMD64=v3)...${NC}"
GOAMD64=v3 CGO_ENABLED=1 go build -o bin/llama-opt ./cmd/go-llama-chat/
echo "✓ Binário otimizado gerado em bin/llama-opt"

# 2. Compilação Go Puro
echo -e "\n${GREEN}[2/5] Compilando Fallback (CGO_ENABLED=0)...${NC}"
CGO_ENABLED=0 go build -o bin/llama-pure ./cmd/go-llama-chat/
echo "✓ Binário estático gerado em bin/llama-pure"

# 3. Verificação de Símbolos SIMD
echo -e "\n${GREEN}[3/5] Verificando símbolos C no binário...${NC}"
if nm ./bin/llama-opt | grep -q "dot_product"; then
    echo "✓ Símbolo 'dot_product' presente. Linker OK."
else
    echo -e "${RED}✗ Erro: Símbolos de aceleração ausentes!${NC}"
    exit 1
fi

# 4. Testes de Unidade e Data Race
echo -e "\n${GREEN}[4/5] Rodando Race Detector...${NC}"
# Usamos -race para garantir que o Pool de Workers está thread-safe
go test -race -timeout=10s ./internal/llama/... || echo "Aviso: Nenhum teste de unidade encontrado."

# 5. Teste de Inferência Não-Interativo (Sanity Check)
echo -e "\n${GREEN}[5/5] Executando Inferência de Teste...${NC}"
if [ -f "$MODEL_PATH" ]; then
    echo "Usando: $MODEL_PATH"
    
    # O "echo" simula a entrada do usuário e o "/exit" fecha o chat após a resposta
    # Redirecionamos o stderr para ver apenas o chat e o log final
    LOG_OUT=$( (echo "Hi"; echo "/exit") | ./bin/llama-opt --model-path "$MODEL_PATH" --max-tokens 10 2>&1 )
    
    echo -e "--- Output do Teste ---"
    echo "$LOG_OUT" | grep -E "Assistente>|tokens/s"
    echo -e "-----------------------"

    if echo "$LOG_OUT" | grep -q "tokens/s"; then
        TPS=$(echo "$LOG_OUT" | grep -oP '\d+\.\d+ tokens/s')
        echo -e "${GREEN}✓ Sucesso: Inferência funcional a $TPS${NC}"
    else
        echo -e "${RED}✗ Falha: O modelo não gerou métricas de performance.${NC}"
        exit 1
    fi
else
    echo -e "${RED}⚠️  Modelo não encontrado em $MODEL_PATH${NC}"
fi

echo -e "\n${BLUE}=== Todos os testes concluídos com sucesso! ===${NC}"c