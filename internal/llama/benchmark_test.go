package llama

import (
	"testing"
)

func BenchmarkForwardPass(b *testing.B) {
	// Mock model para benchmark sem carregar arquivo GGUF
	m := &Model{
		HP: HyperParams{
			VocabSize: 128000,
			NLayers:   32,
			NEmb:      2048,
			NHeads:    32,
			NHeadsKV:  8,
			NFF:       8192,
			RmsEps:    1e-5,
			RopeBase:  500000,
			RopeDim:   64,
			CtxLen:    2048,
		},
		Pool: NewWorkerPool(12),
	}
	defer m.Pool.Close()

	ctx := 256
	headDim := m.HP.NEmb / m.HP.NHeads

	ws := m.NewWorkspace(ctx, 40)
	state := &State{
		Pos:  0,
		KV:   NewKVCache(m.HP.NLayers, ctx, m.HP.NHeadsKV, headDim),
		Rope: NewRopeCache(m.HP.CtxLen, m.HP.RopeDim, m.HP.RopeBase),
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Simular forward pass de um token
		_ = m.ForwardIntoWorkspace(state, 1, nil, ws)
		state.Pos++
		if state.Pos >= ctx {
			state.Pos = 0
		}
	}
}

func BenchmarkMatVecQ8(b *testing.B) {
	m := &Model{
		HP: HyperParams{
			NEmb: 2048,
		},
		Pool: NewWorkerPool(12),
	}
	defer m.Pool.Close()

	// Criar tensores fake de teste
	rows := 2048
	cols := 2048
	rowBytes := (cols / q8BlockSize) * q8_0BytesPerBlock

	W := Tensor{
		Data: make([]byte, rows*rowBytes),
	}
	x := make([]float32, cols)
	out := make([]float32, rows)

	// Popular x com dados aleatórios
	for i := range x {
		x[i] = float32(i%10) * 0.1
	}

	xq := QuantizeQ8_0(x)

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.Pool.ParallelFor(rows, func(start, end int) {
			for r := start; r < end; r++ {
				off := r * rowBytes
				out[r] = dotQ8_0Row(W.Data[off:off+rowBytes], xq)
			}
		})
	}
}

func BenchmarkParallelFor(b *testing.B) {
	m := &Model{
		Pool: NewWorkerPool(12),
	}
	defer m.Pool.Close()

	n := 1024
	counter := 0

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.Pool.ParallelFor(n, func(start, end int) {
			for j := start; j < end; j++ {
				_ = j * 2 // Simular trabalho mínimo
			}
		})
		counter++
	}
}

func BenchmarkTaskPoolRun(b *testing.B) {
	m := &Model{
		Pool: NewWorkerPool(12),
	}
	defer m.Pool.Close()

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.Pool.Run(
			func() { _ = 1 },
			func() { _ = 2 },
			func() { _ = 3 },
		)
	}
}
