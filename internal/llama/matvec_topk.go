package llama

import (
	"fmt"
	"sync"
)

// matVecTopKWS computes y = W*x but only keeps the topK rows by score.
// Only supports Q8_0 weights (fast path). It returns a slice backed by ws.TopKBuf.
func (m *Model) matVecTopKWS(W Tensor, xQ Q8Vector, topK int, ws *Workspace) ([]TopKItem, error) {
	if topK <= 0 {
		return ws.TopKBuf[:0], nil
	}
	if len(W.Info.Dims) != 2 {
		return nil, fmt.Errorf("tensor %q expected 2D", W.Info.Name)
	}
	cols := int(W.Info.Dims[0])
	rows := int(W.Info.Dims[1])
	if cols%q8BlockSize != 0 {
		return nil, fmt.Errorf("matvecTopK %q: cols=%d not multiple of %d", W.Info.Name, cols, q8BlockSize)
	}
	if len(xQ.Qs) < cols {
		return nil, fmt.Errorf("matvecTopK %q: quantized x too short", W.Info.Name)
	}
	rowBytes := (cols / q8BlockSize) * q8_0BytesPerBlock
	need := rows * rowBytes
	if len(W.Data) < need {
		return nil, fmt.Errorf("tensor %q truncated", W.Info.Name)
	}

	workers := m.Pool.Workers()
	if workers <= 0 {
		workers = 1
	}
	chunks := workers
	if rows < chunks {
		chunks = rows
	}

	if ws == nil {
		return nil, fmt.Errorf("workspace is nil")
	}
	needLocal := chunks * topK
	if cap(ws.TopKLocal) < needLocal {
		ws.TopKLocal = make([]TopKItem, needLocal)
	} else {
		ws.TopKLocal = ws.TopKLocal[:needLocal]
	}
	if cap(ws.TopKLens) < chunks {
		ws.TopKLens = make([]int, chunks)
	} else {
		ws.TopKLens = ws.TopKLens[:chunks]
		for i := range ws.TopKLens {
			ws.TopKLens[i] = 0
		}
	}

	var wg sync.WaitGroup
	wg.Add(chunks)
	for ci := 0; ci < chunks; ci++ {
		start := ci * rows / chunks
		end := (ci + 1) * rows / chunks
		go func(ci, start, end int) {
			defer wg.Done()
			local := ws.TopKLocal[ci*topK : (ci+1)*topK]
			best := local[:0]
			for r := start; r < end; r++ {
				off := r * rowBytes
				logit := dotQ8_0Row(W.Data[off:off+rowBytes], xQ)
				best = keepTopKInsert(best, topK, TopKItem{ID: int32(r), Logit: logit})
			}
			copy(local, best)
			ws.TopKLens[ci] = len(best)
		}(ci, start, end)
	}
	wg.Wait()

	if cap(ws.TopKBuf) < topK {
		ws.TopKBuf = make([]TopKItem, 0, topK)
	}
	out := ws.TopKBuf[:0]
	for ci := 0; ci < chunks; ci++ {
		best := ws.TopKLocal[ci*topK : ci*topK+ws.TopKLens[ci]]
		for _, it := range best {
			out = keepTopKInsert(out, topK, it)
		}
	}
	ws.TopKBuf = out
	return out, nil
}
