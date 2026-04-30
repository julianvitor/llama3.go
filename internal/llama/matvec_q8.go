package llama

import "fmt"

func (m *Model) matVecQ8_0(W Tensor, xQ Q8Vector, out []float32) error {
	if len(W.Info.Dims) != 2 {
		return fmt.Errorf("tensor %q expected 2D", W.Info.Name)
	}
	cols := int(W.Info.Dims[0])
	rows := int(W.Info.Dims[1])
	if cols%q8BlockSize != 0 {
		return fmt.Errorf("matvec %q: cols=%d not multiple of %d", W.Info.Name, cols, q8BlockSize)
	}
	if len(xQ.Qs) < cols {
		return fmt.Errorf("matvec %q: quantized x too short", W.Info.Name)
	}
	if len(out) != rows {
		return fmt.Errorf("matvec %q: out len=%d, want %d", W.Info.Name, len(out), rows)
	}
	rowBytes := (cols / q8BlockSize) * q8_0BytesPerBlock
	need := rows * rowBytes
	if len(W.Data) < need {
		return fmt.Errorf("tensor %q truncated", W.Info.Name)
	}

	m.Pool.ParallelFor(rows, func(start, end int) {
		for r := start; r < end; r++ {
			off := r * rowBytes
			out[r] = dotQ8_0Row(W.Data[off:off+rowBytes], xQ)
		}
	})
	return nil
}
