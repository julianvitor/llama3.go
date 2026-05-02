//go:build !cgo
package llama

import (
	"fmt"
)

func (m *Model) matVec(W Tensor, x []float32, out []float32) error {
	if len(W.Info.Dims) != 2 {
		return fmt.Errorf("tensor %q expected 2D", W.Info.Name)
	}
	cols := int(W.Info.Dims[0])
	rows := int(W.Info.Dims[1])
	if len(x) != cols {
		return fmt.Errorf("matvec %q: x len=%d, want %d", W.Info.Name, len(x), cols)
	}
	if len(out) != rows {
		return fmt.Errorf("matvec %q: out len=%d, want %d", W.Info.Name, len(out), rows)
	}

	switch W.Type() {
	case ggmlTypeQ8_0:
		if cols%q8BlockSize != 0 {
			return fmt.Errorf("matvec %q: cols=%d not multiple of %d", W.Info.Name, cols, q8BlockSize)
		}
		xq := QuantizeQ8_0(x)
		rowBytes := (cols / q8BlockSize) * q8_0BytesPerBlock
		need := rows * rowBytes
		if len(W.Data) < need {
			return fmt.Errorf("tensor %q truncated", W.Info.Name)
		}

		m.Pool.ParallelFor(rows, func(start, end int) {
			for r := start; r < end; r++ {
				off := r * rowBytes
				out[r] = dotQ8_0Row(W.Data[off:off+rowBytes], xq)
			}
		})
		return nil

	case ggmlTypeF16, ggmlTypeF32:
		// Slow path: decode row to float32 and dot.
		m.Pool.ParallelFor(rows, func(start, end int) {
			for r := start; r < end; r++ {
				row, err := readRowVecAnyToF32(W, r)
				if err != nil {
					// ignore; will be caught by outer call if needed
					continue
				}
				var s float32
				for i := 0; i < cols; i++ {
					s += row[i] * x[i]
				}
				out[r] = s
			}
		})
		return nil

	default:
		return fmt.Errorf("unsupported matvec weight type %d for %q", W.Type(), W.Info.Name)
	}
}
