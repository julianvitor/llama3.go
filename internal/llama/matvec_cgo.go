//go:build cgo

package llama

/*
#cgo CFLAGS: -O3 -Wall
#include "simd.h"
*/
import "C"
import (
	"unsafe"
)

func (m *Model) matVecQ8_0(W Tensor, xQ Q8Vector, out []float32) error {
	cols := int(W.Info.Dims[0])
	rows := int(W.Info.Dims[1])

	m.Pool.ParallelFor(rows, func(start, end int) {
		C.matvec_q8_c(
			unsafe.Pointer(&W.Data[0]),      // Pesos compactados (34 bytes/bloco)
			(*C.float)(unsafe.Pointer(&xQ.D[0])), // Escalas da ativação (float32)
			(*C.schar)(unsafe.Pointer(&xQ.Qs[0])), // Pesos da ativação (int8)
			(*C.float)(unsafe.Pointer(&out[0])),
			C.int(start),
			C.int(end),
			C.int(cols),
		)
	})
	return nil
}

// matVec: Versão para F32/F16 que estava faltando
func (m *Model) matVec(W Tensor, x []float32, out []float32) error {
	cols := int(W.Info.Dims[0])
	rows := int(W.Info.Dims[1])

	switch W.Type() {
	case ggmlTypeF32:
		m.Pool.ParallelFor(rows, func(start, end int) {
			C.matvec_f32_c(
				(*C.float)(unsafe.Pointer(&W.Data[0])),
				(*C.float)(unsafe.Pointer(&x[0])),
				(*C.float)(unsafe.Pointer(&out[0])),
				C.int(start),
				C.int(end),
				C.int(cols),
			)
		})
		return nil

	case ggmlTypeQ8_0:
		xq := QuantizeQ8_0(x)
		return m.matVecQ8_0(W, xq, out)

	default:
		m.Pool.ParallelFor(rows, func(start, end int) {
			for r := start; r < end; r++ {
				row, _ := readRowVecAnyToF32(W, r)
				var s float32
				for i := 0; i < cols; i++ {
					s += row[i] * x[i]
				}
				out[r] = s
			}
		})
		return nil
	}
}