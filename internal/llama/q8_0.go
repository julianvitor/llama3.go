package llama

import (
	"encoding/binary"
)

const (
	ggmlTypeF32  = 0
	ggmlTypeF16  = 1
	ggmlTypeQ8_0 = 8
)

// Q8_0 block: 32 int8 values with one float16 scale (d).
const q8BlockSize = 32
const q8_0BytesPerBlock = 2 + q8BlockSize // d (f16) + qs[32]

type Q8Vector struct {
	D  []float32 // per-block scale
	Qs []int8    // concatenated blocks
}

func QuantizeQ8_0Into(x []float32, dst *Q8Vector) {
	blocks := (len(x) + q8BlockSize - 1) / q8BlockSize
	needQs := blocks * q8BlockSize
	if cap(dst.D) < blocks {
		dst.D = make([]float32, blocks)
	} else {
		dst.D = dst.D[:blocks]
	}
	if cap(dst.Qs) < needQs {
		dst.Qs = make([]int8, needQs)
	} else {
		dst.Qs = dst.Qs[:needQs]
	}

	for b := 0; b < blocks; b++ {
		start := b * q8BlockSize
		end := start + q8BlockSize
		if end > len(x) {
			end = len(x)
		}
		maxAbs := float32(0)
		for i := start; i < end; i++ {
			v := x[i]
			if v < 0 {
				v = -v
			}
			if v > maxAbs {
				maxAbs = v
			}
		}
		scale := maxAbs / 127.0
		if scale == 0 {
			scale = 1e-8
		}
		dst.D[b] = scale
		inv := float32(1.0) / scale
		for i := 0; i < q8BlockSize; i++ {
			idx := start + i
			var v float32
			if idx < len(x) {
				v = x[idx]
			}
			f := v * inv
			var q int
			if f >= 0 {
				q = int(f + 0.5)
			} else {
				q = int(f - 0.5)
			}
			if q > 127 {
				q = 127
			} else if q < -128 {
				q = -128
			}
			dst.Qs[b*q8BlockSize+i] = int8(q)
		}
	}
}

func QuantizeQ8_0(x []float32) Q8Vector {
	var out Q8Vector
	QuantizeQ8_0Into(x, &out)
	return out
}

func dotQ8_0Row(rowData []byte, xQ Q8Vector) float32 {
	// rowData contains blocks for the row, each block is 34 bytes.
	blocks := len(xQ.D)
	var sum float32
	for b := 0; b < blocks; b++ {
		off := b * q8_0BytesPerBlock
		// weight scale
		dw := f16ToF32(binary.LittleEndian.Uint16(rowData[off : off+2]))
		s := dw * xQ.D[b]
		base := off + 2
		xBase := b * q8BlockSize
		var acc int32
		for i := 0; i < q8BlockSize; i++ {
			acc += int32(int8(rowData[base+i])) * int32(xQ.Qs[xBase+i])
		}
		sum += s * float32(acc)
	}
	return sum
}
