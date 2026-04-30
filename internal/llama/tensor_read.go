package llama

import (
	"encoding/binary"
	"fmt"
	"math"
)

func readVectorF32(t Tensor, n int) ([]float32, error) {
	if err := t.RequireType(ggmlTypeF32); err != nil {
		return nil, err
	}
	need := n * 4
	if len(t.Data) < need {
		return nil, fmt.Errorf("tensor %q truncated", t.Info.Name)
	}
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = float32FromBytes(t.Data[i*4 : i*4+4])
	}
	return out, nil
}

func float32FromBytes(b []byte) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(b))
}

func readVectorF16ToF32(t Tensor, n int) ([]float32, error) {
	if err := t.RequireType(ggmlTypeF16); err != nil {
		return nil, err
	}
	need := n * 2
	if len(t.Data) < need {
		return nil, fmt.Errorf("tensor %q truncated", t.Info.Name)
	}
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = f16ToF32(binary.LittleEndian.Uint16(t.Data[i*2 : i*2+2]))
	}
	return out, nil
}

func readVectorAnyToF32(t Tensor, n int) ([]float32, error) {
	switch t.Type() {
	case ggmlTypeF32:
		return readVectorF32(t, n)
	case ggmlTypeF16:
		return readVectorF16ToF32(t, n)
	default:
		return nil, fmt.Errorf("unsupported vector type %d for %q", t.Type(), t.Info.Name)
	}
}

func readRowVecAnyToF32(t Tensor, row int) ([]float32, error) {
	if len(t.Info.Dims) != 2 {
		return nil, fmt.Errorf("tensor %q expected 2D", t.Info.Name)
	}
	cols := int(t.Info.Dims[0])
	rows := int(t.Info.Dims[1])
	if row < 0 || row >= rows {
		return nil, fmt.Errorf("row out of range")
	}
	// contiguous row
	startElem := row * cols
	switch t.Type() {
	case ggmlTypeQ8_0:
		if cols%q8BlockSize != 0 {
			return nil, fmt.Errorf("tensor %q: cols=%d not multiple of %d", t.Info.Name, cols, q8BlockSize)
		}
		blocks := cols / q8BlockSize
		rowBytes := blocks * q8_0BytesPerBlock
		off := row * rowBytes
		need := off + rowBytes
		if len(t.Data) < need {
			return nil, fmt.Errorf("tensor %q truncated", t.Info.Name)
		}
		out := make([]float32, cols)
		for b := 0; b < blocks; b++ {
			boff := off + b*q8_0BytesPerBlock
			dw := f16ToF32(binary.LittleEndian.Uint16(t.Data[boff : boff+2]))
			base := boff + 2
			for i := 0; i < q8BlockSize; i++ {
				out[b*q8BlockSize+i] = dw * float32(int8(t.Data[base+i]))
			}
		}
		return out, nil
	case ggmlTypeF16:
		off := startElem * 2
		need := off + cols*2
		if len(t.Data) < need {
			return nil, fmt.Errorf("tensor %q truncated", t.Info.Name)
		}
		out := make([]float32, cols)
		for i := 0; i < cols; i++ {
			out[i] = f16ToF32(binary.LittleEndian.Uint16(t.Data[off+i*2 : off+i*2+2]))
		}
		return out, nil
	case ggmlTypeF32:
		off := startElem * 4
		need := off + cols*4
		if len(t.Data) < need {
			return nil, fmt.Errorf("tensor %q truncated", t.Info.Name)
		}
		out := make([]float32, cols)
		for i := 0; i < cols; i++ {
			out[i] = float32FromBytes(t.Data[off+i*4 : off+i*4+4])
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported row type %d for %q", t.Type(), t.Info.Name)
	}
}

func readRowVecAnyToF32Into(dst []float32, t Tensor, row int) error {
	if len(t.Info.Dims) != 2 {
		return fmt.Errorf("tensor %q expected 2D", t.Info.Name)
	}
	cols := int(t.Info.Dims[0])
	rows := int(t.Info.Dims[1])
	if len(dst) != cols {
		return fmt.Errorf("dst len=%d, want %d", len(dst), cols)
	}
	if row < 0 || row >= rows {
		return fmt.Errorf("row out of range")
	}

	switch t.Type() {
	case ggmlTypeQ8_0:
		if cols%q8BlockSize != 0 {
			return fmt.Errorf("tensor %q: cols=%d not multiple of %d", t.Info.Name, cols, q8BlockSize)
		}
		blocks := cols / q8BlockSize
		rowBytes := blocks * q8_0BytesPerBlock
		off := row * rowBytes
		need := off + rowBytes
		if len(t.Data) < need {
			return fmt.Errorf("tensor %q truncated", t.Info.Name)
		}
		for b := 0; b < blocks; b++ {
			boff := off + b*q8_0BytesPerBlock
			dw := f16ToF32(binary.LittleEndian.Uint16(t.Data[boff : boff+2]))
			base := boff + 2
			for i := 0; i < q8BlockSize; i++ {
				dst[b*q8BlockSize+i] = dw * float32(int8(t.Data[base+i]))
			}
		}
		return nil
	case ggmlTypeF16:
		startElem := row * cols
		off := startElem * 2
		need := off + cols*2
		if len(t.Data) < need {
			return fmt.Errorf("tensor %q truncated", t.Info.Name)
		}
		for i := 0; i < cols; i++ {
			dst[i] = f16ToF32(binary.LittleEndian.Uint16(t.Data[off+i*2 : off+i*2+2]))
		}
		return nil
	case ggmlTypeF32:
		startElem := row * cols
		off := startElem * 4
		need := off + cols*4
		if len(t.Data) < need {
			return fmt.Errorf("tensor %q truncated", t.Info.Name)
		}
		for i := 0; i < cols; i++ {
			dst[i] = float32FromBytes(t.Data[off+i*4 : off+i*4+4])
		}
		return nil
	default:
		return fmt.Errorf("unsupported row type %d for %q", t.Type(), t.Info.Name)
	}
}
