package llama

import (
	"fmt"

	"go_llama/internal/gguf"
)

type Tensor struct {
	Info gguf.TensorInfo
	Data []byte // points into mmap'd file
}

func (t Tensor) Dim(i int) int {
	if i < 0 || i >= len(t.Info.Dims) {
		return 0
	}
	return int(t.Info.Dims[i])
}

func (t Tensor) Dims() []uint64 { return t.Info.Shape() }

func (t Tensor) Type() uint32 { return t.Info.Type }

func (t Tensor) RequireType(tt ...uint32) error {
	for _, x := range tt {
		if t.Info.Type == x {
			return nil
		}
	}
	return fmt.Errorf("tensor %q has type %d, want one of %v", t.Info.Name, t.Info.Type, tt)
}
