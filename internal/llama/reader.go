package llama

import (
	"fmt"
	"io"

	"go_llama/internal/gguf"
)

type GGUFModelFile struct {
	Path   string
	MM     *MMapFile
	GGUF   *gguf.File
	ByName map[string]Tensor
}

func OpenGGUF(path string) (*GGUFModelFile, error) {
	mm, err := OpenMMap(path)
	if err != nil {
		return nil, err
	}
	g, err := gguf.Parse(bytesReaderAt(mm.Bytes()))
	if err != nil {
		mm.Close()
		return nil, err
	}

	by := make(map[string]Tensor, len(g.Tensors))
	for _, ti := range g.Tensors {
		abs := g.TensorAbsOffset(ti)
		if abs < 0 || abs > int64(len(mm.Bytes())) {
			mm.Close()
			return nil, fmt.Errorf("tensor %q offset out of range", ti.Name)
		}
		// We don't know tensor byte size without type + dims. We just slice from offset to end.
		// Consumers must compute and slice appropriately.
		by[ti.Name] = Tensor{Info: ti, Data: mm.Bytes()[abs:]}
	}

	return &GGUFModelFile{Path: path, MM: mm, GGUF: g, ByName: by}, nil
}

func (f *GGUFModelFile) Close() error {
	if f.MM != nil {
		return f.MM.Close()
	}
	return nil
}

type byteReaderAt []byte

func bytesReaderAt(b []byte) byteReaderAt { return byteReaderAt(b) }

func (b byteReaderAt) ReadAt(p []byte, off int64) (int, error) {
	if off < 0 {
		return 0, fmt.Errorf("negative offset")
	}
	if off >= int64(len(b)) {
		return 0, io.EOF
	}
	n := copy(p, b[off:])
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}
