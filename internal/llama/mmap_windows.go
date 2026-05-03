//go:build windows

package llama

import (
	"fmt"
	"os"

	"github.com/edsrzf/mmap-go"
)

type MMapFile struct {
	f    *os.File
	data mmap.MMap
	size int64
}

func OpenMMap(path string) (*MMapFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	if st.Size() == 0 {
		f.Close()
		return nil, fmt.Errorf("empty file")
	}

	data, err := mmap.Map(f, mmap.RDONLY, 0)
	if err != nil {
		f.Close()
		return nil, err
	}

	return &MMapFile{f: f, data: data, size: st.Size()}, nil
}

func (m *MMapFile) Close() error {
	if m.data != nil {
		m.data.Unmap()
	}
	if m.f != nil {
		return m.f.Close()
	}
	return nil
}

func (m *MMapFile) Bytes() []byte { return m.data }
func (m *MMapFile) Size() int64   { return m.size }