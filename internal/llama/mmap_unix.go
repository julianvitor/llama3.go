//go:build linux || darwin || freebsd || netbsd || openbsd || dragonfly

package llama

import (
	"fmt"
	"os"
	"syscall"
)

type MMapFile struct {
	f    *os.File
	data []byte
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
	sz := st.Size()
	if sz <= 0 {
		f.Close()
		return nil, fmt.Errorf("empty file")
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(sz), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		f.Close()
		return nil, err
	}
	return &MMapFile{f: f, data: data, size: sz}, nil
}

func (m *MMapFile) Close() error {
	var err1 error
	if m.data != nil {
		err1 = syscall.Munmap(m.data)
		m.data = nil
	}
	var err2 error
	if m.f != nil {
		err2 = m.f.Close()
		m.f = nil
	}
	if err1 != nil {
		return err1
	}
	return err2
}

func (m *MMapFile) Bytes() []byte { return m.data }
func (m *MMapFile) Size() int64   { return m.size }
