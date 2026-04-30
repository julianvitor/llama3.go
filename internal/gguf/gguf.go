package gguf

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

const magic = "GGUF"

type Header struct {
	Version     uint32
	TensorCount uint64
	KVCount     uint64
}

type ValueType uint32

const (
	TypeUint8   ValueType = 0
	TypeInt8    ValueType = 1
	TypeUint16  ValueType = 2
	TypeInt16   ValueType = 3
	TypeUint32  ValueType = 4
	TypeInt32   ValueType = 5
	TypeFloat32 ValueType = 6
	TypeBool    ValueType = 7
	TypeString  ValueType = 8
	TypeArray   ValueType = 9
	TypeUint64  ValueType = 10
	TypeInt64   ValueType = 11
	TypeFloat64 ValueType = 12
)

type KV struct {
	Key   string
	Type  ValueType
	Value any
}

type TensorInfo struct {
	Name   string
	NDims  uint32
	Dims   []uint64
	Type   uint32
	Offset uint64 // offset in bytes from the start of the tensor data section
}

type File struct {
	Header       Header
	KV           map[string]KV
	Tensors      []TensorInfo
	DataOffset   int64 // absolute offset to tensor data section
	Alignment    uint32
	RawHeaderLen int64
}

func Parse(r io.ReaderAt) (*File, error) {
	sr := io.NewSectionReader(r, 0, math.MaxInt64)
	br := bufio.NewReaderSize(sr, 1<<20)

	var m [4]byte
	if _, err := io.ReadFull(br, m[:]); err != nil {
		return nil, err
	}
	if string(m[:]) != magic {
		return nil, fmt.Errorf("not a gguf file (magic=%q)", string(m[:]))
	}

	var h Header
	if err := binary.Read(br, binary.LittleEndian, &h.Version); err != nil {
		return nil, err
	}
	if err := binary.Read(br, binary.LittleEndian, &h.TensorCount); err != nil {
		return nil, err
	}
	if err := binary.Read(br, binary.LittleEndian, &h.KVCount); err != nil {
		return nil, err
	}

	kv := make(map[string]KV, int(h.KVCount))
	for i := uint64(0); i < h.KVCount; i++ {
		key, err := readString(br)
		if err != nil {
			return nil, fmt.Errorf("read kv key: %w", err)
		}
		var t ValueType
		if err := binary.Read(br, binary.LittleEndian, &t); err != nil {
			return nil, err
		}
		val, err := readValue(br, t)
		if err != nil {
			return nil, fmt.Errorf("read kv %q: %w", key, err)
		}
		kv[key] = KV{Key: key, Type: t, Value: val}
	}

	tensors := make([]TensorInfo, 0, int(h.TensorCount))
	for i := uint64(0); i < h.TensorCount; i++ {
		name, err := readString(br)
		if err != nil {
			return nil, fmt.Errorf("read tensor name: %w", err)
		}
		var nDims uint32
		if err := binary.Read(br, binary.LittleEndian, &nDims); err != nil {
			return nil, err
		}
		dims := make([]uint64, nDims)
		for d := uint32(0); d < nDims; d++ {
			if err := binary.Read(br, binary.LittleEndian, &dims[d]); err != nil {
				return nil, err
			}
		}
		var ggmlType uint32
		if err := binary.Read(br, binary.LittleEndian, &ggmlType); err != nil {
			return nil, err
		}
		var off uint64
		if err := binary.Read(br, binary.LittleEndian, &off); err != nil {
			return nil, err
		}
		tensors = append(tensors, TensorInfo{Name: name, NDims: nDims, Dims: dims, Type: ggmlType, Offset: off})
	}

	rawLen := sr.Size() - int64(br.Buffered())
	// rawLen is not correct here; instead compute as current position in section reader.
	// bufio.Reader doesn't expose position, so we reconstruct by reading buffered bytes count.
	// We can compute by creating a small wrapper, but simplest: re-parse with a counting reader.
	// For now, we approximate with a second pass counting bytes.
	_ = rawLen

	// Second pass to compute absolute data offset precisely.
	dataOffset, alignment, err := computeDataOffset(r, h)
	if err != nil {
		return nil, err
	}

	f := &File{
		Header:     h,
		KV:         kv,
		Tensors:    tensors,
		DataOffset: dataOffset,
		Alignment:  alignment,
	}
	return f, nil
}

func computeDataOffset(r io.ReaderAt, h Header) (dataOffset int64, alignment uint32, err error) {
	sr := io.NewSectionReader(r, 0, math.MaxInt64)
	cr := &countingReader{r: sr}
	br := cr

	var m [4]byte
	if _, err := io.ReadFull(br, m[:]); err != nil {
		return 0, 0, err
	}
	var version uint32
	if err := binary.Read(br, binary.LittleEndian, &version); err != nil {
		return 0, 0, err
	}
	var tensorCount uint64
	if err := binary.Read(br, binary.LittleEndian, &tensorCount); err != nil {
		return 0, 0, err
	}
	var kvCount uint64
	if err := binary.Read(br, binary.LittleEndian, &kvCount); err != nil {
		return 0, 0, err
	}
	_ = version
	_ = tensorCount

	alignment = 32
	for i := uint64(0); i < kvCount; i++ {
		key, err := readString(br)
		if err != nil {
			return 0, 0, err
		}
		var t ValueType
		if err := binary.Read(br, binary.LittleEndian, &t); err != nil {
			return 0, 0, err
		}
		val, err := readValue(br, t)
		if err != nil {
			return 0, 0, err
		}
		if key == "general.alignment" {
			if v, ok := val.(uint32); ok {
				alignment = v
			}
		}
	}

	for i := uint64(0); i < h.TensorCount; i++ {
		if _, err := readString(br); err != nil {
			return 0, 0, err
		}
		var nDims uint32
		if err := binary.Read(br, binary.LittleEndian, &nDims); err != nil {
			return 0, 0, err
		}
		for d := uint32(0); d < nDims; d++ {
			var tmp uint64
			if err := binary.Read(br, binary.LittleEndian, &tmp); err != nil {
				return 0, 0, err
			}
		}
		var ggmlType uint32
		if err := binary.Read(br, binary.LittleEndian, &ggmlType); err != nil {
			return 0, 0, err
		}
		var off uint64
		if err := binary.Read(br, binary.LittleEndian, &off); err != nil {
			return 0, 0, err
		}
		_ = ggmlType
		_ = off
	}

	pos := cr.n
	pad := int64(0)
	if a := int64(alignment); a > 0 {
		rem := pos % a
		if rem != 0 {
			pad = a - rem
		}
	}
	return pos + pad, alignment, nil
}

type countingReader struct {
	r io.Reader
	n int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += int64(n)
	return n, err
}

func readString(r io.Reader) (string, error) {
	var n uint64
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return "", err
	}
	if n > 1<<30 {
		return "", fmt.Errorf("string too large: %d", n)
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(r, b); err != nil {
		return "", err
	}
	return string(b), nil
}

func readValue(r io.Reader, t ValueType) (any, error) {
	switch t {
	case TypeUint8:
		var v uint8
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeInt8:
		var v int8
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeUint16:
		var v uint16
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeInt16:
		var v int16
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeUint32:
		var v uint32
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeInt32:
		var v int32
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeUint64:
		var v uint64
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeInt64:
		var v int64
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeFloat32:
		var v float32
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeFloat64:
		var v float64
		return v, binary.Read(r, binary.LittleEndian, &v)
	case TypeBool:
		var v uint8
		if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
			return nil, err
		}
		return v != 0, nil
	case TypeString:
		return readString(r)
	case TypeArray:
		var elemType ValueType
		if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
			return nil, err
		}
		var n uint64
		if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
			return nil, err
		}
		if n > 1<<28 {
			return nil, fmt.Errorf("array too large: %d", n)
		}
		switch elemType {
		case TypeString:
			out := make([]string, 0, int(n))
			for i := uint64(0); i < n; i++ {
				s, err := readString(r)
				if err != nil {
					return nil, err
				}
				out = append(out, s)
			}
			return out, nil
		case TypeInt32:
			out := make([]int32, n)
			for i := uint64(0); i < n; i++ {
				if err := binary.Read(r, binary.LittleEndian, &out[i]); err != nil {
					return nil, err
				}
			}
			return out, nil
		case TypeUint32:
			out := make([]uint32, n)
			for i := uint64(0); i < n; i++ {
				if err := binary.Read(r, binary.LittleEndian, &out[i]); err != nil {
					return nil, err
				}
			}
			return out, nil
		case TypeFloat32:
			out := make([]float32, n)
			for i := uint64(0); i < n; i++ {
				if err := binary.Read(r, binary.LittleEndian, &out[i]); err != nil {
					return nil, err
				}
			}
			return out, nil
		case TypeInt8:
			out := make([]int8, n)
			tmp := make([]byte, n)
			if _, err := io.ReadFull(r, tmp); err != nil {
				return nil, err
			}
			for i := range tmp {
				out[i] = int8(tmp[i])
			}
			return out, nil
		case TypeUint8:
			out := make([]uint8, n)
			if _, err := io.ReadFull(r, out); err != nil {
				return nil, err
			}
			return out, nil
		default:
			// generic path
			out := make([]any, 0, int(n))
			for i := uint64(0); i < n; i++ {
				v, err := readValue(r, elemType)
				if err != nil {
					return nil, err
				}
				out = append(out, v)
			}
			return out, nil
		}
	default:
		return nil, fmt.Errorf("unsupported gguf value type: %d", t)
	}
}

func (f *File) GetString(key string) (string, bool) {
	kv, ok := f.KV[key]
	if !ok {
		return "", false
	}
	s, ok := kv.Value.(string)
	return s, ok
}

func (f *File) GetU32(key string) (uint32, bool) {
	kv, ok := f.KV[key]
	if !ok {
		return 0, false
	}
	switch v := kv.Value.(type) {
	case uint32:
		return v, true
	case uint64:
		if v <= math.MaxUint32 {
			return uint32(v), true
		}
	}
	return 0, false
}

func (f *File) GetI32(key string) (int32, bool) {
	kv, ok := f.KV[key]
	if !ok {
		return 0, false
	}
	v, ok := kv.Value.(int32)
	return v, ok
}

func (f *File) GetU64(key string) (uint64, bool) {
	kv, ok := f.KV[key]
	if !ok {
		return 0, false
	}
	v, ok := kv.Value.(uint64)
	return v, ok
}

func (f *File) GetF32(key string) (float32, bool) {
	kv, ok := f.KV[key]
	if !ok {
		return 0, false
	}
	v, ok := kv.Value.(float32)
	return v, ok
}

func (f *File) GetStrings(key string) ([]string, bool) {
	kv, ok := f.KV[key]
	if !ok {
		return nil, false
	}
	v, ok := kv.Value.([]string)
	return v, ok
}

func (f *File) FindTensor(name string) (TensorInfo, bool) {
	for _, t := range f.Tensors {
		if t.Name == name {
			return t, true
		}
	}
	return TensorInfo{}, false
}

var ErrUnsupported = errors.New("unsupported")

func AlignUp(off int64, alignment uint32) int64 {
	a := int64(alignment)
	if a <= 0 {
		return off
	}
	rem := off % a
	if rem == 0 {
		return off
	}
	return off + (a - rem)
}

func (t TensorInfo) ElemCount() uint64 {
	if t.NDims == 0 {
		return 0
	}
	c := uint64(1)
	for _, d := range t.Dims {
		c *= d
	}
	return c
}

func (t TensorInfo) Shape() []uint64 {
	out := make([]uint64, len(t.Dims))
	copy(out, t.Dims)
	return out
}

// Helper: turns file offset into absolute offset.
func (f *File) TensorAbsOffset(t TensorInfo) int64 {
	return f.DataOffset + int64(t.Offset)
}

func bytesTrimNul(b []byte) []byte {
	return bytes.TrimRight(b, "\x00")
}
