package llama

import (
	"math"
)

type RopeCache struct {
	CtxLen  int
	RopeDim int
	Half    int
	Cos     []float32 // [pos*Half + i]
	Sin     []float32
}

func NewRopeCache(ctxLen, ropeDim int, base float32) *RopeCache {
	half := ropeDim / 2
	cos := make([]float32, ctxLen*half)
	sin := make([]float32, ctxLen*half)

	invFreq := make([]float64, half)
	b := float64(base)
	for i := 0; i < half; i++ {
		// i corresponds to 2*i dim
		exp := float64(2*i) / float64(ropeDim)
		invFreq[i] = 1.0 / math.Pow(b, exp)
	}

	for pos := 0; pos < ctxLen; pos++ {
		for i := 0; i < half; i++ {
			ang := float64(pos) * invFreq[i]
			cos[pos*half+i] = float32(math.Cos(ang))
			sin[pos*half+i] = float32(math.Sin(ang))
		}
	}

	return &RopeCache{CtxLen: ctxLen, RopeDim: ropeDim, Half: half, Cos: cos, Sin: sin}
}

func (r *RopeCache) ApplyInPlace(vec []float32, pos int) {
	// vec length must be >= RopeDim
	h := r.Half
	base := pos * h
	for i := 0; i < h; i++ {
		c := r.Cos[base+i]
		s := r.Sin[base+i]
		idx := 2 * i
		x0 := vec[idx]
		x1 := vec[idx+1]
		vec[idx] = x0*c - x1*s
		vec[idx+1] = x0*s + x1*c
	}
}
