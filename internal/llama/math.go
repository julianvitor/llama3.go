package llama

import "math"

func rmsNorm(out, x, w []float32, eps float32) {
	var ss float32
	for i := 0; i < len(x); i++ {
		ss += x[i] * x[i]
	}
	scale := float32(1.0 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	for i := 0; i < len(x); i++ {
		out[i] = x[i] * scale * w[i]
	}
}

func silu(x float32) float32 {
	// x / (1 + exp(-x))
	return x / (1.0 + float32(math.Exp(float64(-x))))
}
