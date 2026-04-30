package llama

import "math"

// float16 (IEEE 754 half) helpers.

func f16ToF32(u uint16) float32 {
	sign := uint32(u>>15) & 0x1
	exp := uint32(u>>10) & 0x1f
	frac := uint32(u & 0x03ff)

	var out uint32
	switch exp {
	case 0:
		if frac == 0 {
			out = sign << 31
		} else {
			// subnormal
			e := int32(-14)
			m := float32(frac) / 1024.0
			v := float32(math.Ldexp(float64(m), int(e)))
			if sign != 0 {
				v = -v
			}
			return v
		}
	case 0x1f:
		if frac == 0 {
			out = (sign << 31) | 0x7f800000
		} else {
			out = (sign << 31) | 0x7f800000 | (frac << 13)
		}
	default:
		outExp := (exp - 15 + 127) & 0xff
		out = (sign << 31) | (outExp << 23) | (frac << 13)
	}
	return math.Float32frombits(out)
}
