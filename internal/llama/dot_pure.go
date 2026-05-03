//go:build !cgo

package llama

func dot(a, b []float32) float32 {
    var sum float32
    for i := 0; i < len(a); i++ {
        sum += a[i] * b[i]
    }
    return sum
}