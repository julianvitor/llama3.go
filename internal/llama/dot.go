package llama

func (m *Model) Dot(a, b []float32) float32 {
    return dot(a, b)
}