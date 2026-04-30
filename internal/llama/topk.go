package llama

// TopKItem holds a token id and its logit.
// Used to avoid allocating/processing full vocab logits when sampling with top-k.
type TopKItem struct {
	ID    int32
	Logit float32
}

func keepTopKInsert(best []TopKItem, k int, cand TopKItem) []TopKItem {
	if k <= 0 {
		return best
	}
	if len(best) < k {
		return append(best, cand)
	}
	// find min
	minIdx := 0
	minVal := best[0].Logit
	for i := 1; i < len(best); i++ {
		if best[i].Logit < minVal {
			minVal = best[i].Logit
			minIdx = i
		}
	}
	if cand.Logit > minVal {
		best[minIdx] = cand
	}
	return best
}

func mergeTopK(a, b []TopKItem, k int) []TopKItem {
	out := make([]TopKItem, 0, k)
	for _, x := range a {
		out = keepTopKInsert(out, k, x)
	}
	for _, x := range b {
		out = keepTopKInsert(out, k, x)
	}
	return out
}
