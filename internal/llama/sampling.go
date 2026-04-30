package llama

import (
	"math"
	"math/rand"
	"time"
)

type Sampler struct {
	Temperature float32
	TopK        int
	Seed        int64
	rng         *rand.Rand

	tmpPairs []pair
	tmpExps  []float64
}

func NewSampler(temp float32, topK int, seed int64) *Sampler {
	if temp <= 0 {
		temp = 0.8
	}
	if topK <= 0 {
		topK = 40
	}
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	return &Sampler{Temperature: temp, TopK: topK, Seed: seed, rng: rand.New(rand.NewSource(seed))}
}

type pair struct {
	ID    int32
	Logit float32
}

func (s *Sampler) Sample(logits []float32) int32 {
	k := s.TopK
	if k > len(logits) {
		k = len(logits)
	}
	if cap(s.tmpPairs) < k {
		s.tmpPairs = make([]pair, 0, k)
	}
	best := s.tmpPairs[:0]

	// keep top-k by insertion (k small)
	for i, l := range logits {
		p := pair{ID: int32(i), Logit: l}
		if len(best) < k {
			best = append(best, p)
			continue
		}
		// find current min
		minIdx := 0
		minVal := best[0].Logit
		for j := 1; j < len(best); j++ {
			if best[j].Logit < minVal {
				minVal = best[j].Logit
				minIdx = j
			}
		}
		if p.Logit > minVal {
			best[minIdx] = p
		}
	}

	// softmax over top-k
	maxLogit := best[0].Logit
	for i := 1; i < len(best); i++ {
		if best[i].Logit > maxLogit {
			maxLogit = best[i].Logit
		}
	}
	if cap(s.tmpExps) < len(best) {
		s.tmpExps = make([]float64, len(best))
	} else {
		s.tmpExps = s.tmpExps[:len(best)]
	}
	exps := s.tmpExps
	var sum float64
	invT := 1.0 / float64(s.Temperature)
	for i := range best {
		v := float64((best[i].Logit - maxLogit)) * invT
		ev := math.Exp(v)
		exps[i] = ev
		sum += ev
	}

	r := s.rng.Float64() * sum
	acc := float64(0)
	for i := range best {
		acc += exps[i]
		if r <= acc {
			return best[i].ID
		}
	}
	return best[len(best)-1].ID
}

// SampleTopK samples from a precomputed top-k list of (id, logit).
// The list does not need to be sorted.
func (s *Sampler) SampleTopK(items []TopKItem) int32 {
	if len(items) == 0 {
		return 0
	}
	if cap(s.tmpPairs) < len(items) {
		s.tmpPairs = make([]pair, 0, len(items))
	}
	best := s.tmpPairs[:0]
	for _, it := range items {
		best = append(best, pair{ID: it.ID, Logit: it.Logit})
	}
	maxLogit := best[0].Logit
	for i := 1; i < len(best); i++ {
		if best[i].Logit > maxLogit {
			maxLogit = best[i].Logit
		}
	}
	if cap(s.tmpExps) < len(best) {
		s.tmpExps = make([]float64, len(best))
	} else {
		s.tmpExps = s.tmpExps[:len(best)]
	}
	exps := s.tmpExps
	var sum float64
	invT := 1.0 / float64(s.Temperature)
	for i := range best {
		v := float64(best[i].Logit-maxLogit) * invT
		ev := math.Exp(v)
		exps[i] = ev
		sum += ev
	}
	r := s.rng.Float64() * sum
	acc := float64(0)
	for i := range best {
		acc += exps[i]
		if r <= acc {
			return best[i].ID
		}
	}
	return best[len(best)-1].ID
}
