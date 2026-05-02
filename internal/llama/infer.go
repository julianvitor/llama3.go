package llama

import (
	"fmt"
	"math"
)

func (m *Model) Forward(s *State, token int32) ([]float32, error) {
	logits := make([]float32, m.HP.VocabSize)
	if err := m.ForwardInto(s, token, logits); err != nil {
		return nil, err
	}
	return logits, nil
}

func (m *Model) ForwardInto(s *State, token int32, logitsOut []float32) error {
	ws := m.NewWorkspace(s.KV.CtxLen, 0)
	return m.ForwardIntoWorkspace(s, token, logitsOut, ws)
}

func (m *Model) ForwardIntoWorkspace(s *State, token int32, logitsOut []float32, ws *Workspace) error {
	if s.Pos >= s.KV.CtxLen {
		return fmt.Errorf("context length exceeded (%d)", s.KV.CtxLen)
	}
	if logitsOut != nil && len(logitsOut) != m.HP.VocabSize {
		return fmt.Errorf("logitsOut len=%d, want %d", len(logitsOut), m.HP.VocabSize)
	}
	if ws == nil {
		return fmt.Errorf("workspace is nil")
	}
	if ws.CtxLen < s.KV.CtxLen {
		return fmt.Errorf("workspace ctxLen=%d < state ctxLen=%d", ws.CtxLen, s.KV.CtxLen)
	}

	x := ws.X
	if err := m.embedInto(token, x); err != nil {
		return err
	}

	nEmb := m.HP.NEmb
	nHeads := m.HP.NHeads
	nHeadsKV := m.HP.NHeadsKV
	headDim := nEmb / nHeads
	ropeDim := m.HP.RopeDim
	if ropeDim > headDim {
		ropeDim = headDim
	}
	group := nHeads / nHeadsKV
	if group <= 0 {
		group = 1
	}

	xn, attnOut, projOut := ws.XN, ws.Attn, ws.Proj
	ffnIn, ffnAct, ffnUp, ffnGate, ffnDown := ws.FFNIn, ws.FFNAct, ws.FFNUp, ws.FFNGate, ws.FFNDown
	q, k, v := ws.Q, ws.K, ws.V
	scoresScratch := ws.Scores

	for layer := 0; layer < m.HP.NLayers; layer++ {
		lw := m.Layers[layer]

		// attn norm
		rmsNorm(xn, x, m.AttnNormW[layer], m.HP.RmsEps)

		// q,k,v em paralelo usando Pool.Run()
		var errQ, errK, errV error
		isQ8 := lw.Wq.Type() == ggmlTypeQ8_0
		if isQ8 {
			QuantizeQ8_0Into(xn, &ws.Q8Tmp)
		}

		// Usar Pool.Run() para reduzir contention vs manual goroutines
		m.Pool.Run(
			func() {
				if isQ8 {
					errQ = m.matVecQ8_0(lw.Wq, ws.Q8Tmp, q)
				} else {
					errQ = m.matVec(lw.Wq, xn, q)
				}
			},
			func() {
				if isQ8 {
					errK = m.matVecQ8_0(lw.Wk, ws.Q8Tmp, k)
				} else {
					errK = m.matVec(lw.Wk, xn, k)
				}
			},
			func() {
				if isQ8 {
					errV = m.matVecQ8_0(lw.Wv, ws.Q8Tmp, v)
				} else {
					errV = m.matVec(lw.Wv, xn, v)
				}
			},
		)

		if errQ != nil {
			return errQ
		}
		if errK != nil {
			return errK
		}
		if errV != nil {
			return errV
		}

		// apply RoPE to q and k
		for h := 0; h < nHeads; h++ {
			qh := q[h*headDim : (h+1)*headDim]
			s.Rope.ApplyInPlace(qh[:ropeDim], s.Pos)
		}
		for kh := 0; kh < nHeadsKV; kh++ {
			kk := k[kh*headDim : (kh+1)*headDim]
			s.Rope.ApplyInPlace(kk[:ropeDim], s.Pos)
		}

		// write to KV cache
		for kh := 0; kh < nHeadsKV; kh++ {
			kk := k[kh*headDim : (kh+1)*headDim]
			vv := v[kh*headDim : (kh+1)*headDim]
			s.KV.SetK(layer, s.Pos, kh, kk)
			s.KV.SetV(layer, s.Pos, kh, vv)
		}

		// attention (parallel over heads)
		scale := float32(1.0 / math.Sqrt(float64(headDim)))
		m.Pool.ParallelFor(nHeads, func(hStart, hEnd int) {
			for h := hStart; h < hEnd; h++ {
				kvh := h / group
				if kvh >= nHeadsKV {
					kvh = nHeadsKV - 1
				}
				qh := q[h*headDim : (h+1)*headDim]
				base := h * ws.CtxLen
				scores := scoresScratch[base : base+s.Pos+1]
				maxScore := float32(-1e30)

				for t := 0; t <= s.Pos; t++ {
					kt := s.KV.GetK(layer, t, kvh)
					var dot float32
					for i := 0; i < headDim; i++ {
						dot += qh[i] * kt[i]
					}
					sc := dot * scale
					scores[t] = sc
					if sc > maxScore {
						maxScore = sc
					}
				}

				var sumExp float32
				for t := 0; t <= s.Pos; t++ {
					ev := float32(math.Exp(float64(scores[t] - maxScore)))
					scores[t] = ev
					sumExp += ev
				}
				inv := float32(1.0) / sumExp

				outHead := attnOut[h*headDim : (h+1)*headDim]
				for i := 0; i < headDim; i++ {
					outHead[i] = 0
				}
				for t := 0; t <= s.Pos; t++ {
					w := scores[t] * inv
					vt := s.KV.GetV(layer, t, kvh)
					for i := 0; i < headDim; i++ {
						outHead[i] += w * vt[i]
					}
				}
			}
		})

		// project attention output and residual
		if lw.Wo.Type() == ggmlTypeQ8_0 {
			QuantizeQ8_0Into(attnOut, &ws.Q8Tmp)
			if err := m.matVecQ8_0(lw.Wo, ws.Q8Tmp, projOut); err != nil {
				return err
			}
		} else {
			if err := m.matVec(lw.Wo, attnOut, projOut); err != nil {
				return err
			}
		}
		for i := 0; i < nEmb; i++ {
			x[i] += projOut[i]
		}

		// ffn norm
		rmsNorm(ffnIn, x, m.FFNNormW[layer], m.HP.RmsEps)

		// ffn up and gate em paralelo
		var errUp, errGate error
		isFFNQ8 := lw.Wup.Type() == ggmlTypeQ8_0
		if isFFNQ8 {
			QuantizeQ8_0Into(ffnIn, &ws.Q8Tmp)
		}

		// Usar Pool.Run() para paralelizar FFN up e gate
		m.Pool.Run(
			func() {
				if isFFNQ8 {
					errUp = m.matVecQ8_0(lw.Wup, ws.Q8Tmp, ffnUp)
				} else {
					errUp = m.matVec(lw.Wup, ffnIn, ffnUp)
				}
			},
			func() {
				if isFFNQ8 {
					errGate = m.matVecQ8_0(lw.Wgate, ws.Q8Tmp, ffnGate)
				} else {
					errGate = m.matVec(lw.Wgate, ffnIn, ffnGate)
				}
			},
		)

		if errUp != nil {
			return errUp
		}
		if errGate != nil {
			return errGate
		}

		for i := 0; i < m.HP.NFF; i++ {
			ffnAct[i] = silu(ffnGate[i]) * ffnUp[i]
		}

		if lw.Wdown.Type() == ggmlTypeQ8_0 {
			QuantizeQ8_0Into(ffnAct, &ws.Q8Tmp)
			if err := m.matVecQ8_0(lw.Wdown, ws.Q8Tmp, ffnDown); err != nil {
				return err
			}
		} else {
			if err := m.matVec(lw.Wdown, ffnAct, ffnDown); err != nil {
				return err
			}
		}
		for i := 0; i < nEmb; i++ {
			x[i] += ffnDown[i]
		}
	}

	if logitsOut != nil {
		rmsNorm(xn, x, m.OutNormW, m.HP.RmsEps)
		if m.OutProj.Type() == ggmlTypeQ8_0 {
			QuantizeQ8_0Into(xn, &ws.Q8Tmp)
			if err := m.matVecQ8_0(m.OutProj, ws.Q8Tmp, logitsOut); err != nil {
				return err
			}
		} else {
			if err := m.matVec(m.OutProj, xn, logitsOut); err != nil {
				return err
			}
		}
	}

	s.Pos++
	return nil
}

func (m *Model) ForwardTopKIntoWorkspace(s *State, token int32, topK int, ws *Workspace) ([]TopKItem, error) {
	if s.Pos >= s.KV.CtxLen {
		return nil, fmt.Errorf("context length exceeded (%d)", s.KV.CtxLen)
	}
	if ws == nil {
		return nil, fmt.Errorf("workspace is nil")
	}
	if ws.CtxLen < s.KV.CtxLen {
		return nil, fmt.Errorf("workspace ctxLen=%d < state ctxLen=%d", ws.CtxLen, s.KV.CtxLen)
	}
	if topK <= 0 {
		topK = 40
	}

	x := ws.X
	if err := m.embedInto(token, x); err != nil {
		return nil, err
	}

	nEmb := m.HP.NEmb
	nHeads := m.HP.NHeads
	nHeadsKV := m.HP.NHeadsKV
	headDim := nEmb / nHeads
	ropeDim := m.HP.RopeDim
	if ropeDim > headDim {
		ropeDim = headDim
	}
	group := nHeads / nHeadsKV
	if group <= 0 {
		group = 1
	}

	xn, attnOut, projOut := ws.XN, ws.Attn, ws.Proj
	ffnIn, ffnAct, ffnUp, ffnGate, ffnDown := ws.FFNIn, ws.FFNAct, ws.FFNUp, ws.FFNGate, ws.FFNDown
	q, k, v := ws.Q, ws.K, ws.V
	scoresScratch := ws.Scores

	for layer := 0; layer < m.HP.NLayers; layer++ {
		lw := m.Layers[layer]
		rmsNorm(xn, x, m.AttnNormW[layer], m.HP.RmsEps)

		// q,k,v em paralelo usando Pool.Run()
		var errQ, errK, errV error
		isQ8 := lw.Wq.Type() == ggmlTypeQ8_0
		if isQ8 {
			QuantizeQ8_0Into(xn, &ws.Q8Tmp)
		}

		// Usar Pool.Run() para reduzir contention vs manual goroutines
		m.Pool.Run(
			func() {
				if isQ8 {
					errQ = m.matVecQ8_0(lw.Wq, ws.Q8Tmp, q)
				} else {
					errQ = m.matVec(lw.Wq, xn, q)
				}
			},
			func() {
				if isQ8 {
					errK = m.matVecQ8_0(lw.Wk, ws.Q8Tmp, k)
				} else {
					errK = m.matVec(lw.Wk, xn, k)
				}
			},
			func() {
				if isQ8 {
					errV = m.matVecQ8_0(lw.Wv, ws.Q8Tmp, v)
				} else {
					errV = m.matVec(lw.Wv, xn, v)
				}
			},
		)

		if errQ != nil {
			return nil, errQ
		}
		if errK != nil {
			return nil, errK
		}
		if errV != nil {
			return nil, errV
		}

		for h := 0; h < nHeads; h++ {
			qh := q[h*headDim : (h+1)*headDim]
			s.Rope.ApplyInPlace(qh[:ropeDim], s.Pos)
		}
		for kh := 0; kh < nHeadsKV; kh++ {
			kk := k[kh*headDim : (kh+1)*headDim]
			s.Rope.ApplyInPlace(kk[:ropeDim], s.Pos)
		}

		for kh := 0; kh < nHeadsKV; kh++ {
			kk := k[kh*headDim : (kh+1)*headDim]
			vv := v[kh*headDim : (kh+1)*headDim]
			s.KV.SetK(layer, s.Pos, kh, kk)
			s.KV.SetV(layer, s.Pos, kh, vv)
		}

		scale := float32(1.0 / math.Sqrt(float64(headDim)))
		m.Pool.ParallelFor(nHeads, func(hStart, hEnd int) {
			for h := hStart; h < hEnd; h++ {
				kvh := h / group
				if kvh >= nHeadsKV {
					kvh = nHeadsKV - 1
				}
				qh := q[h*headDim : (h+1)*headDim]

				base := h * ws.CtxLen
				scores := scoresScratch[base : base+s.Pos+1]
				maxScore := float32(-1e30)
				for t := 0; t <= s.Pos; t++ {
					kt := s.KV.GetK(layer, t, kvh)
					var dot float32
					for i := 0; i < headDim; i++ {
						dot += qh[i] * kt[i]
					}
					sc := dot * scale
					scores[t] = sc
					if sc > maxScore {
						maxScore = sc
					}
				}
				var sumExp float32
				for t := 0; t <= s.Pos; t++ {
					ev := float32(math.Exp(float64(scores[t] - maxScore)))
					scores[t] = ev
					sumExp += ev
				}
				inv := float32(1.0) / sumExp

				outHead := attnOut[h*headDim : (h+1)*headDim]
				for i := 0; i < headDim; i++ {
					outHead[i] = 0
				}
				for t := 0; t <= s.Pos; t++ {
					w := scores[t] * inv
					vt := s.KV.GetV(layer, t, kvh)
					for i := 0; i < headDim; i++ {
						outHead[i] += w * vt[i]
					}
				}
			}
		})

		if lw.Wo.Type() == ggmlTypeQ8_0 {
			QuantizeQ8_0Into(attnOut, &ws.Q8Tmp)
			if err := m.matVecQ8_0(lw.Wo, ws.Q8Tmp, projOut); err != nil {
				return nil, err
			}
		} else {
			if err := m.matVec(lw.Wo, attnOut, projOut); err != nil {
				return nil, err
			}
		}
		for i := 0; i < nEmb; i++ {
			x[i] += projOut[i]
		}

		rmsNorm(ffnIn, x, m.FFNNormW[layer], m.HP.RmsEps)

		// ffn up and gate em paralelo
		var errUp, errGate error
		isFFNQ8 := lw.Wup.Type() == ggmlTypeQ8_0
		if isFFNQ8 {
			QuantizeQ8_0Into(ffnIn, &ws.Q8Tmp)
		}

		// Usar Pool.Run() para paralelizar FFN up e gate
		m.Pool.Run(
			func() {
				if isFFNQ8 {
					errUp = m.matVecQ8_0(lw.Wup, ws.Q8Tmp, ffnUp)
				} else {
					errUp = m.matVec(lw.Wup, ffnIn, ffnUp)
				}
			},
			func() {
				if isFFNQ8 {
					errGate = m.matVecQ8_0(lw.Wgate, ws.Q8Tmp, ffnGate)
				} else {
					errGate = m.matVec(lw.Wgate, ffnIn, ffnGate)
				}
			},
		)

		if errUp != nil {
			return nil, errUp
		}
		if errGate != nil {
			return nil, errGate
		}

		for i := 0; i < m.HP.NFF; i++ {
			ffnAct[i] = silu(ffnGate[i]) * ffnUp[i]
		}
		if lw.Wdown.Type() == ggmlTypeQ8_0 {
			QuantizeQ8_0Into(ffnAct, &ws.Q8Tmp)
			if err := m.matVecQ8_0(lw.Wdown, ws.Q8Tmp, ffnDown); err != nil {
				return nil, err
			}
		} else {
			if err := m.matVec(lw.Wdown, ffnAct, ffnDown); err != nil {
				return nil, err
			}
		}
		for i := 0; i < nEmb; i++ {
			x[i] += ffnDown[i]
		}
	}

	rmsNorm(xn, x, m.OutNormW, m.HP.RmsEps)
	if m.OutProj.Type() != ggmlTypeQ8_0 {
		return nil, fmt.Errorf("ForwardTopK requires Q8_0 output projection")
	}
	QuantizeQ8_0Into(xn, &ws.Q8Tmp)
	items, err := m.matVecTopKWS(m.OutProj, ws.Q8Tmp, topK, ws)
	if err != nil {
		return nil, err
	}

	s.Pos++
	return items, nil
}

func (m *Model) embedInto(token int32, dst []float32) error {
	if token < 0 || int(token) >= m.HP.VocabSize {
		return fmt.Errorf("token out of range: %d", token)
	}
	return readRowVecAnyToF32Into(dst, m.TokenEmb, int(token))
}
