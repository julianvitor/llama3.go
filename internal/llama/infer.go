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

	// 1. Embedding: converte o token ID em um vetor denso inicial (hidden state)
	hiddenState := ws.X
	if err := m.embedInto(token, hiddenState); err != nil {
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
	// Multi-Query Attention (MQA) ou Grouped-Query Attention (GQA):
	// define quantos heads de query compartilham o mesmo head de key/value
	group := nHeads / nHeadsKV
	if group <= 0 {
		group = 1
	}

	// Buffers temporários do workspace
	normedState, attentionOut, attentionProj := ws.XN, ws.Attn, ws.Proj
	ffnIn, ffnAct, ffnUp, ffnGate, ffnDown := ws.FFNIn, ws.FFNAct, ws.FFNUp, ws.FFNGate, ws.FFNDown
	query, key, value := ws.Q, ws.K, ws.V
	attentionScores := ws.Scores

	// Loop principal sobre as camadas do Transformer
	for layer := 0; layer < m.HP.NLayers; layer++ {
		layerWeights := m.Layers[layer]

		// 2. Pré-Normalização (RMSNorm) antes da camada de Atenção
		rmsNorm(normedState, hiddenState, m.AttnNormW[layer], m.HP.RmsEps)

		// 3. Projeções Lineares Q, K, V (em paralelo)
		var errQ, errK, errV error
		isQ8 := layerWeights.Wq.Type() == ggmlTypeQ8_0
		if isQ8 {
			QuantizeQ8_0Into(normedState, &ws.Q8Tmp)
		}

		// Usar Pool.Run() para reduzir contention vs manual goroutines
		m.Pool.Run(
			func() {
				if isQ8 {
					errQ = m.matVecQ8_0(layerWeights.Wq, ws.Q8Tmp, query)
				} else {
					errQ = m.matVec(layerWeights.Wq, normedState, query)
				}
			},
			func() {
				if isQ8 {
					errK = m.matVecQ8_0(layerWeights.Wk, ws.Q8Tmp, key)
				} else {
					errK = m.matVec(layerWeights.Wk, normedState, key)
				}
			},
			func() {
				if isQ8 {
					errV = m.matVecQ8_0(layerWeights.Wv, ws.Q8Tmp, value)
				} else {
					errV = m.matVec(layerWeights.Wv, normedState, value)
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

		// 4. Rotary Positional Embeddings (RoPE): adiciona informação posicional
		for h := 0; h < nHeads; h++ {
			queryHead := query[h*headDim : (h+1)*headDim]
			s.Rope.ApplyInPlace(queryHead[:ropeDim], s.Pos)
		}
		for kh := 0; kh < nHeadsKV; kh++ {
			keyHead := key[kh*headDim : (kh+1)*headDim]
			s.Rope.ApplyInPlace(keyHead[:ropeDim], s.Pos)
		}

		// 5. Salva Key e Value no cache (KV Cache) para uso futuro em auto-regressão
		for kh := 0; kh < nHeadsKV; kh++ {
			keyHead := key[kh*headDim : (kh+1)*headDim]
			valueHead := value[kh*headDim : (kh+1)*headDim]
			s.KV.SetK(layer, s.Pos, kh, keyHead)
			s.KV.SetV(layer, s.Pos, kh, valueHead)
		}

		// 6. Calcula a Atenção (Multi-Head Attention) paralelizada pelos heads
		scale := float32(1.0 / math.Sqrt(float64(headDim)))
		m.Pool.ParallelFor(nHeads, func(hStart, hEnd int) {
			for h := hStart; h < hEnd; h++ {
				kvHeadIdx := h / group
				if kvHeadIdx >= nHeadsKV {
					kvHeadIdx = nHeadsKV - 1
				}
				queryHead := query[h*headDim : (h+1)*headDim]
				base := h * ws.CtxLen
				scores := attentionScores[base : base+s.Pos+1]
				maxScore := float32(-1e30)

				// Calcula os scores de atenção (Query . Key)
				for t := 0; t <= s.Pos; t++ {
					keyAtT := s.KV.GetK(layer, t, kvHeadIdx)
					dot := m.Dot(queryHead, keyAtT)
					score := dot * scale
					scores[t] = score
					if score > maxScore {
						maxScore = score
					}
				}

				// Softmax sobre os scores para obter os pesos de atenção
				var sumExp float32
				for t := 0; t <= s.Pos; t++ {
					expVal := float32(math.Exp(float64(scores[t] - maxScore)))
					scores[t] = expVal
					sumExp += expVal
				}
				invSumExp := float32(1.0) / sumExp

				// Combina os Values usando os pesos calculados (Attention = Softmax(Q.K) . V)
				outHead := attentionOut[h*headDim : (h+1)*headDim]
				for i := 0; i < headDim; i++ {
					outHead[i] = 0
				}
				for t := 0; t <= s.Pos; t++ {
					attentionWeight := scores[t] * invSumExp
					valueAtT := s.KV.GetV(layer, t, kvHeadIdx)
					for i := 0; i < headDim; i++ {
						outHead[i] += attentionWeight * valueAtT[i]
					}
				}
			}
		})

		// 7. Projeção de saída da atenção (Wo) e conexão residual (Add)
		if layerWeights.Wo.Type() == ggmlTypeQ8_0 {
			QuantizeQ8_0Into(attentionOut, &ws.Q8Tmp)
			if err := m.matVecQ8_0(layerWeights.Wo, ws.Q8Tmp, attentionProj); err != nil {
				return err
			}
		} else {
			if err := m.matVec(layerWeights.Wo, attentionOut, attentionProj); err != nil {
				return err
			}
		}
		// Conexão residual
		for i := 0; i < nEmb; i++ {
			hiddenState[i] += attentionProj[i]
		}

		// 8. Pré-Normalização para Feed-Forward Network (FFN)
		rmsNorm(ffnIn, hiddenState, m.FFNNormW[layer], m.HP.RmsEps)

		// 9. Feed-Forward Network (FFN) - Up e Gate projecions em paralelo
		var errUp, errGate error
		isFFNQ8 := layerWeights.Wup.Type() == ggmlTypeQ8_0
		if isFFNQ8 {
			QuantizeQ8_0Into(ffnIn, &ws.Q8Tmp)
		}

		m.Pool.Run(
			func() {
				if isFFNQ8 {
					errUp = m.matVecQ8_0(layerWeights.Wup, ws.Q8Tmp, ffnUp)
				} else {
					errUp = m.matVec(layerWeights.Wup, ffnIn, ffnUp)
				}
			},
			func() {
				if isFFNQ8 {
					errGate = m.matVecQ8_0(layerWeights.Wgate, ws.Q8Tmp, ffnGate)
				} else {
					errGate = m.matVec(layerWeights.Wgate, ffnIn, ffnGate)
				}
			},
		)

		if errUp != nil {
			return errUp
		}
		if errGate != nil {
			return errGate
		}

		// 10. Função de Ativação (SwiGLU)
		for i := 0; i < m.HP.NFF; i++ {
			ffnAct[i] = silu(ffnGate[i]) * ffnUp[i]
		}

		// 11. Projeção FFN Down e conexão residual
		if layerWeights.Wdown.Type() == ggmlTypeQ8_0 {
			QuantizeQ8_0Into(ffnAct, &ws.Q8Tmp)
			if err := m.matVecQ8_0(layerWeights.Wdown, ws.Q8Tmp, ffnDown); err != nil {
				return err
			}
		} else {
			if err := m.matVec(layerWeights.Wdown, ffnAct, ffnDown); err != nil {
				return err
			}
		}
		// Conexão residual
		for i := 0; i < nEmb; i++ {
			hiddenState[i] += ffnDown[i]
		}
	}

	// 12. Pós-Normalização e projeção final para obter Logits (apenas se solicitado)
	if logitsOut != nil {
		rmsNorm(normedState, hiddenState, m.OutNormW, m.HP.RmsEps)
		if m.OutProj.Type() == ggmlTypeQ8_0 {
			QuantizeQ8_0Into(normedState, &ws.Q8Tmp)
			if err := m.matVecQ8_0(m.OutProj, ws.Q8Tmp, logitsOut); err != nil {
				return err
			}
		} else {
			if err := m.matVec(m.OutProj, normedState, logitsOut); err != nil {
				return err
			}
		}
	}

	// Avança a posição no estado
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

	// 1. Embedding: converte o token ID em um vetor denso inicial (hidden state)
	hiddenState := ws.X
	if err := m.embedInto(token, hiddenState); err != nil {
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
	// Multi-Query Attention (MQA) ou Grouped-Query Attention (GQA)
	group := nHeads / nHeadsKV
	if group <= 0 {
		group = 1
	}

	// Buffers temporários do workspace
	normedState, attentionOut, attentionProj := ws.XN, ws.Attn, ws.Proj
	ffnIn, ffnAct, ffnUp, ffnGate, ffnDown := ws.FFNIn, ws.FFNAct, ws.FFNUp, ws.FFNGate, ws.FFNDown
	query, key, value := ws.Q, ws.K, ws.V
	attentionScores := ws.Scores

	// Loop principal sobre as camadas do Transformer
	for layer := 0; layer < m.HP.NLayers; layer++ {
		layerWeights := m.Layers[layer]

		// 2. Pré-Normalização (RMSNorm) antes da camada de Atenção
		rmsNorm(normedState, hiddenState, m.AttnNormW[layer], m.HP.RmsEps)

		// 3. Projeções Lineares Q, K, V (em paralelo)
		var errQ, errK, errV error
		isQ8 := layerWeights.Wq.Type() == ggmlTypeQ8_0
		if isQ8 {
			QuantizeQ8_0Into(normedState, &ws.Q8Tmp)
		}

		m.Pool.Run(
			func() {
				if isQ8 {
					errQ = m.matVecQ8_0(layerWeights.Wq, ws.Q8Tmp, query)
				} else {
					errQ = m.matVec(layerWeights.Wq, normedState, query)
				}
			},
			func() {
				if isQ8 {
					errK = m.matVecQ8_0(layerWeights.Wk, ws.Q8Tmp, key)
				} else {
					errK = m.matVec(layerWeights.Wk, normedState, key)
				}
			},
			func() {
				if isQ8 {
					errV = m.matVecQ8_0(layerWeights.Wv, ws.Q8Tmp, value)
				} else {
					errV = m.matVec(layerWeights.Wv, normedState, value)
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

		// 4. Rotary Positional Embeddings (RoPE)
		for h := 0; h < nHeads; h++ {
			queryHead := query[h*headDim : (h+1)*headDim]
			s.Rope.ApplyInPlace(queryHead[:ropeDim], s.Pos)
		}
		for kh := 0; kh < nHeadsKV; kh++ {
			keyHead := key[kh*headDim : (kh+1)*headDim]
			s.Rope.ApplyInPlace(keyHead[:ropeDim], s.Pos)
		}

		// 5. Salva Key e Value no cache (KV Cache)
		for kh := 0; kh < nHeadsKV; kh++ {
			keyHead := key[kh*headDim : (kh+1)*headDim]
			valueHead := value[kh*headDim : (kh+1)*headDim]
			s.KV.SetK(layer, s.Pos, kh, keyHead)
			s.KV.SetV(layer, s.Pos, kh, valueHead)
		}

		// 6. Calcula a Atenção (Multi-Head Attention) paralelizada pelos heads
		scale := float32(1.0 / math.Sqrt(float64(headDim)))
		m.Pool.ParallelFor(nHeads, func(hStart, hEnd int) {
			for h := hStart; h < hEnd; h++ {
				kvHeadIdx := h / group
				if kvHeadIdx >= nHeadsKV {
					kvHeadIdx = nHeadsKV - 1
				}
				queryHead := query[h*headDim : (h+1)*headDim]

				base := h * ws.CtxLen
				scores := attentionScores[base : base+s.Pos+1]
				maxScore := float32(-1e30)

				// Calcula os scores de atenção (Query . Key)
				for t := 0; t <= s.Pos; t++ {
					keyAtT := s.KV.GetK(layer, t, kvHeadIdx)
					dot := m.Dot(queryHead, keyAtT)
					score := dot * scale
					scores[t] = score
					if score > maxScore {
						maxScore = score
					}
				}

				// Softmax sobre os scores
				var sumExp float32
				for t := 0; t <= s.Pos; t++ {
					expVal := float32(math.Exp(float64(scores[t] - maxScore)))
					scores[t] = expVal
					sumExp += expVal
				}
				invSumExp := float32(1.0) / sumExp

				// Combina os Values
				outHead := attentionOut[h*headDim : (h+1)*headDim]
				for i := 0; i < headDim; i++ {
					outHead[i] = 0
				}
				for t := 0; t <= s.Pos; t++ {
					attentionWeight := scores[t] * invSumExp
					valueAtT := s.KV.GetV(layer, t, kvHeadIdx)
					for i := 0; i < headDim; i++ {
						outHead[i] += attentionWeight * valueAtT[i]
					}
				}
			}
		})

		// 7. Projeção de saída da atenção (Wo) e conexão residual (Add)
		if layerWeights.Wo.Type() == ggmlTypeQ8_0 {
			QuantizeQ8_0Into(attentionOut, &ws.Q8Tmp)
			if err := m.matVecQ8_0(layerWeights.Wo, ws.Q8Tmp, attentionProj); err != nil {
				return nil, err
			}
		} else {
			if err := m.matVec(layerWeights.Wo, attentionOut, attentionProj); err != nil {
				return nil, err
			}
		}
		// Conexão residual
		for i := 0; i < nEmb; i++ {
			hiddenState[i] += attentionProj[i]
		}

		// 8. Pré-Normalização para Feed-Forward Network (FFN)
		rmsNorm(ffnIn, hiddenState, m.FFNNormW[layer], m.HP.RmsEps)

		// 9. Feed-Forward Network (FFN) - Up e Gate projecions em paralelo
		var errUp, errGate error
		isFFNQ8 := layerWeights.Wup.Type() == ggmlTypeQ8_0
		if isFFNQ8 {
			QuantizeQ8_0Into(ffnIn, &ws.Q8Tmp)
		}

		m.Pool.Run(
			func() {
				if isFFNQ8 {
					errUp = m.matVecQ8_0(layerWeights.Wup, ws.Q8Tmp, ffnUp)
				} else {
					errUp = m.matVec(layerWeights.Wup, ffnIn, ffnUp)
				}
			},
			func() {
				if isFFNQ8 {
					errGate = m.matVecQ8_0(layerWeights.Wgate, ws.Q8Tmp, ffnGate)
				} else {
					errGate = m.matVec(layerWeights.Wgate, ffnIn, ffnGate)
				}
			},
		)

		if errUp != nil {
			return nil, errUp
		}
		if errGate != nil {
			return nil, errGate
		}

		// 10. Função de Ativação (SwiGLU)
		for i := 0; i < m.HP.NFF; i++ {
			ffnAct[i] = silu(ffnGate[i]) * ffnUp[i]
		}

		// 11. Projeção FFN Down e conexão residual
		if layerWeights.Wdown.Type() == ggmlTypeQ8_0 {
			QuantizeQ8_0Into(ffnAct, &ws.Q8Tmp)
			if err := m.matVecQ8_0(layerWeights.Wdown, ws.Q8Tmp, ffnDown); err != nil {
				return nil, err
			}
		} else {
			if err := m.matVec(layerWeights.Wdown, ffnAct, ffnDown); err != nil {
				return nil, err
			}
		}
		// Conexão residual
		for i := 0; i < nEmb; i++ {
			hiddenState[i] += ffnDown[i]
		}
	}

	// 12. Pós-Normalização e extração dos Top-K logits diretos
	rmsNorm(normedState, hiddenState, m.OutNormW, m.HP.RmsEps)
	if m.OutProj.Type() != ggmlTypeQ8_0 {
		return nil, fmt.Errorf("ForwardTopK requires Q8_0 output projection")
	}
	QuantizeQ8_0Into(normedState, &ws.Q8Tmp)
	items, err := m.matVecTopKWS(m.OutProj, ws.Q8Tmp, topK, ws)
	if err != nil {
		return nil, err
	}

	// Avança a posição no estado
	s.Pos++
	return items, nil
}

func (m *Model) embedInto(token int32, dst []float32) error {
	if token < 0 || int(token) >= m.HP.VocabSize {
		return fmt.Errorf("token out of range: %d", token)
	}
	return readRowVecAnyToF32Into(dst, m.TokenEmb, int(token))
}
