package llama

// Workspace holds reusable buffers for inference to avoid per-token allocations.
// It is NOT safe for concurrent use.

type Workspace struct {
	CtxLen int

	X       []float32
	XN      []float32
	Attn    []float32
	Proj    []float32
	FFNIn   []float32
	FFNUp   []float32
	FFNGate []float32
	FFNAct  []float32
	FFNDown []float32

	Q []float32
	K []float32
	V []float32

	// Attention scores scratch (per head): nHeads*ctxLen
	Scores []float32

	// Q8_0 quantization scratch (reused across matvec calls)
	Q8Tmp Q8Vector

	// TopK scratch buffers
	TopKBuf   []TopKItem
	TopKLocal []TopKItem
	TopKLens  []int
}

func (m *Model) NewWorkspace(ctxLen int, topK int) *Workspace {
	if ctxLen <= 0 {
		ctxLen = m.HP.CtxLen
	}
	if ctxLen > m.HP.CtxLen {
		ctxLen = m.HP.CtxLen
	}
	if topK <= 0 {
		topK = 40
	}

	nEmb := m.HP.NEmb
	nFF := m.HP.NFF
	nHeads := m.HP.NHeads
	headDim := nEmb / nHeads

	ws := &Workspace{CtxLen: ctxLen}
	ws.X = make([]float32, nEmb)
	ws.XN = make([]float32, nEmb)
	ws.Attn = make([]float32, nEmb)
	ws.Proj = make([]float32, nEmb)
	ws.FFNIn = make([]float32, nEmb)
	ws.FFNUp = make([]float32, nFF)
	ws.FFNGate = make([]float32, nFF)
	ws.FFNAct = make([]float32, nFF)
	ws.FFNDown = make([]float32, nEmb)

	ws.Q = make([]float32, nEmb)
	ws.K = make([]float32, m.HP.NHeadsKV*headDim)
	ws.V = make([]float32, m.HP.NHeadsKV*headDim)

	ws.Scores = make([]float32, nHeads*ctxLen)
	ws.TopKBuf = make([]TopKItem, 0, topK)
	return ws
}
