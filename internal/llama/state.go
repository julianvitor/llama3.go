package llama

type KVCache struct {
	NLayers  int
	CtxLen   int
	NHeadsKV int
	HeadDim  int
	K        []float32
	V        []float32
}

func NewKVCache(nLayers, ctxLen, nHeadsKV, headDim int) *KVCache {
	sz := nLayers * ctxLen * nHeadsKV * headDim
	return &KVCache{
		NLayers: nLayers, CtxLen: ctxLen, NHeadsKV: nHeadsKV, HeadDim: headDim,
		K: make([]float32, sz),
		V: make([]float32, sz),
	}
}

func (kv *KVCache) idx(layer, pos, head, d int) int {
	return (((layer*kv.CtxLen+pos)*kv.NHeadsKV)+head)*kv.HeadDim + d
}

func (kv *KVCache) SetK(layer, pos, head int, vec []float32) {
	base := kv.idx(layer, pos, head, 0)
	copy(kv.K[base:base+kv.HeadDim], vec)
}

func (kv *KVCache) SetV(layer, pos, head int, vec []float32) {
	base := kv.idx(layer, pos, head, 0)
	copy(kv.V[base:base+kv.HeadDim], vec)
}

func (kv *KVCache) GetK(layer, pos, head int) []float32 {
	base := kv.idx(layer, pos, head, 0)
	return kv.K[base : base+kv.HeadDim]
}

func (kv *KVCache) GetV(layer, pos, head int) []float32 {
	base := kv.idx(layer, pos, head, 0)
	return kv.V[base : base+kv.HeadDim]
}

type State struct {
	Pos  int
	KV   *KVCache
	Rope *RopeCache
}

func (m *Model) NewState(ctxLen int) *State {
	if ctxLen <= 0 {
		ctxLen = m.HP.CtxLen
	}
	if ctxLen > m.HP.CtxLen {
		ctxLen = m.HP.CtxLen
	}
	headDim := m.HP.NEmb / m.HP.NHeads
	return &State{
		Pos:  0,
		KV:   NewKVCache(m.HP.NLayers, ctxLen, m.HP.NHeadsKV, headDim),
		Rope: NewRopeCache(ctxLen, m.HP.RopeDim, m.HP.RopeBase),
	}
}
