package llama

import (
	"runtime"
	"sync"
)

type WorkerPool struct {
	workers int
	ch      chan workItem
	once    sync.Once
}

type workItem struct {
	start int
	end   int
	fn    func(start, end int)
	done  chan struct{}
}

func NewWorkerPool(workers int) *WorkerPool {
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	p := &WorkerPool{workers: workers, ch: make(chan workItem, workers*4)}
	for i := 0; i < p.workers; i++ {
		go func() {
			for it := range p.ch {
				it.fn(it.start, it.end)
				close(it.done)
			}
		}()
	}
	return p
}

func (p *WorkerPool) Workers() int { return p.workers }

func (p *WorkerPool) Close() {
	p.once.Do(func() {
		if p.ch != nil {
			close(p.ch)
		}
	})
}

// Run executa múltiplas tarefas independentes em paralelo.
// Cada tarefa é executada em um worker diferente quando possível.
func (p *WorkerPool) Run(tasks ...func()) {
	if len(tasks) == 0 {
		return
	}
	if len(tasks) == 1 {
		tasks[0]()
		return
	}

	// Para poucas tarefas, usar canais diretos
	if len(tasks) < p.workers {
		dones := make([]chan struct{}, len(tasks))
		for i := range dones {
			dones[i] = make(chan struct{})
		}
		for i, task := range tasks {
			done := dones[i]
			task := task
			p.ch <- workItem{
				start: 0,
				end:   1,
				fn:    func(start, end int) { task() },
				done:  done,
			}
		}
		for _, done := range dones {
			<-done
		}
		return
	}

	// Para muitas tarefas, usar ParallelFor
	p.ParallelFor(len(tasks), func(start, end int) {
		for i := start; i < end; i++ {
			tasks[i]()
		}
	})
}

func (p *WorkerPool) ParallelFor(n int, fn func(start, end int)) {
	if n <= 0 {
		return
	}
	w := p.workers
	if w <= 1 || n < w*64 || p.ch == nil {
		fn(0, n)
		return
	}

	chunk := (n + w - 1) / w
	dones := make([]chan struct{}, 0, w)
	for i := 0; i < w; i++ {
		start := i * chunk
		end := start + chunk
		if start >= n {
			break
		}
		if end > n {
			end = n
		}
		done := make(chan struct{})
		dones = append(dones, done)
		p.ch <- workItem{start: start, end: end, fn: fn, done: done}
	}
	for _, done := range dones {
		<-done
	}
}
