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
	wg    *sync.WaitGroup
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
				it.wg.Done()
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
	var wg sync.WaitGroup
	for i := 0; i < w; i++ {
		start := i * chunk
		end := start + chunk
		if start >= n {
			break
		}
		if end > n {
			end = n
		}
		wg.Add(1)
		p.ch <- workItem{start: start, end: end, fn: fn, wg: &wg}
	}
	wg.Wait()
}
