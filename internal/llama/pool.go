package llama

// TaskPool é um pool de tarefas que reutiliza a infraestrutura de WorkerPool
// para executar múltiplas tarefas independentes em paralelo.
type TaskPool struct {
	workerPool *WorkerPool
}

func NewTaskPool(workers int) *TaskPool {
	return &TaskPool{
		workerPool: NewWorkerPool(workers),
	}
}

// Run executa múltiplas tarefas independentes em paralelo.
// Cada tarefa é executada em um worker diferente quando possível.
func (p *TaskPool) Run(tasks ...func()) {
	if len(tasks) == 0 {
		return
	}
	if len(tasks) == 1 {
		tasks[0]()
		return
	}

	// Para poucas tarefas, usar canais diretos (evitar overhead de ParallelFor)
	if len(tasks) < p.workerPool.workers {
		dones := make([]chan struct{}, len(tasks))
		for i := range dones {
			dones[i] = make(chan struct{})
		}
		for i, task := range tasks {
			done := dones[i]
			task := task
			p.workerPool.ch <- workItem{
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

	// Para muitas tarefas, usar ParallelFor com divisão de trabalho
	p.workerPool.ParallelFor(len(tasks), func(start, end int) {
		for i := start; i < end; i++ {
			tasks[i]()
		}
	})
}

// Close fecha o pool de workers.
func (p *TaskPool) Close() {
	p.workerPool.Close()
}

// Workers retorna o número de workers no pool.
func (p *TaskPool) Workers() int {
	return p.workerPool.Workers()
}
