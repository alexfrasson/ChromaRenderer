#include <ThreadPool.h>
#include <iostream>


// Reserves memory for 4 times the number of processors available of workers 
// and creates number of processors available workers.
// If the number of processors could not be determined, 4 workers will be created.
ThreadPool::ThreadPool() 
	: quit(false)
{
	unsigned int n = std::thread::hardware_concurrency();
	if (n == 0)
		n = 4;
	capacity = n * 4;
	workers.reserve(capacity);
	for (size_t i = 0; i < n; ++i)
		addWorker();
}
// Reserves memory for 'maxNumberWorkers' workers and creates one worker.
// If 'maxNumberWorkers' is 0, the capacity is set to the number of processors.
ThreadPool::ThreadPool(unsigned int maxNumberWorkers)
	: quit(false)
{
	capacity = maxNumberWorkers;
	if (capacity < 1)
	{
		capacity = std::thread::hardware_concurrency();
		if (capacity == 0)
			capacity = 4;
	}
	workers.reserve(capacity);
	addWorker();
}
// Reserves memory for 'maxNumberWorkers' and creates 'nworkers' workers.
// If 'maxNumberWorkers' is 0, the capacity is set to the number of processors.
ThreadPool::ThreadPool(unsigned int maxNumberWorkers, size_t nWorkers)
	: quit(false)
{
	capacity = maxNumberWorkers;
	if (capacity < 1)
	{
		capacity = std::thread::hardware_concurrency();
		if (capacity == 0)
			capacity = 4;
	}
	workers.reserve(capacity);
	for (size_t i = 0; i < nWorkers; ++i)
		addWorker();
}
ThreadPool::~ThreadPool()
{
	abort();
}
void ThreadPool::setNumberWorkers(unsigned int nWorkers)
{
	if (nWorkers == workers.size())
		return;
	// The pool should have at least one worker
	// and no more than capacity() workers.
	if (nWorkers < 1)
		nWorkers = 1;
	else if (nWorkers > capacity)
		nWorkers = capacity;
	// We need to remove some workers.
	while (nWorkers < workers.size())
		removeWorker();
	// We need to add some workers.
	while (nWorkers > workers.size())
		addWorker();
}
// Add one worker to the pool. If the pool is full, no worker is added.
void ThreadPool::addWorker()
{
	if (capacity == workers.size())
		return;
	int i = workers.size();
	workers.emplace_back(
		std::tuple<bool, bool, std::thread>(false, false, std::thread(
		[this, i]
	{
		std::cout << "Worker " << i << " starting." << std::endl;
		for (;;)
		{
			std::function<void(bool&)> task;
			{
				std::unique_lock<std::mutex> lock(this->queue_mutex);
				this->condition.wait(lock,
					[this, i]{ return this->quit || QUIT(this->workers[i]) || !this->tasks.empty(); });
				if (this->quit || QUIT(workers[i]))
				{
					std::cout << "Worker " << i << " quiting." << std::endl;
					return;
				}

				task = std::move(this->tasks.front());
				this->tasks.pop();
			}
			STOP(this->workers[i]) = false;
			task(STOP(this->workers[i]));
			STOP(this->workers[i]) = false;
		}
	}
	)));
}
// Request one of the workers to quit and remove it from the pool.
// If the number of workers is one, no worker is removed.
void ThreadPool::removeWorker()
{
	if (workers.size() <= 1)
		return;
	QUIT(workers.back()) = true;
	STOP(workers.back()) = true;
	condition.notify_all();
	if (THREAD(workers.back()).joinable())
		THREAD(workers.back()).join();
	workers.pop_back();
}
// Clear the tasks queue.
void ThreadPool::clearTaskQueue()
{
	{
		std::unique_lock<std::mutex> lock(queue_mutex);
		while (!tasks.empty())
			tasks.pop();
		stopAll();
	}
}
// Requests and waits for every worker to quit.
void ThreadPool::abort()
{
	{
		std::unique_lock<std::mutex> lock(queue_mutex);
		quit = true;
		for (int i = 0; i < workers.size(); i++)
		{
			QUIT(workers[i]) = true;
			STOP(workers[i]) = true;
		}
	}
	condition.notify_all();
	for (int i = 0; i < workers.size(); i++)
		if (THREAD(workers[i]).joinable())
			THREAD(workers[i]).join();
}
void ThreadPool::stopAll()
{
	for (int i = 0; i < workers.size(); i++)
		STOP(workers[i]) = true;
}
unsigned int ThreadPool::getCapacity()
{
	return capacity;
}
unsigned int ThreadPool::getNumberWorkers()
{
	return workers.size();
}