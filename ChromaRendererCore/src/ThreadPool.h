#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#define STOP(x) std::get<0>(x)   // Boolean for stopping the current task
#define QUIT(x) std::get<1>(x)   // Boolean for "killing" the worker
#define THREAD(x) std::get<2>(x) // The thread object

class ThreadPool
{
  private:
    // Queue of Taks.
    // The boolean parameter is optional. The task can use this boolean to periodically check if the it should abort.
    // If the ThreadPool needs to close one (or all) of the workers, it will not have to wait the worker to complete the
    // task. E.g. void task(bool& stop)
    // {
    //		for(int i = 0; i < verybignumber; i++)
    //		{
    //			computationallyintensivethings;
    //			if(stop)
    //				return;
    //		}
    // }
    std::queue<std::function<void(bool&)>> tasks;
    // The first boolean is used to stop the current task. After the current taks is stopped, the worker will set this
    // boolean to false again and search for another task. The second boolean is used to completely close the worker. It
    // will be delete from the worker list after it finishes.
    std::vector<std::tuple<bool, bool, std::thread>> workers; // Workers
    // Mutex for blocking the access to the tasks queue.
    std::mutex queue_mutex;
    std::condition_variable condition;

    // Maximun number of workers.
    unsigned int capacity;

  public:
    // If this variable is true, all workers will quit.
    bool quit;
    // Reserves memory for 4 times the number of processors available of workers
    // and creates number of processors available workers.
    // If the number of processors could not be determined, 4 workers will be created.
    ThreadPool();
    // Reserves memory for 'maxNumberWorkers' workers and creates one worker.
    ThreadPool(unsigned int maxNumberWorkers);
    // Reserves memory for 'maxNumberWorkers' and creates 'nworkers' workers.
    ThreadPool(unsigned int maxNumberWorkers, size_t nworkers);
    ~ThreadPool();
    template <typename T> void enqueue(const T& t)
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            // Don't allow enqueueing after stopping the pool
            if (quit)
                throw std::runtime_error("Enqueue on stopped ThreadPool.");
            tasks.emplace(t);
        }
        condition.notify_one();
    }
    void setNumberWorkers(unsigned int nWorkers);
    // Add one worker to the pool. If the pool is full, no worker is added.
    void addWorker();
    // Request one of the workers to quit and remove it from the pool.
    // If the number of workers is one, no worker is removed.
    void removeWorker();
    // Clear the tasks queue.
    void clearTaskQueue();
    void stopAll();
    // Requests and waits for every worker to quit.
    void abort();
    unsigned int getCapacity();
    unsigned int getNumberWorkers();
};