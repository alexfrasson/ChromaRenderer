#pragma once

#include <Stopwatch.h>
#include <thread>

class Benchmark
{
  public:
    Benchmark();
    template <typename _Functor> void startBenchmark(const unsigned int n, _Functor f)
    {
        runThread = std::thread([this, f, n]() {
            run(n, f);
            return 0;
        });
        // auto bound = std::bind(&Benchmark::run<_Functor>, this, std::placeholders::_1, std::placeholders::_2);
        // runThread = std::thread(bound, n, f);
    }
    template <typename _Functor, typename _Cmp> void startBenchmark(const unsigned int n, _Functor f, _Cmp isrunning)
    {
        runThread = std::thread([this, isrunning, f, n]() {
            run(n, f, isrunning);
            return 0;
        });
        // auto bound = std::bind(&Benchmark::run<_Functor>, this, std::placeholders::_1, std::placeholders::_2);
        // runThread = std::thread(bound, n, f);
    }
    void stopBenchmark();
    bool isRunning();
    void exportResults();
    void printResults();

    double getMax();
    double getMin();
    double getAverage();

  private:
    template <typename _Functor> void run(const unsigned int& n, _Functor f)
    {
        stop = false;
        running = true;
        reset();
        Stopwatch stopwatch;

        // Run f nTests times
        for (unsigned int i = 0; i < n && !stop; i++)
        {
            stopwatch.restart();
            f();
            stopwatch.stop();
            // Sum for the average
            sumFinalizedTasks += stopwatch.elapsedWallTime;
            // Slower execution
            if (stopwatch.elapsedWallTime > max)
                max = stopwatch.elapsedWallTime;
            // Faster execution
            if (stopwatch.elapsedWallTime < min)
                min = stopwatch.elapsedWallTime;
            nFinalizedTasks++;
            std::cout << stopwatch.elapsedWallTime << std::endl;
        }
        // Average execution time
        average = sumFinalizedTasks / nFinalizedTasks;

        printResults();

        running = false;
    }
    template <typename _Functor, typename _Cmp> void run(const unsigned int& n, _Functor f, _Cmp isrunning)
    {
        stop = false;
        running = true;
        reset();
        Stopwatch stopwatch;

        // Run f nTests times
        for (unsigned int i = 0; i < n && !stop; i++)
        {
            stopwatch.restart();
            f();
            while (isrunning())
                ;
            stopwatch.stop();
            // Sum for the average
            sumFinalizedTasks += stopwatch.elapsedWallTime;
            // Slower execution
            if (stopwatch.elapsedWallTime > max)
                max = stopwatch.elapsedWallTime;
            // Faster execution
            if (stopwatch.elapsedWallTime < min)
                min = stopwatch.elapsedWallTime;
            nFinalizedTasks++;
            std::cout << stopwatch.elapsedWallTime << std::endl;
        }

        // Remove both faster and slower executions from average
        sumFinalizedTasks -= max;
        sumFinalizedTasks -= min;

        // Average execution time
        average = sumFinalizedTasks / (double)(nFinalizedTasks - 2);

        printResults();

        running = false;
    }

    std::thread runThread;

    bool stop;
    bool running;
    double max;
    double min;
    double average;

    double sumFinalizedTasks;
    unsigned int nFinalizedTasks;
    void reset();
};
