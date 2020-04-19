#include "chroma-renderer/core/renderer/Benchmark.h"

#include <iostream>

Benchmark::Benchmark() : stop(false), running(false)
{
}
void Benchmark::stopBenchmark()
{
    stop = true;
    if (runThread.joinable())
        runThread.join();
    running = false;
}
void Benchmark::reset()
{
    min = std::numeric_limits<double>::max();
    max = std::numeric_limits<double>::min();
    average = 0.f;
    sumFinalizedTasks = 0.f;
    nFinalizedTasks = 0;
}

bool Benchmark::isRunning()
{
    return running;
}

double Benchmark::getAverage()
{
    return average;
}
double Benchmark::getMax()
{
    return max;
}
double Benchmark::getMin()
{
    return min;
}

void Benchmark::exportResults()
{
}

void Benchmark::printResults()
{
    std::cout << "Max:     " << getMax() << "s" << std::endl;
    std::cout << "Average: " << getAverage() << "s" << std::endl;
    std::cout << "Min:     " << getMin() << "s" << std::endl;
}
