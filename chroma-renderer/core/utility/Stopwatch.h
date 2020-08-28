#pragma once

#include <glad/glad.h>

#include <chrono>

class Stopwatch
{
  public:
    std::chrono::milliseconds elapsedMillis{0};

    // Starts the stopwatch.
    void start();
    // Sum the elapsed time and stops.
    void stop();
    // Set the elapsed time to zero and start counting immediately.
    void restart();

  private:
    std::chrono::steady_clock::time_point begin;
};

class StopwatchOpenGL
{
  public:
    StopwatchOpenGL();
    ~StopwatchOpenGL();
    StopwatchOpenGL(const StopwatchOpenGL&) = default;
    StopwatchOpenGL(StopwatchOpenGL&&) = default;
    StopwatchOpenGL& operator=(const StopwatchOpenGL&) = default;
    StopwatchOpenGL& operator=(StopwatchOpenGL&&) = default;

    void start();
    void stop();
    double elapsedMillis();

  private:
    GLuint queries[2]{0, 0};
    double _elapsedMillis{0.0};
};