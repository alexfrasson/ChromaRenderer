#include "Stopwatch.h"

#include <iostream>

Stopwatch::Stopwatch()
{
    elapsedMillis = std::chrono::milliseconds(0);
}

void Stopwatch::start()
{
    elapsedMillis = std::chrono::milliseconds(0);
    begin = std::chrono::steady_clock::now();
}

void Stopwatch::stop()
{
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    elapsedMillis = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin); // / 1000000000.0;
}

void Stopwatch::restart()
{
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    elapsedMillis = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin); // / 1000000000.0;
    begin = end;
}

StopwatchOpenGL::StopwatchOpenGL()
{
    glGenQueries(2, queries);
}
StopwatchOpenGL::~StopwatchOpenGL()
{
    glDeleteQueries(2, queries);
}

void StopwatchOpenGL::start()
{
    _elapsedMillis = -1.0;
    glQueryCounter(queries[0], GL_TIMESTAMP);
}

void StopwatchOpenGL::stop()
{
    glQueryCounter(queries[1], GL_TIMESTAMP);
}

double StopwatchOpenGL::elapsedMillis()
{
    if (_elapsedMillis < 0)
    {
        GLint available = 0;
        // Wait for all results to become available
        while (!available)
            glGetQueryObjectiv(queries[1], GL_QUERY_RESULT_AVAILABLE, &available);

        GLuint64 timeStamp0 = 0, timeStamp1 = 0;
        glGetQueryObjectui64v(queries[0], GL_QUERY_RESULT, &timeStamp0);
        glGetQueryObjectui64v(queries[1], GL_QUERY_RESULT, &timeStamp1);
        // OpenGL returns time in nanoseconds.
        _elapsedMillis = (double)(timeStamp1 - timeStamp0) / 1000000.0;
    }
    return _elapsedMillis;
}