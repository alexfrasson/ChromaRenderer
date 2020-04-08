#pragma once

#include <chrono>
#include <glad/glad.h>

class Stopwatch
{
public:

	std::chrono::milliseconds elapsedMillis;

	Stopwatch();

	//Starts the stopwatch.
	void start();
	//Sum the elapsed time and stops.
	void stop();
	//Set the elapsed time to zero and start counting immediately.
	void restart();

private:

	std::chrono::steady_clock::time_point begin;
};



class StopwatchOpenGL
{
public:
	StopwatchOpenGL();
	~StopwatchOpenGL();
	void start();
	void stop();
	double elapsedMillis();

private:
	GLuint queries[2];
	double _elapsedMillis = 0.0;
};