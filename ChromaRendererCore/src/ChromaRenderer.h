#pragma once

#include <thread>
#include <string>
#include <functional>
#include <atomic>

#include <Scene.h>
#include <Image.h>
#include <RendererSettings.h>
#include <ThreadPool.h>
#include <KDTree.h>
#include <Mesh.h>
#include <RayCasting.h>
#include <PathTracing.h>
#include <CudaPathTracer.h>
#include <StopWatch.h>


class ChromaRenderer
{
public:

	enum State
	{
		RENDERING,
		LOADINGSCENE,
		PROCESSINGSCENE,
		IDLE
	};

	enum RendererType
	{
		RAYCAST,
		PATHTRACE,
		CUDAPATHTRACE
	};

//private:
	//bool running;
	RendererType rendererType = RendererType::CUDAPATHTRACE;
	RendererSettings settings;
	Scene scene;
	State state;
	RayCasting renderer;
	PathTracing pathtracing;
	CudaPathTracer cudaPathTracer;
	
	Stopwatch stopwatch;

	// Progress
	bool running = false;
	float invPixelCount;
	int pixelCount;
	//std::atomic<int> donePixelCount;
	// Progress
	void genTasks();
	void start();

public:
	Image image;
	ThreadPool threadPool;
	ChromaRenderer();
	~ChromaRenderer();

	void importScene(std::string filename);
	void importScene(std::string filename, std::function<void()> onLoad);
	void importEnviromentMap(std::string filename);
	void startRender();
	void startRender(RendererSettings& settings);
	void setSettings(RendererSettings& settings);
	RendererSettings getSettings();
	void stopRender(bool block = false);
	void saveLog();
	void setSize(unsigned int width, unsigned int height);
	bool isRunning();
	float getProgress();
	void clearScene();
	bool isIdle()
	{
		if (state == State::RENDERING && !isRunning())
			state = State::IDLE;
		return (state == State::IDLE);
	}
	State getState()
	{
		if (state == ChromaRenderer::RENDERING)
		{
			if (!isRunning())
				state = ChromaRenderer::IDLE;
		}
		/*if (state == ChromaRenderer::PROCESSINGSCENE)
		{
			if (scene.ready)
				state = ChromaRenderer::IDLE;
		}*/
		return state;
	}
	std::string getStateStr()
	{
		std::string s;
		switch (getState())
		{
		case ChromaRenderer::RENDERING:
			s = "Rendering";
			break;
		case ChromaRenderer::LOADINGSCENE:
			s = "Loading Scene";
			break;
		case ChromaRenderer::PROCESSINGSCENE:
			s = "Processing Scene";
			break;
		case ChromaRenderer::IDLE:
			s = "Idle";
			break;
		default:
			s = "";
			break;
		}
		return s;
	}

	void update();

private:
	void cbSceneLoaded(Object o);
	void cbSceneLoadedm(Mesh *m);
	void cbSceneLoadedScene(std::function<void()> onLoad);
	void cbSceneProcessed();
	void printStatistics();
};