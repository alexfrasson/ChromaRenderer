#pragma once

#include "chroma-renderer/core/renderer/CudaPathTracer.h"
#include "chroma-renderer/core/renderer/PathTracing.h"
#include "chroma-renderer/core/renderer/PostProcessor.h"
#include "chroma-renderer/core/renderer/RayCasting.h"
#include "chroma-renderer/core/renderer/RendererSettings.h"
#include "chroma-renderer/core/scene/Scene.h"
#include "chroma-renderer/core/space-partition/ISpacePartitioningStructure.h"
#include "chroma-renderer/core/types/Image.h"
#include "chroma-renderer/core/types/Mesh.h"
#include "chroma-renderer/core/utility/Stopwatch.h"
#include "chroma-renderer/core/utility/ThreadPool.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>

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

    RendererType rendererType = RendererType::CUDAPATHTRACE;
    RendererSettings settings;
    Scene scene;
    State state;
    RayCasting renderer;
    PathTracing pathtracing;
    CudaPathTracer cudaPathTracer;

    std::unique_ptr<ISpacePartitioningStructure> sps;

    PostProcessor post_processor;
    Stopwatch stopwatch;

    bool running = false;
    float invPixelCount;
    int pixelCount;

    void genTasks();
    void start();

  public:
    Image renderer_target;
    Image final_target;
    ThreadPool threadPool;
    ChromaRenderer();
    ~ChromaRenderer();

    void importScene(std::string filename);
    void importScene(std::string filename, std::function<void()> onLoad);
    void importEnviromentMap(std::string filename);
    void setEnvMap(const float* data, const uint32_t width, const uint32_t height, const uint32_t channels);
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
    void cbSceneLoadedm(Mesh* m);
    void cbSceneLoadedScene(std::function<void()> onLoad);
    void cbSceneProcessed();
    void printStatistics();
};