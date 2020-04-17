#pragma once

#include <IRenderer.h>

class PathTracing
{
  public:
    // Progress
    bool running = false;
    float invPixelCount;
    int pixelCount;
    std::atomic<int> donePixelCount;
    // Progress

    uint32_t maxDepth;
    uint32_t targetSamplesPerPixel;
    bool enviromentLight;
    void init()
    {
        donePixelCount = 0;
    }
    PathTracing();
    void trace(Scene& scene, Image& img, Interval interval, bool& abort);
    Color tracePath(Ray& r, Scene& scene, uint32_t depth);
    float calcColor(Intersection& is);
    float getProgress();
    bool isRunning();
    void setSettings(RendererSettings& settings);
};