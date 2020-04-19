#pragma once

#include "chroma-renderer/core/types/Color.h"
#include "chroma-renderer/core/utility/IRenderer.h"

class RayCasting : public IRenderer
{

  public:
    // Progress
    bool running = false;
    float invPixelCount;
    int pixelCount;
    std::atomic<int> donePixelCount;
    // Progress

    RayCasting();
    void init()
    {
        donePixelCount = 0;
    }
    void trace(Scene& scene, Image& img, RendererSettings& settings, Interval interval, bool& abort);
    Color calcColor(Intersection& is);
    float getProgress();
    bool isRunning();
};