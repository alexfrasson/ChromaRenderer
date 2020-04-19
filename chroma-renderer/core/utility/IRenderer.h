#pragma once

#include "chroma-renderer/core/types/Image.h"
#include "chroma-renderer/core/types/RendererSettings.h"
#include "chroma-renderer/core/scene/Scene.h"

#include <atomic>

struct Interval
{
    int fromWidth, fromHeight;
    int toWidth, toHeight;
};

class IRenderer
{
  public:
    virtual void trace(Scene& scene, Image& img, RendererSettings& settings, Interval interval, bool& abort) = 0;
    virtual ~IRenderer(){};
};
