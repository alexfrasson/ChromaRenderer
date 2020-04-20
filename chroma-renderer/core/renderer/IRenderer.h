#pragma once

#include "chroma-renderer/core/scene/Scene.h"
#include "chroma-renderer/core/space-partition/ISpacePartitioningStructure.h"
#include "chroma-renderer/core/types/Image.h"
#include "chroma-renderer/core/renderer/RendererSettings.h"

struct Interval
{
    int fromWidth, fromHeight;
    int toWidth, toHeight;
};

class IRenderer
{
  public:
    virtual void trace(ISpacePartitioningStructure* sps,
                       Scene& scene,
                       Image& img,
                       RendererSettings& settings,
                       Interval interval,
                       bool& abort) = 0;
    virtual ~IRenderer(){};
};
