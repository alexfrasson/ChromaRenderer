#pragma once

#include <Image.h>
#include <RendererSettings.h>
#include <Scene.h>

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
