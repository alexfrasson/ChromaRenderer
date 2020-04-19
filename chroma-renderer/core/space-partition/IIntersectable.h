#pragma once

#include "chroma-renderer/core/types/Intersection.h"
#include "chroma-renderer/core/types/Ray.h"

class IIntersectable
{
  public:
    virtual bool intersect(Ray& r, Intersection& intersection) const = 0;
    virtual ~IIntersectable(){};
};
