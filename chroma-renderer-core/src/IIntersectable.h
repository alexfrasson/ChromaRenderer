#pragma once

#include <Intersection.h>
#include <Ray.h>

class IIntersectable
{
  public:
    virtual bool intersect(Ray& r, Intersection& intersection) const = 0;
    virtual ~IIntersectable(){};
};
