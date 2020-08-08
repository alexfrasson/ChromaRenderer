#pragma once

#include "chroma-renderer/core/space-partition/IIntersectable.h"
#include "chroma-renderer/core/types/Mesh.h"
#include "chroma-renderer/core/types/Ray.h"

#include <vector>

struct Intersection;

class ISpacePartitioningStructure : public IIntersectable
{
  public:
    virtual bool build(std::vector<Mesh*>& meshes) = 0;
    // virtual bool intersect(Ray& r, Intersection& intersection) const = 0;
    virtual size_t sizeInBytes(void) = 0;
    // virtual void abort(void) = 0;
    virtual ~ISpacePartitioningStructure(){};
};