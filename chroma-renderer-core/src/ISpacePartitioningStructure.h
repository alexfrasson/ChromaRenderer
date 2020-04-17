#pragma once

#include <IIntersectable.h>
#include <Mesh.h>
#include <Object.h>
#include <Ray.h>
#include <glm/glm.hpp>
#include <vector>

struct Intersection;

class ISpacePartitioningStructure : public IIntersectable
{
  public:
    // virtual bool build(std::vector<Object>& objects) = 0;
    virtual bool build(std::vector<Mesh*>& meshes) = 0;
    // virtual bool intersect(Ray& r, Intersection& intersection) const = 0;
    virtual size_t sizeInBytes(void) = 0;
    // virtual void abort(void) = 0;
    virtual ~ISpacePartitioningStructure(){};
};