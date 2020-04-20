#pragma once

#include "chroma-renderer/core/types/BoundingBox.h"
#include "chroma-renderer/core/types/Face.h"

#include <glm/vec3.hpp>

#include <vector>

class Object
{
  public:
    BoundingBox boundingBox;

    std::vector<Face> f;
    // std::vector<glm::vec3> v;
    // std::vector<glm::vec3> n;

    Object();

    size_t sizeInBytes();

    void genBoundingBox();
};