#pragma once

#include "chroma-renderer/core/types/BoundingBox.h"
#include "chroma-renderer/core/types/Face.h"

#include <vector>

class Object
{
  public:
    BoundingBox boundingBox;

    std::vector<Face> f;

    Object();

    size_t sizeInBytes();

    void genBoundingBox();
};