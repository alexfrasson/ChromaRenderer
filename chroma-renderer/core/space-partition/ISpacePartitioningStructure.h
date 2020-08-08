#pragma once

#include "chroma-renderer/core/types/Mesh.h"

#include <vector>

class ISpacePartitioningStructure
{
  public:
    virtual bool build(std::vector<Mesh*>& meshes) = 0;
    virtual size_t sizeInBytes(void) = 0;
    virtual ~ISpacePartitioningStructure() = default;
};