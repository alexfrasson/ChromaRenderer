#pragma once

#include "chroma-renderer/core/types/mesh.h"

#include <memory>
#include <vector>

class ISpacePartitioningStructure
{
  public:
    virtual bool build(std::vector<std::unique_ptr<Mesh>>& meshes) = 0;
    virtual size_t sizeInBytes() = 0;
    virtual ~ISpacePartitioningStructure() = default;

  protected:
    ISpacePartitioningStructure() = default;
    ISpacePartitioningStructure(const ISpacePartitioningStructure&) = default;
    ISpacePartitioningStructure(ISpacePartitioningStructure&&) = default;
    ISpacePartitioningStructure& operator=(const ISpacePartitioningStructure&) = default;
    ISpacePartitioningStructure& operator=(ISpacePartitioningStructure&&) = default;
};