#pragma once

#include "chroma-renderer/core/types/Material.h"

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include <vector>

class Triangle
{
  public:
    std::uint32_t v[3]{0, 0, 0};
    std::uint32_t n[3]{0, 0, 0};

    std::vector<glm::vec3>* vdata{nullptr};
    std::vector<glm::vec3>* ndata{nullptr};

    Material* material{nullptr};

    inline glm::vec3* getVertex(std::size_t i) const
    {
        return &(*vdata)[v[i]];
    }

    inline glm::vec3* getNormal(std::size_t i) const
    {
        return &(*ndata)[n[i]];
    }
};