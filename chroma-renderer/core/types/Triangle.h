#pragma once

#include "chroma-renderer/core/types/Material.h"

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include <vector>

#define EPSILON 0.000001f

class Triangle
{
  public:
    uint32_t v[3]; // Vertices' indices
    uint32_t n[3]; // Normals' indices

    std::vector<glm::vec3>* vdata; // Pointer to vertex data
    std::vector<glm::vec3>* ndata; // Pointer to normal data

    Material* material;

    Triangle() : vdata(NULL), ndata(NULL)
    {
    }

    ~Triangle()
    {
    }

    inline glm::vec3* getVertex(size_t i) const
    {
        return &(*vdata)[v[i]];
    }

    inline glm::vec3* getNormal(size_t i) const
    {
        return &(*ndata)[n[i]];
    }
};