#pragma once

#include "chroma-renderer/core/types/bounding_box.h"
#include "chroma-renderer/core/types/material.h"
#include "chroma-renderer/core/types/triangle.h"

#include <glm/vec3.hpp>

#include <vector>

class Mesh
{
  public:
    BoundingBox bounding_box{};
    std::vector<Triangle> t;
    std::vector<glm::vec3> v;
    std::vector<glm::vec3> n;
    std::vector<Material> materials;

    void genBoundingBox();

    size_t sizeInBytes() const;

    void genSmoothNormals();

    inline glm::vec3* getVertex(uint32_t faceindex, uint32_t vertexindex)
    {
        return &v[t[faceindex].v[vertexindex]];
    }
    inline glm::vec3* getNormal(uint32_t faceindex, uint32_t vertexindex)
    {
        return &n[t[faceindex].n[vertexindex]];
    }
};