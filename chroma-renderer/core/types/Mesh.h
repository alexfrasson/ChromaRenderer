#pragma once

#include <glm/glm.hpp>
#include <vector>
//#include "chroma-renderer/core/ChromaRenderer/IIntersectable.h"
#include "chroma-renderer/core/types/BoundingBox.h"
#include "chroma-renderer/core/types/Intersection.h"
#include "chroma-renderer/core/types/Material.h"
#include "chroma-renderer/core/types/Ray.h"
#include "chroma-renderer/core/types/Triangle.h"

class Mesh
//: public IIntersectable
{
  public:
    BoundingBox boundingBox;

    std::vector<Triangle> t;
    std::vector<glm::vec3> v;
    std::vector<glm::vec3> n;

    std::vector<Material> materials;

    Mesh();
    void genBoundingBox();

    size_t sizeInBytes();

    bool intersect(Ray& r, Intersection& intersection) const
    {
        bool hit = false;
        // Para cada triangulo
        const size_t size = t.size();
        for (size_t i = 0; i < size; i++)
        {
            if (t[i].intersect(r, intersection))
                hit = true;
        }
        return hit;
    }

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