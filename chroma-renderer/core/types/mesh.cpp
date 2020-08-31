#include "chroma-renderer/core/types/mesh.h"

size_t Mesh::sizeInBytes() const
{
    size_t size = sizeof(Mesh);
    size += sizeof(Triangle) * t.size();
    size += sizeof(glm::vec3) * v.size();
    size += sizeof(glm::vec3) * n.size();
    return size;
}

void Mesh::genBoundingBox()
{
    size_t size = t.size();

    // find boundaries
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            if (v[t[i].v[j]].x > bounding_box.max.x)
            {
                bounding_box.max.x = v[t[i].v[j]].x;
            }
            if (v[t[i].v[j]].y > bounding_box.max.y)
            {
                bounding_box.max.y = v[t[i].v[j]].y;
            }
            if (v[t[i].v[j]].z > bounding_box.max.z)
            {
                bounding_box.max.z = v[t[i].v[j]].z;
            }
            if (v[t[i].v[j]].x < bounding_box.min.x)
            {
                bounding_box.min.x = v[t[i].v[j]].x;
            }
            if (v[t[i].v[j]].y < bounding_box.min.y)
            {
                bounding_box.min.y = v[t[i].v[j]].y;
            }
            if (v[t[i].v[j]].z < bounding_box.min.z)
            {
                bounding_box.min.z = v[t[i].v[j]].z;
            }
        }
    }
}

void Mesh::genSmoothNormals()
{
    n.clear();
    n.reserve(v.size());

    n.insert(n.begin(), v.size(), glm::vec3());

    // Go through all triangles
    for (size_t i = 0; i < t.size(); i++)
    {
        // Calc triangle normal
        glm::vec3 edge0 = *t[i].getVertex(1) - *t[i].getVertex(0);
        glm::vec3 edge1 = *t[i].getVertex(2) - *t[i].getVertex(0);

        // glm::vec3 tnormal = glm::cross(edge0, edge1);
        edge0 = glm::cross(edge0, edge1);

        // Add it to all 3 vertices' normals
        for (int j = 0; j < 3; j++)
        {
            n[t[i].v[j]] += edge0;
            // n[t[i].v[j]] += tnormal;
            // Update indices
            t[i].n[j] = t[i].v[j];
        }
    }

    // Normalize normals
    for (size_t i = 0; i < n.size(); i++)
    {
        n[i] = glm::normalize(n[i]);
    }
}