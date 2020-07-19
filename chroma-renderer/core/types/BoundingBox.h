#pragma once

#include <glm/vec3.hpp>

#include <limits>

struct BoundingBox
{
    BoundingBox()
    {
        max.x = -std::numeric_limits<float>::infinity();
        max.y = max.x;
        max.z = max.x;

        min.x = std::numeric_limits<float>::infinity();
        min.y = min.x;
        min.z = min.x;
    }
    BoundingBox(const glm::vec3& a_min, const glm::vec3& a_max)
    {
        min = a_min;
        max = a_max;
    }
    float surfaceArea() const
    {
        float x = max.x - min.x;
        float y = max.y - min.y;
        float z = max.z - min.z;
        return (2 * x * y + 2 * x * z + 2 * y * z);
    }
    float volume() const
    {
        float x = max.x - min.x;
        float y = max.y - min.y;
        float z = max.z - min.z;
        return (x * y * z);
    }
    void expand(const glm::vec3& p)
    {
        if (p.x < min.x)
            min.x = p.x;
        if (p.y < min.y)
            min.y = p.y;
        if (p.z < min.z)
            min.z = p.z;
        if (p.x > max.x)
            max.x = p.x;
        if (p.y > max.y)
            max.y = p.y;
        if (p.z > max.z)
            max.z = p.z;
    }
    void expand(const BoundingBox& bb)
    {
        expand(bb.max);
        expand(bb.min);
    }

    bool contains(const glm::vec3& p) const
    {
        if (p.x < min.x || p.y < min.y || p.z < min.z)
            return false;
        if (p.x > max.x || p.y > max.y || p.z > max.z)
            return false;
        return true;
    }
    glm::vec3 centroid() const
    {
        return ((max + min) * 0.5f);
    }

    glm::vec3 getCenter() const
    {
        return (glm::vec3((max + min) * 0.5f));
    }
    glm::vec3 max, min;

    glm::vec3& operator[](const int& i)
    {
        if (i == 0)
            return min;
        return max;
    }

    const glm::vec3& operator[](const int& i) const
    {
        if (i == 0)
            return min;
        return max;
    }
};
