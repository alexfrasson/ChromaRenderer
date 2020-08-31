#pragma once

#include <glm/vec3.hpp>

#include <algorithm>
#include <limits>

struct BoundingBox
{
    BoundingBox() = default;
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
        min.x = std::min(min.x, p.x);
        min.y = std::min(min.y, p.y);
        min.z = std::min(min.z, p.z);

        max.x = std::max(max.x, p.x);
        max.y = std::max(max.y, p.y);
        max.z = std::max(max.z, p.z);
    }

    void expand(const BoundingBox& bb)
    {
        expand(bb.max);
        expand(bb.min);
    }

    bool contains(const glm::vec3& p) const
    {
        if (p.x < min.x || p.y < min.y || p.z < min.z)
        {
            return false;
        }
        if (p.x > max.x || p.y > max.y || p.z > max.z)
        {
            return false;
        }
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

    glm::vec3& operator[](const int& i)
    {
        if (i == 0)
        {
            return min;
        }
        return max;
    }

    const glm::vec3& operator[](const int& i) const
    {
        if (i == 0)
        {
            return min;
        }
        return max;
    }

    glm::vec3 max{std::numeric_limits<float>::lowest()};
    glm::vec3 min{std::numeric_limits<float>::max()};
};
