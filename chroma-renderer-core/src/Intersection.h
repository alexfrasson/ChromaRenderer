#pragma once

#include "Object.h"
#include <Material.h>
#include <glm/glm.hpp>
#include <limits>

struct Intersection
{
    Intersection() : triangle(NULL), distance(std::numeric_limits<float>::infinity())
    {
    }
    const Material* material;
    const Face* triangle;
    float distance;
    glm::vec3 p;
    glm::vec3 n;
};