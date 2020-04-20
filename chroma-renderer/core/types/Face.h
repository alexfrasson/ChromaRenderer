#pragma once

#include <glm/vec3.hpp>

class Face
{
  public:
    glm::vec3 v[3];
    glm::vec3 n[3];
    glm::vec3 tn;
    glm::vec3 edgesMollerTrumbore[2];
};