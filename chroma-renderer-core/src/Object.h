#pragma once

#include <BoundingBox.h>
#include <glm/vec3.hpp>
#include <vector>

class Face
{
  public:
    glm::vec3 v[3];
    glm::vec3 n[3];
    glm::vec3 tn;
    glm::vec3 edgesMollerTrumbore[2];
};

class Object
{
  public:
    BoundingBox boundingBox;

    std::vector<Face> f;
    // std::vector<glm::vec3> v;
    // std::vector<glm::vec3> n;

    Object();

    size_t sizeInBytes();

    void genBoundingBox();
};