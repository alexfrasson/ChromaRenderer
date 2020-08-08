#pragma once

#include "chroma-renderer/core/types/BoundingBox.h"

#include <cmath>
#include <glm/vec3.hpp>
#include <vector>

class Camera
{

  public:
    // Image image;
    int width;
    int height;
    glm::vec3 eye;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 forward;

  private:
    float m_HorizontalFOV;

  public:
    float aspectRatio;
    float d;
    float apperture = 16.0f;
    float shutterTime = 2.0f;
    float iso = 100.0f;

  public:
    Camera();
    ~Camera();

    float horizontalFOV()
    {
        return m_HorizontalFOV;
    }
    void horizontalFOV(float hfov)
    {
        m_HorizontalFOV = hfov;
        d = ((float)width / 2.0f) / tanf(m_HorizontalFOV / 2.0f);
    }

    void setSize(int w, int h);
    void lookAt(glm::vec3 target);

    void fit(const BoundingBox& bb);
    float fov();
};
