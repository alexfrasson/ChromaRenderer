#pragma once

#include "chroma-renderer/core/types/BoundingBox.h"

#include <cmath>
#include <glm/vec3.hpp>
#include <vector>

class Camera
{
  public:
    float horizontalFOV() const
    {
        return m_HorizontalFOV;
    }
    void horizontalFOV(float hfov)
    {
        m_HorizontalFOV = hfov;
        d = ((float)width / 2.0f) / tanf(m_HorizontalFOV / 2.0f);
    }

    void setSize(int pwidth, int pheight);
    void lookAt(glm::vec3 target);

    void fit(const BoundingBox& bb);
    float fov() const;

    int width{1280};
    int height{720};
    glm::vec3 eye{0, 0, 0};
    glm::vec3 up{0, 1, 0};
    glm::vec3 right{1, 0, 0};
    glm::vec3 forward{0, 0, 1};
    float aspectRatio{(float)width / static_cast<float>(height)};
    float m_HorizontalFOV{1.0f};
    float d{((float)width / 2.0f) / tanf(m_HorizontalFOV / 2.0f)};
    float apperture{16.0f};
    float shutterTime{2.0f};
    float iso{100.0f};
};
