#pragma once

#include "chroma-renderer/core/types/BoundingBox.h"
#include "chroma-renderer/core/types/Ray.h"

#include <glm/vec3.hpp>

#include <cmath>
#include <vector>

// tan(FOV/2) = (screenSize/2) / screenPlaneDistance
// tan(FOV_H/2) = (screen_width/2) / screenPlaneDistance
// tan(FOV_V / 2) = (screen_height / 2) / screenPlaneDistance
// tan(FOV_H/2) / screen_width = tan(FOV_V/2) / screen_height

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
        d = ((float)width / 2.0f) / tan(m_HorizontalFOV / 2.0f);
    }

    void setSize(int w, int h);
    void lookAt(glm::vec3 target);

    void randomRayDirection(const int i, const int j, Ray& ray) const;
    void rayDirection(const int i, const int j, Ray& ray) const;
    void rayDirection(const int i, const int j, std::vector<Ray>& rays) const;
    void rayDirection(const int i, const int j, std::vector<Ray>& rays, const unsigned int nRays) const;

    void fit(const BoundingBox& bb);
    float fov();
};
