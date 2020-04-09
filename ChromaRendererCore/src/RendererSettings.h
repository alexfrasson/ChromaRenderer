#pragma once

#include <glm/glm.hpp>

struct RendererSettings
{
  public:
    int width;
    int height;

    float horizontalFOV;

    glm::vec3 enviromentLightColor;
    float enviromentLightIntensity;

    unsigned int nthreads;

    bool supersampling;
    int samplesperpixel;
    int maxPathDepth;

    bool boundingboxtest;

    bool shadowray;

    RendererSettings();

    const bool operator==(const RendererSettings& rs);
    const bool operator!=(const RendererSettings& rs);
};