#pragma once

#include <glm/vec3.hpp>

struct RendererSettings
{
  public:
    int width;
    int height;

    float horizontalFOV;

    unsigned int nthreads;

    bool supersampling;
    int samplesperpixel;
    int maxPathDepth;

    bool boundingboxtest;

    bool shadowray;

    RendererSettings();

    bool operator==(const RendererSettings& rs) const;
    bool operator!=(const RendererSettings& rs) const;
};