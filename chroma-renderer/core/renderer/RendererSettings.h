#pragma once

#include <glm/vec3.hpp>

struct RendererSettings
{
  public:
    int width;
    int height;

    float horizontalFOV;

    int samplesperpixel;
    
    RendererSettings();

    bool operator==(const RendererSettings& rs) const;
    bool operator!=(const RendererSettings& rs) const;
};