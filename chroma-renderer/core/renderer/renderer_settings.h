#pragma once

#include <cstdint>

struct RendererSettings
{
    bool operator==(const RendererSettings& rs) const;
    bool operator!=(const RendererSettings& rs) const;

    int width{640};
    int height{480};
    float horizontal_fov{1.0};
    std::uint32_t samplesperpixel{10000};
};