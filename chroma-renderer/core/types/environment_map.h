#pragma once

#include "chroma-renderer/core/types/distribution.h"

#include <cstdint>

class EnvironmentMap
{
  public:
    EnvironmentMap(const float* data, const uint32_t width, const uint32_t height);
    ~EnvironmentMap();

    const float* GetData()
    {
        return data_;
    }

    const Distribution& GetDistribution() const
    {
        return distribution_;
    }

    const std::vector<float>& GetPdf() const
    {
        return pdf_;
    }

  private:
    float* data_;
    std::vector<float> pdf_;
    uint32_t width_;
    uint32_t height_;
    Distribution distribution_;
};