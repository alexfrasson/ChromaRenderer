#pragma once

#include "chroma-renderer/core/types/distribution.h"

#include <cstdint>

class EnvironmentMap
{
  public:
    EnvironmentMap(const float* data, uint32_t width, uint32_t height);
    EnvironmentMap(const EnvironmentMap&) = delete;
    EnvironmentMap(const EnvironmentMap&&) = delete;
    EnvironmentMap operator=(const EnvironmentMap&) = delete;
    EnvironmentMap operator=(const EnvironmentMap&&) = delete;
    ~EnvironmentMap() = default;

    const float* getData()
    {
        return data_.data();
    }

    const Distribution& getDistribution() const
    {
        return distribution_;
    }

    const std::vector<float>& getPdf() const
    {
        return pdf_;
    }

  private:
    std::vector<float> data_;
    std::vector<float> pdf_;
    uint32_t width_;
    uint32_t height_;
    Distribution distribution_;
};