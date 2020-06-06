#pragma once

#include <vector>

class Distribution
{
  public:
    Distribution() = default;
    explicit Distribution(const std::vector<double>& function);

    size_t Sample(float value);

    const std::vector<float>& GetCdf() const
    {
        return cdf;
    }

  private:
    std::vector<float> cdf;
};