#pragma once

#include <vector>

class Distribution
{
  public:
    Distribution() = default;
    explicit Distribution(const std::vector<double>& pdf);

    size_t sample(float value);

    const std::vector<float>& getCdf() const
    {
        return cdf;
    }

  private:
    std::vector<float> cdf;
};