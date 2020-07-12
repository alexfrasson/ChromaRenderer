#include "chroma-renderer/core/types/distribution.h"

#include <cassert>
#include <stdexcept>

void NormalizeCdf(std::vector<float>& cdf, const float value)
{
    for (auto& cdf_value : cdf)
    {
        cdf_value /= value;
    }
}

std::vector<float> ComputeCdf(const std::vector<double>& pdf)
{
    std::vector<float> cdf;
    cdf.reserve(pdf.size() + 1);
    cdf.push_back(0.0);
    double sum = 0.0;

    for (const auto& value : pdf)
    {
        cdf.push_back(static_cast<float>(sum + value));
        sum += value;
    }
    return cdf;
}

Distribution::Distribution(const std::vector<double>& pdf)
{
    assert(pdf.size() > 0);

    cdf = ComputeCdf(pdf);

    assert(cdf.size() == pdf.size() + 1);

    NormalizeCdf(cdf, cdf.back());
}

size_t Distribution::Sample(float value)
{
    for (size_t i = 0; i < cdf.size(); i++)
    {
        if (cdf[i] > value)
        {
            return i - 1;
        }
    }

    throw std::runtime_error("Value not defined for the current CDF.");
}