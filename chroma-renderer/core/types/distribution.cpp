#include "chroma-renderer/core/types/distribution.h"

#include <cassert>
#include <stdexcept>

void normalizeCdf(std::vector<float>& cdf, const float value)
{
    for (auto& cdf_value : cdf)
    {
        cdf_value /= value;
    }
}

std::vector<float> computeCdf(const std::vector<double>& pdf)
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

    cdf_ = computeCdf(pdf);

    assert(cdf_.size() == pdf.size() + 1);

    normalizeCdf(cdf_, cdf_.back());
}

size_t Distribution::sample(float value)
{
    for (std::size_t i = 0; i < cdf_.size(); i++)
    {
        if (cdf_[i] > value)
        {
            return i - 1;
        }
    }

    throw std::runtime_error("Value not defined for the current CDF.");
}