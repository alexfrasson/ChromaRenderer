#include "chroma-renderer/core/types/environment_map.h"

#define _USE_MATH_DEFINES
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <math.h>

double luma(double r, double g, double b)
{
    return 0.299 * r + 0.587 * g + 0.114 * b;
}

EnvironmentMap::EnvironmentMap(const float* data, const uint32_t width, const uint32_t height)
    : data_{nullptr}, width_{width}, height_{height}
{
    const size_t color_components = 4;
    const size_t pixel_count = width_ * height_;
    data_ = new float[pixel_count * color_components];
    std::memcpy(data_, data, pixel_count * color_components * sizeof(float));

    double sum{0.0};
    double max{std::numeric_limits<double>::min()};
    double min{std::numeric_limits<double>::max()};

    for (size_t i = 0; i < pixel_count; i++)
    {
        const double r = data_[i * color_components];
        const double g = data_[i * color_components + 1];
        const double b = data_[i * color_components + 2];

        double l = luma(r, g, b);
        sum += l;
        max = std::max(max, l);
        min = std::min(min, l);
    }

    std::vector<double> pdf;
    pdf.reserve(pixel_count);
    pdf_.reserve(pixel_count);

    for (size_t i = 0; i < pixel_count; i++)
    {
        const double r = data_[i * color_components];
        const double g = data_[i * color_components + 1];
        const double b = data_[i * color_components + 2];

        double l = luma(r, g, b);

        const double v = floorf((float)i / (float)width) / (float)height;
        const double theta = v * M_PI;
        const double sinTheta = std::sin(theta);
        l *= sinTheta;

        pdf_.push_back(static_cast<float>(l));
        pdf.push_back(l);
    }

    std::cout << "pixel_count: " << pixel_count << std::endl;
    std::cout << std::fixed << std::setprecision(7) << "sum: " << sum << std::endl;
    std::cout << "max: " << max << std::endl;
    std::cout << "min: " << min << std::endl;

    distribution_ = Distribution{pdf};
}

EnvironmentMap::~EnvironmentMap()
{
    delete[] data_;
}