#pragma once

#include <cmath>
#include <limits>

/// @brief Naive implementation of equality comparison of floating-point numbers.
/// The following links show how this could be improved:
/// https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
/// https://github.com/google/googletest/blob/df6b75949b1efab7606ba60c0f0a0125ac95c5af/googletest/include/gtest/internal/gtest-internal.h#L250
/// https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
template <typename T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type almostEquals(const T lhs, const T rhs)
{
    return std::abs(lhs - rhs) <= std::numeric_limits<T>::epsilon();
}