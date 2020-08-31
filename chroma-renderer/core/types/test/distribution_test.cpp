#include "chroma-renderer/core/types/distribution.h"

#include <gtest/gtest.h>

// NOLINTNEXTLINE(hicpp-no-array-decay, cppcoreguidelines-owning-memory, hicpp-special-member-functions)
TEST(DistributionTest, Foo_Bar)
{
    const std::size_t expected_index = 1;
    const float value = 0.5;
    const std::vector<double> function = {0.2, 0.4, 0.2, 0.2};
    Distribution distribution{function};

    const auto actual_index = distribution.sample(value);

    EXPECT_EQ(actual_index, expected_index);
}

// NOLINTNEXTLINE(hicpp-no-array-decay, cppcoreguidelines-owning-memory, hicpp-special-member-functions)
TEST(DistributionTest, Foo_Bar2)
{
    const std::size_t expected_index = 1;
    const float value = 0.5;
    const std::vector<double> function = {2.0, 4.0, 2.0, 2.0};
    Distribution distribution{function};

    const auto actual_index = distribution.sample(value);

    EXPECT_EQ(actual_index, expected_index);
}