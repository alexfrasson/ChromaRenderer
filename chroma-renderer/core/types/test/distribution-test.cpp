#include "chroma-renderer/core/types/distribution.h"

#include <gtest/gtest.h>

TEST(DistributionTest, Foo_Bar)
{
    const size_t expected_index = 1;
    const double value = 0.5;
    const std::vector<double> function = {0.2, 0.4, 0.2, 0.2};
    Distribution distribution{function};

    const auto actual_index = distribution.Sample(value);

    EXPECT_EQ(actual_index, expected_index);
}

TEST(DistributionTest, Foo_Bar2)
{
    const size_t expected_index = 1;
    const double value = 0.5;
    const std::vector<double> function = {2.0, 4.0, 2.0, 2.0};
    Distribution distribution{function};

    const auto actual_index = distribution.Sample(value);

    EXPECT_EQ(actual_index, expected_index);
}