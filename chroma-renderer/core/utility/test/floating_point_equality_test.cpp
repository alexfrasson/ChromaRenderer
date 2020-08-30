#include "chroma-renderer/core/utility/floating_point_equality.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace
{

/// https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
/// https://github.com/google/googletest/blob/df6b75949b1efab7606ba60c0f0a0125ac95c5af/googletest/include/gtest/internal/gtest-internal.h#L250
union Float {
    explicit Float(const float num) : value(num)
    {
    }

    std::int32_t integer;
    float value;
};

float nextDown(const float x)
{
    if (std::isnan(x) || std::isinf(x))
    {
        return x;
    }

    Float f(x);
    if (f.integer > 0)
    {
        f.integer--;
    }
    else if (f.integer < 0)
    {
        f.integer++;
    }
    else
    {
        f.value = -std::numeric_limits<float>::denorm_min();
    }
    return f.value;
}

float nextUp(const float x)
{
    if (std::isnan(x) || std::isinf(x))
    {
        return x;
    }

    Float f(x);
    if (f.integer > 0)
    {
        f.integer++;
    }
    else if (f.integer < 0)
    {
        f.integer--;
    }
    else
    {
        f.value = std::numeric_limits<float>::denorm_min();
    }
    return f.value;
}

// NOLINTNEXTLINE(hicpp-no-array-decay, cppcoreguidelines-owning-memory, hicpp-special-member-functions)
TEST(FloatingPointEqualsTest, ZeroesAreEqual)
{
    EXPECT_TRUE(almostEquals(+0.0f, +0.0f));
    EXPECT_TRUE(almostEquals(+0.0f, -0.0f));
    EXPECT_TRUE(almostEquals(-0.0f, -0.0f));
}

// NOLINTNEXTLINE(hicpp-no-array-decay, cppcoreguidelines-owning-memory, hicpp-special-member-functions)
TEST(FloatingPointEqualsTest, ZeroAndNextRepresentableFloatAreEqual)
{
    const float value{0.0f};
    EXPECT_TRUE(almostEquals(value, nextUp(value)));
}

// NOLINTNEXTLINE(hicpp-no-array-decay, cppcoreguidelines-owning-memory, hicpp-special-member-functions)
TEST(FloatingPointEqualsTest, ZeroAndPreviousRepresentableFloatAreEqual)
{
    const float value{0.0f};
    EXPECT_TRUE(almostEquals(value, nextDown(value)));
}

// NOLINTNEXTLINE(hicpp-no-array-decay, cppcoreguidelines-owning-memory, hicpp-special-member-functions)
TEST(FloatingPointEqualsTest, NextAndPreviousRepresentableFloatAroundZeroAreEqual)
{
    const float value{0.0f};
    EXPECT_TRUE(almostEquals(nextDown(value), nextUp(value)));
}

// NOLINTNEXTLINE(hicpp-no-array-decay, cppcoreguidelines-owning-memory, hicpp-special-member-functions)
TEST(FloatingPointEqualsTest, TwoAndPreviousRepresentableFloatAreEqual)
{
    const float value{2.0f};
    EXPECT_TRUE(almostEquals(value, nextDown(value)));
}

// NOLINTNEXTLINE(hicpp-no-array-decay, cppcoreguidelines-owning-memory, hicpp-special-member-functions)
TEST(FloatingPointEqualsTest, TwoIsEqualToItself)
{
    const float value{2.0f};
    EXPECT_TRUE(almostEquals(value, value));
}

// NOLINTNEXTLINE(hicpp-no-array-decay, cppcoreguidelines-owning-memory, hicpp-special-member-functions)
TEST(FloatingPointEqualsTest, TwoAndNextRepresentableFloatAreDifferent)
{
    // This is were the function starts to report false unless the numbers are exactly the same
    const float value{2.0f};
    EXPECT_FALSE(almostEquals(value, nextUp(value)));
}

// NOLINTNEXTLINE(hicpp-no-array-decay, cppcoreguidelines-owning-memory, hicpp-special-member-functions)
TEST(FloatingPointEqualsTest, LargeNumberIsEqualToItself)
{
    const float value{9999999.0f};
    EXPECT_TRUE(almostEquals(value, value));
}

} // namespace