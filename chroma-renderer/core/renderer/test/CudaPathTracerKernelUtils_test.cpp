#include "chroma-renderer/core/renderer/CudaPathTracerKernelUtils.h"

#include <gtest/gtest.h>

TEST(UtilsTest, Foo_Bar) // NOLINT(hicpp-no-array-decay, cppcoreguidelines-owning-memory,
                         // cppcoreguidelines-special-member-functions, hicpp-special-member-functions)
{
    CudaCamera camera{};
    camera.eye = glm::vec3(0, 0, 0);

    auto ray = rayDirection(0, 0, camera);

    EXPECT_EQ(ray.origin, glm::vec3(0, 0, 0));
}