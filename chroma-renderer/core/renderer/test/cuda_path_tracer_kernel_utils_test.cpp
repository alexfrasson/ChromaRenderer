#include "chroma-renderer/core/renderer/cuda_path_tracer_kernel_utils.h"

#include <gtest/gtest.h>

// NOLINTNEXTLINE(hicpp-no-array-decay, cppcoreguidelines-owning-memory, hicpp-special-member-functions)
TEST(UtilsTest, Foo_Bar)
{
    CudaCamera camera{};
    camera.eye = glm::vec3(0, 0, 0);

    auto ray = rayDirection(0, 0, camera);

    EXPECT_EQ(ray.origin, glm::vec3(0, 0, 0));
}