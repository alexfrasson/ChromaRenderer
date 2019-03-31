#include "gtest/gtest.h"

#include <CudaPathTracerKernel.h>

TEST(TestCaseName, TestName) 
{
	CudaCamera camera;

	CudaRay ray = rayDirection(0, 0, camera);

	EXPECT_NEAR(ray.direction.x, 0.0, 0.0);
}