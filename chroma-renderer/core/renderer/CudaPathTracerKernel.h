#pragma once

#include "chroma-renderer/core/renderer/CudaPathTracerKernelTypes.h"

#include <cuda_runtime.h>
#include <glm/vec4.hpp>

#include <iostream>

constexpr void cudaErrorCheck(const cudaError_t error_code)
{
    if (error_code != cudaSuccess)
    {
        std::cerr << "CudaError: " << cudaGetErrorString(error_code) << "(" << __FILE__ << "(" << __LINE__ << "))"
                  << std::endl;
        exit(-1);
    }
}

// NOLINTNEXTLINE(clang-diagnostic-unused-const-variable)
constexpr uint32_t kMaxPathDepth{3};

extern "C" void setTextureFilterMode(bool b_linear_filter);
extern "C" void bindTextureToArray(cudaArray* aarray);

extern "C" void trace(cudaStream_t& stream,
                      CudaPathIteration* path_iteration_buffer,
                      glm::vec4* accu_buffer,
                      dim3 tex_dim,
                      CudaCamera cam,
                      CudaTriangle* triangles,
                      CudaMaterial* materials,
                      unsigned int seed,
                      CudaLinearBvhNode* linear_bvh,
                      CudaEnviromentSettings enviroment_settings);