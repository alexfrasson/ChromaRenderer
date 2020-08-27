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
constexpr uint32_t MAX_PATH_DEPTH{3};

extern "C" void setTextureFilterMode(bool bLinearFilter);
extern "C" void bindTextureToArray(cudaArray* aarray);

extern "C" void trace(cudaStream_t& stream,
                      CudaPathIteration* pathIterationBuffer,
                      glm::vec4* accuBuffer,
                      dim3 texDim,
                      CudaCamera cam,
                      CudaTriangle* triangles,
                      CudaMaterial* materials,
                      unsigned int seed,
                      CudaLinearBvhNode* linearBVH,
                      CudaEnviromentSettings enviromentSettings);