#pragma once

#include "chroma-renderer/core/renderer/CudaPathTracerKernelTypes.h"

#include <cuda_runtime.h>
#include <glm/vec4.hpp>

#include <iostream>

#define cudaErrorCheck(ans)                                                                                     \
    {                                                                                                           \
        if (ans != cudaSuccess)                                                                                 \
        {                                                                                                       \
            std::cerr << "CudaError: " << cudaGetErrorString(ans) << "(" << __FILE__ << "(" << __LINE__ << "))" \
                      << std::endl;                                                                             \
            exit(-1);                                                                                           \
            return;                                                                                             \
        }                                                                                                       \
    }

#define MAX_PATH_DEPTH 3

extern "C" void setTextureFilterMode(bool bLinearFilter);
extern "C" void bindTextureToArray(cudaArray* aarray);

extern "C" void trace(cudaStream_t& stream,
                      CudaPathIteration* pathIterationBuffer,
                      glm::vec4* accuBuffer,
                      dim3 texDim,
                      CudaCamera cam,
                      CudaTriangle* triangles,
                      unsigned int nTriangles,
                      CudaMaterial* materials,
                      unsigned int nMaterials,
                      unsigned int seed,
                      CudaLinearBvhNode* linearBVH,
                      CudaEnviromentSettings enviromentSettings);