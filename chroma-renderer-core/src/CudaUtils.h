#pragma once

//#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cfloat>
#include <cuda-helpers/helper_math.h>

void writeAccuBufferToTexture(cudaStream_t& stream,
                              float4* accuBuffer,
                              cudaSurfaceObject_t surface,
                              dim3 texDim,
                              unsigned int nFrames);