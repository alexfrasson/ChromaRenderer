#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda-helpers\helper_math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <CudaPathTracerKernel.h>
#include <cfloat>
#include <iostream>

__global__ void writeAccuBufferToTextureKernel(float4* accuBuffer, cudaSurfaceObject_t surface, dim3 texDim, unsigned int nFrames)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= texDim.x || y >= texDim.y)
		return;

	int pos = texDim.x *y + x;

	float4 color = accuBuffer[pos] / (float)nFrames;

	surf2Dwrite(make_uchar4(max(0.0f, min(1.0f, color.x)) * 255, max(0.0f, min(1.0f, color.y)) * 255, max(0.0f, min(1.0f, color.z)) * 255, max(0.0f, min(1.0f, color.w)) * 255), surface, x * sizeof(uchar4), y);
}

void writeAccuBufferToTexture(cudaStream_t &stream, float4* accuBuffer, cudaSurfaceObject_t surface, dim3 texDim, unsigned int nFrames)
{
	dim3 thread(32, 32);
	dim3 block(ceilf((float)texDim.x / (float)thread.x), ceilf((float)texDim.y / (float)thread.y));
	writeAccuBufferToTextureKernel <<< block, thread, 0, stream>>>(accuBuffer, surface, texDim, nFrames);
}