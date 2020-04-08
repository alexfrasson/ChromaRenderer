#pragma once

#define MAX_PATH_DEPTH 3

#define cudaErrorCheck(ans) { if(ans != cudaSuccess){ cerr << "CudaError: " << cudaGetErrorString(ans) << "(" << __FILE__ << "(" << __LINE__ << "))" << endl; exit(-1); return;} }

#include <cfloat>
#include <stdint.h>
#include <cuda-helpers\helper_math.h>

#if defined(__CUDACC__) // NVCC
#define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define MY_ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

struct CudaPathIteration
{
	float3 rayOrigin;
	float3 rayDir;
	float3 mask;
	int bounce;
};

struct CudaEnviromentSettings
{
	float3 enviromentLightColor;
	float enviromentLightIntensity;
	cudaTextureObject_t texObj;
};

struct CudaCamera
{
	int width;
	int height;
	float d;
	float3 right, up, forward;
	float3 eye;
};

struct CudaRay
{
	float3 origin;
	float3 direction;
	float mint;
	float maxt;
};

struct CudaTriangle
{
	int material;
	float3 v[3];
	float3 n[3];
};

struct CudaIntersection
{
	float distance;
	int material;
	float3 p;
	float3 n;
};

struct CudaMaterial
{
	float3 kd;
	float3 ke;
	float3 transparent;
};

struct CudaBoundingBox
{
	float3 max, min;
	CudaBoundingBox()
	{
		max.x = -FLT_MAX;
		max.y = max.x;
		max.z = max.x;

		min.x = FLT_MAX;
		min.y = min.x;
		min.z = min.x;
	}
	CudaBoundingBox(const float3& min, const float3& max)
	{
		this->min = min;
		this->max = max;
	}
	__host__ __device__ float3& operator[](const int& i)
	{
		if (i == 0)
			return min;
		return max;
	}
	__host__ __device__ const float3& operator[](const int& i) const
	{
		if (i == 0)
			return min;
		return max;
	}
};

struct CudaLinearBvhNode
{
	CudaBoundingBox bbox;
	union
	{
		unsigned int primitivesOffset;	// Leaf
		unsigned int secondChildOffset; // Interior
	};
	unsigned char nPrimitives;	// 0 -> interior node
	unsigned char axis;
	//unsigned char pad[2];
};

extern "C"
void setTextureFilterMode(bool bLinearFilter);
extern "C"
void bindTextureToArray(cudaArray* aarray);

//void trace(cudaSurfaceObject_t surface, dim3 texDim, CudaCamera cam, CudaTriangle* triangles, unsigned int nTriangles, CudaMaterial* materials, unsigned int nMaterials, unsigned int seed);
//void trace(cudaSurfaceObject_t surface, dim3 texDim, CudaCamera cam, CudaTriangle* triangles, unsigned int nTriangles, CudaMaterial* materials, unsigned int nMaterials, unsigned int seed, CudaLinearBvhNode* linearBVH);
//void trace(float4* accuBuffer, dim3 texDim, CudaCamera cam, CudaTriangle* triangles, unsigned int nTriangles, CudaMaterial* materials, unsigned int nMaterials, unsigned int seed, CudaLinearBvhNode* linearBVH, bool enviromentLight);
//void trace(cudaStream_t &stream, float4* accuBuffer, dim3 texDim, CudaCamera cam, CudaTriangle* triangles, unsigned int nTriangles, CudaMaterial* materials, unsigned int nMaterials, unsigned int seed, CudaLinearBvhNode* linearBVH, CudaEnviromentSettings enviromentSettings);
extern "C"
void trace(cudaStream_t &stream, CudaPathIteration* pathIterationBuffer, float4* accuBuffer, dim3 texDim, CudaCamera cam, CudaTriangle* triangles, unsigned int nTriangles, CudaMaterial* materials, unsigned int nMaterials, unsigned int seed, CudaLinearBvhNode* linearBVH, CudaEnviromentSettings enviromentSettings);