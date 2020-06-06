#pragma once

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

#include <cuda_runtime.h>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <cfloat>

#if defined(__CUDACC__) // NVCC
#define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define MY_ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

#define MAX_PATH_DEPTH 3

struct CudaPathIteration
{
    glm::vec3 rayOrigin;
    glm::vec3 rayDir;
    glm::vec3 mask;
    int bounce;
};

struct CudaEnviromentSettings
{
    glm::vec3 enviromentLightColor;
    float enviromentLightIntensity;
    cudaTextureObject_t texObj;
};

struct CudaCamera
{
    int width;
    int height;
    float d;
    glm::vec3 right, up, forward;
    glm::vec3 eye;
};

struct CudaRay
{
    glm::vec3 origin;
    glm::vec3 direction;
    float mint;
    float maxt;
};

struct CudaTriangle
{
    int material;
    glm::vec3 v[3];
    glm::vec3 n[3];
};

struct CudaIntersection
{
    float distance;
    int material;
    glm::vec3 p;
    glm::vec3 n;
};

struct CudaMaterial
{
    glm::vec3 kd;
    glm::vec3 ke;
    glm::vec3 transparent;
};

struct CudaBoundingBox
{
    glm::vec3 max, min;
    CudaBoundingBox()
    {
        max.x = -FLT_MAX;
        max.y = max.x;
        max.z = max.x;

        min.x = FLT_MAX;
        min.y = min.x;
        min.z = min.x;
    }
    CudaBoundingBox(const glm::vec3& min, const glm::vec3& max)
    {
        this->min = min;
        this->max = max;
    }
    __host__ __device__ glm::vec3& operator[](const int& i)
    {
        if (i == 0)
            return min;
        return max;
    }
    __host__ __device__ const glm::vec3& operator[](const int& i) const
    {
        if (i == 0)
            return min;
        return max;
    }
};

struct CudaLinearBvhNode
{
    CudaBoundingBox bbox;
    union {
        unsigned int primitivesOffset;  // Leaf
        unsigned int secondChildOffset; // Interior
    };
    unsigned char nPrimitives; // 0 -> interior node
    unsigned char axis;
    // unsigned char pad[2];
};

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