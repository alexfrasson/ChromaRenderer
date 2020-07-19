#pragma once

#include <cfloat>
#include <cuda_runtime.h>
#include <glm/gtc/constants.hpp>
#include <glm/vec3.hpp>

struct CudaPathIteration
{
    glm::vec3 rayOrigin;
    glm::vec3 rayDir;
    glm::vec3 mask;
    glm::vec3 color;
    uint32_t bounce;
    uint32_t samples;
};

struct CudaEnviromentSettings
{
    cudaTextureObject_t texObj;
    int cdf_size;
    int pdf_size;
    float* cdf{nullptr};
    float* pdf{nullptr};
    int width;
    int height;
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

    __device__ glm::vec3 f() const
    {
        return kd * glm::one_over_pi<float>();
    }
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

    CudaBoundingBox(const glm::vec3& a_min, const glm::vec3& a_max)
    {
        min = a_min;
        max = a_max;
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