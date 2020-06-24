#pragma once

#include <cuda_runtime.h>
#include <glm/vec3.hpp>

#include <cfloat>

struct CudaPathIteration
{
    glm::vec3 rayOrigin;
    glm::vec3 rayDir;
    glm::vec3 mask;
    int bounce;
};

struct CudaEnviromentSettings
{
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