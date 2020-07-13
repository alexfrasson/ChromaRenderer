#pragma once

#include "chroma-renderer/core/renderer/CudaPathTracerKernelTypes.h"

#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>

#include <cfloat>

#define EPSILON 0.000001f

__host__ __device__ int binarySearch(const float* cdf, const int cdf_size, const float rand_var)
{
    int start = 1;
    int end = cdf_size;
    int index = -1;
    while (start < end)
    {
        index = static_cast<int>(floorf(((float)start + (float)end) / 2.0f));

        if (rand_var >= cdf[index - 1] && rand_var <= cdf[index])
        {
            break;
        }

        if (rand_var > cdf[index])
        {
            start = index + 1;
        }
        else
        {
            end = index - 1;
        }
    }
    return index;
}

__device__ glm::mat3 basis(glm::vec3 normal)
{
    glm::vec3 binormal;
    if (std::abs(normal.x) > std::abs(normal.z))
    {
        binormal = glm::vec3(-normal.y, normal.x, 0.0f);
    }
    else
    {
        binormal = glm::vec3(0.0f, -normal.z, normal.y);
    }
    binormal = glm::normalize(binormal);
    const glm::vec3 tangent = glm::normalize(glm::cross(binormal, normal));
    return glm::mat3{tangent, binormal, normal};
}

__device__ glm::vec3 toWorld(const glm::vec3 dir, const glm::vec3 normal)
{
    const glm::mat3 base = basis(normal);
    return glm::normalize(base * dir);
}

__device__ glm::vec3 toLocal(const glm::vec3 dir, const glm::vec3 normal)
{
    const glm::mat3 base = glm::transpose(basis(normal));
    return glm::normalize(base * dir);
}

struct SampleDirection
{
    glm::vec3 direction;
    float pdf;
};

__device__ float sphericalTheta(const glm::vec3& unit_vector)
{
    return acosf(unit_vector.y);
}

__device__ float sphericalPhi(const glm::vec3& unit_vector)
{
    const float p = atan2f(unit_vector.z, unit_vector.x);
    return (p < 0.0f) ? (p + 2.0f * glm::pi<float>()) : p;
}

__device__ bool sameHemisphere(const glm::vec3& n, const glm::vec3& a, const glm::vec3& b)
{
    return ((glm::dot(n, a) * glm::dot(n, b)) > 0.0f);
}

__device__ bool sameHemisphere(const glm::vec3& a, const glm::vec3& b)
{
    return (a.z * b.z > 0.0f);
}

__device__ float uniformSampleHemispherePdf(const glm::vec3& n, const glm::vec3& wo, const glm::vec3& wi)
{
    if (!sameHemisphere(n, wo, wi))
    {
        return 0.0f;
    }
    return 1.0f / glm::two_pi<float>();
}

__device__ SampleDirection uniformSampleHemisphere(const float rand0,
                                                   const float rand1,
                                                   const glm::vec3& n,
                                                   const glm::vec3& wo)
{
    // cos(theta) = rand0 = y
    // cos^2(theta) + sin^2(theta) = 1 -> sin(theta) = srtf(1 - cos^2(theta))
    const float sinTheta = sqrtf(1 - rand0 * rand0);
    const float phi = glm::two_pi<float>() * rand1;
    const float x = sinTheta * cosf(phi);
    const float z = sinTheta * sinf(phi);
    const glm::vec3 local_direction = glm::normalize(glm::vec3(x, z, rand0));
    const glm::vec3 wi = toWorld(local_direction, n);
    const float pdf = uniformSampleHemispherePdf(n, wo, wi);
    return SampleDirection{wi, pdf};
}

__device__ float uniformSampleCosineWeightedHemispherePdf(const glm::vec3& n, const glm::vec3& wo, const glm::vec3& wi)
{
    if (!sameHemisphere(n, wo, wi))
    {
        return 0.0f;
    }
    return std::abs(glm::dot(n, wi)) * glm::one_over_pi<float>();
}

__device__ SampleDirection uniformSampleCosineWeightedHemisphere(const float rand0,
                                                                 const float rand1,
                                                                 const glm::vec3& n,
                                                                 const glm::vec3& wo)
{
    const float theta = asinf(sqrtf(rand0));
    const float phi = 2.0f * glm::pi<float>() * rand1;
    const float sinTheta = sinf(theta);
    const float cosTheta = cosf(theta);
    const float sinPhi = sinf(phi);
    const float cosPhi = cosf(phi);
    const float x = sinTheta * cosPhi;
    const float y = sinTheta * sinPhi;
    const float z = cosTheta;

    const glm::vec3 local_direction = glm::normalize(glm::vec3(x, y, z));
    const glm::vec3 wi = toWorld(local_direction, n);
    const float pdf = uniformSampleCosineWeightedHemispherePdf(n, wo, wi);

    return SampleDirection{wi, pdf};
}

__device__ glm::vec3 cosineSampleHemisphere(const glm::vec3 normal, const float rand0, const float rand1)
{
    const glm::vec3 w = glm::normalize(normal);
    const glm::vec3 u = glm::normalize(glm::cross((fabs(w.x) > 0.1f ? glm::vec3{0, 1, 0} : glm::vec3{1, 0, 0}), w));
    const glm::vec3 v = glm::cross(w, u);

    const float phi = 2 * glm::pi<float>() * rand0;
    const float rand1_sqrt = sqrtf(rand1);

    return glm::normalize(u * cosf(phi) * rand1_sqrt + v * sinf(phi) * rand1_sqrt + w * sqrtf(1.0f - rand1));
}

__device__ CudaRay
rayDirectionWithOffset(const int i, const int j, const CudaCamera cam, const float rand0, const float rand1)
{
    CudaRay ray;
    ray.mint = 0;
    ray.maxt = FLT_MAX;
    ray.origin = cam.eye;
    ray.direction = glm::normalize(((float)i + rand0 - (float)cam.width * 0.5f) * cam.right +
                                   ((float)j + rand1 - (float)cam.height * 0.5f) * cam.up + cam.d * cam.forward);
    return ray;
}

// https://people.cs.clemson.edu/~dhouse/courses/405/notes/texture-maps.pdf
__device__ glm::vec2 unitVectorToUv(const glm::vec3& unit_vector)
{
    const float theta = sphericalTheta(unit_vector);
    const float phi = sphericalPhi(unit_vector);
    const float u = (phi + glm::pi<float>()) / glm::two_pi<float>();
    const float v = theta * glm::one_over_pi<float>();
    return glm::vec2(u, v);
}

// https://people.cs.clemson.edu/~dhouse/courses/405/notes/texture-maps.pdf
__host__ __device__ glm::vec3 uvToUnitVector(const glm::vec2& uv)
{
    const float phi = (2.0f * uv.x - 1.0f) * glm::pi<float>();
    const float theta = uv.y * glm::pi<float>();
    const float cosTheta = cosf(theta);
    const float sinTheta = sinf(theta);
    const float sinPhi = sinf(phi);
    const float cosPhi = cosf(phi);
    return glm::vec3(cosPhi * sinTheta, cosTheta, sinPhi * sinTheta);
}

// Computes ray direction given camera and pixel position
__host__ __device__ CudaRay rayDirection(const int i, const int j, const CudaCamera cam)
{
    CudaRay ray;
    ray.mint = 0;
    ray.maxt = FLT_MAX;
    ray.origin = cam.eye;
    ray.direction = ((float)i - (float)cam.width / 2.0f) * cam.right + ((float)j - (float)cam.height / 2.0f) * cam.up +
                    cam.d * cam.forward;
    ray.direction = glm::normalize(ray.direction);
    return ray;
}

__host__ __device__ bool intersectTriangle(const CudaTriangle* triangle, CudaRay* ray, CudaIntersection* intersection)
{
    const glm::vec3 edge0 = triangle->v[1] - triangle->v[0];
    const glm::vec3 edge1 = triangle->v[2] - triangle->v[0];
    const glm::vec3 pvec = glm::cross(ray->direction, edge1);
    const float det = glm::dot(edge0, pvec);

    if (det > -EPSILON && det < EPSILON)
    {
        return false;
    }
    const float invDet = 1.0f / det;
    const glm::vec3 tvec = ray->origin - triangle->v[0];
    const float u = glm::dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f)
    {
        return false;
    }
    const glm::vec3 qvec = glm::cross(tvec, edge0);
    const float v = glm::dot(ray->direction, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f)
    {
        return false;
    }
    const float t = glm::dot(edge1, qvec) * invDet;
    if (t > ray->maxt || t < ray->mint)
    {
        return false;
    }

    ray->maxt = t;

    intersection->distance = t;
    intersection->p = ray->origin + intersection->distance * ray->direction;
    float gama = 1.0f - (u + v);
    intersection->n = u * triangle->n[1] + v * triangle->n[2] + gama * triangle->n[0];
    const bool backface = det < -EPSILON;
    intersection->n = (backface ? -1.0f : 1.0f) * glm::normalize(intersection->n);
    intersection->material = triangle->material;

    return true;
}

__host__ __device__ bool intersectTriangle(const CudaTriangle* triangle, CudaRay* ray)
{
    const glm::vec3 edge0 = triangle->v[1] - triangle->v[0];
    const glm::vec3 edge1 = triangle->v[2] - triangle->v[0];
    const glm::vec3 pvec = glm::cross(ray->direction, edge1);
    const float det = glm::dot(edge0, pvec);

    if (det > -EPSILON && det < EPSILON)
    {
        return false;
    }
    const float invDet = 1.0f / det;
    const glm::vec3 tvec = ray->origin - triangle->v[0];
    const float u = glm::dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f)
    {
        return false;
    }
    const glm::vec3 qvec = glm::cross(tvec, edge0);
    const float v = glm::dot(ray->direction, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f)
    {
        return false;
    }
    const float t = glm::dot(edge1, qvec) * invDet;
    if (t > ray->maxt || t < ray->mint)
    {
        return false;
    }

    ray->maxt = t;
    return true;
}

// [WBMS05] Williams, Amy, Steve Barrus, R.Keith Morley, and Peter Shirley. "An efficient and robust ray-box
// intersection algorithm." In ACM SIGGRAPH 2005 Courses, p. 9. ACM, 2005.
__host__ __device__ bool hitBoundingBoxSlab(const CudaBoundingBox& bb,
                                            const CudaRay& ray,
                                            const glm::vec3& invRayDir,
                                            const bool* dirIsNeg,
                                            float& tmin,
                                            float& tmax)
{
    float min = (bb[dirIsNeg[0]].x - ray.origin.x) * invRayDir.x;
    float max = (bb[1 - dirIsNeg[0]].x - ray.origin.x) * invRayDir.x;
    float tymin = (bb[dirIsNeg[1]].y - ray.origin.y) * invRayDir.y;
    float tymax = (bb[1 - dirIsNeg[1]].y - ray.origin.y) * invRayDir.y;
    if ((min > tymax) || (tymin > max))
    {
        return false;
    }
    if (tymin > min)
    {
        min = tymin;
    }
    if (tymax < max)
    {
        max = tymax;
    }

    tymin = (bb[dirIsNeg[2]].z - ray.origin.z) * invRayDir.z;
    tymax = (bb[1 - dirIsNeg[2]].z - ray.origin.z) * invRayDir.z;

    if ((min > tymax) || (tymin > max))
    {
        return false;
    }
    if (tymin > min)
    {
        min = tymin;
    }
    if (tymax < max)
    {
        max = tymax;
    }

    return (min < tmax) && (max > tmin);
}

__host__ __device__ bool intersectBVH(const CudaTriangle* triangles,
                                      const CudaLinearBvhNode* linearBVH,
                                      CudaRay& ray,
                                      CudaIntersection& intersection)
{
    bool hit = false;
    const glm::vec3 invRayDir = 1.f / ray.direction;
    const bool dirIsNeg[3] = {invRayDir.x < 0, invRayDir.y < 0, invRayDir.z < 0};

    unsigned int todoOffset = 0;
    unsigned int nodeNum = 0;
    unsigned int todo[64];

    intersection.distance = FLT_MAX;
    while (true)
    {
        const CudaLinearBvhNode* node = &linearBVH[nodeNum];

        // Intersect BVH node
        if (hitBoundingBoxSlab(node->bbox, ray, invRayDir, dirIsNeg, ray.mint, ray.maxt))
        {
            // Leaf node
            if (node->nPrimitives > 0)
            {
                // Intersect primitives
                for (unsigned int i = node->primitivesOffset; i < node->nPrimitives + node->primitivesOffset; i++)
                {
                    if (intersectTriangle(&triangles[i], &ray, &intersection))
                    {
                        hit = true;
                    }
                }
                if (todoOffset == 0)
                {
                    break;
                }
                nodeNum = todo[--todoOffset];
            }
            // Internal node
            else
            {
                if (dirIsNeg[node->axis])
                {
                    todo[todoOffset++] = nodeNum + 1;
                    nodeNum = node->secondChildOffset;
                }
                else
                {
                    todo[todoOffset++] = node->secondChildOffset;
                    nodeNum = nodeNum + 1;
                }
            }
        }
        else
        {
            if (todoOffset == 0)
            {
                break;
            }
            nodeNum = todo[--todoOffset];
        }
    }

    return hit;
}

__host__ __device__ bool intersectBVH(const CudaTriangle* triangles, const CudaLinearBvhNode* linearBVH, CudaRay& ray)
{
    const glm::vec3 invRayDir = 1.f / ray.direction;
    const bool dirIsNeg[3] = {invRayDir.x < 0, invRayDir.y < 0, invRayDir.z < 0};

    unsigned int todoOffset = 0;
    unsigned int nodeNum = 0;
    unsigned int todo[64];

    while (true)
    {
        const CudaLinearBvhNode* node = &linearBVH[nodeNum];

        // Intersect BVH node
        if (hitBoundingBoxSlab(node->bbox, ray, invRayDir, dirIsNeg, ray.mint, ray.maxt))
        {
            // Leaf node
            if (node->nPrimitives > 0)
            {
                // Intersect primitives
                for (unsigned int i = node->primitivesOffset; i < node->nPrimitives + node->primitivesOffset; i++)
                {
                    if (intersectTriangle(&triangles[i], &ray))
                    {
                        return true;
                    }
                }
                if (todoOffset == 0)
                {
                    break;
                }
                nodeNum = todo[--todoOffset];
            }
            // Internal node
            else
            {
                if (dirIsNeg[node->axis])
                {
                    todo[todoOffset++] = nodeNum + 1;
                    nodeNum = node->secondChildOffset;
                }
                else
                {
                    todo[todoOffset++] = node->secondChildOffset;
                    nodeNum = nodeNum + 1;
                }
            }
        }
        else
        {
            if (todoOffset == 0)
            {
                break;
            }
            nodeNum = todo[--todoOffset];
        }
    }

    return false;
}