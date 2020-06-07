#pragma once

#include "chroma-renderer/core/renderer/CudaPathTracerKernelTypes.h"

#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>

#include <cfloat>

#define EPSILON 0.000001f

__device__ glm::vec3 cosineSampleHemisphere(const glm::vec3 normal, const float rand0, const float rand1)
{
    float phi = 2 * glm::pi<float>() * rand0;
    float r2 = rand1;
    float r2s = sqrtf(r2);

    // compute orthonormal coordinate frame uvw with hitpoint as origin
    glm::vec3 w = glm::normalize(normal);
    glm::vec3 u = glm::cross((fabs(w.x) > .1 ? glm::vec3{0, 1, 0} : glm::vec3{1, 0, 0}), w);
    u = glm::normalize(u);
    glm::vec3 v = glm::cross(w, u);

    // compute cosine weighted random ray direction on hemisphere
    return glm::normalize(u * cosf(phi) * r2s + v * sinf(phi) * r2s + w * sqrtf(1 - r2));
}

__device__ CudaRay
rayDirectionWithOffset(const int i, const int j, const CudaCamera cam, const float rand0, const float rand1)
{
    CudaRay ray;
    ray.mint = 0;
    ray.maxt = FLT_MAX;
    ray.origin = cam.eye;

    ray.direction = normalize((float)(i + rand0 - cam.width * 0.5f) * cam.right +
                              (float)(j + rand1 - cam.height * 0.5f) * cam.up + cam.d * cam.forward);
    return ray;
}

// Computes ray direction given camera and pixel position
__host__ __device__ CudaRay rayDirection(const int i, const int j, const CudaCamera cam)
{
    CudaRay ray;
    ray.mint = 0;
    ray.maxt = FLT_MAX;
    ray.origin = cam.eye;
    ray.direction =
        (float)(i - cam.width / 2.0f) * cam.right + (float)(j - cam.height / 2.0f) * cam.up + cam.d * cam.forward;
    ray.direction = normalize(ray.direction);
    return ray;
}

__host__ __device__ bool intersectTriangle(const CudaTriangle* triangle, CudaRay* ray, CudaIntersection* intersection)
{
    const glm::vec3 edge0 = triangle->v[1] - triangle->v[0];
    const glm::vec3 edge1 = triangle->v[2] - triangle->v[0];
    const glm::vec3 pvec = glm::cross(ray->direction, edge1);
    const float det = glm::dot(edge0, pvec);

    // If determinant is near zero, ray lies in plane of triangle
    // With backface culling
    // if (det < EPSILON)
    // Without backface culling
    if (det > -EPSILON && det < EPSILON)
    {
        return false;
    }
    bool backface = det < -EPSILON;
    const float invDet = 1.0f / det;
    const glm::vec3 tvec = ray->origin - triangle->v[0];
    float u = glm::dot(tvec, pvec) * invDet;
    // The intersection lies outside of the triangle
    if (u < 0.0f || u > 1.0f)
    {
        return false;
    }
    const glm::vec3 qvec = glm::cross(tvec, edge0);
    float v = glm::dot(ray->direction, qvec) * invDet;
    // The intersection lies outside of the triangle
    if (v < 0.0f || u + v > 1.0f)
    {
        return false;
    }
    float t = glm::dot(edge1, qvec) * invDet;

    // if (t < EPSILON)
    //	return false;
    if (t > ray->maxt || t < ray->mint)
    {
        return false;
    }

    ray->maxt = t;

    // Fill intersection structure
    intersection->distance = t;
    intersection->p = ray->origin + intersection->distance * ray->direction;
    float gama = 1.0f - (u + v);
    intersection->n = u * triangle->n[1] + v * triangle->n[2] + gama * triangle->n[0];
    intersection->n = (backface ? -1.0f : 1.0f) * glm::normalize(intersection->n);

    intersection->material = triangle->material;

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