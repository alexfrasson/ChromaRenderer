#pragma once

#include "chroma-renderer/core/renderer/CudaPathTracerKernelTypes.h"

#include <glm/geometric.hpp>

#include <cfloat>

#define EPSILON 0.000001f

__device__ glm::vec3 cosineSampleHemisphere(const glm::vec3 normal, const float rand0, const float rand1)
{
    // pick two random numbers
    float phi = 2 * M_PI * rand0;
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

// Computes ray direction given camera object and pixel position
__host__ __device__ CudaRay rayDirection(const int i, const int j, CudaCamera cam)
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

// Intersects ray with triangle v0v1v2
__host__ __device__ bool intersectTriangle(const CudaTriangle* tri, CudaRay* ray, CudaIntersection* is)
{
    const glm::vec3 edge0 = tri->v[1] - tri->v[0];
    const glm::vec3 edge1 = tri->v[2] - tri->v[0];
    const glm::vec3 pvec = glm::cross(ray->direction, edge1);
    const float det = glm::dot(edge0, pvec);

    // If determinant is near zero, ray lies in plane of triangle
    // With backface culling
    // if (det < EPSILON)
    // Without backface culling
    if (det > -EPSILON && det < EPSILON)
        return false;
    bool backface = det < -EPSILON;
    const float invDet = 1.0f / det;
    const glm::vec3 tvec = ray->origin - tri->v[0];
    float u = glm::dot(tvec, pvec) * invDet;
    // The intersection lies outside of the triangle
    if (u < 0.0f || u > 1.0f)
        return false;
    const glm::vec3 qvec = glm::cross(tvec, edge0);
    float v = glm::dot(ray->direction, qvec) * invDet;
    // The intersection lies outside of the triangle
    if (v < 0.0f || u + v > 1.0f)
        return false;
    float t = glm::dot(edge1, qvec) * invDet;

    // if (t < EPSILON)
    //	return false;
    if (t > ray->maxt || t < ray->mint)
        return false;

    // Ray intersection
    ray->maxt = t;

    // Fill intersection structure
    is->distance = t;
    // Calcula hitpoint
    is->p = ray->origin + is->distance * ray->direction;
    // Calcula as coordenadas baricentricas
    // const glm::vec3* v0 = is.object->getVertex(is.face, 0);
    // const glm::vec3* v1 = is.object->getVertex(is.face, 1);
    // const glm::vec3* v2 = is.object->getVertex(is.face, 2);
    // float div = 1.0f / glm::dot(glm::cross((*v1 - *v0), (*v2 - *v0)), triangle->tn);
    // float alpha = glm::dot(glm::cross((*v2 - *v1), (hitPoint - *v1)), triangle->tn) * div;
    // float beta = glm::dot(glm::cross((*v0 - *v2), (hitPoint - *v2)), triangle->tn)*div;
    // float gama = glm::dot(glm::cross((f2[i].v[1] - f2[i].v[0]), (point-f2[i].v[0])), f2[i].tn)*div;
    // float gama = 1.0f - (alpha + beta);
    float gama = 1.0f - (u + v);
    // Calcula normal do ponto
    // glm::vec3 hitNormal = alpha * (*n0) + beta * (*n1) + gama * (*n2);
    // is->n = u * (tri->n[1] + v * tri->n[2] + gama * tri->n[0]);
    is->n = u * tri->n[1] + v * tri->n[2] + gama * tri->n[0];
    is->n = (backface ? -1.0f : 1.0f) * glm::normalize(is->n);

    is->material = tri->material;

    return true;
}

// This was taken from
// [WBMS05] Williams, Amy, Steve Barrus, R.Keith Morley, and Peter Shirley. "An efficient and robust ray-box
// intersection algorithm." In ACM SIGGRAPH 2005 Courses, p. 9. ACM, 2005.
__host__ __device__ bool hitBoundingBoxSlab(const CudaBoundingBox& bb,
                                            const CudaRay& r,
                                            const glm::vec3& invRayDir,
                                            const bool* dirIsNeg,
                                            float& tmin,
                                            float& tmax)
{
    float min = (bb[dirIsNeg[0]].x - r.origin.x) * invRayDir.x;
    float max = (bb[1 - dirIsNeg[0]].x - r.origin.x) * invRayDir.x;
    float tymin = (bb[dirIsNeg[1]].y - r.origin.y) * invRayDir.y;
    float tymax = (bb[1 - dirIsNeg[1]].y - r.origin.y) * invRayDir.y;
    if ((min > tymax) || (tymin > max))
        return false;
    if (tymin > min)
        min = tymin;
    if (tymax < max)
        max = tymax;

    tymin = (bb[dirIsNeg[2]].z - r.origin.z) * invRayDir.z;
    tymax = (bb[1 - dirIsNeg[2]].z - r.origin.z) * invRayDir.z;

    if ((min > tymax) || (tymin > max))
        return false;
    if (tymin > min)
        min = tymin;
    if (tymax < max)
        max = tymax;

    return (min < tmax) && (max > tmin);
}

__host__ __device__ bool intersectBVH(const CudaTriangle* triangles,
                                      const CudaLinearBvhNode* linearBVH,
                                      CudaRay& r,
                                      CudaIntersection& intersection)
{
    bool hit = false;
    const glm::vec3 invRayDir = 1.f / r.direction;
    const bool dirIsNeg[3] = {invRayDir.x < 0, invRayDir.y < 0, invRayDir.z < 0};

    unsigned int todoOffset = 0;
    unsigned int nodeNum = 0;
    unsigned int todo[64];

    intersection.distance = FLT_MAX;
    while (true)
    {
        const CudaLinearBvhNode* node = &linearBVH[nodeNum];

        // Intersect BVH node
        if (hitBoundingBoxSlab(node->bbox, r, invRayDir, dirIsNeg, r.mint, r.maxt))
        {
            // Leaf node
            if (node->nPrimitives > 0)
            {
                // Intersect primitives
                for (unsigned int i = node->primitivesOffset; i < node->nPrimitives + node->primitivesOffset; i++)
                {
                    if (intersectTriangle(&triangles[i], &r, &intersection))
                    {
                        hit = true;
                    }
                }
                if (todoOffset == 0)
                    break;
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
                break;
            nodeNum = todo[--todoOffset];
        }
    }

    return hit;
}