#include "chroma-renderer/core/renderer/CudaPathTracerKernel.h"

#include <curand_kernel.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include <glm/geometric.hpp>

#define THREAD_DIM 8
#define EPSILON 0.000001f
#define SAMPLES 1

texture<glm::vec4, cudaTextureType2D, /*cudaReadModeNormalizedFloat*/ cudaReadModeElementType> accuTex;

__device__ __inline__ glm::vec3 cosineSampleHemisphere(curandState* randState, glm::vec3 normal)
{
    // pick two random numbers
    float phi = 2 * M_PI * curand_uniform(randState);
    float r2 = curand_uniform(randState);
    float r2s = sqrtf(r2);

    // compute orthonormal coordinate frame uvw with hitpoint as origin
    glm::vec3 w = glm::normalize(normal);
    glm::vec3 u = glm::cross((fabs(w.x) > .1 ? glm::vec3{0, 1, 0} : glm::vec3{1, 0, 0}), w);
    u = glm::normalize(u);
    glm::vec3 v = glm::cross(w, u);

    // compute cosine weighted random ray direction on hemisphere
    return glm::normalize(u * cosf(phi) * r2s + v * sinf(phi) * r2s + w * sqrtf(1 - r2));
}

__device__ __inline__ CudaRay rayDirectionWithOffset(const int i, const int j, CudaCamera cam, curandState* randState)
{
    CudaRay ray;
    ray.mint = 0;
    ray.maxt = FLT_MAX;
    ray.origin = cam.eye;

    float random0 = curand_uniform(randState);
    float random1 = curand_uniform(randState);

    ray.direction = normalize((float)(i + random0 - cam.width * 0.5f) * cam.right +
                              (float)(j + random1 - cam.height * 0.5f) * cam.up + cam.d * cam.forward);
    return ray;
}

// Computes ray direction given camera object and pixel position
extern "C" __host__ __device__ __inline__ CudaRay rayDirection(const int i, const int j, CudaCamera cam)
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
__host__ __device__ __inline__ bool intersectTriangle(const CudaTriangle* tri, CudaRay* ray, CudaIntersection* is)
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
__host__ __device__ __inline__ bool hitBoundingBoxSlab(const CudaBoundingBox& bb,
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

__global__ void traceKernel(CudaPathIteration* pathIterationBuffer,
                            glm::vec4* accuBuffer,
                            const dim3 texDim,
                            const CudaCamera cam,
                            const CudaTriangle* triangles,
                            int nTriangles,
                            const CudaMaterial* materials,
                            const unsigned int nMaterials,
                            const unsigned int seed,
                            const CudaLinearBvhNode* linearBVH,
                            const CudaEnviromentSettings enviromentSettings)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= texDim.x || y >= texDim.y)
        return;

    int pos = texDim.x * y + x;

    // global threadId, see richiesams blogspot
    int threadId =
        (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    // create random number generator and initialise with hashed frame number, see RichieSams blogspot
    curandState randState; // state of the random number generator, to prevent repetition
    curand_init(seed + threadId, 0, 0, &randState);

    CudaPathIteration pathIteration = pathIterationBuffer[pos];
    CudaRay ray;
    glm::vec3 color;

    // Begin path
    if (pathIteration.bounce == 0)
    {
        ray = rayDirectionWithOffset(x, y, cam, &randState);
        pathIteration.mask = glm::vec3{1.0f, 1.0f, 1.0f};
    }
    else
    {
        ray.mint = 0;
        ray.maxt = FLT_MAX;
        ray.direction = pathIteration.rayDir;
        ray.origin = pathIteration.rayOrigin;
    }

    CudaIntersection is;

    // No intersection
    if (!intersectBVH(triangles, linearBVH, ray, is))
    {
        float u = atan2(ray.direction.x, ray.direction.z) / (2 * M_PI) + 0.5;
        float v = ray.direction.y * 0.5 + 0.5;

        float4 env = tex2D<float4>(enviromentSettings.texObj, u, 1.0f - v);
        color = pathIteration.mask * glm::vec3(env.x, env.y, env.z);

        // float4 env = tex2D<float4>(enviromentSettings.texObj, u, 1.0f - v) *
        // enviromentSettings.enviromentLightIntensity; color = pathIteration.mask * make_float3(pow(env.x, 1.0f
        // / 2.2f), pow(env.y, 1.0f / 2.2f), pow(env.z, 1.0f / 2.2f));

        // Restart from the camera.
        pathIteration.bounce = 0;
    }
    else
    {
        // Pick a random direction from here and keep going.
        if (materials[is.material].transparent.x < 1.0 || materials[is.material].transparent.y < 1.0 ||
            materials[is.material].transparent.z < 1.0)
        {
            color = glm::vec3{0.0f, 0.0f, 0.0f};
            // pathIteration.bounce--;
            ray.origin = is.p + ray.direction * 0.0001f;
        }
        else
        {
            // Intersection
            glm::vec3 emittance;
            glm::vec3 kd = materials[is.material].kd;
            glm::vec3 ke = materials[is.material].ke;

            emittance = ke * kd;

            ray.origin = is.p + is.n * 0.0001f;
            ray.direction = cosineSampleHemisphere(&randState, is.n);

            // Compute the BRDF for this ray (assuming Lambertian reflection)
            float cos_theta = glm::dot(ray.direction, is.n);

            glm::vec3 BRDF = kd * 2.0f * cos_theta;

            // Apply the Rendering Equation here.
            color = pathIteration.mask * emittance;
            pathIteration.mask *= BRDF;
        }

        pathIteration.rayDir = ray.direction;
        pathIteration.rayOrigin = ray.origin;

        pathIteration.bounce++;

        // Reached max depth. Restart from the camera.
        if (pathIteration.bounce == MAX_PATH_DEPTH)
            pathIteration.bounce = 0;
    }

    accuBuffer[pos] += glm::vec4(color.x, color.y, color.z, 1.0f / (float)MAX_PATH_DEPTH);
    pathIterationBuffer[pos] = pathIteration;
}

extern "C" void setTextureFilterMode(bool bLinearFilter)
{
    accuTex.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
}

extern "C" void bindTextureToArray(cudaArray* aarray)
{
    // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc channelDesc;
    cudaErrorCheck(cudaGetChannelDesc(&channelDesc, aarray));

    // set texture parameters
    accuTex.normalized = false;                   // access with normalized texture coordinates
    accuTex.filterMode = cudaFilterModePoint;     // linear interpolation
    accuTex.addressMode[0] = cudaAddressModeWrap; // wrap texture coordinates
    accuTex.addressMode[1] = cudaAddressModeWrap;

    cudaErrorCheck(cudaBindTextureToArray(accuTex, aarray, channelDesc));
}

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
                      CudaEnviromentSettings enviromentSettings)
{
    dim3 thread(THREAD_DIM, THREAD_DIM);
    dim3 block((unsigned int)ceilf((float)texDim.x / (float)thread.x),
               (unsigned int)ceilf((float)texDim.y / (float)thread.y));
    traceKernel<<<block, thread, 0, stream>>>(pathIterationBuffer,
                                              accuBuffer,
                                              texDim,
                                              cam,
                                              triangles,
                                              nTriangles,
                                              materials,
                                              nMaterials,
                                              seed,
                                              linearBVH,
                                              enviromentSettings);
}