#include "chroma-renderer/core/renderer/CudaPathTracerKernel.h"
#include "chroma-renderer/core/renderer/CudaPathTracerKernelUtils.h"

#include <curand_kernel.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include <glm/geometric.hpp>

#define THREAD_DIM 8

texture<glm::vec4, cudaTextureType2D, /*cudaReadModeNormalizedFloat*/ cudaReadModeElementType> accuTex;

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