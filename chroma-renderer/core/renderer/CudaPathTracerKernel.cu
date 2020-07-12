#include "chroma-renderer/core/renderer/CudaPathTracerKernel.h"
#include "chroma-renderer/core/renderer/CudaPathTracerKernelUtils.h"

#include <cmath>
#include <curand_kernel.h>

#define THREAD_DIM 8

__device__ void finishSample(const int pos, glm::vec4* accuBuffer, CudaPathIteration* pathIteration)
{
    glm::vec3 current_avg{accuBuffer[pos].x, accuBuffer[pos].y, accuBuffer[pos].z};

    if (pathIteration->samples == 0)
    {
        current_avg = glm::vec3(0.0f, 0.0f, 0.0f);
    }

    pathIteration->samples++;

    const glm::vec3 new_avg = current_avg + (pathIteration->color - current_avg) / (float)pathIteration->samples;
    accuBuffer[pos] = glm::vec4(new_avg.x, new_avg.y, new_avg.z, 1.0f);

    pathIteration->bounce = 0;
    pathIteration->mask = glm::vec3{1.0f, 1.0f, 1.0f};
    pathIteration->color = glm::vec3{0.0f, 0.0f, 0.0f};
}

struct EnvSample
{
    glm::vec3 direction;
    glm::vec2 uv;
    float pdf;
};

__device__ float envMapPdf(const float pdf, const float v, const float height)
{
    // const float theta = (2.0f * u - 1.0f) * glm::pi<float>();
    // const float theta = glm::pi<float>() * float(v + 0.5f) / float(height);
    const float theta = v * glm::pi<float>();
    const float sinTheta = sinf(theta);
    if (abs(sinTheta) < FLT_EPSILON)
    {
        return pdf / (2.0f * glm::pi<float>() * glm::pi<float>());
    }
    return pdf / (2.0f * glm::pi<float>() * glm::pi<float>() * sinTheta);
}

__device__ float envMapPdf(const glm::vec3 direction, const CudaEnviromentSettings& enviromentSettings)
{
    const glm::vec2 uv = unitVectorToUv(direction);
    const uint32_t ui = uv.x * (float)enviromentSettings.width;
    const uint32_t vi = uv.y * (float)enviromentSettings.height;
    const uint32_t index = vi * enviromentSettings.width + ui;
    return envMapPdf(enviromentSettings.pdf[index], uv.y, enviromentSettings.height);
}

__device__ EnvSample sampleEnv(curandState& randState, const CudaEnviromentSettings& enviromentSettings)
{
    const float random_var = curand_uniform(&randState);
    int index = binarySearch(enviromentSettings.cdf, enviromentSettings.cdf_size, random_var) - 1;
    const int width = enviromentSettings.width;
    const int height = enviromentSettings.height;
    const float u = (index % width) / (float)width;
    const float v = floorf((float)index / (float)width) / (float)height;
    const glm::vec3 direction = glm::normalize(uvToUnitVector(glm::vec2(u, v)));
    return EnvSample{direction,
                     glm::vec2{u, v},
                     envMapPdf(enviromentSettings.pdf[index], v, enviromentSettings.height)};
}

__device__ glm::vec3 envColor(const EnvSample& env_sample,
                              const glm::vec3& hitpoint,
                              const CudaTriangle* triangles,
                              const CudaLinearBvhNode* linearBVH,
                              const CudaEnviromentSettings& enviromentSettings)
{
    const float u = env_sample.uv.x;
    const float v = env_sample.uv.y;
    if (u >= 0.0f && u <= 1.0f && v >= 0.0f && v <= 1.0f)
    {
        CudaRay env_ray;
        env_ray.mint = 0;
        env_ray.maxt = FLT_MAX;
        env_ray.origin = hitpoint;
        env_ray.direction = env_sample.direction;

        CudaIntersection intersection;
        if (!intersectBVH(triangles, linearBVH, env_ray, intersection))
        {
            const float4 env = tex2D<float4>(enviromentSettings.texObj, u, v);
            return glm::vec3(env.x, env.y, env.z);
        }
    }
    else
    {
        return glm::vec3(0, 1, 0);
    }

    return glm::vec3(0, 0, 0);
}

__device__ float misBalanceHeuristic(const float a, const float b)
{
    return a / (a + b);
}

__device__ float misPowerHeuristic(const float af, const float a, const float bf, const float b)
{
    const float as = af * a;
    const float bs = bf * b;
    return (as * as) / (as * as + bs * bs);
}

__global__ void traceKernel(CudaPathIteration* pathIterationBuffer,
                            glm::vec4* accuBuffer,
                            const dim3 texDim,
                            const CudaCamera cam,
                            const CudaTriangle* triangles,
                            const unsigned int nTriangles,
                            const CudaMaterial* materials,
                            const unsigned int nMaterials,
                            const unsigned int seed,
                            const CudaLinearBvhNode* linearBVH,
                            const CudaEnviromentSettings enviromentSettings)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= texDim.x || y >= texDim.y)
    {
        return;
    }

    const int threadId =
        (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    curandState randState;
    curand_init(seed + threadId, 0, 0, &randState);

    const int pos = texDim.x * y + x;
    CudaPathIteration pathIteration = pathIterationBuffer[pos];
    CudaRay ray;

    if (pathIteration.bounce == 0)
    {
        ray = rayDirectionWithOffset(x, y, cam, curand_uniform(&randState), curand_uniform(&randState));
        pathIteration.mask = glm::vec3{1.0f, 1.0f, 1.0f};
        pathIteration.color = glm::vec3{0.0f, 0.0f, 0.0f};
    }
    else
    {
        ray.mint = 0;
        ray.maxt = FLT_MAX;
        ray.direction = pathIteration.rayDir;
        ray.origin = pathIteration.rayOrigin;
    }

    CudaIntersection intersection;
    if (!intersectBVH(triangles, linearBVH, ray, intersection))
    {
        if (pathIteration.bounce == 0)
        {
            const glm::vec2 uv = unitVectorToUv(ray.direction);
            const float4 env = tex2D<float4>(enviromentSettings.texObj, uv.x, uv.y);
            pathIteration.color = glm::vec3(env.x, env.y, env.z);
        }
        finishSample(pos, accuBuffer, &pathIteration);
    }
    else
    {
        const glm::vec3 hitpointNormal = glm::normalize(intersection.n);
        const glm::vec3 hitpoint = intersection.p + hitpointNormal * 0.00001f;
        const CudaMaterial& material = materials[intersection.material];
        const glm::vec3 wo = -ray.direction;

        glm::vec3 direct_light{0.0f, 0.0f, 0.0f};

        // ---------------- BRDF Sample ----------------
        //
        const SampleDirection brdf_sample = uniformSampleCosineWeightedHemisphere(curand_uniform(&randState),
                                                                                  curand_uniform(&randState),
                                                                                  hitpointNormal,
                                                                                  wo);

        const float brdf_cos_theta = glm::dot(brdf_sample.direction, hitpointNormal);
        if (brdf_sample.pdf > 0.0f && brdf_cos_theta > 0.0f)
        {
            CudaRay brdf_ray;
            brdf_ray.mint = 0;
            brdf_ray.maxt = FLT_MAX;
            brdf_ray.origin = hitpoint;
            brdf_ray.direction = brdf_sample.direction;
            if (!intersectBVH(triangles, linearBVH, brdf_ray))
            {
                const glm::vec2 uv = unitVectorToUv(brdf_sample.direction);
                const float4 env = tex2D<float4>(enviromentSettings.texObj, uv.x, uv.y);
                const glm::vec3 li{env.x, env.y, env.z};
                const float env_pdf = envMapPdf(brdf_sample.direction, enviromentSettings);
                const float brdf_weight = misPowerHeuristic(1.0f, brdf_sample.pdf, 1.0f, env_pdf);
                direct_light += (material.f() * li * brdf_cos_theta * brdf_weight) / brdf_sample.pdf;
            }
        }

        // ---------------- Env sample  ----------------
        //
        const EnvSample env_sample = sampleEnv(randState, enviromentSettings);

        const float env_cos_theta = glm::dot(env_sample.direction, hitpointNormal);
        if (env_sample.pdf > 0.0f && env_cos_theta > 0.0f)
        {
            CudaRay env_ray;
            env_ray.mint = 0;
            env_ray.maxt = FLT_MAX;
            env_ray.origin = hitpoint;
            env_ray.direction = env_sample.direction;
            if (!intersectBVH(triangles, linearBVH, env_ray))
            {
                const float brdf_pdf =
                    uniformSampleCosineWeightedHemispherePdf(hitpointNormal, wo, env_sample.direction);
                const float env_weight = misPowerHeuristic(1.0f, env_sample.pdf, 1.0f, brdf_pdf);
                const float4 env = tex2D<float4>(enviromentSettings.texObj, env_sample.uv.x, env_sample.uv.y);
                const glm::vec3 li{env.x, env.y, env.z};
                direct_light += (material.f() * li * env_cos_theta * env_weight) / env_sample.pdf;
            }
        }

        pathIteration.color += pathIteration.mask * direct_light;
        pathIteration.mask *= (material.f() * brdf_cos_theta) / brdf_sample.pdf;
        pathIteration.rayDir = brdf_sample.direction;
        pathIteration.rayOrigin = hitpoint;
        pathIteration.bounce++;

        if (pathIteration.bounce == MAX_PATH_DEPTH || brdf_sample.pdf <= 0.0f || brdf_cos_theta <= 0.0f)
        {
            finishSample(pos, accuBuffer, &pathIteration);
        }
    }

    if (pathIteration.samples == 0)
    {
        accuBuffer[pos] = glm::vec4(pathIteration.color.x, pathIteration.color.y, pathIteration.color.z, 1.0f);
    }

    pathIterationBuffer[pos] = pathIteration;
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