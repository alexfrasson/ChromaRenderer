#include "chroma-renderer/core/renderer/CudaPathTracer.h"
#include "chroma-renderer/core/renderer/CudaPathTracerKernel.h"
#include "chroma-renderer/core/space-partition/BVH.h"
#include "chroma-renderer/core/types/environment_map.h"
#include "chroma-renderer/core/utility/Stopwatch.h"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glad/glad.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <thread>

#define SAFE_CUDA_FREE(x)        \
    cudaErrorCheck(cudaFree(x)); \
    x = nullptr;

#define SAFE_CUDA_FREE_ARRAY(x)       \
    cudaErrorCheck(cudaFreeArray(x)); \
    x = nullptr;

class CudaPathTracer::Impl
{
  public:
    Impl();
    ~Impl();
    Impl(const Impl&) = delete;
    Impl(Impl&&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl& operator=(Impl&&) = delete;

    void render();
    void reset();

    void setSceneGeometry(const ISpacePartitioningStructure* sps, const std::vector<Material>& materials);
    void setEnvMap(const float* hdriEnvData, const int hdriEnvWidth, const int hdriEnvHeight, const int channels);
    void setTargetImage(const Image& img);
    void setCamera(const Camera& cam);
    void setMaterials(const std::vector<Material>& materials, const bool& sync);
    void setSettings(const RendererSettings& settings);

    float getProgress() const;
    std::uint32_t getFinishedSamples() const;
    std::uint32_t getTargetSamplesPerPixel() const;
    float getInstantRaysPerSec() const;

    void copyFrameToTexture();

    struct RegisteredImage
    {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        GLuint texID = 0;
        cudaGraphicsResource* cudaTextureResource{nullptr};

        bool changed(const Image& img);
    };

    int iteration = 0;

    std::uint32_t targetSamplesPerPixel;

    float gammaCorrectionScale = 1.0f;

    RegisteredImage registeredImage;

    cudaStream_t stream;
    cudaStream_t cpystream;

    CudaLinearBvhNode* dev_cudaLinearBVHNodes{nullptr};
    unsigned int nCudaLinearBVHNodes = 0;

    CudaTriangle* dev_cudaTrianglesBVH{nullptr};
    unsigned int nCudaTrianglesBVH = 0;

    CudaMaterial* dev_cudaMaterials{nullptr};
    unsigned int nCudaMaterials = 0;

    CudaCamera cuda_cam;
    CudaEnviromentSettings enviromentSettings;

    cudaArray* envArray{nullptr};

    glm::vec4* dev_accuBuffer{nullptr};
    CudaPathIteration* dev_pathIterationBuffer{nullptr};

    Stopwatch stopwatch;
    std::chrono::milliseconds lastIterationElapsedMillis;
};

CudaPathTracer::CudaPathTracer() : impl_{std::make_unique<CudaPathTracer::Impl>()}
{
}

CudaPathTracer::~CudaPathTracer() = default;

void CudaPathTracer::setEnvMap(const float* hdriEnvData,
                               const int hdriEnvWidth,
                               const int hdriEnvHeight,
                               const int channels)
{
    impl_->setEnvMap(hdriEnvData, hdriEnvWidth, hdriEnvHeight, channels);
}

void CudaPathTracer::setSceneGeometry(const ISpacePartitioningStructure* sps, const std::vector<Material>& materials)
{
    impl_->setSceneGeometry(sps, materials);
}

void CudaPathTracer::setTargetImage(const Image& img)
{
    impl_->setTargetImage(img);
}

void CudaPathTracer::setCamera(const Camera& cam)
{
    impl_->setCamera(cam);
}

void CudaPathTracer::setMaterials(const std::vector<Material>& materials)
{
    impl_->setMaterials(materials, true);
}

void CudaPathTracer::render()
{
    impl_->render();
}

void CudaPathTracer::setSettings(const RendererSettings& settings)
{
    impl_->setSettings(settings);
}

float CudaPathTracer::getProgress() const
{
    return impl_->getProgress();
}

std::uint32_t CudaPathTracer::getFinishedSamples() const
{
    return impl_->getFinishedSamples();
}

std::uint32_t CudaPathTracer::getTargetSamplesPerPixel() const
{
    return impl_->getTargetSamplesPerPixel();
}

float CudaPathTracer::getInstantRaysPerSec() const
{
    return impl_->getInstantRaysPerSec();
}

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n", (int)devProp.major);
    printf("Minor revision number:         %d\n", (int)devProp.minor);
    printf("Name:                          %s\n", devProp.name);
    printf("Total global memory:           %d\n", (int)devProp.totalGlobalMem);
    printf("Total shared memory per block: %d\n", (int)devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n", (int)devProp.regsPerBlock);
    printf("Warp size:                     %d\n", (int)devProp.warpSize);
    printf("Maximum memory pitch:          %d\n", (int)devProp.memPitch);
    printf("Maximum threads per block:     %d\n", (int)devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, (int)devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, (int)devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n", (int)devProp.clockRate);
    printf("Total constant memory:         %d\n", (int)devProp.totalConstMem);
    printf("Texture alignment:             %d\n", (int)devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n", (int)devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}

CudaCamera CameraToCudaCamera(Camera cam)
{
    CudaCamera cudaCam;
    cudaCam.width = cam.width;
    cudaCam.height = cam.height;
    cudaCam.d = cam.d;
    cudaCam.right.x = cam.right.x;
    cudaCam.right.y = cam.right.y;
    cudaCam.right.z = cam.right.z;
    cudaCam.up.x = cam.up.x;
    cudaCam.up.y = cam.up.y;
    cudaCam.up.z = cam.up.z;
    cudaCam.forward.x = cam.forward.x;
    cudaCam.forward.y = cam.forward.y;
    cudaCam.forward.z = cam.forward.z;
    cudaCam.eye.x = cam.eye.x;
    cudaCam.eye.y = cam.eye.y;
    cudaCam.eye.z = cam.eye.z;
    return cudaCam;
}

std::vector<CudaMaterial> SceneToCudaMaterials(const std::vector<Material>& materials)
{
    std::vector<CudaMaterial> cudaMaterials;

    for (const auto& m : materials)
    {
        CudaMaterial cm;
        cm.kd.x = m.kd.r;
        cm.kd.y = m.kd.g;
        cm.kd.z = m.kd.b;
        cm.ke.x = m.ke.r;
        cm.ke.y = m.ke.g;
        cm.ke.z = m.ke.b;
        cm.transparent.x = m.transparent.r;
        cm.transparent.y = m.transparent.g;
        cm.transparent.z = m.transparent.b;
        cudaMaterials.push_back(cm);
    }

    return cudaMaterials;
}

std::vector<CudaLinearBvhNode> SceneToCudaLinearBvhNode(const ISpacePartitioningStructure* sps)
{
    // Lets assume this is a bvh :)
    const BVH* bvh = (const BVH*)sps;

    std::vector<CudaLinearBvhNode> cudaLinearBVH;
    cudaLinearBVH.reserve(bvh->nNodes);

    for (unsigned int i = 0; i < bvh->nNodes; i++)
    {
        CudaLinearBvhNode n;
        n.axis = bvh->lroot[i].axis;
        n.nPrimitives = bvh->lroot[i].nPrimitives;
        n.primitivesOffset = bvh->lroot[i].primitivesOffset;
        n.bbox.max.x = bvh->lroot[i].bbox.max.x;
        n.bbox.max.y = bvh->lroot[i].bbox.max.y;
        n.bbox.max.z = bvh->lroot[i].bbox.max.z;
        n.bbox.min.x = bvh->lroot[i].bbox.min.x;
        n.bbox.min.y = bvh->lroot[i].bbox.min.y;
        n.bbox.min.z = bvh->lroot[i].bbox.min.z;

        cudaLinearBVH.push_back(n);
    }

    return cudaLinearBVH;
}

std::vector<CudaTriangle> SceneToCudaTrianglesBVH(const ISpacePartitioningStructure* sps,
                                                  const std::vector<Material>& materials)
{
    // Lets assume this is a bvh :)
    const BVH* bvh = (const BVH*)sps;

    std::vector<CudaTriangle> cudaTriangles;
    cudaTriangles.reserve(bvh->triangles.size());

    for (Triangle* t : bvh->triangles)
    {
        CudaTriangle ct;

        // Copy vertices and normals
        for (int i = 0; i < 3; i++)
        {
            ct.v[i].x = t->getVertex(i)->x;
            ct.v[i].y = t->getVertex(i)->y;
            ct.v[i].z = t->getVertex(i)->z;

            ct.n[i].x = t->getNormal(i)->x;
            ct.n[i].y = t->getNormal(i)->y;
            ct.n[i].z = t->getNormal(i)->z;
        }

        // Find material index
        for (size_t i = 0; i < materials.size(); i++)
        {
            if (t->material == &materials[i])
            {
                ct.material = (int)i;
                break;
            }
        }

        cudaTriangles.push_back(ct);
    }

    return cudaTriangles;
}

// this hash function calculates a new random number generator seed for each frame, based on framenumber
unsigned int WangHash(unsigned int a)
{
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}

CudaPathTracer::Impl::Impl()
{
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaErrorCheck(cudaSetDevice(0));
    cudaErrorCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaErrorCheck(cudaStreamCreateWithFlags(&cpystream, cudaStreamNonBlocking));

    // Number of CUDA devices
    int devCount;
    cudaErrorCheck(cudaGetDeviceCount(&devCount));
    std::cout << "CUDA Device Query..." << std::endl;
    std::cout << "There are " << devCount << " CUDA devices." << std::endl;

    // Iterate through devices
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        std::cout << std::endl << "CUDA Device " << i << std::endl;
        cudaDeviceProp devProp;
        cudaErrorCheck(cudaGetDeviceProperties(&devProp, i));
        printDevProp(devProp);
        std::cout << std::endl << std::endl;
    }
}

CudaPathTracer::Impl::~Impl()
{
    cudaErrorCheck(cudaDeviceSynchronize());

    if (registeredImage.cudaTextureResource != nullptr)
    {
        cudaErrorCheck(cudaGraphicsUnregisterResource(registeredImage.cudaTextureResource));
    }

    SAFE_CUDA_FREE(dev_cudaTrianglesBVH);
    SAFE_CUDA_FREE(dev_cudaLinearBVHNodes);
    SAFE_CUDA_FREE(dev_cudaMaterials);
    SAFE_CUDA_FREE(dev_accuBuffer);
    SAFE_CUDA_FREE(dev_pathIterationBuffer);

    cudaErrorCheck(cudaDestroyTextureObject(enviromentSettings.texObj));

    SAFE_CUDA_FREE_ARRAY(envArray);
    SAFE_CUDA_FREE(enviromentSettings.cdf);
    SAFE_CUDA_FREE(enviromentSettings.pdf);

    cudaErrorCheck(cudaStreamDestroy(stream));
    cudaErrorCheck(cudaStreamDestroy(cpystream));

    cudaErrorCheck(cudaDeviceReset());
}

void CudaPathTracer::Impl::render()
{
    const bool firstIteration = iteration == 0;

    cudaErrorCheck(cudaGetLastError());

    if (firstIteration)
    {
        cudaErrorCheck(cudaStreamSynchronize(stream));
    }
    else
    {
        cudaError err = cudaStreamQuery(stream);

        if (err == cudaErrorNotReady)
            return;
        else if (err == cudaSuccess)
        {
            stopwatch.stop();
            lastIterationElapsedMillis = stopwatch.elapsedMillis;
            stopwatch.start();
            copyFrameToTexture();
        }
        else
            cudaErrorCheck(err);
    }

    // calculate a new seed for the random number generator, based on the framenumber
    unsigned int hashedframes = WangHash(iteration);

    trace(stream,
          dev_pathIterationBuffer,
          dev_accuBuffer,
          dim3(registeredImage.width, registeredImage.height),
          cuda_cam,
          dev_cudaTrianglesBVH,
          nCudaTrianglesBVH,
          dev_cudaMaterials,
          nCudaMaterials,
          hashedframes,
          dev_cudaLinearBVHNodes,
          enviromentSettings);

    cudaErrorCheck(cudaGetLastError());

    iteration++;

    if (firstIteration)
    {
        cudaErrorCheck(cudaStreamSynchronize(stream));

        copyFrameToTexture();
    }
}

void CudaPathTracer::Impl::reset()
{
    iteration = 0;
}

void CudaPathTracer::Impl::setEnvMap(const float* hdriEnvData,
                                     const int hdriEnvWidth,
                                     const int hdriEnvHeight,
                                     const int channels)
{
    cudaErrorCheck(cudaDestroyTextureObject(enviromentSettings.texObj));
    SAFE_CUDA_FREE_ARRAY(envArray);
    SAFE_CUDA_FREE(enviromentSettings.cdf);
    SAFE_CUDA_FREE(enviromentSettings.pdf);

    unsigned int size = hdriEnvWidth * hdriEnvHeight * channels * sizeof(float);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

    cudaErrorCheck(cudaMallocArray(&envArray, &channelDesc, hdriEnvWidth, hdriEnvHeight));
    cudaErrorCheck(cudaMemcpyToArray(envArray, 0, 0, hdriEnvData, size, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = envArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaErrorCheck(cudaCreateTextureObject(&enviromentSettings.texObj, &resDesc, &texDesc, NULL));

    EnvironmentMap env_map{hdriEnvData, static_cast<uint32_t>(hdriEnvWidth), static_cast<uint32_t>(hdriEnvHeight)};

    const auto& cdf = env_map.GetDistribution().GetCdf();
    enviromentSettings.cdf_size = static_cast<int>(cdf.size());
    cudaErrorCheck(cudaMalloc((void**)&enviromentSettings.cdf, enviromentSettings.cdf_size * sizeof(float)));
    cudaErrorCheck(cudaMemcpyAsync(enviromentSettings.cdf,
                                   &cdf[0],
                                   enviromentSettings.cdf_size * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   stream));

    const auto& pdf = env_map.GetPdf();
    enviromentSettings.pdf_size = static_cast<int>(pdf.size());
    cudaErrorCheck(cudaMalloc((void**)&enviromentSettings.pdf, enviromentSettings.pdf_size * sizeof(float)));
    cudaErrorCheck(cudaMemcpyAsync(enviromentSettings.pdf,
                                   &pdf[0],
                                   enviromentSettings.pdf_size * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   stream));

    cudaErrorCheck(cudaStreamSynchronize(stream));

    enviromentSettings.width = hdriEnvWidth;
    enviromentSettings.height = hdriEnvHeight;

    reset();
}

void CudaPathTracer::Impl::setSceneGeometry(const ISpacePartitioningStructure* sps,
                                            const std::vector<Material>& materials)
{
    SAFE_CUDA_FREE(dev_cudaTrianglesBVH);
    SAFE_CUDA_FREE(dev_cudaLinearBVHNodes);

    size_t free, total;

    cudaErrorCheck(cudaMemGetInfo(&free, &total));
    std::cout << "Free memory: " << free / (1024 * 1024) << "MB" << std::endl;
    std::cout << "Total memory: " << total / (1024 * 1024) << "MB" << std::endl;

    std::vector<CudaTriangle> cudaTrianglesBVH = SceneToCudaTrianglesBVH(sps, materials);
    nCudaTrianglesBVH = (int)cudaTrianglesBVH.size();
    std::cout << "Triangles BVH: " << cudaTrianglesBVH.size() << std::endl;
    std::cout << "Triangles BVH size: " << (cudaTrianglesBVH.size() * sizeof(CudaTriangle)) / (1024) << "KB"
              << std::endl;
    cudaErrorCheck(cudaMalloc((void**)&dev_cudaTrianglesBVH, cudaTrianglesBVH.size() * sizeof(CudaTriangle)));
    cudaErrorCheck(cudaMemcpyAsync(dev_cudaTrianglesBVH,
                                   &cudaTrianglesBVH[0],
                                   cudaTrianglesBVH.size() * sizeof(CudaTriangle),
                                   cudaMemcpyHostToDevice,
                                   stream));

    std::vector<CudaLinearBvhNode> cudaLinearBVH = SceneToCudaLinearBvhNode(sps);
    nCudaLinearBVHNodes = (int)cudaLinearBVH.size();
    std::cout << "CudaLinearBVHNodes: " << cudaLinearBVH.size() << std::endl;
    std::cout << "CudaLinearBVHNodes size: " << (cudaLinearBVH.size() * sizeof(CudaLinearBvhNode)) / (1024) << "KB"
              << std::endl;
    cudaErrorCheck(cudaMalloc((void**)&dev_cudaLinearBVHNodes, cudaLinearBVH.size() * sizeof(CudaLinearBvhNode)));
    cudaErrorCheck(cudaMemcpyAsync(dev_cudaLinearBVHNodes,
                                   &cudaLinearBVH[0],
                                   cudaLinearBVH.size() * sizeof(CudaLinearBvhNode),
                                   cudaMemcpyHostToDevice,
                                   stream));

    setMaterials(materials, false);

    cudaErrorCheck(cudaGetLastError());
    cudaErrorCheck(cudaStreamSynchronize(stream));

    reset();
}

void CudaPathTracer::Impl::setTargetImage(const Image& img)
{
    assert(img.textureID > 0);
    assert(img.getWidth() > 0 && img.getHeight() > 0);

    if (registeredImage.changed(img))
    {
        SAFE_CUDA_FREE(dev_accuBuffer);
        SAFE_CUDA_FREE(dev_pathIterationBuffer);

        cudaErrorCheck(cudaMalloc((void**)&dev_accuBuffer, img.getWidth() * img.getHeight() * sizeof(glm::vec4)));
        cudaErrorCheck(
            cudaMalloc((void**)&dev_pathIterationBuffer, img.getWidth() * img.getHeight() * sizeof(CudaPathIteration)));

        cudaErrorCheck(cudaDeviceSynchronize());

        if (registeredImage.cudaTextureResource != nullptr)
        {
            cudaErrorCheck(cudaGraphicsUnregisterResource(registeredImage.cudaTextureResource));
        }

        // Only call Cuda/OpenGL interop stuff from within the OpenGL context thread!
        cudaErrorCheck(cudaGraphicsGLRegisterImage(&registeredImage.cudaTextureResource,
                                                   img.textureID,
                                                   GL_TEXTURE_2D,
                                                   cudaGraphicsRegisterFlagsWriteDiscard));

        registeredImage.width = img.getWidth();
        registeredImage.height = img.getHeight();
        registeredImage.texID = img.textureID;
    }

    cudaErrorCheck(
        cudaMemsetAsync(dev_accuBuffer, 0, registeredImage.width * registeredImage.height * sizeof(glm::vec4), stream));
    cudaErrorCheck(cudaMemsetAsync(dev_pathIterationBuffer,
                                   0,
                                   registeredImage.width * registeredImage.height * sizeof(CudaPathIteration),
                                   stream));

    cudaErrorCheck(cudaGetLastError());
    cudaErrorCheck(cudaStreamSynchronize(stream));

    reset();
}

void CudaPathTracer::Impl::setCamera(const Camera& cam)
{
    cuda_cam = CameraToCudaCamera(cam);
    reset();
}

void CudaPathTracer::Impl::setMaterials(const std::vector<Material>& materials, const bool& sync)
{
    SAFE_CUDA_FREE(dev_cudaMaterials);

    const std::vector<CudaMaterial> cudaMaterials = SceneToCudaMaterials(materials);
    const std::size_t size_in_bytes = cudaMaterials.size() * sizeof(CudaMaterial);
    nCudaMaterials = static_cast<std::uint32_t>(cudaMaterials.size());

    cudaErrorCheck(cudaMalloc((void**)&dev_cudaMaterials, size_in_bytes));
    cudaErrorCheck(
        cudaMemcpyAsync(dev_cudaMaterials, &cudaMaterials[0], size_in_bytes, cudaMemcpyHostToDevice, stream));

    if (sync)
    {
        cudaErrorCheck(cudaStreamSynchronize(stream));
    }

    reset();
}

void CudaPathTracer::Impl::setSettings(const RendererSettings& settings)
{
    targetSamplesPerPixel = settings.samplesperpixel;
}

void CudaPathTracer::Impl::copyFrameToTexture()
{
    cudaArray* aarray;

    cudaErrorCheck(cudaGraphicsMapResources(1, &registeredImage.cudaTextureResource, cpystream));
    cudaErrorCheck(cudaGraphicsSubResourceGetMappedArray(&aarray, registeredImage.cudaTextureResource, 0, 0));

    cudaErrorCheck(cudaMemcpyToArrayAsync(aarray,
                                          0,
                                          0,
                                          dev_accuBuffer,
                                          registeredImage.width * registeredImage.height * sizeof(glm::vec4),
                                          cudaMemcpyDeviceToDevice,
                                          cpystream));

    cudaErrorCheck(cudaGraphicsUnmapResources(1, &registeredImage.cudaTextureResource, cpystream));
}

float CudaPathTracer::Impl::getProgress() const
{
    return ((float)iteration / (float)targetSamplesPerPixel);
}

float CudaPathTracer::Impl::getInstantRaysPerSec() const
{
    return (registeredImage.width * (float)registeredImage.height) / (lastIterationElapsedMillis.count() * 0.001f);
}

std::uint32_t CudaPathTracer::Impl::getFinishedSamples() const
{
    return iteration;
}

std::uint32_t CudaPathTracer::Impl::getTargetSamplesPerPixel() const
{
    return targetSamplesPerPixel;
}

bool CudaPathTracer::Impl::RegisteredImage::changed(const Image& img)
{
    return img.getWidth() != width || img.getHeight() != height || img.textureID != texID;
}