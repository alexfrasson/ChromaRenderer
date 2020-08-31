#include "chroma-renderer/core/renderer/cuda_path_tracer.h"
#include "chroma-renderer/core/renderer/cuda_path_tracer_kernel.h"
#include "chroma-renderer/core/space-partition/bvh.h"
#include "chroma-renderer/core/types/environment_map.h"
#include "chroma-renderer/core/utility/stopwatch.h"

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

template <typename T>
constexpr inline void safeCudaFree(T& x)
{
    cudaErrorCheck(cudaFree(x));
    x = nullptr;
}

template <typename T>
constexpr inline void safeCudaFreeArray(T& x)
{
    cudaErrorCheck(cudaFreeArray(x));
    x = nullptr;
}

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
    void setEnvMap(const float* hdri_env_data,
                   std::size_t hdri_env_width,
                   std::size_t hdri_env_height,
                   std::size_t channels);
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
        std::uint32_t width{0};
        std::uint32_t height{0};
        GLuint tex_id{0};
        cudaGraphicsResource* cuda_texture_resource{nullptr};

        bool changed(const Image& img) const;
    };

    std::uint32_t iteration{0};
    std::uint32_t target_samples_per_pixel{0};

    float gamma_correction_scale{1.0f};

    RegisteredImage registered_image{};

    cudaStream_t stream{};
    cudaStream_t cpystream{};

    CudaLinearBvhNode* dev_cuda_linear_bvh_nodes{nullptr};
    std::size_t n_cuda_linear_bvh_nodes{0};

    CudaTriangle* dev_cuda_triangles_bvh{nullptr};
    std::size_t n_cuda_triangles_bvh{0};

    CudaMaterial* dev_cuda_materials{nullptr};
    std::size_t n_cuda_materials{0};

    CudaCamera cuda_cam{};
    CudaEnviromentSettings enviroment_settings;

    cudaArray* env_array{nullptr};

    glm::vec4* dev_accu_buffer{nullptr};
    CudaPathIteration* dev_path_iteration_buffer{nullptr};

    Stopwatch stopwatch{};
    std::chrono::milliseconds last_iteration_elapsed_millis{0};
};

CudaPathTracer::CudaPathTracer() : impl_{std::make_unique<CudaPathTracer::Impl>()}
{
}

CudaPathTracer::~CudaPathTracer() = default;

void CudaPathTracer::setEnvMap(const float* hdri_env_data,
                               const std::size_t hdri_env_width,
                               const std::size_t hdri_env_height,
                               const std::size_t channels)
{
    impl_->setEnvMap(hdri_env_data, hdri_env_width, hdri_env_height, channels);
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
void printDevProp(cudaDeviceProp dev_prop)
{
    std::cout << "Major revision number:         " << (int)dev_prop.major << std::endl;
    std::cout << "Minor revision number:         " << (int)dev_prop.minor << std::endl;
    std::cout << "Name:                          " << dev_prop.name << std::endl; // NOLINT
    std::cout << "Total global memory:           " << (int)dev_prop.totalGlobalMem << std::endl;
    std::cout << "Total shared memory per block: " << (int)dev_prop.sharedMemPerBlock << std::endl;
    std::cout << "Total registers per block:     " << (int)dev_prop.regsPerBlock << std::endl;
    std::cout << "Warp size:                     " << (int)dev_prop.warpSize << std::endl;
    std::cout << "Maximum memory pitch:          " << (int)dev_prop.memPitch << std::endl;
    std::cout << "Maximum threads per block:     " << (int)dev_prop.maxThreadsPerBlock << std::endl;
    for (int i = 0; i < 3; ++i)
    {
        std::cout << "Maximum dimension " << i << " of block:  " << (int)dev_prop.maxThreadsDim[i] << std::endl;
    }
    for (int i = 0; i < 3; ++i)
    {
        std::cout << "Maximum dimension " << i << " of grid:   " << (int)dev_prop.maxGridSize[i] << std::endl;
    }
    std::cout << "Clock rate:                    " << (int)dev_prop.clockRate << std::endl;
    std::cout << "Total constant memory:         " << (int)dev_prop.totalConstMem << std::endl;
    std::cout << "Texture alignment:             " << (int)dev_prop.textureAlignment << std::endl;
    std::cout << "Concurrent copy and execution: " << (dev_prop.deviceOverlap == 1 ? "Yes" : "No") << std::endl;
    std::cout << "Number of multiprocessors:     " << (int)dev_prop.multiProcessorCount << std::endl;
    std::cout << "Kernel execution timeout:      " << (dev_prop.kernelExecTimeoutEnabled == 1 ? "Yes" : "No")
              << std::endl;
}

CudaCamera cameraToCudaCamera(Camera cam)
{
    CudaCamera cuda_cam{};
    cuda_cam.width = cam.width;
    cuda_cam.height = cam.height;
    cuda_cam.d = cam.d;
    cuda_cam.right.x = cam.right.x;
    cuda_cam.right.y = cam.right.y;
    cuda_cam.right.z = cam.right.z;
    cuda_cam.up.x = cam.up.x;
    cuda_cam.up.y = cam.up.y;
    cuda_cam.up.z = cam.up.z;
    cuda_cam.forward.x = cam.forward.x;
    cuda_cam.forward.y = cam.forward.y;
    cuda_cam.forward.z = cam.forward.z;
    cuda_cam.eye.x = cam.eye.x;
    cuda_cam.eye.y = cam.eye.y;
    cuda_cam.eye.z = cam.eye.z;
    return cuda_cam;
}

std::vector<CudaMaterial> sceneToCudaMaterials(const std::vector<Material>& materials)
{
    std::vector<CudaMaterial> cuda_materials;

    for (const auto& m : materials)
    {
        CudaMaterial cm{};
        cm.kd.x = m.kd.r;
        cm.kd.y = m.kd.g;
        cm.kd.z = m.kd.b;
        cm.ke.x = m.ke.r;
        cm.ke.y = m.ke.g;
        cm.ke.z = m.ke.b;
        cm.transparent.x = m.transparent.r;
        cm.transparent.y = m.transparent.g;
        cm.transparent.z = m.transparent.b;
        cuda_materials.push_back(cm);
    }

    return cuda_materials;
}

std::vector<CudaLinearBvhNode> sceneToCudaLinearBvhNode(const ISpacePartitioningStructure* sps)
{
    // Lets assume this is a bvh :)
    const BVH* bvh = dynamic_cast<const BVH*>(sps);

    std::vector<CudaLinearBvhNode> cuda_linear_bvh;
    cuda_linear_bvh.reserve(bvh->n_nodes);

    for (unsigned int i = 0; i < bvh->n_nodes; i++)
    {
        CudaLinearBvhNode n;
        n.axis = bvh->lroot[i].axis;
        n.n_primitives = bvh->lroot[i].n_primitives;
        n.primitives_offset = bvh->lroot[i].primitives_offset;
        n.bbox.max.x = bvh->lroot[i].bbox.max.x;
        n.bbox.max.y = bvh->lroot[i].bbox.max.y;
        n.bbox.max.z = bvh->lroot[i].bbox.max.z;
        n.bbox.min.x = bvh->lroot[i].bbox.min.x;
        n.bbox.min.y = bvh->lroot[i].bbox.min.y;
        n.bbox.min.z = bvh->lroot[i].bbox.min.z;

        cuda_linear_bvh.push_back(n);
    }

    return cuda_linear_bvh;
}

std::vector<CudaTriangle> sceneToCudaTrianglesBvh(const ISpacePartitioningStructure* sps,
                                                  const std::vector<Material>& materials)
{
    // Lets assume this is a bvh :)
    const BVH* bvh = dynamic_cast<const BVH*>(sps);

    std::vector<CudaTriangle> cuda_triangles;
    cuda_triangles.reserve(bvh->triangles.size());

    for (Triangle* t : bvh->triangles)
    {
        CudaTriangle ct{};

        // Copy vertices and normals
        for (std::size_t i = 0; i < 3; i++)
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

        cuda_triangles.push_back(ct);
    }

    return cuda_triangles;
}

// this hash function calculates a new random number generator seed for each frame, based on framenumber
unsigned int wangHash(unsigned int a)
{
    a = (a ^ 61u) ^ (a >> 16u);
    a = a + (a << 3u);
    a = a ^ (a >> 4u);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15u);
    return a;
}

CudaPathTracer::Impl::Impl()
{
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaErrorCheck(cudaSetDevice(0));
    cudaErrorCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaErrorCheck(cudaStreamCreateWithFlags(&cpystream, cudaStreamNonBlocking));

    // Number of CUDA devices
    int dev_count{0};
    cudaErrorCheck(cudaGetDeviceCount(&dev_count));
    std::cout << "CUDA Device Query..." << std::endl;
    std::cout << "There are " << dev_count << " CUDA devices." << std::endl;

    // Iterate through devices
    for (int i = 0; i < dev_count; ++i)
    {
        // Get device properties
        std::cout << std::endl << "CUDA Device " << i << std::endl;
        cudaDeviceProp dev_prop{};
        cudaErrorCheck(cudaGetDeviceProperties(&dev_prop, i));
        printDevProp(dev_prop);
        std::cout << std::endl << std::endl;
    }
}

CudaPathTracer::Impl::~Impl()
{
    cudaErrorCheck(cudaDeviceSynchronize());

    if (registered_image.cuda_texture_resource != nullptr)
    {
        cudaErrorCheck(cudaGraphicsUnregisterResource(registered_image.cuda_texture_resource));
    }

    safeCudaFree(dev_cuda_triangles_bvh);
    safeCudaFree(dev_cuda_linear_bvh_nodes);
    safeCudaFree(dev_cuda_materials);
    safeCudaFree(dev_accu_buffer);
    safeCudaFree(dev_path_iteration_buffer);

    cudaErrorCheck(cudaDestroyTextureObject(enviroment_settings.tex_obj));

    safeCudaFreeArray(env_array);
    safeCudaFree(enviroment_settings.cdf);
    safeCudaFree(enviroment_settings.pdf);

    cudaErrorCheck(cudaStreamDestroy(stream));
    cudaErrorCheck(cudaStreamDestroy(cpystream));

    cudaErrorCheck(cudaDeviceReset());
}

void CudaPathTracer::Impl::render()
{
    const bool first_iteration = iteration == 0;

    cudaErrorCheck(cudaGetLastError());

    if (first_iteration)
    {
        cudaErrorCheck(cudaStreamSynchronize(stream));
    }
    else
    {
        cudaError err = cudaStreamQuery(stream);

        if (err == cudaSuccess)
        {
            stopwatch.stop();
            last_iteration_elapsed_millis = stopwatch.elapsed_millis;
            stopwatch.start();
            copyFrameToTexture();
        }
        else if (err == cudaErrorNotReady)
        {
            return;
        }
        else
        {
            cudaErrorCheck(err);
        }
    }

    // calculate a new seed for the random number generator, based on the framenumber
    unsigned int hashedframes = wangHash(iteration);

    trace(stream,
          dev_path_iteration_buffer,
          dev_accu_buffer,
          dim3(registered_image.width, registered_image.height),
          cuda_cam,
          dev_cuda_triangles_bvh,
          dev_cuda_materials,
          hashedframes,
          dev_cuda_linear_bvh_nodes,
          enviroment_settings);

    cudaErrorCheck(cudaGetLastError());

    iteration++;

    if (first_iteration)
    {
        cudaErrorCheck(cudaStreamSynchronize(stream));

        copyFrameToTexture();
    }
}

void CudaPathTracer::Impl::reset()
{
    iteration = 0;
}

void CudaPathTracer::Impl::setEnvMap(const float* hdri_env_data,
                                     const std::size_t hdri_env_width,
                                     const std::size_t hdri_env_height,
                                     const std::size_t channels)
{
    cudaErrorCheck(cudaDestroyTextureObject(enviroment_settings.tex_obj));
    safeCudaFreeArray(env_array);
    safeCudaFree(enviroment_settings.cdf);
    safeCudaFree(enviroment_settings.pdf);

    std::size_t size = hdri_env_width * hdri_env_height * channels * sizeof(float);

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

    cudaErrorCheck(cudaMallocArray(&env_array, &channel_desc, hdri_env_width, hdri_env_height));
    cudaErrorCheck(cudaMemcpyToArray(env_array, 0, 0, hdri_env_data, size, cudaMemcpyHostToDevice));

    cudaResourceDesc res_desc{};
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = env_array;

    cudaTextureDesc tex_desc{};
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModePoint;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;

    cudaErrorCheck(cudaCreateTextureObject(&enviroment_settings.tex_obj, &res_desc, &tex_desc, nullptr));

    EnvironmentMap env_map{hdri_env_data,
                           static_cast<uint32_t>(hdri_env_width),
                           static_cast<uint32_t>(hdri_env_height)};

    const auto& cdf = env_map.getDistribution().getCdf();
    enviroment_settings.cdf_size = cdf.size();
    cudaErrorCheck(cudaMalloc(&enviroment_settings.cdf, enviroment_settings.cdf_size * sizeof(float)));
    cudaErrorCheck(cudaMemcpyAsync(enviroment_settings.cdf,
                                   &cdf[0],
                                   enviroment_settings.cdf_size * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   stream));

    const auto& pdf = env_map.getPdf();
    enviroment_settings.pdf_size = pdf.size();
    cudaErrorCheck(cudaMalloc(&enviroment_settings.pdf, enviroment_settings.pdf_size * sizeof(float)));
    cudaErrorCheck(cudaMemcpyAsync(enviroment_settings.pdf,
                                   &pdf[0],
                                   enviroment_settings.pdf_size * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   stream));

    cudaErrorCheck(cudaStreamSynchronize(stream));

    enviroment_settings.width = hdri_env_width;
    enviroment_settings.height = hdri_env_height;

    reset();
}

void CudaPathTracer::Impl::setSceneGeometry(const ISpacePartitioningStructure* sps,
                                            const std::vector<Material>& materials)
{
    safeCudaFree(dev_cuda_triangles_bvh);
    safeCudaFree(dev_cuda_linear_bvh_nodes);

    std::size_t free{0};
    std::size_t total{0};

    cudaErrorCheck(cudaMemGetInfo(&free, &total));
    std::cout << "Free memory: " << free / (1024 * 1024) << "MB" << std::endl;
    std::cout << "Total memory: " << total / (1024 * 1024) << "MB" << std::endl;

    std::vector<CudaTriangle> cuda_triangles_bvh = sceneToCudaTrianglesBvh(sps, materials);
    n_cuda_triangles_bvh = cuda_triangles_bvh.size();
    std::cout << "Triangles BVH: " << cuda_triangles_bvh.size() << std::endl;
    std::cout << "Triangles BVH size: " << (cuda_triangles_bvh.size() * sizeof(CudaTriangle)) / (1024) << "KB"
              << std::endl;
    cudaErrorCheck(cudaMalloc(&dev_cuda_triangles_bvh, cuda_triangles_bvh.size() * sizeof(CudaTriangle)));
    cudaErrorCheck(cudaMemcpyAsync(dev_cuda_triangles_bvh,
                                   &cuda_triangles_bvh[0],
                                   cuda_triangles_bvh.size() * sizeof(CudaTriangle),
                                   cudaMemcpyHostToDevice,
                                   stream));

    std::vector<CudaLinearBvhNode> cuda_linear_bvh = sceneToCudaLinearBvhNode(sps);
    n_cuda_linear_bvh_nodes = cuda_linear_bvh.size();
    std::cout << "CudaLinearBVHNodes: " << cuda_linear_bvh.size() << std::endl;
    std::cout << "CudaLinearBVHNodes size: " << (cuda_linear_bvh.size() * sizeof(CudaLinearBvhNode)) / (1024) << "KB"
              << std::endl;
    cudaErrorCheck(cudaMalloc(&dev_cuda_linear_bvh_nodes, cuda_linear_bvh.size() * sizeof(CudaLinearBvhNode)));
    cudaErrorCheck(cudaMemcpyAsync(dev_cuda_linear_bvh_nodes,
                                   &cuda_linear_bvh[0],
                                   cuda_linear_bvh.size() * sizeof(CudaLinearBvhNode),
                                   cudaMemcpyHostToDevice,
                                   stream));

    setMaterials(materials, false);

    cudaErrorCheck(cudaGetLastError());
    cudaErrorCheck(cudaStreamSynchronize(stream));

    reset();
}

void CudaPathTracer::Impl::setTargetImage(const Image& img)
{
    assert(img.texture_id > 0);
    assert(img.getWidth() > 0 && img.getHeight() > 0);

    if (registered_image.changed(img))
    {
        safeCudaFree(dev_accu_buffer);
        safeCudaFree(dev_path_iteration_buffer);

        cudaErrorCheck(cudaMalloc(&dev_accu_buffer, img.getWidth() * img.getHeight() * sizeof(glm::vec4)));
        cudaErrorCheck(
            cudaMalloc(&dev_path_iteration_buffer, img.getWidth() * img.getHeight() * sizeof(CudaPathIteration)));

        cudaErrorCheck(cudaDeviceSynchronize());

        if (registered_image.cuda_texture_resource != nullptr)
        {
            cudaErrorCheck(cudaGraphicsUnregisterResource(registered_image.cuda_texture_resource));
        }

        // Only call Cuda/OpenGL interop stuff from within the OpenGL context thread!
        cudaErrorCheck(cudaGraphicsGLRegisterImage(&registered_image.cuda_texture_resource,
                                                   img.texture_id,
                                                   GL_TEXTURE_2D,
                                                   cudaGraphicsRegisterFlagsWriteDiscard));

        registered_image.width = img.getWidth();
        registered_image.height = img.getHeight();
        registered_image.tex_id = img.texture_id;
    }

    cudaErrorCheck(cudaMemsetAsync(dev_accu_buffer,
                                   0,
                                   registered_image.width * registered_image.height * sizeof(glm::vec4),
                                   stream));
    cudaErrorCheck(cudaMemsetAsync(dev_path_iteration_buffer,
                                   0,
                                   registered_image.width * registered_image.height * sizeof(CudaPathIteration),
                                   stream));

    cudaErrorCheck(cudaGetLastError());
    cudaErrorCheck(cudaStreamSynchronize(stream));

    reset();
}

void CudaPathTracer::Impl::setCamera(const Camera& cam)
{
    cuda_cam = cameraToCudaCamera(cam);
    reset();
}

void CudaPathTracer::Impl::setMaterials(const std::vector<Material>& materials, const bool& sync)
{
    safeCudaFree(dev_cuda_materials);

    const std::vector<CudaMaterial> cuda_materials = sceneToCudaMaterials(materials);
    const std::size_t size_in_bytes = cuda_materials.size() * sizeof(CudaMaterial);
    n_cuda_materials = static_cast<std::uint32_t>(cuda_materials.size());

    cudaErrorCheck(cudaMalloc(&dev_cuda_materials, size_in_bytes));
    cudaErrorCheck(
        cudaMemcpyAsync(dev_cuda_materials, &cuda_materials[0], size_in_bytes, cudaMemcpyHostToDevice, stream));

    if (sync)
    {
        cudaErrorCheck(cudaStreamSynchronize(stream));
    }

    reset();
}

void CudaPathTracer::Impl::setSettings(const RendererSettings& settings)
{
    target_samples_per_pixel = settings.samplesperpixel;
}

void CudaPathTracer::Impl::copyFrameToTexture()
{
    cudaArray* aarray{nullptr};

    cudaErrorCheck(cudaGraphicsMapResources(1, &registered_image.cuda_texture_resource, cpystream));
    cudaErrorCheck(cudaGraphicsSubResourceGetMappedArray(&aarray, registered_image.cuda_texture_resource, 0, 0));

    cudaErrorCheck(cudaMemcpyToArrayAsync(aarray,
                                          0,
                                          0,
                                          dev_accu_buffer,
                                          registered_image.width * registered_image.height * sizeof(glm::vec4),
                                          cudaMemcpyDeviceToDevice,
                                          cpystream));

    cudaErrorCheck(cudaGraphicsUnmapResources(1, &registered_image.cuda_texture_resource, cpystream));
}

float CudaPathTracer::Impl::getProgress() const
{
    return ((float)iteration / (float)target_samples_per_pixel);
}

float CudaPathTracer::Impl::getInstantRaysPerSec() const
{
    return ((float)registered_image.width * (float)registered_image.height) /
           ((float)last_iteration_elapsed_millis.count() * 0.001f);
}

std::uint32_t CudaPathTracer::Impl::getFinishedSamples() const
{
    return iteration;
}

std::uint32_t CudaPathTracer::Impl::getTargetSamplesPerPixel() const
{
    return target_samples_per_pixel;
}

bool CudaPathTracer::Impl::RegisteredImage::changed(const Image& img) const
{
    return img.getWidth() != width || img.getHeight() != height || img.texture_id != tex_id;
}