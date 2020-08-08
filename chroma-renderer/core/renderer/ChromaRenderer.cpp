#include "chroma-renderer/core/renderer/ChromaRenderer.h"
#include "chroma-renderer/core/renderer/CudaPathTracer.h"
#include "chroma-renderer/core/renderer/PostProcessor.h"
#include "chroma-renderer/core/renderer/RendererSettings.h"
#include "chroma-renderer/core/scene/ModelImporter.h"
#include "chroma-renderer/core/scene/Scene.h"
#include "chroma-renderer/core/space-partition/BVH.h"
#include "chroma-renderer/core/space-partition/ISpacePartitioningStructure.h"
#include "chroma-renderer/core/types/Image.h"
#include "chroma-renderer/core/types/Mesh.h"
#include "chroma-renderer/core/types/environment_map.h"
#include "chroma-renderer/core/utility/Stopwatch.h"

#define STB_IMAGE_IMPLEMENTATION
#include <atomic>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <stb_image.h>
#include <string>
#include <thread>

class ChromaRenderer::Impl
{
  public:
    Impl();
    ~Impl();

    State getState();
    bool isRunning();
    void stopRender();
    void importScene(const std::string& filename);
    void importEnviromentMap(const std::string& filename);
    void startRender();
    RendererSettings getSettings();
    void setSettings(const RendererSettings& settings);
    void setPostProcessingSettings(const ChromaRenderer::PostProcessingSettings& settings);
    ChromaRenderer::PostProcessingSettings getPostProcessingSettings();
    Scene& getScene();
    Image& getTarget();
    void update();
    Progress getProgress();
    void updateMaterials();

  private:
    void setEnvMap(const float* data, const uint32_t width, const uint32_t height, const uint32_t channels);
    void saveLog();
    void setSize(unsigned int width, unsigned int height);
    void genTasks();
    bool isIdle();
    void cbSceneLoadedScene();

    RendererSettings settings;
    Scene scene;
    State state;
    CudaPathTracer cudaPathTracer;
    PostProcessor post_processor;
    Stopwatch stopwatch;
    Image renderer_target;
    Image final_target;
    Image env_map;
    bool running = false;
    float invPixelCount;
    int pixelCount;
    std::unique_ptr<ISpacePartitioningStructure> sps;
};

ChromaRenderer::Impl::Impl() : state(State::IDLE)
{
    setSettings(settings);
}

ChromaRenderer::Impl::~Impl()
{
}

void ChromaRenderer::Impl::updateMaterials()
{
    cudaPathTracer.setMaterials(scene.materials);
}

void ChromaRenderer::Impl::setPostProcessingSettings(const ChromaRenderer::PostProcessingSettings& a_settings)
{
    post_processor.adjustExposure = a_settings.adjust_exposure;
    post_processor.linearToSrbg = a_settings.linear_to_srgb;
    post_processor.tonemapping = a_settings.tonemapping;
}

ChromaRenderer::PostProcessingSettings ChromaRenderer::Impl::getPostProcessingSettings()
{
    ChromaRenderer::PostProcessingSettings post_processing_settings;
    post_processing_settings.adjust_exposure = post_processor.adjustExposure;
    post_processing_settings.linear_to_srgb = post_processor.linearToSrbg;
    post_processing_settings.tonemapping = post_processor.tonemapping;
    return post_processing_settings;
}

ChromaRenderer::Progress ChromaRenderer::Impl::getProgress()
{
    Progress progress;
    progress.progress = cudaPathTracer.getProgress();
    progress.instant_rays_per_sec = cudaPathTracer.getInstantRaysPerSec();
    progress.finished_samples = cudaPathTracer.getFinishedSamples();
    progress.target_samples_per_pixel = cudaPathTracer.getTargetSamplesPerPixel();
    return progress;
}

Scene& ChromaRenderer::Impl::getScene()
{
    return scene;
}

Image& ChromaRenderer::Impl::getTarget()
{
    return final_target;
}

bool ChromaRenderer::Impl::isIdle()
{
    if (state == State::RENDERING && !isRunning())
        state = State::IDLE;
    return (state == State::IDLE);
}

ChromaRenderer::State ChromaRenderer::Impl::getState()
{
    if (state == ChromaRenderer::RENDERING)
    {
        if (!isRunning())
            state = ChromaRenderer::IDLE;
    }
    /*if (state == ChromaRenderer::PROCESSINGSCENE)
    {
        if (scene.ready)
            state = ChromaRenderer::IDLE;
    }*/
    return state;
}

void ChromaRenderer::Impl::setSize(unsigned int width, unsigned int height)
{
    renderer_target.setSize(width, height);
    final_target.setSize(width, height);
    scene.camera.setSize(width, height);
}

void ChromaRenderer::Impl::setSettings(const RendererSettings& psettings)
{
    settings = psettings;
    scene.camera.horizontalFOV(settings.horizontalFOV);
    setSize(settings.width, settings.height);
    pixelCount = settings.width * settings.height;
    invPixelCount = 1.f / (float)pixelCount;
    cudaPathTracer.setSettings(settings);
}

RendererSettings ChromaRenderer::Impl::getSettings()
{
    return settings;
}

void ChromaRenderer::Impl::startRender()
{
    if (!isIdle())
    {
        return;
    }

    renderer_target.clear();
    final_target.clear();

    cudaPathTracer.setTargetImage(renderer_target);
    cudaPathTracer.setCamera(scene.camera);

    stopwatch.restart();

    state = State::RENDERING;
    running = true;
}

void ChromaRenderer::Impl::stopRender()
{
    running = false;
    state = State::IDLE;
}

void ChromaRenderer::Impl::importScene(const std::string& filename)
{
    if (!isIdle())
    {
        return;
    }

    state = State::LOADINGSCENE;
    ModelImporter::importcbscene(
        filename,
        scene,
        static_cast<std::function<void()>>(std::bind(&ChromaRenderer::Impl::cbSceneLoadedScene, std::ref(*this))));
}

void ChromaRenderer::Impl::setEnvMap(const float* data,
                                     const uint32_t width,
                                     const uint32_t height,
                                     const uint32_t channels)
{
    cudaPathTracer.setEnvMap(data, width, height, channels);
    env_map.setData(data, width, height);
}

void ChromaRenderer::Impl::importEnviromentMap(const std::string& filename)
{
    const uint32_t requested_channels = 4;
    float* data = nullptr;
    int width, height, channels;
    data = stbi_loadf(filename.c_str(), &width, &height, &channels, requested_channels);

    if (data == nullptr)
    {
        std::cout << "Could not load hdri!" << std::endl;
    }

    std::cout << "Width: " << width << " Height: " << height << " Channels: " << channels << std::endl;

    setEnvMap(data, width, height, requested_channels);

    stbi_image_free(data);
}

void ChromaRenderer::Impl::cbSceneLoadedScene()
{
    state = State::PROCESSINGSCENE;
    sps = std::make_unique<BVH>();
    sps->build(scene.meshes);
    cudaPathTracer.setSceneGeometry(sps.get(), scene.materials);

    settings.width = scene.camera.width;
    settings.height = scene.camera.height;
    settings.horizontalFOV = scene.camera.horizontalFOV();

    setSettings(settings);

    cudaPathTracer.setTargetImage(renderer_target);
    cudaPathTracer.setCamera(scene.camera);

    state = State::IDLE;
}

void ChromaRenderer::Impl::update()
{
    if (state == State::RENDERING)
    {
        cudaPathTracer.render();
        post_processor.process(scene.camera, renderer_target, final_target, true);
        if (cudaPathTracer.getProgress() >= 1.0f)
        {
            running = false;
            state = State::IDLE;
        }
    }
}

bool ChromaRenderer::Impl::isRunning()
{
    if (running)
    {
        running = cudaPathTracer.getFinishedSamples() < cudaPathTracer.getTargetSamplesPerPixel();
        if (!running)
        {
            stopwatch.stop();
        }
    }
    return running;
}

ChromaRenderer::ChromaRenderer() : impl_{std::make_unique<ChromaRenderer::Impl>()}
{
}

ChromaRenderer::~ChromaRenderer() = default;

ChromaRenderer::State ChromaRenderer::getState()
{
    return impl_->getState();
}

bool ChromaRenderer::isRunning()
{
    return impl_->isRunning();
}

void ChromaRenderer::update()
{
    impl_->update();
}

void ChromaRenderer::stopRender()
{
    impl_->stopRender();
}

void ChromaRenderer::importScene(const std::string& filename)
{
    impl_->importScene(filename);
}

void ChromaRenderer::importEnviromentMap(const std::string& filename)
{
    impl_->importEnviromentMap(filename);
}

void ChromaRenderer::startRender()
{
    impl_->startRender();
}

RendererSettings ChromaRenderer::getSettings()
{
    return impl_->getSettings();
}

void ChromaRenderer::setSettings(const RendererSettings& settings)
{
    impl_->setSettings(settings);
}

Scene& ChromaRenderer::getScene()
{
    return impl_->getScene();
}

Image& ChromaRenderer::getTarget()
{
    return impl_->getTarget();
}

ChromaRenderer::Progress ChromaRenderer::getProgress()
{
    return impl_->getProgress();
}

void ChromaRenderer::updateMaterials()
{
    impl_->updateMaterials();
}

void ChromaRenderer::setPostProcessingSettings(const ChromaRenderer::PostProcessingSettings& settings)
{
    impl_->setPostProcessingSettings(settings);
}

ChromaRenderer::PostProcessingSettings ChromaRenderer::getPostProcessingSettings()
{
    return impl_->getPostProcessingSettings();
}