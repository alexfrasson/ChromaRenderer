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

#include <stb_image.h>

#include <atomic>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <thread>

class ChromaRenderer::Impl
{
  public:
    Impl();

    State getState();
    bool isRunning();
    void stopRender();
    void importScene(const std::string& filename);
    void importEnviromentMap(const std::string& filename);
    void startRender();
    RendererSettings getSettings();
    void setSettings(const RendererSettings& settings);
    void setPostProcessingSettings(const ChromaRenderer::PostProcessingSettings& settings);
    PostProcessingSettings getPostProcessingSettings() const;
    Scene& getScene();
    Image& getTarget();
    void update();
    Progress getProgress();
    void updateMaterials();

  private:
    void setEnvMap(const float* data, std::uint32_t width, std::uint32_t height, std::uint32_t channels);
    void saveLog();
    void setSize(int width, int height);
    void genTasks();
    bool isIdle();
    void cbSceneLoadedScene();

    RendererSettings settings;
    Scene scene;
    State state{State::IDLE};
    CudaPathTracer cudaPathTracer;
    PostProcessor post_processor;
    Stopwatch stopwatch;
    Image renderer_target;
    Image final_target;
    Image env_map;
    bool running{false};
    float invPixelCount{0.0f};
    int pixelCount{0};
    std::unique_ptr<ISpacePartitioningStructure> sps;
};

ChromaRenderer::Impl::Impl()
{
    setSettings(settings);
}

void ChromaRenderer::Impl::updateMaterials()
{
    cudaPathTracer.setMaterials(scene.materials);
}

void ChromaRenderer::Impl::setPostProcessingSettings(const PostProcessingSettings& a_settings)
{
    post_processor.adjustExposure = a_settings.adjust_exposure;
    post_processor.linearToSrbg = a_settings.linear_to_srgb;
    post_processor.tonemapping = a_settings.tonemapping;
}

ChromaRenderer::PostProcessingSettings ChromaRenderer::Impl::getPostProcessingSettings() const
{
    PostProcessingSettings post_processing_settings{};
    post_processing_settings.adjust_exposure = post_processor.adjustExposure;
    post_processing_settings.linear_to_srgb = post_processor.linearToSrbg;
    post_processing_settings.tonemapping = post_processor.tonemapping;
    return post_processing_settings;
}

ChromaRenderer::Progress ChromaRenderer::Impl::getProgress()
{
    Progress progress{};
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
    {
        state = State::IDLE;
    }
    return (state == State::IDLE);
}

ChromaRenderer::State ChromaRenderer::Impl::getState()
{
    if (state == ChromaRenderer::RENDERING)
    {
        if (!isRunning())
        {
            state = ChromaRenderer::IDLE;
        }
    }
    /*if (state == ChromaRenderer::PROCESSINGSCENE)
    {
        if (scene.ready)
            state = ChromaRenderer::IDLE;
    }*/
    return state;
}

void ChromaRenderer::Impl::setSize(int width, int height)
{
    renderer_target.setSize(static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height));
    final_target.setSize(static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height));
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
    ModelImporter::importcbscene(filename, scene, [&]() { cbSceneLoadedScene(); });
}

void ChromaRenderer::Impl::setEnvMap(const float* data,
                                     const std::uint32_t width,
                                     const std::uint32_t height,
                                     const std::uint32_t channels)
{
    cudaPathTracer.setEnvMap(data, width, height, channels);
    env_map.setData(data, width, height);
}

void ChromaRenderer::Impl::importEnviromentMap(const std::string& filename)
{
    const uint32_t requested_channels{4};
    float* data{nullptr};
    int width{0};
    int height{0};
    int channels{0};
    data = stbi_loadf(filename.c_str(), &width, &height, &channels, requested_channels);

    if (data == nullptr)
    {
        std::cout << "Could not load hdri!" << std::endl;
    }

    std::cout << "Width: " << width << " Height: " << height << " Channels: " << channels << std::endl;

    setEnvMap(data, static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), requested_channels);

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

ChromaRenderer::PostProcessingSettings ChromaRenderer::getPostProcessingSettings() const
{
    return impl_->getPostProcessingSettings();
}