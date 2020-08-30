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

    RendererSettings settings_;
    Scene scene_;
    State state_{State::kIdle};
    CudaPathTracer cuda_path_tracer_;
    PostProcessor post_processor_;
    Stopwatch stopwatch_;
    Image renderer_target_;
    Image final_target_;
    Image env_map_;
    bool running_{false};
    float inv_pixel_count_{0.0f};
    int pixel_count_{0};
    std::unique_ptr<ISpacePartitioningStructure> sps_;
};

ChromaRenderer::Impl::Impl()
{
    setSettings(settings_);
}

void ChromaRenderer::Impl::updateMaterials()
{
    cuda_path_tracer_.setMaterials(scene_.materials);
}

void ChromaRenderer::Impl::setPostProcessingSettings(const PostProcessingSettings& a_settings)
{
    post_processor_.adjust_exposure = a_settings.adjust_exposure;
    post_processor_.linear_to_srbg = a_settings.linear_to_srgb;
    post_processor_.tonemapping = a_settings.tonemapping;
}

ChromaRenderer::PostProcessingSettings ChromaRenderer::Impl::getPostProcessingSettings() const
{
    PostProcessingSettings post_processing_settings{};
    post_processing_settings.adjust_exposure = post_processor_.adjust_exposure;
    post_processing_settings.linear_to_srgb = post_processor_.linear_to_srbg;
    post_processing_settings.tonemapping = post_processor_.tonemapping;
    return post_processing_settings;
}

ChromaRenderer::Progress ChromaRenderer::Impl::getProgress()
{
    Progress progress{};
    progress.progress = cuda_path_tracer_.getProgress();
    progress.instant_rays_per_sec = cuda_path_tracer_.getInstantRaysPerSec();
    progress.finished_samples = cuda_path_tracer_.getFinishedSamples();
    progress.target_samples_per_pixel = cuda_path_tracer_.getTargetSamplesPerPixel();
    return progress;
}

Scene& ChromaRenderer::Impl::getScene()
{
    return scene_;
}

Image& ChromaRenderer::Impl::getTarget()
{
    return final_target_;
}

bool ChromaRenderer::Impl::isIdle()
{
    if (state_ == State::kRendering && !isRunning())
    {
        state_ = State::kIdle;
    }
    return (state_ == State::kIdle);
}

ChromaRenderer::State ChromaRenderer::Impl::getState()
{
    if (state_ == ChromaRenderer::kRendering)
    {
        if (!isRunning())
        {
            state_ = ChromaRenderer::kIdle;
        }
    }
    /*if (state == ChromaRenderer::PROCESSINGSCENE)
    {
        if (scene.ready)
            state = ChromaRenderer::IDLE;
    }*/
    return state_;
}

void ChromaRenderer::Impl::setSize(int width, int height)
{
    renderer_target_.setSize(static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height));
    final_target_.setSize(static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height));
    scene_.camera.setSize(width, height);
}

void ChromaRenderer::Impl::setSettings(const RendererSettings& psettings)
{
    settings_ = psettings;
    scene_.camera.horizontalFOV(settings_.horizontal_fov);
    setSize(settings_.width, settings_.height);
    pixel_count_ = settings_.width * settings_.height;
    inv_pixel_count_ = 1.f / (float)pixel_count_;
    cuda_path_tracer_.setSettings(settings_);
}

RendererSettings ChromaRenderer::Impl::getSettings()
{
    return settings_;
}

void ChromaRenderer::Impl::startRender()
{
    if (!isIdle())
    {
        return;
    }

    renderer_target_.clear();
    final_target_.clear();

    cuda_path_tracer_.setTargetImage(renderer_target_);
    cuda_path_tracer_.setCamera(scene_.camera);

    stopwatch_.restart();

    state_ = State::kRendering;
    running_ = true;
}

void ChromaRenderer::Impl::stopRender()
{
    running_ = false;
    state_ = State::kIdle;
}

void ChromaRenderer::Impl::importScene(const std::string& filename)
{
    if (!isIdle())
    {
        return;
    }

    state_ = State::kLoadingScene;
    model_importer::importcbscene(filename, scene_, [&]() { cbSceneLoadedScene(); });
}

void ChromaRenderer::Impl::setEnvMap(const float* data,
                                     const std::uint32_t width,
                                     const std::uint32_t height,
                                     const std::uint32_t channels)
{
    cuda_path_tracer_.setEnvMap(data, width, height, channels);
    env_map_.setData(data, width, height);
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
    state_ = State::kProcessingScene;
    sps_ = std::make_unique<BVH>();
    sps_->build(scene_.meshes);
    cuda_path_tracer_.setSceneGeometry(sps_.get(), scene_.materials);

    settings_.width = scene_.camera.width;
    settings_.height = scene_.camera.height;
    settings_.horizontal_fov = scene_.camera.horizontalFOV();

    setSettings(settings_);

    cuda_path_tracer_.setTargetImage(renderer_target_);
    cuda_path_tracer_.setCamera(scene_.camera);

    state_ = State::kIdle;
}

void ChromaRenderer::Impl::update()
{
    if (state_ == State::kRendering)
    {
        cuda_path_tracer_.render();
        post_processor_.process(scene_.camera, renderer_target_, final_target_, true);
        if (cuda_path_tracer_.getProgress() >= 1.0f)
        {
            running_ = false;
            state_ = State::kIdle;
        }
    }
}

bool ChromaRenderer::Impl::isRunning()
{
    if (running_)
    {
        running_ = cuda_path_tracer_.getFinishedSamples() < cuda_path_tracer_.getTargetSamplesPerPixel();
        if (!running_)
        {
            stopwatch_.stop();
        }
    }
    return running_;
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