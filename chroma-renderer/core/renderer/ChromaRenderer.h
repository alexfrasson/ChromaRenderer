#pragma once

#include "chroma-renderer/core/renderer/RendererSettings.h"
#include "chroma-renderer/core/scene/Scene.h"
#include "chroma-renderer/core/types/Image.h"

#include <memory>

class ChromaRenderer
{
  public:
    enum State
    {
        kRendering,
        kLoadingScene,
        kProcessingScene,
        kIdle
    };

    struct Progress
    {
        float progress;
        float instant_rays_per_sec;
        std::uint32_t finished_samples;
        std::uint32_t target_samples_per_pixel;
    };

    struct PostProcessingSettings
    {
        bool adjust_exposure;
        bool tonemapping;
        bool linear_to_srgb;
    };

    ChromaRenderer();
    ~ChromaRenderer();

    ChromaRenderer(const ChromaRenderer&) = delete;
    ChromaRenderer(ChromaRenderer&&) = delete;
    ChromaRenderer& operator=(const ChromaRenderer&) = delete;
    ChromaRenderer& operator=(ChromaRenderer&&) = delete;

    void startRender();
    void stopRender();
    void update();
    bool isRunning();
    void importScene(const std::string& filename);
    void importEnviromentMap(const std::string& filename);
    State getState();
    RendererSettings getSettings();
    void setSettings(const RendererSettings& settings);
    void setPostProcessingSettings(const PostProcessingSettings& settings);
    PostProcessingSettings getPostProcessingSettings() const;
    Scene& getScene();
    Image& getTarget();
    Progress getProgress();
    void updateMaterials();

  private:
    class Impl;
    std::unique_ptr<Impl> impl_; // NOLINT
};