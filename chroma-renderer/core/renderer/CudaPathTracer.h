#pragma once

#include "chroma-renderer/core/renderer/RendererSettings.h"
#include "chroma-renderer/core/scene/Camera.h"
#include "chroma-renderer/core/space-partition/ISpacePartitioningStructure.h"
#include "chroma-renderer/core/types/Image.h"
#include "chroma-renderer/core/types/Material.h"

#include <vector>
#include <memory>
#include <cstdint>

class CudaPathTracer
{
  public:
    CudaPathTracer();
    ~CudaPathTracer();
    CudaPathTracer(const CudaPathTracer&) = delete;
    CudaPathTracer(CudaPathTracer&&) = delete;
    CudaPathTracer& operator=(const CudaPathTracer&) = delete;
    CudaPathTracer& operator=(CudaPathTracer&&) = delete;

    void init(const ISpacePartitioningStructure* sps, const std::vector<Material>& materials);
    void init(const float* hdriEnvData, const int hdriEnvWidth, const int hdriEnvHeight, const int channels);
    void init(Image& img, Camera& cam);
    void uploadMaterials(const std::vector<Material>& materials);
    void renderThread(bool& abort);
    void render();
    void setSettings(RendererSettings& settings);
    void copyFrameToTexture();
    void dispatchComputeShader(bool sync);
    float getProgress();
    std::uint32_t getFinishedSamples();
    std::uint32_t getTargetSamplesPerPixel();
    float instantRaysPerSec();

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};