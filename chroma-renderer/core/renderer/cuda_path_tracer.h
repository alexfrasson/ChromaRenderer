#pragma once

#include "chroma-renderer/core/renderer/renderer_settings.h"
#include "chroma-renderer/core/scene/camera.h"
#include "chroma-renderer/core/space-partition/i_space_partitioning_structure.h"
#include "chroma-renderer/core/types/image.h"
#include "chroma-renderer/core/types/material.h"

#include <cstdint>
#include <memory>
#include <vector>

class CudaPathTracer
{
  public:
    CudaPathTracer();
    ~CudaPathTracer();
    CudaPathTracer(const CudaPathTracer&) = delete;
    CudaPathTracer(CudaPathTracer&&) = delete;
    CudaPathTracer& operator=(const CudaPathTracer&) = delete;
    CudaPathTracer& operator=(CudaPathTracer&&) = delete;

    void render();

    void setSceneGeometry(const ISpacePartitioningStructure* sps, const std::vector<Material>& materials);
    void setEnvMap(const float* hdri_env_data,
                   std::size_t hdri_env_width,
                   std::size_t hdri_env_height,
                   std::size_t channels);
    void setCamera(const Camera& cam);
    void setSettings(const RendererSettings& settings);
    void setMaterials(const std::vector<Material>& materials);
    void setTargetImage(const Image& img);

    float getProgress() const;
    float getInstantRaysPerSec() const;
    std::uint32_t getFinishedSamples() const;
    std::uint32_t getTargetSamplesPerPixel() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_; // NOLINT
};