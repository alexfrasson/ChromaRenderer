#pragma once

#include "chroma-renderer/core/scene/camera.h"
#include "chroma-renderer/core/types/material.h"
#include "chroma-renderer/core/types/mesh.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

class Scene
{
  public:
    size_t triangleCount();
    void clear();
    BoundingBox getBoundingBox();

    std::vector<std::unique_ptr<Mesh>> meshes;
    std::vector<Material> materials;
    Camera camera;
    bool ready{false};
    float* hdri_env_data{nullptr};
    int hdri_env_width{0};
    int hdri_env_height{0};
};