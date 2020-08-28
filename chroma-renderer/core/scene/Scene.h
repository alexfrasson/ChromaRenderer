#pragma once

#include "chroma-renderer/core/scene/Camera.h"
#include "chroma-renderer/core/types/Material.h"
#include "chroma-renderer/core/types/Mesh.h"

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
    float* hdriEnvData{nullptr};
    int hdriEnvWidth{0};
    int hdriEnvHeight{0};
};