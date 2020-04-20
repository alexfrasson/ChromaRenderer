#pragma once

#include "chroma-renderer/core/types/Camera.h"
#include "chroma-renderer/core/types/Material.h"
#include "chroma-renderer/core/types/Mesh.h"
#include "chroma-renderer/core/types/Object.h"

#include <functional>
#include <string>
#include <vector>

class Scene
{
  public:
    Scene() = default;
    ~Scene();

    size_t triangleCount();
    void clear();
    BoundingBox getBoundingBox();

    std::vector<Object> objects;
    std::vector<Mesh*> meshes;
    std::vector<Material> materials;
    Camera camera;

    bool ready;

    float* hdriEnvData = nullptr;
    int hdriEnvWidth, hdriEnvHeight;
};