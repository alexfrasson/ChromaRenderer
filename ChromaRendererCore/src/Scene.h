#pragma once

#include "Camera.h"
#include "Object.h"
#include <Mesh.h>
#include <string>
#include <vector>
//#include "Light.h"
#include <ISpacePartitioningStructure.h>
//#include <ChromaRenderer/KDTree.h>
#include <BVH.h>
#include <Material.h>
#include <functional>

class Scene
{
  public:
    std::vector<Object> objects;
    std::vector<Mesh*> meshes;
    std::vector<Material> materials;
    // std::vector<Light> lights;
    Camera camera;
    ISpacePartitioningStructure* sps;
    // BVH* sps;
    bool ready;

    float* hdriEnvData = nullptr;
    int hdriEnvWidth, hdriEnvHeight;

    Scene(void);
    ~Scene(void);
    size_t triangleCount();
    void LoadMesh(std::string file);
    void addObject(Object o);
    void addObject(Object o, std::function<void()> cb);
    void addMesh(Mesh* m);
    void addMesh(Mesh* m, std::function<void()> cb);
    void clear();
    BoundingBox getBoundingBox();
    // void LoadMesh(std::string file, glm::vec3 pos);
};