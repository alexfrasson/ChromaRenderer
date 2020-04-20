#include "chroma-renderer/core/scene/Scene.h"

Scene::~Scene()
{
    clear();
}

size_t Scene::triangleCount()
{
    size_t tc = 0;
    for (size_t i = 0; i < objects.size(); i++)
        tc += objects[i].f.size();
    return tc;
}

void Scene::clear()
{
    objects.clear();
    for (size_t i = 0; i < meshes.size(); i++)
    {
        delete meshes[i];
    }
    meshes.clear();
    materials.clear();
}

BoundingBox Scene::getBoundingBox()
{
    BoundingBox bb;

    for (size_t i = 0; i < objects.size(); i++)
    {
        if (objects[i].boundingBox.max.x > bb.max.x)
            bb.max.x = objects[i].boundingBox.max.x;
        if (objects[i].boundingBox.max.z > bb.max.z)
            bb.max.z = objects[i].boundingBox.max.z;
        if (objects[i].boundingBox.max.y > bb.max.y)
            bb.max.y = objects[i].boundingBox.max.y;

        if (objects[i].boundingBox.min.x < bb.min.x)
            bb.min.x = objects[i].boundingBox.min.x;
        if (objects[i].boundingBox.min.z < bb.min.z)
            bb.min.z = objects[i].boundingBox.min.z;
        if (objects[i].boundingBox.min.y < bb.min.y)
            bb.min.y = objects[i].boundingBox.min.y;
    }

    for (size_t i = 0; i < meshes.size(); i++)
    {
        if (meshes[i]->boundingBox.max.x > bb.max.x)
            bb.max.x = meshes[i]->boundingBox.max.x;
        if (meshes[i]->boundingBox.max.z > bb.max.z)
            bb.max.z = meshes[i]->boundingBox.max.z;
        if (meshes[i]->boundingBox.max.y > bb.max.y)
            bb.max.y = meshes[i]->boundingBox.max.y;

        if (meshes[i]->boundingBox.min.x < bb.min.x)
            bb.min.x = meshes[i]->boundingBox.min.x;
        if (meshes[i]->boundingBox.min.z < bb.min.z)
            bb.min.z = meshes[i]->boundingBox.min.z;
        if (meshes[i]->boundingBox.min.y < bb.min.y)
            bb.min.y = meshes[i]->boundingBox.min.y;
    }

    return bb;
}