#include "chroma-renderer/core/scene/Scene.h"

size_t Scene::triangleCount()
{
    size_t tc = 0;
    for (size_t i = 0; i < meshes.size(); i++)
    {
        tc += meshes[i]->t.size();
    }
    return tc;
}

void Scene::clear()
{
    meshes.clear();
    materials.clear();
}

BoundingBox Scene::getBoundingBox()
{
    BoundingBox bb;

    for (size_t i = 0; i < meshes.size(); i++)
    {
        if (meshes[i]->boundingBox.max.x > bb.max.x)
        {
            bb.max.x = meshes[i]->boundingBox.max.x;
        }
        if (meshes[i]->boundingBox.max.z > bb.max.z)
        {
            bb.max.z = meshes[i]->boundingBox.max.z;
        }
        if (meshes[i]->boundingBox.max.y > bb.max.y)
        {
            bb.max.y = meshes[i]->boundingBox.max.y;
        }

        if (meshes[i]->boundingBox.min.x < bb.min.x)
        {
            bb.min.x = meshes[i]->boundingBox.min.x;
        }
        if (meshes[i]->boundingBox.min.z < bb.min.z)
        {
            bb.min.z = meshes[i]->boundingBox.min.z;
        }
        if (meshes[i]->boundingBox.min.y < bb.min.y)
        {
            bb.min.y = meshes[i]->boundingBox.min.y;
        }
    }

    return bb;
}