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
        if (meshes[i]->bounding_box.max.x > bb.max.x)
        {
            bb.max.x = meshes[i]->bounding_box.max.x;
        }
        if (meshes[i]->bounding_box.max.z > bb.max.z)
        {
            bb.max.z = meshes[i]->bounding_box.max.z;
        }
        if (meshes[i]->bounding_box.max.y > bb.max.y)
        {
            bb.max.y = meshes[i]->bounding_box.max.y;
        }

        if (meshes[i]->bounding_box.min.x < bb.min.x)
        {
            bb.min.x = meshes[i]->bounding_box.min.x;
        }
        if (meshes[i]->bounding_box.min.z < bb.min.z)
        {
            bb.min.z = meshes[i]->bounding_box.min.z;
        }
        if (meshes[i]->bounding_box.min.y < bb.min.y)
        {
            bb.min.y = meshes[i]->bounding_box.min.y;
        }
    }

    return bb;
}