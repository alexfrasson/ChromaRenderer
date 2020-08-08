#pragma once

#include "chroma-renderer/core/scene/Camera.h"
#include "chroma-renderer/core/types/Intersection.h"
#include "chroma-renderer/core/types/Ray.h"

#include <atomic>

namespace RTUtils
{
// void trace(Scene& scene,
//            Image& img,
//            RendererSettings& settings,
//            std::atomic<int>& pixelcount,
//            Interval interval,
//            bool& abort);
bool hitBoundingBox(const Ray& r, const BoundingBox& bb);
bool hitBoundingBox(const Ray& r, const BoundingBox& bb, float& tmin, float& tmax);
bool hitBoundingBoxSlab(const BoundingBox& bb,
                        const Ray& r,
                        const glm::vec3& invRayDir,
                        const glm::bvec3& dirIsNeg,
                        float& tmin,
                        float& tmax);
bool hitBoundingBoxSlab(const BoundingBox& bb, const Ray& r, const glm::vec3& invRayDir, float& tmin, float& tmax);
float calcColor(const Ray& ray, const Face* triangle, const float distance);
float calcColor(const Intersection& is);

void rayDirection(const Camera& camera, const int i, const int j, Ray* rays, const int numRays);

bool intersectRayTrianglesMollerTrumbore(const Ray& r, const std::vector<Face>& f, Intersection& intersection);
bool intersectRayTrianglesMollerTrumbore(const Ray& r, const std::vector<Face*>& f, Intersection& intersection);
bool intersectRayTrianglesMollerTrumbore(const Ray& r, Face** f, const unsigned int nFaces, Intersection& intersection);
bool intersectRayTriangleMollerTrumbore(const Ray& r, const Face& f, float& u, float& v, float& t);
bool intersectRayTriangleMollerTrumbore(const Ray& r, const Face* f, float& u, float& v, float& t);
bool intersectRayTriangleMollerTrumbore(const Ray& r, const Face& f, float& t);
bool intersectRayTriangleMollerTrumboreNOBACKFACECULLING(const Ray& r, const Face& f, float& t);
} // namespace RTUtils