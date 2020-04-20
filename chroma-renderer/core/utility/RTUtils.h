#pragma once

#include "chroma-renderer/core/scene/Camera.h"
#include "chroma-renderer/core/scene/Object.h"
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
bool closestIntersectionRayObjects(const Ray& ray,
                                   const std::vector<Object>& objects,
                                   const Face** hitTriangle,
                                   float& hitDistance,
                                   const bool boundingboxtest);
bool intersectUntilRayObjects(const Ray& ray,
                              const std::vector<Object>& objects,
                              const glm::vec3 point,
                              const bool boundingboxtest);
bool intersectShadowRay(const Ray& r, const float& lightDistance, const Object& o, const bool boundingboxtest);

void rayDirection(const Camera& camera, const int i, const int j, Ray* rays, const int numRays);

bool intersectRayTrianglesMollerTrumbore(const Ray& r, const std::vector<Face>& f, Intersection& intersection);
bool intersectRayTrianglesMollerTrumbore(const Ray& r, const std::vector<Face*>& f, Intersection& intersection);
bool intersectRayTrianglesMollerTrumbore(const Ray& r, Face** f, const unsigned int nFaces, Intersection& intersection);
bool intersectRayTriangleMollerTrumbore(const Ray& r, const Face& f, float& u, float& v, float& t);
bool intersectRayTriangleMollerTrumbore(const Ray& r, const Face* f, float& u, float& v, float& t);
bool intersectRayTriangleMollerTrumbore(const Ray& r, const Face& f, float& t);
bool intersectRayTriangleMollerTrumboreNOBACKFACECULLING(const Ray& r, const Face& f, float& t);
bool intersectRayObjectMollerTrumbore(const Ray& r, const Object& o, const Face** hitFace, float& hitDistance);
bool intersectRayTrianglesMollerTrumboreSIMD128(const Ray& r,
                                                const std::vector<Face>& f,
                                                const Face** hitTriangle,
                                                float& hitDistance);
bool intersectRayObjectMollerTrumboreSIMD128(const Ray& r,
                                             const Object& object,
                                             const Face** hitTriangle,
                                             float& hitDistance);
bool intersectRayObjectMollerTrumboreSIMD256(const Ray& r,
                                             const Object& object,
                                             const Face** hitTriangle,
                                             float& hitDistance);
} // namespace RTUtils