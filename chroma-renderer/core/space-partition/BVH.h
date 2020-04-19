#pragma once

#include "chroma-renderer/core/space-partition/IIntersectable.h"
#include "chroma-renderer/core/space-partition/ISpacePartitioningStructure.h"
#include "chroma-renderer/core/types/Mesh.h"
#include "chroma-renderer/core/types/Object.h"
#include "chroma-renderer/core/types/Ray.h"
#include <vector>

struct BvhNode
{
    BvhNode()
    {
        child[0] = NULL;
        child[1] = NULL;
    }
    // float bbox[2 * 3 * 4];			// Four bounding boxes
    BoundingBox bbox;
    BvhNode* child[2];
    // int axis0, axis1, axis2;
    // int fill;
    int startID;
    int endID;
    bool isLeaf;
    uint8_t axis;
};

struct LinearBvhNode
{
    BoundingBox bbox;
    union {
        uint32_t primitivesOffset;  // Leaf
        uint32_t secondChildOffset; // Interior
    };
    uint8_t nPrimitives; // 0 -> interior node
    uint8_t axis;
    uint8_t pad[2];
};

struct BVHPrimitiveInfo
{
    BVHPrimitiveInfo(BoundingBox bbox, int index)
    {
        this->centroid = bbox.centroid();
        this->bbox = bbox;
        this->index = index;
    }
    glm::vec3 centroid;
    BoundingBox bbox;
    int index;
};

class BVH : public ISpacePartitioningStructure
{
  public:
    BVH();
    ~BVH();
    void clear();
    BvhNode* free(BvhNode* node);
    int flattenBvh(BvhNode* node, int& offset);
    bool build(std::vector<Object>& o);
    bool build(std::vector<Mesh*>& m);
    BvhNode* buildnode(int depth, std::vector<BVHPrimitiveInfo>& primitive, int startID, int endID);
    BvhNode* buildNode(int depth,
                       std::vector<glm::vec3>& centroids,
                       std::vector<BoundingBox>& bboxes,
                       int startID,
                       int endID);
    float cost(float saL, float nL, float saR, float nR);
    float cost(float sa, float n);
    bool intersect(Ray& r, Intersection& intersection) const;
    bool intersectF(const Ray& r, Intersection& intersection, float& nNodeHitsNormalized) const;
    bool intersectF(const Ray& r, Intersection& intersection) const;
    uint32_t nIntersectedNodes(const Ray& r);

    size_t sizeInBytes(void);
    void abort(void)
    {
    }

    unsigned int nLeafs;
    unsigned int nNodes;
    int maxDepth;

    std::vector<Face> tris;
    std::vector<Triangle*> triangles;

    std::vector<int> id;
    BvhNode* root = nullptr;
    LinearBvhNode* lroot = nullptr;

    bool splitMidpoint(std::vector<BVHPrimitiveInfo>& primitive,
                       BoundingBox& trianglesbbox,
                       int startID,
                       int endID,
                       int& splitindex,
                       int& splitdim);
    bool splitMedian(std::vector<BVHPrimitiveInfo>& primitive,
                     BoundingBox& trianglesbbox,
                     int startID,
                     int endID,
                     int& splitindex,
                     int& splitdim);

    uint32_t countPrim(LinearBvhNode* node, int n);
};