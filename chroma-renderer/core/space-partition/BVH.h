#pragma once

#include "chroma-renderer/core/space-partition/ISpacePartitioningStructure.h"
#include "chroma-renderer/core/types/Mesh.h"

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
    BVHPrimitiveInfo(BoundingBox a_bbox, int a_index)
    {
        centroid = a_bbox.centroid();
        bbox = a_bbox;
        index = a_index;
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
    bool build(std::vector<Mesh*>& m);
    BvhNode* buildnode(int depth, std::vector<BVHPrimitiveInfo>& primitive, int startID, int endID);
    BvhNode* buildNode(int depth,
                       std::vector<glm::vec3>& centroids,
                       std::vector<BoundingBox>& bboxes,
                       int startID,
                       int endID);
    float cost(float saL, float nL, float saR, float nR);
    float cost(float sa, float n);

    size_t sizeInBytes(void);
    void abort(void)
    {
    }

    unsigned int nLeafs;
    unsigned int nNodes;
    int maxDepth;

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