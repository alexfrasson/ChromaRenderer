#pragma once

#include "chroma-renderer/core/space-partition/i_space_partitioning_structure.h"
#include "chroma-renderer/core/types/mesh.h"

#include <memory>
#include <vector>

struct BvhNode
{
    BoundingBox bbox{};
    std::unique_ptr<BvhNode> child[2] = {nullptr, nullptr};
    std::int32_t start_id{0};
    std::int32_t end_id{0};
    bool is_leaf{false};
    uint8_t axis{0};
};

// NOLINTNEXTLINE (cppcoreguidelines-pro-type-member-init, hicpp-member-init)
struct LinearBvhNode
{
    BoundingBox bbox{};
    union {

        std::uint32_t primitives_offset;   // Leaf
        std::uint32_t second_child_offset; // Interior
    };
    std::uint8_t n_primitives{0}; // 0 -> interior node
    std::uint8_t axis{0};
    std::uint8_t pad[2];
};

struct BVHPrimitiveInfo
{
    BVHPrimitiveInfo(const BoundingBox a_bbox, const std::size_t a_index)
        : centroid{a_bbox.centroid()}, bbox{a_bbox}, index{a_index}
    {
    }
    glm::vec3 centroid;
    BoundingBox bbox;
    std::size_t index;
};

class BVH : public ISpacePartitioningStructure
{
  public:
    BVH() = default;

    void clear();
    std::uint32_t flattenBvh(const BvhNode& node, std::uint32_t& offset);
    bool build(std::vector<std::unique_ptr<Mesh>>& m) override;
    std::unique_ptr<BvhNode> buildnode(std::int32_t depth,
                                       std::vector<BVHPrimitiveInfo>& primitive,
                                       std::size_t start_id,
                                       std::size_t end_id);
    std::unique_ptr<BvhNode> buildNode(std::int32_t depth,
                                       std::vector<glm::vec3>& centroids,
                                       std::vector<BoundingBox>& bboxes,
                                       std::size_t start_id,
                                       std::size_t end_id);
    static float cost(float sa_l, float n_l, float sa_r, float n_r);
    static float cost(float sa, float n);

    size_t sizeInBytes() override;

    static bool splitMidpoint(std::vector<BVHPrimitiveInfo>& primitive,
                              BoundingBox& trianglesbbox,
                              std::size_t start_id,
                              std::size_t end_id,
                              std::size_t& splitindex,
                              std::int32_t& splitdim);
    static bool splitMedian(std::vector<BVHPrimitiveInfo>& primitive,
                            BoundingBox& trianglesbbox,
                            std::size_t start_id,
                            std::size_t end_id,
                            std::size_t& splitindex,
                            std::int32_t& splitdim);

    std::size_t countPrim(LinearBvhNode* node, std::size_t n);

    std::uint32_t n_leafs{0};
    std::uint32_t n_nodes{0};
    std::int32_t max_depth{0};
    std::vector<Triangle*> triangles{};
    std::vector<std::size_t> id{};
    std::unique_ptr<BvhNode> root{};
    std::vector<LinearBvhNode> lroot{};
};