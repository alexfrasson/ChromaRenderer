#include "chroma-renderer/core/space-partition/BVH.h"
#include "chroma-renderer/core/utility/Stopwatch.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <list>

#define WIDEST_AXIS_SPLIT_ONLY

constexpr std::int32_t NUM_BINS = 8;
constexpr float EPSILON = 0.000001f;
constexpr std::int32_t MIN_LEAF_SIZE = 10;
// constexpr float KT = 1.0f; // Node traversal cost
// constexpr float KI = 1.5f; // Triangle intersection cost

bool primitiveCmpX(const BVHPrimitiveInfo& a, const BVHPrimitiveInfo& b)
{
    return (a.centroid.x < b.centroid.x);
}

bool primitiveCmpY(const BVHPrimitiveInfo& a, const BVHPrimitiveInfo& b)
{
    return (a.centroid.y < b.centroid.y);
}

bool primitiveCmpZ(const BVHPrimitiveInfo& a, const BVHPrimitiveInfo& b)
{
    return (a.centroid.z < b.centroid.z);
}

void BVH::clear()
{
    root.reset();
    lroot.clear();
    triangles.clear();
    id.clear();
}

std::uint32_t BVH::flattenBvh(const BvhNode& node, std::uint32_t& offset)
{
    lroot[offset].bbox = node.bbox;
    std::uint32_t myOffset{offset};
    offset++;
    if (node.is_leaf)
    {
        lroot[myOffset].n_primitives = static_cast<uint8_t>(node.end_id - node.start_id);
        lroot[myOffset].primitives_offset = static_cast<std::uint32_t>(node.start_id);
    }
    else
    {
        lroot[myOffset].axis = node.axis;
        lroot[myOffset].n_primitives = 0;
        flattenBvh(*node.child[0], offset);
        lroot[myOffset].second_child_offset = flattenBvh(*node.child[1], offset);
    }
    return myOffset;
}

std::size_t BVH::countPrim(LinearBvhNode* node, std::size_t n)
{
    if (node->n_primitives > 0)
    {
        return (node->n_primitives);
    }
    return (node->n_primitives + countPrim(&lroot[n + 1], n + 1) +
            countPrim(&lroot[node->second_child_offset], node->second_child_offset));
}

bool BVH::build(std::vector<std::unique_ptr<Mesh>>& m)
{
    if (m.empty())
    {
        return false;
    }

    Mesh& mesh = *m[0];

    std::cout.precision(6);
    std::cout << "Scene triangle count: " << mesh.t.size() << std::endl;
    std::cout << "Building bvh... " << std::endl;

    Stopwatch stopwatch;
    stopwatch.start();

    clear();

    //
    n_leafs = 0;
    n_nodes = 0;
    max_depth = 0;

    // Compute each triangle's bbox, it's centroid and the bbox of all triangles and of all centroids
    std::vector<BVHPrimitiveInfo> primitive;
    primitive.reserve(mesh.t.size());
    for (std::size_t i = 0; i < mesh.t.size(); i++)
    {
        BoundingBox bb;
        bb = BoundingBox();
        for (std::size_t dim = 0; dim < 3; dim++)
        {
            bb.expand(*mesh.t[i].getVertex(dim));
        }
        primitive.emplace_back(bb, i);
    }

    root = buildnode(0, primitive, 0, primitive.size());

    // tris.reserve(primitive.size());
    // for (size_t i = 0; i < primitive.size(); i++)
    //	tris.emplace_back(o[0].f[primitive[i].index]);
    triangles.reserve(primitive.size());
    for (std::size_t i = 0; i < primitive.size(); i++)
    {
        triangles.emplace_back(&mesh.t[primitive[i].index]);
    }

    // Flatten
    lroot.resize(n_nodes);
    std::uint32_t offset = 0;
    flattenBvh(*root, offset);

    stopwatch.stop();

    std::cout << "Node #" << std::endl
              << "    Leaf:        " << n_leafs << std::endl
              << "    Interior:    " << n_nodes - n_leafs << std::endl
              << "    Total:       " << n_nodes << std::endl
              << "Max depth:       " << max_depth << std::endl
              << "Avg. tris/leaf:  " << (float)triangles.size() / (float)n_leafs << std::endl
              << "Building time:   " << (float)stopwatch.elapsed_millis.count() / 1000.0f << "s" << std::endl
              << "Size:            " << sizeInBytes() / 1024 << "KB" << std::endl
              << "Done!" << std::endl;

    // std::cout << "NODESSSS: " << countPrim(lroot, 0) << std::endl;

    float sum = 0.f;
    for (size_t i = 1; i < n_nodes; i++)
    {
        if (lroot[i].n_primitives > 0)
        {
            sum += lroot[i].bbox.volume() / lroot[0].bbox.volume();
        }
    }
    std::cout << "Volume ratio: " << sum << std::endl;

    return true;
}

std::unique_ptr<BvhNode> BVH::buildNode(std::int32_t depth,
                                        std::vector<glm::vec3>& centroids,
                                        std::vector<BoundingBox>& bboxes,
                                        std::size_t startID,
                                        std::size_t endID)
{
    std::unique_ptr<BvhNode> node = std::make_unique<BvhNode>();
    n_nodes++;

    if (depth > max_depth)
    {
        max_depth = depth;
    }

    // Find bbox for the whole set of triangles and for the whole set of centroids
    BoundingBox centroidsbbox;
    BoundingBox trianglesbbox;
    for (std::size_t i = startID; i < endID; i++)
    {
        centroidsbbox.expand(centroids[id[i]]);
        trianglesbbox.expand(bboxes[id[i]].min);
        trianglesbbox.expand(bboxes[id[i]].max);
    }

    // Check if we should make this node a leaf
    std::size_t size = endID - startID;
    if (size <= MIN_LEAF_SIZE)
    {
        node->bbox = trianglesbbox;
        node->start_id = static_cast<std::int32_t>(startID);
        node->end_id = static_cast<std::int32_t>(endID);
        node->is_leaf = true;
        n_leafs++;
        return node;
    }

#ifdef WIDEST_AXIS_SPLIT_ONLY

    // Find widest axis
    std::int32_t widestDim = 0;
    {
        float widest = centroidsbbox.max.x - centroidsbbox.min.x;
        if (centroidsbbox.max.y - centroidsbbox.min.y > widest)
        {
            widest = centroidsbbox.max.y - centroidsbbox.min.y;
            widestDim = 1;
        }
        if (centroidsbbox.max.z - centroidsbbox.min.z > widest)
        {
            widestDim = 2;
        }
    }

    // Precompute constants. This constants will be used to calculate the each centroid's binid.
    // BinID[i] = k1[i] * (tsc[n][i] - k0[i])
    float k1 = (NUM_BINS * (1 - EPSILON)) / (centroidsbbox.max[widestDim] - centroidsbbox.min[widestDim]);
    float k0 = centroidsbbox.min[widestDim];

    // Bins for each axis
    std::int32_t bin[NUM_BINS];
    for (std::int32_t j = 0; j < NUM_BINS; j++)
    {
        bin[j] = 0;
    }

    // Bin's bounds
    BoundingBox binbound[NUM_BINS];

    for (std::size_t i = startID; i < endID; i++)
    {
        // Find the centroid'i''s binid on the axis 'j'
        auto binid = static_cast<std::int32_t>(truncf(k1 * (centroids[id[i]][widestDim] - k0)));
        bin[binid]++;
        // binbound[dim][binid].expand(centroids[id[i]]);
        binbound[binid].expand(bboxes[id[i]].max);
        binbound[binid].expand(bboxes[id[i]].min);
    }

    float minCost = std::numeric_limits<float>::infinity();
    std::int32_t minCostDim = widestDim;
    std::int32_t minCostBin = -1;

    for (std::int32_t i = 0; i < NUM_BINS; i++)
    {
        std::int32_t NL{0};
        std::int32_t NR{0};
        BoundingBox bbl;
        BoundingBox bbr;
        for (std::int32_t j = 0; j <= i; j++)
        {
            if (bin[j] > 0)
            {
                NL += bin[j];
                bbl.expand(binbound[j]);
            }
        }
        for (std::int32_t j = i + 1; j < NUM_BINS; j++)
        {
            if (bin[j] > 0)
            {
                NR += bin[j];
                bbr.expand(binbound[j]);
            }
        }

        float c = cost(bbl.surfaceArea(), static_cast<float>(NL), bbr.surfaceArea(), static_cast<float>(NR));
        if (c < minCost)
        {
            minCost = c;
            minCostBin = i;
        }

        bbl = BoundingBox();
        bbr = BoundingBox();
    }
#else

    // Precompute constants. This constants will be used to calculate the each centroid's binid.
    // BinID[i] = k1[i] * (tsc[n][i] - k0[i])
    float k1[3];
    float k0[3];
    for (std::int32_t i = 0; i < 3; i++)
    {
        k1[i] = (NUM_BINS * (1 - EPSILON)) / (centroidsbbox.max[i] - centroidsbbox.min[i]);
        k0[i] = centroidsbbox.min[i];
    }

    // Bins for each axis
    std::int32_t bin[3][NUM_BINS];
    for (std::int32_t i = 0; i < 3; i++)
    {
        for (std::int32_t j = 0; j < NUM_BINS; j++)
        {
            bin[i][j] = 0;
        }
    }

    // Bin's bounds
    BoundingBox binbound[3][NUM_BINS];

    for (std::int32_t i = startID; i < endID; i++)
    {
        // For each axis
        for (std::int32_t dim = 0; dim < 3; dim++)
        {
            // Find the centroid'i''s binid on the axis 'j'
            std::int32_t binid = truncf(k1[dim] * (centroids[id[i]][dim] - k0[dim]));
            bin[dim][binid]++;
            // binbound[dim][binid].expand(centroids[id[i]]);
            binbound[dim][binid].expand(bboxes[id[i]].max);
            binbound[dim][binid].expand(bboxes[id[i]].min);
        }
    }

    float minCost = std::numeric_limits<float>::infinity();
    std::int32_t minCostDim = -1;
    std::int32_t minCostBin = -1;

    for (std::int32_t dim = 0; dim < 3; dim++)
    {
        for (std::int32_t i = 0; i < NUM_BINS; i++)
        {
            std::int32_t NL, NR;
            BoundingBox bbl, bbr;
            NL = 0;
            NR = 0;
            for (std::int32_t j = 0; j <= i; j++)
            {
                if (bin[dim][j] > 0)
                {
                    NL += bin[dim][j];
                    bbl.expand(binbound[dim][j]);
                }
            }
            for (std::int32_t j = i + 1; j < NUM_BINS; j++)
            {
                if (bin[dim][j] > 0)
                {
                    NR += bin[dim][j];
                    bbr.expand(binbound[dim][j]);
                }
            }

            float c = cost(bbl.surfaceArea(), NL, bbr.surfaceArea(), NR);
            if (c < minCost)
            {
                minCost = c;
                minCostDim = dim;
                minCostBin = i;
            }

            bbl = BoundingBox();
            bbr = BoundingBox();
        }
    }

#endif

    // Check if not splitting is a better option
    if (minCost > cost(trianglesbbox.surfaceArea(), static_cast<float>(size)))
    {
        node->bbox = trianglesbbox;
        node->start_id = static_cast<std::int32_t>(startID);
        node->end_id = static_cast<std::int32_t>(endID);
        node->is_leaf = true;
        n_leafs++;
        return node;
    }

#ifdef WIDEST_AXIS_SPLIT_ONLY
    // Reorganize id array
    std::size_t mid = 0;
    for (std::size_t i = startID, j = endID - 1; i < endID && j >= startID && i <= j;)
    {
        // Find a triangle that is on the left but should be on the right
        for (std::int32_t binID = minCostBin - 1; i < endID && binID <= minCostBin && i != j; i++)
        {
            binID = static_cast<std::int32_t>(truncf(k1 * (centroids[id[i]][minCostDim] - k0)));
        }
        // Find a triangle that is on the right but should be on the left
        for (std::int32_t binID = minCostBin + 1; j >= startID && binID > minCostBin && j != i; j--)
        {
            binID = static_cast<std::int32_t>(truncf(k1 * (centroids[id[j]][minCostDim] - k0)));
        }
        // Where done
        if (i == j)
        {
            mid = i;
            break;
        }
        std::swap(id[i], id[j]);
    }
#else
    // Reorganize id array
    std::size_t mid;
    for (std::size_t i = startID, j = endID - 1; i < endID && j >= startID && i <= j;)
    {
        // Find a triangle that is on the left but should be on the right
        for (std::int32_t binID = minCostBin - 1; i < endID && binID <= minCostBin && i != j; i++)
        {
            binID = truncf(k1[minCostDim] * (centroids[id[i]][minCostDim] - k0[minCostDim]));
        }
        // Find a triangle that is on the right but should be on the left
        for (std::int32_t binID = minCostBin + 1; j >= startID && binID > minCostBin && j != i; j--)
        {
            binID = truncf(k1[minCostDim] * (centroids[id[j]][minCostDim] - k0[minCostDim]));
        }
        // Where done
        if (i == j)
        {
            mid = i;
            break;
        }
        std::swap(id[i], id[j]);
    }
#endif

    node->axis = static_cast<uint8_t>(minCostDim);
    node->bbox = trianglesbbox;
    node->start_id = static_cast<std::int32_t>(startID);
    node->end_id = static_cast<std::int32_t>(endID);
    node->is_leaf = false;

    node->child[0] = buildNode(depth + 1, centroids, bboxes, startID, mid);
    node->child[1] = buildNode(depth + 1, centroids, bboxes, mid, endID);

    return node;
}

std::unique_ptr<BvhNode> BVH::buildnode(std::int32_t depth,
                                        std::vector<BVHPrimitiveInfo>& primitive,
                                        std::size_t startID,
                                        std::size_t endID)
{
    std::unique_ptr<BvhNode> node = std::make_unique<BvhNode>();
    n_nodes++;

    if (depth > max_depth)
    {
        max_depth = depth;
    }

    // Find bbox for the whole set of triangles and for the whole set of centroids
    BoundingBox trianglesbbox;
    BoundingBox centroidsbbox;
    for (std::size_t i = startID; i < endID; i++)
    {
        centroidsbbox.expand(primitive[i].centroid);
        trianglesbbox.expand(primitive[i].bbox.min);
        trianglesbbox.expand(primitive[i].bbox.max);
    }

    // Check if we should make this node a leaf
    std::size_t size = endID - startID;
    if (size <= MIN_LEAF_SIZE)
    {
        node->bbox = trianglesbbox;
        node->start_id = static_cast<std::int32_t>(startID);
        node->end_id = static_cast<std::int32_t>(endID);
        node->is_leaf = true;
        n_leafs++;
        return node;
    }

    // Split
    std::int32_t splitdim{0};
    std::size_t splitindex{0};

    splitMidpoint(primitive, centroidsbbox, startID, endID, splitindex, splitdim);
    // splitMidpoint(primitive, trianglesbbox, startID, endID, splitindex, splitdim);
    // splitMedian(primitive, trianglesbbox, startID, endID, splitindex, splitdim);

    node->axis = static_cast<uint8_t>(splitdim);
    node->bbox = trianglesbbox;
    node->start_id = static_cast<std::int32_t>(startID);
    node->end_id = static_cast<std::int32_t>(endID);
    node->is_leaf = false;

    node->child[0] = buildnode(depth + 1, primitive, startID, splitindex);
    node->child[1] = buildnode(depth + 1, primitive, splitindex, endID);

    return node;
}

bool BVH::splitMidpoint(std::vector<BVHPrimitiveInfo>& primitive,
                        BoundingBox& trianglesbbox,
                        std::size_t startID,
                        std::size_t endID,
                        std::size_t& splitindex,
                        std::int32_t& splitdim)
{
    // Find widest axis
    std::int32_t widestDim = 0;
    {
        float widest = trianglesbbox.max.x - trianglesbbox.min.x;
        if (trianglesbbox.max.y - trianglesbbox.min.y > widest)
        {
            widest = trianglesbbox.max.y - trianglesbbox.min.y;
            widestDim = 1;
        }
        if (trianglesbbox.max.z - trianglesbbox.min.z > widest)
        {
            widestDim = 2;
        }
    }

    // Find midpoint
    float midpoint = (trianglesbbox.max[widestDim] + trianglesbbox.min[widestDim]) / 2.0f;

    // Reorganize array
    std::size_t mid = 0;
    for (std::size_t i = startID, j = endID - 1; i < endID && j >= startID && i <= j;)
    {
        // Find a triangle that is on the left but should be on the right
        while (primitive[i].centroid[widestDim] <= midpoint && i < j)
        {
            i++;
        }
        // Find a triangle that is on the right but should be on the left
        while (primitive[j].centroid[widestDim] > midpoint && i < j)
        {
            j--;
        }
        // Where done
        if (i == j)
        {
            mid = i;
            break;
        }
        // Swap
        BVHPrimitiveInfo aux = primitive[i];
        primitive[i] = primitive[j];
        primitive[j] = aux;
    }

    // This will generate an empty nodes, should do something about it
    // if (mid == startID || mid == endID - 1)
    //	std::cout << "OOPS" << std::endl;

    splitindex = mid;
    splitdim = widestDim;

    return true;
}

bool BVH::splitMedian(std::vector<BVHPrimitiveInfo>& primitive,
                      BoundingBox& trianglesbbox,
                      std::size_t startID,
                      std::size_t endID,
                      std::size_t& splitindex,
                      std::int32_t& splitdim)
{
    // Find widest axis
    std::int32_t widestDim = 0;
    {
        float widest = trianglesbbox.max.x - trianglesbbox.min.x;
        if (trianglesbbox.max.y - trianglesbbox.min.y > widest)
        {
            widest = trianglesbbox.max.y - trianglesbbox.min.y;
            widestDim = 1;
        }
        if (trianglesbbox.max.z - trianglesbbox.min.z > widest)
        {
            widestDim = 2;
        }
    }

    // Find median
    auto begin = primitive.begin();
    std::advance(begin, startID);
    auto end = primitive.begin();
    std::advance(end, endID);
    if (widestDim == 0)
    {
        std::sort(begin, end, primitiveCmpX);
    }
    else if (widestDim == 1)
    {
        std::sort(begin, end, primitiveCmpY);
    }
    else if (widestDim == 2)
    {
        std::sort(begin, end, primitiveCmpZ);
    }

    std::size_t mid = (startID + endID) / 2;

    splitindex = mid;
    splitdim = widestDim;

    return true;
}

float BVH::cost(float saL, float nL, float saR, float nR)
{
    return (saL * nL + saR * nR);
    // return (KT + KI * (saL * nL + saR * nR));
}

float BVH::cost(float sa, float n)
{
    return (sa * n);
}

size_t BVH::sizeInBytes()
{
    size_t size = sizeof(BVH);
    size += sizeof(Triangle*) * triangles.size(); // NOLINT(bugprone-sizeof-expression)
    size += sizeof(std::int32_t) * id.size();
    size += sizeof(BvhNode) * n_nodes;
    size += sizeof(LinearBvhNode) * n_nodes;
    return size;
}