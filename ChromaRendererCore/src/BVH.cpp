#include <BVH.h>
#include <Stopwatch.h>
#include <RTUtils.h>

#include <iostream>
#include <cmath>
#include <list>
#include <algorithm>

#define NUM_BINS		8
#define EPSILON			0.000001f
#define MIN_LEAF_SIZE	10


#define KT 1	// Node traversal cost
#define KI 1.5f	// Triangle intersection cost

#define WIDEST_AXIS_SPLIT_ONLY

struct BvhNodeToDo
{
	BvhNodeToDo()
		: node(NULL)
	{
	}
	BvhNodeToDo(BvhNode* node, float tmin, float tmax)
	{
		this->node = node;
		this->tmin = tmin;
		this->tmax = tmax;
	}
	BvhNode* node;
	float tmin, tmax;
};



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


BVH::BVH()
	: root(NULL)
{
}
BVH::~BVH()
{
	clear();
}

void BVH::clear()
{
	free(root);
	delete[] lroot;
	tris.clear();
	triangles.clear();
	id.clear();
}

BvhNode* BVH::free(BvhNode* node)
{
	if (node == NULL)
		return NULL;

	if (!node->isLeaf)
	{
		if (node->child[0] != NULL)
			node->child[0] = free(node->child[0]);
		if (node->child[1] != NULL)
			node->child[1] = free(node->child[1]);
	}
	delete node;
	return NULL;
}

int BVH::flattenBvh(BvhNode* node, int& offset)
{
	lroot[offset].bbox = node->bbox;
	int myOffset = offset;
	offset++;
	if (node->isLeaf)
	{
		lroot[myOffset].nPrimitives = node->endID - node->startID;
		lroot[myOffset].primitivesOffset = node->startID;
	}
	else
	{
		lroot[myOffset].axis = node->axis;
		lroot[myOffset].nPrimitives = 0;
		flattenBvh(node->child[0], offset);
		lroot[myOffset].secondChildOffset = flattenBvh(node->child[1], offset);
	}
	return myOffset;
}

uint32_t BVH::countPrim(LinearBvhNode* node, int n)
{
	if (node->nPrimitives > 0)
		return (node->nPrimitives);
	return (node->nPrimitives + countPrim(&lroot[n + 1], n + 1) + countPrim(&lroot[node->secondChildOffset], node->secondChildOffset));
}

bool BVH::build(std::vector<Object>& o)
{
	if (o.empty())
		return false;

	std::cout.precision(6);
	std::cout << "Scene triangle count: " << o[0].f.size() << std::endl;
	std::cout << "Building bvh... " << std::endl;

	Stopwatch stopwatch;
	stopwatch.start();

	clear();

	//
	nLeafs = 0;
	nNodes = 0;
	maxDepth = 0;

	// Compute each triangle's bbox, it's centroid and the bbox of all triangles and of all centroids
	std::vector<BVHPrimitiveInfo> primitive;
	primitive.reserve(o[0].f.size());
	for (int i = 0; i < o[0].f.size(); i++)
	{
		BoundingBox bb;
		bb = BoundingBox();
		for (int dim = 0; dim < 3; dim++)
			bb.expand(o[0].f[i].v[dim]);
		primitive.emplace_back(bb, i);
	}

	root = buildnode(0, primitive, 0, primitive.size());

	tris.reserve(primitive.size());
	for (size_t i = 0; i < primitive.size(); i++)
		tris.emplace_back(o[0].f[primitive[i].index]);

	// Flatten
	lroot = new LinearBvhNode[nNodes];
	int offset = 0;
	flattenBvh(root, offset);


	stopwatch.stop();

	std::cout << "Node #"
		<< std::endl
		<< "    Leaf:        " << nLeafs
		<< std::endl
		<< "    Interior:    " << nNodes - nLeafs
		<< std::endl
		<< "    Total:       " << nNodes
		<< std::endl
		<< "Max depth:       " << maxDepth
		<< std::endl
		<< "Avg. tris/leaf:  " << (float)tris.size() / (float)nLeafs
		<< std::endl
		<< "Building time:   " << stopwatch.elapsedMillis / 1000.0 << "s"
		<< std::endl
		<< "Done!"
		<< std::endl;

	//std::cout << "NODESSSS: " << countPrim(lroot, 0) << std::endl;

	float sum = 0.f;
	for (size_t i = 1; i < nNodes; i++)
	{
		if (lroot[i].nPrimitives > 0)
		{
			sum += lroot[i].bbox.volume() / lroot[0].bbox.volume();
		}
	}
	std::cout << "Volume ratio: " << sum << std::endl;

	return true;
}
bool BVH::build(std::vector<Mesh*>& m)
{
	if (m.empty())
		return false;


	Mesh &mesh = *m[0];

	std::cout.precision(6);
	std::cout << "Scene triangle count: " << mesh.t.size() << std::endl;
	std::cout << "Building bvh... " << std::endl;

	Stopwatch stopwatch;
	stopwatch.start();

	clear();

	//
	nLeafs = 0;
	nNodes = 0;
	maxDepth = 0;

	// Compute each triangle's bbox, it's centroid and the bbox of all triangles and of all centroids
	std::vector<BVHPrimitiveInfo> primitive;
	primitive.reserve(mesh.t.size());
	for (int i = 0; i < mesh.t.size(); i++)
	{
		BoundingBox bb;
		bb = BoundingBox();
		for (int dim = 0; dim < 3; dim++)
			bb.expand(*mesh.t[i].getVertex(dim));
		primitive.emplace_back(bb, i);
	}

	root = buildnode(0, primitive, 0, primitive.size());

	//tris.reserve(primitive.size());
	//for (size_t i = 0; i < primitive.size(); i++)
	//	tris.emplace_back(o[0].f[primitive[i].index]);
	triangles.reserve(primitive.size());
	for (size_t i = 0; i < primitive.size(); i++)
		triangles.emplace_back(&mesh.t[primitive[i].index]);

	// Flatten
	lroot = new LinearBvhNode[nNodes];
	int offset = 0;
	flattenBvh(root, offset);


	stopwatch.stop();

	std::cout << "Node #"
		<< std::endl
		<< "    Leaf:        " << nLeafs
		<< std::endl
		<< "    Interior:    " << nNodes - nLeafs
		<< std::endl
		<< "    Total:       " << nNodes
		<< std::endl
		<< "Max depth:       " << maxDepth
		<< std::endl
		<< "Avg. tris/leaf:  " << (float)triangles.size() / (float)nLeafs
		<< std::endl
		<< "Building time:   " << stopwatch.elapsedMillis / 1000.0 << "s"
		<< std::endl
		<< "Size:            " << sizeInBytes() / 1024 << "KB"
		<< std::endl
		<< "Done!"
		<< std::endl;

	//std::cout << "NODESSSS: " << countPrim(lroot, 0) << std::endl;

	float sum = 0.f;
	for (size_t i = 1; i < nNodes; i++)
	{
		if (lroot[i].nPrimitives > 0)
		{
			sum += lroot[i].bbox.volume() / lroot[0].bbox.volume();
		}
	}
	std::cout << "Volume ratio: " << sum << std::endl;

	return true;
}
BvhNode* BVH::buildNode(int depth, std::vector<glm::vec3>& centroids, std::vector<BoundingBox>& bboxes, int startID, int endID)
{
	BvhNode* node = new BvhNode();
	nNodes++;

	if (depth > maxDepth)
		maxDepth = depth;

	// Find bbox for the whole set of triangles and for the whole set of centroids
	BoundingBox centroidsbbox;
	BoundingBox trianglesbbox;
	for (int i = startID; i < endID; i++)
	{
		centroidsbbox.expand(centroids[id[i]]);
		trianglesbbox.expand(bboxes[id[i]].min);
		trianglesbbox.expand(bboxes[id[i]].max);
	}

	// Check if we should make this node a leaf
	int size = endID - startID;
	if (size <= MIN_LEAF_SIZE)
	{
		node->bbox = trianglesbbox;
		node->startID = startID;
		node->endID = endID;
		node->isLeaf = true;
		nLeafs++;
		return node;
	}

#ifdef WIDEST_AXIS_SPLIT_ONLY

	// Find widest axis
	int widestDim = 0;
	{
		float widest = centroidsbbox.max.x - centroidsbbox.min.x;
		if (centroidsbbox.max.y - centroidsbbox.min.y > widest)
		{
			widest = centroidsbbox.max.y - centroidsbbox.min.y;
			widestDim = 1;
		}
		if (centroidsbbox.max.z - centroidsbbox.min.z > widest)
		{
			widest = centroidsbbox.max.z - centroidsbbox.min.z;
			widestDim = 2;
		}
	}

	// Precompute constants. This constants will be used to calculate the each centroid's binid.
	// BinID[i] = k1[i] * (tsc[n][i] - k0[i])
	float k1 = (NUM_BINS * (1 - EPSILON)) / (centroidsbbox.max[widestDim] - centroidsbbox.min[widestDim]);
	float k0 = centroidsbbox.min[widestDim];

	// Bins for each axis
	int bin[NUM_BINS];
	for (int j = 0; j < NUM_BINS; j++)
		bin[j] = 0;

	// Bin's bounds
	BoundingBox binbound[NUM_BINS];

	for (int i = startID; i < endID; i++)
	{
		// Find the centroid'i''s binid on the axis 'j'
		int binid = truncf(k1 * (centroids[id[i]][widestDim] - k0));
		bin[binid]++;
		//binbound[dim][binid].expand(centroids[id[i]]);
		binbound[binid].expand(bboxes[id[i]].max);
		binbound[binid].expand(bboxes[id[i]].min);
	}

	float minCost = std::numeric_limits<float>::infinity();
	int minCostDim = widestDim;
	int minCostBin = -1;

	for (int i = 0; i < NUM_BINS; i++)
	{
		int NL, NR;
		BoundingBox bbl, bbr;
		NL = 0; NR = 0;
		for (int j = 0; j <= i; j++)
		{
			if (bin[j] > 0)
			{
				NL += bin[j];
				bbl.expand(binbound[j]);
			}
		}
		for (int j = i + 1; j < NUM_BINS; j++)
		{
			if (bin[j] > 0)
			{
				NR += bin[j];
				bbr.expand(binbound[j]);
			}
		}

		float c = cost(bbl.surfaceArea(), NL, bbr.surfaceArea(), NR);
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
	for (int i = 0; i < 3; i++)
	{
		k1[i] = (NUM_BINS * (1 - EPSILON)) / (centroidsbbox.max[i] - centroidsbbox.min[i]);
		k0[i] = centroidsbbox.min[i];
	}

	// Bins for each axis
	int bin[3][NUM_BINS];
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < NUM_BINS; j++)
			bin[i][j] = 0;

	// Bin's bounds
	BoundingBox binbound[3][NUM_BINS];

	for (int i = startID; i < endID; i++)
	{
		// For each axis
		for (int dim = 0; dim < 3; dim++)
		{
			// Find the centroid'i''s binid on the axis 'j'
			int binid = truncf(k1[dim] * (centroids[id[i]][dim] - k0[dim]));
			bin[dim][binid]++;
			//binbound[dim][binid].expand(centroids[id[i]]);
			binbound[dim][binid].expand(bboxes[id[i]].max);
			binbound[dim][binid].expand(bboxes[id[i]].min);
		}
	}

	float minCost = std::numeric_limits<float>::infinity();
	int minCostDim = -1;
	int minCostBin = -1;

	for (int dim = 0; dim < 3; dim++)
	{
		for (int i = 0; i < NUM_BINS; i++)
		{
			int NL, NR;
			BoundingBox bbl, bbr;
			NL = 0; NR = 0;
			for (int j = 0; j <= i; j++)
			{
				if (bin[dim][j] > 0)
				{
					NL += bin[dim][j];
					bbl.expand(binbound[dim][j]);
				}
			}
			for (int j = i + 1; j < NUM_BINS; j++)
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
	if (minCost > cost(trianglesbbox.surfaceArea(), size))
	{
		node->bbox = trianglesbbox;
		node->startID = startID;
		node->endID = endID;
		node->isLeaf = true;
		nLeafs++;
		return node;
	}

#ifdef WIDEST_AXIS_SPLIT_ONLY
	// Reorganize id array
	int mid;
	for (int i = startID, j = endID - 1; i < endID && j >= startID && i <= j;)
	{
		// Find a triangle that is on the left but should be on the right
		for (int binID = minCostBin - 1; i < endID && binID <= minCostBin && i != j; i++)
			binID = truncf(k1 * (centroids[id[i]][minCostDim] - k0));
		// Find a triangle that is on the right but should be on the left
		for (int binID = minCostBin + 1; j >= startID && binID > minCostBin && j != i; j--)
			binID = truncf(k1 * (centroids[id[j]][minCostDim] - k0));
		// Where done
		if (i == j)
		{
			mid = i;
			break;
		}
		// Swapp
		int aux = id[i];
		id[i] = id[j];
		id[j] = aux;
	}
#else
	// Reorganize id array
	int mid;
	for (int i = startID, j = endID - 1; i < endID && j >= startID && i <= j;)
	{
		// Find a triangle that is on the left but should be on the right
		for (int binID = minCostBin - 1; i < endID && binID <= minCostBin && i != j; i++)
			binID = truncf(k1[minCostDim] * (centroids[id[i]][minCostDim] - k0[minCostDim]));
		// Find a triangle that is on the right but should be on the left
		for (int binID = minCostBin + 1; j >= startID && binID > minCostBin && j != i; j--)
			binID = truncf(k1[minCostDim] * (centroids[id[j]][minCostDim] - k0[minCostDim]));
		// Where done
		if (i == j)
		{
			mid = i;
			break;
		}
		// Swapp
		int aux = id[i];
		id[i] = id[j];
		id[j] = aux;
	}
#endif

	node->axis = minCostDim;
	node->bbox = trianglesbbox;
	node->startID = startID;
	node->endID = endID;
	node->isLeaf = false;

	node->child[0] = buildNode(depth + 1, centroids, bboxes, startID, mid);
	node->child[1] = buildNode(depth + 1, centroids, bboxes, mid, endID);

	return node;
}
BvhNode* BVH::buildnode(int depth, std::vector<BVHPrimitiveInfo>& primitive, int startID, int endID)
{
	BvhNode* node = new BvhNode();
	nNodes++;

	if (depth > maxDepth)
		maxDepth = depth;

	// Find bbox for the whole set of triangles and for the whole set of centroids
	BoundingBox trianglesbbox;
	BoundingBox centroidsbbox;
	for (int i = startID; i < endID; i++)
	{
		centroidsbbox.expand(primitive[i].centroid);
		trianglesbbox.expand(primitive[i].bbox.min);
		trianglesbbox.expand(primitive[i].bbox.max);
	}

	// Check if we should make this node a leaf
	int size = endID - startID;
	if (size <= MIN_LEAF_SIZE)
	{
		node->bbox = trianglesbbox;
		node->startID = startID;
		node->endID = endID;
		node->isLeaf = true;
		nLeafs++;
		return node;
	}

	// Split
	int splitdim;
	int splitindex;
	
	splitMidpoint(primitive, centroidsbbox, startID, endID, splitindex, splitdim);
	//splitMidpoint(primitive, trianglesbbox, startID, endID, splitindex, splitdim);
	//splitMedian(primitive, trianglesbbox, startID, endID, splitindex, splitdim);

	node->axis = splitdim;
	node->bbox = trianglesbbox;
	node->startID = startID;
	node->endID = endID;
	node->isLeaf = false;

	node->child[0] = buildnode(depth + 1, primitive, startID, splitindex);
	node->child[1] = buildnode(depth + 1, primitive, splitindex, endID);

	return node;
}
bool BVH::splitMidpoint(std::vector<BVHPrimitiveInfo>& primitive, BoundingBox& trianglesbbox, int startID, int endID, int& splitindex, int& splitdim)
{
	// Find widest axis
	int widestDim = 0;
	{
		float widest = trianglesbbox.max.x - trianglesbbox.min.x;
		if (trianglesbbox.max.y - trianglesbbox.min.y > widest)
		{
			widest = trianglesbbox.max.y - trianglesbbox.min.y;
			widestDim = 1;
		}
		if (trianglesbbox.max.z - trianglesbbox.min.z > widest)
		{
			widest = trianglesbbox.max.z - trianglesbbox.min.z;
			widestDim = 2;
		}
	}

	// Find midpoint
	float midpoint = (trianglesbbox.max[widestDim] + trianglesbbox.min[widestDim]) / 2.0f;

	// Reorganize array
	int mid;
	for (int i = startID, j = endID - 1; i < endID && j >= startID && i <= j;)
	{
		// Find a triangle that is on the left but should be on the right
		while (primitive[i].centroid[widestDim] <= midpoint && i < j)
			i++;
		// Find a triangle that is on the right but should be on the left
		while (primitive[j].centroid[widestDim] > midpoint && i < j)
			j--;
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
	//if (mid == startID || mid == endID - 1)
	//	std::cout << "OOPS" << std::endl;

	splitindex = mid;
	splitdim = widestDim;

	return true;
}
bool BVH::splitMedian(std::vector<BVHPrimitiveInfo>& primitive, BoundingBox& trianglesbbox, int startID, int endID, int& splitindex, int& splitdim)
{
	// Find widest axis
	int widestDim = 0;
	{
		float widest = trianglesbbox.max.x - trianglesbbox.min.x;
		if (trianglesbbox.max.y - trianglesbbox.min.y > widest)
		{
			widest = trianglesbbox.max.y - trianglesbbox.min.y;
			widestDim = 1;
		}
		if (trianglesbbox.max.z - trianglesbbox.min.z > widest)
		{
			widest = trianglesbbox.max.z - trianglesbbox.min.z;
			widestDim = 2;
		}
	}

	// Find median
	auto begin = primitive.begin();
	std::advance(begin, startID);
	auto end = primitive.begin();
	std::advance(end, endID);
	if(widestDim == 0)
		std::sort(begin, end, primitiveCmpX);
	else if(widestDim == 1)
		std::sort(begin, end, primitiveCmpY);
	else if(widestDim == 2)
		std::sort(begin, end, primitiveCmpZ);

	
	int mid = (startID + endID) / 2;

	splitindex = mid;
	splitdim = widestDim;

	return true;
}


float BVH::cost(float saL, float nL, float saR, float nR)
{
	return (saL * nL + saR * nR);
	//return (KT + KI * (saL * nL + saR * nR));
}
float BVH::cost(float sa, float n)
{
	return (sa * n);
}

bool cmp(const BvhNodeToDo& a, const BvhNodeToDo& b)
{
	return a.tmin < b.tmin;
}

long BVH::sizeInBytes(void)
{
	long size = sizeof(BVH);
	size += sizeof(Face) * tris.size();
	size += sizeof(Triangle*) * triangles.size();
	size += sizeof(int) * id.size();
	size += sizeof(BvhNode) * nNodes;
	size += sizeof(LinearBvhNode) * nNodes;
	return size;
}

bool BVH::intersect(Ray& r, Intersection& intersection) const
{
	if (lroot == NULL)
		return false;

	bool hit = false;
	glm::vec3 origin = r.origin;
	glm::vec3 invRayDir = 1.f / r.direction;
	glm::bvec3 dirIsNeg(invRayDir.x < 0, invRayDir.y < 0, invRayDir.z < 0);

	uint32_t todoOffset = 0;
	uint32_t nodeNum = 0;
	uint32_t todo[64];

	intersection.distance = std::numeric_limits<float>::max();
	//float tmin = 0, tmax = std::numeric_limits<float>::max();

	//int calls = 0;

	while (true)
	{
		const LinearBvhNode *node = &lroot[nodeNum];

		//calls++;

		if (RTUtils::hitBoundingBoxSlab(node->bbox, r, invRayDir, dirIsNeg, r.mint, r.maxt))
		{
			if (node->nPrimitives > 0)
			{
				// Intersect primitives
				for (int i = node->primitivesOffset; i < node->nPrimitives + node->primitivesOffset; i++)
				{
					//float t;
					//if (RTUtils::intersectRayTriangleMollerTrumbore(r, tris[id[i]], t))
					if (triangles[i]->intersect(r, intersection))
					//if (RTUtils::intersectRayTriangleMollerTrumbore(r, tris[i], t))
					{
						//if (t < intersection.distance)
						//{
							//intersection.triangle = &tris[id[i]];
							//intersection.triangle = &tris[i];
							//intersection.distance = t;
							//tmax = t;
							hit = true;
						//}
					}
				}
				if (todoOffset == 0)
					break;
				nodeNum = todo[--todoOffset];
			}
			else
			{
				if (dirIsNeg[node->axis])
				{
					todo[todoOffset++] = nodeNum + 1;
					nodeNum = node->secondChildOffset;
				}
				else
				{
					todo[todoOffset++] = node->secondChildOffset;
					nodeNum = nodeNum + 1;
				}
			}
		}
		else
		{
			if (todoOffset == 0)
				break;
			nodeNum = todo[--todoOffset];
		}
	}

	//if (calls > 1)
	//	std::cout << "calls " << calls << std::endl;

	return hit;
}
/*bool BVH::intersect(const Ray& r, Intersection& intersection) const
{
	if (lroot == NULL) 
		return false;
	
	bool hit = false;
	glm::vec3 origin = r.origin;
	glm::vec3 invRayDir = 1.f / r.direction;
	glm::bvec3 dirIsNeg(invRayDir.x < 0, invRayDir.y < 0, invRayDir.z < 0);
	
	uint32_t todoOffset = 0;
	uint32_t nodeNum = 0;
	uint32_t todo[64];

	intersection.distance = std::numeric_limits<float>::max();
	float tmin = 0, tmax = std::numeric_limits<float>::max();

	int calls = 0;

	while (true) 
	{
		const LinearBvhNode *node = &lroot[nodeNum];

		calls++;

		if (RTUtils::hitBoundingBoxSlab(node->bbox, r, invRayDir, dirIsNeg, tmin, tmax))
		{
			if (node->nPrimitives > 0)
			{
				// Intersect primitives
				for (int i = node->primitivesOffset; i < node->nPrimitives + node->primitivesOffset; i++)
				{
					float t;
					//if (RTUtils::intersectRayTriangleMollerTrumbore(r, tris[id[i]], t))
					if (RTUtils::intersectRayTriangleMollerTrumbore(r, tris[i], t))
					{
						if (t < intersection.distance)
						{
							//intersection.triangle = &tris[id[i]];
							intersection.triangle = &tris[i];
							intersection.distance = t;
							tmax = t;
							hit = true;
						}
					}
				}
				if (todoOffset == 0)
					break;
				nodeNum = todo[--todoOffset];
			}
			else
			{
				if (dirIsNeg[node->axis]) 
				{
					todo[todoOffset++] = nodeNum + 1;
					nodeNum = node->secondChildOffset;
				}
				else 
				{
					todo[todoOffset++] = node->secondChildOffset;
					nodeNum = nodeNum + 1;
				}
			}
		}
		else 
		{
			if (todoOffset == 0) 
				break;
			nodeNum = todo[--todoOffset];
		}
	}

	//if (calls > 1)
	//	std::cout << "calls " << calls << std::endl;

	return hit;
}*/

bool BVH::intersectF(const Ray& r, Intersection& intersection, float& nNodeHitsNormalized) const
{
	if (lroot == NULL)
		return false;

	nNodeHitsNormalized = 0;
	bool hit = false;
	glm::vec3 origin = r.origin;
	glm::vec3 invRayDir = 1.f / r.direction;
	glm::bvec3 dirIsNeg(invRayDir.x < 0, invRayDir.y < 0, invRayDir.z < 0);

	uint32_t todoOffset = 0;
	uint32_t nodeNum = 0;
	uint32_t todo[64];

	bool leafhit = false;
	uint16_t nNodeTests = 0;
	uint16_t nPrimTests = 0;

	intersection.distance = std::numeric_limits<float>::max();
	float tmin = 0, tmax = std::numeric_limits<float>::max();

	while (true)
	{
		const LinearBvhNode *node = &lroot[nodeNum];

		nNodeTests++;
		if (RTUtils::hitBoundingBoxSlab(node->bbox, r, invRayDir, dirIsNeg, tmin, tmax))
		{
			nNodeHitsNormalized++;
			if (node->nPrimitives > 0)
			{
				leafhit = true;
				// Intersect primitives
				for (int i = node->primitivesOffset; i < node->nPrimitives + node->primitivesOffset; i++)
				{
					float t;
					nPrimTests++;
					if (RTUtils::intersectRayTriangleMollerTrumbore(r, tris[id[i]], t))
					{
						if (t < intersection.distance)
						{
							intersection.triangle = &tris[id[i]];
							intersection.distance = t;
							tmax = t;
							hit = true;
						}
					}
				}
				if (todoOffset == 0)
					break;
				nodeNum = todo[--todoOffset];
			}
			else
			{
				if (dirIsNeg[node->axis])
				{
					todo[todoOffset++] = nodeNum + 1;
					nodeNum = node->secondChildOffset;
				}
				else
				{
					todo[todoOffset++] = node->secondChildOffset;
					nodeNum = nodeNum + 1;
				}
			}
		}
		else
		{
			if (todoOffset == 0)
				break;
			nodeNum = todo[--todoOffset];
		}
	}

	nNodeHitsNormalized /= nNodes;
	//if (leafhit)
	//	nNodeHitsNormalized = 1.f;
	return hit;
}

bool BVH::intersectF(const Ray& r, Intersection& intersection) const
{
	if (root == NULL)
		return false;

	// Intersect ray with the tree's boundingbox
	float tmin, tmax;
	if (!RTUtils::hitBoundingBox(r, root->bbox, tmin, tmax))
		return false;

	std::vector<BvhNodeToDo> todo(maxDepth, BvhNodeToDo());
	int todoPos = -1;

	bool hit = false;
	const Face* hitf = NULL;
	intersection.distance = FLT_MAX;
	const BvhNode* n = root;
	glm::vec3 invRayDir = 1.f / r.direction;

	while (n != NULL)
	{
		//if (hitDistance < tmin)
		//	break;
		// Leaf
		if (n->isLeaf)
		{
			// Intersect primitives
			for (int i = n->startID; i < n->endID; i++)
			{
				float t;
				if (RTUtils::intersectRayTriangleMollerTrumbore(r, tris[i], t))
				{
					if (t < intersection.distance)
					{
						intersection.triangle = &tris[i];
						intersection.distance = t;
						hit = true;
					}
				}
			}
			do
			{
				// No more nodes to trace
				if (todoPos < 0)
				{
					n = NULL;
					break;
				}
				// Get node from todo list
				n = todo[todoPos].node;
				tmin = todo[todoPos].tmin;
				tmax = todo[todoPos].tmax;
				--todoPos;
			} while (intersection.distance <= tmin);
		}
		else
		{
			// Get children tmin and tmax, add them to the todolist and sort the list
			//int child = ;
			float tminchild0, tmaxchild0, tminchild1, tmaxchild1;
			//bool hitchild0 = RTUtils::hitBoundingBox(r, n->child[0]->bbox, tminchild0, tmaxchild0);
			//bool hitchild1 = RTUtils::hitBoundingBox(r, n->child[1]->bbox, tminchild1, tmaxchild1);
			bool hitchild0 = RTUtils::hitBoundingBoxSlab(n->child[0]->bbox, r, invRayDir, tminchild0, tmaxchild0);
			bool hitchild1 = RTUtils::hitBoundingBoxSlab(n->child[1]->bbox, r, invRayDir, tminchild1, tmaxchild1);
			if (hitchild0 && hitchild1)
			{
				if (tminchild0 < tminchild1)
				{
					++todoPos;
					todo[todoPos].node = n->child[1];
					todo[todoPos].tmin = tminchild1;
					todo[todoPos].tmax = tmaxchild1;
					n = n->child[0];
					tmin = tminchild0;
					tmax = tmaxchild0;
				}
				else
				{
					++todoPos;
					todo[todoPos].node = n->child[0];
					todo[todoPos].tmin = tminchild0;
					todo[todoPos].tmax = tmaxchild0;
					n = n->child[1];
					tmin = tminchild1;
					tmax = tmaxchild1;
				}
			}
			else if (hitchild0)
			{
				n = n->child[0];
				tmin = tminchild0;
				tmax = tmaxchild0;
			}
			else if (hitchild1)
			{
				n = n->child[1];
				tmin = tminchild1;
				tmax = tmaxchild1;
			}
			else
			{
				do
				{
					// No more nodes to trace
					if (todoPos < 0)
					{
						n = NULL;
						break;
					}
					// Get node from todo list
					n = todo[todoPos].node;
					tmin = todo[todoPos].tmin;
					tmax = todo[todoPos].tmax;
					--todoPos;
				} while (intersection.distance <= tmin);
			}
		}
	}
	return hit;
}


uint32_t BVH::nIntersectedNodes(const Ray& r)
{
	if (lroot == NULL)
		return 0;

	uint32_t nHits = 0;
	glm::vec3 invRayDir = 1.f / r.direction;
	glm::bvec3 dirIsNeg(invRayDir.x < 0, invRayDir.y < 0, invRayDir.z < 0);

	uint32_t todoOffset = 0;
	uint32_t nodeNum = 0;
	uint32_t todo[64];

	float tmin = 0, tmax = std::numeric_limits<float>::max();

	while (true)
	{
		const LinearBvhNode *node = &lroot[nodeNum];

		if (RTUtils::hitBoundingBoxSlab(node->bbox, r, invRayDir, dirIsNeg, tmin, tmax))
		{
			nHits++;

			if (node->nPrimitives > 0)
			{
				if (todoOffset == 0)
					break;
				nodeNum = todo[--todoOffset];
			}
			else
			{
				if (dirIsNeg[node->axis])
				{
					todo[todoOffset++] = nodeNum + 1;
					nodeNum = node->secondChildOffset;
				}
				else
				{
					todo[todoOffset++] = node->secondChildOffset;
					nodeNum = nodeNum + 1;
				}
			}
		}
		else
		{
			if (todoOffset == 0)
				break;
			nodeNum = todo[--todoOffset];
		}
	}
	return nHits;
}