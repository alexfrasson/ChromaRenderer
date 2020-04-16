#include <KDTree.h>
#include <RTUtils.h>
#include <Stopwatch.h>

#include <algorithm>
#include <fstream>
#include <glm/gtx/string_cast.hpp>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>

#define MAX_DEPTH \
    64 // This is only used in the intersection method, so the stack doesn't have to be dynamically allocated
//#define MIN_LEAF_SIZE	5
//#define EMPTYSPACEBONUS 1.f

#define X 0
#define Y 1
#define Z 2

//#define KT 1	// Node traversal cost
//#define KI 1.5f	// Triangle intersection cost

#define EPSILON 0.000001f

// bool clipTriangleToBoxSH(const Face& t, const BoundingBox& v, BoundingBox& clipped);
bool clipTriangleToBoxSHD(const Face& t, const BoundingBox& v, BoundingBox& clipped);
bool eventCmp(const Event& a, const Event& b);
bool peventCmp(const PEvent& a, const PEvent& b);
bool intersectLineSegmentPlaneD(const glm::dvec3& planeP,
                                const glm::dvec3& planeN,
                                const glm::dvec3& p0,
                                const glm::dvec3& p1,
                                glm::dvec3& intersection);

// <using>
KDTree::KDTree() : root(NULL)
{

    return;
    /*std::cout << std::setprecision(25) << std::fixed;

    glm::vec3 planeP(-2, -4, 0);
    glm::vec3 planeN(1, 0, 0);
    glm::vec3 p0(-4, -5, 0);
    glm::vec3 p1(5, -1, 0);

    glm::vec3 intersection;
    glm::dvec3 intersection2;

    if (intersectLineSegmentPlane(planeP, planeN, p0, p1, intersection))
    //std::cout << "(" << intersection.x << ", " << intersection.y << ", " << intersection.z << ")" << std::endl;
    std::cout << intersection.y << std::endl;
    if (intersectLineSegmentPlane2(planeP, planeN, p0, p1, intersection2))
    //std::cout << "(" << intersection.x << ", " << intersection.y << ", " << intersection.z << ")" << std::endl;
    std::cout << intersection2.y << std::endl;

    return;/**/

    /*BoundingBox b;
    b.max = glm::vec3(2.f, -2.f, 3.f);
    b.min = glm::vec3(-2.f, -4.f, 0.f);

    Face f;
    f.v[0] = glm::vec3(5.f, -1.f, 0.f);
    f.v[1] = glm::vec3(-1.f, -5.f, 0.f);
    f.v[2] = glm::vec3(-4.f, -5.f, 0.f);

    //calcula normal do plano
    f.tn = glm::cross((f.v[1] - f.v[0]), (f.v[2] - f.v[0]));
    f.tn = glm::normalize(f.tn);

    //mollertrumbore edges
    f.edgesMollerTrumbore[0] = f.v[1] - f.v[0];
    f.edgesMollerTrumbore[1] = f.v[2] - f.v[0];

    BoundingBox clipped;
    if (!clipTriangleToBox(f, b, clipped))
    std::cout << "EMPTY" << std::endl;

    std::cout << std::setprecision(10) << std::fixed;

    std::cout << "    Max " << "(" << clipped.max.x << ", " << clipped.max.y << ", " << clipped.max.z << ")" <<
    std::endl
    << "    Min " << "(" << clipped.min.x << ", " << clipped.min.y << ", " << clipped.min.z << ")" << std::endl;/**/

    // return;

    /*int nBBs = 10;
    int nTris = 10;
    int bbmax = 5;
    int bbmin = -5;
    int trimax = bbmax;
    int trimin = bbmin;


    std::random_device rd;
    std::vector<BoundingBox> bbs;
    for (int i = 0; i < nBBs; i++)
    {
        glm::vec3 p;
        BoundingBox b = BoundingBox();
        int n0 = (rd() % (bbmax - bbmin + 1)) + bbmin;
        int n1 = (rd() % (bbmax - bbmin + 1)) + bbmin;
        int n2 = (rd() % (bbmax - bbmin + 1)) + bbmin;

        p.x = n0;
        p.y = n1;
        p.z = n2;

        b.expand(p);
        //std::cout << "P " << glm::to_string(p) << std::endl;
        n0 = (rd() % (bbmax - bbmin + 1)) + bbmin;
        n1 = (rd() % (bbmax - bbmin + 1)) + bbmin;
        n2 = (rd() % (bbmax - bbmin + 1)) + bbmin;

        p.x = n0;
        p.y = n1;
        p.z = n2;
        b.expand(p);
        //std::cout << "P " << glm::to_string(p) << std::endl;
        //std::cout << "Max " << glm::to_string(b.max) << std::endl << "Min " << glm::to_string(b.min) << std::endl;
        //b.max.z = bbmax;
        //b.min.z = bbmin;
        bbs.push_back(b);
    }

    std::vector<Face> tris;
    for (int i = 0; i < nTris; i++)
    {
        glm::vec3 p;
        Face t;
        for (int j = 0; j < 3; j++)
        {
            int n0 = (rd() % (trimax - trimin + 1)) + trimin;
            int n1 = (rd() % (trimax - trimin + 1)) + trimin;
            int n2 = (rd() % (trimax - trimin + 1)) + trimin;
            t.v[j].x = n0;
            t.v[j].y = n1;
            t.v[j].z = n2;
            t.v[j].z = 0;
        }
        t.n[0] = glm::cross((t.v[1] - t.v[0]), (t.v[2] - t.v[0]));
        t.n[0] = glm::normalize(t.n[0]);
        t.n[1] = t.n[0];
        t.n[2] = t.n[0];

        //calcula normal do plano
        //t.tn = glm::cross((t.v[1] - t.v[0]), (t.v[2] - t.v[0]));
        //t.tn = glm::normalize(t.tn);
        t.tn = t.n[0];

        //mollertrumbore edges
        t.edgesMollerTrumbore[0] = t.v[1] - t.v[0];
        t.edgesMollerTrumbore[1] = t.v[2] - t.v[0];

        tris.push_back(t);
    }

    std::ofstream file;
    file.open("C:/lol.txt");

    std::cout << std::setprecision(10) << std::fixed;

    for (int i = 0; i < nBBs; i++)
    {
        for (int j = 0; j < nTris; j++)
        {
            BoundingBox clipped0, clipped1;
            bool intersection = true;

            if (clipTriangleToBox(tris[j], bbs[i], clipped0))
            {
                //file << "Max " << glm::to_string(clipped0.max) << " Min " << glm::to_string(clipped0.min) <<
    std::endl;
                //std::cout << "Max " << glm::to_string(clipped0.max) << std::endl << "Min " <<
    glm::to_string(clipped0.min) << std::endl;
            }
            else
            {
                intersection = false;
                //std::cout << "No intersection" << std::endl;
            }
            if (clipTriangleToBoxSHD(tris[j], bbs[i], clipped1))
            {
                if (!intersection)
                {
                    //std::cout << "Nope" << std::endl;
                    std::cout << "--BoundingBox:" << std::endl;
                    std::cout << "    Max " << "(" << bbs[i].max.x << ", " << bbs[i].max.y << ", " << bbs[i].max.z <<
    ")" << std::endl << "    Min " << "(" << bbs[i].min.x << ", " << bbs[i].min.y << ", " << bbs[i].min.z << ")" <<
    std::endl; std::cout << "--Triangle:" << std::endl; std::cout << "    V0 " << "(" << tris[j].v[0].x << ", " <<
    tris[j].v[0].y << ", " << tris[j].v[0].z << ")" << std::endl; std::cout << "    V1 " << "(" << tris[j].v[1].x << ",
    " << tris[j].v[1].y << ", " << tris[j].v[1].z << ")" << std::endl; std::cout << "    V2 " << "(" << tris[j].v[2].x
    << ", " << tris[j].v[2].y << ", " << tris[j].v[2].z << ")" << std::endl; std::cout << "--clipTriangleToBox found:"
    << std::endl; std::cout << "    No intersection" << std::endl; std::cout << "--clipTriangleToBoxSH found:" <<
    std::endl; std::cout << "    Max " << "(" << clipped1.max.x << ", " << clipped1.max.y << ", " << clipped1.max.z <<
    ")" << std::endl << "    Min " << "(" << clipped1.min.x << ", " << clipped1.min.y << ", " << clipped1.min.z << ")"
    << std::endl;

                }
                else if (clipped0.max.x == clipped1.max.x && clipped0.max.y == clipped1.max.y && clipped0.max.z ==
    clipped1.max.z
                    && clipped0.min.x == clipped1.min.x && clipped0.min.y == clipped1.min.y && clipped0.min.z ==
    clipped1.min.z)
                    //std::cout << "Ok" << std::endl;
                    ;
                else
                {
                    //std::cout << "Nope" << std::endl;
                    std::cout << "--BoundingBox:" << std::endl;
                    std::cout << "    Max " << "(" << bbs[i].max.x << ", " << bbs[i].max.y << ", " << bbs[i].max.z <<
    ")" << std::endl << "    Min " << "(" << bbs[i].min.x << ", " << bbs[i].min.y << ", " << bbs[i].min.z << ")" <<
    std::endl; std::cout << "--Triangle:" << std::endl; std::cout << "    V0 " << "(" << tris[j].v[0].x << ", " <<
    tris[j].v[0].y << ", " << tris[j].v[0].z << ")" << std::endl; std::cout << "    V1 " << "(" << tris[j].v[1].x << ",
    " << tris[j].v[1].y << ", " << tris[j].v[1].z << ")" << std::endl; std::cout << "    V2 " << "(" << tris[j].v[2].x
    << ", " << tris[j].v[2].y << ", " << tris[j].v[2].z << ")" << std::endl; std::cout << "--clipTriangleToBox found:"
    << std::endl; std::cout << "    Max " << "(" << clipped0.max.x << ", " << clipped0.max.y << ", " << clipped0.max.z
    << ")" << std::endl << "    Min " << "(" << clipped0.min.x << ", " << clipped0.min.y << ", " << clipped0.min.z <<
    ")" << std::endl; std::cout << "--clipTriangleToBoxSH found:" << std::endl; std::cout << "    Max " << "(" <<
    clipped1.max.x << ", " << clipped1.max.y << ", " << clipped1.max.z << ")" << std::endl << "    Min " << "(" <<
    clipped1.min.x << ", " << clipped1.min.y << ", " << clipped1.min.z << ")" << std::endl; std::cout << std::endl;
                }
            }
            else
            {
                if (intersection)
                {
                    //std::cout << "Nope" << std::endl;
                    std::cout << "--BoundingBox:" << std::endl;
                    std::cout << "    Max " << "(" << bbs[i].max.x << ", " << bbs[i].max.y << ", " << bbs[i].max.z <<
    ")" << std::endl << "    Min " << "(" << bbs[i].min.x << ", " << bbs[i].min.y << ", " << bbs[i].min.z << ")" <<
    std::endl; std::cout << "--Triangle:" << std::endl; std::cout << "    V0 " << "(" << tris[j].v[0].x << ", " <<
    tris[j].v[0].y << ", " << tris[j].v[0].z << ")" << std::endl; std::cout << "    V1 " << "(" << tris[j].v[1].x << ",
    " << tris[j].v[1].y << ", " << tris[j].v[1].z << ")" << std::endl; std::cout << "    V2 " << "(" << tris[j].v[2].x
    << ", " << tris[j].v[2].y << ", " << tris[j].v[2].z << ")" << std::endl; std::cout << "--clipTriangleToBox found:"
    << std::endl; std::cout << "    Max " << "(" << clipped0.max.x << ", " << clipped0.max.y << ", " << clipped0.max.z
    << ")" << std::endl << "    Min " << "(" << clipped0.min.x << ", " << clipped0.min.y << ", " << clipped0.min.z <<
    ")" << std::endl; std::cout << "--clipTriangleToBoxSH found:" << std::endl; std::cout << "    No intersection" <<
    std::endl;
                }
                else
                    //	std::cout << "Ok no intersection" << std::endl;
                    ;
            }
        }
    }

    file.close();*/
}
KDTree::~KDTree()
{
    free(root);
}
bool KDTree::build(std::vector<Object>& objects)
{
    if (objects.size() == 0)
        return false;

    if (maxDepth > MAX_DEPTH)
    {
        std::cout << "Hardcoded max depth of " << MAX_DEPTH << ". Can't go beyond that" << std::endl;
        maxDepth = MAX_DEPTH;
    }

    std::cout.precision(6);
    std::cout << "Scene triangle count: " << objects[0].f.size() << std::endl;
    std::cout << "Building kd-tree... " << std::endl;

    root = free(root);

    nNodes = 0;
    nLeafs = 0;
    nTriangles = 0;
    depth = 0;

    Stopwatch stopwatch;
    stopwatch.start();

    bb = objects[0].boundingBox;

    // presorted
    // kdtriangles = objects[0].f;

    /**/ std::vector<Side> flags(objects[0].f.size(), Side::BOTH);

    // std::vector<std::pair<Face*, Side*>> triandflag;
    std::vector<TFpointers> triandflag;
    triandflag.reserve(objects[0].f.size());
    for (int i = 0; i < objects[0].f.size(); i++)
        triandflag.emplace_back(objects[0].f.data() + i, flags.data() + i);

    std::vector<PEvent> events;
    presortedGenEvents(triandflag, bb, events);
    std::sort(events.begin(), events.end(), peventCmp);

    root = presortedBuildNodeSah(0, triandflag, events, bb); /**/

    // presorted

    // root = buildNodeSah(0, objects[0].f, bb);

    if (stop)
    {
        free(root);
        std::cout << "Tree building aborted!" << std::endl;
        return false;
    }

    stopwatch.stop();

    printInfo();

    std::cout << "Building time:   " << stopwatch.elapsedMillis.count() / 1000.0 << "s" << std::endl
              << "Done!" << std::endl;

    return true;
}
KDTNode* KDTree::presortedBuildNodeSah(int depth,
                                       std::vector<TFpointers>& triandflag,
                                       std::vector<PEvent>& events,
                                       BoundingBox& nodebb)
{
    if (stop)
        return NULL;

    KDTNode* node = new KDTNode();
    nNodes++;

    // Definitely leaf.
    if (depth >= maxDepth || triandflag.size() <= sizeToBecomeLeaf)
    {
        // Store triangles
        /*for (int i = 0; i < triandflag.size(); i++)
        node->triangles.emplace_back(triandflag[i].triangle);
        node->triangles.shrink_to_fit();
        nTriangles += node->triangles.size();/**/
        /**/ node->leafPayload.nTris = triandflag.size();
        node->leafPayload.tris = new Face*[node->leafPayload.nTris];
        for (int i = 0; i < triandflag.size(); i++)
            node->leafPayload.tris[i] = triandflag[i].triangle;
        nTriangles += node->leafPayload.nTris; /**/
        node->isLeaf = true;
        nLeafs++;
        if (depth > this->depth)
            this->depth = depth;
        return node;
    }

    Side splitPlaneSide;
    Plane splitPlane;
    float splitPlaneCost = presortedEventFindBestPlane(triandflag.size(), events, nodebb, splitPlaneSide, splitPlane);

    // Now that we know the best split is time to check if not splitting at all is better
    // If the cost of not splitting(making this node a leaf) is better, we just... don't split
    if (cost(triandflag.size()) < splitPlaneCost)
    {
        // Store triangles
        /*for (int i = 0; i < triandflag.size(); i++)
        node->triangles.emplace_back(triandflag[i].triangle);
        node->triangles.shrink_to_fit();
        nTriangles += node->triangles.size();/**/
        /**/ node->leafPayload.nTris = triandflag.size();
        node->leafPayload.tris = new Face*[node->leafPayload.nTris];
        for (int i = 0; i < triandflag.size(); i++)
            node->leafPayload.tris[i] = triandflag[i].triangle;
        nTriangles += node->leafPayload.nTris; /**/
        node->isLeaf = true;
        nLeafs++;
        if (depth > this->depth)
            this->depth = depth;
        return node;
    }

    // Otherwise, we have to classify the triangles to the left or right child
    presortedClassify(triandflag, events, splitPlane, splitPlaneSide);

    std::vector<PEvent> eventsl, eventsr;
    presortedSplitEvents(events, triandflag, nodebb, splitPlane, eventsl, eventsr);

    // Deallocate memory as soon as possible
    events = std::vector<PEvent>();

    std::vector<TFpointers> triandflagl, triandflagr;
    for (int i = 0; i < triandflag.size(); i++)
    {
        if (*triandflag[i].flag == Side::LEFT)
        {
            triandflagl.emplace_back(triandflag[i]);
        }
        else if (*triandflag[i].flag == Side::RIGHT)
        {
            triandflagr.emplace_back(triandflag[i]);
        }
        else if (*triandflag[i].flag == Side::BOTH)
        {
            triandflagl.emplace_back(triandflag[i]);
            triandflagr.emplace_back(triandflag[i]);
        }
    }

    // Deallocate memory as soon as possible
    triandflag = std::vector<TFpointers>();

    BoundingBox childbb = nodebb;
    childbb.max[splitPlane.dim] = splitPlane.value;

    node->interiorPayload.children[0] = presortedBuildNodeSah(depth + 1, triandflagl, eventsl, childbb);

    // Deallocate memory as soon as possible
    triandflagl = std::vector<TFpointers>();
    eventsl = std::vector<PEvent>();

    childbb = nodebb;
    childbb.min[splitPlane.dim] = splitPlane.value;

    node->interiorPayload.children[1] = presortedBuildNodeSah(depth + 1, triandflagr, eventsr, childbb);

    node->isLeaf = false;
    node->interiorPayload.plane = splitPlane;

    return node;
} /**/
void KDTree::presortedGenEvents(const std::vector<TFpointers>& tfs,
                                const BoundingBox& nodebb,
                                std::vector<PEvent>& events)
{
    // Reserve all necessary memory at once
    events.reserve(tfs.size() * 2 * 3);

    // Generate the events for all dimensions
    for (int dim = 0; dim < 3; dim++)
    {
        // For each triangle
        for (int i = 0; i < tfs.size(); i++)
        {
            float min = std::numeric_limits<float>::max();
            float max = std::numeric_limits<float>::lowest();
            // Find the limits for dim
            for (int j = 0; j < 3; j++)
            {
                if (tfs[i].triangle->v[j][dim] < min)
                    min = tfs[i].triangle->v[j][dim];
                if (tfs[i].triangle->v[j][dim] > max)
                    max = tfs[i].triangle->v[j][dim];
            }
            // If the triangle is planar
            if (min == max)
            {
                events.emplace_back(Plane(dim, min), tfs[i].triangle, tfs[i].flag, EVENT_TYPE_PLANAR);
            }
            else
            {
                events.emplace_back(Plane(dim, min), tfs[i].triangle, tfs[i].flag, EVENT_TYPE_START);
                events.emplace_back(Plane(dim, max), tfs[i].triangle, tfs[i].flag, EVENT_TYPE_END);
            }
        }
    }
}
float KDTree::presortedEventFindBestPlane(const int triangles,
                                          const std::vector<PEvent>& events,
                                          const BoundingBox& nodebb,
                                          Side& side,
                                          Plane& plane)
{
    float minCost = std::numeric_limits<float>::max();
    Plane minCostPlane;
    Side minCostSide;

    int nl[3] = {0, 0, 0};
    int np[3] = {0, 0, 0};
    int nr[3] = {triangles, triangles, triangles};

    int esize = events.size();
    int i = 0;
    while (i < esize)
    {
        Plane p;
        p = events[i].plane;
        unsigned int pstart = 0, pplanar = 0, pend = 0;

        while ((i < esize) && (events[i].plane.dim == p.dim) && (events[i].plane.value == p.value) &&
               (events[i].type == EVENT_TYPE_END))
        {
            i++;
            pend++;
        }
        while ((i < esize) && (events[i].plane.dim == p.dim) && (events[i].plane.value == p.value) &&
               (events[i].type == EVENT_TYPE_PLANAR))
        {
            i++;
            pplanar++;
        }
        while ((i < esize) && (events[i].plane.dim == p.dim) && (events[i].plane.value == p.value) &&
               (events[i].type == EVENT_TYPE_START))
        {
            i++;
            pstart++;
        }

        np[p.dim] = pplanar;
        nr[p.dim] -= pplanar;
        // nr[p.dim] -= pend;
        nl[p.dim] += pstart;

        float cost;
        Side s;
        cost = sah(p, nodebb, nl[p.dim], nr[p.dim], np[p.dim], s);
        if (cost < minCost)
        {
            minCost = cost;
            minCostPlane = p;
            minCostSide = s;
        }

        nr[p.dim] -= pend;
        // nl[p.dim] += pstart;
        nl[p.dim] += pplanar;
        np[p.dim] = 0;
    }

    side = minCostSide;
    plane = minCostPlane;
    return minCost;

    // Now that we know the best split is time to check if not splitting at all is better
    // If the cost of not splitting(making this node a leaf) is better, we just... don't split
    /*if (cost(ts.size()) < minCost)
    return false;

    // Otherwise, we have to classify the triangles to the left or right child

    std::vector<PEvent> elo, ero;
    std::vector<Face*> both;

    for (int i = 0; i < events.size(); i++)
    {
    if (events[i].type == EVENT_TYPE_END && events[i].plane.dim == minCostPlane.dim && events[i].plane.value <
    minCostPlane.value)
    {
    // left only
    elo.push_back(events[i]);
    }
    else if (events[i].type == EVENT_TYPE_START && events[i].plane.dim == minCostPlane.dim && events[i].plane.value >
    minCostPlane.value)
    {
    // right only
    ero.push_back(events[i]);
    }
    else if (events[i].type == EVENT_TYPE_PLANAR && events[i].plane.dim == minCostPlane.dim)
    {
    if (events[i].plane.value < minCostPlane.value || (events[i].plane.value == minCostPlane.value && minCostSide ==
    Side::LEFT))
    {
    //left only
    elo.push_back(events[i]);
    }
    else if (events[i].plane.value > minCostPlane.value || (events[i].plane.value == minCostPlane.value && minCostSide
    == Side::RIGHT))
    {
    //right only
    ero.push_back(events[i]);
    }
    }
    else if (events[i].plane.dim == minCostPlane.dim)
    {
    // both

    }
    }


    // For each clipped triangle
    /*leftTs.clear();
    rightTs.clear();
    for (int i = 0; i < tsbb.size(); i++)
    {
    // If the triangle is planar to the best spliting plane
    if (tsbb[i].first.min[minCostPlane.dim] == tsbb[i].first.max[minCostPlane.dim])
    {
    if (minCostSide == Side::LEFT)
    leftTs.push_back(ts[tsbb[i].second]);
    else
    rightTs.push_back(ts[tsbb[i].second]);
    }
    else
    {
    // Otherwise, we have to check to which side of the plane the triangle is
    // Only to the left
    if (tsbb[i].first.max[minCostPlane.dim] < minCostPlane.value)
    leftTs.push_back(ts[tsbb[i].second]);
    // Only to the right
    else if (tsbb[i].first.min[minCostPlane.dim] > minCostPlane.value)
    rightTs.push_back(ts[tsbb[i].second]);
    // Both sides
    else
    {
    leftTs.push_back(ts[tsbb[i].second]);
    rightTs.push_back(ts[tsbb[i].second]);
    }
    }
    }*/
    /*
    splitP = minCostPlane;

    return true;*/
}
void KDTree::presortedClassify(const std::vector<TFpointers>& tfs,
                               const std::vector<PEvent>& events,
                               const Plane& splitPlane,
                               const Side& splitPlaneSide)
{
    // Set all triangles to BOTH
    for (int i = 0; i < tfs.size(); i++)
        *tfs[i].flag = Side::BOTH;

    for (int i = 0; i < events.size(); i++)
    {
        if (events[i].type == EVENT_TYPE_END && events[i].plane.dim == splitPlane.dim &&
            events[i].plane.value < splitPlane.value)
        {
            // left only
            *events[i].flag = Side::LEFT;
        }
        else if (events[i].type == EVENT_TYPE_START && events[i].plane.dim == splitPlane.dim &&
                 events[i].plane.value > splitPlane.value)
        {
            // right only
            *events[i].flag = Side::RIGHT;
        }
        else if (events[i].type == EVENT_TYPE_PLANAR && events[i].plane.dim == splitPlane.dim)
        {
            if (events[i].plane.value < splitPlane.value ||
                (events[i].plane.value == splitPlane.value && splitPlaneSide == Side::LEFT))
            {
                // left only
                *events[i].flag = Side::LEFT;
            }
            else if (events[i].plane.value > splitPlane.value ||
                     (events[i].plane.value == splitPlane.value && splitPlaneSide == Side::RIGHT))
            {
                // right only
                *events[i].flag = Side::RIGHT;
            }
        }
    }
}
void KDTree::presortedSplitEvents(const std::vector<PEvent>& events,
                                  const std::vector<TFpointers>& tfs,
                                  const BoundingBox& nodebb,
                                  const Plane& splitplane,
                                  std::vector<PEvent>& eventsl,
                                  std::vector<PEvent>& eventsr)
{
    std::vector<PEvent> leftonly, rightonly, bothleft, bothright;
    leftonly.reserve(events.size() / 2);
    rightonly.reserve(events.size() / 2);
    // Events flagged as LEFT or RIGHT go directly to the new events list
    for (int i = 0; i < events.size(); i++)
    {
        if (*events[i].flag == Side::LEFT)
            leftonly.emplace_back(events[i]);
        else if (*events[i].flag == Side::RIGHT)
            rightonly.emplace_back(events[i]);
    }

    BoundingBox vl = nodebb;
    vl.max[splitplane.dim] = splitplane.value;
    BoundingBox vr = nodebb;
    vr.min[splitplane.dim] = splitplane.value;
    // For every triangle that was flagged as BOTH, generates the new events
    for (int i = 0; i < tfs.size(); i++)
    {
        if (*tfs[i].flag == Side::BOTH)
        {
            // Clip triangle and generate new events
            BoundingBox clippedl, clippedr;
            // if (clipTriangleToBox(*tfs[i].triangle, vl, clippedl))
            if (clipTriangleToBoxSHD(*tfs[i].triangle, vl, clippedl))
            {
                // Generate the events for all dimensions
                for (int dim = 0; dim < 3; dim++)
                {
                    // Left
                    // If the triangle is planar
                    if (clippedl.min[dim] == clippedl.max[dim])
                    {
                        bothleft.emplace_back(Plane(dim, clippedl.min[dim]),
                                              tfs[i].triangle,
                                              tfs[i].flag,
                                              EVENT_TYPE_PLANAR);
                    }
                    else
                    {
                        bothleft.emplace_back(Plane(dim, clippedl.min[dim]),
                                              tfs[i].triangle,
                                              tfs[i].flag,
                                              EVENT_TYPE_START);
                        bothleft.emplace_back(Plane(dim, clippedl.max[dim]),
                                              tfs[i].triangle,
                                              tfs[i].flag,
                                              EVENT_TYPE_END);
                    }
                }
            }
            // if (clipTriangleToBox(*tfs[i].triangle, vr, clippedr))
            if (clipTriangleToBoxSHD(*tfs[i].triangle, vr, clippedr))
            {
                for (int dim = 0; dim < 3; dim++)
                {
                    // Right
                    // If the triangle is planar
                    if (clippedr.min[dim] == clippedr.max[dim])
                    {
                        bothright.emplace_back(Plane(dim, clippedr.min[dim]),
                                               tfs[i].triangle,
                                               tfs[i].flag,
                                               EVENT_TYPE_PLANAR);
                    }
                    else
                    {
                        bothright.emplace_back(Plane(dim, clippedr.min[dim]),
                                               tfs[i].triangle,
                                               tfs[i].flag,
                                               EVENT_TYPE_START);
                        bothright.emplace_back(Plane(dim, clippedr.max[dim]),
                                               tfs[i].triangle,
                                               tfs[i].flag,
                                               EVENT_TYPE_END);
                    }
                }
            }
        }
    }

    // Since events is sorted, leftonly and rightonly are sorted.
    // bothleft and bothright where generated by triangles overlaped by the two child nodes.
    // Since they have new events, they are not sorted.
    std::sort(bothleft.begin(), bothleft.end(), peventCmp);
    std::sort(bothright.begin(), bothright.end(), peventCmp);

    // Now we need to merge
    eventsl.reserve(leftonly.size() + bothleft.size());
    std::merge(leftonly.begin(),
               leftonly.end(),
               bothleft.begin(),
               bothleft.end(),
               std::back_inserter(eventsl),
               peventCmp);

    eventsr.reserve(rightonly.size() + bothright.size());
    std::merge(rightonly.begin(),
               rightonly.end(),
               bothright.begin(),
               bothright.end(),
               std::back_inserter(eventsr),
               peventCmp);
}
bool clipTriangleToBoxSHD(const Face& t, const BoundingBox& v, BoundingBox& clipped)
{
    std::vector<glm::dvec3> vertices;
    std::vector<glm::dvec3> newvertices;

    for (int i = 0; i < 3; i++)
        vertices.emplace_back(t.v[i]);

    glm::dvec3 intersection;

    // Clip min planes
    for (int dim = 0; dim < 3; dim++)
    {
        for (int j = 0; j < vertices.size(); j++)
        {
            int next = (j + 1) % vertices.size();

            if (vertices[j][dim] < v.min[dim])
            {
                // Outside
                if (vertices[next][dim] < v.min[dim])
                {
                    // Do nothing
                }
                // Entering
                else if (vertices[next][dim] > v.min[dim])
                {
                    glm::dvec3 planeN(0.f);
                    planeN[dim] = 1.f;
                    // Add the intersection point and vertices[next]
                    if (intersectLineSegmentPlaneD(glm::dvec3(v.min),
                                                   planeN,
                                                   vertices[j],
                                                   vertices[next],
                                                   intersection))
                    {
                        newvertices.emplace_back(intersection);
                        newvertices.emplace_back(vertices[next]);
                    }
                }
                // Vertice touching the plane
                else
                    newvertices.emplace_back(vertices[next]);
            }
            else
            {
                // Inside
                if (vertices[next][dim] >= v.min[dim])
                {
                    // Add only vertices[next]
                    newvertices.emplace_back(vertices[next]);
                }
                // Leaving, but the intersection is the very vertices[j]
                else if (vertices[j][dim] == v.min[dim])
                {
                    // Add vertices[j]
                    newvertices.emplace_back(vertices[j]);
                }
                // Leaving
                else
                {
                    // Add the intersection point
                    glm::dvec3 planeN(0.f);
                    planeN[dim] = 1.f;
                    // Add the intersection point and vertices[next]
                    if (intersectLineSegmentPlaneD(glm::dvec3(v.min),
                                                   planeN,
                                                   vertices[j],
                                                   vertices[next],
                                                   intersection))
                        newvertices.emplace_back(intersection);
                }
            }
        }
        vertices.swap(newvertices);
        newvertices.clear();
    }

    // Clip max planes
    for (int dim = 0; dim < 3; dim++)
    {
        for (int j = 0; j < vertices.size(); j++)
        {
            int next = (j + 1) % vertices.size();

            // vetices[j] outside
            if (vertices[j][dim] > v.max[dim])
            {
                // Both outside
                if (vertices[next][dim] > v.max[dim])
                {
                    // Do nothing
                }
                // Entering
                else if (vertices[next][dim] < v.max[dim])
                {
                    glm::dvec3 planeN(0.f);
                    planeN[dim] = 1.f;
                    // Add the intersection point and vertices[next]
                    if (intersectLineSegmentPlaneD(glm::dvec3(v.max),
                                                   planeN,
                                                   vertices[j],
                                                   vertices[next],
                                                   intersection))
                    {
                        newvertices.emplace_back(intersection);
                        newvertices.emplace_back(vertices[next]);
                    }
                }
                // Vertice touching the plane
                else
                    newvertices.emplace_back(vertices[next]);
            }
            // vetices[j] inside
            else
            {
                // Both inside
                if (vertices[next][dim] <= v.max[dim])
                {
                    // Add only vertices[next]
                    newvertices.emplace_back(vertices[next]);
                }
                // Leaving, but the intersection is the very vertices[j]
                else if (vertices[j][dim] == v.max[dim])
                {
                    // Add vertices[j]
                    newvertices.emplace_back(vertices[j]);
                }
                else
                {
                    // Add the intersection point
                    glm::dvec3 planeN(0.f);
                    planeN[dim] = 1.f;
                    // Add the intersection point and vertices[next]
                    if (intersectLineSegmentPlaneD(glm::dvec3(v.max),
                                                   planeN,
                                                   vertices[j],
                                                   vertices[next],
                                                   intersection))
                        newvertices.emplace_back(intersection);
                }
            }
        }
        vertices.swap(newvertices);
        newvertices.clear();
    }

    if (vertices.empty())
        return false;

    for (int j = 0; j < vertices.size(); j++)
        clipped.expand(glm::vec3(vertices[j]));

    return true;
}
float KDTree::sah(KDTNode* node)
{
    if (node->isLeaf)
        // return (node->triangles.size() * KI);
        return (node->leafPayload.nTris * KI);

    return (KT + sah(node->interiorPayload.children[0]) + sah(node->interiorPayload.children[1]));
}
float KDTree::sah(const Plane p,
                  const BoundingBox& v,
                  const int triangleCountL,
                  const int triangleCountR,
                  const int triangleCountP,
                  Side& side)
{
    // Split bounding box on p
    BoundingBox vl, vr;
    vl = v;
    vl.max[p.dim] = p.value;
    vr = v;
    vr.min[p.dim] = p.value;
    // Now we have a boundingbox for the left and one for the right
    // Calculate the probability for each volume
    float invVSA = 1.f / v.surfaceArea();
    float probl = vl.surfaceArea() * invVSA;
    float probr = vr.surfaceArea() * invVSA;
    // Calc costs
    float costl = cost(probl, probr, triangleCountL + triangleCountP, triangleCountR);
    float costr = cost(probl, probr, triangleCountL, triangleCountR + triangleCountP);
    // Return the one with the lowest cost
    if (costl < costr)
    {
        side = Side::LEFT;
        return costl;
    }
    side = Side::RIGHT;
    return costr;
}
KDTNode* KDTree::free(KDTNode* node)
{
    /**/ if (node == NULL)
        return NULL;

    if (node->isLeaf)
    {
        delete[] node->leafPayload.tris;
    }
    else
    {
        if (node->interiorPayload.children[0] != NULL)
            node->interiorPayload.children[0] = free(node->interiorPayload.children[0]);
        if (node->interiorPayload.children[1] != NULL)
            node->interiorPayload.children[1] = free(node->interiorPayload.children[1]);
    }
    delete node;
    return NULL; /**/

    /*if (node == NULL)
    return NULL;

    if (!node->isLeaf)
    {
    if (node->interiorPayload.children[0] != NULL)
    node->interiorPayload.children[0] = free(node->interiorPayload.children[0]);
    if (node->interiorPayload.children[1] != NULL)
    node->interiorPayload.children[1] = free(node->interiorPayload.children[1]);
    }
    delete node;
    return NULL;*/
}
float KDTree::cost(const float probabilityL,
                   const float probabilityR,
                   const int triangleCountL,
                   const int triangleCountR)
{
    float bonus = 1.f;

    if (triangleCountL == 0 || triangleCountR == 0)
        bonus *= emptySpaceBonus;

    return (KT + KI * (probabilityL * triangleCountL + probabilityR * triangleCountR)) * bonus;
}
float KDTree::cost(const int triangleCount)
{
    // Leaf cost
    return KI * triangleCount;
}
bool intersectLineSegmentPlaneD(const glm::dvec3& planeP,
                                const glm::dvec3& planeN,
                                const glm::dvec3& p0,
                                const glm::dvec3& p1,
                                glm::dvec3& intersection)
{
    glm::dvec3 vec(p1 - p0);

    double dot = glm::dot(planeN, vec);
    if (dot > -EPSILON && dot < EPSILON)
        return false;

    double t = glm::dot(planeN, planeP - p0) / dot;

    if (t >= 0. && t <= 1.)
    {
        intersection = p0 + t * vec;
        return true;
    }
    return false;
}
long KDTree::sizeInBytes()
{
    long size = sizeof(KDTree);
    size += sizeof(KDTNode) * nNodes;
    size += sizeof(Face*) * nTriangles;
    return size;
}
void KDTree::printInfo()
{
    std::cout << "Node #" << std::endl
              << "    Leaf:        " << nLeafs << std::endl
              << "    Interior:    " << nNodes - nLeafs << std::endl
              << "    Total:       " << nNodes << std::endl
              << "Max. depth:      " << depth << std::endl
              << "Tri. references: " << nTriangles << std::endl
              << "Avg. tris/leaf:  " << (float)nTriangles / (float)nLeafs << std::endl
              << "Size:            " << sizeInBytes() / 1024 << "KB" << std::endl
              << "SAH cost:        " << sah(root) << std::endl;
}
bool KDTree::intersect(const Ray& r, Intersection& intersection) const
{
    float tmin, tmax, tplane;
    // Intersect ray with the tree's boundingbox
    if (!RTUtils::hitBoundingBox(r, bb, tmin, tmax))
        return false;

    // KDTNodeToDo *todo = new KDTNodeToDo[maxDepth];
    KDTNodeToDo todo[MAX_DEPTH];
    int todoPos = -1;

    bool hit = false;
    intersection.distance = FLT_MAX;
    Intersection _intersection;
    const KDTNode* n = root;

    while (n != NULL)
    {
        if (intersection.distance < tmin)
            break;
        // Leaf
        if (n->isLeaf)
        {
            if (RTUtils::intersectRayTrianglesMollerTrumbore(r,
                                                             n->leafPayload.tris,
                                                             n->leafPayload.nTris,
                                                             _intersection))
            {
                if (_intersection.distance < intersection.distance)
                {
                    intersection = _intersection;
                    hit = true;
                    if (intersection.distance <= tmax)
                        break;
                }
            }
            // No more nodes to trace
            if (todoPos < 0)
                break;
            // Get node from todo list
            n = todo[todoPos].node;
            tmin = todo[todoPos].tmin;
            tmax = todo[todoPos].tmax;
            --todoPos;
            // continue;
        }
        else
        {
            // Find first and second child (first to be intersected by the ray)
            KDTNode* first = NULL;
            KDTNode* second = NULL;

            bool belowFirst = (r.origin[n->interiorPayload.plane.dim] < n->interiorPayload.plane.value) ||
                              (r.origin[n->interiorPayload.plane.dim] == n->interiorPayload.plane.value &&
                               r.direction[n->interiorPayload.plane.dim] >= 0);
            if (belowFirst)
            {
                first = n->interiorPayload.children[0];
                second = n->interiorPayload.children[1];
            }
            else
            {
                first = n->interiorPayload.children[1];
                second = n->interiorPayload.children[0];
            }

            // Find t for the split plane
            if (r.direction[n->interiorPayload.plane.dim] != 0.f)
                tplane = (n->interiorPayload.plane.value - r.origin[n->interiorPayload.plane.dim]) /
                         r.direction[n->interiorPayload.plane.dim];
            else
                tplane = -1.f;

            // We test only the first child
            if (tplane > tmax || tplane <= 0)
            {
                n = first;
            }
            // We test only the second child
            else if (tplane < tmin)
            {
                n = second;
            }
            // Ray intersects both children, so we test both.
            else
            {
                ++todoPos;
                todo[todoPos].node = second;
                todo[todoPos].tmin = tplane;
                todo[todoPos].tmax = tmax;
                n = first;
                tmax = tplane;
            }
        }
    }

    // delete[] todo;

    return hit;
}
bool peventCmp(const PEvent& a, const PEvent& b)
{
    return ((a.plane.value < b.plane.value) || (a.plane.value == b.plane.value && a.plane.dim < b.plane.dim) ||
            (a.plane.value == b.plane.value && a.plane.dim == b.plane.dim && a.type < b.type));
}
// </using>

/*

bool KDTree::intersectNonRecursive(const Ray& r, const Face** hitTriangle, float& hitDistance) const
{
    float tmin, tmax, tplane;
    // Intersect ray with the tree's boundingbox
    if (!RTUtils::hitBoundingBox(r, bb, tmin, tmax))
        return false;

    //KDTNodeToDo *todo = new KDTNodeToDo[maxDepth];
    KDTNodeToDo todo[MAX_DEPTH];
    int todoPos = -1;

    bool hit = false;
    const Face* hitf = NULL;
    hitDistance = FLT_MAX;
    const KDTNode* n = root;

    while (n != NULL)
    {
        if (hitDistance < tmin)
            break;
        // Leaf
        if (n->isLeaf)
        {
            float dist;
            if (RTUtils::intersectRayTrianglesMollerTrumbore(r, n->leafPayload.tris, n->leafPayload.nTris, &hitf, dist))
                //if (RTUtils::intersectRayTrianglesMollerTrumbore(r, n->triangles, &hitf, dist))
                //if (RTUtils::intersectRayTrianglesMollerTrumboreSIMD128(r, n->triangles, &hitf, dist))
            {
                if (dist < hitDistance)
                {
                    *hitTriangle = hitf;
                    hitDistance = dist;
                    hit = true;
                    if (hitDistance <= tmax)
                        break;
                }
            }
            // No more nodes to trace
            if (todoPos < 0)
                break;
            // Get node from todo list
            n = todo[todoPos].node;
            tmin = todo[todoPos].tmin;
            tmax = todo[todoPos].tmax;
            --todoPos;
            //continue;
        }
        else
        {
            // Find first and second child (first to be intersected by the ray)
            KDTNode* first = NULL;
            KDTNode* second = NULL;

            bool belowFirst = (r.origin[n->interiorPayload.plane.dim] < n->interiorPayload.plane.value) ||
                (r.origin[n->interiorPayload.plane.dim] == n->interiorPayload.plane.value &&
r.direction[n->interiorPayload.plane.dim] >= 0); if (belowFirst)
            {
                first = n->interiorPayload.children[0];
                second = n->interiorPayload.children[1];
            }
            else
            {
                first = n->interiorPayload.children[1];
                second = n->interiorPayload.children[0];
            }

            // Find t for the split plane
            if (r.direction[n->interiorPayload.plane.dim] != 0.f)
                tplane = (n->interiorPayload.plane.value - r.origin[n->interiorPayload.plane.dim]) /
r.direction[n->interiorPayload.plane.dim]; else tplane = -1.f;

            // We test only the first child
            if (tplane > tmax || tplane <= 0)
            {
                n = first;
            }
            // We test only the second child
            else if (tplane < tmin)
            {
                n = second;
            }
            // Ray intersects both children, so we test both.
            else
            {
                ++todoPos;
                todo[todoPos].node = second;
                todo[todoPos].tmin = tplane;
                todo[todoPos].tmax = tmax;
                n = first;
                tmax = tplane;
            }
        }
    }

    //delete[] todo;

    return hit;
}
bool KDTree::intersectNonRecursiveP(const Ray& r, const Face** hitTriangle, float& hitDistance) const
{
    float tmin, tmax, tplane;
    // Intersect ray with the tree's boundingbox
    if (!RTUtils::hitBoundingBox(r, bb, tmin, tmax))
        return false;

    //KDTNodeToDo *todo = new KDTNodeToDo[maxDepth];
    KDTNodeToDo todo[MAX_DEPTH];
    int todoPos = -1;

    bool hit = false;
    const Face* hitf = NULL;
    hitDistance = FLT_MAX;
    const KDTNode* n = root;

    while (n != NULL)
    {
        if (hitDistance < tmin)
            break;
        // Leaf
        if (n->isLeaf)
        {
            float dist;
            if (RTUtils::intersectRayTrianglesMollerTrumbore(r, n->leafPayload.tris, n->leafPayload.nTris, &hitf, dist))
                //if (RTUtils::intersectRayTrianglesMollerTrumbore(r, n->triangles, &hitf, dist))
                //if (RTUtils::intersectRayTrianglesMollerTrumboreSIMD128(r, n->triangles, &hitf, dist))
            {
                if (dist < hitDistance)
                {
                    *hitTriangle = hitf;
                    hitDistance = dist;
                    hit = true;
                    if (hitDistance <= tmax)
                        break;
                }
            }
            // No more nodes to trace
            if (todoPos < 0)
                break;
            // Get node from todo list
            n = todo[todoPos].node;
            tmin = todo[todoPos].tmin;
            tmax = todo[todoPos].tmax;
            --todoPos;
            //continue;
        }
        else
        {
            // Find first and second child (first to be intersected by the ray)
            KDTNode* first = NULL;
            KDTNode* second = NULL;

            bool belowFirst = (r.origin[n->interiorPayload.plane.dim] < n->interiorPayload.plane.value) ||
                (r.origin[n->interiorPayload.plane.dim] == n->interiorPayload.plane.value &&
r.direction[n->interiorPayload.plane.dim] >= 0); if (belowFirst)
            {
                first = n->interiorPayload.children[0];
                second = n->interiorPayload.children[1];
            }
            else
            {
                first = n->interiorPayload.children[1];
                second = n->interiorPayload.children[0];
            }

            // Find t for the split plane
            if (r.direction[n->interiorPayload.plane.dim] != 0.f)
                tplane = (n->interiorPayload.plane.value - r.origin[n->interiorPayload.plane.dim]) /
r.direction[n->interiorPayload.plane.dim]; else tplane = -1.f;

            // We test only the first child
            if (tplane > tmax || tplane <= 0)
            {
                n = first;
            }
            // We test only the second child
            else if (tplane < tmin)
            {
                n = second;
            }
            // Ray intersects both children, so we test both.
            else
            {
                ++todoPos;
                todo[todoPos].node = second;
                todo[todoPos].tmin = tplane;
                todo[todoPos].tmax = tmax;
                n = first;
                tmax = tplane;
            }
        }
    }

    //delete[] todo;

    return hit;
}

bool intersectLineSegmentPlane(const glm::vec3& planeP, const glm::vec3& planeN, const glm::vec3& p0, const glm::vec3&
p1, glm::vec3& intersection)
{
    const glm::vec3 vec(p1 - p0);
    float dot = glm::dot(planeN, vec);
    if (dot > -EPSILON && dot < EPSILON)
        return false;

    float t = glm::dot(planeN, planeP - p0) / dot;

    if (t >= 0.f && t <= 1.f)
    {
        intersection = p0 + t * vec;
        return true;
    }
    return false;
}




bool eventCmp(const Event& a, const Event& b)
{
    return ((a.plane.value < b.plane.value) ||
            (a.plane.value == b.plane.value && a.plane.dim < b.plane.dim) ||
            (a.plane.value == b.plane.value && a.plane.dim == b.plane.dim && a.type < b.type));
}





// Sutherland - Hodgman
bool clipTriangleToBoxSH(const Face& t, const BoundingBox& v, BoundingBox& clipped)
{
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> newvertices;

    for (int i = 0; i < 3; i++)
        vertices.emplace_back(t.v[i]);

    glm::vec3 intersection;

    // Clip min planes
    for (int dim = 0; dim < 3; dim++)
    {
        for (int j = 0; j < vertices.size(); j++)
        {
            int next = (j + 1) % vertices.size();

            if (vertices[j][dim] < v.min[dim])
            {
                // Outside
                if (vertices[next][dim] < v.min[dim])
                {
                    // Do nothing
                }
                // Entering
                else if (vertices[next][dim] > v.min[dim])
                {
                    glm::vec3 planeN(0.f);
                    planeN[dim] = 1.f;
                    // Add the intersection point and vertices[next]
                    if (intersectLineSegmentPlane(v.min, planeN, vertices[j], vertices[next], intersection))
                    {
                        newvertices.emplace_back(intersection);
                        newvertices.emplace_back(vertices[next]);
                    }
                }
                // Vertice touching the plane
                else
                    newvertices.emplace_back(vertices[next]);
            }
            else
            {
                // Inside
                if (vertices[next][dim] >= v.min[dim])
                {
                    // Add only vertices[next]
                    newvertices.emplace_back(vertices[next]);
                }
                // Leaving, but the intersection is the very vertices[j]
                else if (vertices[j][dim] == v.min[dim])
                {
                    // Add vertices[j]
                    newvertices.emplace_back(vertices[j]);
                }
                // Leaving
                else
                {
                    // Add the intersection point
                    glm::vec3 planeN(0.f);
                    planeN[dim] = 1.f;
                    // Add the intersection point and vertices[next]
                    if (intersectLineSegmentPlane(v.min, planeN, vertices[j], vertices[next], intersection))
                        newvertices.emplace_back(intersection);
                }
            }
        }
        vertices.swap(newvertices);
        newvertices.clear();
    }

    // Clip max planes
    for (int dim = 0; dim < 3; dim++)
    {
        for (int j = 0; j < vertices.size(); j++)
        {
            int next = (j + 1) % vertices.size();

            // vetices[j] outside
            if (vertices[j][dim] > v.max[dim])
            {
                // Both outside
                if (vertices[next][dim] > v.max[dim])
                {
                    // Do nothing
                }
                // Entering
                else if (vertices[next][dim] < v.max[dim])
                {
                    glm::vec3 planeN(0.f);
                    planeN[dim] = 1.f;
                    // Add the intersection point and vertices[next]
                    if (intersectLineSegmentPlane(v.max, planeN, vertices[j], vertices[next], intersection))
                    {
                        newvertices.emplace_back(intersection);
                        newvertices.emplace_back(vertices[next]);
                    }
                }
                // Vertice touching the plane
                else
                    newvertices.emplace_back(vertices[next]);
            }
            // vetices[j] inside
            else
            {
                // Both inside
                if (vertices[next][dim] <= v.max[dim])
                {
                    // Add only vertices[next]
                    newvertices.emplace_back(vertices[next]);
                }
                // Leaving, but the intersection is the very vertices[j]
                else if (vertices[j][dim] == v.max[dim])
                {
                    // Add vertices[j]
                    newvertices.emplace_back(vertices[j]);
                }
                else
                {
                    // Add the intersection point
                    glm::vec3 planeN(0.f);
                    planeN[dim] = 1.f;
                    // Add the intersection point and vertices[next]
                    if (intersectLineSegmentPlane(v.max, planeN, vertices[j], vertices[next], intersection))
                        newvertices.emplace_back(intersection);
                }
            }
        }
        vertices.swap(newvertices);
        newvertices.clear();
    }

    if (vertices.empty())
        return false;

    for (int j = 0; j < vertices.size(); j++)
        clipped.expand(vertices[j]);

    return true;
}

bool clipTriangleToBox2(const Face& t, const BoundingBox& v, BoundingBox& clipped)
{
    // xmin
    std::vector<int> less, greater;
    less.reserve(3);
    greater.reserve(3);

    // divide the triangle vertices in less than and bigger than
    for (int i = 0; i < 3; i++)
    {
        if (t.v[i].x < v.min.x)
            less.emplace_back(i);
        else
            greater.emplace_back(i);
    }

    // we have intersections
    if (!less.empty() && !greater.empty())
    {
        glm::vec3 p[2];
        {
            const glm::vec3* alone;
            const glm::vec3* otherside[2];
            if (less.size() == 1)
            {
                alone = &t.v[less[0]];
                otherside[0] = &t.v[greater[0]];
                otherside[1] = &t.v[greater[1]];
            }
            else
            {
                alone = &t.v[greater[0]];
                otherside[0] = &t.v[less[0]];
                otherside[1] = &t.v[less[1]];
            }
            // Now we know wich vertex is alone in one of the plane sides.
            // So we can find the two vectors of the triangle that certainly intersect the plane.
            glm::vec3 vec[2];
            vec[0] = *otherside[0] - *alone;
            vec[1] = *otherside[1] - *alone;

            glm::vec3 planeP;
            planeP = v.min;

            glm::vec3 planeN;
            planeN.x = 1.f;
            planeN.y = 0.f;
            planeN.z = 0.f;

            // Find the two intersection points
            for (int i = 0; i < 2; i++)
            {
                // Check if the side of the triangle (line segment) intersects the plane
                float dot = glm::dot(planeN, vec[i]);
                if (dot == 0.f)
                    continue;
                float t = glm::dot(planeN, planeP - *alone) / dot;
                if (t >= 0.f && t <= 1.f)
                    p[i] = *alone + t * vec[i];
            }
        }

        // p[0] and p[1] form a line segment that lies on the plane.
        // We clip this line segment with the other planes
        glm::vec3* l;
        glm::vec3* r;

        if (p[0].y < p[1].y)
        {
            l = p; r = p + 1;
        }
        else
        {
            l = p + 1; r = p;
        }

        bool checkedymin = false;
        bool checkedymax = false;

        // Clip with ymin
        if (l->y < v.min.y && r->y > v.min.y)
        {
            glm::vec3 vec(*r - *l);
            glm::vec3 planeP;
            planeP = v.min;

            glm::vec3 planeN;
            planeN.x = 0.f;
            planeN.y = 1.f;
            planeN.z = 0.f;

            // Check if the side of the triangle (line segment) intersects the plane
            float dot = glm::dot(planeN, vec);
            if (dot == 0.f)
                std::cerr << "ClipTriangleToBox: 0 denominator." << std::endl;

            float t = glm::dot(planeN, planeP - *l) / dot;
            if (t >= 0.f && t <= 1.f)
                *l = *l + t * vec;
            checkedymin = true;
        }
        // Clip with ymax
        if (l->y < v.max.y && r->y > v.max.y)
        {
            glm::vec3 vec(*r - *l);
            glm::vec3 planeP;
            planeP = v.max;

            glm::vec3 planeN;
            planeN.x = 0.f;
            planeN.y = 1.f;
            planeN.z = 0.f;

            // Check if the side of the triangle (line segment) intersects the plane
            float dot = glm::dot(planeN, vec);
            if (dot == 0.f)
                std::cerr << "ClipTriangleToBox: 0 denominator." << std::endl;

            float t = glm::dot(planeN, planeP - *l) / dot;
            if (t >= 0.f && t <= 1.f)
                *r = *l + t * vec;
            checkedymax = true;
        }
    }
    return false;

}
bool KDTree::clipTriangleToBox(const Face& t, const BoundingBox& v, BoundingBox& clipped)
{
    bool empty = true;
    // Check which of the vertices are inside the box
    bool allinside = true;
    for (int i = 0; i < 3; i++)
    {
        if (v.contains(t.v[i]))
        {
            clipped.expand(t.v[i]);
            empty = false;
        }
        else
            allinside = false;
    }
    // All vertices are inside the box
    if (allinside)
        return true;


    std::vector<int> left, right;
    left.reserve(3);
    right.reserve(3);

    // We have two planes per dim to test, so 6 iterations.
    // Each consecutive pair of iterations is for one dim. So, to get the dim, we take the integer division of j by 2.
    // And if j is even, we are testing the min plane of the current dim.
    for (int j = 0; j < 6; j++)
    {
        // Get dim
        int dim = j / 2;
        // Min dim plane test.
        left.clear();
        right.clear();

        for (int i = 0; i < 3; i++)
        {
            // Test min plane case j is even, otherwise test max plane
            if (j % 2 == 0)
            {
                if (t.v[i][dim] < v.min[dim])
                    left.push_back(i);
                else
                    right.push_back(i);
            }
            else
            {
                if (t.v[i][dim] > v.max[dim])
                    left.push_back(i);
                else
                    right.push_back(i);
            }
        }

        // Not all vertices reside in the same side.
        // We must find the intersection.
        if (left.size() != 3 && right.size() != 3)
        {
            //std::cout << "Intersection! dim: " << dim << ((j%2 == 0) ? " min" : " max") << std::endl;
            const glm::vec3* alone;
            const glm::vec3* otherside[2];
            if (left.size() == 1)
            {
                alone = &t.v[left[0]];
                otherside[0] = &t.v[right[0]];
                otherside[1] = &t.v[right[1]];
            }
            else
            {
                alone = &t.v[right[0]];
                otherside[0] = &t.v[left[0]];
                otherside[1] = &t.v[left[1]];
            }
            // Now we know wich vertex is alone in one of the plane sides.
            // So we can find the two vectors of the triangle that certainly intersect the plane.
            glm::vec3 vec[2];
            vec[0] = *otherside[0] - *alone;
            vec[1] = *otherside[1] - *alone;

            glm::vec3 planeP;
            if (j % 2 == 0)
                planeP = v.min;
            else
                planeP = v.max;

            glm::vec3 planeN;
            switch (dim)
            {
            case 0:
                planeN.x = 1.f;
                planeN.y = 0.f;
                planeN.z = 0.f;
                break;
            case 1:
                planeN.x = 0.f;
                planeN.y = 1.f;
                planeN.z = 0.f;
                break;
            case 2:
                planeN.x = 0.f;
                planeN.y = 0.f;
                planeN.z = 1.f;
                break;
            }

            // This will tell us how many intersections between the triangle edges and the plane where found
            int intersectionCount = 0;

            for (int i = 0; i < 2; i++)
            {
                // Check if the side of the triangle (line segment) intersects the plane
                float dot = glm::dot(planeN, vec[i]);
                if (dot == 0.f)
                {
                    //std::cout << "ClipTriangleToBox: Triangle edge parallel to plane." << std::endl;
                    continue;
                }
                float t = glm::dot(planeN, planeP - *alone) / dot;
                if (t >= 0.f && t <= 1.f)
                {
                    glm::vec3 p = *alone + t * vec[i];
                    // We must check the the boundaries of the other two dimensions for the point p.
                    // E.g.:
                    // If we are in the dim 1(Y), dim0 will be
                    //		(1 + 1) % 3 = 2
                    // therefore the Z dimension. The dim1 will be
                    //		(1 + 2) % 3 = 0
                    // therefore X dimension.
                    int dim0 = (dim + 1) % 3;
                    int dim1 = (dim + 2) % 3;
                    if (p[dim0] >= v.min[dim0] && p[dim0] <= v.max[dim0] && p[dim1] >= v.min[dim1] && p[dim1] <=
v.max[dim1])
                    {
                        //std::cout << "P (" << p.x << ", " << p.y << ", " << p.z << ")" << std::endl;
                        clipped.expand(p);
                        empty = false;
                        intersectionCount++;
                    }
                }
            }

            // The trinagle will defenitily not be intersected by the edgeds of the box(in this specific plane).
            //if (intersectionCount == 2)
            //	continue;
            // Otherwise, we test all four edges of the box that lie on this plane
        }
    }

    // Now we need to intersect the box's edges with the triangle
    for (int dim = 0; dim < 3; dim++)
    {
        int dim0 = (dim + 1) % 3;
        int dim1 = (dim + 2) % 3;
        //std::cout << "dim " << dim << " dim0 " << dim0 << " dim1 " << dim1 << std::endl;
        Ray r;
        // dir is the same for the 4 edges
        r.direction = glm::vec3(0.f);
        r.direction[dim] = v.max[dim] - v.min[dim];

        float rt;

        r.origin = v.min;
        r.origin.x = v.min.x;
        r.origin.y = v.min.y;
        r.origin.z = v.min.z;
        // intersect r with t
        if (RTUtils::intersectRayTriangleMollerTrumboreNOBACKFACECULLING(r, t, rt))
        {
            if (rt >= 0.f && rt <= 1.0f)
            {
                clipped.expand(r.origin + rt*r.direction);
                empty = false;
            }
        }
        r.origin[dim0] = v.max[dim0];
        // intersect r with t
        if (RTUtils::intersectRayTriangleMollerTrumboreNOBACKFACECULLING(r, t, rt))
        {
            if (rt >= 0.f && rt <= 1.0f)
            {
                clipped.expand(r.origin + rt*r.direction);
                empty = false;
            }
        }
        r.origin[dim0] = v.min[dim0];
        r.origin[dim1] = v.max[dim1];
        // intersect r with t
        if (RTUtils::intersectRayTriangleMollerTrumboreNOBACKFACECULLING(r, t, rt))
        {
            if (rt >= 0.f && rt <= 1.0f)
            {
                clipped.expand(r.origin + rt*r.direction);
                empty = false;
            }
        }
        r.origin[dim0] = v.max[dim0];
        // intersect r with t
        if (RTUtils::intersectRayTriangleMollerTrumboreNOBACKFACECULLING(r, t, rt))
        {
            if (rt >= 0.f && rt <= 1.0f)
            {
                clipped.expand(r.origin + rt*r.direction);
                empty = false;
            }
        }
    }

    return !empty;
}

void KDTree::classify(const std::vector<BoundingBox>& ts, const Plane p, int& tl, int& tr, int& tp)
{
    // Number fo triangles to the left, to the right and the number of triangles that lie on the plane
    tl = 0; tr = 0; tp = 0;

    for (int i = 0; i < ts.size(); i++)
    {
        // If the bounding box lies on the plane, the triangle lies on the plane
        if (ts[i].min[p.dim] == p.value && ts[i].max[p.dim] == p.value)
        {
            tp++;
            continue;
        }
        // Otherwise, we have to check to which side of the plane the triangle is
        // Only to the left
        if (ts[i].max[p.dim] < p.value)
            tl++;
        // Only to the right
        else if (ts[i].min[p.dim] > p.value)
            tr++;
        // Both sides
        else
        {
            tl++;
            tr++;
        }
    }
}
void KDTree::classify(const std::vector<std::pair<BoundingBox, int>>& ts, const Plane p, std::vector<int>& tl,
std::vector<int>& tr, std::vector<int>& tp)
{
    // Number fo triangles to the left, to the right and the number of triangles that lie on the plane
    tl.clear();
    tr.clear();
    tp.clear();

    for (int i = 0; i < ts.size(); i++)
    {
        // If the bounding box lies on the plane, the triangle lies on the plane
        if (ts[i].first.min[p.dim] == p.value && ts[i].first.max[p.dim] == p.value)
        {
            tp.push_back(ts[i].second);
            continue;
        }
        // Otherwise, we have to check to which side of the plane the triangle is
        // Only to the left
        if (ts[i].first.max[p.dim] < p.value)
            tl.push_back(ts[i].second);
        // Only to the right
        else if (ts[i].first.min[p.dim] > p.value)
            tr.push_back(ts[i].second);
        // Both sides
        else
        {
            tl.push_back(ts[i].second);
            tr.push_back(ts[i].second);
        }
    }
}
void KDTree::classify(const std::vector<std::pair<BoundingBox, int>>& ts, const Plane p, int& tl, int& tr, int& tp)
{
    // Number fo triangles to the left, to the right and the number of triangles that lie on the plane
    tl = 0;
    tr = 0;
    tp = 0;
    for (int i = 0; i < ts.size(); i++)
    {
        // If the bounding box lies on the plane, the triangle lies on the plane
        if (ts[i].first.min[p.dim] == p.value && ts[i].first.max[p.dim] == p.value)
        {
            tp++;
            continue;
        }
        // Otherwise, we have to check to which side of the plane the triangle is
        // Only to the left
        if (ts[i].first.max[p.dim] < p.value)
            tl++;
        // Only to the right
        else if (ts[i].first.min[p.dim] > p.value)
            tr++;
        // Both sides
        else
        {
            tl++;
            tr++;
        }
    }
}

std::vector<Plane> KDTree::perfectSplit(const Face& t, const BoundingBox& v)
{
    std::vector<Plane> candidates;
    BoundingBox clippedv;

    if (clipTriangleToBox(t, v, clippedv))
    {
        for (int i = 0; i < 3; i++)
        {
            candidates.push_back(Plane(i, clippedv.max[i]));
            candidates.push_back(Plane(i, clippedv.min[i]));
        }
    }

    return candidates;
}
std::vector<Plane> KDTree::perfectSplit(const BoundingBox& b, const BoundingBox& v)
{
    std::vector<Plane> candidates;
    for (int i = 0; i < 3; i++)
    {
        // Planar triangle generates only one candidate
        if (b.max[i] == b.min[i])
        {
            candidates.push_back(Plane(i, b.min[i]));
        }
        else
        {
            candidates.push_back(Plane(i, b.max[i]));
            candidates.push_back(Plane(i, b.min[i]));
        }
    }
    return candidates;
}





bool KDTree::naiveSahPartition(const std::vector<Face>& ts, BoundingBox nodebb, Plane& splitP, std::vector<Face>&
leftTs, std::vector<Face>& rightTs)
{
    float minCost = std::numeric_limits<float>::max();
    Plane minCostPlane;
    Side minCostSide;
    std::vector<int> minCosttl, minCosttr, minCosttp;

    // Lets save the clipped triangles bounding box and their indices in the 'ts' vector here
    std::vector<std::pair<BoundingBox, int>> tsbb;
    // Clip all triangles and get the clipped triangles bounding box
    for (int i = 0; i < ts.size(); i++)
    {
        BoundingBox clipped;
        if (clipTriangleToBox(ts[i], nodebb, clipped))
            tsbb.push_back(std::pair<BoundingBox, int>(clipped, i));
    }
    // Get the candidate planes from each clipped triangle bounding box and
    // for each candidate calculate the cost
    for (int i = 0; i < tsbb.size(); i++)
    {
        std::vector<Plane> splitCandidates;
        splitCandidates = perfectSplit(tsbb[i].first, nodebb);
        // For each split candidate
        for (int j = 0; j < splitCandidates.size(); j++)
        {
            // Split the current node bb
            //BoundingBox vl, vr;
            //vl = nodebb;
            //vr = nodebb;
            //vl.max[splitCandidates[j].dim] = splitCandidates[j].value;
            //vl.min[splitCandidates[j].dim] = splitCandidates[j].value;
            // Get the number of triangles to the left(tl), to the right(tr) and the number that of t's that lie in the
plane(tp) std::vector<int> tl, tr, tp; classify(tsbb, splitCandidates[j], tl, tr, tp);
            // Get the cost
            // side will tell us to which side the triangles that lie on the plane should go in order to minimize the
cost

            Side side;
            float cost = sah(splitCandidates[j], nodebb, tl.size(), tr.size(), tp.size(), side);
            // This is the new split plane
            if (cost < minCost)
            {
                minCost = cost;
                minCostPlane = splitCandidates[j];
                minCostSide = side;
                minCosttl = tl;
                minCosttr = tr;
                minCosttp = tp;
            }
        }
    }

    // Now that we know the best split is time to check if not splitting at all is better
    // If the cost of not splitting(making this node a leaf) is better, we just... don't split
    if (cost(ts.size()) < minCost)
        return false;

    // Otherwise, we have to classify the triangles to the left or right child
    leftTs.clear();
    for (int i = 0; i < minCosttl.size(); i++)
        leftTs.push_back(ts[minCosttl[i]]);
    rightTs.clear();
    for (int i = 0; i < minCosttr.size(); i++)
        rightTs.push_back(ts[minCosttr[i]]);
    // Triangles laying on the plane go to the left
    if (minCostSide == Side::LEFT)
    {
        for (int i = 0; i < minCosttp.size(); i++)
            leftTs.push_back(ts[minCosttp[i]]);
    }
    // Triangles laying on the plane go to the right
    else
    {
        for (int i = 0; i < minCosttp.size(); i++)
            rightTs.push_back(ts[minCosttp[i]]);
    }

    splitP = minCostPlane;

    return true;
}
bool KDTree::eventSahPartition(const std::vector<Face>& ts, BoundingBox nodebb, Plane& splitP, std::vector<Face>&
leftTs, std::vector<Face>& rightTs)
{
    // Lets save the clipped triangles bounding box and their indices in the 'ts' vector here
    std::vector<std::pair<BoundingBox, int>> tsbb;
    // Clip all triangles and get the clipped triangles bounding box
    for (int i = 0; i < ts.size(); i++)
    {
        BoundingBox clipped;
        if (clipTriangleToBox(ts[i], nodebb, clipped))
            tsbb.push_back(std::pair<BoundingBox, int>(clipped, i));
    }

    float minCost = std::numeric_limits<float>::max();
    Plane minCostPlane;
    Side minCostSide;

    // For each dimension
    for (int dim = 0; dim < 3; dim++)
    {
        std::vector<Event> events;
        // First, compute sorted event list
        events.clear();
        // For each clipped triangle
        for (int i = 0; i < tsbb.size(); i++)
        {
            // If the triangle is planar
            if (tsbb[i].first.min[dim] == tsbb[i].first.max[dim])
            {
                events.push_back(Event(Plane(dim, tsbb[i].first.min[dim]), i, EVENT_TYPE_PLANAR));
            }
            else
            {
                events.push_back(Event(Plane(dim, tsbb[i].first.min[dim]), i, EVENT_TYPE_START));
                events.push_back(Event(Plane(dim, tsbb[i].first.max[dim]), i, EVENT_TYPE_END));
            }
        }
        // Sort
        std::sort(events.begin(), events.end(), eventCmp);



        // Iteratively �sweep� plane over all split candidates
        int nl = 0, np = 0, nr = tsbb.size();
        int esize = events.size();
        int i = 0;
        while (i < esize)
        {
            Plane p;
            p = events[i].plane;
            unsigned int pstart = 0, pplanar = 0, pend = 0;

            while ((i < esize) && (events[i].plane.value == p.value) && (events[i].type == EVENT_TYPE_END))
            {
                i++;
                pend++;
            }
            while ((i < esize) && (events[i].plane.value == p.value) && (events[i].type == EVENT_TYPE_PLANAR))
            {
                i++;
                pplanar++;
            }
            while ((i < esize) && (events[i].plane.value == p.value) && (events[i].type == EVENT_TYPE_START))
            {
                i++;
                pstart++;
            }

            np = pplanar;
            nr -= pplanar;
            //nr -= pend;
            nl += pstart;

            //int tl, tr, tp;
            //classify(tsbb, p, tl, tr, tp);
            //std::cout << "Naive (" << tl << " " << tp << " " << tr << ")" << std::endl;
            //std::cout << "Event (" << nl << " " << np << " " << nr << ")" << std::endl << std::endl;

            float cost;
            Side s;
            cost = sah(p, nodebb, nl, nr, np, s);
            if (cost < minCost)
            {
                minCost = cost;
                minCostPlane = p;
                minCostSide = s;
            }

            nr -= pend;
            //nl += pstart;
            nl += pplanar;
            np = 0;
        }
    }

    // Now that we know the best split is time to check if not splitting at all is better
    // If the cost of not splitting(making this node a leaf) is better, we just... don't split
    if (cost(ts.size()) < minCost)
        return false;

    // Otherwise, we have to classify the triangles to the left or right child


    // For each clipped triangle
    leftTs.clear();
    rightTs.clear();
    for (int i = 0; i < tsbb.size(); i++)
    {
        // If the triangle is planar to the best spliting plane
        if (tsbb[i].first.min[minCostPlane.dim] == tsbb[i].first.max[minCostPlane.dim])
        {
            if (minCostSide == Side::LEFT)
                leftTs.push_back(ts[tsbb[i].second]);
            else
                rightTs.push_back(ts[tsbb[i].second]);
        }
        else
        {
            // Otherwise, we have to check to which side of the plane the triangle is
            // Only to the left
            if (tsbb[i].first.max[minCostPlane.dim] < minCostPlane.value)
                leftTs.push_back(ts[tsbb[i].second]);
            // Only to the right
            else if (tsbb[i].first.min[minCostPlane.dim] > minCostPlane.value)
                rightTs.push_back(ts[tsbb[i].second]);
            // Both sides
            else
            {
                leftTs.push_back(ts[tsbb[i].second]);
                rightTs.push_back(ts[tsbb[i].second]);
            }
        }
    }

    splitP = minCostPlane;

    return true;
}





*/

/*KDTNode* KDTree::buildNode(int depth, int plane, std::vector<Face> faces, BoundingBox nodebb)
{
    KDTNode* node = new KDTNode();
    // Leaf
    if (depth >= maxDepth || faces.size() <= sizeToBecomeLeaf)
    {
        node->triangles = faces;
        node->isLeaf = true;
    }
    // Interior
    else
    {
        node->isLeaf = false;
        plane = plane % 3;
        node->interiorPayload.plane.dim = plane;
        // Find mid
        float mid = (nodebb.max[plane] + nodebb.min[plane]) * .5f;

        // Split triangles
        std::vector<Face> left, right;
        for (int i = 0; i < faces.size(); i++)
        {
            for (int j = 0; j < 3; j++)
            {
                if (faces[i].v[j][plane] <= mid)
                {
                    left.push_back(faces[i]);
                    break;
                }
            }
            for (int j = 0; j < 3; j++)
            {
                if (faces[i].v[j][plane] >= mid)
                {
                    right.push_back(faces[i]);
                    break;
                }
            }
        }
        float aux;
        aux = nodebb.max[plane];
        nodebb.max[plane] = mid;
        node->interiorPayload.children[0] = buildNode(depth + 1, plane + 1, left, nodebb);

        nodebb.max[plane] = aux;
        nodebb.min[plane] = mid;
        node->interiorPayload.children[1] = buildNode(depth + 1, plane + 1, right, nodebb);
    }

    return node;
}
KDTNode* KDTree::buildNodeSah(int depth, std::vector<Face> faces, BoundingBox nodebb)
{
    KDTNode* node = new KDTNode();
    nNodes++;

    // Definitely leaf.
    if (depth >= maxDepth || faces.size() <= sizeToBecomeLeaf)
    {
        node->triangles = faces;
        node->triangles.shrink_to_fit();
        nTriangles += node->triangles.size();
        nLeafs++;
        node->isLeaf = true;
        return node;
    }

    std::vector<Face> left, right;
    Plane splitP;

    // SAH spoke. Lets make this node a leaf.
    //if (!naiveSahPartition(faces, nodebb, splitP, left, right))
    if (!eventSahPartition(faces, nodebb, splitP, left, right))
    {
        node->triangles = faces;
        node->triangles.shrink_to_fit();
        nTriangles += node->triangles.size();
        nLeafs++;
        node->isLeaf = true;
        return node;
    }

    // Or a interior node.
    node->isLeaf = false;
    node->interiorPayload.plane = splitP;

    BoundingBox childbb = nodebb;
    childbb.max[splitP.dim] = splitP.value;
    node->interiorPayload.children[0] = buildNodeSah(depth + 1, left, childbb);

    childbb = nodebb;
    childbb.min[splitP.dim] = splitP.value;
    node->interiorPayload.children[1] = buildNodeSah(depth + 1, right, childbb);

    return node;
}*/

/*
void KDTree::presortedGenEventsFromClipped(const std::vector<std::pair<Face*, Side*>>& tfs, const BoundingBox& nodebb,
std::vector<PEvent>& events)
{
    // Lets save the clipped triangles bounding box and their indices in the 'ts' vector here
    std::vector<std::pair<BoundingBox, int>> tsbb;
    // Clip all triangles and get the clipped triangles bounding box
    for (int i = 0; i < tfs.size(); i++)
    {
        BoundingBox clipped;
        if (clipTriangleToBox(*tfs[i].first, nodebb, clipped))
            tsbb.emplace_back(clipped, i);
    }

    // Reserve all necessary memory at once
    events.reserve(tsbb.size() * 2 * 3);

    // Generate the events for all dimensions
    for (int dim = 0; dim < 3; dim++)
    {
        // For each clipped triangle
        for (int i = 0; i < tsbb.size(); i++)
        {
            // If the triangle is planar
            if (tsbb[i].first.min[dim] == tsbb[i].first.max[dim])
            {
                events.emplace_back(Plane(dim, tsbb[i].first.min[dim]), tfs[tsbb[i].second].first,
tfs[tsbb[i].second].second, EVENT_TYPE_PLANAR);
            }
            else
            {
                events.emplace_back(Plane(dim, tsbb[i].first.min[dim]), tfs[tsbb[i].second].first,
tfs[tsbb[i].second].second, EVENT_TYPE_START); events.emplace_back(Plane(dim, tsbb[i].first.max[dim]),
tfs[tsbb[i].second].first, tfs[tsbb[i].second].second, EVENT_TYPE_END);
            }
        }
    }
}

void KDTree::genEvents(const std::vector<Face>& ts, const BoundingBox& nodebb, std::vector<PEvent>& events)
{
    // Lets save the clipped triangles bounding box and their indices in the 'ts' vector here
    std::vector<std::pair<BoundingBox, const Face*>> tsbb;
    // Clip all triangles and get the clipped triangles bounding box
    for (int i = 0; i < ts.size(); i++)
    {
        BoundingBox clipped;
        if (clipTriangleToBox(ts[i], nodebb, clipped))
            tsbb.push_back(std::pair<BoundingBox, const Face*>(clipped, &ts[i]));
    }

    // Reserve all necessary memory at once
    events.reserve(tsbb.size() * 2 * 3);

    // Generate the events for all dimensions
    for (int dim = 0; dim < 3; dim++)
    {
        // For each clipped triangle
        for (int i = 0; i < tsbb.size(); i++)
        {
            // If the triangle is planar
            if (tsbb[i].first.min[dim] == tsbb[i].first.max[dim])
            {
                //events.push_back(PEvent(Plane(dim, tsbb[i].first.min[dim]), tsbb[i].second, EVENT_TYPE_PLANAR));
            }
            else
            {
                //events.push_back(PEvent(Plane(dim, tsbb[i].first.min[dim]), tsbb[i].second, EVENT_TYPE_START));
                //events.push_back(PEvent(Plane(dim, tsbb[i].first.max[dim]), tsbb[i].second, EVENT_TYPE_END));
            }
        }
    }
}



bool KDTree::intersect(const Ray& r, const Face** hitTriangle, float& hitDistance)
{
    float tmin, tmax;
    // Intersect ray with the tree's boundingbox
    if (!RTUtils::hitBoundingBox(r, bb, tmin, tmax))
        return false;

    float t = FLT_MAX;
    if (intersectNode(root, r, tmin, tmax, t, hitTriangle))
    {
        hitDistance = t;
        return true;
    }
    return false;
}
bool KDTree::intersectNode(KDTNode* node, const Ray& r, float tmin, float tmax, float& hitT, const Face** hitTriangle)
{
    float t = hitT;

    if (node->isLeaf)
    {
        // Intersect leaf
        // Return true if hit something and the t for the closest hit
        //if (RTUtils::intersectRayTrianglesMollerTrumbore(r, node->triangles, hitTriangle, t))
        if (RTUtils::intersectRayTrianglesMollerTrumbore(r, node->leafPayload.tris, node->leafPayload.nTris,
hitTriangle, t))
        {
            if (t < hitT)
                hitT = t;
            return true;
        }
        return false;
    }


    // Find first and second child (first to be intersected by the ray)
    KDTNode* first;
    KDTNode* second;

    int axis = node->interiorPayload.plane.dim;

    bool belowFirst = (r.origin[axis] < node->interiorPayload.plane.value) ||
        (r.origin[axis] == node->interiorPayload.plane.value && r.direction[axis] >= 0);
    if (belowFirst)
    {
        first = node->interiorPayload.children[0];
        second = node->interiorPayload.children[1];
    }
    else
    {
        first = node->interiorPayload.children[1];
        second = node->interiorPayload.children[0];
    }

    // Find t for the split plane
    float tplane;
    if (r.direction[axis] != 0.f)
        tplane = (node->interiorPayload.plane.value - r.origin[axis]) / r.direction[axis];
    else
        tplane = -1.f;

    const Face* hitf = NULL;
    bool hit = false;

    // We test only the first child
    if (tplane > tmax || tplane <= 0)
    {
        if (intersectNode(first, r, tmin, tmax, t, &hitf))
        {
            if (t < hitT)
            {
                *hitTriangle = hitf;
                hitT = t;
            }
            hit = true;
        }
    }
    // We test only the second child
    else if (tplane < tmin)
    {
        if (intersectNode(second, r, tmin, tmax, t, &hitf))
        {
            if (t < hitT)
            {
                *hitTriangle = hitf;
                hitT = t;
            }
            hit = true;
        }
    }
    // Ray intersects both children, so we test both.
    else
    {
        if (intersectNode(first, r, tmin, tplane, t, &hitf))
        {
            if (t < hitT)
            {
                *hitTriangle = hitf;
                hitT = t;
            }
            else
                t = hitT;
            //if (hitT < tplane)
            //	return true;
            hit = true;
        }
        if (intersectNode(second, r, tplane, tmax, t, &hitf))
        {
            if (t < hitT)
            {
                *hitTriangle = hitf;
                hitT = t;
            }
            //return true;
            hit = true;
        }
    }
    return hit;
}


*/