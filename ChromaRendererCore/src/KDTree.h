#pragma once

#include <ISpacePartitioningStructure.h>
#include <Object.h>
#include <Ray.h>
#include <glm/glm.hpp>
#include <vector>

struct Plane
{
    Plane() : dim(0), value(0)
    {
    }
    Plane(int dim, float value)
    {
        this->dim = dim;
        this->value = value;
    }
    int dim; // 0 = X, 1 = Y, 2 = Z
    float value;
};

struct KDTNode
{
    struct InteriorPayload
    {
        Plane plane;
        KDTNode* children[2];
    };

    struct LeafPayload
    {
        // std::vector<Face*> triangles;
        Face** tris;
        unsigned int nTris;
    };

    KDTNode() : isLeaf(false)
    {
        interiorPayload.children[0] = NULL;
        interiorPayload.children[1] = NULL;
    }
    bool isLeaf;

    union {
        InteriorPayload interiorPayload;
        LeafPayload leafPayload;
    };
    /*Plane plane;
    KDTNode* children[2];
    //std::vector<Face*> triangles;
    Face** tris;
    unsigned int nTris;/**/
};

struct KDTNodeToDo
{
    KDTNode* node;
    float tmin, tmax;
};

enum Side
{
    LEFT,
    RIGHT,
    BOTH
};

#define EVENT_TYPE_END 0
#define EVENT_TYPE_PLANAR 1
#define EVENT_TYPE_START 2

struct Event
{
    Event() : triangle(0), type(0)
    {
    }
    Event(Plane p, int tri, int type)
    {
        plane = p;
        triangle = tri;
        this->type = type;
    }
    Plane plane;  // The spliting candidate plane
    int triangle; // Triangle that generated the event
    int type;     // Type +, - or |
};

struct PEvent
{
    PEvent() : triangle(NULL), type(0)
    {
    }
    PEvent(Plane p, Face* tri, Side* s, int type)
    {
        plane = p;
        triangle = tri;
        this->type = type;
        flag = s;
    }
    Plane plane;    // The spliting candidate plane
    Face* triangle; // Triangle that generated the event
    int type;       // Type +, - or |
    Side* flag;
};

struct TFpointers
{
    Face* triangle;
    Side* flag;
    TFpointers(Face* t, Side* f)
    {
        triangle = t;
        flag = f;
    }
};

class KDTree : public ISpacePartitioningStructure
{
  public:
    bool build(std::vector<Mesh*>& meshes)
    {
        return false;
    }

    BoundingBox bb;
    KDTNode* root;

    bool stop = false;

    unsigned int nNodes;
    unsigned int nLeafs;
    unsigned int nTriangles;
    unsigned int depth;

    // Building parameters
    unsigned int maxDepth = 24;
    unsigned int sizeToBecomeLeaf = 5;
    // Cost function parameters
    float emptySpaceBonus = 1.0f;
    float KT = 1.0f; // Node traversal cost
    float KI = 1.5f; // Triangle intersection cost

    KDTree();
    ~KDTree();

    void abort(void)
    {
        stop = true;
    }

    long sizeInBytes(void);
    void printInfo();

    float cost(const float probabilityL, const float probabilityR, const int triangleCountL, const int triangleCountR);
    float cost(const int triangleCount);
    float sah(const Plane p,
              const BoundingBox& v,
              const int triangleCountL,
              const int triangleCountR,
              const int triangleCountP,
              Side& side);
    float sah(KDTNode* node);

    // bool clipTriangleToBox(const Face& t, const BoundingBox& v, BoundingBox& clipped);
    // std::vector<Plane> perfectSplit(const Face& t, const BoundingBox& v);
    // std::vector<Plane> perfectSplit(const BoundingBox& b, const BoundingBox& v);
    // void classify(const std::vector<BoundingBox>& ts, /*const BoundingBox& vleft, const BoundingBox& vright,*/ const
    // Plane p, int& tl, int& tr, int& tp); void classify(const std::vector<std::pair<BoundingBox, int>>& ts, const
    // Plane p, std::vector<int>& tl, std::vector<int>& tr, std::vector<int>& tp); void classify(const
    // std::vector<std::pair<BoundingBox, int>>& ts, const Plane p, int& tl, int& tr, int& tp);

    // bool naiveSahPartition(const std::vector<Face>& ts, BoundingBox nodebb, Plane& splitP, std::vector<Face>& leftTs,
    // std::vector<Face>& rightTs); bool eventSahPartition(const std::vector<Face>& ts, BoundingBox nodebb, Plane&
    // splitP, std::vector<Face>& leftTs, std::vector<Face>& rightTs);
    float presortedEventFindBestPlane(const int triangles,
                                      const std::vector<PEvent>& events,
                                      const BoundingBox& nodebb,
                                      Side& side,
                                      Plane& plane);

    // void presortedGenEventsFromClipped(const std::vector<std::pair<Face*, Side*>>& tfs, const BoundingBox& nodebb,
    // std::vector<PEvent>& events);
    void presortedGenEvents(const std::vector<TFpointers>& tfs, const BoundingBox& nodebb, std::vector<PEvent>& events);
    // void genEvents(const std::vector<Face>& ts, const BoundingBox& nodebb, std::vector<PEvent>& events);

    // KDTNode* buildNode(int depth, int plane, std::vector<Face> faces, BoundingBox nodebb);
    // KDTNode* buildNodeSah(int depth, std::vector<Face> faces, BoundingBox nodebb);
    KDTNode* presortedBuildNodeSah(int depth,
                                   std::vector<TFpointers>& triandflag,
                                   std::vector<PEvent>& events,
                                   BoundingBox& nodebb);
    void presortedClassify(const std::vector<TFpointers>& tfs,
                           const std::vector<PEvent>& events,
                           const Plane& splitPlane,
                           const Side& splitPlaneSide);
    void presortedSplitEvents(const std::vector<PEvent>& events,
                              const std::vector<TFpointers>& tfs,
                              const BoundingBox& nodebb,
                              const Plane& splitplane,
                              std::vector<PEvent>& eventsl,
                              std::vector<PEvent>& eventsr);

    // bool intersectNode(KDTNode* node, const Ray& r, float tmin, float tmax, float& hitT, const Face** hitTriangle);

    bool build(std::vector<Object>& objects);
    // bool intersect(const Ray& r, const Face** hitTriangle, float& hitDistance);
    // bool intersectNonRecursive(const Ray& r, const Face** hitTriangle, float& hitDistance) const;
    // bool intersectNonRecursiveP(const Ray& r, const Face** hitTriangle, float& hitDistance) const;
    bool intersect(const Ray& r, Intersection& intersection) const;

    KDTNode* free(KDTNode* node);
};