#pragma once

#include <Intersection.h>
#include <Material.h>
#include <Ray.h>
#include <glm/glm.hpp>
//#include <ChromaRenderer/IIntersectable.h>

#define EPSILON 0.000001f

class Triangle
//: public IIntersectable
{
  public:
    uint32_t v[3]; // Vertices' indices
    uint32_t n[3]; // Normals' indices

    std::vector<glm::vec3>* vdata; // Pointer to vertex data
    std::vector<glm::vec3>* ndata; // Pointer to normal data

    Material* material;

    // glm::vec3 tn;						// Precalculated plane normal for the intersection test
    // glm::vec3 edgesMollerTrumbore[2];	// Precalculated edges for the intersection test

    Triangle() : vdata(NULL), ndata(NULL)
    {
    }

    ~Triangle()
    {
    }

    bool intersect(Ray& r, Intersection& intersection) const
    {
        // const glm::vec3 &edge0 = edgesMollerTrumbore[0];
        // const glm::vec3 &edge1 = edgesMollerTrumbore[1];
        const glm::vec3 edge0 = *getVertex(1) - *getVertex(0);
        const glm::vec3 edge1 = *getVertex(2) - *getVertex(0);
        const glm::vec3 pvec = glm::cross(r.direction, edge1);
        const float det = glm::dot(edge0, pvec);

        // If determinant is near zero, ray lies in plane of triangle
        // if (det < EPSILON)							// With backface culling
        if (det > -EPSILON && det < EPSILON) // Without backface culling
            return false;
        const float invDet = 1.0f / det;
        const glm::vec3 tvec = r.origin - (*vdata)[v[0]];
        float u = glm::dot(tvec, pvec) * invDet;
        // The intersection lies outside of the triangle
        if (u < 0.0f || u > 1.0f)
            return false;
        const glm::vec3 qvec = glm::cross(tvec, edge0);
        float vv = glm::dot(r.direction, qvec) * invDet;
        // The intersection lies outside of the triangle
        if (vv < 0.0f || u + vv > 1.0f)
            return false;
        float t = glm::dot(edge1, qvec) * invDet;
        // Ray intersection
        // if (t < EPSILON)
        //	return false;
        if (t > r.maxt || t < r.mint)
            return false;

        // Intersection
        r.maxt = t;
        intersection.distance = t;
        // Calcula hitpoint
        intersection.p = r.origin + intersection.distance * r.direction;
        // Calcula as coordenadas baricentricas
        // const glm::vec3* v0 = is.object->getVertex(is.face, 0);
        // const glm::vec3* v1 = is.object->getVertex(is.face, 1);
        // const glm::vec3* v2 = is.object->getVertex(is.face, 2);
        // float div = 1.0f / glm::dot(glm::cross((*v1 - *v0), (*v2 - *v0)), triangle->tn);
        // float alpha = glm::dot(glm::cross((*v2 - *v1), (hitPoint - *v1)), triangle->tn) * div;
        // float beta = glm::dot(glm::cross((*v0 - *v2), (hitPoint - *v2)), triangle->tn)*div;
        // float gama = glm::dot(glm::cross((f2[i].v[1] - f2[i].v[0]), (point-f2[i].v[0])), f2[i].tn)*div;
        // float gama = 1.0f - (alpha + beta);
        float gama = 1.0f - (u + vv);
        // Calcula normal do ponto
        // glm::vec3 hitNormal = alpha * (*n0) + beta * (*n1) + gama * (*n2);
        intersection.n = u * (*getNormal(1)) + vv * (*getNormal(2)) + gama * (*getNormal(0));
        intersection.n = glm::normalize(intersection.n);

        intersection.material = material;

        return true;
    }
    
    void precomputeStuff()
    {
        // Calcula normal do plano
        // tn = glm::cross((*getVertex(1) - *getVertex(0)), (*getVertex(2) - *getVertex(0)));
        // tn = glm::normalize(tn);
        // Mollertrumbore edges
        // edgesMollerTrumbore[0] = *getVertex(1) - *getVertex(0);
        // edgesMollerTrumbore[1] = *getVertex(2) - *getVertex(0);
    }

    inline glm::vec3* getVertex(size_t i) const
    {
        return &(*vdata)[v[i]];
    }

    inline glm::vec3* getNormal(size_t i) const
    {
        return &(*ndata)[n[i]];
    }
};