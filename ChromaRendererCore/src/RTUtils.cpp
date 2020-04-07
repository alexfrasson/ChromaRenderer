#include "RTUtils.h"
#include "Config.h"
#include <algorithm>

#define EPSILON 0.000001f

#define NUMDIM  3
#define RIGHT	0
#define LEFT	1
#define MIDDLE	2




void RTUtils::trace(Scene &scene, Image &img, RendererSettings &settings, std::atomic<int>& pixelcount, Interval interval, bool& abort)
{
	std::vector<Ray> rays;
	rays.reserve(settings.samplesperpixel);

	//Para cada pixel
	for (int i = interval.fromWidth; i < interval.toWidth; i++)
	{
		if (abort)
			return;

		for (int j = interval.fromHeight; j < interval.toHeight; j++)
		{
			//Calcula raios
			scene.camera.rayDirection(i, j, rays, settings.samplesperpixel);

			float diffuseColorSum = 0.0f;
			float treeNodeColor = 0.f;
			bool hitSomething = false;
			//Para todos os raios do pixel
			const int raysVecSize = rays.size();
			for (int k = 0; k < raysVecSize; k++)
			{
				/*const Face* hitTriangle0;
				const Face* hitTriangle1;
				float hitDistance0, hitDistance1;
				if(RTUtils::closestIntersectionRayObjectsKensler(ray, scene.objects, &hitTriangle0, hitDistance0))
				{
				if(RTUtils::closestIntersectionRayObjects2(ray, scene.objects, &hitTriangle1, hitDistance1))
				{
				hitSomething = true;
				//if(hitDistance0 == hitDistance1)
				if(hitTriangle0 == hitTriangle1)
				diffuseColorSum += 1.0f;
				//diffuseColorSum += RTUtils::calcColor(ray, hitTriangle0, hitDistance1);

				}
				}/**/
				/**/
				//Testa colisão com todos objetos
				//const Face* hitTriangle;
				//float hitDistance;
				Intersection intersection = Intersection();
				intersection.n = glm::vec3();
				intersection.p = glm::vec3();

				float nNodeHitsNormalized;
				if (scene.sps->intersect(rays[k], intersection))
				//if (RTUtils::closestIntersectionRayObjects(rays[k], scene.objects, &hitTriangle, hitDistance, settings.boundingboxtest))
				{
					hitSomething = true;
					//diffuseColorSum += RTUtils::calcColor(rays[k], intersection.triangle, intersection.distance);
					diffuseColorSum += RTUtils::calcColor(intersection);
				}
				/**/
				else
					diffuseColorSum += 0.1f;	//Background
				//treeNodeColor += nNodeHitsNormalized;
			}
			float diffuseColorAverage = diffuseColorSum;
			diffuseColorAverage /= raysVecSize;		//Color average
			treeNodeColor /= raysVecSize;
			treeNodeColor = (unsigned char)(255 * treeNodeColor);
			unsigned char color = (unsigned char)(255 * diffuseColorAverage);

			uint8_t red = color * 0.2f;
			uint8_t green = color * 0.2f;
			//green += treeNodeColor * 0.8f;
			uint8_t blue = color * 0.2f;

			//img.setColor(i, j, red, green, blue);
			img.setColor(i, j, color, color, color, 255);
		}
		pixelcount += interval.toHeight - interval.fromHeight;
	}
}

// This was taken from
// [WBMS05] Williams, Amy, Steve Barrus, R.Keith Morley, and Peter Shirley. "An efficient and robust ray-box intersection algorithm." In ACM SIGGRAPH 2005 Courses, p. 9. ACM, 2005.
bool RTUtils::hitBoundingBoxSlab(const BoundingBox& bb, const Ray& r, const glm::vec3& invRayDir, const glm::bvec3& dirIsNeg, float& tmin, float& tmax)
{
	float min = (bb[dirIsNeg[0]].x - r.origin.x) * invRayDir.x;
	float max = (bb[1 - dirIsNeg[0]].x - r.origin.x) * invRayDir.x;
	float tymin = (bb[dirIsNeg[1]].y - r.origin.y) * invRayDir.y;
	float tymax = (bb[1 - dirIsNeg[1]].y - r.origin.y) * invRayDir.y;
	if ((min > tymax) || (tymin > max))
		return false;
	if (tymin > min) min = tymin;
	if (tymax < max) max = tymax;
	
	tymin = (bb[dirIsNeg[2]].z - r.origin.z) * invRayDir.z;
	tymax = (bb[1 - dirIsNeg[2]].z - r.origin.z) * invRayDir.z;

	if ((min > tymax) || (tymin > max))
		return false;
	if (tymin > min) min = tymin;
	if (tymax < max) max = tymax;

	//if (max < 0)
	//	return false;

	return (min < tmax) && (max > tmin);
	//return max >= min;
}
bool RTUtils::hitBoundingBoxSlab(const BoundingBox& bb, const Ray& r, const glm::vec3& invRayDir, float& tmin, float& tmax)
{
	float tx1 = (bb.min.x - r.origin.x) * invRayDir.x;
	float tx2 = (bb.max.x - r.origin.x) * invRayDir.x;

	tmin = std::fminf(tx1, tx2);
	tmax = std::fmaxf(tx1, tx2);

	float ty1 = (bb.min.y - r.origin.y) * invRayDir.y;
	float ty2 = (bb.max.y - r.origin.y) * invRayDir.y;

	tmin = std::fmaxf(tmin, std::fminf(ty1, ty2));
	tmax = std::fminf(tmax, std::fmaxf(ty1, ty2));

	float tz1 = (bb.min.z - r.origin.z) * invRayDir.z;
	float tz2 = (bb.max.z - r.origin.z) * invRayDir.z;

	tmin = std::fmaxf(tmin, std::fminf(tz1, tz2));
	tmax = std::fminf(tmax, std::fmaxf(tz1, tz2));

	if (tmax < 0)
		return false;

	return tmax >= tmin;
}
bool hitBoundingBoxSlab(const BoundingBox& bb, const Ray& r)
{
	float tmin = -INFINITY, tmax = INFINITY;

	if (r.direction.x != 0.0)
	{
		float tx1 = (bb.min.x - r.origin.x) / r.direction.x;
		float tx2 = (bb.max.x - r.origin.x) / r.direction.x;

		tmin = std::fmaxf(tmin, std::fminf(tx1, tx2));
		tmax = std::fminf(tmax, std::fmaxf(tx1, tx2));
	}

	if (r.direction.y != 0.0)
	{
		float ty1 = (bb.min.y - r.origin.y) / r.direction.y;
		float ty2 = (bb.max.y - r.origin.y) / r.direction.y;

		tmin = std::fmaxf(tmin, std::fminf(ty1, ty2));
		tmax = std::fminf(tmax, std::fmaxf(ty1, ty2));
	}

	return tmax >= tmin;
}


bool RTUtils::hitBoundingBox(const Ray& r, const BoundingBox& bb)
{
	//double minB[NUMDIM], maxB[NUMDIM];		/*box */
	//double origin[NUMDIM], dir[NUMDIM];		/*ray */
	//double coord[NUMDIM];				/* hit point */
	
	glm::vec3 coord;

	bool inside = true;
	char quadrant[NUMDIM];
	register int i;
	int whichPlane;
	float maxT[NUMDIM];
	float candidatePlane[NUMDIM];

	/* Find candidate planes; this loop can be avoided if
   	rays cast all from the eye(assume perpsective view) */
	for (i = 0; i < NUMDIM; i++)
	{
		if(r.origin[i] < bb.min[i]) {
			quadrant[i] = LEFT;
			candidatePlane[i] = bb.min[i];
			inside = false;
		}
		else if (r.origin[i] > bb.max[i]) {
			quadrant[i] = RIGHT;
			candidatePlane[i] = bb.max[i];
			inside = false;
		}else	{
			quadrant[i] = MIDDLE;
		}
	}
	
	// Ray origin inside bounding box
	if(inside)
	{
		coord = r.origin;
		return true;
	}


	/* Calculate T distances to candidate planes */
	for (i = 0; i < NUMDIM; i++)
	{
		if (quadrant[i] != MIDDLE && r.direction[i] != 0.)
			maxT[i] = (candidatePlane[i] - r.origin[i]) / r.direction[i];
		else
			maxT[i] = -1.;
	}

	/* Get largest of the maxT's for final choice of intersection */
	whichPlane = 0;
	for (i = 1; i < NUMDIM; i++)
		if (maxT[whichPlane] < maxT[i])
			whichPlane = i;

	/* Check final candidate actually inside box */
	if (maxT[whichPlane] < 0.) 
		return false;
	for (i = 0; i < NUMDIM; i++)
	{
		if (whichPlane != i) {
			coord[i] = r.origin[i] + maxT[whichPlane] * r.direction[i];
			if (coord[i] < bb.min[i] || coord[i] > bb.max[i])
				return false;
		} else {
			coord[i] = candidatePlane[i];
		}
	}
	return true;				/* ray hits box */
}
// Find if there is an intersection between the ray and the box.
// If exists one intersection, find the parametric values for the entry and for the out intersection point.
// The parametric point can be calculated as follows:
//		Point = RayOrigin + RayDir * t
// Where t is the parametric value.
bool RTUtils::hitBoundingBox(const Ray& r, const BoundingBox& bb, float& tnear, float& tfar)
{
	bool inside = true;
	char quadrant[NUMDIM];
	register int i;
	glm::vec3 entryCoord;
	int entryMaxTPlane;
	float entryT[NUMDIM];
	float entryCandidate[NUMDIM];

	/* Find candidate planes; this loop can be avoided if
	rays cast all from the eye(assume perpsective view) */
	for (i = 0; i < NUMDIM; i++)
	{
		if (r.origin[i] < bb.min[i]) 
		{
			quadrant[i] = LEFT;
			entryCandidate[i] = bb.min[i];
			inside = false;
		}
		else if (r.origin[i] > bb.max[i]) 
		{
			quadrant[i] = RIGHT;
			entryCandidate[i] = bb.max[i];
			inside = false;
		}
		else
			quadrant[i] = MIDDLE;
	}

	// Ray origin inside bounding box.
	// The entry point is the actual origin and the t for the origin is 0.
	if (inside)
	{
		tnear = .0f;
		entryCoord = r.origin;
	}
	else
	{
		/* Calculate T distances to candidate planes */
		for (i = 0; i < NUMDIM; i++)
		{
			if (quadrant[i] != MIDDLE && r.direction[i] != 0.)
				entryT[i] = (entryCandidate[i] - r.origin[i]) / r.direction[i];
			else
				entryT[i] = -1.f;
		}

		/* Get largest of the T's for final choice of intersection */
		entryMaxTPlane = 0;
		for (i = 1; i < NUMDIM; i++)
			if (entryT[entryMaxTPlane] < entryT[i])
				entryMaxTPlane = i;

		/* Check final candidate actually inside box */
		if (entryT[entryMaxTPlane] < 0.f)
			return false;
		for (i = 0; i < NUMDIM; i++)
		{
			if (entryMaxTPlane != i) {
				entryCoord[i] = r.origin[i] + entryT[entryMaxTPlane] * r.direction[i];
				if (entryCoord[i] < bb.min[i] || entryCoord[i] > bb.max[i])
					return false;
			}
			else {
				entryCoord[i] = entryCandidate[i];
			}
		}
		tnear = entryT[entryMaxTPlane];
	}

	// If we got til here, the ray hits box. 
	// That means that it must leave the box at some point.

	int outMinTPlane;
	float outT[NUMDIM];
	float outCandidate[NUMDIM];

	for (i = 0; i < NUMDIM; i++)
	{
		if (r.origin[i] < bb.min[i])
			outCandidate[i] = bb.max[i];
		else if (r.origin[i] > bb.max[i])
			outCandidate[i] = bb.min[i];
		else
		{
			// If the origin for this plane is inside, we check the direction.
			if (r.direction[i] > 0)
				outCandidate[i] = bb.max[i];
			else if (r.direction[i] < 0)
				outCandidate[i] = bb.min[i];
		}
	}

	/* Calculate T distances to candidate planes */
	for (i = 0; i < NUMDIM; i++)
	{
		if (r.direction[i] != 0.f)
			outT[i] = (outCandidate[i] - r.origin[i]) / r.direction[i];
		else
			outT[i] = FLT_MAX;
			//outT[i] = -1.f;
	}

	/* Get smallest of the T's for final choice of intersection */
	outMinTPlane = 0;
	for (i = 1; i < NUMDIM; i++)
		if (outT[i] < outT[outMinTPlane])
			outMinTPlane = i;

	// Since we are sure that the ray entered the box, outT[outMinTPlane] 
	// should be the value we where looking for.

	tfar = outT[outMinTPlane];
	return true;
}

float RTUtils::calcColor(const Intersection &is)
{
	Ray lightRay;
	lightRay.origin = glm::vec3(-20, 20, 20);
	lightRay.direction = is.p - lightRay.origin;
	lightRay.direction = glm::normalize(lightRay.direction);

	float diffuseColorIntensity = glm::dot(is.n, lightRay.direction);
	if (diffuseColorIntensity >= 0)
		diffuseColorIntensity = 0;
	else
		diffuseColorIntensity *= -1;
	if (diffuseColorIntensity > 1)
		diffuseColorIntensity = 1;

	return diffuseColorIntensity;
}
float RTUtils::calcColor(const Ray &ray, const Face* triangle, const float distance)
{
	//Calcula hitpoint
	glm::vec3 hitPoint = ray.origin + distance*ray.direction;

	Ray lightRay;
	lightRay.origin = glm::vec3(-20, 20, 20);
	lightRay.direction = hitPoint - lightRay.origin;
	lightRay.direction = glm::normalize(lightRay.direction);
	
	//calcula as coordenadas baricentricass
	float div = 1.0f/glm::dot(glm::cross((triangle->v[1] - triangle->v[0]), (triangle->v[2] - triangle->v[0])), triangle->tn);
	float alpha = glm::dot(glm::cross((triangle->v[2] - triangle->v[1]), (hitPoint-triangle->v[1])), triangle->tn)*div;
	float beta = glm::dot(glm::cross((triangle->v[0] - triangle->v[2]), (hitPoint-triangle->v[2])), triangle->tn)*div;
	//float gama = glm::dot(glm::cross((f2[i].v[1] - f2[i].v[0]), (point-f2[i].v[0])), f2[i].tn)*div;
	float gama = 1.0f - (alpha + beta);
	//calcula normal do ponto
	glm::vec3 hitNormal = alpha*triangle->n[0] + beta*triangle->n[1] + gama*triangle->n[2];
	hitNormal = glm::normalize(hitNormal);

	float diffuseColorIntensity = glm::dot(hitNormal, lightRay.direction);
	if(diffuseColorIntensity >= 0)
		diffuseColorIntensity = 0;
	else
		diffuseColorIntensity *= -1;
	if(diffuseColorIntensity > 1)
		diffuseColorIntensity = 1;
	
	return diffuseColorIntensity;
}
bool RTUtils::closestIntersectionRayObjects(const Ray &ray, const std::vector<Object> &objects, const Face** hitTriangle, float &hitDistance, const bool boundingboxtest)
{
	bool hit = false;
	hitDistance = std::numeric_limits<float>::infinity();
	const unsigned int vectorSize = objects.size();
	for(unsigned int i = 0; i < vectorSize; i++)
	{
		float distance;
		const Face* triangle;
		if (boundingboxtest)
			if(!RTUtils::hitBoundingBox(ray, objects[i].boundingBox))
				continue;
		#ifdef SIMD
		if(RTUtils::intersectRayObjectMollerTrumboreSIMD128(ray, objects[i], &triangle, distance))
		#else
		if(RTUtils::intersectRayObjectMollerTrumbore(ray, objects[i], &triangle, distance))
		#endif
		{
			if(distance < hitDistance)
			{
				hitDistance = distance;
				*hitTriangle = triangle;
				hit = true;
			}
		}
	}
	return hit;
}
bool RTUtils::intersectUntilRayObjects(const Ray &ray, const std::vector<Object> &objects, const glm::vec3 point, const bool boundingboxtest)
{
	float lightDistance = glm::distance(point, ray.origin) - 0.001f;
	const unsigned int vectorSize = objects.size();
	for(unsigned int i = 0; i < vectorSize; i++)
		if(RTUtils::intersectShadowRay(ray, lightDistance, objects[i], boundingboxtest))
				return true;
	return false;
}
bool RTUtils::intersectShadowRay(const Ray &r, const float &lightDistance, const Object &o, const bool boundingboxtest)
{
	if(boundingboxtest)
		if(!RTUtils::hitBoundingBox(r, o.boundingBox))
			return false;

	bool hit = false;
	//para cada triangulo
	const unsigned int size = o.f.size();
	for(unsigned int i = 0; i < size; i++)
	{
		float u, v, t;
		if(RTUtils::intersectRayTriangleMollerTrumbore(r, o.f[i], u, v, t))
			if(t < lightDistance)
				return true;
	}
	return hit;
}

void RTUtils::rayDirection(const Camera &camera, const int i, const int j, Ray* rays, const int numRays)
{
	/*const int numDivs = 2;	//Number of subcells = numDivs^2

	const float step = 1.0f/numDivs;

	for(float k = -0.5f; k < 0.49999f; k += step)
	{
		for(float l = -0.5f; l < 0.49999f; l += step)
		{
			rays[].origin = eye;
			ray.direction = (float)(i + l - width*0.5f) * u + (float)(j + k - height*0.5f) * v - d * w;
			ray.direction = glm::normalize(ray.direction);
			rays.push_back(ray);*
			rays.emplace_back(eye, glm::normalize((float)(i + l - width*0.5f) * u + (float)(j + k - height*0.5f) * v - d * w));
		}
	}
	*/
}


bool RTUtils::intersectRayTrianglesMollerTrumbore(const Ray &r, const std::vector<Face>& f, Intersection& intersection)
{
	float hitU, hitV;											//Hit barycentric coordinates 
	float hitT = std::numeric_limits<float>::infinity();		//Distance of closest hit
	unsigned int hitIndex;										//Index of the closest triangle
	bool hit = false;
	//para cada triangulo
	const unsigned int size = f.size();
	for (unsigned int i = 0; i < size; i++)
	{
		float u, v, t;
		if (intersectRayTriangleMollerTrumbore(r, f[i], u, v, t))
		{
			if (t < hitT)
			{
				hit = true;
				hitU = u;
				hitV = v;
				hitT = t;
				hitIndex = i;
				intersection.distance = t;
			}
		}
	}
	if (hit)
		intersection.triangle = &f[hitIndex];
	return hit;
}
bool RTUtils::intersectRayTrianglesMollerTrumbore(const Ray &r, const std::vector<Face*>& f, Intersection& intersection)
{
	float hitU, hitV;											//Hit barycentric coordinates 
	float hitT = std::numeric_limits<float>::infinity();		//Distance of closest hit
	unsigned int hitIndex;										//Index of the closest triangle
	bool hit = false;
	//para cada triangulo
	const unsigned int size = f.size();
	for (unsigned int i = 0; i < size; i++)
	{
		float u, v, t;
		if (intersectRayTriangleMollerTrumbore(r, *f[i], u, v, t))
		{
			if (t < hitT)
			{
				hit = true;
				hitU = u;
				hitV = v;
				hitT = t;
				hitIndex = i;
				intersection.distance = t;
			}
		}
	}
	if (hit)
		intersection.triangle = f[hitIndex];
	return hit;
}
bool RTUtils::intersectRayTrianglesMollerTrumbore(const Ray &r, Face** f, const unsigned int nFaces, Intersection& intersection)
{
	float hitU, hitV;											//Hit barycentric coordinates 
	float hitT = std::numeric_limits<float>::infinity();		//Distance of closest hit
	unsigned int hitIndex;										//Index of the closest triangle
	bool hit = false;
	//para cada triangulo
	for (unsigned int i = 0; i < nFaces; i++)
	{
		float u, v, t;
		if (intersectRayTriangleMollerTrumbore(r, *f[i], u, v, t))
		{
			if (t < hitT)
			{
				hit = true;
				hitU = u;
				hitV = v;
				hitT = t;
				hitIndex = i;
				intersection.distance = t;
			}
		}
	}
	if (hit)
		intersection.triangle = f[hitIndex];
	return hit;
}
bool RTUtils::intersectRayTriangleMollerTrumbore(const Ray &r, const Face &f, float &u, float &v, float &t)
{	
	const glm::vec3 &edge0 = f.edgesMollerTrumbore[0];
	const glm::vec3 &edge1 = f.edgesMollerTrumbore[1];
	const glm::vec3 pvec = glm::cross(r.direction, edge1);
	const float det = glm::dot(edge0, pvec);

	//if determinant is near zero, ray lies in plane of triangle
	if(det < EPSILON)						//with backface culling
	//if(det > -EPSILON && det < EPSILON)	//without backface culling
		return false;
	const float invDet = 1.0f / det;
	const glm::vec3 tvec = r.origin - f.v[0];
	u = glm::dot(tvec, pvec) * invDet;
	//The intersection lies outside of the triangle
	if(u < 0.0f || u > 1.0f)
		return false;
	const glm::vec3 qvec = glm::cross(tvec, edge0);
	v = glm::dot(r.direction, qvec) * invDet;
	//The intersection lies outside of the triangle
	if(v < 0.0f || u + v > 1.0f)
		return false;
	t = glm::dot(edge1, qvec) * invDet;
	//ray intersection
	if(t > EPSILON) 
		return true;
	return false;
}
bool RTUtils::intersectRayTriangleMollerTrumbore(const Ray &r, const Face* f, float &u, float &v, float &t)
{
	const glm::vec3 &edge0 = f->edgesMollerTrumbore[0];
	const glm::vec3 &edge1 = f->edgesMollerTrumbore[1];
	const glm::vec3 pvec = glm::cross(r.direction, edge1);
	const float det = glm::dot(edge0, pvec);

	//if determinant is near zero, ray lies in plane of triangle
	if (det < EPSILON)						//with backface culling
		//if(det > -EPSILON && det < EPSILON)	//without backface culling
		return false;
	const float invDet = 1.0f / det;
	const glm::vec3 tvec = r.origin - f->v[0];
	u = glm::dot(tvec, pvec) * invDet;
	//The intersection lies outside of the triangle
	if (u < 0.0f || u > 1.0f)
		return false;
	const glm::vec3 qvec = glm::cross(tvec, edge0);
	v = glm::dot(r.direction, qvec) * invDet;
	//The intersection lies outside of the triangle
	if (v < 0.0f || u + v > 1.0f)
		return false;
	t = glm::dot(edge1, qvec) * invDet;
	//ray intersection
	if (t > EPSILON)
		return true;
	return false;
}
bool RTUtils::intersectRayTriangleMollerTrumbore(const Ray &r, const Face &f, float &t)
{
	const glm::vec3 &edge0 = f.edgesMollerTrumbore[0];
	const glm::vec3 &edge1 = f.edgesMollerTrumbore[1];
	const glm::vec3 pvec = glm::cross(r.direction, edge1);
	const float det = glm::dot(edge0, pvec);

	//if determinant is near zero, ray lies in plane of triangle
	if (det < EPSILON)						//with backface culling
		//if(det > -EPSILON && det < EPSILON)	//without backface culling
		return false;
	const float invDet = 1.0f / det;
	const glm::vec3 tvec = r.origin - f.v[0];
	float u = glm::dot(tvec, pvec) * invDet;
	//The intersection lies outside of the triangle
	if (u < 0.0f || u > 1.0f)
		return false;
	const glm::vec3 qvec = glm::cross(tvec, edge0);
	float v = glm::dot(r.direction, qvec) * invDet;
	//The intersection lies outside of the triangle
	if (v < 0.0f || u + v > 1.0f)
		return false;
	t = glm::dot(edge1, qvec) * invDet;
	//ray intersection
	if (t > EPSILON)
		return true;
	return false;
}
bool RTUtils::intersectRayTriangleMollerTrumboreNOBACKFACECULLING(const Ray &r, const Face &f, float &t)
{
	const glm::vec3 &edge0 = f.edgesMollerTrumbore[0];
	const glm::vec3 &edge1 = f.edgesMollerTrumbore[1];
	const glm::vec3 pvec = glm::cross(r.direction, edge1);
	const float det = glm::dot(edge0, pvec);

	//if determinant is near zero, ray lies in plane of triangle
	//if (det < EPSILON)						//with backface culling
	if(det > -EPSILON && det < EPSILON)			//without backface culling
		return false;
	const float invDet = 1.0f / det;
	const glm::vec3 tvec = r.origin - f.v[0];
	float u = glm::dot(tvec, pvec) * invDet;
	//The intersection lies outside of the triangle
	if (u < 0.0f || u > 1.0f)
		return false;
	const glm::vec3 qvec = glm::cross(tvec, edge0);
	float v = glm::dot(r.direction, qvec) * invDet;
	//The intersection lies outside of the triangle
	if (v < 0.0f || u + v > 1.0f)
		return false;
	t = glm::dot(edge1, qvec) * invDet;
	//ray intersection
	if (t > EPSILON)
		return true;
	return false;
}
bool RTUtils::intersectRayObjectMollerTrumbore(const Ray &r, const Object &o, const Face** hitFace, float &hitDistance)
{
	float hitU, hitV;											//Hit barycentric coordinates 
	float hitT = std::numeric_limits<float>::infinity();		//Distance of closest hit
	unsigned int hitIndex;										//Index of the closest triangle
	bool hit = false;
	//para cada triangulo
	const unsigned int size = o.f.size();
	for(unsigned int i = 0; i < size; i++)
	{
		float u, v, t;
		if(intersectRayTriangleMollerTrumbore(r, o.f[i], u, v, t))
		{
			if(t < hitT)
			{
				hit = true;
				hitU = u;
				hitV = v;
				hitT = t;
				hitIndex = i;
				hitDistance = t;
			}
		}
	}
	if(hit)
		*hitFace = &o.f[hitIndex];
	return hit;
}
bool RTUtils::intersectRayTrianglesMollerTrumboreSIMD128(const Ray &r, const std::vector<Face>& f, const Face** hitFace, float &hitDistance)
{
	hitDistance = std::numeric_limits<float>::infinity();
	bool hit = false;
	unsigned int hitIndex;
	
	int sequentialsize = f.size() % 4;

	// Run a sequencial intersection in the remaining triangles
	for (unsigned int i = 0; i < sequentialsize; i++)
	{
		float t;
		if (intersectRayTriangleMollerTrumbore(r, f[i], t))
		{
			if (t < hitDistance)
			{
				hit = true;
				hitIndex = i;
				hitDistance = t;
			}
		}
	}

	//Load Ray Origin
	const __m128 ox = _mm_set_ps1(r.origin.x);
	const __m128 oy = _mm_set_ps1(r.origin.y);
	const __m128 oz = _mm_set_ps1(r.origin.z);
	//Load Ray Direction
	const __m128 dx = _mm_set_ps1(r.direction.x);
	const __m128 dy = _mm_set_ps1(r.direction.y);
	const __m128 dz = _mm_set_ps1(r.direction.z);
	//Constants
	const __m128 epsilon = _mm_set_ps1(EPSILON);
	const __m128 zeroes = _mm_setzero_ps();
	const __m128 ones = _mm_set_ps1(1.0f);

	//Loop over the triangles, computing four triangles at a time
	for (int i = sequentialsize; i + 3 < f.size(); i += 4)
	{
		const __m128 index = _mm_set_ps(i, i + 1, i + 2, i + 3);	//WHY????
		//Load edge 0 of each triangle
		const __m128 edge0x = _mm_set_ps(f[i].edgesMollerTrumbore[0].x,
			f[i + 1].edgesMollerTrumbore[0].x,
			f[i + 2].edgesMollerTrumbore[0].x,
			f[i + 3].edgesMollerTrumbore[0].x);
		const __m128 edge0y = _mm_set_ps(f[i].edgesMollerTrumbore[0].y,
			f[i + 1].edgesMollerTrumbore[0].y,
			f[i + 2].edgesMollerTrumbore[0].y,
			f[i + 3].edgesMollerTrumbore[0].y);
		const __m128 edge0z = _mm_set_ps(f[i].edgesMollerTrumbore[0].z,
			f[i + 1].edgesMollerTrumbore[0].z,
			f[i + 2].edgesMollerTrumbore[0].z,
			f[i + 3].edgesMollerTrumbore[0].z);
		//Load edge 1 of each triangle
		const __m128 edge1x = _mm_set_ps(f[i].edgesMollerTrumbore[1].x,
			f[i + 1].edgesMollerTrumbore[1].x,
			f[i + 2].edgesMollerTrumbore[1].x,
			f[i + 3].edgesMollerTrumbore[1].x);
		const __m128 edge1y = _mm_set_ps(f[i].edgesMollerTrumbore[1].y,
			f[i + 1].edgesMollerTrumbore[1].y,
			f[i + 2].edgesMollerTrumbore[1].y,
			f[i + 3].edgesMollerTrumbore[1].y);
		const __m128 edge1z = _mm_set_ps(f[i].edgesMollerTrumbore[1].z,
			f[i + 1].edgesMollerTrumbore[1].z,
			f[i + 2].edgesMollerTrumbore[1].z,
			f[i + 3].edgesMollerTrumbore[1].z);
		//Compute pvec = cross(raydir, edge1)
		const __m128 pvecx = _mm_sub_ps(
			_mm_mul_ps(dy, edge1z),
			_mm_mul_ps(dz, edge1y));
		const __m128 pvecy = _mm_sub_ps(
			_mm_mul_ps(dz, edge1x),
			_mm_mul_ps(dx, edge1z));
		const __m128 pvecz = _mm_sub_ps(
			_mm_mul_ps(dx, edge1y),
			_mm_mul_ps(dy, edge1x));
		//Compute det = dot(edge0, pvec)
		const __m128 det = _mm_add_ps(_mm_add_ps(
			_mm_mul_ps(pvecx, edge0x),
			_mm_mul_ps(pvecy, edge0y)),
			_mm_mul_ps(pvecz, edge0z));
		//If determinant is near zero, ray lies in plane of triangle
		const __m128 detmaskge = _mm_cmpge_ps(det, epsilon);				//det >= EPSILON
		//const __m128 minusepsilon = _mm_set_ps1(-EPSILON);
		//const __m128 detmasklt = _mm_cmplt_ps(det, minusepsilon);	//det < -EPSILON
		__m128 mask = detmaskge;									//with backface culling	(if(det > EPSILON))
		if (_mm_movemask_ps(mask) == 0x0)
			continue;
		//__m128 mask = _mm_or_ps(detmasklt, detmaskgt);			//without backface culling (if(det < -EPSILON || det > EPSILON))
		//const __m128 rcp = _mm_rcp_ps(det);								//Reciprocal inveDev = 1.0/dev7
		//const __m128 invDet = _mm_sub_ps(
		//		_mm_add_ps(rcp, rcp),
		//		_mm_mul_ps(_mm_mul_ps(rcp, rcp),
		//		det));
		const __m128 invDet = _mm_div_ps(ones, det);
		//const __m128 invDet = _mm_rcp_ps(det);
		//Load vertice 0 of each triangle
		const __m128 p0x = _mm_set_ps(f[i].v[0].x,
			f[i + 1].v[0].x,
			f[i + 2].v[0].x,
			f[i + 3].v[0].x);
		const __m128 p0y = _mm_set_ps(f[i].v[0].y,
			f[i + 1].v[0].y,
			f[i + 2].v[0].y,
			f[i + 3].v[0].y);
		const __m128 p0z = _mm_set_ps(f[i].v[0].z,
			f[i + 1].v[0].z,
			f[i + 2].v[0].z,
			f[i + 3].v[0].z);
		//Compute tvec = rayorigin - vertice0
		const __m128 tvecx = _mm_sub_ps(ox, p0x);
		const __m128 tvecy = _mm_sub_ps(oy, p0y);
		const __m128 tvecz = _mm_sub_ps(oz, p0z);
		//Compute u = dot(tvec, pvec) * invDet;
		const __m128 u = _mm_mul_ps(_mm_add_ps(_mm_add_ps(
			_mm_mul_ps(pvecx, tvecx),
			_mm_mul_ps(pvecy, tvecy)),
			_mm_mul_ps(pvecz, tvecz)), invDet);
		//Test u >= 0.0 && u <= 1.0
		const __m128 umaskge = _mm_cmpge_ps(u, zeroes);
		const __m128 umaskle = _mm_cmple_ps(u, ones);
		mask = _mm_and_ps(mask, _mm_and_ps(umaskge, umaskle));
		if (_mm_movemask_ps(mask) == 0x0)
			continue;
		//Compute qvec = cross(tvec, edge0)
		const __m128 qvecx = _mm_sub_ps(
			_mm_mul_ps(tvecy, edge0z),
			_mm_mul_ps(tvecz, edge0y));
		const __m128 qvecy = _mm_sub_ps(
			_mm_mul_ps(tvecz, edge0x),
			_mm_mul_ps(tvecx, edge0z));
		const __m128 qvecz = _mm_sub_ps(
			_mm_mul_ps(tvecx, edge0y),
			_mm_mul_ps(tvecy, edge0x));
		//Compute v = dot(r.direction, qvec) * invDet;
		const __m128 v = _mm_mul_ps(_mm_add_ps(_mm_add_ps(
			_mm_mul_ps(dx, qvecx),
			_mm_mul_ps(dy, qvecy)),
			_mm_mul_ps(dz, qvecz)), invDet);
		//Test v >= 0.0 && v <= 1.0
		const __m128 vmaskge = _mm_cmpge_ps(v, zeroes);
		const __m128 uvmaskle = _mm_cmple_ps(_mm_add_ps(u, v), ones);
		mask = _mm_and_ps(mask, _mm_and_ps(vmaskge, uvmaskle));
		if (_mm_movemask_ps(mask) == 0x0)
			continue;
		//Compute t = dot(edge1, qvec) * invDet;
		const __m128 t = _mm_mul_ps(_mm_add_ps(_mm_add_ps(
			_mm_mul_ps(qvecx, edge1x),
			_mm_mul_ps(qvecy, edge1y)),
			_mm_mul_ps(qvecz, edge1z)), invDet);
		//Test t > EPSILON
		mask = _mm_and_ps(mask, _mm_cmpgt_ps(t, epsilon));
		if (_mm_movemask_ps(mask) == 0x0)
			continue;

		float distance[4];
		float ind[4];		//WHY????
		_mm_storeu_ps(distance, _mm_and_ps(mask, t));
		_mm_storeu_ps(ind, _mm_and_ps(mask, index));
		for (int j = 0; j < 4; j++)
		{
			if (distance[j] > 0.0f && distance[j] < hitDistance)
			{
				hitIndex = ind[j];
				hitDistance = distance[j];
				hit = true;
			}
		}
	}

	if (hit)
		*hitFace = &f[hitIndex];
	return hit;
}
bool RTUtils::intersectRayObjectMollerTrumboreSIMD128(const Ray &r, const Object &o, const Face** hitFace, float &hitDistance)
{
	hitDistance = std::numeric_limits<float>::infinity();
	bool hit = false;
	unsigned int hitIndex;

	//Load Ray Origin
	const __m128 ox = _mm_set_ps1(r.origin.x);
	const __m128 oy = _mm_set_ps1(r.origin.y);
	const __m128 oz = _mm_set_ps1(r.origin.z);
	//Load Ray Direction
	const __m128 dx = _mm_set_ps1(r.direction.x);
	const __m128 dy = _mm_set_ps1(r.direction.y);
	const __m128 dz = _mm_set_ps1(r.direction.z);
	//Constants
	const __m128 epsilon = _mm_set_ps1(EPSILON);
	const __m128 zeroes = _mm_setzero_ps();
	const __m128 ones = _mm_set_ps1(1.0f);

	//Loop over the triangles, computing four triangles at a time
	for(int i = 0; i+3 < o.f.size(); i += 4) 
	{
		const __m128 index = _mm_set_ps(i, i+1, i+2, i+3);	//WHY????
		//Load edge 0 of each triangle
		const __m128 edge0x = _mm_set_ps(o.f[i  ].edgesMollerTrumbore[0].x, 
								   o.f[i+1].edgesMollerTrumbore[0].x, 
								   o.f[i+2].edgesMollerTrumbore[0].x, 
								   o.f[i+3].edgesMollerTrumbore[0].x);
		const __m128 edge0y = _mm_set_ps(o.f[i  ].edgesMollerTrumbore[0].y, 
								   o.f[i+1].edgesMollerTrumbore[0].y, 
								   o.f[i+2].edgesMollerTrumbore[0].y, 
								   o.f[i+3].edgesMollerTrumbore[0].y);
		const __m128 edge0z = _mm_set_ps(o.f[i  ].edgesMollerTrumbore[0].z, 
								   o.f[i+1].edgesMollerTrumbore[0].z, 
								   o.f[i+2].edgesMollerTrumbore[0].z, 
								   o.f[i+3].edgesMollerTrumbore[0].z);
		//Load edge 1 of each triangle
		const __m128 edge1x = _mm_set_ps(o.f[i  ].edgesMollerTrumbore[1].x, 
								   o.f[i+1].edgesMollerTrumbore[1].x, 
								   o.f[i+2].edgesMollerTrumbore[1].x, 
								   o.f[i+3].edgesMollerTrumbore[1].x);
		const __m128 edge1y = _mm_set_ps(o.f[i  ].edgesMollerTrumbore[1].y, 
								   o.f[i+1].edgesMollerTrumbore[1].y, 
								   o.f[i+2].edgesMollerTrumbore[1].y, 
								   o.f[i+3].edgesMollerTrumbore[1].y);
		const __m128 edge1z = _mm_set_ps(o.f[i  ].edgesMollerTrumbore[1].z, 
								   o.f[i+1].edgesMollerTrumbore[1].z, 
								   o.f[i+2].edgesMollerTrumbore[1].z, 
								   o.f[i+3].edgesMollerTrumbore[1].z);
		//Compute pvec = cross(raydir, edge1)
		const __m128 pvecx = _mm_sub_ps(
		_mm_mul_ps(dy, edge1z),
		_mm_mul_ps(dz, edge1y));
		const __m128 pvecy = _mm_sub_ps(
		_mm_mul_ps(dz, edge1x),
		_mm_mul_ps(dx, edge1z));
		const __m128 pvecz = _mm_sub_ps(
		_mm_mul_ps(dx, edge1y),
		_mm_mul_ps(dy, edge1x));
		//Compute det = dot(edge0, pvec)
		const __m128 det = _mm_add_ps(_mm_add_ps(
		_mm_mul_ps(pvecx, edge0x),
		_mm_mul_ps(pvecy, edge0y)),
		_mm_mul_ps(pvecz, edge0z));
		//If determinant is near zero, ray lies in plane of triangle
		const __m128 detmaskge = _mm_cmpge_ps(det, epsilon);				//det >= EPSILON
		//const __m128 minusepsilon = _mm_set_ps1(-EPSILON);
		//const __m128 detmasklt = _mm_cmplt_ps(det, minusepsilon);	//det < -EPSILON
		__m128 mask = detmaskge;									//with backface culling	(if(det > EPSILON))
		if(_mm_movemask_ps(mask) == 0x0)
			continue;
		//__m128 mask = _mm_or_ps(detmasklt, detmaskgt);			//without backface culling (if(det < -EPSILON || det > EPSILON))
		//const __m128 rcp = _mm_rcp_ps(det);								//Reciprocal inveDev = 1.0/dev7
		//const __m128 invDet = _mm_sub_ps(
		//		_mm_add_ps(rcp, rcp),
		//		_mm_mul_ps(_mm_mul_ps(rcp, rcp),
		//		det));
		const __m128 invDet = _mm_div_ps(ones, det);
		//const __m128 invDet = _mm_rcp_ps(det);
		//Load vertice 0 of each triangle
		const __m128 p0x = _mm_set_ps(o.f[i  ].v[0].x, 
								o.f[i+1].v[0].x, 
								o.f[i+2].v[0].x, 
								o.f[i+3].v[0].x);
		const __m128 p0y = _mm_set_ps(o.f[i  ].v[0].y, 
								o.f[i+1].v[0].y, 
								o.f[i+2].v[0].y, 
								o.f[i+3].v[0].y);
		const __m128 p0z = _mm_set_ps(o.f[i  ].v[0].z, 
								o.f[i+1].v[0].z, 
								o.f[i+2].v[0].z, 
								o.f[i+3].v[0].z);
		//Compute tvec = rayorigin - vertice0
		const __m128 tvecx = _mm_sub_ps(ox, p0x);
		const __m128 tvecy = _mm_sub_ps(oy, p0y);
		const __m128 tvecz = _mm_sub_ps(oz, p0z);
		//Compute u = dot(tvec, pvec) * invDet;
		const __m128 u = _mm_mul_ps(_mm_add_ps(_mm_add_ps(
		_mm_mul_ps(pvecx, tvecx),
		_mm_mul_ps(pvecy, tvecy)),
		_mm_mul_ps(pvecz, tvecz)), invDet);
		//Test u >= 0.0 && u <= 1.0
		const __m128 umaskge = _mm_cmpge_ps(u, zeroes);
		const __m128 umaskle = _mm_cmple_ps(u, ones);
		mask = _mm_and_ps(mask, _mm_and_ps(umaskge, umaskle));
		if(_mm_movemask_ps(mask) == 0x0)
			continue;
		//Compute qvec = cross(tvec, edge0)
		const __m128 qvecx = _mm_sub_ps(
		_mm_mul_ps(tvecy, edge0z),
		_mm_mul_ps(tvecz, edge0y));
		const __m128 qvecy = _mm_sub_ps(
		_mm_mul_ps(tvecz, edge0x),
		_mm_mul_ps(tvecx, edge0z));
		const __m128 qvecz = _mm_sub_ps(
		_mm_mul_ps(tvecx, edge0y),
		_mm_mul_ps(tvecy, edge0x));
		//Compute v = dot(r.direction, qvec) * invDet;
		const __m128 v = _mm_mul_ps(_mm_add_ps(_mm_add_ps(
		_mm_mul_ps(dx, qvecx),
		_mm_mul_ps(dy, qvecy)),
		_mm_mul_ps(dz, qvecz)), invDet);
		//Test v >= 0.0 && v <= 1.0
		const __m128 vmaskge = _mm_cmpge_ps(v, zeroes);
		const __m128 uvmaskle = _mm_cmple_ps(_mm_add_ps(u, v), ones);
		mask = _mm_and_ps(mask, _mm_and_ps(vmaskge, uvmaskle));
		if(_mm_movemask_ps(mask) == 0x0)
			continue;
		//Compute t = dot(edge1, qvec) * invDet;
		const __m128 t = _mm_mul_ps(_mm_add_ps(_mm_add_ps(
		_mm_mul_ps(qvecx, edge1x),
		_mm_mul_ps(qvecy, edge1y)),
		_mm_mul_ps(qvecz, edge1z)), invDet);
		//Test t > EPSILON
		mask = _mm_and_ps(mask, _mm_cmpgt_ps(t, epsilon));
		if(_mm_movemask_ps(mask) == 0x0)
			continue;

		float distance[4];
		float ind[4];		//WHY????
		_mm_storeu_ps(distance, _mm_and_ps(mask, t));
		_mm_storeu_ps(ind, _mm_and_ps(mask, index));
		for(int j = 0; j < 4; j++)
		{
			if(distance[j] > 0.0f && distance[j] < hitDistance)
			{
				hitIndex = ind[j];
				hitDistance = distance[j];
				hit = true;
			}
		}
	}

	if(hit)
		*hitFace = &o.f[hitIndex];
	return hit;
}
bool RTUtils::intersectRayObjectMollerTrumboreSIMD256(const Ray &r, const Object &o, const Face** hitFace, float &hitDistance)
{
	hitDistance = std::numeric_limits<float>::infinity();
	bool hit = false;
	unsigned int hitIndex;

	//Load Ray Origin
	
	const __m256 ox = _mm256_set1_ps(r.origin.x);
	const __m256 oy = _mm256_set1_ps(r.origin.y);
	const __m256 oz = _mm256_set1_ps(r.origin.z);
	//Load Ray Direction
	const __m256 dx = _mm256_set1_ps(r.direction.x);
	const __m256 dy = _mm256_set1_ps(r.direction.y);
	const __m256 dz = _mm256_set1_ps(r.direction.z);
	//Constants
	const __m256 epsilon = _mm256_set1_ps(EPSILON);
	const __m256 zeroes = _mm256_setzero_ps();
	const __m256 ones = _mm256_set1_ps(1.0f);

	//Loop over the triangles, computing eight triangles at a time
	for(int i = 0; i+7 < o.f.size(); i += 8) 
	{
		const __m256 index = _mm256_set_ps(i, i+1, i+2, i+3, i+4, i+5, i+6, i+7);	//WHY????
		//Load edge 0 of each triangle
		const __m256 edge0x = _mm256_set_ps(o.f[i  ].edgesMollerTrumbore[0].x, 
								   o.f[i+1].edgesMollerTrumbore[0].x, 
								   o.f[i+2].edgesMollerTrumbore[0].x, 
								   o.f[i+3].edgesMollerTrumbore[0].x, 
								   o.f[i+4].edgesMollerTrumbore[0].x, 
								   o.f[i+5].edgesMollerTrumbore[0].x, 
								   o.f[i+6].edgesMollerTrumbore[0].x, 
								   o.f[i+7].edgesMollerTrumbore[0].x);
		const __m256 edge0y = _mm256_set_ps(o.f[i  ].edgesMollerTrumbore[0].y, 
								   o.f[i+1].edgesMollerTrumbore[0].y, 
								   o.f[i+2].edgesMollerTrumbore[0].y, 
								   o.f[i+3].edgesMollerTrumbore[0].y, 
								   o.f[i+4].edgesMollerTrumbore[0].y, 
								   o.f[i+5].edgesMollerTrumbore[0].y, 
								   o.f[i+6].edgesMollerTrumbore[0].y, 
								   o.f[i+7].edgesMollerTrumbore[0].y);
		const __m256 edge0z = _mm256_set_ps(o.f[i  ].edgesMollerTrumbore[0].z, 
								   o.f[i+1].edgesMollerTrumbore[0].z, 
								   o.f[i+2].edgesMollerTrumbore[0].z, 
								   o.f[i+3].edgesMollerTrumbore[0].z, 
								   o.f[i+4].edgesMollerTrumbore[0].z, 
								   o.f[i+5].edgesMollerTrumbore[0].z, 
								   o.f[i+6].edgesMollerTrumbore[0].z, 
								   o.f[i+7].edgesMollerTrumbore[0].z);
		//Load edge 1 of each triangle
		const __m256 edge1x = _mm256_set_ps(o.f[i  ].edgesMollerTrumbore[1].x, 
								   o.f[i+1].edgesMollerTrumbore[1].x, 
								   o.f[i+2].edgesMollerTrumbore[1].x, 
								   o.f[i+3].edgesMollerTrumbore[1].x, 
								   o.f[i+4].edgesMollerTrumbore[1].x, 
								   o.f[i+5].edgesMollerTrumbore[1].x, 
								   o.f[i+6].edgesMollerTrumbore[1].x, 
								   o.f[i+7].edgesMollerTrumbore[1].x);
		const __m256 edge1y = _mm256_set_ps(o.f[i  ].edgesMollerTrumbore[1].y, 
								   o.f[i+1].edgesMollerTrumbore[1].y, 
								   o.f[i+2].edgesMollerTrumbore[1].y, 
								   o.f[i+3].edgesMollerTrumbore[1].y, 
								   o.f[i+4].edgesMollerTrumbore[1].y, 
								   o.f[i+5].edgesMollerTrumbore[1].y, 
								   o.f[i+6].edgesMollerTrumbore[1].y, 
								   o.f[i+7].edgesMollerTrumbore[1].y);
		const __m256 edge1z = _mm256_set_ps(o.f[i  ].edgesMollerTrumbore[1].z, 
								   o.f[i+1].edgesMollerTrumbore[1].z, 
								   o.f[i+2].edgesMollerTrumbore[1].z, 
								   o.f[i+3].edgesMollerTrumbore[1].z, 
								   o.f[i+4].edgesMollerTrumbore[1].z, 
								   o.f[i+5].edgesMollerTrumbore[1].z, 
								   o.f[i+6].edgesMollerTrumbore[1].z, 
								   o.f[i+7].edgesMollerTrumbore[1].z);
		//Compute pvec = cross(raydir, edge1)
		const __m256 pvecx = _mm256_sub_ps(
		_mm256_mul_ps(dy, edge1z),
		_mm256_mul_ps(dz, edge1y));
		const __m256 pvecy = _mm256_sub_ps(
		_mm256_mul_ps(dz, edge1x),
		_mm256_mul_ps(dx, edge1z));
		const __m256 pvecz = _mm256_sub_ps(
		_mm256_mul_ps(dx, edge1y),
		_mm256_mul_ps(dy, edge1x));
		//Compute det = dot(edge0, pvec)
		const __m256 det = _mm256_add_ps(_mm256_add_ps(
		_mm256_mul_ps(pvecx, edge0x),
		_mm256_mul_ps(pvecy, edge0y)),
		_mm256_mul_ps(pvecz, edge0z));
		//If determinant is near zero, ray lies in plane of triangle
		const __m256 detmaskge = _mm256_cmp_ps(det, epsilon, _CMP_GE_OS);				//det >= EPSILON
		
		//const __m128 minusepsilon = _mm_set_ps1(-EPSILON);
		//const __m128 detmasklt = _mm_cmplt_ps(det, minusepsilon);	//det < -EPSILON
		__m256 mask = detmaskge;									//with backface culling	(if(det > EPSILON))
		if(_mm256_movemask_ps(mask) == 0x0)
			continue;
		//__m128 mask = _mm_or_ps(detmasklt, detmaskgt);			//without backface culling (if(det < -EPSILON || det > EPSILON))
		//const __m128 rcp = _mm_rcp_ps(det);								//Reciprocal inveDev = 1.0/dev7
		//const __m128 invDet = _mm_sub_ps(
		//		_mm_add_ps(rcp, rcp),
		//		_mm_mul_ps(_mm_mul_ps(rcp, rcp),
		//		det));
		const __m256 invDet = _mm256_div_ps(ones, det);
		//const __m128 invDet = _mm_rcp_ps(det);
		//Load vertice 0 of each triangle
		const __m256 p0x = _mm256_set_ps(o.f[i  ].v[0].x, 
								o.f[i+1].v[0].x, 
								o.f[i+2].v[0].x, 
								o.f[i+3].v[0].x, 
								o.f[i+4].v[0].x, 
								o.f[i+5].v[0].x, 
								o.f[i+6].v[0].x, 
								o.f[i+7].v[0].x);
		const __m256 p0y = _mm256_set_ps(o.f[i  ].v[0].y, 
								o.f[i+1].v[0].y, 
								o.f[i+2].v[0].y, 
								o.f[i+3].v[0].y, 
								o.f[i+4].v[0].y, 
								o.f[i+5].v[0].y, 
								o.f[i+6].v[0].y, 
								o.f[i+7].v[0].y);
		const __m256 p0z = _mm256_set_ps(o.f[i  ].v[0].z, 
								o.f[i+1].v[0].z, 
								o.f[i+2].v[0].z, 
								o.f[i+3].v[0].z, 
								o.f[i+4].v[0].z, 
								o.f[i+5].v[0].z, 
								o.f[i+6].v[0].z, 
								o.f[i+7].v[0].z);
		//Compute tvec = rayorigin - vertice0
		const __m256 tvecx = _mm256_sub_ps(ox, p0x);
		const __m256 tvecy = _mm256_sub_ps(oy, p0y);
		const __m256 tvecz = _mm256_sub_ps(oz, p0z);
		//Compute u = dot(tvec, pvec) * invDet;
		const __m256 u = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(
		_mm256_mul_ps(pvecx, tvecx),
		_mm256_mul_ps(pvecy, tvecy)),
		_mm256_mul_ps(pvecz, tvecz)), invDet);
		//Test u >= 0.0 && u <= 1.0
		const __m256 umaskge = _mm256_cmp_ps(u, zeroes, _CMP_GE_OS);
		const __m256 umaskle = _mm256_cmp_ps(u, ones, _CMP_LE_OS);
		mask = _mm256_and_ps(mask, _mm256_and_ps(umaskge, umaskle));
		if(_mm256_movemask_ps(mask) == 0x0)
			continue;
		//Compute qvec = cross(tvec, edge0)
		const __m256 qvecx = _mm256_sub_ps(
		_mm256_mul_ps(tvecy, edge0z),
		_mm256_mul_ps(tvecz, edge0y));
		const __m256 qvecy = _mm256_sub_ps(
		_mm256_mul_ps(tvecz, edge0x),
		_mm256_mul_ps(tvecx, edge0z));
		const __m256 qvecz = _mm256_sub_ps(
		_mm256_mul_ps(tvecx, edge0y),
		_mm256_mul_ps(tvecy, edge0x));
		//Compute v = dot(r.direction, qvec) * invDet;
		const __m256 v = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(
		_mm256_mul_ps(dx, qvecx),
		_mm256_mul_ps(dy, qvecy)),
		_mm256_mul_ps(dz, qvecz)), invDet);
		//Test v >= 0.0 && v <= 1.0
		const __m256 vmaskge = _mm256_cmp_ps(v, zeroes, _CMP_GE_OS);
		const __m256 uvmaskle = _mm256_cmp_ps(_mm256_add_ps(u, v), ones, _CMP_LE_OS);
		mask = _mm256_and_ps(mask, _mm256_and_ps(vmaskge, uvmaskle));
		if(_mm256_movemask_ps(mask) == 0x0)
			continue;
		//Compute t = dot(edge1, qvec) * invDet;
		const __m256 t = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(
		_mm256_mul_ps(qvecx, edge1x),
		_mm256_mul_ps(qvecy, edge1y)),
		_mm256_mul_ps(qvecz, edge1z)), invDet);
		//Test t > EPSILON
		mask = _mm256_and_ps(mask, _mm256_cmp_ps(t, epsilon, _CMP_GT_OS));
		if(_mm256_movemask_ps(mask) == 0x0)
			continue;

		float distance[8];
		float ind[8];		//WHY????
		_mm256_storeu_ps(distance, _mm256_and_ps(mask, t));
		_mm256_storeu_ps(ind, _mm256_and_ps(mask, index));
		for(int j = 0; j < 8; j++)
		{
			if(distance[j] > 0.0f && distance[j] < hitDistance)
			{
				hitIndex = ind[j];
				hitDistance = distance[j];
				hit = true;
			}
		}
	}

	if(hit)
		*hitFace = &o.f[hitIndex];
	return hit;
}

/*
bool RTUtils::intersectRayTrianglePlucker(const Ray &r, const Face2 &f, float &t)
{
	//Plucker coordinates
	//
	//Given a ray R with direction D and origin O, its plucker coordinates are:
	//	Plucker(R) = {D : D × O} = {U : V}
	//
	//Given two rays R and S, the permuted inner product is:
	//	InnerProduct(Plucker(R), Plucker(S)) = Ur·Vs + Us·Vr
	//
	//The inner product indicates their relative orientation as follows:
	//	InnerProduct(Plucker(R), Plucker(S)) > 0, S goes counterclockwise around R
	//	InnerProduct(Plucker(R), Plucker(S)) < 0, S goes clockwise around R
	//	InnerProduct(Plucker(R), Plucker(S)) = 0, S intersects or is parallel to R
	//
	
	glm::vec3 pluckerRay[2];
	pluckerRay[0] = r.direction;
	pluckerRay[1] = glm::cross(r.direction, r.origin);

	glm::vec3 pluckerEdge0[2];
	pluckerEdge0[0] = f.edges[0];
	pluckerEdge0[1] = glm::cross(f.edges[0], f.v[0]);
	glm::vec3 pluckerEdge1[2];
	pluckerEdge0[0] = f.edges[1];
	pluckerEdge0[1] = glm::cross(f.edges[1], f.v[1]);
	glm::vec3 pluckerEdge2[2];
	pluckerEdge0[0] = f.edges[0];
	pluckerEdge0[1] = glm::cross(f.edges[2], f.v[2]);


	bool side0, side1, side2;	
	side0= glm::dot(pluckerRay[0], pluckerEdge0[1]) + glm::dot(pluckerEdge0[0], pluckerRay[1]) >= 0.0f;
	side1= glm::dot(pluckerRay[0], pluckerEdge1[1]) + glm::dot(pluckerEdge1[0], pluckerRay[1]) >= 0.0f;
	if (side0 != side1)
		return false;
	side2= glm::dot(pluckerRay[0], pluckerEdge2[1]) + glm::dot(pluckerEdge2[0], pluckerRay[1]) >= 0.0f;
	if (side0 != side2)
		return false;

	//RayIntersectTrianglePlane(ray, triangle->plane, &t);
	//calcula o t da equação o raio R(t) = P + t*dir
	t = (f.planeCoeficient - glm::dot(f.tn, r.origin))/glm::dot(r.direction, f.tn);
	//substitui t na esquação
	//glm::vec3 point;
	//point = r.origin + t*r.direction;
	
	if(t <= EPSILON)
		return false;

	return true;
}
bool RTUtils::intersectRayTriangleSimple(const Ray &r, const Face2 &f, glm::vec3 &hitPoint, float &t)
{
	glm::vec3 triangleV[3];
	glm::vec3 verticesN[3];

	//carrega os três pontos do triangulo
	for(int j = 0; j < 3; j++)
	{
		triangleV[j] = f.v[j];
		verticesN[j] = f.n[j];
	}

	//carrega normal do plano
	glm::vec3 planeN = f.tn;

	//carrega coeficiente do plano
	float d = f.planeCoeficient;

	//raio não pode ser paralelo ao plano
	float angle = glm::dot(r.direction, planeN);
	if(angle != 0)
	{
		//calcula o t da equação o raio R(t) = P + t*dir
		t = (d - glm::dot(planeN, r.origin))/angle;
		//substitui t na esquação
		glm::vec3 point;
		point = r.origin + t*r.direction;
		//testa se o ponto está dentro do triangulo
		if( glm::dot(glm::cross((triangleV[1] - triangleV[0]), (point - triangleV[0])), planeN) >= 0 &&
			glm::dot(glm::cross((triangleV[2] - triangleV[1]), (point - triangleV[1])), planeN) >= 0 &&
			glm::dot(glm::cross((triangleV[0] - triangleV[2]), (point - triangleV[2])), planeN) >= 0)
		{
			hitPoint = point;
			return true;
		}
	}
	return false;
}

bool RTUtils::intersectKenslerSIMD(const Ray &r, const Object &object, const Face2** hitTriangle, float &hitDistance)
{
	hitDistance = std::numeric_limits<float>::infinity();
	bool hit = false;

	//Load Ray Origin
	const __m128 ox = _mm_set_ps1(r.origin.x);
	const __m128 oy = _mm_set_ps1(r.origin.y);
	const __m128 oz = _mm_set_ps1(r.origin.z);
	//Load Ray Direction
	const __m128 dx = _mm_set_ps1(r.direction.x);
	const __m128 dy = _mm_set_ps1(r.direction.y);
	const __m128 dz = _mm_set_ps1(r.direction.z);

	//Loop over the triangles, computing four triangles at a time
	for(int i = 0; i+3 < object.f2.size(); i += 4) 
	{
		//Min t
		__m128 oldt = _mm_set_ps1(hitDistance);
		//Load vertice 1 of each triangle
		const __m128 p1x = _mm_set_ps(object.f2[i  ].v[1].x, 
										object.f2[i+1].v[1].x, 
										object.f2[i+2].v[1].x, 
										object.f2[i+3].v[1].x);
		const __m128 p1y = _mm_set_ps(object.f2[i  ].v[1].y, 
										object.f2[i+1].v[1].y, 
										object.f2[i+2].v[1].y, 
										object.f2[i+3].v[1].y);
		const __m128 p1z = _mm_set_ps(object.f2[i  ].v[1].z, 
										object.f2[i+1].v[1].z, 
										object.f2[i+2].v[1].z, 
										object.f2[i+3].v[1].z);
		//Load vertice 0 of each triangle
		const __m128 p0x = _mm_set_ps(object.f2[i  ].v[0].x, 
										object.f2[i+1].v[0].x, 
										object.f2[i+2].v[0].x, 
										object.f2[i+3].v[0].x);
		const __m128 p0y = _mm_set_ps(object.f2[i  ].v[0].y, 
										object.f2[i+1].v[0].y, 
										object.f2[i+2].v[0].y, 
										object.f2[i+3].v[0].y);
		const __m128 p0z = _mm_set_ps(object.f2[i  ].v[0].z, 
										object.f2[i+1].v[0].z, 
										object.f2[i+2].v[0].z, 
										object.f2[i+3].v[0].z);
		//Compute edge 0
		const __m128 edge0x = _mm_sub_ps(p1x, p0x);
		const __m128 edge0y = _mm_sub_ps(p1y, p0y);
		const __m128 edge0z = _mm_sub_ps(p1z, p0z);
		//Load vertice 2 of each triangle
		const __m128 p2x = _mm_set_ps(object.f2[i].v[2].x, 
										object.f2[i+1].v[2].x, 
										object.f2[i+2].v[2].x, 
										object.f2[i+3].v[2].x);
		const __m128 p2y = _mm_set_ps(object.f2[i  ].v[2].y, 
										object.f2[i+1].v[2].y, 
										object.f2[i+2].v[2].y, 
										object.f2[i+3].v[2].y);
		const __m128 p2z = _mm_set_ps(object.f2[i  ].v[2].z, 
										object.f2[i+1].v[2].z, 
										object.f2[i+2].v[2].z, 
										object.f2[i+3].v[2].z);
		//Compute edge 1
		const __m128 edge1x = _mm_sub_ps(p0x, p2x);
		const __m128 edge1y = _mm_sub_ps(p0y, p2y);
		const __m128 edge1z = _mm_sub_ps(p0z, p2z);
		//Compute Normal
		const __m128 normalx = _mm_sub_ps(
		_mm_mul_ps(edge0y, edge1z),
		_mm_mul_ps(edge0z, edge1y));
		const __m128 normaly = _mm_sub_ps(
		_mm_mul_ps(edge0z, edge1x),
		_mm_mul_ps(edge0x, edge1z));
		const __m128 normalz = _mm_sub_ps(
		_mm_mul_ps(edge0x, edge1y),
		_mm_mul_ps(edge0y, edge1x));
		const __m128 zeroes = _mm_setzero_ps();


		//Compute volume V, the denominator
		const __m128 v = _mm_add_ps(
									_mm_add_ps(
												_mm_mul_ps(normalx, dx),
												_mm_mul_ps(normaly, dy)),
									_mm_mul_ps(normalz, dz));
		//Reciprocal estimate of V with one round of Newton
		const __m128 rcpi = _mm_rcp_ps(v);
		const __m128 rcp = _mm_sub_ps(
										_mm_add_ps(rcpi, rcpi),
										_mm_mul_ps(
													_mm_mul_ps(rcpi, rcpi),
													v));
		//Edge from ray origin to first triangle vertex
		const __m128 edge2x = _mm_sub_ps(p0x, ox);
		const __m128 edge2y = _mm_sub_ps(p0y, oy);
		const __m128 edge2z = _mm_sub_ps(p0z, oz);
		//Compute volume Va
		const __m128 va = _mm_add_ps(_mm_add_ps(
		_mm_mul_ps(normalx, edge2x),
		_mm_mul_ps(normaly, edge2y)),
		_mm_mul_ps(normalz, edge2z));
		//Find Va/V to get t-value
		const __m128 t = _mm_mul_ps(rcp, va);
		const __m128 tmaskb = _mm_cmplt_ps(t, oldt);
		const __m128 tmaska = _mm_cmpgt_ps(t, zeroes);
		__m128 mask = _mm_and_ps(tmaska, tmaskb);
		if(_mm_movemask_ps(mask) == 0x0) 
			continue;
		//Compute the single intermediate cross product
		const __m128 intermx = _mm_sub_ps(
		_mm_mul_ps(edge2y, dz),
		_mm_mul_ps(edge2z, dy));
		const __m128 intermy = _mm_sub_ps(
		_mm_mul_ps(edge2z, dx),
		_mm_mul_ps(edge2x, dz));
		const __m128 intermz = _mm_sub_ps(
		_mm_mul_ps(edge2x, dy),
		_mm_mul_ps(edge2y, dx));
		//Compute volume V1
		const __m128 v1 = _mm_add_ps(_mm_add_ps(
		_mm_mul_ps(intermx, edge1x),
		_mm_mul_ps(intermy, edge1y)),
		_mm_mul_ps(intermz, edge1z));
		//Find V1/V to get barycentric beta
		const __m128 beta = _mm_mul_ps(rcp, v1);
		const __m128 bmask = _mm_cmpge_ps(beta, zeroes);
		mask = _mm_and_ps(mask, bmask);
		if(_mm_movemask_ps(mask) == 0x0) 
			continue;
		//Compute volume V2
		const __m128 v2 = _mm_add_ps(_mm_add_ps(
		_mm_mul_ps(intermx, edge0x),
		_mm_mul_ps(intermy, edge0y)),
		_mm_mul_ps(intermz, edge0z));
		//Test if alpha > 0
		const __m128 v1plusv2 = _mm_add_ps(v1, v2);
		const __m128 v12mask = _mm_cmple_ps(
		_mm_mul_ps(v1plusv2, v),
		_mm_mul_ps(v, v));
		//Find V2/V to get barycentric gamma
		const __m128 gamma = _mm_mul_ps(rcp, v2);
		const __m128 gmask = _mm_cmpge_ps(gamma, zeroes);
		mask = _mm_and_ps(mask, v12mask);
		mask = _mm_and_ps(mask, gmask);
		if (_mm_movemask_ps(mask) == 0x0)
			continue;
		//Update stored t-value for closest hits
		//_mm_store_ps(&rtf[pi][ri],
		//				_mm_or_ps(_mm_and_ps(mask, t),
		//				_mm_andnot_ps(mask, oldt)));
		//Optionally store barycentric beta and gamma too
		//float distance;
		//_mm_store_ss(&distance, t);	//Store the smaller t(distance)
		//if(distance < hitDistance)
		//{
		//	hitDistance = distance;
		//	hit = true;
		//}
		float distance[4];
		float finalmask[4];
		_mm_store_ps(distance, t);
		_mm_store_ps(finalmask, mask);
		for(int j = 0; j < 4; j++)
		{
			//std::cout<<finalmask[j]<<std::endl;
			if(finalmask[j] != 0.0f)
			{
				if(distance[j] < hitDistance)
				{
					*hitTriangle = &object.f2[i + j];
					hitDistance = distance[j];
					hit = true;
				}
			}
		}
	}
	return hit;
}*/