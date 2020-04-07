#include "Ray.h"

#include <limits>

#define EPSILON 0.000001f

Ray::Ray (void)
	: 	origin(0.0), 
		direction(0.0, 0.0, 1.0),
		//mint(-std::numeric_limits<float>::infinity()),
		mint(0),
		maxt(std::numeric_limits<float>::infinity())
{
}

Ray::Ray (const glm::vec3& origin, const glm::vec3& dir)
	: 	origin(origin), 
		direction(dir),
		//mint(-std::numeric_limits<float>::infinity()),
		mint(0),
		maxt(std::numeric_limits<float>::infinity())
{
}

Ray::Ray (const Ray& ray)
	:	origin(ray.origin), 
		direction(ray.direction),
		mint(ray.mint),
		maxt(ray.maxt)
{
}

Ray& Ray::operator= (const Ray& rhs) 
{
	if (this == &rhs)
		return (*this);
		
	origin = rhs.origin; 
	direction = rhs.direction; 
	mint = rhs.mint;
	maxt = rhs.maxt;

	return (*this);	
}

Ray::~Ray (void) 
{
}




