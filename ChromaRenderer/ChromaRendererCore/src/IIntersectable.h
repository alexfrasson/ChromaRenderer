#pragma once

#include <Ray.h>
#include <Intersection.h>



class IIntersectable
{
public:
	virtual bool intersect(Ray &r, Intersection &intersection) const = 0;
	virtual ~IIntersectable() {};
};


