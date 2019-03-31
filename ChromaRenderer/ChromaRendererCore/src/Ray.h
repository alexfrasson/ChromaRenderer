#ifndef __RAY__
#define __RAY__

#include <glm/vec3.hpp>

class Ray 
{
	public:
		glm::vec3 origin;
		glm::vec3 direction;
		
		float mint;
		float maxt;

		Ray(void);			
		Ray(const glm::vec3& origin, const glm::vec3& dir);	
		Ray(const Ray& ray); 		
		
		Ray& operator= (const Ray& rhs);
		 								
		~Ray(void);
};

#endif

