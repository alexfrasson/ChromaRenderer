#pragma once

#include <vector>
#include <BoundingBox.h>
#include "Ray.h"
#include <glm/glm.hpp>





//tan(FOV/2) = (screenSize/2) / screenPlaneDistance
//tan(FOV_H/2) = (screen_width/2) / screenPlaneDistance
//tan(FOV_V / 2) = (screen_height / 2) / screenPlaneDistance
//tan(FOV_H/2) / screen_width = tan(FOV_V/2) / screen_height

class Camera
{

public:
	//Image image;
	int width, height;
	glm::vec3 eye;
	glm::vec3 up;
	glm::vec3 right, forward;
	float d;
	float aspectRatio;

private:

	float m_HorizontalFOV;
	
public:

	Camera(void);
	~Camera(void);

	float horizontalFOV()
	{
		return m_HorizontalFOV;
	}
	void horizontalFOV(float hfov)
	{
		m_HorizontalFOV = hfov;
		d = ((float)width / 2.0f) / tan(m_HorizontalFOV / 2.0f);
	}

	void setSize(int w, int h);
	void lookAt(glm::vec3 target);

	void randomRayDirection(const int i, const int j, Ray &ray) const;
	void rayDirection(const int i, const int j, Ray &ray) const;
	void rayDirection(const int i, const int j, std::vector<Ray> &rays) const;
	void rayDirection(const int i, const int j, std::vector<Ray> &rays, const unsigned int nRays) const;

	void fit(const BoundingBox& bb);
	float fov();

};

