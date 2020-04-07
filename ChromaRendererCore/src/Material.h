#pragma once

#include <Color.h>

struct Material
{
	Material() :
		name("Default"),
		kd(0.85, 0.85, 0.85),
		ke(0, 0, 0),
		transparent(1, 1, 1)
	{}

	std::string name;
	Color kd;
	Color ke;
	Color transparent;
};