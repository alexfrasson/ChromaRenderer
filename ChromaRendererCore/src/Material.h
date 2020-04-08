#pragma once

#include <Color.h>

#include <string>

struct Material
{
	Material() :
		name("Default"),
		kd(0.85f, 0.85f, 0.85f),
		ke(0.0f, 0.0f, 0.0f),
		transparent(1.0f, 1.0f, 1.0f)
	{}

	std::string name;
	Color kd;
	Color ke;
	Color transparent;
};