#pragma once

#include <Object.h>
#include <Mesh.h>
#include <Scene.h>
#include <functional>

namespace ModelImporter
{
	bool import(std::string fileName, Mesh& o);
	void importcbm(std::string fileName, std::function<void(Mesh*)> cb);

	bool import(std::string fileName, Scene& s);
	void importcbscene(std::string fileName, Scene &s, std::function<void()> cb);
}