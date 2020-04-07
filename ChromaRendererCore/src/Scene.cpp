#include "Scene.h"
//#include "RTUtils.h"
#include "ModelImporter.h"


Scene::Scene(void)
	: sps(NULL)
{
}
Scene::~Scene(void)
{
	clear();
}

long Scene::triangleCount()
{
	long tc = 0;
	for(unsigned int i = 0; i < objects.size(); i++)
		tc += objects[i].f.size();
	return tc;
}
void Scene::LoadMesh(std::string file)
{
	/*Object o;
	ModelImporter::import(file, o);
	o.genBoundingBox();

	objects.push_back( o );*/
}

void Scene::addObject(Object o)
{
	/*ready = false;
	objects.push_back( o );
	sps->build(objects);
	camera.fit(getBoundingBox());
	ready = true;*/
}
void Scene::addObject(Object o, std::function<void(void)> cb)
{
	/*ready = false;
	objects.push_back(o);
	sps->build(objects);
	camera.fit(getBoundingBox());
	ready = true;

	cb();*/
}

void Scene::addMesh(Mesh *m)
{
	ready = false;
	clear();
	meshes.push_back(m);
	sps = new BVH();
	sps->build(meshes);
	//camera.fit(getBoundingBox());
	// Sponza
	/*camera.eye.x += 100;
	camera.eye.z -= 250;

	camera.eye = getBoundingBox().getCenter();
	camera.eye.z -= 100;

	camera.computeUVW();*/
	// Sponza
	ready = true;
}
void Scene::addMesh(Mesh *m, std::function<void(void)> cb)
{
	addMesh(m);
	cb();
}

void Scene::clear()
{
	objects.clear();
	for (int i = 0; i < meshes.size(); i++)
		delete meshes[i];
	meshes.clear();
	materials.clear();
	delete sps;
}


BoundingBox Scene::getBoundingBox()
{
	BoundingBox bb;

	for (int i = 0; i < objects.size(); i++)
	{
		if (objects[i].boundingBox.max.x > bb.max.x)
			bb.max.x = objects[i].boundingBox.max.x;
		if (objects[i].boundingBox.max.z > bb.max.z)
			bb.max.z = objects[i].boundingBox.max.z;
		if (objects[i].boundingBox.max.y > bb.max.y)
			bb.max.y = objects[i].boundingBox.max.y;
		
		if (objects[i].boundingBox.min.x < bb.min.x)
			bb.min.x = objects[i].boundingBox.min.x;
		if (objects[i].boundingBox.min.z < bb.min.z)
			bb.min.z = objects[i].boundingBox.min.z;
		if (objects[i].boundingBox.min.y < bb.min.y)
			bb.min.y = objects[i].boundingBox.min.y;
	}


	for (int i = 0; i < meshes.size(); i++)
	{
		if (meshes[i]->boundingBox.max.x > bb.max.x)
			bb.max.x = meshes[i]->boundingBox.max.x;
		if (meshes[i]->boundingBox.max.z > bb.max.z)
			bb.max.z = meshes[i]->boundingBox.max.z;
		if (meshes[i]->boundingBox.max.y > bb.max.y)
			bb.max.y = meshes[i]->boundingBox.max.y;

		if (meshes[i]->boundingBox.min.x < bb.min.x)
			bb.min.x = meshes[i]->boundingBox.min.x;
		if (meshes[i]->boundingBox.min.z < bb.min.z)
			bb.min.z = meshes[i]->boundingBox.min.z;
		if (meshes[i]->boundingBox.min.y < bb.min.y)
			bb.min.y = meshes[i]->boundingBox.min.y;
	}

	return bb;
}