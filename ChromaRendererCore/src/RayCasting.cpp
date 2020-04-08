#include <RayCasting.h>



RayCasting::RayCasting()
	: donePixelCount(0), pixelCount(0)
{

}
void RayCasting::trace(Scene &scene, Image &img, RendererSettings &settings, Interval interval, bool& abort)
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

			Color diffuseColor(0.0f, 0.0f, 0.0f);
			//Para todos os raios do pixel
			for (int k = 0; k < rays.size(); k++)
			{
				Intersection intersection = Intersection();
				intersection.n = glm::vec3();
				intersection.p = glm::vec3();

				float nNodeHitsNormalized;
				if (scene.sps->intersect(rays[k], intersection))
					diffuseColor += calcColor(intersection);
				else
					diffuseColor += 0.1f;	//Background
			}
			diffuseColor /= static_cast<float>(rays.size());	//Color average


			img.setColor(i, j, diffuseColor);
		}

		donePixelCount += interval.toHeight - interval.fromHeight;
	}
}
Color RayCasting::calcColor(Intersection& is)
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

	return is.material->kd * diffuseColorIntensity;
}
float RayCasting::getProgress()
{
	return (donePixelCount * invPixelCount);
}
bool RayCasting::isRunning()
{
	if (running)
		running = donePixelCount < pixelCount;
	return running;
}