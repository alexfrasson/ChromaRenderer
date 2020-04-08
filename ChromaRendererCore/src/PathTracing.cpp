#include <PathTracing.h>
#include <halton.hpp>
#include <random>


unsigned my_rand(void)
{
	static unsigned next1 = 1151752134u, next2 = 2070363486u;
	next1 = next1 * 1701532575u + 571550083u;
	next2 = next2 * 3145804233u + 4178903934u;
	return (next1 << 16) ^ next2;
}
float U_m1_p1()
{
	return float(my_rand())*(1.0f / 2147483648.0f) - 1.0f;
}
glm::vec3 pick_random_point_in_sphere()
{
	/**/float x1, x2, x3, d2;
	do{
		x1 = U_m1_p1();
		x2 = U_m1_p1();
		x3 = U_m1_p1();
		d2 = x1*x1 + x2*x2 + x3*x3;
	} while (d2 > 1.0f);
	float scale = 1.0f / sqrt(d2); // or use a fast InvSqrt for this
	return glm::vec3(x1*scale, x2*scale, x3*scale);/**/

	/*double v[2];
	halton(v);
	double x[3];
	u2_to_sphere_unit_3d(v, x);
	return glm::normalize(glm::vec3(x[0], x[1], x[2]));/**/
}
glm::vec3 pick_random_point_in_semisphere(glm::vec3 const &v)
{
	glm::vec3 result = pick_random_point_in_sphere();
	if (glm::dot(result, v) < 0)
	{
		result.x = -result.x;
		result.y = -result.y;
		result.z = -result.z;
	}
	return result;
}


PathTracing::PathTracing()
	: donePixelCount(0), pixelCount(0), maxDepth(5), enviromentLight(true), targetSamplesPerPixel(100)
{
	halton_dim_num_set(2);
}
void PathTracing::trace(Scene &scene, Image &img, Interval interval, bool& abort)
{
	/*std::vector<Ray> rays;
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

			Color multiSampledColor(0.0f);
			//Para todos os raios do pixel
			for (int k = 0; k < rays.size(); k++)
			{
				Color sampledColor(0.0f);
				for (uint32_t lol = 0; lol < maxSamples; lol++)
					sampledColor += tracePath(rays[k], scene, 0);
				multiSampledColor += sampledColor / maxSamples;
			}
			multiSampledColor /= (int)rays.size();

			img.setColor(i, j, multiSampledColor);
		}
		donePixelCount += interval.toHeight - interval.fromHeight;
	}*/

	for (int i = interval.fromWidth; i < interval.toWidth; i++)
	{
		if (abort)
			return;

		for (int j = interval.fromHeight; j < interval.toHeight; j++)
		{	
			Color sampledColor = Color::BLACK;

			for (uint32_t lol = 0; lol < targetSamplesPerPixel; lol++)
			{
				Ray ray;
				scene.camera.randomRayDirection(i, j, ray);
				sampledColor += tracePath(ray, scene, 0);
			}

			sampledColor /= static_cast<float>(targetSamplesPerPixel);

			img.setColor(i, j, sampledColor);
		}
		donePixelCount += interval.toHeight - interval.fromHeight;
	}
}
void PathTracing::setSettings(RendererSettings& settings)
{
	//enviromentLight = settings.enviromentLight;
	targetSamplesPerPixel = settings.samplesperpixel;
}
Color PathTracing::tracePath(Ray &r, Scene &scene, uint32_t depth)
{
	/*
	if (depth == maxDepth)
		return 0.0f;  // Bounced enough times.

	//if (r.hitSomething == false)
	Intersection is;
	if (!scene.sps->intersect(r, is))
		return 0.0f;  // Nothing was hit.

	//Material m = r.thingHit->material;
	//Color emittance = m.emittance;
	float emittance;
	if (is.material->light)
		emittance = 0.93f;
	else
		emittance = 0.0f;

	// Pick a random direction from here and keep going.
	Ray newRay;
	newRay.origin = is.p;
	newRay.direction = pick_random_point_in_semisphere(is.n);  // This is NOT a cosine-weighted distribution!

	// Compute the BRDF for this ray (assuming Lambertian reflection)
	float cos_theta = glm::dot(newRay.direction, is.n);
	//Color BRDF = 2 * m.reflectance * cos_theta;
	//Color reflected = TracePath(newRay, depth + 1);
	float BRDF = 2 * 1.0f * cos_theta;
	float reflected = tracePath(newRay, scene, depth + 1);

	// Apply the Rendering Equation here.
	return emittance + (BRDF * reflected);
	/**/

	if (depth == maxDepth)
		return Color::BLACK;  // Bounced enough times.

	//if (r.hitSomething == false)
	Intersection is;
	if (!scene.sps->intersect(r, is)) // Nothing was hit.
	{
		if (enviromentLight)
			return Color(0.85f);  
		else
			return Color::BLACK;
	}
	//Material m = r.thingHit->material;
	//Color emittance = m.emittance;
	Color emittance;
	emittance = is.material->ke * is.material->kd;

	// Pick a random direction from here and keep going.
	Ray newRay;
	newRay.origin = is.p + is.n * 0.0001f;
	newRay.direction = pick_random_point_in_semisphere(is.n);  // This is NOT a cosine-weighted distribution!

	// Compute the BRDF for this ray (assuming Lambertian reflection)
	float cos_theta = glm::dot(newRay.direction, is.n);
	//Color BRDF = 2 * m.reflectance * cos_theta;
	//Color reflected = TracePath(newRay, depth + 1);
	Color BRDF = is.material->kd * 2.0f * cos_theta;
	Color reflected = tracePath(newRay, scene, depth + 1);

	// Apply the Rendering Equation here.
	return emittance + (reflected * BRDF);
}
float PathTracing::calcColor(Intersection& is)
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
float PathTracing::getProgress()
{
	return (donePixelCount * invPixelCount);
}
bool PathTracing::isRunning()
{
	if (running)
		running = donePixelCount < pixelCount;
	return running;
}