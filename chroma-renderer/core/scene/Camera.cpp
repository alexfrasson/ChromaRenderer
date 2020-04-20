#include "chroma-renderer/core/scene/Camera.h"

#include <cmath>
#include <halton.hpp>
#include <iostream>
#include <random>

Camera::Camera(void)
    : width(1017), height(720), eye(-3, 2, 10), up(0, 1, 0), right(1, 0, 0), forward(0, 0, 1), m_HorizontalFOV(1.0f),
      aspectRatio(width / static_cast<float>(height)), d(((float)width / 2.0f) / tan(m_HorizontalFOV / 2.0f))
{
    // computeUVW();

    // halton_dim_num_set(2);
}
Camera::~Camera(void)
{
}

void Camera::setSize(int pwidth, int pheight)
{
    aspectRatio = pwidth / static_cast<float>(pheight);
    width = pwidth;
    height = pheight;
    d = ((float)pwidth / 2.0f) / tan(m_HorizontalFOV / 2.0f);
}
void Camera::lookAt(glm::vec3 target)
{
    forward = target - eye;
    forward = glm::normalize(forward);
    right = glm::cross(glm::vec3(0, 1, 0), forward);
    right = glm::normalize(right);
    up = glm::cross(forward, right);

    // take care of the singularity by hardwiring in specific camera orientations
    if (eye.x == target.x && eye.z == target.z && eye.y > target.y)
    { // camera looking vertically down
        right = glm::vec3(0, 0, 1);
        up = glm::vec3(1, 0, 0);
        forward = glm::vec3(0, 1, 0);
    }

    if (eye.x == target.x && eye.z == target.z && eye.y < target.y)
    { // camera looking vertically up
        right = glm::vec3(1, 0, 0);
        up = glm::vec3(0, 0, 1);
        forward = glm::vec3(0, -1, 0);
    }
}
void Camera::rayDirection(const int i, const int j, Ray& ray) const
{
    ray.origin = eye;
    ray.direction = (float)(i - width / 2.0f) * right + (float)(j - height / 2.0f) * up - d * forward;
    ray.direction = glm::normalize(ray.direction);
}
void Camera::randomRayDirection(const int i, const int j, Ray& ray) const
{
    ray.origin = eye;

    /*double rand_halton[2];
    halton(rand_halton);

    rand_halton[0] -= 0.5;
    rand_halton[1] -= 0.5;

    ray.direction = glm::normalize((float)(i + rand_halton[0] - width*0.5f) * right + (float)(j + rand_halton[1] -
    height*0.5f) * v - d * forward);*/

    std::mt19937_64 mersenne;

    double random0 = (double)mersenne() / (double)mersenne.max();
    double random1 = (double)mersenne() / (double)mersenne.max();

    ray.direction = glm::normalize((float)(i + random0 - width * 0.5f) * right +
                                   (float)(j + random1 - height * 0.5f) * up - d * forward);

    // if (halton_step_get() > 2000)
    //	halton_step_set(0);
}
void Camera::rayDirection(const int i, const int j, std::vector<Ray>& rays) const
{
    const int numDivs = 2; // Number of subcells = numDivs^2

    const float step = 1.0f / numDivs;

    rays.clear();
    // Ray ray;

    for (float k = -0.5f; k < 0.49999f; k += step)
    {
        for (float l = -0.5f; l < 0.49999f; l += step)
        {

            /*ray.origin = eye;
            ray.direction = (float)(i + l - width*0.5f) * right + (float)(j + k - height*0.5f) * v - d * forward;
            ray.direction = glm::normalize(ray.direction);
            rays.push_back(ray);*/
            rays.emplace_back(eye,
                              glm::normalize((float)(i + l - width * 0.5f) * right +
                                             (float)(j + k - height * 0.5f) * up - d * forward));
        }
    }
}

void Camera::rayDirection(const int i, const int j, std::vector<Ray>& rays, const unsigned int nRays) const
{
    const float numDivs = std::ceil(sqrtf(static_cast<float>(nRays))); // Number of subcells = numDivs^2
    const float step = 1.0f / numDivs;

    unsigned int count = 0;

    rays.clear();

    for (float k = -0.5f; k < 0.49999f; k += step)
    {
        for (float l = -0.5f; l < 0.49999f; l += step)
        {
            rays.emplace_back(eye,
                              glm::normalize((float)(i + l - width * 0.5f) * right +
                                             (float)(j + k - height * 0.5f) * up - d * forward));
            count++;
            if (count >= nRays)
                return;
        }
    }
}
void Camera::fit(const BoundingBox& bb)
{
    glm::vec3 target = bb.getCenter();
    glm::vec3 size = glm::abs((bb.max - bb.min) / 2.f);
    eye = target + glm::vec3(-.5f * size.x, .4f * size.y, 1.f * size.z);
    lookAt(target);

    // return;

    // !Not times two! The fov is actually times two :)
    float verticalfov = atan2(height * .5f, d);
    float horizontalfov = atan2(width * .5f, d);

    glm::vec3 v[8];
    v[0] = bb.max;
    v[1] = bb.min;
    v[2] = glm::vec3(bb.min.x, bb.min.y, bb.max.z);
    v[3] = glm::vec3(bb.min.x, bb.max.y, bb.max.z);
    v[4] = glm::vec3(bb.min.x, bb.max.y, bb.min.z);
    v[5] = glm::vec3(bb.max.x, bb.min.y, bb.min.z);
    v[6] = glm::vec3(bb.max.x, bb.min.y, bb.max.z);
    v[7] = glm::vec3(bb.max.x, bb.max.y, bb.min.z);

    float avg = (size.x + size.y + size.z) / 3.f;

    float step = avg * 0.05f;

    for (int i = 0; i < 8; i++)
    {
        float angle;
        glm::vec3 vectov;
        do
        {
            vectov = v[i] - eye;
            angle = acos(glm::dot(forward, vectov));
            // Back the eye
            eye += forward * step;
        } while (angle >= verticalfov || angle >= horizontalfov);
    }
}
float Camera::fov()
{
    return 2 * atan2(height * .5f, d);
}