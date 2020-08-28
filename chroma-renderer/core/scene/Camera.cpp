#include "chroma-renderer/core/scene/Camera.h"

#include <glm/geometric.hpp>

#include <cmath>
#include <iostream>
#include <random>

void Camera::setSize(int pwidth, int pheight)
{
    aspectRatio = (float)pwidth / static_cast<float>(pheight);
    width = pwidth;
    height = pheight;
    d = ((float)pwidth / 2.0f) / tanf(m_HorizontalFOV / 2.0f);
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

void Camera::fit(const BoundingBox& bb)
{
    glm::vec3 target = bb.getCenter();
    glm::vec3 size = glm::abs((bb.max - bb.min) / 2.f);
    eye = target + glm::vec3(-.5f * size.x, .4f * size.y, 1.f * size.z);
    lookAt(target);

    // return;

    // !Not times two! The fov is actually times two :)
    float verticalfov = atan2f((float)height * 0.5f, d);
    float horizontalfov = atan2f((float)width * 0.5f, d);

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
        float angle{0.0f};
        glm::vec3 vectov;
        do
        {
            vectov = v[i] - eye;
            angle = acosf(glm::dot(forward, vectov));
            // Back the eye
            eye += forward * step;
        } while (angle >= verticalfov || angle >= horizontalfov);
    }
}

float Camera::fov() const
{
    return 2.0f * atan2f((float)height * 0.5f, d);
}