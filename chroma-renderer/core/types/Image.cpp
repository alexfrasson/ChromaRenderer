#include "chroma-renderer/core/types/Image.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>

void Image::update()
{
    if (hasDataChanged)
    {
        updateOpenGLTexture();
    }
}

void Image::genOpenGLTexture()
{
    assert(width > 0 && height > 0);

    glDeleteTextures(1, &textureID);

    // Gen and upload texture
    GLint last_texture{0};
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &last_texture);

    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    // glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    // glPixelStorei(GL_PACK_ALIGNMENT, 1);

    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
    if (colorComponents == 1)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
    }
    else
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    }

    // Restore state
    glBindTexture(GL_TEXTURE_2D, last_texture);

    hasDataChanged = false;
}

GLuint Image::getOpenGLTextureID()
{
    if (textureID == 0)
    {
        genOpenGLTexture();
    }

    return textureID;
}

void Image::readDataFromOpenGLTexture()
{
    if (textureID == 0)
    {
        return;
    }

    glBindTexture(GL_TEXTURE_2D, textureID);

    if (colorComponents == 1)
    {
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, &buffer[0]);
    }
    else
    {
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, &buffer[0]);
    }

    glBindTexture(GL_TEXTURE_2D, 0);
}

void Image::updateOpenGLTexture()
{
    if (textureID == 0)
    {
        return;
    }

    glBindTexture(GL_TEXTURE_2D, textureID);

    if (colorComponents == 1)
    {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_FLOAT, &buffer[0]);
    }
    else
    {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT, &buffer[0]);
    }

    glBindTexture(GL_TEXTURE_2D, 0);

    hasDataChanged = false;
}

void Image::setColor(std::uint32_t widthPixelPos,
                     std::uint32_t heightPixelPos,
                     std::uint32_t r,
                     std::uint32_t g,
                     std::uint32_t b,
                     std::uint32_t a)
{
    if (widthPixelPos >= width || heightPixelPos >= height)
    {
        return;
    }
    buffer[mapPosToArray(widthPixelPos, heightPixelPos) + 0] = std::max(0.0f, std::min(255.0f, (float)r));
    buffer[mapPosToArray(widthPixelPos, heightPixelPos) + 1] = std::max(0.0f, std::min(255.0f, (float)g));
    buffer[mapPosToArray(widthPixelPos, heightPixelPos) + 2] = std::max(0.0f, std::min(255.0f, (float)b));
    buffer[mapPosToArray(widthPixelPos, heightPixelPos) + 3] = std::max(0.0f, std::min(255.0f, (float)a));

    hasDataChanged = true;
}

void Image::setColor(std::uint32_t widthPixelPos, std::uint32_t heightPixelPos, const Color& color)
{
    if (widthPixelPos >= width || heightPixelPos >= height)
    {
        return;
    }
    buffer[mapPosToArray(widthPixelPos, heightPixelPos) + 0] = 255 * std::max(0.0f, std::min(1.0f, color.r));
    buffer[mapPosToArray(widthPixelPos, heightPixelPos) + 1] = 255 * std::max(0.0f, std::min(1.0f, color.g));
    buffer[mapPosToArray(widthPixelPos, heightPixelPos) + 2] = 255 * std::max(0.0f, std::min(1.0f, color.b));
    buffer[mapPosToArray(widthPixelPos, heightPixelPos) + 3] = 255;

    hasDataChanged = true;
}

Color Image::getColor(std::uint32_t widthPixelPos, std::uint32_t heightPixelPos)
{
    if (widthPixelPos >= width || heightPixelPos >= height)
    {
        return Color(0.0f);
    }

    Color c;

    c.r = buffer[mapPosToArray(widthPixelPos, heightPixelPos) + 0];
    c.g = buffer[mapPosToArray(widthPixelPos, heightPixelPos) + 1];
    c.b = buffer[mapPosToArray(widthPixelPos, heightPixelPos) + 2];
    return c;
}

void Image::clear()
{
    memset(&buffer[0], 0, colorComponents * width * height * 4);
    hasDataChanged = true;
}

void Image::setSize(std::uint32_t _width, std::uint32_t _height)
{
    if (_width == width && _height == height)
    {
        return;
    }

    width = _width;
    height = _height;

    createBuffer();
    genOpenGLTexture();
}

void Image::setData(const float* data,
                    const std::uint32_t _width,
                    const std::uint32_t _height,
                    const std::uint8_t components)
{
    colorComponents = components;
    setSize(_width, _height);
    memcpy(&buffer[0], data, colorComponents * width * height * 4);
    genOpenGLTexture();
    updateOpenGLTexture();
}

void Image::createBuffer()
{
    assert(width > 0 && height > 0);
    buffer.assign(colorComponents * width * height, 0);
}