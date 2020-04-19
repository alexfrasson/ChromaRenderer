#include "chroma-renderer/core/types/Image.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>

Image::Image() : buffer(nullptr), width(0), height(0), hasDataChanged(false), textureID(0)
{
}

Image::~Image()
{
    if (buffer != nullptr)
        delete[] buffer;
}

void Image::update()
{
    if (hasDataChanged)
        updateOpenGLTexture();
}

void Image::genOpenGLTexture()
{
    assert(width > 0 && height > 0);
    assert(buffer != nullptr);

    glDeleteTextures(1, &textureID);

    // Gen and upload texture
    GLint last_texture;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &last_texture);

    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    // glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    // glPixelStorei(GL_PACK_ALIGNMENT, 1);

    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);

    // Restore state
    glBindTexture(GL_TEXTURE_2D, last_texture);

    hasDataChanged = false;
}

GLuint Image::getOpenGLTextureID()
{
    if (textureID == 0)
        genOpenGLTexture();

    return textureID;
}

void Image::readDataFromOpenGLTexture()
{
    if (textureID == 0 || buffer == NULL)
        return;

    glBindTexture(GL_TEXTURE_2D, textureID);

    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, buffer);

    glBindTexture(GL_TEXTURE_2D, 0);
}

void Image::updateOpenGLTexture()
{
    if (textureID == 0 || buffer == NULL)
        return;

    glBindTexture(GL_TEXTURE_2D, textureID);

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT, buffer);

    glBindTexture(GL_TEXTURE_2D, 0);

    hasDataChanged = false;
}

void Image::setColor(unsigned int widthPixelPos,
                     unsigned int heightPixelPos,
                     unsigned int r,
                     unsigned int g,
                     unsigned int b,
                     unsigned int a)
{
    if (widthPixelPos >= width || heightPixelPos >= height || buffer == NULL)
        return;
    buffer[mapPosToArray(widthPixelPos, heightPixelPos) + 0] = std::max(0.0f, std::min(255.0f, (float)r));
    buffer[mapPosToArray(widthPixelPos, heightPixelPos) + 1] = std::max(0.0f, std::min(255.0f, (float)g));
    buffer[mapPosToArray(widthPixelPos, heightPixelPos) + 2] = std::max(0.0f, std::min(255.0f, (float)b));
    buffer[mapPosToArray(widthPixelPos, heightPixelPos) + 3] = std::max(0.0f, std::min(255.0f, (float)a));

    hasDataChanged = true;
}

void Image::setColor(unsigned int widthPixelPos, unsigned int heightPixelPos, const Color& color)
{
    if (widthPixelPos >= width || heightPixelPos >= height || buffer == NULL)
        return;
    buffer[mapPosToArray(widthPixelPos, heightPixelPos) + 0] = 255 * std::max(0.0f, std::min(1.0f, color.r));
    buffer[mapPosToArray(widthPixelPos, heightPixelPos) + 1] = 255 * std::max(0.0f, std::min(1.0f, color.g));
    buffer[mapPosToArray(widthPixelPos, heightPixelPos) + 2] = 255 * std::max(0.0f, std::min(1.0f, color.b));
    buffer[mapPosToArray(widthPixelPos, heightPixelPos) + 3] = 255;

    hasDataChanged = true;
}

Color Image::getColor(unsigned int widthPixelPos, unsigned int heightPixelPos)
{
    if (widthPixelPos >= width || heightPixelPos >= height || buffer == NULL)
        return Color(0.0f);

    Color c;

    c.r = buffer[mapPosToArray(widthPixelPos, heightPixelPos) + 0];
    c.g = buffer[mapPosToArray(widthPixelPos, heightPixelPos) + 1];
    c.b = buffer[mapPosToArray(widthPixelPos, heightPixelPos) + 2];
    return c;
}

void Image::clear()
{
    memset(buffer, 0, colorComponents * width * height * 4);

    hasDataChanged = true;
}

void Image::setSize(unsigned int _width, unsigned int _height)
{
    if (_width == this->width && _height == this->height)
        return;

    this->width = _width;
    this->height = _height;

    createBuffer();

    genOpenGLTexture();
}

void Image::createBuffer()
{
    assert(width > 0 && height > 0);

    delete[] buffer;
    buffer = new float[colorComponents * width * height];
    memset(buffer, 0, colorComponents * width * height * 4);
}