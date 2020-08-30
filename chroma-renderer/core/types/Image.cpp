#include "chroma-renderer/core/types/Image.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>

void Image::update()
{
    if (has_data_changed_)
    {
        updateOpenGLTexture();
    }
}

void Image::genOpenGLTexture()
{
    assert(width_ > 0 && height_ > 0);

    glDeleteTextures(1, &texture_id);

    // Gen and upload texture
    GLint last_texture{0};
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &last_texture);

    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    // glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    // glPixelStorei(GL_PACK_ALIGNMENT, 1);

    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, buffer_);
    if (color_components_ == 1)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width_, height_, 0, GL_RED, GL_FLOAT, nullptr);
    }
    else
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width_, height_, 0, GL_RGBA, GL_FLOAT, nullptr);
    }

    // Restore state
    glBindTexture(GL_TEXTURE_2D, last_texture);

    has_data_changed_ = false;
}

GLuint Image::getOpenGLTextureID()
{
    if (texture_id == 0)
    {
        genOpenGLTexture();
    }

    return texture_id;
}

void Image::readDataFromOpenGLTexture()
{
    if (texture_id == 0)
    {
        return;
    }

    glBindTexture(GL_TEXTURE_2D, texture_id);

    if (color_components_ == 1)
    {
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, &buffer_[0]);
    }
    else
    {
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, &buffer_[0]);
    }

    glBindTexture(GL_TEXTURE_2D, 0);
}

void Image::updateOpenGLTexture()
{
    if (texture_id == 0)
    {
        return;
    }

    glBindTexture(GL_TEXTURE_2D, texture_id);

    if (color_components_ == 1)
    {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RED, GL_FLOAT, &buffer_[0]);
    }
    else
    {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGBA, GL_FLOAT, &buffer_[0]);
    }

    glBindTexture(GL_TEXTURE_2D, 0);

    has_data_changed_ = false;
}

void Image::setColor(std::uint32_t widthPixelPos,
                     std::uint32_t heightPixelPos,
                     std::uint32_t r,
                     std::uint32_t g,
                     std::uint32_t b,
                     std::uint32_t a)
{
    if (widthPixelPos >= width_ || heightPixelPos >= height_)
    {
        return;
    }
    buffer_[mapPosToArray(widthPixelPos, heightPixelPos) + 0] = std::max(0.0f, std::min(255.0f, (float)r));
    buffer_[mapPosToArray(widthPixelPos, heightPixelPos) + 1] = std::max(0.0f, std::min(255.0f, (float)g));
    buffer_[mapPosToArray(widthPixelPos, heightPixelPos) + 2] = std::max(0.0f, std::min(255.0f, (float)b));
    buffer_[mapPosToArray(widthPixelPos, heightPixelPos) + 3] = std::max(0.0f, std::min(255.0f, (float)a));

    has_data_changed_ = true;
}

void Image::setColor(std::uint32_t widthPixelPos, std::uint32_t heightPixelPos, const Color& color)
{
    if (widthPixelPos >= width_ || heightPixelPos >= height_)
    {
        return;
    }
    buffer_[mapPosToArray(widthPixelPos, heightPixelPos) + 0] = 255 * std::max(0.0f, std::min(1.0f, color.r));
    buffer_[mapPosToArray(widthPixelPos, heightPixelPos) + 1] = 255 * std::max(0.0f, std::min(1.0f, color.g));
    buffer_[mapPosToArray(widthPixelPos, heightPixelPos) + 2] = 255 * std::max(0.0f, std::min(1.0f, color.b));
    buffer_[mapPosToArray(widthPixelPos, heightPixelPos) + 3] = 255;

    has_data_changed_ = true;
}

Color Image::getColor(std::uint32_t widthPixelPos, std::uint32_t heightPixelPos)
{
    if (widthPixelPos >= width_ || heightPixelPos >= height_)
    {
        return Color(0.0f);
    }

    Color c;

    c.r = buffer_[mapPosToArray(widthPixelPos, heightPixelPos) + 0];
    c.g = buffer_[mapPosToArray(widthPixelPos, heightPixelPos) + 1];
    c.b = buffer_[mapPosToArray(widthPixelPos, heightPixelPos) + 2];
    return c;
}

void Image::clear()
{
    memset(&buffer_[0], 0, color_components_ * width_ * height_ * 4);
    has_data_changed_ = true;
}

void Image::setSize(std::uint32_t _width, std::uint32_t _height)
{
    if (_width == width_ && _height == height_)
    {
        return;
    }

    width_ = _width;
    height_ = _height;

    createBuffer();
    genOpenGLTexture();
}

void Image::setData(const float* data,
                    const std::uint32_t _width,
                    const std::uint32_t _height,
                    const std::uint8_t components)
{
    color_components_ = components;
    setSize(_width, _height);
    memcpy(&buffer_[0], data, color_components_ * width_ * height_ * 4);
    genOpenGLTexture();
    updateOpenGLTexture();
}

void Image::createBuffer()
{
    assert(width_ > 0 && height_ > 0);
    buffer_.assign(color_components_ * width_ * height_, 0);
}