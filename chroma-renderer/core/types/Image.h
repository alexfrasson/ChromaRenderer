#pragma once

#include "chroma-renderer/core/types/Color.h"

#include <cstdint>
#include <glad/glad.h>
#include <vector>

class Image
{
  private:
    std::vector<float> buffer;
    std::uint32_t width{0};
    std::uint32_t height{0};
    std::uint8_t colorComponents = 4;
    bool hasDataChanged{false};

  public:
    GLuint textureID{0};

    Color getColor(std::uint32_t widthPixelPos, std::uint32_t heightPixelPos);
    void setColor(std::uint32_t widthPixelPos,
                  std::uint32_t heightPixelPos,
                  std::uint32_t r,
                  std::uint32_t g,
                  std::uint32_t b,
                  std::uint32_t a);
    void setColor(std::uint32_t widthPixelPos, std::uint32_t heightPixelPos, const Color& color);
    void clear();
    void setSize(std::uint32_t width, std::uint32_t height);
    void setData(const float* data, std::uint32_t width, std::uint32_t height, std::uint8_t components = 4);
    float getAspectRatio() const
    {
        return (float)width / (float)height;
    }
    inline std::uint32_t mapPosToArray(std::uint32_t i, std::uint32_t j) const
    {
        return (width * j + i) * colorComponents;
    }
    inline std::uint32_t getWidth() const
    {
        return width;
    }
    inline std::uint32_t getHeight() const
    {
        return height;
    }
    inline const float* getBuffer()
    {
        return &buffer[0];
    }

    void update();
    void genOpenGLTexture();
    GLuint getOpenGLTextureID();
    void readDataFromOpenGLTexture();
    void updateOpenGLTexture();

  private:
    void createBuffer();
};