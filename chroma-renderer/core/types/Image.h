#pragma once

#include "chroma-renderer/core/types/Color.h"

#include <cstdint>
#include <glad/glad.h>

class Image
{
  private:
    float* buffer;
    unsigned int width;
    unsigned int height;
    unsigned short colorComponents = 4;
    bool hasDataChanged;

  public:
    GLuint textureID;

    Image();
    ~Image();
    Color getColor(unsigned int widthPixelPos, unsigned int heightPixelPos);
    void setColor(unsigned int widthPixelPos,
                  unsigned int heightPixelPos,
                  unsigned int r,
                  unsigned int g,
                  unsigned int b,
                  unsigned int a);
    void setColor(unsigned int widthPixelPos, unsigned int heightPixelPos, const Color& color);
    void clear();
    void setSize(unsigned int width, unsigned int height);
    void setData(const float* data, const uint32_t width, const uint32_t height, const unsigned short components = 4);
    float getAspectRatio() const
    {
        return (float)width / (float)height;
    }
    inline unsigned int mapPosToArray(int i, int j)
    {
        return (width * j + i) * colorComponents;
    }
    inline unsigned int getWidth() const
    {
        return width;
    }
    inline unsigned int getHeight() const
    {
        return height;
    }
    inline const float* getBuffer()
    {
        return buffer;
    }

    void update();
    void genOpenGLTexture();
    GLuint getOpenGLTextureID();
    void readDataFromOpenGLTexture();
    void updateOpenGLTexture();

  private:
    void createBuffer();
};