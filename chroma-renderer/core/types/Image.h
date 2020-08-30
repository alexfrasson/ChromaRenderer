#pragma once

#include "chroma-renderer/core/types/Color.h"

#include <cstdint>
#include <glad/glad.h>
#include <vector>

class Image
{
  private:
    std::vector<float> buffer_;
    std::uint32_t width_{0};
    std::uint32_t height_{0};
    std::uint8_t color_components_{4};
    bool has_data_changed_{false};

  public:
    GLuint texture_id{0};

    Color getColor(std::uint32_t width_pixel_pos, std::uint32_t height_pixel_pos);
    void setColor(std::uint32_t width_pixel_pos,
                  std::uint32_t height_pixel_pos,
                  std::uint32_t r,
                  std::uint32_t g,
                  std::uint32_t b,
                  std::uint32_t a);
    void setColor(std::uint32_t width_pixel_pos, std::uint32_t height_pixel_pos, const Color& color);
    void clear();
    void setSize(std::uint32_t width, std::uint32_t height);
    void setData(const float* data, std::uint32_t width, std::uint32_t height, std::uint8_t components = 4);
    float getAspectRatio() const
    {
        return (float)width_ / (float)height_;
    }
    inline std::uint32_t mapPosToArray(std::uint32_t i, std::uint32_t j) const
    {
        return (width_ * j + i) * color_components_;
    }
    inline std::uint32_t getWidth() const
    {
        return width_;
    }
    inline std::uint32_t getHeight() const
    {
        return height_;
    }
    inline const float* getBuffer()
    {
        return &buffer_[0];
    }

    void update();
    void genOpenGLTexture();
    GLuint getOpenGLTextureID();
    void readDataFromOpenGLTexture();
    void updateOpenGLTexture();

  private:
    void createBuffer();
};