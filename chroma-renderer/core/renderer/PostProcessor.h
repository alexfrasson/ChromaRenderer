#pragma once

#include "chroma-renderer/core/types/Image.h"
#include "chroma-renderer/core/utility/GlslProgram.h"

class PostProcessor
{
  public:
    PostProcessor();

    void process(const Image& src, const Image& dst, const bool sync = true);

  private:

    GLSLProgram computeShader;
};