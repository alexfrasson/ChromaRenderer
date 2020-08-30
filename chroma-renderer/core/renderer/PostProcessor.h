#pragma once

#include "chroma-renderer/core/scene/Camera.h"
#include "chroma-renderer/core/types/Image.h"
#include "chroma-renderer/core/utility/GlslProgram.h"

class PostProcessor
{
  public:
    PostProcessor();

    void process(const Camera& cam, const Image& src, const Image& dst, bool sync = true);

    bool tonemapping{true};
    bool linear_to_srbg{true};
    bool adjust_exposure{true};

  private:
    GLSLProgram compute_shader_;
};