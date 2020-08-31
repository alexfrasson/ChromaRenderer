#include "chroma-renderer/core/renderer/post_processor.h"
#include "chroma-renderer/core/renderer/opengl_error_check.h"

#include <iostream>

PostProcessor::PostProcessor()
{
    try
    {
        compute_shader_.compileShader("./chroma-renderer/shaders/convergence.glsl", glsl_shader::kCompute);
        compute_shader_.link();
        compute_shader_.validate();
        compute_shader_.printActiveAttribs();
    }
    catch (GLSLProgramException& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

void PostProcessor::process(const Camera& cam, const Image& src, const Image& dst, const bool sync)
{
    const std::int32_t src_image_unit = 0;
    const std::int32_t dst_image_unit = 1;

    compute_shader_.use();
    compute_shader_.setUniform("apperture", cam.apperture);
    compute_shader_.setUniform("shutterTime", cam.shutter_time);
    compute_shader_.setUniform("iso", cam.iso);
    compute_shader_.setUniform("tonemapping", tonemapping);
    compute_shader_.setUniform("linearToSrbg", linear_to_srbg);
    compute_shader_.setUniform("adjustExposure", adjust_exposure);
    compute_shader_.setUniform("srcImage", src_image_unit);
    compute_shader_.setUniform("dstImage", dst_image_unit);

    CHECK_OPENGL_ERROR

    glBindImageTexture(src_image_unit, src.texture_id, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindImageTexture(dst_image_unit, dst.texture_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

    CHECK_OPENGL_ERROR

    int n_groups_x = static_cast<int>(ceilf((float)src.getWidth() / 16.0f));
    int n_groups_y = static_cast<int>(ceilf((float)src.getHeight() / 16.0f));

    glDispatchCompute(n_groups_x, n_groups_y, 1);

    if (sync)
    {
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
    }

    CHECK_OPENGL_ERROR
}