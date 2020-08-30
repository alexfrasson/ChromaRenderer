#include "chroma-renderer/core/renderer/PostProcessor.h"
#include "chroma-renderer/core/renderer/OpenGlErrorCheck.h"

#include <iostream>

PostProcessor::PostProcessor()
{
    try
    {
        computeShader.compileShader("./chroma-renderer/shaders/convergence.glsl", GLSLShader::kCompute);
        computeShader.link();
        computeShader.validate();
        computeShader.printActiveAttribs();
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

    computeShader.use();
    computeShader.setUniform("apperture", cam.apperture);
    computeShader.setUniform("shutterTime", cam.shutter_time);
    computeShader.setUniform("iso", cam.iso);
    computeShader.setUniform("tonemapping", tonemapping);
    computeShader.setUniform("linearToSrbg", linearToSrbg);
    computeShader.setUniform("adjustExposure", adjustExposure);
    computeShader.setUniform("srcImage", src_image_unit);
    computeShader.setUniform("dstImage", dst_image_unit);

    CHECK_OPENGL_ERROR

    glBindImageTexture(src_image_unit, src.texture_id, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindImageTexture(dst_image_unit, dst.texture_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

    CHECK_OPENGL_ERROR

    int nGroupsX = static_cast<int>(ceilf((float)src.getWidth() / 16.0f));
    int nGroupsY = static_cast<int>(ceilf((float)src.getHeight() / 16.0f));

    glDispatchCompute(nGroupsX, nGroupsY, 1);

    if (sync)
    {
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
    }

    CHECK_OPENGL_ERROR
}