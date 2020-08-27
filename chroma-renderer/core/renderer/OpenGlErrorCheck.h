#pragma once

#include <glad/glad.h>
#include <iostream>
#include <string>

inline std::string OpenGlErrorToString(const GLenum error)
{
    switch (error)
    {
    case GL_INVALID_ENUM:
        return "GL_INVALID_ENUM";
    case GL_INVALID_VALUE:
        return "GL_INVALID_VALUE";
    case GL_INVALID_OPERATION:
        return "GL_INVALID_OPERATION";
    case GL_STACK_OVERFLOW:
        return "GL_STACK_OVERFLOW";
    case GL_STACK_UNDERFLOW:
        return "GL_STACK_UNDERFLOW";
    case GL_OUT_OF_MEMORY:
        return "GL_OUT_OF_MEMORY";
    case GL_INVALID_FRAMEBUFFER_OPERATION:
        return "GL_INVALID_FRAMEBUFFER_OPERATION";
    case GL_CONTEXT_LOST:
        return "GL_CONTEXT_LOST";
    default:
        return "UNKNOWN";
    }
}

// NOLINTNEXTLINE
#define CHECK_OPENGL_ERROR                                                                                     \
    if (GLenum err = glGetError(); err != GL_NO_ERROR)                                                         \
    {                                                                                                          \
        std::cerr << __FILE__ << ":" << __LINE__ << " OpenGL error " << OpenGlErrorToString(err) << std::endl; \
    }
