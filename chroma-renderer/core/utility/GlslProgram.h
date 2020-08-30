#ifndef GLSLPROGRAM_H
#define GLSLPROGRAM_H
/*
Shader class from glslcookbook 4.0
*/

#ifdef WIN32
#pragma warning(disable : 4290)
#endif

#include <glad/glad.h>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using glm::mat3;
using glm::mat4;
using glm::vec2;
using glm::vec3;
using glm::vec4;
using std::string;
using std::vector;

class GLSLProgramException : public std::runtime_error
{
  public:
    explicit GLSLProgramException(const string& msg) : std::runtime_error(msg)
    {
    }
};

namespace GLSLShader
{
enum GLSLShaderType
{
    kVertex = GL_VERTEX_SHADER,
    kFragment = GL_FRAGMENT_SHADER,
    kGeometry = GL_GEOMETRY_SHADER,
    kTessControl = GL_TESS_CONTROL_SHADER,
    kTessEvaluation = GL_TESS_EVALUATION_SHADER,
    kCompute = GL_COMPUTE_SHADER
};
}

class GLSLProgram
{
  private:
    GLuint handle_{0};
    bool linked_{false};
    std::map<string, int> uniform_locations_;

    GLint getUniformLocation(const char* name);

  public:
    GLSLProgram() = default;
    GLSLProgram(const GLSLProgram&) = delete;
    GLSLProgram(GLSLProgram&&) = delete;
    GLSLProgram& operator=(const GLSLProgram&) = delete;
    GLSLProgram& operator=(GLSLProgram&&) = delete;
    ~GLSLProgram();

    void compileShader(const std::string& file_name);
    void compileShader(const std::string& file_name,
                       GLSLShader::GLSLShaderType type,
                       std::vector<string> defines = std::vector<string>(),
                       const std::map<std::string, int>& defines_int = std::map<std::string, int>());

  private:
    void compileShaderInternal(const std::string& source,
                               GLSLShader::GLSLShaderType type,
                               const std::string& file_name = "");

  public:
    void link();
    void validate() const;
    void use() const;

    GLuint getHandle() const;
    bool isLinked() const;

    void bindAttribLocation(GLuint location, const char* name) const;
    void bindFragDataLocation(GLuint location, const char* name) const;

    void setUniform(const char* name, float x, float y, float z);
    void setUniform(const char* name, const vec2& v);
    void setUniform(const char* name, const vec3& v);
    void setUniform(const char* name, const vec4& v);
    void setUniform(const char* name, const mat4& m);
    void setUniform(const char* name, const mat3& m);
    void setUniform(const char* name, float val);
    void setUniform(const char* name, double val);
    void setUniform(const char* name, int val);
    void setUniform(const char* name, bool val);
    void setUniform(const char* name, GLuint val);

    void printActiveUniforms() const;
    void printActiveUniformBlocks() const;
    void printActiveAttribs() const;

    static const char* getTypeString(GLenum type);
};

#endif // GLSLPROGRAM_H