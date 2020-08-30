#include "chroma-renderer/core/utility/GlslProgram.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

using std::ifstream;
using std::ios;

namespace GLSLShaderInfo
{
struct ShaderFileExtension
{
    const char* ext;
    GLSLShader::GLSLShaderType type;
};

struct ShaderFileExtension extensions[] = {{".vs", GLSLShader::kVertex},
                                           {".vert", GLSLShader::kVertex},
                                           {".gs", GLSLShader::kGeometry},
                                           {".geom", GLSLShader::kGeometry},
                                           {".tcs", GLSLShader::kTessControl},
                                           {".tes", GLSLShader::kTessEvaluation},
                                           {".fs", GLSLShader::kFragment},
                                           {".frag", GLSLShader::kFragment},
                                           {".cs", GLSLShader::kCompute}};
} // namespace GLSLShaderInfo

GLSLProgram::~GLSLProgram()
{
    if (handle_ == 0)
    {
        return;
    }

    // Query the number of attached shaders
    GLint num_shaders = 0;
    glGetProgramiv(handle_, GL_ATTACHED_SHADERS, &num_shaders);

    if (num_shaders > 0)
    {
        // Get the shader names
        std::vector<GLuint> shader_names;
        shader_names.resize(static_cast<unsigned long>(num_shaders));
        glGetAttachedShaders(handle_, num_shaders, nullptr, shader_names.data());

        // Delete the shaders
        for (const auto& shader_name : shader_names)
        {
            glDeleteShader(shader_name);
        }
    }

    // Delete the program
    glDeleteProgram(handle_);
}

void GLSLProgram::compileShader(const std::string& file_name)
{
    int num_exts = sizeof(GLSLShaderInfo::extensions) / sizeof(GLSLShaderInfo::ShaderFileExtension);

    // Check the file name's extension to determine the shader type
    std::string ext = std::filesystem::path(file_name).extension().string();
    GLSLShader::GLSLShaderType type = GLSLShader::kVertex;
    bool match_found = false;
    for (int i = 0; i < num_exts; i++)
    {
        if (ext == GLSLShaderInfo::extensions[i].ext)
        {
            match_found = true;
            type = GLSLShaderInfo::extensions[i].type;
            break;
        }
    }

    // If we didn't find a match, throw an exception
    if (!match_found)
    {
        std::string msg = "Unrecognized extension: " + ext;
        throw GLSLProgramException(msg);
    }

    // Pass the discovered shader type along
    compileShader(file_name, type);
}

void GLSLProgram::compileShader(const std::string& file_name,
                                GLSLShader::GLSLShaderType type,
                                std::vector<std::string> defines,
                                const std::map<std::string, int>& defines_int)
{
    if (!std::filesystem::exists(file_name))
    {
        std::string message = std::string("Shader: ") + file_name + " not found.";
        throw GLSLProgramException(message);
    }

    if (handle_ <= 0)
    {
        handle_ = glCreateProgram();
        if (handle_ == 0)
        {
            throw GLSLProgramException("Unable to create shader program.");
        }
    }

    ifstream in_file(file_name, ios::in);
    if (!in_file)
    {
        std::string message = std::string("Unable to open: ") + file_name;
        throw GLSLProgramException(message);
    }

    // Get file contents
    std::stringstream code;
    code << in_file.rdbuf();
    in_file.close();

    // Insert defines into code
    // Defines must be inserted after '#version' preprocessor

    std::string strcode = code.str();
    size_t pos = strcode.find('\n');

    for (const auto& x : defines_int)
    {
        strcode.insert(pos, std::string("\n#define ") + x.first + std::string(" ") + std::to_string(x.second));
    }

    for (size_t i = 0; i < defines.size(); i++)
    {
        strcode.insert(pos, std::string("\n#define ") + defines[i]);
    }

    compileShaderInternal(strcode, type, file_name);
}

void GLSLProgram::compileShaderInternal(const std::string& source,
                                        GLSLShader::GLSLShaderType type,
                                        const std::string& file_name)
{
    if (handle_ <= 0)
    {
        handle_ = glCreateProgram();
        if (handle_ == 0)
        {
            throw GLSLProgramException("Unable to create shader program.");
        }
    }

    GLuint shader_handle = glCreateShader(type);

    const char* c_code = source.c_str();
    glShaderSource(shader_handle, 1, &c_code, nullptr);

    // Compile the shader
    glCompileShader(shader_handle);

    // Check for errors
    int result{GL_FALSE};
    glGetShaderiv(shader_handle, GL_COMPILE_STATUS, &result);
    if (GL_FALSE != result)
    {
        // Compile succeeded, attach shader
        glAttachShader(handle_, shader_handle);
    }
    else
    {
        // Compile failed, get log
        int length = 0;
        std::string log_string;
        glGetShaderiv(shader_handle, GL_INFO_LOG_LENGTH, &length);
        if (length > 0)
        {
            log_string.resize(static_cast<unsigned long>(length));
            int written = 0;
            glGetShaderInfoLog(shader_handle, length, &written, log_string.data());
        }
        std::string msg = file_name + ": shader compliation failed\n";
        msg += log_string;

        throw GLSLProgramException(msg);
    }
}

void GLSLProgram::link()
{
    if (linked_)
    {
        return;
    }
    if (handle_ <= 0)
    {
        throw GLSLProgramException("Program has not been compiled.");
    }

    glLinkProgram(handle_);

    int status = 0;
    glGetProgramiv(handle_, GL_LINK_STATUS, &status);
    if (GL_FALSE != status)
    {
        uniform_locations_.clear();
        linked_ = true;
    }
    else
    {
        // Store log and return false
        int length = 0;
        std::string log_string;

        glGetProgramiv(handle_, GL_INFO_LOG_LENGTH, &length);

        if (length > 0)
        {
            log_string.resize(static_cast<unsigned long>(length));
            int written = 0;
            glGetProgramInfoLog(handle_, length, &written, log_string.data());
        }

        throw GLSLProgramException(std::string("Program link failed:\n") + log_string);
    }
}

void GLSLProgram::use() const
{
    if (handle_ <= 0 || (!linked_))
    {
        throw GLSLProgramException("Shader has not been linked");
    }
    glUseProgram(handle_);
}

GLuint GLSLProgram::getHandle() const
{
    return handle_;
}

bool GLSLProgram::isLinked() const
{
    return linked_;
}

void GLSLProgram::bindAttribLocation(GLuint location, const char* name) const
{
    glBindAttribLocation(handle_, location, name);
}

void GLSLProgram::bindFragDataLocation(GLuint location, const char* name) const
{
    glBindFragDataLocation(handle_, location, name);
}

void GLSLProgram::setUniform(const char* name, float x, float y, float z)
{
    GLint loc = getUniformLocation(name);
    glUniform3f(loc, x, y, z);
}

void GLSLProgram::setUniform(const char* name, const vec3& v)
{
    this->setUniform(name, v.x, v.y, v.z);
}

void GLSLProgram::setUniform(const char* name, const vec4& v)
{
    GLint loc = getUniformLocation(name);
    glUniform4f(loc, v.x, v.y, v.z, v.w);
}

void GLSLProgram::setUniform(const char* name, const vec2& v)
{
    GLint loc = getUniformLocation(name);
    glUniform2f(loc, v.x, v.y);
}

void GLSLProgram::setUniform(const char* name, const mat4& m)
{
    GLint loc = getUniformLocation(name);
    glUniformMatrix4fv(loc, 1, GL_FALSE, &m[0][0]);
}

void GLSLProgram::setUniform(const char* name, const mat3& m)
{
    GLint loc = getUniformLocation(name);
    glUniformMatrix3fv(loc, 1, GL_FALSE, &m[0][0]);
}

void GLSLProgram::setUniform(const char* name, float val)
{
    GLint loc = getUniformLocation(name);
    glUniform1f(loc, val);
}

void GLSLProgram::setUniform(const char* name, double val)
{
    GLint loc = getUniformLocation(name);
    glUniform1d(loc, val);
}

void GLSLProgram::setUniform(const char* name, int val)
{
    GLint loc = getUniformLocation(name);
    glUniform1i(loc, val);
}

void GLSLProgram::setUniform(const char* name, GLuint val)
{
    GLint loc = getUniformLocation(name);
    glUniform1ui(loc, val);
}

void GLSLProgram::setUniform(const char* name, bool val)
{
    int loc = getUniformLocation(name);
    glUniform1i(loc, static_cast<GLint>(val));
}

void GLSLProgram::printActiveUniforms() const
{
    GLint num_uniforms = 0;
    glGetProgramInterfaceiv(handle_, GL_UNIFORM, GL_ACTIVE_RESOURCES, &num_uniforms);

    std::array<GLenum, 4> properties = {GL_NAME_LENGTH, GL_TYPE, GL_LOCATION, GL_BLOCK_INDEX};

    std::cout << "Active uniforms:" << std::endl;
    for (int i = 0; i < num_uniforms; ++i)
    {
        std::array<GLint, 4> results{};
        glGetProgramResourceiv(handle_, GL_UNIFORM, i, 4, properties.data(), 4, nullptr, results.data());

        if (results[3] != -1)
        {
            continue; // Skip uniforms in blocks
        }
        GLint name_buf_size = results[0] + 1;
        std::string name;
        name.resize(static_cast<unsigned long>(name_buf_size));
        glGetProgramResourceName(handle_, GL_UNIFORM, i, name_buf_size, nullptr, name.data());
        std::cout << results[2] << " " << name << " (" << getTypeString(static_cast<GLenum>(results[1])) << ")"
                  << std::endl;
    }
}

void GLSLProgram::printActiveUniformBlocks() const
{
    GLint num_blocks = 0;

    glGetProgramInterfaceiv(handle_, GL_UNIFORM_BLOCK, GL_ACTIVE_RESOURCES, &num_blocks);
    std::array<GLenum, 2> block_props = {GL_NUM_ACTIVE_VARIABLES, GL_NAME_LENGTH};
    std::array<GLenum, 1> block_index = {GL_ACTIVE_VARIABLES};
    std::array<GLenum, 3> props = {GL_NAME_LENGTH, GL_TYPE, GL_BLOCK_INDEX};

    for (int block = 0; block < num_blocks; ++block)
    {
        std::array<GLint, 2> block_info{};
        glGetProgramResourceiv(handle_, GL_UNIFORM_BLOCK, block, 2, block_props.data(), 2, nullptr, block_info.data());
        GLint num_unis = block_info[0];
        GLint block_name_size = block_info[1] + 1;

        std::string block_name;
        block_name.resize(static_cast<unsigned long>(block_name_size));
        glGetProgramResourceName(handle_, GL_UNIFORM_BLOCK, block, block_name_size, nullptr, block_name.data());
        std::cout << "Uniform block \"" << block_name << "\":" << std::endl;

        std::vector unif_indexes{num_unis};
        glGetProgramResourceiv(handle_,
                               GL_UNIFORM_BLOCK,
                               block,
                               1,
                               block_index.data(),
                               num_unis,
                               nullptr,
                               unif_indexes.data());

        for (int unif = 0; unif < num_unis; ++unif)
        {
            GLint uni_index = unif_indexes[static_cast<unsigned long>(unif)];
            std::array<GLint, 3> results{};
            glGetProgramResourceiv(handle_, GL_UNIFORM, uni_index, 3, props.data(), 3, nullptr, results.data());

            GLint name_buf_size = results[0] + 1;
            std::string name;
            name.resize(static_cast<unsigned long>(name_buf_size));
            glGetProgramResourceName(handle_, GL_UNIFORM, uni_index, name_buf_size, nullptr, name.data());
            std::cout << "    " << name << " (" << getTypeString(static_cast<GLenum>(results[1])) << ")" << std::endl;
        }
    }
}

void GLSLProgram::printActiveAttribs() const
{
    GLint num_attribs{0};
    glGetProgramInterfaceiv(handle_, GL_PROGRAM_INPUT, GL_ACTIVE_RESOURCES, &num_attribs);

    std::array<GLenum, 3> properties = {GL_NAME_LENGTH, GL_TYPE, GL_LOCATION};

    std::cout << "Active attributes:" << std::endl;
    for (int i = 0; i < num_attribs; ++i)
    {
        std::array<GLint, 3> results{};
        glGetProgramResourceiv(handle_, GL_PROGRAM_INPUT, i, 3, properties.data(), 3, nullptr, results.data());

        const GLint name_buf_size = results[0] + 1;
        std::string name;
        name.resize(static_cast<unsigned long>(name_buf_size));
        glGetProgramResourceName(handle_, GL_PROGRAM_INPUT, i, name_buf_size, nullptr, name.data());
        std::cout << results[2] << " " << name.c_str() << " (" << getTypeString(static_cast<GLenum>(results[1])) << ")"
                  << std::endl;
    }
}

const char* GLSLProgram::getTypeString(GLenum type)
{
    // There are many more types than are covered here, but
    // these are the most common in these examples.
    switch (type)
    {
    case GL_FLOAT:
        return "float";
    case GL_FLOAT_VEC2:
        return "vec2";
    case GL_FLOAT_VEC3:
        return "vec3";
    case GL_FLOAT_VEC4:
        return "vec4";
    case GL_DOUBLE:
        return "double";
    case GL_INT:
        return "int";
    case GL_UNSIGNED_INT:
        return "unsigned int";
    case GL_BOOL:
        return "bool";
    case GL_FLOAT_MAT2:
        return "mat2";
    case GL_FLOAT_MAT3:
        return "mat3";
    case GL_FLOAT_MAT4:
        return "mat4";
    default:
        return "?";
    }
}

void GLSLProgram::validate() const
{
    if (!isLinked())
    {
        throw GLSLProgramException("Program is not linked");
    }

    GLint status{GL_FALSE};
    glValidateProgram(handle_);
    glGetProgramiv(handle_, GL_VALIDATE_STATUS, &status);

    if (GL_FALSE == status)
    {
        // Store log and return false
        int length = 0;
        std::string log_string;

        glGetProgramiv(handle_, GL_INFO_LOG_LENGTH, &length);

        if (length > 0)
        {
            log_string.resize(static_cast<unsigned long>(length));
            int written = 0;
            glGetProgramInfoLog(handle_, length, &written, log_string.data());
        }

        throw GLSLProgramException(std::string("Program failed to validate\n") + log_string);
    }
}

int GLSLProgram::getUniformLocation(const char* name)
{
    std::map<std::string, int>::iterator pos;
    pos = uniform_locations_.find(name);

    if (pos == uniform_locations_.end())
    {
        uniform_locations_[name] = glGetUniformLocation(handle_, name);
    }

    return uniform_locations_[name];
}