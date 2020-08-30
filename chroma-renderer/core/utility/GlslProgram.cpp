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
    GLint numShaders = 0;
    glGetProgramiv(handle_, GL_ATTACHED_SHADERS, &numShaders);

    if (numShaders > 0)
    {
        // Get the shader names
        std::vector<GLuint> shaderNames;
        shaderNames.resize(static_cast<unsigned long>(numShaders));
        glGetAttachedShaders(handle_, numShaders, nullptr, shaderNames.data());

        // Delete the shaders
        for (const auto& shaderName : shaderNames)
        {
            glDeleteShader(shaderName);
        }
    }

    // Delete the program
    glDeleteProgram(handle_);
}

void GLSLProgram::compileShader(const std::string& fileName)
{
    int numExts = sizeof(GLSLShaderInfo::extensions) / sizeof(GLSLShaderInfo::ShaderFileExtension);

    // Check the file name's extension to determine the shader type
    std::string ext = std::filesystem::path(fileName).extension().string();
    GLSLShader::GLSLShaderType type = GLSLShader::kVertex;
    bool matchFound = false;
    for (int i = 0; i < numExts; i++)
    {
        if (ext == GLSLShaderInfo::extensions[i].ext)
        {
            matchFound = true;
            type = GLSLShaderInfo::extensions[i].type;
            break;
        }
    }

    // If we didn't find a match, throw an exception
    if (!matchFound)
    {
        std::string msg = "Unrecognized extension: " + ext;
        throw GLSLProgramException(msg);
    }

    // Pass the discovered shader type along
    compileShader(fileName, type);
}

void GLSLProgram::compileShader(const std::string& fileName,
                                GLSLShader::GLSLShaderType type,
                                std::vector<std::string> defines,
                                const std::map<std::string, int>& definesINT)
{
    if (!std::filesystem::exists(fileName))
    {
        std::string message = std::string("Shader: ") + fileName + " not found.";
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

    ifstream inFile(fileName, ios::in);
    if (!inFile)
    {
        std::string message = std::string("Unable to open: ") + fileName;
        throw GLSLProgramException(message);
    }

    // Get file contents
    std::stringstream code;
    code << inFile.rdbuf();
    inFile.close();

    // Insert defines into code
    // Defines must be inserted after '#version' preprocessor

    std::string strcode = code.str();
    size_t pos = strcode.find('\n');

    for (const auto& x : definesINT)
    {
        strcode.insert(pos, std::string("\n#define ") + x.first + std::string(" ") + std::to_string(x.second));
    }

    for (size_t i = 0; i < defines.size(); i++)
    {
        strcode.insert(pos, std::string("\n#define ") + defines[i]);
    }

    compileShaderInternal(strcode, type, fileName);
}

void GLSLProgram::compileShaderInternal(const std::string& source,
                                        GLSLShader::GLSLShaderType type,
                                        const std::string& fileName)
{
    if (handle_ <= 0)
    {
        handle_ = glCreateProgram();
        if (handle_ == 0)
        {
            throw GLSLProgramException("Unable to create shader program.");
        }
    }

    GLuint shaderHandle = glCreateShader(type);

    const char* c_code = source.c_str();
    glShaderSource(shaderHandle, 1, &c_code, nullptr);

    // Compile the shader
    glCompileShader(shaderHandle);

    // Check for errors
    int result{GL_FALSE};
    glGetShaderiv(shaderHandle, GL_COMPILE_STATUS, &result);
    if (GL_FALSE != result)
    {
        // Compile succeeded, attach shader
        glAttachShader(handle_, shaderHandle);
    }
    else
    {
        // Compile failed, get log
        int length = 0;
        std::string logString;
        glGetShaderiv(shaderHandle, GL_INFO_LOG_LENGTH, &length);
        if (length > 0)
        {
            logString.resize(static_cast<unsigned long>(length));
            int written = 0;
            glGetShaderInfoLog(shaderHandle, length, &written, logString.data());
        }
        std::string msg = fileName + ": shader compliation failed\n";
        msg += logString;

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
        std::string logString;

        glGetProgramiv(handle_, GL_INFO_LOG_LENGTH, &length);

        if (length > 0)
        {
            logString.resize(static_cast<unsigned long>(length));
            int written = 0;
            glGetProgramInfoLog(handle_, length, &written, logString.data());
        }

        throw GLSLProgramException(std::string("Program link failed:\n") + logString);
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
    GLint numUniforms = 0;
    glGetProgramInterfaceiv(handle_, GL_UNIFORM, GL_ACTIVE_RESOURCES, &numUniforms);

    std::array<GLenum, 4> properties = {GL_NAME_LENGTH, GL_TYPE, GL_LOCATION, GL_BLOCK_INDEX};

    std::cout << "Active uniforms:" << std::endl;
    for (int i = 0; i < numUniforms; ++i)
    {
        std::array<GLint, 4> results{};
        glGetProgramResourceiv(handle_, GL_UNIFORM, i, 4, properties.data(), 4, nullptr, results.data());

        if (results[3] != -1)
        {
            continue; // Skip uniforms in blocks
        }
        GLint nameBufSize = results[0] + 1;
        std::string name;
        name.resize(static_cast<unsigned long>(nameBufSize));
        glGetProgramResourceName(handle_, GL_UNIFORM, i, nameBufSize, nullptr, name.data());
        std::cout << results[2] << " " << name << " (" << getTypeString(static_cast<GLenum>(results[1])) << ")"
                  << std::endl;
    }
}

void GLSLProgram::printActiveUniformBlocks() const
{
    GLint numBlocks = 0;

    glGetProgramInterfaceiv(handle_, GL_UNIFORM_BLOCK, GL_ACTIVE_RESOURCES, &numBlocks);
    std::array<GLenum, 2> blockProps = {GL_NUM_ACTIVE_VARIABLES, GL_NAME_LENGTH};
    std::array<GLenum, 1> blockIndex = {GL_ACTIVE_VARIABLES};
    std::array<GLenum, 3> props = {GL_NAME_LENGTH, GL_TYPE, GL_BLOCK_INDEX};

    for (int block = 0; block < numBlocks; ++block)
    {
        std::array<GLint, 2> blockInfo{};
        glGetProgramResourceiv(handle_, GL_UNIFORM_BLOCK, block, 2, blockProps.data(), 2, nullptr, blockInfo.data());
        GLint numUnis = blockInfo[0];
        GLint blockNameSize = blockInfo[1] + 1;

        std::string blockName;
        blockName.resize(static_cast<unsigned long>(blockNameSize));
        glGetProgramResourceName(handle_, GL_UNIFORM_BLOCK, block, blockNameSize, nullptr, blockName.data());
        std::cout << "Uniform block \"" << blockName << "\":" << std::endl;

        std::vector unifIndexes{numUnis};
        glGetProgramResourceiv(handle_,
                               GL_UNIFORM_BLOCK,
                               block,
                               1,
                               blockIndex.data(),
                               numUnis,
                               nullptr,
                               unifIndexes.data());

        for (int unif = 0; unif < numUnis; ++unif)
        {
            GLint uniIndex = unifIndexes[static_cast<unsigned long>(unif)];
            std::array<GLint, 3> results{};
            glGetProgramResourceiv(handle_, GL_UNIFORM, uniIndex, 3, props.data(), 3, nullptr, results.data());

            GLint nameBufSize = results[0] + 1;
            std::string name;
            name.resize(static_cast<unsigned long>(nameBufSize));
            glGetProgramResourceName(handle_, GL_UNIFORM, uniIndex, nameBufSize, nullptr, name.data());
            std::cout << "    " << name << " (" << getTypeString(static_cast<GLenum>(results[1])) << ")" << std::endl;
        }
    }
}

void GLSLProgram::printActiveAttribs() const
{
    GLint numAttribs{0};
    glGetProgramInterfaceiv(handle_, GL_PROGRAM_INPUT, GL_ACTIVE_RESOURCES, &numAttribs);

    std::array<GLenum, 3> properties = {GL_NAME_LENGTH, GL_TYPE, GL_LOCATION};

    std::cout << "Active attributes:" << std::endl;
    for (int i = 0; i < numAttribs; ++i)
    {
        std::array<GLint, 3> results{};
        glGetProgramResourceiv(handle_, GL_PROGRAM_INPUT, i, 3, properties.data(), 3, nullptr, results.data());

        const GLint nameBufSize = results[0] + 1;
        std::string name;
        name.resize(static_cast<unsigned long>(nameBufSize));
        glGetProgramResourceName(handle_, GL_PROGRAM_INPUT, i, nameBufSize, nullptr, name.data());
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
        std::string logString;

        glGetProgramiv(handle_, GL_INFO_LOG_LENGTH, &length);

        if (length > 0)
        {
            logString.resize(static_cast<unsigned long>(length));
            int written = 0;
            glGetProgramInfoLog(handle_, length, &written, logString.data());
        }

        throw GLSLProgramException(std::string("Program failed to validate\n") + logString);
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