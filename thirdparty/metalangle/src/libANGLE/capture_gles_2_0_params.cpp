//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// capture_gles2_params.cpp:
//   Pointer parameter capture functions for the OpenGL ES 2.0 entry points.

#include "libANGLE/capture_gles_2_0_autogen.h"

#include "libANGLE/Context.h"
#include "libANGLE/Shader.h"
#include "libANGLE/formatutils.h"

using namespace angle;

namespace gl
{
// Parameter Captures

void CaptureBindAttribLocation_name(const Context *context,
                                    bool isCallValid,
                                    ShaderProgramID program,
                                    GLuint index,
                                    const GLchar *name,
                                    ParamCapture *paramCapture)
{
    CaptureString(name, paramCapture);
}

void CaptureBufferData_data(const Context *context,
                            bool isCallValid,
                            BufferBinding targetPacked,
                            GLsizeiptr size,
                            const void *data,
                            BufferUsage usagePacked,
                            ParamCapture *paramCapture)
{
    if (data)
    {
        CaptureMemory(data, size, paramCapture);
    }
}

void CaptureBufferSubData_data(const Context *context,
                               bool isCallValid,
                               BufferBinding targetPacked,
                               GLintptr offset,
                               GLsizeiptr size,
                               const void *data,
                               ParamCapture *paramCapture)
{
    CaptureMemory(data, size, paramCapture);
}

void CaptureCompressedTexImage2D_data(const Context *context,
                                      bool isCallValid,
                                      TextureTarget targetPacked,
                                      GLint level,
                                      GLenum internalformat,
                                      GLsizei width,
                                      GLsizei height,
                                      GLint border,
                                      GLsizei imageSize,
                                      const void *data,
                                      ParamCapture *paramCapture)
{
    if (context->getState().getTargetBuffer(gl::BufferBinding::PixelUnpack))
    {
        return;
    }

    if (!data)
    {
        return;
    }

    CaptureMemory(data, imageSize, paramCapture);
}

void CaptureCompressedTexSubImage2D_data(const Context *context,
                                         bool isCallValid,
                                         TextureTarget targetPacked,
                                         GLint level,
                                         GLint xoffset,
                                         GLint yoffset,
                                         GLsizei width,
                                         GLsizei height,
                                         GLenum format,
                                         GLsizei imageSize,
                                         const void *data,
                                         ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureDeleteBuffers_buffersPacked(const Context *context,
                                        bool isCallValid,
                                        GLsizei n,
                                        const BufferID *buffers,
                                        ParamCapture *paramCapture)
{
    CaptureMemory(buffers, sizeof(BufferID) * n, paramCapture);
}

void CaptureDeleteFramebuffers_framebuffersPacked(const Context *context,
                                                  bool isCallValid,
                                                  GLsizei n,
                                                  const FramebufferID *framebuffers,
                                                  ParamCapture *paramCapture)
{
    CaptureMemory(framebuffers, sizeof(FramebufferID) * n, paramCapture);
}

void CaptureDeleteRenderbuffers_renderbuffersPacked(const Context *context,
                                                    bool isCallValid,
                                                    GLsizei n,
                                                    const RenderbufferID *renderbuffers,
                                                    ParamCapture *paramCapture)
{
    CaptureMemory(renderbuffers, sizeof(RenderbufferID) * n, paramCapture);
}

void CaptureDeleteTextures_texturesPacked(const Context *context,
                                          bool isCallValid,
                                          GLsizei n,
                                          const TextureID *textures,
                                          ParamCapture *paramCapture)
{
    CaptureMemory(textures, sizeof(TextureID) * n, paramCapture);
}

void CaptureDrawElements_indices(const Context *context,
                                 bool isCallValid,
                                 PrimitiveMode modePacked,
                                 GLsizei count,
                                 DrawElementsType typePacked,
                                 const void *indices,
                                 ParamCapture *paramCapture)
{
    if (context->getState().getVertexArray()->getElementArrayBuffer())
    {
        paramCapture->value.voidConstPointerVal = indices;
    }
    else
    {
        GLuint typeSize = gl::GetDrawElementsTypeSize(typePacked);
        CaptureMemory(indices, typeSize * count, paramCapture);
        paramCapture->value.voidConstPointerVal = paramCapture->data[0].data();
    }
}

void CaptureGenBuffers_buffersPacked(const Context *context,
                                     bool isCallValid,
                                     GLsizei n,
                                     BufferID *buffers,
                                     ParamCapture *paramCapture)
{
    CaptureGenHandles(n, buffers, paramCapture);
}

void CaptureGenFramebuffers_framebuffersPacked(const Context *context,
                                               bool isCallValid,
                                               GLsizei n,
                                               FramebufferID *framebuffers,
                                               ParamCapture *paramCapture)
{
    CaptureGenHandles(n, framebuffers, paramCapture);
}

void CaptureGenRenderbuffers_renderbuffersPacked(const Context *context,
                                                 bool isCallValid,
                                                 GLsizei n,
                                                 RenderbufferID *renderbuffers,
                                                 ParamCapture *paramCapture)
{
    CaptureGenHandles(n, renderbuffers, paramCapture);
}

void CaptureGenTextures_texturesPacked(const Context *context,
                                       bool isCallValid,
                                       GLsizei n,
                                       TextureID *textures,
                                       ParamCapture *paramCapture)
{
    CaptureGenHandles(n, textures, paramCapture);
}

void CaptureGetActiveAttrib_length(const Context *context,
                                   bool isCallValid,
                                   ShaderProgramID program,
                                   GLuint index,
                                   GLsizei bufSize,
                                   GLsizei *length,
                                   GLint *size,
                                   GLenum *type,
                                   GLchar *name,
                                   ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetActiveAttrib_size(const Context *context,
                                 bool isCallValid,
                                 ShaderProgramID program,
                                 GLuint index,
                                 GLsizei bufSize,
                                 GLsizei *length,
                                 GLint *size,
                                 GLenum *type,
                                 GLchar *name,
                                 ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetActiveAttrib_type(const Context *context,
                                 bool isCallValid,
                                 ShaderProgramID program,
                                 GLuint index,
                                 GLsizei bufSize,
                                 GLsizei *length,
                                 GLint *size,
                                 GLenum *type,
                                 GLchar *name,
                                 ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetActiveAttrib_name(const Context *context,
                                 bool isCallValid,
                                 ShaderProgramID program,
                                 GLuint index,
                                 GLsizei bufSize,
                                 GLsizei *length,
                                 GLint *size,
                                 GLenum *type,
                                 GLchar *name,
                                 ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetActiveUniform_length(const Context *context,
                                    bool isCallValid,
                                    ShaderProgramID program,
                                    GLuint index,
                                    GLsizei bufSize,
                                    GLsizei *length,
                                    GLint *size,
                                    GLenum *type,
                                    GLchar *name,
                                    ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetActiveUniform_size(const Context *context,
                                  bool isCallValid,
                                  ShaderProgramID program,
                                  GLuint index,
                                  GLsizei bufSize,
                                  GLsizei *length,
                                  GLint *size,
                                  GLenum *type,
                                  GLchar *name,
                                  ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetActiveUniform_type(const Context *context,
                                  bool isCallValid,
                                  ShaderProgramID program,
                                  GLuint index,
                                  GLsizei bufSize,
                                  GLsizei *length,
                                  GLint *size,
                                  GLenum *type,
                                  GLchar *name,
                                  ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetActiveUniform_name(const Context *context,
                                  bool isCallValid,
                                  ShaderProgramID program,
                                  GLuint index,
                                  GLsizei bufSize,
                                  GLsizei *length,
                                  GLint *size,
                                  GLenum *type,
                                  GLchar *name,
                                  ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetAttachedShaders_count(const Context *context,
                                     bool isCallValid,
                                     ShaderProgramID program,
                                     GLsizei maxCount,
                                     GLsizei *count,
                                     ShaderProgramID *shaders,
                                     ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetAttachedShaders_shadersPacked(const Context *context,
                                             bool isCallValid,
                                             ShaderProgramID program,
                                             GLsizei maxCount,
                                             GLsizei *count,
                                             ShaderProgramID *shaders,
                                             ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetAttribLocation_name(const Context *context,
                                   bool isCallValid,
                                   ShaderProgramID program,
                                   const GLchar *name,
                                   ParamCapture *paramCapture)
{
    CaptureString(name, paramCapture);
}

void CaptureGetBooleanv_data(const Context *context,
                             bool isCallValid,
                             GLenum pname,
                             GLboolean *data,
                             ParamCapture *paramCapture)
{
    GLenum type;
    unsigned int numParams;
    context->getQueryParameterInfo(pname, &type, &numParams);
    paramCapture->readBufferSizeBytes = sizeof(GLboolean) * numParams;
}

void CaptureGetBufferParameteriv_params(const Context *context,
                                        bool isCallValid,
                                        BufferBinding targetPacked,
                                        GLenum pname,
                                        GLint *params,
                                        ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetFloatv_data(const Context *context,
                           bool isCallValid,
                           GLenum pname,
                           GLfloat *data,
                           ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetFramebufferAttachmentParameteriv_params(const Context *context,
                                                       bool isCallValid,
                                                       GLenum target,
                                                       GLenum attachment,
                                                       GLenum pname,
                                                       GLint *params,
                                                       ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetIntegerv_data(const Context *context,
                             bool isCallValid,
                             GLenum pname,
                             GLint *data,
                             ParamCapture *paramCapture)
{
    GLenum type;
    unsigned int numParams;
    context->getQueryParameterInfo(pname, &type, &numParams);
    paramCapture->readBufferSizeBytes = sizeof(GLint) * numParams;
}

void CaptureGetProgramInfoLog_length(const Context *context,
                                     bool isCallValid,
                                     ShaderProgramID program,
                                     GLsizei bufSize,
                                     GLsizei *length,
                                     GLchar *infoLog,
                                     ParamCapture *paramCapture)
{
    paramCapture->readBufferSizeBytes = sizeof(GLsizei);
}

void CaptureGetProgramInfoLog_infoLog(const Context *context,
                                      bool isCallValid,
                                      ShaderProgramID program,
                                      GLsizei bufSize,
                                      GLsizei *length,
                                      GLchar *infoLog,
                                      ParamCapture *paramCapture)
{
    gl::Program *programObj = context->getProgramResolveLink(program);
    ASSERT(programObj);
    paramCapture->readBufferSizeBytes = programObj->getInfoLogLength() + 1;
}

void CaptureGetProgramiv_params(const Context *context,
                                bool isCallValid,
                                ShaderProgramID program,
                                GLenum pname,
                                GLint *params,
                                ParamCapture *paramCapture)
{
    if (params)
    {
        paramCapture->readBufferSizeBytes = sizeof(GLint);
    }
}

void CaptureGetRenderbufferParameteriv_params(const Context *context,
                                              bool isCallValid,
                                              GLenum target,
                                              GLenum pname,
                                              GLint *params,
                                              ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetShaderInfoLog_length(const Context *context,
                                    bool isCallValid,
                                    ShaderProgramID shader,
                                    GLsizei bufSize,
                                    GLsizei *length,
                                    GLchar *infoLog,
                                    ParamCapture *paramCapture)
{
    paramCapture->readBufferSizeBytes = sizeof(GLsizei);
}

void CaptureGetShaderInfoLog_infoLog(const Context *context,
                                     bool isCallValid,
                                     ShaderProgramID shader,
                                     GLsizei bufSize,
                                     GLsizei *length,
                                     GLchar *infoLog,
                                     ParamCapture *paramCapture)
{
    gl::Shader *shaderObj = context->getShader(shader);
    ASSERT(shaderObj);
    paramCapture->readBufferSizeBytes = shaderObj->getInfoLogLength() + 1;
}

void CaptureGetShaderPrecisionFormat_range(const Context *context,
                                           bool isCallValid,
                                           GLenum shadertype,
                                           GLenum precisiontype,
                                           GLint *range,
                                           GLint *precision,
                                           ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetShaderPrecisionFormat_precision(const Context *context,
                                               bool isCallValid,
                                               GLenum shadertype,
                                               GLenum precisiontype,
                                               GLint *range,
                                               GLint *precision,
                                               ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetShaderSource_length(const Context *context,
                                   bool isCallValid,
                                   ShaderProgramID shader,
                                   GLsizei bufSize,
                                   GLsizei *length,
                                   GLchar *source,
                                   ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetShaderSource_source(const Context *context,
                                   bool isCallValid,
                                   ShaderProgramID shader,
                                   GLsizei bufSize,
                                   GLsizei *length,
                                   GLchar *source,
                                   ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetShaderiv_params(const Context *context,
                               bool isCallValid,
                               ShaderProgramID shader,
                               GLenum pname,
                               GLint *params,
                               ParamCapture *paramCapture)
{
    if (params)
    {
        paramCapture->readBufferSizeBytes = sizeof(GLint);
    }
}

void CaptureGetTexParameterfv_params(const Context *context,
                                     bool isCallValid,
                                     TextureType targetPacked,
                                     GLenum pname,
                                     GLfloat *params,
                                     ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetTexParameteriv_params(const Context *context,
                                     bool isCallValid,
                                     TextureType targetPacked,
                                     GLenum pname,
                                     GLint *params,
                                     ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetUniformLocation_name(const Context *context,
                                    bool isCallValid,
                                    ShaderProgramID program,
                                    const GLchar *name,
                                    ParamCapture *paramCapture)
{
    CaptureString(name, paramCapture);
}

void CaptureGetUniformfv_params(const Context *context,
                                bool isCallValid,
                                ShaderProgramID program,
                                GLint location,
                                GLfloat *params,
                                ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetUniformiv_params(const Context *context,
                                bool isCallValid,
                                ShaderProgramID program,
                                GLint location,
                                GLint *params,
                                ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetVertexAttribPointerv_pointer(const Context *context,
                                            bool isCallValid,
                                            GLuint index,
                                            GLenum pname,
                                            void **pointer,
                                            ParamCapture *paramCapture)
{
    paramCapture->readBufferSizeBytes = sizeof(void *);
}

void CaptureGetVertexAttribfv_params(const Context *context,
                                     bool isCallValid,
                                     GLuint index,
                                     GLenum pname,
                                     GLfloat *params,
                                     ParamCapture *paramCapture)
{
    // Can be up to 4 current state values.
    paramCapture->readBufferSizeBytes = sizeof(GLfloat) * 4;
}

void CaptureGetVertexAttribiv_params(const Context *context,
                                     bool isCallValid,
                                     GLuint index,
                                     GLenum pname,
                                     GLint *params,
                                     ParamCapture *paramCapture)
{
    // Can be up to 4 current state values.
    paramCapture->readBufferSizeBytes = sizeof(GLint) * 4;
}

void CaptureReadPixels_pixels(const Context *context,
                              bool isCallValid,
                              GLint x,
                              GLint y,
                              GLsizei width,
                              GLsizei height,
                              GLenum format,
                              GLenum type,
                              void *pixels,
                              ParamCapture *paramCapture)
{
    // Use a conservative upper bound instead of an exact size to be simple.
    static constexpr GLsizei kMaxPixelSize = 32;
    paramCapture->readBufferSizeBytes      = kMaxPixelSize * width * height;
}

void CaptureShaderBinary_shadersPacked(const Context *context,
                                       bool isCallValid,
                                       GLsizei count,
                                       const ShaderProgramID *shaders,
                                       GLenum binaryformat,
                                       const void *binary,
                                       GLsizei length,
                                       ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureShaderBinary_binary(const Context *context,
                                bool isCallValid,
                                GLsizei count,
                                const ShaderProgramID *shaders,
                                GLenum binaryformat,
                                const void *binary,
                                GLsizei length,
                                ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureShaderSource_string(const Context *context,
                                bool isCallValid,
                                ShaderProgramID shader,
                                GLsizei count,
                                const GLchar *const *string,
                                const GLint *length,
                                ParamCapture *paramCapture)
{
    for (GLsizei index = 0; index < count; ++index)
    {
        size_t len = ((length && length[index] >= 0) ? length[index] : strlen(string[index]));
        // includes the '\0' suffix
        std::vector<uint8_t> data(len + 1, 0);
        memcpy(data.data(), string[index], len);
        paramCapture->data.emplace_back(std::move(data));
    }
}

void CaptureShaderSource_length(const Context *context,
                                bool isCallValid,
                                ShaderProgramID shader,
                                GLsizei count,
                                const GLchar *const *string,
                                const GLint *length,
                                ParamCapture *paramCapture)
{
    if (!length)
        return;

    for (GLsizei index = 0; index < count; ++index)
    {
        CaptureMemory(&length[index], sizeof(GLint), paramCapture);
    }
}

void CaptureTexImage2D_pixels(const Context *context,
                              bool isCallValid,
                              TextureTarget targetPacked,
                              GLint level,
                              GLint internalformat,
                              GLsizei width,
                              GLsizei height,
                              GLint border,
                              GLenum format,
                              GLenum type,
                              const void *pixels,
                              ParamCapture *paramCapture)
{
    if (context->getState().getTargetBuffer(gl::BufferBinding::PixelUnpack))
    {
        return;
    }

    if (!pixels)
    {
        return;
    }

    const gl::InternalFormat &internalFormatInfo = gl::GetInternalFormatInfo(format, type);
    const gl::PixelUnpackState &unpack           = context->getState().getUnpackState();

    GLuint srcRowPitch = 0;
    (void)internalFormatInfo.computeRowPitch(type, width, unpack.alignment, unpack.rowLength,
                                             &srcRowPitch);
    GLuint srcDepthPitch = 0;
    (void)internalFormatInfo.computeDepthPitch(height, unpack.imageHeight, srcRowPitch,
                                               &srcDepthPitch);
    GLuint srcSkipBytes = 0;
    (void)internalFormatInfo.computeSkipBytes(type, srcRowPitch, srcDepthPitch, unpack, false,
                                              &srcSkipBytes);

    size_t captureSize = srcRowPitch * height + srcSkipBytes;
    CaptureMemory(pixels, captureSize, paramCapture);
}

void CaptureTexParameterfv_params(const Context *context,
                                  bool isCallValid,
                                  TextureType targetPacked,
                                  GLenum pname,
                                  const GLfloat *params,
                                  ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureTexParameteriv_params(const Context *context,
                                  bool isCallValid,
                                  TextureType targetPacked,
                                  GLenum pname,
                                  const GLint *params,
                                  ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureTexSubImage2D_pixels(const Context *context,
                                 bool isCallValid,
                                 TextureTarget targetPacked,
                                 GLint level,
                                 GLint xoffset,
                                 GLint yoffset,
                                 GLsizei width,
                                 GLsizei height,
                                 GLenum format,
                                 GLenum type,
                                 const void *pixels,
                                 ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureUniform1fv_value(const Context *context,
                             bool isCallValid,
                             GLint location,
                             GLsizei count,
                             const GLfloat *value,
                             ParamCapture *paramCapture)
{
    CaptureMemory(value, count * sizeof(GLfloat), paramCapture);
}

void CaptureUniform1iv_value(const Context *context,
                             bool isCallValid,
                             GLint location,
                             GLsizei count,
                             const GLint *value,
                             ParamCapture *paramCapture)
{
    CaptureMemory(value, count * sizeof(GLint), paramCapture);
}

void CaptureUniform2fv_value(const Context *context,
                             bool isCallValid,
                             GLint location,
                             GLsizei count,
                             const GLfloat *value,
                             ParamCapture *paramCapture)
{
    CaptureMemory(value, count * sizeof(GLfloat) * 2, paramCapture);
}

void CaptureUniform2iv_value(const Context *context,
                             bool isCallValid,
                             GLint location,
                             GLsizei count,
                             const GLint *value,
                             ParamCapture *paramCapture)
{
    CaptureMemory(value, count * sizeof(GLint) * 2, paramCapture);
}

void CaptureUniform3fv_value(const Context *context,
                             bool isCallValid,
                             GLint location,
                             GLsizei count,
                             const GLfloat *value,
                             ParamCapture *paramCapture)
{
    CaptureMemory(value, count * sizeof(GLfloat) * 3, paramCapture);
}

void CaptureUniform3iv_value(const Context *context,
                             bool isCallValid,
                             GLint location,
                             GLsizei count,
                             const GLint *value,
                             ParamCapture *paramCapture)
{
    CaptureMemory(value, count * sizeof(GLint) * 3, paramCapture);
}

void CaptureUniform4fv_value(const Context *context,
                             bool isCallValid,
                             GLint location,
                             GLsizei count,
                             const GLfloat *value,
                             ParamCapture *paramCapture)
{
    CaptureMemory(value, count * sizeof(GLfloat) * 4, paramCapture);
}

void CaptureUniform4iv_value(const Context *context,
                             bool isCallValid,
                             GLint location,
                             GLsizei count,
                             const GLint *value,
                             ParamCapture *paramCapture)
{
    CaptureMemory(value, count * sizeof(GLint) * 4, paramCapture);
}

void CaptureUniformMatrix2fv_value(const Context *context,
                                   bool isCallValid,
                                   GLint location,
                                   GLsizei count,
                                   GLboolean transpose,
                                   const GLfloat *value,
                                   ParamCapture *paramCapture)
{
    CaptureMemory(value, count * sizeof(GLfloat) * 4, paramCapture);
}

void CaptureUniformMatrix3fv_value(const Context *context,
                                   bool isCallValid,
                                   GLint location,
                                   GLsizei count,
                                   GLboolean transpose,
                                   const GLfloat *value,
                                   ParamCapture *paramCapture)
{
    CaptureMemory(value, count * sizeof(GLfloat) * 9, paramCapture);
}

void CaptureUniformMatrix4fv_value(const Context *context,
                                   bool isCallValid,
                                   GLint location,
                                   GLsizei count,
                                   GLboolean transpose,
                                   const GLfloat *value,
                                   ParamCapture *paramCapture)
{
    CaptureMemory(value, count * sizeof(GLfloat) * 16, paramCapture);
}

void CaptureVertexAttrib1fv_v(const Context *context,
                              bool isCallValid,
                              GLuint index,
                              const GLfloat *v,
                              ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureVertexAttrib2fv_v(const Context *context,
                              bool isCallValid,
                              GLuint index,
                              const GLfloat *v,
                              ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureVertexAttrib3fv_v(const Context *context,
                              bool isCallValid,
                              GLuint index,
                              const GLfloat *v,
                              ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureVertexAttrib4fv_v(const Context *context,
                              bool isCallValid,
                              GLuint index,
                              const GLfloat *v,
                              ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureVertexAttribPointer_pointer(const Context *context,
                                        bool isCallValid,
                                        GLuint index,
                                        GLint size,
                                        VertexAttribType typePacked,
                                        GLboolean normalized,
                                        GLsizei stride,
                                        const void *pointer,
                                        ParamCapture *paramCapture)
{
    paramCapture->value.voidConstPointerVal = pointer;
    if (!context->getState().getTargetBuffer(gl::BufferBinding::Array))
    {
        paramCapture->arrayClientPointerIndex = index;
    }
}
}  // namespace gl
