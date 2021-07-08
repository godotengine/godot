//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// capture_gles3_params.cpp:
//   Pointer parameter capture functions for the OpenGL ES 3.0 entry points.

#include "libANGLE/capture_gles_3_0_autogen.h"

using namespace angle;

namespace gl
{
void CaptureClearBufferfv_value(const Context *context,
                                bool isCallValid,
                                GLenum buffer,
                                GLint drawbuffer,
                                const GLfloat *value,
                                ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureClearBufferiv_value(const Context *context,
                                bool isCallValid,
                                GLenum buffer,
                                GLint drawbuffer,
                                const GLint *value,
                                ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureClearBufferuiv_value(const Context *context,
                                 bool isCallValid,
                                 GLenum buffer,
                                 GLint drawbuffer,
                                 const GLuint *value,
                                 ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureCompressedTexImage3D_data(const Context *context,
                                      bool isCallValid,
                                      TextureTarget targetPacked,
                                      GLint level,
                                      GLenum internalformat,
                                      GLsizei width,
                                      GLsizei height,
                                      GLsizei depth,
                                      GLint border,
                                      GLsizei imageSize,
                                      const void *data,
                                      ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureCompressedTexSubImage3D_data(const Context *context,
                                         bool isCallValid,
                                         TextureTarget targetPacked,
                                         GLint level,
                                         GLint xoffset,
                                         GLint yoffset,
                                         GLint zoffset,
                                         GLsizei width,
                                         GLsizei height,
                                         GLsizei depth,
                                         GLenum format,
                                         GLsizei imageSize,
                                         const void *data,
                                         ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureDeleteQueries_idsPacked(const Context *context,
                                    bool isCallValid,
                                    GLsizei n,
                                    const QueryID *ids,
                                    ParamCapture *paramCapture)
{
    CaptureMemory(ids, sizeof(QueryID) * n, paramCapture);
}

void CaptureDeleteSamplers_samplersPacked(const Context *context,
                                          bool isCallValid,
                                          GLsizei count,
                                          const SamplerID *samplers,
                                          ParamCapture *paramCapture)
{
    CaptureMemory(samplers, sizeof(SamplerID) * count, paramCapture);
}

void CaptureDeleteTransformFeedbacks_idsPacked(const Context *context,
                                               bool isCallValid,
                                               GLsizei n,
                                               const TransformFeedbackID *ids,
                                               ParamCapture *paramCapture)
{
    CaptureMemory(ids, sizeof(TransformFeedbackID) * n, paramCapture);
}

void CaptureDeleteVertexArrays_arraysPacked(const Context *context,
                                            bool isCallValid,
                                            GLsizei n,
                                            const VertexArrayID *arrays,
                                            ParamCapture *paramCapture)
{
    CaptureMemory(arrays, sizeof(VertexArrayID) * n, paramCapture);
}

void CaptureDrawBuffers_bufs(const Context *context,
                             bool isCallValid,
                             GLsizei n,
                             const GLenum *bufs,
                             ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureDrawElementsInstanced_indices(const Context *context,
                                          bool isCallValid,
                                          PrimitiveMode modePacked,
                                          GLsizei count,
                                          DrawElementsType typePacked,
                                          const void *indices,
                                          GLsizei instancecount,
                                          ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureDrawRangeElements_indices(const Context *context,
                                      bool isCallValid,
                                      PrimitiveMode modePacked,
                                      GLuint start,
                                      GLuint end,
                                      GLsizei count,
                                      DrawElementsType typePacked,
                                      const void *indices,
                                      ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGenQueries_idsPacked(const Context *context,
                                 bool isCallValid,
                                 GLsizei n,
                                 QueryID *ids,
                                 ParamCapture *paramCapture)
{
    CaptureGenHandles(n, ids, paramCapture);
}

void CaptureGenSamplers_samplersPacked(const Context *context,
                                       bool isCallValid,
                                       GLsizei count,
                                       SamplerID *samplers,
                                       ParamCapture *paramCapture)
{
    CaptureGenHandles(count, samplers, paramCapture);
}

void CaptureGenTransformFeedbacks_idsPacked(const Context *context,
                                            bool isCallValid,
                                            GLsizei n,
                                            TransformFeedbackID *ids,
                                            ParamCapture *paramCapture)
{
    CaptureGenHandles(n, ids, paramCapture);
}

void CaptureGenVertexArrays_arraysPacked(const Context *context,
                                         bool isCallValid,
                                         GLsizei n,
                                         VertexArrayID *arrays,
                                         ParamCapture *paramCapture)
{
    CaptureGenHandles(n, arrays, paramCapture);
}

void CaptureGetActiveUniformBlockName_length(const Context *context,
                                             bool isCallValid,
                                             ShaderProgramID program,
                                             GLuint uniformBlockIndex,
                                             GLsizei bufSize,
                                             GLsizei *length,
                                             GLchar *uniformBlockName,
                                             ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetActiveUniformBlockName_uniformBlockName(const Context *context,
                                                       bool isCallValid,
                                                       ShaderProgramID program,
                                                       GLuint uniformBlockIndex,
                                                       GLsizei bufSize,
                                                       GLsizei *length,
                                                       GLchar *uniformBlockName,
                                                       ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetActiveUniformBlockiv_params(const Context *context,
                                           bool isCallValid,
                                           ShaderProgramID program,
                                           GLuint uniformBlockIndex,
                                           GLenum pname,
                                           GLint *params,
                                           ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetActiveUniformsiv_uniformIndices(const Context *context,
                                               bool isCallValid,
                                               ShaderProgramID program,
                                               GLsizei uniformCount,
                                               const GLuint *uniformIndices,
                                               GLenum pname,
                                               GLint *params,
                                               ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetActiveUniformsiv_params(const Context *context,
                                       bool isCallValid,
                                       ShaderProgramID program,
                                       GLsizei uniformCount,
                                       const GLuint *uniformIndices,
                                       GLenum pname,
                                       GLint *params,
                                       ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetBufferParameteri64v_params(const Context *context,
                                          bool isCallValid,
                                          BufferBinding targetPacked,
                                          GLenum pname,
                                          GLint64 *params,
                                          ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetBufferPointerv_params(const Context *context,
                                     bool isCallValid,
                                     BufferBinding targetPacked,
                                     GLenum pname,
                                     void **params,
                                     ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetFragDataLocation_name(const Context *context,
                                     bool isCallValid,
                                     ShaderProgramID program,
                                     const GLchar *name,
                                     ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetInteger64i_v_data(const Context *context,
                                 bool isCallValid,
                                 GLenum target,
                                 GLuint index,
                                 GLint64 *data,
                                 ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetInteger64v_data(const Context *context,
                               bool isCallValid,
                               GLenum pname,
                               GLint64 *data,
                               ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetIntegeri_v_data(const Context *context,
                               bool isCallValid,
                               GLenum target,
                               GLuint index,
                               GLint *data,
                               ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetInternalformativ_params(const Context *context,
                                       bool isCallValid,
                                       GLenum target,
                                       GLenum internalformat,
                                       GLenum pname,
                                       GLsizei bufSize,
                                       GLint *params,
                                       ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetProgramBinary_length(const Context *context,
                                    bool isCallValid,
                                    ShaderProgramID program,
                                    GLsizei bufSize,
                                    GLsizei *length,
                                    GLenum *binaryFormat,
                                    void *binary,
                                    ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetProgramBinary_binaryFormat(const Context *context,
                                          bool isCallValid,
                                          ShaderProgramID program,
                                          GLsizei bufSize,
                                          GLsizei *length,
                                          GLenum *binaryFormat,
                                          void *binary,
                                          ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetProgramBinary_binary(const Context *context,
                                    bool isCallValid,
                                    ShaderProgramID program,
                                    GLsizei bufSize,
                                    GLsizei *length,
                                    GLenum *binaryFormat,
                                    void *binary,
                                    ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetQueryObjectuiv_params(const Context *context,
                                     bool isCallValid,
                                     QueryID id,
                                     GLenum pname,
                                     GLuint *params,
                                     ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetQueryiv_params(const Context *context,
                              bool isCallValid,
                              QueryType targetPacked,
                              GLenum pname,
                              GLint *params,
                              ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetSamplerParameterfv_params(const Context *context,
                                         bool isCallValid,
                                         SamplerID sampler,
                                         GLenum pname,
                                         GLfloat *params,
                                         ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetSamplerParameteriv_params(const Context *context,
                                         bool isCallValid,
                                         SamplerID sampler,
                                         GLenum pname,
                                         GLint *params,
                                         ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetSynciv_length(const Context *context,
                             bool isCallValid,
                             GLsync sync,
                             GLenum pname,
                             GLsizei bufSize,
                             GLsizei *length,
                             GLint *values,
                             ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetSynciv_values(const Context *context,
                             bool isCallValid,
                             GLsync sync,
                             GLenum pname,
                             GLsizei bufSize,
                             GLsizei *length,
                             GLint *values,
                             ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetTransformFeedbackVarying_length(const Context *context,
                                               bool isCallValid,
                                               ShaderProgramID program,
                                               GLuint index,
                                               GLsizei bufSize,
                                               GLsizei *length,
                                               GLsizei *size,
                                               GLenum *type,
                                               GLchar *name,
                                               ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetTransformFeedbackVarying_size(const Context *context,
                                             bool isCallValid,
                                             ShaderProgramID program,
                                             GLuint index,
                                             GLsizei bufSize,
                                             GLsizei *length,
                                             GLsizei *size,
                                             GLenum *type,
                                             GLchar *name,
                                             ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetTransformFeedbackVarying_type(const Context *context,
                                             bool isCallValid,
                                             ShaderProgramID program,
                                             GLuint index,
                                             GLsizei bufSize,
                                             GLsizei *length,
                                             GLsizei *size,
                                             GLenum *type,
                                             GLchar *name,
                                             ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetTransformFeedbackVarying_name(const Context *context,
                                             bool isCallValid,
                                             ShaderProgramID program,
                                             GLuint index,
                                             GLsizei bufSize,
                                             GLsizei *length,
                                             GLsizei *size,
                                             GLenum *type,
                                             GLchar *name,
                                             ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetUniformBlockIndex_uniformBlockName(const Context *context,
                                                  bool isCallValid,
                                                  ShaderProgramID program,
                                                  const GLchar *uniformBlockName,
                                                  ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetUniformIndices_uniformNames(const Context *context,
                                           bool isCallValid,
                                           ShaderProgramID program,
                                           GLsizei uniformCount,
                                           const GLchar *const *uniformNames,
                                           GLuint *uniformIndices,
                                           ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetUniformIndices_uniformIndices(const Context *context,
                                             bool isCallValid,
                                             ShaderProgramID program,
                                             GLsizei uniformCount,
                                             const GLchar *const *uniformNames,
                                             GLuint *uniformIndices,
                                             ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetUniformuiv_params(const Context *context,
                                 bool isCallValid,
                                 ShaderProgramID program,
                                 GLint location,
                                 GLuint *params,
                                 ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetVertexAttribIiv_params(const Context *context,
                                      bool isCallValid,
                                      GLuint index,
                                      GLenum pname,
                                      GLint *params,
                                      ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetVertexAttribIuiv_params(const Context *context,
                                       bool isCallValid,
                                       GLuint index,
                                       GLenum pname,
                                       GLuint *params,
                                       ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureInvalidateFramebuffer_attachments(const Context *context,
                                              bool isCallValid,
                                              GLenum target,
                                              GLsizei numAttachments,
                                              const GLenum *attachments,
                                              ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureInvalidateSubFramebuffer_attachments(const Context *context,
                                                 bool isCallValid,
                                                 GLenum target,
                                                 GLsizei numAttachments,
                                                 const GLenum *attachments,
                                                 GLint x,
                                                 GLint y,
                                                 GLsizei width,
                                                 GLsizei height,
                                                 ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureProgramBinary_binary(const Context *context,
                                 bool isCallValid,
                                 ShaderProgramID program,
                                 GLenum binaryFormat,
                                 const void *binary,
                                 GLsizei length,
                                 ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureSamplerParameterfv_param(const Context *context,
                                     bool isCallValid,
                                     SamplerID sampler,
                                     GLenum pname,
                                     const GLfloat *param,
                                     ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureSamplerParameteriv_param(const Context *context,
                                     bool isCallValid,
                                     SamplerID sampler,
                                     GLenum pname,
                                     const GLint *param,
                                     ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureTexImage3D_pixels(const Context *context,
                              bool isCallValid,
                              TextureTarget targetPacked,
                              GLint level,
                              GLint internalformat,
                              GLsizei width,
                              GLsizei height,
                              GLsizei depth,
                              GLint border,
                              GLenum format,
                              GLenum type,
                              const void *pixels,
                              ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureTexSubImage3D_pixels(const Context *context,
                                 bool isCallValid,
                                 TextureTarget targetPacked,
                                 GLint level,
                                 GLint xoffset,
                                 GLint yoffset,
                                 GLint zoffset,
                                 GLsizei width,
                                 GLsizei height,
                                 GLsizei depth,
                                 GLenum format,
                                 GLenum type,
                                 const void *pixels,
                                 ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureTransformFeedbackVaryings_varyings(const Context *context,
                                               bool isCallValid,
                                               ShaderProgramID program,
                                               GLsizei count,
                                               const GLchar *const *varyings,
                                               GLenum bufferMode,
                                               ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureUniform1uiv_value(const Context *context,
                              bool isCallValid,
                              GLint location,
                              GLsizei count,
                              const GLuint *value,
                              ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureUniform2uiv_value(const Context *context,
                              bool isCallValid,
                              GLint location,
                              GLsizei count,
                              const GLuint *value,
                              ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureUniform3uiv_value(const Context *context,
                              bool isCallValid,
                              GLint location,
                              GLsizei count,
                              const GLuint *value,
                              ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureUniform4uiv_value(const Context *context,
                              bool isCallValid,
                              GLint location,
                              GLsizei count,
                              const GLuint *value,
                              ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureUniformMatrix2x3fv_value(const Context *context,
                                     bool isCallValid,
                                     GLint location,
                                     GLsizei count,
                                     GLboolean transpose,
                                     const GLfloat *value,
                                     ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureUniformMatrix2x4fv_value(const Context *context,
                                     bool isCallValid,
                                     GLint location,
                                     GLsizei count,
                                     GLboolean transpose,
                                     const GLfloat *value,
                                     ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureUniformMatrix3x2fv_value(const Context *context,
                                     bool isCallValid,
                                     GLint location,
                                     GLsizei count,
                                     GLboolean transpose,
                                     const GLfloat *value,
                                     ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureUniformMatrix3x4fv_value(const Context *context,
                                     bool isCallValid,
                                     GLint location,
                                     GLsizei count,
                                     GLboolean transpose,
                                     const GLfloat *value,
                                     ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureUniformMatrix4x2fv_value(const Context *context,
                                     bool isCallValid,
                                     GLint location,
                                     GLsizei count,
                                     GLboolean transpose,
                                     const GLfloat *value,
                                     ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureUniformMatrix4x3fv_value(const Context *context,
                                     bool isCallValid,
                                     GLint location,
                                     GLsizei count,
                                     GLboolean transpose,
                                     const GLfloat *value,
                                     ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureVertexAttribI4iv_v(const Context *context,
                               bool isCallValid,
                               GLuint index,
                               const GLint *v,
                               ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureVertexAttribI4uiv_v(const Context *context,
                                bool isCallValid,
                                GLuint index,
                                const GLuint *v,
                                ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureVertexAttribIPointer_pointer(const Context *context,
                                         bool isCallValid,
                                         GLuint index,
                                         GLint size,
                                         VertexAttribType typePacked,
                                         GLsizei stride,
                                         const void *pointer,
                                         ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

}  // namespace gl
