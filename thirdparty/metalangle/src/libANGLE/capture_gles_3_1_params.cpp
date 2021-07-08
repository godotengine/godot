//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// capture_gles31_params.cpp:
//   Pointer parameter capture functions for the OpenGL ES 3.1 entry points.

#include "libANGLE/capture_gles_3_1_autogen.h"

using namespace angle;

namespace gl
{

void CaptureCreateShaderProgramv_strings(const Context *context,
                                         bool isCallValid,
                                         ShaderType typePacked,
                                         GLsizei count,
                                         const GLchar *const *strings,
                                         ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureDeleteProgramPipelines_pipelinesPacked(const Context *context,
                                                   bool isCallValid,
                                                   GLsizei n,
                                                   const ProgramPipelineID *pipelines,
                                                   ParamCapture *paramCapture)
{
    CaptureMemory(pipelines, sizeof(ProgramPipelineID) * n, paramCapture);
}

void CaptureDrawArraysIndirect_indirect(const Context *context,
                                        bool isCallValid,
                                        PrimitiveMode modePacked,
                                        const void *indirect,
                                        ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureDrawElementsIndirect_indirect(const Context *context,
                                          bool isCallValid,
                                          PrimitiveMode modePacked,
                                          DrawElementsType typePacked,
                                          const void *indirect,
                                          ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGenProgramPipelines_pipelinesPacked(const Context *context,
                                                bool isCallValid,
                                                GLsizei n,
                                                ProgramPipelineID *pipelines,
                                                ParamCapture *paramCapture)
{
    CaptureGenHandles(n, pipelines, paramCapture);
}

void CaptureGetBooleani_v_data(const Context *context,
                               bool isCallValid,
                               GLenum target,
                               GLuint index,
                               GLboolean *data,
                               ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetFramebufferParameteriv_params(const Context *context,
                                             bool isCallValid,
                                             GLenum target,
                                             GLenum pname,
                                             GLint *params,
                                             ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetMultisamplefv_val(const Context *context,
                                 bool isCallValid,
                                 GLenum pname,
                                 GLuint index,
                                 GLfloat *val,
                                 ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetProgramInterfaceiv_params(const Context *context,
                                         bool isCallValid,
                                         ShaderProgramID program,
                                         GLenum programInterface,
                                         GLenum pname,
                                         GLint *params,
                                         ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetProgramPipelineInfoLog_length(const Context *context,
                                             bool isCallValid,
                                             ProgramPipelineID pipeline,
                                             GLsizei bufSize,
                                             GLsizei *length,
                                             GLchar *infoLog,
                                             ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetProgramPipelineInfoLog_infoLog(const Context *context,
                                              bool isCallValid,
                                              ProgramPipelineID pipeline,
                                              GLsizei bufSize,
                                              GLsizei *length,
                                              GLchar *infoLog,
                                              ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetProgramPipelineiv_params(const Context *context,
                                        bool isCallValid,
                                        ProgramPipelineID pipeline,
                                        GLenum pname,
                                        GLint *params,
                                        ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetProgramResourceIndex_name(const Context *context,
                                         bool isCallValid,
                                         ShaderProgramID program,
                                         GLenum programInterface,
                                         const GLchar *name,
                                         ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetProgramResourceLocation_name(const Context *context,
                                            bool isCallValid,
                                            ShaderProgramID program,
                                            GLenum programInterface,
                                            const GLchar *name,
                                            ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetProgramResourceName_length(const Context *context,
                                          bool isCallValid,
                                          ShaderProgramID program,
                                          GLenum programInterface,
                                          GLuint index,
                                          GLsizei bufSize,
                                          GLsizei *length,
                                          GLchar *name,
                                          ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetProgramResourceName_name(const Context *context,
                                        bool isCallValid,
                                        ShaderProgramID program,
                                        GLenum programInterface,
                                        GLuint index,
                                        GLsizei bufSize,
                                        GLsizei *length,
                                        GLchar *name,
                                        ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetProgramResourceiv_props(const Context *context,
                                       bool isCallValid,
                                       ShaderProgramID program,
                                       GLenum programInterface,
                                       GLuint index,
                                       GLsizei propCount,
                                       const GLenum *props,
                                       GLsizei bufSize,
                                       GLsizei *length,
                                       GLint *params,
                                       ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetProgramResourceiv_length(const Context *context,
                                        bool isCallValid,
                                        ShaderProgramID program,
                                        GLenum programInterface,
                                        GLuint index,
                                        GLsizei propCount,
                                        const GLenum *props,
                                        GLsizei bufSize,
                                        GLsizei *length,
                                        GLint *params,
                                        ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetProgramResourceiv_params(const Context *context,
                                        bool isCallValid,
                                        ShaderProgramID program,
                                        GLenum programInterface,
                                        GLuint index,
                                        GLsizei propCount,
                                        const GLenum *props,
                                        GLsizei bufSize,
                                        GLsizei *length,
                                        GLint *params,
                                        ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetTexLevelParameterfv_params(const Context *context,
                                          bool isCallValid,
                                          TextureTarget targetPacked,
                                          GLint level,
                                          GLenum pname,
                                          GLfloat *params,
                                          ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetTexLevelParameteriv_params(const Context *context,
                                          bool isCallValid,
                                          TextureTarget targetPacked,
                                          GLint level,
                                          GLenum pname,
                                          GLint *params,
                                          ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureProgramUniform1fv_value(const Context *context,
                                    bool isCallValid,
                                    ShaderProgramID program,
                                    GLint location,
                                    GLsizei count,
                                    const GLfloat *value,
                                    ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureProgramUniform1iv_value(const Context *context,
                                    bool isCallValid,
                                    ShaderProgramID program,
                                    GLint location,
                                    GLsizei count,
                                    const GLint *value,
                                    ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureProgramUniform1uiv_value(const Context *context,
                                     bool isCallValid,
                                     ShaderProgramID program,
                                     GLint location,
                                     GLsizei count,
                                     const GLuint *value,
                                     ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureProgramUniform2fv_value(const Context *context,
                                    bool isCallValid,
                                    ShaderProgramID program,
                                    GLint location,
                                    GLsizei count,
                                    const GLfloat *value,
                                    ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureProgramUniform2iv_value(const Context *context,
                                    bool isCallValid,
                                    ShaderProgramID program,
                                    GLint location,
                                    GLsizei count,
                                    const GLint *value,
                                    ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureProgramUniform2uiv_value(const Context *context,
                                     bool isCallValid,
                                     ShaderProgramID program,
                                     GLint location,
                                     GLsizei count,
                                     const GLuint *value,
                                     ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureProgramUniform3fv_value(const Context *context,
                                    bool isCallValid,
                                    ShaderProgramID program,
                                    GLint location,
                                    GLsizei count,
                                    const GLfloat *value,
                                    ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureProgramUniform3iv_value(const Context *context,
                                    bool isCallValid,
                                    ShaderProgramID program,
                                    GLint location,
                                    GLsizei count,
                                    const GLint *value,
                                    ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureProgramUniform3uiv_value(const Context *context,
                                     bool isCallValid,
                                     ShaderProgramID program,
                                     GLint location,
                                     GLsizei count,
                                     const GLuint *value,
                                     ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureProgramUniform4fv_value(const Context *context,
                                    bool isCallValid,
                                    ShaderProgramID program,
                                    GLint location,
                                    GLsizei count,
                                    const GLfloat *value,
                                    ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureProgramUniform4iv_value(const Context *context,
                                    bool isCallValid,
                                    ShaderProgramID program,
                                    GLint location,
                                    GLsizei count,
                                    const GLint *value,
                                    ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureProgramUniform4uiv_value(const Context *context,
                                     bool isCallValid,
                                     ShaderProgramID program,
                                     GLint location,
                                     GLsizei count,
                                     const GLuint *value,
                                     ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureProgramUniformMatrix2fv_value(const Context *context,
                                          bool isCallValid,
                                          ShaderProgramID program,
                                          GLint location,
                                          GLsizei count,
                                          GLboolean transpose,
                                          const GLfloat *value,
                                          ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureProgramUniformMatrix2x3fv_value(const Context *context,
                                            bool isCallValid,
                                            ShaderProgramID program,
                                            GLint location,
                                            GLsizei count,
                                            GLboolean transpose,
                                            const GLfloat *value,
                                            ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureProgramUniformMatrix2x4fv_value(const Context *context,
                                            bool isCallValid,
                                            ShaderProgramID program,
                                            GLint location,
                                            GLsizei count,
                                            GLboolean transpose,
                                            const GLfloat *value,
                                            ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureProgramUniformMatrix3fv_value(const Context *context,
                                          bool isCallValid,
                                          ShaderProgramID program,
                                          GLint location,
                                          GLsizei count,
                                          GLboolean transpose,
                                          const GLfloat *value,
                                          ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureProgramUniformMatrix3x2fv_value(const Context *context,
                                            bool isCallValid,
                                            ShaderProgramID program,
                                            GLint location,
                                            GLsizei count,
                                            GLboolean transpose,
                                            const GLfloat *value,
                                            ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureProgramUniformMatrix3x4fv_value(const Context *context,
                                            bool isCallValid,
                                            ShaderProgramID program,
                                            GLint location,
                                            GLsizei count,
                                            GLboolean transpose,
                                            const GLfloat *value,
                                            ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureProgramUniformMatrix4fv_value(const Context *context,
                                          bool isCallValid,
                                          ShaderProgramID program,
                                          GLint location,
                                          GLsizei count,
                                          GLboolean transpose,
                                          const GLfloat *value,
                                          ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureProgramUniformMatrix4x2fv_value(const Context *context,
                                            bool isCallValid,
                                            ShaderProgramID program,
                                            GLint location,
                                            GLsizei count,
                                            GLboolean transpose,
                                            const GLfloat *value,
                                            ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureProgramUniformMatrix4x3fv_value(const Context *context,
                                            bool isCallValid,
                                            ShaderProgramID program,
                                            GLint location,
                                            GLsizei count,
                                            GLboolean transpose,
                                            const GLfloat *value,
                                            ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

}  // namespace gl
