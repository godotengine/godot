//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// capture_gles1_params.cpp:
//   Pointer parameter capture functions for the OpenGL ES 1.0 entry points.

#include "libANGLE/capture_gles_1_0_autogen.h"

using namespace angle;

namespace gl
{

void CaptureClipPlanef_eqn(const Context *context,
                           bool isCallValid,
                           GLenum p,
                           const GLfloat *eqn,
                           ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureClipPlanex_equation(const Context *context,
                                bool isCallValid,
                                GLenum plane,
                                const GLfixed *equation,
                                ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureColorPointer_pointer(const Context *context,
                                 bool isCallValid,
                                 GLint size,
                                 VertexAttribType typePacked,
                                 GLsizei stride,
                                 const void *pointer,
                                 ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureFogfv_params(const Context *context,
                         bool isCallValid,
                         GLenum pname,
                         const GLfloat *params,
                         ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureFogxv_param(const Context *context,
                        bool isCallValid,
                        GLenum pname,
                        const GLfixed *param,
                        ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetClipPlanef_equation(const Context *context,
                                   bool isCallValid,
                                   GLenum plane,
                                   GLfloat *equation,
                                   ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetClipPlanex_equation(const Context *context,
                                   bool isCallValid,
                                   GLenum plane,
                                   GLfixed *equation,
                                   ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetFixedv_params(const Context *context,
                             bool isCallValid,
                             GLenum pname,
                             GLfixed *params,
                             ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetLightfv_params(const Context *context,
                              bool isCallValid,
                              GLenum light,
                              LightParameter pnamePacked,
                              GLfloat *params,
                              ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetLightxv_params(const Context *context,
                              bool isCallValid,
                              GLenum light,
                              LightParameter pnamePacked,
                              GLfixed *params,
                              ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetMaterialfv_params(const Context *context,
                                 bool isCallValid,
                                 GLenum face,
                                 MaterialParameter pnamePacked,
                                 GLfloat *params,
                                 ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetMaterialxv_params(const Context *context,
                                 bool isCallValid,
                                 GLenum face,
                                 MaterialParameter pnamePacked,
                                 GLfixed *params,
                                 ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetPointerv_params(const Context *context,
                               bool isCallValid,
                               GLenum pname,
                               void **params,
                               ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetTexEnvfv_params(const Context *context,
                               bool isCallValid,
                               TextureEnvTarget targetPacked,
                               TextureEnvParameter pnamePacked,
                               GLfloat *params,
                               ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetTexEnviv_params(const Context *context,
                               bool isCallValid,
                               TextureEnvTarget targetPacked,
                               TextureEnvParameter pnamePacked,
                               GLint *params,
                               ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetTexEnvxv_params(const Context *context,
                               bool isCallValid,
                               TextureEnvTarget targetPacked,
                               TextureEnvParameter pnamePacked,
                               GLfixed *params,
                               ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureGetTexParameterxv_params(const Context *context,
                                     bool isCallValid,
                                     TextureType targetPacked,
                                     GLenum pname,
                                     GLfixed *params,
                                     ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureLightModelfv_params(const Context *context,
                                bool isCallValid,
                                GLenum pname,
                                const GLfloat *params,
                                ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureLightModelxv_param(const Context *context,
                               bool isCallValid,
                               GLenum pname,
                               const GLfixed *param,
                               ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureLightfv_params(const Context *context,
                           bool isCallValid,
                           GLenum light,
                           LightParameter pnamePacked,
                           const GLfloat *params,
                           ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureLightxv_params(const Context *context,
                           bool isCallValid,
                           GLenum light,
                           LightParameter pnamePacked,
                           const GLfixed *params,
                           ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureLoadMatrixf_m(const Context *context,
                          bool isCallValid,
                          const GLfloat *m,
                          ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureLoadMatrixx_m(const Context *context,
                          bool isCallValid,
                          const GLfixed *m,
                          ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureMaterialfv_params(const Context *context,
                              bool isCallValid,
                              GLenum face,
                              MaterialParameter pnamePacked,
                              const GLfloat *params,
                              ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureMaterialxv_param(const Context *context,
                             bool isCallValid,
                             GLenum face,
                             MaterialParameter pnamePacked,
                             const GLfixed *param,
                             ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureMultMatrixf_m(const Context *context,
                          bool isCallValid,
                          const GLfloat *m,
                          ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureMultMatrixx_m(const Context *context,
                          bool isCallValid,
                          const GLfixed *m,
                          ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureNormalPointer_pointer(const Context *context,
                                  bool isCallValid,
                                  VertexAttribType typePacked,
                                  GLsizei stride,
                                  const void *pointer,
                                  ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CapturePointParameterfv_params(const Context *context,
                                    bool isCallValid,
                                    PointParameter pnamePacked,
                                    const GLfloat *params,
                                    ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CapturePointParameterxv_params(const Context *context,
                                    bool isCallValid,
                                    PointParameter pnamePacked,
                                    const GLfixed *params,
                                    ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureTexCoordPointer_pointer(const Context *context,
                                    bool isCallValid,
                                    GLint size,
                                    VertexAttribType typePacked,
                                    GLsizei stride,
                                    const void *pointer,
                                    ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureTexEnvfv_params(const Context *context,
                            bool isCallValid,
                            TextureEnvTarget targetPacked,
                            TextureEnvParameter pnamePacked,
                            const GLfloat *params,
                            ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureTexEnviv_params(const Context *context,
                            bool isCallValid,
                            TextureEnvTarget targetPacked,
                            TextureEnvParameter pnamePacked,
                            const GLint *params,
                            ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureTexEnvxv_params(const Context *context,
                            bool isCallValid,
                            TextureEnvTarget targetPacked,
                            TextureEnvParameter pnamePacked,
                            const GLfixed *params,
                            ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureTexParameterxv_params(const Context *context,
                                  bool isCallValid,
                                  TextureType targetPacked,
                                  GLenum pname,
                                  const GLfixed *params,
                                  ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

void CaptureVertexPointer_pointer(const Context *context,
                                  bool isCallValid,
                                  GLint size,
                                  VertexAttribType typePacked,
                                  GLsizei stride,
                                  const void *pointer,
                                  ParamCapture *paramCapture)
{
    UNIMPLEMENTED();
}

}  // namespace gl
