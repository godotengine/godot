//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// ShaderGL.h: Defines the class interface for ShaderGL.

#ifndef LIBANGLE_RENDERER_GL_SHADERGL_H_
#define LIBANGLE_RENDERER_GL_SHADERGL_H_

#include "libANGLE/renderer/ShaderImpl.h"

namespace rx
{
class RendererGL;
enum class MultiviewImplementationTypeGL;

class ShaderGL : public ShaderImpl
{
  public:
    ShaderGL(const gl::ShaderState &data,
             GLuint shaderID,
             MultiviewImplementationTypeGL multiviewImplementationType,
             const std::shared_ptr<RendererGL> &renderer);
    ~ShaderGL() override;

    void destroy() override;

    std::shared_ptr<WaitableCompileEvent> compile(const gl::Context *context,
                                                  gl::ShCompilerInstance *compilerInstance,
                                                  ShCompileOptions options) override;

    std::string getDebugInfo() const override;

    GLuint getShaderID() const;

  private:
    void compileAndCheckShader(const char *source);
    void compileShader(const char *source);
    void checkShader();
    bool peekCompletion();
    bool compileAndCheckShaderInWorker(const char *source);

    GLuint mShaderID;
    MultiviewImplementationTypeGL mMultiviewImplementationType;
    std::shared_ptr<RendererGL> mRenderer;
    GLint mCompileStatus;
    std::string mInfoLog;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_SHADERGL_H_
