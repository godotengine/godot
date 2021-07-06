//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// GLImplFactory.h:
//   Factory interface for OpenGL ES Impl objects.
//

#ifndef LIBANGLE_RENDERER_GLIMPLFACTORY_H_
#define LIBANGLE_RENDERER_GLIMPLFACTORY_H_

#include <vector>

#include "angle_gl.h"
#include "libANGLE/Framebuffer.h"
#include "libANGLE/Overlay.h"
#include "libANGLE/Program.h"
#include "libANGLE/ProgramPipeline.h"
#include "libANGLE/Renderbuffer.h"
#include "libANGLE/Shader.h"
#include "libANGLE/Texture.h"
#include "libANGLE/TransformFeedback.h"
#include "libANGLE/VertexArray.h"

namespace gl
{
class State;
}  // namespace gl

namespace rx
{
class BufferImpl;
class CompilerImpl;
class ContextImpl;
class FenceNVImpl;
class SyncImpl;
class FramebufferImpl;
class MemoryObjectImpl;
class OverlayImpl;
class PathImpl;
class ProgramImpl;
class ProgramPipelineImpl;
class QueryImpl;
class RenderbufferImpl;
class SamplerImpl;
class SemaphoreImpl;
class ShaderImpl;
class TextureImpl;
class TransformFeedbackImpl;
class VertexArrayImpl;

class GLImplFactory : angle::NonCopyable
{
  public:
    GLImplFactory() {}
    virtual ~GLImplFactory() {}

    // Shader creation
    virtual CompilerImpl *createCompiler()                           = 0;
    virtual ShaderImpl *createShader(const gl::ShaderState &data)    = 0;
    virtual ProgramImpl *createProgram(const gl::ProgramState &data) = 0;

    // Framebuffer creation
    virtual FramebufferImpl *createFramebuffer(const gl::FramebufferState &data) = 0;

    // Texture creation
    virtual TextureImpl *createTexture(const gl::TextureState &state) = 0;

    // Renderbuffer creation
    virtual RenderbufferImpl *createRenderbuffer(const gl::RenderbufferState &state) = 0;

    // Buffer creation
    virtual BufferImpl *createBuffer(const gl::BufferState &state) = 0;

    // Vertex Array creation
    virtual VertexArrayImpl *createVertexArray(const gl::VertexArrayState &data) = 0;

    // Query and Fence creation
    virtual QueryImpl *createQuery(gl::QueryType type) = 0;
    virtual FenceNVImpl *createFenceNV()               = 0;
    virtual SyncImpl *createSync()                     = 0;

    // Transform Feedback creation
    virtual TransformFeedbackImpl *createTransformFeedback(
        const gl::TransformFeedbackState &state) = 0;

    // Sampler object creation
    virtual SamplerImpl *createSampler(const gl::SamplerState &state) = 0;

    // Program Pipeline object creation
    virtual ProgramPipelineImpl *createProgramPipeline(const gl::ProgramPipelineState &data) = 0;

    virtual std::vector<PathImpl *> createPaths(GLsizei range) = 0;

    // Memory object creation
    virtual MemoryObjectImpl *createMemoryObject() = 0;

    // Semaphore creation
    virtual SemaphoreImpl *createSemaphore() = 0;

    // Overlay creation
    virtual OverlayImpl *createOverlay(const gl::OverlayState &state) = 0;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GLIMPLFACTORY_H_
