//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ContextImpl:
//   Implementation-specific functionality associated with a GL Context.
//

#ifndef LIBANGLE_RENDERER_CONTEXTIMPL_H_
#define LIBANGLE_RENDERER_CONTEXTIMPL_H_

#include <vector>

#include "common/angleutils.h"
#include "libANGLE/State.h"
#include "libANGLE/renderer/GLImplFactory.h"

namespace gl
{
class ErrorSet;
class MemoryProgramCache;
class Path;
class Semaphore;
struct Workarounds;
}  // namespace gl

namespace rx
{
class ContextImpl : public GLImplFactory
{
  public:
    ContextImpl(const gl::State &state, gl::ErrorSet *errorSet);
    ~ContextImpl() override;

    virtual void onDestroy(const gl::Context *context) {}

    virtual angle::Result initialize() = 0;

    // Flush and finish.
    virtual angle::Result flush(const gl::Context *context)  = 0;
    virtual angle::Result finish(const gl::Context *context) = 0;

    // Drawing methods.
    virtual angle::Result drawArrays(const gl::Context *context,
                                     gl::PrimitiveMode mode,
                                     GLint first,
                                     GLsizei count)                  = 0;
    virtual angle::Result drawArraysInstanced(const gl::Context *context,
                                              gl::PrimitiveMode mode,
                                              GLint first,
                                              GLsizei count,
                                              GLsizei instanceCount) = 0;
    // Necessary for Vulkan since gl_InstanceIndex includes baseInstance
    virtual angle::Result drawArraysInstancedBaseInstance(const gl::Context *context,
                                                          gl::PrimitiveMode mode,
                                                          GLint first,
                                                          GLsizei count,
                                                          GLsizei instanceCount,
                                                          GLuint baseInstance) = 0;

    virtual angle::Result drawElements(const gl::Context *context,
                                       gl::PrimitiveMode mode,
                                       GLsizei count,
                                       gl::DrawElementsType type,
                                       const void *indices)                                = 0;
    virtual angle::Result drawElementsInstanced(const gl::Context *context,
                                                gl::PrimitiveMode mode,
                                                GLsizei count,
                                                gl::DrawElementsType type,
                                                const void *indices,
                                                GLsizei instances)                         = 0;
    virtual angle::Result drawElementsInstancedBaseVertexBaseInstance(const gl::Context *context,
                                                                      gl::PrimitiveMode mode,
                                                                      GLsizei count,
                                                                      gl::DrawElementsType type,
                                                                      const void *indices,
                                                                      GLsizei instances,
                                                                      GLint baseVertex,
                                                                      GLuint baseInstance) = 0;
    virtual angle::Result drawRangeElements(const gl::Context *context,
                                            gl::PrimitiveMode mode,
                                            GLuint start,
                                            GLuint end,
                                            GLsizei count,
                                            gl::DrawElementsType type,
                                            const void *indices)                           = 0;

    virtual angle::Result drawArraysIndirect(const gl::Context *context,
                                             gl::PrimitiveMode mode,
                                             const void *indirect)   = 0;
    virtual angle::Result drawElementsIndirect(const gl::Context *context,
                                               gl::PrimitiveMode mode,
                                               gl::DrawElementsType type,
                                               const void *indirect) = 0;

    // CHROMIUM_path_rendering path drawing methods.
    virtual void stencilFillPath(const gl::Path *path, GLenum fillMode, GLuint mask);
    virtual void stencilStrokePath(const gl::Path *path, GLint reference, GLuint mask);
    virtual void coverFillPath(const gl::Path *path, GLenum coverMode);
    virtual void coverStrokePath(const gl::Path *path, GLenum coverMode);
    virtual void stencilThenCoverFillPath(const gl::Path *path,
                                          GLenum fillMode,
                                          GLuint mask,
                                          GLenum coverMode);

    virtual void stencilThenCoverStrokePath(const gl::Path *path,
                                            GLint reference,
                                            GLuint mask,
                                            GLenum coverMode);

    virtual void coverFillPathInstanced(const std::vector<gl::Path *> &paths,
                                        GLenum coverMode,
                                        GLenum transformType,
                                        const GLfloat *transformValues);
    virtual void coverStrokePathInstanced(const std::vector<gl::Path *> &paths,
                                          GLenum coverMode,
                                          GLenum transformType,
                                          const GLfloat *transformValues);
    virtual void stencilFillPathInstanced(const std::vector<gl::Path *> &paths,
                                          GLenum fillMode,
                                          GLuint mask,
                                          GLenum transformType,
                                          const GLfloat *transformValues);
    virtual void stencilStrokePathInstanced(const std::vector<gl::Path *> &paths,
                                            GLint reference,
                                            GLuint mask,
                                            GLenum transformType,
                                            const GLfloat *transformValues);
    virtual void stencilThenCoverFillPathInstanced(const std::vector<gl::Path *> &paths,
                                                   GLenum coverMode,
                                                   GLenum fillMode,
                                                   GLuint mask,
                                                   GLenum transformType,
                                                   const GLfloat *transformValues);
    virtual void stencilThenCoverStrokePathInstanced(const std::vector<gl::Path *> &paths,
                                                     GLenum coverMode,
                                                     GLint reference,
                                                     GLuint mask,
                                                     GLenum transformType,
                                                     const GLfloat *transformValues);

    // Device loss
    virtual gl::GraphicsResetStatus getResetStatus() = 0;

    // Vendor and description strings.
    virtual std::string getVendorString() const        = 0;
    virtual std::string getRendererDescription() const = 0;

    // EXT_debug_marker
    virtual void insertEventMarker(GLsizei length, const char *marker) = 0;
    virtual void pushGroupMarker(GLsizei length, const char *marker)   = 0;
    virtual void popGroupMarker()                                      = 0;

    // KHR_debug
    virtual void pushDebugGroup(GLenum source, GLuint id, const std::string &message) = 0;
    virtual void popDebugGroup()                                                      = 0;

    // KHR_parallel_shader_compile
    virtual void setMaxShaderCompilerThreads(GLuint count) {}

    // GL_ANGLE_texture_storage_external
    virtual void invalidateTexture(gl::TextureType target);

    // State sync with dirty bits.
    virtual angle::Result syncState(const gl::Context *context,
                                    const gl::State::DirtyBits &dirtyBits,
                                    const gl::State::DirtyBits &bitMask) = 0;

    // Disjoint timer queries
    virtual GLint getGPUDisjoint() = 0;
    virtual GLint64 getTimestamp() = 0;

    // Context switching
    virtual angle::Result onMakeCurrent(const gl::Context *context) = 0;
    virtual angle::Result onUnMakeCurrent(const gl::Context *context);

    // Native capabilities, unmodified by gl::Context.
    virtual gl::Caps getNativeCaps() const                         = 0;
    virtual const gl::TextureCapsMap &getNativeTextureCaps() const = 0;
    virtual const gl::Extensions &getNativeExtensions() const      = 0;
    virtual const gl::Limitations &getNativeLimitations() const    = 0;

    virtual angle::Result dispatchCompute(const gl::Context *context,
                                          GLuint numGroupsX,
                                          GLuint numGroupsY,
                                          GLuint numGroupsZ)         = 0;
    virtual angle::Result dispatchComputeIndirect(const gl::Context *context,
                                                  GLintptr indirect) = 0;

    virtual angle::Result memoryBarrier(const gl::Context *context, GLbitfield barriers) = 0;
    virtual angle::Result memoryBarrierByRegion(const gl::Context *context,
                                                GLbitfield barriers)                     = 0;

    const gl::State &getState() const { return mState; }
    int getClientMajorVersion() const { return mState.getClientMajorVersion(); }
    int getClientMinorVersion() const { return mState.getClientMinorVersion(); }
    const gl::Caps &getCaps() const { return mState.getCaps(); }
    const gl::TextureCapsMap &getTextureCaps() const { return mState.getTextureCaps(); }
    const gl::Extensions &getExtensions() const { return mState.getExtensions(); }
    const gl::Limitations &getLimitations() const { return mState.getLimitations(); }

    // A common GL driver behaviour is to trigger dynamic shader recompilation on a draw call,
    // based on the current render states. We store a mutable pointer to the program cache so
    // on draw calls we can store the refreshed shaders in the cache.
    void setMemoryProgramCache(gl::MemoryProgramCache *memoryProgramCache);

    void handleError(GLenum errorCode,
                     const char *message,
                     const char *file,
                     const char *function,
                     unsigned int line);

  protected:
    const gl::State &mState;
    gl::MemoryProgramCache *mMemoryProgramCache;
    gl::ErrorSet *mErrors;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_CONTEXTIMPL_H_
