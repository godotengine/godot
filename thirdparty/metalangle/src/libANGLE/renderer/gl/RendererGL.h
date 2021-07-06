//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// RendererGL.h: Defines the class interface for RendererGL.

#ifndef LIBANGLE_RENDERER_GL_RENDERERGL_H_
#define LIBANGLE_RENDERER_GL_RENDERERGL_H_

#include <list>
#include <mutex>
#include <thread>

#include "libANGLE/Caps.h"
#include "libANGLE/Error.h"
#include "libANGLE/Version.h"
#include "libANGLE/renderer/gl/renderergl_utils.h"
#include "platform/FeaturesGL.h"

namespace angle
{
struct FrontendFeatures;
}  // namespace angle

namespace gl
{
struct IndexRange;
class Path;
class State;
}  // namespace gl

namespace egl
{
class AttributeMap;
}  // namespace egl

namespace sh
{
struct BlockMemberInfo;
}  // namespace sh

namespace rx
{
class BlitGL;
class ClearMultiviewGL;
class ContextImpl;
class DisplayGL;
class FunctionsGL;
class RendererGL;
class StateManagerGL;

// WorkerContext wraps a native GL context shared from the main context. It is used by the workers
// for khr_parallel_shader_compile.
class WorkerContext : angle::NonCopyable
{
  public:
    virtual ~WorkerContext() {}

    virtual bool makeCurrent()   = 0;
    virtual void unmakeCurrent() = 0;
};

class ScopedWorkerContextGL
{
  public:
    ScopedWorkerContextGL(RendererGL *renderer, std::string *infoLog);
    ~ScopedWorkerContextGL();

    bool operator()() const;

  private:
    RendererGL *mRenderer = nullptr;
    bool mValid           = false;
};

class RendererGL : angle::NonCopyable
{
  public:
    RendererGL(std::unique_ptr<FunctionsGL> functions,
               const egl::AttributeMap &attribMap,
               DisplayGL *display);
    virtual ~RendererGL();

    angle::Result flush();
    angle::Result finish();

    // CHROMIUM_path_rendering implementation
    void stencilFillPath(const gl::State &state,
                         const gl::Path *path,
                         GLenum fillMode,
                         GLuint mask);
    void stencilStrokePath(const gl::State &state,
                           const gl::Path *path,
                           GLint reference,
                           GLuint mask);
    void coverFillPath(const gl::State &state, const gl::Path *path, GLenum coverMode);
    void coverStrokePath(const gl::State &state, const gl::Path *path, GLenum coverMode);
    void stencilThenCoverFillPath(const gl::State &state,
                                  const gl::Path *path,
                                  GLenum fillMode,
                                  GLuint mask,
                                  GLenum coverMode);
    void stencilThenCoverStrokePath(const gl::State &state,
                                    const gl::Path *path,
                                    GLint reference,
                                    GLuint mask,
                                    GLenum coverMode);
    void coverFillPathInstanced(const gl::State &state,
                                const std::vector<gl::Path *> &paths,
                                GLenum coverMode,
                                GLenum transformType,
                                const GLfloat *transformValues);
    void coverStrokePathInstanced(const gl::State &state,
                                  const std::vector<gl::Path *> &paths,
                                  GLenum coverMode,
                                  GLenum transformType,
                                  const GLfloat *transformValues);
    void stencilFillPathInstanced(const gl::State &state,
                                  const std::vector<gl::Path *> &paths,
                                  GLenum fillMode,
                                  GLuint mask,
                                  GLenum transformType,
                                  const GLfloat *transformValues);
    void stencilStrokePathInstanced(const gl::State &state,
                                    const std::vector<gl::Path *> &paths,
                                    GLint reference,
                                    GLuint mask,
                                    GLenum transformType,
                                    const GLfloat *transformValues);

    void stencilThenCoverFillPathInstanced(const gl::State &state,
                                           const std::vector<gl::Path *> &paths,
                                           GLenum coverMode,
                                           GLenum fillMode,
                                           GLuint mask,
                                           GLenum transformType,
                                           const GLfloat *transformValues);
    void stencilThenCoverStrokePathInstanced(const gl::State &state,
                                             const std::vector<gl::Path *> &paths,
                                             GLenum coverMode,
                                             GLint reference,
                                             GLuint mask,
                                             GLenum transformType,
                                             const GLfloat *transformValues);

    gl::GraphicsResetStatus getResetStatus();

    // EXT_debug_marker
    void insertEventMarker(GLsizei length, const char *marker);
    void pushGroupMarker(GLsizei length, const char *marker);
    void popGroupMarker();

    // KHR_debug
    void pushDebugGroup(GLenum source, GLuint id, const std::string &message);
    void popDebugGroup();

    std::string getVendorString() const;
    std::string getRendererDescription() const;

    GLint getGPUDisjoint();
    GLint64 getTimestamp();

    const gl::Version &getMaxSupportedESVersion() const;
    const FunctionsGL *getFunctions() const { return mFunctions.get(); }
    StateManagerGL *getStateManager() const { return mStateManager; }
    const angle::FeaturesGL &getFeatures() const { return mFeatures; }
    BlitGL *getBlitter() const { return mBlitter; }
    ClearMultiviewGL *getMultiviewClearer() const { return mMultiviewClearer; }

    MultiviewImplementationTypeGL getMultiviewImplementationType() const;
    const gl::Caps &getNativeCaps() const;
    const gl::TextureCapsMap &getNativeTextureCaps() const;
    const gl::Extensions &getNativeExtensions() const;
    const gl::Limitations &getNativeLimitations() const;
    void initializeFrontendFeatures(angle::FrontendFeatures *features) const;

    angle::Result dispatchCompute(const gl::Context *context,
                                  GLuint numGroupsX,
                                  GLuint numGroupsY,
                                  GLuint numGroupsZ);
    angle::Result dispatchComputeIndirect(const gl::Context *context, GLintptr indirect);

    angle::Result memoryBarrier(GLbitfield barriers);
    angle::Result memoryBarrierByRegion(GLbitfield barriers);

    bool bindWorkerContext(std::string *infoLog);
    void unbindWorkerContext();
    // Checks if the driver has the KHR_parallel_shader_compile or ARB_parallel_shader_compile
    // extension.
    bool hasNativeParallelCompile();
    void setMaxShaderCompilerThreads(GLuint count);

    static unsigned int getMaxWorkerContexts();

  protected:
    virtual WorkerContext *createWorkerContext(std::string *infoLog) = 0;

  private:
    void ensureCapsInitialized() const;
    void generateCaps(gl::Caps *outCaps,
                      gl::TextureCapsMap *outTextureCaps,
                      gl::Extensions *outExtensions,
                      gl::Limitations *outLimitations) const;

    mutable gl::Version mMaxSupportedESVersion;

    std::unique_ptr<FunctionsGL> mFunctions;
    StateManagerGL *mStateManager;

    BlitGL *mBlitter;
    ClearMultiviewGL *mMultiviewClearer;

    bool mUseDebugOutput;

    mutable bool mCapsInitialized;
    mutable gl::Caps mNativeCaps;
    mutable gl::TextureCapsMap mNativeTextureCaps;
    mutable gl::Extensions mNativeExtensions;
    mutable gl::Limitations mNativeLimitations;
    mutable MultiviewImplementationTypeGL mMultiviewImplementationType;

    // The thread-to-context mapping for the currently active worker threads.
    std::unordered_map<std::thread::id, std::unique_ptr<WorkerContext>> mCurrentWorkerContexts;
    // The worker contexts available to use.
    std::list<std::unique_ptr<WorkerContext>> mWorkerContextPool;
    // Protect the concurrent accesses to worker contexts.
    std::mutex mWorkerMutex;

    bool mNativeParallelCompileEnabled;

    angle::FeaturesGL mFeatures;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_RENDERERGL_H_
