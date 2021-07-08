//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// RendererGL.cpp: Implements the class methods for RendererGL.

#include "libANGLE/renderer/gl/RendererGL.h"

#include <EGL/eglext.h>

#include "common/debug.h"
#include "libANGLE/AttributeMap.h"
#include "libANGLE/Context.h"
#include "libANGLE/Display.h"
#include "libANGLE/Path.h"
#include "libANGLE/State.h"
#include "libANGLE/Surface.h"
#include "libANGLE/renderer/gl/BlitGL.h"
#include "libANGLE/renderer/gl/BufferGL.h"
#include "libANGLE/renderer/gl/ClearMultiviewGL.h"
#include "libANGLE/renderer/gl/CompilerGL.h"
#include "libANGLE/renderer/gl/ContextGL.h"
#include "libANGLE/renderer/gl/DisplayGL.h"
#include "libANGLE/renderer/gl/FenceNVGL.h"
#include "libANGLE/renderer/gl/FramebufferGL.h"
#include "libANGLE/renderer/gl/FunctionsGL.h"
#include "libANGLE/renderer/gl/PathGL.h"
#include "libANGLE/renderer/gl/ProgramGL.h"
#include "libANGLE/renderer/gl/QueryGL.h"
#include "libANGLE/renderer/gl/RenderbufferGL.h"
#include "libANGLE/renderer/gl/SamplerGL.h"
#include "libANGLE/renderer/gl/ShaderGL.h"
#include "libANGLE/renderer/gl/StateManagerGL.h"
#include "libANGLE/renderer/gl/SurfaceGL.h"
#include "libANGLE/renderer/gl/SyncGL.h"
#include "libANGLE/renderer/gl/TextureGL.h"
#include "libANGLE/renderer/gl/TransformFeedbackGL.h"
#include "libANGLE/renderer/gl/VertexArrayGL.h"
#include "libANGLE/renderer/gl/renderergl_utils.h"
#include "libANGLE/renderer/renderer_utils.h"

namespace
{

std::vector<GLuint> GatherPaths(const std::vector<gl::Path *> &paths)
{
    std::vector<GLuint> ret;
    ret.reserve(paths.size());

    for (const auto *p : paths)
    {
        const auto *pathObj = rx::GetImplAs<rx::PathGL>(p);
        ret.push_back(pathObj->getPathID());
    }
    return ret;
}

void SetMaxShaderCompilerThreads(const rx::FunctionsGL *functions, GLuint count)
{
    if (functions->maxShaderCompilerThreadsKHR != nullptr)
    {
        functions->maxShaderCompilerThreadsKHR(count);
    }
    else
    {
        ASSERT(functions->maxShaderCompilerThreadsARB != nullptr);
        functions->maxShaderCompilerThreadsARB(count);
    }
}

#if defined(ANGLE_PLATFORM_ANDROID)
const char *kIgnoredErrors[] = {
    // Wrong error message on Android Q Pixel 2. http://anglebug.com/3491
    "FreeAllocationOnTimestamp - Reference to buffer created from "
    "different context without a share list. Application failed to pass "
    "share_context to eglCreateContext. Results are undefined.",
};
#endif  // defined(ANGLE_PLATFORM_ANDROID)
}  // namespace

static void INTERNAL_GL_APIENTRY LogGLDebugMessage(GLenum source,
                                                   GLenum type,
                                                   GLuint id,
                                                   GLenum severity,
                                                   GLsizei length,
                                                   const GLchar *message,
                                                   const void *userParam)
{
    std::string sourceText;
    switch (source)
    {
        case GL_DEBUG_SOURCE_API:
            sourceText = "OpenGL";
            break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
            sourceText = "Windows";
            break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER:
            sourceText = "Shader Compiler";
            break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:
            sourceText = "Third Party";
            break;
        case GL_DEBUG_SOURCE_APPLICATION:
            sourceText = "Application";
            break;
        case GL_DEBUG_SOURCE_OTHER:
            sourceText = "Other";
            break;
        default:
            sourceText = "UNKNOWN";
            break;
    }

    std::string typeText;
    switch (type)
    {
        case GL_DEBUG_TYPE_ERROR:
            typeText = "Error";
            break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
            typeText = "Deprecated behavior";
            break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
            typeText = "Undefined behavior";
            break;
        case GL_DEBUG_TYPE_PORTABILITY:
            typeText = "Portability";
            break;
        case GL_DEBUG_TYPE_PERFORMANCE:
            typeText = "Performance";
            break;
        case GL_DEBUG_TYPE_OTHER:
            typeText = "Other";
            break;
        case GL_DEBUG_TYPE_MARKER:
            typeText = "Marker";
            break;
        default:
            typeText = "UNKNOWN";
            break;
    }

    std::string severityText;
    switch (severity)
    {
        case GL_DEBUG_SEVERITY_HIGH:
            severityText = "High";
            break;
        case GL_DEBUG_SEVERITY_MEDIUM:
            severityText = "Medium";
            break;
        case GL_DEBUG_SEVERITY_LOW:
            severityText = "Low";
            break;
        case GL_DEBUG_SEVERITY_NOTIFICATION:
            severityText = "Notification";
            break;
        default:
            severityText = "UNKNOWN";
            break;
    }

#if defined(ANGLE_PLATFORM_ANDROID)
    if (type == GL_DEBUG_TYPE_ERROR)
    {
        for (const char *&err : kIgnoredErrors)
        {
            if (strncmp(err, message, length) == 0)
            {
                // There is only one ignored message right now and it is quite spammy, around 3MB
                // for a complete end2end tests run, so don't print it even as a warning.
                return;
            }
        }
    }
#endif  // defined(ANGLE_PLATFORM_ANDROID)

    if (type == GL_DEBUG_TYPE_ERROR)
    {
        ERR() << std::endl
              << "\tSource: " << sourceText << std::endl
              << "\tType: " << typeText << std::endl
              << "\tID: " << gl::FmtHex(id) << std::endl
              << "\tSeverity: " << severityText << std::endl
              << "\tMessage: " << message;
    }
    else if (type != GL_DEBUG_TYPE_PERFORMANCE)
    {
        // Don't print performance warnings. They tend to be very spammy in the dEQP test suite and
        // there is very little we can do about them.

        // TODO(ynovikov): filter into WARN and INFO if INFO is ever implemented
        WARN() << std::endl
               << "\tSource: " << sourceText << std::endl
               << "\tType: " << typeText << std::endl
               << "\tID: " << gl::FmtHex(id) << std::endl
               << "\tSeverity: " << severityText << std::endl
               << "\tMessage: " << message;
    }
}

namespace rx
{

RendererGL::RendererGL(std::unique_ptr<FunctionsGL> functions,
                       const egl::AttributeMap &attribMap,
                       DisplayGL *display)
    : mMaxSupportedESVersion(0, 0),
      mFunctions(std::move(functions)),
      mStateManager(nullptr),
      mBlitter(nullptr),
      mMultiviewClearer(nullptr),
      mUseDebugOutput(false),
      mCapsInitialized(false),
      mMultiviewImplementationType(MultiviewImplementationTypeGL::UNSPECIFIED),
      mNativeParallelCompileEnabled(false)
{
    ASSERT(mFunctions);
    nativegl_gl::InitializeFeatures(mFunctions.get(), &mFeatures);
    OverrideFeaturesWithDisplayState(&mFeatures, display->getState());
    mStateManager =
        new StateManagerGL(mFunctions.get(), getNativeCaps(), getNativeExtensions(), mFeatures);
    mBlitter          = new BlitGL(mFunctions.get(), mFeatures, mStateManager);
    mMultiviewClearer = new ClearMultiviewGL(mFunctions.get(), mStateManager);

    bool hasDebugOutput = mFunctions->isAtLeastGL(gl::Version(4, 3)) ||
                          mFunctions->hasGLExtension("GL_KHR_debug") ||
                          mFunctions->isAtLeastGLES(gl::Version(3, 2)) ||
                          mFunctions->hasGLESExtension("GL_KHR_debug");

    mUseDebugOutput = hasDebugOutput && ShouldUseDebugLayers(attribMap);

    if (mUseDebugOutput)
    {
        mFunctions->enable(GL_DEBUG_OUTPUT);
        mFunctions->enable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        mFunctions->debugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_HIGH, 0,
                                        nullptr, GL_TRUE);
        mFunctions->debugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_MEDIUM, 0,
                                        nullptr, GL_TRUE);
        mFunctions->debugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_LOW, 0,
                                        nullptr, GL_FALSE);
        mFunctions->debugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION,
                                        0, nullptr, GL_FALSE);
        mFunctions->debugMessageCallback(&LogGLDebugMessage, nullptr);
    }

    if (mFeatures.initializeCurrentVertexAttributes.enabled)
    {
        GLint maxVertexAttribs = 0;
        mFunctions->getIntegerv(GL_MAX_VERTEX_ATTRIBS, &maxVertexAttribs);

        for (GLint i = 0; i < maxVertexAttribs; ++i)
        {
            mFunctions->vertexAttrib4f(i, 0.0f, 0.0f, 0.0f, 1.0f);
        }
    }

    if (hasNativeParallelCompile() && !mNativeParallelCompileEnabled)
    {
        SetMaxShaderCompilerThreads(mFunctions.get(), 0xffffffff);
        mNativeParallelCompileEnabled = true;
    }
}

RendererGL::~RendererGL()
{
    SafeDelete(mBlitter);
    SafeDelete(mMultiviewClearer);
    SafeDelete(mStateManager);

    std::lock_guard<std::mutex> lock(mWorkerMutex);

    ASSERT(mCurrentWorkerContexts.empty());
    mWorkerContextPool.clear();
}

angle::Result RendererGL::flush()
{
    mFunctions->flush();
    return angle::Result::Continue;
}

angle::Result RendererGL::finish()
{
    if (mFeatures.finishDoesNotCauseQueriesToBeAvailable.enabled && mUseDebugOutput)
    {
        mFunctions->enable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    }

    mFunctions->finish();

    if (mFeatures.finishDoesNotCauseQueriesToBeAvailable.enabled && mUseDebugOutput)
    {
        mFunctions->disable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    }

    return angle::Result::Continue;
}

void RendererGL::stencilFillPath(const gl::State &state,
                                 const gl::Path *path,
                                 GLenum fillMode,
                                 GLuint mask)
{
    const auto *pathObj = GetImplAs<PathGL>(path);

    mFunctions->stencilFillPathNV(pathObj->getPathID(), fillMode, mask);

    ASSERT(mFunctions->getError() == GL_NO_ERROR);
}

void RendererGL::stencilStrokePath(const gl::State &state,
                                   const gl::Path *path,
                                   GLint reference,
                                   GLuint mask)
{
    const auto *pathObj = GetImplAs<PathGL>(path);

    mFunctions->stencilStrokePathNV(pathObj->getPathID(), reference, mask);

    ASSERT(mFunctions->getError() == GL_NO_ERROR);
}

void RendererGL::coverFillPath(const gl::State &state, const gl::Path *path, GLenum coverMode)
{

    const auto *pathObj = GetImplAs<PathGL>(path);
    mFunctions->coverFillPathNV(pathObj->getPathID(), coverMode);

    ASSERT(mFunctions->getError() == GL_NO_ERROR);
}

void RendererGL::coverStrokePath(const gl::State &state, const gl::Path *path, GLenum coverMode)
{
    const auto *pathObj = GetImplAs<PathGL>(path);
    mFunctions->coverStrokePathNV(pathObj->getPathID(), coverMode);

    ASSERT(mFunctions->getError() == GL_NO_ERROR);
}

void RendererGL::stencilThenCoverFillPath(const gl::State &state,
                                          const gl::Path *path,
                                          GLenum fillMode,
                                          GLuint mask,
                                          GLenum coverMode)
{

    const auto *pathObj = GetImplAs<PathGL>(path);
    mFunctions->stencilThenCoverFillPathNV(pathObj->getPathID(), fillMode, mask, coverMode);

    ASSERT(mFunctions->getError() == GL_NO_ERROR);
}

void RendererGL::stencilThenCoverStrokePath(const gl::State &state,
                                            const gl::Path *path,
                                            GLint reference,
                                            GLuint mask,
                                            GLenum coverMode)
{

    const auto *pathObj = GetImplAs<PathGL>(path);
    mFunctions->stencilThenCoverStrokePathNV(pathObj->getPathID(), reference, mask, coverMode);

    ASSERT(mFunctions->getError() == GL_NO_ERROR);
}

void RendererGL::coverFillPathInstanced(const gl::State &state,
                                        const std::vector<gl::Path *> &paths,
                                        GLenum coverMode,
                                        GLenum transformType,
                                        const GLfloat *transformValues)
{
    const auto &pathObjs = GatherPaths(paths);

    mFunctions->coverFillPathInstancedNV(static_cast<GLsizei>(pathObjs.size()), GL_UNSIGNED_INT,
                                         &pathObjs[0], 0, coverMode, transformType,
                                         transformValues);

    ASSERT(mFunctions->getError() == GL_NO_ERROR);
}
void RendererGL::coverStrokePathInstanced(const gl::State &state,
                                          const std::vector<gl::Path *> &paths,
                                          GLenum coverMode,
                                          GLenum transformType,
                                          const GLfloat *transformValues)
{
    const auto &pathObjs = GatherPaths(paths);

    mFunctions->coverStrokePathInstancedNV(static_cast<GLsizei>(pathObjs.size()), GL_UNSIGNED_INT,
                                           &pathObjs[0], 0, coverMode, transformType,
                                           transformValues);

    ASSERT(mFunctions->getError() == GL_NO_ERROR);
}
void RendererGL::stencilFillPathInstanced(const gl::State &state,
                                          const std::vector<gl::Path *> &paths,
                                          GLenum fillMode,
                                          GLuint mask,
                                          GLenum transformType,
                                          const GLfloat *transformValues)
{
    const auto &pathObjs = GatherPaths(paths);

    mFunctions->stencilFillPathInstancedNV(static_cast<GLsizei>(pathObjs.size()), GL_UNSIGNED_INT,
                                           &pathObjs[0], 0, fillMode, mask, transformType,
                                           transformValues);

    ASSERT(mFunctions->getError() == GL_NO_ERROR);
}
void RendererGL::stencilStrokePathInstanced(const gl::State &state,
                                            const std::vector<gl::Path *> &paths,
                                            GLint reference,
                                            GLuint mask,
                                            GLenum transformType,
                                            const GLfloat *transformValues)
{
    const auto &pathObjs = GatherPaths(paths);

    mFunctions->stencilStrokePathInstancedNV(static_cast<GLsizei>(pathObjs.size()), GL_UNSIGNED_INT,
                                             &pathObjs[0], 0, reference, mask, transformType,
                                             transformValues);

    ASSERT(mFunctions->getError() == GL_NO_ERROR);
}

void RendererGL::stencilThenCoverFillPathInstanced(const gl::State &state,
                                                   const std::vector<gl::Path *> &paths,
                                                   GLenum coverMode,
                                                   GLenum fillMode,
                                                   GLuint mask,
                                                   GLenum transformType,
                                                   const GLfloat *transformValues)
{
    const auto &pathObjs = GatherPaths(paths);

    mFunctions->stencilThenCoverFillPathInstancedNV(
        static_cast<GLsizei>(pathObjs.size()), GL_UNSIGNED_INT, &pathObjs[0], 0, fillMode, mask,
        coverMode, transformType, transformValues);

    ASSERT(mFunctions->getError() == GL_NO_ERROR);
}
void RendererGL::stencilThenCoverStrokePathInstanced(const gl::State &state,
                                                     const std::vector<gl::Path *> &paths,
                                                     GLenum coverMode,
                                                     GLint reference,
                                                     GLuint mask,
                                                     GLenum transformType,
                                                     const GLfloat *transformValues)
{
    const auto &pathObjs = GatherPaths(paths);

    mFunctions->stencilThenCoverStrokePathInstancedNV(
        static_cast<GLsizei>(pathObjs.size()), GL_UNSIGNED_INT, &pathObjs[0], 0, reference, mask,
        coverMode, transformType, transformValues);

    ASSERT(mFunctions->getError() == GL_NO_ERROR);
}

gl::GraphicsResetStatus RendererGL::getResetStatus()
{
    return gl::FromGLenum<gl::GraphicsResetStatus>(mFunctions->getGraphicsResetStatus());
}

void RendererGL::insertEventMarker(GLsizei length, const char *marker) {}

void RendererGL::pushGroupMarker(GLsizei length, const char *marker) {}

void RendererGL::popGroupMarker() {}

void RendererGL::pushDebugGroup(GLenum source, GLuint id, const std::string &message) {}

void RendererGL::popDebugGroup() {}

std::string RendererGL::getVendorString() const
{
    return std::string(reinterpret_cast<const char *>(mFunctions->getString(GL_VENDOR)));
}

std::string RendererGL::getRendererDescription() const
{
    std::string nativeVendorString(
        reinterpret_cast<const char *>(mFunctions->getString(GL_VENDOR)));
    std::string nativeRendererString(
        reinterpret_cast<const char *>(mFunctions->getString(GL_RENDERER)));

    std::ostringstream rendererString;
    rendererString << nativeVendorString << ", " << nativeRendererString << ", OpenGL";
    if (mFunctions->standard == STANDARD_GL_ES)
    {
        rendererString << " ES";
    }
    rendererString << " " << mFunctions->version.major << "." << mFunctions->version.minor;
    if (mFunctions->standard == STANDARD_GL_DESKTOP)
    {
        // Some drivers (NVIDIA) use a profile mask of 0 when in compatibility profile.
        if ((mFunctions->profile & GL_CONTEXT_COMPATIBILITY_PROFILE_BIT) != 0 ||
            (mFunctions->isAtLeastGL(gl::Version(3, 2)) && mFunctions->profile == 0))
        {
            rendererString << " compatibility";
        }
        else if ((mFunctions->profile & GL_CONTEXT_CORE_PROFILE_BIT) != 0)
        {
            rendererString << " core";
        }
    }

    return rendererString.str();
}

const gl::Version &RendererGL::getMaxSupportedESVersion() const
{
    // Force generation of caps
    getNativeCaps();

    return mMaxSupportedESVersion;
}

void RendererGL::generateCaps(gl::Caps *outCaps,
                              gl::TextureCapsMap *outTextureCaps,
                              gl::Extensions *outExtensions,
                              gl::Limitations * /* outLimitations */) const
{
    nativegl_gl::GenerateCaps(mFunctions.get(), mFeatures, outCaps, outTextureCaps, outExtensions,
                              &mMaxSupportedESVersion, &mMultiviewImplementationType);
}

GLint RendererGL::getGPUDisjoint()
{
    // TODO(ewell): On GLES backends we should find a way to reliably query disjoint events
    return 0;
}

GLint64 RendererGL::getTimestamp()
{
    GLint64 result = 0;
    mFunctions->getInteger64v(GL_TIMESTAMP, &result);
    return result;
}

void RendererGL::ensureCapsInitialized() const
{
    if (!mCapsInitialized)
    {
        generateCaps(&mNativeCaps, &mNativeTextureCaps, &mNativeExtensions, &mNativeLimitations);
        mCapsInitialized = true;
    }
}

const gl::Caps &RendererGL::getNativeCaps() const
{
    ensureCapsInitialized();
    return mNativeCaps;
}

const gl::TextureCapsMap &RendererGL::getNativeTextureCaps() const
{
    ensureCapsInitialized();
    return mNativeTextureCaps;
}

const gl::Extensions &RendererGL::getNativeExtensions() const
{
    ensureCapsInitialized();
    return mNativeExtensions;
}

const gl::Limitations &RendererGL::getNativeLimitations() const
{
    ensureCapsInitialized();
    return mNativeLimitations;
}

MultiviewImplementationTypeGL RendererGL::getMultiviewImplementationType() const
{
    ensureCapsInitialized();
    return mMultiviewImplementationType;
}

void RendererGL::initializeFrontendFeatures(angle::FrontendFeatures *features) const
{
    ensureCapsInitialized();
    nativegl_gl::InitializeFrontendFeatures(mFunctions.get(), features);
}

angle::Result RendererGL::dispatchCompute(const gl::Context *context,
                                          GLuint numGroupsX,
                                          GLuint numGroupsY,
                                          GLuint numGroupsZ)
{
    mFunctions->dispatchCompute(numGroupsX, numGroupsY, numGroupsZ);
    return angle::Result::Continue;
}

angle::Result RendererGL::dispatchComputeIndirect(const gl::Context *context, GLintptr indirect)
{
    mFunctions->dispatchComputeIndirect(indirect);
    return angle::Result::Continue;
}

angle::Result RendererGL::memoryBarrier(GLbitfield barriers)
{
    mFunctions->memoryBarrier(barriers);
    return angle::Result::Continue;
}
angle::Result RendererGL::memoryBarrierByRegion(GLbitfield barriers)
{
    mFunctions->memoryBarrierByRegion(barriers);
    return angle::Result::Continue;
}

bool RendererGL::bindWorkerContext(std::string *infoLog)
{
    if (mFeatures.disableWorkerContexts.enabled)
    {
        return false;
    }

    std::thread::id threadID = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(mWorkerMutex);
    std::unique_ptr<WorkerContext> workerContext;
    if (!mWorkerContextPool.empty())
    {
        auto it       = mWorkerContextPool.begin();
        workerContext = std::move(*it);
        mWorkerContextPool.erase(it);
    }
    else
    {
        WorkerContext *newContext = createWorkerContext(infoLog);
        if (newContext == nullptr)
        {
            return false;
        }
        workerContext.reset(newContext);
    }

    if (!workerContext->makeCurrent())
    {
        mWorkerContextPool.push_back(std::move(workerContext));
        return false;
    }
    mCurrentWorkerContexts[threadID] = std::move(workerContext);
    return true;
}

void RendererGL::unbindWorkerContext()
{
    std::thread::id threadID = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(mWorkerMutex);

    auto it = mCurrentWorkerContexts.find(threadID);
    ASSERT(it != mCurrentWorkerContexts.end());
    (*it).second->unmakeCurrent();
    mWorkerContextPool.push_back(std::move((*it).second));
    mCurrentWorkerContexts.erase(it);
}

unsigned int RendererGL::getMaxWorkerContexts()
{
    // No more than 16 worker contexts.
    return std::min(16u, std::thread::hardware_concurrency());
}

bool RendererGL::hasNativeParallelCompile()
{
    return mFunctions->maxShaderCompilerThreadsKHR != nullptr ||
           mFunctions->maxShaderCompilerThreadsARB != nullptr;
}

void RendererGL::setMaxShaderCompilerThreads(GLuint count)
{
    if (hasNativeParallelCompile())
    {
        SetMaxShaderCompilerThreads(mFunctions.get(), count);
    }
}

ScopedWorkerContextGL::ScopedWorkerContextGL(RendererGL *renderer, std::string *infoLog)
    : mRenderer(renderer)
{
    mValid = mRenderer->bindWorkerContext(infoLog);
}

ScopedWorkerContextGL::~ScopedWorkerContextGL()
{
    if (mValid)
    {
        mRenderer->unbindWorkerContext();
    }
}

bool ScopedWorkerContextGL::operator()() const
{
    return mValid;
}

}  // namespace rx
