//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Context.h: Defines the gl::Context class, managing all GL state and performing
// rendering operations. It is the GLES2 specific implementation of EGLContext.

#ifndef LIBANGLE_CONTEXT_H_
#define LIBANGLE_CONTEXT_H_

#include <set>
#include <string>

#include "angle_gl.h"
#include "common/MemoryBuffer.h"
#include "common/PackedEnums.h"
#include "common/angleutils.h"
#include "libANGLE/Caps.h"
#include "libANGLE/Constants.h"
#include "libANGLE/Context_gl_1_0_autogen.h"
#include "libANGLE/Context_gl_1_1_autogen.h"
#include "libANGLE/Context_gl_1_2_autogen.h"
#include "libANGLE/Context_gl_1_3_autogen.h"
#include "libANGLE/Context_gl_1_4_autogen.h"
#include "libANGLE/Context_gl_1_5_autogen.h"
#include "libANGLE/Context_gl_2_0_autogen.h"
#include "libANGLE/Context_gl_2_1_autogen.h"
#include "libANGLE/Context_gl_3_0_autogen.h"
#include "libANGLE/Context_gl_3_1_autogen.h"
#include "libANGLE/Context_gl_3_2_autogen.h"
#include "libANGLE/Context_gl_3_3_autogen.h"
#include "libANGLE/Context_gl_4_0_autogen.h"
#include "libANGLE/Context_gl_4_1_autogen.h"
#include "libANGLE/Context_gl_4_2_autogen.h"
#include "libANGLE/Context_gl_4_3_autogen.h"
#include "libANGLE/Context_gl_4_4_autogen.h"
#include "libANGLE/Context_gl_4_5_autogen.h"
#include "libANGLE/Context_gl_4_6_autogen.h"
#include "libANGLE/Context_gles_1_0_autogen.h"
#include "libANGLE/Context_gles_2_0_autogen.h"
#include "libANGLE/Context_gles_3_0_autogen.h"
#include "libANGLE/Context_gles_3_1_autogen.h"
#include "libANGLE/Context_gles_ext_autogen.h"
#include "libANGLE/Error.h"
#include "libANGLE/HandleAllocator.h"
#include "libANGLE/RefCountObject.h"
#include "libANGLE/ResourceManager.h"
#include "libANGLE/ResourceMap.h"
#include "libANGLE/State.h"
#include "libANGLE/VertexAttribute.h"
#include "libANGLE/WorkerThread.h"
#include "libANGLE/angletypes.h"

namespace angle
{
class FrameCapture;
struct FrontendFeatures;
}  // namespace angle

namespace rx
{
class ContextImpl;
class EGLImplFactory;
}  // namespace rx

namespace egl
{
class AttributeMap;
class Surface;
struct Config;
class Thread;
}  // namespace egl

namespace gl
{
class Buffer;
class Compiler;
class FenceNV;
class Framebuffer;
class GLES1Renderer;
class MemoryProgramCache;
class MemoryObject;
class Program;
class ProgramPipeline;
class Query;
class Renderbuffer;
class Sampler;
class Semaphore;
class Shader;
class Sync;
class Texture;
class TransformFeedback;
class VertexArray;
struct VertexAttribute;

class ErrorSet : angle::NonCopyable
{
  public:
    explicit ErrorSet(Context *context);
    ~ErrorSet();

    bool empty() const;
    GLenum popError();

    void handleError(GLenum errorCode,
                     const char *message,
                     const char *file,
                     const char *function,
                     unsigned int line);

    void validationError(GLenum errorCode, const char *message);

  private:
    Context *mContext;
    std::set<GLenum> mErrors;
};

enum class VertexAttribTypeCase
{
    Invalid        = 0,
    Valid          = 1,
    ValidSize4Only = 2,
    ValidSize3or4  = 3,
};

// Helper class for managing cache variables and state changes.
class StateCache final : angle::NonCopyable
{
  public:
    StateCache();
    ~StateCache();

    void initialize(Context *context);

    // Places that can trigger updateActiveAttribsMask:
    // 1. onVertexArrayBindingChange.
    // 2. onProgramExecutableChange.
    // 3. onVertexArrayStateChange.
    // 4. onGLES1ClientStateChange.
    AttributesMask getActiveBufferedAttribsMask() const { return mCachedActiveBufferedAttribsMask; }
    AttributesMask getActiveClientAttribsMask() const { return mCachedActiveClientAttribsMask; }
    AttributesMask getActiveDefaultAttribsMask() const { return mCachedActiveDefaultAttribsMask; }
    bool hasAnyEnabledClientAttrib() const { return mCachedHasAnyEnabledClientAttrib; }
    bool hasAnyActiveClientAttrib() const { return mCachedActiveClientAttribsMask.any(); }

    // Places that can trigger updateVertexElementLimits:
    // 1. onVertexArrayBindingChange.
    // 2. onProgramExecutableChange.
    // 3. onVertexArrayFormatChange.
    // 4. onVertexArrayBufferChange.
    // 5. onVertexArrayStateChange.
    GLint64 getNonInstancedVertexElementLimit() const
    {
        return mCachedNonInstancedVertexElementLimit;
    }
    GLint64 getInstancedVertexElementLimit() const { return mCachedInstancedVertexElementLimit; }

    // Places that can trigger updateBasicDrawStatesError:
    // 1. onVertexArrayBindingChange.
    // 2. onProgramExecutableChange.
    // 3. onVertexArrayBufferContentsChange.
    // 4. onVertexArrayStateChange.
    // 5. onVertexArrayBufferStateChange.
    // 6. onDrawFramebufferChange.
    // 7. onContextCapChange.
    // 8. onStencilStateChange.
    // 9. onDefaultVertexAttributeChange.
    // 10. onActiveTextureChange.
    // 11. onQueryChange.
    // 12. onActiveTransformFeedbackChange.
    // 13. onUniformBufferStateChange.
    // 14. onColorMaskChange.
    // 15. onBufferBindingChange.
    bool hasBasicDrawStatesError(Context *context) const
    {
        if (mCachedBasicDrawStatesError == 0)
        {
            return false;
        }
        if (mCachedBasicDrawStatesError != kInvalidPointer)
        {
            return true;
        }
        return getBasicDrawStatesErrorImpl(context) != 0;
    }

    intptr_t getBasicDrawStatesError(Context *context) const
    {
        if (mCachedBasicDrawStatesError != kInvalidPointer)
        {
            return mCachedBasicDrawStatesError;
        }

        return getBasicDrawStatesErrorImpl(context);
    }

    // Places that can trigger updateBasicDrawElementsError:
    // 1. onActiveTransformFeedbackChange.
    // 2. onVertexArrayBufferStateChange.
    // 3. onBufferBindingChange.
    intptr_t getBasicDrawElementsError(Context *context) const
    {
        if (mCachedBasicDrawElementsError != kInvalidPointer)
        {
            return mCachedBasicDrawElementsError;
        }

        return getBasicDrawElementsErrorImpl(context);
    }

    // Places that can trigger updateValidDrawModes:
    // 1. onProgramExecutableChange.
    // 2. onActiveTransformFeedbackChange.
    bool isValidDrawMode(PrimitiveMode primitiveMode) const
    {
        return mCachedValidDrawModes[primitiveMode];
    }

    // Cannot change except on Context/Extension init.
    bool isValidBindTextureType(TextureType type) const
    {
        return mCachedValidBindTextureTypes[type];
    }

    // Cannot change except on Context/Extension init.
    bool isValidDrawElementsType(DrawElementsType type) const
    {
        return mCachedValidDrawElementsTypes[type];
    }

    // Places that can trigger updateTransformFeedbackActiveUnpaused:
    // 1. onActiveTransformFeedbackChange.
    bool isTransformFeedbackActiveUnpaused() const
    {
        return mCachedTransformFeedbackActiveUnpaused;
    }

    // Cannot change except on Context/Extension init.
    VertexAttribTypeCase getVertexAttribTypeValidation(VertexAttribType type) const
    {
        return mCachedVertexAttribTypesValidation[type];
    }

    VertexAttribTypeCase getIntegerVertexAttribTypeValidation(VertexAttribType type) const
    {
        return mCachedIntegerVertexAttribTypesValidation[type];
    }

    // Places that can trigger updateActiveShaderStorageBufferIndices:
    // 1. onProgramExecutableChange.
    StorageBuffersMask getActiveShaderStorageBufferIndices() const
    {
        return mCachedActiveShaderStorageBufferIndices;
    }

    // State change notifications.
    void onVertexArrayBindingChange(Context *context);
    void onProgramExecutableChange(Context *context);
    void onVertexArrayFormatChange(Context *context);
    void onVertexArrayBufferContentsChange(Context *context);
    void onVertexArrayStateChange(Context *context);
    void onVertexArrayBufferStateChange(Context *context);
    void onGLES1ClientStateChange(Context *context);
    void onDrawFramebufferChange(Context *context);
    void onContextCapChange(Context *context);
    void onStencilStateChange(Context *context);
    void onDefaultVertexAttributeChange(Context *context);
    void onActiveTextureChange(Context *context);
    void onQueryChange(Context *context);
    void onActiveTransformFeedbackChange(Context *context);
    void onUniformBufferStateChange(Context *context);
    void onColorMaskChange(Context *context);
    void onBufferBindingChange(Context *context);

  private:
    // Cache update functions.
    void updateActiveAttribsMask(Context *context);
    void updateVertexElementLimits(Context *context);
    void updateVertexElementLimitsImpl(Context *context);
    void updateValidDrawModes(Context *context);
    void updateValidBindTextureTypes(Context *context);
    void updateValidDrawElementsTypes(Context *context);
    void updateBasicDrawStatesError();
    void updateBasicDrawElementsError();
    void updateTransformFeedbackActiveUnpaused(Context *context);
    void updateVertexAttribTypesValidation(Context *context);
    void updateActiveShaderStorageBufferIndices(Context *context);

    void setValidDrawModes(bool pointsOK, bool linesOK, bool trisOK, bool lineAdjOK, bool triAdjOK);

    intptr_t getBasicDrawStatesErrorImpl(Context *context) const;
    intptr_t getBasicDrawElementsErrorImpl(Context *context) const;

    static constexpr intptr_t kInvalidPointer = 1;

    AttributesMask mCachedActiveBufferedAttribsMask;
    AttributesMask mCachedActiveClientAttribsMask;
    AttributesMask mCachedActiveDefaultAttribsMask;
    bool mCachedHasAnyEnabledClientAttrib;
    GLint64 mCachedNonInstancedVertexElementLimit;
    GLint64 mCachedInstancedVertexElementLimit;
    mutable intptr_t mCachedBasicDrawStatesError;
    mutable intptr_t mCachedBasicDrawElementsError;
    bool mCachedTransformFeedbackActiveUnpaused;
    StorageBuffersMask mCachedActiveShaderStorageBufferIndices;

    // Reserve an extra slot at the end of these maps for invalid enum.
    angle::PackedEnumMap<PrimitiveMode, bool, angle::EnumSize<PrimitiveMode>() + 1>
        mCachedValidDrawModes;
    angle::PackedEnumMap<TextureType, bool, angle::EnumSize<TextureType>() + 1>
        mCachedValidBindTextureTypes;
    angle::PackedEnumMap<DrawElementsType, bool, angle::EnumSize<DrawElementsType>() + 1>
        mCachedValidDrawElementsTypes;
    angle::PackedEnumMap<VertexAttribType,
                         VertexAttribTypeCase,
                         angle::EnumSize<VertexAttribType>() + 1>
        mCachedVertexAttribTypesValidation;
    angle::PackedEnumMap<VertexAttribType,
                         VertexAttribTypeCase,
                         angle::EnumSize<VertexAttribType>() + 1>
        mCachedIntegerVertexAttribTypesValidation;
};

class Context final : public egl::LabeledObject, angle::NonCopyable, public angle::ObserverInterface
{
  public:
    Context(egl::Display *display,
            const egl::Config *config,
            const Context *shareContext,
            TextureManager *shareTextures,
            MemoryProgramCache *memoryProgramCache,
            const EGLenum clientType,
            const egl::AttributeMap &attribs,
            const egl::DisplayExtensions &displayExtensions,
            const egl::ClientExtensions &clientExtensions);

    // Use for debugging.
    int id() const { return mState.mID; }

    egl::Error onDestroy(const egl::Display *display);
    ~Context() override;

    void setLabel(EGLLabelKHR label) override;
    EGLLabelKHR getLabel() const override;

    egl::Error makeCurrent(egl::Display *display,
                           egl::Surface *drawSurface,
                           egl::Surface *readSurface);
    egl::Error unMakeCurrent(const egl::Display *display);

    // These create  and destroy methods are merely pass-throughs to
    // ResourceManager, which owns these object types
    BufferID createBuffer();
    TextureID createTexture();
    RenderbufferID createRenderbuffer();
    ProgramPipelineID createProgramPipeline();
    MemoryObjectID createMemoryObject();
    SemaphoreID createSemaphore();

    void deleteBuffer(BufferID buffer);
    void deleteTexture(TextureID texture);
    void deleteRenderbuffer(RenderbufferID renderbuffer);
    void deleteProgramPipeline(ProgramPipelineID pipeline);
    void deleteMemoryObject(MemoryObjectID memoryObject);
    void deleteSemaphore(SemaphoreID semaphore);

    // CHROMIUM_path_rendering
    bool isPathGenerated(PathID path) const;

    void bindReadFramebuffer(FramebufferID framebufferHandle);
    void bindDrawFramebuffer(FramebufferID framebufferHandle);

    Buffer *getBuffer(BufferID handle) const;
    FenceNV *getFenceNV(FenceNVID handle);
    Sync *getSync(GLsync handle) const;
    ANGLE_INLINE Texture *getTexture(TextureID handle) const
    {
        return mState.mTextureManager->getTexture(handle);
    }

    Framebuffer *getFramebuffer(FramebufferID handle) const;
    Renderbuffer *getRenderbuffer(RenderbufferID handle) const;
    VertexArray *getVertexArray(VertexArrayID handle) const;
    Sampler *getSampler(SamplerID handle) const;
    Query *getQuery(QueryID handle, bool create, QueryType type);
    Query *getQuery(QueryID handle) const;
    TransformFeedback *getTransformFeedback(TransformFeedbackID handle) const;
    ProgramPipeline *getProgramPipeline(ProgramPipelineID handle) const;
    MemoryObject *getMemoryObject(MemoryObjectID handle) const;
    Semaphore *getSemaphore(SemaphoreID handle) const;

    Texture *getTextureByType(TextureType type) const;
    Texture *getTextureByTarget(TextureTarget target) const;
    Texture *getSamplerTexture(unsigned int sampler, TextureType type) const;

    Compiler *getCompiler() const;

    bool isVertexArrayGenerated(VertexArrayID vertexArray);
    bool isTransformFeedbackGenerated(TransformFeedbackID transformFeedback);

    void getBooleanvImpl(GLenum pname, GLboolean *params);
    void getFloatvImpl(GLenum pname, GLfloat *params);
    void getIntegervImpl(GLenum pname, GLint *params);
    void getInteger64vImpl(GLenum pname, GLint64 *params);
    void getPointerv(GLenum pname, void **params) const;

    // Framebuffers are owned by the Context, so these methods do not pass through
    FramebufferID createFramebuffer();
    void deleteFramebuffer(FramebufferID framebuffer);

    bool hasActiveTransformFeedback(ShaderProgramID program) const;

    // GL emulation: Interface to entry points
    ANGLE_GL_1_0_CONTEXT_API
    ANGLE_GL_1_1_CONTEXT_API
    ANGLE_GL_1_2_CONTEXT_API
    ANGLE_GL_1_3_CONTEXT_API
    ANGLE_GL_1_4_CONTEXT_API
    ANGLE_GL_1_5_CONTEXT_API
    ANGLE_GL_2_0_CONTEXT_API
    ANGLE_GL_2_1_CONTEXT_API
    ANGLE_GL_3_0_CONTEXT_API
    ANGLE_GL_3_1_CONTEXT_API
    ANGLE_GL_3_2_CONTEXT_API
    ANGLE_GL_3_3_CONTEXT_API
    ANGLE_GL_4_0_CONTEXT_API
    ANGLE_GL_4_1_CONTEXT_API
    ANGLE_GL_4_2_CONTEXT_API
    ANGLE_GL_4_3_CONTEXT_API
    ANGLE_GL_4_4_CONTEXT_API
    ANGLE_GL_4_5_CONTEXT_API
    ANGLE_GL_4_6_CONTEXT_API

    // GLES emulation: Interface to entry points
    ANGLE_GLES_1_0_CONTEXT_API
    ANGLE_GLES_2_0_CONTEXT_API
    ANGLE_GLES_3_0_CONTEXT_API
    ANGLE_GLES_3_1_CONTEXT_API
    ANGLE_GLES_EXT_CONTEXT_API

    // Consumes an error.
    void handleError(GLenum errorCode,
                     const char *message,
                     const char *file,
                     const char *function,
                     unsigned int line);

    void validationError(GLenum errorCode, const char *message);

    void markContextLost(GraphicsResetStatus status);

    bool isContextLost() const { return mContextLost; }

    GLenum getGraphicsResetStrategy() const { return mResetStrategy; }
    bool isResetNotificationEnabled();

    const egl::Config *getConfig() const;
    EGLenum getClientType() const;
    EGLenum getRenderBuffer() const;

    const GLubyte *getString(GLenum name) const;
    const GLubyte *getStringi(GLenum name, GLuint index) const;

    size_t getExtensionStringCount() const;

    bool isExtensionRequestable(const char *name);
    size_t getRequestableExtensionStringCount() const;

    rx::ContextImpl *getImplementation() const { return mImplementation.get(); }

    ANGLE_NO_DISCARD bool getScratchBuffer(size_t requestedSizeBytes,
                                           angle::MemoryBuffer **scratchBufferOut) const;
    ANGLE_NO_DISCARD bool getZeroFilledBuffer(size_t requstedSizeBytes,
                                              angle::MemoryBuffer **zeroBufferOut) const;
    angle::ScratchBuffer *getScratchBuffer() const { return &mScratchBuffer; }

    angle::Result prepareForCopyImage();
    angle::Result prepareForDispatch();

    MemoryProgramCache *getMemoryProgramCache() const { return mMemoryProgramCache; }

    bool hasBeenCurrent() const { return mHasBeenCurrent; }
    egl::Display *getDisplay() const { return mDisplay; }
    egl::Surface *getCurrentDrawSurface() const { return mCurrentDrawSurface; }
    egl::Surface *getCurrentReadSurface() const { return mCurrentReadSurface; }

    bool isRobustResourceInitEnabled() const { return mState.isRobustResourceInitEnabled(); }

    bool isCurrentTransformFeedback(const TransformFeedback *tf) const;

    bool isCurrentVertexArray(const VertexArray *va) const
    {
        return mState.isCurrentVertexArray(va);
    }

    bool isShared() const { return mShared; }
    // Once a context is setShared() it cannot be undone
    void setShared() { mShared = true; }

    const State &getState() const { return mState; }
    GLint getClientMajorVersion() const { return mState.getClientMajorVersion(); }
    GLint getClientMinorVersion() const { return mState.getClientMinorVersion(); }
    const Version &getClientVersion() const { return mState.getClientVersion(); }
    const Caps &getCaps() const { return mState.getCaps(); }
    const TextureCapsMap &getTextureCaps() const { return mState.getTextureCaps(); }
    const Extensions &getExtensions() const { return mState.getExtensions(); }
    const Limitations &getLimitations() const { return mState.getLimitations(); }
    bool skipValidation() const { return mSkipValidation; }
    bool isGLES1() const;

    // Specific methods needed for validation.
    bool getQueryParameterInfo(GLenum pname, GLenum *type, unsigned int *numParams) const;
    bool getIndexedQueryParameterInfo(GLenum target, GLenum *type, unsigned int *numParams) const;

    ANGLE_INLINE Program *getProgramResolveLink(ShaderProgramID handle) const
    {
        Program *program = mState.mShaderProgramManager->getProgram(handle);
        if (program)
        {
            program->resolveLink(this);
        }
        return program;
    }

    Program *getProgramNoResolveLink(ShaderProgramID handle) const;
    Shader *getShader(ShaderProgramID handle) const;

    ANGLE_INLINE bool isTextureGenerated(TextureID texture) const
    {
        return mState.mTextureManager->isHandleGenerated(texture);
    }

    ANGLE_INLINE bool isBufferGenerated(BufferID buffer) const
    {
        return mState.mBufferManager->isHandleGenerated(buffer);
    }

    bool isRenderbufferGenerated(RenderbufferID renderbuffer) const;
    bool isFramebufferGenerated(FramebufferID framebuffer) const;
    bool isProgramPipelineGenerated(ProgramPipelineID pipeline) const;

    bool usingDisplayTextureShareGroup() const;

    // Hack for the special WebGL 1 "DEPTH_STENCIL" internal format.
    GLenum getConvertedRenderbufferFormat(GLenum internalformat) const;

    bool isWebGL() const { return mState.isWebGL(); }
    bool isWebGL1() const { return mState.isWebGL1(); }

    bool isValidBufferBinding(BufferBinding binding) const { return mValidBufferBindings[binding]; }

    // GLES1 emulation: Renderer level (for validation)
    int vertexArrayIndex(ClientVertexArrayType type) const;
    static int TexCoordArrayIndex(unsigned int unit);

    // GL_KHR_parallel_shader_compile
    std::shared_ptr<angle::WorkerThreadPool> getWorkerThreadPool() const { return mThreadPool; }

    const StateCache &getStateCache() const { return mStateCache; }
    StateCache &getStateCache() { return mStateCache; }

    void onSubjectStateChange(angle::SubjectIndex index, angle::SubjectMessage message) override;

    void onSamplerUniformChange(size_t textureUnitIndex);

    bool isBufferAccessValidationEnabled() const { return mBufferAccessValidationEnabled; }

    const angle::FrontendFeatures &getFrontendFeatures() const;

    angle::FrameCapture *getFrameCapture() { return mFrameCapture.get(); }

    void onPostSwap() const;

  private:
    void initialize();

    bool noopDraw(PrimitiveMode mode, GLsizei count);
    bool noopDrawInstanced(PrimitiveMode mode, GLsizei count, GLsizei instanceCount);

    angle::Result prepareForDraw(PrimitiveMode mode);
    angle::Result prepareForClear(GLbitfield mask);
    angle::Result prepareForClearBuffer(GLenum buffer, GLint drawbuffer);
    angle::Result syncState(const State::DirtyBits &bitMask, const State::DirtyObjects &objectMask);
    angle::Result syncDirtyBits();
    angle::Result syncDirtyBits(const State::DirtyBits &bitMask);
    angle::Result syncDirtyObjects(const State::DirtyObjects &objectMask);
    angle::Result syncStateForReadPixels();
    angle::Result syncStateForTexImage();
    angle::Result syncStateForBlit();
    angle::Result syncStateForPathOperation();

    VertexArray *checkVertexArrayAllocation(VertexArrayID vertexArrayHandle);
    TransformFeedback *checkTransformFeedbackAllocation(TransformFeedbackID transformFeedback);

    angle::Result onProgramLink(Program *programObject);

    void detachBuffer(Buffer *buffer);
    void detachTexture(TextureID texture);
    void detachFramebuffer(FramebufferID framebuffer);
    void detachRenderbuffer(RenderbufferID renderbuffer);
    void detachVertexArray(VertexArrayID vertexArray);
    void detachTransformFeedback(TransformFeedbackID transformFeedback);
    void detachSampler(SamplerID sampler);
    void detachProgramPipeline(ProgramPipelineID pipeline);

    // A small helper method to facilitate using the ANGLE_CONTEXT_TRY macro.
    void tryGenPaths(GLsizei range, PathID *createdOut);

    egl::Error setDefaultFramebuffer(egl::Surface *drawSurface, egl::Surface *readSurface);
    egl::Error unsetDefaultFramebuffer();

    void initRendererString();
    void initVersionStrings();
    void initExtensionStrings();

    Extensions generateSupportedExtensions() const;
    void initCaps();
    void updateCaps();

    gl::LabeledObject *getLabeledObject(GLenum identifier, GLuint name) const;
    gl::LabeledObject *getLabeledObjectFromPtr(const void *ptr) const;

    void setUniform1iImpl(Program *program, GLint location, GLsizei count, const GLint *v);

    State mState;
    bool mShared;
    bool mSkipValidation;
    bool mDisplayTextureShareGroup;

    // Recorded errors
    ErrorSet mErrors;

    // Stores for each buffer binding type whether is it allowed to be used in this context.
    angle::PackedEnumBitSet<BufferBinding> mValidBufferBindings;

    std::unique_ptr<rx::ContextImpl> mImplementation;

    EGLLabelKHR mLabel;

    // Extensions supported by the implementation plus extensions that are implemented entirely
    // within the frontend.
    Extensions mSupportedExtensions;

    // Shader compiler. Lazily initialized hence the mutable value.
    mutable BindingPointer<Compiler> mCompiler;

    const egl::Config *mConfig;

    TextureMap mZeroTextures;

    ResourceMap<FenceNV, FenceNVID> mFenceNVMap;
    HandleAllocator mFenceNVHandleAllocator;

    ResourceMap<Query, QueryID> mQueryMap;
    HandleAllocator mQueryHandleAllocator;

    ResourceMap<VertexArray, VertexArrayID> mVertexArrayMap;
    HandleAllocator mVertexArrayHandleAllocator;

    ResourceMap<TransformFeedback, TransformFeedbackID> mTransformFeedbackMap;
    HandleAllocator mTransformFeedbackHandleAllocator;

    const char *mVersionString;
    const char *mShadingLanguageString;
    const char *mRendererString;
    const char *mExtensionString;
    std::vector<const char *> mExtensionStrings;
    const char *mRequestableExtensionString;
    std::vector<const char *> mRequestableExtensionStrings;

    // GLES1 renderer state
    std::unique_ptr<GLES1Renderer> mGLES1Renderer;

    // Current/lost context flags
    bool mHasBeenCurrent;
    bool mContextLost;
    GraphicsResetStatus mResetStatus;
    bool mContextLostForced;
    GLenum mResetStrategy;
    const bool mRobustAccess;
    const bool mSurfacelessSupported;
    const bool mExplicitContextAvailable;
    egl::Surface *mCurrentDrawSurface;
    egl::Surface *mCurrentReadSurface;
    egl::Display *mDisplay;
    const bool mWebGLContext;
    bool mBufferAccessValidationEnabled;
    const bool mExtensionsEnabled;
    MemoryProgramCache *mMemoryProgramCache;

    State::DirtyObjects mDrawDirtyObjects;
    State::DirtyObjects mPathOperationDirtyObjects;

    StateCache mStateCache;

    State::DirtyBits mAllDirtyBits;
    State::DirtyBits mTexImageDirtyBits;
    State::DirtyObjects mTexImageDirtyObjects;
    State::DirtyBits mReadPixelsDirtyBits;
    State::DirtyObjects mReadPixelsDirtyObjects;
    State::DirtyBits mClearDirtyBits;
    State::DirtyObjects mClearDirtyObjects;
    State::DirtyBits mBlitDirtyBits;
    State::DirtyObjects mBlitDirtyObjects;
    State::DirtyBits mComputeDirtyBits;
    State::DirtyObjects mComputeDirtyObjects;
    State::DirtyBits mCopyImageDirtyBits;
    State::DirtyObjects mCopyImageDirtyObjects;

    // Binding to container objects that use dependent state updates.
    angle::ObserverBinding mVertexArrayObserverBinding;
    angle::ObserverBinding mDrawFramebufferObserverBinding;
    angle::ObserverBinding mReadFramebufferObserverBinding;
    std::vector<angle::ObserverBinding> mUniformBufferObserverBindings;
    std::vector<angle::ObserverBinding> mSamplerObserverBindings;
    std::vector<angle::ObserverBinding> mImageObserverBindings;

    // Not really a property of context state. The size and contexts change per-api-call.
    mutable angle::ScratchBuffer mScratchBuffer;
    mutable angle::ScratchBuffer mZeroFilledBuffer;

    std::shared_ptr<angle::WorkerThreadPool> mThreadPool;

    // Note: we use a raw pointer here so we can exclude frame capture sources from the build.
    std::unique_ptr<angle::FrameCapture> mFrameCapture;

    OverlayType mOverlay;
};

}  // namespace gl

#endif  // LIBANGLE_CONTEXT_H_
