//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// State.h: Defines the State class, encapsulating raw GL state

#ifndef LIBANGLE_STATE_H_
#define LIBANGLE_STATE_H_

#include <bitset>
#include <memory>

#include "common/Color.h"
#include "common/angleutils.h"
#include "common/bitset_utils.h"
#include "libANGLE/Debug.h"
#include "libANGLE/GLES1State.h"
#include "libANGLE/Overlay.h"
#include "libANGLE/Program.h"
#include "libANGLE/ProgramPipeline.h"
#include "libANGLE/RefCountObject.h"
#include "libANGLE/Renderbuffer.h"
#include "libANGLE/Sampler.h"
#include "libANGLE/Texture.h"
#include "libANGLE/TransformFeedback.h"
#include "libANGLE/Version.h"
#include "libANGLE/VertexArray.h"
#include "libANGLE/angletypes.h"

namespace gl
{
class BufferManager;
struct Caps;
class Context;
class FramebufferManager;
class MemoryObjectManager;
class PathManager;
class ProgramPipelineManager;
class Query;
class RenderbufferManager;
class SamplerManager;
class SemaphoreManager;
class ShaderProgramManager;
class SyncManager;
class TextureManager;
class VertexArray;

static constexpr Version ES_2_0 = Version(2, 0);
static constexpr Version ES_3_0 = Version(3, 0);
static constexpr Version ES_3_1 = Version(3, 1);

using ContextID = uintptr_t;

class State : angle::NonCopyable
{
  public:
    State(ContextID contextIn,
          const State *shareContextState,
          TextureManager *shareTextures,
          const OverlayType *overlay,
          const EGLenum clientType,
          const Version &clientVersion,
          bool debug,
          bool bindGeneratesResource,
          bool clientArraysEnabled,
          bool robustResourceInit,
          bool programBinaryCacheEnabled);
    ~State();

    int id() const { return mID; }

    void initialize(Context *context);
    void reset(const Context *context);

    // Getters
    ContextID getContextID() const { return mContext; }
    EGLenum getClientType() const { return mClientType; }
    GLint getClientMajorVersion() const { return mClientVersion.major; }
    GLint getClientMinorVersion() const { return mClientVersion.minor; }
    const Version &getClientVersion() const { return mClientVersion; }
    const Caps &getCaps() const { return mCaps; }
    const TextureCapsMap &getTextureCaps() const { return mTextureCaps; }
    const Extensions &getExtensions() const { return mExtensions; }
    const Limitations &getLimitations() const { return mLimitations; }

    bool isWebGL() const { return mExtensions.webglCompatibility; }

    bool isWebGL1() const { return (isWebGL() && mClientVersion.major == 2); }

    const TextureCaps &getTextureCap(GLenum internalFormat) const
    {
        return mTextureCaps.get(internalFormat);
    }

    // State chunk getters
    const RasterizerState &getRasterizerState() const;
    const BlendState &getBlendState() const { return mBlend; }
    const DepthStencilState &getDepthStencilState() const;

    // Clear behavior setters & state parameter block generation function
    void setColorClearValue(float red, float green, float blue, float alpha);
    void setDepthClearValue(float depth);
    void setStencilClearValue(int stencil);

    const ColorF &getColorClearValue() const { return mColorClearValue; }
    float getDepthClearValue() const { return mDepthClearValue; }
    int getStencilClearValue() const { return mStencilClearValue; }

    // Write mask manipulation
    void setColorMask(bool red, bool green, bool blue, bool alpha);
    void setDepthMask(bool mask);

    // Discard toggle & query
    bool isRasterizerDiscardEnabled() const { return mRasterizer.rasterizerDiscard; }
    void setRasterizerDiscard(bool enabled);

    // Primitive restart
    bool isPrimitiveRestartEnabled() const { return mPrimitiveRestart; }
    void setPrimitiveRestart(bool enabled);

    // Face culling state manipulation
    bool isCullFaceEnabled() const { return mRasterizer.cullFace; }
    void setCullFace(bool enabled);
    void setCullMode(CullFaceMode mode);
    void setFrontFace(GLenum front);

    // Depth test state manipulation
    bool isDepthTestEnabled() const { return mDepthStencil.depthTest; }
    void setDepthTest(bool enabled);
    void setDepthFunc(GLenum depthFunc);
    void setDepthRange(float zNear, float zFar);
    float getNearPlane() const { return mNearZ; }
    float getFarPlane() const { return mFarZ; }

    // Blend state manipulation
    bool isBlendEnabled() const { return mBlend.blend; }
    void setBlend(bool enabled);
    void setBlendFactors(GLenum sourceRGB, GLenum destRGB, GLenum sourceAlpha, GLenum destAlpha);
    void setBlendColor(float red, float green, float blue, float alpha);
    void setBlendEquation(GLenum rgbEquation, GLenum alphaEquation);
    const ColorF &getBlendColor() const { return mBlendColor; }

    // Stencil state maniupulation
    bool isStencilTestEnabled() const { return mDepthStencil.stencilTest; }
    void setStencilTest(bool enabled);
    void setStencilParams(GLenum stencilFunc, GLint stencilRef, GLuint stencilMask);
    void setStencilBackParams(GLenum stencilBackFunc, GLint stencilBackRef, GLuint stencilBackMask);
    void setStencilWritemask(GLuint stencilWritemask);
    void setStencilBackWritemask(GLuint stencilBackWritemask);
    void setStencilOperations(GLenum stencilFail,
                              GLenum stencilPassDepthFail,
                              GLenum stencilPassDepthPass);
    void setStencilBackOperations(GLenum stencilBackFail,
                                  GLenum stencilBackPassDepthFail,
                                  GLenum stencilBackPassDepthPass);
    GLint getStencilRef() const { return mStencilRef; }
    GLint getStencilBackRef() const { return mStencilBackRef; }

    // Depth bias/polygon offset state manipulation
    bool isPolygonOffsetFillEnabled() const { return mRasterizer.polygonOffsetFill; }
    void setPolygonOffsetFill(bool enabled);
    void setPolygonOffsetParams(GLfloat factor, GLfloat units);

    // Multisample coverage state manipulation
    bool isSampleAlphaToCoverageEnabled() const { return mBlend.sampleAlphaToCoverage; }
    void setSampleAlphaToCoverage(bool enabled);
    bool isSampleCoverageEnabled() const { return mSampleCoverage; }
    void setSampleCoverage(bool enabled);
    void setSampleCoverageParams(GLclampf value, bool invert);
    GLclampf getSampleCoverageValue() const { return mSampleCoverageValue; }
    bool getSampleCoverageInvert() const { return mSampleCoverageInvert; }

    // Multisample mask state manipulation.
    bool isSampleMaskEnabled() const { return mSampleMask; }
    void setSampleMaskEnabled(bool enabled);
    void setSampleMaskParams(GLuint maskNumber, GLbitfield mask);
    GLbitfield getSampleMaskWord(GLuint maskNumber) const
    {
        ASSERT(maskNumber < mMaxSampleMaskWords);
        return mSampleMaskValues[maskNumber];
    }
    GLuint getMaxSampleMaskWords() const { return mMaxSampleMaskWords; }

    // Multisampling/alpha to one manipulation.
    void setSampleAlphaToOne(bool enabled);
    bool isSampleAlphaToOneEnabled() const { return mSampleAlphaToOne; }
    void setMultisampling(bool enabled);
    bool isMultisamplingEnabled() const { return mMultiSampling; }

    // Scissor test state toggle & query
    bool isScissorTestEnabled() const { return mScissorTest; }
    void setScissorTest(bool enabled);
    void setScissorParams(GLint x, GLint y, GLsizei width, GLsizei height);
    const Rectangle &getScissor() const { return mScissor; }

    // Dither state toggle & query
    bool isDitherEnabled() const { return mBlend.dither; }
    void setDither(bool enabled);

    // Generic state toggle & query
    void setEnableFeature(GLenum feature, bool enabled);
    bool getEnableFeature(GLenum feature) const;

    // Line width state setter
    void setLineWidth(GLfloat width);
    float getLineWidth() const { return mLineWidth; }

    // Hint setters
    void setGenerateMipmapHint(GLenum hint);
    void setFragmentShaderDerivativeHint(GLenum hint);

    // GL_CHROMIUM_bind_generates_resource
    bool isBindGeneratesResourceEnabled() const { return mBindGeneratesResource; }

    // GL_ANGLE_client_arrays
    bool areClientArraysEnabled() const { return mClientArraysEnabled; }

    // Viewport state setter/getter
    void setViewportParams(GLint x, GLint y, GLsizei width, GLsizei height);
    const Rectangle &getViewport() const { return mViewport; }

    // Texture binding & active texture unit manipulation
    void setActiveSampler(unsigned int active);
    unsigned int getActiveSampler() const { return static_cast<unsigned int>(mActiveSampler); }

    void setSamplerTexture(const Context *context, TextureType type, Texture *texture);
    Texture *getTargetTexture(TextureType type) const;

    Texture *getSamplerTexture(unsigned int sampler, TextureType type) const
    {
        ASSERT(sampler < mSamplerTextures[type].size());
        return mSamplerTextures[type][sampler].get();
    }

    TextureID getSamplerTextureId(unsigned int sampler, TextureType type) const;
    void detachTexture(const Context *context, const TextureMap &zeroTextures, TextureID texture);
    void initializeZeroTextures(const Context *context, const TextureMap &zeroTextures);

    void invalidateTexture(TextureType type);

    // Sampler object binding manipulation
    void setSamplerBinding(const Context *context, GLuint textureUnit, Sampler *sampler);
    SamplerID getSamplerId(GLuint textureUnit) const
    {
        ASSERT(textureUnit < mSamplers.size());
        return mSamplers[textureUnit].id();
    }

    Sampler *getSampler(GLuint textureUnit) const { return mSamplers[textureUnit].get(); }

    using SamplerBindingVector = std::vector<BindingPointer<Sampler>>;
    const SamplerBindingVector &getSamplers() const { return mSamplers; }

    void detachSampler(const Context *context, SamplerID sampler);

    // Renderbuffer binding manipulation
    void setRenderbufferBinding(const Context *context, Renderbuffer *renderbuffer);
    RenderbufferID getRenderbufferId() const { return mRenderbuffer.id(); }
    Renderbuffer *getCurrentRenderbuffer() const { return mRenderbuffer.get(); }
    void detachRenderbuffer(const Context *context, RenderbufferID renderbuffer);

    // Framebuffer binding manipulation
    void setReadFramebufferBinding(Framebuffer *framebuffer);
    void setDrawFramebufferBinding(Framebuffer *framebuffer);
    Framebuffer *getTargetFramebuffer(GLenum target) const;
    Framebuffer *getReadFramebuffer() const { return mReadFramebuffer; }
    Framebuffer *getDrawFramebuffer() const { return mDrawFramebuffer; }

    bool removeReadFramebufferBinding(FramebufferID framebuffer);
    bool removeDrawFramebufferBinding(FramebufferID framebuffer);

    // Vertex array object binding manipulation
    void setVertexArrayBinding(const Context *context, VertexArray *vertexArray);
    bool removeVertexArrayBinding(const Context *context, VertexArrayID vertexArray);
    VertexArrayID getVertexArrayId() const;

    VertexArray *getVertexArray() const
    {
        ASSERT(mVertexArray != nullptr);
        return mVertexArray;
    }

    // Program binding manipulation
    angle::Result setProgram(const Context *context, Program *newProgram);

    Program *getProgram() const
    {
        ASSERT(!mProgram || !mProgram->isLinking());
        return mProgram;
    }

    Program *getLinkedProgram(const Context *context) const
    {
        if (mProgram)
        {
            mProgram->resolveLink(context);
        }
        return mProgram;
    }

    // Transform feedback object (not buffer) binding manipulation
    void setTransformFeedbackBinding(const Context *context, TransformFeedback *transformFeedback);
    TransformFeedback *getCurrentTransformFeedback() const { return mTransformFeedback.get(); }

    ANGLE_INLINE bool isTransformFeedbackActive() const
    {
        TransformFeedback *curTransformFeedback = mTransformFeedback.get();
        return curTransformFeedback && curTransformFeedback->isActive();
    }
    ANGLE_INLINE bool isTransformFeedbackActiveUnpaused() const
    {
        TransformFeedback *curTransformFeedback = mTransformFeedback.get();
        return curTransformFeedback && curTransformFeedback->isActive() &&
               !curTransformFeedback->isPaused();
    }

    bool removeTransformFeedbackBinding(const Context *context,
                                        TransformFeedbackID transformFeedback);

    // Query binding manipulation
    bool isQueryActive(QueryType type) const;
    bool isQueryActive(Query *query) const;
    void setActiveQuery(const Context *context, QueryType type, Query *query);
    QueryID getActiveQueryId(QueryType type) const;
    Query *getActiveQuery(QueryType type) const;

    // Program Pipeline binding manipulation
    void setProgramPipelineBinding(const Context *context, ProgramPipeline *pipeline);
    void detachProgramPipeline(const Context *context, ProgramPipelineID pipeline);

    //// Typed buffer binding point manipulation ////
    ANGLE_INLINE void setBufferBinding(const Context *context, BufferBinding target, Buffer *buffer)
    {
        (this->*(kBufferSetters[target]))(context, buffer);
    }

    ANGLE_INLINE Buffer *getTargetBuffer(BufferBinding target) const
    {
        switch (target)
        {
            case BufferBinding::ElementArray:
                return getVertexArray()->getElementArrayBuffer();
            default:
                return mBoundBuffers[target].get();
        }
    }

    angle::Result setIndexedBufferBinding(const Context *context,
                                          BufferBinding target,
                                          GLuint index,
                                          Buffer *buffer,
                                          GLintptr offset,
                                          GLsizeiptr size);

    size_t getAtomicCounterBufferCount() const { return mAtomicCounterBuffers.size(); }

    const OffsetBindingPointer<Buffer> &getIndexedUniformBuffer(size_t index) const;
    const OffsetBindingPointer<Buffer> &getIndexedAtomicCounterBuffer(size_t index) const;
    const OffsetBindingPointer<Buffer> &getIndexedShaderStorageBuffer(size_t index) const;

    // Detach a buffer from all bindings
    angle::Result detachBuffer(Context *context, const Buffer *buffer);

    // Vertex attrib manipulation
    void setEnableVertexAttribArray(unsigned int attribNum, bool enabled);
    void setVertexAttribf(GLuint index, const GLfloat values[4]);
    void setVertexAttribu(GLuint index, const GLuint values[4]);
    void setVertexAttribi(GLuint index, const GLint values[4]);

    ANGLE_INLINE void setVertexAttribPointer(const Context *context,
                                             unsigned int attribNum,
                                             Buffer *boundBuffer,
                                             GLint size,
                                             VertexAttribType type,
                                             bool normalized,
                                             GLsizei stride,
                                             const void *pointer)
    {
        mVertexArray->setVertexAttribPointer(context, attribNum, boundBuffer, size, type,
                                             normalized, stride, pointer);
        mDirtyObjects.set(DIRTY_OBJECT_VERTEX_ARRAY);
    }

    ANGLE_INLINE void setVertexAttribIPointer(const Context *context,
                                              unsigned int attribNum,
                                              Buffer *boundBuffer,
                                              GLint size,
                                              VertexAttribType type,
                                              GLsizei stride,
                                              const void *pointer)
    {
        mVertexArray->setVertexAttribIPointer(context, attribNum, boundBuffer, size, type, stride,
                                              pointer);
        mDirtyObjects.set(DIRTY_OBJECT_VERTEX_ARRAY);
    }

    void setVertexAttribDivisor(const Context *context, GLuint index, GLuint divisor);
    const VertexAttribCurrentValueData &getVertexAttribCurrentValue(size_t attribNum) const
    {
        ASSERT(attribNum < mVertexAttribCurrentValues.size());
        return mVertexAttribCurrentValues[attribNum];
    }

    const std::vector<VertexAttribCurrentValueData> &getVertexAttribCurrentValues() const
    {
        return mVertexAttribCurrentValues;
    }

    const void *getVertexAttribPointer(unsigned int attribNum) const;

    void bindVertexBuffer(const Context *context,
                          GLuint bindingIndex,
                          Buffer *boundBuffer,
                          GLintptr offset,
                          GLsizei stride);
    void setVertexAttribFormat(GLuint attribIndex,
                               GLint size,
                               VertexAttribType type,
                               bool normalized,
                               bool pureInteger,
                               GLuint relativeOffset);

    void setVertexAttribBinding(const Context *context, GLuint attribIndex, GLuint bindingIndex)
    {
        mVertexArray->setVertexAttribBinding(context, attribIndex, bindingIndex);
        mDirtyObjects.set(DIRTY_OBJECT_VERTEX_ARRAY);
    }

    void setVertexBindingDivisor(GLuint bindingIndex, GLuint divisor);

    // Pixel pack state manipulation
    void setPackAlignment(GLint alignment);
    GLint getPackAlignment() const { return mPack.alignment; }
    void setPackReverseRowOrder(bool reverseRowOrder);
    bool getPackReverseRowOrder() const { return mPack.reverseRowOrder; }
    void setPackRowLength(GLint rowLength);
    GLint getPackRowLength() const { return mPack.rowLength; }
    void setPackSkipRows(GLint skipRows);
    GLint getPackSkipRows() const { return mPack.skipRows; }
    void setPackSkipPixels(GLint skipPixels);
    GLint getPackSkipPixels() const { return mPack.skipPixels; }
    const PixelPackState &getPackState() const { return mPack; }
    PixelPackState &getPackState() { return mPack; }

    // Pixel unpack state manipulation
    void setUnpackAlignment(GLint alignment);
    GLint getUnpackAlignment() const { return mUnpack.alignment; }
    void setUnpackRowLength(GLint rowLength);
    GLint getUnpackRowLength() const { return mUnpack.rowLength; }
    void setUnpackImageHeight(GLint imageHeight);
    GLint getUnpackImageHeight() const { return mUnpack.imageHeight; }
    void setUnpackSkipImages(GLint skipImages);
    GLint getUnpackSkipImages() const { return mUnpack.skipImages; }
    void setUnpackSkipRows(GLint skipRows);
    GLint getUnpackSkipRows() const { return mUnpack.skipRows; }
    void setUnpackSkipPixels(GLint skipPixels);
    GLint getUnpackSkipPixels() const { return mUnpack.skipPixels; }
    const PixelUnpackState &getUnpackState() const { return mUnpack; }
    PixelUnpackState &getUnpackState() { return mUnpack; }

    // Debug state
    const Debug &getDebug() const { return mDebug; }
    Debug &getDebug() { return mDebug; }

    // CHROMIUM_framebuffer_mixed_samples coverage modulation
    void setCoverageModulation(GLenum components);
    GLenum getCoverageModulation() const { return mCoverageModulation; }

    // CHROMIUM_path_rendering
    void loadPathRenderingMatrix(GLenum matrixMode, const GLfloat *matrix);
    const GLfloat *getPathRenderingMatrix(GLenum which) const;
    void setPathStencilFunc(GLenum func, GLint ref, GLuint mask);
    GLenum getPathStencilFunc() const { return mPathStencilFunc; }
    GLint getPathStencilRef() const { return mPathStencilRef; }
    GLuint getPathStencilMask() const { return mPathStencilMask; }

    // GL_EXT_sRGB_write_control
    void setFramebufferSRGB(bool sRGB);
    bool getFramebufferSRGB() const { return mFramebufferSRGB; }

    // GL_KHR_parallel_shader_compile
    void setMaxShaderCompilerThreads(GLuint count);
    GLuint getMaxShaderCompilerThreads() const { return mMaxShaderCompilerThreads; }

    // State query functions
    void getBooleanv(GLenum pname, GLboolean *params);
    void getFloatv(GLenum pname, GLfloat *params);
    angle::Result getIntegerv(const Context *context, GLenum pname, GLint *params);
    void getPointerv(const Context *context, GLenum pname, void **params) const;
    void getIntegeri_v(GLenum target, GLuint index, GLint *data);
    void getInteger64i_v(GLenum target, GLuint index, GLint64 *data);
    void getBooleani_v(GLenum target, GLuint index, GLboolean *data);

    bool isRobustResourceInitEnabled() const { return mRobustResourceInit; }

    // Sets the dirty bit for the program executable.
    angle::Result onProgramExecutableChange(const Context *context, Program *program);

    enum DirtyBitType
    {
        // Note: process draw framebuffer binding first, so that other dirty bits whose effect
        // depend on the current draw framebuffer are not processed while the old framebuffer is
        // still bound.
        DIRTY_BIT_DRAW_FRAMEBUFFER_BINDING,
        DIRTY_BIT_READ_FRAMEBUFFER_BINDING,
        DIRTY_BIT_SCISSOR_TEST_ENABLED,
        DIRTY_BIT_SCISSOR,
        DIRTY_BIT_VIEWPORT,
        DIRTY_BIT_DEPTH_RANGE,
        DIRTY_BIT_BLEND_ENABLED,
        DIRTY_BIT_BLEND_COLOR,
        DIRTY_BIT_BLEND_FUNCS,
        DIRTY_BIT_BLEND_EQUATIONS,
        DIRTY_BIT_COLOR_MASK,
        DIRTY_BIT_SAMPLE_ALPHA_TO_COVERAGE_ENABLED,
        DIRTY_BIT_SAMPLE_COVERAGE_ENABLED,
        DIRTY_BIT_SAMPLE_COVERAGE,
        DIRTY_BIT_SAMPLE_MASK_ENABLED,
        DIRTY_BIT_SAMPLE_MASK,
        DIRTY_BIT_DEPTH_TEST_ENABLED,
        DIRTY_BIT_DEPTH_FUNC,
        DIRTY_BIT_DEPTH_MASK,
        DIRTY_BIT_STENCIL_TEST_ENABLED,
        DIRTY_BIT_STENCIL_FUNCS_FRONT,
        DIRTY_BIT_STENCIL_FUNCS_BACK,
        DIRTY_BIT_STENCIL_OPS_FRONT,
        DIRTY_BIT_STENCIL_OPS_BACK,
        DIRTY_BIT_STENCIL_WRITEMASK_FRONT,
        DIRTY_BIT_STENCIL_WRITEMASK_BACK,
        DIRTY_BIT_CULL_FACE_ENABLED,
        DIRTY_BIT_CULL_FACE,
        DIRTY_BIT_FRONT_FACE,
        DIRTY_BIT_POLYGON_OFFSET_FILL_ENABLED,
        DIRTY_BIT_POLYGON_OFFSET,
        DIRTY_BIT_RASTERIZER_DISCARD_ENABLED,
        DIRTY_BIT_LINE_WIDTH,
        DIRTY_BIT_PRIMITIVE_RESTART_ENABLED,
        DIRTY_BIT_CLEAR_COLOR,
        DIRTY_BIT_CLEAR_DEPTH,
        DIRTY_BIT_CLEAR_STENCIL,
        DIRTY_BIT_UNPACK_STATE,
        DIRTY_BIT_UNPACK_BUFFER_BINDING,
        DIRTY_BIT_PACK_STATE,
        DIRTY_BIT_PACK_BUFFER_BINDING,
        DIRTY_BIT_DITHER_ENABLED,
        DIRTY_BIT_RENDERBUFFER_BINDING,
        DIRTY_BIT_VERTEX_ARRAY_BINDING,
        DIRTY_BIT_DRAW_INDIRECT_BUFFER_BINDING,
        DIRTY_BIT_DISPATCH_INDIRECT_BUFFER_BINDING,
        // TODO(jmadill): Fine-grained dirty bits for each index.
        DIRTY_BIT_PROGRAM_BINDING,
        DIRTY_BIT_PROGRAM_EXECUTABLE,
        // TODO(jmadill): Fine-grained dirty bits for each texture/sampler.
        DIRTY_BIT_TEXTURE_BINDINGS,
        DIRTY_BIT_SAMPLER_BINDINGS,
        DIRTY_BIT_IMAGE_BINDINGS,
        DIRTY_BIT_TRANSFORM_FEEDBACK_BINDING,
        DIRTY_BIT_UNIFORM_BUFFER_BINDINGS,
        DIRTY_BIT_SHADER_STORAGE_BUFFER_BINDING,
        DIRTY_BIT_ATOMIC_COUNTER_BUFFER_BINDING,
        DIRTY_BIT_MULTISAMPLING,
        DIRTY_BIT_SAMPLE_ALPHA_TO_ONE,
        DIRTY_BIT_COVERAGE_MODULATION,  // CHROMIUM_framebuffer_mixed_samples
        DIRTY_BIT_PATH_RENDERING,
        DIRTY_BIT_FRAMEBUFFER_SRGB,  // GL_EXT_sRGB_write_control
        DIRTY_BIT_CURRENT_VALUES,
        DIRTY_BIT_PROVOKING_VERTEX,
        DIRTY_BIT_EXTENDED,  // clip distances, mipmap generation hint, derivative hint.
        DIRTY_BIT_INVALID,
        DIRTY_BIT_MAX = DIRTY_BIT_INVALID,
    };

    static_assert(DIRTY_BIT_MAX <= 64, "State dirty bits must be capped at 64");

    // TODO(jmadill): Consider storing dirty objects in a list instead of by binding.
    enum DirtyObjectType
    {
        DIRTY_OBJECT_TEXTURES_INIT,
        DIRTY_OBJECT_IMAGES_INIT,
        DIRTY_OBJECT_READ_ATTACHMENTS,
        DIRTY_OBJECT_DRAW_ATTACHMENTS,
        DIRTY_OBJECT_READ_FRAMEBUFFER,
        DIRTY_OBJECT_DRAW_FRAMEBUFFER,
        DIRTY_OBJECT_VERTEX_ARRAY,
        DIRTY_OBJECT_TEXTURES,  // Top-level dirty bit. Also see mDirtyTextures.
        DIRTY_OBJECT_IMAGES,    // Top-level dirty bit. Also see mDirtyImages.
        DIRTY_OBJECT_SAMPLERS,  // Top-level dirty bit. Also see mDirtySamplers.
        DIRTY_OBJECT_PROGRAM,
        DIRTY_OBJECT_UNKNOWN,
        DIRTY_OBJECT_MAX = DIRTY_OBJECT_UNKNOWN,
    };

    using DirtyBits = angle::BitSet<DIRTY_BIT_MAX>;
    const DirtyBits &getDirtyBits() const { return mDirtyBits; }
    void clearDirtyBits() { mDirtyBits.reset(); }
    void clearDirtyBits(const DirtyBits &bitset) { mDirtyBits &= ~bitset; }
    void setAllDirtyBits()
    {
        mDirtyBits.set();
        mDirtyCurrentValues.set();
    }

    using DirtyObjects = angle::BitSet<DIRTY_OBJECT_MAX>;
    void clearDirtyObjects() { mDirtyObjects.reset(); }
    void setAllDirtyObjects() { mDirtyObjects.set(); }
    angle::Result syncDirtyObjects(const Context *context, const DirtyObjects &bitset);
    angle::Result syncDirtyObject(const Context *context, GLenum target);
    void setObjectDirty(GLenum target);
    void setTextureDirty(size_t textureUnitIndex);
    void setSamplerDirty(size_t samplerIndex);

    ANGLE_INLINE void setReadFramebufferDirty()
    {
        mDirtyObjects.set(DIRTY_OBJECT_READ_FRAMEBUFFER);
        mDirtyObjects.set(DIRTY_OBJECT_READ_ATTACHMENTS);
    }

    ANGLE_INLINE void setDrawFramebufferDirty()
    {
        mDirtyObjects.set(DIRTY_OBJECT_DRAW_FRAMEBUFFER);
        mDirtyObjects.set(DIRTY_OBJECT_DRAW_ATTACHMENTS);
    }

    // This actually clears the current value dirty bits.
    // TODO(jmadill): Pass mutable dirty bits into Impl.
    AttributesMask getAndResetDirtyCurrentValues() const;

    void setImageUnit(const Context *context,
                      size_t unit,
                      Texture *texture,
                      GLint level,
                      GLboolean layered,
                      GLint layer,
                      GLenum access,
                      GLenum format);

    const ImageUnit &getImageUnit(size_t unit) const { return mImageUnits[unit]; }
    const ActiveTexturePointerArray &getActiveTexturesCache() const { return mActiveTexturesCache; }
    ComponentTypeMask getCurrentValuesTypeMask() const { return mCurrentValuesTypeMask; }

    // "onActiveTextureChange" is called when a texture binding changes.
    void onActiveTextureChange(const Context *context, size_t textureUnit);

    // "onActiveTextureStateChange" calls when the Texture itself changed but the binding did not.
    void onActiveTextureStateChange(const Context *context, size_t textureUnit);

    void onImageStateChange(const Context *context, size_t unit);

    void onUniformBufferStateChange(size_t uniformBufferIndex);

    bool isCurrentTransformFeedback(const TransformFeedback *tf) const
    {
        return tf == mTransformFeedback.get();
    }
    bool isCurrentVertexArray(const VertexArray *va) const { return va == mVertexArray; }

    GLES1State &gles1() { return mGLES1State; }
    const GLES1State &gles1() const { return mGLES1State; }

    // Helpers for setting bound buffers. They should all have the same signature.
    // Not meant to be called externally. Used for local helpers in State.cpp.
    template <BufferBinding Target>
    void setGenericBufferBindingWithBit(const Context *context, Buffer *buffer);

    template <BufferBinding Target>
    void setGenericBufferBinding(const Context *context, Buffer *buffer);

    using BufferBindingSetter = void (State::*)(const Context *, Buffer *);

    ANGLE_INLINE bool validateSamplerFormats() const
    {
        return (mTexturesIncompatibleWithSamplers & mProgram->getActiveSamplersMask()).none();
    }

    ProvokingVertexConvention getProvokingVertex() const { return mProvokingVertex; }
    void setProvokingVertex(ProvokingVertexConvention val)
    {
        mDirtyBits.set(State::DIRTY_BIT_PROVOKING_VERTEX);
        mProvokingVertex = val;
    }

    using ClipDistanceEnableBits = angle::BitSet32<IMPLEMENTATION_MAX_CLIP_DISTANCES>;
    const ClipDistanceEnableBits &getEnabledClipDistances() const { return mClipDistancesEnabled; }
    void setClipDistanceEnable(int idx, bool enable);

    const OverlayType *getOverlay() const { return mOverlay; }

  private:
    friend class Context;

    void unsetActiveTextures(ActiveTextureMask textureMask);
    void updateActiveTexture(const Context *context, size_t textureIndex, Texture *texture);
    void updateActiveTextureState(const Context *context,
                                  size_t textureIndex,
                                  const Sampler *sampler,
                                  Texture *texture);

    // Functions to synchronize dirty states
    angle::Result syncTexturesInit(const Context *context);
    angle::Result syncImagesInit(const Context *context);
    angle::Result syncReadAttachments(const Context *context);
    angle::Result syncDrawAttachments(const Context *context);
    angle::Result syncReadFramebuffer(const Context *context);
    angle::Result syncDrawFramebuffer(const Context *context);
    angle::Result syncVertexArray(const Context *context);
    angle::Result syncTextures(const Context *context);
    angle::Result syncImages(const Context *context);
    angle::Result syncSamplers(const Context *context);
    angle::Result syncProgram(const Context *context);

    using DirtyObjectHandler = angle::Result (State::*)(const Context *context);
    static constexpr DirtyObjectHandler kDirtyObjectHandlers[DIRTY_OBJECT_MAX] = {
        &State::syncTexturesInit,    &State::syncImagesInit,      &State::syncReadAttachments,
        &State::syncDrawAttachments, &State::syncReadFramebuffer, &State::syncDrawFramebuffer,
        &State::syncVertexArray,     &State::syncTextures,        &State::syncImages,
        &State::syncSamplers,        &State::syncProgram,
    };

    // Robust init must happen before Framebuffer init for the Vulkan back-end.
    static_assert(DIRTY_OBJECT_TEXTURES_INIT < DIRTY_OBJECT_DRAW_FRAMEBUFFER, "init order");
    static_assert(DIRTY_OBJECT_IMAGES_INIT < DIRTY_OBJECT_DRAW_FRAMEBUFFER, "init order");
    static_assert(DIRTY_OBJECT_DRAW_ATTACHMENTS < DIRTY_OBJECT_DRAW_FRAMEBUFFER, "init order");
    static_assert(DIRTY_OBJECT_READ_ATTACHMENTS < DIRTY_OBJECT_READ_FRAMEBUFFER, "init order");

    static_assert(DIRTY_OBJECT_TEXTURES_INIT == 0, "check DIRTY_OBJECT_TEXTURES_INIT index");
    static_assert(DIRTY_OBJECT_IMAGES_INIT == 1, "check DIRTY_OBJECT_IMAGES_INIT index");
    static_assert(DIRTY_OBJECT_READ_ATTACHMENTS == 2, "check DIRTY_OBJECT_READ_ATTACHMENTS index");
    static_assert(DIRTY_OBJECT_DRAW_ATTACHMENTS == 3, "check DIRTY_OBJECT_DRAW_ATTACHMENTS index");
    static_assert(DIRTY_OBJECT_READ_FRAMEBUFFER == 4, "check DIRTY_OBJECT_READ_FRAMEBUFFER index");
    static_assert(DIRTY_OBJECT_DRAW_FRAMEBUFFER == 5, "check DIRTY_OBJECT_DRAW_FRAMEBUFFER index");
    static_assert(DIRTY_OBJECT_VERTEX_ARRAY == 6, "check DIRTY_OBJECT_VERTEX_ARRAY index");
    static_assert(DIRTY_OBJECT_TEXTURES == 7, "check DIRTY_OBJECT_TEXTURES index");
    static_assert(DIRTY_OBJECT_IMAGES == 8, "check DIRTY_OBJECT_IMAGES index");
    static_assert(DIRTY_OBJECT_SAMPLERS == 9, "check DIRTY_OBJECT_SAMPLERS index");
    static_assert(DIRTY_OBJECT_PROGRAM == 10, "check DIRTY_OBJECT_PROGRAM index");

    // Dispatch table for buffer update functions.
    static const angle::PackedEnumMap<BufferBinding, BufferBindingSetter> kBufferSetters;

    int mID;

    EGLenum mClientType;
    Version mClientVersion;
    ContextID mContext;

    // Caps to use for validation
    Caps mCaps;
    TextureCapsMap mTextureCaps;
    Extensions mExtensions;
    Limitations mLimitations;

    // Resource managers.
    BufferManager *mBufferManager;
    ShaderProgramManager *mShaderProgramManager;
    TextureManager *mTextureManager;
    RenderbufferManager *mRenderbufferManager;
    SamplerManager *mSamplerManager;
    SyncManager *mSyncManager;
    PathManager *mPathManager;
    FramebufferManager *mFramebufferManager;
    ProgramPipelineManager *mProgramPipelineManager;
    MemoryObjectManager *mMemoryObjectManager;
    SemaphoreManager *mSemaphoreManager;

    // Cached values from Context's caps
    GLuint mMaxDrawBuffers;
    GLuint mMaxCombinedTextureImageUnits;

    ColorF mColorClearValue;
    GLfloat mDepthClearValue;
    int mStencilClearValue;

    RasterizerState mRasterizer;
    bool mScissorTest;
    Rectangle mScissor;

    BlendState mBlend;
    ColorF mBlendColor;
    bool mSampleCoverage;
    GLfloat mSampleCoverageValue;
    bool mSampleCoverageInvert;
    bool mSampleMask;
    GLuint mMaxSampleMaskWords;
    std::array<GLbitfield, MAX_SAMPLE_MASK_WORDS> mSampleMaskValues;

    DepthStencilState mDepthStencil;
    GLint mStencilRef;
    GLint mStencilBackRef;

    GLfloat mLineWidth;

    GLenum mGenerateMipmapHint;
    GLenum mFragmentShaderDerivativeHint;

    const bool mBindGeneratesResource;
    const bool mClientArraysEnabled;

    Rectangle mViewport;
    float mNearZ;
    float mFarZ;

    Framebuffer *mReadFramebuffer;
    Framebuffer *mDrawFramebuffer;
    BindingPointer<Renderbuffer> mRenderbuffer;
    Program *mProgram;
    BindingPointer<ProgramPipeline> mProgramPipeline;

    // GL_ANGLE_provoking_vertex
    ProvokingVertexConvention mProvokingVertex;

    using VertexAttribVector = std::vector<VertexAttribCurrentValueData>;
    VertexAttribVector mVertexAttribCurrentValues;  // From glVertexAttrib
    VertexArray *mVertexArray;
    ComponentTypeMask mCurrentValuesTypeMask;

    // Texture and sampler bindings
    size_t mActiveSampler;  // Active texture unit selector - GL_TEXTURE0

    using TextureBindingVector = std::vector<BindingPointer<Texture>>;
    using TextureBindingMap    = angle::PackedEnumMap<TextureType, TextureBindingVector>;
    TextureBindingMap mSamplerTextures;

    // Texture Completeness Caching
    // ----------------------------
    // The texture completeness cache uses dirty bits to avoid having to scan the list of textures
    // each draw call. This gl::State class implements angle::Observer interface. When subject
    // Textures have state changes, messages reach 'State' (also any observing Framebuffers) via the
    // onSubjectStateChange method (above). This then invalidates the completeness cache.
    //
    // Note this requires that we also invalidate the completeness cache manually on events like
    // re-binding textures/samplers or a change in the program. For more information see the
    // Observer.h header and the design doc linked there.

    // A cache of complete textures. nullptr indicates unbound or incomplete.
    // Don't use BindingPointer because this cache is only valid within a draw call.
    // Also stores a notification channel to the texture itself to handle texture change events.
    ActiveTexturePointerArray mActiveTexturesCache;
    std::vector<angle::ObserverBinding> mCompleteTextureBindings;

    ActiveTextureMask mTexturesIncompatibleWithSamplers;

    SamplerBindingVector mSamplers;

    // It would be nice to merge the image and observer binding. Same for textures.
    std::vector<ImageUnit> mImageUnits;

    using ActiveQueryMap = angle::PackedEnumMap<QueryType, BindingPointer<Query>>;
    ActiveQueryMap mActiveQueries;

    // Stores the currently bound buffer for each binding point. It has an entry for the element
    // array buffer but it should not be used. Instead this bind point is owned by the current
    // vertex array object.
    using BoundBufferMap = angle::PackedEnumMap<BufferBinding, BindingPointer<Buffer>>;
    BoundBufferMap mBoundBuffers;

    using BufferVector = std::vector<OffsetBindingPointer<Buffer>>;
    BufferVector mUniformBuffers;
    BufferVector mAtomicCounterBuffers;
    BufferVector mShaderStorageBuffers;

    BindingPointer<TransformFeedback> mTransformFeedback;

    PixelUnpackState mUnpack;
    PixelPackState mPack;

    bool mPrimitiveRestart;

    Debug mDebug;

    bool mMultiSampling;
    bool mSampleAlphaToOne;

    GLenum mCoverageModulation;

    // CHROMIUM_path_rendering
    GLfloat mPathMatrixMV[16];
    GLfloat mPathMatrixProj[16];
    GLenum mPathStencilFunc;
    GLint mPathStencilRef;
    GLuint mPathStencilMask;

    // GL_EXT_sRGB_write_control
    bool mFramebufferSRGB;

    // GL_ANGLE_robust_resource_intialization
    const bool mRobustResourceInit;

    // GL_ANGLE_program_cache_control
    const bool mProgramBinaryCacheEnabled;

    // GL_KHR_parallel_shader_compile
    GLuint mMaxShaderCompilerThreads;

    // GL_APPLE_clip_distance/GL_EXT_clip_cull_distance
    ClipDistanceEnableBits mClipDistancesEnabled;

    // GLES1 emulation: state specific to GLES1
    GLES1State mGLES1State;

    DirtyBits mDirtyBits;
    DirtyObjects mDirtyObjects;
    mutable AttributesMask mDirtyCurrentValues;
    ActiveTextureMask mDirtyTextures;
    ActiveTextureMask mDirtySamplers;
    ImageUnitMask mDirtyImages;

    // The Overlay object, used by the backend to render the overlay.
    const OverlayType *mOverlay;
};

ANGLE_INLINE angle::Result State::syncDirtyObjects(const Context *context,
                                                   const DirtyObjects &bitset)
{
    const DirtyObjects &dirtyObjects = mDirtyObjects & bitset;

    for (size_t dirtyObject : dirtyObjects)
    {
        ANGLE_TRY((this->*kDirtyObjectHandlers[dirtyObject])(context));
    }

    mDirtyObjects &= ~dirtyObjects;
    return angle::Result::Continue;
}

}  // namespace gl

#endif  // LIBANGLE_STATE_H_
