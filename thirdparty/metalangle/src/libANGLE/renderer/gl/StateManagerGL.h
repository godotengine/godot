//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// StateManagerGL.h: Defines a class for caching applied OpenGL state

#ifndef LIBANGLE_RENDERER_GL_STATEMANAGERGL_H_
#define LIBANGLE_RENDERER_GL_STATEMANAGERGL_H_

#include "common/debug.h"
#include "libANGLE/Error.h"
#include "libANGLE/State.h"
#include "libANGLE/angletypes.h"
#include "libANGLE/renderer/gl/functionsgl_typedefs.h"
#include "platform/FeaturesGL.h"

#include <map>

namespace gl
{
struct Caps;
class FramebufferState;
class State;
}  // namespace gl

namespace rx
{

class FramebufferGL;
class FunctionsGL;
class TransformFeedbackGL;
class VertexArrayGL;
class QueryGL;

class StateManagerGL final : angle::NonCopyable
{
  public:
    StateManagerGL(const FunctionsGL *functions,
                   const gl::Caps &rendererCaps,
                   const gl::Extensions &extensions,
                   const angle::FeaturesGL &features);
    ~StateManagerGL();

    void deleteProgram(GLuint program);
    void deleteVertexArray(GLuint vao);
    void deleteTexture(GLuint texture);
    void deleteSampler(GLuint sampler);
    void deleteBuffer(GLuint buffer);
    void deleteFramebuffer(GLuint fbo);
    void deleteRenderbuffer(GLuint rbo);
    void deleteTransformFeedback(GLuint transformFeedback);

    void useProgram(GLuint program);
    void forceUseProgram(GLuint program);
    void bindVertexArray(GLuint vao, GLuint elementArrayBuffer);
    void bindBuffer(gl::BufferBinding target, GLuint buffer);
    void bindBufferBase(gl::BufferBinding target, size_t index, GLuint buffer);
    void bindBufferRange(gl::BufferBinding target,
                         size_t index,
                         GLuint buffer,
                         size_t offset,
                         size_t size);
    void activeTexture(size_t unit);
    void bindTexture(gl::TextureType type, GLuint texture);
    void invalidateTexture(gl::TextureType type);
    void bindSampler(size_t unit, GLuint sampler);
    void bindImageTexture(size_t unit,
                          GLuint texture,
                          GLint level,
                          GLboolean layered,
                          GLint layer,
                          GLenum access,
                          GLenum format);
    void bindFramebuffer(GLenum type, GLuint framebuffer);
    void bindRenderbuffer(GLenum type, GLuint renderbuffer);
    void bindTransformFeedback(GLenum type, GLuint transformFeedback);
    void onTransformFeedbackStateChange();
    void beginQuery(gl::QueryType type, QueryGL *queryObject, GLuint queryId);
    void endQuery(gl::QueryType type, QueryGL *queryObject, GLuint queryId);

    void setAttributeCurrentData(size_t index, const gl::VertexAttribCurrentValueData &data);

    void setScissorTestEnabled(bool enabled);
    void setScissor(const gl::Rectangle &scissor);

    void setViewport(const gl::Rectangle &viewport);
    void setDepthRange(float near, float far);

    void setBlendEnabled(bool enabled);
    void setBlendColor(const gl::ColorF &blendColor);
    void setBlendFuncs(GLenum sourceBlendRGB,
                       GLenum destBlendRGB,
                       GLenum sourceBlendAlpha,
                       GLenum destBlendAlpha);
    void setBlendEquations(GLenum blendEquationRGB, GLenum blendEquationAlpha);
    void setColorMask(bool red, bool green, bool blue, bool alpha);
    void setSampleAlphaToCoverageEnabled(bool enabled);
    void setSampleCoverageEnabled(bool enabled);
    void setSampleCoverage(float value, bool invert);
    void setSampleMaskEnabled(bool enabled);
    void setSampleMaski(GLuint maskNumber, GLbitfield mask);

    void setDepthTestEnabled(bool enabled);
    void setDepthFunc(GLenum depthFunc);
    void setDepthMask(bool mask);
    void setStencilTestEnabled(bool enabled);
    void setStencilFrontWritemask(GLuint mask);
    void setStencilBackWritemask(GLuint mask);
    void setStencilFrontFuncs(GLenum func, GLint ref, GLuint mask);
    void setStencilBackFuncs(GLenum func, GLint ref, GLuint mask);
    void setStencilFrontOps(GLenum sfail, GLenum dpfail, GLenum dppass);
    void setStencilBackOps(GLenum sfail, GLenum dpfail, GLenum dppass);

    void setCullFaceEnabled(bool enabled);
    void setCullFace(gl::CullFaceMode cullFace);
    void setFrontFace(GLenum frontFace);
    void setPolygonOffsetFillEnabled(bool enabled);
    void setPolygonOffset(float factor, float units);
    void setRasterizerDiscardEnabled(bool enabled);
    void setLineWidth(float width);

    void setPrimitiveRestartEnabled(bool enabled);

    void setClearColor(const gl::ColorF &clearColor);
    void setClearDepth(float clearDepth);
    void setClearStencil(GLint clearStencil);

    void setPixelUnpackState(const gl::PixelUnpackState &unpack);
    void setPixelUnpackBuffer(const gl::Buffer *pixelBuffer);
    void setPixelPackState(const gl::PixelPackState &pack);
    void setPixelPackBuffer(const gl::Buffer *pixelBuffer);

    void setFramebufferSRGBEnabled(const gl::Context *context, bool enabled);
    void setFramebufferSRGBEnabledForFramebuffer(const gl::Context *context,
                                                 bool enabled,
                                                 const FramebufferGL *framebuffer);
    void setColorMaskForFramebuffer(bool red,
                                    bool green,
                                    bool blue,
                                    bool alpha,
                                    const FramebufferGL *framebuffer);

    void setDitherEnabled(bool enabled);

    void setMultisamplingStateEnabled(bool enabled);
    void setSampleAlphaToOneStateEnabled(bool enabled);

    void setCoverageModulation(GLenum components);

    void setPathRenderingModelViewMatrix(const GLfloat *m);
    void setPathRenderingProjectionMatrix(const GLfloat *m);
    void setPathRenderingStencilState(GLenum func, GLint ref, GLuint mask);

    void setProvokingVertex(GLenum mode);

    void setClipDistancesEnable(const gl::State::ClipDistanceEnableBits &enables);

    void pauseTransformFeedback();
    angle::Result pauseAllQueries(const gl::Context *context);
    angle::Result pauseQuery(const gl::Context *context, gl::QueryType type);
    angle::Result resumeAllQueries(const gl::Context *context);
    angle::Result resumeQuery(const gl::Context *context, gl::QueryType type);
    angle::Result onMakeCurrent(const gl::Context *context);

    void syncState(const gl::Context *context,
                   const gl::State::DirtyBits &glDirtyBits,
                   const gl::State::DirtyBits &bitMask);

    ANGLE_INLINE void updateMultiviewBaseViewLayerIndexUniform(
        const gl::Program *program,
        const gl::FramebufferState &drawFramebufferState) const
    {
        if (mIsMultiviewEnabled && program && program->usesMultiview())
        {
            updateMultiviewBaseViewLayerIndexUniformImpl(program, drawFramebufferState);
        }
    }

    GLuint getProgramID() const { return mProgram; }
    GLuint getVertexArrayID() const { return mVAO; }
    GLuint getFramebufferID(angle::FramebufferBinding binding) const
    {
        return mFramebuffers[binding];
    }
    GLuint getBufferID(gl::BufferBinding binding) const { return mBuffers[binding]; }

    void validateState() const;

  private:
    void setTextureCubemapSeamlessEnabled(bool enabled);

    void propagateProgramToVAO(const gl::Program *program, VertexArrayGL *vao);

    void updateProgramTextureBindings(const gl::Context *context);
    void updateProgramStorageBufferBindings(const gl::Context *context);
    void updateProgramUniformBufferBindings(const gl::Context *context);
    void updateProgramAtomicCounterBufferBindings(const gl::Context *context);
    void updateProgramImageBindings(const gl::Context *context);

    void updateDispatchIndirectBufferBinding(const gl::Context *context);
    void updateDrawIndirectBufferBinding(const gl::Context *context);

    void syncSamplersState(const gl::Context *context);
    void syncTransformFeedbackState(const gl::Context *context);

    void updateMultiviewBaseViewLayerIndexUniformImpl(
        const gl::Program *program,
        const gl::FramebufferState &drawFramebufferState) const;

    const FunctionsGL *mFunctions;
    const angle::FeaturesGL &mFeatures;

    GLuint mProgram;

    GLuint mVAO;
    std::vector<gl::VertexAttribCurrentValueData> mVertexAttribCurrentValues;

    angle::PackedEnumMap<gl::BufferBinding, GLuint> mBuffers;

    struct IndexedBufferBinding
    {
        IndexedBufferBinding();

        size_t offset;
        size_t size;
        GLuint buffer;
    };
    angle::PackedEnumMap<gl::BufferBinding, std::vector<IndexedBufferBinding>> mIndexedBuffers;

    size_t mTextureUnitIndex;
    angle::PackedEnumMap<gl::TextureType, gl::ActiveTextureArray<GLuint>> mTextures;
    gl::ActiveTextureArray<GLuint> mSamplers;

    struct ImageUnitBinding
    {
        ImageUnitBinding()
            : texture(0), level(0), layered(false), layer(0), access(GL_READ_ONLY), format(GL_R32UI)
        {}

        GLuint texture;
        GLint level;
        GLboolean layered;
        GLint layer;
        GLenum access;
        GLenum format;
    };
    std::vector<ImageUnitBinding> mImages;

    GLuint mTransformFeedback;
    TransformFeedbackGL *mCurrentTransformFeedback;

    // Queries that are currently running on the driver
    angle::PackedEnumMap<gl::QueryType, QueryGL *> mQueries;

    // Queries that are temporarily in the paused state so that their results will not be affected
    // by other operations
    angle::PackedEnumMap<gl::QueryType, QueryGL *> mTemporaryPausedQueries;

    gl::ContextID mPrevDrawContext;

    GLint mUnpackAlignment;
    GLint mUnpackRowLength;
    GLint mUnpackSkipRows;
    GLint mUnpackSkipPixels;
    GLint mUnpackImageHeight;
    GLint mUnpackSkipImages;

    GLint mPackAlignment;
    GLint mPackRowLength;
    GLint mPackSkipRows;
    GLint mPackSkipPixels;

    // TODO(jmadill): Convert to std::array when available
    std::vector<GLenum> mFramebuffers;
    GLuint mRenderbuffer;

    bool mScissorTestEnabled;
    gl::Rectangle mScissor;
    gl::Rectangle mViewport;
    float mNear;
    float mFar;

    bool mBlendEnabled;
    gl::ColorF mBlendColor;
    GLenum mSourceBlendRGB;
    GLenum mDestBlendRGB;
    GLenum mSourceBlendAlpha;
    GLenum mDestBlendAlpha;
    GLenum mBlendEquationRGB;
    GLenum mBlendEquationAlpha;
    bool mColorMaskRed;
    bool mColorMaskGreen;
    bool mColorMaskBlue;
    bool mColorMaskAlpha;
    bool mSampleAlphaToCoverageEnabled;
    bool mSampleCoverageEnabled;
    float mSampleCoverageValue;
    bool mSampleCoverageInvert;
    bool mSampleMaskEnabled;
    std::array<GLbitfield, gl::MAX_SAMPLE_MASK_WORDS> mSampleMaskValues;

    bool mDepthTestEnabled;
    GLenum mDepthFunc;
    bool mDepthMask;
    bool mStencilTestEnabled;
    GLenum mStencilFrontFunc;
    GLint mStencilFrontRef;
    GLuint mStencilFrontValueMask;
    GLenum mStencilFrontStencilFailOp;
    GLenum mStencilFrontStencilPassDepthFailOp;
    GLenum mStencilFrontStencilPassDepthPassOp;
    GLuint mStencilFrontWritemask;
    GLenum mStencilBackFunc;
    GLint mStencilBackRef;
    GLuint mStencilBackValueMask;
    GLenum mStencilBackStencilFailOp;
    GLenum mStencilBackStencilPassDepthFailOp;
    GLenum mStencilBackStencilPassDepthPassOp;
    GLuint mStencilBackWritemask;

    bool mCullFaceEnabled;
    gl::CullFaceMode mCullFace;
    GLenum mFrontFace;
    bool mPolygonOffsetFillEnabled;
    GLfloat mPolygonOffsetFactor;
    GLfloat mPolygonOffsetUnits;
    bool mRasterizerDiscardEnabled;
    float mLineWidth;

    bool mPrimitiveRestartEnabled;

    gl::ColorF mClearColor;
    float mClearDepth;
    GLint mClearStencil;

    bool mFramebufferSRGBEnabled;
    bool mDitherEnabled;
    bool mTextureCubemapSeamlessEnabled;

    bool mMultisamplingEnabled;
    bool mSampleAlphaToOneEnabled;

    GLenum mCoverageModulation;

    GLfloat mPathMatrixMV[16];
    GLfloat mPathMatrixProj[16];
    GLenum mPathStencilFunc;
    GLint mPathStencilRef;
    GLuint mPathStencilMask;

    const bool mIsMultiviewEnabled;

    GLenum mProvokingVertex;

    gl::State::ClipDistanceEnableBits mEnabledClipDistances;
    const size_t mMaxClipDistances;

    gl::State::DirtyBits mLocalDirtyBits;
    gl::AttributesMask mLocalDirtyCurrentValues;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_STATEMANAGERGL_H_
