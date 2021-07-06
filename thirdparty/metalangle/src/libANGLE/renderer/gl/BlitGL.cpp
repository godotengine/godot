//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// BlitGL.cpp: Implements the BlitGL class, a helper for blitting textures

#include "libANGLE/renderer/gl/BlitGL.h"

#include "common/FixedVector.h"
#include "common/utilities.h"
#include "common/vector_utils.h"
#include "image_util/copyimage.h"
#include "libANGLE/Context.h"
#include "libANGLE/Framebuffer.h"
#include "libANGLE/formatutils.h"
#include "libANGLE/renderer/Format.h"
#include "libANGLE/renderer/gl/ContextGL.h"
#include "libANGLE/renderer/gl/FramebufferGL.h"
#include "libANGLE/renderer/gl/FunctionsGL.h"
#include "libANGLE/renderer/gl/RenderbufferGL.h"
#include "libANGLE/renderer/gl/StateManagerGL.h"
#include "libANGLE/renderer/gl/TextureGL.h"
#include "libANGLE/renderer/gl/formatutilsgl.h"
#include "libANGLE/renderer/gl/renderergl_utils.h"
#include "libANGLE/renderer/renderer_utils.h"
#include "platform/FeaturesGL.h"

using angle::Vector2;

namespace rx
{

namespace
{

angle::Result CheckCompileStatus(const gl::Context *context,
                                 const rx::FunctionsGL *functions,
                                 GLuint shader)
{
    GLint compileStatus = GL_FALSE;
    ANGLE_GL_TRY(context, functions->getShaderiv(shader, GL_COMPILE_STATUS, &compileStatus));

    ASSERT(compileStatus == GL_TRUE);
    ANGLE_CHECK(GetImplAs<ContextGL>(context), compileStatus == GL_TRUE,
                "Failed to compile internal blit shader.", GL_OUT_OF_MEMORY);

    return angle::Result::Continue;
}

angle::Result CheckLinkStatus(const gl::Context *context,
                              const rx::FunctionsGL *functions,
                              GLuint program)
{
    GLint linkStatus = GL_FALSE;
    ANGLE_GL_TRY(context, functions->getProgramiv(program, GL_LINK_STATUS, &linkStatus));
    ASSERT(linkStatus == GL_TRUE);
    ANGLE_CHECK(GetImplAs<ContextGL>(context), linkStatus == GL_TRUE,
                "Failed to link internal blit program.", GL_OUT_OF_MEMORY);

    return angle::Result::Continue;
}

class ScopedGLState : angle::NonCopyable
{
  public:
    enum
    {
        KEEP_SCISSOR = 1,
    };

    ScopedGLState() {}

    ~ScopedGLState() { ASSERT(mExited); }

    angle::Result enter(const gl::Context *context, gl::Rectangle viewport, int keepState = 0)
    {
        ContextGL *contextGL         = GetImplAs<ContextGL>(context);
        StateManagerGL *stateManager = contextGL->getStateManager();

        if (!(keepState & KEEP_SCISSOR))
        {
            stateManager->setScissorTestEnabled(false);
        }
        stateManager->setViewport(viewport);
        stateManager->setDepthRange(0.0f, 1.0f);
        stateManager->setBlendEnabled(false);
        stateManager->setColorMask(true, true, true, true);
        stateManager->setSampleAlphaToCoverageEnabled(false);
        stateManager->setSampleCoverageEnabled(false);
        stateManager->setDepthTestEnabled(false);
        stateManager->setStencilTestEnabled(false);
        stateManager->setCullFaceEnabled(false);
        stateManager->setPolygonOffsetFillEnabled(false);
        stateManager->setRasterizerDiscardEnabled(false);

        stateManager->pauseTransformFeedback();
        return stateManager->pauseAllQueries(context);
    }

    angle::Result exit(const gl::Context *context)
    {
        mExited = true;

        ContextGL *contextGL         = GetImplAs<ContextGL>(context);
        StateManagerGL *stateManager = contextGL->getStateManager();

        // XFB resuming will be done automatically
        return stateManager->resumeAllQueries(context);
    }

    void willUseTextureUnit(const gl::Context *context, int unit)
    {
        ContextGL *contextGL = GetImplAs<ContextGL>(context);

        if (contextGL->getFunctions()->bindSampler)
        {
            contextGL->getStateManager()->bindSampler(unit, 0);
        }
    }

  private:
    bool mExited = false;
};

angle::Result SetClearState(StateManagerGL *stateManager,
                            bool colorClear,
                            bool depthClear,
                            bool stencilClear,
                            GLbitfield *outClearMask)
{
    *outClearMask = 0;
    if (colorClear)
    {
        stateManager->setClearColor(gl::ColorF(0.0f, 0.0f, 0.0f, 0.0f));
        stateManager->setColorMask(true, true, true, true);
        *outClearMask |= GL_COLOR_BUFFER_BIT;
    }
    if (depthClear)
    {
        stateManager->setDepthMask(true);
        stateManager->setClearDepth(1.0f);
        *outClearMask |= GL_DEPTH_BUFFER_BIT;
    }
    if (stencilClear)
    {
        stateManager->setClearStencil(0);
        *outClearMask |= GL_STENCIL_BUFFER_BIT;
    }

    stateManager->setScissorTestEnabled(false);

    return angle::Result::Continue;
}

using ClearBindTargetVector = angle::FixedVector<GLenum, 3>;

angle::Result PrepareForClear(StateManagerGL *stateManager,
                              GLenum sizedInternalFormat,
                              ClearBindTargetVector *outBindtargets,
                              ClearBindTargetVector *outUnbindTargets,
                              GLbitfield *outClearMask)
{
    const gl::InternalFormat &internalFormatInfo =
        gl::GetSizedInternalFormatInfo(sizedInternalFormat);
    bool bindDepth   = internalFormatInfo.depthBits > 0;
    bool bindStencil = internalFormatInfo.stencilBits > 0;
    bool bindColor   = !bindDepth && !bindStencil;

    outBindtargets->clear();
    if (bindColor)
    {
        outBindtargets->push_back(GL_COLOR_ATTACHMENT0);
    }
    else
    {
        outUnbindTargets->push_back(GL_COLOR_ATTACHMENT0);
    }
    if (bindDepth)
    {
        outBindtargets->push_back(GL_DEPTH_ATTACHMENT);
    }
    else
    {
        outUnbindTargets->push_back(GL_DEPTH_ATTACHMENT);
    }
    if (bindStencil)
    {
        outBindtargets->push_back(GL_STENCIL_ATTACHMENT);
    }
    else
    {
        outUnbindTargets->push_back(GL_STENCIL_ATTACHMENT);
    }

    ANGLE_TRY(SetClearState(stateManager, bindColor, bindDepth, bindStencil, outClearMask));

    return angle::Result::Continue;
}

angle::Result UnbindAttachments(const gl::Context *context,
                                const FunctionsGL *functions,
                                GLenum framebufferTarget,
                                const ClearBindTargetVector &bindTargets)
{
    for (GLenum bindTarget : bindTargets)
    {
        ANGLE_GL_TRY(context, functions->framebufferRenderbuffer(framebufferTarget, bindTarget,
                                                                 GL_RENDERBUFFER, 0));
    }
    return angle::Result::Continue;
}

}  // anonymous namespace

BlitGL::BlitGL(const FunctionsGL *functions,
               const angle::FeaturesGL &features,
               StateManagerGL *stateManager)
    : mFunctions(functions),
      mFeatures(features),
      mStateManager(stateManager),
      mScratchFBO(0),
      mVAO(0),
      mVertexBuffer(0)
{
    for (size_t i = 0; i < ArraySize(mScratchTextures); i++)
    {
        mScratchTextures[i] = 0;
    }

    ASSERT(mFunctions);
    ASSERT(mStateManager);
}

BlitGL::~BlitGL()
{
    for (const auto &blitProgram : mBlitPrograms)
    {
        mStateManager->deleteProgram(blitProgram.second.program);
    }
    mBlitPrograms.clear();

    for (size_t i = 0; i < ArraySize(mScratchTextures); i++)
    {
        if (mScratchTextures[i] != 0)
        {
            mStateManager->deleteTexture(mScratchTextures[i]);
            mScratchTextures[i] = 0;
        }
    }

    if (mScratchFBO != 0)
    {
        mStateManager->deleteFramebuffer(mScratchFBO);
        mScratchFBO = 0;
    }

    if (mVAO != 0)
    {
        mStateManager->deleteVertexArray(mVAO);
        mVAO = 0;
    }
}

angle::Result BlitGL::copyImageToLUMAWorkaroundTexture(const gl::Context *context,
                                                       GLuint texture,
                                                       gl::TextureType textureType,
                                                       gl::TextureTarget target,
                                                       GLenum lumaFormat,
                                                       size_t level,
                                                       const gl::Rectangle &sourceArea,
                                                       GLenum internalFormat,
                                                       gl::Framebuffer *source)
{
    mStateManager->bindTexture(textureType, texture);

    // Allocate the texture memory
    GLenum format = gl::GetUnsizedFormat(internalFormat);

    GLenum readType = GL_NONE;
    ANGLE_TRY(source->getImplementationColorReadType(context, &readType));

    gl::PixelUnpackState unpack;
    mStateManager->setPixelUnpackState(unpack);
    mStateManager->setPixelUnpackBuffer(
        context->getState().getTargetBuffer(gl::BufferBinding::PixelUnpack));
    ANGLE_GL_TRY_ALWAYS_CHECK(
        context,
        mFunctions->texImage2D(ToGLenum(target), static_cast<GLint>(level), internalFormat,
                               sourceArea.width, sourceArea.height, 0, format, readType, nullptr));

    return copySubImageToLUMAWorkaroundTexture(context, texture, textureType, target, lumaFormat,
                                               level, gl::Offset(0, 0, 0), sourceArea, source);
}

angle::Result BlitGL::copySubImageToLUMAWorkaroundTexture(const gl::Context *context,
                                                          GLuint texture,
                                                          gl::TextureType textureType,
                                                          gl::TextureTarget target,
                                                          GLenum lumaFormat,
                                                          size_t level,
                                                          const gl::Offset &destOffset,
                                                          const gl::Rectangle &sourceArea,
                                                          gl::Framebuffer *source)
{
    ANGLE_TRY(initializeResources(context));

    BlitProgram *blitProgram = nullptr;
    ANGLE_TRY(getBlitProgram(context, gl::TextureType::_2D, GL_FLOAT, GL_FLOAT, &blitProgram));

    // Blit the framebuffer to the first scratch texture
    const FramebufferGL *sourceFramebufferGL = GetImplAs<FramebufferGL>(source);
    mStateManager->bindFramebuffer(GL_FRAMEBUFFER, sourceFramebufferGL->getFramebufferID());

    GLenum readFormat = GL_NONE;
    ANGLE_TRY(source->getImplementationColorReadFormat(context, &readFormat));

    GLenum readType = GL_NONE;
    ANGLE_TRY(source->getImplementationColorReadType(context, &readType));

    nativegl::CopyTexImageImageFormat copyTexImageFormat =
        nativegl::GetCopyTexImageImageFormat(mFunctions, mFeatures, readFormat, readType);

    mStateManager->bindTexture(gl::TextureType::_2D, mScratchTextures[0]);
    ANGLE_GL_TRY_ALWAYS_CHECK(
        context, mFunctions->copyTexImage2D(GL_TEXTURE_2D, 0, copyTexImageFormat.internalFormat,
                                            sourceArea.x, sourceArea.y, sourceArea.width,
                                            sourceArea.height, 0));

    // Set the swizzle of the scratch texture so that the channels sample into the correct emulated
    // LUMA channels.
    GLint swizzle[4] = {
        (lumaFormat == GL_ALPHA) ? GL_ALPHA : GL_RED,
        (lumaFormat == GL_LUMINANCE_ALPHA) ? GL_ALPHA : GL_ZERO,
        GL_ZERO,
        GL_ZERO,
    };
    ANGLE_GL_TRY(context,
                 mFunctions->texParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzle));

    // Make a temporary framebuffer using the second scratch texture to render the swizzled result
    // to.
    mStateManager->bindTexture(gl::TextureType::_2D, mScratchTextures[1]);
    ANGLE_GL_TRY_ALWAYS_CHECK(
        context, mFunctions->texImage2D(GL_TEXTURE_2D, 0, copyTexImageFormat.internalFormat,
                                        sourceArea.width, sourceArea.height, 0,
                                        gl::GetUnsizedFormat(copyTexImageFormat.internalFormat),
                                        readType, nullptr));

    mStateManager->bindFramebuffer(GL_FRAMEBUFFER, mScratchFBO);
    ANGLE_GL_TRY(context, mFunctions->framebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                                           GL_TEXTURE_2D, mScratchTextures[1], 0));

    // Render to the destination texture, sampling from the scratch texture
    ScopedGLState scopedState;
    ANGLE_TRY(scopedState.enter(context, gl::Rectangle(0, 0, sourceArea.width, sourceArea.height)));
    scopedState.willUseTextureUnit(context, 0);

    ANGLE_TRY(setScratchTextureParameter(context, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    ANGLE_TRY(setScratchTextureParameter(context, GL_TEXTURE_MAG_FILTER, GL_NEAREST));

    mStateManager->activeTexture(0);
    mStateManager->bindTexture(gl::TextureType::_2D, mScratchTextures[0]);

    mStateManager->useProgram(blitProgram->program);
    ANGLE_GL_TRY(context, mFunctions->uniform1i(blitProgram->sourceTextureLocation, 0));
    ANGLE_GL_TRY(context, mFunctions->uniform2f(blitProgram->scaleLocation, 1.0, 1.0));
    ANGLE_GL_TRY(context, mFunctions->uniform2f(blitProgram->offsetLocation, 0.0, 0.0));
    ANGLE_GL_TRY(context, mFunctions->uniform1i(blitProgram->multiplyAlphaLocation, 0));
    ANGLE_GL_TRY(context, mFunctions->uniform1i(blitProgram->unMultiplyAlphaLocation, 0));

    mStateManager->bindVertexArray(mVAO, 0);
    ANGLE_GL_TRY(context, mFunctions->drawArrays(GL_TRIANGLES, 0, 3));

    // Copy the swizzled texture to the destination texture
    mStateManager->bindTexture(textureType, texture);

    if (nativegl::UseTexImage3D(textureType))
    {
        ANGLE_GL_TRY(context,
                     mFunctions->copyTexSubImage3D(ToGLenum(target), static_cast<GLint>(level),
                                                   destOffset.x, destOffset.y, destOffset.z, 0, 0,
                                                   sourceArea.width, sourceArea.height));
    }
    else
    {
        ASSERT(nativegl::UseTexImage2D(textureType));
        ANGLE_GL_TRY(context, mFunctions->copyTexSubImage2D(
                                  ToGLenum(target), static_cast<GLint>(level), destOffset.x,
                                  destOffset.y, 0, 0, sourceArea.width, sourceArea.height));
    }

    // Finally orphan the scratch textures so they can be GCed by the driver.
    ANGLE_TRY(orphanScratchTextures(context));

    ANGLE_TRY(scopedState.exit(context));
    return angle::Result::Continue;
}

angle::Result BlitGL::blitColorBufferWithShader(const gl::Context *context,
                                                const gl::Framebuffer *source,
                                                const gl::Framebuffer *dest,
                                                const gl::Rectangle &sourceAreaIn,
                                                const gl::Rectangle &destAreaIn,
                                                GLenum filter)
{
    ANGLE_TRY(initializeResources(context));

    BlitProgram *blitProgram = nullptr;
    ANGLE_TRY(getBlitProgram(context, gl::TextureType::_2D, GL_FLOAT, GL_FLOAT, &blitProgram));

    // We'll keep things simple by removing reversed coordinates from the rectangles. In the end
    // we'll apply the reversal to the source texture coordinates if needed. The destination
    // rectangle will be set to the gl viewport, which can't be reversed.
    bool reverseX            = sourceAreaIn.isReversedX() != destAreaIn.isReversedX();
    bool reverseY            = sourceAreaIn.isReversedY() != destAreaIn.isReversedY();
    gl::Rectangle sourceArea = sourceAreaIn.removeReversal();
    gl::Rectangle destArea   = destAreaIn.removeReversal();

    const gl::FramebufferAttachment *readAttachment = source->getReadColorAttachment();
    ASSERT(readAttachment->getSamples() <= 1);

    // Compute the part of the source that will be sampled.
    gl::Rectangle inBoundsSource;
    {
        gl::Extents sourceSize = readAttachment->getSize();
        gl::Rectangle sourceBounds(0, 0, sourceSize.width, sourceSize.height);
        if (!gl::ClipRectangle(sourceArea, sourceBounds, &inBoundsSource))
        {
            // Early out when the sampled part is empty as the blit will be a noop,
            // and it prevents a division by zero in later computations.
            return angle::Result::Continue;
        }
    }

    // The blit will be emulated by getting the source of the blit in a texture and sampling it
    // with CLAMP_TO_EDGE.

    GLuint textureId;

    // TODO(cwallez) once texture dirty bits are landed, reuse attached texture instead of using
    // CopyTexImage2D
    {
        textureId = mScratchTextures[0];

        GLenum format                 = readAttachment->getFormat().info->internalFormat;
        const FramebufferGL *sourceGL = GetImplAs<FramebufferGL>(source);
        mStateManager->bindFramebuffer(GL_READ_FRAMEBUFFER, sourceGL->getFramebufferID());
        mStateManager->bindTexture(gl::TextureType::_2D, textureId);

        ANGLE_GL_TRY_ALWAYS_CHECK(
            context,
            mFunctions->copyTexImage2D(GL_TEXTURE_2D, 0, format, inBoundsSource.x, inBoundsSource.y,
                                       inBoundsSource.width, inBoundsSource.height, 0));

        // Translate sourceArea to be relative to the copied image.
        sourceArea.x -= inBoundsSource.x;
        sourceArea.y -= inBoundsSource.y;

        ANGLE_TRY(setScratchTextureParameter(context, GL_TEXTURE_MIN_FILTER, filter));
        ANGLE_TRY(setScratchTextureParameter(context, GL_TEXTURE_MAG_FILTER, filter));
        ANGLE_TRY(setScratchTextureParameter(context, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
        ANGLE_TRY(setScratchTextureParameter(context, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    }

    // Transform the source area to the texture coordinate space (where 0.0 and 1.0 correspond to
    // the edges of the texture).
    Vector2 texCoordOffset(
        static_cast<float>(sourceArea.x) / static_cast<float>(inBoundsSource.width),
        static_cast<float>(sourceArea.y) / static_cast<float>(inBoundsSource.height));
    // texCoordScale is equal to the size of the source area in texture coordinates.
    Vector2 texCoordScale(
        static_cast<float>(sourceArea.width) / static_cast<float>(inBoundsSource.width),
        static_cast<float>(sourceArea.height) / static_cast<float>(inBoundsSource.height));

    if (reverseX)
    {
        texCoordOffset.x() = texCoordOffset.x() + texCoordScale.x();
        texCoordScale.x()  = -texCoordScale.x();
    }
    if (reverseY)
    {
        texCoordOffset.y() = texCoordOffset.y() + texCoordScale.y();
        texCoordScale.y()  = -texCoordScale.y();
    }

    // Reset all the state except scissor and use the viewport to draw exactly to the destination
    // rectangle
    ScopedGLState scopedState;
    ANGLE_TRY(scopedState.enter(context, destArea, ScopedGLState::KEEP_SCISSOR));
    scopedState.willUseTextureUnit(context, 0);

    // Set uniforms
    mStateManager->activeTexture(0);
    mStateManager->bindTexture(gl::TextureType::_2D, textureId);

    mStateManager->useProgram(blitProgram->program);
    ANGLE_GL_TRY(context, mFunctions->uniform1i(blitProgram->sourceTextureLocation, 0));
    ANGLE_GL_TRY(context, mFunctions->uniform2f(blitProgram->scaleLocation, texCoordScale.x(),
                                                texCoordScale.y()));
    ANGLE_GL_TRY(context, mFunctions->uniform2f(blitProgram->offsetLocation, texCoordOffset.x(),
                                                texCoordOffset.y()));
    ANGLE_GL_TRY(context, mFunctions->uniform1i(blitProgram->multiplyAlphaLocation, 0));
    ANGLE_GL_TRY(context, mFunctions->uniform1i(blitProgram->unMultiplyAlphaLocation, 0));

    const FramebufferGL *destGL = GetImplAs<FramebufferGL>(dest);
    mStateManager->bindFramebuffer(GL_DRAW_FRAMEBUFFER, destGL->getFramebufferID());

    mStateManager->bindVertexArray(mVAO, 0);
    ANGLE_GL_TRY(context, mFunctions->drawArrays(GL_TRIANGLES, 0, 3));

    ANGLE_TRY(scopedState.exit(context));
    return angle::Result::Continue;
}

angle::Result BlitGL::copySubTexture(const gl::Context *context,
                                     TextureGL *source,
                                     size_t sourceLevel,
                                     GLenum sourceComponentType,
                                     GLuint destID,
                                     gl::TextureTarget destTarget,
                                     size_t destLevel,
                                     GLenum destComponentType,
                                     const gl::Extents &sourceSize,
                                     const gl::Rectangle &sourceArea,
                                     const gl::Offset &destOffset,
                                     bool needsLumaWorkaround,
                                     GLenum lumaFormat,
                                     bool unpackFlipY,
                                     bool unpackPremultiplyAlpha,
                                     bool unpackUnmultiplyAlpha,
                                     bool *copySucceededOut)
{
    ASSERT(source->getType() == gl::TextureType::_2D ||
           source->getType() == gl::TextureType::External ||
           source->getType() == gl::TextureType::Rectangle);
    ANGLE_TRY(initializeResources(context));

    // Make sure the destination texture can be rendered to before setting anything else up.  Some
    // cube maps may not be renderable until all faces have been filled.
    mStateManager->bindFramebuffer(GL_FRAMEBUFFER, mScratchFBO);
    ANGLE_GL_TRY(context, mFunctions->framebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                                           ToGLenum(destTarget), destID,
                                                           static_cast<GLint>(destLevel)));
    GLenum status = ANGLE_GL_TRY(context, mFunctions->checkFramebufferStatus(GL_FRAMEBUFFER));
    if (status != GL_FRAMEBUFFER_COMPLETE)
    {
        *copySucceededOut = false;
        return angle::Result::Continue;
    }

    BlitProgram *blitProgram = nullptr;
    ANGLE_TRY(getBlitProgram(context, source->getType(), sourceComponentType, destComponentType,
                             &blitProgram));

    // Setup the source texture
    if (needsLumaWorkaround)
    {
        GLint luminance = (lumaFormat == GL_ALPHA) ? GL_ZERO : GL_RED;

        GLint alpha = GL_RED;
        if (lumaFormat == GL_LUMINANCE)
        {
            alpha = GL_ONE;
        }
        else if (lumaFormat == GL_LUMINANCE_ALPHA)
        {
            alpha = GL_GREEN;
        }
        else
        {
            ASSERT(lumaFormat == GL_ALPHA);
        }

        GLint swizzle[4] = {luminance, luminance, luminance, alpha};
        ANGLE_TRY(source->setSwizzle(context, swizzle));
    }
    ANGLE_TRY(source->setMinFilter(context, GL_NEAREST));
    ANGLE_TRY(source->setMagFilter(context, GL_NEAREST));
    ANGLE_TRY(source->setBaseLevel(context, static_cast<GLuint>(sourceLevel)));

    // Render to the destination texture, sampling from the source texture
    ScopedGLState scopedState;
    ANGLE_TRY(scopedState.enter(
        context, gl::Rectangle(destOffset.x, destOffset.y, sourceArea.width, sourceArea.height)));
    scopedState.willUseTextureUnit(context, 0);

    mStateManager->activeTexture(0);
    mStateManager->bindTexture(source->getType(), source->getTextureID());

    Vector2 scale(sourceArea.width, sourceArea.height);
    Vector2 offset(sourceArea.x, sourceArea.y);
    if (source->getType() != gl::TextureType::Rectangle)
    {
        scale.x() /= static_cast<float>(sourceSize.width);
        scale.y() /= static_cast<float>(sourceSize.height);
        offset.x() /= static_cast<float>(sourceSize.width);
        offset.y() /= static_cast<float>(sourceSize.height);
    }
    if (unpackFlipY)
    {
        offset.y() += scale.y();
        scale.y() = -scale.y();
    }

    mStateManager->useProgram(blitProgram->program);
    ANGLE_GL_TRY(context, mFunctions->uniform1i(blitProgram->sourceTextureLocation, 0));
    ANGLE_GL_TRY(context, mFunctions->uniform2f(blitProgram->scaleLocation, scale.x(), scale.y()));
    ANGLE_GL_TRY(context,
                 mFunctions->uniform2f(blitProgram->offsetLocation, offset.x(), offset.y()));
    if (unpackPremultiplyAlpha == unpackUnmultiplyAlpha)
    {
        ANGLE_GL_TRY(context, mFunctions->uniform1i(blitProgram->multiplyAlphaLocation, 0));
        ANGLE_GL_TRY(context, mFunctions->uniform1i(blitProgram->unMultiplyAlphaLocation, 0));
    }
    else
    {
        ANGLE_GL_TRY(context, mFunctions->uniform1i(blitProgram->multiplyAlphaLocation,
                                                    unpackPremultiplyAlpha));
        ANGLE_GL_TRY(context, mFunctions->uniform1i(blitProgram->unMultiplyAlphaLocation,
                                                    unpackUnmultiplyAlpha));
    }

    mStateManager->bindVertexArray(mVAO, 0);
    ANGLE_GL_TRY(context, mFunctions->drawArrays(GL_TRIANGLES, 0, 3));

    *copySucceededOut = true;
    ANGLE_TRY(scopedState.exit(context));
    return angle::Result::Continue;
}

angle::Result BlitGL::copySubTextureCPUReadback(const gl::Context *context,
                                                TextureGL *source,
                                                size_t sourceLevel,
                                                GLenum sourceSizedInternalFormat,
                                                TextureGL *dest,
                                                gl::TextureTarget destTarget,
                                                size_t destLevel,
                                                GLenum destFormat,
                                                GLenum destType,
                                                const gl::Extents &sourceSize,
                                                const gl::Rectangle &sourceArea,
                                                const gl::Offset &destOffset,
                                                bool needsLumaWorkaround,
                                                GLenum lumaFormat,
                                                bool unpackFlipY,
                                                bool unpackPremultiplyAlpha,
                                                bool unpackUnmultiplyAlpha)
{
    ANGLE_TRY(initializeResources(context));

    ContextGL *contextGL = GetImplAs<ContextGL>(context);

    ASSERT(source->getType() == gl::TextureType::_2D ||
           source->getType() == gl::TextureType::External ||
           source->getType() == gl::TextureType::Rectangle);
    const auto &destInternalFormatInfo = gl::GetInternalFormatInfo(destFormat, destType);
    const gl::InternalFormat &sourceInternalFormatInfo =
        gl::GetSizedInternalFormatInfo(sourceSizedInternalFormat);

    gl::Rectangle readPixelsArea = sourceArea;

    mStateManager->bindFramebuffer(GL_FRAMEBUFFER, mScratchFBO);
    ANGLE_GL_TRY(context, mFunctions->framebufferTexture2D(
                              GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, ToGLenum(source->getType()),
                              source->getTextureID(), static_cast<GLint>(sourceLevel)));
    GLenum status = ANGLE_GL_TRY(context, mFunctions->checkFramebufferStatus(GL_FRAMEBUFFER));
    if (status != GL_FRAMEBUFFER_COMPLETE)
    {
        // The source texture cannot be read with glReadPixels. Copy it into another RGBA texture
        // and read that back instead.
        nativegl::TexImageFormat texImageFormat = nativegl::GetTexImageFormat(
            mFunctions, mFeatures, sourceInternalFormatInfo.internalFormat,
            sourceInternalFormatInfo.format, sourceInternalFormatInfo.type);

        gl::TextureType scratchTextureType = gl::TextureType::_2D;
        mStateManager->bindTexture(scratchTextureType, mScratchTextures[0]);
        ANGLE_GL_TRY_ALWAYS_CHECK(
            context,
            mFunctions->texImage2D(ToGLenum(scratchTextureType), 0, texImageFormat.internalFormat,
                                   sourceArea.width, sourceArea.height, 0, texImageFormat.format,
                                   texImageFormat.type, nullptr));

        bool copySucceeded = false;
        ANGLE_TRY(copySubTexture(
            context, source, sourceLevel, sourceInternalFormatInfo.componentType,
            mScratchTextures[0], NonCubeTextureTypeToTarget(scratchTextureType), 0,
            sourceInternalFormatInfo.componentType, sourceSize, sourceArea, gl::Offset(0, 0, 0),
            needsLumaWorkaround, lumaFormat, false, false, false, &copySucceeded));
        if (!copySucceeded)
        {
            // No fallback options if we can't render to the scratch texture.
            return angle::Result::Stop;
        }

        // Bind the scratch texture as the readback texture
        mStateManager->bindFramebuffer(GL_FRAMEBUFFER, mScratchFBO);
        ANGLE_GL_TRY(context, mFunctions->framebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                                               ToGLenum(scratchTextureType),
                                                               mScratchTextures[0], 0));

        // The scratch texture sized to sourceArea so adjust the readpixels area
        readPixelsArea.x = 0;
        readPixelsArea.y = 0;

        // Recheck the status
        status = ANGLE_GL_TRY(context, mFunctions->checkFramebufferStatus(GL_FRAMEBUFFER));
    }

    ASSERT(status == GL_FRAMEBUFFER_COMPLETE);

    // Create a buffer for holding the source and destination memory
    const size_t sourcePixelSize = 4;
    size_t sourceBufferSize      = readPixelsArea.width * readPixelsArea.height * sourcePixelSize;
    size_t destBufferSize =
        readPixelsArea.width * readPixelsArea.height * destInternalFormatInfo.pixelBytes;
    angle::MemoryBuffer *buffer = nullptr;
    ANGLE_CHECK_GL_ALLOC(contextGL,
                         context->getScratchBuffer(sourceBufferSize + destBufferSize, &buffer));

    uint8_t *sourceMemory = buffer->data();
    uint8_t *destMemory   = buffer->data() + sourceBufferSize;

    GLenum readPixelsFormat        = GL_NONE;
    PixelReadFunction readFunction = nullptr;
    if (sourceInternalFormatInfo.componentType == GL_UNSIGNED_INT)
    {
        readPixelsFormat = GL_RGBA_INTEGER;
        readFunction     = angle::ReadColor<angle::R8G8B8A8, GLuint>;
    }
    else
    {
        ASSERT(sourceInternalFormatInfo.componentType != GL_INT);
        readPixelsFormat = GL_RGBA;
        readFunction     = angle::ReadColor<angle::R8G8B8A8, GLfloat>;
    }

    gl::PixelUnpackState unpack;
    unpack.alignment = 1;
    mStateManager->setPixelUnpackState(unpack);
    mStateManager->setPixelUnpackBuffer(nullptr);
    ANGLE_GL_TRY(context, mFunctions->readPixels(readPixelsArea.x, readPixelsArea.y,
                                                 readPixelsArea.width, readPixelsArea.height,
                                                 readPixelsFormat, GL_UNSIGNED_BYTE, sourceMemory));

    angle::FormatID destFormatID =
        angle::Format::InternalFormatToID(destInternalFormatInfo.sizedInternalFormat);
    const auto &destFormatInfo = angle::Format::Get(destFormatID);
    CopyImageCHROMIUM(
        sourceMemory, readPixelsArea.width * sourcePixelSize, sourcePixelSize, 0, readFunction,
        destMemory, readPixelsArea.width * destInternalFormatInfo.pixelBytes,
        destInternalFormatInfo.pixelBytes, 0, destFormatInfo.pixelWriteFunction,
        destInternalFormatInfo.format, destInternalFormatInfo.componentType, readPixelsArea.width,
        readPixelsArea.height, 1, unpackFlipY, unpackPremultiplyAlpha, unpackUnmultiplyAlpha);

    gl::PixelPackState pack;
    pack.alignment = 1;
    mStateManager->setPixelPackState(pack);
    mStateManager->setPixelPackBuffer(nullptr);

    nativegl::TexSubImageFormat texSubImageFormat =
        nativegl::GetTexSubImageFormat(mFunctions, mFeatures, destFormat, destType);

    mStateManager->bindTexture(dest->getType(), dest->getTextureID());
    ANGLE_GL_TRY(context, mFunctions->texSubImage2D(
                              ToGLenum(destTarget), static_cast<GLint>(destLevel), destOffset.x,
                              destOffset.y, readPixelsArea.width, readPixelsArea.height,
                              texSubImageFormat.format, texSubImageFormat.type, destMemory));

    return angle::Result::Continue;
}

angle::Result BlitGL::copyTexSubImage(const gl::Context *context,
                                      TextureGL *source,
                                      size_t sourceLevel,
                                      TextureGL *dest,
                                      gl::TextureTarget destTarget,
                                      size_t destLevel,
                                      const gl::Rectangle &sourceArea,
                                      const gl::Offset &destOffset,
                                      bool *copySucceededOut)
{
    ANGLE_TRY(initializeResources(context));

    // Make sure the source texture can create a complete framebuffer before continuing.
    mStateManager->bindFramebuffer(GL_FRAMEBUFFER, mScratchFBO);
    ANGLE_GL_TRY(context, mFunctions->framebufferTexture2D(
                              GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, ToGLenum(source->getType()),
                              source->getTextureID(), static_cast<GLint>(sourceLevel)));
    GLenum status = ANGLE_GL_TRY(context, mFunctions->checkFramebufferStatus(GL_FRAMEBUFFER));
    if (status != GL_FRAMEBUFFER_COMPLETE)
    {
        *copySucceededOut = false;
        return angle::Result::Continue;
    }

    mStateManager->bindTexture(dest->getType(), dest->getTextureID());

    ANGLE_GL_TRY(context,
                 mFunctions->copyTexSubImage2D(ToGLenum(destTarget), static_cast<GLint>(destLevel),
                                               destOffset.x, destOffset.y, sourceArea.x,
                                               sourceArea.y, sourceArea.width, sourceArea.height));

    *copySucceededOut = true;
    return angle::Result::Continue;
}

angle::Result BlitGL::clearRenderableTexture(const gl::Context *context,
                                             TextureGL *source,
                                             GLenum sizedInternalFormat,
                                             int numTextureLayers,
                                             const gl::ImageIndex &imageIndex,
                                             bool *clearSucceededOut)
{
    ANGLE_TRY(initializeResources(context));

    ClearBindTargetVector bindTargets;
    ClearBindTargetVector unbindTargets;
    GLbitfield clearMask = 0;
    ANGLE_TRY(PrepareForClear(mStateManager, sizedInternalFormat, &bindTargets, &unbindTargets,
                              &clearMask));

    mStateManager->bindFramebuffer(GL_FRAMEBUFFER, mScratchFBO);
    ANGLE_TRY(UnbindAttachments(context, mFunctions, GL_FRAMEBUFFER, unbindTargets));

    if (nativegl::UseTexImage2D(source->getType()))
    {
        ASSERT(numTextureLayers == 1);
        for (GLenum bindTarget : bindTargets)
        {
            ANGLE_GL_TRY(context, mFunctions->framebufferTexture2D(
                                      GL_FRAMEBUFFER, bindTarget, ToGLenum(imageIndex.getTarget()),
                                      source->getTextureID(), imageIndex.getLevelIndex()));
        }

        GLenum status = ANGLE_GL_TRY(context, mFunctions->checkFramebufferStatus(GL_FRAMEBUFFER));
        if (status == GL_FRAMEBUFFER_COMPLETE)
        {
            ANGLE_GL_TRY(context, mFunctions->clear(clearMask));
        }
        else
        {
            ANGLE_TRY(UnbindAttachments(context, mFunctions, GL_FRAMEBUFFER, bindTargets));
            *clearSucceededOut = false;
            return angle::Result::Continue;
        }
    }
    else
    {
        ASSERT(nativegl::UseTexImage3D(source->getType()));

        // Check if it's possible to bind all layers of the texture at once
        if (mFunctions->framebufferTexture && !imageIndex.hasLayer())
        {
            for (GLenum bindTarget : bindTargets)
            {
                ANGLE_GL_TRY(context, mFunctions->framebufferTexture(GL_FRAMEBUFFER, bindTarget,
                                                                     source->getTextureID(),
                                                                     imageIndex.getLevelIndex()));
            }

            GLenum status =
                ANGLE_GL_TRY(context, mFunctions->checkFramebufferStatus(GL_FRAMEBUFFER));
            if (status == GL_FRAMEBUFFER_COMPLETE)
            {
                ANGLE_GL_TRY(context, mFunctions->clear(clearMask));
            }
            else
            {
                ANGLE_TRY(UnbindAttachments(context, mFunctions, GL_FRAMEBUFFER, bindTargets));
                *clearSucceededOut = false;
                return angle::Result::Continue;
            }
        }
        else
        {
            GLint firstLayer = 0;
            GLint layerCount = numTextureLayers;
            if (imageIndex.hasLayer())
            {
                firstLayer = imageIndex.getLayerIndex();
                layerCount = imageIndex.getLayerCount();
            }

            for (GLint layer = 0; layer < layerCount; layer++)
            {
                for (GLenum bindTarget : bindTargets)
                {
                    ANGLE_GL_TRY(context, mFunctions->framebufferTextureLayer(
                                              GL_FRAMEBUFFER, bindTarget, source->getTextureID(),
                                              imageIndex.getLevelIndex(), layer + firstLayer));
                }

                GLenum status =
                    ANGLE_GL_TRY(context, mFunctions->checkFramebufferStatus(GL_FRAMEBUFFER));
                if (status == GL_FRAMEBUFFER_COMPLETE)
                {
                    ANGLE_GL_TRY(context, mFunctions->clear(clearMask));
                }
                else
                {
                    ANGLE_TRY(UnbindAttachments(context, mFunctions, GL_FRAMEBUFFER, bindTargets));
                    *clearSucceededOut = false;
                    return angle::Result::Continue;
                }
            }
        }
    }

    ANGLE_TRY(UnbindAttachments(context, mFunctions, GL_FRAMEBUFFER, bindTargets));
    *clearSucceededOut = true;
    return angle::Result::Continue;
}

angle::Result BlitGL::clearRenderbuffer(const gl::Context *context,
                                        RenderbufferGL *source,
                                        GLenum sizedInternalFormat)
{
    ANGLE_TRY(initializeResources(context));

    ClearBindTargetVector bindTargets;
    ClearBindTargetVector unbindTargets;
    GLbitfield clearMask = 0;
    ANGLE_TRY(PrepareForClear(mStateManager, sizedInternalFormat, &bindTargets, &unbindTargets,
                              &clearMask));

    mStateManager->bindFramebuffer(GL_FRAMEBUFFER, mScratchFBO);
    ANGLE_TRY(UnbindAttachments(context, mFunctions, GL_FRAMEBUFFER, unbindTargets));

    for (GLenum bindTarget : bindTargets)
    {
        ANGLE_GL_TRY(context,
                     mFunctions->framebufferRenderbuffer(
                         GL_FRAMEBUFFER, bindTarget, GL_RENDERBUFFER, source->getRenderbufferID()));
    }
    ANGLE_GL_TRY(context, mFunctions->clear(clearMask));

    // Unbind
    for (GLenum bindTarget : bindTargets)
    {
        ANGLE_GL_TRY(context, mFunctions->framebufferRenderbuffer(GL_FRAMEBUFFER, bindTarget,
                                                                  GL_RENDERBUFFER, 0));
    }

    return angle::Result::Continue;
}

angle::Result BlitGL::clearFramebuffer(const gl::Context *context, FramebufferGL *source)
{
    // initializeResources skipped because no local state is used

    // Clear all attachments
    GLbitfield clearMask = 0;
    ANGLE_TRY(SetClearState(mStateManager, true, true, true, &clearMask));

    mStateManager->bindFramebuffer(GL_FRAMEBUFFER, source->getFramebufferID());
    ANGLE_GL_TRY(context, mFunctions->clear(clearMask));

    return angle::Result::Continue;
}

angle::Result BlitGL::clearRenderableTextureAlphaToOne(const gl::Context *context,
                                                       GLuint texture,
                                                       gl::TextureTarget target,
                                                       size_t level)
{
    // Clearing the alpha of 3D textures is not supported/needed yet.
    ASSERT(nativegl::UseTexImage2D(TextureTargetToType(target)));

    ANGLE_TRY(initializeResources(context));

    mStateManager->setClearColor(gl::ColorF(0.0f, 0.0f, 0.0f, 1.0f));
    mStateManager->setColorMask(false, false, false, true);
    mStateManager->setScissorTestEnabled(false);

    mStateManager->bindFramebuffer(GL_FRAMEBUFFER, mScratchFBO);
    ANGLE_GL_TRY(context, mFunctions->framebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                                           ToGLenum(target), texture,
                                                           static_cast<GLint>(level)));
    ANGLE_GL_TRY(context, mFunctions->clear(GL_COLOR_BUFFER_BIT));

    // Unbind the texture from the the scratch framebuffer
    ANGLE_GL_TRY(context, mFunctions->framebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                                              GL_RENDERBUFFER, 0));

    return angle::Result::Continue;
}

angle::Result BlitGL::initializeResources(const gl::Context *context)
{
    for (size_t i = 0; i < ArraySize(mScratchTextures); i++)
    {
        if (mScratchTextures[i] == 0)
        {
            ANGLE_GL_TRY(context, mFunctions->genTextures(1, &mScratchTextures[i]));
        }
    }

    if (mScratchFBO == 0)
    {
        ANGLE_GL_TRY(context, mFunctions->genFramebuffers(1, &mScratchFBO));
    }

    if (mVertexBuffer == 0)
    {
        ANGLE_GL_TRY(context, mFunctions->genBuffers(1, &mVertexBuffer));
        mStateManager->bindBuffer(gl::BufferBinding::Array, mVertexBuffer);

        // Use a single, large triangle, to avoid arithmetic precision issues where fragments
        // with the same Y coordinate don't get exactly the same interpolated texcoord Y.
        float vertexData[] = {
            -0.5f, 0.0f, 1.5f, 0.0f, 0.5f, 2.0f,
        };

        ANGLE_GL_TRY(context, mFunctions->bufferData(GL_ARRAY_BUFFER, sizeof(float) * 6, vertexData,
                                                     GL_STATIC_DRAW));
    }

    if (mVAO == 0)
    {
        ANGLE_GL_TRY(context, mFunctions->genVertexArrays(1, &mVAO));

        mStateManager->bindVertexArray(mVAO, 0);
        mStateManager->bindBuffer(gl::BufferBinding::Array, mVertexBuffer);

        // Enable all attributes with the same buffer so that it doesn't matter what location the
        // texcoord attribute is assigned
        GLint maxAttributes = 0;
        ANGLE_GL_TRY(context, mFunctions->getIntegerv(GL_MAX_VERTEX_ATTRIBS, &maxAttributes));

        for (GLint i = 0; i < maxAttributes; i++)
        {
            ANGLE_GL_TRY(context, mFunctions->enableVertexAttribArray(i));
            ANGLE_GL_TRY(context,
                         mFunctions->vertexAttribPointer(i, 2, GL_FLOAT, GL_FALSE, 0, nullptr));
        }
    }

    return angle::Result::Continue;
}

angle::Result BlitGL::orphanScratchTextures(const gl::Context *context)
{
    for (auto texture : mScratchTextures)
    {
        mStateManager->bindTexture(gl::TextureType::_2D, texture);
        gl::PixelUnpackState unpack;
        mStateManager->setPixelUnpackState(unpack);
        mStateManager->setPixelUnpackBuffer(nullptr);
        ANGLE_GL_TRY(context, mFunctions->texImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 0, 0, 0, GL_RGBA,
                                                     GL_UNSIGNED_BYTE, nullptr));
    }
    return angle::Result::Continue;
}

angle::Result BlitGL::setScratchTextureParameter(const gl::Context *context,
                                                 GLenum param,
                                                 GLenum value)
{
    for (auto texture : mScratchTextures)
    {
        mStateManager->bindTexture(gl::TextureType::_2D, texture);
        ANGLE_GL_TRY(context, mFunctions->texParameteri(GL_TEXTURE_2D, param, value));
        ANGLE_GL_TRY(context, mFunctions->texParameteri(GL_TEXTURE_2D, param, value));
    }
    return angle::Result::Continue;
}

angle::Result BlitGL::getBlitProgram(const gl::Context *context,
                                     gl::TextureType sourceTextureType,
                                     GLenum sourceComponentType,
                                     GLenum destComponentType,
                                     BlitProgram **program)
{

    BlitProgramType programType(sourceTextureType, sourceComponentType, destComponentType);
    BlitProgram &result = mBlitPrograms[programType];
    if (result.program == 0)
    {
        result.program = ANGLE_GL_TRY(context, mFunctions->createProgram());

        // Depending on what types need to be output by the shaders, different versions need to be
        // used.
        std::string version;
        std::string vsInputVariableQualifier;
        std::string vsOutputVariableQualifier;
        std::string fsInputVariableQualifier;
        std::string fsOutputVariableQualifier;
        std::string sampleFunction;
        if (sourceComponentType != GL_UNSIGNED_INT && destComponentType != GL_UNSIGNED_INT &&
            sourceTextureType != gl::TextureType::Rectangle)
        {
            // Simple case, float-to-float with 2D or external textures.  Only needs ESSL/GLSL 100
            version                   = "100";
            vsInputVariableQualifier  = "attribute";
            vsOutputVariableQualifier = "varying";
            fsInputVariableQualifier  = "varying";
            fsOutputVariableQualifier = "";
            sampleFunction            = "texture2D";
        }
        else
        {
            // Need to use a higher version to support non-float output types
            if (mFunctions->standard == STANDARD_GL_DESKTOP)
            {
                version = "330";
            }
            else
            {
                ASSERT(mFunctions->standard == STANDARD_GL_ES);
                version = "300 es";
            }
            vsInputVariableQualifier  = "in";
            vsOutputVariableQualifier = "out";
            fsInputVariableQualifier  = "in";
            fsOutputVariableQualifier = "out";
            sampleFunction            = "texture";
        }

        {
            // Compile the vertex shader
            std::ostringstream vsSourceStream;
            vsSourceStream << "#version " << version << "\n";
            vsSourceStream << vsInputVariableQualifier << " vec2 a_texcoord;\n";
            vsSourceStream << "uniform vec2 u_scale;\n";
            vsSourceStream << "uniform vec2 u_offset;\n";
            vsSourceStream << vsOutputVariableQualifier << " vec2 v_texcoord;\n";
            vsSourceStream << "\n";
            vsSourceStream << "void main()\n";
            vsSourceStream << "{\n";
            vsSourceStream << "    gl_Position = vec4((a_texcoord * 2.0) - 1.0, 0.0, 1.0);\n";
            vsSourceStream << "    v_texcoord = a_texcoord * u_scale + u_offset;\n";
            vsSourceStream << "}\n";

            std::string vsSourceStr  = vsSourceStream.str();
            const char *vsSourceCStr = vsSourceStr.c_str();

            GLuint vs = ANGLE_GL_TRY(context, mFunctions->createShader(GL_VERTEX_SHADER));
            ANGLE_GL_TRY(context, mFunctions->shaderSource(vs, 1, &vsSourceCStr, nullptr));
            ANGLE_GL_TRY(context, mFunctions->compileShader(vs));
            ANGLE_TRY(CheckCompileStatus(context, mFunctions, vs));

            ANGLE_GL_TRY(context, mFunctions->attachShader(result.program, vs));
            ANGLE_GL_TRY(context, mFunctions->deleteShader(vs));
        }

        {
            // Sampling texture uniform changes depending on source texture type.
            std::string samplerType;
            switch (sourceTextureType)
            {
                case gl::TextureType::_2D:
                    switch (sourceComponentType)
                    {
                        case GL_UNSIGNED_INT:
                            samplerType = "usampler2D";
                            break;

                        default:  // Float type
                            samplerType = "sampler2D";
                            break;
                    }
                    break;

                case gl::TextureType::External:
                    ASSERT(sourceComponentType != GL_UNSIGNED_INT);
                    samplerType = "samplerExternalOES";
                    break;

                case gl::TextureType::Rectangle:
                    ASSERT(sourceComponentType != GL_UNSIGNED_INT);
                    samplerType = "sampler2DRect";
                    break;

                default:
                    UNREACHABLE();
                    break;
            }

            std::string samplerResultType;
            switch (sourceComponentType)
            {
                case GL_UNSIGNED_INT:
                    samplerResultType = "uvec4";
                    break;

                default:  // Float type
                    samplerResultType = "vec4";
                    break;
            }

            std::string extensionRequirements;
            switch (sourceTextureType)
            {
                case gl::TextureType::External:
                    extensionRequirements = "#extension GL_OES_EGL_image_external : require";
                    break;

                case gl::TextureType::Rectangle:
                    if (mFunctions->hasGLExtension("GL_ARB_texture_rectangle"))
                    {
                        extensionRequirements = "#extension GL_ARB_texture_rectangle : require";
                    }
                    else
                    {
                        ASSERT(mFunctions->isAtLeastGL(gl::Version(3, 1)));
                    }
                    break;

                default:
                    break;
            }

            // Output variables depend on the output type
            std::string outputType;
            std::string outputVariableName;
            std::string outputMultiplier;
            switch (destComponentType)
            {
                case GL_UNSIGNED_INT:
                    outputType         = "uvec4";
                    outputVariableName = "outputUint";
                    outputMultiplier   = "255.0";
                    break;

                default:  //  float type
                    if (version == "100")
                    {
                        outputType         = "";
                        outputVariableName = "gl_FragColor";
                        outputMultiplier   = "1.0";
                    }
                    else
                    {
                        outputType         = "vec4";
                        outputVariableName = "outputFloat";
                        outputMultiplier   = "1.0";
                    }
                    break;
            }

            // Compile the fragment shader
            std::ostringstream fsSourceStream;
            fsSourceStream << "#version " << version << "\n";
            fsSourceStream << extensionRequirements << "\n";
            fsSourceStream << "precision highp float;\n";
            fsSourceStream << "uniform " << samplerType << " u_source_texture;\n";

            // Write the rest of the uniforms and varyings
            fsSourceStream << "uniform bool u_multiply_alpha;\n";
            fsSourceStream << "uniform bool u_unmultiply_alpha;\n";
            fsSourceStream << fsInputVariableQualifier << " vec2 v_texcoord;\n";
            if (!outputType.empty())
            {
                fsSourceStream << fsOutputVariableQualifier << " " << outputType << " "
                               << outputVariableName << ";\n";
            }

            // Write the main body
            fsSourceStream << "\n";
            fsSourceStream << "void main()\n";
            fsSourceStream << "{\n";

            std::string maxTexcoord;
            switch (sourceTextureType)
            {
                case gl::TextureType::Rectangle:
                    // Valid texcoords are within source texture size
                    maxTexcoord = "vec2(textureSize(u_source_texture))";
                    break;

                default:
                    // Valid texcoords are in [0, 1]
                    maxTexcoord = "vec2(1.0)";
                    break;
            }

            // discard if the texcoord is invalid so the blitframebuffer workaround doesn't
            // write when the point sampled is outside of the source framebuffer.
            fsSourceStream << "    if (clamp(v_texcoord, vec2(0.0), " << maxTexcoord
                           << ") != v_texcoord)\n";
            fsSourceStream << "    {\n";
            fsSourceStream << "        discard;\n";
            fsSourceStream << "    }\n";

            // Sampling code depends on the input data type
            fsSourceStream << "    " << samplerResultType << " color = " << sampleFunction
                           << "(u_source_texture, v_texcoord);\n";

            // Perform the premultiply or unmultiply alpha logic
            fsSourceStream << "    if (u_multiply_alpha)\n";
            fsSourceStream << "    {\n";
            fsSourceStream << "        color.xyz = color.xyz * color.a;\n";
            fsSourceStream << "    }\n";
            fsSourceStream << "    if (u_unmultiply_alpha && color.a != 0.0)\n";
            fsSourceStream << "    {\n";
            fsSourceStream << "         color.xyz = color.xyz / color.a;\n";
            fsSourceStream << "    }\n";

            // Write the conversion to the destionation type
            fsSourceStream << "    color = color * " << outputMultiplier << ";\n";

            // Write the output assignment code
            fsSourceStream << "    " << outputVariableName << " = " << outputType << "(color);\n";
            fsSourceStream << "}\n";

            std::string fsSourceStr  = fsSourceStream.str();
            const char *fsSourceCStr = fsSourceStr.c_str();

            GLuint fs = ANGLE_GL_TRY(context, mFunctions->createShader(GL_FRAGMENT_SHADER));
            ANGLE_GL_TRY(context, mFunctions->shaderSource(fs, 1, &fsSourceCStr, nullptr));
            ANGLE_GL_TRY(context, mFunctions->compileShader(fs));
            ANGLE_TRY(CheckCompileStatus(context, mFunctions, fs));

            ANGLE_GL_TRY(context, mFunctions->attachShader(result.program, fs));
            ANGLE_GL_TRY(context, mFunctions->deleteShader(fs));
        }

        ANGLE_GL_TRY(context, mFunctions->linkProgram(result.program));
        ANGLE_TRY(CheckLinkStatus(context, mFunctions, result.program));

        result.sourceTextureLocation = ANGLE_GL_TRY(
            context, mFunctions->getUniformLocation(result.program, "u_source_texture"));
        result.scaleLocation =
            ANGLE_GL_TRY(context, mFunctions->getUniformLocation(result.program, "u_scale"));
        result.offsetLocation =
            ANGLE_GL_TRY(context, mFunctions->getUniformLocation(result.program, "u_offset"));
        result.multiplyAlphaLocation = ANGLE_GL_TRY(
            context, mFunctions->getUniformLocation(result.program, "u_multiply_alpha"));
        result.unMultiplyAlphaLocation = ANGLE_GL_TRY(
            context, mFunctions->getUniformLocation(result.program, "u_unmultiply_alpha"));
    }

    *program = &result;
    return angle::Result::Continue;
}

}  // namespace rx
