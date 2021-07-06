//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// SurfaceMtl.mm:
//    Implements the class methods for SurfaceMtl.
//

#include "libANGLE/renderer/metal/SurfaceMtl.h"

#include <TargetConditionals.h>

#include "libANGLE/Display.h"
#include "libANGLE/Surface.h"
#include "libANGLE/renderer/metal/ContextMtl.h"
#include "libANGLE/renderer/metal/DisplayMtl.h"
#include "libANGLE/renderer/metal/FrameBufferMtl.h"
#include "libANGLE/renderer/metal/mtl_format_utils.h"

// Compiler can turn on programmatical frame capture in release build by defining
// ANGLE_METAL_FRAME_CAPTURE flag.
#if defined(NDEBUG) && !defined(ANGLE_METAL_FRAME_CAPTURE)
#    define ANGLE_METAL_FRAME_CAPTURE_ENABLED 0
#else
#    define ANGLE_METAL_FRAME_CAPTURE_ENABLED 1
#endif

namespace rx
{

namespace
{

#define ANGLE_TO_EGL_TRY(EXPR)                                 \
    do                                                         \
    {                                                          \
        if (ANGLE_UNLIKELY((EXPR) != angle::Result::Continue)) \
        {                                                      \
            return egl::EglBadSurface();                       \
        }                                                      \
    } while (0)

constexpr angle::FormatID kDefaultFrameBufferDepthFormatId   = angle::FormatID::D32_FLOAT;
constexpr angle::FormatID kDefaultFrameBufferStencilFormatId = angle::FormatID::S8_UINT;
constexpr angle::FormatID kDefaultFrameBufferDepthStencilFormatId =
    angle::FormatID::D24_UNORM_S8_UINT;

void CopyTextureNoScale(ContextMtl *contextMtl,
                        const mtl::TextureRef &src,
                        const mtl::TextureRef *dst)
{
    mtl::BlitCommandEncoder *encoder = contextMtl->getBlitCommandEncoder();
    encoder->copyTexture(src, 0, 0, MTLOriginMake(0, 0, 0),
                         MTLSizeMake(src->width(), src->height(), 1), *dst, 0, 0,
                         MTLOriginMake(0, 0, 0));
    contextMtl->endEncoding(true);
}

angle::Result CreateOrResizeTexture(const gl::Context *context,
                                    const mtl::Format &format,
                                    uint32_t width,
                                    uint32_t height,
                                    uint32_t samples,
                                    bool renderTargetOnly,
                                    mtl::TextureRef *textureOut)
{
    ContextMtl *contextMtl = mtl::GetImpl(context);
    if (*textureOut)
    {
        ANGLE_TRY((*textureOut)->resize(contextMtl, width, height));
    }
    else if (samples > 1)
    {
        ANGLE_TRY(mtl::Texture::Make2DMSTexture(contextMtl, format, width, height, samples,
                                                /** renderTargetOnly */ renderTargetOnly,
                                                /** allowFormatView */ false, textureOut));
    }
    else
    {
        ANGLE_TRY(mtl::Texture::Make2DTexture(contextMtl, format, width, height, 1,
                                              /** renderTargetOnly */ renderTargetOnly,
                                              /** allowFormatView */ false, textureOut));
    }
    return angle::Result::Continue;
}

void InitBlitParams(const mtl::TextureRef &src,
                    const mtl::TextureRef &dst,
                    mtl::BlitParams *paramsOut)
{
    mtl::BlitParams &params = *paramsOut;
    params.src              = src;
    params.srcRect =
        gl::Rectangle(0, 0, static_cast<int>(src->width()), static_cast<int>(src->height()));
    params.dstTextureSize =
        gl::Extents(static_cast<int>(dst->width()), static_cast<int>(dst->height()), 1);
    params.dstRect = params.dstScissorRect =
        gl::Rectangle(0, 0, params.dstTextureSize.width, params.dstTextureSize.height);
}

ANGLE_MTL_UNUSED
bool IsFrameCaptureEnabled()
{
#if !ANGLE_METAL_FRAME_CAPTURE_ENABLED
    return false;
#else
    // We only support frame capture programmatically if the ANGLE_METAL_FRAME_CAPTURE
    // environment flag is set. Otherwise, it will slow down the rendering. This allows user to
    // finely control whether he wants to capture the frame for particular application or not.
    auto var                  = std::getenv("ANGLE_METAL_FRAME_CAPTURE");
    static const bool enabled = var ? (strcmp(var, "1") == 0) : false;

    return enabled;
#endif
}

ANGLE_MTL_UNUSED
size_t MaxAllowedFrameCapture()
{
#if !ANGLE_METAL_FRAME_CAPTURE_ENABLED
    return 0;
#else
    auto var                      = std::getenv("ANGLE_METAL_FRAME_CAPTURE_MAX");
    static const size_t maxFrames = var ? std::atoi(var) : 100;

    return maxFrames;
#endif
}

ANGLE_MTL_UNUSED
size_t MinAllowedFrameCapture()
{
#if !ANGLE_METAL_FRAME_CAPTURE_ENABLED
    return 0;
#else
    auto var                     = std::getenv("ANGLE_METAL_FRAME_CAPTURE_MIN");
    static const size_t minFrame = var ? std::atoi(var) : 0;

    return minFrame;
#endif
}

ANGLE_MTL_UNUSED
bool FrameCaptureDeviceScope()
{
#if !ANGLE_METAL_FRAME_CAPTURE_ENABLED
    return false;
#else
    auto var                      = std::getenv("ANGLE_METAL_FRAME_CAPTURE_SCOPE");
    static const bool scopeDevice = var ? (strcmp(var, "device") == 0) : false;

    return scopeDevice;
#endif
}

ANGLE_MTL_UNUSED
std::atomic<size_t> gFrameCaptured(0);

ANGLE_MTL_UNUSED
void StartFrameCapture(id<MTLDevice> metalDevice, id<MTLCommandQueue> metalCmdQueue)
{
#if ANGLE_METAL_FRAME_CAPTURE_ENABLED
    if (!IsFrameCaptureEnabled())
    {
        return;
    }

    if (gFrameCaptured >= MaxAllowedFrameCapture())
    {
        return;
    }

    MTLCaptureManager *captureManager = [MTLCaptureManager sharedCaptureManager];
    if (captureManager.isCapturing)
    {
        return;
    }

    gFrameCaptured++;

    if (gFrameCaptured < MinAllowedFrameCapture())
    {
        return;
    }

#    ifdef __MAC_10_15
    if (ANGLE_APPLE_AVAILABLE_XCI(10.15, 13.0, 13))
    {
        MTLCaptureDescriptor *captureDescriptor = [[MTLCaptureDescriptor alloc] init];
        captureDescriptor.captureObject         = metalDevice;

        NSError *error;
        if (![captureManager startCaptureWithDescriptor:captureDescriptor error:&error])
        {
            NSLog(@"Failed to start capture, error %@", error);
        }
    }
    else
#    endif  // __MAC_10_15
    {
        if (FrameCaptureDeviceScope())
        {
            [captureManager startCaptureWithDevice:metalDevice];
        }
        else
        {
            [captureManager startCaptureWithCommandQueue:metalCmdQueue];
        }
    }
#endif  // ANGLE_METAL_FRAME_CAPTURE_ENABLED
}

void StartFrameCapture(ContextMtl *context)
{
    StartFrameCapture(context->getMetalDevice(), context->cmdQueue().get());
}

void StopFrameCapture()
{
#if ANGLE_METAL_FRAME_CAPTURE_ENABLED
    if (!IsFrameCaptureEnabled())
    {
        return;
    }
    MTLCaptureManager *captureManager = [MTLCaptureManager sharedCaptureManager];
    if (captureManager.isCapturing)
    {
        [captureManager stopCapture];
    }
#endif
}
}

// SurfaceMtl implementation
SurfaceMtl::SurfaceMtl(DisplayMtl *display,
                       const egl::SurfaceState &state,
                       const egl::AttributeMap &attribs)
    : SurfaceImpl(state)
{
    if (attribs.get(EGL_GL_COLORSPACE, EGL_GL_COLORSPACE_LINEAR) == EGL_GL_COLORSPACE_SRGB_KHR)
    {
        mColorFormat = display->getPixelFormat(angle::FormatID::B8G8R8A8_UNORM_SRGB);
    }
    else
    {
        mColorFormat = display->getPixelFormat(angle::FormatID::B8G8R8A8_UNORM);
    }

    mSamples = state.config->samples;

    int depthBits   = 0;
    int stencilBits = 0;
    if (state.config)
    {
        depthBits   = state.config->depthSize;
        stencilBits = state.config->stencilSize;
    }

    if (depthBits && stencilBits)
    {
        if (display->getFeatures().allowSeparatedDepthStencilBuffers.enabled)
        {
            mDepthFormat   = display->getPixelFormat(kDefaultFrameBufferDepthFormatId);
            mStencilFormat = display->getPixelFormat(kDefaultFrameBufferStencilFormatId);
        }
        else
        {
            // We must use packed depth stencil
            mUsePackedDepthStencil = true;
            mDepthFormat   = display->getPixelFormat(kDefaultFrameBufferDepthStencilFormatId);
            mStencilFormat = mDepthFormat;
        }
    }
    else if (depthBits)
    {
        mDepthFormat = display->getPixelFormat(kDefaultFrameBufferDepthFormatId);
    }
    else if (stencilBits)
    {
        mStencilFormat = display->getPixelFormat(kDefaultFrameBufferStencilFormatId);
    }
}

SurfaceMtl::~SurfaceMtl() {}

void SurfaceMtl::destroy(const egl::Display *display)
{
    mColorTexture   = nullptr;
    mDepthTexture   = nullptr;
    mStencilTexture = nullptr;

    mMSColorTexture = nullptr;

    mColorRenderTarget.reset();
    mColorManualResolveRenderTarget.reset();
    mDepthRenderTarget.reset();
    mStencilRenderTarget.reset();
}

egl::Error SurfaceMtl::initialize(const egl::Display *display)
{
    return egl::NoError();
}

FramebufferImpl *SurfaceMtl::createDefaultFramebuffer(const gl::Context *context,
                                                      const gl::FramebufferState &state)
{
    auto fbo = new FramebufferMtl(state, /* flipY */ false, /* backbuffer */ nullptr);

    return fbo;
}

egl::Error SurfaceMtl::makeCurrent(const gl::Context *context)
{
    ContextMtl *contextMtl = mtl::GetImpl(context);
    StartFrameCapture(contextMtl);

    return egl::NoError();
}

egl::Error SurfaceMtl::unMakeCurrent(const gl::Context *context)
{
    ContextMtl *contextMtl = mtl::GetImpl(context);
    contextMtl->flushCommandBufer();

    StopFrameCapture();
    return egl::NoError();
}

egl::Error SurfaceMtl::swap(const gl::Context *context)
{
    return egl::NoError();
}

egl::Error SurfaceMtl::postSubBuffer(const gl::Context *context,
                                     EGLint x,
                                     EGLint y,
                                     EGLint width,
                                     EGLint height)
{
    UNIMPLEMENTED();
    return egl::EglBadAccess();
}

egl::Error SurfaceMtl::querySurfacePointerANGLE(EGLint attribute, void **value)
{
    UNIMPLEMENTED();
    return egl::EglBadAccess();
}

egl::Error SurfaceMtl::bindTexImage(const gl::Context *context, gl::Texture *texture, EGLint buffer)
{
    UNIMPLEMENTED();
    return egl::EglBadAccess();
}

egl::Error SurfaceMtl::releaseTexImage(const gl::Context *context, EGLint buffer)
{
    UNIMPLEMENTED();
    return egl::EglBadAccess();
}

egl::Error SurfaceMtl::getSyncValues(EGLuint64KHR *ust, EGLuint64KHR *msc, EGLuint64KHR *sbc)
{
    UNIMPLEMENTED();
    return egl::EglBadAccess();
}

void SurfaceMtl::setSwapInterval(EGLint interval) {}

void SurfaceMtl::setFixedWidth(EGLint width)
{
    UNIMPLEMENTED();
}

void SurfaceMtl::setFixedHeight(EGLint height)
{
    UNIMPLEMENTED();
}

EGLint SurfaceMtl::getWidth() const
{
    if (mColorTexture)
    {
        return static_cast<EGLint>(mColorTexture->width());
    }
    return 0;
}

EGLint SurfaceMtl::getHeight() const
{
    if (mColorTexture)
    {
        return static_cast<EGLint>(mColorTexture->height());
    }
    return 0;
}

EGLint SurfaceMtl::isPostSubBufferSupported() const
{
    return EGL_FALSE;
}

EGLint SurfaceMtl::getSwapBehavior() const
{
    return EGL_BUFFER_PRESERVED;
}

angle::Result SurfaceMtl::getAttachmentRenderTarget(const gl::Context *context,
                                                    GLenum binding,
                                                    const gl::ImageIndex &imageIndex,
                                                    GLsizei samples,
                                                    FramebufferAttachmentRenderTarget **rtOut)
{
    ASSERT(mColorTexture);

    switch (binding)
    {
        case GL_BACK:
            *rtOut = &mColorRenderTarget;
            break;
        case GL_DEPTH:
            *rtOut = mDepthFormat.valid() ? &mDepthRenderTarget : nullptr;
            break;
        case GL_STENCIL:
            *rtOut = mStencilFormat.valid() ? &mStencilRenderTarget : nullptr;
            break;
        case GL_DEPTH_STENCIL:
            // NOTE(hqle): ES 3.0 feature
            UNREACHABLE();
            break;
    }

    return angle::Result::Continue;
}

angle::Result SurfaceMtl::ensureCompanionTexturesSizeCorrect(const gl::Context *context,
                                                             const gl::Extents &size)
{
    ContextMtl *contextMtl = mtl::GetImpl(context);

    ASSERT(mColorTexture);

    if (mSamples > 1 && (!mMSColorTexture || mMSColorTexture->size() != size))
    {
        mAutoResolveMSColorTexture =
            contextMtl->getDisplay()->getFeatures().allowMultisampleStoreAndResolve.enabled;
        ANGLE_TRY(CreateOrResizeTexture(context, mColorFormat, size.width, size.height, mSamples,
                                        /** renderTargetOnly */ mAutoResolveMSColorTexture,
                                        &mMSColorTexture));

        if (mAutoResolveMSColorTexture)
        {
            // Use auto MSAA resolve at the end of render pass.
            mColorRenderTarget.setImplicitMSTexture(mMSColorTexture);
        }
        else
        {
            mColorRenderTarget.setTexture(mMSColorTexture);
        }
    }

    if (mDepthFormat.valid() && (!mDepthTexture || mDepthTexture->size() != size))
    {
        ANGLE_TRY(CreateOrResizeTexture(context, mDepthFormat, size.width, size.height, mSamples,
                                        /** renderTargetOnly */ true, &mDepthTexture));

        mDepthRenderTarget.set(mDepthTexture, 0, 0, mDepthFormat);
    }

    if (mStencilFormat.valid() && (!mStencilTexture || mStencilTexture->size() != size))
    {
        if (mUsePackedDepthStencil)
        {
            mStencilTexture = mDepthTexture;
        }
        else
        {
            ANGLE_TRY(CreateOrResizeTexture(context, mStencilFormat, size.width, size.height,
                                            mSamples,
                                            /** renderTargetOnly */ true, &mStencilTexture));
        }

        mStencilRenderTarget.set(mStencilTexture, 0, 0, mStencilFormat);
    }

    return angle::Result::Continue;
}

angle::Result SurfaceMtl::resolveColorTextureIfNeeded(const gl::Context *context)
{
    ASSERT(mMSColorTexture);
    if (!mAutoResolveMSColorTexture)
    {
        // Manually resolve texture
        ContextMtl *contextMtl = mtl::GetImpl(context);

        mColorManualResolveRenderTarget.set(mColorTexture, 0, 0, mColorFormat);
        mtl::RenderCommandEncoder *encoder =
            contextMtl->getRenderCommandEncoder(mColorManualResolveRenderTarget);
        ANGLE_TRY(contextMtl->getDisplay()->getUtils().blitColorWithDraw(
            context, encoder, mColorFormat.actualAngleFormat(), mMSColorTexture));
        contextMtl->endEncoding(true);
        mColorManualResolveRenderTarget.reset();
    }
    return angle::Result::Continue;
}

// WindowSurfaceMtl implementation.
WindowSurfaceMtl::WindowSurfaceMtl(DisplayMtl *display,
                                   const egl::SurfaceState &state,
                                   EGLNativeWindowType window,
                                   const egl::AttributeMap &attribs)
    : SurfaceMtl(display, state, attribs), mLayer((__bridge CALayer *)(window))
{
    // NOTE(hqle): Width and height attributes is ignored for now.
    mCurrentKnownDrawableSize = CGSizeMake(0, 0);
}

WindowSurfaceMtl::~WindowSurfaceMtl() {}

void WindowSurfaceMtl::destroy(const egl::Display *display)
{
    SurfaceMtl::destroy(display);

    mRetainedColorTexture = nullptr;
    mRetainBuffer         = false;

    mCurrentDrawable = nil;
    if (mMetalLayer && mMetalLayer.get() != mLayer)
    {
        // If we created metal layer in WindowSurfaceMtl::initialize(),
        // we need to detach it from super layer now.
        [mMetalLayer.get() removeFromSuperlayer];
    }
    mMetalLayer = nil;
}

egl::Error WindowSurfaceMtl::initialize(const egl::Display *display)
{
    egl::Error re = SurfaceMtl::initialize(display);
    if (re.isError())
    {
        return re;
    }

    DisplayMtl *displayMtl    = mtl::GetImpl(display);
    id<MTLDevice> metalDevice = displayMtl->getMetalDevice();

    StartFrameCapture(metalDevice, displayMtl->cmdQueue().get());

    ANGLE_MTL_OBJC_SCOPE
    {
        if ([mLayer isKindOfClass:CAMetalLayer.class])
        {
            mMetalLayer.retainAssign(static_cast<CAMetalLayer *>(mLayer));
        }
        else
        {
            mMetalLayer             = [[[CAMetalLayer alloc] init] ANGLE_MTL_AUTORELEASE];
            mMetalLayer.get().frame = mLayer.frame;
        }

        mMetalLayer.get().device          = metalDevice;
        mMetalLayer.get().pixelFormat     = mColorFormat.metalFormat;
        mMetalLayer.get().framebufferOnly = NO;  // Support blitting and glReadPixels

#if TARGET_OS_OSX || TARGET_OS_MACCATALYST
        // Autoresize with parent layer.
        mMetalLayer.get().autoresizingMask = kCALayerWidthSizable | kCALayerHeightSizable;
#endif

        if (mMetalLayer.get() != mLayer)
        {
            mMetalLayer.get().contentsScale = mLayer.contentsScale;

            [mLayer addSublayer:mMetalLayer.get()];
        }

        // ensure drawableSize is set to correct value:
        mMetalLayer.get().drawableSize = mCurrentKnownDrawableSize = calcExpectedDrawableSize();
    }

    return egl::NoError();
}

FramebufferImpl *WindowSurfaceMtl::createDefaultFramebuffer(const gl::Context *context,
                                                            const gl::FramebufferState &state)
{
    auto fbo = new FramebufferMtl(state, /* flipY */ true, /* backbuffer */ this);

    return fbo;
}

egl::Error WindowSurfaceMtl::swap(const gl::Context *context)
{
    ANGLE_TO_EGL_TRY(swapImpl(context));

    return egl::NoError();
}

void WindowSurfaceMtl::setSwapInterval(EGLint interval)
{
#if TARGET_OS_OSX || TARGET_OS_MACCATALYST
    mMetalLayer.get().displaySyncEnabled = interval != 0;
#endif
}

void WindowSurfaceMtl::setSwapBehavior(EGLint behavior)
{
    mRetainBuffer = behavior == EGL_BUFFER_PRESERVED;
}

// width and height can change with client window resizing
EGLint WindowSurfaceMtl::getWidth() const
{
    return static_cast<EGLint>(mCurrentKnownDrawableSize.width);
}

EGLint WindowSurfaceMtl::getHeight() const
{
    return static_cast<EGLint>(mCurrentKnownDrawableSize.height);
}

EGLint WindowSurfaceMtl::getSwapBehavior() const
{
    return mRetainBuffer ? EGL_BUFFER_PRESERVED : EGL_BUFFER_DESTROYED;
}

angle::Result WindowSurfaceMtl::getAttachmentRenderTarget(const gl::Context *context,
                                                          GLenum binding,
                                                          const gl::ImageIndex &imageIndex,
                                                          GLsizei samples,
                                                          FramebufferAttachmentRenderTarget **rtOut)
{
    ANGLE_TRY(ensureCurrentDrawableObtained(context));
    ANGLE_TRY(ensureCompanionTexturesSizeCorrect(context));

    return SurfaceMtl::getAttachmentRenderTarget(context, binding, imageIndex, samples, rtOut);
}

angle::Result WindowSurfaceMtl::ensureCurrentDrawableObtained(const gl::Context *context)
{
    if (!mCurrentDrawable)
    {
        ANGLE_TRY(obtainNextDrawable(context));
    }

    return angle::Result::Continue;
}

angle::Result WindowSurfaceMtl::ensureCompanionTexturesSizeCorrect(const gl::Context *context)
{
    ASSERT(mMetalLayer);

    gl::Extents size(static_cast<int>(mMetalLayer.get().drawableSize.width),
                     static_cast<int>(mMetalLayer.get().drawableSize.height), 1);

    ANGLE_TRY(SurfaceMtl::ensureCompanionTexturesSizeCorrect(context, size));

    if (mRetainBuffer && !mMSColorTexture &&
        (!mRetainedColorTexture || mRetainedColorTexture->size() != size))
    {
        // Create retained color texture (only if multisample texture is not used since multisample
        // texture can preserve the content also).
        ANGLE_TRY(CreateOrResizeTexture(context, mColorFormat, size.width, size.height, 1,
                                        /** renderTargetOnly */ true, &mRetainedColorTexture));

        // All drawing will be drawn to this texture instead of the main one.
        mColorRenderTarget.setTexture(mRetainedColorTexture);
    }

    return angle::Result::Continue;
}

CGSize WindowSurfaceMtl::calcExpectedDrawableSize() const
{
    CGSize currentLayerSize           = mMetalLayer.get().bounds.size;
    CGFloat currentLayerContentsScale = mMetalLayer.get().contentsScale;
    CGSize expectedDrawableSize = CGSizeMake(currentLayerSize.width * currentLayerContentsScale,
                                             currentLayerSize.height * currentLayerContentsScale);

    return expectedDrawableSize;
}

bool WindowSurfaceMtl::checkIfLayerResized(const gl::Context *context)
{
    if (mMetalLayer.get() != mLayer && mMetalLayer.get().contentsScale != mLayer.contentsScale)
    {
        // Parent layer's content scale has changed, update Metal layer's scale factor.
        mMetalLayer.get().contentsScale = mLayer.contentsScale;
    }

    CGSize currentLayerDrawableSize = mMetalLayer.get().drawableSize;
    CGSize expectedDrawableSize     = calcExpectedDrawableSize();

    if (currentLayerDrawableSize.width != expectedDrawableSize.width ||
        currentLayerDrawableSize.height != expectedDrawableSize.height ||
        mCurrentKnownDrawableSize.width != expectedDrawableSize.width ||
        mCurrentKnownDrawableSize.height != expectedDrawableSize.height)
    {
        // Resize the internal drawable texture.
        mMetalLayer.get().drawableSize = mCurrentKnownDrawableSize = expectedDrawableSize;

        return true;
    }

    return false;
}

angle::Result WindowSurfaceMtl::obtainNextDrawable(const gl::Context *context)
{
    ANGLE_MTL_OBJC_SCOPE
    {
        ContextMtl *contextMtl = mtl::GetImpl(context);

        ANGLE_MTL_TRY(contextMtl, mMetalLayer);

        mtl::TextureRef preservedColorTexture;
        mtl::TextureRef preservedDepthTexture;
        mtl::TextureRef preservedStencilTexture;

        // Check if layer was resized
        if (checkIfLayerResized(context))
        {
            contextMtl->onBackbufferResized(context, this);

            if (mRetainBuffer)
            {
                preservedColorTexture   = mMSColorTexture ? mMSColorTexture : mRetainedColorTexture;
                preservedDepthTexture   = mDepthTexture;
                preservedStencilTexture = mStencilTexture;
            }
        }

        mCurrentDrawable.retainAssign([mMetalLayer nextDrawable]);
        if (!mCurrentDrawable)
        {
            // The GPU might be taking too long finishing its rendering to the previous frame.
            // Try again, indefinitely wait until the previous frame render finishes.
            // TODO: this may wait forever here
            mMetalLayer.get().allowsNextDrawableTimeout = NO;
            mCurrentDrawable.retainAssign([mMetalLayer nextDrawable]);
            mMetalLayer.get().allowsNextDrawableTimeout = YES;
        }

        if (!mColorTexture)
        {
            mColorTexture = mtl::Texture::MakeFromMetal(mCurrentDrawable.get().texture);
            ASSERT(!mColorRenderTarget.getTexture());
            mColorRenderTarget.set(mColorTexture, mMSColorTexture, 0, 0, mColorFormat);
        }
        else
        {
            mColorTexture->set(mCurrentDrawable.get().texture);
        }

        ANGLE_MTL_LOG("Current metal drawable size=%d,%d", mColorTexture->width(),
                      mColorTexture->height());

        // Now we have to resize depth stencil buffers if required.
        ANGLE_TRY(ensureCompanionTexturesSizeCorrect(context));

        // Copy old content after resize.
        if (preservedColorTexture || preservedDepthTexture || preservedStencilTexture)
        {
            ANGLE_TRY(copyOldContents(context, preservedColorTexture, preservedDepthTexture,
                                      preservedStencilTexture));
        }

        return angle::Result::Continue;
    }
}

angle::Result WindowSurfaceMtl::copyOldContents(const gl::Context *context,
                                                const mtl::TextureRef &oldColorTexture,
                                                const mtl::TextureRef &oldDepthTexture,
                                                const mtl::TextureRef &oldStencilTexture)
{
    ContextMtl *contextMtl = mtl::GetImpl(context);

    if (oldColorTexture)
    {
        mtl::TextureRef readableColorTexture = oldColorTexture->getReadableCopy(
            contextMtl, contextMtl->getBlitCommandEncoder(), 0, 0,
            MTLRegionMake2D(0, 0, oldColorTexture->width(), oldColorTexture->height()));

        mtl::ColorBlitParams params;
        InitBlitParams(readableColorTexture, mColorRenderTarget.getTexture(), &params);
        params.enabledBuffers.set(0);

        mtl::RenderCommandEncoder *encoder =
            contextMtl->getRenderCommandEncoder(mColorRenderTarget);

        ANGLE_TRY(contextMtl->getDisplay()->getUtils().blitColorWithDraw(
            context, encoder, mColorFormat.actualAngleFormat(), params));

        contextMtl->endEncoding(true);
    }

    if (oldDepthTexture)
    {
        mtl::TextureRef readableDepthTexture = oldDepthTexture->getReadableCopy(
            contextMtl, contextMtl->getBlitCommandEncoder(), 0, 0,
            MTLRegionMake2D(0, 0, oldDepthTexture->width(), oldDepthTexture->height()));

        mtl::DepthStencilBlitParams params;
        InitBlitParams(readableDepthTexture, mDepthTexture, &params);

        mtl::RenderPassDesc rpDesc;
        mDepthRenderTarget.toRenderPassAttachmentDesc(&rpDesc.depthAttachment);
        rpDesc.sampleCount = mDepthTexture->samples();

        mtl::RenderCommandEncoder *encoder = contextMtl->getRenderCommandEncoder(rpDesc);

        ANGLE_TRY(contextMtl->getDisplay()->getUtils().blitDepthStencilWithDraw(context, encoder,
                                                                                params));

        contextMtl->endEncoding(true);
    }

    if (oldStencilTexture)
    {
        mtl::TextureRef readableStencilTexture = oldStencilTexture->getReadableCopy(
            contextMtl, contextMtl->getBlitCommandEncoder(), 0, 0,
            MTLRegionMake2D(0, 0, oldStencilTexture->width(), oldStencilTexture->height()));

        if (contextMtl->getDisplay()->getFeatures().hasStencilOutput.enabled)
        {
            mtl::DepthStencilBlitParams params;
            InitBlitParams(readableStencilTexture, mStencilTexture, &params);
            params.src        = nullptr;
            params.srcStencil = readableStencilTexture->getStencilView();

            mtl::RenderPassDesc rpDesc;
            mStencilRenderTarget.toRenderPassAttachmentDesc(&rpDesc.stencilAttachment);
            rpDesc.sampleCount = mStencilTexture->samples();

            mtl::RenderCommandEncoder *encoder = contextMtl->getRenderCommandEncoder(rpDesc);

            ANGLE_TRY(contextMtl->getDisplay()->getUtils().blitDepthStencilWithDraw(
                context, encoder, params));
        }
        else
        {
            mtl::StencilBlitViaBufferParams params;
            InitBlitParams(readableStencilTexture, mStencilTexture, &params);
            params.src                         = nullptr;
            params.srcStencil                  = readableStencilTexture->getStencilView();
            params.dstStencil                  = mStencilTexture;
            params.dstPackedDepthStencilFormat = mUsePackedDepthStencil;

            ANGLE_TRY(
                contextMtl->getDisplay()->getUtils().blitStencilViaCopyBuffer(context, params));
        }
        contextMtl->endEncoding(true);
    }

    return angle::Result::Continue;
}

angle::Result WindowSurfaceMtl::swapImpl(const gl::Context *context)
{
    if (mCurrentDrawable)
    {
        ASSERT(mColorTexture);

        ContextMtl *contextMtl = mtl::GetImpl(context);

        if (mMSColorTexture)
        {
            ANGLE_TRY(resolveColorTextureIfNeeded(context));
        }
        else if (mRetainedColorTexture)
        {
            CopyTextureNoScale(contextMtl, mRetainedColorTexture, &mColorTexture);
        }

        contextMtl->present(context, mCurrentDrawable);

        StopFrameCapture();
        StartFrameCapture(contextMtl);

        // Invalidate current drawable
        mColorTexture->set(nil);
        mCurrentDrawable = nil;
    }

    return angle::Result::Continue;
}

// OffscreenSurfaceMtl implementation
OffscreenSurfaceMtl::OffscreenSurfaceMtl(DisplayMtl *display,
                                         const egl::SurfaceState &state,
                                         const egl::AttributeMap &attribs)
    : SurfaceMtl(display, state, attribs)
{
    mSize = gl::Extents(attribs.getAsInt(EGL_WIDTH, 1), attribs.getAsInt(EGL_HEIGHT, 1), 1);
}

OffscreenSurfaceMtl::~OffscreenSurfaceMtl() {}

void OffscreenSurfaceMtl::destroy(const egl::Display *display)
{
    mAttachmentMSColorTextures.clear();
    SurfaceMtl::destroy(display);
}

egl::Error OffscreenSurfaceMtl::swap(const gl::Context *context)
{
    // Check for surface resize.
    ANGLE_TO_EGL_TRY(ensureTexturesSizeCorrect(context));

    return egl::NoError();
}

egl::Error OffscreenSurfaceMtl::bindTexImage(const gl::Context *context,
                                             gl::Texture *texture,
                                             EGLint buffer)
{
    ContextMtl *contextMtl = mtl::GetImpl(context);
    contextMtl->flushCommandBufer();

    // Initialize offscreen textures if needed:
    ANGLE_TO_EGL_TRY(ensureTexturesSizeCorrect(context));

    if (mMSColorTexture)
    {
        ANGLE_TO_EGL_TRY(resolveColorTextureIfNeeded(context));
    }

    return egl::NoError();
}

egl::Error OffscreenSurfaceMtl::releaseTexImage(const gl::Context *context, EGLint buffer)
{
    ContextMtl *contextMtl = mtl::GetImpl(context);
    // NOTE(hqle): Should we finishCommandBuffer or flush is enough?
    contextMtl->flushCommandBufer();
    return egl::NoError();
}

angle::Result OffscreenSurfaceMtl::getAttachmentRenderTarget(
    const gl::Context *context,
    GLenum binding,
    const gl::ImageIndex &imageIndex,
    GLsizei samples,
    FramebufferAttachmentRenderTarget **rtOut)
{
    // Initialize offscreen textures if needed:
    ANGLE_TRY(ensureTexturesSizeCorrect(context));

    return SurfaceMtl::getAttachmentRenderTarget(context, binding, imageIndex, samples, rtOut);
}

angle::Result OffscreenSurfaceMtl::getAttachmentMSColorTexture(const gl::Context *context,
                                                               GLsizei samples,
                                                               mtl::TextureRef *texOut)
{
    mtl::TextureRef &msTexture = mAttachmentMSColorTextures[samples];
    if (msTexture)
    {
        if (msTexture->size() == mColorTexture->size())
        {
            *texOut = msTexture;
            return angle::Result::Continue;
        }
    }

    ContextMtl *contextMtl = mtl::GetImpl(context);

    ANGLE_TRY(mtl::Texture::Make2DMSTexture(contextMtl, mColorFormat, mSize.width, mSize.height,
                                            samples,
                                            /* renderTargetOnly */ true,
                                            /* allowFormatView */ false, &msTexture));
    *texOut = msTexture;
    return angle::Result::Continue;
}

angle::Result OffscreenSurfaceMtl::ensureTexturesSizeCorrect(const gl::Context *context)
{
    if (!mColorTexture || mColorTexture->size() != mSize)
    {
        ANGLE_TRY(CreateOrResizeTexture(context, mColorFormat, mSize.width, mSize.height, 1,
                                        /** renderTargetOnly */ false, &mColorTexture));

        mColorRenderTarget.set(mColorTexture, 0, 0, mColorFormat);
    }

    return ensureCompanionTexturesSizeCorrect(context, mSize);
}

// PBufferSurfaceMtl implementation
PBufferSurfaceMtl::PBufferSurfaceMtl(DisplayMtl *display,
                                     const egl::SurfaceState &state,
                                     const egl::AttributeMap &attribs)
    : OffscreenSurfaceMtl(display, state, attribs)
{}

void PBufferSurfaceMtl::setFixedWidth(EGLint width)
{
    mSize.width = width;
}

void PBufferSurfaceMtl::setFixedHeight(EGLint height)
{
    mSize.height = height;
}
}
