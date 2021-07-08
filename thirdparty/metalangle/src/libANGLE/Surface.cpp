//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Surface.cpp: Implements the egl::Surface class, representing a drawing surface
// such as the client area of a window, including any back buffers.
// Implements EGLSurface and related functionality. [EGL 1.4] section 2.2 page 3.

#include "libANGLE/Surface.h"

#include <EGL/eglext.h>

#include "libANGLE/Config.h"
#include "libANGLE/Context.h"
#include "libANGLE/Display.h"
#include "libANGLE/Framebuffer.h"
#include "libANGLE/Texture.h"
#include "libANGLE/Thread.h"
#include "libANGLE/formatutils.h"
#include "libANGLE/renderer/EGLImplFactory.h"
#include "libANGLE/trace.h"

namespace egl
{

SurfaceState::SurfaceState(const egl::Config *configIn, const AttributeMap &attributesIn)
    : label(nullptr),
      config((configIn != nullptr) ? new egl::Config(*configIn) : nullptr),
      attributes(attributesIn),
      timestampsEnabled(false)
{}

SurfaceState::~SurfaceState()
{
    delete config;
}

Surface::Surface(EGLint surfaceType,
                 const egl::Config *config,
                 const AttributeMap &attributes,
                 EGLenum buftype)
    : FramebufferAttachmentObject(),
      mState(config, attributes),
      mImplementation(nullptr),
      mRefCount(0),
      mDestroyed(false),
      mType(surfaceType),
      mBuftype(buftype),
      mPostSubBufferRequested(false),
      mLargestPbuffer(false),
      mGLColorspace(EGL_GL_COLORSPACE_LINEAR),
      mVGAlphaFormat(EGL_VG_ALPHA_FORMAT_NONPRE),
      mVGColorspace(EGL_VG_COLORSPACE_sRGB),
      mMipmapTexture(false),
      mMipmapLevel(0),
      mHorizontalResolution(EGL_UNKNOWN),
      mVerticalResolution(EGL_UNKNOWN),
      mMultisampleResolve(EGL_MULTISAMPLE_RESOLVE_DEFAULT),
      mFixedSize(false),
      mFixedWidth(0),
      mFixedHeight(0),
      mTextureFormat(TextureFormat::NoTexture),
      mTextureTarget(EGL_NO_TEXTURE),
      // FIXME: Determine actual pixel aspect ratio
      mPixelAspectRatio(static_cast<EGLint>(1.0 * EGL_DISPLAY_SCALING)),
      mRenderBuffer(EGL_BACK_BUFFER),
      mSwapBehavior(EGL_NONE),
      mOrientation(0),
      mTexture(nullptr),
      mColorFormat(config->renderTargetFormat),
      mDSFormat(config->depthStencilFormat),
      mInitState(gl::InitState::Initialized)
{
    mPostSubBufferRequested =
        (attributes.get(EGL_POST_SUB_BUFFER_SUPPORTED_NV, EGL_FALSE) == EGL_TRUE);
    mFlexibleSurfaceCompatibilityRequested =
        (attributes.get(EGL_FLEXIBLE_SURFACE_COMPATIBILITY_SUPPORTED_ANGLE, EGL_FALSE) == EGL_TRUE);

    if (mType == EGL_PBUFFER_BIT)
    {
        mLargestPbuffer = (attributes.get(EGL_LARGEST_PBUFFER, EGL_FALSE) == EGL_TRUE);
    }

    mGLColorspace =
        static_cast<EGLenum>(attributes.get(EGL_GL_COLORSPACE, EGL_GL_COLORSPACE_LINEAR));
    mVGAlphaFormat =
        static_cast<EGLenum>(attributes.get(EGL_VG_ALPHA_FORMAT, EGL_VG_ALPHA_FORMAT_NONPRE));
    mVGColorspace = static_cast<EGLenum>(attributes.get(EGL_VG_COLORSPACE, EGL_VG_COLORSPACE_sRGB));
    mMipmapTexture = (attributes.get(EGL_MIPMAP_TEXTURE, EGL_FALSE) == EGL_TRUE);

    mDirectComposition = (attributes.get(EGL_DIRECT_COMPOSITION_ANGLE, EGL_FALSE) == EGL_TRUE);

    mRobustResourceInitialization =
        (attributes.get(EGL_ROBUST_RESOURCE_INITIALIZATION_ANGLE, EGL_FALSE) == EGL_TRUE);
    if (mRobustResourceInitialization)
    {
        mInitState = gl::InitState::MayNeedInit;
    }

    mFixedSize = (attributes.get(EGL_FIXED_SIZE_ANGLE, EGL_FALSE) == EGL_TRUE);
    if (mFixedSize)
    {
        mFixedWidth  = static_cast<size_t>(attributes.get(EGL_WIDTH, 0));
        mFixedHeight = static_cast<size_t>(attributes.get(EGL_HEIGHT, 0));
    }

    if (mType != EGL_WINDOW_BIT)
    {
        mTextureFormat = attributes.getAsPackedEnum(EGL_TEXTURE_FORMAT, TextureFormat::NoTexture);
        mTextureTarget = static_cast<EGLenum>(attributes.get(EGL_TEXTURE_TARGET, EGL_NO_TEXTURE));
    }

    mOrientation = static_cast<EGLint>(attributes.get(EGL_SURFACE_ORIENTATION_ANGLE, 0));
}

Surface::~Surface() {}

rx::FramebufferAttachmentObjectImpl *Surface::getAttachmentImpl() const
{
    return mImplementation;
}

Error Surface::destroyImpl(const Display *display)
{
    if (mImplementation)
    {
        mImplementation->destroy(display);
    }

    ASSERT(!mTexture);

    SafeDelete(mImplementation);

    delete this;
    return NoError();
}

void Surface::postSwap(const gl::Context *context)
{
    if (mRobustResourceInitialization && mSwapBehavior != EGL_BUFFER_PRESERVED)
    {
        mInitState = gl::InitState::MayNeedInit;
        onStateChange(angle::SubjectMessage::SubjectChanged);
    }

    context->onPostSwap();
}

Error Surface::initialize(const Display *display)
{
    GLenum overrideRenderTargetFormat = mState.config->renderTargetFormat;

    // To account for color space differences, override the renderTargetFormat with the
    // non-linear format. If no suitable non-linear format was found, return
    // EGL_BAD_MATCH error
    if (!gl::ColorspaceFormatOverride(mGLColorspace, &overrideRenderTargetFormat))
    {
        return egl::EglBadMatch();
    }

    // If an override is required update mState.config as well
    if (mState.config->renderTargetFormat != overrideRenderTargetFormat)
    {
        egl::Config *overrideConfig        = new egl::Config(*(mState.config));
        overrideConfig->renderTargetFormat = overrideRenderTargetFormat;
        delete mState.config;
        mState.config = overrideConfig;

        mColorFormat = gl::Format(mState.config->renderTargetFormat);
        mDSFormat    = gl::Format(mState.config->depthStencilFormat);
    }

    ANGLE_TRY(mImplementation->initialize(display));

    // Initialized here since impl is nullptr in the constructor.
    // Must happen after implementation initialize for Android.
    mSwapBehavior = mImplementation->getSwapBehavior();

    if (mBuftype == EGL_IOSURFACE_ANGLE)
    {
        GLenum internalFormat =
            static_cast<GLenum>(mState.attributes.get(EGL_TEXTURE_INTERNAL_FORMAT_ANGLE));
        GLenum type  = static_cast<GLenum>(mState.attributes.get(EGL_TEXTURE_TYPE_ANGLE));
        mColorFormat = gl::Format(internalFormat, type);
    }
    if (mBuftype == EGL_D3D_TEXTURE_ANGLE)
    {
        const angle::Format *colorFormat = mImplementation->getD3DTextureColorFormat();
        ASSERT(colorFormat != nullptr);
        GLenum internalFormat = colorFormat->fboImplementationInternalFormat;
        mColorFormat          = gl::Format(internalFormat, colorFormat->componentType);
        mGLColorspace         = EGL_GL_COLORSPACE_LINEAR;
        if (mColorFormat.info->colorEncoding == GL_SRGB)
        {
            mGLColorspace = EGL_GL_COLORSPACE_SRGB;
        }
    }

    if (mType == EGL_WINDOW_BIT && display->getExtensions().getFrameTimestamps)
    {
        mState.supportedCompositorTimings = mImplementation->getSupportedCompositorTimings();
        mState.supportedTimestamps        = mImplementation->getSupportedTimestamps();
    }

    return NoError();
}

Error Surface::makeCurrent(const gl::Context *context)
{
    ANGLE_TRY(mImplementation->makeCurrent(context));

    mRefCount++;
    return NoError();
}

Error Surface::unMakeCurrent(const gl::Context *context)
{
    ANGLE_TRY(mImplementation->unMakeCurrent(context));
    return releaseRef(context->getDisplay());
}

Error Surface::releaseRef(const Display *display)
{
    ASSERT(mRefCount > 0);
    mRefCount--;
    if (mRefCount == 0 && mDestroyed)
    {
        ASSERT(display);
        return destroyImpl(display);
    }

    return NoError();
}

Error Surface::onDestroy(const Display *display)
{
    mDestroyed = true;
    if (mRefCount == 0)
    {
        return destroyImpl(display);
    }
    return NoError();
}

void Surface::setLabel(EGLLabelKHR label)
{
    mState.label = label;
}

EGLLabelKHR Surface::getLabel() const
{
    return mState.label;
}

EGLint Surface::getType() const
{
    return mType;
}

Error Surface::swap(const gl::Context *context)
{
    ANGLE_TRACE_EVENT0("gpu.angle", "egl::Surface::swap");

    context->getState().getOverlay()->onSwap();

    ANGLE_TRY(mImplementation->swap(context));
    postSwap(context);
    return NoError();
}

Error Surface::swapWithDamage(const gl::Context *context, EGLint *rects, EGLint n_rects)
{
    context->getState().getOverlay()->onSwap();

    ANGLE_TRY(mImplementation->swapWithDamage(context, rects, n_rects));
    postSwap(context);
    return NoError();
}

Error Surface::postSubBuffer(const gl::Context *context,
                             EGLint x,
                             EGLint y,
                             EGLint width,
                             EGLint height)
{
    if (width == 0 || height == 0)
    {
        return egl::NoError();
    }

    context->getState().getOverlay()->onSwap();

    ANGLE_TRY(mImplementation->postSubBuffer(context, x, y, width, height));
    postSwap(context);
    return NoError();
}

Error Surface::setPresentationTime(EGLnsecsANDROID time)
{
    return mImplementation->setPresentationTime(time);
}

Error Surface::querySurfacePointerANGLE(EGLint attribute, void **value)
{
    return mImplementation->querySurfacePointerANGLE(attribute, value);
}

EGLint Surface::isPostSubBufferSupported() const
{
    return mPostSubBufferRequested && mImplementation->isPostSubBufferSupported();
}

void Surface::setSwapInterval(EGLint interval)
{
    mImplementation->setSwapInterval(interval);
}

void Surface::setMipmapLevel(EGLint level)
{
    // Level is set but ignored
    UNIMPLEMENTED();
    mMipmapLevel = level;
}

void Surface::setMultisampleResolve(EGLenum resolve)
{
    // Behaviour is set but ignored
    UNIMPLEMENTED();
    mMultisampleResolve = resolve;
}

void Surface::setSwapBehavior(EGLenum behavior)
{
    mSwapBehavior = behavior;
    mImplementation->setSwapBehavior(behavior);
}

void Surface::setFixedWidth(EGLint width)
{
    mFixedWidth = width;
    mImplementation->setFixedWidth(width);
}

void Surface::setFixedHeight(EGLint height)
{
    mFixedHeight = height;
    mImplementation->setFixedHeight(height);
}

const Config *Surface::getConfig() const
{
    return mState.config;
}

EGLint Surface::getPixelAspectRatio() const
{
    return mPixelAspectRatio;
}

EGLenum Surface::getRenderBuffer() const
{
    return mRenderBuffer;
}

EGLenum Surface::getSwapBehavior() const
{
    return mSwapBehavior;
}

TextureFormat Surface::getTextureFormat() const
{
    return mTextureFormat;
}

EGLenum Surface::getTextureTarget() const
{
    return mTextureTarget;
}

bool Surface::getLargestPbuffer() const
{
    return mLargestPbuffer;
}

EGLenum Surface::getGLColorspace() const
{
    return mGLColorspace;
}

EGLenum Surface::getVGAlphaFormat() const
{
    return mVGAlphaFormat;
}

EGLenum Surface::getVGColorspace() const
{
    return mVGColorspace;
}

bool Surface::getMipmapTexture() const
{
    return mMipmapTexture;
}

EGLint Surface::getMipmapLevel() const
{
    return mMipmapLevel;
}

EGLint Surface::getHorizontalResolution() const
{
    return mHorizontalResolution;
}

EGLint Surface::getVerticalResolution() const
{
    return mVerticalResolution;
}

EGLenum Surface::getMultisampleResolve() const
{
    return mMultisampleResolve;
}

EGLint Surface::isFixedSize() const
{
    return mFixedSize;
}

EGLint Surface::getWidth() const
{
    return mFixedSize ? static_cast<EGLint>(mFixedWidth) : mImplementation->getWidth();
}

EGLint Surface::getHeight() const
{
    return mFixedSize ? static_cast<EGLint>(mFixedHeight) : mImplementation->getHeight();
}

Error Surface::bindTexImage(gl::Context *context, gl::Texture *texture, EGLint buffer)
{
    ASSERT(!mTexture);
    ANGLE_TRY(mImplementation->bindTexImage(context, texture, buffer));

    if (texture->bindTexImageFromSurface(context, this) == angle::Result::Stop)
    {
        return Error(EGL_BAD_SURFACE);
    }
    mTexture = texture;
    mRefCount++;

    return NoError();
}

Error Surface::releaseTexImage(const gl::Context *context, EGLint buffer)
{
    ASSERT(context);

    ANGLE_TRY(mImplementation->releaseTexImage(context, buffer));

    ASSERT(mTexture);
    ANGLE_TRY(ResultToEGL(mTexture->releaseTexImageFromSurface(context)));

    return releaseTexImageFromTexture(context);
}

Error Surface::getSyncValues(EGLuint64KHR *ust, EGLuint64KHR *msc, EGLuint64KHR *sbc)
{
    return mImplementation->getSyncValues(ust, msc, sbc);
}

Error Surface::releaseTexImageFromTexture(const gl::Context *context)
{
    ASSERT(mTexture);
    mTexture = nullptr;
    return releaseRef(context->getDisplay());
}

gl::Extents Surface::getAttachmentSize(const gl::ImageIndex & /*target*/) const
{
    return gl::Extents(getWidth(), getHeight(), 1);
}

gl::Format Surface::getAttachmentFormat(GLenum binding, const gl::ImageIndex &target) const
{
    return (binding == GL_BACK ? mColorFormat : mDSFormat);
}

GLsizei Surface::getAttachmentSamples(const gl::ImageIndex &target) const
{
    return getConfig()->samples;
}

bool Surface::isRenderable(const gl::Context *context,
                           GLenum binding,
                           const gl::ImageIndex &imageIndex) const
{
    return true;
}

GLuint Surface::getId() const
{
    UNREACHABLE();
    return 0;
}

gl::Framebuffer *Surface::createDefaultFramebuffer(const gl::Context *context,
                                                   egl::Surface *readSurface)
{
    return new gl::Framebuffer(context, this, readSurface);
}

gl::InitState Surface::initState(const gl::ImageIndex & /*imageIndex*/) const
{
    return mInitState;
}

void Surface::setInitState(const gl::ImageIndex & /*imageIndex*/, gl::InitState initState)
{
    mInitState = initState;
}

void Surface::setTimestampsEnabled(bool enabled)
{
    mImplementation->setTimestampsEnabled(enabled);
    mState.timestampsEnabled = enabled;
}

bool Surface::isTimestampsEnabled() const
{
    return mState.timestampsEnabled;
}

const SupportedCompositorTiming &Surface::getSupportedCompositorTimings() const
{
    return mState.supportedCompositorTimings;
}

Error Surface::getCompositorTiming(EGLint numTimestamps,
                                   const EGLint *names,
                                   EGLnsecsANDROID *values) const
{
    return mImplementation->getCompositorTiming(numTimestamps, names, values);
}

Error Surface::getNextFrameId(EGLuint64KHR *frameId) const
{
    return mImplementation->getNextFrameId(frameId);
}

const SupportedTimestamps &Surface::getSupportedTimestamps() const
{
    return mState.supportedTimestamps;
}

Error Surface::getFrameTimestamps(EGLuint64KHR frameId,
                                  EGLint numTimestamps,
                                  const EGLint *timestamps,
                                  EGLnsecsANDROID *values) const
{
    return mImplementation->getFrameTimestamps(frameId, numTimestamps, timestamps, values);
}

WindowSurface::WindowSurface(rx::EGLImplFactory *implFactory,
                             const egl::Config *config,
                             EGLNativeWindowType window,
                             const AttributeMap &attribs)
    : Surface(EGL_WINDOW_BIT, config, attribs)
{
    mImplementation = implFactory->createWindowSurface(mState, window, attribs);
}

WindowSurface::~WindowSurface() {}

PbufferSurface::PbufferSurface(rx::EGLImplFactory *implFactory,
                               const Config *config,
                               const AttributeMap &attribs)
    : Surface(EGL_PBUFFER_BIT, config, attribs)
{
    mImplementation = implFactory->createPbufferSurface(mState, attribs);
}

PbufferSurface::PbufferSurface(rx::EGLImplFactory *implFactory,
                               const Config *config,
                               EGLenum buftype,
                               EGLClientBuffer clientBuffer,
                               const AttributeMap &attribs)
    : Surface(EGL_PBUFFER_BIT, config, attribs, buftype)
{
    mImplementation =
        implFactory->createPbufferFromClientBuffer(mState, buftype, clientBuffer, attribs);
}

PbufferSurface::~PbufferSurface() {}

PixmapSurface::PixmapSurface(rx::EGLImplFactory *implFactory,
                             const Config *config,
                             NativePixmapType nativePixmap,
                             const AttributeMap &attribs)
    : Surface(EGL_PIXMAP_BIT, config, attribs)
{
    mImplementation = implFactory->createPixmapSurface(mState, nativePixmap, attribs);
}

PixmapSurface::~PixmapSurface() {}

// SurfaceDeleter implementation.

SurfaceDeleter::SurfaceDeleter(const Display *display) : mDisplay(display) {}

SurfaceDeleter::~SurfaceDeleter() {}

void SurfaceDeleter::operator()(Surface *surface)
{
    ANGLE_SWALLOW_ERR(surface->onDestroy(mDisplay));
}

}  // namespace egl
