//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// PBufferSurfaceCGL.cpp: an implementation of PBuffers created from IOSurfaces using
//                        EGL_ANGLE_iosurface_client_buffer

#if __has_include(<Cocoa/Cocoa.h>)

#    include "libANGLE/renderer/gl/cgl/IOSurfaceSurfaceCGL.h"

#    import <Cocoa/Cocoa.h>
#    include <IOSurface/IOSurface.h>
#    include <OpenGL/CGLIOSurface.h>

#    include "common/debug.h"
#    include "libANGLE/AttributeMap.h"
#    include "libANGLE/renderer/gl/BlitGL.h"
#    include "libANGLE/renderer/gl/FramebufferGL.h"
#    include "libANGLE/renderer/gl/FunctionsGL.h"
#    include "libANGLE/renderer/gl/RendererGL.h"
#    include "libANGLE/renderer/gl/StateManagerGL.h"
#    include "libANGLE/renderer/gl/TextureGL.h"
#    include "libANGLE/renderer/gl/cgl/DisplayCGL.h"

namespace rx
{

namespace
{

struct IOSurfaceFormatInfo
{
    GLenum internalFormat;
    GLenum type;

    size_t componentBytes;

    GLenum nativeInternalFormat;
    GLenum nativeFormat;
    GLenum nativeType;
};

// clang-format off
static const IOSurfaceFormatInfo kIOSurfaceFormats[] = {
    {GL_RED,      GL_UNSIGNED_BYTE,  1, GL_RED,  GL_RED,  GL_UNSIGNED_BYTE           },
    {GL_R16UI,    GL_UNSIGNED_SHORT, 2, GL_RED,  GL_RED,  GL_UNSIGNED_SHORT          },
    {GL_RG,       GL_UNSIGNED_BYTE,  2, GL_RG,   GL_RG,   GL_UNSIGNED_BYTE           },
    {GL_RGB,      GL_UNSIGNED_BYTE,  4, GL_BGRA, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV},
    {GL_BGRA_EXT, GL_UNSIGNED_BYTE,  4, GL_BGRA, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV},
    {GL_RGBA,     GL_HALF_FLOAT,     8, GL_RGBA, GL_RGBA, GL_HALF_FLOAT              },
};
// clang-format on

int FindIOSurfaceFormatIndex(GLenum internalFormat, GLenum type)
{
    for (int i = 0; i < static_cast<int>(ArraySize(kIOSurfaceFormats)); ++i)
    {
        const auto &formatInfo = kIOSurfaceFormats[i];
        if (formatInfo.internalFormat == internalFormat && formatInfo.type == type)
        {
            return i;
        }
    }
    return -1;
}

}  // anonymous namespace

IOSurfaceSurfaceCGL::IOSurfaceSurfaceCGL(const egl::SurfaceState &state,
                                         CGLContextObj cglContext,
                                         EGLClientBuffer buffer,
                                         const egl::AttributeMap &attribs)
    : SurfaceGL(state),
      mCGLContext(cglContext),
      mIOSurface(nullptr),
      mWidth(0),
      mHeight(0),
      mPlane(0),
      mFormatIndex(-1),
      mAlphaInitialized(false)
{
    // Keep reference to the IOSurface so it doesn't get deleted while the pbuffer exists.
    mIOSurface = reinterpret_cast<IOSurfaceRef>(buffer);
    CFRetain(mIOSurface);

    // Extract attribs useful for the call to CGLTexImageIOSurface2D
    mWidth  = static_cast<int>(attribs.get(EGL_WIDTH));
    mHeight = static_cast<int>(attribs.get(EGL_HEIGHT));
    mPlane  = static_cast<int>(attribs.get(EGL_IOSURFACE_PLANE_ANGLE));

    EGLAttrib internalFormat = attribs.get(EGL_TEXTURE_INTERNAL_FORMAT_ANGLE);
    EGLAttrib type           = attribs.get(EGL_TEXTURE_TYPE_ANGLE);
    mFormatIndex =
        FindIOSurfaceFormatIndex(static_cast<GLenum>(internalFormat), static_cast<GLenum>(type));
    ASSERT(mFormatIndex >= 0);

    mAlphaInitialized = !hasEmulatedAlphaChannel();
}

IOSurfaceSurfaceCGL::~IOSurfaceSurfaceCGL()
{
    if (mIOSurface != nullptr)
    {
        CFRelease(mIOSurface);
        mIOSurface = nullptr;
    }
}

egl::Error IOSurfaceSurfaceCGL::initialize(const egl::Display *display)
{
    return egl::NoError();
}

egl::Error IOSurfaceSurfaceCGL::makeCurrent(const gl::Context *context)
{
    return egl::NoError();
}

egl::Error IOSurfaceSurfaceCGL::unMakeCurrent(const gl::Context *context)
{
    GetFunctionsGL(context)->flush();
    return egl::NoError();
}

egl::Error IOSurfaceSurfaceCGL::swap(const gl::Context *context)
{
    return egl::NoError();
}

egl::Error IOSurfaceSurfaceCGL::postSubBuffer(const gl::Context *context,
                                              EGLint x,
                                              EGLint y,
                                              EGLint width,
                                              EGLint height)
{
    UNREACHABLE();
    return egl::NoError();
}

egl::Error IOSurfaceSurfaceCGL::querySurfacePointerANGLE(EGLint attribute, void **value)
{
    UNREACHABLE();
    return egl::NoError();
}

egl::Error IOSurfaceSurfaceCGL::bindTexImage(const gl::Context *context,
                                             gl::Texture *texture,
                                             EGLint buffer)
{
    StateManagerGL *stateManager = GetStateManagerGL(context);

    const TextureGL *textureGL = GetImplAs<TextureGL>(texture);
    GLuint textureID           = textureGL->getTextureID();
    stateManager->bindTexture(gl::TextureType::Rectangle, textureID);

    const auto &format = kIOSurfaceFormats[mFormatIndex];
    CGLError error = CGLTexImageIOSurface2D(mCGLContext, GL_TEXTURE_RECTANGLE, format.nativeFormat,
                                            mWidth, mHeight, format.nativeInternalFormat,
                                            format.nativeType, mIOSurface, mPlane);

    if (error != kCGLNoError)
    {
        return egl::EglContextLost() << "CGLTexImageIOSurface2D failed: " << CGLErrorString(error);
    }

    if (IsError(initializeAlphaChannel(context, textureID)))
    {
        return egl::EglContextLost() << "Failed to initialize IOSurface alpha channel.";
    }

    return egl::NoError();
}

egl::Error IOSurfaceSurfaceCGL::releaseTexImage(const gl::Context *context, EGLint buffer)
{
    const FunctionsGL *functions = GetFunctionsGL(context);
    functions->flush();
    return egl::NoError();
}

void IOSurfaceSurfaceCGL::setSwapInterval(EGLint interval)
{
    UNREACHABLE();
}

EGLint IOSurfaceSurfaceCGL::getWidth() const
{
    return mWidth;
}

EGLint IOSurfaceSurfaceCGL::getHeight() const
{
    return mHeight;
}

EGLint IOSurfaceSurfaceCGL::isPostSubBufferSupported() const
{
    UNREACHABLE();
    return EGL_FALSE;
}

EGLint IOSurfaceSurfaceCGL::getSwapBehavior() const
{
    // N/A because you can't MakeCurrent an IOSurface, return any valid value.
    return EGL_BUFFER_PRESERVED;
}

// static
bool IOSurfaceSurfaceCGL::validateAttributes(EGLClientBuffer buffer,
                                             const egl::AttributeMap &attribs)
{
    IOSurfaceRef ioSurface = reinterpret_cast<IOSurfaceRef>(buffer);

    // The plane must exist for this IOSurface. IOSurfaceGetPlaneCount can return 0 for non-planar
    // ioSurfaces but we will treat non-planar like it is a single plane.
    size_t surfacePlaneCount = std::max(size_t(1), IOSurfaceGetPlaneCount(ioSurface));
    EGLAttrib plane          = attribs.get(EGL_IOSURFACE_PLANE_ANGLE);
    if (plane < 0 || static_cast<size_t>(plane) >= surfacePlaneCount)
    {
        return false;
    }

    // The width height specified must be at least (1, 1) and at most the plane size
    EGLAttrib width  = attribs.get(EGL_WIDTH);
    EGLAttrib height = attribs.get(EGL_HEIGHT);
    if (width <= 0 || static_cast<size_t>(width) > IOSurfaceGetWidthOfPlane(ioSurface, plane) ||
        height <= 0 || static_cast<size_t>(height) > IOSurfaceGetHeightOfPlane(ioSurface, plane))
    {
        return false;
    }

    // Find this IOSurface format
    EGLAttrib internalFormat = attribs.get(EGL_TEXTURE_INTERNAL_FORMAT_ANGLE);
    EGLAttrib type           = attribs.get(EGL_TEXTURE_TYPE_ANGLE);

    int formatIndex =
        FindIOSurfaceFormatIndex(static_cast<GLenum>(internalFormat), static_cast<GLenum>(type));

    if (formatIndex < 0)
    {
        return false;
    }

    // Check that the format matches this IOSurface plane
    if (IOSurfaceGetBytesPerElementOfPlane(ioSurface, plane) !=
        kIOSurfaceFormats[formatIndex].componentBytes)
    {
        return false;
    }

    return true;
}

// Wraps a FramebufferGL to hook the destroy function to delete the texture associated with the
// framebuffer.
class IOSurfaceFramebuffer : public FramebufferGL
{
  public:
    IOSurfaceFramebuffer(const gl::FramebufferState &data,
                         GLuint id,
                         GLuint textureId,
                         bool isDefault,
                         bool emulatedAlpha)
        : FramebufferGL(data, id, isDefault, emulatedAlpha), mTextureId(textureId)
    {}
    void destroy(const gl::Context *context) override
    {
        GetFunctionsGL(context)->deleteTextures(1, &mTextureId);
        FramebufferGL::destroy(context);
    }

  private:
    GLuint mTextureId;
};

FramebufferImpl *IOSurfaceSurfaceCGL::createDefaultFramebuffer(const gl::Context *context,
                                                               const gl::FramebufferState &state)
{
    const FunctionsGL *functions = GetFunctionsGL(context);
    StateManagerGL *stateManager = GetStateManagerGL(context);

    GLuint texture = 0;
    functions->genTextures(1, &texture);
    const auto &format = kIOSurfaceFormats[mFormatIndex];
    stateManager->bindTexture(gl::TextureType::Rectangle, texture);
    CGLError error = CGLTexImageIOSurface2D(mCGLContext, GL_TEXTURE_RECTANGLE, format.nativeFormat,
                                            mWidth, mHeight, format.nativeInternalFormat,
                                            format.nativeType, mIOSurface, mPlane);
    if (error != kCGLNoError)
    {
        ERR() << "CGLTexImageIOSurface2D failed: " << CGLErrorString(error);
    }
    ASSERT(error == kCGLNoError);

    if (IsError(initializeAlphaChannel(context, texture)))
    {
        ERR() << "Failed to initialize IOSurface alpha channel.";
    }

    GLuint framebuffer = 0;
    functions->genFramebuffers(1, &framebuffer);
    stateManager->bindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    stateManager->bindTexture(gl::TextureType::Rectangle, texture);
    functions->framebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE,
                                    texture, 0);

    return new IOSurfaceFramebuffer(state, framebuffer, texture, true, hasEmulatedAlphaChannel());
}

angle::Result IOSurfaceSurfaceCGL::initializeAlphaChannel(const gl::Context *context,
                                                          GLuint texture)
{
    if (mAlphaInitialized)
    {
        return angle::Result::Continue;
    }

    BlitGL *blitter = GetBlitGL(context);
    ANGLE_TRY(blitter->clearRenderableTextureAlphaToOne(context, texture,
                                                        gl::TextureTarget::Rectangle, 0));
    mAlphaInitialized = true;
    return angle::Result::Continue;
}

bool IOSurfaceSurfaceCGL::hasEmulatedAlphaChannel() const
{
    const auto &format = kIOSurfaceFormats[mFormatIndex];
    return format.internalFormat == GL_RGB;
}

}  // namespace rx

#endif  // __has_include(<Cocoa/Cocoa.h>)
