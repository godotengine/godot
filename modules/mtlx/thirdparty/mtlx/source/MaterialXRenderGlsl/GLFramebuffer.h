//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_GLFRAMEBUFFER_H
#define MATERIALX_GLFRAMEBUFFER_H

/// @file
/// OpenGL framebuffer handling

#include <MaterialXRenderGlsl/Export.h>

#include <MaterialXRender/ImageHandler.h>

MATERIALX_NAMESPACE_BEGIN

class GLFramebuffer;

/// Shared pointer to a GLFramebuffer
using GLFramebufferPtr = std::shared_ptr<GLFramebuffer>;

/// @class GLFramebuffer
/// Wrapper for an OpenGL framebuffer
class MX_RENDERGLSL_API GLFramebuffer
{
  public:
    /// Create a new framebuffer
    static GLFramebufferPtr create(unsigned int width, unsigned int height, unsigned int channelCount, Image::BaseType baseType);

    /// Destructor
    virtual ~GLFramebuffer();

    /// Return the width of the framebuffer.
    unsigned int getWidth() const
    {
        return _width;
    }

    /// Return the height of the framebuffer.
    unsigned int getHeight() const
    {
        return _height;
    }

    /// Set the encode sRGB flag, which controls whether values written
    /// to the framebuffer are encoded to the sRGB color space.
    void setEncodeSrgb(bool encode)
    {
        _encodeSrgb = encode;
    }

    /// Return the encode sRGB flag.
    bool getEncodeSrgb()
    {
        return _encodeSrgb;
    }

    /// Bind the framebuffer for rendering.
    void bind();

    /// Unbind the frame buffer after rendering.
    void unbind();

    /// Return our color texture handle.
    unsigned int getColorTexture() const
    {
        return _colorTexture;
    }

    /// Return our depth texture handle.
    unsigned int getDepthTexture() const
    {
        return _depthTexture;
    }

    /// Return the color data of this framebuffer as an image.
    /// If an input image is provided, it will be used to store the color data;
    /// otherwise a new image of the required format will be created.
    ImagePtr getColorImage(ImagePtr image = nullptr);

    /// Blit our color texture to the back buffer.
    void blit();

  protected:
    GLFramebuffer(unsigned int width, unsigned int height, unsigned int channelCount, Image::BaseType baseType);

  protected:
    unsigned int _width;
    unsigned int _height;
    unsigned int _channelCount;
    Image::BaseType _baseType;
    bool _encodeSrgb;

    unsigned int _framebuffer;
    unsigned int _colorTexture;
    unsigned int _depthTexture;
};

MATERIALX_NAMESPACE_END

#endif
