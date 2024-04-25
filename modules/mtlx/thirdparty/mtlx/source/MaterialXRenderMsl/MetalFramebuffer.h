//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_METALFRAMEBUFFER_H
#define MATERIALX_METALFRAMEBUFFER_H

/// @file
/// Metal framebuffer handling

#include <MaterialXRenderMsl/Export.h>

#include <MaterialXRender/ImageHandler.h>

#import <Metal/Metal.h>

MATERIALX_NAMESPACE_BEGIN

class MetalFramebuffer;

/// Shared pointer to a MetalFramebuffer
using MetalFramebufferPtr = std::shared_ptr<MetalFramebuffer>;

/// @class MetalFramebuffer
/// Wrapper for an Metal framebuffer
class MX_RENDERMSL_API MetalFramebuffer
{
  public:
    /// Create a new framebuffer
    static MetalFramebufferPtr create(id<MTLDevice> device,
                                      unsigned int width, unsigned int height,
                                      unsigned int channelCount,
                                      Image::BaseType baseType,
                                      id<MTLTexture> colorTexture = nil,
                                      bool encodeSrgb = false,
                                      MTLPixelFormat pixelFormat = MTLPixelFormatInvalid);

    /// Destructor
    virtual ~MetalFramebuffer();

    /// Resize the framebuffer
    void resize(unsigned int width, unsigned int height, bool forceRecreate = false,
                MTLPixelFormat pixelFormat = MTLPixelFormatInvalid,
                id<MTLTexture> extColorTexture = nil);

    /// Set the encode sRGB flag, which controls whether values written
    /// to the framebuffer are encoded to the sRGB color space.
    void setEncodeSrgb(bool encode)
    {
        if(encode != _encodeSrgb)
        {
            _encodeSrgb = encode;
            resize(_width, _height, true);
        }
    }

    /// Return the encode sRGB flag.
    bool getEncodeSrgb()
    {
        return _encodeSrgb;
    }
    
    /// Return the framebuffer width
    unsigned int getWidth() const { return _width; }

    /// Return the framebuffer height
    unsigned int getHeight() const { return _height; }
    
    /// Bind the framebuffer for rendering.
    void bind(MTLRenderPassDescriptor* renderpassDesc);

    /// Unbind the frame buffer after rendering.
    void unbind();

    /// Return our color texture handle.
    id<MTLTexture> getColorTexture() const
    {
        return _colorTexture;
    }
    
    void setColorTexture(id<MTLTexture> newColorTexture)
    {
        auto sameDim = [](id<MTLTexture> tex0, id<MTLTexture> tex1) -> bool
        {
            return  [tex0 width]  == [tex1 width] &&
                    [tex0 height] == [tex1 height];
        };
        if((!_colorTextureOwned || sameDim(_colorTexture, newColorTexture)) &&
           sameDim(newColorTexture, _depthTexture))
        {
            if(_colorTextureOwned)
                [_colorTexture release];
            _colorTexture = newColorTexture;
        }
    }

    /// Return our depth texture handle.
    id<MTLTexture> getDepthTexture() const
    {
        return _depthTexture;
    }

    /// Return the color data of this framebuffer as an image.
    /// If an input image is provided, it will be used to store the color data;
    /// otherwise a new image of the required format will be created.
    ImagePtr getColorImage(id<MTLCommandQueue> cmdQueue = nil, ImagePtr image = nullptr);

  protected:
    MetalFramebuffer(id<MTLDevice> device,
                     unsigned int width, unsigned int height,
                     unsigned int channelCount,
                     Image::BaseType baseType,
                     id<MTLTexture> colorTexture = nil,
                     bool encodeSrgb = false,
                     MTLPixelFormat pixelFormat = MTLPixelFormatInvalid);

  protected:
    unsigned int _width;
    unsigned int _height;
    unsigned int _channelCount;
    Image::BaseType _baseType;
    bool _encodeSrgb;

    id<MTLDevice>  _device = nil;
    id<MTLTexture> _colorTexture = nil;
    id<MTLTexture> _depthTexture = nil;
    
    bool _colorTextureOwned = false;
};

MATERIALX_NAMESPACE_END

#endif
