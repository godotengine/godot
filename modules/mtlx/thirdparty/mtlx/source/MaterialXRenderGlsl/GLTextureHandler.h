//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_GLTEXTUREHANDLER_H
#define MATERIALX_GLTEXTUREHANDLER_H

/// @file
/// OpenGL texture handler

#include <MaterialXRenderGlsl/Export.h>

#include <MaterialXRender/ImageHandler.h>

MATERIALX_NAMESPACE_BEGIN

/// Shared pointer to an OpenGL texture handler
using GLTextureHandlerPtr = std::shared_ptr<class GLTextureHandler>;

/// @class GLTextureHandler
/// An OpenGL texture handler class
class MX_RENDERGLSL_API GLTextureHandler : public ImageHandler
{
  public:
    static ImageHandlerPtr create(ImageLoaderPtr imageLoader)
    {
        return ImageHandlerPtr(new GLTextureHandler(imageLoader));
    }

    /// Bind an image. This method will bind the texture to an active texture
    /// unit as defined by the corresponding image description. The method
    /// will fail if there are not enough available image units to bind to.
    bool bindImage(ImagePtr image, const ImageSamplingProperties& samplingProperties) override;

    /// Unbind an image.
    bool unbindImage(ImagePtr image) override;

    /// Create rendering resources for the given image.
    bool createRenderResources(ImagePtr image, bool generateMipMaps, bool useAsRenderTarget = false) override;

    /// Release rendering resources for the given image, or for all cached images
    /// if no image pointer is specified.
    void releaseRenderResources(ImagePtr image = nullptr) override;

    /// Return the bound texture location for a given resource
    int getBoundTextureLocation(unsigned int resourceId);

    /// Utility to map an address mode enumeration to an OpenGL address mode
    static int mapAddressModeToGL(ImageSamplingProperties::AddressMode addressModeEnum);

    /// Utility to map a filter type enumeration to an OpenGL filter type
    static int mapFilterTypeToGL(ImageSamplingProperties::FilterType filterTypeEnum, bool enableMipmaps);

    /// Utility to map generic texture properties to OpenGL texture formats.
    static void mapTextureFormatToGL(Image::BaseType baseType, unsigned int channelCount, bool srgb,
                                     int& glType, int& glFormat, int& glInternalFormat);

  protected:
    // Protected constructor
    GLTextureHandler(ImageLoaderPtr imageLoader);

    // Return the first free texture location that can be bound to.
    int getNextAvailableTextureLocation();

  protected:
    std::vector<unsigned int> _boundTextureLocations;
};

MATERIALX_NAMESPACE_END

#endif
