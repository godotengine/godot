//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_OIIOIMAGELOADER_H
#define MATERIALX_OIIOIMAGELOADER_H

/// @file
/// Image loader wrapper using OpenImageIO

#include <MaterialXRender/ImageHandler.h>

MATERIALX_NAMESPACE_BEGIN

/// Shared pointer to an OiioImageLoader
using OiioImageLoaderPtr = std::shared_ptr<class OiioImageLoader>;

/// @class OiioImageLoader
/// OpenImageIO image file loader
class MX_RENDER_API OiioImageLoader : public ImageLoader
{
  public:
    OiioImageLoader()
    {
        // Set all extensions supported by OpenImageIO
        _extensions.insert(BMP_EXTENSION);
        _extensions.insert(GIF_EXTENSION);
        _extensions.insert(HDR_EXTENSION);
        _extensions.insert(JPG_EXTENSION);
        _extensions.insert(JPEG_EXTENSION);
        _extensions.insert(PIC_EXTENSION);
        _extensions.insert(PNG_EXTENSION);
        _extensions.insert(PSD_EXTENSION);
        _extensions.insert(TGA_EXTENSION);
        _extensions.insert(EXR_EXTENSION);
        _extensions.insert(TIF_EXTENSION);
        _extensions.insert(TIFF_EXTENSION);
        _extensions.insert(TX_EXTENSION);
        _extensions.insert(TXT_EXTENSION);
        _extensions.insert(TXR_EXTENSION);
    }
    virtual ~OiioImageLoader() { }

    /// Create a new OpenImageIO image loader
    static OiioImageLoaderPtr create() { return std::make_shared<OiioImageLoader>(); }

    /// Save an image to the file system.
    bool saveImage(const FilePath& filePath,
                   ConstImagePtr image,
                   bool verticalFlip = false) override;

    /// Load an image from the file system.
    ImagePtr loadImage(const FilePath& filePath) override;
};

MATERIALX_NAMESPACE_END

#endif
