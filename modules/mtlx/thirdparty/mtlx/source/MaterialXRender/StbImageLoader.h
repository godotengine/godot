//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_STBIMAGELOADER_H
#define MATERIALX_STBIMAGELOADER_H

/// @file
/// Image loader using the stb image library

#include <MaterialXRender/ImageHandler.h>

MATERIALX_NAMESPACE_BEGIN

/// Shared pointer to an StbImageLoader
using StbImageLoaderPtr = std::shared_ptr<class StbImageLoader>;

/// @class StbImageLoader
/// Stb image file loader
class MX_RENDER_API StbImageLoader : public ImageLoader
{
  public:
    StbImageLoader()
    {
        // Set all extensions supported by stb image
        _extensions.insert(BMP_EXTENSION);
        _extensions.insert(GIF_EXTENSION);
        _extensions.insert(HDR_EXTENSION);
        _extensions.insert(JPG_EXTENSION);
        _extensions.insert(JPEG_EXTENSION);
        _extensions.insert(PIC_EXTENSION);
        _extensions.insert(PNG_EXTENSION);
        _extensions.insert(PSD_EXTENSION);
        _extensions.insert(TGA_EXTENSION);
    }
    virtual ~StbImageLoader() { }

    /// Create a new stb image loader
    static StbImageLoaderPtr create() { return std::make_shared<StbImageLoader>(); }

    /// Save an image to the file system.
    bool saveImage(const FilePath& filePath,
                   ConstImagePtr image,
                   bool verticalFlip = false) override;

    /// Load an image from the file system.
    ImagePtr loadImage(const FilePath& filePath) override;
};

MATERIALX_NAMESPACE_END

#endif
