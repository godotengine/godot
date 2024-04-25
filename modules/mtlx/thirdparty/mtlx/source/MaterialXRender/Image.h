//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_IMAGE_H
#define MATERIALX_IMAGE_H

/// @file
/// Image class

#include <MaterialXRender/Export.h>

#include <MaterialXFormat/File.h>

#include <MaterialXCore/Types.h>

MATERIALX_NAMESPACE_BEGIN

class Image;

/// A shared pointer to an image
using ImagePtr = shared_ptr<Image>;

/// A shared pointer to a const image
using ConstImagePtr = shared_ptr<const Image>;

/// A map from strings to images.
using ImageMap = std::unordered_map<string, ImagePtr>;

/// A vetor of images.
using ImageVec = std::vector<ImagePtr>;

/// A pair of images.
using ImagePair = std::pair<ImagePtr, ImagePtr>;

/// A function to perform image buffer deallocation
using ImageBufferDeallocator = std::function<void(void*)>;

/// A pair of unsigned integers.
using UnsignedIntPair = std::pair<unsigned int, unsigned int>;

/// @class Image
/// Class representing an image in system memory
class MX_RENDER_API Image
{
  public:
    enum class BaseType
    {
        UINT8,
        INT8,
        UINT16,
        INT16,
        HALF,
        FLOAT
    };

  public:
    /// Create an empty image with the given properties.
    static ImagePtr create(unsigned int width, unsigned int height, unsigned int channelCount, BaseType baseType = BaseType::UINT8)
    {
        return ImagePtr(new Image(width, height, channelCount, baseType));
    }

    ~Image();

    /// @name Property Accessors
    /// @{

    /// Return the width of the image.
    unsigned int getWidth() const
    {
        return _width;
    }

    /// Return the height of the image.
    unsigned int getHeight() const
    {
        return _height;
    }

    /// Return the channel count of the image.
    unsigned int getChannelCount() const
    {
        return _channelCount;
    }

    /// Return the base type of the image.
    BaseType getBaseType() const
    {
        return _baseType;
    }

    /// Return the stride of our base type in bytes.
    unsigned int getBaseStride() const;

    /// Return the stride of an image row in bytes.
    unsigned int getRowStride() const
    {
        return _width * _channelCount * getBaseStride();
    }

    /// Return the maximum number of mipmaps for this image.
    unsigned int getMaxMipCount() const;

    /// @}
    /// @name Texel Accessors
    /// @{

    /// Set the texel color at the given coordinates.  If the coordinates
    /// or image resource buffer are invalid, then an exception is thrown.
    void setTexelColor(unsigned int x, unsigned int y, const Color4& color);

    /// Return the texel color at the given coordinates.  If the coordinates
    /// or image resource buffer are invalid, then an exception is thrown.
    Color4 getTexelColor(unsigned int x, unsigned int y) const;

    /// @}
    /// @name Image Analysis
    /// @{

    /// Compute the average color of the image.
    Color4 getAverageColor();

    /// Return true if all texels of this image are identical in color.
    /// @param uniformColor Return the uniform color of the image, if any.
    bool isUniformColor(Color4* uniformColor = nullptr);

    /// @}
    /// @name Image Processing
    /// @{

    /// Set all texels of this image to a uniform color.
    void setUniformColor(const Color4& color);

    /// Apply the given matrix transform to all texels of this image.
    void applyMatrixTransform(const Matrix33& mat);

    /// Apply the given gamma transform to all texels of this image.
    void applyGammaTransform(float gamma);

    /// Create a copy of this image with the given channel count and base type.
    ImagePtr copy(unsigned int channelCount, BaseType baseType) const;

    /// Apply a 3x3 box blur to this image, returning a new blurred image.
    ImagePtr applyBoxBlur();

    /// Apply a 7x7 Gaussian blur to this image, returning a new blurred image.
    ImagePtr applyGaussianBlur();

    /// Split this image by the given luminance threshold, returning the
    /// resulting underflow and overflow images.
    ImagePair splitByLuminance(float luminance);

    /// Save a channel of this image to disk as a text table, in a format
    /// that can be used for curve and surface fitting.
    void writeTable(const FilePath& filePath, unsigned int channel);

    /// @}
    /// @name Resource Buffers
    /// @{

    /// Set the resource buffer for this image.
    void setResourceBuffer(void* buffer)
    {
        _resourceBuffer = buffer;
    }

    /// Return the resource buffer for this image.
    void* getResourceBuffer() const
    {
        return _resourceBuffer;
    }

    /// Allocate a resource buffer for this image that matches its properties.
    void createResourceBuffer();

    /// Release the resource buffer for this image.
    void releaseResourceBuffer();

    /// Set the resource buffer deallocator for this image.
    void setResourceBufferDeallocator(ImageBufferDeallocator deallocator)
    {
        _resourceBufferDeallocator = deallocator;
    }

    /// Return the resource buffer deallocator for this image.
    ImageBufferDeallocator getResourceBufferDeallocator() const
    {
        return _resourceBufferDeallocator;
    }

    /// @}
    /// @name Resource IDs
    /// @{

    /// Set the resource ID for this image.
    void setResourceId(unsigned int id)
    {
        _resourceId = id;
    }

    /// Return the resource ID for this image.
    unsigned int getResourceId() const
    {
        return _resourceId;
    }

    /// @}

  protected:
    Image(unsigned int width, unsigned int height, unsigned int channelCount, BaseType baseType);

  protected:
    unsigned int _width;
    unsigned int _height;
    unsigned int _channelCount;
    BaseType _baseType;

    void* _resourceBuffer;
    ImageBufferDeallocator _resourceBufferDeallocator;
    unsigned int _resourceId = 0;
};

/// Create a uniform-color image with the given properties.
MX_RENDER_API ImagePtr createUniformImage(unsigned int width, unsigned int height, unsigned int channelCount, Image::BaseType baseType, const Color4& color);

/// Create a horizontal image strip from a vector of images with identical resolutions and formats.
MX_RENDER_API ImagePtr createImageStrip(const vector<ImagePtr>& imageVec);

/// Compute the maximum width and height of all images in the given vector.
MX_RENDER_API UnsignedIntPair getMaxDimensions(const vector<ImagePtr>& imageVec);

MATERIALX_NAMESPACE_END

#endif
