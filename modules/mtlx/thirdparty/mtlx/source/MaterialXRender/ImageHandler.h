//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_IMAGEHANDLER_H
#define MATERIALX_IMAGEHANDLER_H

/// @file
/// Image handler interfaces

#include <MaterialXRender/Export.h>
#include <MaterialXRender/Image.h>

#include <MaterialXFormat/File.h>

#include <MaterialXCore/Document.h>

MATERIALX_NAMESPACE_BEGIN

extern MX_RENDER_API const string IMAGE_PROPERTY_SEPARATOR;
extern MX_RENDER_API const string UADDRESS_MODE_SUFFIX;
extern MX_RENDER_API const string VADDRESS_MODE_SUFFIX;
extern MX_RENDER_API const string FILTER_TYPE_SUFFIX;
extern MX_RENDER_API const string DEFAULT_COLOR_SUFFIX;

class ImageHandler;
class ImageLoader;
class VariableBlock;

/// Shared pointer to an ImageHandler
using ImageHandlerPtr = std::shared_ptr<ImageHandler>;

/// Shared pointer to an ImageLoader
using ImageLoaderPtr = std::shared_ptr<ImageLoader>;

/// Map from strings to vectors of image loaders
using ImageLoaderMap = std::unordered_map<string, std::vector<ImageLoaderPtr>>;

/// @class ImageSamplingProperties
/// Interface to describe sampling properties for images.
class MX_RENDER_API ImageSamplingProperties
{
  public:
    /// Set the properties based on data in a uniform block.
    /// @param fileNameUniform Name of the file name uniform. Used to find
    ///        corresponding sampler data in the uniform block
    /// @param uniformBlock Block containing sampler uniforms
    void setProperties(const string& fileNameUniform,
                       const VariableBlock& uniformBlock);

    bool operator==(const ImageSamplingProperties& r) const;

    /// Address mode options. Matches enumerations allowed for image address
    /// modes, except UNSPECIFIED which indicates no explicit mode was defined.
    enum class AddressMode : int
    {
        UNSPECIFIED = -1,
        CONSTANT = 0,
        CLAMP = 1,
        PERIODIC = 2,
        MIRROR = 3
    };

    /// Address mode in U
    AddressMode uaddressMode = AddressMode::UNSPECIFIED;
    /// Address mode in V
    AddressMode vaddressMode = AddressMode::UNSPECIFIED;

    /// Filter type options. Matches enumerations allowed for image filter
    /// types, except UNSPECIFIED which indicates no explicit type was defined.
    enum class FilterType : int
    {
        UNSPECIFIED = -1,
        CLOSEST = 0,
        LINEAR = 1,
        CUBIC = 2
    };

    /// Filter type
    FilterType filterType = FilterType::UNSPECIFIED;

    /// Enable mipmaps
    bool enableMipmaps = true;

    /// Default color. Corresponds to the "default" value on the image
    /// node definition.
    Color4 defaultColor = { 0.0f, 0.0f, 0.0f, 1.0f };
};

/// @struct ImageSamplingKeyHasher
/// Class used for hashing ImageSamplingProperties in an unordered_map
struct MX_RENDER_API ImageSamplingKeyHasher
{
    size_t operator()(const ImageSamplingProperties& k) const
    {
        return (size_t) k.enableMipmaps                   + // 1 bit
               (((size_t) k.filterType         & 3) << 1) + // 2 bit
               ((((size_t) k.uaddressMode + 1) & 7) << 3) + // 3 bit
               ((((size_t) k.vaddressMode + 1) & 7) << 6) + // 3 bit
               ((((size_t) k.defaultColor[0] + 1))  << 9) ;
    }
};

/// @class ImageLoader
/// Abstract base class for file-system image loaders
class MX_RENDER_API ImageLoader
{
  public:
    ImageLoader()
    {
    }
    virtual ~ImageLoader() { }

    /// Standard image file extensions
    static const string BMP_EXTENSION;
    static const string EXR_EXTENSION;
    static const string GIF_EXTENSION;
    static const string HDR_EXTENSION;
    static const string JPG_EXTENSION;
    static const string JPEG_EXTENSION;
    static const string PIC_EXTENSION;
    static const string PNG_EXTENSION;
    static const string PSD_EXTENSION;
    static const string TGA_EXTENSION;
    static const string TIF_EXTENSION;
    static const string TIFF_EXTENSION;
    static const string TXT_EXTENSION;
    static const string TX_EXTENSION;
    static const string TXR_EXTENSION;

    /// Returns a list of supported extensions
    /// @return List of support extensions
    const StringSet& supportedExtensions() const
    {
        return _extensions;
    }

    /// Save an image to the file system. This method must be implemented by derived classes.
    /// @param filePath File path to be written
    /// @param image The image to be saved
    /// @param verticalFlip Whether the image should be flipped in Y during save
    /// @return if save succeeded
    virtual bool saveImage(const FilePath& filePath,
                           ConstImagePtr image,
                           bool verticalFlip = false);

    /// Load an image from the file system. This method must be implemented by derived classes.
    /// @param filePath The requested image file path.
    /// @return On success, a shared pointer to the loaded image; otherwise an empty shared pointer.
    virtual ImagePtr loadImage(const FilePath& filePath);

  protected:
    // List of supported string extensions
    StringSet _extensions;
};

/// @class ImageHandler
/// Base image handler class. Keeps track of images which are loaded from
/// disk via supplied ImageLoader. Derived classes are responsible for
/// determinining how to perform the logic for "binding" of these resources
/// for a given target (such as a given shading language).
class MX_RENDER_API ImageHandler
{
  public:
    static ImageHandlerPtr create(ImageLoaderPtr imageLoader)
    {
        return ImageHandlerPtr(new ImageHandler(imageLoader));
    }
    virtual ~ImageHandler() { }

    /// Add another image loader to the handler, which will be invoked if
    /// existing loaders cannot load a given image.
    void addLoader(ImageLoaderPtr loader);

    /// Get a list of extensions supported by the handler.
    StringSet supportedExtensions();

    /// Save image to disk. This method must be implemented by derived classes.
    /// The first image loader which supports the file name extension will be used.
    /// @param filePath File path to be written
    /// @param image The image to be saved
    /// @param verticalFlip Whether the image should be flipped in Y during save
    /// @return if save succeeded
    bool saveImage(const FilePath& filePath, ConstImagePtr image, bool verticalFlip = false);

    /// Acquire an image from the cache or file system.  If the image is not
    /// found in the cache, then each image loader will be applied in turn.
    /// If the image cannot be found by any loader, then a uniform image of the
    /// given default color will be returned.
    /// @param filePath File path of the image.
    /// @param defaultColor Default color to use as a fallback for missing images.
    /// @return On success, a shared pointer to the acquired image.
    ImagePtr acquireImage(const FilePath& filePath, const Color4& defaultColor = Color4(0.0f));

    /// Bind an image for rendering.
    /// @param image The image to bind.
    /// @param samplingProperties Sampling properties for the image.
    virtual bool bindImage(ImagePtr image, const ImageSamplingProperties& samplingProperties);

    /// Unbind an image, making it no longer active for rendering.
    /// @param image The image to unbind.
    virtual bool unbindImage(ImagePtr image);

    /// Unbind all images that are currently stored in the cache.
    void unbindImages();

    /// Set the search path to be used for finding images on the file system.
    void setSearchPath(const FileSearchPath& path)
    {
        _searchPath = path;
    }

    /// Return the image search path.
    const FileSearchPath& getSearchPath() const
    {
        return _searchPath;
    }

    /// Set the filename resolver for images.
    void setFilenameResolver(StringResolverPtr resolver)
    {
        _resolver = resolver;
    }

    /// Return the filename resolver for images.
    StringResolverPtr getFilenameResolver() const
    {
        return _resolver;
    }

    /// Create rendering resources for the given image.
    virtual bool createRenderResources(ImagePtr image, bool generateMipMaps, bool useAsRenderTarget = false);

    /// Release rendering resources for the given image, or for all cached images
    /// if no image pointer is specified.
    virtual void releaseRenderResources(ImagePtr image = nullptr);

    /// Clear the contents of the image cache, first releasing any render
    /// resources associated with cached images.
    void clearImageCache()
    {
        releaseRenderResources();
        _imageCache.clear();
    }

    /// Return a fallback image with zeroes in all channels.
    ImagePtr getZeroImage() const
    {
        return _zeroImage;
    }

    /// Acquire all images referenced by the given document, and return the
    /// images in a vector.
    ImageVec getReferencedImages(ConstDocumentPtr doc);

  protected:
    // Protected constructor.
    ImageHandler(ImageLoaderPtr imageLoader);

    // Load an image from the file system.
    ImagePtr loadImage(const FilePath& filePath);

    // Add an image to the cache.
    void cacheImage(const string& filePath, ImagePtr image);

    // Return the cached image, if found; otherwise return an empty
    // shared pointer.
    ImagePtr getCachedImage(const FilePath& filePath);

  protected:
    ImageLoaderMap _imageLoaders;
    ImageMap _imageCache;
    FileSearchPath _searchPath;
    StringResolverPtr _resolver;
    ImagePtr _zeroImage;
};

MATERIALX_NAMESPACE_END

#endif
