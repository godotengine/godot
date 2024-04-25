//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRender/ImageHandler.h>

#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/Util.h>

#include <iostream>

MATERIALX_NAMESPACE_BEGIN

const string IMAGE_PROPERTY_SEPARATOR("_");
const string UADDRESS_MODE_SUFFIX(IMAGE_PROPERTY_SEPARATOR + "uaddressmode");
const string VADDRESS_MODE_SUFFIX(IMAGE_PROPERTY_SEPARATOR + "vaddressmode");
const string FILTER_TYPE_SUFFIX(IMAGE_PROPERTY_SEPARATOR + "filtertype");
const string DEFAULT_COLOR_SUFFIX(IMAGE_PROPERTY_SEPARATOR + "default");

const string ImageLoader::BMP_EXTENSION = "bmp";
const string ImageLoader::EXR_EXTENSION = "exr";
const string ImageLoader::GIF_EXTENSION = "gif";
const string ImageLoader::HDR_EXTENSION = "hdr";
const string ImageLoader::JPG_EXTENSION = "jpg";
const string ImageLoader::JPEG_EXTENSION = "jpeg";
const string ImageLoader::PIC_EXTENSION = "pic";
const string ImageLoader::PNG_EXTENSION = "png";
const string ImageLoader::PSD_EXTENSION = "psd";
const string ImageLoader::TGA_EXTENSION = "tga";
const string ImageLoader::TIF_EXTENSION = "tif";
const string ImageLoader::TIFF_EXTENSION = "tiff";
const string ImageLoader::TX_EXTENSION = "tx";
const string ImageLoader::TXT_EXTENSION = "txt";
const string ImageLoader::TXR_EXTENSION = "txr";

//
// ImageLoader methods
//

bool ImageLoader::saveImage(const FilePath&, ConstImagePtr, bool)
{
    return false;
}

ImagePtr ImageLoader::loadImage(const FilePath&)
{
    return nullptr;
}

//
// ImageHandler methods
//

ImageHandler::ImageHandler(ImageLoaderPtr imageLoader)
{
    addLoader(imageLoader);
    _zeroImage = createUniformImage(2, 2, 4, Image::BaseType::UINT8, Color4(0.0f));
}

void ImageHandler::addLoader(ImageLoaderPtr loader)
{
    if (loader)
    {
        const StringSet& extensions = loader->supportedExtensions();
        for (const auto& extension : extensions)
        {
            _imageLoaders[extension].push_back(loader);
        }
    }
}

StringSet ImageHandler::supportedExtensions()
{
    StringSet extensions;
    for (const auto& pair : _imageLoaders)
    {
        extensions.insert(pair.first);
    }
    return extensions;
}

bool ImageHandler::saveImage(const FilePath& filePath,
                             ConstImagePtr image,
                             bool verticalFlip)
{
    if (!image)
    {
        return false;
    }

    FilePath foundFilePath = _searchPath.find(filePath);
    if (foundFilePath.isEmpty())
    {
        return false;
    }

    string extension = foundFilePath.getExtension();
    for (ImageLoaderPtr loader : _imageLoaders[extension])
    {
        bool saved = false;
        try
        {
            saved = loader->saveImage(foundFilePath, image, verticalFlip);
        }
        catch (std::exception& e)
        {
            std::cerr << "Exception in image I/O library: " << e.what() << std::endl;
        }
        if (saved)
        {
            return true;
        }
    }
    return false;
}

ImagePtr ImageHandler::acquireImage(const FilePath& filePath, const Color4& defaultColor)
{
    // Resolve the input filepath.
    FilePath resolvedFilePath = filePath;
    if (_resolver)
    {
        resolvedFilePath = _resolver->resolve(resolvedFilePath, FILENAME_TYPE_STRING);
    }

    // Return a cached image if available.
    ImagePtr cachedImage = getCachedImage(resolvedFilePath);
    if (cachedImage)
    {
        return cachedImage;
    }

    // Load and cache the requested image.
    ImagePtr image = loadImage(_searchPath.find(resolvedFilePath));
    if (image)
    {
        cacheImage(resolvedFilePath, image);
        return image;
    }

    // No valid image was found, so generate a uniform texture with the given default color.
    // TODO: This step assumes that the missing image and its default color are in the same
    //       color space, which is not always the case.
    ImagePtr defaultImage = createUniformImage(1, 1, 4, Image::BaseType::UINT8, defaultColor);
    cacheImage(resolvedFilePath, defaultImage);
    return defaultImage;
}

bool ImageHandler::bindImage(ImagePtr, const ImageSamplingProperties&)
{
    return false;
}

bool ImageHandler::unbindImage(ImagePtr)
{
    return false;
}

void ImageHandler::unbindImages()
{
    for (auto iter : _imageCache)
    {
        unbindImage(iter.second);
    }
}

bool ImageHandler::createRenderResources(ImagePtr, bool, bool)
{
    return false;
}

void ImageHandler::releaseRenderResources(ImagePtr)
{
}

ImageVec ImageHandler::getReferencedImages(ConstDocumentPtr doc)
{
    ImageVec imageVec;
    for (ElementPtr elem : doc->traverseTree())
    {
        if (elem->getActiveSourceUri() != doc->getSourceUri())
        {
            continue;
        }

        InputPtr input = elem->asA<Input>();
        if (input && input->getType() == FILENAME_TYPE_STRING)
        {
            ImagePtr image = acquireImage(input->getResolvedValueString());
            if (image)
            {
                imageVec.push_back(image);
            }
        }
    }
    return imageVec;
}

ImagePtr ImageHandler::loadImage(const FilePath& filePath)
{
    string extension = stringToLower(filePath.getExtension());
    for (ImageLoaderPtr loader : _imageLoaders[extension])
    {
        ImagePtr image;
        try
        {
            image = loader->loadImage(filePath);
        }
        catch (std::exception& e)
        {
            std::cerr << "Exception in image I/O library: " << e.what() << std::endl;
        }
        if (image)
        {
            return image;
        }
    }

    if (!filePath.isEmpty())
    {
        if (!filePath.exists())
        {
            std::cerr << string("Image file not found: ") + filePath.asString() << std::endl;
        }
        else if (!_imageLoaders.count(extension))
        {
            std::cerr << string("Unsupported image extension: ") + filePath.asString() << std::endl;
        }
        else
        {
            std::cerr << string("Image loader failed to parse image: ") + filePath.asString() << std::endl;
        }
    }

    return nullptr;
}

void ImageHandler::cacheImage(const string& filePath, ImagePtr image)
{
    _imageCache[filePath] = image;
}

ImagePtr ImageHandler::getCachedImage(const FilePath& filePath)
{
    if (_imageCache.count(filePath))
    {
        return _imageCache[filePath];
    }
    if (!filePath.isAbsolute())
    {
        for (const FilePath& path : _searchPath)
        {
            FilePath combined = path / filePath;
            if (_imageCache.count(combined))
            {
                return _imageCache[combined];
            }
        }
    }
    return nullptr;
}

//
// ImageSamplingProperties methods
//

void ImageSamplingProperties::setProperties(const string& fileNameUniform,
                                            const VariableBlock& uniformBlock)
{
    const int INVALID_MAPPED_INT_VALUE = -1; // Any value < 0 is not considered to be invalid

    // Get the additional texture parameters based on image uniform name
    // excluding the trailing "_file" postfix string
    string root = fileNameUniform;
    size_t pos = root.find_last_of(IMAGE_PROPERTY_SEPARATOR);
    if (pos != string::npos)
    {
        root = root.substr(0, pos);
    }

    const ShaderPort* port = uniformBlock.find(root + UADDRESS_MODE_SUFFIX);
    ValuePtr intValue = port ? port->getValue() : nullptr;
    uaddressMode = ImageSamplingProperties::AddressMode(intValue && intValue->isA<int>() ? intValue->asA<int>() : INVALID_MAPPED_INT_VALUE);

    port = uniformBlock.find(root + VADDRESS_MODE_SUFFIX);
    intValue = port ? port->getValue() : nullptr;
    vaddressMode = ImageSamplingProperties::AddressMode(intValue && intValue->isA<int>() ? intValue->asA<int>() : INVALID_MAPPED_INT_VALUE);

    port = uniformBlock.find(root + FILTER_TYPE_SUFFIX);
    intValue = port ? port->getValue() : nullptr;
    filterType = ImageSamplingProperties::FilterType(intValue && intValue->isA<int>() ? intValue->asA<int>() : INVALID_MAPPED_INT_VALUE);

    port = uniformBlock.find(root + DEFAULT_COLOR_SUFFIX);
    if (!port)
    {
        port = uniformBlock.find(root + DEFAULT_COLOR_SUFFIX + "_cm_in");
    }
    ValuePtr colorValue = port ? port->getValue() : nullptr;
    if (colorValue)
    {
        mapValueToColor(colorValue, defaultColor);
    }
}

bool ImageSamplingProperties::operator==(const ImageSamplingProperties& r) const
{
    return (enableMipmaps == r.enableMipmaps &&
            uaddressMode == r.uaddressMode &&
            vaddressMode == r.vaddressMode &&
            filterType == r.filterType &&
            defaultColor == r.defaultColor);
}

MATERIALX_NAMESPACE_END
