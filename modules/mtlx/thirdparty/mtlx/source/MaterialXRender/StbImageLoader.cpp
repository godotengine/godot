//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRender/StbImageLoader.h>

#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable : 4100)
    #pragma warning(disable : 4505)
    #pragma warning(disable : 4996)
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC 1
#include <MaterialXRender/External/StbImage/stb_image.h>

#if defined(__APPLE__)
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC 1
#include <MaterialXRender/External/StbImage/stb_image_write.h>

#if defined(_MSC_VER)
    #pragma warning(pop)
#endif

MATERIALX_NAMESPACE_BEGIN

bool StbImageLoader::saveImage(const FilePath& filePath,
                               ConstImagePtr image,
                               bool verticalFlip)
{
    bool isChar = image->getBaseType() == Image::BaseType::UINT8;
    bool isFloat = image->getBaseType() == Image::BaseType::FLOAT;
    if (!isChar && !isFloat)
    {
        return false;
    }

    int returnValue = -1;

    // Set global "flip" flag
    int prevFlip = stbi__flip_vertically_on_write;
    stbi__flip_vertically_on_write = verticalFlip ? 1 : 0;

    int w = static_cast<int>(image->getWidth());
    int h = static_cast<int>(image->getHeight());
    int channels = static_cast<int>(image->getChannelCount());
    void* data = image->getResourceBuffer();

    const string filePathName = filePath.asString();

    string extension = filePath.getExtension();
    if (!isFloat)
    {
        if (extension == PNG_EXTENSION)
        {
            returnValue = stbi_write_png(filePathName.c_str(), w, h, channels, data, w * image->getChannelCount());
        }
        else if (extension == BMP_EXTENSION)
        {
            returnValue = stbi_write_bmp(filePathName.c_str(), w, h, channels, data);
        }
        else if (extension == TGA_EXTENSION)
        {
            returnValue = stbi_write_tga(filePathName.c_str(), w, h, channels, data);
        }
        else if (extension == JPG_EXTENSION || extension == JPEG_EXTENSION)
        {
            returnValue = stbi_write_jpg(filePathName.c_str(), w, h, channels, data, 100);
        }
    }
    else
    {
        if (extension == HDR_EXTENSION)
        {
            returnValue = stbi_write_hdr(filePathName.c_str(), w, h, channels, static_cast<float*>(data));
        }
    }

    if (verticalFlip)
    {
        stbi__flip_vertically_on_write = prevFlip;
    }
    return (returnValue == 1);
}

ImagePtr StbImageLoader::loadImage(const FilePath& filePath)
{
    int width = 0;
    int height = 0;
    int channelCount = 0;
    Image::BaseType baseType = Image::BaseType::UINT8;
    void* buffer = nullptr;

    // Select standard or float reader based on file extension.
    string extension = filePath.getExtension();
    if (extension == HDR_EXTENSION)
    {
        buffer = stbi_loadf(filePath.asString().c_str(), &width, &height, &channelCount, 0);
        baseType = Image::BaseType::FLOAT;
    }
    else
    {
        buffer = stbi_load(filePath.asString().c_str(), &width, &height, &channelCount, 0);
        baseType = Image::BaseType::UINT8;
    }
    if (!buffer)
    {
        return nullptr;
    }

    // Create the image object.
    ImagePtr image = Image::create(width, height, channelCount, baseType);
    image->setResourceBuffer(buffer);
    image->setResourceBufferDeallocator(&stbi_image_free);
    return image;
}

#if defined(__APPLE__)
    #pragma clang diagnostic pop
#endif

MATERIALX_NAMESPACE_END
