//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRender/Image.h>

#include <MaterialXRender/Types.h>

#include <MaterialXGenShader/Nodes/ConvolutionNode.h>

#include <cstring>
#include <fstream>
#include <limits>

MATERIALX_NAMESPACE_BEGIN

//
// Global functions
//

ImagePtr createUniformImage(unsigned int width, unsigned int height, unsigned int channelCount, Image::BaseType baseType, const Color4& color)
{
    ImagePtr image = Image::create(width, height, channelCount, baseType);
    image->createResourceBuffer();
    image->setUniformColor(color);
    return image;
}

ImagePtr createImageStrip(const vector<ImagePtr>& imageVec)
{
    ImagePtr refImage = imageVec.empty() ? nullptr : imageVec[0];
    if (!refImage)
    {
        return nullptr;
    }

    unsigned int srcWidth = refImage->getWidth();
    unsigned int srcHeight = refImage->getHeight();
    unsigned int destWidth = srcWidth * (unsigned int) imageVec.size();
    unsigned int destHeight = srcHeight;

    ImagePtr imageStrip = Image::create(destWidth, destHeight, refImage->getChannelCount(), refImage->getBaseType());
    imageStrip->createResourceBuffer();

    unsigned int srcRowStride = refImage->getRowStride();
    unsigned int destRowStride = imageStrip->getRowStride();

    for (unsigned int i = 0; i < imageVec.size(); i++)
    {
        ConstImagePtr srcImage = imageVec[i];
        if (!srcImage ||
            srcImage->getWidth() != srcWidth ||
            srcImage->getHeight() != srcHeight ||
            srcImage->getChannelCount() != refImage->getChannelCount() ||
            srcImage->getBaseType() != refImage->getBaseType())
        {
            throw Exception("Source images must have identical resolutions and formats in createImageStrip");
        }

        unsigned int xOffset = i * srcRowStride;

        uint8_t* src = (uint8_t*) srcImage->getResourceBuffer();
        uint8_t* dest = (uint8_t*) imageStrip->getResourceBuffer() + xOffset;

        for (unsigned int y = 0; y < srcHeight; y++)
        {
            memcpy(dest, src, srcRowStride);
            src += srcRowStride;
            dest += destRowStride;
        }
    }

    return imageStrip;
}

UnsignedIntPair getMaxDimensions(const vector<ImagePtr>& imageVec)
{
    UnsignedIntPair maxSize(0, 0);
    for (ImagePtr image : imageVec)
    {
        maxSize.first = std::max(maxSize.first, image->getWidth());
        maxSize.second = std::max(maxSize.second, image->getHeight());
    }
    return maxSize;
}

//
// Image methods
//

Image::Image(unsigned int width, unsigned int height, unsigned int channelCount, BaseType baseType) :
    _width(width),
    _height(height),
    _channelCount(channelCount),
    _baseType(baseType),
    _resourceBuffer(nullptr),
    _resourceBufferDeallocator(nullptr),
    _resourceId(0)
{
}

Image::~Image()
{
    releaseResourceBuffer();
}

unsigned int Image::getBaseStride() const
{
    if (_baseType == BaseType::FLOAT)
    {
        return 4;
    }
    if (_baseType == BaseType::HALF ||
        _baseType == BaseType::UINT16 ||
        _baseType == BaseType::INT16)
    {
        return 2;
    }
    if (_baseType == BaseType::UINT8 ||
        _baseType == BaseType::INT8)
    {
        return 1;
    }
    throw Exception("Unsupported base type in getBaseStride");
}

unsigned int Image::getMaxMipCount() const
{
    return (unsigned int) std::log2(std::max(_width, _height)) + 1;
}

void Image::setTexelColor(unsigned int x, unsigned int y, const Color4& color)
{
    if (x >= _width || y >= _height)
    {
        throw Exception("Invalid coordinates in setTexelColor");
    }
    if (!_resourceBuffer)
    {
        throw Exception("Invalid resource buffer in setTexelColor");
    }

    unsigned int writeChannels = std::min(_channelCount, (unsigned int) 4);
    if (_baseType == BaseType::FLOAT)
    {
        float* data = static_cast<float*>(_resourceBuffer) + (y * _width + x) * _channelCount;
        for (unsigned int c = 0; c < writeChannels; c++)
        {
            data[c] = color[c];
        }
    }
    else if (_baseType == BaseType::HALF)
    {
        Half* data = static_cast<Half*>(_resourceBuffer) + (y * _width + x) * _channelCount;
        for (unsigned int c = 0; c < writeChannels; c++)
        {
            data[c] = (Half) color[c];
        }
    }
    else if (_baseType == BaseType::UINT16)
    {
        uint16_t* data = static_cast<uint16_t*>(_resourceBuffer) + (y * _width + x) * _channelCount;
        for (unsigned int c = 0; c < writeChannels; c++)
        {
            data[c] = (uint16_t) std::round(color[c] * (float) std::numeric_limits<uint16_t>::max());
        }
    }
    else if (_baseType == BaseType::INT16)
    {
        int16_t* data = static_cast<int16_t*>(_resourceBuffer) + (y * _width + x) * _channelCount;
        for (unsigned int c = 0; c < writeChannels; c++)
        {
            data[c] = (int16_t) std::round(color[c] * (float) std::numeric_limits<int16_t>::max());
        }
    }
    else if (_baseType == BaseType::UINT8)
    {
        uint8_t* data = static_cast<uint8_t*>(_resourceBuffer) + (y * _width + x) * _channelCount;
        for (unsigned int c = 0; c < writeChannels; c++)
        {
            data[c] = (uint8_t) std::round(color[c] * (float) std::numeric_limits<uint8_t>::max());
        }
    }
    else if (_baseType == BaseType::INT8)
    {
        int8_t* data = static_cast<int8_t*>(_resourceBuffer) + (y * _width + x) * _channelCount;
        for (unsigned int c = 0; c < writeChannels; c++)
        {
            data[c] = (int8_t) std::round(color[c] * (float) std::numeric_limits<int8_t>::max());
        }
    }
    else
    {
        throw Exception("Unsupported base type in setTexelColor");
    }
}

Color4 Image::getTexelColor(unsigned int x, unsigned int y) const
{
    if (x >= _width || y >= _height)
    {
        throw Exception("Invalid coordinates in getTexelColor");
    }
    if (!_resourceBuffer)
    {
        throw Exception("Invalid resource buffer in getTexelColor");
    }

    if (_baseType == BaseType::FLOAT)
    {
        float* data = static_cast<float*>(_resourceBuffer) + (y * _width + x) * _channelCount;
        if (_channelCount == 4)
        {
            return Color4(data[0], data[1], data[2], data[3]);
        }
        else if (_channelCount == 3)
        {
            return Color4(data[0], data[1], data[2], 1.0f);
        }
        else if (_channelCount == 2)
        {
            return Color4(data[0], data[1], 0.0f, 1.0f);
        }
        else if (_channelCount == 1)
        {
            return Color4(data[0], data[0], data[0], 1.0f);
        }
        else
        {
            throw Exception("Unsupported channel count in getTexelColor");
        }
    }
    else if (_baseType == BaseType::HALF)
    {
        Half* data = static_cast<Half*>(_resourceBuffer) + (y * _width + x) * _channelCount;
        if (_channelCount == 4)
        {
            return Color4(data[0], data[1], data[2], data[3]);
        }
        else if (_channelCount == 3)
        {
            return Color4(data[0], data[1], data[2], 1.0f);
        }
        else if (_channelCount == 2)
        {
            return Color4(data[0], data[1], 0.0f, 1.0f);
        }
        else if (_channelCount == 1)
        {
            return Color4(data[0], data[0], data[0], 1.0f);
        }
        else
        {
            throw Exception("Unsupported channel count in getTexelColor");
        }
    }
    else if (_baseType == BaseType::UINT16)
    {
        uint16_t* data = static_cast<uint16_t*>(_resourceBuffer) + (y * _width + x) * _channelCount;
        const float MAX_VALUE = (float) std::numeric_limits<uint16_t>::max();
        if (_channelCount == 4)
        {
            return Color4(data[0] / MAX_VALUE, data[1] / MAX_VALUE, data[2] / MAX_VALUE, data[3] / MAX_VALUE);
        }
        else if (_channelCount == 3)
        {
            return Color4(data[0] / MAX_VALUE, data[1] / MAX_VALUE, data[2] / MAX_VALUE, 1.0f);
        }
        else if (_channelCount == 2)
        {
            return Color4(data[0] / MAX_VALUE, data[1] / MAX_VALUE, 0.0f, 1.0f);
        }
        else if (_channelCount == 1)
        {
            float scalar = data[0] / MAX_VALUE;
            return Color4(scalar, scalar, scalar, 1.0f);
        }
        else
        {
            throw Exception("Unsupported channel count in getTexelColor");
        }
    }
    else if (_baseType == BaseType::INT16)
    {
        int16_t* data = static_cast<int16_t*>(_resourceBuffer) + (y * _width + x) * _channelCount;
        const float MAX_VALUE = (float) std::numeric_limits<int16_t>::max();
        if (_channelCount == 4)
        {
            return Color4(data[0] / MAX_VALUE, data[1] / MAX_VALUE, data[2] / MAX_VALUE, data[3] / MAX_VALUE);
        }
        else if (_channelCount == 3)
        {
            return Color4(data[0] / MAX_VALUE, data[1] / MAX_VALUE, data[2] / MAX_VALUE, 1.0f);
        }
        else if (_channelCount == 2)
        {
            return Color4(data[0] / MAX_VALUE, data[1] / MAX_VALUE, 0.0f, 1.0f);
        }
        else if (_channelCount == 1)
        {
            float scalar = data[0] / MAX_VALUE;
            return Color4(scalar, scalar, scalar, 1.0f);
        }
        else
        {
            throw Exception("Unsupported channel count in getTexelColor");
        }
    }
    else if (_baseType == BaseType::UINT8)
    {
        uint8_t* data = static_cast<uint8_t*>(_resourceBuffer) + (y * _width + x) * _channelCount;
        const float MAX_VALUE = (float) std::numeric_limits<uint8_t>::max();
        if (_channelCount == 4)
        {
            return Color4(data[0] / MAX_VALUE, data[1] / MAX_VALUE, data[2] / MAX_VALUE, data[3] / MAX_VALUE);
        }
        else if (_channelCount == 3)
        {
            return Color4(data[0] / MAX_VALUE, data[1] / MAX_VALUE, data[2] / MAX_VALUE, 1.0f);
        }
        else if (_channelCount == 2)
        {
            return Color4(data[0] / MAX_VALUE, data[1] / MAX_VALUE, 0.0f, 1.0f);
        }
        else if (_channelCount == 1)
        {
            float scalar = data[0] / MAX_VALUE;
            return Color4(scalar, scalar, scalar, 1.0f);
        }
        else
        {
            throw Exception("Unsupported channel count in getTexelColor");
        }
    }
    else if (_baseType == BaseType::INT8)
    {
        int8_t* data = static_cast<int8_t*>(_resourceBuffer) + (y * _width + x) * _channelCount;
        const float MAX_VALUE = (float) std::numeric_limits<int8_t>::max();
        if (_channelCount == 4)
        {
            return Color4(data[0] / MAX_VALUE, data[1] / MAX_VALUE, data[2] / MAX_VALUE, data[3] / MAX_VALUE);
        }
        else if (_channelCount == 3)
        {
            return Color4(data[0] / MAX_VALUE, data[1] / MAX_VALUE, data[2] / MAX_VALUE, 1.0f);
        }
        else if (_channelCount == 2)
        {
            return Color4(data[0] / MAX_VALUE, data[1] / MAX_VALUE, 0.0f, 1.0f);
        }
        else if (_channelCount == 1)
        {
            float scalar = data[0] / MAX_VALUE;
            return Color4(scalar, scalar, scalar, 1.0f);
        }
        else
        {
            throw Exception("Unsupported channel count in getTexelColor");
        }
    }
    else
    {
        throw Exception("Unsupported base type in getTexelColor");
    }
}

Color4 Image::getAverageColor()
{
    Color4 averageColor;
    for (unsigned int y = 0; y < getHeight(); y++)
    {
        for (unsigned int x = 0; x < getWidth(); x++)
        {
            averageColor += getTexelColor(x, y);
        }
    }
    unsigned int sampleCount = getWidth() * getHeight();
    averageColor /= (float) sampleCount;
    return averageColor;
}

bool Image::isUniformColor(Color4* uniformColor)
{
    Color4 refColor = getTexelColor(0, 0);
    for (unsigned int y = 0; y < getHeight(); y++)
    {
        for (unsigned int x = 0; x < getWidth(); x++)
        {
            if (!x && !y)
            {
                continue;
            }
            if (getTexelColor(x, y) != refColor)
            {
                return false;
            }
        }
    }
    if (uniformColor)
    {
        *uniformColor = refColor;
    }
    return true;
}

void Image::setUniformColor(const Color4& color)
{
    for (unsigned int y = 0; y < getHeight(); y++)
    {
        for (unsigned int x = 0; x < getWidth(); x++)
        {
            setTexelColor(x, y, color);
        }
    }
}

void Image::applyMatrixTransform(const Matrix33& mat)
{
    for (unsigned int y = 0; y < getHeight(); y++)
    {
        for (unsigned int x = 0; x < getWidth(); x++)
        {
            Color4 color = getTexelColor(x, y);
            Vector3 vec(color[0], color[1], color[2]);
            vec = mat.multiply(vec);
            setTexelColor(x, y, Color4(vec[0], vec[1], vec[2], color[3]));
        }
    }
}

void Image::applyGammaTransform(float gamma)
{
    for (unsigned int y = 0; y < getHeight(); y++)
    {
        for (unsigned int x = 0; x < getWidth(); x++)
        {
            Color4 color = getTexelColor(x, y);
            Vector3 vec(color[0], color[1], color[2]);
            vec[0] = std::pow(std::max(vec[0], 0.0f), gamma);
            vec[1] = std::pow(std::max(vec[1], 0.0f), gamma);
            vec[2] = std::pow(std::max(vec[2], 0.0f), gamma);
            setTexelColor(x, y, Color4(vec[0], vec[1], vec[2], color[3]));
        }
    }
}

ImagePtr Image::copy(unsigned int channelCount, BaseType baseType) const
{
    ImagePtr newImage = Image::create(getWidth(), getHeight(), channelCount, baseType);
    newImage->createResourceBuffer();

    for (int y = 0; y < (int) getHeight(); y++)
    {
        for (int x = 0; x < (int) getWidth(); x++)
        {
            newImage->setTexelColor(x, y, getTexelColor(x, y));
        }
    }

    return newImage;
}

ImagePtr Image::applyBoxBlur()
{
    ImagePtr blurImage = Image::create(getWidth(), getHeight(), getChannelCount(), getBaseType());
    blurImage->createResourceBuffer();

    for (int y = 0; y < (int) getHeight(); y++)
    {
        for (int x = 0; x < (int) getWidth(); x++)
        {
            Color4 blurColor;
            for (int dy = -1; dy <= 1; dy++)
            {
                int sy = std::min(std::max(y + dy, 0), (int) getHeight() - 1);
                for (int dx = -1; dx <= 1; dx++)
                {
                    int sx = std::min(std::max(x + dx, 0), (int) getWidth() - 1);
                    blurColor += getTexelColor(sx, sy);
                }
            }
            blurColor /= 9.0f;
            blurImage->setTexelColor(x, y, blurColor);
        }
    }

    return blurImage;
}

ImagePtr Image::applyGaussianBlur()
{
    ImagePtr blurImage1 = Image::create(getWidth(), getHeight(), getChannelCount(), getBaseType());
    ImagePtr blurImage2 = Image::create(getWidth(), getHeight(), getChannelCount(), getBaseType());
    blurImage1->createResourceBuffer();
    blurImage2->createResourceBuffer();

    for (int y = 0; y < (int) getHeight(); y++)
    {
        for (int x = 0; x < (int) getWidth(); x++)
        {
            Color4 blurColor;
            unsigned int weightIndex = 0;
            for (int dy = -3; dy <= 3; dy++, weightIndex++)
            {
                int sy = std::min(std::max(y + dy, 0), (int) getHeight() - 1);
                blurColor += getTexelColor(x, sy) * GAUSSIAN_KERNEL_7[weightIndex];
            }
            blurImage1->setTexelColor(x, y, blurColor);
        }
    }

    for (int y = 0; y < (int) getHeight(); y++)
    {
        for (int x = 0; x < (int) getWidth(); x++)
        {
            Color4 blurColor;
            unsigned int weightIndex = 0;
            for (int dx = -3; dx <= 3; dx++, weightIndex++)
            {
                int sx = std::min(std::max(x + dx, 0), (int) getWidth() - 1);
                blurColor += blurImage1->getTexelColor(sx, y) * GAUSSIAN_KERNEL_7[weightIndex];
            }
            blurImage2->setTexelColor(x, y, blurColor);
        }
    }

    return blurImage2;
}

ImagePair Image::splitByLuminance(float luminance)
{
    ImagePtr underflowImage = Image::create(getWidth(), getHeight(), getChannelCount(), getBaseType());
    ImagePtr overflowImage = Image::create(getWidth(), getHeight(), getChannelCount(), getBaseType());
    underflowImage->createResourceBuffer();
    overflowImage->createResourceBuffer();

    for (unsigned int y = 0; y < getHeight(); y++)
    {
        for (unsigned int x = 0; x < getWidth(); x++)
        {
            Color4 envColor = getTexelColor(x, y);
            Color4 underflowColor(
                std::min(envColor[0], luminance),
                std::min(envColor[1], luminance),
                std::min(envColor[2], luminance), 1.0f);
            Color4 overflowColor(
                std::max(envColor[0] - underflowColor[0], 0.0f),
                std::max(envColor[1] - underflowColor[1], 0.0f),
                std::max(envColor[2] - underflowColor[2], 0.0f), 1.0f);
            underflowImage->setTexelColor(x, y, underflowColor);
            overflowImage->setTexelColor(x, y, overflowColor);
        }
    }

    return std::make_pair(underflowImage, overflowImage);
}

void Image::writeTable(const FilePath& filePath, unsigned int channel)
{
    std::ofstream ofs(filePath.asString());
    ofs << "X Y Z" << std::endl;
    for (unsigned int y = 0; y < getHeight(); y++)
    {
        for (unsigned int x = 0; x < getWidth(); x++)
        {
            double dx = ((double) x + 0.5) / (double) getWidth();
            double dy = ((double) y + 0.5) / (double) getHeight();
            double dz = getTexelColor(x, y)[channel];
            ofs << dx << " " << dy << " " << dz << std::endl;
        }
    }
}

void Image::createResourceBuffer()
{
    releaseResourceBuffer();
    _resourceBuffer = malloc(_width * _height * _channelCount * getBaseStride());
    _resourceBufferDeallocator = nullptr;
}

void Image::releaseResourceBuffer()
{
    if (_resourceBuffer)
    {
        if (_resourceBufferDeallocator)
        {
            _resourceBufferDeallocator(_resourceBuffer);
        }
        else
        {
            free(_resourceBuffer);
        }
        _resourceBuffer = nullptr;
    }
}

MATERIALX_NAMESPACE_END
