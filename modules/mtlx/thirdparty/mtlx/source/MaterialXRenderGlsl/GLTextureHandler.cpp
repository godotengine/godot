//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRenderGlsl/GLTextureHandler.h>

#include <MaterialXRenderGlsl/GlslProgram.h>
#include <MaterialXRenderGlsl/External/Glad/glad.h>

#include <MaterialXRender/ShaderRenderer.h>

#include <iostream>

MATERIALX_NAMESPACE_BEGIN

GLTextureHandler::GLTextureHandler(ImageLoaderPtr imageLoader) :
    ImageHandler(imageLoader)
{
    if (!glActiveTexture)
    {
        gladLoadGL();
    }

    int maxTextureUnits;
    glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &maxTextureUnits);
    _boundTextureLocations.resize(maxTextureUnits, GlslProgram::UNDEFINED_OPENGL_RESOURCE_ID);
}

bool GLTextureHandler::bindImage(ImagePtr image, const ImageSamplingProperties& samplingProperties)
{
    if (!image)
    {
        return false;
    }

    // Create renderer resources if needed.
    if (image->getResourceId() == GlslProgram::UNDEFINED_OPENGL_RESOURCE_ID)
    {
        if (!createRenderResources(image, true))
        {
            return false;
        }
    }

    // Update bound location if not already bound
    int textureUnit = getBoundTextureLocation(image->getResourceId());
    if (textureUnit < 0)
    {
        textureUnit = getNextAvailableTextureLocation();
    }
    if (textureUnit < 0)
    {
        std::cerr << "Exceeded maximum number of bound textures in GLTextureHandler::bindImage" << std::endl;
        return false;
    }      
    _boundTextureLocations[textureUnit] = image->getResourceId();

    glActiveTexture(GL_TEXTURE0 + textureUnit);
    glBindTexture(GL_TEXTURE_2D, image->getResourceId());

    // Set up texture properties
    GLint minFilterType = mapFilterTypeToGL(samplingProperties.filterType, samplingProperties.enableMipmaps);
    GLint magFilterType = GL_LINEAR; // Magnification filters are more restrictive than minification
    GLint uaddressMode = mapAddressModeToGL(samplingProperties.uaddressMode);
    GLint vaddressMode = mapAddressModeToGL(samplingProperties.vaddressMode);
    Color4 borderColor(samplingProperties.defaultColor);
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, uaddressMode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, vaddressMode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilterType);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilterType);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, 16.0f);

    return true;
}

bool GLTextureHandler::unbindImage(ImagePtr image)
{
    if (image->getResourceId() != GlslProgram::UNDEFINED_OPENGL_RESOURCE_ID)
    {
        int textureUnit = getBoundTextureLocation(image->getResourceId());
        if (textureUnit >= 0)
        {
            glActiveTexture(GL_TEXTURE0 + textureUnit);
            glBindTexture(GL_TEXTURE_2D, GlslProgram::UNDEFINED_OPENGL_RESOURCE_ID);
            _boundTextureLocations[textureUnit] = GlslProgram::UNDEFINED_OPENGL_RESOURCE_ID;
            return true;
        }
    }
    return false;
}

bool GLTextureHandler::createRenderResources(ImagePtr image, bool generateMipMaps, bool)
{
    if (image->getResourceId() == GlslProgram::UNDEFINED_OPENGL_RESOURCE_ID)
    {
        unsigned int resourceId;
        glGenTextures(1, &resourceId);
        if (resourceId == GlslProgram::UNDEFINED_OPENGL_RESOURCE_ID)
        {
            std::cerr << "Failed to generate render resource for texture" << std::endl;
            return false;
        }
        image->setResourceId(resourceId);
    }

    int textureUnit = getNextAvailableTextureLocation();
    if (textureUnit < 0)
    {
        return false;
    }

    glActiveTexture(GL_TEXTURE0 + textureUnit);
    glBindTexture(GL_TEXTURE_2D, image->getResourceId());

    int glType, glFormat, glInternalFormat;
    mapTextureFormatToGL(image->getBaseType(), image->getChannelCount(), false,
        glType, glFormat, glInternalFormat);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, glInternalFormat, image->getWidth(), image->getHeight(),
        0, glFormat, glType, image->getResourceBuffer());
    if (image->getChannelCount() == 1)
    {
        GLint swizzleMask[] = { GL_RED, GL_RED, GL_RED, GL_ONE };
        glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask);
    }

    if (generateMipMaps)
    {
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    glBindTexture(GL_TEXTURE_2D, 0);

    return true;
}
    
void GLTextureHandler::releaseRenderResources(ImagePtr image)
{
    if (!image)
    {
        for (auto iter : _imageCache)
        {
            if (iter.second)
            {
                releaseRenderResources(iter.second);
            }
        }
        return;
    }
    if (image->getResourceId() == GlslProgram::UNDEFINED_OPENGL_RESOURCE_ID)
    {
        return;
    }

    unbindImage(image);
    unsigned int resourceId = image->getResourceId();
    glDeleteTextures(1, &resourceId);
    image->setResourceId(GlslProgram::UNDEFINED_OPENGL_RESOURCE_ID);
}

int GLTextureHandler::getBoundTextureLocation(unsigned int resourceId)
{
    for (size_t i = 0; i < _boundTextureLocations.size(); i++)
    {
        if (_boundTextureLocations[i] == resourceId)
        {
            return static_cast<int>(i);
        }
    }
    return -1;
}

int GLTextureHandler::getNextAvailableTextureLocation()
{
    for (size_t i = 0; i < _boundTextureLocations.size(); i++)
    {
        if (_boundTextureLocations[i] == GlslProgram::UNDEFINED_OPENGL_RESOURCE_ID)
        {
            return static_cast<int>(i);
        }
    }
    return -1;
}

int GLTextureHandler::mapAddressModeToGL(ImageSamplingProperties::AddressMode addressModeEnum)
{
    const std::array<int, 4> ADDRESS_MODES
    {
        // Constant color. Use clamp to border
        // with border color to achieve this
        GL_CLAMP_TO_BORDER,

        // Clamp
        GL_CLAMP_TO_EDGE,

        // Repeat
        GL_REPEAT,

        // Mirror
        GL_MIRRORED_REPEAT
    };

    int addressMode = GL_REPEAT;
    if (addressModeEnum != ImageSamplingProperties::AddressMode::UNSPECIFIED)
    {
        addressMode = ADDRESS_MODES[static_cast<int>(addressModeEnum)];
    }
    return addressMode;
}

int GLTextureHandler::mapFilterTypeToGL(ImageSamplingProperties::FilterType filterTypeEnum, bool enableMipmaps)
{
    if (filterTypeEnum == ImageSamplingProperties::FilterType::CLOSEST)
    {
        return enableMipmaps ? GL_NEAREST_MIPMAP_NEAREST : GL_NEAREST;
    }
    return enableMipmaps ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR;
}

void GLTextureHandler::mapTextureFormatToGL(Image::BaseType baseType, unsigned int channelCount, bool srgb,
                                            int& glType, int& glFormat, int& glInternalFormat)
{
    switch (channelCount)
    {
        case 4: glFormat = GL_RGBA; break;
        case 3: glFormat = GL_RGB; break;
        case 2: glFormat = GL_RG; break;
        case 1: glFormat = GL_RED; break;
        default: throw Exception("Unsupported channel count in mapTextureFormatToGL");
    }

    if (baseType == Image::BaseType::UINT8 || baseType == Image::BaseType::INT8)
    {
        glType = baseType == Image::BaseType::UINT8 ? GL_UNSIGNED_BYTE : GL_BYTE;
        switch (channelCount)
        {
            case 4: glInternalFormat = srgb ? GL_SRGB8_ALPHA8 : GL_RGBA8; break;
            case 3: glInternalFormat = srgb ? GL_SRGB8 : GL_RGB8; break;
            case 2: glInternalFormat = GL_RG8; break;
            case 1: glInternalFormat = GL_R8; break;
            default: throw Exception("Unsupported channel count in mapTextureFormatToGL");
        }
    }
    else if (baseType == Image::BaseType::UINT16 || baseType == Image::BaseType::INT16)
    {
        glType = baseType == Image::BaseType::UINT16 ? GL_UNSIGNED_SHORT : GL_SHORT;
        switch (channelCount)
        {
            case 4: glInternalFormat = GL_RGBA16; break;
            case 3: glInternalFormat = GL_RGB16; break;
            case 2: glInternalFormat = GL_RG16; break;
            case 1: glInternalFormat = GL_R16; break;
            default: throw Exception("Unsupported channel count in mapTextureFormatToGL");
        }
    }
    else if (baseType == Image::BaseType::HALF)
    {
        glType = GL_HALF_FLOAT;
        switch (channelCount)
        {
            case 4: glInternalFormat = GL_RGBA16F; break;
            case 3: glInternalFormat = GL_RGB16F; break;
            case 2: glInternalFormat = GL_RG16F; break;
            case 1: glInternalFormat = GL_R16F; break;
            default: throw Exception("Unsupported channel count in mapTextureFormatToGL");
        }
    }
    else if (baseType == Image::BaseType::FLOAT)
    {
        glType = GL_FLOAT;
        switch (channelCount)
        {
            case 4: glInternalFormat = GL_RGBA32F; break;
            case 3: glInternalFormat = GL_RGB32F; break;
            case 2: glInternalFormat = GL_RG32F; break;
            case 1: glInternalFormat = GL_R32F; break;
            default: throw Exception("Unsupported channel count in mapTextureFormatToGL");
        }
    }
    else
    {
        throw Exception("Unsupported base type in mapTextureFormatToGL");
    }
}

MATERIALX_NAMESPACE_END
