//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRenderMsl/MetalTextureHandler.h>
#include <MaterialXRenderMsl/MslPipelineStateObject.h>
#include <MaterialXRender/ShaderRenderer.h>

#include <iostream>

MATERIALX_NAMESPACE_BEGIN

MetalTextureHandler::MetalTextureHandler(id<MTLDevice> device, ImageLoaderPtr imageLoader) :
    ImageHandler(imageLoader)
{
    int maxTextureUnits = 31;
    _boundTextureLocations.resize(maxTextureUnits, MslProgram::UNDEFINED_METAL_RESOURCE_ID);
    _device = device;
}

bool MetalTextureHandler::bindImage(ImagePtr image, const ImageSamplingProperties& samplingProperties)
{
    // Create renderer resources if needed.
    if (image->getResourceId() == MslProgram::UNDEFINED_METAL_RESOURCE_ID)
    {
        if (!createRenderResources(image, true))
        {
            return false;
        }
    }
    _imageBindingInfo[image->getResourceId()] = std::make_pair(image, samplingProperties);
    return true;
}

id<MTLSamplerState> MetalTextureHandler::getSamplerState(const ImageSamplingProperties& samplingProperties)
{
    if(_imageSamplerStateMap.find(samplingProperties) == _imageSamplerStateMap.end())
    {
        MTLSamplerDescriptor* samplerDesc = [MTLSamplerDescriptor new];
        [samplerDesc setSAddressMode:mapAddressModeToMetal(samplingProperties.uaddressMode)];
        [samplerDesc setRAddressMode:mapAddressModeToMetal(samplingProperties.uaddressMode)];
        [samplerDesc setTAddressMode:mapAddressModeToMetal(samplingProperties.vaddressMode)];
        [samplerDesc setBorderColor:samplingProperties.defaultColor[0] == 0 ? MTLSamplerBorderColorOpaqueBlack : MTLSamplerBorderColorOpaqueWhite];
        MTLSamplerMinMagFilter minmagFilter;
        MTLSamplerMipFilter    mipFilter;
        mapFilterTypeToMetal(samplingProperties.filterType, samplingProperties.enableMipmaps, minmagFilter, mipFilter);
        // Magnification filters are more restrictive than minification
        [samplerDesc setMagFilter:MTLSamplerMinMagFilterLinear];
        [samplerDesc setMinFilter:minmagFilter];
        [samplerDesc setMipFilter:mipFilter];
        [samplerDesc setMaxAnisotropy:16];
    
        _imageSamplerStateMap[samplingProperties] = [_device newSamplerStateWithDescriptor:samplerDesc];
    }
    
    return _imageSamplerStateMap[samplingProperties];
}

bool MetalTextureHandler::bindImage(id<MTLRenderCommandEncoder> renderCmdEncoder,
                                    int textureUnit, ImagePtr image)
{
    // Create renderer resources if needed.
    if (image->getResourceId() == MslProgram::UNDEFINED_METAL_RESOURCE_ID)
    {
        if (!createRenderResources(image, true))
        {
            return false;
        }
    }

    _boundTextureLocations[textureUnit] = image->getResourceId();
    
    [renderCmdEncoder setFragmentTexture:_metalTextureMap[image->getResourceId()] atIndex:textureUnit];
    [renderCmdEncoder setFragmentSamplerState:getSamplerState(_imageBindingInfo[image->getResourceId()].second) atIndex:textureUnit];

    return true;
}

id<MTLTexture> MetalTextureHandler::getAssociatedMetalTexture(ImagePtr image)
{
    if(image)
    {
        auto tex = _metalTextureMap.find(image->getResourceId());
        if(tex != _metalTextureMap.end())
            return (tex->second);
    }
    return nil;
}

id<MTLTexture> MetalTextureHandler::getMTLTextureForImage(unsigned int index) const
{
    auto imageInfo = _imageBindingInfo.find(index);
    if(imageInfo != _imageBindingInfo.end())
    {
        if(!imageInfo->second.first)
            return nil;
        
        auto metalTexture = _metalTextureMap.find(imageInfo->second.first->getResourceId());
        if(metalTexture != _metalTextureMap.end())
            return metalTexture->second;
    }
    
    return nil;
}

id<MTLSamplerState> MetalTextureHandler::getMTLSamplerStateForImage(unsigned int index)
{
    auto imageInfo = _imageBindingInfo.find(index);
    if(imageInfo != _imageBindingInfo.end())
    {
        return getSamplerState(imageInfo->second.second);
    }
    return nil;
}

bool MetalTextureHandler::unbindImage(ImagePtr image)
{
    if (image->getResourceId() != MslProgram::UNDEFINED_METAL_RESOURCE_ID)
    {
        int textureUnit = getBoundTextureLocation(image->getResourceId());
        if (textureUnit >= 0)
        {
            _boundTextureLocations[textureUnit] = MslProgram::UNDEFINED_METAL_RESOURCE_ID;
            return true;
        }
    }
    return false;
}

bool MetalTextureHandler::createRenderResources(ImagePtr image, bool generateMipMaps, bool useAsRenderTarget)
{
    id<MTLTexture> texture = nil;
    
    MTLPixelFormat pixelFormat;
    MTLDataType    dataType;
    
    if (image->getResourceId() == MslProgram::UNDEFINED_METAL_RESOURCE_ID)
    {
        static unsigned int resourceId = 0;
        ++resourceId;
        
        mapTextureFormatToMetal(image->getBaseType(), image->getChannelCount(),
                                false, dataType, pixelFormat);
        
        MTLTextureDescriptor* texDesc = [MTLTextureDescriptor new];
        [texDesc setTextureType:MTLTextureType2D];
        texDesc.width = image->getWidth();
        texDesc.height = image->getHeight();
        texDesc.mipmapLevelCount = generateMipMaps ? image->getMaxMipCount() : 1;
        texDesc.usage = MTLTextureUsageShaderRead |
                    (useAsRenderTarget ? MTLTextureUsageRenderTarget : 0);
        texDesc.resourceOptions = MTLResourceStorageModePrivate;
        texDesc.pixelFormat = pixelFormat;
        if(generateMipMaps)
        {
            if(image->getChannelCount() == 1)
            {
                texDesc.swizzle = MTLTextureSwizzleChannelsMake(
                        MTLTextureSwizzleRed,
                        MTLTextureSwizzleRed,
                        MTLTextureSwizzleRed,
                        MTLTextureSwizzleRed);
            }
            else if(image->getChannelCount() == 2)
            {
                texDesc.swizzle = MTLTextureSwizzleChannelsMake(
                        MTLTextureSwizzleRed,
                        MTLTextureSwizzleGreen,
                        MTLTextureSwizzleRed,
                        MTLTextureSwizzleGreen);
            }
        }
        texture = [_device newTextureWithDescriptor:texDesc];
        _metalTextureMap[resourceId] = texture;
        image->setResourceId(resourceId);
    }
    else
    {
        mapTextureFormatToMetal(image->getBaseType(), image->getChannelCount(), false,
                                dataType, pixelFormat);
        
        texture = _metalTextureMap[image->getResourceId()];
    }
    

    id<MTLCommandQueue> cmdQueue = [_device newCommandQueue];
    id<MTLCommandBuffer> cmdBuffer = [cmdQueue commandBuffer];
    
    id<MTLBlitCommandEncoder> blitCmdEncoder = [cmdBuffer blitCommandEncoder];
    
    NSUInteger channelCount = image->getChannelCount();
    
    NSUInteger sourceBytesPerRow =
        image->getWidth() *
        channelCount *
        getTextureBaseTypeSize(image->getBaseType());
    NSUInteger sourceBytesPerImage =
        sourceBytesPerRow *
        image->getHeight();
    
    std::vector<float>         rearrangedDataF;
    std::vector<unsigned char> rearrangedDataC;
    void* imageData = image->getResourceBuffer();
    
    if ((pixelFormat == MTLPixelFormatRGBA32Float || pixelFormat == MTLPixelFormatRGBA8Unorm) && channelCount == 3)
    {
        bool isFloat = pixelFormat == MTLPixelFormatRGBA32Float;
        
        sourceBytesPerRow   = sourceBytesPerRow   / 3 * 4;
        sourceBytesPerImage = sourceBytesPerImage / 3 * 4;
        
        size_t srcIdx = 0;
        
        if(isFloat)
        {
            rearrangedDataF.resize(sourceBytesPerImage / sizeof(float));
            for(size_t dstIdx = 0; dstIdx < rearrangedDataF.size(); ++dstIdx)
            {
                if((dstIdx & 0x3) == 3)
                {
                    rearrangedDataF[dstIdx] = 1.0f;
                    continue;
                }
                
                rearrangedDataF[dstIdx] = ((float*)imageData)[srcIdx++];
            }
            
            imageData = rearrangedDataF.data();
        }
        else
        {
            rearrangedDataC.resize(sourceBytesPerImage);
            for(size_t dstIdx = 0; dstIdx < rearrangedDataC.size(); ++dstIdx)
            {
                if((dstIdx & 0x3) == 3)
                {
                    rearrangedDataC[dstIdx] = 255;
                    continue;
                }
                
                rearrangedDataC[dstIdx] = ((unsigned char*)imageData)[srcIdx++];
            }
            
            imageData = rearrangedDataC.data();
        }
 
        channelCount = 4;
    }
    
    id<MTLBuffer> buffer = nil;
    if(imageData)
    {
        buffer = [_device newBufferWithBytes:imageData
                                                    length:sourceBytesPerImage
                                                   options:MTLStorageModeShared];
        [blitCmdEncoder copyFromBuffer:buffer sourceOffset:0
                     sourceBytesPerRow:sourceBytesPerRow
                   sourceBytesPerImage:sourceBytesPerImage
                            sourceSize:MTLSizeMake(image->getWidth(), image->getHeight(), 1)
                             toTexture:texture
                      destinationSlice:0
                      destinationLevel:0
                     destinationOrigin:MTLOriginMake(0, 0, 0)];
    }
    
    if(generateMipMaps &&  image->getMaxMipCount() > 1)
        [blitCmdEncoder generateMipmapsForTexture:texture];
    
    [blitCmdEncoder endEncoding];
    
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];
    
    if(buffer)
        [buffer release];
  
    return true;
}
    
void MetalTextureHandler::releaseRenderResources(ImagePtr image)
{
    if(!image)
        return;
    
    if (image->getResourceId() == MslProgram::UNDEFINED_METAL_RESOURCE_ID)
    {
        return;
    }

    unbindImage(image);
    unsigned int resourceId = image->getResourceId();
    auto tex = _metalTextureMap.find(resourceId);
    if(tex != _metalTextureMap.end())
    {
        [tex->second release];
    }
    _metalTextureMap.erase(resourceId);
    image->setResourceId(MslProgram::UNDEFINED_METAL_RESOURCE_ID);
}

int MetalTextureHandler::getBoundTextureLocation(unsigned int resourceId)
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

MTLSamplerAddressMode MetalTextureHandler::mapAddressModeToMetal(ImageSamplingProperties::AddressMode addressModeEnum)
{
    const vector<MTLSamplerAddressMode> addressModes
    {
        // Constant color. Use clamp to border
        // with border color to achieve this
        MTLSamplerAddressModeClampToBorderColor,

        // Clamp
        MTLSamplerAddressModeClampToEdge,

        // Repeat
        MTLSamplerAddressModeRepeat,

        // Mirror
        MTLSamplerAddressModeMirrorRepeat
    };

    MTLSamplerAddressMode addressMode = MTLSamplerAddressModeRepeat;
    if (addressModeEnum != ImageSamplingProperties::AddressMode::UNSPECIFIED)
    {
        addressMode = addressModes[static_cast<int>(addressModeEnum)];
    }
    return addressMode;
}

void MetalTextureHandler::mapFilterTypeToMetal(ImageSamplingProperties::FilterType filterTypeEnum, bool enableMipmaps, MTLSamplerMinMagFilter& minMagFilter, MTLSamplerMipFilter& mipFilter)
{
    if(enableMipmaps)
    {
        if(filterTypeEnum == ImageSamplingProperties::FilterType::LINEAR ||
           filterTypeEnum == ImageSamplingProperties::FilterType::CUBIC  ||
           filterTypeEnum == ImageSamplingProperties::FilterType::UNSPECIFIED)
        {
            minMagFilter = MTLSamplerMinMagFilterLinear;
            mipFilter = MTLSamplerMipFilterLinear;
        }
        else
        {
            minMagFilter = MTLSamplerMinMagFilterNearest;
            mipFilter = MTLSamplerMipFilterNearest;
        }
    }
    else
    {
        if(filterTypeEnum == ImageSamplingProperties::FilterType::LINEAR ||
           filterTypeEnum == ImageSamplingProperties::FilterType::CUBIC  ||
           filterTypeEnum == ImageSamplingProperties::FilterType::UNSPECIFIED)
        {
            minMagFilter = MTLSamplerMinMagFilterLinear;
            mipFilter = MTLSamplerMipFilterNotMipmapped;
        }
        else
        {
            minMagFilter = MTLSamplerMinMagFilterNearest;
            mipFilter = MTLSamplerMipFilterNotMipmapped;
        }
    }
}

void MetalTextureHandler::mapTextureFormatToMetal(Image::BaseType baseType, unsigned int channelCount, bool srgb,
                                            MTLDataType& dataType, MTLPixelFormat& pixelFormat)
{
    if (baseType == Image::BaseType::UINT8)
    {
        dataType = MTLDataTypeChar;
        switch (channelCount)
        {
            case 4: pixelFormat = srgb ? MTLPixelFormatRGBA8Unorm_sRGB : MTLPixelFormatRGBA8Unorm; dataType = MTLDataTypeChar4; break;
            case 3: pixelFormat = srgb ? MTLPixelFormatRGBA8Unorm_sRGB : MTLPixelFormatRGBA8Unorm; dataType = MTLDataTypeChar3; break;
            case 2: pixelFormat = MTLPixelFormatRG8Unorm;                                          dataType = MTLDataTypeChar2; break;
            case 1: pixelFormat = MTLPixelFormatR8Unorm;                                           dataType = MTLDataTypeChar;  break;
            default: throw Exception("Unsupported channel count in mapTextureFormatToMetal");
        }
    }
    else if (baseType == Image::BaseType::UINT16)
    {
        switch (channelCount)
        {
            case 4: pixelFormat = MTLPixelFormatRGBA16Uint; dataType = MTLDataTypeShort4; break;
            case 3: pixelFormat = MTLPixelFormatRGBA16Uint; dataType = MTLDataTypeShort3; break;
            case 2: pixelFormat = MTLPixelFormatRG16Uint;   dataType = MTLDataTypeShort2; break;
            case 1: pixelFormat = MTLPixelFormatR16Uint;    dataType = MTLDataTypeShort;  break;
            default: throw Exception("Unsupported channel count in mapTextureFormatToMetal");
        }
    }
    else if (baseType == Image::BaseType::HALF)
    {
        switch (channelCount)
        {
            case 4: pixelFormat = MTLPixelFormatRGBA16Float; dataType = MTLDataTypeHalf4; break;
            case 3: pixelFormat = MTLPixelFormatRGBA16Float; dataType = MTLDataTypeHalf3; break;
            case 2: pixelFormat = MTLPixelFormatRG16Float;   dataType = MTLDataTypeHalf2; break;
            case 1: pixelFormat = MTLPixelFormatR16Float;    dataType = MTLDataTypeHalf ; break;
            default: throw Exception("Unsupported channel count in mapTextureFormatToMetal");
        }
    }
    else if (baseType == Image::BaseType::FLOAT)
    {
        switch (channelCount)
        {
            case 4: pixelFormat = MTLPixelFormatRGBA32Float; dataType = MTLDataTypeFloat4; break;
            case 3: pixelFormat = MTLPixelFormatRGBA32Float;  dataType = MTLDataTypeFloat3; break;
            case 2: pixelFormat = MTLPixelFormatRG32Float;   dataType = MTLDataTypeFloat2; break;
            case 1: pixelFormat = MTLPixelFormatR32Float;    dataType = MTLDataTypeFloat;  break;
            default: throw Exception("Unsupported channel count in mapTextureFormatToMetal");
        }
    }
    else
    {
        throw Exception("Unsupported base type in mapTextureFormatToMetal");
    }
}

size_t MetalTextureHandler::getTextureBaseTypeSize(Image::BaseType baseType)
{
    if (baseType == Image::BaseType::UINT8)
    {
        return 1;
    }
    else if (baseType == Image::BaseType::UINT16 || baseType == Image::BaseType::HALF)
    {
        return 2;
    }
    else if (baseType == Image::BaseType::FLOAT)
    {
        return 4;
    }
    else
    {
        throw Exception("Unsupported base type in mapTextureFormatToMetal");
    }
}


MATERIALX_NAMESPACE_END
