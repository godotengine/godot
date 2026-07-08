//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLTexture.hpp
//
// Copyright 2020-2025 Apple Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#pragma once

#include "../Foundation/Foundation.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPixelFormat.hpp"
#include "MTLPrivate.hpp"
#include "MTLResource.hpp"
#include "MTLTypes.hpp"
#include <IOSurface/IOSurfaceRef.h>

namespace MTL
{
class Buffer;
class Device;
class Resource;
class SharedTextureHandle;
class Texture;
class TextureDescriptor;
class TextureViewDescriptor;
}

namespace MTL
{
_MTL_ENUM(NS::UInteger, TextureType) {
    TextureType1D = 0,
    TextureType1DArray = 1,
    TextureType2D = 2,
    TextureType2DArray = 3,
    TextureType2DMultisample = 4,
    TextureTypeCube = 5,
    TextureTypeCubeArray = 6,
    TextureType3D = 7,
    TextureType2DMultisampleArray = 8,
    TextureTypeTextureBuffer = 9,
};

_MTL_ENUM(uint8_t, TextureSwizzle) {
    TextureSwizzleZero = 0,
    TextureSwizzleOne = 1,
    TextureSwizzleRed = 2,
    TextureSwizzleGreen = 3,
    TextureSwizzleBlue = 4,
    TextureSwizzleAlpha = 5,
};

_MTL_ENUM(NS::Integer, TextureCompressionType) {
    TextureCompressionTypeLossless = 0,
    TextureCompressionTypeLossy = 1,
};

_MTL_OPTIONS(NS::UInteger, TextureUsage) {
    TextureUsageUnknown = 0,
    TextureUsageShaderRead = 1,
    TextureUsageShaderWrite = 1 << 1,
    TextureUsageRenderTarget = 1 << 2,
    TextureUsagePixelFormatView = 1 << 4,
    TextureUsageShaderAtomic = 1 << 5,
};

struct TextureSwizzleChannels
{

    TextureSwizzleChannels(MTL::TextureSwizzle r, MTL::TextureSwizzle g, MTL::TextureSwizzle b, MTL::TextureSwizzle a);

    TextureSwizzleChannels();

    static TextureSwizzleChannels Default();

    static TextureSwizzleChannels Make(MTL::TextureSwizzle r, MTL::TextureSwizzle g, MTL::TextureSwizzle b, MTL::TextureSwizzle a);

    MTL::TextureSwizzle           red;
    MTL::TextureSwizzle           green;
    MTL::TextureSwizzle           blue;
    MTL::TextureSwizzle           alpha;
} _MTL_PACKED;

class SharedTextureHandle : public NS::SecureCoding<SharedTextureHandle>
{
public:
    static SharedTextureHandle* alloc();

    Device*                     device() const;

    SharedTextureHandle*        init();

    NS::String*                 label() const;
};
class TextureDescriptor : public NS::Copying<TextureDescriptor>
{
public:
    static TextureDescriptor* alloc();

    bool                      allowGPUOptimizedContents() const;

    NS::UInteger              arrayLength() const;

    TextureCompressionType    compressionType() const;

    CPUCacheMode              cpuCacheMode() const;

    NS::UInteger              depth() const;

    HazardTrackingMode        hazardTrackingMode() const;

    NS::UInteger              height() const;

    TextureDescriptor*        init();

    NS::UInteger              mipmapLevelCount() const;

    PixelFormat               pixelFormat() const;

    SparsePageSize            placementSparsePageSize() const;

    ResourceOptions           resourceOptions() const;

    NS::UInteger              sampleCount() const;

    void                      setAllowGPUOptimizedContents(bool allowGPUOptimizedContents);

    void                      setArrayLength(NS::UInteger arrayLength);

    void                      setCompressionType(MTL::TextureCompressionType compressionType);

    void                      setCpuCacheMode(MTL::CPUCacheMode cpuCacheMode);

    void                      setDepth(NS::UInteger depth);

    void                      setHazardTrackingMode(MTL::HazardTrackingMode hazardTrackingMode);

    void                      setHeight(NS::UInteger height);

    void                      setMipmapLevelCount(NS::UInteger mipmapLevelCount);

    void                      setPixelFormat(MTL::PixelFormat pixelFormat);

    void                      setPlacementSparsePageSize(MTL::SparsePageSize placementSparsePageSize);

    void                      setResourceOptions(MTL::ResourceOptions resourceOptions);

    void                      setSampleCount(NS::UInteger sampleCount);

    void                      setStorageMode(MTL::StorageMode storageMode);

    void                      setSwizzle(MTL::TextureSwizzleChannels swizzle);

    void                      setTextureType(MTL::TextureType textureType);

    void                      setUsage(MTL::TextureUsage usage);

    void                      setWidth(NS::UInteger width);

    StorageMode               storageMode() const;

    TextureSwizzleChannels    swizzle() const;

    static TextureDescriptor* texture2DDescriptor(MTL::PixelFormat pixelFormat, NS::UInteger width, NS::UInteger height, bool mipmapped);

    static TextureDescriptor* textureBufferDescriptor(MTL::PixelFormat pixelFormat, NS::UInteger width, MTL::ResourceOptions resourceOptions, MTL::TextureUsage usage);

    static TextureDescriptor* textureCubeDescriptor(MTL::PixelFormat pixelFormat, NS::UInteger size, bool mipmapped);

    TextureType               textureType() const;

    TextureUsage              usage() const;

    NS::UInteger              width() const;
};
class TextureViewDescriptor : public NS::Copying<TextureViewDescriptor>
{
public:
    static TextureViewDescriptor* alloc();

    TextureViewDescriptor*        init();

    NS::Range                     levelRange() const;

    PixelFormat                   pixelFormat() const;

    void                          setLevelRange(NS::Range levelRange);

    void                          setPixelFormat(MTL::PixelFormat pixelFormat);

    void                          setSliceRange(NS::Range sliceRange);

    void                          setSwizzle(MTL::TextureSwizzleChannels swizzle);

    void                          setTextureType(MTL::TextureType textureType);

    NS::Range                     sliceRange() const;

    TextureSwizzleChannels        swizzle() const;

    TextureType                   textureType() const;
};
class Texture : public NS::Referencing<Texture, Resource>
{
public:
    bool                   allowGPUOptimizedContents() const;

    NS::UInteger           arrayLength() const;

    Buffer*                buffer() const;
    NS::UInteger           bufferBytesPerRow() const;

    NS::UInteger           bufferOffset() const;

    TextureCompressionType compressionType() const;

    NS::UInteger           depth() const;

    NS::UInteger           firstMipmapInTail() const;

    [[deprecated("please use isFramebufferOnly instead")]]
    bool                 framebufferOnly() const;

    void                 getBytes(void* pixelBytes, NS::UInteger bytesPerRow, NS::UInteger bytesPerImage, MTL::Region region, NS::UInteger level, NS::UInteger slice);
    void                 getBytes(void* pixelBytes, NS::UInteger bytesPerRow, MTL::Region region, NS::UInteger level);

    ResourceID           gpuResourceID() const;

    NS::UInteger         height() const;

    IOSurfaceRef         iosurface() const;
    NS::UInteger         iosurfacePlane() const;

    bool                 isFramebufferOnly() const;

    bool                 isShareable() const;

    bool                 isSparse() const;

    NS::UInteger         mipmapLevelCount() const;

    Texture*             newRemoteTextureViewForDevice(const MTL::Device* device);

    SharedTextureHandle* newSharedTextureHandle();

    Texture*             newTextureView(MTL::PixelFormat pixelFormat);
    Texture*             newTextureView(MTL::PixelFormat pixelFormat, MTL::TextureType textureType, NS::Range levelRange, NS::Range sliceRange);
    Texture*             newTextureView(const MTL::TextureViewDescriptor* descriptor);
    Texture*             newTextureView(MTL::PixelFormat pixelFormat, MTL::TextureType textureType, NS::Range levelRange, NS::Range sliceRange, MTL::TextureSwizzleChannels swizzle);

    NS::UInteger         parentRelativeLevel() const;

    NS::UInteger         parentRelativeSlice() const;

    Texture*             parentTexture() const;

    PixelFormat          pixelFormat() const;

    Texture*             remoteStorageTexture() const;

    void                 replaceRegion(MTL::Region region, NS::UInteger level, NS::UInteger slice, const void* pixelBytes, NS::UInteger bytesPerRow, NS::UInteger bytesPerImage);
    void                 replaceRegion(MTL::Region region, NS::UInteger level, const void* pixelBytes, NS::UInteger bytesPerRow);

    Resource*            rootResource() const;

    NS::UInteger         sampleCount() const;

    [[deprecated("please use isShareable instead")]]
    bool                   shareable() const;

    TextureSparseTier      sparseTextureTier() const;

    TextureSwizzleChannels swizzle() const;

    NS::UInteger           tailSizeInBytes() const;

    TextureType            textureType() const;

    TextureUsage           usage() const;

    NS::UInteger           width() const;
};

}
_MTL_INLINE MTL::TextureSwizzleChannels::TextureSwizzleChannels(MTL::TextureSwizzle r, MTL::TextureSwizzle g, MTL::TextureSwizzle b, MTL::TextureSwizzle a)
    : red(r)
    , green(g)
    , blue(b)
    , alpha(a)
{
}

_MTL_INLINE MTL::TextureSwizzleChannels::TextureSwizzleChannels()
    : red(MTL::TextureSwizzleRed)
    , green(MTL::TextureSwizzleGreen)
    , blue(MTL::TextureSwizzleBlue)
    , alpha(MTL::TextureSwizzleAlpha)
{
}

_MTL_INLINE MTL::TextureSwizzleChannels MTL::TextureSwizzleChannels::Default()
{
    return MTL::TextureSwizzleChannels();
}

_MTL_INLINE MTL::TextureSwizzleChannels MTL::TextureSwizzleChannels::Make(MTL::TextureSwizzle r, MTL::TextureSwizzle g, MTL::TextureSwizzle b, MTL::TextureSwizzle a)
{
    return TextureSwizzleChannels(r, g, b, a);
}

_MTL_INLINE MTL::SharedTextureHandle* MTL::SharedTextureHandle::alloc()
{
    return NS::Object::alloc<MTL::SharedTextureHandle>(_MTL_PRIVATE_CLS(MTLSharedTextureHandle));
}

_MTL_INLINE MTL::Device* MTL::SharedTextureHandle::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE MTL::SharedTextureHandle* MTL::SharedTextureHandle::init()
{
    return NS::Object::init<MTL::SharedTextureHandle>();
}

_MTL_INLINE NS::String* MTL::SharedTextureHandle::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE MTL::TextureDescriptor* MTL::TextureDescriptor::alloc()
{
    return NS::Object::alloc<MTL::TextureDescriptor>(_MTL_PRIVATE_CLS(MTLTextureDescriptor));
}

_MTL_INLINE bool MTL::TextureDescriptor::allowGPUOptimizedContents() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(allowGPUOptimizedContents));
}

_MTL_INLINE NS::UInteger MTL::TextureDescriptor::arrayLength() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(arrayLength));
}

_MTL_INLINE MTL::TextureCompressionType MTL::TextureDescriptor::compressionType() const
{
    return Object::sendMessage<MTL::TextureCompressionType>(this, _MTL_PRIVATE_SEL(compressionType));
}

_MTL_INLINE MTL::CPUCacheMode MTL::TextureDescriptor::cpuCacheMode() const
{
    return Object::sendMessage<MTL::CPUCacheMode>(this, _MTL_PRIVATE_SEL(cpuCacheMode));
}

_MTL_INLINE NS::UInteger MTL::TextureDescriptor::depth() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(depth));
}

_MTL_INLINE MTL::HazardTrackingMode MTL::TextureDescriptor::hazardTrackingMode() const
{
    return Object::sendMessage<MTL::HazardTrackingMode>(this, _MTL_PRIVATE_SEL(hazardTrackingMode));
}

_MTL_INLINE NS::UInteger MTL::TextureDescriptor::height() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(height));
}

_MTL_INLINE MTL::TextureDescriptor* MTL::TextureDescriptor::init()
{
    return NS::Object::init<MTL::TextureDescriptor>();
}

_MTL_INLINE NS::UInteger MTL::TextureDescriptor::mipmapLevelCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(mipmapLevelCount));
}

_MTL_INLINE MTL::PixelFormat MTL::TextureDescriptor::pixelFormat() const
{
    return Object::sendMessage<MTL::PixelFormat>(this, _MTL_PRIVATE_SEL(pixelFormat));
}

_MTL_INLINE MTL::SparsePageSize MTL::TextureDescriptor::placementSparsePageSize() const
{
    return Object::sendMessage<MTL::SparsePageSize>(this, _MTL_PRIVATE_SEL(placementSparsePageSize));
}

_MTL_INLINE MTL::ResourceOptions MTL::TextureDescriptor::resourceOptions() const
{
    return Object::sendMessage<MTL::ResourceOptions>(this, _MTL_PRIVATE_SEL(resourceOptions));
}

_MTL_INLINE NS::UInteger MTL::TextureDescriptor::sampleCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(sampleCount));
}

_MTL_INLINE void MTL::TextureDescriptor::setAllowGPUOptimizedContents(bool allowGPUOptimizedContents)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setAllowGPUOptimizedContents_), allowGPUOptimizedContents);
}

_MTL_INLINE void MTL::TextureDescriptor::setArrayLength(NS::UInteger arrayLength)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setArrayLength_), arrayLength);
}

_MTL_INLINE void MTL::TextureDescriptor::setCompressionType(MTL::TextureCompressionType compressionType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCompressionType_), compressionType);
}

_MTL_INLINE void MTL::TextureDescriptor::setCpuCacheMode(MTL::CPUCacheMode cpuCacheMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCpuCacheMode_), cpuCacheMode);
}

_MTL_INLINE void MTL::TextureDescriptor::setDepth(NS::UInteger depth)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepth_), depth);
}

_MTL_INLINE void MTL::TextureDescriptor::setHazardTrackingMode(MTL::HazardTrackingMode hazardTrackingMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setHazardTrackingMode_), hazardTrackingMode);
}

_MTL_INLINE void MTL::TextureDescriptor::setHeight(NS::UInteger height)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setHeight_), height);
}

_MTL_INLINE void MTL::TextureDescriptor::setMipmapLevelCount(NS::UInteger mipmapLevelCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMipmapLevelCount_), mipmapLevelCount);
}

_MTL_INLINE void MTL::TextureDescriptor::setPixelFormat(MTL::PixelFormat pixelFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPixelFormat_), pixelFormat);
}

_MTL_INLINE void MTL::TextureDescriptor::setPlacementSparsePageSize(MTL::SparsePageSize placementSparsePageSize)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPlacementSparsePageSize_), placementSparsePageSize);
}

_MTL_INLINE void MTL::TextureDescriptor::setResourceOptions(MTL::ResourceOptions resourceOptions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setResourceOptions_), resourceOptions);
}

_MTL_INLINE void MTL::TextureDescriptor::setSampleCount(NS::UInteger sampleCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSampleCount_), sampleCount);
}

_MTL_INLINE void MTL::TextureDescriptor::setStorageMode(MTL::StorageMode storageMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStorageMode_), storageMode);
}

_MTL_INLINE void MTL::TextureDescriptor::setSwizzle(MTL::TextureSwizzleChannels swizzle)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSwizzle_), swizzle);
}

_MTL_INLINE void MTL::TextureDescriptor::setTextureType(MTL::TextureType textureType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTextureType_), textureType);
}

_MTL_INLINE void MTL::TextureDescriptor::setUsage(MTL::TextureUsage usage)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setUsage_), usage);
}

_MTL_INLINE void MTL::TextureDescriptor::setWidth(NS::UInteger width)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setWidth_), width);
}

_MTL_INLINE MTL::StorageMode MTL::TextureDescriptor::storageMode() const
{
    return Object::sendMessage<MTL::StorageMode>(this, _MTL_PRIVATE_SEL(storageMode));
}

_MTL_INLINE MTL::TextureSwizzleChannels MTL::TextureDescriptor::swizzle() const
{
    return Object::sendMessage<MTL::TextureSwizzleChannels>(this, _MTL_PRIVATE_SEL(swizzle));
}

_MTL_INLINE MTL::TextureDescriptor* MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormat pixelFormat, NS::UInteger width, NS::UInteger height, bool mipmapped)
{
    return Object::sendMessage<MTL::TextureDescriptor*>(_MTL_PRIVATE_CLS(MTLTextureDescriptor), _MTL_PRIVATE_SEL(texture2DDescriptorWithPixelFormat_width_height_mipmapped_), pixelFormat, width, height, mipmapped);
}

_MTL_INLINE MTL::TextureDescriptor* MTL::TextureDescriptor::textureBufferDescriptor(MTL::PixelFormat pixelFormat, NS::UInteger width, MTL::ResourceOptions resourceOptions, MTL::TextureUsage usage)
{
    return Object::sendMessage<MTL::TextureDescriptor*>(_MTL_PRIVATE_CLS(MTLTextureDescriptor), _MTL_PRIVATE_SEL(textureBufferDescriptorWithPixelFormat_width_resourceOptions_usage_), pixelFormat, width, resourceOptions, usage);
}

_MTL_INLINE MTL::TextureDescriptor* MTL::TextureDescriptor::textureCubeDescriptor(MTL::PixelFormat pixelFormat, NS::UInteger size, bool mipmapped)
{
    return Object::sendMessage<MTL::TextureDescriptor*>(_MTL_PRIVATE_CLS(MTLTextureDescriptor), _MTL_PRIVATE_SEL(textureCubeDescriptorWithPixelFormat_size_mipmapped_), pixelFormat, size, mipmapped);
}

_MTL_INLINE MTL::TextureType MTL::TextureDescriptor::textureType() const
{
    return Object::sendMessage<MTL::TextureType>(this, _MTL_PRIVATE_SEL(textureType));
}

_MTL_INLINE MTL::TextureUsage MTL::TextureDescriptor::usage() const
{
    return Object::sendMessage<MTL::TextureUsage>(this, _MTL_PRIVATE_SEL(usage));
}

_MTL_INLINE NS::UInteger MTL::TextureDescriptor::width() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(width));
}

_MTL_INLINE MTL::TextureViewDescriptor* MTL::TextureViewDescriptor::alloc()
{
    return NS::Object::alloc<MTL::TextureViewDescriptor>(_MTL_PRIVATE_CLS(MTLTextureViewDescriptor));
}

_MTL_INLINE MTL::TextureViewDescriptor* MTL::TextureViewDescriptor::init()
{
    return NS::Object::init<MTL::TextureViewDescriptor>();
}

_MTL_INLINE NS::Range MTL::TextureViewDescriptor::levelRange() const
{
    return Object::sendMessage<NS::Range>(this, _MTL_PRIVATE_SEL(levelRange));
}

_MTL_INLINE MTL::PixelFormat MTL::TextureViewDescriptor::pixelFormat() const
{
    return Object::sendMessage<MTL::PixelFormat>(this, _MTL_PRIVATE_SEL(pixelFormat));
}

_MTL_INLINE void MTL::TextureViewDescriptor::setLevelRange(NS::Range levelRange)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLevelRange_), levelRange);
}

_MTL_INLINE void MTL::TextureViewDescriptor::setPixelFormat(MTL::PixelFormat pixelFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPixelFormat_), pixelFormat);
}

_MTL_INLINE void MTL::TextureViewDescriptor::setSliceRange(NS::Range sliceRange)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSliceRange_), sliceRange);
}

_MTL_INLINE void MTL::TextureViewDescriptor::setSwizzle(MTL::TextureSwizzleChannels swizzle)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSwizzle_), swizzle);
}

_MTL_INLINE void MTL::TextureViewDescriptor::setTextureType(MTL::TextureType textureType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTextureType_), textureType);
}

_MTL_INLINE NS::Range MTL::TextureViewDescriptor::sliceRange() const
{
    return Object::sendMessage<NS::Range>(this, _MTL_PRIVATE_SEL(sliceRange));
}

_MTL_INLINE MTL::TextureSwizzleChannels MTL::TextureViewDescriptor::swizzle() const
{
    return Object::sendMessage<MTL::TextureSwizzleChannels>(this, _MTL_PRIVATE_SEL(swizzle));
}

_MTL_INLINE MTL::TextureType MTL::TextureViewDescriptor::textureType() const
{
    return Object::sendMessage<MTL::TextureType>(this, _MTL_PRIVATE_SEL(textureType));
}

_MTL_INLINE bool MTL::Texture::allowGPUOptimizedContents() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(allowGPUOptimizedContents));
}

_MTL_INLINE NS::UInteger MTL::Texture::arrayLength() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(arrayLength));
}

_MTL_INLINE MTL::Buffer* MTL::Texture::buffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(buffer));
}

_MTL_INLINE NS::UInteger MTL::Texture::bufferBytesPerRow() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(bufferBytesPerRow));
}

_MTL_INLINE NS::UInteger MTL::Texture::bufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(bufferOffset));
}

_MTL_INLINE MTL::TextureCompressionType MTL::Texture::compressionType() const
{
    return Object::sendMessage<MTL::TextureCompressionType>(this, _MTL_PRIVATE_SEL(compressionType));
}

_MTL_INLINE NS::UInteger MTL::Texture::depth() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(depth));
}

_MTL_INLINE NS::UInteger MTL::Texture::firstMipmapInTail() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(firstMipmapInTail));
}

_MTL_INLINE bool MTL::Texture::framebufferOnly() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isFramebufferOnly));
}

_MTL_INLINE void MTL::Texture::getBytes(void* pixelBytes, NS::UInteger bytesPerRow, NS::UInteger bytesPerImage, MTL::Region region, NS::UInteger level, NS::UInteger slice)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(getBytes_bytesPerRow_bytesPerImage_fromRegion_mipmapLevel_slice_), pixelBytes, bytesPerRow, bytesPerImage, region, level, slice);
}

_MTL_INLINE void MTL::Texture::getBytes(void* pixelBytes, NS::UInteger bytesPerRow, MTL::Region region, NS::UInteger level)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(getBytes_bytesPerRow_fromRegion_mipmapLevel_), pixelBytes, bytesPerRow, region, level);
}

_MTL_INLINE MTL::ResourceID MTL::Texture::gpuResourceID() const
{
    return Object::sendMessage<MTL::ResourceID>(this, _MTL_PRIVATE_SEL(gpuResourceID));
}

_MTL_INLINE NS::UInteger MTL::Texture::height() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(height));
}

_MTL_INLINE IOSurfaceRef MTL::Texture::iosurface() const
{
    return Object::sendMessage<IOSurfaceRef>(this, _MTL_PRIVATE_SEL(iosurface));
}

_MTL_INLINE NS::UInteger MTL::Texture::iosurfacePlane() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(iosurfacePlane));
}

_MTL_INLINE bool MTL::Texture::isFramebufferOnly() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isFramebufferOnly));
}

_MTL_INLINE bool MTL::Texture::isShareable() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isShareable));
}

_MTL_INLINE bool MTL::Texture::isSparse() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isSparse));
}

_MTL_INLINE NS::UInteger MTL::Texture::mipmapLevelCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(mipmapLevelCount));
}

_MTL_INLINE MTL::Texture* MTL::Texture::newRemoteTextureViewForDevice(const MTL::Device* device)
{
    return Object::sendMessage<MTL::Texture*>(this, _MTL_PRIVATE_SEL(newRemoteTextureViewForDevice_), device);
}

_MTL_INLINE MTL::SharedTextureHandle* MTL::Texture::newSharedTextureHandle()
{
    return Object::sendMessage<MTL::SharedTextureHandle*>(this, _MTL_PRIVATE_SEL(newSharedTextureHandle));
}

_MTL_INLINE MTL::Texture* MTL::Texture::newTextureView(MTL::PixelFormat pixelFormat)
{
    return Object::sendMessage<MTL::Texture*>(this, _MTL_PRIVATE_SEL(newTextureViewWithPixelFormat_), pixelFormat);
}

_MTL_INLINE MTL::Texture* MTL::Texture::newTextureView(MTL::PixelFormat pixelFormat, MTL::TextureType textureType, NS::Range levelRange, NS::Range sliceRange)
{
    return Object::sendMessage<MTL::Texture*>(this, _MTL_PRIVATE_SEL(newTextureViewWithPixelFormat_textureType_levels_slices_), pixelFormat, textureType, levelRange, sliceRange);
}

_MTL_INLINE MTL::Texture* MTL::Texture::newTextureView(const MTL::TextureViewDescriptor* descriptor)
{
    return Object::sendMessage<MTL::Texture*>(this, _MTL_PRIVATE_SEL(newTextureViewWithDescriptor_), descriptor);
}

_MTL_INLINE MTL::Texture* MTL::Texture::newTextureView(MTL::PixelFormat pixelFormat, MTL::TextureType textureType, NS::Range levelRange, NS::Range sliceRange, MTL::TextureSwizzleChannels swizzle)
{
    return Object::sendMessage<MTL::Texture*>(this, _MTL_PRIVATE_SEL(newTextureViewWithPixelFormat_textureType_levels_slices_swizzle_), pixelFormat, textureType, levelRange, sliceRange, swizzle);
}

_MTL_INLINE NS::UInteger MTL::Texture::parentRelativeLevel() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(parentRelativeLevel));
}

_MTL_INLINE NS::UInteger MTL::Texture::parentRelativeSlice() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(parentRelativeSlice));
}

_MTL_INLINE MTL::Texture* MTL::Texture::parentTexture() const
{
    return Object::sendMessage<MTL::Texture*>(this, _MTL_PRIVATE_SEL(parentTexture));
}

_MTL_INLINE MTL::PixelFormat MTL::Texture::pixelFormat() const
{
    return Object::sendMessage<MTL::PixelFormat>(this, _MTL_PRIVATE_SEL(pixelFormat));
}

_MTL_INLINE MTL::Texture* MTL::Texture::remoteStorageTexture() const
{
    return Object::sendMessage<MTL::Texture*>(this, _MTL_PRIVATE_SEL(remoteStorageTexture));
}

_MTL_INLINE void MTL::Texture::replaceRegion(MTL::Region region, NS::UInteger level, NS::UInteger slice, const void* pixelBytes, NS::UInteger bytesPerRow, NS::UInteger bytesPerImage)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage_), region, level, slice, pixelBytes, bytesPerRow, bytesPerImage);
}

_MTL_INLINE void MTL::Texture::replaceRegion(MTL::Region region, NS::UInteger level, const void* pixelBytes, NS::UInteger bytesPerRow)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(replaceRegion_mipmapLevel_withBytes_bytesPerRow_), region, level, pixelBytes, bytesPerRow);
}

_MTL_INLINE MTL::Resource* MTL::Texture::rootResource() const
{
    return Object::sendMessage<MTL::Resource*>(this, _MTL_PRIVATE_SEL(rootResource));
}

_MTL_INLINE NS::UInteger MTL::Texture::sampleCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(sampleCount));
}

_MTL_INLINE bool MTL::Texture::shareable() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isShareable));
}

_MTL_INLINE MTL::TextureSparseTier MTL::Texture::sparseTextureTier() const
{
    return Object::sendMessage<MTL::TextureSparseTier>(this, _MTL_PRIVATE_SEL(sparseTextureTier));
}

_MTL_INLINE MTL::TextureSwizzleChannels MTL::Texture::swizzle() const
{
    return Object::sendMessage<MTL::TextureSwizzleChannels>(this, _MTL_PRIVATE_SEL(swizzle));
}

_MTL_INLINE NS::UInteger MTL::Texture::tailSizeInBytes() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(tailSizeInBytes));
}

_MTL_INLINE MTL::TextureType MTL::Texture::textureType() const
{
    return Object::sendMessage<MTL::TextureType>(this, _MTL_PRIVATE_SEL(textureType));
}

_MTL_INLINE MTL::TextureUsage MTL::Texture::usage() const
{
    return Object::sendMessage<MTL::TextureUsage>(this, _MTL_PRIVATE_SEL(usage));
}

_MTL_INLINE NS::UInteger MTL::Texture::width() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(width));
}
