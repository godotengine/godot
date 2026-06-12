#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTLResource.hpp"
#include <IOSurface/IOSurfaceRef.h>

namespace MTL {
    class Buffer;
    class Device;
    class Resource;
    enum CPUCacheMode : NS::UInteger;
    enum HazardTrackingMode : NS::UInteger;
    enum PixelFormat : NS::UInteger;
    using ResourceOptions = NS::UInteger;
    enum SparsePageSize : NS::Integer;
    enum StorageMode : NS::UInteger;
    enum TextureSparseTier : NS::Integer;
}
namespace NS {
    class String;
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

_MTL_OPTIONS(NS::UInteger, TextureUsage) {
    TextureUsageUnknown = 0x0000,
    TextureUsageShaderRead = 0x0001,
    TextureUsageShaderWrite = 0x0002,
    TextureUsageRenderTarget = 0x0004,
    TextureUsagePixelFormatView = 0x0010,
    TextureUsageShaderAtomic = 0x0020,
};

_MTL_ENUM(NS::Integer, TextureCompressionType) {
    TextureCompressionTypeLossless = 0,
    TextureCompressionTypeLossy = 1,
};


class SharedTextureHandle;
class TextureDescriptor;
class TextureViewDescriptor;
class Texture;

class SharedTextureHandle : public NS::SecureCoding<SharedTextureHandle>
{
public:
    static SharedTextureHandle* alloc();
    SharedTextureHandle*        init() const;

    MTL::Device* device() const;
    NS::String*  label() const;

};

class TextureDescriptor : public NS::Copying<TextureDescriptor>
{
public:
    static TextureDescriptor* alloc();
    TextureDescriptor*        init() const;

    static MTL::TextureDescriptor* texture2DDescriptor(MTL::PixelFormat pixelFormat, NS::UInteger width, NS::UInteger height, bool mipmapped);
    static MTL::TextureDescriptor* textureBufferDescriptor(MTL::PixelFormat pixelFormat, NS::UInteger width, MTL::ResourceOptions resourceOptions, MTL::TextureUsage usage);
    static MTL::TextureDescriptor* textureCubeDescriptor(MTL::PixelFormat pixelFormat, NS::UInteger size, bool mipmapped);

    bool                        allowGPUOptimizedContents() const;
    NS::UInteger                arrayLength() const;
    MTL::TextureCompressionType compressionType() const;
    MTL::CPUCacheMode           cpuCacheMode() const;
    NS::UInteger                depth() const;
    MTL::HazardTrackingMode     hazardTrackingMode() const;
    NS::UInteger                height() const;
    NS::UInteger                mipmapLevelCount() const;
    MTL::PixelFormat            pixelFormat() const;
    MTL::SparsePageSize         placementSparsePageSize() const;
    MTL::ResourceOptions        resourceOptions() const;
    NS::UInteger                sampleCount() const;
    void                        setAllowGPUOptimizedContents(bool allowGPUOptimizedContents);
    void                        setArrayLength(NS::UInteger arrayLength);
    void                        setCompressionType(MTL::TextureCompressionType compressionType);
    void                        setCpuCacheMode(MTL::CPUCacheMode cpuCacheMode);
    void                        setDepth(NS::UInteger depth);
    void                        setHazardTrackingMode(MTL::HazardTrackingMode hazardTrackingMode);
    void                        setHeight(NS::UInteger height);
    void                        setMipmapLevelCount(NS::UInteger mipmapLevelCount);
    void                        setPixelFormat(MTL::PixelFormat pixelFormat);
    void                        setPlacementSparsePageSize(MTL::SparsePageSize placementSparsePageSize);
    void                        setResourceOptions(MTL::ResourceOptions resourceOptions);
    void                        setSampleCount(NS::UInteger sampleCount);
    void                        setStorageMode(MTL::StorageMode storageMode);
    void                        setSwizzle(MTL::TextureSwizzleChannels swizzle);
    void                        setTextureType(MTL::TextureType textureType);
    void                        setUsage(MTL::TextureUsage usage);
    void                        setWidth(NS::UInteger width);
    MTL::StorageMode            storageMode() const;
    MTL::TextureSwizzleChannels swizzle() const;
    MTL::TextureType            textureType() const;
    MTL::TextureUsage           usage() const;
    NS::UInteger                width() const;

};

class TextureViewDescriptor : public NS::Copying<TextureViewDescriptor>
{
public:
    static TextureViewDescriptor* alloc();
    TextureViewDescriptor*        init() const;

    NS::Range                   levelRange() const;
    MTL::PixelFormat            pixelFormat() const;
    void                        setLevelRange(NS::Range levelRange);
    void                        setPixelFormat(MTL::PixelFormat pixelFormat);
    void                        setSliceRange(NS::Range sliceRange);
    void                        setSwizzle(MTL::TextureSwizzleChannels swizzle);
    void                        setTextureType(MTL::TextureType textureType);
    NS::Range                   sliceRange() const;
    MTL::TextureSwizzleChannels swizzle() const;
    MTL::TextureType            textureType() const;

};

class Texture : public NS::Referencing<Texture, MTL::Resource>
{
public:
    bool                        allowGPUOptimizedContents() const;
    NS::UInteger                arrayLength() const;
    MTL::Buffer*                buffer() const;
    NS::UInteger                bufferBytesPerRow() const;
    NS::UInteger                bufferOffset() const;
    MTL::TextureCompressionType compressionType() const;
    NS::UInteger                depth() const;
    NS::UInteger                firstMipmapInTail() const;
    bool                        framebufferOnly() const;
    void                        getBytes(void * pixelBytes, NS::UInteger bytesPerRow, NS::UInteger bytesPerImage, MTL::Region region, NS::UInteger level, NS::UInteger slice);
    void                        getBytes(void * pixelBytes, NS::UInteger bytesPerRow, MTL::Region region, NS::UInteger level);
    MTL::ResourceID             gpuResourceID() const;
    NS::UInteger                height() const;
    IOSurfaceRef                iosurface() const;
    NS::UInteger                iosurfacePlane() const;
    bool                        isFramebufferOnly();
    bool                        isShareable();
    bool                        isSparse() const;
    NS::UInteger                mipmapLevelCount() const;
    MTL::Texture*               newRemoteTextureView(MTL::Device* device);
    MTL::SharedTextureHandle*   newSharedTextureHandle();
    MTL::Texture*               newTextureView(MTL::PixelFormat pixelFormat);
    MTL::Texture*               newTextureView(MTL::PixelFormat pixelFormat, MTL::TextureType textureType, NS::Range levelRange, NS::Range sliceRange);
    MTL::Texture*               newTextureView(MTL::TextureViewDescriptor* descriptor);
    MTL::Texture*               newTextureView(MTL::PixelFormat pixelFormat, MTL::TextureType textureType, NS::Range levelRange, NS::Range sliceRange, MTL::TextureSwizzleChannels swizzle);
    NS::UInteger                parentRelativeLevel() const;
    NS::UInteger                parentRelativeSlice() const;
    MTL::Texture*               parentTexture() const;
    MTL::PixelFormat            pixelFormat() const;
    MTL::Texture*               remoteStorageTexture() const;
    void                        replace(MTL::Region region, NS::UInteger level, NS::UInteger slice, const void * pixelBytes, NS::UInteger bytesPerRow, NS::UInteger bytesPerImage);
    void                        replace(MTL::Region region, NS::UInteger level, const void * pixelBytes, NS::UInteger bytesPerRow);
    MTL::Resource*              rootResource() const;
    NS::UInteger                sampleCount() const;
    bool                        shareable() const;
    MTL::TextureSparseTier      sparseTextureTier() const;
    MTL::TextureSwizzleChannels swizzle() const;
    NS::UInteger                tailSizeInBytes() const;
    MTL::TextureType            textureType() const;
    MTL::TextureUsage           usage() const;
    NS::UInteger                width() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLSharedTextureHandle;
extern "C" void *OBJC_CLASS_$_MTLTextureDescriptor;
extern "C" void *OBJC_CLASS_$_MTLTextureViewDescriptor;
extern "C" void *OBJC_CLASS_$_MTLTexture;

_MTL_INLINE MTL::SharedTextureHandle* MTL::SharedTextureHandle::alloc()
{
    return _MTL_msg_MTL__SharedTextureHandlep_alloc((const void*)&OBJC_CLASS_$_MTLSharedTextureHandle, nullptr);
}

_MTL_INLINE MTL::SharedTextureHandle* MTL::SharedTextureHandle::init() const
{
    return _MTL_msg_MTL__SharedTextureHandlep_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::Device* MTL::SharedTextureHandle::device() const
{
    return _MTL_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::SharedTextureHandle::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE MTL::TextureDescriptor* MTL::TextureDescriptor::alloc()
{
    return _MTL_msg_MTL__TextureDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLTextureDescriptor, nullptr);
}

_MTL_INLINE MTL::TextureDescriptor* MTL::TextureDescriptor::init() const
{
    return _MTL_msg_MTL__TextureDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::TextureDescriptor* MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormat pixelFormat, NS::UInteger width, NS::UInteger height, bool mipmapped)
{
    return _MTL_msg_MTL__TextureDescriptorp_texture2DDescriptorWithPixelFormat_width_height_mipmapped__MTL__PixelFormat_NS__UInteger_NS__UInteger_bool((const void*)&OBJC_CLASS_$_MTLTextureDescriptor, nullptr, pixelFormat, width, height, mipmapped);
}

_MTL_INLINE MTL::TextureDescriptor* MTL::TextureDescriptor::textureCubeDescriptor(MTL::PixelFormat pixelFormat, NS::UInteger size, bool mipmapped)
{
    return _MTL_msg_MTL__TextureDescriptorp_textureCubeDescriptorWithPixelFormat_size_mipmapped__MTL__PixelFormat_NS__UInteger_bool((const void*)&OBJC_CLASS_$_MTLTextureDescriptor, nullptr, pixelFormat, size, mipmapped);
}

_MTL_INLINE MTL::TextureDescriptor* MTL::TextureDescriptor::textureBufferDescriptor(MTL::PixelFormat pixelFormat, NS::UInteger width, MTL::ResourceOptions resourceOptions, MTL::TextureUsage usage)
{
    return _MTL_msg_MTL__TextureDescriptorp_textureBufferDescriptorWithPixelFormat_width_resourceOptions_usage__MTL__PixelFormat_NS__UInteger_MTL__ResourceOptions_MTL__TextureUsage((const void*)&OBJC_CLASS_$_MTLTextureDescriptor, nullptr, pixelFormat, width, resourceOptions, usage);
}

_MTL_INLINE MTL::TextureType MTL::TextureDescriptor::textureType() const
{
    return _MTL_msg_MTL__TextureType_textureType((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TextureDescriptor::setTextureType(MTL::TextureType textureType)
{
    _MTL_msg_v_setTextureType__MTL__TextureType((const void*)this, nullptr, textureType);
}

_MTL_INLINE MTL::PixelFormat MTL::TextureDescriptor::pixelFormat() const
{
    return _MTL_msg_MTL__PixelFormat_pixelFormat((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TextureDescriptor::setPixelFormat(MTL::PixelFormat pixelFormat)
{
    _MTL_msg_v_setPixelFormat__MTL__PixelFormat((const void*)this, nullptr, pixelFormat);
}

_MTL_INLINE NS::UInteger MTL::TextureDescriptor::width() const
{
    return _MTL_msg_NS__UInteger_width((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TextureDescriptor::setWidth(NS::UInteger width)
{
    _MTL_msg_v_setWidth__NS__UInteger((const void*)this, nullptr, width);
}

_MTL_INLINE NS::UInteger MTL::TextureDescriptor::height() const
{
    return _MTL_msg_NS__UInteger_height((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TextureDescriptor::setHeight(NS::UInteger height)
{
    _MTL_msg_v_setHeight__NS__UInteger((const void*)this, nullptr, height);
}

_MTL_INLINE NS::UInteger MTL::TextureDescriptor::depth() const
{
    return _MTL_msg_NS__UInteger_depth((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TextureDescriptor::setDepth(NS::UInteger depth)
{
    _MTL_msg_v_setDepth__NS__UInteger((const void*)this, nullptr, depth);
}

_MTL_INLINE NS::UInteger MTL::TextureDescriptor::mipmapLevelCount() const
{
    return _MTL_msg_NS__UInteger_mipmapLevelCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TextureDescriptor::setMipmapLevelCount(NS::UInteger mipmapLevelCount)
{
    _MTL_msg_v_setMipmapLevelCount__NS__UInteger((const void*)this, nullptr, mipmapLevelCount);
}

_MTL_INLINE NS::UInteger MTL::TextureDescriptor::sampleCount() const
{
    return _MTL_msg_NS__UInteger_sampleCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TextureDescriptor::setSampleCount(NS::UInteger sampleCount)
{
    _MTL_msg_v_setSampleCount__NS__UInteger((const void*)this, nullptr, sampleCount);
}

_MTL_INLINE NS::UInteger MTL::TextureDescriptor::arrayLength() const
{
    return _MTL_msg_NS__UInteger_arrayLength((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TextureDescriptor::setArrayLength(NS::UInteger arrayLength)
{
    _MTL_msg_v_setArrayLength__NS__UInteger((const void*)this, nullptr, arrayLength);
}

_MTL_INLINE MTL::ResourceOptions MTL::TextureDescriptor::resourceOptions() const
{
    return _MTL_msg_MTL__ResourceOptions_resourceOptions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TextureDescriptor::setResourceOptions(MTL::ResourceOptions resourceOptions)
{
    _MTL_msg_v_setResourceOptions__MTL__ResourceOptions((const void*)this, nullptr, resourceOptions);
}

_MTL_INLINE MTL::CPUCacheMode MTL::TextureDescriptor::cpuCacheMode() const
{
    return _MTL_msg_MTL__CPUCacheMode_cpuCacheMode((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TextureDescriptor::setCpuCacheMode(MTL::CPUCacheMode cpuCacheMode)
{
    _MTL_msg_v_setCpuCacheMode__MTL__CPUCacheMode((const void*)this, nullptr, cpuCacheMode);
}

_MTL_INLINE MTL::StorageMode MTL::TextureDescriptor::storageMode() const
{
    return _MTL_msg_MTL__StorageMode_storageMode((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TextureDescriptor::setStorageMode(MTL::StorageMode storageMode)
{
    _MTL_msg_v_setStorageMode__MTL__StorageMode((const void*)this, nullptr, storageMode);
}

_MTL_INLINE MTL::HazardTrackingMode MTL::TextureDescriptor::hazardTrackingMode() const
{
    return _MTL_msg_MTL__HazardTrackingMode_hazardTrackingMode((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TextureDescriptor::setHazardTrackingMode(MTL::HazardTrackingMode hazardTrackingMode)
{
    _MTL_msg_v_setHazardTrackingMode__MTL__HazardTrackingMode((const void*)this, nullptr, hazardTrackingMode);
}

_MTL_INLINE MTL::TextureUsage MTL::TextureDescriptor::usage() const
{
    return _MTL_msg_MTL__TextureUsage_usage((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TextureDescriptor::setUsage(MTL::TextureUsage usage)
{
    _MTL_msg_v_setUsage__MTL__TextureUsage((const void*)this, nullptr, usage);
}

_MTL_INLINE bool MTL::TextureDescriptor::allowGPUOptimizedContents() const
{
    return _MTL_msg_bool_allowGPUOptimizedContents((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TextureDescriptor::setAllowGPUOptimizedContents(bool allowGPUOptimizedContents)
{
    _MTL_msg_v_setAllowGPUOptimizedContents__bool((const void*)this, nullptr, allowGPUOptimizedContents);
}

_MTL_INLINE MTL::TextureCompressionType MTL::TextureDescriptor::compressionType() const
{
    return _MTL_msg_MTL__TextureCompressionType_compressionType((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TextureDescriptor::setCompressionType(MTL::TextureCompressionType compressionType)
{
    _MTL_msg_v_setCompressionType__MTL__TextureCompressionType((const void*)this, nullptr, compressionType);
}

_MTL_INLINE MTL::TextureSwizzleChannels MTL::TextureDescriptor::swizzle() const
{
    return _MTL_msg_MTL__TextureSwizzleChannels_swizzle((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TextureDescriptor::setSwizzle(MTL::TextureSwizzleChannels swizzle)
{
    _MTL_msg_v_setSwizzle__MTL__TextureSwizzleChannels((const void*)this, nullptr, swizzle);
}

_MTL_INLINE MTL::SparsePageSize MTL::TextureDescriptor::placementSparsePageSize() const
{
    return _MTL_msg_MTL__SparsePageSize_placementSparsePageSize((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TextureDescriptor::setPlacementSparsePageSize(MTL::SparsePageSize placementSparsePageSize)
{
    _MTL_msg_v_setPlacementSparsePageSize__MTL__SparsePageSize((const void*)this, nullptr, placementSparsePageSize);
}

_MTL_INLINE MTL::TextureViewDescriptor* MTL::TextureViewDescriptor::alloc()
{
    return _MTL_msg_MTL__TextureViewDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLTextureViewDescriptor, nullptr);
}

_MTL_INLINE MTL::TextureViewDescriptor* MTL::TextureViewDescriptor::init() const
{
    return _MTL_msg_MTL__TextureViewDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::PixelFormat MTL::TextureViewDescriptor::pixelFormat() const
{
    return _MTL_msg_MTL__PixelFormat_pixelFormat((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TextureViewDescriptor::setPixelFormat(MTL::PixelFormat pixelFormat)
{
    _MTL_msg_v_setPixelFormat__MTL__PixelFormat((const void*)this, nullptr, pixelFormat);
}

_MTL_INLINE MTL::TextureType MTL::TextureViewDescriptor::textureType() const
{
    return _MTL_msg_MTL__TextureType_textureType((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TextureViewDescriptor::setTextureType(MTL::TextureType textureType)
{
    _MTL_msg_v_setTextureType__MTL__TextureType((const void*)this, nullptr, textureType);
}

_MTL_INLINE NS::Range MTL::TextureViewDescriptor::levelRange() const
{
    return _MTL_msg_NS__Range_levelRange((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TextureViewDescriptor::setLevelRange(NS::Range levelRange)
{
    _MTL_msg_v_setLevelRange__NS__Range((const void*)this, nullptr, levelRange);
}

_MTL_INLINE NS::Range MTL::TextureViewDescriptor::sliceRange() const
{
    return _MTL_msg_NS__Range_sliceRange((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TextureViewDescriptor::setSliceRange(NS::Range sliceRange)
{
    _MTL_msg_v_setSliceRange__NS__Range((const void*)this, nullptr, sliceRange);
}

_MTL_INLINE MTL::TextureSwizzleChannels MTL::TextureViewDescriptor::swizzle() const
{
    return _MTL_msg_MTL__TextureSwizzleChannels_swizzle((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TextureViewDescriptor::setSwizzle(MTL::TextureSwizzleChannels swizzle)
{
    _MTL_msg_v_setSwizzle__MTL__TextureSwizzleChannels((const void*)this, nullptr, swizzle);
}

_MTL_INLINE MTL::Resource* MTL::Texture::rootResource() const
{
    return _MTL_msg_MTL__Resourcep_rootResource((const void*)this, nullptr);
}

_MTL_INLINE MTL::Texture* MTL::Texture::parentTexture() const
{
    return _MTL_msg_MTL__Texturep_parentTexture((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Texture::parentRelativeLevel() const
{
    return _MTL_msg_NS__UInteger_parentRelativeLevel((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Texture::parentRelativeSlice() const
{
    return _MTL_msg_NS__UInteger_parentRelativeSlice((const void*)this, nullptr);
}

_MTL_INLINE MTL::Buffer* MTL::Texture::buffer() const
{
    return _MTL_msg_MTL__Bufferp_buffer((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Texture::bufferOffset() const
{
    return _MTL_msg_NS__UInteger_bufferOffset((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Texture::bufferBytesPerRow() const
{
    return _MTL_msg_NS__UInteger_bufferBytesPerRow((const void*)this, nullptr);
}

_MTL_INLINE IOSurfaceRef MTL::Texture::iosurface() const
{
    return _MTL_msg_IOSurfaceRef_iosurface((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Texture::iosurfacePlane() const
{
    return _MTL_msg_NS__UInteger_iosurfacePlane((const void*)this, nullptr);
}

_MTL_INLINE MTL::TextureType MTL::Texture::textureType() const
{
    return _MTL_msg_MTL__TextureType_textureType((const void*)this, nullptr);
}

_MTL_INLINE MTL::PixelFormat MTL::Texture::pixelFormat() const
{
    return _MTL_msg_MTL__PixelFormat_pixelFormat((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Texture::width() const
{
    return _MTL_msg_NS__UInteger_width((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Texture::height() const
{
    return _MTL_msg_NS__UInteger_height((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Texture::depth() const
{
    return _MTL_msg_NS__UInteger_depth((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Texture::mipmapLevelCount() const
{
    return _MTL_msg_NS__UInteger_mipmapLevelCount((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Texture::sampleCount() const
{
    return _MTL_msg_NS__UInteger_sampleCount((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Texture::arrayLength() const
{
    return _MTL_msg_NS__UInteger_arrayLength((const void*)this, nullptr);
}

_MTL_INLINE MTL::TextureUsage MTL::Texture::usage() const
{
    return _MTL_msg_MTL__TextureUsage_usage((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Texture::shareable() const
{
    return _MTL_msg_bool_shareable((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Texture::framebufferOnly() const
{
    return _MTL_msg_bool_framebufferOnly((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Texture::firstMipmapInTail() const
{
    return _MTL_msg_NS__UInteger_firstMipmapInTail((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Texture::tailSizeInBytes() const
{
    return _MTL_msg_NS__UInteger_tailSizeInBytes((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Texture::isSparse() const
{
    return _MTL_msg_bool_isSparse((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Texture::allowGPUOptimizedContents() const
{
    return _MTL_msg_bool_allowGPUOptimizedContents((const void*)this, nullptr);
}

_MTL_INLINE MTL::TextureCompressionType MTL::Texture::compressionType() const
{
    return _MTL_msg_MTL__TextureCompressionType_compressionType((const void*)this, nullptr);
}

_MTL_INLINE MTL::ResourceID MTL::Texture::gpuResourceID() const
{
    return _MTL_msg_MTL__ResourceID_gpuResourceID((const void*)this, nullptr);
}

_MTL_INLINE MTL::Texture* MTL::Texture::remoteStorageTexture() const
{
    return _MTL_msg_MTL__Texturep_remoteStorageTexture((const void*)this, nullptr);
}

_MTL_INLINE MTL::TextureSwizzleChannels MTL::Texture::swizzle() const
{
    return _MTL_msg_MTL__TextureSwizzleChannels_swizzle((const void*)this, nullptr);
}

_MTL_INLINE MTL::TextureSparseTier MTL::Texture::sparseTextureTier() const
{
    return _MTL_msg_MTL__TextureSparseTier_sparseTextureTier((const void*)this, nullptr);
}

_MTL_INLINE void MTL::Texture::getBytes(void * pixelBytes, NS::UInteger bytesPerRow, NS::UInteger bytesPerImage, MTL::Region region, NS::UInteger level, NS::UInteger slice)
{
    _MTL_msg_v_getBytes_bytesPerRow_bytesPerImage_fromRegion_mipmapLevel_slice__voidp_NS__UInteger_NS__UInteger_MTL__Region_NS__UInteger_NS__UInteger((const void*)this, nullptr, pixelBytes, bytesPerRow, bytesPerImage, region, level, slice);
}

_MTL_INLINE void MTL::Texture::replace(MTL::Region region, NS::UInteger level, NS::UInteger slice, const void * pixelBytes, NS::UInteger bytesPerRow, NS::UInteger bytesPerImage)
{
    _MTL_msg_v_replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage__MTL__Region_NS__UInteger_NS__UInteger_constvoidp_NS__UInteger_NS__UInteger((const void*)this, nullptr, region, level, slice, pixelBytes, bytesPerRow, bytesPerImage);
}

_MTL_INLINE void MTL::Texture::getBytes(void * pixelBytes, NS::UInteger bytesPerRow, MTL::Region region, NS::UInteger level)
{
    _MTL_msg_v_getBytes_bytesPerRow_fromRegion_mipmapLevel__voidp_NS__UInteger_MTL__Region_NS__UInteger((const void*)this, nullptr, pixelBytes, bytesPerRow, region, level);
}

_MTL_INLINE void MTL::Texture::replace(MTL::Region region, NS::UInteger level, const void * pixelBytes, NS::UInteger bytesPerRow)
{
    _MTL_msg_v_replaceRegion_mipmapLevel_withBytes_bytesPerRow__MTL__Region_NS__UInteger_constvoidp_NS__UInteger((const void*)this, nullptr, region, level, pixelBytes, bytesPerRow);
}

_MTL_INLINE MTL::Texture* MTL::Texture::newTextureView(MTL::PixelFormat pixelFormat)
{
    return _MTL_msg_MTL__Texturep_newTextureViewWithPixelFormat__MTL__PixelFormat((const void*)this, nullptr, pixelFormat);
}

_MTL_INLINE MTL::Texture* MTL::Texture::newTextureView(MTL::PixelFormat pixelFormat, MTL::TextureType textureType, NS::Range levelRange, NS::Range sliceRange)
{
    return _MTL_msg_MTL__Texturep_newTextureViewWithPixelFormat_textureType_levels_slices__MTL__PixelFormat_MTL__TextureType_NS__Range_NS__Range((const void*)this, nullptr, pixelFormat, textureType, levelRange, sliceRange);
}

_MTL_INLINE MTL::SharedTextureHandle* MTL::Texture::newSharedTextureHandle()
{
    return _MTL_msg_MTL__SharedTextureHandlep_newSharedTextureHandle((const void*)this, nullptr);
}

_MTL_INLINE MTL::Texture* MTL::Texture::newTextureView(MTL::TextureViewDescriptor* descriptor)
{
    return _MTL_msg_MTL__Texturep_newTextureViewWithDescriptor__MTL__TextureViewDescriptorp((const void*)this, nullptr, descriptor);
}

_MTL_INLINE MTL::Texture* MTL::Texture::newRemoteTextureView(MTL::Device* device)
{
    return _MTL_msg_MTL__Texturep_newRemoteTextureViewForDevice__MTL__Devicep((const void*)this, nullptr, device);
}

_MTL_INLINE MTL::Texture* MTL::Texture::newTextureView(MTL::PixelFormat pixelFormat, MTL::TextureType textureType, NS::Range levelRange, NS::Range sliceRange, MTL::TextureSwizzleChannels swizzle)
{
    return _MTL_msg_MTL__Texturep_newTextureViewWithPixelFormat_textureType_levels_slices_swizzle__MTL__PixelFormat_MTL__TextureType_NS__Range_NS__Range_MTL__TextureSwizzleChannels((const void*)this, nullptr, pixelFormat, textureType, levelRange, sliceRange, swizzle);
}

_MTL_INLINE bool MTL::Texture::isShareable()
{
    return _MTL_msg_bool_isShareable((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::Texture::isFramebufferOnly()
{
    return _MTL_msg_bool_isFramebufferOnly((const void*)this, nullptr);
}
