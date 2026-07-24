#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTLCommandEncoder.hpp"

namespace MTL {
    class Buffer;
    class CounterSampleBuffer;
    class Fence;
    class IndirectCommandBuffer;
    class Resource;
    class Tensor;
    class TensorExtents;
    class Texture;
}

namespace MTL
{

_MTL_OPTIONS(NS::UInteger, BlitOption) {
    BlitOptionNone = 0,
    BlitOptionDepthFromDepthStencil = 1 << 0,
    BlitOptionStencilFromDepthStencil = 1 << 1,
    BlitOptionRowLinearPVRTC = 1 << 2,
};


class BlitCommandEncoder : public NS::Referencing<BlitCommandEncoder, MTL::CommandEncoder>
{
public:
    void copyFromBuffer(MTL::Buffer* sourceBuffer, NS::UInteger sourceOffset, NS::UInteger sourceBytesPerRow, NS::UInteger sourceBytesPerImage, MTL::Size sourceSize, MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin);
    void copyFromBuffer(MTL::Buffer* sourceBuffer, NS::UInteger sourceOffset, NS::UInteger sourceBytesPerRow, NS::UInteger sourceBytesPerImage, MTL::Size sourceSize, MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin, MTL::BlitOption options);
    void copyFromBuffer(MTL::Buffer* sourceBuffer, NS::UInteger sourceOffset, MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset, NS::UInteger size);
    void copyFromTensor(MTL::Tensor* sourceTensor, MTL::TensorExtents* sourceOrigin, MTL::TensorExtents* sourceDimensions, MTL::Tensor* destinationTensor, MTL::TensorExtents* destinationOrigin, MTL::TensorExtents* destinationDimensions);
    void copyFromTexture(MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin);
    void copyFromTexture(MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset, NS::UInteger destinationBytesPerRow, NS::UInteger destinationBytesPerImage);
    void copyFromTexture(MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset, NS::UInteger destinationBytesPerRow, NS::UInteger destinationBytesPerImage, MTL::BlitOption options);
    void copyFromTexture(MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, NS::UInteger sliceCount, NS::UInteger levelCount);
    void copyFromTexture(MTL::Texture* sourceTexture, MTL::Texture* destinationTexture);
    void copyIndirectCommandBuffer(MTL::IndirectCommandBuffer* source, NS::Range sourceRange, MTL::IndirectCommandBuffer* destination, NS::UInteger destinationIndex);
    void fillBuffer(MTL::Buffer* buffer, NS::Range range, uint8_t value);
    void generateMipmaps(MTL::Texture* texture);
    void getTextureAccessCounters(MTL::Texture* texture, MTL::Region region, NS::UInteger mipLevel, NS::UInteger slice, bool resetCounters, MTL::Buffer* countersBuffer, NS::UInteger countersBufferOffset);
    void optimizeContents(MTL::Texture* texture);
    void optimizeContents(MTL::Texture* texture, NS::UInteger slice, NS::UInteger level);
    void optimizeIndirectCommandBuffer(MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::Range range);
    void resetCommandsInBuffer(MTL::IndirectCommandBuffer* buffer, NS::Range range);
    void resetTextureAccessCounters(MTL::Texture* texture, MTL::Region region, NS::UInteger mipLevel, NS::UInteger slice);
    void resolveCounters(MTL::CounterSampleBuffer* sampleBuffer, NS::Range range, MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset);
    void sampleCountersInBuffer(MTL::CounterSampleBuffer* sampleBuffer, NS::UInteger sampleIndex, bool barrier);
    void synchronize(MTL::Resource* resource);
    void synchronize(MTL::Texture* texture, NS::UInteger slice, NS::UInteger level);
    void updateFence(MTL::Fence* fence);
    void waitForFence(MTL::Fence* fence);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLBlitCommandEncoder;

_MTL_INLINE void MTL::BlitCommandEncoder::synchronize(MTL::Resource* resource)
{
    _MTL_msg_v_synchronizeResource__MTL__Resourcep((const void*)this, nullptr, resource);
}

_MTL_INLINE void MTL::BlitCommandEncoder::synchronize(MTL::Texture* texture, NS::UInteger slice, NS::UInteger level)
{
    _MTL_msg_v_synchronizeTexture_slice_level__MTL__Texturep_NS__UInteger_NS__UInteger((const void*)this, nullptr, texture, slice, level);
}

_MTL_INLINE void MTL::BlitCommandEncoder::copyFromTexture(MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin)
{
    _MTL_msg_v_copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin__MTL__Texturep_NS__UInteger_NS__UInteger_MTL__Origin_MTL__Size_MTL__Texturep_NS__UInteger_NS__UInteger_MTL__Origin((const void*)this, nullptr, sourceTexture, sourceSlice, sourceLevel, sourceOrigin, sourceSize, destinationTexture, destinationSlice, destinationLevel, destinationOrigin);
}

_MTL_INLINE void MTL::BlitCommandEncoder::copyFromBuffer(MTL::Buffer* sourceBuffer, NS::UInteger sourceOffset, NS::UInteger sourceBytesPerRow, NS::UInteger sourceBytesPerImage, MTL::Size sourceSize, MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin)
{
    _MTL_msg_v_copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin__MTL__Bufferp_NS__UInteger_NS__UInteger_NS__UInteger_MTL__Size_MTL__Texturep_NS__UInteger_NS__UInteger_MTL__Origin((const void*)this, nullptr, sourceBuffer, sourceOffset, sourceBytesPerRow, sourceBytesPerImage, sourceSize, destinationTexture, destinationSlice, destinationLevel, destinationOrigin);
}

_MTL_INLINE void MTL::BlitCommandEncoder::copyFromBuffer(MTL::Buffer* sourceBuffer, NS::UInteger sourceOffset, NS::UInteger sourceBytesPerRow, NS::UInteger sourceBytesPerImage, MTL::Size sourceSize, MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin, MTL::BlitOption options)
{
    _MTL_msg_v_copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options__MTL__Bufferp_NS__UInteger_NS__UInteger_NS__UInteger_MTL__Size_MTL__Texturep_NS__UInteger_NS__UInteger_MTL__Origin_MTL__BlitOption((const void*)this, nullptr, sourceBuffer, sourceOffset, sourceBytesPerRow, sourceBytesPerImage, sourceSize, destinationTexture, destinationSlice, destinationLevel, destinationOrigin, options);
}

_MTL_INLINE void MTL::BlitCommandEncoder::copyFromTexture(MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset, NS::UInteger destinationBytesPerRow, NS::UInteger destinationBytesPerImage)
{
    _MTL_msg_v_copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage__MTL__Texturep_NS__UInteger_NS__UInteger_MTL__Origin_MTL__Size_MTL__Bufferp_NS__UInteger_NS__UInteger_NS__UInteger((const void*)this, nullptr, sourceTexture, sourceSlice, sourceLevel, sourceOrigin, sourceSize, destinationBuffer, destinationOffset, destinationBytesPerRow, destinationBytesPerImage);
}

_MTL_INLINE void MTL::BlitCommandEncoder::copyFromTexture(MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset, NS::UInteger destinationBytesPerRow, NS::UInteger destinationBytesPerImage, MTL::BlitOption options)
{
    _MTL_msg_v_copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options__MTL__Texturep_NS__UInteger_NS__UInteger_MTL__Origin_MTL__Size_MTL__Bufferp_NS__UInteger_NS__UInteger_NS__UInteger_MTL__BlitOption((const void*)this, nullptr, sourceTexture, sourceSlice, sourceLevel, sourceOrigin, sourceSize, destinationBuffer, destinationOffset, destinationBytesPerRow, destinationBytesPerImage, options);
}

_MTL_INLINE void MTL::BlitCommandEncoder::generateMipmaps(MTL::Texture* texture)
{
    _MTL_msg_v_generateMipmapsForTexture__MTL__Texturep((const void*)this, nullptr, texture);
}

_MTL_INLINE void MTL::BlitCommandEncoder::fillBuffer(MTL::Buffer* buffer, NS::Range range, uint8_t value)
{
    _MTL_msg_v_fillBuffer_range_value__MTL__Bufferp_NS__Range_uint8_t((const void*)this, nullptr, buffer, range, value);
}

_MTL_INLINE void MTL::BlitCommandEncoder::copyFromTexture(MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, NS::UInteger sliceCount, NS::UInteger levelCount)
{
    _MTL_msg_v_copyFromTexture_sourceSlice_sourceLevel_toTexture_destinationSlice_destinationLevel_sliceCount_levelCount__MTL__Texturep_NS__UInteger_NS__UInteger_MTL__Texturep_NS__UInteger_NS__UInteger_NS__UInteger_NS__UInteger((const void*)this, nullptr, sourceTexture, sourceSlice, sourceLevel, destinationTexture, destinationSlice, destinationLevel, sliceCount, levelCount);
}

_MTL_INLINE void MTL::BlitCommandEncoder::copyFromTexture(MTL::Texture* sourceTexture, MTL::Texture* destinationTexture)
{
    _MTL_msg_v_copyFromTexture_toTexture__MTL__Texturep_MTL__Texturep((const void*)this, nullptr, sourceTexture, destinationTexture);
}

_MTL_INLINE void MTL::BlitCommandEncoder::copyFromBuffer(MTL::Buffer* sourceBuffer, NS::UInteger sourceOffset, MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset, NS::UInteger size)
{
    _MTL_msg_v_copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size__MTL__Bufferp_NS__UInteger_MTL__Bufferp_NS__UInteger_NS__UInteger((const void*)this, nullptr, sourceBuffer, sourceOffset, destinationBuffer, destinationOffset, size);
}

_MTL_INLINE void MTL::BlitCommandEncoder::updateFence(MTL::Fence* fence)
{
    _MTL_msg_v_updateFence__MTL__Fencep((const void*)this, nullptr, fence);
}

_MTL_INLINE void MTL::BlitCommandEncoder::waitForFence(MTL::Fence* fence)
{
    _MTL_msg_v_waitForFence__MTL__Fencep((const void*)this, nullptr, fence);
}

_MTL_INLINE void MTL::BlitCommandEncoder::getTextureAccessCounters(MTL::Texture* texture, MTL::Region region, NS::UInteger mipLevel, NS::UInteger slice, bool resetCounters, MTL::Buffer* countersBuffer, NS::UInteger countersBufferOffset)
{
    _MTL_msg_v_getTextureAccessCounters_region_mipLevel_slice_resetCounters_countersBuffer_countersBufferOffset__MTL__Texturep_MTL__Region_NS__UInteger_NS__UInteger_bool_MTL__Bufferp_NS__UInteger((const void*)this, nullptr, texture, region, mipLevel, slice, resetCounters, countersBuffer, countersBufferOffset);
}

_MTL_INLINE void MTL::BlitCommandEncoder::resetTextureAccessCounters(MTL::Texture* texture, MTL::Region region, NS::UInteger mipLevel, NS::UInteger slice)
{
    _MTL_msg_v_resetTextureAccessCounters_region_mipLevel_slice__MTL__Texturep_MTL__Region_NS__UInteger_NS__UInteger((const void*)this, nullptr, texture, region, mipLevel, slice);
}

_MTL_INLINE void MTL::BlitCommandEncoder::optimizeContents(MTL::Texture* texture)
{
    _MTL_msg_v_optimizeContentsForGPUAccess__MTL__Texturep((const void*)this, nullptr, texture);
}

_MTL_INLINE void MTL::BlitCommandEncoder::optimizeContents(MTL::Texture* texture, NS::UInteger slice, NS::UInteger level)
{
    _MTL_msg_v_optimizeContentsForGPUAccess_slice_level__MTL__Texturep_NS__UInteger_NS__UInteger((const void*)this, nullptr, texture, slice, level);
}

_MTL_INLINE void MTL::BlitCommandEncoder::resetCommandsInBuffer(MTL::IndirectCommandBuffer* buffer, NS::Range range)
{
    _MTL_msg_v_resetCommandsInBuffer_withRange__MTL__IndirectCommandBufferp_NS__Range((const void*)this, nullptr, buffer, range);
}

_MTL_INLINE void MTL::BlitCommandEncoder::copyIndirectCommandBuffer(MTL::IndirectCommandBuffer* source, NS::Range sourceRange, MTL::IndirectCommandBuffer* destination, NS::UInteger destinationIndex)
{
    _MTL_msg_v_copyIndirectCommandBuffer_sourceRange_destination_destinationIndex__MTL__IndirectCommandBufferp_NS__Range_MTL__IndirectCommandBufferp_NS__UInteger((const void*)this, nullptr, source, sourceRange, destination, destinationIndex);
}

_MTL_INLINE void MTL::BlitCommandEncoder::optimizeIndirectCommandBuffer(MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::Range range)
{
    _MTL_msg_v_optimizeIndirectCommandBuffer_withRange__MTL__IndirectCommandBufferp_NS__Range((const void*)this, nullptr, indirectCommandBuffer, range);
}

_MTL_INLINE void MTL::BlitCommandEncoder::sampleCountersInBuffer(MTL::CounterSampleBuffer* sampleBuffer, NS::UInteger sampleIndex, bool barrier)
{
    _MTL_msg_v_sampleCountersInBuffer_atSampleIndex_withBarrier__MTL__CounterSampleBufferp_NS__UInteger_bool((const void*)this, nullptr, sampleBuffer, sampleIndex, barrier);
}

_MTL_INLINE void MTL::BlitCommandEncoder::resolveCounters(MTL::CounterSampleBuffer* sampleBuffer, NS::Range range, MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset)
{
    _MTL_msg_v_resolveCounters_inRange_destinationBuffer_destinationOffset__MTL__CounterSampleBufferp_NS__Range_MTL__Bufferp_NS__UInteger((const void*)this, nullptr, sampleBuffer, range, destinationBuffer, destinationOffset);
}

_MTL_INLINE void MTL::BlitCommandEncoder::copyFromTensor(MTL::Tensor* sourceTensor, MTL::TensorExtents* sourceOrigin, MTL::TensorExtents* sourceDimensions, MTL::Tensor* destinationTensor, MTL::TensorExtents* destinationOrigin, MTL::TensorExtents* destinationDimensions)
{
    _MTL_msg_v_copyFromTensor_sourceOrigin_sourceDimensions_toTensor_destinationOrigin_destinationDimensions__MTL__Tensorp_MTL__TensorExtentsp_MTL__TensorExtentsp_MTL__Tensorp_MTL__TensorExtentsp_MTL__TensorExtentsp((const void*)this, nullptr, sourceTensor, sourceOrigin, sourceDimensions, destinationTensor, destinationOrigin, destinationDimensions);
}
