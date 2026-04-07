//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLBlitCommandEncoder.hpp
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
#include "MTLCommandEncoder.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include "MTLTypes.hpp"
#include <cstdint>

namespace MTL
{
class Buffer;
class CounterSampleBuffer;
class Fence;
class IndirectCommandBuffer;
class Resource;
class Tensor;
class TensorExtents;
class Texture;

_MTL_OPTIONS(NS::UInteger, BlitOption) {
    BlitOptionNone = 0,
    BlitOptionDepthFromDepthStencil = 1,
    BlitOptionStencilFromDepthStencil = 1 << 1,
    BlitOptionRowLinearPVRTC = 1 << 2,
};

class BlitCommandEncoder : public NS::Referencing<BlitCommandEncoder, CommandEncoder>
{
public:
    void copyFromBuffer(const MTL::Buffer* sourceBuffer, NS::UInteger sourceOffset, NS::UInteger sourceBytesPerRow, NS::UInteger sourceBytesPerImage, MTL::Size sourceSize, const MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin);
    void copyFromBuffer(const MTL::Buffer* sourceBuffer, NS::UInteger sourceOffset, NS::UInteger sourceBytesPerRow, NS::UInteger sourceBytesPerImage, MTL::Size sourceSize, const MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin, MTL::BlitOption options);
    void copyFromBuffer(const MTL::Buffer* sourceBuffer, NS::UInteger sourceOffset, const MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset, NS::UInteger size);

    void copyFromTensor(const MTL::Tensor* sourceTensor, const MTL::TensorExtents* sourceOrigin, const MTL::TensorExtents* sourceDimensions, const MTL::Tensor* destinationTensor, const MTL::TensorExtents* destinationOrigin, const MTL::TensorExtents* destinationDimensions);

    void copyFromTexture(const MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, const MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin);
    void copyFromTexture(const MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, const MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset, NS::UInteger destinationBytesPerRow, NS::UInteger destinationBytesPerImage);
    void copyFromTexture(const MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, const MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset, NS::UInteger destinationBytesPerRow, NS::UInteger destinationBytesPerImage, MTL::BlitOption options);
    void copyFromTexture(const MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, const MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, NS::UInteger sliceCount, NS::UInteger levelCount);
    void copyFromTexture(const MTL::Texture* sourceTexture, const MTL::Texture* destinationTexture);

    void copyIndirectCommandBuffer(const MTL::IndirectCommandBuffer* source, NS::Range sourceRange, const MTL::IndirectCommandBuffer* destination, NS::UInteger destinationIndex);

    void fillBuffer(const MTL::Buffer* buffer, NS::Range range, uint8_t value);

    void generateMipmaps(const MTL::Texture* texture);

    void getTextureAccessCounters(const MTL::Texture* texture, MTL::Region region, NS::UInteger mipLevel, NS::UInteger slice, bool resetCounters, const MTL::Buffer* countersBuffer, NS::UInteger countersBufferOffset);

    void optimizeContentsForCPUAccess(const MTL::Texture* texture);
    void optimizeContentsForCPUAccess(const MTL::Texture* texture, NS::UInteger slice, NS::UInteger level);

    void optimizeContentsForGPUAccess(const MTL::Texture* texture);
    void optimizeContentsForGPUAccess(const MTL::Texture* texture, NS::UInteger slice, NS::UInteger level);

    void optimizeIndirectCommandBuffer(const MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::Range range);

    void resetCommandsInBuffer(const MTL::IndirectCommandBuffer* buffer, NS::Range range);

    void resetTextureAccessCounters(const MTL::Texture* texture, MTL::Region region, NS::UInteger mipLevel, NS::UInteger slice);

    void resolveCounters(const MTL::CounterSampleBuffer* sampleBuffer, NS::Range range, const MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset);

    void sampleCountersInBuffer(const MTL::CounterSampleBuffer* sampleBuffer, NS::UInteger sampleIndex, bool barrier);

    void synchronizeResource(const MTL::Resource* resource);

    void synchronizeTexture(const MTL::Texture* texture, NS::UInteger slice, NS::UInteger level);

    void updateFence(const MTL::Fence* fence);

    void waitForFence(const MTL::Fence* fence);
};

}
_MTL_INLINE void MTL::BlitCommandEncoder::copyFromBuffer(const MTL::Buffer* sourceBuffer, NS::UInteger sourceOffset, NS::UInteger sourceBytesPerRow, NS::UInteger sourceBytesPerImage, MTL::Size sourceSize, const MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_), sourceBuffer, sourceOffset, sourceBytesPerRow, sourceBytesPerImage, sourceSize, destinationTexture, destinationSlice, destinationLevel, destinationOrigin);
}

_MTL_INLINE void MTL::BlitCommandEncoder::copyFromBuffer(const MTL::Buffer* sourceBuffer, NS::UInteger sourceOffset, NS::UInteger sourceBytesPerRow, NS::UInteger sourceBytesPerImage, MTL::Size sourceSize, const MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin, MTL::BlitOption options)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options_), sourceBuffer, sourceOffset, sourceBytesPerRow, sourceBytesPerImage, sourceSize, destinationTexture, destinationSlice, destinationLevel, destinationOrigin, options);
}

_MTL_INLINE void MTL::BlitCommandEncoder::copyFromBuffer(const MTL::Buffer* sourceBuffer, NS::UInteger sourceOffset, const MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset, NS::UInteger size)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_), sourceBuffer, sourceOffset, destinationBuffer, destinationOffset, size);
}

_MTL_INLINE void MTL::BlitCommandEncoder::copyFromTensor(const MTL::Tensor* sourceTensor, const MTL::TensorExtents* sourceOrigin, const MTL::TensorExtents* sourceDimensions, const MTL::Tensor* destinationTensor, const MTL::TensorExtents* destinationOrigin, const MTL::TensorExtents* destinationDimensions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyFromTensor_sourceOrigin_sourceDimensions_toTensor_destinationOrigin_destinationDimensions_), sourceTensor, sourceOrigin, sourceDimensions, destinationTensor, destinationOrigin, destinationDimensions);
}

_MTL_INLINE void MTL::BlitCommandEncoder::copyFromTexture(const MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, const MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_), sourceTexture, sourceSlice, sourceLevel, sourceOrigin, sourceSize, destinationTexture, destinationSlice, destinationLevel, destinationOrigin);
}

_MTL_INLINE void MTL::BlitCommandEncoder::copyFromTexture(const MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, const MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset, NS::UInteger destinationBytesPerRow, NS::UInteger destinationBytesPerImage)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_), sourceTexture, sourceSlice, sourceLevel, sourceOrigin, sourceSize, destinationBuffer, destinationOffset, destinationBytesPerRow, destinationBytesPerImage);
}

_MTL_INLINE void MTL::BlitCommandEncoder::copyFromTexture(const MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, const MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset, NS::UInteger destinationBytesPerRow, NS::UInteger destinationBytesPerImage, MTL::BlitOption options)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options_), sourceTexture, sourceSlice, sourceLevel, sourceOrigin, sourceSize, destinationBuffer, destinationOffset, destinationBytesPerRow, destinationBytesPerImage, options);
}

_MTL_INLINE void MTL::BlitCommandEncoder::copyFromTexture(const MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, const MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, NS::UInteger sliceCount, NS::UInteger levelCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyFromTexture_sourceSlice_sourceLevel_toTexture_destinationSlice_destinationLevel_sliceCount_levelCount_), sourceTexture, sourceSlice, sourceLevel, destinationTexture, destinationSlice, destinationLevel, sliceCount, levelCount);
}

_MTL_INLINE void MTL::BlitCommandEncoder::copyFromTexture(const MTL::Texture* sourceTexture, const MTL::Texture* destinationTexture)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyFromTexture_toTexture_), sourceTexture, destinationTexture);
}

_MTL_INLINE void MTL::BlitCommandEncoder::copyIndirectCommandBuffer(const MTL::IndirectCommandBuffer* source, NS::Range sourceRange, const MTL::IndirectCommandBuffer* destination, NS::UInteger destinationIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyIndirectCommandBuffer_sourceRange_destination_destinationIndex_), source, sourceRange, destination, destinationIndex);
}

_MTL_INLINE void MTL::BlitCommandEncoder::fillBuffer(const MTL::Buffer* buffer, NS::Range range, uint8_t value)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(fillBuffer_range_value_), buffer, range, value);
}

_MTL_INLINE void MTL::BlitCommandEncoder::generateMipmaps(const MTL::Texture* texture)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(generateMipmapsForTexture_), texture);
}

_MTL_INLINE void MTL::BlitCommandEncoder::getTextureAccessCounters(const MTL::Texture* texture, MTL::Region region, NS::UInteger mipLevel, NS::UInteger slice, bool resetCounters, const MTL::Buffer* countersBuffer, NS::UInteger countersBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(getTextureAccessCounters_region_mipLevel_slice_resetCounters_countersBuffer_countersBufferOffset_), texture, region, mipLevel, slice, resetCounters, countersBuffer, countersBufferOffset);
}

_MTL_INLINE void MTL::BlitCommandEncoder::optimizeContentsForCPUAccess(const MTL::Texture* texture)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(optimizeContentsForCPUAccess_), texture);
}

_MTL_INLINE void MTL::BlitCommandEncoder::optimizeContentsForCPUAccess(const MTL::Texture* texture, NS::UInteger slice, NS::UInteger level)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(optimizeContentsForCPUAccess_slice_level_), texture, slice, level);
}

_MTL_INLINE void MTL::BlitCommandEncoder::optimizeContentsForGPUAccess(const MTL::Texture* texture)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(optimizeContentsForGPUAccess_), texture);
}

_MTL_INLINE void MTL::BlitCommandEncoder::optimizeContentsForGPUAccess(const MTL::Texture* texture, NS::UInteger slice, NS::UInteger level)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(optimizeContentsForGPUAccess_slice_level_), texture, slice, level);
}

_MTL_INLINE void MTL::BlitCommandEncoder::optimizeIndirectCommandBuffer(const MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(optimizeIndirectCommandBuffer_withRange_), indirectCommandBuffer, range);
}

_MTL_INLINE void MTL::BlitCommandEncoder::resetCommandsInBuffer(const MTL::IndirectCommandBuffer* buffer, NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(resetCommandsInBuffer_withRange_), buffer, range);
}

_MTL_INLINE void MTL::BlitCommandEncoder::resetTextureAccessCounters(const MTL::Texture* texture, MTL::Region region, NS::UInteger mipLevel, NS::UInteger slice)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(resetTextureAccessCounters_region_mipLevel_slice_), texture, region, mipLevel, slice);
}

_MTL_INLINE void MTL::BlitCommandEncoder::resolveCounters(const MTL::CounterSampleBuffer* sampleBuffer, NS::Range range, const MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(resolveCounters_inRange_destinationBuffer_destinationOffset_), sampleBuffer, range, destinationBuffer, destinationOffset);
}

_MTL_INLINE void MTL::BlitCommandEncoder::sampleCountersInBuffer(const MTL::CounterSampleBuffer* sampleBuffer, NS::UInteger sampleIndex, bool barrier)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(sampleCountersInBuffer_atSampleIndex_withBarrier_), sampleBuffer, sampleIndex, barrier);
}

_MTL_INLINE void MTL::BlitCommandEncoder::synchronizeResource(const MTL::Resource* resource)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(synchronizeResource_), resource);
}

_MTL_INLINE void MTL::BlitCommandEncoder::synchronizeTexture(const MTL::Texture* texture, NS::UInteger slice, NS::UInteger level)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(synchronizeTexture_slice_level_), texture, slice, level);
}

_MTL_INLINE void MTL::BlitCommandEncoder::updateFence(const MTL::Fence* fence)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(updateFence_), fence);
}

_MTL_INLINE void MTL::BlitCommandEncoder::waitForFence(const MTL::Fence* fence)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(waitForFence_), fence);
}
