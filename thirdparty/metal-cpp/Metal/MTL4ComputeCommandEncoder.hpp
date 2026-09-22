//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTL4ComputeCommandEncoder.hpp
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
#include "MTL4CommandEncoder.hpp"
#include "MTL4Counters.hpp"
#include "MTLAccelerationStructure.hpp"
#include "MTLAccelerationStructureTypes.hpp"
#include "MTLBlitCommandEncoder.hpp"
#include "MTLCommandEncoder.hpp"
#include "MTLDefines.hpp"
#include "MTLGPUAddress.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include "MTLTypes.hpp"
#include <cstdint>

namespace MTL4
{
class AccelerationStructureDescriptor;
class ArgumentTable;
class CounterHeap;
}

namespace MTL
{
class AccelerationStructure;
class Buffer;
class ComputePipelineState;
class IndirectCommandBuffer;
class Tensor;
class TensorExtents;
class Texture;
}

namespace MTL4
{
class ComputeCommandEncoder : public NS::Referencing<ComputeCommandEncoder, CommandEncoder>
{
public:
    void        buildAccelerationStructure(const MTL::AccelerationStructure* accelerationStructure, const MTL4::AccelerationStructureDescriptor* descriptor, const MTL4::BufferRange scratchBuffer);

    void        copyAccelerationStructure(const MTL::AccelerationStructure* sourceAccelerationStructure, const MTL::AccelerationStructure* destinationAccelerationStructure);

    void        copyAndCompactAccelerationStructure(const MTL::AccelerationStructure* sourceAccelerationStructure, const MTL::AccelerationStructure* destinationAccelerationStructure);

    void        copyFromBuffer(const MTL::Buffer* sourceBuffer, NS::UInteger sourceOffset, const MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset, NS::UInteger size);
    void        copyFromBuffer(const MTL::Buffer* sourceBuffer, NS::UInteger sourceOffset, NS::UInteger sourceBytesPerRow, NS::UInteger sourceBytesPerImage, MTL::Size sourceSize, const MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin);
    void        copyFromBuffer(const MTL::Buffer* sourceBuffer, NS::UInteger sourceOffset, NS::UInteger sourceBytesPerRow, NS::UInteger sourceBytesPerImage, MTL::Size sourceSize, const MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin, MTL::BlitOption options);

    void        copyFromTensor(const MTL::Tensor* sourceTensor, const MTL::TensorExtents* sourceOrigin, const MTL::TensorExtents* sourceDimensions, const MTL::Tensor* destinationTensor, const MTL::TensorExtents* destinationOrigin, const MTL::TensorExtents* destinationDimensions);

    void        copyFromTexture(const MTL::Texture* sourceTexture, const MTL::Texture* destinationTexture);
    void        copyFromTexture(const MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, const MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, NS::UInteger sliceCount, NS::UInteger levelCount);
    void        copyFromTexture(const MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, const MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin);
    void        copyFromTexture(const MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, const MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset, NS::UInteger destinationBytesPerRow, NS::UInteger destinationBytesPerImage);
    void        copyFromTexture(const MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, const MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset, NS::UInteger destinationBytesPerRow, NS::UInteger destinationBytesPerImage, MTL::BlitOption options);

    void        copyIndirectCommandBuffer(const MTL::IndirectCommandBuffer* source, NS::Range sourceRange, const MTL::IndirectCommandBuffer* destination, NS::UInteger destinationIndex);

    void        dispatchThreadgroups(MTL::Size threadgroupsPerGrid, MTL::Size threadsPerThreadgroup);
    void        dispatchThreadgroups(MTL::GPUAddress indirectBuffer, MTL::Size threadsPerThreadgroup);

    void        dispatchThreads(MTL::Size threadsPerGrid, MTL::Size threadsPerThreadgroup);
    void        dispatchThreads(MTL::GPUAddress indirectBuffer);

    void        executeCommandsInBuffer(const MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::Range executionRange);
    void        executeCommandsInBuffer(const MTL::IndirectCommandBuffer* indirectCommandbuffer, MTL::GPUAddress indirectRangeBuffer);

    void        fillBuffer(const MTL::Buffer* buffer, NS::Range range, uint8_t value);

    void        generateMipmaps(const MTL::Texture* texture);

    void        optimizeContentsForCPUAccess(const MTL::Texture* texture);
    void        optimizeContentsForCPUAccess(const MTL::Texture* texture, NS::UInteger slice, NS::UInteger level);

    void        optimizeContentsForGPUAccess(const MTL::Texture* texture);
    void        optimizeContentsForGPUAccess(const MTL::Texture* texture, NS::UInteger slice, NS::UInteger level);

    void        optimizeIndirectCommandBuffer(const MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::Range range);

    void        refitAccelerationStructure(const MTL::AccelerationStructure* sourceAccelerationStructure, const MTL4::AccelerationStructureDescriptor* descriptor, const MTL::AccelerationStructure* destinationAccelerationStructure, const MTL4::BufferRange scratchBuffer);
    void        refitAccelerationStructure(const MTL::AccelerationStructure* sourceAccelerationStructure, const MTL4::AccelerationStructureDescriptor* descriptor, const MTL::AccelerationStructure* destinationAccelerationStructure, const MTL4::BufferRange scratchBuffer, MTL::AccelerationStructureRefitOptions options);

    void        resetCommandsInBuffer(const MTL::IndirectCommandBuffer* buffer, NS::Range range);

    void        setArgumentTable(const MTL4::ArgumentTable* argumentTable);

    void        setComputePipelineState(const MTL::ComputePipelineState* state);

    void        setImageblockWidth(NS::UInteger width, NS::UInteger height);

    void        setThreadgroupMemoryLength(NS::UInteger length, NS::UInteger index);

    MTL::Stages stages();

    void        writeCompactedAccelerationStructureSize(const MTL::AccelerationStructure* accelerationStructure, const MTL4::BufferRange buffer);

    void        writeTimestamp(MTL4::TimestampGranularity granularity, const MTL4::CounterHeap* counterHeap, NS::UInteger index);
};

}
_MTL_INLINE void MTL4::ComputeCommandEncoder::buildAccelerationStructure(const MTL::AccelerationStructure* accelerationStructure, const MTL4::AccelerationStructureDescriptor* descriptor, const MTL4::BufferRange scratchBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(buildAccelerationStructure_descriptor_scratchBuffer_), accelerationStructure, descriptor, scratchBuffer);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::copyAccelerationStructure(const MTL::AccelerationStructure* sourceAccelerationStructure, const MTL::AccelerationStructure* destinationAccelerationStructure)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyAccelerationStructure_toAccelerationStructure_), sourceAccelerationStructure, destinationAccelerationStructure);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::copyAndCompactAccelerationStructure(const MTL::AccelerationStructure* sourceAccelerationStructure, const MTL::AccelerationStructure* destinationAccelerationStructure)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyAndCompactAccelerationStructure_toAccelerationStructure_), sourceAccelerationStructure, destinationAccelerationStructure);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::copyFromBuffer(const MTL::Buffer* sourceBuffer, NS::UInteger sourceOffset, const MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset, NS::UInteger size)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_), sourceBuffer, sourceOffset, destinationBuffer, destinationOffset, size);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::copyFromBuffer(const MTL::Buffer* sourceBuffer, NS::UInteger sourceOffset, NS::UInteger sourceBytesPerRow, NS::UInteger sourceBytesPerImage, MTL::Size sourceSize, const MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_), sourceBuffer, sourceOffset, sourceBytesPerRow, sourceBytesPerImage, sourceSize, destinationTexture, destinationSlice, destinationLevel, destinationOrigin);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::copyFromBuffer(const MTL::Buffer* sourceBuffer, NS::UInteger sourceOffset, NS::UInteger sourceBytesPerRow, NS::UInteger sourceBytesPerImage, MTL::Size sourceSize, const MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin, MTL::BlitOption options)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options_), sourceBuffer, sourceOffset, sourceBytesPerRow, sourceBytesPerImage, sourceSize, destinationTexture, destinationSlice, destinationLevel, destinationOrigin, options);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::copyFromTensor(const MTL::Tensor* sourceTensor, const MTL::TensorExtents* sourceOrigin, const MTL::TensorExtents* sourceDimensions, const MTL::Tensor* destinationTensor, const MTL::TensorExtents* destinationOrigin, const MTL::TensorExtents* destinationDimensions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyFromTensor_sourceOrigin_sourceDimensions_toTensor_destinationOrigin_destinationDimensions_), sourceTensor, sourceOrigin, sourceDimensions, destinationTensor, destinationOrigin, destinationDimensions);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::copyFromTexture(const MTL::Texture* sourceTexture, const MTL::Texture* destinationTexture)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyFromTexture_toTexture_), sourceTexture, destinationTexture);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::copyFromTexture(const MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, const MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, NS::UInteger sliceCount, NS::UInteger levelCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyFromTexture_sourceSlice_sourceLevel_toTexture_destinationSlice_destinationLevel_sliceCount_levelCount_), sourceTexture, sourceSlice, sourceLevel, destinationTexture, destinationSlice, destinationLevel, sliceCount, levelCount);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::copyFromTexture(const MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, const MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_), sourceTexture, sourceSlice, sourceLevel, sourceOrigin, sourceSize, destinationTexture, destinationSlice, destinationLevel, destinationOrigin);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::copyFromTexture(const MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, const MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset, NS::UInteger destinationBytesPerRow, NS::UInteger destinationBytesPerImage)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_), sourceTexture, sourceSlice, sourceLevel, sourceOrigin, sourceSize, destinationBuffer, destinationOffset, destinationBytesPerRow, destinationBytesPerImage);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::copyFromTexture(const MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, const MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset, NS::UInteger destinationBytesPerRow, NS::UInteger destinationBytesPerImage, MTL::BlitOption options)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options_), sourceTexture, sourceSlice, sourceLevel, sourceOrigin, sourceSize, destinationBuffer, destinationOffset, destinationBytesPerRow, destinationBytesPerImage, options);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::copyIndirectCommandBuffer(const MTL::IndirectCommandBuffer* source, NS::Range sourceRange, const MTL::IndirectCommandBuffer* destination, NS::UInteger destinationIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyIndirectCommandBuffer_sourceRange_destination_destinationIndex_), source, sourceRange, destination, destinationIndex);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::dispatchThreadgroups(MTL::Size threadgroupsPerGrid, MTL::Size threadsPerThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(dispatchThreadgroups_threadsPerThreadgroup_), threadgroupsPerGrid, threadsPerThreadgroup);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::dispatchThreadgroups(MTL::GPUAddress indirectBuffer, MTL::Size threadsPerThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(dispatchThreadgroupsWithIndirectBuffer_threadsPerThreadgroup_), indirectBuffer, threadsPerThreadgroup);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::dispatchThreads(MTL::Size threadsPerGrid, MTL::Size threadsPerThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(dispatchThreads_threadsPerThreadgroup_), threadsPerGrid, threadsPerThreadgroup);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::dispatchThreads(MTL::GPUAddress indirectBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(dispatchThreadsWithIndirectBuffer_), indirectBuffer);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::executeCommandsInBuffer(const MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::Range executionRange)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(executeCommandsInBuffer_withRange_), indirectCommandBuffer, executionRange);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::executeCommandsInBuffer(const MTL::IndirectCommandBuffer* indirectCommandbuffer, MTL::GPUAddress indirectRangeBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(executeCommandsInBuffer_indirectBuffer_), indirectCommandbuffer, indirectRangeBuffer);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::fillBuffer(const MTL::Buffer* buffer, NS::Range range, uint8_t value)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(fillBuffer_range_value_), buffer, range, value);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::generateMipmaps(const MTL::Texture* texture)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(generateMipmapsForTexture_), texture);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::optimizeContentsForCPUAccess(const MTL::Texture* texture)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(optimizeContentsForCPUAccess_), texture);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::optimizeContentsForCPUAccess(const MTL::Texture* texture, NS::UInteger slice, NS::UInteger level)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(optimizeContentsForCPUAccess_slice_level_), texture, slice, level);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::optimizeContentsForGPUAccess(const MTL::Texture* texture)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(optimizeContentsForGPUAccess_), texture);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::optimizeContentsForGPUAccess(const MTL::Texture* texture, NS::UInteger slice, NS::UInteger level)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(optimizeContentsForGPUAccess_slice_level_), texture, slice, level);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::optimizeIndirectCommandBuffer(const MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(optimizeIndirectCommandBuffer_withRange_), indirectCommandBuffer, range);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::refitAccelerationStructure(const MTL::AccelerationStructure* sourceAccelerationStructure, const MTL4::AccelerationStructureDescriptor* descriptor, const MTL::AccelerationStructure* destinationAccelerationStructure, const MTL4::BufferRange scratchBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(refitAccelerationStructure_descriptor_destination_scratchBuffer_), sourceAccelerationStructure, descriptor, destinationAccelerationStructure, scratchBuffer);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::refitAccelerationStructure(const MTL::AccelerationStructure* sourceAccelerationStructure, const MTL4::AccelerationStructureDescriptor* descriptor, const MTL::AccelerationStructure* destinationAccelerationStructure, const MTL4::BufferRange scratchBuffer, MTL::AccelerationStructureRefitOptions options)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(refitAccelerationStructure_descriptor_destination_scratchBuffer_options_), sourceAccelerationStructure, descriptor, destinationAccelerationStructure, scratchBuffer, options);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::resetCommandsInBuffer(const MTL::IndirectCommandBuffer* buffer, NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(resetCommandsInBuffer_withRange_), buffer, range);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::setArgumentTable(const MTL4::ArgumentTable* argumentTable)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setArgumentTable_), argumentTable);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::setComputePipelineState(const MTL::ComputePipelineState* state)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setComputePipelineState_), state);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::setImageblockWidth(NS::UInteger width, NS::UInteger height)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setImageblockWidth_height_), width, height);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::setThreadgroupMemoryLength(NS::UInteger length, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setThreadgroupMemoryLength_atIndex_), length, index);
}

_MTL_INLINE MTL::Stages MTL4::ComputeCommandEncoder::stages()
{
    return Object::sendMessage<MTL::Stages>(this, _MTL_PRIVATE_SEL(stages));
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::writeCompactedAccelerationStructureSize(const MTL::AccelerationStructure* accelerationStructure, const MTL4::BufferRange buffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(writeCompactedAccelerationStructureSize_toBuffer_), accelerationStructure, buffer);
}

_MTL_INLINE void MTL4::ComputeCommandEncoder::writeTimestamp(MTL4::TimestampGranularity granularity, const MTL4::CounterHeap* counterHeap, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(writeTimestampWithGranularity_intoHeap_atIndex_), granularity, counterHeap, index);
}
