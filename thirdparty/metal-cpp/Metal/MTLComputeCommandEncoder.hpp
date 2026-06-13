//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLComputeCommandEncoder.hpp
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
#include "MTLCommandBuffer.hpp"
#include "MTLCommandEncoder.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include "MTLTypes.hpp"
#include <cstdint>

namespace MTL
{
class AccelerationStructure;
class Buffer;
class ComputePipelineState;
class CounterSampleBuffer;
class Fence;
class Heap;
class IndirectCommandBuffer;
class IntersectionFunctionTable;
class Resource;
class SamplerState;
class Texture;
class VisibleFunctionTable;

struct DispatchThreadgroupsIndirectArguments
{
    uint32_t threadgroupsPerGrid[3];
} _MTL_PACKED;

struct DispatchThreadsIndirectArguments
{
    uint32_t threadsPerGrid[3];
    uint32_t threadsPerThreadgroup[3];
} _MTL_PACKED;

struct StageInRegionIndirectArguments
{
    uint32_t stageInOrigin[3];
    uint32_t stageInSize[3];
} _MTL_PACKED;

class ComputeCommandEncoder : public NS::Referencing<ComputeCommandEncoder, CommandEncoder>
{
public:
    void         dispatchThreadgroups(MTL::Size threadgroupsPerGrid, MTL::Size threadsPerThreadgroup);
    void         dispatchThreadgroups(const MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset, MTL::Size threadsPerThreadgroup);

    void         dispatchThreads(MTL::Size threadsPerGrid, MTL::Size threadsPerThreadgroup);

    DispatchType dispatchType() const;

    void         executeCommandsInBuffer(const MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::Range executionRange);
    void         executeCommandsInBuffer(const MTL::IndirectCommandBuffer* indirectCommandbuffer, const MTL::Buffer* indirectRangeBuffer, NS::UInteger indirectBufferOffset);

    void         memoryBarrier(MTL::BarrierScope scope);
    void         memoryBarrier(const MTL::Resource* const resources[], NS::UInteger count);

    void         sampleCountersInBuffer(const MTL::CounterSampleBuffer* sampleBuffer, NS::UInteger sampleIndex, bool barrier);

    void         setAccelerationStructure(const MTL::AccelerationStructure* accelerationStructure, NS::UInteger bufferIndex);

    void         setBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);
    void         setBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger stride, NS::UInteger index);
    void         setBufferOffset(NS::UInteger offset, NS::UInteger index);
    void         setBufferOffset(NS::UInteger offset, NS::UInteger stride, NS::UInteger index);

    void         setBuffers(const MTL::Buffer* const buffers[], const NS::UInteger offsets[], NS::Range range);
    void         setBuffers(const MTL::Buffer* const buffers[], const NS::UInteger* offsets, const NS::UInteger* strides, NS::Range range);

    void         setBytes(const void* bytes, NS::UInteger length, NS::UInteger index);
    void         setBytes(const void* bytes, NS::UInteger length, NS::UInteger stride, NS::UInteger index);

    void         setComputePipelineState(const MTL::ComputePipelineState* state);

    void         setImageblockWidth(NS::UInteger width, NS::UInteger height);

    void         setIntersectionFunctionTable(const MTL::IntersectionFunctionTable* intersectionFunctionTable, NS::UInteger bufferIndex);
    void         setIntersectionFunctionTables(const MTL::IntersectionFunctionTable* const intersectionFunctionTables[], NS::Range range);

    void         setSamplerState(const MTL::SamplerState* sampler, NS::UInteger index);
    void         setSamplerState(const MTL::SamplerState* sampler, float lodMinClamp, float lodMaxClamp, NS::UInteger index);
    void         setSamplerStates(const MTL::SamplerState* const samplers[], NS::Range range);
    void         setSamplerStates(const MTL::SamplerState* const samplers[], const float lodMinClamps[], const float lodMaxClamps[], NS::Range range);

    void         setStageInRegion(MTL::Region region);
    void         setStageInRegion(const MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset);

    void         setTexture(const MTL::Texture* texture, NS::UInteger index);
    void         setTextures(const MTL::Texture* const textures[], NS::Range range);

    void         setThreadgroupMemoryLength(NS::UInteger length, NS::UInteger index);

    void         setVisibleFunctionTable(const MTL::VisibleFunctionTable* visibleFunctionTable, NS::UInteger bufferIndex);
    void         setVisibleFunctionTables(const MTL::VisibleFunctionTable* const visibleFunctionTables[], NS::Range range);

    void         updateFence(const MTL::Fence* fence);

    void         useHeap(const MTL::Heap* heap);
    void         useHeaps(const MTL::Heap* const heaps[], NS::UInteger count);

    void         useResource(const MTL::Resource* resource, MTL::ResourceUsage usage);
    void         useResources(const MTL::Resource* const resources[], NS::UInteger count, MTL::ResourceUsage usage);

    void         waitForFence(const MTL::Fence* fence);
};

}

_MTL_INLINE void MTL::ComputeCommandEncoder::dispatchThreadgroups(MTL::Size threadgroupsPerGrid, MTL::Size threadsPerThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(dispatchThreadgroups_threadsPerThreadgroup_), threadgroupsPerGrid, threadsPerThreadgroup);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::dispatchThreadgroups(const MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset, MTL::Size threadsPerThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup_), indirectBuffer, indirectBufferOffset, threadsPerThreadgroup);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::dispatchThreads(MTL::Size threadsPerGrid, MTL::Size threadsPerThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(dispatchThreads_threadsPerThreadgroup_), threadsPerGrid, threadsPerThreadgroup);
}

_MTL_INLINE MTL::DispatchType MTL::ComputeCommandEncoder::dispatchType() const
{
    return Object::sendMessage<MTL::DispatchType>(this, _MTL_PRIVATE_SEL(dispatchType));
}

_MTL_INLINE void MTL::ComputeCommandEncoder::executeCommandsInBuffer(const MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::Range executionRange)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(executeCommandsInBuffer_withRange_), indirectCommandBuffer, executionRange);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::executeCommandsInBuffer(const MTL::IndirectCommandBuffer* indirectCommandbuffer, const MTL::Buffer* indirectRangeBuffer, NS::UInteger indirectBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(executeCommandsInBuffer_indirectBuffer_indirectBufferOffset_), indirectCommandbuffer, indirectRangeBuffer, indirectBufferOffset);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::memoryBarrier(MTL::BarrierScope scope)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(memoryBarrierWithScope_), scope);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::memoryBarrier(const MTL::Resource* const resources[], NS::UInteger count)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(memoryBarrierWithResources_count_), resources, count);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::sampleCountersInBuffer(const MTL::CounterSampleBuffer* sampleBuffer, NS::UInteger sampleIndex, bool barrier)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(sampleCountersInBuffer_atSampleIndex_withBarrier_), sampleBuffer, sampleIndex, barrier);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setAccelerationStructure(const MTL::AccelerationStructure* accelerationStructure, NS::UInteger bufferIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setAccelerationStructure_atBufferIndex_), accelerationStructure, bufferIndex);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBuffer_offset_atIndex_), buffer, offset, index);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger stride, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBuffer_offset_attributeStride_atIndex_), buffer, offset, stride, index);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setBufferOffset(NS::UInteger offset, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBufferOffset_atIndex_), offset, index);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setBufferOffset(NS::UInteger offset, NS::UInteger stride, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBufferOffset_attributeStride_atIndex_), offset, stride, index);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setBuffers(const MTL::Buffer* const buffers[], const NS::UInteger offsets[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBuffers_offsets_withRange_), buffers, offsets, range);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setBuffers(const MTL::Buffer* const buffers[], const NS::UInteger* offsets, const NS::UInteger* strides, NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBuffers_offsets_attributeStrides_withRange_), buffers, offsets, strides, range);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setBytes(const void* bytes, NS::UInteger length, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBytes_length_atIndex_), bytes, length, index);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setBytes(const void* bytes, NS::UInteger length, NS::UInteger stride, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBytes_length_attributeStride_atIndex_), bytes, length, stride, index);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setComputePipelineState(const MTL::ComputePipelineState* state)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setComputePipelineState_), state);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setImageblockWidth(NS::UInteger width, NS::UInteger height)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setImageblockWidth_height_), width, height);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setIntersectionFunctionTable(const MTL::IntersectionFunctionTable* intersectionFunctionTable, NS::UInteger bufferIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIntersectionFunctionTable_atBufferIndex_), intersectionFunctionTable, bufferIndex);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setIntersectionFunctionTables(const MTL::IntersectionFunctionTable* const intersectionFunctionTables[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIntersectionFunctionTables_withBufferRange_), intersectionFunctionTables, range);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setSamplerState(const MTL::SamplerState* sampler, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSamplerState_atIndex_), sampler, index);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setSamplerState(const MTL::SamplerState* sampler, float lodMinClamp, float lodMaxClamp, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSamplerState_lodMinClamp_lodMaxClamp_atIndex_), sampler, lodMinClamp, lodMaxClamp, index);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setSamplerStates(const MTL::SamplerState* const samplers[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSamplerStates_withRange_), samplers, range);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setSamplerStates(const MTL::SamplerState* const samplers[], const float lodMinClamps[], const float lodMaxClamps[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSamplerStates_lodMinClamps_lodMaxClamps_withRange_), samplers, lodMinClamps, lodMaxClamps, range);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setStageInRegion(MTL::Region region)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStageInRegion_), region);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setStageInRegion(const MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStageInRegionWithIndirectBuffer_indirectBufferOffset_), indirectBuffer, indirectBufferOffset);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setTexture(const MTL::Texture* texture, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTexture_atIndex_), texture, index);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setTextures(const MTL::Texture* const textures[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTextures_withRange_), textures, range);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setThreadgroupMemoryLength(NS::UInteger length, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setThreadgroupMemoryLength_atIndex_), length, index);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setVisibleFunctionTable(const MTL::VisibleFunctionTable* visibleFunctionTable, NS::UInteger bufferIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVisibleFunctionTable_atBufferIndex_), visibleFunctionTable, bufferIndex);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setVisibleFunctionTables(const MTL::VisibleFunctionTable* const visibleFunctionTables[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVisibleFunctionTables_withBufferRange_), visibleFunctionTables, range);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::updateFence(const MTL::Fence* fence)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(updateFence_), fence);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::useHeap(const MTL::Heap* heap)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(useHeap_), heap);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::useHeaps(const MTL::Heap* const heaps[], NS::UInteger count)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(useHeaps_count_), heaps, count);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::useResource(const MTL::Resource* resource, MTL::ResourceUsage usage)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(useResource_usage_), resource, usage);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::useResources(const MTL::Resource* const resources[], NS::UInteger count, MTL::ResourceUsage usage)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(useResources_count_usage_), resources, count, usage);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::waitForFence(const MTL::Fence* fence)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(waitForFence_), fence);
}
