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
    using BarrierScope = NS::UInteger;
    enum DispatchType : NS::UInteger;
    using ResourceUsage = NS::UInteger;
}

namespace MTL
{

class ComputeCommandEncoder : public NS::Referencing<ComputeCommandEncoder, MTL::CommandEncoder>
{
public:
    void              dispatchThreadgroups(MTL::Size threadgroupsPerGrid, MTL::Size threadsPerThreadgroup);
    void              dispatchThreadgroups(MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset, MTL::Size threadsPerThreadgroup);
    void              dispatchThreads(MTL::Size threadsPerGrid, MTL::Size threadsPerThreadgroup);
    MTL::DispatchType dispatchType() const;
    void              executeCommandsInBuffer(MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::Range executionRange);
    void              executeCommandsInBuffer(MTL::IndirectCommandBuffer* indirectCommandbuffer, MTL::Buffer* indirectRangeBuffer, NS::UInteger indirectBufferOffset);
    void              memoryBarrier(MTL::BarrierScope scope);
    void              memoryBarrier(const MTL::Resource* const * resources, NS::UInteger count);
    void              sampleCountersInBuffer(MTL::CounterSampleBuffer* sampleBuffer, NS::UInteger sampleIndex, bool barrier);
    void              setAccelerationStructure(MTL::AccelerationStructure* accelerationStructure, NS::UInteger bufferIndex);
    void              setBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);
    void              setBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger stride, NS::UInteger index);
    void              setBufferOffset(NS::UInteger offset, NS::UInteger index);
    void              setBufferOffset(NS::UInteger offset, NS::UInteger stride, NS::UInteger index);
    void              setBuffers(const MTL::Buffer* const * buffers, const NS::UInteger * offsets, NS::Range range);
    void              setBuffers(const MTL::Buffer* const * buffers, const NS::UInteger * offsets, const NS::UInteger * strides, NS::Range range);
    void              setBytes(const void * bytes, NS::UInteger length, NS::UInteger index);
    void              setBytes(const void * bytes, NS::UInteger length, NS::UInteger stride, NS::UInteger index);
    void              setComputePipelineState(MTL::ComputePipelineState* state);
    void              setImageblockWidth(NS::UInteger width, NS::UInteger height);
    void              setIntersectionFunctionTable(MTL::IntersectionFunctionTable* intersectionFunctionTable, NS::UInteger bufferIndex);
    void              setIntersectionFunctionTables(const MTL::IntersectionFunctionTable* const * intersectionFunctionTables, NS::Range range);
    void              setSamplerState(MTL::SamplerState* sampler, NS::UInteger index);
    void              setSamplerState(MTL::SamplerState* sampler, float lodMinClamp, float lodMaxClamp, NS::UInteger index);
    void              setSamplerStates(const MTL::SamplerState* const * samplers, NS::Range range);
    void              setSamplerStates(const MTL::SamplerState* const * samplers, const float * lodMinClamps, const float * lodMaxClamps, NS::Range range);
    void              setStageInRegion(MTL::Region region);
    void              setStageInRegion(MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset);
    void              setTexture(MTL::Texture* texture, NS::UInteger index);
    void              setTextures(const MTL::Texture* const * textures, NS::Range range);
    void              setThreadgroupMemoryLength(NS::UInteger length, NS::UInteger index);
    void              setVisibleFunctionTable(MTL::VisibleFunctionTable* visibleFunctionTable, NS::UInteger bufferIndex);
    void              setVisibleFunctionTables(const MTL::VisibleFunctionTable* const * visibleFunctionTables, NS::Range range);
    void              updateFence(MTL::Fence* fence);
    void              useHeap(MTL::Heap* heap);
    void              useHeaps(const MTL::Heap* const * heaps, NS::UInteger count);
    void              useResource(MTL::Resource* resource, MTL::ResourceUsage usage);
    void              useResources(const MTL::Resource* const * resources, NS::UInteger count, MTL::ResourceUsage usage);
    void              waitForFence(MTL::Fence* fence);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLComputeCommandEncoder;

_MTL_INLINE MTL::DispatchType MTL::ComputeCommandEncoder::dispatchType() const
{
    return _MTL_msg_MTL__DispatchType_dispatchType((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setComputePipelineState(MTL::ComputePipelineState* state)
{
    _MTL_msg_v_setComputePipelineState__MTL__ComputePipelineStatep((const void*)this, nullptr, state);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setBytes(const void * bytes, NS::UInteger length, NS::UInteger index)
{
    _MTL_msg_v_setBytes_length_atIndex__constvoidp_NS__UInteger_NS__UInteger((const void*)this, nullptr, bytes, length, index);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    _MTL_msg_v_setBuffer_offset_atIndex__MTL__Bufferp_NS__UInteger_NS__UInteger((const void*)this, nullptr, buffer, offset, index);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setBufferOffset(NS::UInteger offset, NS::UInteger index)
{
    _MTL_msg_v_setBufferOffset_atIndex__NS__UInteger_NS__UInteger((const void*)this, nullptr, offset, index);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setBuffers(const MTL::Buffer* const * buffers, const NS::UInteger * offsets, NS::Range range)
{
    _MTL_msg_v_setBuffers_offsets_withRange__constMTL__Bufferpconstp_constNS__UIntegerp_NS__Range((const void*)this, nullptr, buffers, offsets, range);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger stride, NS::UInteger index)
{
    _MTL_msg_v_setBuffer_offset_attributeStride_atIndex__MTL__Bufferp_NS__UInteger_NS__UInteger_NS__UInteger((const void*)this, nullptr, buffer, offset, stride, index);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setBuffers(const MTL::Buffer* const * buffers, const NS::UInteger * offsets, const NS::UInteger * strides, NS::Range range)
{
    _MTL_msg_v_setBuffers_offsets_attributeStrides_withRange__constMTL__Bufferpconstp_constNS__UIntegerp_constNS__UIntegerp_NS__Range((const void*)this, nullptr, buffers, offsets, strides, range);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setBufferOffset(NS::UInteger offset, NS::UInteger stride, NS::UInteger index)
{
    _MTL_msg_v_setBufferOffset_attributeStride_atIndex__NS__UInteger_NS__UInteger_NS__UInteger((const void*)this, nullptr, offset, stride, index);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setBytes(const void * bytes, NS::UInteger length, NS::UInteger stride, NS::UInteger index)
{
    _MTL_msg_v_setBytes_length_attributeStride_atIndex__constvoidp_NS__UInteger_NS__UInteger_NS__UInteger((const void*)this, nullptr, bytes, length, stride, index);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setVisibleFunctionTable(MTL::VisibleFunctionTable* visibleFunctionTable, NS::UInteger bufferIndex)
{
    _MTL_msg_v_setVisibleFunctionTable_atBufferIndex__MTL__VisibleFunctionTablep_NS__UInteger((const void*)this, nullptr, visibleFunctionTable, bufferIndex);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setVisibleFunctionTables(const MTL::VisibleFunctionTable* const * visibleFunctionTables, NS::Range range)
{
    _MTL_msg_v_setVisibleFunctionTables_withBufferRange__constMTL__VisibleFunctionTablepconstp_NS__Range((const void*)this, nullptr, visibleFunctionTables, range);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setIntersectionFunctionTable(MTL::IntersectionFunctionTable* intersectionFunctionTable, NS::UInteger bufferIndex)
{
    _MTL_msg_v_setIntersectionFunctionTable_atBufferIndex__MTL__IntersectionFunctionTablep_NS__UInteger((const void*)this, nullptr, intersectionFunctionTable, bufferIndex);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setIntersectionFunctionTables(const MTL::IntersectionFunctionTable* const * intersectionFunctionTables, NS::Range range)
{
    _MTL_msg_v_setIntersectionFunctionTables_withBufferRange__constMTL__IntersectionFunctionTablepconstp_NS__Range((const void*)this, nullptr, intersectionFunctionTables, range);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setAccelerationStructure(MTL::AccelerationStructure* accelerationStructure, NS::UInteger bufferIndex)
{
    _MTL_msg_v_setAccelerationStructure_atBufferIndex__MTL__AccelerationStructurep_NS__UInteger((const void*)this, nullptr, accelerationStructure, bufferIndex);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setTexture(MTL::Texture* texture, NS::UInteger index)
{
    _MTL_msg_v_setTexture_atIndex__MTL__Texturep_NS__UInteger((const void*)this, nullptr, texture, index);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setTextures(const MTL::Texture* const * textures, NS::Range range)
{
    _MTL_msg_v_setTextures_withRange__constMTL__Texturepconstp_NS__Range((const void*)this, nullptr, textures, range);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setSamplerState(MTL::SamplerState* sampler, NS::UInteger index)
{
    _MTL_msg_v_setSamplerState_atIndex__MTL__SamplerStatep_NS__UInteger((const void*)this, nullptr, sampler, index);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setSamplerStates(const MTL::SamplerState* const * samplers, NS::Range range)
{
    _MTL_msg_v_setSamplerStates_withRange__constMTL__SamplerStatepconstp_NS__Range((const void*)this, nullptr, samplers, range);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setSamplerState(MTL::SamplerState* sampler, float lodMinClamp, float lodMaxClamp, NS::UInteger index)
{
    _MTL_msg_v_setSamplerState_lodMinClamp_lodMaxClamp_atIndex__MTL__SamplerStatep_float_float_NS__UInteger((const void*)this, nullptr, sampler, lodMinClamp, lodMaxClamp, index);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setSamplerStates(const MTL::SamplerState* const * samplers, const float * lodMinClamps, const float * lodMaxClamps, NS::Range range)
{
    _MTL_msg_v_setSamplerStates_lodMinClamps_lodMaxClamps_withRange__constMTL__SamplerStatepconstp_constfloatp_constfloatp_NS__Range((const void*)this, nullptr, samplers, lodMinClamps, lodMaxClamps, range);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setThreadgroupMemoryLength(NS::UInteger length, NS::UInteger index)
{
    _MTL_msg_v_setThreadgroupMemoryLength_atIndex__NS__UInteger_NS__UInteger((const void*)this, nullptr, length, index);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setImageblockWidth(NS::UInteger width, NS::UInteger height)
{
    _MTL_msg_v_setImageblockWidth_height__NS__UInteger_NS__UInteger((const void*)this, nullptr, width, height);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setStageInRegion(MTL::Region region)
{
    _MTL_msg_v_setStageInRegion__MTL__Region((const void*)this, nullptr, region);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::setStageInRegion(MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset)
{
    _MTL_msg_v_setStageInRegionWithIndirectBuffer_indirectBufferOffset__MTL__Bufferp_NS__UInteger((const void*)this, nullptr, indirectBuffer, indirectBufferOffset);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::dispatchThreadgroups(MTL::Size threadgroupsPerGrid, MTL::Size threadsPerThreadgroup)
{
    _MTL_msg_v_dispatchThreadgroups_threadsPerThreadgroup__MTL__Size_MTL__Size((const void*)this, nullptr, threadgroupsPerGrid, threadsPerThreadgroup);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::dispatchThreadgroups(MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset, MTL::Size threadsPerThreadgroup)
{
    _MTL_msg_v_dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup__MTL__Bufferp_NS__UInteger_MTL__Size((const void*)this, nullptr, indirectBuffer, indirectBufferOffset, threadsPerThreadgroup);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::dispatchThreads(MTL::Size threadsPerGrid, MTL::Size threadsPerThreadgroup)
{
    _MTL_msg_v_dispatchThreads_threadsPerThreadgroup__MTL__Size_MTL__Size((const void*)this, nullptr, threadsPerGrid, threadsPerThreadgroup);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::updateFence(MTL::Fence* fence)
{
    _MTL_msg_v_updateFence__MTL__Fencep((const void*)this, nullptr, fence);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::waitForFence(MTL::Fence* fence)
{
    _MTL_msg_v_waitForFence__MTL__Fencep((const void*)this, nullptr, fence);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::useResource(MTL::Resource* resource, MTL::ResourceUsage usage)
{
    _MTL_msg_v_useResource_usage__MTL__Resourcep_MTL__ResourceUsage((const void*)this, nullptr, resource, usage);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::useResources(const MTL::Resource* const * resources, NS::UInteger count, MTL::ResourceUsage usage)
{
    _MTL_msg_v_useResources_count_usage__constMTL__Resourcepconstp_NS__UInteger_MTL__ResourceUsage((const void*)this, nullptr, resources, count, usage);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::useHeap(MTL::Heap* heap)
{
    _MTL_msg_v_useHeap__MTL__Heapp((const void*)this, nullptr, heap);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::useHeaps(const MTL::Heap* const * heaps, NS::UInteger count)
{
    _MTL_msg_v_useHeaps_count__constMTL__Heappconstp_NS__UInteger((const void*)this, nullptr, heaps, count);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::executeCommandsInBuffer(MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::Range executionRange)
{
    _MTL_msg_v_executeCommandsInBuffer_withRange__MTL__IndirectCommandBufferp_NS__Range((const void*)this, nullptr, indirectCommandBuffer, executionRange);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::executeCommandsInBuffer(MTL::IndirectCommandBuffer* indirectCommandbuffer, MTL::Buffer* indirectRangeBuffer, NS::UInteger indirectBufferOffset)
{
    _MTL_msg_v_executeCommandsInBuffer_indirectBuffer_indirectBufferOffset__MTL__IndirectCommandBufferp_MTL__Bufferp_NS__UInteger((const void*)this, nullptr, indirectCommandbuffer, indirectRangeBuffer, indirectBufferOffset);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::memoryBarrier(MTL::BarrierScope scope)
{
    _MTL_msg_v_memoryBarrierWithScope__MTL__BarrierScope((const void*)this, nullptr, scope);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::memoryBarrier(const MTL::Resource* const * resources, NS::UInteger count)
{
    _MTL_msg_v_memoryBarrierWithResources_count__constMTL__Resourcepconstp_NS__UInteger((const void*)this, nullptr, resources, count);
}

_MTL_INLINE void MTL::ComputeCommandEncoder::sampleCountersInBuffer(MTL::CounterSampleBuffer* sampleBuffer, NS::UInteger sampleIndex, bool barrier)
{
    _MTL_msg_v_sampleCountersInBuffer_atSampleIndex_withBarrier__MTL__CounterSampleBufferp_NS__UInteger_bool((const void*)this, nullptr, sampleBuffer, sampleIndex, barrier);
}
