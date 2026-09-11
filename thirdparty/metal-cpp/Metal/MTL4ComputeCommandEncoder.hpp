#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTL4CommandEncoder.hpp"
#include "MTLStructs.hpp"

namespace MTL {
    class AccelerationStructure;
    class Buffer;
    class ComputePipelineState;
    class IndirectCommandBuffer;
    class Tensor;
    class TensorExtents;
    class Texture;
    using AccelerationStructureRefitOptions = NS::UInteger;
    using BlitOption = NS::UInteger;
    using Stages = NS::UInteger;
}
namespace MTL4 {
    class AccelerationStructureDescriptor;
    class ArgumentTable;
    class CounterHeap;
    enum TimestampGranularity : NS::Integer;
}

namespace MTL4
{

class ComputeCommandEncoder : public NS::Referencing<ComputeCommandEncoder, MTL4::CommandEncoder>
{
public:
    void        buildAccelerationStructure(MTL::AccelerationStructure* accelerationStructure, MTL4::AccelerationStructureDescriptor* descriptor, MTL4::BufferRange scratchBuffer);
    void        copyAccelerationStructure(MTL::AccelerationStructure* sourceAccelerationStructure, MTL::AccelerationStructure* destinationAccelerationStructure);
    void        copyAndCompactAccelerationStructure(MTL::AccelerationStructure* sourceAccelerationStructure, MTL::AccelerationStructure* destinationAccelerationStructure);
    void        copyFromBuffer(MTL::Buffer* sourceBuffer, NS::UInteger sourceOffset, MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset, NS::UInteger size);
    void        copyFromBuffer(MTL::Buffer* sourceBuffer, NS::UInteger sourceOffset, NS::UInteger sourceBytesPerRow, NS::UInteger sourceBytesPerImage, MTL::Size sourceSize, MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin);
    void        copyFromBuffer(MTL::Buffer* sourceBuffer, NS::UInteger sourceOffset, NS::UInteger sourceBytesPerRow, NS::UInteger sourceBytesPerImage, MTL::Size sourceSize, MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin, MTL::BlitOption options);
    void        copyFromTensor(MTL::Tensor* sourceTensor, MTL::TensorExtents* sourceOrigin, MTL::TensorExtents* sourceDimensions, MTL::Tensor* destinationTensor, MTL::TensorExtents* destinationOrigin, MTL::TensorExtents* destinationDimensions);
    void        copyFromTexture(MTL::Texture* sourceTexture, MTL::Texture* destinationTexture);
    void        copyFromTexture(MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, NS::UInteger sliceCount, NS::UInteger levelCount);
    void        copyFromTexture(MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin);
    void        copyFromTexture(MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset, NS::UInteger destinationBytesPerRow, NS::UInteger destinationBytesPerImage);
    void        copyFromTexture(MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset, NS::UInteger destinationBytesPerRow, NS::UInteger destinationBytesPerImage, MTL::BlitOption options);
    void        copyIndirectCommandBuffer(MTL::IndirectCommandBuffer* source, NS::Range sourceRange, MTL::IndirectCommandBuffer* destination, NS::UInteger destinationIndex);
    void        dispatchThreadgroups(MTL::Size threadgroupsPerGrid, MTL::Size threadsPerThreadgroup);
    void        dispatchThreadgroups(MTL::GPUAddress indirectBuffer, MTL::Size threadsPerThreadgroup);
    void        dispatchThreads(MTL::Size threadsPerGrid, MTL::Size threadsPerThreadgroup);
    void        dispatchThreads(MTL::GPUAddress indirectBuffer);
    void        executeCommandsInBuffer(MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::Range executionRange);
    void        executeCommandsInBuffer(MTL::IndirectCommandBuffer* indirectCommandbuffer, MTL::GPUAddress indirectRangeBuffer);
    void        fillBuffer(MTL::Buffer* buffer, NS::Range range, uint8_t value);
    void        generateMipmaps(MTL::Texture* texture);
    void        optimizeContents(MTL::Texture* texture);
    void        optimizeContents(MTL::Texture* texture, NS::UInteger slice, NS::UInteger level);
    void        optimizeIndirectCommandBuffer(MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::Range range);
    void        refitAccelerationStructure(MTL::AccelerationStructure* sourceAccelerationStructure, MTL4::AccelerationStructureDescriptor* descriptor, MTL::AccelerationStructure* destinationAccelerationStructure, MTL4::BufferRange scratchBuffer);
    void        refitAccelerationStructure(MTL::AccelerationStructure* sourceAccelerationStructure, MTL4::AccelerationStructureDescriptor* descriptor, MTL::AccelerationStructure* destinationAccelerationStructure, MTL4::BufferRange scratchBuffer, MTL::AccelerationStructureRefitOptions options);
    void        resetCommandsInBuffer(MTL::IndirectCommandBuffer* buffer, NS::Range range);
    void        setArgumentTable(MTL4::ArgumentTable* argumentTable);
    void        setComputePipelineState(MTL::ComputePipelineState* state);
    void        setImageblockWidth(NS::UInteger width, NS::UInteger height);
    void        setThreadgroupMemoryLength(NS::UInteger length, NS::UInteger index);
    MTL::Stages stages();
    void        writeCompactedAccelerationStructureSize(MTL::AccelerationStructure* accelerationStructure, MTL4::BufferRange buffer);
    void        writeTimestamp(MTL4::TimestampGranularity granularity, MTL4::CounterHeap* counterHeap, NS::UInteger index);

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4ComputeCommandEncoder;

_MTL4_INLINE MTL::Stages MTL4::ComputeCommandEncoder::stages()
{
    return _MTL4_msg_MTL__Stages_stages((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::setComputePipelineState(MTL::ComputePipelineState* state)
{
    _MTL4_msg_v_setComputePipelineState__MTL__ComputePipelineStatep((const void*)this, nullptr, state);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::setThreadgroupMemoryLength(NS::UInteger length, NS::UInteger index)
{
    _MTL4_msg_v_setThreadgroupMemoryLength_atIndex__NS__UInteger_NS__UInteger((const void*)this, nullptr, length, index);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::setImageblockWidth(NS::UInteger width, NS::UInteger height)
{
    _MTL4_msg_v_setImageblockWidth_height__NS__UInteger_NS__UInteger((const void*)this, nullptr, width, height);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::dispatchThreads(MTL::Size threadsPerGrid, MTL::Size threadsPerThreadgroup)
{
    _MTL4_msg_v_dispatchThreads_threadsPerThreadgroup__MTL__Size_MTL__Size((const void*)this, nullptr, threadsPerGrid, threadsPerThreadgroup);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::dispatchThreadgroups(MTL::Size threadgroupsPerGrid, MTL::Size threadsPerThreadgroup)
{
    _MTL4_msg_v_dispatchThreadgroups_threadsPerThreadgroup__MTL__Size_MTL__Size((const void*)this, nullptr, threadgroupsPerGrid, threadsPerThreadgroup);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::dispatchThreadgroups(MTL::GPUAddress indirectBuffer, MTL::Size threadsPerThreadgroup)
{
    _MTL4_msg_v_dispatchThreadgroupsWithIndirectBuffer_threadsPerThreadgroup__MTL__GPUAddress_MTL__Size((const void*)this, nullptr, indirectBuffer, threadsPerThreadgroup);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::dispatchThreads(MTL::GPUAddress indirectBuffer)
{
    _MTL4_msg_v_dispatchThreadsWithIndirectBuffer__MTL__GPUAddress((const void*)this, nullptr, indirectBuffer);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::executeCommandsInBuffer(MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::Range executionRange)
{
    _MTL4_msg_v_executeCommandsInBuffer_withRange__MTL__IndirectCommandBufferp_NS__Range((const void*)this, nullptr, indirectCommandBuffer, executionRange);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::executeCommandsInBuffer(MTL::IndirectCommandBuffer* indirectCommandbuffer, MTL::GPUAddress indirectRangeBuffer)
{
    _MTL4_msg_v_executeCommandsInBuffer_indirectBuffer__MTL__IndirectCommandBufferp_MTL__GPUAddress((const void*)this, nullptr, indirectCommandbuffer, indirectRangeBuffer);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::copyFromTexture(MTL::Texture* sourceTexture, MTL::Texture* destinationTexture)
{
    _MTL4_msg_v_copyFromTexture_toTexture__MTL__Texturep_MTL__Texturep((const void*)this, nullptr, sourceTexture, destinationTexture);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::copyFromTexture(MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, NS::UInteger sliceCount, NS::UInteger levelCount)
{
    _MTL4_msg_v_copyFromTexture_sourceSlice_sourceLevel_toTexture_destinationSlice_destinationLevel_sliceCount_levelCount__MTL__Texturep_NS__UInteger_NS__UInteger_MTL__Texturep_NS__UInteger_NS__UInteger_NS__UInteger_NS__UInteger((const void*)this, nullptr, sourceTexture, sourceSlice, sourceLevel, destinationTexture, destinationSlice, destinationLevel, sliceCount, levelCount);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::copyFromTexture(MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin)
{
    _MTL4_msg_v_copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin__MTL__Texturep_NS__UInteger_NS__UInteger_MTL__Origin_MTL__Size_MTL__Texturep_NS__UInteger_NS__UInteger_MTL__Origin((const void*)this, nullptr, sourceTexture, sourceSlice, sourceLevel, sourceOrigin, sourceSize, destinationTexture, destinationSlice, destinationLevel, destinationOrigin);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::copyFromTexture(MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset, NS::UInteger destinationBytesPerRow, NS::UInteger destinationBytesPerImage)
{
    _MTL4_msg_v_copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage__MTL__Texturep_NS__UInteger_NS__UInteger_MTL__Origin_MTL__Size_MTL__Bufferp_NS__UInteger_NS__UInteger_NS__UInteger((const void*)this, nullptr, sourceTexture, sourceSlice, sourceLevel, sourceOrigin, sourceSize, destinationBuffer, destinationOffset, destinationBytesPerRow, destinationBytesPerImage);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::copyFromTexture(MTL::Texture* sourceTexture, NS::UInteger sourceSlice, NS::UInteger sourceLevel, MTL::Origin sourceOrigin, MTL::Size sourceSize, MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset, NS::UInteger destinationBytesPerRow, NS::UInteger destinationBytesPerImage, MTL::BlitOption options)
{
    _MTL4_msg_v_copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options__MTL__Texturep_NS__UInteger_NS__UInteger_MTL__Origin_MTL__Size_MTL__Bufferp_NS__UInteger_NS__UInteger_NS__UInteger_MTL__BlitOption((const void*)this, nullptr, sourceTexture, sourceSlice, sourceLevel, sourceOrigin, sourceSize, destinationBuffer, destinationOffset, destinationBytesPerRow, destinationBytesPerImage, options);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::copyFromBuffer(MTL::Buffer* sourceBuffer, NS::UInteger sourceOffset, MTL::Buffer* destinationBuffer, NS::UInteger destinationOffset, NS::UInteger size)
{
    _MTL4_msg_v_copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size__MTL__Bufferp_NS__UInteger_MTL__Bufferp_NS__UInteger_NS__UInteger((const void*)this, nullptr, sourceBuffer, sourceOffset, destinationBuffer, destinationOffset, size);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::copyFromBuffer(MTL::Buffer* sourceBuffer, NS::UInteger sourceOffset, NS::UInteger sourceBytesPerRow, NS::UInteger sourceBytesPerImage, MTL::Size sourceSize, MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin)
{
    _MTL4_msg_v_copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin__MTL__Bufferp_NS__UInteger_NS__UInteger_NS__UInteger_MTL__Size_MTL__Texturep_NS__UInteger_NS__UInteger_MTL__Origin((const void*)this, nullptr, sourceBuffer, sourceOffset, sourceBytesPerRow, sourceBytesPerImage, sourceSize, destinationTexture, destinationSlice, destinationLevel, destinationOrigin);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::copyFromBuffer(MTL::Buffer* sourceBuffer, NS::UInteger sourceOffset, NS::UInteger sourceBytesPerRow, NS::UInteger sourceBytesPerImage, MTL::Size sourceSize, MTL::Texture* destinationTexture, NS::UInteger destinationSlice, NS::UInteger destinationLevel, MTL::Origin destinationOrigin, MTL::BlitOption options)
{
    _MTL4_msg_v_copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options__MTL__Bufferp_NS__UInteger_NS__UInteger_NS__UInteger_MTL__Size_MTL__Texturep_NS__UInteger_NS__UInteger_MTL__Origin_MTL__BlitOption((const void*)this, nullptr, sourceBuffer, sourceOffset, sourceBytesPerRow, sourceBytesPerImage, sourceSize, destinationTexture, destinationSlice, destinationLevel, destinationOrigin, options);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::copyFromTensor(MTL::Tensor* sourceTensor, MTL::TensorExtents* sourceOrigin, MTL::TensorExtents* sourceDimensions, MTL::Tensor* destinationTensor, MTL::TensorExtents* destinationOrigin, MTL::TensorExtents* destinationDimensions)
{
    _MTL4_msg_v_copyFromTensor_sourceOrigin_sourceDimensions_toTensor_destinationOrigin_destinationDimensions__MTL__Tensorp_MTL__TensorExtentsp_MTL__TensorExtentsp_MTL__Tensorp_MTL__TensorExtentsp_MTL__TensorExtentsp((const void*)this, nullptr, sourceTensor, sourceOrigin, sourceDimensions, destinationTensor, destinationOrigin, destinationDimensions);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::generateMipmaps(MTL::Texture* texture)
{
    _MTL4_msg_v_generateMipmapsForTexture__MTL__Texturep((const void*)this, nullptr, texture);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::fillBuffer(MTL::Buffer* buffer, NS::Range range, uint8_t value)
{
    _MTL4_msg_v_fillBuffer_range_value__MTL__Bufferp_NS__Range_uint8_t((const void*)this, nullptr, buffer, range, value);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::optimizeContents(MTL::Texture* texture)
{
    _MTL4_msg_v_optimizeContentsForGPUAccess__MTL__Texturep((const void*)this, nullptr, texture);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::optimizeContents(MTL::Texture* texture, NS::UInteger slice, NS::UInteger level)
{
    _MTL4_msg_v_optimizeContentsForGPUAccess_slice_level__MTL__Texturep_NS__UInteger_NS__UInteger((const void*)this, nullptr, texture, slice, level);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::resetCommandsInBuffer(MTL::IndirectCommandBuffer* buffer, NS::Range range)
{
    _MTL4_msg_v_resetCommandsInBuffer_withRange__MTL__IndirectCommandBufferp_NS__Range((const void*)this, nullptr, buffer, range);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::copyIndirectCommandBuffer(MTL::IndirectCommandBuffer* source, NS::Range sourceRange, MTL::IndirectCommandBuffer* destination, NS::UInteger destinationIndex)
{
    _MTL4_msg_v_copyIndirectCommandBuffer_sourceRange_destination_destinationIndex__MTL__IndirectCommandBufferp_NS__Range_MTL__IndirectCommandBufferp_NS__UInteger((const void*)this, nullptr, source, sourceRange, destination, destinationIndex);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::optimizeIndirectCommandBuffer(MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::Range range)
{
    _MTL4_msg_v_optimizeIndirectCommandBuffer_withRange__MTL__IndirectCommandBufferp_NS__Range((const void*)this, nullptr, indirectCommandBuffer, range);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::setArgumentTable(MTL4::ArgumentTable* argumentTable)
{
    _MTL4_msg_v_setArgumentTable__MTL4__ArgumentTablep((const void*)this, nullptr, argumentTable);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::buildAccelerationStructure(MTL::AccelerationStructure* accelerationStructure, MTL4::AccelerationStructureDescriptor* descriptor, MTL4::BufferRange scratchBuffer)
{
    _MTL4_msg_v_buildAccelerationStructure_descriptor_scratchBuffer__MTL__AccelerationStructurep_MTL4__AccelerationStructureDescriptorp_MTL4__BufferRange((const void*)this, nullptr, accelerationStructure, descriptor, scratchBuffer);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::refitAccelerationStructure(MTL::AccelerationStructure* sourceAccelerationStructure, MTL4::AccelerationStructureDescriptor* descriptor, MTL::AccelerationStructure* destinationAccelerationStructure, MTL4::BufferRange scratchBuffer)
{
    _MTL4_msg_v_refitAccelerationStructure_descriptor_destination_scratchBuffer__MTL__AccelerationStructurep_MTL4__AccelerationStructureDescriptorp_MTL__AccelerationStructurep_MTL4__BufferRange((const void*)this, nullptr, sourceAccelerationStructure, descriptor, destinationAccelerationStructure, scratchBuffer);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::refitAccelerationStructure(MTL::AccelerationStructure* sourceAccelerationStructure, MTL4::AccelerationStructureDescriptor* descriptor, MTL::AccelerationStructure* destinationAccelerationStructure, MTL4::BufferRange scratchBuffer, MTL::AccelerationStructureRefitOptions options)
{
    _MTL4_msg_v_refitAccelerationStructure_descriptor_destination_scratchBuffer_options__MTL__AccelerationStructurep_MTL4__AccelerationStructureDescriptorp_MTL__AccelerationStructurep_MTL4__BufferRange_MTL__AccelerationStructureRefitOptions((const void*)this, nullptr, sourceAccelerationStructure, descriptor, destinationAccelerationStructure, scratchBuffer, options);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::copyAccelerationStructure(MTL::AccelerationStructure* sourceAccelerationStructure, MTL::AccelerationStructure* destinationAccelerationStructure)
{
    _MTL4_msg_v_copyAccelerationStructure_toAccelerationStructure__MTL__AccelerationStructurep_MTL__AccelerationStructurep((const void*)this, nullptr, sourceAccelerationStructure, destinationAccelerationStructure);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::writeCompactedAccelerationStructureSize(MTL::AccelerationStructure* accelerationStructure, MTL4::BufferRange buffer)
{
    _MTL4_msg_v_writeCompactedAccelerationStructureSize_toBuffer__MTL__AccelerationStructurep_MTL4__BufferRange((const void*)this, nullptr, accelerationStructure, buffer);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::copyAndCompactAccelerationStructure(MTL::AccelerationStructure* sourceAccelerationStructure, MTL::AccelerationStructure* destinationAccelerationStructure)
{
    _MTL4_msg_v_copyAndCompactAccelerationStructure_toAccelerationStructure__MTL__AccelerationStructurep_MTL__AccelerationStructurep((const void*)this, nullptr, sourceAccelerationStructure, destinationAccelerationStructure);
}

_MTL4_INLINE void MTL4::ComputeCommandEncoder::writeTimestamp(MTL4::TimestampGranularity granularity, MTL4::CounterHeap* counterHeap, NS::UInteger index)
{
    _MTL4_msg_v_writeTimestampWithGranularity_intoHeap_atIndex__MTL4__TimestampGranularity_MTL4__CounterHeapp_NS__UInteger((const void*)this, nullptr, granularity, counterHeap, index);
}
