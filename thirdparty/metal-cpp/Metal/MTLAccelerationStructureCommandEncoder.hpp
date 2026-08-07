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
    class AccelerationStructureDescriptor;
    class Buffer;
    class CounterSampleBuffer;
    class Fence;
    class Heap;
    class Resource;
    using AccelerationStructureRefitOptions = NS::UInteger;
    enum DataType : NS::UInteger;
    using ResourceUsage = NS::UInteger;
}

namespace MTL
{

class AccelerationStructureCommandEncoder;
class AccelerationStructurePassSampleBufferAttachmentDescriptor;
class AccelerationStructurePassSampleBufferAttachmentDescriptorArray;
class AccelerationStructurePassDescriptor;

class AccelerationStructureCommandEncoder : public NS::Referencing<AccelerationStructureCommandEncoder, MTL::CommandEncoder>
{
public:
    void build(MTL::AccelerationStructure* accelerationStructure, MTL::AccelerationStructureDescriptor* descriptor, MTL::Buffer* scratchBuffer, NS::UInteger scratchBufferOffset);
    void copyAccelerationStructure(MTL::AccelerationStructure* sourceAccelerationStructure, MTL::AccelerationStructure* destinationAccelerationStructure);
    void copyAndCompactAccelerationStructure(MTL::AccelerationStructure* sourceAccelerationStructure, MTL::AccelerationStructure* destinationAccelerationStructure);
    void refitAccelerationStructure(MTL::AccelerationStructure* sourceAccelerationStructure, MTL::AccelerationStructureDescriptor* descriptor, MTL::AccelerationStructure* destinationAccelerationStructure, MTL::Buffer* scratchBuffer, NS::UInteger scratchBufferOffset);
    void refitAccelerationStructure(MTL::AccelerationStructure* sourceAccelerationStructure, MTL::AccelerationStructureDescriptor* descriptor, MTL::AccelerationStructure* destinationAccelerationStructure, MTL::Buffer* scratchBuffer, NS::UInteger scratchBufferOffset, MTL::AccelerationStructureRefitOptions options);
    void sampleCountersInBuffer(MTL::CounterSampleBuffer* sampleBuffer, NS::UInteger sampleIndex, bool barrier);
    void updateFence(MTL::Fence* fence);
    void useHeap(MTL::Heap* heap);
    void useHeaps(const MTL::Heap* const * heaps, NS::UInteger count);
    void useResource(MTL::Resource* resource, MTL::ResourceUsage usage);
    void useResources(const MTL::Resource* const * resources, NS::UInteger count, MTL::ResourceUsage usage);
    void waitForFence(MTL::Fence* fence);
    void writeCompactedAccelerationStructureSize(MTL::AccelerationStructure* accelerationStructure, MTL::Buffer* buffer, NS::UInteger offset);
    void writeCompactedAccelerationStructureSize(MTL::AccelerationStructure* accelerationStructure, MTL::Buffer* buffer, NS::UInteger offset, MTL::DataType sizeDataType);

};

class AccelerationStructurePassSampleBufferAttachmentDescriptor : public NS::Copying<AccelerationStructurePassSampleBufferAttachmentDescriptor>
{
public:
    static AccelerationStructurePassSampleBufferAttachmentDescriptor* alloc();
    AccelerationStructurePassSampleBufferAttachmentDescriptor*        init() const;

    NS::UInteger              endOfEncoderSampleIndex() const;
    MTL::CounterSampleBuffer* sampleBuffer() const;
    void                      setEndOfEncoderSampleIndex(NS::UInteger endOfEncoderSampleIndex);
    void                      setSampleBuffer(MTL::CounterSampleBuffer* sampleBuffer);
    void                      setStartOfEncoderSampleIndex(NS::UInteger startOfEncoderSampleIndex);
    NS::UInteger              startOfEncoderSampleIndex() const;

};

class AccelerationStructurePassSampleBufferAttachmentDescriptorArray : public NS::Referencing<AccelerationStructurePassSampleBufferAttachmentDescriptorArray>
{
public:
    static AccelerationStructurePassSampleBufferAttachmentDescriptorArray* alloc();
    AccelerationStructurePassSampleBufferAttachmentDescriptorArray*        init() const;

    MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor* object(NS::UInteger attachmentIndex);
    void                                                            setObject(MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor* attachment, NS::UInteger attachmentIndex);

};

class AccelerationStructurePassDescriptor : public NS::Copying<AccelerationStructurePassDescriptor>
{
public:
    static AccelerationStructurePassDescriptor* alloc();
    AccelerationStructurePassDescriptor*        init() const;

    static MTL::AccelerationStructurePassDescriptor* accelerationStructurePassDescriptor();

    MTL::AccelerationStructurePassSampleBufferAttachmentDescriptorArray* sampleBufferAttachments() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLAccelerationStructureCommandEncoder;
extern "C" void *OBJC_CLASS_$_MTLAccelerationStructurePassSampleBufferAttachmentDescriptor;
extern "C" void *OBJC_CLASS_$_MTLAccelerationStructurePassSampleBufferAttachmentDescriptorArray;
extern "C" void *OBJC_CLASS_$_MTLAccelerationStructurePassDescriptor;

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::build(MTL::AccelerationStructure* accelerationStructure, MTL::AccelerationStructureDescriptor* descriptor, MTL::Buffer* scratchBuffer, NS::UInteger scratchBufferOffset)
{
    _MTL_msg_v_buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset__MTL__AccelerationStructurep_MTL__AccelerationStructureDescriptorp_MTL__Bufferp_NS__UInteger((const void*)this, nullptr, accelerationStructure, descriptor, scratchBuffer, scratchBufferOffset);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::refitAccelerationStructure(MTL::AccelerationStructure* sourceAccelerationStructure, MTL::AccelerationStructureDescriptor* descriptor, MTL::AccelerationStructure* destinationAccelerationStructure, MTL::Buffer* scratchBuffer, NS::UInteger scratchBufferOffset)
{
    _MTL_msg_v_refitAccelerationStructure_descriptor_destination_scratchBuffer_scratchBufferOffset__MTL__AccelerationStructurep_MTL__AccelerationStructureDescriptorp_MTL__AccelerationStructurep_MTL__Bufferp_NS__UInteger((const void*)this, nullptr, sourceAccelerationStructure, descriptor, destinationAccelerationStructure, scratchBuffer, scratchBufferOffset);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::refitAccelerationStructure(MTL::AccelerationStructure* sourceAccelerationStructure, MTL::AccelerationStructureDescriptor* descriptor, MTL::AccelerationStructure* destinationAccelerationStructure, MTL::Buffer* scratchBuffer, NS::UInteger scratchBufferOffset, MTL::AccelerationStructureRefitOptions options)
{
    _MTL_msg_v_refitAccelerationStructure_descriptor_destination_scratchBuffer_scratchBufferOffset_options__MTL__AccelerationStructurep_MTL__AccelerationStructureDescriptorp_MTL__AccelerationStructurep_MTL__Bufferp_NS__UInteger_MTL__AccelerationStructureRefitOptions((const void*)this, nullptr, sourceAccelerationStructure, descriptor, destinationAccelerationStructure, scratchBuffer, scratchBufferOffset, options);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::copyAccelerationStructure(MTL::AccelerationStructure* sourceAccelerationStructure, MTL::AccelerationStructure* destinationAccelerationStructure)
{
    _MTL_msg_v_copyAccelerationStructure_toAccelerationStructure__MTL__AccelerationStructurep_MTL__AccelerationStructurep((const void*)this, nullptr, sourceAccelerationStructure, destinationAccelerationStructure);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::writeCompactedAccelerationStructureSize(MTL::AccelerationStructure* accelerationStructure, MTL::Buffer* buffer, NS::UInteger offset)
{
    _MTL_msg_v_writeCompactedAccelerationStructureSize_toBuffer_offset__MTL__AccelerationStructurep_MTL__Bufferp_NS__UInteger((const void*)this, nullptr, accelerationStructure, buffer, offset);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::writeCompactedAccelerationStructureSize(MTL::AccelerationStructure* accelerationStructure, MTL::Buffer* buffer, NS::UInteger offset, MTL::DataType sizeDataType)
{
    _MTL_msg_v_writeCompactedAccelerationStructureSize_toBuffer_offset_sizeDataType__MTL__AccelerationStructurep_MTL__Bufferp_NS__UInteger_MTL__DataType((const void*)this, nullptr, accelerationStructure, buffer, offset, sizeDataType);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::copyAndCompactAccelerationStructure(MTL::AccelerationStructure* sourceAccelerationStructure, MTL::AccelerationStructure* destinationAccelerationStructure)
{
    _MTL_msg_v_copyAndCompactAccelerationStructure_toAccelerationStructure__MTL__AccelerationStructurep_MTL__AccelerationStructurep((const void*)this, nullptr, sourceAccelerationStructure, destinationAccelerationStructure);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::updateFence(MTL::Fence* fence)
{
    _MTL_msg_v_updateFence__MTL__Fencep((const void*)this, nullptr, fence);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::waitForFence(MTL::Fence* fence)
{
    _MTL_msg_v_waitForFence__MTL__Fencep((const void*)this, nullptr, fence);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::useResource(MTL::Resource* resource, MTL::ResourceUsage usage)
{
    _MTL_msg_v_useResource_usage__MTL__Resourcep_MTL__ResourceUsage((const void*)this, nullptr, resource, usage);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::useResources(const MTL::Resource* const * resources, NS::UInteger count, MTL::ResourceUsage usage)
{
    _MTL_msg_v_useResources_count_usage__constMTL__Resourcepconstp_NS__UInteger_MTL__ResourceUsage((const void*)this, nullptr, resources, count, usage);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::useHeap(MTL::Heap* heap)
{
    _MTL_msg_v_useHeap__MTL__Heapp((const void*)this, nullptr, heap);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::useHeaps(const MTL::Heap* const * heaps, NS::UInteger count)
{
    _MTL_msg_v_useHeaps_count__constMTL__Heappconstp_NS__UInteger((const void*)this, nullptr, heaps, count);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::sampleCountersInBuffer(MTL::CounterSampleBuffer* sampleBuffer, NS::UInteger sampleIndex, bool barrier)
{
    _MTL_msg_v_sampleCountersInBuffer_atSampleIndex_withBarrier__MTL__CounterSampleBufferp_NS__UInteger_bool((const void*)this, nullptr, sampleBuffer, sampleIndex, barrier);
}

_MTL_INLINE MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor* MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor::alloc()
{
    return _MTL_msg_MTL__AccelerationStructurePassSampleBufferAttachmentDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLAccelerationStructurePassSampleBufferAttachmentDescriptor, nullptr);
}

_MTL_INLINE MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor* MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor::init() const
{
    return _MTL_msg_MTL__AccelerationStructurePassSampleBufferAttachmentDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::CounterSampleBuffer* MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor::sampleBuffer() const
{
    return _MTL_msg_MTL__CounterSampleBufferp_sampleBuffer((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor::setSampleBuffer(MTL::CounterSampleBuffer* sampleBuffer)
{
    _MTL_msg_v_setSampleBuffer__MTL__CounterSampleBufferp((const void*)this, nullptr, sampleBuffer);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor::startOfEncoderSampleIndex() const
{
    return _MTL_msg_NS__UInteger_startOfEncoderSampleIndex((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor::setStartOfEncoderSampleIndex(NS::UInteger startOfEncoderSampleIndex)
{
    _MTL_msg_v_setStartOfEncoderSampleIndex__NS__UInteger((const void*)this, nullptr, startOfEncoderSampleIndex);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor::endOfEncoderSampleIndex() const
{
    return _MTL_msg_NS__UInteger_endOfEncoderSampleIndex((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor::setEndOfEncoderSampleIndex(NS::UInteger endOfEncoderSampleIndex)
{
    _MTL_msg_v_setEndOfEncoderSampleIndex__NS__UInteger((const void*)this, nullptr, endOfEncoderSampleIndex);
}

_MTL_INLINE MTL::AccelerationStructurePassSampleBufferAttachmentDescriptorArray* MTL::AccelerationStructurePassSampleBufferAttachmentDescriptorArray::alloc()
{
    return _MTL_msg_MTL__AccelerationStructurePassSampleBufferAttachmentDescriptorArrayp_alloc((const void*)&OBJC_CLASS_$_MTLAccelerationStructurePassSampleBufferAttachmentDescriptorArray, nullptr);
}

_MTL_INLINE MTL::AccelerationStructurePassSampleBufferAttachmentDescriptorArray* MTL::AccelerationStructurePassSampleBufferAttachmentDescriptorArray::init() const
{
    return _MTL_msg_MTL__AccelerationStructurePassSampleBufferAttachmentDescriptorArrayp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor* MTL::AccelerationStructurePassSampleBufferAttachmentDescriptorArray::object(NS::UInteger attachmentIndex)
{
    return _MTL_msg_MTL__AccelerationStructurePassSampleBufferAttachmentDescriptorp_objectAtIndexedSubscript__NS__UInteger((const void*)this, nullptr, attachmentIndex);
}

_MTL_INLINE void MTL::AccelerationStructurePassSampleBufferAttachmentDescriptorArray::setObject(MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor* attachment, NS::UInteger attachmentIndex)
{
    _MTL_msg_v_setObject_atIndexedSubscript__MTL__AccelerationStructurePassSampleBufferAttachmentDescriptorp_NS__UInteger((const void*)this, nullptr, attachment, attachmentIndex);
}

_MTL_INLINE MTL::AccelerationStructurePassDescriptor* MTL::AccelerationStructurePassDescriptor::alloc()
{
    return _MTL_msg_MTL__AccelerationStructurePassDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLAccelerationStructurePassDescriptor, nullptr);
}

_MTL_INLINE MTL::AccelerationStructurePassDescriptor* MTL::AccelerationStructurePassDescriptor::init() const
{
    return _MTL_msg_MTL__AccelerationStructurePassDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::AccelerationStructurePassDescriptor* MTL::AccelerationStructurePassDescriptor::accelerationStructurePassDescriptor()
{
    return _MTL_msg_MTL__AccelerationStructurePassDescriptorp_accelerationStructurePassDescriptor((const void*)&OBJC_CLASS_$_MTLAccelerationStructurePassDescriptor, nullptr);
}

_MTL_INLINE MTL::AccelerationStructurePassSampleBufferAttachmentDescriptorArray* MTL::AccelerationStructurePassDescriptor::sampleBufferAttachments() const
{
    return _MTL_msg_MTL__AccelerationStructurePassSampleBufferAttachmentDescriptorArrayp_sampleBufferAttachments((const void*)this, nullptr);
}
