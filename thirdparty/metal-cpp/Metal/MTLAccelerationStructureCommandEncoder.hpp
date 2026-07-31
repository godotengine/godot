//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLAccelerationStructureCommandEncoder.hpp
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
#include "MTLAccelerationStructure.hpp"
#include "MTLCommandEncoder.hpp"
#include "MTLDataType.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"

namespace MTL
{
class AccelerationStructure;
class AccelerationStructureDescriptor;
class AccelerationStructurePassDescriptor;
class AccelerationStructurePassSampleBufferAttachmentDescriptor;
class AccelerationStructurePassSampleBufferAttachmentDescriptorArray;
class Buffer;
class CounterSampleBuffer;
class Fence;
class Heap;
class Resource;

class AccelerationStructureCommandEncoder : public NS::Referencing<AccelerationStructureCommandEncoder, CommandEncoder>
{
public:
    void buildAccelerationStructure(const MTL::AccelerationStructure* accelerationStructure, const MTL::AccelerationStructureDescriptor* descriptor, const MTL::Buffer* scratchBuffer, NS::UInteger scratchBufferOffset);

    void copyAccelerationStructure(const MTL::AccelerationStructure* sourceAccelerationStructure, const MTL::AccelerationStructure* destinationAccelerationStructure);

    void copyAndCompactAccelerationStructure(const MTL::AccelerationStructure* sourceAccelerationStructure, const MTL::AccelerationStructure* destinationAccelerationStructure);

    void refitAccelerationStructure(const MTL::AccelerationStructure* sourceAccelerationStructure, const MTL::AccelerationStructureDescriptor* descriptor, const MTL::AccelerationStructure* destinationAccelerationStructure, const MTL::Buffer* scratchBuffer, NS::UInteger scratchBufferOffset);
    void refitAccelerationStructure(const MTL::AccelerationStructure* sourceAccelerationStructure, const MTL::AccelerationStructureDescriptor* descriptor, const MTL::AccelerationStructure* destinationAccelerationStructure, const MTL::Buffer* scratchBuffer, NS::UInteger scratchBufferOffset, MTL::AccelerationStructureRefitOptions options);

    void sampleCountersInBuffer(const MTL::CounterSampleBuffer* sampleBuffer, NS::UInteger sampleIndex, bool barrier);

    void updateFence(const MTL::Fence* fence);

    void useHeap(const MTL::Heap* heap);
    void useHeaps(const MTL::Heap* const heaps[], NS::UInteger count);

    void useResource(const MTL::Resource* resource, MTL::ResourceUsage usage);
    void useResources(const MTL::Resource* const resources[], NS::UInteger count, MTL::ResourceUsage usage);

    void waitForFence(const MTL::Fence* fence);

    void writeCompactedAccelerationStructureSize(const MTL::AccelerationStructure* accelerationStructure, const MTL::Buffer* buffer, NS::UInteger offset);
    void writeCompactedAccelerationStructureSize(const MTL::AccelerationStructure* accelerationStructure, const MTL::Buffer* buffer, NS::UInteger offset, MTL::DataType sizeDataType);
};
class AccelerationStructurePassSampleBufferAttachmentDescriptor : public NS::Copying<AccelerationStructurePassSampleBufferAttachmentDescriptor>
{
public:
    static AccelerationStructurePassSampleBufferAttachmentDescriptor* alloc();

    NS::UInteger                                                      endOfEncoderSampleIndex() const;

    AccelerationStructurePassSampleBufferAttachmentDescriptor*        init();

    CounterSampleBuffer*                                              sampleBuffer() const;

    void                                                              setEndOfEncoderSampleIndex(NS::UInteger endOfEncoderSampleIndex);

    void                                                              setSampleBuffer(const MTL::CounterSampleBuffer* sampleBuffer);

    void                                                              setStartOfEncoderSampleIndex(NS::UInteger startOfEncoderSampleIndex);
    NS::UInteger                                                      startOfEncoderSampleIndex() const;
};
class AccelerationStructurePassSampleBufferAttachmentDescriptorArray : public NS::Referencing<AccelerationStructurePassSampleBufferAttachmentDescriptorArray>
{
public:
    static AccelerationStructurePassSampleBufferAttachmentDescriptorArray* alloc();

    AccelerationStructurePassSampleBufferAttachmentDescriptorArray*        init();

    AccelerationStructurePassSampleBufferAttachmentDescriptor*             object(NS::UInteger attachmentIndex);
    void                                                                   setObject(const MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor* attachment, NS::UInteger attachmentIndex);
};
class AccelerationStructurePassDescriptor : public NS::Copying<AccelerationStructurePassDescriptor>
{
public:
    static AccelerationStructurePassDescriptor*                     accelerationStructurePassDescriptor();

    static AccelerationStructurePassDescriptor*                     alloc();

    AccelerationStructurePassDescriptor*                            init();

    AccelerationStructurePassSampleBufferAttachmentDescriptorArray* sampleBufferAttachments() const;
};

}
_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::buildAccelerationStructure(const MTL::AccelerationStructure* accelerationStructure, const MTL::AccelerationStructureDescriptor* descriptor, const MTL::Buffer* scratchBuffer, NS::UInteger scratchBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset_), accelerationStructure, descriptor, scratchBuffer, scratchBufferOffset);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::copyAccelerationStructure(const MTL::AccelerationStructure* sourceAccelerationStructure, const MTL::AccelerationStructure* destinationAccelerationStructure)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyAccelerationStructure_toAccelerationStructure_), sourceAccelerationStructure, destinationAccelerationStructure);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::copyAndCompactAccelerationStructure(const MTL::AccelerationStructure* sourceAccelerationStructure, const MTL::AccelerationStructure* destinationAccelerationStructure)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyAndCompactAccelerationStructure_toAccelerationStructure_), sourceAccelerationStructure, destinationAccelerationStructure);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::refitAccelerationStructure(const MTL::AccelerationStructure* sourceAccelerationStructure, const MTL::AccelerationStructureDescriptor* descriptor, const MTL::AccelerationStructure* destinationAccelerationStructure, const MTL::Buffer* scratchBuffer, NS::UInteger scratchBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(refitAccelerationStructure_descriptor_destination_scratchBuffer_scratchBufferOffset_), sourceAccelerationStructure, descriptor, destinationAccelerationStructure, scratchBuffer, scratchBufferOffset);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::refitAccelerationStructure(const MTL::AccelerationStructure* sourceAccelerationStructure, const MTL::AccelerationStructureDescriptor* descriptor, const MTL::AccelerationStructure* destinationAccelerationStructure, const MTL::Buffer* scratchBuffer, NS::UInteger scratchBufferOffset, MTL::AccelerationStructureRefitOptions options)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(refitAccelerationStructure_descriptor_destination_scratchBuffer_scratchBufferOffset_options_), sourceAccelerationStructure, descriptor, destinationAccelerationStructure, scratchBuffer, scratchBufferOffset, options);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::sampleCountersInBuffer(const MTL::CounterSampleBuffer* sampleBuffer, NS::UInteger sampleIndex, bool barrier)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(sampleCountersInBuffer_atSampleIndex_withBarrier_), sampleBuffer, sampleIndex, barrier);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::updateFence(const MTL::Fence* fence)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(updateFence_), fence);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::useHeap(const MTL::Heap* heap)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(useHeap_), heap);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::useHeaps(const MTL::Heap* const heaps[], NS::UInteger count)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(useHeaps_count_), heaps, count);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::useResource(const MTL::Resource* resource, MTL::ResourceUsage usage)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(useResource_usage_), resource, usage);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::useResources(const MTL::Resource* const resources[], NS::UInteger count, MTL::ResourceUsage usage)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(useResources_count_usage_), resources, count, usage);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::waitForFence(const MTL::Fence* fence)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(waitForFence_), fence);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::writeCompactedAccelerationStructureSize(const MTL::AccelerationStructure* accelerationStructure, const MTL::Buffer* buffer, NS::UInteger offset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(writeCompactedAccelerationStructureSize_toBuffer_offset_), accelerationStructure, buffer, offset);
}

_MTL_INLINE void MTL::AccelerationStructureCommandEncoder::writeCompactedAccelerationStructureSize(const MTL::AccelerationStructure* accelerationStructure, const MTL::Buffer* buffer, NS::UInteger offset, MTL::DataType sizeDataType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(writeCompactedAccelerationStructureSize_toBuffer_offset_sizeDataType_), accelerationStructure, buffer, offset, sizeDataType);
}

_MTL_INLINE MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor* MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor::alloc()
{
    return NS::Object::alloc<MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor>(_MTL_PRIVATE_CLS(MTLAccelerationStructurePassSampleBufferAttachmentDescriptor));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor::endOfEncoderSampleIndex() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(endOfEncoderSampleIndex));
}

_MTL_INLINE MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor* MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor::init()
{
    return NS::Object::init<MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor>();
}

_MTL_INLINE MTL::CounterSampleBuffer* MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor::sampleBuffer() const
{
    return Object::sendMessage<MTL::CounterSampleBuffer*>(this, _MTL_PRIVATE_SEL(sampleBuffer));
}

_MTL_INLINE void MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor::setEndOfEncoderSampleIndex(NS::UInteger endOfEncoderSampleIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setEndOfEncoderSampleIndex_), endOfEncoderSampleIndex);
}

_MTL_INLINE void MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor::setSampleBuffer(const MTL::CounterSampleBuffer* sampleBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSampleBuffer_), sampleBuffer);
}

_MTL_INLINE void MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor::setStartOfEncoderSampleIndex(NS::UInteger startOfEncoderSampleIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStartOfEncoderSampleIndex_), startOfEncoderSampleIndex);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor::startOfEncoderSampleIndex() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(startOfEncoderSampleIndex));
}

_MTL_INLINE MTL::AccelerationStructurePassSampleBufferAttachmentDescriptorArray* MTL::AccelerationStructurePassSampleBufferAttachmentDescriptorArray::alloc()
{
    return NS::Object::alloc<MTL::AccelerationStructurePassSampleBufferAttachmentDescriptorArray>(_MTL_PRIVATE_CLS(MTLAccelerationStructurePassSampleBufferAttachmentDescriptorArray));
}

_MTL_INLINE MTL::AccelerationStructurePassSampleBufferAttachmentDescriptorArray* MTL::AccelerationStructurePassSampleBufferAttachmentDescriptorArray::init()
{
    return NS::Object::init<MTL::AccelerationStructurePassSampleBufferAttachmentDescriptorArray>();
}

_MTL_INLINE MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor* MTL::AccelerationStructurePassSampleBufferAttachmentDescriptorArray::object(NS::UInteger attachmentIndex)
{
    return Object::sendMessage<MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor*>(this, _MTL_PRIVATE_SEL(objectAtIndexedSubscript_), attachmentIndex);
}

_MTL_INLINE void MTL::AccelerationStructurePassSampleBufferAttachmentDescriptorArray::setObject(const MTL::AccelerationStructurePassSampleBufferAttachmentDescriptor* attachment, NS::UInteger attachmentIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObject_atIndexedSubscript_), attachment, attachmentIndex);
}

_MTL_INLINE MTL::AccelerationStructurePassDescriptor* MTL::AccelerationStructurePassDescriptor::accelerationStructurePassDescriptor()
{
    return Object::sendMessage<MTL::AccelerationStructurePassDescriptor*>(_MTL_PRIVATE_CLS(MTLAccelerationStructurePassDescriptor), _MTL_PRIVATE_SEL(accelerationStructurePassDescriptor));
}

_MTL_INLINE MTL::AccelerationStructurePassDescriptor* MTL::AccelerationStructurePassDescriptor::alloc()
{
    return NS::Object::alloc<MTL::AccelerationStructurePassDescriptor>(_MTL_PRIVATE_CLS(MTLAccelerationStructurePassDescriptor));
}

_MTL_INLINE MTL::AccelerationStructurePassDescriptor* MTL::AccelerationStructurePassDescriptor::init()
{
    return NS::Object::init<MTL::AccelerationStructurePassDescriptor>();
}

_MTL_INLINE MTL::AccelerationStructurePassSampleBufferAttachmentDescriptorArray* MTL::AccelerationStructurePassDescriptor::sampleBufferAttachments() const
{
    return Object::sendMessage<MTL::AccelerationStructurePassSampleBufferAttachmentDescriptorArray*>(this, _MTL_PRIVATE_SEL(sampleBufferAttachments));
}
