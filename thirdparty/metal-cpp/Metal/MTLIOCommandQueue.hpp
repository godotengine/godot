//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLIOCommandQueue.hpp
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
#include "MTLPrivate.hpp"

namespace MTL
{
class Buffer;
class IOCommandBuffer;
class IOCommandQueueDescriptor;
class IOScratchBuffer;
class IOScratchBufferAllocator;
_MTL_ENUM(NS::Integer, IOPriority) {
    IOPriorityHigh = 0,
    IOPriorityNormal = 1,
    IOPriorityLow = 2,
};

_MTL_ENUM(NS::Integer, IOCommandQueueType) {
    IOCommandQueueTypeConcurrent = 0,
    IOCommandQueueTypeSerial = 1,
};

_MTL_ENUM(NS::Integer, IOError) {
    IOErrorURLInvalid = 1,
    IOErrorInternal = 2,
};

_MTL_CONST(NS::ErrorDomain, IOErrorDomain);
class IOCommandQueue : public NS::Referencing<IOCommandQueue>
{
public:
    IOCommandBuffer* commandBuffer();
    IOCommandBuffer* commandBufferWithUnretainedReferences();

    void             enqueueBarrier();

    NS::String*      label() const;
    void             setLabel(const NS::String* label);
};
class IOScratchBuffer : public NS::Referencing<IOScratchBuffer>
{
public:
    Buffer* buffer() const;
};
class IOScratchBufferAllocator : public NS::Referencing<IOScratchBufferAllocator>
{
public:
    IOScratchBuffer* newScratchBuffer(NS::UInteger minimumSize);
};
class IOCommandQueueDescriptor : public NS::Copying<IOCommandQueueDescriptor>
{
public:
    static IOCommandQueueDescriptor* alloc();

    IOCommandQueueDescriptor*        init();

    NS::UInteger                     maxCommandBufferCount() const;

    NS::UInteger                     maxCommandsInFlight() const;

    IOPriority                       priority() const;

    IOScratchBufferAllocator*        scratchBufferAllocator() const;

    void                             setMaxCommandBufferCount(NS::UInteger maxCommandBufferCount);

    void                             setMaxCommandsInFlight(NS::UInteger maxCommandsInFlight);

    void                             setPriority(MTL::IOPriority priority);

    void                             setScratchBufferAllocator(const MTL::IOScratchBufferAllocator* scratchBufferAllocator);

    void                             setType(MTL::IOCommandQueueType type);
    IOCommandQueueType               type() const;
};
class IOFileHandle : public NS::Referencing<IOFileHandle>
{
public:
    NS::String* label() const;
    void        setLabel(const NS::String* label);
};

}
_MTL_PRIVATE_DEF_CONST(NS::ErrorDomain, IOErrorDomain);
_MTL_INLINE MTL::IOCommandBuffer* MTL::IOCommandQueue::commandBuffer()
{
    return Object::sendMessage<MTL::IOCommandBuffer*>(this, _MTL_PRIVATE_SEL(commandBuffer));
}

_MTL_INLINE MTL::IOCommandBuffer* MTL::IOCommandQueue::commandBufferWithUnretainedReferences()
{
    return Object::sendMessage<MTL::IOCommandBuffer*>(this, _MTL_PRIVATE_SEL(commandBufferWithUnretainedReferences));
}

_MTL_INLINE void MTL::IOCommandQueue::enqueueBarrier()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(enqueueBarrier));
}

_MTL_INLINE NS::String* MTL::IOCommandQueue::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE void MTL::IOCommandQueue::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE MTL::Buffer* MTL::IOScratchBuffer::buffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(buffer));
}

_MTL_INLINE MTL::IOScratchBuffer* MTL::IOScratchBufferAllocator::newScratchBuffer(NS::UInteger minimumSize)
{
    return Object::sendMessage<MTL::IOScratchBuffer*>(this, _MTL_PRIVATE_SEL(newScratchBufferWithMinimumSize_), minimumSize);
}

_MTL_INLINE MTL::IOCommandQueueDescriptor* MTL::IOCommandQueueDescriptor::alloc()
{
    return NS::Object::alloc<MTL::IOCommandQueueDescriptor>(_MTL_PRIVATE_CLS(MTLIOCommandQueueDescriptor));
}

_MTL_INLINE MTL::IOCommandQueueDescriptor* MTL::IOCommandQueueDescriptor::init()
{
    return NS::Object::init<MTL::IOCommandQueueDescriptor>();
}

_MTL_INLINE NS::UInteger MTL::IOCommandQueueDescriptor::maxCommandBufferCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxCommandBufferCount));
}

_MTL_INLINE NS::UInteger MTL::IOCommandQueueDescriptor::maxCommandsInFlight() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxCommandsInFlight));
}

_MTL_INLINE MTL::IOPriority MTL::IOCommandQueueDescriptor::priority() const
{
    return Object::sendMessage<MTL::IOPriority>(this, _MTL_PRIVATE_SEL(priority));
}

_MTL_INLINE MTL::IOScratchBufferAllocator* MTL::IOCommandQueueDescriptor::scratchBufferAllocator() const
{
    return Object::sendMessage<MTL::IOScratchBufferAllocator*>(this, _MTL_PRIVATE_SEL(scratchBufferAllocator));
}

_MTL_INLINE void MTL::IOCommandQueueDescriptor::setMaxCommandBufferCount(NS::UInteger maxCommandBufferCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxCommandBufferCount_), maxCommandBufferCount);
}

_MTL_INLINE void MTL::IOCommandQueueDescriptor::setMaxCommandsInFlight(NS::UInteger maxCommandsInFlight)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxCommandsInFlight_), maxCommandsInFlight);
}

_MTL_INLINE void MTL::IOCommandQueueDescriptor::setPriority(MTL::IOPriority priority)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPriority_), priority);
}

_MTL_INLINE void MTL::IOCommandQueueDescriptor::setScratchBufferAllocator(const MTL::IOScratchBufferAllocator* scratchBufferAllocator)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setScratchBufferAllocator_), scratchBufferAllocator);
}

_MTL_INLINE void MTL::IOCommandQueueDescriptor::setType(MTL::IOCommandQueueType type)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setType_), type);
}

_MTL_INLINE MTL::IOCommandQueueType MTL::IOCommandQueueDescriptor::type() const
{
    return Object::sendMessage<MTL::IOCommandQueueType>(this, _MTL_PRIVATE_SEL(type));
}

_MTL_INLINE NS::String* MTL::IOFileHandle::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE void MTL::IOFileHandle::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}
