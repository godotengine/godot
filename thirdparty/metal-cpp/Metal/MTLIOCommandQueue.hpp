#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class Buffer;
    class IOCommandBuffer;
}
namespace NS {
    class String;
}

namespace MTL
{

extern NS::ErrorDomain const IOErrorDomain __asm__("_MTLIOErrorDomain");
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


class IOCommandQueue;
class IOScratchBuffer;
class IOScratchBufferAllocator;
class IOCommandQueueDescriptor;
class IOFileHandle;

class IOCommandQueue : public NS::Referencing<IOCommandQueue>
{
public:
    MTL::IOCommandBuffer* commandBuffer();
    MTL::IOCommandBuffer* commandBufferWithUnretainedReferences();
    void                  enqueueBarrier();
    NS::String*           label() const;
    void                  setLabel(NS::String* label);

};

class IOScratchBuffer : public NS::Referencing<IOScratchBuffer>
{
public:
    MTL::Buffer* buffer() const;

};

class IOScratchBufferAllocator : public NS::Referencing<IOScratchBufferAllocator>
{
public:
    MTL::IOScratchBuffer* newScratchBuffer(NS::UInteger minimumSize);

};

class IOCommandQueueDescriptor : public NS::Copying<IOCommandQueueDescriptor>
{
public:
    static IOCommandQueueDescriptor* alloc();
    IOCommandQueueDescriptor*        init() const;

    NS::UInteger                   maxCommandBufferCount() const;
    NS::UInteger                   maxCommandsInFlight() const;
    MTL::IOPriority                priority() const;
    MTL::IOScratchBufferAllocator* scratchBufferAllocator() const;
    void                           setMaxCommandBufferCount(NS::UInteger maxCommandBufferCount);
    void                           setMaxCommandsInFlight(NS::UInteger maxCommandsInFlight);
    void                           setPriority(MTL::IOPriority priority);
    void                           setScratchBufferAllocator(MTL::IOScratchBufferAllocator* scratchBufferAllocator);
    void                           setType(MTL::IOCommandQueueType type);
    MTL::IOCommandQueueType        type() const;

};

class IOFileHandle : public NS::Referencing<IOFileHandle>
{
public:
    NS::String* label() const;
    void        setLabel(NS::String* label);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLIOCommandQueue;
extern "C" void *OBJC_CLASS_$_MTLIOScratchBuffer;
extern "C" void *OBJC_CLASS_$_MTLIOScratchBufferAllocator;
extern "C" void *OBJC_CLASS_$_MTLIOCommandQueueDescriptor;
extern "C" void *OBJC_CLASS_$_MTLIOFileHandle;

_MTL_INLINE NS::String* MTL::IOCommandQueue::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IOCommandQueue::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE void MTL::IOCommandQueue::enqueueBarrier()
{
    _MTL_msg_v_enqueueBarrier((const void*)this, nullptr);
}

_MTL_INLINE MTL::IOCommandBuffer* MTL::IOCommandQueue::commandBuffer()
{
    return _MTL_msg_MTL__IOCommandBufferp_commandBuffer((const void*)this, nullptr);
}

_MTL_INLINE MTL::IOCommandBuffer* MTL::IOCommandQueue::commandBufferWithUnretainedReferences()
{
    return _MTL_msg_MTL__IOCommandBufferp_commandBufferWithUnretainedReferences((const void*)this, nullptr);
}

_MTL_INLINE MTL::Buffer* MTL::IOScratchBuffer::buffer() const
{
    return _MTL_msg_MTL__Bufferp_buffer((const void*)this, nullptr);
}

_MTL_INLINE MTL::IOScratchBuffer* MTL::IOScratchBufferAllocator::newScratchBuffer(NS::UInteger minimumSize)
{
    return _MTL_msg_MTL__IOScratchBufferp_newScratchBufferWithMinimumSize__NS__UInteger((const void*)this, nullptr, minimumSize);
}

_MTL_INLINE MTL::IOCommandQueueDescriptor* MTL::IOCommandQueueDescriptor::alloc()
{
    return _MTL_msg_MTL__IOCommandQueueDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLIOCommandQueueDescriptor, nullptr);
}

_MTL_INLINE MTL::IOCommandQueueDescriptor* MTL::IOCommandQueueDescriptor::init() const
{
    return _MTL_msg_MTL__IOCommandQueueDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::IOCommandQueueDescriptor::maxCommandBufferCount() const
{
    return _MTL_msg_NS__UInteger_maxCommandBufferCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IOCommandQueueDescriptor::setMaxCommandBufferCount(NS::UInteger maxCommandBufferCount)
{
    _MTL_msg_v_setMaxCommandBufferCount__NS__UInteger((const void*)this, nullptr, maxCommandBufferCount);
}

_MTL_INLINE MTL::IOPriority MTL::IOCommandQueueDescriptor::priority() const
{
    return _MTL_msg_MTL__IOPriority_priority((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IOCommandQueueDescriptor::setPriority(MTL::IOPriority priority)
{
    _MTL_msg_v_setPriority__MTL__IOPriority((const void*)this, nullptr, priority);
}

_MTL_INLINE MTL::IOCommandQueueType MTL::IOCommandQueueDescriptor::type() const
{
    return _MTL_msg_MTL__IOCommandQueueType_type((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IOCommandQueueDescriptor::setType(MTL::IOCommandQueueType type)
{
    _MTL_msg_v_setType__MTL__IOCommandQueueType((const void*)this, nullptr, type);
}

_MTL_INLINE NS::UInteger MTL::IOCommandQueueDescriptor::maxCommandsInFlight() const
{
    return _MTL_msg_NS__UInteger_maxCommandsInFlight((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IOCommandQueueDescriptor::setMaxCommandsInFlight(NS::UInteger maxCommandsInFlight)
{
    _MTL_msg_v_setMaxCommandsInFlight__NS__UInteger((const void*)this, nullptr, maxCommandsInFlight);
}

_MTL_INLINE MTL::IOScratchBufferAllocator* MTL::IOCommandQueueDescriptor::scratchBufferAllocator() const
{
    return _MTL_msg_MTL__IOScratchBufferAllocatorp_scratchBufferAllocator((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IOCommandQueueDescriptor::setScratchBufferAllocator(MTL::IOScratchBufferAllocator* scratchBufferAllocator)
{
    _MTL_msg_v_setScratchBufferAllocator__MTL__IOScratchBufferAllocatorp((const void*)this, nullptr, scratchBufferAllocator);
}

_MTL_INLINE NS::String* MTL::IOFileHandle::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IOFileHandle::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}
