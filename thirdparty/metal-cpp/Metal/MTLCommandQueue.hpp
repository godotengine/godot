#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class CommandBuffer;
    class CommandBufferDescriptor;
    class Device;
    class LogState;
    class ResidencySet;
}
namespace NS {
    class String;
}

namespace MTL
{

class CommandQueue;
class CommandQueueDescriptor;

class CommandQueue : public NS::Referencing<CommandQueue>
{
public:
    void                addResidencySet(MTL::ResidencySet* residencySet);
    void                addResidencySets(const MTL::ResidencySet* const * residencySets, NS::UInteger count);
    MTL::CommandBuffer* commandBuffer();
    MTL::CommandBuffer* commandBuffer(MTL::CommandBufferDescriptor* descriptor);
    MTL::CommandBuffer* commandBufferWithUnretainedReferences();
    MTL::Device*        device() const;
    void                insertDebugCaptureBoundary();
    NS::String*         label() const;
    void                removeResidencySet(MTL::ResidencySet* residencySet);
    void                removeResidencySets(const MTL::ResidencySet* const * residencySets, NS::UInteger count);
    void                setLabel(NS::String* label);

};

class CommandQueueDescriptor : public NS::Copying<CommandQueueDescriptor>
{
public:
    static CommandQueueDescriptor* alloc();
    CommandQueueDescriptor*        init() const;

    MTL::LogState* logState() const;
    NS::UInteger   maxCommandBufferCount() const;
    void           setLogState(MTL::LogState* logState);
    void           setMaxCommandBufferCount(NS::UInteger maxCommandBufferCount);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLCommandQueue;
extern "C" void *OBJC_CLASS_$_MTLCommandQueueDescriptor;

_MTL_INLINE NS::String* MTL::CommandQueue::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CommandQueue::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE MTL::Device* MTL::CommandQueue::device() const
{
    return _MTL_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL_INLINE MTL::CommandBuffer* MTL::CommandQueue::commandBuffer()
{
    return _MTL_msg_MTL__CommandBufferp_commandBuffer((const void*)this, nullptr);
}

_MTL_INLINE MTL::CommandBuffer* MTL::CommandQueue::commandBuffer(MTL::CommandBufferDescriptor* descriptor)
{
    return _MTL_msg_MTL__CommandBufferp_commandBufferWithDescriptor__MTL__CommandBufferDescriptorp((const void*)this, nullptr, descriptor);
}

_MTL_INLINE MTL::CommandBuffer* MTL::CommandQueue::commandBufferWithUnretainedReferences()
{
    return _MTL_msg_MTL__CommandBufferp_commandBufferWithUnretainedReferences((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CommandQueue::insertDebugCaptureBoundary()
{
    _MTL_msg_v_insertDebugCaptureBoundary((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CommandQueue::addResidencySet(MTL::ResidencySet* residencySet)
{
    _MTL_msg_v_addResidencySet__MTL__ResidencySetp((const void*)this, nullptr, residencySet);
}

_MTL_INLINE void MTL::CommandQueue::addResidencySets(const MTL::ResidencySet* const * residencySets, NS::UInteger count)
{
    _MTL_msg_v_addResidencySets_count__constMTL__ResidencySetpconstp_NS__UInteger((const void*)this, nullptr, residencySets, count);
}

_MTL_INLINE void MTL::CommandQueue::removeResidencySet(MTL::ResidencySet* residencySet)
{
    _MTL_msg_v_removeResidencySet__MTL__ResidencySetp((const void*)this, nullptr, residencySet);
}

_MTL_INLINE void MTL::CommandQueue::removeResidencySets(const MTL::ResidencySet* const * residencySets, NS::UInteger count)
{
    _MTL_msg_v_removeResidencySets_count__constMTL__ResidencySetpconstp_NS__UInteger((const void*)this, nullptr, residencySets, count);
}

_MTL_INLINE MTL::CommandQueueDescriptor* MTL::CommandQueueDescriptor::alloc()
{
    return _MTL_msg_MTL__CommandQueueDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLCommandQueueDescriptor, nullptr);
}

_MTL_INLINE MTL::CommandQueueDescriptor* MTL::CommandQueueDescriptor::init() const
{
    return _MTL_msg_MTL__CommandQueueDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::CommandQueueDescriptor::maxCommandBufferCount() const
{
    return _MTL_msg_NS__UInteger_maxCommandBufferCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CommandQueueDescriptor::setMaxCommandBufferCount(NS::UInteger maxCommandBufferCount)
{
    _MTL_msg_v_setMaxCommandBufferCount__NS__UInteger((const void*)this, nullptr, maxCommandBufferCount);
}

_MTL_INLINE MTL::LogState* MTL::CommandQueueDescriptor::logState() const
{
    return _MTL_msg_MTL__LogStatep_logState((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CommandQueueDescriptor::setLogState(MTL::LogState* logState)
{
    _MTL_msg_v_setLogState__MTL__LogStatep((const void*)this, nullptr, logState);
}
