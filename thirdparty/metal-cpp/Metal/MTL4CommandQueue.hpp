#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class Buffer;
    class Device;
    class Drawable;
    class Event;
    class Heap;
    class ResidencySet;
    class Texture;
}
namespace MTL4 {
    class CommandBuffer;
}
namespace NS {
    class String;
}

namespace MTL4
{

extern NS::ErrorDomain const CommandQueueErrorDomain __asm__("_MTL4CommandQueueErrorDomain");
_MTL4_ENUM(NS::Integer, CommandQueueError) {
    CommandQueueErrorNone = 0,
    CommandQueueErrorTimeout = 1,
    CommandQueueErrorNotPermitted = 2,
    CommandQueueErrorOutOfMemory = 3,
    CommandQueueErrorDeviceRemoved = 4,
    CommandQueueErrorAccessRevoked = 5,
    CommandQueueErrorInternal = 6,
};


class CommitOptions;
class CommandQueueDescriptor;
class CommandQueue;

class CommitOptions : public NS::Referencing<CommitOptions>
{
public:
    static CommitOptions* alloc();
    CommitOptions*        init() const;

    void addFeedbackHandler(MTL4::CommitFeedbackHandler block);
    void addFeedbackHandler(const MTL4::CommitFeedbackHandlerFunction& block);

};

class CommandQueueDescriptor : public NS::Copying<CommandQueueDescriptor>
{
public:
    static CommandQueueDescriptor* alloc();
    CommandQueueDescriptor*        init() const;

    dispatch_queue_t feedbackQueue() const;
    NS::String*      label() const;
    void             setFeedbackQueue(dispatch_queue_t feedbackQueue);
    void             setLabel(NS::String* label);

};

class CommandQueue : public NS::Referencing<CommandQueue>
{
public:
    void         addResidencySet(MTL::ResidencySet* residencySet);
    void         addResidencySets(const MTL::ResidencySet* const * residencySets, NS::UInteger count);
    void         commit(const MTL4::CommandBuffer* const * commandBuffers, NS::UInteger count);
    void         commit(const MTL4::CommandBuffer* const * commandBuffers, NS::UInteger count, MTL4::CommitOptions* options);
    void         copyBufferMappingsFromBuffer(MTL::Buffer* sourceBuffer, MTL::Buffer* destinationBuffer, const MTL4::CopySparseBufferMappingOperation * operations, NS::UInteger count);
    void         copyTextureMappingsFromTexture(MTL::Texture* sourceTexture, MTL::Texture* destinationTexture, const MTL4::CopySparseTextureMappingOperation * operations, NS::UInteger count);
    MTL::Device* device() const;
    NS::String*  label() const;
    void         removeResidencySet(MTL::ResidencySet* residencySet);
    void         removeResidencySets(const MTL::ResidencySet* const * residencySets, NS::UInteger count);
    void         signalDrawable(MTL::Drawable* drawable);
    void         signalEvent(MTL::Event* event, uint64_t value);
    void         updateBufferMappings(MTL::Buffer* buffer, MTL::Heap* heap, const MTL4::UpdateSparseBufferMappingOperation * operations, NS::UInteger count);
    void         updateTextureMappings(MTL::Texture* texture, MTL::Heap* heap, const MTL4::UpdateSparseTextureMappingOperation * operations, NS::UInteger count);
    void         wait(MTL::Event* event, uint64_t value);
    void         wait(MTL::Drawable* drawable);

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4CommitOptions;
extern "C" void *OBJC_CLASS_$_MTL4CommandQueueDescriptor;
extern "C" void *OBJC_CLASS_$_MTL4CommandQueue;

_MTL4_INLINE MTL4::CommitOptions* MTL4::CommitOptions::alloc()
{
    return _MTL4_msg_MTL4__CommitOptionsp_alloc((const void*)&OBJC_CLASS_$_MTL4CommitOptions, nullptr);
}

_MTL4_INLINE MTL4::CommitOptions* MTL4::CommitOptions::init() const
{
    return _MTL4_msg_MTL4__CommitOptionsp_init((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::CommitOptions::addFeedbackHandler(MTL4::CommitFeedbackHandler block)
{
    _MTL4_msg_v_addFeedbackHandler__MTL4__CommitFeedbackHandler((const void*)this, nullptr, block);
}

_MTL4_INLINE void MTL4::CommitOptions::addFeedbackHandler(const MTL4::CommitFeedbackHandlerFunction& block)
{
    __block MTL4::CommitFeedbackHandlerFunction blockFunction = block;
    addFeedbackHandler(^(MTL4::CommitFeedback* x0) { blockFunction(x0); });
}

_MTL4_INLINE MTL4::CommandQueueDescriptor* MTL4::CommandQueueDescriptor::alloc()
{
    return _MTL4_msg_MTL4__CommandQueueDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4CommandQueueDescriptor, nullptr);
}

_MTL4_INLINE MTL4::CommandQueueDescriptor* MTL4::CommandQueueDescriptor::init() const
{
    return _MTL4_msg_MTL4__CommandQueueDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE NS::String* MTL4::CommandQueueDescriptor::label() const
{
    return _MTL4_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::CommandQueueDescriptor::setLabel(NS::String* label)
{
    _MTL4_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL4_INLINE dispatch_queue_t MTL4::CommandQueueDescriptor::feedbackQueue() const
{
    return _MTL4_msg_dispatch_queue_t_feedbackQueue((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::CommandQueueDescriptor::setFeedbackQueue(dispatch_queue_t feedbackQueue)
{
    _MTL4_msg_v_setFeedbackQueue__dispatch_queue_t((const void*)this, nullptr, feedbackQueue);
}

_MTL4_INLINE MTL::Device* MTL4::CommandQueue::device() const
{
    return _MTL4_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL4_INLINE NS::String* MTL4::CommandQueue::label() const
{
    return _MTL4_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::CommandQueue::commit(const MTL4::CommandBuffer* const * commandBuffers, NS::UInteger count)
{
    _MTL4_msg_v_commit_count__constMTL4__CommandBufferpconstp_NS__UInteger((const void*)this, nullptr, commandBuffers, count);
}

_MTL4_INLINE void MTL4::CommandQueue::commit(const MTL4::CommandBuffer* const * commandBuffers, NS::UInteger count, MTL4::CommitOptions* options)
{
    _MTL4_msg_v_commit_count_options__constMTL4__CommandBufferpconstp_NS__UInteger_MTL4__CommitOptionsp((const void*)this, nullptr, commandBuffers, count, options);
}

_MTL4_INLINE void MTL4::CommandQueue::signalEvent(MTL::Event* event, uint64_t value)
{
    _MTL4_msg_v_signalEvent_value__MTL__Eventp_uint64_t((const void*)this, nullptr, event, value);
}

_MTL4_INLINE void MTL4::CommandQueue::wait(MTL::Event* event, uint64_t value)
{
    _MTL4_msg_v_waitForEvent_value__MTL__Eventp_uint64_t((const void*)this, nullptr, event, value);
}

_MTL4_INLINE void MTL4::CommandQueue::signalDrawable(MTL::Drawable* drawable)
{
    _MTL4_msg_v_signalDrawable__MTL__Drawablep((const void*)this, nullptr, drawable);
}

_MTL4_INLINE void MTL4::CommandQueue::wait(MTL::Drawable* drawable)
{
    _MTL4_msg_v_waitForDrawable__MTL__Drawablep((const void*)this, nullptr, drawable);
}

_MTL4_INLINE void MTL4::CommandQueue::addResidencySet(MTL::ResidencySet* residencySet)
{
    _MTL4_msg_v_addResidencySet__MTL__ResidencySetp((const void*)this, nullptr, residencySet);
}

_MTL4_INLINE void MTL4::CommandQueue::addResidencySets(const MTL::ResidencySet* const * residencySets, NS::UInteger count)
{
    _MTL4_msg_v_addResidencySets_count__constMTL__ResidencySetpconstp_NS__UInteger((const void*)this, nullptr, residencySets, count);
}

_MTL4_INLINE void MTL4::CommandQueue::removeResidencySet(MTL::ResidencySet* residencySet)
{
    _MTL4_msg_v_removeResidencySet__MTL__ResidencySetp((const void*)this, nullptr, residencySet);
}

_MTL4_INLINE void MTL4::CommandQueue::removeResidencySets(const MTL::ResidencySet* const * residencySets, NS::UInteger count)
{
    _MTL4_msg_v_removeResidencySets_count__constMTL__ResidencySetpconstp_NS__UInteger((const void*)this, nullptr, residencySets, count);
}

_MTL4_INLINE void MTL4::CommandQueue::updateTextureMappings(MTL::Texture* texture, MTL::Heap* heap, const MTL4::UpdateSparseTextureMappingOperation * operations, NS::UInteger count)
{
    _MTL4_msg_v_updateTextureMappings_heap_operations_count__MTL__Texturep_MTL__Heapp_constMTL4__UpdateSparseTextureMappingOperationp_NS__UInteger((const void*)this, nullptr, texture, heap, operations, count);
}

_MTL4_INLINE void MTL4::CommandQueue::copyTextureMappingsFromTexture(MTL::Texture* sourceTexture, MTL::Texture* destinationTexture, const MTL4::CopySparseTextureMappingOperation * operations, NS::UInteger count)
{
    _MTL4_msg_v_copyTextureMappingsFromTexture_toTexture_operations_count__MTL__Texturep_MTL__Texturep_constMTL4__CopySparseTextureMappingOperationp_NS__UInteger((const void*)this, nullptr, sourceTexture, destinationTexture, operations, count);
}

_MTL4_INLINE void MTL4::CommandQueue::updateBufferMappings(MTL::Buffer* buffer, MTL::Heap* heap, const MTL4::UpdateSparseBufferMappingOperation * operations, NS::UInteger count)
{
    _MTL4_msg_v_updateBufferMappings_heap_operations_count__MTL__Bufferp_MTL__Heapp_constMTL4__UpdateSparseBufferMappingOperationp_NS__UInteger((const void*)this, nullptr, buffer, heap, operations, count);
}

_MTL4_INLINE void MTL4::CommandQueue::copyBufferMappingsFromBuffer(MTL::Buffer* sourceBuffer, MTL::Buffer* destinationBuffer, const MTL4::CopySparseBufferMappingOperation * operations, NS::UInteger count)
{
    _MTL4_msg_v_copyBufferMappingsFromBuffer_toBuffer_operations_count__MTL__Bufferp_MTL__Bufferp_constMTL4__CopySparseBufferMappingOperationp_NS__UInteger((const void*)this, nullptr, sourceBuffer, destinationBuffer, operations, count);
}
