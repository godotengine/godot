#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class Device;
    class Fence;
    class LogState;
    class ResidencySet;
}
namespace MTL4 {
    class CommandAllocator;
    class ComputeCommandEncoder;
    class CounterHeap;
    class MachineLearningCommandEncoder;
    class RenderCommandEncoder;
    class RenderPassDescriptor;
    using RenderEncoderOptions = NS::UInteger;
}
namespace NS {
    class String;
}

namespace MTL4
{

class CommandBufferOptions;
class CommandBuffer;

class CommandBufferOptions : public NS::Copying<CommandBufferOptions>
{
public:
    static CommandBufferOptions* alloc();
    CommandBufferOptions*        init() const;

    MTL::LogState* logState() const;
    void           setLogState(MTL::LogState* logState);

};

class CommandBuffer : public NS::Referencing<CommandBuffer>
{
public:
    void                                 beginCommandBuffer(MTL4::CommandAllocator* allocator);
    void                                 beginCommandBuffer(MTL4::CommandAllocator* allocator, MTL4::CommandBufferOptions* options);
    MTL4::ComputeCommandEncoder*         computeCommandEncoder();
    MTL::Device*                         device() const;
    void                                 endCommandBuffer();
    NS::String*                          label() const;
    MTL4::MachineLearningCommandEncoder* machineLearningCommandEncoder();
    void                                 popDebugGroup();
    void                                 pushDebugGroup(NS::String* string);
    MTL4::RenderCommandEncoder*          renderCommandEncoder(MTL4::RenderPassDescriptor* descriptor);
    MTL4::RenderCommandEncoder*          renderCommandEncoder(MTL4::RenderPassDescriptor* descriptor, MTL4::RenderEncoderOptions options);
    void                                 resolveCounterHeap(MTL4::CounterHeap* counterHeap, NS::Range range, MTL4::BufferRange bufferRange, MTL::Fence* fenceToWait, MTL::Fence* fenceToUpdate);
    void                                 setLabel(NS::String* label);
    void                                 useResidencySet(MTL::ResidencySet* residencySet);
    void                                 useResidencySets(const MTL::ResidencySet* const * residencySets, NS::UInteger count);
    void                                 writeTimestampIntoHeap(MTL4::CounterHeap* counterHeap, NS::UInteger index);

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4CommandBufferOptions;
extern "C" void *OBJC_CLASS_$_MTL4CommandBuffer;

_MTL4_INLINE MTL4::CommandBufferOptions* MTL4::CommandBufferOptions::alloc()
{
    return _MTL4_msg_MTL4__CommandBufferOptionsp_alloc((const void*)&OBJC_CLASS_$_MTL4CommandBufferOptions, nullptr);
}

_MTL4_INLINE MTL4::CommandBufferOptions* MTL4::CommandBufferOptions::init() const
{
    return _MTL4_msg_MTL4__CommandBufferOptionsp_init((const void*)this, nullptr);
}

_MTL4_INLINE MTL::LogState* MTL4::CommandBufferOptions::logState() const
{
    return _MTL4_msg_MTL__LogStatep_logState((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::CommandBufferOptions::setLogState(MTL::LogState* logState)
{
    _MTL4_msg_v_setLogState__MTL__LogStatep((const void*)this, nullptr, logState);
}

_MTL4_INLINE MTL::Device* MTL4::CommandBuffer::device() const
{
    return _MTL4_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL4_INLINE NS::String* MTL4::CommandBuffer::label() const
{
    return _MTL4_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::CommandBuffer::setLabel(NS::String* label)
{
    _MTL4_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL4_INLINE void MTL4::CommandBuffer::beginCommandBuffer(MTL4::CommandAllocator* allocator)
{
    _MTL4_msg_v_beginCommandBufferWithAllocator__MTL4__CommandAllocatorp((const void*)this, nullptr, allocator);
}

_MTL4_INLINE void MTL4::CommandBuffer::beginCommandBuffer(MTL4::CommandAllocator* allocator, MTL4::CommandBufferOptions* options)
{
    _MTL4_msg_v_beginCommandBufferWithAllocator_options__MTL4__CommandAllocatorp_MTL4__CommandBufferOptionsp((const void*)this, nullptr, allocator, options);
}

_MTL4_INLINE void MTL4::CommandBuffer::endCommandBuffer()
{
    _MTL4_msg_v_endCommandBuffer((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::RenderCommandEncoder* MTL4::CommandBuffer::renderCommandEncoder(MTL4::RenderPassDescriptor* descriptor)
{
    return _MTL4_msg_MTL4__RenderCommandEncoderp_renderCommandEncoderWithDescriptor__MTL4__RenderPassDescriptorp((const void*)this, nullptr, descriptor);
}

_MTL4_INLINE MTL4::RenderCommandEncoder* MTL4::CommandBuffer::renderCommandEncoder(MTL4::RenderPassDescriptor* descriptor, MTL4::RenderEncoderOptions options)
{
    return _MTL4_msg_MTL4__RenderCommandEncoderp_renderCommandEncoderWithDescriptor_options__MTL4__RenderPassDescriptorp_MTL4__RenderEncoderOptions((const void*)this, nullptr, descriptor, options);
}

_MTL4_INLINE MTL4::ComputeCommandEncoder* MTL4::CommandBuffer::computeCommandEncoder()
{
    return _MTL4_msg_MTL4__ComputeCommandEncoderp_computeCommandEncoder((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::MachineLearningCommandEncoder* MTL4::CommandBuffer::machineLearningCommandEncoder()
{
    return _MTL4_msg_MTL4__MachineLearningCommandEncoderp_machineLearningCommandEncoder((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::CommandBuffer::useResidencySet(MTL::ResidencySet* residencySet)
{
    _MTL4_msg_v_useResidencySet__MTL__ResidencySetp((const void*)this, nullptr, residencySet);
}

_MTL4_INLINE void MTL4::CommandBuffer::useResidencySets(const MTL::ResidencySet* const * residencySets, NS::UInteger count)
{
    _MTL4_msg_v_useResidencySets_count__constMTL__ResidencySetpconstp_NS__UInteger((const void*)this, nullptr, residencySets, count);
}

_MTL4_INLINE void MTL4::CommandBuffer::pushDebugGroup(NS::String* string)
{
    _MTL4_msg_v_pushDebugGroup__NS__Stringp((const void*)this, nullptr, string);
}

_MTL4_INLINE void MTL4::CommandBuffer::popDebugGroup()
{
    _MTL4_msg_v_popDebugGroup((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::CommandBuffer::writeTimestampIntoHeap(MTL4::CounterHeap* counterHeap, NS::UInteger index)
{
    _MTL4_msg_v_writeTimestampIntoHeap_atIndex__MTL4__CounterHeapp_NS__UInteger((const void*)this, nullptr, counterHeap, index);
}

_MTL4_INLINE void MTL4::CommandBuffer::resolveCounterHeap(MTL4::CounterHeap* counterHeap, NS::Range range, MTL4::BufferRange bufferRange, MTL::Fence* fenceToWait, MTL::Fence* fenceToUpdate)
{
    _MTL4_msg_v_resolveCounterHeap_withRange_intoBuffer_waitFence_updateFence__MTL4__CounterHeapp_NS__Range_MTL4__BufferRange_MTL__Fencep_MTL__Fencep((const void*)this, nullptr, counterHeap, range, bufferRange, fenceToWait, fenceToUpdate);
}
