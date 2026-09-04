#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class AccelerationStructureCommandEncoder;
    class AccelerationStructurePassDescriptor;
    class BlitCommandEncoder;
    class BlitPassDescriptor;
    class CommandQueue;
    class ComputeCommandEncoder;
    class ComputePassDescriptor;
    class Device;
    class Drawable;
    class Event;
    class LogContainer;
    class LogState;
    class ParallelRenderCommandEncoder;
    class RenderCommandEncoder;
    class RenderPassDescriptor;
    class ResidencySet;
    class ResourceStateCommandEncoder;
    class ResourceStatePassDescriptor;
}
namespace NS {
    class Array;
    class Error;
    class String;
}

namespace MTL
{

extern NS::ErrorDomain const CommandBufferErrorDomain __asm__("_MTLCommandBufferErrorDomain");
extern NS::ErrorUserInfoKey const CommandBufferEncoderInfoErrorKey __asm__("_MTLCommandBufferEncoderInfoErrorKey");
_MTL_ENUM(NS::UInteger, CommandBufferStatus) {
    CommandBufferStatusNotEnqueued = 0,
    CommandBufferStatusEnqueued = 1,
    CommandBufferStatusCommitted = 2,
    CommandBufferStatusScheduled = 3,
    CommandBufferStatusCompleted = 4,
    CommandBufferStatusError = 5,
};

_MTL_ENUM(NS::UInteger, CommandBufferError) {
    CommandBufferErrorNone = 0,
    CommandBufferErrorInternal = 1,
    CommandBufferErrorTimeout = 2,
    CommandBufferErrorPageFault = 3,
    CommandBufferErrorBlacklisted = 4,
    CommandBufferErrorAccessRevoked = 4,
    CommandBufferErrorNotPermitted = 7,
    CommandBufferErrorOutOfMemory = 8,
    CommandBufferErrorInvalidResource = 9,
    CommandBufferErrorMemoryless = 10,
    CommandBufferErrorDeviceRemoved = 11,
    CommandBufferErrorStackOverflow = 12,
};

_MTL_OPTIONS(NS::UInteger, CommandBufferErrorOption) {
    CommandBufferErrorOptionNone = 0,
    CommandBufferErrorOptionEncoderExecutionStatus = 1 << 0,
};

_MTL_ENUM(NS::Integer, CommandEncoderErrorState) {
    CommandEncoderErrorStateUnknown = 0,
    CommandEncoderErrorStateCompleted = 1,
    CommandEncoderErrorStateAffected = 2,
    CommandEncoderErrorStatePending = 3,
    CommandEncoderErrorStateFaulted = 4,
};

_MTL_ENUM(NS::UInteger, DispatchType) {
    DispatchTypeSerial = 0,
    DispatchTypeConcurrent = 1,
};


class CommandBufferDescriptor;
class CommandBufferEncoderInfo;
class CommandBuffer;

class CommandBufferDescriptor : public NS::Copying<CommandBufferDescriptor>
{
public:
    static CommandBufferDescriptor* alloc();
    CommandBufferDescriptor*        init() const;

    MTL::CommandBufferErrorOption errorOptions() const;
    MTL::LogState*                logState() const;
    bool                          retainedReferences() const;
    void                          setErrorOptions(MTL::CommandBufferErrorOption errorOptions);
    void                          setLogState(MTL::LogState* logState);
    void                          setRetainedReferences(bool retainedReferences);

};

class CommandBufferEncoderInfo : public NS::Referencing<CommandBufferEncoderInfo>
{
public:
    NS::Array*                    debugSignposts() const;
    MTL::CommandEncoderErrorState errorState() const;
    NS::String*                   label() const;

};

class CommandBuffer : public NS::Referencing<CommandBuffer>
{
public:
    CFTimeInterval                            GPUEndTime() const;
    CFTimeInterval                            GPUStartTime() const;
    MTL::AccelerationStructureCommandEncoder* accelerationStructureCommandEncoder();
    MTL::AccelerationStructureCommandEncoder* accelerationStructureCommandEncoder(MTL::AccelerationStructurePassDescriptor* descriptor);
    void                                      addCompletedHandler(MTL::CommandBufferHandler block);
    void                                      addCompletedHandler(const MTL::CommandBufferHandlerFunction& block);
    void                                      addScheduledHandler(MTL::CommandBufferHandler block);
    void                                      addScheduledHandler(const MTL::CommandBufferHandlerFunction& block);
    MTL::BlitCommandEncoder*                  blitCommandEncoder();
    MTL::BlitCommandEncoder*                  blitCommandEncoder(MTL::BlitPassDescriptor* blitPassDescriptor);
    MTL::CommandQueue*                        commandQueue() const;
    void                                      commit();
    MTL::ComputeCommandEncoder*               computeCommandEncoder(MTL::ComputePassDescriptor* computePassDescriptor);
    MTL::ComputeCommandEncoder*               computeCommandEncoder();
    MTL::ComputeCommandEncoder*               computeCommandEncoder(MTL::DispatchType dispatchType);
    MTL::Device*                              device() const;
    void                                      encodeSignalEvent(MTL::Event* event, uint64_t value);
    void                                      encodeWait(MTL::Event* event, uint64_t value);
    void                                      enqueue();
    NS::Error*                                error() const;
    MTL::CommandBufferErrorOption             errorOptions() const;
    CFTimeInterval                            kernelEndTime() const;
    CFTimeInterval                            kernelStartTime() const;
    NS::String*                               label() const;
    MTL::LogContainer*                        logs() const;
    MTL::ParallelRenderCommandEncoder*        parallelRenderCommandEncoder(MTL::RenderPassDescriptor* renderPassDescriptor);
    void                                      popDebugGroup();
    void                                      presentDrawable(MTL::Drawable* drawable);
    void                                      presentDrawableAfterMinimumDuration(MTL::Drawable* drawable, CFTimeInterval duration);
    void                                      presentDrawableAtTime(MTL::Drawable* drawable, CFTimeInterval presentationTime);
    void                                      pushDebugGroup(NS::String* string);
    MTL::RenderCommandEncoder*                renderCommandEncoder(MTL::RenderPassDescriptor* renderPassDescriptor);
    MTL::ResourceStateCommandEncoder*         resourceStateCommandEncoder();
    MTL::ResourceStateCommandEncoder*         resourceStateCommandEncoder(MTL::ResourceStatePassDescriptor* resourceStatePassDescriptor);
    bool                                      retainedReferences() const;
    void                                      setLabel(NS::String* label);
    MTL::CommandBufferStatus                  status() const;
    void                                      useResidencySet(MTL::ResidencySet* residencySet);
    void                                      useResidencySets(const MTL::ResidencySet* const * residencySets, NS::UInteger count);
    void                                      waitUntilCompleted();
    void                                      waitUntilScheduled();

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLCommandBufferDescriptor;
extern "C" void *OBJC_CLASS_$_MTLCommandBufferEncoderInfo;
extern "C" void *OBJC_CLASS_$_MTLCommandBuffer;

_MTL_INLINE MTL::CommandBufferDescriptor* MTL::CommandBufferDescriptor::alloc()
{
    return _MTL_msg_MTL__CommandBufferDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLCommandBufferDescriptor, nullptr);
}

_MTL_INLINE MTL::CommandBufferDescriptor* MTL::CommandBufferDescriptor::init() const
{
    return _MTL_msg_MTL__CommandBufferDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::CommandBufferDescriptor::retainedReferences() const
{
    return _MTL_msg_bool_retainedReferences((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CommandBufferDescriptor::setRetainedReferences(bool retainedReferences)
{
    _MTL_msg_v_setRetainedReferences__bool((const void*)this, nullptr, retainedReferences);
}

_MTL_INLINE MTL::CommandBufferErrorOption MTL::CommandBufferDescriptor::errorOptions() const
{
    return _MTL_msg_MTL__CommandBufferErrorOption_errorOptions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CommandBufferDescriptor::setErrorOptions(MTL::CommandBufferErrorOption errorOptions)
{
    _MTL_msg_v_setErrorOptions__MTL__CommandBufferErrorOption((const void*)this, nullptr, errorOptions);
}

_MTL_INLINE MTL::LogState* MTL::CommandBufferDescriptor::logState() const
{
    return _MTL_msg_MTL__LogStatep_logState((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CommandBufferDescriptor::setLogState(MTL::LogState* logState)
{
    _MTL_msg_v_setLogState__MTL__LogStatep((const void*)this, nullptr, logState);
}

_MTL_INLINE NS::String* MTL::CommandBufferEncoderInfo::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE NS::Array* MTL::CommandBufferEncoderInfo::debugSignposts() const
{
    return _MTL_msg_NS__Arrayp_debugSignposts((const void*)this, nullptr);
}

_MTL_INLINE MTL::CommandEncoderErrorState MTL::CommandBufferEncoderInfo::errorState() const
{
    return _MTL_msg_MTL__CommandEncoderErrorState_errorState((const void*)this, nullptr);
}

_MTL_INLINE MTL::Device* MTL::CommandBuffer::device() const
{
    return _MTL_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL_INLINE MTL::CommandQueue* MTL::CommandBuffer::commandQueue() const
{
    return _MTL_msg_MTL__CommandQueuep_commandQueue((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::CommandBuffer::retainedReferences() const
{
    return _MTL_msg_bool_retainedReferences((const void*)this, nullptr);
}

_MTL_INLINE MTL::CommandBufferErrorOption MTL::CommandBuffer::errorOptions() const
{
    return _MTL_msg_MTL__CommandBufferErrorOption_errorOptions((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::CommandBuffer::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CommandBuffer::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE CFTimeInterval MTL::CommandBuffer::kernelStartTime() const
{
    return _MTL_msg_CFTimeInterval_kernelStartTime((const void*)this, nullptr);
}

_MTL_INLINE CFTimeInterval MTL::CommandBuffer::kernelEndTime() const
{
    return _MTL_msg_CFTimeInterval_kernelEndTime((const void*)this, nullptr);
}

_MTL_INLINE MTL::LogContainer* MTL::CommandBuffer::logs() const
{
    return _MTL_msg_MTL__LogContainerp_logs((const void*)this, nullptr);
}

_MTL_INLINE CFTimeInterval MTL::CommandBuffer::GPUStartTime() const
{
    return _MTL_msg_CFTimeInterval_GPUStartTime((const void*)this, nullptr);
}

_MTL_INLINE CFTimeInterval MTL::CommandBuffer::GPUEndTime() const
{
    return _MTL_msg_CFTimeInterval_GPUEndTime((const void*)this, nullptr);
}

_MTL_INLINE MTL::CommandBufferStatus MTL::CommandBuffer::status() const
{
    return _MTL_msg_MTL__CommandBufferStatus_status((const void*)this, nullptr);
}

_MTL_INLINE NS::Error* MTL::CommandBuffer::error() const
{
    return _MTL_msg_NS__Errorp_error((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CommandBuffer::enqueue()
{
    _MTL_msg_v_enqueue((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CommandBuffer::commit()
{
    _MTL_msg_v_commit((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CommandBuffer::addScheduledHandler(MTL::CommandBufferHandler block)
{
    _MTL_msg_v_addScheduledHandler__MTL__CommandBufferHandler((const void*)this, nullptr, block);
}

_MTL_INLINE void MTL::CommandBuffer::addScheduledHandler(const MTL::CommandBufferHandlerFunction& block)
{
    __block MTL::CommandBufferHandlerFunction blockFunction = block;
    addScheduledHandler(^(MTL::CommandBuffer* x0) { blockFunction(x0); });
}

_MTL_INLINE void MTL::CommandBuffer::presentDrawable(MTL::Drawable* drawable)
{
    _MTL_msg_v_presentDrawable__MTL__Drawablep((const void*)this, nullptr, drawable);
}

_MTL_INLINE void MTL::CommandBuffer::presentDrawableAtTime(MTL::Drawable* drawable, CFTimeInterval presentationTime)
{
    _MTL_msg_v_presentDrawable_atTime__MTL__Drawablep_CFTimeInterval((const void*)this, nullptr, drawable, presentationTime);
}

_MTL_INLINE void MTL::CommandBuffer::presentDrawableAfterMinimumDuration(MTL::Drawable* drawable, CFTimeInterval duration)
{
    _MTL_msg_v_presentDrawable_afterMinimumDuration__MTL__Drawablep_CFTimeInterval((const void*)this, nullptr, drawable, duration);
}

_MTL_INLINE void MTL::CommandBuffer::waitUntilScheduled()
{
    _MTL_msg_v_waitUntilScheduled((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CommandBuffer::addCompletedHandler(MTL::CommandBufferHandler block)
{
    _MTL_msg_v_addCompletedHandler__MTL__CommandBufferHandler((const void*)this, nullptr, block);
}

_MTL_INLINE void MTL::CommandBuffer::addCompletedHandler(const MTL::CommandBufferHandlerFunction& block)
{
    __block MTL::CommandBufferHandlerFunction blockFunction = block;
    addCompletedHandler(^(MTL::CommandBuffer* x0) { blockFunction(x0); });
}

_MTL_INLINE void MTL::CommandBuffer::waitUntilCompleted()
{
    _MTL_msg_v_waitUntilCompleted((const void*)this, nullptr);
}

_MTL_INLINE MTL::BlitCommandEncoder* MTL::CommandBuffer::blitCommandEncoder()
{
    return _MTL_msg_MTL__BlitCommandEncoderp_blitCommandEncoder((const void*)this, nullptr);
}

_MTL_INLINE MTL::RenderCommandEncoder* MTL::CommandBuffer::renderCommandEncoder(MTL::RenderPassDescriptor* renderPassDescriptor)
{
    return _MTL_msg_MTL__RenderCommandEncoderp_renderCommandEncoderWithDescriptor__MTL__RenderPassDescriptorp((const void*)this, nullptr, renderPassDescriptor);
}

_MTL_INLINE MTL::ComputeCommandEncoder* MTL::CommandBuffer::computeCommandEncoder(MTL::ComputePassDescriptor* computePassDescriptor)
{
    return _MTL_msg_MTL__ComputeCommandEncoderp_computeCommandEncoderWithDescriptor__MTL__ComputePassDescriptorp((const void*)this, nullptr, computePassDescriptor);
}

_MTL_INLINE MTL::BlitCommandEncoder* MTL::CommandBuffer::blitCommandEncoder(MTL::BlitPassDescriptor* blitPassDescriptor)
{
    return _MTL_msg_MTL__BlitCommandEncoderp_blitCommandEncoderWithDescriptor__MTL__BlitPassDescriptorp((const void*)this, nullptr, blitPassDescriptor);
}

_MTL_INLINE MTL::ComputeCommandEncoder* MTL::CommandBuffer::computeCommandEncoder()
{
    return _MTL_msg_MTL__ComputeCommandEncoderp_computeCommandEncoder((const void*)this, nullptr);
}

_MTL_INLINE MTL::ComputeCommandEncoder* MTL::CommandBuffer::computeCommandEncoder(MTL::DispatchType dispatchType)
{
    return _MTL_msg_MTL__ComputeCommandEncoderp_computeCommandEncoderWithDispatchType__MTL__DispatchType((const void*)this, nullptr, dispatchType);
}

_MTL_INLINE void MTL::CommandBuffer::encodeWait(MTL::Event* event, uint64_t value)
{
    _MTL_msg_v_encodeWaitForEvent_value__MTL__Eventp_uint64_t((const void*)this, nullptr, event, value);
}

_MTL_INLINE void MTL::CommandBuffer::encodeSignalEvent(MTL::Event* event, uint64_t value)
{
    _MTL_msg_v_encodeSignalEvent_value__MTL__Eventp_uint64_t((const void*)this, nullptr, event, value);
}

_MTL_INLINE MTL::ParallelRenderCommandEncoder* MTL::CommandBuffer::parallelRenderCommandEncoder(MTL::RenderPassDescriptor* renderPassDescriptor)
{
    return _MTL_msg_MTL__ParallelRenderCommandEncoderp_parallelRenderCommandEncoderWithDescriptor__MTL__RenderPassDescriptorp((const void*)this, nullptr, renderPassDescriptor);
}

_MTL_INLINE MTL::ResourceStateCommandEncoder* MTL::CommandBuffer::resourceStateCommandEncoder()
{
    return _MTL_msg_MTL__ResourceStateCommandEncoderp_resourceStateCommandEncoder((const void*)this, nullptr);
}

_MTL_INLINE MTL::ResourceStateCommandEncoder* MTL::CommandBuffer::resourceStateCommandEncoder(MTL::ResourceStatePassDescriptor* resourceStatePassDescriptor)
{
    return _MTL_msg_MTL__ResourceStateCommandEncoderp_resourceStateCommandEncoderWithDescriptor__MTL__ResourceStatePassDescriptorp((const void*)this, nullptr, resourceStatePassDescriptor);
}

_MTL_INLINE MTL::AccelerationStructureCommandEncoder* MTL::CommandBuffer::accelerationStructureCommandEncoder()
{
    return _MTL_msg_MTL__AccelerationStructureCommandEncoderp_accelerationStructureCommandEncoder((const void*)this, nullptr);
}

_MTL_INLINE MTL::AccelerationStructureCommandEncoder* MTL::CommandBuffer::accelerationStructureCommandEncoder(MTL::AccelerationStructurePassDescriptor* descriptor)
{
    return _MTL_msg_MTL__AccelerationStructureCommandEncoderp_accelerationStructureCommandEncoderWithDescriptor__MTL__AccelerationStructurePassDescriptorp((const void*)this, nullptr, descriptor);
}

_MTL_INLINE void MTL::CommandBuffer::pushDebugGroup(NS::String* string)
{
    _MTL_msg_v_pushDebugGroup__NS__Stringp((const void*)this, nullptr, string);
}

_MTL_INLINE void MTL::CommandBuffer::popDebugGroup()
{
    _MTL_msg_v_popDebugGroup((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CommandBuffer::useResidencySet(MTL::ResidencySet* residencySet)
{
    _MTL_msg_v_useResidencySet__MTL__ResidencySetp((const void*)this, nullptr, residencySet);
}

_MTL_INLINE void MTL::CommandBuffer::useResidencySets(const MTL::ResidencySet* const * residencySets, NS::UInteger count)
{
    _MTL_msg_v_useResidencySets_count__constMTL__ResidencySetpconstp_NS__UInteger((const void*)this, nullptr, residencySets, count);
}
