//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLCommandBuffer.hpp
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
#include <CoreFoundation/CoreFoundation.h>
#include <cstdint>

#include <functional>

namespace MTL
{
class AccelerationStructureCommandEncoder;
class AccelerationStructurePassDescriptor;
class BlitCommandEncoder;
class BlitPassDescriptor;
class CommandBuffer;
class CommandBufferDescriptor;
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

_MTL_OPTIONS(NS::UInteger, CommandBufferErrorOption) {
    CommandBufferErrorOptionNone = 0,
    CommandBufferErrorOptionEncoderExecutionStatus = 1,
};

using CommandBufferHandler = void (^)(CommandBuffer*);
using HandlerFunction = std::function<void(CommandBuffer*)>;

class CommandBufferDescriptor : public NS::Copying<CommandBufferDescriptor>
{
public:
    static CommandBufferDescriptor* alloc();

    CommandBufferErrorOption        errorOptions() const;

    CommandBufferDescriptor*        init();

    LogState*                       logState() const;

    bool                            retainedReferences() const;

    void                            setErrorOptions(MTL::CommandBufferErrorOption errorOptions);

    void                            setLogState(const MTL::LogState* logState);

    void                            setRetainedReferences(bool retainedReferences);
};
class CommandBufferEncoderInfo : public NS::Referencing<CommandBufferEncoderInfo>
{
public:
    NS::Array*               debugSignposts() const;

    CommandEncoderErrorState errorState() const;

    NS::String*              label() const;
};
class CommandBuffer : public NS::Referencing<CommandBuffer>
{
public:
    CFTimeInterval                       GPUEndTime() const;

    CFTimeInterval                       GPUStartTime() const;

    AccelerationStructureCommandEncoder* accelerationStructureCommandEncoder();
    AccelerationStructureCommandEncoder* accelerationStructureCommandEncoder(const MTL::AccelerationStructurePassDescriptor* descriptor);

    void                                 addCompletedHandler(const MTL::CommandBufferHandler block);
    void                                 addCompletedHandler(const MTL::HandlerFunction& function);

    void                                 addScheduledHandler(const MTL::CommandBufferHandler block);
    void                                 addScheduledHandler(const MTL::HandlerFunction& function);

    BlitCommandEncoder*                  blitCommandEncoder();
    BlitCommandEncoder*                  blitCommandEncoder(const MTL::BlitPassDescriptor* blitPassDescriptor);

    CommandQueue*                        commandQueue() const;

    void                                 commit();

    ComputeCommandEncoder*               computeCommandEncoder(const MTL::ComputePassDescriptor* computePassDescriptor);
    ComputeCommandEncoder*               computeCommandEncoder();
    ComputeCommandEncoder*               computeCommandEncoder(MTL::DispatchType dispatchType);

    Device*                              device() const;

    void                                 encodeSignalEvent(const MTL::Event* event, uint64_t value);

    void                                 encodeWait(const MTL::Event* event, uint64_t value);

    void                                 enqueue();

    NS::Error*                           error() const;
    CommandBufferErrorOption             errorOptions() const;

    CFTimeInterval                       kernelEndTime() const;

    CFTimeInterval                       kernelStartTime() const;

    NS::String*                          label() const;

    LogContainer*                        logs() const;

    ParallelRenderCommandEncoder*        parallelRenderCommandEncoder(const MTL::RenderPassDescriptor* renderPassDescriptor);

    void                                 popDebugGroup();

    void                                 presentDrawable(const MTL::Drawable* drawable);
    void                                 presentDrawableAfterMinimumDuration(const MTL::Drawable* drawable, CFTimeInterval duration);

    void                                 presentDrawableAtTime(const MTL::Drawable* drawable, CFTimeInterval presentationTime);

    void                                 pushDebugGroup(const NS::String* string);

    RenderCommandEncoder*                renderCommandEncoder(const MTL::RenderPassDescriptor* renderPassDescriptor);

    ResourceStateCommandEncoder*         resourceStateCommandEncoder();
    ResourceStateCommandEncoder*         resourceStateCommandEncoder(const MTL::ResourceStatePassDescriptor* resourceStatePassDescriptor);

    bool                                 retainedReferences() const;

    void                                 setLabel(const NS::String* label);

    CommandBufferStatus                  status() const;

    void                                 useResidencySet(const MTL::ResidencySet* residencySet);
    void                                 useResidencySets(const MTL::ResidencySet* const residencySets[], NS::UInteger count);

    void                                 waitUntilCompleted();

    void                                 waitUntilScheduled();
};

}
_MTL_INLINE MTL::CommandBufferDescriptor* MTL::CommandBufferDescriptor::alloc()
{
    return NS::Object::alloc<MTL::CommandBufferDescriptor>(_MTL_PRIVATE_CLS(MTLCommandBufferDescriptor));
}

_MTL_INLINE MTL::CommandBufferErrorOption MTL::CommandBufferDescriptor::errorOptions() const
{
    return Object::sendMessage<MTL::CommandBufferErrorOption>(this, _MTL_PRIVATE_SEL(errorOptions));
}

_MTL_INLINE MTL::CommandBufferDescriptor* MTL::CommandBufferDescriptor::init()
{
    return NS::Object::init<MTL::CommandBufferDescriptor>();
}

_MTL_INLINE MTL::LogState* MTL::CommandBufferDescriptor::logState() const
{
    return Object::sendMessage<MTL::LogState*>(this, _MTL_PRIVATE_SEL(logState));
}

_MTL_INLINE bool MTL::CommandBufferDescriptor::retainedReferences() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(retainedReferences));
}

_MTL_INLINE void MTL::CommandBufferDescriptor::setErrorOptions(MTL::CommandBufferErrorOption errorOptions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setErrorOptions_), errorOptions);
}

_MTL_INLINE void MTL::CommandBufferDescriptor::setLogState(const MTL::LogState* logState)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLogState_), logState);
}

_MTL_INLINE void MTL::CommandBufferDescriptor::setRetainedReferences(bool retainedReferences)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRetainedReferences_), retainedReferences);
}

_MTL_INLINE NS::Array* MTL::CommandBufferEncoderInfo::debugSignposts() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(debugSignposts));
}

_MTL_INLINE MTL::CommandEncoderErrorState MTL::CommandBufferEncoderInfo::errorState() const
{
    return Object::sendMessage<MTL::CommandEncoderErrorState>(this, _MTL_PRIVATE_SEL(errorState));
}

_MTL_INLINE NS::String* MTL::CommandBufferEncoderInfo::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE CFTimeInterval MTL::CommandBuffer::GPUEndTime() const
{
    return Object::sendMessage<CFTimeInterval>(this, _MTL_PRIVATE_SEL(GPUEndTime));
}

_MTL_INLINE CFTimeInterval MTL::CommandBuffer::GPUStartTime() const
{
    return Object::sendMessage<CFTimeInterval>(this, _MTL_PRIVATE_SEL(GPUStartTime));
}

_MTL_INLINE MTL::AccelerationStructureCommandEncoder* MTL::CommandBuffer::accelerationStructureCommandEncoder()
{
    return Object::sendMessage<MTL::AccelerationStructureCommandEncoder*>(this, _MTL_PRIVATE_SEL(accelerationStructureCommandEncoder));
}

_MTL_INLINE MTL::AccelerationStructureCommandEncoder* MTL::CommandBuffer::accelerationStructureCommandEncoder(const MTL::AccelerationStructurePassDescriptor* descriptor)
{
    return Object::sendMessage<MTL::AccelerationStructureCommandEncoder*>(this, _MTL_PRIVATE_SEL(accelerationStructureCommandEncoderWithDescriptor_), descriptor);
}

_MTL_INLINE void MTL::CommandBuffer::addCompletedHandler(const MTL::CommandBufferHandler block)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(addCompletedHandler_), block);
}

_MTL_INLINE void MTL::CommandBuffer::addCompletedHandler(const MTL::HandlerFunction& function)
{
    __block HandlerFunction blockFunction = function;
    addCompletedHandler(^(MTL::CommandBuffer* pCommandBuffer) { blockFunction(pCommandBuffer); });
}

_MTL_INLINE void MTL::CommandBuffer::addScheduledHandler(const MTL::CommandBufferHandler block)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(addScheduledHandler_), block);
}

_MTL_INLINE void MTL::CommandBuffer::addScheduledHandler(const MTL::HandlerFunction& function)
{
    __block HandlerFunction blockFunction = function;
    addScheduledHandler(^(MTL::CommandBuffer* pCommandBuffer) { blockFunction(pCommandBuffer); });
}

_MTL_INLINE MTL::BlitCommandEncoder* MTL::CommandBuffer::blitCommandEncoder()
{
    return Object::sendMessage<MTL::BlitCommandEncoder*>(this, _MTL_PRIVATE_SEL(blitCommandEncoder));
}

_MTL_INLINE MTL::BlitCommandEncoder* MTL::CommandBuffer::blitCommandEncoder(const MTL::BlitPassDescriptor* blitPassDescriptor)
{
    return Object::sendMessage<MTL::BlitCommandEncoder*>(this, _MTL_PRIVATE_SEL(blitCommandEncoderWithDescriptor_), blitPassDescriptor);
}

_MTL_INLINE MTL::CommandQueue* MTL::CommandBuffer::commandQueue() const
{
    return Object::sendMessage<MTL::CommandQueue*>(this, _MTL_PRIVATE_SEL(commandQueue));
}

_MTL_INLINE void MTL::CommandBuffer::commit()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(commit));
}

_MTL_INLINE MTL::ComputeCommandEncoder* MTL::CommandBuffer::computeCommandEncoder(const MTL::ComputePassDescriptor* computePassDescriptor)
{
    return Object::sendMessage<MTL::ComputeCommandEncoder*>(this, _MTL_PRIVATE_SEL(computeCommandEncoderWithDescriptor_), computePassDescriptor);
}

_MTL_INLINE MTL::ComputeCommandEncoder* MTL::CommandBuffer::computeCommandEncoder()
{
    return Object::sendMessage<MTL::ComputeCommandEncoder*>(this, _MTL_PRIVATE_SEL(computeCommandEncoder));
}

_MTL_INLINE MTL::ComputeCommandEncoder* MTL::CommandBuffer::computeCommandEncoder(MTL::DispatchType dispatchType)
{
    return Object::sendMessage<MTL::ComputeCommandEncoder*>(this, _MTL_PRIVATE_SEL(computeCommandEncoderWithDispatchType_), dispatchType);
}

_MTL_INLINE MTL::Device* MTL::CommandBuffer::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE void MTL::CommandBuffer::encodeSignalEvent(const MTL::Event* event, uint64_t value)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(encodeSignalEvent_value_), event, value);
}

_MTL_INLINE void MTL::CommandBuffer::encodeWait(const MTL::Event* event, uint64_t value)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(encodeWaitForEvent_value_), event, value);
}

_MTL_INLINE void MTL::CommandBuffer::enqueue()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(enqueue));
}

_MTL_INLINE NS::Error* MTL::CommandBuffer::error() const
{
    return Object::sendMessage<NS::Error*>(this, _MTL_PRIVATE_SEL(error));
}

_MTL_INLINE MTL::CommandBufferErrorOption MTL::CommandBuffer::errorOptions() const
{
    return Object::sendMessage<MTL::CommandBufferErrorOption>(this, _MTL_PRIVATE_SEL(errorOptions));
}

_MTL_INLINE CFTimeInterval MTL::CommandBuffer::kernelEndTime() const
{
    return Object::sendMessage<CFTimeInterval>(this, _MTL_PRIVATE_SEL(kernelEndTime));
}

_MTL_INLINE CFTimeInterval MTL::CommandBuffer::kernelStartTime() const
{
    return Object::sendMessage<CFTimeInterval>(this, _MTL_PRIVATE_SEL(kernelStartTime));
}

_MTL_INLINE NS::String* MTL::CommandBuffer::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE MTL::LogContainer* MTL::CommandBuffer::logs() const
{
    return Object::sendMessage<MTL::LogContainer*>(this, _MTL_PRIVATE_SEL(logs));
}

_MTL_INLINE MTL::ParallelRenderCommandEncoder* MTL::CommandBuffer::parallelRenderCommandEncoder(const MTL::RenderPassDescriptor* renderPassDescriptor)
{
    return Object::sendMessage<MTL::ParallelRenderCommandEncoder*>(this, _MTL_PRIVATE_SEL(parallelRenderCommandEncoderWithDescriptor_), renderPassDescriptor);
}

_MTL_INLINE void MTL::CommandBuffer::popDebugGroup()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(popDebugGroup));
}

_MTL_INLINE void MTL::CommandBuffer::presentDrawable(const MTL::Drawable* drawable)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(presentDrawable_), drawable);
}

_MTL_INLINE void MTL::CommandBuffer::presentDrawableAfterMinimumDuration(const MTL::Drawable* drawable, CFTimeInterval duration)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(presentDrawable_afterMinimumDuration_), drawable, duration);
}

_MTL_INLINE void MTL::CommandBuffer::presentDrawableAtTime(const MTL::Drawable* drawable, CFTimeInterval presentationTime)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(presentDrawable_atTime_), drawable, presentationTime);
}

_MTL_INLINE void MTL::CommandBuffer::pushDebugGroup(const NS::String* string)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(pushDebugGroup_), string);
}

_MTL_INLINE MTL::RenderCommandEncoder* MTL::CommandBuffer::renderCommandEncoder(const MTL::RenderPassDescriptor* renderPassDescriptor)
{
    return Object::sendMessage<MTL::RenderCommandEncoder*>(this, _MTL_PRIVATE_SEL(renderCommandEncoderWithDescriptor_), renderPassDescriptor);
}

_MTL_INLINE MTL::ResourceStateCommandEncoder* MTL::CommandBuffer::resourceStateCommandEncoder()
{
    return Object::sendMessage<MTL::ResourceStateCommandEncoder*>(this, _MTL_PRIVATE_SEL(resourceStateCommandEncoder));
}

_MTL_INLINE MTL::ResourceStateCommandEncoder* MTL::CommandBuffer::resourceStateCommandEncoder(const MTL::ResourceStatePassDescriptor* resourceStatePassDescriptor)
{
    return Object::sendMessage<MTL::ResourceStateCommandEncoder*>(this, _MTL_PRIVATE_SEL(resourceStateCommandEncoderWithDescriptor_), resourceStatePassDescriptor);
}

_MTL_INLINE bool MTL::CommandBuffer::retainedReferences() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(retainedReferences));
}

_MTL_INLINE void MTL::CommandBuffer::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE MTL::CommandBufferStatus MTL::CommandBuffer::status() const
{
    return Object::sendMessage<MTL::CommandBufferStatus>(this, _MTL_PRIVATE_SEL(status));
}

_MTL_INLINE void MTL::CommandBuffer::useResidencySet(const MTL::ResidencySet* residencySet)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(useResidencySet_), residencySet);
}

_MTL_INLINE void MTL::CommandBuffer::useResidencySets(const MTL::ResidencySet* const residencySets[], NS::UInteger count)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(useResidencySets_count_), residencySets, count);
}

_MTL_INLINE void MTL::CommandBuffer::waitUntilCompleted()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(waitUntilCompleted));
}

_MTL_INLINE void MTL::CommandBuffer::waitUntilScheduled()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(waitUntilScheduled));
}
