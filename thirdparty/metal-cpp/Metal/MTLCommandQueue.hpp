//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLCommandQueue.hpp
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
class CommandBuffer;
class CommandBufferDescriptor;
class CommandQueueDescriptor;
class Device;
class LogState;
class ResidencySet;

class CommandQueue : public NS::Referencing<CommandQueue>
{
public:
    void           addResidencySet(const MTL::ResidencySet* residencySet);
    void           addResidencySets(const MTL::ResidencySet* const residencySets[], NS::UInteger count);

    CommandBuffer* commandBuffer();
    CommandBuffer* commandBuffer(const MTL::CommandBufferDescriptor* descriptor);
    CommandBuffer* commandBufferWithUnretainedReferences();

    Device*        device() const;

    void           insertDebugCaptureBoundary();

    NS::String*    label() const;

    void           removeResidencySet(const MTL::ResidencySet* residencySet);
    void           removeResidencySets(const MTL::ResidencySet* const residencySets[], NS::UInteger count);

    void           setLabel(const NS::String* label);
};
class CommandQueueDescriptor : public NS::Copying<CommandQueueDescriptor>
{
public:
    static CommandQueueDescriptor* alloc();

    CommandQueueDescriptor*        init();

    LogState*                      logState() const;

    NS::UInteger                   maxCommandBufferCount() const;

    void                           setLogState(const MTL::LogState* logState);

    void                           setMaxCommandBufferCount(NS::UInteger maxCommandBufferCount);
};

}
_MTL_INLINE void MTL::CommandQueue::addResidencySet(const MTL::ResidencySet* residencySet)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(addResidencySet_), residencySet);
}

_MTL_INLINE void MTL::CommandQueue::addResidencySets(const MTL::ResidencySet* const residencySets[], NS::UInteger count)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(addResidencySets_count_), residencySets, count);
}

_MTL_INLINE MTL::CommandBuffer* MTL::CommandQueue::commandBuffer()
{
    return Object::sendMessage<MTL::CommandBuffer*>(this, _MTL_PRIVATE_SEL(commandBuffer));
}

_MTL_INLINE MTL::CommandBuffer* MTL::CommandQueue::commandBuffer(const MTL::CommandBufferDescriptor* descriptor)
{
    return Object::sendMessage<MTL::CommandBuffer*>(this, _MTL_PRIVATE_SEL(commandBufferWithDescriptor_), descriptor);
}

_MTL_INLINE MTL::CommandBuffer* MTL::CommandQueue::commandBufferWithUnretainedReferences()
{
    return Object::sendMessage<MTL::CommandBuffer*>(this, _MTL_PRIVATE_SEL(commandBufferWithUnretainedReferences));
}

_MTL_INLINE MTL::Device* MTL::CommandQueue::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE void MTL::CommandQueue::insertDebugCaptureBoundary()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(insertDebugCaptureBoundary));
}

_MTL_INLINE NS::String* MTL::CommandQueue::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE void MTL::CommandQueue::removeResidencySet(const MTL::ResidencySet* residencySet)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(removeResidencySet_), residencySet);
}

_MTL_INLINE void MTL::CommandQueue::removeResidencySets(const MTL::ResidencySet* const residencySets[], NS::UInteger count)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(removeResidencySets_count_), residencySets, count);
}

_MTL_INLINE void MTL::CommandQueue::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE MTL::CommandQueueDescriptor* MTL::CommandQueueDescriptor::alloc()
{
    return NS::Object::alloc<MTL::CommandQueueDescriptor>(_MTL_PRIVATE_CLS(MTLCommandQueueDescriptor));
}

_MTL_INLINE MTL::CommandQueueDescriptor* MTL::CommandQueueDescriptor::init()
{
    return NS::Object::init<MTL::CommandQueueDescriptor>();
}

_MTL_INLINE MTL::LogState* MTL::CommandQueueDescriptor::logState() const
{
    return Object::sendMessage<MTL::LogState*>(this, _MTL_PRIVATE_SEL(logState));
}

_MTL_INLINE NS::UInteger MTL::CommandQueueDescriptor::maxCommandBufferCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxCommandBufferCount));
}

_MTL_INLINE void MTL::CommandQueueDescriptor::setLogState(const MTL::LogState* logState)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLogState_), logState);
}

_MTL_INLINE void MTL::CommandQueueDescriptor::setMaxCommandBufferCount(NS::UInteger maxCommandBufferCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxCommandBufferCount_), maxCommandBufferCount);
}
