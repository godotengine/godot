//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTL4CommandBuffer.hpp
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
#include "MTL4RenderCommandEncoder.hpp"
#include "MTLAccelerationStructureTypes.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"

namespace MTL4
{
class CommandAllocator;
class CommandBufferOptions;
class ComputeCommandEncoder;
class CounterHeap;
class MachineLearningCommandEncoder;
class RenderCommandEncoder;
class RenderPassDescriptor;
}

namespace MTL
{
class Device;
class Fence;
class LogState;
class ResidencySet;
}

namespace MTL4
{
class CommandBufferOptions : public NS::Copying<CommandBufferOptions>
{
public:
    static CommandBufferOptions* alloc();

    CommandBufferOptions*        init();

    MTL::LogState*               logState() const;
    void                         setLogState(const MTL::LogState* logState);
};
class CommandBuffer : public NS::Referencing<CommandBuffer>
{
public:
    void                           beginCommandBuffer(const MTL4::CommandAllocator* allocator);
    void                           beginCommandBuffer(const MTL4::CommandAllocator* allocator, const MTL4::CommandBufferOptions* options);

    ComputeCommandEncoder*         computeCommandEncoder();

    MTL::Device*                   device() const;

    void                           endCommandBuffer();

    NS::String*                    label() const;

    MachineLearningCommandEncoder* machineLearningCommandEncoder();

    void                           popDebugGroup();

    void                           pushDebugGroup(const NS::String* string);

    RenderCommandEncoder*          renderCommandEncoder(const MTL4::RenderPassDescriptor* descriptor);
    RenderCommandEncoder*          renderCommandEncoder(const MTL4::RenderPassDescriptor* descriptor, MTL4::RenderEncoderOptions options);

    void                           resolveCounterHeap(const MTL4::CounterHeap* counterHeap, NS::Range range, const MTL4::BufferRange bufferRange, const MTL::Fence* fenceToWait, const MTL::Fence* fenceToUpdate);

    void                           setLabel(const NS::String* label);

    void                           useResidencySet(const MTL::ResidencySet* residencySet);
    void                           useResidencySets(const MTL::ResidencySet* const residencySets[], NS::UInteger count);

    void                           writeTimestampIntoHeap(const MTL4::CounterHeap* counterHeap, NS::UInteger index);
};

}
_MTL_INLINE MTL4::CommandBufferOptions* MTL4::CommandBufferOptions::alloc()
{
    return NS::Object::alloc<MTL4::CommandBufferOptions>(_MTL_PRIVATE_CLS(MTL4CommandBufferOptions));
}

_MTL_INLINE MTL4::CommandBufferOptions* MTL4::CommandBufferOptions::init()
{
    return NS::Object::init<MTL4::CommandBufferOptions>();
}

_MTL_INLINE MTL::LogState* MTL4::CommandBufferOptions::logState() const
{
    return Object::sendMessage<MTL::LogState*>(this, _MTL_PRIVATE_SEL(logState));
}

_MTL_INLINE void MTL4::CommandBufferOptions::setLogState(const MTL::LogState* logState)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLogState_), logState);
}

_MTL_INLINE void MTL4::CommandBuffer::beginCommandBuffer(const MTL4::CommandAllocator* allocator)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(beginCommandBufferWithAllocator_), allocator);
}

_MTL_INLINE void MTL4::CommandBuffer::beginCommandBuffer(const MTL4::CommandAllocator* allocator, const MTL4::CommandBufferOptions* options)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(beginCommandBufferWithAllocator_options_), allocator, options);
}

_MTL_INLINE MTL4::ComputeCommandEncoder* MTL4::CommandBuffer::computeCommandEncoder()
{
    return Object::sendMessage<MTL4::ComputeCommandEncoder*>(this, _MTL_PRIVATE_SEL(computeCommandEncoder));
}

_MTL_INLINE MTL::Device* MTL4::CommandBuffer::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE void MTL4::CommandBuffer::endCommandBuffer()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(endCommandBuffer));
}

_MTL_INLINE NS::String* MTL4::CommandBuffer::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE MTL4::MachineLearningCommandEncoder* MTL4::CommandBuffer::machineLearningCommandEncoder()
{
    return Object::sendMessage<MTL4::MachineLearningCommandEncoder*>(this, _MTL_PRIVATE_SEL(machineLearningCommandEncoder));
}

_MTL_INLINE void MTL4::CommandBuffer::popDebugGroup()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(popDebugGroup));
}

_MTL_INLINE void MTL4::CommandBuffer::pushDebugGroup(const NS::String* string)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(pushDebugGroup_), string);
}

_MTL_INLINE MTL4::RenderCommandEncoder* MTL4::CommandBuffer::renderCommandEncoder(const MTL4::RenderPassDescriptor* descriptor)
{
    return Object::sendMessage<MTL4::RenderCommandEncoder*>(this, _MTL_PRIVATE_SEL(renderCommandEncoderWithDescriptor_), descriptor);
}

_MTL_INLINE MTL4::RenderCommandEncoder* MTL4::CommandBuffer::renderCommandEncoder(const MTL4::RenderPassDescriptor* descriptor, MTL4::RenderEncoderOptions options)
{
    return Object::sendMessage<MTL4::RenderCommandEncoder*>(this, _MTL_PRIVATE_SEL(renderCommandEncoderWithDescriptor_options_), descriptor, options);
}

_MTL_INLINE void MTL4::CommandBuffer::resolveCounterHeap(const MTL4::CounterHeap* counterHeap, NS::Range range, const MTL4::BufferRange bufferRange, const MTL::Fence* fenceToWait, const MTL::Fence* fenceToUpdate)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(resolveCounterHeap_withRange_intoBuffer_waitFence_updateFence_), counterHeap, range, bufferRange, fenceToWait, fenceToUpdate);
}

_MTL_INLINE void MTL4::CommandBuffer::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE void MTL4::CommandBuffer::useResidencySet(const MTL::ResidencySet* residencySet)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(useResidencySet_), residencySet);
}

_MTL_INLINE void MTL4::CommandBuffer::useResidencySets(const MTL::ResidencySet* const residencySets[], NS::UInteger count)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(useResidencySets_count_), residencySets, count);
}

_MTL_INLINE void MTL4::CommandBuffer::writeTimestampIntoHeap(const MTL4::CounterHeap* counterHeap, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(writeTimestampIntoHeap_atIndex_), counterHeap, index);
}
