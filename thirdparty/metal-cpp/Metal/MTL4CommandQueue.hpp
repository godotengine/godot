//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTL4CommandQueue.hpp
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
#include "MTL4CommitFeedback.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include "MTLResourceStateCommandEncoder.hpp"
#include "MTLTypes.hpp"
#include <cstdint>
#include <dispatch/dispatch.h>

namespace MTL
{
class Buffer;
class Device;
class Drawable;
class Event;
class Heap;
class ResidencySet;
class Texture;
}

namespace MTL4
{
class CommandBuffer;
class CommandQueueDescriptor;
class CommitOptions;
struct CopySparseBufferMappingOperation;
struct CopySparseTextureMappingOperation;
struct UpdateSparseBufferMappingOperation;
struct UpdateSparseTextureMappingOperation;
_MTL_ENUM(NS::Integer, CommandQueueError) {
    CommandQueueErrorNone = 0,
    CommandQueueErrorTimeout = 1,
    CommandQueueErrorNotPermitted = 2,
    CommandQueueErrorOutOfMemory = 3,
    CommandQueueErrorDeviceRemoved = 4,
    CommandQueueErrorAccessRevoked = 5,
    CommandQueueErrorInternal = 6,
};

struct UpdateSparseTextureMappingOperation
{
    MTL::SparseTextureMappingMode mode;
    MTL::Region                   textureRegion;
    NS::UInteger                  textureLevel;
    NS::UInteger                  textureSlice;
    NS::UInteger                  heapOffset;
} _MTL_PACKED;

struct CopySparseTextureMappingOperation
{
    MTL::Region  sourceRegion;
    NS::UInteger sourceLevel;
    NS::UInteger sourceSlice;
    MTL::Origin  destinationOrigin;
    NS::UInteger destinationLevel;
    NS::UInteger destinationSlice;
} _MTL_PACKED;

struct UpdateSparseBufferMappingOperation
{
    MTL::SparseTextureMappingMode mode;
    NS::Range                     bufferRange;
    NS::UInteger                  heapOffset;
} _MTL_PACKED;

struct CopySparseBufferMappingOperation
{
    NS::Range    sourceRange;
    NS::UInteger destinationOffset;
} _MTL_PACKED;

class CommitOptions : public NS::Referencing<CommitOptions>
{
public:
    void                  addFeedbackHandler(const MTL4::CommitFeedbackHandler block);
    void                  addFeedbackHandler(const MTL4::CommitFeedbackHandlerFunction& function);

    static CommitOptions* alloc();

    CommitOptions*        init();
};
class CommandQueueDescriptor : public NS::Copying<CommandQueueDescriptor>
{
public:
    static CommandQueueDescriptor* alloc();

    dispatch_queue_t               feedbackQueue() const;

    CommandQueueDescriptor*        init();

    NS::String*                    label() const;

    void                           setFeedbackQueue(const dispatch_queue_t feedbackQueue);

    void                           setLabel(const NS::String* label);
};
class CommandQueue : public NS::Referencing<CommandQueue>
{
public:
    void         addResidencySet(const MTL::ResidencySet* residencySet);
    void         addResidencySets(const MTL::ResidencySet* const residencySets[], NS::UInteger count);

    void         commit(const MTL4::CommandBuffer* const commandBuffers[], NS::UInteger count);
    void         commit(const MTL4::CommandBuffer* const commandBuffers[], NS::UInteger count, const MTL4::CommitOptions* options);

    void         copyBufferMappingsFromBuffer(const MTL::Buffer* sourceBuffer, const MTL::Buffer* destinationBuffer, const MTL4::CopySparseBufferMappingOperation* operations, NS::UInteger count);

    void         copyTextureMappingsFromTexture(const MTL::Texture* sourceTexture, const MTL::Texture* destinationTexture, const MTL4::CopySparseTextureMappingOperation* operations, NS::UInteger count);

    MTL::Device* device() const;

    NS::String*  label() const;

    void         removeResidencySet(const MTL::ResidencySet* residencySet);
    void         removeResidencySets(const MTL::ResidencySet* const residencySets[], NS::UInteger count);

    void         signalDrawable(const MTL::Drawable* drawable);

    void         signalEvent(const MTL::Event* event, uint64_t value);

    void         updateBufferMappings(const MTL::Buffer* buffer, const MTL::Heap* heap, const MTL4::UpdateSparseBufferMappingOperation* operations, NS::UInteger count);

    void         updateTextureMappings(const MTL::Texture* texture, const MTL::Heap* heap, const MTL4::UpdateSparseTextureMappingOperation* operations, NS::UInteger count);

    void         wait(const MTL::Event* event, uint64_t value);
    void         wait(const MTL::Drawable* drawable);
};

}

_MTL_INLINE void MTL4::CommitOptions::addFeedbackHandler(const MTL4::CommitFeedbackHandler block)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(addFeedbackHandler_), block);
}

_MTL_INLINE void MTL4::CommitOptions::addFeedbackHandler(const MTL4::CommitFeedbackHandlerFunction& function)
{
    __block MTL4::CommitFeedbackHandlerFunction blockFunction = function;
    addFeedbackHandler(^(MTL4::CommitFeedback* pFeedback) { blockFunction(pFeedback); });
}

_MTL_INLINE MTL4::CommitOptions* MTL4::CommitOptions::alloc()
{
    return NS::Object::alloc<MTL4::CommitOptions>(_MTL_PRIVATE_CLS(MTL4CommitOptions));
}

_MTL_INLINE MTL4::CommitOptions* MTL4::CommitOptions::init()
{
    return NS::Object::init<MTL4::CommitOptions>();
}

_MTL_INLINE MTL4::CommandQueueDescriptor* MTL4::CommandQueueDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::CommandQueueDescriptor>(_MTL_PRIVATE_CLS(MTL4CommandQueueDescriptor));
}

_MTL_INLINE dispatch_queue_t MTL4::CommandQueueDescriptor::feedbackQueue() const
{
    return Object::sendMessage<dispatch_queue_t>(this, _MTL_PRIVATE_SEL(feedbackQueue));
}

_MTL_INLINE MTL4::CommandQueueDescriptor* MTL4::CommandQueueDescriptor::init()
{
    return NS::Object::init<MTL4::CommandQueueDescriptor>();
}

_MTL_INLINE NS::String* MTL4::CommandQueueDescriptor::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE void MTL4::CommandQueueDescriptor::setFeedbackQueue(const dispatch_queue_t feedbackQueue)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFeedbackQueue_), feedbackQueue);
}

_MTL_INLINE void MTL4::CommandQueueDescriptor::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE void MTL4::CommandQueue::addResidencySet(const MTL::ResidencySet* residencySet)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(addResidencySet_), residencySet);
}

_MTL_INLINE void MTL4::CommandQueue::addResidencySets(const MTL::ResidencySet* const residencySets[], NS::UInteger count)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(addResidencySets_count_), residencySets, count);
}

_MTL_INLINE void MTL4::CommandQueue::commit(const MTL4::CommandBuffer* const commandBuffers[], NS::UInteger count)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(commit_count_), commandBuffers, count);
}

_MTL_INLINE void MTL4::CommandQueue::commit(const MTL4::CommandBuffer* const commandBuffers[], NS::UInteger count, const MTL4::CommitOptions* options)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(commit_count_options_), commandBuffers, count, options);
}

_MTL_INLINE void MTL4::CommandQueue::copyBufferMappingsFromBuffer(const MTL::Buffer* sourceBuffer, const MTL::Buffer* destinationBuffer, const MTL4::CopySparseBufferMappingOperation* operations, NS::UInteger count)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyBufferMappingsFromBuffer_toBuffer_operations_count_), sourceBuffer, destinationBuffer, operations, count);
}

_MTL_INLINE void MTL4::CommandQueue::copyTextureMappingsFromTexture(const MTL::Texture* sourceTexture, const MTL::Texture* destinationTexture, const MTL4::CopySparseTextureMappingOperation* operations, NS::UInteger count)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyTextureMappingsFromTexture_toTexture_operations_count_), sourceTexture, destinationTexture, operations, count);
}

_MTL_INLINE MTL::Device* MTL4::CommandQueue::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE NS::String* MTL4::CommandQueue::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE void MTL4::CommandQueue::removeResidencySet(const MTL::ResidencySet* residencySet)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(removeResidencySet_), residencySet);
}

_MTL_INLINE void MTL4::CommandQueue::removeResidencySets(const MTL::ResidencySet* const residencySets[], NS::UInteger count)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(removeResidencySets_count_), residencySets, count);
}

_MTL_INLINE void MTL4::CommandQueue::signalDrawable(const MTL::Drawable* drawable)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(signalDrawable_), drawable);
}

_MTL_INLINE void MTL4::CommandQueue::signalEvent(const MTL::Event* event, uint64_t value)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(signalEvent_value_), event, value);
}

_MTL_INLINE void MTL4::CommandQueue::updateBufferMappings(const MTL::Buffer* buffer, const MTL::Heap* heap, const MTL4::UpdateSparseBufferMappingOperation* operations, NS::UInteger count)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(updateBufferMappings_heap_operations_count_), buffer, heap, operations, count);
}

_MTL_INLINE void MTL4::CommandQueue::updateTextureMappings(const MTL::Texture* texture, const MTL::Heap* heap, const MTL4::UpdateSparseTextureMappingOperation* operations, NS::UInteger count)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(updateTextureMappings_heap_operations_count_), texture, heap, operations, count);
}

_MTL_INLINE void MTL4::CommandQueue::wait(const MTL::Event* event, uint64_t value)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(waitForEvent_value_), event, value);
}

_MTL_INLINE void MTL4::CommandQueue::wait(const MTL::Drawable* drawable)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(waitForDrawable_), drawable);
}
