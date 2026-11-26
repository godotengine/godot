//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLIOCommandBuffer.hpp
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
#include "MTLTypes.hpp"
#include <cstdint>

namespace MTL
{
class Buffer;
class IOCommandBuffer;
class IOFileHandle;
class SharedEvent;
class Texture;
_MTL_ENUM(NS::Integer, IOStatus) {
    IOStatusPending = 0,
    IOStatusCancelled = 1,
    IOStatusError = 2,
    IOStatusComplete = 3,
};

using IOCommandBufferHandler = void (^)(MTL::IOCommandBuffer*);
using IOCommandBufferHandlerFunction = std::function<void(MTL::IOCommandBuffer*)>;

class IOCommandBuffer : public NS::Referencing<IOCommandBuffer>
{
public:
    void        addBarrier();

    void        addCompletedHandler(const MTL::IOCommandBufferHandler block);
    void        addCompletedHandler(const MTL::IOCommandBufferHandlerFunction& function);

    void        commit();

    void        copyStatusToBuffer(const MTL::Buffer* buffer, NS::UInteger offset);

    void        enqueue();

    NS::Error*  error() const;

    NS::String* label() const;

    void        loadBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger size, const MTL::IOFileHandle* sourceHandle, NS::UInteger sourceHandleOffset);

    void        loadBytes(const void* pointer, NS::UInteger size, const MTL::IOFileHandle* sourceHandle, NS::UInteger sourceHandleOffset);

    void        loadTexture(const MTL::Texture* texture, NS::UInteger slice, NS::UInteger level, MTL::Size size, NS::UInteger sourceBytesPerRow, NS::UInteger sourceBytesPerImage, MTL::Origin destinationOrigin, const MTL::IOFileHandle* sourceHandle, NS::UInteger sourceHandleOffset);

    void        popDebugGroup();

    void        pushDebugGroup(const NS::String* string);

    void        setLabel(const NS::String* label);

    void        signalEvent(const MTL::SharedEvent* event, uint64_t value);

    IOStatus    status() const;

    void        tryCancel();

    void        wait(const MTL::SharedEvent* event, uint64_t value);
    void        waitUntilCompleted();
};

}
_MTL_INLINE void MTL::IOCommandBuffer::addBarrier()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(addBarrier));
}

_MTL_INLINE void MTL::IOCommandBuffer::addCompletedHandler(const MTL::IOCommandBufferHandler block)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(addCompletedHandler_), block);
}

_MTL_INLINE void MTL::IOCommandBuffer::addCompletedHandler(const MTL::IOCommandBufferHandlerFunction& function)
{
    __block MTL::IOCommandBufferHandlerFunction blockFunction = function;
    addCompletedHandler(^(MTL::IOCommandBuffer* pCommandBuffer) { blockFunction(pCommandBuffer); });
}

_MTL_INLINE void MTL::IOCommandBuffer::commit()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(commit));
}

_MTL_INLINE void MTL::IOCommandBuffer::copyStatusToBuffer(const MTL::Buffer* buffer, NS::UInteger offset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyStatusToBuffer_offset_), buffer, offset);
}

_MTL_INLINE void MTL::IOCommandBuffer::enqueue()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(enqueue));
}

_MTL_INLINE NS::Error* MTL::IOCommandBuffer::error() const
{
    return Object::sendMessage<NS::Error*>(this, _MTL_PRIVATE_SEL(error));
}

_MTL_INLINE NS::String* MTL::IOCommandBuffer::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE void MTL::IOCommandBuffer::loadBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger size, const MTL::IOFileHandle* sourceHandle, NS::UInteger sourceHandleOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(loadBuffer_offset_size_sourceHandle_sourceHandleOffset_), buffer, offset, size, sourceHandle, sourceHandleOffset);
}

_MTL_INLINE void MTL::IOCommandBuffer::loadBytes(const void* pointer, NS::UInteger size, const MTL::IOFileHandle* sourceHandle, NS::UInteger sourceHandleOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(loadBytes_size_sourceHandle_sourceHandleOffset_), pointer, size, sourceHandle, sourceHandleOffset);
}

_MTL_INLINE void MTL::IOCommandBuffer::loadTexture(const MTL::Texture* texture, NS::UInteger slice, NS::UInteger level, MTL::Size size, NS::UInteger sourceBytesPerRow, NS::UInteger sourceBytesPerImage, MTL::Origin destinationOrigin, const MTL::IOFileHandle* sourceHandle, NS::UInteger sourceHandleOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(loadTexture_slice_level_size_sourceBytesPerRow_sourceBytesPerImage_destinationOrigin_sourceHandle_sourceHandleOffset_), texture, slice, level, size, sourceBytesPerRow, sourceBytesPerImage, destinationOrigin, sourceHandle, sourceHandleOffset);
}

_MTL_INLINE void MTL::IOCommandBuffer::popDebugGroup()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(popDebugGroup));
}

_MTL_INLINE void MTL::IOCommandBuffer::pushDebugGroup(const NS::String* string)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(pushDebugGroup_), string);
}

_MTL_INLINE void MTL::IOCommandBuffer::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE void MTL::IOCommandBuffer::signalEvent(const MTL::SharedEvent* event, uint64_t value)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(signalEvent_value_), event, value);
}

_MTL_INLINE MTL::IOStatus MTL::IOCommandBuffer::status() const
{
    return Object::sendMessage<MTL::IOStatus>(this, _MTL_PRIVATE_SEL(status));
}

_MTL_INLINE void MTL::IOCommandBuffer::tryCancel()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(tryCancel));
}

_MTL_INLINE void MTL::IOCommandBuffer::wait(const MTL::SharedEvent* event, uint64_t value)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(waitForEvent_value_), event, value);
}

_MTL_INLINE void MTL::IOCommandBuffer::waitUntilCompleted()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(waitUntilCompleted));
}
