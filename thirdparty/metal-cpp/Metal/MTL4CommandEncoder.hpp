//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTL4CommandEncoder.hpp
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
#include "MTLCommandEncoder.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"

namespace MTL4
{
class CommandBuffer;
}

namespace MTL
{
class Fence;
}

namespace MTL4
{
_MTL_OPTIONS(NS::UInteger, VisibilityOptions) {
    VisibilityOptionNone = 0,
    VisibilityOptionDevice = 1,
    VisibilityOptionResourceAlias = 1 << 1,
};

class CommandEncoder : public NS::Referencing<CommandEncoder>
{
public:
    void           barrierAfterEncoderStages(MTL::Stages afterEncoderStages, MTL::Stages beforeEncoderStages, MTL4::VisibilityOptions visibilityOptions);

    void           barrierAfterQueueStages(MTL::Stages afterQueueStages, MTL::Stages beforeStages, MTL4::VisibilityOptions visibilityOptions);

    void           barrierAfterStages(MTL::Stages afterStages, MTL::Stages beforeQueueStages, MTL4::VisibilityOptions visibilityOptions);

    CommandBuffer* commandBuffer() const;

    void           endEncoding();

    void           insertDebugSignpost(const NS::String* string);

    NS::String*    label() const;

    void           popDebugGroup();

    void           pushDebugGroup(const NS::String* string);

    void           setLabel(const NS::String* label);

    void           updateFence(const MTL::Fence* fence, MTL::Stages afterEncoderStages);

    void           waitForFence(const MTL::Fence* fence, MTL::Stages beforeEncoderStages);
};

}
_MTL_INLINE void MTL4::CommandEncoder::barrierAfterEncoderStages(MTL::Stages afterEncoderStages, MTL::Stages beforeEncoderStages, MTL4::VisibilityOptions visibilityOptions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions_), afterEncoderStages, beforeEncoderStages, visibilityOptions);
}

_MTL_INLINE void MTL4::CommandEncoder::barrierAfterQueueStages(MTL::Stages afterQueueStages, MTL::Stages beforeStages, MTL4::VisibilityOptions visibilityOptions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(barrierAfterQueueStages_beforeStages_visibilityOptions_), afterQueueStages, beforeStages, visibilityOptions);
}

_MTL_INLINE void MTL4::CommandEncoder::barrierAfterStages(MTL::Stages afterStages, MTL::Stages beforeQueueStages, MTL4::VisibilityOptions visibilityOptions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(barrierAfterStages_beforeQueueStages_visibilityOptions_), afterStages, beforeQueueStages, visibilityOptions);
}

_MTL_INLINE MTL4::CommandBuffer* MTL4::CommandEncoder::commandBuffer() const
{
    return Object::sendMessage<MTL4::CommandBuffer*>(this, _MTL_PRIVATE_SEL(commandBuffer));
}

_MTL_INLINE void MTL4::CommandEncoder::endEncoding()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(endEncoding));
}

_MTL_INLINE void MTL4::CommandEncoder::insertDebugSignpost(const NS::String* string)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(insertDebugSignpost_), string);
}

_MTL_INLINE NS::String* MTL4::CommandEncoder::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE void MTL4::CommandEncoder::popDebugGroup()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(popDebugGroup));
}

_MTL_INLINE void MTL4::CommandEncoder::pushDebugGroup(const NS::String* string)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(pushDebugGroup_), string);
}

_MTL_INLINE void MTL4::CommandEncoder::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE void MTL4::CommandEncoder::updateFence(const MTL::Fence* fence, MTL::Stages afterEncoderStages)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(updateFence_afterEncoderStages_), fence, afterEncoderStages);
}

_MTL_INLINE void MTL4::CommandEncoder::waitForFence(const MTL::Fence* fence, MTL::Stages beforeEncoderStages)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(waitForFence_beforeEncoderStages_), fence, beforeEncoderStages);
}
