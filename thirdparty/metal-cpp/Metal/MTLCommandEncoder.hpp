//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLCommandEncoder.hpp
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
class Device;

_MTL_OPTIONS(NS::UInteger, ResourceUsage) {
    ResourceUsageRead = 1,
    ResourceUsageWrite = 1 << 1,
    ResourceUsageSample = 1 << 2,
};

_MTL_OPTIONS(NS::UInteger, BarrierScope) {
    BarrierScopeBuffers = 1,
    BarrierScopeTextures = 1 << 1,
    BarrierScopeRenderTargets = 1 << 2,
};

_MTL_OPTIONS(NS::UInteger, Stages) {
    StageVertex = 1,
    StageFragment = 1 << 1,
    StageTile = 1 << 2,
    StageObject = 1 << 3,
    StageMesh = 1 << 4,
    StageResourceState = 1 << 26,
    StageDispatch = 1 << 27,
    StageBlit = 1 << 28,
    StageAccelerationStructure = 1 << 29,
    StageMachineLearning = 1 << 30,
    StageAll = 9223372036854775807,
};

class CommandEncoder : public NS::Referencing<CommandEncoder>
{
public:
    void        barrierAfterQueueStages(MTL::Stages afterQueueStages, MTL::Stages beforeStages);

    Device*     device() const;

    void        endEncoding();

    void        insertDebugSignpost(const NS::String* string);

    NS::String* label() const;

    void        popDebugGroup();

    void        pushDebugGroup(const NS::String* string);

    void        setLabel(const NS::String* label);
};

}
_MTL_INLINE void MTL::CommandEncoder::barrierAfterQueueStages(MTL::Stages afterQueueStages, MTL::Stages beforeStages)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(barrierAfterQueueStages_beforeStages_), afterQueueStages, beforeStages);
}

_MTL_INLINE MTL::Device* MTL::CommandEncoder::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE void MTL::CommandEncoder::endEncoding()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(endEncoding));
}

_MTL_INLINE void MTL::CommandEncoder::insertDebugSignpost(const NS::String* string)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(insertDebugSignpost_), string);
}

_MTL_INLINE NS::String* MTL::CommandEncoder::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE void MTL::CommandEncoder::popDebugGroup()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(popDebugGroup));
}

_MTL_INLINE void MTL::CommandEncoder::pushDebugGroup(const NS::String* string)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(pushDebugGroup_), string);
}

_MTL_INLINE void MTL::CommandEncoder::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}
