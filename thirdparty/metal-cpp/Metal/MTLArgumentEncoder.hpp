//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLArgumentEncoder.hpp
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
#include "MTLDepthStencil.hpp"

namespace MTL
{
class AccelerationStructure;
class ArgumentEncoder;
class Buffer;
class ComputePipelineState;
class Device;
class IndirectCommandBuffer;
class IntersectionFunctionTable;
class RenderPipelineState;
class SamplerState;
class Texture;
class VisibleFunctionTable;

static const NS::UInteger AttributeStrideStatic = NS::UIntegerMax;

class ArgumentEncoder : public NS::Referencing<ArgumentEncoder>
{
public:
    NS::UInteger     alignment() const;

    void*            constantData(NS::UInteger index);

    Device*          device() const;

    NS::UInteger     encodedLength() const;

    NS::String*      label() const;

    ArgumentEncoder* newArgumentEncoder(NS::UInteger index);

    void             setAccelerationStructure(const MTL::AccelerationStructure* accelerationStructure, NS::UInteger index);

    void             setArgumentBuffer(const MTL::Buffer* argumentBuffer, NS::UInteger offset);
    void             setArgumentBuffer(const MTL::Buffer* argumentBuffer, NS::UInteger startOffset, NS::UInteger arrayElement);

    void             setBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);
    void             setBuffers(const MTL::Buffer* const buffers[], const NS::UInteger offsets[], NS::Range range);

    void             setComputePipelineState(const MTL::ComputePipelineState* pipeline, NS::UInteger index);
    void             setComputePipelineStates(const MTL::ComputePipelineState* const pipelines[], NS::Range range);

    void             setDepthStencilState(const MTL::DepthStencilState* depthStencilState, NS::UInteger index);
    void             setDepthStencilStates(const MTL::DepthStencilState* const depthStencilStates[], NS::Range range);

    void             setIndirectCommandBuffer(const MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::UInteger index);
    void             setIndirectCommandBuffers(const MTL::IndirectCommandBuffer* const buffers[], NS::Range range);

    void             setIntersectionFunctionTable(const MTL::IntersectionFunctionTable* intersectionFunctionTable, NS::UInteger index);
    void             setIntersectionFunctionTables(const MTL::IntersectionFunctionTable* const intersectionFunctionTables[], NS::Range range);

    void             setLabel(const NS::String* label);

    void             setRenderPipelineState(const MTL::RenderPipelineState* pipeline, NS::UInteger index);
    void             setRenderPipelineStates(const MTL::RenderPipelineState* const pipelines[], NS::Range range);

    void             setSamplerState(const MTL::SamplerState* sampler, NS::UInteger index);
    void             setSamplerStates(const MTL::SamplerState* const samplers[], NS::Range range);

    void             setTexture(const MTL::Texture* texture, NS::UInteger index);
    void             setTextures(const MTL::Texture* const textures[], NS::Range range);

    void             setVisibleFunctionTable(const MTL::VisibleFunctionTable* visibleFunctionTable, NS::UInteger index);
    void             setVisibleFunctionTables(const MTL::VisibleFunctionTable* const visibleFunctionTables[], NS::Range range);
};

}

_MTL_INLINE NS::UInteger MTL::ArgumentEncoder::alignment() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(alignment));
}

_MTL_INLINE void* MTL::ArgumentEncoder::constantData(NS::UInteger index)
{
    return Object::sendMessage<void*>(this, _MTL_PRIVATE_SEL(constantDataAtIndex_), index);
}

_MTL_INLINE MTL::Device* MTL::ArgumentEncoder::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE NS::UInteger MTL::ArgumentEncoder::encodedLength() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(encodedLength));
}

_MTL_INLINE NS::String* MTL::ArgumentEncoder::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE MTL::ArgumentEncoder* MTL::ArgumentEncoder::newArgumentEncoder(NS::UInteger index)
{
    return Object::sendMessage<MTL::ArgumentEncoder*>(this, _MTL_PRIVATE_SEL(newArgumentEncoderForBufferAtIndex_), index);
}

_MTL_INLINE void MTL::ArgumentEncoder::setAccelerationStructure(const MTL::AccelerationStructure* accelerationStructure, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setAccelerationStructure_atIndex_), accelerationStructure, index);
}

_MTL_INLINE void MTL::ArgumentEncoder::setArgumentBuffer(const MTL::Buffer* argumentBuffer, NS::UInteger offset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setArgumentBuffer_offset_), argumentBuffer, offset);
}

_MTL_INLINE void MTL::ArgumentEncoder::setArgumentBuffer(const MTL::Buffer* argumentBuffer, NS::UInteger startOffset, NS::UInteger arrayElement)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setArgumentBuffer_startOffset_arrayElement_), argumentBuffer, startOffset, arrayElement);
}

_MTL_INLINE void MTL::ArgumentEncoder::setBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBuffer_offset_atIndex_), buffer, offset, index);
}

_MTL_INLINE void MTL::ArgumentEncoder::setBuffers(const MTL::Buffer* const buffers[], const NS::UInteger offsets[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBuffers_offsets_withRange_), buffers, offsets, range);
}

_MTL_INLINE void MTL::ArgumentEncoder::setComputePipelineState(const MTL::ComputePipelineState* pipeline, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setComputePipelineState_atIndex_), pipeline, index);
}

_MTL_INLINE void MTL::ArgumentEncoder::setComputePipelineStates(const MTL::ComputePipelineState* const pipelines[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setComputePipelineStates_withRange_), pipelines, range);
}

_MTL_INLINE void MTL::ArgumentEncoder::setDepthStencilState(const MTL::DepthStencilState* depthStencilState, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthStencilState_atIndex_), depthStencilState, index);
}

_MTL_INLINE void MTL::ArgumentEncoder::setDepthStencilStates(const MTL::DepthStencilState* const depthStencilStates[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthStencilStates_withRange_), depthStencilStates, range);
}

_MTL_INLINE void MTL::ArgumentEncoder::setIndirectCommandBuffer(const MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndirectCommandBuffer_atIndex_), indirectCommandBuffer, index);
}

_MTL_INLINE void MTL::ArgumentEncoder::setIndirectCommandBuffers(const MTL::IndirectCommandBuffer* const buffers[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndirectCommandBuffers_withRange_), buffers, range);
}

_MTL_INLINE void MTL::ArgumentEncoder::setIntersectionFunctionTable(const MTL::IntersectionFunctionTable* intersectionFunctionTable, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIntersectionFunctionTable_atIndex_), intersectionFunctionTable, index);
}

_MTL_INLINE void MTL::ArgumentEncoder::setIntersectionFunctionTables(const MTL::IntersectionFunctionTable* const intersectionFunctionTables[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIntersectionFunctionTables_withRange_), intersectionFunctionTables, range);
}

_MTL_INLINE void MTL::ArgumentEncoder::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE void MTL::ArgumentEncoder::setRenderPipelineState(const MTL::RenderPipelineState* pipeline, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRenderPipelineState_atIndex_), pipeline, index);
}

_MTL_INLINE void MTL::ArgumentEncoder::setRenderPipelineStates(const MTL::RenderPipelineState* const pipelines[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRenderPipelineStates_withRange_), pipelines, range);
}

_MTL_INLINE void MTL::ArgumentEncoder::setSamplerState(const MTL::SamplerState* sampler, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSamplerState_atIndex_), sampler, index);
}

_MTL_INLINE void MTL::ArgumentEncoder::setSamplerStates(const MTL::SamplerState* const samplers[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSamplerStates_withRange_), samplers, range);
}

_MTL_INLINE void MTL::ArgumentEncoder::setTexture(const MTL::Texture* texture, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTexture_atIndex_), texture, index);
}

_MTL_INLINE void MTL::ArgumentEncoder::setTextures(const MTL::Texture* const textures[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTextures_withRange_), textures, range);
}

_MTL_INLINE void MTL::ArgumentEncoder::setVisibleFunctionTable(const MTL::VisibleFunctionTable* visibleFunctionTable, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVisibleFunctionTable_atIndex_), visibleFunctionTable, index);
}

_MTL_INLINE void MTL::ArgumentEncoder::setVisibleFunctionTables(const MTL::VisibleFunctionTable* const visibleFunctionTables[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVisibleFunctionTables_withRange_), visibleFunctionTables, range);
}
