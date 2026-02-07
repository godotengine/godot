//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTL4RenderCommandEncoder.hpp
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
#include "MTL4CommandEncoder.hpp"
#include "MTL4Counters.hpp"
#include "MTLArgument.hpp"
#include "MTLDefines.hpp"
#include "MTLGPUAddress.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include "MTLRenderCommandEncoder.hpp"
#include "MTLRenderPass.hpp"
#include "MTLTypes.hpp"
#include <cstdint>

namespace MTL4
{
class ArgumentTable;
class CounterHeap;
}

namespace MTL
{
class DepthStencilState;
class IndirectCommandBuffer;
class LogicalToPhysicalColorAttachmentMap;
class RenderPipelineState;
struct ScissorRect;
struct VertexAmplificationViewMapping;
struct Viewport;

}
namespace MTL4
{
_MTL_OPTIONS(NS::UInteger, RenderEncoderOptions) {
    RenderEncoderOptionNone = 0,
    RenderEncoderOptionSuspending = 1,
    RenderEncoderOptionResuming = 1 << 1,
};

class RenderCommandEncoder : public NS::Referencing<RenderCommandEncoder, CommandEncoder>
{
public:
    void         dispatchThreadsPerTile(MTL::Size threadsPerTile);

    void         drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, MTL::GPUAddress indexBuffer, NS::UInteger indexBufferLength);
    void         drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, MTL::GPUAddress indexBuffer, NS::UInteger indexBufferLength, NS::UInteger instanceCount);
    void         drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, MTL::GPUAddress indexBuffer, NS::UInteger indexBufferLength, NS::UInteger instanceCount, NS::Integer baseVertex, NS::UInteger baseInstance);
    void         drawIndexedPrimitives(MTL::PrimitiveType primitiveType, MTL::IndexType indexType, MTL::GPUAddress indexBuffer, NS::UInteger indexBufferLength, MTL::GPUAddress indirectBuffer);

    void         drawMeshThreadgroups(MTL::Size threadgroupsPerGrid, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup);
    void         drawMeshThreadgroups(MTL::GPUAddress indirectBuffer, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup);

    void         drawMeshThreads(MTL::Size threadsPerGrid, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup);

    void         drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount);
    void         drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount, NS::UInteger instanceCount);
    void         drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount, NS::UInteger instanceCount, NS::UInteger baseInstance);
    void         drawPrimitives(MTL::PrimitiveType primitiveType, MTL::GPUAddress indirectBuffer);

    void         executeCommandsInBuffer(const MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::Range executionRange);
    void         executeCommandsInBuffer(const MTL::IndirectCommandBuffer* indirectCommandBuffer, MTL::GPUAddress indirectRangeBuffer);

    void         setArgumentTable(const MTL4::ArgumentTable* argumentTable, MTL::RenderStages stages);

    void         setBlendColor(float red, float green, float blue, float alpha);

    void         setColorAttachmentMap(const MTL::LogicalToPhysicalColorAttachmentMap* mapping);

    void         setColorStoreAction(MTL::StoreAction storeAction, NS::UInteger colorAttachmentIndex);

    void         setCullMode(MTL::CullMode cullMode);

    void         setDepthBias(float depthBias, float slopeScale, float clamp);

    void         setDepthClipMode(MTL::DepthClipMode depthClipMode);

    void         setDepthStencilState(const MTL::DepthStencilState* depthStencilState);

    void         setDepthStoreAction(MTL::StoreAction storeAction);

    void         setDepthTestBounds(float minBound, float maxBound);

    void         setFrontFacingWinding(MTL::Winding frontFacingWinding);

    void         setObjectThreadgroupMemoryLength(NS::UInteger length, NS::UInteger index);

    void         setRenderPipelineState(const MTL::RenderPipelineState* pipelineState);

    void         setScissorRect(MTL::ScissorRect rect);
    void         setScissorRects(const MTL::ScissorRect* scissorRects, NS::UInteger count);

    void         setStencilReferenceValue(uint32_t referenceValue);
    void         setStencilReferenceValues(uint32_t frontReferenceValue, uint32_t backReferenceValue);

    void         setStencilStoreAction(MTL::StoreAction storeAction);

    void         setThreadgroupMemoryLength(NS::UInteger length, NS::UInteger offset, NS::UInteger index);

    void         setTriangleFillMode(MTL::TriangleFillMode fillMode);

    void         setVertexAmplificationCount(NS::UInteger count, const MTL::VertexAmplificationViewMapping* viewMappings);

    void         setViewport(MTL::Viewport viewport);
    void         setViewports(const MTL::Viewport* viewports, NS::UInteger count);

    void         setVisibilityResultMode(MTL::VisibilityResultMode mode, NS::UInteger offset);

    NS::UInteger tileHeight() const;

    NS::UInteger tileWidth() const;

    void         writeTimestamp(MTL4::TimestampGranularity granularity, MTL::RenderStages stage, const MTL4::CounterHeap* counterHeap, NS::UInteger index);
};

}
_MTL_INLINE void MTL4::RenderCommandEncoder::dispatchThreadsPerTile(MTL::Size threadsPerTile)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(dispatchThreadsPerTile_), threadsPerTile);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, MTL::GPUAddress indexBuffer, NS::UInteger indexBufferLength)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferLength_), primitiveType, indexCount, indexType, indexBuffer, indexBufferLength);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, MTL::GPUAddress indexBuffer, NS::UInteger indexBufferLength, NS::UInteger instanceCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferLength_instanceCount_), primitiveType, indexCount, indexType, indexBuffer, indexBufferLength, instanceCount);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, MTL::GPUAddress indexBuffer, NS::UInteger indexBufferLength, NS::UInteger instanceCount, NS::Integer baseVertex, NS::UInteger baseInstance)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferLength_instanceCount_baseVertex_baseInstance_), primitiveType, indexCount, indexType, indexBuffer, indexBufferLength, instanceCount, baseVertex, baseInstance);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::drawIndexedPrimitives(MTL::PrimitiveType primitiveType, MTL::IndexType indexType, MTL::GPUAddress indexBuffer, NS::UInteger indexBufferLength, MTL::GPUAddress indirectBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawIndexedPrimitives_indexType_indexBuffer_indexBufferLength_indirectBuffer_), primitiveType, indexType, indexBuffer, indexBufferLength, indirectBuffer);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::drawMeshThreadgroups(MTL::Size threadgroupsPerGrid, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup_), threadgroupsPerGrid, threadsPerObjectThreadgroup, threadsPerMeshThreadgroup);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::drawMeshThreadgroups(MTL::GPUAddress indirectBuffer, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawMeshThreadgroupsWithIndirectBuffer_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup_), indirectBuffer, threadsPerObjectThreadgroup, threadsPerMeshThreadgroup);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::drawMeshThreads(MTL::Size threadsPerGrid, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawMeshThreads_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup_), threadsPerGrid, threadsPerObjectThreadgroup, threadsPerMeshThreadgroup);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawPrimitives_vertexStart_vertexCount_), primitiveType, vertexStart, vertexCount);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount, NS::UInteger instanceCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawPrimitives_vertexStart_vertexCount_instanceCount_), primitiveType, vertexStart, vertexCount, instanceCount);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount, NS::UInteger instanceCount, NS::UInteger baseInstance)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance_), primitiveType, vertexStart, vertexCount, instanceCount, baseInstance);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::drawPrimitives(MTL::PrimitiveType primitiveType, MTL::GPUAddress indirectBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawPrimitives_indirectBuffer_), primitiveType, indirectBuffer);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::executeCommandsInBuffer(const MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::Range executionRange)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(executeCommandsInBuffer_withRange_), indirectCommandBuffer, executionRange);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::executeCommandsInBuffer(const MTL::IndirectCommandBuffer* indirectCommandBuffer, MTL::GPUAddress indirectRangeBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(executeCommandsInBuffer_indirectBuffer_), indirectCommandBuffer, indirectRangeBuffer);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::setArgumentTable(const MTL4::ArgumentTable* argumentTable, MTL::RenderStages stages)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setArgumentTable_atStages_), argumentTable, stages);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::setBlendColor(float red, float green, float blue, float alpha)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBlendColorRed_green_blue_alpha_), red, green, blue, alpha);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::setColorAttachmentMap(const MTL::LogicalToPhysicalColorAttachmentMap* mapping)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setColorAttachmentMap_), mapping);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::setColorStoreAction(MTL::StoreAction storeAction, NS::UInteger colorAttachmentIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setColorStoreAction_atIndex_), storeAction, colorAttachmentIndex);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::setCullMode(MTL::CullMode cullMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCullMode_), cullMode);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::setDepthBias(float depthBias, float slopeScale, float clamp)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthBias_slopeScale_clamp_), depthBias, slopeScale, clamp);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::setDepthClipMode(MTL::DepthClipMode depthClipMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthClipMode_), depthClipMode);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::setDepthStencilState(const MTL::DepthStencilState* depthStencilState)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthStencilState_), depthStencilState);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::setDepthStoreAction(MTL::StoreAction storeAction)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthStoreAction_), storeAction);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::setDepthTestBounds(float minBound, float maxBound)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthTestMinBound_maxBound_), minBound, maxBound);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::setFrontFacingWinding(MTL::Winding frontFacingWinding)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFrontFacingWinding_), frontFacingWinding);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::setObjectThreadgroupMemoryLength(NS::UInteger length, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObjectThreadgroupMemoryLength_atIndex_), length, index);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::setRenderPipelineState(const MTL::RenderPipelineState* pipelineState)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRenderPipelineState_), pipelineState);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::setScissorRect(MTL::ScissorRect rect)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setScissorRect_), rect);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::setScissorRects(const MTL::ScissorRect* scissorRects, NS::UInteger count)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setScissorRects_count_), scissorRects, count);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::setStencilReferenceValue(uint32_t referenceValue)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStencilReferenceValue_), referenceValue);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::setStencilReferenceValues(uint32_t frontReferenceValue, uint32_t backReferenceValue)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStencilFrontReferenceValue_backReferenceValue_), frontReferenceValue, backReferenceValue);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::setStencilStoreAction(MTL::StoreAction storeAction)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStencilStoreAction_), storeAction);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::setThreadgroupMemoryLength(NS::UInteger length, NS::UInteger offset, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setThreadgroupMemoryLength_offset_atIndex_), length, offset, index);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::setTriangleFillMode(MTL::TriangleFillMode fillMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTriangleFillMode_), fillMode);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::setVertexAmplificationCount(NS::UInteger count, const MTL::VertexAmplificationViewMapping* viewMappings)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexAmplificationCount_viewMappings_), count, viewMappings);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::setViewport(MTL::Viewport viewport)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setViewport_), viewport);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::setViewports(const MTL::Viewport* viewports, NS::UInteger count)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setViewports_count_), viewports, count);
}

_MTL_INLINE void MTL4::RenderCommandEncoder::setVisibilityResultMode(MTL::VisibilityResultMode mode, NS::UInteger offset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVisibilityResultMode_offset_), mode, offset);
}

_MTL_INLINE NS::UInteger MTL4::RenderCommandEncoder::tileHeight() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(tileHeight));
}

_MTL_INLINE NS::UInteger MTL4::RenderCommandEncoder::tileWidth() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(tileWidth));
}

_MTL_INLINE void MTL4::RenderCommandEncoder::writeTimestamp(MTL4::TimestampGranularity granularity, MTL::RenderStages stage, const MTL4::CounterHeap* counterHeap, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(writeTimestampWithGranularity_afterStage_intoHeap_atIndex_), granularity, stage, counterHeap, index);
}
