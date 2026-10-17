//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLResourceStatePass.hpp
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
#include "MTLArgument.hpp"
#include "MTLCommandEncoder.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include "MTLRenderPass.hpp"
#include "MTLTypes.hpp"
#include <cstdint>

namespace MTL
{
class AccelerationStructure;
class Buffer;
class CounterSampleBuffer;
class DepthStencilState;
class Fence;
class Heap;
class IndirectCommandBuffer;
class IntersectionFunctionTable;
class LogicalToPhysicalColorAttachmentMap;
class RenderPipelineState;
class Resource;
class SamplerState;
struct ScissorRect;
class Texture;
struct VertexAmplificationViewMapping;
struct Viewport;
class VisibleFunctionTable;
_MTL_ENUM(NS::UInteger, PrimitiveType) {
    PrimitiveTypePoint = 0,
    PrimitiveTypeLine = 1,
    PrimitiveTypeLineStrip = 2,
    PrimitiveTypeTriangle = 3,
    PrimitiveTypeTriangleStrip = 4,
};

_MTL_ENUM(NS::UInteger, VisibilityResultMode) {
    VisibilityResultModeDisabled = 0,
    VisibilityResultModeBoolean = 1,
    VisibilityResultModeCounting = 2,
};

_MTL_ENUM(NS::UInteger, CullMode) {
    CullModeNone = 0,
    CullModeFront = 1,
    CullModeBack = 2,
};

_MTL_ENUM(NS::UInteger, Winding) {
    WindingClockwise = 0,
    WindingCounterClockwise = 1,
};

_MTL_ENUM(NS::UInteger, DepthClipMode) {
    DepthClipModeClip = 0,
    DepthClipModeClamp = 1,
};

_MTL_ENUM(NS::UInteger, TriangleFillMode) {
    TriangleFillModeFill = 0,
    TriangleFillModeLines = 1,
};

_MTL_OPTIONS(NS::UInteger, RenderStages) {
    RenderStageVertex = 1,
    RenderStageFragment = 1 << 1,
    RenderStageTile = 1 << 2,
    RenderStageObject = 1 << 3,
    RenderStageMesh = 1 << 4,
};

struct ScissorRect
{
    NS::UInteger x;
    NS::UInteger y;
    NS::UInteger width;
    NS::UInteger height;
} _MTL_PACKED;

struct Viewport
{
    double originX;
    double originY;
    double width;
    double height;
    double znear;
    double zfar;
} _MTL_PACKED;

struct DrawPrimitivesIndirectArguments
{
    uint32_t vertexCount;
    uint32_t instanceCount;
    uint32_t vertexStart;
    uint32_t baseInstance;
} _MTL_PACKED;

struct DrawIndexedPrimitivesIndirectArguments
{
    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t indexStart;
    int32_t  baseVertex;
    uint32_t baseInstance;
} _MTL_PACKED;

struct VertexAmplificationViewMapping
{
    uint32_t viewportArrayIndexOffset;
    uint32_t renderTargetArrayIndexOffset;
} _MTL_PACKED;

struct DrawPatchIndirectArguments
{
    uint32_t patchCount;
    uint32_t instanceCount;
    uint32_t patchStart;
    uint32_t baseInstance;
} _MTL_PACKED;

struct QuadTessellationFactorsHalf
{
    uint16_t edgeTessellationFactor[4];
    uint16_t insideTessellationFactor[2];
} _MTL_PACKED;

struct TriangleTessellationFactorsHalf
{
    uint16_t edgeTessellationFactor[3];
    uint16_t insideTessellationFactor;
} _MTL_PACKED;

class RenderCommandEncoder : public NS::Referencing<RenderCommandEncoder, CommandEncoder>
{
public:
    void         dispatchThreadsPerTile(MTL::Size threadsPerTile);

    void         drawIndexedPatches(NS::UInteger numberOfPatchControlPoints, NS::UInteger patchStart, NS::UInteger patchCount, const MTL::Buffer* patchIndexBuffer, NS::UInteger patchIndexBufferOffset, const MTL::Buffer* controlPointIndexBuffer, NS::UInteger controlPointIndexBufferOffset, NS::UInteger instanceCount, NS::UInteger baseInstance);
    void         drawIndexedPatches(NS::UInteger numberOfPatchControlPoints, const MTL::Buffer* patchIndexBuffer, NS::UInteger patchIndexBufferOffset, const MTL::Buffer* controlPointIndexBuffer, NS::UInteger controlPointIndexBufferOffset, const MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset);

    void         drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, const MTL::Buffer* indexBuffer, NS::UInteger indexBufferOffset, NS::UInteger instanceCount);
    void         drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, const MTL::Buffer* indexBuffer, NS::UInteger indexBufferOffset);
    void         drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, const MTL::Buffer* indexBuffer, NS::UInteger indexBufferOffset, NS::UInteger instanceCount, NS::Integer baseVertex, NS::UInteger baseInstance);
    void         drawIndexedPrimitives(MTL::PrimitiveType primitiveType, MTL::IndexType indexType, const MTL::Buffer* indexBuffer, NS::UInteger indexBufferOffset, const MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset);

    void         drawMeshThreadgroups(MTL::Size threadgroupsPerGrid, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup);
    void         drawMeshThreadgroups(const MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup);

    void         drawMeshThreads(MTL::Size threadsPerGrid, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup);

    void         drawPatches(NS::UInteger numberOfPatchControlPoints, NS::UInteger patchStart, NS::UInteger patchCount, const MTL::Buffer* patchIndexBuffer, NS::UInteger patchIndexBufferOffset, NS::UInteger instanceCount, NS::UInteger baseInstance);
    void         drawPatches(NS::UInteger numberOfPatchControlPoints, const MTL::Buffer* patchIndexBuffer, NS::UInteger patchIndexBufferOffset, const MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset);

    void         drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount, NS::UInteger instanceCount);
    void         drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount);
    void         drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount, NS::UInteger instanceCount, NS::UInteger baseInstance);
    void         drawPrimitives(MTL::PrimitiveType primitiveType, const MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset);

    void         executeCommandsInBuffer(const MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::Range executionRange);
    void         executeCommandsInBuffer(const MTL::IndirectCommandBuffer* indirectCommandbuffer, const MTL::Buffer* indirectRangeBuffer, NS::UInteger indirectBufferOffset);

    void         memoryBarrier(MTL::BarrierScope scope, MTL::RenderStages after, MTL::RenderStages before);
    void         memoryBarrier(const MTL::Resource* const resources[], NS::UInteger count, MTL::RenderStages after, MTL::RenderStages before);

    void         sampleCountersInBuffer(const MTL::CounterSampleBuffer* sampleBuffer, NS::UInteger sampleIndex, bool barrier);

    void         setBlendColor(float red, float green, float blue, float alpha);

    void         setColorAttachmentMap(const MTL::LogicalToPhysicalColorAttachmentMap* mapping);

    void         setColorStoreAction(MTL::StoreAction storeAction, NS::UInteger colorAttachmentIndex);
    void         setColorStoreActionOptions(MTL::StoreActionOptions storeActionOptions, NS::UInteger colorAttachmentIndex);

    void         setCullMode(MTL::CullMode cullMode);

    void         setDepthBias(float depthBias, float slopeScale, float clamp);

    void         setDepthClipMode(MTL::DepthClipMode depthClipMode);

    void         setDepthStencilState(const MTL::DepthStencilState* depthStencilState);

    void         setDepthStoreAction(MTL::StoreAction storeAction);
    void         setDepthStoreActionOptions(MTL::StoreActionOptions storeActionOptions);

    void         setDepthTestBounds(float minBound, float maxBound);

    void         setFragmentAccelerationStructure(const MTL::AccelerationStructure* accelerationStructure, NS::UInteger bufferIndex);

    void         setFragmentBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);
    void         setFragmentBufferOffset(NS::UInteger offset, NS::UInteger index);

    void         setFragmentBuffers(const MTL::Buffer* const buffers[], const NS::UInteger offsets[], NS::Range range);

    void         setFragmentBytes(const void* bytes, NS::UInteger length, NS::UInteger index);

    void         setFragmentIntersectionFunctionTable(const MTL::IntersectionFunctionTable* intersectionFunctionTable, NS::UInteger bufferIndex);
    void         setFragmentIntersectionFunctionTables(const MTL::IntersectionFunctionTable* const intersectionFunctionTables[], NS::Range range);

    void         setFragmentSamplerState(const MTL::SamplerState* sampler, NS::UInteger index);
    void         setFragmentSamplerState(const MTL::SamplerState* sampler, float lodMinClamp, float lodMaxClamp, NS::UInteger index);
    void         setFragmentSamplerStates(const MTL::SamplerState* const samplers[], NS::Range range);
    void         setFragmentSamplerStates(const MTL::SamplerState* const samplers[], const float lodMinClamps[], const float lodMaxClamps[], NS::Range range);

    void         setFragmentTexture(const MTL::Texture* texture, NS::UInteger index);
    void         setFragmentTextures(const MTL::Texture* const textures[], NS::Range range);

    void         setFragmentVisibleFunctionTable(const MTL::VisibleFunctionTable* functionTable, NS::UInteger bufferIndex);
    void         setFragmentVisibleFunctionTables(const MTL::VisibleFunctionTable* const functionTables[], NS::Range range);

    void         setFrontFacingWinding(MTL::Winding frontFacingWinding);

    void         setMeshBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);
    void         setMeshBufferOffset(NS::UInteger offset, NS::UInteger index);

    void         setMeshBuffers(const MTL::Buffer* const buffers[], const NS::UInteger* offsets, NS::Range range);

    void         setMeshBytes(const void* bytes, NS::UInteger length, NS::UInteger index);

    void         setMeshSamplerState(const MTL::SamplerState* sampler, NS::UInteger index);
    void         setMeshSamplerState(const MTL::SamplerState* sampler, float lodMinClamp, float lodMaxClamp, NS::UInteger index);
    void         setMeshSamplerStates(const MTL::SamplerState* const samplers[], NS::Range range);
    void         setMeshSamplerStates(const MTL::SamplerState* const samplers[], const float* lodMinClamps, const float* lodMaxClamps, NS::Range range);

    void         setMeshTexture(const MTL::Texture* texture, NS::UInteger index);
    void         setMeshTextures(const MTL::Texture* const textures[], NS::Range range);

    void         setObjectBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);
    void         setObjectBufferOffset(NS::UInteger offset, NS::UInteger index);

    void         setObjectBuffers(const MTL::Buffer* const buffers[], const NS::UInteger* offsets, NS::Range range);

    void         setObjectBytes(const void* bytes, NS::UInteger length, NS::UInteger index);

    void         setObjectSamplerState(const MTL::SamplerState* sampler, NS::UInteger index);
    void         setObjectSamplerState(const MTL::SamplerState* sampler, float lodMinClamp, float lodMaxClamp, NS::UInteger index);
    void         setObjectSamplerStates(const MTL::SamplerState* const samplers[], NS::Range range);
    void         setObjectSamplerStates(const MTL::SamplerState* const samplers[], const float* lodMinClamps, const float* lodMaxClamps, NS::Range range);

    void         setObjectTexture(const MTL::Texture* texture, NS::UInteger index);
    void         setObjectTextures(const MTL::Texture* const textures[], NS::Range range);

    void         setObjectThreadgroupMemoryLength(NS::UInteger length, NS::UInteger index);

    void         setRenderPipelineState(const MTL::RenderPipelineState* pipelineState);

    void         setScissorRect(MTL::ScissorRect rect);
    void         setScissorRects(const MTL::ScissorRect* scissorRects, NS::UInteger count);

    void         setStencilReferenceValue(uint32_t referenceValue);
    void         setStencilReferenceValues(uint32_t frontReferenceValue, uint32_t backReferenceValue);

    void         setStencilStoreAction(MTL::StoreAction storeAction);
    void         setStencilStoreActionOptions(MTL::StoreActionOptions storeActionOptions);

    void         setTessellationFactorBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger instanceStride);

    void         setTessellationFactorScale(float scale);

    void         setThreadgroupMemoryLength(NS::UInteger length, NS::UInteger offset, NS::UInteger index);

    void         setTileAccelerationStructure(const MTL::AccelerationStructure* accelerationStructure, NS::UInteger bufferIndex);

    void         setTileBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);
    void         setTileBufferOffset(NS::UInteger offset, NS::UInteger index);

    void         setTileBuffers(const MTL::Buffer* const buffers[], const NS::UInteger* offsets, NS::Range range);

    void         setTileBytes(const void* bytes, NS::UInteger length, NS::UInteger index);

    void         setTileIntersectionFunctionTable(const MTL::IntersectionFunctionTable* intersectionFunctionTable, NS::UInteger bufferIndex);
    void         setTileIntersectionFunctionTables(const MTL::IntersectionFunctionTable* const intersectionFunctionTables[], NS::Range range);

    void         setTileSamplerState(const MTL::SamplerState* sampler, NS::UInteger index);
    void         setTileSamplerState(const MTL::SamplerState* sampler, float lodMinClamp, float lodMaxClamp, NS::UInteger index);
    void         setTileSamplerStates(const MTL::SamplerState* const samplers[], NS::Range range);
    void         setTileSamplerStates(const MTL::SamplerState* const samplers[], const float lodMinClamps[], const float lodMaxClamps[], NS::Range range);

    void         setTileTexture(const MTL::Texture* texture, NS::UInteger index);
    void         setTileTextures(const MTL::Texture* const textures[], NS::Range range);

    void         setTileVisibleFunctionTable(const MTL::VisibleFunctionTable* functionTable, NS::UInteger bufferIndex);
    void         setTileVisibleFunctionTables(const MTL::VisibleFunctionTable* const functionTables[], NS::Range range);

    void         setTriangleFillMode(MTL::TriangleFillMode fillMode);

    void         setVertexAccelerationStructure(const MTL::AccelerationStructure* accelerationStructure, NS::UInteger bufferIndex);

    void         setVertexAmplificationCount(NS::UInteger count, const MTL::VertexAmplificationViewMapping* viewMappings);

    void         setVertexBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);
    void         setVertexBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger stride, NS::UInteger index);
    void         setVertexBufferOffset(NS::UInteger offset, NS::UInteger index);
    void         setVertexBufferOffset(NS::UInteger offset, NS::UInteger stride, NS::UInteger index);

    void         setVertexBuffers(const MTL::Buffer* const buffers[], const NS::UInteger offsets[], NS::Range range);
    void         setVertexBuffers(const MTL::Buffer* const buffers[], const NS::UInteger* offsets, const NS::UInteger* strides, NS::Range range);

    void         setVertexBytes(const void* bytes, NS::UInteger length, NS::UInteger index);
    void         setVertexBytes(const void* bytes, NS::UInteger length, NS::UInteger stride, NS::UInteger index);

    void         setVertexIntersectionFunctionTable(const MTL::IntersectionFunctionTable* intersectionFunctionTable, NS::UInteger bufferIndex);
    void         setVertexIntersectionFunctionTables(const MTL::IntersectionFunctionTable* const intersectionFunctionTables[], NS::Range range);

    void         setVertexSamplerState(const MTL::SamplerState* sampler, NS::UInteger index);
    void         setVertexSamplerState(const MTL::SamplerState* sampler, float lodMinClamp, float lodMaxClamp, NS::UInteger index);
    void         setVertexSamplerStates(const MTL::SamplerState* const samplers[], NS::Range range);
    void         setVertexSamplerStates(const MTL::SamplerState* const samplers[], const float lodMinClamps[], const float lodMaxClamps[], NS::Range range);

    void         setVertexTexture(const MTL::Texture* texture, NS::UInteger index);
    void         setVertexTextures(const MTL::Texture* const textures[], NS::Range range);

    void         setVertexVisibleFunctionTable(const MTL::VisibleFunctionTable* functionTable, NS::UInteger bufferIndex);
    void         setVertexVisibleFunctionTables(const MTL::VisibleFunctionTable* const functionTables[], NS::Range range);

    void         setViewport(MTL::Viewport viewport);
    void         setViewports(const MTL::Viewport* viewports, NS::UInteger count);

    void         setVisibilityResultMode(MTL::VisibilityResultMode mode, NS::UInteger offset);

    void         textureBarrier();

    NS::UInteger tileHeight() const;

    NS::UInteger tileWidth() const;

    void         updateFence(const MTL::Fence* fence, MTL::RenderStages stages);

    void         useHeap(const MTL::Heap* heap);
    void         useHeap(const MTL::Heap* heap, MTL::RenderStages stages);
    void         useHeaps(const MTL::Heap* const heaps[], NS::UInteger count);
    void         useHeaps(const MTL::Heap* const heaps[], NS::UInteger count, MTL::RenderStages stages);

    void         useResource(const MTL::Resource* resource, MTL::ResourceUsage usage);
    void         useResource(const MTL::Resource* resource, MTL::ResourceUsage usage, MTL::RenderStages stages);
    void         useResources(const MTL::Resource* const resources[], NS::UInteger count, MTL::ResourceUsage usage);
    void         useResources(const MTL::Resource* const resources[], NS::UInteger count, MTL::ResourceUsage usage, MTL::RenderStages stages);

    void         waitForFence(const MTL::Fence* fence, MTL::RenderStages stages);
};

}

_MTL_INLINE void MTL::RenderCommandEncoder::dispatchThreadsPerTile(MTL::Size threadsPerTile)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(dispatchThreadsPerTile_), threadsPerTile);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawIndexedPatches(NS::UInteger numberOfPatchControlPoints, NS::UInteger patchStart, NS::UInteger patchCount, const MTL::Buffer* patchIndexBuffer, NS::UInteger patchIndexBufferOffset, const MTL::Buffer* controlPointIndexBuffer, NS::UInteger controlPointIndexBufferOffset, NS::UInteger instanceCount, NS::UInteger baseInstance)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawIndexedPatches_patchStart_patchCount_patchIndexBuffer_patchIndexBufferOffset_controlPointIndexBuffer_controlPointIndexBufferOffset_instanceCount_baseInstance_), numberOfPatchControlPoints, patchStart, patchCount, patchIndexBuffer, patchIndexBufferOffset, controlPointIndexBuffer, controlPointIndexBufferOffset, instanceCount, baseInstance);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawIndexedPatches(NS::UInteger numberOfPatchControlPoints, const MTL::Buffer* patchIndexBuffer, NS::UInteger patchIndexBufferOffset, const MTL::Buffer* controlPointIndexBuffer, NS::UInteger controlPointIndexBufferOffset, const MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawIndexedPatches_patchIndexBuffer_patchIndexBufferOffset_controlPointIndexBuffer_controlPointIndexBufferOffset_indirectBuffer_indirectBufferOffset_), numberOfPatchControlPoints, patchIndexBuffer, patchIndexBufferOffset, controlPointIndexBuffer, controlPointIndexBufferOffset, indirectBuffer, indirectBufferOffset);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, const MTL::Buffer* indexBuffer, NS::UInteger indexBufferOffset, NS::UInteger instanceCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_), primitiveType, indexCount, indexType, indexBuffer, indexBufferOffset, instanceCount);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, const MTL::Buffer* indexBuffer, NS::UInteger indexBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_), primitiveType, indexCount, indexType, indexBuffer, indexBufferOffset);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, const MTL::Buffer* indexBuffer, NS::UInteger indexBufferOffset, NS::UInteger instanceCount, NS::Integer baseVertex, NS::UInteger baseInstance)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance_), primitiveType, indexCount, indexType, indexBuffer, indexBufferOffset, instanceCount, baseVertex, baseInstance);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawIndexedPrimitives(MTL::PrimitiveType primitiveType, MTL::IndexType indexType, const MTL::Buffer* indexBuffer, NS::UInteger indexBufferOffset, const MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset_), primitiveType, indexType, indexBuffer, indexBufferOffset, indirectBuffer, indirectBufferOffset);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawMeshThreadgroups(MTL::Size threadgroupsPerGrid, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup_), threadgroupsPerGrid, threadsPerObjectThreadgroup, threadsPerMeshThreadgroup);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawMeshThreadgroups(const MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawMeshThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup_), indirectBuffer, indirectBufferOffset, threadsPerObjectThreadgroup, threadsPerMeshThreadgroup);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawMeshThreads(MTL::Size threadsPerGrid, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawMeshThreads_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup_), threadsPerGrid, threadsPerObjectThreadgroup, threadsPerMeshThreadgroup);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawPatches(NS::UInteger numberOfPatchControlPoints, NS::UInteger patchStart, NS::UInteger patchCount, const MTL::Buffer* patchIndexBuffer, NS::UInteger patchIndexBufferOffset, NS::UInteger instanceCount, NS::UInteger baseInstance)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawPatches_patchStart_patchCount_patchIndexBuffer_patchIndexBufferOffset_instanceCount_baseInstance_), numberOfPatchControlPoints, patchStart, patchCount, patchIndexBuffer, patchIndexBufferOffset, instanceCount, baseInstance);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawPatches(NS::UInteger numberOfPatchControlPoints, const MTL::Buffer* patchIndexBuffer, NS::UInteger patchIndexBufferOffset, const MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawPatches_patchIndexBuffer_patchIndexBufferOffset_indirectBuffer_indirectBufferOffset_), numberOfPatchControlPoints, patchIndexBuffer, patchIndexBufferOffset, indirectBuffer, indirectBufferOffset);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount, NS::UInteger instanceCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawPrimitives_vertexStart_vertexCount_instanceCount_), primitiveType, vertexStart, vertexCount, instanceCount);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawPrimitives_vertexStart_vertexCount_), primitiveType, vertexStart, vertexCount);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount, NS::UInteger instanceCount, NS::UInteger baseInstance)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance_), primitiveType, vertexStart, vertexCount, instanceCount, baseInstance);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawPrimitives(MTL::PrimitiveType primitiveType, const MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawPrimitives_indirectBuffer_indirectBufferOffset_), primitiveType, indirectBuffer, indirectBufferOffset);
}

_MTL_INLINE void MTL::RenderCommandEncoder::executeCommandsInBuffer(const MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::Range executionRange)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(executeCommandsInBuffer_withRange_), indirectCommandBuffer, executionRange);
}

_MTL_INLINE void MTL::RenderCommandEncoder::executeCommandsInBuffer(const MTL::IndirectCommandBuffer* indirectCommandbuffer, const MTL::Buffer* indirectRangeBuffer, NS::UInteger indirectBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(executeCommandsInBuffer_indirectBuffer_indirectBufferOffset_), indirectCommandbuffer, indirectRangeBuffer, indirectBufferOffset);
}

_MTL_INLINE void MTL::RenderCommandEncoder::memoryBarrier(MTL::BarrierScope scope, MTL::RenderStages after, MTL::RenderStages before)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(memoryBarrierWithScope_afterStages_beforeStages_), scope, after, before);
}

_MTL_INLINE void MTL::RenderCommandEncoder::memoryBarrier(const MTL::Resource* const resources[], NS::UInteger count, MTL::RenderStages after, MTL::RenderStages before)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(memoryBarrierWithResources_count_afterStages_beforeStages_), resources, count, after, before);
}

_MTL_INLINE void MTL::RenderCommandEncoder::sampleCountersInBuffer(const MTL::CounterSampleBuffer* sampleBuffer, NS::UInteger sampleIndex, bool barrier)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(sampleCountersInBuffer_atSampleIndex_withBarrier_), sampleBuffer, sampleIndex, barrier);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setBlendColor(float red, float green, float blue, float alpha)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBlendColorRed_green_blue_alpha_), red, green, blue, alpha);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setColorAttachmentMap(const MTL::LogicalToPhysicalColorAttachmentMap* mapping)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setColorAttachmentMap_), mapping);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setColorStoreAction(MTL::StoreAction storeAction, NS::UInteger colorAttachmentIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setColorStoreAction_atIndex_), storeAction, colorAttachmentIndex);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setColorStoreActionOptions(MTL::StoreActionOptions storeActionOptions, NS::UInteger colorAttachmentIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setColorStoreActionOptions_atIndex_), storeActionOptions, colorAttachmentIndex);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setCullMode(MTL::CullMode cullMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCullMode_), cullMode);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setDepthBias(float depthBias, float slopeScale, float clamp)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthBias_slopeScale_clamp_), depthBias, slopeScale, clamp);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setDepthClipMode(MTL::DepthClipMode depthClipMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthClipMode_), depthClipMode);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setDepthStencilState(const MTL::DepthStencilState* depthStencilState)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthStencilState_), depthStencilState);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setDepthStoreAction(MTL::StoreAction storeAction)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthStoreAction_), storeAction);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setDepthStoreActionOptions(MTL::StoreActionOptions storeActionOptions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthStoreActionOptions_), storeActionOptions);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setDepthTestBounds(float minBound, float maxBound)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthTestMinBound_maxBound_), minBound, maxBound);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentAccelerationStructure(const MTL::AccelerationStructure* accelerationStructure, NS::UInteger bufferIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentAccelerationStructure_atBufferIndex_), accelerationStructure, bufferIndex);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentBuffer_offset_atIndex_), buffer, offset, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentBufferOffset(NS::UInteger offset, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentBufferOffset_atIndex_), offset, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentBuffers(const MTL::Buffer* const buffers[], const NS::UInteger offsets[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentBuffers_offsets_withRange_), buffers, offsets, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentBytes(const void* bytes, NS::UInteger length, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentBytes_length_atIndex_), bytes, length, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentIntersectionFunctionTable(const MTL::IntersectionFunctionTable* intersectionFunctionTable, NS::UInteger bufferIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentIntersectionFunctionTable_atBufferIndex_), intersectionFunctionTable, bufferIndex);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentIntersectionFunctionTables(const MTL::IntersectionFunctionTable* const intersectionFunctionTables[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentIntersectionFunctionTables_withBufferRange_), intersectionFunctionTables, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentSamplerState(const MTL::SamplerState* sampler, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentSamplerState_atIndex_), sampler, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentSamplerState(const MTL::SamplerState* sampler, float lodMinClamp, float lodMaxClamp, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentSamplerState_lodMinClamp_lodMaxClamp_atIndex_), sampler, lodMinClamp, lodMaxClamp, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentSamplerStates(const MTL::SamplerState* const samplers[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentSamplerStates_withRange_), samplers, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentSamplerStates(const MTL::SamplerState* const samplers[], const float lodMinClamps[], const float lodMaxClamps[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentSamplerStates_lodMinClamps_lodMaxClamps_withRange_), samplers, lodMinClamps, lodMaxClamps, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentTexture(const MTL::Texture* texture, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentTexture_atIndex_), texture, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentTextures(const MTL::Texture* const textures[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentTextures_withRange_), textures, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentVisibleFunctionTable(const MTL::VisibleFunctionTable* functionTable, NS::UInteger bufferIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentVisibleFunctionTable_atBufferIndex_), functionTable, bufferIndex);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentVisibleFunctionTables(const MTL::VisibleFunctionTable* const functionTables[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentVisibleFunctionTables_withBufferRange_), functionTables, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFrontFacingWinding(MTL::Winding frontFacingWinding)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFrontFacingWinding_), frontFacingWinding);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setMeshBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMeshBuffer_offset_atIndex_), buffer, offset, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setMeshBufferOffset(NS::UInteger offset, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMeshBufferOffset_atIndex_), offset, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setMeshBuffers(const MTL::Buffer* const buffers[], const NS::UInteger* offsets, NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMeshBuffers_offsets_withRange_), buffers, offsets, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setMeshBytes(const void* bytes, NS::UInteger length, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMeshBytes_length_atIndex_), bytes, length, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setMeshSamplerState(const MTL::SamplerState* sampler, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMeshSamplerState_atIndex_), sampler, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setMeshSamplerState(const MTL::SamplerState* sampler, float lodMinClamp, float lodMaxClamp, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMeshSamplerState_lodMinClamp_lodMaxClamp_atIndex_), sampler, lodMinClamp, lodMaxClamp, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setMeshSamplerStates(const MTL::SamplerState* const samplers[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMeshSamplerStates_withRange_), samplers, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setMeshSamplerStates(const MTL::SamplerState* const samplers[], const float* lodMinClamps, const float* lodMaxClamps, NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMeshSamplerStates_lodMinClamps_lodMaxClamps_withRange_), samplers, lodMinClamps, lodMaxClamps, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setMeshTexture(const MTL::Texture* texture, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMeshTexture_atIndex_), texture, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setMeshTextures(const MTL::Texture* const textures[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMeshTextures_withRange_), textures, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setObjectBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObjectBuffer_offset_atIndex_), buffer, offset, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setObjectBufferOffset(NS::UInteger offset, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObjectBufferOffset_atIndex_), offset, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setObjectBuffers(const MTL::Buffer* const buffers[], const NS::UInteger* offsets, NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObjectBuffers_offsets_withRange_), buffers, offsets, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setObjectBytes(const void* bytes, NS::UInteger length, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObjectBytes_length_atIndex_), bytes, length, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setObjectSamplerState(const MTL::SamplerState* sampler, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObjectSamplerState_atIndex_), sampler, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setObjectSamplerState(const MTL::SamplerState* sampler, float lodMinClamp, float lodMaxClamp, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObjectSamplerState_lodMinClamp_lodMaxClamp_atIndex_), sampler, lodMinClamp, lodMaxClamp, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setObjectSamplerStates(const MTL::SamplerState* const samplers[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObjectSamplerStates_withRange_), samplers, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setObjectSamplerStates(const MTL::SamplerState* const samplers[], const float* lodMinClamps, const float* lodMaxClamps, NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObjectSamplerStates_lodMinClamps_lodMaxClamps_withRange_), samplers, lodMinClamps, lodMaxClamps, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setObjectTexture(const MTL::Texture* texture, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObjectTexture_atIndex_), texture, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setObjectTextures(const MTL::Texture* const textures[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObjectTextures_withRange_), textures, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setObjectThreadgroupMemoryLength(NS::UInteger length, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObjectThreadgroupMemoryLength_atIndex_), length, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setRenderPipelineState(const MTL::RenderPipelineState* pipelineState)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRenderPipelineState_), pipelineState);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setScissorRect(MTL::ScissorRect rect)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setScissorRect_), rect);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setScissorRects(const MTL::ScissorRect* scissorRects, NS::UInteger count)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setScissorRects_count_), scissorRects, count);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setStencilReferenceValue(uint32_t referenceValue)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStencilReferenceValue_), referenceValue);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setStencilReferenceValues(uint32_t frontReferenceValue, uint32_t backReferenceValue)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStencilFrontReferenceValue_backReferenceValue_), frontReferenceValue, backReferenceValue);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setStencilStoreAction(MTL::StoreAction storeAction)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStencilStoreAction_), storeAction);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setStencilStoreActionOptions(MTL::StoreActionOptions storeActionOptions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStencilStoreActionOptions_), storeActionOptions);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTessellationFactorBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger instanceStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTessellationFactorBuffer_offset_instanceStride_), buffer, offset, instanceStride);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTessellationFactorScale(float scale)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTessellationFactorScale_), scale);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setThreadgroupMemoryLength(NS::UInteger length, NS::UInteger offset, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setThreadgroupMemoryLength_offset_atIndex_), length, offset, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileAccelerationStructure(const MTL::AccelerationStructure* accelerationStructure, NS::UInteger bufferIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTileAccelerationStructure_atBufferIndex_), accelerationStructure, bufferIndex);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTileBuffer_offset_atIndex_), buffer, offset, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileBufferOffset(NS::UInteger offset, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTileBufferOffset_atIndex_), offset, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileBuffers(const MTL::Buffer* const buffers[], const NS::UInteger* offsets, NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTileBuffers_offsets_withRange_), buffers, offsets, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileBytes(const void* bytes, NS::UInteger length, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTileBytes_length_atIndex_), bytes, length, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileIntersectionFunctionTable(const MTL::IntersectionFunctionTable* intersectionFunctionTable, NS::UInteger bufferIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTileIntersectionFunctionTable_atBufferIndex_), intersectionFunctionTable, bufferIndex);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileIntersectionFunctionTables(const MTL::IntersectionFunctionTable* const intersectionFunctionTables[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTileIntersectionFunctionTables_withBufferRange_), intersectionFunctionTables, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileSamplerState(const MTL::SamplerState* sampler, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTileSamplerState_atIndex_), sampler, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileSamplerState(const MTL::SamplerState* sampler, float lodMinClamp, float lodMaxClamp, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTileSamplerState_lodMinClamp_lodMaxClamp_atIndex_), sampler, lodMinClamp, lodMaxClamp, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileSamplerStates(const MTL::SamplerState* const samplers[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTileSamplerStates_withRange_), samplers, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileSamplerStates(const MTL::SamplerState* const samplers[], const float lodMinClamps[], const float lodMaxClamps[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTileSamplerStates_lodMinClamps_lodMaxClamps_withRange_), samplers, lodMinClamps, lodMaxClamps, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileTexture(const MTL::Texture* texture, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTileTexture_atIndex_), texture, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileTextures(const MTL::Texture* const textures[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTileTextures_withRange_), textures, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileVisibleFunctionTable(const MTL::VisibleFunctionTable* functionTable, NS::UInteger bufferIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTileVisibleFunctionTable_atBufferIndex_), functionTable, bufferIndex);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileVisibleFunctionTables(const MTL::VisibleFunctionTable* const functionTables[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTileVisibleFunctionTables_withBufferRange_), functionTables, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTriangleFillMode(MTL::TriangleFillMode fillMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTriangleFillMode_), fillMode);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexAccelerationStructure(const MTL::AccelerationStructure* accelerationStructure, NS::UInteger bufferIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexAccelerationStructure_atBufferIndex_), accelerationStructure, bufferIndex);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexAmplificationCount(NS::UInteger count, const MTL::VertexAmplificationViewMapping* viewMappings)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexAmplificationCount_viewMappings_), count, viewMappings);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexBuffer_offset_atIndex_), buffer, offset, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger stride, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexBuffer_offset_attributeStride_atIndex_), buffer, offset, stride, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexBufferOffset(NS::UInteger offset, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexBufferOffset_atIndex_), offset, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexBufferOffset(NS::UInteger offset, NS::UInteger stride, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexBufferOffset_attributeStride_atIndex_), offset, stride, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexBuffers(const MTL::Buffer* const buffers[], const NS::UInteger offsets[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexBuffers_offsets_withRange_), buffers, offsets, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexBuffers(const MTL::Buffer* const buffers[], const NS::UInteger* offsets, const NS::UInteger* strides, NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexBuffers_offsets_attributeStrides_withRange_), buffers, offsets, strides, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexBytes(const void* bytes, NS::UInteger length, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexBytes_length_atIndex_), bytes, length, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexBytes(const void* bytes, NS::UInteger length, NS::UInteger stride, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexBytes_length_attributeStride_atIndex_), bytes, length, stride, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexIntersectionFunctionTable(const MTL::IntersectionFunctionTable* intersectionFunctionTable, NS::UInteger bufferIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexIntersectionFunctionTable_atBufferIndex_), intersectionFunctionTable, bufferIndex);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexIntersectionFunctionTables(const MTL::IntersectionFunctionTable* const intersectionFunctionTables[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexIntersectionFunctionTables_withBufferRange_), intersectionFunctionTables, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexSamplerState(const MTL::SamplerState* sampler, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexSamplerState_atIndex_), sampler, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexSamplerState(const MTL::SamplerState* sampler, float lodMinClamp, float lodMaxClamp, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexSamplerState_lodMinClamp_lodMaxClamp_atIndex_), sampler, lodMinClamp, lodMaxClamp, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexSamplerStates(const MTL::SamplerState* const samplers[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexSamplerStates_withRange_), samplers, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexSamplerStates(const MTL::SamplerState* const samplers[], const float lodMinClamps[], const float lodMaxClamps[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexSamplerStates_lodMinClamps_lodMaxClamps_withRange_), samplers, lodMinClamps, lodMaxClamps, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexTexture(const MTL::Texture* texture, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexTexture_atIndex_), texture, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexTextures(const MTL::Texture* const textures[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexTextures_withRange_), textures, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexVisibleFunctionTable(const MTL::VisibleFunctionTable* functionTable, NS::UInteger bufferIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexVisibleFunctionTable_atBufferIndex_), functionTable, bufferIndex);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexVisibleFunctionTables(const MTL::VisibleFunctionTable* const functionTables[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexVisibleFunctionTables_withBufferRange_), functionTables, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setViewport(MTL::Viewport viewport)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setViewport_), viewport);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setViewports(const MTL::Viewport* viewports, NS::UInteger count)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setViewports_count_), viewports, count);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVisibilityResultMode(MTL::VisibilityResultMode mode, NS::UInteger offset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVisibilityResultMode_offset_), mode, offset);
}

_MTL_INLINE void MTL::RenderCommandEncoder::textureBarrier()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(textureBarrier));
}

_MTL_INLINE NS::UInteger MTL::RenderCommandEncoder::tileHeight() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(tileHeight));
}

_MTL_INLINE NS::UInteger MTL::RenderCommandEncoder::tileWidth() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(tileWidth));
}

_MTL_INLINE void MTL::RenderCommandEncoder::updateFence(const MTL::Fence* fence, MTL::RenderStages stages)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(updateFence_afterStages_), fence, stages);
}

_MTL_INLINE void MTL::RenderCommandEncoder::useHeap(const MTL::Heap* heap)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(useHeap_), heap);
}

_MTL_INLINE void MTL::RenderCommandEncoder::useHeap(const MTL::Heap* heap, MTL::RenderStages stages)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(useHeap_stages_), heap, stages);
}

_MTL_INLINE void MTL::RenderCommandEncoder::useHeaps(const MTL::Heap* const heaps[], NS::UInteger count)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(useHeaps_count_), heaps, count);
}

_MTL_INLINE void MTL::RenderCommandEncoder::useHeaps(const MTL::Heap* const heaps[], NS::UInteger count, MTL::RenderStages stages)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(useHeaps_count_stages_), heaps, count, stages);
}

_MTL_INLINE void MTL::RenderCommandEncoder::useResource(const MTL::Resource* resource, MTL::ResourceUsage usage)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(useResource_usage_), resource, usage);
}

_MTL_INLINE void MTL::RenderCommandEncoder::useResource(const MTL::Resource* resource, MTL::ResourceUsage usage, MTL::RenderStages stages)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(useResource_usage_stages_), resource, usage, stages);
}

_MTL_INLINE void MTL::RenderCommandEncoder::useResources(const MTL::Resource* const resources[], NS::UInteger count, MTL::ResourceUsage usage)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(useResources_count_usage_), resources, count, usage);
}

_MTL_INLINE void MTL::RenderCommandEncoder::useResources(const MTL::Resource* const resources[], NS::UInteger count, MTL::ResourceUsage usage, MTL::RenderStages stages)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(useResources_count_usage_stages_), resources, count, usage, stages);
}

_MTL_INLINE void MTL::RenderCommandEncoder::waitForFence(const MTL::Fence* fence, MTL::RenderStages stages)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(waitForFence_beforeStages_), fence, stages);
}
