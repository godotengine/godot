#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTLCommandEncoder.hpp"

namespace MTL {
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
    class Texture;
    class VisibleFunctionTable;
    using BarrierScope = NS::UInteger;
    enum IndexType : NS::UInteger;
    using ResourceUsage = NS::UInteger;
    enum StoreAction : NS::UInteger;
    using StoreActionOptions = NS::UInteger;
}

namespace MTL
{

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
    RenderStageVertex = (1UL << 0),
    RenderStageFragment = (1UL << 1),
    RenderStageTile = (1UL << 2),
    RenderStageObject = (1UL << 3),
    RenderStageMesh = (1UL << 4),
};


class RenderCommandEncoder : public NS::Referencing<RenderCommandEncoder, MTL::CommandEncoder>
{
public:
    void         dispatchThreadsPerTile(MTL::Size threadsPerTile);
    void         drawIndexedPatches(NS::UInteger numberOfPatchControlPoints, NS::UInteger patchStart, NS::UInteger patchCount, MTL::Buffer* patchIndexBuffer, NS::UInteger patchIndexBufferOffset, MTL::Buffer* controlPointIndexBuffer, NS::UInteger controlPointIndexBufferOffset, NS::UInteger instanceCount, NS::UInteger baseInstance);
    void         drawIndexedPatches(NS::UInteger numberOfPatchControlPoints, MTL::Buffer* patchIndexBuffer, NS::UInteger patchIndexBufferOffset, MTL::Buffer* controlPointIndexBuffer, NS::UInteger controlPointIndexBufferOffset, MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset);
    void         drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, MTL::Buffer* indexBuffer, NS::UInteger indexBufferOffset, NS::UInteger instanceCount);
    void         drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, MTL::Buffer* indexBuffer, NS::UInteger indexBufferOffset);
    void         drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, MTL::Buffer* indexBuffer, NS::UInteger indexBufferOffset, NS::UInteger instanceCount, NS::Integer baseVertex, NS::UInteger baseInstance);
    void         drawIndexedPrimitives(MTL::PrimitiveType primitiveType, MTL::IndexType indexType, MTL::Buffer* indexBuffer, NS::UInteger indexBufferOffset, MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset);
    void         drawMeshThreadgroups(MTL::Size threadgroupsPerGrid, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup);
    void         drawMeshThreadgroups(MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup);
    void         drawMeshThreads(MTL::Size threadsPerGrid, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup);
    void         drawPatches(NS::UInteger numberOfPatchControlPoints, NS::UInteger patchStart, NS::UInteger patchCount, MTL::Buffer* patchIndexBuffer, NS::UInteger patchIndexBufferOffset, NS::UInteger instanceCount, NS::UInteger baseInstance);
    void         drawPatches(NS::UInteger numberOfPatchControlPoints, MTL::Buffer* patchIndexBuffer, NS::UInteger patchIndexBufferOffset, MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset);
    void         drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount, NS::UInteger instanceCount);
    void         drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount);
    void         drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount, NS::UInteger instanceCount, NS::UInteger baseInstance);
    void         drawPrimitives(MTL::PrimitiveType primitiveType, MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset);
    void         executeCommandsInBuffer(MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::Range executionRange);
    void         executeCommandsInBuffer(MTL::IndirectCommandBuffer* indirectCommandbuffer, MTL::Buffer* indirectRangeBuffer, NS::UInteger indirectBufferOffset);
    void         memoryBarrier(MTL::BarrierScope scope, MTL::RenderStages after, MTL::RenderStages before);
    void         memoryBarrier(const MTL::Resource* const * resources, NS::UInteger count, MTL::RenderStages after, MTL::RenderStages before);
    void         sampleCountersInBuffer(MTL::CounterSampleBuffer* sampleBuffer, NS::UInteger sampleIndex, bool barrier);
    void         setBlendColor(float red, float green, float blue, float alpha);
    void         setColorAttachmentMap(MTL::LogicalToPhysicalColorAttachmentMap* mapping);
    void         setColorStoreAction(MTL::StoreAction storeAction, NS::UInteger colorAttachmentIndex);
    void         setColorStoreActionOptions(MTL::StoreActionOptions storeActionOptions, NS::UInteger colorAttachmentIndex);
    void         setCullMode(MTL::CullMode cullMode);
    void         setDepthBias(float depthBias, float slopeScale, float clamp);
    void         setDepthClipMode(MTL::DepthClipMode depthClipMode);
    void         setDepthStencilState(MTL::DepthStencilState* depthStencilState);
    void         setDepthStoreAction(MTL::StoreAction storeAction);
    void         setDepthStoreActionOptions(MTL::StoreActionOptions storeActionOptions);
    void         setDepthTestMinBound(float minBound, float maxBound);
    void         setFragmentAccelerationStructure(MTL::AccelerationStructure* accelerationStructure, NS::UInteger bufferIndex);
    void         setFragmentBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);
    void         setFragmentBufferOffset(NS::UInteger offset, NS::UInteger index);
    void         setFragmentBuffers(const MTL::Buffer* const * buffers, const NS::UInteger * offsets, NS::Range range);
    void         setFragmentBytes(const void * bytes, NS::UInteger length, NS::UInteger index);
    void         setFragmentIntersectionFunctionTable(MTL::IntersectionFunctionTable* intersectionFunctionTable, NS::UInteger bufferIndex);
    void         setFragmentIntersectionFunctionTables(const MTL::IntersectionFunctionTable* const * intersectionFunctionTables, NS::Range range);
    void         setFragmentSamplerState(MTL::SamplerState* sampler, NS::UInteger index);
    void         setFragmentSamplerState(MTL::SamplerState* sampler, float lodMinClamp, float lodMaxClamp, NS::UInteger index);
    void         setFragmentSamplerStates(const MTL::SamplerState* const * samplers, NS::Range range);
    void         setFragmentSamplerStates(const MTL::SamplerState* const * samplers, const float * lodMinClamps, const float * lodMaxClamps, NS::Range range);
    void         setFragmentTexture(MTL::Texture* texture, NS::UInteger index);
    void         setFragmentTextures(const MTL::Texture* const * textures, NS::Range range);
    void         setFragmentVisibleFunctionTable(MTL::VisibleFunctionTable* functionTable, NS::UInteger bufferIndex);
    void         setFragmentVisibleFunctionTables(const MTL::VisibleFunctionTable* const * functionTables, NS::Range range);
    void         setFrontFacingWinding(MTL::Winding frontFacingWinding);
    void         setMeshBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);
    void         setMeshBufferOffset(NS::UInteger offset, NS::UInteger index);
    void         setMeshBuffers(const MTL::Buffer* const * buffers, const NS::UInteger * offsets, NS::Range range);
    void         setMeshBytes(const void * bytes, NS::UInteger length, NS::UInteger index);
    void         setMeshSamplerState(MTL::SamplerState* sampler, NS::UInteger index);
    void         setMeshSamplerState(MTL::SamplerState* sampler, float lodMinClamp, float lodMaxClamp, NS::UInteger index);
    void         setMeshSamplerStates(const MTL::SamplerState* const * samplers, NS::Range range);
    void         setMeshSamplerStates(const MTL::SamplerState* const * samplers, const float * lodMinClamps, const float * lodMaxClamps, NS::Range range);
    void         setMeshTexture(MTL::Texture* texture, NS::UInteger index);
    void         setMeshTextures(const MTL::Texture* const * textures, NS::Range range);
    void         setObjectBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);
    void         setObjectBufferOffset(NS::UInteger offset, NS::UInteger index);
    void         setObjectBuffers(const MTL::Buffer* const * buffers, const NS::UInteger * offsets, NS::Range range);
    void         setObjectBytes(const void * bytes, NS::UInteger length, NS::UInteger index);
    void         setObjectSamplerState(MTL::SamplerState* sampler, NS::UInteger index);
    void         setObjectSamplerState(MTL::SamplerState* sampler, float lodMinClamp, float lodMaxClamp, NS::UInteger index);
    void         setObjectSamplerStates(const MTL::SamplerState* const * samplers, NS::Range range);
    void         setObjectSamplerStates(const MTL::SamplerState* const * samplers, const float * lodMinClamps, const float * lodMaxClamps, NS::Range range);
    void         setObjectTexture(MTL::Texture* texture, NS::UInteger index);
    void         setObjectTextures(const MTL::Texture* const * textures, NS::Range range);
    void         setObjectThreadgroupMemoryLength(NS::UInteger length, NS::UInteger index);
    void         setRenderPipelineState(MTL::RenderPipelineState* pipelineState);
    void         setScissorRect(MTL::ScissorRect rect);
    void         setScissorRects(const MTL::ScissorRect * scissorRects, NS::UInteger count);
    void         setStencilReferenceValue(uint32_t referenceValue);
    void         setStencilReferenceValues(uint32_t frontReferenceValue, uint32_t backReferenceValue);
    void         setStencilStoreAction(MTL::StoreAction storeAction);
    void         setStencilStoreActionOptions(MTL::StoreActionOptions storeActionOptions);
    void         setTessellationFactorBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger instanceStride);
    void         setTessellationFactorScale(float scale);
    void         setThreadgroupMemoryLength(NS::UInteger length, NS::UInteger offset, NS::UInteger index);
    void         setTileAccelerationStructure(MTL::AccelerationStructure* accelerationStructure, NS::UInteger bufferIndex);
    void         setTileBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);
    void         setTileBufferOffset(NS::UInteger offset, NS::UInteger index);
    void         setTileBuffers(const MTL::Buffer* const * buffers, const NS::UInteger * offsets, NS::Range range);
    void         setTileBytes(const void * bytes, NS::UInteger length, NS::UInteger index);
    void         setTileIntersectionFunctionTable(MTL::IntersectionFunctionTable* intersectionFunctionTable, NS::UInteger bufferIndex);
    void         setTileIntersectionFunctionTables(const MTL::IntersectionFunctionTable* const * intersectionFunctionTables, NS::Range range);
    void         setTileSamplerState(MTL::SamplerState* sampler, NS::UInteger index);
    void         setTileSamplerState(MTL::SamplerState* sampler, float lodMinClamp, float lodMaxClamp, NS::UInteger index);
    void         setTileSamplerStates(const MTL::SamplerState* const * samplers, NS::Range range);
    void         setTileSamplerStates(const MTL::SamplerState* const * samplers, const float * lodMinClamps, const float * lodMaxClamps, NS::Range range);
    void         setTileTexture(MTL::Texture* texture, NS::UInteger index);
    void         setTileTextures(const MTL::Texture* const * textures, NS::Range range);
    void         setTileVisibleFunctionTable(MTL::VisibleFunctionTable* functionTable, NS::UInteger bufferIndex);
    void         setTileVisibleFunctionTables(const MTL::VisibleFunctionTable* const * functionTables, NS::Range range);
    void         setTriangleFillMode(MTL::TriangleFillMode fillMode);
    void         setVertexAccelerationStructure(MTL::AccelerationStructure* accelerationStructure, NS::UInteger bufferIndex);
    void         setVertexAmplificationCount(NS::UInteger count, const MTL::VertexAmplificationViewMapping * viewMappings);
    void         setVertexBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);
    void         setVertexBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger stride, NS::UInteger index);
    void         setVertexBufferOffset(NS::UInteger offset, NS::UInteger index);
    void         setVertexBufferOffset(NS::UInteger offset, NS::UInteger stride, NS::UInteger index);
    void         setVertexBuffers(const MTL::Buffer* const * buffers, const NS::UInteger * offsets, NS::Range range);
    void         setVertexBuffers(const MTL::Buffer* const * buffers, const NS::UInteger * offsets, const NS::UInteger * strides, NS::Range range);
    void         setVertexBytes(const void * bytes, NS::UInteger length, NS::UInteger index);
    void         setVertexBytes(const void * bytes, NS::UInteger length, NS::UInteger stride, NS::UInteger index);
    void         setVertexIntersectionFunctionTable(MTL::IntersectionFunctionTable* intersectionFunctionTable, NS::UInteger bufferIndex);
    void         setVertexIntersectionFunctionTables(const MTL::IntersectionFunctionTable* const * intersectionFunctionTables, NS::Range range);
    void         setVertexSamplerState(MTL::SamplerState* sampler, NS::UInteger index);
    void         setVertexSamplerState(MTL::SamplerState* sampler, float lodMinClamp, float lodMaxClamp, NS::UInteger index);
    void         setVertexSamplerStates(const MTL::SamplerState* const * samplers, NS::Range range);
    void         setVertexSamplerStates(const MTL::SamplerState* const * samplers, const float * lodMinClamps, const float * lodMaxClamps, NS::Range range);
    void         setVertexTexture(MTL::Texture* texture, NS::UInteger index);
    void         setVertexTextures(const MTL::Texture* const * textures, NS::Range range);
    void         setVertexVisibleFunctionTable(MTL::VisibleFunctionTable* functionTable, NS::UInteger bufferIndex);
    void         setVertexVisibleFunctionTables(const MTL::VisibleFunctionTable* const * functionTables, NS::Range range);
    void         setViewport(MTL::Viewport viewport);
    void         setViewports(const MTL::Viewport * viewports, NS::UInteger count);
    void         setVisibilityResultMode(MTL::VisibilityResultMode mode, NS::UInteger offset);
    void         textureBarrier();
    NS::UInteger tileHeight() const;
    NS::UInteger tileWidth() const;
    void         updateFence(MTL::Fence* fence, MTL::RenderStages stages);
    void         useHeap(MTL::Heap* heap);
    void         useHeap(MTL::Heap* heap, MTL::RenderStages stages);
    void         useHeaps(const MTL::Heap* const * heaps, NS::UInteger count);
    void         useHeaps(const MTL::Heap* const * heaps, NS::UInteger count, MTL::RenderStages stages);
    void         useResource(MTL::Resource* resource, MTL::ResourceUsage usage);
    void         useResource(MTL::Resource* resource, MTL::ResourceUsage usage, MTL::RenderStages stages);
    void         useResources(const MTL::Resource* const * resources, NS::UInteger count, MTL::ResourceUsage usage);
    void         useResources(const MTL::Resource* const * resources, NS::UInteger count, MTL::ResourceUsage usage, MTL::RenderStages stages);
    void         waitForFence(MTL::Fence* fence, MTL::RenderStages stages);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLRenderCommandEncoder;

_MTL_INLINE NS::UInteger MTL::RenderCommandEncoder::tileWidth() const
{
    return _MTL_msg_NS__UInteger_tileWidth((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::RenderCommandEncoder::tileHeight() const
{
    return _MTL_msg_NS__UInteger_tileHeight((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setRenderPipelineState(MTL::RenderPipelineState* pipelineState)
{
    _MTL_msg_v_setRenderPipelineState__MTL__RenderPipelineStatep((const void*)this, nullptr, pipelineState);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexBytes(const void * bytes, NS::UInteger length, NS::UInteger index)
{
    _MTL_msg_v_setVertexBytes_length_atIndex__constvoidp_NS__UInteger_NS__UInteger((const void*)this, nullptr, bytes, length, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    _MTL_msg_v_setVertexBuffer_offset_atIndex__MTL__Bufferp_NS__UInteger_NS__UInteger((const void*)this, nullptr, buffer, offset, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexBufferOffset(NS::UInteger offset, NS::UInteger index)
{
    _MTL_msg_v_setVertexBufferOffset_atIndex__NS__UInteger_NS__UInteger((const void*)this, nullptr, offset, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexBuffers(const MTL::Buffer* const * buffers, const NS::UInteger * offsets, NS::Range range)
{
    _MTL_msg_v_setVertexBuffers_offsets_withRange__constMTL__Bufferpconstp_constNS__UIntegerp_NS__Range((const void*)this, nullptr, buffers, offsets, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger stride, NS::UInteger index)
{
    _MTL_msg_v_setVertexBuffer_offset_attributeStride_atIndex__MTL__Bufferp_NS__UInteger_NS__UInteger_NS__UInteger((const void*)this, nullptr, buffer, offset, stride, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexBuffers(const MTL::Buffer* const * buffers, const NS::UInteger * offsets, const NS::UInteger * strides, NS::Range range)
{
    _MTL_msg_v_setVertexBuffers_offsets_attributeStrides_withRange__constMTL__Bufferpconstp_constNS__UIntegerp_constNS__UIntegerp_NS__Range((const void*)this, nullptr, buffers, offsets, strides, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexBufferOffset(NS::UInteger offset, NS::UInteger stride, NS::UInteger index)
{
    _MTL_msg_v_setVertexBufferOffset_attributeStride_atIndex__NS__UInteger_NS__UInteger_NS__UInteger((const void*)this, nullptr, offset, stride, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexBytes(const void * bytes, NS::UInteger length, NS::UInteger stride, NS::UInteger index)
{
    _MTL_msg_v_setVertexBytes_length_attributeStride_atIndex__constvoidp_NS__UInteger_NS__UInteger_NS__UInteger((const void*)this, nullptr, bytes, length, stride, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexTexture(MTL::Texture* texture, NS::UInteger index)
{
    _MTL_msg_v_setVertexTexture_atIndex__MTL__Texturep_NS__UInteger((const void*)this, nullptr, texture, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexTextures(const MTL::Texture* const * textures, NS::Range range)
{
    _MTL_msg_v_setVertexTextures_withRange__constMTL__Texturepconstp_NS__Range((const void*)this, nullptr, textures, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexSamplerState(MTL::SamplerState* sampler, NS::UInteger index)
{
    _MTL_msg_v_setVertexSamplerState_atIndex__MTL__SamplerStatep_NS__UInteger((const void*)this, nullptr, sampler, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexSamplerStates(const MTL::SamplerState* const * samplers, NS::Range range)
{
    _MTL_msg_v_setVertexSamplerStates_withRange__constMTL__SamplerStatepconstp_NS__Range((const void*)this, nullptr, samplers, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexSamplerState(MTL::SamplerState* sampler, float lodMinClamp, float lodMaxClamp, NS::UInteger index)
{
    _MTL_msg_v_setVertexSamplerState_lodMinClamp_lodMaxClamp_atIndex__MTL__SamplerStatep_float_float_NS__UInteger((const void*)this, nullptr, sampler, lodMinClamp, lodMaxClamp, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexSamplerStates(const MTL::SamplerState* const * samplers, const float * lodMinClamps, const float * lodMaxClamps, NS::Range range)
{
    _MTL_msg_v_setVertexSamplerStates_lodMinClamps_lodMaxClamps_withRange__constMTL__SamplerStatepconstp_constfloatp_constfloatp_NS__Range((const void*)this, nullptr, samplers, lodMinClamps, lodMaxClamps, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexVisibleFunctionTable(MTL::VisibleFunctionTable* functionTable, NS::UInteger bufferIndex)
{
    _MTL_msg_v_setVertexVisibleFunctionTable_atBufferIndex__MTL__VisibleFunctionTablep_NS__UInteger((const void*)this, nullptr, functionTable, bufferIndex);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexVisibleFunctionTables(const MTL::VisibleFunctionTable* const * functionTables, NS::Range range)
{
    _MTL_msg_v_setVertexVisibleFunctionTables_withBufferRange__constMTL__VisibleFunctionTablepconstp_NS__Range((const void*)this, nullptr, functionTables, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexIntersectionFunctionTable(MTL::IntersectionFunctionTable* intersectionFunctionTable, NS::UInteger bufferIndex)
{
    _MTL_msg_v_setVertexIntersectionFunctionTable_atBufferIndex__MTL__IntersectionFunctionTablep_NS__UInteger((const void*)this, nullptr, intersectionFunctionTable, bufferIndex);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexIntersectionFunctionTables(const MTL::IntersectionFunctionTable* const * intersectionFunctionTables, NS::Range range)
{
    _MTL_msg_v_setVertexIntersectionFunctionTables_withBufferRange__constMTL__IntersectionFunctionTablepconstp_NS__Range((const void*)this, nullptr, intersectionFunctionTables, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexAccelerationStructure(MTL::AccelerationStructure* accelerationStructure, NS::UInteger bufferIndex)
{
    _MTL_msg_v_setVertexAccelerationStructure_atBufferIndex__MTL__AccelerationStructurep_NS__UInteger((const void*)this, nullptr, accelerationStructure, bufferIndex);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setViewport(MTL::Viewport viewport)
{
    _MTL_msg_v_setViewport__MTL__Viewport((const void*)this, nullptr, viewport);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setViewports(const MTL::Viewport * viewports, NS::UInteger count)
{
    _MTL_msg_v_setViewports_count__constMTL__Viewportp_NS__UInteger((const void*)this, nullptr, viewports, count);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFrontFacingWinding(MTL::Winding frontFacingWinding)
{
    _MTL_msg_v_setFrontFacingWinding__MTL__Winding((const void*)this, nullptr, frontFacingWinding);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVertexAmplificationCount(NS::UInteger count, const MTL::VertexAmplificationViewMapping * viewMappings)
{
    _MTL_msg_v_setVertexAmplificationCount_viewMappings__NS__UInteger_constMTL__VertexAmplificationViewMappingp((const void*)this, nullptr, count, viewMappings);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setCullMode(MTL::CullMode cullMode)
{
    _MTL_msg_v_setCullMode__MTL__CullMode((const void*)this, nullptr, cullMode);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setDepthClipMode(MTL::DepthClipMode depthClipMode)
{
    _MTL_msg_v_setDepthClipMode__MTL__DepthClipMode((const void*)this, nullptr, depthClipMode);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setDepthBias(float depthBias, float slopeScale, float clamp)
{
    _MTL_msg_v_setDepthBias_slopeScale_clamp__float_float_float((const void*)this, nullptr, depthBias, slopeScale, clamp);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setDepthTestMinBound(float minBound, float maxBound)
{
    _MTL_msg_v_setDepthTestMinBound_maxBound__float_float((const void*)this, nullptr, minBound, maxBound);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setScissorRect(MTL::ScissorRect rect)
{
    _MTL_msg_v_setScissorRect__MTL__ScissorRect((const void*)this, nullptr, rect);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setScissorRects(const MTL::ScissorRect * scissorRects, NS::UInteger count)
{
    _MTL_msg_v_setScissorRects_count__constMTL__ScissorRectp_NS__UInteger((const void*)this, nullptr, scissorRects, count);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTriangleFillMode(MTL::TriangleFillMode fillMode)
{
    _MTL_msg_v_setTriangleFillMode__MTL__TriangleFillMode((const void*)this, nullptr, fillMode);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentBytes(const void * bytes, NS::UInteger length, NS::UInteger index)
{
    _MTL_msg_v_setFragmentBytes_length_atIndex__constvoidp_NS__UInteger_NS__UInteger((const void*)this, nullptr, bytes, length, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    _MTL_msg_v_setFragmentBuffer_offset_atIndex__MTL__Bufferp_NS__UInteger_NS__UInteger((const void*)this, nullptr, buffer, offset, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentBufferOffset(NS::UInteger offset, NS::UInteger index)
{
    _MTL_msg_v_setFragmentBufferOffset_atIndex__NS__UInteger_NS__UInteger((const void*)this, nullptr, offset, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentBuffers(const MTL::Buffer* const * buffers, const NS::UInteger * offsets, NS::Range range)
{
    _MTL_msg_v_setFragmentBuffers_offsets_withRange__constMTL__Bufferpconstp_constNS__UIntegerp_NS__Range((const void*)this, nullptr, buffers, offsets, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentTexture(MTL::Texture* texture, NS::UInteger index)
{
    _MTL_msg_v_setFragmentTexture_atIndex__MTL__Texturep_NS__UInteger((const void*)this, nullptr, texture, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentTextures(const MTL::Texture* const * textures, NS::Range range)
{
    _MTL_msg_v_setFragmentTextures_withRange__constMTL__Texturepconstp_NS__Range((const void*)this, nullptr, textures, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentSamplerState(MTL::SamplerState* sampler, NS::UInteger index)
{
    _MTL_msg_v_setFragmentSamplerState_atIndex__MTL__SamplerStatep_NS__UInteger((const void*)this, nullptr, sampler, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentSamplerStates(const MTL::SamplerState* const * samplers, NS::Range range)
{
    _MTL_msg_v_setFragmentSamplerStates_withRange__constMTL__SamplerStatepconstp_NS__Range((const void*)this, nullptr, samplers, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentSamplerState(MTL::SamplerState* sampler, float lodMinClamp, float lodMaxClamp, NS::UInteger index)
{
    _MTL_msg_v_setFragmentSamplerState_lodMinClamp_lodMaxClamp_atIndex__MTL__SamplerStatep_float_float_NS__UInteger((const void*)this, nullptr, sampler, lodMinClamp, lodMaxClamp, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentSamplerStates(const MTL::SamplerState* const * samplers, const float * lodMinClamps, const float * lodMaxClamps, NS::Range range)
{
    _MTL_msg_v_setFragmentSamplerStates_lodMinClamps_lodMaxClamps_withRange__constMTL__SamplerStatepconstp_constfloatp_constfloatp_NS__Range((const void*)this, nullptr, samplers, lodMinClamps, lodMaxClamps, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentVisibleFunctionTable(MTL::VisibleFunctionTable* functionTable, NS::UInteger bufferIndex)
{
    _MTL_msg_v_setFragmentVisibleFunctionTable_atBufferIndex__MTL__VisibleFunctionTablep_NS__UInteger((const void*)this, nullptr, functionTable, bufferIndex);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentVisibleFunctionTables(const MTL::VisibleFunctionTable* const * functionTables, NS::Range range)
{
    _MTL_msg_v_setFragmentVisibleFunctionTables_withBufferRange__constMTL__VisibleFunctionTablepconstp_NS__Range((const void*)this, nullptr, functionTables, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentIntersectionFunctionTable(MTL::IntersectionFunctionTable* intersectionFunctionTable, NS::UInteger bufferIndex)
{
    _MTL_msg_v_setFragmentIntersectionFunctionTable_atBufferIndex__MTL__IntersectionFunctionTablep_NS__UInteger((const void*)this, nullptr, intersectionFunctionTable, bufferIndex);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentIntersectionFunctionTables(const MTL::IntersectionFunctionTable* const * intersectionFunctionTables, NS::Range range)
{
    _MTL_msg_v_setFragmentIntersectionFunctionTables_withBufferRange__constMTL__IntersectionFunctionTablepconstp_NS__Range((const void*)this, nullptr, intersectionFunctionTables, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setFragmentAccelerationStructure(MTL::AccelerationStructure* accelerationStructure, NS::UInteger bufferIndex)
{
    _MTL_msg_v_setFragmentAccelerationStructure_atBufferIndex__MTL__AccelerationStructurep_NS__UInteger((const void*)this, nullptr, accelerationStructure, bufferIndex);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setBlendColor(float red, float green, float blue, float alpha)
{
    _MTL_msg_v_setBlendColorRed_green_blue_alpha__float_float_float_float((const void*)this, nullptr, red, green, blue, alpha);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setDepthStencilState(MTL::DepthStencilState* depthStencilState)
{
    _MTL_msg_v_setDepthStencilState__MTL__DepthStencilStatep((const void*)this, nullptr, depthStencilState);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setStencilReferenceValue(uint32_t referenceValue)
{
    _MTL_msg_v_setStencilReferenceValue__uint32_t((const void*)this, nullptr, referenceValue);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setStencilReferenceValues(uint32_t frontReferenceValue, uint32_t backReferenceValue)
{
    _MTL_msg_v_setStencilFrontReferenceValue_backReferenceValue__uint32_t_uint32_t((const void*)this, nullptr, frontReferenceValue, backReferenceValue);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setVisibilityResultMode(MTL::VisibilityResultMode mode, NS::UInteger offset)
{
    _MTL_msg_v_setVisibilityResultMode_offset__MTL__VisibilityResultMode_NS__UInteger((const void*)this, nullptr, mode, offset);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setColorStoreAction(MTL::StoreAction storeAction, NS::UInteger colorAttachmentIndex)
{
    _MTL_msg_v_setColorStoreAction_atIndex__MTL__StoreAction_NS__UInteger((const void*)this, nullptr, storeAction, colorAttachmentIndex);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setDepthStoreAction(MTL::StoreAction storeAction)
{
    _MTL_msg_v_setDepthStoreAction__MTL__StoreAction((const void*)this, nullptr, storeAction);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setStencilStoreAction(MTL::StoreAction storeAction)
{
    _MTL_msg_v_setStencilStoreAction__MTL__StoreAction((const void*)this, nullptr, storeAction);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setColorStoreActionOptions(MTL::StoreActionOptions storeActionOptions, NS::UInteger colorAttachmentIndex)
{
    _MTL_msg_v_setColorStoreActionOptions_atIndex__MTL__StoreActionOptions_NS__UInteger((const void*)this, nullptr, storeActionOptions, colorAttachmentIndex);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setDepthStoreActionOptions(MTL::StoreActionOptions storeActionOptions)
{
    _MTL_msg_v_setDepthStoreActionOptions__MTL__StoreActionOptions((const void*)this, nullptr, storeActionOptions);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setStencilStoreActionOptions(MTL::StoreActionOptions storeActionOptions)
{
    _MTL_msg_v_setStencilStoreActionOptions__MTL__StoreActionOptions((const void*)this, nullptr, storeActionOptions);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setObjectBytes(const void * bytes, NS::UInteger length, NS::UInteger index)
{
    _MTL_msg_v_setObjectBytes_length_atIndex__constvoidp_NS__UInteger_NS__UInteger((const void*)this, nullptr, bytes, length, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setObjectBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    _MTL_msg_v_setObjectBuffer_offset_atIndex__MTL__Bufferp_NS__UInteger_NS__UInteger((const void*)this, nullptr, buffer, offset, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setObjectBufferOffset(NS::UInteger offset, NS::UInteger index)
{
    _MTL_msg_v_setObjectBufferOffset_atIndex__NS__UInteger_NS__UInteger((const void*)this, nullptr, offset, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setObjectBuffers(const MTL::Buffer* const * buffers, const NS::UInteger * offsets, NS::Range range)
{
    _MTL_msg_v_setObjectBuffers_offsets_withRange__constMTL__Bufferpconstp_constNS__UIntegerp_NS__Range((const void*)this, nullptr, buffers, offsets, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setObjectTexture(MTL::Texture* texture, NS::UInteger index)
{
    _MTL_msg_v_setObjectTexture_atIndex__MTL__Texturep_NS__UInteger((const void*)this, nullptr, texture, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setObjectTextures(const MTL::Texture* const * textures, NS::Range range)
{
    _MTL_msg_v_setObjectTextures_withRange__constMTL__Texturepconstp_NS__Range((const void*)this, nullptr, textures, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setObjectSamplerState(MTL::SamplerState* sampler, NS::UInteger index)
{
    _MTL_msg_v_setObjectSamplerState_atIndex__MTL__SamplerStatep_NS__UInteger((const void*)this, nullptr, sampler, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setObjectSamplerStates(const MTL::SamplerState* const * samplers, NS::Range range)
{
    _MTL_msg_v_setObjectSamplerStates_withRange__constMTL__SamplerStatepconstp_NS__Range((const void*)this, nullptr, samplers, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setObjectSamplerState(MTL::SamplerState* sampler, float lodMinClamp, float lodMaxClamp, NS::UInteger index)
{
    _MTL_msg_v_setObjectSamplerState_lodMinClamp_lodMaxClamp_atIndex__MTL__SamplerStatep_float_float_NS__UInteger((const void*)this, nullptr, sampler, lodMinClamp, lodMaxClamp, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setObjectSamplerStates(const MTL::SamplerState* const * samplers, const float * lodMinClamps, const float * lodMaxClamps, NS::Range range)
{
    _MTL_msg_v_setObjectSamplerStates_lodMinClamps_lodMaxClamps_withRange__constMTL__SamplerStatepconstp_constfloatp_constfloatp_NS__Range((const void*)this, nullptr, samplers, lodMinClamps, lodMaxClamps, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setObjectThreadgroupMemoryLength(NS::UInteger length, NS::UInteger index)
{
    _MTL_msg_v_setObjectThreadgroupMemoryLength_atIndex__NS__UInteger_NS__UInteger((const void*)this, nullptr, length, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setMeshBytes(const void * bytes, NS::UInteger length, NS::UInteger index)
{
    _MTL_msg_v_setMeshBytes_length_atIndex__constvoidp_NS__UInteger_NS__UInteger((const void*)this, nullptr, bytes, length, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setMeshBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    _MTL_msg_v_setMeshBuffer_offset_atIndex__MTL__Bufferp_NS__UInteger_NS__UInteger((const void*)this, nullptr, buffer, offset, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setMeshBufferOffset(NS::UInteger offset, NS::UInteger index)
{
    _MTL_msg_v_setMeshBufferOffset_atIndex__NS__UInteger_NS__UInteger((const void*)this, nullptr, offset, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setMeshBuffers(const MTL::Buffer* const * buffers, const NS::UInteger * offsets, NS::Range range)
{
    _MTL_msg_v_setMeshBuffers_offsets_withRange__constMTL__Bufferpconstp_constNS__UIntegerp_NS__Range((const void*)this, nullptr, buffers, offsets, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setMeshTexture(MTL::Texture* texture, NS::UInteger index)
{
    _MTL_msg_v_setMeshTexture_atIndex__MTL__Texturep_NS__UInteger((const void*)this, nullptr, texture, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setMeshTextures(const MTL::Texture* const * textures, NS::Range range)
{
    _MTL_msg_v_setMeshTextures_withRange__constMTL__Texturepconstp_NS__Range((const void*)this, nullptr, textures, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setMeshSamplerState(MTL::SamplerState* sampler, NS::UInteger index)
{
    _MTL_msg_v_setMeshSamplerState_atIndex__MTL__SamplerStatep_NS__UInteger((const void*)this, nullptr, sampler, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setMeshSamplerStates(const MTL::SamplerState* const * samplers, NS::Range range)
{
    _MTL_msg_v_setMeshSamplerStates_withRange__constMTL__SamplerStatepconstp_NS__Range((const void*)this, nullptr, samplers, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setMeshSamplerState(MTL::SamplerState* sampler, float lodMinClamp, float lodMaxClamp, NS::UInteger index)
{
    _MTL_msg_v_setMeshSamplerState_lodMinClamp_lodMaxClamp_atIndex__MTL__SamplerStatep_float_float_NS__UInteger((const void*)this, nullptr, sampler, lodMinClamp, lodMaxClamp, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setMeshSamplerStates(const MTL::SamplerState* const * samplers, const float * lodMinClamps, const float * lodMaxClamps, NS::Range range)
{
    _MTL_msg_v_setMeshSamplerStates_lodMinClamps_lodMaxClamps_withRange__constMTL__SamplerStatepconstp_constfloatp_constfloatp_NS__Range((const void*)this, nullptr, samplers, lodMinClamps, lodMaxClamps, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawMeshThreadgroups(MTL::Size threadgroupsPerGrid, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup)
{
    _MTL_msg_v_drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup__MTL__Size_MTL__Size_MTL__Size((const void*)this, nullptr, threadgroupsPerGrid, threadsPerObjectThreadgroup, threadsPerMeshThreadgroup);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawMeshThreads(MTL::Size threadsPerGrid, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup)
{
    _MTL_msg_v_drawMeshThreads_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup__MTL__Size_MTL__Size_MTL__Size((const void*)this, nullptr, threadsPerGrid, threadsPerObjectThreadgroup, threadsPerMeshThreadgroup);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawMeshThreadgroups(MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup)
{
    _MTL_msg_v_drawMeshThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup__MTL__Bufferp_NS__UInteger_MTL__Size_MTL__Size((const void*)this, nullptr, indirectBuffer, indirectBufferOffset, threadsPerObjectThreadgroup, threadsPerMeshThreadgroup);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount, NS::UInteger instanceCount)
{
    _MTL_msg_v_drawPrimitives_vertexStart_vertexCount_instanceCount__MTL__PrimitiveType_NS__UInteger_NS__UInteger_NS__UInteger((const void*)this, nullptr, primitiveType, vertexStart, vertexCount, instanceCount);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount)
{
    _MTL_msg_v_drawPrimitives_vertexStart_vertexCount__MTL__PrimitiveType_NS__UInteger_NS__UInteger((const void*)this, nullptr, primitiveType, vertexStart, vertexCount);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, MTL::Buffer* indexBuffer, NS::UInteger indexBufferOffset, NS::UInteger instanceCount)
{
    _MTL_msg_v_drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount__MTL__PrimitiveType_NS__UInteger_MTL__IndexType_MTL__Bufferp_NS__UInteger_NS__UInteger((const void*)this, nullptr, primitiveType, indexCount, indexType, indexBuffer, indexBufferOffset, instanceCount);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, MTL::Buffer* indexBuffer, NS::UInteger indexBufferOffset)
{
    _MTL_msg_v_drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset__MTL__PrimitiveType_NS__UInteger_MTL__IndexType_MTL__Bufferp_NS__UInteger((const void*)this, nullptr, primitiveType, indexCount, indexType, indexBuffer, indexBufferOffset);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount, NS::UInteger instanceCount, NS::UInteger baseInstance)
{
    _MTL_msg_v_drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance__MTL__PrimitiveType_NS__UInteger_NS__UInteger_NS__UInteger_NS__UInteger((const void*)this, nullptr, primitiveType, vertexStart, vertexCount, instanceCount, baseInstance);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, MTL::Buffer* indexBuffer, NS::UInteger indexBufferOffset, NS::UInteger instanceCount, NS::Integer baseVertex, NS::UInteger baseInstance)
{
    _MTL_msg_v_drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance__MTL__PrimitiveType_NS__UInteger_MTL__IndexType_MTL__Bufferp_NS__UInteger_NS__UInteger_NS__Integer_NS__UInteger((const void*)this, nullptr, primitiveType, indexCount, indexType, indexBuffer, indexBufferOffset, instanceCount, baseVertex, baseInstance);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawPrimitives(MTL::PrimitiveType primitiveType, MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset)
{
    _MTL_msg_v_drawPrimitives_indirectBuffer_indirectBufferOffset__MTL__PrimitiveType_MTL__Bufferp_NS__UInteger((const void*)this, nullptr, primitiveType, indirectBuffer, indirectBufferOffset);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawIndexedPrimitives(MTL::PrimitiveType primitiveType, MTL::IndexType indexType, MTL::Buffer* indexBuffer, NS::UInteger indexBufferOffset, MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset)
{
    _MTL_msg_v_drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset__MTL__PrimitiveType_MTL__IndexType_MTL__Bufferp_NS__UInteger_MTL__Bufferp_NS__UInteger((const void*)this, nullptr, primitiveType, indexType, indexBuffer, indexBufferOffset, indirectBuffer, indirectBufferOffset);
}

_MTL_INLINE void MTL::RenderCommandEncoder::textureBarrier()
{
    _MTL_msg_v_textureBarrier((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderCommandEncoder::updateFence(MTL::Fence* fence, MTL::RenderStages stages)
{
    _MTL_msg_v_updateFence_afterStages__MTL__Fencep_MTL__RenderStages((const void*)this, nullptr, fence, stages);
}

_MTL_INLINE void MTL::RenderCommandEncoder::waitForFence(MTL::Fence* fence, MTL::RenderStages stages)
{
    _MTL_msg_v_waitForFence_beforeStages__MTL__Fencep_MTL__RenderStages((const void*)this, nullptr, fence, stages);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTessellationFactorBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger instanceStride)
{
    _MTL_msg_v_setTessellationFactorBuffer_offset_instanceStride__MTL__Bufferp_NS__UInteger_NS__UInteger((const void*)this, nullptr, buffer, offset, instanceStride);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTessellationFactorScale(float scale)
{
    _MTL_msg_v_setTessellationFactorScale__float((const void*)this, nullptr, scale);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawPatches(NS::UInteger numberOfPatchControlPoints, NS::UInteger patchStart, NS::UInteger patchCount, MTL::Buffer* patchIndexBuffer, NS::UInteger patchIndexBufferOffset, NS::UInteger instanceCount, NS::UInteger baseInstance)
{
    _MTL_msg_v_drawPatches_patchStart_patchCount_patchIndexBuffer_patchIndexBufferOffset_instanceCount_baseInstance__NS__UInteger_NS__UInteger_NS__UInteger_MTL__Bufferp_NS__UInteger_NS__UInteger_NS__UInteger((const void*)this, nullptr, numberOfPatchControlPoints, patchStart, patchCount, patchIndexBuffer, patchIndexBufferOffset, instanceCount, baseInstance);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawPatches(NS::UInteger numberOfPatchControlPoints, MTL::Buffer* patchIndexBuffer, NS::UInteger patchIndexBufferOffset, MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset)
{
    _MTL_msg_v_drawPatches_patchIndexBuffer_patchIndexBufferOffset_indirectBuffer_indirectBufferOffset__NS__UInteger_MTL__Bufferp_NS__UInteger_MTL__Bufferp_NS__UInteger((const void*)this, nullptr, numberOfPatchControlPoints, patchIndexBuffer, patchIndexBufferOffset, indirectBuffer, indirectBufferOffset);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawIndexedPatches(NS::UInteger numberOfPatchControlPoints, NS::UInteger patchStart, NS::UInteger patchCount, MTL::Buffer* patchIndexBuffer, NS::UInteger patchIndexBufferOffset, MTL::Buffer* controlPointIndexBuffer, NS::UInteger controlPointIndexBufferOffset, NS::UInteger instanceCount, NS::UInteger baseInstance)
{
    _MTL_msg_v_drawIndexedPatches_patchStart_patchCount_patchIndexBuffer_patchIndexBufferOffset_controlPointIndexBuffer_controlPointIndexBufferOffset_instanceCount_baseInstance__NS__UInteger_NS__UInteger_NS__UInteger_MTL__Bufferp_NS__UInteger_MTL__Bufferp_NS__UInteger_NS__UInteger_NS__UInteger((const void*)this, nullptr, numberOfPatchControlPoints, patchStart, patchCount, patchIndexBuffer, patchIndexBufferOffset, controlPointIndexBuffer, controlPointIndexBufferOffset, instanceCount, baseInstance);
}

_MTL_INLINE void MTL::RenderCommandEncoder::drawIndexedPatches(NS::UInteger numberOfPatchControlPoints, MTL::Buffer* patchIndexBuffer, NS::UInteger patchIndexBufferOffset, MTL::Buffer* controlPointIndexBuffer, NS::UInteger controlPointIndexBufferOffset, MTL::Buffer* indirectBuffer, NS::UInteger indirectBufferOffset)
{
    _MTL_msg_v_drawIndexedPatches_patchIndexBuffer_patchIndexBufferOffset_controlPointIndexBuffer_controlPointIndexBufferOffset_indirectBuffer_indirectBufferOffset__NS__UInteger_MTL__Bufferp_NS__UInteger_MTL__Bufferp_NS__UInteger_MTL__Bufferp_NS__UInteger((const void*)this, nullptr, numberOfPatchControlPoints, patchIndexBuffer, patchIndexBufferOffset, controlPointIndexBuffer, controlPointIndexBufferOffset, indirectBuffer, indirectBufferOffset);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileBytes(const void * bytes, NS::UInteger length, NS::UInteger index)
{
    _MTL_msg_v_setTileBytes_length_atIndex__constvoidp_NS__UInteger_NS__UInteger((const void*)this, nullptr, bytes, length, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    _MTL_msg_v_setTileBuffer_offset_atIndex__MTL__Bufferp_NS__UInteger_NS__UInteger((const void*)this, nullptr, buffer, offset, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileBufferOffset(NS::UInteger offset, NS::UInteger index)
{
    _MTL_msg_v_setTileBufferOffset_atIndex__NS__UInteger_NS__UInteger((const void*)this, nullptr, offset, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileBuffers(const MTL::Buffer* const * buffers, const NS::UInteger * offsets, NS::Range range)
{
    _MTL_msg_v_setTileBuffers_offsets_withRange__constMTL__Bufferpconstp_constNS__UIntegerp_NS__Range((const void*)this, nullptr, buffers, offsets, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileTexture(MTL::Texture* texture, NS::UInteger index)
{
    _MTL_msg_v_setTileTexture_atIndex__MTL__Texturep_NS__UInteger((const void*)this, nullptr, texture, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileTextures(const MTL::Texture* const * textures, NS::Range range)
{
    _MTL_msg_v_setTileTextures_withRange__constMTL__Texturepconstp_NS__Range((const void*)this, nullptr, textures, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileSamplerState(MTL::SamplerState* sampler, NS::UInteger index)
{
    _MTL_msg_v_setTileSamplerState_atIndex__MTL__SamplerStatep_NS__UInteger((const void*)this, nullptr, sampler, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileSamplerStates(const MTL::SamplerState* const * samplers, NS::Range range)
{
    _MTL_msg_v_setTileSamplerStates_withRange__constMTL__SamplerStatepconstp_NS__Range((const void*)this, nullptr, samplers, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileSamplerState(MTL::SamplerState* sampler, float lodMinClamp, float lodMaxClamp, NS::UInteger index)
{
    _MTL_msg_v_setTileSamplerState_lodMinClamp_lodMaxClamp_atIndex__MTL__SamplerStatep_float_float_NS__UInteger((const void*)this, nullptr, sampler, lodMinClamp, lodMaxClamp, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileSamplerStates(const MTL::SamplerState* const * samplers, const float * lodMinClamps, const float * lodMaxClamps, NS::Range range)
{
    _MTL_msg_v_setTileSamplerStates_lodMinClamps_lodMaxClamps_withRange__constMTL__SamplerStatepconstp_constfloatp_constfloatp_NS__Range((const void*)this, nullptr, samplers, lodMinClamps, lodMaxClamps, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileVisibleFunctionTable(MTL::VisibleFunctionTable* functionTable, NS::UInteger bufferIndex)
{
    _MTL_msg_v_setTileVisibleFunctionTable_atBufferIndex__MTL__VisibleFunctionTablep_NS__UInteger((const void*)this, nullptr, functionTable, bufferIndex);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileVisibleFunctionTables(const MTL::VisibleFunctionTable* const * functionTables, NS::Range range)
{
    _MTL_msg_v_setTileVisibleFunctionTables_withBufferRange__constMTL__VisibleFunctionTablepconstp_NS__Range((const void*)this, nullptr, functionTables, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileIntersectionFunctionTable(MTL::IntersectionFunctionTable* intersectionFunctionTable, NS::UInteger bufferIndex)
{
    _MTL_msg_v_setTileIntersectionFunctionTable_atBufferIndex__MTL__IntersectionFunctionTablep_NS__UInteger((const void*)this, nullptr, intersectionFunctionTable, bufferIndex);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileIntersectionFunctionTables(const MTL::IntersectionFunctionTable* const * intersectionFunctionTables, NS::Range range)
{
    _MTL_msg_v_setTileIntersectionFunctionTables_withBufferRange__constMTL__IntersectionFunctionTablepconstp_NS__Range((const void*)this, nullptr, intersectionFunctionTables, range);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setTileAccelerationStructure(MTL::AccelerationStructure* accelerationStructure, NS::UInteger bufferIndex)
{
    _MTL_msg_v_setTileAccelerationStructure_atBufferIndex__MTL__AccelerationStructurep_NS__UInteger((const void*)this, nullptr, accelerationStructure, bufferIndex);
}

_MTL_INLINE void MTL::RenderCommandEncoder::dispatchThreadsPerTile(MTL::Size threadsPerTile)
{
    _MTL_msg_v_dispatchThreadsPerTile__MTL__Size((const void*)this, nullptr, threadsPerTile);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setThreadgroupMemoryLength(NS::UInteger length, NS::UInteger offset, NS::UInteger index)
{
    _MTL_msg_v_setThreadgroupMemoryLength_offset_atIndex__NS__UInteger_NS__UInteger_NS__UInteger((const void*)this, nullptr, length, offset, index);
}

_MTL_INLINE void MTL::RenderCommandEncoder::useResource(MTL::Resource* resource, MTL::ResourceUsage usage)
{
    _MTL_msg_v_useResource_usage__MTL__Resourcep_MTL__ResourceUsage((const void*)this, nullptr, resource, usage);
}

_MTL_INLINE void MTL::RenderCommandEncoder::useResources(const MTL::Resource* const * resources, NS::UInteger count, MTL::ResourceUsage usage)
{
    _MTL_msg_v_useResources_count_usage__constMTL__Resourcepconstp_NS__UInteger_MTL__ResourceUsage((const void*)this, nullptr, resources, count, usage);
}

_MTL_INLINE void MTL::RenderCommandEncoder::useResource(MTL::Resource* resource, MTL::ResourceUsage usage, MTL::RenderStages stages)
{
    _MTL_msg_v_useResource_usage_stages__MTL__Resourcep_MTL__ResourceUsage_MTL__RenderStages((const void*)this, nullptr, resource, usage, stages);
}

_MTL_INLINE void MTL::RenderCommandEncoder::useResources(const MTL::Resource* const * resources, NS::UInteger count, MTL::ResourceUsage usage, MTL::RenderStages stages)
{
    _MTL_msg_v_useResources_count_usage_stages__constMTL__Resourcepconstp_NS__UInteger_MTL__ResourceUsage_MTL__RenderStages((const void*)this, nullptr, resources, count, usage, stages);
}

_MTL_INLINE void MTL::RenderCommandEncoder::useHeap(MTL::Heap* heap)
{
    _MTL_msg_v_useHeap__MTL__Heapp((const void*)this, nullptr, heap);
}

_MTL_INLINE void MTL::RenderCommandEncoder::useHeaps(const MTL::Heap* const * heaps, NS::UInteger count)
{
    _MTL_msg_v_useHeaps_count__constMTL__Heappconstp_NS__UInteger((const void*)this, nullptr, heaps, count);
}

_MTL_INLINE void MTL::RenderCommandEncoder::useHeap(MTL::Heap* heap, MTL::RenderStages stages)
{
    _MTL_msg_v_useHeap_stages__MTL__Heapp_MTL__RenderStages((const void*)this, nullptr, heap, stages);
}

_MTL_INLINE void MTL::RenderCommandEncoder::useHeaps(const MTL::Heap* const * heaps, NS::UInteger count, MTL::RenderStages stages)
{
    _MTL_msg_v_useHeaps_count_stages__constMTL__Heappconstp_NS__UInteger_MTL__RenderStages((const void*)this, nullptr, heaps, count, stages);
}

_MTL_INLINE void MTL::RenderCommandEncoder::executeCommandsInBuffer(MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::Range executionRange)
{
    _MTL_msg_v_executeCommandsInBuffer_withRange__MTL__IndirectCommandBufferp_NS__Range((const void*)this, nullptr, indirectCommandBuffer, executionRange);
}

_MTL_INLINE void MTL::RenderCommandEncoder::executeCommandsInBuffer(MTL::IndirectCommandBuffer* indirectCommandbuffer, MTL::Buffer* indirectRangeBuffer, NS::UInteger indirectBufferOffset)
{
    _MTL_msg_v_executeCommandsInBuffer_indirectBuffer_indirectBufferOffset__MTL__IndirectCommandBufferp_MTL__Bufferp_NS__UInteger((const void*)this, nullptr, indirectCommandbuffer, indirectRangeBuffer, indirectBufferOffset);
}

_MTL_INLINE void MTL::RenderCommandEncoder::memoryBarrier(MTL::BarrierScope scope, MTL::RenderStages after, MTL::RenderStages before)
{
    _MTL_msg_v_memoryBarrierWithScope_afterStages_beforeStages__MTL__BarrierScope_MTL__RenderStages_MTL__RenderStages((const void*)this, nullptr, scope, after, before);
}

_MTL_INLINE void MTL::RenderCommandEncoder::memoryBarrier(const MTL::Resource* const * resources, NS::UInteger count, MTL::RenderStages after, MTL::RenderStages before)
{
    _MTL_msg_v_memoryBarrierWithResources_count_afterStages_beforeStages__constMTL__Resourcepconstp_NS__UInteger_MTL__RenderStages_MTL__RenderStages((const void*)this, nullptr, resources, count, after, before);
}

_MTL_INLINE void MTL::RenderCommandEncoder::sampleCountersInBuffer(MTL::CounterSampleBuffer* sampleBuffer, NS::UInteger sampleIndex, bool barrier)
{
    _MTL_msg_v_sampleCountersInBuffer_atSampleIndex_withBarrier__MTL__CounterSampleBufferp_NS__UInteger_bool((const void*)this, nullptr, sampleBuffer, sampleIndex, barrier);
}

_MTL_INLINE void MTL::RenderCommandEncoder::setColorAttachmentMap(MTL::LogicalToPhysicalColorAttachmentMap* mapping)
{
    _MTL_msg_v_setColorAttachmentMap__MTL__LogicalToPhysicalColorAttachmentMapp((const void*)this, nullptr, mapping);
}
