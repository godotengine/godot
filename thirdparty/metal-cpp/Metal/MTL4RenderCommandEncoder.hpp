#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTL4CommandEncoder.hpp"
#include "MTLStructs.hpp"

namespace MTL {
    class DepthStencilState;
    class IndirectCommandBuffer;
    class LogicalToPhysicalColorAttachmentMap;
    class RenderPipelineState;
    enum CullMode : NS::UInteger;
    enum DepthClipMode : NS::UInteger;
    enum IndexType : NS::UInteger;
    enum PrimitiveType : NS::UInteger;
    using RenderStages = NS::UInteger;
    enum StoreAction : NS::UInteger;
    enum TriangleFillMode : NS::UInteger;
    enum VisibilityResultMode : NS::UInteger;
    enum Winding : NS::UInteger;
}
namespace MTL4 {
    class ArgumentTable;
    class CounterHeap;
    enum TimestampGranularity : NS::Integer;
}

namespace MTL4
{

_MTL4_OPTIONS(NS::UInteger, RenderEncoderOptions) {
    RenderEncoderOptionNone = 0,
    RenderEncoderOptionSuspending = (1 << 0),
    RenderEncoderOptionResuming = (1 << 1),
};


class RenderCommandEncoder : public NS::Referencing<RenderCommandEncoder, MTL4::CommandEncoder>
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
    void         executeCommandsInBuffer(MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::Range executionRange);
    void         executeCommandsInBuffer(MTL::IndirectCommandBuffer* indirectCommandBuffer, MTL::GPUAddress indirectRangeBuffer);
    void         setArgumentTable(MTL4::ArgumentTable* argumentTable, MTL::RenderStages stages);
    void         setBlendColor(float red, float green, float blue, float alpha);
    void         setColorAttachmentMap(MTL::LogicalToPhysicalColorAttachmentMap* mapping);
    void         setColorStoreAction(MTL::StoreAction storeAction, NS::UInteger colorAttachmentIndex);
    void         setCullMode(MTL::CullMode cullMode);
    void         setDepthBias(float depthBias, float slopeScale, float clamp);
    void         setDepthClipMode(MTL::DepthClipMode depthClipMode);
    void         setDepthStencilState(MTL::DepthStencilState* depthStencilState);
    void         setDepthStoreAction(MTL::StoreAction storeAction);
    void         setDepthTestMinBound(float minBound, float maxBound);
    void         setFrontFacingWinding(MTL::Winding frontFacingWinding);
    void         setObjectThreadgroupMemoryLength(NS::UInteger length, NS::UInteger index);
    void         setRenderPipelineState(MTL::RenderPipelineState* pipelineState);
    void         setScissorRect(MTL::ScissorRect rect);
    void         setScissorRects(const MTL::ScissorRect * scissorRects, NS::UInteger count);
    void         setStencilReferenceValue(uint32_t referenceValue);
    void         setStencilReferenceValues(uint32_t frontReferenceValue, uint32_t backReferenceValue);
    void         setStencilStoreAction(MTL::StoreAction storeAction);
    void         setThreadgroupMemoryLength(NS::UInteger length, NS::UInteger offset, NS::UInteger index);
    void         setTriangleFillMode(MTL::TriangleFillMode fillMode);
    void         setVertexAmplificationCount(NS::UInteger count, const MTL::VertexAmplificationViewMapping * viewMappings);
    void         setViewport(MTL::Viewport viewport);
    void         setViewports(const MTL::Viewport * viewports, NS::UInteger count);
    void         setVisibilityResultMode(MTL::VisibilityResultMode mode, NS::UInteger offset);
    NS::UInteger tileHeight() const;
    NS::UInteger tileWidth() const;
    void         writeTimestamp(MTL4::TimestampGranularity granularity, MTL::RenderStages stage, MTL4::CounterHeap* counterHeap, NS::UInteger index);

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4RenderCommandEncoder;

_MTL4_INLINE NS::UInteger MTL4::RenderCommandEncoder::tileWidth() const
{
    return _MTL4_msg_NS__UInteger_tileWidth((const void*)this, nullptr);
}

_MTL4_INLINE NS::UInteger MTL4::RenderCommandEncoder::tileHeight() const
{
    return _MTL4_msg_NS__UInteger_tileHeight((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::setColorAttachmentMap(MTL::LogicalToPhysicalColorAttachmentMap* mapping)
{
    _MTL4_msg_v_setColorAttachmentMap__MTL__LogicalToPhysicalColorAttachmentMapp((const void*)this, nullptr, mapping);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::setRenderPipelineState(MTL::RenderPipelineState* pipelineState)
{
    _MTL4_msg_v_setRenderPipelineState__MTL__RenderPipelineStatep((const void*)this, nullptr, pipelineState);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::setViewport(MTL::Viewport viewport)
{
    _MTL4_msg_v_setViewport__MTL__Viewport((const void*)this, nullptr, viewport);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::setViewports(const MTL::Viewport * viewports, NS::UInteger count)
{
    _MTL4_msg_v_setViewports_count__constMTL__Viewportp_NS__UInteger((const void*)this, nullptr, viewports, count);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::setVertexAmplificationCount(NS::UInteger count, const MTL::VertexAmplificationViewMapping * viewMappings)
{
    _MTL4_msg_v_setVertexAmplificationCount_viewMappings__NS__UInteger_constMTL__VertexAmplificationViewMappingp((const void*)this, nullptr, count, viewMappings);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::setCullMode(MTL::CullMode cullMode)
{
    _MTL4_msg_v_setCullMode__MTL__CullMode((const void*)this, nullptr, cullMode);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::setDepthClipMode(MTL::DepthClipMode depthClipMode)
{
    _MTL4_msg_v_setDepthClipMode__MTL__DepthClipMode((const void*)this, nullptr, depthClipMode);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::setDepthBias(float depthBias, float slopeScale, float clamp)
{
    _MTL4_msg_v_setDepthBias_slopeScale_clamp__float_float_float((const void*)this, nullptr, depthBias, slopeScale, clamp);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::setDepthTestMinBound(float minBound, float maxBound)
{
    _MTL4_msg_v_setDepthTestMinBound_maxBound__float_float((const void*)this, nullptr, minBound, maxBound);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::setScissorRect(MTL::ScissorRect rect)
{
    _MTL4_msg_v_setScissorRect__MTL__ScissorRect((const void*)this, nullptr, rect);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::setScissorRects(const MTL::ScissorRect * scissorRects, NS::UInteger count)
{
    _MTL4_msg_v_setScissorRects_count__constMTL__ScissorRectp_NS__UInteger((const void*)this, nullptr, scissorRects, count);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::setTriangleFillMode(MTL::TriangleFillMode fillMode)
{
    _MTL4_msg_v_setTriangleFillMode__MTL__TriangleFillMode((const void*)this, nullptr, fillMode);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::setBlendColor(float red, float green, float blue, float alpha)
{
    _MTL4_msg_v_setBlendColorRed_green_blue_alpha__float_float_float_float((const void*)this, nullptr, red, green, blue, alpha);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::setDepthStencilState(MTL::DepthStencilState* depthStencilState)
{
    _MTL4_msg_v_setDepthStencilState__MTL__DepthStencilStatep((const void*)this, nullptr, depthStencilState);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::setStencilReferenceValue(uint32_t referenceValue)
{
    _MTL4_msg_v_setStencilReferenceValue__uint32_t((const void*)this, nullptr, referenceValue);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::setStencilReferenceValues(uint32_t frontReferenceValue, uint32_t backReferenceValue)
{
    _MTL4_msg_v_setStencilFrontReferenceValue_backReferenceValue__uint32_t_uint32_t((const void*)this, nullptr, frontReferenceValue, backReferenceValue);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::setVisibilityResultMode(MTL::VisibilityResultMode mode, NS::UInteger offset)
{
    _MTL4_msg_v_setVisibilityResultMode_offset__MTL__VisibilityResultMode_NS__UInteger((const void*)this, nullptr, mode, offset);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::setColorStoreAction(MTL::StoreAction storeAction, NS::UInteger colorAttachmentIndex)
{
    _MTL4_msg_v_setColorStoreAction_atIndex__MTL__StoreAction_NS__UInteger((const void*)this, nullptr, storeAction, colorAttachmentIndex);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::setDepthStoreAction(MTL::StoreAction storeAction)
{
    _MTL4_msg_v_setDepthStoreAction__MTL__StoreAction((const void*)this, nullptr, storeAction);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::setStencilStoreAction(MTL::StoreAction storeAction)
{
    _MTL4_msg_v_setStencilStoreAction__MTL__StoreAction((const void*)this, nullptr, storeAction);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount)
{
    _MTL4_msg_v_drawPrimitives_vertexStart_vertexCount__MTL__PrimitiveType_NS__UInteger_NS__UInteger((const void*)this, nullptr, primitiveType, vertexStart, vertexCount);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount, NS::UInteger instanceCount)
{
    _MTL4_msg_v_drawPrimitives_vertexStart_vertexCount_instanceCount__MTL__PrimitiveType_NS__UInteger_NS__UInteger_NS__UInteger((const void*)this, nullptr, primitiveType, vertexStart, vertexCount, instanceCount);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount, NS::UInteger instanceCount, NS::UInteger baseInstance)
{
    _MTL4_msg_v_drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance__MTL__PrimitiveType_NS__UInteger_NS__UInteger_NS__UInteger_NS__UInteger((const void*)this, nullptr, primitiveType, vertexStart, vertexCount, instanceCount, baseInstance);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, MTL::GPUAddress indexBuffer, NS::UInteger indexBufferLength)
{
    _MTL4_msg_v_drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferLength__MTL__PrimitiveType_NS__UInteger_MTL__IndexType_MTL__GPUAddress_NS__UInteger((const void*)this, nullptr, primitiveType, indexCount, indexType, indexBuffer, indexBufferLength);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, MTL::GPUAddress indexBuffer, NS::UInteger indexBufferLength, NS::UInteger instanceCount)
{
    _MTL4_msg_v_drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferLength_instanceCount__MTL__PrimitiveType_NS__UInteger_MTL__IndexType_MTL__GPUAddress_NS__UInteger_NS__UInteger((const void*)this, nullptr, primitiveType, indexCount, indexType, indexBuffer, indexBufferLength, instanceCount);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, MTL::GPUAddress indexBuffer, NS::UInteger indexBufferLength, NS::UInteger instanceCount, NS::Integer baseVertex, NS::UInteger baseInstance)
{
    _MTL4_msg_v_drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferLength_instanceCount_baseVertex_baseInstance__MTL__PrimitiveType_NS__UInteger_MTL__IndexType_MTL__GPUAddress_NS__UInteger_NS__UInteger_NS__Integer_NS__UInteger((const void*)this, nullptr, primitiveType, indexCount, indexType, indexBuffer, indexBufferLength, instanceCount, baseVertex, baseInstance);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::drawPrimitives(MTL::PrimitiveType primitiveType, MTL::GPUAddress indirectBuffer)
{
    _MTL4_msg_v_drawPrimitives_indirectBuffer__MTL__PrimitiveType_MTL__GPUAddress((const void*)this, nullptr, primitiveType, indirectBuffer);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::drawIndexedPrimitives(MTL::PrimitiveType primitiveType, MTL::IndexType indexType, MTL::GPUAddress indexBuffer, NS::UInteger indexBufferLength, MTL::GPUAddress indirectBuffer)
{
    _MTL4_msg_v_drawIndexedPrimitives_indexType_indexBuffer_indexBufferLength_indirectBuffer__MTL__PrimitiveType_MTL__IndexType_MTL__GPUAddress_NS__UInteger_MTL__GPUAddress((const void*)this, nullptr, primitiveType, indexType, indexBuffer, indexBufferLength, indirectBuffer);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::executeCommandsInBuffer(MTL::IndirectCommandBuffer* indirectCommandBuffer, NS::Range executionRange)
{
    _MTL4_msg_v_executeCommandsInBuffer_withRange__MTL__IndirectCommandBufferp_NS__Range((const void*)this, nullptr, indirectCommandBuffer, executionRange);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::executeCommandsInBuffer(MTL::IndirectCommandBuffer* indirectCommandBuffer, MTL::GPUAddress indirectRangeBuffer)
{
    _MTL4_msg_v_executeCommandsInBuffer_indirectBuffer__MTL__IndirectCommandBufferp_MTL__GPUAddress((const void*)this, nullptr, indirectCommandBuffer, indirectRangeBuffer);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::setObjectThreadgroupMemoryLength(NS::UInteger length, NS::UInteger index)
{
    _MTL4_msg_v_setObjectThreadgroupMemoryLength_atIndex__NS__UInteger_NS__UInteger((const void*)this, nullptr, length, index);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::drawMeshThreadgroups(MTL::Size threadgroupsPerGrid, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup)
{
    _MTL4_msg_v_drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup__MTL__Size_MTL__Size_MTL__Size((const void*)this, nullptr, threadgroupsPerGrid, threadsPerObjectThreadgroup, threadsPerMeshThreadgroup);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::drawMeshThreads(MTL::Size threadsPerGrid, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup)
{
    _MTL4_msg_v_drawMeshThreads_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup__MTL__Size_MTL__Size_MTL__Size((const void*)this, nullptr, threadsPerGrid, threadsPerObjectThreadgroup, threadsPerMeshThreadgroup);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::drawMeshThreadgroups(MTL::GPUAddress indirectBuffer, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup)
{
    _MTL4_msg_v_drawMeshThreadgroupsWithIndirectBuffer_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup__MTL__GPUAddress_MTL__Size_MTL__Size((const void*)this, nullptr, indirectBuffer, threadsPerObjectThreadgroup, threadsPerMeshThreadgroup);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::dispatchThreadsPerTile(MTL::Size threadsPerTile)
{
    _MTL4_msg_v_dispatchThreadsPerTile__MTL__Size((const void*)this, nullptr, threadsPerTile);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::setThreadgroupMemoryLength(NS::UInteger length, NS::UInteger offset, NS::UInteger index)
{
    _MTL4_msg_v_setThreadgroupMemoryLength_offset_atIndex__NS__UInteger_NS__UInteger_NS__UInteger((const void*)this, nullptr, length, offset, index);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::setArgumentTable(MTL4::ArgumentTable* argumentTable, MTL::RenderStages stages)
{
    _MTL4_msg_v_setArgumentTable_atStages__MTL4__ArgumentTablep_MTL__RenderStages((const void*)this, nullptr, argumentTable, stages);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::setFrontFacingWinding(MTL::Winding frontFacingWinding)
{
    _MTL4_msg_v_setFrontFacingWinding__MTL__Winding((const void*)this, nullptr, frontFacingWinding);
}

_MTL4_INLINE void MTL4::RenderCommandEncoder::writeTimestamp(MTL4::TimestampGranularity granularity, MTL::RenderStages stage, MTL4::CounterHeap* counterHeap, NS::UInteger index)
{
    _MTL4_msg_v_writeTimestampWithGranularity_afterStage_intoHeap_atIndex__MTL4__TimestampGranularity_MTL__RenderStages_MTL4__CounterHeapp_NS__UInteger((const void*)this, nullptr, granularity, stage, counterHeap, index);
}
