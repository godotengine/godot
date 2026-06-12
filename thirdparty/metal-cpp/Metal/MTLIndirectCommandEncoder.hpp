#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class Buffer;
    class ComputePipelineState;
    class DepthStencilState;
    class RenderPipelineState;
    enum CullMode : NS::UInteger;
    enum DepthClipMode : NS::UInteger;
    enum IndexType : NS::UInteger;
    enum PrimitiveType : NS::UInteger;
    enum TriangleFillMode : NS::UInteger;
    enum Winding : NS::UInteger;
}

namespace MTL
{

class IndirectRenderCommand;
class IndirectComputeCommand;

class IndirectRenderCommand : public NS::Referencing<IndirectRenderCommand>
{
public:
    void clearBarrier();
    void drawIndexedPatches(NS::UInteger numberOfPatchControlPoints, NS::UInteger patchStart, NS::UInteger patchCount, MTL::Buffer* patchIndexBuffer, NS::UInteger patchIndexBufferOffset, MTL::Buffer* controlPointIndexBuffer, NS::UInteger controlPointIndexBufferOffset, NS::UInteger instanceCount, NS::UInteger baseInstance, MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger instanceStride);
    void drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, MTL::Buffer* indexBuffer, NS::UInteger indexBufferOffset, NS::UInteger instanceCount, NS::Integer baseVertex, NS::UInteger baseInstance);
    void drawMeshThreadgroups(MTL::Size threadgroupsPerGrid, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup);
    void drawMeshThreads(MTL::Size threadsPerGrid, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup);
    void drawPatches(NS::UInteger numberOfPatchControlPoints, NS::UInteger patchStart, NS::UInteger patchCount, MTL::Buffer* patchIndexBuffer, NS::UInteger patchIndexBufferOffset, NS::UInteger instanceCount, NS::UInteger baseInstance, MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger instanceStride);
    void drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount, NS::UInteger instanceCount, NS::UInteger baseInstance);
    void reset();
    void setBarrier();
    void setCullMode(MTL::CullMode cullMode);
    void setDepthBias(float depthBias, float slopeScale, float clamp);
    void setDepthClipMode(MTL::DepthClipMode depthClipMode);
    void setDepthStencilState(MTL::DepthStencilState* depthStencilState);
    void setFragmentBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);
    void setFrontFacingWinding(MTL::Winding frontFacingWindning);
    void setMeshBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);
    void setObjectBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);
    void setObjectThreadgroupMemoryLength(NS::UInteger length, NS::UInteger index);
    void setRenderPipelineState(MTL::RenderPipelineState* pipelineState);
    void setTriangleFillMode(MTL::TriangleFillMode fillMode);
    void setVertexBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);
    void setVertexBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger stride, NS::UInteger index);

};

class IndirectComputeCommand : public NS::Referencing<IndirectComputeCommand>
{
public:
    void clearBarrier();
    void concurrentDispatchThreadgroups(MTL::Size threadgroupsPerGrid, MTL::Size threadsPerThreadgroup);
    void concurrentDispatchThreads(MTL::Size threadsPerGrid, MTL::Size threadsPerThreadgroup);
    void reset();
    void setBarrier();
    void setComputePipelineState(MTL::ComputePipelineState* pipelineState);
    void setImageblockWidth(NS::UInteger width, NS::UInteger height);
    void setKernelBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);
    void setKernelBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger stride, NS::UInteger index);
    void setStageInRegion(MTL::Region region);
    void setThreadgroupMemoryLength(NS::UInteger length, NS::UInteger index);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLIndirectRenderCommand;
extern "C" void *OBJC_CLASS_$_MTLIndirectComputeCommand;

_MTL_INLINE void MTL::IndirectRenderCommand::setRenderPipelineState(MTL::RenderPipelineState* pipelineState)
{
    _MTL_msg_v_setRenderPipelineState__MTL__RenderPipelineStatep((const void*)this, nullptr, pipelineState);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setVertexBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    _MTL_msg_v_setVertexBuffer_offset_atIndex__MTL__Bufferp_NS__UInteger_NS__UInteger((const void*)this, nullptr, buffer, offset, index);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setFragmentBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    _MTL_msg_v_setFragmentBuffer_offset_atIndex__MTL__Bufferp_NS__UInteger_NS__UInteger((const void*)this, nullptr, buffer, offset, index);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setVertexBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger stride, NS::UInteger index)
{
    _MTL_msg_v_setVertexBuffer_offset_attributeStride_atIndex__MTL__Bufferp_NS__UInteger_NS__UInteger_NS__UInteger((const void*)this, nullptr, buffer, offset, stride, index);
}

_MTL_INLINE void MTL::IndirectRenderCommand::drawPatches(NS::UInteger numberOfPatchControlPoints, NS::UInteger patchStart, NS::UInteger patchCount, MTL::Buffer* patchIndexBuffer, NS::UInteger patchIndexBufferOffset, NS::UInteger instanceCount, NS::UInteger baseInstance, MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger instanceStride)
{
    _MTL_msg_v_drawPatches_patchStart_patchCount_patchIndexBuffer_patchIndexBufferOffset_instanceCount_baseInstance_tessellationFactorBuffer_tessellationFactorBufferOffset_tessellationFactorBufferInstanceStride__NS__UInteger_NS__UInteger_NS__UInteger_MTL__Bufferp_NS__UInteger_NS__UInteger_NS__UInteger_MTL__Bufferp_NS__UInteger_NS__UInteger((const void*)this, nullptr, numberOfPatchControlPoints, patchStart, patchCount, patchIndexBuffer, patchIndexBufferOffset, instanceCount, baseInstance, buffer, offset, instanceStride);
}

_MTL_INLINE void MTL::IndirectRenderCommand::drawIndexedPatches(NS::UInteger numberOfPatchControlPoints, NS::UInteger patchStart, NS::UInteger patchCount, MTL::Buffer* patchIndexBuffer, NS::UInteger patchIndexBufferOffset, MTL::Buffer* controlPointIndexBuffer, NS::UInteger controlPointIndexBufferOffset, NS::UInteger instanceCount, NS::UInteger baseInstance, MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger instanceStride)
{
    _MTL_msg_v_drawIndexedPatches_patchStart_patchCount_patchIndexBuffer_patchIndexBufferOffset_controlPointIndexBuffer_controlPointIndexBufferOffset_instanceCount_baseInstance_tessellationFactorBuffer_tessellationFactorBufferOffset_tessellationFactorBufferInstanceStride__NS__UInteger_NS__UInteger_NS__UInteger_MTL__Bufferp_NS__UInteger_MTL__Bufferp_NS__UInteger_NS__UInteger_NS__UInteger_MTL__Bufferp_NS__UInteger_NS__UInteger((const void*)this, nullptr, numberOfPatchControlPoints, patchStart, patchCount, patchIndexBuffer, patchIndexBufferOffset, controlPointIndexBuffer, controlPointIndexBufferOffset, instanceCount, baseInstance, buffer, offset, instanceStride);
}

_MTL_INLINE void MTL::IndirectRenderCommand::drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount, NS::UInteger instanceCount, NS::UInteger baseInstance)
{
    _MTL_msg_v_drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance__MTL__PrimitiveType_NS__UInteger_NS__UInteger_NS__UInteger_NS__UInteger((const void*)this, nullptr, primitiveType, vertexStart, vertexCount, instanceCount, baseInstance);
}

_MTL_INLINE void MTL::IndirectRenderCommand::drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, MTL::Buffer* indexBuffer, NS::UInteger indexBufferOffset, NS::UInteger instanceCount, NS::Integer baseVertex, NS::UInteger baseInstance)
{
    _MTL_msg_v_drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance__MTL__PrimitiveType_NS__UInteger_MTL__IndexType_MTL__Bufferp_NS__UInteger_NS__UInteger_NS__Integer_NS__UInteger((const void*)this, nullptr, primitiveType, indexCount, indexType, indexBuffer, indexBufferOffset, instanceCount, baseVertex, baseInstance);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setObjectThreadgroupMemoryLength(NS::UInteger length, NS::UInteger index)
{
    _MTL_msg_v_setObjectThreadgroupMemoryLength_atIndex__NS__UInteger_NS__UInteger((const void*)this, nullptr, length, index);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setObjectBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    _MTL_msg_v_setObjectBuffer_offset_atIndex__MTL__Bufferp_NS__UInteger_NS__UInteger((const void*)this, nullptr, buffer, offset, index);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setMeshBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    _MTL_msg_v_setMeshBuffer_offset_atIndex__MTL__Bufferp_NS__UInteger_NS__UInteger((const void*)this, nullptr, buffer, offset, index);
}

_MTL_INLINE void MTL::IndirectRenderCommand::drawMeshThreadgroups(MTL::Size threadgroupsPerGrid, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup)
{
    _MTL_msg_v_drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup__MTL__Size_MTL__Size_MTL__Size((const void*)this, nullptr, threadgroupsPerGrid, threadsPerObjectThreadgroup, threadsPerMeshThreadgroup);
}

_MTL_INLINE void MTL::IndirectRenderCommand::drawMeshThreads(MTL::Size threadsPerGrid, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup)
{
    _MTL_msg_v_drawMeshThreads_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup__MTL__Size_MTL__Size_MTL__Size((const void*)this, nullptr, threadsPerGrid, threadsPerObjectThreadgroup, threadsPerMeshThreadgroup);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setBarrier()
{
    _MTL_msg_v_setBarrier((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectRenderCommand::clearBarrier()
{
    _MTL_msg_v_clearBarrier((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setDepthStencilState(MTL::DepthStencilState* depthStencilState)
{
    _MTL_msg_v_setDepthStencilState__MTL__DepthStencilStatep((const void*)this, nullptr, depthStencilState);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setDepthBias(float depthBias, float slopeScale, float clamp)
{
    _MTL_msg_v_setDepthBias_slopeScale_clamp__float_float_float((const void*)this, nullptr, depthBias, slopeScale, clamp);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setDepthClipMode(MTL::DepthClipMode depthClipMode)
{
    _MTL_msg_v_setDepthClipMode__MTL__DepthClipMode((const void*)this, nullptr, depthClipMode);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setCullMode(MTL::CullMode cullMode)
{
    _MTL_msg_v_setCullMode__MTL__CullMode((const void*)this, nullptr, cullMode);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setFrontFacingWinding(MTL::Winding frontFacingWindning)
{
    _MTL_msg_v_setFrontFacingWinding__MTL__Winding((const void*)this, nullptr, frontFacingWindning);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setTriangleFillMode(MTL::TriangleFillMode fillMode)
{
    _MTL_msg_v_setTriangleFillMode__MTL__TriangleFillMode((const void*)this, nullptr, fillMode);
}

_MTL_INLINE void MTL::IndirectRenderCommand::reset()
{
    _MTL_msg_v_reset((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectComputeCommand::setComputePipelineState(MTL::ComputePipelineState* pipelineState)
{
    _MTL_msg_v_setComputePipelineState__MTL__ComputePipelineStatep((const void*)this, nullptr, pipelineState);
}

_MTL_INLINE void MTL::IndirectComputeCommand::setKernelBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    _MTL_msg_v_setKernelBuffer_offset_atIndex__MTL__Bufferp_NS__UInteger_NS__UInteger((const void*)this, nullptr, buffer, offset, index);
}

_MTL_INLINE void MTL::IndirectComputeCommand::setKernelBuffer(MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger stride, NS::UInteger index)
{
    _MTL_msg_v_setKernelBuffer_offset_attributeStride_atIndex__MTL__Bufferp_NS__UInteger_NS__UInteger_NS__UInteger((const void*)this, nullptr, buffer, offset, stride, index);
}

_MTL_INLINE void MTL::IndirectComputeCommand::concurrentDispatchThreadgroups(MTL::Size threadgroupsPerGrid, MTL::Size threadsPerThreadgroup)
{
    _MTL_msg_v_concurrentDispatchThreadgroups_threadsPerThreadgroup__MTL__Size_MTL__Size((const void*)this, nullptr, threadgroupsPerGrid, threadsPerThreadgroup);
}

_MTL_INLINE void MTL::IndirectComputeCommand::concurrentDispatchThreads(MTL::Size threadsPerGrid, MTL::Size threadsPerThreadgroup)
{
    _MTL_msg_v_concurrentDispatchThreads_threadsPerThreadgroup__MTL__Size_MTL__Size((const void*)this, nullptr, threadsPerGrid, threadsPerThreadgroup);
}

_MTL_INLINE void MTL::IndirectComputeCommand::setBarrier()
{
    _MTL_msg_v_setBarrier((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectComputeCommand::clearBarrier()
{
    _MTL_msg_v_clearBarrier((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectComputeCommand::setImageblockWidth(NS::UInteger width, NS::UInteger height)
{
    _MTL_msg_v_setImageblockWidth_height__NS__UInteger_NS__UInteger((const void*)this, nullptr, width, height);
}

_MTL_INLINE void MTL::IndirectComputeCommand::reset()
{
    _MTL_msg_v_reset((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectComputeCommand::setThreadgroupMemoryLength(NS::UInteger length, NS::UInteger index)
{
    _MTL_msg_v_setThreadgroupMemoryLength_atIndex__NS__UInteger_NS__UInteger((const void*)this, nullptr, length, index);
}

_MTL_INLINE void MTL::IndirectComputeCommand::setStageInRegion(MTL::Region region)
{
    _MTL_msg_v_setStageInRegion__MTL__Region((const void*)this, nullptr, region);
}
