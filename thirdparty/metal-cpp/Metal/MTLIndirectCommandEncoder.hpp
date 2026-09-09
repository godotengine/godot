//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLIndirectCommandEncoder.hpp
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
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include "MTLRenderCommandEncoder.hpp"
#include "MTLTypes.hpp"

namespace MTL
{
class Buffer;
class ComputePipelineState;
class RenderPipelineState;

class IndirectRenderCommand : public NS::Referencing<IndirectRenderCommand>
{
public:
    void clearBarrier();

    void drawIndexedPatches(NS::UInteger numberOfPatchControlPoints, NS::UInteger patchStart, NS::UInteger patchCount, const MTL::Buffer* patchIndexBuffer, NS::UInteger patchIndexBufferOffset, const MTL::Buffer* controlPointIndexBuffer, NS::UInteger controlPointIndexBufferOffset, NS::UInteger instanceCount, NS::UInteger baseInstance, const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger instanceStride);

    void drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, const MTL::Buffer* indexBuffer, NS::UInteger indexBufferOffset, NS::UInteger instanceCount, NS::Integer baseVertex, NS::UInteger baseInstance);

    void drawMeshThreadgroups(MTL::Size threadgroupsPerGrid, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup);

    void drawMeshThreads(MTL::Size threadsPerGrid, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup);

    void drawPatches(NS::UInteger numberOfPatchControlPoints, NS::UInteger patchStart, NS::UInteger patchCount, const MTL::Buffer* patchIndexBuffer, NS::UInteger patchIndexBufferOffset, NS::UInteger instanceCount, NS::UInteger baseInstance, const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger instanceStride);

    void drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount, NS::UInteger instanceCount, NS::UInteger baseInstance);

    void reset();

    void setBarrier();

    void setCullMode(MTL::CullMode cullMode);

    void setDepthBias(float depthBias, float slopeScale, float clamp);

    void setDepthClipMode(MTL::DepthClipMode depthClipMode);

    void setDepthStencilState(const MTL::DepthStencilState* depthStencilState);

    void setFragmentBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);

    void setFrontFacingWinding(MTL::Winding frontFacingWindning);

    void setMeshBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);

    void setObjectBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);

    void setObjectThreadgroupMemoryLength(NS::UInteger length, NS::UInteger index);

    void setRenderPipelineState(const MTL::RenderPipelineState* pipelineState);

    void setTriangleFillMode(MTL::TriangleFillMode fillMode);

    void setVertexBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);
    void setVertexBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger stride, NS::UInteger index);
};
class IndirectComputeCommand : public NS::Referencing<IndirectComputeCommand>
{
public:
    void clearBarrier();

    void concurrentDispatchThreadgroups(MTL::Size threadgroupsPerGrid, MTL::Size threadsPerThreadgroup);

    void concurrentDispatchThreads(MTL::Size threadsPerGrid, MTL::Size threadsPerThreadgroup);

    void reset();

    void setBarrier();

    void setComputePipelineState(const MTL::ComputePipelineState* pipelineState);

    void setImageblockWidth(NS::UInteger width, NS::UInteger height);

    void setKernelBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index);
    void setKernelBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger stride, NS::UInteger index);

    void setStageInRegion(MTL::Region region);

    void setThreadgroupMemoryLength(NS::UInteger length, NS::UInteger index);
};

}
_MTL_INLINE void MTL::IndirectRenderCommand::clearBarrier()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(clearBarrier));
}

_MTL_INLINE void MTL::IndirectRenderCommand::drawIndexedPatches(NS::UInteger numberOfPatchControlPoints, NS::UInteger patchStart, NS::UInteger patchCount, const MTL::Buffer* patchIndexBuffer, NS::UInteger patchIndexBufferOffset, const MTL::Buffer* controlPointIndexBuffer, NS::UInteger controlPointIndexBufferOffset, NS::UInteger instanceCount, NS::UInteger baseInstance, const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger instanceStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawIndexedPatches_patchStart_patchCount_patchIndexBuffer_patchIndexBufferOffset_controlPointIndexBuffer_controlPointIndexBufferOffset_instanceCount_baseInstance_tessellationFactorBuffer_tessellationFactorBufferOffset_tessellationFactorBufferInstanceStride_), numberOfPatchControlPoints, patchStart, patchCount, patchIndexBuffer, patchIndexBufferOffset, controlPointIndexBuffer, controlPointIndexBufferOffset, instanceCount, baseInstance, buffer, offset, instanceStride);
}

_MTL_INLINE void MTL::IndirectRenderCommand::drawIndexedPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger indexCount, MTL::IndexType indexType, const MTL::Buffer* indexBuffer, NS::UInteger indexBufferOffset, NS::UInteger instanceCount, NS::Integer baseVertex, NS::UInteger baseInstance)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance_), primitiveType, indexCount, indexType, indexBuffer, indexBufferOffset, instanceCount, baseVertex, baseInstance);
}

_MTL_INLINE void MTL::IndirectRenderCommand::drawMeshThreadgroups(MTL::Size threadgroupsPerGrid, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup_), threadgroupsPerGrid, threadsPerObjectThreadgroup, threadsPerMeshThreadgroup);
}

_MTL_INLINE void MTL::IndirectRenderCommand::drawMeshThreads(MTL::Size threadsPerGrid, MTL::Size threadsPerObjectThreadgroup, MTL::Size threadsPerMeshThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawMeshThreads_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup_), threadsPerGrid, threadsPerObjectThreadgroup, threadsPerMeshThreadgroup);
}

_MTL_INLINE void MTL::IndirectRenderCommand::drawPatches(NS::UInteger numberOfPatchControlPoints, NS::UInteger patchStart, NS::UInteger patchCount, const MTL::Buffer* patchIndexBuffer, NS::UInteger patchIndexBufferOffset, NS::UInteger instanceCount, NS::UInteger baseInstance, const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger instanceStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawPatches_patchStart_patchCount_patchIndexBuffer_patchIndexBufferOffset_instanceCount_baseInstance_tessellationFactorBuffer_tessellationFactorBufferOffset_tessellationFactorBufferInstanceStride_), numberOfPatchControlPoints, patchStart, patchCount, patchIndexBuffer, patchIndexBufferOffset, instanceCount, baseInstance, buffer, offset, instanceStride);
}

_MTL_INLINE void MTL::IndirectRenderCommand::drawPrimitives(MTL::PrimitiveType primitiveType, NS::UInteger vertexStart, NS::UInteger vertexCount, NS::UInteger instanceCount, NS::UInteger baseInstance)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance_), primitiveType, vertexStart, vertexCount, instanceCount, baseInstance);
}

_MTL_INLINE void MTL::IndirectRenderCommand::reset()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(reset));
}

_MTL_INLINE void MTL::IndirectRenderCommand::setBarrier()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBarrier));
}

_MTL_INLINE void MTL::IndirectRenderCommand::setCullMode(MTL::CullMode cullMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCullMode_), cullMode);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setDepthBias(float depthBias, float slopeScale, float clamp)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthBias_slopeScale_clamp_), depthBias, slopeScale, clamp);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setDepthClipMode(MTL::DepthClipMode depthClipMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthClipMode_), depthClipMode);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setDepthStencilState(const MTL::DepthStencilState* depthStencilState)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDepthStencilState_), depthStencilState);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setFragmentBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentBuffer_offset_atIndex_), buffer, offset, index);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setFrontFacingWinding(MTL::Winding frontFacingWindning)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFrontFacingWinding_), frontFacingWindning);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setMeshBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMeshBuffer_offset_atIndex_), buffer, offset, index);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setObjectBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObjectBuffer_offset_atIndex_), buffer, offset, index);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setObjectThreadgroupMemoryLength(NS::UInteger length, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObjectThreadgroupMemoryLength_atIndex_), length, index);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setRenderPipelineState(const MTL::RenderPipelineState* pipelineState)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRenderPipelineState_), pipelineState);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setTriangleFillMode(MTL::TriangleFillMode fillMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTriangleFillMode_), fillMode);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setVertexBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexBuffer_offset_atIndex_), buffer, offset, index);
}

_MTL_INLINE void MTL::IndirectRenderCommand::setVertexBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger stride, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexBuffer_offset_attributeStride_atIndex_), buffer, offset, stride, index);
}

_MTL_INLINE void MTL::IndirectComputeCommand::clearBarrier()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(clearBarrier));
}

_MTL_INLINE void MTL::IndirectComputeCommand::concurrentDispatchThreadgroups(MTL::Size threadgroupsPerGrid, MTL::Size threadsPerThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(concurrentDispatchThreadgroups_threadsPerThreadgroup_), threadgroupsPerGrid, threadsPerThreadgroup);
}

_MTL_INLINE void MTL::IndirectComputeCommand::concurrentDispatchThreads(MTL::Size threadsPerGrid, MTL::Size threadsPerThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(concurrentDispatchThreads_threadsPerThreadgroup_), threadsPerGrid, threadsPerThreadgroup);
}

_MTL_INLINE void MTL::IndirectComputeCommand::reset()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(reset));
}

_MTL_INLINE void MTL::IndirectComputeCommand::setBarrier()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBarrier));
}

_MTL_INLINE void MTL::IndirectComputeCommand::setComputePipelineState(const MTL::ComputePipelineState* pipelineState)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setComputePipelineState_), pipelineState);
}

_MTL_INLINE void MTL::IndirectComputeCommand::setImageblockWidth(NS::UInteger width, NS::UInteger height)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setImageblockWidth_height_), width, height);
}

_MTL_INLINE void MTL::IndirectComputeCommand::setKernelBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setKernelBuffer_offset_atIndex_), buffer, offset, index);
}

_MTL_INLINE void MTL::IndirectComputeCommand::setKernelBuffer(const MTL::Buffer* buffer, NS::UInteger offset, NS::UInteger stride, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setKernelBuffer_offset_attributeStride_atIndex_), buffer, offset, stride, index);
}

_MTL_INLINE void MTL::IndirectComputeCommand::setStageInRegion(MTL::Region region)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStageInRegion_), region);
}

_MTL_INLINE void MTL::IndirectComputeCommand::setThreadgroupMemoryLength(NS::UInteger length, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setThreadgroupMemoryLength_atIndex_), length, index);
}
