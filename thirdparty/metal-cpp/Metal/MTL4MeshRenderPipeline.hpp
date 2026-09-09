//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTL4MeshRenderPipeline.hpp
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
#include "MTL4PipelineState.hpp"
#include "MTL4RenderPipeline.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include "MTLTypes.hpp"

namespace MTL4
{
class FunctionDescriptor;
class MeshRenderPipelineDescriptor;
class RenderPipelineColorAttachmentDescriptorArray;
class StaticLinkingDescriptor;

class MeshRenderPipelineDescriptor : public NS::Copying<MeshRenderPipelineDescriptor, PipelineDescriptor>
{
public:
    static MeshRenderPipelineDescriptor*          alloc();

    AlphaToCoverageState                          alphaToCoverageState() const;

    AlphaToOneState                               alphaToOneState() const;

    LogicalToPhysicalColorAttachmentMappingState  colorAttachmentMappingState() const;

    RenderPipelineColorAttachmentDescriptorArray* colorAttachments() const;

    FunctionDescriptor*                           fragmentFunctionDescriptor() const;

    StaticLinkingDescriptor*                      fragmentStaticLinkingDescriptor() const;

    MeshRenderPipelineDescriptor*                 init();

    bool                                          isRasterizationEnabled() const;

    NS::UInteger                                  maxTotalThreadgroupsPerMeshGrid() const;

    NS::UInteger                                  maxTotalThreadsPerMeshThreadgroup() const;

    NS::UInteger                                  maxTotalThreadsPerObjectThreadgroup() const;

    NS::UInteger                                  maxVertexAmplificationCount() const;

    FunctionDescriptor*                           meshFunctionDescriptor() const;

    StaticLinkingDescriptor*                      meshStaticLinkingDescriptor() const;

    bool                                          meshThreadgroupSizeIsMultipleOfThreadExecutionWidth() const;

    FunctionDescriptor*                           objectFunctionDescriptor() const;

    StaticLinkingDescriptor*                      objectStaticLinkingDescriptor() const;

    bool                                          objectThreadgroupSizeIsMultipleOfThreadExecutionWidth() const;

    NS::UInteger                                  payloadMemoryLength() const;

    NS::UInteger                                  rasterSampleCount() const;

    [[deprecated("please use isRasterizationEnabled instead")]]
    bool                              rasterizationEnabled() const;

    MTL::Size                         requiredThreadsPerMeshThreadgroup() const;

    MTL::Size                         requiredThreadsPerObjectThreadgroup() const;

    void                              reset();

    void                              setAlphaToCoverageState(MTL4::AlphaToCoverageState alphaToCoverageState);

    void                              setAlphaToOneState(MTL4::AlphaToOneState alphaToOneState);

    void                              setColorAttachmentMappingState(MTL4::LogicalToPhysicalColorAttachmentMappingState colorAttachmentMappingState);

    void                              setFragmentFunctionDescriptor(const MTL4::FunctionDescriptor* fragmentFunctionDescriptor);

    void                              setFragmentStaticLinkingDescriptor(const MTL4::StaticLinkingDescriptor* fragmentStaticLinkingDescriptor);

    void                              setMaxTotalThreadgroupsPerMeshGrid(NS::UInteger maxTotalThreadgroupsPerMeshGrid);

    void                              setMaxTotalThreadsPerMeshThreadgroup(NS::UInteger maxTotalThreadsPerMeshThreadgroup);

    void                              setMaxTotalThreadsPerObjectThreadgroup(NS::UInteger maxTotalThreadsPerObjectThreadgroup);

    void                              setMaxVertexAmplificationCount(NS::UInteger maxVertexAmplificationCount);

    void                              setMeshFunctionDescriptor(const MTL4::FunctionDescriptor* meshFunctionDescriptor);

    void                              setMeshStaticLinkingDescriptor(const MTL4::StaticLinkingDescriptor* meshStaticLinkingDescriptor);

    void                              setMeshThreadgroupSizeIsMultipleOfThreadExecutionWidth(bool meshThreadgroupSizeIsMultipleOfThreadExecutionWidth);

    void                              setObjectFunctionDescriptor(const MTL4::FunctionDescriptor* objectFunctionDescriptor);

    void                              setObjectStaticLinkingDescriptor(const MTL4::StaticLinkingDescriptor* objectStaticLinkingDescriptor);

    void                              setObjectThreadgroupSizeIsMultipleOfThreadExecutionWidth(bool objectThreadgroupSizeIsMultipleOfThreadExecutionWidth);

    void                              setPayloadMemoryLength(NS::UInteger payloadMemoryLength);

    void                              setRasterSampleCount(NS::UInteger rasterSampleCount);

    void                              setRasterizationEnabled(bool rasterizationEnabled);

    void                              setRequiredThreadsPerMeshThreadgroup(MTL::Size requiredThreadsPerMeshThreadgroup);

    void                              setRequiredThreadsPerObjectThreadgroup(MTL::Size requiredThreadsPerObjectThreadgroup);

    void                              setSupportFragmentBinaryLinking(bool supportFragmentBinaryLinking);

    void                              setSupportIndirectCommandBuffers(MTL4::IndirectCommandBufferSupportState supportIndirectCommandBuffers);

    void                              setSupportMeshBinaryLinking(bool supportMeshBinaryLinking);

    void                              setSupportObjectBinaryLinking(bool supportObjectBinaryLinking);

    bool                              supportFragmentBinaryLinking() const;

    IndirectCommandBufferSupportState supportIndirectCommandBuffers() const;

    bool                              supportMeshBinaryLinking() const;

    bool                              supportObjectBinaryLinking() const;
};

}
_MTL_INLINE MTL4::MeshRenderPipelineDescriptor* MTL4::MeshRenderPipelineDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::MeshRenderPipelineDescriptor>(_MTL_PRIVATE_CLS(MTL4MeshRenderPipelineDescriptor));
}

_MTL_INLINE MTL4::AlphaToCoverageState MTL4::MeshRenderPipelineDescriptor::alphaToCoverageState() const
{
    return Object::sendMessage<MTL4::AlphaToCoverageState>(this, _MTL_PRIVATE_SEL(alphaToCoverageState));
}

_MTL_INLINE MTL4::AlphaToOneState MTL4::MeshRenderPipelineDescriptor::alphaToOneState() const
{
    return Object::sendMessage<MTL4::AlphaToOneState>(this, _MTL_PRIVATE_SEL(alphaToOneState));
}

_MTL_INLINE MTL4::LogicalToPhysicalColorAttachmentMappingState MTL4::MeshRenderPipelineDescriptor::colorAttachmentMappingState() const
{
    return Object::sendMessage<MTL4::LogicalToPhysicalColorAttachmentMappingState>(this, _MTL_PRIVATE_SEL(colorAttachmentMappingState));
}

_MTL_INLINE MTL4::RenderPipelineColorAttachmentDescriptorArray* MTL4::MeshRenderPipelineDescriptor::colorAttachments() const
{
    return Object::sendMessage<MTL4::RenderPipelineColorAttachmentDescriptorArray*>(this, _MTL_PRIVATE_SEL(colorAttachments));
}

_MTL_INLINE MTL4::FunctionDescriptor* MTL4::MeshRenderPipelineDescriptor::fragmentFunctionDescriptor() const
{
    return Object::sendMessage<MTL4::FunctionDescriptor*>(this, _MTL_PRIVATE_SEL(fragmentFunctionDescriptor));
}

_MTL_INLINE MTL4::StaticLinkingDescriptor* MTL4::MeshRenderPipelineDescriptor::fragmentStaticLinkingDescriptor() const
{
    return Object::sendMessage<MTL4::StaticLinkingDescriptor*>(this, _MTL_PRIVATE_SEL(fragmentStaticLinkingDescriptor));
}

_MTL_INLINE MTL4::MeshRenderPipelineDescriptor* MTL4::MeshRenderPipelineDescriptor::init()
{
    return NS::Object::init<MTL4::MeshRenderPipelineDescriptor>();
}

_MTL_INLINE bool MTL4::MeshRenderPipelineDescriptor::isRasterizationEnabled() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isRasterizationEnabled));
}

_MTL_INLINE NS::UInteger MTL4::MeshRenderPipelineDescriptor::maxTotalThreadgroupsPerMeshGrid() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxTotalThreadgroupsPerMeshGrid));
}

_MTL_INLINE NS::UInteger MTL4::MeshRenderPipelineDescriptor::maxTotalThreadsPerMeshThreadgroup() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxTotalThreadsPerMeshThreadgroup));
}

_MTL_INLINE NS::UInteger MTL4::MeshRenderPipelineDescriptor::maxTotalThreadsPerObjectThreadgroup() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxTotalThreadsPerObjectThreadgroup));
}

_MTL_INLINE NS::UInteger MTL4::MeshRenderPipelineDescriptor::maxVertexAmplificationCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxVertexAmplificationCount));
}

_MTL_INLINE MTL4::FunctionDescriptor* MTL4::MeshRenderPipelineDescriptor::meshFunctionDescriptor() const
{
    return Object::sendMessage<MTL4::FunctionDescriptor*>(this, _MTL_PRIVATE_SEL(meshFunctionDescriptor));
}

_MTL_INLINE MTL4::StaticLinkingDescriptor* MTL4::MeshRenderPipelineDescriptor::meshStaticLinkingDescriptor() const
{
    return Object::sendMessage<MTL4::StaticLinkingDescriptor*>(this, _MTL_PRIVATE_SEL(meshStaticLinkingDescriptor));
}

_MTL_INLINE bool MTL4::MeshRenderPipelineDescriptor::meshThreadgroupSizeIsMultipleOfThreadExecutionWidth() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(meshThreadgroupSizeIsMultipleOfThreadExecutionWidth));
}

_MTL_INLINE MTL4::FunctionDescriptor* MTL4::MeshRenderPipelineDescriptor::objectFunctionDescriptor() const
{
    return Object::sendMessage<MTL4::FunctionDescriptor*>(this, _MTL_PRIVATE_SEL(objectFunctionDescriptor));
}

_MTL_INLINE MTL4::StaticLinkingDescriptor* MTL4::MeshRenderPipelineDescriptor::objectStaticLinkingDescriptor() const
{
    return Object::sendMessage<MTL4::StaticLinkingDescriptor*>(this, _MTL_PRIVATE_SEL(objectStaticLinkingDescriptor));
}

_MTL_INLINE bool MTL4::MeshRenderPipelineDescriptor::objectThreadgroupSizeIsMultipleOfThreadExecutionWidth() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(objectThreadgroupSizeIsMultipleOfThreadExecutionWidth));
}

_MTL_INLINE NS::UInteger MTL4::MeshRenderPipelineDescriptor::payloadMemoryLength() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(payloadMemoryLength));
}

_MTL_INLINE NS::UInteger MTL4::MeshRenderPipelineDescriptor::rasterSampleCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(rasterSampleCount));
}

_MTL_INLINE bool MTL4::MeshRenderPipelineDescriptor::rasterizationEnabled() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isRasterizationEnabled));
}

_MTL_INLINE MTL::Size MTL4::MeshRenderPipelineDescriptor::requiredThreadsPerMeshThreadgroup() const
{
    return Object::sendMessage<MTL::Size>(this, _MTL_PRIVATE_SEL(requiredThreadsPerMeshThreadgroup));
}

_MTL_INLINE MTL::Size MTL4::MeshRenderPipelineDescriptor::requiredThreadsPerObjectThreadgroup() const
{
    return Object::sendMessage<MTL::Size>(this, _MTL_PRIVATE_SEL(requiredThreadsPerObjectThreadgroup));
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::reset()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(reset));
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::setAlphaToCoverageState(MTL4::AlphaToCoverageState alphaToCoverageState)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setAlphaToCoverageState_), alphaToCoverageState);
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::setAlphaToOneState(MTL4::AlphaToOneState alphaToOneState)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setAlphaToOneState_), alphaToOneState);
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::setColorAttachmentMappingState(MTL4::LogicalToPhysicalColorAttachmentMappingState colorAttachmentMappingState)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setColorAttachmentMappingState_), colorAttachmentMappingState);
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::setFragmentFunctionDescriptor(const MTL4::FunctionDescriptor* fragmentFunctionDescriptor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentFunctionDescriptor_), fragmentFunctionDescriptor);
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::setFragmentStaticLinkingDescriptor(const MTL4::StaticLinkingDescriptor* fragmentStaticLinkingDescriptor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFragmentStaticLinkingDescriptor_), fragmentStaticLinkingDescriptor);
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::setMaxTotalThreadgroupsPerMeshGrid(NS::UInteger maxTotalThreadgroupsPerMeshGrid)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxTotalThreadgroupsPerMeshGrid_), maxTotalThreadgroupsPerMeshGrid);
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::setMaxTotalThreadsPerMeshThreadgroup(NS::UInteger maxTotalThreadsPerMeshThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxTotalThreadsPerMeshThreadgroup_), maxTotalThreadsPerMeshThreadgroup);
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::setMaxTotalThreadsPerObjectThreadgroup(NS::UInteger maxTotalThreadsPerObjectThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxTotalThreadsPerObjectThreadgroup_), maxTotalThreadsPerObjectThreadgroup);
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::setMaxVertexAmplificationCount(NS::UInteger maxVertexAmplificationCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxVertexAmplificationCount_), maxVertexAmplificationCount);
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::setMeshFunctionDescriptor(const MTL4::FunctionDescriptor* meshFunctionDescriptor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMeshFunctionDescriptor_), meshFunctionDescriptor);
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::setMeshStaticLinkingDescriptor(const MTL4::StaticLinkingDescriptor* meshStaticLinkingDescriptor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMeshStaticLinkingDescriptor_), meshStaticLinkingDescriptor);
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::setMeshThreadgroupSizeIsMultipleOfThreadExecutionWidth(bool meshThreadgroupSizeIsMultipleOfThreadExecutionWidth)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMeshThreadgroupSizeIsMultipleOfThreadExecutionWidth_), meshThreadgroupSizeIsMultipleOfThreadExecutionWidth);
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::setObjectFunctionDescriptor(const MTL4::FunctionDescriptor* objectFunctionDescriptor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObjectFunctionDescriptor_), objectFunctionDescriptor);
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::setObjectStaticLinkingDescriptor(const MTL4::StaticLinkingDescriptor* objectStaticLinkingDescriptor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObjectStaticLinkingDescriptor_), objectStaticLinkingDescriptor);
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::setObjectThreadgroupSizeIsMultipleOfThreadExecutionWidth(bool objectThreadgroupSizeIsMultipleOfThreadExecutionWidth)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObjectThreadgroupSizeIsMultipleOfThreadExecutionWidth_), objectThreadgroupSizeIsMultipleOfThreadExecutionWidth);
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::setPayloadMemoryLength(NS::UInteger payloadMemoryLength)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPayloadMemoryLength_), payloadMemoryLength);
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::setRasterSampleCount(NS::UInteger rasterSampleCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRasterSampleCount_), rasterSampleCount);
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::setRasterizationEnabled(bool rasterizationEnabled)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRasterizationEnabled_), rasterizationEnabled);
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::setRequiredThreadsPerMeshThreadgroup(MTL::Size requiredThreadsPerMeshThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRequiredThreadsPerMeshThreadgroup_), requiredThreadsPerMeshThreadgroup);
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::setRequiredThreadsPerObjectThreadgroup(MTL::Size requiredThreadsPerObjectThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRequiredThreadsPerObjectThreadgroup_), requiredThreadsPerObjectThreadgroup);
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::setSupportFragmentBinaryLinking(bool supportFragmentBinaryLinking)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSupportFragmentBinaryLinking_), supportFragmentBinaryLinking);
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::setSupportIndirectCommandBuffers(MTL4::IndirectCommandBufferSupportState supportIndirectCommandBuffers)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSupportIndirectCommandBuffers_), supportIndirectCommandBuffers);
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::setSupportMeshBinaryLinking(bool supportMeshBinaryLinking)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSupportMeshBinaryLinking_), supportMeshBinaryLinking);
}

_MTL_INLINE void MTL4::MeshRenderPipelineDescriptor::setSupportObjectBinaryLinking(bool supportObjectBinaryLinking)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSupportObjectBinaryLinking_), supportObjectBinaryLinking);
}

_MTL_INLINE bool MTL4::MeshRenderPipelineDescriptor::supportFragmentBinaryLinking() const
{
    return Object::sendMessageSafe<bool>(this, _MTL_PRIVATE_SEL(supportFragmentBinaryLinking));
}

_MTL_INLINE MTL4::IndirectCommandBufferSupportState MTL4::MeshRenderPipelineDescriptor::supportIndirectCommandBuffers() const
{
    return Object::sendMessage<MTL4::IndirectCommandBufferSupportState>(this, _MTL_PRIVATE_SEL(supportIndirectCommandBuffers));
}

_MTL_INLINE bool MTL4::MeshRenderPipelineDescriptor::supportMeshBinaryLinking() const
{
    return Object::sendMessageSafe<bool>(this, _MTL_PRIVATE_SEL(supportMeshBinaryLinking));
}

_MTL_INLINE bool MTL4::MeshRenderPipelineDescriptor::supportObjectBinaryLinking() const
{
    return Object::sendMessageSafe<bool>(this, _MTL_PRIVATE_SEL(supportObjectBinaryLinking));
}
