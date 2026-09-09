//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLIndirectCommandBuffer.hpp
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
#include "MTLResource.hpp"
#include "MTLTypes.hpp"
#include <cstdint>

namespace MTL
{
class IndirectCommandBufferDescriptor;
class IndirectComputeCommand;
class IndirectRenderCommand;

_MTL_OPTIONS(NS::UInteger, IndirectCommandType) {
    IndirectCommandTypeDraw = 1,
    IndirectCommandTypeDrawIndexed = 1 << 1,
    IndirectCommandTypeDrawPatches = 1 << 2,
    IndirectCommandTypeDrawIndexedPatches = 1 << 3,
    IndirectCommandTypeConcurrentDispatch = 1 << 5,
    IndirectCommandTypeConcurrentDispatchThreads = 1 << 6,
    IndirectCommandTypeDrawMeshThreadgroups = 1 << 7,
    IndirectCommandTypeDrawMeshThreads = 1 << 8,
};

struct IndirectCommandBufferExecutionRange
{
    uint32_t location;
    uint32_t length;
} _MTL_PACKED;

class IndirectCommandBufferDescriptor : public NS::Copying<IndirectCommandBufferDescriptor>
{
public:
    static IndirectCommandBufferDescriptor* alloc();

    IndirectCommandType                     commandTypes() const;

    bool                                    inheritBuffers() const;

    bool                                    inheritCullMode() const;

    bool                                    inheritDepthBias() const;

    bool                                    inheritDepthClipMode() const;

    bool                                    inheritDepthStencilState() const;

    bool                                    inheritFrontFacingWinding() const;

    bool                                    inheritPipelineState() const;

    bool                                    inheritTriangleFillMode() const;

    IndirectCommandBufferDescriptor*        init();

    NS::UInteger                            maxFragmentBufferBindCount() const;

    NS::UInteger                            maxKernelBufferBindCount() const;

    NS::UInteger                            maxKernelThreadgroupMemoryBindCount() const;

    NS::UInteger                            maxMeshBufferBindCount() const;

    NS::UInteger                            maxObjectBufferBindCount() const;

    NS::UInteger                            maxObjectThreadgroupMemoryBindCount() const;

    NS::UInteger                            maxVertexBufferBindCount() const;

    void                                    setCommandTypes(MTL::IndirectCommandType commandTypes);

    void                                    setInheritBuffers(bool inheritBuffers);

    void                                    setInheritCullMode(bool inheritCullMode);

    void                                    setInheritDepthBias(bool inheritDepthBias);

    void                                    setInheritDepthClipMode(bool inheritDepthClipMode);

    void                                    setInheritDepthStencilState(bool inheritDepthStencilState);

    void                                    setInheritFrontFacingWinding(bool inheritFrontFacingWinding);

    void                                    setInheritPipelineState(bool inheritPipelineState);

    void                                    setInheritTriangleFillMode(bool inheritTriangleFillMode);

    void                                    setMaxFragmentBufferBindCount(NS::UInteger maxFragmentBufferBindCount);

    void                                    setMaxKernelBufferBindCount(NS::UInteger maxKernelBufferBindCount);

    void                                    setMaxKernelThreadgroupMemoryBindCount(NS::UInteger maxKernelThreadgroupMemoryBindCount);

    void                                    setMaxMeshBufferBindCount(NS::UInteger maxMeshBufferBindCount);

    void                                    setMaxObjectBufferBindCount(NS::UInteger maxObjectBufferBindCount);

    void                                    setMaxObjectThreadgroupMemoryBindCount(NS::UInteger maxObjectThreadgroupMemoryBindCount);

    void                                    setMaxVertexBufferBindCount(NS::UInteger maxVertexBufferBindCount);

    void                                    setSupportColorAttachmentMapping(bool supportColorAttachmentMapping);

    void                                    setSupportDynamicAttributeStride(bool supportDynamicAttributeStride);

    void                                    setSupportRayTracing(bool supportRayTracing);

    bool                                    supportColorAttachmentMapping() const;

    bool                                    supportDynamicAttributeStride() const;

    bool                                    supportRayTracing() const;
};
class IndirectCommandBuffer : public NS::Referencing<IndirectCommandBuffer, Resource>
{
public:
    ResourceID              gpuResourceID() const;

    IndirectComputeCommand* indirectComputeCommand(NS::UInteger commandIndex);

    IndirectRenderCommand*  indirectRenderCommand(NS::UInteger commandIndex);

    void                    reset(NS::Range range);

    NS::UInteger            size() const;
};

}

_MTL_INLINE MTL::IndirectCommandBufferDescriptor* MTL::IndirectCommandBufferDescriptor::alloc()
{
    return NS::Object::alloc<MTL::IndirectCommandBufferDescriptor>(_MTL_PRIVATE_CLS(MTLIndirectCommandBufferDescriptor));
}

_MTL_INLINE MTL::IndirectCommandType MTL::IndirectCommandBufferDescriptor::commandTypes() const
{
    return Object::sendMessage<MTL::IndirectCommandType>(this, _MTL_PRIVATE_SEL(commandTypes));
}

_MTL_INLINE bool MTL::IndirectCommandBufferDescriptor::inheritBuffers() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(inheritBuffers));
}

_MTL_INLINE bool MTL::IndirectCommandBufferDescriptor::inheritCullMode() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(inheritCullMode));
}

_MTL_INLINE bool MTL::IndirectCommandBufferDescriptor::inheritDepthBias() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(inheritDepthBias));
}

_MTL_INLINE bool MTL::IndirectCommandBufferDescriptor::inheritDepthClipMode() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(inheritDepthClipMode));
}

_MTL_INLINE bool MTL::IndirectCommandBufferDescriptor::inheritDepthStencilState() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(inheritDepthStencilState));
}

_MTL_INLINE bool MTL::IndirectCommandBufferDescriptor::inheritFrontFacingWinding() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(inheritFrontFacingWinding));
}

_MTL_INLINE bool MTL::IndirectCommandBufferDescriptor::inheritPipelineState() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(inheritPipelineState));
}

_MTL_INLINE bool MTL::IndirectCommandBufferDescriptor::inheritTriangleFillMode() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(inheritTriangleFillMode));
}

_MTL_INLINE MTL::IndirectCommandBufferDescriptor* MTL::IndirectCommandBufferDescriptor::init()
{
    return NS::Object::init<MTL::IndirectCommandBufferDescriptor>();
}

_MTL_INLINE NS::UInteger MTL::IndirectCommandBufferDescriptor::maxFragmentBufferBindCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxFragmentBufferBindCount));
}

_MTL_INLINE NS::UInteger MTL::IndirectCommandBufferDescriptor::maxKernelBufferBindCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxKernelBufferBindCount));
}

_MTL_INLINE NS::UInteger MTL::IndirectCommandBufferDescriptor::maxKernelThreadgroupMemoryBindCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxKernelThreadgroupMemoryBindCount));
}

_MTL_INLINE NS::UInteger MTL::IndirectCommandBufferDescriptor::maxMeshBufferBindCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxMeshBufferBindCount));
}

_MTL_INLINE NS::UInteger MTL::IndirectCommandBufferDescriptor::maxObjectBufferBindCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxObjectBufferBindCount));
}

_MTL_INLINE NS::UInteger MTL::IndirectCommandBufferDescriptor::maxObjectThreadgroupMemoryBindCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxObjectThreadgroupMemoryBindCount));
}

_MTL_INLINE NS::UInteger MTL::IndirectCommandBufferDescriptor::maxVertexBufferBindCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxVertexBufferBindCount));
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setCommandTypes(MTL::IndirectCommandType commandTypes)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCommandTypes_), commandTypes);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setInheritBuffers(bool inheritBuffers)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInheritBuffers_), inheritBuffers);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setInheritCullMode(bool inheritCullMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInheritCullMode_), inheritCullMode);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setInheritDepthBias(bool inheritDepthBias)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInheritDepthBias_), inheritDepthBias);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setInheritDepthClipMode(bool inheritDepthClipMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInheritDepthClipMode_), inheritDepthClipMode);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setInheritDepthStencilState(bool inheritDepthStencilState)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInheritDepthStencilState_), inheritDepthStencilState);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setInheritFrontFacingWinding(bool inheritFrontFacingWinding)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInheritFrontFacingWinding_), inheritFrontFacingWinding);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setInheritPipelineState(bool inheritPipelineState)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInheritPipelineState_), inheritPipelineState);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setInheritTriangleFillMode(bool inheritTriangleFillMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInheritTriangleFillMode_), inheritTriangleFillMode);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setMaxFragmentBufferBindCount(NS::UInteger maxFragmentBufferBindCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxFragmentBufferBindCount_), maxFragmentBufferBindCount);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setMaxKernelBufferBindCount(NS::UInteger maxKernelBufferBindCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxKernelBufferBindCount_), maxKernelBufferBindCount);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setMaxKernelThreadgroupMemoryBindCount(NS::UInteger maxKernelThreadgroupMemoryBindCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxKernelThreadgroupMemoryBindCount_), maxKernelThreadgroupMemoryBindCount);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setMaxMeshBufferBindCount(NS::UInteger maxMeshBufferBindCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxMeshBufferBindCount_), maxMeshBufferBindCount);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setMaxObjectBufferBindCount(NS::UInteger maxObjectBufferBindCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxObjectBufferBindCount_), maxObjectBufferBindCount);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setMaxObjectThreadgroupMemoryBindCount(NS::UInteger maxObjectThreadgroupMemoryBindCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxObjectThreadgroupMemoryBindCount_), maxObjectThreadgroupMemoryBindCount);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setMaxVertexBufferBindCount(NS::UInteger maxVertexBufferBindCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxVertexBufferBindCount_), maxVertexBufferBindCount);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setSupportColorAttachmentMapping(bool supportColorAttachmentMapping)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSupportColorAttachmentMapping_), supportColorAttachmentMapping);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setSupportDynamicAttributeStride(bool supportDynamicAttributeStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSupportDynamicAttributeStride_), supportDynamicAttributeStride);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setSupportRayTracing(bool supportRayTracing)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSupportRayTracing_), supportRayTracing);
}

_MTL_INLINE bool MTL::IndirectCommandBufferDescriptor::supportColorAttachmentMapping() const
{
    return Object::sendMessageSafe<bool>(this, _MTL_PRIVATE_SEL(supportColorAttachmentMapping));
}

_MTL_INLINE bool MTL::IndirectCommandBufferDescriptor::supportDynamicAttributeStride() const
{
    return Object::sendMessageSafe<bool>(this, _MTL_PRIVATE_SEL(supportDynamicAttributeStride));
}

_MTL_INLINE bool MTL::IndirectCommandBufferDescriptor::supportRayTracing() const
{
    return Object::sendMessageSafe<bool>(this, _MTL_PRIVATE_SEL(supportRayTracing));
}

_MTL_INLINE MTL::ResourceID MTL::IndirectCommandBuffer::gpuResourceID() const
{
    return Object::sendMessage<MTL::ResourceID>(this, _MTL_PRIVATE_SEL(gpuResourceID));
}

_MTL_INLINE MTL::IndirectComputeCommand* MTL::IndirectCommandBuffer::indirectComputeCommand(NS::UInteger commandIndex)
{
    return Object::sendMessage<MTL::IndirectComputeCommand*>(this, _MTL_PRIVATE_SEL(indirectComputeCommandAtIndex_), commandIndex);
}

_MTL_INLINE MTL::IndirectRenderCommand* MTL::IndirectCommandBuffer::indirectRenderCommand(NS::UInteger commandIndex)
{
    return Object::sendMessage<MTL::IndirectRenderCommand*>(this, _MTL_PRIVATE_SEL(indirectRenderCommandAtIndex_), commandIndex);
}

_MTL_INLINE void MTL::IndirectCommandBuffer::reset(NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(resetWithRange_), range);
}

_MTL_INLINE NS::UInteger MTL::IndirectCommandBuffer::size() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(size));
}
