#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTLResource.hpp"

namespace MTL {
    class IndirectComputeCommand;
    class IndirectRenderCommand;
}

namespace MTL
{

_MTL_OPTIONS(NS::UInteger, IndirectCommandType) {
    IndirectCommandTypeDraw = (1 << 0),
    IndirectCommandTypeDrawIndexed = (1 << 1),
    IndirectCommandTypeDrawPatches = (1 << 2),
    IndirectCommandTypeDrawIndexedPatches = (1 << 3),
    IndirectCommandTypeConcurrentDispatch = (1 << 5),
    IndirectCommandTypeConcurrentDispatchThreads = (1 << 6),
    IndirectCommandTypeDrawMeshThreadgroups = (1 << 7),
    IndirectCommandTypeDrawMeshThreads = (1 << 8),
};


class IndirectCommandBufferDescriptor;
class IndirectCommandBuffer;

class IndirectCommandBufferDescriptor : public NS::Copying<IndirectCommandBufferDescriptor>
{
public:
    static IndirectCommandBufferDescriptor* alloc();
    IndirectCommandBufferDescriptor*        init() const;

    MTL::IndirectCommandType commandTypes() const;
    bool                     inheritBuffers() const;
    bool                     inheritCullMode() const;
    bool                     inheritDepthBias() const;
    bool                     inheritDepthClipMode() const;
    bool                     inheritDepthStencilState() const;
    bool                     inheritFrontFacingWinding() const;
    bool                     inheritPipelineState() const;
    bool                     inheritTriangleFillMode() const;
    NS::UInteger             maxFragmentBufferBindCount() const;
    NS::UInteger             maxKernelBufferBindCount() const;
    NS::UInteger             maxKernelThreadgroupMemoryBindCount() const;
    NS::UInteger             maxMeshBufferBindCount() const;
    NS::UInteger             maxObjectBufferBindCount() const;
    NS::UInteger             maxObjectThreadgroupMemoryBindCount() const;
    NS::UInteger             maxVertexBufferBindCount() const;
    void                     setCommandTypes(MTL::IndirectCommandType commandTypes);
    void                     setInheritBuffers(bool inheritBuffers);
    void                     setInheritCullMode(bool inheritCullMode);
    void                     setInheritDepthBias(bool inheritDepthBias);
    void                     setInheritDepthClipMode(bool inheritDepthClipMode);
    void                     setInheritDepthStencilState(bool inheritDepthStencilState);
    void                     setInheritFrontFacingWinding(bool inheritFrontFacingWinding);
    void                     setInheritPipelineState(bool inheritPipelineState);
    void                     setInheritTriangleFillMode(bool inheritTriangleFillMode);
    void                     setMaxFragmentBufferBindCount(NS::UInteger maxFragmentBufferBindCount);
    void                     setMaxKernelBufferBindCount(NS::UInteger maxKernelBufferBindCount);
    void                     setMaxKernelThreadgroupMemoryBindCount(NS::UInteger maxKernelThreadgroupMemoryBindCount);
    void                     setMaxMeshBufferBindCount(NS::UInteger maxMeshBufferBindCount);
    void                     setMaxObjectBufferBindCount(NS::UInteger maxObjectBufferBindCount);
    void                     setMaxObjectThreadgroupMemoryBindCount(NS::UInteger maxObjectThreadgroupMemoryBindCount);
    void                     setMaxVertexBufferBindCount(NS::UInteger maxVertexBufferBindCount);
    void                     setSupportColorAttachmentMapping(bool supportColorAttachmentMapping);
    void                     setSupportDynamicAttributeStride(bool supportDynamicAttributeStride);
    void                     setSupportRayTracing(bool supportRayTracing);
    bool                     supportColorAttachmentMapping() const;
    bool                     supportDynamicAttributeStride() const;
    bool                     supportRayTracing() const;

};

class IndirectCommandBuffer : public NS::Referencing<IndirectCommandBuffer, MTL::Resource>
{
public:
    MTL::ResourceID              gpuResourceID() const;
    MTL::IndirectComputeCommand* indirectComputeCommand(NS::UInteger commandIndex);
    MTL::IndirectRenderCommand*  indirectRenderCommand(NS::UInteger commandIndex);
    void                         reset(NS::Range range);
    NS::UInteger                 size() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLIndirectCommandBufferDescriptor;
extern "C" void *OBJC_CLASS_$_MTLIndirectCommandBuffer;

_MTL_INLINE MTL::IndirectCommandBufferDescriptor* MTL::IndirectCommandBufferDescriptor::alloc()
{
    return _MTL_msg_MTL__IndirectCommandBufferDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLIndirectCommandBufferDescriptor, nullptr);
}

_MTL_INLINE MTL::IndirectCommandBufferDescriptor* MTL::IndirectCommandBufferDescriptor::init() const
{
    return _MTL_msg_MTL__IndirectCommandBufferDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::IndirectCommandType MTL::IndirectCommandBufferDescriptor::commandTypes() const
{
    return _MTL_msg_MTL__IndirectCommandType_commandTypes((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setCommandTypes(MTL::IndirectCommandType commandTypes)
{
    _MTL_msg_v_setCommandTypes__MTL__IndirectCommandType((const void*)this, nullptr, commandTypes);
}

_MTL_INLINE bool MTL::IndirectCommandBufferDescriptor::inheritPipelineState() const
{
    return _MTL_msg_bool_inheritPipelineState((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setInheritPipelineState(bool inheritPipelineState)
{
    _MTL_msg_v_setInheritPipelineState__bool((const void*)this, nullptr, inheritPipelineState);
}

_MTL_INLINE bool MTL::IndirectCommandBufferDescriptor::inheritBuffers() const
{
    return _MTL_msg_bool_inheritBuffers((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setInheritBuffers(bool inheritBuffers)
{
    _MTL_msg_v_setInheritBuffers__bool((const void*)this, nullptr, inheritBuffers);
}

_MTL_INLINE bool MTL::IndirectCommandBufferDescriptor::inheritDepthStencilState() const
{
    return _MTL_msg_bool_inheritDepthStencilState((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setInheritDepthStencilState(bool inheritDepthStencilState)
{
    _MTL_msg_v_setInheritDepthStencilState__bool((const void*)this, nullptr, inheritDepthStencilState);
}

_MTL_INLINE bool MTL::IndirectCommandBufferDescriptor::inheritDepthBias() const
{
    return _MTL_msg_bool_inheritDepthBias((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setInheritDepthBias(bool inheritDepthBias)
{
    _MTL_msg_v_setInheritDepthBias__bool((const void*)this, nullptr, inheritDepthBias);
}

_MTL_INLINE bool MTL::IndirectCommandBufferDescriptor::inheritDepthClipMode() const
{
    return _MTL_msg_bool_inheritDepthClipMode((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setInheritDepthClipMode(bool inheritDepthClipMode)
{
    _MTL_msg_v_setInheritDepthClipMode__bool((const void*)this, nullptr, inheritDepthClipMode);
}

_MTL_INLINE bool MTL::IndirectCommandBufferDescriptor::inheritCullMode() const
{
    return _MTL_msg_bool_inheritCullMode((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setInheritCullMode(bool inheritCullMode)
{
    _MTL_msg_v_setInheritCullMode__bool((const void*)this, nullptr, inheritCullMode);
}

_MTL_INLINE bool MTL::IndirectCommandBufferDescriptor::inheritFrontFacingWinding() const
{
    return _MTL_msg_bool_inheritFrontFacingWinding((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setInheritFrontFacingWinding(bool inheritFrontFacingWinding)
{
    _MTL_msg_v_setInheritFrontFacingWinding__bool((const void*)this, nullptr, inheritFrontFacingWinding);
}

_MTL_INLINE bool MTL::IndirectCommandBufferDescriptor::inheritTriangleFillMode() const
{
    return _MTL_msg_bool_inheritTriangleFillMode((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setInheritTriangleFillMode(bool inheritTriangleFillMode)
{
    _MTL_msg_v_setInheritTriangleFillMode__bool((const void*)this, nullptr, inheritTriangleFillMode);
}

_MTL_INLINE NS::UInteger MTL::IndirectCommandBufferDescriptor::maxVertexBufferBindCount() const
{
    return _MTL_msg_NS__UInteger_maxVertexBufferBindCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setMaxVertexBufferBindCount(NS::UInteger maxVertexBufferBindCount)
{
    _MTL_msg_v_setMaxVertexBufferBindCount__NS__UInteger((const void*)this, nullptr, maxVertexBufferBindCount);
}

_MTL_INLINE NS::UInteger MTL::IndirectCommandBufferDescriptor::maxFragmentBufferBindCount() const
{
    return _MTL_msg_NS__UInteger_maxFragmentBufferBindCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setMaxFragmentBufferBindCount(NS::UInteger maxFragmentBufferBindCount)
{
    _MTL_msg_v_setMaxFragmentBufferBindCount__NS__UInteger((const void*)this, nullptr, maxFragmentBufferBindCount);
}

_MTL_INLINE NS::UInteger MTL::IndirectCommandBufferDescriptor::maxKernelBufferBindCount() const
{
    return _MTL_msg_NS__UInteger_maxKernelBufferBindCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setMaxKernelBufferBindCount(NS::UInteger maxKernelBufferBindCount)
{
    _MTL_msg_v_setMaxKernelBufferBindCount__NS__UInteger((const void*)this, nullptr, maxKernelBufferBindCount);
}

_MTL_INLINE NS::UInteger MTL::IndirectCommandBufferDescriptor::maxKernelThreadgroupMemoryBindCount() const
{
    return _MTL_msg_NS__UInteger_maxKernelThreadgroupMemoryBindCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setMaxKernelThreadgroupMemoryBindCount(NS::UInteger maxKernelThreadgroupMemoryBindCount)
{
    _MTL_msg_v_setMaxKernelThreadgroupMemoryBindCount__NS__UInteger((const void*)this, nullptr, maxKernelThreadgroupMemoryBindCount);
}

_MTL_INLINE NS::UInteger MTL::IndirectCommandBufferDescriptor::maxObjectBufferBindCount() const
{
    return _MTL_msg_NS__UInteger_maxObjectBufferBindCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setMaxObjectBufferBindCount(NS::UInteger maxObjectBufferBindCount)
{
    _MTL_msg_v_setMaxObjectBufferBindCount__NS__UInteger((const void*)this, nullptr, maxObjectBufferBindCount);
}

_MTL_INLINE NS::UInteger MTL::IndirectCommandBufferDescriptor::maxMeshBufferBindCount() const
{
    return _MTL_msg_NS__UInteger_maxMeshBufferBindCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setMaxMeshBufferBindCount(NS::UInteger maxMeshBufferBindCount)
{
    _MTL_msg_v_setMaxMeshBufferBindCount__NS__UInteger((const void*)this, nullptr, maxMeshBufferBindCount);
}

_MTL_INLINE NS::UInteger MTL::IndirectCommandBufferDescriptor::maxObjectThreadgroupMemoryBindCount() const
{
    return _MTL_msg_NS__UInteger_maxObjectThreadgroupMemoryBindCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setMaxObjectThreadgroupMemoryBindCount(NS::UInteger maxObjectThreadgroupMemoryBindCount)
{
    _MTL_msg_v_setMaxObjectThreadgroupMemoryBindCount__NS__UInteger((const void*)this, nullptr, maxObjectThreadgroupMemoryBindCount);
}

_MTL_INLINE bool MTL::IndirectCommandBufferDescriptor::supportRayTracing() const
{
    return _MTL_msg_bool_supportRayTracing((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setSupportRayTracing(bool supportRayTracing)
{
    _MTL_msg_v_setSupportRayTracing__bool((const void*)this, nullptr, supportRayTracing);
}

_MTL_INLINE bool MTL::IndirectCommandBufferDescriptor::supportDynamicAttributeStride() const
{
    return _MTL_msg_bool_supportDynamicAttributeStride((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setSupportDynamicAttributeStride(bool supportDynamicAttributeStride)
{
    _MTL_msg_v_setSupportDynamicAttributeStride__bool((const void*)this, nullptr, supportDynamicAttributeStride);
}

_MTL_INLINE bool MTL::IndirectCommandBufferDescriptor::supportColorAttachmentMapping() const
{
    return _MTL_msg_bool_supportColorAttachmentMapping((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectCommandBufferDescriptor::setSupportColorAttachmentMapping(bool supportColorAttachmentMapping)
{
    _MTL_msg_v_setSupportColorAttachmentMapping__bool((const void*)this, nullptr, supportColorAttachmentMapping);
}

_MTL_INLINE NS::UInteger MTL::IndirectCommandBuffer::size() const
{
    return _MTL_msg_NS__UInteger_size((const void*)this, nullptr);
}

_MTL_INLINE MTL::ResourceID MTL::IndirectCommandBuffer::gpuResourceID() const
{
    return _MTL_msg_MTL__ResourceID_gpuResourceID((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectCommandBuffer::reset(NS::Range range)
{
    _MTL_msg_v_resetWithRange__NS__Range((const void*)this, nullptr, range);
}

_MTL_INLINE MTL::IndirectRenderCommand* MTL::IndirectCommandBuffer::indirectRenderCommand(NS::UInteger commandIndex)
{
    return _MTL_msg_MTL__IndirectRenderCommandp_indirectRenderCommandAtIndex__NS__UInteger((const void*)this, nullptr, commandIndex);
}

_MTL_INLINE MTL::IndirectComputeCommand* MTL::IndirectCommandBuffer::indirectComputeCommand(NS::UInteger commandIndex)
{
    return _MTL_msg_MTL__IndirectComputeCommandp_indirectComputeCommandAtIndex__NS__UInteger((const void*)this, nullptr, commandIndex);
}
