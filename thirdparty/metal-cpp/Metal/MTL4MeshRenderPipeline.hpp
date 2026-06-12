#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTL4PipelineState.hpp"
#include "MTLStructs.hpp"

namespace MTL4 {
    class FunctionDescriptor;
    class RenderPipelineColorAttachmentDescriptorArray;
    class StaticLinkingDescriptor;
    enum AlphaToCoverageState : NS::Integer;
    enum AlphaToOneState : NS::Integer;
    enum IndirectCommandBufferSupportState : NS::Integer;
    enum LogicalToPhysicalColorAttachmentMappingState : NS::Integer;
}

namespace MTL4
{

class MeshRenderPipelineDescriptor : public NS::Referencing<MeshRenderPipelineDescriptor, MTL4::PipelineDescriptor>
{
public:
    static MeshRenderPipelineDescriptor* alloc();
    MeshRenderPipelineDescriptor*        init() const;

    MTL4::AlphaToCoverageState                          alphaToCoverageState() const;
    MTL4::AlphaToOneState                               alphaToOneState() const;
    MTL4::LogicalToPhysicalColorAttachmentMappingState  colorAttachmentMappingState() const;
    MTL4::RenderPipelineColorAttachmentDescriptorArray* colorAttachments() const;
    MTL4::FunctionDescriptor*                           fragmentFunctionDescriptor() const;
    MTL4::StaticLinkingDescriptor*                      fragmentStaticLinkingDescriptor() const;
    bool                                                isRasterizationEnabled();
    NS::UInteger                                        maxTotalThreadgroupsPerMeshGrid() const;
    NS::UInteger                                        maxTotalThreadsPerMeshThreadgroup() const;
    NS::UInteger                                        maxTotalThreadsPerObjectThreadgroup() const;
    NS::UInteger                                        maxVertexAmplificationCount() const;
    MTL4::FunctionDescriptor*                           meshFunctionDescriptor() const;
    MTL4::StaticLinkingDescriptor*                      meshStaticLinkingDescriptor() const;
    bool                                                meshThreadgroupSizeIsMultipleOfThreadExecutionWidth() const;
    MTL4::FunctionDescriptor*                           objectFunctionDescriptor() const;
    MTL4::StaticLinkingDescriptor*                      objectStaticLinkingDescriptor() const;
    bool                                                objectThreadgroupSizeIsMultipleOfThreadExecutionWidth() const;
    NS::UInteger                                        payloadMemoryLength() const;
    NS::UInteger                                        rasterSampleCount() const;
    bool                                                rasterizationEnabled() const;
    MTL::Size                                           requiredThreadsPerMeshThreadgroup() const;
    MTL::Size                                           requiredThreadsPerObjectThreadgroup() const;
    void                                                reset();
    void                                                setAlphaToCoverageState(MTL4::AlphaToCoverageState alphaToCoverageState);
    void                                                setAlphaToOneState(MTL4::AlphaToOneState alphaToOneState);
    void                                                setColorAttachmentMappingState(MTL4::LogicalToPhysicalColorAttachmentMappingState colorAttachmentMappingState);
    void                                                setFragmentFunctionDescriptor(MTL4::FunctionDescriptor* fragmentFunctionDescriptor);
    void                                                setFragmentStaticLinkingDescriptor(MTL4::StaticLinkingDescriptor* fragmentStaticLinkingDescriptor);
    void                                                setMaxTotalThreadgroupsPerMeshGrid(NS::UInteger maxTotalThreadgroupsPerMeshGrid);
    void                                                setMaxTotalThreadsPerMeshThreadgroup(NS::UInteger maxTotalThreadsPerMeshThreadgroup);
    void                                                setMaxTotalThreadsPerObjectThreadgroup(NS::UInteger maxTotalThreadsPerObjectThreadgroup);
    void                                                setMaxVertexAmplificationCount(NS::UInteger maxVertexAmplificationCount);
    void                                                setMeshFunctionDescriptor(MTL4::FunctionDescriptor* meshFunctionDescriptor);
    void                                                setMeshStaticLinkingDescriptor(MTL4::StaticLinkingDescriptor* meshStaticLinkingDescriptor);
    void                                                setMeshThreadgroupSizeIsMultipleOfThreadExecutionWidth(bool meshThreadgroupSizeIsMultipleOfThreadExecutionWidth);
    void                                                setObjectFunctionDescriptor(MTL4::FunctionDescriptor* objectFunctionDescriptor);
    void                                                setObjectStaticLinkingDescriptor(MTL4::StaticLinkingDescriptor* objectStaticLinkingDescriptor);
    void                                                setObjectThreadgroupSizeIsMultipleOfThreadExecutionWidth(bool objectThreadgroupSizeIsMultipleOfThreadExecutionWidth);
    void                                                setPayloadMemoryLength(NS::UInteger payloadMemoryLength);
    void                                                setRasterSampleCount(NS::UInteger rasterSampleCount);
    void                                                setRasterizationEnabled(bool rasterizationEnabled);
    void                                                setRequiredThreadsPerMeshThreadgroup(MTL::Size requiredThreadsPerMeshThreadgroup);
    void                                                setRequiredThreadsPerObjectThreadgroup(MTL::Size requiredThreadsPerObjectThreadgroup);
    void                                                setSupportFragmentBinaryLinking(bool supportFragmentBinaryLinking);
    void                                                setSupportIndirectCommandBuffers(MTL4::IndirectCommandBufferSupportState supportIndirectCommandBuffers);
    void                                                setSupportMeshBinaryLinking(bool supportMeshBinaryLinking);
    void                                                setSupportObjectBinaryLinking(bool supportObjectBinaryLinking);
    bool                                                supportFragmentBinaryLinking() const;
    MTL4::IndirectCommandBufferSupportState             supportIndirectCommandBuffers() const;
    bool                                                supportMeshBinaryLinking() const;
    bool                                                supportObjectBinaryLinking() const;

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4MeshRenderPipelineDescriptor;

_MTL4_INLINE MTL4::MeshRenderPipelineDescriptor* MTL4::MeshRenderPipelineDescriptor::alloc()
{
    return _MTL4_msg_MTL4__MeshRenderPipelineDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4MeshRenderPipelineDescriptor, nullptr);
}

_MTL4_INLINE MTL4::MeshRenderPipelineDescriptor* MTL4::MeshRenderPipelineDescriptor::init() const
{
    return _MTL4_msg_MTL4__MeshRenderPipelineDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::FunctionDescriptor* MTL4::MeshRenderPipelineDescriptor::objectFunctionDescriptor() const
{
    return _MTL4_msg_MTL4__FunctionDescriptorp_objectFunctionDescriptor((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::setObjectFunctionDescriptor(MTL4::FunctionDescriptor* objectFunctionDescriptor)
{
    _MTL4_msg_v_setObjectFunctionDescriptor__MTL4__FunctionDescriptorp((const void*)this, nullptr, objectFunctionDescriptor);
}

_MTL4_INLINE MTL4::FunctionDescriptor* MTL4::MeshRenderPipelineDescriptor::meshFunctionDescriptor() const
{
    return _MTL4_msg_MTL4__FunctionDescriptorp_meshFunctionDescriptor((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::setMeshFunctionDescriptor(MTL4::FunctionDescriptor* meshFunctionDescriptor)
{
    _MTL4_msg_v_setMeshFunctionDescriptor__MTL4__FunctionDescriptorp((const void*)this, nullptr, meshFunctionDescriptor);
}

_MTL4_INLINE MTL4::FunctionDescriptor* MTL4::MeshRenderPipelineDescriptor::fragmentFunctionDescriptor() const
{
    return _MTL4_msg_MTL4__FunctionDescriptorp_fragmentFunctionDescriptor((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::setFragmentFunctionDescriptor(MTL4::FunctionDescriptor* fragmentFunctionDescriptor)
{
    _MTL4_msg_v_setFragmentFunctionDescriptor__MTL4__FunctionDescriptorp((const void*)this, nullptr, fragmentFunctionDescriptor);
}

_MTL4_INLINE NS::UInteger MTL4::MeshRenderPipelineDescriptor::maxTotalThreadsPerObjectThreadgroup() const
{
    return _MTL4_msg_NS__UInteger_maxTotalThreadsPerObjectThreadgroup((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::setMaxTotalThreadsPerObjectThreadgroup(NS::UInteger maxTotalThreadsPerObjectThreadgroup)
{
    _MTL4_msg_v_setMaxTotalThreadsPerObjectThreadgroup__NS__UInteger((const void*)this, nullptr, maxTotalThreadsPerObjectThreadgroup);
}

_MTL4_INLINE NS::UInteger MTL4::MeshRenderPipelineDescriptor::maxTotalThreadsPerMeshThreadgroup() const
{
    return _MTL4_msg_NS__UInteger_maxTotalThreadsPerMeshThreadgroup((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::setMaxTotalThreadsPerMeshThreadgroup(NS::UInteger maxTotalThreadsPerMeshThreadgroup)
{
    _MTL4_msg_v_setMaxTotalThreadsPerMeshThreadgroup__NS__UInteger((const void*)this, nullptr, maxTotalThreadsPerMeshThreadgroup);
}

_MTL4_INLINE MTL::Size MTL4::MeshRenderPipelineDescriptor::requiredThreadsPerObjectThreadgroup() const
{
    return _MTL4_msg_MTL__Size_requiredThreadsPerObjectThreadgroup((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::setRequiredThreadsPerObjectThreadgroup(MTL::Size requiredThreadsPerObjectThreadgroup)
{
    _MTL4_msg_v_setRequiredThreadsPerObjectThreadgroup__MTL__Size((const void*)this, nullptr, requiredThreadsPerObjectThreadgroup);
}

_MTL4_INLINE MTL::Size MTL4::MeshRenderPipelineDescriptor::requiredThreadsPerMeshThreadgroup() const
{
    return _MTL4_msg_MTL__Size_requiredThreadsPerMeshThreadgroup((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::setRequiredThreadsPerMeshThreadgroup(MTL::Size requiredThreadsPerMeshThreadgroup)
{
    _MTL4_msg_v_setRequiredThreadsPerMeshThreadgroup__MTL__Size((const void*)this, nullptr, requiredThreadsPerMeshThreadgroup);
}

_MTL4_INLINE bool MTL4::MeshRenderPipelineDescriptor::objectThreadgroupSizeIsMultipleOfThreadExecutionWidth() const
{
    return _MTL4_msg_bool_objectThreadgroupSizeIsMultipleOfThreadExecutionWidth((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::setObjectThreadgroupSizeIsMultipleOfThreadExecutionWidth(bool objectThreadgroupSizeIsMultipleOfThreadExecutionWidth)
{
    _MTL4_msg_v_setObjectThreadgroupSizeIsMultipleOfThreadExecutionWidth__bool((const void*)this, nullptr, objectThreadgroupSizeIsMultipleOfThreadExecutionWidth);
}

_MTL4_INLINE bool MTL4::MeshRenderPipelineDescriptor::meshThreadgroupSizeIsMultipleOfThreadExecutionWidth() const
{
    return _MTL4_msg_bool_meshThreadgroupSizeIsMultipleOfThreadExecutionWidth((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::setMeshThreadgroupSizeIsMultipleOfThreadExecutionWidth(bool meshThreadgroupSizeIsMultipleOfThreadExecutionWidth)
{
    _MTL4_msg_v_setMeshThreadgroupSizeIsMultipleOfThreadExecutionWidth__bool((const void*)this, nullptr, meshThreadgroupSizeIsMultipleOfThreadExecutionWidth);
}

_MTL4_INLINE NS::UInteger MTL4::MeshRenderPipelineDescriptor::payloadMemoryLength() const
{
    return _MTL4_msg_NS__UInteger_payloadMemoryLength((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::setPayloadMemoryLength(NS::UInteger payloadMemoryLength)
{
    _MTL4_msg_v_setPayloadMemoryLength__NS__UInteger((const void*)this, nullptr, payloadMemoryLength);
}

_MTL4_INLINE NS::UInteger MTL4::MeshRenderPipelineDescriptor::maxTotalThreadgroupsPerMeshGrid() const
{
    return _MTL4_msg_NS__UInteger_maxTotalThreadgroupsPerMeshGrid((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::setMaxTotalThreadgroupsPerMeshGrid(NS::UInteger maxTotalThreadgroupsPerMeshGrid)
{
    _MTL4_msg_v_setMaxTotalThreadgroupsPerMeshGrid__NS__UInteger((const void*)this, nullptr, maxTotalThreadgroupsPerMeshGrid);
}

_MTL4_INLINE NS::UInteger MTL4::MeshRenderPipelineDescriptor::rasterSampleCount() const
{
    return _MTL4_msg_NS__UInteger_rasterSampleCount((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::setRasterSampleCount(NS::UInteger rasterSampleCount)
{
    _MTL4_msg_v_setRasterSampleCount__NS__UInteger((const void*)this, nullptr, rasterSampleCount);
}

_MTL4_INLINE MTL4::AlphaToCoverageState MTL4::MeshRenderPipelineDescriptor::alphaToCoverageState() const
{
    return _MTL4_msg_MTL4__AlphaToCoverageState_alphaToCoverageState((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::setAlphaToCoverageState(MTL4::AlphaToCoverageState alphaToCoverageState)
{
    _MTL4_msg_v_setAlphaToCoverageState__MTL4__AlphaToCoverageState((const void*)this, nullptr, alphaToCoverageState);
}

_MTL4_INLINE MTL4::AlphaToOneState MTL4::MeshRenderPipelineDescriptor::alphaToOneState() const
{
    return _MTL4_msg_MTL4__AlphaToOneState_alphaToOneState((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::setAlphaToOneState(MTL4::AlphaToOneState alphaToOneState)
{
    _MTL4_msg_v_setAlphaToOneState__MTL4__AlphaToOneState((const void*)this, nullptr, alphaToOneState);
}

_MTL4_INLINE bool MTL4::MeshRenderPipelineDescriptor::rasterizationEnabled() const
{
    return _MTL4_msg_bool_rasterizationEnabled((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::setRasterizationEnabled(bool rasterizationEnabled)
{
    _MTL4_msg_v_setRasterizationEnabled__bool((const void*)this, nullptr, rasterizationEnabled);
}

_MTL4_INLINE NS::UInteger MTL4::MeshRenderPipelineDescriptor::maxVertexAmplificationCount() const
{
    return _MTL4_msg_NS__UInteger_maxVertexAmplificationCount((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::setMaxVertexAmplificationCount(NS::UInteger maxVertexAmplificationCount)
{
    _MTL4_msg_v_setMaxVertexAmplificationCount__NS__UInteger((const void*)this, nullptr, maxVertexAmplificationCount);
}

_MTL4_INLINE MTL4::RenderPipelineColorAttachmentDescriptorArray* MTL4::MeshRenderPipelineDescriptor::colorAttachments() const
{
    return _MTL4_msg_MTL4__RenderPipelineColorAttachmentDescriptorArrayp_colorAttachments((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::StaticLinkingDescriptor* MTL4::MeshRenderPipelineDescriptor::objectStaticLinkingDescriptor() const
{
    return _MTL4_msg_MTL4__StaticLinkingDescriptorp_objectStaticLinkingDescriptor((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::setObjectStaticLinkingDescriptor(MTL4::StaticLinkingDescriptor* objectStaticLinkingDescriptor)
{
    _MTL4_msg_v_setObjectStaticLinkingDescriptor__MTL4__StaticLinkingDescriptorp((const void*)this, nullptr, objectStaticLinkingDescriptor);
}

_MTL4_INLINE MTL4::StaticLinkingDescriptor* MTL4::MeshRenderPipelineDescriptor::meshStaticLinkingDescriptor() const
{
    return _MTL4_msg_MTL4__StaticLinkingDescriptorp_meshStaticLinkingDescriptor((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::setMeshStaticLinkingDescriptor(MTL4::StaticLinkingDescriptor* meshStaticLinkingDescriptor)
{
    _MTL4_msg_v_setMeshStaticLinkingDescriptor__MTL4__StaticLinkingDescriptorp((const void*)this, nullptr, meshStaticLinkingDescriptor);
}

_MTL4_INLINE MTL4::StaticLinkingDescriptor* MTL4::MeshRenderPipelineDescriptor::fragmentStaticLinkingDescriptor() const
{
    return _MTL4_msg_MTL4__StaticLinkingDescriptorp_fragmentStaticLinkingDescriptor((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::setFragmentStaticLinkingDescriptor(MTL4::StaticLinkingDescriptor* fragmentStaticLinkingDescriptor)
{
    _MTL4_msg_v_setFragmentStaticLinkingDescriptor__MTL4__StaticLinkingDescriptorp((const void*)this, nullptr, fragmentStaticLinkingDescriptor);
}

_MTL4_INLINE bool MTL4::MeshRenderPipelineDescriptor::supportObjectBinaryLinking() const
{
    return _MTL4_msg_bool_supportObjectBinaryLinking((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::setSupportObjectBinaryLinking(bool supportObjectBinaryLinking)
{
    _MTL4_msg_v_setSupportObjectBinaryLinking__bool((const void*)this, nullptr, supportObjectBinaryLinking);
}

_MTL4_INLINE bool MTL4::MeshRenderPipelineDescriptor::supportMeshBinaryLinking() const
{
    return _MTL4_msg_bool_supportMeshBinaryLinking((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::setSupportMeshBinaryLinking(bool supportMeshBinaryLinking)
{
    _MTL4_msg_v_setSupportMeshBinaryLinking__bool((const void*)this, nullptr, supportMeshBinaryLinking);
}

_MTL4_INLINE bool MTL4::MeshRenderPipelineDescriptor::supportFragmentBinaryLinking() const
{
    return _MTL4_msg_bool_supportFragmentBinaryLinking((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::setSupportFragmentBinaryLinking(bool supportFragmentBinaryLinking)
{
    _MTL4_msg_v_setSupportFragmentBinaryLinking__bool((const void*)this, nullptr, supportFragmentBinaryLinking);
}

_MTL4_INLINE MTL4::LogicalToPhysicalColorAttachmentMappingState MTL4::MeshRenderPipelineDescriptor::colorAttachmentMappingState() const
{
    return _MTL4_msg_MTL4__LogicalToPhysicalColorAttachmentMappingState_colorAttachmentMappingState((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::setColorAttachmentMappingState(MTL4::LogicalToPhysicalColorAttachmentMappingState colorAttachmentMappingState)
{
    _MTL4_msg_v_setColorAttachmentMappingState__MTL4__LogicalToPhysicalColorAttachmentMappingState((const void*)this, nullptr, colorAttachmentMappingState);
}

_MTL4_INLINE MTL4::IndirectCommandBufferSupportState MTL4::MeshRenderPipelineDescriptor::supportIndirectCommandBuffers() const
{
    return _MTL4_msg_MTL4__IndirectCommandBufferSupportState_supportIndirectCommandBuffers((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::setSupportIndirectCommandBuffers(MTL4::IndirectCommandBufferSupportState supportIndirectCommandBuffers)
{
    _MTL4_msg_v_setSupportIndirectCommandBuffers__MTL4__IndirectCommandBufferSupportState((const void*)this, nullptr, supportIndirectCommandBuffers);
}

_MTL4_INLINE void MTL4::MeshRenderPipelineDescriptor::reset()
{
    _MTL4_msg_v_reset((const void*)this, nullptr);
}

_MTL4_INLINE bool MTL4::MeshRenderPipelineDescriptor::isRasterizationEnabled()
{
    return _MTL4_msg_bool_isRasterizationEnabled((const void*)this, nullptr);
}
