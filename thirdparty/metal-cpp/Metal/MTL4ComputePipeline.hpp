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
    class StaticLinkingDescriptor;
    enum IndirectCommandBufferSupportState : NS::Integer;
}

namespace MTL4
{

class ComputePipelineDescriptor : public NS::Referencing<ComputePipelineDescriptor, MTL4::PipelineDescriptor>
{
public:
    static ComputePipelineDescriptor* alloc();
    ComputePipelineDescriptor*        init() const;

    MTL4::FunctionDescriptor*               computeFunctionDescriptor() const;
    NS::UInteger                            maxTotalThreadsPerThreadgroup() const;
    MTL::Size                               requiredThreadsPerThreadgroup() const;
    void                                    reset();
    void                                    setComputeFunctionDescriptor(MTL4::FunctionDescriptor* computeFunctionDescriptor);
    void                                    setMaxTotalThreadsPerThreadgroup(NS::UInteger maxTotalThreadsPerThreadgroup);
    void                                    setRequiredThreadsPerThreadgroup(MTL::Size requiredThreadsPerThreadgroup);
    void                                    setStaticLinkingDescriptor(MTL4::StaticLinkingDescriptor* staticLinkingDescriptor);
    void                                    setSupportBinaryLinking(bool supportBinaryLinking);
    void                                    setSupportIndirectCommandBuffers(MTL4::IndirectCommandBufferSupportState supportIndirectCommandBuffers);
    void                                    setThreadGroupSizeIsMultipleOfThreadExecutionWidth(bool threadGroupSizeIsMultipleOfThreadExecutionWidth);
    MTL4::StaticLinkingDescriptor*          staticLinkingDescriptor() const;
    bool                                    supportBinaryLinking() const;
    MTL4::IndirectCommandBufferSupportState supportIndirectCommandBuffers() const;
    bool                                    threadGroupSizeIsMultipleOfThreadExecutionWidth() const;

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4ComputePipelineDescriptor;

_MTL4_INLINE MTL4::ComputePipelineDescriptor* MTL4::ComputePipelineDescriptor::alloc()
{
    return _MTL4_msg_MTL4__ComputePipelineDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4ComputePipelineDescriptor, nullptr);
}

_MTL4_INLINE MTL4::ComputePipelineDescriptor* MTL4::ComputePipelineDescriptor::init() const
{
    return _MTL4_msg_MTL4__ComputePipelineDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::FunctionDescriptor* MTL4::ComputePipelineDescriptor::computeFunctionDescriptor() const
{
    return _MTL4_msg_MTL4__FunctionDescriptorp_computeFunctionDescriptor((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::ComputePipelineDescriptor::setComputeFunctionDescriptor(MTL4::FunctionDescriptor* computeFunctionDescriptor)
{
    _MTL4_msg_v_setComputeFunctionDescriptor__MTL4__FunctionDescriptorp((const void*)this, nullptr, computeFunctionDescriptor);
}

_MTL4_INLINE bool MTL4::ComputePipelineDescriptor::threadGroupSizeIsMultipleOfThreadExecutionWidth() const
{
    return _MTL4_msg_bool_threadGroupSizeIsMultipleOfThreadExecutionWidth((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::ComputePipelineDescriptor::setThreadGroupSizeIsMultipleOfThreadExecutionWidth(bool threadGroupSizeIsMultipleOfThreadExecutionWidth)
{
    _MTL4_msg_v_setThreadGroupSizeIsMultipleOfThreadExecutionWidth__bool((const void*)this, nullptr, threadGroupSizeIsMultipleOfThreadExecutionWidth);
}

_MTL4_INLINE NS::UInteger MTL4::ComputePipelineDescriptor::maxTotalThreadsPerThreadgroup() const
{
    return _MTL4_msg_NS__UInteger_maxTotalThreadsPerThreadgroup((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::ComputePipelineDescriptor::setMaxTotalThreadsPerThreadgroup(NS::UInteger maxTotalThreadsPerThreadgroup)
{
    _MTL4_msg_v_setMaxTotalThreadsPerThreadgroup__NS__UInteger((const void*)this, nullptr, maxTotalThreadsPerThreadgroup);
}

_MTL4_INLINE MTL::Size MTL4::ComputePipelineDescriptor::requiredThreadsPerThreadgroup() const
{
    return _MTL4_msg_MTL__Size_requiredThreadsPerThreadgroup((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::ComputePipelineDescriptor::setRequiredThreadsPerThreadgroup(MTL::Size requiredThreadsPerThreadgroup)
{
    _MTL4_msg_v_setRequiredThreadsPerThreadgroup__MTL__Size((const void*)this, nullptr, requiredThreadsPerThreadgroup);
}

_MTL4_INLINE bool MTL4::ComputePipelineDescriptor::supportBinaryLinking() const
{
    return _MTL4_msg_bool_supportBinaryLinking((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::ComputePipelineDescriptor::setSupportBinaryLinking(bool supportBinaryLinking)
{
    _MTL4_msg_v_setSupportBinaryLinking__bool((const void*)this, nullptr, supportBinaryLinking);
}

_MTL4_INLINE MTL4::StaticLinkingDescriptor* MTL4::ComputePipelineDescriptor::staticLinkingDescriptor() const
{
    return _MTL4_msg_MTL4__StaticLinkingDescriptorp_staticLinkingDescriptor((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::ComputePipelineDescriptor::setStaticLinkingDescriptor(MTL4::StaticLinkingDescriptor* staticLinkingDescriptor)
{
    _MTL4_msg_v_setStaticLinkingDescriptor__MTL4__StaticLinkingDescriptorp((const void*)this, nullptr, staticLinkingDescriptor);
}

_MTL4_INLINE MTL4::IndirectCommandBufferSupportState MTL4::ComputePipelineDescriptor::supportIndirectCommandBuffers() const
{
    return _MTL4_msg_MTL4__IndirectCommandBufferSupportState_supportIndirectCommandBuffers((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::ComputePipelineDescriptor::setSupportIndirectCommandBuffers(MTL4::IndirectCommandBufferSupportState supportIndirectCommandBuffers)
{
    _MTL4_msg_v_setSupportIndirectCommandBuffers__MTL4__IndirectCommandBufferSupportState((const void*)this, nullptr, supportIndirectCommandBuffers);
}

_MTL4_INLINE void MTL4::ComputePipelineDescriptor::reset()
{
    _MTL4_msg_v_reset((const void*)this, nullptr);
}
