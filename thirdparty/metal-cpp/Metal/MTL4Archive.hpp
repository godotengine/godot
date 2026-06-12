#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class ComputePipelineState;
    class RenderPipelineState;
}
namespace MTL4 {
    class BinaryFunction;
    class BinaryFunctionDescriptor;
    class ComputePipelineDescriptor;
    class PipelineDescriptor;
    class PipelineStageDynamicLinkingDescriptor;
    class RenderPipelineDynamicLinkingDescriptor;
}
namespace NS {
    class Error;
    class String;
}

namespace MTL4
{

class Archive : public NS::Referencing<Archive>
{
public:
    NS::String*                label() const;
    MTL4::BinaryFunction*      newBinaryFunction(MTL4::BinaryFunctionDescriptor* descriptor, NS::Error** error);
    MTL::ComputePipelineState* newComputePipelineState(MTL4::ComputePipelineDescriptor* descriptor, NS::Error** error);
    MTL::ComputePipelineState* newComputePipelineState(MTL4::ComputePipelineDescriptor* descriptor, MTL4::PipelineStageDynamicLinkingDescriptor* dynamicLinkingDescriptor, NS::Error** error);
    MTL::RenderPipelineState*  newRenderPipelineState(MTL4::PipelineDescriptor* descriptor, NS::Error** error);
    MTL::RenderPipelineState*  newRenderPipelineState(MTL4::PipelineDescriptor* descriptor, MTL4::RenderPipelineDynamicLinkingDescriptor* dynamicLinkingDescriptor, NS::Error** error);
    void                       setLabel(NS::String* label);

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4Archive;

_MTL4_INLINE NS::String* MTL4::Archive::label() const
{
    return _MTL4_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::Archive::setLabel(NS::String* label)
{
    _MTL4_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL4_INLINE MTL::ComputePipelineState* MTL4::Archive::newComputePipelineState(MTL4::ComputePipelineDescriptor* descriptor, NS::Error** error)
{
    return _MTL4_msg_MTL__ComputePipelineStatep_newComputePipelineStateWithDescriptor_error__MTL4__ComputePipelineDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}

_MTL4_INLINE MTL::ComputePipelineState* MTL4::Archive::newComputePipelineState(MTL4::ComputePipelineDescriptor* descriptor, MTL4::PipelineStageDynamicLinkingDescriptor* dynamicLinkingDescriptor, NS::Error** error)
{
    return _MTL4_msg_MTL__ComputePipelineStatep_newComputePipelineStateWithDescriptor_dynamicLinkingDescriptor_error__MTL4__ComputePipelineDescriptorp_MTL4__PipelineStageDynamicLinkingDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, dynamicLinkingDescriptor, error);
}

_MTL4_INLINE MTL::RenderPipelineState* MTL4::Archive::newRenderPipelineState(MTL4::PipelineDescriptor* descriptor, NS::Error** error)
{
    return _MTL4_msg_MTL__RenderPipelineStatep_newRenderPipelineStateWithDescriptor_error__MTL4__PipelineDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}

_MTL4_INLINE MTL::RenderPipelineState* MTL4::Archive::newRenderPipelineState(MTL4::PipelineDescriptor* descriptor, MTL4::RenderPipelineDynamicLinkingDescriptor* dynamicLinkingDescriptor, NS::Error** error)
{
    return _MTL4_msg_MTL__RenderPipelineStatep_newRenderPipelineStateWithDescriptor_dynamicLinkingDescriptor_error__MTL4__PipelineDescriptorp_MTL4__RenderPipelineDynamicLinkingDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, dynamicLinkingDescriptor, error);
}

_MTL4_INLINE MTL4::BinaryFunction* MTL4::Archive::newBinaryFunction(MTL4::BinaryFunctionDescriptor* descriptor, NS::Error** error)
{
    return _MTL4_msg_MTL4__BinaryFunctionp_newBinaryFunctionWithDescriptor_error__MTL4__BinaryFunctionDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}
