#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    enum ShaderValidation : NS::Integer;
}
namespace NS {
    class String;
}

namespace MTL4
{

_MTL4_OPTIONS(NS::UInteger, ShaderReflection) {
    ShaderReflectionNone = 0,
    ShaderReflectionBindingInfo = 1 << 0,
    ShaderReflectionBufferTypeInfo = 1 << 1,
};

_MTL4_ENUM(NS::Integer, AlphaToOneState) {
    AlphaToOneStateDisabled = 0,
    AlphaToOneStateEnabled = 1,
};

_MTL4_ENUM(NS::Integer, AlphaToCoverageState) {
    AlphaToCoverageStateDisabled = 0,
    AlphaToCoverageStateEnabled = 1,
};

_MTL4_ENUM(NS::Integer, BlendState) {
    BlendStateDisabled = 0,
    BlendStateEnabled = 1,
    BlendStateUnspecialized = 2,
};

_MTL4_ENUM(NS::Integer, IndirectCommandBufferSupportState) {
    IndirectCommandBufferSupportStateDisabled = 0,
    IndirectCommandBufferSupportStateEnabled = 1,
};


class PipelineOptions;
class PipelineDescriptor;

class PipelineOptions : public NS::Copying<PipelineOptions>
{
public:
    static PipelineOptions* alloc();
    PipelineOptions*        init() const;

    void                   setShaderReflection(MTL4::ShaderReflection shaderReflection);
    void                   setShaderValidation(MTL::ShaderValidation shaderValidation);
    MTL4::ShaderReflection shaderReflection() const;
    MTL::ShaderValidation  shaderValidation() const;

};

class PipelineDescriptor : public NS::Copying<PipelineDescriptor>
{
public:
    static PipelineDescriptor* alloc();
    PipelineDescriptor*        init() const;

    NS::String*            label() const;
    MTL4::PipelineOptions* options() const;
    void                   setLabel(NS::String* label);
    void                   setOptions(MTL4::PipelineOptions* options);

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4PipelineOptions;
extern "C" void *OBJC_CLASS_$_MTL4PipelineDescriptor;

_MTL4_INLINE MTL4::PipelineOptions* MTL4::PipelineOptions::alloc()
{
    return _MTL4_msg_MTL4__PipelineOptionsp_alloc((const void*)&OBJC_CLASS_$_MTL4PipelineOptions, nullptr);
}

_MTL4_INLINE MTL4::PipelineOptions* MTL4::PipelineOptions::init() const
{
    return _MTL4_msg_MTL4__PipelineOptionsp_init((const void*)this, nullptr);
}

_MTL4_INLINE MTL::ShaderValidation MTL4::PipelineOptions::shaderValidation() const
{
    return _MTL4_msg_MTL__ShaderValidation_shaderValidation((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::PipelineOptions::setShaderValidation(MTL::ShaderValidation shaderValidation)
{
    _MTL4_msg_v_setShaderValidation__MTL__ShaderValidation((const void*)this, nullptr, shaderValidation);
}

_MTL4_INLINE MTL4::ShaderReflection MTL4::PipelineOptions::shaderReflection() const
{
    return _MTL4_msg_MTL4__ShaderReflection_shaderReflection((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::PipelineOptions::setShaderReflection(MTL4::ShaderReflection shaderReflection)
{
    _MTL4_msg_v_setShaderReflection__MTL4__ShaderReflection((const void*)this, nullptr, shaderReflection);
}

_MTL4_INLINE MTL4::PipelineDescriptor* MTL4::PipelineDescriptor::alloc()
{
    return _MTL4_msg_MTL4__PipelineDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4PipelineDescriptor, nullptr);
}

_MTL4_INLINE MTL4::PipelineDescriptor* MTL4::PipelineDescriptor::init() const
{
    return _MTL4_msg_MTL4__PipelineDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE NS::String* MTL4::PipelineDescriptor::label() const
{
    return _MTL4_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::PipelineDescriptor::setLabel(NS::String* label)
{
    _MTL4_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL4_INLINE MTL4::PipelineOptions* MTL4::PipelineDescriptor::options() const
{
    return _MTL4_msg_MTL4__PipelineOptionsp_options((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::PipelineDescriptor::setOptions(MTL4::PipelineOptions* options)
{
    _MTL4_msg_v_setOptions__MTL4__PipelineOptionsp((const void*)this, nullptr, options);
}
