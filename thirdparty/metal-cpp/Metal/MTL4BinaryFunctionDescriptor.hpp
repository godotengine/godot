#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL4 {
    class FunctionDescriptor;
}
namespace NS {
    class String;
}

namespace MTL4
{

_MTL4_OPTIONS(NS::UInteger, BinaryFunctionOptions) {
    BinaryFunctionOptionNone = 0,
    BinaryFunctionOptionPipelineIndependent = 1 << 1,
};


class BinaryFunctionDescriptor : public NS::Copying<BinaryFunctionDescriptor>
{
public:
    static BinaryFunctionDescriptor* alloc();
    BinaryFunctionDescriptor*        init() const;

    MTL4::FunctionDescriptor*   functionDescriptor() const;
    NS::String*                 name() const;
    MTL4::BinaryFunctionOptions options() const;
    void                        setFunctionDescriptor(MTL4::FunctionDescriptor* functionDescriptor);
    void                        setName(NS::String* name);
    void                        setOptions(MTL4::BinaryFunctionOptions options);

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4BinaryFunctionDescriptor;

_MTL4_INLINE MTL4::BinaryFunctionDescriptor* MTL4::BinaryFunctionDescriptor::alloc()
{
    return _MTL4_msg_MTL4__BinaryFunctionDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4BinaryFunctionDescriptor, nullptr);
}

_MTL4_INLINE MTL4::BinaryFunctionDescriptor* MTL4::BinaryFunctionDescriptor::init() const
{
    return _MTL4_msg_MTL4__BinaryFunctionDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE NS::String* MTL4::BinaryFunctionDescriptor::name() const
{
    return _MTL4_msg_NS__Stringp_name((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::BinaryFunctionDescriptor::setName(NS::String* name)
{
    _MTL4_msg_v_setName__NS__Stringp((const void*)this, nullptr, name);
}

_MTL4_INLINE MTL4::FunctionDescriptor* MTL4::BinaryFunctionDescriptor::functionDescriptor() const
{
    return _MTL4_msg_MTL4__FunctionDescriptorp_functionDescriptor((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::BinaryFunctionDescriptor::setFunctionDescriptor(MTL4::FunctionDescriptor* functionDescriptor)
{
    _MTL4_msg_v_setFunctionDescriptor__MTL4__FunctionDescriptorp((const void*)this, nullptr, functionDescriptor);
}

_MTL4_INLINE MTL4::BinaryFunctionOptions MTL4::BinaryFunctionDescriptor::options() const
{
    return _MTL4_msg_MTL4__BinaryFunctionOptions_options((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::BinaryFunctionDescriptor::setOptions(MTL4::BinaryFunctionOptions options)
{
    _MTL4_msg_v_setOptions__MTL4__BinaryFunctionOptions((const void*)this, nullptr, options);
}
