#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTL4FunctionDescriptor.hpp"

namespace MTL {
    class FunctionConstantValues;
}
namespace MTL4 {
    class FunctionDescriptor;
}
namespace NS {
    class String;
}

namespace MTL4
{

class SpecializedFunctionDescriptor : public NS::Referencing<SpecializedFunctionDescriptor, MTL4::FunctionDescriptor>
{
public:
    static SpecializedFunctionDescriptor* alloc();
    SpecializedFunctionDescriptor*        init() const;

    MTL::FunctionConstantValues* constantValues() const;
    MTL4::FunctionDescriptor*    functionDescriptor() const;
    void                         setConstantValues(MTL::FunctionConstantValues* constantValues);
    void                         setFunctionDescriptor(MTL4::FunctionDescriptor* functionDescriptor);
    void                         setSpecializedName(NS::String* specializedName);
    NS::String*                  specializedName() const;

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4SpecializedFunctionDescriptor;

_MTL4_INLINE MTL4::SpecializedFunctionDescriptor* MTL4::SpecializedFunctionDescriptor::alloc()
{
    return _MTL4_msg_MTL4__SpecializedFunctionDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4SpecializedFunctionDescriptor, nullptr);
}

_MTL4_INLINE MTL4::SpecializedFunctionDescriptor* MTL4::SpecializedFunctionDescriptor::init() const
{
    return _MTL4_msg_MTL4__SpecializedFunctionDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::FunctionDescriptor* MTL4::SpecializedFunctionDescriptor::functionDescriptor() const
{
    return _MTL4_msg_MTL4__FunctionDescriptorp_functionDescriptor((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::SpecializedFunctionDescriptor::setFunctionDescriptor(MTL4::FunctionDescriptor* functionDescriptor)
{
    _MTL4_msg_v_setFunctionDescriptor__MTL4__FunctionDescriptorp((const void*)this, nullptr, functionDescriptor);
}

_MTL4_INLINE NS::String* MTL4::SpecializedFunctionDescriptor::specializedName() const
{
    return _MTL4_msg_NS__Stringp_specializedName((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::SpecializedFunctionDescriptor::setSpecializedName(NS::String* specializedName)
{
    _MTL4_msg_v_setSpecializedName__NS__Stringp((const void*)this, nullptr, specializedName);
}

_MTL4_INLINE MTL::FunctionConstantValues* MTL4::SpecializedFunctionDescriptor::constantValues() const
{
    return _MTL4_msg_MTL__FunctionConstantValuesp_constantValues((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::SpecializedFunctionDescriptor::setConstantValues(MTL::FunctionConstantValues* constantValues)
{
    _MTL4_msg_v_setConstantValues__MTL__FunctionConstantValuesp((const void*)this, nullptr, constantValues);
}
