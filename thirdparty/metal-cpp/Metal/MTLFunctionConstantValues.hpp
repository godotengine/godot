#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    enum DataType : NS::UInteger;
}
namespace NS {
    class String;
}

namespace MTL
{

class FunctionConstantValues : public NS::Copying<FunctionConstantValues>
{
public:
    static FunctionConstantValues* alloc();
    FunctionConstantValues*        init() const;

    void reset();
    void setConstantValue(const void * value, MTL::DataType type, NS::UInteger index);
    void setConstantValue(const void * value, MTL::DataType type, NS::String* name);
    void setConstantValues(const void * values, MTL::DataType type, NS::Range range);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLFunctionConstantValues;

_MTL_INLINE MTL::FunctionConstantValues* MTL::FunctionConstantValues::alloc()
{
    return _MTL_msg_MTL__FunctionConstantValuesp_alloc((const void*)&OBJC_CLASS_$_MTLFunctionConstantValues, nullptr);
}

_MTL_INLINE MTL::FunctionConstantValues* MTL::FunctionConstantValues::init() const
{
    return _MTL_msg_MTL__FunctionConstantValuesp_init((const void*)this, nullptr);
}

_MTL_INLINE void MTL::FunctionConstantValues::setConstantValue(const void * value, MTL::DataType type, NS::UInteger index)
{
    _MTL_msg_v_setConstantValue_type_atIndex__constvoidp_MTL__DataType_NS__UInteger((const void*)this, nullptr, value, type, index);
}

_MTL_INLINE void MTL::FunctionConstantValues::setConstantValues(const void * values, MTL::DataType type, NS::Range range)
{
    _MTL_msg_v_setConstantValues_type_withRange__constvoidp_MTL__DataType_NS__Range((const void*)this, nullptr, values, type, range);
}

_MTL_INLINE void MTL::FunctionConstantValues::setConstantValue(const void * value, MTL::DataType type, NS::String* name)
{
    _MTL_msg_v_setConstantValue_type_withName__constvoidp_MTL__DataType_NS__Stringp((const void*)this, nullptr, value, type, name);
}

_MTL_INLINE void MTL::FunctionConstantValues::reset()
{
    _MTL_msg_v_reset((const void*)this, nullptr);
}
