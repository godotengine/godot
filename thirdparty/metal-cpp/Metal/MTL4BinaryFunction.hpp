#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    enum FunctionType : NS::UInteger;
}
namespace NS {
    class String;
}

namespace MTL4
{

class BinaryFunction : public NS::Referencing<BinaryFunction>
{
public:
    MTL::FunctionType functionType() const;
    NS::String*       name() const;

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4BinaryFunction;

_MTL4_INLINE NS::String* MTL4::BinaryFunction::name() const
{
    return _MTL4_msg_NS__Stringp_name((const void*)this, nullptr);
}

_MTL4_INLINE MTL::FunctionType MTL4::BinaryFunction::functionType() const
{
    return _MTL4_msg_MTL__FunctionType_functionType((const void*)this, nullptr);
}
