#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL4
{

class FunctionDescriptor : public NS::Copying<FunctionDescriptor>
{
public:
    static FunctionDescriptor* alloc();
    FunctionDescriptor*        init() const;

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4FunctionDescriptor;

_MTL4_INLINE MTL4::FunctionDescriptor* MTL4::FunctionDescriptor::alloc()
{
    return _MTL4_msg_MTL4__FunctionDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4FunctionDescriptor, nullptr);
}

_MTL4_INLINE MTL4::FunctionDescriptor* MTL4::FunctionDescriptor::init() const
{
    return _MTL4_msg_MTL4__FunctionDescriptorp_init((const void*)this, nullptr);
}
