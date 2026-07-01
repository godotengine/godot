#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class Device;
    enum FunctionType : NS::UInteger;
}
namespace NS {
    class String;
}

namespace MTL
{

class FunctionHandle : public NS::Referencing<FunctionHandle>
{
public:
    MTL::Device*      device() const;
    MTL::FunctionType functionType() const;
    MTL::ResourceID   gpuResourceID() const;
    NS::String*       name() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLFunctionHandle;

_MTL_INLINE MTL::FunctionType MTL::FunctionHandle::functionType() const
{
    return _MTL_msg_MTL__FunctionType_functionType((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::FunctionHandle::name() const
{
    return _MTL_msg_NS__Stringp_name((const void*)this, nullptr);
}

_MTL_INLINE MTL::Device* MTL::FunctionHandle::device() const
{
    return _MTL_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL_INLINE MTL::ResourceID MTL::FunctionHandle::gpuResourceID() const
{
    return _MTL_msg_MTL__ResourceID_gpuResourceID((const void*)this, nullptr);
}
