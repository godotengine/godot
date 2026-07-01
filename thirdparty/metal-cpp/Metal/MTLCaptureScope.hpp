#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class CommandQueue;
    class Device;
}
namespace NS {
    class String;
}

namespace MTL
{

class CaptureScope : public NS::Referencing<CaptureScope>
{
public:
    void               beginScope();
    MTL::CommandQueue* commandQueue() const;
    MTL::Device*       device() const;
    void               endScope();
    NS::String*        label() const;
    void               setLabel(NS::String* label);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLCaptureScope;

_MTL_INLINE NS::String* MTL::CaptureScope::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CaptureScope::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE MTL::Device* MTL::CaptureScope::device() const
{
    return _MTL_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL_INLINE MTL::CommandQueue* MTL::CaptureScope::commandQueue() const
{
    return _MTL_msg_MTL__CommandQueuep_commandQueue((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CaptureScope::beginScope()
{
    _MTL_msg_v_beginScope((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CaptureScope::endScope()
{
    _MTL_msg_v_endScope((const void*)this, nullptr);
}
