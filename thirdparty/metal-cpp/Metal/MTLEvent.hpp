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
}
namespace NS {
    class String;
}

namespace MTL
{

class Event;
class SharedEventListener;
class SharedEvent;
class SharedEventHandle;

class Event : public NS::Referencing<Event>
{
public:
    MTL::Device* device() const;
    NS::String*  label() const;
    void         setLabel(NS::String* label);

};

class SharedEventListener : public NS::Referencing<SharedEventListener>
{
public:
    static SharedEventListener* alloc();
    SharedEventListener*        init() const;

    static MTL::SharedEventListener* sharedListener();

    dispatch_queue_t          dispatchQueue() const;
    MTL::SharedEventListener* init(dispatch_queue_t dispatchQueue);

};

class SharedEvent : public NS::Referencing<SharedEvent, MTL::Event>
{
public:
    MTL::SharedEventHandle* newSharedEventHandle();
    void                    notifyListener(MTL::SharedEventListener* listener, uint64_t value, MTL::SharedEventNotificationBlock block);
    void                    notifyListener(MTL::SharedEventListener* listener, uint64_t value, const MTL::SharedEventNotificationFunction& block);
    void                    setSignaledValue(uint64_t signaledValue);
    uint64_t                signaledValue() const;
    bool                    waitUntilSignaledValue(uint64_t value, uint64_t milliseconds);

};

class SharedEventHandle : public NS::SecureCoding<SharedEventHandle>
{
public:
    static SharedEventHandle* alloc();
    SharedEventHandle*        init() const;

    NS::String* label() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLEvent;
extern "C" void *OBJC_CLASS_$_MTLSharedEventListener;
extern "C" void *OBJC_CLASS_$_MTLSharedEvent;
extern "C" void *OBJC_CLASS_$_MTLSharedEventHandle;

_MTL_INLINE MTL::Device* MTL::Event::device() const
{
    return _MTL_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::Event::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::Event::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE MTL::SharedEventListener* MTL::SharedEventListener::alloc()
{
    return _MTL_msg_MTL__SharedEventListenerp_alloc((const void*)&OBJC_CLASS_$_MTLSharedEventListener, nullptr);
}

_MTL_INLINE MTL::SharedEventListener* MTL::SharedEventListener::init() const
{
    return _MTL_msg_MTL__SharedEventListenerp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::SharedEventListener* MTL::SharedEventListener::sharedListener()
{
    return _MTL_msg_MTL__SharedEventListenerp_sharedListener((const void*)&OBJC_CLASS_$_MTLSharedEventListener, nullptr);
}

_MTL_INLINE dispatch_queue_t MTL::SharedEventListener::dispatchQueue() const
{
    return _MTL_msg_dispatch_queue_t_dispatchQueue((const void*)this, nullptr);
}

_MTL_INLINE MTL::SharedEventListener* MTL::SharedEventListener::init(dispatch_queue_t dispatchQueue)
{
    return _MTL_msg_MTL__SharedEventListenerp_initWithDispatchQueue__dispatch_queue_t((const void*)this, nullptr, dispatchQueue);
}

_MTL_INLINE uint64_t MTL::SharedEvent::signaledValue() const
{
    return _MTL_msg_uint64_t_signaledValue((const void*)this, nullptr);
}

_MTL_INLINE void MTL::SharedEvent::setSignaledValue(uint64_t signaledValue)
{
    _MTL_msg_v_setSignaledValue__uint64_t((const void*)this, nullptr, signaledValue);
}

_MTL_INLINE void MTL::SharedEvent::notifyListener(MTL::SharedEventListener* listener, uint64_t value, MTL::SharedEventNotificationBlock block)
{
    _MTL_msg_v_notifyListener_atValue_block__MTL__SharedEventListenerp_uint64_t_MTL__SharedEventNotificationBlock((const void*)this, nullptr, listener, value, block);
}

_MTL_INLINE void MTL::SharedEvent::notifyListener(MTL::SharedEventListener* listener, uint64_t value, const MTL::SharedEventNotificationFunction& block)
{
    __block MTL::SharedEventNotificationFunction blockFunction = block;
    notifyListener(listener, value, ^(MTL::SharedEvent* x0, uint64_t x1) { blockFunction(x0, x1); });
}

_MTL_INLINE MTL::SharedEventHandle* MTL::SharedEvent::newSharedEventHandle()
{
    return _MTL_msg_MTL__SharedEventHandlep_newSharedEventHandle((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::SharedEvent::waitUntilSignaledValue(uint64_t value, uint64_t milliseconds)
{
    return _MTL_msg_bool_waitUntilSignaledValue_timeoutMS__uint64_t_uint64_t((const void*)this, nullptr, value, milliseconds);
}

_MTL_INLINE MTL::SharedEventHandle* MTL::SharedEventHandle::alloc()
{
    return _MTL_msg_MTL__SharedEventHandlep_alloc((const void*)&OBJC_CLASS_$_MTLSharedEventHandle, nullptr);
}

_MTL_INLINE MTL::SharedEventHandle* MTL::SharedEventHandle::init() const
{
    return _MTL_msg_MTL__SharedEventHandlep_init((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::SharedEventHandle::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}
