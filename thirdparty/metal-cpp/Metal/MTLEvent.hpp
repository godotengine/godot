//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLEvent.hpp
//
// Copyright 2020-2025 Apple Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#pragma once

#include "../Foundation/Foundation.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include <cstdint>
#include <dispatch/dispatch.h>

#include <cstdint>
#include <functional>

namespace MTL
{
class Device;
class SharedEvent;
class SharedEventHandle;
class SharedEventListener;

using SharedEventNotificationBlock = void (^)(SharedEvent* pEvent, std::uint64_t value);
using SharedEventNotificationFunction = std::function<void(SharedEvent* pEvent, std::uint64_t value)>;

class Event : public NS::Referencing<Event>
{
public:
    Device*     device() const;

    NS::String* label() const;
    void        setLabel(const NS::String* label);
};
class SharedEventListener : public NS::Referencing<SharedEventListener>
{
public:
    static SharedEventListener* alloc();

    dispatch_queue_t            dispatchQueue() const;

    SharedEventListener*        init();
    SharedEventListener*        init(const dispatch_queue_t dispatchQueue);

    static SharedEventListener* sharedListener();
};
class SharedEvent : public NS::Referencing<SharedEvent, Event>
{
public:
    SharedEventHandle* newSharedEventHandle();

    void               notifyListener(const MTL::SharedEventListener* listener, uint64_t value, const MTL::SharedEventNotificationBlock block);
    void               notifyListener(const MTL::SharedEventListener* listener, uint64_t value, const MTL::SharedEventNotificationFunction& function);

    void               setSignaledValue(uint64_t signaledValue);
    uint64_t           signaledValue() const;
    bool               waitUntilSignaledValue(uint64_t value, uint64_t milliseconds);
};
class SharedEventHandle : public NS::SecureCoding<SharedEventHandle>
{
public:
    static SharedEventHandle* alloc();

    SharedEventHandle*        init();

    NS::String*               label() const;
};

}
_MTL_INLINE MTL::Device* MTL::Event::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE NS::String* MTL::Event::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE void MTL::Event::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE MTL::SharedEventListener* MTL::SharedEventListener::alloc()
{
    return NS::Object::alloc<MTL::SharedEventListener>(_MTL_PRIVATE_CLS(MTLSharedEventListener));
}

_MTL_INLINE dispatch_queue_t MTL::SharedEventListener::dispatchQueue() const
{
    return Object::sendMessage<dispatch_queue_t>(this, _MTL_PRIVATE_SEL(dispatchQueue));
}

_MTL_INLINE MTL::SharedEventListener* MTL::SharedEventListener::init()
{
    return NS::Object::init<MTL::SharedEventListener>();
}

_MTL_INLINE MTL::SharedEventListener* MTL::SharedEventListener::init(const dispatch_queue_t dispatchQueue)
{
    return Object::sendMessage<MTL::SharedEventListener*>(this, _MTL_PRIVATE_SEL(initWithDispatchQueue_), dispatchQueue);
}

_MTL_INLINE MTL::SharedEventListener* MTL::SharedEventListener::sharedListener()
{
    return Object::sendMessage<MTL::SharedEventListener*>(_MTL_PRIVATE_CLS(MTLSharedEventListener), _MTL_PRIVATE_SEL(sharedListener));
}

_MTL_INLINE MTL::SharedEventHandle* MTL::SharedEvent::newSharedEventHandle()
{
    return Object::sendMessage<MTL::SharedEventHandle*>(this, _MTL_PRIVATE_SEL(newSharedEventHandle));
}

_MTL_INLINE void MTL::SharedEvent::notifyListener(const MTL::SharedEventListener* listener, uint64_t value, const MTL::SharedEventNotificationBlock block)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(notifyListener_atValue_block_), listener, value, block);
}

_MTL_INLINE void MTL::SharedEvent::notifyListener(const MTL::SharedEventListener* listener, uint64_t value, const MTL::SharedEventNotificationFunction& function)
{
    __block MTL::SharedEventNotificationFunction callback = function;
    notifyListener(listener, value, ^void(SharedEvent* pEvent, std::uint64_t innerValue) { callback(pEvent, innerValue); });
}

_MTL_INLINE void MTL::SharedEvent::setSignaledValue(uint64_t signaledValue)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSignaledValue_), signaledValue);
}

_MTL_INLINE uint64_t MTL::SharedEvent::signaledValue() const
{
    return Object::sendMessage<uint64_t>(this, _MTL_PRIVATE_SEL(signaledValue));
}

_MTL_INLINE bool MTL::SharedEvent::waitUntilSignaledValue(uint64_t value, uint64_t milliseconds)
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(waitUntilSignaledValue_timeoutMS_), value, milliseconds);
}

_MTL_INLINE MTL::SharedEventHandle* MTL::SharedEventHandle::alloc()
{
    return NS::Object::alloc<MTL::SharedEventHandle>(_MTL_PRIVATE_CLS(MTLSharedEventHandle));
}

_MTL_INLINE MTL::SharedEventHandle* MTL::SharedEventHandle::init()
{
    return NS::Object::init<MTL::SharedEventHandle>();
}

_MTL_INLINE NS::String* MTL::SharedEventHandle::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}
