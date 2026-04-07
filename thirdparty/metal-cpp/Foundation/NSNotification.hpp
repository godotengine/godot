//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Foundation/NSNotification.hpp
//
// Copyright 2020-2024 Apple Inc.
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

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#include "NSDefines.hpp"
#include "NSDictionary.hpp"
#include "NSObject.hpp"
#include "NSString.hpp"
#include "NSTypes.hpp"
#include <functional>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace NS
{
using NotificationName = class String*;

class Notification : public NS::Referencing<Notification>
{
public:
    NS::String*     name() const;
    NS::Object*     object() const;
    NS::Dictionary* userInfo() const;
};

using ObserverBlock = void(^)(Notification*);
using ObserverFunction = std::function<void(Notification*)>;

class NotificationCenter : public NS::Referencing<NotificationCenter>
{
    public:
        static class NotificationCenter* defaultCenter();
        Object* addObserver(NotificationName name, Object* pObj, void* pQueue, ObserverBlock block);
        Object* addObserver(NotificationName name, Object* pObj, void* pQueue, ObserverFunction &handler);
        void removeObserver(Object* pObserver);

};
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::Notification::name() const
{
    return Object::sendMessage<NS::String*>(this, _NS_PRIVATE_SEL(name));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Object* NS::Notification::object() const
{
    return Object::sendMessage<NS::Object*>(this, _NS_PRIVATE_SEL(object));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Dictionary* NS::Notification::userInfo() const
{
    return Object::sendMessage<NS::Dictionary*>(this, _NS_PRIVATE_SEL(userInfo));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::NotificationCenter* NS::NotificationCenter::defaultCenter()
{
    return NS::Object::sendMessage<NS::NotificationCenter*>(_NS_PRIVATE_CLS(NSNotificationCenter), _NS_PRIVATE_SEL(defaultCenter));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Object* NS::NotificationCenter::addObserver(NS::NotificationName name, Object* pObj, void* pQueue, NS::ObserverBlock block)
{
    return NS::Object::sendMessage<Object*>(this, _NS_PRIVATE_SEL(addObserverName_object_queue_block_), name, pObj, pQueue, block);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Object* NS::NotificationCenter::addObserver(NS::NotificationName name, Object* pObj, void* pQueue, NS::ObserverFunction &handler)
{
    __block ObserverFunction blockFunction = handler;

    return addObserver(name, pObj, pQueue, ^(NS::Notification* pNotif) {blockFunction(pNotif);});
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE void NS::NotificationCenter::removeObserver(Object* pObserver)
{
    return NS::Object::sendMessage<void>(this, _NS_PRIVATE_SEL(removeObserver_), pObserver);
}

