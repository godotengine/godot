//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Foundation/NSLock.hpp
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
#include "NSObject.hpp"
#include "NSPrivate.hpp"
#include "NSTypes.hpp"
#include "NSDate.hpp"

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace NS
{

template <class _Class, class _Base = class Object>
class Locking : public _Base
{
public:
    void lock();
    void unlock();
};

class Condition : public Locking<Condition>
{
public:
    static Condition* alloc();

    Condition*        init();

    void              wait();
    bool              waitUntilDate(Date* pLimit);
    void              signal();
    void              broadcast();
};

} // NS

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

template<class _Class, class _Base /* = NS::Object */>
_NS_INLINE void NS::Locking<_Class, _Base>::lock()
{
    NS::Object::sendMessage<void>(this, _NS_PRIVATE_SEL(lock));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

template<class _Class, class _Base /* = NS::Object */>
_NS_INLINE void NS::Locking<_Class, _Base>::unlock()
{
    NS::Object::sendMessage<void>(this, _NS_PRIVATE_SEL(unlock));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Condition* NS::Condition::alloc()
{
    return NS::Object::alloc<NS::Condition>(_NS_PRIVATE_CLS(NSCondition));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Condition* NS::Condition::init()
{
    return NS::Object::init<NS::Condition>();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE void NS::Condition::wait()
{
    NS::Object::sendMessage<void>(this, _NS_PRIVATE_SEL(wait));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE bool NS::Condition::waitUntilDate(NS::Date* pLimit)
{
    return NS::Object::sendMessage<bool>(this, _NS_PRIVATE_SEL(waitUntilDate_), pLimit);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE void NS::Condition::signal()
{
    NS::Object::sendMessage<void>(this, _NS_PRIVATE_SEL(signal));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE void NS::Condition::broadcast()
{
    NS::Object::sendMessage<void>(this, _NS_PRIVATE_SEL(broadcast));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------