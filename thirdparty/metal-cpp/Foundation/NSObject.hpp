//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Foundation/NSObject.hpp
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
#include "NSPrivate.hpp"
#include "NSTypes.hpp"

#include <objc/message.h>
#include <objc/runtime.h>

#include <type_traits>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace NS
{
template <class _Class, class _Base = class Object>
class _NS_EXPORT Referencing : public _Base
{
public:
    _Class*  retain();
    void     release();

    _Class*  autorelease();

    UInteger retainCount() const;
};

template <class _Class, class _Base = class Object>
class Copying : public Referencing<_Class, _Base>
{
public:
    _Class* copy() const;
};

template <class _Class, class _Base = class Object>
class SecureCoding : public Referencing<_Class, _Base>
{
};

class Object : public Referencing<Object, objc_object>
{
public:
    UInteger      hash() const;
    bool          isEqual(const Object* pObject) const;

    class String* description() const;
    class String* debugDescription() const;

    template <typename _Ret, typename... _Args>
    static _Ret sendMessage(const void* pObj, SEL selector, _Args... args);

    template <typename _Ret, typename... _Args>
    static _Ret sendMessageSafe(const void* pObj, SEL selector, _Args... args);

protected:
    friend class Referencing<Object, objc_object>;

    template <class _Class>
    static _Class* alloc(const char* pClassName);
    template <class _Class>
    static _Class* alloc(const void* pClass);
    template <class _Class>
    _Class* init();

    template <class _Dst>
    static _Dst                   bridgingCast(const void* pObj);
    static class MethodSignature* methodSignatureForSelector(const void* pObj, SEL selector);
    static bool                   respondsToSelector(const void* pObj, SEL selector);
    template <typename _Type>
    static constexpr bool doesRequireMsgSendStret();

private:
    Object() = delete;
    Object(const Object&) = delete;
    ~Object() = delete;

    Object& operator=(const Object&) = delete;
};
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

template <class _Class, class _Base /* = Object */>
_NS_INLINE _Class* NS::Referencing<_Class, _Base>::retain()
{
    return Object::sendMessage<_Class*>(this, _NS_PRIVATE_SEL(retain));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

template <class _Class, class _Base /* = Object */>
_NS_INLINE void NS::Referencing<_Class, _Base>::release()
{
    Object::sendMessage<void>(this, _NS_PRIVATE_SEL(release));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

template <class _Class, class _Base /* = Object */>
_NS_INLINE _Class* NS::Referencing<_Class, _Base>::autorelease()
{
    return Object::sendMessage<_Class*>(this, _NS_PRIVATE_SEL(autorelease));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

template <class _Class, class _Base /* = Object */>
_NS_INLINE NS::UInteger NS::Referencing<_Class, _Base>::retainCount() const
{
    return Object::sendMessage<UInteger>(this, _NS_PRIVATE_SEL(retainCount));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

template <class _Class, class _Base /* = Object */>
_NS_INLINE _Class* NS::Copying<_Class, _Base>::copy() const
{
    return Object::sendMessage<_Class*>(this, _NS_PRIVATE_SEL(copy));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

template <class _Dst>
_NS_INLINE _Dst NS::Object::bridgingCast(const void* pObj)
{
#ifdef __OBJC__
    return (__bridge _Dst)pObj;
#else
    return (_Dst)pObj;
#endif // __OBJC__
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename _Type>
_NS_INLINE constexpr bool NS::Object::doesRequireMsgSendStret()
{
#if (defined(__i386__) || defined(__x86_64__))
    constexpr size_t kStructLimit = (sizeof(std::uintptr_t) << 1);

    return sizeof(_Type) > kStructLimit;
#elif defined(__arm64__)
    return false;
#elif defined(__arm__)
    constexpr size_t kStructLimit = sizeof(std::uintptr_t);

    return std::is_class_v<_Type> && (sizeof(_Type) > kStructLimit);
#else
#error "Unsupported architecture!"
#endif
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

template <>
_NS_INLINE constexpr bool NS::Object::doesRequireMsgSendStret<void>()
{
    return false;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename _Ret, typename... _Args>
_NS_INLINE _Ret NS::Object::sendMessage(const void* pObj, SEL selector, _Args... args)
{
#if (defined(__i386__) || defined(__x86_64__))
    if constexpr (std::is_floating_point<_Ret>())
    {
        using SendMessageProcFpret = _Ret (*)(const void*, SEL, _Args...);

        const SendMessageProcFpret pProc = reinterpret_cast<SendMessageProcFpret>(&objc_msgSend_fpret);

        return (*pProc)(pObj, selector, args...);
    }
    else
#endif // ( defined( __i386__ )  || defined( __x86_64__ )  )
#if !defined(__arm64__)
        if constexpr (doesRequireMsgSendStret<_Ret>())
    {
        using SendMessageProcStret = void (*)(_Ret*, const void*, SEL, _Args...);

        const SendMessageProcStret pProc = reinterpret_cast<SendMessageProcStret>(&objc_msgSend_stret);
        _Ret                       ret;

        (*pProc)(&ret, pObj, selector, args...);

        return ret;
    }
    else
#endif // !defined( __arm64__ )
    {
        using SendMessageProc = _Ret (*)(const void*, SEL, _Args...);

        const SendMessageProc pProc = reinterpret_cast<SendMessageProc>(&objc_msgSend);

        return (*pProc)(pObj, selector, args...);
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::MethodSignature* NS::Object::methodSignatureForSelector(const void* pObj, SEL selector)
{
    return sendMessage<MethodSignature*>(pObj, _NS_PRIVATE_SEL(methodSignatureForSelector_), selector);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE bool NS::Object::respondsToSelector(const void* pObj, SEL selector)
{
    return sendMessage<bool>(pObj, _NS_PRIVATE_SEL(respondsToSelector_), selector);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename _Ret, typename... _Args>
_NS_INLINE _Ret NS::Object::sendMessageSafe(const void* pObj, SEL selector, _Args... args)
{
    if ((respondsToSelector(pObj, selector)) || (nullptr != methodSignatureForSelector(pObj, selector)))
    {
        return sendMessage<_Ret>(pObj, selector, args...);
    }

    if constexpr (!std::is_void<_Ret>::value)
    {
        return _Ret(0);
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

template <class _Class>
_NS_INLINE _Class* NS::Object::alloc(const char* pClassName)
{
    return sendMessage<_Class*>(objc_lookUpClass(pClassName), _NS_PRIVATE_SEL(alloc));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

template <class _Class>
_NS_INLINE _Class* NS::Object::alloc(const void* pClass)
{
    return sendMessage<_Class*>(pClass, _NS_PRIVATE_SEL(alloc));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

template <class _Class>
_NS_INLINE _Class* NS::Object::init()
{
    return sendMessage<_Class*>(this, _NS_PRIVATE_SEL(init));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::UInteger NS::Object::hash() const
{
    return sendMessage<UInteger>(this, _NS_PRIVATE_SEL(hash));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE bool NS::Object::isEqual(const Object* pObject) const
{
    return sendMessage<bool>(this, _NS_PRIVATE_SEL(isEqual_), pObject);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::Object::description() const
{
    return sendMessage<String*>(this, _NS_PRIVATE_SEL(description));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::Object::debugDescription() const
{
    return sendMessageSafe<String*>(this, _NS_PRIVATE_SEL(debugDescription));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
