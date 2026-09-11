#pragma once

// NS::Object — the CRTP root. Emitted by tools/generate.py; was formerly
// copied verbatim from Apple's metal-cpp/Foundation/NSObject.hpp.
//
// Directly-named selectors (retain/release/autorelease/retainCount/copy/
// hash/isEqual:/description/debugDescription/init/alloc/
// respondsToSelector:/methodSignatureForSelector:) dispatch through the
// linker-synthesized `_objc_msgSend$<sel>` trampolines — same shape every
// other generated class uses. `sendMessage` / `sendMessageSafe` are kept
// for callers that need to dispatch an arbitrary runtime SEL.

#include "NSDefines.hpp"
#include "NSTypes.hpp"
#include "NSBridge.hpp"

#include <objc/message.h>
#include <objc/runtime.h>

#include <type_traits>

namespace NS
{
class Object;
class String;
class MethodSignature;
} // namespace NS

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
} // namespace NS

// --- Inline implementations ---

template <class _Class, class _Base>
_NS_INLINE _Class* NS::Referencing<_Class, _Base>::retain()
{
    return reinterpret_cast<_Class*>(_NS_msg_voidp_retain((const void*)this, nullptr));
}

template <class _Class, class _Base>
_NS_INLINE void NS::Referencing<_Class, _Base>::release()
{
    _NS_msg_v_release((const void*)this, nullptr);
}

template <class _Class, class _Base>
_NS_INLINE _Class* NS::Referencing<_Class, _Base>::autorelease()
{
    return reinterpret_cast<_Class*>(_NS_msg_voidp_autorelease((const void*)this, nullptr));
}

template <class _Class, class _Base>
_NS_INLINE NS::UInteger NS::Referencing<_Class, _Base>::retainCount() const
{
    return _NS_msg_NS__UInteger_retainCount((const void*)this, nullptr);
}

template <class _Class, class _Base>
_NS_INLINE _Class* NS::Copying<_Class, _Base>::copy() const
{
    return reinterpret_cast<_Class*>(_NS_msg_voidp_copy((const void*)this, nullptr));
}

template <class _Dst>
_NS_INLINE _Dst NS::Object::bridgingCast(const void* pObj)
{
#ifdef __OBJC__
    return (__bridge _Dst)pObj;
#else
    return (_Dst)pObj;
#endif // __OBJC__
}

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

template <>
_NS_INLINE constexpr bool NS::Object::doesRequireMsgSendStret<void>()
{
    return false;
}

// `sendMessage` keeps the upstream architecture-aware dispatch: x86 uses
// objc_msgSend_fpret for floating-point returns and objc_msgSend_stret
// for large structs, arm64 always uses regular objc_msgSend.
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
#endif
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
#endif
    {
        using SendMessageProc = _Ret (*)(const void*, SEL, _Args...);
        const SendMessageProc pProc = reinterpret_cast<SendMessageProc>(&objc_msgSend);
        return (*pProc)(pObj, selector, args...);
    }
}

_NS_INLINE NS::MethodSignature* NS::Object::methodSignatureForSelector(const void* pObj, SEL selector)
{
    return _NS_msg_NS__MethodSignaturep_methodSignatureForSelector__SEL(pObj, nullptr, selector);
}

_NS_INLINE bool NS::Object::respondsToSelector(const void* pObj, SEL selector)
{
    return _NS_msg_bool_respondsToSelector__SEL(pObj, nullptr, selector);
}

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

template <class _Class>
_NS_INLINE _Class* NS::Object::alloc(const char* pClassName)
{
    // objc_lookUpClass returns `Class` (objc_class*). Under ObjC ARC,
    // bridging a Class to a non-retainable pointer needs `__bridge`;
    // in pure C++ translation units the macro expands to nothing.
#if __has_feature(objc_arc)
    const void* cls = (__bridge const void*)objc_lookUpClass(pClassName);
#else
    const void* cls = (const void*)objc_lookUpClass(pClassName);
#endif
    return reinterpret_cast<_Class*>(_NS_msg_voidp_alloc(cls, nullptr));
}

template <class _Class>
_NS_INLINE _Class* NS::Object::alloc(const void* pClass)
{
    return reinterpret_cast<_Class*>(_NS_msg_voidp_alloc(pClass, nullptr));
}

template <class _Class>
_NS_INLINE _Class* NS::Object::init()
{
    return reinterpret_cast<_Class*>(_NS_msg_voidp_init((const void*)this, nullptr));
}

_NS_INLINE NS::UInteger NS::Object::hash() const
{
    return _NS_msg_NS__UInteger_hash((const void*)this, nullptr);
}

_NS_INLINE bool NS::Object::isEqual(const Object* pObject) const
{
    return _NS_msg_bool_isEqual__constNS__Objectp((const void*)this, nullptr, pObject);
}

_NS_INLINE NS::String* NS::Object::description() const
{
    return _NS_msg_NS__Stringp_description((const void*)this, nullptr);
}

_NS_INLINE NS::String* NS::Object::debugDescription() const
{
    return _NS_msg_NS__Stringp_debugDescription((const void*)this, nullptr);
}
