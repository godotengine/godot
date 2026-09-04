#pragma once

#include "NSDefines.hpp"
#include "NSBlocks.hpp"
#include "NSStructs.hpp"
#include "NSBridge.hpp"
#include "NSObject.hpp"
#include "NSTypes.hpp"
#include "NSRange.hpp"
#include "NSEnumerator.hpp"

namespace NS {
    class Object;
}

namespace NS
{

class Dictionary : public NS::SecureCoding<Dictionary>
{
public:
    static Dictionary* alloc();
    Dictionary*        init() const;

    static NS::Dictionary* dictionary();
    static NS::Dictionary* dictionary(NS::Object* object, NS::Object* key);
    static NS::Dictionary* dictionary(const NS::Object* const * objects, const NS::Object* const * keys, NS::UInteger cnt);

    NS::UInteger          count() const;
    NS::Dictionary*       init(const NS::Object* const * objects, const NS::Object* const * keys, NS::UInteger cnt);
    NS::Dictionary*       init(void* coder);
    template <class _KeyType = Object>
    Enumerator<_KeyType>* keyEnumerator();
    template <class _Object = Object>
    _Object*              object(NS::Object* aKey);

};

} // namespace NS

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_NSDictionary;

_NS_INLINE NS::Dictionary* NS::Dictionary::alloc()
{
    return _NS_msg_NS__Dictionaryp_alloc((const void*)&OBJC_CLASS_$_NSDictionary, nullptr);
}

_NS_INLINE NS::Dictionary* NS::Dictionary::init() const
{
    return _NS_msg_NS__Dictionaryp_init((const void*)this, nullptr);
}

_NS_INLINE NS::Dictionary* NS::Dictionary::dictionary()
{
    return _NS_msg_NS__Dictionaryp_dictionary((const void*)&OBJC_CLASS_$_NSDictionary, nullptr);
}

_NS_INLINE NS::Dictionary* NS::Dictionary::dictionary(NS::Object* object, NS::Object* key)
{
    return _NS_msg_NS__Dictionaryp_dictionaryWithObject_forKey__NS__Objectp_NS__Objectp((const void*)&OBJC_CLASS_$_NSDictionary, nullptr, object, key);
}

_NS_INLINE NS::Dictionary* NS::Dictionary::dictionary(const NS::Object* const * objects, const NS::Object* const * keys, NS::UInteger cnt)
{
    return _NS_msg_NS__Dictionaryp_dictionaryWithObjects_forKeys_count__constNS__Objectpconstp_constNS__Objectpconstp_NS__UInteger((const void*)&OBJC_CLASS_$_NSDictionary, nullptr, objects, keys, cnt);
}

_NS_INLINE NS::UInteger NS::Dictionary::count() const
{
    return _NS_msg_NS__UInteger_count((const void*)this, nullptr);
}

template <class _Object>
_NS_INLINE _Object* NS::Dictionary::object(NS::Object* aKey)
{
    return reinterpret_cast<_Object*>(_NS_msg_NS__Objectp_objectForKey__NS__Objectp((const void*)this, nullptr, aKey));
}

template <class _KeyType>
_NS_INLINE NS::Enumerator<_KeyType>* NS::Dictionary::keyEnumerator()
{
    return reinterpret_cast<NS::Enumerator<_KeyType>*>(_NS_msg_NS__EnumeratorLNS__ObjectGp_keyEnumerator((const void*)this, nullptr));
}

_NS_INLINE NS::Dictionary* NS::Dictionary::init(const NS::Object* const * objects, const NS::Object* const * keys, NS::UInteger cnt)
{
    return _NS_msg_NS__Dictionaryp_initWithObjects_forKeys_count__constNS__Objectpconstp_constNS__Objectpconstp_NS__UInteger((const void*)this, nullptr, objects, keys, cnt);
}

_NS_INLINE NS::Dictionary* NS::Dictionary::init(void* coder)
{
    return _NS_msg_NS__Dictionaryp_initWithCoder__voidp((const void*)this, nullptr, coder);
}
