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

_NS_OPTIONS(NS::UInteger, BinarySearchingOptions) {
    BinarySearchingFirstEqual = (1UL << 8),
    BinarySearchingLastEqual = (1UL << 9),
    BinarySearchingInsertionIndex = (1UL << 10),
};


class Array : public NS::SecureCoding<Array>
{
public:
    static Array* alloc();
    Array*        init() const;

    static NS::Array* array();
    static NS::Array* array(NS::Object* anObject);
    static NS::Array* array(const NS::Object* const * objects, NS::UInteger cnt);

    NS::UInteger         count() const;
    NS::Array*           init(const NS::Object* const * objects, NS::UInteger cnt);
    NS::Array*           init(void* coder);
    template <class _Object = Object>
    _Object*             object(NS::UInteger index);
    template <class _Object = Object>
    Enumerator<_Object>* objectEnumerator();

};

} // namespace NS

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_NSArray;

_NS_INLINE NS::Array* NS::Array::alloc()
{
    return _NS_msg_NS__Arrayp_alloc((const void*)&OBJC_CLASS_$_NSArray, nullptr);
}

_NS_INLINE NS::Array* NS::Array::init() const
{
    return _NS_msg_NS__Arrayp_init((const void*)this, nullptr);
}

_NS_INLINE NS::Array* NS::Array::array()
{
    return _NS_msg_NS__Arrayp_array((const void*)&OBJC_CLASS_$_NSArray, nullptr);
}

_NS_INLINE NS::Array* NS::Array::array(NS::Object* anObject)
{
    return _NS_msg_NS__Arrayp_arrayWithObject__NS__Objectp((const void*)&OBJC_CLASS_$_NSArray, nullptr, anObject);
}

_NS_INLINE NS::Array* NS::Array::array(const NS::Object* const * objects, NS::UInteger cnt)
{
    return _NS_msg_NS__Arrayp_arrayWithObjects_count__constNS__Objectpconstp_NS__UInteger((const void*)&OBJC_CLASS_$_NSArray, nullptr, objects, cnt);
}

_NS_INLINE NS::UInteger NS::Array::count() const
{
    return _NS_msg_NS__UInteger_count((const void*)this, nullptr);
}

template <class _Object>
_NS_INLINE _Object* NS::Array::object(NS::UInteger index)
{
    return reinterpret_cast<_Object*>(_NS_msg_NS__Objectp_objectAtIndex__NS__UInteger((const void*)this, nullptr, index));
}

_NS_INLINE NS::Array* NS::Array::init(const NS::Object* const * objects, NS::UInteger cnt)
{
    return _NS_msg_NS__Arrayp_initWithObjects_count__constNS__Objectpconstp_NS__UInteger((const void*)this, nullptr, objects, cnt);
}

_NS_INLINE NS::Array* NS::Array::init(void* coder)
{
    return _NS_msg_NS__Arrayp_initWithCoder__voidp((const void*)this, nullptr, coder);
}

template <class _Object>
_NS_INLINE NS::Enumerator<_Object>* NS::Array::objectEnumerator()
{
    return reinterpret_cast<NS::Enumerator<_Object>*>(_NS_msg_NS__EnumeratorLNS__ObjectGp_objectEnumerator((const void*)this, nullptr));
}
