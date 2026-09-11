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

class Set : public NS::SecureCoding<Set>
{
public:
    static Set* alloc();
    Set*        init() const;

    NS::UInteger         count() const;
    NS::Set*             init(const NS::Object* const * objects, NS::UInteger cnt);
    NS::Set*             init(void* coder);
    template <class _Object = Object>
    Enumerator<_Object>* objectEnumerator();

};

} // namespace NS

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_NSSet;

_NS_INLINE NS::Set* NS::Set::alloc()
{
    return _NS_msg_NS__Setp_alloc((const void*)&OBJC_CLASS_$_NSSet, nullptr);
}

_NS_INLINE NS::Set* NS::Set::init() const
{
    return _NS_msg_NS__Setp_init((const void*)this, nullptr);
}

_NS_INLINE NS::UInteger NS::Set::count() const
{
    return _NS_msg_NS__UInteger_count((const void*)this, nullptr);
}

template <class _Object>
_NS_INLINE NS::Enumerator<_Object>* NS::Set::objectEnumerator()
{
    return reinterpret_cast<NS::Enumerator<_Object>*>(_NS_msg_NS__EnumeratorLNS__ObjectGp_objectEnumerator((const void*)this, nullptr));
}

_NS_INLINE NS::Set* NS::Set::init(const NS::Object* const * objects, NS::UInteger cnt)
{
    return _NS_msg_NS__Setp_initWithObjects_count__constNS__Objectpconstp_NS__UInteger((const void*)this, nullptr, objects, cnt);
}

_NS_INLINE NS::Set* NS::Set::init(void* coder)
{
    return _NS_msg_NS__Setp_initWithCoder__voidp((const void*)this, nullptr, coder);
}
