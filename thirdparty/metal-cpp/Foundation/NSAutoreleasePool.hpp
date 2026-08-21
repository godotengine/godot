#pragma once

#include "NSDefines.hpp"
#include "NSBlocks.hpp"
#include "NSStructs.hpp"
#include "NSBridge.hpp"
#include "NSObject.hpp"
#include "NSTypes.hpp"
#include "NSRange.hpp"

namespace NS {
    class Object;
}

namespace NS
{

class AutoreleasePool : public NS::Referencing<AutoreleasePool>
{
public:
    static AutoreleasePool* alloc();
    AutoreleasePool*        init() const;

    static void addObject(NS::Object* anObject);

    void drain();

};

} // namespace NS

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_NSAutoreleasePool;

_NS_INLINE NS::AutoreleasePool* NS::AutoreleasePool::alloc()
{
    return _NS_msg_NS__AutoreleasePoolp_alloc((const void*)&OBJC_CLASS_$_NSAutoreleasePool, nullptr);
}

_NS_INLINE NS::AutoreleasePool* NS::AutoreleasePool::init() const
{
    return _NS_msg_NS__AutoreleasePoolp_init((const void*)this, nullptr);
}

_NS_INLINE void NS::AutoreleasePool::addObject(NS::Object* anObject)
{
    _NS_msg_v_addObject__NS__Objectp((const void*)&OBJC_CLASS_$_NSAutoreleasePool, nullptr, anObject);
}

_NS_INLINE void NS::AutoreleasePool::drain()
{
    _NS_msg_v_drain((const void*)this, nullptr);
}
