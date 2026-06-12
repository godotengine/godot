#pragma once

// NS::Enumerator / NS::FastEnumeration — emitted by tools/generate.py
// (used to live verbatim under metal-cpp-apple). The kept-upstream
// version routed through `_NS_PRIVATE_SEL`; this version dispatches
// directly through the linker-synthesized `_objc_msgSend$<sel>` stubs
// so it stops pulling Apple's selector-registration machinery into the
// tree.

#include "NSDefines.hpp"
#include "NSObject.hpp"
#include "NSTypes.hpp"
#include "NSBridge.hpp"

namespace NS
{
class Array;
class FastEnumeration;
template <class _ObjectType> class Enumerator;
} // namespace NS

namespace NS
{
struct FastEnumerationState
{
    unsigned long  state;
    Object**       itemsPtr;
    unsigned long* mutationsPtr;
    unsigned long  extra[5];
} _NS_PACKED;

class FastEnumeration : public Referencing<FastEnumeration>
{
public:
    NS::UInteger countByEnumerating(FastEnumerationState* pState, Object** pBuffer, NS::UInteger len);
};

template <class _ObjectType>
class Enumerator : public Referencing<Enumerator<_ObjectType>, FastEnumeration>
{
public:
    _ObjectType* nextObject();
    class Array* allObjects();
};
} // namespace NS

// --- Inline implementations ---

_NS_INLINE NS::UInteger NS::FastEnumeration::countByEnumerating(
    FastEnumerationState* pState, Object** pBuffer, NS::UInteger len)
{
    return _NS_msg_NS__UInteger_countByEnumeratingWithState_objects_count__voidp_NS__Objectpp_NS__UInteger(
        (const void*)this, nullptr, pState, pBuffer, len);
}

template <class _ObjectType>
_NS_INLINE _ObjectType* NS::Enumerator<_ObjectType>::nextObject()
{
    return reinterpret_cast<_ObjectType*>(
        _NS_msg_voidp_nextObject((const void*)this, nullptr));
}

template <class _ObjectType>
_NS_INLINE NS::Array* NS::Enumerator<_ObjectType>::allObjects()
{
    return _NS_msg_NS__Arrayp_allObjects((const void*)this, nullptr);
}
