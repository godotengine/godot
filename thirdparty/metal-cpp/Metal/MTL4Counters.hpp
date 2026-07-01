#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace NS {
    class Data;
    class String;
}

namespace MTL4
{

_MTL4_ENUM(NS::Integer, CounterHeapType) {
    CounterHeapTypeInvalid = 0,
    CounterHeapTypeTimestamp = 1,
};

_MTL4_ENUM(NS::Integer, TimestampGranularity) {
    TimestampGranularityRelaxed = 0,
    TimestampGranularityPrecise = 1,
};


class CounterHeapDescriptor;
class CounterHeap;

class CounterHeapDescriptor : public NS::Copying<CounterHeapDescriptor>
{
public:
    static CounterHeapDescriptor* alloc();
    CounterHeapDescriptor*        init() const;

    NS::UInteger          count() const;
    void                  setCount(NS::UInteger count);
    void                  setType(MTL4::CounterHeapType type);
    MTL4::CounterHeapType type() const;

};

class CounterHeap : public NS::Referencing<CounterHeap>
{
public:
    NS::UInteger          count() const;
    void                  invalidateCounterRange(NS::Range range);
    NS::String*           label() const;
    NS::Data*             resolveCounterRange(NS::Range range);
    void                  setLabel(NS::String* label);
    MTL4::CounterHeapType type() const;

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4CounterHeapDescriptor;
extern "C" void *OBJC_CLASS_$_MTL4CounterHeap;

_MTL4_INLINE MTL4::CounterHeapDescriptor* MTL4::CounterHeapDescriptor::alloc()
{
    return _MTL4_msg_MTL4__CounterHeapDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4CounterHeapDescriptor, nullptr);
}

_MTL4_INLINE MTL4::CounterHeapDescriptor* MTL4::CounterHeapDescriptor::init() const
{
    return _MTL4_msg_MTL4__CounterHeapDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::CounterHeapType MTL4::CounterHeapDescriptor::type() const
{
    return _MTL4_msg_MTL4__CounterHeapType_type((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::CounterHeapDescriptor::setType(MTL4::CounterHeapType type)
{
    _MTL4_msg_v_setType__MTL4__CounterHeapType((const void*)this, nullptr, type);
}

_MTL4_INLINE NS::UInteger MTL4::CounterHeapDescriptor::count() const
{
    return _MTL4_msg_NS__UInteger_count((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::CounterHeapDescriptor::setCount(NS::UInteger count)
{
    _MTL4_msg_v_setCount__NS__UInteger((const void*)this, nullptr, count);
}

_MTL4_INLINE NS::String* MTL4::CounterHeap::label() const
{
    return _MTL4_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::CounterHeap::setLabel(NS::String* label)
{
    _MTL4_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL4_INLINE NS::UInteger MTL4::CounterHeap::count() const
{
    return _MTL4_msg_NS__UInteger_count((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::CounterHeapType MTL4::CounterHeap::type() const
{
    return _MTL4_msg_MTL4__CounterHeapType_type((const void*)this, nullptr);
}

_MTL4_INLINE NS::Data* MTL4::CounterHeap::resolveCounterRange(NS::Range range)
{
    return _MTL4_msg_NS__Datap_resolveCounterRange__NS__Range((const void*)this, nullptr, range);
}

_MTL4_INLINE void MTL4::CounterHeap::invalidateCounterRange(NS::Range range)
{
    _MTL4_msg_v_invalidateCounterRange__NS__Range((const void*)this, nullptr, range);
}
