//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTL4Counters.hpp
//
// Copyright 2020-2025 Apple Inc.
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

#include "../Foundation/Foundation.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include <cstdint>

#include <cstdint>

namespace MTL4
{
class CounterHeapDescriptor;
_MTL_ENUM(NS::Integer, CounterHeapType) {
    CounterHeapTypeInvalid,
    CounterHeapTypeTimestamp,
};

_MTL_ENUM(NS::Integer, TimestampGranularity) {
    TimestampGranularityRelaxed = 0,
    TimestampGranularityPrecise = 1,
};

struct TimestampHeapEntry
{
    uint64_t timestamp;
} _MTL_PACKED;

class CounterHeapDescriptor : public NS::Copying<CounterHeapDescriptor>
{
public:
    static CounterHeapDescriptor* alloc();

    NS::UInteger                  count() const;

    CounterHeapDescriptor*        init();

    void                          setCount(NS::UInteger count);

    void                          setType(MTL4::CounterHeapType type);
    CounterHeapType               type() const;
};
class CounterHeap : public NS::Referencing<CounterHeap>
{
public:
    NS::UInteger    count() const;
    void            invalidateCounterRange(NS::Range range);

    NS::String*     label() const;

    NS::Data*       resolveCounterRange(NS::Range range);

    void            setLabel(const NS::String* label);

    CounterHeapType type() const;
};

}

_MTL_INLINE MTL4::CounterHeapDescriptor* MTL4::CounterHeapDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::CounterHeapDescriptor>(_MTL_PRIVATE_CLS(MTL4CounterHeapDescriptor));
}

_MTL_INLINE NS::UInteger MTL4::CounterHeapDescriptor::count() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(count));
}

_MTL_INLINE MTL4::CounterHeapDescriptor* MTL4::CounterHeapDescriptor::init()
{
    return NS::Object::init<MTL4::CounterHeapDescriptor>();
}

_MTL_INLINE void MTL4::CounterHeapDescriptor::setCount(NS::UInteger count)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCount_), count);
}

_MTL_INLINE void MTL4::CounterHeapDescriptor::setType(MTL4::CounterHeapType type)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setType_), type);
}

_MTL_INLINE MTL4::CounterHeapType MTL4::CounterHeapDescriptor::type() const
{
    return Object::sendMessage<MTL4::CounterHeapType>(this, _MTL_PRIVATE_SEL(type));
}

_MTL_INLINE NS::UInteger MTL4::CounterHeap::count() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(count));
}

_MTL_INLINE void MTL4::CounterHeap::invalidateCounterRange(NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(invalidateCounterRange_), range);
}

_MTL_INLINE NS::String* MTL4::CounterHeap::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE NS::Data* MTL4::CounterHeap::resolveCounterRange(NS::Range range)
{
    return Object::sendMessage<NS::Data*>(this, _MTL_PRIVATE_SEL(resolveCounterRange_), range);
}

_MTL_INLINE void MTL4::CounterHeap::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE MTL4::CounterHeapType MTL4::CounterHeap::type() const
{
    return Object::sendMessage<MTL4::CounterHeapType>(this, _MTL_PRIVATE_SEL(type));
}
