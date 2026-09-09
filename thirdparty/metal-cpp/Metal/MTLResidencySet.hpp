//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLResidencySet.hpp
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

namespace MTL
{
class Allocation;
class Device;
class ResidencySetDescriptor;

class ResidencySetDescriptor : public NS::Copying<ResidencySetDescriptor>
{
public:
    static ResidencySetDescriptor* alloc();

    ResidencySetDescriptor*        init();
    NS::UInteger                   initialCapacity() const;

    NS::String*                    label() const;

    void                           setInitialCapacity(NS::UInteger initialCapacity);

    void                           setLabel(const NS::String* label);
};
class ResidencySet : public NS::Referencing<ResidencySet>
{
public:
    void         addAllocation(const MTL::Allocation* allocation);
    void         addAllocations(const MTL::Allocation* const allocations[], NS::UInteger count);

    NS::Array*   allAllocations() const;

    uint64_t     allocatedSize() const;

    NS::UInteger allocationCount() const;

    void         commit();

    bool         containsAllocation(const MTL::Allocation* anAllocation);

    Device*      device() const;

    void         endResidency();

    NS::String*  label() const;

    void         removeAllAllocations();

    void         removeAllocation(const MTL::Allocation* allocation);
    void         removeAllocations(const MTL::Allocation* const allocations[], NS::UInteger count);

    void         requestResidency();
};

}
_MTL_INLINE MTL::ResidencySetDescriptor* MTL::ResidencySetDescriptor::alloc()
{
    return NS::Object::alloc<MTL::ResidencySetDescriptor>(_MTL_PRIVATE_CLS(MTLResidencySetDescriptor));
}

_MTL_INLINE MTL::ResidencySetDescriptor* MTL::ResidencySetDescriptor::init()
{
    return NS::Object::init<MTL::ResidencySetDescriptor>();
}

_MTL_INLINE NS::UInteger MTL::ResidencySetDescriptor::initialCapacity() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(initialCapacity));
}

_MTL_INLINE NS::String* MTL::ResidencySetDescriptor::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE void MTL::ResidencySetDescriptor::setInitialCapacity(NS::UInteger initialCapacity)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInitialCapacity_), initialCapacity);
}

_MTL_INLINE void MTL::ResidencySetDescriptor::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE void MTL::ResidencySet::addAllocation(const MTL::Allocation* allocation)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(addAllocation_), allocation);
}

_MTL_INLINE void MTL::ResidencySet::addAllocations(const MTL::Allocation* const allocations[], NS::UInteger count)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(addAllocations_count_), allocations, count);
}

_MTL_INLINE NS::Array* MTL::ResidencySet::allAllocations() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(allAllocations));
}

_MTL_INLINE uint64_t MTL::ResidencySet::allocatedSize() const
{
    return Object::sendMessage<uint64_t>(this, _MTL_PRIVATE_SEL(allocatedSize));
}

_MTL_INLINE NS::UInteger MTL::ResidencySet::allocationCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(allocationCount));
}

_MTL_INLINE void MTL::ResidencySet::commit()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(commit));
}

_MTL_INLINE bool MTL::ResidencySet::containsAllocation(const MTL::Allocation* anAllocation)
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(containsAllocation_), anAllocation);
}

_MTL_INLINE MTL::Device* MTL::ResidencySet::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE void MTL::ResidencySet::endResidency()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(endResidency));
}

_MTL_INLINE NS::String* MTL::ResidencySet::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE void MTL::ResidencySet::removeAllAllocations()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(removeAllAllocations));
}

_MTL_INLINE void MTL::ResidencySet::removeAllocation(const MTL::Allocation* allocation)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(removeAllocation_), allocation);
}

_MTL_INLINE void MTL::ResidencySet::removeAllocations(const MTL::Allocation* const allocations[], NS::UInteger count)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(removeAllocations_count_), allocations, count);
}

_MTL_INLINE void MTL::ResidencySet::requestResidency()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(requestResidency));
}
