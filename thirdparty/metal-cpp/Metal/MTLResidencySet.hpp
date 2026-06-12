#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class Allocation;
    class Device;
}
namespace NS {
    class Array;
    class String;
}

namespace MTL
{

class ResidencySetDescriptor;
class ResidencySet;

class ResidencySetDescriptor : public NS::Copying<ResidencySetDescriptor>
{
public:
    static ResidencySetDescriptor* alloc();
    ResidencySetDescriptor*        init() const;

    NS::UInteger initialCapacity() const;
    NS::String*  label() const;
    void         setInitialCapacity(NS::UInteger initialCapacity);
    void         setLabel(NS::String* label);

};

class ResidencySet : public NS::Referencing<ResidencySet>
{
public:
    void         addAllocation(MTL::Allocation* allocation);
    void         addAllocations(const MTL::Allocation* const * allocations, NS::UInteger count);
    NS::Array*   allAllocations() const;
    uint64_t     allocatedSize() const;
    NS::UInteger allocationCount() const;
    void         commit();
    bool         containsAllocation(MTL::Allocation* anAllocation);
    MTL::Device* device() const;
    void         endResidency();
    NS::String*  label() const;
    void         removeAllAllocations();
    void         removeAllocation(MTL::Allocation* allocation);
    void         removeAllocations(const MTL::Allocation* const * allocations, NS::UInteger count);
    void         requestResidency();

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLResidencySetDescriptor;
extern "C" void *OBJC_CLASS_$_MTLResidencySet;

_MTL_INLINE MTL::ResidencySetDescriptor* MTL::ResidencySetDescriptor::alloc()
{
    return _MTL_msg_MTL__ResidencySetDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLResidencySetDescriptor, nullptr);
}

_MTL_INLINE MTL::ResidencySetDescriptor* MTL::ResidencySetDescriptor::init() const
{
    return _MTL_msg_MTL__ResidencySetDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::ResidencySetDescriptor::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ResidencySetDescriptor::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE NS::UInteger MTL::ResidencySetDescriptor::initialCapacity() const
{
    return _MTL_msg_NS__UInteger_initialCapacity((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ResidencySetDescriptor::setInitialCapacity(NS::UInteger initialCapacity)
{
    _MTL_msg_v_setInitialCapacity__NS__UInteger((const void*)this, nullptr, initialCapacity);
}

_MTL_INLINE MTL::Device* MTL::ResidencySet::device() const
{
    return _MTL_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::ResidencySet::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE uint64_t MTL::ResidencySet::allocatedSize() const
{
    return _MTL_msg_uint64_t_allocatedSize((const void*)this, nullptr);
}

_MTL_INLINE NS::Array* MTL::ResidencySet::allAllocations() const
{
    return _MTL_msg_NS__Arrayp_allAllocations((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::ResidencySet::allocationCount() const
{
    return _MTL_msg_NS__UInteger_allocationCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ResidencySet::requestResidency()
{
    _MTL_msg_v_requestResidency((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ResidencySet::endResidency()
{
    _MTL_msg_v_endResidency((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ResidencySet::addAllocation(MTL::Allocation* allocation)
{
    _MTL_msg_v_addAllocation__MTL__Allocationp((const void*)this, nullptr, allocation);
}

_MTL_INLINE void MTL::ResidencySet::addAllocations(const MTL::Allocation* const * allocations, NS::UInteger count)
{
    _MTL_msg_v_addAllocations_count__constMTL__Allocationpconstp_NS__UInteger((const void*)this, nullptr, allocations, count);
}

_MTL_INLINE void MTL::ResidencySet::removeAllocation(MTL::Allocation* allocation)
{
    _MTL_msg_v_removeAllocation__MTL__Allocationp((const void*)this, nullptr, allocation);
}

_MTL_INLINE void MTL::ResidencySet::removeAllocations(const MTL::Allocation* const * allocations, NS::UInteger count)
{
    _MTL_msg_v_removeAllocations_count__constMTL__Allocationpconstp_NS__UInteger((const void*)this, nullptr, allocations, count);
}

_MTL_INLINE void MTL::ResidencySet::removeAllAllocations()
{
    _MTL_msg_v_removeAllAllocations((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::ResidencySet::containsAllocation(MTL::Allocation* anAllocation)
{
    return _MTL_msg_bool_containsAllocation__MTL__Allocationp((const void*)this, nullptr, anAllocation);
}

_MTL_INLINE void MTL::ResidencySet::commit()
{
    _MTL_msg_v_commit((const void*)this, nullptr);
}
