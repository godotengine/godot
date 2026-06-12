#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class Device;
}
namespace NS {
    class String;
}

namespace MTL4
{

class CommandAllocatorDescriptor;
class CommandAllocator;

class CommandAllocatorDescriptor : public NS::Copying<CommandAllocatorDescriptor>
{
public:
    static CommandAllocatorDescriptor* alloc();
    CommandAllocatorDescriptor*        init() const;

    NS::String* label() const;
    void        setLabel(NS::String* label);

};

class CommandAllocator : public NS::Referencing<CommandAllocator>
{
public:
    uint64_t     allocatedSize();
    MTL::Device* device() const;
    NS::String*  label() const;
    void         reset();

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4CommandAllocatorDescriptor;
extern "C" void *OBJC_CLASS_$_MTL4CommandAllocator;

_MTL4_INLINE MTL4::CommandAllocatorDescriptor* MTL4::CommandAllocatorDescriptor::alloc()
{
    return _MTL4_msg_MTL4__CommandAllocatorDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4CommandAllocatorDescriptor, nullptr);
}

_MTL4_INLINE MTL4::CommandAllocatorDescriptor* MTL4::CommandAllocatorDescriptor::init() const
{
    return _MTL4_msg_MTL4__CommandAllocatorDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE NS::String* MTL4::CommandAllocatorDescriptor::label() const
{
    return _MTL4_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::CommandAllocatorDescriptor::setLabel(NS::String* label)
{
    _MTL4_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL4_INLINE MTL::Device* MTL4::CommandAllocator::device() const
{
    return _MTL4_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL4_INLINE NS::String* MTL4::CommandAllocator::label() const
{
    return _MTL4_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL4_INLINE uint64_t MTL4::CommandAllocator::allocatedSize()
{
    return _MTL4_msg_uint64_t_allocatedSize((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::CommandAllocator::reset()
{
    _MTL4_msg_v_reset((const void*)this, nullptr);
}
