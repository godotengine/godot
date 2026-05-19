#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class Device;
}
namespace NS {
    class String;
}

namespace MTL
{

class ResourceViewPoolDescriptor;
class ResourceViewPool;

class ResourceViewPoolDescriptor : public NS::Copying<ResourceViewPoolDescriptor>
{
public:
    static ResourceViewPoolDescriptor* alloc();
    ResourceViewPoolDescriptor*        init() const;

    NS::String*  label() const;
    NS::UInteger resourceViewCount() const;
    void         setLabel(NS::String* label);
    void         setResourceViewCount(NS::UInteger resourceViewCount);

};

class ResourceViewPool : public NS::Referencing<ResourceViewPool>
{
public:
    MTL::ResourceID baseResourceID() const;
    MTL::ResourceID copyResourceViewsFromPool(MTL::ResourceViewPool* sourcePool, NS::Range sourceRange, NS::UInteger destinationIndex);
    MTL::Device*    device() const;
    NS::String*     label() const;
    NS::UInteger    resourceViewCount() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLResourceViewPoolDescriptor;
extern "C" void *OBJC_CLASS_$_MTLResourceViewPool;

_MTL_INLINE MTL::ResourceViewPoolDescriptor* MTL::ResourceViewPoolDescriptor::alloc()
{
    return _MTL_msg_MTL__ResourceViewPoolDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLResourceViewPoolDescriptor, nullptr);
}

_MTL_INLINE MTL::ResourceViewPoolDescriptor* MTL::ResourceViewPoolDescriptor::init() const
{
    return _MTL_msg_MTL__ResourceViewPoolDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::ResourceViewPoolDescriptor::resourceViewCount() const
{
    return _MTL_msg_NS__UInteger_resourceViewCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ResourceViewPoolDescriptor::setResourceViewCount(NS::UInteger resourceViewCount)
{
    _MTL_msg_v_setResourceViewCount__NS__UInteger((const void*)this, nullptr, resourceViewCount);
}

_MTL_INLINE NS::String* MTL::ResourceViewPoolDescriptor::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ResourceViewPoolDescriptor::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE MTL::ResourceID MTL::ResourceViewPool::baseResourceID() const
{
    return _MTL_msg_MTL__ResourceID_baseResourceID((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::ResourceViewPool::resourceViewCount() const
{
    return _MTL_msg_NS__UInteger_resourceViewCount((const void*)this, nullptr);
}

_MTL_INLINE MTL::Device* MTL::ResourceViewPool::device() const
{
    return _MTL_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::ResourceViewPool::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE MTL::ResourceID MTL::ResourceViewPool::copyResourceViewsFromPool(MTL::ResourceViewPool* sourcePool, NS::Range sourceRange, NS::UInteger destinationIndex)
{
    return _MTL_msg_MTL__ResourceID_copyResourceViewsFromPool_sourceRange_destinationIndex__MTL__ResourceViewPoolp_NS__Range_NS__UInteger((const void*)this, nullptr, sourcePool, sourceRange, destinationIndex);
}
