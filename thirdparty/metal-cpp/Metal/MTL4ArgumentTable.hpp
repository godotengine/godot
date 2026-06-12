#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTLStructs.hpp"

namespace MTL {
    class Device;
}
namespace NS {
    class String;
}

namespace MTL4
{

class ArgumentTableDescriptor;
class ArgumentTable;

class ArgumentTableDescriptor : public NS::Copying<ArgumentTableDescriptor>
{
public:
    static ArgumentTableDescriptor* alloc();
    ArgumentTableDescriptor*        init() const;

    bool         initializeBindings() const;
    NS::String*  label() const;
    NS::UInteger maxBufferBindCount() const;
    NS::UInteger maxSamplerStateBindCount() const;
    NS::UInteger maxTextureBindCount() const;
    void         setInitializeBindings(bool initializeBindings);
    void         setLabel(NS::String* label);
    void         setMaxBufferBindCount(NS::UInteger maxBufferBindCount);
    void         setMaxSamplerStateBindCount(NS::UInteger maxSamplerStateBindCount);
    void         setMaxTextureBindCount(NS::UInteger maxTextureBindCount);
    void         setSupportAttributeStrides(bool supportAttributeStrides);
    bool         supportAttributeStrides() const;

};

class ArgumentTable : public NS::Referencing<ArgumentTable>
{
public:
    MTL::Device* device() const;
    NS::String*  label() const;
    void         setAddress(MTL::GPUAddress gpuAddress, NS::UInteger bindingIndex);
    void         setAddress(MTL::GPUAddress gpuAddress, NS::UInteger stride, NS::UInteger bindingIndex);
    void         setResource(MTL::ResourceID resourceID, NS::UInteger bindingIndex);
    void         setSamplerState(MTL::ResourceID resourceID, NS::UInteger bindingIndex);
    void         setTexture(MTL::ResourceID resourceID, NS::UInteger bindingIndex);

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4ArgumentTableDescriptor;
extern "C" void *OBJC_CLASS_$_MTL4ArgumentTable;

_MTL4_INLINE MTL4::ArgumentTableDescriptor* MTL4::ArgumentTableDescriptor::alloc()
{
    return _MTL4_msg_MTL4__ArgumentTableDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4ArgumentTableDescriptor, nullptr);
}

_MTL4_INLINE MTL4::ArgumentTableDescriptor* MTL4::ArgumentTableDescriptor::init() const
{
    return _MTL4_msg_MTL4__ArgumentTableDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE NS::UInteger MTL4::ArgumentTableDescriptor::maxBufferBindCount() const
{
    return _MTL4_msg_NS__UInteger_maxBufferBindCount((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::ArgumentTableDescriptor::setMaxBufferBindCount(NS::UInteger maxBufferBindCount)
{
    _MTL4_msg_v_setMaxBufferBindCount__NS__UInteger((const void*)this, nullptr, maxBufferBindCount);
}

_MTL4_INLINE NS::UInteger MTL4::ArgumentTableDescriptor::maxTextureBindCount() const
{
    return _MTL4_msg_NS__UInteger_maxTextureBindCount((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::ArgumentTableDescriptor::setMaxTextureBindCount(NS::UInteger maxTextureBindCount)
{
    _MTL4_msg_v_setMaxTextureBindCount__NS__UInteger((const void*)this, nullptr, maxTextureBindCount);
}

_MTL4_INLINE NS::UInteger MTL4::ArgumentTableDescriptor::maxSamplerStateBindCount() const
{
    return _MTL4_msg_NS__UInteger_maxSamplerStateBindCount((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::ArgumentTableDescriptor::setMaxSamplerStateBindCount(NS::UInteger maxSamplerStateBindCount)
{
    _MTL4_msg_v_setMaxSamplerStateBindCount__NS__UInteger((const void*)this, nullptr, maxSamplerStateBindCount);
}

_MTL4_INLINE bool MTL4::ArgumentTableDescriptor::initializeBindings() const
{
    return _MTL4_msg_bool_initializeBindings((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::ArgumentTableDescriptor::setInitializeBindings(bool initializeBindings)
{
    _MTL4_msg_v_setInitializeBindings__bool((const void*)this, nullptr, initializeBindings);
}

_MTL4_INLINE bool MTL4::ArgumentTableDescriptor::supportAttributeStrides() const
{
    return _MTL4_msg_bool_supportAttributeStrides((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::ArgumentTableDescriptor::setSupportAttributeStrides(bool supportAttributeStrides)
{
    _MTL4_msg_v_setSupportAttributeStrides__bool((const void*)this, nullptr, supportAttributeStrides);
}

_MTL4_INLINE NS::String* MTL4::ArgumentTableDescriptor::label() const
{
    return _MTL4_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::ArgumentTableDescriptor::setLabel(NS::String* label)
{
    _MTL4_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL4_INLINE MTL::Device* MTL4::ArgumentTable::device() const
{
    return _MTL4_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL4_INLINE NS::String* MTL4::ArgumentTable::label() const
{
    return _MTL4_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::ArgumentTable::setAddress(MTL::GPUAddress gpuAddress, NS::UInteger bindingIndex)
{
    _MTL4_msg_v_setAddress_atIndex__MTL__GPUAddress_NS__UInteger((const void*)this, nullptr, gpuAddress, bindingIndex);
}

_MTL4_INLINE void MTL4::ArgumentTable::setAddress(MTL::GPUAddress gpuAddress, NS::UInteger stride, NS::UInteger bindingIndex)
{
    _MTL4_msg_v_setAddress_attributeStride_atIndex__MTL__GPUAddress_NS__UInteger_NS__UInteger((const void*)this, nullptr, gpuAddress, stride, bindingIndex);
}

_MTL4_INLINE void MTL4::ArgumentTable::setResource(MTL::ResourceID resourceID, NS::UInteger bindingIndex)
{
    _MTL4_msg_v_setResource_atBufferIndex__MTL__ResourceID_NS__UInteger((const void*)this, nullptr, resourceID, bindingIndex);
}

_MTL4_INLINE void MTL4::ArgumentTable::setTexture(MTL::ResourceID resourceID, NS::UInteger bindingIndex)
{
    _MTL4_msg_v_setTexture_atIndex__MTL__ResourceID_NS__UInteger((const void*)this, nullptr, resourceID, bindingIndex);
}

_MTL4_INLINE void MTL4::ArgumentTable::setSamplerState(MTL::ResourceID resourceID, NS::UInteger bindingIndex)
{
    _MTL4_msg_v_setSamplerState_atIndex__MTL__ResourceID_NS__UInteger((const void*)this, nullptr, resourceID, bindingIndex);
}
