//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTL4ArgumentTable.hpp
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
#include "MTLGPUAddress.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include "MTLTypes.hpp"

namespace MTL
{
class Device;
}

namespace MTL4
{
class ArgumentTableDescriptor : public NS::Copying<ArgumentTableDescriptor>
{
public:
    static ArgumentTableDescriptor* alloc();

    ArgumentTableDescriptor*        init();
    bool                            initializeBindings() const;

    NS::String*                     label() const;

    NS::UInteger                    maxBufferBindCount() const;

    NS::UInteger                    maxSamplerStateBindCount() const;

    NS::UInteger                    maxTextureBindCount() const;

    void                            setInitializeBindings(bool initializeBindings);

    void                            setLabel(const NS::String* label);

    void                            setMaxBufferBindCount(NS::UInteger maxBufferBindCount);

    void                            setMaxSamplerStateBindCount(NS::UInteger maxSamplerStateBindCount);

    void                            setMaxTextureBindCount(NS::UInteger maxTextureBindCount);

    void                            setSupportAttributeStrides(bool supportAttributeStrides);
    bool                            supportAttributeStrides() const;
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

}
_MTL_INLINE MTL4::ArgumentTableDescriptor* MTL4::ArgumentTableDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::ArgumentTableDescriptor>(_MTL_PRIVATE_CLS(MTL4ArgumentTableDescriptor));
}

_MTL_INLINE MTL4::ArgumentTableDescriptor* MTL4::ArgumentTableDescriptor::init()
{
    return NS::Object::init<MTL4::ArgumentTableDescriptor>();
}

_MTL_INLINE bool MTL4::ArgumentTableDescriptor::initializeBindings() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(initializeBindings));
}

_MTL_INLINE NS::String* MTL4::ArgumentTableDescriptor::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE NS::UInteger MTL4::ArgumentTableDescriptor::maxBufferBindCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxBufferBindCount));
}

_MTL_INLINE NS::UInteger MTL4::ArgumentTableDescriptor::maxSamplerStateBindCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxSamplerStateBindCount));
}

_MTL_INLINE NS::UInteger MTL4::ArgumentTableDescriptor::maxTextureBindCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxTextureBindCount));
}

_MTL_INLINE void MTL4::ArgumentTableDescriptor::setInitializeBindings(bool initializeBindings)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInitializeBindings_), initializeBindings);
}

_MTL_INLINE void MTL4::ArgumentTableDescriptor::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE void MTL4::ArgumentTableDescriptor::setMaxBufferBindCount(NS::UInteger maxBufferBindCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxBufferBindCount_), maxBufferBindCount);
}

_MTL_INLINE void MTL4::ArgumentTableDescriptor::setMaxSamplerStateBindCount(NS::UInteger maxSamplerStateBindCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxSamplerStateBindCount_), maxSamplerStateBindCount);
}

_MTL_INLINE void MTL4::ArgumentTableDescriptor::setMaxTextureBindCount(NS::UInteger maxTextureBindCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxTextureBindCount_), maxTextureBindCount);
}

_MTL_INLINE void MTL4::ArgumentTableDescriptor::setSupportAttributeStrides(bool supportAttributeStrides)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSupportAttributeStrides_), supportAttributeStrides);
}

_MTL_INLINE bool MTL4::ArgumentTableDescriptor::supportAttributeStrides() const
{
    return Object::sendMessageSafe<bool>(this, _MTL_PRIVATE_SEL(supportAttributeStrides));
}

_MTL_INLINE MTL::Device* MTL4::ArgumentTable::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE NS::String* MTL4::ArgumentTable::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE void MTL4::ArgumentTable::setAddress(MTL::GPUAddress gpuAddress, NS::UInteger bindingIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setAddress_atIndex_), gpuAddress, bindingIndex);
}

_MTL_INLINE void MTL4::ArgumentTable::setAddress(MTL::GPUAddress gpuAddress, NS::UInteger stride, NS::UInteger bindingIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setAddress_attributeStride_atIndex_), gpuAddress, stride, bindingIndex);
}

_MTL_INLINE void MTL4::ArgumentTable::setResource(MTL::ResourceID resourceID, NS::UInteger bindingIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setResource_atBufferIndex_), resourceID, bindingIndex);
}

_MTL_INLINE void MTL4::ArgumentTable::setSamplerState(MTL::ResourceID resourceID, NS::UInteger bindingIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSamplerState_atIndex_), resourceID, bindingIndex);
}

_MTL_INLINE void MTL4::ArgumentTable::setTexture(MTL::ResourceID resourceID, NS::UInteger bindingIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTexture_atIndex_), resourceID, bindingIndex);
}
