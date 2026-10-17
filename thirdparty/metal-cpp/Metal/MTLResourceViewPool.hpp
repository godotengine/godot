//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLResourceViewPool.hpp
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
#include "MTLTypes.hpp"

namespace MTL
{
class Device;
class ResourceViewPool;
class ResourceViewPoolDescriptor;

class ResourceViewPoolDescriptor : public NS::Copying<ResourceViewPoolDescriptor>
{
public:
    static ResourceViewPoolDescriptor* alloc();

    ResourceViewPoolDescriptor*        init();

    NS::String*                        label() const;

    NS::UInteger                       resourceViewCount() const;

    void                               setLabel(const NS::String* label);

    void                               setResourceViewCount(NS::UInteger resourceViewCount);
};
class ResourceViewPool : public NS::Referencing<ResourceViewPool>
{
public:
    ResourceID   baseResourceID() const;

    ResourceID   copyResourceViewsFromPool(const MTL::ResourceViewPool* sourcePool, NS::Range sourceRange, NS::UInteger destinationIndex);

    Device*      device() const;

    NS::String*  label() const;

    NS::UInteger resourceViewCount() const;
};

}
_MTL_INLINE MTL::ResourceViewPoolDescriptor* MTL::ResourceViewPoolDescriptor::alloc()
{
    return NS::Object::alloc<MTL::ResourceViewPoolDescriptor>(_MTL_PRIVATE_CLS(MTLResourceViewPoolDescriptor));
}

_MTL_INLINE MTL::ResourceViewPoolDescriptor* MTL::ResourceViewPoolDescriptor::init()
{
    return NS::Object::init<MTL::ResourceViewPoolDescriptor>();
}

_MTL_INLINE NS::String* MTL::ResourceViewPoolDescriptor::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE NS::UInteger MTL::ResourceViewPoolDescriptor::resourceViewCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(resourceViewCount));
}

_MTL_INLINE void MTL::ResourceViewPoolDescriptor::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE void MTL::ResourceViewPoolDescriptor::setResourceViewCount(NS::UInteger resourceViewCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setResourceViewCount_), resourceViewCount);
}

_MTL_INLINE MTL::ResourceID MTL::ResourceViewPool::baseResourceID() const
{
    return Object::sendMessage<MTL::ResourceID>(this, _MTL_PRIVATE_SEL(baseResourceID));
}

_MTL_INLINE MTL::ResourceID MTL::ResourceViewPool::copyResourceViewsFromPool(const MTL::ResourceViewPool* sourcePool, NS::Range sourceRange, NS::UInteger destinationIndex)
{
    return Object::sendMessage<MTL::ResourceID>(this, _MTL_PRIVATE_SEL(copyResourceViewsFromPool_sourceRange_destinationIndex_), sourcePool, sourceRange, destinationIndex);
}

_MTL_INLINE MTL::Device* MTL::ResourceViewPool::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE NS::String* MTL::ResourceViewPool::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE NS::UInteger MTL::ResourceViewPool::resourceViewCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(resourceViewCount));
}
