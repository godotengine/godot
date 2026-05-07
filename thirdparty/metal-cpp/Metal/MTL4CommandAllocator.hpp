//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTL4CommandAllocator.hpp
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
class Device;
}

namespace MTL4
{

class CommandAllocatorDescriptor : public NS::Copying<CommandAllocatorDescriptor>
{
public:
    static CommandAllocatorDescriptor* alloc();

    CommandAllocatorDescriptor*        init();

    NS::String*                        label() const;
    void                               setLabel(const NS::String* label);
};

class CommandAllocator : public NS::Referencing<CommandAllocator>
{
public:
    uint64_t     allocatedSize();

    MTL::Device* device() const;

    NS::String*  label() const;

    void         reset();
};

}

_MTL_INLINE MTL4::CommandAllocatorDescriptor* MTL4::CommandAllocatorDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::CommandAllocatorDescriptor>(_MTL_PRIVATE_CLS(MTL4CommandAllocatorDescriptor));
}

_MTL_INLINE MTL4::CommandAllocatorDescriptor* MTL4::CommandAllocatorDescriptor::init()
{
    return NS::Object::init<MTL4::CommandAllocatorDescriptor>();
}

_MTL_INLINE NS::String* MTL4::CommandAllocatorDescriptor::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE void MTL4::CommandAllocatorDescriptor::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE uint64_t MTL4::CommandAllocator::allocatedSize()
{
    return Object::sendMessage<uint64_t>(this, _MTL_PRIVATE_SEL(allocatedSize));
}

_MTL_INLINE MTL::Device* MTL4::CommandAllocator::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE NS::String* MTL4::CommandAllocator::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE void MTL4::CommandAllocator::reset()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(reset));
}
