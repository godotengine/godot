//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLFence.hpp
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

namespace MTL
{
class Device;

class Fence : public NS::Referencing<Fence>
{
public:
    Device*     device() const;

    NS::String* label() const;
    void        setLabel(const NS::String* label);
};

}
_MTL_INLINE MTL::Device* MTL::Fence::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE NS::String* MTL::Fence::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE void MTL::Fence::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}
