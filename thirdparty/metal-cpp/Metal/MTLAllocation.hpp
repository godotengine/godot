//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLAllocation.hpp
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
class Allocation : public NS::Referencing<Allocation>
{
public:
    NS::UInteger allocatedSize() const;
};

}
_MTL_INLINE NS::UInteger MTL::Allocation::allocatedSize() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(allocatedSize));
}
