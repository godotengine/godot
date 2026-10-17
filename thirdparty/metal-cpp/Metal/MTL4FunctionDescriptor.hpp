//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal//MTL4FunctionDescriptor.hpp
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

namespace MTL4
{
class FunctionDescriptor;

class FunctionDescriptor : public NS::Copying<FunctionDescriptor>
{
public:
    static FunctionDescriptor* alloc();

    FunctionDescriptor*        init();
};

}
_MTL_INLINE MTL4::FunctionDescriptor* MTL4::FunctionDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::FunctionDescriptor>(_MTL_PRIVATE_CLS(MTL4FunctionDescriptor));
}

_MTL_INLINE MTL4::FunctionDescriptor* MTL4::FunctionDescriptor::init()
{
    return NS::Object::init<MTL4::FunctionDescriptor>();
}
