//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLVisibleFunctionTable.hpp
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
#include "MTLResource.hpp"
#include "MTLTypes.hpp"

namespace MTL
{
class FunctionHandle;
class VisibleFunctionTableDescriptor;

class VisibleFunctionTableDescriptor : public NS::Copying<VisibleFunctionTableDescriptor>
{
public:
    static VisibleFunctionTableDescriptor* alloc();

    NS::UInteger                           functionCount() const;

    VisibleFunctionTableDescriptor*        init();

    void                                   setFunctionCount(NS::UInteger functionCount);

    static VisibleFunctionTableDescriptor* visibleFunctionTableDescriptor();
};
class VisibleFunctionTable : public NS::Referencing<VisibleFunctionTable, Resource>
{
public:
    ResourceID gpuResourceID() const;

    void       setFunction(const MTL::FunctionHandle* function, NS::UInteger index);
    void       setFunctions(const MTL::FunctionHandle* const functions[], NS::Range range);
};

}
_MTL_INLINE MTL::VisibleFunctionTableDescriptor* MTL::VisibleFunctionTableDescriptor::alloc()
{
    return NS::Object::alloc<MTL::VisibleFunctionTableDescriptor>(_MTL_PRIVATE_CLS(MTLVisibleFunctionTableDescriptor));
}

_MTL_INLINE NS::UInteger MTL::VisibleFunctionTableDescriptor::functionCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(functionCount));
}

_MTL_INLINE MTL::VisibleFunctionTableDescriptor* MTL::VisibleFunctionTableDescriptor::init()
{
    return NS::Object::init<MTL::VisibleFunctionTableDescriptor>();
}

_MTL_INLINE void MTL::VisibleFunctionTableDescriptor::setFunctionCount(NS::UInteger functionCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFunctionCount_), functionCount);
}

_MTL_INLINE MTL::VisibleFunctionTableDescriptor* MTL::VisibleFunctionTableDescriptor::visibleFunctionTableDescriptor()
{
    return Object::sendMessage<MTL::VisibleFunctionTableDescriptor*>(_MTL_PRIVATE_CLS(MTLVisibleFunctionTableDescriptor), _MTL_PRIVATE_SEL(visibleFunctionTableDescriptor));
}

_MTL_INLINE MTL::ResourceID MTL::VisibleFunctionTable::gpuResourceID() const
{
    return Object::sendMessage<MTL::ResourceID>(this, _MTL_PRIVATE_SEL(gpuResourceID));
}

_MTL_INLINE void MTL::VisibleFunctionTable::setFunction(const MTL::FunctionHandle* function, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFunction_atIndex_), function, index);
}

_MTL_INLINE void MTL::VisibleFunctionTable::setFunctions(const MTL::FunctionHandle* const functions[], NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFunctions_withRange_), functions, range);
}
