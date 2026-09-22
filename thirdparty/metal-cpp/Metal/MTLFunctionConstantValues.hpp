//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLFunctionConstantValues.hpp
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
#include "MTLDataType.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"

namespace MTL
{
class FunctionConstantValues;

class FunctionConstantValues : public NS::Copying<FunctionConstantValues>
{
public:
    static FunctionConstantValues* alloc();

    FunctionConstantValues*        init();

    void                           reset();

    void                           setConstantValue(const void* value, MTL::DataType type, NS::UInteger index);
    void                           setConstantValue(const void* value, MTL::DataType type, const NS::String* name);
    void                           setConstantValues(const void* values, MTL::DataType type, NS::Range range);
};

}
_MTL_INLINE MTL::FunctionConstantValues* MTL::FunctionConstantValues::alloc()
{
    return NS::Object::alloc<MTL::FunctionConstantValues>(_MTL_PRIVATE_CLS(MTLFunctionConstantValues));
}

_MTL_INLINE MTL::FunctionConstantValues* MTL::FunctionConstantValues::init()
{
    return NS::Object::init<MTL::FunctionConstantValues>();
}

_MTL_INLINE void MTL::FunctionConstantValues::reset()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(reset));
}

_MTL_INLINE void MTL::FunctionConstantValues::setConstantValue(const void* value, MTL::DataType type, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setConstantValue_type_atIndex_), value, type, index);
}

_MTL_INLINE void MTL::FunctionConstantValues::setConstantValue(const void* value, MTL::DataType type, const NS::String* name)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setConstantValue_type_withName_), value, type, name);
}

_MTL_INLINE void MTL::FunctionConstantValues::setConstantValues(const void* values, MTL::DataType type, NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setConstantValues_type_withRange_), values, type, range);
}
