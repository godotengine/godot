//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLFunctionDescriptor.hpp
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
class FunctionConstantValues;
class FunctionDescriptor;
class IntersectionFunctionDescriptor;

_MTL_OPTIONS(NS::UInteger, FunctionOptions) {
    FunctionOptionNone = 0,
    FunctionOptionCompileToBinary = 1,
    FunctionOptionStoreFunctionInMetalPipelinesScript = 1 << 1,
    FunctionOptionStoreFunctionInMetalScript = 1 << 1,
    FunctionOptionFailOnBinaryArchiveMiss = 1 << 2,
    FunctionOptionPipelineIndependent = 1 << 3,
};

class FunctionDescriptor : public NS::Copying<FunctionDescriptor>
{
public:
    static FunctionDescriptor* alloc();

    NS::Array*                 binaryArchives() const;

    FunctionConstantValues*    constantValues() const;

    static FunctionDescriptor* functionDescriptor();

    FunctionDescriptor*        init();

    NS::String*                name() const;

    FunctionOptions            options() const;

    void                       setBinaryArchives(const NS::Array* binaryArchives);

    void                       setConstantValues(const MTL::FunctionConstantValues* constantValues);

    void                       setName(const NS::String* name);

    void                       setOptions(MTL::FunctionOptions options);

    void                       setSpecializedName(const NS::String* specializedName);
    NS::String*                specializedName() const;
};
class IntersectionFunctionDescriptor : public NS::Copying<IntersectionFunctionDescriptor, FunctionDescriptor>
{
public:
    static IntersectionFunctionDescriptor* alloc();

    IntersectionFunctionDescriptor*        init();
};

}
_MTL_INLINE MTL::FunctionDescriptor* MTL::FunctionDescriptor::alloc()
{
    return NS::Object::alloc<MTL::FunctionDescriptor>(_MTL_PRIVATE_CLS(MTLFunctionDescriptor));
}

_MTL_INLINE NS::Array* MTL::FunctionDescriptor::binaryArchives() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(binaryArchives));
}

_MTL_INLINE MTL::FunctionConstantValues* MTL::FunctionDescriptor::constantValues() const
{
    return Object::sendMessage<MTL::FunctionConstantValues*>(this, _MTL_PRIVATE_SEL(constantValues));
}

_MTL_INLINE MTL::FunctionDescriptor* MTL::FunctionDescriptor::functionDescriptor()
{
    return Object::sendMessage<MTL::FunctionDescriptor*>(_MTL_PRIVATE_CLS(MTLFunctionDescriptor), _MTL_PRIVATE_SEL(functionDescriptor));
}

_MTL_INLINE MTL::FunctionDescriptor* MTL::FunctionDescriptor::init()
{
    return NS::Object::init<MTL::FunctionDescriptor>();
}

_MTL_INLINE NS::String* MTL::FunctionDescriptor::name() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(name));
}

_MTL_INLINE MTL::FunctionOptions MTL::FunctionDescriptor::options() const
{
    return Object::sendMessage<MTL::FunctionOptions>(this, _MTL_PRIVATE_SEL(options));
}

_MTL_INLINE void MTL::FunctionDescriptor::setBinaryArchives(const NS::Array* binaryArchives)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBinaryArchives_), binaryArchives);
}

_MTL_INLINE void MTL::FunctionDescriptor::setConstantValues(const MTL::FunctionConstantValues* constantValues)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setConstantValues_), constantValues);
}

_MTL_INLINE void MTL::FunctionDescriptor::setName(const NS::String* name)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setName_), name);
}

_MTL_INLINE void MTL::FunctionDescriptor::setOptions(MTL::FunctionOptions options)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setOptions_), options);
}

_MTL_INLINE void MTL::FunctionDescriptor::setSpecializedName(const NS::String* specializedName)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSpecializedName_), specializedName);
}

_MTL_INLINE NS::String* MTL::FunctionDescriptor::specializedName() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(specializedName));
}

_MTL_INLINE MTL::IntersectionFunctionDescriptor* MTL::IntersectionFunctionDescriptor::alloc()
{
    return NS::Object::alloc<MTL::IntersectionFunctionDescriptor>(_MTL_PRIVATE_CLS(MTLIntersectionFunctionDescriptor));
}

_MTL_INLINE MTL::IntersectionFunctionDescriptor* MTL::IntersectionFunctionDescriptor::init()
{
    return NS::Object::init<MTL::IntersectionFunctionDescriptor>();
}
