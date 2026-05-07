//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTL4PipelineState.hpp
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
#include "MTLPipeline.hpp"
#include "MTLPrivate.hpp"

namespace MTL4
{
class PipelineDescriptor;
class PipelineOptions;
_MTL_ENUM(NS::Integer, AlphaToOneState) {
    AlphaToOneStateDisabled = 0,
    AlphaToOneStateEnabled = 1,
};

_MTL_ENUM(NS::Integer, AlphaToCoverageState) {
    AlphaToCoverageStateDisabled = 0,
    AlphaToCoverageStateEnabled = 1,
};

_MTL_ENUM(NS::Integer, BlendState) {
    BlendStateDisabled = 0,
    BlendStateEnabled = 1,
    BlendStateUnspecialized = 2,
};

_MTL_ENUM(NS::Integer, IndirectCommandBufferSupportState) {
    IndirectCommandBufferSupportStateDisabled = 0,
    IndirectCommandBufferSupportStateEnabled = 1,
};

_MTL_OPTIONS(NS::UInteger, ShaderReflection) {
    ShaderReflectionNone = 0,
    ShaderReflectionBindingInfo = 1,
    ShaderReflectionBufferTypeInfo = 1 << 1,
};

class PipelineOptions : public NS::Copying<PipelineOptions>
{
public:
    static PipelineOptions* alloc();

    PipelineOptions*        init();

    void                    setShaderReflection(MTL4::ShaderReflection shaderReflection);

    void                    setShaderValidation(MTL::ShaderValidation shaderValidation);

    ShaderReflection        shaderReflection() const;

    MTL::ShaderValidation   shaderValidation() const;
};
class PipelineDescriptor : public NS::Copying<PipelineDescriptor>
{
public:
    static PipelineDescriptor* alloc();

    PipelineDescriptor*        init();

    NS::String*                label() const;

    PipelineOptions*           options() const;

    void                       setLabel(const NS::String* label);

    void                       setOptions(const MTL4::PipelineOptions* options);
};

}
_MTL_INLINE MTL4::PipelineOptions* MTL4::PipelineOptions::alloc()
{
    return NS::Object::alloc<MTL4::PipelineOptions>(_MTL_PRIVATE_CLS(MTL4PipelineOptions));
}

_MTL_INLINE MTL4::PipelineOptions* MTL4::PipelineOptions::init()
{
    return NS::Object::init<MTL4::PipelineOptions>();
}

_MTL_INLINE void MTL4::PipelineOptions::setShaderReflection(MTL4::ShaderReflection shaderReflection)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setShaderReflection_), shaderReflection);
}

_MTL_INLINE void MTL4::PipelineOptions::setShaderValidation(MTL::ShaderValidation shaderValidation)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setShaderValidation_), shaderValidation);
}

_MTL_INLINE MTL4::ShaderReflection MTL4::PipelineOptions::shaderReflection() const
{
    return Object::sendMessage<MTL4::ShaderReflection>(this, _MTL_PRIVATE_SEL(shaderReflection));
}

_MTL_INLINE MTL::ShaderValidation MTL4::PipelineOptions::shaderValidation() const
{
    return Object::sendMessage<MTL::ShaderValidation>(this, _MTL_PRIVATE_SEL(shaderValidation));
}

_MTL_INLINE MTL4::PipelineDescriptor* MTL4::PipelineDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::PipelineDescriptor>(_MTL_PRIVATE_CLS(MTL4PipelineDescriptor));
}

_MTL_INLINE MTL4::PipelineDescriptor* MTL4::PipelineDescriptor::init()
{
    return NS::Object::init<MTL4::PipelineDescriptor>();
}

_MTL_INLINE NS::String* MTL4::PipelineDescriptor::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE MTL4::PipelineOptions* MTL4::PipelineDescriptor::options() const
{
    return Object::sendMessage<MTL4::PipelineOptions*>(this, _MTL_PRIVATE_SEL(options));
}

_MTL_INLINE void MTL4::PipelineDescriptor::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE void MTL4::PipelineDescriptor::setOptions(const MTL4::PipelineOptions* options)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setOptions_), options);
}
