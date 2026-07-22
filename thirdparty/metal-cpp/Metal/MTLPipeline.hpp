//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLPipeline.hpp
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
class PipelineBufferDescriptor;
class PipelineBufferDescriptorArray;
_MTL_ENUM(NS::UInteger, Mutability) {
    MutabilityDefault = 0,
    MutabilityMutable = 1,
    MutabilityImmutable = 2,
};

_MTL_ENUM(NS::Integer, ShaderValidation) {
    ShaderValidationDefault = 0,
    ShaderValidationEnabled = 1,
    ShaderValidationDisabled = 2,
};

class PipelineBufferDescriptor : public NS::Copying<PipelineBufferDescriptor>
{
public:
    static PipelineBufferDescriptor* alloc();

    PipelineBufferDescriptor*        init();

    Mutability                       mutability() const;
    void                             setMutability(MTL::Mutability mutability);
};
class PipelineBufferDescriptorArray : public NS::Referencing<PipelineBufferDescriptorArray>
{
public:
    static PipelineBufferDescriptorArray* alloc();

    PipelineBufferDescriptorArray*        init();

    PipelineBufferDescriptor*             object(NS::UInteger bufferIndex);
    void                                  setObject(const MTL::PipelineBufferDescriptor* buffer, NS::UInteger bufferIndex);
};

}
_MTL_INLINE MTL::PipelineBufferDescriptor* MTL::PipelineBufferDescriptor::alloc()
{
    return NS::Object::alloc<MTL::PipelineBufferDescriptor>(_MTL_PRIVATE_CLS(MTLPipelineBufferDescriptor));
}

_MTL_INLINE MTL::PipelineBufferDescriptor* MTL::PipelineBufferDescriptor::init()
{
    return NS::Object::init<MTL::PipelineBufferDescriptor>();
}

_MTL_INLINE MTL::Mutability MTL::PipelineBufferDescriptor::mutability() const
{
    return Object::sendMessage<MTL::Mutability>(this, _MTL_PRIVATE_SEL(mutability));
}

_MTL_INLINE void MTL::PipelineBufferDescriptor::setMutability(MTL::Mutability mutability)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMutability_), mutability);
}

_MTL_INLINE MTL::PipelineBufferDescriptorArray* MTL::PipelineBufferDescriptorArray::alloc()
{
    return NS::Object::alloc<MTL::PipelineBufferDescriptorArray>(_MTL_PRIVATE_CLS(MTLPipelineBufferDescriptorArray));
}

_MTL_INLINE MTL::PipelineBufferDescriptorArray* MTL::PipelineBufferDescriptorArray::init()
{
    return NS::Object::init<MTL::PipelineBufferDescriptorArray>();
}

_MTL_INLINE MTL::PipelineBufferDescriptor* MTL::PipelineBufferDescriptorArray::object(NS::UInteger bufferIndex)
{
    return Object::sendMessage<MTL::PipelineBufferDescriptor*>(this, _MTL_PRIVATE_SEL(objectAtIndexedSubscript_), bufferIndex);
}

_MTL_INLINE void MTL::PipelineBufferDescriptorArray::setObject(const MTL::PipelineBufferDescriptor* buffer, NS::UInteger bufferIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObject_atIndexedSubscript_), buffer, bufferIndex);
}
