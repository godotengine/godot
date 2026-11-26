//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTL4MachineLearningPipeline.hpp
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
#include "MTL4PipelineState.hpp"
#include "MTLAllocation.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"

namespace MTL4
{
class FunctionDescriptor;
class MachineLearningPipelineDescriptor;
class MachineLearningPipelineReflection;
}

namespace MTL
{
class Device;
class TensorExtents;
}

namespace MTL4
{
class MachineLearningPipelineDescriptor : public NS::Copying<MachineLearningPipelineDescriptor, PipelineDescriptor>
{
public:
    static MachineLearningPipelineDescriptor* alloc();

    MachineLearningPipelineDescriptor*        init();

    MTL::TensorExtents*                       inputDimensionsAtBufferIndex(NS::Integer bufferIndex);

    NS::String*                               label() const;

    FunctionDescriptor*                       machineLearningFunctionDescriptor() const;

    void                                      reset();

    void                                      setInputDimensions(const MTL::TensorExtents* dimensions, NS::Integer bufferIndex);
    void                                      setInputDimensions(const NS::Array* dimensions, NS::Range range);

    void                                      setLabel(const NS::String* label);

    void                                      setMachineLearningFunctionDescriptor(const MTL4::FunctionDescriptor* machineLearningFunctionDescriptor);
};
class MachineLearningPipelineReflection : public NS::Referencing<MachineLearningPipelineReflection>
{
public:
    static MachineLearningPipelineReflection* alloc();

    NS::Array*                                bindings() const;

    MachineLearningPipelineReflection*        init();
};
class MachineLearningPipelineState : public NS::Referencing<MachineLearningPipelineState, MTL::Allocation>
{
public:
    MTL::Device*                       device() const;

    NS::UInteger                       intermediatesHeapSize() const;

    NS::String*                        label() const;

    MachineLearningPipelineReflection* reflection() const;
};

}
_MTL_INLINE MTL4::MachineLearningPipelineDescriptor* MTL4::MachineLearningPipelineDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::MachineLearningPipelineDescriptor>(_MTL_PRIVATE_CLS(MTL4MachineLearningPipelineDescriptor));
}

_MTL_INLINE MTL4::MachineLearningPipelineDescriptor* MTL4::MachineLearningPipelineDescriptor::init()
{
    return NS::Object::init<MTL4::MachineLearningPipelineDescriptor>();
}

_MTL_INLINE MTL::TensorExtents* MTL4::MachineLearningPipelineDescriptor::inputDimensionsAtBufferIndex(NS::Integer bufferIndex)
{
    return Object::sendMessage<MTL::TensorExtents*>(this, _MTL_PRIVATE_SEL(inputDimensionsAtBufferIndex_), bufferIndex);
}

_MTL_INLINE NS::String* MTL4::MachineLearningPipelineDescriptor::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE MTL4::FunctionDescriptor* MTL4::MachineLearningPipelineDescriptor::machineLearningFunctionDescriptor() const
{
    return Object::sendMessage<MTL4::FunctionDescriptor*>(this, _MTL_PRIVATE_SEL(machineLearningFunctionDescriptor));
}

_MTL_INLINE void MTL4::MachineLearningPipelineDescriptor::reset()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(reset));
}

_MTL_INLINE void MTL4::MachineLearningPipelineDescriptor::setInputDimensions(const MTL::TensorExtents* dimensions, NS::Integer bufferIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInputDimensions_atBufferIndex_), dimensions, bufferIndex);
}

_MTL_INLINE void MTL4::MachineLearningPipelineDescriptor::setInputDimensions(const NS::Array* dimensions, NS::Range range)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInputDimensions_withRange_), dimensions, range);
}

_MTL_INLINE void MTL4::MachineLearningPipelineDescriptor::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE void MTL4::MachineLearningPipelineDescriptor::setMachineLearningFunctionDescriptor(const MTL4::FunctionDescriptor* machineLearningFunctionDescriptor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMachineLearningFunctionDescriptor_), machineLearningFunctionDescriptor);
}

_MTL_INLINE MTL4::MachineLearningPipelineReflection* MTL4::MachineLearningPipelineReflection::alloc()
{
    return NS::Object::alloc<MTL4::MachineLearningPipelineReflection>(_MTL_PRIVATE_CLS(MTL4MachineLearningPipelineReflection));
}

_MTL_INLINE NS::Array* MTL4::MachineLearningPipelineReflection::bindings() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(bindings));
}

_MTL_INLINE MTL4::MachineLearningPipelineReflection* MTL4::MachineLearningPipelineReflection::init()
{
    return NS::Object::init<MTL4::MachineLearningPipelineReflection>();
}

_MTL_INLINE MTL::Device* MTL4::MachineLearningPipelineState::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE NS::UInteger MTL4::MachineLearningPipelineState::intermediatesHeapSize() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(intermediatesHeapSize));
}

_MTL_INLINE NS::String* MTL4::MachineLearningPipelineState::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE MTL4::MachineLearningPipelineReflection* MTL4::MachineLearningPipelineState::reflection() const
{
    return Object::sendMessage<MTL4::MachineLearningPipelineReflection*>(this, _MTL_PRIVATE_SEL(reflection));
}
