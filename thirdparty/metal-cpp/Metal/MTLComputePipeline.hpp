//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLComputePipeline.hpp
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
#include "MTLAllocation.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPipeline.hpp"
#include "MTLPrivate.hpp"
#include "MTLTypes.hpp"

namespace MTL
{
class ComputePipelineDescriptor;
class ComputePipelineReflection;
class ComputePipelineState;
class Device;
class Function;
class FunctionHandle;
class IntersectionFunctionTable;
class IntersectionFunctionTableDescriptor;
class LinkedFunctions;
class PipelineBufferDescriptorArray;
class StageInputOutputDescriptor;
class VisibleFunctionTable;
class VisibleFunctionTableDescriptor;

}
namespace MTL4
{
class BinaryFunction;

}
namespace MTL
{
class ComputePipelineReflection : public NS::Referencing<ComputePipelineReflection>
{
public:
    static ComputePipelineReflection* alloc();

    NS::Array*                        arguments() const;

    NS::Array*                        bindings() const;

    ComputePipelineReflection*        init();
};
class ComputePipelineDescriptor : public NS::Copying<ComputePipelineDescriptor>
{
public:
    static ComputePipelineDescriptor* alloc();

    NS::Array*                        binaryArchives() const;

    PipelineBufferDescriptorArray*    buffers() const;

    Function*                         computeFunction() const;

    ComputePipelineDescriptor*        init();

    NS::Array*                        insertLibraries() const;

    NS::String*                       label() const;

    LinkedFunctions*                  linkedFunctions() const;

    NS::UInteger                      maxCallStackDepth() const;

    NS::UInteger                      maxTotalThreadsPerThreadgroup() const;

    NS::Array*                        preloadedLibraries() const;

    Size                              requiredThreadsPerThreadgroup() const;

    void                              reset();

    void                              setBinaryArchives(const NS::Array* binaryArchives);

    void                              setComputeFunction(const MTL::Function* computeFunction);

    void                              setInsertLibraries(const NS::Array* insertLibraries);

    void                              setLabel(const NS::String* label);

    void                              setLinkedFunctions(const MTL::LinkedFunctions* linkedFunctions);

    void                              setMaxCallStackDepth(NS::UInteger maxCallStackDepth);

    void                              setMaxTotalThreadsPerThreadgroup(NS::UInteger maxTotalThreadsPerThreadgroup);

    void                              setPreloadedLibraries(const NS::Array* preloadedLibraries);

    void                              setRequiredThreadsPerThreadgroup(MTL::Size requiredThreadsPerThreadgroup);

    void                              setShaderValidation(MTL::ShaderValidation shaderValidation);

    void                              setStageInputDescriptor(const MTL::StageInputOutputDescriptor* stageInputDescriptor);

    void                              setSupportAddingBinaryFunctions(bool supportAddingBinaryFunctions);

    void                              setSupportIndirectCommandBuffers(bool supportIndirectCommandBuffers);

    void                              setThreadGroupSizeIsMultipleOfThreadExecutionWidth(bool threadGroupSizeIsMultipleOfThreadExecutionWidth);

    ShaderValidation                  shaderValidation() const;

    StageInputOutputDescriptor*       stageInputDescriptor() const;

    bool                              supportAddingBinaryFunctions() const;

    bool                              supportIndirectCommandBuffers() const;

    bool                              threadGroupSizeIsMultipleOfThreadExecutionWidth() const;
};
class ComputePipelineState : public NS::Referencing<ComputePipelineState, Allocation>
{
public:
    Device*                    device() const;

    FunctionHandle*            functionHandle(const NS::String* name);
    FunctionHandle*            functionHandle(const MTL4::BinaryFunction* function);
    FunctionHandle*            functionHandle(const MTL::Function* function);

    ResourceID                 gpuResourceID() const;

    NS::UInteger               imageblockMemoryLength(MTL::Size imageblockDimensions);

    NS::String*                label() const;

    NS::UInteger               maxTotalThreadsPerThreadgroup() const;

    ComputePipelineState*      newComputePipelineStateWithBinaryFunctions(const NS::Array* additionalBinaryFunctions, NS::Error** error);
    ComputePipelineState*      newComputePipelineState(const NS::Array* functions, NS::Error** error);

    IntersectionFunctionTable* newIntersectionFunctionTable(const MTL::IntersectionFunctionTableDescriptor* descriptor);

    VisibleFunctionTable*      newVisibleFunctionTable(const MTL::VisibleFunctionTableDescriptor* descriptor);

    ComputePipelineReflection* reflection() const;

    Size                       requiredThreadsPerThreadgroup() const;

    ShaderValidation           shaderValidation() const;

    NS::UInteger               staticThreadgroupMemoryLength() const;

    bool                       supportIndirectCommandBuffers() const;

    NS::UInteger               threadExecutionWidth() const;
};

}
_MTL_INLINE MTL::ComputePipelineReflection* MTL::ComputePipelineReflection::alloc()
{
    return NS::Object::alloc<MTL::ComputePipelineReflection>(_MTL_PRIVATE_CLS(MTLComputePipelineReflection));
}

_MTL_INLINE NS::Array* MTL::ComputePipelineReflection::arguments() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(arguments));
}

_MTL_INLINE NS::Array* MTL::ComputePipelineReflection::bindings() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(bindings));
}

_MTL_INLINE MTL::ComputePipelineReflection* MTL::ComputePipelineReflection::init()
{
    return NS::Object::init<MTL::ComputePipelineReflection>();
}

_MTL_INLINE MTL::ComputePipelineDescriptor* MTL::ComputePipelineDescriptor::alloc()
{
    return NS::Object::alloc<MTL::ComputePipelineDescriptor>(_MTL_PRIVATE_CLS(MTLComputePipelineDescriptor));
}

_MTL_INLINE NS::Array* MTL::ComputePipelineDescriptor::binaryArchives() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(binaryArchives));
}

_MTL_INLINE MTL::PipelineBufferDescriptorArray* MTL::ComputePipelineDescriptor::buffers() const
{
    return Object::sendMessage<MTL::PipelineBufferDescriptorArray*>(this, _MTL_PRIVATE_SEL(buffers));
}

_MTL_INLINE MTL::Function* MTL::ComputePipelineDescriptor::computeFunction() const
{
    return Object::sendMessage<MTL::Function*>(this, _MTL_PRIVATE_SEL(computeFunction));
}

_MTL_INLINE MTL::ComputePipelineDescriptor* MTL::ComputePipelineDescriptor::init()
{
    return NS::Object::init<MTL::ComputePipelineDescriptor>();
}

_MTL_INLINE NS::Array* MTL::ComputePipelineDescriptor::insertLibraries() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(insertLibraries));
}

_MTL_INLINE NS::String* MTL::ComputePipelineDescriptor::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE MTL::LinkedFunctions* MTL::ComputePipelineDescriptor::linkedFunctions() const
{
    return Object::sendMessage<MTL::LinkedFunctions*>(this, _MTL_PRIVATE_SEL(linkedFunctions));
}

_MTL_INLINE NS::UInteger MTL::ComputePipelineDescriptor::maxCallStackDepth() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxCallStackDepth));
}

_MTL_INLINE NS::UInteger MTL::ComputePipelineDescriptor::maxTotalThreadsPerThreadgroup() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxTotalThreadsPerThreadgroup));
}

_MTL_INLINE NS::Array* MTL::ComputePipelineDescriptor::preloadedLibraries() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(preloadedLibraries));
}

_MTL_INLINE MTL::Size MTL::ComputePipelineDescriptor::requiredThreadsPerThreadgroup() const
{
    return Object::sendMessage<MTL::Size>(this, _MTL_PRIVATE_SEL(requiredThreadsPerThreadgroup));
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::reset()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(reset));
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setBinaryArchives(const NS::Array* binaryArchives)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBinaryArchives_), binaryArchives);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setComputeFunction(const MTL::Function* computeFunction)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setComputeFunction_), computeFunction);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setInsertLibraries(const NS::Array* insertLibraries)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInsertLibraries_), insertLibraries);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setLinkedFunctions(const MTL::LinkedFunctions* linkedFunctions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLinkedFunctions_), linkedFunctions);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setMaxCallStackDepth(NS::UInteger maxCallStackDepth)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxCallStackDepth_), maxCallStackDepth);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setMaxTotalThreadsPerThreadgroup(NS::UInteger maxTotalThreadsPerThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxTotalThreadsPerThreadgroup_), maxTotalThreadsPerThreadgroup);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setPreloadedLibraries(const NS::Array* preloadedLibraries)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPreloadedLibraries_), preloadedLibraries);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setRequiredThreadsPerThreadgroup(MTL::Size requiredThreadsPerThreadgroup)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRequiredThreadsPerThreadgroup_), requiredThreadsPerThreadgroup);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setShaderValidation(MTL::ShaderValidation shaderValidation)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setShaderValidation_), shaderValidation);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setStageInputDescriptor(const MTL::StageInputOutputDescriptor* stageInputDescriptor)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStageInputDescriptor_), stageInputDescriptor);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setSupportAddingBinaryFunctions(bool supportAddingBinaryFunctions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSupportAddingBinaryFunctions_), supportAddingBinaryFunctions);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setSupportIndirectCommandBuffers(bool supportIndirectCommandBuffers)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSupportIndirectCommandBuffers_), supportIndirectCommandBuffers);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setThreadGroupSizeIsMultipleOfThreadExecutionWidth(bool threadGroupSizeIsMultipleOfThreadExecutionWidth)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setThreadGroupSizeIsMultipleOfThreadExecutionWidth_), threadGroupSizeIsMultipleOfThreadExecutionWidth);
}

_MTL_INLINE MTL::ShaderValidation MTL::ComputePipelineDescriptor::shaderValidation() const
{
    return Object::sendMessage<MTL::ShaderValidation>(this, _MTL_PRIVATE_SEL(shaderValidation));
}

_MTL_INLINE MTL::StageInputOutputDescriptor* MTL::ComputePipelineDescriptor::stageInputDescriptor() const
{
    return Object::sendMessage<MTL::StageInputOutputDescriptor*>(this, _MTL_PRIVATE_SEL(stageInputDescriptor));
}

_MTL_INLINE bool MTL::ComputePipelineDescriptor::supportAddingBinaryFunctions() const
{
    return Object::sendMessageSafe<bool>(this, _MTL_PRIVATE_SEL(supportAddingBinaryFunctions));
}

_MTL_INLINE bool MTL::ComputePipelineDescriptor::supportIndirectCommandBuffers() const
{
    return Object::sendMessageSafe<bool>(this, _MTL_PRIVATE_SEL(supportIndirectCommandBuffers));
}

_MTL_INLINE bool MTL::ComputePipelineDescriptor::threadGroupSizeIsMultipleOfThreadExecutionWidth() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(threadGroupSizeIsMultipleOfThreadExecutionWidth));
}

_MTL_INLINE MTL::Device* MTL::ComputePipelineState::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE MTL::FunctionHandle* MTL::ComputePipelineState::functionHandle(const NS::String* name)
{
    return Object::sendMessage<MTL::FunctionHandle*>(this, _MTL_PRIVATE_SEL(functionHandleWithName_), name);
}

_MTL_INLINE MTL::FunctionHandle* MTL::ComputePipelineState::functionHandle(const MTL4::BinaryFunction* function)
{
    return Object::sendMessage<MTL::FunctionHandle*>(this, _MTL_PRIVATE_SEL(functionHandleWithBinaryFunction_), function);
}

_MTL_INLINE MTL::FunctionHandle* MTL::ComputePipelineState::functionHandle(const MTL::Function* function)
{
    return Object::sendMessage<MTL::FunctionHandle*>(this, _MTL_PRIVATE_SEL(functionHandleWithFunction_), function);
}

_MTL_INLINE MTL::ResourceID MTL::ComputePipelineState::gpuResourceID() const
{
    return Object::sendMessage<MTL::ResourceID>(this, _MTL_PRIVATE_SEL(gpuResourceID));
}

_MTL_INLINE NS::UInteger MTL::ComputePipelineState::imageblockMemoryLength(MTL::Size imageblockDimensions)
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(imageblockMemoryLengthForDimensions_), imageblockDimensions);
}

_MTL_INLINE NS::String* MTL::ComputePipelineState::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE NS::UInteger MTL::ComputePipelineState::maxTotalThreadsPerThreadgroup() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxTotalThreadsPerThreadgroup));
}

_MTL_INLINE MTL::ComputePipelineState* MTL::ComputePipelineState::newComputePipelineStateWithBinaryFunctions(const NS::Array* additionalBinaryFunctions, NS::Error** error)
{
    return Object::sendMessage<MTL::ComputePipelineState*>(this, _MTL_PRIVATE_SEL(newComputePipelineStateWithBinaryFunctions_error_), additionalBinaryFunctions, error);
}

_MTL_INLINE MTL::ComputePipelineState* MTL::ComputePipelineState::newComputePipelineState(const NS::Array* functions, NS::Error** error)
{
    return Object::sendMessage<MTL::ComputePipelineState*>(this, _MTL_PRIVATE_SEL(newComputePipelineStateWithAdditionalBinaryFunctions_error_), functions, error);
}

_MTL_INLINE MTL::IntersectionFunctionTable* MTL::ComputePipelineState::newIntersectionFunctionTable(const MTL::IntersectionFunctionTableDescriptor* descriptor)
{
    return Object::sendMessage<MTL::IntersectionFunctionTable*>(this, _MTL_PRIVATE_SEL(newIntersectionFunctionTableWithDescriptor_), descriptor);
}

_MTL_INLINE MTL::VisibleFunctionTable* MTL::ComputePipelineState::newVisibleFunctionTable(const MTL::VisibleFunctionTableDescriptor* descriptor)
{
    return Object::sendMessage<MTL::VisibleFunctionTable*>(this, _MTL_PRIVATE_SEL(newVisibleFunctionTableWithDescriptor_), descriptor);
}

_MTL_INLINE MTL::ComputePipelineReflection* MTL::ComputePipelineState::reflection() const
{
    return Object::sendMessage<MTL::ComputePipelineReflection*>(this, _MTL_PRIVATE_SEL(reflection));
}

_MTL_INLINE MTL::Size MTL::ComputePipelineState::requiredThreadsPerThreadgroup() const
{
    return Object::sendMessage<MTL::Size>(this, _MTL_PRIVATE_SEL(requiredThreadsPerThreadgroup));
}

_MTL_INLINE MTL::ShaderValidation MTL::ComputePipelineState::shaderValidation() const
{
    return Object::sendMessage<MTL::ShaderValidation>(this, _MTL_PRIVATE_SEL(shaderValidation));
}

_MTL_INLINE NS::UInteger MTL::ComputePipelineState::staticThreadgroupMemoryLength() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(staticThreadgroupMemoryLength));
}

_MTL_INLINE bool MTL::ComputePipelineState::supportIndirectCommandBuffers() const
{
    return Object::sendMessageSafe<bool>(this, _MTL_PRIVATE_SEL(supportIndirectCommandBuffers));
}

_MTL_INLINE NS::UInteger MTL::ComputePipelineState::threadExecutionWidth() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(threadExecutionWidth));
}
