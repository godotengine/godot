//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTL4Compiler.hpp
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
#include "MTLDevice.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"

#include <functional>

namespace MTL4
{
class BinaryFunction;
class BinaryFunctionDescriptor;
class CompilerDescriptor;
class CompilerTask;
class CompilerTaskOptions;
class ComputePipelineDescriptor;
class LibraryDescriptor;
class MachineLearningPipelineDescriptor;
class MachineLearningPipelineState;
class PipelineDataSetSerializer;
class PipelineDescriptor;
class PipelineStageDynamicLinkingDescriptor;
class RenderPipelineDynamicLinkingDescriptor;
}

namespace MTL
{
class ComputePipelineState;
class Device;
class DynamicLibrary;
class Library;
class RenderPipelineState;

using NewDynamicLibraryCompletionHandler = void (^)(MTL::DynamicLibrary*, NS::Error*);
using NewDynamicLibraryCompletionHandlerFunction = std::function<void(MTL::DynamicLibrary*, NS::Error*)>;
}

namespace MTL4
{
using NewComputePipelineStateCompletionHandler = void (^)(MTL::ComputePipelineState*, NS::Error*);
using NewComputePipelineStateCompletionHandlerFunction = std::function<void(MTL::ComputePipelineState*, NS::Error*)>;
using NewRenderPipelineStateCompletionHandler = void (^)(MTL::RenderPipelineState*, NS::Error*);
using NewRenderPipelineStateCompletionHandlerFunction = std::function<void(MTL::RenderPipelineState*, NS::Error*)>;
using NewBinaryFunctionCompletionHandler = void (^)(MTL4::BinaryFunction*, NS::Error*);
using NewBinaryFunctionCompletionHandlerFunction = std::function<void(MTL4::BinaryFunction*, NS::Error*)>;
using NewMachineLearningPipelineStateCompletionHandler = void (^)(MTL4::MachineLearningPipelineState*, NS::Error*);
using NewMachineLearningPipelineStateCompletionHandlerFunction = std::function<void(MTL4::MachineLearningPipelineState*, NS::Error*)>;

class CompilerDescriptor : public NS::Copying<CompilerDescriptor>
{
public:
    static CompilerDescriptor* alloc();

    CompilerDescriptor*        init();

    NS::String*                label() const;

    PipelineDataSetSerializer* pipelineDataSetSerializer() const;

    void                       setLabel(const NS::String* label);

    void                       setPipelineDataSetSerializer(const MTL4::PipelineDataSetSerializer* pipelineDataSetSerializer);
};
class CompilerTaskOptions : public NS::Copying<CompilerTaskOptions>
{
public:
    static CompilerTaskOptions* alloc();

    CompilerTaskOptions*        init();

    NS::Array*                  lookupArchives() const;
    void                        setLookupArchives(const NS::Array* lookupArchives);
};
class Compiler : public NS::Referencing<Compiler>
{
public:
    MTL::Device*                  device() const;

    NS::String*                   label() const;

    BinaryFunction*               newBinaryFunction(const MTL4::BinaryFunctionDescriptor* descriptor, const MTL4::CompilerTaskOptions* compilerTaskOptions, NS::Error** error);
    CompilerTask*                 newBinaryFunction(const MTL4::BinaryFunctionDescriptor* descriptor, const MTL4::CompilerTaskOptions* compilerTaskOptions, const MTL4::NewBinaryFunctionCompletionHandler completionHandler);

    MTL::ComputePipelineState*    newComputePipelineState(const MTL4::ComputePipelineDescriptor* descriptor, const MTL4::CompilerTaskOptions* compilerTaskOptions, NS::Error** error);
    MTL::ComputePipelineState*    newComputePipelineState(const MTL4::ComputePipelineDescriptor* descriptor, const MTL4::PipelineStageDynamicLinkingDescriptor* dynamicLinkingDescriptor, const MTL4::CompilerTaskOptions* compilerTaskOptions, NS::Error** error);
    CompilerTask*                 newComputePipelineState(const MTL4::ComputePipelineDescriptor* descriptor, const MTL4::CompilerTaskOptions* compilerTaskOptions, const MTL::NewComputePipelineStateCompletionHandler completionHandler);
    CompilerTask*                 newComputePipelineState(const MTL4::ComputePipelineDescriptor* descriptor, const MTL4::PipelineStageDynamicLinkingDescriptor* dynamicLinkingDescriptor, const MTL4::CompilerTaskOptions* compilerTaskOptions, const MTL::NewComputePipelineStateCompletionHandler completionHandler);
    CompilerTask*                 newComputePipelineState(const MTL4::ComputePipelineDescriptor* pDescriptor, const MTL4::CompilerTaskOptions* options, const MTL4::NewComputePipelineStateCompletionHandlerFunction& function);

    MTL::DynamicLibrary*          newDynamicLibrary(const MTL::Library* library, NS::Error** error);
    MTL::DynamicLibrary*          newDynamicLibrary(const NS::URL* url, NS::Error** error);
    CompilerTask*                 newDynamicLibrary(const MTL::Library* library, const MTL::NewDynamicLibraryCompletionHandler completionHandler);
    CompilerTask*                 newDynamicLibrary(const NS::URL* url, const MTL::NewDynamicLibraryCompletionHandler completionHandler);
    CompilerTask*                 newDynamicLibrary(const MTL::Library* pLibrary, const MTL::NewDynamicLibraryCompletionHandlerFunction& function);
    CompilerTask*                 newDynamicLibrary(const NS::URL* pURL, const MTL::NewDynamicLibraryCompletionHandlerFunction& function);

    MTL::Library*                 newLibrary(const MTL4::LibraryDescriptor* descriptor, NS::Error** error);
    CompilerTask*                 newLibrary(const MTL4::LibraryDescriptor* descriptor, const MTL::NewLibraryCompletionHandler completionHandler);
    CompilerTask*                 newLibrary(const MTL4::LibraryDescriptor* pDescriptor, const MTL::NewLibraryCompletionHandlerFunction& function);

    MachineLearningPipelineState* newMachineLearningPipelineState(const MTL4::MachineLearningPipelineDescriptor* descriptor, NS::Error** error);
    CompilerTask*                 newMachineLearningPipelineState(const MTL4::MachineLearningPipelineDescriptor* descriptor, const MTL4::NewMachineLearningPipelineStateCompletionHandler completionHandler);
    CompilerTask*                 newMachineLearningPipelineState(const MTL4::MachineLearningPipelineDescriptor* pDescriptor, const MTL4::NewMachineLearningPipelineStateCompletionHandlerFunction& function);

    MTL::RenderPipelineState*     newRenderPipelineState(const MTL4::PipelineDescriptor* descriptor, const MTL4::CompilerTaskOptions* compilerTaskOptions, NS::Error** error);
    MTL::RenderPipelineState*     newRenderPipelineState(const MTL4::PipelineDescriptor* descriptor, const MTL4::RenderPipelineDynamicLinkingDescriptor* dynamicLinkingDescriptor, const MTL4::CompilerTaskOptions* compilerTaskOptions, NS::Error** error);
    CompilerTask*                 newRenderPipelineState(const MTL4::PipelineDescriptor* descriptor, const MTL4::CompilerTaskOptions* compilerTaskOptions, const MTL::NewRenderPipelineStateCompletionHandler completionHandler);
    CompilerTask*                 newRenderPipelineState(const MTL4::PipelineDescriptor* descriptor, const MTL4::RenderPipelineDynamicLinkingDescriptor* dynamicLinkingDescriptor, const MTL4::CompilerTaskOptions* compilerTaskOptions, const MTL::NewRenderPipelineStateCompletionHandler completionHandler);
    CompilerTask*                 newRenderPipelineState(const MTL4::PipelineDescriptor* pDescriptor, const MTL4::CompilerTaskOptions* options, const MTL4::NewRenderPipelineStateCompletionHandlerFunction& function);
    MTL::RenderPipelineState*     newRenderPipelineStateBySpecialization(const MTL4::PipelineDescriptor* descriptor, const MTL::RenderPipelineState* pipeline, NS::Error** error);
    CompilerTask*                 newRenderPipelineStateBySpecialization(const MTL4::PipelineDescriptor* descriptor, const MTL::RenderPipelineState* pipeline, const MTL::NewRenderPipelineStateCompletionHandler completionHandler);
    CompilerTask*                 newRenderPipelineStateBySpecialization(const MTL4::PipelineDescriptor* pDescriptor, const MTL::RenderPipelineState* pPipeline, const MTL4::NewRenderPipelineStateCompletionHandlerFunction& function);

    PipelineDataSetSerializer*    pipelineDataSetSerializer() const;
};

}
_MTL_INLINE MTL4::CompilerDescriptor* MTL4::CompilerDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::CompilerDescriptor>(_MTL_PRIVATE_CLS(MTL4CompilerDescriptor));
}

_MTL_INLINE MTL4::CompilerDescriptor* MTL4::CompilerDescriptor::init()
{
    return NS::Object::init<MTL4::CompilerDescriptor>();
}

_MTL_INLINE NS::String* MTL4::CompilerDescriptor::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE MTL4::PipelineDataSetSerializer* MTL4::CompilerDescriptor::pipelineDataSetSerializer() const
{
    return Object::sendMessage<MTL4::PipelineDataSetSerializer*>(this, _MTL_PRIVATE_SEL(pipelineDataSetSerializer));
}

_MTL_INLINE void MTL4::CompilerDescriptor::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE void MTL4::CompilerDescriptor::setPipelineDataSetSerializer(const MTL4::PipelineDataSetSerializer* pipelineDataSetSerializer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPipelineDataSetSerializer_), pipelineDataSetSerializer);
}

_MTL_INLINE MTL4::CompilerTaskOptions* MTL4::CompilerTaskOptions::alloc()
{
    return NS::Object::alloc<MTL4::CompilerTaskOptions>(_MTL_PRIVATE_CLS(MTL4CompilerTaskOptions));
}

_MTL_INLINE MTL4::CompilerTaskOptions* MTL4::CompilerTaskOptions::init()
{
    return NS::Object::init<MTL4::CompilerTaskOptions>();
}

_MTL_INLINE NS::Array* MTL4::CompilerTaskOptions::lookupArchives() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(lookupArchives));
}

_MTL_INLINE void MTL4::CompilerTaskOptions::setLookupArchives(const NS::Array* lookupArchives)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLookupArchives_), lookupArchives);
}

_MTL_INLINE MTL::Device* MTL4::Compiler::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE NS::String* MTL4::Compiler::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE MTL4::BinaryFunction* MTL4::Compiler::newBinaryFunction(const MTL4::BinaryFunctionDescriptor* descriptor, const MTL4::CompilerTaskOptions* compilerTaskOptions, NS::Error** error)
{
    return Object::sendMessage<MTL4::BinaryFunction*>(this, _MTL_PRIVATE_SEL(newBinaryFunctionWithDescriptor_compilerTaskOptions_error_), descriptor, compilerTaskOptions, error);
}

_MTL_INLINE MTL4::CompilerTask* MTL4::Compiler::newBinaryFunction(const MTL4::BinaryFunctionDescriptor* descriptor, const MTL4::CompilerTaskOptions* compilerTaskOptions, const MTL4::NewBinaryFunctionCompletionHandler completionHandler)
{
    return Object::sendMessage<MTL4::CompilerTask*>(this, _MTL_PRIVATE_SEL(newBinaryFunctionWithDescriptor_compilerTaskOptions_completionHandler_), descriptor, compilerTaskOptions, completionHandler);
}

_MTL_INLINE MTL::ComputePipelineState* MTL4::Compiler::newComputePipelineState(const MTL4::ComputePipelineDescriptor* descriptor, const MTL4::CompilerTaskOptions* compilerTaskOptions, NS::Error** error)
{
    return Object::sendMessage<MTL::ComputePipelineState*>(this, _MTL_PRIVATE_SEL(newComputePipelineStateWithDescriptor_compilerTaskOptions_error_), descriptor, compilerTaskOptions, error);
}

_MTL_INLINE MTL::ComputePipelineState* MTL4::Compiler::newComputePipelineState(const MTL4::ComputePipelineDescriptor* descriptor, const MTL4::PipelineStageDynamicLinkingDescriptor* dynamicLinkingDescriptor, const MTL4::CompilerTaskOptions* compilerTaskOptions, NS::Error** error)
{
    return Object::sendMessage<MTL::ComputePipelineState*>(this, _MTL_PRIVATE_SEL(newComputePipelineStateWithDescriptor_dynamicLinkingDescriptor_compilerTaskOptions_error_), descriptor, dynamicLinkingDescriptor, compilerTaskOptions, error);
}

_MTL_INLINE MTL4::CompilerTask* MTL4::Compiler::newComputePipelineState(const MTL4::ComputePipelineDescriptor* descriptor, const MTL4::CompilerTaskOptions* compilerTaskOptions, const MTL::NewComputePipelineStateCompletionHandler completionHandler)
{
    return Object::sendMessage<MTL4::CompilerTask*>(this, _MTL_PRIVATE_SEL(newComputePipelineStateWithDescriptor_compilerTaskOptions_completionHandler_), descriptor, compilerTaskOptions, completionHandler);
}

_MTL_INLINE MTL4::CompilerTask* MTL4::Compiler::newComputePipelineState(const MTL4::ComputePipelineDescriptor* descriptor, const MTL4::PipelineStageDynamicLinkingDescriptor* dynamicLinkingDescriptor, const MTL4::CompilerTaskOptions* compilerTaskOptions, const MTL::NewComputePipelineStateCompletionHandler completionHandler)
{
    return Object::sendMessage<MTL4::CompilerTask*>(this, _MTL_PRIVATE_SEL(newComputePipelineStateWithDescriptor_dynamicLinkingDescriptor_compilerTaskOptions_completionHandler_), descriptor, dynamicLinkingDescriptor, compilerTaskOptions, completionHandler);
}

_MTL_INLINE MTL4::CompilerTask* MTL4::Compiler::newComputePipelineState(const MTL4::ComputePipelineDescriptor* pDescriptor, const MTL4::CompilerTaskOptions* options, const MTL4::NewComputePipelineStateCompletionHandlerFunction& function)
{
    __block MTL4::NewComputePipelineStateCompletionHandlerFunction blockFunction = function;
    return newComputePipelineState(pDescriptor, options, ^(MTL::ComputePipelineState* pPipeline, NS::Error* pError) { blockFunction(pPipeline, pError); });
}

_MTL_INLINE MTL::DynamicLibrary* MTL4::Compiler::newDynamicLibrary(const MTL::Library* library, NS::Error** error)
{
    return Object::sendMessage<MTL::DynamicLibrary*>(this, _MTL_PRIVATE_SEL(newDynamicLibrary_error_), library, error);
}

_MTL_INLINE MTL::DynamicLibrary* MTL4::Compiler::newDynamicLibrary(const NS::URL* url, NS::Error** error)
{
    return Object::sendMessage<MTL::DynamicLibrary*>(this, _MTL_PRIVATE_SEL(newDynamicLibraryWithURL_error_), url, error);
}

_MTL_INLINE MTL4::CompilerTask* MTL4::Compiler::newDynamicLibrary(const MTL::Library* library, const MTL::NewDynamicLibraryCompletionHandler completionHandler)
{
    return Object::sendMessage<MTL4::CompilerTask*>(this, _MTL_PRIVATE_SEL(newDynamicLibrary_completionHandler_), library, completionHandler);
}

_MTL_INLINE MTL4::CompilerTask* MTL4::Compiler::newDynamicLibrary(const NS::URL* url, const MTL::NewDynamicLibraryCompletionHandler completionHandler)
{
    return Object::sendMessage<MTL4::CompilerTask*>(this, _MTL_PRIVATE_SEL(newDynamicLibraryWithURL_completionHandler_), url, completionHandler);
}

_MTL_INLINE MTL4::CompilerTask* MTL4::Compiler::newDynamicLibrary(const MTL::Library* pLibrary, const MTL::NewDynamicLibraryCompletionHandlerFunction& function)
{
    __block MTL::NewDynamicLibraryCompletionHandlerFunction blockFunction = function;
    return newDynamicLibrary(pLibrary, ^(MTL::DynamicLibrary* pLibraryRef, NS::Error* pError) { blockFunction(pLibraryRef, pError); });
}

_MTL_INLINE MTL4::CompilerTask* MTL4::Compiler::newDynamicLibrary(const NS::URL* pURL, const MTL::NewDynamicLibraryCompletionHandlerFunction& function)
{
    __block MTL::NewDynamicLibraryCompletionHandlerFunction blockFunction = function;
    return newDynamicLibrary(pURL, ^(MTL::DynamicLibrary* pLibrary, NS::Error* pError) { blockFunction(pLibrary, pError); });
}

_MTL_INLINE MTL::Library* MTL4::Compiler::newLibrary(const MTL4::LibraryDescriptor* descriptor, NS::Error** error)
{
    return Object::sendMessage<MTL::Library*>(this, _MTL_PRIVATE_SEL(newLibraryWithDescriptor_error_), descriptor, error);
}

_MTL_INLINE MTL4::CompilerTask* MTL4::Compiler::newLibrary(const MTL4::LibraryDescriptor* descriptor, const MTL::NewLibraryCompletionHandler completionHandler)
{
    return Object::sendMessage<MTL4::CompilerTask*>(this, _MTL_PRIVATE_SEL(newLibraryWithDescriptor_completionHandler_), descriptor, completionHandler);
}

_MTL_INLINE MTL4::CompilerTask* MTL4::Compiler::newLibrary(const MTL4::LibraryDescriptor* pDescriptor, const MTL::NewLibraryCompletionHandlerFunction& function)
{
    __block MTL::NewLibraryCompletionHandlerFunction blockFunction = function;
    return newLibrary(pDescriptor, ^(MTL::Library* pLibrary, NS::Error* pError) { blockFunction(pLibrary, pError); });
}

_MTL_INLINE MTL4::MachineLearningPipelineState* MTL4::Compiler::newMachineLearningPipelineState(const MTL4::MachineLearningPipelineDescriptor* descriptor, NS::Error** error)
{
    return Object::sendMessage<MTL4::MachineLearningPipelineState*>(this, _MTL_PRIVATE_SEL(newMachineLearningPipelineStateWithDescriptor_error_), descriptor, error);
}

_MTL_INLINE MTL4::CompilerTask* MTL4::Compiler::newMachineLearningPipelineState(const MTL4::MachineLearningPipelineDescriptor* descriptor, const MTL4::NewMachineLearningPipelineStateCompletionHandler completionHandler)
{
    return Object::sendMessage<MTL4::CompilerTask*>(this, _MTL_PRIVATE_SEL(newMachineLearningPipelineStateWithDescriptor_completionHandler_), descriptor, completionHandler);
}

_MTL_INLINE MTL4::CompilerTask* MTL4::Compiler::newMachineLearningPipelineState(const MTL4::MachineLearningPipelineDescriptor* pDescriptor, const MTL4::NewMachineLearningPipelineStateCompletionHandlerFunction& function)
{
    __block MTL4::NewMachineLearningPipelineStateCompletionHandlerFunction blockFunction = function;
    return newMachineLearningPipelineState(pDescriptor, ^(MTL4::MachineLearningPipelineState* pPipeline, NS::Error* pError) { blockFunction(pPipeline, pError); });
}

_MTL_INLINE MTL::RenderPipelineState* MTL4::Compiler::newRenderPipelineState(const MTL4::PipelineDescriptor* descriptor, const MTL4::CompilerTaskOptions* compilerTaskOptions, NS::Error** error)
{
    return Object::sendMessage<MTL::RenderPipelineState*>(this, _MTL_PRIVATE_SEL(newRenderPipelineStateWithDescriptor_compilerTaskOptions_error_), descriptor, compilerTaskOptions, error);
}

_MTL_INLINE MTL::RenderPipelineState* MTL4::Compiler::newRenderPipelineState(const MTL4::PipelineDescriptor* descriptor, const MTL4::RenderPipelineDynamicLinkingDescriptor* dynamicLinkingDescriptor, const MTL4::CompilerTaskOptions* compilerTaskOptions, NS::Error** error)
{
    return Object::sendMessage<MTL::RenderPipelineState*>(this, _MTL_PRIVATE_SEL(newRenderPipelineStateWithDescriptor_dynamicLinkingDescriptor_compilerTaskOptions_error_), descriptor, dynamicLinkingDescriptor, compilerTaskOptions, error);
}

_MTL_INLINE MTL4::CompilerTask* MTL4::Compiler::newRenderPipelineState(const MTL4::PipelineDescriptor* descriptor, const MTL4::CompilerTaskOptions* compilerTaskOptions, const MTL::NewRenderPipelineStateCompletionHandler completionHandler)
{
    return Object::sendMessage<MTL4::CompilerTask*>(this, _MTL_PRIVATE_SEL(newRenderPipelineStateWithDescriptor_compilerTaskOptions_completionHandler_), descriptor, compilerTaskOptions, completionHandler);
}

_MTL_INLINE MTL4::CompilerTask* MTL4::Compiler::newRenderPipelineState(const MTL4::PipelineDescriptor* descriptor, const MTL4::RenderPipelineDynamicLinkingDescriptor* dynamicLinkingDescriptor, const MTL4::CompilerTaskOptions* compilerTaskOptions, const MTL::NewRenderPipelineStateCompletionHandler completionHandler)
{
    return Object::sendMessage<MTL4::CompilerTask*>(this, _MTL_PRIVATE_SEL(newRenderPipelineStateWithDescriptor_dynamicLinkingDescriptor_compilerTaskOptions_completionHandler_), descriptor, dynamicLinkingDescriptor, compilerTaskOptions, completionHandler);
}

_MTL_INLINE MTL4::CompilerTask* MTL4::Compiler::newRenderPipelineState(const MTL4::PipelineDescriptor* pDescriptor, const MTL4::CompilerTaskOptions* options, const MTL4::NewRenderPipelineStateCompletionHandlerFunction& function)
{
    __block MTL4::NewRenderPipelineStateCompletionHandlerFunction blockFunction = function;
    return newRenderPipelineState(pDescriptor, options, ^(MTL::RenderPipelineState* pPipeline, NS::Error* pError) { blockFunction(pPipeline, pError); });
}

_MTL_INLINE MTL::RenderPipelineState* MTL4::Compiler::newRenderPipelineStateBySpecialization(const MTL4::PipelineDescriptor* descriptor, const MTL::RenderPipelineState* pipeline, NS::Error** error)
{
    return Object::sendMessage<MTL::RenderPipelineState*>(this, _MTL_PRIVATE_SEL(newRenderPipelineStateBySpecializationWithDescriptor_pipeline_error_), descriptor, pipeline, error);
}

_MTL_INLINE MTL4::CompilerTask* MTL4::Compiler::newRenderPipelineStateBySpecialization(const MTL4::PipelineDescriptor* descriptor, const MTL::RenderPipelineState* pipeline, const MTL::NewRenderPipelineStateCompletionHandler completionHandler)
{
    return Object::sendMessage<MTL4::CompilerTask*>(this, _MTL_PRIVATE_SEL(newRenderPipelineStateBySpecializationWithDescriptor_pipeline_completionHandler_), descriptor, pipeline, completionHandler);
}

_MTL_INLINE MTL4::CompilerTask* MTL4::Compiler::newRenderPipelineStateBySpecialization(const MTL4::PipelineDescriptor* pDescriptor, const MTL::RenderPipelineState* pPipeline, const MTL4::NewRenderPipelineStateCompletionHandlerFunction& function)
{
    __block MTL4::NewRenderPipelineStateCompletionHandlerFunction blockFunction = function;
    return newRenderPipelineStateBySpecialization(pDescriptor, pPipeline, ^(MTL::RenderPipelineState* pPipelineRef, NS::Error* pError) { blockFunction(pPipelineRef, pError); });
}

_MTL_INLINE MTL4::PipelineDataSetSerializer* MTL4::Compiler::pipelineDataSetSerializer() const
{
    return Object::sendMessage<MTL4::PipelineDataSetSerializer*>(this, _MTL_PRIVATE_SEL(pipelineDataSetSerializer));
}
