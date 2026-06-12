#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTLBlocks.hpp"

namespace MTL {
    class ComputePipelineState;
    class Device;
    class DynamicLibrary;
    class Library;
    class RenderPipelineState;
}
namespace MTL4 {
    class BinaryFunction;
    class BinaryFunctionDescriptor;
    class CompilerTask;
    class ComputePipelineDescriptor;
    class LibraryDescriptor;
    class MachineLearningPipelineDescriptor;
    class MachineLearningPipelineState;
    class PipelineDataSetSerializer;
    class PipelineDescriptor;
    class PipelineStageDynamicLinkingDescriptor;
    class RenderPipelineDynamicLinkingDescriptor;
}
namespace NS {
    class Array;
    class Error;
    class String;
    class URL;
}

namespace MTL4
{

class CompilerDescriptor;
class CompilerTaskOptions;
class Compiler;

class CompilerDescriptor : public NS::Copying<CompilerDescriptor>
{
public:
    static CompilerDescriptor* alloc();
    CompilerDescriptor*        init() const;

    NS::String*                      label() const;
    MTL4::PipelineDataSetSerializer* pipelineDataSetSerializer() const;
    void                             setLabel(NS::String* label);
    void                             setPipelineDataSetSerializer(MTL4::PipelineDataSetSerializer* pipelineDataSetSerializer);

};

class CompilerTaskOptions : public NS::Copying<CompilerTaskOptions>
{
public:
    static CompilerTaskOptions* alloc();
    CompilerTaskOptions*        init() const;

    NS::Array* lookupArchives() const;
    void       setLookupArchives(NS::Array* lookupArchives);

};

class Compiler : public NS::Referencing<Compiler>
{
public:
    MTL::Device*                        device() const;
    NS::String*                         label() const;
    MTL4::BinaryFunction*               newBinaryFunction(MTL4::BinaryFunctionDescriptor* descriptor, MTL4::CompilerTaskOptions* compilerTaskOptions, NS::Error** error);
    MTL4::CompilerTask*                 newBinaryFunction(MTL4::BinaryFunctionDescriptor* descriptor, MTL4::CompilerTaskOptions* compilerTaskOptions, MTL4::NewBinaryFunctionCompletionHandler completionHandler);
    MTL4::CompilerTask*                 newBinaryFunction(MTL4::BinaryFunctionDescriptor* descriptor, MTL4::CompilerTaskOptions* compilerTaskOptions, const MTL4::NewBinaryFunctionCompletionHandlerFunction& completionHandler);
    MTL::ComputePipelineState*          newComputePipelineState(MTL4::ComputePipelineDescriptor* descriptor, MTL4::CompilerTaskOptions* compilerTaskOptions, NS::Error** error);
    MTL::ComputePipelineState*          newComputePipelineState(MTL4::ComputePipelineDescriptor* descriptor, MTL4::PipelineStageDynamicLinkingDescriptor* dynamicLinkingDescriptor, MTL4::CompilerTaskOptions* compilerTaskOptions, NS::Error** error);
    MTL4::CompilerTask*                 newComputePipelineState(MTL4::ComputePipelineDescriptor* descriptor, MTL4::CompilerTaskOptions* compilerTaskOptions, MTL::NewComputePipelineStateCompletionHandler completionHandler);
    MTL4::CompilerTask*                 newComputePipelineState(MTL4::ComputePipelineDescriptor* descriptor, MTL4::PipelineStageDynamicLinkingDescriptor* dynamicLinkingDescriptor, MTL4::CompilerTaskOptions* compilerTaskOptions, MTL::NewComputePipelineStateCompletionHandler completionHandler);
    MTL::DynamicLibrary*                newDynamicLibrary(MTL::Library* library, NS::Error** error);
    MTL::DynamicLibrary*                newDynamicLibrary(NS::URL* url, NS::Error** error);
    MTL4::CompilerTask*                 newDynamicLibrary(MTL::Library* library, MTL::NewDynamicLibraryCompletionHandler completionHandler);
    MTL4::CompilerTask*                 newDynamicLibrary(NS::URL* url, MTL::NewDynamicLibraryCompletionHandler completionHandler);
    MTL::Library*                       newLibrary(MTL4::LibraryDescriptor* descriptor, NS::Error** error);
    MTL4::CompilerTask*                 newLibrary(MTL4::LibraryDescriptor* descriptor, MTL::NewLibraryCompletionHandler completionHandler);
    MTL4::MachineLearningPipelineState* newMachineLearningPipelineState(MTL4::MachineLearningPipelineDescriptor* descriptor, NS::Error** error);
    MTL4::CompilerTask*                 newMachineLearningPipelineState(MTL4::MachineLearningPipelineDescriptor* descriptor, MTL4::NewMachineLearningPipelineStateCompletionHandler completionHandler);
    MTL4::CompilerTask*                 newMachineLearningPipelineState(MTL4::MachineLearningPipelineDescriptor* descriptor, const MTL4::NewMachineLearningPipelineStateCompletionHandlerFunction& completionHandler);
    MTL::RenderPipelineState*           newRenderPipelineState(MTL4::PipelineDescriptor* descriptor, MTL4::CompilerTaskOptions* compilerTaskOptions, NS::Error** error);
    MTL::RenderPipelineState*           newRenderPipelineState(MTL4::PipelineDescriptor* descriptor, MTL4::RenderPipelineDynamicLinkingDescriptor* dynamicLinkingDescriptor, MTL4::CompilerTaskOptions* compilerTaskOptions, NS::Error** error);
    MTL4::CompilerTask*                 newRenderPipelineState(MTL4::PipelineDescriptor* descriptor, MTL4::CompilerTaskOptions* compilerTaskOptions, MTL::NewRenderPipelineStateCompletionHandler completionHandler);
    MTL4::CompilerTask*                 newRenderPipelineState(MTL4::PipelineDescriptor* descriptor, MTL4::RenderPipelineDynamicLinkingDescriptor* dynamicLinkingDescriptor, MTL4::CompilerTaskOptions* compilerTaskOptions, MTL::NewRenderPipelineStateCompletionHandler completionHandler);
    MTL::RenderPipelineState*           newRenderPipelineStateBySpecialization(MTL4::PipelineDescriptor* descriptor, MTL::RenderPipelineState* pipeline, NS::Error** error);
    MTL4::CompilerTask*                 newRenderPipelineStateBySpecialization(MTL4::PipelineDescriptor* descriptor, MTL::RenderPipelineState* pipeline, MTL::NewRenderPipelineStateCompletionHandler completionHandler);
    MTL4::PipelineDataSetSerializer*    pipelineDataSetSerializer() const;

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4CompilerDescriptor;
extern "C" void *OBJC_CLASS_$_MTL4CompilerTaskOptions;
extern "C" void *OBJC_CLASS_$_MTL4Compiler;

_MTL4_INLINE MTL4::CompilerDescriptor* MTL4::CompilerDescriptor::alloc()
{
    return _MTL4_msg_MTL4__CompilerDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4CompilerDescriptor, nullptr);
}

_MTL4_INLINE MTL4::CompilerDescriptor* MTL4::CompilerDescriptor::init() const
{
    return _MTL4_msg_MTL4__CompilerDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE NS::String* MTL4::CompilerDescriptor::label() const
{
    return _MTL4_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::CompilerDescriptor::setLabel(NS::String* label)
{
    _MTL4_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL4_INLINE MTL4::PipelineDataSetSerializer* MTL4::CompilerDescriptor::pipelineDataSetSerializer() const
{
    return _MTL4_msg_MTL4__PipelineDataSetSerializerp_pipelineDataSetSerializer((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::CompilerDescriptor::setPipelineDataSetSerializer(MTL4::PipelineDataSetSerializer* pipelineDataSetSerializer)
{
    _MTL4_msg_v_setPipelineDataSetSerializer__MTL4__PipelineDataSetSerializerp((const void*)this, nullptr, pipelineDataSetSerializer);
}

_MTL4_INLINE MTL4::CompilerTaskOptions* MTL4::CompilerTaskOptions::alloc()
{
    return _MTL4_msg_MTL4__CompilerTaskOptionsp_alloc((const void*)&OBJC_CLASS_$_MTL4CompilerTaskOptions, nullptr);
}

_MTL4_INLINE MTL4::CompilerTaskOptions* MTL4::CompilerTaskOptions::init() const
{
    return _MTL4_msg_MTL4__CompilerTaskOptionsp_init((const void*)this, nullptr);
}

_MTL4_INLINE NS::Array* MTL4::CompilerTaskOptions::lookupArchives() const
{
    return _MTL4_msg_NS__Arrayp_lookupArchives((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::CompilerTaskOptions::setLookupArchives(NS::Array* lookupArchives)
{
    _MTL4_msg_v_setLookupArchives__NS__Arrayp((const void*)this, nullptr, lookupArchives);
}

_MTL4_INLINE MTL::Device* MTL4::Compiler::device() const
{
    return _MTL4_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL4_INLINE NS::String* MTL4::Compiler::label() const
{
    return _MTL4_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::PipelineDataSetSerializer* MTL4::Compiler::pipelineDataSetSerializer() const
{
    return _MTL4_msg_MTL4__PipelineDataSetSerializerp_pipelineDataSetSerializer((const void*)this, nullptr);
}

_MTL4_INLINE MTL::Library* MTL4::Compiler::newLibrary(MTL4::LibraryDescriptor* descriptor, NS::Error** error)
{
    return _MTL4_msg_MTL__Libraryp_newLibraryWithDescriptor_error__MTL4__LibraryDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}

_MTL4_INLINE MTL::DynamicLibrary* MTL4::Compiler::newDynamicLibrary(MTL::Library* library, NS::Error** error)
{
    return _MTL4_msg_MTL__DynamicLibraryp_newDynamicLibrary_error__MTL__Libraryp_NS__Errorpp((const void*)this, nullptr, library, error);
}

_MTL4_INLINE MTL::DynamicLibrary* MTL4::Compiler::newDynamicLibrary(NS::URL* url, NS::Error** error)
{
    return _MTL4_msg_MTL__DynamicLibraryp_newDynamicLibraryWithURL_error__NS__URLp_NS__Errorpp((const void*)this, nullptr, url, error);
}

_MTL4_INLINE MTL::ComputePipelineState* MTL4::Compiler::newComputePipelineState(MTL4::ComputePipelineDescriptor* descriptor, MTL4::CompilerTaskOptions* compilerTaskOptions, NS::Error** error)
{
    return _MTL4_msg_MTL__ComputePipelineStatep_newComputePipelineStateWithDescriptor_compilerTaskOptions_error__MTL4__ComputePipelineDescriptorp_MTL4__CompilerTaskOptionsp_NS__Errorpp((const void*)this, nullptr, descriptor, compilerTaskOptions, error);
}

_MTL4_INLINE MTL::ComputePipelineState* MTL4::Compiler::newComputePipelineState(MTL4::ComputePipelineDescriptor* descriptor, MTL4::PipelineStageDynamicLinkingDescriptor* dynamicLinkingDescriptor, MTL4::CompilerTaskOptions* compilerTaskOptions, NS::Error** error)
{
    return _MTL4_msg_MTL__ComputePipelineStatep_newComputePipelineStateWithDescriptor_dynamicLinkingDescriptor_compilerTaskOptions_error__MTL4__ComputePipelineDescriptorp_MTL4__PipelineStageDynamicLinkingDescriptorp_MTL4__CompilerTaskOptionsp_NS__Errorpp((const void*)this, nullptr, descriptor, dynamicLinkingDescriptor, compilerTaskOptions, error);
}

_MTL4_INLINE MTL::RenderPipelineState* MTL4::Compiler::newRenderPipelineState(MTL4::PipelineDescriptor* descriptor, MTL4::CompilerTaskOptions* compilerTaskOptions, NS::Error** error)
{
    return _MTL4_msg_MTL__RenderPipelineStatep_newRenderPipelineStateWithDescriptor_compilerTaskOptions_error__MTL4__PipelineDescriptorp_MTL4__CompilerTaskOptionsp_NS__Errorpp((const void*)this, nullptr, descriptor, compilerTaskOptions, error);
}

_MTL4_INLINE MTL::RenderPipelineState* MTL4::Compiler::newRenderPipelineState(MTL4::PipelineDescriptor* descriptor, MTL4::RenderPipelineDynamicLinkingDescriptor* dynamicLinkingDescriptor, MTL4::CompilerTaskOptions* compilerTaskOptions, NS::Error** error)
{
    return _MTL4_msg_MTL__RenderPipelineStatep_newRenderPipelineStateWithDescriptor_dynamicLinkingDescriptor_compilerTaskOptions_error__MTL4__PipelineDescriptorp_MTL4__RenderPipelineDynamicLinkingDescriptorp_MTL4__CompilerTaskOptionsp_NS__Errorpp((const void*)this, nullptr, descriptor, dynamicLinkingDescriptor, compilerTaskOptions, error);
}

_MTL4_INLINE MTL::RenderPipelineState* MTL4::Compiler::newRenderPipelineStateBySpecialization(MTL4::PipelineDescriptor* descriptor, MTL::RenderPipelineState* pipeline, NS::Error** error)
{
    return _MTL4_msg_MTL__RenderPipelineStatep_newRenderPipelineStateBySpecializationWithDescriptor_pipeline_error__MTL4__PipelineDescriptorp_MTL__RenderPipelineStatep_NS__Errorpp((const void*)this, nullptr, descriptor, pipeline, error);
}

_MTL4_INLINE MTL4::BinaryFunction* MTL4::Compiler::newBinaryFunction(MTL4::BinaryFunctionDescriptor* descriptor, MTL4::CompilerTaskOptions* compilerTaskOptions, NS::Error** error)
{
    return _MTL4_msg_MTL4__BinaryFunctionp_newBinaryFunctionWithDescriptor_compilerTaskOptions_error__MTL4__BinaryFunctionDescriptorp_MTL4__CompilerTaskOptionsp_NS__Errorpp((const void*)this, nullptr, descriptor, compilerTaskOptions, error);
}

_MTL4_INLINE MTL4::CompilerTask* MTL4::Compiler::newLibrary(MTL4::LibraryDescriptor* descriptor, MTL::NewLibraryCompletionHandler completionHandler)
{
    return _MTL4_msg_MTL4__CompilerTaskp_newLibraryWithDescriptor_completionHandler__MTL4__LibraryDescriptorp_MTL__NewLibraryCompletionHandler((const void*)this, nullptr, descriptor, completionHandler);
}

_MTL4_INLINE MTL4::CompilerTask* MTL4::Compiler::newDynamicLibrary(MTL::Library* library, MTL::NewDynamicLibraryCompletionHandler completionHandler)
{
    return _MTL4_msg_MTL4__CompilerTaskp_newDynamicLibrary_completionHandler__MTL__Libraryp_MTL__NewDynamicLibraryCompletionHandler((const void*)this, nullptr, library, completionHandler);
}

_MTL4_INLINE MTL4::CompilerTask* MTL4::Compiler::newDynamicLibrary(NS::URL* url, MTL::NewDynamicLibraryCompletionHandler completionHandler)
{
    return _MTL4_msg_MTL4__CompilerTaskp_newDynamicLibraryWithURL_completionHandler__NS__URLp_MTL__NewDynamicLibraryCompletionHandler((const void*)this, nullptr, url, completionHandler);
}

_MTL4_INLINE MTL4::CompilerTask* MTL4::Compiler::newComputePipelineState(MTL4::ComputePipelineDescriptor* descriptor, MTL4::CompilerTaskOptions* compilerTaskOptions, MTL::NewComputePipelineStateCompletionHandler completionHandler)
{
    return _MTL4_msg_MTL4__CompilerTaskp_newComputePipelineStateWithDescriptor_compilerTaskOptions_completionHandler__MTL4__ComputePipelineDescriptorp_MTL4__CompilerTaskOptionsp_MTL__NewComputePipelineStateCompletionHandler((const void*)this, nullptr, descriptor, compilerTaskOptions, completionHandler);
}

_MTL4_INLINE MTL4::CompilerTask* MTL4::Compiler::newComputePipelineState(MTL4::ComputePipelineDescriptor* descriptor, MTL4::PipelineStageDynamicLinkingDescriptor* dynamicLinkingDescriptor, MTL4::CompilerTaskOptions* compilerTaskOptions, MTL::NewComputePipelineStateCompletionHandler completionHandler)
{
    return _MTL4_msg_MTL4__CompilerTaskp_newComputePipelineStateWithDescriptor_dynamicLinkingDescriptor_compilerTaskOptions_completionHandler__MTL4__ComputePipelineDescriptorp_MTL4__PipelineStageDynamicLinkingDescriptorp_MTL4__CompilerTaskOptionsp_MTL__NewComputePipelineStateCompletionHandler((const void*)this, nullptr, descriptor, dynamicLinkingDescriptor, compilerTaskOptions, completionHandler);
}

_MTL4_INLINE MTL4::CompilerTask* MTL4::Compiler::newRenderPipelineState(MTL4::PipelineDescriptor* descriptor, MTL4::CompilerTaskOptions* compilerTaskOptions, MTL::NewRenderPipelineStateCompletionHandler completionHandler)
{
    return _MTL4_msg_MTL4__CompilerTaskp_newRenderPipelineStateWithDescriptor_compilerTaskOptions_completionHandler__MTL4__PipelineDescriptorp_MTL4__CompilerTaskOptionsp_MTL__NewRenderPipelineStateCompletionHandler((const void*)this, nullptr, descriptor, compilerTaskOptions, completionHandler);
}

_MTL4_INLINE MTL4::CompilerTask* MTL4::Compiler::newRenderPipelineState(MTL4::PipelineDescriptor* descriptor, MTL4::RenderPipelineDynamicLinkingDescriptor* dynamicLinkingDescriptor, MTL4::CompilerTaskOptions* compilerTaskOptions, MTL::NewRenderPipelineStateCompletionHandler completionHandler)
{
    return _MTL4_msg_MTL4__CompilerTaskp_newRenderPipelineStateWithDescriptor_dynamicLinkingDescriptor_compilerTaskOptions_completionHandler__MTL4__PipelineDescriptorp_MTL4__RenderPipelineDynamicLinkingDescriptorp_MTL4__CompilerTaskOptionsp_MTL__NewRenderPipelineStateCompletionHandler((const void*)this, nullptr, descriptor, dynamicLinkingDescriptor, compilerTaskOptions, completionHandler);
}

_MTL4_INLINE MTL4::CompilerTask* MTL4::Compiler::newRenderPipelineStateBySpecialization(MTL4::PipelineDescriptor* descriptor, MTL::RenderPipelineState* pipeline, MTL::NewRenderPipelineStateCompletionHandler completionHandler)
{
    return _MTL4_msg_MTL4__CompilerTaskp_newRenderPipelineStateBySpecializationWithDescriptor_pipeline_completionHandler__MTL4__PipelineDescriptorp_MTL__RenderPipelineStatep_MTL__NewRenderPipelineStateCompletionHandler((const void*)this, nullptr, descriptor, pipeline, completionHandler);
}

_MTL4_INLINE MTL4::CompilerTask* MTL4::Compiler::newBinaryFunction(MTL4::BinaryFunctionDescriptor* descriptor, MTL4::CompilerTaskOptions* compilerTaskOptions, MTL4::NewBinaryFunctionCompletionHandler completionHandler)
{
    return _MTL4_msg_MTL4__CompilerTaskp_newBinaryFunctionWithDescriptor_compilerTaskOptions_completionHandler__MTL4__BinaryFunctionDescriptorp_MTL4__CompilerTaskOptionsp_MTL4__NewBinaryFunctionCompletionHandler((const void*)this, nullptr, descriptor, compilerTaskOptions, completionHandler);
}

_MTL4_INLINE MTL4::CompilerTask* MTL4::Compiler::newBinaryFunction(MTL4::BinaryFunctionDescriptor* descriptor, MTL4::CompilerTaskOptions* compilerTaskOptions, const MTL4::NewBinaryFunctionCompletionHandlerFunction& completionHandler)
{
    __block MTL4::NewBinaryFunctionCompletionHandlerFunction blockFunction = completionHandler;
    return newBinaryFunction(descriptor, compilerTaskOptions, ^(MTL4::BinaryFunction* x0, NS::Error* x1) { blockFunction(x0, x1); });
}

_MTL4_INLINE MTL4::MachineLearningPipelineState* MTL4::Compiler::newMachineLearningPipelineState(MTL4::MachineLearningPipelineDescriptor* descriptor, NS::Error** error)
{
    return _MTL4_msg_MTL4__MachineLearningPipelineStatep_newMachineLearningPipelineStateWithDescriptor_error__MTL4__MachineLearningPipelineDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}

_MTL4_INLINE MTL4::CompilerTask* MTL4::Compiler::newMachineLearningPipelineState(MTL4::MachineLearningPipelineDescriptor* descriptor, MTL4::NewMachineLearningPipelineStateCompletionHandler completionHandler)
{
    return _MTL4_msg_MTL4__CompilerTaskp_newMachineLearningPipelineStateWithDescriptor_completionHandler__MTL4__MachineLearningPipelineDescriptorp_MTL4__NewMachineLearningPipelineStateCompletionHandler((const void*)this, nullptr, descriptor, completionHandler);
}

_MTL4_INLINE MTL4::CompilerTask* MTL4::Compiler::newMachineLearningPipelineState(MTL4::MachineLearningPipelineDescriptor* descriptor, const MTL4::NewMachineLearningPipelineStateCompletionHandlerFunction& completionHandler)
{
    __block MTL4::NewMachineLearningPipelineStateCompletionHandlerFunction blockFunction = completionHandler;
    return newMachineLearningPipelineState(descriptor, ^(MTL4::MachineLearningPipelineState* x0, NS::Error* x1) { blockFunction(x0, x1); });
}
