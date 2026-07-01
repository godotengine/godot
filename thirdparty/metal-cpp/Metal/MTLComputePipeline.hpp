#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTLAllocation.hpp"

namespace MTL {
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
    enum ShaderValidation : NS::Integer;
}
namespace MTL4 {
    class BinaryFunction;
}
namespace NS {
    class Array;
    class Error;
    class String;
}

namespace MTL
{

class ComputePipelineReflection;
class ComputePipelineDescriptor;
class ComputePipelineState;

class ComputePipelineReflection : public NS::Referencing<ComputePipelineReflection>
{
public:
    static ComputePipelineReflection* alloc();
    ComputePipelineReflection*        init() const;

    NS::Array* arguments() const;
    NS::Array* bindings() const;

};

class ComputePipelineDescriptor : public NS::Copying<ComputePipelineDescriptor>
{
public:
    static ComputePipelineDescriptor* alloc();
    ComputePipelineDescriptor*        init() const;

    NS::Array*                          binaryArchives() const;
    MTL::PipelineBufferDescriptorArray* buffers() const;
    MTL::Function*                      computeFunction() const;
    NS::Array*                          insertLibraries() const;
    NS::String*                         label() const;
    MTL::LinkedFunctions*               linkedFunctions() const;
    NS::UInteger                        maxCallStackDepth() const;
    NS::UInteger                        maxTotalThreadsPerThreadgroup() const;
    NS::Array*                          preloadedLibraries() const;
    MTL::Size                           requiredThreadsPerThreadgroup() const;
    void                                reset();
    void                                setBinaryArchives(NS::Array* binaryArchives);
    void                                setComputeFunction(MTL::Function* computeFunction);
    void                                setInsertLibraries(NS::Array* insertLibraries);
    void                                setLabel(NS::String* label);
    void                                setLinkedFunctions(MTL::LinkedFunctions* linkedFunctions);
    void                                setMaxCallStackDepth(NS::UInteger maxCallStackDepth);
    void                                setMaxTotalThreadsPerThreadgroup(NS::UInteger maxTotalThreadsPerThreadgroup);
    void                                setPreloadedLibraries(NS::Array* preloadedLibraries);
    void                                setRequiredThreadsPerThreadgroup(MTL::Size requiredThreadsPerThreadgroup);
    void                                setShaderValidation(MTL::ShaderValidation shaderValidation);
    void                                setStageInputDescriptor(MTL::StageInputOutputDescriptor* stageInputDescriptor);
    void                                setSupportAddingBinaryFunctions(bool supportAddingBinaryFunctions);
    void                                setSupportIndirectCommandBuffers(bool supportIndirectCommandBuffers);
    void                                setThreadGroupSizeIsMultipleOfThreadExecutionWidth(bool threadGroupSizeIsMultipleOfThreadExecutionWidth);
    MTL::ShaderValidation               shaderValidation() const;
    MTL::StageInputOutputDescriptor*    stageInputDescriptor() const;
    bool                                supportAddingBinaryFunctions() const;
    bool                                supportIndirectCommandBuffers() const;
    bool                                threadGroupSizeIsMultipleOfThreadExecutionWidth() const;

};

class ComputePipelineState : public NS::Referencing<ComputePipelineState, MTL::Allocation>
{
public:
    MTL::Device*                    device() const;
    MTL::FunctionHandle*            functionHandle(NS::String* name);
    MTL::FunctionHandle*            functionHandle(MTL4::BinaryFunction* function);
    MTL::FunctionHandle*            functionHandle(MTL::Function* function);
    MTL::ResourceID                 gpuResourceID() const;
    NS::UInteger                    imageblockMemoryLength(MTL::Size imageblockDimensions);
    NS::String*                     label() const;
    NS::UInteger                    maxTotalThreadsPerThreadgroup() const;
    MTL::ComputePipelineState*      newComputePipelineState(NS::Array* additionalBinaryFunctions, NS::Error** error);
    MTL::IntersectionFunctionTable* newIntersectionFunctionTable(MTL::IntersectionFunctionTableDescriptor* descriptor);
    MTL::VisibleFunctionTable*      newVisibleFunctionTable(MTL::VisibleFunctionTableDescriptor* descriptor);
    MTL::ComputePipelineReflection* reflection() const;
    MTL::Size                       requiredThreadsPerThreadgroup() const;
    MTL::ShaderValidation           shaderValidation() const;
    NS::UInteger                    staticThreadgroupMemoryLength() const;
    bool                            supportIndirectCommandBuffers() const;
    NS::UInteger                    threadExecutionWidth() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLComputePipelineReflection;
extern "C" void *OBJC_CLASS_$_MTLComputePipelineDescriptor;
extern "C" void *OBJC_CLASS_$_MTLComputePipelineState;

_MTL_INLINE MTL::ComputePipelineReflection* MTL::ComputePipelineReflection::alloc()
{
    return _MTL_msg_MTL__ComputePipelineReflectionp_alloc((const void*)&OBJC_CLASS_$_MTLComputePipelineReflection, nullptr);
}

_MTL_INLINE MTL::ComputePipelineReflection* MTL::ComputePipelineReflection::init() const
{
    return _MTL_msg_MTL__ComputePipelineReflectionp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::Array* MTL::ComputePipelineReflection::bindings() const
{
    return _MTL_msg_NS__Arrayp_bindings((const void*)this, nullptr);
}

_MTL_INLINE NS::Array* MTL::ComputePipelineReflection::arguments() const
{
    return _MTL_msg_NS__Arrayp_arguments((const void*)this, nullptr);
}

_MTL_INLINE MTL::ComputePipelineDescriptor* MTL::ComputePipelineDescriptor::alloc()
{
    return _MTL_msg_MTL__ComputePipelineDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLComputePipelineDescriptor, nullptr);
}

_MTL_INLINE MTL::ComputePipelineDescriptor* MTL::ComputePipelineDescriptor::init() const
{
    return _MTL_msg_MTL__ComputePipelineDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::ComputePipelineDescriptor::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE MTL::Function* MTL::ComputePipelineDescriptor::computeFunction() const
{
    return _MTL_msg_MTL__Functionp_computeFunction((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setComputeFunction(MTL::Function* computeFunction)
{
    _MTL_msg_v_setComputeFunction__MTL__Functionp((const void*)this, nullptr, computeFunction);
}

_MTL_INLINE bool MTL::ComputePipelineDescriptor::threadGroupSizeIsMultipleOfThreadExecutionWidth() const
{
    return _MTL_msg_bool_threadGroupSizeIsMultipleOfThreadExecutionWidth((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setThreadGroupSizeIsMultipleOfThreadExecutionWidth(bool threadGroupSizeIsMultipleOfThreadExecutionWidth)
{
    _MTL_msg_v_setThreadGroupSizeIsMultipleOfThreadExecutionWidth__bool((const void*)this, nullptr, threadGroupSizeIsMultipleOfThreadExecutionWidth);
}

_MTL_INLINE NS::UInteger MTL::ComputePipelineDescriptor::maxTotalThreadsPerThreadgroup() const
{
    return _MTL_msg_NS__UInteger_maxTotalThreadsPerThreadgroup((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setMaxTotalThreadsPerThreadgroup(NS::UInteger maxTotalThreadsPerThreadgroup)
{
    _MTL_msg_v_setMaxTotalThreadsPerThreadgroup__NS__UInteger((const void*)this, nullptr, maxTotalThreadsPerThreadgroup);
}

_MTL_INLINE MTL::StageInputOutputDescriptor* MTL::ComputePipelineDescriptor::stageInputDescriptor() const
{
    return _MTL_msg_MTL__StageInputOutputDescriptorp_stageInputDescriptor((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setStageInputDescriptor(MTL::StageInputOutputDescriptor* stageInputDescriptor)
{
    _MTL_msg_v_setStageInputDescriptor__MTL__StageInputOutputDescriptorp((const void*)this, nullptr, stageInputDescriptor);
}

_MTL_INLINE MTL::PipelineBufferDescriptorArray* MTL::ComputePipelineDescriptor::buffers() const
{
    return _MTL_msg_MTL__PipelineBufferDescriptorArrayp_buffers((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::ComputePipelineDescriptor::supportIndirectCommandBuffers() const
{
    return _MTL_msg_bool_supportIndirectCommandBuffers((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setSupportIndirectCommandBuffers(bool supportIndirectCommandBuffers)
{
    _MTL_msg_v_setSupportIndirectCommandBuffers__bool((const void*)this, nullptr, supportIndirectCommandBuffers);
}

_MTL_INLINE NS::Array* MTL::ComputePipelineDescriptor::insertLibraries() const
{
    return _MTL_msg_NS__Arrayp_insertLibraries((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setInsertLibraries(NS::Array* insertLibraries)
{
    _MTL_msg_v_setInsertLibraries__NS__Arrayp((const void*)this, nullptr, insertLibraries);
}

_MTL_INLINE NS::Array* MTL::ComputePipelineDescriptor::preloadedLibraries() const
{
    return _MTL_msg_NS__Arrayp_preloadedLibraries((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setPreloadedLibraries(NS::Array* preloadedLibraries)
{
    _MTL_msg_v_setPreloadedLibraries__NS__Arrayp((const void*)this, nullptr, preloadedLibraries);
}

_MTL_INLINE NS::Array* MTL::ComputePipelineDescriptor::binaryArchives() const
{
    return _MTL_msg_NS__Arrayp_binaryArchives((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setBinaryArchives(NS::Array* binaryArchives)
{
    _MTL_msg_v_setBinaryArchives__NS__Arrayp((const void*)this, nullptr, binaryArchives);
}

_MTL_INLINE MTL::LinkedFunctions* MTL::ComputePipelineDescriptor::linkedFunctions() const
{
    return _MTL_msg_MTL__LinkedFunctionsp_linkedFunctions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setLinkedFunctions(MTL::LinkedFunctions* linkedFunctions)
{
    _MTL_msg_v_setLinkedFunctions__MTL__LinkedFunctionsp((const void*)this, nullptr, linkedFunctions);
}

_MTL_INLINE bool MTL::ComputePipelineDescriptor::supportAddingBinaryFunctions() const
{
    return _MTL_msg_bool_supportAddingBinaryFunctions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setSupportAddingBinaryFunctions(bool supportAddingBinaryFunctions)
{
    _MTL_msg_v_setSupportAddingBinaryFunctions__bool((const void*)this, nullptr, supportAddingBinaryFunctions);
}

_MTL_INLINE NS::UInteger MTL::ComputePipelineDescriptor::maxCallStackDepth() const
{
    return _MTL_msg_NS__UInteger_maxCallStackDepth((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setMaxCallStackDepth(NS::UInteger maxCallStackDepth)
{
    _MTL_msg_v_setMaxCallStackDepth__NS__UInteger((const void*)this, nullptr, maxCallStackDepth);
}

_MTL_INLINE MTL::ShaderValidation MTL::ComputePipelineDescriptor::shaderValidation() const
{
    return _MTL_msg_MTL__ShaderValidation_shaderValidation((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setShaderValidation(MTL::ShaderValidation shaderValidation)
{
    _MTL_msg_v_setShaderValidation__MTL__ShaderValidation((const void*)this, nullptr, shaderValidation);
}

_MTL_INLINE MTL::Size MTL::ComputePipelineDescriptor::requiredThreadsPerThreadgroup() const
{
    return _MTL_msg_MTL__Size_requiredThreadsPerThreadgroup((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::setRequiredThreadsPerThreadgroup(MTL::Size requiredThreadsPerThreadgroup)
{
    _MTL_msg_v_setRequiredThreadsPerThreadgroup__MTL__Size((const void*)this, nullptr, requiredThreadsPerThreadgroup);
}

_MTL_INLINE void MTL::ComputePipelineDescriptor::reset()
{
    _MTL_msg_v_reset((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::ComputePipelineState::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE MTL::ComputePipelineReflection* MTL::ComputePipelineState::reflection() const
{
    return _MTL_msg_MTL__ComputePipelineReflectionp_reflection((const void*)this, nullptr);
}

_MTL_INLINE MTL::Device* MTL::ComputePipelineState::device() const
{
    return _MTL_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::ComputePipelineState::maxTotalThreadsPerThreadgroup() const
{
    return _MTL_msg_NS__UInteger_maxTotalThreadsPerThreadgroup((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::ComputePipelineState::threadExecutionWidth() const
{
    return _MTL_msg_NS__UInteger_threadExecutionWidth((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::ComputePipelineState::staticThreadgroupMemoryLength() const
{
    return _MTL_msg_NS__UInteger_staticThreadgroupMemoryLength((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::ComputePipelineState::supportIndirectCommandBuffers() const
{
    return _MTL_msg_bool_supportIndirectCommandBuffers((const void*)this, nullptr);
}

_MTL_INLINE MTL::ResourceID MTL::ComputePipelineState::gpuResourceID() const
{
    return _MTL_msg_MTL__ResourceID_gpuResourceID((const void*)this, nullptr);
}

_MTL_INLINE MTL::ShaderValidation MTL::ComputePipelineState::shaderValidation() const
{
    return _MTL_msg_MTL__ShaderValidation_shaderValidation((const void*)this, nullptr);
}

_MTL_INLINE MTL::Size MTL::ComputePipelineState::requiredThreadsPerThreadgroup() const
{
    return _MTL_msg_MTL__Size_requiredThreadsPerThreadgroup((const void*)this, nullptr);
}

_MTL_INLINE MTL::FunctionHandle* MTL::ComputePipelineState::functionHandle(NS::String* name)
{
    return _MTL_msg_MTL__FunctionHandlep_functionHandleWithName__NS__Stringp((const void*)this, nullptr, name);
}

_MTL_INLINE MTL::FunctionHandle* MTL::ComputePipelineState::functionHandle(MTL4::BinaryFunction* function)
{
    return _MTL_msg_MTL__FunctionHandlep_functionHandleWithBinaryFunction__MTL4__BinaryFunctionp((const void*)this, nullptr, function);
}

_MTL_INLINE MTL::ComputePipelineState* MTL::ComputePipelineState::newComputePipelineState(NS::Array* additionalBinaryFunctions, NS::Error** error)
{
    return _MTL_msg_MTL__ComputePipelineStatep_newComputePipelineStateWithBinaryFunctions_error__NS__Arrayp_NS__Errorpp((const void*)this, nullptr, additionalBinaryFunctions, error);
}

_MTL_INLINE NS::UInteger MTL::ComputePipelineState::imageblockMemoryLength(MTL::Size imageblockDimensions)
{
    return _MTL_msg_NS__UInteger_imageblockMemoryLengthForDimensions__MTL__Size((const void*)this, nullptr, imageblockDimensions);
}

_MTL_INLINE MTL::FunctionHandle* MTL::ComputePipelineState::functionHandle(MTL::Function* function)
{
    return _MTL_msg_MTL__FunctionHandlep_functionHandleWithFunction__MTL__Functionp((const void*)this, nullptr, function);
}

_MTL_INLINE MTL::VisibleFunctionTable* MTL::ComputePipelineState::newVisibleFunctionTable(MTL::VisibleFunctionTableDescriptor* descriptor)
{
    return _MTL_msg_MTL__VisibleFunctionTablep_newVisibleFunctionTableWithDescriptor__MTL__VisibleFunctionTableDescriptorp((const void*)this, nullptr, descriptor);
}

_MTL_INLINE MTL::IntersectionFunctionTable* MTL::ComputePipelineState::newIntersectionFunctionTable(MTL::IntersectionFunctionTableDescriptor* descriptor)
{
    return _MTL_msg_MTL__IntersectionFunctionTablep_newIntersectionFunctionTableWithDescriptor__MTL__IntersectionFunctionTableDescriptorp((const void*)this, nullptr, descriptor);
}
