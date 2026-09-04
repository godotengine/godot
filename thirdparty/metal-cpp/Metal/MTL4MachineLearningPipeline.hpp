#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTL4PipelineState.hpp"
#include "MTLAllocation.hpp"

namespace MTL {
    class Device;
    class TensorExtents;
}
namespace MTL4 {
    class FunctionDescriptor;
}
namespace NS {
    class Array;
    class String;
}

namespace MTL4
{

class MachineLearningPipelineDescriptor;
class MachineLearningPipelineReflection;
class MachineLearningPipelineState;

class MachineLearningPipelineDescriptor : public NS::Referencing<MachineLearningPipelineDescriptor, MTL4::PipelineDescriptor>
{
public:
    static MachineLearningPipelineDescriptor* alloc();
    MachineLearningPipelineDescriptor*        init() const;

    MTL::TensorExtents*       inputDimensions(NS::Integer bufferIndex);
    NS::String*               label() const;
    MTL4::FunctionDescriptor* machineLearningFunctionDescriptor() const;
    void                      reset();
    void                      setInputDimensions(MTL::TensorExtents* dimensions, NS::Integer bufferIndex);
    void                      setInputDimensions(NS::Array* dimensions, NS::Range range);
    void                      setLabel(NS::String* label);
    void                      setMachineLearningFunctionDescriptor(MTL4::FunctionDescriptor* machineLearningFunctionDescriptor);

};

class MachineLearningPipelineReflection : public NS::Referencing<MachineLearningPipelineReflection>
{
public:
    static MachineLearningPipelineReflection* alloc();
    MachineLearningPipelineReflection*        init() const;

    NS::Array* bindings() const;

};

class MachineLearningPipelineState : public NS::Referencing<MachineLearningPipelineState, MTL::Allocation>
{
public:
    MTL::Device*                             device() const;
    NS::UInteger                             intermediatesHeapSize() const;
    NS::String*                              label() const;
    MTL4::MachineLearningPipelineReflection* reflection() const;

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4MachineLearningPipelineDescriptor;
extern "C" void *OBJC_CLASS_$_MTL4MachineLearningPipelineReflection;
extern "C" void *OBJC_CLASS_$_MTL4MachineLearningPipelineState;

_MTL4_INLINE MTL4::MachineLearningPipelineDescriptor* MTL4::MachineLearningPipelineDescriptor::alloc()
{
    return _MTL4_msg_MTL4__MachineLearningPipelineDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4MachineLearningPipelineDescriptor, nullptr);
}

_MTL4_INLINE MTL4::MachineLearningPipelineDescriptor* MTL4::MachineLearningPipelineDescriptor::init() const
{
    return _MTL4_msg_MTL4__MachineLearningPipelineDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE NS::String* MTL4::MachineLearningPipelineDescriptor::label() const
{
    return _MTL4_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MachineLearningPipelineDescriptor::setLabel(NS::String* label)
{
    _MTL4_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL4_INLINE MTL4::FunctionDescriptor* MTL4::MachineLearningPipelineDescriptor::machineLearningFunctionDescriptor() const
{
    return _MTL4_msg_MTL4__FunctionDescriptorp_machineLearningFunctionDescriptor((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::MachineLearningPipelineDescriptor::setMachineLearningFunctionDescriptor(MTL4::FunctionDescriptor* machineLearningFunctionDescriptor)
{
    _MTL4_msg_v_setMachineLearningFunctionDescriptor__MTL4__FunctionDescriptorp((const void*)this, nullptr, machineLearningFunctionDescriptor);
}

_MTL4_INLINE void MTL4::MachineLearningPipelineDescriptor::setInputDimensions(MTL::TensorExtents* dimensions, NS::Integer bufferIndex)
{
    _MTL4_msg_v_setInputDimensions_atBufferIndex__MTL__TensorExtentsp_NS__Integer((const void*)this, nullptr, dimensions, bufferIndex);
}

_MTL4_INLINE void MTL4::MachineLearningPipelineDescriptor::setInputDimensions(NS::Array* dimensions, NS::Range range)
{
    _MTL4_msg_v_setInputDimensions_withRange__NS__Arrayp_NS__Range((const void*)this, nullptr, dimensions, range);
}

_MTL4_INLINE MTL::TensorExtents* MTL4::MachineLearningPipelineDescriptor::inputDimensions(NS::Integer bufferIndex)
{
    return _MTL4_msg_MTL__TensorExtentsp_inputDimensionsAtBufferIndex__NS__Integer((const void*)this, nullptr, bufferIndex);
}

_MTL4_INLINE void MTL4::MachineLearningPipelineDescriptor::reset()
{
    _MTL4_msg_v_reset((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::MachineLearningPipelineReflection* MTL4::MachineLearningPipelineReflection::alloc()
{
    return _MTL4_msg_MTL4__MachineLearningPipelineReflectionp_alloc((const void*)&OBJC_CLASS_$_MTL4MachineLearningPipelineReflection, nullptr);
}

_MTL4_INLINE MTL4::MachineLearningPipelineReflection* MTL4::MachineLearningPipelineReflection::init() const
{
    return _MTL4_msg_MTL4__MachineLearningPipelineReflectionp_init((const void*)this, nullptr);
}

_MTL4_INLINE NS::Array* MTL4::MachineLearningPipelineReflection::bindings() const
{
    return _MTL4_msg_NS__Arrayp_bindings((const void*)this, nullptr);
}

_MTL4_INLINE NS::String* MTL4::MachineLearningPipelineState::label() const
{
    return _MTL4_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL4_INLINE MTL::Device* MTL4::MachineLearningPipelineState::device() const
{
    return _MTL4_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::MachineLearningPipelineReflection* MTL4::MachineLearningPipelineState::reflection() const
{
    return _MTL4_msg_MTL4__MachineLearningPipelineReflectionp_reflection((const void*)this, nullptr);
}

_MTL4_INLINE NS::UInteger MTL4::MachineLearningPipelineState::intermediatesHeapSize() const
{
    return _MTL4_msg_NS__UInteger_intermediatesHeapSize((const void*)this, nullptr);
}
