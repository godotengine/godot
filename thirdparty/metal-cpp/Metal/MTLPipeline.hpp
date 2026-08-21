#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL
{

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


class PipelineBufferDescriptor;
class PipelineBufferDescriptorArray;

class PipelineBufferDescriptor : public NS::Copying<PipelineBufferDescriptor>
{
public:
    static PipelineBufferDescriptor* alloc();
    PipelineBufferDescriptor*        init() const;

    MTL::Mutability mutability() const;
    void            setMutability(MTL::Mutability mutability);

};

class PipelineBufferDescriptorArray : public NS::Referencing<PipelineBufferDescriptorArray>
{
public:
    static PipelineBufferDescriptorArray* alloc();
    PipelineBufferDescriptorArray*        init() const;

    MTL::PipelineBufferDescriptor* object(NS::UInteger bufferIndex);
    void                           setObject(MTL::PipelineBufferDescriptor* buffer, NS::UInteger bufferIndex);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLPipelineBufferDescriptor;
extern "C" void *OBJC_CLASS_$_MTLPipelineBufferDescriptorArray;

_MTL_INLINE MTL::PipelineBufferDescriptor* MTL::PipelineBufferDescriptor::alloc()
{
    return _MTL_msg_MTL__PipelineBufferDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLPipelineBufferDescriptor, nullptr);
}

_MTL_INLINE MTL::PipelineBufferDescriptor* MTL::PipelineBufferDescriptor::init() const
{
    return _MTL_msg_MTL__PipelineBufferDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::Mutability MTL::PipelineBufferDescriptor::mutability() const
{
    return _MTL_msg_MTL__Mutability_mutability((const void*)this, nullptr);
}

_MTL_INLINE void MTL::PipelineBufferDescriptor::setMutability(MTL::Mutability mutability)
{
    _MTL_msg_v_setMutability__MTL__Mutability((const void*)this, nullptr, mutability);
}

_MTL_INLINE MTL::PipelineBufferDescriptorArray* MTL::PipelineBufferDescriptorArray::alloc()
{
    return _MTL_msg_MTL__PipelineBufferDescriptorArrayp_alloc((const void*)&OBJC_CLASS_$_MTLPipelineBufferDescriptorArray, nullptr);
}

_MTL_INLINE MTL::PipelineBufferDescriptorArray* MTL::PipelineBufferDescriptorArray::init() const
{
    return _MTL_msg_MTL__PipelineBufferDescriptorArrayp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::PipelineBufferDescriptor* MTL::PipelineBufferDescriptorArray::object(NS::UInteger bufferIndex)
{
    return _MTL_msg_MTL__PipelineBufferDescriptorp_objectAtIndexedSubscript__NS__UInteger((const void*)this, nullptr, bufferIndex);
}

_MTL_INLINE void MTL::PipelineBufferDescriptorArray::setObject(MTL::PipelineBufferDescriptor* buffer, NS::UInteger bufferIndex)
{
    _MTL_msg_v_setObject_atIndexedSubscript__MTL__PipelineBufferDescriptorp_NS__UInteger((const void*)this, nullptr, buffer, bufferIndex);
}
