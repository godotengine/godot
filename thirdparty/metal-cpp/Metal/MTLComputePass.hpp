#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class CounterSampleBuffer;
    enum DispatchType : NS::UInteger;
}

namespace MTL
{

class ComputePassSampleBufferAttachmentDescriptor;
class ComputePassSampleBufferAttachmentDescriptorArray;
class ComputePassDescriptor;

class ComputePassSampleBufferAttachmentDescriptor : public NS::Copying<ComputePassSampleBufferAttachmentDescriptor>
{
public:
    static ComputePassSampleBufferAttachmentDescriptor* alloc();
    ComputePassSampleBufferAttachmentDescriptor*        init() const;

    NS::UInteger              endOfEncoderSampleIndex() const;
    MTL::CounterSampleBuffer* sampleBuffer() const;
    void                      setEndOfEncoderSampleIndex(NS::UInteger endOfEncoderSampleIndex);
    void                      setSampleBuffer(MTL::CounterSampleBuffer* sampleBuffer);
    void                      setStartOfEncoderSampleIndex(NS::UInteger startOfEncoderSampleIndex);
    NS::UInteger              startOfEncoderSampleIndex() const;

};

class ComputePassSampleBufferAttachmentDescriptorArray : public NS::Referencing<ComputePassSampleBufferAttachmentDescriptorArray>
{
public:
    static ComputePassSampleBufferAttachmentDescriptorArray* alloc();
    ComputePassSampleBufferAttachmentDescriptorArray*        init() const;

    MTL::ComputePassSampleBufferAttachmentDescriptor* object(NS::UInteger attachmentIndex);
    void                                              setObject(MTL::ComputePassSampleBufferAttachmentDescriptor* attachment, NS::UInteger attachmentIndex);

};

class ComputePassDescriptor : public NS::Copying<ComputePassDescriptor>
{
public:
    static ComputePassDescriptor* alloc();
    ComputePassDescriptor*        init() const;

    static MTL::ComputePassDescriptor* computePassDescriptor();

    MTL::DispatchType                                      dispatchType() const;
    MTL::ComputePassSampleBufferAttachmentDescriptorArray* sampleBufferAttachments() const;
    void                                                   setDispatchType(MTL::DispatchType dispatchType);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLComputePassSampleBufferAttachmentDescriptor;
extern "C" void *OBJC_CLASS_$_MTLComputePassSampleBufferAttachmentDescriptorArray;
extern "C" void *OBJC_CLASS_$_MTLComputePassDescriptor;

_MTL_INLINE MTL::ComputePassSampleBufferAttachmentDescriptor* MTL::ComputePassSampleBufferAttachmentDescriptor::alloc()
{
    return _MTL_msg_MTL__ComputePassSampleBufferAttachmentDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLComputePassSampleBufferAttachmentDescriptor, nullptr);
}

_MTL_INLINE MTL::ComputePassSampleBufferAttachmentDescriptor* MTL::ComputePassSampleBufferAttachmentDescriptor::init() const
{
    return _MTL_msg_MTL__ComputePassSampleBufferAttachmentDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::CounterSampleBuffer* MTL::ComputePassSampleBufferAttachmentDescriptor::sampleBuffer() const
{
    return _MTL_msg_MTL__CounterSampleBufferp_sampleBuffer((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ComputePassSampleBufferAttachmentDescriptor::setSampleBuffer(MTL::CounterSampleBuffer* sampleBuffer)
{
    _MTL_msg_v_setSampleBuffer__MTL__CounterSampleBufferp((const void*)this, nullptr, sampleBuffer);
}

_MTL_INLINE NS::UInteger MTL::ComputePassSampleBufferAttachmentDescriptor::startOfEncoderSampleIndex() const
{
    return _MTL_msg_NS__UInteger_startOfEncoderSampleIndex((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ComputePassSampleBufferAttachmentDescriptor::setStartOfEncoderSampleIndex(NS::UInteger startOfEncoderSampleIndex)
{
    _MTL_msg_v_setStartOfEncoderSampleIndex__NS__UInteger((const void*)this, nullptr, startOfEncoderSampleIndex);
}

_MTL_INLINE NS::UInteger MTL::ComputePassSampleBufferAttachmentDescriptor::endOfEncoderSampleIndex() const
{
    return _MTL_msg_NS__UInteger_endOfEncoderSampleIndex((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ComputePassSampleBufferAttachmentDescriptor::setEndOfEncoderSampleIndex(NS::UInteger endOfEncoderSampleIndex)
{
    _MTL_msg_v_setEndOfEncoderSampleIndex__NS__UInteger((const void*)this, nullptr, endOfEncoderSampleIndex);
}

_MTL_INLINE MTL::ComputePassSampleBufferAttachmentDescriptorArray* MTL::ComputePassSampleBufferAttachmentDescriptorArray::alloc()
{
    return _MTL_msg_MTL__ComputePassSampleBufferAttachmentDescriptorArrayp_alloc((const void*)&OBJC_CLASS_$_MTLComputePassSampleBufferAttachmentDescriptorArray, nullptr);
}

_MTL_INLINE MTL::ComputePassSampleBufferAttachmentDescriptorArray* MTL::ComputePassSampleBufferAttachmentDescriptorArray::init() const
{
    return _MTL_msg_MTL__ComputePassSampleBufferAttachmentDescriptorArrayp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::ComputePassSampleBufferAttachmentDescriptor* MTL::ComputePassSampleBufferAttachmentDescriptorArray::object(NS::UInteger attachmentIndex)
{
    return _MTL_msg_MTL__ComputePassSampleBufferAttachmentDescriptorp_objectAtIndexedSubscript__NS__UInteger((const void*)this, nullptr, attachmentIndex);
}

_MTL_INLINE void MTL::ComputePassSampleBufferAttachmentDescriptorArray::setObject(MTL::ComputePassSampleBufferAttachmentDescriptor* attachment, NS::UInteger attachmentIndex)
{
    _MTL_msg_v_setObject_atIndexedSubscript__MTL__ComputePassSampleBufferAttachmentDescriptorp_NS__UInteger((const void*)this, nullptr, attachment, attachmentIndex);
}

_MTL_INLINE MTL::ComputePassDescriptor* MTL::ComputePassDescriptor::alloc()
{
    return _MTL_msg_MTL__ComputePassDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLComputePassDescriptor, nullptr);
}

_MTL_INLINE MTL::ComputePassDescriptor* MTL::ComputePassDescriptor::init() const
{
    return _MTL_msg_MTL__ComputePassDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::ComputePassDescriptor* MTL::ComputePassDescriptor::computePassDescriptor()
{
    return _MTL_msg_MTL__ComputePassDescriptorp_computePassDescriptor((const void*)&OBJC_CLASS_$_MTLComputePassDescriptor, nullptr);
}

_MTL_INLINE MTL::DispatchType MTL::ComputePassDescriptor::dispatchType() const
{
    return _MTL_msg_MTL__DispatchType_dispatchType((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ComputePassDescriptor::setDispatchType(MTL::DispatchType dispatchType)
{
    _MTL_msg_v_setDispatchType__MTL__DispatchType((const void*)this, nullptr, dispatchType);
}

_MTL_INLINE MTL::ComputePassSampleBufferAttachmentDescriptorArray* MTL::ComputePassDescriptor::sampleBufferAttachments() const
{
    return _MTL_msg_MTL__ComputePassSampleBufferAttachmentDescriptorArrayp_sampleBufferAttachments((const void*)this, nullptr);
}
