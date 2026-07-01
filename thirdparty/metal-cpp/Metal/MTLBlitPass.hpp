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
}

namespace MTL
{

class BlitPassSampleBufferAttachmentDescriptor;
class BlitPassSampleBufferAttachmentDescriptorArray;
class BlitPassDescriptor;

class BlitPassSampleBufferAttachmentDescriptor : public NS::Copying<BlitPassSampleBufferAttachmentDescriptor>
{
public:
    static BlitPassSampleBufferAttachmentDescriptor* alloc();
    BlitPassSampleBufferAttachmentDescriptor*        init() const;

    NS::UInteger              endOfEncoderSampleIndex() const;
    MTL::CounterSampleBuffer* sampleBuffer() const;
    void                      setEndOfEncoderSampleIndex(NS::UInteger endOfEncoderSampleIndex);
    void                      setSampleBuffer(MTL::CounterSampleBuffer* sampleBuffer);
    void                      setStartOfEncoderSampleIndex(NS::UInteger startOfEncoderSampleIndex);
    NS::UInteger              startOfEncoderSampleIndex() const;

};

class BlitPassSampleBufferAttachmentDescriptorArray : public NS::Referencing<BlitPassSampleBufferAttachmentDescriptorArray>
{
public:
    static BlitPassSampleBufferAttachmentDescriptorArray* alloc();
    BlitPassSampleBufferAttachmentDescriptorArray*        init() const;

    MTL::BlitPassSampleBufferAttachmentDescriptor* object(NS::UInteger attachmentIndex);
    void                                           setObject(MTL::BlitPassSampleBufferAttachmentDescriptor* attachment, NS::UInteger attachmentIndex);

};

class BlitPassDescriptor : public NS::Copying<BlitPassDescriptor>
{
public:
    static BlitPassDescriptor* alloc();
    BlitPassDescriptor*        init() const;

    static MTL::BlitPassDescriptor* blitPassDescriptor();

    MTL::BlitPassSampleBufferAttachmentDescriptorArray* sampleBufferAttachments() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLBlitPassSampleBufferAttachmentDescriptor;
extern "C" void *OBJC_CLASS_$_MTLBlitPassSampleBufferAttachmentDescriptorArray;
extern "C" void *OBJC_CLASS_$_MTLBlitPassDescriptor;

_MTL_INLINE MTL::BlitPassSampleBufferAttachmentDescriptor* MTL::BlitPassSampleBufferAttachmentDescriptor::alloc()
{
    return _MTL_msg_MTL__BlitPassSampleBufferAttachmentDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLBlitPassSampleBufferAttachmentDescriptor, nullptr);
}

_MTL_INLINE MTL::BlitPassSampleBufferAttachmentDescriptor* MTL::BlitPassSampleBufferAttachmentDescriptor::init() const
{
    return _MTL_msg_MTL__BlitPassSampleBufferAttachmentDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::CounterSampleBuffer* MTL::BlitPassSampleBufferAttachmentDescriptor::sampleBuffer() const
{
    return _MTL_msg_MTL__CounterSampleBufferp_sampleBuffer((const void*)this, nullptr);
}

_MTL_INLINE void MTL::BlitPassSampleBufferAttachmentDescriptor::setSampleBuffer(MTL::CounterSampleBuffer* sampleBuffer)
{
    _MTL_msg_v_setSampleBuffer__MTL__CounterSampleBufferp((const void*)this, nullptr, sampleBuffer);
}

_MTL_INLINE NS::UInteger MTL::BlitPassSampleBufferAttachmentDescriptor::startOfEncoderSampleIndex() const
{
    return _MTL_msg_NS__UInteger_startOfEncoderSampleIndex((const void*)this, nullptr);
}

_MTL_INLINE void MTL::BlitPassSampleBufferAttachmentDescriptor::setStartOfEncoderSampleIndex(NS::UInteger startOfEncoderSampleIndex)
{
    _MTL_msg_v_setStartOfEncoderSampleIndex__NS__UInteger((const void*)this, nullptr, startOfEncoderSampleIndex);
}

_MTL_INLINE NS::UInteger MTL::BlitPassSampleBufferAttachmentDescriptor::endOfEncoderSampleIndex() const
{
    return _MTL_msg_NS__UInteger_endOfEncoderSampleIndex((const void*)this, nullptr);
}

_MTL_INLINE void MTL::BlitPassSampleBufferAttachmentDescriptor::setEndOfEncoderSampleIndex(NS::UInteger endOfEncoderSampleIndex)
{
    _MTL_msg_v_setEndOfEncoderSampleIndex__NS__UInteger((const void*)this, nullptr, endOfEncoderSampleIndex);
}

_MTL_INLINE MTL::BlitPassSampleBufferAttachmentDescriptorArray* MTL::BlitPassSampleBufferAttachmentDescriptorArray::alloc()
{
    return _MTL_msg_MTL__BlitPassSampleBufferAttachmentDescriptorArrayp_alloc((const void*)&OBJC_CLASS_$_MTLBlitPassSampleBufferAttachmentDescriptorArray, nullptr);
}

_MTL_INLINE MTL::BlitPassSampleBufferAttachmentDescriptorArray* MTL::BlitPassSampleBufferAttachmentDescriptorArray::init() const
{
    return _MTL_msg_MTL__BlitPassSampleBufferAttachmentDescriptorArrayp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::BlitPassSampleBufferAttachmentDescriptor* MTL::BlitPassSampleBufferAttachmentDescriptorArray::object(NS::UInteger attachmentIndex)
{
    return _MTL_msg_MTL__BlitPassSampleBufferAttachmentDescriptorp_objectAtIndexedSubscript__NS__UInteger((const void*)this, nullptr, attachmentIndex);
}

_MTL_INLINE void MTL::BlitPassSampleBufferAttachmentDescriptorArray::setObject(MTL::BlitPassSampleBufferAttachmentDescriptor* attachment, NS::UInteger attachmentIndex)
{
    _MTL_msg_v_setObject_atIndexedSubscript__MTL__BlitPassSampleBufferAttachmentDescriptorp_NS__UInteger((const void*)this, nullptr, attachment, attachmentIndex);
}

_MTL_INLINE MTL::BlitPassDescriptor* MTL::BlitPassDescriptor::alloc()
{
    return _MTL_msg_MTL__BlitPassDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLBlitPassDescriptor, nullptr);
}

_MTL_INLINE MTL::BlitPassDescriptor* MTL::BlitPassDescriptor::init() const
{
    return _MTL_msg_MTL__BlitPassDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::BlitPassDescriptor* MTL::BlitPassDescriptor::blitPassDescriptor()
{
    return _MTL_msg_MTL__BlitPassDescriptorp_blitPassDescriptor((const void*)&OBJC_CLASS_$_MTLBlitPassDescriptor, nullptr);
}

_MTL_INLINE MTL::BlitPassSampleBufferAttachmentDescriptorArray* MTL::BlitPassDescriptor::sampleBufferAttachments() const
{
    return _MTL_msg_MTL__BlitPassSampleBufferAttachmentDescriptorArrayp_sampleBufferAttachments((const void*)this, nullptr);
}
