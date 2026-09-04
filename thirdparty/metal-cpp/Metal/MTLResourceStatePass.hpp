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

class ResourceStatePassSampleBufferAttachmentDescriptor;
class ResourceStatePassSampleBufferAttachmentDescriptorArray;
class ResourceStatePassDescriptor;

class ResourceStatePassSampleBufferAttachmentDescriptor : public NS::Copying<ResourceStatePassSampleBufferAttachmentDescriptor>
{
public:
    static ResourceStatePassSampleBufferAttachmentDescriptor* alloc();
    ResourceStatePassSampleBufferAttachmentDescriptor*        init() const;

    NS::UInteger              endOfEncoderSampleIndex() const;
    MTL::CounterSampleBuffer* sampleBuffer() const;
    void                      setEndOfEncoderSampleIndex(NS::UInteger endOfEncoderSampleIndex);
    void                      setSampleBuffer(MTL::CounterSampleBuffer* sampleBuffer);
    void                      setStartOfEncoderSampleIndex(NS::UInteger startOfEncoderSampleIndex);
    NS::UInteger              startOfEncoderSampleIndex() const;

};

class ResourceStatePassSampleBufferAttachmentDescriptorArray : public NS::Referencing<ResourceStatePassSampleBufferAttachmentDescriptorArray>
{
public:
    static ResourceStatePassSampleBufferAttachmentDescriptorArray* alloc();
    ResourceStatePassSampleBufferAttachmentDescriptorArray*        init() const;

    MTL::ResourceStatePassSampleBufferAttachmentDescriptor* object(NS::UInteger attachmentIndex);
    void                                                    setObject(MTL::ResourceStatePassSampleBufferAttachmentDescriptor* attachment, NS::UInteger attachmentIndex);

};

class ResourceStatePassDescriptor : public NS::Copying<ResourceStatePassDescriptor>
{
public:
    static ResourceStatePassDescriptor* alloc();
    ResourceStatePassDescriptor*        init() const;

    static MTL::ResourceStatePassDescriptor* resourceStatePassDescriptor();

    MTL::ResourceStatePassSampleBufferAttachmentDescriptorArray* sampleBufferAttachments() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLResourceStatePassSampleBufferAttachmentDescriptor;
extern "C" void *OBJC_CLASS_$_MTLResourceStatePassSampleBufferAttachmentDescriptorArray;
extern "C" void *OBJC_CLASS_$_MTLResourceStatePassDescriptor;

_MTL_INLINE MTL::ResourceStatePassSampleBufferAttachmentDescriptor* MTL::ResourceStatePassSampleBufferAttachmentDescriptor::alloc()
{
    return _MTL_msg_MTL__ResourceStatePassSampleBufferAttachmentDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLResourceStatePassSampleBufferAttachmentDescriptor, nullptr);
}

_MTL_INLINE MTL::ResourceStatePassSampleBufferAttachmentDescriptor* MTL::ResourceStatePassSampleBufferAttachmentDescriptor::init() const
{
    return _MTL_msg_MTL__ResourceStatePassSampleBufferAttachmentDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::CounterSampleBuffer* MTL::ResourceStatePassSampleBufferAttachmentDescriptor::sampleBuffer() const
{
    return _MTL_msg_MTL__CounterSampleBufferp_sampleBuffer((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ResourceStatePassSampleBufferAttachmentDescriptor::setSampleBuffer(MTL::CounterSampleBuffer* sampleBuffer)
{
    _MTL_msg_v_setSampleBuffer__MTL__CounterSampleBufferp((const void*)this, nullptr, sampleBuffer);
}

_MTL_INLINE NS::UInteger MTL::ResourceStatePassSampleBufferAttachmentDescriptor::startOfEncoderSampleIndex() const
{
    return _MTL_msg_NS__UInteger_startOfEncoderSampleIndex((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ResourceStatePassSampleBufferAttachmentDescriptor::setStartOfEncoderSampleIndex(NS::UInteger startOfEncoderSampleIndex)
{
    _MTL_msg_v_setStartOfEncoderSampleIndex__NS__UInteger((const void*)this, nullptr, startOfEncoderSampleIndex);
}

_MTL_INLINE NS::UInteger MTL::ResourceStatePassSampleBufferAttachmentDescriptor::endOfEncoderSampleIndex() const
{
    return _MTL_msg_NS__UInteger_endOfEncoderSampleIndex((const void*)this, nullptr);
}

_MTL_INLINE void MTL::ResourceStatePassSampleBufferAttachmentDescriptor::setEndOfEncoderSampleIndex(NS::UInteger endOfEncoderSampleIndex)
{
    _MTL_msg_v_setEndOfEncoderSampleIndex__NS__UInteger((const void*)this, nullptr, endOfEncoderSampleIndex);
}

_MTL_INLINE MTL::ResourceStatePassSampleBufferAttachmentDescriptorArray* MTL::ResourceStatePassSampleBufferAttachmentDescriptorArray::alloc()
{
    return _MTL_msg_MTL__ResourceStatePassSampleBufferAttachmentDescriptorArrayp_alloc((const void*)&OBJC_CLASS_$_MTLResourceStatePassSampleBufferAttachmentDescriptorArray, nullptr);
}

_MTL_INLINE MTL::ResourceStatePassSampleBufferAttachmentDescriptorArray* MTL::ResourceStatePassSampleBufferAttachmentDescriptorArray::init() const
{
    return _MTL_msg_MTL__ResourceStatePassSampleBufferAttachmentDescriptorArrayp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::ResourceStatePassSampleBufferAttachmentDescriptor* MTL::ResourceStatePassSampleBufferAttachmentDescriptorArray::object(NS::UInteger attachmentIndex)
{
    return _MTL_msg_MTL__ResourceStatePassSampleBufferAttachmentDescriptorp_objectAtIndexedSubscript__NS__UInteger((const void*)this, nullptr, attachmentIndex);
}

_MTL_INLINE void MTL::ResourceStatePassSampleBufferAttachmentDescriptorArray::setObject(MTL::ResourceStatePassSampleBufferAttachmentDescriptor* attachment, NS::UInteger attachmentIndex)
{
    _MTL_msg_v_setObject_atIndexedSubscript__MTL__ResourceStatePassSampleBufferAttachmentDescriptorp_NS__UInteger((const void*)this, nullptr, attachment, attachmentIndex);
}

_MTL_INLINE MTL::ResourceStatePassDescriptor* MTL::ResourceStatePassDescriptor::alloc()
{
    return _MTL_msg_MTL__ResourceStatePassDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLResourceStatePassDescriptor, nullptr);
}

_MTL_INLINE MTL::ResourceStatePassDescriptor* MTL::ResourceStatePassDescriptor::init() const
{
    return _MTL_msg_MTL__ResourceStatePassDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::ResourceStatePassDescriptor* MTL::ResourceStatePassDescriptor::resourceStatePassDescriptor()
{
    return _MTL_msg_MTL__ResourceStatePassDescriptorp_resourceStatePassDescriptor((const void*)&OBJC_CLASS_$_MTLResourceStatePassDescriptor, nullptr);
}

_MTL_INLINE MTL::ResourceStatePassSampleBufferAttachmentDescriptorArray* MTL::ResourceStatePassDescriptor::sampleBufferAttachments() const
{
    return _MTL_msg_MTL__ResourceStatePassSampleBufferAttachmentDescriptorArrayp_sampleBufferAttachments((const void*)this, nullptr);
}
