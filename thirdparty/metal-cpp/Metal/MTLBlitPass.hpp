//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLBlitPass.hpp
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
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"

namespace MTL
{
class BlitPassDescriptor;
class BlitPassSampleBufferAttachmentDescriptor;
class BlitPassSampleBufferAttachmentDescriptorArray;
class CounterSampleBuffer;

class BlitPassSampleBufferAttachmentDescriptor : public NS::Copying<BlitPassSampleBufferAttachmentDescriptor>
{
public:
    static BlitPassSampleBufferAttachmentDescriptor* alloc();

    NS::UInteger                                     endOfEncoderSampleIndex() const;

    BlitPassSampleBufferAttachmentDescriptor*        init();

    CounterSampleBuffer*                             sampleBuffer() const;

    void                                             setEndOfEncoderSampleIndex(NS::UInteger endOfEncoderSampleIndex);

    void                                             setSampleBuffer(const MTL::CounterSampleBuffer* sampleBuffer);

    void                                             setStartOfEncoderSampleIndex(NS::UInteger startOfEncoderSampleIndex);
    NS::UInteger                                     startOfEncoderSampleIndex() const;
};
class BlitPassSampleBufferAttachmentDescriptorArray : public NS::Referencing<BlitPassSampleBufferAttachmentDescriptorArray>
{
public:
    static BlitPassSampleBufferAttachmentDescriptorArray* alloc();

    BlitPassSampleBufferAttachmentDescriptorArray*        init();

    BlitPassSampleBufferAttachmentDescriptor*             object(NS::UInteger attachmentIndex);
    void                                                  setObject(const MTL::BlitPassSampleBufferAttachmentDescriptor* attachment, NS::UInteger attachmentIndex);
};
class BlitPassDescriptor : public NS::Copying<BlitPassDescriptor>
{
public:
    static BlitPassDescriptor*                     alloc();

    static BlitPassDescriptor*                     blitPassDescriptor();

    BlitPassDescriptor*                            init();

    BlitPassSampleBufferAttachmentDescriptorArray* sampleBufferAttachments() const;
};

}
_MTL_INLINE MTL::BlitPassSampleBufferAttachmentDescriptor* MTL::BlitPassSampleBufferAttachmentDescriptor::alloc()
{
    return NS::Object::alloc<MTL::BlitPassSampleBufferAttachmentDescriptor>(_MTL_PRIVATE_CLS(MTLBlitPassSampleBufferAttachmentDescriptor));
}

_MTL_INLINE NS::UInteger MTL::BlitPassSampleBufferAttachmentDescriptor::endOfEncoderSampleIndex() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(endOfEncoderSampleIndex));
}

_MTL_INLINE MTL::BlitPassSampleBufferAttachmentDescriptor* MTL::BlitPassSampleBufferAttachmentDescriptor::init()
{
    return NS::Object::init<MTL::BlitPassSampleBufferAttachmentDescriptor>();
}

_MTL_INLINE MTL::CounterSampleBuffer* MTL::BlitPassSampleBufferAttachmentDescriptor::sampleBuffer() const
{
    return Object::sendMessage<MTL::CounterSampleBuffer*>(this, _MTL_PRIVATE_SEL(sampleBuffer));
}

_MTL_INLINE void MTL::BlitPassSampleBufferAttachmentDescriptor::setEndOfEncoderSampleIndex(NS::UInteger endOfEncoderSampleIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setEndOfEncoderSampleIndex_), endOfEncoderSampleIndex);
}

_MTL_INLINE void MTL::BlitPassSampleBufferAttachmentDescriptor::setSampleBuffer(const MTL::CounterSampleBuffer* sampleBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSampleBuffer_), sampleBuffer);
}

_MTL_INLINE void MTL::BlitPassSampleBufferAttachmentDescriptor::setStartOfEncoderSampleIndex(NS::UInteger startOfEncoderSampleIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStartOfEncoderSampleIndex_), startOfEncoderSampleIndex);
}

_MTL_INLINE NS::UInteger MTL::BlitPassSampleBufferAttachmentDescriptor::startOfEncoderSampleIndex() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(startOfEncoderSampleIndex));
}

_MTL_INLINE MTL::BlitPassSampleBufferAttachmentDescriptorArray* MTL::BlitPassSampleBufferAttachmentDescriptorArray::alloc()
{
    return NS::Object::alloc<MTL::BlitPassSampleBufferAttachmentDescriptorArray>(_MTL_PRIVATE_CLS(MTLBlitPassSampleBufferAttachmentDescriptorArray));
}

_MTL_INLINE MTL::BlitPassSampleBufferAttachmentDescriptorArray* MTL::BlitPassSampleBufferAttachmentDescriptorArray::init()
{
    return NS::Object::init<MTL::BlitPassSampleBufferAttachmentDescriptorArray>();
}

_MTL_INLINE MTL::BlitPassSampleBufferAttachmentDescriptor* MTL::BlitPassSampleBufferAttachmentDescriptorArray::object(NS::UInteger attachmentIndex)
{
    return Object::sendMessage<MTL::BlitPassSampleBufferAttachmentDescriptor*>(this, _MTL_PRIVATE_SEL(objectAtIndexedSubscript_), attachmentIndex);
}

_MTL_INLINE void MTL::BlitPassSampleBufferAttachmentDescriptorArray::setObject(const MTL::BlitPassSampleBufferAttachmentDescriptor* attachment, NS::UInteger attachmentIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObject_atIndexedSubscript_), attachment, attachmentIndex);
}

_MTL_INLINE MTL::BlitPassDescriptor* MTL::BlitPassDescriptor::alloc()
{
    return NS::Object::alloc<MTL::BlitPassDescriptor>(_MTL_PRIVATE_CLS(MTLBlitPassDescriptor));
}

_MTL_INLINE MTL::BlitPassDescriptor* MTL::BlitPassDescriptor::blitPassDescriptor()
{
    return Object::sendMessage<MTL::BlitPassDescriptor*>(_MTL_PRIVATE_CLS(MTLBlitPassDescriptor), _MTL_PRIVATE_SEL(blitPassDescriptor));
}

_MTL_INLINE MTL::BlitPassDescriptor* MTL::BlitPassDescriptor::init()
{
    return NS::Object::init<MTL::BlitPassDescriptor>();
}

_MTL_INLINE MTL::BlitPassSampleBufferAttachmentDescriptorArray* MTL::BlitPassDescriptor::sampleBufferAttachments() const
{
    return Object::sendMessage<MTL::BlitPassSampleBufferAttachmentDescriptorArray*>(this, _MTL_PRIVATE_SEL(sampleBufferAttachments));
}
