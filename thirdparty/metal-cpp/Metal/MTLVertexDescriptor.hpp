//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLVertexDescriptor.hpp
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
class VertexAttributeDescriptor;
class VertexAttributeDescriptorArray;
class VertexBufferLayoutDescriptor;
class VertexBufferLayoutDescriptorArray;
class VertexDescriptor;
_MTL_ENUM(NS::UInteger, VertexFormat) {
    VertexFormatInvalid = 0,
    VertexFormatUChar2 = 1,
    VertexFormatUChar3 = 2,
    VertexFormatUChar4 = 3,
    VertexFormatChar2 = 4,
    VertexFormatChar3 = 5,
    VertexFormatChar4 = 6,
    VertexFormatUChar2Normalized = 7,
    VertexFormatUChar3Normalized = 8,
    VertexFormatUChar4Normalized = 9,
    VertexFormatChar2Normalized = 10,
    VertexFormatChar3Normalized = 11,
    VertexFormatChar4Normalized = 12,
    VertexFormatUShort2 = 13,
    VertexFormatUShort3 = 14,
    VertexFormatUShort4 = 15,
    VertexFormatShort2 = 16,
    VertexFormatShort3 = 17,
    VertexFormatShort4 = 18,
    VertexFormatUShort2Normalized = 19,
    VertexFormatUShort3Normalized = 20,
    VertexFormatUShort4Normalized = 21,
    VertexFormatShort2Normalized = 22,
    VertexFormatShort3Normalized = 23,
    VertexFormatShort4Normalized = 24,
    VertexFormatHalf2 = 25,
    VertexFormatHalf3 = 26,
    VertexFormatHalf4 = 27,
    VertexFormatFloat = 28,
    VertexFormatFloat2 = 29,
    VertexFormatFloat3 = 30,
    VertexFormatFloat4 = 31,
    VertexFormatInt = 32,
    VertexFormatInt2 = 33,
    VertexFormatInt3 = 34,
    VertexFormatInt4 = 35,
    VertexFormatUInt = 36,
    VertexFormatUInt2 = 37,
    VertexFormatUInt3 = 38,
    VertexFormatUInt4 = 39,
    VertexFormatInt1010102Normalized = 40,
    VertexFormatUInt1010102Normalized = 41,
    VertexFormatUChar4Normalized_BGRA = 42,
    VertexFormatUChar = 45,
    VertexFormatChar = 46,
    VertexFormatUCharNormalized = 47,
    VertexFormatCharNormalized = 48,
    VertexFormatUShort = 49,
    VertexFormatShort = 50,
    VertexFormatUShortNormalized = 51,
    VertexFormatShortNormalized = 52,
    VertexFormatHalf = 53,
    VertexFormatFloatRG11B10 = 54,
    VertexFormatFloatRGB9E5 = 55,
};

_MTL_ENUM(NS::UInteger, VertexStepFunction) {
    VertexStepFunctionConstant = 0,
    VertexStepFunctionPerVertex = 1,
    VertexStepFunctionPerInstance = 2,
    VertexStepFunctionPerPatch = 3,
    VertexStepFunctionPerPatchControlPoint = 4,
};

static const NS::UInteger BufferLayoutStrideDynamic = NS::UIntegerMax;

class VertexBufferLayoutDescriptor : public NS::Copying<VertexBufferLayoutDescriptor>
{
public:
    static VertexBufferLayoutDescriptor* alloc();

    VertexBufferLayoutDescriptor*        init();

    void                                 setStepFunction(MTL::VertexStepFunction stepFunction);

    void                                 setStepRate(NS::UInteger stepRate);

    void                                 setStride(NS::UInteger stride);

    VertexStepFunction                   stepFunction() const;

    NS::UInteger                         stepRate() const;

    NS::UInteger                         stride() const;
};
class VertexBufferLayoutDescriptorArray : public NS::Referencing<VertexBufferLayoutDescriptorArray>
{
public:
    static VertexBufferLayoutDescriptorArray* alloc();

    VertexBufferLayoutDescriptorArray*        init();

    VertexBufferLayoutDescriptor*             object(NS::UInteger index);
    void                                      setObject(const MTL::VertexBufferLayoutDescriptor* bufferDesc, NS::UInteger index);
};
class VertexAttributeDescriptor : public NS::Copying<VertexAttributeDescriptor>
{
public:
    static VertexAttributeDescriptor* alloc();

    NS::UInteger                      bufferIndex() const;

    VertexFormat                      format() const;

    VertexAttributeDescriptor*        init();

    NS::UInteger                      offset() const;

    void                              setBufferIndex(NS::UInteger bufferIndex);

    void                              setFormat(MTL::VertexFormat format);

    void                              setOffset(NS::UInteger offset);
};
class VertexAttributeDescriptorArray : public NS::Referencing<VertexAttributeDescriptorArray>
{
public:
    static VertexAttributeDescriptorArray* alloc();

    VertexAttributeDescriptorArray*        init();

    VertexAttributeDescriptor*             object(NS::UInteger index);
    void                                   setObject(const MTL::VertexAttributeDescriptor* attributeDesc, NS::UInteger index);
};
class VertexDescriptor : public NS::Copying<VertexDescriptor>
{
public:
    static VertexDescriptor*           alloc();

    VertexAttributeDescriptorArray*    attributes() const;

    VertexDescriptor*                  init();

    VertexBufferLayoutDescriptorArray* layouts() const;

    void                               reset();

    static VertexDescriptor*           vertexDescriptor();
};

}

_MTL_INLINE MTL::VertexBufferLayoutDescriptor* MTL::VertexBufferLayoutDescriptor::alloc()
{
    return NS::Object::alloc<MTL::VertexBufferLayoutDescriptor>(_MTL_PRIVATE_CLS(MTLVertexBufferLayoutDescriptor));
}

_MTL_INLINE MTL::VertexBufferLayoutDescriptor* MTL::VertexBufferLayoutDescriptor::init()
{
    return NS::Object::init<MTL::VertexBufferLayoutDescriptor>();
}

_MTL_INLINE void MTL::VertexBufferLayoutDescriptor::setStepFunction(MTL::VertexStepFunction stepFunction)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStepFunction_), stepFunction);
}

_MTL_INLINE void MTL::VertexBufferLayoutDescriptor::setStepRate(NS::UInteger stepRate)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStepRate_), stepRate);
}

_MTL_INLINE void MTL::VertexBufferLayoutDescriptor::setStride(NS::UInteger stride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStride_), stride);
}

_MTL_INLINE MTL::VertexStepFunction MTL::VertexBufferLayoutDescriptor::stepFunction() const
{
    return Object::sendMessage<MTL::VertexStepFunction>(this, _MTL_PRIVATE_SEL(stepFunction));
}

_MTL_INLINE NS::UInteger MTL::VertexBufferLayoutDescriptor::stepRate() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(stepRate));
}

_MTL_INLINE NS::UInteger MTL::VertexBufferLayoutDescriptor::stride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(stride));
}

_MTL_INLINE MTL::VertexBufferLayoutDescriptorArray* MTL::VertexBufferLayoutDescriptorArray::alloc()
{
    return NS::Object::alloc<MTL::VertexBufferLayoutDescriptorArray>(_MTL_PRIVATE_CLS(MTLVertexBufferLayoutDescriptorArray));
}

_MTL_INLINE MTL::VertexBufferLayoutDescriptorArray* MTL::VertexBufferLayoutDescriptorArray::init()
{
    return NS::Object::init<MTL::VertexBufferLayoutDescriptorArray>();
}

_MTL_INLINE MTL::VertexBufferLayoutDescriptor* MTL::VertexBufferLayoutDescriptorArray::object(NS::UInteger index)
{
    return Object::sendMessage<MTL::VertexBufferLayoutDescriptor*>(this, _MTL_PRIVATE_SEL(objectAtIndexedSubscript_), index);
}

_MTL_INLINE void MTL::VertexBufferLayoutDescriptorArray::setObject(const MTL::VertexBufferLayoutDescriptor* bufferDesc, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObject_atIndexedSubscript_), bufferDesc, index);
}

_MTL_INLINE MTL::VertexAttributeDescriptor* MTL::VertexAttributeDescriptor::alloc()
{
    return NS::Object::alloc<MTL::VertexAttributeDescriptor>(_MTL_PRIVATE_CLS(MTLVertexAttributeDescriptor));
}

_MTL_INLINE NS::UInteger MTL::VertexAttributeDescriptor::bufferIndex() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(bufferIndex));
}

_MTL_INLINE MTL::VertexFormat MTL::VertexAttributeDescriptor::format() const
{
    return Object::sendMessage<MTL::VertexFormat>(this, _MTL_PRIVATE_SEL(format));
}

_MTL_INLINE MTL::VertexAttributeDescriptor* MTL::VertexAttributeDescriptor::init()
{
    return NS::Object::init<MTL::VertexAttributeDescriptor>();
}

_MTL_INLINE NS::UInteger MTL::VertexAttributeDescriptor::offset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(offset));
}

_MTL_INLINE void MTL::VertexAttributeDescriptor::setBufferIndex(NS::UInteger bufferIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBufferIndex_), bufferIndex);
}

_MTL_INLINE void MTL::VertexAttributeDescriptor::setFormat(MTL::VertexFormat format)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setFormat_), format);
}

_MTL_INLINE void MTL::VertexAttributeDescriptor::setOffset(NS::UInteger offset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setOffset_), offset);
}

_MTL_INLINE MTL::VertexAttributeDescriptorArray* MTL::VertexAttributeDescriptorArray::alloc()
{
    return NS::Object::alloc<MTL::VertexAttributeDescriptorArray>(_MTL_PRIVATE_CLS(MTLVertexAttributeDescriptorArray));
}

_MTL_INLINE MTL::VertexAttributeDescriptorArray* MTL::VertexAttributeDescriptorArray::init()
{
    return NS::Object::init<MTL::VertexAttributeDescriptorArray>();
}

_MTL_INLINE MTL::VertexAttributeDescriptor* MTL::VertexAttributeDescriptorArray::object(NS::UInteger index)
{
    return Object::sendMessage<MTL::VertexAttributeDescriptor*>(this, _MTL_PRIVATE_SEL(objectAtIndexedSubscript_), index);
}

_MTL_INLINE void MTL::VertexAttributeDescriptorArray::setObject(const MTL::VertexAttributeDescriptor* attributeDesc, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObject_atIndexedSubscript_), attributeDesc, index);
}

_MTL_INLINE MTL::VertexDescriptor* MTL::VertexDescriptor::alloc()
{
    return NS::Object::alloc<MTL::VertexDescriptor>(_MTL_PRIVATE_CLS(MTLVertexDescriptor));
}

_MTL_INLINE MTL::VertexAttributeDescriptorArray* MTL::VertexDescriptor::attributes() const
{
    return Object::sendMessage<MTL::VertexAttributeDescriptorArray*>(this, _MTL_PRIVATE_SEL(attributes));
}

_MTL_INLINE MTL::VertexDescriptor* MTL::VertexDescriptor::init()
{
    return NS::Object::init<MTL::VertexDescriptor>();
}

_MTL_INLINE MTL::VertexBufferLayoutDescriptorArray* MTL::VertexDescriptor::layouts() const
{
    return Object::sendMessage<MTL::VertexBufferLayoutDescriptorArray*>(this, _MTL_PRIVATE_SEL(layouts));
}

_MTL_INLINE void MTL::VertexDescriptor::reset()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(reset));
}

_MTL_INLINE MTL::VertexDescriptor* MTL::VertexDescriptor::vertexDescriptor()
{
    return Object::sendMessage<MTL::VertexDescriptor*>(_MTL_PRIVATE_CLS(MTLVertexDescriptor), _MTL_PRIVATE_SEL(vertexDescriptor));
}
