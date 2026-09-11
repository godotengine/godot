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


class VertexBufferLayoutDescriptor;
class VertexBufferLayoutDescriptorArray;
class VertexAttributeDescriptor;
class VertexAttributeDescriptorArray;
class VertexDescriptor;

class VertexBufferLayoutDescriptor : public NS::Copying<VertexBufferLayoutDescriptor>
{
public:
    static VertexBufferLayoutDescriptor* alloc();
    VertexBufferLayoutDescriptor*        init() const;

    void                    setStepFunction(MTL::VertexStepFunction stepFunction);
    void                    setStepRate(NS::UInteger stepRate);
    void                    setStride(NS::UInteger stride);
    MTL::VertexStepFunction stepFunction() const;
    NS::UInteger            stepRate() const;
    NS::UInteger            stride() const;

};

class VertexBufferLayoutDescriptorArray : public NS::Referencing<VertexBufferLayoutDescriptorArray>
{
public:
    static VertexBufferLayoutDescriptorArray* alloc();
    VertexBufferLayoutDescriptorArray*        init() const;

    MTL::VertexBufferLayoutDescriptor* object(NS::UInteger index);
    void                               setObject(MTL::VertexBufferLayoutDescriptor* bufferDesc, NS::UInteger index);

};

class VertexAttributeDescriptor : public NS::Copying<VertexAttributeDescriptor>
{
public:
    static VertexAttributeDescriptor* alloc();
    VertexAttributeDescriptor*        init() const;

    NS::UInteger      bufferIndex() const;
    MTL::VertexFormat format() const;
    NS::UInteger      offset() const;
    void              setBufferIndex(NS::UInteger bufferIndex);
    void              setFormat(MTL::VertexFormat format);
    void              setOffset(NS::UInteger offset);

};

class VertexAttributeDescriptorArray : public NS::Referencing<VertexAttributeDescriptorArray>
{
public:
    static VertexAttributeDescriptorArray* alloc();
    VertexAttributeDescriptorArray*        init() const;

    MTL::VertexAttributeDescriptor* object(NS::UInteger index);
    void                            setObject(MTL::VertexAttributeDescriptor* attributeDesc, NS::UInteger index);

};

class VertexDescriptor : public NS::Copying<VertexDescriptor>
{
public:
    static VertexDescriptor* alloc();
    VertexDescriptor*        init() const;

    static MTL::VertexDescriptor* vertexDescriptor();

    MTL::VertexAttributeDescriptorArray*    attributes() const;
    MTL::VertexBufferLayoutDescriptorArray* layouts() const;
    void                                    reset();

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLVertexBufferLayoutDescriptor;
extern "C" void *OBJC_CLASS_$_MTLVertexBufferLayoutDescriptorArray;
extern "C" void *OBJC_CLASS_$_MTLVertexAttributeDescriptor;
extern "C" void *OBJC_CLASS_$_MTLVertexAttributeDescriptorArray;
extern "C" void *OBJC_CLASS_$_MTLVertexDescriptor;

_MTL_INLINE MTL::VertexBufferLayoutDescriptor* MTL::VertexBufferLayoutDescriptor::alloc()
{
    return _MTL_msg_MTL__VertexBufferLayoutDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLVertexBufferLayoutDescriptor, nullptr);
}

_MTL_INLINE MTL::VertexBufferLayoutDescriptor* MTL::VertexBufferLayoutDescriptor::init() const
{
    return _MTL_msg_MTL__VertexBufferLayoutDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::VertexBufferLayoutDescriptor::stride() const
{
    return _MTL_msg_NS__UInteger_stride((const void*)this, nullptr);
}

_MTL_INLINE void MTL::VertexBufferLayoutDescriptor::setStride(NS::UInteger stride)
{
    _MTL_msg_v_setStride__NS__UInteger((const void*)this, nullptr, stride);
}

_MTL_INLINE MTL::VertexStepFunction MTL::VertexBufferLayoutDescriptor::stepFunction() const
{
    return _MTL_msg_MTL__VertexStepFunction_stepFunction((const void*)this, nullptr);
}

_MTL_INLINE void MTL::VertexBufferLayoutDescriptor::setStepFunction(MTL::VertexStepFunction stepFunction)
{
    _MTL_msg_v_setStepFunction__MTL__VertexStepFunction((const void*)this, nullptr, stepFunction);
}

_MTL_INLINE NS::UInteger MTL::VertexBufferLayoutDescriptor::stepRate() const
{
    return _MTL_msg_NS__UInteger_stepRate((const void*)this, nullptr);
}

_MTL_INLINE void MTL::VertexBufferLayoutDescriptor::setStepRate(NS::UInteger stepRate)
{
    _MTL_msg_v_setStepRate__NS__UInteger((const void*)this, nullptr, stepRate);
}

_MTL_INLINE MTL::VertexBufferLayoutDescriptorArray* MTL::VertexBufferLayoutDescriptorArray::alloc()
{
    return _MTL_msg_MTL__VertexBufferLayoutDescriptorArrayp_alloc((const void*)&OBJC_CLASS_$_MTLVertexBufferLayoutDescriptorArray, nullptr);
}

_MTL_INLINE MTL::VertexBufferLayoutDescriptorArray* MTL::VertexBufferLayoutDescriptorArray::init() const
{
    return _MTL_msg_MTL__VertexBufferLayoutDescriptorArrayp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::VertexBufferLayoutDescriptor* MTL::VertexBufferLayoutDescriptorArray::object(NS::UInteger index)
{
    return _MTL_msg_MTL__VertexBufferLayoutDescriptorp_objectAtIndexedSubscript__NS__UInteger((const void*)this, nullptr, index);
}

_MTL_INLINE void MTL::VertexBufferLayoutDescriptorArray::setObject(MTL::VertexBufferLayoutDescriptor* bufferDesc, NS::UInteger index)
{
    _MTL_msg_v_setObject_atIndexedSubscript__MTL__VertexBufferLayoutDescriptorp_NS__UInteger((const void*)this, nullptr, bufferDesc, index);
}

_MTL_INLINE MTL::VertexAttributeDescriptor* MTL::VertexAttributeDescriptor::alloc()
{
    return _MTL_msg_MTL__VertexAttributeDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLVertexAttributeDescriptor, nullptr);
}

_MTL_INLINE MTL::VertexAttributeDescriptor* MTL::VertexAttributeDescriptor::init() const
{
    return _MTL_msg_MTL__VertexAttributeDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::VertexFormat MTL::VertexAttributeDescriptor::format() const
{
    return _MTL_msg_MTL__VertexFormat_format((const void*)this, nullptr);
}

_MTL_INLINE void MTL::VertexAttributeDescriptor::setFormat(MTL::VertexFormat format)
{
    _MTL_msg_v_setFormat__MTL__VertexFormat((const void*)this, nullptr, format);
}

_MTL_INLINE NS::UInteger MTL::VertexAttributeDescriptor::offset() const
{
    return _MTL_msg_NS__UInteger_offset((const void*)this, nullptr);
}

_MTL_INLINE void MTL::VertexAttributeDescriptor::setOffset(NS::UInteger offset)
{
    _MTL_msg_v_setOffset__NS__UInteger((const void*)this, nullptr, offset);
}

_MTL_INLINE NS::UInteger MTL::VertexAttributeDescriptor::bufferIndex() const
{
    return _MTL_msg_NS__UInteger_bufferIndex((const void*)this, nullptr);
}

_MTL_INLINE void MTL::VertexAttributeDescriptor::setBufferIndex(NS::UInteger bufferIndex)
{
    _MTL_msg_v_setBufferIndex__NS__UInteger((const void*)this, nullptr, bufferIndex);
}

_MTL_INLINE MTL::VertexAttributeDescriptorArray* MTL::VertexAttributeDescriptorArray::alloc()
{
    return _MTL_msg_MTL__VertexAttributeDescriptorArrayp_alloc((const void*)&OBJC_CLASS_$_MTLVertexAttributeDescriptorArray, nullptr);
}

_MTL_INLINE MTL::VertexAttributeDescriptorArray* MTL::VertexAttributeDescriptorArray::init() const
{
    return _MTL_msg_MTL__VertexAttributeDescriptorArrayp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::VertexAttributeDescriptor* MTL::VertexAttributeDescriptorArray::object(NS::UInteger index)
{
    return _MTL_msg_MTL__VertexAttributeDescriptorp_objectAtIndexedSubscript__NS__UInteger((const void*)this, nullptr, index);
}

_MTL_INLINE void MTL::VertexAttributeDescriptorArray::setObject(MTL::VertexAttributeDescriptor* attributeDesc, NS::UInteger index)
{
    _MTL_msg_v_setObject_atIndexedSubscript__MTL__VertexAttributeDescriptorp_NS__UInteger((const void*)this, nullptr, attributeDesc, index);
}

_MTL_INLINE MTL::VertexDescriptor* MTL::VertexDescriptor::alloc()
{
    return _MTL_msg_MTL__VertexDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLVertexDescriptor, nullptr);
}

_MTL_INLINE MTL::VertexDescriptor* MTL::VertexDescriptor::init() const
{
    return _MTL_msg_MTL__VertexDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::VertexDescriptor* MTL::VertexDescriptor::vertexDescriptor()
{
    return _MTL_msg_MTL__VertexDescriptorp_vertexDescriptor((const void*)&OBJC_CLASS_$_MTLVertexDescriptor, nullptr);
}

_MTL_INLINE MTL::VertexBufferLayoutDescriptorArray* MTL::VertexDescriptor::layouts() const
{
    return _MTL_msg_MTL__VertexBufferLayoutDescriptorArrayp_layouts((const void*)this, nullptr);
}

_MTL_INLINE MTL::VertexAttributeDescriptorArray* MTL::VertexDescriptor::attributes() const
{
    return _MTL_msg_MTL__VertexAttributeDescriptorArrayp_attributes((const void*)this, nullptr);
}

_MTL_INLINE void MTL::VertexDescriptor::reset()
{
    _MTL_msg_v_reset((const void*)this, nullptr);
}
