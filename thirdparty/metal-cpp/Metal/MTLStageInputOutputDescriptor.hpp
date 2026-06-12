#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    enum IndexType : NS::UInteger;
}

namespace MTL
{

_MTL_ENUM(NS::UInteger, AttributeFormat) {
    AttributeFormatInvalid = 0,
    AttributeFormatUChar2 = 1,
    AttributeFormatUChar3 = 2,
    AttributeFormatUChar4 = 3,
    AttributeFormatChar2 = 4,
    AttributeFormatChar3 = 5,
    AttributeFormatChar4 = 6,
    AttributeFormatUChar2Normalized = 7,
    AttributeFormatUChar3Normalized = 8,
    AttributeFormatUChar4Normalized = 9,
    AttributeFormatChar2Normalized = 10,
    AttributeFormatChar3Normalized = 11,
    AttributeFormatChar4Normalized = 12,
    AttributeFormatUShort2 = 13,
    AttributeFormatUShort3 = 14,
    AttributeFormatUShort4 = 15,
    AttributeFormatShort2 = 16,
    AttributeFormatShort3 = 17,
    AttributeFormatShort4 = 18,
    AttributeFormatUShort2Normalized = 19,
    AttributeFormatUShort3Normalized = 20,
    AttributeFormatUShort4Normalized = 21,
    AttributeFormatShort2Normalized = 22,
    AttributeFormatShort3Normalized = 23,
    AttributeFormatShort4Normalized = 24,
    AttributeFormatHalf2 = 25,
    AttributeFormatHalf3 = 26,
    AttributeFormatHalf4 = 27,
    AttributeFormatFloat = 28,
    AttributeFormatFloat2 = 29,
    AttributeFormatFloat3 = 30,
    AttributeFormatFloat4 = 31,
    AttributeFormatInt = 32,
    AttributeFormatInt2 = 33,
    AttributeFormatInt3 = 34,
    AttributeFormatInt4 = 35,
    AttributeFormatUInt = 36,
    AttributeFormatUInt2 = 37,
    AttributeFormatUInt3 = 38,
    AttributeFormatUInt4 = 39,
    AttributeFormatInt1010102Normalized = 40,
    AttributeFormatUInt1010102Normalized = 41,
    AttributeFormatUChar4Normalized_BGRA = 42,
    AttributeFormatUChar = 45,
    AttributeFormatChar = 46,
    AttributeFormatUCharNormalized = 47,
    AttributeFormatCharNormalized = 48,
    AttributeFormatUShort = 49,
    AttributeFormatShort = 50,
    AttributeFormatUShortNormalized = 51,
    AttributeFormatShortNormalized = 52,
    AttributeFormatHalf = 53,
    AttributeFormatFloatRG11B10 = 54,
    AttributeFormatFloatRGB9E5 = 55,
};

_MTL_ENUM(NS::UInteger, StepFunction) {
    StepFunctionConstant = 0,
    StepFunctionPerVertex = 1,
    StepFunctionPerInstance = 2,
    StepFunctionPerPatch = 3,
    StepFunctionPerPatchControlPoint = 4,
    StepFunctionThreadPositionInGridX = 5,
    StepFunctionThreadPositionInGridY = 6,
    StepFunctionThreadPositionInGridXIndexed = 7,
    StepFunctionThreadPositionInGridYIndexed = 8,
};


class BufferLayoutDescriptor;
class BufferLayoutDescriptorArray;
class AttributeDescriptor;
class AttributeDescriptorArray;
class StageInputOutputDescriptor;

class BufferLayoutDescriptor : public NS::Copying<BufferLayoutDescriptor>
{
public:
    static BufferLayoutDescriptor* alloc();
    BufferLayoutDescriptor*        init() const;

    void              setStepFunction(MTL::StepFunction stepFunction);
    void              setStepRate(NS::UInteger stepRate);
    void              setStride(NS::UInteger stride);
    MTL::StepFunction stepFunction() const;
    NS::UInteger      stepRate() const;
    NS::UInteger      stride() const;

};

class BufferLayoutDescriptorArray : public NS::Referencing<BufferLayoutDescriptorArray>
{
public:
    static BufferLayoutDescriptorArray* alloc();
    BufferLayoutDescriptorArray*        init() const;

    MTL::BufferLayoutDescriptor* object(NS::UInteger index);
    void                         setObject(MTL::BufferLayoutDescriptor* bufferDesc, NS::UInteger index);

};

class AttributeDescriptor : public NS::Copying<AttributeDescriptor>
{
public:
    static AttributeDescriptor* alloc();
    AttributeDescriptor*        init() const;

    NS::UInteger         bufferIndex() const;
    MTL::AttributeFormat format() const;
    NS::UInteger         offset() const;
    void                 setBufferIndex(NS::UInteger bufferIndex);
    void                 setFormat(MTL::AttributeFormat format);
    void                 setOffset(NS::UInteger offset);

};

class AttributeDescriptorArray : public NS::Referencing<AttributeDescriptorArray>
{
public:
    static AttributeDescriptorArray* alloc();
    AttributeDescriptorArray*        init() const;

    MTL::AttributeDescriptor* object(NS::UInteger index);
    void                      setObject(MTL::AttributeDescriptor* attributeDesc, NS::UInteger index);

};

class StageInputOutputDescriptor : public NS::Copying<StageInputOutputDescriptor>
{
public:
    static StageInputOutputDescriptor* alloc();
    StageInputOutputDescriptor*        init() const;

    static MTL::StageInputOutputDescriptor* stageInputOutputDescriptor();

    MTL::AttributeDescriptorArray*    attributes() const;
    NS::UInteger                      indexBufferIndex() const;
    MTL::IndexType                    indexType() const;
    MTL::BufferLayoutDescriptorArray* layouts() const;
    void                              reset();
    void                              setIndexBufferIndex(NS::UInteger indexBufferIndex);
    void                              setIndexType(MTL::IndexType indexType);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLBufferLayoutDescriptor;
extern "C" void *OBJC_CLASS_$_MTLBufferLayoutDescriptorArray;
extern "C" void *OBJC_CLASS_$_MTLAttributeDescriptor;
extern "C" void *OBJC_CLASS_$_MTLAttributeDescriptorArray;
extern "C" void *OBJC_CLASS_$_MTLStageInputOutputDescriptor;

_MTL_INLINE MTL::BufferLayoutDescriptor* MTL::BufferLayoutDescriptor::alloc()
{
    return _MTL_msg_MTL__BufferLayoutDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLBufferLayoutDescriptor, nullptr);
}

_MTL_INLINE MTL::BufferLayoutDescriptor* MTL::BufferLayoutDescriptor::init() const
{
    return _MTL_msg_MTL__BufferLayoutDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::BufferLayoutDescriptor::stride() const
{
    return _MTL_msg_NS__UInteger_stride((const void*)this, nullptr);
}

_MTL_INLINE void MTL::BufferLayoutDescriptor::setStride(NS::UInteger stride)
{
    _MTL_msg_v_setStride__NS__UInteger((const void*)this, nullptr, stride);
}

_MTL_INLINE MTL::StepFunction MTL::BufferLayoutDescriptor::stepFunction() const
{
    return _MTL_msg_MTL__StepFunction_stepFunction((const void*)this, nullptr);
}

_MTL_INLINE void MTL::BufferLayoutDescriptor::setStepFunction(MTL::StepFunction stepFunction)
{
    _MTL_msg_v_setStepFunction__MTL__StepFunction((const void*)this, nullptr, stepFunction);
}

_MTL_INLINE NS::UInteger MTL::BufferLayoutDescriptor::stepRate() const
{
    return _MTL_msg_NS__UInteger_stepRate((const void*)this, nullptr);
}

_MTL_INLINE void MTL::BufferLayoutDescriptor::setStepRate(NS::UInteger stepRate)
{
    _MTL_msg_v_setStepRate__NS__UInteger((const void*)this, nullptr, stepRate);
}

_MTL_INLINE MTL::BufferLayoutDescriptorArray* MTL::BufferLayoutDescriptorArray::alloc()
{
    return _MTL_msg_MTL__BufferLayoutDescriptorArrayp_alloc((const void*)&OBJC_CLASS_$_MTLBufferLayoutDescriptorArray, nullptr);
}

_MTL_INLINE MTL::BufferLayoutDescriptorArray* MTL::BufferLayoutDescriptorArray::init() const
{
    return _MTL_msg_MTL__BufferLayoutDescriptorArrayp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::BufferLayoutDescriptor* MTL::BufferLayoutDescriptorArray::object(NS::UInteger index)
{
    return _MTL_msg_MTL__BufferLayoutDescriptorp_objectAtIndexedSubscript__NS__UInteger((const void*)this, nullptr, index);
}

_MTL_INLINE void MTL::BufferLayoutDescriptorArray::setObject(MTL::BufferLayoutDescriptor* bufferDesc, NS::UInteger index)
{
    _MTL_msg_v_setObject_atIndexedSubscript__MTL__BufferLayoutDescriptorp_NS__UInteger((const void*)this, nullptr, bufferDesc, index);
}

_MTL_INLINE MTL::AttributeDescriptor* MTL::AttributeDescriptor::alloc()
{
    return _MTL_msg_MTL__AttributeDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLAttributeDescriptor, nullptr);
}

_MTL_INLINE MTL::AttributeDescriptor* MTL::AttributeDescriptor::init() const
{
    return _MTL_msg_MTL__AttributeDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::AttributeFormat MTL::AttributeDescriptor::format() const
{
    return _MTL_msg_MTL__AttributeFormat_format((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AttributeDescriptor::setFormat(MTL::AttributeFormat format)
{
    _MTL_msg_v_setFormat__MTL__AttributeFormat((const void*)this, nullptr, format);
}

_MTL_INLINE NS::UInteger MTL::AttributeDescriptor::offset() const
{
    return _MTL_msg_NS__UInteger_offset((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AttributeDescriptor::setOffset(NS::UInteger offset)
{
    _MTL_msg_v_setOffset__NS__UInteger((const void*)this, nullptr, offset);
}

_MTL_INLINE NS::UInteger MTL::AttributeDescriptor::bufferIndex() const
{
    return _MTL_msg_NS__UInteger_bufferIndex((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AttributeDescriptor::setBufferIndex(NS::UInteger bufferIndex)
{
    _MTL_msg_v_setBufferIndex__NS__UInteger((const void*)this, nullptr, bufferIndex);
}

_MTL_INLINE MTL::AttributeDescriptorArray* MTL::AttributeDescriptorArray::alloc()
{
    return _MTL_msg_MTL__AttributeDescriptorArrayp_alloc((const void*)&OBJC_CLASS_$_MTLAttributeDescriptorArray, nullptr);
}

_MTL_INLINE MTL::AttributeDescriptorArray* MTL::AttributeDescriptorArray::init() const
{
    return _MTL_msg_MTL__AttributeDescriptorArrayp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::AttributeDescriptor* MTL::AttributeDescriptorArray::object(NS::UInteger index)
{
    return _MTL_msg_MTL__AttributeDescriptorp_objectAtIndexedSubscript__NS__UInteger((const void*)this, nullptr, index);
}

_MTL_INLINE void MTL::AttributeDescriptorArray::setObject(MTL::AttributeDescriptor* attributeDesc, NS::UInteger index)
{
    _MTL_msg_v_setObject_atIndexedSubscript__MTL__AttributeDescriptorp_NS__UInteger((const void*)this, nullptr, attributeDesc, index);
}

_MTL_INLINE MTL::StageInputOutputDescriptor* MTL::StageInputOutputDescriptor::alloc()
{
    return _MTL_msg_MTL__StageInputOutputDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLStageInputOutputDescriptor, nullptr);
}

_MTL_INLINE MTL::StageInputOutputDescriptor* MTL::StageInputOutputDescriptor::init() const
{
    return _MTL_msg_MTL__StageInputOutputDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::StageInputOutputDescriptor* MTL::StageInputOutputDescriptor::stageInputOutputDescriptor()
{
    return _MTL_msg_MTL__StageInputOutputDescriptorp_stageInputOutputDescriptor((const void*)&OBJC_CLASS_$_MTLStageInputOutputDescriptor, nullptr);
}

_MTL_INLINE MTL::BufferLayoutDescriptorArray* MTL::StageInputOutputDescriptor::layouts() const
{
    return _MTL_msg_MTL__BufferLayoutDescriptorArrayp_layouts((const void*)this, nullptr);
}

_MTL_INLINE MTL::AttributeDescriptorArray* MTL::StageInputOutputDescriptor::attributes() const
{
    return _MTL_msg_MTL__AttributeDescriptorArrayp_attributes((const void*)this, nullptr);
}

_MTL_INLINE MTL::IndexType MTL::StageInputOutputDescriptor::indexType() const
{
    return _MTL_msg_MTL__IndexType_indexType((const void*)this, nullptr);
}

_MTL_INLINE void MTL::StageInputOutputDescriptor::setIndexType(MTL::IndexType indexType)
{
    _MTL_msg_v_setIndexType__MTL__IndexType((const void*)this, nullptr, indexType);
}

_MTL_INLINE NS::UInteger MTL::StageInputOutputDescriptor::indexBufferIndex() const
{
    return _MTL_msg_NS__UInteger_indexBufferIndex((const void*)this, nullptr);
}

_MTL_INLINE void MTL::StageInputOutputDescriptor::setIndexBufferIndex(NS::UInteger indexBufferIndex)
{
    _MTL_msg_v_setIndexBufferIndex__NS__UInteger((const void*)this, nullptr, indexBufferIndex);
}

_MTL_INLINE void MTL::StageInputOutputDescriptor::reset()
{
    _MTL_msg_v_reset((const void*)this, nullptr);
}
