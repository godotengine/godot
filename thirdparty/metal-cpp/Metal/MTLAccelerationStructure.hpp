#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTLResource.hpp"

namespace MTL {
    class Buffer;
    enum AttributeFormat : NS::UInteger;
    enum IndexType : NS::UInteger;
}
namespace NS {
    class Array;
    class String;
}

namespace MTL
{

_MTL_OPTIONS(NS::UInteger, AccelerationStructureRefitOptions) {
    AccelerationStructureRefitOptionVertexData = (1 << 0),
    AccelerationStructureRefitOptionPerPrimitiveData = (1 << 1),
};

_MTL_OPTIONS(NS::UInteger, AccelerationStructureUsage) {
    AccelerationStructureUsageNone = 0,
    AccelerationStructureUsageRefit = (1 << 0),
    AccelerationStructureUsagePreferFastBuild = (1 << 1),
    AccelerationStructureUsageExtendedLimits = (1 << 2),
    AccelerationStructureUsagePreferFastIntersection = (1 << 4),
    AccelerationStructureUsageMinimizeMemory = (1 << 5),
};

_MTL_OPTIONS(uint32_t, AccelerationStructureInstanceOptions) {
    AccelerationStructureInstanceOptionNone = 0,
    AccelerationStructureInstanceOptionDisableTriangleCulling = (1 << 0),
    AccelerationStructureInstanceOptionTriangleFrontFacingWindingCounterClockwise = (1 << 1),
    AccelerationStructureInstanceOptionOpaque = (1 << 2),
    AccelerationStructureInstanceOptionNonOpaque = (1 << 3),
};

_MTL_ENUM(NS::Integer, MatrixLayout) {
    MatrixLayoutColumnMajor = 0,
    MatrixLayoutRowMajor = 1,
};

_MTL_ENUM(uint32_t, MotionBorderMode) {
    MotionBorderModeClamp = 0,
    MotionBorderModeVanish = 1,
};

_MTL_ENUM(NS::Integer, CurveType) {
    CurveTypeRound = 0,
    CurveTypeFlat = 1,
};

_MTL_ENUM(NS::Integer, CurveBasis) {
    CurveBasisBSpline = 0,
    CurveBasisCatmullRom = 1,
    CurveBasisLinear = 2,
    CurveBasisBezier = 3,
};

_MTL_ENUM(NS::Integer, CurveEndCaps) {
    CurveEndCapsNone = 0,
    CurveEndCapsDisk = 1,
    CurveEndCapsSphere = 2,
};

_MTL_ENUM(NS::UInteger, AccelerationStructureInstanceDescriptorType) {
    AccelerationStructureInstanceDescriptorTypeDefault = 0,
    AccelerationStructureInstanceDescriptorTypeUserID = 1,
    AccelerationStructureInstanceDescriptorTypeMotion = 2,
    AccelerationStructureInstanceDescriptorTypeIndirect = 3,
    AccelerationStructureInstanceDescriptorTypeIndirectMotion = 4,
};

_MTL_ENUM(NS::Integer, TransformType) {
    TransformTypePackedFloat4x3 = 0,
    TransformTypeComponent = 1,
};


class AccelerationStructureDescriptor;
class AccelerationStructureGeometryDescriptor;
class PrimitiveAccelerationStructureDescriptor;
class AccelerationStructureTriangleGeometryDescriptor;
class AccelerationStructureBoundingBoxGeometryDescriptor;
class MotionKeyframeData;
class AccelerationStructureMotionTriangleGeometryDescriptor;
class AccelerationStructureMotionBoundingBoxGeometryDescriptor;
class AccelerationStructureCurveGeometryDescriptor;
class AccelerationStructureMotionCurveGeometryDescriptor;
class InstanceAccelerationStructureDescriptor;
class IndirectInstanceAccelerationStructureDescriptor;
class AccelerationStructure;

class AccelerationStructureDescriptor : public NS::Copying<AccelerationStructureDescriptor>
{
public:
    static AccelerationStructureDescriptor* alloc();
    AccelerationStructureDescriptor*        init() const;

    void                            setUsage(MTL::AccelerationStructureUsage usage);
    MTL::AccelerationStructureUsage usage() const;

};

class AccelerationStructureGeometryDescriptor : public NS::Copying<AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureGeometryDescriptor* alloc();
    AccelerationStructureGeometryDescriptor*        init() const;

    bool         allowDuplicateIntersectionFunctionInvocation() const;
    NS::UInteger intersectionFunctionTableOffset() const;
    NS::String*  label() const;
    bool         opaque() const;
    MTL::Buffer* primitiveDataBuffer() const;
    NS::UInteger primitiveDataBufferOffset() const;
    NS::UInteger primitiveDataElementSize() const;
    NS::UInteger primitiveDataStride() const;
    void         setAllowDuplicateIntersectionFunctionInvocation(bool allowDuplicateIntersectionFunctionInvocation);
    void         setIntersectionFunctionTableOffset(NS::UInteger intersectionFunctionTableOffset);
    void         setLabel(NS::String* label);
    void         setOpaque(bool opaque);
    void         setPrimitiveDataBuffer(MTL::Buffer* primitiveDataBuffer);
    void         setPrimitiveDataBufferOffset(NS::UInteger primitiveDataBufferOffset);
    void         setPrimitiveDataElementSize(NS::UInteger primitiveDataElementSize);
    void         setPrimitiveDataStride(NS::UInteger primitiveDataStride);

};

class PrimitiveAccelerationStructureDescriptor : public NS::Referencing<PrimitiveAccelerationStructureDescriptor, MTL::AccelerationStructureDescriptor>
{
public:
    static PrimitiveAccelerationStructureDescriptor* alloc();
    PrimitiveAccelerationStructureDescriptor*        init() const;

    static MTL::PrimitiveAccelerationStructureDescriptor* descriptor();

    NS::Array*            geometryDescriptors() const;
    MTL::MotionBorderMode motionEndBorderMode() const;
    float                 motionEndTime() const;
    NS::UInteger          motionKeyframeCount() const;
    MTL::MotionBorderMode motionStartBorderMode() const;
    float                 motionStartTime() const;
    void                  setGeometryDescriptors(NS::Array* geometryDescriptors);
    void                  setMotionEndBorderMode(MTL::MotionBorderMode motionEndBorderMode);
    void                  setMotionEndTime(float motionEndTime);
    void                  setMotionKeyframeCount(NS::UInteger motionKeyframeCount);
    void                  setMotionStartBorderMode(MTL::MotionBorderMode motionStartBorderMode);
    void                  setMotionStartTime(float motionStartTime);

};

class AccelerationStructureTriangleGeometryDescriptor : public NS::Referencing<AccelerationStructureTriangleGeometryDescriptor, MTL::AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureTriangleGeometryDescriptor* alloc();
    AccelerationStructureTriangleGeometryDescriptor*        init() const;

    static MTL::AccelerationStructureTriangleGeometryDescriptor* descriptor();

    MTL::Buffer*         indexBuffer() const;
    NS::UInteger         indexBufferOffset() const;
    MTL::IndexType       indexType() const;
    void                 setIndexBuffer(MTL::Buffer* indexBuffer);
    void                 setIndexBufferOffset(NS::UInteger indexBufferOffset);
    void                 setIndexType(MTL::IndexType indexType);
    void                 setTransformationMatrixBuffer(MTL::Buffer* transformationMatrixBuffer);
    void                 setTransformationMatrixBufferOffset(NS::UInteger transformationMatrixBufferOffset);
    void                 setTransformationMatrixLayout(MTL::MatrixLayout transformationMatrixLayout);
    void                 setTriangleCount(NS::UInteger triangleCount);
    void                 setVertexBuffer(MTL::Buffer* vertexBuffer);
    void                 setVertexBufferOffset(NS::UInteger vertexBufferOffset);
    void                 setVertexFormat(MTL::AttributeFormat vertexFormat);
    void                 setVertexStride(NS::UInteger vertexStride);
    MTL::Buffer*         transformationMatrixBuffer() const;
    NS::UInteger         transformationMatrixBufferOffset() const;
    MTL::MatrixLayout    transformationMatrixLayout() const;
    NS::UInteger         triangleCount() const;
    MTL::Buffer*         vertexBuffer() const;
    NS::UInteger         vertexBufferOffset() const;
    MTL::AttributeFormat vertexFormat() const;
    NS::UInteger         vertexStride() const;

};

class AccelerationStructureBoundingBoxGeometryDescriptor : public NS::Referencing<AccelerationStructureBoundingBoxGeometryDescriptor, MTL::AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureBoundingBoxGeometryDescriptor* alloc();
    AccelerationStructureBoundingBoxGeometryDescriptor*        init() const;

    static MTL::AccelerationStructureBoundingBoxGeometryDescriptor* descriptor();

    MTL::Buffer* boundingBoxBuffer() const;
    NS::UInteger boundingBoxBufferOffset() const;
    NS::UInteger boundingBoxCount() const;
    NS::UInteger boundingBoxStride() const;
    void         setBoundingBoxBuffer(MTL::Buffer* boundingBoxBuffer);
    void         setBoundingBoxBufferOffset(NS::UInteger boundingBoxBufferOffset);
    void         setBoundingBoxCount(NS::UInteger boundingBoxCount);
    void         setBoundingBoxStride(NS::UInteger boundingBoxStride);

};

class MotionKeyframeData : public NS::Referencing<MotionKeyframeData>
{
public:
    static MotionKeyframeData* alloc();
    MotionKeyframeData*        init() const;

    static MTL::MotionKeyframeData* data();

    MTL::Buffer* buffer() const;
    NS::UInteger offset() const;
    void         setBuffer(MTL::Buffer* buffer);
    void         setOffset(NS::UInteger offset);

};

class AccelerationStructureMotionTriangleGeometryDescriptor : public NS::Referencing<AccelerationStructureMotionTriangleGeometryDescriptor, MTL::AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureMotionTriangleGeometryDescriptor* alloc();
    AccelerationStructureMotionTriangleGeometryDescriptor*        init() const;

    static MTL::AccelerationStructureMotionTriangleGeometryDescriptor* descriptor();

    MTL::Buffer*         indexBuffer() const;
    NS::UInteger         indexBufferOffset() const;
    MTL::IndexType       indexType() const;
    void                 setIndexBuffer(MTL::Buffer* indexBuffer);
    void                 setIndexBufferOffset(NS::UInteger indexBufferOffset);
    void                 setIndexType(MTL::IndexType indexType);
    void                 setTransformationMatrixBuffer(MTL::Buffer* transformationMatrixBuffer);
    void                 setTransformationMatrixBufferOffset(NS::UInteger transformationMatrixBufferOffset);
    void                 setTransformationMatrixLayout(MTL::MatrixLayout transformationMatrixLayout);
    void                 setTriangleCount(NS::UInteger triangleCount);
    void                 setVertexBuffers(NS::Array* vertexBuffers);
    void                 setVertexFormat(MTL::AttributeFormat vertexFormat);
    void                 setVertexStride(NS::UInteger vertexStride);
    MTL::Buffer*         transformationMatrixBuffer() const;
    NS::UInteger         transformationMatrixBufferOffset() const;
    MTL::MatrixLayout    transformationMatrixLayout() const;
    NS::UInteger         triangleCount() const;
    NS::Array*           vertexBuffers() const;
    MTL::AttributeFormat vertexFormat() const;
    NS::UInteger         vertexStride() const;

};

class AccelerationStructureMotionBoundingBoxGeometryDescriptor : public NS::Referencing<AccelerationStructureMotionBoundingBoxGeometryDescriptor, MTL::AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureMotionBoundingBoxGeometryDescriptor* alloc();
    AccelerationStructureMotionBoundingBoxGeometryDescriptor*        init() const;

    static MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor* descriptor();

    NS::Array*   boundingBoxBuffers() const;
    NS::UInteger boundingBoxCount() const;
    NS::UInteger boundingBoxStride() const;
    void         setBoundingBoxBuffers(NS::Array* boundingBoxBuffers);
    void         setBoundingBoxCount(NS::UInteger boundingBoxCount);
    void         setBoundingBoxStride(NS::UInteger boundingBoxStride);

};

class AccelerationStructureCurveGeometryDescriptor : public NS::Referencing<AccelerationStructureCurveGeometryDescriptor, MTL::AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureCurveGeometryDescriptor* alloc();
    AccelerationStructureCurveGeometryDescriptor*        init() const;

    static MTL::AccelerationStructureCurveGeometryDescriptor* descriptor();

    MTL::Buffer*         controlPointBuffer() const;
    NS::UInteger         controlPointBufferOffset() const;
    NS::UInteger         controlPointCount() const;
    MTL::AttributeFormat controlPointFormat() const;
    NS::UInteger         controlPointStride() const;
    MTL::CurveBasis      curveBasis() const;
    MTL::CurveEndCaps    curveEndCaps() const;
    MTL::CurveType       curveType() const;
    MTL::Buffer*         indexBuffer() const;
    NS::UInteger         indexBufferOffset() const;
    MTL::IndexType       indexType() const;
    MTL::Buffer*         radiusBuffer() const;
    NS::UInteger         radiusBufferOffset() const;
    MTL::AttributeFormat radiusFormat() const;
    NS::UInteger         radiusStride() const;
    NS::UInteger         segmentControlPointCount() const;
    NS::UInteger         segmentCount() const;
    void                 setControlPointBuffer(MTL::Buffer* controlPointBuffer);
    void                 setControlPointBufferOffset(NS::UInteger controlPointBufferOffset);
    void                 setControlPointCount(NS::UInteger controlPointCount);
    void                 setControlPointFormat(MTL::AttributeFormat controlPointFormat);
    void                 setControlPointStride(NS::UInteger controlPointStride);
    void                 setCurveBasis(MTL::CurveBasis curveBasis);
    void                 setCurveEndCaps(MTL::CurveEndCaps curveEndCaps);
    void                 setCurveType(MTL::CurveType curveType);
    void                 setIndexBuffer(MTL::Buffer* indexBuffer);
    void                 setIndexBufferOffset(NS::UInteger indexBufferOffset);
    void                 setIndexType(MTL::IndexType indexType);
    void                 setRadiusBuffer(MTL::Buffer* radiusBuffer);
    void                 setRadiusBufferOffset(NS::UInteger radiusBufferOffset);
    void                 setRadiusFormat(MTL::AttributeFormat radiusFormat);
    void                 setRadiusStride(NS::UInteger radiusStride);
    void                 setSegmentControlPointCount(NS::UInteger segmentControlPointCount);
    void                 setSegmentCount(NS::UInteger segmentCount);

};

class AccelerationStructureMotionCurveGeometryDescriptor : public NS::Referencing<AccelerationStructureMotionCurveGeometryDescriptor, MTL::AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureMotionCurveGeometryDescriptor* alloc();
    AccelerationStructureMotionCurveGeometryDescriptor*        init() const;

    static MTL::AccelerationStructureMotionCurveGeometryDescriptor* descriptor();

    NS::Array*           controlPointBuffers() const;
    NS::UInteger         controlPointCount() const;
    MTL::AttributeFormat controlPointFormat() const;
    NS::UInteger         controlPointStride() const;
    MTL::CurveBasis      curveBasis() const;
    MTL::CurveEndCaps    curveEndCaps() const;
    MTL::CurveType       curveType() const;
    MTL::Buffer*         indexBuffer() const;
    NS::UInteger         indexBufferOffset() const;
    MTL::IndexType       indexType() const;
    NS::Array*           radiusBuffers() const;
    MTL::AttributeFormat radiusFormat() const;
    NS::UInteger         radiusStride() const;
    NS::UInteger         segmentControlPointCount() const;
    NS::UInteger         segmentCount() const;
    void                 setControlPointBuffers(NS::Array* controlPointBuffers);
    void                 setControlPointCount(NS::UInteger controlPointCount);
    void                 setControlPointFormat(MTL::AttributeFormat controlPointFormat);
    void                 setControlPointStride(NS::UInteger controlPointStride);
    void                 setCurveBasis(MTL::CurveBasis curveBasis);
    void                 setCurveEndCaps(MTL::CurveEndCaps curveEndCaps);
    void                 setCurveType(MTL::CurveType curveType);
    void                 setIndexBuffer(MTL::Buffer* indexBuffer);
    void                 setIndexBufferOffset(NS::UInteger indexBufferOffset);
    void                 setIndexType(MTL::IndexType indexType);
    void                 setRadiusBuffers(NS::Array* radiusBuffers);
    void                 setRadiusFormat(MTL::AttributeFormat radiusFormat);
    void                 setRadiusStride(NS::UInteger radiusStride);
    void                 setSegmentControlPointCount(NS::UInteger segmentControlPointCount);
    void                 setSegmentCount(NS::UInteger segmentCount);

};

class InstanceAccelerationStructureDescriptor : public NS::Referencing<InstanceAccelerationStructureDescriptor, MTL::AccelerationStructureDescriptor>
{
public:
    static InstanceAccelerationStructureDescriptor* alloc();
    InstanceAccelerationStructureDescriptor*        init() const;

    static MTL::InstanceAccelerationStructureDescriptor* descriptor();

    NS::UInteger                                     instanceCount() const;
    MTL::Buffer*                                     instanceDescriptorBuffer() const;
    NS::UInteger                                     instanceDescriptorBufferOffset() const;
    NS::UInteger                                     instanceDescriptorStride() const;
    MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType() const;
    MTL::MatrixLayout                                instanceTransformationMatrixLayout() const;
    NS::Array*                                       instancedAccelerationStructures() const;
    MTL::Buffer*                                     motionTransformBuffer() const;
    NS::UInteger                                     motionTransformBufferOffset() const;
    NS::UInteger                                     motionTransformCount() const;
    NS::UInteger                                     motionTransformStride() const;
    MTL::TransformType                               motionTransformType() const;
    void                                             setInstanceCount(NS::UInteger instanceCount);
    void                                             setInstanceDescriptorBuffer(MTL::Buffer* instanceDescriptorBuffer);
    void                                             setInstanceDescriptorBufferOffset(NS::UInteger instanceDescriptorBufferOffset);
    void                                             setInstanceDescriptorStride(NS::UInteger instanceDescriptorStride);
    void                                             setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType);
    void                                             setInstanceTransformationMatrixLayout(MTL::MatrixLayout instanceTransformationMatrixLayout);
    void                                             setInstancedAccelerationStructures(NS::Array* instancedAccelerationStructures);
    void                                             setMotionTransformBuffer(MTL::Buffer* motionTransformBuffer);
    void                                             setMotionTransformBufferOffset(NS::UInteger motionTransformBufferOffset);
    void                                             setMotionTransformCount(NS::UInteger motionTransformCount);
    void                                             setMotionTransformStride(NS::UInteger motionTransformStride);
    void                                             setMotionTransformType(MTL::TransformType motionTransformType);

};

class IndirectInstanceAccelerationStructureDescriptor : public NS::Referencing<IndirectInstanceAccelerationStructureDescriptor, MTL::AccelerationStructureDescriptor>
{
public:
    static IndirectInstanceAccelerationStructureDescriptor* alloc();
    IndirectInstanceAccelerationStructureDescriptor*        init() const;

    static MTL::IndirectInstanceAccelerationStructureDescriptor* descriptor();

    MTL::Buffer*                                     instanceCountBuffer() const;
    NS::UInteger                                     instanceCountBufferOffset() const;
    MTL::Buffer*                                     instanceDescriptorBuffer() const;
    NS::UInteger                                     instanceDescriptorBufferOffset() const;
    NS::UInteger                                     instanceDescriptorStride() const;
    MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType() const;
    MTL::MatrixLayout                                instanceTransformationMatrixLayout() const;
    NS::UInteger                                     maxInstanceCount() const;
    NS::UInteger                                     maxMotionTransformCount() const;
    MTL::Buffer*                                     motionTransformBuffer() const;
    NS::UInteger                                     motionTransformBufferOffset() const;
    MTL::Buffer*                                     motionTransformCountBuffer() const;
    NS::UInteger                                     motionTransformCountBufferOffset() const;
    NS::UInteger                                     motionTransformStride() const;
    MTL::TransformType                               motionTransformType() const;
    void                                             setInstanceCountBuffer(MTL::Buffer* instanceCountBuffer);
    void                                             setInstanceCountBufferOffset(NS::UInteger instanceCountBufferOffset);
    void                                             setInstanceDescriptorBuffer(MTL::Buffer* instanceDescriptorBuffer);
    void                                             setInstanceDescriptorBufferOffset(NS::UInteger instanceDescriptorBufferOffset);
    void                                             setInstanceDescriptorStride(NS::UInteger instanceDescriptorStride);
    void                                             setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType);
    void                                             setInstanceTransformationMatrixLayout(MTL::MatrixLayout instanceTransformationMatrixLayout);
    void                                             setMaxInstanceCount(NS::UInteger maxInstanceCount);
    void                                             setMaxMotionTransformCount(NS::UInteger maxMotionTransformCount);
    void                                             setMotionTransformBuffer(MTL::Buffer* motionTransformBuffer);
    void                                             setMotionTransformBufferOffset(NS::UInteger motionTransformBufferOffset);
    void                                             setMotionTransformCountBuffer(MTL::Buffer* motionTransformCountBuffer);
    void                                             setMotionTransformCountBufferOffset(NS::UInteger motionTransformCountBufferOffset);
    void                                             setMotionTransformStride(NS::UInteger motionTransformStride);
    void                                             setMotionTransformType(MTL::TransformType motionTransformType);

};

class AccelerationStructure : public NS::Referencing<AccelerationStructure, MTL::Resource>
{
public:
    MTL::ResourceID gpuResourceID() const;
    NS::UInteger    size() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLAccelerationStructureDescriptor;
extern "C" void *OBJC_CLASS_$_MTLAccelerationStructureGeometryDescriptor;
extern "C" void *OBJC_CLASS_$_MTLPrimitiveAccelerationStructureDescriptor;
extern "C" void *OBJC_CLASS_$_MTLAccelerationStructureTriangleGeometryDescriptor;
extern "C" void *OBJC_CLASS_$_MTLAccelerationStructureBoundingBoxGeometryDescriptor;
extern "C" void *OBJC_CLASS_$_MTLMotionKeyframeData;
extern "C" void *OBJC_CLASS_$_MTLAccelerationStructureMotionTriangleGeometryDescriptor;
extern "C" void *OBJC_CLASS_$_MTLAccelerationStructureMotionBoundingBoxGeometryDescriptor;
extern "C" void *OBJC_CLASS_$_MTLAccelerationStructureCurveGeometryDescriptor;
extern "C" void *OBJC_CLASS_$_MTLAccelerationStructureMotionCurveGeometryDescriptor;
extern "C" void *OBJC_CLASS_$_MTLInstanceAccelerationStructureDescriptor;
extern "C" void *OBJC_CLASS_$_MTLIndirectInstanceAccelerationStructureDescriptor;
extern "C" void *OBJC_CLASS_$_MTLAccelerationStructure;

_MTL_INLINE MTL::AccelerationStructureDescriptor* MTL::AccelerationStructureDescriptor::alloc()
{
    return _MTL_msg_MTL__AccelerationStructureDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLAccelerationStructureDescriptor, nullptr);
}

_MTL_INLINE MTL::AccelerationStructureDescriptor* MTL::AccelerationStructureDescriptor::init() const
{
    return _MTL_msg_MTL__AccelerationStructureDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::AccelerationStructureUsage MTL::AccelerationStructureDescriptor::usage() const
{
    return _MTL_msg_MTL__AccelerationStructureUsage_usage((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureDescriptor::setUsage(MTL::AccelerationStructureUsage usage)
{
    _MTL_msg_v_setUsage__MTL__AccelerationStructureUsage((const void*)this, nullptr, usage);
}

_MTL_INLINE MTL::AccelerationStructureGeometryDescriptor* MTL::AccelerationStructureGeometryDescriptor::alloc()
{
    return _MTL_msg_MTL__AccelerationStructureGeometryDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLAccelerationStructureGeometryDescriptor, nullptr);
}

_MTL_INLINE MTL::AccelerationStructureGeometryDescriptor* MTL::AccelerationStructureGeometryDescriptor::init() const
{
    return _MTL_msg_MTL__AccelerationStructureGeometryDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureGeometryDescriptor::intersectionFunctionTableOffset() const
{
    return _MTL_msg_NS__UInteger_intersectionFunctionTableOffset((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureGeometryDescriptor::setIntersectionFunctionTableOffset(NS::UInteger intersectionFunctionTableOffset)
{
    _MTL_msg_v_setIntersectionFunctionTableOffset__NS__UInteger((const void*)this, nullptr, intersectionFunctionTableOffset);
}

_MTL_INLINE bool MTL::AccelerationStructureGeometryDescriptor::opaque() const
{
    return _MTL_msg_bool_opaque((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureGeometryDescriptor::setOpaque(bool opaque)
{
    _MTL_msg_v_setOpaque__bool((const void*)this, nullptr, opaque);
}

_MTL_INLINE bool MTL::AccelerationStructureGeometryDescriptor::allowDuplicateIntersectionFunctionInvocation() const
{
    return _MTL_msg_bool_allowDuplicateIntersectionFunctionInvocation((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureGeometryDescriptor::setAllowDuplicateIntersectionFunctionInvocation(bool allowDuplicateIntersectionFunctionInvocation)
{
    _MTL_msg_v_setAllowDuplicateIntersectionFunctionInvocation__bool((const void*)this, nullptr, allowDuplicateIntersectionFunctionInvocation);
}

_MTL_INLINE NS::String* MTL::AccelerationStructureGeometryDescriptor::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureGeometryDescriptor::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureGeometryDescriptor::primitiveDataBuffer() const
{
    return _MTL_msg_MTL__Bufferp_primitiveDataBuffer((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureGeometryDescriptor::setPrimitiveDataBuffer(MTL::Buffer* primitiveDataBuffer)
{
    _MTL_msg_v_setPrimitiveDataBuffer__MTL__Bufferp((const void*)this, nullptr, primitiveDataBuffer);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureGeometryDescriptor::primitiveDataBufferOffset() const
{
    return _MTL_msg_NS__UInteger_primitiveDataBufferOffset((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureGeometryDescriptor::setPrimitiveDataBufferOffset(NS::UInteger primitiveDataBufferOffset)
{
    _MTL_msg_v_setPrimitiveDataBufferOffset__NS__UInteger((const void*)this, nullptr, primitiveDataBufferOffset);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureGeometryDescriptor::primitiveDataStride() const
{
    return _MTL_msg_NS__UInteger_primitiveDataStride((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureGeometryDescriptor::setPrimitiveDataStride(NS::UInteger primitiveDataStride)
{
    _MTL_msg_v_setPrimitiveDataStride__NS__UInteger((const void*)this, nullptr, primitiveDataStride);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureGeometryDescriptor::primitiveDataElementSize() const
{
    return _MTL_msg_NS__UInteger_primitiveDataElementSize((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureGeometryDescriptor::setPrimitiveDataElementSize(NS::UInteger primitiveDataElementSize)
{
    _MTL_msg_v_setPrimitiveDataElementSize__NS__UInteger((const void*)this, nullptr, primitiveDataElementSize);
}

_MTL_INLINE MTL::PrimitiveAccelerationStructureDescriptor* MTL::PrimitiveAccelerationStructureDescriptor::alloc()
{
    return _MTL_msg_MTL__PrimitiveAccelerationStructureDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLPrimitiveAccelerationStructureDescriptor, nullptr);
}

_MTL_INLINE MTL::PrimitiveAccelerationStructureDescriptor* MTL::PrimitiveAccelerationStructureDescriptor::init() const
{
    return _MTL_msg_MTL__PrimitiveAccelerationStructureDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::PrimitiveAccelerationStructureDescriptor* MTL::PrimitiveAccelerationStructureDescriptor::descriptor()
{
    return _MTL_msg_MTL__PrimitiveAccelerationStructureDescriptorp_descriptor((const void*)&OBJC_CLASS_$_MTLPrimitiveAccelerationStructureDescriptor, nullptr);
}

_MTL_INLINE NS::Array* MTL::PrimitiveAccelerationStructureDescriptor::geometryDescriptors() const
{
    return _MTL_msg_NS__Arrayp_geometryDescriptors((const void*)this, nullptr);
}

_MTL_INLINE void MTL::PrimitiveAccelerationStructureDescriptor::setGeometryDescriptors(NS::Array* geometryDescriptors)
{
    _MTL_msg_v_setGeometryDescriptors__NS__Arrayp((const void*)this, nullptr, geometryDescriptors);
}

_MTL_INLINE MTL::MotionBorderMode MTL::PrimitiveAccelerationStructureDescriptor::motionStartBorderMode() const
{
    return _MTL_msg_MTL__MotionBorderMode_motionStartBorderMode((const void*)this, nullptr);
}

_MTL_INLINE void MTL::PrimitiveAccelerationStructureDescriptor::setMotionStartBorderMode(MTL::MotionBorderMode motionStartBorderMode)
{
    _MTL_msg_v_setMotionStartBorderMode__MTL__MotionBorderMode((const void*)this, nullptr, motionStartBorderMode);
}

_MTL_INLINE MTL::MotionBorderMode MTL::PrimitiveAccelerationStructureDescriptor::motionEndBorderMode() const
{
    return _MTL_msg_MTL__MotionBorderMode_motionEndBorderMode((const void*)this, nullptr);
}

_MTL_INLINE void MTL::PrimitiveAccelerationStructureDescriptor::setMotionEndBorderMode(MTL::MotionBorderMode motionEndBorderMode)
{
    _MTL_msg_v_setMotionEndBorderMode__MTL__MotionBorderMode((const void*)this, nullptr, motionEndBorderMode);
}

_MTL_INLINE float MTL::PrimitiveAccelerationStructureDescriptor::motionStartTime() const
{
    return _MTL_msg_float_motionStartTime((const void*)this, nullptr);
}

_MTL_INLINE void MTL::PrimitiveAccelerationStructureDescriptor::setMotionStartTime(float motionStartTime)
{
    _MTL_msg_v_setMotionStartTime__float((const void*)this, nullptr, motionStartTime);
}

_MTL_INLINE float MTL::PrimitiveAccelerationStructureDescriptor::motionEndTime() const
{
    return _MTL_msg_float_motionEndTime((const void*)this, nullptr);
}

_MTL_INLINE void MTL::PrimitiveAccelerationStructureDescriptor::setMotionEndTime(float motionEndTime)
{
    _MTL_msg_v_setMotionEndTime__float((const void*)this, nullptr, motionEndTime);
}

_MTL_INLINE NS::UInteger MTL::PrimitiveAccelerationStructureDescriptor::motionKeyframeCount() const
{
    return _MTL_msg_NS__UInteger_motionKeyframeCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::PrimitiveAccelerationStructureDescriptor::setMotionKeyframeCount(NS::UInteger motionKeyframeCount)
{
    _MTL_msg_v_setMotionKeyframeCount__NS__UInteger((const void*)this, nullptr, motionKeyframeCount);
}

_MTL_INLINE MTL::AccelerationStructureTriangleGeometryDescriptor* MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()
{
    return _MTL_msg_MTL__AccelerationStructureTriangleGeometryDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLAccelerationStructureTriangleGeometryDescriptor, nullptr);
}

_MTL_INLINE MTL::AccelerationStructureTriangleGeometryDescriptor* MTL::AccelerationStructureTriangleGeometryDescriptor::init() const
{
    return _MTL_msg_MTL__AccelerationStructureTriangleGeometryDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::AccelerationStructureTriangleGeometryDescriptor* MTL::AccelerationStructureTriangleGeometryDescriptor::descriptor()
{
    return _MTL_msg_MTL__AccelerationStructureTriangleGeometryDescriptorp_descriptor((const void*)&OBJC_CLASS_$_MTLAccelerationStructureTriangleGeometryDescriptor, nullptr);
}

_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureTriangleGeometryDescriptor::vertexBuffer() const
{
    return _MTL_msg_MTL__Bufferp_vertexBuffer((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setVertexBuffer(MTL::Buffer* vertexBuffer)
{
    _MTL_msg_v_setVertexBuffer__MTL__Bufferp((const void*)this, nullptr, vertexBuffer);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureTriangleGeometryDescriptor::vertexBufferOffset() const
{
    return _MTL_msg_NS__UInteger_vertexBufferOffset((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setVertexBufferOffset(NS::UInteger vertexBufferOffset)
{
    _MTL_msg_v_setVertexBufferOffset__NS__UInteger((const void*)this, nullptr, vertexBufferOffset);
}

_MTL_INLINE MTL::AttributeFormat MTL::AccelerationStructureTriangleGeometryDescriptor::vertexFormat() const
{
    return _MTL_msg_MTL__AttributeFormat_vertexFormat((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setVertexFormat(MTL::AttributeFormat vertexFormat)
{
    _MTL_msg_v_setVertexFormat__MTL__AttributeFormat((const void*)this, nullptr, vertexFormat);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureTriangleGeometryDescriptor::vertexStride() const
{
    return _MTL_msg_NS__UInteger_vertexStride((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setVertexStride(NS::UInteger vertexStride)
{
    _MTL_msg_v_setVertexStride__NS__UInteger((const void*)this, nullptr, vertexStride);
}

_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureTriangleGeometryDescriptor::indexBuffer() const
{
    return _MTL_msg_MTL__Bufferp_indexBuffer((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setIndexBuffer(MTL::Buffer* indexBuffer)
{
    _MTL_msg_v_setIndexBuffer__MTL__Bufferp((const void*)this, nullptr, indexBuffer);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureTriangleGeometryDescriptor::indexBufferOffset() const
{
    return _MTL_msg_NS__UInteger_indexBufferOffset((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setIndexBufferOffset(NS::UInteger indexBufferOffset)
{
    _MTL_msg_v_setIndexBufferOffset__NS__UInteger((const void*)this, nullptr, indexBufferOffset);
}

_MTL_INLINE MTL::IndexType MTL::AccelerationStructureTriangleGeometryDescriptor::indexType() const
{
    return _MTL_msg_MTL__IndexType_indexType((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setIndexType(MTL::IndexType indexType)
{
    _MTL_msg_v_setIndexType__MTL__IndexType((const void*)this, nullptr, indexType);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureTriangleGeometryDescriptor::triangleCount() const
{
    return _MTL_msg_NS__UInteger_triangleCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setTriangleCount(NS::UInteger triangleCount)
{
    _MTL_msg_v_setTriangleCount__NS__UInteger((const void*)this, nullptr, triangleCount);
}

_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureTriangleGeometryDescriptor::transformationMatrixBuffer() const
{
    return _MTL_msg_MTL__Bufferp_transformationMatrixBuffer((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setTransformationMatrixBuffer(MTL::Buffer* transformationMatrixBuffer)
{
    _MTL_msg_v_setTransformationMatrixBuffer__MTL__Bufferp((const void*)this, nullptr, transformationMatrixBuffer);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureTriangleGeometryDescriptor::transformationMatrixBufferOffset() const
{
    return _MTL_msg_NS__UInteger_transformationMatrixBufferOffset((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setTransformationMatrixBufferOffset(NS::UInteger transformationMatrixBufferOffset)
{
    _MTL_msg_v_setTransformationMatrixBufferOffset__NS__UInteger((const void*)this, nullptr, transformationMatrixBufferOffset);
}

_MTL_INLINE MTL::MatrixLayout MTL::AccelerationStructureTriangleGeometryDescriptor::transformationMatrixLayout() const
{
    return _MTL_msg_MTL__MatrixLayout_transformationMatrixLayout((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setTransformationMatrixLayout(MTL::MatrixLayout transformationMatrixLayout)
{
    _MTL_msg_v_setTransformationMatrixLayout__MTL__MatrixLayout((const void*)this, nullptr, transformationMatrixLayout);
}

_MTL_INLINE MTL::AccelerationStructureBoundingBoxGeometryDescriptor* MTL::AccelerationStructureBoundingBoxGeometryDescriptor::alloc()
{
    return _MTL_msg_MTL__AccelerationStructureBoundingBoxGeometryDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLAccelerationStructureBoundingBoxGeometryDescriptor, nullptr);
}

_MTL_INLINE MTL::AccelerationStructureBoundingBoxGeometryDescriptor* MTL::AccelerationStructureBoundingBoxGeometryDescriptor::init() const
{
    return _MTL_msg_MTL__AccelerationStructureBoundingBoxGeometryDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::AccelerationStructureBoundingBoxGeometryDescriptor* MTL::AccelerationStructureBoundingBoxGeometryDescriptor::descriptor()
{
    return _MTL_msg_MTL__AccelerationStructureBoundingBoxGeometryDescriptorp_descriptor((const void*)&OBJC_CLASS_$_MTLAccelerationStructureBoundingBoxGeometryDescriptor, nullptr);
}

_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureBoundingBoxGeometryDescriptor::boundingBoxBuffer() const
{
    return _MTL_msg_MTL__Bufferp_boundingBoxBuffer((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureBoundingBoxGeometryDescriptor::setBoundingBoxBuffer(MTL::Buffer* boundingBoxBuffer)
{
    _MTL_msg_v_setBoundingBoxBuffer__MTL__Bufferp((const void*)this, nullptr, boundingBoxBuffer);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureBoundingBoxGeometryDescriptor::boundingBoxBufferOffset() const
{
    return _MTL_msg_NS__UInteger_boundingBoxBufferOffset((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureBoundingBoxGeometryDescriptor::setBoundingBoxBufferOffset(NS::UInteger boundingBoxBufferOffset)
{
    _MTL_msg_v_setBoundingBoxBufferOffset__NS__UInteger((const void*)this, nullptr, boundingBoxBufferOffset);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureBoundingBoxGeometryDescriptor::boundingBoxStride() const
{
    return _MTL_msg_NS__UInteger_boundingBoxStride((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureBoundingBoxGeometryDescriptor::setBoundingBoxStride(NS::UInteger boundingBoxStride)
{
    _MTL_msg_v_setBoundingBoxStride__NS__UInteger((const void*)this, nullptr, boundingBoxStride);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureBoundingBoxGeometryDescriptor::boundingBoxCount() const
{
    return _MTL_msg_NS__UInteger_boundingBoxCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureBoundingBoxGeometryDescriptor::setBoundingBoxCount(NS::UInteger boundingBoxCount)
{
    _MTL_msg_v_setBoundingBoxCount__NS__UInteger((const void*)this, nullptr, boundingBoxCount);
}

_MTL_INLINE MTL::MotionKeyframeData* MTL::MotionKeyframeData::alloc()
{
    return _MTL_msg_MTL__MotionKeyframeDatap_alloc((const void*)&OBJC_CLASS_$_MTLMotionKeyframeData, nullptr);
}

_MTL_INLINE MTL::MotionKeyframeData* MTL::MotionKeyframeData::init() const
{
    return _MTL_msg_MTL__MotionKeyframeDatap_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::MotionKeyframeData* MTL::MotionKeyframeData::data()
{
    return _MTL_msg_MTL__MotionKeyframeDatap_data((const void*)&OBJC_CLASS_$_MTLMotionKeyframeData, nullptr);
}

_MTL_INLINE MTL::Buffer* MTL::MotionKeyframeData::buffer() const
{
    return _MTL_msg_MTL__Bufferp_buffer((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MotionKeyframeData::setBuffer(MTL::Buffer* buffer)
{
    _MTL_msg_v_setBuffer__MTL__Bufferp((const void*)this, nullptr, buffer);
}

_MTL_INLINE NS::UInteger MTL::MotionKeyframeData::offset() const
{
    return _MTL_msg_NS__UInteger_offset((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MotionKeyframeData::setOffset(NS::UInteger offset)
{
    _MTL_msg_v_setOffset__NS__UInteger((const void*)this, nullptr, offset);
}

_MTL_INLINE MTL::AccelerationStructureMotionTriangleGeometryDescriptor* MTL::AccelerationStructureMotionTriangleGeometryDescriptor::alloc()
{
    return _MTL_msg_MTL__AccelerationStructureMotionTriangleGeometryDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLAccelerationStructureMotionTriangleGeometryDescriptor, nullptr);
}

_MTL_INLINE MTL::AccelerationStructureMotionTriangleGeometryDescriptor* MTL::AccelerationStructureMotionTriangleGeometryDescriptor::init() const
{
    return _MTL_msg_MTL__AccelerationStructureMotionTriangleGeometryDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::AccelerationStructureMotionTriangleGeometryDescriptor* MTL::AccelerationStructureMotionTriangleGeometryDescriptor::descriptor()
{
    return _MTL_msg_MTL__AccelerationStructureMotionTriangleGeometryDescriptorp_descriptor((const void*)&OBJC_CLASS_$_MTLAccelerationStructureMotionTriangleGeometryDescriptor, nullptr);
}

_MTL_INLINE NS::Array* MTL::AccelerationStructureMotionTriangleGeometryDescriptor::vertexBuffers() const
{
    return _MTL_msg_NS__Arrayp_vertexBuffers((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setVertexBuffers(NS::Array* vertexBuffers)
{
    _MTL_msg_v_setVertexBuffers__NS__Arrayp((const void*)this, nullptr, vertexBuffers);
}

_MTL_INLINE MTL::AttributeFormat MTL::AccelerationStructureMotionTriangleGeometryDescriptor::vertexFormat() const
{
    return _MTL_msg_MTL__AttributeFormat_vertexFormat((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setVertexFormat(MTL::AttributeFormat vertexFormat)
{
    _MTL_msg_v_setVertexFormat__MTL__AttributeFormat((const void*)this, nullptr, vertexFormat);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionTriangleGeometryDescriptor::vertexStride() const
{
    return _MTL_msg_NS__UInteger_vertexStride((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setVertexStride(NS::UInteger vertexStride)
{
    _MTL_msg_v_setVertexStride__NS__UInteger((const void*)this, nullptr, vertexStride);
}

_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureMotionTriangleGeometryDescriptor::indexBuffer() const
{
    return _MTL_msg_MTL__Bufferp_indexBuffer((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setIndexBuffer(MTL::Buffer* indexBuffer)
{
    _MTL_msg_v_setIndexBuffer__MTL__Bufferp((const void*)this, nullptr, indexBuffer);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionTriangleGeometryDescriptor::indexBufferOffset() const
{
    return _MTL_msg_NS__UInteger_indexBufferOffset((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setIndexBufferOffset(NS::UInteger indexBufferOffset)
{
    _MTL_msg_v_setIndexBufferOffset__NS__UInteger((const void*)this, nullptr, indexBufferOffset);
}

_MTL_INLINE MTL::IndexType MTL::AccelerationStructureMotionTriangleGeometryDescriptor::indexType() const
{
    return _MTL_msg_MTL__IndexType_indexType((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setIndexType(MTL::IndexType indexType)
{
    _MTL_msg_v_setIndexType__MTL__IndexType((const void*)this, nullptr, indexType);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionTriangleGeometryDescriptor::triangleCount() const
{
    return _MTL_msg_NS__UInteger_triangleCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setTriangleCount(NS::UInteger triangleCount)
{
    _MTL_msg_v_setTriangleCount__NS__UInteger((const void*)this, nullptr, triangleCount);
}

_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureMotionTriangleGeometryDescriptor::transformationMatrixBuffer() const
{
    return _MTL_msg_MTL__Bufferp_transformationMatrixBuffer((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setTransformationMatrixBuffer(MTL::Buffer* transformationMatrixBuffer)
{
    _MTL_msg_v_setTransformationMatrixBuffer__MTL__Bufferp((const void*)this, nullptr, transformationMatrixBuffer);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionTriangleGeometryDescriptor::transformationMatrixBufferOffset() const
{
    return _MTL_msg_NS__UInteger_transformationMatrixBufferOffset((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setTransformationMatrixBufferOffset(NS::UInteger transformationMatrixBufferOffset)
{
    _MTL_msg_v_setTransformationMatrixBufferOffset__NS__UInteger((const void*)this, nullptr, transformationMatrixBufferOffset);
}

_MTL_INLINE MTL::MatrixLayout MTL::AccelerationStructureMotionTriangleGeometryDescriptor::transformationMatrixLayout() const
{
    return _MTL_msg_MTL__MatrixLayout_transformationMatrixLayout((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setTransformationMatrixLayout(MTL::MatrixLayout transformationMatrixLayout)
{
    _MTL_msg_v_setTransformationMatrixLayout__MTL__MatrixLayout((const void*)this, nullptr, transformationMatrixLayout);
}

_MTL_INLINE MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor* MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::alloc()
{
    return _MTL_msg_MTL__AccelerationStructureMotionBoundingBoxGeometryDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLAccelerationStructureMotionBoundingBoxGeometryDescriptor, nullptr);
}

_MTL_INLINE MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor* MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::init() const
{
    return _MTL_msg_MTL__AccelerationStructureMotionBoundingBoxGeometryDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor* MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::descriptor()
{
    return _MTL_msg_MTL__AccelerationStructureMotionBoundingBoxGeometryDescriptorp_descriptor((const void*)&OBJC_CLASS_$_MTLAccelerationStructureMotionBoundingBoxGeometryDescriptor, nullptr);
}

_MTL_INLINE NS::Array* MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::boundingBoxBuffers() const
{
    return _MTL_msg_NS__Arrayp_boundingBoxBuffers((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::setBoundingBoxBuffers(NS::Array* boundingBoxBuffers)
{
    _MTL_msg_v_setBoundingBoxBuffers__NS__Arrayp((const void*)this, nullptr, boundingBoxBuffers);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::boundingBoxStride() const
{
    return _MTL_msg_NS__UInteger_boundingBoxStride((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::setBoundingBoxStride(NS::UInteger boundingBoxStride)
{
    _MTL_msg_v_setBoundingBoxStride__NS__UInteger((const void*)this, nullptr, boundingBoxStride);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::boundingBoxCount() const
{
    return _MTL_msg_NS__UInteger_boundingBoxCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::setBoundingBoxCount(NS::UInteger boundingBoxCount)
{
    _MTL_msg_v_setBoundingBoxCount__NS__UInteger((const void*)this, nullptr, boundingBoxCount);
}

_MTL_INLINE MTL::AccelerationStructureCurveGeometryDescriptor* MTL::AccelerationStructureCurveGeometryDescriptor::alloc()
{
    return _MTL_msg_MTL__AccelerationStructureCurveGeometryDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLAccelerationStructureCurveGeometryDescriptor, nullptr);
}

_MTL_INLINE MTL::AccelerationStructureCurveGeometryDescriptor* MTL::AccelerationStructureCurveGeometryDescriptor::init() const
{
    return _MTL_msg_MTL__AccelerationStructureCurveGeometryDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::AccelerationStructureCurveGeometryDescriptor* MTL::AccelerationStructureCurveGeometryDescriptor::descriptor()
{
    return _MTL_msg_MTL__AccelerationStructureCurveGeometryDescriptorp_descriptor((const void*)&OBJC_CLASS_$_MTLAccelerationStructureCurveGeometryDescriptor, nullptr);
}

_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureCurveGeometryDescriptor::controlPointBuffer() const
{
    return _MTL_msg_MTL__Bufferp_controlPointBuffer((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setControlPointBuffer(MTL::Buffer* controlPointBuffer)
{
    _MTL_msg_v_setControlPointBuffer__MTL__Bufferp((const void*)this, nullptr, controlPointBuffer);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureCurveGeometryDescriptor::controlPointBufferOffset() const
{
    return _MTL_msg_NS__UInteger_controlPointBufferOffset((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setControlPointBufferOffset(NS::UInteger controlPointBufferOffset)
{
    _MTL_msg_v_setControlPointBufferOffset__NS__UInteger((const void*)this, nullptr, controlPointBufferOffset);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureCurveGeometryDescriptor::controlPointCount() const
{
    return _MTL_msg_NS__UInteger_controlPointCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setControlPointCount(NS::UInteger controlPointCount)
{
    _MTL_msg_v_setControlPointCount__NS__UInteger((const void*)this, nullptr, controlPointCount);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureCurveGeometryDescriptor::controlPointStride() const
{
    return _MTL_msg_NS__UInteger_controlPointStride((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setControlPointStride(NS::UInteger controlPointStride)
{
    _MTL_msg_v_setControlPointStride__NS__UInteger((const void*)this, nullptr, controlPointStride);
}

_MTL_INLINE MTL::AttributeFormat MTL::AccelerationStructureCurveGeometryDescriptor::controlPointFormat() const
{
    return _MTL_msg_MTL__AttributeFormat_controlPointFormat((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setControlPointFormat(MTL::AttributeFormat controlPointFormat)
{
    _MTL_msg_v_setControlPointFormat__MTL__AttributeFormat((const void*)this, nullptr, controlPointFormat);
}

_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureCurveGeometryDescriptor::radiusBuffer() const
{
    return _MTL_msg_MTL__Bufferp_radiusBuffer((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setRadiusBuffer(MTL::Buffer* radiusBuffer)
{
    _MTL_msg_v_setRadiusBuffer__MTL__Bufferp((const void*)this, nullptr, radiusBuffer);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureCurveGeometryDescriptor::radiusBufferOffset() const
{
    return _MTL_msg_NS__UInteger_radiusBufferOffset((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setRadiusBufferOffset(NS::UInteger radiusBufferOffset)
{
    _MTL_msg_v_setRadiusBufferOffset__NS__UInteger((const void*)this, nullptr, radiusBufferOffset);
}

_MTL_INLINE MTL::AttributeFormat MTL::AccelerationStructureCurveGeometryDescriptor::radiusFormat() const
{
    return _MTL_msg_MTL__AttributeFormat_radiusFormat((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setRadiusFormat(MTL::AttributeFormat radiusFormat)
{
    _MTL_msg_v_setRadiusFormat__MTL__AttributeFormat((const void*)this, nullptr, radiusFormat);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureCurveGeometryDescriptor::radiusStride() const
{
    return _MTL_msg_NS__UInteger_radiusStride((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setRadiusStride(NS::UInteger radiusStride)
{
    _MTL_msg_v_setRadiusStride__NS__UInteger((const void*)this, nullptr, radiusStride);
}

_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureCurveGeometryDescriptor::indexBuffer() const
{
    return _MTL_msg_MTL__Bufferp_indexBuffer((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setIndexBuffer(MTL::Buffer* indexBuffer)
{
    _MTL_msg_v_setIndexBuffer__MTL__Bufferp((const void*)this, nullptr, indexBuffer);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureCurveGeometryDescriptor::indexBufferOffset() const
{
    return _MTL_msg_NS__UInteger_indexBufferOffset((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setIndexBufferOffset(NS::UInteger indexBufferOffset)
{
    _MTL_msg_v_setIndexBufferOffset__NS__UInteger((const void*)this, nullptr, indexBufferOffset);
}

_MTL_INLINE MTL::IndexType MTL::AccelerationStructureCurveGeometryDescriptor::indexType() const
{
    return _MTL_msg_MTL__IndexType_indexType((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setIndexType(MTL::IndexType indexType)
{
    _MTL_msg_v_setIndexType__MTL__IndexType((const void*)this, nullptr, indexType);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureCurveGeometryDescriptor::segmentCount() const
{
    return _MTL_msg_NS__UInteger_segmentCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setSegmentCount(NS::UInteger segmentCount)
{
    _MTL_msg_v_setSegmentCount__NS__UInteger((const void*)this, nullptr, segmentCount);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureCurveGeometryDescriptor::segmentControlPointCount() const
{
    return _MTL_msg_NS__UInteger_segmentControlPointCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setSegmentControlPointCount(NS::UInteger segmentControlPointCount)
{
    _MTL_msg_v_setSegmentControlPointCount__NS__UInteger((const void*)this, nullptr, segmentControlPointCount);
}

_MTL_INLINE MTL::CurveType MTL::AccelerationStructureCurveGeometryDescriptor::curveType() const
{
    return _MTL_msg_MTL__CurveType_curveType((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setCurveType(MTL::CurveType curveType)
{
    _MTL_msg_v_setCurveType__MTL__CurveType((const void*)this, nullptr, curveType);
}

_MTL_INLINE MTL::CurveBasis MTL::AccelerationStructureCurveGeometryDescriptor::curveBasis() const
{
    return _MTL_msg_MTL__CurveBasis_curveBasis((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setCurveBasis(MTL::CurveBasis curveBasis)
{
    _MTL_msg_v_setCurveBasis__MTL__CurveBasis((const void*)this, nullptr, curveBasis);
}

_MTL_INLINE MTL::CurveEndCaps MTL::AccelerationStructureCurveGeometryDescriptor::curveEndCaps() const
{
    return _MTL_msg_MTL__CurveEndCaps_curveEndCaps((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setCurveEndCaps(MTL::CurveEndCaps curveEndCaps)
{
    _MTL_msg_v_setCurveEndCaps__MTL__CurveEndCaps((const void*)this, nullptr, curveEndCaps);
}

_MTL_INLINE MTL::AccelerationStructureMotionCurveGeometryDescriptor* MTL::AccelerationStructureMotionCurveGeometryDescriptor::alloc()
{
    return _MTL_msg_MTL__AccelerationStructureMotionCurveGeometryDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLAccelerationStructureMotionCurveGeometryDescriptor, nullptr);
}

_MTL_INLINE MTL::AccelerationStructureMotionCurveGeometryDescriptor* MTL::AccelerationStructureMotionCurveGeometryDescriptor::init() const
{
    return _MTL_msg_MTL__AccelerationStructureMotionCurveGeometryDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::AccelerationStructureMotionCurveGeometryDescriptor* MTL::AccelerationStructureMotionCurveGeometryDescriptor::descriptor()
{
    return _MTL_msg_MTL__AccelerationStructureMotionCurveGeometryDescriptorp_descriptor((const void*)&OBJC_CLASS_$_MTLAccelerationStructureMotionCurveGeometryDescriptor, nullptr);
}

_MTL_INLINE NS::Array* MTL::AccelerationStructureMotionCurveGeometryDescriptor::controlPointBuffers() const
{
    return _MTL_msg_NS__Arrayp_controlPointBuffers((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setControlPointBuffers(NS::Array* controlPointBuffers)
{
    _MTL_msg_v_setControlPointBuffers__NS__Arrayp((const void*)this, nullptr, controlPointBuffers);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionCurveGeometryDescriptor::controlPointCount() const
{
    return _MTL_msg_NS__UInteger_controlPointCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setControlPointCount(NS::UInteger controlPointCount)
{
    _MTL_msg_v_setControlPointCount__NS__UInteger((const void*)this, nullptr, controlPointCount);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionCurveGeometryDescriptor::controlPointStride() const
{
    return _MTL_msg_NS__UInteger_controlPointStride((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setControlPointStride(NS::UInteger controlPointStride)
{
    _MTL_msg_v_setControlPointStride__NS__UInteger((const void*)this, nullptr, controlPointStride);
}

_MTL_INLINE MTL::AttributeFormat MTL::AccelerationStructureMotionCurveGeometryDescriptor::controlPointFormat() const
{
    return _MTL_msg_MTL__AttributeFormat_controlPointFormat((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setControlPointFormat(MTL::AttributeFormat controlPointFormat)
{
    _MTL_msg_v_setControlPointFormat__MTL__AttributeFormat((const void*)this, nullptr, controlPointFormat);
}

_MTL_INLINE NS::Array* MTL::AccelerationStructureMotionCurveGeometryDescriptor::radiusBuffers() const
{
    return _MTL_msg_NS__Arrayp_radiusBuffers((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setRadiusBuffers(NS::Array* radiusBuffers)
{
    _MTL_msg_v_setRadiusBuffers__NS__Arrayp((const void*)this, nullptr, radiusBuffers);
}

_MTL_INLINE MTL::AttributeFormat MTL::AccelerationStructureMotionCurveGeometryDescriptor::radiusFormat() const
{
    return _MTL_msg_MTL__AttributeFormat_radiusFormat((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setRadiusFormat(MTL::AttributeFormat radiusFormat)
{
    _MTL_msg_v_setRadiusFormat__MTL__AttributeFormat((const void*)this, nullptr, radiusFormat);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionCurveGeometryDescriptor::radiusStride() const
{
    return _MTL_msg_NS__UInteger_radiusStride((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setRadiusStride(NS::UInteger radiusStride)
{
    _MTL_msg_v_setRadiusStride__NS__UInteger((const void*)this, nullptr, radiusStride);
}

_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureMotionCurveGeometryDescriptor::indexBuffer() const
{
    return _MTL_msg_MTL__Bufferp_indexBuffer((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setIndexBuffer(MTL::Buffer* indexBuffer)
{
    _MTL_msg_v_setIndexBuffer__MTL__Bufferp((const void*)this, nullptr, indexBuffer);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionCurveGeometryDescriptor::indexBufferOffset() const
{
    return _MTL_msg_NS__UInteger_indexBufferOffset((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setIndexBufferOffset(NS::UInteger indexBufferOffset)
{
    _MTL_msg_v_setIndexBufferOffset__NS__UInteger((const void*)this, nullptr, indexBufferOffset);
}

_MTL_INLINE MTL::IndexType MTL::AccelerationStructureMotionCurveGeometryDescriptor::indexType() const
{
    return _MTL_msg_MTL__IndexType_indexType((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setIndexType(MTL::IndexType indexType)
{
    _MTL_msg_v_setIndexType__MTL__IndexType((const void*)this, nullptr, indexType);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionCurveGeometryDescriptor::segmentCount() const
{
    return _MTL_msg_NS__UInteger_segmentCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setSegmentCount(NS::UInteger segmentCount)
{
    _MTL_msg_v_setSegmentCount__NS__UInteger((const void*)this, nullptr, segmentCount);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionCurveGeometryDescriptor::segmentControlPointCount() const
{
    return _MTL_msg_NS__UInteger_segmentControlPointCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setSegmentControlPointCount(NS::UInteger segmentControlPointCount)
{
    _MTL_msg_v_setSegmentControlPointCount__NS__UInteger((const void*)this, nullptr, segmentControlPointCount);
}

_MTL_INLINE MTL::CurveType MTL::AccelerationStructureMotionCurveGeometryDescriptor::curveType() const
{
    return _MTL_msg_MTL__CurveType_curveType((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setCurveType(MTL::CurveType curveType)
{
    _MTL_msg_v_setCurveType__MTL__CurveType((const void*)this, nullptr, curveType);
}

_MTL_INLINE MTL::CurveBasis MTL::AccelerationStructureMotionCurveGeometryDescriptor::curveBasis() const
{
    return _MTL_msg_MTL__CurveBasis_curveBasis((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setCurveBasis(MTL::CurveBasis curveBasis)
{
    _MTL_msg_v_setCurveBasis__MTL__CurveBasis((const void*)this, nullptr, curveBasis);
}

_MTL_INLINE MTL::CurveEndCaps MTL::AccelerationStructureMotionCurveGeometryDescriptor::curveEndCaps() const
{
    return _MTL_msg_MTL__CurveEndCaps_curveEndCaps((const void*)this, nullptr);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setCurveEndCaps(MTL::CurveEndCaps curveEndCaps)
{
    _MTL_msg_v_setCurveEndCaps__MTL__CurveEndCaps((const void*)this, nullptr, curveEndCaps);
}

_MTL_INLINE MTL::InstanceAccelerationStructureDescriptor* MTL::InstanceAccelerationStructureDescriptor::alloc()
{
    return _MTL_msg_MTL__InstanceAccelerationStructureDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLInstanceAccelerationStructureDescriptor, nullptr);
}

_MTL_INLINE MTL::InstanceAccelerationStructureDescriptor* MTL::InstanceAccelerationStructureDescriptor::init() const
{
    return _MTL_msg_MTL__InstanceAccelerationStructureDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::InstanceAccelerationStructureDescriptor* MTL::InstanceAccelerationStructureDescriptor::descriptor()
{
    return _MTL_msg_MTL__InstanceAccelerationStructureDescriptorp_descriptor((const void*)&OBJC_CLASS_$_MTLInstanceAccelerationStructureDescriptor, nullptr);
}

_MTL_INLINE MTL::Buffer* MTL::InstanceAccelerationStructureDescriptor::instanceDescriptorBuffer() const
{
    return _MTL_msg_MTL__Bufferp_instanceDescriptorBuffer((const void*)this, nullptr);
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setInstanceDescriptorBuffer(MTL::Buffer* instanceDescriptorBuffer)
{
    _MTL_msg_v_setInstanceDescriptorBuffer__MTL__Bufferp((const void*)this, nullptr, instanceDescriptorBuffer);
}

_MTL_INLINE NS::UInteger MTL::InstanceAccelerationStructureDescriptor::instanceDescriptorBufferOffset() const
{
    return _MTL_msg_NS__UInteger_instanceDescriptorBufferOffset((const void*)this, nullptr);
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setInstanceDescriptorBufferOffset(NS::UInteger instanceDescriptorBufferOffset)
{
    _MTL_msg_v_setInstanceDescriptorBufferOffset__NS__UInteger((const void*)this, nullptr, instanceDescriptorBufferOffset);
}

_MTL_INLINE NS::UInteger MTL::InstanceAccelerationStructureDescriptor::instanceDescriptorStride() const
{
    return _MTL_msg_NS__UInteger_instanceDescriptorStride((const void*)this, nullptr);
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setInstanceDescriptorStride(NS::UInteger instanceDescriptorStride)
{
    _MTL_msg_v_setInstanceDescriptorStride__NS__UInteger((const void*)this, nullptr, instanceDescriptorStride);
}

_MTL_INLINE NS::UInteger MTL::InstanceAccelerationStructureDescriptor::instanceCount() const
{
    return _MTL_msg_NS__UInteger_instanceCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setInstanceCount(NS::UInteger instanceCount)
{
    _MTL_msg_v_setInstanceCount__NS__UInteger((const void*)this, nullptr, instanceCount);
}

_MTL_INLINE NS::Array* MTL::InstanceAccelerationStructureDescriptor::instancedAccelerationStructures() const
{
    return _MTL_msg_NS__Arrayp_instancedAccelerationStructures((const void*)this, nullptr);
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setInstancedAccelerationStructures(NS::Array* instancedAccelerationStructures)
{
    _MTL_msg_v_setInstancedAccelerationStructures__NS__Arrayp((const void*)this, nullptr, instancedAccelerationStructures);
}

_MTL_INLINE MTL::AccelerationStructureInstanceDescriptorType MTL::InstanceAccelerationStructureDescriptor::instanceDescriptorType() const
{
    return _MTL_msg_MTL__AccelerationStructureInstanceDescriptorType_instanceDescriptorType((const void*)this, nullptr);
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType)
{
    _MTL_msg_v_setInstanceDescriptorType__MTL__AccelerationStructureInstanceDescriptorType((const void*)this, nullptr, instanceDescriptorType);
}

_MTL_INLINE MTL::Buffer* MTL::InstanceAccelerationStructureDescriptor::motionTransformBuffer() const
{
    return _MTL_msg_MTL__Bufferp_motionTransformBuffer((const void*)this, nullptr);
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setMotionTransformBuffer(MTL::Buffer* motionTransformBuffer)
{
    _MTL_msg_v_setMotionTransformBuffer__MTL__Bufferp((const void*)this, nullptr, motionTransformBuffer);
}

_MTL_INLINE NS::UInteger MTL::InstanceAccelerationStructureDescriptor::motionTransformBufferOffset() const
{
    return _MTL_msg_NS__UInteger_motionTransformBufferOffset((const void*)this, nullptr);
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setMotionTransformBufferOffset(NS::UInteger motionTransformBufferOffset)
{
    _MTL_msg_v_setMotionTransformBufferOffset__NS__UInteger((const void*)this, nullptr, motionTransformBufferOffset);
}

_MTL_INLINE NS::UInteger MTL::InstanceAccelerationStructureDescriptor::motionTransformCount() const
{
    return _MTL_msg_NS__UInteger_motionTransformCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setMotionTransformCount(NS::UInteger motionTransformCount)
{
    _MTL_msg_v_setMotionTransformCount__NS__UInteger((const void*)this, nullptr, motionTransformCount);
}

_MTL_INLINE MTL::MatrixLayout MTL::InstanceAccelerationStructureDescriptor::instanceTransformationMatrixLayout() const
{
    return _MTL_msg_MTL__MatrixLayout_instanceTransformationMatrixLayout((const void*)this, nullptr);
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setInstanceTransformationMatrixLayout(MTL::MatrixLayout instanceTransformationMatrixLayout)
{
    _MTL_msg_v_setInstanceTransformationMatrixLayout__MTL__MatrixLayout((const void*)this, nullptr, instanceTransformationMatrixLayout);
}

_MTL_INLINE MTL::TransformType MTL::InstanceAccelerationStructureDescriptor::motionTransformType() const
{
    return _MTL_msg_MTL__TransformType_motionTransformType((const void*)this, nullptr);
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setMotionTransformType(MTL::TransformType motionTransformType)
{
    _MTL_msg_v_setMotionTransformType__MTL__TransformType((const void*)this, nullptr, motionTransformType);
}

_MTL_INLINE NS::UInteger MTL::InstanceAccelerationStructureDescriptor::motionTransformStride() const
{
    return _MTL_msg_NS__UInteger_motionTransformStride((const void*)this, nullptr);
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setMotionTransformStride(NS::UInteger motionTransformStride)
{
    _MTL_msg_v_setMotionTransformStride__NS__UInteger((const void*)this, nullptr, motionTransformStride);
}

_MTL_INLINE MTL::IndirectInstanceAccelerationStructureDescriptor* MTL::IndirectInstanceAccelerationStructureDescriptor::alloc()
{
    return _MTL_msg_MTL__IndirectInstanceAccelerationStructureDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLIndirectInstanceAccelerationStructureDescriptor, nullptr);
}

_MTL_INLINE MTL::IndirectInstanceAccelerationStructureDescriptor* MTL::IndirectInstanceAccelerationStructureDescriptor::init() const
{
    return _MTL_msg_MTL__IndirectInstanceAccelerationStructureDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::IndirectInstanceAccelerationStructureDescriptor* MTL::IndirectInstanceAccelerationStructureDescriptor::descriptor()
{
    return _MTL_msg_MTL__IndirectInstanceAccelerationStructureDescriptorp_descriptor((const void*)&OBJC_CLASS_$_MTLIndirectInstanceAccelerationStructureDescriptor, nullptr);
}

_MTL_INLINE MTL::Buffer* MTL::IndirectInstanceAccelerationStructureDescriptor::instanceDescriptorBuffer() const
{
    return _MTL_msg_MTL__Bufferp_instanceDescriptorBuffer((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setInstanceDescriptorBuffer(MTL::Buffer* instanceDescriptorBuffer)
{
    _MTL_msg_v_setInstanceDescriptorBuffer__MTL__Bufferp((const void*)this, nullptr, instanceDescriptorBuffer);
}

_MTL_INLINE NS::UInteger MTL::IndirectInstanceAccelerationStructureDescriptor::instanceDescriptorBufferOffset() const
{
    return _MTL_msg_NS__UInteger_instanceDescriptorBufferOffset((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setInstanceDescriptorBufferOffset(NS::UInteger instanceDescriptorBufferOffset)
{
    _MTL_msg_v_setInstanceDescriptorBufferOffset__NS__UInteger((const void*)this, nullptr, instanceDescriptorBufferOffset);
}

_MTL_INLINE NS::UInteger MTL::IndirectInstanceAccelerationStructureDescriptor::instanceDescriptorStride() const
{
    return _MTL_msg_NS__UInteger_instanceDescriptorStride((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setInstanceDescriptorStride(NS::UInteger instanceDescriptorStride)
{
    _MTL_msg_v_setInstanceDescriptorStride__NS__UInteger((const void*)this, nullptr, instanceDescriptorStride);
}

_MTL_INLINE NS::UInteger MTL::IndirectInstanceAccelerationStructureDescriptor::maxInstanceCount() const
{
    return _MTL_msg_NS__UInteger_maxInstanceCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setMaxInstanceCount(NS::UInteger maxInstanceCount)
{
    _MTL_msg_v_setMaxInstanceCount__NS__UInteger((const void*)this, nullptr, maxInstanceCount);
}

_MTL_INLINE MTL::Buffer* MTL::IndirectInstanceAccelerationStructureDescriptor::instanceCountBuffer() const
{
    return _MTL_msg_MTL__Bufferp_instanceCountBuffer((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setInstanceCountBuffer(MTL::Buffer* instanceCountBuffer)
{
    _MTL_msg_v_setInstanceCountBuffer__MTL__Bufferp((const void*)this, nullptr, instanceCountBuffer);
}

_MTL_INLINE NS::UInteger MTL::IndirectInstanceAccelerationStructureDescriptor::instanceCountBufferOffset() const
{
    return _MTL_msg_NS__UInteger_instanceCountBufferOffset((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setInstanceCountBufferOffset(NS::UInteger instanceCountBufferOffset)
{
    _MTL_msg_v_setInstanceCountBufferOffset__NS__UInteger((const void*)this, nullptr, instanceCountBufferOffset);
}

_MTL_INLINE MTL::AccelerationStructureInstanceDescriptorType MTL::IndirectInstanceAccelerationStructureDescriptor::instanceDescriptorType() const
{
    return _MTL_msg_MTL__AccelerationStructureInstanceDescriptorType_instanceDescriptorType((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType)
{
    _MTL_msg_v_setInstanceDescriptorType__MTL__AccelerationStructureInstanceDescriptorType((const void*)this, nullptr, instanceDescriptorType);
}

_MTL_INLINE MTL::Buffer* MTL::IndirectInstanceAccelerationStructureDescriptor::motionTransformBuffer() const
{
    return _MTL_msg_MTL__Bufferp_motionTransformBuffer((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setMotionTransformBuffer(MTL::Buffer* motionTransformBuffer)
{
    _MTL_msg_v_setMotionTransformBuffer__MTL__Bufferp((const void*)this, nullptr, motionTransformBuffer);
}

_MTL_INLINE NS::UInteger MTL::IndirectInstanceAccelerationStructureDescriptor::motionTransformBufferOffset() const
{
    return _MTL_msg_NS__UInteger_motionTransformBufferOffset((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setMotionTransformBufferOffset(NS::UInteger motionTransformBufferOffset)
{
    _MTL_msg_v_setMotionTransformBufferOffset__NS__UInteger((const void*)this, nullptr, motionTransformBufferOffset);
}

_MTL_INLINE NS::UInteger MTL::IndirectInstanceAccelerationStructureDescriptor::maxMotionTransformCount() const
{
    return _MTL_msg_NS__UInteger_maxMotionTransformCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setMaxMotionTransformCount(NS::UInteger maxMotionTransformCount)
{
    _MTL_msg_v_setMaxMotionTransformCount__NS__UInteger((const void*)this, nullptr, maxMotionTransformCount);
}

_MTL_INLINE MTL::Buffer* MTL::IndirectInstanceAccelerationStructureDescriptor::motionTransformCountBuffer() const
{
    return _MTL_msg_MTL__Bufferp_motionTransformCountBuffer((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setMotionTransformCountBuffer(MTL::Buffer* motionTransformCountBuffer)
{
    _MTL_msg_v_setMotionTransformCountBuffer__MTL__Bufferp((const void*)this, nullptr, motionTransformCountBuffer);
}

_MTL_INLINE NS::UInteger MTL::IndirectInstanceAccelerationStructureDescriptor::motionTransformCountBufferOffset() const
{
    return _MTL_msg_NS__UInteger_motionTransformCountBufferOffset((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setMotionTransformCountBufferOffset(NS::UInteger motionTransformCountBufferOffset)
{
    _MTL_msg_v_setMotionTransformCountBufferOffset__NS__UInteger((const void*)this, nullptr, motionTransformCountBufferOffset);
}

_MTL_INLINE MTL::MatrixLayout MTL::IndirectInstanceAccelerationStructureDescriptor::instanceTransformationMatrixLayout() const
{
    return _MTL_msg_MTL__MatrixLayout_instanceTransformationMatrixLayout((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setInstanceTransformationMatrixLayout(MTL::MatrixLayout instanceTransformationMatrixLayout)
{
    _MTL_msg_v_setInstanceTransformationMatrixLayout__MTL__MatrixLayout((const void*)this, nullptr, instanceTransformationMatrixLayout);
}

_MTL_INLINE MTL::TransformType MTL::IndirectInstanceAccelerationStructureDescriptor::motionTransformType() const
{
    return _MTL_msg_MTL__TransformType_motionTransformType((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setMotionTransformType(MTL::TransformType motionTransformType)
{
    _MTL_msg_v_setMotionTransformType__MTL__TransformType((const void*)this, nullptr, motionTransformType);
}

_MTL_INLINE NS::UInteger MTL::IndirectInstanceAccelerationStructureDescriptor::motionTransformStride() const
{
    return _MTL_msg_NS__UInteger_motionTransformStride((const void*)this, nullptr);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setMotionTransformStride(NS::UInteger motionTransformStride)
{
    _MTL_msg_v_setMotionTransformStride__NS__UInteger((const void*)this, nullptr, motionTransformStride);
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructure::size() const
{
    return _MTL_msg_NS__UInteger_size((const void*)this, nullptr);
}

_MTL_INLINE MTL::ResourceID MTL::AccelerationStructure::gpuResourceID() const
{
    return _MTL_msg_MTL__ResourceID_gpuResourceID((const void*)this, nullptr);
}
