#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTLAccelerationStructure.hpp"

namespace MTL {
    enum AccelerationStructureInstanceDescriptorType : NS::UInteger;
    enum AttributeFormat : NS::UInteger;
    enum CurveBasis : NS::Integer;
    enum CurveEndCaps : NS::Integer;
    enum CurveType : NS::Integer;
    enum IndexType : NS::UInteger;
    enum MatrixLayout : NS::Integer;
    enum MotionBorderMode : uint32_t;
    enum TransformType : NS::Integer;
}
namespace NS {
    class Array;
    class String;
}

namespace MTL4
{

class AccelerationStructureDescriptor;
class AccelerationStructureGeometryDescriptor;
class PrimitiveAccelerationStructureDescriptor;
class AccelerationStructureTriangleGeometryDescriptor;
class AccelerationStructureBoundingBoxGeometryDescriptor;
class AccelerationStructureMotionTriangleGeometryDescriptor;
class AccelerationStructureMotionBoundingBoxGeometryDescriptor;
class AccelerationStructureCurveGeometryDescriptor;
class AccelerationStructureMotionCurveGeometryDescriptor;
class InstanceAccelerationStructureDescriptor;
class IndirectInstanceAccelerationStructureDescriptor;

class AccelerationStructureDescriptor : public NS::Referencing<AccelerationStructureDescriptor, MTL::AccelerationStructureDescriptor>
{
public:
    static AccelerationStructureDescriptor* alloc();
    AccelerationStructureDescriptor*        init() const;

};

class AccelerationStructureGeometryDescriptor : public NS::Copying<AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureGeometryDescriptor* alloc();
    AccelerationStructureGeometryDescriptor*        init() const;

    bool              allowDuplicateIntersectionFunctionInvocation() const;
    NS::UInteger      intersectionFunctionTableOffset() const;
    NS::String*       label() const;
    bool              opaque() const;
    MTL4::BufferRange primitiveDataBuffer() const;
    NS::UInteger      primitiveDataElementSize() const;
    NS::UInteger      primitiveDataStride() const;
    void              setAllowDuplicateIntersectionFunctionInvocation(bool allowDuplicateIntersectionFunctionInvocation);
    void              setIntersectionFunctionTableOffset(NS::UInteger intersectionFunctionTableOffset);
    void              setLabel(NS::String* label);
    void              setOpaque(bool opaque);
    void              setPrimitiveDataBuffer(MTL4::BufferRange primitiveDataBuffer);
    void              setPrimitiveDataElementSize(NS::UInteger primitiveDataElementSize);
    void              setPrimitiveDataStride(NS::UInteger primitiveDataStride);

};

class PrimitiveAccelerationStructureDescriptor : public NS::Referencing<PrimitiveAccelerationStructureDescriptor, MTL4::AccelerationStructureDescriptor>
{
public:
    static PrimitiveAccelerationStructureDescriptor* alloc();
    PrimitiveAccelerationStructureDescriptor*        init() const;

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

class AccelerationStructureTriangleGeometryDescriptor : public NS::Referencing<AccelerationStructureTriangleGeometryDescriptor, MTL4::AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureTriangleGeometryDescriptor* alloc();
    AccelerationStructureTriangleGeometryDescriptor*        init() const;

    MTL4::BufferRange    indexBuffer() const;
    MTL::IndexType       indexType() const;
    void                 setIndexBuffer(MTL4::BufferRange indexBuffer);
    void                 setIndexType(MTL::IndexType indexType);
    void                 setTransformationMatrixBuffer(MTL4::BufferRange transformationMatrixBuffer);
    void                 setTransformationMatrixLayout(MTL::MatrixLayout transformationMatrixLayout);
    void                 setTriangleCount(NS::UInteger triangleCount);
    void                 setVertexBuffer(MTL4::BufferRange vertexBuffer);
    void                 setVertexFormat(MTL::AttributeFormat vertexFormat);
    void                 setVertexStride(NS::UInteger vertexStride);
    MTL4::BufferRange    transformationMatrixBuffer() const;
    MTL::MatrixLayout    transformationMatrixLayout() const;
    NS::UInteger         triangleCount() const;
    MTL4::BufferRange    vertexBuffer() const;
    MTL::AttributeFormat vertexFormat() const;
    NS::UInteger         vertexStride() const;

};

class AccelerationStructureBoundingBoxGeometryDescriptor : public NS::Referencing<AccelerationStructureBoundingBoxGeometryDescriptor, MTL4::AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureBoundingBoxGeometryDescriptor* alloc();
    AccelerationStructureBoundingBoxGeometryDescriptor*        init() const;

    MTL4::BufferRange boundingBoxBuffer() const;
    NS::UInteger      boundingBoxCount() const;
    NS::UInteger      boundingBoxStride() const;
    void              setBoundingBoxBuffer(MTL4::BufferRange boundingBoxBuffer);
    void              setBoundingBoxCount(NS::UInteger boundingBoxCount);
    void              setBoundingBoxStride(NS::UInteger boundingBoxStride);

};

class AccelerationStructureMotionTriangleGeometryDescriptor : public NS::Referencing<AccelerationStructureMotionTriangleGeometryDescriptor, MTL4::AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureMotionTriangleGeometryDescriptor* alloc();
    AccelerationStructureMotionTriangleGeometryDescriptor*        init() const;

    MTL4::BufferRange    indexBuffer() const;
    MTL::IndexType       indexType() const;
    void                 setIndexBuffer(MTL4::BufferRange indexBuffer);
    void                 setIndexType(MTL::IndexType indexType);
    void                 setTransformationMatrixBuffer(MTL4::BufferRange transformationMatrixBuffer);
    void                 setTransformationMatrixLayout(MTL::MatrixLayout transformationMatrixLayout);
    void                 setTriangleCount(NS::UInteger triangleCount);
    void                 setVertexBuffers(MTL4::BufferRange vertexBuffers);
    void                 setVertexFormat(MTL::AttributeFormat vertexFormat);
    void                 setVertexStride(NS::UInteger vertexStride);
    MTL4::BufferRange    transformationMatrixBuffer() const;
    MTL::MatrixLayout    transformationMatrixLayout() const;
    NS::UInteger         triangleCount() const;
    MTL4::BufferRange    vertexBuffers() const;
    MTL::AttributeFormat vertexFormat() const;
    NS::UInteger         vertexStride() const;

};

class AccelerationStructureMotionBoundingBoxGeometryDescriptor : public NS::Referencing<AccelerationStructureMotionBoundingBoxGeometryDescriptor, MTL4::AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureMotionBoundingBoxGeometryDescriptor* alloc();
    AccelerationStructureMotionBoundingBoxGeometryDescriptor*        init() const;

    MTL4::BufferRange boundingBoxBuffers() const;
    NS::UInteger      boundingBoxCount() const;
    NS::UInteger      boundingBoxStride() const;
    void              setBoundingBoxBuffers(MTL4::BufferRange boundingBoxBuffers);
    void              setBoundingBoxCount(NS::UInteger boundingBoxCount);
    void              setBoundingBoxStride(NS::UInteger boundingBoxStride);

};

class AccelerationStructureCurveGeometryDescriptor : public NS::Referencing<AccelerationStructureCurveGeometryDescriptor, MTL4::AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureCurveGeometryDescriptor* alloc();
    AccelerationStructureCurveGeometryDescriptor*        init() const;

    MTL4::BufferRange    controlPointBuffer() const;
    NS::UInteger         controlPointCount() const;
    MTL::AttributeFormat controlPointFormat() const;
    NS::UInteger         controlPointStride() const;
    MTL::CurveBasis      curveBasis() const;
    MTL::CurveEndCaps    curveEndCaps() const;
    MTL::CurveType       curveType() const;
    MTL4::BufferRange    indexBuffer() const;
    MTL::IndexType       indexType() const;
    MTL4::BufferRange    radiusBuffer() const;
    MTL::AttributeFormat radiusFormat() const;
    NS::UInteger         radiusStride() const;
    NS::UInteger         segmentControlPointCount() const;
    NS::UInteger         segmentCount() const;
    void                 setControlPointBuffer(MTL4::BufferRange controlPointBuffer);
    void                 setControlPointCount(NS::UInteger controlPointCount);
    void                 setControlPointFormat(MTL::AttributeFormat controlPointFormat);
    void                 setControlPointStride(NS::UInteger controlPointStride);
    void                 setCurveBasis(MTL::CurveBasis curveBasis);
    void                 setCurveEndCaps(MTL::CurveEndCaps curveEndCaps);
    void                 setCurveType(MTL::CurveType curveType);
    void                 setIndexBuffer(MTL4::BufferRange indexBuffer);
    void                 setIndexType(MTL::IndexType indexType);
    void                 setRadiusBuffer(MTL4::BufferRange radiusBuffer);
    void                 setRadiusFormat(MTL::AttributeFormat radiusFormat);
    void                 setRadiusStride(NS::UInteger radiusStride);
    void                 setSegmentControlPointCount(NS::UInteger segmentControlPointCount);
    void                 setSegmentCount(NS::UInteger segmentCount);

};

class AccelerationStructureMotionCurveGeometryDescriptor : public NS::Referencing<AccelerationStructureMotionCurveGeometryDescriptor, MTL4::AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureMotionCurveGeometryDescriptor* alloc();
    AccelerationStructureMotionCurveGeometryDescriptor*        init() const;

    MTL4::BufferRange    controlPointBuffers() const;
    NS::UInteger         controlPointCount() const;
    MTL::AttributeFormat controlPointFormat() const;
    NS::UInteger         controlPointStride() const;
    MTL::CurveBasis      curveBasis() const;
    MTL::CurveEndCaps    curveEndCaps() const;
    MTL::CurveType       curveType() const;
    MTL4::BufferRange    indexBuffer() const;
    MTL::IndexType       indexType() const;
    MTL4::BufferRange    radiusBuffers() const;
    MTL::AttributeFormat radiusFormat() const;
    NS::UInteger         radiusStride() const;
    NS::UInteger         segmentControlPointCount() const;
    NS::UInteger         segmentCount() const;
    void                 setControlPointBuffers(MTL4::BufferRange controlPointBuffers);
    void                 setControlPointCount(NS::UInteger controlPointCount);
    void                 setControlPointFormat(MTL::AttributeFormat controlPointFormat);
    void                 setControlPointStride(NS::UInteger controlPointStride);
    void                 setCurveBasis(MTL::CurveBasis curveBasis);
    void                 setCurveEndCaps(MTL::CurveEndCaps curveEndCaps);
    void                 setCurveType(MTL::CurveType curveType);
    void                 setIndexBuffer(MTL4::BufferRange indexBuffer);
    void                 setIndexType(MTL::IndexType indexType);
    void                 setRadiusBuffers(MTL4::BufferRange radiusBuffers);
    void                 setRadiusFormat(MTL::AttributeFormat radiusFormat);
    void                 setRadiusStride(NS::UInteger radiusStride);
    void                 setSegmentControlPointCount(NS::UInteger segmentControlPointCount);
    void                 setSegmentCount(NS::UInteger segmentCount);

};

class InstanceAccelerationStructureDescriptor : public NS::Referencing<InstanceAccelerationStructureDescriptor, MTL4::AccelerationStructureDescriptor>
{
public:
    static InstanceAccelerationStructureDescriptor* alloc();
    InstanceAccelerationStructureDescriptor*        init() const;

    NS::UInteger                                     instanceCount() const;
    MTL4::BufferRange                                instanceDescriptorBuffer() const;
    NS::UInteger                                     instanceDescriptorStride() const;
    MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType() const;
    MTL::MatrixLayout                                instanceTransformationMatrixLayout() const;
    MTL4::BufferRange                                motionTransformBuffer() const;
    NS::UInteger                                     motionTransformCount() const;
    NS::UInteger                                     motionTransformStride() const;
    MTL::TransformType                               motionTransformType() const;
    void                                             setInstanceCount(NS::UInteger instanceCount);
    void                                             setInstanceDescriptorBuffer(MTL4::BufferRange instanceDescriptorBuffer);
    void                                             setInstanceDescriptorStride(NS::UInteger instanceDescriptorStride);
    void                                             setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType);
    void                                             setInstanceTransformationMatrixLayout(MTL::MatrixLayout instanceTransformationMatrixLayout);
    void                                             setMotionTransformBuffer(MTL4::BufferRange motionTransformBuffer);
    void                                             setMotionTransformCount(NS::UInteger motionTransformCount);
    void                                             setMotionTransformStride(NS::UInteger motionTransformStride);
    void                                             setMotionTransformType(MTL::TransformType motionTransformType);

};

class IndirectInstanceAccelerationStructureDescriptor : public NS::Referencing<IndirectInstanceAccelerationStructureDescriptor, MTL4::AccelerationStructureDescriptor>
{
public:
    static IndirectInstanceAccelerationStructureDescriptor* alloc();
    IndirectInstanceAccelerationStructureDescriptor*        init() const;

    MTL4::BufferRange                                instanceCountBuffer() const;
    MTL4::BufferRange                                instanceDescriptorBuffer() const;
    NS::UInteger                                     instanceDescriptorStride() const;
    MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType() const;
    MTL::MatrixLayout                                instanceTransformationMatrixLayout() const;
    NS::UInteger                                     maxInstanceCount() const;
    NS::UInteger                                     maxMotionTransformCount() const;
    MTL4::BufferRange                                motionTransformBuffer() const;
    MTL4::BufferRange                                motionTransformCountBuffer() const;
    NS::UInteger                                     motionTransformStride() const;
    MTL::TransformType                               motionTransformType() const;
    void                                             setInstanceCountBuffer(MTL4::BufferRange instanceCountBuffer);
    void                                             setInstanceDescriptorBuffer(MTL4::BufferRange instanceDescriptorBuffer);
    void                                             setInstanceDescriptorStride(NS::UInteger instanceDescriptorStride);
    void                                             setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType);
    void                                             setInstanceTransformationMatrixLayout(MTL::MatrixLayout instanceTransformationMatrixLayout);
    void                                             setMaxInstanceCount(NS::UInteger maxInstanceCount);
    void                                             setMaxMotionTransformCount(NS::UInteger maxMotionTransformCount);
    void                                             setMotionTransformBuffer(MTL4::BufferRange motionTransformBuffer);
    void                                             setMotionTransformCountBuffer(MTL4::BufferRange motionTransformCountBuffer);
    void                                             setMotionTransformStride(NS::UInteger motionTransformStride);
    void                                             setMotionTransformType(MTL::TransformType motionTransformType);

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4AccelerationStructureDescriptor;
extern "C" void *OBJC_CLASS_$_MTL4AccelerationStructureGeometryDescriptor;
extern "C" void *OBJC_CLASS_$_MTL4PrimitiveAccelerationStructureDescriptor;
extern "C" void *OBJC_CLASS_$_MTL4AccelerationStructureTriangleGeometryDescriptor;
extern "C" void *OBJC_CLASS_$_MTL4AccelerationStructureBoundingBoxGeometryDescriptor;
extern "C" void *OBJC_CLASS_$_MTL4AccelerationStructureMotionTriangleGeometryDescriptor;
extern "C" void *OBJC_CLASS_$_MTL4AccelerationStructureMotionBoundingBoxGeometryDescriptor;
extern "C" void *OBJC_CLASS_$_MTL4AccelerationStructureCurveGeometryDescriptor;
extern "C" void *OBJC_CLASS_$_MTL4AccelerationStructureMotionCurveGeometryDescriptor;
extern "C" void *OBJC_CLASS_$_MTL4InstanceAccelerationStructureDescriptor;
extern "C" void *OBJC_CLASS_$_MTL4IndirectInstanceAccelerationStructureDescriptor;

_MTL4_INLINE MTL4::AccelerationStructureDescriptor* MTL4::AccelerationStructureDescriptor::alloc()
{
    return _MTL4_msg_MTL4__AccelerationStructureDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4AccelerationStructureDescriptor, nullptr);
}

_MTL4_INLINE MTL4::AccelerationStructureDescriptor* MTL4::AccelerationStructureDescriptor::init() const
{
    return _MTL4_msg_MTL4__AccelerationStructureDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::AccelerationStructureGeometryDescriptor* MTL4::AccelerationStructureGeometryDescriptor::alloc()
{
    return _MTL4_msg_MTL4__AccelerationStructureGeometryDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4AccelerationStructureGeometryDescriptor, nullptr);
}

_MTL4_INLINE MTL4::AccelerationStructureGeometryDescriptor* MTL4::AccelerationStructureGeometryDescriptor::init() const
{
    return _MTL4_msg_MTL4__AccelerationStructureGeometryDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE NS::UInteger MTL4::AccelerationStructureGeometryDescriptor::intersectionFunctionTableOffset() const
{
    return _MTL4_msg_NS__UInteger_intersectionFunctionTableOffset((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureGeometryDescriptor::setIntersectionFunctionTableOffset(NS::UInteger intersectionFunctionTableOffset)
{
    _MTL4_msg_v_setIntersectionFunctionTableOffset__NS__UInteger((const void*)this, nullptr, intersectionFunctionTableOffset);
}

_MTL4_INLINE bool MTL4::AccelerationStructureGeometryDescriptor::opaque() const
{
    return _MTL4_msg_bool_opaque((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureGeometryDescriptor::setOpaque(bool opaque)
{
    _MTL4_msg_v_setOpaque__bool((const void*)this, nullptr, opaque);
}

_MTL4_INLINE bool MTL4::AccelerationStructureGeometryDescriptor::allowDuplicateIntersectionFunctionInvocation() const
{
    return _MTL4_msg_bool_allowDuplicateIntersectionFunctionInvocation((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureGeometryDescriptor::setAllowDuplicateIntersectionFunctionInvocation(bool allowDuplicateIntersectionFunctionInvocation)
{
    _MTL4_msg_v_setAllowDuplicateIntersectionFunctionInvocation__bool((const void*)this, nullptr, allowDuplicateIntersectionFunctionInvocation);
}

_MTL4_INLINE NS::String* MTL4::AccelerationStructureGeometryDescriptor::label() const
{
    return _MTL4_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureGeometryDescriptor::setLabel(NS::String* label)
{
    _MTL4_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL4_INLINE MTL4::BufferRange MTL4::AccelerationStructureGeometryDescriptor::primitiveDataBuffer() const
{
    return _MTL4_msg_MTL4__BufferRange_primitiveDataBuffer((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureGeometryDescriptor::setPrimitiveDataBuffer(MTL4::BufferRange primitiveDataBuffer)
{
    _MTL4_msg_v_setPrimitiveDataBuffer__MTL4__BufferRange((const void*)this, nullptr, primitiveDataBuffer);
}

_MTL4_INLINE NS::UInteger MTL4::AccelerationStructureGeometryDescriptor::primitiveDataStride() const
{
    return _MTL4_msg_NS__UInteger_primitiveDataStride((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureGeometryDescriptor::setPrimitiveDataStride(NS::UInteger primitiveDataStride)
{
    _MTL4_msg_v_setPrimitiveDataStride__NS__UInteger((const void*)this, nullptr, primitiveDataStride);
}

_MTL4_INLINE NS::UInteger MTL4::AccelerationStructureGeometryDescriptor::primitiveDataElementSize() const
{
    return _MTL4_msg_NS__UInteger_primitiveDataElementSize((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureGeometryDescriptor::setPrimitiveDataElementSize(NS::UInteger primitiveDataElementSize)
{
    _MTL4_msg_v_setPrimitiveDataElementSize__NS__UInteger((const void*)this, nullptr, primitiveDataElementSize);
}

_MTL4_INLINE MTL4::PrimitiveAccelerationStructureDescriptor* MTL4::PrimitiveAccelerationStructureDescriptor::alloc()
{
    return _MTL4_msg_MTL4__PrimitiveAccelerationStructureDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4PrimitiveAccelerationStructureDescriptor, nullptr);
}

_MTL4_INLINE MTL4::PrimitiveAccelerationStructureDescriptor* MTL4::PrimitiveAccelerationStructureDescriptor::init() const
{
    return _MTL4_msg_MTL4__PrimitiveAccelerationStructureDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE NS::Array* MTL4::PrimitiveAccelerationStructureDescriptor::geometryDescriptors() const
{
    return _MTL4_msg_NS__Arrayp_geometryDescriptors((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::PrimitiveAccelerationStructureDescriptor::setGeometryDescriptors(NS::Array* geometryDescriptors)
{
    _MTL4_msg_v_setGeometryDescriptors__NS__Arrayp((const void*)this, nullptr, geometryDescriptors);
}

_MTL4_INLINE MTL::MotionBorderMode MTL4::PrimitiveAccelerationStructureDescriptor::motionStartBorderMode() const
{
    return _MTL4_msg_MTL__MotionBorderMode_motionStartBorderMode((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::PrimitiveAccelerationStructureDescriptor::setMotionStartBorderMode(MTL::MotionBorderMode motionStartBorderMode)
{
    _MTL4_msg_v_setMotionStartBorderMode__MTL__MotionBorderMode((const void*)this, nullptr, motionStartBorderMode);
}

_MTL4_INLINE MTL::MotionBorderMode MTL4::PrimitiveAccelerationStructureDescriptor::motionEndBorderMode() const
{
    return _MTL4_msg_MTL__MotionBorderMode_motionEndBorderMode((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::PrimitiveAccelerationStructureDescriptor::setMotionEndBorderMode(MTL::MotionBorderMode motionEndBorderMode)
{
    _MTL4_msg_v_setMotionEndBorderMode__MTL__MotionBorderMode((const void*)this, nullptr, motionEndBorderMode);
}

_MTL4_INLINE float MTL4::PrimitiveAccelerationStructureDescriptor::motionStartTime() const
{
    return _MTL4_msg_float_motionStartTime((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::PrimitiveAccelerationStructureDescriptor::setMotionStartTime(float motionStartTime)
{
    _MTL4_msg_v_setMotionStartTime__float((const void*)this, nullptr, motionStartTime);
}

_MTL4_INLINE float MTL4::PrimitiveAccelerationStructureDescriptor::motionEndTime() const
{
    return _MTL4_msg_float_motionEndTime((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::PrimitiveAccelerationStructureDescriptor::setMotionEndTime(float motionEndTime)
{
    _MTL4_msg_v_setMotionEndTime__float((const void*)this, nullptr, motionEndTime);
}

_MTL4_INLINE NS::UInteger MTL4::PrimitiveAccelerationStructureDescriptor::motionKeyframeCount() const
{
    return _MTL4_msg_NS__UInteger_motionKeyframeCount((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::PrimitiveAccelerationStructureDescriptor::setMotionKeyframeCount(NS::UInteger motionKeyframeCount)
{
    _MTL4_msg_v_setMotionKeyframeCount__NS__UInteger((const void*)this, nullptr, motionKeyframeCount);
}

_MTL4_INLINE MTL4::AccelerationStructureTriangleGeometryDescriptor* MTL4::AccelerationStructureTriangleGeometryDescriptor::alloc()
{
    return _MTL4_msg_MTL4__AccelerationStructureTriangleGeometryDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4AccelerationStructureTriangleGeometryDescriptor, nullptr);
}

_MTL4_INLINE MTL4::AccelerationStructureTriangleGeometryDescriptor* MTL4::AccelerationStructureTriangleGeometryDescriptor::init() const
{
    return _MTL4_msg_MTL4__AccelerationStructureTriangleGeometryDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::BufferRange MTL4::AccelerationStructureTriangleGeometryDescriptor::vertexBuffer() const
{
    return _MTL4_msg_MTL4__BufferRange_vertexBuffer((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureTriangleGeometryDescriptor::setVertexBuffer(MTL4::BufferRange vertexBuffer)
{
    _MTL4_msg_v_setVertexBuffer__MTL4__BufferRange((const void*)this, nullptr, vertexBuffer);
}

_MTL4_INLINE MTL::AttributeFormat MTL4::AccelerationStructureTriangleGeometryDescriptor::vertexFormat() const
{
    return _MTL4_msg_MTL__AttributeFormat_vertexFormat((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureTriangleGeometryDescriptor::setVertexFormat(MTL::AttributeFormat vertexFormat)
{
    _MTL4_msg_v_setVertexFormat__MTL__AttributeFormat((const void*)this, nullptr, vertexFormat);
}

_MTL4_INLINE NS::UInteger MTL4::AccelerationStructureTriangleGeometryDescriptor::vertexStride() const
{
    return _MTL4_msg_NS__UInteger_vertexStride((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureTriangleGeometryDescriptor::setVertexStride(NS::UInteger vertexStride)
{
    _MTL4_msg_v_setVertexStride__NS__UInteger((const void*)this, nullptr, vertexStride);
}

_MTL4_INLINE MTL4::BufferRange MTL4::AccelerationStructureTriangleGeometryDescriptor::indexBuffer() const
{
    return _MTL4_msg_MTL4__BufferRange_indexBuffer((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureTriangleGeometryDescriptor::setIndexBuffer(MTL4::BufferRange indexBuffer)
{
    _MTL4_msg_v_setIndexBuffer__MTL4__BufferRange((const void*)this, nullptr, indexBuffer);
}

_MTL4_INLINE MTL::IndexType MTL4::AccelerationStructureTriangleGeometryDescriptor::indexType() const
{
    return _MTL4_msg_MTL__IndexType_indexType((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureTriangleGeometryDescriptor::setIndexType(MTL::IndexType indexType)
{
    _MTL4_msg_v_setIndexType__MTL__IndexType((const void*)this, nullptr, indexType);
}

_MTL4_INLINE NS::UInteger MTL4::AccelerationStructureTriangleGeometryDescriptor::triangleCount() const
{
    return _MTL4_msg_NS__UInteger_triangleCount((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureTriangleGeometryDescriptor::setTriangleCount(NS::UInteger triangleCount)
{
    _MTL4_msg_v_setTriangleCount__NS__UInteger((const void*)this, nullptr, triangleCount);
}

_MTL4_INLINE MTL4::BufferRange MTL4::AccelerationStructureTriangleGeometryDescriptor::transformationMatrixBuffer() const
{
    return _MTL4_msg_MTL4__BufferRange_transformationMatrixBuffer((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureTriangleGeometryDescriptor::setTransformationMatrixBuffer(MTL4::BufferRange transformationMatrixBuffer)
{
    _MTL4_msg_v_setTransformationMatrixBuffer__MTL4__BufferRange((const void*)this, nullptr, transformationMatrixBuffer);
}

_MTL4_INLINE MTL::MatrixLayout MTL4::AccelerationStructureTriangleGeometryDescriptor::transformationMatrixLayout() const
{
    return _MTL4_msg_MTL__MatrixLayout_transformationMatrixLayout((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureTriangleGeometryDescriptor::setTransformationMatrixLayout(MTL::MatrixLayout transformationMatrixLayout)
{
    _MTL4_msg_v_setTransformationMatrixLayout__MTL__MatrixLayout((const void*)this, nullptr, transformationMatrixLayout);
}

_MTL4_INLINE MTL4::AccelerationStructureBoundingBoxGeometryDescriptor* MTL4::AccelerationStructureBoundingBoxGeometryDescriptor::alloc()
{
    return _MTL4_msg_MTL4__AccelerationStructureBoundingBoxGeometryDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4AccelerationStructureBoundingBoxGeometryDescriptor, nullptr);
}

_MTL4_INLINE MTL4::AccelerationStructureBoundingBoxGeometryDescriptor* MTL4::AccelerationStructureBoundingBoxGeometryDescriptor::init() const
{
    return _MTL4_msg_MTL4__AccelerationStructureBoundingBoxGeometryDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::BufferRange MTL4::AccelerationStructureBoundingBoxGeometryDescriptor::boundingBoxBuffer() const
{
    return _MTL4_msg_MTL4__BufferRange_boundingBoxBuffer((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureBoundingBoxGeometryDescriptor::setBoundingBoxBuffer(MTL4::BufferRange boundingBoxBuffer)
{
    _MTL4_msg_v_setBoundingBoxBuffer__MTL4__BufferRange((const void*)this, nullptr, boundingBoxBuffer);
}

_MTL4_INLINE NS::UInteger MTL4::AccelerationStructureBoundingBoxGeometryDescriptor::boundingBoxStride() const
{
    return _MTL4_msg_NS__UInteger_boundingBoxStride((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureBoundingBoxGeometryDescriptor::setBoundingBoxStride(NS::UInteger boundingBoxStride)
{
    _MTL4_msg_v_setBoundingBoxStride__NS__UInteger((const void*)this, nullptr, boundingBoxStride);
}

_MTL4_INLINE NS::UInteger MTL4::AccelerationStructureBoundingBoxGeometryDescriptor::boundingBoxCount() const
{
    return _MTL4_msg_NS__UInteger_boundingBoxCount((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureBoundingBoxGeometryDescriptor::setBoundingBoxCount(NS::UInteger boundingBoxCount)
{
    _MTL4_msg_v_setBoundingBoxCount__NS__UInteger((const void*)this, nullptr, boundingBoxCount);
}

_MTL4_INLINE MTL4::AccelerationStructureMotionTriangleGeometryDescriptor* MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::alloc()
{
    return _MTL4_msg_MTL4__AccelerationStructureMotionTriangleGeometryDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4AccelerationStructureMotionTriangleGeometryDescriptor, nullptr);
}

_MTL4_INLINE MTL4::AccelerationStructureMotionTriangleGeometryDescriptor* MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::init() const
{
    return _MTL4_msg_MTL4__AccelerationStructureMotionTriangleGeometryDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::BufferRange MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::vertexBuffers() const
{
    return _MTL4_msg_MTL4__BufferRange_vertexBuffers((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::setVertexBuffers(MTL4::BufferRange vertexBuffers)
{
    _MTL4_msg_v_setVertexBuffers__MTL4__BufferRange((const void*)this, nullptr, vertexBuffers);
}

_MTL4_INLINE MTL::AttributeFormat MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::vertexFormat() const
{
    return _MTL4_msg_MTL__AttributeFormat_vertexFormat((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::setVertexFormat(MTL::AttributeFormat vertexFormat)
{
    _MTL4_msg_v_setVertexFormat__MTL__AttributeFormat((const void*)this, nullptr, vertexFormat);
}

_MTL4_INLINE NS::UInteger MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::vertexStride() const
{
    return _MTL4_msg_NS__UInteger_vertexStride((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::setVertexStride(NS::UInteger vertexStride)
{
    _MTL4_msg_v_setVertexStride__NS__UInteger((const void*)this, nullptr, vertexStride);
}

_MTL4_INLINE MTL4::BufferRange MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::indexBuffer() const
{
    return _MTL4_msg_MTL4__BufferRange_indexBuffer((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::setIndexBuffer(MTL4::BufferRange indexBuffer)
{
    _MTL4_msg_v_setIndexBuffer__MTL4__BufferRange((const void*)this, nullptr, indexBuffer);
}

_MTL4_INLINE MTL::IndexType MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::indexType() const
{
    return _MTL4_msg_MTL__IndexType_indexType((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::setIndexType(MTL::IndexType indexType)
{
    _MTL4_msg_v_setIndexType__MTL__IndexType((const void*)this, nullptr, indexType);
}

_MTL4_INLINE NS::UInteger MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::triangleCount() const
{
    return _MTL4_msg_NS__UInteger_triangleCount((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::setTriangleCount(NS::UInteger triangleCount)
{
    _MTL4_msg_v_setTriangleCount__NS__UInteger((const void*)this, nullptr, triangleCount);
}

_MTL4_INLINE MTL4::BufferRange MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::transformationMatrixBuffer() const
{
    return _MTL4_msg_MTL4__BufferRange_transformationMatrixBuffer((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::setTransformationMatrixBuffer(MTL4::BufferRange transformationMatrixBuffer)
{
    _MTL4_msg_v_setTransformationMatrixBuffer__MTL4__BufferRange((const void*)this, nullptr, transformationMatrixBuffer);
}

_MTL4_INLINE MTL::MatrixLayout MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::transformationMatrixLayout() const
{
    return _MTL4_msg_MTL__MatrixLayout_transformationMatrixLayout((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::setTransformationMatrixLayout(MTL::MatrixLayout transformationMatrixLayout)
{
    _MTL4_msg_v_setTransformationMatrixLayout__MTL__MatrixLayout((const void*)this, nullptr, transformationMatrixLayout);
}

_MTL4_INLINE MTL4::AccelerationStructureMotionBoundingBoxGeometryDescriptor* MTL4::AccelerationStructureMotionBoundingBoxGeometryDescriptor::alloc()
{
    return _MTL4_msg_MTL4__AccelerationStructureMotionBoundingBoxGeometryDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4AccelerationStructureMotionBoundingBoxGeometryDescriptor, nullptr);
}

_MTL4_INLINE MTL4::AccelerationStructureMotionBoundingBoxGeometryDescriptor* MTL4::AccelerationStructureMotionBoundingBoxGeometryDescriptor::init() const
{
    return _MTL4_msg_MTL4__AccelerationStructureMotionBoundingBoxGeometryDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::BufferRange MTL4::AccelerationStructureMotionBoundingBoxGeometryDescriptor::boundingBoxBuffers() const
{
    return _MTL4_msg_MTL4__BufferRange_boundingBoxBuffers((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionBoundingBoxGeometryDescriptor::setBoundingBoxBuffers(MTL4::BufferRange boundingBoxBuffers)
{
    _MTL4_msg_v_setBoundingBoxBuffers__MTL4__BufferRange((const void*)this, nullptr, boundingBoxBuffers);
}

_MTL4_INLINE NS::UInteger MTL4::AccelerationStructureMotionBoundingBoxGeometryDescriptor::boundingBoxStride() const
{
    return _MTL4_msg_NS__UInteger_boundingBoxStride((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionBoundingBoxGeometryDescriptor::setBoundingBoxStride(NS::UInteger boundingBoxStride)
{
    _MTL4_msg_v_setBoundingBoxStride__NS__UInteger((const void*)this, nullptr, boundingBoxStride);
}

_MTL4_INLINE NS::UInteger MTL4::AccelerationStructureMotionBoundingBoxGeometryDescriptor::boundingBoxCount() const
{
    return _MTL4_msg_NS__UInteger_boundingBoxCount((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionBoundingBoxGeometryDescriptor::setBoundingBoxCount(NS::UInteger boundingBoxCount)
{
    _MTL4_msg_v_setBoundingBoxCount__NS__UInteger((const void*)this, nullptr, boundingBoxCount);
}

_MTL4_INLINE MTL4::AccelerationStructureCurveGeometryDescriptor* MTL4::AccelerationStructureCurveGeometryDescriptor::alloc()
{
    return _MTL4_msg_MTL4__AccelerationStructureCurveGeometryDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4AccelerationStructureCurveGeometryDescriptor, nullptr);
}

_MTL4_INLINE MTL4::AccelerationStructureCurveGeometryDescriptor* MTL4::AccelerationStructureCurveGeometryDescriptor::init() const
{
    return _MTL4_msg_MTL4__AccelerationStructureCurveGeometryDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::BufferRange MTL4::AccelerationStructureCurveGeometryDescriptor::controlPointBuffer() const
{
    return _MTL4_msg_MTL4__BufferRange_controlPointBuffer((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setControlPointBuffer(MTL4::BufferRange controlPointBuffer)
{
    _MTL4_msg_v_setControlPointBuffer__MTL4__BufferRange((const void*)this, nullptr, controlPointBuffer);
}

_MTL4_INLINE NS::UInteger MTL4::AccelerationStructureCurveGeometryDescriptor::controlPointCount() const
{
    return _MTL4_msg_NS__UInteger_controlPointCount((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setControlPointCount(NS::UInteger controlPointCount)
{
    _MTL4_msg_v_setControlPointCount__NS__UInteger((const void*)this, nullptr, controlPointCount);
}

_MTL4_INLINE NS::UInteger MTL4::AccelerationStructureCurveGeometryDescriptor::controlPointStride() const
{
    return _MTL4_msg_NS__UInteger_controlPointStride((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setControlPointStride(NS::UInteger controlPointStride)
{
    _MTL4_msg_v_setControlPointStride__NS__UInteger((const void*)this, nullptr, controlPointStride);
}

_MTL4_INLINE MTL::AttributeFormat MTL4::AccelerationStructureCurveGeometryDescriptor::controlPointFormat() const
{
    return _MTL4_msg_MTL__AttributeFormat_controlPointFormat((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setControlPointFormat(MTL::AttributeFormat controlPointFormat)
{
    _MTL4_msg_v_setControlPointFormat__MTL__AttributeFormat((const void*)this, nullptr, controlPointFormat);
}

_MTL4_INLINE MTL4::BufferRange MTL4::AccelerationStructureCurveGeometryDescriptor::radiusBuffer() const
{
    return _MTL4_msg_MTL4__BufferRange_radiusBuffer((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setRadiusBuffer(MTL4::BufferRange radiusBuffer)
{
    _MTL4_msg_v_setRadiusBuffer__MTL4__BufferRange((const void*)this, nullptr, radiusBuffer);
}

_MTL4_INLINE MTL::AttributeFormat MTL4::AccelerationStructureCurveGeometryDescriptor::radiusFormat() const
{
    return _MTL4_msg_MTL__AttributeFormat_radiusFormat((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setRadiusFormat(MTL::AttributeFormat radiusFormat)
{
    _MTL4_msg_v_setRadiusFormat__MTL__AttributeFormat((const void*)this, nullptr, radiusFormat);
}

_MTL4_INLINE NS::UInteger MTL4::AccelerationStructureCurveGeometryDescriptor::radiusStride() const
{
    return _MTL4_msg_NS__UInteger_radiusStride((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setRadiusStride(NS::UInteger radiusStride)
{
    _MTL4_msg_v_setRadiusStride__NS__UInteger((const void*)this, nullptr, radiusStride);
}

_MTL4_INLINE MTL4::BufferRange MTL4::AccelerationStructureCurveGeometryDescriptor::indexBuffer() const
{
    return _MTL4_msg_MTL4__BufferRange_indexBuffer((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setIndexBuffer(MTL4::BufferRange indexBuffer)
{
    _MTL4_msg_v_setIndexBuffer__MTL4__BufferRange((const void*)this, nullptr, indexBuffer);
}

_MTL4_INLINE MTL::IndexType MTL4::AccelerationStructureCurveGeometryDescriptor::indexType() const
{
    return _MTL4_msg_MTL__IndexType_indexType((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setIndexType(MTL::IndexType indexType)
{
    _MTL4_msg_v_setIndexType__MTL__IndexType((const void*)this, nullptr, indexType);
}

_MTL4_INLINE NS::UInteger MTL4::AccelerationStructureCurveGeometryDescriptor::segmentCount() const
{
    return _MTL4_msg_NS__UInteger_segmentCount((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setSegmentCount(NS::UInteger segmentCount)
{
    _MTL4_msg_v_setSegmentCount__NS__UInteger((const void*)this, nullptr, segmentCount);
}

_MTL4_INLINE NS::UInteger MTL4::AccelerationStructureCurveGeometryDescriptor::segmentControlPointCount() const
{
    return _MTL4_msg_NS__UInteger_segmentControlPointCount((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setSegmentControlPointCount(NS::UInteger segmentControlPointCount)
{
    _MTL4_msg_v_setSegmentControlPointCount__NS__UInteger((const void*)this, nullptr, segmentControlPointCount);
}

_MTL4_INLINE MTL::CurveType MTL4::AccelerationStructureCurveGeometryDescriptor::curveType() const
{
    return _MTL4_msg_MTL__CurveType_curveType((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setCurveType(MTL::CurveType curveType)
{
    _MTL4_msg_v_setCurveType__MTL__CurveType((const void*)this, nullptr, curveType);
}

_MTL4_INLINE MTL::CurveBasis MTL4::AccelerationStructureCurveGeometryDescriptor::curveBasis() const
{
    return _MTL4_msg_MTL__CurveBasis_curveBasis((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setCurveBasis(MTL::CurveBasis curveBasis)
{
    _MTL4_msg_v_setCurveBasis__MTL__CurveBasis((const void*)this, nullptr, curveBasis);
}

_MTL4_INLINE MTL::CurveEndCaps MTL4::AccelerationStructureCurveGeometryDescriptor::curveEndCaps() const
{
    return _MTL4_msg_MTL__CurveEndCaps_curveEndCaps((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setCurveEndCaps(MTL::CurveEndCaps curveEndCaps)
{
    _MTL4_msg_v_setCurveEndCaps__MTL__CurveEndCaps((const void*)this, nullptr, curveEndCaps);
}

_MTL4_INLINE MTL4::AccelerationStructureMotionCurveGeometryDescriptor* MTL4::AccelerationStructureMotionCurveGeometryDescriptor::alloc()
{
    return _MTL4_msg_MTL4__AccelerationStructureMotionCurveGeometryDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4AccelerationStructureMotionCurveGeometryDescriptor, nullptr);
}

_MTL4_INLINE MTL4::AccelerationStructureMotionCurveGeometryDescriptor* MTL4::AccelerationStructureMotionCurveGeometryDescriptor::init() const
{
    return _MTL4_msg_MTL4__AccelerationStructureMotionCurveGeometryDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::BufferRange MTL4::AccelerationStructureMotionCurveGeometryDescriptor::controlPointBuffers() const
{
    return _MTL4_msg_MTL4__BufferRange_controlPointBuffers((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setControlPointBuffers(MTL4::BufferRange controlPointBuffers)
{
    _MTL4_msg_v_setControlPointBuffers__MTL4__BufferRange((const void*)this, nullptr, controlPointBuffers);
}

_MTL4_INLINE NS::UInteger MTL4::AccelerationStructureMotionCurveGeometryDescriptor::controlPointCount() const
{
    return _MTL4_msg_NS__UInteger_controlPointCount((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setControlPointCount(NS::UInteger controlPointCount)
{
    _MTL4_msg_v_setControlPointCount__NS__UInteger((const void*)this, nullptr, controlPointCount);
}

_MTL4_INLINE NS::UInteger MTL4::AccelerationStructureMotionCurveGeometryDescriptor::controlPointStride() const
{
    return _MTL4_msg_NS__UInteger_controlPointStride((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setControlPointStride(NS::UInteger controlPointStride)
{
    _MTL4_msg_v_setControlPointStride__NS__UInteger((const void*)this, nullptr, controlPointStride);
}

_MTL4_INLINE MTL::AttributeFormat MTL4::AccelerationStructureMotionCurveGeometryDescriptor::controlPointFormat() const
{
    return _MTL4_msg_MTL__AttributeFormat_controlPointFormat((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setControlPointFormat(MTL::AttributeFormat controlPointFormat)
{
    _MTL4_msg_v_setControlPointFormat__MTL__AttributeFormat((const void*)this, nullptr, controlPointFormat);
}

_MTL4_INLINE MTL4::BufferRange MTL4::AccelerationStructureMotionCurveGeometryDescriptor::radiusBuffers() const
{
    return _MTL4_msg_MTL4__BufferRange_radiusBuffers((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setRadiusBuffers(MTL4::BufferRange radiusBuffers)
{
    _MTL4_msg_v_setRadiusBuffers__MTL4__BufferRange((const void*)this, nullptr, radiusBuffers);
}

_MTL4_INLINE MTL::AttributeFormat MTL4::AccelerationStructureMotionCurveGeometryDescriptor::radiusFormat() const
{
    return _MTL4_msg_MTL__AttributeFormat_radiusFormat((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setRadiusFormat(MTL::AttributeFormat radiusFormat)
{
    _MTL4_msg_v_setRadiusFormat__MTL__AttributeFormat((const void*)this, nullptr, radiusFormat);
}

_MTL4_INLINE NS::UInteger MTL4::AccelerationStructureMotionCurveGeometryDescriptor::radiusStride() const
{
    return _MTL4_msg_NS__UInteger_radiusStride((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setRadiusStride(NS::UInteger radiusStride)
{
    _MTL4_msg_v_setRadiusStride__NS__UInteger((const void*)this, nullptr, radiusStride);
}

_MTL4_INLINE MTL4::BufferRange MTL4::AccelerationStructureMotionCurveGeometryDescriptor::indexBuffer() const
{
    return _MTL4_msg_MTL4__BufferRange_indexBuffer((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setIndexBuffer(MTL4::BufferRange indexBuffer)
{
    _MTL4_msg_v_setIndexBuffer__MTL4__BufferRange((const void*)this, nullptr, indexBuffer);
}

_MTL4_INLINE MTL::IndexType MTL4::AccelerationStructureMotionCurveGeometryDescriptor::indexType() const
{
    return _MTL4_msg_MTL__IndexType_indexType((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setIndexType(MTL::IndexType indexType)
{
    _MTL4_msg_v_setIndexType__MTL__IndexType((const void*)this, nullptr, indexType);
}

_MTL4_INLINE NS::UInteger MTL4::AccelerationStructureMotionCurveGeometryDescriptor::segmentCount() const
{
    return _MTL4_msg_NS__UInteger_segmentCount((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setSegmentCount(NS::UInteger segmentCount)
{
    _MTL4_msg_v_setSegmentCount__NS__UInteger((const void*)this, nullptr, segmentCount);
}

_MTL4_INLINE NS::UInteger MTL4::AccelerationStructureMotionCurveGeometryDescriptor::segmentControlPointCount() const
{
    return _MTL4_msg_NS__UInteger_segmentControlPointCount((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setSegmentControlPointCount(NS::UInteger segmentControlPointCount)
{
    _MTL4_msg_v_setSegmentControlPointCount__NS__UInteger((const void*)this, nullptr, segmentControlPointCount);
}

_MTL4_INLINE MTL::CurveType MTL4::AccelerationStructureMotionCurveGeometryDescriptor::curveType() const
{
    return _MTL4_msg_MTL__CurveType_curveType((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setCurveType(MTL::CurveType curveType)
{
    _MTL4_msg_v_setCurveType__MTL__CurveType((const void*)this, nullptr, curveType);
}

_MTL4_INLINE MTL::CurveBasis MTL4::AccelerationStructureMotionCurveGeometryDescriptor::curveBasis() const
{
    return _MTL4_msg_MTL__CurveBasis_curveBasis((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setCurveBasis(MTL::CurveBasis curveBasis)
{
    _MTL4_msg_v_setCurveBasis__MTL__CurveBasis((const void*)this, nullptr, curveBasis);
}

_MTL4_INLINE MTL::CurveEndCaps MTL4::AccelerationStructureMotionCurveGeometryDescriptor::curveEndCaps() const
{
    return _MTL4_msg_MTL__CurveEndCaps_curveEndCaps((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setCurveEndCaps(MTL::CurveEndCaps curveEndCaps)
{
    _MTL4_msg_v_setCurveEndCaps__MTL__CurveEndCaps((const void*)this, nullptr, curveEndCaps);
}

_MTL4_INLINE MTL4::InstanceAccelerationStructureDescriptor* MTL4::InstanceAccelerationStructureDescriptor::alloc()
{
    return _MTL4_msg_MTL4__InstanceAccelerationStructureDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4InstanceAccelerationStructureDescriptor, nullptr);
}

_MTL4_INLINE MTL4::InstanceAccelerationStructureDescriptor* MTL4::InstanceAccelerationStructureDescriptor::init() const
{
    return _MTL4_msg_MTL4__InstanceAccelerationStructureDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::BufferRange MTL4::InstanceAccelerationStructureDescriptor::instanceDescriptorBuffer() const
{
    return _MTL4_msg_MTL4__BufferRange_instanceDescriptorBuffer((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::InstanceAccelerationStructureDescriptor::setInstanceDescriptorBuffer(MTL4::BufferRange instanceDescriptorBuffer)
{
    _MTL4_msg_v_setInstanceDescriptorBuffer__MTL4__BufferRange((const void*)this, nullptr, instanceDescriptorBuffer);
}

_MTL4_INLINE NS::UInteger MTL4::InstanceAccelerationStructureDescriptor::instanceDescriptorStride() const
{
    return _MTL4_msg_NS__UInteger_instanceDescriptorStride((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::InstanceAccelerationStructureDescriptor::setInstanceDescriptorStride(NS::UInteger instanceDescriptorStride)
{
    _MTL4_msg_v_setInstanceDescriptorStride__NS__UInteger((const void*)this, nullptr, instanceDescriptorStride);
}

_MTL4_INLINE NS::UInteger MTL4::InstanceAccelerationStructureDescriptor::instanceCount() const
{
    return _MTL4_msg_NS__UInteger_instanceCount((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::InstanceAccelerationStructureDescriptor::setInstanceCount(NS::UInteger instanceCount)
{
    _MTL4_msg_v_setInstanceCount__NS__UInteger((const void*)this, nullptr, instanceCount);
}

_MTL4_INLINE MTL::AccelerationStructureInstanceDescriptorType MTL4::InstanceAccelerationStructureDescriptor::instanceDescriptorType() const
{
    return _MTL4_msg_MTL__AccelerationStructureInstanceDescriptorType_instanceDescriptorType((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::InstanceAccelerationStructureDescriptor::setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType)
{
    _MTL4_msg_v_setInstanceDescriptorType__MTL__AccelerationStructureInstanceDescriptorType((const void*)this, nullptr, instanceDescriptorType);
}

_MTL4_INLINE MTL4::BufferRange MTL4::InstanceAccelerationStructureDescriptor::motionTransformBuffer() const
{
    return _MTL4_msg_MTL4__BufferRange_motionTransformBuffer((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::InstanceAccelerationStructureDescriptor::setMotionTransformBuffer(MTL4::BufferRange motionTransformBuffer)
{
    _MTL4_msg_v_setMotionTransformBuffer__MTL4__BufferRange((const void*)this, nullptr, motionTransformBuffer);
}

_MTL4_INLINE NS::UInteger MTL4::InstanceAccelerationStructureDescriptor::motionTransformCount() const
{
    return _MTL4_msg_NS__UInteger_motionTransformCount((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::InstanceAccelerationStructureDescriptor::setMotionTransformCount(NS::UInteger motionTransformCount)
{
    _MTL4_msg_v_setMotionTransformCount__NS__UInteger((const void*)this, nullptr, motionTransformCount);
}

_MTL4_INLINE MTL::MatrixLayout MTL4::InstanceAccelerationStructureDescriptor::instanceTransformationMatrixLayout() const
{
    return _MTL4_msg_MTL__MatrixLayout_instanceTransformationMatrixLayout((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::InstanceAccelerationStructureDescriptor::setInstanceTransformationMatrixLayout(MTL::MatrixLayout instanceTransformationMatrixLayout)
{
    _MTL4_msg_v_setInstanceTransformationMatrixLayout__MTL__MatrixLayout((const void*)this, nullptr, instanceTransformationMatrixLayout);
}

_MTL4_INLINE MTL::TransformType MTL4::InstanceAccelerationStructureDescriptor::motionTransformType() const
{
    return _MTL4_msg_MTL__TransformType_motionTransformType((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::InstanceAccelerationStructureDescriptor::setMotionTransformType(MTL::TransformType motionTransformType)
{
    _MTL4_msg_v_setMotionTransformType__MTL__TransformType((const void*)this, nullptr, motionTransformType);
}

_MTL4_INLINE NS::UInteger MTL4::InstanceAccelerationStructureDescriptor::motionTransformStride() const
{
    return _MTL4_msg_NS__UInteger_motionTransformStride((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::InstanceAccelerationStructureDescriptor::setMotionTransformStride(NS::UInteger motionTransformStride)
{
    _MTL4_msg_v_setMotionTransformStride__NS__UInteger((const void*)this, nullptr, motionTransformStride);
}

_MTL4_INLINE MTL4::IndirectInstanceAccelerationStructureDescriptor* MTL4::IndirectInstanceAccelerationStructureDescriptor::alloc()
{
    return _MTL4_msg_MTL4__IndirectInstanceAccelerationStructureDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4IndirectInstanceAccelerationStructureDescriptor, nullptr);
}

_MTL4_INLINE MTL4::IndirectInstanceAccelerationStructureDescriptor* MTL4::IndirectInstanceAccelerationStructureDescriptor::init() const
{
    return _MTL4_msg_MTL4__IndirectInstanceAccelerationStructureDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::BufferRange MTL4::IndirectInstanceAccelerationStructureDescriptor::instanceDescriptorBuffer() const
{
    return _MTL4_msg_MTL4__BufferRange_instanceDescriptorBuffer((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::IndirectInstanceAccelerationStructureDescriptor::setInstanceDescriptorBuffer(MTL4::BufferRange instanceDescriptorBuffer)
{
    _MTL4_msg_v_setInstanceDescriptorBuffer__MTL4__BufferRange((const void*)this, nullptr, instanceDescriptorBuffer);
}

_MTL4_INLINE NS::UInteger MTL4::IndirectInstanceAccelerationStructureDescriptor::instanceDescriptorStride() const
{
    return _MTL4_msg_NS__UInteger_instanceDescriptorStride((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::IndirectInstanceAccelerationStructureDescriptor::setInstanceDescriptorStride(NS::UInteger instanceDescriptorStride)
{
    _MTL4_msg_v_setInstanceDescriptorStride__NS__UInteger((const void*)this, nullptr, instanceDescriptorStride);
}

_MTL4_INLINE NS::UInteger MTL4::IndirectInstanceAccelerationStructureDescriptor::maxInstanceCount() const
{
    return _MTL4_msg_NS__UInteger_maxInstanceCount((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::IndirectInstanceAccelerationStructureDescriptor::setMaxInstanceCount(NS::UInteger maxInstanceCount)
{
    _MTL4_msg_v_setMaxInstanceCount__NS__UInteger((const void*)this, nullptr, maxInstanceCount);
}

_MTL4_INLINE MTL4::BufferRange MTL4::IndirectInstanceAccelerationStructureDescriptor::instanceCountBuffer() const
{
    return _MTL4_msg_MTL4__BufferRange_instanceCountBuffer((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::IndirectInstanceAccelerationStructureDescriptor::setInstanceCountBuffer(MTL4::BufferRange instanceCountBuffer)
{
    _MTL4_msg_v_setInstanceCountBuffer__MTL4__BufferRange((const void*)this, nullptr, instanceCountBuffer);
}

_MTL4_INLINE MTL::AccelerationStructureInstanceDescriptorType MTL4::IndirectInstanceAccelerationStructureDescriptor::instanceDescriptorType() const
{
    return _MTL4_msg_MTL__AccelerationStructureInstanceDescriptorType_instanceDescriptorType((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::IndirectInstanceAccelerationStructureDescriptor::setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType)
{
    _MTL4_msg_v_setInstanceDescriptorType__MTL__AccelerationStructureInstanceDescriptorType((const void*)this, nullptr, instanceDescriptorType);
}

_MTL4_INLINE MTL4::BufferRange MTL4::IndirectInstanceAccelerationStructureDescriptor::motionTransformBuffer() const
{
    return _MTL4_msg_MTL4__BufferRange_motionTransformBuffer((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::IndirectInstanceAccelerationStructureDescriptor::setMotionTransformBuffer(MTL4::BufferRange motionTransformBuffer)
{
    _MTL4_msg_v_setMotionTransformBuffer__MTL4__BufferRange((const void*)this, nullptr, motionTransformBuffer);
}

_MTL4_INLINE NS::UInteger MTL4::IndirectInstanceAccelerationStructureDescriptor::maxMotionTransformCount() const
{
    return _MTL4_msg_NS__UInteger_maxMotionTransformCount((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::IndirectInstanceAccelerationStructureDescriptor::setMaxMotionTransformCount(NS::UInteger maxMotionTransformCount)
{
    _MTL4_msg_v_setMaxMotionTransformCount__NS__UInteger((const void*)this, nullptr, maxMotionTransformCount);
}

_MTL4_INLINE MTL4::BufferRange MTL4::IndirectInstanceAccelerationStructureDescriptor::motionTransformCountBuffer() const
{
    return _MTL4_msg_MTL4__BufferRange_motionTransformCountBuffer((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::IndirectInstanceAccelerationStructureDescriptor::setMotionTransformCountBuffer(MTL4::BufferRange motionTransformCountBuffer)
{
    _MTL4_msg_v_setMotionTransformCountBuffer__MTL4__BufferRange((const void*)this, nullptr, motionTransformCountBuffer);
}

_MTL4_INLINE MTL::MatrixLayout MTL4::IndirectInstanceAccelerationStructureDescriptor::instanceTransformationMatrixLayout() const
{
    return _MTL4_msg_MTL__MatrixLayout_instanceTransformationMatrixLayout((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::IndirectInstanceAccelerationStructureDescriptor::setInstanceTransformationMatrixLayout(MTL::MatrixLayout instanceTransformationMatrixLayout)
{
    _MTL4_msg_v_setInstanceTransformationMatrixLayout__MTL__MatrixLayout((const void*)this, nullptr, instanceTransformationMatrixLayout);
}

_MTL4_INLINE MTL::TransformType MTL4::IndirectInstanceAccelerationStructureDescriptor::motionTransformType() const
{
    return _MTL4_msg_MTL__TransformType_motionTransformType((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::IndirectInstanceAccelerationStructureDescriptor::setMotionTransformType(MTL::TransformType motionTransformType)
{
    _MTL4_msg_v_setMotionTransformType__MTL__TransformType((const void*)this, nullptr, motionTransformType);
}

_MTL4_INLINE NS::UInteger MTL4::IndirectInstanceAccelerationStructureDescriptor::motionTransformStride() const
{
    return _MTL4_msg_NS__UInteger_motionTransformStride((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::IndirectInstanceAccelerationStructureDescriptor::setMotionTransformStride(NS::UInteger motionTransformStride)
{
    _MTL4_msg_v_setMotionTransformStride__NS__UInteger((const void*)this, nullptr, motionTransformStride);
}
