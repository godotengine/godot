//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTL4AccelerationStructure.hpp
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
#include "MTLAccelerationStructure.hpp"
#include "MTLAccelerationStructureTypes.hpp"
#include "MTLArgument.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include "MTLStageInputOutputDescriptor.hpp"

namespace MTL4
{
class AccelerationStructureBoundingBoxGeometryDescriptor;
class AccelerationStructureCurveGeometryDescriptor;
class AccelerationStructureDescriptor;
class AccelerationStructureGeometryDescriptor;
class AccelerationStructureMotionBoundingBoxGeometryDescriptor;
class AccelerationStructureMotionCurveGeometryDescriptor;
class AccelerationStructureMotionTriangleGeometryDescriptor;
class AccelerationStructureTriangleGeometryDescriptor;
class IndirectInstanceAccelerationStructureDescriptor;
class InstanceAccelerationStructureDescriptor;
class PrimitiveAccelerationStructureDescriptor;

class AccelerationStructureDescriptor : public NS::Copying<AccelerationStructureDescriptor, MTL::AccelerationStructureDescriptor>
{
public:
    static AccelerationStructureDescriptor* alloc();

    AccelerationStructureDescriptor*        init();
};
class AccelerationStructureGeometryDescriptor : public NS::Copying<AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureGeometryDescriptor* alloc();

    bool                                            allowDuplicateIntersectionFunctionInvocation() const;

    AccelerationStructureGeometryDescriptor*        init();

    NS::UInteger                                    intersectionFunctionTableOffset() const;

    NS::String*                                     label() const;

    bool                                            opaque() const;

    BufferRange                                     primitiveDataBuffer() const;

    NS::UInteger                                    primitiveDataElementSize() const;

    NS::UInteger                                    primitiveDataStride() const;

    void                                            setAllowDuplicateIntersectionFunctionInvocation(bool allowDuplicateIntersectionFunctionInvocation);

    void                                            setIntersectionFunctionTableOffset(NS::UInteger intersectionFunctionTableOffset);

    void                                            setLabel(const NS::String* label);

    void                                            setOpaque(bool opaque);

    void                                            setPrimitiveDataBuffer(const MTL4::BufferRange primitiveDataBuffer);

    void                                            setPrimitiveDataElementSize(NS::UInteger primitiveDataElementSize);

    void                                            setPrimitiveDataStride(NS::UInteger primitiveDataStride);
};
class PrimitiveAccelerationStructureDescriptor : public NS::Copying<PrimitiveAccelerationStructureDescriptor, AccelerationStructureDescriptor>
{
public:
    static PrimitiveAccelerationStructureDescriptor* alloc();

    NS::Array*                                       geometryDescriptors() const;

    PrimitiveAccelerationStructureDescriptor*        init();

    MTL::MotionBorderMode                            motionEndBorderMode() const;

    float                                            motionEndTime() const;

    NS::UInteger                                     motionKeyframeCount() const;

    MTL::MotionBorderMode                            motionStartBorderMode() const;

    float                                            motionStartTime() const;

    void                                             setGeometryDescriptors(const NS::Array* geometryDescriptors);

    void                                             setMotionEndBorderMode(MTL::MotionBorderMode motionEndBorderMode);

    void                                             setMotionEndTime(float motionEndTime);

    void                                             setMotionKeyframeCount(NS::UInteger motionKeyframeCount);

    void                                             setMotionStartBorderMode(MTL::MotionBorderMode motionStartBorderMode);

    void                                             setMotionStartTime(float motionStartTime);
};
class AccelerationStructureTriangleGeometryDescriptor : public NS::Copying<AccelerationStructureTriangleGeometryDescriptor, AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureTriangleGeometryDescriptor* alloc();

    BufferRange                                             indexBuffer() const;

    MTL::IndexType                                          indexType() const;

    AccelerationStructureTriangleGeometryDescriptor*        init();

    void                                                    setIndexBuffer(const MTL4::BufferRange indexBuffer);

    void                                                    setIndexType(MTL::IndexType indexType);

    void                                                    setTransformationMatrixBuffer(const MTL4::BufferRange transformationMatrixBuffer);

    void                                                    setTransformationMatrixLayout(MTL::MatrixLayout transformationMatrixLayout);

    void                                                    setTriangleCount(NS::UInteger triangleCount);

    void                                                    setVertexBuffer(const MTL4::BufferRange vertexBuffer);

    void                                                    setVertexFormat(MTL::AttributeFormat vertexFormat);

    void                                                    setVertexStride(NS::UInteger vertexStride);

    BufferRange                                             transformationMatrixBuffer() const;

    MTL::MatrixLayout                                       transformationMatrixLayout() const;

    NS::UInteger                                            triangleCount() const;

    BufferRange                                             vertexBuffer() const;

    MTL::AttributeFormat                                    vertexFormat() const;

    NS::UInteger                                            vertexStride() const;
};
class AccelerationStructureBoundingBoxGeometryDescriptor : public NS::Copying<AccelerationStructureBoundingBoxGeometryDescriptor, AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureBoundingBoxGeometryDescriptor* alloc();

    BufferRange                                                boundingBoxBuffer() const;

    NS::UInteger                                               boundingBoxCount() const;

    NS::UInteger                                               boundingBoxStride() const;

    AccelerationStructureBoundingBoxGeometryDescriptor*        init();

    void                                                       setBoundingBoxBuffer(const MTL4::BufferRange boundingBoxBuffer);

    void                                                       setBoundingBoxCount(NS::UInteger boundingBoxCount);

    void                                                       setBoundingBoxStride(NS::UInteger boundingBoxStride);
};
class AccelerationStructureMotionTriangleGeometryDescriptor : public NS::Copying<AccelerationStructureMotionTriangleGeometryDescriptor, AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureMotionTriangleGeometryDescriptor* alloc();

    BufferRange                                                   indexBuffer() const;

    MTL::IndexType                                                indexType() const;

    AccelerationStructureMotionTriangleGeometryDescriptor*        init();

    void                                                          setIndexBuffer(const MTL4::BufferRange indexBuffer);

    void                                                          setIndexType(MTL::IndexType indexType);

    void                                                          setTransformationMatrixBuffer(const MTL4::BufferRange transformationMatrixBuffer);

    void                                                          setTransformationMatrixLayout(MTL::MatrixLayout transformationMatrixLayout);

    void                                                          setTriangleCount(NS::UInteger triangleCount);

    void                                                          setVertexBuffers(const MTL4::BufferRange vertexBuffers);

    void                                                          setVertexFormat(MTL::AttributeFormat vertexFormat);

    void                                                          setVertexStride(NS::UInteger vertexStride);

    BufferRange                                                   transformationMatrixBuffer() const;

    MTL::MatrixLayout                                             transformationMatrixLayout() const;

    NS::UInteger                                                  triangleCount() const;

    BufferRange                                                   vertexBuffers() const;

    MTL::AttributeFormat                                          vertexFormat() const;

    NS::UInteger                                                  vertexStride() const;
};
class AccelerationStructureMotionBoundingBoxGeometryDescriptor : public NS::Copying<AccelerationStructureMotionBoundingBoxGeometryDescriptor, AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureMotionBoundingBoxGeometryDescriptor* alloc();

    BufferRange                                                      boundingBoxBuffers() const;

    NS::UInteger                                                     boundingBoxCount() const;

    NS::UInteger                                                     boundingBoxStride() const;

    AccelerationStructureMotionBoundingBoxGeometryDescriptor*        init();

    void                                                             setBoundingBoxBuffers(const MTL4::BufferRange boundingBoxBuffers);

    void                                                             setBoundingBoxCount(NS::UInteger boundingBoxCount);

    void                                                             setBoundingBoxStride(NS::UInteger boundingBoxStride);
};
class AccelerationStructureCurveGeometryDescriptor : public NS::Copying<AccelerationStructureCurveGeometryDescriptor, AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureCurveGeometryDescriptor* alloc();

    BufferRange                                          controlPointBuffer() const;

    NS::UInteger                                         controlPointCount() const;

    MTL::AttributeFormat                                 controlPointFormat() const;

    NS::UInteger                                         controlPointStride() const;

    MTL::CurveBasis                                      curveBasis() const;

    MTL::CurveEndCaps                                    curveEndCaps() const;

    MTL::CurveType                                       curveType() const;

    BufferRange                                          indexBuffer() const;

    MTL::IndexType                                       indexType() const;

    AccelerationStructureCurveGeometryDescriptor*        init();

    BufferRange                                          radiusBuffer() const;

    MTL::AttributeFormat                                 radiusFormat() const;

    NS::UInteger                                         radiusStride() const;

    NS::UInteger                                         segmentControlPointCount() const;

    NS::UInteger                                         segmentCount() const;

    void                                                 setControlPointBuffer(const MTL4::BufferRange controlPointBuffer);

    void                                                 setControlPointCount(NS::UInteger controlPointCount);

    void                                                 setControlPointFormat(MTL::AttributeFormat controlPointFormat);

    void                                                 setControlPointStride(NS::UInteger controlPointStride);

    void                                                 setCurveBasis(MTL::CurveBasis curveBasis);

    void                                                 setCurveEndCaps(MTL::CurveEndCaps curveEndCaps);

    void                                                 setCurveType(MTL::CurveType curveType);

    void                                                 setIndexBuffer(const MTL4::BufferRange indexBuffer);

    void                                                 setIndexType(MTL::IndexType indexType);

    void                                                 setRadiusBuffer(const MTL4::BufferRange radiusBuffer);

    void                                                 setRadiusFormat(MTL::AttributeFormat radiusFormat);

    void                                                 setRadiusStride(NS::UInteger radiusStride);

    void                                                 setSegmentControlPointCount(NS::UInteger segmentControlPointCount);

    void                                                 setSegmentCount(NS::UInteger segmentCount);
};
class AccelerationStructureMotionCurveGeometryDescriptor : public NS::Copying<AccelerationStructureMotionCurveGeometryDescriptor, AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureMotionCurveGeometryDescriptor* alloc();

    BufferRange                                                controlPointBuffers() const;

    NS::UInteger                                               controlPointCount() const;

    MTL::AttributeFormat                                       controlPointFormat() const;

    NS::UInteger                                               controlPointStride() const;

    MTL::CurveBasis                                            curveBasis() const;

    MTL::CurveEndCaps                                          curveEndCaps() const;

    MTL::CurveType                                             curveType() const;

    BufferRange                                                indexBuffer() const;

    MTL::IndexType                                             indexType() const;

    AccelerationStructureMotionCurveGeometryDescriptor*        init();

    BufferRange                                                radiusBuffers() const;

    MTL::AttributeFormat                                       radiusFormat() const;

    NS::UInteger                                               radiusStride() const;

    NS::UInteger                                               segmentControlPointCount() const;

    NS::UInteger                                               segmentCount() const;

    void                                                       setControlPointBuffers(const MTL4::BufferRange controlPointBuffers);

    void                                                       setControlPointCount(NS::UInteger controlPointCount);

    void                                                       setControlPointFormat(MTL::AttributeFormat controlPointFormat);

    void                                                       setControlPointStride(NS::UInteger controlPointStride);

    void                                                       setCurveBasis(MTL::CurveBasis curveBasis);

    void                                                       setCurveEndCaps(MTL::CurveEndCaps curveEndCaps);

    void                                                       setCurveType(MTL::CurveType curveType);

    void                                                       setIndexBuffer(const MTL4::BufferRange indexBuffer);

    void                                                       setIndexType(MTL::IndexType indexType);

    void                                                       setRadiusBuffers(const MTL4::BufferRange radiusBuffers);

    void                                                       setRadiusFormat(MTL::AttributeFormat radiusFormat);

    void                                                       setRadiusStride(NS::UInteger radiusStride);

    void                                                       setSegmentControlPointCount(NS::UInteger segmentControlPointCount);

    void                                                       setSegmentCount(NS::UInteger segmentCount);
};
class InstanceAccelerationStructureDescriptor : public NS::Copying<InstanceAccelerationStructureDescriptor, AccelerationStructureDescriptor>
{
public:
    static InstanceAccelerationStructureDescriptor*  alloc();

    InstanceAccelerationStructureDescriptor*         init();

    NS::UInteger                                     instanceCount() const;

    BufferRange                                      instanceDescriptorBuffer() const;

    NS::UInteger                                     instanceDescriptorStride() const;

    MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType() const;

    MTL::MatrixLayout                                instanceTransformationMatrixLayout() const;

    BufferRange                                      motionTransformBuffer() const;

    NS::UInteger                                     motionTransformCount() const;

    NS::UInteger                                     motionTransformStride() const;

    MTL::TransformType                               motionTransformType() const;

    void                                             setInstanceCount(NS::UInteger instanceCount);

    void                                             setInstanceDescriptorBuffer(const MTL4::BufferRange instanceDescriptorBuffer);

    void                                             setInstanceDescriptorStride(NS::UInteger instanceDescriptorStride);

    void                                             setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType);

    void                                             setInstanceTransformationMatrixLayout(MTL::MatrixLayout instanceTransformationMatrixLayout);

    void                                             setMotionTransformBuffer(const MTL4::BufferRange motionTransformBuffer);

    void                                             setMotionTransformCount(NS::UInteger motionTransformCount);

    void                                             setMotionTransformStride(NS::UInteger motionTransformStride);

    void                                             setMotionTransformType(MTL::TransformType motionTransformType);
};
class IndirectInstanceAccelerationStructureDescriptor : public NS::Copying<IndirectInstanceAccelerationStructureDescriptor, AccelerationStructureDescriptor>
{
public:
    static IndirectInstanceAccelerationStructureDescriptor* alloc();

    IndirectInstanceAccelerationStructureDescriptor*        init();

    BufferRange                                             instanceCountBuffer() const;

    BufferRange                                             instanceDescriptorBuffer() const;

    NS::UInteger                                            instanceDescriptorStride() const;

    MTL::AccelerationStructureInstanceDescriptorType        instanceDescriptorType() const;

    MTL::MatrixLayout                                       instanceTransformationMatrixLayout() const;

    NS::UInteger                                            maxInstanceCount() const;

    NS::UInteger                                            maxMotionTransformCount() const;

    BufferRange                                             motionTransformBuffer() const;

    BufferRange                                             motionTransformCountBuffer() const;

    NS::UInteger                                            motionTransformStride() const;

    MTL::TransformType                                      motionTransformType() const;

    void                                                    setInstanceCountBuffer(const MTL4::BufferRange instanceCountBuffer);

    void                                                    setInstanceDescriptorBuffer(const MTL4::BufferRange instanceDescriptorBuffer);

    void                                                    setInstanceDescriptorStride(NS::UInteger instanceDescriptorStride);

    void                                                    setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType);

    void                                                    setInstanceTransformationMatrixLayout(MTL::MatrixLayout instanceTransformationMatrixLayout);

    void                                                    setMaxInstanceCount(NS::UInteger maxInstanceCount);

    void                                                    setMaxMotionTransformCount(NS::UInteger maxMotionTransformCount);

    void                                                    setMotionTransformBuffer(const MTL4::BufferRange motionTransformBuffer);

    void                                                    setMotionTransformCountBuffer(const MTL4::BufferRange motionTransformCountBuffer);

    void                                                    setMotionTransformStride(NS::UInteger motionTransformStride);

    void                                                    setMotionTransformType(MTL::TransformType motionTransformType);
};

}
_MTL_INLINE MTL4::AccelerationStructureDescriptor* MTL4::AccelerationStructureDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::AccelerationStructureDescriptor>(_MTL_PRIVATE_CLS(MTL4AccelerationStructureDescriptor));
}

_MTL_INLINE MTL4::AccelerationStructureDescriptor* MTL4::AccelerationStructureDescriptor::init()
{
    return NS::Object::init<MTL4::AccelerationStructureDescriptor>();
}

_MTL_INLINE MTL4::AccelerationStructureGeometryDescriptor* MTL4::AccelerationStructureGeometryDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::AccelerationStructureGeometryDescriptor>(_MTL_PRIVATE_CLS(MTL4AccelerationStructureGeometryDescriptor));
}

_MTL_INLINE bool MTL4::AccelerationStructureGeometryDescriptor::allowDuplicateIntersectionFunctionInvocation() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(allowDuplicateIntersectionFunctionInvocation));
}

_MTL_INLINE MTL4::AccelerationStructureGeometryDescriptor* MTL4::AccelerationStructureGeometryDescriptor::init()
{
    return NS::Object::init<MTL4::AccelerationStructureGeometryDescriptor>();
}

_MTL_INLINE NS::UInteger MTL4::AccelerationStructureGeometryDescriptor::intersectionFunctionTableOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(intersectionFunctionTableOffset));
}

_MTL_INLINE NS::String* MTL4::AccelerationStructureGeometryDescriptor::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE bool MTL4::AccelerationStructureGeometryDescriptor::opaque() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(opaque));
}

_MTL_INLINE MTL4::BufferRange MTL4::AccelerationStructureGeometryDescriptor::primitiveDataBuffer() const
{
    return Object::sendMessage<MTL4::BufferRange>(this, _MTL_PRIVATE_SEL(primitiveDataBuffer));
}

_MTL_INLINE NS::UInteger MTL4::AccelerationStructureGeometryDescriptor::primitiveDataElementSize() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(primitiveDataElementSize));
}

_MTL_INLINE NS::UInteger MTL4::AccelerationStructureGeometryDescriptor::primitiveDataStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(primitiveDataStride));
}

_MTL_INLINE void MTL4::AccelerationStructureGeometryDescriptor::setAllowDuplicateIntersectionFunctionInvocation(bool allowDuplicateIntersectionFunctionInvocation)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setAllowDuplicateIntersectionFunctionInvocation_), allowDuplicateIntersectionFunctionInvocation);
}

_MTL_INLINE void MTL4::AccelerationStructureGeometryDescriptor::setIntersectionFunctionTableOffset(NS::UInteger intersectionFunctionTableOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIntersectionFunctionTableOffset_), intersectionFunctionTableOffset);
}

_MTL_INLINE void MTL4::AccelerationStructureGeometryDescriptor::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE void MTL4::AccelerationStructureGeometryDescriptor::setOpaque(bool opaque)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setOpaque_), opaque);
}

_MTL_INLINE void MTL4::AccelerationStructureGeometryDescriptor::setPrimitiveDataBuffer(const MTL4::BufferRange primitiveDataBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPrimitiveDataBuffer_), primitiveDataBuffer);
}

_MTL_INLINE void MTL4::AccelerationStructureGeometryDescriptor::setPrimitiveDataElementSize(NS::UInteger primitiveDataElementSize)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPrimitiveDataElementSize_), primitiveDataElementSize);
}

_MTL_INLINE void MTL4::AccelerationStructureGeometryDescriptor::setPrimitiveDataStride(NS::UInteger primitiveDataStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPrimitiveDataStride_), primitiveDataStride);
}

_MTL_INLINE MTL4::PrimitiveAccelerationStructureDescriptor* MTL4::PrimitiveAccelerationStructureDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::PrimitiveAccelerationStructureDescriptor>(_MTL_PRIVATE_CLS(MTL4PrimitiveAccelerationStructureDescriptor));
}

_MTL_INLINE NS::Array* MTL4::PrimitiveAccelerationStructureDescriptor::geometryDescriptors() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(geometryDescriptors));
}

_MTL_INLINE MTL4::PrimitiveAccelerationStructureDescriptor* MTL4::PrimitiveAccelerationStructureDescriptor::init()
{
    return NS::Object::init<MTL4::PrimitiveAccelerationStructureDescriptor>();
}

_MTL_INLINE MTL::MotionBorderMode MTL4::PrimitiveAccelerationStructureDescriptor::motionEndBorderMode() const
{
    return Object::sendMessage<MTL::MotionBorderMode>(this, _MTL_PRIVATE_SEL(motionEndBorderMode));
}

_MTL_INLINE float MTL4::PrimitiveAccelerationStructureDescriptor::motionEndTime() const
{
    return Object::sendMessage<float>(this, _MTL_PRIVATE_SEL(motionEndTime));
}

_MTL_INLINE NS::UInteger MTL4::PrimitiveAccelerationStructureDescriptor::motionKeyframeCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(motionKeyframeCount));
}

_MTL_INLINE MTL::MotionBorderMode MTL4::PrimitiveAccelerationStructureDescriptor::motionStartBorderMode() const
{
    return Object::sendMessage<MTL::MotionBorderMode>(this, _MTL_PRIVATE_SEL(motionStartBorderMode));
}

_MTL_INLINE float MTL4::PrimitiveAccelerationStructureDescriptor::motionStartTime() const
{
    return Object::sendMessage<float>(this, _MTL_PRIVATE_SEL(motionStartTime));
}

_MTL_INLINE void MTL4::PrimitiveAccelerationStructureDescriptor::setGeometryDescriptors(const NS::Array* geometryDescriptors)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setGeometryDescriptors_), geometryDescriptors);
}

_MTL_INLINE void MTL4::PrimitiveAccelerationStructureDescriptor::setMotionEndBorderMode(MTL::MotionBorderMode motionEndBorderMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionEndBorderMode_), motionEndBorderMode);
}

_MTL_INLINE void MTL4::PrimitiveAccelerationStructureDescriptor::setMotionEndTime(float motionEndTime)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionEndTime_), motionEndTime);
}

_MTL_INLINE void MTL4::PrimitiveAccelerationStructureDescriptor::setMotionKeyframeCount(NS::UInteger motionKeyframeCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionKeyframeCount_), motionKeyframeCount);
}

_MTL_INLINE void MTL4::PrimitiveAccelerationStructureDescriptor::setMotionStartBorderMode(MTL::MotionBorderMode motionStartBorderMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionStartBorderMode_), motionStartBorderMode);
}

_MTL_INLINE void MTL4::PrimitiveAccelerationStructureDescriptor::setMotionStartTime(float motionStartTime)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionStartTime_), motionStartTime);
}

_MTL_INLINE MTL4::AccelerationStructureTriangleGeometryDescriptor* MTL4::AccelerationStructureTriangleGeometryDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::AccelerationStructureTriangleGeometryDescriptor>(_MTL_PRIVATE_CLS(MTL4AccelerationStructureTriangleGeometryDescriptor));
}

_MTL_INLINE MTL4::BufferRange MTL4::AccelerationStructureTriangleGeometryDescriptor::indexBuffer() const
{
    return Object::sendMessage<MTL4::BufferRange>(this, _MTL_PRIVATE_SEL(indexBuffer));
}

_MTL_INLINE MTL::IndexType MTL4::AccelerationStructureTriangleGeometryDescriptor::indexType() const
{
    return Object::sendMessage<MTL::IndexType>(this, _MTL_PRIVATE_SEL(indexType));
}

_MTL_INLINE MTL4::AccelerationStructureTriangleGeometryDescriptor* MTL4::AccelerationStructureTriangleGeometryDescriptor::init()
{
    return NS::Object::init<MTL4::AccelerationStructureTriangleGeometryDescriptor>();
}

_MTL_INLINE void MTL4::AccelerationStructureTriangleGeometryDescriptor::setIndexBuffer(const MTL4::BufferRange indexBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexBuffer_), indexBuffer);
}

_MTL_INLINE void MTL4::AccelerationStructureTriangleGeometryDescriptor::setIndexType(MTL::IndexType indexType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexType_), indexType);
}

_MTL_INLINE void MTL4::AccelerationStructureTriangleGeometryDescriptor::setTransformationMatrixBuffer(const MTL4::BufferRange transformationMatrixBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTransformationMatrixBuffer_), transformationMatrixBuffer);
}

_MTL_INLINE void MTL4::AccelerationStructureTriangleGeometryDescriptor::setTransformationMatrixLayout(MTL::MatrixLayout transformationMatrixLayout)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTransformationMatrixLayout_), transformationMatrixLayout);
}

_MTL_INLINE void MTL4::AccelerationStructureTriangleGeometryDescriptor::setTriangleCount(NS::UInteger triangleCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTriangleCount_), triangleCount);
}

_MTL_INLINE void MTL4::AccelerationStructureTriangleGeometryDescriptor::setVertexBuffer(const MTL4::BufferRange vertexBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexBuffer_), vertexBuffer);
}

_MTL_INLINE void MTL4::AccelerationStructureTriangleGeometryDescriptor::setVertexFormat(MTL::AttributeFormat vertexFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexFormat_), vertexFormat);
}

_MTL_INLINE void MTL4::AccelerationStructureTriangleGeometryDescriptor::setVertexStride(NS::UInteger vertexStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexStride_), vertexStride);
}

_MTL_INLINE MTL4::BufferRange MTL4::AccelerationStructureTriangleGeometryDescriptor::transformationMatrixBuffer() const
{
    return Object::sendMessage<MTL4::BufferRange>(this, _MTL_PRIVATE_SEL(transformationMatrixBuffer));
}

_MTL_INLINE MTL::MatrixLayout MTL4::AccelerationStructureTriangleGeometryDescriptor::transformationMatrixLayout() const
{
    return Object::sendMessage<MTL::MatrixLayout>(this, _MTL_PRIVATE_SEL(transformationMatrixLayout));
}

_MTL_INLINE NS::UInteger MTL4::AccelerationStructureTriangleGeometryDescriptor::triangleCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(triangleCount));
}

_MTL_INLINE MTL4::BufferRange MTL4::AccelerationStructureTriangleGeometryDescriptor::vertexBuffer() const
{
    return Object::sendMessage<MTL4::BufferRange>(this, _MTL_PRIVATE_SEL(vertexBuffer));
}

_MTL_INLINE MTL::AttributeFormat MTL4::AccelerationStructureTriangleGeometryDescriptor::vertexFormat() const
{
    return Object::sendMessage<MTL::AttributeFormat>(this, _MTL_PRIVATE_SEL(vertexFormat));
}

_MTL_INLINE NS::UInteger MTL4::AccelerationStructureTriangleGeometryDescriptor::vertexStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(vertexStride));
}

_MTL_INLINE MTL4::AccelerationStructureBoundingBoxGeometryDescriptor* MTL4::AccelerationStructureBoundingBoxGeometryDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::AccelerationStructureBoundingBoxGeometryDescriptor>(_MTL_PRIVATE_CLS(MTL4AccelerationStructureBoundingBoxGeometryDescriptor));
}

_MTL_INLINE MTL4::BufferRange MTL4::AccelerationStructureBoundingBoxGeometryDescriptor::boundingBoxBuffer() const
{
    return Object::sendMessage<MTL4::BufferRange>(this, _MTL_PRIVATE_SEL(boundingBoxBuffer));
}

_MTL_INLINE NS::UInteger MTL4::AccelerationStructureBoundingBoxGeometryDescriptor::boundingBoxCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(boundingBoxCount));
}

_MTL_INLINE NS::UInteger MTL4::AccelerationStructureBoundingBoxGeometryDescriptor::boundingBoxStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(boundingBoxStride));
}

_MTL_INLINE MTL4::AccelerationStructureBoundingBoxGeometryDescriptor* MTL4::AccelerationStructureBoundingBoxGeometryDescriptor::init()
{
    return NS::Object::init<MTL4::AccelerationStructureBoundingBoxGeometryDescriptor>();
}

_MTL_INLINE void MTL4::AccelerationStructureBoundingBoxGeometryDescriptor::setBoundingBoxBuffer(const MTL4::BufferRange boundingBoxBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBoundingBoxBuffer_), boundingBoxBuffer);
}

_MTL_INLINE void MTL4::AccelerationStructureBoundingBoxGeometryDescriptor::setBoundingBoxCount(NS::UInteger boundingBoxCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBoundingBoxCount_), boundingBoxCount);
}

_MTL_INLINE void MTL4::AccelerationStructureBoundingBoxGeometryDescriptor::setBoundingBoxStride(NS::UInteger boundingBoxStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBoundingBoxStride_), boundingBoxStride);
}

_MTL_INLINE MTL4::AccelerationStructureMotionTriangleGeometryDescriptor* MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::AccelerationStructureMotionTriangleGeometryDescriptor>(_MTL_PRIVATE_CLS(MTL4AccelerationStructureMotionTriangleGeometryDescriptor));
}

_MTL_INLINE MTL4::BufferRange MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::indexBuffer() const
{
    return Object::sendMessage<MTL4::BufferRange>(this, _MTL_PRIVATE_SEL(indexBuffer));
}

_MTL_INLINE MTL::IndexType MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::indexType() const
{
    return Object::sendMessage<MTL::IndexType>(this, _MTL_PRIVATE_SEL(indexType));
}

_MTL_INLINE MTL4::AccelerationStructureMotionTriangleGeometryDescriptor* MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::init()
{
    return NS::Object::init<MTL4::AccelerationStructureMotionTriangleGeometryDescriptor>();
}

_MTL_INLINE void MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::setIndexBuffer(const MTL4::BufferRange indexBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexBuffer_), indexBuffer);
}

_MTL_INLINE void MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::setIndexType(MTL::IndexType indexType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexType_), indexType);
}

_MTL_INLINE void MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::setTransformationMatrixBuffer(const MTL4::BufferRange transformationMatrixBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTransformationMatrixBuffer_), transformationMatrixBuffer);
}

_MTL_INLINE void MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::setTransformationMatrixLayout(MTL::MatrixLayout transformationMatrixLayout)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTransformationMatrixLayout_), transformationMatrixLayout);
}

_MTL_INLINE void MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::setTriangleCount(NS::UInteger triangleCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTriangleCount_), triangleCount);
}

_MTL_INLINE void MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::setVertexBuffers(const MTL4::BufferRange vertexBuffers)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexBuffers_), vertexBuffers);
}

_MTL_INLINE void MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::setVertexFormat(MTL::AttributeFormat vertexFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexFormat_), vertexFormat);
}

_MTL_INLINE void MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::setVertexStride(NS::UInteger vertexStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexStride_), vertexStride);
}

_MTL_INLINE MTL4::BufferRange MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::transformationMatrixBuffer() const
{
    return Object::sendMessage<MTL4::BufferRange>(this, _MTL_PRIVATE_SEL(transformationMatrixBuffer));
}

_MTL_INLINE MTL::MatrixLayout MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::transformationMatrixLayout() const
{
    return Object::sendMessage<MTL::MatrixLayout>(this, _MTL_PRIVATE_SEL(transformationMatrixLayout));
}

_MTL_INLINE NS::UInteger MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::triangleCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(triangleCount));
}

_MTL_INLINE MTL4::BufferRange MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::vertexBuffers() const
{
    return Object::sendMessage<MTL4::BufferRange>(this, _MTL_PRIVATE_SEL(vertexBuffers));
}

_MTL_INLINE MTL::AttributeFormat MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::vertexFormat() const
{
    return Object::sendMessage<MTL::AttributeFormat>(this, _MTL_PRIVATE_SEL(vertexFormat));
}

_MTL_INLINE NS::UInteger MTL4::AccelerationStructureMotionTriangleGeometryDescriptor::vertexStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(vertexStride));
}

_MTL_INLINE MTL4::AccelerationStructureMotionBoundingBoxGeometryDescriptor* MTL4::AccelerationStructureMotionBoundingBoxGeometryDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::AccelerationStructureMotionBoundingBoxGeometryDescriptor>(_MTL_PRIVATE_CLS(MTL4AccelerationStructureMotionBoundingBoxGeometryDescriptor));
}

_MTL_INLINE MTL4::BufferRange MTL4::AccelerationStructureMotionBoundingBoxGeometryDescriptor::boundingBoxBuffers() const
{
    return Object::sendMessage<MTL4::BufferRange>(this, _MTL_PRIVATE_SEL(boundingBoxBuffers));
}

_MTL_INLINE NS::UInteger MTL4::AccelerationStructureMotionBoundingBoxGeometryDescriptor::boundingBoxCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(boundingBoxCount));
}

_MTL_INLINE NS::UInteger MTL4::AccelerationStructureMotionBoundingBoxGeometryDescriptor::boundingBoxStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(boundingBoxStride));
}

_MTL_INLINE MTL4::AccelerationStructureMotionBoundingBoxGeometryDescriptor* MTL4::AccelerationStructureMotionBoundingBoxGeometryDescriptor::init()
{
    return NS::Object::init<MTL4::AccelerationStructureMotionBoundingBoxGeometryDescriptor>();
}

_MTL_INLINE void MTL4::AccelerationStructureMotionBoundingBoxGeometryDescriptor::setBoundingBoxBuffers(const MTL4::BufferRange boundingBoxBuffers)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBoundingBoxBuffers_), boundingBoxBuffers);
}

_MTL_INLINE void MTL4::AccelerationStructureMotionBoundingBoxGeometryDescriptor::setBoundingBoxCount(NS::UInteger boundingBoxCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBoundingBoxCount_), boundingBoxCount);
}

_MTL_INLINE void MTL4::AccelerationStructureMotionBoundingBoxGeometryDescriptor::setBoundingBoxStride(NS::UInteger boundingBoxStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBoundingBoxStride_), boundingBoxStride);
}

_MTL_INLINE MTL4::AccelerationStructureCurveGeometryDescriptor* MTL4::AccelerationStructureCurveGeometryDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::AccelerationStructureCurveGeometryDescriptor>(_MTL_PRIVATE_CLS(MTL4AccelerationStructureCurveGeometryDescriptor));
}

_MTL_INLINE MTL4::BufferRange MTL4::AccelerationStructureCurveGeometryDescriptor::controlPointBuffer() const
{
    return Object::sendMessage<MTL4::BufferRange>(this, _MTL_PRIVATE_SEL(controlPointBuffer));
}

_MTL_INLINE NS::UInteger MTL4::AccelerationStructureCurveGeometryDescriptor::controlPointCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(controlPointCount));
}

_MTL_INLINE MTL::AttributeFormat MTL4::AccelerationStructureCurveGeometryDescriptor::controlPointFormat() const
{
    return Object::sendMessage<MTL::AttributeFormat>(this, _MTL_PRIVATE_SEL(controlPointFormat));
}

_MTL_INLINE NS::UInteger MTL4::AccelerationStructureCurveGeometryDescriptor::controlPointStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(controlPointStride));
}

_MTL_INLINE MTL::CurveBasis MTL4::AccelerationStructureCurveGeometryDescriptor::curveBasis() const
{
    return Object::sendMessage<MTL::CurveBasis>(this, _MTL_PRIVATE_SEL(curveBasis));
}

_MTL_INLINE MTL::CurveEndCaps MTL4::AccelerationStructureCurveGeometryDescriptor::curveEndCaps() const
{
    return Object::sendMessage<MTL::CurveEndCaps>(this, _MTL_PRIVATE_SEL(curveEndCaps));
}

_MTL_INLINE MTL::CurveType MTL4::AccelerationStructureCurveGeometryDescriptor::curveType() const
{
    return Object::sendMessage<MTL::CurveType>(this, _MTL_PRIVATE_SEL(curveType));
}

_MTL_INLINE MTL4::BufferRange MTL4::AccelerationStructureCurveGeometryDescriptor::indexBuffer() const
{
    return Object::sendMessage<MTL4::BufferRange>(this, _MTL_PRIVATE_SEL(indexBuffer));
}

_MTL_INLINE MTL::IndexType MTL4::AccelerationStructureCurveGeometryDescriptor::indexType() const
{
    return Object::sendMessage<MTL::IndexType>(this, _MTL_PRIVATE_SEL(indexType));
}

_MTL_INLINE MTL4::AccelerationStructureCurveGeometryDescriptor* MTL4::AccelerationStructureCurveGeometryDescriptor::init()
{
    return NS::Object::init<MTL4::AccelerationStructureCurveGeometryDescriptor>();
}

_MTL_INLINE MTL4::BufferRange MTL4::AccelerationStructureCurveGeometryDescriptor::radiusBuffer() const
{
    return Object::sendMessage<MTL4::BufferRange>(this, _MTL_PRIVATE_SEL(radiusBuffer));
}

_MTL_INLINE MTL::AttributeFormat MTL4::AccelerationStructureCurveGeometryDescriptor::radiusFormat() const
{
    return Object::sendMessage<MTL::AttributeFormat>(this, _MTL_PRIVATE_SEL(radiusFormat));
}

_MTL_INLINE NS::UInteger MTL4::AccelerationStructureCurveGeometryDescriptor::radiusStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(radiusStride));
}

_MTL_INLINE NS::UInteger MTL4::AccelerationStructureCurveGeometryDescriptor::segmentControlPointCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(segmentControlPointCount));
}

_MTL_INLINE NS::UInteger MTL4::AccelerationStructureCurveGeometryDescriptor::segmentCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(segmentCount));
}

_MTL_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setControlPointBuffer(const MTL4::BufferRange controlPointBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointBuffer_), controlPointBuffer);
}

_MTL_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setControlPointCount(NS::UInteger controlPointCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointCount_), controlPointCount);
}

_MTL_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setControlPointFormat(MTL::AttributeFormat controlPointFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointFormat_), controlPointFormat);
}

_MTL_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setControlPointStride(NS::UInteger controlPointStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointStride_), controlPointStride);
}

_MTL_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setCurveBasis(MTL::CurveBasis curveBasis)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCurveBasis_), curveBasis);
}

_MTL_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setCurveEndCaps(MTL::CurveEndCaps curveEndCaps)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCurveEndCaps_), curveEndCaps);
}

_MTL_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setCurveType(MTL::CurveType curveType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCurveType_), curveType);
}

_MTL_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setIndexBuffer(const MTL4::BufferRange indexBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexBuffer_), indexBuffer);
}

_MTL_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setIndexType(MTL::IndexType indexType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexType_), indexType);
}

_MTL_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setRadiusBuffer(const MTL4::BufferRange radiusBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRadiusBuffer_), radiusBuffer);
}

_MTL_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setRadiusFormat(MTL::AttributeFormat radiusFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRadiusFormat_), radiusFormat);
}

_MTL_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setRadiusStride(NS::UInteger radiusStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRadiusStride_), radiusStride);
}

_MTL_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setSegmentControlPointCount(NS::UInteger segmentControlPointCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSegmentControlPointCount_), segmentControlPointCount);
}

_MTL_INLINE void MTL4::AccelerationStructureCurveGeometryDescriptor::setSegmentCount(NS::UInteger segmentCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSegmentCount_), segmentCount);
}

_MTL_INLINE MTL4::AccelerationStructureMotionCurveGeometryDescriptor* MTL4::AccelerationStructureMotionCurveGeometryDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::AccelerationStructureMotionCurveGeometryDescriptor>(_MTL_PRIVATE_CLS(MTL4AccelerationStructureMotionCurveGeometryDescriptor));
}

_MTL_INLINE MTL4::BufferRange MTL4::AccelerationStructureMotionCurveGeometryDescriptor::controlPointBuffers() const
{
    return Object::sendMessage<MTL4::BufferRange>(this, _MTL_PRIVATE_SEL(controlPointBuffers));
}

_MTL_INLINE NS::UInteger MTL4::AccelerationStructureMotionCurveGeometryDescriptor::controlPointCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(controlPointCount));
}

_MTL_INLINE MTL::AttributeFormat MTL4::AccelerationStructureMotionCurveGeometryDescriptor::controlPointFormat() const
{
    return Object::sendMessage<MTL::AttributeFormat>(this, _MTL_PRIVATE_SEL(controlPointFormat));
}

_MTL_INLINE NS::UInteger MTL4::AccelerationStructureMotionCurveGeometryDescriptor::controlPointStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(controlPointStride));
}

_MTL_INLINE MTL::CurveBasis MTL4::AccelerationStructureMotionCurveGeometryDescriptor::curveBasis() const
{
    return Object::sendMessage<MTL::CurveBasis>(this, _MTL_PRIVATE_SEL(curveBasis));
}

_MTL_INLINE MTL::CurveEndCaps MTL4::AccelerationStructureMotionCurveGeometryDescriptor::curveEndCaps() const
{
    return Object::sendMessage<MTL::CurveEndCaps>(this, _MTL_PRIVATE_SEL(curveEndCaps));
}

_MTL_INLINE MTL::CurveType MTL4::AccelerationStructureMotionCurveGeometryDescriptor::curveType() const
{
    return Object::sendMessage<MTL::CurveType>(this, _MTL_PRIVATE_SEL(curveType));
}

_MTL_INLINE MTL4::BufferRange MTL4::AccelerationStructureMotionCurveGeometryDescriptor::indexBuffer() const
{
    return Object::sendMessage<MTL4::BufferRange>(this, _MTL_PRIVATE_SEL(indexBuffer));
}

_MTL_INLINE MTL::IndexType MTL4::AccelerationStructureMotionCurveGeometryDescriptor::indexType() const
{
    return Object::sendMessage<MTL::IndexType>(this, _MTL_PRIVATE_SEL(indexType));
}

_MTL_INLINE MTL4::AccelerationStructureMotionCurveGeometryDescriptor* MTL4::AccelerationStructureMotionCurveGeometryDescriptor::init()
{
    return NS::Object::init<MTL4::AccelerationStructureMotionCurveGeometryDescriptor>();
}

_MTL_INLINE MTL4::BufferRange MTL4::AccelerationStructureMotionCurveGeometryDescriptor::radiusBuffers() const
{
    return Object::sendMessage<MTL4::BufferRange>(this, _MTL_PRIVATE_SEL(radiusBuffers));
}

_MTL_INLINE MTL::AttributeFormat MTL4::AccelerationStructureMotionCurveGeometryDescriptor::radiusFormat() const
{
    return Object::sendMessage<MTL::AttributeFormat>(this, _MTL_PRIVATE_SEL(radiusFormat));
}

_MTL_INLINE NS::UInteger MTL4::AccelerationStructureMotionCurveGeometryDescriptor::radiusStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(radiusStride));
}

_MTL_INLINE NS::UInteger MTL4::AccelerationStructureMotionCurveGeometryDescriptor::segmentControlPointCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(segmentControlPointCount));
}

_MTL_INLINE NS::UInteger MTL4::AccelerationStructureMotionCurveGeometryDescriptor::segmentCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(segmentCount));
}

_MTL_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setControlPointBuffers(const MTL4::BufferRange controlPointBuffers)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointBuffers_), controlPointBuffers);
}

_MTL_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setControlPointCount(NS::UInteger controlPointCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointCount_), controlPointCount);
}

_MTL_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setControlPointFormat(MTL::AttributeFormat controlPointFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointFormat_), controlPointFormat);
}

_MTL_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setControlPointStride(NS::UInteger controlPointStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointStride_), controlPointStride);
}

_MTL_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setCurveBasis(MTL::CurveBasis curveBasis)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCurveBasis_), curveBasis);
}

_MTL_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setCurveEndCaps(MTL::CurveEndCaps curveEndCaps)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCurveEndCaps_), curveEndCaps);
}

_MTL_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setCurveType(MTL::CurveType curveType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCurveType_), curveType);
}

_MTL_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setIndexBuffer(const MTL4::BufferRange indexBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexBuffer_), indexBuffer);
}

_MTL_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setIndexType(MTL::IndexType indexType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexType_), indexType);
}

_MTL_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setRadiusBuffers(const MTL4::BufferRange radiusBuffers)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRadiusBuffers_), radiusBuffers);
}

_MTL_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setRadiusFormat(MTL::AttributeFormat radiusFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRadiusFormat_), radiusFormat);
}

_MTL_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setRadiusStride(NS::UInteger radiusStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRadiusStride_), radiusStride);
}

_MTL_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setSegmentControlPointCount(NS::UInteger segmentControlPointCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSegmentControlPointCount_), segmentControlPointCount);
}

_MTL_INLINE void MTL4::AccelerationStructureMotionCurveGeometryDescriptor::setSegmentCount(NS::UInteger segmentCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSegmentCount_), segmentCount);
}

_MTL_INLINE MTL4::InstanceAccelerationStructureDescriptor* MTL4::InstanceAccelerationStructureDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::InstanceAccelerationStructureDescriptor>(_MTL_PRIVATE_CLS(MTL4InstanceAccelerationStructureDescriptor));
}

_MTL_INLINE MTL4::InstanceAccelerationStructureDescriptor* MTL4::InstanceAccelerationStructureDescriptor::init()
{
    return NS::Object::init<MTL4::InstanceAccelerationStructureDescriptor>();
}

_MTL_INLINE NS::UInteger MTL4::InstanceAccelerationStructureDescriptor::instanceCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(instanceCount));
}

_MTL_INLINE MTL4::BufferRange MTL4::InstanceAccelerationStructureDescriptor::instanceDescriptorBuffer() const
{
    return Object::sendMessage<MTL4::BufferRange>(this, _MTL_PRIVATE_SEL(instanceDescriptorBuffer));
}

_MTL_INLINE NS::UInteger MTL4::InstanceAccelerationStructureDescriptor::instanceDescriptorStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(instanceDescriptorStride));
}

_MTL_INLINE MTL::AccelerationStructureInstanceDescriptorType MTL4::InstanceAccelerationStructureDescriptor::instanceDescriptorType() const
{
    return Object::sendMessage<MTL::AccelerationStructureInstanceDescriptorType>(this, _MTL_PRIVATE_SEL(instanceDescriptorType));
}

_MTL_INLINE MTL::MatrixLayout MTL4::InstanceAccelerationStructureDescriptor::instanceTransformationMatrixLayout() const
{
    return Object::sendMessage<MTL::MatrixLayout>(this, _MTL_PRIVATE_SEL(instanceTransformationMatrixLayout));
}

_MTL_INLINE MTL4::BufferRange MTL4::InstanceAccelerationStructureDescriptor::motionTransformBuffer() const
{
    return Object::sendMessage<MTL4::BufferRange>(this, _MTL_PRIVATE_SEL(motionTransformBuffer));
}

_MTL_INLINE NS::UInteger MTL4::InstanceAccelerationStructureDescriptor::motionTransformCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(motionTransformCount));
}

_MTL_INLINE NS::UInteger MTL4::InstanceAccelerationStructureDescriptor::motionTransformStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(motionTransformStride));
}

_MTL_INLINE MTL::TransformType MTL4::InstanceAccelerationStructureDescriptor::motionTransformType() const
{
    return Object::sendMessage<MTL::TransformType>(this, _MTL_PRIVATE_SEL(motionTransformType));
}

_MTL_INLINE void MTL4::InstanceAccelerationStructureDescriptor::setInstanceCount(NS::UInteger instanceCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceCount_), instanceCount);
}

_MTL_INLINE void MTL4::InstanceAccelerationStructureDescriptor::setInstanceDescriptorBuffer(const MTL4::BufferRange instanceDescriptorBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceDescriptorBuffer_), instanceDescriptorBuffer);
}

_MTL_INLINE void MTL4::InstanceAccelerationStructureDescriptor::setInstanceDescriptorStride(NS::UInteger instanceDescriptorStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceDescriptorStride_), instanceDescriptorStride);
}

_MTL_INLINE void MTL4::InstanceAccelerationStructureDescriptor::setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceDescriptorType_), instanceDescriptorType);
}

_MTL_INLINE void MTL4::InstanceAccelerationStructureDescriptor::setInstanceTransformationMatrixLayout(MTL::MatrixLayout instanceTransformationMatrixLayout)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceTransformationMatrixLayout_), instanceTransformationMatrixLayout);
}

_MTL_INLINE void MTL4::InstanceAccelerationStructureDescriptor::setMotionTransformBuffer(const MTL4::BufferRange motionTransformBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformBuffer_), motionTransformBuffer);
}

_MTL_INLINE void MTL4::InstanceAccelerationStructureDescriptor::setMotionTransformCount(NS::UInteger motionTransformCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformCount_), motionTransformCount);
}

_MTL_INLINE void MTL4::InstanceAccelerationStructureDescriptor::setMotionTransformStride(NS::UInteger motionTransformStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformStride_), motionTransformStride);
}

_MTL_INLINE void MTL4::InstanceAccelerationStructureDescriptor::setMotionTransformType(MTL::TransformType motionTransformType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformType_), motionTransformType);
}

_MTL_INLINE MTL4::IndirectInstanceAccelerationStructureDescriptor* MTL4::IndirectInstanceAccelerationStructureDescriptor::alloc()
{
    return NS::Object::alloc<MTL4::IndirectInstanceAccelerationStructureDescriptor>(_MTL_PRIVATE_CLS(MTL4IndirectInstanceAccelerationStructureDescriptor));
}

_MTL_INLINE MTL4::IndirectInstanceAccelerationStructureDescriptor* MTL4::IndirectInstanceAccelerationStructureDescriptor::init()
{
    return NS::Object::init<MTL4::IndirectInstanceAccelerationStructureDescriptor>();
}

_MTL_INLINE MTL4::BufferRange MTL4::IndirectInstanceAccelerationStructureDescriptor::instanceCountBuffer() const
{
    return Object::sendMessage<MTL4::BufferRange>(this, _MTL_PRIVATE_SEL(instanceCountBuffer));
}

_MTL_INLINE MTL4::BufferRange MTL4::IndirectInstanceAccelerationStructureDescriptor::instanceDescriptorBuffer() const
{
    return Object::sendMessage<MTL4::BufferRange>(this, _MTL_PRIVATE_SEL(instanceDescriptorBuffer));
}

_MTL_INLINE NS::UInteger MTL4::IndirectInstanceAccelerationStructureDescriptor::instanceDescriptorStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(instanceDescriptorStride));
}

_MTL_INLINE MTL::AccelerationStructureInstanceDescriptorType MTL4::IndirectInstanceAccelerationStructureDescriptor::instanceDescriptorType() const
{
    return Object::sendMessage<MTL::AccelerationStructureInstanceDescriptorType>(this, _MTL_PRIVATE_SEL(instanceDescriptorType));
}

_MTL_INLINE MTL::MatrixLayout MTL4::IndirectInstanceAccelerationStructureDescriptor::instanceTransformationMatrixLayout() const
{
    return Object::sendMessage<MTL::MatrixLayout>(this, _MTL_PRIVATE_SEL(instanceTransformationMatrixLayout));
}

_MTL_INLINE NS::UInteger MTL4::IndirectInstanceAccelerationStructureDescriptor::maxInstanceCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxInstanceCount));
}

_MTL_INLINE NS::UInteger MTL4::IndirectInstanceAccelerationStructureDescriptor::maxMotionTransformCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxMotionTransformCount));
}

_MTL_INLINE MTL4::BufferRange MTL4::IndirectInstanceAccelerationStructureDescriptor::motionTransformBuffer() const
{
    return Object::sendMessage<MTL4::BufferRange>(this, _MTL_PRIVATE_SEL(motionTransformBuffer));
}

_MTL_INLINE MTL4::BufferRange MTL4::IndirectInstanceAccelerationStructureDescriptor::motionTransformCountBuffer() const
{
    return Object::sendMessage<MTL4::BufferRange>(this, _MTL_PRIVATE_SEL(motionTransformCountBuffer));
}

_MTL_INLINE NS::UInteger MTL4::IndirectInstanceAccelerationStructureDescriptor::motionTransformStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(motionTransformStride));
}

_MTL_INLINE MTL::TransformType MTL4::IndirectInstanceAccelerationStructureDescriptor::motionTransformType() const
{
    return Object::sendMessage<MTL::TransformType>(this, _MTL_PRIVATE_SEL(motionTransformType));
}

_MTL_INLINE void MTL4::IndirectInstanceAccelerationStructureDescriptor::setInstanceCountBuffer(const MTL4::BufferRange instanceCountBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceCountBuffer_), instanceCountBuffer);
}

_MTL_INLINE void MTL4::IndirectInstanceAccelerationStructureDescriptor::setInstanceDescriptorBuffer(const MTL4::BufferRange instanceDescriptorBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceDescriptorBuffer_), instanceDescriptorBuffer);
}

_MTL_INLINE void MTL4::IndirectInstanceAccelerationStructureDescriptor::setInstanceDescriptorStride(NS::UInteger instanceDescriptorStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceDescriptorStride_), instanceDescriptorStride);
}

_MTL_INLINE void MTL4::IndirectInstanceAccelerationStructureDescriptor::setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceDescriptorType_), instanceDescriptorType);
}

_MTL_INLINE void MTL4::IndirectInstanceAccelerationStructureDescriptor::setInstanceTransformationMatrixLayout(MTL::MatrixLayout instanceTransformationMatrixLayout)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceTransformationMatrixLayout_), instanceTransformationMatrixLayout);
}

_MTL_INLINE void MTL4::IndirectInstanceAccelerationStructureDescriptor::setMaxInstanceCount(NS::UInteger maxInstanceCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxInstanceCount_), maxInstanceCount);
}

_MTL_INLINE void MTL4::IndirectInstanceAccelerationStructureDescriptor::setMaxMotionTransformCount(NS::UInteger maxMotionTransformCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxMotionTransformCount_), maxMotionTransformCount);
}

_MTL_INLINE void MTL4::IndirectInstanceAccelerationStructureDescriptor::setMotionTransformBuffer(const MTL4::BufferRange motionTransformBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformBuffer_), motionTransformBuffer);
}

_MTL_INLINE void MTL4::IndirectInstanceAccelerationStructureDescriptor::setMotionTransformCountBuffer(const MTL4::BufferRange motionTransformCountBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformCountBuffer_), motionTransformCountBuffer);
}

_MTL_INLINE void MTL4::IndirectInstanceAccelerationStructureDescriptor::setMotionTransformStride(NS::UInteger motionTransformStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformStride_), motionTransformStride);
}

_MTL_INLINE void MTL4::IndirectInstanceAccelerationStructureDescriptor::setMotionTransformType(MTL::TransformType motionTransformType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformType_), motionTransformType);
}
