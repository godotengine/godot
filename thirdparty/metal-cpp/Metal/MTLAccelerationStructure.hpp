//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLAccelerationStructure.hpp
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
#include "MTLAccelerationStructureTypes.hpp"
#include "MTLArgument.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include "MTLResource.hpp"
#include "MTLStageInputOutputDescriptor.hpp"
#include "MTLTypes.hpp"
#include <cstdint>

namespace MTL
{
class AccelerationStructureBoundingBoxGeometryDescriptor;
class AccelerationStructureCurveGeometryDescriptor;
class AccelerationStructureDescriptor;
class AccelerationStructureGeometryDescriptor;
class AccelerationStructureMotionBoundingBoxGeometryDescriptor;
class AccelerationStructureMotionCurveGeometryDescriptor;
class AccelerationStructureMotionTriangleGeometryDescriptor;
class AccelerationStructureTriangleGeometryDescriptor;
class Buffer;
class IndirectInstanceAccelerationStructureDescriptor;
class InstanceAccelerationStructureDescriptor;
class MotionKeyframeData;
class PrimitiveAccelerationStructureDescriptor;
}

namespace MTL
{
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

_MTL_OPTIONS(NS::UInteger, AccelerationStructureRefitOptions) {
    AccelerationStructureRefitOptionVertexData = 1,
    AccelerationStructureRefitOptionPerPrimitiveData = 1 << 1,
};

_MTL_OPTIONS(NS::UInteger, AccelerationStructureUsage) {
    AccelerationStructureUsageNone = 0,
    AccelerationStructureUsageRefit = 1,
    AccelerationStructureUsagePreferFastBuild = 1 << 1,
    AccelerationStructureUsageExtendedLimits = 1 << 2,
    AccelerationStructureUsagePreferFastIntersection = 1 << 4,
    AccelerationStructureUsageMinimizeMemory = 1 << 5,
};

_MTL_OPTIONS(uint32_t, AccelerationStructureInstanceOptions) {
    AccelerationStructureInstanceOptionNone = 0,
    AccelerationStructureInstanceOptionDisableTriangleCulling = 1,
    AccelerationStructureInstanceOptionTriangleFrontFacingWindingCounterClockwise = 1 << 1,
    AccelerationStructureInstanceOptionOpaque = 1 << 2,
    AccelerationStructureInstanceOptionNonOpaque = 1 << 3,
};

struct AccelerationStructureInstanceDescriptor
{
    MTL::PackedFloat4x3                       transformationMatrix;
    MTL::AccelerationStructureInstanceOptions options;
    uint32_t                                  mask;
    uint32_t                                  intersectionFunctionTableOffset;
    uint32_t                                  accelerationStructureIndex;
} _MTL_PACKED;

struct AccelerationStructureUserIDInstanceDescriptor
{
    MTL::PackedFloat4x3                       transformationMatrix;
    MTL::AccelerationStructureInstanceOptions options;
    uint32_t                                  mask;
    uint32_t                                  intersectionFunctionTableOffset;
    uint32_t                                  accelerationStructureIndex;
    uint32_t                                  userID;
} _MTL_PACKED;

struct AccelerationStructureMotionInstanceDescriptor
{
    MTL::AccelerationStructureInstanceOptions options;
    uint32_t                                  mask;
    uint32_t                                  intersectionFunctionTableOffset;
    uint32_t                                  accelerationStructureIndex;
    uint32_t                                  userID;
    uint32_t                                  motionTransformsStartIndex;
    uint32_t                                  motionTransformsCount;
    MTL::MotionBorderMode                     motionStartBorderMode;
    MTL::MotionBorderMode                     motionEndBorderMode;
    float                                     motionStartTime;
    float                                     motionEndTime;
} _MTL_PACKED;

struct IndirectAccelerationStructureInstanceDescriptor
{
    MTL::PackedFloat4x3                       transformationMatrix;
    MTL::AccelerationStructureInstanceOptions options;
    uint32_t                                  mask;
    uint32_t                                  intersectionFunctionTableOffset;
    uint32_t                                  userID;
    MTL::ResourceID                           accelerationStructureID;
} _MTL_PACKED;

struct IndirectAccelerationStructureMotionInstanceDescriptor
{
    MTL::AccelerationStructureInstanceOptions options;
    uint32_t                                  mask;
    uint32_t                                  intersectionFunctionTableOffset;
    uint32_t                                  userID;
    MTL::ResourceID                           accelerationStructureID;
    uint32_t                                  motionTransformsStartIndex;
    uint32_t                                  motionTransformsCount;
    MTL::MotionBorderMode                     motionStartBorderMode;
    MTL::MotionBorderMode                     motionEndBorderMode;
    float                                     motionStartTime;
    float                                     motionEndTime;
} _MTL_PACKED;

class AccelerationStructureDescriptor : public NS::Copying<AccelerationStructureDescriptor>
{
public:
    static AccelerationStructureDescriptor* alloc();

    AccelerationStructureDescriptor*        init();

    void                                    setUsage(MTL::AccelerationStructureUsage usage);
    AccelerationStructureUsage              usage() const;
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

    Buffer*                                         primitiveDataBuffer() const;
    NS::UInteger                                    primitiveDataBufferOffset() const;

    NS::UInteger                                    primitiveDataElementSize() const;

    NS::UInteger                                    primitiveDataStride() const;

    void                                            setAllowDuplicateIntersectionFunctionInvocation(bool allowDuplicateIntersectionFunctionInvocation);

    void                                            setIntersectionFunctionTableOffset(NS::UInteger intersectionFunctionTableOffset);

    void                                            setLabel(const NS::String* label);

    void                                            setOpaque(bool opaque);

    void                                            setPrimitiveDataBuffer(const MTL::Buffer* primitiveDataBuffer);
    void                                            setPrimitiveDataBufferOffset(NS::UInteger primitiveDataBufferOffset);

    void                                            setPrimitiveDataElementSize(NS::UInteger primitiveDataElementSize);

    void                                            setPrimitiveDataStride(NS::UInteger primitiveDataStride);
};
class PrimitiveAccelerationStructureDescriptor : public NS::Copying<PrimitiveAccelerationStructureDescriptor, AccelerationStructureDescriptor>
{
public:
    static PrimitiveAccelerationStructureDescriptor* alloc();

    static PrimitiveAccelerationStructureDescriptor* descriptor();
    NS::Array*                                       geometryDescriptors() const;

    PrimitiveAccelerationStructureDescriptor*        init();

    MotionBorderMode                                 motionEndBorderMode() const;

    float                                            motionEndTime() const;

    NS::UInteger                                     motionKeyframeCount() const;

    MotionBorderMode                                 motionStartBorderMode() const;

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

    static AccelerationStructureTriangleGeometryDescriptor* descriptor();

    Buffer*                                                 indexBuffer() const;
    NS::UInteger                                            indexBufferOffset() const;

    IndexType                                               indexType() const;

    AccelerationStructureTriangleGeometryDescriptor*        init();

    void                                                    setIndexBuffer(const MTL::Buffer* indexBuffer);
    void                                                    setIndexBufferOffset(NS::UInteger indexBufferOffset);

    void                                                    setIndexType(MTL::IndexType indexType);

    void                                                    setTransformationMatrixBuffer(const MTL::Buffer* transformationMatrixBuffer);
    void                                                    setTransformationMatrixBufferOffset(NS::UInteger transformationMatrixBufferOffset);

    void                                                    setTransformationMatrixLayout(MTL::MatrixLayout transformationMatrixLayout);

    void                                                    setTriangleCount(NS::UInteger triangleCount);

    void                                                    setVertexBuffer(const MTL::Buffer* vertexBuffer);
    void                                                    setVertexBufferOffset(NS::UInteger vertexBufferOffset);

    void                                                    setVertexFormat(MTL::AttributeFormat vertexFormat);

    void                                                    setVertexStride(NS::UInteger vertexStride);

    Buffer*                                                 transformationMatrixBuffer() const;
    NS::UInteger                                            transformationMatrixBufferOffset() const;

    MatrixLayout                                            transformationMatrixLayout() const;

    NS::UInteger                                            triangleCount() const;

    Buffer*                                                 vertexBuffer() const;
    NS::UInteger                                            vertexBufferOffset() const;

    AttributeFormat                                         vertexFormat() const;

    NS::UInteger                                            vertexStride() const;
};
class AccelerationStructureBoundingBoxGeometryDescriptor : public NS::Copying<AccelerationStructureBoundingBoxGeometryDescriptor, AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureBoundingBoxGeometryDescriptor* alloc();

    Buffer*                                                    boundingBoxBuffer() const;
    NS::UInteger                                               boundingBoxBufferOffset() const;

    NS::UInteger                                               boundingBoxCount() const;

    NS::UInteger                                               boundingBoxStride() const;

    static AccelerationStructureBoundingBoxGeometryDescriptor* descriptor();

    AccelerationStructureBoundingBoxGeometryDescriptor*        init();

    void                                                       setBoundingBoxBuffer(const MTL::Buffer* boundingBoxBuffer);
    void                                                       setBoundingBoxBufferOffset(NS::UInteger boundingBoxBufferOffset);

    void                                                       setBoundingBoxCount(NS::UInteger boundingBoxCount);

    void                                                       setBoundingBoxStride(NS::UInteger boundingBoxStride);
};
class MotionKeyframeData : public NS::Referencing<MotionKeyframeData>
{
public:
    static MotionKeyframeData* alloc();

    Buffer*                    buffer() const;

    static MotionKeyframeData* data();

    MotionKeyframeData*        init();

    NS::UInteger               offset() const;

    void                       setBuffer(const MTL::Buffer* buffer);

    void                       setOffset(NS::UInteger offset);
};
class AccelerationStructureMotionTriangleGeometryDescriptor : public NS::Copying<AccelerationStructureMotionTriangleGeometryDescriptor, AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureMotionTriangleGeometryDescriptor* alloc();

    static AccelerationStructureMotionTriangleGeometryDescriptor* descriptor();

    Buffer*                                                       indexBuffer() const;
    NS::UInteger                                                  indexBufferOffset() const;

    IndexType                                                     indexType() const;

    AccelerationStructureMotionTriangleGeometryDescriptor*        init();

    void                                                          setIndexBuffer(const MTL::Buffer* indexBuffer);
    void                                                          setIndexBufferOffset(NS::UInteger indexBufferOffset);

    void                                                          setIndexType(MTL::IndexType indexType);

    void                                                          setTransformationMatrixBuffer(const MTL::Buffer* transformationMatrixBuffer);
    void                                                          setTransformationMatrixBufferOffset(NS::UInteger transformationMatrixBufferOffset);

    void                                                          setTransformationMatrixLayout(MTL::MatrixLayout transformationMatrixLayout);

    void                                                          setTriangleCount(NS::UInteger triangleCount);

    void                                                          setVertexBuffers(const NS::Array* vertexBuffers);

    void                                                          setVertexFormat(MTL::AttributeFormat vertexFormat);

    void                                                          setVertexStride(NS::UInteger vertexStride);

    Buffer*                                                       transformationMatrixBuffer() const;
    NS::UInteger                                                  transformationMatrixBufferOffset() const;

    MatrixLayout                                                  transformationMatrixLayout() const;

    NS::UInteger                                                  triangleCount() const;

    NS::Array*                                                    vertexBuffers() const;

    AttributeFormat                                               vertexFormat() const;

    NS::UInteger                                                  vertexStride() const;
};
class AccelerationStructureMotionBoundingBoxGeometryDescriptor : public NS::Copying<AccelerationStructureMotionBoundingBoxGeometryDescriptor, AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureMotionBoundingBoxGeometryDescriptor* alloc();

    NS::Array*                                                       boundingBoxBuffers() const;

    NS::UInteger                                                     boundingBoxCount() const;

    NS::UInteger                                                     boundingBoxStride() const;

    static AccelerationStructureMotionBoundingBoxGeometryDescriptor* descriptor();

    AccelerationStructureMotionBoundingBoxGeometryDescriptor*        init();

    void                                                             setBoundingBoxBuffers(const NS::Array* boundingBoxBuffers);

    void                                                             setBoundingBoxCount(NS::UInteger boundingBoxCount);

    void                                                             setBoundingBoxStride(NS::UInteger boundingBoxStride);
};
class AccelerationStructureCurveGeometryDescriptor : public NS::Copying<AccelerationStructureCurveGeometryDescriptor, AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureCurveGeometryDescriptor* alloc();

    Buffer*                                              controlPointBuffer() const;
    NS::UInteger                                         controlPointBufferOffset() const;

    NS::UInteger                                         controlPointCount() const;

    AttributeFormat                                      controlPointFormat() const;

    NS::UInteger                                         controlPointStride() const;

    CurveBasis                                           curveBasis() const;

    CurveEndCaps                                         curveEndCaps() const;

    CurveType                                            curveType() const;

    static AccelerationStructureCurveGeometryDescriptor* descriptor();

    Buffer*                                              indexBuffer() const;
    NS::UInteger                                         indexBufferOffset() const;

    IndexType                                            indexType() const;

    AccelerationStructureCurveGeometryDescriptor*        init();

    Buffer*                                              radiusBuffer() const;
    NS::UInteger                                         radiusBufferOffset() const;

    AttributeFormat                                      radiusFormat() const;

    NS::UInteger                                         radiusStride() const;

    NS::UInteger                                         segmentControlPointCount() const;

    NS::UInteger                                         segmentCount() const;

    void                                                 setControlPointBuffer(const MTL::Buffer* controlPointBuffer);
    void                                                 setControlPointBufferOffset(NS::UInteger controlPointBufferOffset);

    void                                                 setControlPointCount(NS::UInteger controlPointCount);

    void                                                 setControlPointFormat(MTL::AttributeFormat controlPointFormat);

    void                                                 setControlPointStride(NS::UInteger controlPointStride);

    void                                                 setCurveBasis(MTL::CurveBasis curveBasis);

    void                                                 setCurveEndCaps(MTL::CurveEndCaps curveEndCaps);

    void                                                 setCurveType(MTL::CurveType curveType);

    void                                                 setIndexBuffer(const MTL::Buffer* indexBuffer);
    void                                                 setIndexBufferOffset(NS::UInteger indexBufferOffset);

    void                                                 setIndexType(MTL::IndexType indexType);

    void                                                 setRadiusBuffer(const MTL::Buffer* radiusBuffer);
    void                                                 setRadiusBufferOffset(NS::UInteger radiusBufferOffset);

    void                                                 setRadiusFormat(MTL::AttributeFormat radiusFormat);

    void                                                 setRadiusStride(NS::UInteger radiusStride);

    void                                                 setSegmentControlPointCount(NS::UInteger segmentControlPointCount);

    void                                                 setSegmentCount(NS::UInteger segmentCount);
};
class AccelerationStructureMotionCurveGeometryDescriptor : public NS::Copying<AccelerationStructureMotionCurveGeometryDescriptor, AccelerationStructureGeometryDescriptor>
{
public:
    static AccelerationStructureMotionCurveGeometryDescriptor* alloc();

    NS::Array*                                                 controlPointBuffers() const;

    NS::UInteger                                               controlPointCount() const;

    AttributeFormat                                            controlPointFormat() const;

    NS::UInteger                                               controlPointStride() const;

    CurveBasis                                                 curveBasis() const;

    CurveEndCaps                                               curveEndCaps() const;

    CurveType                                                  curveType() const;

    static AccelerationStructureMotionCurveGeometryDescriptor* descriptor();

    Buffer*                                                    indexBuffer() const;
    NS::UInteger                                               indexBufferOffset() const;

    IndexType                                                  indexType() const;

    AccelerationStructureMotionCurveGeometryDescriptor*        init();

    NS::Array*                                                 radiusBuffers() const;

    AttributeFormat                                            radiusFormat() const;

    NS::UInteger                                               radiusStride() const;

    NS::UInteger                                               segmentControlPointCount() const;

    NS::UInteger                                               segmentCount() const;

    void                                                       setControlPointBuffers(const NS::Array* controlPointBuffers);

    void                                                       setControlPointCount(NS::UInteger controlPointCount);

    void                                                       setControlPointFormat(MTL::AttributeFormat controlPointFormat);

    void                                                       setControlPointStride(NS::UInteger controlPointStride);

    void                                                       setCurveBasis(MTL::CurveBasis curveBasis);

    void                                                       setCurveEndCaps(MTL::CurveEndCaps curveEndCaps);

    void                                                       setCurveType(MTL::CurveType curveType);

    void                                                       setIndexBuffer(const MTL::Buffer* indexBuffer);
    void                                                       setIndexBufferOffset(NS::UInteger indexBufferOffset);

    void                                                       setIndexType(MTL::IndexType indexType);

    void                                                       setRadiusBuffers(const NS::Array* radiusBuffers);

    void                                                       setRadiusFormat(MTL::AttributeFormat radiusFormat);

    void                                                       setRadiusStride(NS::UInteger radiusStride);

    void                                                       setSegmentControlPointCount(NS::UInteger segmentControlPointCount);

    void                                                       setSegmentCount(NS::UInteger segmentCount);
};
class InstanceAccelerationStructureDescriptor : public NS::Copying<InstanceAccelerationStructureDescriptor, AccelerationStructureDescriptor>
{
public:
    static InstanceAccelerationStructureDescriptor* alloc();

    static InstanceAccelerationStructureDescriptor* descriptor();

    InstanceAccelerationStructureDescriptor*        init();

    NS::UInteger                                    instanceCount() const;

    Buffer*                                         instanceDescriptorBuffer() const;
    NS::UInteger                                    instanceDescriptorBufferOffset() const;

    NS::UInteger                                    instanceDescriptorStride() const;

    AccelerationStructureInstanceDescriptorType     instanceDescriptorType() const;

    MatrixLayout                                    instanceTransformationMatrixLayout() const;

    NS::Array*                                      instancedAccelerationStructures() const;

    Buffer*                                         motionTransformBuffer() const;
    NS::UInteger                                    motionTransformBufferOffset() const;

    NS::UInteger                                    motionTransformCount() const;

    NS::UInteger                                    motionTransformStride() const;

    TransformType                                   motionTransformType() const;

    void                                            setInstanceCount(NS::UInteger instanceCount);

    void                                            setInstanceDescriptorBuffer(const MTL::Buffer* instanceDescriptorBuffer);
    void                                            setInstanceDescriptorBufferOffset(NS::UInteger instanceDescriptorBufferOffset);

    void                                            setInstanceDescriptorStride(NS::UInteger instanceDescriptorStride);

    void                                            setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType);

    void                                            setInstanceTransformationMatrixLayout(MTL::MatrixLayout instanceTransformationMatrixLayout);

    void                                            setInstancedAccelerationStructures(const NS::Array* instancedAccelerationStructures);

    void                                            setMotionTransformBuffer(const MTL::Buffer* motionTransformBuffer);
    void                                            setMotionTransformBufferOffset(NS::UInteger motionTransformBufferOffset);

    void                                            setMotionTransformCount(NS::UInteger motionTransformCount);

    void                                            setMotionTransformStride(NS::UInteger motionTransformStride);

    void                                            setMotionTransformType(MTL::TransformType motionTransformType);
};
class IndirectInstanceAccelerationStructureDescriptor : public NS::Copying<IndirectInstanceAccelerationStructureDescriptor, AccelerationStructureDescriptor>
{
public:
    static IndirectInstanceAccelerationStructureDescriptor* alloc();

    static IndirectInstanceAccelerationStructureDescriptor* descriptor();

    IndirectInstanceAccelerationStructureDescriptor*        init();

    Buffer*                                                 instanceCountBuffer() const;
    NS::UInteger                                            instanceCountBufferOffset() const;

    Buffer*                                                 instanceDescriptorBuffer() const;
    NS::UInteger                                            instanceDescriptorBufferOffset() const;

    NS::UInteger                                            instanceDescriptorStride() const;

    AccelerationStructureInstanceDescriptorType             instanceDescriptorType() const;

    MatrixLayout                                            instanceTransformationMatrixLayout() const;

    NS::UInteger                                            maxInstanceCount() const;

    NS::UInteger                                            maxMotionTransformCount() const;

    Buffer*                                                 motionTransformBuffer() const;
    NS::UInteger                                            motionTransformBufferOffset() const;

    Buffer*                                                 motionTransformCountBuffer() const;
    NS::UInteger                                            motionTransformCountBufferOffset() const;

    NS::UInteger                                            motionTransformStride() const;

    TransformType                                           motionTransformType() const;

    void                                                    setInstanceCountBuffer(const MTL::Buffer* instanceCountBuffer);
    void                                                    setInstanceCountBufferOffset(NS::UInteger instanceCountBufferOffset);

    void                                                    setInstanceDescriptorBuffer(const MTL::Buffer* instanceDescriptorBuffer);
    void                                                    setInstanceDescriptorBufferOffset(NS::UInteger instanceDescriptorBufferOffset);

    void                                                    setInstanceDescriptorStride(NS::UInteger instanceDescriptorStride);

    void                                                    setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType);

    void                                                    setInstanceTransformationMatrixLayout(MTL::MatrixLayout instanceTransformationMatrixLayout);

    void                                                    setMaxInstanceCount(NS::UInteger maxInstanceCount);

    void                                                    setMaxMotionTransformCount(NS::UInteger maxMotionTransformCount);

    void                                                    setMotionTransformBuffer(const MTL::Buffer* motionTransformBuffer);
    void                                                    setMotionTransformBufferOffset(NS::UInteger motionTransformBufferOffset);

    void                                                    setMotionTransformCountBuffer(const MTL::Buffer* motionTransformCountBuffer);
    void                                                    setMotionTransformCountBufferOffset(NS::UInteger motionTransformCountBufferOffset);

    void                                                    setMotionTransformStride(NS::UInteger motionTransformStride);

    void                                                    setMotionTransformType(MTL::TransformType motionTransformType);
};
class AccelerationStructure : public NS::Referencing<AccelerationStructure, Resource>
{
public:
    ResourceID   gpuResourceID() const;

    NS::UInteger size() const;
};

}

_MTL_INLINE MTL::AccelerationStructureDescriptor* MTL::AccelerationStructureDescriptor::alloc()
{
    return NS::Object::alloc<MTL::AccelerationStructureDescriptor>(_MTL_PRIVATE_CLS(MTLAccelerationStructureDescriptor));
}

_MTL_INLINE MTL::AccelerationStructureDescriptor* MTL::AccelerationStructureDescriptor::init()
{
    return NS::Object::init<MTL::AccelerationStructureDescriptor>();
}

_MTL_INLINE void MTL::AccelerationStructureDescriptor::setUsage(MTL::AccelerationStructureUsage usage)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setUsage_), usage);
}

_MTL_INLINE MTL::AccelerationStructureUsage MTL::AccelerationStructureDescriptor::usage() const
{
    return Object::sendMessage<MTL::AccelerationStructureUsage>(this, _MTL_PRIVATE_SEL(usage));
}

_MTL_INLINE MTL::AccelerationStructureGeometryDescriptor* MTL::AccelerationStructureGeometryDescriptor::alloc()
{
    return NS::Object::alloc<MTL::AccelerationStructureGeometryDescriptor>(_MTL_PRIVATE_CLS(MTLAccelerationStructureGeometryDescriptor));
}

_MTL_INLINE bool MTL::AccelerationStructureGeometryDescriptor::allowDuplicateIntersectionFunctionInvocation() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(allowDuplicateIntersectionFunctionInvocation));
}

_MTL_INLINE MTL::AccelerationStructureGeometryDescriptor* MTL::AccelerationStructureGeometryDescriptor::init()
{
    return NS::Object::init<MTL::AccelerationStructureGeometryDescriptor>();
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureGeometryDescriptor::intersectionFunctionTableOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(intersectionFunctionTableOffset));
}

_MTL_INLINE NS::String* MTL::AccelerationStructureGeometryDescriptor::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE bool MTL::AccelerationStructureGeometryDescriptor::opaque() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(opaque));
}

_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureGeometryDescriptor::primitiveDataBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(primitiveDataBuffer));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureGeometryDescriptor::primitiveDataBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(primitiveDataBufferOffset));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureGeometryDescriptor::primitiveDataElementSize() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(primitiveDataElementSize));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureGeometryDescriptor::primitiveDataStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(primitiveDataStride));
}

_MTL_INLINE void MTL::AccelerationStructureGeometryDescriptor::setAllowDuplicateIntersectionFunctionInvocation(bool allowDuplicateIntersectionFunctionInvocation)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setAllowDuplicateIntersectionFunctionInvocation_), allowDuplicateIntersectionFunctionInvocation);
}

_MTL_INLINE void MTL::AccelerationStructureGeometryDescriptor::setIntersectionFunctionTableOffset(NS::UInteger intersectionFunctionTableOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIntersectionFunctionTableOffset_), intersectionFunctionTableOffset);
}

_MTL_INLINE void MTL::AccelerationStructureGeometryDescriptor::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE void MTL::AccelerationStructureGeometryDescriptor::setOpaque(bool opaque)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setOpaque_), opaque);
}

_MTL_INLINE void MTL::AccelerationStructureGeometryDescriptor::setPrimitiveDataBuffer(const MTL::Buffer* primitiveDataBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPrimitiveDataBuffer_), primitiveDataBuffer);
}

_MTL_INLINE void MTL::AccelerationStructureGeometryDescriptor::setPrimitiveDataBufferOffset(NS::UInteger primitiveDataBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPrimitiveDataBufferOffset_), primitiveDataBufferOffset);
}

_MTL_INLINE void MTL::AccelerationStructureGeometryDescriptor::setPrimitiveDataElementSize(NS::UInteger primitiveDataElementSize)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPrimitiveDataElementSize_), primitiveDataElementSize);
}

_MTL_INLINE void MTL::AccelerationStructureGeometryDescriptor::setPrimitiveDataStride(NS::UInteger primitiveDataStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setPrimitiveDataStride_), primitiveDataStride);
}

_MTL_INLINE MTL::PrimitiveAccelerationStructureDescriptor* MTL::PrimitiveAccelerationStructureDescriptor::alloc()
{
    return NS::Object::alloc<MTL::PrimitiveAccelerationStructureDescriptor>(_MTL_PRIVATE_CLS(MTLPrimitiveAccelerationStructureDescriptor));
}

_MTL_INLINE MTL::PrimitiveAccelerationStructureDescriptor* MTL::PrimitiveAccelerationStructureDescriptor::descriptor()
{
    return Object::sendMessage<MTL::PrimitiveAccelerationStructureDescriptor*>(_MTL_PRIVATE_CLS(MTLPrimitiveAccelerationStructureDescriptor), _MTL_PRIVATE_SEL(descriptor));
}

_MTL_INLINE NS::Array* MTL::PrimitiveAccelerationStructureDescriptor::geometryDescriptors() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(geometryDescriptors));
}

_MTL_INLINE MTL::PrimitiveAccelerationStructureDescriptor* MTL::PrimitiveAccelerationStructureDescriptor::init()
{
    return NS::Object::init<MTL::PrimitiveAccelerationStructureDescriptor>();
}

_MTL_INLINE MTL::MotionBorderMode MTL::PrimitiveAccelerationStructureDescriptor::motionEndBorderMode() const
{
    return Object::sendMessage<MTL::MotionBorderMode>(this, _MTL_PRIVATE_SEL(motionEndBorderMode));
}

_MTL_INLINE float MTL::PrimitiveAccelerationStructureDescriptor::motionEndTime() const
{
    return Object::sendMessage<float>(this, _MTL_PRIVATE_SEL(motionEndTime));
}

_MTL_INLINE NS::UInteger MTL::PrimitiveAccelerationStructureDescriptor::motionKeyframeCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(motionKeyframeCount));
}

_MTL_INLINE MTL::MotionBorderMode MTL::PrimitiveAccelerationStructureDescriptor::motionStartBorderMode() const
{
    return Object::sendMessage<MTL::MotionBorderMode>(this, _MTL_PRIVATE_SEL(motionStartBorderMode));
}

_MTL_INLINE float MTL::PrimitiveAccelerationStructureDescriptor::motionStartTime() const
{
    return Object::sendMessage<float>(this, _MTL_PRIVATE_SEL(motionStartTime));
}

_MTL_INLINE void MTL::PrimitiveAccelerationStructureDescriptor::setGeometryDescriptors(const NS::Array* geometryDescriptors)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setGeometryDescriptors_), geometryDescriptors);
}

_MTL_INLINE void MTL::PrimitiveAccelerationStructureDescriptor::setMotionEndBorderMode(MTL::MotionBorderMode motionEndBorderMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionEndBorderMode_), motionEndBorderMode);
}

_MTL_INLINE void MTL::PrimitiveAccelerationStructureDescriptor::setMotionEndTime(float motionEndTime)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionEndTime_), motionEndTime);
}

_MTL_INLINE void MTL::PrimitiveAccelerationStructureDescriptor::setMotionKeyframeCount(NS::UInteger motionKeyframeCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionKeyframeCount_), motionKeyframeCount);
}

_MTL_INLINE void MTL::PrimitiveAccelerationStructureDescriptor::setMotionStartBorderMode(MTL::MotionBorderMode motionStartBorderMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionStartBorderMode_), motionStartBorderMode);
}

_MTL_INLINE void MTL::PrimitiveAccelerationStructureDescriptor::setMotionStartTime(float motionStartTime)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionStartTime_), motionStartTime);
}

_MTL_INLINE MTL::AccelerationStructureTriangleGeometryDescriptor* MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()
{
    return NS::Object::alloc<MTL::AccelerationStructureTriangleGeometryDescriptor>(_MTL_PRIVATE_CLS(MTLAccelerationStructureTriangleGeometryDescriptor));
}

_MTL_INLINE MTL::AccelerationStructureTriangleGeometryDescriptor* MTL::AccelerationStructureTriangleGeometryDescriptor::descriptor()
{
    return Object::sendMessage<MTL::AccelerationStructureTriangleGeometryDescriptor*>(_MTL_PRIVATE_CLS(MTLAccelerationStructureTriangleGeometryDescriptor), _MTL_PRIVATE_SEL(descriptor));
}

_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureTriangleGeometryDescriptor::indexBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(indexBuffer));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureTriangleGeometryDescriptor::indexBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(indexBufferOffset));
}

_MTL_INLINE MTL::IndexType MTL::AccelerationStructureTriangleGeometryDescriptor::indexType() const
{
    return Object::sendMessage<MTL::IndexType>(this, _MTL_PRIVATE_SEL(indexType));
}

_MTL_INLINE MTL::AccelerationStructureTriangleGeometryDescriptor* MTL::AccelerationStructureTriangleGeometryDescriptor::init()
{
    return NS::Object::init<MTL::AccelerationStructureTriangleGeometryDescriptor>();
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setIndexBuffer(const MTL::Buffer* indexBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexBuffer_), indexBuffer);
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setIndexBufferOffset(NS::UInteger indexBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexBufferOffset_), indexBufferOffset);
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setIndexType(MTL::IndexType indexType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexType_), indexType);
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setTransformationMatrixBuffer(const MTL::Buffer* transformationMatrixBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTransformationMatrixBuffer_), transformationMatrixBuffer);
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setTransformationMatrixBufferOffset(NS::UInteger transformationMatrixBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTransformationMatrixBufferOffset_), transformationMatrixBufferOffset);
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setTransformationMatrixLayout(MTL::MatrixLayout transformationMatrixLayout)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTransformationMatrixLayout_), transformationMatrixLayout);
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setTriangleCount(NS::UInteger triangleCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTriangleCount_), triangleCount);
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setVertexBuffer(const MTL::Buffer* vertexBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexBuffer_), vertexBuffer);
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setVertexBufferOffset(NS::UInteger vertexBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexBufferOffset_), vertexBufferOffset);
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setVertexFormat(MTL::AttributeFormat vertexFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexFormat_), vertexFormat);
}

_MTL_INLINE void MTL::AccelerationStructureTriangleGeometryDescriptor::setVertexStride(NS::UInteger vertexStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexStride_), vertexStride);
}

_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureTriangleGeometryDescriptor::transformationMatrixBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(transformationMatrixBuffer));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureTriangleGeometryDescriptor::transformationMatrixBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(transformationMatrixBufferOffset));
}

_MTL_INLINE MTL::MatrixLayout MTL::AccelerationStructureTriangleGeometryDescriptor::transformationMatrixLayout() const
{
    return Object::sendMessage<MTL::MatrixLayout>(this, _MTL_PRIVATE_SEL(transformationMatrixLayout));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureTriangleGeometryDescriptor::triangleCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(triangleCount));
}

_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureTriangleGeometryDescriptor::vertexBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(vertexBuffer));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureTriangleGeometryDescriptor::vertexBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(vertexBufferOffset));
}

_MTL_INLINE MTL::AttributeFormat MTL::AccelerationStructureTriangleGeometryDescriptor::vertexFormat() const
{
    return Object::sendMessage<MTL::AttributeFormat>(this, _MTL_PRIVATE_SEL(vertexFormat));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureTriangleGeometryDescriptor::vertexStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(vertexStride));
}

_MTL_INLINE MTL::AccelerationStructureBoundingBoxGeometryDescriptor* MTL::AccelerationStructureBoundingBoxGeometryDescriptor::alloc()
{
    return NS::Object::alloc<MTL::AccelerationStructureBoundingBoxGeometryDescriptor>(_MTL_PRIVATE_CLS(MTLAccelerationStructureBoundingBoxGeometryDescriptor));
}

_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureBoundingBoxGeometryDescriptor::boundingBoxBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(boundingBoxBuffer));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureBoundingBoxGeometryDescriptor::boundingBoxBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(boundingBoxBufferOffset));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureBoundingBoxGeometryDescriptor::boundingBoxCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(boundingBoxCount));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureBoundingBoxGeometryDescriptor::boundingBoxStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(boundingBoxStride));
}

_MTL_INLINE MTL::AccelerationStructureBoundingBoxGeometryDescriptor* MTL::AccelerationStructureBoundingBoxGeometryDescriptor::descriptor()
{
    return Object::sendMessage<MTL::AccelerationStructureBoundingBoxGeometryDescriptor*>(_MTL_PRIVATE_CLS(MTLAccelerationStructureBoundingBoxGeometryDescriptor), _MTL_PRIVATE_SEL(descriptor));
}

_MTL_INLINE MTL::AccelerationStructureBoundingBoxGeometryDescriptor* MTL::AccelerationStructureBoundingBoxGeometryDescriptor::init()
{
    return NS::Object::init<MTL::AccelerationStructureBoundingBoxGeometryDescriptor>();
}

_MTL_INLINE void MTL::AccelerationStructureBoundingBoxGeometryDescriptor::setBoundingBoxBuffer(const MTL::Buffer* boundingBoxBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBoundingBoxBuffer_), boundingBoxBuffer);
}

_MTL_INLINE void MTL::AccelerationStructureBoundingBoxGeometryDescriptor::setBoundingBoxBufferOffset(NS::UInteger boundingBoxBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBoundingBoxBufferOffset_), boundingBoxBufferOffset);
}

_MTL_INLINE void MTL::AccelerationStructureBoundingBoxGeometryDescriptor::setBoundingBoxCount(NS::UInteger boundingBoxCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBoundingBoxCount_), boundingBoxCount);
}

_MTL_INLINE void MTL::AccelerationStructureBoundingBoxGeometryDescriptor::setBoundingBoxStride(NS::UInteger boundingBoxStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBoundingBoxStride_), boundingBoxStride);
}

_MTL_INLINE MTL::MotionKeyframeData* MTL::MotionKeyframeData::alloc()
{
    return NS::Object::alloc<MTL::MotionKeyframeData>(_MTL_PRIVATE_CLS(MTLMotionKeyframeData));
}

_MTL_INLINE MTL::Buffer* MTL::MotionKeyframeData::buffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(buffer));
}

_MTL_INLINE MTL::MotionKeyframeData* MTL::MotionKeyframeData::data()
{
    return Object::sendMessage<MTL::MotionKeyframeData*>(_MTL_PRIVATE_CLS(MTLMotionKeyframeData), _MTL_PRIVATE_SEL(data));
}

_MTL_INLINE MTL::MotionKeyframeData* MTL::MotionKeyframeData::init()
{
    return NS::Object::init<MTL::MotionKeyframeData>();
}

_MTL_INLINE NS::UInteger MTL::MotionKeyframeData::offset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(offset));
}

_MTL_INLINE void MTL::MotionKeyframeData::setBuffer(const MTL::Buffer* buffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBuffer_), buffer);
}

_MTL_INLINE void MTL::MotionKeyframeData::setOffset(NS::UInteger offset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setOffset_), offset);
}

_MTL_INLINE MTL::AccelerationStructureMotionTriangleGeometryDescriptor* MTL::AccelerationStructureMotionTriangleGeometryDescriptor::alloc()
{
    return NS::Object::alloc<MTL::AccelerationStructureMotionTriangleGeometryDescriptor>(_MTL_PRIVATE_CLS(MTLAccelerationStructureMotionTriangleGeometryDescriptor));
}

_MTL_INLINE MTL::AccelerationStructureMotionTriangleGeometryDescriptor* MTL::AccelerationStructureMotionTriangleGeometryDescriptor::descriptor()
{
    return Object::sendMessage<MTL::AccelerationStructureMotionTriangleGeometryDescriptor*>(_MTL_PRIVATE_CLS(MTLAccelerationStructureMotionTriangleGeometryDescriptor), _MTL_PRIVATE_SEL(descriptor));
}

_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureMotionTriangleGeometryDescriptor::indexBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(indexBuffer));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionTriangleGeometryDescriptor::indexBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(indexBufferOffset));
}

_MTL_INLINE MTL::IndexType MTL::AccelerationStructureMotionTriangleGeometryDescriptor::indexType() const
{
    return Object::sendMessage<MTL::IndexType>(this, _MTL_PRIVATE_SEL(indexType));
}

_MTL_INLINE MTL::AccelerationStructureMotionTriangleGeometryDescriptor* MTL::AccelerationStructureMotionTriangleGeometryDescriptor::init()
{
    return NS::Object::init<MTL::AccelerationStructureMotionTriangleGeometryDescriptor>();
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setIndexBuffer(const MTL::Buffer* indexBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexBuffer_), indexBuffer);
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setIndexBufferOffset(NS::UInteger indexBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexBufferOffset_), indexBufferOffset);
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setIndexType(MTL::IndexType indexType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexType_), indexType);
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setTransformationMatrixBuffer(const MTL::Buffer* transformationMatrixBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTransformationMatrixBuffer_), transformationMatrixBuffer);
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setTransformationMatrixBufferOffset(NS::UInteger transformationMatrixBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTransformationMatrixBufferOffset_), transformationMatrixBufferOffset);
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setTransformationMatrixLayout(MTL::MatrixLayout transformationMatrixLayout)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTransformationMatrixLayout_), transformationMatrixLayout);
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setTriangleCount(NS::UInteger triangleCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setTriangleCount_), triangleCount);
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setVertexBuffers(const NS::Array* vertexBuffers)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexBuffers_), vertexBuffers);
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setVertexFormat(MTL::AttributeFormat vertexFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexFormat_), vertexFormat);
}

_MTL_INLINE void MTL::AccelerationStructureMotionTriangleGeometryDescriptor::setVertexStride(NS::UInteger vertexStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setVertexStride_), vertexStride);
}

_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureMotionTriangleGeometryDescriptor::transformationMatrixBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(transformationMatrixBuffer));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionTriangleGeometryDescriptor::transformationMatrixBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(transformationMatrixBufferOffset));
}

_MTL_INLINE MTL::MatrixLayout MTL::AccelerationStructureMotionTriangleGeometryDescriptor::transformationMatrixLayout() const
{
    return Object::sendMessage<MTL::MatrixLayout>(this, _MTL_PRIVATE_SEL(transformationMatrixLayout));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionTriangleGeometryDescriptor::triangleCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(triangleCount));
}

_MTL_INLINE NS::Array* MTL::AccelerationStructureMotionTriangleGeometryDescriptor::vertexBuffers() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(vertexBuffers));
}

_MTL_INLINE MTL::AttributeFormat MTL::AccelerationStructureMotionTriangleGeometryDescriptor::vertexFormat() const
{
    return Object::sendMessage<MTL::AttributeFormat>(this, _MTL_PRIVATE_SEL(vertexFormat));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionTriangleGeometryDescriptor::vertexStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(vertexStride));
}

_MTL_INLINE MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor* MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::alloc()
{
    return NS::Object::alloc<MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor>(_MTL_PRIVATE_CLS(MTLAccelerationStructureMotionBoundingBoxGeometryDescriptor));
}

_MTL_INLINE NS::Array* MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::boundingBoxBuffers() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(boundingBoxBuffers));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::boundingBoxCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(boundingBoxCount));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::boundingBoxStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(boundingBoxStride));
}

_MTL_INLINE MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor* MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::descriptor()
{
    return Object::sendMessage<MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor*>(_MTL_PRIVATE_CLS(MTLAccelerationStructureMotionBoundingBoxGeometryDescriptor), _MTL_PRIVATE_SEL(descriptor));
}

_MTL_INLINE MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor* MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::init()
{
    return NS::Object::init<MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor>();
}

_MTL_INLINE void MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::setBoundingBoxBuffers(const NS::Array* boundingBoxBuffers)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBoundingBoxBuffers_), boundingBoxBuffers);
}

_MTL_INLINE void MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::setBoundingBoxCount(NS::UInteger boundingBoxCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBoundingBoxCount_), boundingBoxCount);
}

_MTL_INLINE void MTL::AccelerationStructureMotionBoundingBoxGeometryDescriptor::setBoundingBoxStride(NS::UInteger boundingBoxStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBoundingBoxStride_), boundingBoxStride);
}

_MTL_INLINE MTL::AccelerationStructureCurveGeometryDescriptor* MTL::AccelerationStructureCurveGeometryDescriptor::alloc()
{
    return NS::Object::alloc<MTL::AccelerationStructureCurveGeometryDescriptor>(_MTL_PRIVATE_CLS(MTLAccelerationStructureCurveGeometryDescriptor));
}

_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureCurveGeometryDescriptor::controlPointBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(controlPointBuffer));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureCurveGeometryDescriptor::controlPointBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(controlPointBufferOffset));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureCurveGeometryDescriptor::controlPointCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(controlPointCount));
}

_MTL_INLINE MTL::AttributeFormat MTL::AccelerationStructureCurveGeometryDescriptor::controlPointFormat() const
{
    return Object::sendMessage<MTL::AttributeFormat>(this, _MTL_PRIVATE_SEL(controlPointFormat));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureCurveGeometryDescriptor::controlPointStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(controlPointStride));
}

_MTL_INLINE MTL::CurveBasis MTL::AccelerationStructureCurveGeometryDescriptor::curveBasis() const
{
    return Object::sendMessage<MTL::CurveBasis>(this, _MTL_PRIVATE_SEL(curveBasis));
}

_MTL_INLINE MTL::CurveEndCaps MTL::AccelerationStructureCurveGeometryDescriptor::curveEndCaps() const
{
    return Object::sendMessage<MTL::CurveEndCaps>(this, _MTL_PRIVATE_SEL(curveEndCaps));
}

_MTL_INLINE MTL::CurveType MTL::AccelerationStructureCurveGeometryDescriptor::curveType() const
{
    return Object::sendMessage<MTL::CurveType>(this, _MTL_PRIVATE_SEL(curveType));
}

_MTL_INLINE MTL::AccelerationStructureCurveGeometryDescriptor* MTL::AccelerationStructureCurveGeometryDescriptor::descriptor()
{
    return Object::sendMessage<MTL::AccelerationStructureCurveGeometryDescriptor*>(_MTL_PRIVATE_CLS(MTLAccelerationStructureCurveGeometryDescriptor), _MTL_PRIVATE_SEL(descriptor));
}

_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureCurveGeometryDescriptor::indexBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(indexBuffer));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureCurveGeometryDescriptor::indexBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(indexBufferOffset));
}

_MTL_INLINE MTL::IndexType MTL::AccelerationStructureCurveGeometryDescriptor::indexType() const
{
    return Object::sendMessage<MTL::IndexType>(this, _MTL_PRIVATE_SEL(indexType));
}

_MTL_INLINE MTL::AccelerationStructureCurveGeometryDescriptor* MTL::AccelerationStructureCurveGeometryDescriptor::init()
{
    return NS::Object::init<MTL::AccelerationStructureCurveGeometryDescriptor>();
}

_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureCurveGeometryDescriptor::radiusBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(radiusBuffer));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureCurveGeometryDescriptor::radiusBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(radiusBufferOffset));
}

_MTL_INLINE MTL::AttributeFormat MTL::AccelerationStructureCurveGeometryDescriptor::radiusFormat() const
{
    return Object::sendMessage<MTL::AttributeFormat>(this, _MTL_PRIVATE_SEL(radiusFormat));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureCurveGeometryDescriptor::radiusStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(radiusStride));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureCurveGeometryDescriptor::segmentControlPointCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(segmentControlPointCount));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureCurveGeometryDescriptor::segmentCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(segmentCount));
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setControlPointBuffer(const MTL::Buffer* controlPointBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointBuffer_), controlPointBuffer);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setControlPointBufferOffset(NS::UInteger controlPointBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointBufferOffset_), controlPointBufferOffset);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setControlPointCount(NS::UInteger controlPointCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointCount_), controlPointCount);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setControlPointFormat(MTL::AttributeFormat controlPointFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointFormat_), controlPointFormat);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setControlPointStride(NS::UInteger controlPointStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointStride_), controlPointStride);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setCurveBasis(MTL::CurveBasis curveBasis)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCurveBasis_), curveBasis);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setCurveEndCaps(MTL::CurveEndCaps curveEndCaps)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCurveEndCaps_), curveEndCaps);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setCurveType(MTL::CurveType curveType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCurveType_), curveType);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setIndexBuffer(const MTL::Buffer* indexBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexBuffer_), indexBuffer);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setIndexBufferOffset(NS::UInteger indexBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexBufferOffset_), indexBufferOffset);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setIndexType(MTL::IndexType indexType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexType_), indexType);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setRadiusBuffer(const MTL::Buffer* radiusBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRadiusBuffer_), radiusBuffer);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setRadiusBufferOffset(NS::UInteger radiusBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRadiusBufferOffset_), radiusBufferOffset);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setRadiusFormat(MTL::AttributeFormat radiusFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRadiusFormat_), radiusFormat);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setRadiusStride(NS::UInteger radiusStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRadiusStride_), radiusStride);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setSegmentControlPointCount(NS::UInteger segmentControlPointCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSegmentControlPointCount_), segmentControlPointCount);
}

_MTL_INLINE void MTL::AccelerationStructureCurveGeometryDescriptor::setSegmentCount(NS::UInteger segmentCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSegmentCount_), segmentCount);
}

_MTL_INLINE MTL::AccelerationStructureMotionCurveGeometryDescriptor* MTL::AccelerationStructureMotionCurveGeometryDescriptor::alloc()
{
    return NS::Object::alloc<MTL::AccelerationStructureMotionCurveGeometryDescriptor>(_MTL_PRIVATE_CLS(MTLAccelerationStructureMotionCurveGeometryDescriptor));
}

_MTL_INLINE NS::Array* MTL::AccelerationStructureMotionCurveGeometryDescriptor::controlPointBuffers() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(controlPointBuffers));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionCurveGeometryDescriptor::controlPointCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(controlPointCount));
}

_MTL_INLINE MTL::AttributeFormat MTL::AccelerationStructureMotionCurveGeometryDescriptor::controlPointFormat() const
{
    return Object::sendMessage<MTL::AttributeFormat>(this, _MTL_PRIVATE_SEL(controlPointFormat));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionCurveGeometryDescriptor::controlPointStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(controlPointStride));
}

_MTL_INLINE MTL::CurveBasis MTL::AccelerationStructureMotionCurveGeometryDescriptor::curveBasis() const
{
    return Object::sendMessage<MTL::CurveBasis>(this, _MTL_PRIVATE_SEL(curveBasis));
}

_MTL_INLINE MTL::CurveEndCaps MTL::AccelerationStructureMotionCurveGeometryDescriptor::curveEndCaps() const
{
    return Object::sendMessage<MTL::CurveEndCaps>(this, _MTL_PRIVATE_SEL(curveEndCaps));
}

_MTL_INLINE MTL::CurveType MTL::AccelerationStructureMotionCurveGeometryDescriptor::curveType() const
{
    return Object::sendMessage<MTL::CurveType>(this, _MTL_PRIVATE_SEL(curveType));
}

_MTL_INLINE MTL::AccelerationStructureMotionCurveGeometryDescriptor* MTL::AccelerationStructureMotionCurveGeometryDescriptor::descriptor()
{
    return Object::sendMessage<MTL::AccelerationStructureMotionCurveGeometryDescriptor*>(_MTL_PRIVATE_CLS(MTLAccelerationStructureMotionCurveGeometryDescriptor), _MTL_PRIVATE_SEL(descriptor));
}

_MTL_INLINE MTL::Buffer* MTL::AccelerationStructureMotionCurveGeometryDescriptor::indexBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(indexBuffer));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionCurveGeometryDescriptor::indexBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(indexBufferOffset));
}

_MTL_INLINE MTL::IndexType MTL::AccelerationStructureMotionCurveGeometryDescriptor::indexType() const
{
    return Object::sendMessage<MTL::IndexType>(this, _MTL_PRIVATE_SEL(indexType));
}

_MTL_INLINE MTL::AccelerationStructureMotionCurveGeometryDescriptor* MTL::AccelerationStructureMotionCurveGeometryDescriptor::init()
{
    return NS::Object::init<MTL::AccelerationStructureMotionCurveGeometryDescriptor>();
}

_MTL_INLINE NS::Array* MTL::AccelerationStructureMotionCurveGeometryDescriptor::radiusBuffers() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(radiusBuffers));
}

_MTL_INLINE MTL::AttributeFormat MTL::AccelerationStructureMotionCurveGeometryDescriptor::radiusFormat() const
{
    return Object::sendMessage<MTL::AttributeFormat>(this, _MTL_PRIVATE_SEL(radiusFormat));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionCurveGeometryDescriptor::radiusStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(radiusStride));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionCurveGeometryDescriptor::segmentControlPointCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(segmentControlPointCount));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructureMotionCurveGeometryDescriptor::segmentCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(segmentCount));
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setControlPointBuffers(const NS::Array* controlPointBuffers)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointBuffers_), controlPointBuffers);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setControlPointCount(NS::UInteger controlPointCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointCount_), controlPointCount);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setControlPointFormat(MTL::AttributeFormat controlPointFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointFormat_), controlPointFormat);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setControlPointStride(NS::UInteger controlPointStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setControlPointStride_), controlPointStride);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setCurveBasis(MTL::CurveBasis curveBasis)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCurveBasis_), curveBasis);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setCurveEndCaps(MTL::CurveEndCaps curveEndCaps)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCurveEndCaps_), curveEndCaps);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setCurveType(MTL::CurveType curveType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCurveType_), curveType);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setIndexBuffer(const MTL::Buffer* indexBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexBuffer_), indexBuffer);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setIndexBufferOffset(NS::UInteger indexBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexBufferOffset_), indexBufferOffset);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setIndexType(MTL::IndexType indexType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setIndexType_), indexType);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setRadiusBuffers(const NS::Array* radiusBuffers)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRadiusBuffers_), radiusBuffers);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setRadiusFormat(MTL::AttributeFormat radiusFormat)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRadiusFormat_), radiusFormat);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setRadiusStride(NS::UInteger radiusStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setRadiusStride_), radiusStride);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setSegmentControlPointCount(NS::UInteger segmentControlPointCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSegmentControlPointCount_), segmentControlPointCount);
}

_MTL_INLINE void MTL::AccelerationStructureMotionCurveGeometryDescriptor::setSegmentCount(NS::UInteger segmentCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSegmentCount_), segmentCount);
}

_MTL_INLINE MTL::InstanceAccelerationStructureDescriptor* MTL::InstanceAccelerationStructureDescriptor::alloc()
{
    return NS::Object::alloc<MTL::InstanceAccelerationStructureDescriptor>(_MTL_PRIVATE_CLS(MTLInstanceAccelerationStructureDescriptor));
}

_MTL_INLINE MTL::InstanceAccelerationStructureDescriptor* MTL::InstanceAccelerationStructureDescriptor::descriptor()
{
    return Object::sendMessage<MTL::InstanceAccelerationStructureDescriptor*>(_MTL_PRIVATE_CLS(MTLInstanceAccelerationStructureDescriptor), _MTL_PRIVATE_SEL(descriptor));
}

_MTL_INLINE MTL::InstanceAccelerationStructureDescriptor* MTL::InstanceAccelerationStructureDescriptor::init()
{
    return NS::Object::init<MTL::InstanceAccelerationStructureDescriptor>();
}

_MTL_INLINE NS::UInteger MTL::InstanceAccelerationStructureDescriptor::instanceCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(instanceCount));
}

_MTL_INLINE MTL::Buffer* MTL::InstanceAccelerationStructureDescriptor::instanceDescriptorBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(instanceDescriptorBuffer));
}

_MTL_INLINE NS::UInteger MTL::InstanceAccelerationStructureDescriptor::instanceDescriptorBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(instanceDescriptorBufferOffset));
}

_MTL_INLINE NS::UInteger MTL::InstanceAccelerationStructureDescriptor::instanceDescriptorStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(instanceDescriptorStride));
}

_MTL_INLINE MTL::AccelerationStructureInstanceDescriptorType MTL::InstanceAccelerationStructureDescriptor::instanceDescriptorType() const
{
    return Object::sendMessage<MTL::AccelerationStructureInstanceDescriptorType>(this, _MTL_PRIVATE_SEL(instanceDescriptorType));
}

_MTL_INLINE MTL::MatrixLayout MTL::InstanceAccelerationStructureDescriptor::instanceTransformationMatrixLayout() const
{
    return Object::sendMessage<MTL::MatrixLayout>(this, _MTL_PRIVATE_SEL(instanceTransformationMatrixLayout));
}

_MTL_INLINE NS::Array* MTL::InstanceAccelerationStructureDescriptor::instancedAccelerationStructures() const
{
    return Object::sendMessage<NS::Array*>(this, _MTL_PRIVATE_SEL(instancedAccelerationStructures));
}

_MTL_INLINE MTL::Buffer* MTL::InstanceAccelerationStructureDescriptor::motionTransformBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(motionTransformBuffer));
}

_MTL_INLINE NS::UInteger MTL::InstanceAccelerationStructureDescriptor::motionTransformBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(motionTransformBufferOffset));
}

_MTL_INLINE NS::UInteger MTL::InstanceAccelerationStructureDescriptor::motionTransformCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(motionTransformCount));
}

_MTL_INLINE NS::UInteger MTL::InstanceAccelerationStructureDescriptor::motionTransformStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(motionTransformStride));
}

_MTL_INLINE MTL::TransformType MTL::InstanceAccelerationStructureDescriptor::motionTransformType() const
{
    return Object::sendMessage<MTL::TransformType>(this, _MTL_PRIVATE_SEL(motionTransformType));
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setInstanceCount(NS::UInteger instanceCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceCount_), instanceCount);
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setInstanceDescriptorBuffer(const MTL::Buffer* instanceDescriptorBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceDescriptorBuffer_), instanceDescriptorBuffer);
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setInstanceDescriptorBufferOffset(NS::UInteger instanceDescriptorBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceDescriptorBufferOffset_), instanceDescriptorBufferOffset);
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setInstanceDescriptorStride(NS::UInteger instanceDescriptorStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceDescriptorStride_), instanceDescriptorStride);
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceDescriptorType_), instanceDescriptorType);
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setInstanceTransformationMatrixLayout(MTL::MatrixLayout instanceTransformationMatrixLayout)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceTransformationMatrixLayout_), instanceTransformationMatrixLayout);
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setInstancedAccelerationStructures(const NS::Array* instancedAccelerationStructures)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstancedAccelerationStructures_), instancedAccelerationStructures);
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setMotionTransformBuffer(const MTL::Buffer* motionTransformBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformBuffer_), motionTransformBuffer);
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setMotionTransformBufferOffset(NS::UInteger motionTransformBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformBufferOffset_), motionTransformBufferOffset);
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setMotionTransformCount(NS::UInteger motionTransformCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformCount_), motionTransformCount);
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setMotionTransformStride(NS::UInteger motionTransformStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformStride_), motionTransformStride);
}

_MTL_INLINE void MTL::InstanceAccelerationStructureDescriptor::setMotionTransformType(MTL::TransformType motionTransformType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformType_), motionTransformType);
}

_MTL_INLINE MTL::IndirectInstanceAccelerationStructureDescriptor* MTL::IndirectInstanceAccelerationStructureDescriptor::alloc()
{
    return NS::Object::alloc<MTL::IndirectInstanceAccelerationStructureDescriptor>(_MTL_PRIVATE_CLS(MTLIndirectInstanceAccelerationStructureDescriptor));
}

_MTL_INLINE MTL::IndirectInstanceAccelerationStructureDescriptor* MTL::IndirectInstanceAccelerationStructureDescriptor::descriptor()
{
    return Object::sendMessage<MTL::IndirectInstanceAccelerationStructureDescriptor*>(_MTL_PRIVATE_CLS(MTLIndirectInstanceAccelerationStructureDescriptor), _MTL_PRIVATE_SEL(descriptor));
}

_MTL_INLINE MTL::IndirectInstanceAccelerationStructureDescriptor* MTL::IndirectInstanceAccelerationStructureDescriptor::init()
{
    return NS::Object::init<MTL::IndirectInstanceAccelerationStructureDescriptor>();
}

_MTL_INLINE MTL::Buffer* MTL::IndirectInstanceAccelerationStructureDescriptor::instanceCountBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(instanceCountBuffer));
}

_MTL_INLINE NS::UInteger MTL::IndirectInstanceAccelerationStructureDescriptor::instanceCountBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(instanceCountBufferOffset));
}

_MTL_INLINE MTL::Buffer* MTL::IndirectInstanceAccelerationStructureDescriptor::instanceDescriptorBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(instanceDescriptorBuffer));
}

_MTL_INLINE NS::UInteger MTL::IndirectInstanceAccelerationStructureDescriptor::instanceDescriptorBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(instanceDescriptorBufferOffset));
}

_MTL_INLINE NS::UInteger MTL::IndirectInstanceAccelerationStructureDescriptor::instanceDescriptorStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(instanceDescriptorStride));
}

_MTL_INLINE MTL::AccelerationStructureInstanceDescriptorType MTL::IndirectInstanceAccelerationStructureDescriptor::instanceDescriptorType() const
{
    return Object::sendMessage<MTL::AccelerationStructureInstanceDescriptorType>(this, _MTL_PRIVATE_SEL(instanceDescriptorType));
}

_MTL_INLINE MTL::MatrixLayout MTL::IndirectInstanceAccelerationStructureDescriptor::instanceTransformationMatrixLayout() const
{
    return Object::sendMessage<MTL::MatrixLayout>(this, _MTL_PRIVATE_SEL(instanceTransformationMatrixLayout));
}

_MTL_INLINE NS::UInteger MTL::IndirectInstanceAccelerationStructureDescriptor::maxInstanceCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxInstanceCount));
}

_MTL_INLINE NS::UInteger MTL::IndirectInstanceAccelerationStructureDescriptor::maxMotionTransformCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(maxMotionTransformCount));
}

_MTL_INLINE MTL::Buffer* MTL::IndirectInstanceAccelerationStructureDescriptor::motionTransformBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(motionTransformBuffer));
}

_MTL_INLINE NS::UInteger MTL::IndirectInstanceAccelerationStructureDescriptor::motionTransformBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(motionTransformBufferOffset));
}

_MTL_INLINE MTL::Buffer* MTL::IndirectInstanceAccelerationStructureDescriptor::motionTransformCountBuffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(motionTransformCountBuffer));
}

_MTL_INLINE NS::UInteger MTL::IndirectInstanceAccelerationStructureDescriptor::motionTransformCountBufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(motionTransformCountBufferOffset));
}

_MTL_INLINE NS::UInteger MTL::IndirectInstanceAccelerationStructureDescriptor::motionTransformStride() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(motionTransformStride));
}

_MTL_INLINE MTL::TransformType MTL::IndirectInstanceAccelerationStructureDescriptor::motionTransformType() const
{
    return Object::sendMessage<MTL::TransformType>(this, _MTL_PRIVATE_SEL(motionTransformType));
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setInstanceCountBuffer(const MTL::Buffer* instanceCountBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceCountBuffer_), instanceCountBuffer);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setInstanceCountBufferOffset(NS::UInteger instanceCountBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceCountBufferOffset_), instanceCountBufferOffset);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setInstanceDescriptorBuffer(const MTL::Buffer* instanceDescriptorBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceDescriptorBuffer_), instanceDescriptorBuffer);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setInstanceDescriptorBufferOffset(NS::UInteger instanceDescriptorBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceDescriptorBufferOffset_), instanceDescriptorBufferOffset);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setInstanceDescriptorStride(NS::UInteger instanceDescriptorStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceDescriptorStride_), instanceDescriptorStride);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorType instanceDescriptorType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceDescriptorType_), instanceDescriptorType);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setInstanceTransformationMatrixLayout(MTL::MatrixLayout instanceTransformationMatrixLayout)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setInstanceTransformationMatrixLayout_), instanceTransformationMatrixLayout);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setMaxInstanceCount(NS::UInteger maxInstanceCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxInstanceCount_), maxInstanceCount);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setMaxMotionTransformCount(NS::UInteger maxMotionTransformCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMaxMotionTransformCount_), maxMotionTransformCount);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setMotionTransformBuffer(const MTL::Buffer* motionTransformBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformBuffer_), motionTransformBuffer);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setMotionTransformBufferOffset(NS::UInteger motionTransformBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformBufferOffset_), motionTransformBufferOffset);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setMotionTransformCountBuffer(const MTL::Buffer* motionTransformCountBuffer)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformCountBuffer_), motionTransformCountBuffer);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setMotionTransformCountBufferOffset(NS::UInteger motionTransformCountBufferOffset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformCountBufferOffset_), motionTransformCountBufferOffset);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setMotionTransformStride(NS::UInteger motionTransformStride)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformStride_), motionTransformStride);
}

_MTL_INLINE void MTL::IndirectInstanceAccelerationStructureDescriptor::setMotionTransformType(MTL::TransformType motionTransformType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setMotionTransformType_), motionTransformType);
}

_MTL_INLINE MTL::ResourceID MTL::AccelerationStructure::gpuResourceID() const
{
    return Object::sendMessage<MTL::ResourceID>(this, _MTL_PRIVATE_SEL(gpuResourceID));
}

_MTL_INLINE NS::UInteger MTL::AccelerationStructure::size() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(size));
}
