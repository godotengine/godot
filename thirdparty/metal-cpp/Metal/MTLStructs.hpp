#pragma once

#include "MTLDefines.hpp"
#include "../Foundation/NSTypes.hpp"

namespace MTL {

using AccelerationStructureInstanceOptions = uint32_t;
enum MotionBorderMode : uint32_t;
enum TextureSwizzle : uint8_t;

struct Origin {
Origin() = default;
Origin(NS::UInteger x_, NS::UInteger y_, NS::UInteger z_) : x(x_), y(y_), z(z_) {}
static Origin Make(NS::UInteger x_, NS::UInteger y_, NS::UInteger z_) { return Origin(x_, y_, z_); }
    NS::UInteger x;
    NS::UInteger y;
    NS::UInteger z;
} _MTL_PACKED;

struct Size {
Size() = default;
Size(NS::UInteger w, NS::UInteger h, NS::UInteger d) : width(w), height(h), depth(d) {}
static Size Make(NS::UInteger w, NS::UInteger h, NS::UInteger d) { return Size(w, h, d); }
    NS::UInteger width;
    NS::UInteger height;
    NS::UInteger depth;
} _MTL_PACKED;

struct Region {
Region() = default;
Region(NS::UInteger x, NS::UInteger width) : origin(x, 0, 0), size(width, 1, 1) {}
Region(NS::UInteger x, NS::UInteger y, NS::UInteger width, NS::UInteger height) : origin(x, y, 0), size(width, height, 1) {}
Region(NS::UInteger x, NS::UInteger y, NS::UInteger z, NS::UInteger width, NS::UInteger height, NS::UInteger depth)
    : origin(x, y, z), size(width, height, depth) {}
static Region Make1D(NS::UInteger x, NS::UInteger width) { return Region(x, width); }
static Region Make2D(NS::UInteger x, NS::UInteger y, NS::UInteger width, NS::UInteger height) { return Region(x, y, width, height); }
static Region Make3D(NS::UInteger x, NS::UInteger y, NS::UInteger z, NS::UInteger width, NS::UInteger height, NS::UInteger depth) {
    return Region(x, y, z, width, height, depth);
}
    MTL::Origin origin;
    MTL::Size size;
} _MTL_PACKED;

struct SamplePosition {
SamplePosition() = default;
SamplePosition(float x_, float y_) : x(x_), y(y_) {}
static SamplePosition Make(float x_, float y_) { return SamplePosition(x_, y_); }
    float x;
    float y;
} _MTL_PACKED;

struct ResourceID {
    uint64_t _impl;
} _MTL_PACKED;

struct ClearColor {
ClearColor() = default;
ClearColor(double r, double g, double b, double a) : red(r), green(g), blue(b), alpha(a) {}
static ClearColor Make(double r, double g, double b, double a) { return ClearColor(r, g, b, a); }
    double red;
    double green;
    double blue;
    double alpha;
} _MTL_PACKED;

struct AccelerationStructureMotionInstanceDescriptor {
    MTL::AccelerationStructureInstanceOptions options;
    uint32_t mask;
    uint32_t intersectionFunctionTableOffset;
    uint32_t accelerationStructureIndex;
    uint32_t userID;
    uint32_t motionTransformsStartIndex;
    uint32_t motionTransformsCount;
    MTL::MotionBorderMode motionStartBorderMode;
    MTL::MotionBorderMode motionEndBorderMode;
    float motionStartTime;
    float motionEndTime;
} _MTL_PACKED;

struct IndirectAccelerationStructureMotionInstanceDescriptor {
    MTL::AccelerationStructureInstanceOptions options;
    uint32_t mask;
    uint32_t intersectionFunctionTableOffset;
    uint32_t userID;
    MTL::ResourceID accelerationStructureID;
    uint32_t motionTransformsStartIndex;
    uint32_t motionTransformsCount;
    MTL::MotionBorderMode motionStartBorderMode;
    MTL::MotionBorderMode motionEndBorderMode;
    float motionStartTime;
    float motionEndTime;
} _MTL_PACKED;

struct DispatchThreadgroupsIndirectArguments {
    uint32_t threadgroupsPerGrid[3];
} _MTL_PACKED;

struct DispatchThreadsIndirectArguments {
    uint32_t threadsPerGrid[3];
    uint32_t threadsPerThreadgroup[3];
} _MTL_PACKED;

struct StageInRegionIndirectArguments {
    uint32_t stageInOrigin[3];
    uint32_t stageInSize[3];
} _MTL_PACKED;

struct CounterResultTimestamp {
    uint64_t timestamp;
} _MTL_PACKED;

struct CounterResultStageUtilization {
    uint64_t totalCycles;
    uint64_t vertexCycles;
    uint64_t tessellationCycles;
    uint64_t postTessellationVertexCycles;
    uint64_t fragmentCycles;
    uint64_t renderTargetCycles;
} _MTL_PACKED;

struct CounterResultStatistic {
    uint64_t tessellationInputPatches;
    uint64_t vertexInvocations;
    uint64_t postTessellationVertexInvocations;
    uint64_t clipperInvocations;
    uint64_t clipperPrimitivesOut;
    uint64_t fragmentInvocations;
    uint64_t fragmentsPassed;
    uint64_t computeKernelInvocations;
} _MTL_PACKED;

struct AccelerationStructureSizes {
    NS::UInteger accelerationStructureSize;
    NS::UInteger buildScratchBufferSize;
    NS::UInteger refitScratchBufferSize;
} _MTL_PACKED;

struct SizeAndAlign {
    NS::UInteger size;
    NS::UInteger align;
} _MTL_PACKED;

struct IndirectCommandBufferExecutionRange {
    uint32_t location;
    uint32_t length;
} _MTL_PACKED;

struct IntersectionFunctionBufferArguments {
    uint64_t intersectionFunctionBuffer;
    uint64_t intersectionFunctionBufferSize;
    uint64_t intersectionFunctionStride;
} _MTL_PACKED;

struct ScissorRect {
    NS::UInteger x;
    NS::UInteger y;
    NS::UInteger width;
    NS::UInteger height;
} _MTL_PACKED;

struct Viewport {
    double originX;
    double originY;
    double width;
    double height;
    double znear;
    double zfar;
} _MTL_PACKED;

struct DrawPrimitivesIndirectArguments {
    uint32_t vertexCount;
    uint32_t instanceCount;
    uint32_t vertexStart;
    uint32_t baseInstance;
} _MTL_PACKED;

struct DrawIndexedPrimitivesIndirectArguments {
    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t indexStart;
    int32_t baseVertex;
    uint32_t baseInstance;
} _MTL_PACKED;

struct VertexAmplificationViewMapping {
    uint32_t viewportArrayIndexOffset;
    uint32_t renderTargetArrayIndexOffset;
} _MTL_PACKED;

struct DrawPatchIndirectArguments {
    uint32_t patchCount;
    uint32_t instanceCount;
    uint32_t patchStart;
    uint32_t baseInstance;
} _MTL_PACKED;

struct QuadTessellationFactorsHalf {
    uint16_t edgeTessellationFactor[4];
    uint16_t insideTessellationFactor[2];
} _MTL_PACKED;

struct TriangleTessellationFactorsHalf {
    uint16_t edgeTessellationFactor[3];
    uint16_t insideTessellationFactor;
} _MTL_PACKED;

struct MapIndirectArguments {
    uint32_t regionOriginX;
    uint32_t regionOriginY;
    uint32_t regionOriginZ;
    uint32_t regionSizeWidth;
    uint32_t regionSizeHeight;
    uint32_t regionSizeDepth;
    uint32_t mipMapLevel;
    uint32_t sliceId;
} _MTL_PACKED;

struct TextureSwizzleChannels {
TextureSwizzleChannels() = default;
TextureSwizzleChannels(MTL::TextureSwizzle r, MTL::TextureSwizzle g, MTL::TextureSwizzle b, MTL::TextureSwizzle a)
    : red(r), green(g), blue(b), alpha(a) {}
static TextureSwizzleChannels Default() { return TextureSwizzleChannels(); }
static TextureSwizzleChannels Make(MTL::TextureSwizzle r, MTL::TextureSwizzle g, MTL::TextureSwizzle b, MTL::TextureSwizzle a) {
    return TextureSwizzleChannels(r, g, b, a);
}
    MTL::TextureSwizzle red;
    MTL::TextureSwizzle green;
    MTL::TextureSwizzle blue;
    MTL::TextureSwizzle alpha;
} _MTL_PACKED;

using Coordinate2D = SamplePosition;

} // MTL

#include "MTLDefines.hpp"
#include "MTLGPUAddress.hpp"
#include "MTLAccelerationStructureTypes.hpp"
#include "MTLPrivate.hpp"

namespace MTL {

struct AccelerationStructureInstanceDescriptor {
    MTL::PackedFloat4x3 transformationMatrix;
    MTL::AccelerationStructureInstanceOptions options;
    uint32_t mask;
    uint32_t intersectionFunctionTableOffset;
    uint32_t accelerationStructureIndex;
} _MTL_PACKED;

struct AccelerationStructureUserIDInstanceDescriptor {
    MTL::PackedFloat4x3 transformationMatrix;
    MTL::AccelerationStructureInstanceOptions options;
    uint32_t mask;
    uint32_t intersectionFunctionTableOffset;
    uint32_t accelerationStructureIndex;
    uint32_t userID;
} _MTL_PACKED;

struct IndirectAccelerationStructureInstanceDescriptor {
    MTL::PackedFloat4x3 transformationMatrix;
    MTL::AccelerationStructureInstanceOptions options;
    uint32_t mask;
    uint32_t intersectionFunctionTableOffset;
    uint32_t userID;
    MTL::ResourceID accelerationStructureID;
} _MTL_PACKED;

using PackedFloat4x3 = PackedFloat4x3;
using AxisAlignedBoundingBox = AxisAlignedBoundingBox;
using CommonCounter = NS::String*;
using CommonCounterSet = NS::String*;
using DeviceNotificationName = NS::String*;
using Timestamp = uint64_t;
using NSDeviceCertification = NS::Integer;
using NSProcessPerformanceProfile = NS::Integer;
using AutoreleasedRenderPipelineReflection = RenderPipelineReflection*;
using AutoreleasedComputePipelineReflection = ComputePipelineReflection*;
using AutoreleasedArgument = Argument*;

} // MTL
