//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2012-2013 LunarG, Inc.
// Copyright (C) 2017 ARM Limited.
// Modifications Copyright (C) 2020 Advanced Micro Devices, Inc. All rights reserved.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#ifndef _BASICTYPES_INCLUDED_
#define _BASICTYPES_INCLUDED_

namespace glslang {

//
// Basic type.  Arrays, vectors, sampler details, etc., are orthogonal to this.
//
enum TBasicType {
    EbtVoid,
    EbtFloat,
    EbtDouble,
    EbtFloat16,
    EbtInt8,
    EbtUint8,
    EbtInt16,
    EbtUint16,
    EbtInt,
    EbtUint,
    EbtInt64,
    EbtUint64,
    EbtBool,
    EbtAtomicUint,
    EbtSampler,
    EbtStruct,
    EbtBlock,
    EbtAccStruct,
    EbtReference,
    EbtRayQuery,
#ifndef GLSLANG_WEB
    // SPIR-V type defined by spirv_type
    EbtSpirvType,
#endif

    // HLSL types that live only temporarily.
    EbtString,

    EbtNumTypes
};

//
// Storage qualifiers.  Should align with different kinds of storage or
// resource or GLSL storage qualifier.  Expansion is deprecated.
//
// N.B.: You probably DON'T want to add anything here, but rather just add it
// to the built-in variables.  See the comment above TBuiltInVariable.
//
// A new built-in variable will normally be an existing qualifier, like 'in', 'out', etc.
// DO NOT follow the design pattern of, say EvqInstanceId, etc.
//
enum TStorageQualifier {
    EvqTemporary,     // For temporaries (within a function), read/write
    EvqGlobal,        // For globals read/write
    EvqConst,         // User-defined constant values, will be semantically constant and constant folded
    EvqVaryingIn,     // pipeline input, read only, also supercategory for all built-ins not included in this enum (see TBuiltInVariable)
    EvqVaryingOut,    // pipeline output, read/write, also supercategory for all built-ins not included in this enum (see TBuiltInVariable)
    EvqUniform,       // read only, shared with app
    EvqBuffer,        // read/write, shared with app
    EvqShared,        // compute shader's read/write 'shared' qualifier
#ifndef GLSLANG_WEB
    EvqSpirvStorageClass, // spirv_storage_class
#endif

    EvqPayload,
    EvqPayloadIn,
    EvqHitAttr,
    EvqCallableData,
    EvqCallableDataIn,

    // parameters
    EvqIn,            // also, for 'in' in the grammar before we know if it's a pipeline input or an 'in' parameter
    EvqOut,           // also, for 'out' in the grammar before we know if it's a pipeline output or an 'out' parameter
    EvqInOut,
    EvqConstReadOnly, // input; also other read-only types having neither a constant value nor constant-value semantics

    // built-ins read by vertex shader
    EvqVertexId,
    EvqInstanceId,

    // built-ins written by vertex shader
    EvqPosition,
    EvqPointSize,
    EvqClipVertex,

    // built-ins read by fragment shader
    EvqFace,
    EvqFragCoord,
    EvqPointCoord,

    // built-ins written by fragment shader
    EvqFragColor,
    EvqFragDepth,
    EvqFragStencil,

    // end of list
    EvqLast
};

//
// Subcategories of the TStorageQualifier, simply to give a direct mapping
// between built-in variable names and an numerical value (the enum).
//
// For backward compatibility, there is some redundancy between the
// TStorageQualifier and these.  Existing members should both be maintained accurately.
// However, any new built-in variable (and any existing non-redundant one)
// must follow the pattern that the specific built-in is here, and only its
// general qualifier is in TStorageQualifier.
//
// Something like gl_Position, which is sometimes 'in' and sometimes 'out'
// shows up as two different built-in variables in a single stage, but
// only has a single enum in TBuiltInVariable, so both the
// TStorageQualifier and the TBuitinVariable are needed to distinguish
// between them.
//
enum TBuiltInVariable {
    EbvNone,
    EbvNumWorkGroups,
    EbvWorkGroupSize,
    EbvWorkGroupId,
    EbvLocalInvocationId,
    EbvGlobalInvocationId,
    EbvLocalInvocationIndex,
    EbvNumSubgroups,
    EbvSubgroupID,
    EbvSubGroupSize,
    EbvSubGroupInvocation,
    EbvSubGroupEqMask,
    EbvSubGroupGeMask,
    EbvSubGroupGtMask,
    EbvSubGroupLeMask,
    EbvSubGroupLtMask,
    EbvSubgroupSize2,
    EbvSubgroupInvocation2,
    EbvSubgroupEqMask2,
    EbvSubgroupGeMask2,
    EbvSubgroupGtMask2,
    EbvSubgroupLeMask2,
    EbvSubgroupLtMask2,
    EbvVertexId,
    EbvInstanceId,
    EbvVertexIndex,
    EbvInstanceIndex,
    EbvBaseVertex,
    EbvBaseInstance,
    EbvDrawId,
    EbvPosition,
    EbvPointSize,
    EbvClipVertex,
    EbvClipDistance,
    EbvCullDistance,
    EbvNormal,
    EbvVertex,
    EbvMultiTexCoord0,
    EbvMultiTexCoord1,
    EbvMultiTexCoord2,
    EbvMultiTexCoord3,
    EbvMultiTexCoord4,
    EbvMultiTexCoord5,
    EbvMultiTexCoord6,
    EbvMultiTexCoord7,
    EbvFrontColor,
    EbvBackColor,
    EbvFrontSecondaryColor,
    EbvBackSecondaryColor,
    EbvTexCoord,
    EbvFogFragCoord,
    EbvInvocationId,
    EbvPrimitiveId,
    EbvLayer,
    EbvViewportIndex,
    EbvPatchVertices,
    EbvTessLevelOuter,
    EbvTessLevelInner,
    EbvBoundingBox,
    EbvTessCoord,
    EbvColor,
    EbvSecondaryColor,
    EbvFace,
    EbvFragCoord,
    EbvPointCoord,
    EbvFragColor,
    EbvFragData,
    EbvFragDepth,
    EbvFragStencilRef,
    EbvSampleId,
    EbvSamplePosition,
    EbvSampleMask,
    EbvHelperInvocation,

    EbvBaryCoordNoPersp,
    EbvBaryCoordNoPerspCentroid,
    EbvBaryCoordNoPerspSample,
    EbvBaryCoordSmooth,
    EbvBaryCoordSmoothCentroid,
    EbvBaryCoordSmoothSample,
    EbvBaryCoordPullModel,

    EbvViewIndex,
    EbvDeviceIndex,

    EbvShadingRateKHR,
    EbvPrimitiveShadingRateKHR,

    EbvFragSizeEXT,
    EbvFragInvocationCountEXT,

    EbvSecondaryFragDataEXT,
    EbvSecondaryFragColorEXT,

    EbvViewportMaskNV,
    EbvSecondaryPositionNV,
    EbvSecondaryViewportMaskNV,
    EbvPositionPerViewNV,
    EbvViewportMaskPerViewNV,
    EbvFragFullyCoveredNV,
    EbvFragmentSizeNV,
    EbvInvocationsPerPixelNV,
    // ray tracing
    EbvLaunchId,
    EbvLaunchSize,
    EbvInstanceCustomIndex,
    EbvGeometryIndex,
    EbvWorldRayOrigin,
    EbvWorldRayDirection,
    EbvObjectRayOrigin,
    EbvObjectRayDirection,
    EbvRayTmin,
    EbvRayTmax,
    EbvCullMask,
    EbvHitT,
    EbvHitKind,
    EbvObjectToWorld,
    EbvObjectToWorld3x4,
    EbvWorldToObject,
    EbvWorldToObject3x4,
    EbvIncomingRayFlags,
    EbvCurrentRayTimeNV,
    // barycentrics
    EbvBaryCoordNV,
    EbvBaryCoordNoPerspNV,
    EbvBaryCoordEXT,
    EbvBaryCoordNoPerspEXT,
    // mesh shaders
    EbvTaskCountNV,
    EbvPrimitiveCountNV,
    EbvPrimitiveIndicesNV,
    EbvClipDistancePerViewNV,
    EbvCullDistancePerViewNV,
    EbvLayerPerViewNV,
    EbvMeshViewCountNV,
    EbvMeshViewIndicesNV,

    // sm builtins
    EbvWarpsPerSM,
    EbvSMCount,
    EbvWarpID,
    EbvSMID,

    // HLSL built-ins that live only temporarily, until they get remapped
    // to one of the above.
    EbvFragDepthGreater,
    EbvFragDepthLesser,
    EbvGsOutputStream,
    EbvOutputPatch,
    EbvInputPatch,

    // structbuffer types
    EbvAppendConsume, // no need to differentiate append and consume
    EbvRWStructuredBuffer,
    EbvStructuredBuffer,
    EbvByteAddressBuffer,
    EbvRWByteAddressBuffer,

    EbvLast
};

// In this enum, order matters; users can assume higher precision is a bigger value
// and EpqNone is 0.
enum TPrecisionQualifier {
    EpqNone = 0,
    EpqLow,
    EpqMedium,
    EpqHigh
};

#ifdef GLSLANG_WEB
__inline const char* GetStorageQualifierString(TStorageQualifier q) { return ""; }
__inline const char* GetPrecisionQualifierString(TPrecisionQualifier p) { return ""; }
#else
// These will show up in error messages
__inline const char* GetStorageQualifierString(TStorageQualifier q)
{
    switch (q) {
    case EvqTemporary:      return "temp";           break;
    case EvqGlobal:         return "global";         break;
    case EvqConst:          return "const";          break;
    case EvqConstReadOnly:  return "const (read only)"; break;
#ifndef GLSLANG_WEB
    case EvqSpirvStorageClass: return "spirv_storage_class"; break;
#endif
    case EvqVaryingIn:      return "in";             break;
    case EvqVaryingOut:     return "out";            break;
    case EvqUniform:        return "uniform";        break;
    case EvqBuffer:         return "buffer";         break;
    case EvqShared:         return "shared";         break;
    case EvqIn:             return "in";             break;
    case EvqOut:            return "out";            break;
    case EvqInOut:          return "inout";          break;
    case EvqVertexId:       return "gl_VertexId";    break;
    case EvqInstanceId:     return "gl_InstanceId";  break;
    case EvqPosition:       return "gl_Position";    break;
    case EvqPointSize:      return "gl_PointSize";   break;
    case EvqClipVertex:     return "gl_ClipVertex";  break;
    case EvqFace:           return "gl_FrontFacing"; break;
    case EvqFragCoord:      return "gl_FragCoord";   break;
    case EvqPointCoord:     return "gl_PointCoord";  break;
    case EvqFragColor:      return "fragColor";      break;
    case EvqFragDepth:      return "gl_FragDepth";   break;
    case EvqFragStencil:    return "gl_FragStencilRefARB"; break;
    case EvqPayload:        return "rayPayloadNV";     break;
    case EvqPayloadIn:      return "rayPayloadInNV";   break;
    case EvqHitAttr:        return "hitAttributeNV";   break;
    case EvqCallableData:   return "callableDataNV";   break;
    case EvqCallableDataIn: return "callableDataInNV"; break;
    default:                return "unknown qualifier";
    }
}

__inline const char* GetBuiltInVariableString(TBuiltInVariable v)
{
    switch (v) {
    case EbvNone:                 return "";
    case EbvNumWorkGroups:        return "NumWorkGroups";
    case EbvWorkGroupSize:        return "WorkGroupSize";
    case EbvWorkGroupId:          return "WorkGroupID";
    case EbvLocalInvocationId:    return "LocalInvocationID";
    case EbvGlobalInvocationId:   return "GlobalInvocationID";
    case EbvLocalInvocationIndex: return "LocalInvocationIndex";
    case EbvNumSubgroups:         return "NumSubgroups";
    case EbvSubgroupID:           return "SubgroupID";
    case EbvSubGroupSize:         return "SubGroupSize";
    case EbvSubGroupInvocation:   return "SubGroupInvocation";
    case EbvSubGroupEqMask:       return "SubGroupEqMask";
    case EbvSubGroupGeMask:       return "SubGroupGeMask";
    case EbvSubGroupGtMask:       return "SubGroupGtMask";
    case EbvSubGroupLeMask:       return "SubGroupLeMask";
    case EbvSubGroupLtMask:       return "SubGroupLtMask";
    case EbvSubgroupSize2:        return "SubgroupSize";
    case EbvSubgroupInvocation2:  return "SubgroupInvocationID";
    case EbvSubgroupEqMask2:      return "SubgroupEqMask";
    case EbvSubgroupGeMask2:      return "SubgroupGeMask";
    case EbvSubgroupGtMask2:      return "SubgroupGtMask";
    case EbvSubgroupLeMask2:      return "SubgroupLeMask";
    case EbvSubgroupLtMask2:      return "SubgroupLtMask";
    case EbvVertexId:             return "VertexId";
    case EbvInstanceId:           return "InstanceId";
    case EbvVertexIndex:          return "VertexIndex";
    case EbvInstanceIndex:        return "InstanceIndex";
    case EbvBaseVertex:           return "BaseVertex";
    case EbvBaseInstance:         return "BaseInstance";
    case EbvDrawId:               return "DrawId";
    case EbvPosition:             return "Position";
    case EbvPointSize:            return "PointSize";
    case EbvClipVertex:           return "ClipVertex";
    case EbvClipDistance:         return "ClipDistance";
    case EbvCullDistance:         return "CullDistance";
    case EbvNormal:               return "Normal";
    case EbvVertex:               return "Vertex";
    case EbvMultiTexCoord0:       return "MultiTexCoord0";
    case EbvMultiTexCoord1:       return "MultiTexCoord1";
    case EbvMultiTexCoord2:       return "MultiTexCoord2";
    case EbvMultiTexCoord3:       return "MultiTexCoord3";
    case EbvMultiTexCoord4:       return "MultiTexCoord4";
    case EbvMultiTexCoord5:       return "MultiTexCoord5";
    case EbvMultiTexCoord6:       return "MultiTexCoord6";
    case EbvMultiTexCoord7:       return "MultiTexCoord7";
    case EbvFrontColor:           return "FrontColor";
    case EbvBackColor:            return "BackColor";
    case EbvFrontSecondaryColor:  return "FrontSecondaryColor";
    case EbvBackSecondaryColor:   return "BackSecondaryColor";
    case EbvTexCoord:             return "TexCoord";
    case EbvFogFragCoord:         return "FogFragCoord";
    case EbvInvocationId:         return "InvocationID";
    case EbvPrimitiveId:          return "PrimitiveID";
    case EbvLayer:                return "Layer";
    case EbvViewportIndex:        return "ViewportIndex";
    case EbvPatchVertices:        return "PatchVertices";
    case EbvTessLevelOuter:       return "TessLevelOuter";
    case EbvTessLevelInner:       return "TessLevelInner";
    case EbvBoundingBox:          return "BoundingBox";
    case EbvTessCoord:            return "TessCoord";
    case EbvColor:                return "Color";
    case EbvSecondaryColor:       return "SecondaryColor";
    case EbvFace:                 return "Face";
    case EbvFragCoord:            return "FragCoord";
    case EbvPointCoord:           return "PointCoord";
    case EbvFragColor:            return "FragColor";
    case EbvFragData:             return "FragData";
    case EbvFragDepth:            return "FragDepth";
    case EbvFragStencilRef:       return "FragStencilRef";
    case EbvSampleId:             return "SampleId";
    case EbvSamplePosition:       return "SamplePosition";
    case EbvSampleMask:           return "SampleMaskIn";
    case EbvHelperInvocation:     return "HelperInvocation";

    case EbvBaryCoordNoPersp:           return "BaryCoordNoPersp";
    case EbvBaryCoordNoPerspCentroid:   return "BaryCoordNoPerspCentroid";
    case EbvBaryCoordNoPerspSample:     return "BaryCoordNoPerspSample";
    case EbvBaryCoordSmooth:            return "BaryCoordSmooth";
    case EbvBaryCoordSmoothCentroid:    return "BaryCoordSmoothCentroid";
    case EbvBaryCoordSmoothSample:      return "BaryCoordSmoothSample";
    case EbvBaryCoordPullModel:         return "BaryCoordPullModel";

    case EbvViewIndex:                  return "ViewIndex";
    case EbvDeviceIndex:                return "DeviceIndex";

    case EbvFragSizeEXT:                return "FragSizeEXT";
    case EbvFragInvocationCountEXT:     return "FragInvocationCountEXT";

    case EbvSecondaryFragDataEXT:       return "SecondaryFragDataEXT";
    case EbvSecondaryFragColorEXT:      return "SecondaryFragColorEXT";

    case EbvViewportMaskNV:             return "ViewportMaskNV";
    case EbvSecondaryPositionNV:        return "SecondaryPositionNV";
    case EbvSecondaryViewportMaskNV:    return "SecondaryViewportMaskNV";
    case EbvPositionPerViewNV:          return "PositionPerViewNV";
    case EbvViewportMaskPerViewNV:      return "ViewportMaskPerViewNV";
    case EbvFragFullyCoveredNV:         return "FragFullyCoveredNV";
    case EbvFragmentSizeNV:             return "FragmentSizeNV";
    case EbvInvocationsPerPixelNV:      return "InvocationsPerPixelNV";
    case EbvLaunchId:                   return "LaunchIdNV";
    case EbvLaunchSize:                 return "LaunchSizeNV";
    case EbvInstanceCustomIndex:        return "InstanceCustomIndexNV";
    case EbvGeometryIndex:              return "GeometryIndexEXT";
    case EbvWorldRayOrigin:             return "WorldRayOriginNV";
    case EbvWorldRayDirection:          return "WorldRayDirectionNV";
    case EbvObjectRayOrigin:            return "ObjectRayOriginNV";
    case EbvObjectRayDirection:         return "ObjectRayDirectionNV";
    case EbvRayTmin:                    return "ObjectRayTminNV";
    case EbvRayTmax:                    return "ObjectRayTmaxNV";
    case EbvHitT:                       return "HitTNV";
    case EbvHitKind:                    return "HitKindNV";
    case EbvIncomingRayFlags:           return "IncomingRayFlagsNV";
    case EbvObjectToWorld:              return "ObjectToWorldNV";
    case EbvWorldToObject:              return "WorldToObjectNV";
    case EbvCurrentRayTimeNV:           return "CurrentRayTimeNV";

    case EbvBaryCoordEXT:
    case EbvBaryCoordNV:                return "BaryCoordKHR";
    case EbvBaryCoordNoPerspEXT:
    case EbvBaryCoordNoPerspNV:         return "BaryCoordNoPerspKHR";

    case EbvTaskCountNV:                return "TaskCountNV";
    case EbvPrimitiveCountNV:           return "PrimitiveCountNV";
    case EbvPrimitiveIndicesNV:         return "PrimitiveIndicesNV";
    case EbvClipDistancePerViewNV:      return "ClipDistancePerViewNV";
    case EbvCullDistancePerViewNV:      return "CullDistancePerViewNV";
    case EbvLayerPerViewNV:             return "LayerPerViewNV";
    case EbvMeshViewCountNV:            return "MeshViewCountNV";
    case EbvMeshViewIndicesNV:          return "MeshViewIndicesNV";

    case EbvWarpsPerSM:                 return "WarpsPerSMNV";
    case EbvSMCount:                    return "SMCountNV";
    case EbvWarpID:                     return "WarpIDNV";
    case EbvSMID:                       return "SMIDNV";

    case EbvShadingRateKHR:             return "ShadingRateKHR";
    case EbvPrimitiveShadingRateKHR:    return "PrimitiveShadingRateKHR";

    default:                      return "unknown built-in variable";
    }
}

__inline const char* GetPrecisionQualifierString(TPrecisionQualifier p)
{
    switch (p) {
    case EpqNone:   return "";        break;
    case EpqLow:    return "lowp";    break;
    case EpqMedium: return "mediump"; break;
    case EpqHigh:   return "highp";   break;
    default:        return "unknown precision qualifier";
    }
}
#endif

__inline bool isTypeSignedInt(TBasicType type)
{
    switch (type) {
    case EbtInt8:
    case EbtInt16:
    case EbtInt:
    case EbtInt64:
        return true;
    default:
        return false;
    }
}

__inline bool isTypeUnsignedInt(TBasicType type)
{
    switch (type) {
    case EbtUint8:
    case EbtUint16:
    case EbtUint:
    case EbtUint64:
        return true;
    default:
        return false;
    }
}

__inline bool isTypeInt(TBasicType type)
{
    return isTypeSignedInt(type) || isTypeUnsignedInt(type);
}

__inline bool isTypeFloat(TBasicType type)
{
    switch (type) {
    case EbtFloat:
    case EbtDouble:
    case EbtFloat16:
        return true;
    default:
        return false;
    }
}

__inline int getTypeRank(TBasicType type)
{
    int res = -1;
    switch(type) {
    case EbtInt8:
    case EbtUint8:
        res = 0;
        break;
    case EbtInt16:
    case EbtUint16:
        res = 1;
        break;
    case EbtInt:
    case EbtUint:
        res = 2;
        break;
    case EbtInt64:
    case EbtUint64:
        res = 3;
        break;
    default:
        assert(false);
        break;
    }
    return res;
}

} // end namespace glslang

#endif // _BASICTYPES_INCLUDED_
