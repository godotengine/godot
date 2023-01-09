/* DO NOT EDIT - This file is generated automatically by spirv_info_c.py script */

/*
 * Copyright (C) 2017 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include "spirv_info.h"

const char *
spirv_addressingmodel_to_string(SpvAddressingModel v)
{
   switch (v) {
   case SpvAddressingModelLogical: return "SpvAddressingModelLogical";
   case SpvAddressingModelPhysical32: return "SpvAddressingModelPhysical32";
   case SpvAddressingModelPhysical64: return "SpvAddressingModelPhysical64";
   case SpvAddressingModelPhysicalStorageBuffer64: return "SpvAddressingModelPhysicalStorageBuffer64";
   case SpvAddressingModelMax: break; /* silence warnings about unhandled enums. */
   }

   return "unknown";
}

const char *
spirv_builtin_to_string(SpvBuiltIn v)
{
   switch (v) {
   case SpvBuiltInPosition: return "SpvBuiltInPosition";
   case SpvBuiltInPointSize: return "SpvBuiltInPointSize";
   case SpvBuiltInClipDistance: return "SpvBuiltInClipDistance";
   case SpvBuiltInCullDistance: return "SpvBuiltInCullDistance";
   case SpvBuiltInVertexId: return "SpvBuiltInVertexId";
   case SpvBuiltInInstanceId: return "SpvBuiltInInstanceId";
   case SpvBuiltInPrimitiveId: return "SpvBuiltInPrimitiveId";
   case SpvBuiltInInvocationId: return "SpvBuiltInInvocationId";
   case SpvBuiltInLayer: return "SpvBuiltInLayer";
   case SpvBuiltInViewportIndex: return "SpvBuiltInViewportIndex";
   case SpvBuiltInTessLevelOuter: return "SpvBuiltInTessLevelOuter";
   case SpvBuiltInTessLevelInner: return "SpvBuiltInTessLevelInner";
   case SpvBuiltInTessCoord: return "SpvBuiltInTessCoord";
   case SpvBuiltInPatchVertices: return "SpvBuiltInPatchVertices";
   case SpvBuiltInFragCoord: return "SpvBuiltInFragCoord";
   case SpvBuiltInPointCoord: return "SpvBuiltInPointCoord";
   case SpvBuiltInFrontFacing: return "SpvBuiltInFrontFacing";
   case SpvBuiltInSampleId: return "SpvBuiltInSampleId";
   case SpvBuiltInSamplePosition: return "SpvBuiltInSamplePosition";
   case SpvBuiltInSampleMask: return "SpvBuiltInSampleMask";
   case SpvBuiltInFragDepth: return "SpvBuiltInFragDepth";
   case SpvBuiltInHelperInvocation: return "SpvBuiltInHelperInvocation";
   case SpvBuiltInNumWorkgroups: return "SpvBuiltInNumWorkgroups";
   case SpvBuiltInWorkgroupSize: return "SpvBuiltInWorkgroupSize";
   case SpvBuiltInWorkgroupId: return "SpvBuiltInWorkgroupId";
   case SpvBuiltInLocalInvocationId: return "SpvBuiltInLocalInvocationId";
   case SpvBuiltInGlobalInvocationId: return "SpvBuiltInGlobalInvocationId";
   case SpvBuiltInLocalInvocationIndex: return "SpvBuiltInLocalInvocationIndex";
   case SpvBuiltInWorkDim: return "SpvBuiltInWorkDim";
   case SpvBuiltInGlobalSize: return "SpvBuiltInGlobalSize";
   case SpvBuiltInEnqueuedWorkgroupSize: return "SpvBuiltInEnqueuedWorkgroupSize";
   case SpvBuiltInGlobalOffset: return "SpvBuiltInGlobalOffset";
   case SpvBuiltInGlobalLinearId: return "SpvBuiltInGlobalLinearId";
   case SpvBuiltInSubgroupSize: return "SpvBuiltInSubgroupSize";
   case SpvBuiltInSubgroupMaxSize: return "SpvBuiltInSubgroupMaxSize";
   case SpvBuiltInNumSubgroups: return "SpvBuiltInNumSubgroups";
   case SpvBuiltInNumEnqueuedSubgroups: return "SpvBuiltInNumEnqueuedSubgroups";
   case SpvBuiltInSubgroupId: return "SpvBuiltInSubgroupId";
   case SpvBuiltInSubgroupLocalInvocationId: return "SpvBuiltInSubgroupLocalInvocationId";
   case SpvBuiltInVertexIndex: return "SpvBuiltInVertexIndex";
   case SpvBuiltInInstanceIndex: return "SpvBuiltInInstanceIndex";
   case SpvBuiltInSubgroupEqMask: return "SpvBuiltInSubgroupEqMask";
   case SpvBuiltInSubgroupGeMask: return "SpvBuiltInSubgroupGeMask";
   case SpvBuiltInSubgroupGtMask: return "SpvBuiltInSubgroupGtMask";
   case SpvBuiltInSubgroupLeMask: return "SpvBuiltInSubgroupLeMask";
   case SpvBuiltInSubgroupLtMask: return "SpvBuiltInSubgroupLtMask";
   case SpvBuiltInBaseVertex: return "SpvBuiltInBaseVertex";
   case SpvBuiltInBaseInstance: return "SpvBuiltInBaseInstance";
   case SpvBuiltInDrawIndex: return "SpvBuiltInDrawIndex";
   case SpvBuiltInPrimitiveShadingRateKHR: return "SpvBuiltInPrimitiveShadingRateKHR";
   case SpvBuiltInDeviceIndex: return "SpvBuiltInDeviceIndex";
   case SpvBuiltInViewIndex: return "SpvBuiltInViewIndex";
   case SpvBuiltInShadingRateKHR: return "SpvBuiltInShadingRateKHR";
   case SpvBuiltInBaryCoordNoPerspAMD: return "SpvBuiltInBaryCoordNoPerspAMD";
   case SpvBuiltInBaryCoordNoPerspCentroidAMD: return "SpvBuiltInBaryCoordNoPerspCentroidAMD";
   case SpvBuiltInBaryCoordNoPerspSampleAMD: return "SpvBuiltInBaryCoordNoPerspSampleAMD";
   case SpvBuiltInBaryCoordSmoothAMD: return "SpvBuiltInBaryCoordSmoothAMD";
   case SpvBuiltInBaryCoordSmoothCentroidAMD: return "SpvBuiltInBaryCoordSmoothCentroidAMD";
   case SpvBuiltInBaryCoordSmoothSampleAMD: return "SpvBuiltInBaryCoordSmoothSampleAMD";
   case SpvBuiltInBaryCoordPullModelAMD: return "SpvBuiltInBaryCoordPullModelAMD";
   case SpvBuiltInFragStencilRefEXT: return "SpvBuiltInFragStencilRefEXT";
   case SpvBuiltInViewportMaskNV: return "SpvBuiltInViewportMaskNV";
   case SpvBuiltInSecondaryPositionNV: return "SpvBuiltInSecondaryPositionNV";
   case SpvBuiltInSecondaryViewportMaskNV: return "SpvBuiltInSecondaryViewportMaskNV";
   case SpvBuiltInPositionPerViewNV: return "SpvBuiltInPositionPerViewNV";
   case SpvBuiltInViewportMaskPerViewNV: return "SpvBuiltInViewportMaskPerViewNV";
   case SpvBuiltInFullyCoveredEXT: return "SpvBuiltInFullyCoveredEXT";
   case SpvBuiltInTaskCountNV: return "SpvBuiltInTaskCountNV";
   case SpvBuiltInPrimitiveCountNV: return "SpvBuiltInPrimitiveCountNV";
   case SpvBuiltInPrimitiveIndicesNV: return "SpvBuiltInPrimitiveIndicesNV";
   case SpvBuiltInClipDistancePerViewNV: return "SpvBuiltInClipDistancePerViewNV";
   case SpvBuiltInCullDistancePerViewNV: return "SpvBuiltInCullDistancePerViewNV";
   case SpvBuiltInLayerPerViewNV: return "SpvBuiltInLayerPerViewNV";
   case SpvBuiltInMeshViewCountNV: return "SpvBuiltInMeshViewCountNV";
   case SpvBuiltInMeshViewIndicesNV: return "SpvBuiltInMeshViewIndicesNV";
   case SpvBuiltInBaryCoordKHR: return "SpvBuiltInBaryCoordKHR";
   case SpvBuiltInBaryCoordNoPerspKHR: return "SpvBuiltInBaryCoordNoPerspKHR";
   case SpvBuiltInFragSizeEXT: return "SpvBuiltInFragSizeEXT";
   case SpvBuiltInFragInvocationCountEXT: return "SpvBuiltInFragInvocationCountEXT";
   case SpvBuiltInPrimitivePointIndicesEXT: return "SpvBuiltInPrimitivePointIndicesEXT";
   case SpvBuiltInPrimitiveLineIndicesEXT: return "SpvBuiltInPrimitiveLineIndicesEXT";
   case SpvBuiltInPrimitiveTriangleIndicesEXT: return "SpvBuiltInPrimitiveTriangleIndicesEXT";
   case SpvBuiltInCullPrimitiveEXT: return "SpvBuiltInCullPrimitiveEXT";
   case SpvBuiltInLaunchIdNV: return "SpvBuiltInLaunchIdNV";
   case SpvBuiltInLaunchSizeNV: return "SpvBuiltInLaunchSizeNV";
   case SpvBuiltInWorldRayOriginNV: return "SpvBuiltInWorldRayOriginNV";
   case SpvBuiltInWorldRayDirectionNV: return "SpvBuiltInWorldRayDirectionNV";
   case SpvBuiltInObjectRayOriginNV: return "SpvBuiltInObjectRayOriginNV";
   case SpvBuiltInObjectRayDirectionNV: return "SpvBuiltInObjectRayDirectionNV";
   case SpvBuiltInRayTminNV: return "SpvBuiltInRayTminNV";
   case SpvBuiltInRayTmaxNV: return "SpvBuiltInRayTmaxNV";
   case SpvBuiltInInstanceCustomIndexNV: return "SpvBuiltInInstanceCustomIndexNV";
   case SpvBuiltInObjectToWorldNV: return "SpvBuiltInObjectToWorldNV";
   case SpvBuiltInWorldToObjectNV: return "SpvBuiltInWorldToObjectNV";
   case SpvBuiltInHitTNV: return "SpvBuiltInHitTNV";
   case SpvBuiltInHitKindNV: return "SpvBuiltInHitKindNV";
   case SpvBuiltInCurrentRayTimeNV: return "SpvBuiltInCurrentRayTimeNV";
   case SpvBuiltInIncomingRayFlagsNV: return "SpvBuiltInIncomingRayFlagsNV";
   case SpvBuiltInRayGeometryIndexKHR: return "SpvBuiltInRayGeometryIndexKHR";
   case SpvBuiltInWarpsPerSMNV: return "SpvBuiltInWarpsPerSMNV";
   case SpvBuiltInSMCountNV: return "SpvBuiltInSMCountNV";
   case SpvBuiltInWarpIDNV: return "SpvBuiltInWarpIDNV";
   case SpvBuiltInSMIDNV: return "SpvBuiltInSMIDNV";
   case SpvBuiltInCullMaskKHR: return "SpvBuiltInCullMaskKHR";
   case SpvBuiltInMax: break; /* silence warnings about unhandled enums. */
   }

   return "unknown";
}

const char *
spirv_capability_to_string(SpvCapability v)
{
   switch (v) {
   case SpvCapabilityMatrix: return "SpvCapabilityMatrix";
   case SpvCapabilityShader: return "SpvCapabilityShader";
   case SpvCapabilityGeometry: return "SpvCapabilityGeometry";
   case SpvCapabilityTessellation: return "SpvCapabilityTessellation";
   case SpvCapabilityAddresses: return "SpvCapabilityAddresses";
   case SpvCapabilityLinkage: return "SpvCapabilityLinkage";
   case SpvCapabilityKernel: return "SpvCapabilityKernel";
   case SpvCapabilityVector16: return "SpvCapabilityVector16";
   case SpvCapabilityFloat16Buffer: return "SpvCapabilityFloat16Buffer";
   case SpvCapabilityFloat16: return "SpvCapabilityFloat16";
   case SpvCapabilityFloat64: return "SpvCapabilityFloat64";
   case SpvCapabilityInt64: return "SpvCapabilityInt64";
   case SpvCapabilityInt64Atomics: return "SpvCapabilityInt64Atomics";
   case SpvCapabilityImageBasic: return "SpvCapabilityImageBasic";
   case SpvCapabilityImageReadWrite: return "SpvCapabilityImageReadWrite";
   case SpvCapabilityImageMipmap: return "SpvCapabilityImageMipmap";
   case SpvCapabilityPipes: return "SpvCapabilityPipes";
   case SpvCapabilityGroups: return "SpvCapabilityGroups";
   case SpvCapabilityDeviceEnqueue: return "SpvCapabilityDeviceEnqueue";
   case SpvCapabilityLiteralSampler: return "SpvCapabilityLiteralSampler";
   case SpvCapabilityAtomicStorage: return "SpvCapabilityAtomicStorage";
   case SpvCapabilityInt16: return "SpvCapabilityInt16";
   case SpvCapabilityTessellationPointSize: return "SpvCapabilityTessellationPointSize";
   case SpvCapabilityGeometryPointSize: return "SpvCapabilityGeometryPointSize";
   case SpvCapabilityImageGatherExtended: return "SpvCapabilityImageGatherExtended";
   case SpvCapabilityStorageImageMultisample: return "SpvCapabilityStorageImageMultisample";
   case SpvCapabilityUniformBufferArrayDynamicIndexing: return "SpvCapabilityUniformBufferArrayDynamicIndexing";
   case SpvCapabilitySampledImageArrayDynamicIndexing: return "SpvCapabilitySampledImageArrayDynamicIndexing";
   case SpvCapabilityStorageBufferArrayDynamicIndexing: return "SpvCapabilityStorageBufferArrayDynamicIndexing";
   case SpvCapabilityStorageImageArrayDynamicIndexing: return "SpvCapabilityStorageImageArrayDynamicIndexing";
   case SpvCapabilityClipDistance: return "SpvCapabilityClipDistance";
   case SpvCapabilityCullDistance: return "SpvCapabilityCullDistance";
   case SpvCapabilityImageCubeArray: return "SpvCapabilityImageCubeArray";
   case SpvCapabilitySampleRateShading: return "SpvCapabilitySampleRateShading";
   case SpvCapabilityImageRect: return "SpvCapabilityImageRect";
   case SpvCapabilitySampledRect: return "SpvCapabilitySampledRect";
   case SpvCapabilityGenericPointer: return "SpvCapabilityGenericPointer";
   case SpvCapabilityInt8: return "SpvCapabilityInt8";
   case SpvCapabilityInputAttachment: return "SpvCapabilityInputAttachment";
   case SpvCapabilitySparseResidency: return "SpvCapabilitySparseResidency";
   case SpvCapabilityMinLod: return "SpvCapabilityMinLod";
   case SpvCapabilitySampled1D: return "SpvCapabilitySampled1D";
   case SpvCapabilityImage1D: return "SpvCapabilityImage1D";
   case SpvCapabilitySampledCubeArray: return "SpvCapabilitySampledCubeArray";
   case SpvCapabilitySampledBuffer: return "SpvCapabilitySampledBuffer";
   case SpvCapabilityImageBuffer: return "SpvCapabilityImageBuffer";
   case SpvCapabilityImageMSArray: return "SpvCapabilityImageMSArray";
   case SpvCapabilityStorageImageExtendedFormats: return "SpvCapabilityStorageImageExtendedFormats";
   case SpvCapabilityImageQuery: return "SpvCapabilityImageQuery";
   case SpvCapabilityDerivativeControl: return "SpvCapabilityDerivativeControl";
   case SpvCapabilityInterpolationFunction: return "SpvCapabilityInterpolationFunction";
   case SpvCapabilityTransformFeedback: return "SpvCapabilityTransformFeedback";
   case SpvCapabilityGeometryStreams: return "SpvCapabilityGeometryStreams";
   case SpvCapabilityStorageImageReadWithoutFormat: return "SpvCapabilityStorageImageReadWithoutFormat";
   case SpvCapabilityStorageImageWriteWithoutFormat: return "SpvCapabilityStorageImageWriteWithoutFormat";
   case SpvCapabilityMultiViewport: return "SpvCapabilityMultiViewport";
   case SpvCapabilitySubgroupDispatch: return "SpvCapabilitySubgroupDispatch";
   case SpvCapabilityNamedBarrier: return "SpvCapabilityNamedBarrier";
   case SpvCapabilityPipeStorage: return "SpvCapabilityPipeStorage";
   case SpvCapabilityGroupNonUniform: return "SpvCapabilityGroupNonUniform";
   case SpvCapabilityGroupNonUniformVote: return "SpvCapabilityGroupNonUniformVote";
   case SpvCapabilityGroupNonUniformArithmetic: return "SpvCapabilityGroupNonUniformArithmetic";
   case SpvCapabilityGroupNonUniformBallot: return "SpvCapabilityGroupNonUniformBallot";
   case SpvCapabilityGroupNonUniformShuffle: return "SpvCapabilityGroupNonUniformShuffle";
   case SpvCapabilityGroupNonUniformShuffleRelative: return "SpvCapabilityGroupNonUniformShuffleRelative";
   case SpvCapabilityGroupNonUniformClustered: return "SpvCapabilityGroupNonUniformClustered";
   case SpvCapabilityGroupNonUniformQuad: return "SpvCapabilityGroupNonUniformQuad";
   case SpvCapabilityShaderLayer: return "SpvCapabilityShaderLayer";
   case SpvCapabilityShaderViewportIndex: return "SpvCapabilityShaderViewportIndex";
   case SpvCapabilityUniformDecoration: return "SpvCapabilityUniformDecoration";
   case SpvCapabilityFragmentShadingRateKHR: return "SpvCapabilityFragmentShadingRateKHR";
   case SpvCapabilitySubgroupBallotKHR: return "SpvCapabilitySubgroupBallotKHR";
   case SpvCapabilityDrawParameters: return "SpvCapabilityDrawParameters";
   case SpvCapabilityWorkgroupMemoryExplicitLayoutKHR: return "SpvCapabilityWorkgroupMemoryExplicitLayoutKHR";
   case SpvCapabilityWorkgroupMemoryExplicitLayout8BitAccessKHR: return "SpvCapabilityWorkgroupMemoryExplicitLayout8BitAccessKHR";
   case SpvCapabilityWorkgroupMemoryExplicitLayout16BitAccessKHR: return "SpvCapabilityWorkgroupMemoryExplicitLayout16BitAccessKHR";
   case SpvCapabilitySubgroupVoteKHR: return "SpvCapabilitySubgroupVoteKHR";
   case SpvCapabilityStorageBuffer16BitAccess: return "SpvCapabilityStorageBuffer16BitAccess";
   case SpvCapabilityUniformAndStorageBuffer16BitAccess: return "SpvCapabilityUniformAndStorageBuffer16BitAccess";
   case SpvCapabilityStoragePushConstant16: return "SpvCapabilityStoragePushConstant16";
   case SpvCapabilityStorageInputOutput16: return "SpvCapabilityStorageInputOutput16";
   case SpvCapabilityDeviceGroup: return "SpvCapabilityDeviceGroup";
   case SpvCapabilityMultiView: return "SpvCapabilityMultiView";
   case SpvCapabilityVariablePointersStorageBuffer: return "SpvCapabilityVariablePointersStorageBuffer";
   case SpvCapabilityVariablePointers: return "SpvCapabilityVariablePointers";
   case SpvCapabilityAtomicStorageOps: return "SpvCapabilityAtomicStorageOps";
   case SpvCapabilitySampleMaskPostDepthCoverage: return "SpvCapabilitySampleMaskPostDepthCoverage";
   case SpvCapabilityStorageBuffer8BitAccess: return "SpvCapabilityStorageBuffer8BitAccess";
   case SpvCapabilityUniformAndStorageBuffer8BitAccess: return "SpvCapabilityUniformAndStorageBuffer8BitAccess";
   case SpvCapabilityStoragePushConstant8: return "SpvCapabilityStoragePushConstant8";
   case SpvCapabilityDenormPreserve: return "SpvCapabilityDenormPreserve";
   case SpvCapabilityDenormFlushToZero: return "SpvCapabilityDenormFlushToZero";
   case SpvCapabilitySignedZeroInfNanPreserve: return "SpvCapabilitySignedZeroInfNanPreserve";
   case SpvCapabilityRoundingModeRTE: return "SpvCapabilityRoundingModeRTE";
   case SpvCapabilityRoundingModeRTZ: return "SpvCapabilityRoundingModeRTZ";
   case SpvCapabilityRayQueryProvisionalKHR: return "SpvCapabilityRayQueryProvisionalKHR";
   case SpvCapabilityRayQueryKHR: return "SpvCapabilityRayQueryKHR";
   case SpvCapabilityRayTraversalPrimitiveCullingKHR: return "SpvCapabilityRayTraversalPrimitiveCullingKHR";
   case SpvCapabilityRayTracingKHR: return "SpvCapabilityRayTracingKHR";
   case SpvCapabilityFloat16ImageAMD: return "SpvCapabilityFloat16ImageAMD";
   case SpvCapabilityImageGatherBiasLodAMD: return "SpvCapabilityImageGatherBiasLodAMD";
   case SpvCapabilityFragmentMaskAMD: return "SpvCapabilityFragmentMaskAMD";
   case SpvCapabilityStencilExportEXT: return "SpvCapabilityStencilExportEXT";
   case SpvCapabilityImageReadWriteLodAMD: return "SpvCapabilityImageReadWriteLodAMD";
   case SpvCapabilityInt64ImageEXT: return "SpvCapabilityInt64ImageEXT";
   case SpvCapabilityShaderClockKHR: return "SpvCapabilityShaderClockKHR";
   case SpvCapabilitySampleMaskOverrideCoverageNV: return "SpvCapabilitySampleMaskOverrideCoverageNV";
   case SpvCapabilityGeometryShaderPassthroughNV: return "SpvCapabilityGeometryShaderPassthroughNV";
   case SpvCapabilityShaderViewportIndexLayerEXT: return "SpvCapabilityShaderViewportIndexLayerEXT";
   case SpvCapabilityShaderViewportMaskNV: return "SpvCapabilityShaderViewportMaskNV";
   case SpvCapabilityShaderStereoViewNV: return "SpvCapabilityShaderStereoViewNV";
   case SpvCapabilityPerViewAttributesNV: return "SpvCapabilityPerViewAttributesNV";
   case SpvCapabilityFragmentFullyCoveredEXT: return "SpvCapabilityFragmentFullyCoveredEXT";
   case SpvCapabilityMeshShadingNV: return "SpvCapabilityMeshShadingNV";
   case SpvCapabilityImageFootprintNV: return "SpvCapabilityImageFootprintNV";
   case SpvCapabilityMeshShadingEXT: return "SpvCapabilityMeshShadingEXT";
   case SpvCapabilityFragmentBarycentricKHR: return "SpvCapabilityFragmentBarycentricKHR";
   case SpvCapabilityComputeDerivativeGroupQuadsNV: return "SpvCapabilityComputeDerivativeGroupQuadsNV";
   case SpvCapabilityFragmentDensityEXT: return "SpvCapabilityFragmentDensityEXT";
   case SpvCapabilityGroupNonUniformPartitionedNV: return "SpvCapabilityGroupNonUniformPartitionedNV";
   case SpvCapabilityShaderNonUniform: return "SpvCapabilityShaderNonUniform";
   case SpvCapabilityRuntimeDescriptorArray: return "SpvCapabilityRuntimeDescriptorArray";
   case SpvCapabilityInputAttachmentArrayDynamicIndexing: return "SpvCapabilityInputAttachmentArrayDynamicIndexing";
   case SpvCapabilityUniformTexelBufferArrayDynamicIndexing: return "SpvCapabilityUniformTexelBufferArrayDynamicIndexing";
   case SpvCapabilityStorageTexelBufferArrayDynamicIndexing: return "SpvCapabilityStorageTexelBufferArrayDynamicIndexing";
   case SpvCapabilityUniformBufferArrayNonUniformIndexing: return "SpvCapabilityUniformBufferArrayNonUniformIndexing";
   case SpvCapabilitySampledImageArrayNonUniformIndexing: return "SpvCapabilitySampledImageArrayNonUniformIndexing";
   case SpvCapabilityStorageBufferArrayNonUniformIndexing: return "SpvCapabilityStorageBufferArrayNonUniformIndexing";
   case SpvCapabilityStorageImageArrayNonUniformIndexing: return "SpvCapabilityStorageImageArrayNonUniformIndexing";
   case SpvCapabilityInputAttachmentArrayNonUniformIndexing: return "SpvCapabilityInputAttachmentArrayNonUniformIndexing";
   case SpvCapabilityUniformTexelBufferArrayNonUniformIndexing: return "SpvCapabilityUniformTexelBufferArrayNonUniformIndexing";
   case SpvCapabilityStorageTexelBufferArrayNonUniformIndexing: return "SpvCapabilityStorageTexelBufferArrayNonUniformIndexing";
   case SpvCapabilityRayTracingNV: return "SpvCapabilityRayTracingNV";
   case SpvCapabilityRayTracingMotionBlurNV: return "SpvCapabilityRayTracingMotionBlurNV";
   case SpvCapabilityVulkanMemoryModel: return "SpvCapabilityVulkanMemoryModel";
   case SpvCapabilityVulkanMemoryModelDeviceScope: return "SpvCapabilityVulkanMemoryModelDeviceScope";
   case SpvCapabilityPhysicalStorageBufferAddresses: return "SpvCapabilityPhysicalStorageBufferAddresses";
   case SpvCapabilityComputeDerivativeGroupLinearNV: return "SpvCapabilityComputeDerivativeGroupLinearNV";
   case SpvCapabilityRayTracingProvisionalKHR: return "SpvCapabilityRayTracingProvisionalKHR";
   case SpvCapabilityCooperativeMatrixNV: return "SpvCapabilityCooperativeMatrixNV";
   case SpvCapabilityFragmentShaderSampleInterlockEXT: return "SpvCapabilityFragmentShaderSampleInterlockEXT";
   case SpvCapabilityFragmentShaderShadingRateInterlockEXT: return "SpvCapabilityFragmentShaderShadingRateInterlockEXT";
   case SpvCapabilityShaderSMBuiltinsNV: return "SpvCapabilityShaderSMBuiltinsNV";
   case SpvCapabilityFragmentShaderPixelInterlockEXT: return "SpvCapabilityFragmentShaderPixelInterlockEXT";
   case SpvCapabilityDemoteToHelperInvocation: return "SpvCapabilityDemoteToHelperInvocation";
   case SpvCapabilityBindlessTextureNV: return "SpvCapabilityBindlessTextureNV";
   case SpvCapabilitySubgroupShuffleINTEL: return "SpvCapabilitySubgroupShuffleINTEL";
   case SpvCapabilitySubgroupBufferBlockIOINTEL: return "SpvCapabilitySubgroupBufferBlockIOINTEL";
   case SpvCapabilitySubgroupImageBlockIOINTEL: return "SpvCapabilitySubgroupImageBlockIOINTEL";
   case SpvCapabilitySubgroupImageMediaBlockIOINTEL: return "SpvCapabilitySubgroupImageMediaBlockIOINTEL";
   case SpvCapabilityRoundToInfinityINTEL: return "SpvCapabilityRoundToInfinityINTEL";
   case SpvCapabilityFloatingPointModeINTEL: return "SpvCapabilityFloatingPointModeINTEL";
   case SpvCapabilityIntegerFunctions2INTEL: return "SpvCapabilityIntegerFunctions2INTEL";
   case SpvCapabilityFunctionPointersINTEL: return "SpvCapabilityFunctionPointersINTEL";
   case SpvCapabilityIndirectReferencesINTEL: return "SpvCapabilityIndirectReferencesINTEL";
   case SpvCapabilityAsmINTEL: return "SpvCapabilityAsmINTEL";
   case SpvCapabilityAtomicFloat32MinMaxEXT: return "SpvCapabilityAtomicFloat32MinMaxEXT";
   case SpvCapabilityAtomicFloat64MinMaxEXT: return "SpvCapabilityAtomicFloat64MinMaxEXT";
   case SpvCapabilityAtomicFloat16MinMaxEXT: return "SpvCapabilityAtomicFloat16MinMaxEXT";
   case SpvCapabilityVectorComputeINTEL: return "SpvCapabilityVectorComputeINTEL";
   case SpvCapabilityVectorAnyINTEL: return "SpvCapabilityVectorAnyINTEL";
   case SpvCapabilityExpectAssumeKHR: return "SpvCapabilityExpectAssumeKHR";
   case SpvCapabilitySubgroupAvcMotionEstimationINTEL: return "SpvCapabilitySubgroupAvcMotionEstimationINTEL";
   case SpvCapabilitySubgroupAvcMotionEstimationIntraINTEL: return "SpvCapabilitySubgroupAvcMotionEstimationIntraINTEL";
   case SpvCapabilitySubgroupAvcMotionEstimationChromaINTEL: return "SpvCapabilitySubgroupAvcMotionEstimationChromaINTEL";
   case SpvCapabilityVariableLengthArrayINTEL: return "SpvCapabilityVariableLengthArrayINTEL";
   case SpvCapabilityFunctionFloatControlINTEL: return "SpvCapabilityFunctionFloatControlINTEL";
   case SpvCapabilityFPGAMemoryAttributesINTEL: return "SpvCapabilityFPGAMemoryAttributesINTEL";
   case SpvCapabilityFPFastMathModeINTEL: return "SpvCapabilityFPFastMathModeINTEL";
   case SpvCapabilityArbitraryPrecisionIntegersINTEL: return "SpvCapabilityArbitraryPrecisionIntegersINTEL";
   case SpvCapabilityArbitraryPrecisionFloatingPointINTEL: return "SpvCapabilityArbitraryPrecisionFloatingPointINTEL";
   case SpvCapabilityUnstructuredLoopControlsINTEL: return "SpvCapabilityUnstructuredLoopControlsINTEL";
   case SpvCapabilityFPGALoopControlsINTEL: return "SpvCapabilityFPGALoopControlsINTEL";
   case SpvCapabilityKernelAttributesINTEL: return "SpvCapabilityKernelAttributesINTEL";
   case SpvCapabilityFPGAKernelAttributesINTEL: return "SpvCapabilityFPGAKernelAttributesINTEL";
   case SpvCapabilityFPGAMemoryAccessesINTEL: return "SpvCapabilityFPGAMemoryAccessesINTEL";
   case SpvCapabilityFPGAClusterAttributesINTEL: return "SpvCapabilityFPGAClusterAttributesINTEL";
   case SpvCapabilityLoopFuseINTEL: return "SpvCapabilityLoopFuseINTEL";
   case SpvCapabilityMemoryAccessAliasingINTEL: return "SpvCapabilityMemoryAccessAliasingINTEL";
   case SpvCapabilityFPGABufferLocationINTEL: return "SpvCapabilityFPGABufferLocationINTEL";
   case SpvCapabilityArbitraryPrecisionFixedPointINTEL: return "SpvCapabilityArbitraryPrecisionFixedPointINTEL";
   case SpvCapabilityUSMStorageClassesINTEL: return "SpvCapabilityUSMStorageClassesINTEL";
   case SpvCapabilityIOPipesINTEL: return "SpvCapabilityIOPipesINTEL";
   case SpvCapabilityBlockingPipesINTEL: return "SpvCapabilityBlockingPipesINTEL";
   case SpvCapabilityFPGARegINTEL: return "SpvCapabilityFPGARegINTEL";
   case SpvCapabilityDotProductInputAll: return "SpvCapabilityDotProductInputAll";
   case SpvCapabilityDotProductInput4x8Bit: return "SpvCapabilityDotProductInput4x8Bit";
   case SpvCapabilityDotProductInput4x8BitPacked: return "SpvCapabilityDotProductInput4x8BitPacked";
   case SpvCapabilityDotProduct: return "SpvCapabilityDotProduct";
   case SpvCapabilityRayCullMaskKHR: return "SpvCapabilityRayCullMaskKHR";
   case SpvCapabilityBitInstructions: return "SpvCapabilityBitInstructions";
   case SpvCapabilityGroupNonUniformRotateKHR: return "SpvCapabilityGroupNonUniformRotateKHR";
   case SpvCapabilityAtomicFloat32AddEXT: return "SpvCapabilityAtomicFloat32AddEXT";
   case SpvCapabilityAtomicFloat64AddEXT: return "SpvCapabilityAtomicFloat64AddEXT";
   case SpvCapabilityLongConstantCompositeINTEL: return "SpvCapabilityLongConstantCompositeINTEL";
   case SpvCapabilityOptNoneINTEL: return "SpvCapabilityOptNoneINTEL";
   case SpvCapabilityAtomicFloat16AddEXT: return "SpvCapabilityAtomicFloat16AddEXT";
   case SpvCapabilityDebugInfoModuleINTEL: return "SpvCapabilityDebugInfoModuleINTEL";
   case SpvCapabilitySplitBarrierINTEL: return "SpvCapabilitySplitBarrierINTEL";
   case SpvCapabilityGroupUniformArithmeticKHR: return "SpvCapabilityGroupUniformArithmeticKHR";
   case SpvCapabilityMax: break; /* silence warnings about unhandled enums. */
   }

   return "unknown";
}

const char *
spirv_decoration_to_string(SpvDecoration v)
{
   switch (v) {
   case SpvDecorationRelaxedPrecision: return "SpvDecorationRelaxedPrecision";
   case SpvDecorationSpecId: return "SpvDecorationSpecId";
   case SpvDecorationBlock: return "SpvDecorationBlock";
   case SpvDecorationBufferBlock: return "SpvDecorationBufferBlock";
   case SpvDecorationRowMajor: return "SpvDecorationRowMajor";
   case SpvDecorationColMajor: return "SpvDecorationColMajor";
   case SpvDecorationArrayStride: return "SpvDecorationArrayStride";
   case SpvDecorationMatrixStride: return "SpvDecorationMatrixStride";
   case SpvDecorationGLSLShared: return "SpvDecorationGLSLShared";
   case SpvDecorationGLSLPacked: return "SpvDecorationGLSLPacked";
   case SpvDecorationCPacked: return "SpvDecorationCPacked";
   case SpvDecorationBuiltIn: return "SpvDecorationBuiltIn";
   case SpvDecorationNoPerspective: return "SpvDecorationNoPerspective";
   case SpvDecorationFlat: return "SpvDecorationFlat";
   case SpvDecorationPatch: return "SpvDecorationPatch";
   case SpvDecorationCentroid: return "SpvDecorationCentroid";
   case SpvDecorationSample: return "SpvDecorationSample";
   case SpvDecorationInvariant: return "SpvDecorationInvariant";
   case SpvDecorationRestrict: return "SpvDecorationRestrict";
   case SpvDecorationAliased: return "SpvDecorationAliased";
   case SpvDecorationVolatile: return "SpvDecorationVolatile";
   case SpvDecorationConstant: return "SpvDecorationConstant";
   case SpvDecorationCoherent: return "SpvDecorationCoherent";
   case SpvDecorationNonWritable: return "SpvDecorationNonWritable";
   case SpvDecorationNonReadable: return "SpvDecorationNonReadable";
   case SpvDecorationUniform: return "SpvDecorationUniform";
   case SpvDecorationUniformId: return "SpvDecorationUniformId";
   case SpvDecorationSaturatedConversion: return "SpvDecorationSaturatedConversion";
   case SpvDecorationStream: return "SpvDecorationStream";
   case SpvDecorationLocation: return "SpvDecorationLocation";
   case SpvDecorationComponent: return "SpvDecorationComponent";
   case SpvDecorationIndex: return "SpvDecorationIndex";
   case SpvDecorationBinding: return "SpvDecorationBinding";
   case SpvDecorationDescriptorSet: return "SpvDecorationDescriptorSet";
   case SpvDecorationOffset: return "SpvDecorationOffset";
   case SpvDecorationXfbBuffer: return "SpvDecorationXfbBuffer";
   case SpvDecorationXfbStride: return "SpvDecorationXfbStride";
   case SpvDecorationFuncParamAttr: return "SpvDecorationFuncParamAttr";
   case SpvDecorationFPRoundingMode: return "SpvDecorationFPRoundingMode";
   case SpvDecorationFPFastMathMode: return "SpvDecorationFPFastMathMode";
   case SpvDecorationLinkageAttributes: return "SpvDecorationLinkageAttributes";
   case SpvDecorationNoContraction: return "SpvDecorationNoContraction";
   case SpvDecorationInputAttachmentIndex: return "SpvDecorationInputAttachmentIndex";
   case SpvDecorationAlignment: return "SpvDecorationAlignment";
   case SpvDecorationMaxByteOffset: return "SpvDecorationMaxByteOffset";
   case SpvDecorationAlignmentId: return "SpvDecorationAlignmentId";
   case SpvDecorationMaxByteOffsetId: return "SpvDecorationMaxByteOffsetId";
   case SpvDecorationNoSignedWrap: return "SpvDecorationNoSignedWrap";
   case SpvDecorationNoUnsignedWrap: return "SpvDecorationNoUnsignedWrap";
   case SpvDecorationExplicitInterpAMD: return "SpvDecorationExplicitInterpAMD";
   case SpvDecorationOverrideCoverageNV: return "SpvDecorationOverrideCoverageNV";
   case SpvDecorationPassthroughNV: return "SpvDecorationPassthroughNV";
   case SpvDecorationViewportRelativeNV: return "SpvDecorationViewportRelativeNV";
   case SpvDecorationSecondaryViewportRelativeNV: return "SpvDecorationSecondaryViewportRelativeNV";
   case SpvDecorationPerPrimitiveNV: return "SpvDecorationPerPrimitiveNV";
   case SpvDecorationPerViewNV: return "SpvDecorationPerViewNV";
   case SpvDecorationPerTaskNV: return "SpvDecorationPerTaskNV";
   case SpvDecorationPerVertexKHR: return "SpvDecorationPerVertexKHR";
   case SpvDecorationNonUniform: return "SpvDecorationNonUniform";
   case SpvDecorationRestrictPointer: return "SpvDecorationRestrictPointer";
   case SpvDecorationAliasedPointer: return "SpvDecorationAliasedPointer";
   case SpvDecorationBindlessSamplerNV: return "SpvDecorationBindlessSamplerNV";
   case SpvDecorationBindlessImageNV: return "SpvDecorationBindlessImageNV";
   case SpvDecorationBoundSamplerNV: return "SpvDecorationBoundSamplerNV";
   case SpvDecorationBoundImageNV: return "SpvDecorationBoundImageNV";
   case SpvDecorationSIMTCallINTEL: return "SpvDecorationSIMTCallINTEL";
   case SpvDecorationReferencedIndirectlyINTEL: return "SpvDecorationReferencedIndirectlyINTEL";
   case SpvDecorationClobberINTEL: return "SpvDecorationClobberINTEL";
   case SpvDecorationSideEffectsINTEL: return "SpvDecorationSideEffectsINTEL";
   case SpvDecorationVectorComputeVariableINTEL: return "SpvDecorationVectorComputeVariableINTEL";
   case SpvDecorationFuncParamIOKindINTEL: return "SpvDecorationFuncParamIOKindINTEL";
   case SpvDecorationVectorComputeFunctionINTEL: return "SpvDecorationVectorComputeFunctionINTEL";
   case SpvDecorationStackCallINTEL: return "SpvDecorationStackCallINTEL";
   case SpvDecorationGlobalVariableOffsetINTEL: return "SpvDecorationGlobalVariableOffsetINTEL";
   case SpvDecorationCounterBuffer: return "SpvDecorationCounterBuffer";
   case SpvDecorationUserSemantic: return "SpvDecorationUserSemantic";
   case SpvDecorationUserTypeGOOGLE: return "SpvDecorationUserTypeGOOGLE";
   case SpvDecorationFunctionRoundingModeINTEL: return "SpvDecorationFunctionRoundingModeINTEL";
   case SpvDecorationFunctionDenormModeINTEL: return "SpvDecorationFunctionDenormModeINTEL";
   case SpvDecorationRegisterINTEL: return "SpvDecorationRegisterINTEL";
   case SpvDecorationMemoryINTEL: return "SpvDecorationMemoryINTEL";
   case SpvDecorationNumbanksINTEL: return "SpvDecorationNumbanksINTEL";
   case SpvDecorationBankwidthINTEL: return "SpvDecorationBankwidthINTEL";
   case SpvDecorationMaxPrivateCopiesINTEL: return "SpvDecorationMaxPrivateCopiesINTEL";
   case SpvDecorationSinglepumpINTEL: return "SpvDecorationSinglepumpINTEL";
   case SpvDecorationDoublepumpINTEL: return "SpvDecorationDoublepumpINTEL";
   case SpvDecorationMaxReplicatesINTEL: return "SpvDecorationMaxReplicatesINTEL";
   case SpvDecorationSimpleDualPortINTEL: return "SpvDecorationSimpleDualPortINTEL";
   case SpvDecorationMergeINTEL: return "SpvDecorationMergeINTEL";
   case SpvDecorationBankBitsINTEL: return "SpvDecorationBankBitsINTEL";
   case SpvDecorationForcePow2DepthINTEL: return "SpvDecorationForcePow2DepthINTEL";
   case SpvDecorationBurstCoalesceINTEL: return "SpvDecorationBurstCoalesceINTEL";
   case SpvDecorationCacheSizeINTEL: return "SpvDecorationCacheSizeINTEL";
   case SpvDecorationDontStaticallyCoalesceINTEL: return "SpvDecorationDontStaticallyCoalesceINTEL";
   case SpvDecorationPrefetchINTEL: return "SpvDecorationPrefetchINTEL";
   case SpvDecorationStallEnableINTEL: return "SpvDecorationStallEnableINTEL";
   case SpvDecorationFuseLoopsInFunctionINTEL: return "SpvDecorationFuseLoopsInFunctionINTEL";
   case SpvDecorationAliasScopeINTEL: return "SpvDecorationAliasScopeINTEL";
   case SpvDecorationNoAliasINTEL: return "SpvDecorationNoAliasINTEL";
   case SpvDecorationBufferLocationINTEL: return "SpvDecorationBufferLocationINTEL";
   case SpvDecorationIOPipeStorageINTEL: return "SpvDecorationIOPipeStorageINTEL";
   case SpvDecorationFunctionFloatingPointModeINTEL: return "SpvDecorationFunctionFloatingPointModeINTEL";
   case SpvDecorationSingleElementVectorINTEL: return "SpvDecorationSingleElementVectorINTEL";
   case SpvDecorationVectorComputeCallableFunctionINTEL: return "SpvDecorationVectorComputeCallableFunctionINTEL";
   case SpvDecorationMediaBlockIOINTEL: return "SpvDecorationMediaBlockIOINTEL";
   case SpvDecorationMax: break; /* silence warnings about unhandled enums. */
   }

   return "unknown";
}

const char *
spirv_dim_to_string(SpvDim v)
{
   switch (v) {
   case SpvDim1D: return "SpvDim1D";
   case SpvDim2D: return "SpvDim2D";
   case SpvDim3D: return "SpvDim3D";
   case SpvDimCube: return "SpvDimCube";
   case SpvDimRect: return "SpvDimRect";
   case SpvDimBuffer: return "SpvDimBuffer";
   case SpvDimSubpassData: return "SpvDimSubpassData";
   case SpvDimMax: break; /* silence warnings about unhandled enums. */
   }

   return "unknown";
}

const char *
spirv_executionmode_to_string(SpvExecutionMode v)
{
   switch (v) {
   case SpvExecutionModeInvocations: return "SpvExecutionModeInvocations";
   case SpvExecutionModeSpacingEqual: return "SpvExecutionModeSpacingEqual";
   case SpvExecutionModeSpacingFractionalEven: return "SpvExecutionModeSpacingFractionalEven";
   case SpvExecutionModeSpacingFractionalOdd: return "SpvExecutionModeSpacingFractionalOdd";
   case SpvExecutionModeVertexOrderCw: return "SpvExecutionModeVertexOrderCw";
   case SpvExecutionModeVertexOrderCcw: return "SpvExecutionModeVertexOrderCcw";
   case SpvExecutionModePixelCenterInteger: return "SpvExecutionModePixelCenterInteger";
   case SpvExecutionModeOriginUpperLeft: return "SpvExecutionModeOriginUpperLeft";
   case SpvExecutionModeOriginLowerLeft: return "SpvExecutionModeOriginLowerLeft";
   case SpvExecutionModeEarlyFragmentTests: return "SpvExecutionModeEarlyFragmentTests";
   case SpvExecutionModePointMode: return "SpvExecutionModePointMode";
   case SpvExecutionModeXfb: return "SpvExecutionModeXfb";
   case SpvExecutionModeDepthReplacing: return "SpvExecutionModeDepthReplacing";
   case SpvExecutionModeDepthGreater: return "SpvExecutionModeDepthGreater";
   case SpvExecutionModeDepthLess: return "SpvExecutionModeDepthLess";
   case SpvExecutionModeDepthUnchanged: return "SpvExecutionModeDepthUnchanged";
   case SpvExecutionModeLocalSize: return "SpvExecutionModeLocalSize";
   case SpvExecutionModeLocalSizeHint: return "SpvExecutionModeLocalSizeHint";
   case SpvExecutionModeInputPoints: return "SpvExecutionModeInputPoints";
   case SpvExecutionModeInputLines: return "SpvExecutionModeInputLines";
   case SpvExecutionModeInputLinesAdjacency: return "SpvExecutionModeInputLinesAdjacency";
   case SpvExecutionModeTriangles: return "SpvExecutionModeTriangles";
   case SpvExecutionModeInputTrianglesAdjacency: return "SpvExecutionModeInputTrianglesAdjacency";
   case SpvExecutionModeQuads: return "SpvExecutionModeQuads";
   case SpvExecutionModeIsolines: return "SpvExecutionModeIsolines";
   case SpvExecutionModeOutputVertices: return "SpvExecutionModeOutputVertices";
   case SpvExecutionModeOutputPoints: return "SpvExecutionModeOutputPoints";
   case SpvExecutionModeOutputLineStrip: return "SpvExecutionModeOutputLineStrip";
   case SpvExecutionModeOutputTriangleStrip: return "SpvExecutionModeOutputTriangleStrip";
   case SpvExecutionModeVecTypeHint: return "SpvExecutionModeVecTypeHint";
   case SpvExecutionModeContractionOff: return "SpvExecutionModeContractionOff";
   case SpvExecutionModeInitializer: return "SpvExecutionModeInitializer";
   case SpvExecutionModeFinalizer: return "SpvExecutionModeFinalizer";
   case SpvExecutionModeSubgroupSize: return "SpvExecutionModeSubgroupSize";
   case SpvExecutionModeSubgroupsPerWorkgroup: return "SpvExecutionModeSubgroupsPerWorkgroup";
   case SpvExecutionModeSubgroupsPerWorkgroupId: return "SpvExecutionModeSubgroupsPerWorkgroupId";
   case SpvExecutionModeLocalSizeId: return "SpvExecutionModeLocalSizeId";
   case SpvExecutionModeLocalSizeHintId: return "SpvExecutionModeLocalSizeHintId";
   case SpvExecutionModeSubgroupUniformControlFlowKHR: return "SpvExecutionModeSubgroupUniformControlFlowKHR";
   case SpvExecutionModePostDepthCoverage: return "SpvExecutionModePostDepthCoverage";
   case SpvExecutionModeDenormPreserve: return "SpvExecutionModeDenormPreserve";
   case SpvExecutionModeDenormFlushToZero: return "SpvExecutionModeDenormFlushToZero";
   case SpvExecutionModeSignedZeroInfNanPreserve: return "SpvExecutionModeSignedZeroInfNanPreserve";
   case SpvExecutionModeRoundingModeRTE: return "SpvExecutionModeRoundingModeRTE";
   case SpvExecutionModeRoundingModeRTZ: return "SpvExecutionModeRoundingModeRTZ";
   case SpvExecutionModeEarlyAndLateFragmentTestsAMD: return "SpvExecutionModeEarlyAndLateFragmentTestsAMD";
   case SpvExecutionModeStencilRefReplacingEXT: return "SpvExecutionModeStencilRefReplacingEXT";
   case SpvExecutionModeStencilRefUnchangedFrontAMD: return "SpvExecutionModeStencilRefUnchangedFrontAMD";
   case SpvExecutionModeStencilRefGreaterFrontAMD: return "SpvExecutionModeStencilRefGreaterFrontAMD";
   case SpvExecutionModeStencilRefLessFrontAMD: return "SpvExecutionModeStencilRefLessFrontAMD";
   case SpvExecutionModeStencilRefUnchangedBackAMD: return "SpvExecutionModeStencilRefUnchangedBackAMD";
   case SpvExecutionModeStencilRefGreaterBackAMD: return "SpvExecutionModeStencilRefGreaterBackAMD";
   case SpvExecutionModeStencilRefLessBackAMD: return "SpvExecutionModeStencilRefLessBackAMD";
   case SpvExecutionModeOutputLinesNV: return "SpvExecutionModeOutputLinesNV";
   case SpvExecutionModeOutputPrimitivesNV: return "SpvExecutionModeOutputPrimitivesNV";
   case SpvExecutionModeDerivativeGroupQuadsNV: return "SpvExecutionModeDerivativeGroupQuadsNV";
   case SpvExecutionModeDerivativeGroupLinearNV: return "SpvExecutionModeDerivativeGroupLinearNV";
   case SpvExecutionModeOutputTrianglesNV: return "SpvExecutionModeOutputTrianglesNV";
   case SpvExecutionModePixelInterlockOrderedEXT: return "SpvExecutionModePixelInterlockOrderedEXT";
   case SpvExecutionModePixelInterlockUnorderedEXT: return "SpvExecutionModePixelInterlockUnorderedEXT";
   case SpvExecutionModeSampleInterlockOrderedEXT: return "SpvExecutionModeSampleInterlockOrderedEXT";
   case SpvExecutionModeSampleInterlockUnorderedEXT: return "SpvExecutionModeSampleInterlockUnorderedEXT";
   case SpvExecutionModeShadingRateInterlockOrderedEXT: return "SpvExecutionModeShadingRateInterlockOrderedEXT";
   case SpvExecutionModeShadingRateInterlockUnorderedEXT: return "SpvExecutionModeShadingRateInterlockUnorderedEXT";
   case SpvExecutionModeSharedLocalMemorySizeINTEL: return "SpvExecutionModeSharedLocalMemorySizeINTEL";
   case SpvExecutionModeRoundingModeRTPINTEL: return "SpvExecutionModeRoundingModeRTPINTEL";
   case SpvExecutionModeRoundingModeRTNINTEL: return "SpvExecutionModeRoundingModeRTNINTEL";
   case SpvExecutionModeFloatingPointModeALTINTEL: return "SpvExecutionModeFloatingPointModeALTINTEL";
   case SpvExecutionModeFloatingPointModeIEEEINTEL: return "SpvExecutionModeFloatingPointModeIEEEINTEL";
   case SpvExecutionModeMaxWorkgroupSizeINTEL: return "SpvExecutionModeMaxWorkgroupSizeINTEL";
   case SpvExecutionModeMaxWorkDimINTEL: return "SpvExecutionModeMaxWorkDimINTEL";
   case SpvExecutionModeNoGlobalOffsetINTEL: return "SpvExecutionModeNoGlobalOffsetINTEL";
   case SpvExecutionModeNumSIMDWorkitemsINTEL: return "SpvExecutionModeNumSIMDWorkitemsINTEL";
   case SpvExecutionModeSchedulerTargetFmaxMhzINTEL: return "SpvExecutionModeSchedulerTargetFmaxMhzINTEL";
   case SpvExecutionModeNamedBarrierCountINTEL: return "SpvExecutionModeNamedBarrierCountINTEL";
   case SpvExecutionModeMax: break; /* silence warnings about unhandled enums. */
   }

   return "unknown";
}

const char *
spirv_executionmodel_to_string(SpvExecutionModel v)
{
   switch (v) {
   case SpvExecutionModelVertex: return "SpvExecutionModelVertex";
   case SpvExecutionModelTessellationControl: return "SpvExecutionModelTessellationControl";
   case SpvExecutionModelTessellationEvaluation: return "SpvExecutionModelTessellationEvaluation";
   case SpvExecutionModelGeometry: return "SpvExecutionModelGeometry";
   case SpvExecutionModelFragment: return "SpvExecutionModelFragment";
   case SpvExecutionModelGLCompute: return "SpvExecutionModelGLCompute";
   case SpvExecutionModelKernel: return "SpvExecutionModelKernel";
   case SpvExecutionModelTaskNV: return "SpvExecutionModelTaskNV";
   case SpvExecutionModelMeshNV: return "SpvExecutionModelMeshNV";
   case SpvExecutionModelRayGenerationNV: return "SpvExecutionModelRayGenerationNV";
   case SpvExecutionModelIntersectionNV: return "SpvExecutionModelIntersectionNV";
   case SpvExecutionModelAnyHitNV: return "SpvExecutionModelAnyHitNV";
   case SpvExecutionModelClosestHitNV: return "SpvExecutionModelClosestHitNV";
   case SpvExecutionModelMissNV: return "SpvExecutionModelMissNV";
   case SpvExecutionModelCallableNV: return "SpvExecutionModelCallableNV";
   case SpvExecutionModelTaskEXT: return "SpvExecutionModelTaskEXT";
   case SpvExecutionModelMeshEXT: return "SpvExecutionModelMeshEXT";
   case SpvExecutionModelMax: break; /* silence warnings about unhandled enums. */
   }

   return "unknown";
}

const char *
spirv_imageformat_to_string(SpvImageFormat v)
{
   switch (v) {
   case SpvImageFormatUnknown: return "SpvImageFormatUnknown";
   case SpvImageFormatRgba32f: return "SpvImageFormatRgba32f";
   case SpvImageFormatRgba16f: return "SpvImageFormatRgba16f";
   case SpvImageFormatR32f: return "SpvImageFormatR32f";
   case SpvImageFormatRgba8: return "SpvImageFormatRgba8";
   case SpvImageFormatRgba8Snorm: return "SpvImageFormatRgba8Snorm";
   case SpvImageFormatRg32f: return "SpvImageFormatRg32f";
   case SpvImageFormatRg16f: return "SpvImageFormatRg16f";
   case SpvImageFormatR11fG11fB10f: return "SpvImageFormatR11fG11fB10f";
   case SpvImageFormatR16f: return "SpvImageFormatR16f";
   case SpvImageFormatRgba16: return "SpvImageFormatRgba16";
   case SpvImageFormatRgb10A2: return "SpvImageFormatRgb10A2";
   case SpvImageFormatRg16: return "SpvImageFormatRg16";
   case SpvImageFormatRg8: return "SpvImageFormatRg8";
   case SpvImageFormatR16: return "SpvImageFormatR16";
   case SpvImageFormatR8: return "SpvImageFormatR8";
   case SpvImageFormatRgba16Snorm: return "SpvImageFormatRgba16Snorm";
   case SpvImageFormatRg16Snorm: return "SpvImageFormatRg16Snorm";
   case SpvImageFormatRg8Snorm: return "SpvImageFormatRg8Snorm";
   case SpvImageFormatR16Snorm: return "SpvImageFormatR16Snorm";
   case SpvImageFormatR8Snorm: return "SpvImageFormatR8Snorm";
   case SpvImageFormatRgba32i: return "SpvImageFormatRgba32i";
   case SpvImageFormatRgba16i: return "SpvImageFormatRgba16i";
   case SpvImageFormatRgba8i: return "SpvImageFormatRgba8i";
   case SpvImageFormatR32i: return "SpvImageFormatR32i";
   case SpvImageFormatRg32i: return "SpvImageFormatRg32i";
   case SpvImageFormatRg16i: return "SpvImageFormatRg16i";
   case SpvImageFormatRg8i: return "SpvImageFormatRg8i";
   case SpvImageFormatR16i: return "SpvImageFormatR16i";
   case SpvImageFormatR8i: return "SpvImageFormatR8i";
   case SpvImageFormatRgba32ui: return "SpvImageFormatRgba32ui";
   case SpvImageFormatRgba16ui: return "SpvImageFormatRgba16ui";
   case SpvImageFormatRgba8ui: return "SpvImageFormatRgba8ui";
   case SpvImageFormatR32ui: return "SpvImageFormatR32ui";
   case SpvImageFormatRgb10a2ui: return "SpvImageFormatRgb10a2ui";
   case SpvImageFormatRg32ui: return "SpvImageFormatRg32ui";
   case SpvImageFormatRg16ui: return "SpvImageFormatRg16ui";
   case SpvImageFormatRg8ui: return "SpvImageFormatRg8ui";
   case SpvImageFormatR16ui: return "SpvImageFormatR16ui";
   case SpvImageFormatR8ui: return "SpvImageFormatR8ui";
   case SpvImageFormatR64ui: return "SpvImageFormatR64ui";
   case SpvImageFormatR64i: return "SpvImageFormatR64i";
   case SpvImageFormatMax: break; /* silence warnings about unhandled enums. */
   }

   return "unknown";
}

const char *
spirv_memorymodel_to_string(SpvMemoryModel v)
{
   switch (v) {
   case SpvMemoryModelSimple: return "SpvMemoryModelSimple";
   case SpvMemoryModelGLSL450: return "SpvMemoryModelGLSL450";
   case SpvMemoryModelOpenCL: return "SpvMemoryModelOpenCL";
   case SpvMemoryModelVulkan: return "SpvMemoryModelVulkan";
   case SpvMemoryModelMax: break; /* silence warnings about unhandled enums. */
   }

   return "unknown";
}

const char *
spirv_storageclass_to_string(SpvStorageClass v)
{
   switch (v) {
   case SpvStorageClassUniformConstant: return "SpvStorageClassUniformConstant";
   case SpvStorageClassInput: return "SpvStorageClassInput";
   case SpvStorageClassUniform: return "SpvStorageClassUniform";
   case SpvStorageClassOutput: return "SpvStorageClassOutput";
   case SpvStorageClassWorkgroup: return "SpvStorageClassWorkgroup";
   case SpvStorageClassCrossWorkgroup: return "SpvStorageClassCrossWorkgroup";
   case SpvStorageClassPrivate: return "SpvStorageClassPrivate";
   case SpvStorageClassFunction: return "SpvStorageClassFunction";
   case SpvStorageClassGeneric: return "SpvStorageClassGeneric";
   case SpvStorageClassPushConstant: return "SpvStorageClassPushConstant";
   case SpvStorageClassAtomicCounter: return "SpvStorageClassAtomicCounter";
   case SpvStorageClassImage: return "SpvStorageClassImage";
   case SpvStorageClassStorageBuffer: return "SpvStorageClassStorageBuffer";
   case SpvStorageClassCallableDataNV: return "SpvStorageClassCallableDataNV";
   case SpvStorageClassIncomingCallableDataNV: return "SpvStorageClassIncomingCallableDataNV";
   case SpvStorageClassRayPayloadNV: return "SpvStorageClassRayPayloadNV";
   case SpvStorageClassHitAttributeNV: return "SpvStorageClassHitAttributeNV";
   case SpvStorageClassIncomingRayPayloadNV: return "SpvStorageClassIncomingRayPayloadNV";
   case SpvStorageClassShaderRecordBufferNV: return "SpvStorageClassShaderRecordBufferNV";
   case SpvStorageClassPhysicalStorageBuffer: return "SpvStorageClassPhysicalStorageBuffer";
   case SpvStorageClassTaskPayloadWorkgroupEXT: return "SpvStorageClassTaskPayloadWorkgroupEXT";
   case SpvStorageClassCodeSectionINTEL: return "SpvStorageClassCodeSectionINTEL";
   case SpvStorageClassDeviceOnlyINTEL: return "SpvStorageClassDeviceOnlyINTEL";
   case SpvStorageClassHostOnlyINTEL: return "SpvStorageClassHostOnlyINTEL";
   case SpvStorageClassMax: break; /* silence warnings about unhandled enums. */
   }

   return "unknown";
}

const char *
spirv_imageoperands_to_string(SpvImageOperandsMask v)
{
   switch (v) {
   case SpvImageOperandsMaskNone: return "SpvImageOperandsNone";
   case SpvImageOperandsBiasMask: return "SpvImageOperandsBias";
   case SpvImageOperandsLodMask: return "SpvImageOperandsLod";
   case SpvImageOperandsGradMask: return "SpvImageOperandsGrad";
   case SpvImageOperandsConstOffsetMask: return "SpvImageOperandsConstOffset";
   case SpvImageOperandsOffsetMask: return "SpvImageOperandsOffset";
   case SpvImageOperandsConstOffsetsMask: return "SpvImageOperandsConstOffsets";
   case SpvImageOperandsSampleMask: return "SpvImageOperandsSample";
   case SpvImageOperandsMinLodMask: return "SpvImageOperandsMinLod";
   case SpvImageOperandsMakeTexelAvailableMask: return "SpvImageOperandsMakeTexelAvailable";
   case SpvImageOperandsMakeTexelVisibleMask: return "SpvImageOperandsMakeTexelVisible";
   case SpvImageOperandsNonPrivateTexelMask: return "SpvImageOperandsNonPrivateTexel";
   case SpvImageOperandsVolatileTexelMask: return "SpvImageOperandsVolatileTexel";
   case SpvImageOperandsSignExtendMask: return "SpvImageOperandsSignExtend";
   case SpvImageOperandsZeroExtendMask: return "SpvImageOperandsZeroExtend";
   case SpvImageOperandsNontemporalMask: return "SpvImageOperandsNontemporal";
   case SpvImageOperandsOffsetsMask: return "SpvImageOperandsOffsets";
   }

   return "unknown";
}

const char *
spirv_fproundingmode_to_string(SpvFPRoundingMode v)
{
   switch (v) {
   case SpvFPRoundingModeRTE: return "SpvFPRoundingModeRTE";
   case SpvFPRoundingModeRTZ: return "SpvFPRoundingModeRTZ";
   case SpvFPRoundingModeRTP: return "SpvFPRoundingModeRTP";
   case SpvFPRoundingModeRTN: return "SpvFPRoundingModeRTN";
   case SpvFPRoundingModeMax: break; /* silence warnings about unhandled enums. */
   }

   return "unknown";
}

const char *
spirv_op_to_string(SpvOp v)
{
   switch (v) {
   case SpvOpNop: return "SpvOpNop";
   case SpvOpUndef: return "SpvOpUndef";
   case SpvOpSourceContinued: return "SpvOpSourceContinued";
   case SpvOpSource: return "SpvOpSource";
   case SpvOpSourceExtension: return "SpvOpSourceExtension";
   case SpvOpName: return "SpvOpName";
   case SpvOpMemberName: return "SpvOpMemberName";
   case SpvOpString: return "SpvOpString";
   case SpvOpLine: return "SpvOpLine";
   case SpvOpExtension: return "SpvOpExtension";
   case SpvOpExtInstImport: return "SpvOpExtInstImport";
   case SpvOpExtInst: return "SpvOpExtInst";
   case SpvOpMemoryModel: return "SpvOpMemoryModel";
   case SpvOpEntryPoint: return "SpvOpEntryPoint";
   case SpvOpExecutionMode: return "SpvOpExecutionMode";
   case SpvOpCapability: return "SpvOpCapability";
   case SpvOpTypeVoid: return "SpvOpTypeVoid";
   case SpvOpTypeBool: return "SpvOpTypeBool";
   case SpvOpTypeInt: return "SpvOpTypeInt";
   case SpvOpTypeFloat: return "SpvOpTypeFloat";
   case SpvOpTypeVector: return "SpvOpTypeVector";
   case SpvOpTypeMatrix: return "SpvOpTypeMatrix";
   case SpvOpTypeImage: return "SpvOpTypeImage";
   case SpvOpTypeSampler: return "SpvOpTypeSampler";
   case SpvOpTypeSampledImage: return "SpvOpTypeSampledImage";
   case SpvOpTypeArray: return "SpvOpTypeArray";
   case SpvOpTypeRuntimeArray: return "SpvOpTypeRuntimeArray";
   case SpvOpTypeStruct: return "SpvOpTypeStruct";
   case SpvOpTypeOpaque: return "SpvOpTypeOpaque";
   case SpvOpTypePointer: return "SpvOpTypePointer";
   case SpvOpTypeFunction: return "SpvOpTypeFunction";
   case SpvOpTypeEvent: return "SpvOpTypeEvent";
   case SpvOpTypeDeviceEvent: return "SpvOpTypeDeviceEvent";
   case SpvOpTypeReserveId: return "SpvOpTypeReserveId";
   case SpvOpTypeQueue: return "SpvOpTypeQueue";
   case SpvOpTypePipe: return "SpvOpTypePipe";
   case SpvOpTypeForwardPointer: return "SpvOpTypeForwardPointer";
   case SpvOpConstantTrue: return "SpvOpConstantTrue";
   case SpvOpConstantFalse: return "SpvOpConstantFalse";
   case SpvOpConstant: return "SpvOpConstant";
   case SpvOpConstantComposite: return "SpvOpConstantComposite";
   case SpvOpConstantSampler: return "SpvOpConstantSampler";
   case SpvOpConstantNull: return "SpvOpConstantNull";
   case SpvOpSpecConstantTrue: return "SpvOpSpecConstantTrue";
   case SpvOpSpecConstantFalse: return "SpvOpSpecConstantFalse";
   case SpvOpSpecConstant: return "SpvOpSpecConstant";
   case SpvOpSpecConstantComposite: return "SpvOpSpecConstantComposite";
   case SpvOpSpecConstantOp: return "SpvOpSpecConstantOp";
   case SpvOpFunction: return "SpvOpFunction";
   case SpvOpFunctionParameter: return "SpvOpFunctionParameter";
   case SpvOpFunctionEnd: return "SpvOpFunctionEnd";
   case SpvOpFunctionCall: return "SpvOpFunctionCall";
   case SpvOpVariable: return "SpvOpVariable";
   case SpvOpImageTexelPointer: return "SpvOpImageTexelPointer";
   case SpvOpLoad: return "SpvOpLoad";
   case SpvOpStore: return "SpvOpStore";
   case SpvOpCopyMemory: return "SpvOpCopyMemory";
   case SpvOpCopyMemorySized: return "SpvOpCopyMemorySized";
   case SpvOpAccessChain: return "SpvOpAccessChain";
   case SpvOpInBoundsAccessChain: return "SpvOpInBoundsAccessChain";
   case SpvOpPtrAccessChain: return "SpvOpPtrAccessChain";
   case SpvOpArrayLength: return "SpvOpArrayLength";
   case SpvOpGenericPtrMemSemantics: return "SpvOpGenericPtrMemSemantics";
   case SpvOpInBoundsPtrAccessChain: return "SpvOpInBoundsPtrAccessChain";
   case SpvOpDecorate: return "SpvOpDecorate";
   case SpvOpMemberDecorate: return "SpvOpMemberDecorate";
   case SpvOpDecorationGroup: return "SpvOpDecorationGroup";
   case SpvOpGroupDecorate: return "SpvOpGroupDecorate";
   case SpvOpGroupMemberDecorate: return "SpvOpGroupMemberDecorate";
   case SpvOpVectorExtractDynamic: return "SpvOpVectorExtractDynamic";
   case SpvOpVectorInsertDynamic: return "SpvOpVectorInsertDynamic";
   case SpvOpVectorShuffle: return "SpvOpVectorShuffle";
   case SpvOpCompositeConstruct: return "SpvOpCompositeConstruct";
   case SpvOpCompositeExtract: return "SpvOpCompositeExtract";
   case SpvOpCompositeInsert: return "SpvOpCompositeInsert";
   case SpvOpCopyObject: return "SpvOpCopyObject";
   case SpvOpTranspose: return "SpvOpTranspose";
   case SpvOpSampledImage: return "SpvOpSampledImage";
   case SpvOpImageSampleImplicitLod: return "SpvOpImageSampleImplicitLod";
   case SpvOpImageSampleExplicitLod: return "SpvOpImageSampleExplicitLod";
   case SpvOpImageSampleDrefImplicitLod: return "SpvOpImageSampleDrefImplicitLod";
   case SpvOpImageSampleDrefExplicitLod: return "SpvOpImageSampleDrefExplicitLod";
   case SpvOpImageSampleProjImplicitLod: return "SpvOpImageSampleProjImplicitLod";
   case SpvOpImageSampleProjExplicitLod: return "SpvOpImageSampleProjExplicitLod";
   case SpvOpImageSampleProjDrefImplicitLod: return "SpvOpImageSampleProjDrefImplicitLod";
   case SpvOpImageSampleProjDrefExplicitLod: return "SpvOpImageSampleProjDrefExplicitLod";
   case SpvOpImageFetch: return "SpvOpImageFetch";
   case SpvOpImageGather: return "SpvOpImageGather";
   case SpvOpImageDrefGather: return "SpvOpImageDrefGather";
   case SpvOpImageRead: return "SpvOpImageRead";
   case SpvOpImageWrite: return "SpvOpImageWrite";
   case SpvOpImage: return "SpvOpImage";
   case SpvOpImageQueryFormat: return "SpvOpImageQueryFormat";
   case SpvOpImageQueryOrder: return "SpvOpImageQueryOrder";
   case SpvOpImageQuerySizeLod: return "SpvOpImageQuerySizeLod";
   case SpvOpImageQuerySize: return "SpvOpImageQuerySize";
   case SpvOpImageQueryLod: return "SpvOpImageQueryLod";
   case SpvOpImageQueryLevels: return "SpvOpImageQueryLevels";
   case SpvOpImageQuerySamples: return "SpvOpImageQuerySamples";
   case SpvOpConvertFToU: return "SpvOpConvertFToU";
   case SpvOpConvertFToS: return "SpvOpConvertFToS";
   case SpvOpConvertSToF: return "SpvOpConvertSToF";
   case SpvOpConvertUToF: return "SpvOpConvertUToF";
   case SpvOpUConvert: return "SpvOpUConvert";
   case SpvOpSConvert: return "SpvOpSConvert";
   case SpvOpFConvert: return "SpvOpFConvert";
   case SpvOpQuantizeToF16: return "SpvOpQuantizeToF16";
   case SpvOpConvertPtrToU: return "SpvOpConvertPtrToU";
   case SpvOpSatConvertSToU: return "SpvOpSatConvertSToU";
   case SpvOpSatConvertUToS: return "SpvOpSatConvertUToS";
   case SpvOpConvertUToPtr: return "SpvOpConvertUToPtr";
   case SpvOpPtrCastToGeneric: return "SpvOpPtrCastToGeneric";
   case SpvOpGenericCastToPtr: return "SpvOpGenericCastToPtr";
   case SpvOpGenericCastToPtrExplicit: return "SpvOpGenericCastToPtrExplicit";
   case SpvOpBitcast: return "SpvOpBitcast";
   case SpvOpSNegate: return "SpvOpSNegate";
   case SpvOpFNegate: return "SpvOpFNegate";
   case SpvOpIAdd: return "SpvOpIAdd";
   case SpvOpFAdd: return "SpvOpFAdd";
   case SpvOpISub: return "SpvOpISub";
   case SpvOpFSub: return "SpvOpFSub";
   case SpvOpIMul: return "SpvOpIMul";
   case SpvOpFMul: return "SpvOpFMul";
   case SpvOpUDiv: return "SpvOpUDiv";
   case SpvOpSDiv: return "SpvOpSDiv";
   case SpvOpFDiv: return "SpvOpFDiv";
   case SpvOpUMod: return "SpvOpUMod";
   case SpvOpSRem: return "SpvOpSRem";
   case SpvOpSMod: return "SpvOpSMod";
   case SpvOpFRem: return "SpvOpFRem";
   case SpvOpFMod: return "SpvOpFMod";
   case SpvOpVectorTimesScalar: return "SpvOpVectorTimesScalar";
   case SpvOpMatrixTimesScalar: return "SpvOpMatrixTimesScalar";
   case SpvOpVectorTimesMatrix: return "SpvOpVectorTimesMatrix";
   case SpvOpMatrixTimesVector: return "SpvOpMatrixTimesVector";
   case SpvOpMatrixTimesMatrix: return "SpvOpMatrixTimesMatrix";
   case SpvOpOuterProduct: return "SpvOpOuterProduct";
   case SpvOpDot: return "SpvOpDot";
   case SpvOpIAddCarry: return "SpvOpIAddCarry";
   case SpvOpISubBorrow: return "SpvOpISubBorrow";
   case SpvOpUMulExtended: return "SpvOpUMulExtended";
   case SpvOpSMulExtended: return "SpvOpSMulExtended";
   case SpvOpAny: return "SpvOpAny";
   case SpvOpAll: return "SpvOpAll";
   case SpvOpIsNan: return "SpvOpIsNan";
   case SpvOpIsInf: return "SpvOpIsInf";
   case SpvOpIsFinite: return "SpvOpIsFinite";
   case SpvOpIsNormal: return "SpvOpIsNormal";
   case SpvOpSignBitSet: return "SpvOpSignBitSet";
   case SpvOpLessOrGreater: return "SpvOpLessOrGreater";
   case SpvOpOrdered: return "SpvOpOrdered";
   case SpvOpUnordered: return "SpvOpUnordered";
   case SpvOpLogicalEqual: return "SpvOpLogicalEqual";
   case SpvOpLogicalNotEqual: return "SpvOpLogicalNotEqual";
   case SpvOpLogicalOr: return "SpvOpLogicalOr";
   case SpvOpLogicalAnd: return "SpvOpLogicalAnd";
   case SpvOpLogicalNot: return "SpvOpLogicalNot";
   case SpvOpSelect: return "SpvOpSelect";
   case SpvOpIEqual: return "SpvOpIEqual";
   case SpvOpINotEqual: return "SpvOpINotEqual";
   case SpvOpUGreaterThan: return "SpvOpUGreaterThan";
   case SpvOpSGreaterThan: return "SpvOpSGreaterThan";
   case SpvOpUGreaterThanEqual: return "SpvOpUGreaterThanEqual";
   case SpvOpSGreaterThanEqual: return "SpvOpSGreaterThanEqual";
   case SpvOpULessThan: return "SpvOpULessThan";
   case SpvOpSLessThan: return "SpvOpSLessThan";
   case SpvOpULessThanEqual: return "SpvOpULessThanEqual";
   case SpvOpSLessThanEqual: return "SpvOpSLessThanEqual";
   case SpvOpFOrdEqual: return "SpvOpFOrdEqual";
   case SpvOpFUnordEqual: return "SpvOpFUnordEqual";
   case SpvOpFOrdNotEqual: return "SpvOpFOrdNotEqual";
   case SpvOpFUnordNotEqual: return "SpvOpFUnordNotEqual";
   case SpvOpFOrdLessThan: return "SpvOpFOrdLessThan";
   case SpvOpFUnordLessThan: return "SpvOpFUnordLessThan";
   case SpvOpFOrdGreaterThan: return "SpvOpFOrdGreaterThan";
   case SpvOpFUnordGreaterThan: return "SpvOpFUnordGreaterThan";
   case SpvOpFOrdLessThanEqual: return "SpvOpFOrdLessThanEqual";
   case SpvOpFUnordLessThanEqual: return "SpvOpFUnordLessThanEqual";
   case SpvOpFOrdGreaterThanEqual: return "SpvOpFOrdGreaterThanEqual";
   case SpvOpFUnordGreaterThanEqual: return "SpvOpFUnordGreaterThanEqual";
   case SpvOpShiftRightLogical: return "SpvOpShiftRightLogical";
   case SpvOpShiftRightArithmetic: return "SpvOpShiftRightArithmetic";
   case SpvOpShiftLeftLogical: return "SpvOpShiftLeftLogical";
   case SpvOpBitwiseOr: return "SpvOpBitwiseOr";
   case SpvOpBitwiseXor: return "SpvOpBitwiseXor";
   case SpvOpBitwiseAnd: return "SpvOpBitwiseAnd";
   case SpvOpNot: return "SpvOpNot";
   case SpvOpBitFieldInsert: return "SpvOpBitFieldInsert";
   case SpvOpBitFieldSExtract: return "SpvOpBitFieldSExtract";
   case SpvOpBitFieldUExtract: return "SpvOpBitFieldUExtract";
   case SpvOpBitReverse: return "SpvOpBitReverse";
   case SpvOpBitCount: return "SpvOpBitCount";
   case SpvOpDPdx: return "SpvOpDPdx";
   case SpvOpDPdy: return "SpvOpDPdy";
   case SpvOpFwidth: return "SpvOpFwidth";
   case SpvOpDPdxFine: return "SpvOpDPdxFine";
   case SpvOpDPdyFine: return "SpvOpDPdyFine";
   case SpvOpFwidthFine: return "SpvOpFwidthFine";
   case SpvOpDPdxCoarse: return "SpvOpDPdxCoarse";
   case SpvOpDPdyCoarse: return "SpvOpDPdyCoarse";
   case SpvOpFwidthCoarse: return "SpvOpFwidthCoarse";
   case SpvOpEmitVertex: return "SpvOpEmitVertex";
   case SpvOpEndPrimitive: return "SpvOpEndPrimitive";
   case SpvOpEmitStreamVertex: return "SpvOpEmitStreamVertex";
   case SpvOpEndStreamPrimitive: return "SpvOpEndStreamPrimitive";
   case SpvOpControlBarrier: return "SpvOpControlBarrier";
   case SpvOpMemoryBarrier: return "SpvOpMemoryBarrier";
   case SpvOpAtomicLoad: return "SpvOpAtomicLoad";
   case SpvOpAtomicStore: return "SpvOpAtomicStore";
   case SpvOpAtomicExchange: return "SpvOpAtomicExchange";
   case SpvOpAtomicCompareExchange: return "SpvOpAtomicCompareExchange";
   case SpvOpAtomicCompareExchangeWeak: return "SpvOpAtomicCompareExchangeWeak";
   case SpvOpAtomicIIncrement: return "SpvOpAtomicIIncrement";
   case SpvOpAtomicIDecrement: return "SpvOpAtomicIDecrement";
   case SpvOpAtomicIAdd: return "SpvOpAtomicIAdd";
   case SpvOpAtomicISub: return "SpvOpAtomicISub";
   case SpvOpAtomicSMin: return "SpvOpAtomicSMin";
   case SpvOpAtomicUMin: return "SpvOpAtomicUMin";
   case SpvOpAtomicSMax: return "SpvOpAtomicSMax";
   case SpvOpAtomicUMax: return "SpvOpAtomicUMax";
   case SpvOpAtomicAnd: return "SpvOpAtomicAnd";
   case SpvOpAtomicOr: return "SpvOpAtomicOr";
   case SpvOpAtomicXor: return "SpvOpAtomicXor";
   case SpvOpPhi: return "SpvOpPhi";
   case SpvOpLoopMerge: return "SpvOpLoopMerge";
   case SpvOpSelectionMerge: return "SpvOpSelectionMerge";
   case SpvOpLabel: return "SpvOpLabel";
   case SpvOpBranch: return "SpvOpBranch";
   case SpvOpBranchConditional: return "SpvOpBranchConditional";
   case SpvOpSwitch: return "SpvOpSwitch";
   case SpvOpKill: return "SpvOpKill";
   case SpvOpReturn: return "SpvOpReturn";
   case SpvOpReturnValue: return "SpvOpReturnValue";
   case SpvOpUnreachable: return "SpvOpUnreachable";
   case SpvOpLifetimeStart: return "SpvOpLifetimeStart";
   case SpvOpLifetimeStop: return "SpvOpLifetimeStop";
   case SpvOpGroupAsyncCopy: return "SpvOpGroupAsyncCopy";
   case SpvOpGroupWaitEvents: return "SpvOpGroupWaitEvents";
   case SpvOpGroupAll: return "SpvOpGroupAll";
   case SpvOpGroupAny: return "SpvOpGroupAny";
   case SpvOpGroupBroadcast: return "SpvOpGroupBroadcast";
   case SpvOpGroupIAdd: return "SpvOpGroupIAdd";
   case SpvOpGroupFAdd: return "SpvOpGroupFAdd";
   case SpvOpGroupFMin: return "SpvOpGroupFMin";
   case SpvOpGroupUMin: return "SpvOpGroupUMin";
   case SpvOpGroupSMin: return "SpvOpGroupSMin";
   case SpvOpGroupFMax: return "SpvOpGroupFMax";
   case SpvOpGroupUMax: return "SpvOpGroupUMax";
   case SpvOpGroupSMax: return "SpvOpGroupSMax";
   case SpvOpReadPipe: return "SpvOpReadPipe";
   case SpvOpWritePipe: return "SpvOpWritePipe";
   case SpvOpReservedReadPipe: return "SpvOpReservedReadPipe";
   case SpvOpReservedWritePipe: return "SpvOpReservedWritePipe";
   case SpvOpReserveReadPipePackets: return "SpvOpReserveReadPipePackets";
   case SpvOpReserveWritePipePackets: return "SpvOpReserveWritePipePackets";
   case SpvOpCommitReadPipe: return "SpvOpCommitReadPipe";
   case SpvOpCommitWritePipe: return "SpvOpCommitWritePipe";
   case SpvOpIsValidReserveId: return "SpvOpIsValidReserveId";
   case SpvOpGetNumPipePackets: return "SpvOpGetNumPipePackets";
   case SpvOpGetMaxPipePackets: return "SpvOpGetMaxPipePackets";
   case SpvOpGroupReserveReadPipePackets: return "SpvOpGroupReserveReadPipePackets";
   case SpvOpGroupReserveWritePipePackets: return "SpvOpGroupReserveWritePipePackets";
   case SpvOpGroupCommitReadPipe: return "SpvOpGroupCommitReadPipe";
   case SpvOpGroupCommitWritePipe: return "SpvOpGroupCommitWritePipe";
   case SpvOpEnqueueMarker: return "SpvOpEnqueueMarker";
   case SpvOpEnqueueKernel: return "SpvOpEnqueueKernel";
   case SpvOpGetKernelNDrangeSubGroupCount: return "SpvOpGetKernelNDrangeSubGroupCount";
   case SpvOpGetKernelNDrangeMaxSubGroupSize: return "SpvOpGetKernelNDrangeMaxSubGroupSize";
   case SpvOpGetKernelWorkGroupSize: return "SpvOpGetKernelWorkGroupSize";
   case SpvOpGetKernelPreferredWorkGroupSizeMultiple: return "SpvOpGetKernelPreferredWorkGroupSizeMultiple";
   case SpvOpRetainEvent: return "SpvOpRetainEvent";
   case SpvOpReleaseEvent: return "SpvOpReleaseEvent";
   case SpvOpCreateUserEvent: return "SpvOpCreateUserEvent";
   case SpvOpIsValidEvent: return "SpvOpIsValidEvent";
   case SpvOpSetUserEventStatus: return "SpvOpSetUserEventStatus";
   case SpvOpCaptureEventProfilingInfo: return "SpvOpCaptureEventProfilingInfo";
   case SpvOpGetDefaultQueue: return "SpvOpGetDefaultQueue";
   case SpvOpBuildNDRange: return "SpvOpBuildNDRange";
   case SpvOpImageSparseSampleImplicitLod: return "SpvOpImageSparseSampleImplicitLod";
   case SpvOpImageSparseSampleExplicitLod: return "SpvOpImageSparseSampleExplicitLod";
   case SpvOpImageSparseSampleDrefImplicitLod: return "SpvOpImageSparseSampleDrefImplicitLod";
   case SpvOpImageSparseSampleDrefExplicitLod: return "SpvOpImageSparseSampleDrefExplicitLod";
   case SpvOpImageSparseSampleProjImplicitLod: return "SpvOpImageSparseSampleProjImplicitLod";
   case SpvOpImageSparseSampleProjExplicitLod: return "SpvOpImageSparseSampleProjExplicitLod";
   case SpvOpImageSparseSampleProjDrefImplicitLod: return "SpvOpImageSparseSampleProjDrefImplicitLod";
   case SpvOpImageSparseSampleProjDrefExplicitLod: return "SpvOpImageSparseSampleProjDrefExplicitLod";
   case SpvOpImageSparseFetch: return "SpvOpImageSparseFetch";
   case SpvOpImageSparseGather: return "SpvOpImageSparseGather";
   case SpvOpImageSparseDrefGather: return "SpvOpImageSparseDrefGather";
   case SpvOpImageSparseTexelsResident: return "SpvOpImageSparseTexelsResident";
   case SpvOpNoLine: return "SpvOpNoLine";
   case SpvOpAtomicFlagTestAndSet: return "SpvOpAtomicFlagTestAndSet";
   case SpvOpAtomicFlagClear: return "SpvOpAtomicFlagClear";
   case SpvOpImageSparseRead: return "SpvOpImageSparseRead";
   case SpvOpSizeOf: return "SpvOpSizeOf";
   case SpvOpTypePipeStorage: return "SpvOpTypePipeStorage";
   case SpvOpConstantPipeStorage: return "SpvOpConstantPipeStorage";
   case SpvOpCreatePipeFromPipeStorage: return "SpvOpCreatePipeFromPipeStorage";
   case SpvOpGetKernelLocalSizeForSubgroupCount: return "SpvOpGetKernelLocalSizeForSubgroupCount";
   case SpvOpGetKernelMaxNumSubgroups: return "SpvOpGetKernelMaxNumSubgroups";
   case SpvOpTypeNamedBarrier: return "SpvOpTypeNamedBarrier";
   case SpvOpNamedBarrierInitialize: return "SpvOpNamedBarrierInitialize";
   case SpvOpMemoryNamedBarrier: return "SpvOpMemoryNamedBarrier";
   case SpvOpModuleProcessed: return "SpvOpModuleProcessed";
   case SpvOpExecutionModeId: return "SpvOpExecutionModeId";
   case SpvOpDecorateId: return "SpvOpDecorateId";
   case SpvOpGroupNonUniformElect: return "SpvOpGroupNonUniformElect";
   case SpvOpGroupNonUniformAll: return "SpvOpGroupNonUniformAll";
   case SpvOpGroupNonUniformAny: return "SpvOpGroupNonUniformAny";
   case SpvOpGroupNonUniformAllEqual: return "SpvOpGroupNonUniformAllEqual";
   case SpvOpGroupNonUniformBroadcast: return "SpvOpGroupNonUniformBroadcast";
   case SpvOpGroupNonUniformBroadcastFirst: return "SpvOpGroupNonUniformBroadcastFirst";
   case SpvOpGroupNonUniformBallot: return "SpvOpGroupNonUniformBallot";
   case SpvOpGroupNonUniformInverseBallot: return "SpvOpGroupNonUniformInverseBallot";
   case SpvOpGroupNonUniformBallotBitExtract: return "SpvOpGroupNonUniformBallotBitExtract";
   case SpvOpGroupNonUniformBallotBitCount: return "SpvOpGroupNonUniformBallotBitCount";
   case SpvOpGroupNonUniformBallotFindLSB: return "SpvOpGroupNonUniformBallotFindLSB";
   case SpvOpGroupNonUniformBallotFindMSB: return "SpvOpGroupNonUniformBallotFindMSB";
   case SpvOpGroupNonUniformShuffle: return "SpvOpGroupNonUniformShuffle";
   case SpvOpGroupNonUniformShuffleXor: return "SpvOpGroupNonUniformShuffleXor";
   case SpvOpGroupNonUniformShuffleUp: return "SpvOpGroupNonUniformShuffleUp";
   case SpvOpGroupNonUniformShuffleDown: return "SpvOpGroupNonUniformShuffleDown";
   case SpvOpGroupNonUniformIAdd: return "SpvOpGroupNonUniformIAdd";
   case SpvOpGroupNonUniformFAdd: return "SpvOpGroupNonUniformFAdd";
   case SpvOpGroupNonUniformIMul: return "SpvOpGroupNonUniformIMul";
   case SpvOpGroupNonUniformFMul: return "SpvOpGroupNonUniformFMul";
   case SpvOpGroupNonUniformSMin: return "SpvOpGroupNonUniformSMin";
   case SpvOpGroupNonUniformUMin: return "SpvOpGroupNonUniformUMin";
   case SpvOpGroupNonUniformFMin: return "SpvOpGroupNonUniformFMin";
   case SpvOpGroupNonUniformSMax: return "SpvOpGroupNonUniformSMax";
   case SpvOpGroupNonUniformUMax: return "SpvOpGroupNonUniformUMax";
   case SpvOpGroupNonUniformFMax: return "SpvOpGroupNonUniformFMax";
   case SpvOpGroupNonUniformBitwiseAnd: return "SpvOpGroupNonUniformBitwiseAnd";
   case SpvOpGroupNonUniformBitwiseOr: return "SpvOpGroupNonUniformBitwiseOr";
   case SpvOpGroupNonUniformBitwiseXor: return "SpvOpGroupNonUniformBitwiseXor";
   case SpvOpGroupNonUniformLogicalAnd: return "SpvOpGroupNonUniformLogicalAnd";
   case SpvOpGroupNonUniformLogicalOr: return "SpvOpGroupNonUniformLogicalOr";
   case SpvOpGroupNonUniformLogicalXor: return "SpvOpGroupNonUniformLogicalXor";
   case SpvOpGroupNonUniformQuadBroadcast: return "SpvOpGroupNonUniformQuadBroadcast";
   case SpvOpGroupNonUniformQuadSwap: return "SpvOpGroupNonUniformQuadSwap";
   case SpvOpCopyLogical: return "SpvOpCopyLogical";
   case SpvOpPtrEqual: return "SpvOpPtrEqual";
   case SpvOpPtrNotEqual: return "SpvOpPtrNotEqual";
   case SpvOpPtrDiff: return "SpvOpPtrDiff";
   case SpvOpTerminateInvocation: return "SpvOpTerminateInvocation";
   case SpvOpSubgroupBallotKHR: return "SpvOpSubgroupBallotKHR";
   case SpvOpSubgroupFirstInvocationKHR: return "SpvOpSubgroupFirstInvocationKHR";
   case SpvOpSubgroupAllKHR: return "SpvOpSubgroupAllKHR";
   case SpvOpSubgroupAnyKHR: return "SpvOpSubgroupAnyKHR";
   case SpvOpSubgroupAllEqualKHR: return "SpvOpSubgroupAllEqualKHR";
   case SpvOpGroupNonUniformRotateKHR: return "SpvOpGroupNonUniformRotateKHR";
   case SpvOpSubgroupReadInvocationKHR: return "SpvOpSubgroupReadInvocationKHR";
   case SpvOpTraceRayKHR: return "SpvOpTraceRayKHR";
   case SpvOpExecuteCallableKHR: return "SpvOpExecuteCallableKHR";
   case SpvOpConvertUToAccelerationStructureKHR: return "SpvOpConvertUToAccelerationStructureKHR";
   case SpvOpIgnoreIntersectionKHR: return "SpvOpIgnoreIntersectionKHR";
   case SpvOpTerminateRayKHR: return "SpvOpTerminateRayKHR";
   case SpvOpSDot: return "SpvOpSDot";
   case SpvOpUDot: return "SpvOpUDot";
   case SpvOpSUDot: return "SpvOpSUDot";
   case SpvOpSDotAccSat: return "SpvOpSDotAccSat";
   case SpvOpUDotAccSat: return "SpvOpUDotAccSat";
   case SpvOpSUDotAccSat: return "SpvOpSUDotAccSat";
   case SpvOpTypeRayQueryKHR: return "SpvOpTypeRayQueryKHR";
   case SpvOpRayQueryInitializeKHR: return "SpvOpRayQueryInitializeKHR";
   case SpvOpRayQueryTerminateKHR: return "SpvOpRayQueryTerminateKHR";
   case SpvOpRayQueryGenerateIntersectionKHR: return "SpvOpRayQueryGenerateIntersectionKHR";
   case SpvOpRayQueryConfirmIntersectionKHR: return "SpvOpRayQueryConfirmIntersectionKHR";
   case SpvOpRayQueryProceedKHR: return "SpvOpRayQueryProceedKHR";
   case SpvOpRayQueryGetIntersectionTypeKHR: return "SpvOpRayQueryGetIntersectionTypeKHR";
   case SpvOpGroupIAddNonUniformAMD: return "SpvOpGroupIAddNonUniformAMD";
   case SpvOpGroupFAddNonUniformAMD: return "SpvOpGroupFAddNonUniformAMD";
   case SpvOpGroupFMinNonUniformAMD: return "SpvOpGroupFMinNonUniformAMD";
   case SpvOpGroupUMinNonUniformAMD: return "SpvOpGroupUMinNonUniformAMD";
   case SpvOpGroupSMinNonUniformAMD: return "SpvOpGroupSMinNonUniformAMD";
   case SpvOpGroupFMaxNonUniformAMD: return "SpvOpGroupFMaxNonUniformAMD";
   case SpvOpGroupUMaxNonUniformAMD: return "SpvOpGroupUMaxNonUniformAMD";
   case SpvOpGroupSMaxNonUniformAMD: return "SpvOpGroupSMaxNonUniformAMD";
   case SpvOpFragmentMaskFetchAMD: return "SpvOpFragmentMaskFetchAMD";
   case SpvOpFragmentFetchAMD: return "SpvOpFragmentFetchAMD";
   case SpvOpReadClockKHR: return "SpvOpReadClockKHR";
   case SpvOpImageSampleFootprintNV: return "SpvOpImageSampleFootprintNV";
   case SpvOpEmitMeshTasksEXT: return "SpvOpEmitMeshTasksEXT";
   case SpvOpSetMeshOutputsEXT: return "SpvOpSetMeshOutputsEXT";
   case SpvOpGroupNonUniformPartitionNV: return "SpvOpGroupNonUniformPartitionNV";
   case SpvOpWritePackedPrimitiveIndices4x8NV: return "SpvOpWritePackedPrimitiveIndices4x8NV";
   case SpvOpReportIntersectionNV: return "SpvOpReportIntersectionNV";
   case SpvOpIgnoreIntersectionNV: return "SpvOpIgnoreIntersectionNV";
   case SpvOpTerminateRayNV: return "SpvOpTerminateRayNV";
   case SpvOpTraceNV: return "SpvOpTraceNV";
   case SpvOpTraceMotionNV: return "SpvOpTraceMotionNV";
   case SpvOpTraceRayMotionNV: return "SpvOpTraceRayMotionNV";
   case SpvOpTypeAccelerationStructureNV: return "SpvOpTypeAccelerationStructureNV";
   case SpvOpExecuteCallableNV: return "SpvOpExecuteCallableNV";
   case SpvOpTypeCooperativeMatrixNV: return "SpvOpTypeCooperativeMatrixNV";
   case SpvOpCooperativeMatrixLoadNV: return "SpvOpCooperativeMatrixLoadNV";
   case SpvOpCooperativeMatrixStoreNV: return "SpvOpCooperativeMatrixStoreNV";
   case SpvOpCooperativeMatrixMulAddNV: return "SpvOpCooperativeMatrixMulAddNV";
   case SpvOpCooperativeMatrixLengthNV: return "SpvOpCooperativeMatrixLengthNV";
   case SpvOpBeginInvocationInterlockEXT: return "SpvOpBeginInvocationInterlockEXT";
   case SpvOpEndInvocationInterlockEXT: return "SpvOpEndInvocationInterlockEXT";
   case SpvOpDemoteToHelperInvocation: return "SpvOpDemoteToHelperInvocation";
   case SpvOpIsHelperInvocationEXT: return "SpvOpIsHelperInvocationEXT";
   case SpvOpConvertUToImageNV: return "SpvOpConvertUToImageNV";
   case SpvOpConvertUToSamplerNV: return "SpvOpConvertUToSamplerNV";
   case SpvOpConvertImageToUNV: return "SpvOpConvertImageToUNV";
   case SpvOpConvertSamplerToUNV: return "SpvOpConvertSamplerToUNV";
   case SpvOpConvertUToSampledImageNV: return "SpvOpConvertUToSampledImageNV";
   case SpvOpConvertSampledImageToUNV: return "SpvOpConvertSampledImageToUNV";
   case SpvOpSamplerImageAddressingModeNV: return "SpvOpSamplerImageAddressingModeNV";
   case SpvOpSubgroupShuffleINTEL: return "SpvOpSubgroupShuffleINTEL";
   case SpvOpSubgroupShuffleDownINTEL: return "SpvOpSubgroupShuffleDownINTEL";
   case SpvOpSubgroupShuffleUpINTEL: return "SpvOpSubgroupShuffleUpINTEL";
   case SpvOpSubgroupShuffleXorINTEL: return "SpvOpSubgroupShuffleXorINTEL";
   case SpvOpSubgroupBlockReadINTEL: return "SpvOpSubgroupBlockReadINTEL";
   case SpvOpSubgroupBlockWriteINTEL: return "SpvOpSubgroupBlockWriteINTEL";
   case SpvOpSubgroupImageBlockReadINTEL: return "SpvOpSubgroupImageBlockReadINTEL";
   case SpvOpSubgroupImageBlockWriteINTEL: return "SpvOpSubgroupImageBlockWriteINTEL";
   case SpvOpSubgroupImageMediaBlockReadINTEL: return "SpvOpSubgroupImageMediaBlockReadINTEL";
   case SpvOpSubgroupImageMediaBlockWriteINTEL: return "SpvOpSubgroupImageMediaBlockWriteINTEL";
   case SpvOpUCountLeadingZerosINTEL: return "SpvOpUCountLeadingZerosINTEL";
   case SpvOpUCountTrailingZerosINTEL: return "SpvOpUCountTrailingZerosINTEL";
   case SpvOpAbsISubINTEL: return "SpvOpAbsISubINTEL";
   case SpvOpAbsUSubINTEL: return "SpvOpAbsUSubINTEL";
   case SpvOpIAddSatINTEL: return "SpvOpIAddSatINTEL";
   case SpvOpUAddSatINTEL: return "SpvOpUAddSatINTEL";
   case SpvOpIAverageINTEL: return "SpvOpIAverageINTEL";
   case SpvOpUAverageINTEL: return "SpvOpUAverageINTEL";
   case SpvOpIAverageRoundedINTEL: return "SpvOpIAverageRoundedINTEL";
   case SpvOpUAverageRoundedINTEL: return "SpvOpUAverageRoundedINTEL";
   case SpvOpISubSatINTEL: return "SpvOpISubSatINTEL";
   case SpvOpUSubSatINTEL: return "SpvOpUSubSatINTEL";
   case SpvOpIMul32x16INTEL: return "SpvOpIMul32x16INTEL";
   case SpvOpUMul32x16INTEL: return "SpvOpUMul32x16INTEL";
   case SpvOpConstantFunctionPointerINTEL: return "SpvOpConstantFunctionPointerINTEL";
   case SpvOpFunctionPointerCallINTEL: return "SpvOpFunctionPointerCallINTEL";
   case SpvOpAsmTargetINTEL: return "SpvOpAsmTargetINTEL";
   case SpvOpAsmINTEL: return "SpvOpAsmINTEL";
   case SpvOpAsmCallINTEL: return "SpvOpAsmCallINTEL";
   case SpvOpAtomicFMinEXT: return "SpvOpAtomicFMinEXT";
   case SpvOpAtomicFMaxEXT: return "SpvOpAtomicFMaxEXT";
   case SpvOpAssumeTrueKHR: return "SpvOpAssumeTrueKHR";
   case SpvOpExpectKHR: return "SpvOpExpectKHR";
   case SpvOpDecorateString: return "SpvOpDecorateString";
   case SpvOpMemberDecorateString: return "SpvOpMemberDecorateString";
   case SpvOpVmeImageINTEL: return "SpvOpVmeImageINTEL";
   case SpvOpTypeVmeImageINTEL: return "SpvOpTypeVmeImageINTEL";
   case SpvOpTypeAvcImePayloadINTEL: return "SpvOpTypeAvcImePayloadINTEL";
   case SpvOpTypeAvcRefPayloadINTEL: return "SpvOpTypeAvcRefPayloadINTEL";
   case SpvOpTypeAvcSicPayloadINTEL: return "SpvOpTypeAvcSicPayloadINTEL";
   case SpvOpTypeAvcMcePayloadINTEL: return "SpvOpTypeAvcMcePayloadINTEL";
   case SpvOpTypeAvcMceResultINTEL: return "SpvOpTypeAvcMceResultINTEL";
   case SpvOpTypeAvcImeResultINTEL: return "SpvOpTypeAvcImeResultINTEL";
   case SpvOpTypeAvcImeResultSingleReferenceStreamoutINTEL: return "SpvOpTypeAvcImeResultSingleReferenceStreamoutINTEL";
   case SpvOpTypeAvcImeResultDualReferenceStreamoutINTEL: return "SpvOpTypeAvcImeResultDualReferenceStreamoutINTEL";
   case SpvOpTypeAvcImeSingleReferenceStreaminINTEL: return "SpvOpTypeAvcImeSingleReferenceStreaminINTEL";
   case SpvOpTypeAvcImeDualReferenceStreaminINTEL: return "SpvOpTypeAvcImeDualReferenceStreaminINTEL";
   case SpvOpTypeAvcRefResultINTEL: return "SpvOpTypeAvcRefResultINTEL";
   case SpvOpTypeAvcSicResultINTEL: return "SpvOpTypeAvcSicResultINTEL";
   case SpvOpSubgroupAvcMceGetDefaultInterBaseMultiReferencePenaltyINTEL: return "SpvOpSubgroupAvcMceGetDefaultInterBaseMultiReferencePenaltyINTEL";
   case SpvOpSubgroupAvcMceSetInterBaseMultiReferencePenaltyINTEL: return "SpvOpSubgroupAvcMceSetInterBaseMultiReferencePenaltyINTEL";
   case SpvOpSubgroupAvcMceGetDefaultInterShapePenaltyINTEL: return "SpvOpSubgroupAvcMceGetDefaultInterShapePenaltyINTEL";
   case SpvOpSubgroupAvcMceSetInterShapePenaltyINTEL: return "SpvOpSubgroupAvcMceSetInterShapePenaltyINTEL";
   case SpvOpSubgroupAvcMceGetDefaultInterDirectionPenaltyINTEL: return "SpvOpSubgroupAvcMceGetDefaultInterDirectionPenaltyINTEL";
   case SpvOpSubgroupAvcMceSetInterDirectionPenaltyINTEL: return "SpvOpSubgroupAvcMceSetInterDirectionPenaltyINTEL";
   case SpvOpSubgroupAvcMceGetDefaultIntraLumaShapePenaltyINTEL: return "SpvOpSubgroupAvcMceGetDefaultIntraLumaShapePenaltyINTEL";
   case SpvOpSubgroupAvcMceGetDefaultInterMotionVectorCostTableINTEL: return "SpvOpSubgroupAvcMceGetDefaultInterMotionVectorCostTableINTEL";
   case SpvOpSubgroupAvcMceGetDefaultHighPenaltyCostTableINTEL: return "SpvOpSubgroupAvcMceGetDefaultHighPenaltyCostTableINTEL";
   case SpvOpSubgroupAvcMceGetDefaultMediumPenaltyCostTableINTEL: return "SpvOpSubgroupAvcMceGetDefaultMediumPenaltyCostTableINTEL";
   case SpvOpSubgroupAvcMceGetDefaultLowPenaltyCostTableINTEL: return "SpvOpSubgroupAvcMceGetDefaultLowPenaltyCostTableINTEL";
   case SpvOpSubgroupAvcMceSetMotionVectorCostFunctionINTEL: return "SpvOpSubgroupAvcMceSetMotionVectorCostFunctionINTEL";
   case SpvOpSubgroupAvcMceGetDefaultIntraLumaModePenaltyINTEL: return "SpvOpSubgroupAvcMceGetDefaultIntraLumaModePenaltyINTEL";
   case SpvOpSubgroupAvcMceGetDefaultNonDcLumaIntraPenaltyINTEL: return "SpvOpSubgroupAvcMceGetDefaultNonDcLumaIntraPenaltyINTEL";
   case SpvOpSubgroupAvcMceGetDefaultIntraChromaModeBasePenaltyINTEL: return "SpvOpSubgroupAvcMceGetDefaultIntraChromaModeBasePenaltyINTEL";
   case SpvOpSubgroupAvcMceSetAcOnlyHaarINTEL: return "SpvOpSubgroupAvcMceSetAcOnlyHaarINTEL";
   case SpvOpSubgroupAvcMceSetSourceInterlacedFieldPolarityINTEL: return "SpvOpSubgroupAvcMceSetSourceInterlacedFieldPolarityINTEL";
   case SpvOpSubgroupAvcMceSetSingleReferenceInterlacedFieldPolarityINTEL: return "SpvOpSubgroupAvcMceSetSingleReferenceInterlacedFieldPolarityINTEL";
   case SpvOpSubgroupAvcMceSetDualReferenceInterlacedFieldPolaritiesINTEL: return "SpvOpSubgroupAvcMceSetDualReferenceInterlacedFieldPolaritiesINTEL";
   case SpvOpSubgroupAvcMceConvertToImePayloadINTEL: return "SpvOpSubgroupAvcMceConvertToImePayloadINTEL";
   case SpvOpSubgroupAvcMceConvertToImeResultINTEL: return "SpvOpSubgroupAvcMceConvertToImeResultINTEL";
   case SpvOpSubgroupAvcMceConvertToRefPayloadINTEL: return "SpvOpSubgroupAvcMceConvertToRefPayloadINTEL";
   case SpvOpSubgroupAvcMceConvertToRefResultINTEL: return "SpvOpSubgroupAvcMceConvertToRefResultINTEL";
   case SpvOpSubgroupAvcMceConvertToSicPayloadINTEL: return "SpvOpSubgroupAvcMceConvertToSicPayloadINTEL";
   case SpvOpSubgroupAvcMceConvertToSicResultINTEL: return "SpvOpSubgroupAvcMceConvertToSicResultINTEL";
   case SpvOpSubgroupAvcMceGetMotionVectorsINTEL: return "SpvOpSubgroupAvcMceGetMotionVectorsINTEL";
   case SpvOpSubgroupAvcMceGetInterDistortionsINTEL: return "SpvOpSubgroupAvcMceGetInterDistortionsINTEL";
   case SpvOpSubgroupAvcMceGetBestInterDistortionsINTEL: return "SpvOpSubgroupAvcMceGetBestInterDistortionsINTEL";
   case SpvOpSubgroupAvcMceGetInterMajorShapeINTEL: return "SpvOpSubgroupAvcMceGetInterMajorShapeINTEL";
   case SpvOpSubgroupAvcMceGetInterMinorShapeINTEL: return "SpvOpSubgroupAvcMceGetInterMinorShapeINTEL";
   case SpvOpSubgroupAvcMceGetInterDirectionsINTEL: return "SpvOpSubgroupAvcMceGetInterDirectionsINTEL";
   case SpvOpSubgroupAvcMceGetInterMotionVectorCountINTEL: return "SpvOpSubgroupAvcMceGetInterMotionVectorCountINTEL";
   case SpvOpSubgroupAvcMceGetInterReferenceIdsINTEL: return "SpvOpSubgroupAvcMceGetInterReferenceIdsINTEL";
   case SpvOpSubgroupAvcMceGetInterReferenceInterlacedFieldPolaritiesINTEL: return "SpvOpSubgroupAvcMceGetInterReferenceInterlacedFieldPolaritiesINTEL";
   case SpvOpSubgroupAvcImeInitializeINTEL: return "SpvOpSubgroupAvcImeInitializeINTEL";
   case SpvOpSubgroupAvcImeSetSingleReferenceINTEL: return "SpvOpSubgroupAvcImeSetSingleReferenceINTEL";
   case SpvOpSubgroupAvcImeSetDualReferenceINTEL: return "SpvOpSubgroupAvcImeSetDualReferenceINTEL";
   case SpvOpSubgroupAvcImeRefWindowSizeINTEL: return "SpvOpSubgroupAvcImeRefWindowSizeINTEL";
   case SpvOpSubgroupAvcImeAdjustRefOffsetINTEL: return "SpvOpSubgroupAvcImeAdjustRefOffsetINTEL";
   case SpvOpSubgroupAvcImeConvertToMcePayloadINTEL: return "SpvOpSubgroupAvcImeConvertToMcePayloadINTEL";
   case SpvOpSubgroupAvcImeSetMaxMotionVectorCountINTEL: return "SpvOpSubgroupAvcImeSetMaxMotionVectorCountINTEL";
   case SpvOpSubgroupAvcImeSetUnidirectionalMixDisableINTEL: return "SpvOpSubgroupAvcImeSetUnidirectionalMixDisableINTEL";
   case SpvOpSubgroupAvcImeSetEarlySearchTerminationThresholdINTEL: return "SpvOpSubgroupAvcImeSetEarlySearchTerminationThresholdINTEL";
   case SpvOpSubgroupAvcImeSetWeightedSadINTEL: return "SpvOpSubgroupAvcImeSetWeightedSadINTEL";
   case SpvOpSubgroupAvcImeEvaluateWithSingleReferenceINTEL: return "SpvOpSubgroupAvcImeEvaluateWithSingleReferenceINTEL";
   case SpvOpSubgroupAvcImeEvaluateWithDualReferenceINTEL: return "SpvOpSubgroupAvcImeEvaluateWithDualReferenceINTEL";
   case SpvOpSubgroupAvcImeEvaluateWithSingleReferenceStreaminINTEL: return "SpvOpSubgroupAvcImeEvaluateWithSingleReferenceStreaminINTEL";
   case SpvOpSubgroupAvcImeEvaluateWithDualReferenceStreaminINTEL: return "SpvOpSubgroupAvcImeEvaluateWithDualReferenceStreaminINTEL";
   case SpvOpSubgroupAvcImeEvaluateWithSingleReferenceStreamoutINTEL: return "SpvOpSubgroupAvcImeEvaluateWithSingleReferenceStreamoutINTEL";
   case SpvOpSubgroupAvcImeEvaluateWithDualReferenceStreamoutINTEL: return "SpvOpSubgroupAvcImeEvaluateWithDualReferenceStreamoutINTEL";
   case SpvOpSubgroupAvcImeEvaluateWithSingleReferenceStreaminoutINTEL: return "SpvOpSubgroupAvcImeEvaluateWithSingleReferenceStreaminoutINTEL";
   case SpvOpSubgroupAvcImeEvaluateWithDualReferenceStreaminoutINTEL: return "SpvOpSubgroupAvcImeEvaluateWithDualReferenceStreaminoutINTEL";
   case SpvOpSubgroupAvcImeConvertToMceResultINTEL: return "SpvOpSubgroupAvcImeConvertToMceResultINTEL";
   case SpvOpSubgroupAvcImeGetSingleReferenceStreaminINTEL: return "SpvOpSubgroupAvcImeGetSingleReferenceStreaminINTEL";
   case SpvOpSubgroupAvcImeGetDualReferenceStreaminINTEL: return "SpvOpSubgroupAvcImeGetDualReferenceStreaminINTEL";
   case SpvOpSubgroupAvcImeStripSingleReferenceStreamoutINTEL: return "SpvOpSubgroupAvcImeStripSingleReferenceStreamoutINTEL";
   case SpvOpSubgroupAvcImeStripDualReferenceStreamoutINTEL: return "SpvOpSubgroupAvcImeStripDualReferenceStreamoutINTEL";
   case SpvOpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeMotionVectorsINTEL: return "SpvOpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeMotionVectorsINTEL";
   case SpvOpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeDistortionsINTEL: return "SpvOpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeDistortionsINTEL";
   case SpvOpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeReferenceIdsINTEL: return "SpvOpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeReferenceIdsINTEL";
   case SpvOpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeMotionVectorsINTEL: return "SpvOpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeMotionVectorsINTEL";
   case SpvOpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeDistortionsINTEL: return "SpvOpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeDistortionsINTEL";
   case SpvOpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeReferenceIdsINTEL: return "SpvOpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeReferenceIdsINTEL";
   case SpvOpSubgroupAvcImeGetBorderReachedINTEL: return "SpvOpSubgroupAvcImeGetBorderReachedINTEL";
   case SpvOpSubgroupAvcImeGetTruncatedSearchIndicationINTEL: return "SpvOpSubgroupAvcImeGetTruncatedSearchIndicationINTEL";
   case SpvOpSubgroupAvcImeGetUnidirectionalEarlySearchTerminationINTEL: return "SpvOpSubgroupAvcImeGetUnidirectionalEarlySearchTerminationINTEL";
   case SpvOpSubgroupAvcImeGetWeightingPatternMinimumMotionVectorINTEL: return "SpvOpSubgroupAvcImeGetWeightingPatternMinimumMotionVectorINTEL";
   case SpvOpSubgroupAvcImeGetWeightingPatternMinimumDistortionINTEL: return "SpvOpSubgroupAvcImeGetWeightingPatternMinimumDistortionINTEL";
   case SpvOpSubgroupAvcFmeInitializeINTEL: return "SpvOpSubgroupAvcFmeInitializeINTEL";
   case SpvOpSubgroupAvcBmeInitializeINTEL: return "SpvOpSubgroupAvcBmeInitializeINTEL";
   case SpvOpSubgroupAvcRefConvertToMcePayloadINTEL: return "SpvOpSubgroupAvcRefConvertToMcePayloadINTEL";
   case SpvOpSubgroupAvcRefSetBidirectionalMixDisableINTEL: return "SpvOpSubgroupAvcRefSetBidirectionalMixDisableINTEL";
   case SpvOpSubgroupAvcRefSetBilinearFilterEnableINTEL: return "SpvOpSubgroupAvcRefSetBilinearFilterEnableINTEL";
   case SpvOpSubgroupAvcRefEvaluateWithSingleReferenceINTEL: return "SpvOpSubgroupAvcRefEvaluateWithSingleReferenceINTEL";
   case SpvOpSubgroupAvcRefEvaluateWithDualReferenceINTEL: return "SpvOpSubgroupAvcRefEvaluateWithDualReferenceINTEL";
   case SpvOpSubgroupAvcRefEvaluateWithMultiReferenceINTEL: return "SpvOpSubgroupAvcRefEvaluateWithMultiReferenceINTEL";
   case SpvOpSubgroupAvcRefEvaluateWithMultiReferenceInterlacedINTEL: return "SpvOpSubgroupAvcRefEvaluateWithMultiReferenceInterlacedINTEL";
   case SpvOpSubgroupAvcRefConvertToMceResultINTEL: return "SpvOpSubgroupAvcRefConvertToMceResultINTEL";
   case SpvOpSubgroupAvcSicInitializeINTEL: return "SpvOpSubgroupAvcSicInitializeINTEL";
   case SpvOpSubgroupAvcSicConfigureSkcINTEL: return "SpvOpSubgroupAvcSicConfigureSkcINTEL";
   case SpvOpSubgroupAvcSicConfigureIpeLumaINTEL: return "SpvOpSubgroupAvcSicConfigureIpeLumaINTEL";
   case SpvOpSubgroupAvcSicConfigureIpeLumaChromaINTEL: return "SpvOpSubgroupAvcSicConfigureIpeLumaChromaINTEL";
   case SpvOpSubgroupAvcSicGetMotionVectorMaskINTEL: return "SpvOpSubgroupAvcSicGetMotionVectorMaskINTEL";
   case SpvOpSubgroupAvcSicConvertToMcePayloadINTEL: return "SpvOpSubgroupAvcSicConvertToMcePayloadINTEL";
   case SpvOpSubgroupAvcSicSetIntraLumaShapePenaltyINTEL: return "SpvOpSubgroupAvcSicSetIntraLumaShapePenaltyINTEL";
   case SpvOpSubgroupAvcSicSetIntraLumaModeCostFunctionINTEL: return "SpvOpSubgroupAvcSicSetIntraLumaModeCostFunctionINTEL";
   case SpvOpSubgroupAvcSicSetIntraChromaModeCostFunctionINTEL: return "SpvOpSubgroupAvcSicSetIntraChromaModeCostFunctionINTEL";
   case SpvOpSubgroupAvcSicSetBilinearFilterEnableINTEL: return "SpvOpSubgroupAvcSicSetBilinearFilterEnableINTEL";
   case SpvOpSubgroupAvcSicSetSkcForwardTransformEnableINTEL: return "SpvOpSubgroupAvcSicSetSkcForwardTransformEnableINTEL";
   case SpvOpSubgroupAvcSicSetBlockBasedRawSkipSadINTEL: return "SpvOpSubgroupAvcSicSetBlockBasedRawSkipSadINTEL";
   case SpvOpSubgroupAvcSicEvaluateIpeINTEL: return "SpvOpSubgroupAvcSicEvaluateIpeINTEL";
   case SpvOpSubgroupAvcSicEvaluateWithSingleReferenceINTEL: return "SpvOpSubgroupAvcSicEvaluateWithSingleReferenceINTEL";
   case SpvOpSubgroupAvcSicEvaluateWithDualReferenceINTEL: return "SpvOpSubgroupAvcSicEvaluateWithDualReferenceINTEL";
   case SpvOpSubgroupAvcSicEvaluateWithMultiReferenceINTEL: return "SpvOpSubgroupAvcSicEvaluateWithMultiReferenceINTEL";
   case SpvOpSubgroupAvcSicEvaluateWithMultiReferenceInterlacedINTEL: return "SpvOpSubgroupAvcSicEvaluateWithMultiReferenceInterlacedINTEL";
   case SpvOpSubgroupAvcSicConvertToMceResultINTEL: return "SpvOpSubgroupAvcSicConvertToMceResultINTEL";
   case SpvOpSubgroupAvcSicGetIpeLumaShapeINTEL: return "SpvOpSubgroupAvcSicGetIpeLumaShapeINTEL";
   case SpvOpSubgroupAvcSicGetBestIpeLumaDistortionINTEL: return "SpvOpSubgroupAvcSicGetBestIpeLumaDistortionINTEL";
   case SpvOpSubgroupAvcSicGetBestIpeChromaDistortionINTEL: return "SpvOpSubgroupAvcSicGetBestIpeChromaDistortionINTEL";
   case SpvOpSubgroupAvcSicGetPackedIpeLumaModesINTEL: return "SpvOpSubgroupAvcSicGetPackedIpeLumaModesINTEL";
   case SpvOpSubgroupAvcSicGetIpeChromaModeINTEL: return "SpvOpSubgroupAvcSicGetIpeChromaModeINTEL";
   case SpvOpSubgroupAvcSicGetPackedSkcLumaCountThresholdINTEL: return "SpvOpSubgroupAvcSicGetPackedSkcLumaCountThresholdINTEL";
   case SpvOpSubgroupAvcSicGetPackedSkcLumaSumThresholdINTEL: return "SpvOpSubgroupAvcSicGetPackedSkcLumaSumThresholdINTEL";
   case SpvOpSubgroupAvcSicGetInterRawSadsINTEL: return "SpvOpSubgroupAvcSicGetInterRawSadsINTEL";
   case SpvOpVariableLengthArrayINTEL: return "SpvOpVariableLengthArrayINTEL";
   case SpvOpSaveMemoryINTEL: return "SpvOpSaveMemoryINTEL";
   case SpvOpRestoreMemoryINTEL: return "SpvOpRestoreMemoryINTEL";
   case SpvOpArbitraryFloatSinCosPiINTEL: return "SpvOpArbitraryFloatSinCosPiINTEL";
   case SpvOpArbitraryFloatCastINTEL: return "SpvOpArbitraryFloatCastINTEL";
   case SpvOpArbitraryFloatCastFromIntINTEL: return "SpvOpArbitraryFloatCastFromIntINTEL";
   case SpvOpArbitraryFloatCastToIntINTEL: return "SpvOpArbitraryFloatCastToIntINTEL";
   case SpvOpArbitraryFloatAddINTEL: return "SpvOpArbitraryFloatAddINTEL";
   case SpvOpArbitraryFloatSubINTEL: return "SpvOpArbitraryFloatSubINTEL";
   case SpvOpArbitraryFloatMulINTEL: return "SpvOpArbitraryFloatMulINTEL";
   case SpvOpArbitraryFloatDivINTEL: return "SpvOpArbitraryFloatDivINTEL";
   case SpvOpArbitraryFloatGTINTEL: return "SpvOpArbitraryFloatGTINTEL";
   case SpvOpArbitraryFloatGEINTEL: return "SpvOpArbitraryFloatGEINTEL";
   case SpvOpArbitraryFloatLTINTEL: return "SpvOpArbitraryFloatLTINTEL";
   case SpvOpArbitraryFloatLEINTEL: return "SpvOpArbitraryFloatLEINTEL";
   case SpvOpArbitraryFloatEQINTEL: return "SpvOpArbitraryFloatEQINTEL";
   case SpvOpArbitraryFloatRecipINTEL: return "SpvOpArbitraryFloatRecipINTEL";
   case SpvOpArbitraryFloatRSqrtINTEL: return "SpvOpArbitraryFloatRSqrtINTEL";
   case SpvOpArbitraryFloatCbrtINTEL: return "SpvOpArbitraryFloatCbrtINTEL";
   case SpvOpArbitraryFloatHypotINTEL: return "SpvOpArbitraryFloatHypotINTEL";
   case SpvOpArbitraryFloatSqrtINTEL: return "SpvOpArbitraryFloatSqrtINTEL";
   case SpvOpArbitraryFloatLogINTEL: return "SpvOpArbitraryFloatLogINTEL";
   case SpvOpArbitraryFloatLog2INTEL: return "SpvOpArbitraryFloatLog2INTEL";
   case SpvOpArbitraryFloatLog10INTEL: return "SpvOpArbitraryFloatLog10INTEL";
   case SpvOpArbitraryFloatLog1pINTEL: return "SpvOpArbitraryFloatLog1pINTEL";
   case SpvOpArbitraryFloatExpINTEL: return "SpvOpArbitraryFloatExpINTEL";
   case SpvOpArbitraryFloatExp2INTEL: return "SpvOpArbitraryFloatExp2INTEL";
   case SpvOpArbitraryFloatExp10INTEL: return "SpvOpArbitraryFloatExp10INTEL";
   case SpvOpArbitraryFloatExpm1INTEL: return "SpvOpArbitraryFloatExpm1INTEL";
   case SpvOpArbitraryFloatSinINTEL: return "SpvOpArbitraryFloatSinINTEL";
   case SpvOpArbitraryFloatCosINTEL: return "SpvOpArbitraryFloatCosINTEL";
   case SpvOpArbitraryFloatSinCosINTEL: return "SpvOpArbitraryFloatSinCosINTEL";
   case SpvOpArbitraryFloatSinPiINTEL: return "SpvOpArbitraryFloatSinPiINTEL";
   case SpvOpArbitraryFloatCosPiINTEL: return "SpvOpArbitraryFloatCosPiINTEL";
   case SpvOpArbitraryFloatASinINTEL: return "SpvOpArbitraryFloatASinINTEL";
   case SpvOpArbitraryFloatASinPiINTEL: return "SpvOpArbitraryFloatASinPiINTEL";
   case SpvOpArbitraryFloatACosINTEL: return "SpvOpArbitraryFloatACosINTEL";
   case SpvOpArbitraryFloatACosPiINTEL: return "SpvOpArbitraryFloatACosPiINTEL";
   case SpvOpArbitraryFloatATanINTEL: return "SpvOpArbitraryFloatATanINTEL";
   case SpvOpArbitraryFloatATanPiINTEL: return "SpvOpArbitraryFloatATanPiINTEL";
   case SpvOpArbitraryFloatATan2INTEL: return "SpvOpArbitraryFloatATan2INTEL";
   case SpvOpArbitraryFloatPowINTEL: return "SpvOpArbitraryFloatPowINTEL";
   case SpvOpArbitraryFloatPowRINTEL: return "SpvOpArbitraryFloatPowRINTEL";
   case SpvOpArbitraryFloatPowNINTEL: return "SpvOpArbitraryFloatPowNINTEL";
   case SpvOpLoopControlINTEL: return "SpvOpLoopControlINTEL";
   case SpvOpAliasDomainDeclINTEL: return "SpvOpAliasDomainDeclINTEL";
   case SpvOpAliasScopeDeclINTEL: return "SpvOpAliasScopeDeclINTEL";
   case SpvOpAliasScopeListDeclINTEL: return "SpvOpAliasScopeListDeclINTEL";
   case SpvOpFixedSqrtINTEL: return "SpvOpFixedSqrtINTEL";
   case SpvOpFixedRecipINTEL: return "SpvOpFixedRecipINTEL";
   case SpvOpFixedRsqrtINTEL: return "SpvOpFixedRsqrtINTEL";
   case SpvOpFixedSinINTEL: return "SpvOpFixedSinINTEL";
   case SpvOpFixedCosINTEL: return "SpvOpFixedCosINTEL";
   case SpvOpFixedSinCosINTEL: return "SpvOpFixedSinCosINTEL";
   case SpvOpFixedSinPiINTEL: return "SpvOpFixedSinPiINTEL";
   case SpvOpFixedCosPiINTEL: return "SpvOpFixedCosPiINTEL";
   case SpvOpFixedSinCosPiINTEL: return "SpvOpFixedSinCosPiINTEL";
   case SpvOpFixedLogINTEL: return "SpvOpFixedLogINTEL";
   case SpvOpFixedExpINTEL: return "SpvOpFixedExpINTEL";
   case SpvOpPtrCastToCrossWorkgroupINTEL: return "SpvOpPtrCastToCrossWorkgroupINTEL";
   case SpvOpCrossWorkgroupCastToPtrINTEL: return "SpvOpCrossWorkgroupCastToPtrINTEL";
   case SpvOpReadPipeBlockingINTEL: return "SpvOpReadPipeBlockingINTEL";
   case SpvOpWritePipeBlockingINTEL: return "SpvOpWritePipeBlockingINTEL";
   case SpvOpFPGARegINTEL: return "SpvOpFPGARegINTEL";
   case SpvOpRayQueryGetRayTMinKHR: return "SpvOpRayQueryGetRayTMinKHR";
   case SpvOpRayQueryGetRayFlagsKHR: return "SpvOpRayQueryGetRayFlagsKHR";
   case SpvOpRayQueryGetIntersectionTKHR: return "SpvOpRayQueryGetIntersectionTKHR";
   case SpvOpRayQueryGetIntersectionInstanceCustomIndexKHR: return "SpvOpRayQueryGetIntersectionInstanceCustomIndexKHR";
   case SpvOpRayQueryGetIntersectionInstanceIdKHR: return "SpvOpRayQueryGetIntersectionInstanceIdKHR";
   case SpvOpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR: return "SpvOpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR";
   case SpvOpRayQueryGetIntersectionGeometryIndexKHR: return "SpvOpRayQueryGetIntersectionGeometryIndexKHR";
   case SpvOpRayQueryGetIntersectionPrimitiveIndexKHR: return "SpvOpRayQueryGetIntersectionPrimitiveIndexKHR";
   case SpvOpRayQueryGetIntersectionBarycentricsKHR: return "SpvOpRayQueryGetIntersectionBarycentricsKHR";
   case SpvOpRayQueryGetIntersectionFrontFaceKHR: return "SpvOpRayQueryGetIntersectionFrontFaceKHR";
   case SpvOpRayQueryGetIntersectionCandidateAABBOpaqueKHR: return "SpvOpRayQueryGetIntersectionCandidateAABBOpaqueKHR";
   case SpvOpRayQueryGetIntersectionObjectRayDirectionKHR: return "SpvOpRayQueryGetIntersectionObjectRayDirectionKHR";
   case SpvOpRayQueryGetIntersectionObjectRayOriginKHR: return "SpvOpRayQueryGetIntersectionObjectRayOriginKHR";
   case SpvOpRayQueryGetWorldRayDirectionKHR: return "SpvOpRayQueryGetWorldRayDirectionKHR";
   case SpvOpRayQueryGetWorldRayOriginKHR: return "SpvOpRayQueryGetWorldRayOriginKHR";
   case SpvOpRayQueryGetIntersectionObjectToWorldKHR: return "SpvOpRayQueryGetIntersectionObjectToWorldKHR";
   case SpvOpRayQueryGetIntersectionWorldToObjectKHR: return "SpvOpRayQueryGetIntersectionWorldToObjectKHR";
   case SpvOpAtomicFAddEXT: return "SpvOpAtomicFAddEXT";
   case SpvOpTypeBufferSurfaceINTEL: return "SpvOpTypeBufferSurfaceINTEL";
   case SpvOpTypeStructContinuedINTEL: return "SpvOpTypeStructContinuedINTEL";
   case SpvOpConstantCompositeContinuedINTEL: return "SpvOpConstantCompositeContinuedINTEL";
   case SpvOpSpecConstantCompositeContinuedINTEL: return "SpvOpSpecConstantCompositeContinuedINTEL";
   case SpvOpControlBarrierArriveINTEL: return "SpvOpControlBarrierArriveINTEL";
   case SpvOpControlBarrierWaitINTEL: return "SpvOpControlBarrierWaitINTEL";
   case SpvOpGroupIMulKHR: return "SpvOpGroupIMulKHR";
   case SpvOpGroupFMulKHR: return "SpvOpGroupFMulKHR";
   case SpvOpGroupBitwiseAndKHR: return "SpvOpGroupBitwiseAndKHR";
   case SpvOpGroupBitwiseOrKHR: return "SpvOpGroupBitwiseOrKHR";
   case SpvOpGroupBitwiseXorKHR: return "SpvOpGroupBitwiseXorKHR";
   case SpvOpGroupLogicalAndKHR: return "SpvOpGroupLogicalAndKHR";
   case SpvOpGroupLogicalOrKHR: return "SpvOpGroupLogicalOrKHR";
   case SpvOpGroupLogicalXorKHR: return "SpvOpGroupLogicalXorKHR";
   case SpvOpMax: break; /* silence warnings about unhandled enums. */
   }

   return "unknown";
}
