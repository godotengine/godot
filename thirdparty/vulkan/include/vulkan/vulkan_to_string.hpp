// Copyright 2015-2024 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//

// This header is generated from the Khronos Vulkan XML API Registry.

#ifndef VULKAN_TO_STRING_HPP
#define VULKAN_TO_STRING_HPP

#include <vulkan/vulkan_enums.hpp>

#if __cpp_lib_format
#  include <format>  // std::format
#else
#  include <sstream>  // std::stringstream
#endif

namespace VULKAN_HPP_NAMESPACE
{

  //==========================
  //=== BITMASKs to_string ===
  //==========================

  //=== VK_VERSION_1_0 ===

  VULKAN_HPP_INLINE std::string to_string( FormatFeatureFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & FormatFeatureFlagBits::eSampledImage )
      result += "SampledImage | ";
    if ( value & FormatFeatureFlagBits::eStorageImage )
      result += "StorageImage | ";
    if ( value & FormatFeatureFlagBits::eStorageImageAtomic )
      result += "StorageImageAtomic | ";
    if ( value & FormatFeatureFlagBits::eUniformTexelBuffer )
      result += "UniformTexelBuffer | ";
    if ( value & FormatFeatureFlagBits::eStorageTexelBuffer )
      result += "StorageTexelBuffer | ";
    if ( value & FormatFeatureFlagBits::eStorageTexelBufferAtomic )
      result += "StorageTexelBufferAtomic | ";
    if ( value & FormatFeatureFlagBits::eVertexBuffer )
      result += "VertexBuffer | ";
    if ( value & FormatFeatureFlagBits::eColorAttachment )
      result += "ColorAttachment | ";
    if ( value & FormatFeatureFlagBits::eColorAttachmentBlend )
      result += "ColorAttachmentBlend | ";
    if ( value & FormatFeatureFlagBits::eDepthStencilAttachment )
      result += "DepthStencilAttachment | ";
    if ( value & FormatFeatureFlagBits::eBlitSrc )
      result += "BlitSrc | ";
    if ( value & FormatFeatureFlagBits::eBlitDst )
      result += "BlitDst | ";
    if ( value & FormatFeatureFlagBits::eSampledImageFilterLinear )
      result += "SampledImageFilterLinear | ";
    if ( value & FormatFeatureFlagBits::eTransferSrc )
      result += "TransferSrc | ";
    if ( value & FormatFeatureFlagBits::eTransferDst )
      result += "TransferDst | ";
    if ( value & FormatFeatureFlagBits::eMidpointChromaSamples )
      result += "MidpointChromaSamples | ";
    if ( value & FormatFeatureFlagBits::eSampledImageYcbcrConversionLinearFilter )
      result += "SampledImageYcbcrConversionLinearFilter | ";
    if ( value & FormatFeatureFlagBits::eSampledImageYcbcrConversionSeparateReconstructionFilter )
      result += "SampledImageYcbcrConversionSeparateReconstructionFilter | ";
    if ( value & FormatFeatureFlagBits::eSampledImageYcbcrConversionChromaReconstructionExplicit )
      result += "SampledImageYcbcrConversionChromaReconstructionExplicit | ";
    if ( value & FormatFeatureFlagBits::eSampledImageYcbcrConversionChromaReconstructionExplicitForceable )
      result += "SampledImageYcbcrConversionChromaReconstructionExplicitForceable | ";
    if ( value & FormatFeatureFlagBits::eDisjoint )
      result += "Disjoint | ";
    if ( value & FormatFeatureFlagBits::eCositedChromaSamples )
      result += "CositedChromaSamples | ";
    if ( value & FormatFeatureFlagBits::eSampledImageFilterMinmax )
      result += "SampledImageFilterMinmax | ";
    if ( value & FormatFeatureFlagBits::eVideoDecodeOutputKHR )
      result += "VideoDecodeOutputKHR | ";
    if ( value & FormatFeatureFlagBits::eVideoDecodeDpbKHR )
      result += "VideoDecodeDpbKHR | ";
    if ( value & FormatFeatureFlagBits::eAccelerationStructureVertexBufferKHR )
      result += "AccelerationStructureVertexBufferKHR | ";
    if ( value & FormatFeatureFlagBits::eSampledImageFilterCubicEXT )
      result += "SampledImageFilterCubicEXT | ";
    if ( value & FormatFeatureFlagBits::eFragmentDensityMapEXT )
      result += "FragmentDensityMapEXT | ";
    if ( value & FormatFeatureFlagBits::eFragmentShadingRateAttachmentKHR )
      result += "FragmentShadingRateAttachmentKHR | ";
    if ( value & FormatFeatureFlagBits::eVideoEncodeInputKHR )
      result += "VideoEncodeInputKHR | ";
    if ( value & FormatFeatureFlagBits::eVideoEncodeDpbKHR )
      result += "VideoEncodeDpbKHR | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( ImageCreateFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & ImageCreateFlagBits::eSparseBinding )
      result += "SparseBinding | ";
    if ( value & ImageCreateFlagBits::eSparseResidency )
      result += "SparseResidency | ";
    if ( value & ImageCreateFlagBits::eSparseAliased )
      result += "SparseAliased | ";
    if ( value & ImageCreateFlagBits::eMutableFormat )
      result += "MutableFormat | ";
    if ( value & ImageCreateFlagBits::eCubeCompatible )
      result += "CubeCompatible | ";
    if ( value & ImageCreateFlagBits::eAlias )
      result += "Alias | ";
    if ( value & ImageCreateFlagBits::eSplitInstanceBindRegions )
      result += "SplitInstanceBindRegions | ";
    if ( value & ImageCreateFlagBits::e2DArrayCompatible )
      result += "2DArrayCompatible | ";
    if ( value & ImageCreateFlagBits::eBlockTexelViewCompatible )
      result += "BlockTexelViewCompatible | ";
    if ( value & ImageCreateFlagBits::eExtendedUsage )
      result += "ExtendedUsage | ";
    if ( value & ImageCreateFlagBits::eProtected )
      result += "Protected | ";
    if ( value & ImageCreateFlagBits::eDisjoint )
      result += "Disjoint | ";
    if ( value & ImageCreateFlagBits::eCornerSampledNV )
      result += "CornerSampledNV | ";
    if ( value & ImageCreateFlagBits::eSampleLocationsCompatibleDepthEXT )
      result += "SampleLocationsCompatibleDepthEXT | ";
    if ( value & ImageCreateFlagBits::eSubsampledEXT )
      result += "SubsampledEXT | ";
    if ( value & ImageCreateFlagBits::eDescriptorBufferCaptureReplayEXT )
      result += "DescriptorBufferCaptureReplayEXT | ";
    if ( value & ImageCreateFlagBits::eMultisampledRenderToSingleSampledEXT )
      result += "MultisampledRenderToSingleSampledEXT | ";
    if ( value & ImageCreateFlagBits::e2DViewCompatibleEXT )
      result += "2DViewCompatibleEXT | ";
    if ( value & ImageCreateFlagBits::eFragmentDensityMapOffsetQCOM )
      result += "FragmentDensityMapOffsetQCOM | ";
    if ( value & ImageCreateFlagBits::eVideoProfileIndependentKHR )
      result += "VideoProfileIndependentKHR | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( ImageUsageFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & ImageUsageFlagBits::eTransferSrc )
      result += "TransferSrc | ";
    if ( value & ImageUsageFlagBits::eTransferDst )
      result += "TransferDst | ";
    if ( value & ImageUsageFlagBits::eSampled )
      result += "Sampled | ";
    if ( value & ImageUsageFlagBits::eStorage )
      result += "Storage | ";
    if ( value & ImageUsageFlagBits::eColorAttachment )
      result += "ColorAttachment | ";
    if ( value & ImageUsageFlagBits::eDepthStencilAttachment )
      result += "DepthStencilAttachment | ";
    if ( value & ImageUsageFlagBits::eTransientAttachment )
      result += "TransientAttachment | ";
    if ( value & ImageUsageFlagBits::eInputAttachment )
      result += "InputAttachment | ";
    if ( value & ImageUsageFlagBits::eVideoDecodeDstKHR )
      result += "VideoDecodeDstKHR | ";
    if ( value & ImageUsageFlagBits::eVideoDecodeSrcKHR )
      result += "VideoDecodeSrcKHR | ";
    if ( value & ImageUsageFlagBits::eVideoDecodeDpbKHR )
      result += "VideoDecodeDpbKHR | ";
    if ( value & ImageUsageFlagBits::eFragmentDensityMapEXT )
      result += "FragmentDensityMapEXT | ";
    if ( value & ImageUsageFlagBits::eFragmentShadingRateAttachmentKHR )
      result += "FragmentShadingRateAttachmentKHR | ";
    if ( value & ImageUsageFlagBits::eHostTransferEXT )
      result += "HostTransferEXT | ";
    if ( value & ImageUsageFlagBits::eVideoEncodeDstKHR )
      result += "VideoEncodeDstKHR | ";
    if ( value & ImageUsageFlagBits::eVideoEncodeSrcKHR )
      result += "VideoEncodeSrcKHR | ";
    if ( value & ImageUsageFlagBits::eVideoEncodeDpbKHR )
      result += "VideoEncodeDpbKHR | ";
    if ( value & ImageUsageFlagBits::eAttachmentFeedbackLoopEXT )
      result += "AttachmentFeedbackLoopEXT | ";
    if ( value & ImageUsageFlagBits::eInvocationMaskHUAWEI )
      result += "InvocationMaskHUAWEI | ";
    if ( value & ImageUsageFlagBits::eSampleWeightQCOM )
      result += "SampleWeightQCOM | ";
    if ( value & ImageUsageFlagBits::eSampleBlockMatchQCOM )
      result += "SampleBlockMatchQCOM | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( InstanceCreateFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & InstanceCreateFlagBits::eEnumeratePortabilityKHR )
      result += "EnumeratePortabilityKHR | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( MemoryHeapFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & MemoryHeapFlagBits::eDeviceLocal )
      result += "DeviceLocal | ";
    if ( value & MemoryHeapFlagBits::eMultiInstance )
      result += "MultiInstance | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( MemoryPropertyFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & MemoryPropertyFlagBits::eDeviceLocal )
      result += "DeviceLocal | ";
    if ( value & MemoryPropertyFlagBits::eHostVisible )
      result += "HostVisible | ";
    if ( value & MemoryPropertyFlagBits::eHostCoherent )
      result += "HostCoherent | ";
    if ( value & MemoryPropertyFlagBits::eHostCached )
      result += "HostCached | ";
    if ( value & MemoryPropertyFlagBits::eLazilyAllocated )
      result += "LazilyAllocated | ";
    if ( value & MemoryPropertyFlagBits::eProtected )
      result += "Protected | ";
    if ( value & MemoryPropertyFlagBits::eDeviceCoherentAMD )
      result += "DeviceCoherentAMD | ";
    if ( value & MemoryPropertyFlagBits::eDeviceUncachedAMD )
      result += "DeviceUncachedAMD | ";
    if ( value & MemoryPropertyFlagBits::eRdmaCapableNV )
      result += "RdmaCapableNV | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( QueueFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & QueueFlagBits::eGraphics )
      result += "Graphics | ";
    if ( value & QueueFlagBits::eCompute )
      result += "Compute | ";
    if ( value & QueueFlagBits::eTransfer )
      result += "Transfer | ";
    if ( value & QueueFlagBits::eSparseBinding )
      result += "SparseBinding | ";
    if ( value & QueueFlagBits::eProtected )
      result += "Protected | ";
    if ( value & QueueFlagBits::eVideoDecodeKHR )
      result += "VideoDecodeKHR | ";
    if ( value & QueueFlagBits::eVideoEncodeKHR )
      result += "VideoEncodeKHR | ";
    if ( value & QueueFlagBits::eOpticalFlowNV )
      result += "OpticalFlowNV | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( SampleCountFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & SampleCountFlagBits::e1 )
      result += "1 | ";
    if ( value & SampleCountFlagBits::e2 )
      result += "2 | ";
    if ( value & SampleCountFlagBits::e4 )
      result += "4 | ";
    if ( value & SampleCountFlagBits::e8 )
      result += "8 | ";
    if ( value & SampleCountFlagBits::e16 )
      result += "16 | ";
    if ( value & SampleCountFlagBits::e32 )
      result += "32 | ";
    if ( value & SampleCountFlagBits::e64 )
      result += "64 | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( DeviceCreateFlags )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( DeviceQueueCreateFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & DeviceQueueCreateFlagBits::eProtected )
      result += "Protected | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineStageFlags value )
  {
    if ( !value )
      return "None";

    std::string result;
    if ( value & PipelineStageFlagBits::eTopOfPipe )
      result += "TopOfPipe | ";
    if ( value & PipelineStageFlagBits::eDrawIndirect )
      result += "DrawIndirect | ";
    if ( value & PipelineStageFlagBits::eVertexInput )
      result += "VertexInput | ";
    if ( value & PipelineStageFlagBits::eVertexShader )
      result += "VertexShader | ";
    if ( value & PipelineStageFlagBits::eTessellationControlShader )
      result += "TessellationControlShader | ";
    if ( value & PipelineStageFlagBits::eTessellationEvaluationShader )
      result += "TessellationEvaluationShader | ";
    if ( value & PipelineStageFlagBits::eGeometryShader )
      result += "GeometryShader | ";
    if ( value & PipelineStageFlagBits::eFragmentShader )
      result += "FragmentShader | ";
    if ( value & PipelineStageFlagBits::eEarlyFragmentTests )
      result += "EarlyFragmentTests | ";
    if ( value & PipelineStageFlagBits::eLateFragmentTests )
      result += "LateFragmentTests | ";
    if ( value & PipelineStageFlagBits::eColorAttachmentOutput )
      result += "ColorAttachmentOutput | ";
    if ( value & PipelineStageFlagBits::eComputeShader )
      result += "ComputeShader | ";
    if ( value & PipelineStageFlagBits::eTransfer )
      result += "Transfer | ";
    if ( value & PipelineStageFlagBits::eBottomOfPipe )
      result += "BottomOfPipe | ";
    if ( value & PipelineStageFlagBits::eHost )
      result += "Host | ";
    if ( value & PipelineStageFlagBits::eAllGraphics )
      result += "AllGraphics | ";
    if ( value & PipelineStageFlagBits::eAllCommands )
      result += "AllCommands | ";
    if ( value & PipelineStageFlagBits::eTransformFeedbackEXT )
      result += "TransformFeedbackEXT | ";
    if ( value & PipelineStageFlagBits::eConditionalRenderingEXT )
      result += "ConditionalRenderingEXT | ";
    if ( value & PipelineStageFlagBits::eAccelerationStructureBuildKHR )
      result += "AccelerationStructureBuildKHR | ";
    if ( value & PipelineStageFlagBits::eRayTracingShaderKHR )
      result += "RayTracingShaderKHR | ";
    if ( value & PipelineStageFlagBits::eFragmentDensityProcessEXT )
      result += "FragmentDensityProcessEXT | ";
    if ( value & PipelineStageFlagBits::eFragmentShadingRateAttachmentKHR )
      result += "FragmentShadingRateAttachmentKHR | ";
    if ( value & PipelineStageFlagBits::eCommandPreprocessNV )
      result += "CommandPreprocessNV | ";
    if ( value & PipelineStageFlagBits::eTaskShaderEXT )
      result += "TaskShaderEXT | ";
    if ( value & PipelineStageFlagBits::eMeshShaderEXT )
      result += "MeshShaderEXT | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( MemoryMapFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & MemoryMapFlagBits::ePlacedEXT )
      result += "PlacedEXT | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( ImageAspectFlags value )
  {
    if ( !value )
      return "None";

    std::string result;
    if ( value & ImageAspectFlagBits::eColor )
      result += "Color | ";
    if ( value & ImageAspectFlagBits::eDepth )
      result += "Depth | ";
    if ( value & ImageAspectFlagBits::eStencil )
      result += "Stencil | ";
    if ( value & ImageAspectFlagBits::eMetadata )
      result += "Metadata | ";
    if ( value & ImageAspectFlagBits::ePlane0 )
      result += "Plane0 | ";
    if ( value & ImageAspectFlagBits::ePlane1 )
      result += "Plane1 | ";
    if ( value & ImageAspectFlagBits::ePlane2 )
      result += "Plane2 | ";
    if ( value & ImageAspectFlagBits::eMemoryPlane0EXT )
      result += "MemoryPlane0EXT | ";
    if ( value & ImageAspectFlagBits::eMemoryPlane1EXT )
      result += "MemoryPlane1EXT | ";
    if ( value & ImageAspectFlagBits::eMemoryPlane2EXT )
      result += "MemoryPlane2EXT | ";
    if ( value & ImageAspectFlagBits::eMemoryPlane3EXT )
      result += "MemoryPlane3EXT | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( SparseImageFormatFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & SparseImageFormatFlagBits::eSingleMiptail )
      result += "SingleMiptail | ";
    if ( value & SparseImageFormatFlagBits::eAlignedMipSize )
      result += "AlignedMipSize | ";
    if ( value & SparseImageFormatFlagBits::eNonstandardBlockSize )
      result += "NonstandardBlockSize | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( SparseMemoryBindFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & SparseMemoryBindFlagBits::eMetadata )
      result += "Metadata | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( FenceCreateFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & FenceCreateFlagBits::eSignaled )
      result += "Signaled | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( SemaphoreCreateFlags )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( EventCreateFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & EventCreateFlagBits::eDeviceOnly )
      result += "DeviceOnly | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( QueryPipelineStatisticFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & QueryPipelineStatisticFlagBits::eInputAssemblyVertices )
      result += "InputAssemblyVertices | ";
    if ( value & QueryPipelineStatisticFlagBits::eInputAssemblyPrimitives )
      result += "InputAssemblyPrimitives | ";
    if ( value & QueryPipelineStatisticFlagBits::eVertexShaderInvocations )
      result += "VertexShaderInvocations | ";
    if ( value & QueryPipelineStatisticFlagBits::eGeometryShaderInvocations )
      result += "GeometryShaderInvocations | ";
    if ( value & QueryPipelineStatisticFlagBits::eGeometryShaderPrimitives )
      result += "GeometryShaderPrimitives | ";
    if ( value & QueryPipelineStatisticFlagBits::eClippingInvocations )
      result += "ClippingInvocations | ";
    if ( value & QueryPipelineStatisticFlagBits::eClippingPrimitives )
      result += "ClippingPrimitives | ";
    if ( value & QueryPipelineStatisticFlagBits::eFragmentShaderInvocations )
      result += "FragmentShaderInvocations | ";
    if ( value & QueryPipelineStatisticFlagBits::eTessellationControlShaderPatches )
      result += "TessellationControlShaderPatches | ";
    if ( value & QueryPipelineStatisticFlagBits::eTessellationEvaluationShaderInvocations )
      result += "TessellationEvaluationShaderInvocations | ";
    if ( value & QueryPipelineStatisticFlagBits::eComputeShaderInvocations )
      result += "ComputeShaderInvocations | ";
    if ( value & QueryPipelineStatisticFlagBits::eTaskShaderInvocationsEXT )
      result += "TaskShaderInvocationsEXT | ";
    if ( value & QueryPipelineStatisticFlagBits::eMeshShaderInvocationsEXT )
      result += "MeshShaderInvocationsEXT | ";
    if ( value & QueryPipelineStatisticFlagBits::eClusterCullingShaderInvocationsHUAWEI )
      result += "ClusterCullingShaderInvocationsHUAWEI | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( QueryPoolCreateFlags )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( QueryResultFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & QueryResultFlagBits::e64 )
      result += "64 | ";
    if ( value & QueryResultFlagBits::eWait )
      result += "Wait | ";
    if ( value & QueryResultFlagBits::eWithAvailability )
      result += "WithAvailability | ";
    if ( value & QueryResultFlagBits::ePartial )
      result += "Partial | ";
    if ( value & QueryResultFlagBits::eWithStatusKHR )
      result += "WithStatusKHR | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( BufferCreateFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & BufferCreateFlagBits::eSparseBinding )
      result += "SparseBinding | ";
    if ( value & BufferCreateFlagBits::eSparseResidency )
      result += "SparseResidency | ";
    if ( value & BufferCreateFlagBits::eSparseAliased )
      result += "SparseAliased | ";
    if ( value & BufferCreateFlagBits::eProtected )
      result += "Protected | ";
    if ( value & BufferCreateFlagBits::eDeviceAddressCaptureReplay )
      result += "DeviceAddressCaptureReplay | ";
    if ( value & BufferCreateFlagBits::eDescriptorBufferCaptureReplayEXT )
      result += "DescriptorBufferCaptureReplayEXT | ";
    if ( value & BufferCreateFlagBits::eVideoProfileIndependentKHR )
      result += "VideoProfileIndependentKHR | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( BufferUsageFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & BufferUsageFlagBits::eTransferSrc )
      result += "TransferSrc | ";
    if ( value & BufferUsageFlagBits::eTransferDst )
      result += "TransferDst | ";
    if ( value & BufferUsageFlagBits::eUniformTexelBuffer )
      result += "UniformTexelBuffer | ";
    if ( value & BufferUsageFlagBits::eStorageTexelBuffer )
      result += "StorageTexelBuffer | ";
    if ( value & BufferUsageFlagBits::eUniformBuffer )
      result += "UniformBuffer | ";
    if ( value & BufferUsageFlagBits::eStorageBuffer )
      result += "StorageBuffer | ";
    if ( value & BufferUsageFlagBits::eIndexBuffer )
      result += "IndexBuffer | ";
    if ( value & BufferUsageFlagBits::eVertexBuffer )
      result += "VertexBuffer | ";
    if ( value & BufferUsageFlagBits::eIndirectBuffer )
      result += "IndirectBuffer | ";
    if ( value & BufferUsageFlagBits::eShaderDeviceAddress )
      result += "ShaderDeviceAddress | ";
    if ( value & BufferUsageFlagBits::eVideoDecodeSrcKHR )
      result += "VideoDecodeSrcKHR | ";
    if ( value & BufferUsageFlagBits::eVideoDecodeDstKHR )
      result += "VideoDecodeDstKHR | ";
    if ( value & BufferUsageFlagBits::eTransformFeedbackBufferEXT )
      result += "TransformFeedbackBufferEXT | ";
    if ( value & BufferUsageFlagBits::eTransformFeedbackCounterBufferEXT )
      result += "TransformFeedbackCounterBufferEXT | ";
    if ( value & BufferUsageFlagBits::eConditionalRenderingEXT )
      result += "ConditionalRenderingEXT | ";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & BufferUsageFlagBits::eExecutionGraphScratchAMDX )
      result += "ExecutionGraphScratchAMDX | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    if ( value & BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR )
      result += "AccelerationStructureBuildInputReadOnlyKHR | ";
    if ( value & BufferUsageFlagBits::eAccelerationStructureStorageKHR )
      result += "AccelerationStructureStorageKHR | ";
    if ( value & BufferUsageFlagBits::eShaderBindingTableKHR )
      result += "ShaderBindingTableKHR | ";
    if ( value & BufferUsageFlagBits::eVideoEncodeDstKHR )
      result += "VideoEncodeDstKHR | ";
    if ( value & BufferUsageFlagBits::eVideoEncodeSrcKHR )
      result += "VideoEncodeSrcKHR | ";
    if ( value & BufferUsageFlagBits::eSamplerDescriptorBufferEXT )
      result += "SamplerDescriptorBufferEXT | ";
    if ( value & BufferUsageFlagBits::eResourceDescriptorBufferEXT )
      result += "ResourceDescriptorBufferEXT | ";
    if ( value & BufferUsageFlagBits::ePushDescriptorsDescriptorBufferEXT )
      result += "PushDescriptorsDescriptorBufferEXT | ";
    if ( value & BufferUsageFlagBits::eMicromapBuildInputReadOnlyEXT )
      result += "MicromapBuildInputReadOnlyEXT | ";
    if ( value & BufferUsageFlagBits::eMicromapStorageEXT )
      result += "MicromapStorageEXT | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( BufferViewCreateFlags )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( ImageViewCreateFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & ImageViewCreateFlagBits::eFragmentDensityMapDynamicEXT )
      result += "FragmentDensityMapDynamicEXT | ";
    if ( value & ImageViewCreateFlagBits::eDescriptorBufferCaptureReplayEXT )
      result += "DescriptorBufferCaptureReplayEXT | ";
    if ( value & ImageViewCreateFlagBits::eFragmentDensityMapDeferredEXT )
      result += "FragmentDensityMapDeferredEXT | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( ShaderModuleCreateFlags )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineCacheCreateFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & PipelineCacheCreateFlagBits::eExternallySynchronized )
      result += "ExternallySynchronized | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( ColorComponentFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & ColorComponentFlagBits::eR )
      result += "R | ";
    if ( value & ColorComponentFlagBits::eG )
      result += "G | ";
    if ( value & ColorComponentFlagBits::eB )
      result += "B | ";
    if ( value & ColorComponentFlagBits::eA )
      result += "A | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( CullModeFlags value )
  {
    if ( !value )
      return "None";

    std::string result;
    if ( value & CullModeFlagBits::eFront )
      result += "Front | ";
    if ( value & CullModeFlagBits::eBack )
      result += "Back | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineColorBlendStateCreateFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & PipelineColorBlendStateCreateFlagBits::eRasterizationOrderAttachmentAccessEXT )
      result += "RasterizationOrderAttachmentAccessEXT | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineCreateFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & PipelineCreateFlagBits::eDisableOptimization )
      result += "DisableOptimization | ";
    if ( value & PipelineCreateFlagBits::eAllowDerivatives )
      result += "AllowDerivatives | ";
    if ( value & PipelineCreateFlagBits::eDerivative )
      result += "Derivative | ";
    if ( value & PipelineCreateFlagBits::eViewIndexFromDeviceIndex )
      result += "ViewIndexFromDeviceIndex | ";
    if ( value & PipelineCreateFlagBits::eDispatchBase )
      result += "DispatchBase | ";
    if ( value & PipelineCreateFlagBits::eFailOnPipelineCompileRequired )
      result += "FailOnPipelineCompileRequired | ";
    if ( value & PipelineCreateFlagBits::eEarlyReturnOnFailure )
      result += "EarlyReturnOnFailure | ";
    if ( value & PipelineCreateFlagBits::eRenderingFragmentShadingRateAttachmentKHR )
      result += "RenderingFragmentShadingRateAttachmentKHR | ";
    if ( value & PipelineCreateFlagBits::eRenderingFragmentDensityMapAttachmentEXT )
      result += "RenderingFragmentDensityMapAttachmentEXT | ";
    if ( value & PipelineCreateFlagBits::eRayTracingNoNullAnyHitShadersKHR )
      result += "RayTracingNoNullAnyHitShadersKHR | ";
    if ( value & PipelineCreateFlagBits::eRayTracingNoNullClosestHitShadersKHR )
      result += "RayTracingNoNullClosestHitShadersKHR | ";
    if ( value & PipelineCreateFlagBits::eRayTracingNoNullMissShadersKHR )
      result += "RayTracingNoNullMissShadersKHR | ";
    if ( value & PipelineCreateFlagBits::eRayTracingNoNullIntersectionShadersKHR )
      result += "RayTracingNoNullIntersectionShadersKHR | ";
    if ( value & PipelineCreateFlagBits::eRayTracingSkipTrianglesKHR )
      result += "RayTracingSkipTrianglesKHR | ";
    if ( value & PipelineCreateFlagBits::eRayTracingSkipAabbsKHR )
      result += "RayTracingSkipAabbsKHR | ";
    if ( value & PipelineCreateFlagBits::eRayTracingShaderGroupHandleCaptureReplayKHR )
      result += "RayTracingShaderGroupHandleCaptureReplayKHR | ";
    if ( value & PipelineCreateFlagBits::eDeferCompileNV )
      result += "DeferCompileNV | ";
    if ( value & PipelineCreateFlagBits::eCaptureStatisticsKHR )
      result += "CaptureStatisticsKHR | ";
    if ( value & PipelineCreateFlagBits::eCaptureInternalRepresentationsKHR )
      result += "CaptureInternalRepresentationsKHR | ";
    if ( value & PipelineCreateFlagBits::eIndirectBindableNV )
      result += "IndirectBindableNV | ";
    if ( value & PipelineCreateFlagBits::eLibraryKHR )
      result += "LibraryKHR | ";
    if ( value & PipelineCreateFlagBits::eDescriptorBufferEXT )
      result += "DescriptorBufferEXT | ";
    if ( value & PipelineCreateFlagBits::eRetainLinkTimeOptimizationInfoEXT )
      result += "RetainLinkTimeOptimizationInfoEXT | ";
    if ( value & PipelineCreateFlagBits::eLinkTimeOptimizationEXT )
      result += "LinkTimeOptimizationEXT | ";
    if ( value & PipelineCreateFlagBits::eRayTracingAllowMotionNV )
      result += "RayTracingAllowMotionNV | ";
    if ( value & PipelineCreateFlagBits::eColorAttachmentFeedbackLoopEXT )
      result += "ColorAttachmentFeedbackLoopEXT | ";
    if ( value & PipelineCreateFlagBits::eDepthStencilAttachmentFeedbackLoopEXT )
      result += "DepthStencilAttachmentFeedbackLoopEXT | ";
    if ( value & PipelineCreateFlagBits::eRayTracingOpacityMicromapEXT )
      result += "RayTracingOpacityMicromapEXT | ";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & PipelineCreateFlagBits::eRayTracingDisplacementMicromapNV )
      result += "RayTracingDisplacementMicromapNV | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    if ( value & PipelineCreateFlagBits::eNoProtectedAccessEXT )
      result += "NoProtectedAccessEXT | ";
    if ( value & PipelineCreateFlagBits::eProtectedAccessOnlyEXT )
      result += "ProtectedAccessOnlyEXT | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineDepthStencilStateCreateFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & PipelineDepthStencilStateCreateFlagBits::eRasterizationOrderAttachmentDepthAccessEXT )
      result += "RasterizationOrderAttachmentDepthAccessEXT | ";
    if ( value & PipelineDepthStencilStateCreateFlagBits::eRasterizationOrderAttachmentStencilAccessEXT )
      result += "RasterizationOrderAttachmentStencilAccessEXT | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineDynamicStateCreateFlags )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineInputAssemblyStateCreateFlags )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineLayoutCreateFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & PipelineLayoutCreateFlagBits::eIndependentSetsEXT )
      result += "IndependentSetsEXT | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineMultisampleStateCreateFlags )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineRasterizationStateCreateFlags )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineShaderStageCreateFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & PipelineShaderStageCreateFlagBits::eAllowVaryingSubgroupSize )
      result += "AllowVaryingSubgroupSize | ";
    if ( value & PipelineShaderStageCreateFlagBits::eRequireFullSubgroups )
      result += "RequireFullSubgroups | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineTessellationStateCreateFlags )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineVertexInputStateCreateFlags )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineViewportStateCreateFlags )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( ShaderStageFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & ShaderStageFlagBits::eVertex )
      result += "Vertex | ";
    if ( value & ShaderStageFlagBits::eTessellationControl )
      result += "TessellationControl | ";
    if ( value & ShaderStageFlagBits::eTessellationEvaluation )
      result += "TessellationEvaluation | ";
    if ( value & ShaderStageFlagBits::eGeometry )
      result += "Geometry | ";
    if ( value & ShaderStageFlagBits::eFragment )
      result += "Fragment | ";
    if ( value & ShaderStageFlagBits::eCompute )
      result += "Compute | ";
    if ( value & ShaderStageFlagBits::eRaygenKHR )
      result += "RaygenKHR | ";
    if ( value & ShaderStageFlagBits::eAnyHitKHR )
      result += "AnyHitKHR | ";
    if ( value & ShaderStageFlagBits::eClosestHitKHR )
      result += "ClosestHitKHR | ";
    if ( value & ShaderStageFlagBits::eMissKHR )
      result += "MissKHR | ";
    if ( value & ShaderStageFlagBits::eIntersectionKHR )
      result += "IntersectionKHR | ";
    if ( value & ShaderStageFlagBits::eCallableKHR )
      result += "CallableKHR | ";
    if ( value & ShaderStageFlagBits::eTaskEXT )
      result += "TaskEXT | ";
    if ( value & ShaderStageFlagBits::eMeshEXT )
      result += "MeshEXT | ";
    if ( value & ShaderStageFlagBits::eSubpassShadingHUAWEI )
      result += "SubpassShadingHUAWEI | ";
    if ( value & ShaderStageFlagBits::eClusterCullingHUAWEI )
      result += "ClusterCullingHUAWEI | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( SamplerCreateFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & SamplerCreateFlagBits::eSubsampledEXT )
      result += "SubsampledEXT | ";
    if ( value & SamplerCreateFlagBits::eSubsampledCoarseReconstructionEXT )
      result += "SubsampledCoarseReconstructionEXT | ";
    if ( value & SamplerCreateFlagBits::eDescriptorBufferCaptureReplayEXT )
      result += "DescriptorBufferCaptureReplayEXT | ";
    if ( value & SamplerCreateFlagBits::eNonSeamlessCubeMapEXT )
      result += "NonSeamlessCubeMapEXT | ";
    if ( value & SamplerCreateFlagBits::eImageProcessingQCOM )
      result += "ImageProcessingQCOM | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( DescriptorPoolCreateFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & DescriptorPoolCreateFlagBits::eFreeDescriptorSet )
      result += "FreeDescriptorSet | ";
    if ( value & DescriptorPoolCreateFlagBits::eUpdateAfterBind )
      result += "UpdateAfterBind | ";
    if ( value & DescriptorPoolCreateFlagBits::eHostOnlyEXT )
      result += "HostOnlyEXT | ";
    if ( value & DescriptorPoolCreateFlagBits::eAllowOverallocationSetsNV )
      result += "AllowOverallocationSetsNV | ";
    if ( value & DescriptorPoolCreateFlagBits::eAllowOverallocationPoolsNV )
      result += "AllowOverallocationPoolsNV | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( DescriptorPoolResetFlags )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( DescriptorSetLayoutCreateFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool )
      result += "UpdateAfterBindPool | ";
    if ( value & DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR )
      result += "PushDescriptorKHR | ";
    if ( value & DescriptorSetLayoutCreateFlagBits::eDescriptorBufferEXT )
      result += "DescriptorBufferEXT | ";
    if ( value & DescriptorSetLayoutCreateFlagBits::eEmbeddedImmutableSamplersEXT )
      result += "EmbeddedImmutableSamplersEXT | ";
    if ( value & DescriptorSetLayoutCreateFlagBits::eIndirectBindableNV )
      result += "IndirectBindableNV | ";
    if ( value & DescriptorSetLayoutCreateFlagBits::eHostOnlyPoolEXT )
      result += "HostOnlyPoolEXT | ";
    if ( value & DescriptorSetLayoutCreateFlagBits::ePerStageNV )
      result += "PerStageNV | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( AccessFlags value )
  {
    if ( !value )
      return "None";

    std::string result;
    if ( value & AccessFlagBits::eIndirectCommandRead )
      result += "IndirectCommandRead | ";
    if ( value & AccessFlagBits::eIndexRead )
      result += "IndexRead | ";
    if ( value & AccessFlagBits::eVertexAttributeRead )
      result += "VertexAttributeRead | ";
    if ( value & AccessFlagBits::eUniformRead )
      result += "UniformRead | ";
    if ( value & AccessFlagBits::eInputAttachmentRead )
      result += "InputAttachmentRead | ";
    if ( value & AccessFlagBits::eShaderRead )
      result += "ShaderRead | ";
    if ( value & AccessFlagBits::eShaderWrite )
      result += "ShaderWrite | ";
    if ( value & AccessFlagBits::eColorAttachmentRead )
      result += "ColorAttachmentRead | ";
    if ( value & AccessFlagBits::eColorAttachmentWrite )
      result += "ColorAttachmentWrite | ";
    if ( value & AccessFlagBits::eDepthStencilAttachmentRead )
      result += "DepthStencilAttachmentRead | ";
    if ( value & AccessFlagBits::eDepthStencilAttachmentWrite )
      result += "DepthStencilAttachmentWrite | ";
    if ( value & AccessFlagBits::eTransferRead )
      result += "TransferRead | ";
    if ( value & AccessFlagBits::eTransferWrite )
      result += "TransferWrite | ";
    if ( value & AccessFlagBits::eHostRead )
      result += "HostRead | ";
    if ( value & AccessFlagBits::eHostWrite )
      result += "HostWrite | ";
    if ( value & AccessFlagBits::eMemoryRead )
      result += "MemoryRead | ";
    if ( value & AccessFlagBits::eMemoryWrite )
      result += "MemoryWrite | ";
    if ( value & AccessFlagBits::eTransformFeedbackWriteEXT )
      result += "TransformFeedbackWriteEXT | ";
    if ( value & AccessFlagBits::eTransformFeedbackCounterReadEXT )
      result += "TransformFeedbackCounterReadEXT | ";
    if ( value & AccessFlagBits::eTransformFeedbackCounterWriteEXT )
      result += "TransformFeedbackCounterWriteEXT | ";
    if ( value & AccessFlagBits::eConditionalRenderingReadEXT )
      result += "ConditionalRenderingReadEXT | ";
    if ( value & AccessFlagBits::eColorAttachmentReadNoncoherentEXT )
      result += "ColorAttachmentReadNoncoherentEXT | ";
    if ( value & AccessFlagBits::eAccelerationStructureReadKHR )
      result += "AccelerationStructureReadKHR | ";
    if ( value & AccessFlagBits::eAccelerationStructureWriteKHR )
      result += "AccelerationStructureWriteKHR | ";
    if ( value & AccessFlagBits::eFragmentDensityMapReadEXT )
      result += "FragmentDensityMapReadEXT | ";
    if ( value & AccessFlagBits::eFragmentShadingRateAttachmentReadKHR )
      result += "FragmentShadingRateAttachmentReadKHR | ";
    if ( value & AccessFlagBits::eCommandPreprocessReadNV )
      result += "CommandPreprocessReadNV | ";
    if ( value & AccessFlagBits::eCommandPreprocessWriteNV )
      result += "CommandPreprocessWriteNV | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( AttachmentDescriptionFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & AttachmentDescriptionFlagBits::eMayAlias )
      result += "MayAlias | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( DependencyFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & DependencyFlagBits::eByRegion )
      result += "ByRegion | ";
    if ( value & DependencyFlagBits::eDeviceGroup )
      result += "DeviceGroup | ";
    if ( value & DependencyFlagBits::eViewLocal )
      result += "ViewLocal | ";
    if ( value & DependencyFlagBits::eFeedbackLoopEXT )
      result += "FeedbackLoopEXT | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( FramebufferCreateFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & FramebufferCreateFlagBits::eImageless )
      result += "Imageless | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( RenderPassCreateFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & RenderPassCreateFlagBits::eTransformQCOM )
      result += "TransformQCOM | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( SubpassDescriptionFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & SubpassDescriptionFlagBits::ePerViewAttributesNVX )
      result += "PerViewAttributesNVX | ";
    if ( value & SubpassDescriptionFlagBits::ePerViewPositionXOnlyNVX )
      result += "PerViewPositionXOnlyNVX | ";
    if ( value & SubpassDescriptionFlagBits::eFragmentRegionQCOM )
      result += "FragmentRegionQCOM | ";
    if ( value & SubpassDescriptionFlagBits::eShaderResolveQCOM )
      result += "ShaderResolveQCOM | ";
    if ( value & SubpassDescriptionFlagBits::eRasterizationOrderAttachmentColorAccessEXT )
      result += "RasterizationOrderAttachmentColorAccessEXT | ";
    if ( value & SubpassDescriptionFlagBits::eRasterizationOrderAttachmentDepthAccessEXT )
      result += "RasterizationOrderAttachmentDepthAccessEXT | ";
    if ( value & SubpassDescriptionFlagBits::eRasterizationOrderAttachmentStencilAccessEXT )
      result += "RasterizationOrderAttachmentStencilAccessEXT | ";
    if ( value & SubpassDescriptionFlagBits::eEnableLegacyDitheringEXT )
      result += "EnableLegacyDitheringEXT | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( CommandPoolCreateFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & CommandPoolCreateFlagBits::eTransient )
      result += "Transient | ";
    if ( value & CommandPoolCreateFlagBits::eResetCommandBuffer )
      result += "ResetCommandBuffer | ";
    if ( value & CommandPoolCreateFlagBits::eProtected )
      result += "Protected | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( CommandPoolResetFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & CommandPoolResetFlagBits::eReleaseResources )
      result += "ReleaseResources | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( CommandBufferResetFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & CommandBufferResetFlagBits::eReleaseResources )
      result += "ReleaseResources | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( CommandBufferUsageFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & CommandBufferUsageFlagBits::eOneTimeSubmit )
      result += "OneTimeSubmit | ";
    if ( value & CommandBufferUsageFlagBits::eRenderPassContinue )
      result += "RenderPassContinue | ";
    if ( value & CommandBufferUsageFlagBits::eSimultaneousUse )
      result += "SimultaneousUse | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( QueryControlFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & QueryControlFlagBits::ePrecise )
      result += "Precise | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( StencilFaceFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & StencilFaceFlagBits::eFront )
      result += "Front | ";
    if ( value & StencilFaceFlagBits::eBack )
      result += "Back | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_VERSION_1_1 ===

  VULKAN_HPP_INLINE std::string to_string( SubgroupFeatureFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & SubgroupFeatureFlagBits::eBasic )
      result += "Basic | ";
    if ( value & SubgroupFeatureFlagBits::eVote )
      result += "Vote | ";
    if ( value & SubgroupFeatureFlagBits::eArithmetic )
      result += "Arithmetic | ";
    if ( value & SubgroupFeatureFlagBits::eBallot )
      result += "Ballot | ";
    if ( value & SubgroupFeatureFlagBits::eShuffle )
      result += "Shuffle | ";
    if ( value & SubgroupFeatureFlagBits::eShuffleRelative )
      result += "ShuffleRelative | ";
    if ( value & SubgroupFeatureFlagBits::eClustered )
      result += "Clustered | ";
    if ( value & SubgroupFeatureFlagBits::eQuad )
      result += "Quad | ";
    if ( value & SubgroupFeatureFlagBits::ePartitionedNV )
      result += "PartitionedNV | ";
    if ( value & SubgroupFeatureFlagBits::eRotateKHR )
      result += "RotateKHR | ";
    if ( value & SubgroupFeatureFlagBits::eRotateClusteredKHR )
      result += "RotateClusteredKHR | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( PeerMemoryFeatureFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & PeerMemoryFeatureFlagBits::eCopySrc )
      result += "CopySrc | ";
    if ( value & PeerMemoryFeatureFlagBits::eCopyDst )
      result += "CopyDst | ";
    if ( value & PeerMemoryFeatureFlagBits::eGenericSrc )
      result += "GenericSrc | ";
    if ( value & PeerMemoryFeatureFlagBits::eGenericDst )
      result += "GenericDst | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( MemoryAllocateFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & MemoryAllocateFlagBits::eDeviceMask )
      result += "DeviceMask | ";
    if ( value & MemoryAllocateFlagBits::eDeviceAddress )
      result += "DeviceAddress | ";
    if ( value & MemoryAllocateFlagBits::eDeviceAddressCaptureReplay )
      result += "DeviceAddressCaptureReplay | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( CommandPoolTrimFlags )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( DescriptorUpdateTemplateCreateFlags )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( ExternalMemoryHandleTypeFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & ExternalMemoryHandleTypeFlagBits::eOpaqueFd )
      result += "OpaqueFd | ";
    if ( value & ExternalMemoryHandleTypeFlagBits::eOpaqueWin32 )
      result += "OpaqueWin32 | ";
    if ( value & ExternalMemoryHandleTypeFlagBits::eOpaqueWin32Kmt )
      result += "OpaqueWin32Kmt | ";
    if ( value & ExternalMemoryHandleTypeFlagBits::eD3D11Texture )
      result += "D3D11Texture | ";
    if ( value & ExternalMemoryHandleTypeFlagBits::eD3D11TextureKmt )
      result += "D3D11TextureKmt | ";
    if ( value & ExternalMemoryHandleTypeFlagBits::eD3D12Heap )
      result += "D3D12Heap | ";
    if ( value & ExternalMemoryHandleTypeFlagBits::eD3D12Resource )
      result += "D3D12Resource | ";
    if ( value & ExternalMemoryHandleTypeFlagBits::eDmaBufEXT )
      result += "DmaBufEXT | ";
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
    if ( value & ExternalMemoryHandleTypeFlagBits::eAndroidHardwareBufferANDROID )
      result += "AndroidHardwareBufferANDROID | ";
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
    if ( value & ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT )
      result += "HostAllocationEXT | ";
    if ( value & ExternalMemoryHandleTypeFlagBits::eHostMappedForeignMemoryEXT )
      result += "HostMappedForeignMemoryEXT | ";
#if defined( VK_USE_PLATFORM_FUCHSIA )
    if ( value & ExternalMemoryHandleTypeFlagBits::eZirconVmoFUCHSIA )
      result += "ZirconVmoFUCHSIA | ";
#endif /*VK_USE_PLATFORM_FUCHSIA*/
    if ( value & ExternalMemoryHandleTypeFlagBits::eRdmaAddressNV )
      result += "RdmaAddressNV | ";
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
    if ( value & ExternalMemoryHandleTypeFlagBits::eScreenBufferQNX )
      result += "ScreenBufferQNX | ";
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( ExternalMemoryFeatureFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & ExternalMemoryFeatureFlagBits::eDedicatedOnly )
      result += "DedicatedOnly | ";
    if ( value & ExternalMemoryFeatureFlagBits::eExportable )
      result += "Exportable | ";
    if ( value & ExternalMemoryFeatureFlagBits::eImportable )
      result += "Importable | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( ExternalFenceHandleTypeFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & ExternalFenceHandleTypeFlagBits::eOpaqueFd )
      result += "OpaqueFd | ";
    if ( value & ExternalFenceHandleTypeFlagBits::eOpaqueWin32 )
      result += "OpaqueWin32 | ";
    if ( value & ExternalFenceHandleTypeFlagBits::eOpaqueWin32Kmt )
      result += "OpaqueWin32Kmt | ";
    if ( value & ExternalFenceHandleTypeFlagBits::eSyncFd )
      result += "SyncFd | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( ExternalFenceFeatureFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & ExternalFenceFeatureFlagBits::eExportable )
      result += "Exportable | ";
    if ( value & ExternalFenceFeatureFlagBits::eImportable )
      result += "Importable | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( FenceImportFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & FenceImportFlagBits::eTemporary )
      result += "Temporary | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( SemaphoreImportFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & SemaphoreImportFlagBits::eTemporary )
      result += "Temporary | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( ExternalSemaphoreHandleTypeFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd )
      result += "OpaqueFd | ";
    if ( value & ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32 )
      result += "OpaqueWin32 | ";
    if ( value & ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32Kmt )
      result += "OpaqueWin32Kmt | ";
    if ( value & ExternalSemaphoreHandleTypeFlagBits::eD3D12Fence )
      result += "D3D12Fence | ";
    if ( value & ExternalSemaphoreHandleTypeFlagBits::eSyncFd )
      result += "SyncFd | ";
#if defined( VK_USE_PLATFORM_FUCHSIA )
    if ( value & ExternalSemaphoreHandleTypeFlagBits::eZirconEventFUCHSIA )
      result += "ZirconEventFUCHSIA | ";
#endif /*VK_USE_PLATFORM_FUCHSIA*/

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( ExternalSemaphoreFeatureFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & ExternalSemaphoreFeatureFlagBits::eExportable )
      result += "Exportable | ";
    if ( value & ExternalSemaphoreFeatureFlagBits::eImportable )
      result += "Importable | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_VERSION_1_2 ===

  VULKAN_HPP_INLINE std::string to_string( DescriptorBindingFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & DescriptorBindingFlagBits::eUpdateAfterBind )
      result += "UpdateAfterBind | ";
    if ( value & DescriptorBindingFlagBits::eUpdateUnusedWhilePending )
      result += "UpdateUnusedWhilePending | ";
    if ( value & DescriptorBindingFlagBits::ePartiallyBound )
      result += "PartiallyBound | ";
    if ( value & DescriptorBindingFlagBits::eVariableDescriptorCount )
      result += "VariableDescriptorCount | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( ResolveModeFlags value )
  {
    if ( !value )
      return "None";

    std::string result;
    if ( value & ResolveModeFlagBits::eSampleZero )
      result += "SampleZero | ";
    if ( value & ResolveModeFlagBits::eAverage )
      result += "Average | ";
    if ( value & ResolveModeFlagBits::eMin )
      result += "Min | ";
    if ( value & ResolveModeFlagBits::eMax )
      result += "Max | ";
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
    if ( value & ResolveModeFlagBits::eExternalFormatDownsampleANDROID )
      result += "ExternalFormatDownsampleANDROID | ";
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( SemaphoreWaitFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & SemaphoreWaitFlagBits::eAny )
      result += "Any | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_VERSION_1_3 ===

  VULKAN_HPP_INLINE std::string to_string( PipelineCreationFeedbackFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & PipelineCreationFeedbackFlagBits::eValid )
      result += "Valid | ";
    if ( value & PipelineCreationFeedbackFlagBits::eApplicationPipelineCacheHit )
      result += "ApplicationPipelineCacheHit | ";
    if ( value & PipelineCreationFeedbackFlagBits::eBasePipelineAcceleration )
      result += "BasePipelineAcceleration | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( ToolPurposeFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & ToolPurposeFlagBits::eValidation )
      result += "Validation | ";
    if ( value & ToolPurposeFlagBits::eProfiling )
      result += "Profiling | ";
    if ( value & ToolPurposeFlagBits::eTracing )
      result += "Tracing | ";
    if ( value & ToolPurposeFlagBits::eAdditionalFeatures )
      result += "AdditionalFeatures | ";
    if ( value & ToolPurposeFlagBits::eModifyingFeatures )
      result += "ModifyingFeatures | ";
    if ( value & ToolPurposeFlagBits::eDebugReportingEXT )
      result += "DebugReportingEXT | ";
    if ( value & ToolPurposeFlagBits::eDebugMarkersEXT )
      result += "DebugMarkersEXT | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( PrivateDataSlotCreateFlags )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineStageFlags2 value )
  {
    if ( !value )
      return "None";

    std::string result;
    if ( value & PipelineStageFlagBits2::eTopOfPipe )
      result += "TopOfPipe | ";
    if ( value & PipelineStageFlagBits2::eDrawIndirect )
      result += "DrawIndirect | ";
    if ( value & PipelineStageFlagBits2::eVertexInput )
      result += "VertexInput | ";
    if ( value & PipelineStageFlagBits2::eVertexShader )
      result += "VertexShader | ";
    if ( value & PipelineStageFlagBits2::eTessellationControlShader )
      result += "TessellationControlShader | ";
    if ( value & PipelineStageFlagBits2::eTessellationEvaluationShader )
      result += "TessellationEvaluationShader | ";
    if ( value & PipelineStageFlagBits2::eGeometryShader )
      result += "GeometryShader | ";
    if ( value & PipelineStageFlagBits2::eFragmentShader )
      result += "FragmentShader | ";
    if ( value & PipelineStageFlagBits2::eEarlyFragmentTests )
      result += "EarlyFragmentTests | ";
    if ( value & PipelineStageFlagBits2::eLateFragmentTests )
      result += "LateFragmentTests | ";
    if ( value & PipelineStageFlagBits2::eColorAttachmentOutput )
      result += "ColorAttachmentOutput | ";
    if ( value & PipelineStageFlagBits2::eComputeShader )
      result += "ComputeShader | ";
    if ( value & PipelineStageFlagBits2::eAllTransfer )
      result += "AllTransfer | ";
    if ( value & PipelineStageFlagBits2::eBottomOfPipe )
      result += "BottomOfPipe | ";
    if ( value & PipelineStageFlagBits2::eHost )
      result += "Host | ";
    if ( value & PipelineStageFlagBits2::eAllGraphics )
      result += "AllGraphics | ";
    if ( value & PipelineStageFlagBits2::eAllCommands )
      result += "AllCommands | ";
    if ( value & PipelineStageFlagBits2::eCopy )
      result += "Copy | ";
    if ( value & PipelineStageFlagBits2::eResolve )
      result += "Resolve | ";
    if ( value & PipelineStageFlagBits2::eBlit )
      result += "Blit | ";
    if ( value & PipelineStageFlagBits2::eClear )
      result += "Clear | ";
    if ( value & PipelineStageFlagBits2::eIndexInput )
      result += "IndexInput | ";
    if ( value & PipelineStageFlagBits2::eVertexAttributeInput )
      result += "VertexAttributeInput | ";
    if ( value & PipelineStageFlagBits2::ePreRasterizationShaders )
      result += "PreRasterizationShaders | ";
    if ( value & PipelineStageFlagBits2::eVideoDecodeKHR )
      result += "VideoDecodeKHR | ";
    if ( value & PipelineStageFlagBits2::eVideoEncodeKHR )
      result += "VideoEncodeKHR | ";
    if ( value & PipelineStageFlagBits2::eTransformFeedbackEXT )
      result += "TransformFeedbackEXT | ";
    if ( value & PipelineStageFlagBits2::eConditionalRenderingEXT )
      result += "ConditionalRenderingEXT | ";
    if ( value & PipelineStageFlagBits2::eCommandPreprocessNV )
      result += "CommandPreprocessNV | ";
    if ( value & PipelineStageFlagBits2::eFragmentShadingRateAttachmentKHR )
      result += "FragmentShadingRateAttachmentKHR | ";
    if ( value & PipelineStageFlagBits2::eAccelerationStructureBuildKHR )
      result += "AccelerationStructureBuildKHR | ";
    if ( value & PipelineStageFlagBits2::eRayTracingShaderKHR )
      result += "RayTracingShaderKHR | ";
    if ( value & PipelineStageFlagBits2::eFragmentDensityProcessEXT )
      result += "FragmentDensityProcessEXT | ";
    if ( value & PipelineStageFlagBits2::eTaskShaderEXT )
      result += "TaskShaderEXT | ";
    if ( value & PipelineStageFlagBits2::eMeshShaderEXT )
      result += "MeshShaderEXT | ";
    if ( value & PipelineStageFlagBits2::eSubpassShaderHUAWEI )
      result += "SubpassShaderHUAWEI | ";
    if ( value & PipelineStageFlagBits2::eInvocationMaskHUAWEI )
      result += "InvocationMaskHUAWEI | ";
    if ( value & PipelineStageFlagBits2::eAccelerationStructureCopyKHR )
      result += "AccelerationStructureCopyKHR | ";
    if ( value & PipelineStageFlagBits2::eMicromapBuildEXT )
      result += "MicromapBuildEXT | ";
    if ( value & PipelineStageFlagBits2::eClusterCullingShaderHUAWEI )
      result += "ClusterCullingShaderHUAWEI | ";
    if ( value & PipelineStageFlagBits2::eOpticalFlowNV )
      result += "OpticalFlowNV | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( AccessFlags2 value )
  {
    if ( !value )
      return "None";

    std::string result;
    if ( value & AccessFlagBits2::eIndirectCommandRead )
      result += "IndirectCommandRead | ";
    if ( value & AccessFlagBits2::eIndexRead )
      result += "IndexRead | ";
    if ( value & AccessFlagBits2::eVertexAttributeRead )
      result += "VertexAttributeRead | ";
    if ( value & AccessFlagBits2::eUniformRead )
      result += "UniformRead | ";
    if ( value & AccessFlagBits2::eInputAttachmentRead )
      result += "InputAttachmentRead | ";
    if ( value & AccessFlagBits2::eShaderRead )
      result += "ShaderRead | ";
    if ( value & AccessFlagBits2::eShaderWrite )
      result += "ShaderWrite | ";
    if ( value & AccessFlagBits2::eColorAttachmentRead )
      result += "ColorAttachmentRead | ";
    if ( value & AccessFlagBits2::eColorAttachmentWrite )
      result += "ColorAttachmentWrite | ";
    if ( value & AccessFlagBits2::eDepthStencilAttachmentRead )
      result += "DepthStencilAttachmentRead | ";
    if ( value & AccessFlagBits2::eDepthStencilAttachmentWrite )
      result += "DepthStencilAttachmentWrite | ";
    if ( value & AccessFlagBits2::eTransferRead )
      result += "TransferRead | ";
    if ( value & AccessFlagBits2::eTransferWrite )
      result += "TransferWrite | ";
    if ( value & AccessFlagBits2::eHostRead )
      result += "HostRead | ";
    if ( value & AccessFlagBits2::eHostWrite )
      result += "HostWrite | ";
    if ( value & AccessFlagBits2::eMemoryRead )
      result += "MemoryRead | ";
    if ( value & AccessFlagBits2::eMemoryWrite )
      result += "MemoryWrite | ";
    if ( value & AccessFlagBits2::eShaderSampledRead )
      result += "ShaderSampledRead | ";
    if ( value & AccessFlagBits2::eShaderStorageRead )
      result += "ShaderStorageRead | ";
    if ( value & AccessFlagBits2::eShaderStorageWrite )
      result += "ShaderStorageWrite | ";
    if ( value & AccessFlagBits2::eVideoDecodeReadKHR )
      result += "VideoDecodeReadKHR | ";
    if ( value & AccessFlagBits2::eVideoDecodeWriteKHR )
      result += "VideoDecodeWriteKHR | ";
    if ( value & AccessFlagBits2::eVideoEncodeReadKHR )
      result += "VideoEncodeReadKHR | ";
    if ( value & AccessFlagBits2::eVideoEncodeWriteKHR )
      result += "VideoEncodeWriteKHR | ";
    if ( value & AccessFlagBits2::eTransformFeedbackWriteEXT )
      result += "TransformFeedbackWriteEXT | ";
    if ( value & AccessFlagBits2::eTransformFeedbackCounterReadEXT )
      result += "TransformFeedbackCounterReadEXT | ";
    if ( value & AccessFlagBits2::eTransformFeedbackCounterWriteEXT )
      result += "TransformFeedbackCounterWriteEXT | ";
    if ( value & AccessFlagBits2::eConditionalRenderingReadEXT )
      result += "ConditionalRenderingReadEXT | ";
    if ( value & AccessFlagBits2::eCommandPreprocessReadNV )
      result += "CommandPreprocessReadNV | ";
    if ( value & AccessFlagBits2::eCommandPreprocessWriteNV )
      result += "CommandPreprocessWriteNV | ";
    if ( value & AccessFlagBits2::eFragmentShadingRateAttachmentReadKHR )
      result += "FragmentShadingRateAttachmentReadKHR | ";
    if ( value & AccessFlagBits2::eAccelerationStructureReadKHR )
      result += "AccelerationStructureReadKHR | ";
    if ( value & AccessFlagBits2::eAccelerationStructureWriteKHR )
      result += "AccelerationStructureWriteKHR | ";
    if ( value & AccessFlagBits2::eFragmentDensityMapReadEXT )
      result += "FragmentDensityMapReadEXT | ";
    if ( value & AccessFlagBits2::eColorAttachmentReadNoncoherentEXT )
      result += "ColorAttachmentReadNoncoherentEXT | ";
    if ( value & AccessFlagBits2::eDescriptorBufferReadEXT )
      result += "DescriptorBufferReadEXT | ";
    if ( value & AccessFlagBits2::eInvocationMaskReadHUAWEI )
      result += "InvocationMaskReadHUAWEI | ";
    if ( value & AccessFlagBits2::eShaderBindingTableReadKHR )
      result += "ShaderBindingTableReadKHR | ";
    if ( value & AccessFlagBits2::eMicromapReadEXT )
      result += "MicromapReadEXT | ";
    if ( value & AccessFlagBits2::eMicromapWriteEXT )
      result += "MicromapWriteEXT | ";
    if ( value & AccessFlagBits2::eOpticalFlowReadNV )
      result += "OpticalFlowReadNV | ";
    if ( value & AccessFlagBits2::eOpticalFlowWriteNV )
      result += "OpticalFlowWriteNV | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( SubmitFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & SubmitFlagBits::eProtected )
      result += "Protected | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( RenderingFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & RenderingFlagBits::eContentsSecondaryCommandBuffers )
      result += "ContentsSecondaryCommandBuffers | ";
    if ( value & RenderingFlagBits::eSuspending )
      result += "Suspending | ";
    if ( value & RenderingFlagBits::eResuming )
      result += "Resuming | ";
    if ( value & RenderingFlagBits::eContentsInlineEXT )
      result += "ContentsInlineEXT | ";
    if ( value & RenderingFlagBits::eEnableLegacyDitheringEXT )
      result += "EnableLegacyDitheringEXT | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( FormatFeatureFlags2 value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & FormatFeatureFlagBits2::eSampledImage )
      result += "SampledImage | ";
    if ( value & FormatFeatureFlagBits2::eStorageImage )
      result += "StorageImage | ";
    if ( value & FormatFeatureFlagBits2::eStorageImageAtomic )
      result += "StorageImageAtomic | ";
    if ( value & FormatFeatureFlagBits2::eUniformTexelBuffer )
      result += "UniformTexelBuffer | ";
    if ( value & FormatFeatureFlagBits2::eStorageTexelBuffer )
      result += "StorageTexelBuffer | ";
    if ( value & FormatFeatureFlagBits2::eStorageTexelBufferAtomic )
      result += "StorageTexelBufferAtomic | ";
    if ( value & FormatFeatureFlagBits2::eVertexBuffer )
      result += "VertexBuffer | ";
    if ( value & FormatFeatureFlagBits2::eColorAttachment )
      result += "ColorAttachment | ";
    if ( value & FormatFeatureFlagBits2::eColorAttachmentBlend )
      result += "ColorAttachmentBlend | ";
    if ( value & FormatFeatureFlagBits2::eDepthStencilAttachment )
      result += "DepthStencilAttachment | ";
    if ( value & FormatFeatureFlagBits2::eBlitSrc )
      result += "BlitSrc | ";
    if ( value & FormatFeatureFlagBits2::eBlitDst )
      result += "BlitDst | ";
    if ( value & FormatFeatureFlagBits2::eSampledImageFilterLinear )
      result += "SampledImageFilterLinear | ";
    if ( value & FormatFeatureFlagBits2::eSampledImageFilterCubic )
      result += "SampledImageFilterCubic | ";
    if ( value & FormatFeatureFlagBits2::eTransferSrc )
      result += "TransferSrc | ";
    if ( value & FormatFeatureFlagBits2::eTransferDst )
      result += "TransferDst | ";
    if ( value & FormatFeatureFlagBits2::eSampledImageFilterMinmax )
      result += "SampledImageFilterMinmax | ";
    if ( value & FormatFeatureFlagBits2::eMidpointChromaSamples )
      result += "MidpointChromaSamples | ";
    if ( value & FormatFeatureFlagBits2::eSampledImageYcbcrConversionLinearFilter )
      result += "SampledImageYcbcrConversionLinearFilter | ";
    if ( value & FormatFeatureFlagBits2::eSampledImageYcbcrConversionSeparateReconstructionFilter )
      result += "SampledImageYcbcrConversionSeparateReconstructionFilter | ";
    if ( value & FormatFeatureFlagBits2::eSampledImageYcbcrConversionChromaReconstructionExplicit )
      result += "SampledImageYcbcrConversionChromaReconstructionExplicit | ";
    if ( value & FormatFeatureFlagBits2::eSampledImageYcbcrConversionChromaReconstructionExplicitForceable )
      result += "SampledImageYcbcrConversionChromaReconstructionExplicitForceable | ";
    if ( value & FormatFeatureFlagBits2::eDisjoint )
      result += "Disjoint | ";
    if ( value & FormatFeatureFlagBits2::eCositedChromaSamples )
      result += "CositedChromaSamples | ";
    if ( value & FormatFeatureFlagBits2::eStorageReadWithoutFormat )
      result += "StorageReadWithoutFormat | ";
    if ( value & FormatFeatureFlagBits2::eStorageWriteWithoutFormat )
      result += "StorageWriteWithoutFormat | ";
    if ( value & FormatFeatureFlagBits2::eSampledImageDepthComparison )
      result += "SampledImageDepthComparison | ";
    if ( value & FormatFeatureFlagBits2::eVideoDecodeOutputKHR )
      result += "VideoDecodeOutputKHR | ";
    if ( value & FormatFeatureFlagBits2::eVideoDecodeDpbKHR )
      result += "VideoDecodeDpbKHR | ";
    if ( value & FormatFeatureFlagBits2::eAccelerationStructureVertexBufferKHR )
      result += "AccelerationStructureVertexBufferKHR | ";
    if ( value & FormatFeatureFlagBits2::eFragmentDensityMapEXT )
      result += "FragmentDensityMapEXT | ";
    if ( value & FormatFeatureFlagBits2::eFragmentShadingRateAttachmentKHR )
      result += "FragmentShadingRateAttachmentKHR | ";
    if ( value & FormatFeatureFlagBits2::eHostImageTransferEXT )
      result += "HostImageTransferEXT | ";
    if ( value & FormatFeatureFlagBits2::eVideoEncodeInputKHR )
      result += "VideoEncodeInputKHR | ";
    if ( value & FormatFeatureFlagBits2::eVideoEncodeDpbKHR )
      result += "VideoEncodeDpbKHR | ";
    if ( value & FormatFeatureFlagBits2::eLinearColorAttachmentNV )
      result += "LinearColorAttachmentNV | ";
    if ( value & FormatFeatureFlagBits2::eWeightImageQCOM )
      result += "WeightImageQCOM | ";
    if ( value & FormatFeatureFlagBits2::eWeightSampledImageQCOM )
      result += "WeightSampledImageQCOM | ";
    if ( value & FormatFeatureFlagBits2::eBlockMatchingQCOM )
      result += "BlockMatchingQCOM | ";
    if ( value & FormatFeatureFlagBits2::eBoxFilterSampledQCOM )
      result += "BoxFilterSampledQCOM | ";
    if ( value & FormatFeatureFlagBits2::eOpticalFlowImageNV )
      result += "OpticalFlowImageNV | ";
    if ( value & FormatFeatureFlagBits2::eOpticalFlowVectorNV )
      result += "OpticalFlowVectorNV | ";
    if ( value & FormatFeatureFlagBits2::eOpticalFlowCostNV )
      result += "OpticalFlowCostNV | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_KHR_surface ===

  VULKAN_HPP_INLINE std::string to_string( CompositeAlphaFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & CompositeAlphaFlagBitsKHR::eOpaque )
      result += "Opaque | ";
    if ( value & CompositeAlphaFlagBitsKHR::ePreMultiplied )
      result += "PreMultiplied | ";
    if ( value & CompositeAlphaFlagBitsKHR::ePostMultiplied )
      result += "PostMultiplied | ";
    if ( value & CompositeAlphaFlagBitsKHR::eInherit )
      result += "Inherit | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_KHR_swapchain ===

  VULKAN_HPP_INLINE std::string to_string( SwapchainCreateFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & SwapchainCreateFlagBitsKHR::eSplitInstanceBindRegions )
      result += "SplitInstanceBindRegions | ";
    if ( value & SwapchainCreateFlagBitsKHR::eProtected )
      result += "Protected | ";
    if ( value & SwapchainCreateFlagBitsKHR::eMutableFormat )
      result += "MutableFormat | ";
    if ( value & SwapchainCreateFlagBitsKHR::eDeferredMemoryAllocationEXT )
      result += "DeferredMemoryAllocationEXT | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( DeviceGroupPresentModeFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & DeviceGroupPresentModeFlagBitsKHR::eLocal )
      result += "Local | ";
    if ( value & DeviceGroupPresentModeFlagBitsKHR::eRemote )
      result += "Remote | ";
    if ( value & DeviceGroupPresentModeFlagBitsKHR::eSum )
      result += "Sum | ";
    if ( value & DeviceGroupPresentModeFlagBitsKHR::eLocalMultiDevice )
      result += "LocalMultiDevice | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_KHR_display ===

  VULKAN_HPP_INLINE std::string to_string( DisplayModeCreateFlagsKHR )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( DisplayPlaneAlphaFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & DisplayPlaneAlphaFlagBitsKHR::eOpaque )
      result += "Opaque | ";
    if ( value & DisplayPlaneAlphaFlagBitsKHR::eGlobal )
      result += "Global | ";
    if ( value & DisplayPlaneAlphaFlagBitsKHR::ePerPixel )
      result += "PerPixel | ";
    if ( value & DisplayPlaneAlphaFlagBitsKHR::ePerPixelPremultiplied )
      result += "PerPixelPremultiplied | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( DisplaySurfaceCreateFlagsKHR )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( SurfaceTransformFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & SurfaceTransformFlagBitsKHR::eIdentity )
      result += "Identity | ";
    if ( value & SurfaceTransformFlagBitsKHR::eRotate90 )
      result += "Rotate90 | ";
    if ( value & SurfaceTransformFlagBitsKHR::eRotate180 )
      result += "Rotate180 | ";
    if ( value & SurfaceTransformFlagBitsKHR::eRotate270 )
      result += "Rotate270 | ";
    if ( value & SurfaceTransformFlagBitsKHR::eHorizontalMirror )
      result += "HorizontalMirror | ";
    if ( value & SurfaceTransformFlagBitsKHR::eHorizontalMirrorRotate90 )
      result += "HorizontalMirrorRotate90 | ";
    if ( value & SurfaceTransformFlagBitsKHR::eHorizontalMirrorRotate180 )
      result += "HorizontalMirrorRotate180 | ";
    if ( value & SurfaceTransformFlagBitsKHR::eHorizontalMirrorRotate270 )
      result += "HorizontalMirrorRotate270 | ";
    if ( value & SurfaceTransformFlagBitsKHR::eInherit )
      result += "Inherit | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

#if defined( VK_USE_PLATFORM_XLIB_KHR )
  //=== VK_KHR_xlib_surface ===

  VULKAN_HPP_INLINE std::string to_string( XlibSurfaceCreateFlagsKHR )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_XLIB_KHR*/

#if defined( VK_USE_PLATFORM_XCB_KHR )
  //=== VK_KHR_xcb_surface ===

  VULKAN_HPP_INLINE std::string to_string( XcbSurfaceCreateFlagsKHR )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_XCB_KHR*/

#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
  //=== VK_KHR_wayland_surface ===

  VULKAN_HPP_INLINE std::string to_string( WaylandSurfaceCreateFlagsKHR )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  //=== VK_KHR_android_surface ===

  VULKAN_HPP_INLINE std::string to_string( AndroidSurfaceCreateFlagsKHR )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_win32_surface ===

  VULKAN_HPP_INLINE std::string to_string( Win32SurfaceCreateFlagsKHR )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_EXT_debug_report ===

  VULKAN_HPP_INLINE std::string to_string( DebugReportFlagsEXT value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & DebugReportFlagBitsEXT::eInformation )
      result += "Information | ";
    if ( value & DebugReportFlagBitsEXT::eWarning )
      result += "Warning | ";
    if ( value & DebugReportFlagBitsEXT::ePerformanceWarning )
      result += "PerformanceWarning | ";
    if ( value & DebugReportFlagBitsEXT::eError )
      result += "Error | ";
    if ( value & DebugReportFlagBitsEXT::eDebug )
      result += "Debug | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_KHR_video_queue ===

  VULKAN_HPP_INLINE std::string to_string( VideoCodecOperationFlagsKHR value )
  {
    if ( !value )
      return "None";

    std::string result;
    if ( value & VideoCodecOperationFlagBitsKHR::eEncodeH264 )
      result += "EncodeH264 | ";
    if ( value & VideoCodecOperationFlagBitsKHR::eEncodeH265 )
      result += "EncodeH265 | ";
    if ( value & VideoCodecOperationFlagBitsKHR::eDecodeH264 )
      result += "DecodeH264 | ";
    if ( value & VideoCodecOperationFlagBitsKHR::eDecodeH265 )
      result += "DecodeH265 | ";
    if ( value & VideoCodecOperationFlagBitsKHR::eDecodeAv1 )
      result += "DecodeAv1 | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoChromaSubsamplingFlagsKHR value )
  {
    if ( !value )
      return "Invalid";

    std::string result;
    if ( value & VideoChromaSubsamplingFlagBitsKHR::eMonochrome )
      result += "Monochrome | ";
    if ( value & VideoChromaSubsamplingFlagBitsKHR::e420 )
      result += "420 | ";
    if ( value & VideoChromaSubsamplingFlagBitsKHR::e422 )
      result += "422 | ";
    if ( value & VideoChromaSubsamplingFlagBitsKHR::e444 )
      result += "444 | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoComponentBitDepthFlagsKHR value )
  {
    if ( !value )
      return "Invalid";

    std::string result;
    if ( value & VideoComponentBitDepthFlagBitsKHR::e8 )
      result += "8 | ";
    if ( value & VideoComponentBitDepthFlagBitsKHR::e10 )
      result += "10 | ";
    if ( value & VideoComponentBitDepthFlagBitsKHR::e12 )
      result += "12 | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoCapabilityFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoCapabilityFlagBitsKHR::eProtectedContent )
      result += "ProtectedContent | ";
    if ( value & VideoCapabilityFlagBitsKHR::eSeparateReferenceImages )
      result += "SeparateReferenceImages | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoSessionCreateFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoSessionCreateFlagBitsKHR::eProtectedContent )
      result += "ProtectedContent | ";
    if ( value & VideoSessionCreateFlagBitsKHR::eAllowEncodeParameterOptimizations )
      result += "AllowEncodeParameterOptimizations | ";
    if ( value & VideoSessionCreateFlagBitsKHR::eInlineQueries )
      result += "InlineQueries | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoSessionParametersCreateFlagsKHR )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoBeginCodingFlagsKHR )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEndCodingFlagsKHR )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoCodingControlFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoCodingControlFlagBitsKHR::eReset )
      result += "Reset | ";
    if ( value & VideoCodingControlFlagBitsKHR::eEncodeRateControl )
      result += "EncodeRateControl | ";
    if ( value & VideoCodingControlFlagBitsKHR::eEncodeQualityLevel )
      result += "EncodeQualityLevel | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_KHR_video_decode_queue ===

  VULKAN_HPP_INLINE std::string to_string( VideoDecodeCapabilityFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoDecodeCapabilityFlagBitsKHR::eDpbAndOutputCoincide )
      result += "DpbAndOutputCoincide | ";
    if ( value & VideoDecodeCapabilityFlagBitsKHR::eDpbAndOutputDistinct )
      result += "DpbAndOutputDistinct | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoDecodeUsageFlagsKHR value )
  {
    if ( !value )
      return "Default";

    std::string result;
    if ( value & VideoDecodeUsageFlagBitsKHR::eTranscoding )
      result += "Transcoding | ";
    if ( value & VideoDecodeUsageFlagBitsKHR::eOffline )
      result += "Offline | ";
    if ( value & VideoDecodeUsageFlagBitsKHR::eStreaming )
      result += "Streaming | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoDecodeFlagsKHR )
  {
    return "{}";
  }

  //=== VK_EXT_transform_feedback ===

  VULKAN_HPP_INLINE std::string to_string( PipelineRasterizationStateStreamCreateFlagsEXT )
  {
    return "{}";
  }

  //=== VK_KHR_video_encode_h264 ===

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeH264CapabilityFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoEncodeH264CapabilityFlagBitsKHR::eHrdCompliance )
      result += "HrdCompliance | ";
    if ( value & VideoEncodeH264CapabilityFlagBitsKHR::ePredictionWeightTableGenerated )
      result += "PredictionWeightTableGenerated | ";
    if ( value & VideoEncodeH264CapabilityFlagBitsKHR::eRowUnalignedSlice )
      result += "RowUnalignedSlice | ";
    if ( value & VideoEncodeH264CapabilityFlagBitsKHR::eDifferentSliceType )
      result += "DifferentSliceType | ";
    if ( value & VideoEncodeH264CapabilityFlagBitsKHR::eBFrameInL0List )
      result += "BFrameInL0List | ";
    if ( value & VideoEncodeH264CapabilityFlagBitsKHR::eBFrameInL1List )
      result += "BFrameInL1List | ";
    if ( value & VideoEncodeH264CapabilityFlagBitsKHR::ePerPictureTypeMinMaxQp )
      result += "PerPictureTypeMinMaxQp | ";
    if ( value & VideoEncodeH264CapabilityFlagBitsKHR::ePerSliceConstantQp )
      result += "PerSliceConstantQp | ";
    if ( value & VideoEncodeH264CapabilityFlagBitsKHR::eGeneratePrefixNalu )
      result += "GeneratePrefixNalu | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeH264StdFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoEncodeH264StdFlagBitsKHR::eSeparateColorPlaneFlagSet )
      result += "SeparateColorPlaneFlagSet | ";
    if ( value & VideoEncodeH264StdFlagBitsKHR::eQpprimeYZeroTransformBypassFlagSet )
      result += "QpprimeYZeroTransformBypassFlagSet | ";
    if ( value & VideoEncodeH264StdFlagBitsKHR::eScalingMatrixPresentFlagSet )
      result += "ScalingMatrixPresentFlagSet | ";
    if ( value & VideoEncodeH264StdFlagBitsKHR::eChromaQpIndexOffset )
      result += "ChromaQpIndexOffset | ";
    if ( value & VideoEncodeH264StdFlagBitsKHR::eSecondChromaQpIndexOffset )
      result += "SecondChromaQpIndexOffset | ";
    if ( value & VideoEncodeH264StdFlagBitsKHR::ePicInitQpMinus26 )
      result += "PicInitQpMinus26 | ";
    if ( value & VideoEncodeH264StdFlagBitsKHR::eWeightedPredFlagSet )
      result += "WeightedPredFlagSet | ";
    if ( value & VideoEncodeH264StdFlagBitsKHR::eWeightedBipredIdcExplicit )
      result += "WeightedBipredIdcExplicit | ";
    if ( value & VideoEncodeH264StdFlagBitsKHR::eWeightedBipredIdcImplicit )
      result += "WeightedBipredIdcImplicit | ";
    if ( value & VideoEncodeH264StdFlagBitsKHR::eTransform8X8ModeFlagSet )
      result += "Transform8X8ModeFlagSet | ";
    if ( value & VideoEncodeH264StdFlagBitsKHR::eDirectSpatialMvPredFlagUnset )
      result += "DirectSpatialMvPredFlagUnset | ";
    if ( value & VideoEncodeH264StdFlagBitsKHR::eEntropyCodingModeFlagUnset )
      result += "EntropyCodingModeFlagUnset | ";
    if ( value & VideoEncodeH264StdFlagBitsKHR::eEntropyCodingModeFlagSet )
      result += "EntropyCodingModeFlagSet | ";
    if ( value & VideoEncodeH264StdFlagBitsKHR::eDirect8X8InferenceFlagUnset )
      result += "Direct8X8InferenceFlagUnset | ";
    if ( value & VideoEncodeH264StdFlagBitsKHR::eConstrainedIntraPredFlagSet )
      result += "ConstrainedIntraPredFlagSet | ";
    if ( value & VideoEncodeH264StdFlagBitsKHR::eDeblockingFilterDisabled )
      result += "DeblockingFilterDisabled | ";
    if ( value & VideoEncodeH264StdFlagBitsKHR::eDeblockingFilterEnabled )
      result += "DeblockingFilterEnabled | ";
    if ( value & VideoEncodeH264StdFlagBitsKHR::eDeblockingFilterPartial )
      result += "DeblockingFilterPartial | ";
    if ( value & VideoEncodeH264StdFlagBitsKHR::eSliceQpDelta )
      result += "SliceQpDelta | ";
    if ( value & VideoEncodeH264StdFlagBitsKHR::eDifferentSliceQpDelta )
      result += "DifferentSliceQpDelta | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeH264RateControlFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoEncodeH264RateControlFlagBitsKHR::eAttemptHrdCompliance )
      result += "AttemptHrdCompliance | ";
    if ( value & VideoEncodeH264RateControlFlagBitsKHR::eRegularGop )
      result += "RegularGop | ";
    if ( value & VideoEncodeH264RateControlFlagBitsKHR::eReferencePatternFlat )
      result += "ReferencePatternFlat | ";
    if ( value & VideoEncodeH264RateControlFlagBitsKHR::eReferencePatternDyadic )
      result += "ReferencePatternDyadic | ";
    if ( value & VideoEncodeH264RateControlFlagBitsKHR::eTemporalLayerPatternDyadic )
      result += "TemporalLayerPatternDyadic | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_KHR_video_encode_h265 ===

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeH265CapabilityFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoEncodeH265CapabilityFlagBitsKHR::eHrdCompliance )
      result += "HrdCompliance | ";
    if ( value & VideoEncodeH265CapabilityFlagBitsKHR::ePredictionWeightTableGenerated )
      result += "PredictionWeightTableGenerated | ";
    if ( value & VideoEncodeH265CapabilityFlagBitsKHR::eRowUnalignedSliceSegment )
      result += "RowUnalignedSliceSegment | ";
    if ( value & VideoEncodeH265CapabilityFlagBitsKHR::eDifferentSliceSegmentType )
      result += "DifferentSliceSegmentType | ";
    if ( value & VideoEncodeH265CapabilityFlagBitsKHR::eBFrameInL0List )
      result += "BFrameInL0List | ";
    if ( value & VideoEncodeH265CapabilityFlagBitsKHR::eBFrameInL1List )
      result += "BFrameInL1List | ";
    if ( value & VideoEncodeH265CapabilityFlagBitsKHR::ePerPictureTypeMinMaxQp )
      result += "PerPictureTypeMinMaxQp | ";
    if ( value & VideoEncodeH265CapabilityFlagBitsKHR::ePerSliceSegmentConstantQp )
      result += "PerSliceSegmentConstantQp | ";
    if ( value & VideoEncodeH265CapabilityFlagBitsKHR::eMultipleTilesPerSliceSegment )
      result += "MultipleTilesPerSliceSegment | ";
    if ( value & VideoEncodeH265CapabilityFlagBitsKHR::eMultipleSliceSegmentsPerTile )
      result += "MultipleSliceSegmentsPerTile | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeH265StdFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoEncodeH265StdFlagBitsKHR::eSeparateColorPlaneFlagSet )
      result += "SeparateColorPlaneFlagSet | ";
    if ( value & VideoEncodeH265StdFlagBitsKHR::eSampleAdaptiveOffsetEnabledFlagSet )
      result += "SampleAdaptiveOffsetEnabledFlagSet | ";
    if ( value & VideoEncodeH265StdFlagBitsKHR::eScalingListDataPresentFlagSet )
      result += "ScalingListDataPresentFlagSet | ";
    if ( value & VideoEncodeH265StdFlagBitsKHR::ePcmEnabledFlagSet )
      result += "PcmEnabledFlagSet | ";
    if ( value & VideoEncodeH265StdFlagBitsKHR::eSpsTemporalMvpEnabledFlagSet )
      result += "SpsTemporalMvpEnabledFlagSet | ";
    if ( value & VideoEncodeH265StdFlagBitsKHR::eInitQpMinus26 )
      result += "InitQpMinus26 | ";
    if ( value & VideoEncodeH265StdFlagBitsKHR::eWeightedPredFlagSet )
      result += "WeightedPredFlagSet | ";
    if ( value & VideoEncodeH265StdFlagBitsKHR::eWeightedBipredFlagSet )
      result += "WeightedBipredFlagSet | ";
    if ( value & VideoEncodeH265StdFlagBitsKHR::eLog2ParallelMergeLevelMinus2 )
      result += "Log2ParallelMergeLevelMinus2 | ";
    if ( value & VideoEncodeH265StdFlagBitsKHR::eSignDataHidingEnabledFlagSet )
      result += "SignDataHidingEnabledFlagSet | ";
    if ( value & VideoEncodeH265StdFlagBitsKHR::eTransformSkipEnabledFlagSet )
      result += "TransformSkipEnabledFlagSet | ";
    if ( value & VideoEncodeH265StdFlagBitsKHR::eTransformSkipEnabledFlagUnset )
      result += "TransformSkipEnabledFlagUnset | ";
    if ( value & VideoEncodeH265StdFlagBitsKHR::ePpsSliceChromaQpOffsetsPresentFlagSet )
      result += "PpsSliceChromaQpOffsetsPresentFlagSet | ";
    if ( value & VideoEncodeH265StdFlagBitsKHR::eTransquantBypassEnabledFlagSet )
      result += "TransquantBypassEnabledFlagSet | ";
    if ( value & VideoEncodeH265StdFlagBitsKHR::eConstrainedIntraPredFlagSet )
      result += "ConstrainedIntraPredFlagSet | ";
    if ( value & VideoEncodeH265StdFlagBitsKHR::eEntropyCodingSyncEnabledFlagSet )
      result += "EntropyCodingSyncEnabledFlagSet | ";
    if ( value & VideoEncodeH265StdFlagBitsKHR::eDeblockingFilterOverrideEnabledFlagSet )
      result += "DeblockingFilterOverrideEnabledFlagSet | ";
    if ( value & VideoEncodeH265StdFlagBitsKHR::eDependentSliceSegmentsEnabledFlagSet )
      result += "DependentSliceSegmentsEnabledFlagSet | ";
    if ( value & VideoEncodeH265StdFlagBitsKHR::eDependentSliceSegmentFlagSet )
      result += "DependentSliceSegmentFlagSet | ";
    if ( value & VideoEncodeH265StdFlagBitsKHR::eSliceQpDelta )
      result += "SliceQpDelta | ";
    if ( value & VideoEncodeH265StdFlagBitsKHR::eDifferentSliceQpDelta )
      result += "DifferentSliceQpDelta | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeH265CtbSizeFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoEncodeH265CtbSizeFlagBitsKHR::e16 )
      result += "16 | ";
    if ( value & VideoEncodeH265CtbSizeFlagBitsKHR::e32 )
      result += "32 | ";
    if ( value & VideoEncodeH265CtbSizeFlagBitsKHR::e64 )
      result += "64 | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeH265TransformBlockSizeFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoEncodeH265TransformBlockSizeFlagBitsKHR::e4 )
      result += "4 | ";
    if ( value & VideoEncodeH265TransformBlockSizeFlagBitsKHR::e8 )
      result += "8 | ";
    if ( value & VideoEncodeH265TransformBlockSizeFlagBitsKHR::e16 )
      result += "16 | ";
    if ( value & VideoEncodeH265TransformBlockSizeFlagBitsKHR::e32 )
      result += "32 | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeH265RateControlFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoEncodeH265RateControlFlagBitsKHR::eAttemptHrdCompliance )
      result += "AttemptHrdCompliance | ";
    if ( value & VideoEncodeH265RateControlFlagBitsKHR::eRegularGop )
      result += "RegularGop | ";
    if ( value & VideoEncodeH265RateControlFlagBitsKHR::eReferencePatternFlat )
      result += "ReferencePatternFlat | ";
    if ( value & VideoEncodeH265RateControlFlagBitsKHR::eReferencePatternDyadic )
      result += "ReferencePatternDyadic | ";
    if ( value & VideoEncodeH265RateControlFlagBitsKHR::eTemporalSubLayerPatternDyadic )
      result += "TemporalSubLayerPatternDyadic | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_KHR_video_decode_h264 ===

  VULKAN_HPP_INLINE std::string to_string( VideoDecodeH264PictureLayoutFlagsKHR value )
  {
    if ( !value )
      return "Progressive";

    std::string result;
    if ( value & VideoDecodeH264PictureLayoutFlagBitsKHR::eInterlacedInterleavedLines )
      result += "InterlacedInterleavedLines | ";
    if ( value & VideoDecodeH264PictureLayoutFlagBitsKHR::eInterlacedSeparatePlanes )
      result += "InterlacedSeparatePlanes | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

#if defined( VK_USE_PLATFORM_GGP )
  //=== VK_GGP_stream_descriptor_surface ===

  VULKAN_HPP_INLINE std::string to_string( StreamDescriptorSurfaceCreateFlagsGGP )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_GGP*/

  //=== VK_NV_external_memory_capabilities ===

  VULKAN_HPP_INLINE std::string to_string( ExternalMemoryHandleTypeFlagsNV value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & ExternalMemoryHandleTypeFlagBitsNV::eOpaqueWin32 )
      result += "OpaqueWin32 | ";
    if ( value & ExternalMemoryHandleTypeFlagBitsNV::eOpaqueWin32Kmt )
      result += "OpaqueWin32Kmt | ";
    if ( value & ExternalMemoryHandleTypeFlagBitsNV::eD3D11Image )
      result += "D3D11Image | ";
    if ( value & ExternalMemoryHandleTypeFlagBitsNV::eD3D11ImageKmt )
      result += "D3D11ImageKmt | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( ExternalMemoryFeatureFlagsNV value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & ExternalMemoryFeatureFlagBitsNV::eDedicatedOnly )
      result += "DedicatedOnly | ";
    if ( value & ExternalMemoryFeatureFlagBitsNV::eExportable )
      result += "Exportable | ";
    if ( value & ExternalMemoryFeatureFlagBitsNV::eImportable )
      result += "Importable | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

#if defined( VK_USE_PLATFORM_VI_NN )
  //=== VK_NN_vi_surface ===

  VULKAN_HPP_INLINE std::string to_string( ViSurfaceCreateFlagsNN )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_VI_NN*/

  //=== VK_EXT_conditional_rendering ===

  VULKAN_HPP_INLINE std::string to_string( ConditionalRenderingFlagsEXT value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & ConditionalRenderingFlagBitsEXT::eInverted )
      result += "Inverted | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_EXT_display_surface_counter ===

  VULKAN_HPP_INLINE std::string to_string( SurfaceCounterFlagsEXT value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & SurfaceCounterFlagBitsEXT::eVblank )
      result += "Vblank | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_NV_viewport_swizzle ===

  VULKAN_HPP_INLINE std::string to_string( PipelineViewportSwizzleStateCreateFlagsNV )
  {
    return "{}";
  }

  //=== VK_EXT_discard_rectangles ===

  VULKAN_HPP_INLINE std::string to_string( PipelineDiscardRectangleStateCreateFlagsEXT )
  {
    return "{}";
  }

  //=== VK_EXT_conservative_rasterization ===

  VULKAN_HPP_INLINE std::string to_string( PipelineRasterizationConservativeStateCreateFlagsEXT )
  {
    return "{}";
  }

  //=== VK_EXT_depth_clip_enable ===

  VULKAN_HPP_INLINE std::string to_string( PipelineRasterizationDepthClipStateCreateFlagsEXT )
  {
    return "{}";
  }

  //=== VK_KHR_performance_query ===

  VULKAN_HPP_INLINE std::string to_string( PerformanceCounterDescriptionFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & PerformanceCounterDescriptionFlagBitsKHR::ePerformanceImpacting )
      result += "PerformanceImpacting | ";
    if ( value & PerformanceCounterDescriptionFlagBitsKHR::eConcurrentlyImpacted )
      result += "ConcurrentlyImpacted | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( AcquireProfilingLockFlagsKHR )
  {
    return "{}";
  }

#if defined( VK_USE_PLATFORM_IOS_MVK )
  //=== VK_MVK_ios_surface ===

  VULKAN_HPP_INLINE std::string to_string( IOSSurfaceCreateFlagsMVK )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_IOS_MVK*/

#if defined( VK_USE_PLATFORM_MACOS_MVK )
  //=== VK_MVK_macos_surface ===

  VULKAN_HPP_INLINE std::string to_string( MacOSSurfaceCreateFlagsMVK )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_MACOS_MVK*/

  //=== VK_EXT_debug_utils ===

  VULKAN_HPP_INLINE std::string to_string( DebugUtilsMessageSeverityFlagsEXT value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & DebugUtilsMessageSeverityFlagBitsEXT::eVerbose )
      result += "Verbose | ";
    if ( value & DebugUtilsMessageSeverityFlagBitsEXT::eInfo )
      result += "Info | ";
    if ( value & DebugUtilsMessageSeverityFlagBitsEXT::eWarning )
      result += "Warning | ";
    if ( value & DebugUtilsMessageSeverityFlagBitsEXT::eError )
      result += "Error | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( DebugUtilsMessageTypeFlagsEXT value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & DebugUtilsMessageTypeFlagBitsEXT::eGeneral )
      result += "General | ";
    if ( value & DebugUtilsMessageTypeFlagBitsEXT::eValidation )
      result += "Validation | ";
    if ( value & DebugUtilsMessageTypeFlagBitsEXT::ePerformance )
      result += "Performance | ";
    if ( value & DebugUtilsMessageTypeFlagBitsEXT::eDeviceAddressBinding )
      result += "DeviceAddressBinding | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( DebugUtilsMessengerCallbackDataFlagsEXT )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( DebugUtilsMessengerCreateFlagsEXT )
  {
    return "{}";
  }

  //=== VK_NV_fragment_coverage_to_color ===

  VULKAN_HPP_INLINE std::string to_string( PipelineCoverageToColorStateCreateFlagsNV )
  {
    return "{}";
  }

  //=== VK_KHR_acceleration_structure ===

  VULKAN_HPP_INLINE std::string to_string( GeometryFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & GeometryFlagBitsKHR::eOpaque )
      result += "Opaque | ";
    if ( value & GeometryFlagBitsKHR::eNoDuplicateAnyHitInvocation )
      result += "NoDuplicateAnyHitInvocation | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( GeometryInstanceFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable )
      result += "TriangleFacingCullDisable | ";
    if ( value & GeometryInstanceFlagBitsKHR::eTriangleFlipFacing )
      result += "TriangleFlipFacing | ";
    if ( value & GeometryInstanceFlagBitsKHR::eForceOpaque )
      result += "ForceOpaque | ";
    if ( value & GeometryInstanceFlagBitsKHR::eForceNoOpaque )
      result += "ForceNoOpaque | ";
    if ( value & GeometryInstanceFlagBitsKHR::eForceOpacityMicromap2StateEXT )
      result += "ForceOpacityMicromap2StateEXT | ";
    if ( value & GeometryInstanceFlagBitsKHR::eDisableOpacityMicromapsEXT )
      result += "DisableOpacityMicromapsEXT | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( BuildAccelerationStructureFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & BuildAccelerationStructureFlagBitsKHR::eAllowUpdate )
      result += "AllowUpdate | ";
    if ( value & BuildAccelerationStructureFlagBitsKHR::eAllowCompaction )
      result += "AllowCompaction | ";
    if ( value & BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace )
      result += "PreferFastTrace | ";
    if ( value & BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild )
      result += "PreferFastBuild | ";
    if ( value & BuildAccelerationStructureFlagBitsKHR::eLowMemory )
      result += "LowMemory | ";
    if ( value & BuildAccelerationStructureFlagBitsKHR::eMotionNV )
      result += "MotionNV | ";
    if ( value & BuildAccelerationStructureFlagBitsKHR::eAllowOpacityMicromapUpdateEXT )
      result += "AllowOpacityMicromapUpdateEXT | ";
    if ( value & BuildAccelerationStructureFlagBitsKHR::eAllowDisableOpacityMicromapsEXT )
      result += "AllowDisableOpacityMicromapsEXT | ";
    if ( value & BuildAccelerationStructureFlagBitsKHR::eAllowOpacityMicromapDataUpdateEXT )
      result += "AllowOpacityMicromapDataUpdateEXT | ";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & BuildAccelerationStructureFlagBitsKHR::eAllowDisplacementMicromapUpdateNV )
      result += "AllowDisplacementMicromapUpdateNV | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    if ( value & BuildAccelerationStructureFlagBitsKHR::eAllowDataAccess )
      result += "AllowDataAccess | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( AccelerationStructureCreateFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & AccelerationStructureCreateFlagBitsKHR::eDeviceAddressCaptureReplay )
      result += "DeviceAddressCaptureReplay | ";
    if ( value & AccelerationStructureCreateFlagBitsKHR::eDescriptorBufferCaptureReplayEXT )
      result += "DescriptorBufferCaptureReplayEXT | ";
    if ( value & AccelerationStructureCreateFlagBitsKHR::eMotionNV )
      result += "MotionNV | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_NV_framebuffer_mixed_samples ===

  VULKAN_HPP_INLINE std::string to_string( PipelineCoverageModulationStateCreateFlagsNV )
  {
    return "{}";
  }

  //=== VK_EXT_validation_cache ===

  VULKAN_HPP_INLINE std::string to_string( ValidationCacheCreateFlagsEXT )
  {
    return "{}";
  }

  //=== VK_AMD_pipeline_compiler_control ===

  VULKAN_HPP_INLINE std::string to_string( PipelineCompilerControlFlagsAMD )
  {
    return "{}";
  }

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_imagepipe_surface ===

  VULKAN_HPP_INLINE std::string to_string( ImagePipeSurfaceCreateFlagsFUCHSIA )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_surface ===

  VULKAN_HPP_INLINE std::string to_string( MetalSurfaceCreateFlagsEXT )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_AMD_shader_core_properties2 ===

  VULKAN_HPP_INLINE std::string to_string( ShaderCorePropertiesFlagsAMD )
  {
    return "{}";
  }

  //=== VK_NV_coverage_reduction_mode ===

  VULKAN_HPP_INLINE std::string to_string( PipelineCoverageReductionStateCreateFlagsNV )
  {
    return "{}";
  }

  //=== VK_EXT_headless_surface ===

  VULKAN_HPP_INLINE std::string to_string( HeadlessSurfaceCreateFlagsEXT )
  {
    return "{}";
  }

  //=== VK_EXT_host_image_copy ===

  VULKAN_HPP_INLINE std::string to_string( HostImageCopyFlagsEXT value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & HostImageCopyFlagBitsEXT::eMemcpy )
      result += "Memcpy | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_KHR_map_memory2 ===

  VULKAN_HPP_INLINE std::string to_string( MemoryUnmapFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & MemoryUnmapFlagBitsKHR::eReserveEXT )
      result += "ReserveEXT | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_EXT_surface_maintenance1 ===

  VULKAN_HPP_INLINE std::string to_string( PresentScalingFlagsEXT value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & PresentScalingFlagBitsEXT::eOneToOne )
      result += "OneToOne | ";
    if ( value & PresentScalingFlagBitsEXT::eAspectRatioStretch )
      result += "AspectRatioStretch | ";
    if ( value & PresentScalingFlagBitsEXT::eStretch )
      result += "Stretch | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( PresentGravityFlagsEXT value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & PresentGravityFlagBitsEXT::eMin )
      result += "Min | ";
    if ( value & PresentGravityFlagBitsEXT::eMax )
      result += "Max | ";
    if ( value & PresentGravityFlagBitsEXT::eCentered )
      result += "Centered | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_NV_device_generated_commands ===

  VULKAN_HPP_INLINE std::string to_string( IndirectStateFlagsNV value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & IndirectStateFlagBitsNV::eFlagFrontface )
      result += "FlagFrontface | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( IndirectCommandsLayoutUsageFlagsNV value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & IndirectCommandsLayoutUsageFlagBitsNV::eExplicitPreprocess )
      result += "ExplicitPreprocess | ";
    if ( value & IndirectCommandsLayoutUsageFlagBitsNV::eIndexedSequences )
      result += "IndexedSequences | ";
    if ( value & IndirectCommandsLayoutUsageFlagBitsNV::eUnorderedSequences )
      result += "UnorderedSequences | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_EXT_device_memory_report ===

  VULKAN_HPP_INLINE std::string to_string( DeviceMemoryReportFlagsEXT )
  {
    return "{}";
  }

  //=== VK_KHR_video_encode_queue ===

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeFlagsKHR )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeCapabilityFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoEncodeCapabilityFlagBitsKHR::ePrecedingExternallyEncodedBytes )
      result += "PrecedingExternallyEncodedBytes | ";
    if ( value & VideoEncodeCapabilityFlagBitsKHR::eInsufficientBitstreamBufferRangeDetection )
      result += "InsufficientBitstreamBufferRangeDetection | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeFeedbackFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoEncodeFeedbackFlagBitsKHR::eBitstreamBufferOffset )
      result += "BitstreamBufferOffset | ";
    if ( value & VideoEncodeFeedbackFlagBitsKHR::eBitstreamBytesWritten )
      result += "BitstreamBytesWritten | ";
    if ( value & VideoEncodeFeedbackFlagBitsKHR::eBitstreamHasOverrides )
      result += "BitstreamHasOverrides | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeUsageFlagsKHR value )
  {
    if ( !value )
      return "Default";

    std::string result;
    if ( value & VideoEncodeUsageFlagBitsKHR::eTranscoding )
      result += "Transcoding | ";
    if ( value & VideoEncodeUsageFlagBitsKHR::eStreaming )
      result += "Streaming | ";
    if ( value & VideoEncodeUsageFlagBitsKHR::eRecording )
      result += "Recording | ";
    if ( value & VideoEncodeUsageFlagBitsKHR::eConferencing )
      result += "Conferencing | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeContentFlagsKHR value )
  {
    if ( !value )
      return "Default";

    std::string result;
    if ( value & VideoEncodeContentFlagBitsKHR::eCamera )
      result += "Camera | ";
    if ( value & VideoEncodeContentFlagBitsKHR::eDesktop )
      result += "Desktop | ";
    if ( value & VideoEncodeContentFlagBitsKHR::eRendered )
      result += "Rendered | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeRateControlFlagsKHR )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeRateControlModeFlagsKHR value )
  {
    if ( !value )
      return "Default";

    std::string result;
    if ( value & VideoEncodeRateControlModeFlagBitsKHR::eDisabled )
      result += "Disabled | ";
    if ( value & VideoEncodeRateControlModeFlagBitsKHR::eCbr )
      result += "Cbr | ";
    if ( value & VideoEncodeRateControlModeFlagBitsKHR::eVbr )
      result += "Vbr | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_NV_device_diagnostics_config ===

  VULKAN_HPP_INLINE std::string to_string( DeviceDiagnosticsConfigFlagsNV value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & DeviceDiagnosticsConfigFlagBitsNV::eEnableShaderDebugInfo )
      result += "EnableShaderDebugInfo | ";
    if ( value & DeviceDiagnosticsConfigFlagBitsNV::eEnableResourceTracking )
      result += "EnableResourceTracking | ";
    if ( value & DeviceDiagnosticsConfigFlagBitsNV::eEnableAutomaticCheckpoints )
      result += "EnableAutomaticCheckpoints | ";
    if ( value & DeviceDiagnosticsConfigFlagBitsNV::eEnableShaderErrorReporting )
      result += "EnableShaderErrorReporting | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_objects ===

  VULKAN_HPP_INLINE std::string to_string( ExportMetalObjectTypeFlagsEXT value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & ExportMetalObjectTypeFlagBitsEXT::eMetalDevice )
      result += "MetalDevice | ";
    if ( value & ExportMetalObjectTypeFlagBitsEXT::eMetalCommandQueue )
      result += "MetalCommandQueue | ";
    if ( value & ExportMetalObjectTypeFlagBitsEXT::eMetalBuffer )
      result += "MetalBuffer | ";
    if ( value & ExportMetalObjectTypeFlagBitsEXT::eMetalTexture )
      result += "MetalTexture | ";
    if ( value & ExportMetalObjectTypeFlagBitsEXT::eMetalIosurface )
      result += "MetalIosurface | ";
    if ( value & ExportMetalObjectTypeFlagBitsEXT::eMetalSharedEvent )
      result += "MetalSharedEvent | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_EXT_graphics_pipeline_library ===

  VULKAN_HPP_INLINE std::string to_string( GraphicsPipelineLibraryFlagsEXT value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & GraphicsPipelineLibraryFlagBitsEXT::eVertexInputInterface )
      result += "VertexInputInterface | ";
    if ( value & GraphicsPipelineLibraryFlagBitsEXT::ePreRasterizationShaders )
      result += "PreRasterizationShaders | ";
    if ( value & GraphicsPipelineLibraryFlagBitsEXT::eFragmentShader )
      result += "FragmentShader | ";
    if ( value & GraphicsPipelineLibraryFlagBitsEXT::eFragmentOutputInterface )
      result += "FragmentOutputInterface | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_NV_ray_tracing_motion_blur ===

  VULKAN_HPP_INLINE std::string to_string( AccelerationStructureMotionInfoFlagsNV )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( AccelerationStructureMotionInstanceFlagsNV )
  {
    return "{}";
  }

  //=== VK_EXT_image_compression_control ===

  VULKAN_HPP_INLINE std::string to_string( ImageCompressionFlagsEXT value )
  {
    if ( !value )
      return "Default";

    std::string result;
    if ( value & ImageCompressionFlagBitsEXT::eFixedRateDefault )
      result += "FixedRateDefault | ";
    if ( value & ImageCompressionFlagBitsEXT::eFixedRateExplicit )
      result += "FixedRateExplicit | ";
    if ( value & ImageCompressionFlagBitsEXT::eDisabled )
      result += "Disabled | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( ImageCompressionFixedRateFlagsEXT value )
  {
    if ( !value )
      return "None";

    std::string result;
    if ( value & ImageCompressionFixedRateFlagBitsEXT::e1Bpc )
      result += "1Bpc | ";
    if ( value & ImageCompressionFixedRateFlagBitsEXT::e2Bpc )
      result += "2Bpc | ";
    if ( value & ImageCompressionFixedRateFlagBitsEXT::e3Bpc )
      result += "3Bpc | ";
    if ( value & ImageCompressionFixedRateFlagBitsEXT::e4Bpc )
      result += "4Bpc | ";
    if ( value & ImageCompressionFixedRateFlagBitsEXT::e5Bpc )
      result += "5Bpc | ";
    if ( value & ImageCompressionFixedRateFlagBitsEXT::e6Bpc )
      result += "6Bpc | ";
    if ( value & ImageCompressionFixedRateFlagBitsEXT::e7Bpc )
      result += "7Bpc | ";
    if ( value & ImageCompressionFixedRateFlagBitsEXT::e8Bpc )
      result += "8Bpc | ";
    if ( value & ImageCompressionFixedRateFlagBitsEXT::e9Bpc )
      result += "9Bpc | ";
    if ( value & ImageCompressionFixedRateFlagBitsEXT::e10Bpc )
      result += "10Bpc | ";
    if ( value & ImageCompressionFixedRateFlagBitsEXT::e11Bpc )
      result += "11Bpc | ";
    if ( value & ImageCompressionFixedRateFlagBitsEXT::e12Bpc )
      result += "12Bpc | ";
    if ( value & ImageCompressionFixedRateFlagBitsEXT::e13Bpc )
      result += "13Bpc | ";
    if ( value & ImageCompressionFixedRateFlagBitsEXT::e14Bpc )
      result += "14Bpc | ";
    if ( value & ImageCompressionFixedRateFlagBitsEXT::e15Bpc )
      result += "15Bpc | ";
    if ( value & ImageCompressionFixedRateFlagBitsEXT::e16Bpc )
      result += "16Bpc | ";
    if ( value & ImageCompressionFixedRateFlagBitsEXT::e17Bpc )
      result += "17Bpc | ";
    if ( value & ImageCompressionFixedRateFlagBitsEXT::e18Bpc )
      result += "18Bpc | ";
    if ( value & ImageCompressionFixedRateFlagBitsEXT::e19Bpc )
      result += "19Bpc | ";
    if ( value & ImageCompressionFixedRateFlagBitsEXT::e20Bpc )
      result += "20Bpc | ";
    if ( value & ImageCompressionFixedRateFlagBitsEXT::e21Bpc )
      result += "21Bpc | ";
    if ( value & ImageCompressionFixedRateFlagBitsEXT::e22Bpc )
      result += "22Bpc | ";
    if ( value & ImageCompressionFixedRateFlagBitsEXT::e23Bpc )
      result += "23Bpc | ";
    if ( value & ImageCompressionFixedRateFlagBitsEXT::e24Bpc )
      result += "24Bpc | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
  //=== VK_EXT_directfb_surface ===

  VULKAN_HPP_INLINE std::string to_string( DirectFBSurfaceCreateFlagsEXT )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

  //=== VK_EXT_device_address_binding_report ===

  VULKAN_HPP_INLINE std::string to_string( DeviceAddressBindingFlagsEXT value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & DeviceAddressBindingFlagBitsEXT::eInternalObject )
      result += "InternalObject | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_buffer_collection ===

  VULKAN_HPP_INLINE std::string to_string( ImageFormatConstraintsFlagsFUCHSIA )
  {
    return "{}";
  }

  VULKAN_HPP_INLINE std::string to_string( ImageConstraintsInfoFlagsFUCHSIA value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & ImageConstraintsInfoFlagBitsFUCHSIA::eCpuReadRarely )
      result += "CpuReadRarely | ";
    if ( value & ImageConstraintsInfoFlagBitsFUCHSIA::eCpuReadOften )
      result += "CpuReadOften | ";
    if ( value & ImageConstraintsInfoFlagBitsFUCHSIA::eCpuWriteRarely )
      result += "CpuWriteRarely | ";
    if ( value & ImageConstraintsInfoFlagBitsFUCHSIA::eCpuWriteOften )
      result += "CpuWriteOften | ";
    if ( value & ImageConstraintsInfoFlagBitsFUCHSIA::eProtectedOptional )
      result += "ProtectedOptional | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }
#endif /*VK_USE_PLATFORM_FUCHSIA*/

  //=== VK_EXT_frame_boundary ===

  VULKAN_HPP_INLINE std::string to_string( FrameBoundaryFlagsEXT value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & FrameBoundaryFlagBitsEXT::eFrameEnd )
      result += "FrameEnd | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
  //=== VK_QNX_screen_surface ===

  VULKAN_HPP_INLINE std::string to_string( ScreenSurfaceCreateFlagsQNX )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

  //=== VK_EXT_opacity_micromap ===

  VULKAN_HPP_INLINE std::string to_string( BuildMicromapFlagsEXT value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & BuildMicromapFlagBitsEXT::ePreferFastTrace )
      result += "PreferFastTrace | ";
    if ( value & BuildMicromapFlagBitsEXT::ePreferFastBuild )
      result += "PreferFastBuild | ";
    if ( value & BuildMicromapFlagBitsEXT::eAllowCompaction )
      result += "AllowCompaction | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( MicromapCreateFlagsEXT value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & MicromapCreateFlagBitsEXT::eDeviceAddressCaptureReplay )
      result += "DeviceAddressCaptureReplay | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_ARM_scheduling_controls ===

  VULKAN_HPP_INLINE std::string to_string( PhysicalDeviceSchedulingControlsFlagsARM value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & PhysicalDeviceSchedulingControlsFlagBitsARM::eShaderCoreCount )
      result += "ShaderCoreCount | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_NV_memory_decompression ===

  VULKAN_HPP_INLINE std::string to_string( MemoryDecompressionMethodFlagsNV value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & MemoryDecompressionMethodFlagBitsNV::eGdeflate10 )
      result += "Gdeflate10 | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_LUNARG_direct_driver_loading ===

  VULKAN_HPP_INLINE std::string to_string( DirectDriverLoadingFlagsLUNARG )
  {
    return "{}";
  }

  //=== VK_NV_optical_flow ===

  VULKAN_HPP_INLINE std::string to_string( OpticalFlowUsageFlagsNV value )
  {
    if ( !value )
      return "Unknown";

    std::string result;
    if ( value & OpticalFlowUsageFlagBitsNV::eInput )
      result += "Input | ";
    if ( value & OpticalFlowUsageFlagBitsNV::eOutput )
      result += "Output | ";
    if ( value & OpticalFlowUsageFlagBitsNV::eHint )
      result += "Hint | ";
    if ( value & OpticalFlowUsageFlagBitsNV::eCost )
      result += "Cost | ";
    if ( value & OpticalFlowUsageFlagBitsNV::eGlobalFlow )
      result += "GlobalFlow | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( OpticalFlowGridSizeFlagsNV value )
  {
    if ( !value )
      return "Unknown";

    std::string result;
    if ( value & OpticalFlowGridSizeFlagBitsNV::e1X1 )
      result += "1X1 | ";
    if ( value & OpticalFlowGridSizeFlagBitsNV::e2X2 )
      result += "2X2 | ";
    if ( value & OpticalFlowGridSizeFlagBitsNV::e4X4 )
      result += "4X4 | ";
    if ( value & OpticalFlowGridSizeFlagBitsNV::e8X8 )
      result += "8X8 | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( OpticalFlowSessionCreateFlagsNV value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & OpticalFlowSessionCreateFlagBitsNV::eEnableHint )
      result += "EnableHint | ";
    if ( value & OpticalFlowSessionCreateFlagBitsNV::eEnableCost )
      result += "EnableCost | ";
    if ( value & OpticalFlowSessionCreateFlagBitsNV::eEnableGlobalFlow )
      result += "EnableGlobalFlow | ";
    if ( value & OpticalFlowSessionCreateFlagBitsNV::eAllowRegions )
      result += "AllowRegions | ";
    if ( value & OpticalFlowSessionCreateFlagBitsNV::eBothDirections )
      result += "BothDirections | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( OpticalFlowExecuteFlagsNV value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & OpticalFlowExecuteFlagBitsNV::eDisableTemporalHints )
      result += "DisableTemporalHints | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_KHR_maintenance5 ===

  VULKAN_HPP_INLINE std::string to_string( PipelineCreateFlags2KHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & PipelineCreateFlagBits2KHR::eDisableOptimization )
      result += "DisableOptimization | ";
    if ( value & PipelineCreateFlagBits2KHR::eAllowDerivatives )
      result += "AllowDerivatives | ";
    if ( value & PipelineCreateFlagBits2KHR::eDerivative )
      result += "Derivative | ";
    if ( value & PipelineCreateFlagBits2KHR::eEnableLegacyDitheringEXT )
      result += "EnableLegacyDitheringEXT | ";
    if ( value & PipelineCreateFlagBits2KHR::eViewIndexFromDeviceIndex )
      result += "ViewIndexFromDeviceIndex | ";
    if ( value & PipelineCreateFlagBits2KHR::eDispatchBase )
      result += "DispatchBase | ";
    if ( value & PipelineCreateFlagBits2KHR::eDeferCompileNV )
      result += "DeferCompileNV | ";
    if ( value & PipelineCreateFlagBits2KHR::eCaptureStatistics )
      result += "CaptureStatistics | ";
    if ( value & PipelineCreateFlagBits2KHR::eCaptureInternalRepresentations )
      result += "CaptureInternalRepresentations | ";
    if ( value & PipelineCreateFlagBits2KHR::eFailOnPipelineCompileRequired )
      result += "FailOnPipelineCompileRequired | ";
    if ( value & PipelineCreateFlagBits2KHR::eEarlyReturnOnFailure )
      result += "EarlyReturnOnFailure | ";
    if ( value & PipelineCreateFlagBits2KHR::eLinkTimeOptimizationEXT )
      result += "LinkTimeOptimizationEXT | ";
    if ( value & PipelineCreateFlagBits2KHR::eRetainLinkTimeOptimizationInfoEXT )
      result += "RetainLinkTimeOptimizationInfoEXT | ";
    if ( value & PipelineCreateFlagBits2KHR::eLibrary )
      result += "Library | ";
    if ( value & PipelineCreateFlagBits2KHR::eRayTracingSkipTriangles )
      result += "RayTracingSkipTriangles | ";
    if ( value & PipelineCreateFlagBits2KHR::eRayTracingSkipAabbs )
      result += "RayTracingSkipAabbs | ";
    if ( value & PipelineCreateFlagBits2KHR::eRayTracingNoNullAnyHitShaders )
      result += "RayTracingNoNullAnyHitShaders | ";
    if ( value & PipelineCreateFlagBits2KHR::eRayTracingNoNullClosestHitShaders )
      result += "RayTracingNoNullClosestHitShaders | ";
    if ( value & PipelineCreateFlagBits2KHR::eRayTracingNoNullMissShaders )
      result += "RayTracingNoNullMissShaders | ";
    if ( value & PipelineCreateFlagBits2KHR::eRayTracingNoNullIntersectionShaders )
      result += "RayTracingNoNullIntersectionShaders | ";
    if ( value & PipelineCreateFlagBits2KHR::eRayTracingShaderGroupHandleCaptureReplay )
      result += "RayTracingShaderGroupHandleCaptureReplay | ";
    if ( value & PipelineCreateFlagBits2KHR::eIndirectBindableNV )
      result += "IndirectBindableNV | ";
    if ( value & PipelineCreateFlagBits2KHR::eRayTracingAllowMotionNV )
      result += "RayTracingAllowMotionNV | ";
    if ( value & PipelineCreateFlagBits2KHR::eRenderingFragmentShadingRateAttachment )
      result += "RenderingFragmentShadingRateAttachment | ";
    if ( value & PipelineCreateFlagBits2KHR::eRenderingFragmentDensityMapAttachmentEXT )
      result += "RenderingFragmentDensityMapAttachmentEXT | ";
    if ( value & PipelineCreateFlagBits2KHR::eRayTracingOpacityMicromapEXT )
      result += "RayTracingOpacityMicromapEXT | ";
    if ( value & PipelineCreateFlagBits2KHR::eColorAttachmentFeedbackLoopEXT )
      result += "ColorAttachmentFeedbackLoopEXT | ";
    if ( value & PipelineCreateFlagBits2KHR::eDepthStencilAttachmentFeedbackLoopEXT )
      result += "DepthStencilAttachmentFeedbackLoopEXT | ";
    if ( value & PipelineCreateFlagBits2KHR::eNoProtectedAccessEXT )
      result += "NoProtectedAccessEXT | ";
    if ( value & PipelineCreateFlagBits2KHR::eProtectedAccessOnlyEXT )
      result += "ProtectedAccessOnlyEXT | ";
    if ( value & PipelineCreateFlagBits2KHR::eRayTracingDisplacementMicromapNV )
      result += "RayTracingDisplacementMicromapNV | ";
    if ( value & PipelineCreateFlagBits2KHR::eDescriptorBufferEXT )
      result += "DescriptorBufferEXT | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  VULKAN_HPP_INLINE std::string to_string( BufferUsageFlags2KHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & BufferUsageFlagBits2KHR::eTransferSrc )
      result += "TransferSrc | ";
    if ( value & BufferUsageFlagBits2KHR::eTransferDst )
      result += "TransferDst | ";
    if ( value & BufferUsageFlagBits2KHR::eUniformTexelBuffer )
      result += "UniformTexelBuffer | ";
    if ( value & BufferUsageFlagBits2KHR::eStorageTexelBuffer )
      result += "StorageTexelBuffer | ";
    if ( value & BufferUsageFlagBits2KHR::eUniformBuffer )
      result += "UniformBuffer | ";
    if ( value & BufferUsageFlagBits2KHR::eStorageBuffer )
      result += "StorageBuffer | ";
    if ( value & BufferUsageFlagBits2KHR::eIndexBuffer )
      result += "IndexBuffer | ";
    if ( value & BufferUsageFlagBits2KHR::eVertexBuffer )
      result += "VertexBuffer | ";
    if ( value & BufferUsageFlagBits2KHR::eIndirectBuffer )
      result += "IndirectBuffer | ";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & BufferUsageFlagBits2KHR::eExecutionGraphScratchAMDX )
      result += "ExecutionGraphScratchAMDX | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    if ( value & BufferUsageFlagBits2KHR::eConditionalRenderingEXT )
      result += "ConditionalRenderingEXT | ";
    if ( value & BufferUsageFlagBits2KHR::eShaderBindingTable )
      result += "ShaderBindingTable | ";
    if ( value & BufferUsageFlagBits2KHR::eTransformFeedbackBufferEXT )
      result += "TransformFeedbackBufferEXT | ";
    if ( value & BufferUsageFlagBits2KHR::eTransformFeedbackCounterBufferEXT )
      result += "TransformFeedbackCounterBufferEXT | ";
    if ( value & BufferUsageFlagBits2KHR::eVideoDecodeSrc )
      result += "VideoDecodeSrc | ";
    if ( value & BufferUsageFlagBits2KHR::eVideoDecodeDst )
      result += "VideoDecodeDst | ";
    if ( value & BufferUsageFlagBits2KHR::eVideoEncodeDst )
      result += "VideoEncodeDst | ";
    if ( value & BufferUsageFlagBits2KHR::eVideoEncodeSrc )
      result += "VideoEncodeSrc | ";
    if ( value & BufferUsageFlagBits2KHR::eShaderDeviceAddress )
      result += "ShaderDeviceAddress | ";
    if ( value & BufferUsageFlagBits2KHR::eAccelerationStructureBuildInputReadOnly )
      result += "AccelerationStructureBuildInputReadOnly | ";
    if ( value & BufferUsageFlagBits2KHR::eAccelerationStructureStorage )
      result += "AccelerationStructureStorage | ";
    if ( value & BufferUsageFlagBits2KHR::eSamplerDescriptorBufferEXT )
      result += "SamplerDescriptorBufferEXT | ";
    if ( value & BufferUsageFlagBits2KHR::eResourceDescriptorBufferEXT )
      result += "ResourceDescriptorBufferEXT | ";
    if ( value & BufferUsageFlagBits2KHR::ePushDescriptorsDescriptorBufferEXT )
      result += "PushDescriptorsDescriptorBufferEXT | ";
    if ( value & BufferUsageFlagBits2KHR::eMicromapBuildInputReadOnlyEXT )
      result += "MicromapBuildInputReadOnlyEXT | ";
    if ( value & BufferUsageFlagBits2KHR::eMicromapStorageEXT )
      result += "MicromapStorageEXT | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_EXT_shader_object ===

  VULKAN_HPP_INLINE std::string to_string( ShaderCreateFlagsEXT value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & ShaderCreateFlagBitsEXT::eLinkStage )
      result += "LinkStage | ";
    if ( value & ShaderCreateFlagBitsEXT::eAllowVaryingSubgroupSize )
      result += "AllowVaryingSubgroupSize | ";
    if ( value & ShaderCreateFlagBitsEXT::eRequireFullSubgroups )
      result += "RequireFullSubgroups | ";
    if ( value & ShaderCreateFlagBitsEXT::eNoTaskShader )
      result += "NoTaskShader | ";
    if ( value & ShaderCreateFlagBitsEXT::eDispatchBase )
      result += "DispatchBase | ";
    if ( value & ShaderCreateFlagBitsEXT::eFragmentShadingRateAttachment )
      result += "FragmentShadingRateAttachment | ";
    if ( value & ShaderCreateFlagBitsEXT::eFragmentDensityMapAttachment )
      result += "FragmentDensityMapAttachment | ";

    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=======================
  //=== ENUMs to_string ===
  //=======================

  VULKAN_HPP_INLINE std::string toHexString( uint32_t value )
  {
#if __cpp_lib_format
    return std::format( "{:x}", value );
#else
    std::stringstream stream;
    stream << std::hex << value;
    return stream.str();
#endif
  }

  //=== VK_VERSION_1_0 ===

  VULKAN_HPP_INLINE std::string to_string( Result value )
  {
    switch ( value )
    {
      case Result::eSuccess: return "Success";
      case Result::eNotReady: return "NotReady";
      case Result::eTimeout: return "Timeout";
      case Result::eEventSet: return "EventSet";
      case Result::eEventReset: return "EventReset";
      case Result::eIncomplete: return "Incomplete";
      case Result::eErrorOutOfHostMemory: return "ErrorOutOfHostMemory";
      case Result::eErrorOutOfDeviceMemory: return "ErrorOutOfDeviceMemory";
      case Result::eErrorInitializationFailed: return "ErrorInitializationFailed";
      case Result::eErrorDeviceLost: return "ErrorDeviceLost";
      case Result::eErrorMemoryMapFailed: return "ErrorMemoryMapFailed";
      case Result::eErrorLayerNotPresent: return "ErrorLayerNotPresent";
      case Result::eErrorExtensionNotPresent: return "ErrorExtensionNotPresent";
      case Result::eErrorFeatureNotPresent: return "ErrorFeatureNotPresent";
      case Result::eErrorIncompatibleDriver: return "ErrorIncompatibleDriver";
      case Result::eErrorTooManyObjects: return "ErrorTooManyObjects";
      case Result::eErrorFormatNotSupported: return "ErrorFormatNotSupported";
      case Result::eErrorFragmentedPool: return "ErrorFragmentedPool";
      case Result::eErrorUnknown: return "ErrorUnknown";
      case Result::eErrorOutOfPoolMemory: return "ErrorOutOfPoolMemory";
      case Result::eErrorInvalidExternalHandle: return "ErrorInvalidExternalHandle";
      case Result::eErrorFragmentation: return "ErrorFragmentation";
      case Result::eErrorInvalidOpaqueCaptureAddress: return "ErrorInvalidOpaqueCaptureAddress";
      case Result::ePipelineCompileRequired: return "PipelineCompileRequired";
      case Result::eErrorSurfaceLostKHR: return "ErrorSurfaceLostKHR";
      case Result::eErrorNativeWindowInUseKHR: return "ErrorNativeWindowInUseKHR";
      case Result::eSuboptimalKHR: return "SuboptimalKHR";
      case Result::eErrorOutOfDateKHR: return "ErrorOutOfDateKHR";
      case Result::eErrorIncompatibleDisplayKHR: return "ErrorIncompatibleDisplayKHR";
      case Result::eErrorValidationFailedEXT: return "ErrorValidationFailedEXT";
      case Result::eErrorInvalidShaderNV: return "ErrorInvalidShaderNV";
      case Result::eErrorImageUsageNotSupportedKHR: return "ErrorImageUsageNotSupportedKHR";
      case Result::eErrorVideoPictureLayoutNotSupportedKHR: return "ErrorVideoPictureLayoutNotSupportedKHR";
      case Result::eErrorVideoProfileOperationNotSupportedKHR: return "ErrorVideoProfileOperationNotSupportedKHR";
      case Result::eErrorVideoProfileFormatNotSupportedKHR: return "ErrorVideoProfileFormatNotSupportedKHR";
      case Result::eErrorVideoProfileCodecNotSupportedKHR: return "ErrorVideoProfileCodecNotSupportedKHR";
      case Result::eErrorVideoStdVersionNotSupportedKHR: return "ErrorVideoStdVersionNotSupportedKHR";
      case Result::eErrorInvalidDrmFormatModifierPlaneLayoutEXT: return "ErrorInvalidDrmFormatModifierPlaneLayoutEXT";
      case Result::eErrorNotPermittedKHR: return "ErrorNotPermittedKHR";
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      case Result::eErrorFullScreenExclusiveModeLostEXT: return "ErrorFullScreenExclusiveModeLostEXT";
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      case Result::eThreadIdleKHR: return "ThreadIdleKHR";
      case Result::eThreadDoneKHR: return "ThreadDoneKHR";
      case Result::eOperationDeferredKHR: return "OperationDeferredKHR";
      case Result::eOperationNotDeferredKHR: return "OperationNotDeferredKHR";
      case Result::eErrorInvalidVideoStdParametersKHR: return "ErrorInvalidVideoStdParametersKHR";
      case Result::eErrorCompressionExhaustedEXT: return "ErrorCompressionExhaustedEXT";
      case Result::eIncompatibleShaderBinaryEXT: return "IncompatibleShaderBinaryEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( StructureType value )
  {
    switch ( value )
    {
      case StructureType::eApplicationInfo: return "ApplicationInfo";
      case StructureType::eInstanceCreateInfo: return "InstanceCreateInfo";
      case StructureType::eDeviceQueueCreateInfo: return "DeviceQueueCreateInfo";
      case StructureType::eDeviceCreateInfo: return "DeviceCreateInfo";
      case StructureType::eSubmitInfo: return "SubmitInfo";
      case StructureType::eMemoryAllocateInfo: return "MemoryAllocateInfo";
      case StructureType::eMappedMemoryRange: return "MappedMemoryRange";
      case StructureType::eBindSparseInfo: return "BindSparseInfo";
      case StructureType::eFenceCreateInfo: return "FenceCreateInfo";
      case StructureType::eSemaphoreCreateInfo: return "SemaphoreCreateInfo";
      case StructureType::eEventCreateInfo: return "EventCreateInfo";
      case StructureType::eQueryPoolCreateInfo: return "QueryPoolCreateInfo";
      case StructureType::eBufferCreateInfo: return "BufferCreateInfo";
      case StructureType::eBufferViewCreateInfo: return "BufferViewCreateInfo";
      case StructureType::eImageCreateInfo: return "ImageCreateInfo";
      case StructureType::eImageViewCreateInfo: return "ImageViewCreateInfo";
      case StructureType::eShaderModuleCreateInfo: return "ShaderModuleCreateInfo";
      case StructureType::ePipelineCacheCreateInfo: return "PipelineCacheCreateInfo";
      case StructureType::ePipelineShaderStageCreateInfo: return "PipelineShaderStageCreateInfo";
      case StructureType::ePipelineVertexInputStateCreateInfo: return "PipelineVertexInputStateCreateInfo";
      case StructureType::ePipelineInputAssemblyStateCreateInfo: return "PipelineInputAssemblyStateCreateInfo";
      case StructureType::ePipelineTessellationStateCreateInfo: return "PipelineTessellationStateCreateInfo";
      case StructureType::ePipelineViewportStateCreateInfo: return "PipelineViewportStateCreateInfo";
      case StructureType::ePipelineRasterizationStateCreateInfo: return "PipelineRasterizationStateCreateInfo";
      case StructureType::ePipelineMultisampleStateCreateInfo: return "PipelineMultisampleStateCreateInfo";
      case StructureType::ePipelineDepthStencilStateCreateInfo: return "PipelineDepthStencilStateCreateInfo";
      case StructureType::ePipelineColorBlendStateCreateInfo: return "PipelineColorBlendStateCreateInfo";
      case StructureType::ePipelineDynamicStateCreateInfo: return "PipelineDynamicStateCreateInfo";
      case StructureType::eGraphicsPipelineCreateInfo: return "GraphicsPipelineCreateInfo";
      case StructureType::eComputePipelineCreateInfo: return "ComputePipelineCreateInfo";
      case StructureType::ePipelineLayoutCreateInfo: return "PipelineLayoutCreateInfo";
      case StructureType::eSamplerCreateInfo: return "SamplerCreateInfo";
      case StructureType::eDescriptorSetLayoutCreateInfo: return "DescriptorSetLayoutCreateInfo";
      case StructureType::eDescriptorPoolCreateInfo: return "DescriptorPoolCreateInfo";
      case StructureType::eDescriptorSetAllocateInfo: return "DescriptorSetAllocateInfo";
      case StructureType::eWriteDescriptorSet: return "WriteDescriptorSet";
      case StructureType::eCopyDescriptorSet: return "CopyDescriptorSet";
      case StructureType::eFramebufferCreateInfo: return "FramebufferCreateInfo";
      case StructureType::eRenderPassCreateInfo: return "RenderPassCreateInfo";
      case StructureType::eCommandPoolCreateInfo: return "CommandPoolCreateInfo";
      case StructureType::eCommandBufferAllocateInfo: return "CommandBufferAllocateInfo";
      case StructureType::eCommandBufferInheritanceInfo: return "CommandBufferInheritanceInfo";
      case StructureType::eCommandBufferBeginInfo: return "CommandBufferBeginInfo";
      case StructureType::eRenderPassBeginInfo: return "RenderPassBeginInfo";
      case StructureType::eBufferMemoryBarrier: return "BufferMemoryBarrier";
      case StructureType::eImageMemoryBarrier: return "ImageMemoryBarrier";
      case StructureType::eMemoryBarrier: return "MemoryBarrier";
      case StructureType::eLoaderInstanceCreateInfo: return "LoaderInstanceCreateInfo";
      case StructureType::eLoaderDeviceCreateInfo: return "LoaderDeviceCreateInfo";
      case StructureType::ePhysicalDeviceSubgroupProperties: return "PhysicalDeviceSubgroupProperties";
      case StructureType::eBindBufferMemoryInfo: return "BindBufferMemoryInfo";
      case StructureType::eBindImageMemoryInfo: return "BindImageMemoryInfo";
      case StructureType::ePhysicalDevice16BitStorageFeatures: return "PhysicalDevice16BitStorageFeatures";
      case StructureType::eMemoryDedicatedRequirements: return "MemoryDedicatedRequirements";
      case StructureType::eMemoryDedicatedAllocateInfo: return "MemoryDedicatedAllocateInfo";
      case StructureType::eMemoryAllocateFlagsInfo: return "MemoryAllocateFlagsInfo";
      case StructureType::eDeviceGroupRenderPassBeginInfo: return "DeviceGroupRenderPassBeginInfo";
      case StructureType::eDeviceGroupCommandBufferBeginInfo: return "DeviceGroupCommandBufferBeginInfo";
      case StructureType::eDeviceGroupSubmitInfo: return "DeviceGroupSubmitInfo";
      case StructureType::eDeviceGroupBindSparseInfo: return "DeviceGroupBindSparseInfo";
      case StructureType::eBindBufferMemoryDeviceGroupInfo: return "BindBufferMemoryDeviceGroupInfo";
      case StructureType::eBindImageMemoryDeviceGroupInfo: return "BindImageMemoryDeviceGroupInfo";
      case StructureType::ePhysicalDeviceGroupProperties: return "PhysicalDeviceGroupProperties";
      case StructureType::eDeviceGroupDeviceCreateInfo: return "DeviceGroupDeviceCreateInfo";
      case StructureType::eBufferMemoryRequirementsInfo2: return "BufferMemoryRequirementsInfo2";
      case StructureType::eImageMemoryRequirementsInfo2: return "ImageMemoryRequirementsInfo2";
      case StructureType::eImageSparseMemoryRequirementsInfo2: return "ImageSparseMemoryRequirementsInfo2";
      case StructureType::eMemoryRequirements2: return "MemoryRequirements2";
      case StructureType::eSparseImageMemoryRequirements2: return "SparseImageMemoryRequirements2";
      case StructureType::ePhysicalDeviceFeatures2: return "PhysicalDeviceFeatures2";
      case StructureType::ePhysicalDeviceProperties2: return "PhysicalDeviceProperties2";
      case StructureType::eFormatProperties2: return "FormatProperties2";
      case StructureType::eImageFormatProperties2: return "ImageFormatProperties2";
      case StructureType::ePhysicalDeviceImageFormatInfo2: return "PhysicalDeviceImageFormatInfo2";
      case StructureType::eQueueFamilyProperties2: return "QueueFamilyProperties2";
      case StructureType::ePhysicalDeviceMemoryProperties2: return "PhysicalDeviceMemoryProperties2";
      case StructureType::eSparseImageFormatProperties2: return "SparseImageFormatProperties2";
      case StructureType::ePhysicalDeviceSparseImageFormatInfo2: return "PhysicalDeviceSparseImageFormatInfo2";
      case StructureType::ePhysicalDevicePointClippingProperties: return "PhysicalDevicePointClippingProperties";
      case StructureType::eRenderPassInputAttachmentAspectCreateInfo: return "RenderPassInputAttachmentAspectCreateInfo";
      case StructureType::eImageViewUsageCreateInfo: return "ImageViewUsageCreateInfo";
      case StructureType::ePipelineTessellationDomainOriginStateCreateInfo: return "PipelineTessellationDomainOriginStateCreateInfo";
      case StructureType::eRenderPassMultiviewCreateInfo: return "RenderPassMultiviewCreateInfo";
      case StructureType::ePhysicalDeviceMultiviewFeatures: return "PhysicalDeviceMultiviewFeatures";
      case StructureType::ePhysicalDeviceMultiviewProperties: return "PhysicalDeviceMultiviewProperties";
      case StructureType::ePhysicalDeviceVariablePointersFeatures: return "PhysicalDeviceVariablePointersFeatures";
      case StructureType::eProtectedSubmitInfo: return "ProtectedSubmitInfo";
      case StructureType::ePhysicalDeviceProtectedMemoryFeatures: return "PhysicalDeviceProtectedMemoryFeatures";
      case StructureType::ePhysicalDeviceProtectedMemoryProperties: return "PhysicalDeviceProtectedMemoryProperties";
      case StructureType::eDeviceQueueInfo2: return "DeviceQueueInfo2";
      case StructureType::eSamplerYcbcrConversionCreateInfo: return "SamplerYcbcrConversionCreateInfo";
      case StructureType::eSamplerYcbcrConversionInfo: return "SamplerYcbcrConversionInfo";
      case StructureType::eBindImagePlaneMemoryInfo: return "BindImagePlaneMemoryInfo";
      case StructureType::eImagePlaneMemoryRequirementsInfo: return "ImagePlaneMemoryRequirementsInfo";
      case StructureType::ePhysicalDeviceSamplerYcbcrConversionFeatures: return "PhysicalDeviceSamplerYcbcrConversionFeatures";
      case StructureType::eSamplerYcbcrConversionImageFormatProperties: return "SamplerYcbcrConversionImageFormatProperties";
      case StructureType::eDescriptorUpdateTemplateCreateInfo: return "DescriptorUpdateTemplateCreateInfo";
      case StructureType::ePhysicalDeviceExternalImageFormatInfo: return "PhysicalDeviceExternalImageFormatInfo";
      case StructureType::eExternalImageFormatProperties: return "ExternalImageFormatProperties";
      case StructureType::ePhysicalDeviceExternalBufferInfo: return "PhysicalDeviceExternalBufferInfo";
      case StructureType::eExternalBufferProperties: return "ExternalBufferProperties";
      case StructureType::ePhysicalDeviceIdProperties: return "PhysicalDeviceIdProperties";
      case StructureType::eExternalMemoryBufferCreateInfo: return "ExternalMemoryBufferCreateInfo";
      case StructureType::eExternalMemoryImageCreateInfo: return "ExternalMemoryImageCreateInfo";
      case StructureType::eExportMemoryAllocateInfo: return "ExportMemoryAllocateInfo";
      case StructureType::ePhysicalDeviceExternalFenceInfo: return "PhysicalDeviceExternalFenceInfo";
      case StructureType::eExternalFenceProperties: return "ExternalFenceProperties";
      case StructureType::eExportFenceCreateInfo: return "ExportFenceCreateInfo";
      case StructureType::eExportSemaphoreCreateInfo: return "ExportSemaphoreCreateInfo";
      case StructureType::ePhysicalDeviceExternalSemaphoreInfo: return "PhysicalDeviceExternalSemaphoreInfo";
      case StructureType::eExternalSemaphoreProperties: return "ExternalSemaphoreProperties";
      case StructureType::ePhysicalDeviceMaintenance3Properties: return "PhysicalDeviceMaintenance3Properties";
      case StructureType::eDescriptorSetLayoutSupport: return "DescriptorSetLayoutSupport";
      case StructureType::ePhysicalDeviceShaderDrawParametersFeatures: return "PhysicalDeviceShaderDrawParametersFeatures";
      case StructureType::ePhysicalDeviceVulkan11Features: return "PhysicalDeviceVulkan11Features";
      case StructureType::ePhysicalDeviceVulkan11Properties: return "PhysicalDeviceVulkan11Properties";
      case StructureType::ePhysicalDeviceVulkan12Features: return "PhysicalDeviceVulkan12Features";
      case StructureType::ePhysicalDeviceVulkan12Properties: return "PhysicalDeviceVulkan12Properties";
      case StructureType::eImageFormatListCreateInfo: return "ImageFormatListCreateInfo";
      case StructureType::eAttachmentDescription2: return "AttachmentDescription2";
      case StructureType::eAttachmentReference2: return "AttachmentReference2";
      case StructureType::eSubpassDescription2: return "SubpassDescription2";
      case StructureType::eSubpassDependency2: return "SubpassDependency2";
      case StructureType::eRenderPassCreateInfo2: return "RenderPassCreateInfo2";
      case StructureType::eSubpassBeginInfo: return "SubpassBeginInfo";
      case StructureType::eSubpassEndInfo: return "SubpassEndInfo";
      case StructureType::ePhysicalDevice8BitStorageFeatures: return "PhysicalDevice8BitStorageFeatures";
      case StructureType::ePhysicalDeviceDriverProperties: return "PhysicalDeviceDriverProperties";
      case StructureType::ePhysicalDeviceShaderAtomicInt64Features: return "PhysicalDeviceShaderAtomicInt64Features";
      case StructureType::ePhysicalDeviceShaderFloat16Int8Features: return "PhysicalDeviceShaderFloat16Int8Features";
      case StructureType::ePhysicalDeviceFloatControlsProperties: return "PhysicalDeviceFloatControlsProperties";
      case StructureType::eDescriptorSetLayoutBindingFlagsCreateInfo: return "DescriptorSetLayoutBindingFlagsCreateInfo";
      case StructureType::ePhysicalDeviceDescriptorIndexingFeatures: return "PhysicalDeviceDescriptorIndexingFeatures";
      case StructureType::ePhysicalDeviceDescriptorIndexingProperties: return "PhysicalDeviceDescriptorIndexingProperties";
      case StructureType::eDescriptorSetVariableDescriptorCountAllocateInfo: return "DescriptorSetVariableDescriptorCountAllocateInfo";
      case StructureType::eDescriptorSetVariableDescriptorCountLayoutSupport: return "DescriptorSetVariableDescriptorCountLayoutSupport";
      case StructureType::ePhysicalDeviceDepthStencilResolveProperties: return "PhysicalDeviceDepthStencilResolveProperties";
      case StructureType::eSubpassDescriptionDepthStencilResolve: return "SubpassDescriptionDepthStencilResolve";
      case StructureType::ePhysicalDeviceScalarBlockLayoutFeatures: return "PhysicalDeviceScalarBlockLayoutFeatures";
      case StructureType::eImageStencilUsageCreateInfo: return "ImageStencilUsageCreateInfo";
      case StructureType::ePhysicalDeviceSamplerFilterMinmaxProperties: return "PhysicalDeviceSamplerFilterMinmaxProperties";
      case StructureType::eSamplerReductionModeCreateInfo: return "SamplerReductionModeCreateInfo";
      case StructureType::ePhysicalDeviceVulkanMemoryModelFeatures: return "PhysicalDeviceVulkanMemoryModelFeatures";
      case StructureType::ePhysicalDeviceImagelessFramebufferFeatures: return "PhysicalDeviceImagelessFramebufferFeatures";
      case StructureType::eFramebufferAttachmentsCreateInfo: return "FramebufferAttachmentsCreateInfo";
      case StructureType::eFramebufferAttachmentImageInfo: return "FramebufferAttachmentImageInfo";
      case StructureType::eRenderPassAttachmentBeginInfo: return "RenderPassAttachmentBeginInfo";
      case StructureType::ePhysicalDeviceUniformBufferStandardLayoutFeatures: return "PhysicalDeviceUniformBufferStandardLayoutFeatures";
      case StructureType::ePhysicalDeviceShaderSubgroupExtendedTypesFeatures: return "PhysicalDeviceShaderSubgroupExtendedTypesFeatures";
      case StructureType::ePhysicalDeviceSeparateDepthStencilLayoutsFeatures: return "PhysicalDeviceSeparateDepthStencilLayoutsFeatures";
      case StructureType::eAttachmentReferenceStencilLayout: return "AttachmentReferenceStencilLayout";
      case StructureType::eAttachmentDescriptionStencilLayout: return "AttachmentDescriptionStencilLayout";
      case StructureType::ePhysicalDeviceHostQueryResetFeatures: return "PhysicalDeviceHostQueryResetFeatures";
      case StructureType::ePhysicalDeviceTimelineSemaphoreFeatures: return "PhysicalDeviceTimelineSemaphoreFeatures";
      case StructureType::ePhysicalDeviceTimelineSemaphoreProperties: return "PhysicalDeviceTimelineSemaphoreProperties";
      case StructureType::eSemaphoreTypeCreateInfo: return "SemaphoreTypeCreateInfo";
      case StructureType::eTimelineSemaphoreSubmitInfo: return "TimelineSemaphoreSubmitInfo";
      case StructureType::eSemaphoreWaitInfo: return "SemaphoreWaitInfo";
      case StructureType::eSemaphoreSignalInfo: return "SemaphoreSignalInfo";
      case StructureType::ePhysicalDeviceBufferDeviceAddressFeatures: return "PhysicalDeviceBufferDeviceAddressFeatures";
      case StructureType::eBufferDeviceAddressInfo: return "BufferDeviceAddressInfo";
      case StructureType::eBufferOpaqueCaptureAddressCreateInfo: return "BufferOpaqueCaptureAddressCreateInfo";
      case StructureType::eMemoryOpaqueCaptureAddressAllocateInfo: return "MemoryOpaqueCaptureAddressAllocateInfo";
      case StructureType::eDeviceMemoryOpaqueCaptureAddressInfo: return "DeviceMemoryOpaqueCaptureAddressInfo";
      case StructureType::ePhysicalDeviceVulkan13Features: return "PhysicalDeviceVulkan13Features";
      case StructureType::ePhysicalDeviceVulkan13Properties: return "PhysicalDeviceVulkan13Properties";
      case StructureType::ePipelineCreationFeedbackCreateInfo: return "PipelineCreationFeedbackCreateInfo";
      case StructureType::ePhysicalDeviceShaderTerminateInvocationFeatures: return "PhysicalDeviceShaderTerminateInvocationFeatures";
      case StructureType::ePhysicalDeviceToolProperties: return "PhysicalDeviceToolProperties";
      case StructureType::ePhysicalDeviceShaderDemoteToHelperInvocationFeatures: return "PhysicalDeviceShaderDemoteToHelperInvocationFeatures";
      case StructureType::ePhysicalDevicePrivateDataFeatures: return "PhysicalDevicePrivateDataFeatures";
      case StructureType::eDevicePrivateDataCreateInfo: return "DevicePrivateDataCreateInfo";
      case StructureType::ePrivateDataSlotCreateInfo: return "PrivateDataSlotCreateInfo";
      case StructureType::ePhysicalDevicePipelineCreationCacheControlFeatures: return "PhysicalDevicePipelineCreationCacheControlFeatures";
      case StructureType::eMemoryBarrier2: return "MemoryBarrier2";
      case StructureType::eBufferMemoryBarrier2: return "BufferMemoryBarrier2";
      case StructureType::eImageMemoryBarrier2: return "ImageMemoryBarrier2";
      case StructureType::eDependencyInfo: return "DependencyInfo";
      case StructureType::eSubmitInfo2: return "SubmitInfo2";
      case StructureType::eSemaphoreSubmitInfo: return "SemaphoreSubmitInfo";
      case StructureType::eCommandBufferSubmitInfo: return "CommandBufferSubmitInfo";
      case StructureType::ePhysicalDeviceSynchronization2Features: return "PhysicalDeviceSynchronization2Features";
      case StructureType::ePhysicalDeviceZeroInitializeWorkgroupMemoryFeatures: return "PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures";
      case StructureType::ePhysicalDeviceImageRobustnessFeatures: return "PhysicalDeviceImageRobustnessFeatures";
      case StructureType::eCopyBufferInfo2: return "CopyBufferInfo2";
      case StructureType::eCopyImageInfo2: return "CopyImageInfo2";
      case StructureType::eCopyBufferToImageInfo2: return "CopyBufferToImageInfo2";
      case StructureType::eCopyImageToBufferInfo2: return "CopyImageToBufferInfo2";
      case StructureType::eBlitImageInfo2: return "BlitImageInfo2";
      case StructureType::eResolveImageInfo2: return "ResolveImageInfo2";
      case StructureType::eBufferCopy2: return "BufferCopy2";
      case StructureType::eImageCopy2: return "ImageCopy2";
      case StructureType::eImageBlit2: return "ImageBlit2";
      case StructureType::eBufferImageCopy2: return "BufferImageCopy2";
      case StructureType::eImageResolve2: return "ImageResolve2";
      case StructureType::ePhysicalDeviceSubgroupSizeControlProperties: return "PhysicalDeviceSubgroupSizeControlProperties";
      case StructureType::ePipelineShaderStageRequiredSubgroupSizeCreateInfo: return "PipelineShaderStageRequiredSubgroupSizeCreateInfo";
      case StructureType::ePhysicalDeviceSubgroupSizeControlFeatures: return "PhysicalDeviceSubgroupSizeControlFeatures";
      case StructureType::ePhysicalDeviceInlineUniformBlockFeatures: return "PhysicalDeviceInlineUniformBlockFeatures";
      case StructureType::ePhysicalDeviceInlineUniformBlockProperties: return "PhysicalDeviceInlineUniformBlockProperties";
      case StructureType::eWriteDescriptorSetInlineUniformBlock: return "WriteDescriptorSetInlineUniformBlock";
      case StructureType::eDescriptorPoolInlineUniformBlockCreateInfo: return "DescriptorPoolInlineUniformBlockCreateInfo";
      case StructureType::ePhysicalDeviceTextureCompressionAstcHdrFeatures: return "PhysicalDeviceTextureCompressionAstcHdrFeatures";
      case StructureType::eRenderingInfo: return "RenderingInfo";
      case StructureType::eRenderingAttachmentInfo: return "RenderingAttachmentInfo";
      case StructureType::ePipelineRenderingCreateInfo: return "PipelineRenderingCreateInfo";
      case StructureType::ePhysicalDeviceDynamicRenderingFeatures: return "PhysicalDeviceDynamicRenderingFeatures";
      case StructureType::eCommandBufferInheritanceRenderingInfo: return "CommandBufferInheritanceRenderingInfo";
      case StructureType::ePhysicalDeviceShaderIntegerDotProductFeatures: return "PhysicalDeviceShaderIntegerDotProductFeatures";
      case StructureType::ePhysicalDeviceShaderIntegerDotProductProperties: return "PhysicalDeviceShaderIntegerDotProductProperties";
      case StructureType::ePhysicalDeviceTexelBufferAlignmentProperties: return "PhysicalDeviceTexelBufferAlignmentProperties";
      case StructureType::eFormatProperties3: return "FormatProperties3";
      case StructureType::ePhysicalDeviceMaintenance4Features: return "PhysicalDeviceMaintenance4Features";
      case StructureType::ePhysicalDeviceMaintenance4Properties: return "PhysicalDeviceMaintenance4Properties";
      case StructureType::eDeviceBufferMemoryRequirements: return "DeviceBufferMemoryRequirements";
      case StructureType::eDeviceImageMemoryRequirements: return "DeviceImageMemoryRequirements";
      case StructureType::eSwapchainCreateInfoKHR: return "SwapchainCreateInfoKHR";
      case StructureType::ePresentInfoKHR: return "PresentInfoKHR";
      case StructureType::eDeviceGroupPresentCapabilitiesKHR: return "DeviceGroupPresentCapabilitiesKHR";
      case StructureType::eImageSwapchainCreateInfoKHR: return "ImageSwapchainCreateInfoKHR";
      case StructureType::eBindImageMemorySwapchainInfoKHR: return "BindImageMemorySwapchainInfoKHR";
      case StructureType::eAcquireNextImageInfoKHR: return "AcquireNextImageInfoKHR";
      case StructureType::eDeviceGroupPresentInfoKHR: return "DeviceGroupPresentInfoKHR";
      case StructureType::eDeviceGroupSwapchainCreateInfoKHR: return "DeviceGroupSwapchainCreateInfoKHR";
      case StructureType::eDisplayModeCreateInfoKHR: return "DisplayModeCreateInfoKHR";
      case StructureType::eDisplaySurfaceCreateInfoKHR: return "DisplaySurfaceCreateInfoKHR";
      case StructureType::eDisplayPresentInfoKHR: return "DisplayPresentInfoKHR";
#if defined( VK_USE_PLATFORM_XLIB_KHR )
      case StructureType::eXlibSurfaceCreateInfoKHR: return "XlibSurfaceCreateInfoKHR";
#endif /*VK_USE_PLATFORM_XLIB_KHR*/
#if defined( VK_USE_PLATFORM_XCB_KHR )
      case StructureType::eXcbSurfaceCreateInfoKHR: return "XcbSurfaceCreateInfoKHR";
#endif /*VK_USE_PLATFORM_XCB_KHR*/
#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
      case StructureType::eWaylandSurfaceCreateInfoKHR: return "WaylandSurfaceCreateInfoKHR";
#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
      case StructureType::eAndroidSurfaceCreateInfoKHR: return "AndroidSurfaceCreateInfoKHR";
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      case StructureType::eWin32SurfaceCreateInfoKHR: return "Win32SurfaceCreateInfoKHR";
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      case StructureType::eDebugReportCallbackCreateInfoEXT: return "DebugReportCallbackCreateInfoEXT";
      case StructureType::ePipelineRasterizationStateRasterizationOrderAMD: return "PipelineRasterizationStateRasterizationOrderAMD";
      case StructureType::eDebugMarkerObjectNameInfoEXT: return "DebugMarkerObjectNameInfoEXT";
      case StructureType::eDebugMarkerObjectTagInfoEXT: return "DebugMarkerObjectTagInfoEXT";
      case StructureType::eDebugMarkerMarkerInfoEXT: return "DebugMarkerMarkerInfoEXT";
      case StructureType::eVideoProfileInfoKHR: return "VideoProfileInfoKHR";
      case StructureType::eVideoCapabilitiesKHR: return "VideoCapabilitiesKHR";
      case StructureType::eVideoPictureResourceInfoKHR: return "VideoPictureResourceInfoKHR";
      case StructureType::eVideoSessionMemoryRequirementsKHR: return "VideoSessionMemoryRequirementsKHR";
      case StructureType::eBindVideoSessionMemoryInfoKHR: return "BindVideoSessionMemoryInfoKHR";
      case StructureType::eVideoSessionCreateInfoKHR: return "VideoSessionCreateInfoKHR";
      case StructureType::eVideoSessionParametersCreateInfoKHR: return "VideoSessionParametersCreateInfoKHR";
      case StructureType::eVideoSessionParametersUpdateInfoKHR: return "VideoSessionParametersUpdateInfoKHR";
      case StructureType::eVideoBeginCodingInfoKHR: return "VideoBeginCodingInfoKHR";
      case StructureType::eVideoEndCodingInfoKHR: return "VideoEndCodingInfoKHR";
      case StructureType::eVideoCodingControlInfoKHR: return "VideoCodingControlInfoKHR";
      case StructureType::eVideoReferenceSlotInfoKHR: return "VideoReferenceSlotInfoKHR";
      case StructureType::eQueueFamilyVideoPropertiesKHR: return "QueueFamilyVideoPropertiesKHR";
      case StructureType::eVideoProfileListInfoKHR: return "VideoProfileListInfoKHR";
      case StructureType::ePhysicalDeviceVideoFormatInfoKHR: return "PhysicalDeviceVideoFormatInfoKHR";
      case StructureType::eVideoFormatPropertiesKHR: return "VideoFormatPropertiesKHR";
      case StructureType::eQueueFamilyQueryResultStatusPropertiesKHR: return "QueueFamilyQueryResultStatusPropertiesKHR";
      case StructureType::eVideoDecodeInfoKHR: return "VideoDecodeInfoKHR";
      case StructureType::eVideoDecodeCapabilitiesKHR: return "VideoDecodeCapabilitiesKHR";
      case StructureType::eVideoDecodeUsageInfoKHR: return "VideoDecodeUsageInfoKHR";
      case StructureType::eDedicatedAllocationImageCreateInfoNV: return "DedicatedAllocationImageCreateInfoNV";
      case StructureType::eDedicatedAllocationBufferCreateInfoNV: return "DedicatedAllocationBufferCreateInfoNV";
      case StructureType::eDedicatedAllocationMemoryAllocateInfoNV: return "DedicatedAllocationMemoryAllocateInfoNV";
      case StructureType::ePhysicalDeviceTransformFeedbackFeaturesEXT: return "PhysicalDeviceTransformFeedbackFeaturesEXT";
      case StructureType::ePhysicalDeviceTransformFeedbackPropertiesEXT: return "PhysicalDeviceTransformFeedbackPropertiesEXT";
      case StructureType::ePipelineRasterizationStateStreamCreateInfoEXT: return "PipelineRasterizationStateStreamCreateInfoEXT";
      case StructureType::eCuModuleCreateInfoNVX: return "CuModuleCreateInfoNVX";
      case StructureType::eCuFunctionCreateInfoNVX: return "CuFunctionCreateInfoNVX";
      case StructureType::eCuLaunchInfoNVX: return "CuLaunchInfoNVX";
      case StructureType::eImageViewHandleInfoNVX: return "ImageViewHandleInfoNVX";
      case StructureType::eImageViewAddressPropertiesNVX: return "ImageViewAddressPropertiesNVX";
      case StructureType::eVideoEncodeH264CapabilitiesKHR: return "VideoEncodeH264CapabilitiesKHR";
      case StructureType::eVideoEncodeH264SessionParametersCreateInfoKHR: return "VideoEncodeH264SessionParametersCreateInfoKHR";
      case StructureType::eVideoEncodeH264SessionParametersAddInfoKHR: return "VideoEncodeH264SessionParametersAddInfoKHR";
      case StructureType::eVideoEncodeH264PictureInfoKHR: return "VideoEncodeH264PictureInfoKHR";
      case StructureType::eVideoEncodeH264DpbSlotInfoKHR: return "VideoEncodeH264DpbSlotInfoKHR";
      case StructureType::eVideoEncodeH264NaluSliceInfoKHR: return "VideoEncodeH264NaluSliceInfoKHR";
      case StructureType::eVideoEncodeH264GopRemainingFrameInfoKHR: return "VideoEncodeH264GopRemainingFrameInfoKHR";
      case StructureType::eVideoEncodeH264ProfileInfoKHR: return "VideoEncodeH264ProfileInfoKHR";
      case StructureType::eVideoEncodeH264RateControlInfoKHR: return "VideoEncodeH264RateControlInfoKHR";
      case StructureType::eVideoEncodeH264RateControlLayerInfoKHR: return "VideoEncodeH264RateControlLayerInfoKHR";
      case StructureType::eVideoEncodeH264SessionCreateInfoKHR: return "VideoEncodeH264SessionCreateInfoKHR";
      case StructureType::eVideoEncodeH264QualityLevelPropertiesKHR: return "VideoEncodeH264QualityLevelPropertiesKHR";
      case StructureType::eVideoEncodeH264SessionParametersGetInfoKHR: return "VideoEncodeH264SessionParametersGetInfoKHR";
      case StructureType::eVideoEncodeH264SessionParametersFeedbackInfoKHR: return "VideoEncodeH264SessionParametersFeedbackInfoKHR";
      case StructureType::eVideoEncodeH265CapabilitiesKHR: return "VideoEncodeH265CapabilitiesKHR";
      case StructureType::eVideoEncodeH265SessionParametersCreateInfoKHR: return "VideoEncodeH265SessionParametersCreateInfoKHR";
      case StructureType::eVideoEncodeH265SessionParametersAddInfoKHR: return "VideoEncodeH265SessionParametersAddInfoKHR";
      case StructureType::eVideoEncodeH265PictureInfoKHR: return "VideoEncodeH265PictureInfoKHR";
      case StructureType::eVideoEncodeH265DpbSlotInfoKHR: return "VideoEncodeH265DpbSlotInfoKHR";
      case StructureType::eVideoEncodeH265NaluSliceSegmentInfoKHR: return "VideoEncodeH265NaluSliceSegmentInfoKHR";
      case StructureType::eVideoEncodeH265GopRemainingFrameInfoKHR: return "VideoEncodeH265GopRemainingFrameInfoKHR";
      case StructureType::eVideoEncodeH265ProfileInfoKHR: return "VideoEncodeH265ProfileInfoKHR";
      case StructureType::eVideoEncodeH265RateControlInfoKHR: return "VideoEncodeH265RateControlInfoKHR";
      case StructureType::eVideoEncodeH265RateControlLayerInfoKHR: return "VideoEncodeH265RateControlLayerInfoKHR";
      case StructureType::eVideoEncodeH265SessionCreateInfoKHR: return "VideoEncodeH265SessionCreateInfoKHR";
      case StructureType::eVideoEncodeH265QualityLevelPropertiesKHR: return "VideoEncodeH265QualityLevelPropertiesKHR";
      case StructureType::eVideoEncodeH265SessionParametersGetInfoKHR: return "VideoEncodeH265SessionParametersGetInfoKHR";
      case StructureType::eVideoEncodeH265SessionParametersFeedbackInfoKHR: return "VideoEncodeH265SessionParametersFeedbackInfoKHR";
      case StructureType::eVideoDecodeH264CapabilitiesKHR: return "VideoDecodeH264CapabilitiesKHR";
      case StructureType::eVideoDecodeH264PictureInfoKHR: return "VideoDecodeH264PictureInfoKHR";
      case StructureType::eVideoDecodeH264ProfileInfoKHR: return "VideoDecodeH264ProfileInfoKHR";
      case StructureType::eVideoDecodeH264SessionParametersCreateInfoKHR: return "VideoDecodeH264SessionParametersCreateInfoKHR";
      case StructureType::eVideoDecodeH264SessionParametersAddInfoKHR: return "VideoDecodeH264SessionParametersAddInfoKHR";
      case StructureType::eVideoDecodeH264DpbSlotInfoKHR: return "VideoDecodeH264DpbSlotInfoKHR";
      case StructureType::eTextureLodGatherFormatPropertiesAMD: return "TextureLodGatherFormatPropertiesAMD";
      case StructureType::eRenderingFragmentShadingRateAttachmentInfoKHR: return "RenderingFragmentShadingRateAttachmentInfoKHR";
      case StructureType::eRenderingFragmentDensityMapAttachmentInfoEXT: return "RenderingFragmentDensityMapAttachmentInfoEXT";
      case StructureType::eAttachmentSampleCountInfoAMD: return "AttachmentSampleCountInfoAMD";
      case StructureType::eMultiviewPerViewAttributesInfoNVX: return "MultiviewPerViewAttributesInfoNVX";
#if defined( VK_USE_PLATFORM_GGP )
      case StructureType::eStreamDescriptorSurfaceCreateInfoGGP: return "StreamDescriptorSurfaceCreateInfoGGP";
#endif /*VK_USE_PLATFORM_GGP*/
      case StructureType::ePhysicalDeviceCornerSampledImageFeaturesNV: return "PhysicalDeviceCornerSampledImageFeaturesNV";
      case StructureType::eExternalMemoryImageCreateInfoNV: return "ExternalMemoryImageCreateInfoNV";
      case StructureType::eExportMemoryAllocateInfoNV: return "ExportMemoryAllocateInfoNV";
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      case StructureType::eImportMemoryWin32HandleInfoNV: return "ImportMemoryWin32HandleInfoNV";
      case StructureType::eExportMemoryWin32HandleInfoNV: return "ExportMemoryWin32HandleInfoNV";
      case StructureType::eWin32KeyedMutexAcquireReleaseInfoNV: return "Win32KeyedMutexAcquireReleaseInfoNV";
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      case StructureType::eValidationFlagsEXT: return "ValidationFlagsEXT";
#if defined( VK_USE_PLATFORM_VI_NN )
      case StructureType::eViSurfaceCreateInfoNN: return "ViSurfaceCreateInfoNN";
#endif /*VK_USE_PLATFORM_VI_NN*/
      case StructureType::eImageViewAstcDecodeModeEXT: return "ImageViewAstcDecodeModeEXT";
      case StructureType::ePhysicalDeviceAstcDecodeFeaturesEXT: return "PhysicalDeviceAstcDecodeFeaturesEXT";
      case StructureType::ePipelineRobustnessCreateInfoEXT: return "PipelineRobustnessCreateInfoEXT";
      case StructureType::ePhysicalDevicePipelineRobustnessFeaturesEXT: return "PhysicalDevicePipelineRobustnessFeaturesEXT";
      case StructureType::ePhysicalDevicePipelineRobustnessPropertiesEXT: return "PhysicalDevicePipelineRobustnessPropertiesEXT";
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      case StructureType::eImportMemoryWin32HandleInfoKHR: return "ImportMemoryWin32HandleInfoKHR";
      case StructureType::eExportMemoryWin32HandleInfoKHR: return "ExportMemoryWin32HandleInfoKHR";
      case StructureType::eMemoryWin32HandlePropertiesKHR: return "MemoryWin32HandlePropertiesKHR";
      case StructureType::eMemoryGetWin32HandleInfoKHR: return "MemoryGetWin32HandleInfoKHR";
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      case StructureType::eImportMemoryFdInfoKHR: return "ImportMemoryFdInfoKHR";
      case StructureType::eMemoryFdPropertiesKHR: return "MemoryFdPropertiesKHR";
      case StructureType::eMemoryGetFdInfoKHR: return "MemoryGetFdInfoKHR";
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      case StructureType::eWin32KeyedMutexAcquireReleaseInfoKHR: return "Win32KeyedMutexAcquireReleaseInfoKHR";
      case StructureType::eImportSemaphoreWin32HandleInfoKHR: return "ImportSemaphoreWin32HandleInfoKHR";
      case StructureType::eExportSemaphoreWin32HandleInfoKHR: return "ExportSemaphoreWin32HandleInfoKHR";
      case StructureType::eD3D12FenceSubmitInfoKHR: return "D3D12FenceSubmitInfoKHR";
      case StructureType::eSemaphoreGetWin32HandleInfoKHR: return "SemaphoreGetWin32HandleInfoKHR";
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      case StructureType::eImportSemaphoreFdInfoKHR: return "ImportSemaphoreFdInfoKHR";
      case StructureType::eSemaphoreGetFdInfoKHR: return "SemaphoreGetFdInfoKHR";
      case StructureType::ePhysicalDevicePushDescriptorPropertiesKHR: return "PhysicalDevicePushDescriptorPropertiesKHR";
      case StructureType::eCommandBufferInheritanceConditionalRenderingInfoEXT: return "CommandBufferInheritanceConditionalRenderingInfoEXT";
      case StructureType::ePhysicalDeviceConditionalRenderingFeaturesEXT: return "PhysicalDeviceConditionalRenderingFeaturesEXT";
      case StructureType::eConditionalRenderingBeginInfoEXT: return "ConditionalRenderingBeginInfoEXT";
      case StructureType::ePresentRegionsKHR: return "PresentRegionsKHR";
      case StructureType::ePipelineViewportWScalingStateCreateInfoNV: return "PipelineViewportWScalingStateCreateInfoNV";
      case StructureType::eSurfaceCapabilities2EXT: return "SurfaceCapabilities2EXT";
      case StructureType::eDisplayPowerInfoEXT: return "DisplayPowerInfoEXT";
      case StructureType::eDeviceEventInfoEXT: return "DeviceEventInfoEXT";
      case StructureType::eDisplayEventInfoEXT: return "DisplayEventInfoEXT";
      case StructureType::eSwapchainCounterCreateInfoEXT: return "SwapchainCounterCreateInfoEXT";
      case StructureType::ePresentTimesInfoGOOGLE: return "PresentTimesInfoGOOGLE";
      case StructureType::ePhysicalDeviceMultiviewPerViewAttributesPropertiesNVX: return "PhysicalDeviceMultiviewPerViewAttributesPropertiesNVX";
      case StructureType::ePipelineViewportSwizzleStateCreateInfoNV: return "PipelineViewportSwizzleStateCreateInfoNV";
      case StructureType::ePhysicalDeviceDiscardRectanglePropertiesEXT: return "PhysicalDeviceDiscardRectanglePropertiesEXT";
      case StructureType::ePipelineDiscardRectangleStateCreateInfoEXT: return "PipelineDiscardRectangleStateCreateInfoEXT";
      case StructureType::ePhysicalDeviceConservativeRasterizationPropertiesEXT: return "PhysicalDeviceConservativeRasterizationPropertiesEXT";
      case StructureType::ePipelineRasterizationConservativeStateCreateInfoEXT: return "PipelineRasterizationConservativeStateCreateInfoEXT";
      case StructureType::ePhysicalDeviceDepthClipEnableFeaturesEXT: return "PhysicalDeviceDepthClipEnableFeaturesEXT";
      case StructureType::ePipelineRasterizationDepthClipStateCreateInfoEXT: return "PipelineRasterizationDepthClipStateCreateInfoEXT";
      case StructureType::eHdrMetadataEXT: return "HdrMetadataEXT";
      case StructureType::ePhysicalDeviceRelaxedLineRasterizationFeaturesIMG: return "PhysicalDeviceRelaxedLineRasterizationFeaturesIMG";
      case StructureType::eSharedPresentSurfaceCapabilitiesKHR: return "SharedPresentSurfaceCapabilitiesKHR";
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      case StructureType::eImportFenceWin32HandleInfoKHR: return "ImportFenceWin32HandleInfoKHR";
      case StructureType::eExportFenceWin32HandleInfoKHR: return "ExportFenceWin32HandleInfoKHR";
      case StructureType::eFenceGetWin32HandleInfoKHR: return "FenceGetWin32HandleInfoKHR";
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      case StructureType::eImportFenceFdInfoKHR: return "ImportFenceFdInfoKHR";
      case StructureType::eFenceGetFdInfoKHR: return "FenceGetFdInfoKHR";
      case StructureType::ePhysicalDevicePerformanceQueryFeaturesKHR: return "PhysicalDevicePerformanceQueryFeaturesKHR";
      case StructureType::ePhysicalDevicePerformanceQueryPropertiesKHR: return "PhysicalDevicePerformanceQueryPropertiesKHR";
      case StructureType::eQueryPoolPerformanceCreateInfoKHR: return "QueryPoolPerformanceCreateInfoKHR";
      case StructureType::ePerformanceQuerySubmitInfoKHR: return "PerformanceQuerySubmitInfoKHR";
      case StructureType::eAcquireProfilingLockInfoKHR: return "AcquireProfilingLockInfoKHR";
      case StructureType::ePerformanceCounterKHR: return "PerformanceCounterKHR";
      case StructureType::ePerformanceCounterDescriptionKHR: return "PerformanceCounterDescriptionKHR";
      case StructureType::ePhysicalDeviceSurfaceInfo2KHR: return "PhysicalDeviceSurfaceInfo2KHR";
      case StructureType::eSurfaceCapabilities2KHR: return "SurfaceCapabilities2KHR";
      case StructureType::eSurfaceFormat2KHR: return "SurfaceFormat2KHR";
      case StructureType::eDisplayProperties2KHR: return "DisplayProperties2KHR";
      case StructureType::eDisplayPlaneProperties2KHR: return "DisplayPlaneProperties2KHR";
      case StructureType::eDisplayModeProperties2KHR: return "DisplayModeProperties2KHR";
      case StructureType::eDisplayPlaneInfo2KHR: return "DisplayPlaneInfo2KHR";
      case StructureType::eDisplayPlaneCapabilities2KHR: return "DisplayPlaneCapabilities2KHR";
#if defined( VK_USE_PLATFORM_IOS_MVK )
      case StructureType::eIosSurfaceCreateInfoMVK: return "IosSurfaceCreateInfoMVK";
#endif /*VK_USE_PLATFORM_IOS_MVK*/
#if defined( VK_USE_PLATFORM_MACOS_MVK )
      case StructureType::eMacosSurfaceCreateInfoMVK: return "MacosSurfaceCreateInfoMVK";
#endif /*VK_USE_PLATFORM_MACOS_MVK*/
      case StructureType::eDebugUtilsObjectNameInfoEXT: return "DebugUtilsObjectNameInfoEXT";
      case StructureType::eDebugUtilsObjectTagInfoEXT: return "DebugUtilsObjectTagInfoEXT";
      case StructureType::eDebugUtilsLabelEXT: return "DebugUtilsLabelEXT";
      case StructureType::eDebugUtilsMessengerCallbackDataEXT: return "DebugUtilsMessengerCallbackDataEXT";
      case StructureType::eDebugUtilsMessengerCreateInfoEXT: return "DebugUtilsMessengerCreateInfoEXT";
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
      case StructureType::eAndroidHardwareBufferUsageANDROID: return "AndroidHardwareBufferUsageANDROID";
      case StructureType::eAndroidHardwareBufferPropertiesANDROID: return "AndroidHardwareBufferPropertiesANDROID";
      case StructureType::eAndroidHardwareBufferFormatPropertiesANDROID: return "AndroidHardwareBufferFormatPropertiesANDROID";
      case StructureType::eImportAndroidHardwareBufferInfoANDROID: return "ImportAndroidHardwareBufferInfoANDROID";
      case StructureType::eMemoryGetAndroidHardwareBufferInfoANDROID: return "MemoryGetAndroidHardwareBufferInfoANDROID";
      case StructureType::eExternalFormatANDROID: return "ExternalFormatANDROID";
      case StructureType::eAndroidHardwareBufferFormatProperties2ANDROID: return "AndroidHardwareBufferFormatProperties2ANDROID";
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case StructureType::ePhysicalDeviceShaderEnqueueFeaturesAMDX: return "PhysicalDeviceShaderEnqueueFeaturesAMDX";
      case StructureType::ePhysicalDeviceShaderEnqueuePropertiesAMDX: return "PhysicalDeviceShaderEnqueuePropertiesAMDX";
      case StructureType::eExecutionGraphPipelineScratchSizeAMDX: return "ExecutionGraphPipelineScratchSizeAMDX";
      case StructureType::eExecutionGraphPipelineCreateInfoAMDX: return "ExecutionGraphPipelineCreateInfoAMDX";
      case StructureType::ePipelineShaderStageNodeCreateInfoAMDX: return "PipelineShaderStageNodeCreateInfoAMDX";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      case StructureType::eSampleLocationsInfoEXT: return "SampleLocationsInfoEXT";
      case StructureType::eRenderPassSampleLocationsBeginInfoEXT: return "RenderPassSampleLocationsBeginInfoEXT";
      case StructureType::ePipelineSampleLocationsStateCreateInfoEXT: return "PipelineSampleLocationsStateCreateInfoEXT";
      case StructureType::ePhysicalDeviceSampleLocationsPropertiesEXT: return "PhysicalDeviceSampleLocationsPropertiesEXT";
      case StructureType::eMultisamplePropertiesEXT: return "MultisamplePropertiesEXT";
      case StructureType::ePhysicalDeviceBlendOperationAdvancedFeaturesEXT: return "PhysicalDeviceBlendOperationAdvancedFeaturesEXT";
      case StructureType::ePhysicalDeviceBlendOperationAdvancedPropertiesEXT: return "PhysicalDeviceBlendOperationAdvancedPropertiesEXT";
      case StructureType::ePipelineColorBlendAdvancedStateCreateInfoEXT: return "PipelineColorBlendAdvancedStateCreateInfoEXT";
      case StructureType::ePipelineCoverageToColorStateCreateInfoNV: return "PipelineCoverageToColorStateCreateInfoNV";
      case StructureType::eWriteDescriptorSetAccelerationStructureKHR: return "WriteDescriptorSetAccelerationStructureKHR";
      case StructureType::eAccelerationStructureBuildGeometryInfoKHR: return "AccelerationStructureBuildGeometryInfoKHR";
      case StructureType::eAccelerationStructureDeviceAddressInfoKHR: return "AccelerationStructureDeviceAddressInfoKHR";
      case StructureType::eAccelerationStructureGeometryAabbsDataKHR: return "AccelerationStructureGeometryAabbsDataKHR";
      case StructureType::eAccelerationStructureGeometryInstancesDataKHR: return "AccelerationStructureGeometryInstancesDataKHR";
      case StructureType::eAccelerationStructureGeometryTrianglesDataKHR: return "AccelerationStructureGeometryTrianglesDataKHR";
      case StructureType::eAccelerationStructureGeometryKHR: return "AccelerationStructureGeometryKHR";
      case StructureType::eAccelerationStructureVersionInfoKHR: return "AccelerationStructureVersionInfoKHR";
      case StructureType::eCopyAccelerationStructureInfoKHR: return "CopyAccelerationStructureInfoKHR";
      case StructureType::eCopyAccelerationStructureToMemoryInfoKHR: return "CopyAccelerationStructureToMemoryInfoKHR";
      case StructureType::eCopyMemoryToAccelerationStructureInfoKHR: return "CopyMemoryToAccelerationStructureInfoKHR";
      case StructureType::ePhysicalDeviceAccelerationStructureFeaturesKHR: return "PhysicalDeviceAccelerationStructureFeaturesKHR";
      case StructureType::ePhysicalDeviceAccelerationStructurePropertiesKHR: return "PhysicalDeviceAccelerationStructurePropertiesKHR";
      case StructureType::eAccelerationStructureCreateInfoKHR: return "AccelerationStructureCreateInfoKHR";
      case StructureType::eAccelerationStructureBuildSizesInfoKHR: return "AccelerationStructureBuildSizesInfoKHR";
      case StructureType::ePhysicalDeviceRayTracingPipelineFeaturesKHR: return "PhysicalDeviceRayTracingPipelineFeaturesKHR";
      case StructureType::ePhysicalDeviceRayTracingPipelinePropertiesKHR: return "PhysicalDeviceRayTracingPipelinePropertiesKHR";
      case StructureType::eRayTracingPipelineCreateInfoKHR: return "RayTracingPipelineCreateInfoKHR";
      case StructureType::eRayTracingShaderGroupCreateInfoKHR: return "RayTracingShaderGroupCreateInfoKHR";
      case StructureType::eRayTracingPipelineInterfaceCreateInfoKHR: return "RayTracingPipelineInterfaceCreateInfoKHR";
      case StructureType::ePhysicalDeviceRayQueryFeaturesKHR: return "PhysicalDeviceRayQueryFeaturesKHR";
      case StructureType::ePipelineCoverageModulationStateCreateInfoNV: return "PipelineCoverageModulationStateCreateInfoNV";
      case StructureType::ePhysicalDeviceShaderSmBuiltinsFeaturesNV: return "PhysicalDeviceShaderSmBuiltinsFeaturesNV";
      case StructureType::ePhysicalDeviceShaderSmBuiltinsPropertiesNV: return "PhysicalDeviceShaderSmBuiltinsPropertiesNV";
      case StructureType::eDrmFormatModifierPropertiesListEXT: return "DrmFormatModifierPropertiesListEXT";
      case StructureType::ePhysicalDeviceImageDrmFormatModifierInfoEXT: return "PhysicalDeviceImageDrmFormatModifierInfoEXT";
      case StructureType::eImageDrmFormatModifierListCreateInfoEXT: return "ImageDrmFormatModifierListCreateInfoEXT";
      case StructureType::eImageDrmFormatModifierExplicitCreateInfoEXT: return "ImageDrmFormatModifierExplicitCreateInfoEXT";
      case StructureType::eImageDrmFormatModifierPropertiesEXT: return "ImageDrmFormatModifierPropertiesEXT";
      case StructureType::eDrmFormatModifierPropertiesList2EXT: return "DrmFormatModifierPropertiesList2EXT";
      case StructureType::eValidationCacheCreateInfoEXT: return "ValidationCacheCreateInfoEXT";
      case StructureType::eShaderModuleValidationCacheCreateInfoEXT: return "ShaderModuleValidationCacheCreateInfoEXT";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case StructureType::ePhysicalDevicePortabilitySubsetFeaturesKHR: return "PhysicalDevicePortabilitySubsetFeaturesKHR";
      case StructureType::ePhysicalDevicePortabilitySubsetPropertiesKHR: return "PhysicalDevicePortabilitySubsetPropertiesKHR";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      case StructureType::ePipelineViewportShadingRateImageStateCreateInfoNV: return "PipelineViewportShadingRateImageStateCreateInfoNV";
      case StructureType::ePhysicalDeviceShadingRateImageFeaturesNV: return "PhysicalDeviceShadingRateImageFeaturesNV";
      case StructureType::ePhysicalDeviceShadingRateImagePropertiesNV: return "PhysicalDeviceShadingRateImagePropertiesNV";
      case StructureType::ePipelineViewportCoarseSampleOrderStateCreateInfoNV: return "PipelineViewportCoarseSampleOrderStateCreateInfoNV";
      case StructureType::eRayTracingPipelineCreateInfoNV: return "RayTracingPipelineCreateInfoNV";
      case StructureType::eAccelerationStructureCreateInfoNV: return "AccelerationStructureCreateInfoNV";
      case StructureType::eGeometryNV: return "GeometryNV";
      case StructureType::eGeometryTrianglesNV: return "GeometryTrianglesNV";
      case StructureType::eGeometryAabbNV: return "GeometryAabbNV";
      case StructureType::eBindAccelerationStructureMemoryInfoNV: return "BindAccelerationStructureMemoryInfoNV";
      case StructureType::eWriteDescriptorSetAccelerationStructureNV: return "WriteDescriptorSetAccelerationStructureNV";
      case StructureType::eAccelerationStructureMemoryRequirementsInfoNV: return "AccelerationStructureMemoryRequirementsInfoNV";
      case StructureType::ePhysicalDeviceRayTracingPropertiesNV: return "PhysicalDeviceRayTracingPropertiesNV";
      case StructureType::eRayTracingShaderGroupCreateInfoNV: return "RayTracingShaderGroupCreateInfoNV";
      case StructureType::eAccelerationStructureInfoNV: return "AccelerationStructureInfoNV";
      case StructureType::ePhysicalDeviceRepresentativeFragmentTestFeaturesNV: return "PhysicalDeviceRepresentativeFragmentTestFeaturesNV";
      case StructureType::ePipelineRepresentativeFragmentTestStateCreateInfoNV: return "PipelineRepresentativeFragmentTestStateCreateInfoNV";
      case StructureType::ePhysicalDeviceImageViewImageFormatInfoEXT: return "PhysicalDeviceImageViewImageFormatInfoEXT";
      case StructureType::eFilterCubicImageViewImageFormatPropertiesEXT: return "FilterCubicImageViewImageFormatPropertiesEXT";
      case StructureType::eImportMemoryHostPointerInfoEXT: return "ImportMemoryHostPointerInfoEXT";
      case StructureType::eMemoryHostPointerPropertiesEXT: return "MemoryHostPointerPropertiesEXT";
      case StructureType::ePhysicalDeviceExternalMemoryHostPropertiesEXT: return "PhysicalDeviceExternalMemoryHostPropertiesEXT";
      case StructureType::ePhysicalDeviceShaderClockFeaturesKHR: return "PhysicalDeviceShaderClockFeaturesKHR";
      case StructureType::ePipelineCompilerControlCreateInfoAMD: return "PipelineCompilerControlCreateInfoAMD";
      case StructureType::ePhysicalDeviceShaderCorePropertiesAMD: return "PhysicalDeviceShaderCorePropertiesAMD";
      case StructureType::eVideoDecodeH265CapabilitiesKHR: return "VideoDecodeH265CapabilitiesKHR";
      case StructureType::eVideoDecodeH265SessionParametersCreateInfoKHR: return "VideoDecodeH265SessionParametersCreateInfoKHR";
      case StructureType::eVideoDecodeH265SessionParametersAddInfoKHR: return "VideoDecodeH265SessionParametersAddInfoKHR";
      case StructureType::eVideoDecodeH265ProfileInfoKHR: return "VideoDecodeH265ProfileInfoKHR";
      case StructureType::eVideoDecodeH265PictureInfoKHR: return "VideoDecodeH265PictureInfoKHR";
      case StructureType::eVideoDecodeH265DpbSlotInfoKHR: return "VideoDecodeH265DpbSlotInfoKHR";
      case StructureType::eDeviceQueueGlobalPriorityCreateInfoKHR: return "DeviceQueueGlobalPriorityCreateInfoKHR";
      case StructureType::ePhysicalDeviceGlobalPriorityQueryFeaturesKHR: return "PhysicalDeviceGlobalPriorityQueryFeaturesKHR";
      case StructureType::eQueueFamilyGlobalPriorityPropertiesKHR: return "QueueFamilyGlobalPriorityPropertiesKHR";
      case StructureType::eDeviceMemoryOverallocationCreateInfoAMD: return "DeviceMemoryOverallocationCreateInfoAMD";
      case StructureType::ePhysicalDeviceVertexAttributeDivisorPropertiesEXT: return "PhysicalDeviceVertexAttributeDivisorPropertiesEXT";
#if defined( VK_USE_PLATFORM_GGP )
      case StructureType::ePresentFrameTokenGGP: return "PresentFrameTokenGGP";
#endif /*VK_USE_PLATFORM_GGP*/
      case StructureType::ePhysicalDeviceComputeShaderDerivativesFeaturesNV: return "PhysicalDeviceComputeShaderDerivativesFeaturesNV";
      case StructureType::ePhysicalDeviceMeshShaderFeaturesNV: return "PhysicalDeviceMeshShaderFeaturesNV";
      case StructureType::ePhysicalDeviceMeshShaderPropertiesNV: return "PhysicalDeviceMeshShaderPropertiesNV";
      case StructureType::ePhysicalDeviceShaderImageFootprintFeaturesNV: return "PhysicalDeviceShaderImageFootprintFeaturesNV";
      case StructureType::ePipelineViewportExclusiveScissorStateCreateInfoNV: return "PipelineViewportExclusiveScissorStateCreateInfoNV";
      case StructureType::ePhysicalDeviceExclusiveScissorFeaturesNV: return "PhysicalDeviceExclusiveScissorFeaturesNV";
      case StructureType::eCheckpointDataNV: return "CheckpointDataNV";
      case StructureType::eQueueFamilyCheckpointPropertiesNV: return "QueueFamilyCheckpointPropertiesNV";
      case StructureType::ePhysicalDeviceShaderIntegerFunctions2FeaturesINTEL: return "PhysicalDeviceShaderIntegerFunctions2FeaturesINTEL";
      case StructureType::eQueryPoolPerformanceQueryCreateInfoINTEL: return "QueryPoolPerformanceQueryCreateInfoINTEL";
      case StructureType::eInitializePerformanceApiInfoINTEL: return "InitializePerformanceApiInfoINTEL";
      case StructureType::ePerformanceMarkerInfoINTEL: return "PerformanceMarkerInfoINTEL";
      case StructureType::ePerformanceStreamMarkerInfoINTEL: return "PerformanceStreamMarkerInfoINTEL";
      case StructureType::ePerformanceOverrideInfoINTEL: return "PerformanceOverrideInfoINTEL";
      case StructureType::ePerformanceConfigurationAcquireInfoINTEL: return "PerformanceConfigurationAcquireInfoINTEL";
      case StructureType::ePhysicalDevicePciBusInfoPropertiesEXT: return "PhysicalDevicePciBusInfoPropertiesEXT";
      case StructureType::eDisplayNativeHdrSurfaceCapabilitiesAMD: return "DisplayNativeHdrSurfaceCapabilitiesAMD";
      case StructureType::eSwapchainDisplayNativeHdrCreateInfoAMD: return "SwapchainDisplayNativeHdrCreateInfoAMD";
#if defined( VK_USE_PLATFORM_FUCHSIA )
      case StructureType::eImagepipeSurfaceCreateInfoFUCHSIA: return "ImagepipeSurfaceCreateInfoFUCHSIA";
#endif /*VK_USE_PLATFORM_FUCHSIA*/
#if defined( VK_USE_PLATFORM_METAL_EXT )
      case StructureType::eMetalSurfaceCreateInfoEXT: return "MetalSurfaceCreateInfoEXT";
#endif /*VK_USE_PLATFORM_METAL_EXT*/
      case StructureType::ePhysicalDeviceFragmentDensityMapFeaturesEXT: return "PhysicalDeviceFragmentDensityMapFeaturesEXT";
      case StructureType::ePhysicalDeviceFragmentDensityMapPropertiesEXT: return "PhysicalDeviceFragmentDensityMapPropertiesEXT";
      case StructureType::eRenderPassFragmentDensityMapCreateInfoEXT: return "RenderPassFragmentDensityMapCreateInfoEXT";
      case StructureType::eFragmentShadingRateAttachmentInfoKHR: return "FragmentShadingRateAttachmentInfoKHR";
      case StructureType::ePipelineFragmentShadingRateStateCreateInfoKHR: return "PipelineFragmentShadingRateStateCreateInfoKHR";
      case StructureType::ePhysicalDeviceFragmentShadingRatePropertiesKHR: return "PhysicalDeviceFragmentShadingRatePropertiesKHR";
      case StructureType::ePhysicalDeviceFragmentShadingRateFeaturesKHR: return "PhysicalDeviceFragmentShadingRateFeaturesKHR";
      case StructureType::ePhysicalDeviceFragmentShadingRateKHR: return "PhysicalDeviceFragmentShadingRateKHR";
      case StructureType::ePhysicalDeviceShaderCoreProperties2AMD: return "PhysicalDeviceShaderCoreProperties2AMD";
      case StructureType::ePhysicalDeviceCoherentMemoryFeaturesAMD: return "PhysicalDeviceCoherentMemoryFeaturesAMD";
      case StructureType::ePhysicalDeviceDynamicRenderingLocalReadFeaturesKHR: return "PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR";
      case StructureType::eRenderingAttachmentLocationInfoKHR: return "RenderingAttachmentLocationInfoKHR";
      case StructureType::eRenderingInputAttachmentIndexInfoKHR: return "RenderingInputAttachmentIndexInfoKHR";
      case StructureType::ePhysicalDeviceShaderImageAtomicInt64FeaturesEXT: return "PhysicalDeviceShaderImageAtomicInt64FeaturesEXT";
      case StructureType::ePhysicalDeviceShaderQuadControlFeaturesKHR: return "PhysicalDeviceShaderQuadControlFeaturesKHR";
      case StructureType::ePhysicalDeviceMemoryBudgetPropertiesEXT: return "PhysicalDeviceMemoryBudgetPropertiesEXT";
      case StructureType::ePhysicalDeviceMemoryPriorityFeaturesEXT: return "PhysicalDeviceMemoryPriorityFeaturesEXT";
      case StructureType::eMemoryPriorityAllocateInfoEXT: return "MemoryPriorityAllocateInfoEXT";
      case StructureType::eSurfaceProtectedCapabilitiesKHR: return "SurfaceProtectedCapabilitiesKHR";
      case StructureType::ePhysicalDeviceDedicatedAllocationImageAliasingFeaturesNV: return "PhysicalDeviceDedicatedAllocationImageAliasingFeaturesNV";
      case StructureType::ePhysicalDeviceBufferDeviceAddressFeaturesEXT: return "PhysicalDeviceBufferDeviceAddressFeaturesEXT";
      case StructureType::eBufferDeviceAddressCreateInfoEXT: return "BufferDeviceAddressCreateInfoEXT";
      case StructureType::eValidationFeaturesEXT: return "ValidationFeaturesEXT";
      case StructureType::ePhysicalDevicePresentWaitFeaturesKHR: return "PhysicalDevicePresentWaitFeaturesKHR";
      case StructureType::ePhysicalDeviceCooperativeMatrixFeaturesNV: return "PhysicalDeviceCooperativeMatrixFeaturesNV";
      case StructureType::eCooperativeMatrixPropertiesNV: return "CooperativeMatrixPropertiesNV";
      case StructureType::ePhysicalDeviceCooperativeMatrixPropertiesNV: return "PhysicalDeviceCooperativeMatrixPropertiesNV";
      case StructureType::ePhysicalDeviceCoverageReductionModeFeaturesNV: return "PhysicalDeviceCoverageReductionModeFeaturesNV";
      case StructureType::ePipelineCoverageReductionStateCreateInfoNV: return "PipelineCoverageReductionStateCreateInfoNV";
      case StructureType::eFramebufferMixedSamplesCombinationNV: return "FramebufferMixedSamplesCombinationNV";
      case StructureType::ePhysicalDeviceFragmentShaderInterlockFeaturesEXT: return "PhysicalDeviceFragmentShaderInterlockFeaturesEXT";
      case StructureType::ePhysicalDeviceYcbcrImageArraysFeaturesEXT: return "PhysicalDeviceYcbcrImageArraysFeaturesEXT";
      case StructureType::ePhysicalDeviceProvokingVertexFeaturesEXT: return "PhysicalDeviceProvokingVertexFeaturesEXT";
      case StructureType::ePipelineRasterizationProvokingVertexStateCreateInfoEXT: return "PipelineRasterizationProvokingVertexStateCreateInfoEXT";
      case StructureType::ePhysicalDeviceProvokingVertexPropertiesEXT: return "PhysicalDeviceProvokingVertexPropertiesEXT";
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      case StructureType::eSurfaceFullScreenExclusiveInfoEXT: return "SurfaceFullScreenExclusiveInfoEXT";
      case StructureType::eSurfaceCapabilitiesFullScreenExclusiveEXT: return "SurfaceCapabilitiesFullScreenExclusiveEXT";
      case StructureType::eSurfaceFullScreenExclusiveWin32InfoEXT: return "SurfaceFullScreenExclusiveWin32InfoEXT";
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      case StructureType::eHeadlessSurfaceCreateInfoEXT: return "HeadlessSurfaceCreateInfoEXT";
      case StructureType::ePhysicalDeviceShaderAtomicFloatFeaturesEXT: return "PhysicalDeviceShaderAtomicFloatFeaturesEXT";
      case StructureType::ePhysicalDeviceExtendedDynamicStateFeaturesEXT: return "PhysicalDeviceExtendedDynamicStateFeaturesEXT";
      case StructureType::ePhysicalDevicePipelineExecutablePropertiesFeaturesKHR: return "PhysicalDevicePipelineExecutablePropertiesFeaturesKHR";
      case StructureType::ePipelineInfoKHR: return "PipelineInfoKHR";
      case StructureType::ePipelineExecutablePropertiesKHR: return "PipelineExecutablePropertiesKHR";
      case StructureType::ePipelineExecutableInfoKHR: return "PipelineExecutableInfoKHR";
      case StructureType::ePipelineExecutableStatisticKHR: return "PipelineExecutableStatisticKHR";
      case StructureType::ePipelineExecutableInternalRepresentationKHR: return "PipelineExecutableInternalRepresentationKHR";
      case StructureType::ePhysicalDeviceHostImageCopyFeaturesEXT: return "PhysicalDeviceHostImageCopyFeaturesEXT";
      case StructureType::ePhysicalDeviceHostImageCopyPropertiesEXT: return "PhysicalDeviceHostImageCopyPropertiesEXT";
      case StructureType::eMemoryToImageCopyEXT: return "MemoryToImageCopyEXT";
      case StructureType::eImageToMemoryCopyEXT: return "ImageToMemoryCopyEXT";
      case StructureType::eCopyImageToMemoryInfoEXT: return "CopyImageToMemoryInfoEXT";
      case StructureType::eCopyMemoryToImageInfoEXT: return "CopyMemoryToImageInfoEXT";
      case StructureType::eHostImageLayoutTransitionInfoEXT: return "HostImageLayoutTransitionInfoEXT";
      case StructureType::eCopyImageToImageInfoEXT: return "CopyImageToImageInfoEXT";
      case StructureType::eSubresourceHostMemcpySizeEXT: return "SubresourceHostMemcpySizeEXT";
      case StructureType::eHostImageCopyDevicePerformanceQueryEXT: return "HostImageCopyDevicePerformanceQueryEXT";
      case StructureType::eMemoryMapInfoKHR: return "MemoryMapInfoKHR";
      case StructureType::eMemoryUnmapInfoKHR: return "MemoryUnmapInfoKHR";
      case StructureType::ePhysicalDeviceMapMemoryPlacedFeaturesEXT: return "PhysicalDeviceMapMemoryPlacedFeaturesEXT";
      case StructureType::ePhysicalDeviceMapMemoryPlacedPropertiesEXT: return "PhysicalDeviceMapMemoryPlacedPropertiesEXT";
      case StructureType::eMemoryMapPlacedInfoEXT: return "MemoryMapPlacedInfoEXT";
      case StructureType::ePhysicalDeviceShaderAtomicFloat2FeaturesEXT: return "PhysicalDeviceShaderAtomicFloat2FeaturesEXT";
      case StructureType::eSurfacePresentModeEXT: return "SurfacePresentModeEXT";
      case StructureType::eSurfacePresentScalingCapabilitiesEXT: return "SurfacePresentScalingCapabilitiesEXT";
      case StructureType::eSurfacePresentModeCompatibilityEXT: return "SurfacePresentModeCompatibilityEXT";
      case StructureType::ePhysicalDeviceSwapchainMaintenance1FeaturesEXT: return "PhysicalDeviceSwapchainMaintenance1FeaturesEXT";
      case StructureType::eSwapchainPresentFenceInfoEXT: return "SwapchainPresentFenceInfoEXT";
      case StructureType::eSwapchainPresentModesCreateInfoEXT: return "SwapchainPresentModesCreateInfoEXT";
      case StructureType::eSwapchainPresentModeInfoEXT: return "SwapchainPresentModeInfoEXT";
      case StructureType::eSwapchainPresentScalingCreateInfoEXT: return "SwapchainPresentScalingCreateInfoEXT";
      case StructureType::eReleaseSwapchainImagesInfoEXT: return "ReleaseSwapchainImagesInfoEXT";
      case StructureType::ePhysicalDeviceDeviceGeneratedCommandsPropertiesNV: return "PhysicalDeviceDeviceGeneratedCommandsPropertiesNV";
      case StructureType::eGraphicsShaderGroupCreateInfoNV: return "GraphicsShaderGroupCreateInfoNV";
      case StructureType::eGraphicsPipelineShaderGroupsCreateInfoNV: return "GraphicsPipelineShaderGroupsCreateInfoNV";
      case StructureType::eIndirectCommandsLayoutTokenNV: return "IndirectCommandsLayoutTokenNV";
      case StructureType::eIndirectCommandsLayoutCreateInfoNV: return "IndirectCommandsLayoutCreateInfoNV";
      case StructureType::eGeneratedCommandsInfoNV: return "GeneratedCommandsInfoNV";
      case StructureType::eGeneratedCommandsMemoryRequirementsInfoNV: return "GeneratedCommandsMemoryRequirementsInfoNV";
      case StructureType::ePhysicalDeviceDeviceGeneratedCommandsFeaturesNV: return "PhysicalDeviceDeviceGeneratedCommandsFeaturesNV";
      case StructureType::ePhysicalDeviceInheritedViewportScissorFeaturesNV: return "PhysicalDeviceInheritedViewportScissorFeaturesNV";
      case StructureType::eCommandBufferInheritanceViewportScissorInfoNV: return "CommandBufferInheritanceViewportScissorInfoNV";
      case StructureType::ePhysicalDeviceTexelBufferAlignmentFeaturesEXT: return "PhysicalDeviceTexelBufferAlignmentFeaturesEXT";
      case StructureType::eCommandBufferInheritanceRenderPassTransformInfoQCOM: return "CommandBufferInheritanceRenderPassTransformInfoQCOM";
      case StructureType::eRenderPassTransformBeginInfoQCOM: return "RenderPassTransformBeginInfoQCOM";
      case StructureType::ePhysicalDeviceDepthBiasControlFeaturesEXT: return "PhysicalDeviceDepthBiasControlFeaturesEXT";
      case StructureType::eDepthBiasInfoEXT: return "DepthBiasInfoEXT";
      case StructureType::eDepthBiasRepresentationInfoEXT: return "DepthBiasRepresentationInfoEXT";
      case StructureType::ePhysicalDeviceDeviceMemoryReportFeaturesEXT: return "PhysicalDeviceDeviceMemoryReportFeaturesEXT";
      case StructureType::eDeviceDeviceMemoryReportCreateInfoEXT: return "DeviceDeviceMemoryReportCreateInfoEXT";
      case StructureType::eDeviceMemoryReportCallbackDataEXT: return "DeviceMemoryReportCallbackDataEXT";
      case StructureType::ePhysicalDeviceRobustness2FeaturesEXT: return "PhysicalDeviceRobustness2FeaturesEXT";
      case StructureType::ePhysicalDeviceRobustness2PropertiesEXT: return "PhysicalDeviceRobustness2PropertiesEXT";
      case StructureType::eSamplerCustomBorderColorCreateInfoEXT: return "SamplerCustomBorderColorCreateInfoEXT";
      case StructureType::ePhysicalDeviceCustomBorderColorPropertiesEXT: return "PhysicalDeviceCustomBorderColorPropertiesEXT";
      case StructureType::ePhysicalDeviceCustomBorderColorFeaturesEXT: return "PhysicalDeviceCustomBorderColorFeaturesEXT";
      case StructureType::ePipelineLibraryCreateInfoKHR: return "PipelineLibraryCreateInfoKHR";
      case StructureType::ePhysicalDevicePresentBarrierFeaturesNV: return "PhysicalDevicePresentBarrierFeaturesNV";
      case StructureType::eSurfaceCapabilitiesPresentBarrierNV: return "SurfaceCapabilitiesPresentBarrierNV";
      case StructureType::eSwapchainPresentBarrierCreateInfoNV: return "SwapchainPresentBarrierCreateInfoNV";
      case StructureType::ePresentIdKHR: return "PresentIdKHR";
      case StructureType::ePhysicalDevicePresentIdFeaturesKHR: return "PhysicalDevicePresentIdFeaturesKHR";
      case StructureType::eVideoEncodeInfoKHR: return "VideoEncodeInfoKHR";
      case StructureType::eVideoEncodeRateControlInfoKHR: return "VideoEncodeRateControlInfoKHR";
      case StructureType::eVideoEncodeRateControlLayerInfoKHR: return "VideoEncodeRateControlLayerInfoKHR";
      case StructureType::eVideoEncodeCapabilitiesKHR: return "VideoEncodeCapabilitiesKHR";
      case StructureType::eVideoEncodeUsageInfoKHR: return "VideoEncodeUsageInfoKHR";
      case StructureType::eQueryPoolVideoEncodeFeedbackCreateInfoKHR: return "QueryPoolVideoEncodeFeedbackCreateInfoKHR";
      case StructureType::ePhysicalDeviceVideoEncodeQualityLevelInfoKHR: return "PhysicalDeviceVideoEncodeQualityLevelInfoKHR";
      case StructureType::eVideoEncodeQualityLevelPropertiesKHR: return "VideoEncodeQualityLevelPropertiesKHR";
      case StructureType::eVideoEncodeQualityLevelInfoKHR: return "VideoEncodeQualityLevelInfoKHR";
      case StructureType::eVideoEncodeSessionParametersGetInfoKHR: return "VideoEncodeSessionParametersGetInfoKHR";
      case StructureType::eVideoEncodeSessionParametersFeedbackInfoKHR: return "VideoEncodeSessionParametersFeedbackInfoKHR";
      case StructureType::ePhysicalDeviceDiagnosticsConfigFeaturesNV: return "PhysicalDeviceDiagnosticsConfigFeaturesNV";
      case StructureType::eDeviceDiagnosticsConfigCreateInfoNV: return "DeviceDiagnosticsConfigCreateInfoNV";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case StructureType::eCudaModuleCreateInfoNV: return "CudaModuleCreateInfoNV";
      case StructureType::eCudaFunctionCreateInfoNV: return "CudaFunctionCreateInfoNV";
      case StructureType::eCudaLaunchInfoNV: return "CudaLaunchInfoNV";
      case StructureType::ePhysicalDeviceCudaKernelLaunchFeaturesNV: return "PhysicalDeviceCudaKernelLaunchFeaturesNV";
      case StructureType::ePhysicalDeviceCudaKernelLaunchPropertiesNV: return "PhysicalDeviceCudaKernelLaunchPropertiesNV";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      case StructureType::eQueryLowLatencySupportNV: return "QueryLowLatencySupportNV";
#if defined( VK_USE_PLATFORM_METAL_EXT )
      case StructureType::eExportMetalObjectCreateInfoEXT: return "ExportMetalObjectCreateInfoEXT";
      case StructureType::eExportMetalObjectsInfoEXT: return "ExportMetalObjectsInfoEXT";
      case StructureType::eExportMetalDeviceInfoEXT: return "ExportMetalDeviceInfoEXT";
      case StructureType::eExportMetalCommandQueueInfoEXT: return "ExportMetalCommandQueueInfoEXT";
      case StructureType::eExportMetalBufferInfoEXT: return "ExportMetalBufferInfoEXT";
      case StructureType::eImportMetalBufferInfoEXT: return "ImportMetalBufferInfoEXT";
      case StructureType::eExportMetalTextureInfoEXT: return "ExportMetalTextureInfoEXT";
      case StructureType::eImportMetalTextureInfoEXT: return "ImportMetalTextureInfoEXT";
      case StructureType::eExportMetalIoSurfaceInfoEXT: return "ExportMetalIoSurfaceInfoEXT";
      case StructureType::eImportMetalIoSurfaceInfoEXT: return "ImportMetalIoSurfaceInfoEXT";
      case StructureType::eExportMetalSharedEventInfoEXT: return "ExportMetalSharedEventInfoEXT";
      case StructureType::eImportMetalSharedEventInfoEXT: return "ImportMetalSharedEventInfoEXT";
#endif /*VK_USE_PLATFORM_METAL_EXT*/
      case StructureType::eQueueFamilyCheckpointProperties2NV: return "QueueFamilyCheckpointProperties2NV";
      case StructureType::eCheckpointData2NV: return "CheckpointData2NV";
      case StructureType::ePhysicalDeviceDescriptorBufferPropertiesEXT: return "PhysicalDeviceDescriptorBufferPropertiesEXT";
      case StructureType::ePhysicalDeviceDescriptorBufferDensityMapPropertiesEXT: return "PhysicalDeviceDescriptorBufferDensityMapPropertiesEXT";
      case StructureType::ePhysicalDeviceDescriptorBufferFeaturesEXT: return "PhysicalDeviceDescriptorBufferFeaturesEXT";
      case StructureType::eDescriptorAddressInfoEXT: return "DescriptorAddressInfoEXT";
      case StructureType::eDescriptorGetInfoEXT: return "DescriptorGetInfoEXT";
      case StructureType::eBufferCaptureDescriptorDataInfoEXT: return "BufferCaptureDescriptorDataInfoEXT";
      case StructureType::eImageCaptureDescriptorDataInfoEXT: return "ImageCaptureDescriptorDataInfoEXT";
      case StructureType::eImageViewCaptureDescriptorDataInfoEXT: return "ImageViewCaptureDescriptorDataInfoEXT";
      case StructureType::eSamplerCaptureDescriptorDataInfoEXT: return "SamplerCaptureDescriptorDataInfoEXT";
      case StructureType::eOpaqueCaptureDescriptorDataCreateInfoEXT: return "OpaqueCaptureDescriptorDataCreateInfoEXT";
      case StructureType::eDescriptorBufferBindingInfoEXT: return "DescriptorBufferBindingInfoEXT";
      case StructureType::eDescriptorBufferBindingPushDescriptorBufferHandleEXT: return "DescriptorBufferBindingPushDescriptorBufferHandleEXT";
      case StructureType::eAccelerationStructureCaptureDescriptorDataInfoEXT: return "AccelerationStructureCaptureDescriptorDataInfoEXT";
      case StructureType::ePhysicalDeviceGraphicsPipelineLibraryFeaturesEXT: return "PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT";
      case StructureType::ePhysicalDeviceGraphicsPipelineLibraryPropertiesEXT: return "PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT";
      case StructureType::eGraphicsPipelineLibraryCreateInfoEXT: return "GraphicsPipelineLibraryCreateInfoEXT";
      case StructureType::ePhysicalDeviceShaderEarlyAndLateFragmentTestsFeaturesAMD: return "PhysicalDeviceShaderEarlyAndLateFragmentTestsFeaturesAMD";
      case StructureType::ePhysicalDeviceFragmentShaderBarycentricFeaturesKHR: return "PhysicalDeviceFragmentShaderBarycentricFeaturesKHR";
      case StructureType::ePhysicalDeviceFragmentShaderBarycentricPropertiesKHR: return "PhysicalDeviceFragmentShaderBarycentricPropertiesKHR";
      case StructureType::ePhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR: return "PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR";
      case StructureType::ePhysicalDeviceFragmentShadingRateEnumsPropertiesNV: return "PhysicalDeviceFragmentShadingRateEnumsPropertiesNV";
      case StructureType::ePhysicalDeviceFragmentShadingRateEnumsFeaturesNV: return "PhysicalDeviceFragmentShadingRateEnumsFeaturesNV";
      case StructureType::ePipelineFragmentShadingRateEnumStateCreateInfoNV: return "PipelineFragmentShadingRateEnumStateCreateInfoNV";
      case StructureType::eAccelerationStructureGeometryMotionTrianglesDataNV: return "AccelerationStructureGeometryMotionTrianglesDataNV";
      case StructureType::ePhysicalDeviceRayTracingMotionBlurFeaturesNV: return "PhysicalDeviceRayTracingMotionBlurFeaturesNV";
      case StructureType::eAccelerationStructureMotionInfoNV: return "AccelerationStructureMotionInfoNV";
      case StructureType::ePhysicalDeviceMeshShaderFeaturesEXT: return "PhysicalDeviceMeshShaderFeaturesEXT";
      case StructureType::ePhysicalDeviceMeshShaderPropertiesEXT: return "PhysicalDeviceMeshShaderPropertiesEXT";
      case StructureType::ePhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT: return "PhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT";
      case StructureType::ePhysicalDeviceFragmentDensityMap2FeaturesEXT: return "PhysicalDeviceFragmentDensityMap2FeaturesEXT";
      case StructureType::ePhysicalDeviceFragmentDensityMap2PropertiesEXT: return "PhysicalDeviceFragmentDensityMap2PropertiesEXT";
      case StructureType::eCopyCommandTransformInfoQCOM: return "CopyCommandTransformInfoQCOM";
      case StructureType::ePhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR: return "PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR";
      case StructureType::ePhysicalDeviceImageCompressionControlFeaturesEXT: return "PhysicalDeviceImageCompressionControlFeaturesEXT";
      case StructureType::eImageCompressionControlEXT: return "ImageCompressionControlEXT";
      case StructureType::eImageCompressionPropertiesEXT: return "ImageCompressionPropertiesEXT";
      case StructureType::ePhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT: return "PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT";
      case StructureType::ePhysicalDevice4444FormatsFeaturesEXT: return "PhysicalDevice4444FormatsFeaturesEXT";
      case StructureType::ePhysicalDeviceFaultFeaturesEXT: return "PhysicalDeviceFaultFeaturesEXT";
      case StructureType::eDeviceFaultCountsEXT: return "DeviceFaultCountsEXT";
      case StructureType::eDeviceFaultInfoEXT: return "DeviceFaultInfoEXT";
      case StructureType::ePhysicalDeviceRgba10X6FormatsFeaturesEXT: return "PhysicalDeviceRgba10X6FormatsFeaturesEXT";
#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
      case StructureType::eDirectfbSurfaceCreateInfoEXT: return "DirectfbSurfaceCreateInfoEXT";
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/
      case StructureType::ePhysicalDeviceVertexInputDynamicStateFeaturesEXT: return "PhysicalDeviceVertexInputDynamicStateFeaturesEXT";
      case StructureType::eVertexInputBindingDescription2EXT: return "VertexInputBindingDescription2EXT";
      case StructureType::eVertexInputAttributeDescription2EXT: return "VertexInputAttributeDescription2EXT";
      case StructureType::ePhysicalDeviceDrmPropertiesEXT: return "PhysicalDeviceDrmPropertiesEXT";
      case StructureType::ePhysicalDeviceAddressBindingReportFeaturesEXT: return "PhysicalDeviceAddressBindingReportFeaturesEXT";
      case StructureType::eDeviceAddressBindingCallbackDataEXT: return "DeviceAddressBindingCallbackDataEXT";
      case StructureType::ePhysicalDeviceDepthClipControlFeaturesEXT: return "PhysicalDeviceDepthClipControlFeaturesEXT";
      case StructureType::ePipelineViewportDepthClipControlCreateInfoEXT: return "PipelineViewportDepthClipControlCreateInfoEXT";
      case StructureType::ePhysicalDevicePrimitiveTopologyListRestartFeaturesEXT: return "PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT";
#if defined( VK_USE_PLATFORM_FUCHSIA )
      case StructureType::eImportMemoryZirconHandleInfoFUCHSIA: return "ImportMemoryZirconHandleInfoFUCHSIA";
      case StructureType::eMemoryZirconHandlePropertiesFUCHSIA: return "MemoryZirconHandlePropertiesFUCHSIA";
      case StructureType::eMemoryGetZirconHandleInfoFUCHSIA: return "MemoryGetZirconHandleInfoFUCHSIA";
      case StructureType::eImportSemaphoreZirconHandleInfoFUCHSIA: return "ImportSemaphoreZirconHandleInfoFUCHSIA";
      case StructureType::eSemaphoreGetZirconHandleInfoFUCHSIA: return "SemaphoreGetZirconHandleInfoFUCHSIA";
      case StructureType::eBufferCollectionCreateInfoFUCHSIA: return "BufferCollectionCreateInfoFUCHSIA";
      case StructureType::eImportMemoryBufferCollectionFUCHSIA: return "ImportMemoryBufferCollectionFUCHSIA";
      case StructureType::eBufferCollectionImageCreateInfoFUCHSIA: return "BufferCollectionImageCreateInfoFUCHSIA";
      case StructureType::eBufferCollectionPropertiesFUCHSIA: return "BufferCollectionPropertiesFUCHSIA";
      case StructureType::eBufferConstraintsInfoFUCHSIA: return "BufferConstraintsInfoFUCHSIA";
      case StructureType::eBufferCollectionBufferCreateInfoFUCHSIA: return "BufferCollectionBufferCreateInfoFUCHSIA";
      case StructureType::eImageConstraintsInfoFUCHSIA: return "ImageConstraintsInfoFUCHSIA";
      case StructureType::eImageFormatConstraintsInfoFUCHSIA: return "ImageFormatConstraintsInfoFUCHSIA";
      case StructureType::eSysmemColorSpaceFUCHSIA: return "SysmemColorSpaceFUCHSIA";
      case StructureType::eBufferCollectionConstraintsInfoFUCHSIA: return "BufferCollectionConstraintsInfoFUCHSIA";
#endif /*VK_USE_PLATFORM_FUCHSIA*/
      case StructureType::eSubpassShadingPipelineCreateInfoHUAWEI: return "SubpassShadingPipelineCreateInfoHUAWEI";
      case StructureType::ePhysicalDeviceSubpassShadingFeaturesHUAWEI: return "PhysicalDeviceSubpassShadingFeaturesHUAWEI";
      case StructureType::ePhysicalDeviceSubpassShadingPropertiesHUAWEI: return "PhysicalDeviceSubpassShadingPropertiesHUAWEI";
      case StructureType::ePhysicalDeviceInvocationMaskFeaturesHUAWEI: return "PhysicalDeviceInvocationMaskFeaturesHUAWEI";
      case StructureType::eMemoryGetRemoteAddressInfoNV: return "MemoryGetRemoteAddressInfoNV";
      case StructureType::ePhysicalDeviceExternalMemoryRdmaFeaturesNV: return "PhysicalDeviceExternalMemoryRdmaFeaturesNV";
      case StructureType::ePipelinePropertiesIdentifierEXT: return "PipelinePropertiesIdentifierEXT";
      case StructureType::ePhysicalDevicePipelinePropertiesFeaturesEXT: return "PhysicalDevicePipelinePropertiesFeaturesEXT";
      case StructureType::ePhysicalDeviceFrameBoundaryFeaturesEXT: return "PhysicalDeviceFrameBoundaryFeaturesEXT";
      case StructureType::eFrameBoundaryEXT: return "FrameBoundaryEXT";
      case StructureType::ePhysicalDeviceMultisampledRenderToSingleSampledFeaturesEXT: return "PhysicalDeviceMultisampledRenderToSingleSampledFeaturesEXT";
      case StructureType::eSubpassResolvePerformanceQueryEXT: return "SubpassResolvePerformanceQueryEXT";
      case StructureType::eMultisampledRenderToSingleSampledInfoEXT: return "MultisampledRenderToSingleSampledInfoEXT";
      case StructureType::ePhysicalDeviceExtendedDynamicState2FeaturesEXT: return "PhysicalDeviceExtendedDynamicState2FeaturesEXT";
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
      case StructureType::eScreenSurfaceCreateInfoQNX: return "ScreenSurfaceCreateInfoQNX";
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/
      case StructureType::ePhysicalDeviceColorWriteEnableFeaturesEXT: return "PhysicalDeviceColorWriteEnableFeaturesEXT";
      case StructureType::ePipelineColorWriteCreateInfoEXT: return "PipelineColorWriteCreateInfoEXT";
      case StructureType::ePhysicalDevicePrimitivesGeneratedQueryFeaturesEXT: return "PhysicalDevicePrimitivesGeneratedQueryFeaturesEXT";
      case StructureType::ePhysicalDeviceRayTracingMaintenance1FeaturesKHR: return "PhysicalDeviceRayTracingMaintenance1FeaturesKHR";
      case StructureType::ePhysicalDeviceImageViewMinLodFeaturesEXT: return "PhysicalDeviceImageViewMinLodFeaturesEXT";
      case StructureType::eImageViewMinLodCreateInfoEXT: return "ImageViewMinLodCreateInfoEXT";
      case StructureType::ePhysicalDeviceMultiDrawFeaturesEXT: return "PhysicalDeviceMultiDrawFeaturesEXT";
      case StructureType::ePhysicalDeviceMultiDrawPropertiesEXT: return "PhysicalDeviceMultiDrawPropertiesEXT";
      case StructureType::ePhysicalDeviceImage2DViewOf3DFeaturesEXT: return "PhysicalDeviceImage2DViewOf3DFeaturesEXT";
      case StructureType::ePhysicalDeviceShaderTileImageFeaturesEXT: return "PhysicalDeviceShaderTileImageFeaturesEXT";
      case StructureType::ePhysicalDeviceShaderTileImagePropertiesEXT: return "PhysicalDeviceShaderTileImagePropertiesEXT";
      case StructureType::eMicromapBuildInfoEXT: return "MicromapBuildInfoEXT";
      case StructureType::eMicromapVersionInfoEXT: return "MicromapVersionInfoEXT";
      case StructureType::eCopyMicromapInfoEXT: return "CopyMicromapInfoEXT";
      case StructureType::eCopyMicromapToMemoryInfoEXT: return "CopyMicromapToMemoryInfoEXT";
      case StructureType::eCopyMemoryToMicromapInfoEXT: return "CopyMemoryToMicromapInfoEXT";
      case StructureType::ePhysicalDeviceOpacityMicromapFeaturesEXT: return "PhysicalDeviceOpacityMicromapFeaturesEXT";
      case StructureType::ePhysicalDeviceOpacityMicromapPropertiesEXT: return "PhysicalDeviceOpacityMicromapPropertiesEXT";
      case StructureType::eMicromapCreateInfoEXT: return "MicromapCreateInfoEXT";
      case StructureType::eMicromapBuildSizesInfoEXT: return "MicromapBuildSizesInfoEXT";
      case StructureType::eAccelerationStructureTrianglesOpacityMicromapEXT: return "AccelerationStructureTrianglesOpacityMicromapEXT";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case StructureType::ePhysicalDeviceDisplacementMicromapFeaturesNV: return "PhysicalDeviceDisplacementMicromapFeaturesNV";
      case StructureType::ePhysicalDeviceDisplacementMicromapPropertiesNV: return "PhysicalDeviceDisplacementMicromapPropertiesNV";
      case StructureType::eAccelerationStructureTrianglesDisplacementMicromapNV: return "AccelerationStructureTrianglesDisplacementMicromapNV";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      case StructureType::ePhysicalDeviceClusterCullingShaderFeaturesHUAWEI: return "PhysicalDeviceClusterCullingShaderFeaturesHUAWEI";
      case StructureType::ePhysicalDeviceClusterCullingShaderPropertiesHUAWEI: return "PhysicalDeviceClusterCullingShaderPropertiesHUAWEI";
      case StructureType::ePhysicalDeviceClusterCullingShaderVrsFeaturesHUAWEI: return "PhysicalDeviceClusterCullingShaderVrsFeaturesHUAWEI";
      case StructureType::ePhysicalDeviceBorderColorSwizzleFeaturesEXT: return "PhysicalDeviceBorderColorSwizzleFeaturesEXT";
      case StructureType::eSamplerBorderColorComponentMappingCreateInfoEXT: return "SamplerBorderColorComponentMappingCreateInfoEXT";
      case StructureType::ePhysicalDevicePageableDeviceLocalMemoryFeaturesEXT: return "PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT";
      case StructureType::ePhysicalDeviceShaderCorePropertiesARM: return "PhysicalDeviceShaderCorePropertiesARM";
      case StructureType::ePhysicalDeviceShaderSubgroupRotateFeaturesKHR: return "PhysicalDeviceShaderSubgroupRotateFeaturesKHR";
      case StructureType::eDeviceQueueShaderCoreControlCreateInfoARM: return "DeviceQueueShaderCoreControlCreateInfoARM";
      case StructureType::ePhysicalDeviceSchedulingControlsFeaturesARM: return "PhysicalDeviceSchedulingControlsFeaturesARM";
      case StructureType::ePhysicalDeviceSchedulingControlsPropertiesARM: return "PhysicalDeviceSchedulingControlsPropertiesARM";
      case StructureType::ePhysicalDeviceImageSlicedViewOf3DFeaturesEXT: return "PhysicalDeviceImageSlicedViewOf3DFeaturesEXT";
      case StructureType::eImageViewSlicedCreateInfoEXT: return "ImageViewSlicedCreateInfoEXT";
      case StructureType::ePhysicalDeviceDescriptorSetHostMappingFeaturesVALVE: return "PhysicalDeviceDescriptorSetHostMappingFeaturesVALVE";
      case StructureType::eDescriptorSetBindingReferenceVALVE: return "DescriptorSetBindingReferenceVALVE";
      case StructureType::eDescriptorSetLayoutHostMappingInfoVALVE: return "DescriptorSetLayoutHostMappingInfoVALVE";
      case StructureType::ePhysicalDeviceDepthClampZeroOneFeaturesEXT: return "PhysicalDeviceDepthClampZeroOneFeaturesEXT";
      case StructureType::ePhysicalDeviceNonSeamlessCubeMapFeaturesEXT: return "PhysicalDeviceNonSeamlessCubeMapFeaturesEXT";
      case StructureType::ePhysicalDeviceRenderPassStripedFeaturesARM: return "PhysicalDeviceRenderPassStripedFeaturesARM";
      case StructureType::ePhysicalDeviceRenderPassStripedPropertiesARM: return "PhysicalDeviceRenderPassStripedPropertiesARM";
      case StructureType::eRenderPassStripeBeginInfoARM: return "RenderPassStripeBeginInfoARM";
      case StructureType::eRenderPassStripeInfoARM: return "RenderPassStripeInfoARM";
      case StructureType::eRenderPassStripeSubmitInfoARM: return "RenderPassStripeSubmitInfoARM";
      case StructureType::ePhysicalDeviceFragmentDensityMapOffsetFeaturesQCOM: return "PhysicalDeviceFragmentDensityMapOffsetFeaturesQCOM";
      case StructureType::ePhysicalDeviceFragmentDensityMapOffsetPropertiesQCOM: return "PhysicalDeviceFragmentDensityMapOffsetPropertiesQCOM";
      case StructureType::eSubpassFragmentDensityMapOffsetEndInfoQCOM: return "SubpassFragmentDensityMapOffsetEndInfoQCOM";
      case StructureType::ePhysicalDeviceCopyMemoryIndirectFeaturesNV: return "PhysicalDeviceCopyMemoryIndirectFeaturesNV";
      case StructureType::ePhysicalDeviceCopyMemoryIndirectPropertiesNV: return "PhysicalDeviceCopyMemoryIndirectPropertiesNV";
      case StructureType::ePhysicalDeviceMemoryDecompressionFeaturesNV: return "PhysicalDeviceMemoryDecompressionFeaturesNV";
      case StructureType::ePhysicalDeviceMemoryDecompressionPropertiesNV: return "PhysicalDeviceMemoryDecompressionPropertiesNV";
      case StructureType::ePhysicalDeviceDeviceGeneratedCommandsComputeFeaturesNV: return "PhysicalDeviceDeviceGeneratedCommandsComputeFeaturesNV";
      case StructureType::eComputePipelineIndirectBufferInfoNV: return "ComputePipelineIndirectBufferInfoNV";
      case StructureType::ePipelineIndirectDeviceAddressInfoNV: return "PipelineIndirectDeviceAddressInfoNV";
      case StructureType::ePhysicalDeviceLinearColorAttachmentFeaturesNV: return "PhysicalDeviceLinearColorAttachmentFeaturesNV";
      case StructureType::ePhysicalDeviceShaderMaximalReconvergenceFeaturesKHR: return "PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR";
      case StructureType::ePhysicalDeviceImageCompressionControlSwapchainFeaturesEXT: return "PhysicalDeviceImageCompressionControlSwapchainFeaturesEXT";
      case StructureType::ePhysicalDeviceImageProcessingFeaturesQCOM: return "PhysicalDeviceImageProcessingFeaturesQCOM";
      case StructureType::ePhysicalDeviceImageProcessingPropertiesQCOM: return "PhysicalDeviceImageProcessingPropertiesQCOM";
      case StructureType::eImageViewSampleWeightCreateInfoQCOM: return "ImageViewSampleWeightCreateInfoQCOM";
      case StructureType::ePhysicalDeviceNestedCommandBufferFeaturesEXT: return "PhysicalDeviceNestedCommandBufferFeaturesEXT";
      case StructureType::ePhysicalDeviceNestedCommandBufferPropertiesEXT: return "PhysicalDeviceNestedCommandBufferPropertiesEXT";
      case StructureType::eExternalMemoryAcquireUnmodifiedEXT: return "ExternalMemoryAcquireUnmodifiedEXT";
      case StructureType::ePhysicalDeviceExtendedDynamicState3FeaturesEXT: return "PhysicalDeviceExtendedDynamicState3FeaturesEXT";
      case StructureType::ePhysicalDeviceExtendedDynamicState3PropertiesEXT: return "PhysicalDeviceExtendedDynamicState3PropertiesEXT";
      case StructureType::ePhysicalDeviceSubpassMergeFeedbackFeaturesEXT: return "PhysicalDeviceSubpassMergeFeedbackFeaturesEXT";
      case StructureType::eRenderPassCreationControlEXT: return "RenderPassCreationControlEXT";
      case StructureType::eRenderPassCreationFeedbackCreateInfoEXT: return "RenderPassCreationFeedbackCreateInfoEXT";
      case StructureType::eRenderPassSubpassFeedbackCreateInfoEXT: return "RenderPassSubpassFeedbackCreateInfoEXT";
      case StructureType::eDirectDriverLoadingInfoLUNARG: return "DirectDriverLoadingInfoLUNARG";
      case StructureType::eDirectDriverLoadingListLUNARG: return "DirectDriverLoadingListLUNARG";
      case StructureType::ePhysicalDeviceShaderModuleIdentifierFeaturesEXT: return "PhysicalDeviceShaderModuleIdentifierFeaturesEXT";
      case StructureType::ePhysicalDeviceShaderModuleIdentifierPropertiesEXT: return "PhysicalDeviceShaderModuleIdentifierPropertiesEXT";
      case StructureType::ePipelineShaderStageModuleIdentifierCreateInfoEXT: return "PipelineShaderStageModuleIdentifierCreateInfoEXT";
      case StructureType::eShaderModuleIdentifierEXT: return "ShaderModuleIdentifierEXT";
      case StructureType::ePhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT: return "PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT";
      case StructureType::ePhysicalDeviceOpticalFlowFeaturesNV: return "PhysicalDeviceOpticalFlowFeaturesNV";
      case StructureType::ePhysicalDeviceOpticalFlowPropertiesNV: return "PhysicalDeviceOpticalFlowPropertiesNV";
      case StructureType::eOpticalFlowImageFormatInfoNV: return "OpticalFlowImageFormatInfoNV";
      case StructureType::eOpticalFlowImageFormatPropertiesNV: return "OpticalFlowImageFormatPropertiesNV";
      case StructureType::eOpticalFlowSessionCreateInfoNV: return "OpticalFlowSessionCreateInfoNV";
      case StructureType::eOpticalFlowExecuteInfoNV: return "OpticalFlowExecuteInfoNV";
      case StructureType::eOpticalFlowSessionCreatePrivateDataInfoNV: return "OpticalFlowSessionCreatePrivateDataInfoNV";
      case StructureType::ePhysicalDeviceLegacyDitheringFeaturesEXT: return "PhysicalDeviceLegacyDitheringFeaturesEXT";
      case StructureType::ePhysicalDevicePipelineProtectedAccessFeaturesEXT: return "PhysicalDevicePipelineProtectedAccessFeaturesEXT";
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
      case StructureType::ePhysicalDeviceExternalFormatResolveFeaturesANDROID: return "PhysicalDeviceExternalFormatResolveFeaturesANDROID";
      case StructureType::ePhysicalDeviceExternalFormatResolvePropertiesANDROID: return "PhysicalDeviceExternalFormatResolvePropertiesANDROID";
      case StructureType::eAndroidHardwareBufferFormatResolvePropertiesANDROID: return "AndroidHardwareBufferFormatResolvePropertiesANDROID";
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
      case StructureType::ePhysicalDeviceMaintenance5FeaturesKHR: return "PhysicalDeviceMaintenance5FeaturesKHR";
      case StructureType::ePhysicalDeviceMaintenance5PropertiesKHR: return "PhysicalDeviceMaintenance5PropertiesKHR";
      case StructureType::eRenderingAreaInfoKHR: return "RenderingAreaInfoKHR";
      case StructureType::eDeviceImageSubresourceInfoKHR: return "DeviceImageSubresourceInfoKHR";
      case StructureType::eSubresourceLayout2KHR: return "SubresourceLayout2KHR";
      case StructureType::eImageSubresource2KHR: return "ImageSubresource2KHR";
      case StructureType::ePipelineCreateFlags2CreateInfoKHR: return "PipelineCreateFlags2CreateInfoKHR";
      case StructureType::eBufferUsageFlags2CreateInfoKHR: return "BufferUsageFlags2CreateInfoKHR";
      case StructureType::ePhysicalDeviceRayTracingPositionFetchFeaturesKHR: return "PhysicalDeviceRayTracingPositionFetchFeaturesKHR";
      case StructureType::ePhysicalDeviceShaderObjectFeaturesEXT: return "PhysicalDeviceShaderObjectFeaturesEXT";
      case StructureType::ePhysicalDeviceShaderObjectPropertiesEXT: return "PhysicalDeviceShaderObjectPropertiesEXT";
      case StructureType::eShaderCreateInfoEXT: return "ShaderCreateInfoEXT";
      case StructureType::ePhysicalDeviceTilePropertiesFeaturesQCOM: return "PhysicalDeviceTilePropertiesFeaturesQCOM";
      case StructureType::eTilePropertiesQCOM: return "TilePropertiesQCOM";
      case StructureType::ePhysicalDeviceAmigoProfilingFeaturesSEC: return "PhysicalDeviceAmigoProfilingFeaturesSEC";
      case StructureType::eAmigoProfilingSubmitInfoSEC: return "AmigoProfilingSubmitInfoSEC";
      case StructureType::ePhysicalDeviceMultiviewPerViewViewportsFeaturesQCOM: return "PhysicalDeviceMultiviewPerViewViewportsFeaturesQCOM";
      case StructureType::ePhysicalDeviceRayTracingInvocationReorderFeaturesNV: return "PhysicalDeviceRayTracingInvocationReorderFeaturesNV";
      case StructureType::ePhysicalDeviceRayTracingInvocationReorderPropertiesNV: return "PhysicalDeviceRayTracingInvocationReorderPropertiesNV";
      case StructureType::ePhysicalDeviceExtendedSparseAddressSpaceFeaturesNV: return "PhysicalDeviceExtendedSparseAddressSpaceFeaturesNV";
      case StructureType::ePhysicalDeviceExtendedSparseAddressSpacePropertiesNV: return "PhysicalDeviceExtendedSparseAddressSpacePropertiesNV";
      case StructureType::ePhysicalDeviceMutableDescriptorTypeFeaturesEXT: return "PhysicalDeviceMutableDescriptorTypeFeaturesEXT";
      case StructureType::eMutableDescriptorTypeCreateInfoEXT: return "MutableDescriptorTypeCreateInfoEXT";
      case StructureType::eLayerSettingsCreateInfoEXT: return "LayerSettingsCreateInfoEXT";
      case StructureType::ePhysicalDeviceShaderCoreBuiltinsFeaturesARM: return "PhysicalDeviceShaderCoreBuiltinsFeaturesARM";
      case StructureType::ePhysicalDeviceShaderCoreBuiltinsPropertiesARM: return "PhysicalDeviceShaderCoreBuiltinsPropertiesARM";
      case StructureType::ePhysicalDevicePipelineLibraryGroupHandlesFeaturesEXT: return "PhysicalDevicePipelineLibraryGroupHandlesFeaturesEXT";
      case StructureType::ePhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT: return "PhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT";
      case StructureType::eLatencySleepModeInfoNV: return "LatencySleepModeInfoNV";
      case StructureType::eLatencySleepInfoNV: return "LatencySleepInfoNV";
      case StructureType::eSetLatencyMarkerInfoNV: return "SetLatencyMarkerInfoNV";
      case StructureType::eGetLatencyMarkerInfoNV: return "GetLatencyMarkerInfoNV";
      case StructureType::eLatencyTimingsFrameReportNV: return "LatencyTimingsFrameReportNV";
      case StructureType::eLatencySubmissionPresentIdNV: return "LatencySubmissionPresentIdNV";
      case StructureType::eOutOfBandQueueTypeInfoNV: return "OutOfBandQueueTypeInfoNV";
      case StructureType::eSwapchainLatencyCreateInfoNV: return "SwapchainLatencyCreateInfoNV";
      case StructureType::eLatencySurfaceCapabilitiesNV: return "LatencySurfaceCapabilitiesNV";
      case StructureType::ePhysicalDeviceCooperativeMatrixFeaturesKHR: return "PhysicalDeviceCooperativeMatrixFeaturesKHR";
      case StructureType::eCooperativeMatrixPropertiesKHR: return "CooperativeMatrixPropertiesKHR";
      case StructureType::ePhysicalDeviceCooperativeMatrixPropertiesKHR: return "PhysicalDeviceCooperativeMatrixPropertiesKHR";
      case StructureType::ePhysicalDeviceMultiviewPerViewRenderAreasFeaturesQCOM: return "PhysicalDeviceMultiviewPerViewRenderAreasFeaturesQCOM";
      case StructureType::eMultiviewPerViewRenderAreasRenderPassBeginInfoQCOM: return "MultiviewPerViewRenderAreasRenderPassBeginInfoQCOM";
      case StructureType::eVideoDecodeAv1CapabilitiesKHR: return "VideoDecodeAv1CapabilitiesKHR";
      case StructureType::eVideoDecodeAv1PictureInfoKHR: return "VideoDecodeAv1PictureInfoKHR";
      case StructureType::eVideoDecodeAv1ProfileInfoKHR: return "VideoDecodeAv1ProfileInfoKHR";
      case StructureType::eVideoDecodeAv1SessionParametersCreateInfoKHR: return "VideoDecodeAv1SessionParametersCreateInfoKHR";
      case StructureType::eVideoDecodeAv1DpbSlotInfoKHR: return "VideoDecodeAv1DpbSlotInfoKHR";
      case StructureType::ePhysicalDeviceVideoMaintenance1FeaturesKHR: return "PhysicalDeviceVideoMaintenance1FeaturesKHR";
      case StructureType::eVideoInlineQueryInfoKHR: return "VideoInlineQueryInfoKHR";
      case StructureType::ePhysicalDevicePerStageDescriptorSetFeaturesNV: return "PhysicalDevicePerStageDescriptorSetFeaturesNV";
      case StructureType::ePhysicalDeviceImageProcessing2FeaturesQCOM: return "PhysicalDeviceImageProcessing2FeaturesQCOM";
      case StructureType::ePhysicalDeviceImageProcessing2PropertiesQCOM: return "PhysicalDeviceImageProcessing2PropertiesQCOM";
      case StructureType::eSamplerBlockMatchWindowCreateInfoQCOM: return "SamplerBlockMatchWindowCreateInfoQCOM";
      case StructureType::eSamplerCubicWeightsCreateInfoQCOM: return "SamplerCubicWeightsCreateInfoQCOM";
      case StructureType::ePhysicalDeviceCubicWeightsFeaturesQCOM: return "PhysicalDeviceCubicWeightsFeaturesQCOM";
      case StructureType::eBlitImageCubicWeightsInfoQCOM: return "BlitImageCubicWeightsInfoQCOM";
      case StructureType::ePhysicalDeviceYcbcrDegammaFeaturesQCOM: return "PhysicalDeviceYcbcrDegammaFeaturesQCOM";
      case StructureType::eSamplerYcbcrConversionYcbcrDegammaCreateInfoQCOM: return "SamplerYcbcrConversionYcbcrDegammaCreateInfoQCOM";
      case StructureType::ePhysicalDeviceCubicClampFeaturesQCOM: return "PhysicalDeviceCubicClampFeaturesQCOM";
      case StructureType::ePhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT: return "PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT";
      case StructureType::ePhysicalDeviceVertexAttributeDivisorPropertiesKHR: return "PhysicalDeviceVertexAttributeDivisorPropertiesKHR";
      case StructureType::ePipelineVertexInputDivisorStateCreateInfoKHR: return "PipelineVertexInputDivisorStateCreateInfoKHR";
      case StructureType::ePhysicalDeviceVertexAttributeDivisorFeaturesKHR: return "PhysicalDeviceVertexAttributeDivisorFeaturesKHR";
      case StructureType::ePhysicalDeviceShaderFloatControls2FeaturesKHR: return "PhysicalDeviceShaderFloatControls2FeaturesKHR";
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
      case StructureType::eScreenBufferPropertiesQNX: return "ScreenBufferPropertiesQNX";
      case StructureType::eScreenBufferFormatPropertiesQNX: return "ScreenBufferFormatPropertiesQNX";
      case StructureType::eImportScreenBufferInfoQNX: return "ImportScreenBufferInfoQNX";
      case StructureType::eExternalFormatQNX: return "ExternalFormatQNX";
      case StructureType::ePhysicalDeviceExternalMemoryScreenBufferFeaturesQNX: return "PhysicalDeviceExternalMemoryScreenBufferFeaturesQNX";
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/
      case StructureType::ePhysicalDeviceLayeredDriverPropertiesMSFT: return "PhysicalDeviceLayeredDriverPropertiesMSFT";
      case StructureType::ePhysicalDeviceIndexTypeUint8FeaturesKHR: return "PhysicalDeviceIndexTypeUint8FeaturesKHR";
      case StructureType::ePhysicalDeviceLineRasterizationFeaturesKHR: return "PhysicalDeviceLineRasterizationFeaturesKHR";
      case StructureType::ePipelineRasterizationLineStateCreateInfoKHR: return "PipelineRasterizationLineStateCreateInfoKHR";
      case StructureType::ePhysicalDeviceLineRasterizationPropertiesKHR: return "PhysicalDeviceLineRasterizationPropertiesKHR";
      case StructureType::eCalibratedTimestampInfoKHR: return "CalibratedTimestampInfoKHR";
      case StructureType::ePhysicalDeviceShaderExpectAssumeFeaturesKHR: return "PhysicalDeviceShaderExpectAssumeFeaturesKHR";
      case StructureType::ePhysicalDeviceMaintenance6FeaturesKHR: return "PhysicalDeviceMaintenance6FeaturesKHR";
      case StructureType::ePhysicalDeviceMaintenance6PropertiesKHR: return "PhysicalDeviceMaintenance6PropertiesKHR";
      case StructureType::eBindMemoryStatusKHR: return "BindMemoryStatusKHR";
      case StructureType::eBindDescriptorSetsInfoKHR: return "BindDescriptorSetsInfoKHR";
      case StructureType::ePushConstantsInfoKHR: return "PushConstantsInfoKHR";
      case StructureType::ePushDescriptorSetInfoKHR: return "PushDescriptorSetInfoKHR";
      case StructureType::ePushDescriptorSetWithTemplateInfoKHR: return "PushDescriptorSetWithTemplateInfoKHR";
      case StructureType::eSetDescriptorBufferOffsetsInfoEXT: return "SetDescriptorBufferOffsetsInfoEXT";
      case StructureType::eBindDescriptorBufferEmbeddedSamplersInfoEXT: return "BindDescriptorBufferEmbeddedSamplersInfoEXT";
      case StructureType::ePhysicalDeviceDescriptorPoolOverallocationFeaturesNV: return "PhysicalDeviceDescriptorPoolOverallocationFeaturesNV";
      case StructureType::ePhysicalDeviceRawAccessChainsFeaturesNV: return "PhysicalDeviceRawAccessChainsFeaturesNV";
      case StructureType::ePhysicalDeviceShaderAtomicFloat16VectorFeaturesNV: return "PhysicalDeviceShaderAtomicFloat16VectorFeaturesNV";
      case StructureType::ePhysicalDeviceRayTracingValidationFeaturesNV: return "PhysicalDeviceRayTracingValidationFeaturesNV";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineCacheHeaderVersion value )
  {
    switch ( value )
    {
      case PipelineCacheHeaderVersion::eOne: return "One";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ObjectType value )
  {
    switch ( value )
    {
      case ObjectType::eUnknown: return "Unknown";
      case ObjectType::eInstance: return "Instance";
      case ObjectType::ePhysicalDevice: return "PhysicalDevice";
      case ObjectType::eDevice: return "Device";
      case ObjectType::eQueue: return "Queue";
      case ObjectType::eSemaphore: return "Semaphore";
      case ObjectType::eCommandBuffer: return "CommandBuffer";
      case ObjectType::eFence: return "Fence";
      case ObjectType::eDeviceMemory: return "DeviceMemory";
      case ObjectType::eBuffer: return "Buffer";
      case ObjectType::eImage: return "Image";
      case ObjectType::eEvent: return "Event";
      case ObjectType::eQueryPool: return "QueryPool";
      case ObjectType::eBufferView: return "BufferView";
      case ObjectType::eImageView: return "ImageView";
      case ObjectType::eShaderModule: return "ShaderModule";
      case ObjectType::ePipelineCache: return "PipelineCache";
      case ObjectType::ePipelineLayout: return "PipelineLayout";
      case ObjectType::eRenderPass: return "RenderPass";
      case ObjectType::ePipeline: return "Pipeline";
      case ObjectType::eDescriptorSetLayout: return "DescriptorSetLayout";
      case ObjectType::eSampler: return "Sampler";
      case ObjectType::eDescriptorPool: return "DescriptorPool";
      case ObjectType::eDescriptorSet: return "DescriptorSet";
      case ObjectType::eFramebuffer: return "Framebuffer";
      case ObjectType::eCommandPool: return "CommandPool";
      case ObjectType::eSamplerYcbcrConversion: return "SamplerYcbcrConversion";
      case ObjectType::eDescriptorUpdateTemplate: return "DescriptorUpdateTemplate";
      case ObjectType::ePrivateDataSlot: return "PrivateDataSlot";
      case ObjectType::eSurfaceKHR: return "SurfaceKHR";
      case ObjectType::eSwapchainKHR: return "SwapchainKHR";
      case ObjectType::eDisplayKHR: return "DisplayKHR";
      case ObjectType::eDisplayModeKHR: return "DisplayModeKHR";
      case ObjectType::eDebugReportCallbackEXT: return "DebugReportCallbackEXT";
      case ObjectType::eVideoSessionKHR: return "VideoSessionKHR";
      case ObjectType::eVideoSessionParametersKHR: return "VideoSessionParametersKHR";
      case ObjectType::eCuModuleNVX: return "CuModuleNVX";
      case ObjectType::eCuFunctionNVX: return "CuFunctionNVX";
      case ObjectType::eDebugUtilsMessengerEXT: return "DebugUtilsMessengerEXT";
      case ObjectType::eAccelerationStructureKHR: return "AccelerationStructureKHR";
      case ObjectType::eValidationCacheEXT: return "ValidationCacheEXT";
      case ObjectType::eAccelerationStructureNV: return "AccelerationStructureNV";
      case ObjectType::ePerformanceConfigurationINTEL: return "PerformanceConfigurationINTEL";
      case ObjectType::eDeferredOperationKHR: return "DeferredOperationKHR";
      case ObjectType::eIndirectCommandsLayoutNV: return "IndirectCommandsLayoutNV";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case ObjectType::eCudaModuleNV: return "CudaModuleNV";
      case ObjectType::eCudaFunctionNV: return "CudaFunctionNV";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_USE_PLATFORM_FUCHSIA )
      case ObjectType::eBufferCollectionFUCHSIA: return "BufferCollectionFUCHSIA";
#endif /*VK_USE_PLATFORM_FUCHSIA*/
      case ObjectType::eMicromapEXT: return "MicromapEXT";
      case ObjectType::eOpticalFlowSessionNV: return "OpticalFlowSessionNV";
      case ObjectType::eShaderEXT: return "ShaderEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( VendorId value )
  {
    switch ( value )
    {
      case VendorId::eVIV: return "VIV";
      case VendorId::eVSI: return "VSI";
      case VendorId::eKazan: return "Kazan";
      case VendorId::eCodeplay: return "Codeplay";
      case VendorId::eMESA: return "MESA";
      case VendorId::ePocl: return "Pocl";
      case VendorId::eMobileye: return "Mobileye";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( Format value )
  {
    switch ( value )
    {
      case Format::eUndefined: return "Undefined";
      case Format::eR4G4UnormPack8: return "R4G4UnormPack8";
      case Format::eR4G4B4A4UnormPack16: return "R4G4B4A4UnormPack16";
      case Format::eB4G4R4A4UnormPack16: return "B4G4R4A4UnormPack16";
      case Format::eR5G6B5UnormPack16: return "R5G6B5UnormPack16";
      case Format::eB5G6R5UnormPack16: return "B5G6R5UnormPack16";
      case Format::eR5G5B5A1UnormPack16: return "R5G5B5A1UnormPack16";
      case Format::eB5G5R5A1UnormPack16: return "B5G5R5A1UnormPack16";
      case Format::eA1R5G5B5UnormPack16: return "A1R5G5B5UnormPack16";
      case Format::eR8Unorm: return "R8Unorm";
      case Format::eR8Snorm: return "R8Snorm";
      case Format::eR8Uscaled: return "R8Uscaled";
      case Format::eR8Sscaled: return "R8Sscaled";
      case Format::eR8Uint: return "R8Uint";
      case Format::eR8Sint: return "R8Sint";
      case Format::eR8Srgb: return "R8Srgb";
      case Format::eR8G8Unorm: return "R8G8Unorm";
      case Format::eR8G8Snorm: return "R8G8Snorm";
      case Format::eR8G8Uscaled: return "R8G8Uscaled";
      case Format::eR8G8Sscaled: return "R8G8Sscaled";
      case Format::eR8G8Uint: return "R8G8Uint";
      case Format::eR8G8Sint: return "R8G8Sint";
      case Format::eR8G8Srgb: return "R8G8Srgb";
      case Format::eR8G8B8Unorm: return "R8G8B8Unorm";
      case Format::eR8G8B8Snorm: return "R8G8B8Snorm";
      case Format::eR8G8B8Uscaled: return "R8G8B8Uscaled";
      case Format::eR8G8B8Sscaled: return "R8G8B8Sscaled";
      case Format::eR8G8B8Uint: return "R8G8B8Uint";
      case Format::eR8G8B8Sint: return "R8G8B8Sint";
      case Format::eR8G8B8Srgb: return "R8G8B8Srgb";
      case Format::eB8G8R8Unorm: return "B8G8R8Unorm";
      case Format::eB8G8R8Snorm: return "B8G8R8Snorm";
      case Format::eB8G8R8Uscaled: return "B8G8R8Uscaled";
      case Format::eB8G8R8Sscaled: return "B8G8R8Sscaled";
      case Format::eB8G8R8Uint: return "B8G8R8Uint";
      case Format::eB8G8R8Sint: return "B8G8R8Sint";
      case Format::eB8G8R8Srgb: return "B8G8R8Srgb";
      case Format::eR8G8B8A8Unorm: return "R8G8B8A8Unorm";
      case Format::eR8G8B8A8Snorm: return "R8G8B8A8Snorm";
      case Format::eR8G8B8A8Uscaled: return "R8G8B8A8Uscaled";
      case Format::eR8G8B8A8Sscaled: return "R8G8B8A8Sscaled";
      case Format::eR8G8B8A8Uint: return "R8G8B8A8Uint";
      case Format::eR8G8B8A8Sint: return "R8G8B8A8Sint";
      case Format::eR8G8B8A8Srgb: return "R8G8B8A8Srgb";
      case Format::eB8G8R8A8Unorm: return "B8G8R8A8Unorm";
      case Format::eB8G8R8A8Snorm: return "B8G8R8A8Snorm";
      case Format::eB8G8R8A8Uscaled: return "B8G8R8A8Uscaled";
      case Format::eB8G8R8A8Sscaled: return "B8G8R8A8Sscaled";
      case Format::eB8G8R8A8Uint: return "B8G8R8A8Uint";
      case Format::eB8G8R8A8Sint: return "B8G8R8A8Sint";
      case Format::eB8G8R8A8Srgb: return "B8G8R8A8Srgb";
      case Format::eA8B8G8R8UnormPack32: return "A8B8G8R8UnormPack32";
      case Format::eA8B8G8R8SnormPack32: return "A8B8G8R8SnormPack32";
      case Format::eA8B8G8R8UscaledPack32: return "A8B8G8R8UscaledPack32";
      case Format::eA8B8G8R8SscaledPack32: return "A8B8G8R8SscaledPack32";
      case Format::eA8B8G8R8UintPack32: return "A8B8G8R8UintPack32";
      case Format::eA8B8G8R8SintPack32: return "A8B8G8R8SintPack32";
      case Format::eA8B8G8R8SrgbPack32: return "A8B8G8R8SrgbPack32";
      case Format::eA2R10G10B10UnormPack32: return "A2R10G10B10UnormPack32";
      case Format::eA2R10G10B10SnormPack32: return "A2R10G10B10SnormPack32";
      case Format::eA2R10G10B10UscaledPack32: return "A2R10G10B10UscaledPack32";
      case Format::eA2R10G10B10SscaledPack32: return "A2R10G10B10SscaledPack32";
      case Format::eA2R10G10B10UintPack32: return "A2R10G10B10UintPack32";
      case Format::eA2R10G10B10SintPack32: return "A2R10G10B10SintPack32";
      case Format::eA2B10G10R10UnormPack32: return "A2B10G10R10UnormPack32";
      case Format::eA2B10G10R10SnormPack32: return "A2B10G10R10SnormPack32";
      case Format::eA2B10G10R10UscaledPack32: return "A2B10G10R10UscaledPack32";
      case Format::eA2B10G10R10SscaledPack32: return "A2B10G10R10SscaledPack32";
      case Format::eA2B10G10R10UintPack32: return "A2B10G10R10UintPack32";
      case Format::eA2B10G10R10SintPack32: return "A2B10G10R10SintPack32";
      case Format::eR16Unorm: return "R16Unorm";
      case Format::eR16Snorm: return "R16Snorm";
      case Format::eR16Uscaled: return "R16Uscaled";
      case Format::eR16Sscaled: return "R16Sscaled";
      case Format::eR16Uint: return "R16Uint";
      case Format::eR16Sint: return "R16Sint";
      case Format::eR16Sfloat: return "R16Sfloat";
      case Format::eR16G16Unorm: return "R16G16Unorm";
      case Format::eR16G16Snorm: return "R16G16Snorm";
      case Format::eR16G16Uscaled: return "R16G16Uscaled";
      case Format::eR16G16Sscaled: return "R16G16Sscaled";
      case Format::eR16G16Uint: return "R16G16Uint";
      case Format::eR16G16Sint: return "R16G16Sint";
      case Format::eR16G16Sfloat: return "R16G16Sfloat";
      case Format::eR16G16B16Unorm: return "R16G16B16Unorm";
      case Format::eR16G16B16Snorm: return "R16G16B16Snorm";
      case Format::eR16G16B16Uscaled: return "R16G16B16Uscaled";
      case Format::eR16G16B16Sscaled: return "R16G16B16Sscaled";
      case Format::eR16G16B16Uint: return "R16G16B16Uint";
      case Format::eR16G16B16Sint: return "R16G16B16Sint";
      case Format::eR16G16B16Sfloat: return "R16G16B16Sfloat";
      case Format::eR16G16B16A16Unorm: return "R16G16B16A16Unorm";
      case Format::eR16G16B16A16Snorm: return "R16G16B16A16Snorm";
      case Format::eR16G16B16A16Uscaled: return "R16G16B16A16Uscaled";
      case Format::eR16G16B16A16Sscaled: return "R16G16B16A16Sscaled";
      case Format::eR16G16B16A16Uint: return "R16G16B16A16Uint";
      case Format::eR16G16B16A16Sint: return "R16G16B16A16Sint";
      case Format::eR16G16B16A16Sfloat: return "R16G16B16A16Sfloat";
      case Format::eR32Uint: return "R32Uint";
      case Format::eR32Sint: return "R32Sint";
      case Format::eR32Sfloat: return "R32Sfloat";
      case Format::eR32G32Uint: return "R32G32Uint";
      case Format::eR32G32Sint: return "R32G32Sint";
      case Format::eR32G32Sfloat: return "R32G32Sfloat";
      case Format::eR32G32B32Uint: return "R32G32B32Uint";
      case Format::eR32G32B32Sint: return "R32G32B32Sint";
      case Format::eR32G32B32Sfloat: return "R32G32B32Sfloat";
      case Format::eR32G32B32A32Uint: return "R32G32B32A32Uint";
      case Format::eR32G32B32A32Sint: return "R32G32B32A32Sint";
      case Format::eR32G32B32A32Sfloat: return "R32G32B32A32Sfloat";
      case Format::eR64Uint: return "R64Uint";
      case Format::eR64Sint: return "R64Sint";
      case Format::eR64Sfloat: return "R64Sfloat";
      case Format::eR64G64Uint: return "R64G64Uint";
      case Format::eR64G64Sint: return "R64G64Sint";
      case Format::eR64G64Sfloat: return "R64G64Sfloat";
      case Format::eR64G64B64Uint: return "R64G64B64Uint";
      case Format::eR64G64B64Sint: return "R64G64B64Sint";
      case Format::eR64G64B64Sfloat: return "R64G64B64Sfloat";
      case Format::eR64G64B64A64Uint: return "R64G64B64A64Uint";
      case Format::eR64G64B64A64Sint: return "R64G64B64A64Sint";
      case Format::eR64G64B64A64Sfloat: return "R64G64B64A64Sfloat";
      case Format::eB10G11R11UfloatPack32: return "B10G11R11UfloatPack32";
      case Format::eE5B9G9R9UfloatPack32: return "E5B9G9R9UfloatPack32";
      case Format::eD16Unorm: return "D16Unorm";
      case Format::eX8D24UnormPack32: return "X8D24UnormPack32";
      case Format::eD32Sfloat: return "D32Sfloat";
      case Format::eS8Uint: return "S8Uint";
      case Format::eD16UnormS8Uint: return "D16UnormS8Uint";
      case Format::eD24UnormS8Uint: return "D24UnormS8Uint";
      case Format::eD32SfloatS8Uint: return "D32SfloatS8Uint";
      case Format::eBc1RgbUnormBlock: return "Bc1RgbUnormBlock";
      case Format::eBc1RgbSrgbBlock: return "Bc1RgbSrgbBlock";
      case Format::eBc1RgbaUnormBlock: return "Bc1RgbaUnormBlock";
      case Format::eBc1RgbaSrgbBlock: return "Bc1RgbaSrgbBlock";
      case Format::eBc2UnormBlock: return "Bc2UnormBlock";
      case Format::eBc2SrgbBlock: return "Bc2SrgbBlock";
      case Format::eBc3UnormBlock: return "Bc3UnormBlock";
      case Format::eBc3SrgbBlock: return "Bc3SrgbBlock";
      case Format::eBc4UnormBlock: return "Bc4UnormBlock";
      case Format::eBc4SnormBlock: return "Bc4SnormBlock";
      case Format::eBc5UnormBlock: return "Bc5UnormBlock";
      case Format::eBc5SnormBlock: return "Bc5SnormBlock";
      case Format::eBc6HUfloatBlock: return "Bc6HUfloatBlock";
      case Format::eBc6HSfloatBlock: return "Bc6HSfloatBlock";
      case Format::eBc7UnormBlock: return "Bc7UnormBlock";
      case Format::eBc7SrgbBlock: return "Bc7SrgbBlock";
      case Format::eEtc2R8G8B8UnormBlock: return "Etc2R8G8B8UnormBlock";
      case Format::eEtc2R8G8B8SrgbBlock: return "Etc2R8G8B8SrgbBlock";
      case Format::eEtc2R8G8B8A1UnormBlock: return "Etc2R8G8B8A1UnormBlock";
      case Format::eEtc2R8G8B8A1SrgbBlock: return "Etc2R8G8B8A1SrgbBlock";
      case Format::eEtc2R8G8B8A8UnormBlock: return "Etc2R8G8B8A8UnormBlock";
      case Format::eEtc2R8G8B8A8SrgbBlock: return "Etc2R8G8B8A8SrgbBlock";
      case Format::eEacR11UnormBlock: return "EacR11UnormBlock";
      case Format::eEacR11SnormBlock: return "EacR11SnormBlock";
      case Format::eEacR11G11UnormBlock: return "EacR11G11UnormBlock";
      case Format::eEacR11G11SnormBlock: return "EacR11G11SnormBlock";
      case Format::eAstc4x4UnormBlock: return "Astc4x4UnormBlock";
      case Format::eAstc4x4SrgbBlock: return "Astc4x4SrgbBlock";
      case Format::eAstc5x4UnormBlock: return "Astc5x4UnormBlock";
      case Format::eAstc5x4SrgbBlock: return "Astc5x4SrgbBlock";
      case Format::eAstc5x5UnormBlock: return "Astc5x5UnormBlock";
      case Format::eAstc5x5SrgbBlock: return "Astc5x5SrgbBlock";
      case Format::eAstc6x5UnormBlock: return "Astc6x5UnormBlock";
      case Format::eAstc6x5SrgbBlock: return "Astc6x5SrgbBlock";
      case Format::eAstc6x6UnormBlock: return "Astc6x6UnormBlock";
      case Format::eAstc6x6SrgbBlock: return "Astc6x6SrgbBlock";
      case Format::eAstc8x5UnormBlock: return "Astc8x5UnormBlock";
      case Format::eAstc8x5SrgbBlock: return "Astc8x5SrgbBlock";
      case Format::eAstc8x6UnormBlock: return "Astc8x6UnormBlock";
      case Format::eAstc8x6SrgbBlock: return "Astc8x6SrgbBlock";
      case Format::eAstc8x8UnormBlock: return "Astc8x8UnormBlock";
      case Format::eAstc8x8SrgbBlock: return "Astc8x8SrgbBlock";
      case Format::eAstc10x5UnormBlock: return "Astc10x5UnormBlock";
      case Format::eAstc10x5SrgbBlock: return "Astc10x5SrgbBlock";
      case Format::eAstc10x6UnormBlock: return "Astc10x6UnormBlock";
      case Format::eAstc10x6SrgbBlock: return "Astc10x6SrgbBlock";
      case Format::eAstc10x8UnormBlock: return "Astc10x8UnormBlock";
      case Format::eAstc10x8SrgbBlock: return "Astc10x8SrgbBlock";
      case Format::eAstc10x10UnormBlock: return "Astc10x10UnormBlock";
      case Format::eAstc10x10SrgbBlock: return "Astc10x10SrgbBlock";
      case Format::eAstc12x10UnormBlock: return "Astc12x10UnormBlock";
      case Format::eAstc12x10SrgbBlock: return "Astc12x10SrgbBlock";
      case Format::eAstc12x12UnormBlock: return "Astc12x12UnormBlock";
      case Format::eAstc12x12SrgbBlock: return "Astc12x12SrgbBlock";
      case Format::eG8B8G8R8422Unorm: return "G8B8G8R8422Unorm";
      case Format::eB8G8R8G8422Unorm: return "B8G8R8G8422Unorm";
      case Format::eG8B8R83Plane420Unorm: return "G8B8R83Plane420Unorm";
      case Format::eG8B8R82Plane420Unorm: return "G8B8R82Plane420Unorm";
      case Format::eG8B8R83Plane422Unorm: return "G8B8R83Plane422Unorm";
      case Format::eG8B8R82Plane422Unorm: return "G8B8R82Plane422Unorm";
      case Format::eG8B8R83Plane444Unorm: return "G8B8R83Plane444Unorm";
      case Format::eR10X6UnormPack16: return "R10X6UnormPack16";
      case Format::eR10X6G10X6Unorm2Pack16: return "R10X6G10X6Unorm2Pack16";
      case Format::eR10X6G10X6B10X6A10X6Unorm4Pack16: return "R10X6G10X6B10X6A10X6Unorm4Pack16";
      case Format::eG10X6B10X6G10X6R10X6422Unorm4Pack16: return "G10X6B10X6G10X6R10X6422Unorm4Pack16";
      case Format::eB10X6G10X6R10X6G10X6422Unorm4Pack16: return "B10X6G10X6R10X6G10X6422Unorm4Pack16";
      case Format::eG10X6B10X6R10X63Plane420Unorm3Pack16: return "G10X6B10X6R10X63Plane420Unorm3Pack16";
      case Format::eG10X6B10X6R10X62Plane420Unorm3Pack16: return "G10X6B10X6R10X62Plane420Unorm3Pack16";
      case Format::eG10X6B10X6R10X63Plane422Unorm3Pack16: return "G10X6B10X6R10X63Plane422Unorm3Pack16";
      case Format::eG10X6B10X6R10X62Plane422Unorm3Pack16: return "G10X6B10X6R10X62Plane422Unorm3Pack16";
      case Format::eG10X6B10X6R10X63Plane444Unorm3Pack16: return "G10X6B10X6R10X63Plane444Unorm3Pack16";
      case Format::eR12X4UnormPack16: return "R12X4UnormPack16";
      case Format::eR12X4G12X4Unorm2Pack16: return "R12X4G12X4Unorm2Pack16";
      case Format::eR12X4G12X4B12X4A12X4Unorm4Pack16: return "R12X4G12X4B12X4A12X4Unorm4Pack16";
      case Format::eG12X4B12X4G12X4R12X4422Unorm4Pack16: return "G12X4B12X4G12X4R12X4422Unorm4Pack16";
      case Format::eB12X4G12X4R12X4G12X4422Unorm4Pack16: return "B12X4G12X4R12X4G12X4422Unorm4Pack16";
      case Format::eG12X4B12X4R12X43Plane420Unorm3Pack16: return "G12X4B12X4R12X43Plane420Unorm3Pack16";
      case Format::eG12X4B12X4R12X42Plane420Unorm3Pack16: return "G12X4B12X4R12X42Plane420Unorm3Pack16";
      case Format::eG12X4B12X4R12X43Plane422Unorm3Pack16: return "G12X4B12X4R12X43Plane422Unorm3Pack16";
      case Format::eG12X4B12X4R12X42Plane422Unorm3Pack16: return "G12X4B12X4R12X42Plane422Unorm3Pack16";
      case Format::eG12X4B12X4R12X43Plane444Unorm3Pack16: return "G12X4B12X4R12X43Plane444Unorm3Pack16";
      case Format::eG16B16G16R16422Unorm: return "G16B16G16R16422Unorm";
      case Format::eB16G16R16G16422Unorm: return "B16G16R16G16422Unorm";
      case Format::eG16B16R163Plane420Unorm: return "G16B16R163Plane420Unorm";
      case Format::eG16B16R162Plane420Unorm: return "G16B16R162Plane420Unorm";
      case Format::eG16B16R163Plane422Unorm: return "G16B16R163Plane422Unorm";
      case Format::eG16B16R162Plane422Unorm: return "G16B16R162Plane422Unorm";
      case Format::eG16B16R163Plane444Unorm: return "G16B16R163Plane444Unorm";
      case Format::eG8B8R82Plane444Unorm: return "G8B8R82Plane444Unorm";
      case Format::eG10X6B10X6R10X62Plane444Unorm3Pack16: return "G10X6B10X6R10X62Plane444Unorm3Pack16";
      case Format::eG12X4B12X4R12X42Plane444Unorm3Pack16: return "G12X4B12X4R12X42Plane444Unorm3Pack16";
      case Format::eG16B16R162Plane444Unorm: return "G16B16R162Plane444Unorm";
      case Format::eA4R4G4B4UnormPack16: return "A4R4G4B4UnormPack16";
      case Format::eA4B4G4R4UnormPack16: return "A4B4G4R4UnormPack16";
      case Format::eAstc4x4SfloatBlock: return "Astc4x4SfloatBlock";
      case Format::eAstc5x4SfloatBlock: return "Astc5x4SfloatBlock";
      case Format::eAstc5x5SfloatBlock: return "Astc5x5SfloatBlock";
      case Format::eAstc6x5SfloatBlock: return "Astc6x5SfloatBlock";
      case Format::eAstc6x6SfloatBlock: return "Astc6x6SfloatBlock";
      case Format::eAstc8x5SfloatBlock: return "Astc8x5SfloatBlock";
      case Format::eAstc8x6SfloatBlock: return "Astc8x6SfloatBlock";
      case Format::eAstc8x8SfloatBlock: return "Astc8x8SfloatBlock";
      case Format::eAstc10x5SfloatBlock: return "Astc10x5SfloatBlock";
      case Format::eAstc10x6SfloatBlock: return "Astc10x6SfloatBlock";
      case Format::eAstc10x8SfloatBlock: return "Astc10x8SfloatBlock";
      case Format::eAstc10x10SfloatBlock: return "Astc10x10SfloatBlock";
      case Format::eAstc12x10SfloatBlock: return "Astc12x10SfloatBlock";
      case Format::eAstc12x12SfloatBlock: return "Astc12x12SfloatBlock";
      case Format::ePvrtc12BppUnormBlockIMG: return "Pvrtc12BppUnormBlockIMG";
      case Format::ePvrtc14BppUnormBlockIMG: return "Pvrtc14BppUnormBlockIMG";
      case Format::ePvrtc22BppUnormBlockIMG: return "Pvrtc22BppUnormBlockIMG";
      case Format::ePvrtc24BppUnormBlockIMG: return "Pvrtc24BppUnormBlockIMG";
      case Format::ePvrtc12BppSrgbBlockIMG: return "Pvrtc12BppSrgbBlockIMG";
      case Format::ePvrtc14BppSrgbBlockIMG: return "Pvrtc14BppSrgbBlockIMG";
      case Format::ePvrtc22BppSrgbBlockIMG: return "Pvrtc22BppSrgbBlockIMG";
      case Format::ePvrtc24BppSrgbBlockIMG: return "Pvrtc24BppSrgbBlockIMG";
      case Format::eR16G16Sfixed5NV: return "R16G16Sfixed5NV";
      case Format::eA1B5G5R5UnormPack16KHR: return "A1B5G5R5UnormPack16KHR";
      case Format::eA8UnormKHR: return "A8UnormKHR";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( FormatFeatureFlagBits value )
  {
    switch ( value )
    {
      case FormatFeatureFlagBits::eSampledImage: return "SampledImage";
      case FormatFeatureFlagBits::eStorageImage: return "StorageImage";
      case FormatFeatureFlagBits::eStorageImageAtomic: return "StorageImageAtomic";
      case FormatFeatureFlagBits::eUniformTexelBuffer: return "UniformTexelBuffer";
      case FormatFeatureFlagBits::eStorageTexelBuffer: return "StorageTexelBuffer";
      case FormatFeatureFlagBits::eStorageTexelBufferAtomic: return "StorageTexelBufferAtomic";
      case FormatFeatureFlagBits::eVertexBuffer: return "VertexBuffer";
      case FormatFeatureFlagBits::eColorAttachment: return "ColorAttachment";
      case FormatFeatureFlagBits::eColorAttachmentBlend: return "ColorAttachmentBlend";
      case FormatFeatureFlagBits::eDepthStencilAttachment: return "DepthStencilAttachment";
      case FormatFeatureFlagBits::eBlitSrc: return "BlitSrc";
      case FormatFeatureFlagBits::eBlitDst: return "BlitDst";
      case FormatFeatureFlagBits::eSampledImageFilterLinear: return "SampledImageFilterLinear";
      case FormatFeatureFlagBits::eTransferSrc: return "TransferSrc";
      case FormatFeatureFlagBits::eTransferDst: return "TransferDst";
      case FormatFeatureFlagBits::eMidpointChromaSamples: return "MidpointChromaSamples";
      case FormatFeatureFlagBits::eSampledImageYcbcrConversionLinearFilter: return "SampledImageYcbcrConversionLinearFilter";
      case FormatFeatureFlagBits::eSampledImageYcbcrConversionSeparateReconstructionFilter: return "SampledImageYcbcrConversionSeparateReconstructionFilter";
      case FormatFeatureFlagBits::eSampledImageYcbcrConversionChromaReconstructionExplicit: return "SampledImageYcbcrConversionChromaReconstructionExplicit";
      case FormatFeatureFlagBits::eSampledImageYcbcrConversionChromaReconstructionExplicitForceable:
        return "SampledImageYcbcrConversionChromaReconstructionExplicitForceable";
      case FormatFeatureFlagBits::eDisjoint: return "Disjoint";
      case FormatFeatureFlagBits::eCositedChromaSamples: return "CositedChromaSamples";
      case FormatFeatureFlagBits::eSampledImageFilterMinmax: return "SampledImageFilterMinmax";
      case FormatFeatureFlagBits::eVideoDecodeOutputKHR: return "VideoDecodeOutputKHR";
      case FormatFeatureFlagBits::eVideoDecodeDpbKHR: return "VideoDecodeDpbKHR";
      case FormatFeatureFlagBits::eAccelerationStructureVertexBufferKHR: return "AccelerationStructureVertexBufferKHR";
      case FormatFeatureFlagBits::eSampledImageFilterCubicEXT: return "SampledImageFilterCubicEXT";
      case FormatFeatureFlagBits::eFragmentDensityMapEXT: return "FragmentDensityMapEXT";
      case FormatFeatureFlagBits::eFragmentShadingRateAttachmentKHR: return "FragmentShadingRateAttachmentKHR";
      case FormatFeatureFlagBits::eVideoEncodeInputKHR: return "VideoEncodeInputKHR";
      case FormatFeatureFlagBits::eVideoEncodeDpbKHR: return "VideoEncodeDpbKHR";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ImageCreateFlagBits value )
  {
    switch ( value )
    {
      case ImageCreateFlagBits::eSparseBinding: return "SparseBinding";
      case ImageCreateFlagBits::eSparseResidency: return "SparseResidency";
      case ImageCreateFlagBits::eSparseAliased: return "SparseAliased";
      case ImageCreateFlagBits::eMutableFormat: return "MutableFormat";
      case ImageCreateFlagBits::eCubeCompatible: return "CubeCompatible";
      case ImageCreateFlagBits::eAlias: return "Alias";
      case ImageCreateFlagBits::eSplitInstanceBindRegions: return "SplitInstanceBindRegions";
      case ImageCreateFlagBits::e2DArrayCompatible: return "2DArrayCompatible";
      case ImageCreateFlagBits::eBlockTexelViewCompatible: return "BlockTexelViewCompatible";
      case ImageCreateFlagBits::eExtendedUsage: return "ExtendedUsage";
      case ImageCreateFlagBits::eProtected: return "Protected";
      case ImageCreateFlagBits::eDisjoint: return "Disjoint";
      case ImageCreateFlagBits::eCornerSampledNV: return "CornerSampledNV";
      case ImageCreateFlagBits::eSampleLocationsCompatibleDepthEXT: return "SampleLocationsCompatibleDepthEXT";
      case ImageCreateFlagBits::eSubsampledEXT: return "SubsampledEXT";
      case ImageCreateFlagBits::eDescriptorBufferCaptureReplayEXT: return "DescriptorBufferCaptureReplayEXT";
      case ImageCreateFlagBits::eMultisampledRenderToSingleSampledEXT: return "MultisampledRenderToSingleSampledEXT";
      case ImageCreateFlagBits::e2DViewCompatibleEXT: return "2DViewCompatibleEXT";
      case ImageCreateFlagBits::eFragmentDensityMapOffsetQCOM: return "FragmentDensityMapOffsetQCOM";
      case ImageCreateFlagBits::eVideoProfileIndependentKHR: return "VideoProfileIndependentKHR";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ImageTiling value )
  {
    switch ( value )
    {
      case ImageTiling::eOptimal: return "Optimal";
      case ImageTiling::eLinear: return "Linear";
      case ImageTiling::eDrmFormatModifierEXT: return "DrmFormatModifierEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ImageType value )
  {
    switch ( value )
    {
      case ImageType::e1D: return "1D";
      case ImageType::e2D: return "2D";
      case ImageType::e3D: return "3D";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ImageUsageFlagBits value )
  {
    switch ( value )
    {
      case ImageUsageFlagBits::eTransferSrc: return "TransferSrc";
      case ImageUsageFlagBits::eTransferDst: return "TransferDst";
      case ImageUsageFlagBits::eSampled: return "Sampled";
      case ImageUsageFlagBits::eStorage: return "Storage";
      case ImageUsageFlagBits::eColorAttachment: return "ColorAttachment";
      case ImageUsageFlagBits::eDepthStencilAttachment: return "DepthStencilAttachment";
      case ImageUsageFlagBits::eTransientAttachment: return "TransientAttachment";
      case ImageUsageFlagBits::eInputAttachment: return "InputAttachment";
      case ImageUsageFlagBits::eVideoDecodeDstKHR: return "VideoDecodeDstKHR";
      case ImageUsageFlagBits::eVideoDecodeSrcKHR: return "VideoDecodeSrcKHR";
      case ImageUsageFlagBits::eVideoDecodeDpbKHR: return "VideoDecodeDpbKHR";
      case ImageUsageFlagBits::eFragmentDensityMapEXT: return "FragmentDensityMapEXT";
      case ImageUsageFlagBits::eFragmentShadingRateAttachmentKHR: return "FragmentShadingRateAttachmentKHR";
      case ImageUsageFlagBits::eHostTransferEXT: return "HostTransferEXT";
      case ImageUsageFlagBits::eVideoEncodeDstKHR: return "VideoEncodeDstKHR";
      case ImageUsageFlagBits::eVideoEncodeSrcKHR: return "VideoEncodeSrcKHR";
      case ImageUsageFlagBits::eVideoEncodeDpbKHR: return "VideoEncodeDpbKHR";
      case ImageUsageFlagBits::eAttachmentFeedbackLoopEXT: return "AttachmentFeedbackLoopEXT";
      case ImageUsageFlagBits::eInvocationMaskHUAWEI: return "InvocationMaskHUAWEI";
      case ImageUsageFlagBits::eSampleWeightQCOM: return "SampleWeightQCOM";
      case ImageUsageFlagBits::eSampleBlockMatchQCOM: return "SampleBlockMatchQCOM";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( InstanceCreateFlagBits value )
  {
    switch ( value )
    {
      case InstanceCreateFlagBits::eEnumeratePortabilityKHR: return "EnumeratePortabilityKHR";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( InternalAllocationType value )
  {
    switch ( value )
    {
      case InternalAllocationType::eExecutable: return "Executable";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( MemoryHeapFlagBits value )
  {
    switch ( value )
    {
      case MemoryHeapFlagBits::eDeviceLocal: return "DeviceLocal";
      case MemoryHeapFlagBits::eMultiInstance: return "MultiInstance";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( MemoryPropertyFlagBits value )
  {
    switch ( value )
    {
      case MemoryPropertyFlagBits::eDeviceLocal: return "DeviceLocal";
      case MemoryPropertyFlagBits::eHostVisible: return "HostVisible";
      case MemoryPropertyFlagBits::eHostCoherent: return "HostCoherent";
      case MemoryPropertyFlagBits::eHostCached: return "HostCached";
      case MemoryPropertyFlagBits::eLazilyAllocated: return "LazilyAllocated";
      case MemoryPropertyFlagBits::eProtected: return "Protected";
      case MemoryPropertyFlagBits::eDeviceCoherentAMD: return "DeviceCoherentAMD";
      case MemoryPropertyFlagBits::eDeviceUncachedAMD: return "DeviceUncachedAMD";
      case MemoryPropertyFlagBits::eRdmaCapableNV: return "RdmaCapableNV";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PhysicalDeviceType value )
  {
    switch ( value )
    {
      case PhysicalDeviceType::eOther: return "Other";
      case PhysicalDeviceType::eIntegratedGpu: return "IntegratedGpu";
      case PhysicalDeviceType::eDiscreteGpu: return "DiscreteGpu";
      case PhysicalDeviceType::eVirtualGpu: return "VirtualGpu";
      case PhysicalDeviceType::eCpu: return "Cpu";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( QueueFlagBits value )
  {
    switch ( value )
    {
      case QueueFlagBits::eGraphics: return "Graphics";
      case QueueFlagBits::eCompute: return "Compute";
      case QueueFlagBits::eTransfer: return "Transfer";
      case QueueFlagBits::eSparseBinding: return "SparseBinding";
      case QueueFlagBits::eProtected: return "Protected";
      case QueueFlagBits::eVideoDecodeKHR: return "VideoDecodeKHR";
      case QueueFlagBits::eVideoEncodeKHR: return "VideoEncodeKHR";
      case QueueFlagBits::eOpticalFlowNV: return "OpticalFlowNV";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( SampleCountFlagBits value )
  {
    switch ( value )
    {
      case SampleCountFlagBits::e1: return "1";
      case SampleCountFlagBits::e2: return "2";
      case SampleCountFlagBits::e4: return "4";
      case SampleCountFlagBits::e8: return "8";
      case SampleCountFlagBits::e16: return "16";
      case SampleCountFlagBits::e32: return "32";
      case SampleCountFlagBits::e64: return "64";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( SystemAllocationScope value )
  {
    switch ( value )
    {
      case SystemAllocationScope::eCommand: return "Command";
      case SystemAllocationScope::eObject: return "Object";
      case SystemAllocationScope::eCache: return "Cache";
      case SystemAllocationScope::eDevice: return "Device";
      case SystemAllocationScope::eInstance: return "Instance";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( DeviceCreateFlagBits )
  {
    return "(void)";
  }

  VULKAN_HPP_INLINE std::string to_string( DeviceQueueCreateFlagBits value )
  {
    switch ( value )
    {
      case DeviceQueueCreateFlagBits::eProtected: return "Protected";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineStageFlagBits value )
  {
    switch ( value )
    {
      case PipelineStageFlagBits::eTopOfPipe: return "TopOfPipe";
      case PipelineStageFlagBits::eDrawIndirect: return "DrawIndirect";
      case PipelineStageFlagBits::eVertexInput: return "VertexInput";
      case PipelineStageFlagBits::eVertexShader: return "VertexShader";
      case PipelineStageFlagBits::eTessellationControlShader: return "TessellationControlShader";
      case PipelineStageFlagBits::eTessellationEvaluationShader: return "TessellationEvaluationShader";
      case PipelineStageFlagBits::eGeometryShader: return "GeometryShader";
      case PipelineStageFlagBits::eFragmentShader: return "FragmentShader";
      case PipelineStageFlagBits::eEarlyFragmentTests: return "EarlyFragmentTests";
      case PipelineStageFlagBits::eLateFragmentTests: return "LateFragmentTests";
      case PipelineStageFlagBits::eColorAttachmentOutput: return "ColorAttachmentOutput";
      case PipelineStageFlagBits::eComputeShader: return "ComputeShader";
      case PipelineStageFlagBits::eTransfer: return "Transfer";
      case PipelineStageFlagBits::eBottomOfPipe: return "BottomOfPipe";
      case PipelineStageFlagBits::eHost: return "Host";
      case PipelineStageFlagBits::eAllGraphics: return "AllGraphics";
      case PipelineStageFlagBits::eAllCommands: return "AllCommands";
      case PipelineStageFlagBits::eNone: return "None";
      case PipelineStageFlagBits::eTransformFeedbackEXT: return "TransformFeedbackEXT";
      case PipelineStageFlagBits::eConditionalRenderingEXT: return "ConditionalRenderingEXT";
      case PipelineStageFlagBits::eAccelerationStructureBuildKHR: return "AccelerationStructureBuildKHR";
      case PipelineStageFlagBits::eRayTracingShaderKHR: return "RayTracingShaderKHR";
      case PipelineStageFlagBits::eFragmentDensityProcessEXT: return "FragmentDensityProcessEXT";
      case PipelineStageFlagBits::eFragmentShadingRateAttachmentKHR: return "FragmentShadingRateAttachmentKHR";
      case PipelineStageFlagBits::eCommandPreprocessNV: return "CommandPreprocessNV";
      case PipelineStageFlagBits::eTaskShaderEXT: return "TaskShaderEXT";
      case PipelineStageFlagBits::eMeshShaderEXT: return "MeshShaderEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( MemoryMapFlagBits value )
  {
    switch ( value )
    {
      case MemoryMapFlagBits::ePlacedEXT: return "PlacedEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ImageAspectFlagBits value )
  {
    switch ( value )
    {
      case ImageAspectFlagBits::eColor: return "Color";
      case ImageAspectFlagBits::eDepth: return "Depth";
      case ImageAspectFlagBits::eStencil: return "Stencil";
      case ImageAspectFlagBits::eMetadata: return "Metadata";
      case ImageAspectFlagBits::ePlane0: return "Plane0";
      case ImageAspectFlagBits::ePlane1: return "Plane1";
      case ImageAspectFlagBits::ePlane2: return "Plane2";
      case ImageAspectFlagBits::eNone: return "None";
      case ImageAspectFlagBits::eMemoryPlane0EXT: return "MemoryPlane0EXT";
      case ImageAspectFlagBits::eMemoryPlane1EXT: return "MemoryPlane1EXT";
      case ImageAspectFlagBits::eMemoryPlane2EXT: return "MemoryPlane2EXT";
      case ImageAspectFlagBits::eMemoryPlane3EXT: return "MemoryPlane3EXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( SparseImageFormatFlagBits value )
  {
    switch ( value )
    {
      case SparseImageFormatFlagBits::eSingleMiptail: return "SingleMiptail";
      case SparseImageFormatFlagBits::eAlignedMipSize: return "AlignedMipSize";
      case SparseImageFormatFlagBits::eNonstandardBlockSize: return "NonstandardBlockSize";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( SparseMemoryBindFlagBits value )
  {
    switch ( value )
    {
      case SparseMemoryBindFlagBits::eMetadata: return "Metadata";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( FenceCreateFlagBits value )
  {
    switch ( value )
    {
      case FenceCreateFlagBits::eSignaled: return "Signaled";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( SemaphoreCreateFlagBits )
  {
    return "(void)";
  }

  VULKAN_HPP_INLINE std::string to_string( EventCreateFlagBits value )
  {
    switch ( value )
    {
      case EventCreateFlagBits::eDeviceOnly: return "DeviceOnly";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( QueryPipelineStatisticFlagBits value )
  {
    switch ( value )
    {
      case QueryPipelineStatisticFlagBits::eInputAssemblyVertices: return "InputAssemblyVertices";
      case QueryPipelineStatisticFlagBits::eInputAssemblyPrimitives: return "InputAssemblyPrimitives";
      case QueryPipelineStatisticFlagBits::eVertexShaderInvocations: return "VertexShaderInvocations";
      case QueryPipelineStatisticFlagBits::eGeometryShaderInvocations: return "GeometryShaderInvocations";
      case QueryPipelineStatisticFlagBits::eGeometryShaderPrimitives: return "GeometryShaderPrimitives";
      case QueryPipelineStatisticFlagBits::eClippingInvocations: return "ClippingInvocations";
      case QueryPipelineStatisticFlagBits::eClippingPrimitives: return "ClippingPrimitives";
      case QueryPipelineStatisticFlagBits::eFragmentShaderInvocations: return "FragmentShaderInvocations";
      case QueryPipelineStatisticFlagBits::eTessellationControlShaderPatches: return "TessellationControlShaderPatches";
      case QueryPipelineStatisticFlagBits::eTessellationEvaluationShaderInvocations: return "TessellationEvaluationShaderInvocations";
      case QueryPipelineStatisticFlagBits::eComputeShaderInvocations: return "ComputeShaderInvocations";
      case QueryPipelineStatisticFlagBits::eTaskShaderInvocationsEXT: return "TaskShaderInvocationsEXT";
      case QueryPipelineStatisticFlagBits::eMeshShaderInvocationsEXT: return "MeshShaderInvocationsEXT";
      case QueryPipelineStatisticFlagBits::eClusterCullingShaderInvocationsHUAWEI: return "ClusterCullingShaderInvocationsHUAWEI";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( QueryResultFlagBits value )
  {
    switch ( value )
    {
      case QueryResultFlagBits::e64: return "64";
      case QueryResultFlagBits::eWait: return "Wait";
      case QueryResultFlagBits::eWithAvailability: return "WithAvailability";
      case QueryResultFlagBits::ePartial: return "Partial";
      case QueryResultFlagBits::eWithStatusKHR: return "WithStatusKHR";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( QueryType value )
  {
    switch ( value )
    {
      case QueryType::eOcclusion: return "Occlusion";
      case QueryType::ePipelineStatistics: return "PipelineStatistics";
      case QueryType::eTimestamp: return "Timestamp";
      case QueryType::eResultStatusOnlyKHR: return "ResultStatusOnlyKHR";
      case QueryType::eTransformFeedbackStreamEXT: return "TransformFeedbackStreamEXT";
      case QueryType::ePerformanceQueryKHR: return "PerformanceQueryKHR";
      case QueryType::eAccelerationStructureCompactedSizeKHR: return "AccelerationStructureCompactedSizeKHR";
      case QueryType::eAccelerationStructureSerializationSizeKHR: return "AccelerationStructureSerializationSizeKHR";
      case QueryType::eAccelerationStructureCompactedSizeNV: return "AccelerationStructureCompactedSizeNV";
      case QueryType::ePerformanceQueryINTEL: return "PerformanceQueryINTEL";
      case QueryType::eVideoEncodeFeedbackKHR: return "VideoEncodeFeedbackKHR";
      case QueryType::eMeshPrimitivesGeneratedEXT: return "MeshPrimitivesGeneratedEXT";
      case QueryType::ePrimitivesGeneratedEXT: return "PrimitivesGeneratedEXT";
      case QueryType::eAccelerationStructureSerializationBottomLevelPointersKHR: return "AccelerationStructureSerializationBottomLevelPointersKHR";
      case QueryType::eAccelerationStructureSizeKHR: return "AccelerationStructureSizeKHR";
      case QueryType::eMicromapSerializationSizeEXT: return "MicromapSerializationSizeEXT";
      case QueryType::eMicromapCompactedSizeEXT: return "MicromapCompactedSizeEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( QueryPoolCreateFlagBits )
  {
    return "(void)";
  }

  VULKAN_HPP_INLINE std::string to_string( BufferCreateFlagBits value )
  {
    switch ( value )
    {
      case BufferCreateFlagBits::eSparseBinding: return "SparseBinding";
      case BufferCreateFlagBits::eSparseResidency: return "SparseResidency";
      case BufferCreateFlagBits::eSparseAliased: return "SparseAliased";
      case BufferCreateFlagBits::eProtected: return "Protected";
      case BufferCreateFlagBits::eDeviceAddressCaptureReplay: return "DeviceAddressCaptureReplay";
      case BufferCreateFlagBits::eDescriptorBufferCaptureReplayEXT: return "DescriptorBufferCaptureReplayEXT";
      case BufferCreateFlagBits::eVideoProfileIndependentKHR: return "VideoProfileIndependentKHR";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( BufferUsageFlagBits value )
  {
    switch ( value )
    {
      case BufferUsageFlagBits::eTransferSrc: return "TransferSrc";
      case BufferUsageFlagBits::eTransferDst: return "TransferDst";
      case BufferUsageFlagBits::eUniformTexelBuffer: return "UniformTexelBuffer";
      case BufferUsageFlagBits::eStorageTexelBuffer: return "StorageTexelBuffer";
      case BufferUsageFlagBits::eUniformBuffer: return "UniformBuffer";
      case BufferUsageFlagBits::eStorageBuffer: return "StorageBuffer";
      case BufferUsageFlagBits::eIndexBuffer: return "IndexBuffer";
      case BufferUsageFlagBits::eVertexBuffer: return "VertexBuffer";
      case BufferUsageFlagBits::eIndirectBuffer: return "IndirectBuffer";
      case BufferUsageFlagBits::eShaderDeviceAddress: return "ShaderDeviceAddress";
      case BufferUsageFlagBits::eVideoDecodeSrcKHR: return "VideoDecodeSrcKHR";
      case BufferUsageFlagBits::eVideoDecodeDstKHR: return "VideoDecodeDstKHR";
      case BufferUsageFlagBits::eTransformFeedbackBufferEXT: return "TransformFeedbackBufferEXT";
      case BufferUsageFlagBits::eTransformFeedbackCounterBufferEXT: return "TransformFeedbackCounterBufferEXT";
      case BufferUsageFlagBits::eConditionalRenderingEXT: return "ConditionalRenderingEXT";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case BufferUsageFlagBits::eExecutionGraphScratchAMDX: return "ExecutionGraphScratchAMDX";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      case BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR: return "AccelerationStructureBuildInputReadOnlyKHR";
      case BufferUsageFlagBits::eAccelerationStructureStorageKHR: return "AccelerationStructureStorageKHR";
      case BufferUsageFlagBits::eShaderBindingTableKHR: return "ShaderBindingTableKHR";
      case BufferUsageFlagBits::eVideoEncodeDstKHR: return "VideoEncodeDstKHR";
      case BufferUsageFlagBits::eVideoEncodeSrcKHR: return "VideoEncodeSrcKHR";
      case BufferUsageFlagBits::eSamplerDescriptorBufferEXT: return "SamplerDescriptorBufferEXT";
      case BufferUsageFlagBits::eResourceDescriptorBufferEXT: return "ResourceDescriptorBufferEXT";
      case BufferUsageFlagBits::ePushDescriptorsDescriptorBufferEXT: return "PushDescriptorsDescriptorBufferEXT";
      case BufferUsageFlagBits::eMicromapBuildInputReadOnlyEXT: return "MicromapBuildInputReadOnlyEXT";
      case BufferUsageFlagBits::eMicromapStorageEXT: return "MicromapStorageEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( SharingMode value )
  {
    switch ( value )
    {
      case SharingMode::eExclusive: return "Exclusive";
      case SharingMode::eConcurrent: return "Concurrent";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( BufferViewCreateFlagBits )
  {
    return "(void)";
  }

  VULKAN_HPP_INLINE std::string to_string( ImageLayout value )
  {
    switch ( value )
    {
      case ImageLayout::eUndefined: return "Undefined";
      case ImageLayout::eGeneral: return "General";
      case ImageLayout::eColorAttachmentOptimal: return "ColorAttachmentOptimal";
      case ImageLayout::eDepthStencilAttachmentOptimal: return "DepthStencilAttachmentOptimal";
      case ImageLayout::eDepthStencilReadOnlyOptimal: return "DepthStencilReadOnlyOptimal";
      case ImageLayout::eShaderReadOnlyOptimal: return "ShaderReadOnlyOptimal";
      case ImageLayout::eTransferSrcOptimal: return "TransferSrcOptimal";
      case ImageLayout::eTransferDstOptimal: return "TransferDstOptimal";
      case ImageLayout::ePreinitialized: return "Preinitialized";
      case ImageLayout::eDepthReadOnlyStencilAttachmentOptimal: return "DepthReadOnlyStencilAttachmentOptimal";
      case ImageLayout::eDepthAttachmentStencilReadOnlyOptimal: return "DepthAttachmentStencilReadOnlyOptimal";
      case ImageLayout::eDepthAttachmentOptimal: return "DepthAttachmentOptimal";
      case ImageLayout::eDepthReadOnlyOptimal: return "DepthReadOnlyOptimal";
      case ImageLayout::eStencilAttachmentOptimal: return "StencilAttachmentOptimal";
      case ImageLayout::eStencilReadOnlyOptimal: return "StencilReadOnlyOptimal";
      case ImageLayout::eReadOnlyOptimal: return "ReadOnlyOptimal";
      case ImageLayout::eAttachmentOptimal: return "AttachmentOptimal";
      case ImageLayout::ePresentSrcKHR: return "PresentSrcKHR";
      case ImageLayout::eVideoDecodeDstKHR: return "VideoDecodeDstKHR";
      case ImageLayout::eVideoDecodeSrcKHR: return "VideoDecodeSrcKHR";
      case ImageLayout::eVideoDecodeDpbKHR: return "VideoDecodeDpbKHR";
      case ImageLayout::eSharedPresentKHR: return "SharedPresentKHR";
      case ImageLayout::eFragmentDensityMapOptimalEXT: return "FragmentDensityMapOptimalEXT";
      case ImageLayout::eFragmentShadingRateAttachmentOptimalKHR: return "FragmentShadingRateAttachmentOptimalKHR";
      case ImageLayout::eRenderingLocalReadKHR: return "RenderingLocalReadKHR";
      case ImageLayout::eVideoEncodeDstKHR: return "VideoEncodeDstKHR";
      case ImageLayout::eVideoEncodeSrcKHR: return "VideoEncodeSrcKHR";
      case ImageLayout::eVideoEncodeDpbKHR: return "VideoEncodeDpbKHR";
      case ImageLayout::eAttachmentFeedbackLoopOptimalEXT: return "AttachmentFeedbackLoopOptimalEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ComponentSwizzle value )
  {
    switch ( value )
    {
      case ComponentSwizzle::eIdentity: return "Identity";
      case ComponentSwizzle::eZero: return "Zero";
      case ComponentSwizzle::eOne: return "One";
      case ComponentSwizzle::eR: return "R";
      case ComponentSwizzle::eG: return "G";
      case ComponentSwizzle::eB: return "B";
      case ComponentSwizzle::eA: return "A";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ImageViewCreateFlagBits value )
  {
    switch ( value )
    {
      case ImageViewCreateFlagBits::eFragmentDensityMapDynamicEXT: return "FragmentDensityMapDynamicEXT";
      case ImageViewCreateFlagBits::eDescriptorBufferCaptureReplayEXT: return "DescriptorBufferCaptureReplayEXT";
      case ImageViewCreateFlagBits::eFragmentDensityMapDeferredEXT: return "FragmentDensityMapDeferredEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ImageViewType value )
  {
    switch ( value )
    {
      case ImageViewType::e1D: return "1D";
      case ImageViewType::e2D: return "2D";
      case ImageViewType::e3D: return "3D";
      case ImageViewType::eCube: return "Cube";
      case ImageViewType::e1DArray: return "1DArray";
      case ImageViewType::e2DArray: return "2DArray";
      case ImageViewType::eCubeArray: return "CubeArray";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ShaderModuleCreateFlagBits )
  {
    return "(void)";
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineCacheCreateFlagBits value )
  {
    switch ( value )
    {
      case PipelineCacheCreateFlagBits::eExternallySynchronized: return "ExternallySynchronized";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( BlendFactor value )
  {
    switch ( value )
    {
      case BlendFactor::eZero: return "Zero";
      case BlendFactor::eOne: return "One";
      case BlendFactor::eSrcColor: return "SrcColor";
      case BlendFactor::eOneMinusSrcColor: return "OneMinusSrcColor";
      case BlendFactor::eDstColor: return "DstColor";
      case BlendFactor::eOneMinusDstColor: return "OneMinusDstColor";
      case BlendFactor::eSrcAlpha: return "SrcAlpha";
      case BlendFactor::eOneMinusSrcAlpha: return "OneMinusSrcAlpha";
      case BlendFactor::eDstAlpha: return "DstAlpha";
      case BlendFactor::eOneMinusDstAlpha: return "OneMinusDstAlpha";
      case BlendFactor::eConstantColor: return "ConstantColor";
      case BlendFactor::eOneMinusConstantColor: return "OneMinusConstantColor";
      case BlendFactor::eConstantAlpha: return "ConstantAlpha";
      case BlendFactor::eOneMinusConstantAlpha: return "OneMinusConstantAlpha";
      case BlendFactor::eSrcAlphaSaturate: return "SrcAlphaSaturate";
      case BlendFactor::eSrc1Color: return "Src1Color";
      case BlendFactor::eOneMinusSrc1Color: return "OneMinusSrc1Color";
      case BlendFactor::eSrc1Alpha: return "Src1Alpha";
      case BlendFactor::eOneMinusSrc1Alpha: return "OneMinusSrc1Alpha";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( BlendOp value )
  {
    switch ( value )
    {
      case BlendOp::eAdd: return "Add";
      case BlendOp::eSubtract: return "Subtract";
      case BlendOp::eReverseSubtract: return "ReverseSubtract";
      case BlendOp::eMin: return "Min";
      case BlendOp::eMax: return "Max";
      case BlendOp::eZeroEXT: return "ZeroEXT";
      case BlendOp::eSrcEXT: return "SrcEXT";
      case BlendOp::eDstEXT: return "DstEXT";
      case BlendOp::eSrcOverEXT: return "SrcOverEXT";
      case BlendOp::eDstOverEXT: return "DstOverEXT";
      case BlendOp::eSrcInEXT: return "SrcInEXT";
      case BlendOp::eDstInEXT: return "DstInEXT";
      case BlendOp::eSrcOutEXT: return "SrcOutEXT";
      case BlendOp::eDstOutEXT: return "DstOutEXT";
      case BlendOp::eSrcAtopEXT: return "SrcAtopEXT";
      case BlendOp::eDstAtopEXT: return "DstAtopEXT";
      case BlendOp::eXorEXT: return "XorEXT";
      case BlendOp::eMultiplyEXT: return "MultiplyEXT";
      case BlendOp::eScreenEXT: return "ScreenEXT";
      case BlendOp::eOverlayEXT: return "OverlayEXT";
      case BlendOp::eDarkenEXT: return "DarkenEXT";
      case BlendOp::eLightenEXT: return "LightenEXT";
      case BlendOp::eColordodgeEXT: return "ColordodgeEXT";
      case BlendOp::eColorburnEXT: return "ColorburnEXT";
      case BlendOp::eHardlightEXT: return "HardlightEXT";
      case BlendOp::eSoftlightEXT: return "SoftlightEXT";
      case BlendOp::eDifferenceEXT: return "DifferenceEXT";
      case BlendOp::eExclusionEXT: return "ExclusionEXT";
      case BlendOp::eInvertEXT: return "InvertEXT";
      case BlendOp::eInvertRgbEXT: return "InvertRgbEXT";
      case BlendOp::eLineardodgeEXT: return "LineardodgeEXT";
      case BlendOp::eLinearburnEXT: return "LinearburnEXT";
      case BlendOp::eVividlightEXT: return "VividlightEXT";
      case BlendOp::eLinearlightEXT: return "LinearlightEXT";
      case BlendOp::ePinlightEXT: return "PinlightEXT";
      case BlendOp::eHardmixEXT: return "HardmixEXT";
      case BlendOp::eHslHueEXT: return "HslHueEXT";
      case BlendOp::eHslSaturationEXT: return "HslSaturationEXT";
      case BlendOp::eHslColorEXT: return "HslColorEXT";
      case BlendOp::eHslLuminosityEXT: return "HslLuminosityEXT";
      case BlendOp::ePlusEXT: return "PlusEXT";
      case BlendOp::ePlusClampedEXT: return "PlusClampedEXT";
      case BlendOp::ePlusClampedAlphaEXT: return "PlusClampedAlphaEXT";
      case BlendOp::ePlusDarkerEXT: return "PlusDarkerEXT";
      case BlendOp::eMinusEXT: return "MinusEXT";
      case BlendOp::eMinusClampedEXT: return "MinusClampedEXT";
      case BlendOp::eContrastEXT: return "ContrastEXT";
      case BlendOp::eInvertOvgEXT: return "InvertOvgEXT";
      case BlendOp::eRedEXT: return "RedEXT";
      case BlendOp::eGreenEXT: return "GreenEXT";
      case BlendOp::eBlueEXT: return "BlueEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ColorComponentFlagBits value )
  {
    switch ( value )
    {
      case ColorComponentFlagBits::eR: return "R";
      case ColorComponentFlagBits::eG: return "G";
      case ColorComponentFlagBits::eB: return "B";
      case ColorComponentFlagBits::eA: return "A";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( CompareOp value )
  {
    switch ( value )
    {
      case CompareOp::eNever: return "Never";
      case CompareOp::eLess: return "Less";
      case CompareOp::eEqual: return "Equal";
      case CompareOp::eLessOrEqual: return "LessOrEqual";
      case CompareOp::eGreater: return "Greater";
      case CompareOp::eNotEqual: return "NotEqual";
      case CompareOp::eGreaterOrEqual: return "GreaterOrEqual";
      case CompareOp::eAlways: return "Always";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( CullModeFlagBits value )
  {
    switch ( value )
    {
      case CullModeFlagBits::eNone: return "None";
      case CullModeFlagBits::eFront: return "Front";
      case CullModeFlagBits::eBack: return "Back";
      case CullModeFlagBits::eFrontAndBack: return "FrontAndBack";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( DynamicState value )
  {
    switch ( value )
    {
      case DynamicState::eViewport: return "Viewport";
      case DynamicState::eScissor: return "Scissor";
      case DynamicState::eLineWidth: return "LineWidth";
      case DynamicState::eDepthBias: return "DepthBias";
      case DynamicState::eBlendConstants: return "BlendConstants";
      case DynamicState::eDepthBounds: return "DepthBounds";
      case DynamicState::eStencilCompareMask: return "StencilCompareMask";
      case DynamicState::eStencilWriteMask: return "StencilWriteMask";
      case DynamicState::eStencilReference: return "StencilReference";
      case DynamicState::eCullMode: return "CullMode";
      case DynamicState::eFrontFace: return "FrontFace";
      case DynamicState::ePrimitiveTopology: return "PrimitiveTopology";
      case DynamicState::eViewportWithCount: return "ViewportWithCount";
      case DynamicState::eScissorWithCount: return "ScissorWithCount";
      case DynamicState::eVertexInputBindingStride: return "VertexInputBindingStride";
      case DynamicState::eDepthTestEnable: return "DepthTestEnable";
      case DynamicState::eDepthWriteEnable: return "DepthWriteEnable";
      case DynamicState::eDepthCompareOp: return "DepthCompareOp";
      case DynamicState::eDepthBoundsTestEnable: return "DepthBoundsTestEnable";
      case DynamicState::eStencilTestEnable: return "StencilTestEnable";
      case DynamicState::eStencilOp: return "StencilOp";
      case DynamicState::eRasterizerDiscardEnable: return "RasterizerDiscardEnable";
      case DynamicState::eDepthBiasEnable: return "DepthBiasEnable";
      case DynamicState::ePrimitiveRestartEnable: return "PrimitiveRestartEnable";
      case DynamicState::eViewportWScalingNV: return "ViewportWScalingNV";
      case DynamicState::eDiscardRectangleEXT: return "DiscardRectangleEXT";
      case DynamicState::eDiscardRectangleEnableEXT: return "DiscardRectangleEnableEXT";
      case DynamicState::eDiscardRectangleModeEXT: return "DiscardRectangleModeEXT";
      case DynamicState::eSampleLocationsEXT: return "SampleLocationsEXT";
      case DynamicState::eRayTracingPipelineStackSizeKHR: return "RayTracingPipelineStackSizeKHR";
      case DynamicState::eViewportShadingRatePaletteNV: return "ViewportShadingRatePaletteNV";
      case DynamicState::eViewportCoarseSampleOrderNV: return "ViewportCoarseSampleOrderNV";
      case DynamicState::eExclusiveScissorEnableNV: return "ExclusiveScissorEnableNV";
      case DynamicState::eExclusiveScissorNV: return "ExclusiveScissorNV";
      case DynamicState::eFragmentShadingRateKHR: return "FragmentShadingRateKHR";
      case DynamicState::eVertexInputEXT: return "VertexInputEXT";
      case DynamicState::ePatchControlPointsEXT: return "PatchControlPointsEXT";
      case DynamicState::eLogicOpEXT: return "LogicOpEXT";
      case DynamicState::eColorWriteEnableEXT: return "ColorWriteEnableEXT";
      case DynamicState::eDepthClampEnableEXT: return "DepthClampEnableEXT";
      case DynamicState::ePolygonModeEXT: return "PolygonModeEXT";
      case DynamicState::eRasterizationSamplesEXT: return "RasterizationSamplesEXT";
      case DynamicState::eSampleMaskEXT: return "SampleMaskEXT";
      case DynamicState::eAlphaToCoverageEnableEXT: return "AlphaToCoverageEnableEXT";
      case DynamicState::eAlphaToOneEnableEXT: return "AlphaToOneEnableEXT";
      case DynamicState::eLogicOpEnableEXT: return "LogicOpEnableEXT";
      case DynamicState::eColorBlendEnableEXT: return "ColorBlendEnableEXT";
      case DynamicState::eColorBlendEquationEXT: return "ColorBlendEquationEXT";
      case DynamicState::eColorWriteMaskEXT: return "ColorWriteMaskEXT";
      case DynamicState::eTessellationDomainOriginEXT: return "TessellationDomainOriginEXT";
      case DynamicState::eRasterizationStreamEXT: return "RasterizationStreamEXT";
      case DynamicState::eConservativeRasterizationModeEXT: return "ConservativeRasterizationModeEXT";
      case DynamicState::eExtraPrimitiveOverestimationSizeEXT: return "ExtraPrimitiveOverestimationSizeEXT";
      case DynamicState::eDepthClipEnableEXT: return "DepthClipEnableEXT";
      case DynamicState::eSampleLocationsEnableEXT: return "SampleLocationsEnableEXT";
      case DynamicState::eColorBlendAdvancedEXT: return "ColorBlendAdvancedEXT";
      case DynamicState::eProvokingVertexModeEXT: return "ProvokingVertexModeEXT";
      case DynamicState::eLineRasterizationModeEXT: return "LineRasterizationModeEXT";
      case DynamicState::eLineStippleEnableEXT: return "LineStippleEnableEXT";
      case DynamicState::eDepthClipNegativeOneToOneEXT: return "DepthClipNegativeOneToOneEXT";
      case DynamicState::eViewportWScalingEnableNV: return "ViewportWScalingEnableNV";
      case DynamicState::eViewportSwizzleNV: return "ViewportSwizzleNV";
      case DynamicState::eCoverageToColorEnableNV: return "CoverageToColorEnableNV";
      case DynamicState::eCoverageToColorLocationNV: return "CoverageToColorLocationNV";
      case DynamicState::eCoverageModulationModeNV: return "CoverageModulationModeNV";
      case DynamicState::eCoverageModulationTableEnableNV: return "CoverageModulationTableEnableNV";
      case DynamicState::eCoverageModulationTableNV: return "CoverageModulationTableNV";
      case DynamicState::eShadingRateImageEnableNV: return "ShadingRateImageEnableNV";
      case DynamicState::eRepresentativeFragmentTestEnableNV: return "RepresentativeFragmentTestEnableNV";
      case DynamicState::eCoverageReductionModeNV: return "CoverageReductionModeNV";
      case DynamicState::eAttachmentFeedbackLoopEnableEXT: return "AttachmentFeedbackLoopEnableEXT";
      case DynamicState::eLineStippleKHR: return "LineStippleKHR";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( FrontFace value )
  {
    switch ( value )
    {
      case FrontFace::eCounterClockwise: return "CounterClockwise";
      case FrontFace::eClockwise: return "Clockwise";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( LogicOp value )
  {
    switch ( value )
    {
      case LogicOp::eClear: return "Clear";
      case LogicOp::eAnd: return "And";
      case LogicOp::eAndReverse: return "AndReverse";
      case LogicOp::eCopy: return "Copy";
      case LogicOp::eAndInverted: return "AndInverted";
      case LogicOp::eNoOp: return "NoOp";
      case LogicOp::eXor: return "Xor";
      case LogicOp::eOr: return "Or";
      case LogicOp::eNor: return "Nor";
      case LogicOp::eEquivalent: return "Equivalent";
      case LogicOp::eInvert: return "Invert";
      case LogicOp::eOrReverse: return "OrReverse";
      case LogicOp::eCopyInverted: return "CopyInverted";
      case LogicOp::eOrInverted: return "OrInverted";
      case LogicOp::eNand: return "Nand";
      case LogicOp::eSet: return "Set";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineCreateFlagBits value )
  {
    switch ( value )
    {
      case PipelineCreateFlagBits::eDisableOptimization: return "DisableOptimization";
      case PipelineCreateFlagBits::eAllowDerivatives: return "AllowDerivatives";
      case PipelineCreateFlagBits::eDerivative: return "Derivative";
      case PipelineCreateFlagBits::eViewIndexFromDeviceIndex: return "ViewIndexFromDeviceIndex";
      case PipelineCreateFlagBits::eDispatchBase: return "DispatchBase";
      case PipelineCreateFlagBits::eFailOnPipelineCompileRequired: return "FailOnPipelineCompileRequired";
      case PipelineCreateFlagBits::eEarlyReturnOnFailure: return "EarlyReturnOnFailure";
      case PipelineCreateFlagBits::eRenderingFragmentShadingRateAttachmentKHR: return "RenderingFragmentShadingRateAttachmentKHR";
      case PipelineCreateFlagBits::eRenderingFragmentDensityMapAttachmentEXT: return "RenderingFragmentDensityMapAttachmentEXT";
      case PipelineCreateFlagBits::eRayTracingNoNullAnyHitShadersKHR: return "RayTracingNoNullAnyHitShadersKHR";
      case PipelineCreateFlagBits::eRayTracingNoNullClosestHitShadersKHR: return "RayTracingNoNullClosestHitShadersKHR";
      case PipelineCreateFlagBits::eRayTracingNoNullMissShadersKHR: return "RayTracingNoNullMissShadersKHR";
      case PipelineCreateFlagBits::eRayTracingNoNullIntersectionShadersKHR: return "RayTracingNoNullIntersectionShadersKHR";
      case PipelineCreateFlagBits::eRayTracingSkipTrianglesKHR: return "RayTracingSkipTrianglesKHR";
      case PipelineCreateFlagBits::eRayTracingSkipAabbsKHR: return "RayTracingSkipAabbsKHR";
      case PipelineCreateFlagBits::eRayTracingShaderGroupHandleCaptureReplayKHR: return "RayTracingShaderGroupHandleCaptureReplayKHR";
      case PipelineCreateFlagBits::eDeferCompileNV: return "DeferCompileNV";
      case PipelineCreateFlagBits::eCaptureStatisticsKHR: return "CaptureStatisticsKHR";
      case PipelineCreateFlagBits::eCaptureInternalRepresentationsKHR: return "CaptureInternalRepresentationsKHR";
      case PipelineCreateFlagBits::eIndirectBindableNV: return "IndirectBindableNV";
      case PipelineCreateFlagBits::eLibraryKHR: return "LibraryKHR";
      case PipelineCreateFlagBits::eDescriptorBufferEXT: return "DescriptorBufferEXT";
      case PipelineCreateFlagBits::eRetainLinkTimeOptimizationInfoEXT: return "RetainLinkTimeOptimizationInfoEXT";
      case PipelineCreateFlagBits::eLinkTimeOptimizationEXT: return "LinkTimeOptimizationEXT";
      case PipelineCreateFlagBits::eRayTracingAllowMotionNV: return "RayTracingAllowMotionNV";
      case PipelineCreateFlagBits::eColorAttachmentFeedbackLoopEXT: return "ColorAttachmentFeedbackLoopEXT";
      case PipelineCreateFlagBits::eDepthStencilAttachmentFeedbackLoopEXT: return "DepthStencilAttachmentFeedbackLoopEXT";
      case PipelineCreateFlagBits::eRayTracingOpacityMicromapEXT: return "RayTracingOpacityMicromapEXT";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case PipelineCreateFlagBits::eRayTracingDisplacementMicromapNV: return "RayTracingDisplacementMicromapNV";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      case PipelineCreateFlagBits::eNoProtectedAccessEXT: return "NoProtectedAccessEXT";
      case PipelineCreateFlagBits::eProtectedAccessOnlyEXT: return "ProtectedAccessOnlyEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineShaderStageCreateFlagBits value )
  {
    switch ( value )
    {
      case PipelineShaderStageCreateFlagBits::eAllowVaryingSubgroupSize: return "AllowVaryingSubgroupSize";
      case PipelineShaderStageCreateFlagBits::eRequireFullSubgroups: return "RequireFullSubgroups";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PolygonMode value )
  {
    switch ( value )
    {
      case PolygonMode::eFill: return "Fill";
      case PolygonMode::eLine: return "Line";
      case PolygonMode::ePoint: return "Point";
      case PolygonMode::eFillRectangleNV: return "FillRectangleNV";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PrimitiveTopology value )
  {
    switch ( value )
    {
      case PrimitiveTopology::ePointList: return "PointList";
      case PrimitiveTopology::eLineList: return "LineList";
      case PrimitiveTopology::eLineStrip: return "LineStrip";
      case PrimitiveTopology::eTriangleList: return "TriangleList";
      case PrimitiveTopology::eTriangleStrip: return "TriangleStrip";
      case PrimitiveTopology::eTriangleFan: return "TriangleFan";
      case PrimitiveTopology::eLineListWithAdjacency: return "LineListWithAdjacency";
      case PrimitiveTopology::eLineStripWithAdjacency: return "LineStripWithAdjacency";
      case PrimitiveTopology::eTriangleListWithAdjacency: return "TriangleListWithAdjacency";
      case PrimitiveTopology::eTriangleStripWithAdjacency: return "TriangleStripWithAdjacency";
      case PrimitiveTopology::ePatchList: return "PatchList";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ShaderStageFlagBits value )
  {
    switch ( value )
    {
      case ShaderStageFlagBits::eVertex: return "Vertex";
      case ShaderStageFlagBits::eTessellationControl: return "TessellationControl";
      case ShaderStageFlagBits::eTessellationEvaluation: return "TessellationEvaluation";
      case ShaderStageFlagBits::eGeometry: return "Geometry";
      case ShaderStageFlagBits::eFragment: return "Fragment";
      case ShaderStageFlagBits::eCompute: return "Compute";
      case ShaderStageFlagBits::eAllGraphics: return "AllGraphics";
      case ShaderStageFlagBits::eAll: return "All";
      case ShaderStageFlagBits::eRaygenKHR: return "RaygenKHR";
      case ShaderStageFlagBits::eAnyHitKHR: return "AnyHitKHR";
      case ShaderStageFlagBits::eClosestHitKHR: return "ClosestHitKHR";
      case ShaderStageFlagBits::eMissKHR: return "MissKHR";
      case ShaderStageFlagBits::eIntersectionKHR: return "IntersectionKHR";
      case ShaderStageFlagBits::eCallableKHR: return "CallableKHR";
      case ShaderStageFlagBits::eTaskEXT: return "TaskEXT";
      case ShaderStageFlagBits::eMeshEXT: return "MeshEXT";
      case ShaderStageFlagBits::eSubpassShadingHUAWEI: return "SubpassShadingHUAWEI";
      case ShaderStageFlagBits::eClusterCullingHUAWEI: return "ClusterCullingHUAWEI";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( StencilOp value )
  {
    switch ( value )
    {
      case StencilOp::eKeep: return "Keep";
      case StencilOp::eZero: return "Zero";
      case StencilOp::eReplace: return "Replace";
      case StencilOp::eIncrementAndClamp: return "IncrementAndClamp";
      case StencilOp::eDecrementAndClamp: return "DecrementAndClamp";
      case StencilOp::eInvert: return "Invert";
      case StencilOp::eIncrementAndWrap: return "IncrementAndWrap";
      case StencilOp::eDecrementAndWrap: return "DecrementAndWrap";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( VertexInputRate value )
  {
    switch ( value )
    {
      case VertexInputRate::eVertex: return "Vertex";
      case VertexInputRate::eInstance: return "Instance";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineColorBlendStateCreateFlagBits value )
  {
    switch ( value )
    {
      case PipelineColorBlendStateCreateFlagBits::eRasterizationOrderAttachmentAccessEXT: return "RasterizationOrderAttachmentAccessEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineDepthStencilStateCreateFlagBits value )
  {
    switch ( value )
    {
      case PipelineDepthStencilStateCreateFlagBits::eRasterizationOrderAttachmentDepthAccessEXT: return "RasterizationOrderAttachmentDepthAccessEXT";
      case PipelineDepthStencilStateCreateFlagBits::eRasterizationOrderAttachmentStencilAccessEXT: return "RasterizationOrderAttachmentStencilAccessEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineDynamicStateCreateFlagBits )
  {
    return "(void)";
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineInputAssemblyStateCreateFlagBits )
  {
    return "(void)";
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineLayoutCreateFlagBits value )
  {
    switch ( value )
    {
      case PipelineLayoutCreateFlagBits::eIndependentSetsEXT: return "IndependentSetsEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineMultisampleStateCreateFlagBits )
  {
    return "(void)";
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineRasterizationStateCreateFlagBits )
  {
    return "(void)";
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineTessellationStateCreateFlagBits )
  {
    return "(void)";
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineVertexInputStateCreateFlagBits )
  {
    return "(void)";
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineViewportStateCreateFlagBits )
  {
    return "(void)";
  }

  VULKAN_HPP_INLINE std::string to_string( BorderColor value )
  {
    switch ( value )
    {
      case BorderColor::eFloatTransparentBlack: return "FloatTransparentBlack";
      case BorderColor::eIntTransparentBlack: return "IntTransparentBlack";
      case BorderColor::eFloatOpaqueBlack: return "FloatOpaqueBlack";
      case BorderColor::eIntOpaqueBlack: return "IntOpaqueBlack";
      case BorderColor::eFloatOpaqueWhite: return "FloatOpaqueWhite";
      case BorderColor::eIntOpaqueWhite: return "IntOpaqueWhite";
      case BorderColor::eFloatCustomEXT: return "FloatCustomEXT";
      case BorderColor::eIntCustomEXT: return "IntCustomEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( Filter value )
  {
    switch ( value )
    {
      case Filter::eNearest: return "Nearest";
      case Filter::eLinear: return "Linear";
      case Filter::eCubicEXT: return "CubicEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( SamplerAddressMode value )
  {
    switch ( value )
    {
      case SamplerAddressMode::eRepeat: return "Repeat";
      case SamplerAddressMode::eMirroredRepeat: return "MirroredRepeat";
      case SamplerAddressMode::eClampToEdge: return "ClampToEdge";
      case SamplerAddressMode::eClampToBorder: return "ClampToBorder";
      case SamplerAddressMode::eMirrorClampToEdge: return "MirrorClampToEdge";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( SamplerCreateFlagBits value )
  {
    switch ( value )
    {
      case SamplerCreateFlagBits::eSubsampledEXT: return "SubsampledEXT";
      case SamplerCreateFlagBits::eSubsampledCoarseReconstructionEXT: return "SubsampledCoarseReconstructionEXT";
      case SamplerCreateFlagBits::eDescriptorBufferCaptureReplayEXT: return "DescriptorBufferCaptureReplayEXT";
      case SamplerCreateFlagBits::eNonSeamlessCubeMapEXT: return "NonSeamlessCubeMapEXT";
      case SamplerCreateFlagBits::eImageProcessingQCOM: return "ImageProcessingQCOM";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( SamplerMipmapMode value )
  {
    switch ( value )
    {
      case SamplerMipmapMode::eNearest: return "Nearest";
      case SamplerMipmapMode::eLinear: return "Linear";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( DescriptorPoolCreateFlagBits value )
  {
    switch ( value )
    {
      case DescriptorPoolCreateFlagBits::eFreeDescriptorSet: return "FreeDescriptorSet";
      case DescriptorPoolCreateFlagBits::eUpdateAfterBind: return "UpdateAfterBind";
      case DescriptorPoolCreateFlagBits::eHostOnlyEXT: return "HostOnlyEXT";
      case DescriptorPoolCreateFlagBits::eAllowOverallocationSetsNV: return "AllowOverallocationSetsNV";
      case DescriptorPoolCreateFlagBits::eAllowOverallocationPoolsNV: return "AllowOverallocationPoolsNV";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( DescriptorSetLayoutCreateFlagBits value )
  {
    switch ( value )
    {
      case DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool: return "UpdateAfterBindPool";
      case DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR: return "PushDescriptorKHR";
      case DescriptorSetLayoutCreateFlagBits::eDescriptorBufferEXT: return "DescriptorBufferEXT";
      case DescriptorSetLayoutCreateFlagBits::eEmbeddedImmutableSamplersEXT: return "EmbeddedImmutableSamplersEXT";
      case DescriptorSetLayoutCreateFlagBits::eIndirectBindableNV: return "IndirectBindableNV";
      case DescriptorSetLayoutCreateFlagBits::eHostOnlyPoolEXT: return "HostOnlyPoolEXT";
      case DescriptorSetLayoutCreateFlagBits::ePerStageNV: return "PerStageNV";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( DescriptorType value )
  {
    switch ( value )
    {
      case DescriptorType::eSampler: return "Sampler";
      case DescriptorType::eCombinedImageSampler: return "CombinedImageSampler";
      case DescriptorType::eSampledImage: return "SampledImage";
      case DescriptorType::eStorageImage: return "StorageImage";
      case DescriptorType::eUniformTexelBuffer: return "UniformTexelBuffer";
      case DescriptorType::eStorageTexelBuffer: return "StorageTexelBuffer";
      case DescriptorType::eUniformBuffer: return "UniformBuffer";
      case DescriptorType::eStorageBuffer: return "StorageBuffer";
      case DescriptorType::eUniformBufferDynamic: return "UniformBufferDynamic";
      case DescriptorType::eStorageBufferDynamic: return "StorageBufferDynamic";
      case DescriptorType::eInputAttachment: return "InputAttachment";
      case DescriptorType::eInlineUniformBlock: return "InlineUniformBlock";
      case DescriptorType::eAccelerationStructureKHR: return "AccelerationStructureKHR";
      case DescriptorType::eAccelerationStructureNV: return "AccelerationStructureNV";
      case DescriptorType::eSampleWeightImageQCOM: return "SampleWeightImageQCOM";
      case DescriptorType::eBlockMatchImageQCOM: return "BlockMatchImageQCOM";
      case DescriptorType::eMutableEXT: return "MutableEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( DescriptorPoolResetFlagBits )
  {
    return "(void)";
  }

  VULKAN_HPP_INLINE std::string to_string( AccessFlagBits value )
  {
    switch ( value )
    {
      case AccessFlagBits::eIndirectCommandRead: return "IndirectCommandRead";
      case AccessFlagBits::eIndexRead: return "IndexRead";
      case AccessFlagBits::eVertexAttributeRead: return "VertexAttributeRead";
      case AccessFlagBits::eUniformRead: return "UniformRead";
      case AccessFlagBits::eInputAttachmentRead: return "InputAttachmentRead";
      case AccessFlagBits::eShaderRead: return "ShaderRead";
      case AccessFlagBits::eShaderWrite: return "ShaderWrite";
      case AccessFlagBits::eColorAttachmentRead: return "ColorAttachmentRead";
      case AccessFlagBits::eColorAttachmentWrite: return "ColorAttachmentWrite";
      case AccessFlagBits::eDepthStencilAttachmentRead: return "DepthStencilAttachmentRead";
      case AccessFlagBits::eDepthStencilAttachmentWrite: return "DepthStencilAttachmentWrite";
      case AccessFlagBits::eTransferRead: return "TransferRead";
      case AccessFlagBits::eTransferWrite: return "TransferWrite";
      case AccessFlagBits::eHostRead: return "HostRead";
      case AccessFlagBits::eHostWrite: return "HostWrite";
      case AccessFlagBits::eMemoryRead: return "MemoryRead";
      case AccessFlagBits::eMemoryWrite: return "MemoryWrite";
      case AccessFlagBits::eNone: return "None";
      case AccessFlagBits::eTransformFeedbackWriteEXT: return "TransformFeedbackWriteEXT";
      case AccessFlagBits::eTransformFeedbackCounterReadEXT: return "TransformFeedbackCounterReadEXT";
      case AccessFlagBits::eTransformFeedbackCounterWriteEXT: return "TransformFeedbackCounterWriteEXT";
      case AccessFlagBits::eConditionalRenderingReadEXT: return "ConditionalRenderingReadEXT";
      case AccessFlagBits::eColorAttachmentReadNoncoherentEXT: return "ColorAttachmentReadNoncoherentEXT";
      case AccessFlagBits::eAccelerationStructureReadKHR: return "AccelerationStructureReadKHR";
      case AccessFlagBits::eAccelerationStructureWriteKHR: return "AccelerationStructureWriteKHR";
      case AccessFlagBits::eFragmentDensityMapReadEXT: return "FragmentDensityMapReadEXT";
      case AccessFlagBits::eFragmentShadingRateAttachmentReadKHR: return "FragmentShadingRateAttachmentReadKHR";
      case AccessFlagBits::eCommandPreprocessReadNV: return "CommandPreprocessReadNV";
      case AccessFlagBits::eCommandPreprocessWriteNV: return "CommandPreprocessWriteNV";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( AttachmentDescriptionFlagBits value )
  {
    switch ( value )
    {
      case AttachmentDescriptionFlagBits::eMayAlias: return "MayAlias";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( AttachmentLoadOp value )
  {
    switch ( value )
    {
      case AttachmentLoadOp::eLoad: return "Load";
      case AttachmentLoadOp::eClear: return "Clear";
      case AttachmentLoadOp::eDontCare: return "DontCare";
      case AttachmentLoadOp::eNoneKHR: return "NoneKHR";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( AttachmentStoreOp value )
  {
    switch ( value )
    {
      case AttachmentStoreOp::eStore: return "Store";
      case AttachmentStoreOp::eDontCare: return "DontCare";
      case AttachmentStoreOp::eNone: return "None";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( DependencyFlagBits value )
  {
    switch ( value )
    {
      case DependencyFlagBits::eByRegion: return "ByRegion";
      case DependencyFlagBits::eDeviceGroup: return "DeviceGroup";
      case DependencyFlagBits::eViewLocal: return "ViewLocal";
      case DependencyFlagBits::eFeedbackLoopEXT: return "FeedbackLoopEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( FramebufferCreateFlagBits value )
  {
    switch ( value )
    {
      case FramebufferCreateFlagBits::eImageless: return "Imageless";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineBindPoint value )
  {
    switch ( value )
    {
      case PipelineBindPoint::eGraphics: return "Graphics";
      case PipelineBindPoint::eCompute: return "Compute";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case PipelineBindPoint::eExecutionGraphAMDX: return "ExecutionGraphAMDX";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      case PipelineBindPoint::eRayTracingKHR: return "RayTracingKHR";
      case PipelineBindPoint::eSubpassShadingHUAWEI: return "SubpassShadingHUAWEI";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( RenderPassCreateFlagBits value )
  {
    switch ( value )
    {
      case RenderPassCreateFlagBits::eTransformQCOM: return "TransformQCOM";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( SubpassDescriptionFlagBits value )
  {
    switch ( value )
    {
      case SubpassDescriptionFlagBits::ePerViewAttributesNVX: return "PerViewAttributesNVX";
      case SubpassDescriptionFlagBits::ePerViewPositionXOnlyNVX: return "PerViewPositionXOnlyNVX";
      case SubpassDescriptionFlagBits::eFragmentRegionQCOM: return "FragmentRegionQCOM";
      case SubpassDescriptionFlagBits::eShaderResolveQCOM: return "ShaderResolveQCOM";
      case SubpassDescriptionFlagBits::eRasterizationOrderAttachmentColorAccessEXT: return "RasterizationOrderAttachmentColorAccessEXT";
      case SubpassDescriptionFlagBits::eRasterizationOrderAttachmentDepthAccessEXT: return "RasterizationOrderAttachmentDepthAccessEXT";
      case SubpassDescriptionFlagBits::eRasterizationOrderAttachmentStencilAccessEXT: return "RasterizationOrderAttachmentStencilAccessEXT";
      case SubpassDescriptionFlagBits::eEnableLegacyDitheringEXT: return "EnableLegacyDitheringEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( CommandPoolCreateFlagBits value )
  {
    switch ( value )
    {
      case CommandPoolCreateFlagBits::eTransient: return "Transient";
      case CommandPoolCreateFlagBits::eResetCommandBuffer: return "ResetCommandBuffer";
      case CommandPoolCreateFlagBits::eProtected: return "Protected";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( CommandPoolResetFlagBits value )
  {
    switch ( value )
    {
      case CommandPoolResetFlagBits::eReleaseResources: return "ReleaseResources";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( CommandBufferLevel value )
  {
    switch ( value )
    {
      case CommandBufferLevel::ePrimary: return "Primary";
      case CommandBufferLevel::eSecondary: return "Secondary";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( CommandBufferResetFlagBits value )
  {
    switch ( value )
    {
      case CommandBufferResetFlagBits::eReleaseResources: return "ReleaseResources";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( CommandBufferUsageFlagBits value )
  {
    switch ( value )
    {
      case CommandBufferUsageFlagBits::eOneTimeSubmit: return "OneTimeSubmit";
      case CommandBufferUsageFlagBits::eRenderPassContinue: return "RenderPassContinue";
      case CommandBufferUsageFlagBits::eSimultaneousUse: return "SimultaneousUse";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( QueryControlFlagBits value )
  {
    switch ( value )
    {
      case QueryControlFlagBits::ePrecise: return "Precise";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( IndexType value )
  {
    switch ( value )
    {
      case IndexType::eUint16: return "Uint16";
      case IndexType::eUint32: return "Uint32";
      case IndexType::eNoneKHR: return "NoneKHR";
      case IndexType::eUint8KHR: return "Uint8KHR";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( StencilFaceFlagBits value )
  {
    switch ( value )
    {
      case StencilFaceFlagBits::eFront: return "Front";
      case StencilFaceFlagBits::eBack: return "Back";
      case StencilFaceFlagBits::eFrontAndBack: return "FrontAndBack";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( SubpassContents value )
  {
    switch ( value )
    {
      case SubpassContents::eInline: return "Inline";
      case SubpassContents::eSecondaryCommandBuffers: return "SecondaryCommandBuffers";
      case SubpassContents::eInlineAndSecondaryCommandBuffersEXT: return "InlineAndSecondaryCommandBuffersEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_VERSION_1_1 ===

  VULKAN_HPP_INLINE std::string to_string( SubgroupFeatureFlagBits value )
  {
    switch ( value )
    {
      case SubgroupFeatureFlagBits::eBasic: return "Basic";
      case SubgroupFeatureFlagBits::eVote: return "Vote";
      case SubgroupFeatureFlagBits::eArithmetic: return "Arithmetic";
      case SubgroupFeatureFlagBits::eBallot: return "Ballot";
      case SubgroupFeatureFlagBits::eShuffle: return "Shuffle";
      case SubgroupFeatureFlagBits::eShuffleRelative: return "ShuffleRelative";
      case SubgroupFeatureFlagBits::eClustered: return "Clustered";
      case SubgroupFeatureFlagBits::eQuad: return "Quad";
      case SubgroupFeatureFlagBits::ePartitionedNV: return "PartitionedNV";
      case SubgroupFeatureFlagBits::eRotateKHR: return "RotateKHR";
      case SubgroupFeatureFlagBits::eRotateClusteredKHR: return "RotateClusteredKHR";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PeerMemoryFeatureFlagBits value )
  {
    switch ( value )
    {
      case PeerMemoryFeatureFlagBits::eCopySrc: return "CopySrc";
      case PeerMemoryFeatureFlagBits::eCopyDst: return "CopyDst";
      case PeerMemoryFeatureFlagBits::eGenericSrc: return "GenericSrc";
      case PeerMemoryFeatureFlagBits::eGenericDst: return "GenericDst";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( MemoryAllocateFlagBits value )
  {
    switch ( value )
    {
      case MemoryAllocateFlagBits::eDeviceMask: return "DeviceMask";
      case MemoryAllocateFlagBits::eDeviceAddress: return "DeviceAddress";
      case MemoryAllocateFlagBits::eDeviceAddressCaptureReplay: return "DeviceAddressCaptureReplay";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( CommandPoolTrimFlagBits )
  {
    return "(void)";
  }

  VULKAN_HPP_INLINE std::string to_string( PointClippingBehavior value )
  {
    switch ( value )
    {
      case PointClippingBehavior::eAllClipPlanes: return "AllClipPlanes";
      case PointClippingBehavior::eUserClipPlanesOnly: return "UserClipPlanesOnly";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( TessellationDomainOrigin value )
  {
    switch ( value )
    {
      case TessellationDomainOrigin::eUpperLeft: return "UpperLeft";
      case TessellationDomainOrigin::eLowerLeft: return "LowerLeft";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( SamplerYcbcrModelConversion value )
  {
    switch ( value )
    {
      case SamplerYcbcrModelConversion::eRgbIdentity: return "RgbIdentity";
      case SamplerYcbcrModelConversion::eYcbcrIdentity: return "YcbcrIdentity";
      case SamplerYcbcrModelConversion::eYcbcr709: return "Ycbcr709";
      case SamplerYcbcrModelConversion::eYcbcr601: return "Ycbcr601";
      case SamplerYcbcrModelConversion::eYcbcr2020: return "Ycbcr2020";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( SamplerYcbcrRange value )
  {
    switch ( value )
    {
      case SamplerYcbcrRange::eItuFull: return "ItuFull";
      case SamplerYcbcrRange::eItuNarrow: return "ItuNarrow";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ChromaLocation value )
  {
    switch ( value )
    {
      case ChromaLocation::eCositedEven: return "CositedEven";
      case ChromaLocation::eMidpoint: return "Midpoint";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( DescriptorUpdateTemplateType value )
  {
    switch ( value )
    {
      case DescriptorUpdateTemplateType::eDescriptorSet: return "DescriptorSet";
      case DescriptorUpdateTemplateType::ePushDescriptorsKHR: return "PushDescriptorsKHR";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( DescriptorUpdateTemplateCreateFlagBits )
  {
    return "(void)";
  }

  VULKAN_HPP_INLINE std::string to_string( ExternalMemoryHandleTypeFlagBits value )
  {
    switch ( value )
    {
      case ExternalMemoryHandleTypeFlagBits::eOpaqueFd: return "OpaqueFd";
      case ExternalMemoryHandleTypeFlagBits::eOpaqueWin32: return "OpaqueWin32";
      case ExternalMemoryHandleTypeFlagBits::eOpaqueWin32Kmt: return "OpaqueWin32Kmt";
      case ExternalMemoryHandleTypeFlagBits::eD3D11Texture: return "D3D11Texture";
      case ExternalMemoryHandleTypeFlagBits::eD3D11TextureKmt: return "D3D11TextureKmt";
      case ExternalMemoryHandleTypeFlagBits::eD3D12Heap: return "D3D12Heap";
      case ExternalMemoryHandleTypeFlagBits::eD3D12Resource: return "D3D12Resource";
      case ExternalMemoryHandleTypeFlagBits::eDmaBufEXT: return "DmaBufEXT";
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
      case ExternalMemoryHandleTypeFlagBits::eAndroidHardwareBufferANDROID: return "AndroidHardwareBufferANDROID";
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
      case ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT: return "HostAllocationEXT";
      case ExternalMemoryHandleTypeFlagBits::eHostMappedForeignMemoryEXT: return "HostMappedForeignMemoryEXT";
#if defined( VK_USE_PLATFORM_FUCHSIA )
      case ExternalMemoryHandleTypeFlagBits::eZirconVmoFUCHSIA: return "ZirconVmoFUCHSIA";
#endif /*VK_USE_PLATFORM_FUCHSIA*/
      case ExternalMemoryHandleTypeFlagBits::eRdmaAddressNV: return "RdmaAddressNV";
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
      case ExternalMemoryHandleTypeFlagBits::eScreenBufferQNX: return "ScreenBufferQNX";
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ExternalMemoryFeatureFlagBits value )
  {
    switch ( value )
    {
      case ExternalMemoryFeatureFlagBits::eDedicatedOnly: return "DedicatedOnly";
      case ExternalMemoryFeatureFlagBits::eExportable: return "Exportable";
      case ExternalMemoryFeatureFlagBits::eImportable: return "Importable";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ExternalFenceHandleTypeFlagBits value )
  {
    switch ( value )
    {
      case ExternalFenceHandleTypeFlagBits::eOpaqueFd: return "OpaqueFd";
      case ExternalFenceHandleTypeFlagBits::eOpaqueWin32: return "OpaqueWin32";
      case ExternalFenceHandleTypeFlagBits::eOpaqueWin32Kmt: return "OpaqueWin32Kmt";
      case ExternalFenceHandleTypeFlagBits::eSyncFd: return "SyncFd";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ExternalFenceFeatureFlagBits value )
  {
    switch ( value )
    {
      case ExternalFenceFeatureFlagBits::eExportable: return "Exportable";
      case ExternalFenceFeatureFlagBits::eImportable: return "Importable";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( FenceImportFlagBits value )
  {
    switch ( value )
    {
      case FenceImportFlagBits::eTemporary: return "Temporary";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( SemaphoreImportFlagBits value )
  {
    switch ( value )
    {
      case SemaphoreImportFlagBits::eTemporary: return "Temporary";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ExternalSemaphoreHandleTypeFlagBits value )
  {
    switch ( value )
    {
      case ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd: return "OpaqueFd";
      case ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32: return "OpaqueWin32";
      case ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32Kmt: return "OpaqueWin32Kmt";
      case ExternalSemaphoreHandleTypeFlagBits::eD3D12Fence: return "D3D12Fence";
      case ExternalSemaphoreHandleTypeFlagBits::eSyncFd: return "SyncFd";
#if defined( VK_USE_PLATFORM_FUCHSIA )
      case ExternalSemaphoreHandleTypeFlagBits::eZirconEventFUCHSIA: return "ZirconEventFUCHSIA";
#endif /*VK_USE_PLATFORM_FUCHSIA*/
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ExternalSemaphoreFeatureFlagBits value )
  {
    switch ( value )
    {
      case ExternalSemaphoreFeatureFlagBits::eExportable: return "Exportable";
      case ExternalSemaphoreFeatureFlagBits::eImportable: return "Importable";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_VERSION_1_2 ===

  VULKAN_HPP_INLINE std::string to_string( DriverId value )
  {
    switch ( value )
    {
      case DriverId::eAmdProprietary: return "AmdProprietary";
      case DriverId::eAmdOpenSource: return "AmdOpenSource";
      case DriverId::eMesaRadv: return "MesaRadv";
      case DriverId::eNvidiaProprietary: return "NvidiaProprietary";
      case DriverId::eIntelProprietaryWindows: return "IntelProprietaryWindows";
      case DriverId::eIntelOpenSourceMESA: return "IntelOpenSourceMESA";
      case DriverId::eImaginationProprietary: return "ImaginationProprietary";
      case DriverId::eQualcommProprietary: return "QualcommProprietary";
      case DriverId::eArmProprietary: return "ArmProprietary";
      case DriverId::eGoogleSwiftshader: return "GoogleSwiftshader";
      case DriverId::eGgpProprietary: return "GgpProprietary";
      case DriverId::eBroadcomProprietary: return "BroadcomProprietary";
      case DriverId::eMesaLlvmpipe: return "MesaLlvmpipe";
      case DriverId::eMoltenvk: return "Moltenvk";
      case DriverId::eCoreaviProprietary: return "CoreaviProprietary";
      case DriverId::eJuiceProprietary: return "JuiceProprietary";
      case DriverId::eVerisiliconProprietary: return "VerisiliconProprietary";
      case DriverId::eMesaTurnip: return "MesaTurnip";
      case DriverId::eMesaV3Dv: return "MesaV3Dv";
      case DriverId::eMesaPanvk: return "MesaPanvk";
      case DriverId::eSamsungProprietary: return "SamsungProprietary";
      case DriverId::eMesaVenus: return "MesaVenus";
      case DriverId::eMesaDozen: return "MesaDozen";
      case DriverId::eMesaNvk: return "MesaNvk";
      case DriverId::eImaginationOpenSourceMESA: return "ImaginationOpenSourceMESA";
      case DriverId::eMesaAgxv: return "MesaAgxv";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ShaderFloatControlsIndependence value )
  {
    switch ( value )
    {
      case ShaderFloatControlsIndependence::e32BitOnly: return "32BitOnly";
      case ShaderFloatControlsIndependence::eAll: return "All";
      case ShaderFloatControlsIndependence::eNone: return "None";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( DescriptorBindingFlagBits value )
  {
    switch ( value )
    {
      case DescriptorBindingFlagBits::eUpdateAfterBind: return "UpdateAfterBind";
      case DescriptorBindingFlagBits::eUpdateUnusedWhilePending: return "UpdateUnusedWhilePending";
      case DescriptorBindingFlagBits::ePartiallyBound: return "PartiallyBound";
      case DescriptorBindingFlagBits::eVariableDescriptorCount: return "VariableDescriptorCount";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ResolveModeFlagBits value )
  {
    switch ( value )
    {
      case ResolveModeFlagBits::eNone: return "None";
      case ResolveModeFlagBits::eSampleZero: return "SampleZero";
      case ResolveModeFlagBits::eAverage: return "Average";
      case ResolveModeFlagBits::eMin: return "Min";
      case ResolveModeFlagBits::eMax: return "Max";
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
      case ResolveModeFlagBits::eExternalFormatDownsampleANDROID: return "ExternalFormatDownsampleANDROID";
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( SamplerReductionMode value )
  {
    switch ( value )
    {
      case SamplerReductionMode::eWeightedAverage: return "WeightedAverage";
      case SamplerReductionMode::eMin: return "Min";
      case SamplerReductionMode::eMax: return "Max";
      case SamplerReductionMode::eWeightedAverageRangeclampQCOM: return "WeightedAverageRangeclampQCOM";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( SemaphoreType value )
  {
    switch ( value )
    {
      case SemaphoreType::eBinary: return "Binary";
      case SemaphoreType::eTimeline: return "Timeline";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( SemaphoreWaitFlagBits value )
  {
    switch ( value )
    {
      case SemaphoreWaitFlagBits::eAny: return "Any";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_VERSION_1_3 ===

  VULKAN_HPP_INLINE std::string to_string( PipelineCreationFeedbackFlagBits value )
  {
    switch ( value )
    {
      case PipelineCreationFeedbackFlagBits::eValid: return "Valid";
      case PipelineCreationFeedbackFlagBits::eApplicationPipelineCacheHit: return "ApplicationPipelineCacheHit";
      case PipelineCreationFeedbackFlagBits::eBasePipelineAcceleration: return "BasePipelineAcceleration";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ToolPurposeFlagBits value )
  {
    switch ( value )
    {
      case ToolPurposeFlagBits::eValidation: return "Validation";
      case ToolPurposeFlagBits::eProfiling: return "Profiling";
      case ToolPurposeFlagBits::eTracing: return "Tracing";
      case ToolPurposeFlagBits::eAdditionalFeatures: return "AdditionalFeatures";
      case ToolPurposeFlagBits::eModifyingFeatures: return "ModifyingFeatures";
      case ToolPurposeFlagBits::eDebugReportingEXT: return "DebugReportingEXT";
      case ToolPurposeFlagBits::eDebugMarkersEXT: return "DebugMarkersEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PrivateDataSlotCreateFlagBits )
  {
    return "(void)";
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineStageFlagBits2 value )
  {
    switch ( value )
    {
      case PipelineStageFlagBits2::eNone: return "None";
      case PipelineStageFlagBits2::eTopOfPipe: return "TopOfPipe";
      case PipelineStageFlagBits2::eDrawIndirect: return "DrawIndirect";
      case PipelineStageFlagBits2::eVertexInput: return "VertexInput";
      case PipelineStageFlagBits2::eVertexShader: return "VertexShader";
      case PipelineStageFlagBits2::eTessellationControlShader: return "TessellationControlShader";
      case PipelineStageFlagBits2::eTessellationEvaluationShader: return "TessellationEvaluationShader";
      case PipelineStageFlagBits2::eGeometryShader: return "GeometryShader";
      case PipelineStageFlagBits2::eFragmentShader: return "FragmentShader";
      case PipelineStageFlagBits2::eEarlyFragmentTests: return "EarlyFragmentTests";
      case PipelineStageFlagBits2::eLateFragmentTests: return "LateFragmentTests";
      case PipelineStageFlagBits2::eColorAttachmentOutput: return "ColorAttachmentOutput";
      case PipelineStageFlagBits2::eComputeShader: return "ComputeShader";
      case PipelineStageFlagBits2::eAllTransfer: return "AllTransfer";
      case PipelineStageFlagBits2::eBottomOfPipe: return "BottomOfPipe";
      case PipelineStageFlagBits2::eHost: return "Host";
      case PipelineStageFlagBits2::eAllGraphics: return "AllGraphics";
      case PipelineStageFlagBits2::eAllCommands: return "AllCommands";
      case PipelineStageFlagBits2::eCopy: return "Copy";
      case PipelineStageFlagBits2::eResolve: return "Resolve";
      case PipelineStageFlagBits2::eBlit: return "Blit";
      case PipelineStageFlagBits2::eClear: return "Clear";
      case PipelineStageFlagBits2::eIndexInput: return "IndexInput";
      case PipelineStageFlagBits2::eVertexAttributeInput: return "VertexAttributeInput";
      case PipelineStageFlagBits2::ePreRasterizationShaders: return "PreRasterizationShaders";
      case PipelineStageFlagBits2::eVideoDecodeKHR: return "VideoDecodeKHR";
      case PipelineStageFlagBits2::eVideoEncodeKHR: return "VideoEncodeKHR";
      case PipelineStageFlagBits2::eTransformFeedbackEXT: return "TransformFeedbackEXT";
      case PipelineStageFlagBits2::eConditionalRenderingEXT: return "ConditionalRenderingEXT";
      case PipelineStageFlagBits2::eCommandPreprocessNV: return "CommandPreprocessNV";
      case PipelineStageFlagBits2::eFragmentShadingRateAttachmentKHR: return "FragmentShadingRateAttachmentKHR";
      case PipelineStageFlagBits2::eAccelerationStructureBuildKHR: return "AccelerationStructureBuildKHR";
      case PipelineStageFlagBits2::eRayTracingShaderKHR: return "RayTracingShaderKHR";
      case PipelineStageFlagBits2::eFragmentDensityProcessEXT: return "FragmentDensityProcessEXT";
      case PipelineStageFlagBits2::eTaskShaderEXT: return "TaskShaderEXT";
      case PipelineStageFlagBits2::eMeshShaderEXT: return "MeshShaderEXT";
      case PipelineStageFlagBits2::eSubpassShaderHUAWEI: return "SubpassShaderHUAWEI";
      case PipelineStageFlagBits2::eInvocationMaskHUAWEI: return "InvocationMaskHUAWEI";
      case PipelineStageFlagBits2::eAccelerationStructureCopyKHR: return "AccelerationStructureCopyKHR";
      case PipelineStageFlagBits2::eMicromapBuildEXT: return "MicromapBuildEXT";
      case PipelineStageFlagBits2::eClusterCullingShaderHUAWEI: return "ClusterCullingShaderHUAWEI";
      case PipelineStageFlagBits2::eOpticalFlowNV: return "OpticalFlowNV";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( AccessFlagBits2 value )
  {
    switch ( value )
    {
      case AccessFlagBits2::eNone: return "None";
      case AccessFlagBits2::eIndirectCommandRead: return "IndirectCommandRead";
      case AccessFlagBits2::eIndexRead: return "IndexRead";
      case AccessFlagBits2::eVertexAttributeRead: return "VertexAttributeRead";
      case AccessFlagBits2::eUniformRead: return "UniformRead";
      case AccessFlagBits2::eInputAttachmentRead: return "InputAttachmentRead";
      case AccessFlagBits2::eShaderRead: return "ShaderRead";
      case AccessFlagBits2::eShaderWrite: return "ShaderWrite";
      case AccessFlagBits2::eColorAttachmentRead: return "ColorAttachmentRead";
      case AccessFlagBits2::eColorAttachmentWrite: return "ColorAttachmentWrite";
      case AccessFlagBits2::eDepthStencilAttachmentRead: return "DepthStencilAttachmentRead";
      case AccessFlagBits2::eDepthStencilAttachmentWrite: return "DepthStencilAttachmentWrite";
      case AccessFlagBits2::eTransferRead: return "TransferRead";
      case AccessFlagBits2::eTransferWrite: return "TransferWrite";
      case AccessFlagBits2::eHostRead: return "HostRead";
      case AccessFlagBits2::eHostWrite: return "HostWrite";
      case AccessFlagBits2::eMemoryRead: return "MemoryRead";
      case AccessFlagBits2::eMemoryWrite: return "MemoryWrite";
      case AccessFlagBits2::eShaderSampledRead: return "ShaderSampledRead";
      case AccessFlagBits2::eShaderStorageRead: return "ShaderStorageRead";
      case AccessFlagBits2::eShaderStorageWrite: return "ShaderStorageWrite";
      case AccessFlagBits2::eVideoDecodeReadKHR: return "VideoDecodeReadKHR";
      case AccessFlagBits2::eVideoDecodeWriteKHR: return "VideoDecodeWriteKHR";
      case AccessFlagBits2::eVideoEncodeReadKHR: return "VideoEncodeReadKHR";
      case AccessFlagBits2::eVideoEncodeWriteKHR: return "VideoEncodeWriteKHR";
      case AccessFlagBits2::eTransformFeedbackWriteEXT: return "TransformFeedbackWriteEXT";
      case AccessFlagBits2::eTransformFeedbackCounterReadEXT: return "TransformFeedbackCounterReadEXT";
      case AccessFlagBits2::eTransformFeedbackCounterWriteEXT: return "TransformFeedbackCounterWriteEXT";
      case AccessFlagBits2::eConditionalRenderingReadEXT: return "ConditionalRenderingReadEXT";
      case AccessFlagBits2::eCommandPreprocessReadNV: return "CommandPreprocessReadNV";
      case AccessFlagBits2::eCommandPreprocessWriteNV: return "CommandPreprocessWriteNV";
      case AccessFlagBits2::eFragmentShadingRateAttachmentReadKHR: return "FragmentShadingRateAttachmentReadKHR";
      case AccessFlagBits2::eAccelerationStructureReadKHR: return "AccelerationStructureReadKHR";
      case AccessFlagBits2::eAccelerationStructureWriteKHR: return "AccelerationStructureWriteKHR";
      case AccessFlagBits2::eFragmentDensityMapReadEXT: return "FragmentDensityMapReadEXT";
      case AccessFlagBits2::eColorAttachmentReadNoncoherentEXT: return "ColorAttachmentReadNoncoherentEXT";
      case AccessFlagBits2::eDescriptorBufferReadEXT: return "DescriptorBufferReadEXT";
      case AccessFlagBits2::eInvocationMaskReadHUAWEI: return "InvocationMaskReadHUAWEI";
      case AccessFlagBits2::eShaderBindingTableReadKHR: return "ShaderBindingTableReadKHR";
      case AccessFlagBits2::eMicromapReadEXT: return "MicromapReadEXT";
      case AccessFlagBits2::eMicromapWriteEXT: return "MicromapWriteEXT";
      case AccessFlagBits2::eOpticalFlowReadNV: return "OpticalFlowReadNV";
      case AccessFlagBits2::eOpticalFlowWriteNV: return "OpticalFlowWriteNV";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( SubmitFlagBits value )
  {
    switch ( value )
    {
      case SubmitFlagBits::eProtected: return "Protected";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( RenderingFlagBits value )
  {
    switch ( value )
    {
      case RenderingFlagBits::eContentsSecondaryCommandBuffers: return "ContentsSecondaryCommandBuffers";
      case RenderingFlagBits::eSuspending: return "Suspending";
      case RenderingFlagBits::eResuming: return "Resuming";
      case RenderingFlagBits::eContentsInlineEXT: return "ContentsInlineEXT";
      case RenderingFlagBits::eEnableLegacyDitheringEXT: return "EnableLegacyDitheringEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( FormatFeatureFlagBits2 value )
  {
    switch ( value )
    {
      case FormatFeatureFlagBits2::eSampledImage: return "SampledImage";
      case FormatFeatureFlagBits2::eStorageImage: return "StorageImage";
      case FormatFeatureFlagBits2::eStorageImageAtomic: return "StorageImageAtomic";
      case FormatFeatureFlagBits2::eUniformTexelBuffer: return "UniformTexelBuffer";
      case FormatFeatureFlagBits2::eStorageTexelBuffer: return "StorageTexelBuffer";
      case FormatFeatureFlagBits2::eStorageTexelBufferAtomic: return "StorageTexelBufferAtomic";
      case FormatFeatureFlagBits2::eVertexBuffer: return "VertexBuffer";
      case FormatFeatureFlagBits2::eColorAttachment: return "ColorAttachment";
      case FormatFeatureFlagBits2::eColorAttachmentBlend: return "ColorAttachmentBlend";
      case FormatFeatureFlagBits2::eDepthStencilAttachment: return "DepthStencilAttachment";
      case FormatFeatureFlagBits2::eBlitSrc: return "BlitSrc";
      case FormatFeatureFlagBits2::eBlitDst: return "BlitDst";
      case FormatFeatureFlagBits2::eSampledImageFilterLinear: return "SampledImageFilterLinear";
      case FormatFeatureFlagBits2::eSampledImageFilterCubic: return "SampledImageFilterCubic";
      case FormatFeatureFlagBits2::eTransferSrc: return "TransferSrc";
      case FormatFeatureFlagBits2::eTransferDst: return "TransferDst";
      case FormatFeatureFlagBits2::eSampledImageFilterMinmax: return "SampledImageFilterMinmax";
      case FormatFeatureFlagBits2::eMidpointChromaSamples: return "MidpointChromaSamples";
      case FormatFeatureFlagBits2::eSampledImageYcbcrConversionLinearFilter: return "SampledImageYcbcrConversionLinearFilter";
      case FormatFeatureFlagBits2::eSampledImageYcbcrConversionSeparateReconstructionFilter: return "SampledImageYcbcrConversionSeparateReconstructionFilter";
      case FormatFeatureFlagBits2::eSampledImageYcbcrConversionChromaReconstructionExplicit: return "SampledImageYcbcrConversionChromaReconstructionExplicit";
      case FormatFeatureFlagBits2::eSampledImageYcbcrConversionChromaReconstructionExplicitForceable:
        return "SampledImageYcbcrConversionChromaReconstructionExplicitForceable";
      case FormatFeatureFlagBits2::eDisjoint: return "Disjoint";
      case FormatFeatureFlagBits2::eCositedChromaSamples: return "CositedChromaSamples";
      case FormatFeatureFlagBits2::eStorageReadWithoutFormat: return "StorageReadWithoutFormat";
      case FormatFeatureFlagBits2::eStorageWriteWithoutFormat: return "StorageWriteWithoutFormat";
      case FormatFeatureFlagBits2::eSampledImageDepthComparison: return "SampledImageDepthComparison";
      case FormatFeatureFlagBits2::eVideoDecodeOutputKHR: return "VideoDecodeOutputKHR";
      case FormatFeatureFlagBits2::eVideoDecodeDpbKHR: return "VideoDecodeDpbKHR";
      case FormatFeatureFlagBits2::eAccelerationStructureVertexBufferKHR: return "AccelerationStructureVertexBufferKHR";
      case FormatFeatureFlagBits2::eFragmentDensityMapEXT: return "FragmentDensityMapEXT";
      case FormatFeatureFlagBits2::eFragmentShadingRateAttachmentKHR: return "FragmentShadingRateAttachmentKHR";
      case FormatFeatureFlagBits2::eHostImageTransferEXT: return "HostImageTransferEXT";
      case FormatFeatureFlagBits2::eVideoEncodeInputKHR: return "VideoEncodeInputKHR";
      case FormatFeatureFlagBits2::eVideoEncodeDpbKHR: return "VideoEncodeDpbKHR";
      case FormatFeatureFlagBits2::eLinearColorAttachmentNV: return "LinearColorAttachmentNV";
      case FormatFeatureFlagBits2::eWeightImageQCOM: return "WeightImageQCOM";
      case FormatFeatureFlagBits2::eWeightSampledImageQCOM: return "WeightSampledImageQCOM";
      case FormatFeatureFlagBits2::eBlockMatchingQCOM: return "BlockMatchingQCOM";
      case FormatFeatureFlagBits2::eBoxFilterSampledQCOM: return "BoxFilterSampledQCOM";
      case FormatFeatureFlagBits2::eOpticalFlowImageNV: return "OpticalFlowImageNV";
      case FormatFeatureFlagBits2::eOpticalFlowVectorNV: return "OpticalFlowVectorNV";
      case FormatFeatureFlagBits2::eOpticalFlowCostNV: return "OpticalFlowCostNV";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_KHR_surface ===

  VULKAN_HPP_INLINE std::string to_string( SurfaceTransformFlagBitsKHR value )
  {
    switch ( value )
    {
      case SurfaceTransformFlagBitsKHR::eIdentity: return "Identity";
      case SurfaceTransformFlagBitsKHR::eRotate90: return "Rotate90";
      case SurfaceTransformFlagBitsKHR::eRotate180: return "Rotate180";
      case SurfaceTransformFlagBitsKHR::eRotate270: return "Rotate270";
      case SurfaceTransformFlagBitsKHR::eHorizontalMirror: return "HorizontalMirror";
      case SurfaceTransformFlagBitsKHR::eHorizontalMirrorRotate90: return "HorizontalMirrorRotate90";
      case SurfaceTransformFlagBitsKHR::eHorizontalMirrorRotate180: return "HorizontalMirrorRotate180";
      case SurfaceTransformFlagBitsKHR::eHorizontalMirrorRotate270: return "HorizontalMirrorRotate270";
      case SurfaceTransformFlagBitsKHR::eInherit: return "Inherit";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PresentModeKHR value )
  {
    switch ( value )
    {
      case PresentModeKHR::eImmediate: return "Immediate";
      case PresentModeKHR::eMailbox: return "Mailbox";
      case PresentModeKHR::eFifo: return "Fifo";
      case PresentModeKHR::eFifoRelaxed: return "FifoRelaxed";
      case PresentModeKHR::eSharedDemandRefresh: return "SharedDemandRefresh";
      case PresentModeKHR::eSharedContinuousRefresh: return "SharedContinuousRefresh";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ColorSpaceKHR value )
  {
    switch ( value )
    {
      case ColorSpaceKHR::eSrgbNonlinear: return "SrgbNonlinear";
      case ColorSpaceKHR::eDisplayP3NonlinearEXT: return "DisplayP3NonlinearEXT";
      case ColorSpaceKHR::eExtendedSrgbLinearEXT: return "ExtendedSrgbLinearEXT";
      case ColorSpaceKHR::eDisplayP3LinearEXT: return "DisplayP3LinearEXT";
      case ColorSpaceKHR::eDciP3NonlinearEXT: return "DciP3NonlinearEXT";
      case ColorSpaceKHR::eBt709LinearEXT: return "Bt709LinearEXT";
      case ColorSpaceKHR::eBt709NonlinearEXT: return "Bt709NonlinearEXT";
      case ColorSpaceKHR::eBt2020LinearEXT: return "Bt2020LinearEXT";
      case ColorSpaceKHR::eHdr10St2084EXT: return "Hdr10St2084EXT";
      case ColorSpaceKHR::eDolbyvisionEXT: return "DolbyvisionEXT";
      case ColorSpaceKHR::eHdr10HlgEXT: return "Hdr10HlgEXT";
      case ColorSpaceKHR::eAdobergbLinearEXT: return "AdobergbLinearEXT";
      case ColorSpaceKHR::eAdobergbNonlinearEXT: return "AdobergbNonlinearEXT";
      case ColorSpaceKHR::ePassThroughEXT: return "PassThroughEXT";
      case ColorSpaceKHR::eExtendedSrgbNonlinearEXT: return "ExtendedSrgbNonlinearEXT";
      case ColorSpaceKHR::eDisplayNativeAMD: return "DisplayNativeAMD";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( CompositeAlphaFlagBitsKHR value )
  {
    switch ( value )
    {
      case CompositeAlphaFlagBitsKHR::eOpaque: return "Opaque";
      case CompositeAlphaFlagBitsKHR::ePreMultiplied: return "PreMultiplied";
      case CompositeAlphaFlagBitsKHR::ePostMultiplied: return "PostMultiplied";
      case CompositeAlphaFlagBitsKHR::eInherit: return "Inherit";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_KHR_swapchain ===

  VULKAN_HPP_INLINE std::string to_string( SwapchainCreateFlagBitsKHR value )
  {
    switch ( value )
    {
      case SwapchainCreateFlagBitsKHR::eSplitInstanceBindRegions: return "SplitInstanceBindRegions";
      case SwapchainCreateFlagBitsKHR::eProtected: return "Protected";
      case SwapchainCreateFlagBitsKHR::eMutableFormat: return "MutableFormat";
      case SwapchainCreateFlagBitsKHR::eDeferredMemoryAllocationEXT: return "DeferredMemoryAllocationEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( DeviceGroupPresentModeFlagBitsKHR value )
  {
    switch ( value )
    {
      case DeviceGroupPresentModeFlagBitsKHR::eLocal: return "Local";
      case DeviceGroupPresentModeFlagBitsKHR::eRemote: return "Remote";
      case DeviceGroupPresentModeFlagBitsKHR::eSum: return "Sum";
      case DeviceGroupPresentModeFlagBitsKHR::eLocalMultiDevice: return "LocalMultiDevice";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_KHR_display ===

  VULKAN_HPP_INLINE std::string to_string( DisplayPlaneAlphaFlagBitsKHR value )
  {
    switch ( value )
    {
      case DisplayPlaneAlphaFlagBitsKHR::eOpaque: return "Opaque";
      case DisplayPlaneAlphaFlagBitsKHR::eGlobal: return "Global";
      case DisplayPlaneAlphaFlagBitsKHR::ePerPixel: return "PerPixel";
      case DisplayPlaneAlphaFlagBitsKHR::ePerPixelPremultiplied: return "PerPixelPremultiplied";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( DisplayModeCreateFlagBitsKHR )
  {
    return "(void)";
  }

  VULKAN_HPP_INLINE std::string to_string( DisplaySurfaceCreateFlagBitsKHR )
  {
    return "(void)";
  }

#if defined( VK_USE_PLATFORM_XLIB_KHR )
  //=== VK_KHR_xlib_surface ===

  VULKAN_HPP_INLINE std::string to_string( XlibSurfaceCreateFlagBitsKHR )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_XLIB_KHR*/

#if defined( VK_USE_PLATFORM_XCB_KHR )
  //=== VK_KHR_xcb_surface ===

  VULKAN_HPP_INLINE std::string to_string( XcbSurfaceCreateFlagBitsKHR )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_XCB_KHR*/

#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
  //=== VK_KHR_wayland_surface ===

  VULKAN_HPP_INLINE std::string to_string( WaylandSurfaceCreateFlagBitsKHR )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  //=== VK_KHR_android_surface ===

  VULKAN_HPP_INLINE std::string to_string( AndroidSurfaceCreateFlagBitsKHR )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_win32_surface ===

  VULKAN_HPP_INLINE std::string to_string( Win32SurfaceCreateFlagBitsKHR )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_EXT_debug_report ===

  VULKAN_HPP_INLINE std::string to_string( DebugReportFlagBitsEXT value )
  {
    switch ( value )
    {
      case DebugReportFlagBitsEXT::eInformation: return "Information";
      case DebugReportFlagBitsEXT::eWarning: return "Warning";
      case DebugReportFlagBitsEXT::ePerformanceWarning: return "PerformanceWarning";
      case DebugReportFlagBitsEXT::eError: return "Error";
      case DebugReportFlagBitsEXT::eDebug: return "Debug";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( DebugReportObjectTypeEXT value )
  {
    switch ( value )
    {
      case DebugReportObjectTypeEXT::eUnknown: return "Unknown";
      case DebugReportObjectTypeEXT::eInstance: return "Instance";
      case DebugReportObjectTypeEXT::ePhysicalDevice: return "PhysicalDevice";
      case DebugReportObjectTypeEXT::eDevice: return "Device";
      case DebugReportObjectTypeEXT::eQueue: return "Queue";
      case DebugReportObjectTypeEXT::eSemaphore: return "Semaphore";
      case DebugReportObjectTypeEXT::eCommandBuffer: return "CommandBuffer";
      case DebugReportObjectTypeEXT::eFence: return "Fence";
      case DebugReportObjectTypeEXT::eDeviceMemory: return "DeviceMemory";
      case DebugReportObjectTypeEXT::eBuffer: return "Buffer";
      case DebugReportObjectTypeEXT::eImage: return "Image";
      case DebugReportObjectTypeEXT::eEvent: return "Event";
      case DebugReportObjectTypeEXT::eQueryPool: return "QueryPool";
      case DebugReportObjectTypeEXT::eBufferView: return "BufferView";
      case DebugReportObjectTypeEXT::eImageView: return "ImageView";
      case DebugReportObjectTypeEXT::eShaderModule: return "ShaderModule";
      case DebugReportObjectTypeEXT::ePipelineCache: return "PipelineCache";
      case DebugReportObjectTypeEXT::ePipelineLayout: return "PipelineLayout";
      case DebugReportObjectTypeEXT::eRenderPass: return "RenderPass";
      case DebugReportObjectTypeEXT::ePipeline: return "Pipeline";
      case DebugReportObjectTypeEXT::eDescriptorSetLayout: return "DescriptorSetLayout";
      case DebugReportObjectTypeEXT::eSampler: return "Sampler";
      case DebugReportObjectTypeEXT::eDescriptorPool: return "DescriptorPool";
      case DebugReportObjectTypeEXT::eDescriptorSet: return "DescriptorSet";
      case DebugReportObjectTypeEXT::eFramebuffer: return "Framebuffer";
      case DebugReportObjectTypeEXT::eCommandPool: return "CommandPool";
      case DebugReportObjectTypeEXT::eSurfaceKHR: return "SurfaceKHR";
      case DebugReportObjectTypeEXT::eSwapchainKHR: return "SwapchainKHR";
      case DebugReportObjectTypeEXT::eDebugReportCallbackEXT: return "DebugReportCallbackEXT";
      case DebugReportObjectTypeEXT::eDisplayKHR: return "DisplayKHR";
      case DebugReportObjectTypeEXT::eDisplayModeKHR: return "DisplayModeKHR";
      case DebugReportObjectTypeEXT::eValidationCacheEXT: return "ValidationCacheEXT";
      case DebugReportObjectTypeEXT::eSamplerYcbcrConversion: return "SamplerYcbcrConversion";
      case DebugReportObjectTypeEXT::eDescriptorUpdateTemplate: return "DescriptorUpdateTemplate";
      case DebugReportObjectTypeEXT::eCuModuleNVX: return "CuModuleNVX";
      case DebugReportObjectTypeEXT::eCuFunctionNVX: return "CuFunctionNVX";
      case DebugReportObjectTypeEXT::eAccelerationStructureKHR: return "AccelerationStructureKHR";
      case DebugReportObjectTypeEXT::eAccelerationStructureNV: return "AccelerationStructureNV";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case DebugReportObjectTypeEXT::eCudaModuleNV: return "CudaModuleNV";
      case DebugReportObjectTypeEXT::eCudaFunctionNV: return "CudaFunctionNV";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_USE_PLATFORM_FUCHSIA )
      case DebugReportObjectTypeEXT::eBufferCollectionFUCHSIA: return "BufferCollectionFUCHSIA";
#endif /*VK_USE_PLATFORM_FUCHSIA*/
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_AMD_rasterization_order ===

  VULKAN_HPP_INLINE std::string to_string( RasterizationOrderAMD value )
  {
    switch ( value )
    {
      case RasterizationOrderAMD::eStrict: return "Strict";
      case RasterizationOrderAMD::eRelaxed: return "Relaxed";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_KHR_video_queue ===

  VULKAN_HPP_INLINE std::string to_string( VideoCodecOperationFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoCodecOperationFlagBitsKHR::eNone: return "None";
      case VideoCodecOperationFlagBitsKHR::eEncodeH264: return "EncodeH264";
      case VideoCodecOperationFlagBitsKHR::eEncodeH265: return "EncodeH265";
      case VideoCodecOperationFlagBitsKHR::eDecodeH264: return "DecodeH264";
      case VideoCodecOperationFlagBitsKHR::eDecodeH265: return "DecodeH265";
      case VideoCodecOperationFlagBitsKHR::eDecodeAv1: return "DecodeAv1";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( VideoChromaSubsamplingFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoChromaSubsamplingFlagBitsKHR::eInvalid: return "Invalid";
      case VideoChromaSubsamplingFlagBitsKHR::eMonochrome: return "Monochrome";
      case VideoChromaSubsamplingFlagBitsKHR::e420: return "420";
      case VideoChromaSubsamplingFlagBitsKHR::e422: return "422";
      case VideoChromaSubsamplingFlagBitsKHR::e444: return "444";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( VideoComponentBitDepthFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoComponentBitDepthFlagBitsKHR::eInvalid: return "Invalid";
      case VideoComponentBitDepthFlagBitsKHR::e8: return "8";
      case VideoComponentBitDepthFlagBitsKHR::e10: return "10";
      case VideoComponentBitDepthFlagBitsKHR::e12: return "12";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( VideoCapabilityFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoCapabilityFlagBitsKHR::eProtectedContent: return "ProtectedContent";
      case VideoCapabilityFlagBitsKHR::eSeparateReferenceImages: return "SeparateReferenceImages";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( VideoSessionCreateFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoSessionCreateFlagBitsKHR::eProtectedContent: return "ProtectedContent";
      case VideoSessionCreateFlagBitsKHR::eAllowEncodeParameterOptimizations: return "AllowEncodeParameterOptimizations";
      case VideoSessionCreateFlagBitsKHR::eInlineQueries: return "InlineQueries";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( VideoCodingControlFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoCodingControlFlagBitsKHR::eReset: return "Reset";
      case VideoCodingControlFlagBitsKHR::eEncodeRateControl: return "EncodeRateControl";
      case VideoCodingControlFlagBitsKHR::eEncodeQualityLevel: return "EncodeQualityLevel";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( QueryResultStatusKHR value )
  {
    switch ( value )
    {
      case QueryResultStatusKHR::eError: return "Error";
      case QueryResultStatusKHR::eNotReady: return "NotReady";
      case QueryResultStatusKHR::eComplete: return "Complete";
      case QueryResultStatusKHR::eInsufficientBitstreamBufferRange: return "InsufficientBitstreamBufferRange";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( VideoSessionParametersCreateFlagBitsKHR )
  {
    return "(void)";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoBeginCodingFlagBitsKHR )
  {
    return "(void)";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEndCodingFlagBitsKHR )
  {
    return "(void)";
  }

  //=== VK_KHR_video_decode_queue ===

  VULKAN_HPP_INLINE std::string to_string( VideoDecodeCapabilityFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoDecodeCapabilityFlagBitsKHR::eDpbAndOutputCoincide: return "DpbAndOutputCoincide";
      case VideoDecodeCapabilityFlagBitsKHR::eDpbAndOutputDistinct: return "DpbAndOutputDistinct";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( VideoDecodeUsageFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoDecodeUsageFlagBitsKHR::eDefault: return "Default";
      case VideoDecodeUsageFlagBitsKHR::eTranscoding: return "Transcoding";
      case VideoDecodeUsageFlagBitsKHR::eOffline: return "Offline";
      case VideoDecodeUsageFlagBitsKHR::eStreaming: return "Streaming";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( VideoDecodeFlagBitsKHR )
  {
    return "(void)";
  }

  //=== VK_EXT_transform_feedback ===

  VULKAN_HPP_INLINE std::string to_string( PipelineRasterizationStateStreamCreateFlagBitsEXT )
  {
    return "(void)";
  }

  //=== VK_KHR_video_encode_h264 ===

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeH264CapabilityFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoEncodeH264CapabilityFlagBitsKHR::eHrdCompliance: return "HrdCompliance";
      case VideoEncodeH264CapabilityFlagBitsKHR::ePredictionWeightTableGenerated: return "PredictionWeightTableGenerated";
      case VideoEncodeH264CapabilityFlagBitsKHR::eRowUnalignedSlice: return "RowUnalignedSlice";
      case VideoEncodeH264CapabilityFlagBitsKHR::eDifferentSliceType: return "DifferentSliceType";
      case VideoEncodeH264CapabilityFlagBitsKHR::eBFrameInL0List: return "BFrameInL0List";
      case VideoEncodeH264CapabilityFlagBitsKHR::eBFrameInL1List: return "BFrameInL1List";
      case VideoEncodeH264CapabilityFlagBitsKHR::ePerPictureTypeMinMaxQp: return "PerPictureTypeMinMaxQp";
      case VideoEncodeH264CapabilityFlagBitsKHR::ePerSliceConstantQp: return "PerSliceConstantQp";
      case VideoEncodeH264CapabilityFlagBitsKHR::eGeneratePrefixNalu: return "GeneratePrefixNalu";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeH264StdFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoEncodeH264StdFlagBitsKHR::eSeparateColorPlaneFlagSet: return "SeparateColorPlaneFlagSet";
      case VideoEncodeH264StdFlagBitsKHR::eQpprimeYZeroTransformBypassFlagSet: return "QpprimeYZeroTransformBypassFlagSet";
      case VideoEncodeH264StdFlagBitsKHR::eScalingMatrixPresentFlagSet: return "ScalingMatrixPresentFlagSet";
      case VideoEncodeH264StdFlagBitsKHR::eChromaQpIndexOffset: return "ChromaQpIndexOffset";
      case VideoEncodeH264StdFlagBitsKHR::eSecondChromaQpIndexOffset: return "SecondChromaQpIndexOffset";
      case VideoEncodeH264StdFlagBitsKHR::ePicInitQpMinus26: return "PicInitQpMinus26";
      case VideoEncodeH264StdFlagBitsKHR::eWeightedPredFlagSet: return "WeightedPredFlagSet";
      case VideoEncodeH264StdFlagBitsKHR::eWeightedBipredIdcExplicit: return "WeightedBipredIdcExplicit";
      case VideoEncodeH264StdFlagBitsKHR::eWeightedBipredIdcImplicit: return "WeightedBipredIdcImplicit";
      case VideoEncodeH264StdFlagBitsKHR::eTransform8X8ModeFlagSet: return "Transform8X8ModeFlagSet";
      case VideoEncodeH264StdFlagBitsKHR::eDirectSpatialMvPredFlagUnset: return "DirectSpatialMvPredFlagUnset";
      case VideoEncodeH264StdFlagBitsKHR::eEntropyCodingModeFlagUnset: return "EntropyCodingModeFlagUnset";
      case VideoEncodeH264StdFlagBitsKHR::eEntropyCodingModeFlagSet: return "EntropyCodingModeFlagSet";
      case VideoEncodeH264StdFlagBitsKHR::eDirect8X8InferenceFlagUnset: return "Direct8X8InferenceFlagUnset";
      case VideoEncodeH264StdFlagBitsKHR::eConstrainedIntraPredFlagSet: return "ConstrainedIntraPredFlagSet";
      case VideoEncodeH264StdFlagBitsKHR::eDeblockingFilterDisabled: return "DeblockingFilterDisabled";
      case VideoEncodeH264StdFlagBitsKHR::eDeblockingFilterEnabled: return "DeblockingFilterEnabled";
      case VideoEncodeH264StdFlagBitsKHR::eDeblockingFilterPartial: return "DeblockingFilterPartial";
      case VideoEncodeH264StdFlagBitsKHR::eSliceQpDelta: return "SliceQpDelta";
      case VideoEncodeH264StdFlagBitsKHR::eDifferentSliceQpDelta: return "DifferentSliceQpDelta";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeH264RateControlFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoEncodeH264RateControlFlagBitsKHR::eAttemptHrdCompliance: return "AttemptHrdCompliance";
      case VideoEncodeH264RateControlFlagBitsKHR::eRegularGop: return "RegularGop";
      case VideoEncodeH264RateControlFlagBitsKHR::eReferencePatternFlat: return "ReferencePatternFlat";
      case VideoEncodeH264RateControlFlagBitsKHR::eReferencePatternDyadic: return "ReferencePatternDyadic";
      case VideoEncodeH264RateControlFlagBitsKHR::eTemporalLayerPatternDyadic: return "TemporalLayerPatternDyadic";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_KHR_video_encode_h265 ===

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeH265CapabilityFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoEncodeH265CapabilityFlagBitsKHR::eHrdCompliance: return "HrdCompliance";
      case VideoEncodeH265CapabilityFlagBitsKHR::ePredictionWeightTableGenerated: return "PredictionWeightTableGenerated";
      case VideoEncodeH265CapabilityFlagBitsKHR::eRowUnalignedSliceSegment: return "RowUnalignedSliceSegment";
      case VideoEncodeH265CapabilityFlagBitsKHR::eDifferentSliceSegmentType: return "DifferentSliceSegmentType";
      case VideoEncodeH265CapabilityFlagBitsKHR::eBFrameInL0List: return "BFrameInL0List";
      case VideoEncodeH265CapabilityFlagBitsKHR::eBFrameInL1List: return "BFrameInL1List";
      case VideoEncodeH265CapabilityFlagBitsKHR::ePerPictureTypeMinMaxQp: return "PerPictureTypeMinMaxQp";
      case VideoEncodeH265CapabilityFlagBitsKHR::ePerSliceSegmentConstantQp: return "PerSliceSegmentConstantQp";
      case VideoEncodeH265CapabilityFlagBitsKHR::eMultipleTilesPerSliceSegment: return "MultipleTilesPerSliceSegment";
      case VideoEncodeH265CapabilityFlagBitsKHR::eMultipleSliceSegmentsPerTile: return "MultipleSliceSegmentsPerTile";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeH265StdFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoEncodeH265StdFlagBitsKHR::eSeparateColorPlaneFlagSet: return "SeparateColorPlaneFlagSet";
      case VideoEncodeH265StdFlagBitsKHR::eSampleAdaptiveOffsetEnabledFlagSet: return "SampleAdaptiveOffsetEnabledFlagSet";
      case VideoEncodeH265StdFlagBitsKHR::eScalingListDataPresentFlagSet: return "ScalingListDataPresentFlagSet";
      case VideoEncodeH265StdFlagBitsKHR::ePcmEnabledFlagSet: return "PcmEnabledFlagSet";
      case VideoEncodeH265StdFlagBitsKHR::eSpsTemporalMvpEnabledFlagSet: return "SpsTemporalMvpEnabledFlagSet";
      case VideoEncodeH265StdFlagBitsKHR::eInitQpMinus26: return "InitQpMinus26";
      case VideoEncodeH265StdFlagBitsKHR::eWeightedPredFlagSet: return "WeightedPredFlagSet";
      case VideoEncodeH265StdFlagBitsKHR::eWeightedBipredFlagSet: return "WeightedBipredFlagSet";
      case VideoEncodeH265StdFlagBitsKHR::eLog2ParallelMergeLevelMinus2: return "Log2ParallelMergeLevelMinus2";
      case VideoEncodeH265StdFlagBitsKHR::eSignDataHidingEnabledFlagSet: return "SignDataHidingEnabledFlagSet";
      case VideoEncodeH265StdFlagBitsKHR::eTransformSkipEnabledFlagSet: return "TransformSkipEnabledFlagSet";
      case VideoEncodeH265StdFlagBitsKHR::eTransformSkipEnabledFlagUnset: return "TransformSkipEnabledFlagUnset";
      case VideoEncodeH265StdFlagBitsKHR::ePpsSliceChromaQpOffsetsPresentFlagSet: return "PpsSliceChromaQpOffsetsPresentFlagSet";
      case VideoEncodeH265StdFlagBitsKHR::eTransquantBypassEnabledFlagSet: return "TransquantBypassEnabledFlagSet";
      case VideoEncodeH265StdFlagBitsKHR::eConstrainedIntraPredFlagSet: return "ConstrainedIntraPredFlagSet";
      case VideoEncodeH265StdFlagBitsKHR::eEntropyCodingSyncEnabledFlagSet: return "EntropyCodingSyncEnabledFlagSet";
      case VideoEncodeH265StdFlagBitsKHR::eDeblockingFilterOverrideEnabledFlagSet: return "DeblockingFilterOverrideEnabledFlagSet";
      case VideoEncodeH265StdFlagBitsKHR::eDependentSliceSegmentsEnabledFlagSet: return "DependentSliceSegmentsEnabledFlagSet";
      case VideoEncodeH265StdFlagBitsKHR::eDependentSliceSegmentFlagSet: return "DependentSliceSegmentFlagSet";
      case VideoEncodeH265StdFlagBitsKHR::eSliceQpDelta: return "SliceQpDelta";
      case VideoEncodeH265StdFlagBitsKHR::eDifferentSliceQpDelta: return "DifferentSliceQpDelta";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeH265CtbSizeFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoEncodeH265CtbSizeFlagBitsKHR::e16: return "16";
      case VideoEncodeH265CtbSizeFlagBitsKHR::e32: return "32";
      case VideoEncodeH265CtbSizeFlagBitsKHR::e64: return "64";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeH265TransformBlockSizeFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoEncodeH265TransformBlockSizeFlagBitsKHR::e4: return "4";
      case VideoEncodeH265TransformBlockSizeFlagBitsKHR::e8: return "8";
      case VideoEncodeH265TransformBlockSizeFlagBitsKHR::e16: return "16";
      case VideoEncodeH265TransformBlockSizeFlagBitsKHR::e32: return "32";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeH265RateControlFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoEncodeH265RateControlFlagBitsKHR::eAttemptHrdCompliance: return "AttemptHrdCompliance";
      case VideoEncodeH265RateControlFlagBitsKHR::eRegularGop: return "RegularGop";
      case VideoEncodeH265RateControlFlagBitsKHR::eReferencePatternFlat: return "ReferencePatternFlat";
      case VideoEncodeH265RateControlFlagBitsKHR::eReferencePatternDyadic: return "ReferencePatternDyadic";
      case VideoEncodeH265RateControlFlagBitsKHR::eTemporalSubLayerPatternDyadic: return "TemporalSubLayerPatternDyadic";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_KHR_video_decode_h264 ===

  VULKAN_HPP_INLINE std::string to_string( VideoDecodeH264PictureLayoutFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoDecodeH264PictureLayoutFlagBitsKHR::eProgressive: return "Progressive";
      case VideoDecodeH264PictureLayoutFlagBitsKHR::eInterlacedInterleavedLines: return "InterlacedInterleavedLines";
      case VideoDecodeH264PictureLayoutFlagBitsKHR::eInterlacedSeparatePlanes: return "InterlacedSeparatePlanes";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_AMD_shader_info ===

  VULKAN_HPP_INLINE std::string to_string( ShaderInfoTypeAMD value )
  {
    switch ( value )
    {
      case ShaderInfoTypeAMD::eStatistics: return "Statistics";
      case ShaderInfoTypeAMD::eBinary: return "Binary";
      case ShaderInfoTypeAMD::eDisassembly: return "Disassembly";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

#if defined( VK_USE_PLATFORM_GGP )
  //=== VK_GGP_stream_descriptor_surface ===

  VULKAN_HPP_INLINE std::string to_string( StreamDescriptorSurfaceCreateFlagBitsGGP )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_GGP*/

  //=== VK_NV_external_memory_capabilities ===

  VULKAN_HPP_INLINE std::string to_string( ExternalMemoryHandleTypeFlagBitsNV value )
  {
    switch ( value )
    {
      case ExternalMemoryHandleTypeFlagBitsNV::eOpaqueWin32: return "OpaqueWin32";
      case ExternalMemoryHandleTypeFlagBitsNV::eOpaqueWin32Kmt: return "OpaqueWin32Kmt";
      case ExternalMemoryHandleTypeFlagBitsNV::eD3D11Image: return "D3D11Image";
      case ExternalMemoryHandleTypeFlagBitsNV::eD3D11ImageKmt: return "D3D11ImageKmt";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ExternalMemoryFeatureFlagBitsNV value )
  {
    switch ( value )
    {
      case ExternalMemoryFeatureFlagBitsNV::eDedicatedOnly: return "DedicatedOnly";
      case ExternalMemoryFeatureFlagBitsNV::eExportable: return "Exportable";
      case ExternalMemoryFeatureFlagBitsNV::eImportable: return "Importable";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_EXT_validation_flags ===

  VULKAN_HPP_INLINE std::string to_string( ValidationCheckEXT value )
  {
    switch ( value )
    {
      case ValidationCheckEXT::eAll: return "All";
      case ValidationCheckEXT::eShaders: return "Shaders";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

#if defined( VK_USE_PLATFORM_VI_NN )
  //=== VK_NN_vi_surface ===

  VULKAN_HPP_INLINE std::string to_string( ViSurfaceCreateFlagBitsNN )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_VI_NN*/

  //=== VK_EXT_pipeline_robustness ===

  VULKAN_HPP_INLINE std::string to_string( PipelineRobustnessBufferBehaviorEXT value )
  {
    switch ( value )
    {
      case PipelineRobustnessBufferBehaviorEXT::eDeviceDefault: return "DeviceDefault";
      case PipelineRobustnessBufferBehaviorEXT::eDisabled: return "Disabled";
      case PipelineRobustnessBufferBehaviorEXT::eRobustBufferAccess: return "RobustBufferAccess";
      case PipelineRobustnessBufferBehaviorEXT::eRobustBufferAccess2: return "RobustBufferAccess2";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineRobustnessImageBehaviorEXT value )
  {
    switch ( value )
    {
      case PipelineRobustnessImageBehaviorEXT::eDeviceDefault: return "DeviceDefault";
      case PipelineRobustnessImageBehaviorEXT::eDisabled: return "Disabled";
      case PipelineRobustnessImageBehaviorEXT::eRobustImageAccess: return "RobustImageAccess";
      case PipelineRobustnessImageBehaviorEXT::eRobustImageAccess2: return "RobustImageAccess2";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_EXT_conditional_rendering ===

  VULKAN_HPP_INLINE std::string to_string( ConditionalRenderingFlagBitsEXT value )
  {
    switch ( value )
    {
      case ConditionalRenderingFlagBitsEXT::eInverted: return "Inverted";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_EXT_display_surface_counter ===

  VULKAN_HPP_INLINE std::string to_string( SurfaceCounterFlagBitsEXT value )
  {
    switch ( value )
    {
      case SurfaceCounterFlagBitsEXT::eVblank: return "Vblank";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_EXT_display_control ===

  VULKAN_HPP_INLINE std::string to_string( DisplayPowerStateEXT value )
  {
    switch ( value )
    {
      case DisplayPowerStateEXT::eOff: return "Off";
      case DisplayPowerStateEXT::eSuspend: return "Suspend";
      case DisplayPowerStateEXT::eOn: return "On";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( DeviceEventTypeEXT value )
  {
    switch ( value )
    {
      case DeviceEventTypeEXT::eDisplayHotplug: return "DisplayHotplug";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( DisplayEventTypeEXT value )
  {
    switch ( value )
    {
      case DisplayEventTypeEXT::eFirstPixelOut: return "FirstPixelOut";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_NV_viewport_swizzle ===

  VULKAN_HPP_INLINE std::string to_string( ViewportCoordinateSwizzleNV value )
  {
    switch ( value )
    {
      case ViewportCoordinateSwizzleNV::ePositiveX: return "PositiveX";
      case ViewportCoordinateSwizzleNV::eNegativeX: return "NegativeX";
      case ViewportCoordinateSwizzleNV::ePositiveY: return "PositiveY";
      case ViewportCoordinateSwizzleNV::eNegativeY: return "NegativeY";
      case ViewportCoordinateSwizzleNV::ePositiveZ: return "PositiveZ";
      case ViewportCoordinateSwizzleNV::eNegativeZ: return "NegativeZ";
      case ViewportCoordinateSwizzleNV::ePositiveW: return "PositiveW";
      case ViewportCoordinateSwizzleNV::eNegativeW: return "NegativeW";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineViewportSwizzleStateCreateFlagBitsNV )
  {
    return "(void)";
  }

  //=== VK_EXT_discard_rectangles ===

  VULKAN_HPP_INLINE std::string to_string( DiscardRectangleModeEXT value )
  {
    switch ( value )
    {
      case DiscardRectangleModeEXT::eInclusive: return "Inclusive";
      case DiscardRectangleModeEXT::eExclusive: return "Exclusive";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineDiscardRectangleStateCreateFlagBitsEXT )
  {
    return "(void)";
  }

  //=== VK_EXT_conservative_rasterization ===

  VULKAN_HPP_INLINE std::string to_string( ConservativeRasterizationModeEXT value )
  {
    switch ( value )
    {
      case ConservativeRasterizationModeEXT::eDisabled: return "Disabled";
      case ConservativeRasterizationModeEXT::eOverestimate: return "Overestimate";
      case ConservativeRasterizationModeEXT::eUnderestimate: return "Underestimate";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineRasterizationConservativeStateCreateFlagBitsEXT )
  {
    return "(void)";
  }

  //=== VK_EXT_depth_clip_enable ===

  VULKAN_HPP_INLINE std::string to_string( PipelineRasterizationDepthClipStateCreateFlagBitsEXT )
  {
    return "(void)";
  }

  //=== VK_KHR_performance_query ===

  VULKAN_HPP_INLINE std::string to_string( PerformanceCounterDescriptionFlagBitsKHR value )
  {
    switch ( value )
    {
      case PerformanceCounterDescriptionFlagBitsKHR::ePerformanceImpacting: return "PerformanceImpacting";
      case PerformanceCounterDescriptionFlagBitsKHR::eConcurrentlyImpacted: return "ConcurrentlyImpacted";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PerformanceCounterScopeKHR value )
  {
    switch ( value )
    {
      case PerformanceCounterScopeKHR::eCommandBuffer: return "CommandBuffer";
      case PerformanceCounterScopeKHR::eRenderPass: return "RenderPass";
      case PerformanceCounterScopeKHR::eCommand: return "Command";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PerformanceCounterStorageKHR value )
  {
    switch ( value )
    {
      case PerformanceCounterStorageKHR::eInt32: return "Int32";
      case PerformanceCounterStorageKHR::eInt64: return "Int64";
      case PerformanceCounterStorageKHR::eUint32: return "Uint32";
      case PerformanceCounterStorageKHR::eUint64: return "Uint64";
      case PerformanceCounterStorageKHR::eFloat32: return "Float32";
      case PerformanceCounterStorageKHR::eFloat64: return "Float64";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PerformanceCounterUnitKHR value )
  {
    switch ( value )
    {
      case PerformanceCounterUnitKHR::eGeneric: return "Generic";
      case PerformanceCounterUnitKHR::ePercentage: return "Percentage";
      case PerformanceCounterUnitKHR::eNanoseconds: return "Nanoseconds";
      case PerformanceCounterUnitKHR::eBytes: return "Bytes";
      case PerformanceCounterUnitKHR::eBytesPerSecond: return "BytesPerSecond";
      case PerformanceCounterUnitKHR::eKelvin: return "Kelvin";
      case PerformanceCounterUnitKHR::eWatts: return "Watts";
      case PerformanceCounterUnitKHR::eVolts: return "Volts";
      case PerformanceCounterUnitKHR::eAmps: return "Amps";
      case PerformanceCounterUnitKHR::eHertz: return "Hertz";
      case PerformanceCounterUnitKHR::eCycles: return "Cycles";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( AcquireProfilingLockFlagBitsKHR )
  {
    return "(void)";
  }

#if defined( VK_USE_PLATFORM_IOS_MVK )
  //=== VK_MVK_ios_surface ===

  VULKAN_HPP_INLINE std::string to_string( IOSSurfaceCreateFlagBitsMVK )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_IOS_MVK*/

#if defined( VK_USE_PLATFORM_MACOS_MVK )
  //=== VK_MVK_macos_surface ===

  VULKAN_HPP_INLINE std::string to_string( MacOSSurfaceCreateFlagBitsMVK )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_MACOS_MVK*/

  //=== VK_EXT_debug_utils ===

  VULKAN_HPP_INLINE std::string to_string( DebugUtilsMessageSeverityFlagBitsEXT value )
  {
    switch ( value )
    {
      case DebugUtilsMessageSeverityFlagBitsEXT::eVerbose: return "Verbose";
      case DebugUtilsMessageSeverityFlagBitsEXT::eInfo: return "Info";
      case DebugUtilsMessageSeverityFlagBitsEXT::eWarning: return "Warning";
      case DebugUtilsMessageSeverityFlagBitsEXT::eError: return "Error";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( DebugUtilsMessageTypeFlagBitsEXT value )
  {
    switch ( value )
    {
      case DebugUtilsMessageTypeFlagBitsEXT::eGeneral: return "General";
      case DebugUtilsMessageTypeFlagBitsEXT::eValidation: return "Validation";
      case DebugUtilsMessageTypeFlagBitsEXT::ePerformance: return "Performance";
      case DebugUtilsMessageTypeFlagBitsEXT::eDeviceAddressBinding: return "DeviceAddressBinding";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( DebugUtilsMessengerCallbackDataFlagBitsEXT )
  {
    return "(void)";
  }

  VULKAN_HPP_INLINE std::string to_string( DebugUtilsMessengerCreateFlagBitsEXT )
  {
    return "(void)";
  }

  //=== VK_EXT_blend_operation_advanced ===

  VULKAN_HPP_INLINE std::string to_string( BlendOverlapEXT value )
  {
    switch ( value )
    {
      case BlendOverlapEXT::eUncorrelated: return "Uncorrelated";
      case BlendOverlapEXT::eDisjoint: return "Disjoint";
      case BlendOverlapEXT::eConjoint: return "Conjoint";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_NV_fragment_coverage_to_color ===

  VULKAN_HPP_INLINE std::string to_string( PipelineCoverageToColorStateCreateFlagBitsNV )
  {
    return "(void)";
  }

  //=== VK_KHR_acceleration_structure ===

  VULKAN_HPP_INLINE std::string to_string( AccelerationStructureTypeKHR value )
  {
    switch ( value )
    {
      case AccelerationStructureTypeKHR::eTopLevel: return "TopLevel";
      case AccelerationStructureTypeKHR::eBottomLevel: return "BottomLevel";
      case AccelerationStructureTypeKHR::eGeneric: return "Generic";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( AccelerationStructureBuildTypeKHR value )
  {
    switch ( value )
    {
      case AccelerationStructureBuildTypeKHR::eHost: return "Host";
      case AccelerationStructureBuildTypeKHR::eDevice: return "Device";
      case AccelerationStructureBuildTypeKHR::eHostOrDevice: return "HostOrDevice";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( GeometryFlagBitsKHR value )
  {
    switch ( value )
    {
      case GeometryFlagBitsKHR::eOpaque: return "Opaque";
      case GeometryFlagBitsKHR::eNoDuplicateAnyHitInvocation: return "NoDuplicateAnyHitInvocation";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( GeometryInstanceFlagBitsKHR value )
  {
    switch ( value )
    {
      case GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable: return "TriangleFacingCullDisable";
      case GeometryInstanceFlagBitsKHR::eTriangleFlipFacing: return "TriangleFlipFacing";
      case GeometryInstanceFlagBitsKHR::eForceOpaque: return "ForceOpaque";
      case GeometryInstanceFlagBitsKHR::eForceNoOpaque: return "ForceNoOpaque";
      case GeometryInstanceFlagBitsKHR::eForceOpacityMicromap2StateEXT: return "ForceOpacityMicromap2StateEXT";
      case GeometryInstanceFlagBitsKHR::eDisableOpacityMicromapsEXT: return "DisableOpacityMicromapsEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( BuildAccelerationStructureFlagBitsKHR value )
  {
    switch ( value )
    {
      case BuildAccelerationStructureFlagBitsKHR::eAllowUpdate: return "AllowUpdate";
      case BuildAccelerationStructureFlagBitsKHR::eAllowCompaction: return "AllowCompaction";
      case BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace: return "PreferFastTrace";
      case BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild: return "PreferFastBuild";
      case BuildAccelerationStructureFlagBitsKHR::eLowMemory: return "LowMemory";
      case BuildAccelerationStructureFlagBitsKHR::eMotionNV: return "MotionNV";
      case BuildAccelerationStructureFlagBitsKHR::eAllowOpacityMicromapUpdateEXT: return "AllowOpacityMicromapUpdateEXT";
      case BuildAccelerationStructureFlagBitsKHR::eAllowDisableOpacityMicromapsEXT: return "AllowDisableOpacityMicromapsEXT";
      case BuildAccelerationStructureFlagBitsKHR::eAllowOpacityMicromapDataUpdateEXT: return "AllowOpacityMicromapDataUpdateEXT";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case BuildAccelerationStructureFlagBitsKHR::eAllowDisplacementMicromapUpdateNV: return "AllowDisplacementMicromapUpdateNV";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      case BuildAccelerationStructureFlagBitsKHR::eAllowDataAccess: return "AllowDataAccess";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( CopyAccelerationStructureModeKHR value )
  {
    switch ( value )
    {
      case CopyAccelerationStructureModeKHR::eClone: return "Clone";
      case CopyAccelerationStructureModeKHR::eCompact: return "Compact";
      case CopyAccelerationStructureModeKHR::eSerialize: return "Serialize";
      case CopyAccelerationStructureModeKHR::eDeserialize: return "Deserialize";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( GeometryTypeKHR value )
  {
    switch ( value )
    {
      case GeometryTypeKHR::eTriangles: return "Triangles";
      case GeometryTypeKHR::eAabbs: return "Aabbs";
      case GeometryTypeKHR::eInstances: return "Instances";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( AccelerationStructureCompatibilityKHR value )
  {
    switch ( value )
    {
      case AccelerationStructureCompatibilityKHR::eCompatible: return "Compatible";
      case AccelerationStructureCompatibilityKHR::eIncompatible: return "Incompatible";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( AccelerationStructureCreateFlagBitsKHR value )
  {
    switch ( value )
    {
      case AccelerationStructureCreateFlagBitsKHR::eDeviceAddressCaptureReplay: return "DeviceAddressCaptureReplay";
      case AccelerationStructureCreateFlagBitsKHR::eDescriptorBufferCaptureReplayEXT: return "DescriptorBufferCaptureReplayEXT";
      case AccelerationStructureCreateFlagBitsKHR::eMotionNV: return "MotionNV";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( BuildAccelerationStructureModeKHR value )
  {
    switch ( value )
    {
      case BuildAccelerationStructureModeKHR::eBuild: return "Build";
      case BuildAccelerationStructureModeKHR::eUpdate: return "Update";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_KHR_ray_tracing_pipeline ===

  VULKAN_HPP_INLINE std::string to_string( RayTracingShaderGroupTypeKHR value )
  {
    switch ( value )
    {
      case RayTracingShaderGroupTypeKHR::eGeneral: return "General";
      case RayTracingShaderGroupTypeKHR::eTrianglesHitGroup: return "TrianglesHitGroup";
      case RayTracingShaderGroupTypeKHR::eProceduralHitGroup: return "ProceduralHitGroup";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ShaderGroupShaderKHR value )
  {
    switch ( value )
    {
      case ShaderGroupShaderKHR::eGeneral: return "General";
      case ShaderGroupShaderKHR::eClosestHit: return "ClosestHit";
      case ShaderGroupShaderKHR::eAnyHit: return "AnyHit";
      case ShaderGroupShaderKHR::eIntersection: return "Intersection";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_NV_framebuffer_mixed_samples ===

  VULKAN_HPP_INLINE std::string to_string( CoverageModulationModeNV value )
  {
    switch ( value )
    {
      case CoverageModulationModeNV::eNone: return "None";
      case CoverageModulationModeNV::eRgb: return "Rgb";
      case CoverageModulationModeNV::eAlpha: return "Alpha";
      case CoverageModulationModeNV::eRgba: return "Rgba";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineCoverageModulationStateCreateFlagBitsNV )
  {
    return "(void)";
  }

  //=== VK_EXT_validation_cache ===

  VULKAN_HPP_INLINE std::string to_string( ValidationCacheHeaderVersionEXT value )
  {
    switch ( value )
    {
      case ValidationCacheHeaderVersionEXT::eOne: return "One";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ValidationCacheCreateFlagBitsEXT )
  {
    return "(void)";
  }

  //=== VK_NV_shading_rate_image ===

  VULKAN_HPP_INLINE std::string to_string( ShadingRatePaletteEntryNV value )
  {
    switch ( value )
    {
      case ShadingRatePaletteEntryNV::eNoInvocations: return "NoInvocations";
      case ShadingRatePaletteEntryNV::e16InvocationsPerPixel: return "16InvocationsPerPixel";
      case ShadingRatePaletteEntryNV::e8InvocationsPerPixel: return "8InvocationsPerPixel";
      case ShadingRatePaletteEntryNV::e4InvocationsPerPixel: return "4InvocationsPerPixel";
      case ShadingRatePaletteEntryNV::e2InvocationsPerPixel: return "2InvocationsPerPixel";
      case ShadingRatePaletteEntryNV::e1InvocationPerPixel: return "1InvocationPerPixel";
      case ShadingRatePaletteEntryNV::e1InvocationPer2X1Pixels: return "1InvocationPer2X1Pixels";
      case ShadingRatePaletteEntryNV::e1InvocationPer1X2Pixels: return "1InvocationPer1X2Pixels";
      case ShadingRatePaletteEntryNV::e1InvocationPer2X2Pixels: return "1InvocationPer2X2Pixels";
      case ShadingRatePaletteEntryNV::e1InvocationPer4X2Pixels: return "1InvocationPer4X2Pixels";
      case ShadingRatePaletteEntryNV::e1InvocationPer2X4Pixels: return "1InvocationPer2X4Pixels";
      case ShadingRatePaletteEntryNV::e1InvocationPer4X4Pixels: return "1InvocationPer4X4Pixels";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( CoarseSampleOrderTypeNV value )
  {
    switch ( value )
    {
      case CoarseSampleOrderTypeNV::eDefault: return "Default";
      case CoarseSampleOrderTypeNV::eCustom: return "Custom";
      case CoarseSampleOrderTypeNV::ePixelMajor: return "PixelMajor";
      case CoarseSampleOrderTypeNV::eSampleMajor: return "SampleMajor";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_NV_ray_tracing ===

  VULKAN_HPP_INLINE std::string to_string( AccelerationStructureMemoryRequirementsTypeNV value )
  {
    switch ( value )
    {
      case AccelerationStructureMemoryRequirementsTypeNV::eObject: return "Object";
      case AccelerationStructureMemoryRequirementsTypeNV::eBuildScratch: return "BuildScratch";
      case AccelerationStructureMemoryRequirementsTypeNV::eUpdateScratch: return "UpdateScratch";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_AMD_pipeline_compiler_control ===

  VULKAN_HPP_INLINE std::string to_string( PipelineCompilerControlFlagBitsAMD )
  {
    return "(void)";
  }

  //=== VK_KHR_global_priority ===

  VULKAN_HPP_INLINE std::string to_string( QueueGlobalPriorityKHR value )
  {
    switch ( value )
    {
      case QueueGlobalPriorityKHR::eLow: return "Low";
      case QueueGlobalPriorityKHR::eMedium: return "Medium";
      case QueueGlobalPriorityKHR::eHigh: return "High";
      case QueueGlobalPriorityKHR::eRealtime: return "Realtime";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_AMD_memory_overallocation_behavior ===

  VULKAN_HPP_INLINE std::string to_string( MemoryOverallocationBehaviorAMD value )
  {
    switch ( value )
    {
      case MemoryOverallocationBehaviorAMD::eDefault: return "Default";
      case MemoryOverallocationBehaviorAMD::eAllowed: return "Allowed";
      case MemoryOverallocationBehaviorAMD::eDisallowed: return "Disallowed";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_INTEL_performance_query ===

  VULKAN_HPP_INLINE std::string to_string( PerformanceConfigurationTypeINTEL value )
  {
    switch ( value )
    {
      case PerformanceConfigurationTypeINTEL::eCommandQueueMetricsDiscoveryActivated: return "CommandQueueMetricsDiscoveryActivated";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( QueryPoolSamplingModeINTEL value )
  {
    switch ( value )
    {
      case QueryPoolSamplingModeINTEL::eManual: return "Manual";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PerformanceOverrideTypeINTEL value )
  {
    switch ( value )
    {
      case PerformanceOverrideTypeINTEL::eNullHardware: return "NullHardware";
      case PerformanceOverrideTypeINTEL::eFlushGpuCaches: return "FlushGpuCaches";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PerformanceParameterTypeINTEL value )
  {
    switch ( value )
    {
      case PerformanceParameterTypeINTEL::eHwCountersSupported: return "HwCountersSupported";
      case PerformanceParameterTypeINTEL::eStreamMarkerValidBits: return "StreamMarkerValidBits";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PerformanceValueTypeINTEL value )
  {
    switch ( value )
    {
      case PerformanceValueTypeINTEL::eUint32: return "Uint32";
      case PerformanceValueTypeINTEL::eUint64: return "Uint64";
      case PerformanceValueTypeINTEL::eFloat: return "Float";
      case PerformanceValueTypeINTEL::eBool: return "Bool";
      case PerformanceValueTypeINTEL::eString: return "String";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_imagepipe_surface ===

  VULKAN_HPP_INLINE std::string to_string( ImagePipeSurfaceCreateFlagBitsFUCHSIA )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_surface ===

  VULKAN_HPP_INLINE std::string to_string( MetalSurfaceCreateFlagBitsEXT )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_KHR_fragment_shading_rate ===

  VULKAN_HPP_INLINE std::string to_string( FragmentShadingRateCombinerOpKHR value )
  {
    switch ( value )
    {
      case FragmentShadingRateCombinerOpKHR::eKeep: return "Keep";
      case FragmentShadingRateCombinerOpKHR::eReplace: return "Replace";
      case FragmentShadingRateCombinerOpKHR::eMin: return "Min";
      case FragmentShadingRateCombinerOpKHR::eMax: return "Max";
      case FragmentShadingRateCombinerOpKHR::eMul: return "Mul";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_AMD_shader_core_properties2 ===

  VULKAN_HPP_INLINE std::string to_string( ShaderCorePropertiesFlagBitsAMD )
  {
    return "(void)";
  }

  //=== VK_EXT_validation_features ===

  VULKAN_HPP_INLINE std::string to_string( ValidationFeatureEnableEXT value )
  {
    switch ( value )
    {
      case ValidationFeatureEnableEXT::eGpuAssisted: return "GpuAssisted";
      case ValidationFeatureEnableEXT::eGpuAssistedReserveBindingSlot: return "GpuAssistedReserveBindingSlot";
      case ValidationFeatureEnableEXT::eBestPractices: return "BestPractices";
      case ValidationFeatureEnableEXT::eDebugPrintf: return "DebugPrintf";
      case ValidationFeatureEnableEXT::eSynchronizationValidation: return "SynchronizationValidation";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ValidationFeatureDisableEXT value )
  {
    switch ( value )
    {
      case ValidationFeatureDisableEXT::eAll: return "All";
      case ValidationFeatureDisableEXT::eShaders: return "Shaders";
      case ValidationFeatureDisableEXT::eThreadSafety: return "ThreadSafety";
      case ValidationFeatureDisableEXT::eApiParameters: return "ApiParameters";
      case ValidationFeatureDisableEXT::eObjectLifetimes: return "ObjectLifetimes";
      case ValidationFeatureDisableEXT::eCoreChecks: return "CoreChecks";
      case ValidationFeatureDisableEXT::eUniqueHandles: return "UniqueHandles";
      case ValidationFeatureDisableEXT::eShaderValidationCache: return "ShaderValidationCache";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_NV_coverage_reduction_mode ===

  VULKAN_HPP_INLINE std::string to_string( CoverageReductionModeNV value )
  {
    switch ( value )
    {
      case CoverageReductionModeNV::eMerge: return "Merge";
      case CoverageReductionModeNV::eTruncate: return "Truncate";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineCoverageReductionStateCreateFlagBitsNV )
  {
    return "(void)";
  }

  //=== VK_EXT_provoking_vertex ===

  VULKAN_HPP_INLINE std::string to_string( ProvokingVertexModeEXT value )
  {
    switch ( value )
    {
      case ProvokingVertexModeEXT::eFirstVertex: return "FirstVertex";
      case ProvokingVertexModeEXT::eLastVertex: return "LastVertex";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_EXT_full_screen_exclusive ===

  VULKAN_HPP_INLINE std::string to_string( FullScreenExclusiveEXT value )
  {
    switch ( value )
    {
      case FullScreenExclusiveEXT::eDefault: return "Default";
      case FullScreenExclusiveEXT::eAllowed: return "Allowed";
      case FullScreenExclusiveEXT::eDisallowed: return "Disallowed";
      case FullScreenExclusiveEXT::eApplicationControlled: return "ApplicationControlled";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_EXT_headless_surface ===

  VULKAN_HPP_INLINE std::string to_string( HeadlessSurfaceCreateFlagBitsEXT )
  {
    return "(void)";
  }

  //=== VK_KHR_pipeline_executable_properties ===

  VULKAN_HPP_INLINE std::string to_string( PipelineExecutableStatisticFormatKHR value )
  {
    switch ( value )
    {
      case PipelineExecutableStatisticFormatKHR::eBool32: return "Bool32";
      case PipelineExecutableStatisticFormatKHR::eInt64: return "Int64";
      case PipelineExecutableStatisticFormatKHR::eUint64: return "Uint64";
      case PipelineExecutableStatisticFormatKHR::eFloat64: return "Float64";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_EXT_host_image_copy ===

  VULKAN_HPP_INLINE std::string to_string( HostImageCopyFlagBitsEXT value )
  {
    switch ( value )
    {
      case HostImageCopyFlagBitsEXT::eMemcpy: return "Memcpy";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_KHR_map_memory2 ===

  VULKAN_HPP_INLINE std::string to_string( MemoryUnmapFlagBitsKHR value )
  {
    switch ( value )
    {
      case MemoryUnmapFlagBitsKHR::eReserveEXT: return "ReserveEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_EXT_surface_maintenance1 ===

  VULKAN_HPP_INLINE std::string to_string( PresentScalingFlagBitsEXT value )
  {
    switch ( value )
    {
      case PresentScalingFlagBitsEXT::eOneToOne: return "OneToOne";
      case PresentScalingFlagBitsEXT::eAspectRatioStretch: return "AspectRatioStretch";
      case PresentScalingFlagBitsEXT::eStretch: return "Stretch";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( PresentGravityFlagBitsEXT value )
  {
    switch ( value )
    {
      case PresentGravityFlagBitsEXT::eMin: return "Min";
      case PresentGravityFlagBitsEXT::eMax: return "Max";
      case PresentGravityFlagBitsEXT::eCentered: return "Centered";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_NV_device_generated_commands ===

  VULKAN_HPP_INLINE std::string to_string( IndirectStateFlagBitsNV value )
  {
    switch ( value )
    {
      case IndirectStateFlagBitsNV::eFlagFrontface: return "FlagFrontface";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( IndirectCommandsTokenTypeNV value )
  {
    switch ( value )
    {
      case IndirectCommandsTokenTypeNV::eShaderGroup: return "ShaderGroup";
      case IndirectCommandsTokenTypeNV::eStateFlags: return "StateFlags";
      case IndirectCommandsTokenTypeNV::eIndexBuffer: return "IndexBuffer";
      case IndirectCommandsTokenTypeNV::eVertexBuffer: return "VertexBuffer";
      case IndirectCommandsTokenTypeNV::ePushConstant: return "PushConstant";
      case IndirectCommandsTokenTypeNV::eDrawIndexed: return "DrawIndexed";
      case IndirectCommandsTokenTypeNV::eDraw: return "Draw";
      case IndirectCommandsTokenTypeNV::eDrawTasks: return "DrawTasks";
      case IndirectCommandsTokenTypeNV::eDrawMeshTasks: return "DrawMeshTasks";
      case IndirectCommandsTokenTypeNV::ePipeline: return "Pipeline";
      case IndirectCommandsTokenTypeNV::eDispatch: return "Dispatch";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( IndirectCommandsLayoutUsageFlagBitsNV value )
  {
    switch ( value )
    {
      case IndirectCommandsLayoutUsageFlagBitsNV::eExplicitPreprocess: return "ExplicitPreprocess";
      case IndirectCommandsLayoutUsageFlagBitsNV::eIndexedSequences: return "IndexedSequences";
      case IndirectCommandsLayoutUsageFlagBitsNV::eUnorderedSequences: return "UnorderedSequences";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_EXT_depth_bias_control ===

  VULKAN_HPP_INLINE std::string to_string( DepthBiasRepresentationEXT value )
  {
    switch ( value )
    {
      case DepthBiasRepresentationEXT::eLeastRepresentableValueFormat: return "LeastRepresentableValueFormat";
      case DepthBiasRepresentationEXT::eLeastRepresentableValueForceUnorm: return "LeastRepresentableValueForceUnorm";
      case DepthBiasRepresentationEXT::eFloat: return "Float";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_EXT_device_memory_report ===

  VULKAN_HPP_INLINE std::string to_string( DeviceMemoryReportEventTypeEXT value )
  {
    switch ( value )
    {
      case DeviceMemoryReportEventTypeEXT::eAllocate: return "Allocate";
      case DeviceMemoryReportEventTypeEXT::eFree: return "Free";
      case DeviceMemoryReportEventTypeEXT::eImport: return "Import";
      case DeviceMemoryReportEventTypeEXT::eUnimport: return "Unimport";
      case DeviceMemoryReportEventTypeEXT::eAllocationFailed: return "AllocationFailed";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( DeviceMemoryReportFlagBitsEXT )
  {
    return "(void)";
  }

  //=== VK_KHR_video_encode_queue ===

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeCapabilityFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoEncodeCapabilityFlagBitsKHR::ePrecedingExternallyEncodedBytes: return "PrecedingExternallyEncodedBytes";
      case VideoEncodeCapabilityFlagBitsKHR::eInsufficientBitstreamBufferRangeDetection: return "InsufficientBitstreamBufferRangeDetection";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeFeedbackFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoEncodeFeedbackFlagBitsKHR::eBitstreamBufferOffset: return "BitstreamBufferOffset";
      case VideoEncodeFeedbackFlagBitsKHR::eBitstreamBytesWritten: return "BitstreamBytesWritten";
      case VideoEncodeFeedbackFlagBitsKHR::eBitstreamHasOverrides: return "BitstreamHasOverrides";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeUsageFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoEncodeUsageFlagBitsKHR::eDefault: return "Default";
      case VideoEncodeUsageFlagBitsKHR::eTranscoding: return "Transcoding";
      case VideoEncodeUsageFlagBitsKHR::eStreaming: return "Streaming";
      case VideoEncodeUsageFlagBitsKHR::eRecording: return "Recording";
      case VideoEncodeUsageFlagBitsKHR::eConferencing: return "Conferencing";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeContentFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoEncodeContentFlagBitsKHR::eDefault: return "Default";
      case VideoEncodeContentFlagBitsKHR::eCamera: return "Camera";
      case VideoEncodeContentFlagBitsKHR::eDesktop: return "Desktop";
      case VideoEncodeContentFlagBitsKHR::eRendered: return "Rendered";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeTuningModeKHR value )
  {
    switch ( value )
    {
      case VideoEncodeTuningModeKHR::eDefault: return "Default";
      case VideoEncodeTuningModeKHR::eHighQuality: return "HighQuality";
      case VideoEncodeTuningModeKHR::eLowLatency: return "LowLatency";
      case VideoEncodeTuningModeKHR::eUltraLowLatency: return "UltraLowLatency";
      case VideoEncodeTuningModeKHR::eLossless: return "Lossless";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeRateControlModeFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoEncodeRateControlModeFlagBitsKHR::eDefault: return "Default";
      case VideoEncodeRateControlModeFlagBitsKHR::eDisabled: return "Disabled";
      case VideoEncodeRateControlModeFlagBitsKHR::eCbr: return "Cbr";
      case VideoEncodeRateControlModeFlagBitsKHR::eVbr: return "Vbr";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeFlagBitsKHR )
  {
    return "(void)";
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeRateControlFlagBitsKHR )
  {
    return "(void)";
  }

  //=== VK_NV_device_diagnostics_config ===

  VULKAN_HPP_INLINE std::string to_string( DeviceDiagnosticsConfigFlagBitsNV value )
  {
    switch ( value )
    {
      case DeviceDiagnosticsConfigFlagBitsNV::eEnableShaderDebugInfo: return "EnableShaderDebugInfo";
      case DeviceDiagnosticsConfigFlagBitsNV::eEnableResourceTracking: return "EnableResourceTracking";
      case DeviceDiagnosticsConfigFlagBitsNV::eEnableAutomaticCheckpoints: return "EnableAutomaticCheckpoints";
      case DeviceDiagnosticsConfigFlagBitsNV::eEnableShaderErrorReporting: return "EnableShaderErrorReporting";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_objects ===

  VULKAN_HPP_INLINE std::string to_string( ExportMetalObjectTypeFlagBitsEXT value )
  {
    switch ( value )
    {
      case ExportMetalObjectTypeFlagBitsEXT::eMetalDevice: return "MetalDevice";
      case ExportMetalObjectTypeFlagBitsEXT::eMetalCommandQueue: return "MetalCommandQueue";
      case ExportMetalObjectTypeFlagBitsEXT::eMetalBuffer: return "MetalBuffer";
      case ExportMetalObjectTypeFlagBitsEXT::eMetalTexture: return "MetalTexture";
      case ExportMetalObjectTypeFlagBitsEXT::eMetalIosurface: return "MetalIosurface";
      case ExportMetalObjectTypeFlagBitsEXT::eMetalSharedEvent: return "MetalSharedEvent";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_EXT_graphics_pipeline_library ===

  VULKAN_HPP_INLINE std::string to_string( GraphicsPipelineLibraryFlagBitsEXT value )
  {
    switch ( value )
    {
      case GraphicsPipelineLibraryFlagBitsEXT::eVertexInputInterface: return "VertexInputInterface";
      case GraphicsPipelineLibraryFlagBitsEXT::ePreRasterizationShaders: return "PreRasterizationShaders";
      case GraphicsPipelineLibraryFlagBitsEXT::eFragmentShader: return "FragmentShader";
      case GraphicsPipelineLibraryFlagBitsEXT::eFragmentOutputInterface: return "FragmentOutputInterface";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_NV_fragment_shading_rate_enums ===

  VULKAN_HPP_INLINE std::string to_string( FragmentShadingRateNV value )
  {
    switch ( value )
    {
      case FragmentShadingRateNV::e1InvocationPerPixel: return "1InvocationPerPixel";
      case FragmentShadingRateNV::e1InvocationPer1X2Pixels: return "1InvocationPer1X2Pixels";
      case FragmentShadingRateNV::e1InvocationPer2X1Pixels: return "1InvocationPer2X1Pixels";
      case FragmentShadingRateNV::e1InvocationPer2X2Pixels: return "1InvocationPer2X2Pixels";
      case FragmentShadingRateNV::e1InvocationPer2X4Pixels: return "1InvocationPer2X4Pixels";
      case FragmentShadingRateNV::e1InvocationPer4X2Pixels: return "1InvocationPer4X2Pixels";
      case FragmentShadingRateNV::e1InvocationPer4X4Pixels: return "1InvocationPer4X4Pixels";
      case FragmentShadingRateNV::e2InvocationsPerPixel: return "2InvocationsPerPixel";
      case FragmentShadingRateNV::e4InvocationsPerPixel: return "4InvocationsPerPixel";
      case FragmentShadingRateNV::e8InvocationsPerPixel: return "8InvocationsPerPixel";
      case FragmentShadingRateNV::e16InvocationsPerPixel: return "16InvocationsPerPixel";
      case FragmentShadingRateNV::eNoInvocations: return "NoInvocations";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( FragmentShadingRateTypeNV value )
  {
    switch ( value )
    {
      case FragmentShadingRateTypeNV::eFragmentSize: return "FragmentSize";
      case FragmentShadingRateTypeNV::eEnums: return "Enums";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_NV_ray_tracing_motion_blur ===

  VULKAN_HPP_INLINE std::string to_string( AccelerationStructureMotionInstanceTypeNV value )
  {
    switch ( value )
    {
      case AccelerationStructureMotionInstanceTypeNV::eStatic: return "Static";
      case AccelerationStructureMotionInstanceTypeNV::eMatrixMotion: return "MatrixMotion";
      case AccelerationStructureMotionInstanceTypeNV::eSrtMotion: return "SrtMotion";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( AccelerationStructureMotionInfoFlagBitsNV )
  {
    return "(void)";
  }

  VULKAN_HPP_INLINE std::string to_string( AccelerationStructureMotionInstanceFlagBitsNV )
  {
    return "(void)";
  }

  //=== VK_EXT_image_compression_control ===

  VULKAN_HPP_INLINE std::string to_string( ImageCompressionFlagBitsEXT value )
  {
    switch ( value )
    {
      case ImageCompressionFlagBitsEXT::eDefault: return "Default";
      case ImageCompressionFlagBitsEXT::eFixedRateDefault: return "FixedRateDefault";
      case ImageCompressionFlagBitsEXT::eFixedRateExplicit: return "FixedRateExplicit";
      case ImageCompressionFlagBitsEXT::eDisabled: return "Disabled";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ImageCompressionFixedRateFlagBitsEXT value )
  {
    switch ( value )
    {
      case ImageCompressionFixedRateFlagBitsEXT::eNone: return "None";
      case ImageCompressionFixedRateFlagBitsEXT::e1Bpc: return "1Bpc";
      case ImageCompressionFixedRateFlagBitsEXT::e2Bpc: return "2Bpc";
      case ImageCompressionFixedRateFlagBitsEXT::e3Bpc: return "3Bpc";
      case ImageCompressionFixedRateFlagBitsEXT::e4Bpc: return "4Bpc";
      case ImageCompressionFixedRateFlagBitsEXT::e5Bpc: return "5Bpc";
      case ImageCompressionFixedRateFlagBitsEXT::e6Bpc: return "6Bpc";
      case ImageCompressionFixedRateFlagBitsEXT::e7Bpc: return "7Bpc";
      case ImageCompressionFixedRateFlagBitsEXT::e8Bpc: return "8Bpc";
      case ImageCompressionFixedRateFlagBitsEXT::e9Bpc: return "9Bpc";
      case ImageCompressionFixedRateFlagBitsEXT::e10Bpc: return "10Bpc";
      case ImageCompressionFixedRateFlagBitsEXT::e11Bpc: return "11Bpc";
      case ImageCompressionFixedRateFlagBitsEXT::e12Bpc: return "12Bpc";
      case ImageCompressionFixedRateFlagBitsEXT::e13Bpc: return "13Bpc";
      case ImageCompressionFixedRateFlagBitsEXT::e14Bpc: return "14Bpc";
      case ImageCompressionFixedRateFlagBitsEXT::e15Bpc: return "15Bpc";
      case ImageCompressionFixedRateFlagBitsEXT::e16Bpc: return "16Bpc";
      case ImageCompressionFixedRateFlagBitsEXT::e17Bpc: return "17Bpc";
      case ImageCompressionFixedRateFlagBitsEXT::e18Bpc: return "18Bpc";
      case ImageCompressionFixedRateFlagBitsEXT::e19Bpc: return "19Bpc";
      case ImageCompressionFixedRateFlagBitsEXT::e20Bpc: return "20Bpc";
      case ImageCompressionFixedRateFlagBitsEXT::e21Bpc: return "21Bpc";
      case ImageCompressionFixedRateFlagBitsEXT::e22Bpc: return "22Bpc";
      case ImageCompressionFixedRateFlagBitsEXT::e23Bpc: return "23Bpc";
      case ImageCompressionFixedRateFlagBitsEXT::e24Bpc: return "24Bpc";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_EXT_device_fault ===

  VULKAN_HPP_INLINE std::string to_string( DeviceFaultAddressTypeEXT value )
  {
    switch ( value )
    {
      case DeviceFaultAddressTypeEXT::eNone: return "None";
      case DeviceFaultAddressTypeEXT::eReadInvalid: return "ReadInvalid";
      case DeviceFaultAddressTypeEXT::eWriteInvalid: return "WriteInvalid";
      case DeviceFaultAddressTypeEXT::eExecuteInvalid: return "ExecuteInvalid";
      case DeviceFaultAddressTypeEXT::eInstructionPointerUnknown: return "InstructionPointerUnknown";
      case DeviceFaultAddressTypeEXT::eInstructionPointerInvalid: return "InstructionPointerInvalid";
      case DeviceFaultAddressTypeEXT::eInstructionPointerFault: return "InstructionPointerFault";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( DeviceFaultVendorBinaryHeaderVersionEXT value )
  {
    switch ( value )
    {
      case DeviceFaultVendorBinaryHeaderVersionEXT::eOne: return "One";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
  //=== VK_EXT_directfb_surface ===

  VULKAN_HPP_INLINE std::string to_string( DirectFBSurfaceCreateFlagBitsEXT )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

  //=== VK_EXT_device_address_binding_report ===

  VULKAN_HPP_INLINE std::string to_string( DeviceAddressBindingFlagBitsEXT value )
  {
    switch ( value )
    {
      case DeviceAddressBindingFlagBitsEXT::eInternalObject: return "InternalObject";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( DeviceAddressBindingTypeEXT value )
  {
    switch ( value )
    {
      case DeviceAddressBindingTypeEXT::eBind: return "Bind";
      case DeviceAddressBindingTypeEXT::eUnbind: return "Unbind";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_buffer_collection ===

  VULKAN_HPP_INLINE std::string to_string( ImageConstraintsInfoFlagBitsFUCHSIA value )
  {
    switch ( value )
    {
      case ImageConstraintsInfoFlagBitsFUCHSIA::eCpuReadRarely: return "CpuReadRarely";
      case ImageConstraintsInfoFlagBitsFUCHSIA::eCpuReadOften: return "CpuReadOften";
      case ImageConstraintsInfoFlagBitsFUCHSIA::eCpuWriteRarely: return "CpuWriteRarely";
      case ImageConstraintsInfoFlagBitsFUCHSIA::eCpuWriteOften: return "CpuWriteOften";
      case ImageConstraintsInfoFlagBitsFUCHSIA::eProtectedOptional: return "ProtectedOptional";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ImageFormatConstraintsFlagBitsFUCHSIA )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_FUCHSIA*/

  //=== VK_EXT_frame_boundary ===

  VULKAN_HPP_INLINE std::string to_string( FrameBoundaryFlagBitsEXT value )
  {
    switch ( value )
    {
      case FrameBoundaryFlagBitsEXT::eFrameEnd: return "FrameEnd";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
  //=== VK_QNX_screen_surface ===

  VULKAN_HPP_INLINE std::string to_string( ScreenSurfaceCreateFlagBitsQNX )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

  //=== VK_EXT_opacity_micromap ===

  VULKAN_HPP_INLINE std::string to_string( MicromapTypeEXT value )
  {
    switch ( value )
    {
      case MicromapTypeEXT::eOpacityMicromap: return "OpacityMicromap";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case MicromapTypeEXT::eDisplacementMicromapNV: return "DisplacementMicromapNV";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( BuildMicromapFlagBitsEXT value )
  {
    switch ( value )
    {
      case BuildMicromapFlagBitsEXT::ePreferFastTrace: return "PreferFastTrace";
      case BuildMicromapFlagBitsEXT::ePreferFastBuild: return "PreferFastBuild";
      case BuildMicromapFlagBitsEXT::eAllowCompaction: return "AllowCompaction";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( CopyMicromapModeEXT value )
  {
    switch ( value )
    {
      case CopyMicromapModeEXT::eClone: return "Clone";
      case CopyMicromapModeEXT::eSerialize: return "Serialize";
      case CopyMicromapModeEXT::eDeserialize: return "Deserialize";
      case CopyMicromapModeEXT::eCompact: return "Compact";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( MicromapCreateFlagBitsEXT value )
  {
    switch ( value )
    {
      case MicromapCreateFlagBitsEXT::eDeviceAddressCaptureReplay: return "DeviceAddressCaptureReplay";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( BuildMicromapModeEXT value )
  {
    switch ( value )
    {
      case BuildMicromapModeEXT::eBuild: return "Build";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( OpacityMicromapFormatEXT value )
  {
    switch ( value )
    {
      case OpacityMicromapFormatEXT::e2State: return "2State";
      case OpacityMicromapFormatEXT::e4State: return "4State";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( OpacityMicromapSpecialIndexEXT value )
  {
    switch ( value )
    {
      case OpacityMicromapSpecialIndexEXT::eFullyTransparent: return "FullyTransparent";
      case OpacityMicromapSpecialIndexEXT::eFullyOpaque: return "FullyOpaque";
      case OpacityMicromapSpecialIndexEXT::eFullyUnknownTransparent: return "FullyUnknownTransparent";
      case OpacityMicromapSpecialIndexEXT::eFullyUnknownOpaque: return "FullyUnknownOpaque";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_NV_displacement_micromap ===

  VULKAN_HPP_INLINE std::string to_string( DisplacementMicromapFormatNV value )
  {
    switch ( value )
    {
      case DisplacementMicromapFormatNV::e64Triangles64Bytes: return "64Triangles64Bytes";
      case DisplacementMicromapFormatNV::e256Triangles128Bytes: return "256Triangles128Bytes";
      case DisplacementMicromapFormatNV::e1024Triangles128Bytes: return "1024Triangles128Bytes";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_ARM_scheduling_controls ===

  VULKAN_HPP_INLINE std::string to_string( PhysicalDeviceSchedulingControlsFlagBitsARM value )
  {
    switch ( value )
    {
      case PhysicalDeviceSchedulingControlsFlagBitsARM::eShaderCoreCount: return "ShaderCoreCount";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_NV_memory_decompression ===

  VULKAN_HPP_INLINE std::string to_string( MemoryDecompressionMethodFlagBitsNV value )
  {
    switch ( value )
    {
      case MemoryDecompressionMethodFlagBitsNV::eGdeflate10: return "Gdeflate10";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_EXT_subpass_merge_feedback ===

  VULKAN_HPP_INLINE std::string to_string( SubpassMergeStatusEXT value )
  {
    switch ( value )
    {
      case SubpassMergeStatusEXT::eMerged: return "Merged";
      case SubpassMergeStatusEXT::eDisallowed: return "Disallowed";
      case SubpassMergeStatusEXT::eNotMergedSideEffects: return "NotMergedSideEffects";
      case SubpassMergeStatusEXT::eNotMergedSamplesMismatch: return "NotMergedSamplesMismatch";
      case SubpassMergeStatusEXT::eNotMergedViewsMismatch: return "NotMergedViewsMismatch";
      case SubpassMergeStatusEXT::eNotMergedAliasing: return "NotMergedAliasing";
      case SubpassMergeStatusEXT::eNotMergedDependencies: return "NotMergedDependencies";
      case SubpassMergeStatusEXT::eNotMergedIncompatibleInputAttachment: return "NotMergedIncompatibleInputAttachment";
      case SubpassMergeStatusEXT::eNotMergedTooManyAttachments: return "NotMergedTooManyAttachments";
      case SubpassMergeStatusEXT::eNotMergedInsufficientStorage: return "NotMergedInsufficientStorage";
      case SubpassMergeStatusEXT::eNotMergedDepthStencilCount: return "NotMergedDepthStencilCount";
      case SubpassMergeStatusEXT::eNotMergedResolveAttachmentReuse: return "NotMergedResolveAttachmentReuse";
      case SubpassMergeStatusEXT::eNotMergedSingleSubpass: return "NotMergedSingleSubpass";
      case SubpassMergeStatusEXT::eNotMergedUnspecified: return "NotMergedUnspecified";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_LUNARG_direct_driver_loading ===

  VULKAN_HPP_INLINE std::string to_string( DirectDriverLoadingModeLUNARG value )
  {
    switch ( value )
    {
      case DirectDriverLoadingModeLUNARG::eExclusive: return "Exclusive";
      case DirectDriverLoadingModeLUNARG::eInclusive: return "Inclusive";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( DirectDriverLoadingFlagBitsLUNARG )
  {
    return "(void)";
  }

  //=== VK_NV_optical_flow ===

  VULKAN_HPP_INLINE std::string to_string( OpticalFlowUsageFlagBitsNV value )
  {
    switch ( value )
    {
      case OpticalFlowUsageFlagBitsNV::eUnknown: return "Unknown";
      case OpticalFlowUsageFlagBitsNV::eInput: return "Input";
      case OpticalFlowUsageFlagBitsNV::eOutput: return "Output";
      case OpticalFlowUsageFlagBitsNV::eHint: return "Hint";
      case OpticalFlowUsageFlagBitsNV::eCost: return "Cost";
      case OpticalFlowUsageFlagBitsNV::eGlobalFlow: return "GlobalFlow";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( OpticalFlowGridSizeFlagBitsNV value )
  {
    switch ( value )
    {
      case OpticalFlowGridSizeFlagBitsNV::eUnknown: return "Unknown";
      case OpticalFlowGridSizeFlagBitsNV::e1X1: return "1X1";
      case OpticalFlowGridSizeFlagBitsNV::e2X2: return "2X2";
      case OpticalFlowGridSizeFlagBitsNV::e4X4: return "4X4";
      case OpticalFlowGridSizeFlagBitsNV::e8X8: return "8X8";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( OpticalFlowPerformanceLevelNV value )
  {
    switch ( value )
    {
      case OpticalFlowPerformanceLevelNV::eUnknown: return "Unknown";
      case OpticalFlowPerformanceLevelNV::eSlow: return "Slow";
      case OpticalFlowPerformanceLevelNV::eMedium: return "Medium";
      case OpticalFlowPerformanceLevelNV::eFast: return "Fast";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( OpticalFlowSessionBindingPointNV value )
  {
    switch ( value )
    {
      case OpticalFlowSessionBindingPointNV::eUnknown: return "Unknown";
      case OpticalFlowSessionBindingPointNV::eInput: return "Input";
      case OpticalFlowSessionBindingPointNV::eReference: return "Reference";
      case OpticalFlowSessionBindingPointNV::eHint: return "Hint";
      case OpticalFlowSessionBindingPointNV::eFlowVector: return "FlowVector";
      case OpticalFlowSessionBindingPointNV::eBackwardFlowVector: return "BackwardFlowVector";
      case OpticalFlowSessionBindingPointNV::eCost: return "Cost";
      case OpticalFlowSessionBindingPointNV::eBackwardCost: return "BackwardCost";
      case OpticalFlowSessionBindingPointNV::eGlobalFlow: return "GlobalFlow";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( OpticalFlowSessionCreateFlagBitsNV value )
  {
    switch ( value )
    {
      case OpticalFlowSessionCreateFlagBitsNV::eEnableHint: return "EnableHint";
      case OpticalFlowSessionCreateFlagBitsNV::eEnableCost: return "EnableCost";
      case OpticalFlowSessionCreateFlagBitsNV::eEnableGlobalFlow: return "EnableGlobalFlow";
      case OpticalFlowSessionCreateFlagBitsNV::eAllowRegions: return "AllowRegions";
      case OpticalFlowSessionCreateFlagBitsNV::eBothDirections: return "BothDirections";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( OpticalFlowExecuteFlagBitsNV value )
  {
    switch ( value )
    {
      case OpticalFlowExecuteFlagBitsNV::eDisableTemporalHints: return "DisableTemporalHints";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_KHR_maintenance5 ===

  VULKAN_HPP_INLINE std::string to_string( PipelineCreateFlagBits2KHR value )
  {
    switch ( value )
    {
      case PipelineCreateFlagBits2KHR::eDisableOptimization: return "DisableOptimization";
      case PipelineCreateFlagBits2KHR::eAllowDerivatives: return "AllowDerivatives";
      case PipelineCreateFlagBits2KHR::eDerivative: return "Derivative";
      case PipelineCreateFlagBits2KHR::eEnableLegacyDitheringEXT: return "EnableLegacyDitheringEXT";
      case PipelineCreateFlagBits2KHR::eViewIndexFromDeviceIndex: return "ViewIndexFromDeviceIndex";
      case PipelineCreateFlagBits2KHR::eDispatchBase: return "DispatchBase";
      case PipelineCreateFlagBits2KHR::eDeferCompileNV: return "DeferCompileNV";
      case PipelineCreateFlagBits2KHR::eCaptureStatistics: return "CaptureStatistics";
      case PipelineCreateFlagBits2KHR::eCaptureInternalRepresentations: return "CaptureInternalRepresentations";
      case PipelineCreateFlagBits2KHR::eFailOnPipelineCompileRequired: return "FailOnPipelineCompileRequired";
      case PipelineCreateFlagBits2KHR::eEarlyReturnOnFailure: return "EarlyReturnOnFailure";
      case PipelineCreateFlagBits2KHR::eLinkTimeOptimizationEXT: return "LinkTimeOptimizationEXT";
      case PipelineCreateFlagBits2KHR::eRetainLinkTimeOptimizationInfoEXT: return "RetainLinkTimeOptimizationInfoEXT";
      case PipelineCreateFlagBits2KHR::eLibrary: return "Library";
      case PipelineCreateFlagBits2KHR::eRayTracingSkipTriangles: return "RayTracingSkipTriangles";
      case PipelineCreateFlagBits2KHR::eRayTracingSkipAabbs: return "RayTracingSkipAabbs";
      case PipelineCreateFlagBits2KHR::eRayTracingNoNullAnyHitShaders: return "RayTracingNoNullAnyHitShaders";
      case PipelineCreateFlagBits2KHR::eRayTracingNoNullClosestHitShaders: return "RayTracingNoNullClosestHitShaders";
      case PipelineCreateFlagBits2KHR::eRayTracingNoNullMissShaders: return "RayTracingNoNullMissShaders";
      case PipelineCreateFlagBits2KHR::eRayTracingNoNullIntersectionShaders: return "RayTracingNoNullIntersectionShaders";
      case PipelineCreateFlagBits2KHR::eRayTracingShaderGroupHandleCaptureReplay: return "RayTracingShaderGroupHandleCaptureReplay";
      case PipelineCreateFlagBits2KHR::eIndirectBindableNV: return "IndirectBindableNV";
      case PipelineCreateFlagBits2KHR::eRayTracingAllowMotionNV: return "RayTracingAllowMotionNV";
      case PipelineCreateFlagBits2KHR::eRenderingFragmentShadingRateAttachment: return "RenderingFragmentShadingRateAttachment";
      case PipelineCreateFlagBits2KHR::eRenderingFragmentDensityMapAttachmentEXT: return "RenderingFragmentDensityMapAttachmentEXT";
      case PipelineCreateFlagBits2KHR::eRayTracingOpacityMicromapEXT: return "RayTracingOpacityMicromapEXT";
      case PipelineCreateFlagBits2KHR::eColorAttachmentFeedbackLoopEXT: return "ColorAttachmentFeedbackLoopEXT";
      case PipelineCreateFlagBits2KHR::eDepthStencilAttachmentFeedbackLoopEXT: return "DepthStencilAttachmentFeedbackLoopEXT";
      case PipelineCreateFlagBits2KHR::eNoProtectedAccessEXT: return "NoProtectedAccessEXT";
      case PipelineCreateFlagBits2KHR::eProtectedAccessOnlyEXT: return "ProtectedAccessOnlyEXT";
      case PipelineCreateFlagBits2KHR::eRayTracingDisplacementMicromapNV: return "RayTracingDisplacementMicromapNV";
      case PipelineCreateFlagBits2KHR::eDescriptorBufferEXT: return "DescriptorBufferEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( BufferUsageFlagBits2KHR value )
  {
    switch ( value )
    {
      case BufferUsageFlagBits2KHR::eTransferSrc: return "TransferSrc";
      case BufferUsageFlagBits2KHR::eTransferDst: return "TransferDst";
      case BufferUsageFlagBits2KHR::eUniformTexelBuffer: return "UniformTexelBuffer";
      case BufferUsageFlagBits2KHR::eStorageTexelBuffer: return "StorageTexelBuffer";
      case BufferUsageFlagBits2KHR::eUniformBuffer: return "UniformBuffer";
      case BufferUsageFlagBits2KHR::eStorageBuffer: return "StorageBuffer";
      case BufferUsageFlagBits2KHR::eIndexBuffer: return "IndexBuffer";
      case BufferUsageFlagBits2KHR::eVertexBuffer: return "VertexBuffer";
      case BufferUsageFlagBits2KHR::eIndirectBuffer: return "IndirectBuffer";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case BufferUsageFlagBits2KHR::eExecutionGraphScratchAMDX: return "ExecutionGraphScratchAMDX";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      case BufferUsageFlagBits2KHR::eConditionalRenderingEXT: return "ConditionalRenderingEXT";
      case BufferUsageFlagBits2KHR::eShaderBindingTable: return "ShaderBindingTable";
      case BufferUsageFlagBits2KHR::eTransformFeedbackBufferEXT: return "TransformFeedbackBufferEXT";
      case BufferUsageFlagBits2KHR::eTransformFeedbackCounterBufferEXT: return "TransformFeedbackCounterBufferEXT";
      case BufferUsageFlagBits2KHR::eVideoDecodeSrc: return "VideoDecodeSrc";
      case BufferUsageFlagBits2KHR::eVideoDecodeDst: return "VideoDecodeDst";
      case BufferUsageFlagBits2KHR::eVideoEncodeDst: return "VideoEncodeDst";
      case BufferUsageFlagBits2KHR::eVideoEncodeSrc: return "VideoEncodeSrc";
      case BufferUsageFlagBits2KHR::eShaderDeviceAddress: return "ShaderDeviceAddress";
      case BufferUsageFlagBits2KHR::eAccelerationStructureBuildInputReadOnly: return "AccelerationStructureBuildInputReadOnly";
      case BufferUsageFlagBits2KHR::eAccelerationStructureStorage: return "AccelerationStructureStorage";
      case BufferUsageFlagBits2KHR::eSamplerDescriptorBufferEXT: return "SamplerDescriptorBufferEXT";
      case BufferUsageFlagBits2KHR::eResourceDescriptorBufferEXT: return "ResourceDescriptorBufferEXT";
      case BufferUsageFlagBits2KHR::ePushDescriptorsDescriptorBufferEXT: return "PushDescriptorsDescriptorBufferEXT";
      case BufferUsageFlagBits2KHR::eMicromapBuildInputReadOnlyEXT: return "MicromapBuildInputReadOnlyEXT";
      case BufferUsageFlagBits2KHR::eMicromapStorageEXT: return "MicromapStorageEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_EXT_shader_object ===

  VULKAN_HPP_INLINE std::string to_string( ShaderCreateFlagBitsEXT value )
  {
    switch ( value )
    {
      case ShaderCreateFlagBitsEXT::eLinkStage: return "LinkStage";
      case ShaderCreateFlagBitsEXT::eAllowVaryingSubgroupSize: return "AllowVaryingSubgroupSize";
      case ShaderCreateFlagBitsEXT::eRequireFullSubgroups: return "RequireFullSubgroups";
      case ShaderCreateFlagBitsEXT::eNoTaskShader: return "NoTaskShader";
      case ShaderCreateFlagBitsEXT::eDispatchBase: return "DispatchBase";
      case ShaderCreateFlagBitsEXT::eFragmentShadingRateAttachment: return "FragmentShadingRateAttachment";
      case ShaderCreateFlagBitsEXT::eFragmentDensityMapAttachment: return "FragmentDensityMapAttachment";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ShaderCodeTypeEXT value )
  {
    switch ( value )
    {
      case ShaderCodeTypeEXT::eBinary: return "Binary";
      case ShaderCodeTypeEXT::eSpirv: return "Spirv";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_NV_ray_tracing_invocation_reorder ===

  VULKAN_HPP_INLINE std::string to_string( RayTracingInvocationReorderModeNV value )
  {
    switch ( value )
    {
      case RayTracingInvocationReorderModeNV::eNone: return "None";
      case RayTracingInvocationReorderModeNV::eReorder: return "Reorder";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_EXT_layer_settings ===

  VULKAN_HPP_INLINE std::string to_string( LayerSettingTypeEXT value )
  {
    switch ( value )
    {
      case LayerSettingTypeEXT::eBool32: return "Bool32";
      case LayerSettingTypeEXT::eInt32: return "Int32";
      case LayerSettingTypeEXT::eInt64: return "Int64";
      case LayerSettingTypeEXT::eUint32: return "Uint32";
      case LayerSettingTypeEXT::eUint64: return "Uint64";
      case LayerSettingTypeEXT::eFloat32: return "Float32";
      case LayerSettingTypeEXT::eFloat64: return "Float64";
      case LayerSettingTypeEXT::eString: return "String";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_NV_low_latency2 ===

  VULKAN_HPP_INLINE std::string to_string( LatencyMarkerNV value )
  {
    switch ( value )
    {
      case LatencyMarkerNV::eSimulationStart: return "SimulationStart";
      case LatencyMarkerNV::eSimulationEnd: return "SimulationEnd";
      case LatencyMarkerNV::eRendersubmitStart: return "RendersubmitStart";
      case LatencyMarkerNV::eRendersubmitEnd: return "RendersubmitEnd";
      case LatencyMarkerNV::ePresentStart: return "PresentStart";
      case LatencyMarkerNV::ePresentEnd: return "PresentEnd";
      case LatencyMarkerNV::eInputSample: return "InputSample";
      case LatencyMarkerNV::eTriggerFlash: return "TriggerFlash";
      case LatencyMarkerNV::eOutOfBandRendersubmitStart: return "OutOfBandRendersubmitStart";
      case LatencyMarkerNV::eOutOfBandRendersubmitEnd: return "OutOfBandRendersubmitEnd";
      case LatencyMarkerNV::eOutOfBandPresentStart: return "OutOfBandPresentStart";
      case LatencyMarkerNV::eOutOfBandPresentEnd: return "OutOfBandPresentEnd";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( OutOfBandQueueTypeNV value )
  {
    switch ( value )
    {
      case OutOfBandQueueTypeNV::eRender: return "Render";
      case OutOfBandQueueTypeNV::ePresent: return "Present";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_KHR_cooperative_matrix ===

  VULKAN_HPP_INLINE std::string to_string( ScopeKHR value )
  {
    switch ( value )
    {
      case ScopeKHR::eDevice: return "Device";
      case ScopeKHR::eWorkgroup: return "Workgroup";
      case ScopeKHR::eSubgroup: return "Subgroup";
      case ScopeKHR::eQueueFamily: return "QueueFamily";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  VULKAN_HPP_INLINE std::string to_string( ComponentTypeKHR value )
  {
    switch ( value )
    {
      case ComponentTypeKHR::eFloat16: return "Float16";
      case ComponentTypeKHR::eFloat32: return "Float32";
      case ComponentTypeKHR::eFloat64: return "Float64";
      case ComponentTypeKHR::eSint8: return "Sint8";
      case ComponentTypeKHR::eSint16: return "Sint16";
      case ComponentTypeKHR::eSint32: return "Sint32";
      case ComponentTypeKHR::eSint64: return "Sint64";
      case ComponentTypeKHR::eUint8: return "Uint8";
      case ComponentTypeKHR::eUint16: return "Uint16";
      case ComponentTypeKHR::eUint32: return "Uint32";
      case ComponentTypeKHR::eUint64: return "Uint64";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_QCOM_image_processing2 ===

  VULKAN_HPP_INLINE std::string to_string( BlockMatchWindowCompareModeQCOM value )
  {
    switch ( value )
    {
      case BlockMatchWindowCompareModeQCOM::eMin: return "Min";
      case BlockMatchWindowCompareModeQCOM::eMax: return "Max";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_QCOM_filter_cubic_weights ===

  VULKAN_HPP_INLINE std::string to_string( CubicFilterWeightsQCOM value )
  {
    switch ( value )
    {
      case CubicFilterWeightsQCOM::eCatmullRom: return "CatmullRom";
      case CubicFilterWeightsQCOM::eZeroTangentCardinal: return "ZeroTangentCardinal";
      case CubicFilterWeightsQCOM::eBSpline: return "BSpline";
      case CubicFilterWeightsQCOM::eMitchellNetravali: return "MitchellNetravali";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_MSFT_layered_driver ===

  VULKAN_HPP_INLINE std::string to_string( LayeredDriverUnderlyingApiMSFT value )
  {
    switch ( value )
    {
      case LayeredDriverUnderlyingApiMSFT::eNone: return "None";
      case LayeredDriverUnderlyingApiMSFT::eD3D12: return "D3D12";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_KHR_line_rasterization ===

  VULKAN_HPP_INLINE std::string to_string( LineRasterizationModeKHR value )
  {
    switch ( value )
    {
      case LineRasterizationModeKHR::eDefault: return "Default";
      case LineRasterizationModeKHR::eRectangular: return "Rectangular";
      case LineRasterizationModeKHR::eBresenham: return "Bresenham";
      case LineRasterizationModeKHR::eRectangularSmooth: return "RectangularSmooth";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_KHR_calibrated_timestamps ===

  VULKAN_HPP_INLINE std::string to_string( TimeDomainKHR value )
  {
    switch ( value )
    {
      case TimeDomainKHR::eDevice: return "Device";
      case TimeDomainKHR::eClockMonotonic: return "ClockMonotonic";
      case TimeDomainKHR::eClockMonotonicRaw: return "ClockMonotonicRaw";
      case TimeDomainKHR::eQueryPerformanceCounter: return "QueryPerformanceCounter";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

}  // namespace VULKAN_HPP_NAMESPACE
#endif
