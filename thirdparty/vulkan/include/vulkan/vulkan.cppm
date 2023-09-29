// Copyright 2015-2023 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//

// This header is generated from the Khronos Vulkan XML API Registry.

module;

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_extension_inspection.hpp>
#include <vulkan/vulkan_format_traits.hpp>
#include <vulkan/vulkan_hash.hpp>
#include <vulkan/vulkan_raii.hpp>

export module vulkan;

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

export namespace VULKAN_HPP_NAMESPACE
{
  //=====================================
  //=== HARDCODED TYPEs AND FUNCTIONs ===
  //=====================================
  using VULKAN_HPP_NAMESPACE::ArrayWrapper1D;
  using VULKAN_HPP_NAMESPACE::ArrayWrapper2D;
  using VULKAN_HPP_NAMESPACE::DispatchLoaderBase;
  using VULKAN_HPP_NAMESPACE::Flags;
  using VULKAN_HPP_NAMESPACE::FlagTraits;

#if !defined( VK_NO_PROTOTYPES )
  using VULKAN_HPP_NAMESPACE::DispatchLoaderStatic;
#endif /*VK_NO_PROTOTYPES*/

  using VULKAN_HPP_NAMESPACE::operator&;
  using VULKAN_HPP_NAMESPACE::operator|;
  using VULKAN_HPP_NAMESPACE::operator^;
  using VULKAN_HPP_NAMESPACE::operator~;
  using VULKAN_HPP_DEFAULT_DISPATCHER_TYPE;

#if !defined( VULKAN_HPP_DISABLE_ENHANCED_MODE )
  using VULKAN_HPP_NAMESPACE::ArrayProxy;
  using VULKAN_HPP_NAMESPACE::ArrayProxyNoTemporaries;
  using VULKAN_HPP_NAMESPACE::Optional;
  using VULKAN_HPP_NAMESPACE::StridedArrayProxy;
  using VULKAN_HPP_NAMESPACE::StructureChain;
  using VULKAN_HPP_NAMESPACE::UniqueHandle;
#endif /*VULKAN_HPP_DISABLE_ENHANCED_MODE*/

#if !defined( VULKAN_HPP_NO_SMART_HANDLE )
  using VULKAN_HPP_NAMESPACE::ObjectDestroy;
  using VULKAN_HPP_NAMESPACE::ObjectFree;
  using VULKAN_HPP_NAMESPACE::ObjectRelease;
  using VULKAN_HPP_NAMESPACE::PoolFree;
#endif /*VULKAN_HPP_NO_SMART_HANDLE*/

  //==================
  //=== BASE TYPEs ===
  //==================
  using VULKAN_HPP_NAMESPACE::Bool32;
  using VULKAN_HPP_NAMESPACE::DeviceAddress;
  using VULKAN_HPP_NAMESPACE::DeviceSize;
  using VULKAN_HPP_NAMESPACE::RemoteAddressNV;
  using VULKAN_HPP_NAMESPACE::SampleMask;

  //=============
  //=== ENUMs ===
  //=============
  using VULKAN_HPP_NAMESPACE::CppType;

  //=== VK_VERSION_1_0 ===
  using VULKAN_HPP_NAMESPACE::AccessFlagBits;
  using VULKAN_HPP_NAMESPACE::AccessFlags;
  using VULKAN_HPP_NAMESPACE::AttachmentDescriptionFlagBits;
  using VULKAN_HPP_NAMESPACE::AttachmentDescriptionFlags;
  using VULKAN_HPP_NAMESPACE::AttachmentLoadOp;
  using VULKAN_HPP_NAMESPACE::AttachmentStoreOp;
  using VULKAN_HPP_NAMESPACE::BlendFactor;
  using VULKAN_HPP_NAMESPACE::BlendOp;
  using VULKAN_HPP_NAMESPACE::BorderColor;
  using VULKAN_HPP_NAMESPACE::BufferCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::BufferCreateFlags;
  using VULKAN_HPP_NAMESPACE::BufferUsageFlagBits;
  using VULKAN_HPP_NAMESPACE::BufferUsageFlags;
  using VULKAN_HPP_NAMESPACE::BufferViewCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::BufferViewCreateFlags;
  using VULKAN_HPP_NAMESPACE::ColorComponentFlagBits;
  using VULKAN_HPP_NAMESPACE::ColorComponentFlags;
  using VULKAN_HPP_NAMESPACE::CommandBufferLevel;
  using VULKAN_HPP_NAMESPACE::CommandBufferResetFlagBits;
  using VULKAN_HPP_NAMESPACE::CommandBufferResetFlags;
  using VULKAN_HPP_NAMESPACE::CommandBufferUsageFlagBits;
  using VULKAN_HPP_NAMESPACE::CommandBufferUsageFlags;
  using VULKAN_HPP_NAMESPACE::CommandPoolCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::CommandPoolCreateFlags;
  using VULKAN_HPP_NAMESPACE::CommandPoolResetFlagBits;
  using VULKAN_HPP_NAMESPACE::CommandPoolResetFlags;
  using VULKAN_HPP_NAMESPACE::CompareOp;
  using VULKAN_HPP_NAMESPACE::ComponentSwizzle;
  using VULKAN_HPP_NAMESPACE::CullModeFlagBits;
  using VULKAN_HPP_NAMESPACE::CullModeFlags;
  using VULKAN_HPP_NAMESPACE::DependencyFlagBits;
  using VULKAN_HPP_NAMESPACE::DependencyFlags;
  using VULKAN_HPP_NAMESPACE::DescriptorPoolCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::DescriptorPoolCreateFlags;
  using VULKAN_HPP_NAMESPACE::DescriptorPoolResetFlagBits;
  using VULKAN_HPP_NAMESPACE::DescriptorPoolResetFlags;
  using VULKAN_HPP_NAMESPACE::DescriptorSetLayoutCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::DescriptorSetLayoutCreateFlags;
  using VULKAN_HPP_NAMESPACE::DescriptorType;
  using VULKAN_HPP_NAMESPACE::DeviceCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::DeviceCreateFlags;
  using VULKAN_HPP_NAMESPACE::DynamicState;
  using VULKAN_HPP_NAMESPACE::EventCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::EventCreateFlags;
  using VULKAN_HPP_NAMESPACE::FenceCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::FenceCreateFlags;
  using VULKAN_HPP_NAMESPACE::Filter;
  using VULKAN_HPP_NAMESPACE::Format;
  using VULKAN_HPP_NAMESPACE::FormatFeatureFlagBits;
  using VULKAN_HPP_NAMESPACE::FormatFeatureFlags;
  using VULKAN_HPP_NAMESPACE::FramebufferCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::FramebufferCreateFlags;
  using VULKAN_HPP_NAMESPACE::FrontFace;
  using VULKAN_HPP_NAMESPACE::ImageAspectFlagBits;
  using VULKAN_HPP_NAMESPACE::ImageAspectFlags;
  using VULKAN_HPP_NAMESPACE::ImageCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::ImageCreateFlags;
  using VULKAN_HPP_NAMESPACE::ImageLayout;
  using VULKAN_HPP_NAMESPACE::ImageTiling;
  using VULKAN_HPP_NAMESPACE::ImageType;
  using VULKAN_HPP_NAMESPACE::ImageUsageFlagBits;
  using VULKAN_HPP_NAMESPACE::ImageUsageFlags;
  using VULKAN_HPP_NAMESPACE::ImageViewCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::ImageViewCreateFlags;
  using VULKAN_HPP_NAMESPACE::ImageViewType;
  using VULKAN_HPP_NAMESPACE::IndexType;
  using VULKAN_HPP_NAMESPACE::InstanceCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::InstanceCreateFlags;
  using VULKAN_HPP_NAMESPACE::InternalAllocationType;
  using VULKAN_HPP_NAMESPACE::LogicOp;
  using VULKAN_HPP_NAMESPACE::MemoryHeapFlagBits;
  using VULKAN_HPP_NAMESPACE::MemoryHeapFlags;
  using VULKAN_HPP_NAMESPACE::MemoryMapFlagBits;
  using VULKAN_HPP_NAMESPACE::MemoryMapFlags;
  using VULKAN_HPP_NAMESPACE::MemoryPropertyFlagBits;
  using VULKAN_HPP_NAMESPACE::MemoryPropertyFlags;
  using VULKAN_HPP_NAMESPACE::ObjectType;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceType;
  using VULKAN_HPP_NAMESPACE::PipelineBindPoint;
  using VULKAN_HPP_NAMESPACE::PipelineCacheHeaderVersion;
  using VULKAN_HPP_NAMESPACE::PipelineCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::PipelineCreateFlags;
  using VULKAN_HPP_NAMESPACE::PipelineDynamicStateCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::PipelineDynamicStateCreateFlags;
  using VULKAN_HPP_NAMESPACE::PipelineInputAssemblyStateCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::PipelineInputAssemblyStateCreateFlags;
  using VULKAN_HPP_NAMESPACE::PipelineMultisampleStateCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::PipelineMultisampleStateCreateFlags;
  using VULKAN_HPP_NAMESPACE::PipelineRasterizationStateCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::PipelineRasterizationStateCreateFlags;
  using VULKAN_HPP_NAMESPACE::PipelineShaderStageCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::PipelineShaderStageCreateFlags;
  using VULKAN_HPP_NAMESPACE::PipelineStageFlagBits;
  using VULKAN_HPP_NAMESPACE::PipelineStageFlags;
  using VULKAN_HPP_NAMESPACE::PipelineTessellationStateCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::PipelineTessellationStateCreateFlags;
  using VULKAN_HPP_NAMESPACE::PipelineVertexInputStateCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::PipelineVertexInputStateCreateFlags;
  using VULKAN_HPP_NAMESPACE::PipelineViewportStateCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::PipelineViewportStateCreateFlags;
  using VULKAN_HPP_NAMESPACE::PolygonMode;
  using VULKAN_HPP_NAMESPACE::PrimitiveTopology;
  using VULKAN_HPP_NAMESPACE::QueryControlFlagBits;
  using VULKAN_HPP_NAMESPACE::QueryControlFlags;
  using VULKAN_HPP_NAMESPACE::QueryPipelineStatisticFlagBits;
  using VULKAN_HPP_NAMESPACE::QueryPipelineStatisticFlags;
  using VULKAN_HPP_NAMESPACE::QueryPoolCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::QueryPoolCreateFlags;
  using VULKAN_HPP_NAMESPACE::QueryResultFlagBits;
  using VULKAN_HPP_NAMESPACE::QueryResultFlags;
  using VULKAN_HPP_NAMESPACE::QueryType;
  using VULKAN_HPP_NAMESPACE::QueueFlagBits;
  using VULKAN_HPP_NAMESPACE::QueueFlags;
  using VULKAN_HPP_NAMESPACE::RenderPassCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::RenderPassCreateFlags;
  using VULKAN_HPP_NAMESPACE::Result;
  using VULKAN_HPP_NAMESPACE::SampleCountFlagBits;
  using VULKAN_HPP_NAMESPACE::SampleCountFlags;
  using VULKAN_HPP_NAMESPACE::SamplerAddressMode;
  using VULKAN_HPP_NAMESPACE::SamplerCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::SamplerCreateFlags;
  using VULKAN_HPP_NAMESPACE::SamplerMipmapMode;
  using VULKAN_HPP_NAMESPACE::SemaphoreCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::SemaphoreCreateFlags;
  using VULKAN_HPP_NAMESPACE::ShaderModuleCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::ShaderModuleCreateFlags;
  using VULKAN_HPP_NAMESPACE::ShaderStageFlagBits;
  using VULKAN_HPP_NAMESPACE::ShaderStageFlags;
  using VULKAN_HPP_NAMESPACE::SharingMode;
  using VULKAN_HPP_NAMESPACE::SparseImageFormatFlagBits;
  using VULKAN_HPP_NAMESPACE::SparseImageFormatFlags;
  using VULKAN_HPP_NAMESPACE::SparseMemoryBindFlagBits;
  using VULKAN_HPP_NAMESPACE::SparseMemoryBindFlags;
  using VULKAN_HPP_NAMESPACE::StencilFaceFlagBits;
  using VULKAN_HPP_NAMESPACE::StencilFaceFlags;
  using VULKAN_HPP_NAMESPACE::StencilOp;
  using VULKAN_HPP_NAMESPACE::StructureType;
  using VULKAN_HPP_NAMESPACE::SubpassContents;
  using VULKAN_HPP_NAMESPACE::SubpassDescriptionFlagBits;
  using VULKAN_HPP_NAMESPACE::SubpassDescriptionFlags;
  using VULKAN_HPP_NAMESPACE::SystemAllocationScope;
  using VULKAN_HPP_NAMESPACE::VendorId;
  using VULKAN_HPP_NAMESPACE::VertexInputRate;

  //=== VK_VERSION_1_1 ===
  using VULKAN_HPP_NAMESPACE::ChromaLocation;
  using VULKAN_HPP_NAMESPACE::ChromaLocationKHR;
  using VULKAN_HPP_NAMESPACE::CommandPoolTrimFlagBits;
  using VULKAN_HPP_NAMESPACE::CommandPoolTrimFlags;
  using VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateCreateFlags;
  using VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateType;
  using VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateTypeKHR;
  using VULKAN_HPP_NAMESPACE::DeviceQueueCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::DeviceQueueCreateFlags;
  using VULKAN_HPP_NAMESPACE::ExternalFenceFeatureFlagBits;
  using VULKAN_HPP_NAMESPACE::ExternalFenceFeatureFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::ExternalFenceFeatureFlags;
  using VULKAN_HPP_NAMESPACE::ExternalFenceHandleTypeFlagBits;
  using VULKAN_HPP_NAMESPACE::ExternalFenceHandleTypeFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::ExternalFenceHandleTypeFlags;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryFeatureFlagBits;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryFeatureFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryFeatureFlags;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryHandleTypeFlagBits;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryHandleTypeFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryHandleTypeFlags;
  using VULKAN_HPP_NAMESPACE::ExternalSemaphoreFeatureFlagBits;
  using VULKAN_HPP_NAMESPACE::ExternalSemaphoreFeatureFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::ExternalSemaphoreFeatureFlags;
  using VULKAN_HPP_NAMESPACE::ExternalSemaphoreHandleTypeFlagBits;
  using VULKAN_HPP_NAMESPACE::ExternalSemaphoreHandleTypeFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::ExternalSemaphoreHandleTypeFlags;
  using VULKAN_HPP_NAMESPACE::FenceImportFlagBits;
  using VULKAN_HPP_NAMESPACE::FenceImportFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::FenceImportFlags;
  using VULKAN_HPP_NAMESPACE::MemoryAllocateFlagBits;
  using VULKAN_HPP_NAMESPACE::MemoryAllocateFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::MemoryAllocateFlags;
  using VULKAN_HPP_NAMESPACE::PeerMemoryFeatureFlagBits;
  using VULKAN_HPP_NAMESPACE::PeerMemoryFeatureFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::PeerMemoryFeatureFlags;
  using VULKAN_HPP_NAMESPACE::PointClippingBehavior;
  using VULKAN_HPP_NAMESPACE::PointClippingBehaviorKHR;
  using VULKAN_HPP_NAMESPACE::SamplerYcbcrModelConversion;
  using VULKAN_HPP_NAMESPACE::SamplerYcbcrModelConversionKHR;
  using VULKAN_HPP_NAMESPACE::SamplerYcbcrRange;
  using VULKAN_HPP_NAMESPACE::SamplerYcbcrRangeKHR;
  using VULKAN_HPP_NAMESPACE::SemaphoreImportFlagBits;
  using VULKAN_HPP_NAMESPACE::SemaphoreImportFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::SemaphoreImportFlags;
  using VULKAN_HPP_NAMESPACE::SubgroupFeatureFlagBits;
  using VULKAN_HPP_NAMESPACE::SubgroupFeatureFlags;
  using VULKAN_HPP_NAMESPACE::TessellationDomainOrigin;
  using VULKAN_HPP_NAMESPACE::TessellationDomainOriginKHR;

  //=== VK_VERSION_1_2 ===
  using VULKAN_HPP_NAMESPACE::DescriptorBindingFlagBits;
  using VULKAN_HPP_NAMESPACE::DescriptorBindingFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::DescriptorBindingFlags;
  using VULKAN_HPP_NAMESPACE::DriverId;
  using VULKAN_HPP_NAMESPACE::DriverIdKHR;
  using VULKAN_HPP_NAMESPACE::ResolveModeFlagBits;
  using VULKAN_HPP_NAMESPACE::ResolveModeFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::ResolveModeFlags;
  using VULKAN_HPP_NAMESPACE::SamplerReductionMode;
  using VULKAN_HPP_NAMESPACE::SamplerReductionModeEXT;
  using VULKAN_HPP_NAMESPACE::SemaphoreType;
  using VULKAN_HPP_NAMESPACE::SemaphoreTypeKHR;
  using VULKAN_HPP_NAMESPACE::SemaphoreWaitFlagBits;
  using VULKAN_HPP_NAMESPACE::SemaphoreWaitFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::SemaphoreWaitFlags;
  using VULKAN_HPP_NAMESPACE::ShaderFloatControlsIndependence;
  using VULKAN_HPP_NAMESPACE::ShaderFloatControlsIndependenceKHR;

  //=== VK_VERSION_1_3 ===
  using VULKAN_HPP_NAMESPACE::AccessFlagBits2;
  using VULKAN_HPP_NAMESPACE::AccessFlagBits2KHR;
  using VULKAN_HPP_NAMESPACE::AccessFlags2;
  using VULKAN_HPP_NAMESPACE::FormatFeatureFlagBits2;
  using VULKAN_HPP_NAMESPACE::FormatFeatureFlagBits2KHR;
  using VULKAN_HPP_NAMESPACE::FormatFeatureFlags2;
  using VULKAN_HPP_NAMESPACE::PipelineCreationFeedbackFlagBits;
  using VULKAN_HPP_NAMESPACE::PipelineCreationFeedbackFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::PipelineCreationFeedbackFlags;
  using VULKAN_HPP_NAMESPACE::PipelineStageFlagBits2;
  using VULKAN_HPP_NAMESPACE::PipelineStageFlagBits2KHR;
  using VULKAN_HPP_NAMESPACE::PipelineStageFlags2;
  using VULKAN_HPP_NAMESPACE::PrivateDataSlotCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::PrivateDataSlotCreateFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::PrivateDataSlotCreateFlags;
  using VULKAN_HPP_NAMESPACE::RenderingFlagBits;
  using VULKAN_HPP_NAMESPACE::RenderingFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::RenderingFlags;
  using VULKAN_HPP_NAMESPACE::SubmitFlagBits;
  using VULKAN_HPP_NAMESPACE::SubmitFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::SubmitFlags;
  using VULKAN_HPP_NAMESPACE::ToolPurposeFlagBits;
  using VULKAN_HPP_NAMESPACE::ToolPurposeFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::ToolPurposeFlags;

  //=== VK_KHR_surface ===
  using VULKAN_HPP_NAMESPACE::ColorSpaceKHR;
  using VULKAN_HPP_NAMESPACE::CompositeAlphaFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::CompositeAlphaFlagsKHR;
  using VULKAN_HPP_NAMESPACE::PresentModeKHR;
  using VULKAN_HPP_NAMESPACE::SurfaceTransformFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::SurfaceTransformFlagsKHR;

  //=== VK_KHR_swapchain ===
  using VULKAN_HPP_NAMESPACE::DeviceGroupPresentModeFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::DeviceGroupPresentModeFlagsKHR;
  using VULKAN_HPP_NAMESPACE::SwapchainCreateFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::SwapchainCreateFlagsKHR;

  //=== VK_KHR_display ===
  using VULKAN_HPP_NAMESPACE::DisplayModeCreateFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::DisplayModeCreateFlagsKHR;
  using VULKAN_HPP_NAMESPACE::DisplayPlaneAlphaFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::DisplayPlaneAlphaFlagsKHR;
  using VULKAN_HPP_NAMESPACE::DisplaySurfaceCreateFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::DisplaySurfaceCreateFlagsKHR;

#if defined( VK_USE_PLATFORM_XLIB_KHR )
  //=== VK_KHR_xlib_surface ===
  using VULKAN_HPP_NAMESPACE::XlibSurfaceCreateFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::XlibSurfaceCreateFlagsKHR;
#endif /*VK_USE_PLATFORM_XLIB_KHR*/

#if defined( VK_USE_PLATFORM_XCB_KHR )
  //=== VK_KHR_xcb_surface ===
  using VULKAN_HPP_NAMESPACE::XcbSurfaceCreateFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::XcbSurfaceCreateFlagsKHR;
#endif /*VK_USE_PLATFORM_XCB_KHR*/

#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
  //=== VK_KHR_wayland_surface ===
  using VULKAN_HPP_NAMESPACE::WaylandSurfaceCreateFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::WaylandSurfaceCreateFlagsKHR;
#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  //=== VK_KHR_android_surface ===
  using VULKAN_HPP_NAMESPACE::AndroidSurfaceCreateFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::AndroidSurfaceCreateFlagsKHR;
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_win32_surface ===
  using VULKAN_HPP_NAMESPACE::Win32SurfaceCreateFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::Win32SurfaceCreateFlagsKHR;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_EXT_debug_report ===
  using VULKAN_HPP_NAMESPACE::DebugReportFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::DebugReportFlagsEXT;
  using VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT;

  //=== VK_AMD_rasterization_order ===
  using VULKAN_HPP_NAMESPACE::RasterizationOrderAMD;

  //=== VK_KHR_video_queue ===
  using VULKAN_HPP_NAMESPACE::QueryResultStatusKHR;
  using VULKAN_HPP_NAMESPACE::VideoBeginCodingFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoBeginCodingFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoCapabilityFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoCapabilityFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoChromaSubsamplingFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoChromaSubsamplingFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoCodecOperationFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoCodecOperationFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoCodingControlFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoCodingControlFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoComponentBitDepthFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoComponentBitDepthFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEndCodingFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEndCodingFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoSessionCreateFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoSessionCreateFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoSessionParametersCreateFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoSessionParametersCreateFlagsKHR;

  //=== VK_KHR_video_decode_queue ===
  using VULKAN_HPP_NAMESPACE::VideoDecodeCapabilityFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeCapabilityFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeUsageFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeUsageFlagsKHR;

  //=== VK_EXT_transform_feedback ===
  using VULKAN_HPP_NAMESPACE::PipelineRasterizationStateStreamCreateFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::PipelineRasterizationStateStreamCreateFlagsEXT;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_EXT_video_encode_h264 ===
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264CapabilityFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264CapabilityFlagsEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264RateControlFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264RateControlFlagsEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264StdFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264StdFlagsEXT;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_EXT_video_encode_h265 ===
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265CapabilityFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265CapabilityFlagsEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265CtbSizeFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265CtbSizeFlagsEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265RateControlFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265RateControlFlagsEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265StdFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265StdFlagsEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265TransformBlockSizeFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265TransformBlockSizeFlagsEXT;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_KHR_video_decode_h264 ===
  using VULKAN_HPP_NAMESPACE::VideoDecodeH264PictureLayoutFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeH264PictureLayoutFlagsKHR;

  //=== VK_AMD_shader_info ===
  using VULKAN_HPP_NAMESPACE::ShaderInfoTypeAMD;

#if defined( VK_USE_PLATFORM_GGP )
  //=== VK_GGP_stream_descriptor_surface ===
  using VULKAN_HPP_NAMESPACE::StreamDescriptorSurfaceCreateFlagBitsGGP;
  using VULKAN_HPP_NAMESPACE::StreamDescriptorSurfaceCreateFlagsGGP;
#endif /*VK_USE_PLATFORM_GGP*/

  //=== VK_NV_external_memory_capabilities ===
  using VULKAN_HPP_NAMESPACE::ExternalMemoryFeatureFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryFeatureFlagsNV;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryHandleTypeFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryHandleTypeFlagsNV;

  //=== VK_EXT_validation_flags ===
  using VULKAN_HPP_NAMESPACE::ValidationCheckEXT;

#if defined( VK_USE_PLATFORM_VI_NN )
  //=== VK_NN_vi_surface ===
  using VULKAN_HPP_NAMESPACE::ViSurfaceCreateFlagBitsNN;
  using VULKAN_HPP_NAMESPACE::ViSurfaceCreateFlagsNN;
#endif /*VK_USE_PLATFORM_VI_NN*/

  //=== VK_EXT_pipeline_robustness ===
  using VULKAN_HPP_NAMESPACE::PipelineRobustnessBufferBehaviorEXT;
  using VULKAN_HPP_NAMESPACE::PipelineRobustnessImageBehaviorEXT;

  //=== VK_EXT_conditional_rendering ===
  using VULKAN_HPP_NAMESPACE::ConditionalRenderingFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::ConditionalRenderingFlagsEXT;

  //=== VK_EXT_display_surface_counter ===
  using VULKAN_HPP_NAMESPACE::SurfaceCounterFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::SurfaceCounterFlagsEXT;

  //=== VK_EXT_display_control ===
  using VULKAN_HPP_NAMESPACE::DeviceEventTypeEXT;
  using VULKAN_HPP_NAMESPACE::DisplayEventTypeEXT;
  using VULKAN_HPP_NAMESPACE::DisplayPowerStateEXT;

  //=== VK_NV_viewport_swizzle ===
  using VULKAN_HPP_NAMESPACE::PipelineViewportSwizzleStateCreateFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::PipelineViewportSwizzleStateCreateFlagsNV;
  using VULKAN_HPP_NAMESPACE::ViewportCoordinateSwizzleNV;

  //=== VK_EXT_discard_rectangles ===
  using VULKAN_HPP_NAMESPACE::DiscardRectangleModeEXT;
  using VULKAN_HPP_NAMESPACE::PipelineDiscardRectangleStateCreateFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::PipelineDiscardRectangleStateCreateFlagsEXT;

  //=== VK_EXT_conservative_rasterization ===
  using VULKAN_HPP_NAMESPACE::ConservativeRasterizationModeEXT;
  using VULKAN_HPP_NAMESPACE::PipelineRasterizationConservativeStateCreateFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::PipelineRasterizationConservativeStateCreateFlagsEXT;

  //=== VK_EXT_depth_clip_enable ===
  using VULKAN_HPP_NAMESPACE::PipelineRasterizationDepthClipStateCreateFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::PipelineRasterizationDepthClipStateCreateFlagsEXT;

  //=== VK_KHR_performance_query ===
  using VULKAN_HPP_NAMESPACE::AcquireProfilingLockFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::AcquireProfilingLockFlagsKHR;
  using VULKAN_HPP_NAMESPACE::PerformanceCounterDescriptionFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::PerformanceCounterDescriptionFlagsKHR;
  using VULKAN_HPP_NAMESPACE::PerformanceCounterScopeKHR;
  using VULKAN_HPP_NAMESPACE::PerformanceCounterStorageKHR;
  using VULKAN_HPP_NAMESPACE::PerformanceCounterUnitKHR;

#if defined( VK_USE_PLATFORM_IOS_MVK )
  //=== VK_MVK_ios_surface ===
  using VULKAN_HPP_NAMESPACE::IOSSurfaceCreateFlagBitsMVK;
  using VULKAN_HPP_NAMESPACE::IOSSurfaceCreateFlagsMVK;
#endif /*VK_USE_PLATFORM_IOS_MVK*/

#if defined( VK_USE_PLATFORM_MACOS_MVK )
  //=== VK_MVK_macos_surface ===
  using VULKAN_HPP_NAMESPACE::MacOSSurfaceCreateFlagBitsMVK;
  using VULKAN_HPP_NAMESPACE::MacOSSurfaceCreateFlagsMVK;
#endif /*VK_USE_PLATFORM_MACOS_MVK*/

  //=== VK_EXT_debug_utils ===
  using VULKAN_HPP_NAMESPACE::DebugUtilsMessageSeverityFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::DebugUtilsMessageSeverityFlagsEXT;
  using VULKAN_HPP_NAMESPACE::DebugUtilsMessageTypeFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::DebugUtilsMessageTypeFlagsEXT;
  using VULKAN_HPP_NAMESPACE::DebugUtilsMessengerCallbackDataFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::DebugUtilsMessengerCallbackDataFlagsEXT;
  using VULKAN_HPP_NAMESPACE::DebugUtilsMessengerCreateFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::DebugUtilsMessengerCreateFlagsEXT;

  //=== VK_EXT_blend_operation_advanced ===
  using VULKAN_HPP_NAMESPACE::BlendOverlapEXT;

  //=== VK_NV_fragment_coverage_to_color ===
  using VULKAN_HPP_NAMESPACE::PipelineCoverageToColorStateCreateFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::PipelineCoverageToColorStateCreateFlagsNV;

  //=== VK_KHR_acceleration_structure ===
  using VULKAN_HPP_NAMESPACE::AccelerationStructureBuildTypeKHR;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureCompatibilityKHR;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureCreateFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureCreateFlagsKHR;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureTypeKHR;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureTypeNV;
  using VULKAN_HPP_NAMESPACE::BuildAccelerationStructureFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::BuildAccelerationStructureFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::BuildAccelerationStructureFlagsKHR;
  using VULKAN_HPP_NAMESPACE::BuildAccelerationStructureModeKHR;
  using VULKAN_HPP_NAMESPACE::CopyAccelerationStructureModeKHR;
  using VULKAN_HPP_NAMESPACE::CopyAccelerationStructureModeNV;
  using VULKAN_HPP_NAMESPACE::GeometryFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::GeometryFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::GeometryFlagsKHR;
  using VULKAN_HPP_NAMESPACE::GeometryInstanceFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::GeometryInstanceFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::GeometryInstanceFlagsKHR;
  using VULKAN_HPP_NAMESPACE::GeometryTypeKHR;
  using VULKAN_HPP_NAMESPACE::GeometryTypeNV;

  //=== VK_KHR_ray_tracing_pipeline ===
  using VULKAN_HPP_NAMESPACE::RayTracingShaderGroupTypeKHR;
  using VULKAN_HPP_NAMESPACE::RayTracingShaderGroupTypeNV;
  using VULKAN_HPP_NAMESPACE::ShaderGroupShaderKHR;

  //=== VK_NV_framebuffer_mixed_samples ===
  using VULKAN_HPP_NAMESPACE::CoverageModulationModeNV;
  using VULKAN_HPP_NAMESPACE::PipelineCoverageModulationStateCreateFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::PipelineCoverageModulationStateCreateFlagsNV;

  //=== VK_EXT_validation_cache ===
  using VULKAN_HPP_NAMESPACE::ValidationCacheCreateFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::ValidationCacheCreateFlagsEXT;
  using VULKAN_HPP_NAMESPACE::ValidationCacheHeaderVersionEXT;

  //=== VK_NV_shading_rate_image ===
  using VULKAN_HPP_NAMESPACE::CoarseSampleOrderTypeNV;
  using VULKAN_HPP_NAMESPACE::ShadingRatePaletteEntryNV;

  //=== VK_NV_ray_tracing ===
  using VULKAN_HPP_NAMESPACE::AccelerationStructureMemoryRequirementsTypeNV;

  //=== VK_AMD_pipeline_compiler_control ===
  using VULKAN_HPP_NAMESPACE::PipelineCompilerControlFlagBitsAMD;
  using VULKAN_HPP_NAMESPACE::PipelineCompilerControlFlagsAMD;

  //=== VK_EXT_calibrated_timestamps ===
  using VULKAN_HPP_NAMESPACE::TimeDomainEXT;

  //=== VK_KHR_global_priority ===
  using VULKAN_HPP_NAMESPACE::QueueGlobalPriorityEXT;
  using VULKAN_HPP_NAMESPACE::QueueGlobalPriorityKHR;

  //=== VK_AMD_memory_overallocation_behavior ===
  using VULKAN_HPP_NAMESPACE::MemoryOverallocationBehaviorAMD;

  //=== VK_INTEL_performance_query ===
  using VULKAN_HPP_NAMESPACE::PerformanceConfigurationTypeINTEL;
  using VULKAN_HPP_NAMESPACE::PerformanceOverrideTypeINTEL;
  using VULKAN_HPP_NAMESPACE::PerformanceParameterTypeINTEL;
  using VULKAN_HPP_NAMESPACE::PerformanceValueTypeINTEL;
  using VULKAN_HPP_NAMESPACE::QueryPoolSamplingModeINTEL;

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_imagepipe_surface ===
  using VULKAN_HPP_NAMESPACE::ImagePipeSurfaceCreateFlagBitsFUCHSIA;
  using VULKAN_HPP_NAMESPACE::ImagePipeSurfaceCreateFlagsFUCHSIA;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_surface ===
  using VULKAN_HPP_NAMESPACE::MetalSurfaceCreateFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::MetalSurfaceCreateFlagsEXT;
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_KHR_fragment_shading_rate ===
  using VULKAN_HPP_NAMESPACE::FragmentShadingRateCombinerOpKHR;

  //=== VK_AMD_shader_core_properties2 ===
  using VULKAN_HPP_NAMESPACE::ShaderCorePropertiesFlagBitsAMD;
  using VULKAN_HPP_NAMESPACE::ShaderCorePropertiesFlagsAMD;

  //=== VK_EXT_validation_features ===
  using VULKAN_HPP_NAMESPACE::ValidationFeatureDisableEXT;
  using VULKAN_HPP_NAMESPACE::ValidationFeatureEnableEXT;

  //=== VK_NV_coverage_reduction_mode ===
  using VULKAN_HPP_NAMESPACE::CoverageReductionModeNV;
  using VULKAN_HPP_NAMESPACE::PipelineCoverageReductionStateCreateFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::PipelineCoverageReductionStateCreateFlagsNV;

  //=== VK_EXT_provoking_vertex ===
  using VULKAN_HPP_NAMESPACE::ProvokingVertexModeEXT;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_EXT_full_screen_exclusive ===
  using VULKAN_HPP_NAMESPACE::FullScreenExclusiveEXT;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_EXT_headless_surface ===
  using VULKAN_HPP_NAMESPACE::HeadlessSurfaceCreateFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::HeadlessSurfaceCreateFlagsEXT;

  //=== VK_EXT_line_rasterization ===
  using VULKAN_HPP_NAMESPACE::LineRasterizationModeEXT;

  //=== VK_KHR_pipeline_executable_properties ===
  using VULKAN_HPP_NAMESPACE::PipelineExecutableStatisticFormatKHR;

  //=== VK_EXT_host_image_copy ===
  using VULKAN_HPP_NAMESPACE::HostImageCopyFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::HostImageCopyFlagsEXT;

  //=== VK_KHR_map_memory2 ===
  using VULKAN_HPP_NAMESPACE::MemoryUnmapFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::MemoryUnmapFlagsKHR;

  //=== VK_EXT_surface_maintenance1 ===
  using VULKAN_HPP_NAMESPACE::PresentGravityFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::PresentGravityFlagsEXT;
  using VULKAN_HPP_NAMESPACE::PresentScalingFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::PresentScalingFlagsEXT;

  //=== VK_NV_device_generated_commands ===
  using VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutUsageFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutUsageFlagsNV;
  using VULKAN_HPP_NAMESPACE::IndirectCommandsTokenTypeNV;
  using VULKAN_HPP_NAMESPACE::IndirectStateFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::IndirectStateFlagsNV;

  //=== VK_EXT_depth_bias_control ===
  using VULKAN_HPP_NAMESPACE::DepthBiasRepresentationEXT;

  //=== VK_EXT_device_memory_report ===
  using VULKAN_HPP_NAMESPACE::DeviceMemoryReportEventTypeEXT;
  using VULKAN_HPP_NAMESPACE::DeviceMemoryReportFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::DeviceMemoryReportFlagsEXT;

  //=== VK_EXT_pipeline_creation_cache_control ===
  using VULKAN_HPP_NAMESPACE::PipelineCacheCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::PipelineCacheCreateFlags;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_KHR_video_encode_queue ===
  using VULKAN_HPP_NAMESPACE::VideoEncodeCapabilityFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeCapabilityFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeContentFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeContentFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeFeedbackFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeFeedbackFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeRateControlFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeRateControlFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeRateControlModeFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeRateControlModeFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeTuningModeKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeUsageFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeUsageFlagsKHR;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_NV_device_diagnostics_config ===
  using VULKAN_HPP_NAMESPACE::DeviceDiagnosticsConfigFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::DeviceDiagnosticsConfigFlagsNV;

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_objects ===
  using VULKAN_HPP_NAMESPACE::ExportMetalObjectTypeFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::ExportMetalObjectTypeFlagsEXT;
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_EXT_graphics_pipeline_library ===
  using VULKAN_HPP_NAMESPACE::GraphicsPipelineLibraryFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::GraphicsPipelineLibraryFlagsEXT;
  using VULKAN_HPP_NAMESPACE::PipelineLayoutCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::PipelineLayoutCreateFlags;

  //=== VK_NV_fragment_shading_rate_enums ===
  using VULKAN_HPP_NAMESPACE::FragmentShadingRateNV;
  using VULKAN_HPP_NAMESPACE::FragmentShadingRateTypeNV;

  //=== VK_NV_ray_tracing_motion_blur ===
  using VULKAN_HPP_NAMESPACE::AccelerationStructureMotionInfoFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureMotionInfoFlagsNV;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureMotionInstanceFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureMotionInstanceFlagsNV;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureMotionInstanceTypeNV;

  //=== VK_EXT_image_compression_control ===
  using VULKAN_HPP_NAMESPACE::ImageCompressionFixedRateFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::ImageCompressionFixedRateFlagsEXT;
  using VULKAN_HPP_NAMESPACE::ImageCompressionFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::ImageCompressionFlagsEXT;

  //=== VK_EXT_device_fault ===
  using VULKAN_HPP_NAMESPACE::DeviceFaultAddressTypeEXT;
  using VULKAN_HPP_NAMESPACE::DeviceFaultVendorBinaryHeaderVersionEXT;

#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
  //=== VK_EXT_directfb_surface ===
  using VULKAN_HPP_NAMESPACE::DirectFBSurfaceCreateFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::DirectFBSurfaceCreateFlagsEXT;
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

  //=== VK_EXT_device_address_binding_report ===
  using VULKAN_HPP_NAMESPACE::DeviceAddressBindingFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::DeviceAddressBindingFlagsEXT;
  using VULKAN_HPP_NAMESPACE::DeviceAddressBindingTypeEXT;

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_buffer_collection ===
  using VULKAN_HPP_NAMESPACE::ImageConstraintsInfoFlagBitsFUCHSIA;
  using VULKAN_HPP_NAMESPACE::ImageConstraintsInfoFlagsFUCHSIA;
  using VULKAN_HPP_NAMESPACE::ImageFormatConstraintsFlagBitsFUCHSIA;
  using VULKAN_HPP_NAMESPACE::ImageFormatConstraintsFlagsFUCHSIA;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
  //=== VK_QNX_screen_surface ===
  using VULKAN_HPP_NAMESPACE::ScreenSurfaceCreateFlagBitsQNX;
  using VULKAN_HPP_NAMESPACE::ScreenSurfaceCreateFlagsQNX;
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

  //=== VK_EXT_opacity_micromap ===
  using VULKAN_HPP_NAMESPACE::BuildMicromapFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::BuildMicromapFlagsEXT;
  using VULKAN_HPP_NAMESPACE::BuildMicromapModeEXT;
  using VULKAN_HPP_NAMESPACE::CopyMicromapModeEXT;
  using VULKAN_HPP_NAMESPACE::MicromapCreateFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::MicromapCreateFlagsEXT;
  using VULKAN_HPP_NAMESPACE::MicromapTypeEXT;
  using VULKAN_HPP_NAMESPACE::OpacityMicromapFormatEXT;
  using VULKAN_HPP_NAMESPACE::OpacityMicromapSpecialIndexEXT;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_NV_displacement_micromap ===
  using VULKAN_HPP_NAMESPACE::DisplacementMicromapFormatNV;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_NV_memory_decompression ===
  using VULKAN_HPP_NAMESPACE::MemoryDecompressionMethodFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::MemoryDecompressionMethodFlagsNV;

  //=== VK_EXT_subpass_merge_feedback ===
  using VULKAN_HPP_NAMESPACE::SubpassMergeStatusEXT;

  //=== VK_LUNARG_direct_driver_loading ===
  using VULKAN_HPP_NAMESPACE::DirectDriverLoadingFlagBitsLUNARG;
  using VULKAN_HPP_NAMESPACE::DirectDriverLoadingFlagsLUNARG;
  using VULKAN_HPP_NAMESPACE::DirectDriverLoadingModeLUNARG;

  //=== VK_EXT_rasterization_order_attachment_access ===
  using VULKAN_HPP_NAMESPACE::PipelineColorBlendStateCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::PipelineColorBlendStateCreateFlags;
  using VULKAN_HPP_NAMESPACE::PipelineDepthStencilStateCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::PipelineDepthStencilStateCreateFlags;

  //=== VK_NV_optical_flow ===
  using VULKAN_HPP_NAMESPACE::OpticalFlowExecuteFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::OpticalFlowExecuteFlagsNV;
  using VULKAN_HPP_NAMESPACE::OpticalFlowGridSizeFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::OpticalFlowGridSizeFlagsNV;
  using VULKAN_HPP_NAMESPACE::OpticalFlowPerformanceLevelNV;
  using VULKAN_HPP_NAMESPACE::OpticalFlowSessionBindingPointNV;
  using VULKAN_HPP_NAMESPACE::OpticalFlowSessionCreateFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::OpticalFlowSessionCreateFlagsNV;
  using VULKAN_HPP_NAMESPACE::OpticalFlowUsageFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::OpticalFlowUsageFlagsNV;

  //=== VK_KHR_maintenance5 ===
  using VULKAN_HPP_NAMESPACE::BufferUsageFlagBits2KHR;
  using VULKAN_HPP_NAMESPACE::BufferUsageFlags2KHR;
  using VULKAN_HPP_NAMESPACE::PipelineCreateFlagBits2KHR;
  using VULKAN_HPP_NAMESPACE::PipelineCreateFlags2KHR;

  //=== VK_EXT_shader_object ===
  using VULKAN_HPP_NAMESPACE::ShaderCodeTypeEXT;
  using VULKAN_HPP_NAMESPACE::ShaderCreateFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::ShaderCreateFlagsEXT;

  //=== VK_NV_ray_tracing_invocation_reorder ===
  using VULKAN_HPP_NAMESPACE::RayTracingInvocationReorderModeNV;

  //=== VK_KHR_cooperative_matrix ===
  using VULKAN_HPP_NAMESPACE::ComponentTypeKHR;
  using VULKAN_HPP_NAMESPACE::ComponentTypeNV;
  using VULKAN_HPP_NAMESPACE::ScopeKHR;
  using VULKAN_HPP_NAMESPACE::ScopeNV;

  //=========================
  //=== Index Type Traits ===
  //=========================
  using VULKAN_HPP_NAMESPACE::IndexTypeValue;

  //======================
  //=== ENUM to_string ===
  //======================
#if !defined( VULKAN_HPP_NO_TO_STRING )
  using VULKAN_HPP_NAMESPACE::to_string;
  using VULKAN_HPP_NAMESPACE::toHexString;
#endif /*VULKAN_HPP_NO_TO_STRING*/

  //=============================
  //=== EXCEPTIONs AND ERRORs ===
  //=============================
#if !defined( VULKAN_HPP_NO_EXCEPTIONS )
  using VULKAN_HPP_NAMESPACE::DeviceLostError;
  using VULKAN_HPP_NAMESPACE::Error;
  using VULKAN_HPP_NAMESPACE::errorCategory;
  using VULKAN_HPP_NAMESPACE::ErrorCategoryImpl;
  using VULKAN_HPP_NAMESPACE::ExtensionNotPresentError;
  using VULKAN_HPP_NAMESPACE::FeatureNotPresentError;
  using VULKAN_HPP_NAMESPACE::FormatNotSupportedError;
  using VULKAN_HPP_NAMESPACE::FragmentationError;
  using VULKAN_HPP_NAMESPACE::FragmentedPoolError;
  using VULKAN_HPP_NAMESPACE::ImageUsageNotSupportedKHRError;
  using VULKAN_HPP_NAMESPACE::IncompatibleDisplayKHRError;
  using VULKAN_HPP_NAMESPACE::IncompatibleDriverError;
  using VULKAN_HPP_NAMESPACE::InitializationFailedError;
  using VULKAN_HPP_NAMESPACE::InvalidDrmFormatModifierPlaneLayoutEXTError;
  using VULKAN_HPP_NAMESPACE::InvalidExternalHandleError;
  using VULKAN_HPP_NAMESPACE::InvalidOpaqueCaptureAddressError;
  using VULKAN_HPP_NAMESPACE::InvalidShaderNVError;
  using VULKAN_HPP_NAMESPACE::LayerNotPresentError;
  using VULKAN_HPP_NAMESPACE::LogicError;
  using VULKAN_HPP_NAMESPACE::make_error_code;
  using VULKAN_HPP_NAMESPACE::make_error_condition;
  using VULKAN_HPP_NAMESPACE::MemoryMapFailedError;
  using VULKAN_HPP_NAMESPACE::NativeWindowInUseKHRError;
  using VULKAN_HPP_NAMESPACE::NotPermittedKHRError;
  using VULKAN_HPP_NAMESPACE::OutOfDateKHRError;
  using VULKAN_HPP_NAMESPACE::OutOfDeviceMemoryError;
  using VULKAN_HPP_NAMESPACE::OutOfHostMemoryError;
  using VULKAN_HPP_NAMESPACE::OutOfPoolMemoryError;
  using VULKAN_HPP_NAMESPACE::SurfaceLostKHRError;
  using VULKAN_HPP_NAMESPACE::SystemError;
  using VULKAN_HPP_NAMESPACE::TooManyObjectsError;
  using VULKAN_HPP_NAMESPACE::UnknownError;
  using VULKAN_HPP_NAMESPACE::ValidationFailedEXTError;
  using VULKAN_HPP_NAMESPACE::VideoPictureLayoutNotSupportedKHRError;
  using VULKAN_HPP_NAMESPACE::VideoProfileCodecNotSupportedKHRError;
  using VULKAN_HPP_NAMESPACE::VideoProfileFormatNotSupportedKHRError;
  using VULKAN_HPP_NAMESPACE::VideoProfileOperationNotSupportedKHRError;
  using VULKAN_HPP_NAMESPACE::VideoStdVersionNotSupportedKHRError;

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  using VULKAN_HPP_NAMESPACE::FullScreenExclusiveModeLostEXTError;
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  using VULKAN_HPP_NAMESPACE::InvalidVideoStdParametersKHRError;
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

  using VULKAN_HPP_NAMESPACE::CompressionExhaustedEXTError;
  using VULKAN_HPP_NAMESPACE::IncompatibleShaderBinaryEXTError;
#endif /*VULKAN_HPP_NO_EXCEPTIONS*/

  using VULKAN_HPP_NAMESPACE::createResultValueType;
  using VULKAN_HPP_NAMESPACE::ignore;
  using VULKAN_HPP_NAMESPACE::resultCheck;
  using VULKAN_HPP_NAMESPACE::ResultValue;
  using VULKAN_HPP_NAMESPACE::ResultValueType;

  //===========================
  //=== CONSTEXPR CONSTANTs ===
  //===========================

  //=== VK_VERSION_1_0 ===
  using VULKAN_HPP_NAMESPACE::AttachmentUnused;
  using VULKAN_HPP_NAMESPACE::False;
  using VULKAN_HPP_NAMESPACE::LodClampNone;
  using VULKAN_HPP_NAMESPACE::MaxDescriptionSize;
  using VULKAN_HPP_NAMESPACE::MaxExtensionNameSize;
  using VULKAN_HPP_NAMESPACE::MaxMemoryHeaps;
  using VULKAN_HPP_NAMESPACE::MaxMemoryTypes;
  using VULKAN_HPP_NAMESPACE::MaxPhysicalDeviceNameSize;
  using VULKAN_HPP_NAMESPACE::QueueFamilyIgnored;
  using VULKAN_HPP_NAMESPACE::RemainingArrayLayers;
  using VULKAN_HPP_NAMESPACE::RemainingMipLevels;
  using VULKAN_HPP_NAMESPACE::SubpassExternal;
  using VULKAN_HPP_NAMESPACE::True;
  using VULKAN_HPP_NAMESPACE::UuidSize;
  using VULKAN_HPP_NAMESPACE::WholeSize;

  //=== VK_VERSION_1_1 ===
  using VULKAN_HPP_NAMESPACE::LuidSize;
  using VULKAN_HPP_NAMESPACE::MaxDeviceGroupSize;
  using VULKAN_HPP_NAMESPACE::QueueFamilyExternal;

  //=== VK_VERSION_1_2 ===
  using VULKAN_HPP_NAMESPACE::MaxDriverInfoSize;
  using VULKAN_HPP_NAMESPACE::MaxDriverNameSize;

  //=== VK_KHR_device_group_creation ===
  using VULKAN_HPP_NAMESPACE::MaxDeviceGroupSizeKhr;

  //=== VK_KHR_external_memory_capabilities ===
  using VULKAN_HPP_NAMESPACE::LuidSizeKhr;

  //=== VK_KHR_external_memory ===
  using VULKAN_HPP_NAMESPACE::QueueFamilyExternalKhr;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_AMDX_shader_enqueue ===
  using VULKAN_HPP_NAMESPACE::ShaderIndexUnusedAmdx;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_KHR_ray_tracing_pipeline ===
  using VULKAN_HPP_NAMESPACE::ShaderUnusedKhr;

  //=== VK_NV_ray_tracing ===
  using VULKAN_HPP_NAMESPACE::ShaderUnusedNv;

  //=== VK_KHR_global_priority ===
  using VULKAN_HPP_NAMESPACE::MaxGlobalPrioritySizeKhr;

  //=== VK_KHR_driver_properties ===
  using VULKAN_HPP_NAMESPACE::MaxDriverInfoSizeKhr;
  using VULKAN_HPP_NAMESPACE::MaxDriverNameSizeKhr;

  //=== VK_EXT_global_priority_query ===
  using VULKAN_HPP_NAMESPACE::MaxGlobalPrioritySizeExt;

  //=== VK_EXT_image_sliced_view_of_3d ===
  using VULKAN_HPP_NAMESPACE::Remaining3DSlicesExt;

  //=== VK_EXT_shader_module_identifier ===
  using VULKAN_HPP_NAMESPACE::MaxShaderModuleIdentifierSizeExt;

  //========================
  //=== CONSTEXPR VALUEs ===
  //========================
  using VULKAN_HPP_NAMESPACE::HeaderVersion;

  //=========================
  //=== CONSTEXPR CALLEEs ===
  //=========================
  using VULKAN_HPP_NAMESPACE::apiVersionMajor;
  using VULKAN_HPP_NAMESPACE::apiVersionMinor;
  using VULKAN_HPP_NAMESPACE::apiVersionPatch;
  using VULKAN_HPP_NAMESPACE::apiVersionVariant;
  using VULKAN_HPP_NAMESPACE::makeApiVersion;
  using VULKAN_HPP_NAMESPACE::makeVersion;
  using VULKAN_HPP_NAMESPACE::versionMajor;
  using VULKAN_HPP_NAMESPACE::versionMinor;
  using VULKAN_HPP_NAMESPACE::versionPatch;

  //==========================
  //=== CONSTEXPR CALLERSs ===
  //==========================
  using VULKAN_HPP_NAMESPACE::ApiVersion;
  using VULKAN_HPP_NAMESPACE::ApiVersion10;
  using VULKAN_HPP_NAMESPACE::ApiVersion11;
  using VULKAN_HPP_NAMESPACE::ApiVersion12;
  using VULKAN_HPP_NAMESPACE::ApiVersion13;
  using VULKAN_HPP_NAMESPACE::HeaderVersionComplete;

  //===============
  //=== STRUCTs ===
  //===============

  //=== VK_VERSION_1_0 ===
  using VULKAN_HPP_NAMESPACE::AllocationCallbacks;
  using VULKAN_HPP_NAMESPACE::ApplicationInfo;
  using VULKAN_HPP_NAMESPACE::AttachmentDescription;
  using VULKAN_HPP_NAMESPACE::AttachmentReference;
  using VULKAN_HPP_NAMESPACE::BaseInStructure;
  using VULKAN_HPP_NAMESPACE::BaseOutStructure;
  using VULKAN_HPP_NAMESPACE::BindSparseInfo;
  using VULKAN_HPP_NAMESPACE::BufferCopy;
  using VULKAN_HPP_NAMESPACE::BufferCreateInfo;
  using VULKAN_HPP_NAMESPACE::BufferImageCopy;
  using VULKAN_HPP_NAMESPACE::BufferMemoryBarrier;
  using VULKAN_HPP_NAMESPACE::BufferViewCreateInfo;
  using VULKAN_HPP_NAMESPACE::ClearAttachment;
  using VULKAN_HPP_NAMESPACE::ClearColorValue;
  using VULKAN_HPP_NAMESPACE::ClearDepthStencilValue;
  using VULKAN_HPP_NAMESPACE::ClearRect;
  using VULKAN_HPP_NAMESPACE::ClearValue;
  using VULKAN_HPP_NAMESPACE::CommandBufferAllocateInfo;
  using VULKAN_HPP_NAMESPACE::CommandBufferBeginInfo;
  using VULKAN_HPP_NAMESPACE::CommandBufferInheritanceInfo;
  using VULKAN_HPP_NAMESPACE::CommandPoolCreateInfo;
  using VULKAN_HPP_NAMESPACE::ComponentMapping;
  using VULKAN_HPP_NAMESPACE::ComputePipelineCreateInfo;
  using VULKAN_HPP_NAMESPACE::CopyDescriptorSet;
  using VULKAN_HPP_NAMESPACE::DescriptorBufferInfo;
  using VULKAN_HPP_NAMESPACE::DescriptorImageInfo;
  using VULKAN_HPP_NAMESPACE::DescriptorPoolCreateInfo;
  using VULKAN_HPP_NAMESPACE::DescriptorPoolSize;
  using VULKAN_HPP_NAMESPACE::DescriptorSetAllocateInfo;
  using VULKAN_HPP_NAMESPACE::DescriptorSetLayoutBinding;
  using VULKAN_HPP_NAMESPACE::DescriptorSetLayoutCreateInfo;
  using VULKAN_HPP_NAMESPACE::DeviceCreateInfo;
  using VULKAN_HPP_NAMESPACE::DeviceQueueCreateInfo;
  using VULKAN_HPP_NAMESPACE::DispatchIndirectCommand;
  using VULKAN_HPP_NAMESPACE::DrawIndexedIndirectCommand;
  using VULKAN_HPP_NAMESPACE::DrawIndirectCommand;
  using VULKAN_HPP_NAMESPACE::EventCreateInfo;
  using VULKAN_HPP_NAMESPACE::ExtensionProperties;
  using VULKAN_HPP_NAMESPACE::Extent2D;
  using VULKAN_HPP_NAMESPACE::Extent3D;
  using VULKAN_HPP_NAMESPACE::FenceCreateInfo;
  using VULKAN_HPP_NAMESPACE::FormatProperties;
  using VULKAN_HPP_NAMESPACE::FramebufferCreateInfo;
  using VULKAN_HPP_NAMESPACE::GraphicsPipelineCreateInfo;
  using VULKAN_HPP_NAMESPACE::ImageBlit;
  using VULKAN_HPP_NAMESPACE::ImageCopy;
  using VULKAN_HPP_NAMESPACE::ImageCreateInfo;
  using VULKAN_HPP_NAMESPACE::ImageFormatProperties;
  using VULKAN_HPP_NAMESPACE::ImageMemoryBarrier;
  using VULKAN_HPP_NAMESPACE::ImageResolve;
  using VULKAN_HPP_NAMESPACE::ImageSubresource;
  using VULKAN_HPP_NAMESPACE::ImageSubresourceLayers;
  using VULKAN_HPP_NAMESPACE::ImageSubresourceRange;
  using VULKAN_HPP_NAMESPACE::ImageViewCreateInfo;
  using VULKAN_HPP_NAMESPACE::InstanceCreateInfo;
  using VULKAN_HPP_NAMESPACE::LayerProperties;
  using VULKAN_HPP_NAMESPACE::MappedMemoryRange;
  using VULKAN_HPP_NAMESPACE::MemoryAllocateInfo;
  using VULKAN_HPP_NAMESPACE::MemoryBarrier;
  using VULKAN_HPP_NAMESPACE::MemoryHeap;
  using VULKAN_HPP_NAMESPACE::MemoryRequirements;
  using VULKAN_HPP_NAMESPACE::MemoryType;
  using VULKAN_HPP_NAMESPACE::Offset2D;
  using VULKAN_HPP_NAMESPACE::Offset3D;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceLimits;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSparseProperties;
  using VULKAN_HPP_NAMESPACE::PipelineCacheCreateInfo;
  using VULKAN_HPP_NAMESPACE::PipelineCacheHeaderVersionOne;
  using VULKAN_HPP_NAMESPACE::PipelineColorBlendAttachmentState;
  using VULKAN_HPP_NAMESPACE::PipelineColorBlendStateCreateInfo;
  using VULKAN_HPP_NAMESPACE::PipelineDepthStencilStateCreateInfo;
  using VULKAN_HPP_NAMESPACE::PipelineDynamicStateCreateInfo;
  using VULKAN_HPP_NAMESPACE::PipelineInputAssemblyStateCreateInfo;
  using VULKAN_HPP_NAMESPACE::PipelineLayoutCreateInfo;
  using VULKAN_HPP_NAMESPACE::PipelineMultisampleStateCreateInfo;
  using VULKAN_HPP_NAMESPACE::PipelineRasterizationStateCreateInfo;
  using VULKAN_HPP_NAMESPACE::PipelineShaderStageCreateInfo;
  using VULKAN_HPP_NAMESPACE::PipelineTessellationStateCreateInfo;
  using VULKAN_HPP_NAMESPACE::PipelineVertexInputStateCreateInfo;
  using VULKAN_HPP_NAMESPACE::PipelineViewportStateCreateInfo;
  using VULKAN_HPP_NAMESPACE::PushConstantRange;
  using VULKAN_HPP_NAMESPACE::QueryPoolCreateInfo;
  using VULKAN_HPP_NAMESPACE::QueueFamilyProperties;
  using VULKAN_HPP_NAMESPACE::Rect2D;
  using VULKAN_HPP_NAMESPACE::RenderPassBeginInfo;
  using VULKAN_HPP_NAMESPACE::RenderPassCreateInfo;
  using VULKAN_HPP_NAMESPACE::SamplerCreateInfo;
  using VULKAN_HPP_NAMESPACE::SemaphoreCreateInfo;
  using VULKAN_HPP_NAMESPACE::ShaderModuleCreateInfo;
  using VULKAN_HPP_NAMESPACE::SparseBufferMemoryBindInfo;
  using VULKAN_HPP_NAMESPACE::SparseImageFormatProperties;
  using VULKAN_HPP_NAMESPACE::SparseImageMemoryBind;
  using VULKAN_HPP_NAMESPACE::SparseImageMemoryBindInfo;
  using VULKAN_HPP_NAMESPACE::SparseImageMemoryRequirements;
  using VULKAN_HPP_NAMESPACE::SparseImageOpaqueMemoryBindInfo;
  using VULKAN_HPP_NAMESPACE::SparseMemoryBind;
  using VULKAN_HPP_NAMESPACE::SpecializationInfo;
  using VULKAN_HPP_NAMESPACE::SpecializationMapEntry;
  using VULKAN_HPP_NAMESPACE::StencilOpState;
  using VULKAN_HPP_NAMESPACE::SubmitInfo;
  using VULKAN_HPP_NAMESPACE::SubpassDependency;
  using VULKAN_HPP_NAMESPACE::SubpassDescription;
  using VULKAN_HPP_NAMESPACE::SubresourceLayout;
  using VULKAN_HPP_NAMESPACE::VertexInputAttributeDescription;
  using VULKAN_HPP_NAMESPACE::VertexInputBindingDescription;
  using VULKAN_HPP_NAMESPACE::Viewport;
  using VULKAN_HPP_NAMESPACE::WriteDescriptorSet;

  //=== VK_VERSION_1_1 ===
  using VULKAN_HPP_NAMESPACE::BindBufferMemoryDeviceGroupInfo;
  using VULKAN_HPP_NAMESPACE::BindBufferMemoryDeviceGroupInfoKHR;
  using VULKAN_HPP_NAMESPACE::BindBufferMemoryInfo;
  using VULKAN_HPP_NAMESPACE::BindBufferMemoryInfoKHR;
  using VULKAN_HPP_NAMESPACE::BindImageMemoryDeviceGroupInfo;
  using VULKAN_HPP_NAMESPACE::BindImageMemoryDeviceGroupInfoKHR;
  using VULKAN_HPP_NAMESPACE::BindImageMemoryInfo;
  using VULKAN_HPP_NAMESPACE::BindImageMemoryInfoKHR;
  using VULKAN_HPP_NAMESPACE::BindImagePlaneMemoryInfo;
  using VULKAN_HPP_NAMESPACE::BindImagePlaneMemoryInfoKHR;
  using VULKAN_HPP_NAMESPACE::BufferMemoryRequirementsInfo2;
  using VULKAN_HPP_NAMESPACE::BufferMemoryRequirementsInfo2KHR;
  using VULKAN_HPP_NAMESPACE::DescriptorSetLayoutSupport;
  using VULKAN_HPP_NAMESPACE::DescriptorSetLayoutSupportKHR;
  using VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateCreateInfo;
  using VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateEntry;
  using VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateEntryKHR;
  using VULKAN_HPP_NAMESPACE::DeviceGroupBindSparseInfo;
  using VULKAN_HPP_NAMESPACE::DeviceGroupBindSparseInfoKHR;
  using VULKAN_HPP_NAMESPACE::DeviceGroupCommandBufferBeginInfo;
  using VULKAN_HPP_NAMESPACE::DeviceGroupCommandBufferBeginInfoKHR;
  using VULKAN_HPP_NAMESPACE::DeviceGroupDeviceCreateInfo;
  using VULKAN_HPP_NAMESPACE::DeviceGroupDeviceCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::DeviceGroupRenderPassBeginInfo;
  using VULKAN_HPP_NAMESPACE::DeviceGroupRenderPassBeginInfoKHR;
  using VULKAN_HPP_NAMESPACE::DeviceGroupSubmitInfo;
  using VULKAN_HPP_NAMESPACE::DeviceGroupSubmitInfoKHR;
  using VULKAN_HPP_NAMESPACE::DeviceQueueInfo2;
  using VULKAN_HPP_NAMESPACE::ExportFenceCreateInfo;
  using VULKAN_HPP_NAMESPACE::ExportFenceCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::ExportMemoryAllocateInfo;
  using VULKAN_HPP_NAMESPACE::ExportMemoryAllocateInfoKHR;
  using VULKAN_HPP_NAMESPACE::ExportSemaphoreCreateInfo;
  using VULKAN_HPP_NAMESPACE::ExportSemaphoreCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::ExternalBufferProperties;
  using VULKAN_HPP_NAMESPACE::ExternalBufferPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::ExternalFenceProperties;
  using VULKAN_HPP_NAMESPACE::ExternalFencePropertiesKHR;
  using VULKAN_HPP_NAMESPACE::ExternalImageFormatProperties;
  using VULKAN_HPP_NAMESPACE::ExternalImageFormatPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryBufferCreateInfo;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryBufferCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryImageCreateInfo;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryImageCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryProperties;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::ExternalSemaphoreProperties;
  using VULKAN_HPP_NAMESPACE::ExternalSemaphorePropertiesKHR;
  using VULKAN_HPP_NAMESPACE::FormatProperties2;
  using VULKAN_HPP_NAMESPACE::FormatProperties2KHR;
  using VULKAN_HPP_NAMESPACE::ImageFormatProperties2;
  using VULKAN_HPP_NAMESPACE::ImageFormatProperties2KHR;
  using VULKAN_HPP_NAMESPACE::ImageMemoryRequirementsInfo2;
  using VULKAN_HPP_NAMESPACE::ImageMemoryRequirementsInfo2KHR;
  using VULKAN_HPP_NAMESPACE::ImagePlaneMemoryRequirementsInfo;
  using VULKAN_HPP_NAMESPACE::ImagePlaneMemoryRequirementsInfoKHR;
  using VULKAN_HPP_NAMESPACE::ImageSparseMemoryRequirementsInfo2;
  using VULKAN_HPP_NAMESPACE::ImageSparseMemoryRequirementsInfo2KHR;
  using VULKAN_HPP_NAMESPACE::ImageViewUsageCreateInfo;
  using VULKAN_HPP_NAMESPACE::ImageViewUsageCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::InputAttachmentAspectReference;
  using VULKAN_HPP_NAMESPACE::InputAttachmentAspectReferenceKHR;
  using VULKAN_HPP_NAMESPACE::MemoryAllocateFlagsInfo;
  using VULKAN_HPP_NAMESPACE::MemoryAllocateFlagsInfoKHR;
  using VULKAN_HPP_NAMESPACE::MemoryDedicatedAllocateInfo;
  using VULKAN_HPP_NAMESPACE::MemoryDedicatedAllocateInfoKHR;
  using VULKAN_HPP_NAMESPACE::MemoryDedicatedRequirements;
  using VULKAN_HPP_NAMESPACE::MemoryDedicatedRequirementsKHR;
  using VULKAN_HPP_NAMESPACE::MemoryRequirements2;
  using VULKAN_HPP_NAMESPACE::MemoryRequirements2KHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDevice16BitStorageFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDevice16BitStorageFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalBufferInfo;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalBufferInfoKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalFenceInfo;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalFenceInfoKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalImageFormatInfo;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalImageFormatInfoKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalSemaphoreInfo;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalSemaphoreInfoKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures2;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures2KHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceGroupProperties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceGroupPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceIDProperties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceIDPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceImageFormatInfo2;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceImageFormatInfo2KHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance3Properties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance3PropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties2;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties2KHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewProperties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePointClippingProperties;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePointClippingPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties2;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties2KHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceProtectedMemoryFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceProtectedMemoryProperties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSamplerYcbcrConversionFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSamplerYcbcrConversionFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderDrawParameterFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderDrawParametersFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSparseImageFormatInfo2;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSparseImageFormatInfo2KHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSubgroupProperties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVariablePointerFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVariablePointerFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVariablePointersFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVariablePointersFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PipelineTessellationDomainOriginStateCreateInfo;
  using VULKAN_HPP_NAMESPACE::PipelineTessellationDomainOriginStateCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::ProtectedSubmitInfo;
  using VULKAN_HPP_NAMESPACE::QueueFamilyProperties2;
  using VULKAN_HPP_NAMESPACE::QueueFamilyProperties2KHR;
  using VULKAN_HPP_NAMESPACE::RenderPassInputAttachmentAspectCreateInfo;
  using VULKAN_HPP_NAMESPACE::RenderPassInputAttachmentAspectCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::RenderPassMultiviewCreateInfo;
  using VULKAN_HPP_NAMESPACE::RenderPassMultiviewCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionCreateInfo;
  using VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionImageFormatProperties;
  using VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionImageFormatPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionInfo;
  using VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionInfoKHR;
  using VULKAN_HPP_NAMESPACE::SparseImageFormatProperties2;
  using VULKAN_HPP_NAMESPACE::SparseImageFormatProperties2KHR;
  using VULKAN_HPP_NAMESPACE::SparseImageMemoryRequirements2;
  using VULKAN_HPP_NAMESPACE::SparseImageMemoryRequirements2KHR;

  //=== VK_VERSION_1_2 ===
  using VULKAN_HPP_NAMESPACE::AttachmentDescription2;
  using VULKAN_HPP_NAMESPACE::AttachmentDescription2KHR;
  using VULKAN_HPP_NAMESPACE::AttachmentDescriptionStencilLayout;
  using VULKAN_HPP_NAMESPACE::AttachmentDescriptionStencilLayoutKHR;
  using VULKAN_HPP_NAMESPACE::AttachmentReference2;
  using VULKAN_HPP_NAMESPACE::AttachmentReference2KHR;
  using VULKAN_HPP_NAMESPACE::AttachmentReferenceStencilLayout;
  using VULKAN_HPP_NAMESPACE::AttachmentReferenceStencilLayoutKHR;
  using VULKAN_HPP_NAMESPACE::BufferDeviceAddressInfo;
  using VULKAN_HPP_NAMESPACE::BufferDeviceAddressInfoEXT;
  using VULKAN_HPP_NAMESPACE::BufferDeviceAddressInfoKHR;
  using VULKAN_HPP_NAMESPACE::BufferOpaqueCaptureAddressCreateInfo;
  using VULKAN_HPP_NAMESPACE::BufferOpaqueCaptureAddressCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::ConformanceVersion;
  using VULKAN_HPP_NAMESPACE::ConformanceVersionKHR;
  using VULKAN_HPP_NAMESPACE::DescriptorSetLayoutBindingFlagsCreateInfo;
  using VULKAN_HPP_NAMESPACE::DescriptorSetLayoutBindingFlagsCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::DescriptorSetVariableDescriptorCountAllocateInfo;
  using VULKAN_HPP_NAMESPACE::DescriptorSetVariableDescriptorCountAllocateInfoEXT;
  using VULKAN_HPP_NAMESPACE::DescriptorSetVariableDescriptorCountLayoutSupport;
  using VULKAN_HPP_NAMESPACE::DescriptorSetVariableDescriptorCountLayoutSupportEXT;
  using VULKAN_HPP_NAMESPACE::DeviceMemoryOpaqueCaptureAddressInfo;
  using VULKAN_HPP_NAMESPACE::DeviceMemoryOpaqueCaptureAddressInfoKHR;
  using VULKAN_HPP_NAMESPACE::FramebufferAttachmentImageInfo;
  using VULKAN_HPP_NAMESPACE::FramebufferAttachmentImageInfoKHR;
  using VULKAN_HPP_NAMESPACE::FramebufferAttachmentsCreateInfo;
  using VULKAN_HPP_NAMESPACE::FramebufferAttachmentsCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::ImageFormatListCreateInfo;
  using VULKAN_HPP_NAMESPACE::ImageFormatListCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::ImageStencilUsageCreateInfo;
  using VULKAN_HPP_NAMESPACE::ImageStencilUsageCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::MemoryOpaqueCaptureAddressAllocateInfo;
  using VULKAN_HPP_NAMESPACE::MemoryOpaqueCaptureAddressAllocateInfoKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDevice8BitStorageFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDevice8BitStorageFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceBufferDeviceAddressFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceBufferDeviceAddressFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthStencilResolveProperties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthStencilResolvePropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorIndexingFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorIndexingFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorIndexingProperties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorIndexingPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDriverProperties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDriverPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFloat16Int8FeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFloatControlsProperties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFloatControlsPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceHostQueryResetFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceHostQueryResetFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceImagelessFramebufferFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceImagelessFramebufferFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSamplerFilterMinmaxProperties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSamplerFilterMinmaxPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceScalarBlockLayoutFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceScalarBlockLayoutFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSeparateDepthStencilLayoutsFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSeparateDepthStencilLayoutsFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicInt64Features;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicInt64FeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderFloat16Int8Features;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderFloat16Int8FeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSubgroupExtendedTypesFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSubgroupExtendedTypesFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceTimelineSemaphoreFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceTimelineSemaphoreFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceTimelineSemaphoreProperties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceTimelineSemaphorePropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceUniformBufferStandardLayoutFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceUniformBufferStandardLayoutFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan11Features;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan11Properties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan12Features;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan12Properties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkanMemoryModelFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkanMemoryModelFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::RenderPassAttachmentBeginInfo;
  using VULKAN_HPP_NAMESPACE::RenderPassAttachmentBeginInfoKHR;
  using VULKAN_HPP_NAMESPACE::RenderPassCreateInfo2;
  using VULKAN_HPP_NAMESPACE::RenderPassCreateInfo2KHR;
  using VULKAN_HPP_NAMESPACE::SamplerReductionModeCreateInfo;
  using VULKAN_HPP_NAMESPACE::SamplerReductionModeCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::SemaphoreSignalInfo;
  using VULKAN_HPP_NAMESPACE::SemaphoreSignalInfoKHR;
  using VULKAN_HPP_NAMESPACE::SemaphoreTypeCreateInfo;
  using VULKAN_HPP_NAMESPACE::SemaphoreTypeCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::SemaphoreWaitInfo;
  using VULKAN_HPP_NAMESPACE::SemaphoreWaitInfoKHR;
  using VULKAN_HPP_NAMESPACE::SubpassBeginInfo;
  using VULKAN_HPP_NAMESPACE::SubpassBeginInfoKHR;
  using VULKAN_HPP_NAMESPACE::SubpassDependency2;
  using VULKAN_HPP_NAMESPACE::SubpassDependency2KHR;
  using VULKAN_HPP_NAMESPACE::SubpassDescription2;
  using VULKAN_HPP_NAMESPACE::SubpassDescription2KHR;
  using VULKAN_HPP_NAMESPACE::SubpassDescriptionDepthStencilResolve;
  using VULKAN_HPP_NAMESPACE::SubpassDescriptionDepthStencilResolveKHR;
  using VULKAN_HPP_NAMESPACE::SubpassEndInfo;
  using VULKAN_HPP_NAMESPACE::SubpassEndInfoKHR;
  using VULKAN_HPP_NAMESPACE::TimelineSemaphoreSubmitInfo;
  using VULKAN_HPP_NAMESPACE::TimelineSemaphoreSubmitInfoKHR;

  //=== VK_VERSION_1_3 ===
  using VULKAN_HPP_NAMESPACE::BlitImageInfo2;
  using VULKAN_HPP_NAMESPACE::BlitImageInfo2KHR;
  using VULKAN_HPP_NAMESPACE::BufferCopy2;
  using VULKAN_HPP_NAMESPACE::BufferCopy2KHR;
  using VULKAN_HPP_NAMESPACE::BufferImageCopy2;
  using VULKAN_HPP_NAMESPACE::BufferImageCopy2KHR;
  using VULKAN_HPP_NAMESPACE::BufferMemoryBarrier2;
  using VULKAN_HPP_NAMESPACE::BufferMemoryBarrier2KHR;
  using VULKAN_HPP_NAMESPACE::CommandBufferInheritanceRenderingInfo;
  using VULKAN_HPP_NAMESPACE::CommandBufferInheritanceRenderingInfoKHR;
  using VULKAN_HPP_NAMESPACE::CommandBufferSubmitInfo;
  using VULKAN_HPP_NAMESPACE::CommandBufferSubmitInfoKHR;
  using VULKAN_HPP_NAMESPACE::CopyBufferInfo2;
  using VULKAN_HPP_NAMESPACE::CopyBufferInfo2KHR;
  using VULKAN_HPP_NAMESPACE::CopyBufferToImageInfo2;
  using VULKAN_HPP_NAMESPACE::CopyBufferToImageInfo2KHR;
  using VULKAN_HPP_NAMESPACE::CopyImageInfo2;
  using VULKAN_HPP_NAMESPACE::CopyImageInfo2KHR;
  using VULKAN_HPP_NAMESPACE::CopyImageToBufferInfo2;
  using VULKAN_HPP_NAMESPACE::CopyImageToBufferInfo2KHR;
  using VULKAN_HPP_NAMESPACE::DependencyInfo;
  using VULKAN_HPP_NAMESPACE::DependencyInfoKHR;
  using VULKAN_HPP_NAMESPACE::DescriptorPoolInlineUniformBlockCreateInfo;
  using VULKAN_HPP_NAMESPACE::DescriptorPoolInlineUniformBlockCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::DeviceBufferMemoryRequirements;
  using VULKAN_HPP_NAMESPACE::DeviceBufferMemoryRequirementsKHR;
  using VULKAN_HPP_NAMESPACE::DeviceImageMemoryRequirements;
  using VULKAN_HPP_NAMESPACE::DeviceImageMemoryRequirementsKHR;
  using VULKAN_HPP_NAMESPACE::DevicePrivateDataCreateInfo;
  using VULKAN_HPP_NAMESPACE::DevicePrivateDataCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::FormatProperties3;
  using VULKAN_HPP_NAMESPACE::FormatProperties3KHR;
  using VULKAN_HPP_NAMESPACE::ImageBlit2;
  using VULKAN_HPP_NAMESPACE::ImageBlit2KHR;
  using VULKAN_HPP_NAMESPACE::ImageCopy2;
  using VULKAN_HPP_NAMESPACE::ImageCopy2KHR;
  using VULKAN_HPP_NAMESPACE::ImageMemoryBarrier2;
  using VULKAN_HPP_NAMESPACE::ImageMemoryBarrier2KHR;
  using VULKAN_HPP_NAMESPACE::ImageResolve2;
  using VULKAN_HPP_NAMESPACE::ImageResolve2KHR;
  using VULKAN_HPP_NAMESPACE::MemoryBarrier2;
  using VULKAN_HPP_NAMESPACE::MemoryBarrier2KHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDynamicRenderingFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDynamicRenderingFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceImageRobustnessFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceImageRobustnessFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceInlineUniformBlockFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceInlineUniformBlockFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceInlineUniformBlockProperties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceInlineUniformBlockPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance4Features;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance4FeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance4Properties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance4PropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineCreationCacheControlFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineCreationCacheControlFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePrivateDataFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePrivateDataFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderDemoteToHelperInvocationFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderIntegerDotProductFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderIntegerDotProductFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderIntegerDotProductProperties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderIntegerDotProductPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderTerminateInvocationFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderTerminateInvocationFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSubgroupSizeControlFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSubgroupSizeControlFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSubgroupSizeControlProperties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSubgroupSizeControlPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSynchronization2Features;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSynchronization2FeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceTexelBufferAlignmentProperties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceTexelBufferAlignmentPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceTextureCompressionASTCHDRFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceTextureCompressionASTCHDRFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceToolProperties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceToolPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan13Features;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan13Properties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceZeroInitializeWorkgroupMemoryFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PipelineCreationFeedback;
  using VULKAN_HPP_NAMESPACE::PipelineCreationFeedbackCreateInfo;
  using VULKAN_HPP_NAMESPACE::PipelineCreationFeedbackCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::PipelineCreationFeedbackEXT;
  using VULKAN_HPP_NAMESPACE::PipelineRenderingCreateInfo;
  using VULKAN_HPP_NAMESPACE::PipelineRenderingCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::PipelineShaderStageRequiredSubgroupSizeCreateInfo;
  using VULKAN_HPP_NAMESPACE::PipelineShaderStageRequiredSubgroupSizeCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::PrivateDataSlotCreateInfo;
  using VULKAN_HPP_NAMESPACE::PrivateDataSlotCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::RenderingAttachmentInfo;
  using VULKAN_HPP_NAMESPACE::RenderingAttachmentInfoKHR;
  using VULKAN_HPP_NAMESPACE::RenderingInfo;
  using VULKAN_HPP_NAMESPACE::RenderingInfoKHR;
  using VULKAN_HPP_NAMESPACE::ResolveImageInfo2;
  using VULKAN_HPP_NAMESPACE::ResolveImageInfo2KHR;
  using VULKAN_HPP_NAMESPACE::SemaphoreSubmitInfo;
  using VULKAN_HPP_NAMESPACE::SemaphoreSubmitInfoKHR;
  using VULKAN_HPP_NAMESPACE::ShaderRequiredSubgroupSizeCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::SubmitInfo2;
  using VULKAN_HPP_NAMESPACE::SubmitInfo2KHR;
  using VULKAN_HPP_NAMESPACE::WriteDescriptorSetInlineUniformBlock;
  using VULKAN_HPP_NAMESPACE::WriteDescriptorSetInlineUniformBlockEXT;

  //=== VK_KHR_surface ===
  using VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesKHR;
  using VULKAN_HPP_NAMESPACE::SurfaceFormatKHR;

  //=== VK_KHR_swapchain ===
  using VULKAN_HPP_NAMESPACE::AcquireNextImageInfoKHR;
  using VULKAN_HPP_NAMESPACE::BindImageMemorySwapchainInfoKHR;
  using VULKAN_HPP_NAMESPACE::DeviceGroupPresentCapabilitiesKHR;
  using VULKAN_HPP_NAMESPACE::DeviceGroupPresentInfoKHR;
  using VULKAN_HPP_NAMESPACE::DeviceGroupSwapchainCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::ImageSwapchainCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::PresentInfoKHR;
  using VULKAN_HPP_NAMESPACE::SwapchainCreateInfoKHR;

  //=== VK_KHR_display ===
  using VULKAN_HPP_NAMESPACE::DisplayModeCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::DisplayModeParametersKHR;
  using VULKAN_HPP_NAMESPACE::DisplayModePropertiesKHR;
  using VULKAN_HPP_NAMESPACE::DisplayPlaneCapabilitiesKHR;
  using VULKAN_HPP_NAMESPACE::DisplayPlanePropertiesKHR;
  using VULKAN_HPP_NAMESPACE::DisplayPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::DisplaySurfaceCreateInfoKHR;

  //=== VK_KHR_display_swapchain ===
  using VULKAN_HPP_NAMESPACE::DisplayPresentInfoKHR;

#if defined( VK_USE_PLATFORM_XLIB_KHR )
  //=== VK_KHR_xlib_surface ===
  using VULKAN_HPP_NAMESPACE::XlibSurfaceCreateInfoKHR;
#endif /*VK_USE_PLATFORM_XLIB_KHR*/

#if defined( VK_USE_PLATFORM_XCB_KHR )
  //=== VK_KHR_xcb_surface ===
  using VULKAN_HPP_NAMESPACE::XcbSurfaceCreateInfoKHR;
#endif /*VK_USE_PLATFORM_XCB_KHR*/

#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
  //=== VK_KHR_wayland_surface ===
  using VULKAN_HPP_NAMESPACE::WaylandSurfaceCreateInfoKHR;
#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  //=== VK_KHR_android_surface ===
  using VULKAN_HPP_NAMESPACE::AndroidSurfaceCreateInfoKHR;
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_win32_surface ===
  using VULKAN_HPP_NAMESPACE::Win32SurfaceCreateInfoKHR;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_EXT_debug_report ===
  using VULKAN_HPP_NAMESPACE::DebugReportCallbackCreateInfoEXT;

  //=== VK_AMD_rasterization_order ===
  using VULKAN_HPP_NAMESPACE::PipelineRasterizationStateRasterizationOrderAMD;

  //=== VK_EXT_debug_marker ===
  using VULKAN_HPP_NAMESPACE::DebugMarkerMarkerInfoEXT;
  using VULKAN_HPP_NAMESPACE::DebugMarkerObjectNameInfoEXT;
  using VULKAN_HPP_NAMESPACE::DebugMarkerObjectTagInfoEXT;

  //=== VK_KHR_video_queue ===
  using VULKAN_HPP_NAMESPACE::BindVideoSessionMemoryInfoKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoFormatInfoKHR;
  using VULKAN_HPP_NAMESPACE::QueueFamilyQueryResultStatusPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::QueueFamilyVideoPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::VideoBeginCodingInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoCapabilitiesKHR;
  using VULKAN_HPP_NAMESPACE::VideoCodingControlInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEndCodingInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoFormatPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::VideoPictureResourceInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoProfileInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoProfileListInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoReferenceSlotInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoSessionCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoSessionMemoryRequirementsKHR;
  using VULKAN_HPP_NAMESPACE::VideoSessionParametersCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoSessionParametersUpdateInfoKHR;

  //=== VK_KHR_video_decode_queue ===
  using VULKAN_HPP_NAMESPACE::VideoDecodeCapabilitiesKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeUsageInfoKHR;

  //=== VK_NV_dedicated_allocation ===
  using VULKAN_HPP_NAMESPACE::DedicatedAllocationBufferCreateInfoNV;
  using VULKAN_HPP_NAMESPACE::DedicatedAllocationImageCreateInfoNV;
  using VULKAN_HPP_NAMESPACE::DedicatedAllocationMemoryAllocateInfoNV;

  //=== VK_EXT_transform_feedback ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceTransformFeedbackFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceTransformFeedbackPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PipelineRasterizationStateStreamCreateInfoEXT;

  //=== VK_NVX_binary_import ===
  using VULKAN_HPP_NAMESPACE::CuFunctionCreateInfoNVX;
  using VULKAN_HPP_NAMESPACE::CuLaunchInfoNVX;
  using VULKAN_HPP_NAMESPACE::CuModuleCreateInfoNVX;

  //=== VK_NVX_image_view_handle ===
  using VULKAN_HPP_NAMESPACE::ImageViewAddressPropertiesNVX;
  using VULKAN_HPP_NAMESPACE::ImageViewHandleInfoNVX;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_EXT_video_encode_h264 ===
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264CapabilitiesEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264DpbSlotInfoEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264FrameSizeEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264GopRemainingFrameInfoEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264NaluSliceInfoEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264PictureInfoEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264ProfileInfoEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264QpEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264QualityLevelPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264RateControlInfoEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264RateControlLayerInfoEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersAddInfoEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersFeedbackInfoEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersGetInfoEXT;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_EXT_video_encode_h265 ===
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265CapabilitiesEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265DpbSlotInfoEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265FrameSizeEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265GopRemainingFrameInfoEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265NaluSliceSegmentInfoEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265PictureInfoEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265ProfileInfoEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265QpEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265QualityLevelPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265RateControlInfoEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265RateControlLayerInfoEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersAddInfoEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersFeedbackInfoEXT;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersGetInfoEXT;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_KHR_video_decode_h264 ===
  using VULKAN_HPP_NAMESPACE::VideoDecodeH264CapabilitiesKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeH264DpbSlotInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeH264PictureInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeH264ProfileInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeH264SessionParametersAddInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeH264SessionParametersCreateInfoKHR;

  //=== VK_AMD_texture_gather_bias_lod ===
  using VULKAN_HPP_NAMESPACE::TextureLODGatherFormatPropertiesAMD;

  //=== VK_AMD_shader_info ===
  using VULKAN_HPP_NAMESPACE::ShaderResourceUsageAMD;
  using VULKAN_HPP_NAMESPACE::ShaderStatisticsInfoAMD;

  //=== VK_KHR_dynamic_rendering ===
  using VULKAN_HPP_NAMESPACE::AttachmentSampleCountInfoAMD;
  using VULKAN_HPP_NAMESPACE::AttachmentSampleCountInfoNV;
  using VULKAN_HPP_NAMESPACE::MultiviewPerViewAttributesInfoNVX;
  using VULKAN_HPP_NAMESPACE::RenderingFragmentDensityMapAttachmentInfoEXT;
  using VULKAN_HPP_NAMESPACE::RenderingFragmentShadingRateAttachmentInfoKHR;

#if defined( VK_USE_PLATFORM_GGP )
  //=== VK_GGP_stream_descriptor_surface ===
  using VULKAN_HPP_NAMESPACE::StreamDescriptorSurfaceCreateInfoGGP;
#endif /*VK_USE_PLATFORM_GGP*/

  //=== VK_NV_corner_sampled_image ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCornerSampledImageFeaturesNV;

  //=== VK_NV_external_memory_capabilities ===
  using VULKAN_HPP_NAMESPACE::ExternalImageFormatPropertiesNV;

  //=== VK_NV_external_memory ===
  using VULKAN_HPP_NAMESPACE::ExportMemoryAllocateInfoNV;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryImageCreateInfoNV;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_NV_external_memory_win32 ===
  using VULKAN_HPP_NAMESPACE::ExportMemoryWin32HandleInfoNV;
  using VULKAN_HPP_NAMESPACE::ImportMemoryWin32HandleInfoNV;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_NV_win32_keyed_mutex ===
  using VULKAN_HPP_NAMESPACE::Win32KeyedMutexAcquireReleaseInfoNV;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_EXT_validation_flags ===
  using VULKAN_HPP_NAMESPACE::ValidationFlagsEXT;

#if defined( VK_USE_PLATFORM_VI_NN )
  //=== VK_NN_vi_surface ===
  using VULKAN_HPP_NAMESPACE::ViSurfaceCreateInfoNN;
#endif /*VK_USE_PLATFORM_VI_NN*/

  //=== VK_EXT_astc_decode_mode ===
  using VULKAN_HPP_NAMESPACE::ImageViewASTCDecodeModeEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceASTCDecodeFeaturesEXT;

  //=== VK_EXT_pipeline_robustness ===
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineRobustnessFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineRobustnessPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PipelineRobustnessCreateInfoEXT;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_external_memory_win32 ===
  using VULKAN_HPP_NAMESPACE::ExportMemoryWin32HandleInfoKHR;
  using VULKAN_HPP_NAMESPACE::ImportMemoryWin32HandleInfoKHR;
  using VULKAN_HPP_NAMESPACE::MemoryGetWin32HandleInfoKHR;
  using VULKAN_HPP_NAMESPACE::MemoryWin32HandlePropertiesKHR;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_KHR_external_memory_fd ===
  using VULKAN_HPP_NAMESPACE::ImportMemoryFdInfoKHR;
  using VULKAN_HPP_NAMESPACE::MemoryFdPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::MemoryGetFdInfoKHR;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_win32_keyed_mutex ===
  using VULKAN_HPP_NAMESPACE::Win32KeyedMutexAcquireReleaseInfoKHR;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_external_semaphore_win32 ===
  using VULKAN_HPP_NAMESPACE::D3D12FenceSubmitInfoKHR;
  using VULKAN_HPP_NAMESPACE::ExportSemaphoreWin32HandleInfoKHR;
  using VULKAN_HPP_NAMESPACE::ImportSemaphoreWin32HandleInfoKHR;
  using VULKAN_HPP_NAMESPACE::SemaphoreGetWin32HandleInfoKHR;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_KHR_external_semaphore_fd ===
  using VULKAN_HPP_NAMESPACE::ImportSemaphoreFdInfoKHR;
  using VULKAN_HPP_NAMESPACE::SemaphoreGetFdInfoKHR;

  //=== VK_KHR_push_descriptor ===
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePushDescriptorPropertiesKHR;

  //=== VK_EXT_conditional_rendering ===
  using VULKAN_HPP_NAMESPACE::CommandBufferInheritanceConditionalRenderingInfoEXT;
  using VULKAN_HPP_NAMESPACE::ConditionalRenderingBeginInfoEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceConditionalRenderingFeaturesEXT;

  //=== VK_KHR_incremental_present ===
  using VULKAN_HPP_NAMESPACE::PresentRegionKHR;
  using VULKAN_HPP_NAMESPACE::PresentRegionsKHR;
  using VULKAN_HPP_NAMESPACE::RectLayerKHR;

  //=== VK_NV_clip_space_w_scaling ===
  using VULKAN_HPP_NAMESPACE::PipelineViewportWScalingStateCreateInfoNV;
  using VULKAN_HPP_NAMESPACE::ViewportWScalingNV;

  //=== VK_EXT_display_surface_counter ===
  using VULKAN_HPP_NAMESPACE::SurfaceCapabilities2EXT;

  //=== VK_EXT_display_control ===
  using VULKAN_HPP_NAMESPACE::DeviceEventInfoEXT;
  using VULKAN_HPP_NAMESPACE::DisplayEventInfoEXT;
  using VULKAN_HPP_NAMESPACE::DisplayPowerInfoEXT;
  using VULKAN_HPP_NAMESPACE::SwapchainCounterCreateInfoEXT;

  //=== VK_GOOGLE_display_timing ===
  using VULKAN_HPP_NAMESPACE::PastPresentationTimingGOOGLE;
  using VULKAN_HPP_NAMESPACE::PresentTimeGOOGLE;
  using VULKAN_HPP_NAMESPACE::PresentTimesInfoGOOGLE;
  using VULKAN_HPP_NAMESPACE::RefreshCycleDurationGOOGLE;

  //=== VK_NVX_multiview_per_view_attributes ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewPerViewAttributesPropertiesNVX;

  //=== VK_NV_viewport_swizzle ===
  using VULKAN_HPP_NAMESPACE::PipelineViewportSwizzleStateCreateInfoNV;
  using VULKAN_HPP_NAMESPACE::ViewportSwizzleNV;

  //=== VK_EXT_discard_rectangles ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDiscardRectanglePropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PipelineDiscardRectangleStateCreateInfoEXT;

  //=== VK_EXT_conservative_rasterization ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceConservativeRasterizationPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PipelineRasterizationConservativeStateCreateInfoEXT;

  //=== VK_EXT_depth_clip_enable ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClipEnableFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PipelineRasterizationDepthClipStateCreateInfoEXT;

  //=== VK_EXT_hdr_metadata ===
  using VULKAN_HPP_NAMESPACE::HdrMetadataEXT;
  using VULKAN_HPP_NAMESPACE::XYColorEXT;

  //=== VK_KHR_shared_presentable_image ===
  using VULKAN_HPP_NAMESPACE::SharedPresentSurfaceCapabilitiesKHR;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_external_fence_win32 ===
  using VULKAN_HPP_NAMESPACE::ExportFenceWin32HandleInfoKHR;
  using VULKAN_HPP_NAMESPACE::FenceGetWin32HandleInfoKHR;
  using VULKAN_HPP_NAMESPACE::ImportFenceWin32HandleInfoKHR;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_KHR_external_fence_fd ===
  using VULKAN_HPP_NAMESPACE::FenceGetFdInfoKHR;
  using VULKAN_HPP_NAMESPACE::ImportFenceFdInfoKHR;

  //=== VK_KHR_performance_query ===
  using VULKAN_HPP_NAMESPACE::AcquireProfilingLockInfoKHR;
  using VULKAN_HPP_NAMESPACE::PerformanceCounterDescriptionKHR;
  using VULKAN_HPP_NAMESPACE::PerformanceCounterKHR;
  using VULKAN_HPP_NAMESPACE::PerformanceCounterResultKHR;
  using VULKAN_HPP_NAMESPACE::PerformanceQuerySubmitInfoKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePerformanceQueryFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePerformanceQueryPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::QueryPoolPerformanceCreateInfoKHR;

  //=== VK_KHR_get_surface_capabilities2 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSurfaceInfo2KHR;
  using VULKAN_HPP_NAMESPACE::SurfaceCapabilities2KHR;
  using VULKAN_HPP_NAMESPACE::SurfaceFormat2KHR;

  //=== VK_KHR_get_display_properties2 ===
  using VULKAN_HPP_NAMESPACE::DisplayModeProperties2KHR;
  using VULKAN_HPP_NAMESPACE::DisplayPlaneCapabilities2KHR;
  using VULKAN_HPP_NAMESPACE::DisplayPlaneInfo2KHR;
  using VULKAN_HPP_NAMESPACE::DisplayPlaneProperties2KHR;
  using VULKAN_HPP_NAMESPACE::DisplayProperties2KHR;

#if defined( VK_USE_PLATFORM_IOS_MVK )
  //=== VK_MVK_ios_surface ===
  using VULKAN_HPP_NAMESPACE::IOSSurfaceCreateInfoMVK;
#endif /*VK_USE_PLATFORM_IOS_MVK*/

#if defined( VK_USE_PLATFORM_MACOS_MVK )
  //=== VK_MVK_macos_surface ===
  using VULKAN_HPP_NAMESPACE::MacOSSurfaceCreateInfoMVK;
#endif /*VK_USE_PLATFORM_MACOS_MVK*/

  //=== VK_EXT_debug_utils ===
  using VULKAN_HPP_NAMESPACE::DebugUtilsLabelEXT;
  using VULKAN_HPP_NAMESPACE::DebugUtilsMessengerCallbackDataEXT;
  using VULKAN_HPP_NAMESPACE::DebugUtilsMessengerCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::DebugUtilsObjectNameInfoEXT;
  using VULKAN_HPP_NAMESPACE::DebugUtilsObjectTagInfoEXT;

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  //=== VK_ANDROID_external_memory_android_hardware_buffer ===
  using VULKAN_HPP_NAMESPACE::AndroidHardwareBufferFormatProperties2ANDROID;
  using VULKAN_HPP_NAMESPACE::AndroidHardwareBufferFormatPropertiesANDROID;
  using VULKAN_HPP_NAMESPACE::AndroidHardwareBufferPropertiesANDROID;
  using VULKAN_HPP_NAMESPACE::AndroidHardwareBufferUsageANDROID;
  using VULKAN_HPP_NAMESPACE::ExternalFormatANDROID;
  using VULKAN_HPP_NAMESPACE::ImportAndroidHardwareBufferInfoANDROID;
  using VULKAN_HPP_NAMESPACE::MemoryGetAndroidHardwareBufferInfoANDROID;
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_AMDX_shader_enqueue ===
  using VULKAN_HPP_NAMESPACE::DeviceOrHostAddressConstAMDX;
  using VULKAN_HPP_NAMESPACE::DispatchGraphCountInfoAMDX;
  using VULKAN_HPP_NAMESPACE::DispatchGraphInfoAMDX;
  using VULKAN_HPP_NAMESPACE::ExecutionGraphPipelineCreateInfoAMDX;
  using VULKAN_HPP_NAMESPACE::ExecutionGraphPipelineScratchSizeAMDX;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderEnqueueFeaturesAMDX;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderEnqueuePropertiesAMDX;
  using VULKAN_HPP_NAMESPACE::PipelineShaderStageNodeCreateInfoAMDX;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_EXT_sample_locations ===
  using VULKAN_HPP_NAMESPACE::AttachmentSampleLocationsEXT;
  using VULKAN_HPP_NAMESPACE::MultisamplePropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSampleLocationsPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PipelineSampleLocationsStateCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::RenderPassSampleLocationsBeginInfoEXT;
  using VULKAN_HPP_NAMESPACE::SampleLocationEXT;
  using VULKAN_HPP_NAMESPACE::SampleLocationsInfoEXT;
  using VULKAN_HPP_NAMESPACE::SubpassSampleLocationsEXT;

  //=== VK_EXT_blend_operation_advanced ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceBlendOperationAdvancedFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceBlendOperationAdvancedPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PipelineColorBlendAdvancedStateCreateInfoEXT;

  //=== VK_NV_fragment_coverage_to_color ===
  using VULKAN_HPP_NAMESPACE::PipelineCoverageToColorStateCreateInfoNV;

  //=== VK_KHR_acceleration_structure ===
  using VULKAN_HPP_NAMESPACE::AabbPositionsKHR;
  using VULKAN_HPP_NAMESPACE::AabbPositionsNV;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureBuildGeometryInfoKHR;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureBuildRangeInfoKHR;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureBuildSizesInfoKHR;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureDeviceAddressInfoKHR;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryAabbsDataKHR;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryDataKHR;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryInstancesDataKHR;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryKHR;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryTrianglesDataKHR;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureInstanceKHR;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureInstanceNV;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureVersionInfoKHR;
  using VULKAN_HPP_NAMESPACE::CopyAccelerationStructureInfoKHR;
  using VULKAN_HPP_NAMESPACE::CopyAccelerationStructureToMemoryInfoKHR;
  using VULKAN_HPP_NAMESPACE::CopyMemoryToAccelerationStructureInfoKHR;
  using VULKAN_HPP_NAMESPACE::DeviceOrHostAddressConstKHR;
  using VULKAN_HPP_NAMESPACE::DeviceOrHostAddressKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceAccelerationStructureFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceAccelerationStructurePropertiesKHR;
  using VULKAN_HPP_NAMESPACE::TransformMatrixKHR;
  using VULKAN_HPP_NAMESPACE::TransformMatrixNV;
  using VULKAN_HPP_NAMESPACE::WriteDescriptorSetAccelerationStructureKHR;

  //=== VK_KHR_ray_tracing_pipeline ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPipelineFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPipelinePropertiesKHR;
  using VULKAN_HPP_NAMESPACE::RayTracingPipelineCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::RayTracingPipelineInterfaceCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::RayTracingShaderGroupCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::StridedDeviceAddressRegionKHR;
  using VULKAN_HPP_NAMESPACE::TraceRaysIndirectCommandKHR;

  //=== VK_KHR_ray_query ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRayQueryFeaturesKHR;

  //=== VK_NV_framebuffer_mixed_samples ===
  using VULKAN_HPP_NAMESPACE::PipelineCoverageModulationStateCreateInfoNV;

  //=== VK_NV_shader_sm_builtins ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSMBuiltinsFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSMBuiltinsPropertiesNV;

  //=== VK_EXT_image_drm_format_modifier ===
  using VULKAN_HPP_NAMESPACE::DrmFormatModifierProperties2EXT;
  using VULKAN_HPP_NAMESPACE::DrmFormatModifierPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::DrmFormatModifierPropertiesList2EXT;
  using VULKAN_HPP_NAMESPACE::DrmFormatModifierPropertiesListEXT;
  using VULKAN_HPP_NAMESPACE::ImageDrmFormatModifierExplicitCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::ImageDrmFormatModifierListCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::ImageDrmFormatModifierPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceImageDrmFormatModifierInfoEXT;

  //=== VK_EXT_validation_cache ===
  using VULKAN_HPP_NAMESPACE::ShaderModuleValidationCacheCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::ValidationCacheCreateInfoEXT;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_KHR_portability_subset ===
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePortabilitySubsetFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePortabilitySubsetPropertiesKHR;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_NV_shading_rate_image ===
  using VULKAN_HPP_NAMESPACE::CoarseSampleLocationNV;
  using VULKAN_HPP_NAMESPACE::CoarseSampleOrderCustomNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShadingRateImageFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShadingRateImagePropertiesNV;
  using VULKAN_HPP_NAMESPACE::PipelineViewportCoarseSampleOrderStateCreateInfoNV;
  using VULKAN_HPP_NAMESPACE::PipelineViewportShadingRateImageStateCreateInfoNV;
  using VULKAN_HPP_NAMESPACE::ShadingRatePaletteNV;

  //=== VK_NV_ray_tracing ===
  using VULKAN_HPP_NAMESPACE::AccelerationStructureCreateInfoNV;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureInfoNV;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureMemoryRequirementsInfoNV;
  using VULKAN_HPP_NAMESPACE::BindAccelerationStructureMemoryInfoNV;
  using VULKAN_HPP_NAMESPACE::GeometryAABBNV;
  using VULKAN_HPP_NAMESPACE::GeometryDataNV;
  using VULKAN_HPP_NAMESPACE::GeometryNV;
  using VULKAN_HPP_NAMESPACE::GeometryTrianglesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPropertiesNV;
  using VULKAN_HPP_NAMESPACE::RayTracingPipelineCreateInfoNV;
  using VULKAN_HPP_NAMESPACE::RayTracingShaderGroupCreateInfoNV;
  using VULKAN_HPP_NAMESPACE::WriteDescriptorSetAccelerationStructureNV;

  //=== VK_NV_representative_fragment_test ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRepresentativeFragmentTestFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PipelineRepresentativeFragmentTestStateCreateInfoNV;

  //=== VK_EXT_filter_cubic ===
  using VULKAN_HPP_NAMESPACE::FilterCubicImageViewImageFormatPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceImageViewImageFormatInfoEXT;

  //=== VK_EXT_external_memory_host ===
  using VULKAN_HPP_NAMESPACE::ImportMemoryHostPointerInfoEXT;
  using VULKAN_HPP_NAMESPACE::MemoryHostPointerPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalMemoryHostPropertiesEXT;

  //=== VK_KHR_shader_clock ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderClockFeaturesKHR;

  //=== VK_AMD_pipeline_compiler_control ===
  using VULKAN_HPP_NAMESPACE::PipelineCompilerControlCreateInfoAMD;

  //=== VK_EXT_calibrated_timestamps ===
  using VULKAN_HPP_NAMESPACE::CalibratedTimestampInfoEXT;

  //=== VK_AMD_shader_core_properties ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCorePropertiesAMD;

  //=== VK_KHR_video_decode_h265 ===
  using VULKAN_HPP_NAMESPACE::VideoDecodeH265CapabilitiesKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeH265DpbSlotInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeH265PictureInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeH265ProfileInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeH265SessionParametersAddInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeH265SessionParametersCreateInfoKHR;

  //=== VK_KHR_global_priority ===
  using VULKAN_HPP_NAMESPACE::DeviceQueueGlobalPriorityCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::DeviceQueueGlobalPriorityCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceGlobalPriorityQueryFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceGlobalPriorityQueryFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::QueueFamilyGlobalPriorityPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::QueueFamilyGlobalPriorityPropertiesKHR;

  //=== VK_AMD_memory_overallocation_behavior ===
  using VULKAN_HPP_NAMESPACE::DeviceMemoryOverallocationCreateInfoAMD;

  //=== VK_EXT_vertex_attribute_divisor ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeDivisorFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeDivisorPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PipelineVertexInputDivisorStateCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::VertexInputBindingDivisorDescriptionEXT;

#if defined( VK_USE_PLATFORM_GGP )
  //=== VK_GGP_frame_token ===
  using VULKAN_HPP_NAMESPACE::PresentFrameTokenGGP;
#endif /*VK_USE_PLATFORM_GGP*/

  //=== VK_NV_compute_shader_derivatives ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceComputeShaderDerivativesFeaturesNV;

  //=== VK_NV_mesh_shader ===
  using VULKAN_HPP_NAMESPACE::DrawMeshTasksIndirectCommandNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderPropertiesNV;

  //=== VK_NV_shader_image_footprint ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderImageFootprintFeaturesNV;

  //=== VK_NV_scissor_exclusive ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceExclusiveScissorFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PipelineViewportExclusiveScissorStateCreateInfoNV;

  //=== VK_NV_device_diagnostic_checkpoints ===
  using VULKAN_HPP_NAMESPACE::CheckpointDataNV;
  using VULKAN_HPP_NAMESPACE::QueueFamilyCheckpointPropertiesNV;

  //=== VK_INTEL_shader_integer_functions2 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderIntegerFunctions2FeaturesINTEL;

  //=== VK_INTEL_performance_query ===
  using VULKAN_HPP_NAMESPACE::InitializePerformanceApiInfoINTEL;
  using VULKAN_HPP_NAMESPACE::PerformanceConfigurationAcquireInfoINTEL;
  using VULKAN_HPP_NAMESPACE::PerformanceMarkerInfoINTEL;
  using VULKAN_HPP_NAMESPACE::PerformanceOverrideInfoINTEL;
  using VULKAN_HPP_NAMESPACE::PerformanceStreamMarkerInfoINTEL;
  using VULKAN_HPP_NAMESPACE::PerformanceValueDataINTEL;
  using VULKAN_HPP_NAMESPACE::PerformanceValueINTEL;
  using VULKAN_HPP_NAMESPACE::QueryPoolCreateInfoINTEL;
  using VULKAN_HPP_NAMESPACE::QueryPoolPerformanceQueryCreateInfoINTEL;

  //=== VK_EXT_pci_bus_info ===
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePCIBusInfoPropertiesEXT;

  //=== VK_AMD_display_native_hdr ===
  using VULKAN_HPP_NAMESPACE::DisplayNativeHdrSurfaceCapabilitiesAMD;
  using VULKAN_HPP_NAMESPACE::SwapchainDisplayNativeHdrCreateInfoAMD;

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_imagepipe_surface ===
  using VULKAN_HPP_NAMESPACE::ImagePipeSurfaceCreateInfoFUCHSIA;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_surface ===
  using VULKAN_HPP_NAMESPACE::MetalSurfaceCreateInfoEXT;
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_EXT_fragment_density_map ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::RenderPassFragmentDensityMapCreateInfoEXT;

  //=== VK_KHR_fragment_shading_rate ===
  using VULKAN_HPP_NAMESPACE::FragmentShadingRateAttachmentInfoKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRatePropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PipelineFragmentShadingRateStateCreateInfoKHR;

  //=== VK_AMD_shader_core_properties2 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCoreProperties2AMD;

  //=== VK_AMD_device_coherent_memory ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCoherentMemoryFeaturesAMD;

  //=== VK_EXT_shader_image_atomic_int64 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT;

  //=== VK_EXT_memory_budget ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryBudgetPropertiesEXT;

  //=== VK_EXT_memory_priority ===
  using VULKAN_HPP_NAMESPACE::MemoryPriorityAllocateInfoEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryPriorityFeaturesEXT;

  //=== VK_KHR_surface_protected_capabilities ===
  using VULKAN_HPP_NAMESPACE::SurfaceProtectedCapabilitiesKHR;

  //=== VK_NV_dedicated_allocation_image_aliasing ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDedicatedAllocationImageAliasingFeaturesNV;

  //=== VK_EXT_buffer_device_address ===
  using VULKAN_HPP_NAMESPACE::BufferDeviceAddressCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceBufferAddressFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceBufferDeviceAddressFeaturesEXT;

  //=== VK_EXT_validation_features ===
  using VULKAN_HPP_NAMESPACE::ValidationFeaturesEXT;

  //=== VK_KHR_present_wait ===
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePresentWaitFeaturesKHR;

  //=== VK_NV_cooperative_matrix ===
  using VULKAN_HPP_NAMESPACE::CooperativeMatrixPropertiesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixPropertiesNV;

  //=== VK_NV_coverage_reduction_mode ===
  using VULKAN_HPP_NAMESPACE::FramebufferMixedSamplesCombinationNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCoverageReductionModeFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PipelineCoverageReductionStateCreateInfoNV;

  //=== VK_EXT_fragment_shader_interlock ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShaderInterlockFeaturesEXT;

  //=== VK_EXT_ycbcr_image_arrays ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceYcbcrImageArraysFeaturesEXT;

  //=== VK_EXT_provoking_vertex ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceProvokingVertexFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceProvokingVertexPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PipelineRasterizationProvokingVertexStateCreateInfoEXT;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_EXT_full_screen_exclusive ===
  using VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesFullScreenExclusiveEXT;
  using VULKAN_HPP_NAMESPACE::SurfaceFullScreenExclusiveInfoEXT;
  using VULKAN_HPP_NAMESPACE::SurfaceFullScreenExclusiveWin32InfoEXT;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_EXT_headless_surface ===
  using VULKAN_HPP_NAMESPACE::HeadlessSurfaceCreateInfoEXT;

  //=== VK_EXT_line_rasterization ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceLineRasterizationFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceLineRasterizationPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PipelineRasterizationLineStateCreateInfoEXT;

  //=== VK_EXT_shader_atomic_float ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicFloatFeaturesEXT;

  //=== VK_EXT_index_type_uint8 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceIndexTypeUint8FeaturesEXT;

  //=== VK_EXT_extended_dynamic_state ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicStateFeaturesEXT;

  //=== VK_KHR_pipeline_executable_properties ===
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PipelineExecutableInfoKHR;
  using VULKAN_HPP_NAMESPACE::PipelineExecutableInternalRepresentationKHR;
  using VULKAN_HPP_NAMESPACE::PipelineExecutablePropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PipelineExecutableStatisticKHR;
  using VULKAN_HPP_NAMESPACE::PipelineExecutableStatisticValueKHR;
  using VULKAN_HPP_NAMESPACE::PipelineInfoEXT;
  using VULKAN_HPP_NAMESPACE::PipelineInfoKHR;

  //=== VK_EXT_host_image_copy ===
  using VULKAN_HPP_NAMESPACE::CopyImageToImageInfoEXT;
  using VULKAN_HPP_NAMESPACE::CopyImageToMemoryInfoEXT;
  using VULKAN_HPP_NAMESPACE::CopyMemoryToImageInfoEXT;
  using VULKAN_HPP_NAMESPACE::HostImageCopyDevicePerformanceQueryEXT;
  using VULKAN_HPP_NAMESPACE::HostImageLayoutTransitionInfoEXT;
  using VULKAN_HPP_NAMESPACE::ImageToMemoryCopyEXT;
  using VULKAN_HPP_NAMESPACE::MemoryToImageCopyEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceHostImageCopyFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceHostImageCopyPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::SubresourceHostMemcpySizeEXT;

  //=== VK_KHR_map_memory2 ===
  using VULKAN_HPP_NAMESPACE::MemoryMapInfoKHR;
  using VULKAN_HPP_NAMESPACE::MemoryUnmapInfoKHR;

  //=== VK_EXT_shader_atomic_float2 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicFloat2FeaturesEXT;

  //=== VK_EXT_surface_maintenance1 ===
  using VULKAN_HPP_NAMESPACE::SurfacePresentModeCompatibilityEXT;
  using VULKAN_HPP_NAMESPACE::SurfacePresentModeEXT;
  using VULKAN_HPP_NAMESPACE::SurfacePresentScalingCapabilitiesEXT;

  //=== VK_EXT_swapchain_maintenance1 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSwapchainMaintenance1FeaturesEXT;
  using VULKAN_HPP_NAMESPACE::ReleaseSwapchainImagesInfoEXT;
  using VULKAN_HPP_NAMESPACE::SwapchainPresentFenceInfoEXT;
  using VULKAN_HPP_NAMESPACE::SwapchainPresentModeInfoEXT;
  using VULKAN_HPP_NAMESPACE::SwapchainPresentModesCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::SwapchainPresentScalingCreateInfoEXT;

  //=== VK_NV_device_generated_commands ===
  using VULKAN_HPP_NAMESPACE::BindIndexBufferIndirectCommandNV;
  using VULKAN_HPP_NAMESPACE::BindShaderGroupIndirectCommandNV;
  using VULKAN_HPP_NAMESPACE::BindVertexBufferIndirectCommandNV;
  using VULKAN_HPP_NAMESPACE::GeneratedCommandsInfoNV;
  using VULKAN_HPP_NAMESPACE::GeneratedCommandsMemoryRequirementsInfoNV;
  using VULKAN_HPP_NAMESPACE::GraphicsPipelineShaderGroupsCreateInfoNV;
  using VULKAN_HPP_NAMESPACE::GraphicsShaderGroupCreateInfoNV;
  using VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutCreateInfoNV;
  using VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutTokenNV;
  using VULKAN_HPP_NAMESPACE::IndirectCommandsStreamNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceGeneratedCommandsFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceGeneratedCommandsPropertiesNV;
  using VULKAN_HPP_NAMESPACE::SetStateFlagsIndirectCommandNV;

  //=== VK_NV_inherited_viewport_scissor ===
  using VULKAN_HPP_NAMESPACE::CommandBufferInheritanceViewportScissorInfoNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceInheritedViewportScissorFeaturesNV;

  //=== VK_EXT_texel_buffer_alignment ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceTexelBufferAlignmentFeaturesEXT;

  //=== VK_QCOM_render_pass_transform ===
  using VULKAN_HPP_NAMESPACE::CommandBufferInheritanceRenderPassTransformInfoQCOM;
  using VULKAN_HPP_NAMESPACE::RenderPassTransformBeginInfoQCOM;

  //=== VK_EXT_depth_bias_control ===
  using VULKAN_HPP_NAMESPACE::DepthBiasInfoEXT;
  using VULKAN_HPP_NAMESPACE::DepthBiasRepresentationInfoEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthBiasControlFeaturesEXT;

  //=== VK_EXT_device_memory_report ===
  using VULKAN_HPP_NAMESPACE::DeviceDeviceMemoryReportCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::DeviceMemoryReportCallbackDataEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceMemoryReportFeaturesEXT;

  //=== VK_EXT_robustness2 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRobustness2FeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRobustness2PropertiesEXT;

  //=== VK_EXT_custom_border_color ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCustomBorderColorFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCustomBorderColorPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::SamplerCustomBorderColorCreateInfoEXT;

  //=== VK_KHR_pipeline_library ===
  using VULKAN_HPP_NAMESPACE::PipelineLibraryCreateInfoKHR;

  //=== VK_NV_present_barrier ===
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePresentBarrierFeaturesNV;
  using VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesPresentBarrierNV;
  using VULKAN_HPP_NAMESPACE::SwapchainPresentBarrierCreateInfoNV;

  //=== VK_KHR_present_id ===
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePresentIdFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PresentIdKHR;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_KHR_video_encode_queue ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoEncodeQualityLevelInfoKHR;
  using VULKAN_HPP_NAMESPACE::QueryPoolVideoEncodeFeedbackCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeCapabilitiesKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeQualityLevelInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeQualityLevelPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeRateControlInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeRateControlLayerInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeSessionParametersFeedbackInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeSessionParametersGetInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeUsageInfoKHR;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_NV_device_diagnostics_config ===
  using VULKAN_HPP_NAMESPACE::DeviceDiagnosticsConfigCreateInfoNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDiagnosticsConfigFeaturesNV;

  //=== VK_NV_low_latency ===
  using VULKAN_HPP_NAMESPACE::QueryLowLatencySupportNV;

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_objects ===
  using VULKAN_HPP_NAMESPACE::ExportMetalBufferInfoEXT;
  using VULKAN_HPP_NAMESPACE::ExportMetalCommandQueueInfoEXT;
  using VULKAN_HPP_NAMESPACE::ExportMetalDeviceInfoEXT;
  using VULKAN_HPP_NAMESPACE::ExportMetalIOSurfaceInfoEXT;
  using VULKAN_HPP_NAMESPACE::ExportMetalObjectCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::ExportMetalObjectsInfoEXT;
  using VULKAN_HPP_NAMESPACE::ExportMetalSharedEventInfoEXT;
  using VULKAN_HPP_NAMESPACE::ExportMetalTextureInfoEXT;
  using VULKAN_HPP_NAMESPACE::ImportMetalBufferInfoEXT;
  using VULKAN_HPP_NAMESPACE::ImportMetalIOSurfaceInfoEXT;
  using VULKAN_HPP_NAMESPACE::ImportMetalSharedEventInfoEXT;
  using VULKAN_HPP_NAMESPACE::ImportMetalTextureInfoEXT;
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_KHR_synchronization2 ===
  using VULKAN_HPP_NAMESPACE::CheckpointData2NV;
  using VULKAN_HPP_NAMESPACE::QueueFamilyCheckpointProperties2NV;

  //=== VK_EXT_descriptor_buffer ===
  using VULKAN_HPP_NAMESPACE::AccelerationStructureCaptureDescriptorDataInfoEXT;
  using VULKAN_HPP_NAMESPACE::BufferCaptureDescriptorDataInfoEXT;
  using VULKAN_HPP_NAMESPACE::DescriptorAddressInfoEXT;
  using VULKAN_HPP_NAMESPACE::DescriptorBufferBindingInfoEXT;
  using VULKAN_HPP_NAMESPACE::DescriptorBufferBindingPushDescriptorBufferHandleEXT;
  using VULKAN_HPP_NAMESPACE::DescriptorDataEXT;
  using VULKAN_HPP_NAMESPACE::DescriptorGetInfoEXT;
  using VULKAN_HPP_NAMESPACE::ImageCaptureDescriptorDataInfoEXT;
  using VULKAN_HPP_NAMESPACE::ImageViewCaptureDescriptorDataInfoEXT;
  using VULKAN_HPP_NAMESPACE::OpaqueCaptureDescriptorDataCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorBufferDensityMapPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorBufferFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorBufferPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::SamplerCaptureDescriptorDataInfoEXT;

  //=== VK_EXT_graphics_pipeline_library ===
  using VULKAN_HPP_NAMESPACE::GraphicsPipelineLibraryCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT;

  //=== VK_AMD_shader_early_and_late_fragment_tests ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderEarlyAndLateFragmentTestsFeaturesAMD;

  //=== VK_KHR_fragment_shader_barycentric ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShaderBarycentricFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShaderBarycentricPropertiesKHR;

  //=== VK_KHR_shader_subgroup_uniform_control_flow ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR;

  //=== VK_NV_fragment_shading_rate_enums ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateEnumsFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateEnumsPropertiesNV;
  using VULKAN_HPP_NAMESPACE::PipelineFragmentShadingRateEnumStateCreateInfoNV;

  //=== VK_NV_ray_tracing_motion_blur ===
  using VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryMotionTrianglesDataNV;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureMatrixMotionInstanceNV;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureMotionInfoNV;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureMotionInstanceDataNV;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureMotionInstanceNV;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureSRTMotionInstanceNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingMotionBlurFeaturesNV;
  using VULKAN_HPP_NAMESPACE::SRTDataNV;

  //=== VK_EXT_mesh_shader ===
  using VULKAN_HPP_NAMESPACE::DrawMeshTasksIndirectCommandEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderPropertiesEXT;

  //=== VK_EXT_ycbcr_2plane_444_formats ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT;

  //=== VK_EXT_fragment_density_map2 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMap2FeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMap2PropertiesEXT;

  //=== VK_QCOM_rotated_copy_commands ===
  using VULKAN_HPP_NAMESPACE::CopyCommandTransformInfoQCOM;

  //=== VK_KHR_workgroup_memory_explicit_layout ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR;

  //=== VK_EXT_image_compression_control ===
  using VULKAN_HPP_NAMESPACE::ImageCompressionControlEXT;
  using VULKAN_HPP_NAMESPACE::ImageCompressionPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceImageCompressionControlFeaturesEXT;

  //=== VK_EXT_attachment_feedback_loop_layout ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT;

  //=== VK_EXT_4444_formats ===
  using VULKAN_HPP_NAMESPACE::PhysicalDevice4444FormatsFeaturesEXT;

  //=== VK_EXT_device_fault ===
  using VULKAN_HPP_NAMESPACE::DeviceFaultAddressInfoEXT;
  using VULKAN_HPP_NAMESPACE::DeviceFaultCountsEXT;
  using VULKAN_HPP_NAMESPACE::DeviceFaultInfoEXT;
  using VULKAN_HPP_NAMESPACE::DeviceFaultVendorBinaryHeaderVersionOneEXT;
  using VULKAN_HPP_NAMESPACE::DeviceFaultVendorInfoEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFaultFeaturesEXT;

  //=== VK_EXT_rgba10x6_formats ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRGBA10X6FormatsFeaturesEXT;

#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
  //=== VK_EXT_directfb_surface ===
  using VULKAN_HPP_NAMESPACE::DirectFBSurfaceCreateInfoEXT;
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

  //=== VK_EXT_vertex_input_dynamic_state ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexInputDynamicStateFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::VertexInputAttributeDescription2EXT;
  using VULKAN_HPP_NAMESPACE::VertexInputBindingDescription2EXT;

  //=== VK_EXT_physical_device_drm ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDrmPropertiesEXT;

  //=== VK_EXT_device_address_binding_report ===
  using VULKAN_HPP_NAMESPACE::DeviceAddressBindingCallbackDataEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceAddressBindingReportFeaturesEXT;

  //=== VK_EXT_depth_clip_control ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClipControlFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PipelineViewportDepthClipControlCreateInfoEXT;

  //=== VK_EXT_primitive_topology_list_restart ===
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT;

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_external_memory ===
  using VULKAN_HPP_NAMESPACE::ImportMemoryZirconHandleInfoFUCHSIA;
  using VULKAN_HPP_NAMESPACE::MemoryGetZirconHandleInfoFUCHSIA;
  using VULKAN_HPP_NAMESPACE::MemoryZirconHandlePropertiesFUCHSIA;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_external_semaphore ===
  using VULKAN_HPP_NAMESPACE::ImportSemaphoreZirconHandleInfoFUCHSIA;
  using VULKAN_HPP_NAMESPACE::SemaphoreGetZirconHandleInfoFUCHSIA;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_buffer_collection ===
  using VULKAN_HPP_NAMESPACE::BufferCollectionBufferCreateInfoFUCHSIA;
  using VULKAN_HPP_NAMESPACE::BufferCollectionConstraintsInfoFUCHSIA;
  using VULKAN_HPP_NAMESPACE::BufferCollectionCreateInfoFUCHSIA;
  using VULKAN_HPP_NAMESPACE::BufferCollectionImageCreateInfoFUCHSIA;
  using VULKAN_HPP_NAMESPACE::BufferCollectionPropertiesFUCHSIA;
  using VULKAN_HPP_NAMESPACE::BufferConstraintsInfoFUCHSIA;
  using VULKAN_HPP_NAMESPACE::ImageConstraintsInfoFUCHSIA;
  using VULKAN_HPP_NAMESPACE::ImageFormatConstraintsInfoFUCHSIA;
  using VULKAN_HPP_NAMESPACE::ImportMemoryBufferCollectionFUCHSIA;
  using VULKAN_HPP_NAMESPACE::SysmemColorSpaceFUCHSIA;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

  //=== VK_HUAWEI_subpass_shading ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSubpassShadingFeaturesHUAWEI;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSubpassShadingPropertiesHUAWEI;
  using VULKAN_HPP_NAMESPACE::SubpassShadingPipelineCreateInfoHUAWEI;

  //=== VK_HUAWEI_invocation_mask ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceInvocationMaskFeaturesHUAWEI;

  //=== VK_NV_external_memory_rdma ===
  using VULKAN_HPP_NAMESPACE::MemoryGetRemoteAddressInfoNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalMemoryRDMAFeaturesNV;

  //=== VK_EXT_pipeline_properties ===
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePipelinePropertiesFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PipelinePropertiesIdentifierEXT;

  //=== VK_EXT_multisampled_render_to_single_sampled ===
  using VULKAN_HPP_NAMESPACE::MultisampledRenderToSingleSampledInfoEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMultisampledRenderToSingleSampledFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::SubpassResolvePerformanceQueryEXT;

  //=== VK_EXT_extended_dynamic_state2 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicState2FeaturesEXT;

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
  //=== VK_QNX_screen_surface ===
  using VULKAN_HPP_NAMESPACE::ScreenSurfaceCreateInfoQNX;
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

  //=== VK_EXT_color_write_enable ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceColorWriteEnableFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PipelineColorWriteCreateInfoEXT;

  //=== VK_EXT_primitives_generated_query ===
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePrimitivesGeneratedQueryFeaturesEXT;

  //=== VK_KHR_ray_tracing_maintenance1 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingMaintenance1FeaturesKHR;
  using VULKAN_HPP_NAMESPACE::TraceRaysIndirectCommand2KHR;

  //=== VK_EXT_image_view_min_lod ===
  using VULKAN_HPP_NAMESPACE::ImageViewMinLodCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceImageViewMinLodFeaturesEXT;

  //=== VK_EXT_multi_draw ===
  using VULKAN_HPP_NAMESPACE::MultiDrawIndexedInfoEXT;
  using VULKAN_HPP_NAMESPACE::MultiDrawInfoEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiDrawFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiDrawPropertiesEXT;

  //=== VK_EXT_image_2d_view_of_3d ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceImage2DViewOf3DFeaturesEXT;

  //=== VK_EXT_shader_tile_image ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderTileImageFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderTileImagePropertiesEXT;

  //=== VK_EXT_opacity_micromap ===
  using VULKAN_HPP_NAMESPACE::AccelerationStructureTrianglesOpacityMicromapEXT;
  using VULKAN_HPP_NAMESPACE::CopyMemoryToMicromapInfoEXT;
  using VULKAN_HPP_NAMESPACE::CopyMicromapInfoEXT;
  using VULKAN_HPP_NAMESPACE::CopyMicromapToMemoryInfoEXT;
  using VULKAN_HPP_NAMESPACE::MicromapBuildInfoEXT;
  using VULKAN_HPP_NAMESPACE::MicromapBuildSizesInfoEXT;
  using VULKAN_HPP_NAMESPACE::MicromapCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::MicromapTriangleEXT;
  using VULKAN_HPP_NAMESPACE::MicromapUsageEXT;
  using VULKAN_HPP_NAMESPACE::MicromapVersionInfoEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceOpacityMicromapFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceOpacityMicromapPropertiesEXT;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_NV_displacement_micromap ===
  using VULKAN_HPP_NAMESPACE::AccelerationStructureTrianglesDisplacementMicromapNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDisplacementMicromapFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDisplacementMicromapPropertiesNV;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_HUAWEI_cluster_culling_shader ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceClusterCullingShaderFeaturesHUAWEI;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceClusterCullingShaderPropertiesHUAWEI;

  //=== VK_EXT_border_color_swizzle ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceBorderColorSwizzleFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::SamplerBorderColorComponentMappingCreateInfoEXT;

  //=== VK_EXT_pageable_device_local_memory ===
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT;

  //=== VK_ARM_shader_core_properties ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCorePropertiesARM;

  //=== VK_EXT_image_sliced_view_of_3d ===
  using VULKAN_HPP_NAMESPACE::ImageViewSlicedCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceImageSlicedViewOf3DFeaturesEXT;

  //=== VK_VALVE_descriptor_set_host_mapping ===
  using VULKAN_HPP_NAMESPACE::DescriptorSetBindingReferenceVALVE;
  using VULKAN_HPP_NAMESPACE::DescriptorSetLayoutHostMappingInfoVALVE;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorSetHostMappingFeaturesVALVE;

  //=== VK_EXT_depth_clamp_zero_one ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClampZeroOneFeaturesEXT;

  //=== VK_EXT_non_seamless_cube_map ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceNonSeamlessCubeMapFeaturesEXT;

  //=== VK_QCOM_fragment_density_map_offset ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapOffsetFeaturesQCOM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapOffsetPropertiesQCOM;
  using VULKAN_HPP_NAMESPACE::SubpassFragmentDensityMapOffsetEndInfoQCOM;

  //=== VK_NV_copy_memory_indirect ===
  using VULKAN_HPP_NAMESPACE::CopyMemoryIndirectCommandNV;
  using VULKAN_HPP_NAMESPACE::CopyMemoryToImageIndirectCommandNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCopyMemoryIndirectFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCopyMemoryIndirectPropertiesNV;

  //=== VK_NV_memory_decompression ===
  using VULKAN_HPP_NAMESPACE::DecompressMemoryRegionNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryDecompressionFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryDecompressionPropertiesNV;

  //=== VK_NV_device_generated_commands_compute ===
  using VULKAN_HPP_NAMESPACE::BindPipelineIndirectCommandNV;
  using VULKAN_HPP_NAMESPACE::ComputePipelineIndirectBufferInfoNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceGeneratedCommandsComputeFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PipelineIndirectDeviceAddressInfoNV;

  //=== VK_NV_linear_color_attachment ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceLinearColorAttachmentFeaturesNV;

  //=== VK_EXT_image_compression_control_swapchain ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceImageCompressionControlSwapchainFeaturesEXT;

  //=== VK_QCOM_image_processing ===
  using VULKAN_HPP_NAMESPACE::ImageViewSampleWeightCreateInfoQCOM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessingFeaturesQCOM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessingPropertiesQCOM;

  //=== VK_EXT_external_memory_acquire_unmodified ===
  using VULKAN_HPP_NAMESPACE::ExternalMemoryAcquireUnmodifiedEXT;

  //=== VK_EXT_extended_dynamic_state3 ===
  using VULKAN_HPP_NAMESPACE::ColorBlendAdvancedEXT;
  using VULKAN_HPP_NAMESPACE::ColorBlendEquationEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicState3FeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicState3PropertiesEXT;

  //=== VK_EXT_subpass_merge_feedback ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSubpassMergeFeedbackFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::RenderPassCreationControlEXT;
  using VULKAN_HPP_NAMESPACE::RenderPassCreationFeedbackCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::RenderPassCreationFeedbackInfoEXT;
  using VULKAN_HPP_NAMESPACE::RenderPassSubpassFeedbackCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::RenderPassSubpassFeedbackInfoEXT;

  //=== VK_LUNARG_direct_driver_loading ===
  using VULKAN_HPP_NAMESPACE::DirectDriverLoadingInfoLUNARG;
  using VULKAN_HPP_NAMESPACE::DirectDriverLoadingListLUNARG;

  //=== VK_EXT_shader_module_identifier ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderModuleIdentifierFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderModuleIdentifierPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PipelineShaderStageModuleIdentifierCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::ShaderModuleIdentifierEXT;

  //=== VK_EXT_rasterization_order_attachment_access ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesARM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT;

  //=== VK_NV_optical_flow ===
  using VULKAN_HPP_NAMESPACE::OpticalFlowExecuteInfoNV;
  using VULKAN_HPP_NAMESPACE::OpticalFlowImageFormatInfoNV;
  using VULKAN_HPP_NAMESPACE::OpticalFlowImageFormatPropertiesNV;
  using VULKAN_HPP_NAMESPACE::OpticalFlowSessionCreateInfoNV;
  using VULKAN_HPP_NAMESPACE::OpticalFlowSessionCreatePrivateDataInfoNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceOpticalFlowFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceOpticalFlowPropertiesNV;

  //=== VK_EXT_legacy_dithering ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceLegacyDitheringFeaturesEXT;

  //=== VK_EXT_pipeline_protected_access ===
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineProtectedAccessFeaturesEXT;

  //=== VK_KHR_maintenance5 ===
  using VULKAN_HPP_NAMESPACE::BufferUsageFlags2CreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::DeviceImageSubresourceInfoKHR;
  using VULKAN_HPP_NAMESPACE::ImageSubresource2EXT;
  using VULKAN_HPP_NAMESPACE::ImageSubresource2KHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance5FeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance5PropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PipelineCreateFlags2CreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::RenderingAreaInfoKHR;
  using VULKAN_HPP_NAMESPACE::SubresourceLayout2EXT;
  using VULKAN_HPP_NAMESPACE::SubresourceLayout2KHR;

  //=== VK_KHR_ray_tracing_position_fetch ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPositionFetchFeaturesKHR;

  //=== VK_EXT_shader_object ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderObjectFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderObjectPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::ShaderCreateInfoEXT;

  //=== VK_QCOM_tile_properties ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceTilePropertiesFeaturesQCOM;
  using VULKAN_HPP_NAMESPACE::TilePropertiesQCOM;

  //=== VK_SEC_amigo_profiling ===
  using VULKAN_HPP_NAMESPACE::AmigoProfilingSubmitInfoSEC;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceAmigoProfilingFeaturesSEC;

  //=== VK_QCOM_multiview_per_view_viewports ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewPerViewViewportsFeaturesQCOM;

  //=== VK_NV_ray_tracing_invocation_reorder ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingInvocationReorderFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingInvocationReorderPropertiesNV;

  //=== VK_EXT_mutable_descriptor_type ===
  using VULKAN_HPP_NAMESPACE::MutableDescriptorTypeCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::MutableDescriptorTypeCreateInfoVALVE;
  using VULKAN_HPP_NAMESPACE::MutableDescriptorTypeListEXT;
  using VULKAN_HPP_NAMESPACE::MutableDescriptorTypeListVALVE;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMutableDescriptorTypeFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMutableDescriptorTypeFeaturesVALVE;

  //=== VK_ARM_shader_core_builtins ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCoreBuiltinsFeaturesARM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCoreBuiltinsPropertiesARM;

  //=== VK_EXT_pipeline_library_group_handles ===
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineLibraryGroupHandlesFeaturesEXT;

  //=== VK_EXT_dynamic_rendering_unused_attachments ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT;

  //=== VK_KHR_cooperative_matrix ===
  using VULKAN_HPP_NAMESPACE::CooperativeMatrixPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixPropertiesKHR;

  //=== VK_QCOM_multiview_per_view_render_areas ===
  using VULKAN_HPP_NAMESPACE::MultiviewPerViewRenderAreasRenderPassBeginInfoQCOM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewPerViewRenderAreasFeaturesQCOM;

  //=== VK_EXT_attachment_feedback_loop_dynamic_state ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT;

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
  //=== VK_QNX_external_memory_screen_buffer ===
  using VULKAN_HPP_NAMESPACE::ExternalFormatQNX;
  using VULKAN_HPP_NAMESPACE::ImportScreenBufferInfoQNX;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalMemoryScreenBufferFeaturesQNX;
  using VULKAN_HPP_NAMESPACE::ScreenBufferFormatPropertiesQNX;
  using VULKAN_HPP_NAMESPACE::ScreenBufferPropertiesQNX;
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

  //===============
  //=== HANDLEs ===
  //===============

  using VULKAN_HPP_NAMESPACE::isVulkanHandleType;

  //=== VK_VERSION_1_0 ===
  using VULKAN_HPP_NAMESPACE::Buffer;
  using VULKAN_HPP_NAMESPACE::BufferView;
  using VULKAN_HPP_NAMESPACE::CommandBuffer;
  using VULKAN_HPP_NAMESPACE::CommandPool;
  using VULKAN_HPP_NAMESPACE::DescriptorPool;
  using VULKAN_HPP_NAMESPACE::DescriptorSet;
  using VULKAN_HPP_NAMESPACE::DescriptorSetLayout;
  using VULKAN_HPP_NAMESPACE::Device;
  using VULKAN_HPP_NAMESPACE::DeviceMemory;
  using VULKAN_HPP_NAMESPACE::Event;
  using VULKAN_HPP_NAMESPACE::Fence;
  using VULKAN_HPP_NAMESPACE::Framebuffer;
  using VULKAN_HPP_NAMESPACE::Image;
  using VULKAN_HPP_NAMESPACE::ImageView;
  using VULKAN_HPP_NAMESPACE::Instance;
  using VULKAN_HPP_NAMESPACE::PhysicalDevice;
  using VULKAN_HPP_NAMESPACE::Pipeline;
  using VULKAN_HPP_NAMESPACE::PipelineCache;
  using VULKAN_HPP_NAMESPACE::PipelineLayout;
  using VULKAN_HPP_NAMESPACE::QueryPool;
  using VULKAN_HPP_NAMESPACE::Queue;
  using VULKAN_HPP_NAMESPACE::RenderPass;
  using VULKAN_HPP_NAMESPACE::Sampler;
  using VULKAN_HPP_NAMESPACE::Semaphore;
  using VULKAN_HPP_NAMESPACE::ShaderModule;

  //=== VK_VERSION_1_1 ===
  using VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplate;
  using VULKAN_HPP_NAMESPACE::SamplerYcbcrConversion;

  //=== VK_VERSION_1_3 ===
  using VULKAN_HPP_NAMESPACE::PrivateDataSlot;

  //=== VK_KHR_surface ===
  using VULKAN_HPP_NAMESPACE::SurfaceKHR;

  //=== VK_KHR_swapchain ===
  using VULKAN_HPP_NAMESPACE::SwapchainKHR;

  //=== VK_KHR_display ===
  using VULKAN_HPP_NAMESPACE::DisplayKHR;
  using VULKAN_HPP_NAMESPACE::DisplayModeKHR;

  //=== VK_EXT_debug_report ===
  using VULKAN_HPP_NAMESPACE::DebugReportCallbackEXT;

  //=== VK_KHR_video_queue ===
  using VULKAN_HPP_NAMESPACE::VideoSessionKHR;
  using VULKAN_HPP_NAMESPACE::VideoSessionParametersKHR;

  //=== VK_NVX_binary_import ===
  using VULKAN_HPP_NAMESPACE::CuFunctionNVX;
  using VULKAN_HPP_NAMESPACE::CuModuleNVX;

  //=== VK_EXT_debug_utils ===
  using VULKAN_HPP_NAMESPACE::DebugUtilsMessengerEXT;

  //=== VK_KHR_acceleration_structure ===
  using VULKAN_HPP_NAMESPACE::AccelerationStructureKHR;

  //=== VK_EXT_validation_cache ===
  using VULKAN_HPP_NAMESPACE::ValidationCacheEXT;

  //=== VK_NV_ray_tracing ===
  using VULKAN_HPP_NAMESPACE::AccelerationStructureNV;

  //=== VK_INTEL_performance_query ===
  using VULKAN_HPP_NAMESPACE::PerformanceConfigurationINTEL;

  //=== VK_KHR_deferred_host_operations ===
  using VULKAN_HPP_NAMESPACE::DeferredOperationKHR;

  //=== VK_NV_device_generated_commands ===
  using VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutNV;

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_buffer_collection ===
  using VULKAN_HPP_NAMESPACE::BufferCollectionFUCHSIA;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

  //=== VK_EXT_opacity_micromap ===
  using VULKAN_HPP_NAMESPACE::MicromapEXT;

  //=== VK_NV_optical_flow ===
  using VULKAN_HPP_NAMESPACE::OpticalFlowSessionNV;

  //=== VK_EXT_shader_object ===
  using VULKAN_HPP_NAMESPACE::ShaderEXT;

  //======================
  //=== UNIQUE HANDLEs ===
  //======================

#if !defined( VULKAN_HPP_NO_SMART_HANDLE )

  //=== VK_VERSION_1_0 ===
  using VULKAN_HPP_NAMESPACE::UniqueBuffer;
  using VULKAN_HPP_NAMESPACE::UniqueBufferView;
  using VULKAN_HPP_NAMESPACE::UniqueCommandBuffer;
  using VULKAN_HPP_NAMESPACE::UniqueCommandPool;
  using VULKAN_HPP_NAMESPACE::UniqueDescriptorPool;
  using VULKAN_HPP_NAMESPACE::UniqueDescriptorSet;
  using VULKAN_HPP_NAMESPACE::UniqueDescriptorSetLayout;
  using VULKAN_HPP_NAMESPACE::UniqueDevice;
  using VULKAN_HPP_NAMESPACE::UniqueDeviceMemory;
  using VULKAN_HPP_NAMESPACE::UniqueEvent;
  using VULKAN_HPP_NAMESPACE::UniqueFence;
  using VULKAN_HPP_NAMESPACE::UniqueFramebuffer;
  using VULKAN_HPP_NAMESPACE::UniqueImage;
  using VULKAN_HPP_NAMESPACE::UniqueImageView;
  using VULKAN_HPP_NAMESPACE::UniqueInstance;
  using VULKAN_HPP_NAMESPACE::UniquePipeline;
  using VULKAN_HPP_NAMESPACE::UniquePipelineCache;
  using VULKAN_HPP_NAMESPACE::UniquePipelineLayout;
  using VULKAN_HPP_NAMESPACE::UniqueQueryPool;
  using VULKAN_HPP_NAMESPACE::UniqueRenderPass;
  using VULKAN_HPP_NAMESPACE::UniqueSampler;
  using VULKAN_HPP_NAMESPACE::UniqueSemaphore;
  using VULKAN_HPP_NAMESPACE::UniqueShaderModule;

  //=== VK_VERSION_1_1 ===
  using VULKAN_HPP_NAMESPACE::UniqueDescriptorUpdateTemplate;
  using VULKAN_HPP_NAMESPACE::UniqueSamplerYcbcrConversion;

  //=== VK_VERSION_1_3 ===
  using VULKAN_HPP_NAMESPACE::UniquePrivateDataSlot;

  //=== VK_KHR_surface ===
  using VULKAN_HPP_NAMESPACE::UniqueSurfaceKHR;

  //=== VK_KHR_swapchain ===
  using VULKAN_HPP_NAMESPACE::UniqueSwapchainKHR;

  //=== VK_EXT_debug_report ===
  using VULKAN_HPP_NAMESPACE::UniqueDebugReportCallbackEXT;

  //=== VK_KHR_video_queue ===
  using VULKAN_HPP_NAMESPACE::UniqueVideoSessionKHR;
  using VULKAN_HPP_NAMESPACE::UniqueVideoSessionParametersKHR;

  //=== VK_NVX_binary_import ===
  using VULKAN_HPP_NAMESPACE::UniqueCuFunctionNVX;
  using VULKAN_HPP_NAMESPACE::UniqueCuModuleNVX;

  //=== VK_EXT_debug_utils ===
  using VULKAN_HPP_NAMESPACE::UniqueDebugUtilsMessengerEXT;

  //=== VK_KHR_acceleration_structure ===
  using VULKAN_HPP_NAMESPACE::UniqueAccelerationStructureKHR;

  //=== VK_EXT_validation_cache ===
  using VULKAN_HPP_NAMESPACE::UniqueValidationCacheEXT;

  //=== VK_NV_ray_tracing ===
  using VULKAN_HPP_NAMESPACE::UniqueAccelerationStructureNV;

  //=== VK_KHR_deferred_host_operations ===
  using VULKAN_HPP_NAMESPACE::UniqueDeferredOperationKHR;

  //=== VK_NV_device_generated_commands ===
  using VULKAN_HPP_NAMESPACE::UniqueIndirectCommandsLayoutNV;

#  if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_buffer_collection ===
  using VULKAN_HPP_NAMESPACE::UniqueBufferCollectionFUCHSIA;
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

  //=== VK_EXT_opacity_micromap ===
  using VULKAN_HPP_NAMESPACE::UniqueMicromapEXT;

  //=== VK_NV_optical_flow ===
  using VULKAN_HPP_NAMESPACE::UniqueOpticalFlowSessionNV;

  //=== VK_EXT_shader_object ===
  using VULKAN_HPP_NAMESPACE::UniqueHandleTraits;
  using VULKAN_HPP_NAMESPACE::UniqueShaderEXT;
#endif /*VULKAN_HPP_NO_SMART_HANDLE*/

  //===========================
  //=== COMMAND Definitions ===
  //===========================
  using VULKAN_HPP_NAMESPACE::createInstance;
  using VULKAN_HPP_NAMESPACE::enumerateInstanceExtensionProperties;
  using VULKAN_HPP_NAMESPACE::enumerateInstanceLayerProperties;
  using VULKAN_HPP_NAMESPACE::enumerateInstanceVersion;

#if !defined( VULKAN_HPP_NO_SMART_HANDLE )
  using VULKAN_HPP_NAMESPACE::createInstanceUnique;
#endif /*VULKAN_HPP_NO_SMART_HANDLE*/

#if !defined( VULKAN_HPP_DISABLE_ENHANCED_MODE )
  using VULKAN_HPP_NAMESPACE::StructExtends;
#endif /*VULKAN_HPP_DISABLE_ENHANCED_MODE*/

#if defined( VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL )
  using VULKAN_HPP_NAMESPACE::DynamicLoader;
#endif /*VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL*/

  using VULKAN_HPP_NAMESPACE::DispatchLoaderDynamic;

  //=====================
  //=== Format Traits ===
  //=====================
  using VULKAN_HPP_NAMESPACE::blockExtent;
  using VULKAN_HPP_NAMESPACE::blockSize;
  using VULKAN_HPP_NAMESPACE::compatibilityClass;
  using VULKAN_HPP_NAMESPACE::componentBits;
  using VULKAN_HPP_NAMESPACE::componentCount;
  using VULKAN_HPP_NAMESPACE::componentName;
  using VULKAN_HPP_NAMESPACE::componentNumericFormat;
  using VULKAN_HPP_NAMESPACE::componentPlaneIndex;
  using VULKAN_HPP_NAMESPACE::componentsAreCompressed;
  using VULKAN_HPP_NAMESPACE::compressionScheme;
  using VULKAN_HPP_NAMESPACE::isCompressed;
  using VULKAN_HPP_NAMESPACE::packed;
  using VULKAN_HPP_NAMESPACE::planeCompatibleFormat;
  using VULKAN_HPP_NAMESPACE::planeCount;
  using VULKAN_HPP_NAMESPACE::planeHeightDivisor;
  using VULKAN_HPP_NAMESPACE::planeWidthDivisor;
  using VULKAN_HPP_NAMESPACE::texelsPerBlock;

  //======================================
  //=== Extension inspection functions ===
  //======================================
  using VULKAN_HPP_NAMESPACE::getDeprecatedExtensions;
  using VULKAN_HPP_NAMESPACE::getDeviceExtensions;
  using VULKAN_HPP_NAMESPACE::getExtensionDepends;
  using VULKAN_HPP_NAMESPACE::getExtensionDeprecatedBy;
  using VULKAN_HPP_NAMESPACE::getExtensionObsoletedBy;
  using VULKAN_HPP_NAMESPACE::getExtensionPromotedTo;
  using VULKAN_HPP_NAMESPACE::getInstanceExtensions;
  using VULKAN_HPP_NAMESPACE::getObsoletedExtensions;
  using VULKAN_HPP_NAMESPACE::getPromotedExtensions;
  using VULKAN_HPP_NAMESPACE::isDeprecatedExtension;
  using VULKAN_HPP_NAMESPACE::isDeviceExtension;
  using VULKAN_HPP_NAMESPACE::isInstanceExtension;
  using VULKAN_HPP_NAMESPACE::isObsoletedExtension;
  using VULKAN_HPP_NAMESPACE::isPromotedExtension;

  namespace VULKAN_HPP_RAII_NAMESPACE
  {
    //======================
    //=== RAII HARDCODED ===
    //======================

    using VULKAN_HPP_RAII_NAMESPACE::Context;
    using VULKAN_HPP_RAII_NAMESPACE::ContextDispatcher;
    using VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher;
    using VULKAN_HPP_RAII_NAMESPACE::exchange;
    using VULKAN_HPP_RAII_NAMESPACE::InstanceDispatcher;

    //====================
    //=== RAII HANDLEs ===
    //====================

    //=== VK_VERSION_1_0 ===
    using VULKAN_HPP_RAII_NAMESPACE::Buffer;
    using VULKAN_HPP_RAII_NAMESPACE::BufferView;
    using VULKAN_HPP_RAII_NAMESPACE::CommandBuffer;
    using VULKAN_HPP_RAII_NAMESPACE::CommandBuffers;
    using VULKAN_HPP_RAII_NAMESPACE::CommandPool;
    using VULKAN_HPP_RAII_NAMESPACE::DescriptorPool;
    using VULKAN_HPP_RAII_NAMESPACE::DescriptorSet;
    using VULKAN_HPP_RAII_NAMESPACE::DescriptorSetLayout;
    using VULKAN_HPP_RAII_NAMESPACE::DescriptorSets;
    using VULKAN_HPP_RAII_NAMESPACE::Device;
    using VULKAN_HPP_RAII_NAMESPACE::DeviceMemory;
    using VULKAN_HPP_RAII_NAMESPACE::Event;
    using VULKAN_HPP_RAII_NAMESPACE::Fence;
    using VULKAN_HPP_RAII_NAMESPACE::Framebuffer;
    using VULKAN_HPP_RAII_NAMESPACE::Image;
    using VULKAN_HPP_RAII_NAMESPACE::ImageView;
    using VULKAN_HPP_RAII_NAMESPACE::Instance;
    using VULKAN_HPP_RAII_NAMESPACE::PhysicalDevice;
    using VULKAN_HPP_RAII_NAMESPACE::PhysicalDevices;
    using VULKAN_HPP_RAII_NAMESPACE::Pipeline;
    using VULKAN_HPP_RAII_NAMESPACE::PipelineCache;
    using VULKAN_HPP_RAII_NAMESPACE::PipelineLayout;
    using VULKAN_HPP_RAII_NAMESPACE::Pipelines;
    using VULKAN_HPP_RAII_NAMESPACE::QueryPool;
    using VULKAN_HPP_RAII_NAMESPACE::Queue;
    using VULKAN_HPP_RAII_NAMESPACE::RenderPass;
    using VULKAN_HPP_RAII_NAMESPACE::Sampler;
    using VULKAN_HPP_RAII_NAMESPACE::Semaphore;
    using VULKAN_HPP_RAII_NAMESPACE::ShaderModule;

    //=== VK_VERSION_1_1 ===
    using VULKAN_HPP_RAII_NAMESPACE::DescriptorUpdateTemplate;
    using VULKAN_HPP_RAII_NAMESPACE::SamplerYcbcrConversion;

    //=== VK_VERSION_1_3 ===
    using VULKAN_HPP_RAII_NAMESPACE::PrivateDataSlot;

    //=== VK_KHR_surface ===
    using VULKAN_HPP_RAII_NAMESPACE::SurfaceKHR;

    //=== VK_KHR_swapchain ===
    using VULKAN_HPP_RAII_NAMESPACE::SwapchainKHR;
    using VULKAN_HPP_RAII_NAMESPACE::SwapchainKHRs;

    //=== VK_KHR_display ===
    using VULKAN_HPP_RAII_NAMESPACE::DisplayKHR;
    using VULKAN_HPP_RAII_NAMESPACE::DisplayKHRs;
    using VULKAN_HPP_RAII_NAMESPACE::DisplayModeKHR;

    //=== VK_EXT_debug_report ===
    using VULKAN_HPP_RAII_NAMESPACE::DebugReportCallbackEXT;

    //=== VK_KHR_video_queue ===
    using VULKAN_HPP_RAII_NAMESPACE::VideoSessionKHR;
    using VULKAN_HPP_RAII_NAMESPACE::VideoSessionParametersKHR;

    //=== VK_NVX_binary_import ===
    using VULKAN_HPP_RAII_NAMESPACE::CuFunctionNVX;
    using VULKAN_HPP_RAII_NAMESPACE::CuModuleNVX;

    //=== VK_EXT_debug_utils ===
    using VULKAN_HPP_RAII_NAMESPACE::DebugUtilsMessengerEXT;

    //=== VK_KHR_acceleration_structure ===
    using VULKAN_HPP_RAII_NAMESPACE::AccelerationStructureKHR;

    //=== VK_EXT_validation_cache ===
    using VULKAN_HPP_RAII_NAMESPACE::ValidationCacheEXT;

    //=== VK_NV_ray_tracing ===
    using VULKAN_HPP_RAII_NAMESPACE::AccelerationStructureNV;

    //=== VK_INTEL_performance_query ===
    using VULKAN_HPP_RAII_NAMESPACE::PerformanceConfigurationINTEL;

    //=== VK_KHR_deferred_host_operations ===
    using VULKAN_HPP_RAII_NAMESPACE::DeferredOperationKHR;

    //=== VK_NV_device_generated_commands ===
    using VULKAN_HPP_RAII_NAMESPACE::IndirectCommandsLayoutNV;

#if defined( VK_USE_PLATFORM_FUCHSIA )
    //=== VK_FUCHSIA_buffer_collection ===
    using VULKAN_HPP_RAII_NAMESPACE::BufferCollectionFUCHSIA;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

    //=== VK_EXT_opacity_micromap ===
    using VULKAN_HPP_RAII_NAMESPACE::MicromapEXT;

    //=== VK_NV_optical_flow ===
    using VULKAN_HPP_RAII_NAMESPACE::OpticalFlowSessionNV;

    //=== VK_EXT_shader_object ===
    using VULKAN_HPP_RAII_NAMESPACE::ShaderEXT;
    using VULKAN_HPP_RAII_NAMESPACE::ShaderEXTs;

  }  // namespace VULKAN_HPP_RAII_NAMESPACE
}  // namespace VULKAN_HPP_NAMESPACE
