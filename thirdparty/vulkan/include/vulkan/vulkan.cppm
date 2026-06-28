// Copyright 2015-2025 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//

// This header is generated from the Khronos Vulkan XML API Registry.

module;

#define VULKAN_HPP_CXX_MODULE 1

#include <vulkan/vulkan_hpp_macros.hpp>

#if !defined( VULKAN_HPP_CXX_MODULE_EXPERIMENTAL_WARNING )
#  define VULKAN_HPP_CXX_MODULE_EXPERIMENTAL_WARNING \
    "\n\tThe Vulkan-Hpp C++ named module is experimental. It is subject to change without prior notice.\n" \
  "\tTo silence this warning, define the VULKAN_HPP_CXX_MODULE_EXPERIMENTAL_WARNING macro.\n" \
  "\tFor feedback, go to: https://github.com/KhronosGroup/Vulkan-Hpp/issues"

VULKAN_HPP_COMPILE_WARNING( VULKAN_HPP_CXX_MODULE_EXPERIMENTAL_WARNING )
#endif

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_extension_inspection.hpp>
#include <vulkan/vulkan_format_traits.hpp>
#include <vulkan/vulkan_hash.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_shared.hpp>

export module vulkan;
export import :video;
export import std;

export namespace VULKAN_HPP_NAMESPACE
{
  //=====================================
  //=== HARDCODED TYPEs AND FUNCTIONs ===
  //=====================================
  using VULKAN_HPP_NAMESPACE::ArrayWrapper1D;
  using VULKAN_HPP_NAMESPACE::ArrayWrapper2D;
  using VULKAN_HPP_NAMESPACE::Flags;
  using VULKAN_HPP_NAMESPACE::FlagTraits;

  namespace detail
  {
    using VULKAN_HPP_NAMESPACE::detail::DispatchLoaderBase;
    using VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic;
#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC == 1
    using VULKAN_HPP_NAMESPACE::detail::defaultDispatchLoaderDynamic;
#endif
#if !defined( VK_NO_PROTOTYPES )
    using VULKAN_HPP_NAMESPACE::detail::DispatchLoaderStatic;
    using VULKAN_HPP_NAMESPACE::detail::getDispatchLoaderStatic;
#endif /*VK_NO_PROTOTYPES*/
    using VULKAN_HPP_NAMESPACE::detail::createResultValueType;
    using VULKAN_HPP_NAMESPACE::detail::isDispatchLoader;
    using VULKAN_HPP_NAMESPACE::detail::resultCheck;
  }  // namespace detail
#if !defined( VULKAN_HPP_DISABLE_ENHANCED_MODE )
  namespace VULKAN_HPP_RAII_NAMESPACE
  {
    using VULKAN_HPP_RAII_NAMESPACE::operator==;
    using VULKAN_HPP_RAII_NAMESPACE::operator!=;
#  if defined( VULKAN_HPP_HAS_SPACESHIP_OPERATOR )
    using VULKAN_HPP_RAII_NAMESPACE::operator<=>;
#  else
    using VULKAN_HPP_RAII_NAMESPACE::operator<;
#  endif
  }  // namespace VULKAN_HPP_RAII_NAMESPACE
#endif
  using VULKAN_HPP_NAMESPACE::operator&;
  using VULKAN_HPP_NAMESPACE::operator|;
  using VULKAN_HPP_NAMESPACE::operator^;
  using VULKAN_HPP_NAMESPACE::operator~;
  using VULKAN_HPP_NAMESPACE::operator<;
  using VULKAN_HPP_NAMESPACE::operator<=;
  using VULKAN_HPP_NAMESPACE::operator>;
  using VULKAN_HPP_NAMESPACE::operator>=;
  using VULKAN_HPP_NAMESPACE::operator==;
  using VULKAN_HPP_NAMESPACE::operator!=;
  using VULKAN_HPP_DEFAULT_DISPATCHER_TYPE;

#if !defined( VULKAN_HPP_DISABLE_ENHANCED_MODE )
  using VULKAN_HPP_NAMESPACE::ArrayProxy;
  using VULKAN_HPP_NAMESPACE::ArrayProxyNoTemporaries;
  using VULKAN_HPP_NAMESPACE::Optional;
  using VULKAN_HPP_NAMESPACE::StridedArrayProxy;
  using VULKAN_HPP_NAMESPACE::StructureChain;
#endif /*VULKAN_HPP_DISABLE_ENHANCED_MODE*/

#if !defined( VULKAN_HPP_NO_SMART_HANDLE )
  namespace detail
  {
    using VULKAN_HPP_NAMESPACE::detail::ObjectDestroy;
    using VULKAN_HPP_NAMESPACE::detail::ObjectDestroyShared;
    using VULKAN_HPP_NAMESPACE::detail::ObjectFree;
    using VULKAN_HPP_NAMESPACE::detail::ObjectFreeShared;
    using VULKAN_HPP_NAMESPACE::detail::ObjectRelease;
    using VULKAN_HPP_NAMESPACE::detail::ObjectReleaseShared;
    using VULKAN_HPP_NAMESPACE::detail::PoolFree;
    using VULKAN_HPP_NAMESPACE::detail::PoolFreeShared;
  }  // namespace detail

  using VULKAN_HPP_NAMESPACE::SharedHandle;
  using VULKAN_HPP_NAMESPACE::UniqueHandle;
#endif /*VULKAN_HPP_NO_SMART_HANDLE*/

  using VULKAN_HPP_NAMESPACE::exchange;

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
  using VULKAN_HPP_NAMESPACE::DeviceQueueCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::DeviceQueueCreateFlags;
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
  using VULKAN_HPP_NAMESPACE::PipelineCacheCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::PipelineCacheCreateFlags;
  using VULKAN_HPP_NAMESPACE::PipelineCacheHeaderVersion;
  using VULKAN_HPP_NAMESPACE::PipelineColorBlendStateCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::PipelineColorBlendStateCreateFlags;
  using VULKAN_HPP_NAMESPACE::PipelineCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::PipelineCreateFlags;
  using VULKAN_HPP_NAMESPACE::PipelineDepthStencilStateCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::PipelineDepthStencilStateCreateFlags;
  using VULKAN_HPP_NAMESPACE::PipelineDynamicStateCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::PipelineDynamicStateCreateFlags;
  using VULKAN_HPP_NAMESPACE::PipelineInputAssemblyStateCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::PipelineInputAssemblyStateCreateFlags;
  using VULKAN_HPP_NAMESPACE::PipelineLayoutCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::PipelineLayoutCreateFlags;
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
  using VULKAN_HPP_NAMESPACE::CommandPoolTrimFlagsKHR;
  using VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateCreateFlags;
  using VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateCreateFlagsKHR;
  using VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateType;
  using VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateTypeKHR;
  using VULKAN_HPP_NAMESPACE::ExternalFenceFeatureFlagBits;
  using VULKAN_HPP_NAMESPACE::ExternalFenceFeatureFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::ExternalFenceFeatureFlags;
  using VULKAN_HPP_NAMESPACE::ExternalFenceFeatureFlagsKHR;
  using VULKAN_HPP_NAMESPACE::ExternalFenceHandleTypeFlagBits;
  using VULKAN_HPP_NAMESPACE::ExternalFenceHandleTypeFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::ExternalFenceHandleTypeFlags;
  using VULKAN_HPP_NAMESPACE::ExternalFenceHandleTypeFlagsKHR;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryFeatureFlagBits;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryFeatureFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryFeatureFlags;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryFeatureFlagsKHR;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryHandleTypeFlagBits;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryHandleTypeFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryHandleTypeFlags;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryHandleTypeFlagsKHR;
  using VULKAN_HPP_NAMESPACE::ExternalSemaphoreFeatureFlagBits;
  using VULKAN_HPP_NAMESPACE::ExternalSemaphoreFeatureFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::ExternalSemaphoreFeatureFlags;
  using VULKAN_HPP_NAMESPACE::ExternalSemaphoreFeatureFlagsKHR;
  using VULKAN_HPP_NAMESPACE::ExternalSemaphoreHandleTypeFlagBits;
  using VULKAN_HPP_NAMESPACE::ExternalSemaphoreHandleTypeFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::ExternalSemaphoreHandleTypeFlags;
  using VULKAN_HPP_NAMESPACE::ExternalSemaphoreHandleTypeFlagsKHR;
  using VULKAN_HPP_NAMESPACE::FenceImportFlagBits;
  using VULKAN_HPP_NAMESPACE::FenceImportFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::FenceImportFlags;
  using VULKAN_HPP_NAMESPACE::FenceImportFlagsKHR;
  using VULKAN_HPP_NAMESPACE::MemoryAllocateFlagBits;
  using VULKAN_HPP_NAMESPACE::MemoryAllocateFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::MemoryAllocateFlags;
  using VULKAN_HPP_NAMESPACE::MemoryAllocateFlagsKHR;
  using VULKAN_HPP_NAMESPACE::PeerMemoryFeatureFlagBits;
  using VULKAN_HPP_NAMESPACE::PeerMemoryFeatureFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::PeerMemoryFeatureFlags;
  using VULKAN_HPP_NAMESPACE::PeerMemoryFeatureFlagsKHR;
  using VULKAN_HPP_NAMESPACE::PointClippingBehavior;
  using VULKAN_HPP_NAMESPACE::PointClippingBehaviorKHR;
  using VULKAN_HPP_NAMESPACE::SamplerYcbcrModelConversion;
  using VULKAN_HPP_NAMESPACE::SamplerYcbcrModelConversionKHR;
  using VULKAN_HPP_NAMESPACE::SamplerYcbcrRange;
  using VULKAN_HPP_NAMESPACE::SamplerYcbcrRangeKHR;
  using VULKAN_HPP_NAMESPACE::SemaphoreImportFlagBits;
  using VULKAN_HPP_NAMESPACE::SemaphoreImportFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::SemaphoreImportFlags;
  using VULKAN_HPP_NAMESPACE::SemaphoreImportFlagsKHR;
  using VULKAN_HPP_NAMESPACE::SubgroupFeatureFlagBits;
  using VULKAN_HPP_NAMESPACE::SubgroupFeatureFlags;
  using VULKAN_HPP_NAMESPACE::TessellationDomainOrigin;
  using VULKAN_HPP_NAMESPACE::TessellationDomainOriginKHR;

  //=== VK_VERSION_1_2 ===
  using VULKAN_HPP_NAMESPACE::DescriptorBindingFlagBits;
  using VULKAN_HPP_NAMESPACE::DescriptorBindingFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::DescriptorBindingFlags;
  using VULKAN_HPP_NAMESPACE::DescriptorBindingFlagsEXT;
  using VULKAN_HPP_NAMESPACE::DriverId;
  using VULKAN_HPP_NAMESPACE::DriverIdKHR;
  using VULKAN_HPP_NAMESPACE::ResolveModeFlagBits;
  using VULKAN_HPP_NAMESPACE::ResolveModeFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::ResolveModeFlags;
  using VULKAN_HPP_NAMESPACE::ResolveModeFlagsKHR;
  using VULKAN_HPP_NAMESPACE::SamplerReductionMode;
  using VULKAN_HPP_NAMESPACE::SamplerReductionModeEXT;
  using VULKAN_HPP_NAMESPACE::SemaphoreType;
  using VULKAN_HPP_NAMESPACE::SemaphoreTypeKHR;
  using VULKAN_HPP_NAMESPACE::SemaphoreWaitFlagBits;
  using VULKAN_HPP_NAMESPACE::SemaphoreWaitFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::SemaphoreWaitFlags;
  using VULKAN_HPP_NAMESPACE::SemaphoreWaitFlagsKHR;
  using VULKAN_HPP_NAMESPACE::ShaderFloatControlsIndependence;
  using VULKAN_HPP_NAMESPACE::ShaderFloatControlsIndependenceKHR;

  //=== VK_VERSION_1_3 ===
  using VULKAN_HPP_NAMESPACE::AccessFlagBits2;
  using VULKAN_HPP_NAMESPACE::AccessFlagBits2KHR;
  using VULKAN_HPP_NAMESPACE::AccessFlags2;
  using VULKAN_HPP_NAMESPACE::AccessFlags2KHR;
  using VULKAN_HPP_NAMESPACE::FormatFeatureFlagBits2;
  using VULKAN_HPP_NAMESPACE::FormatFeatureFlagBits2KHR;
  using VULKAN_HPP_NAMESPACE::FormatFeatureFlags2;
  using VULKAN_HPP_NAMESPACE::FormatFeatureFlags2KHR;
  using VULKAN_HPP_NAMESPACE::PipelineCreationFeedbackFlagBits;
  using VULKAN_HPP_NAMESPACE::PipelineCreationFeedbackFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::PipelineCreationFeedbackFlags;
  using VULKAN_HPP_NAMESPACE::PipelineCreationFeedbackFlagsEXT;
  using VULKAN_HPP_NAMESPACE::PipelineStageFlagBits2;
  using VULKAN_HPP_NAMESPACE::PipelineStageFlagBits2KHR;
  using VULKAN_HPP_NAMESPACE::PipelineStageFlags2;
  using VULKAN_HPP_NAMESPACE::PipelineStageFlags2KHR;
  using VULKAN_HPP_NAMESPACE::PrivateDataSlotCreateFlagBits;
  using VULKAN_HPP_NAMESPACE::PrivateDataSlotCreateFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::PrivateDataSlotCreateFlags;
  using VULKAN_HPP_NAMESPACE::PrivateDataSlotCreateFlagsEXT;
  using VULKAN_HPP_NAMESPACE::RenderingFlagBits;
  using VULKAN_HPP_NAMESPACE::RenderingFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::RenderingFlags;
  using VULKAN_HPP_NAMESPACE::RenderingFlagsKHR;
  using VULKAN_HPP_NAMESPACE::SubmitFlagBits;
  using VULKAN_HPP_NAMESPACE::SubmitFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::SubmitFlags;
  using VULKAN_HPP_NAMESPACE::SubmitFlagsKHR;
  using VULKAN_HPP_NAMESPACE::ToolPurposeFlagBits;
  using VULKAN_HPP_NAMESPACE::ToolPurposeFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::ToolPurposeFlags;
  using VULKAN_HPP_NAMESPACE::ToolPurposeFlagsEXT;

  //=== VK_VERSION_1_4 ===
  using VULKAN_HPP_NAMESPACE::BufferUsageFlagBits2;
  using VULKAN_HPP_NAMESPACE::BufferUsageFlagBits2KHR;
  using VULKAN_HPP_NAMESPACE::BufferUsageFlags2;
  using VULKAN_HPP_NAMESPACE::BufferUsageFlags2KHR;
  using VULKAN_HPP_NAMESPACE::HostImageCopyFlagBits;
  using VULKAN_HPP_NAMESPACE::HostImageCopyFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::HostImageCopyFlags;
  using VULKAN_HPP_NAMESPACE::HostImageCopyFlagsEXT;
  using VULKAN_HPP_NAMESPACE::LineRasterizationMode;
  using VULKAN_HPP_NAMESPACE::LineRasterizationModeEXT;
  using VULKAN_HPP_NAMESPACE::LineRasterizationModeKHR;
  using VULKAN_HPP_NAMESPACE::MemoryUnmapFlagBits;
  using VULKAN_HPP_NAMESPACE::MemoryUnmapFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::MemoryUnmapFlags;
  using VULKAN_HPP_NAMESPACE::MemoryUnmapFlagsKHR;
  using VULKAN_HPP_NAMESPACE::PipelineCreateFlagBits2;
  using VULKAN_HPP_NAMESPACE::PipelineCreateFlagBits2KHR;
  using VULKAN_HPP_NAMESPACE::PipelineCreateFlags2;
  using VULKAN_HPP_NAMESPACE::PipelineCreateFlags2KHR;
  using VULKAN_HPP_NAMESPACE::PipelineRobustnessBufferBehavior;
  using VULKAN_HPP_NAMESPACE::PipelineRobustnessBufferBehaviorEXT;
  using VULKAN_HPP_NAMESPACE::PipelineRobustnessImageBehavior;
  using VULKAN_HPP_NAMESPACE::PipelineRobustnessImageBehaviorEXT;
  using VULKAN_HPP_NAMESPACE::QueueGlobalPriority;
  using VULKAN_HPP_NAMESPACE::QueueGlobalPriorityEXT;
  using VULKAN_HPP_NAMESPACE::QueueGlobalPriorityKHR;

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

  //=== VK_KHR_video_encode_h264 ===
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264CapabilityFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264CapabilityFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264RateControlFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264RateControlFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264StdFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264StdFlagsKHR;

  //=== VK_KHR_video_encode_h265 ===
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265CapabilityFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265CapabilityFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265CtbSizeFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265CtbSizeFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265RateControlFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265RateControlFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265StdFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265StdFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265TransformBlockSizeFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265TransformBlockSizeFlagsKHR;

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
  using VULKAN_HPP_NAMESPACE::BuildAccelerationStructureFlagsNV;
  using VULKAN_HPP_NAMESPACE::BuildAccelerationStructureModeKHR;
  using VULKAN_HPP_NAMESPACE::CopyAccelerationStructureModeKHR;
  using VULKAN_HPP_NAMESPACE::CopyAccelerationStructureModeNV;
  using VULKAN_HPP_NAMESPACE::GeometryFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::GeometryFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::GeometryFlagsKHR;
  using VULKAN_HPP_NAMESPACE::GeometryFlagsNV;
  using VULKAN_HPP_NAMESPACE::GeometryInstanceFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::GeometryInstanceFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::GeometryInstanceFlagsKHR;
  using VULKAN_HPP_NAMESPACE::GeometryInstanceFlagsNV;
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

  //=== VK_AMD_memory_overallocation_behavior ===
  using VULKAN_HPP_NAMESPACE::MemoryOverallocationBehaviorAMD;

  //=== VK_EXT_present_timing ===
  using VULKAN_HPP_NAMESPACE::PastPresentationTimingFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::PastPresentationTimingFlagsEXT;
  using VULKAN_HPP_NAMESPACE::PresentStageFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::PresentStageFlagsEXT;
  using VULKAN_HPP_NAMESPACE::PresentTimingInfoFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::PresentTimingInfoFlagsEXT;

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

  //=== VK_KHR_pipeline_executable_properties ===
  using VULKAN_HPP_NAMESPACE::PipelineExecutableStatisticFormatKHR;

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

  //=== VK_NV_device_diagnostics_config ===
  using VULKAN_HPP_NAMESPACE::DeviceDiagnosticsConfigFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::DeviceDiagnosticsConfigFlagsNV;

  //=== VK_QCOM_tile_shading ===
  using VULKAN_HPP_NAMESPACE::TileShadingRenderPassFlagBitsQCOM;
  using VULKAN_HPP_NAMESPACE::TileShadingRenderPassFlagsQCOM;

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_objects ===
  using VULKAN_HPP_NAMESPACE::ExportMetalObjectTypeFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::ExportMetalObjectTypeFlagsEXT;
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_EXT_graphics_pipeline_library ===
  using VULKAN_HPP_NAMESPACE::GraphicsPipelineLibraryFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::GraphicsPipelineLibraryFlagsEXT;

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

  //=== VK_EXT_frame_boundary ===
  using VULKAN_HPP_NAMESPACE::FrameBoundaryFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::FrameBoundaryFlagsEXT;

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
  //=== VK_QNX_screen_surface ===
  using VULKAN_HPP_NAMESPACE::ScreenSurfaceCreateFlagBitsQNX;
  using VULKAN_HPP_NAMESPACE::ScreenSurfaceCreateFlagsQNX;
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

  //=== VK_VALVE_video_encode_rgb_conversion ===
  using VULKAN_HPP_NAMESPACE::VideoEncodeRgbChromaOffsetFlagBitsVALVE;
  using VULKAN_HPP_NAMESPACE::VideoEncodeRgbChromaOffsetFlagsVALVE;
  using VULKAN_HPP_NAMESPACE::VideoEncodeRgbModelConversionFlagBitsVALVE;
  using VULKAN_HPP_NAMESPACE::VideoEncodeRgbModelConversionFlagsVALVE;
  using VULKAN_HPP_NAMESPACE::VideoEncodeRgbRangeCompressionFlagBitsVALVE;
  using VULKAN_HPP_NAMESPACE::VideoEncodeRgbRangeCompressionFlagsVALVE;

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

  //=== VK_ARM_scheduling_controls ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSchedulingControlsFlagBitsARM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSchedulingControlsFlagsARM;

  //=== VK_NV_ray_tracing_linear_swept_spheres ===
  using VULKAN_HPP_NAMESPACE::RayTracingLssIndexingModeNV;
  using VULKAN_HPP_NAMESPACE::RayTracingLssPrimitiveEndCapsModeNV;

  //=== VK_EXT_subpass_merge_feedback ===
  using VULKAN_HPP_NAMESPACE::SubpassMergeStatusEXT;

  //=== VK_LUNARG_direct_driver_loading ===
  using VULKAN_HPP_NAMESPACE::DirectDriverLoadingFlagBitsLUNARG;
  using VULKAN_HPP_NAMESPACE::DirectDriverLoadingFlagsLUNARG;
  using VULKAN_HPP_NAMESPACE::DirectDriverLoadingModeLUNARG;

  //=== VK_ARM_tensors ===
  using VULKAN_HPP_NAMESPACE::TensorCreateFlagBitsARM;
  using VULKAN_HPP_NAMESPACE::TensorCreateFlagsARM;
  using VULKAN_HPP_NAMESPACE::TensorTilingARM;
  using VULKAN_HPP_NAMESPACE::TensorUsageFlagBitsARM;
  using VULKAN_HPP_NAMESPACE::TensorUsageFlagsARM;
  using VULKAN_HPP_NAMESPACE::TensorViewCreateFlagBitsARM;
  using VULKAN_HPP_NAMESPACE::TensorViewCreateFlagsARM;

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

  //=== VK_AMD_anti_lag ===
  using VULKAN_HPP_NAMESPACE::AntiLagModeAMD;
  using VULKAN_HPP_NAMESPACE::AntiLagStageAMD;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_AMDX_dense_geometry_format ===
  using VULKAN_HPP_NAMESPACE::CompressedTriangleFormatAMDX;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_EXT_shader_object ===
  using VULKAN_HPP_NAMESPACE::ShaderCodeTypeEXT;
  using VULKAN_HPP_NAMESPACE::ShaderCreateFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::ShaderCreateFlagsEXT;

  //=== VK_KHR_surface_maintenance1 ===
  using VULKAN_HPP_NAMESPACE::PresentGravityFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::PresentGravityFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::PresentGravityFlagsEXT;
  using VULKAN_HPP_NAMESPACE::PresentGravityFlagsKHR;
  using VULKAN_HPP_NAMESPACE::PresentScalingFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::PresentScalingFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::PresentScalingFlagsEXT;
  using VULKAN_HPP_NAMESPACE::PresentScalingFlagsKHR;

  //=== VK_NV_cooperative_vector ===
  using VULKAN_HPP_NAMESPACE::ComponentTypeKHR;
  using VULKAN_HPP_NAMESPACE::ComponentTypeNV;
  using VULKAN_HPP_NAMESPACE::CooperativeVectorMatrixLayoutNV;

  //=== VK_EXT_layer_settings ===
  using VULKAN_HPP_NAMESPACE::LayerSettingTypeEXT;

  //=== VK_NV_low_latency2 ===
  using VULKAN_HPP_NAMESPACE::LatencyMarkerNV;
  using VULKAN_HPP_NAMESPACE::OutOfBandQueueTypeNV;

  //=== VK_KHR_cooperative_matrix ===
  using VULKAN_HPP_NAMESPACE::ScopeKHR;
  using VULKAN_HPP_NAMESPACE::ScopeNV;

  //=== VK_ARM_data_graph ===
  using VULKAN_HPP_NAMESPACE::DataGraphPipelineDispatchFlagBitsARM;
  using VULKAN_HPP_NAMESPACE::DataGraphPipelineDispatchFlagsARM;
  using VULKAN_HPP_NAMESPACE::DataGraphPipelinePropertyARM;
  using VULKAN_HPP_NAMESPACE::DataGraphPipelineSessionBindPointARM;
  using VULKAN_HPP_NAMESPACE::DataGraphPipelineSessionBindPointTypeARM;
  using VULKAN_HPP_NAMESPACE::DataGraphPipelineSessionCreateFlagBitsARM;
  using VULKAN_HPP_NAMESPACE::DataGraphPipelineSessionCreateFlagsARM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDataGraphOperationTypeARM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDataGraphProcessingEngineTypeARM;

  //=== VK_KHR_video_encode_av1 ===
  using VULKAN_HPP_NAMESPACE::VideoEncodeAV1CapabilityFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeAV1CapabilityFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeAV1PredictionModeKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeAV1RateControlFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeAV1RateControlFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeAV1RateControlGroupKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeAV1StdFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeAV1StdFlagsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeAV1SuperblockSizeFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeAV1SuperblockSizeFlagsKHR;

  //=== VK_QCOM_image_processing2 ===
  using VULKAN_HPP_NAMESPACE::BlockMatchWindowCompareModeQCOM;

  //=== VK_QCOM_filter_cubic_weights ===
  using VULKAN_HPP_NAMESPACE::CubicFilterWeightsQCOM;

  //=== VK_MSFT_layered_driver ===
  using VULKAN_HPP_NAMESPACE::LayeredDriverUnderlyingApiMSFT;

  //=== VK_KHR_calibrated_timestamps ===
  using VULKAN_HPP_NAMESPACE::TimeDomainEXT;
  using VULKAN_HPP_NAMESPACE::TimeDomainKHR;

  //=== VK_KHR_copy_memory_indirect ===
  using VULKAN_HPP_NAMESPACE::AddressCopyFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::AddressCopyFlagsKHR;

  //=== VK_EXT_memory_decompression ===
  using VULKAN_HPP_NAMESPACE::MemoryDecompressionMethodFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::MemoryDecompressionMethodFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::MemoryDecompressionMethodFlagsEXT;
  using VULKAN_HPP_NAMESPACE::MemoryDecompressionMethodFlagsNV;

  //=== VK_NV_display_stereo ===
  using VULKAN_HPP_NAMESPACE::DisplaySurfaceStereoTypeNV;

  //=== VK_KHR_video_encode_intra_refresh ===
  using VULKAN_HPP_NAMESPACE::VideoEncodeIntraRefreshModeFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeIntraRefreshModeFlagsKHR;

  //=== VK_KHR_maintenance7 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceLayeredApiKHR;

  //=== VK_NV_cluster_acceleration_structure ===
  using VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureAddressResolutionFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureAddressResolutionFlagsNV;
  using VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureClusterFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureClusterFlagsNV;
  using VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureGeometryFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureGeometryFlagsNV;
  using VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureIndexFormatFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureIndexFormatFlagsNV;
  using VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureOpModeNV;
  using VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureOpTypeNV;
  using VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureTypeNV;

  //=== VK_NV_partitioned_acceleration_structure ===
  using VULKAN_HPP_NAMESPACE::PartitionedAccelerationStructureInstanceFlagBitsNV;
  using VULKAN_HPP_NAMESPACE::PartitionedAccelerationStructureInstanceFlagsNV;
  using VULKAN_HPP_NAMESPACE::PartitionedAccelerationStructureOpTypeNV;

  //=== VK_EXT_device_generated_commands ===
  using VULKAN_HPP_NAMESPACE::IndirectCommandsInputModeFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::IndirectCommandsInputModeFlagsEXT;
  using VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutUsageFlagBitsEXT;
  using VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutUsageFlagsEXT;
  using VULKAN_HPP_NAMESPACE::IndirectCommandsTokenTypeEXT;
  using VULKAN_HPP_NAMESPACE::IndirectExecutionSetInfoTypeEXT;

  //=== VK_KHR_maintenance8 ===
  using VULKAN_HPP_NAMESPACE::AccessFlagBits3KHR;
  using VULKAN_HPP_NAMESPACE::AccessFlags3KHR;

  //=== VK_EXT_ray_tracing_invocation_reorder ===
  using VULKAN_HPP_NAMESPACE::RayTracingInvocationReorderModeEXT;
  using VULKAN_HPP_NAMESPACE::RayTracingInvocationReorderModeNV;

  //=== VK_EXT_depth_clamp_control ===
  using VULKAN_HPP_NAMESPACE::DepthClampModeEXT;

  //=== VK_KHR_maintenance9 ===
  using VULKAN_HPP_NAMESPACE::DefaultVertexAttributeValueKHR;

#if defined( VK_USE_PLATFORM_OHOS )
  //=== VK_OHOS_surface ===
  using VULKAN_HPP_NAMESPACE::SurfaceCreateFlagBitsOHOS;
  using VULKAN_HPP_NAMESPACE::SurfaceCreateFlagsOHOS;
#endif /*VK_USE_PLATFORM_OHOS*/

#if defined( VK_USE_PLATFORM_OHOS )
  //=== VK_OHOS_native_buffer ===
  using VULKAN_HPP_NAMESPACE::SwapchainImageUsageFlagBitsOHOS;
  using VULKAN_HPP_NAMESPACE::SwapchainImageUsageFlagsOHOS;
#endif /*VK_USE_PLATFORM_OHOS*/

  //=== VK_ARM_performance_counters_by_region ===
  using VULKAN_HPP_NAMESPACE::PerformanceCounterDescriptionFlagBitsARM;
  using VULKAN_HPP_NAMESPACE::PerformanceCounterDescriptionFlagsARM;

  //=== VK_QCOM_data_graph_model ===
  using VULKAN_HPP_NAMESPACE::DataGraphModelCacheTypeQCOM;

  //=== VK_KHR_maintenance10 ===
  using VULKAN_HPP_NAMESPACE::RenderingAttachmentFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::RenderingAttachmentFlagsKHR;
  using VULKAN_HPP_NAMESPACE::ResolveImageFlagBitsKHR;
  using VULKAN_HPP_NAMESPACE::ResolveImageFlagsKHR;

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
  using VULKAN_HPP_NAMESPACE::NotPermittedError;
  using VULKAN_HPP_NAMESPACE::OutOfDateKHRError;
  using VULKAN_HPP_NAMESPACE::OutOfDeviceMemoryError;
  using VULKAN_HPP_NAMESPACE::OutOfHostMemoryError;
  using VULKAN_HPP_NAMESPACE::OutOfPoolMemoryError;
  using VULKAN_HPP_NAMESPACE::PresentTimingQueueFullEXTError;
  using VULKAN_HPP_NAMESPACE::SurfaceLostKHRError;
  using VULKAN_HPP_NAMESPACE::SystemError;
  using VULKAN_HPP_NAMESPACE::TooManyObjectsError;
  using VULKAN_HPP_NAMESPACE::UnknownError;
  using VULKAN_HPP_NAMESPACE::ValidationFailedError;
  using VULKAN_HPP_NAMESPACE::VideoPictureLayoutNotSupportedKHRError;
  using VULKAN_HPP_NAMESPACE::VideoProfileCodecNotSupportedKHRError;
  using VULKAN_HPP_NAMESPACE::VideoProfileFormatNotSupportedKHRError;
  using VULKAN_HPP_NAMESPACE::VideoProfileOperationNotSupportedKHRError;
  using VULKAN_HPP_NAMESPACE::VideoStdVersionNotSupportedKHRError;

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  using VULKAN_HPP_NAMESPACE::FullScreenExclusiveModeLostEXTError;
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

  using VULKAN_HPP_NAMESPACE::CompressionExhaustedEXTError;
  using VULKAN_HPP_NAMESPACE::InvalidVideoStdParametersKHRError;
  using VULKAN_HPP_NAMESPACE::NotEnoughSpaceKHRError;
#endif /*VULKAN_HPP_NO_EXCEPTIONS*/

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

  //=== VK_VERSION_1_4 ===
  using VULKAN_HPP_NAMESPACE::MaxGlobalPrioritySize;

  //=== VK_KHR_surface ===
  using VULKAN_HPP_NAMESPACE::KHRSurfaceExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRSurfaceSpecVersion;

  //=== VK_KHR_swapchain ===
  using VULKAN_HPP_NAMESPACE::KHRSwapchainExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRSwapchainSpecVersion;

  //=== VK_KHR_display ===
  using VULKAN_HPP_NAMESPACE::KHRDisplayExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRDisplaySpecVersion;

  //=== VK_KHR_display_swapchain ===
  using VULKAN_HPP_NAMESPACE::KHRDisplaySwapchainExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRDisplaySwapchainSpecVersion;

#if defined( VK_USE_PLATFORM_XLIB_KHR )
  //=== VK_KHR_xlib_surface ===
  using VULKAN_HPP_NAMESPACE::KHRXlibSurfaceExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRXlibSurfaceSpecVersion;
#endif /*VK_USE_PLATFORM_XLIB_KHR*/

#if defined( VK_USE_PLATFORM_XCB_KHR )
  //=== VK_KHR_xcb_surface ===
  using VULKAN_HPP_NAMESPACE::KHRXcbSurfaceExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRXcbSurfaceSpecVersion;
#endif /*VK_USE_PLATFORM_XCB_KHR*/

#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
  //=== VK_KHR_wayland_surface ===
  using VULKAN_HPP_NAMESPACE::KHRWaylandSurfaceExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRWaylandSurfaceSpecVersion;
#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  //=== VK_KHR_android_surface ===
  using VULKAN_HPP_NAMESPACE::KHRAndroidSurfaceExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRAndroidSurfaceSpecVersion;
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_win32_surface ===
  using VULKAN_HPP_NAMESPACE::KHRWin32SurfaceExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRWin32SurfaceSpecVersion;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_EXT_debug_report ===
  using VULKAN_HPP_NAMESPACE::EXTDebugReportExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTDebugReportSpecVersion;

  //=== VK_NV_glsl_shader ===
  using VULKAN_HPP_NAMESPACE::NVGlslShaderExtensionName;
  using VULKAN_HPP_NAMESPACE::NVGlslShaderSpecVersion;

  //=== VK_EXT_depth_range_unrestricted ===
  using VULKAN_HPP_NAMESPACE::EXTDepthRangeUnrestrictedExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTDepthRangeUnrestrictedSpecVersion;

  //=== VK_KHR_sampler_mirror_clamp_to_edge ===
  using VULKAN_HPP_NAMESPACE::KHRSamplerMirrorClampToEdgeExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRSamplerMirrorClampToEdgeSpecVersion;

  //=== VK_IMG_filter_cubic ===
  using VULKAN_HPP_NAMESPACE::IMGFilterCubicExtensionName;
  using VULKAN_HPP_NAMESPACE::IMGFilterCubicSpecVersion;

  //=== VK_AMD_rasterization_order ===
  using VULKAN_HPP_NAMESPACE::AMDRasterizationOrderExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDRasterizationOrderSpecVersion;

  //=== VK_AMD_shader_trinary_minmax ===
  using VULKAN_HPP_NAMESPACE::AMDShaderTrinaryMinmaxExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDShaderTrinaryMinmaxSpecVersion;

  //=== VK_AMD_shader_explicit_vertex_parameter ===
  using VULKAN_HPP_NAMESPACE::AMDShaderExplicitVertexParameterExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDShaderExplicitVertexParameterSpecVersion;

  //=== VK_EXT_debug_marker ===
  using VULKAN_HPP_NAMESPACE::EXTDebugMarkerExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTDebugMarkerSpecVersion;

  //=== VK_KHR_video_queue ===
  using VULKAN_HPP_NAMESPACE::KHRVideoQueueExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRVideoQueueSpecVersion;

  //=== VK_KHR_video_decode_queue ===
  using VULKAN_HPP_NAMESPACE::KHRVideoDecodeQueueExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRVideoDecodeQueueSpecVersion;

  //=== VK_AMD_gcn_shader ===
  using VULKAN_HPP_NAMESPACE::AMDGcnShaderExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDGcnShaderSpecVersion;

  //=== VK_NV_dedicated_allocation ===
  using VULKAN_HPP_NAMESPACE::NVDedicatedAllocationExtensionName;
  using VULKAN_HPP_NAMESPACE::NVDedicatedAllocationSpecVersion;

  //=== VK_EXT_transform_feedback ===
  using VULKAN_HPP_NAMESPACE::EXTTransformFeedbackExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTTransformFeedbackSpecVersion;

  //=== VK_NVX_binary_import ===
  using VULKAN_HPP_NAMESPACE::NVXBinaryImportExtensionName;
  using VULKAN_HPP_NAMESPACE::NVXBinaryImportSpecVersion;

  //=== VK_NVX_image_view_handle ===
  using VULKAN_HPP_NAMESPACE::NVXImageViewHandleExtensionName;
  using VULKAN_HPP_NAMESPACE::NVXImageViewHandleSpecVersion;

  //=== VK_AMD_draw_indirect_count ===
  using VULKAN_HPP_NAMESPACE::AMDDrawIndirectCountExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDDrawIndirectCountSpecVersion;

  //=== VK_AMD_negative_viewport_height ===
  using VULKAN_HPP_NAMESPACE::AMDNegativeViewportHeightExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDNegativeViewportHeightSpecVersion;

  //=== VK_AMD_gpu_shader_half_float ===
  using VULKAN_HPP_NAMESPACE::AMDGpuShaderHalfFloatExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDGpuShaderHalfFloatSpecVersion;

  //=== VK_AMD_shader_ballot ===
  using VULKAN_HPP_NAMESPACE::AMDShaderBallotExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDShaderBallotSpecVersion;

  //=== VK_KHR_video_encode_h264 ===
  using VULKAN_HPP_NAMESPACE::KHRVideoEncodeH264ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRVideoEncodeH264SpecVersion;

  //=== VK_KHR_video_encode_h265 ===
  using VULKAN_HPP_NAMESPACE::KHRVideoEncodeH265ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRVideoEncodeH265SpecVersion;

  //=== VK_KHR_video_decode_h264 ===
  using VULKAN_HPP_NAMESPACE::KHRVideoDecodeH264ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRVideoDecodeH264SpecVersion;

  //=== VK_AMD_texture_gather_bias_lod ===
  using VULKAN_HPP_NAMESPACE::AMDTextureGatherBiasLodExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDTextureGatherBiasLodSpecVersion;

  //=== VK_AMD_shader_info ===
  using VULKAN_HPP_NAMESPACE::AMDShaderInfoExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDShaderInfoSpecVersion;

  //=== VK_KHR_dynamic_rendering ===
  using VULKAN_HPP_NAMESPACE::KHRDynamicRenderingExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRDynamicRenderingSpecVersion;

  //=== VK_AMD_shader_image_load_store_lod ===
  using VULKAN_HPP_NAMESPACE::AMDShaderImageLoadStoreLodExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDShaderImageLoadStoreLodSpecVersion;

#if defined( VK_USE_PLATFORM_GGP )
  //=== VK_GGP_stream_descriptor_surface ===
  using VULKAN_HPP_NAMESPACE::GGPStreamDescriptorSurfaceExtensionName;
  using VULKAN_HPP_NAMESPACE::GGPStreamDescriptorSurfaceSpecVersion;
#endif /*VK_USE_PLATFORM_GGP*/

  //=== VK_NV_corner_sampled_image ===
  using VULKAN_HPP_NAMESPACE::NVCornerSampledImageExtensionName;
  using VULKAN_HPP_NAMESPACE::NVCornerSampledImageSpecVersion;

  //=== VK_KHR_multiview ===
  using VULKAN_HPP_NAMESPACE::KHRMultiviewExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRMultiviewSpecVersion;

  //=== VK_IMG_format_pvrtc ===
  using VULKAN_HPP_NAMESPACE::IMGFormatPvrtcExtensionName;
  using VULKAN_HPP_NAMESPACE::IMGFormatPvrtcSpecVersion;

  //=== VK_NV_external_memory_capabilities ===
  using VULKAN_HPP_NAMESPACE::NVExternalMemoryCapabilitiesExtensionName;
  using VULKAN_HPP_NAMESPACE::NVExternalMemoryCapabilitiesSpecVersion;

  //=== VK_NV_external_memory ===
  using VULKAN_HPP_NAMESPACE::NVExternalMemoryExtensionName;
  using VULKAN_HPP_NAMESPACE::NVExternalMemorySpecVersion;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_NV_external_memory_win32 ===
  using VULKAN_HPP_NAMESPACE::NVExternalMemoryWin32ExtensionName;
  using VULKAN_HPP_NAMESPACE::NVExternalMemoryWin32SpecVersion;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_NV_win32_keyed_mutex ===
  using VULKAN_HPP_NAMESPACE::NVWin32KeyedMutexExtensionName;
  using VULKAN_HPP_NAMESPACE::NVWin32KeyedMutexSpecVersion;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_KHR_get_physical_device_properties2 ===
  using VULKAN_HPP_NAMESPACE::KHRGetPhysicalDeviceProperties2ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRGetPhysicalDeviceProperties2SpecVersion;

  //=== VK_KHR_device_group ===
  using VULKAN_HPP_NAMESPACE::KHRDeviceGroupExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRDeviceGroupSpecVersion;

  //=== VK_EXT_validation_flags ===
  using VULKAN_HPP_NAMESPACE::EXTValidationFlagsExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTValidationFlagsSpecVersion;

#if defined( VK_USE_PLATFORM_VI_NN )
  //=== VK_NN_vi_surface ===
  using VULKAN_HPP_NAMESPACE::NNViSurfaceExtensionName;
  using VULKAN_HPP_NAMESPACE::NNViSurfaceSpecVersion;
#endif /*VK_USE_PLATFORM_VI_NN*/

  //=== VK_KHR_shader_draw_parameters ===
  using VULKAN_HPP_NAMESPACE::KHRShaderDrawParametersExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRShaderDrawParametersSpecVersion;

  //=== VK_EXT_shader_subgroup_ballot ===
  using VULKAN_HPP_NAMESPACE::EXTShaderSubgroupBallotExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTShaderSubgroupBallotSpecVersion;

  //=== VK_EXT_shader_subgroup_vote ===
  using VULKAN_HPP_NAMESPACE::EXTShaderSubgroupVoteExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTShaderSubgroupVoteSpecVersion;

  //=== VK_EXT_texture_compression_astc_hdr ===
  using VULKAN_HPP_NAMESPACE::EXTTextureCompressionAstcHdrExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTTextureCompressionAstcHdrSpecVersion;

  //=== VK_EXT_astc_decode_mode ===
  using VULKAN_HPP_NAMESPACE::EXTAstcDecodeModeExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTAstcDecodeModeSpecVersion;

  //=== VK_EXT_pipeline_robustness ===
  using VULKAN_HPP_NAMESPACE::EXTPipelineRobustnessExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTPipelineRobustnessSpecVersion;

  //=== VK_KHR_maintenance1 ===
  using VULKAN_HPP_NAMESPACE::KHRMaintenance1ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRMaintenance1SpecVersion;

  //=== VK_KHR_device_group_creation ===
  using VULKAN_HPP_NAMESPACE::KHRDeviceGroupCreationExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRDeviceGroupCreationSpecVersion;
  using VULKAN_HPP_NAMESPACE::MaxDeviceGroupSizeKHR;

  //=== VK_KHR_external_memory_capabilities ===
  using VULKAN_HPP_NAMESPACE::KHRExternalMemoryCapabilitiesExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRExternalMemoryCapabilitiesSpecVersion;
  using VULKAN_HPP_NAMESPACE::LuidSizeKHR;

  //=== VK_KHR_external_memory ===
  using VULKAN_HPP_NAMESPACE::KHRExternalMemoryExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRExternalMemorySpecVersion;
  using VULKAN_HPP_NAMESPACE::QueueFamilyExternalKHR;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_external_memory_win32 ===
  using VULKAN_HPP_NAMESPACE::KHRExternalMemoryWin32ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRExternalMemoryWin32SpecVersion;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_KHR_external_memory_fd ===
  using VULKAN_HPP_NAMESPACE::KHRExternalMemoryFdExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRExternalMemoryFdSpecVersion;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_win32_keyed_mutex ===
  using VULKAN_HPP_NAMESPACE::KHRWin32KeyedMutexExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRWin32KeyedMutexSpecVersion;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_KHR_external_semaphore_capabilities ===
  using VULKAN_HPP_NAMESPACE::KHRExternalSemaphoreCapabilitiesExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRExternalSemaphoreCapabilitiesSpecVersion;

  //=== VK_KHR_external_semaphore ===
  using VULKAN_HPP_NAMESPACE::KHRExternalSemaphoreExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRExternalSemaphoreSpecVersion;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_external_semaphore_win32 ===
  using VULKAN_HPP_NAMESPACE::KHRExternalSemaphoreWin32ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRExternalSemaphoreWin32SpecVersion;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_KHR_external_semaphore_fd ===
  using VULKAN_HPP_NAMESPACE::KHRExternalSemaphoreFdExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRExternalSemaphoreFdSpecVersion;

  //=== VK_KHR_push_descriptor ===
  using VULKAN_HPP_NAMESPACE::KHRPushDescriptorExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRPushDescriptorSpecVersion;

  //=== VK_EXT_conditional_rendering ===
  using VULKAN_HPP_NAMESPACE::EXTConditionalRenderingExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTConditionalRenderingSpecVersion;

  //=== VK_KHR_shader_float16_int8 ===
  using VULKAN_HPP_NAMESPACE::KHRShaderFloat16Int8ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRShaderFloat16Int8SpecVersion;

  //=== VK_KHR_16bit_storage ===
  using VULKAN_HPP_NAMESPACE::KHR16BitStorageExtensionName;
  using VULKAN_HPP_NAMESPACE::KHR16BitStorageSpecVersion;

  //=== VK_KHR_incremental_present ===
  using VULKAN_HPP_NAMESPACE::KHRIncrementalPresentExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRIncrementalPresentSpecVersion;

  //=== VK_KHR_descriptor_update_template ===
  using VULKAN_HPP_NAMESPACE::KHRDescriptorUpdateTemplateExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRDescriptorUpdateTemplateSpecVersion;

  //=== VK_NV_clip_space_w_scaling ===
  using VULKAN_HPP_NAMESPACE::NVClipSpaceWScalingExtensionName;
  using VULKAN_HPP_NAMESPACE::NVClipSpaceWScalingSpecVersion;

  //=== VK_EXT_direct_mode_display ===
  using VULKAN_HPP_NAMESPACE::EXTDirectModeDisplayExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTDirectModeDisplaySpecVersion;

#if defined( VK_USE_PLATFORM_XLIB_XRANDR_EXT )
  //=== VK_EXT_acquire_xlib_display ===
  using VULKAN_HPP_NAMESPACE::EXTAcquireXlibDisplayExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTAcquireXlibDisplaySpecVersion;
#endif /*VK_USE_PLATFORM_XLIB_XRANDR_EXT*/

  //=== VK_EXT_display_surface_counter ===
  using VULKAN_HPP_NAMESPACE::EXTDisplaySurfaceCounterExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTDisplaySurfaceCounterSpecVersion;

  //=== VK_EXT_display_control ===
  using VULKAN_HPP_NAMESPACE::EXTDisplayControlExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTDisplayControlSpecVersion;

  //=== VK_GOOGLE_display_timing ===
  using VULKAN_HPP_NAMESPACE::GOOGLEDisplayTimingExtensionName;
  using VULKAN_HPP_NAMESPACE::GOOGLEDisplayTimingSpecVersion;

  //=== VK_NV_sample_mask_override_coverage ===
  using VULKAN_HPP_NAMESPACE::NVSampleMaskOverrideCoverageExtensionName;
  using VULKAN_HPP_NAMESPACE::NVSampleMaskOverrideCoverageSpecVersion;

  //=== VK_NV_geometry_shader_passthrough ===
  using VULKAN_HPP_NAMESPACE::NVGeometryShaderPassthroughExtensionName;
  using VULKAN_HPP_NAMESPACE::NVGeometryShaderPassthroughSpecVersion;

  //=== VK_NV_viewport_array2 ===
  using VULKAN_HPP_NAMESPACE::NVViewportArray2ExtensionName;
  using VULKAN_HPP_NAMESPACE::NVViewportArray2SpecVersion;

  //=== VK_NVX_multiview_per_view_attributes ===
  using VULKAN_HPP_NAMESPACE::NVXMultiviewPerViewAttributesExtensionName;
  using VULKAN_HPP_NAMESPACE::NVXMultiviewPerViewAttributesSpecVersion;

  //=== VK_NV_viewport_swizzle ===
  using VULKAN_HPP_NAMESPACE::NVViewportSwizzleExtensionName;
  using VULKAN_HPP_NAMESPACE::NVViewportSwizzleSpecVersion;

  //=== VK_EXT_discard_rectangles ===
  using VULKAN_HPP_NAMESPACE::EXTDiscardRectanglesExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTDiscardRectanglesSpecVersion;

  //=== VK_EXT_conservative_rasterization ===
  using VULKAN_HPP_NAMESPACE::EXTConservativeRasterizationExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTConservativeRasterizationSpecVersion;

  //=== VK_EXT_depth_clip_enable ===
  using VULKAN_HPP_NAMESPACE::EXTDepthClipEnableExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTDepthClipEnableSpecVersion;

  //=== VK_EXT_swapchain_colorspace ===
  using VULKAN_HPP_NAMESPACE::EXTSwapchainColorSpaceExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTSwapchainColorSpaceSpecVersion;

  //=== VK_EXT_hdr_metadata ===
  using VULKAN_HPP_NAMESPACE::EXTHdrMetadataExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTHdrMetadataSpecVersion;

  //=== VK_KHR_imageless_framebuffer ===
  using VULKAN_HPP_NAMESPACE::KHRImagelessFramebufferExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRImagelessFramebufferSpecVersion;

  //=== VK_KHR_create_renderpass2 ===
  using VULKAN_HPP_NAMESPACE::KHRCreateRenderpass2ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRCreateRenderpass2SpecVersion;

  //=== VK_IMG_relaxed_line_rasterization ===
  using VULKAN_HPP_NAMESPACE::IMGRelaxedLineRasterizationExtensionName;
  using VULKAN_HPP_NAMESPACE::IMGRelaxedLineRasterizationSpecVersion;

  //=== VK_KHR_shared_presentable_image ===
  using VULKAN_HPP_NAMESPACE::KHRSharedPresentableImageExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRSharedPresentableImageSpecVersion;

  //=== VK_KHR_external_fence_capabilities ===
  using VULKAN_HPP_NAMESPACE::KHRExternalFenceCapabilitiesExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRExternalFenceCapabilitiesSpecVersion;

  //=== VK_KHR_external_fence ===
  using VULKAN_HPP_NAMESPACE::KHRExternalFenceExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRExternalFenceSpecVersion;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_external_fence_win32 ===
  using VULKAN_HPP_NAMESPACE::KHRExternalFenceWin32ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRExternalFenceWin32SpecVersion;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_KHR_external_fence_fd ===
  using VULKAN_HPP_NAMESPACE::KHRExternalFenceFdExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRExternalFenceFdSpecVersion;

  //=== VK_KHR_performance_query ===
  using VULKAN_HPP_NAMESPACE::KHRPerformanceQueryExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRPerformanceQuerySpecVersion;

  //=== VK_KHR_maintenance2 ===
  using VULKAN_HPP_NAMESPACE::KHRMaintenance2ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRMaintenance2SpecVersion;

  //=== VK_KHR_get_surface_capabilities2 ===
  using VULKAN_HPP_NAMESPACE::KHRGetSurfaceCapabilities2ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRGetSurfaceCapabilities2SpecVersion;

  //=== VK_KHR_variable_pointers ===
  using VULKAN_HPP_NAMESPACE::KHRVariablePointersExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRVariablePointersSpecVersion;

  //=== VK_KHR_get_display_properties2 ===
  using VULKAN_HPP_NAMESPACE::KHRGetDisplayProperties2ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRGetDisplayProperties2SpecVersion;

#if defined( VK_USE_PLATFORM_IOS_MVK )
  //=== VK_MVK_ios_surface ===
  using VULKAN_HPP_NAMESPACE::MVKIosSurfaceExtensionName;
  using VULKAN_HPP_NAMESPACE::MVKIosSurfaceSpecVersion;
#endif /*VK_USE_PLATFORM_IOS_MVK*/

#if defined( VK_USE_PLATFORM_MACOS_MVK )
  //=== VK_MVK_macos_surface ===
  using VULKAN_HPP_NAMESPACE::MVKMacosSurfaceExtensionName;
  using VULKAN_HPP_NAMESPACE::MVKMacosSurfaceSpecVersion;
#endif /*VK_USE_PLATFORM_MACOS_MVK*/

  //=== VK_EXT_external_memory_dma_buf ===
  using VULKAN_HPP_NAMESPACE::EXTExternalMemoryDmaBufExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTExternalMemoryDmaBufSpecVersion;

  //=== VK_EXT_queue_family_foreign ===
  using VULKAN_HPP_NAMESPACE::EXTQueueFamilyForeignExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTQueueFamilyForeignSpecVersion;
  using VULKAN_HPP_NAMESPACE::QueueFamilyForeignEXT;

  //=== VK_KHR_dedicated_allocation ===
  using VULKAN_HPP_NAMESPACE::KHRDedicatedAllocationExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRDedicatedAllocationSpecVersion;

  //=== VK_EXT_debug_utils ===
  using VULKAN_HPP_NAMESPACE::EXTDebugUtilsExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTDebugUtilsSpecVersion;

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  //=== VK_ANDROID_external_memory_android_hardware_buffer ===
  using VULKAN_HPP_NAMESPACE::ANDROIDExternalMemoryAndroidHardwareBufferExtensionName;
  using VULKAN_HPP_NAMESPACE::ANDROIDExternalMemoryAndroidHardwareBufferSpecVersion;
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

  //=== VK_EXT_sampler_filter_minmax ===
  using VULKAN_HPP_NAMESPACE::EXTSamplerFilterMinmaxExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTSamplerFilterMinmaxSpecVersion;

  //=== VK_KHR_storage_buffer_storage_class ===
  using VULKAN_HPP_NAMESPACE::KHRStorageBufferStorageClassExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRStorageBufferStorageClassSpecVersion;

  //=== VK_AMD_gpu_shader_int16 ===
  using VULKAN_HPP_NAMESPACE::AMDGpuShaderInt16ExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDGpuShaderInt16SpecVersion;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_AMDX_shader_enqueue ===
  using VULKAN_HPP_NAMESPACE::AMDXShaderEnqueueExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDXShaderEnqueueSpecVersion;
  using VULKAN_HPP_NAMESPACE::ShaderIndexUnusedAMDX;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_AMD_mixed_attachment_samples ===
  using VULKAN_HPP_NAMESPACE::AMDMixedAttachmentSamplesExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDMixedAttachmentSamplesSpecVersion;

  //=== VK_AMD_shader_fragment_mask ===
  using VULKAN_HPP_NAMESPACE::AMDShaderFragmentMaskExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDShaderFragmentMaskSpecVersion;

  //=== VK_EXT_inline_uniform_block ===
  using VULKAN_HPP_NAMESPACE::EXTInlineUniformBlockExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTInlineUniformBlockSpecVersion;

  //=== VK_EXT_shader_stencil_export ===
  using VULKAN_HPP_NAMESPACE::EXTShaderStencilExportExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTShaderStencilExportSpecVersion;

  //=== VK_KHR_shader_bfloat16 ===
  using VULKAN_HPP_NAMESPACE::KHRShaderBfloat16ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRShaderBfloat16SpecVersion;

  //=== VK_EXT_sample_locations ===
  using VULKAN_HPP_NAMESPACE::EXTSampleLocationsExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTSampleLocationsSpecVersion;

  //=== VK_KHR_relaxed_block_layout ===
  using VULKAN_HPP_NAMESPACE::KHRRelaxedBlockLayoutExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRRelaxedBlockLayoutSpecVersion;

  //=== VK_KHR_get_memory_requirements2 ===
  using VULKAN_HPP_NAMESPACE::KHRGetMemoryRequirements2ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRGetMemoryRequirements2SpecVersion;

  //=== VK_KHR_image_format_list ===
  using VULKAN_HPP_NAMESPACE::KHRImageFormatListExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRImageFormatListSpecVersion;

  //=== VK_EXT_blend_operation_advanced ===
  using VULKAN_HPP_NAMESPACE::EXTBlendOperationAdvancedExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTBlendOperationAdvancedSpecVersion;

  //=== VK_NV_fragment_coverage_to_color ===
  using VULKAN_HPP_NAMESPACE::NVFragmentCoverageToColorExtensionName;
  using VULKAN_HPP_NAMESPACE::NVFragmentCoverageToColorSpecVersion;

  //=== VK_KHR_acceleration_structure ===
  using VULKAN_HPP_NAMESPACE::KHRAccelerationStructureExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRAccelerationStructureSpecVersion;

  //=== VK_KHR_ray_tracing_pipeline ===
  using VULKAN_HPP_NAMESPACE::KHRRayTracingPipelineExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRRayTracingPipelineSpecVersion;
  using VULKAN_HPP_NAMESPACE::ShaderUnusedKHR;

  //=== VK_KHR_ray_query ===
  using VULKAN_HPP_NAMESPACE::KHRRayQueryExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRRayQuerySpecVersion;

  //=== VK_NV_framebuffer_mixed_samples ===
  using VULKAN_HPP_NAMESPACE::NVFramebufferMixedSamplesExtensionName;
  using VULKAN_HPP_NAMESPACE::NVFramebufferMixedSamplesSpecVersion;

  //=== VK_NV_fill_rectangle ===
  using VULKAN_HPP_NAMESPACE::NVFillRectangleExtensionName;
  using VULKAN_HPP_NAMESPACE::NVFillRectangleSpecVersion;

  //=== VK_NV_shader_sm_builtins ===
  using VULKAN_HPP_NAMESPACE::NVShaderSmBuiltinsExtensionName;
  using VULKAN_HPP_NAMESPACE::NVShaderSmBuiltinsSpecVersion;

  //=== VK_EXT_post_depth_coverage ===
  using VULKAN_HPP_NAMESPACE::EXTPostDepthCoverageExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTPostDepthCoverageSpecVersion;

  //=== VK_KHR_sampler_ycbcr_conversion ===
  using VULKAN_HPP_NAMESPACE::KHRSamplerYcbcrConversionExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRSamplerYcbcrConversionSpecVersion;

  //=== VK_KHR_bind_memory2 ===
  using VULKAN_HPP_NAMESPACE::KHRBindMemory2ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRBindMemory2SpecVersion;

  //=== VK_EXT_image_drm_format_modifier ===
  using VULKAN_HPP_NAMESPACE::EXTImageDrmFormatModifierExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTImageDrmFormatModifierSpecVersion;

  //=== VK_EXT_validation_cache ===
  using VULKAN_HPP_NAMESPACE::EXTValidationCacheExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTValidationCacheSpecVersion;

  //=== VK_EXT_descriptor_indexing ===
  using VULKAN_HPP_NAMESPACE::EXTDescriptorIndexingExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTDescriptorIndexingSpecVersion;

  //=== VK_EXT_shader_viewport_index_layer ===
  using VULKAN_HPP_NAMESPACE::EXTShaderViewportIndexLayerExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTShaderViewportIndexLayerSpecVersion;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_KHR_portability_subset ===
  using VULKAN_HPP_NAMESPACE::KHRPortabilitySubsetExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRPortabilitySubsetSpecVersion;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_NV_shading_rate_image ===
  using VULKAN_HPP_NAMESPACE::NVShadingRateImageExtensionName;
  using VULKAN_HPP_NAMESPACE::NVShadingRateImageSpecVersion;

  //=== VK_NV_ray_tracing ===
  using VULKAN_HPP_NAMESPACE::NVRayTracingExtensionName;
  using VULKAN_HPP_NAMESPACE::NVRayTracingSpecVersion;
  using VULKAN_HPP_NAMESPACE::ShaderUnusedNV;

  //=== VK_NV_representative_fragment_test ===
  using VULKAN_HPP_NAMESPACE::NVRepresentativeFragmentTestExtensionName;
  using VULKAN_HPP_NAMESPACE::NVRepresentativeFragmentTestSpecVersion;

  //=== VK_KHR_maintenance3 ===
  using VULKAN_HPP_NAMESPACE::KHRMaintenance3ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRMaintenance3SpecVersion;

  //=== VK_KHR_draw_indirect_count ===
  using VULKAN_HPP_NAMESPACE::KHRDrawIndirectCountExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRDrawIndirectCountSpecVersion;

  //=== VK_EXT_filter_cubic ===
  using VULKAN_HPP_NAMESPACE::EXTFilterCubicExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTFilterCubicSpecVersion;

  //=== VK_QCOM_render_pass_shader_resolve ===
  using VULKAN_HPP_NAMESPACE::QCOMRenderPassShaderResolveExtensionName;
  using VULKAN_HPP_NAMESPACE::QCOMRenderPassShaderResolveSpecVersion;

  //=== VK_EXT_global_priority ===
  using VULKAN_HPP_NAMESPACE::EXTGlobalPriorityExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTGlobalPrioritySpecVersion;

  //=== VK_KHR_shader_subgroup_extended_types ===
  using VULKAN_HPP_NAMESPACE::KHRShaderSubgroupExtendedTypesExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRShaderSubgroupExtendedTypesSpecVersion;

  //=== VK_KHR_8bit_storage ===
  using VULKAN_HPP_NAMESPACE::KHR8BitStorageExtensionName;
  using VULKAN_HPP_NAMESPACE::KHR8BitStorageSpecVersion;

  //=== VK_EXT_external_memory_host ===
  using VULKAN_HPP_NAMESPACE::EXTExternalMemoryHostExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTExternalMemoryHostSpecVersion;

  //=== VK_AMD_buffer_marker ===
  using VULKAN_HPP_NAMESPACE::AMDBufferMarkerExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDBufferMarkerSpecVersion;

  //=== VK_KHR_shader_atomic_int64 ===
  using VULKAN_HPP_NAMESPACE::KHRShaderAtomicInt64ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRShaderAtomicInt64SpecVersion;

  //=== VK_KHR_shader_clock ===
  using VULKAN_HPP_NAMESPACE::KHRShaderClockExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRShaderClockSpecVersion;

  //=== VK_AMD_pipeline_compiler_control ===
  using VULKAN_HPP_NAMESPACE::AMDPipelineCompilerControlExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDPipelineCompilerControlSpecVersion;

  //=== VK_EXT_calibrated_timestamps ===
  using VULKAN_HPP_NAMESPACE::EXTCalibratedTimestampsExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTCalibratedTimestampsSpecVersion;

  //=== VK_AMD_shader_core_properties ===
  using VULKAN_HPP_NAMESPACE::AMDShaderCorePropertiesExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDShaderCorePropertiesSpecVersion;

  //=== VK_KHR_video_decode_h265 ===
  using VULKAN_HPP_NAMESPACE::KHRVideoDecodeH265ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRVideoDecodeH265SpecVersion;

  //=== VK_KHR_global_priority ===
  using VULKAN_HPP_NAMESPACE::KHRGlobalPriorityExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRGlobalPrioritySpecVersion;
  using VULKAN_HPP_NAMESPACE::MaxGlobalPrioritySizeKHR;

  //=== VK_AMD_memory_overallocation_behavior ===
  using VULKAN_HPP_NAMESPACE::AMDMemoryOverallocationBehaviorExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDMemoryOverallocationBehaviorSpecVersion;

  //=== VK_EXT_vertex_attribute_divisor ===
  using VULKAN_HPP_NAMESPACE::EXTVertexAttributeDivisorExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTVertexAttributeDivisorSpecVersion;

#if defined( VK_USE_PLATFORM_GGP )
  //=== VK_GGP_frame_token ===
  using VULKAN_HPP_NAMESPACE::GGPFrameTokenExtensionName;
  using VULKAN_HPP_NAMESPACE::GGPFrameTokenSpecVersion;
#endif /*VK_USE_PLATFORM_GGP*/

  //=== VK_EXT_pipeline_creation_feedback ===
  using VULKAN_HPP_NAMESPACE::EXTPipelineCreationFeedbackExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTPipelineCreationFeedbackSpecVersion;

  //=== VK_KHR_driver_properties ===
  using VULKAN_HPP_NAMESPACE::KHRDriverPropertiesExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRDriverPropertiesSpecVersion;
  using VULKAN_HPP_NAMESPACE::MaxDriverInfoSizeKHR;
  using VULKAN_HPP_NAMESPACE::MaxDriverNameSizeKHR;

  //=== VK_KHR_shader_float_controls ===
  using VULKAN_HPP_NAMESPACE::KHRShaderFloatControlsExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRShaderFloatControlsSpecVersion;

  //=== VK_NV_shader_subgroup_partitioned ===
  using VULKAN_HPP_NAMESPACE::NVShaderSubgroupPartitionedExtensionName;
  using VULKAN_HPP_NAMESPACE::NVShaderSubgroupPartitionedSpecVersion;

  //=== VK_KHR_depth_stencil_resolve ===
  using VULKAN_HPP_NAMESPACE::KHRDepthStencilResolveExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRDepthStencilResolveSpecVersion;

  //=== VK_KHR_swapchain_mutable_format ===
  using VULKAN_HPP_NAMESPACE::KHRSwapchainMutableFormatExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRSwapchainMutableFormatSpecVersion;

  //=== VK_NV_compute_shader_derivatives ===
  using VULKAN_HPP_NAMESPACE::NVComputeShaderDerivativesExtensionName;
  using VULKAN_HPP_NAMESPACE::NVComputeShaderDerivativesSpecVersion;

  //=== VK_NV_mesh_shader ===
  using VULKAN_HPP_NAMESPACE::NVMeshShaderExtensionName;
  using VULKAN_HPP_NAMESPACE::NVMeshShaderSpecVersion;

  //=== VK_NV_fragment_shader_barycentric ===
  using VULKAN_HPP_NAMESPACE::NVFragmentShaderBarycentricExtensionName;
  using VULKAN_HPP_NAMESPACE::NVFragmentShaderBarycentricSpecVersion;

  //=== VK_NV_shader_image_footprint ===
  using VULKAN_HPP_NAMESPACE::NVShaderImageFootprintExtensionName;
  using VULKAN_HPP_NAMESPACE::NVShaderImageFootprintSpecVersion;

  //=== VK_NV_scissor_exclusive ===
  using VULKAN_HPP_NAMESPACE::NVScissorExclusiveExtensionName;
  using VULKAN_HPP_NAMESPACE::NVScissorExclusiveSpecVersion;

  //=== VK_NV_device_diagnostic_checkpoints ===
  using VULKAN_HPP_NAMESPACE::NVDeviceDiagnosticCheckpointsExtensionName;
  using VULKAN_HPP_NAMESPACE::NVDeviceDiagnosticCheckpointsSpecVersion;

  //=== VK_KHR_timeline_semaphore ===
  using VULKAN_HPP_NAMESPACE::KHRTimelineSemaphoreExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRTimelineSemaphoreSpecVersion;

  //=== VK_EXT_present_timing ===
  using VULKAN_HPP_NAMESPACE::EXTPresentTimingExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTPresentTimingSpecVersion;

  //=== VK_INTEL_shader_integer_functions2 ===
  using VULKAN_HPP_NAMESPACE::INTELShaderIntegerFunctions2ExtensionName;
  using VULKAN_HPP_NAMESPACE::INTELShaderIntegerFunctions2SpecVersion;

  //=== VK_INTEL_performance_query ===
  using VULKAN_HPP_NAMESPACE::INTELPerformanceQueryExtensionName;
  using VULKAN_HPP_NAMESPACE::INTELPerformanceQuerySpecVersion;

  //=== VK_KHR_vulkan_memory_model ===
  using VULKAN_HPP_NAMESPACE::KHRVulkanMemoryModelExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRVulkanMemoryModelSpecVersion;

  //=== VK_EXT_pci_bus_info ===
  using VULKAN_HPP_NAMESPACE::EXTPciBusInfoExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTPciBusInfoSpecVersion;

  //=== VK_AMD_display_native_hdr ===
  using VULKAN_HPP_NAMESPACE::AMDDisplayNativeHdrExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDDisplayNativeHdrSpecVersion;

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_imagepipe_surface ===
  using VULKAN_HPP_NAMESPACE::FUCHSIAImagepipeSurfaceExtensionName;
  using VULKAN_HPP_NAMESPACE::FUCHSIAImagepipeSurfaceSpecVersion;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

  //=== VK_KHR_shader_terminate_invocation ===
  using VULKAN_HPP_NAMESPACE::KHRShaderTerminateInvocationExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRShaderTerminateInvocationSpecVersion;

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_surface ===
  using VULKAN_HPP_NAMESPACE::EXTMetalSurfaceExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTMetalSurfaceSpecVersion;
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_EXT_fragment_density_map ===
  using VULKAN_HPP_NAMESPACE::EXTFragmentDensityMapExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTFragmentDensityMapSpecVersion;

  //=== VK_EXT_scalar_block_layout ===
  using VULKAN_HPP_NAMESPACE::EXTScalarBlockLayoutExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTScalarBlockLayoutSpecVersion;

  //=== VK_GOOGLE_hlsl_functionality1 ===
  using VULKAN_HPP_NAMESPACE::GOOGLEHlslFunctionality1ExtensionName;
  using VULKAN_HPP_NAMESPACE::GOOGLEHlslFunctionality1SpecVersion;

  //=== VK_GOOGLE_decorate_string ===
  using VULKAN_HPP_NAMESPACE::GOOGLEDecorateStringExtensionName;
  using VULKAN_HPP_NAMESPACE::GOOGLEDecorateStringSpecVersion;

  //=== VK_EXT_subgroup_size_control ===
  using VULKAN_HPP_NAMESPACE::EXTSubgroupSizeControlExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTSubgroupSizeControlSpecVersion;

  //=== VK_KHR_fragment_shading_rate ===
  using VULKAN_HPP_NAMESPACE::KHRFragmentShadingRateExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRFragmentShadingRateSpecVersion;

  //=== VK_AMD_shader_core_properties2 ===
  using VULKAN_HPP_NAMESPACE::AMDShaderCoreProperties2ExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDShaderCoreProperties2SpecVersion;

  //=== VK_AMD_device_coherent_memory ===
  using VULKAN_HPP_NAMESPACE::AMDDeviceCoherentMemoryExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDDeviceCoherentMemorySpecVersion;

  //=== VK_KHR_dynamic_rendering_local_read ===
  using VULKAN_HPP_NAMESPACE::KHRDynamicRenderingLocalReadExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRDynamicRenderingLocalReadSpecVersion;

  //=== VK_EXT_shader_image_atomic_int64 ===
  using VULKAN_HPP_NAMESPACE::EXTShaderImageAtomicInt64ExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTShaderImageAtomicInt64SpecVersion;

  //=== VK_KHR_shader_quad_control ===
  using VULKAN_HPP_NAMESPACE::KHRShaderQuadControlExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRShaderQuadControlSpecVersion;

  //=== VK_KHR_spirv_1_4 ===
  using VULKAN_HPP_NAMESPACE::KHRSpirv14ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRSpirv14SpecVersion;

  //=== VK_EXT_memory_budget ===
  using VULKAN_HPP_NAMESPACE::EXTMemoryBudgetExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTMemoryBudgetSpecVersion;

  //=== VK_EXT_memory_priority ===
  using VULKAN_HPP_NAMESPACE::EXTMemoryPriorityExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTMemoryPrioritySpecVersion;

  //=== VK_KHR_surface_protected_capabilities ===
  using VULKAN_HPP_NAMESPACE::KHRSurfaceProtectedCapabilitiesExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRSurfaceProtectedCapabilitiesSpecVersion;

  //=== VK_NV_dedicated_allocation_image_aliasing ===
  using VULKAN_HPP_NAMESPACE::NVDedicatedAllocationImageAliasingExtensionName;
  using VULKAN_HPP_NAMESPACE::NVDedicatedAllocationImageAliasingSpecVersion;

  //=== VK_KHR_separate_depth_stencil_layouts ===
  using VULKAN_HPP_NAMESPACE::KHRSeparateDepthStencilLayoutsExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRSeparateDepthStencilLayoutsSpecVersion;

  //=== VK_EXT_buffer_device_address ===
  using VULKAN_HPP_NAMESPACE::EXTBufferDeviceAddressExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTBufferDeviceAddressSpecVersion;

  //=== VK_EXT_tooling_info ===
  using VULKAN_HPP_NAMESPACE::EXTToolingInfoExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTToolingInfoSpecVersion;

  //=== VK_EXT_separate_stencil_usage ===
  using VULKAN_HPP_NAMESPACE::EXTSeparateStencilUsageExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTSeparateStencilUsageSpecVersion;

  //=== VK_EXT_validation_features ===
  using VULKAN_HPP_NAMESPACE::EXTValidationFeaturesExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTValidationFeaturesSpecVersion;

  //=== VK_KHR_present_wait ===
  using VULKAN_HPP_NAMESPACE::KHRPresentWaitExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRPresentWaitSpecVersion;

  //=== VK_NV_cooperative_matrix ===
  using VULKAN_HPP_NAMESPACE::NVCooperativeMatrixExtensionName;
  using VULKAN_HPP_NAMESPACE::NVCooperativeMatrixSpecVersion;

  //=== VK_NV_coverage_reduction_mode ===
  using VULKAN_HPP_NAMESPACE::NVCoverageReductionModeExtensionName;
  using VULKAN_HPP_NAMESPACE::NVCoverageReductionModeSpecVersion;

  //=== VK_EXT_fragment_shader_interlock ===
  using VULKAN_HPP_NAMESPACE::EXTFragmentShaderInterlockExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTFragmentShaderInterlockSpecVersion;

  //=== VK_EXT_ycbcr_image_arrays ===
  using VULKAN_HPP_NAMESPACE::EXTYcbcrImageArraysExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTYcbcrImageArraysSpecVersion;

  //=== VK_KHR_uniform_buffer_standard_layout ===
  using VULKAN_HPP_NAMESPACE::KHRUniformBufferStandardLayoutExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRUniformBufferStandardLayoutSpecVersion;

  //=== VK_EXT_provoking_vertex ===
  using VULKAN_HPP_NAMESPACE::EXTProvokingVertexExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTProvokingVertexSpecVersion;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_EXT_full_screen_exclusive ===
  using VULKAN_HPP_NAMESPACE::EXTFullScreenExclusiveExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTFullScreenExclusiveSpecVersion;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_EXT_headless_surface ===
  using VULKAN_HPP_NAMESPACE::EXTHeadlessSurfaceExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTHeadlessSurfaceSpecVersion;

  //=== VK_KHR_buffer_device_address ===
  using VULKAN_HPP_NAMESPACE::KHRBufferDeviceAddressExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRBufferDeviceAddressSpecVersion;

  //=== VK_EXT_line_rasterization ===
  using VULKAN_HPP_NAMESPACE::EXTLineRasterizationExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTLineRasterizationSpecVersion;

  //=== VK_EXT_shader_atomic_float ===
  using VULKAN_HPP_NAMESPACE::EXTShaderAtomicFloatExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTShaderAtomicFloatSpecVersion;

  //=== VK_EXT_host_query_reset ===
  using VULKAN_HPP_NAMESPACE::EXTHostQueryResetExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTHostQueryResetSpecVersion;

  //=== VK_EXT_index_type_uint8 ===
  using VULKAN_HPP_NAMESPACE::EXTIndexTypeUint8ExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTIndexTypeUint8SpecVersion;

  //=== VK_EXT_extended_dynamic_state ===
  using VULKAN_HPP_NAMESPACE::EXTExtendedDynamicStateExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTExtendedDynamicStateSpecVersion;

  //=== VK_KHR_deferred_host_operations ===
  using VULKAN_HPP_NAMESPACE::KHRDeferredHostOperationsExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRDeferredHostOperationsSpecVersion;

  //=== VK_KHR_pipeline_executable_properties ===
  using VULKAN_HPP_NAMESPACE::KHRPipelineExecutablePropertiesExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRPipelineExecutablePropertiesSpecVersion;

  //=== VK_EXT_host_image_copy ===
  using VULKAN_HPP_NAMESPACE::EXTHostImageCopyExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTHostImageCopySpecVersion;

  //=== VK_KHR_map_memory2 ===
  using VULKAN_HPP_NAMESPACE::KHRMapMemory2ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRMapMemory2SpecVersion;

  //=== VK_EXT_map_memory_placed ===
  using VULKAN_HPP_NAMESPACE::EXTMapMemoryPlacedExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTMapMemoryPlacedSpecVersion;

  //=== VK_EXT_shader_atomic_float2 ===
  using VULKAN_HPP_NAMESPACE::EXTShaderAtomicFloat2ExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTShaderAtomicFloat2SpecVersion;

  //=== VK_EXT_surface_maintenance1 ===
  using VULKAN_HPP_NAMESPACE::EXTSurfaceMaintenance1ExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTSurfaceMaintenance1SpecVersion;

  //=== VK_EXT_swapchain_maintenance1 ===
  using VULKAN_HPP_NAMESPACE::EXTSwapchainMaintenance1ExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTSwapchainMaintenance1SpecVersion;

  //=== VK_EXT_shader_demote_to_helper_invocation ===
  using VULKAN_HPP_NAMESPACE::EXTShaderDemoteToHelperInvocationExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTShaderDemoteToHelperInvocationSpecVersion;

  //=== VK_NV_device_generated_commands ===
  using VULKAN_HPP_NAMESPACE::NVDeviceGeneratedCommandsExtensionName;
  using VULKAN_HPP_NAMESPACE::NVDeviceGeneratedCommandsSpecVersion;

  //=== VK_NV_inherited_viewport_scissor ===
  using VULKAN_HPP_NAMESPACE::NVInheritedViewportScissorExtensionName;
  using VULKAN_HPP_NAMESPACE::NVInheritedViewportScissorSpecVersion;

  //=== VK_KHR_shader_integer_dot_product ===
  using VULKAN_HPP_NAMESPACE::KHRShaderIntegerDotProductExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRShaderIntegerDotProductSpecVersion;

  //=== VK_EXT_texel_buffer_alignment ===
  using VULKAN_HPP_NAMESPACE::EXTTexelBufferAlignmentExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTTexelBufferAlignmentSpecVersion;

  //=== VK_QCOM_render_pass_transform ===
  using VULKAN_HPP_NAMESPACE::QCOMRenderPassTransformExtensionName;
  using VULKAN_HPP_NAMESPACE::QCOMRenderPassTransformSpecVersion;

  //=== VK_EXT_depth_bias_control ===
  using VULKAN_HPP_NAMESPACE::EXTDepthBiasControlExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTDepthBiasControlSpecVersion;

  //=== VK_EXT_device_memory_report ===
  using VULKAN_HPP_NAMESPACE::EXTDeviceMemoryReportExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTDeviceMemoryReportSpecVersion;

  //=== VK_EXT_acquire_drm_display ===
  using VULKAN_HPP_NAMESPACE::EXTAcquireDrmDisplayExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTAcquireDrmDisplaySpecVersion;

  //=== VK_EXT_robustness2 ===
  using VULKAN_HPP_NAMESPACE::EXTRobustness2ExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTRobustness2SpecVersion;

  //=== VK_EXT_custom_border_color ===
  using VULKAN_HPP_NAMESPACE::EXTCustomBorderColorExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTCustomBorderColorSpecVersion;

  //=== VK_GOOGLE_user_type ===
  using VULKAN_HPP_NAMESPACE::GOOGLEUserTypeExtensionName;
  using VULKAN_HPP_NAMESPACE::GOOGLEUserTypeSpecVersion;

  //=== VK_KHR_pipeline_library ===
  using VULKAN_HPP_NAMESPACE::KHRPipelineLibraryExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRPipelineLibrarySpecVersion;

  //=== VK_NV_present_barrier ===
  using VULKAN_HPP_NAMESPACE::NVPresentBarrierExtensionName;
  using VULKAN_HPP_NAMESPACE::NVPresentBarrierSpecVersion;

  //=== VK_KHR_shader_non_semantic_info ===
  using VULKAN_HPP_NAMESPACE::KHRShaderNonSemanticInfoExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRShaderNonSemanticInfoSpecVersion;

  //=== VK_KHR_present_id ===
  using VULKAN_HPP_NAMESPACE::KHRPresentIdExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRPresentIdSpecVersion;

  //=== VK_EXT_private_data ===
  using VULKAN_HPP_NAMESPACE::EXTPrivateDataExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTPrivateDataSpecVersion;

  //=== VK_EXT_pipeline_creation_cache_control ===
  using VULKAN_HPP_NAMESPACE::EXTPipelineCreationCacheControlExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTPipelineCreationCacheControlSpecVersion;

  //=== VK_KHR_video_encode_queue ===
  using VULKAN_HPP_NAMESPACE::KHRVideoEncodeQueueExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRVideoEncodeQueueSpecVersion;

  //=== VK_NV_device_diagnostics_config ===
  using VULKAN_HPP_NAMESPACE::NVDeviceDiagnosticsConfigExtensionName;
  using VULKAN_HPP_NAMESPACE::NVDeviceDiagnosticsConfigSpecVersion;

  //=== VK_QCOM_render_pass_store_ops ===
  using VULKAN_HPP_NAMESPACE::QCOMRenderPassStoreOpsExtensionName;
  using VULKAN_HPP_NAMESPACE::QCOMRenderPassStoreOpsSpecVersion;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_NV_cuda_kernel_launch ===
  using VULKAN_HPP_NAMESPACE::NVCudaKernelLaunchExtensionName;
  using VULKAN_HPP_NAMESPACE::NVCudaKernelLaunchSpecVersion;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_QCOM_tile_shading ===
  using VULKAN_HPP_NAMESPACE::QCOMTileShadingExtensionName;
  using VULKAN_HPP_NAMESPACE::QCOMTileShadingSpecVersion;

  //=== VK_NV_low_latency ===
  using VULKAN_HPP_NAMESPACE::NVLowLatencyExtensionName;
  using VULKAN_HPP_NAMESPACE::NVLowLatencySpecVersion;

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_objects ===
  using VULKAN_HPP_NAMESPACE::EXTMetalObjectsExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTMetalObjectsSpecVersion;
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_KHR_synchronization2 ===
  using VULKAN_HPP_NAMESPACE::KHRSynchronization2ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRSynchronization2SpecVersion;

  //=== VK_EXT_descriptor_buffer ===
  using VULKAN_HPP_NAMESPACE::EXTDescriptorBufferExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTDescriptorBufferSpecVersion;

  //=== VK_EXT_graphics_pipeline_library ===
  using VULKAN_HPP_NAMESPACE::EXTGraphicsPipelineLibraryExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTGraphicsPipelineLibrarySpecVersion;

  //=== VK_AMD_shader_early_and_late_fragment_tests ===
  using VULKAN_HPP_NAMESPACE::AMDShaderEarlyAndLateFragmentTestsExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDShaderEarlyAndLateFragmentTestsSpecVersion;

  //=== VK_KHR_fragment_shader_barycentric ===
  using VULKAN_HPP_NAMESPACE::KHRFragmentShaderBarycentricExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRFragmentShaderBarycentricSpecVersion;

  //=== VK_KHR_shader_subgroup_uniform_control_flow ===
  using VULKAN_HPP_NAMESPACE::KHRShaderSubgroupUniformControlFlowExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRShaderSubgroupUniformControlFlowSpecVersion;

  //=== VK_KHR_zero_initialize_workgroup_memory ===
  using VULKAN_HPP_NAMESPACE::KHRZeroInitializeWorkgroupMemoryExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRZeroInitializeWorkgroupMemorySpecVersion;

  //=== VK_NV_fragment_shading_rate_enums ===
  using VULKAN_HPP_NAMESPACE::NVFragmentShadingRateEnumsExtensionName;
  using VULKAN_HPP_NAMESPACE::NVFragmentShadingRateEnumsSpecVersion;

  //=== VK_NV_ray_tracing_motion_blur ===
  using VULKAN_HPP_NAMESPACE::NVRayTracingMotionBlurExtensionName;
  using VULKAN_HPP_NAMESPACE::NVRayTracingMotionBlurSpecVersion;

  //=== VK_EXT_mesh_shader ===
  using VULKAN_HPP_NAMESPACE::EXTMeshShaderExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTMeshShaderSpecVersion;

  //=== VK_EXT_ycbcr_2plane_444_formats ===
  using VULKAN_HPP_NAMESPACE::EXTYcbcr2Plane444FormatsExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTYcbcr2Plane444FormatsSpecVersion;

  //=== VK_EXT_fragment_density_map2 ===
  using VULKAN_HPP_NAMESPACE::EXTFragmentDensityMap2ExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTFragmentDensityMap2SpecVersion;

  //=== VK_QCOM_rotated_copy_commands ===
  using VULKAN_HPP_NAMESPACE::QCOMRotatedCopyCommandsExtensionName;
  using VULKAN_HPP_NAMESPACE::QCOMRotatedCopyCommandsSpecVersion;

  //=== VK_EXT_image_robustness ===
  using VULKAN_HPP_NAMESPACE::EXTImageRobustnessExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTImageRobustnessSpecVersion;

  //=== VK_KHR_workgroup_memory_explicit_layout ===
  using VULKAN_HPP_NAMESPACE::KHRWorkgroupMemoryExplicitLayoutExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRWorkgroupMemoryExplicitLayoutSpecVersion;

  //=== VK_KHR_copy_commands2 ===
  using VULKAN_HPP_NAMESPACE::KHRCopyCommands2ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRCopyCommands2SpecVersion;

  //=== VK_EXT_image_compression_control ===
  using VULKAN_HPP_NAMESPACE::EXTImageCompressionControlExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTImageCompressionControlSpecVersion;

  //=== VK_EXT_attachment_feedback_loop_layout ===
  using VULKAN_HPP_NAMESPACE::EXTAttachmentFeedbackLoopLayoutExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTAttachmentFeedbackLoopLayoutSpecVersion;

  //=== VK_EXT_4444_formats ===
  using VULKAN_HPP_NAMESPACE::EXT4444FormatsExtensionName;
  using VULKAN_HPP_NAMESPACE::EXT4444FormatsSpecVersion;

  //=== VK_EXT_device_fault ===
  using VULKAN_HPP_NAMESPACE::EXTDeviceFaultExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTDeviceFaultSpecVersion;

  //=== VK_ARM_rasterization_order_attachment_access ===
  using VULKAN_HPP_NAMESPACE::ARMRasterizationOrderAttachmentAccessExtensionName;
  using VULKAN_HPP_NAMESPACE::ARMRasterizationOrderAttachmentAccessSpecVersion;

  //=== VK_EXT_rgba10x6_formats ===
  using VULKAN_HPP_NAMESPACE::EXTRgba10X6FormatsExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTRgba10X6FormatsSpecVersion;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_NV_acquire_winrt_display ===
  using VULKAN_HPP_NAMESPACE::NVAcquireWinrtDisplayExtensionName;
  using VULKAN_HPP_NAMESPACE::NVAcquireWinrtDisplaySpecVersion;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
  //=== VK_EXT_directfb_surface ===
  using VULKAN_HPP_NAMESPACE::EXTDirectfbSurfaceExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTDirectfbSurfaceSpecVersion;
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

  //=== VK_VALVE_mutable_descriptor_type ===
  using VULKAN_HPP_NAMESPACE::VALVEMutableDescriptorTypeExtensionName;
  using VULKAN_HPP_NAMESPACE::VALVEMutableDescriptorTypeSpecVersion;

  //=== VK_EXT_vertex_input_dynamic_state ===
  using VULKAN_HPP_NAMESPACE::EXTVertexInputDynamicStateExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTVertexInputDynamicStateSpecVersion;

  //=== VK_EXT_physical_device_drm ===
  using VULKAN_HPP_NAMESPACE::EXTPhysicalDeviceDrmExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTPhysicalDeviceDrmSpecVersion;

  //=== VK_EXT_device_address_binding_report ===
  using VULKAN_HPP_NAMESPACE::EXTDeviceAddressBindingReportExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTDeviceAddressBindingReportSpecVersion;

  //=== VK_EXT_depth_clip_control ===
  using VULKAN_HPP_NAMESPACE::EXTDepthClipControlExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTDepthClipControlSpecVersion;

  //=== VK_EXT_primitive_topology_list_restart ===
  using VULKAN_HPP_NAMESPACE::EXTPrimitiveTopologyListRestartExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTPrimitiveTopologyListRestartSpecVersion;

  //=== VK_KHR_format_feature_flags2 ===
  using VULKAN_HPP_NAMESPACE::KHRFormatFeatureFlags2ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRFormatFeatureFlags2SpecVersion;

  //=== VK_EXT_present_mode_fifo_latest_ready ===
  using VULKAN_HPP_NAMESPACE::EXTPresentModeFifoLatestReadyExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTPresentModeFifoLatestReadySpecVersion;

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_external_memory ===
  using VULKAN_HPP_NAMESPACE::FUCHSIAExternalMemoryExtensionName;
  using VULKAN_HPP_NAMESPACE::FUCHSIAExternalMemorySpecVersion;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_external_semaphore ===
  using VULKAN_HPP_NAMESPACE::FUCHSIAExternalSemaphoreExtensionName;
  using VULKAN_HPP_NAMESPACE::FUCHSIAExternalSemaphoreSpecVersion;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_buffer_collection ===
  using VULKAN_HPP_NAMESPACE::FUCHSIABufferCollectionExtensionName;
  using VULKAN_HPP_NAMESPACE::FUCHSIABufferCollectionSpecVersion;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

  //=== VK_HUAWEI_subpass_shading ===
  using VULKAN_HPP_NAMESPACE::HUAWEISubpassShadingExtensionName;
  using VULKAN_HPP_NAMESPACE::HUAWEISubpassShadingSpecVersion;

  //=== VK_HUAWEI_invocation_mask ===
  using VULKAN_HPP_NAMESPACE::HUAWEIInvocationMaskExtensionName;
  using VULKAN_HPP_NAMESPACE::HUAWEIInvocationMaskSpecVersion;

  //=== VK_NV_external_memory_rdma ===
  using VULKAN_HPP_NAMESPACE::NVExternalMemoryRdmaExtensionName;
  using VULKAN_HPP_NAMESPACE::NVExternalMemoryRdmaSpecVersion;

  //=== VK_EXT_pipeline_properties ===
  using VULKAN_HPP_NAMESPACE::EXTPipelinePropertiesExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTPipelinePropertiesSpecVersion;

  //=== VK_EXT_frame_boundary ===
  using VULKAN_HPP_NAMESPACE::EXTFrameBoundaryExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTFrameBoundarySpecVersion;

  //=== VK_EXT_multisampled_render_to_single_sampled ===
  using VULKAN_HPP_NAMESPACE::EXTMultisampledRenderToSingleSampledExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTMultisampledRenderToSingleSampledSpecVersion;

  //=== VK_EXT_extended_dynamic_state2 ===
  using VULKAN_HPP_NAMESPACE::EXTExtendedDynamicState2ExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTExtendedDynamicState2SpecVersion;

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
  //=== VK_QNX_screen_surface ===
  using VULKAN_HPP_NAMESPACE::QNXScreenSurfaceExtensionName;
  using VULKAN_HPP_NAMESPACE::QNXScreenSurfaceSpecVersion;
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

  //=== VK_EXT_color_write_enable ===
  using VULKAN_HPP_NAMESPACE::EXTColorWriteEnableExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTColorWriteEnableSpecVersion;

  //=== VK_EXT_primitives_generated_query ===
  using VULKAN_HPP_NAMESPACE::EXTPrimitivesGeneratedQueryExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTPrimitivesGeneratedQuerySpecVersion;

  //=== VK_KHR_ray_tracing_maintenance1 ===
  using VULKAN_HPP_NAMESPACE::KHRRayTracingMaintenance1ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRRayTracingMaintenance1SpecVersion;

  //=== VK_KHR_shader_untyped_pointers ===
  using VULKAN_HPP_NAMESPACE::KHRShaderUntypedPointersExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRShaderUntypedPointersSpecVersion;

  //=== VK_EXT_global_priority_query ===
  using VULKAN_HPP_NAMESPACE::EXTGlobalPriorityQueryExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTGlobalPriorityQuerySpecVersion;
  using VULKAN_HPP_NAMESPACE::MaxGlobalPrioritySizeEXT;

  //=== VK_VALVE_video_encode_rgb_conversion ===
  using VULKAN_HPP_NAMESPACE::VALVEVideoEncodeRgbConversionExtensionName;
  using VULKAN_HPP_NAMESPACE::VALVEVideoEncodeRgbConversionSpecVersion;

  //=== VK_EXT_image_view_min_lod ===
  using VULKAN_HPP_NAMESPACE::EXTImageViewMinLodExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTImageViewMinLodSpecVersion;

  //=== VK_EXT_multi_draw ===
  using VULKAN_HPP_NAMESPACE::EXTMultiDrawExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTMultiDrawSpecVersion;

  //=== VK_EXT_image_2d_view_of_3d ===
  using VULKAN_HPP_NAMESPACE::EXTImage2DViewOf3DExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTImage2DViewOf3DSpecVersion;

  //=== VK_KHR_portability_enumeration ===
  using VULKAN_HPP_NAMESPACE::KHRPortabilityEnumerationExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRPortabilityEnumerationSpecVersion;

  //=== VK_EXT_shader_tile_image ===
  using VULKAN_HPP_NAMESPACE::EXTShaderTileImageExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTShaderTileImageSpecVersion;

  //=== VK_EXT_opacity_micromap ===
  using VULKAN_HPP_NAMESPACE::EXTOpacityMicromapExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTOpacityMicromapSpecVersion;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_NV_displacement_micromap ===
  using VULKAN_HPP_NAMESPACE::NVDisplacementMicromapExtensionName;
  using VULKAN_HPP_NAMESPACE::NVDisplacementMicromapSpecVersion;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_EXT_load_store_op_none ===
  using VULKAN_HPP_NAMESPACE::EXTLoadStoreOpNoneExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTLoadStoreOpNoneSpecVersion;

  //=== VK_HUAWEI_cluster_culling_shader ===
  using VULKAN_HPP_NAMESPACE::HUAWEIClusterCullingShaderExtensionName;
  using VULKAN_HPP_NAMESPACE::HUAWEIClusterCullingShaderSpecVersion;

  //=== VK_EXT_border_color_swizzle ===
  using VULKAN_HPP_NAMESPACE::EXTBorderColorSwizzleExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTBorderColorSwizzleSpecVersion;

  //=== VK_EXT_pageable_device_local_memory ===
  using VULKAN_HPP_NAMESPACE::EXTPageableDeviceLocalMemoryExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTPageableDeviceLocalMemorySpecVersion;

  //=== VK_KHR_maintenance4 ===
  using VULKAN_HPP_NAMESPACE::KHRMaintenance4ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRMaintenance4SpecVersion;

  //=== VK_ARM_shader_core_properties ===
  using VULKAN_HPP_NAMESPACE::ARMShaderCorePropertiesExtensionName;
  using VULKAN_HPP_NAMESPACE::ARMShaderCorePropertiesSpecVersion;

  //=== VK_KHR_shader_subgroup_rotate ===
  using VULKAN_HPP_NAMESPACE::KHRShaderSubgroupRotateExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRShaderSubgroupRotateSpecVersion;

  //=== VK_ARM_scheduling_controls ===
  using VULKAN_HPP_NAMESPACE::ARMSchedulingControlsExtensionName;
  using VULKAN_HPP_NAMESPACE::ARMSchedulingControlsSpecVersion;

  //=== VK_EXT_image_sliced_view_of_3d ===
  using VULKAN_HPP_NAMESPACE::EXTImageSlicedViewOf3DExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTImageSlicedViewOf3DSpecVersion;
  using VULKAN_HPP_NAMESPACE::Remaining3DSlicesEXT;

  //=== VK_VALVE_descriptor_set_host_mapping ===
  using VULKAN_HPP_NAMESPACE::VALVEDescriptorSetHostMappingExtensionName;
  using VULKAN_HPP_NAMESPACE::VALVEDescriptorSetHostMappingSpecVersion;

  //=== VK_EXT_depth_clamp_zero_one ===
  using VULKAN_HPP_NAMESPACE::EXTDepthClampZeroOneExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTDepthClampZeroOneSpecVersion;

  //=== VK_EXT_non_seamless_cube_map ===
  using VULKAN_HPP_NAMESPACE::EXTNonSeamlessCubeMapExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTNonSeamlessCubeMapSpecVersion;

  //=== VK_ARM_render_pass_striped ===
  using VULKAN_HPP_NAMESPACE::ARMRenderPassStripedExtensionName;
  using VULKAN_HPP_NAMESPACE::ARMRenderPassStripedSpecVersion;

  //=== VK_QCOM_fragment_density_map_offset ===
  using VULKAN_HPP_NAMESPACE::QCOMFragmentDensityMapOffsetExtensionName;
  using VULKAN_HPP_NAMESPACE::QCOMFragmentDensityMapOffsetSpecVersion;

  //=== VK_NV_copy_memory_indirect ===
  using VULKAN_HPP_NAMESPACE::NVCopyMemoryIndirectExtensionName;
  using VULKAN_HPP_NAMESPACE::NVCopyMemoryIndirectSpecVersion;

  //=== VK_NV_memory_decompression ===
  using VULKAN_HPP_NAMESPACE::NVMemoryDecompressionExtensionName;
  using VULKAN_HPP_NAMESPACE::NVMemoryDecompressionSpecVersion;

  //=== VK_NV_device_generated_commands_compute ===
  using VULKAN_HPP_NAMESPACE::NVDeviceGeneratedCommandsComputeExtensionName;
  using VULKAN_HPP_NAMESPACE::NVDeviceGeneratedCommandsComputeSpecVersion;

  //=== VK_NV_ray_tracing_linear_swept_spheres ===
  using VULKAN_HPP_NAMESPACE::NVRayTracingLinearSweptSpheresExtensionName;
  using VULKAN_HPP_NAMESPACE::NVRayTracingLinearSweptSpheresSpecVersion;

  //=== VK_NV_linear_color_attachment ===
  using VULKAN_HPP_NAMESPACE::NVLinearColorAttachmentExtensionName;
  using VULKAN_HPP_NAMESPACE::NVLinearColorAttachmentSpecVersion;

  //=== VK_GOOGLE_surfaceless_query ===
  using VULKAN_HPP_NAMESPACE::GOOGLESurfacelessQueryExtensionName;
  using VULKAN_HPP_NAMESPACE::GOOGLESurfacelessQuerySpecVersion;

  //=== VK_KHR_shader_maximal_reconvergence ===
  using VULKAN_HPP_NAMESPACE::KHRShaderMaximalReconvergenceExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRShaderMaximalReconvergenceSpecVersion;

  //=== VK_EXT_image_compression_control_swapchain ===
  using VULKAN_HPP_NAMESPACE::EXTImageCompressionControlSwapchainExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTImageCompressionControlSwapchainSpecVersion;

  //=== VK_QCOM_image_processing ===
  using VULKAN_HPP_NAMESPACE::QCOMImageProcessingExtensionName;
  using VULKAN_HPP_NAMESPACE::QCOMImageProcessingSpecVersion;

  //=== VK_EXT_nested_command_buffer ===
  using VULKAN_HPP_NAMESPACE::EXTNestedCommandBufferExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTNestedCommandBufferSpecVersion;

#if defined( VK_USE_PLATFORM_OHOS )
  //=== VK_OHOS_external_memory ===
  using VULKAN_HPP_NAMESPACE::OHOSExternalMemoryExtensionName;
  using VULKAN_HPP_NAMESPACE::OHOSExternalMemorySpecVersion;
#endif /*VK_USE_PLATFORM_OHOS*/

  //=== VK_EXT_external_memory_acquire_unmodified ===
  using VULKAN_HPP_NAMESPACE::EXTExternalMemoryAcquireUnmodifiedExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTExternalMemoryAcquireUnmodifiedSpecVersion;

  //=== VK_EXT_extended_dynamic_state3 ===
  using VULKAN_HPP_NAMESPACE::EXTExtendedDynamicState3ExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTExtendedDynamicState3SpecVersion;

  //=== VK_EXT_subpass_merge_feedback ===
  using VULKAN_HPP_NAMESPACE::EXTSubpassMergeFeedbackExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTSubpassMergeFeedbackSpecVersion;

  //=== VK_LUNARG_direct_driver_loading ===
  using VULKAN_HPP_NAMESPACE::LUNARGDirectDriverLoadingExtensionName;
  using VULKAN_HPP_NAMESPACE::LUNARGDirectDriverLoadingSpecVersion;

  //=== VK_ARM_tensors ===
  using VULKAN_HPP_NAMESPACE::ARMTensorsExtensionName;
  using VULKAN_HPP_NAMESPACE::ARMTensorsSpecVersion;

  //=== VK_EXT_shader_module_identifier ===
  using VULKAN_HPP_NAMESPACE::EXTShaderModuleIdentifierExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTShaderModuleIdentifierSpecVersion;
  using VULKAN_HPP_NAMESPACE::MaxShaderModuleIdentifierSizeEXT;

  //=== VK_EXT_rasterization_order_attachment_access ===
  using VULKAN_HPP_NAMESPACE::EXTRasterizationOrderAttachmentAccessExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTRasterizationOrderAttachmentAccessSpecVersion;

  //=== VK_NV_optical_flow ===
  using VULKAN_HPP_NAMESPACE::NVOpticalFlowExtensionName;
  using VULKAN_HPP_NAMESPACE::NVOpticalFlowSpecVersion;

  //=== VK_EXT_legacy_dithering ===
  using VULKAN_HPP_NAMESPACE::EXTLegacyDitheringExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTLegacyDitheringSpecVersion;

  //=== VK_EXT_pipeline_protected_access ===
  using VULKAN_HPP_NAMESPACE::EXTPipelineProtectedAccessExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTPipelineProtectedAccessSpecVersion;

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  //=== VK_ANDROID_external_format_resolve ===
  using VULKAN_HPP_NAMESPACE::ANDROIDExternalFormatResolveExtensionName;
  using VULKAN_HPP_NAMESPACE::ANDROIDExternalFormatResolveSpecVersion;
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

  //=== VK_KHR_maintenance5 ===
  using VULKAN_HPP_NAMESPACE::KHRMaintenance5ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRMaintenance5SpecVersion;

  //=== VK_AMD_anti_lag ===
  using VULKAN_HPP_NAMESPACE::AMDAntiLagExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDAntiLagSpecVersion;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_AMDX_dense_geometry_format ===
  using VULKAN_HPP_NAMESPACE::AMDXDenseGeometryFormatExtensionName;
  using VULKAN_HPP_NAMESPACE::AMDXDenseGeometryFormatSpecVersion;
  using VULKAN_HPP_NAMESPACE::CompressedTriangleFormatDgf1ByteAlignmentAMDX;
  using VULKAN_HPP_NAMESPACE::CompressedTriangleFormatDgf1ByteStrideAMDX;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_KHR_present_id2 ===
  using VULKAN_HPP_NAMESPACE::KHRPresentId2ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRPresentId2SpecVersion;

  //=== VK_KHR_present_wait2 ===
  using VULKAN_HPP_NAMESPACE::KHRPresentWait2ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRPresentWait2SpecVersion;

  //=== VK_KHR_ray_tracing_position_fetch ===
  using VULKAN_HPP_NAMESPACE::KHRRayTracingPositionFetchExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRRayTracingPositionFetchSpecVersion;

  //=== VK_EXT_shader_object ===
  using VULKAN_HPP_NAMESPACE::EXTShaderObjectExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTShaderObjectSpecVersion;

  //=== VK_KHR_pipeline_binary ===
  using VULKAN_HPP_NAMESPACE::KHRPipelineBinaryExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRPipelineBinarySpecVersion;
  using VULKAN_HPP_NAMESPACE::MaxPipelineBinaryKeySizeKHR;

  //=== VK_QCOM_tile_properties ===
  using VULKAN_HPP_NAMESPACE::QCOMTilePropertiesExtensionName;
  using VULKAN_HPP_NAMESPACE::QCOMTilePropertiesSpecVersion;

  //=== VK_SEC_amigo_profiling ===
  using VULKAN_HPP_NAMESPACE::SECAmigoProfilingExtensionName;
  using VULKAN_HPP_NAMESPACE::SECAmigoProfilingSpecVersion;

  //=== VK_KHR_surface_maintenance1 ===
  using VULKAN_HPP_NAMESPACE::KHRSurfaceMaintenance1ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRSurfaceMaintenance1SpecVersion;

  //=== VK_KHR_swapchain_maintenance1 ===
  using VULKAN_HPP_NAMESPACE::KHRSwapchainMaintenance1ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRSwapchainMaintenance1SpecVersion;

  //=== VK_QCOM_multiview_per_view_viewports ===
  using VULKAN_HPP_NAMESPACE::QCOMMultiviewPerViewViewportsExtensionName;
  using VULKAN_HPP_NAMESPACE::QCOMMultiviewPerViewViewportsSpecVersion;

  //=== VK_NV_ray_tracing_invocation_reorder ===
  using VULKAN_HPP_NAMESPACE::NVRayTracingInvocationReorderExtensionName;
  using VULKAN_HPP_NAMESPACE::NVRayTracingInvocationReorderSpecVersion;

  //=== VK_NV_cooperative_vector ===
  using VULKAN_HPP_NAMESPACE::NVCooperativeVectorExtensionName;
  using VULKAN_HPP_NAMESPACE::NVCooperativeVectorSpecVersion;

  //=== VK_NV_extended_sparse_address_space ===
  using VULKAN_HPP_NAMESPACE::NVExtendedSparseAddressSpaceExtensionName;
  using VULKAN_HPP_NAMESPACE::NVExtendedSparseAddressSpaceSpecVersion;

  //=== VK_EXT_mutable_descriptor_type ===
  using VULKAN_HPP_NAMESPACE::EXTMutableDescriptorTypeExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTMutableDescriptorTypeSpecVersion;

  //=== VK_EXT_legacy_vertex_attributes ===
  using VULKAN_HPP_NAMESPACE::EXTLegacyVertexAttributesExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTLegacyVertexAttributesSpecVersion;

  //=== VK_EXT_layer_settings ===
  using VULKAN_HPP_NAMESPACE::EXTLayerSettingsExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTLayerSettingsSpecVersion;

  //=== VK_ARM_shader_core_builtins ===
  using VULKAN_HPP_NAMESPACE::ARMShaderCoreBuiltinsExtensionName;
  using VULKAN_HPP_NAMESPACE::ARMShaderCoreBuiltinsSpecVersion;

  //=== VK_EXT_pipeline_library_group_handles ===
  using VULKAN_HPP_NAMESPACE::EXTPipelineLibraryGroupHandlesExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTPipelineLibraryGroupHandlesSpecVersion;

  //=== VK_EXT_dynamic_rendering_unused_attachments ===
  using VULKAN_HPP_NAMESPACE::EXTDynamicRenderingUnusedAttachmentsExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTDynamicRenderingUnusedAttachmentsSpecVersion;

  //=== VK_NV_low_latency2 ===
  using VULKAN_HPP_NAMESPACE::NVLowLatency2ExtensionName;
  using VULKAN_HPP_NAMESPACE::NVLowLatency2SpecVersion;

  //=== VK_KHR_cooperative_matrix ===
  using VULKAN_HPP_NAMESPACE::KHRCooperativeMatrixExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRCooperativeMatrixSpecVersion;

  //=== VK_ARM_data_graph ===
  using VULKAN_HPP_NAMESPACE::ARMDataGraphExtensionName;
  using VULKAN_HPP_NAMESPACE::ARMDataGraphSpecVersion;
  using VULKAN_HPP_NAMESPACE::MaxPhysicalDeviceDataGraphOperationSetNameSizeARM;

  //=== VK_QCOM_multiview_per_view_render_areas ===
  using VULKAN_HPP_NAMESPACE::QCOMMultiviewPerViewRenderAreasExtensionName;
  using VULKAN_HPP_NAMESPACE::QCOMMultiviewPerViewRenderAreasSpecVersion;

  //=== VK_KHR_compute_shader_derivatives ===
  using VULKAN_HPP_NAMESPACE::KHRComputeShaderDerivativesExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRComputeShaderDerivativesSpecVersion;

  //=== VK_KHR_video_decode_av1 ===
  using VULKAN_HPP_NAMESPACE::KHRVideoDecodeAv1ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRVideoDecodeAv1SpecVersion;
  using VULKAN_HPP_NAMESPACE::MaxVideoAv1ReferencesPerFrameKHR;

  //=== VK_KHR_video_encode_av1 ===
  using VULKAN_HPP_NAMESPACE::KHRVideoEncodeAv1ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRVideoEncodeAv1SpecVersion;

  //=== VK_KHR_video_decode_vp9 ===
  using VULKAN_HPP_NAMESPACE::KHRVideoDecodeVp9ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRVideoDecodeVp9SpecVersion;
  using VULKAN_HPP_NAMESPACE::MaxVideoVp9ReferencesPerFrameKHR;

  //=== VK_KHR_video_maintenance1 ===
  using VULKAN_HPP_NAMESPACE::KHRVideoMaintenance1ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRVideoMaintenance1SpecVersion;

  //=== VK_NV_per_stage_descriptor_set ===
  using VULKAN_HPP_NAMESPACE::NVPerStageDescriptorSetExtensionName;
  using VULKAN_HPP_NAMESPACE::NVPerStageDescriptorSetSpecVersion;

  //=== VK_QCOM_image_processing2 ===
  using VULKAN_HPP_NAMESPACE::QCOMImageProcessing2ExtensionName;
  using VULKAN_HPP_NAMESPACE::QCOMImageProcessing2SpecVersion;

  //=== VK_QCOM_filter_cubic_weights ===
  using VULKAN_HPP_NAMESPACE::QCOMFilterCubicWeightsExtensionName;
  using VULKAN_HPP_NAMESPACE::QCOMFilterCubicWeightsSpecVersion;

  //=== VK_QCOM_ycbcr_degamma ===
  using VULKAN_HPP_NAMESPACE::QCOMYcbcrDegammaExtensionName;
  using VULKAN_HPP_NAMESPACE::QCOMYcbcrDegammaSpecVersion;

  //=== VK_QCOM_filter_cubic_clamp ===
  using VULKAN_HPP_NAMESPACE::QCOMFilterCubicClampExtensionName;
  using VULKAN_HPP_NAMESPACE::QCOMFilterCubicClampSpecVersion;

  //=== VK_EXT_attachment_feedback_loop_dynamic_state ===
  using VULKAN_HPP_NAMESPACE::EXTAttachmentFeedbackLoopDynamicStateExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTAttachmentFeedbackLoopDynamicStateSpecVersion;

  //=== VK_KHR_vertex_attribute_divisor ===
  using VULKAN_HPP_NAMESPACE::KHRVertexAttributeDivisorExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRVertexAttributeDivisorSpecVersion;

  //=== VK_KHR_load_store_op_none ===
  using VULKAN_HPP_NAMESPACE::KHRLoadStoreOpNoneExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRLoadStoreOpNoneSpecVersion;

  //=== VK_KHR_unified_image_layouts ===
  using VULKAN_HPP_NAMESPACE::KHRUnifiedImageLayoutsExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRUnifiedImageLayoutsSpecVersion;

  //=== VK_KHR_shader_float_controls2 ===
  using VULKAN_HPP_NAMESPACE::KHRShaderFloatControls2ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRShaderFloatControls2SpecVersion;

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
  //=== VK_QNX_external_memory_screen_buffer ===
  using VULKAN_HPP_NAMESPACE::QNXExternalMemoryScreenBufferExtensionName;
  using VULKAN_HPP_NAMESPACE::QNXExternalMemoryScreenBufferSpecVersion;
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

  //=== VK_MSFT_layered_driver ===
  using VULKAN_HPP_NAMESPACE::MSFTLayeredDriverExtensionName;
  using VULKAN_HPP_NAMESPACE::MSFTLayeredDriverSpecVersion;

  //=== VK_KHR_index_type_uint8 ===
  using VULKAN_HPP_NAMESPACE::KHRIndexTypeUint8ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRIndexTypeUint8SpecVersion;

  //=== VK_KHR_line_rasterization ===
  using VULKAN_HPP_NAMESPACE::KHRLineRasterizationExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRLineRasterizationSpecVersion;

  //=== VK_KHR_calibrated_timestamps ===
  using VULKAN_HPP_NAMESPACE::KHRCalibratedTimestampsExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRCalibratedTimestampsSpecVersion;

  //=== VK_KHR_shader_expect_assume ===
  using VULKAN_HPP_NAMESPACE::KHRShaderExpectAssumeExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRShaderExpectAssumeSpecVersion;

  //=== VK_KHR_maintenance6 ===
  using VULKAN_HPP_NAMESPACE::KHRMaintenance6ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRMaintenance6SpecVersion;

  //=== VK_NV_descriptor_pool_overallocation ===
  using VULKAN_HPP_NAMESPACE::NVDescriptorPoolOverallocationExtensionName;
  using VULKAN_HPP_NAMESPACE::NVDescriptorPoolOverallocationSpecVersion;

  //=== VK_QCOM_tile_memory_heap ===
  using VULKAN_HPP_NAMESPACE::QCOMTileMemoryHeapExtensionName;
  using VULKAN_HPP_NAMESPACE::QCOMTileMemoryHeapSpecVersion;

  //=== VK_KHR_copy_memory_indirect ===
  using VULKAN_HPP_NAMESPACE::KHRCopyMemoryIndirectExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRCopyMemoryIndirectSpecVersion;

  //=== VK_EXT_memory_decompression ===
  using VULKAN_HPP_NAMESPACE::EXTMemoryDecompressionExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTMemoryDecompressionSpecVersion;

  //=== VK_NV_display_stereo ===
  using VULKAN_HPP_NAMESPACE::NVDisplayStereoExtensionName;
  using VULKAN_HPP_NAMESPACE::NVDisplayStereoSpecVersion;

  //=== VK_KHR_video_encode_intra_refresh ===
  using VULKAN_HPP_NAMESPACE::KHRVideoEncodeIntraRefreshExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRVideoEncodeIntraRefreshSpecVersion;

  //=== VK_KHR_video_encode_quantization_map ===
  using VULKAN_HPP_NAMESPACE::KHRVideoEncodeQuantizationMapExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRVideoEncodeQuantizationMapSpecVersion;

  //=== VK_NV_raw_access_chains ===
  using VULKAN_HPP_NAMESPACE::NVRawAccessChainsExtensionName;
  using VULKAN_HPP_NAMESPACE::NVRawAccessChainsSpecVersion;

  //=== VK_NV_external_compute_queue ===
  using VULKAN_HPP_NAMESPACE::NVExternalComputeQueueExtensionName;
  using VULKAN_HPP_NAMESPACE::NVExternalComputeQueueSpecVersion;

  //=== VK_KHR_shader_relaxed_extended_instruction ===
  using VULKAN_HPP_NAMESPACE::KHRShaderRelaxedExtendedInstructionExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRShaderRelaxedExtendedInstructionSpecVersion;

  //=== VK_NV_command_buffer_inheritance ===
  using VULKAN_HPP_NAMESPACE::NVCommandBufferInheritanceExtensionName;
  using VULKAN_HPP_NAMESPACE::NVCommandBufferInheritanceSpecVersion;

  //=== VK_KHR_maintenance7 ===
  using VULKAN_HPP_NAMESPACE::KHRMaintenance7ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRMaintenance7SpecVersion;

  //=== VK_NV_shader_atomic_float16_vector ===
  using VULKAN_HPP_NAMESPACE::NVShaderAtomicFloat16VectorExtensionName;
  using VULKAN_HPP_NAMESPACE::NVShaderAtomicFloat16VectorSpecVersion;

  //=== VK_EXT_shader_replicated_composites ===
  using VULKAN_HPP_NAMESPACE::EXTShaderReplicatedCompositesExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTShaderReplicatedCompositesSpecVersion;

  //=== VK_EXT_shader_float8 ===
  using VULKAN_HPP_NAMESPACE::EXTShaderFloat8ExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTShaderFloat8SpecVersion;

  //=== VK_NV_ray_tracing_validation ===
  using VULKAN_HPP_NAMESPACE::NVRayTracingValidationExtensionName;
  using VULKAN_HPP_NAMESPACE::NVRayTracingValidationSpecVersion;

  //=== VK_NV_cluster_acceleration_structure ===
  using VULKAN_HPP_NAMESPACE::NVClusterAccelerationStructureExtensionName;
  using VULKAN_HPP_NAMESPACE::NVClusterAccelerationStructureSpecVersion;

  //=== VK_NV_partitioned_acceleration_structure ===
  using VULKAN_HPP_NAMESPACE::NVPartitionedAccelerationStructureExtensionName;
  using VULKAN_HPP_NAMESPACE::NVPartitionedAccelerationStructureSpecVersion;
  using VULKAN_HPP_NAMESPACE::PartitionedAccelerationStructurePartitionIndexGlobalNV;

  //=== VK_EXT_device_generated_commands ===
  using VULKAN_HPP_NAMESPACE::EXTDeviceGeneratedCommandsExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTDeviceGeneratedCommandsSpecVersion;

  //=== VK_KHR_maintenance8 ===
  using VULKAN_HPP_NAMESPACE::KHRMaintenance8ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRMaintenance8SpecVersion;

  //=== VK_MESA_image_alignment_control ===
  using VULKAN_HPP_NAMESPACE::MESAImageAlignmentControlExtensionName;
  using VULKAN_HPP_NAMESPACE::MESAImageAlignmentControlSpecVersion;

  //=== VK_KHR_shader_fma ===
  using VULKAN_HPP_NAMESPACE::KHRShaderFmaExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRShaderFmaSpecVersion;

  //=== VK_EXT_ray_tracing_invocation_reorder ===
  using VULKAN_HPP_NAMESPACE::EXTRayTracingInvocationReorderExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTRayTracingInvocationReorderSpecVersion;

  //=== VK_EXT_depth_clamp_control ===
  using VULKAN_HPP_NAMESPACE::EXTDepthClampControlExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTDepthClampControlSpecVersion;

  //=== VK_KHR_maintenance9 ===
  using VULKAN_HPP_NAMESPACE::KHRMaintenance9ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRMaintenance9SpecVersion;

  //=== VK_KHR_video_maintenance2 ===
  using VULKAN_HPP_NAMESPACE::KHRVideoMaintenance2ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRVideoMaintenance2SpecVersion;

#if defined( VK_USE_PLATFORM_OHOS )
  //=== VK_OHOS_surface ===
  using VULKAN_HPP_NAMESPACE::OHOSSurfaceExtensionName;
  using VULKAN_HPP_NAMESPACE::OHOSSurfaceSpecVersion;
#endif /*VK_USE_PLATFORM_OHOS*/

#if defined( VK_USE_PLATFORM_OHOS )
  //=== VK_OHOS_native_buffer ===
  using VULKAN_HPP_NAMESPACE::OHOSNativeBufferExtensionName;
  using VULKAN_HPP_NAMESPACE::OHOSNativeBufferSpecVersion;
#endif /*VK_USE_PLATFORM_OHOS*/

  //=== VK_HUAWEI_hdr_vivid ===
  using VULKAN_HPP_NAMESPACE::HUAWEIHdrVividExtensionName;
  using VULKAN_HPP_NAMESPACE::HUAWEIHdrVividSpecVersion;

  //=== VK_NV_cooperative_matrix2 ===
  using VULKAN_HPP_NAMESPACE::NVCooperativeMatrix2ExtensionName;
  using VULKAN_HPP_NAMESPACE::NVCooperativeMatrix2SpecVersion;

  //=== VK_ARM_pipeline_opacity_micromap ===
  using VULKAN_HPP_NAMESPACE::ARMPipelineOpacityMicromapExtensionName;
  using VULKAN_HPP_NAMESPACE::ARMPipelineOpacityMicromapSpecVersion;

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_external_memory_metal ===
  using VULKAN_HPP_NAMESPACE::EXTExternalMemoryMetalExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTExternalMemoryMetalSpecVersion;
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_KHR_depth_clamp_zero_one ===
  using VULKAN_HPP_NAMESPACE::KHRDepthClampZeroOneExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRDepthClampZeroOneSpecVersion;

  //=== VK_ARM_performance_counters_by_region ===
  using VULKAN_HPP_NAMESPACE::ARMPerformanceCountersByRegionExtensionName;
  using VULKAN_HPP_NAMESPACE::ARMPerformanceCountersByRegionSpecVersion;

  //=== VK_EXT_vertex_attribute_robustness ===
  using VULKAN_HPP_NAMESPACE::EXTVertexAttributeRobustnessExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTVertexAttributeRobustnessSpecVersion;

  //=== VK_ARM_format_pack ===
  using VULKAN_HPP_NAMESPACE::ARMFormatPackExtensionName;
  using VULKAN_HPP_NAMESPACE::ARMFormatPackSpecVersion;

  //=== VK_VALVE_fragment_density_map_layered ===
  using VULKAN_HPP_NAMESPACE::VALVEFragmentDensityMapLayeredExtensionName;
  using VULKAN_HPP_NAMESPACE::VALVEFragmentDensityMapLayeredSpecVersion;

  //=== VK_KHR_robustness2 ===
  using VULKAN_HPP_NAMESPACE::KHRRobustness2ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRRobustness2SpecVersion;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_NV_present_metering ===
  using VULKAN_HPP_NAMESPACE::NVPresentMeteringExtensionName;
  using VULKAN_HPP_NAMESPACE::NVPresentMeteringSpecVersion;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_EXT_fragment_density_map_offset ===
  using VULKAN_HPP_NAMESPACE::EXTFragmentDensityMapOffsetExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTFragmentDensityMapOffsetSpecVersion;

  //=== VK_EXT_zero_initialize_device_memory ===
  using VULKAN_HPP_NAMESPACE::EXTZeroInitializeDeviceMemoryExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTZeroInitializeDeviceMemorySpecVersion;

  //=== VK_KHR_present_mode_fifo_latest_ready ===
  using VULKAN_HPP_NAMESPACE::KHRPresentModeFifoLatestReadyExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRPresentModeFifoLatestReadySpecVersion;

  //=== VK_EXT_shader_64bit_indexing ===
  using VULKAN_HPP_NAMESPACE::EXTShader64BitIndexingExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTShader64BitIndexingSpecVersion;

  //=== VK_EXT_custom_resolve ===
  using VULKAN_HPP_NAMESPACE::EXTCustomResolveExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTCustomResolveSpecVersion;

  //=== VK_QCOM_data_graph_model ===
  using VULKAN_HPP_NAMESPACE::DataGraphModelToolchainVersionLengthQCOM;
  using VULKAN_HPP_NAMESPACE::QCOMDataGraphModelExtensionName;
  using VULKAN_HPP_NAMESPACE::QCOMDataGraphModelSpecVersion;

  //=== VK_KHR_maintenance10 ===
  using VULKAN_HPP_NAMESPACE::KHRMaintenance10ExtensionName;
  using VULKAN_HPP_NAMESPACE::KHRMaintenance10SpecVersion;

  //=== VK_SEC_pipeline_cache_incremental_mode ===
  using VULKAN_HPP_NAMESPACE::SECPipelineCacheIncrementalModeExtensionName;
  using VULKAN_HPP_NAMESPACE::SECPipelineCacheIncrementalModeSpecVersion;

  //=== VK_EXT_shader_uniform_buffer_unsized_array ===
  using VULKAN_HPP_NAMESPACE::EXTShaderUniformBufferUnsizedArrayExtensionName;
  using VULKAN_HPP_NAMESPACE::EXTShaderUniformBufferUnsizedArraySpecVersion;

  //========================
  //=== CONSTEXPR VALUEs ===
  //========================
  using VULKAN_HPP_NAMESPACE::HeaderVersion;
  using VULKAN_HPP_NAMESPACE::Use64BitPtrDefines;

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
  //=== CONSTEXPR CALLERs ===
  //==========================
  using VULKAN_HPP_NAMESPACE::ApiVersion;
  using VULKAN_HPP_NAMESPACE::ApiVersion10;
  using VULKAN_HPP_NAMESPACE::ApiVersion11;
  using VULKAN_HPP_NAMESPACE::ApiVersion12;
  using VULKAN_HPP_NAMESPACE::ApiVersion13;
  using VULKAN_HPP_NAMESPACE::ApiVersion14;
  using VULKAN_HPP_NAMESPACE::HeaderVersionComplete;

  //====================
  //=== FUNCPOINTERs ===
  //====================

  //=== VK_VERSION_1_0 ===
  using VULKAN_HPP_NAMESPACE::PFN_AllocationFunction;
  using VULKAN_HPP_NAMESPACE::PFN_FreeFunction;
  using VULKAN_HPP_NAMESPACE::PFN_InternalAllocationNotification;
  using VULKAN_HPP_NAMESPACE::PFN_InternalFreeNotification;
  using VULKAN_HPP_NAMESPACE::PFN_ReallocationFunction;
  using VULKAN_HPP_NAMESPACE::PFN_VoidFunction;

  //=== VK_EXT_debug_report ===
  using VULKAN_HPP_NAMESPACE::PFN_DebugReportCallbackEXT;

  //=== VK_EXT_debug_utils ===
  using VULKAN_HPP_NAMESPACE::PFN_DebugUtilsMessengerCallbackEXT;

  //=== VK_EXT_device_memory_report ===
  using VULKAN_HPP_NAMESPACE::PFN_DeviceMemoryReportCallbackEXT;

  //=== VK_LUNARG_direct_driver_loading ===
  using VULKAN_HPP_NAMESPACE::PFN_GetInstanceProcAddrLUNARG;

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

  //=== VK_VERSION_1_4 ===
  using VULKAN_HPP_NAMESPACE::BindDescriptorSetsInfo;
  using VULKAN_HPP_NAMESPACE::BindDescriptorSetsInfoKHR;
  using VULKAN_HPP_NAMESPACE::BindMemoryStatus;
  using VULKAN_HPP_NAMESPACE::BindMemoryStatusKHR;
  using VULKAN_HPP_NAMESPACE::BufferUsageFlags2CreateInfo;
  using VULKAN_HPP_NAMESPACE::BufferUsageFlags2CreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::CopyImageToImageInfo;
  using VULKAN_HPP_NAMESPACE::CopyImageToImageInfoEXT;
  using VULKAN_HPP_NAMESPACE::CopyImageToMemoryInfo;
  using VULKAN_HPP_NAMESPACE::CopyImageToMemoryInfoEXT;
  using VULKAN_HPP_NAMESPACE::CopyMemoryToImageInfo;
  using VULKAN_HPP_NAMESPACE::CopyMemoryToImageInfoEXT;
  using VULKAN_HPP_NAMESPACE::DeviceImageSubresourceInfo;
  using VULKAN_HPP_NAMESPACE::DeviceImageSubresourceInfoKHR;
  using VULKAN_HPP_NAMESPACE::DeviceQueueGlobalPriorityCreateInfo;
  using VULKAN_HPP_NAMESPACE::DeviceQueueGlobalPriorityCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::DeviceQueueGlobalPriorityCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::HostImageCopyDevicePerformanceQuery;
  using VULKAN_HPP_NAMESPACE::HostImageCopyDevicePerformanceQueryEXT;
  using VULKAN_HPP_NAMESPACE::HostImageLayoutTransitionInfo;
  using VULKAN_HPP_NAMESPACE::HostImageLayoutTransitionInfoEXT;
  using VULKAN_HPP_NAMESPACE::ImageSubresource2;
  using VULKAN_HPP_NAMESPACE::ImageSubresource2EXT;
  using VULKAN_HPP_NAMESPACE::ImageSubresource2KHR;
  using VULKAN_HPP_NAMESPACE::ImageToMemoryCopy;
  using VULKAN_HPP_NAMESPACE::ImageToMemoryCopyEXT;
  using VULKAN_HPP_NAMESPACE::MemoryMapInfo;
  using VULKAN_HPP_NAMESPACE::MemoryMapInfoKHR;
  using VULKAN_HPP_NAMESPACE::MemoryToImageCopy;
  using VULKAN_HPP_NAMESPACE::MemoryToImageCopyEXT;
  using VULKAN_HPP_NAMESPACE::MemoryUnmapInfo;
  using VULKAN_HPP_NAMESPACE::MemoryUnmapInfoKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDynamicRenderingLocalReadFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceGlobalPriorityQueryFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceGlobalPriorityQueryFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceGlobalPriorityQueryFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceHostImageCopyFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceHostImageCopyFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceHostImageCopyProperties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceHostImageCopyPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceIndexTypeUint8Features;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceIndexTypeUint8FeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceIndexTypeUint8FeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceLineRasterizationFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceLineRasterizationFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceLineRasterizationFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceLineRasterizationProperties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceLineRasterizationPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceLineRasterizationPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance5Features;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance5FeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance5Properties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance5PropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance6Features;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance6FeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance6Properties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance6PropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineProtectedAccessFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineProtectedAccessFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineRobustnessFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineRobustnessFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineRobustnessProperties;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineRobustnessPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePushDescriptorProperties;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePushDescriptorPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderExpectAssumeFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderExpectAssumeFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderFloatControls2Features;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderFloatControls2FeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSubgroupRotateFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSubgroupRotateFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeDivisorFeatures;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeDivisorFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeDivisorFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeDivisorProperties;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeDivisorPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan14Features;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan14Properties;
  using VULKAN_HPP_NAMESPACE::PipelineCreateFlags2CreateInfo;
  using VULKAN_HPP_NAMESPACE::PipelineCreateFlags2CreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::PipelineRasterizationLineStateCreateInfo;
  using VULKAN_HPP_NAMESPACE::PipelineRasterizationLineStateCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::PipelineRasterizationLineStateCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::PipelineRobustnessCreateInfo;
  using VULKAN_HPP_NAMESPACE::PipelineRobustnessCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::PipelineVertexInputDivisorStateCreateInfo;
  using VULKAN_HPP_NAMESPACE::PipelineVertexInputDivisorStateCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::PipelineVertexInputDivisorStateCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::PushConstantsInfo;
  using VULKAN_HPP_NAMESPACE::PushConstantsInfoKHR;
  using VULKAN_HPP_NAMESPACE::PushDescriptorSetInfo;
  using VULKAN_HPP_NAMESPACE::PushDescriptorSetInfoKHR;
  using VULKAN_HPP_NAMESPACE::PushDescriptorSetWithTemplateInfo;
  using VULKAN_HPP_NAMESPACE::PushDescriptorSetWithTemplateInfoKHR;
  using VULKAN_HPP_NAMESPACE::QueueFamilyGlobalPriorityProperties;
  using VULKAN_HPP_NAMESPACE::QueueFamilyGlobalPriorityPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::QueueFamilyGlobalPriorityPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::RenderingAreaInfo;
  using VULKAN_HPP_NAMESPACE::RenderingAreaInfoKHR;
  using VULKAN_HPP_NAMESPACE::RenderingAttachmentLocationInfo;
  using VULKAN_HPP_NAMESPACE::RenderingAttachmentLocationInfoKHR;
  using VULKAN_HPP_NAMESPACE::RenderingInputAttachmentIndexInfo;
  using VULKAN_HPP_NAMESPACE::RenderingInputAttachmentIndexInfoKHR;
  using VULKAN_HPP_NAMESPACE::SubresourceHostMemcpySize;
  using VULKAN_HPP_NAMESPACE::SubresourceHostMemcpySizeEXT;
  using VULKAN_HPP_NAMESPACE::SubresourceLayout2;
  using VULKAN_HPP_NAMESPACE::SubresourceLayout2EXT;
  using VULKAN_HPP_NAMESPACE::SubresourceLayout2KHR;
  using VULKAN_HPP_NAMESPACE::VertexInputBindingDivisorDescription;
  using VULKAN_HPP_NAMESPACE::VertexInputBindingDivisorDescriptionEXT;
  using VULKAN_HPP_NAMESPACE::VertexInputBindingDivisorDescriptionKHR;

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
  using VULKAN_HPP_NAMESPACE::CuModuleTexturingModeCreateInfoNVX;

  //=== VK_NVX_image_view_handle ===
  using VULKAN_HPP_NAMESPACE::ImageViewAddressPropertiesNVX;
  using VULKAN_HPP_NAMESPACE::ImageViewHandleInfoNVX;

  //=== VK_KHR_video_encode_h264 ===
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264CapabilitiesKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264DpbSlotInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264FrameSizeKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264GopRemainingFrameInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264NaluSliceInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264PictureInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264ProfileInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264QpKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264QualityLevelPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264RateControlInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264RateControlLayerInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersAddInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersFeedbackInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersGetInfoKHR;

  //=== VK_KHR_video_encode_h265 ===
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265CapabilitiesKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265DpbSlotInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265FrameSizeKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265GopRemainingFrameInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265NaluSliceSegmentInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265PictureInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265ProfileInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265QpKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265QualityLevelPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265RateControlInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265RateControlLayerInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersAddInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersFeedbackInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersGetInfoKHR;

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
  using VULKAN_HPP_NAMESPACE::MultiviewPerViewAttributesInfoNVX;
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

  //=== VK_IMG_relaxed_line_rasterization ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRelaxedLineRasterizationFeaturesIMG;

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

  //=== VK_AMD_mixed_attachment_samples ===
  using VULKAN_HPP_NAMESPACE::AttachmentSampleCountInfoAMD;
  using VULKAN_HPP_NAMESPACE::AttachmentSampleCountInfoNV;

  //=== VK_KHR_shader_bfloat16 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderBfloat16FeaturesKHR;

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

  //=== VK_AMD_shader_core_properties ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCorePropertiesAMD;

  //=== VK_KHR_video_decode_h265 ===
  using VULKAN_HPP_NAMESPACE::VideoDecodeH265CapabilitiesKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeH265DpbSlotInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeH265PictureInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeH265ProfileInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeH265SessionParametersAddInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeH265SessionParametersCreateInfoKHR;

  //=== VK_AMD_memory_overallocation_behavior ===
  using VULKAN_HPP_NAMESPACE::DeviceMemoryOverallocationCreateInfoAMD;

  //=== VK_EXT_vertex_attribute_divisor ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeDivisorPropertiesEXT;

#if defined( VK_USE_PLATFORM_GGP )
  //=== VK_GGP_frame_token ===
  using VULKAN_HPP_NAMESPACE::PresentFrameTokenGGP;
#endif /*VK_USE_PLATFORM_GGP*/

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
  using VULKAN_HPP_NAMESPACE::CheckpointData2NV;
  using VULKAN_HPP_NAMESPACE::CheckpointDataNV;
  using VULKAN_HPP_NAMESPACE::QueueFamilyCheckpointProperties2NV;
  using VULKAN_HPP_NAMESPACE::QueueFamilyCheckpointPropertiesNV;

  //=== VK_EXT_present_timing ===
  using VULKAN_HPP_NAMESPACE::PastPresentationTimingEXT;
  using VULKAN_HPP_NAMESPACE::PastPresentationTimingInfoEXT;
  using VULKAN_HPP_NAMESPACE::PastPresentationTimingPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePresentTimingFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PresentStageTimeEXT;
  using VULKAN_HPP_NAMESPACE::PresentTimingInfoEXT;
  using VULKAN_HPP_NAMESPACE::PresentTimingsInfoEXT;
  using VULKAN_HPP_NAMESPACE::PresentTimingSurfaceCapabilitiesEXT;
  using VULKAN_HPP_NAMESPACE::SwapchainCalibratedTimestampInfoEXT;
  using VULKAN_HPP_NAMESPACE::SwapchainTimeDomainPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::SwapchainTimingPropertiesEXT;

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
  using VULKAN_HPP_NAMESPACE::RenderingFragmentDensityMapAttachmentInfoEXT;
  using VULKAN_HPP_NAMESPACE::RenderPassFragmentDensityMapCreateInfoEXT;

  //=== VK_KHR_fragment_shading_rate ===
  using VULKAN_HPP_NAMESPACE::FragmentShadingRateAttachmentInfoKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRatePropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PipelineFragmentShadingRateStateCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::RenderingFragmentShadingRateAttachmentInfoKHR;

  //=== VK_AMD_shader_core_properties2 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCoreProperties2AMD;

  //=== VK_AMD_device_coherent_memory ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCoherentMemoryFeaturesAMD;

  //=== VK_EXT_shader_image_atomic_int64 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT;

  //=== VK_KHR_shader_quad_control ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderQuadControlFeaturesKHR;

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

  //=== VK_EXT_shader_atomic_float ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicFloatFeaturesEXT;

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

  //=== VK_EXT_map_memory_placed ===
  using VULKAN_HPP_NAMESPACE::MemoryMapPlacedInfoEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMapMemoryPlacedFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMapMemoryPlacedPropertiesEXT;

  //=== VK_EXT_shader_atomic_float2 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicFloat2FeaturesEXT;

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

  //=== VK_NV_device_diagnostics_config ===
  using VULKAN_HPP_NAMESPACE::DeviceDiagnosticsConfigCreateInfoNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDiagnosticsConfigFeaturesNV;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_NV_cuda_kernel_launch ===
  using VULKAN_HPP_NAMESPACE::CudaFunctionCreateInfoNV;
  using VULKAN_HPP_NAMESPACE::CudaLaunchInfoNV;
  using VULKAN_HPP_NAMESPACE::CudaModuleCreateInfoNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCudaKernelLaunchFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCudaKernelLaunchPropertiesNV;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_QCOM_tile_shading ===
  using VULKAN_HPP_NAMESPACE::DispatchTileInfoQCOM;
  using VULKAN_HPP_NAMESPACE::PerTileBeginInfoQCOM;
  using VULKAN_HPP_NAMESPACE::PerTileEndInfoQCOM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceTileShadingFeaturesQCOM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceTileShadingPropertiesQCOM;
  using VULKAN_HPP_NAMESPACE::RenderPassTileShadingCreateInfoQCOM;

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

  //=== VK_EXT_frame_boundary ===
  using VULKAN_HPP_NAMESPACE::FrameBoundaryEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFrameBoundaryFeaturesEXT;

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

  //=== VK_KHR_shader_untyped_pointers ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderUntypedPointersFeaturesKHR;

  //=== VK_VALVE_video_encode_rgb_conversion ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoEncodeRgbConversionFeaturesVALVE;
  using VULKAN_HPP_NAMESPACE::VideoEncodeProfileRgbConversionInfoVALVE;
  using VULKAN_HPP_NAMESPACE::VideoEncodeRgbConversionCapabilitiesVALVE;
  using VULKAN_HPP_NAMESPACE::VideoEncodeSessionRgbConversionCreateInfoVALVE;

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
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceClusterCullingShaderVrsFeaturesHUAWEI;

  //=== VK_EXT_border_color_swizzle ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceBorderColorSwizzleFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::SamplerBorderColorComponentMappingCreateInfoEXT;

  //=== VK_EXT_pageable_device_local_memory ===
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT;

  //=== VK_ARM_shader_core_properties ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCorePropertiesARM;

  //=== VK_ARM_scheduling_controls ===
  using VULKAN_HPP_NAMESPACE::DeviceQueueShaderCoreControlCreateInfoARM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSchedulingControlsFeaturesARM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSchedulingControlsPropertiesARM;

  //=== VK_EXT_image_sliced_view_of_3d ===
  using VULKAN_HPP_NAMESPACE::ImageViewSlicedCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceImageSlicedViewOf3DFeaturesEXT;

  //=== VK_VALVE_descriptor_set_host_mapping ===
  using VULKAN_HPP_NAMESPACE::DescriptorSetBindingReferenceVALVE;
  using VULKAN_HPP_NAMESPACE::DescriptorSetLayoutHostMappingInfoVALVE;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorSetHostMappingFeaturesVALVE;

  //=== VK_EXT_non_seamless_cube_map ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceNonSeamlessCubeMapFeaturesEXT;

  //=== VK_ARM_render_pass_striped ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRenderPassStripedFeaturesARM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRenderPassStripedPropertiesARM;
  using VULKAN_HPP_NAMESPACE::RenderPassStripeBeginInfoARM;
  using VULKAN_HPP_NAMESPACE::RenderPassStripeInfoARM;
  using VULKAN_HPP_NAMESPACE::RenderPassStripeSubmitInfoARM;

  //=== VK_NV_copy_memory_indirect ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCopyMemoryIndirectFeaturesNV;

  //=== VK_NV_memory_decompression ===
  using VULKAN_HPP_NAMESPACE::DecompressMemoryRegionNV;

  //=== VK_NV_device_generated_commands_compute ===
  using VULKAN_HPP_NAMESPACE::BindPipelineIndirectCommandNV;
  using VULKAN_HPP_NAMESPACE::ComputePipelineIndirectBufferInfoNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceGeneratedCommandsComputeFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PipelineIndirectDeviceAddressInfoNV;

  //=== VK_NV_ray_tracing_linear_swept_spheres ===
  using VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryLinearSweptSpheresDataNV;
  using VULKAN_HPP_NAMESPACE::AccelerationStructureGeometrySpheresDataNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingLinearSweptSpheresFeaturesNV;

  //=== VK_NV_linear_color_attachment ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceLinearColorAttachmentFeaturesNV;

  //=== VK_KHR_shader_maximal_reconvergence ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR;

  //=== VK_EXT_image_compression_control_swapchain ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceImageCompressionControlSwapchainFeaturesEXT;

  //=== VK_QCOM_image_processing ===
  using VULKAN_HPP_NAMESPACE::ImageViewSampleWeightCreateInfoQCOM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessingFeaturesQCOM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessingPropertiesQCOM;

  //=== VK_EXT_nested_command_buffer ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceNestedCommandBufferFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceNestedCommandBufferPropertiesEXT;

#if defined( VK_USE_PLATFORM_OHOS )
  //=== VK_OHOS_external_memory ===
  using VULKAN_HPP_NAMESPACE::ExternalFormatOHOS;
  using VULKAN_HPP_NAMESPACE::ImportNativeBufferInfoOHOS;
  using VULKAN_HPP_NAMESPACE::MemoryGetNativeBufferInfoOHOS;
  using VULKAN_HPP_NAMESPACE::NativeBufferFormatPropertiesOHOS;
  using VULKAN_HPP_NAMESPACE::NativeBufferPropertiesOHOS;
  using VULKAN_HPP_NAMESPACE::NativeBufferUsageOHOS;
#endif /*VK_USE_PLATFORM_OHOS*/

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

  //=== VK_ARM_tensors ===
  using VULKAN_HPP_NAMESPACE::BindTensorMemoryInfoARM;
  using VULKAN_HPP_NAMESPACE::CopyTensorInfoARM;
  using VULKAN_HPP_NAMESPACE::DescriptorGetTensorInfoARM;
  using VULKAN_HPP_NAMESPACE::DeviceTensorMemoryRequirementsARM;
  using VULKAN_HPP_NAMESPACE::ExternalMemoryTensorCreateInfoARM;
  using VULKAN_HPP_NAMESPACE::ExternalTensorPropertiesARM;
  using VULKAN_HPP_NAMESPACE::FrameBoundaryTensorsARM;
  using VULKAN_HPP_NAMESPACE::MemoryDedicatedAllocateInfoTensorARM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorBufferTensorFeaturesARM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorBufferTensorPropertiesARM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalTensorInfoARM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceTensorFeaturesARM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceTensorPropertiesARM;
  using VULKAN_HPP_NAMESPACE::TensorCaptureDescriptorDataInfoARM;
  using VULKAN_HPP_NAMESPACE::TensorCopyARM;
  using VULKAN_HPP_NAMESPACE::TensorCreateInfoARM;
  using VULKAN_HPP_NAMESPACE::TensorDependencyInfoARM;
  using VULKAN_HPP_NAMESPACE::TensorDescriptionARM;
  using VULKAN_HPP_NAMESPACE::TensorFormatPropertiesARM;
  using VULKAN_HPP_NAMESPACE::TensorMemoryBarrierARM;
  using VULKAN_HPP_NAMESPACE::TensorMemoryRequirementsInfoARM;
  using VULKAN_HPP_NAMESPACE::TensorViewCaptureDescriptorDataInfoARM;
  using VULKAN_HPP_NAMESPACE::TensorViewCreateInfoARM;
  using VULKAN_HPP_NAMESPACE::WriteDescriptorSetTensorARM;

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

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  //=== VK_ANDROID_external_format_resolve ===
  using VULKAN_HPP_NAMESPACE::AndroidHardwareBufferFormatResolvePropertiesANDROID;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalFormatResolveFeaturesANDROID;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalFormatResolvePropertiesANDROID;
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

  //=== VK_AMD_anti_lag ===
  using VULKAN_HPP_NAMESPACE::AntiLagDataAMD;
  using VULKAN_HPP_NAMESPACE::AntiLagPresentationInfoAMD;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceAntiLagFeaturesAMD;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_AMDX_dense_geometry_format ===
  using VULKAN_HPP_NAMESPACE::AccelerationStructureDenseGeometryFormatTrianglesDataAMDX;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDenseGeometryFormatFeaturesAMDX;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_KHR_present_id2 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePresentId2FeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PresentId2KHR;
  using VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesPresentId2KHR;

  //=== VK_KHR_present_wait2 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePresentWait2FeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PresentWait2InfoKHR;
  using VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesPresentWait2KHR;

  //=== VK_KHR_ray_tracing_position_fetch ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPositionFetchFeaturesKHR;

  //=== VK_EXT_shader_object ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderObjectFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderObjectPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::ShaderCreateInfoEXT;

  //=== VK_KHR_pipeline_binary ===
  using VULKAN_HPP_NAMESPACE::DevicePipelineBinaryInternalCacheControlKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineBinaryFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineBinaryPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PipelineBinaryCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::PipelineBinaryDataInfoKHR;
  using VULKAN_HPP_NAMESPACE::PipelineBinaryDataKHR;
  using VULKAN_HPP_NAMESPACE::PipelineBinaryHandlesInfoKHR;
  using VULKAN_HPP_NAMESPACE::PipelineBinaryInfoKHR;
  using VULKAN_HPP_NAMESPACE::PipelineBinaryKeyKHR;
  using VULKAN_HPP_NAMESPACE::PipelineBinaryKeysAndDataKHR;
  using VULKAN_HPP_NAMESPACE::PipelineCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::ReleaseCapturedPipelineDataInfoKHR;

  //=== VK_QCOM_tile_properties ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceTilePropertiesFeaturesQCOM;
  using VULKAN_HPP_NAMESPACE::TilePropertiesQCOM;

  //=== VK_SEC_amigo_profiling ===
  using VULKAN_HPP_NAMESPACE::AmigoProfilingSubmitInfoSEC;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceAmigoProfilingFeaturesSEC;

  //=== VK_KHR_surface_maintenance1 ===
  using VULKAN_HPP_NAMESPACE::SurfacePresentModeCompatibilityEXT;
  using VULKAN_HPP_NAMESPACE::SurfacePresentModeCompatibilityKHR;
  using VULKAN_HPP_NAMESPACE::SurfacePresentModeEXT;
  using VULKAN_HPP_NAMESPACE::SurfacePresentModeKHR;
  using VULKAN_HPP_NAMESPACE::SurfacePresentScalingCapabilitiesEXT;
  using VULKAN_HPP_NAMESPACE::SurfacePresentScalingCapabilitiesKHR;

  //=== VK_KHR_swapchain_maintenance1 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSwapchainMaintenance1FeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceSwapchainMaintenance1FeaturesKHR;
  using VULKAN_HPP_NAMESPACE::ReleaseSwapchainImagesInfoEXT;
  using VULKAN_HPP_NAMESPACE::ReleaseSwapchainImagesInfoKHR;
  using VULKAN_HPP_NAMESPACE::SwapchainPresentFenceInfoEXT;
  using VULKAN_HPP_NAMESPACE::SwapchainPresentFenceInfoKHR;
  using VULKAN_HPP_NAMESPACE::SwapchainPresentModeInfoEXT;
  using VULKAN_HPP_NAMESPACE::SwapchainPresentModeInfoKHR;
  using VULKAN_HPP_NAMESPACE::SwapchainPresentModesCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::SwapchainPresentModesCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::SwapchainPresentScalingCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::SwapchainPresentScalingCreateInfoKHR;

  //=== VK_QCOM_multiview_per_view_viewports ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewPerViewViewportsFeaturesQCOM;

  //=== VK_NV_ray_tracing_invocation_reorder ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingInvocationReorderFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingInvocationReorderPropertiesNV;

  //=== VK_NV_cooperative_vector ===
  using VULKAN_HPP_NAMESPACE::ConvertCooperativeVectorMatrixInfoNV;
  using VULKAN_HPP_NAMESPACE::CooperativeVectorPropertiesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeVectorFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeVectorPropertiesNV;

  //=== VK_NV_extended_sparse_address_space ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedSparseAddressSpaceFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedSparseAddressSpacePropertiesNV;

  //=== VK_EXT_mutable_descriptor_type ===
  using VULKAN_HPP_NAMESPACE::MutableDescriptorTypeCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::MutableDescriptorTypeCreateInfoVALVE;
  using VULKAN_HPP_NAMESPACE::MutableDescriptorTypeListEXT;
  using VULKAN_HPP_NAMESPACE::MutableDescriptorTypeListVALVE;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMutableDescriptorTypeFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMutableDescriptorTypeFeaturesVALVE;

  //=== VK_EXT_legacy_vertex_attributes ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceLegacyVertexAttributesFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceLegacyVertexAttributesPropertiesEXT;

  //=== VK_EXT_layer_settings ===
  using VULKAN_HPP_NAMESPACE::LayerSettingEXT;
  using VULKAN_HPP_NAMESPACE::LayerSettingsCreateInfoEXT;

  //=== VK_ARM_shader_core_builtins ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCoreBuiltinsFeaturesARM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCoreBuiltinsPropertiesARM;

  //=== VK_EXT_pipeline_library_group_handles ===
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineLibraryGroupHandlesFeaturesEXT;

  //=== VK_EXT_dynamic_rendering_unused_attachments ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT;

  //=== VK_NV_low_latency2 ===
  using VULKAN_HPP_NAMESPACE::GetLatencyMarkerInfoNV;
  using VULKAN_HPP_NAMESPACE::LatencySleepInfoNV;
  using VULKAN_HPP_NAMESPACE::LatencySleepModeInfoNV;
  using VULKAN_HPP_NAMESPACE::LatencySubmissionPresentIdNV;
  using VULKAN_HPP_NAMESPACE::LatencySurfaceCapabilitiesNV;
  using VULKAN_HPP_NAMESPACE::LatencyTimingsFrameReportNV;
  using VULKAN_HPP_NAMESPACE::OutOfBandQueueTypeInfoNV;
  using VULKAN_HPP_NAMESPACE::SetLatencyMarkerInfoNV;
  using VULKAN_HPP_NAMESPACE::SwapchainLatencyCreateInfoNV;

  //=== VK_KHR_cooperative_matrix ===
  using VULKAN_HPP_NAMESPACE::CooperativeMatrixPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixPropertiesKHR;

  //=== VK_ARM_data_graph ===
  using VULKAN_HPP_NAMESPACE::BindDataGraphPipelineSessionMemoryInfoARM;
  using VULKAN_HPP_NAMESPACE::DataGraphPipelineCompilerControlCreateInfoARM;
  using VULKAN_HPP_NAMESPACE::DataGraphPipelineConstantARM;
  using VULKAN_HPP_NAMESPACE::DataGraphPipelineConstantTensorSemiStructuredSparsityInfoARM;
  using VULKAN_HPP_NAMESPACE::DataGraphPipelineCreateInfoARM;
  using VULKAN_HPP_NAMESPACE::DataGraphPipelineDispatchInfoARM;
  using VULKAN_HPP_NAMESPACE::DataGraphPipelineIdentifierCreateInfoARM;
  using VULKAN_HPP_NAMESPACE::DataGraphPipelineInfoARM;
  using VULKAN_HPP_NAMESPACE::DataGraphPipelinePropertyQueryResultARM;
  using VULKAN_HPP_NAMESPACE::DataGraphPipelineResourceInfoARM;
  using VULKAN_HPP_NAMESPACE::DataGraphPipelineSessionBindPointRequirementARM;
  using VULKAN_HPP_NAMESPACE::DataGraphPipelineSessionBindPointRequirementsInfoARM;
  using VULKAN_HPP_NAMESPACE::DataGraphPipelineSessionCreateInfoARM;
  using VULKAN_HPP_NAMESPACE::DataGraphPipelineSessionMemoryRequirementsInfoARM;
  using VULKAN_HPP_NAMESPACE::DataGraphPipelineShaderModuleCreateInfoARM;
  using VULKAN_HPP_NAMESPACE::DataGraphProcessingEngineCreateInfoARM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDataGraphFeaturesARM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDataGraphOperationSupportARM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDataGraphProcessingEngineARM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceQueueFamilyDataGraphProcessingEngineInfoARM;
  using VULKAN_HPP_NAMESPACE::QueueFamilyDataGraphProcessingEnginePropertiesARM;
  using VULKAN_HPP_NAMESPACE::QueueFamilyDataGraphPropertiesARM;

  //=== VK_QCOM_multiview_per_view_render_areas ===
  using VULKAN_HPP_NAMESPACE::MultiviewPerViewRenderAreasRenderPassBeginInfoQCOM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewPerViewRenderAreasFeaturesQCOM;

  //=== VK_KHR_compute_shader_derivatives ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceComputeShaderDerivativesFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceComputeShaderDerivativesFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceComputeShaderDerivativesPropertiesKHR;

  //=== VK_KHR_video_decode_av1 ===
  using VULKAN_HPP_NAMESPACE::VideoDecodeAV1CapabilitiesKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeAV1DpbSlotInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeAV1PictureInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeAV1ProfileInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeAV1SessionParametersCreateInfoKHR;

  //=== VK_KHR_video_encode_av1 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoEncodeAV1FeaturesKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeAV1CapabilitiesKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeAV1DpbSlotInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeAV1FrameSizeKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeAV1GopRemainingFrameInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeAV1PictureInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeAV1ProfileInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeAV1QIndexKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeAV1QualityLevelPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeAV1RateControlInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeAV1RateControlLayerInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeAV1SessionCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeAV1SessionParametersCreateInfoKHR;

  //=== VK_KHR_video_decode_vp9 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoDecodeVP9FeaturesKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeVP9CapabilitiesKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeVP9PictureInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeVP9ProfileInfoKHR;

  //=== VK_KHR_video_maintenance1 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoMaintenance1FeaturesKHR;
  using VULKAN_HPP_NAMESPACE::VideoInlineQueryInfoKHR;

  //=== VK_NV_per_stage_descriptor_set ===
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePerStageDescriptorSetFeaturesNV;

  //=== VK_QCOM_image_processing2 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessing2FeaturesQCOM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessing2PropertiesQCOM;
  using VULKAN_HPP_NAMESPACE::SamplerBlockMatchWindowCreateInfoQCOM;

  //=== VK_QCOM_filter_cubic_weights ===
  using VULKAN_HPP_NAMESPACE::BlitImageCubicWeightsInfoQCOM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCubicWeightsFeaturesQCOM;
  using VULKAN_HPP_NAMESPACE::SamplerCubicWeightsCreateInfoQCOM;

  //=== VK_QCOM_ycbcr_degamma ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceYcbcrDegammaFeaturesQCOM;
  using VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionYcbcrDegammaCreateInfoQCOM;

  //=== VK_QCOM_filter_cubic_clamp ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCubicClampFeaturesQCOM;

  //=== VK_EXT_attachment_feedback_loop_dynamic_state ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT;

  //=== VK_KHR_unified_image_layouts ===
  using VULKAN_HPP_NAMESPACE::AttachmentFeedbackLoopInfoEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR;

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
  //=== VK_QNX_external_memory_screen_buffer ===
  using VULKAN_HPP_NAMESPACE::ExternalFormatQNX;
  using VULKAN_HPP_NAMESPACE::ImportScreenBufferInfoQNX;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalMemoryScreenBufferFeaturesQNX;
  using VULKAN_HPP_NAMESPACE::ScreenBufferFormatPropertiesQNX;
  using VULKAN_HPP_NAMESPACE::ScreenBufferPropertiesQNX;
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

  //=== VK_MSFT_layered_driver ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceLayeredDriverPropertiesMSFT;

  //=== VK_KHR_calibrated_timestamps ===
  using VULKAN_HPP_NAMESPACE::CalibratedTimestampInfoEXT;
  using VULKAN_HPP_NAMESPACE::CalibratedTimestampInfoKHR;

  //=== VK_KHR_maintenance6 ===
  using VULKAN_HPP_NAMESPACE::BindDescriptorBufferEmbeddedSamplersInfoEXT;
  using VULKAN_HPP_NAMESPACE::SetDescriptorBufferOffsetsInfoEXT;

  //=== VK_NV_descriptor_pool_overallocation ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorPoolOverallocationFeaturesNV;

  //=== VK_QCOM_tile_memory_heap ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceTileMemoryHeapFeaturesQCOM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceTileMemoryHeapPropertiesQCOM;
  using VULKAN_HPP_NAMESPACE::TileMemoryBindInfoQCOM;
  using VULKAN_HPP_NAMESPACE::TileMemoryRequirementsQCOM;
  using VULKAN_HPP_NAMESPACE::TileMemorySizeInfoQCOM;

  //=== VK_KHR_copy_memory_indirect ===
  using VULKAN_HPP_NAMESPACE::CopyMemoryIndirectCommandKHR;
  using VULKAN_HPP_NAMESPACE::CopyMemoryIndirectCommandNV;
  using VULKAN_HPP_NAMESPACE::CopyMemoryIndirectInfoKHR;
  using VULKAN_HPP_NAMESPACE::CopyMemoryToImageIndirectCommandKHR;
  using VULKAN_HPP_NAMESPACE::CopyMemoryToImageIndirectCommandNV;
  using VULKAN_HPP_NAMESPACE::CopyMemoryToImageIndirectInfoKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCopyMemoryIndirectFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCopyMemoryIndirectPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCopyMemoryIndirectPropertiesNV;
  using VULKAN_HPP_NAMESPACE::StridedDeviceAddressRangeKHR;

  //=== VK_EXT_memory_decompression ===
  using VULKAN_HPP_NAMESPACE::DecompressMemoryInfoEXT;
  using VULKAN_HPP_NAMESPACE::DecompressMemoryRegionEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryDecompressionFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryDecompressionFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryDecompressionPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryDecompressionPropertiesNV;

  //=== VK_NV_display_stereo ===
  using VULKAN_HPP_NAMESPACE::DisplayModeStereoPropertiesNV;
  using VULKAN_HPP_NAMESPACE::DisplaySurfaceStereoCreateInfoNV;

  //=== VK_KHR_video_encode_intra_refresh ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoEncodeIntraRefreshFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeIntraRefreshCapabilitiesKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeIntraRefreshInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeSessionIntraRefreshCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoReferenceIntraRefreshInfoKHR;

  //=== VK_KHR_video_encode_quantization_map ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoEncodeQuantizationMapFeaturesKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeAV1QuantizationMapCapabilitiesKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH264QuantizationMapCapabilitiesKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeH265QuantizationMapCapabilitiesKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeQuantizationMapCapabilitiesKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeQuantizationMapInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoEncodeQuantizationMapSessionParametersCreateInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoFormatAV1QuantizationMapPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::VideoFormatH265QuantizationMapPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::VideoFormatQuantizationMapPropertiesKHR;

  //=== VK_NV_raw_access_chains ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRawAccessChainsFeaturesNV;

  //=== VK_NV_external_compute_queue ===
  using VULKAN_HPP_NAMESPACE::ExternalComputeQueueCreateInfoNV;
  using VULKAN_HPP_NAMESPACE::ExternalComputeQueueDataParamsNV;
  using VULKAN_HPP_NAMESPACE::ExternalComputeQueueDeviceCreateInfoNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalComputeQueuePropertiesNV;

  //=== VK_KHR_shader_relaxed_extended_instruction ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderRelaxedExtendedInstructionFeaturesKHR;

  //=== VK_NV_command_buffer_inheritance ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCommandBufferInheritanceFeaturesNV;

  //=== VK_KHR_maintenance7 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceLayeredApiPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceLayeredApiPropertiesListKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceLayeredApiVulkanPropertiesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance7FeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance7PropertiesKHR;

  //=== VK_NV_shader_atomic_float16_vector ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicFloat16VectorFeaturesNV;

  //=== VK_EXT_shader_replicated_composites ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderReplicatedCompositesFeaturesEXT;

  //=== VK_EXT_shader_float8 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderFloat8FeaturesEXT;

  //=== VK_NV_ray_tracing_validation ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingValidationFeaturesNV;

  //=== VK_NV_cluster_acceleration_structure ===
  using VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureBuildClustersBottomLevelInfoNV;
  using VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureBuildTriangleClusterInfoNV;
  using VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV;
  using VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureClustersBottomLevelInputNV;
  using VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureCommandsInfoNV;
  using VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureGeometryIndexAndGeometryFlagsNV;
  using VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureGetTemplateIndicesInfoNV;
  using VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureInputInfoNV;
  using VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureInstantiateClusterInfoNV;
  using VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureMoveObjectsInfoNV;
  using VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureMoveObjectsInputNV;
  using VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureOpInputNV;
  using VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureTriangleClusterInputNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceClusterAccelerationStructureFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceClusterAccelerationStructurePropertiesNV;
  using VULKAN_HPP_NAMESPACE::RayTracingPipelineClusterAccelerationStructureCreateInfoNV;
  using VULKAN_HPP_NAMESPACE::StridedDeviceAddressNV;

  //=== VK_NV_partitioned_acceleration_structure ===
  using VULKAN_HPP_NAMESPACE::BuildPartitionedAccelerationStructureIndirectCommandNV;
  using VULKAN_HPP_NAMESPACE::BuildPartitionedAccelerationStructureInfoNV;
  using VULKAN_HPP_NAMESPACE::PartitionedAccelerationStructureFlagsNV;
  using VULKAN_HPP_NAMESPACE::PartitionedAccelerationStructureInstancesInputNV;
  using VULKAN_HPP_NAMESPACE::PartitionedAccelerationStructureUpdateInstanceDataNV;
  using VULKAN_HPP_NAMESPACE::PartitionedAccelerationStructureWriteInstanceDataNV;
  using VULKAN_HPP_NAMESPACE::PartitionedAccelerationStructureWritePartitionTranslationDataNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePartitionedAccelerationStructureFeaturesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePartitionedAccelerationStructurePropertiesNV;
  using VULKAN_HPP_NAMESPACE::WriteDescriptorSetPartitionedAccelerationStructureNV;

  //=== VK_EXT_device_generated_commands ===
  using VULKAN_HPP_NAMESPACE::BindIndexBufferIndirectCommandEXT;
  using VULKAN_HPP_NAMESPACE::BindVertexBufferIndirectCommandEXT;
  using VULKAN_HPP_NAMESPACE::DrawIndirectCountIndirectCommandEXT;
  using VULKAN_HPP_NAMESPACE::GeneratedCommandsInfoEXT;
  using VULKAN_HPP_NAMESPACE::GeneratedCommandsMemoryRequirementsInfoEXT;
  using VULKAN_HPP_NAMESPACE::GeneratedCommandsPipelineInfoEXT;
  using VULKAN_HPP_NAMESPACE::GeneratedCommandsShaderInfoEXT;
  using VULKAN_HPP_NAMESPACE::IndirectCommandsExecutionSetTokenEXT;
  using VULKAN_HPP_NAMESPACE::IndirectCommandsIndexBufferTokenEXT;
  using VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutTokenEXT;
  using VULKAN_HPP_NAMESPACE::IndirectCommandsPushConstantTokenEXT;
  using VULKAN_HPP_NAMESPACE::IndirectCommandsTokenDataEXT;
  using VULKAN_HPP_NAMESPACE::IndirectCommandsVertexBufferTokenEXT;
  using VULKAN_HPP_NAMESPACE::IndirectExecutionSetCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::IndirectExecutionSetInfoEXT;
  using VULKAN_HPP_NAMESPACE::IndirectExecutionSetPipelineInfoEXT;
  using VULKAN_HPP_NAMESPACE::IndirectExecutionSetShaderInfoEXT;
  using VULKAN_HPP_NAMESPACE::IndirectExecutionSetShaderLayoutInfoEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceGeneratedCommandsFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceGeneratedCommandsPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::WriteIndirectExecutionSetPipelineEXT;
  using VULKAN_HPP_NAMESPACE::WriteIndirectExecutionSetShaderEXT;

  //=== VK_KHR_maintenance8 ===
  using VULKAN_HPP_NAMESPACE::MemoryBarrierAccessFlags3KHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance8FeaturesKHR;

  //=== VK_MESA_image_alignment_control ===
  using VULKAN_HPP_NAMESPACE::ImageAlignmentControlCreateInfoMESA;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceImageAlignmentControlFeaturesMESA;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceImageAlignmentControlPropertiesMESA;

  //=== VK_KHR_shader_fma ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderFmaFeaturesKHR;

  //=== VK_EXT_ray_tracing_invocation_reorder ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingInvocationReorderFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingInvocationReorderPropertiesEXT;

  //=== VK_EXT_depth_clamp_control ===
  using VULKAN_HPP_NAMESPACE::DepthClampRangeEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClampControlFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PipelineViewportDepthClampControlCreateInfoEXT;

  //=== VK_KHR_maintenance9 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance9FeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance9PropertiesKHR;
  using VULKAN_HPP_NAMESPACE::QueueFamilyOwnershipTransferPropertiesKHR;

  //=== VK_KHR_video_maintenance2 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoMaintenance2FeaturesKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeAV1InlineSessionParametersInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeH264InlineSessionParametersInfoKHR;
  using VULKAN_HPP_NAMESPACE::VideoDecodeH265InlineSessionParametersInfoKHR;

#if defined( VK_USE_PLATFORM_OHOS )
  //=== VK_OHOS_surface ===
  using VULKAN_HPP_NAMESPACE::SurfaceCreateInfoOHOS;
#endif /*VK_USE_PLATFORM_OHOS*/

#if defined( VK_USE_PLATFORM_OHOS )
  //=== VK_OHOS_native_buffer ===
  using VULKAN_HPP_NAMESPACE::NativeBufferOHOS;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePresentationPropertiesOHOS;
  using VULKAN_HPP_NAMESPACE::SwapchainImageCreateInfoOHOS;
#endif /*VK_USE_PLATFORM_OHOS*/

  //=== VK_HUAWEI_hdr_vivid ===
  using VULKAN_HPP_NAMESPACE::HdrVividDynamicMetadataHUAWEI;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceHdrVividFeaturesHUAWEI;

  //=== VK_NV_cooperative_matrix2 ===
  using VULKAN_HPP_NAMESPACE::CooperativeMatrixFlexibleDimensionsPropertiesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrix2FeaturesNV;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrix2PropertiesNV;

  //=== VK_ARM_pipeline_opacity_micromap ===
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineOpacityMicromapFeaturesARM;

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_external_memory_metal ===
  using VULKAN_HPP_NAMESPACE::ImportMemoryMetalHandleInfoEXT;
  using VULKAN_HPP_NAMESPACE::MemoryGetMetalHandleInfoEXT;
  using VULKAN_HPP_NAMESPACE::MemoryMetalHandlePropertiesEXT;
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_KHR_depth_clamp_zero_one ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClampZeroOneFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClampZeroOneFeaturesKHR;

  //=== VK_ARM_performance_counters_by_region ===
  using VULKAN_HPP_NAMESPACE::PerformanceCounterARM;
  using VULKAN_HPP_NAMESPACE::PerformanceCounterDescriptionARM;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePerformanceCountersByRegionFeaturesARM;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePerformanceCountersByRegionPropertiesARM;
  using VULKAN_HPP_NAMESPACE::RenderPassPerformanceCountersByRegionBeginInfoARM;

  //=== VK_EXT_vertex_attribute_robustness ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeRobustnessFeaturesEXT;

  //=== VK_ARM_format_pack ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFormatPackFeaturesARM;

  //=== VK_VALVE_fragment_density_map_layered ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapLayeredFeaturesVALVE;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapLayeredPropertiesVALVE;
  using VULKAN_HPP_NAMESPACE::PipelineFragmentDensityMapLayeredCreateInfoVALVE;

  //=== VK_KHR_robustness2 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRobustness2FeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRobustness2FeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRobustness2PropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceRobustness2PropertiesKHR;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_NV_present_metering ===
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePresentMeteringFeaturesNV;
  using VULKAN_HPP_NAMESPACE::SetPresentConfigNV;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_EXT_fragment_density_map_offset ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapOffsetFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapOffsetFeaturesQCOM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapOffsetPropertiesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapOffsetPropertiesQCOM;
  using VULKAN_HPP_NAMESPACE::RenderPassFragmentDensityMapOffsetEndInfoEXT;
  using VULKAN_HPP_NAMESPACE::SubpassFragmentDensityMapOffsetEndInfoQCOM;

  //=== VK_EXT_zero_initialize_device_memory ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceZeroInitializeDeviceMemoryFeaturesEXT;

  //=== VK_KHR_present_mode_fifo_latest_ready ===
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePresentModeFifoLatestReadyFeaturesEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePresentModeFifoLatestReadyFeaturesKHR;

  //=== VK_EXT_shader_64bit_indexing ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShader64BitIndexingFeaturesEXT;

  //=== VK_EXT_custom_resolve ===
  using VULKAN_HPP_NAMESPACE::BeginCustomResolveInfoEXT;
  using VULKAN_HPP_NAMESPACE::CustomResolveCreateInfoEXT;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceCustomResolveFeaturesEXT;

  //=== VK_QCOM_data_graph_model ===
  using VULKAN_HPP_NAMESPACE::DataGraphPipelineBuiltinModelCreateInfoQCOM;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceDataGraphModelFeaturesQCOM;
  using VULKAN_HPP_NAMESPACE::PipelineCacheHeaderVersionDataGraphQCOM;

  //=== VK_KHR_maintenance10 ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance10FeaturesKHR;
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance10PropertiesKHR;
  using VULKAN_HPP_NAMESPACE::RenderingAttachmentFlagsInfoKHR;
  using VULKAN_HPP_NAMESPACE::RenderingEndInfoEXT;
  using VULKAN_HPP_NAMESPACE::RenderingEndInfoKHR;
  using VULKAN_HPP_NAMESPACE::ResolveImageModeInfoKHR;

  //=== VK_SEC_pipeline_cache_incremental_mode ===
  using VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineCacheIncrementalModeFeaturesSEC;

  //=== VK_EXT_shader_uniform_buffer_unsized_array ===
  using VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderUniformBufferUnsizedArrayFeaturesEXT;

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

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_NV_cuda_kernel_launch ===
  using VULKAN_HPP_NAMESPACE::CudaFunctionNV;
  using VULKAN_HPP_NAMESPACE::CudaModuleNV;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_buffer_collection ===
  using VULKAN_HPP_NAMESPACE::BufferCollectionFUCHSIA;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

  //=== VK_EXT_opacity_micromap ===
  using VULKAN_HPP_NAMESPACE::MicromapEXT;

  //=== VK_ARM_tensors ===
  using VULKAN_HPP_NAMESPACE::TensorARM;
  using VULKAN_HPP_NAMESPACE::TensorViewARM;

  //=== VK_NV_optical_flow ===
  using VULKAN_HPP_NAMESPACE::OpticalFlowSessionNV;

  //=== VK_EXT_shader_object ===
  using VULKAN_HPP_NAMESPACE::ShaderEXT;

  //=== VK_KHR_pipeline_binary ===
  using VULKAN_HPP_NAMESPACE::PipelineBinaryKHR;

  //=== VK_ARM_data_graph ===
  using VULKAN_HPP_NAMESPACE::DataGraphPipelineSessionARM;

  //=== VK_NV_external_compute_queue ===
  using VULKAN_HPP_NAMESPACE::ExternalComputeQueueNV;

  //=== VK_EXT_device_generated_commands ===
  using VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutEXT;
  using VULKAN_HPP_NAMESPACE::IndirectExecutionSetEXT;

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

  //=== VK_KHR_display ===
  using VULKAN_HPP_NAMESPACE::UniqueDisplayKHR;

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

  //=== VK_INTEL_performance_query ===
  using VULKAN_HPP_NAMESPACE::UniquePerformanceConfigurationINTEL;

  //=== VK_KHR_deferred_host_operations ===
  using VULKAN_HPP_NAMESPACE::UniqueDeferredOperationKHR;

  //=== VK_NV_device_generated_commands ===
  using VULKAN_HPP_NAMESPACE::UniqueIndirectCommandsLayoutNV;

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_NV_cuda_kernel_launch ===
  using VULKAN_HPP_NAMESPACE::UniqueCudaFunctionNV;
  using VULKAN_HPP_NAMESPACE::UniqueCudaModuleNV;
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_buffer_collection ===
  using VULKAN_HPP_NAMESPACE::UniqueBufferCollectionFUCHSIA;
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

  //=== VK_EXT_opacity_micromap ===
  using VULKAN_HPP_NAMESPACE::UniqueMicromapEXT;

  //=== VK_ARM_tensors ===
  using VULKAN_HPP_NAMESPACE::UniqueTensorARM;
  using VULKAN_HPP_NAMESPACE::UniqueTensorViewARM;

  //=== VK_NV_optical_flow ===
  using VULKAN_HPP_NAMESPACE::UniqueOpticalFlowSessionNV;

  //=== VK_EXT_shader_object ===
  using VULKAN_HPP_NAMESPACE::UniqueShaderEXT;

  //=== VK_KHR_pipeline_binary ===
  using VULKAN_HPP_NAMESPACE::UniquePipelineBinaryKHR;

  //=== VK_ARM_data_graph ===
  using VULKAN_HPP_NAMESPACE::UniqueDataGraphPipelineSessionARM;

  //=== VK_NV_external_compute_queue ===
  using VULKAN_HPP_NAMESPACE::UniqueExternalComputeQueueNV;

  //=== VK_EXT_device_generated_commands ===
  using VULKAN_HPP_NAMESPACE::UniqueHandleTraits;
  using VULKAN_HPP_NAMESPACE::UniqueIndirectCommandsLayoutEXT;
  using VULKAN_HPP_NAMESPACE::UniqueIndirectExecutionSetEXT;
#endif /*VULKAN_HPP_NO_SMART_HANDLE*/

  //======================
  //=== SHARED HANDLEs ===
  //======================

#if !defined( VULKAN_HPP_NO_SMART_HANDLE )

  //=== VK_VERSION_1_0 ===
  using VULKAN_HPP_NAMESPACE::SharedBuffer;
  using VULKAN_HPP_NAMESPACE::SharedBufferView;
  using VULKAN_HPP_NAMESPACE::SharedCommandBuffer;
  using VULKAN_HPP_NAMESPACE::SharedCommandPool;
  using VULKAN_HPP_NAMESPACE::SharedDescriptorPool;
  using VULKAN_HPP_NAMESPACE::SharedDescriptorSet;
  using VULKAN_HPP_NAMESPACE::SharedDescriptorSetLayout;
  using VULKAN_HPP_NAMESPACE::SharedDevice;
  using VULKAN_HPP_NAMESPACE::SharedDeviceMemory;
  using VULKAN_HPP_NAMESPACE::SharedEvent;
  using VULKAN_HPP_NAMESPACE::SharedFence;
  using VULKAN_HPP_NAMESPACE::SharedFramebuffer;
  using VULKAN_HPP_NAMESPACE::SharedImage;
  using VULKAN_HPP_NAMESPACE::SharedImageView;
  using VULKAN_HPP_NAMESPACE::SharedInstance;
  using VULKAN_HPP_NAMESPACE::SharedPhysicalDevice;
  using VULKAN_HPP_NAMESPACE::SharedPipeline;
  using VULKAN_HPP_NAMESPACE::SharedPipelineCache;
  using VULKAN_HPP_NAMESPACE::SharedPipelineLayout;
  using VULKAN_HPP_NAMESPACE::SharedQueryPool;
  using VULKAN_HPP_NAMESPACE::SharedQueue;
  using VULKAN_HPP_NAMESPACE::SharedRenderPass;
  using VULKAN_HPP_NAMESPACE::SharedSampler;
  using VULKAN_HPP_NAMESPACE::SharedSemaphore;
  using VULKAN_HPP_NAMESPACE::SharedShaderModule;

  //=== VK_VERSION_1_1 ===
  using VULKAN_HPP_NAMESPACE::SharedDescriptorUpdateTemplate;
  using VULKAN_HPP_NAMESPACE::SharedSamplerYcbcrConversion;

  //=== VK_VERSION_1_3 ===
  using VULKAN_HPP_NAMESPACE::SharedPrivateDataSlot;

  //=== VK_KHR_surface ===
  using VULKAN_HPP_NAMESPACE::SharedSurfaceKHR;

  //=== VK_KHR_swapchain ===
  using VULKAN_HPP_NAMESPACE::SharedSwapchainKHR;

  //=== VK_KHR_display ===
  using VULKAN_HPP_NAMESPACE::SharedDisplayKHR;
  using VULKAN_HPP_NAMESPACE::SharedDisplayModeKHR;

  //=== VK_EXT_debug_report ===
  using VULKAN_HPP_NAMESPACE::SharedDebugReportCallbackEXT;

  //=== VK_KHR_video_queue ===
  using VULKAN_HPP_NAMESPACE::SharedVideoSessionKHR;
  using VULKAN_HPP_NAMESPACE::SharedVideoSessionParametersKHR;

  //=== VK_NVX_binary_import ===
  using VULKAN_HPP_NAMESPACE::SharedCuFunctionNVX;
  using VULKAN_HPP_NAMESPACE::SharedCuModuleNVX;

  //=== VK_EXT_debug_utils ===
  using VULKAN_HPP_NAMESPACE::SharedDebugUtilsMessengerEXT;

  //=== VK_KHR_acceleration_structure ===
  using VULKAN_HPP_NAMESPACE::SharedAccelerationStructureKHR;

  //=== VK_EXT_validation_cache ===
  using VULKAN_HPP_NAMESPACE::SharedValidationCacheEXT;

  //=== VK_NV_ray_tracing ===
  using VULKAN_HPP_NAMESPACE::SharedAccelerationStructureNV;

  //=== VK_INTEL_performance_query ===
  using VULKAN_HPP_NAMESPACE::SharedPerformanceConfigurationINTEL;

  //=== VK_KHR_deferred_host_operations ===
  using VULKAN_HPP_NAMESPACE::SharedDeferredOperationKHR;

  //=== VK_NV_device_generated_commands ===
  using VULKAN_HPP_NAMESPACE::SharedIndirectCommandsLayoutNV;

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_NV_cuda_kernel_launch ===
  using VULKAN_HPP_NAMESPACE::SharedCudaFunctionNV;
  using VULKAN_HPP_NAMESPACE::SharedCudaModuleNV;
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_buffer_collection ===
  using VULKAN_HPP_NAMESPACE::SharedBufferCollectionFUCHSIA;
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

  //=== VK_EXT_opacity_micromap ===
  using VULKAN_HPP_NAMESPACE::SharedMicromapEXT;

  //=== VK_ARM_tensors ===
  using VULKAN_HPP_NAMESPACE::SharedTensorARM;
  using VULKAN_HPP_NAMESPACE::SharedTensorViewARM;

  //=== VK_NV_optical_flow ===
  using VULKAN_HPP_NAMESPACE::SharedOpticalFlowSessionNV;

  //=== VK_EXT_shader_object ===
  using VULKAN_HPP_NAMESPACE::SharedShaderEXT;

  //=== VK_KHR_pipeline_binary ===
  using VULKAN_HPP_NAMESPACE::SharedPipelineBinaryKHR;

  //=== VK_ARM_data_graph ===
  using VULKAN_HPP_NAMESPACE::SharedDataGraphPipelineSessionARM;

  //=== VK_NV_external_compute_queue ===
  using VULKAN_HPP_NAMESPACE::SharedExternalComputeQueueNV;

  //=== VK_EXT_device_generated_commands ===
  using VULKAN_HPP_NAMESPACE::SharedHandleTraits;
  using VULKAN_HPP_NAMESPACE::SharedIndirectCommandsLayoutEXT;
  using VULKAN_HPP_NAMESPACE::SharedIndirectExecutionSetEXT;

  //=== VK_KHR_swapchain enum ===
  using VULKAN_HPP_NAMESPACE::SwapchainOwns;
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

#if VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL
  namespace detail
  {
    using VULKAN_HPP_NAMESPACE::detail::DynamicLoader;
  }  // namespace detail
#endif /*VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL*/

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
  using VULKAN_HPP_NAMESPACE::getDepthFormats;
  using VULKAN_HPP_NAMESPACE::getDepthStencilFormats;
  using VULKAN_HPP_NAMESPACE::getStencilFormats;
  using VULKAN_HPP_NAMESPACE::hasDepthComponent;
  using VULKAN_HPP_NAMESPACE::hasStencilComponent;
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

#if !defined( VULKAN_HPP_DISABLE_ENHANCED_MODE )
  namespace VULKAN_HPP_RAII_NAMESPACE
  {
    //======================
    //=== RAII HARDCODED ===
    //======================

    using VULKAN_HPP_RAII_NAMESPACE::Context;
    using VULKAN_HPP_RAII_NAMESPACE::isVulkanRAIIHandleType;

    namespace detail
    {
      using VULKAN_HPP_RAII_NAMESPACE::detail::ContextDispatcher;
      using VULKAN_HPP_RAII_NAMESPACE::detail::DeviceDispatcher;
      using VULKAN_HPP_RAII_NAMESPACE::detail::InstanceDispatcher;
    }  // namespace detail

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

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
    //=== VK_NV_cuda_kernel_launch ===
    using VULKAN_HPP_RAII_NAMESPACE::CudaFunctionNV;
    using VULKAN_HPP_RAII_NAMESPACE::CudaModuleNV;
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
    //=== VK_FUCHSIA_buffer_collection ===
    using VULKAN_HPP_RAII_NAMESPACE::BufferCollectionFUCHSIA;
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

    //=== VK_EXT_opacity_micromap ===
    using VULKAN_HPP_RAII_NAMESPACE::MicromapEXT;

    //=== VK_ARM_tensors ===
    using VULKAN_HPP_RAII_NAMESPACE::TensorARM;
    using VULKAN_HPP_RAII_NAMESPACE::TensorViewARM;

    //=== VK_NV_optical_flow ===
    using VULKAN_HPP_RAII_NAMESPACE::OpticalFlowSessionNV;

    //=== VK_EXT_shader_object ===
    using VULKAN_HPP_RAII_NAMESPACE::ShaderEXT;
    using VULKAN_HPP_RAII_NAMESPACE::ShaderEXTs;

    //=== VK_KHR_pipeline_binary ===
    using VULKAN_HPP_RAII_NAMESPACE::PipelineBinaryKHR;
    using VULKAN_HPP_RAII_NAMESPACE::PipelineBinaryKHRs;

    //=== VK_ARM_data_graph ===
    using VULKAN_HPP_RAII_NAMESPACE::DataGraphPipelineSessionARM;

    //=== VK_NV_external_compute_queue ===
    using VULKAN_HPP_RAII_NAMESPACE::ExternalComputeQueueNV;

    //=== VK_EXT_device_generated_commands ===
    using VULKAN_HPP_RAII_NAMESPACE::IndirectCommandsLayoutEXT;
    using VULKAN_HPP_RAII_NAMESPACE::IndirectExecutionSetEXT;

  }  // namespace VULKAN_HPP_RAII_NAMESPACE
#endif
}  // namespace VULKAN_HPP_NAMESPACE

export namespace std
{

  //=======================================
  //=== HASH specialization for Flags types ===
  //=======================================

  template <typename BitType>
  struct hash<VULKAN_HPP_NAMESPACE::Flags<BitType>>;

  //========================================
  //=== HASH specializations for handles ===
  //========================================

  //=== VK_VERSION_1_0 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Instance>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevice>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Device>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Queue>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceMemory>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Fence>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Semaphore>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueryPool>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Buffer>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Image>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageView>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CommandPool>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CommandBuffer>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Event>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferView>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ShaderModule>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineCache>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Pipeline>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineLayout>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Sampler>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorPool>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorSet>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorSetLayout>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Framebuffer>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPass>;

  //=== VK_VERSION_1_1 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplate>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SamplerYcbcrConversion>;

  //=== VK_VERSION_1_3 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PrivateDataSlot>;

  //=== VK_KHR_surface ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceKHR>;

  //=== VK_KHR_swapchain ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SwapchainKHR>;

  //=== VK_KHR_display ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayModeKHR>;

  //=== VK_EXT_debug_report ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DebugReportCallbackEXT>;

  //=== VK_KHR_video_queue ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoSessionKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoSessionParametersKHR>;

  //=== VK_NVX_binary_import ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CuModuleNVX>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CuFunctionNVX>;

  //=== VK_EXT_debug_utils ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DebugUtilsMessengerEXT>;

  //=== VK_KHR_acceleration_structure ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureKHR>;

  //=== VK_EXT_validation_cache ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ValidationCacheEXT>;

  //=== VK_NV_ray_tracing ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureNV>;

  //=== VK_INTEL_performance_query ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PerformanceConfigurationINTEL>;

  //=== VK_KHR_deferred_host_operations ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeferredOperationKHR>;

  //=== VK_NV_device_generated_commands ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutNV>;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_NV_cuda_kernel_launch ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CudaModuleNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CudaFunctionNV>;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_buffer_collection ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferCollectionFUCHSIA>;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

  //=== VK_EXT_opacity_micromap ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MicromapEXT>;

  //=== VK_ARM_tensors ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TensorARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TensorViewARM>;

  //=== VK_NV_optical_flow ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::OpticalFlowSessionNV>;

  //=== VK_EXT_shader_object ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ShaderEXT>;

  //=== VK_KHR_pipeline_binary ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineBinaryKHR>;

  //=== VK_ARM_data_graph ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DataGraphPipelineSessionARM>;

  //=== VK_NV_external_compute_queue ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalComputeQueueNV>;

  //=== VK_EXT_device_generated_commands ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IndirectExecutionSetEXT>;

  //========================================
  //=== HASH specializations for structs ===
  //========================================

  //=== VK_VERSION_1_0 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Extent2D>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Extent3D>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Offset2D>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Offset3D>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Rect2D>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BaseInStructure>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BaseOutStructure>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferMemoryBarrier>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageMemoryBarrier>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryBarrier>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AllocationCallbacks>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ApplicationInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FormatProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageFormatProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::InstanceCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryHeap>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryType>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceLimits>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSparseProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueueFamilyProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceQueueCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExtensionProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::LayerProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubmitInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MappedMemoryRange>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryAllocateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryRequirements>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindSparseInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageSubresource>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SparseBufferMemoryBindInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SparseImageFormatProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SparseImageMemoryBind>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SparseImageMemoryBindInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SparseImageMemoryRequirements>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SparseImageOpaqueMemoryBindInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SparseMemoryBind>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FenceCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SemaphoreCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueryPoolCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubresourceLayout>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ComponentMapping>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageSubresourceRange>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageViewCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CommandPoolCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CommandBufferAllocateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CommandBufferBeginInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CommandBufferInheritanceInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferCopy>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferImageCopy>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageCopy>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageSubresourceLayers>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DispatchIndirectCommand>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineCacheHeaderVersionOne>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::EventCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferViewCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ShaderModuleCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineCacheCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ComputePipelineCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineShaderStageCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SpecializationInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SpecializationMapEntry>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineLayoutCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PushConstantRange>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SamplerCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyDescriptorSet>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorBufferInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorImageInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorPoolCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorPoolSize>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorSetAllocateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorSetLayoutBinding>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorSetLayoutCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::WriteDescriptorSet>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ClearColorValue>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DrawIndexedIndirectCommand>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DrawIndirectCommand>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GraphicsPipelineCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineColorBlendAttachmentState>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineColorBlendStateCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineDepthStencilStateCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineDynamicStateCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineInputAssemblyStateCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineMultisampleStateCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineRasterizationStateCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineTessellationStateCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineVertexInputStateCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineViewportStateCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::StencilOpState>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VertexInputAttributeDescription>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VertexInputBindingDescription>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Viewport>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AttachmentDescription>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AttachmentReference>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FramebufferCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubpassDependency>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubpassDescription>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ClearAttachment>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ClearDepthStencilValue>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ClearRect>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ClearValue>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageBlit>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageResolve>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassBeginInfo>;

  //=== VK_VERSION_1_1 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindBufferMemoryInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindImageMemoryInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryDedicatedRequirements>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryDedicatedAllocateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryAllocateFlagsInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceGroupCommandBufferBeginInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceGroupSubmitInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceGroupBindSparseInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindBufferMemoryDeviceGroupInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindImageMemoryDeviceGroupInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceGroupProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceGroupDeviceCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferMemoryRequirementsInfo2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageMemoryRequirementsInfo2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageSparseMemoryRequirementsInfo2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryRequirements2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SparseImageMemoryRequirements2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FormatProperties2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageFormatProperties2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageFormatInfo2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueueFamilyProperties2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SparseImageFormatProperties2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSparseImageFormatInfo2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageViewUsageCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceProtectedMemoryFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceProtectedMemoryProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceQueueInfo2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ProtectedSubmitInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindImagePlaneMemoryInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImagePlaneMemoryRequirementsInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalMemoryProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalImageFormatInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalImageFormatProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalBufferInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalBufferProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceIDProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalMemoryImageCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalMemoryBufferCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportMemoryAllocateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalFenceInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalFenceProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportFenceCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportSemaphoreCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalSemaphoreInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalSemaphoreProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSubgroupProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevice16BitStorageFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVariablePointersFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateEntry>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance3Properties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorSetLayoutSupport>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSamplerYcbcrConversionFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionImageFormatProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceGroupRenderPassBeginInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePointClippingProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassInputAttachmentAspectCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::InputAttachmentAspectReference>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineTessellationDomainOriginStateCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassMultiviewCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderDrawParametersFeatures>;

  //=== VK_VERSION_1_2 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan11Features>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan11Properties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan12Features>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan12Properties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageFormatListCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ConformanceVersion>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDriverProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkanMemoryModelFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceHostQueryResetFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceTimelineSemaphoreFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceTimelineSemaphoreProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SemaphoreTypeCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TimelineSemaphoreSubmitInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SemaphoreWaitInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SemaphoreSignalInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceBufferDeviceAddressFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferDeviceAddressInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferOpaqueCaptureAddressCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryOpaqueCaptureAddressAllocateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceMemoryOpaqueCaptureAddressInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevice8BitStorageFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicInt64Features>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderFloat16Int8Features>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFloatControlsProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorSetLayoutBindingFlagsCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorIndexingFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorIndexingProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorSetVariableDescriptorCountAllocateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorSetVariableDescriptorCountLayoutSupport>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceScalarBlockLayoutFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SamplerReductionModeCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSamplerFilterMinmaxProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceUniformBufferStandardLayoutFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSubgroupExtendedTypesFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassCreateInfo2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AttachmentDescription2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AttachmentReference2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubpassDescription2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubpassDependency2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubpassBeginInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubpassEndInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubpassDescriptionDepthStencilResolve>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthStencilResolveProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageStencilUsageCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImagelessFramebufferFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FramebufferAttachmentsCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FramebufferAttachmentImageInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassAttachmentBeginInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSeparateDepthStencilLayoutsFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AttachmentReferenceStencilLayout>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AttachmentDescriptionStencilLayout>;

  //=== VK_VERSION_1_3 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan13Features>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan13Properties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceToolProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePrivateDataFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DevicePrivateDataCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PrivateDataSlotCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryBarrier2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferMemoryBarrier2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageMemoryBarrier2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DependencyInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubmitInfo2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SemaphoreSubmitInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CommandBufferSubmitInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSynchronization2Features>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyBufferInfo2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyImageInfo2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyBufferToImageInfo2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyImageToBufferInfo2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferCopy2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageCopy2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferImageCopy2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceTextureCompressionASTCHDRFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FormatProperties3>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance4Features>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance4Properties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceBufferMemoryRequirements>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceImageMemoryRequirements>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineCreationFeedbackCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineCreationFeedback>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderTerminateInvocationFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderDemoteToHelperInvocationFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineCreationCacheControlFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageRobustnessFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSubgroupSizeControlFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSubgroupSizeControlProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineShaderStageRequiredSubgroupSizeCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceInlineUniformBlockFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceInlineUniformBlockProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::WriteDescriptorSetInlineUniformBlock>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorPoolInlineUniformBlockCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderIntegerDotProductFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderIntegerDotProductProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceTexelBufferAlignmentProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BlitImageInfo2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageBlit2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ResolveImageInfo2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageResolve2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderingInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderingAttachmentInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineRenderingCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDynamicRenderingFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CommandBufferInheritanceRenderingInfo>;

  //=== VK_VERSION_1_4 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan14Features>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan14Properties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceQueueGlobalPriorityCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceGlobalPriorityQueryFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueueFamilyGlobalPriorityProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceIndexTypeUint8Features>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryMapInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryUnmapInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance5Features>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance5Properties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceImageSubresourceInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageSubresource2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubresourceLayout2>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferUsageFlags2CreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance6Features>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance6Properties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindMemoryStatus>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceHostImageCopyFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceHostImageCopyProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryToImageCopy>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageToMemoryCopy>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyMemoryToImageInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyImageToMemoryInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyImageToImageInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::HostImageLayoutTransitionInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubresourceHostMemcpySize>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::HostImageCopyDevicePerformanceQuery>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSubgroupRotateFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderFloatControls2Features>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderExpectAssumeFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineCreateFlags2CreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePushDescriptorProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindDescriptorSetsInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PushConstantsInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PushDescriptorSetInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PushDescriptorSetWithTemplateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineProtectedAccessFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineRobustnessFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineRobustnessProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineRobustnessCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceLineRasterizationFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceLineRasterizationProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineRasterizationLineStateCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeDivisorProperties>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VertexInputBindingDivisorDescription>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineVertexInputDivisorStateCreateInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeDivisorFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderingAreaInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDynamicRenderingLocalReadFeatures>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderingAttachmentLocationInfo>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderingInputAttachmentIndexInfo>;

  //=== VK_KHR_surface ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceFormatKHR>;

  //=== VK_KHR_swapchain ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SwapchainCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PresentInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageSwapchainCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindImageMemorySwapchainInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AcquireNextImageInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceGroupPresentCapabilitiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceGroupPresentInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceGroupSwapchainCreateInfoKHR>;

  //=== VK_KHR_display ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayModeCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayModeParametersKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayModePropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayPlaneCapabilitiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayPlanePropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayPropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplaySurfaceCreateInfoKHR>;

  //=== VK_KHR_display_swapchain ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayPresentInfoKHR>;

#if defined( VK_USE_PLATFORM_XLIB_KHR )
  //=== VK_KHR_xlib_surface ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::XlibSurfaceCreateInfoKHR>;
#endif /*VK_USE_PLATFORM_XLIB_KHR*/

#if defined( VK_USE_PLATFORM_XCB_KHR )
  //=== VK_KHR_xcb_surface ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::XcbSurfaceCreateInfoKHR>;
#endif /*VK_USE_PLATFORM_XCB_KHR*/

#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
  //=== VK_KHR_wayland_surface ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::WaylandSurfaceCreateInfoKHR>;
#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  //=== VK_KHR_android_surface ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AndroidSurfaceCreateInfoKHR>;
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_win32_surface ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Win32SurfaceCreateInfoKHR>;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_EXT_debug_report ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DebugReportCallbackCreateInfoEXT>;

  //=== VK_AMD_rasterization_order ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineRasterizationStateRasterizationOrderAMD>;

  //=== VK_EXT_debug_marker ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DebugMarkerObjectNameInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DebugMarkerObjectTagInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DebugMarkerMarkerInfoEXT>;

  //=== VK_KHR_video_queue ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueueFamilyQueryResultStatusPropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueueFamilyVideoPropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoProfileInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoProfileListInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoCapabilitiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoFormatInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoFormatPropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoPictureResourceInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoReferenceSlotInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoSessionMemoryRequirementsKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindVideoSessionMemoryInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoSessionCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoSessionParametersCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoSessionParametersUpdateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoBeginCodingInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEndCodingInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoCodingControlInfoKHR>;

  //=== VK_KHR_video_decode_queue ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeCapabilitiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeUsageInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeInfoKHR>;

  //=== VK_NV_dedicated_allocation ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DedicatedAllocationImageCreateInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DedicatedAllocationBufferCreateInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DedicatedAllocationMemoryAllocateInfoNV>;

  //=== VK_EXT_transform_feedback ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceTransformFeedbackFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceTransformFeedbackPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineRasterizationStateStreamCreateInfoEXT>;

  //=== VK_NVX_binary_import ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CuModuleCreateInfoNVX>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CuModuleTexturingModeCreateInfoNVX>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CuFunctionCreateInfoNVX>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CuLaunchInfoNVX>;

  //=== VK_NVX_image_view_handle ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageViewHandleInfoNVX>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageViewAddressPropertiesNVX>;

  //=== VK_KHR_video_encode_h264 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264CapabilitiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264QualityLevelPropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersAddInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersGetInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersFeedbackInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264PictureInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264DpbSlotInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264NaluSliceInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264ProfileInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264RateControlInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264RateControlLayerInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264QpKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264FrameSizeKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264GopRemainingFrameInfoKHR>;

  //=== VK_KHR_video_encode_h265 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265CapabilitiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265QualityLevelPropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersAddInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersGetInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersFeedbackInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265PictureInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265DpbSlotInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265NaluSliceSegmentInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265ProfileInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265RateControlInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265RateControlLayerInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265QpKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265FrameSizeKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265GopRemainingFrameInfoKHR>;

  //=== VK_KHR_video_decode_h264 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH264ProfileInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH264CapabilitiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH264SessionParametersCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH264SessionParametersAddInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH264PictureInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH264DpbSlotInfoKHR>;

  //=== VK_AMD_texture_gather_bias_lod ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TextureLODGatherFormatPropertiesAMD>;

  //=== VK_AMD_shader_info ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ShaderResourceUsageAMD>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ShaderStatisticsInfoAMD>;

#if defined( VK_USE_PLATFORM_GGP )
  //=== VK_GGP_stream_descriptor_surface ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::StreamDescriptorSurfaceCreateInfoGGP>;
#endif /*VK_USE_PLATFORM_GGP*/

  //=== VK_NV_corner_sampled_image ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCornerSampledImageFeaturesNV>;

  //=== VK_NV_external_memory_capabilities ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalImageFormatPropertiesNV>;

  //=== VK_NV_external_memory ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalMemoryImageCreateInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportMemoryAllocateInfoNV>;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_NV_external_memory_win32 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportMemoryWin32HandleInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportMemoryWin32HandleInfoNV>;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_NV_win32_keyed_mutex ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Win32KeyedMutexAcquireReleaseInfoNV>;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_KHR_device_group ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceGroupPresentCapabilitiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageSwapchainCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindImageMemorySwapchainInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AcquireNextImageInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceGroupPresentInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceGroupSwapchainCreateInfoKHR>;

  //=== VK_EXT_validation_flags ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ValidationFlagsEXT>;

#if defined( VK_USE_PLATFORM_VI_NN )
  //=== VK_NN_vi_surface ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ViSurfaceCreateInfoNN>;
#endif /*VK_USE_PLATFORM_VI_NN*/

  //=== VK_EXT_astc_decode_mode ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageViewASTCDecodeModeEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceASTCDecodeFeaturesEXT>;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_external_memory_win32 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportMemoryWin32HandleInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportMemoryWin32HandleInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryWin32HandlePropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryGetWin32HandleInfoKHR>;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_KHR_external_memory_fd ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportMemoryFdInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryFdPropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryGetFdInfoKHR>;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_win32_keyed_mutex ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Win32KeyedMutexAcquireReleaseInfoKHR>;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_external_semaphore_win32 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportSemaphoreWin32HandleInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportSemaphoreWin32HandleInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::D3D12FenceSubmitInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SemaphoreGetWin32HandleInfoKHR>;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_KHR_external_semaphore_fd ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportSemaphoreFdInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SemaphoreGetFdInfoKHR>;

  //=== VK_EXT_conditional_rendering ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ConditionalRenderingBeginInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceConditionalRenderingFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CommandBufferInheritanceConditionalRenderingInfoEXT>;

  //=== VK_KHR_incremental_present ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PresentRegionsKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PresentRegionKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RectLayerKHR>;

  //=== VK_NV_clip_space_w_scaling ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ViewportWScalingNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineViewportWScalingStateCreateInfoNV>;

  //=== VK_EXT_display_surface_counter ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceCapabilities2EXT>;

  //=== VK_EXT_display_control ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayPowerInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceEventInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayEventInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SwapchainCounterCreateInfoEXT>;

  //=== VK_GOOGLE_display_timing ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RefreshCycleDurationGOOGLE>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PastPresentationTimingGOOGLE>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PresentTimesInfoGOOGLE>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PresentTimeGOOGLE>;

  //=== VK_NVX_multiview_per_view_attributes ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewPerViewAttributesPropertiesNVX>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MultiviewPerViewAttributesInfoNVX>;

  //=== VK_NV_viewport_swizzle ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ViewportSwizzleNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineViewportSwizzleStateCreateInfoNV>;

  //=== VK_EXT_discard_rectangles ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDiscardRectanglePropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineDiscardRectangleStateCreateInfoEXT>;

  //=== VK_EXT_conservative_rasterization ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceConservativeRasterizationPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineRasterizationConservativeStateCreateInfoEXT>;

  //=== VK_EXT_depth_clip_enable ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClipEnableFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineRasterizationDepthClipStateCreateInfoEXT>;

  //=== VK_EXT_hdr_metadata ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::HdrMetadataEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::XYColorEXT>;

  //=== VK_IMG_relaxed_line_rasterization ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRelaxedLineRasterizationFeaturesIMG>;

  //=== VK_KHR_shared_presentable_image ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SharedPresentSurfaceCapabilitiesKHR>;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_external_fence_win32 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportFenceWin32HandleInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportFenceWin32HandleInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FenceGetWin32HandleInfoKHR>;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_KHR_external_fence_fd ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportFenceFdInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FenceGetFdInfoKHR>;

  //=== VK_KHR_performance_query ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePerformanceQueryFeaturesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePerformanceQueryPropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PerformanceCounterKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PerformanceCounterDescriptionKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueryPoolPerformanceCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PerformanceCounterResultKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AcquireProfilingLockInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PerformanceQuerySubmitInfoKHR>;

  //=== VK_KHR_get_surface_capabilities2 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSurfaceInfo2KHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceCapabilities2KHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceFormat2KHR>;

  //=== VK_KHR_get_display_properties2 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayProperties2KHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayPlaneProperties2KHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayModeProperties2KHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayPlaneInfo2KHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayPlaneCapabilities2KHR>;

#if defined( VK_USE_PLATFORM_IOS_MVK )
  //=== VK_MVK_ios_surface ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IOSSurfaceCreateInfoMVK>;
#endif /*VK_USE_PLATFORM_IOS_MVK*/

#if defined( VK_USE_PLATFORM_MACOS_MVK )
  //=== VK_MVK_macos_surface ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MacOSSurfaceCreateInfoMVK>;
#endif /*VK_USE_PLATFORM_MACOS_MVK*/

  //=== VK_EXT_debug_utils ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DebugUtilsLabelEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DebugUtilsMessengerCallbackDataEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DebugUtilsMessengerCreateInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DebugUtilsObjectNameInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DebugUtilsObjectTagInfoEXT>;

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  //=== VK_ANDROID_external_memory_android_hardware_buffer ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AndroidHardwareBufferUsageANDROID>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AndroidHardwareBufferPropertiesANDROID>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AndroidHardwareBufferFormatPropertiesANDROID>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportAndroidHardwareBufferInfoANDROID>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryGetAndroidHardwareBufferInfoANDROID>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalFormatANDROID>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AndroidHardwareBufferFormatProperties2ANDROID>;
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_AMDX_shader_enqueue ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderEnqueueFeaturesAMDX>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderEnqueuePropertiesAMDX>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExecutionGraphPipelineScratchSizeAMDX>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExecutionGraphPipelineCreateInfoAMDX>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DispatchGraphInfoAMDX>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DispatchGraphCountInfoAMDX>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineShaderStageNodeCreateInfoAMDX>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceOrHostAddressConstAMDX>;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_AMD_mixed_attachment_samples ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AttachmentSampleCountInfoAMD>;

  //=== VK_KHR_shader_bfloat16 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderBfloat16FeaturesKHR>;

  //=== VK_EXT_sample_locations ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SampleLocationEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SampleLocationsInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AttachmentSampleLocationsEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubpassSampleLocationsEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassSampleLocationsBeginInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineSampleLocationsStateCreateInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSampleLocationsPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MultisamplePropertiesEXT>;

  //=== VK_EXT_blend_operation_advanced ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceBlendOperationAdvancedFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceBlendOperationAdvancedPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineColorBlendAdvancedStateCreateInfoEXT>;

  //=== VK_NV_fragment_coverage_to_color ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineCoverageToColorStateCreateInfoNV>;

  //=== VK_KHR_acceleration_structure ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceOrHostAddressKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceOrHostAddressConstKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureBuildRangeInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AabbPositionsKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryTrianglesDataKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TransformMatrixKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureBuildGeometryInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryAabbsDataKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureInstanceKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryInstancesDataKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryDataKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::WriteDescriptorSetAccelerationStructureKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceAccelerationStructureFeaturesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceAccelerationStructurePropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureDeviceAddressInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureVersionInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyAccelerationStructureToMemoryInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyMemoryToAccelerationStructureInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyAccelerationStructureInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureBuildSizesInfoKHR>;

  //=== VK_KHR_ray_tracing_pipeline ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RayTracingShaderGroupCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RayTracingPipelineCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPipelineFeaturesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPipelinePropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::StridedDeviceAddressRegionKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TraceRaysIndirectCommandKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RayTracingPipelineInterfaceCreateInfoKHR>;

  //=== VK_KHR_ray_query ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayQueryFeaturesKHR>;

  //=== VK_NV_framebuffer_mixed_samples ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineCoverageModulationStateCreateInfoNV>;

  //=== VK_NV_shader_sm_builtins ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSMBuiltinsPropertiesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSMBuiltinsFeaturesNV>;

  //=== VK_EXT_image_drm_format_modifier ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DrmFormatModifierPropertiesListEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DrmFormatModifierPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageDrmFormatModifierInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageDrmFormatModifierListCreateInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageDrmFormatModifierExplicitCreateInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageDrmFormatModifierPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DrmFormatModifierPropertiesList2EXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DrmFormatModifierProperties2EXT>;

  //=== VK_EXT_validation_cache ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ValidationCacheCreateInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ShaderModuleValidationCacheCreateInfoEXT>;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_KHR_portability_subset ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePortabilitySubsetFeaturesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePortabilitySubsetPropertiesKHR>;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_NV_shading_rate_image ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ShadingRatePaletteNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineViewportShadingRateImageStateCreateInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShadingRateImageFeaturesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShadingRateImagePropertiesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CoarseSampleLocationNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CoarseSampleOrderCustomNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineViewportCoarseSampleOrderStateCreateInfoNV>;

  //=== VK_NV_ray_tracing ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RayTracingShaderGroupCreateInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RayTracingPipelineCreateInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GeometryTrianglesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GeometryAABBNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GeometryDataNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GeometryNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureCreateInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindAccelerationStructureMemoryInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::WriteDescriptorSetAccelerationStructureNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureMemoryRequirementsInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPropertiesNV>;

  //=== VK_NV_representative_fragment_test ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRepresentativeFragmentTestFeaturesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineRepresentativeFragmentTestStateCreateInfoNV>;

  //=== VK_EXT_filter_cubic ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageViewImageFormatInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FilterCubicImageViewImageFormatPropertiesEXT>;

  //=== VK_EXT_external_memory_host ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportMemoryHostPointerInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryHostPointerPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalMemoryHostPropertiesEXT>;

  //=== VK_KHR_shader_clock ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderClockFeaturesKHR>;

  //=== VK_AMD_pipeline_compiler_control ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineCompilerControlCreateInfoAMD>;

  //=== VK_AMD_shader_core_properties ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCorePropertiesAMD>;

  //=== VK_KHR_video_decode_h265 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH265ProfileInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH265CapabilitiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH265SessionParametersCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH265SessionParametersAddInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH265PictureInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH265DpbSlotInfoKHR>;

  //=== VK_AMD_memory_overallocation_behavior ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceMemoryOverallocationCreateInfoAMD>;

  //=== VK_EXT_vertex_attribute_divisor ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeDivisorPropertiesEXT>;

#if defined( VK_USE_PLATFORM_GGP )
  //=== VK_GGP_frame_token ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PresentFrameTokenGGP>;
#endif /*VK_USE_PLATFORM_GGP*/

  //=== VK_NV_mesh_shader ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderFeaturesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderPropertiesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DrawMeshTasksIndirectCommandNV>;

  //=== VK_NV_shader_image_footprint ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderImageFootprintFeaturesNV>;

  //=== VK_NV_scissor_exclusive ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineViewportExclusiveScissorStateCreateInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExclusiveScissorFeaturesNV>;

  //=== VK_NV_device_diagnostic_checkpoints ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueueFamilyCheckpointPropertiesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CheckpointDataNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueueFamilyCheckpointProperties2NV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CheckpointData2NV>;

  //=== VK_EXT_present_timing ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePresentTimingFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PresentTimingSurfaceCapabilitiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SwapchainCalibratedTimestampInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SwapchainTimingPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SwapchainTimeDomainPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PastPresentationTimingInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PastPresentationTimingPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PastPresentationTimingEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PresentTimingsInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PresentTimingInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PresentStageTimeEXT>;

  //=== VK_INTEL_shader_integer_functions2 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderIntegerFunctions2FeaturesINTEL>;

  //=== VK_INTEL_performance_query ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PerformanceValueDataINTEL>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PerformanceValueINTEL>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::InitializePerformanceApiInfoINTEL>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueryPoolPerformanceQueryCreateInfoINTEL>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PerformanceMarkerInfoINTEL>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PerformanceStreamMarkerInfoINTEL>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PerformanceOverrideInfoINTEL>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PerformanceConfigurationAcquireInfoINTEL>;

  //=== VK_EXT_pci_bus_info ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePCIBusInfoPropertiesEXT>;

  //=== VK_AMD_display_native_hdr ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayNativeHdrSurfaceCapabilitiesAMD>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SwapchainDisplayNativeHdrCreateInfoAMD>;

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_imagepipe_surface ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImagePipeSurfaceCreateInfoFUCHSIA>;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_surface ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MetalSurfaceCreateInfoEXT>;
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_EXT_fragment_density_map ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassFragmentDensityMapCreateInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderingFragmentDensityMapAttachmentInfoEXT>;

  //=== VK_KHR_fragment_shading_rate ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FragmentShadingRateAttachmentInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineFragmentShadingRateStateCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateFeaturesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRatePropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderingFragmentShadingRateAttachmentInfoKHR>;

  //=== VK_AMD_shader_core_properties2 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCoreProperties2AMD>;

  //=== VK_AMD_device_coherent_memory ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCoherentMemoryFeaturesAMD>;

  //=== VK_EXT_shader_image_atomic_int64 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT>;

  //=== VK_KHR_shader_quad_control ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderQuadControlFeaturesKHR>;

  //=== VK_EXT_memory_budget ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryBudgetPropertiesEXT>;

  //=== VK_EXT_memory_priority ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryPriorityFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryPriorityAllocateInfoEXT>;

  //=== VK_KHR_surface_protected_capabilities ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceProtectedCapabilitiesKHR>;

  //=== VK_NV_dedicated_allocation_image_aliasing ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDedicatedAllocationImageAliasingFeaturesNV>;

  //=== VK_EXT_buffer_device_address ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceBufferDeviceAddressFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferDeviceAddressCreateInfoEXT>;

  //=== VK_EXT_validation_features ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ValidationFeaturesEXT>;

  //=== VK_KHR_present_wait ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePresentWaitFeaturesKHR>;

  //=== VK_NV_cooperative_matrix ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CooperativeMatrixPropertiesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixFeaturesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixPropertiesNV>;

  //=== VK_NV_coverage_reduction_mode ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCoverageReductionModeFeaturesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineCoverageReductionStateCreateInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FramebufferMixedSamplesCombinationNV>;

  //=== VK_EXT_fragment_shader_interlock ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShaderInterlockFeaturesEXT>;

  //=== VK_EXT_ycbcr_image_arrays ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceYcbcrImageArraysFeaturesEXT>;

  //=== VK_EXT_provoking_vertex ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceProvokingVertexFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceProvokingVertexPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineRasterizationProvokingVertexStateCreateInfoEXT>;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_EXT_full_screen_exclusive ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceFullScreenExclusiveInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesFullScreenExclusiveEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceFullScreenExclusiveWin32InfoEXT>;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_EXT_headless_surface ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::HeadlessSurfaceCreateInfoEXT>;

  //=== VK_EXT_shader_atomic_float ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicFloatFeaturesEXT>;

  //=== VK_EXT_extended_dynamic_state ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicStateFeaturesEXT>;

  //=== VK_KHR_pipeline_executable_properties ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineExecutablePropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineExecutableInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineExecutableStatisticValueKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineExecutableStatisticKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineExecutableInternalRepresentationKHR>;

  //=== VK_EXT_map_memory_placed ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMapMemoryPlacedFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMapMemoryPlacedPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryMapPlacedInfoEXT>;

  //=== VK_EXT_shader_atomic_float2 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicFloat2FeaturesEXT>;

  //=== VK_NV_device_generated_commands ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceGeneratedCommandsPropertiesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceGeneratedCommandsFeaturesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GraphicsShaderGroupCreateInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GraphicsPipelineShaderGroupsCreateInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindShaderGroupIndirectCommandNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindIndexBufferIndirectCommandNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindVertexBufferIndirectCommandNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SetStateFlagsIndirectCommandNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IndirectCommandsStreamNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutTokenNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutCreateInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GeneratedCommandsInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GeneratedCommandsMemoryRequirementsInfoNV>;

  //=== VK_NV_inherited_viewport_scissor ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceInheritedViewportScissorFeaturesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CommandBufferInheritanceViewportScissorInfoNV>;

  //=== VK_EXT_texel_buffer_alignment ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceTexelBufferAlignmentFeaturesEXT>;

  //=== VK_QCOM_render_pass_transform ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassTransformBeginInfoQCOM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CommandBufferInheritanceRenderPassTransformInfoQCOM>;

  //=== VK_EXT_depth_bias_control ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthBiasControlFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DepthBiasInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DepthBiasRepresentationInfoEXT>;

  //=== VK_EXT_device_memory_report ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceMemoryReportFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceDeviceMemoryReportCreateInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceMemoryReportCallbackDataEXT>;

  //=== VK_EXT_custom_border_color ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SamplerCustomBorderColorCreateInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCustomBorderColorPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCustomBorderColorFeaturesEXT>;

  //=== VK_KHR_pipeline_library ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineLibraryCreateInfoKHR>;

  //=== VK_NV_present_barrier ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePresentBarrierFeaturesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesPresentBarrierNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SwapchainPresentBarrierCreateInfoNV>;

  //=== VK_KHR_present_id ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PresentIdKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePresentIdFeaturesKHR>;

  //=== VK_KHR_video_encode_queue ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeCapabilitiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueryPoolVideoEncodeFeedbackCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeUsageInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeRateControlInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeRateControlLayerInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoEncodeQualityLevelInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeQualityLevelPropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeQualityLevelInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeSessionParametersGetInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeSessionParametersFeedbackInfoKHR>;

  //=== VK_NV_device_diagnostics_config ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDiagnosticsConfigFeaturesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceDiagnosticsConfigCreateInfoNV>;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_NV_cuda_kernel_launch ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CudaModuleCreateInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CudaFunctionCreateInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CudaLaunchInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCudaKernelLaunchFeaturesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCudaKernelLaunchPropertiesNV>;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_QCOM_tile_shading ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceTileShadingFeaturesQCOM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceTileShadingPropertiesQCOM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassTileShadingCreateInfoQCOM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PerTileBeginInfoQCOM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PerTileEndInfoQCOM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DispatchTileInfoQCOM>;

  //=== VK_NV_low_latency ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueryLowLatencySupportNV>;

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_objects ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportMetalObjectCreateInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportMetalObjectsInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportMetalDeviceInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportMetalCommandQueueInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportMetalBufferInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportMetalBufferInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportMetalTextureInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportMetalTextureInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportMetalIOSurfaceInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportMetalIOSurfaceInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportMetalSharedEventInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportMetalSharedEventInfoEXT>;
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_EXT_descriptor_buffer ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorBufferPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorBufferDensityMapPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorBufferFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorAddressInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorBufferBindingInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorBufferBindingPushDescriptorBufferHandleEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorDataEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorGetInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferCaptureDescriptorDataInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageCaptureDescriptorDataInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageViewCaptureDescriptorDataInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SamplerCaptureDescriptorDataInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::OpaqueCaptureDescriptorDataCreateInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureCaptureDescriptorDataInfoEXT>;

  //=== VK_EXT_graphics_pipeline_library ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GraphicsPipelineLibraryCreateInfoEXT>;

  //=== VK_AMD_shader_early_and_late_fragment_tests ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderEarlyAndLateFragmentTestsFeaturesAMD>;

  //=== VK_KHR_fragment_shader_barycentric ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShaderBarycentricPropertiesKHR>;

  //=== VK_KHR_shader_subgroup_uniform_control_flow ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR>;

  //=== VK_NV_fragment_shading_rate_enums ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateEnumsFeaturesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateEnumsPropertiesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineFragmentShadingRateEnumStateCreateInfoNV>;

  //=== VK_NV_ray_tracing_motion_blur ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryMotionTrianglesDataNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureMotionInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureMotionInstanceNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureMotionInstanceDataNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureMatrixMotionInstanceNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureSRTMotionInstanceNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SRTDataNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingMotionBlurFeaturesNV>;

  //=== VK_EXT_mesh_shader ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DrawMeshTasksIndirectCommandEXT>;

  //=== VK_EXT_ycbcr_2plane_444_formats ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT>;

  //=== VK_EXT_fragment_density_map2 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMap2FeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMap2PropertiesEXT>;

  //=== VK_QCOM_rotated_copy_commands ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyCommandTransformInfoQCOM>;

  //=== VK_KHR_workgroup_memory_explicit_layout ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR>;

  //=== VK_EXT_image_compression_control ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageCompressionControlFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageCompressionControlEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageCompressionPropertiesEXT>;

  //=== VK_EXT_attachment_feedback_loop_layout ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT>;

  //=== VK_EXT_4444_formats ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevice4444FormatsFeaturesEXT>;

  //=== VK_EXT_device_fault ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFaultFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceFaultCountsEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceFaultInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceFaultAddressInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceFaultVendorInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceFaultVendorBinaryHeaderVersionOneEXT>;

  //=== VK_EXT_rgba10x6_formats ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRGBA10X6FormatsFeaturesEXT>;

#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
  //=== VK_EXT_directfb_surface ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DirectFBSurfaceCreateInfoEXT>;
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

  //=== VK_EXT_vertex_input_dynamic_state ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexInputDynamicStateFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VertexInputBindingDescription2EXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VertexInputAttributeDescription2EXT>;

  //=== VK_EXT_physical_device_drm ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDrmPropertiesEXT>;

  //=== VK_EXT_device_address_binding_report ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceAddressBindingReportFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceAddressBindingCallbackDataEXT>;

  //=== VK_EXT_depth_clip_control ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClipControlFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineViewportDepthClipControlCreateInfoEXT>;

  //=== VK_EXT_primitive_topology_list_restart ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT>;

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_external_memory ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportMemoryZirconHandleInfoFUCHSIA>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryZirconHandlePropertiesFUCHSIA>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryGetZirconHandleInfoFUCHSIA>;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_external_semaphore ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportSemaphoreZirconHandleInfoFUCHSIA>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SemaphoreGetZirconHandleInfoFUCHSIA>;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_buffer_collection ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferCollectionCreateInfoFUCHSIA>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportMemoryBufferCollectionFUCHSIA>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferCollectionImageCreateInfoFUCHSIA>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferConstraintsInfoFUCHSIA>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferCollectionBufferCreateInfoFUCHSIA>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferCollectionPropertiesFUCHSIA>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SysmemColorSpaceFUCHSIA>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageConstraintsInfoFUCHSIA>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageFormatConstraintsInfoFUCHSIA>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferCollectionConstraintsInfoFUCHSIA>;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

  //=== VK_HUAWEI_subpass_shading ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubpassShadingPipelineCreateInfoHUAWEI>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSubpassShadingFeaturesHUAWEI>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSubpassShadingPropertiesHUAWEI>;

  //=== VK_HUAWEI_invocation_mask ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceInvocationMaskFeaturesHUAWEI>;

  //=== VK_NV_external_memory_rdma ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryGetRemoteAddressInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalMemoryRDMAFeaturesNV>;

  //=== VK_EXT_pipeline_properties ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelinePropertiesIdentifierEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelinePropertiesFeaturesEXT>;

  //=== VK_EXT_frame_boundary ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFrameBoundaryFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FrameBoundaryEXT>;

  //=== VK_EXT_multisampled_render_to_single_sampled ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultisampledRenderToSingleSampledFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubpassResolvePerformanceQueryEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MultisampledRenderToSingleSampledInfoEXT>;

  //=== VK_EXT_extended_dynamic_state2 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicState2FeaturesEXT>;

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
  //=== VK_QNX_screen_surface ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ScreenSurfaceCreateInfoQNX>;
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

  //=== VK_EXT_color_write_enable ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceColorWriteEnableFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineColorWriteCreateInfoEXT>;

  //=== VK_EXT_primitives_generated_query ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePrimitivesGeneratedQueryFeaturesEXT>;

  //=== VK_KHR_ray_tracing_maintenance1 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingMaintenance1FeaturesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TraceRaysIndirectCommand2KHR>;

  //=== VK_KHR_shader_untyped_pointers ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderUntypedPointersFeaturesKHR>;

  //=== VK_VALVE_video_encode_rgb_conversion ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoEncodeRgbConversionFeaturesVALVE>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeRgbConversionCapabilitiesVALVE>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeProfileRgbConversionInfoVALVE>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeSessionRgbConversionCreateInfoVALVE>;

  //=== VK_EXT_image_view_min_lod ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageViewMinLodFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageViewMinLodCreateInfoEXT>;

  //=== VK_EXT_multi_draw ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiDrawFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiDrawPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MultiDrawInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MultiDrawIndexedInfoEXT>;

  //=== VK_EXT_image_2d_view_of_3d ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImage2DViewOf3DFeaturesEXT>;

  //=== VK_EXT_shader_tile_image ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderTileImageFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderTileImagePropertiesEXT>;

  //=== VK_EXT_opacity_micromap ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MicromapBuildInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MicromapUsageEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MicromapCreateInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceOpacityMicromapFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceOpacityMicromapPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MicromapVersionInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyMicromapToMemoryInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyMemoryToMicromapInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyMicromapInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MicromapBuildSizesInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureTrianglesOpacityMicromapEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MicromapTriangleEXT>;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_NV_displacement_micromap ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDisplacementMicromapFeaturesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDisplacementMicromapPropertiesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureTrianglesDisplacementMicromapNV>;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_HUAWEI_cluster_culling_shader ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceClusterCullingShaderFeaturesHUAWEI>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceClusterCullingShaderPropertiesHUAWEI>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceClusterCullingShaderVrsFeaturesHUAWEI>;

  //=== VK_EXT_border_color_swizzle ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceBorderColorSwizzleFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SamplerBorderColorComponentMappingCreateInfoEXT>;

  //=== VK_EXT_pageable_device_local_memory ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT>;

  //=== VK_ARM_shader_core_properties ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCorePropertiesARM>;

  //=== VK_ARM_scheduling_controls ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceQueueShaderCoreControlCreateInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSchedulingControlsFeaturesARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSchedulingControlsPropertiesARM>;

  //=== VK_EXT_image_sliced_view_of_3d ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageSlicedViewOf3DFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageViewSlicedCreateInfoEXT>;

  //=== VK_VALVE_descriptor_set_host_mapping ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorSetHostMappingFeaturesVALVE>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorSetBindingReferenceVALVE>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorSetLayoutHostMappingInfoVALVE>;

  //=== VK_EXT_non_seamless_cube_map ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceNonSeamlessCubeMapFeaturesEXT>;

  //=== VK_ARM_render_pass_striped ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRenderPassStripedFeaturesARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRenderPassStripedPropertiesARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassStripeBeginInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassStripeInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassStripeSubmitInfoARM>;

  //=== VK_NV_copy_memory_indirect ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCopyMemoryIndirectFeaturesNV>;

  //=== VK_NV_memory_decompression ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DecompressMemoryRegionNV>;

  //=== VK_NV_device_generated_commands_compute ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceGeneratedCommandsComputeFeaturesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ComputePipelineIndirectBufferInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineIndirectDeviceAddressInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindPipelineIndirectCommandNV>;

  //=== VK_NV_ray_tracing_linear_swept_spheres ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingLinearSweptSpheresFeaturesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureGeometryLinearSweptSpheresDataNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureGeometrySpheresDataNV>;

  //=== VK_NV_linear_color_attachment ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceLinearColorAttachmentFeaturesNV>;

  //=== VK_KHR_shader_maximal_reconvergence ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR>;

  //=== VK_EXT_image_compression_control_swapchain ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageCompressionControlSwapchainFeaturesEXT>;

  //=== VK_QCOM_image_processing ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageViewSampleWeightCreateInfoQCOM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessingFeaturesQCOM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessingPropertiesQCOM>;

  //=== VK_EXT_nested_command_buffer ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceNestedCommandBufferFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceNestedCommandBufferPropertiesEXT>;

#if defined( VK_USE_PLATFORM_OHOS )
  //=== VK_OHOS_external_memory ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::NativeBufferUsageOHOS>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::NativeBufferPropertiesOHOS>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::NativeBufferFormatPropertiesOHOS>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportNativeBufferInfoOHOS>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryGetNativeBufferInfoOHOS>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalFormatOHOS>;
#endif /*VK_USE_PLATFORM_OHOS*/

  //=== VK_EXT_external_memory_acquire_unmodified ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalMemoryAcquireUnmodifiedEXT>;

  //=== VK_EXT_extended_dynamic_state3 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicState3FeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicState3PropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ColorBlendEquationEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ColorBlendAdvancedEXT>;

  //=== VK_EXT_subpass_merge_feedback ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSubpassMergeFeedbackFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassCreationControlEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassCreationFeedbackInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassCreationFeedbackCreateInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassSubpassFeedbackInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassSubpassFeedbackCreateInfoEXT>;

  //=== VK_LUNARG_direct_driver_loading ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DirectDriverLoadingInfoLUNARG>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DirectDriverLoadingListLUNARG>;

  //=== VK_ARM_tensors ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TensorDescriptionARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TensorCreateInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TensorViewCreateInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TensorMemoryRequirementsInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindTensorMemoryInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::WriteDescriptorSetTensorARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TensorFormatPropertiesARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceTensorPropertiesARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TensorMemoryBarrierARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TensorDependencyInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceTensorFeaturesARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceTensorMemoryRequirementsARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyTensorInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TensorCopyARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryDedicatedAllocateInfoTensorARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalTensorInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalTensorPropertiesARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalMemoryTensorCreateInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorBufferTensorFeaturesARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorBufferTensorPropertiesARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorGetTensorInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TensorCaptureDescriptorDataInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TensorViewCaptureDescriptorDataInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FrameBoundaryTensorsARM>;

  //=== VK_EXT_shader_module_identifier ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderModuleIdentifierFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderModuleIdentifierPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineShaderStageModuleIdentifierCreateInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ShaderModuleIdentifierEXT>;

  //=== VK_EXT_rasterization_order_attachment_access ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT>;

  //=== VK_NV_optical_flow ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceOpticalFlowFeaturesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceOpticalFlowPropertiesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::OpticalFlowImageFormatInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::OpticalFlowImageFormatPropertiesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::OpticalFlowSessionCreateInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::OpticalFlowSessionCreatePrivateDataInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::OpticalFlowExecuteInfoNV>;

  //=== VK_EXT_legacy_dithering ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceLegacyDitheringFeaturesEXT>;

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  //=== VK_ANDROID_external_format_resolve ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalFormatResolveFeaturesANDROID>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalFormatResolvePropertiesANDROID>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AndroidHardwareBufferFormatResolvePropertiesANDROID>;
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

  //=== VK_AMD_anti_lag ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceAntiLagFeaturesAMD>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AntiLagDataAMD>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AntiLagPresentationInfoAMD>;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_AMDX_dense_geometry_format ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDenseGeometryFormatFeaturesAMDX>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureDenseGeometryFormatTrianglesDataAMDX>;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_KHR_present_id2 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesPresentId2KHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PresentId2KHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePresentId2FeaturesKHR>;

  //=== VK_KHR_present_wait2 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesPresentWait2KHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePresentWait2FeaturesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PresentWait2InfoKHR>;

  //=== VK_KHR_ray_tracing_position_fetch ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPositionFetchFeaturesKHR>;

  //=== VK_EXT_shader_object ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderObjectFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderObjectPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ShaderCreateInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VertexInputBindingDescription2EXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VertexInputAttributeDescription2EXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ColorBlendEquationEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ColorBlendAdvancedEXT>;

  //=== VK_KHR_pipeline_binary ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineBinaryFeaturesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineBinaryPropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DevicePipelineBinaryInternalCacheControlKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineBinaryKeyKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineBinaryDataKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineBinaryKeysAndDataKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineBinaryCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineBinaryInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ReleaseCapturedPipelineDataInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineBinaryDataInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineBinaryHandlesInfoKHR>;

  //=== VK_QCOM_tile_properties ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceTilePropertiesFeaturesQCOM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TilePropertiesQCOM>;

  //=== VK_SEC_amigo_profiling ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceAmigoProfilingFeaturesSEC>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AmigoProfilingSubmitInfoSEC>;

  //=== VK_KHR_surface_maintenance1 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfacePresentModeKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfacePresentScalingCapabilitiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfacePresentModeCompatibilityKHR>;

  //=== VK_KHR_swapchain_maintenance1 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSwapchainMaintenance1FeaturesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SwapchainPresentFenceInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SwapchainPresentModesCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SwapchainPresentModeInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SwapchainPresentScalingCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ReleaseSwapchainImagesInfoKHR>;

  //=== VK_QCOM_multiview_per_view_viewports ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewPerViewViewportsFeaturesQCOM>;

  //=== VK_NV_ray_tracing_invocation_reorder ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingInvocationReorderPropertiesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingInvocationReorderFeaturesNV>;

  //=== VK_NV_cooperative_vector ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeVectorPropertiesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeVectorFeaturesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CooperativeVectorPropertiesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ConvertCooperativeVectorMatrixInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceOrHostAddressKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceOrHostAddressConstKHR>;

  //=== VK_NV_extended_sparse_address_space ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedSparseAddressSpaceFeaturesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedSparseAddressSpacePropertiesNV>;

  //=== VK_EXT_mutable_descriptor_type ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMutableDescriptorTypeFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MutableDescriptorTypeListEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MutableDescriptorTypeCreateInfoEXT>;

  //=== VK_EXT_legacy_vertex_attributes ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceLegacyVertexAttributesFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceLegacyVertexAttributesPropertiesEXT>;

  //=== VK_EXT_layer_settings ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::LayerSettingsCreateInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::LayerSettingEXT>;

  //=== VK_ARM_shader_core_builtins ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCoreBuiltinsFeaturesARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCoreBuiltinsPropertiesARM>;

  //=== VK_EXT_pipeline_library_group_handles ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineLibraryGroupHandlesFeaturesEXT>;

  //=== VK_EXT_dynamic_rendering_unused_attachments ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT>;

  //=== VK_NV_low_latency2 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::LatencySleepModeInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::LatencySleepInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SetLatencyMarkerInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GetLatencyMarkerInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::LatencyTimingsFrameReportNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::LatencySubmissionPresentIdNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SwapchainLatencyCreateInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::OutOfBandQueueTypeInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::LatencySurfaceCapabilitiesNV>;

  //=== VK_KHR_cooperative_matrix ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CooperativeMatrixPropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixFeaturesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixPropertiesKHR>;

  //=== VK_ARM_data_graph ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDataGraphFeaturesARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DataGraphPipelineConstantARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DataGraphPipelineResourceInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DataGraphPipelineCompilerControlCreateInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DataGraphPipelineCreateInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DataGraphPipelineShaderModuleCreateInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DataGraphPipelineSessionCreateInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DataGraphPipelineSessionBindPointRequirementsInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DataGraphPipelineSessionBindPointRequirementARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DataGraphPipelineSessionMemoryRequirementsInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindDataGraphPipelineSessionMemoryInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DataGraphPipelineInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DataGraphPipelinePropertyQueryResultARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DataGraphPipelineIdentifierCreateInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DataGraphPipelineDispatchInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDataGraphProcessingEngineARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueueFamilyDataGraphPropertiesARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DataGraphProcessingEngineCreateInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceQueueFamilyDataGraphProcessingEngineInfoARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueueFamilyDataGraphProcessingEnginePropertiesARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDataGraphOperationSupportARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DataGraphPipelineConstantTensorSemiStructuredSparsityInfoARM>;

  //=== VK_QCOM_multiview_per_view_render_areas ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewPerViewRenderAreasFeaturesQCOM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MultiviewPerViewRenderAreasRenderPassBeginInfoQCOM>;

  //=== VK_KHR_compute_shader_derivatives ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceComputeShaderDerivativesFeaturesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceComputeShaderDerivativesPropertiesKHR>;

  //=== VK_KHR_video_decode_av1 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeAV1ProfileInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeAV1CapabilitiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeAV1SessionParametersCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeAV1PictureInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeAV1DpbSlotInfoKHR>;

  //=== VK_KHR_video_encode_av1 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoEncodeAV1FeaturesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeAV1CapabilitiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeAV1QualityLevelPropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeAV1SessionCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeAV1SessionParametersCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeAV1PictureInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeAV1DpbSlotInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeAV1ProfileInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeAV1QIndexKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeAV1FrameSizeKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeAV1GopRemainingFrameInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeAV1RateControlInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeAV1RateControlLayerInfoKHR>;

  //=== VK_KHR_video_decode_vp9 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoDecodeVP9FeaturesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeVP9ProfileInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeVP9CapabilitiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeVP9PictureInfoKHR>;

  //=== VK_KHR_video_maintenance1 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoMaintenance1FeaturesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoInlineQueryInfoKHR>;

  //=== VK_NV_per_stage_descriptor_set ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePerStageDescriptorSetFeaturesNV>;

  //=== VK_QCOM_image_processing2 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessing2FeaturesQCOM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessing2PropertiesQCOM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SamplerBlockMatchWindowCreateInfoQCOM>;

  //=== VK_QCOM_filter_cubic_weights ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCubicWeightsFeaturesQCOM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SamplerCubicWeightsCreateInfoQCOM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BlitImageCubicWeightsInfoQCOM>;

  //=== VK_QCOM_ycbcr_degamma ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceYcbcrDegammaFeaturesQCOM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionYcbcrDegammaCreateInfoQCOM>;

  //=== VK_QCOM_filter_cubic_clamp ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCubicClampFeaturesQCOM>;

  //=== VK_EXT_attachment_feedback_loop_dynamic_state ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT>;

  //=== VK_KHR_unified_image_layouts ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AttachmentFeedbackLoopInfoEXT>;

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
  //=== VK_QNX_external_memory_screen_buffer ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ScreenBufferPropertiesQNX>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ScreenBufferFormatPropertiesQNX>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportScreenBufferInfoQNX>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalFormatQNX>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalMemoryScreenBufferFeaturesQNX>;
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

  //=== VK_MSFT_layered_driver ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceLayeredDriverPropertiesMSFT>;

  //=== VK_KHR_calibrated_timestamps ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CalibratedTimestampInfoKHR>;

  //=== VK_KHR_maintenance6 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SetDescriptorBufferOffsetsInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindDescriptorBufferEmbeddedSamplersInfoEXT>;

  //=== VK_NV_descriptor_pool_overallocation ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorPoolOverallocationFeaturesNV>;

  //=== VK_QCOM_tile_memory_heap ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceTileMemoryHeapFeaturesQCOM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceTileMemoryHeapPropertiesQCOM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TileMemoryRequirementsQCOM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TileMemoryBindInfoQCOM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TileMemorySizeInfoQCOM>;

  //=== VK_KHR_copy_memory_indirect ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::StridedDeviceAddressRangeKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyMemoryIndirectCommandKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyMemoryIndirectInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyMemoryToImageIndirectCommandKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyMemoryToImageIndirectInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCopyMemoryIndirectFeaturesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCopyMemoryIndirectPropertiesKHR>;

  //=== VK_EXT_memory_decompression ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DecompressMemoryInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DecompressMemoryRegionEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryDecompressionFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryDecompressionPropertiesEXT>;

  //=== VK_NV_display_stereo ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplaySurfaceStereoCreateInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayModeStereoPropertiesNV>;

  //=== VK_KHR_video_encode_intra_refresh ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeIntraRefreshCapabilitiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeSessionIntraRefreshCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeIntraRefreshInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoReferenceIntraRefreshInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoEncodeIntraRefreshFeaturesKHR>;

  //=== VK_KHR_video_encode_quantization_map ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeQuantizationMapCapabilitiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoFormatQuantizationMapPropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeQuantizationMapInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeQuantizationMapSessionParametersCreateInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoEncodeQuantizationMapFeaturesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264QuantizationMapCapabilitiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265QuantizationMapCapabilitiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoFormatH265QuantizationMapPropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeAV1QuantizationMapCapabilitiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoFormatAV1QuantizationMapPropertiesKHR>;

  //=== VK_NV_raw_access_chains ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRawAccessChainsFeaturesNV>;

  //=== VK_NV_external_compute_queue ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalComputeQueueDeviceCreateInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalComputeQueueCreateInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalComputeQueueDataParamsNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalComputeQueuePropertiesNV>;

  //=== VK_KHR_shader_relaxed_extended_instruction ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderRelaxedExtendedInstructionFeaturesKHR>;

  //=== VK_NV_command_buffer_inheritance ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCommandBufferInheritanceFeaturesNV>;

  //=== VK_KHR_maintenance7 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance7FeaturesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance7PropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceLayeredApiPropertiesListKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceLayeredApiPropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceLayeredApiVulkanPropertiesKHR>;

  //=== VK_NV_shader_atomic_float16_vector ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicFloat16VectorFeaturesNV>;

  //=== VK_EXT_shader_replicated_composites ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderReplicatedCompositesFeaturesEXT>;

  //=== VK_EXT_shader_float8 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderFloat8FeaturesEXT>;

  //=== VK_NV_ray_tracing_validation ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingValidationFeaturesNV>;

  //=== VK_NV_cluster_acceleration_structure ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceClusterAccelerationStructureFeaturesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceClusterAccelerationStructurePropertiesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureClustersBottomLevelInputNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureTriangleClusterInputNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureMoveObjectsInputNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureOpInputNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureInputInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureCommandsInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::StridedDeviceAddressNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureGeometryIndexAndGeometryFlagsNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureMoveObjectsInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureBuildClustersBottomLevelInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureBuildTriangleClusterInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureInstantiateClusterInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ClusterAccelerationStructureGetTemplateIndicesInfoNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RayTracingPipelineClusterAccelerationStructureCreateInfoNV>;

  //=== VK_NV_partitioned_acceleration_structure ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePartitionedAccelerationStructureFeaturesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePartitionedAccelerationStructurePropertiesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PartitionedAccelerationStructureFlagsNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BuildPartitionedAccelerationStructureIndirectCommandNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PartitionedAccelerationStructureWriteInstanceDataNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PartitionedAccelerationStructureUpdateInstanceDataNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PartitionedAccelerationStructureWritePartitionTranslationDataNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::WriteDescriptorSetPartitionedAccelerationStructureNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PartitionedAccelerationStructureInstancesInputNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BuildPartitionedAccelerationStructureInfoNV>;

  //=== VK_EXT_device_generated_commands ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceGeneratedCommandsFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceGeneratedCommandsPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GeneratedCommandsMemoryRequirementsInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IndirectExecutionSetCreateInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IndirectExecutionSetInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IndirectExecutionSetPipelineInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IndirectExecutionSetShaderInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GeneratedCommandsInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::WriteIndirectExecutionSetPipelineEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutCreateInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutTokenEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DrawIndirectCountIndirectCommandEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IndirectCommandsVertexBufferTokenEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindVertexBufferIndirectCommandEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IndirectCommandsIndexBufferTokenEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindIndexBufferIndirectCommandEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IndirectCommandsPushConstantTokenEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IndirectCommandsExecutionSetTokenEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IndirectCommandsTokenDataEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IndirectExecutionSetShaderLayoutInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GeneratedCommandsPipelineInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GeneratedCommandsShaderInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::WriteIndirectExecutionSetShaderEXT>;

  //=== VK_KHR_maintenance8 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryBarrierAccessFlags3KHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance8FeaturesKHR>;

  //=== VK_MESA_image_alignment_control ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageAlignmentControlFeaturesMESA>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageAlignmentControlPropertiesMESA>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageAlignmentControlCreateInfoMESA>;

  //=== VK_KHR_shader_fma ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderFmaFeaturesKHR>;

  //=== VK_EXT_ray_tracing_invocation_reorder ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingInvocationReorderPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingInvocationReorderFeaturesEXT>;

  //=== VK_EXT_depth_clamp_control ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClampControlFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineViewportDepthClampControlCreateInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DepthClampRangeEXT>;

  //=== VK_KHR_maintenance9 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance9FeaturesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance9PropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueueFamilyOwnershipTransferPropertiesKHR>;

  //=== VK_KHR_video_maintenance2 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoMaintenance2FeaturesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH264InlineSessionParametersInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH265InlineSessionParametersInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeAV1InlineSessionParametersInfoKHR>;

#if defined( VK_USE_PLATFORM_OHOS )
  //=== VK_OHOS_surface ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceCreateInfoOHOS>;
#endif /*VK_USE_PLATFORM_OHOS*/

#if defined( VK_USE_PLATFORM_OHOS )
  //=== VK_OHOS_native_buffer ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::NativeBufferOHOS>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SwapchainImageCreateInfoOHOS>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePresentationPropertiesOHOS>;
#endif /*VK_USE_PLATFORM_OHOS*/

  //=== VK_HUAWEI_hdr_vivid ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceHdrVividFeaturesHUAWEI>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::HdrVividDynamicMetadataHUAWEI>;

  //=== VK_NV_cooperative_matrix2 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CooperativeMatrixFlexibleDimensionsPropertiesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrix2FeaturesNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrix2PropertiesNV>;

  //=== VK_ARM_pipeline_opacity_micromap ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineOpacityMicromapFeaturesARM>;

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_external_memory_metal ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportMemoryMetalHandleInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryMetalHandlePropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryGetMetalHandleInfoEXT>;
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_KHR_depth_clamp_zero_one ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClampZeroOneFeaturesKHR>;

  //=== VK_ARM_performance_counters_by_region ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePerformanceCountersByRegionFeaturesARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePerformanceCountersByRegionPropertiesARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PerformanceCounterARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PerformanceCounterDescriptionARM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassPerformanceCountersByRegionBeginInfoARM>;

  //=== VK_EXT_vertex_attribute_robustness ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeRobustnessFeaturesEXT>;

  //=== VK_ARM_format_pack ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFormatPackFeaturesARM>;

  //=== VK_VALVE_fragment_density_map_layered ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapLayeredFeaturesVALVE>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapLayeredPropertiesVALVE>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineFragmentDensityMapLayeredCreateInfoVALVE>;

  //=== VK_KHR_robustness2 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRobustness2FeaturesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRobustness2PropertiesKHR>;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_NV_present_metering ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SetPresentConfigNV>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePresentMeteringFeaturesNV>;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_EXT_fragment_density_map_offset ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapOffsetFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapOffsetPropertiesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassFragmentDensityMapOffsetEndInfoEXT>;

  //=== VK_EXT_zero_initialize_device_memory ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceZeroInitializeDeviceMemoryFeaturesEXT>;

  //=== VK_KHR_present_mode_fifo_latest_ready ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePresentModeFifoLatestReadyFeaturesKHR>;

  //=== VK_EXT_shader_64bit_indexing ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShader64BitIndexingFeaturesEXT>;

  //=== VK_EXT_custom_resolve ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCustomResolveFeaturesEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BeginCustomResolveInfoEXT>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CustomResolveCreateInfoEXT>;

  //=== VK_QCOM_data_graph_model ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineCacheHeaderVersionDataGraphQCOM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DataGraphPipelineBuiltinModelCreateInfoQCOM>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDataGraphModelFeaturesQCOM>;

  //=== VK_KHR_maintenance10 ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance10FeaturesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance10PropertiesKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderingEndInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderingAttachmentFlagsInfoKHR>;
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ResolveImageModeInfoKHR>;

  //=== VK_SEC_pipeline_cache_incremental_mode ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineCacheIncrementalModeFeaturesSEC>;

  //=== VK_EXT_shader_uniform_buffer_unsized_array ===
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderUniformBufferUnsizedArrayFeaturesEXT>;

  //=================================================================
  //=== Required exports for VULKAN_HPP_NAMESPACE::StructureChain ===
  //=================================================================

#if !defined( VULKAN_HPP_DISABLE_ENHANCED_MODE )
  using std::tuple_element;
  using std::tuple_size;
#endif
}  // namespace std

export
{
  // This VkFlags type is used as part of a bitfield in some structures.
  // As it can't be mimicked by vk-data types, we need to export just that.
  using ::VkGeometryInstanceFlagsKHR;

  //==================
  //=== PFN TYPEs ===
  //==================

  //=== VK_VERSION_1_0 ===
  using ::PFN_vkAllocateCommandBuffers;
  using ::PFN_vkAllocateDescriptorSets;
  using ::PFN_vkAllocateMemory;
  using ::PFN_vkBeginCommandBuffer;
  using ::PFN_vkBindBufferMemory;
  using ::PFN_vkBindImageMemory;
  using ::PFN_vkCmdBeginQuery;
  using ::PFN_vkCmdBeginRenderPass;
  using ::PFN_vkCmdBindDescriptorSets;
  using ::PFN_vkCmdBindIndexBuffer;
  using ::PFN_vkCmdBindPipeline;
  using ::PFN_vkCmdBindVertexBuffers;
  using ::PFN_vkCmdBlitImage;
  using ::PFN_vkCmdClearAttachments;
  using ::PFN_vkCmdClearColorImage;
  using ::PFN_vkCmdClearDepthStencilImage;
  using ::PFN_vkCmdCopyBuffer;
  using ::PFN_vkCmdCopyBufferToImage;
  using ::PFN_vkCmdCopyImage;
  using ::PFN_vkCmdCopyImageToBuffer;
  using ::PFN_vkCmdCopyQueryPoolResults;
  using ::PFN_vkCmdDispatch;
  using ::PFN_vkCmdDispatchIndirect;
  using ::PFN_vkCmdDraw;
  using ::PFN_vkCmdDrawIndexed;
  using ::PFN_vkCmdDrawIndexedIndirect;
  using ::PFN_vkCmdDrawIndirect;
  using ::PFN_vkCmdEndQuery;
  using ::PFN_vkCmdEndRenderPass;
  using ::PFN_vkCmdExecuteCommands;
  using ::PFN_vkCmdFillBuffer;
  using ::PFN_vkCmdNextSubpass;
  using ::PFN_vkCmdPipelineBarrier;
  using ::PFN_vkCmdPushConstants;
  using ::PFN_vkCmdResetEvent;
  using ::PFN_vkCmdResetQueryPool;
  using ::PFN_vkCmdResolveImage;
  using ::PFN_vkCmdSetBlendConstants;
  using ::PFN_vkCmdSetDepthBias;
  using ::PFN_vkCmdSetDepthBounds;
  using ::PFN_vkCmdSetEvent;
  using ::PFN_vkCmdSetLineWidth;
  using ::PFN_vkCmdSetScissor;
  using ::PFN_vkCmdSetStencilCompareMask;
  using ::PFN_vkCmdSetStencilReference;
  using ::PFN_vkCmdSetStencilWriteMask;
  using ::PFN_vkCmdSetViewport;
  using ::PFN_vkCmdUpdateBuffer;
  using ::PFN_vkCmdWaitEvents;
  using ::PFN_vkCmdWriteTimestamp;
  using ::PFN_vkCreateBuffer;
  using ::PFN_vkCreateBufferView;
  using ::PFN_vkCreateCommandPool;
  using ::PFN_vkCreateComputePipelines;
  using ::PFN_vkCreateDescriptorPool;
  using ::PFN_vkCreateDescriptorSetLayout;
  using ::PFN_vkCreateDevice;
  using ::PFN_vkCreateEvent;
  using ::PFN_vkCreateFence;
  using ::PFN_vkCreateFramebuffer;
  using ::PFN_vkCreateGraphicsPipelines;
  using ::PFN_vkCreateImage;
  using ::PFN_vkCreateImageView;
  using ::PFN_vkCreateInstance;
  using ::PFN_vkCreatePipelineCache;
  using ::PFN_vkCreatePipelineLayout;
  using ::PFN_vkCreateQueryPool;
  using ::PFN_vkCreateRenderPass;
  using ::PFN_vkCreateSampler;
  using ::PFN_vkCreateSemaphore;
  using ::PFN_vkCreateShaderModule;
  using ::PFN_vkDestroyBuffer;
  using ::PFN_vkDestroyBufferView;
  using ::PFN_vkDestroyCommandPool;
  using ::PFN_vkDestroyDescriptorPool;
  using ::PFN_vkDestroyDescriptorSetLayout;
  using ::PFN_vkDestroyDevice;
  using ::PFN_vkDestroyEvent;
  using ::PFN_vkDestroyFence;
  using ::PFN_vkDestroyFramebuffer;
  using ::PFN_vkDestroyImage;
  using ::PFN_vkDestroyImageView;
  using ::PFN_vkDestroyInstance;
  using ::PFN_vkDestroyPipeline;
  using ::PFN_vkDestroyPipelineCache;
  using ::PFN_vkDestroyPipelineLayout;
  using ::PFN_vkDestroyQueryPool;
  using ::PFN_vkDestroyRenderPass;
  using ::PFN_vkDestroySampler;
  using ::PFN_vkDestroySemaphore;
  using ::PFN_vkDestroyShaderModule;
  using ::PFN_vkDeviceWaitIdle;
  using ::PFN_vkEndCommandBuffer;
  using ::PFN_vkEnumerateDeviceExtensionProperties;
  using ::PFN_vkEnumerateDeviceLayerProperties;
  using ::PFN_vkEnumerateInstanceExtensionProperties;
  using ::PFN_vkEnumerateInstanceLayerProperties;
  using ::PFN_vkEnumeratePhysicalDevices;
  using ::PFN_vkFlushMappedMemoryRanges;
  using ::PFN_vkFreeCommandBuffers;
  using ::PFN_vkFreeDescriptorSets;
  using ::PFN_vkFreeMemory;
  using ::PFN_vkGetBufferMemoryRequirements;
  using ::PFN_vkGetDeviceMemoryCommitment;
  using ::PFN_vkGetDeviceProcAddr;
  using ::PFN_vkGetDeviceQueue;
  using ::PFN_vkGetEventStatus;
  using ::PFN_vkGetFenceStatus;
  using ::PFN_vkGetImageMemoryRequirements;
  using ::PFN_vkGetImageSparseMemoryRequirements;
  using ::PFN_vkGetImageSubresourceLayout;
  using ::PFN_vkGetInstanceProcAddr;
  using ::PFN_vkGetPhysicalDeviceFeatures;
  using ::PFN_vkGetPhysicalDeviceFormatProperties;
  using ::PFN_vkGetPhysicalDeviceImageFormatProperties;
  using ::PFN_vkGetPhysicalDeviceMemoryProperties;
  using ::PFN_vkGetPhysicalDeviceProperties;
  using ::PFN_vkGetPhysicalDeviceQueueFamilyProperties;
  using ::PFN_vkGetPhysicalDeviceSparseImageFormatProperties;
  using ::PFN_vkGetPipelineCacheData;
  using ::PFN_vkGetQueryPoolResults;
  using ::PFN_vkGetRenderAreaGranularity;
  using ::PFN_vkInvalidateMappedMemoryRanges;
  using ::PFN_vkMapMemory;
  using ::PFN_vkMergePipelineCaches;
  using ::PFN_vkQueueBindSparse;
  using ::PFN_vkQueueSubmit;
  using ::PFN_vkQueueWaitIdle;
  using ::PFN_vkResetCommandBuffer;
  using ::PFN_vkResetCommandPool;
  using ::PFN_vkResetDescriptorPool;
  using ::PFN_vkResetEvent;
  using ::PFN_vkResetFences;
  using ::PFN_vkSetEvent;
  using ::PFN_vkUnmapMemory;
  using ::PFN_vkUpdateDescriptorSets;
  using ::PFN_vkWaitForFences;

  //=== VK_VERSION_1_1 ===
  using ::PFN_vkBindBufferMemory2;
  using ::PFN_vkBindImageMemory2;
  using ::PFN_vkCmdDispatchBase;
  using ::PFN_vkCmdSetDeviceMask;
  using ::PFN_vkCreateDescriptorUpdateTemplate;
  using ::PFN_vkCreateSamplerYcbcrConversion;
  using ::PFN_vkDestroyDescriptorUpdateTemplate;
  using ::PFN_vkDestroySamplerYcbcrConversion;
  using ::PFN_vkEnumerateInstanceVersion;
  using ::PFN_vkEnumeratePhysicalDeviceGroups;
  using ::PFN_vkGetBufferMemoryRequirements2;
  using ::PFN_vkGetDescriptorSetLayoutSupport;
  using ::PFN_vkGetDeviceGroupPeerMemoryFeatures;
  using ::PFN_vkGetDeviceQueue2;
  using ::PFN_vkGetImageMemoryRequirements2;
  using ::PFN_vkGetImageSparseMemoryRequirements2;
  using ::PFN_vkGetPhysicalDeviceExternalBufferProperties;
  using ::PFN_vkGetPhysicalDeviceExternalFenceProperties;
  using ::PFN_vkGetPhysicalDeviceExternalSemaphoreProperties;
  using ::PFN_vkGetPhysicalDeviceFeatures2;
  using ::PFN_vkGetPhysicalDeviceFormatProperties2;
  using ::PFN_vkGetPhysicalDeviceImageFormatProperties2;
  using ::PFN_vkGetPhysicalDeviceMemoryProperties2;
  using ::PFN_vkGetPhysicalDeviceProperties2;
  using ::PFN_vkGetPhysicalDeviceQueueFamilyProperties2;
  using ::PFN_vkGetPhysicalDeviceSparseImageFormatProperties2;
  using ::PFN_vkTrimCommandPool;
  using ::PFN_vkUpdateDescriptorSetWithTemplate;

  //=== VK_VERSION_1_2 ===
  using ::PFN_vkCmdBeginRenderPass2;
  using ::PFN_vkCmdDrawIndexedIndirectCount;
  using ::PFN_vkCmdDrawIndirectCount;
  using ::PFN_vkCmdEndRenderPass2;
  using ::PFN_vkCmdNextSubpass2;
  using ::PFN_vkCreateRenderPass2;
  using ::PFN_vkGetBufferDeviceAddress;
  using ::PFN_vkGetBufferOpaqueCaptureAddress;
  using ::PFN_vkGetDeviceMemoryOpaqueCaptureAddress;
  using ::PFN_vkGetSemaphoreCounterValue;
  using ::PFN_vkResetQueryPool;
  using ::PFN_vkSignalSemaphore;
  using ::PFN_vkWaitSemaphores;

  //=== VK_VERSION_1_3 ===
  using ::PFN_vkCmdBeginRendering;
  using ::PFN_vkCmdBindVertexBuffers2;
  using ::PFN_vkCmdBlitImage2;
  using ::PFN_vkCmdCopyBuffer2;
  using ::PFN_vkCmdCopyBufferToImage2;
  using ::PFN_vkCmdCopyImage2;
  using ::PFN_vkCmdCopyImageToBuffer2;
  using ::PFN_vkCmdEndRendering;
  using ::PFN_vkCmdPipelineBarrier2;
  using ::PFN_vkCmdResetEvent2;
  using ::PFN_vkCmdResolveImage2;
  using ::PFN_vkCmdSetCullMode;
  using ::PFN_vkCmdSetDepthBiasEnable;
  using ::PFN_vkCmdSetDepthBoundsTestEnable;
  using ::PFN_vkCmdSetDepthCompareOp;
  using ::PFN_vkCmdSetDepthTestEnable;
  using ::PFN_vkCmdSetDepthWriteEnable;
  using ::PFN_vkCmdSetEvent2;
  using ::PFN_vkCmdSetFrontFace;
  using ::PFN_vkCmdSetPrimitiveRestartEnable;
  using ::PFN_vkCmdSetPrimitiveTopology;
  using ::PFN_vkCmdSetRasterizerDiscardEnable;
  using ::PFN_vkCmdSetScissorWithCount;
  using ::PFN_vkCmdSetStencilOp;
  using ::PFN_vkCmdSetStencilTestEnable;
  using ::PFN_vkCmdSetViewportWithCount;
  using ::PFN_vkCmdWaitEvents2;
  using ::PFN_vkCmdWriteTimestamp2;
  using ::PFN_vkCreatePrivateDataSlot;
  using ::PFN_vkDestroyPrivateDataSlot;
  using ::PFN_vkGetDeviceBufferMemoryRequirements;
  using ::PFN_vkGetDeviceImageMemoryRequirements;
  using ::PFN_vkGetDeviceImageSparseMemoryRequirements;
  using ::PFN_vkGetPhysicalDeviceToolProperties;
  using ::PFN_vkGetPrivateData;
  using ::PFN_vkQueueSubmit2;
  using ::PFN_vkSetPrivateData;

  //=== VK_VERSION_1_4 ===
  using ::PFN_vkCmdBindDescriptorSets2;
  using ::PFN_vkCmdBindIndexBuffer2;
  using ::PFN_vkCmdPushConstants2;
  using ::PFN_vkCmdPushDescriptorSet;
  using ::PFN_vkCmdPushDescriptorSet2;
  using ::PFN_vkCmdPushDescriptorSetWithTemplate;
  using ::PFN_vkCmdPushDescriptorSetWithTemplate2;
  using ::PFN_vkCmdSetLineStipple;
  using ::PFN_vkCmdSetRenderingAttachmentLocations;
  using ::PFN_vkCmdSetRenderingInputAttachmentIndices;
  using ::PFN_vkCopyImageToImage;
  using ::PFN_vkCopyImageToMemory;
  using ::PFN_vkCopyMemoryToImage;
  using ::PFN_vkGetDeviceImageSubresourceLayout;
  using ::PFN_vkGetImageSubresourceLayout2;
  using ::PFN_vkGetRenderingAreaGranularity;
  using ::PFN_vkMapMemory2;
  using ::PFN_vkTransitionImageLayout;
  using ::PFN_vkUnmapMemory2;

  //=== VK_KHR_surface ===
  using ::PFN_vkDestroySurfaceKHR;
  using ::PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR;
  using ::PFN_vkGetPhysicalDeviceSurfaceFormatsKHR;
  using ::PFN_vkGetPhysicalDeviceSurfacePresentModesKHR;
  using ::PFN_vkGetPhysicalDeviceSurfaceSupportKHR;

  //=== VK_KHR_swapchain ===
  using ::PFN_vkAcquireNextImage2KHR;
  using ::PFN_vkAcquireNextImageKHR;
  using ::PFN_vkCreateSwapchainKHR;
  using ::PFN_vkDestroySwapchainKHR;
  using ::PFN_vkGetDeviceGroupPresentCapabilitiesKHR;
  using ::PFN_vkGetDeviceGroupSurfacePresentModesKHR;
  using ::PFN_vkGetPhysicalDevicePresentRectanglesKHR;
  using ::PFN_vkGetSwapchainImagesKHR;
  using ::PFN_vkQueuePresentKHR;

  //=== VK_KHR_display ===
  using ::PFN_vkCreateDisplayModeKHR;
  using ::PFN_vkCreateDisplayPlaneSurfaceKHR;
  using ::PFN_vkGetDisplayModePropertiesKHR;
  using ::PFN_vkGetDisplayPlaneCapabilitiesKHR;
  using ::PFN_vkGetDisplayPlaneSupportedDisplaysKHR;
  using ::PFN_vkGetPhysicalDeviceDisplayPlanePropertiesKHR;
  using ::PFN_vkGetPhysicalDeviceDisplayPropertiesKHR;

  //=== VK_KHR_display_swapchain ===
  using ::PFN_vkCreateSharedSwapchainsKHR;

#if defined( VK_USE_PLATFORM_XLIB_KHR )
  //=== VK_KHR_xlib_surface ===
  using ::PFN_vkCreateXlibSurfaceKHR;
  using ::PFN_vkGetPhysicalDeviceXlibPresentationSupportKHR;
#endif /*VK_USE_PLATFORM_XLIB_KHR*/

#if defined( VK_USE_PLATFORM_XCB_KHR )
  //=== VK_KHR_xcb_surface ===
  using ::PFN_vkCreateXcbSurfaceKHR;
  using ::PFN_vkGetPhysicalDeviceXcbPresentationSupportKHR;
#endif /*VK_USE_PLATFORM_XCB_KHR*/

#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
  //=== VK_KHR_wayland_surface ===
  using ::PFN_vkCreateWaylandSurfaceKHR;
  using ::PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR;
#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  //=== VK_KHR_android_surface ===
  using ::PFN_vkCreateAndroidSurfaceKHR;
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_win32_surface ===
  using ::PFN_vkCreateWin32SurfaceKHR;
  using ::PFN_vkGetPhysicalDeviceWin32PresentationSupportKHR;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_EXT_debug_report ===
  using ::PFN_vkCreateDebugReportCallbackEXT;
  using ::PFN_vkDebugReportMessageEXT;
  using ::PFN_vkDestroyDebugReportCallbackEXT;

  //=== VK_EXT_debug_marker ===
  using ::PFN_vkCmdDebugMarkerBeginEXT;
  using ::PFN_vkCmdDebugMarkerEndEXT;
  using ::PFN_vkCmdDebugMarkerInsertEXT;
  using ::PFN_vkDebugMarkerSetObjectNameEXT;
  using ::PFN_vkDebugMarkerSetObjectTagEXT;

  //=== VK_KHR_video_queue ===
  using ::PFN_vkBindVideoSessionMemoryKHR;
  using ::PFN_vkCmdBeginVideoCodingKHR;
  using ::PFN_vkCmdControlVideoCodingKHR;
  using ::PFN_vkCmdEndVideoCodingKHR;
  using ::PFN_vkCreateVideoSessionKHR;
  using ::PFN_vkCreateVideoSessionParametersKHR;
  using ::PFN_vkDestroyVideoSessionKHR;
  using ::PFN_vkDestroyVideoSessionParametersKHR;
  using ::PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR;
  using ::PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR;
  using ::PFN_vkGetVideoSessionMemoryRequirementsKHR;
  using ::PFN_vkUpdateVideoSessionParametersKHR;

  //=== VK_KHR_video_decode_queue ===
  using ::PFN_vkCmdDecodeVideoKHR;

  //=== VK_EXT_transform_feedback ===
  using ::PFN_vkCmdBeginQueryIndexedEXT;
  using ::PFN_vkCmdBeginTransformFeedbackEXT;
  using ::PFN_vkCmdBindTransformFeedbackBuffersEXT;
  using ::PFN_vkCmdDrawIndirectByteCountEXT;
  using ::PFN_vkCmdEndQueryIndexedEXT;
  using ::PFN_vkCmdEndTransformFeedbackEXT;

  //=== VK_NVX_binary_import ===
  using ::PFN_vkCmdCuLaunchKernelNVX;
  using ::PFN_vkCreateCuFunctionNVX;
  using ::PFN_vkCreateCuModuleNVX;
  using ::PFN_vkDestroyCuFunctionNVX;
  using ::PFN_vkDestroyCuModuleNVX;

  //=== VK_NVX_image_view_handle ===
  using ::PFN_vkGetImageViewAddressNVX;
  using ::PFN_vkGetImageViewHandle64NVX;
  using ::PFN_vkGetImageViewHandleNVX;

  //=== VK_AMD_draw_indirect_count ===
  using ::PFN_vkCmdDrawIndexedIndirectCountAMD;
  using ::PFN_vkCmdDrawIndirectCountAMD;

  //=== VK_AMD_shader_info ===
  using ::PFN_vkGetShaderInfoAMD;

  //=== VK_KHR_dynamic_rendering ===
  using ::PFN_vkCmdBeginRenderingKHR;
  using ::PFN_vkCmdEndRenderingKHR;

#if defined( VK_USE_PLATFORM_GGP )
  //=== VK_GGP_stream_descriptor_surface ===
  using ::PFN_vkCreateStreamDescriptorSurfaceGGP;
#endif /*VK_USE_PLATFORM_GGP*/

  //=== VK_NV_external_memory_capabilities ===
  using ::PFN_vkGetPhysicalDeviceExternalImageFormatPropertiesNV;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_NV_external_memory_win32 ===
  using ::PFN_vkGetMemoryWin32HandleNV;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_KHR_get_physical_device_properties2 ===
  using ::PFN_vkGetPhysicalDeviceFeatures2KHR;
  using ::PFN_vkGetPhysicalDeviceFormatProperties2KHR;
  using ::PFN_vkGetPhysicalDeviceImageFormatProperties2KHR;
  using ::PFN_vkGetPhysicalDeviceMemoryProperties2KHR;
  using ::PFN_vkGetPhysicalDeviceProperties2KHR;
  using ::PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR;
  using ::PFN_vkGetPhysicalDeviceSparseImageFormatProperties2KHR;

  //=== VK_KHR_device_group ===
  using ::PFN_vkCmdDispatchBaseKHR;
  using ::PFN_vkCmdSetDeviceMaskKHR;
  using ::PFN_vkGetDeviceGroupPeerMemoryFeaturesKHR;

#if defined( VK_USE_PLATFORM_VI_NN )
  //=== VK_NN_vi_surface ===
  using ::PFN_vkCreateViSurfaceNN;
#endif /*VK_USE_PLATFORM_VI_NN*/

  //=== VK_KHR_maintenance1 ===
  using ::PFN_vkTrimCommandPoolKHR;

  //=== VK_KHR_device_group_creation ===
  using ::PFN_vkEnumeratePhysicalDeviceGroupsKHR;

  //=== VK_KHR_external_memory_capabilities ===
  using ::PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_external_memory_win32 ===
  using ::PFN_vkGetMemoryWin32HandleKHR;
  using ::PFN_vkGetMemoryWin32HandlePropertiesKHR;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_KHR_external_memory_fd ===
  using ::PFN_vkGetMemoryFdKHR;
  using ::PFN_vkGetMemoryFdPropertiesKHR;

  //=== VK_KHR_external_semaphore_capabilities ===
  using ::PFN_vkGetPhysicalDeviceExternalSemaphorePropertiesKHR;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_external_semaphore_win32 ===
  using ::PFN_vkGetSemaphoreWin32HandleKHR;
  using ::PFN_vkImportSemaphoreWin32HandleKHR;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_KHR_external_semaphore_fd ===
  using ::PFN_vkGetSemaphoreFdKHR;
  using ::PFN_vkImportSemaphoreFdKHR;

  //=== VK_KHR_push_descriptor ===
  using ::PFN_vkCmdPushDescriptorSetKHR;
  using ::PFN_vkCmdPushDescriptorSetWithTemplateKHR;

  //=== VK_EXT_conditional_rendering ===
  using ::PFN_vkCmdBeginConditionalRenderingEXT;
  using ::PFN_vkCmdEndConditionalRenderingEXT;

  //=== VK_KHR_descriptor_update_template ===
  using ::PFN_vkCreateDescriptorUpdateTemplateKHR;
  using ::PFN_vkDestroyDescriptorUpdateTemplateKHR;
  using ::PFN_vkUpdateDescriptorSetWithTemplateKHR;

  //=== VK_NV_clip_space_w_scaling ===
  using ::PFN_vkCmdSetViewportWScalingNV;

  //=== VK_EXT_direct_mode_display ===
  using ::PFN_vkReleaseDisplayEXT;

#if defined( VK_USE_PLATFORM_XLIB_XRANDR_EXT )
  //=== VK_EXT_acquire_xlib_display ===
  using ::PFN_vkAcquireXlibDisplayEXT;
  using ::PFN_vkGetRandROutputDisplayEXT;
#endif /*VK_USE_PLATFORM_XLIB_XRANDR_EXT*/

  //=== VK_EXT_display_surface_counter ===
  using ::PFN_vkGetPhysicalDeviceSurfaceCapabilities2EXT;

  //=== VK_EXT_display_control ===
  using ::PFN_vkDisplayPowerControlEXT;
  using ::PFN_vkGetSwapchainCounterEXT;
  using ::PFN_vkRegisterDeviceEventEXT;
  using ::PFN_vkRegisterDisplayEventEXT;

  //=== VK_GOOGLE_display_timing ===
  using ::PFN_vkGetPastPresentationTimingGOOGLE;
  using ::PFN_vkGetRefreshCycleDurationGOOGLE;

  //=== VK_EXT_discard_rectangles ===
  using ::PFN_vkCmdSetDiscardRectangleEnableEXT;
  using ::PFN_vkCmdSetDiscardRectangleEXT;
  using ::PFN_vkCmdSetDiscardRectangleModeEXT;

  //=== VK_EXT_hdr_metadata ===
  using ::PFN_vkSetHdrMetadataEXT;

  //=== VK_KHR_create_renderpass2 ===
  using ::PFN_vkCmdBeginRenderPass2KHR;
  using ::PFN_vkCmdEndRenderPass2KHR;
  using ::PFN_vkCmdNextSubpass2KHR;
  using ::PFN_vkCreateRenderPass2KHR;

  //=== VK_KHR_shared_presentable_image ===
  using ::PFN_vkGetSwapchainStatusKHR;

  //=== VK_KHR_external_fence_capabilities ===
  using ::PFN_vkGetPhysicalDeviceExternalFencePropertiesKHR;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_external_fence_win32 ===
  using ::PFN_vkGetFenceWin32HandleKHR;
  using ::PFN_vkImportFenceWin32HandleKHR;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_KHR_external_fence_fd ===
  using ::PFN_vkGetFenceFdKHR;
  using ::PFN_vkImportFenceFdKHR;

  //=== VK_KHR_performance_query ===
  using ::PFN_vkAcquireProfilingLockKHR;
  using ::PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR;
  using ::PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR;
  using ::PFN_vkReleaseProfilingLockKHR;

  //=== VK_KHR_get_surface_capabilities2 ===
  using ::PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR;
  using ::PFN_vkGetPhysicalDeviceSurfaceFormats2KHR;

  //=== VK_KHR_get_display_properties2 ===
  using ::PFN_vkGetDisplayModeProperties2KHR;
  using ::PFN_vkGetDisplayPlaneCapabilities2KHR;
  using ::PFN_vkGetPhysicalDeviceDisplayPlaneProperties2KHR;
  using ::PFN_vkGetPhysicalDeviceDisplayProperties2KHR;

#if defined( VK_USE_PLATFORM_IOS_MVK )
  //=== VK_MVK_ios_surface ===
  using ::PFN_vkCreateIOSSurfaceMVK;
#endif /*VK_USE_PLATFORM_IOS_MVK*/

#if defined( VK_USE_PLATFORM_MACOS_MVK )
  //=== VK_MVK_macos_surface ===
  using ::PFN_vkCreateMacOSSurfaceMVK;
#endif /*VK_USE_PLATFORM_MACOS_MVK*/

  //=== VK_EXT_debug_utils ===
  using ::PFN_vkCmdBeginDebugUtilsLabelEXT;
  using ::PFN_vkCmdEndDebugUtilsLabelEXT;
  using ::PFN_vkCmdInsertDebugUtilsLabelEXT;
  using ::PFN_vkCreateDebugUtilsMessengerEXT;
  using ::PFN_vkDestroyDebugUtilsMessengerEXT;
  using ::PFN_vkQueueBeginDebugUtilsLabelEXT;
  using ::PFN_vkQueueEndDebugUtilsLabelEXT;
  using ::PFN_vkQueueInsertDebugUtilsLabelEXT;
  using ::PFN_vkSetDebugUtilsObjectNameEXT;
  using ::PFN_vkSetDebugUtilsObjectTagEXT;
  using ::PFN_vkSubmitDebugUtilsMessageEXT;

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  //=== VK_ANDROID_external_memory_android_hardware_buffer ===
  using ::PFN_vkGetAndroidHardwareBufferPropertiesANDROID;
  using ::PFN_vkGetMemoryAndroidHardwareBufferANDROID;
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_AMDX_shader_enqueue ===
  using ::PFN_vkCmdDispatchGraphAMDX;
  using ::PFN_vkCmdDispatchGraphIndirectAMDX;
  using ::PFN_vkCmdDispatchGraphIndirectCountAMDX;
  using ::PFN_vkCmdInitializeGraphScratchMemoryAMDX;
  using ::PFN_vkCreateExecutionGraphPipelinesAMDX;
  using ::PFN_vkGetExecutionGraphPipelineNodeIndexAMDX;
  using ::PFN_vkGetExecutionGraphPipelineScratchSizeAMDX;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_EXT_sample_locations ===
  using ::PFN_vkCmdSetSampleLocationsEXT;
  using ::PFN_vkGetPhysicalDeviceMultisamplePropertiesEXT;

  //=== VK_KHR_get_memory_requirements2 ===
  using ::PFN_vkGetBufferMemoryRequirements2KHR;
  using ::PFN_vkGetImageMemoryRequirements2KHR;
  using ::PFN_vkGetImageSparseMemoryRequirements2KHR;

  //=== VK_KHR_acceleration_structure ===
  using ::PFN_vkBuildAccelerationStructuresKHR;
  using ::PFN_vkCmdBuildAccelerationStructuresIndirectKHR;
  using ::PFN_vkCmdBuildAccelerationStructuresKHR;
  using ::PFN_vkCmdCopyAccelerationStructureKHR;
  using ::PFN_vkCmdCopyAccelerationStructureToMemoryKHR;
  using ::PFN_vkCmdCopyMemoryToAccelerationStructureKHR;
  using ::PFN_vkCmdWriteAccelerationStructuresPropertiesKHR;
  using ::PFN_vkCopyAccelerationStructureKHR;
  using ::PFN_vkCopyAccelerationStructureToMemoryKHR;
  using ::PFN_vkCopyMemoryToAccelerationStructureKHR;
  using ::PFN_vkCreateAccelerationStructureKHR;
  using ::PFN_vkDestroyAccelerationStructureKHR;
  using ::PFN_vkGetAccelerationStructureBuildSizesKHR;
  using ::PFN_vkGetAccelerationStructureDeviceAddressKHR;
  using ::PFN_vkGetDeviceAccelerationStructureCompatibilityKHR;
  using ::PFN_vkWriteAccelerationStructuresPropertiesKHR;

  //=== VK_KHR_ray_tracing_pipeline ===
  using ::PFN_vkCmdSetRayTracingPipelineStackSizeKHR;
  using ::PFN_vkCmdTraceRaysIndirectKHR;
  using ::PFN_vkCmdTraceRaysKHR;
  using ::PFN_vkCreateRayTracingPipelinesKHR;
  using ::PFN_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR;
  using ::PFN_vkGetRayTracingShaderGroupHandlesKHR;
  using ::PFN_vkGetRayTracingShaderGroupStackSizeKHR;

  //=== VK_KHR_sampler_ycbcr_conversion ===
  using ::PFN_vkCreateSamplerYcbcrConversionKHR;
  using ::PFN_vkDestroySamplerYcbcrConversionKHR;

  //=== VK_KHR_bind_memory2 ===
  using ::PFN_vkBindBufferMemory2KHR;
  using ::PFN_vkBindImageMemory2KHR;

  //=== VK_EXT_image_drm_format_modifier ===
  using ::PFN_vkGetImageDrmFormatModifierPropertiesEXT;

  //=== VK_EXT_validation_cache ===
  using ::PFN_vkCreateValidationCacheEXT;
  using ::PFN_vkDestroyValidationCacheEXT;
  using ::PFN_vkGetValidationCacheDataEXT;
  using ::PFN_vkMergeValidationCachesEXT;

  //=== VK_NV_shading_rate_image ===
  using ::PFN_vkCmdBindShadingRateImageNV;
  using ::PFN_vkCmdSetCoarseSampleOrderNV;
  using ::PFN_vkCmdSetViewportShadingRatePaletteNV;

  //=== VK_NV_ray_tracing ===
  using ::PFN_vkBindAccelerationStructureMemoryNV;
  using ::PFN_vkCmdBuildAccelerationStructureNV;
  using ::PFN_vkCmdCopyAccelerationStructureNV;
  using ::PFN_vkCmdTraceRaysNV;
  using ::PFN_vkCmdWriteAccelerationStructuresPropertiesNV;
  using ::PFN_vkCompileDeferredNV;
  using ::PFN_vkCreateAccelerationStructureNV;
  using ::PFN_vkCreateRayTracingPipelinesNV;
  using ::PFN_vkDestroyAccelerationStructureNV;
  using ::PFN_vkGetAccelerationStructureHandleNV;
  using ::PFN_vkGetAccelerationStructureMemoryRequirementsNV;
  using ::PFN_vkGetRayTracingShaderGroupHandlesNV;

  //=== VK_KHR_maintenance3 ===
  using ::PFN_vkGetDescriptorSetLayoutSupportKHR;

  //=== VK_KHR_draw_indirect_count ===
  using ::PFN_vkCmdDrawIndexedIndirectCountKHR;
  using ::PFN_vkCmdDrawIndirectCountKHR;

  //=== VK_EXT_external_memory_host ===
  using ::PFN_vkGetMemoryHostPointerPropertiesEXT;

  //=== VK_AMD_buffer_marker ===
  using ::PFN_vkCmdWriteBufferMarker2AMD;
  using ::PFN_vkCmdWriteBufferMarkerAMD;

  //=== VK_EXT_calibrated_timestamps ===
  using ::PFN_vkGetCalibratedTimestampsEXT;
  using ::PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT;

  //=== VK_NV_mesh_shader ===
  using ::PFN_vkCmdDrawMeshTasksIndirectCountNV;
  using ::PFN_vkCmdDrawMeshTasksIndirectNV;
  using ::PFN_vkCmdDrawMeshTasksNV;

  //=== VK_NV_scissor_exclusive ===
  using ::PFN_vkCmdSetExclusiveScissorEnableNV;
  using ::PFN_vkCmdSetExclusiveScissorNV;

  //=== VK_NV_device_diagnostic_checkpoints ===
  using ::PFN_vkCmdSetCheckpointNV;
  using ::PFN_vkGetQueueCheckpointData2NV;
  using ::PFN_vkGetQueueCheckpointDataNV;

  //=== VK_KHR_timeline_semaphore ===
  using ::PFN_vkGetSemaphoreCounterValueKHR;
  using ::PFN_vkSignalSemaphoreKHR;
  using ::PFN_vkWaitSemaphoresKHR;

  //=== VK_EXT_present_timing ===
  using ::PFN_vkGetPastPresentationTimingEXT;
  using ::PFN_vkGetSwapchainTimeDomainPropertiesEXT;
  using ::PFN_vkGetSwapchainTimingPropertiesEXT;
  using ::PFN_vkSetSwapchainPresentTimingQueueSizeEXT;

  //=== VK_INTEL_performance_query ===
  using ::PFN_vkAcquirePerformanceConfigurationINTEL;
  using ::PFN_vkCmdSetPerformanceMarkerINTEL;
  using ::PFN_vkCmdSetPerformanceOverrideINTEL;
  using ::PFN_vkCmdSetPerformanceStreamMarkerINTEL;
  using ::PFN_vkGetPerformanceParameterINTEL;
  using ::PFN_vkInitializePerformanceApiINTEL;
  using ::PFN_vkQueueSetPerformanceConfigurationINTEL;
  using ::PFN_vkReleasePerformanceConfigurationINTEL;
  using ::PFN_vkUninitializePerformanceApiINTEL;

  //=== VK_AMD_display_native_hdr ===
  using ::PFN_vkSetLocalDimmingAMD;

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_imagepipe_surface ===
  using ::PFN_vkCreateImagePipeSurfaceFUCHSIA;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_surface ===
  using ::PFN_vkCreateMetalSurfaceEXT;
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_KHR_fragment_shading_rate ===
  using ::PFN_vkCmdSetFragmentShadingRateKHR;
  using ::PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR;

  //=== VK_KHR_dynamic_rendering_local_read ===
  using ::PFN_vkCmdSetRenderingAttachmentLocationsKHR;
  using ::PFN_vkCmdSetRenderingInputAttachmentIndicesKHR;

  //=== VK_EXT_buffer_device_address ===
  using ::PFN_vkGetBufferDeviceAddressEXT;

  //=== VK_EXT_tooling_info ===
  using ::PFN_vkGetPhysicalDeviceToolPropertiesEXT;

  //=== VK_KHR_present_wait ===
  using ::PFN_vkWaitForPresentKHR;

  //=== VK_NV_cooperative_matrix ===
  using ::PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV;

  //=== VK_NV_coverage_reduction_mode ===
  using ::PFN_vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_EXT_full_screen_exclusive ===
  using ::PFN_vkAcquireFullScreenExclusiveModeEXT;
  using ::PFN_vkGetDeviceGroupSurfacePresentModes2EXT;
  using ::PFN_vkGetPhysicalDeviceSurfacePresentModes2EXT;
  using ::PFN_vkReleaseFullScreenExclusiveModeEXT;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_EXT_headless_surface ===
  using ::PFN_vkCreateHeadlessSurfaceEXT;

  //=== VK_KHR_buffer_device_address ===
  using ::PFN_vkGetBufferDeviceAddressKHR;
  using ::PFN_vkGetBufferOpaqueCaptureAddressKHR;
  using ::PFN_vkGetDeviceMemoryOpaqueCaptureAddressKHR;

  //=== VK_EXT_line_rasterization ===
  using ::PFN_vkCmdSetLineStippleEXT;

  //=== VK_EXT_host_query_reset ===
  using ::PFN_vkResetQueryPoolEXT;

  //=== VK_EXT_extended_dynamic_state ===
  using ::PFN_vkCmdBindVertexBuffers2EXT;
  using ::PFN_vkCmdSetCullModeEXT;
  using ::PFN_vkCmdSetDepthBoundsTestEnableEXT;
  using ::PFN_vkCmdSetDepthCompareOpEXT;
  using ::PFN_vkCmdSetDepthTestEnableEXT;
  using ::PFN_vkCmdSetDepthWriteEnableEXT;
  using ::PFN_vkCmdSetFrontFaceEXT;
  using ::PFN_vkCmdSetPrimitiveTopologyEXT;
  using ::PFN_vkCmdSetScissorWithCountEXT;
  using ::PFN_vkCmdSetStencilOpEXT;
  using ::PFN_vkCmdSetStencilTestEnableEXT;
  using ::PFN_vkCmdSetViewportWithCountEXT;

  //=== VK_KHR_deferred_host_operations ===
  using ::PFN_vkCreateDeferredOperationKHR;
  using ::PFN_vkDeferredOperationJoinKHR;
  using ::PFN_vkDestroyDeferredOperationKHR;
  using ::PFN_vkGetDeferredOperationMaxConcurrencyKHR;
  using ::PFN_vkGetDeferredOperationResultKHR;

  //=== VK_KHR_pipeline_executable_properties ===
  using ::PFN_vkGetPipelineExecutableInternalRepresentationsKHR;
  using ::PFN_vkGetPipelineExecutablePropertiesKHR;
  using ::PFN_vkGetPipelineExecutableStatisticsKHR;

  //=== VK_EXT_host_image_copy ===
  using ::PFN_vkCopyImageToImageEXT;
  using ::PFN_vkCopyImageToMemoryEXT;
  using ::PFN_vkCopyMemoryToImageEXT;
  using ::PFN_vkGetImageSubresourceLayout2EXT;
  using ::PFN_vkTransitionImageLayoutEXT;

  //=== VK_KHR_map_memory2 ===
  using ::PFN_vkMapMemory2KHR;
  using ::PFN_vkUnmapMemory2KHR;

  //=== VK_EXT_swapchain_maintenance1 ===
  using ::PFN_vkReleaseSwapchainImagesEXT;

  //=== VK_NV_device_generated_commands ===
  using ::PFN_vkCmdBindPipelineShaderGroupNV;
  using ::PFN_vkCmdExecuteGeneratedCommandsNV;
  using ::PFN_vkCmdPreprocessGeneratedCommandsNV;
  using ::PFN_vkCreateIndirectCommandsLayoutNV;
  using ::PFN_vkDestroyIndirectCommandsLayoutNV;
  using ::PFN_vkGetGeneratedCommandsMemoryRequirementsNV;

  //=== VK_EXT_depth_bias_control ===
  using ::PFN_vkCmdSetDepthBias2EXT;

  //=== VK_EXT_acquire_drm_display ===
  using ::PFN_vkAcquireDrmDisplayEXT;
  using ::PFN_vkGetDrmDisplayEXT;

  //=== VK_EXT_private_data ===
  using ::PFN_vkCreatePrivateDataSlotEXT;
  using ::PFN_vkDestroyPrivateDataSlotEXT;
  using ::PFN_vkGetPrivateDataEXT;
  using ::PFN_vkSetPrivateDataEXT;

  //=== VK_KHR_video_encode_queue ===
  using ::PFN_vkCmdEncodeVideoKHR;
  using ::PFN_vkGetEncodedVideoSessionParametersKHR;
  using ::PFN_vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_NV_cuda_kernel_launch ===
  using ::PFN_vkCmdCudaLaunchKernelNV;
  using ::PFN_vkCreateCudaFunctionNV;
  using ::PFN_vkCreateCudaModuleNV;
  using ::PFN_vkDestroyCudaFunctionNV;
  using ::PFN_vkDestroyCudaModuleNV;
  using ::PFN_vkGetCudaModuleCacheNV;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_QCOM_tile_shading ===
  using ::PFN_vkCmdBeginPerTileExecutionQCOM;
  using ::PFN_vkCmdDispatchTileQCOM;
  using ::PFN_vkCmdEndPerTileExecutionQCOM;

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_objects ===
  using ::PFN_vkExportMetalObjectsEXT;
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_KHR_synchronization2 ===
  using ::PFN_vkCmdPipelineBarrier2KHR;
  using ::PFN_vkCmdResetEvent2KHR;
  using ::PFN_vkCmdSetEvent2KHR;
  using ::PFN_vkCmdWaitEvents2KHR;
  using ::PFN_vkCmdWriteTimestamp2KHR;
  using ::PFN_vkQueueSubmit2KHR;

  //=== VK_EXT_descriptor_buffer ===
  using ::PFN_vkCmdBindDescriptorBufferEmbeddedSamplersEXT;
  using ::PFN_vkCmdBindDescriptorBuffersEXT;
  using ::PFN_vkCmdSetDescriptorBufferOffsetsEXT;
  using ::PFN_vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT;
  using ::PFN_vkGetBufferOpaqueCaptureDescriptorDataEXT;
  using ::PFN_vkGetDescriptorEXT;
  using ::PFN_vkGetDescriptorSetLayoutBindingOffsetEXT;
  using ::PFN_vkGetDescriptorSetLayoutSizeEXT;
  using ::PFN_vkGetImageOpaqueCaptureDescriptorDataEXT;
  using ::PFN_vkGetImageViewOpaqueCaptureDescriptorDataEXT;
  using ::PFN_vkGetSamplerOpaqueCaptureDescriptorDataEXT;

  //=== VK_NV_fragment_shading_rate_enums ===
  using ::PFN_vkCmdSetFragmentShadingRateEnumNV;

  //=== VK_EXT_mesh_shader ===
  using ::PFN_vkCmdDrawMeshTasksEXT;
  using ::PFN_vkCmdDrawMeshTasksIndirectCountEXT;
  using ::PFN_vkCmdDrawMeshTasksIndirectEXT;

  //=== VK_KHR_copy_commands2 ===
  using ::PFN_vkCmdBlitImage2KHR;
  using ::PFN_vkCmdCopyBuffer2KHR;
  using ::PFN_vkCmdCopyBufferToImage2KHR;
  using ::PFN_vkCmdCopyImage2KHR;
  using ::PFN_vkCmdCopyImageToBuffer2KHR;
  using ::PFN_vkCmdResolveImage2KHR;

  //=== VK_EXT_device_fault ===
  using ::PFN_vkGetDeviceFaultInfoEXT;

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_NV_acquire_winrt_display ===
  using ::PFN_vkAcquireWinrtDisplayNV;
  using ::PFN_vkGetWinrtDisplayNV;
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
  //=== VK_EXT_directfb_surface ===
  using ::PFN_vkCreateDirectFBSurfaceEXT;
  using ::PFN_vkGetPhysicalDeviceDirectFBPresentationSupportEXT;
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

  //=== VK_EXT_vertex_input_dynamic_state ===
  using ::PFN_vkCmdSetVertexInputEXT;

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_external_memory ===
  using ::PFN_vkGetMemoryZirconHandleFUCHSIA;
  using ::PFN_vkGetMemoryZirconHandlePropertiesFUCHSIA;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_external_semaphore ===
  using ::PFN_vkGetSemaphoreZirconHandleFUCHSIA;
  using ::PFN_vkImportSemaphoreZirconHandleFUCHSIA;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_buffer_collection ===
  using ::PFN_vkCreateBufferCollectionFUCHSIA;
  using ::PFN_vkDestroyBufferCollectionFUCHSIA;
  using ::PFN_vkGetBufferCollectionPropertiesFUCHSIA;
  using ::PFN_vkSetBufferCollectionBufferConstraintsFUCHSIA;
  using ::PFN_vkSetBufferCollectionImageConstraintsFUCHSIA;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

  //=== VK_HUAWEI_subpass_shading ===
  using ::PFN_vkCmdSubpassShadingHUAWEI;
  using ::PFN_vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI;

  //=== VK_HUAWEI_invocation_mask ===
  using ::PFN_vkCmdBindInvocationMaskHUAWEI;

  //=== VK_NV_external_memory_rdma ===
  using ::PFN_vkGetMemoryRemoteAddressNV;

  //=== VK_EXT_pipeline_properties ===
  using ::PFN_vkGetPipelinePropertiesEXT;

  //=== VK_EXT_extended_dynamic_state2 ===
  using ::PFN_vkCmdSetDepthBiasEnableEXT;
  using ::PFN_vkCmdSetLogicOpEXT;
  using ::PFN_vkCmdSetPatchControlPointsEXT;
  using ::PFN_vkCmdSetPrimitiveRestartEnableEXT;
  using ::PFN_vkCmdSetRasterizerDiscardEnableEXT;

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
  //=== VK_QNX_screen_surface ===
  using ::PFN_vkCreateScreenSurfaceQNX;
  using ::PFN_vkGetPhysicalDeviceScreenPresentationSupportQNX;
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

  //=== VK_EXT_color_write_enable ===
  using ::PFN_vkCmdSetColorWriteEnableEXT;

  //=== VK_KHR_ray_tracing_maintenance1 ===
  using ::PFN_vkCmdTraceRaysIndirect2KHR;

  //=== VK_EXT_multi_draw ===
  using ::PFN_vkCmdDrawMultiEXT;
  using ::PFN_vkCmdDrawMultiIndexedEXT;

  //=== VK_EXT_opacity_micromap ===
  using ::PFN_vkBuildMicromapsEXT;
  using ::PFN_vkCmdBuildMicromapsEXT;
  using ::PFN_vkCmdCopyMemoryToMicromapEXT;
  using ::PFN_vkCmdCopyMicromapEXT;
  using ::PFN_vkCmdCopyMicromapToMemoryEXT;
  using ::PFN_vkCmdWriteMicromapsPropertiesEXT;
  using ::PFN_vkCopyMemoryToMicromapEXT;
  using ::PFN_vkCopyMicromapEXT;
  using ::PFN_vkCopyMicromapToMemoryEXT;
  using ::PFN_vkCreateMicromapEXT;
  using ::PFN_vkDestroyMicromapEXT;
  using ::PFN_vkGetDeviceMicromapCompatibilityEXT;
  using ::PFN_vkGetMicromapBuildSizesEXT;
  using ::PFN_vkWriteMicromapsPropertiesEXT;

  //=== VK_HUAWEI_cluster_culling_shader ===
  using ::PFN_vkCmdDrawClusterHUAWEI;
  using ::PFN_vkCmdDrawClusterIndirectHUAWEI;

  //=== VK_EXT_pageable_device_local_memory ===
  using ::PFN_vkSetDeviceMemoryPriorityEXT;

  //=== VK_KHR_maintenance4 ===
  using ::PFN_vkGetDeviceBufferMemoryRequirementsKHR;
  using ::PFN_vkGetDeviceImageMemoryRequirementsKHR;
  using ::PFN_vkGetDeviceImageSparseMemoryRequirementsKHR;

  //=== VK_VALVE_descriptor_set_host_mapping ===
  using ::PFN_vkGetDescriptorSetHostMappingVALVE;
  using ::PFN_vkGetDescriptorSetLayoutHostMappingInfoVALVE;

  //=== VK_NV_copy_memory_indirect ===
  using ::PFN_vkCmdCopyMemoryIndirectNV;
  using ::PFN_vkCmdCopyMemoryToImageIndirectNV;

  //=== VK_NV_memory_decompression ===
  using ::PFN_vkCmdDecompressMemoryIndirectCountNV;
  using ::PFN_vkCmdDecompressMemoryNV;

  //=== VK_NV_device_generated_commands_compute ===
  using ::PFN_vkCmdUpdatePipelineIndirectBufferNV;
  using ::PFN_vkGetPipelineIndirectDeviceAddressNV;
  using ::PFN_vkGetPipelineIndirectMemoryRequirementsNV;

#if defined( VK_USE_PLATFORM_OHOS )
  //=== VK_OHOS_external_memory ===
  using ::PFN_vkGetMemoryNativeBufferOHOS;
  using ::PFN_vkGetNativeBufferPropertiesOHOS;
#endif /*VK_USE_PLATFORM_OHOS*/

  //=== VK_EXT_extended_dynamic_state3 ===
  using ::PFN_vkCmdSetAlphaToCoverageEnableEXT;
  using ::PFN_vkCmdSetAlphaToOneEnableEXT;
  using ::PFN_vkCmdSetColorBlendAdvancedEXT;
  using ::PFN_vkCmdSetColorBlendEnableEXT;
  using ::PFN_vkCmdSetColorBlendEquationEXT;
  using ::PFN_vkCmdSetColorWriteMaskEXT;
  using ::PFN_vkCmdSetConservativeRasterizationModeEXT;
  using ::PFN_vkCmdSetCoverageModulationModeNV;
  using ::PFN_vkCmdSetCoverageModulationTableEnableNV;
  using ::PFN_vkCmdSetCoverageModulationTableNV;
  using ::PFN_vkCmdSetCoverageReductionModeNV;
  using ::PFN_vkCmdSetCoverageToColorEnableNV;
  using ::PFN_vkCmdSetCoverageToColorLocationNV;
  using ::PFN_vkCmdSetDepthClampEnableEXT;
  using ::PFN_vkCmdSetDepthClipEnableEXT;
  using ::PFN_vkCmdSetDepthClipNegativeOneToOneEXT;
  using ::PFN_vkCmdSetExtraPrimitiveOverestimationSizeEXT;
  using ::PFN_vkCmdSetLineRasterizationModeEXT;
  using ::PFN_vkCmdSetLineStippleEnableEXT;
  using ::PFN_vkCmdSetLogicOpEnableEXT;
  using ::PFN_vkCmdSetPolygonModeEXT;
  using ::PFN_vkCmdSetProvokingVertexModeEXT;
  using ::PFN_vkCmdSetRasterizationSamplesEXT;
  using ::PFN_vkCmdSetRasterizationStreamEXT;
  using ::PFN_vkCmdSetRepresentativeFragmentTestEnableNV;
  using ::PFN_vkCmdSetSampleLocationsEnableEXT;
  using ::PFN_vkCmdSetSampleMaskEXT;
  using ::PFN_vkCmdSetShadingRateImageEnableNV;
  using ::PFN_vkCmdSetTessellationDomainOriginEXT;
  using ::PFN_vkCmdSetViewportSwizzleNV;
  using ::PFN_vkCmdSetViewportWScalingEnableNV;

  //=== VK_ARM_tensors ===
  using ::PFN_vkBindTensorMemoryARM;
  using ::PFN_vkCmdCopyTensorARM;
  using ::PFN_vkCreateTensorARM;
  using ::PFN_vkCreateTensorViewARM;
  using ::PFN_vkDestroyTensorARM;
  using ::PFN_vkDestroyTensorViewARM;
  using ::PFN_vkGetDeviceTensorMemoryRequirementsARM;
  using ::PFN_vkGetPhysicalDeviceExternalTensorPropertiesARM;
  using ::PFN_vkGetTensorMemoryRequirementsARM;
  using ::PFN_vkGetTensorOpaqueCaptureDescriptorDataARM;
  using ::PFN_vkGetTensorViewOpaqueCaptureDescriptorDataARM;

  //=== VK_EXT_shader_module_identifier ===
  using ::PFN_vkGetShaderModuleCreateInfoIdentifierEXT;
  using ::PFN_vkGetShaderModuleIdentifierEXT;

  //=== VK_NV_optical_flow ===
  using ::PFN_vkBindOpticalFlowSessionImageNV;
  using ::PFN_vkCmdOpticalFlowExecuteNV;
  using ::PFN_vkCreateOpticalFlowSessionNV;
  using ::PFN_vkDestroyOpticalFlowSessionNV;
  using ::PFN_vkGetPhysicalDeviceOpticalFlowImageFormatsNV;

  //=== VK_KHR_maintenance5 ===
  using ::PFN_vkCmdBindIndexBuffer2KHR;
  using ::PFN_vkGetDeviceImageSubresourceLayoutKHR;
  using ::PFN_vkGetImageSubresourceLayout2KHR;
  using ::PFN_vkGetRenderingAreaGranularityKHR;

  //=== VK_AMD_anti_lag ===
  using ::PFN_vkAntiLagUpdateAMD;

  //=== VK_KHR_present_wait2 ===
  using ::PFN_vkWaitForPresent2KHR;

  //=== VK_EXT_shader_object ===
  using ::PFN_vkCmdBindShadersEXT;
  using ::PFN_vkCmdSetDepthClampRangeEXT;
  using ::PFN_vkCreateShadersEXT;
  using ::PFN_vkDestroyShaderEXT;
  using ::PFN_vkGetShaderBinaryDataEXT;

  //=== VK_KHR_pipeline_binary ===
  using ::PFN_vkCreatePipelineBinariesKHR;
  using ::PFN_vkDestroyPipelineBinaryKHR;
  using ::PFN_vkGetPipelineBinaryDataKHR;
  using ::PFN_vkGetPipelineKeyKHR;
  using ::PFN_vkReleaseCapturedPipelineDataKHR;

  //=== VK_QCOM_tile_properties ===
  using ::PFN_vkGetDynamicRenderingTilePropertiesQCOM;
  using ::PFN_vkGetFramebufferTilePropertiesQCOM;

  //=== VK_KHR_swapchain_maintenance1 ===
  using ::PFN_vkReleaseSwapchainImagesKHR;

  //=== VK_NV_cooperative_vector ===
  using ::PFN_vkCmdConvertCooperativeVectorMatrixNV;
  using ::PFN_vkConvertCooperativeVectorMatrixNV;
  using ::PFN_vkGetPhysicalDeviceCooperativeVectorPropertiesNV;

  //=== VK_NV_low_latency2 ===
  using ::PFN_vkGetLatencyTimingsNV;
  using ::PFN_vkLatencySleepNV;
  using ::PFN_vkQueueNotifyOutOfBandNV;
  using ::PFN_vkSetLatencyMarkerNV;
  using ::PFN_vkSetLatencySleepModeNV;

  //=== VK_KHR_cooperative_matrix ===
  using ::PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR;

  //=== VK_ARM_data_graph ===
  using ::PFN_vkBindDataGraphPipelineSessionMemoryARM;
  using ::PFN_vkCmdDispatchDataGraphARM;
  using ::PFN_vkCreateDataGraphPipelinesARM;
  using ::PFN_vkCreateDataGraphPipelineSessionARM;
  using ::PFN_vkDestroyDataGraphPipelineSessionARM;
  using ::PFN_vkGetDataGraphPipelineAvailablePropertiesARM;
  using ::PFN_vkGetDataGraphPipelinePropertiesARM;
  using ::PFN_vkGetDataGraphPipelineSessionBindPointRequirementsARM;
  using ::PFN_vkGetDataGraphPipelineSessionMemoryRequirementsARM;
  using ::PFN_vkGetPhysicalDeviceQueueFamilyDataGraphProcessingEnginePropertiesARM;
  using ::PFN_vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM;

  //=== VK_EXT_attachment_feedback_loop_dynamic_state ===
  using ::PFN_vkCmdSetAttachmentFeedbackLoopEnableEXT;

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
  //=== VK_QNX_external_memory_screen_buffer ===
  using ::PFN_vkGetScreenBufferPropertiesQNX;
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

  //=== VK_KHR_line_rasterization ===
  using ::PFN_vkCmdSetLineStippleKHR;

  //=== VK_KHR_calibrated_timestamps ===
  using ::PFN_vkGetCalibratedTimestampsKHR;
  using ::PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR;

  //=== VK_KHR_maintenance6 ===
  using ::PFN_vkCmdBindDescriptorBufferEmbeddedSamplers2EXT;
  using ::PFN_vkCmdBindDescriptorSets2KHR;
  using ::PFN_vkCmdPushConstants2KHR;
  using ::PFN_vkCmdPushDescriptorSet2KHR;
  using ::PFN_vkCmdPushDescriptorSetWithTemplate2KHR;
  using ::PFN_vkCmdSetDescriptorBufferOffsets2EXT;

  //=== VK_QCOM_tile_memory_heap ===
  using ::PFN_vkCmdBindTileMemoryQCOM;

  //=== VK_KHR_copy_memory_indirect ===
  using ::PFN_vkCmdCopyMemoryIndirectKHR;
  using ::PFN_vkCmdCopyMemoryToImageIndirectKHR;

  //=== VK_EXT_memory_decompression ===
  using ::PFN_vkCmdDecompressMemoryEXT;
  using ::PFN_vkCmdDecompressMemoryIndirectCountEXT;

  //=== VK_NV_external_compute_queue ===
  using ::PFN_vkCreateExternalComputeQueueNV;
  using ::PFN_vkDestroyExternalComputeQueueNV;
  using ::PFN_vkGetExternalComputeQueueDataNV;

  //=== VK_NV_cluster_acceleration_structure ===
  using ::PFN_vkCmdBuildClusterAccelerationStructureIndirectNV;
  using ::PFN_vkGetClusterAccelerationStructureBuildSizesNV;

  //=== VK_NV_partitioned_acceleration_structure ===
  using ::PFN_vkCmdBuildPartitionedAccelerationStructuresNV;
  using ::PFN_vkGetPartitionedAccelerationStructuresBuildSizesNV;

  //=== VK_EXT_device_generated_commands ===
  using ::PFN_vkCmdExecuteGeneratedCommandsEXT;
  using ::PFN_vkCmdPreprocessGeneratedCommandsEXT;
  using ::PFN_vkCreateIndirectCommandsLayoutEXT;
  using ::PFN_vkCreateIndirectExecutionSetEXT;
  using ::PFN_vkDestroyIndirectCommandsLayoutEXT;
  using ::PFN_vkDestroyIndirectExecutionSetEXT;
  using ::PFN_vkGetGeneratedCommandsMemoryRequirementsEXT;
  using ::PFN_vkUpdateIndirectExecutionSetPipelineEXT;
  using ::PFN_vkUpdateIndirectExecutionSetShaderEXT;

#if defined( VK_USE_PLATFORM_OHOS )
  //=== VK_OHOS_surface ===
  using ::PFN_vkCreateSurfaceOHOS;
#endif /*VK_USE_PLATFORM_OHOS*/

#if defined( VK_USE_PLATFORM_OHOS )
  //=== VK_OHOS_native_buffer ===
  using ::PFN_vkAcquireImageOHOS;
  using ::PFN_vkGetSwapchainGrallocUsageOHOS;
  using ::PFN_vkQueueSignalReleaseImageOHOS;
#endif /*VK_USE_PLATFORM_OHOS*/

  //=== VK_NV_cooperative_matrix2 ===
  using ::PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV;

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_external_memory_metal ===
  using ::PFN_vkGetMemoryMetalHandleEXT;
  using ::PFN_vkGetMemoryMetalHandlePropertiesEXT;
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_ARM_performance_counters_by_region ===
  using ::PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceCountersByRegionARM;

  //=== VK_EXT_fragment_density_map_offset ===
  using ::PFN_vkCmdEndRendering2EXT;

  //=== VK_EXT_custom_resolve ===
  using ::PFN_vkCmdBeginCustomResolveEXT;

  //=== VK_KHR_maintenance10 ===
  using ::PFN_vkCmdEndRendering2KHR;
}
