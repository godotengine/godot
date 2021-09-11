// Copyright 2015-2021 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//

// This header is generated from the Khronos Vulkan XML API Registry.

#ifndef VULKAN_ENUMS_HPP
#define VULKAN_ENUMS_HPP

namespace VULKAN_HPP_NAMESPACE
{
  template <typename EnumType, EnumType value>
  struct CppType
  {};

  template <typename Type>
  struct isVulkanHandleType
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool value = false;
  };

  VULKAN_HPP_INLINE std::string toHexString( uint32_t value )
  {
    std::stringstream stream;
    stream << std::hex << value;
    return stream.str();
  }

  //=============
  //=== ENUMs ===
  //=============

  //=== VK_VERSION_1_0 ===

  enum class Result
  {
    eSuccess                                     = VK_SUCCESS,
    eNotReady                                    = VK_NOT_READY,
    eTimeout                                     = VK_TIMEOUT,
    eEventSet                                    = VK_EVENT_SET,
    eEventReset                                  = VK_EVENT_RESET,
    eIncomplete                                  = VK_INCOMPLETE,
    eErrorOutOfHostMemory                        = VK_ERROR_OUT_OF_HOST_MEMORY,
    eErrorOutOfDeviceMemory                      = VK_ERROR_OUT_OF_DEVICE_MEMORY,
    eErrorInitializationFailed                   = VK_ERROR_INITIALIZATION_FAILED,
    eErrorDeviceLost                             = VK_ERROR_DEVICE_LOST,
    eErrorMemoryMapFailed                        = VK_ERROR_MEMORY_MAP_FAILED,
    eErrorLayerNotPresent                        = VK_ERROR_LAYER_NOT_PRESENT,
    eErrorExtensionNotPresent                    = VK_ERROR_EXTENSION_NOT_PRESENT,
    eErrorFeatureNotPresent                      = VK_ERROR_FEATURE_NOT_PRESENT,
    eErrorIncompatibleDriver                     = VK_ERROR_INCOMPATIBLE_DRIVER,
    eErrorTooManyObjects                         = VK_ERROR_TOO_MANY_OBJECTS,
    eErrorFormatNotSupported                     = VK_ERROR_FORMAT_NOT_SUPPORTED,
    eErrorFragmentedPool                         = VK_ERROR_FRAGMENTED_POOL,
    eErrorUnknown                                = VK_ERROR_UNKNOWN,
    eErrorOutOfPoolMemory                        = VK_ERROR_OUT_OF_POOL_MEMORY,
    eErrorInvalidExternalHandle                  = VK_ERROR_INVALID_EXTERNAL_HANDLE,
    eErrorFragmentation                          = VK_ERROR_FRAGMENTATION,
    eErrorInvalidOpaqueCaptureAddress            = VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS,
    eErrorSurfaceLostKHR                         = VK_ERROR_SURFACE_LOST_KHR,
    eErrorNativeWindowInUseKHR                   = VK_ERROR_NATIVE_WINDOW_IN_USE_KHR,
    eSuboptimalKHR                               = VK_SUBOPTIMAL_KHR,
    eErrorOutOfDateKHR                           = VK_ERROR_OUT_OF_DATE_KHR,
    eErrorIncompatibleDisplayKHR                 = VK_ERROR_INCOMPATIBLE_DISPLAY_KHR,
    eErrorValidationFailedEXT                    = VK_ERROR_VALIDATION_FAILED_EXT,
    eErrorInvalidShaderNV                        = VK_ERROR_INVALID_SHADER_NV,
    eErrorInvalidDrmFormatModifierPlaneLayoutEXT = VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT,
    eErrorNotPermittedEXT                        = VK_ERROR_NOT_PERMITTED_EXT,
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    eErrorFullScreenExclusiveModeLostEXT = VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT,
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    eThreadIdleKHR                       = VK_THREAD_IDLE_KHR,
    eThreadDoneKHR                       = VK_THREAD_DONE_KHR,
    eOperationDeferredKHR                = VK_OPERATION_DEFERRED_KHR,
    eOperationNotDeferredKHR             = VK_OPERATION_NOT_DEFERRED_KHR,
    ePipelineCompileRequiredEXT          = VK_PIPELINE_COMPILE_REQUIRED_EXT,
    eErrorFragmentationEXT               = VK_ERROR_FRAGMENTATION_EXT,
    eErrorInvalidDeviceAddressEXT        = VK_ERROR_INVALID_DEVICE_ADDRESS_EXT,
    eErrorInvalidExternalHandleKHR       = VK_ERROR_INVALID_EXTERNAL_HANDLE_KHR,
    eErrorInvalidOpaqueCaptureAddressKHR = VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR,
    eErrorOutOfPoolMemoryKHR             = VK_ERROR_OUT_OF_POOL_MEMORY_KHR,
    eErrorPipelineCompileRequiredEXT     = VK_ERROR_PIPELINE_COMPILE_REQUIRED_EXT
  };

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
      case Result::eErrorSurfaceLostKHR: return "ErrorSurfaceLostKHR";
      case Result::eErrorNativeWindowInUseKHR: return "ErrorNativeWindowInUseKHR";
      case Result::eSuboptimalKHR: return "SuboptimalKHR";
      case Result::eErrorOutOfDateKHR: return "ErrorOutOfDateKHR";
      case Result::eErrorIncompatibleDisplayKHR: return "ErrorIncompatibleDisplayKHR";
      case Result::eErrorValidationFailedEXT: return "ErrorValidationFailedEXT";
      case Result::eErrorInvalidShaderNV: return "ErrorInvalidShaderNV";
      case Result::eErrorInvalidDrmFormatModifierPlaneLayoutEXT: return "ErrorInvalidDrmFormatModifierPlaneLayoutEXT";
      case Result::eErrorNotPermittedEXT: return "ErrorNotPermittedEXT";
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      case Result::eErrorFullScreenExclusiveModeLostEXT: return "ErrorFullScreenExclusiveModeLostEXT";
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      case Result::eThreadIdleKHR: return "ThreadIdleKHR";
      case Result::eThreadDoneKHR: return "ThreadDoneKHR";
      case Result::eOperationDeferredKHR: return "OperationDeferredKHR";
      case Result::eOperationNotDeferredKHR: return "OperationNotDeferredKHR";
      case Result::ePipelineCompileRequiredEXT: return "PipelineCompileRequiredEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class StructureType
  {
    eApplicationInfo                           = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    eInstanceCreateInfo                        = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    eDeviceQueueCreateInfo                     = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    eDeviceCreateInfo                          = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    eSubmitInfo                                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    eMemoryAllocateInfo                        = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    eMappedMemoryRange                         = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
    eBindSparseInfo                            = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
    eFenceCreateInfo                           = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    eSemaphoreCreateInfo                       = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    eEventCreateInfo                           = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO,
    eQueryPoolCreateInfo                       = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
    eBufferCreateInfo                          = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    eBufferViewCreateInfo                      = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
    eImageCreateInfo                           = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
    eImageViewCreateInfo                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
    eShaderModuleCreateInfo                    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    ePipelineCacheCreateInfo                   = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
    ePipelineShaderStageCreateInfo             = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    ePipelineVertexInputStateCreateInfo        = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    ePipelineInputAssemblyStateCreateInfo      = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
    ePipelineTessellationStateCreateInfo       = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
    ePipelineViewportStateCreateInfo           = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
    ePipelineRasterizationStateCreateInfo      = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
    ePipelineMultisampleStateCreateInfo        = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
    ePipelineDepthStencilStateCreateInfo       = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
    ePipelineColorBlendStateCreateInfo         = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
    ePipelineDynamicStateCreateInfo            = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
    eGraphicsPipelineCreateInfo                = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
    eComputePipelineCreateInfo                 = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    ePipelineLayoutCreateInfo                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    eSamplerCreateInfo                         = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
    eDescriptorSetLayoutCreateInfo             = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    eDescriptorPoolCreateInfo                  = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    eDescriptorSetAllocateInfo                 = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    eWriteDescriptorSet                        = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    eCopyDescriptorSet                         = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET,
    eFramebufferCreateInfo                     = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
    eRenderPassCreateInfo                      = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
    eCommandPoolCreateInfo                     = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    eCommandBufferAllocateInfo                 = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    eCommandBufferInheritanceInfo              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
    eCommandBufferBeginInfo                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    eRenderPassBeginInfo                       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
    eBufferMemoryBarrier                       = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
    eImageMemoryBarrier                        = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
    eMemoryBarrier                             = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    eLoaderInstanceCreateInfo                  = VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO,
    eLoaderDeviceCreateInfo                    = VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO,
    ePhysicalDeviceSubgroupProperties          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    eBindBufferMemoryInfo                      = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
    eBindImageMemoryInfo                       = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,
    ePhysicalDevice16BitStorageFeatures        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
    eMemoryDedicatedRequirements               = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
    eMemoryDedicatedAllocateInfo               = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
    eMemoryAllocateFlagsInfo                   = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
    eDeviceGroupRenderPassBeginInfo            = VK_STRUCTURE_TYPE_DEVICE_GROUP_RENDER_PASS_BEGIN_INFO,
    eDeviceGroupCommandBufferBeginInfo         = VK_STRUCTURE_TYPE_DEVICE_GROUP_COMMAND_BUFFER_BEGIN_INFO,
    eDeviceGroupSubmitInfo                     = VK_STRUCTURE_TYPE_DEVICE_GROUP_SUBMIT_INFO,
    eDeviceGroupBindSparseInfo                 = VK_STRUCTURE_TYPE_DEVICE_GROUP_BIND_SPARSE_INFO,
    eBindBufferMemoryDeviceGroupInfo           = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_DEVICE_GROUP_INFO,
    eBindImageMemoryDeviceGroupInfo            = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_DEVICE_GROUP_INFO,
    ePhysicalDeviceGroupProperties             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES,
    eDeviceGroupDeviceCreateInfo               = VK_STRUCTURE_TYPE_DEVICE_GROUP_DEVICE_CREATE_INFO,
    eBufferMemoryRequirementsInfo2             = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
    eImageMemoryRequirementsInfo2              = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
    eImageSparseMemoryRequirementsInfo2        = VK_STRUCTURE_TYPE_IMAGE_SPARSE_MEMORY_REQUIREMENTS_INFO_2,
    eMemoryRequirements2                       = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
    eSparseImageMemoryRequirements2            = VK_STRUCTURE_TYPE_SPARSE_IMAGE_MEMORY_REQUIREMENTS_2,
    ePhysicalDeviceFeatures2                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
    ePhysicalDeviceProperties2                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
    eFormatProperties2                         = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
    eImageFormatProperties2                    = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
    ePhysicalDeviceImageFormatInfo2            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
    eQueueFamilyProperties2                    = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2,
    ePhysicalDeviceMemoryProperties2           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
    eSparseImageFormatProperties2              = VK_STRUCTURE_TYPE_SPARSE_IMAGE_FORMAT_PROPERTIES_2,
    ePhysicalDeviceSparseImageFormatInfo2      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SPARSE_IMAGE_FORMAT_INFO_2,
    ePhysicalDevicePointClippingProperties     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_POINT_CLIPPING_PROPERTIES,
    eRenderPassInputAttachmentAspectCreateInfo = VK_STRUCTURE_TYPE_RENDER_PASS_INPUT_ATTACHMENT_ASPECT_CREATE_INFO,
    eImageViewUsageCreateInfo                  = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
    ePipelineTessellationDomainOriginStateCreateInfo =
      VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_DOMAIN_ORIGIN_STATE_CREATE_INFO,
    eRenderPassMultiviewCreateInfo                = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO,
    ePhysicalDeviceMultiviewFeatures              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES,
    ePhysicalDeviceMultiviewProperties            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES,
    ePhysicalDeviceVariablePointersFeatures       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES,
    eProtectedSubmitInfo                          = VK_STRUCTURE_TYPE_PROTECTED_SUBMIT_INFO,
    ePhysicalDeviceProtectedMemoryFeatures        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES,
    ePhysicalDeviceProtectedMemoryProperties      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_PROPERTIES,
    eDeviceQueueInfo2                             = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2,
    eSamplerYcbcrConversionCreateInfo             = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
    eSamplerYcbcrConversionInfo                   = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO,
    eBindImagePlaneMemoryInfo                     = VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO,
    eImagePlaneMemoryRequirementsInfo             = VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO,
    ePhysicalDeviceSamplerYcbcrConversionFeatures = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES,
    eSamplerYcbcrConversionImageFormatProperties  = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES,
    eDescriptorUpdateTemplateCreateInfo           = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO,
    ePhysicalDeviceExternalImageFormatInfo        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
    eExternalImageFormatProperties                = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES,
    ePhysicalDeviceExternalBufferInfo             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
    eExternalBufferProperties                     = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES,
    ePhysicalDeviceIdProperties                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
    eExternalMemoryBufferCreateInfo               = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
    eExternalMemoryImageCreateInfo                = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
    eExportMemoryAllocateInfo                     = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
    ePhysicalDeviceExternalFenceInfo              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_FENCE_INFO,
    eExternalFenceProperties                      = VK_STRUCTURE_TYPE_EXTERNAL_FENCE_PROPERTIES,
    eExportFenceCreateInfo                        = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO,
    eExportSemaphoreCreateInfo                    = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
    ePhysicalDeviceExternalSemaphoreInfo          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
    eExternalSemaphoreProperties                  = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES,
    ePhysicalDeviceMaintenance3Properties         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES,
    eDescriptorSetLayoutSupport                   = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT,
    ePhysicalDeviceShaderDrawParametersFeatures   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES,
    ePhysicalDeviceVulkan11Features               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
    ePhysicalDeviceVulkan11Properties             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES,
    ePhysicalDeviceVulkan12Features               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
    ePhysicalDeviceVulkan12Properties             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES,
    eImageFormatListCreateInfo                    = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
    eAttachmentDescription2                       = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
    eAttachmentReference2                         = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2,
    eSubpassDescription2                          = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2,
    eSubpassDependency2                           = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2,
    eRenderPassCreateInfo2                        = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2,
    eSubpassBeginInfo                             = VK_STRUCTURE_TYPE_SUBPASS_BEGIN_INFO,
    eSubpassEndInfo                               = VK_STRUCTURE_TYPE_SUBPASS_END_INFO,
    ePhysicalDevice8BitStorageFeatures            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
    ePhysicalDeviceDriverProperties               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
    ePhysicalDeviceShaderAtomicInt64Features      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES,
    ePhysicalDeviceShaderFloat16Int8Features      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
    ePhysicalDeviceFloatControlsProperties        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES,
    eDescriptorSetLayoutBindingFlagsCreateInfo    = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
    ePhysicalDeviceDescriptorIndexingFeatures     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
    ePhysicalDeviceDescriptorIndexingProperties   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES,
    eDescriptorSetVariableDescriptorCountAllocateInfo =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO,
    eDescriptorSetVariableDescriptorCountLayoutSupport =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_LAYOUT_SUPPORT,
    ePhysicalDeviceDepthStencilResolveProperties = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES,
    eSubpassDescriptionDepthStencilResolve       = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE,
    ePhysicalDeviceScalarBlockLayoutFeatures     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES,
    eImageStencilUsageCreateInfo                 = VK_STRUCTURE_TYPE_IMAGE_STENCIL_USAGE_CREATE_INFO,
    ePhysicalDeviceSamplerFilterMinmaxProperties = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_FILTER_MINMAX_PROPERTIES,
    eSamplerReductionModeCreateInfo              = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO,
    ePhysicalDeviceVulkanMemoryModelFeatures     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES,
    ePhysicalDeviceImagelessFramebufferFeatures  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES,
    eFramebufferAttachmentsCreateInfo            = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENTS_CREATE_INFO,
    eFramebufferAttachmentImageInfo              = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENT_IMAGE_INFO,
    eRenderPassAttachmentBeginInfo               = VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO,
    ePhysicalDeviceUniformBufferStandardLayoutFeatures =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES,
    ePhysicalDeviceShaderSubgroupExtendedTypesFeatures =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES,
    ePhysicalDeviceSeparateDepthStencilLayoutsFeatures =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES,
    eAttachmentReferenceStencilLayout          = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_STENCIL_LAYOUT,
    eAttachmentDescriptionStencilLayout        = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_STENCIL_LAYOUT,
    ePhysicalDeviceHostQueryResetFeatures      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES,
    ePhysicalDeviceTimelineSemaphoreFeatures   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
    ePhysicalDeviceTimelineSemaphoreProperties = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES,
    eSemaphoreTypeCreateInfo                   = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
    eTimelineSemaphoreSubmitInfo               = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
    eSemaphoreWaitInfo                         = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
    eSemaphoreSignalInfo                       = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
    ePhysicalDeviceBufferDeviceAddressFeatures = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
    eBufferDeviceAddressInfo                   = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
    eBufferOpaqueCaptureAddressCreateInfo      = VK_STRUCTURE_TYPE_BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO,
    eMemoryOpaqueCaptureAddressAllocateInfo    = VK_STRUCTURE_TYPE_MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO,
    eDeviceMemoryOpaqueCaptureAddressInfo      = VK_STRUCTURE_TYPE_DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO,
    eSwapchainCreateInfoKHR                    = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
    ePresentInfoKHR                            = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
    eDeviceGroupPresentCapabilitiesKHR         = VK_STRUCTURE_TYPE_DEVICE_GROUP_PRESENT_CAPABILITIES_KHR,
    eImageSwapchainCreateInfoKHR               = VK_STRUCTURE_TYPE_IMAGE_SWAPCHAIN_CREATE_INFO_KHR,
    eBindImageMemorySwapchainInfoKHR           = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_SWAPCHAIN_INFO_KHR,
    eAcquireNextImageInfoKHR                   = VK_STRUCTURE_TYPE_ACQUIRE_NEXT_IMAGE_INFO_KHR,
    eDeviceGroupPresentInfoKHR                 = VK_STRUCTURE_TYPE_DEVICE_GROUP_PRESENT_INFO_KHR,
    eDeviceGroupSwapchainCreateInfoKHR         = VK_STRUCTURE_TYPE_DEVICE_GROUP_SWAPCHAIN_CREATE_INFO_KHR,
    eDisplayModeCreateInfoKHR                  = VK_STRUCTURE_TYPE_DISPLAY_MODE_CREATE_INFO_KHR,
    eDisplaySurfaceCreateInfoKHR               = VK_STRUCTURE_TYPE_DISPLAY_SURFACE_CREATE_INFO_KHR,
    eDisplayPresentInfoKHR                     = VK_STRUCTURE_TYPE_DISPLAY_PRESENT_INFO_KHR,
#if defined( VK_USE_PLATFORM_XLIB_KHR )
    eXlibSurfaceCreateInfoKHR = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
#endif /*VK_USE_PLATFORM_XLIB_KHR*/
#if defined( VK_USE_PLATFORM_XCB_KHR )
    eXcbSurfaceCreateInfoKHR = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
#endif /*VK_USE_PLATFORM_XCB_KHR*/
#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
    eWaylandSurfaceCreateInfoKHR = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
    eAndroidSurfaceCreateInfoKHR = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR,
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    eWin32SurfaceCreateInfoKHR = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    eDebugReportCallbackCreateInfoEXT = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
    ePipelineRasterizationStateRasterizationOrderAMD =
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_RASTERIZATION_ORDER_AMD,
    eDebugMarkerObjectNameInfoEXT = VK_STRUCTURE_TYPE_DEBUG_MARKER_OBJECT_NAME_INFO_EXT,
    eDebugMarkerObjectTagInfoEXT  = VK_STRUCTURE_TYPE_DEBUG_MARKER_OBJECT_TAG_INFO_EXT,
    eDebugMarkerMarkerInfoEXT     = VK_STRUCTURE_TYPE_DEBUG_MARKER_MARKER_INFO_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoProfileKHR                     = VK_STRUCTURE_TYPE_VIDEO_PROFILE_KHR,
    eVideoCapabilitiesKHR                = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR,
    eVideoPictureResourceKHR             = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_KHR,
    eVideoGetMemoryPropertiesKHR         = VK_STRUCTURE_TYPE_VIDEO_GET_MEMORY_PROPERTIES_KHR,
    eVideoBindMemoryKHR                  = VK_STRUCTURE_TYPE_VIDEO_BIND_MEMORY_KHR,
    eVideoSessionCreateInfoKHR           = VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR,
    eVideoSessionParametersCreateInfoKHR = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
    eVideoSessionParametersUpdateInfoKHR = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_UPDATE_INFO_KHR,
    eVideoBeginCodingInfoKHR             = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR,
    eVideoEndCodingInfoKHR               = VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR,
    eVideoCodingControlInfoKHR           = VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR,
    eVideoReferenceSlotKHR               = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_KHR,
    eVideoQueueFamilyProperties2KHR      = VK_STRUCTURE_TYPE_VIDEO_QUEUE_FAMILY_PROPERTIES_2_KHR,
    eVideoProfilesKHR                    = VK_STRUCTURE_TYPE_VIDEO_PROFILES_KHR,
    ePhysicalDeviceVideoFormatInfoKHR    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_FORMAT_INFO_KHR,
    eVideoFormatPropertiesKHR            = VK_STRUCTURE_TYPE_VIDEO_FORMAT_PROPERTIES_KHR,
    eVideoDecodeInfoKHR                  = VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eDedicatedAllocationImageCreateInfoNV         = VK_STRUCTURE_TYPE_DEDICATED_ALLOCATION_IMAGE_CREATE_INFO_NV,
    eDedicatedAllocationBufferCreateInfoNV        = VK_STRUCTURE_TYPE_DEDICATED_ALLOCATION_BUFFER_CREATE_INFO_NV,
    eDedicatedAllocationMemoryAllocateInfoNV      = VK_STRUCTURE_TYPE_DEDICATED_ALLOCATION_MEMORY_ALLOCATE_INFO_NV,
    ePhysicalDeviceTransformFeedbackFeaturesEXT   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT,
    ePhysicalDeviceTransformFeedbackPropertiesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_PROPERTIES_EXT,
    ePipelineRasterizationStateStreamCreateInfoEXT =
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_STREAM_CREATE_INFO_EXT,
    eCuModuleCreateInfoNVX         = VK_STRUCTURE_TYPE_CU_MODULE_CREATE_INFO_NVX,
    eCuFunctionCreateInfoNVX       = VK_STRUCTURE_TYPE_CU_FUNCTION_CREATE_INFO_NVX,
    eCuLaunchInfoNVX               = VK_STRUCTURE_TYPE_CU_LAUNCH_INFO_NVX,
    eImageViewHandleInfoNVX        = VK_STRUCTURE_TYPE_IMAGE_VIEW_HANDLE_INFO_NVX,
    eImageViewAddressPropertiesNVX = VK_STRUCTURE_TYPE_IMAGE_VIEW_ADDRESS_PROPERTIES_NVX,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoEncodeH264CapabilitiesEXT      = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_CAPABILITIES_EXT,
    eVideoEncodeH264SessionCreateInfoEXT = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_CREATE_INFO_EXT,
    eVideoEncodeH264SessionParametersCreateInfoEXT =
      VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_CREATE_INFO_EXT,
    eVideoEncodeH264SessionParametersAddInfoEXT = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_ADD_INFO_EXT,
    eVideoEncodeH264VclFrameInfoEXT             = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_VCL_FRAME_INFO_EXT,
    eVideoEncodeH264DpbSlotInfoEXT              = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_DPB_SLOT_INFO_EXT,
    eVideoEncodeH264NaluSliceEXT                = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_NALU_SLICE_EXT,
    eVideoEncodeH264EmitPictureParametersEXT    = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_EMIT_PICTURE_PARAMETERS_EXT,
    eVideoEncodeH264ProfileEXT                  = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PROFILE_EXT,
    eVideoDecodeH264CapabilitiesEXT             = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_CAPABILITIES_EXT,
    eVideoDecodeH264SessionCreateInfoEXT        = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_CREATE_INFO_EXT,
    eVideoDecodeH264PictureInfoEXT              = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PICTURE_INFO_EXT,
    eVideoDecodeH264MvcEXT                      = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_MVC_EXT,
    eVideoDecodeH264ProfileEXT                  = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PROFILE_EXT,
    eVideoDecodeH264SessionParametersCreateInfoEXT =
      VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO_EXT,
    eVideoDecodeH264SessionParametersAddInfoEXT = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_EXT,
    eVideoDecodeH264DpbSlotInfoEXT              = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_EXT,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eTextureLodGatherFormatPropertiesAMD = VK_STRUCTURE_TYPE_TEXTURE_LOD_GATHER_FORMAT_PROPERTIES_AMD,
#if defined( VK_USE_PLATFORM_GGP )
    eStreamDescriptorSurfaceCreateInfoGGP = VK_STRUCTURE_TYPE_STREAM_DESCRIPTOR_SURFACE_CREATE_INFO_GGP,
#endif /*VK_USE_PLATFORM_GGP*/
    ePhysicalDeviceCornerSampledImageFeaturesNV = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CORNER_SAMPLED_IMAGE_FEATURES_NV,
    eExternalMemoryImageCreateInfoNV            = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO_NV,
    eExportMemoryAllocateInfoNV                 = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_NV,
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    eImportMemoryWin32HandleInfoNV       = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_NV,
    eExportMemoryWin32HandleInfoNV       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_NV,
    eWin32KeyedMutexAcquireReleaseInfoNV = VK_STRUCTURE_TYPE_WIN32_KEYED_MUTEX_ACQUIRE_RELEASE_INFO_NV,
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    eValidationFlagsEXT = VK_STRUCTURE_TYPE_VALIDATION_FLAGS_EXT,
#if defined( VK_USE_PLATFORM_VI_NN )
    eViSurfaceCreateInfoNN = VK_STRUCTURE_TYPE_VI_SURFACE_CREATE_INFO_NN,
#endif /*VK_USE_PLATFORM_VI_NN*/
    ePhysicalDeviceTextureCompressionAstcHdrFeaturesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES_EXT,
    eImageViewAstcDecodeModeEXT          = VK_STRUCTURE_TYPE_IMAGE_VIEW_ASTC_DECODE_MODE_EXT,
    ePhysicalDeviceAstcDecodeFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ASTC_DECODE_FEATURES_EXT,
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    eImportMemoryWin32HandleInfoKHR = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR,
    eExportMemoryWin32HandleInfoKHR = VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR,
    eMemoryWin32HandlePropertiesKHR = VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR,
    eMemoryGetWin32HandleInfoKHR    = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    eImportMemoryFdInfoKHR = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
    eMemoryFdPropertiesKHR = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR,
    eMemoryGetFdInfoKHR    = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    eWin32KeyedMutexAcquireReleaseInfoKHR = VK_STRUCTURE_TYPE_WIN32_KEYED_MUTEX_ACQUIRE_RELEASE_INFO_KHR,
    eImportSemaphoreWin32HandleInfoKHR    = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR,
    eExportSemaphoreWin32HandleInfoKHR    = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR,
    eD3D12FenceSubmitInfoKHR              = VK_STRUCTURE_TYPE_D3D12_FENCE_SUBMIT_INFO_KHR,
    eSemaphoreGetWin32HandleInfoKHR       = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR,
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    eImportSemaphoreFdInfoKHR                  = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
    eSemaphoreGetFdInfoKHR                     = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
    ePhysicalDevicePushDescriptorPropertiesKHR = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR,
    eCommandBufferInheritanceConditionalRenderingInfoEXT =
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_CONDITIONAL_RENDERING_INFO_EXT,
    ePhysicalDeviceConditionalRenderingFeaturesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT,
    eConditionalRenderingBeginInfoEXT          = VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT,
    ePresentRegionsKHR                         = VK_STRUCTURE_TYPE_PRESENT_REGIONS_KHR,
    ePipelineViewportWScalingStateCreateInfoNV = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_W_SCALING_STATE_CREATE_INFO_NV,
    eSurfaceCapabilities2EXT                   = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_EXT,
    eDisplayPowerInfoEXT                       = VK_STRUCTURE_TYPE_DISPLAY_POWER_INFO_EXT,
    eDeviceEventInfoEXT                        = VK_STRUCTURE_TYPE_DEVICE_EVENT_INFO_EXT,
    eDisplayEventInfoEXT                       = VK_STRUCTURE_TYPE_DISPLAY_EVENT_INFO_EXT,
    eSwapchainCounterCreateInfoEXT             = VK_STRUCTURE_TYPE_SWAPCHAIN_COUNTER_CREATE_INFO_EXT,
    ePresentTimesInfoGOOGLE                    = VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE,
    ePhysicalDeviceMultiviewPerViewAttributesPropertiesNVX =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PER_VIEW_ATTRIBUTES_PROPERTIES_NVX,
    ePipelineViewportSwizzleStateCreateInfoNV    = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_SWIZZLE_STATE_CREATE_INFO_NV,
    ePhysicalDeviceDiscardRectanglePropertiesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DISCARD_RECTANGLE_PROPERTIES_EXT,
    ePipelineDiscardRectangleStateCreateInfoEXT  = VK_STRUCTURE_TYPE_PIPELINE_DISCARD_RECTANGLE_STATE_CREATE_INFO_EXT,
    ePhysicalDeviceConservativeRasterizationPropertiesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES_EXT,
    ePipelineRasterizationConservativeStateCreateInfoEXT =
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT,
    ePhysicalDeviceDepthClipEnableFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT,
    ePipelineRasterizationDepthClipStateCreateInfoEXT =
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT,
    eHdrMetadataEXT                      = VK_STRUCTURE_TYPE_HDR_METADATA_EXT,
    eSharedPresentSurfaceCapabilitiesKHR = VK_STRUCTURE_TYPE_SHARED_PRESENT_SURFACE_CAPABILITIES_KHR,
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    eImportFenceWin32HandleInfoKHR = VK_STRUCTURE_TYPE_IMPORT_FENCE_WIN32_HANDLE_INFO_KHR,
    eExportFenceWin32HandleInfoKHR = VK_STRUCTURE_TYPE_EXPORT_FENCE_WIN32_HANDLE_INFO_KHR,
    eFenceGetWin32HandleInfoKHR    = VK_STRUCTURE_TYPE_FENCE_GET_WIN32_HANDLE_INFO_KHR,
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    eImportFenceFdInfoKHR                        = VK_STRUCTURE_TYPE_IMPORT_FENCE_FD_INFO_KHR,
    eFenceGetFdInfoKHR                           = VK_STRUCTURE_TYPE_FENCE_GET_FD_INFO_KHR,
    ePhysicalDevicePerformanceQueryFeaturesKHR   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PERFORMANCE_QUERY_FEATURES_KHR,
    ePhysicalDevicePerformanceQueryPropertiesKHR = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PERFORMANCE_QUERY_PROPERTIES_KHR,
    eQueryPoolPerformanceCreateInfoKHR           = VK_STRUCTURE_TYPE_QUERY_POOL_PERFORMANCE_CREATE_INFO_KHR,
    ePerformanceQuerySubmitInfoKHR               = VK_STRUCTURE_TYPE_PERFORMANCE_QUERY_SUBMIT_INFO_KHR,
    eAcquireProfilingLockInfoKHR                 = VK_STRUCTURE_TYPE_ACQUIRE_PROFILING_LOCK_INFO_KHR,
    ePerformanceCounterKHR                       = VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_KHR,
    ePerformanceCounterDescriptionKHR            = VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_DESCRIPTION_KHR,
    ePhysicalDeviceSurfaceInfo2KHR               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR,
    eSurfaceCapabilities2KHR                     = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR,
    eSurfaceFormat2KHR                           = VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR,
    eDisplayProperties2KHR                       = VK_STRUCTURE_TYPE_DISPLAY_PROPERTIES_2_KHR,
    eDisplayPlaneProperties2KHR                  = VK_STRUCTURE_TYPE_DISPLAY_PLANE_PROPERTIES_2_KHR,
    eDisplayModeProperties2KHR                   = VK_STRUCTURE_TYPE_DISPLAY_MODE_PROPERTIES_2_KHR,
    eDisplayPlaneInfo2KHR                        = VK_STRUCTURE_TYPE_DISPLAY_PLANE_INFO_2_KHR,
    eDisplayPlaneCapabilities2KHR                = VK_STRUCTURE_TYPE_DISPLAY_PLANE_CAPABILITIES_2_KHR,
#if defined( VK_USE_PLATFORM_IOS_MVK )
    eIosSurfaceCreateInfoMVK = VK_STRUCTURE_TYPE_IOS_SURFACE_CREATE_INFO_MVK,
#endif /*VK_USE_PLATFORM_IOS_MVK*/
#if defined( VK_USE_PLATFORM_MACOS_MVK )
    eMacosSurfaceCreateInfoMVK = VK_STRUCTURE_TYPE_MACOS_SURFACE_CREATE_INFO_MVK,
#endif /*VK_USE_PLATFORM_MACOS_MVK*/
    eDebugUtilsObjectNameInfoEXT        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
    eDebugUtilsObjectTagInfoEXT         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_TAG_INFO_EXT,
    eDebugUtilsLabelEXT                 = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
    eDebugUtilsMessengerCallbackDataEXT = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT,
    eDebugUtilsMessengerCreateInfoEXT   = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
    eAndroidHardwareBufferUsageANDROID            = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_USAGE_ANDROID,
    eAndroidHardwareBufferPropertiesANDROID       = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID,
    eAndroidHardwareBufferFormatPropertiesANDROID = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID,
    eImportAndroidHardwareBufferInfoANDROID       = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
    eMemoryGetAndroidHardwareBufferInfoANDROID    = VK_STRUCTURE_TYPE_MEMORY_GET_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
    eExternalFormatANDROID                        = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID,
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
    ePhysicalDeviceInlineUniformBlockFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES_EXT,
    ePhysicalDeviceInlineUniformBlockPropertiesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_PROPERTIES_EXT,
    eWriteDescriptorSetInlineUniformBlockEXT = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK_EXT,
    eDescriptorPoolInlineUniformBlockCreateInfoEXT =
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_INLINE_UNIFORM_BLOCK_CREATE_INFO_EXT,
    eSampleLocationsInfoEXT                     = VK_STRUCTURE_TYPE_SAMPLE_LOCATIONS_INFO_EXT,
    eRenderPassSampleLocationsBeginInfoEXT      = VK_STRUCTURE_TYPE_RENDER_PASS_SAMPLE_LOCATIONS_BEGIN_INFO_EXT,
    ePipelineSampleLocationsStateCreateInfoEXT  = VK_STRUCTURE_TYPE_PIPELINE_SAMPLE_LOCATIONS_STATE_CREATE_INFO_EXT,
    ePhysicalDeviceSampleLocationsPropertiesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLE_LOCATIONS_PROPERTIES_EXT,
    eMultisamplePropertiesEXT                   = VK_STRUCTURE_TYPE_MULTISAMPLE_PROPERTIES_EXT,
    ePhysicalDeviceBlendOperationAdvancedFeaturesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BLEND_OPERATION_ADVANCED_FEATURES_EXT,
    ePhysicalDeviceBlendOperationAdvancedPropertiesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BLEND_OPERATION_ADVANCED_PROPERTIES_EXT,
    ePipelineColorBlendAdvancedStateCreateInfoEXT =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_ADVANCED_STATE_CREATE_INFO_EXT,
    ePipelineCoverageToColorStateCreateInfoNV   = VK_STRUCTURE_TYPE_PIPELINE_COVERAGE_TO_COLOR_STATE_CREATE_INFO_NV,
    eWriteDescriptorSetAccelerationStructureKHR = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
    eAccelerationStructureBuildGeometryInfoKHR  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
    eAccelerationStructureDeviceAddressInfoKHR  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
    eAccelerationStructureGeometryAabbsDataKHR  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR,
    eAccelerationStructureGeometryInstancesDataKHR =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
    eAccelerationStructureGeometryTrianglesDataKHR =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
    eAccelerationStructureGeometryKHR         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
    eAccelerationStructureVersionInfoKHR      = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_VERSION_INFO_KHR,
    eCopyAccelerationStructureInfoKHR         = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR,
    eCopyAccelerationStructureToMemoryInfoKHR = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_TO_MEMORY_INFO_KHR,
    eCopyMemoryToAccelerationStructureInfoKHR = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_ACCELERATION_STRUCTURE_INFO_KHR,
    ePhysicalDeviceAccelerationStructureFeaturesKHR =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
    ePhysicalDeviceAccelerationStructurePropertiesKHR =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR,
    eAccelerationStructureCreateInfoKHR          = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
    eAccelerationStructureBuildSizesInfoKHR      = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
    ePhysicalDeviceRayTracingPipelineFeaturesKHR = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
    ePhysicalDeviceRayTracingPipelinePropertiesKHR =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
    eRayTracingPipelineCreateInfoKHR             = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
    eRayTracingShaderGroupCreateInfoKHR          = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
    eRayTracingPipelineInterfaceCreateInfoKHR    = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_INTERFACE_CREATE_INFO_KHR,
    ePhysicalDeviceRayQueryFeaturesKHR           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
    ePipelineCoverageModulationStateCreateInfoNV = VK_STRUCTURE_TYPE_PIPELINE_COVERAGE_MODULATION_STATE_CREATE_INFO_NV,
    ePhysicalDeviceShaderSmBuiltinsFeaturesNV    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_FEATURES_NV,
    ePhysicalDeviceShaderSmBuiltinsPropertiesNV  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV,
    eDrmFormatModifierPropertiesListEXT          = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT,
    ePhysicalDeviceImageDrmFormatModifierInfoEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT,
    eImageDrmFormatModifierListCreateInfoEXT     = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
    eImageDrmFormatModifierExplicitCreateInfoEXT = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
    eImageDrmFormatModifierPropertiesEXT         = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT,
    eValidationCacheCreateInfoEXT                = VK_STRUCTURE_TYPE_VALIDATION_CACHE_CREATE_INFO_EXT,
    eShaderModuleValidationCacheCreateInfoEXT    = VK_STRUCTURE_TYPE_SHADER_MODULE_VALIDATION_CACHE_CREATE_INFO_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    ePhysicalDevicePortabilitySubsetFeaturesKHR   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR,
    ePhysicalDevicePortabilitySubsetPropertiesKHR = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_PROPERTIES_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    ePipelineViewportShadingRateImageStateCreateInfoNV =
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_SHADING_RATE_IMAGE_STATE_CREATE_INFO_NV,
    ePhysicalDeviceShadingRateImageFeaturesNV   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADING_RATE_IMAGE_FEATURES_NV,
    ePhysicalDeviceShadingRateImagePropertiesNV = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADING_RATE_IMAGE_PROPERTIES_NV,
    ePipelineViewportCoarseSampleOrderStateCreateInfoNV =
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_COARSE_SAMPLE_ORDER_STATE_CREATE_INFO_NV,
    eRayTracingPipelineCreateInfoNV            = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_NV,
    eAccelerationStructureCreateInfoNV         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV,
    eGeometryNV                                = VK_STRUCTURE_TYPE_GEOMETRY_NV,
    eGeometryTrianglesNV                       = VK_STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV,
    eGeometryAabbNV                            = VK_STRUCTURE_TYPE_GEOMETRY_AABB_NV,
    eBindAccelerationStructureMemoryInfoNV     = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV,
    eWriteDescriptorSetAccelerationStructureNV = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_NV,
    eAccelerationStructureMemoryRequirementsInfoNV =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV,
    ePhysicalDeviceRayTracingPropertiesNV = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PROPERTIES_NV,
    eRayTracingShaderGroupCreateInfoNV    = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV,
    eAccelerationStructureInfoNV          = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV,
    ePhysicalDeviceRepresentativeFragmentTestFeaturesNV =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_REPRESENTATIVE_FRAGMENT_TEST_FEATURES_NV,
    ePipelineRepresentativeFragmentTestStateCreateInfoNV =
      VK_STRUCTURE_TYPE_PIPELINE_REPRESENTATIVE_FRAGMENT_TEST_STATE_CREATE_INFO_NV,
    ePhysicalDeviceImageViewImageFormatInfoEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_VIEW_IMAGE_FORMAT_INFO_EXT,
    eFilterCubicImageViewImageFormatPropertiesEXT =
      VK_STRUCTURE_TYPE_FILTER_CUBIC_IMAGE_VIEW_IMAGE_FORMAT_PROPERTIES_EXT,
    eDeviceQueueGlobalPriorityCreateInfoEXT = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_EXT,
    eImportMemoryHostPointerInfoEXT         = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
    eMemoryHostPointerPropertiesEXT         = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT,
    ePhysicalDeviceExternalMemoryHostPropertiesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT,
    ePhysicalDeviceShaderClockFeaturesKHR  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR,
    ePipelineCompilerControlCreateInfoAMD  = VK_STRUCTURE_TYPE_PIPELINE_COMPILER_CONTROL_CREATE_INFO_AMD,
    eCalibratedTimestampInfoEXT            = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT,
    ePhysicalDeviceShaderCorePropertiesAMD = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoDecodeH265CapabilitiesEXT      = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_CAPABILITIES_EXT,
    eVideoDecodeH265SessionCreateInfoEXT = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_CREATE_INFO_EXT,
    eVideoDecodeH265SessionParametersCreateInfoEXT =
      VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_CREATE_INFO_EXT,
    eVideoDecodeH265SessionParametersAddInfoEXT = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_ADD_INFO_EXT,
    eVideoDecodeH265ProfileEXT                  = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PROFILE_EXT,
    eVideoDecodeH265PictureInfoEXT              = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PICTURE_INFO_EXT,
    eVideoDecodeH265DpbSlotInfoEXT              = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_DPB_SLOT_INFO_EXT,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eDeviceMemoryOverallocationCreateInfoAMD = VK_STRUCTURE_TYPE_DEVICE_MEMORY_OVERALLOCATION_CREATE_INFO_AMD,
    ePhysicalDeviceVertexAttributeDivisorPropertiesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_PROPERTIES_EXT,
    ePipelineVertexInputDivisorStateCreateInfoEXT =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT,
    ePhysicalDeviceVertexAttributeDivisorFeaturesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT,
#if defined( VK_USE_PLATFORM_GGP )
    ePresentFrameTokenGGP = VK_STRUCTURE_TYPE_PRESENT_FRAME_TOKEN_GGP,
#endif /*VK_USE_PLATFORM_GGP*/
    ePipelineCreationFeedbackCreateInfoEXT = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO_EXT,
    ePhysicalDeviceComputeShaderDerivativesFeaturesNV =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_NV,
    ePhysicalDeviceMeshShaderFeaturesNV   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV,
    ePhysicalDeviceMeshShaderPropertiesNV = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_NV,
    ePhysicalDeviceFragmentShaderBarycentricFeaturesNV =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_NV,
    ePhysicalDeviceShaderImageFootprintFeaturesNV =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_FOOTPRINT_FEATURES_NV,
    ePipelineViewportExclusiveScissorStateCreateInfoNV =
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_EXCLUSIVE_SCISSOR_STATE_CREATE_INFO_NV,
    ePhysicalDeviceExclusiveScissorFeaturesNV = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXCLUSIVE_SCISSOR_FEATURES_NV,
    eCheckpointDataNV                         = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV,
    eQueueFamilyCheckpointPropertiesNV        = VK_STRUCTURE_TYPE_QUEUE_FAMILY_CHECKPOINT_PROPERTIES_NV,
    ePhysicalDeviceShaderIntegerFunctions2FeaturesINTEL =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_FUNCTIONS_2_FEATURES_INTEL,
    eQueryPoolPerformanceQueryCreateInfoINTEL = VK_STRUCTURE_TYPE_QUERY_POOL_PERFORMANCE_QUERY_CREATE_INFO_INTEL,
    eInitializePerformanceApiInfoINTEL        = VK_STRUCTURE_TYPE_INITIALIZE_PERFORMANCE_API_INFO_INTEL,
    ePerformanceMarkerInfoINTEL               = VK_STRUCTURE_TYPE_PERFORMANCE_MARKER_INFO_INTEL,
    ePerformanceStreamMarkerInfoINTEL         = VK_STRUCTURE_TYPE_PERFORMANCE_STREAM_MARKER_INFO_INTEL,
    ePerformanceOverrideInfoINTEL             = VK_STRUCTURE_TYPE_PERFORMANCE_OVERRIDE_INFO_INTEL,
    ePerformanceConfigurationAcquireInfoINTEL = VK_STRUCTURE_TYPE_PERFORMANCE_CONFIGURATION_ACQUIRE_INFO_INTEL,
    ePhysicalDevicePciBusInfoPropertiesEXT    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT,
    eDisplayNativeHdrSurfaceCapabilitiesAMD   = VK_STRUCTURE_TYPE_DISPLAY_NATIVE_HDR_SURFACE_CAPABILITIES_AMD,
    eSwapchainDisplayNativeHdrCreateInfoAMD   = VK_STRUCTURE_TYPE_SWAPCHAIN_DISPLAY_NATIVE_HDR_CREATE_INFO_AMD,
#if defined( VK_USE_PLATFORM_FUCHSIA )
    eImagepipeSurfaceCreateInfoFUCHSIA = VK_STRUCTURE_TYPE_IMAGEPIPE_SURFACE_CREATE_INFO_FUCHSIA,
#endif /*VK_USE_PLATFORM_FUCHSIA*/
    ePhysicalDeviceShaderTerminateInvocationFeaturesKHR =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES_KHR,
#if defined( VK_USE_PLATFORM_METAL_EXT )
    eMetalSurfaceCreateInfoEXT = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT,
#endif /*VK_USE_PLATFORM_METAL_EXT*/
    ePhysicalDeviceFragmentDensityMapFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_FEATURES_EXT,
    ePhysicalDeviceFragmentDensityMapPropertiesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_PROPERTIES_EXT,
    eRenderPassFragmentDensityMapCreateInfoEXT = VK_STRUCTURE_TYPE_RENDER_PASS_FRAGMENT_DENSITY_MAP_CREATE_INFO_EXT,
    ePhysicalDeviceSubgroupSizeControlPropertiesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT,
    ePipelineShaderStageRequiredSubgroupSizeCreateInfoEXT =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
    ePhysicalDeviceSubgroupSizeControlFeaturesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT,
    eFragmentShadingRateAttachmentInfoKHR = VK_STRUCTURE_TYPE_FRAGMENT_SHADING_RATE_ATTACHMENT_INFO_KHR,
    ePipelineFragmentShadingRateStateCreateInfoKHR =
      VK_STRUCTURE_TYPE_PIPELINE_FRAGMENT_SHADING_RATE_STATE_CREATE_INFO_KHR,
    ePhysicalDeviceFragmentShadingRatePropertiesKHR =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_PROPERTIES_KHR,
    ePhysicalDeviceFragmentShadingRateFeaturesKHR =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR,
    ePhysicalDeviceFragmentShadingRateKHR    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_KHR,
    ePhysicalDeviceShaderCoreProperties2AMD  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_2_AMD,
    ePhysicalDeviceCoherentMemoryFeaturesAMD = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD,
    ePhysicalDeviceShaderImageAtomicInt64FeaturesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT,
    ePhysicalDeviceMemoryBudgetPropertiesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT,
    ePhysicalDeviceMemoryPriorityFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT,
    eMemoryPriorityAllocateInfoEXT           = VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT,
    eSurfaceProtectedCapabilitiesKHR         = VK_STRUCTURE_TYPE_SURFACE_PROTECTED_CAPABILITIES_KHR,
    ePhysicalDeviceDedicatedAllocationImageAliasingFeaturesNV =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEDICATED_ALLOCATION_IMAGE_ALIASING_FEATURES_NV,
    ePhysicalDeviceBufferDeviceAddressFeaturesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_EXT,
    eBufferDeviceAddressCreateInfoEXT            = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_CREATE_INFO_EXT,
    ePhysicalDeviceToolPropertiesEXT             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TOOL_PROPERTIES_EXT,
    eValidationFeaturesEXT                       = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
    ePhysicalDeviceCooperativeMatrixFeaturesNV   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_NV,
    eCooperativeMatrixPropertiesNV               = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_NV,
    ePhysicalDeviceCooperativeMatrixPropertiesNV = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_NV,
    ePhysicalDeviceCoverageReductionModeFeaturesNV =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COVERAGE_REDUCTION_MODE_FEATURES_NV,
    ePipelineCoverageReductionStateCreateInfoNV = VK_STRUCTURE_TYPE_PIPELINE_COVERAGE_REDUCTION_STATE_CREATE_INFO_NV,
    eFramebufferMixedSamplesCombinationNV       = VK_STRUCTURE_TYPE_FRAMEBUFFER_MIXED_SAMPLES_COMBINATION_NV,
    ePhysicalDeviceFragmentShaderInterlockFeaturesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT,
    ePhysicalDeviceYcbcrImageArraysFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_YCBCR_IMAGE_ARRAYS_FEATURES_EXT,
    ePhysicalDeviceProvokingVertexFeaturesEXT  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT,
    ePipelineRasterizationProvokingVertexStateCreateInfoEXT =
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT,
    ePhysicalDeviceProvokingVertexPropertiesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_PROPERTIES_EXT,
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    eSurfaceFullScreenExclusiveInfoEXT         = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_INFO_EXT,
    eSurfaceCapabilitiesFullScreenExclusiveEXT = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_FULL_SCREEN_EXCLUSIVE_EXT,
    eSurfaceFullScreenExclusiveWin32InfoEXT    = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_WIN32_INFO_EXT,
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    eHeadlessSurfaceCreateInfoEXT                 = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT,
    ePhysicalDeviceLineRasterizationFeaturesEXT   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_EXT,
    ePipelineRasterizationLineStateCreateInfoEXT  = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO_EXT,
    ePhysicalDeviceLineRasterizationPropertiesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_PROPERTIES_EXT,
    ePhysicalDeviceShaderAtomicFloatFeaturesEXT   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
    ePhysicalDeviceIndexTypeUint8FeaturesEXT      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT,
    ePhysicalDeviceExtendedDynamicStateFeaturesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT,
    ePhysicalDevicePipelineExecutablePropertiesFeaturesKHR =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR,
    ePipelineInfoKHR                             = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR,
    ePipelineExecutablePropertiesKHR             = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR,
    ePipelineExecutableInfoKHR                   = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR,
    ePipelineExecutableStatisticKHR              = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR,
    ePipelineExecutableInternalRepresentationKHR = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR,
    ePhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES_EXT,
    ePhysicalDeviceDeviceGeneratedCommandsPropertiesNV =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_PROPERTIES_NV,
    eGraphicsShaderGroupCreateInfoNV           = VK_STRUCTURE_TYPE_GRAPHICS_SHADER_GROUP_CREATE_INFO_NV,
    eGraphicsPipelineShaderGroupsCreateInfoNV  = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_SHADER_GROUPS_CREATE_INFO_NV,
    eIndirectCommandsLayoutTokenNV             = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_NV,
    eIndirectCommandsLayoutCreateInfoNV        = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_NV,
    eGeneratedCommandsInfoNV                   = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_NV,
    eGeneratedCommandsMemoryRequirementsInfoNV = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_NV,
    ePhysicalDeviceDeviceGeneratedCommandsFeaturesNV =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_NV,
    ePhysicalDeviceInheritedViewportScissorFeaturesNV =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INHERITED_VIEWPORT_SCISSOR_FEATURES_NV,
    eCommandBufferInheritanceViewportScissorInfoNV =
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_VIEWPORT_SCISSOR_INFO_NV,
    ePhysicalDeviceTexelBufferAlignmentFeaturesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_FEATURES_EXT,
    ePhysicalDeviceTexelBufferAlignmentPropertiesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_PROPERTIES_EXT,
    eCommandBufferInheritanceRenderPassTransformInfoQCOM =
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDER_PASS_TRANSFORM_INFO_QCOM,
    eRenderPassTransformBeginInfoQCOM            = VK_STRUCTURE_TYPE_RENDER_PASS_TRANSFORM_BEGIN_INFO_QCOM,
    ePhysicalDeviceDeviceMemoryReportFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_MEMORY_REPORT_FEATURES_EXT,
    eDeviceDeviceMemoryReportCreateInfoEXT       = VK_STRUCTURE_TYPE_DEVICE_DEVICE_MEMORY_REPORT_CREATE_INFO_EXT,
    eDeviceMemoryReportCallbackDataEXT           = VK_STRUCTURE_TYPE_DEVICE_MEMORY_REPORT_CALLBACK_DATA_EXT,
    ePhysicalDeviceRobustness2FeaturesEXT        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
    ePhysicalDeviceRobustness2PropertiesEXT      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_PROPERTIES_EXT,
    eSamplerCustomBorderColorCreateInfoEXT       = VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT,
    ePhysicalDeviceCustomBorderColorPropertiesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_PROPERTIES_EXT,
    ePhysicalDeviceCustomBorderColorFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT,
    ePipelineLibraryCreateInfoKHR               = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR,
    ePhysicalDevicePrivateDataFeaturesEXT       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES_EXT,
    eDevicePrivateDataCreateInfoEXT             = VK_STRUCTURE_TYPE_DEVICE_PRIVATE_DATA_CREATE_INFO_EXT,
    ePrivateDataSlotCreateInfoEXT               = VK_STRUCTURE_TYPE_PRIVATE_DATA_SLOT_CREATE_INFO_EXT,
    ePhysicalDevicePipelineCreationCacheControlFeaturesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoEncodeInfoKHR            = VK_STRUCTURE_TYPE_VIDEO_ENCODE_INFO_KHR,
    eVideoEncodeRateControlInfoKHR = VK_STRUCTURE_TYPE_VIDEO_ENCODE_RATE_CONTROL_INFO_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    ePhysicalDeviceDiagnosticsConfigFeaturesNV = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DIAGNOSTICS_CONFIG_FEATURES_NV,
    eDeviceDiagnosticsConfigCreateInfoNV       = VK_STRUCTURE_TYPE_DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO_NV,
    eMemoryBarrier2KHR                         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR,
    eBufferMemoryBarrier2KHR                   = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR,
    eImageMemoryBarrier2KHR                    = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR,
    eDependencyInfoKHR                         = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
    eSubmitInfo2KHR                            = VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR,
    eSemaphoreSubmitInfoKHR                    = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR,
    eCommandBufferSubmitInfoKHR                = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR,
    ePhysicalDeviceSynchronization2FeaturesKHR = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR,
    eQueueFamilyCheckpointProperties2NV        = VK_STRUCTURE_TYPE_QUEUE_FAMILY_CHECKPOINT_PROPERTIES_2_NV,
    eCheckpointData2NV                         = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_2_NV,
    ePhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_FEATURES_KHR,
    ePhysicalDeviceZeroInitializeWorkgroupMemoryFeaturesKHR =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES_KHR,
    ePhysicalDeviceFragmentShadingRateEnumsPropertiesNV =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_ENUMS_PROPERTIES_NV,
    ePhysicalDeviceFragmentShadingRateEnumsFeaturesNV =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_ENUMS_FEATURES_NV,
    ePipelineFragmentShadingRateEnumStateCreateInfoNV =
      VK_STRUCTURE_TYPE_PIPELINE_FRAGMENT_SHADING_RATE_ENUM_STATE_CREATE_INFO_NV,
    eAccelerationStructureGeometryMotionTrianglesDataNV =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_MOTION_TRIANGLES_DATA_NV,
    ePhysicalDeviceRayTracingMotionBlurFeaturesNV =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MOTION_BLUR_FEATURES_NV,
    eAccelerationStructureMotionInfoNV = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MOTION_INFO_NV,
    ePhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_YCBCR_2_PLANE_444_FORMATS_FEATURES_EXT,
    ePhysicalDeviceFragmentDensityMap2FeaturesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_2_FEATURES_EXT,
    ePhysicalDeviceFragmentDensityMap2PropertiesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_2_PROPERTIES_EXT,
    eCopyCommandTransformInfoQCOM             = VK_STRUCTURE_TYPE_COPY_COMMAND_TRANSFORM_INFO_QCOM,
    ePhysicalDeviceImageRobustnessFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES_EXT,
    ePhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR,
    eCopyBufferInfo2KHR                   = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2_KHR,
    eCopyImageInfo2KHR                    = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2_KHR,
    eCopyBufferToImageInfo2KHR            = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2_KHR,
    eCopyImageToBufferInfo2KHR            = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2_KHR,
    eBlitImageInfo2KHR                    = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2_KHR,
    eResolveImageInfo2KHR                 = VK_STRUCTURE_TYPE_RESOLVE_IMAGE_INFO_2_KHR,
    eBufferCopy2KHR                       = VK_STRUCTURE_TYPE_BUFFER_COPY_2_KHR,
    eImageCopy2KHR                        = VK_STRUCTURE_TYPE_IMAGE_COPY_2_KHR,
    eImageBlit2KHR                        = VK_STRUCTURE_TYPE_IMAGE_BLIT_2_KHR,
    eBufferImageCopy2KHR                  = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2_KHR,
    eImageResolve2KHR                     = VK_STRUCTURE_TYPE_IMAGE_RESOLVE_2_KHR,
    ePhysicalDevice4444FormatsFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_4444_FORMATS_FEATURES_EXT,
#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
    eDirectfbSurfaceCreateInfoEXT = VK_STRUCTURE_TYPE_DIRECTFB_SURFACE_CREATE_INFO_EXT,
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/
    ePhysicalDeviceMutableDescriptorTypeFeaturesVALVE =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_VALVE,
    eMutableDescriptorTypeCreateInfoVALVE = VK_STRUCTURE_TYPE_MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_VALVE,
    ePhysicalDeviceVertexInputDynamicStateFeaturesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT,
    eVertexInputBindingDescription2EXT   = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
    eVertexInputAttributeDescription2EXT = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
    ePhysicalDeviceDrmPropertiesEXT      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT,
#if defined( VK_USE_PLATFORM_FUCHSIA )
    eImportMemoryZirconHandleInfoFUCHSIA    = VK_STRUCTURE_TYPE_IMPORT_MEMORY_ZIRCON_HANDLE_INFO_FUCHSIA,
    eMemoryZirconHandlePropertiesFUCHSIA    = VK_STRUCTURE_TYPE_MEMORY_ZIRCON_HANDLE_PROPERTIES_FUCHSIA,
    eMemoryGetZirconHandleInfoFUCHSIA       = VK_STRUCTURE_TYPE_MEMORY_GET_ZIRCON_HANDLE_INFO_FUCHSIA,
    eImportSemaphoreZirconHandleInfoFUCHSIA = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_ZIRCON_HANDLE_INFO_FUCHSIA,
    eSemaphoreGetZirconHandleInfoFUCHSIA    = VK_STRUCTURE_TYPE_SEMAPHORE_GET_ZIRCON_HANDLE_INFO_FUCHSIA,
#endif /*VK_USE_PLATFORM_FUCHSIA*/
    eSubpasssShadingPipelineCreateInfoHUAWEI      = VK_STRUCTURE_TYPE_SUBPASSS_SHADING_PIPELINE_CREATE_INFO_HUAWEI,
    ePhysicalDeviceSubpassShadingFeaturesHUAWEI   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBPASS_SHADING_FEATURES_HUAWEI,
    ePhysicalDeviceSubpassShadingPropertiesHUAWEI = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBPASS_SHADING_PROPERTIES_HUAWEI,
    ePhysicalDeviceExtendedDynamicState2FeaturesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT,
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
    eScreenSurfaceCreateInfoQNX = VK_STRUCTURE_TYPE_SCREEN_SURFACE_CREATE_INFO_QNX,
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/
    ePhysicalDeviceColorWriteEnableFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COLOR_WRITE_ENABLE_FEATURES_EXT,
    ePipelineColorWriteCreateInfoEXT           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_WRITE_CREATE_INFO_EXT,
    ePhysicalDeviceGlobalPriorityQueryFeaturesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GLOBAL_PRIORITY_QUERY_FEATURES_EXT,
    eQueueFamilyGlobalPriorityPropertiesEXT  = VK_STRUCTURE_TYPE_QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES_EXT,
    ePhysicalDeviceMultiDrawFeaturesEXT      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTI_DRAW_FEATURES_EXT,
    ePhysicalDeviceMultiDrawPropertiesEXT    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTI_DRAW_PROPERTIES_EXT,
    eAttachmentDescription2KHR               = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2_KHR,
    eAttachmentDescriptionStencilLayoutKHR   = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_STENCIL_LAYOUT_KHR,
    eAttachmentReference2KHR                 = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2_KHR,
    eAttachmentReferenceStencilLayoutKHR     = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_STENCIL_LAYOUT_KHR,
    eBindBufferMemoryDeviceGroupInfoKHR      = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_DEVICE_GROUP_INFO_KHR,
    eBindBufferMemoryInfoKHR                 = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO_KHR,
    eBindImageMemoryDeviceGroupInfoKHR       = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_DEVICE_GROUP_INFO_KHR,
    eBindImageMemoryInfoKHR                  = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO_KHR,
    eBindImagePlaneMemoryInfoKHR             = VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO_KHR,
    eBufferDeviceAddressInfoEXT              = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_EXT,
    eBufferDeviceAddressInfoKHR              = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR,
    eBufferMemoryRequirementsInfo2KHR        = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2_KHR,
    eBufferOpaqueCaptureAddressCreateInfoKHR = VK_STRUCTURE_TYPE_BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO_KHR,
    eDebugReportCreateInfoEXT                = VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT,
    eDescriptorSetLayoutBindingFlagsCreateInfoEXT =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT,
    eDescriptorSetLayoutSupportKHR = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT_KHR,
    eDescriptorSetVariableDescriptorCountAllocateInfoEXT =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT,
    eDescriptorSetVariableDescriptorCountLayoutSupportEXT =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_LAYOUT_SUPPORT_EXT,
    eDescriptorUpdateTemplateCreateInfoKHR     = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO_KHR,
    eDeviceGroupBindSparseInfoKHR              = VK_STRUCTURE_TYPE_DEVICE_GROUP_BIND_SPARSE_INFO_KHR,
    eDeviceGroupCommandBufferBeginInfoKHR      = VK_STRUCTURE_TYPE_DEVICE_GROUP_COMMAND_BUFFER_BEGIN_INFO_KHR,
    eDeviceGroupDeviceCreateInfoKHR            = VK_STRUCTURE_TYPE_DEVICE_GROUP_DEVICE_CREATE_INFO_KHR,
    eDeviceGroupRenderPassBeginInfoKHR         = VK_STRUCTURE_TYPE_DEVICE_GROUP_RENDER_PASS_BEGIN_INFO_KHR,
    eDeviceGroupSubmitInfoKHR                  = VK_STRUCTURE_TYPE_DEVICE_GROUP_SUBMIT_INFO_KHR,
    eDeviceMemoryOpaqueCaptureAddressInfoKHR   = VK_STRUCTURE_TYPE_DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO_KHR,
    eExportFenceCreateInfoKHR                  = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO_KHR,
    eExportMemoryAllocateInfoKHR               = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR,
    eExportSemaphoreCreateInfoKHR              = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR,
    eExternalBufferPropertiesKHR               = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES_KHR,
    eExternalFencePropertiesKHR                = VK_STRUCTURE_TYPE_EXTERNAL_FENCE_PROPERTIES_KHR,
    eExternalImageFormatPropertiesKHR          = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES_KHR,
    eExternalMemoryBufferCreateInfoKHR         = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO_KHR,
    eExternalMemoryImageCreateInfoKHR          = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO_KHR,
    eExternalSemaphorePropertiesKHR            = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES_KHR,
    eFormatProperties2KHR                      = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2_KHR,
    eFramebufferAttachmentsCreateInfoKHR       = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENTS_CREATE_INFO_KHR,
    eFramebufferAttachmentImageInfoKHR         = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENT_IMAGE_INFO_KHR,
    eImageFormatListCreateInfoKHR              = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO_KHR,
    eImageFormatProperties2KHR                 = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2_KHR,
    eImageMemoryRequirementsInfo2KHR           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2_KHR,
    eImagePlaneMemoryRequirementsInfoKHR       = VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO_KHR,
    eImageSparseMemoryRequirementsInfo2KHR     = VK_STRUCTURE_TYPE_IMAGE_SPARSE_MEMORY_REQUIREMENTS_INFO_2_KHR,
    eImageStencilUsageCreateInfoEXT            = VK_STRUCTURE_TYPE_IMAGE_STENCIL_USAGE_CREATE_INFO_EXT,
    eImageViewUsageCreateInfoKHR               = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO_KHR,
    eMemoryAllocateFlagsInfoKHR                = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_KHR,
    eMemoryDedicatedAllocateInfoKHR            = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR,
    eMemoryDedicatedRequirementsKHR            = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR,
    eMemoryOpaqueCaptureAddressAllocateInfoKHR = VK_STRUCTURE_TYPE_MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO_KHR,
    eMemoryRequirements2KHR                    = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR,
    ePhysicalDevice16BitStorageFeaturesKHR     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR,
    ePhysicalDevice8BitStorageFeaturesKHR      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR,
    ePhysicalDeviceBufferAddressFeaturesEXT    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_ADDRESS_FEATURES_EXT,
    ePhysicalDeviceBufferDeviceAddressFeaturesKHR =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR,
    ePhysicalDeviceDepthStencilResolvePropertiesKHR =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES_KHR,
    ePhysicalDeviceDescriptorIndexingFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT,
    ePhysicalDeviceDescriptorIndexingPropertiesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES_EXT,
    ePhysicalDeviceDriverPropertiesKHR        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES_KHR,
    ePhysicalDeviceExternalBufferInfoKHR      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO_KHR,
    ePhysicalDeviceExternalFenceInfoKHR       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_FENCE_INFO_KHR,
    ePhysicalDeviceExternalImageFormatInfoKHR = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO_KHR,
    ePhysicalDeviceExternalSemaphoreInfoKHR   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO_KHR,
    ePhysicalDeviceFeatures2KHR               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR,
    ePhysicalDeviceFloat16Int8FeaturesKHR     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR,
    ePhysicalDeviceFloatControlsPropertiesKHR = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES_KHR,
    ePhysicalDeviceGroupPropertiesKHR         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES_KHR,
    ePhysicalDeviceHostQueryResetFeaturesEXT  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES_EXT,
    ePhysicalDeviceIdPropertiesKHR            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES_KHR,
    ePhysicalDeviceImagelessFramebufferFeaturesKHR =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES_KHR,
    ePhysicalDeviceImageFormatInfo2KHR        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2_KHR,
    ePhysicalDeviceMaintenance3PropertiesKHR  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES_KHR,
    ePhysicalDeviceMemoryProperties2KHR       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR,
    ePhysicalDeviceMultiviewFeaturesKHR       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES_KHR,
    ePhysicalDeviceMultiviewPropertiesKHR     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES_KHR,
    ePhysicalDevicePointClippingPropertiesKHR = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_POINT_CLIPPING_PROPERTIES_KHR,
    ePhysicalDeviceProperties2KHR             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR,
    ePhysicalDeviceSamplerFilterMinmaxPropertiesEXT =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_FILTER_MINMAX_PROPERTIES_EXT,
    ePhysicalDeviceSamplerYcbcrConversionFeaturesKHR =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES_KHR,
    ePhysicalDeviceScalarBlockLayoutFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES_EXT,
    ePhysicalDeviceSeparateDepthStencilLayoutsFeaturesKHR =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES_KHR,
    ePhysicalDeviceShaderAtomicInt64FeaturesKHR = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES_KHR,
    ePhysicalDeviceShaderDrawParameterFeatures  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETER_FEATURES,
    ePhysicalDeviceShaderFloat16Int8FeaturesKHR = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR,
    ePhysicalDeviceShaderSubgroupExtendedTypesFeaturesKHR =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES_KHR,
    ePhysicalDeviceSparseImageFormatInfo2KHR      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SPARSE_IMAGE_FORMAT_INFO_2_KHR,
    ePhysicalDeviceTimelineSemaphoreFeaturesKHR   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR,
    ePhysicalDeviceTimelineSemaphorePropertiesKHR = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES_KHR,
    ePhysicalDeviceUniformBufferStandardLayoutFeaturesKHR =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES_KHR,
    ePhysicalDeviceVariablePointersFeaturesKHR  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES_KHR,
    ePhysicalDeviceVariablePointerFeatures      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTER_FEATURES,
    ePhysicalDeviceVariablePointerFeaturesKHR   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTER_FEATURES_KHR,
    ePhysicalDeviceVulkanMemoryModelFeaturesKHR = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES_KHR,
    ePipelineTessellationDomainOriginStateCreateInfoKHR =
      VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_DOMAIN_ORIGIN_STATE_CREATE_INFO_KHR,
    eQueryPoolCreateInfoINTEL         = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO_INTEL,
    eQueueFamilyProperties2KHR        = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2_KHR,
    eRenderPassAttachmentBeginInfoKHR = VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO_KHR,
    eRenderPassCreateInfo2KHR         = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2_KHR,
    eRenderPassInputAttachmentAspectCreateInfoKHR =
      VK_STRUCTURE_TYPE_RENDER_PASS_INPUT_ATTACHMENT_ASPECT_CREATE_INFO_KHR,
    eRenderPassMultiviewCreateInfoKHR    = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO_KHR,
    eSamplerReductionModeCreateInfoEXT   = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO_EXT,
    eSamplerYcbcrConversionCreateInfoKHR = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO_KHR,
    eSamplerYcbcrConversionImageFormatPropertiesKHR =
      VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES_KHR,
    eSamplerYcbcrConversionInfoKHR            = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO_KHR,
    eSemaphoreSignalInfoKHR                   = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO_KHR,
    eSemaphoreTypeCreateInfoKHR               = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR,
    eSemaphoreWaitInfoKHR                     = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO_KHR,
    eSparseImageFormatProperties2KHR          = VK_STRUCTURE_TYPE_SPARSE_IMAGE_FORMAT_PROPERTIES_2_KHR,
    eSparseImageMemoryRequirements2KHR        = VK_STRUCTURE_TYPE_SPARSE_IMAGE_MEMORY_REQUIREMENTS_2_KHR,
    eSubpassBeginInfoKHR                      = VK_STRUCTURE_TYPE_SUBPASS_BEGIN_INFO_KHR,
    eSubpassDependency2KHR                    = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2_KHR,
    eSubpassDescription2KHR                   = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2_KHR,
    eSubpassDescriptionDepthStencilResolveKHR = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE_KHR,
    eSubpassEndInfoKHR                        = VK_STRUCTURE_TYPE_SUBPASS_END_INFO_KHR,
    eTimelineSemaphoreSubmitInfoKHR           = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR
  };

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
      case StructureType::eRenderPassInputAttachmentAspectCreateInfo:
        return "RenderPassInputAttachmentAspectCreateInfo";
      case StructureType::eImageViewUsageCreateInfo: return "ImageViewUsageCreateInfo";
      case StructureType::ePipelineTessellationDomainOriginStateCreateInfo:
        return "PipelineTessellationDomainOriginStateCreateInfo";
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
      case StructureType::ePhysicalDeviceSamplerYcbcrConversionFeatures:
        return "PhysicalDeviceSamplerYcbcrConversionFeatures";
      case StructureType::eSamplerYcbcrConversionImageFormatProperties:
        return "SamplerYcbcrConversionImageFormatProperties";
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
      case StructureType::ePhysicalDeviceShaderDrawParametersFeatures:
        return "PhysicalDeviceShaderDrawParametersFeatures";
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
      case StructureType::eDescriptorSetLayoutBindingFlagsCreateInfo:
        return "DescriptorSetLayoutBindingFlagsCreateInfo";
      case StructureType::ePhysicalDeviceDescriptorIndexingFeatures: return "PhysicalDeviceDescriptorIndexingFeatures";
      case StructureType::ePhysicalDeviceDescriptorIndexingProperties:
        return "PhysicalDeviceDescriptorIndexingProperties";
      case StructureType::eDescriptorSetVariableDescriptorCountAllocateInfo:
        return "DescriptorSetVariableDescriptorCountAllocateInfo";
      case StructureType::eDescriptorSetVariableDescriptorCountLayoutSupport:
        return "DescriptorSetVariableDescriptorCountLayoutSupport";
      case StructureType::ePhysicalDeviceDepthStencilResolveProperties:
        return "PhysicalDeviceDepthStencilResolveProperties";
      case StructureType::eSubpassDescriptionDepthStencilResolve: return "SubpassDescriptionDepthStencilResolve";
      case StructureType::ePhysicalDeviceScalarBlockLayoutFeatures: return "PhysicalDeviceScalarBlockLayoutFeatures";
      case StructureType::eImageStencilUsageCreateInfo: return "ImageStencilUsageCreateInfo";
      case StructureType::ePhysicalDeviceSamplerFilterMinmaxProperties:
        return "PhysicalDeviceSamplerFilterMinmaxProperties";
      case StructureType::eSamplerReductionModeCreateInfo: return "SamplerReductionModeCreateInfo";
      case StructureType::ePhysicalDeviceVulkanMemoryModelFeatures: return "PhysicalDeviceVulkanMemoryModelFeatures";
      case StructureType::ePhysicalDeviceImagelessFramebufferFeatures:
        return "PhysicalDeviceImagelessFramebufferFeatures";
      case StructureType::eFramebufferAttachmentsCreateInfo: return "FramebufferAttachmentsCreateInfo";
      case StructureType::eFramebufferAttachmentImageInfo: return "FramebufferAttachmentImageInfo";
      case StructureType::eRenderPassAttachmentBeginInfo: return "RenderPassAttachmentBeginInfo";
      case StructureType::ePhysicalDeviceUniformBufferStandardLayoutFeatures:
        return "PhysicalDeviceUniformBufferStandardLayoutFeatures";
      case StructureType::ePhysicalDeviceShaderSubgroupExtendedTypesFeatures:
        return "PhysicalDeviceShaderSubgroupExtendedTypesFeatures";
      case StructureType::ePhysicalDeviceSeparateDepthStencilLayoutsFeatures:
        return "PhysicalDeviceSeparateDepthStencilLayoutsFeatures";
      case StructureType::eAttachmentReferenceStencilLayout: return "AttachmentReferenceStencilLayout";
      case StructureType::eAttachmentDescriptionStencilLayout: return "AttachmentDescriptionStencilLayout";
      case StructureType::ePhysicalDeviceHostQueryResetFeatures: return "PhysicalDeviceHostQueryResetFeatures";
      case StructureType::ePhysicalDeviceTimelineSemaphoreFeatures: return "PhysicalDeviceTimelineSemaphoreFeatures";
      case StructureType::ePhysicalDeviceTimelineSemaphoreProperties:
        return "PhysicalDeviceTimelineSemaphoreProperties";
      case StructureType::eSemaphoreTypeCreateInfo: return "SemaphoreTypeCreateInfo";
      case StructureType::eTimelineSemaphoreSubmitInfo: return "TimelineSemaphoreSubmitInfo";
      case StructureType::eSemaphoreWaitInfo: return "SemaphoreWaitInfo";
      case StructureType::eSemaphoreSignalInfo: return "SemaphoreSignalInfo";
      case StructureType::ePhysicalDeviceBufferDeviceAddressFeatures:
        return "PhysicalDeviceBufferDeviceAddressFeatures";
      case StructureType::eBufferDeviceAddressInfo: return "BufferDeviceAddressInfo";
      case StructureType::eBufferOpaqueCaptureAddressCreateInfo: return "BufferOpaqueCaptureAddressCreateInfo";
      case StructureType::eMemoryOpaqueCaptureAddressAllocateInfo: return "MemoryOpaqueCaptureAddressAllocateInfo";
      case StructureType::eDeviceMemoryOpaqueCaptureAddressInfo: return "DeviceMemoryOpaqueCaptureAddressInfo";
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
      case StructureType::ePipelineRasterizationStateRasterizationOrderAMD:
        return "PipelineRasterizationStateRasterizationOrderAMD";
      case StructureType::eDebugMarkerObjectNameInfoEXT: return "DebugMarkerObjectNameInfoEXT";
      case StructureType::eDebugMarkerObjectTagInfoEXT: return "DebugMarkerObjectTagInfoEXT";
      case StructureType::eDebugMarkerMarkerInfoEXT: return "DebugMarkerMarkerInfoEXT";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case StructureType::eVideoProfileKHR: return "VideoProfileKHR";
      case StructureType::eVideoCapabilitiesKHR: return "VideoCapabilitiesKHR";
      case StructureType::eVideoPictureResourceKHR: return "VideoPictureResourceKHR";
      case StructureType::eVideoGetMemoryPropertiesKHR: return "VideoGetMemoryPropertiesKHR";
      case StructureType::eVideoBindMemoryKHR: return "VideoBindMemoryKHR";
      case StructureType::eVideoSessionCreateInfoKHR: return "VideoSessionCreateInfoKHR";
      case StructureType::eVideoSessionParametersCreateInfoKHR: return "VideoSessionParametersCreateInfoKHR";
      case StructureType::eVideoSessionParametersUpdateInfoKHR: return "VideoSessionParametersUpdateInfoKHR";
      case StructureType::eVideoBeginCodingInfoKHR: return "VideoBeginCodingInfoKHR";
      case StructureType::eVideoEndCodingInfoKHR: return "VideoEndCodingInfoKHR";
      case StructureType::eVideoCodingControlInfoKHR: return "VideoCodingControlInfoKHR";
      case StructureType::eVideoReferenceSlotKHR: return "VideoReferenceSlotKHR";
      case StructureType::eVideoQueueFamilyProperties2KHR: return "VideoQueueFamilyProperties2KHR";
      case StructureType::eVideoProfilesKHR: return "VideoProfilesKHR";
      case StructureType::ePhysicalDeviceVideoFormatInfoKHR: return "PhysicalDeviceVideoFormatInfoKHR";
      case StructureType::eVideoFormatPropertiesKHR: return "VideoFormatPropertiesKHR";
      case StructureType::eVideoDecodeInfoKHR: return "VideoDecodeInfoKHR";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      case StructureType::eDedicatedAllocationImageCreateInfoNV: return "DedicatedAllocationImageCreateInfoNV";
      case StructureType::eDedicatedAllocationBufferCreateInfoNV: return "DedicatedAllocationBufferCreateInfoNV";
      case StructureType::eDedicatedAllocationMemoryAllocateInfoNV: return "DedicatedAllocationMemoryAllocateInfoNV";
      case StructureType::ePhysicalDeviceTransformFeedbackFeaturesEXT:
        return "PhysicalDeviceTransformFeedbackFeaturesEXT";
      case StructureType::ePhysicalDeviceTransformFeedbackPropertiesEXT:
        return "PhysicalDeviceTransformFeedbackPropertiesEXT";
      case StructureType::ePipelineRasterizationStateStreamCreateInfoEXT:
        return "PipelineRasterizationStateStreamCreateInfoEXT";
      case StructureType::eCuModuleCreateInfoNVX: return "CuModuleCreateInfoNVX";
      case StructureType::eCuFunctionCreateInfoNVX: return "CuFunctionCreateInfoNVX";
      case StructureType::eCuLaunchInfoNVX: return "CuLaunchInfoNVX";
      case StructureType::eImageViewHandleInfoNVX: return "ImageViewHandleInfoNVX";
      case StructureType::eImageViewAddressPropertiesNVX: return "ImageViewAddressPropertiesNVX";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case StructureType::eVideoEncodeH264CapabilitiesEXT: return "VideoEncodeH264CapabilitiesEXT";
      case StructureType::eVideoEncodeH264SessionCreateInfoEXT: return "VideoEncodeH264SessionCreateInfoEXT";
      case StructureType::eVideoEncodeH264SessionParametersCreateInfoEXT:
        return "VideoEncodeH264SessionParametersCreateInfoEXT";
      case StructureType::eVideoEncodeH264SessionParametersAddInfoEXT:
        return "VideoEncodeH264SessionParametersAddInfoEXT";
      case StructureType::eVideoEncodeH264VclFrameInfoEXT: return "VideoEncodeH264VclFrameInfoEXT";
      case StructureType::eVideoEncodeH264DpbSlotInfoEXT: return "VideoEncodeH264DpbSlotInfoEXT";
      case StructureType::eVideoEncodeH264NaluSliceEXT: return "VideoEncodeH264NaluSliceEXT";
      case StructureType::eVideoEncodeH264EmitPictureParametersEXT: return "VideoEncodeH264EmitPictureParametersEXT";
      case StructureType::eVideoEncodeH264ProfileEXT: return "VideoEncodeH264ProfileEXT";
      case StructureType::eVideoDecodeH264CapabilitiesEXT: return "VideoDecodeH264CapabilitiesEXT";
      case StructureType::eVideoDecodeH264SessionCreateInfoEXT: return "VideoDecodeH264SessionCreateInfoEXT";
      case StructureType::eVideoDecodeH264PictureInfoEXT: return "VideoDecodeH264PictureInfoEXT";
      case StructureType::eVideoDecodeH264MvcEXT: return "VideoDecodeH264MvcEXT";
      case StructureType::eVideoDecodeH264ProfileEXT: return "VideoDecodeH264ProfileEXT";
      case StructureType::eVideoDecodeH264SessionParametersCreateInfoEXT:
        return "VideoDecodeH264SessionParametersCreateInfoEXT";
      case StructureType::eVideoDecodeH264SessionParametersAddInfoEXT:
        return "VideoDecodeH264SessionParametersAddInfoEXT";
      case StructureType::eVideoDecodeH264DpbSlotInfoEXT: return "VideoDecodeH264DpbSlotInfoEXT";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      case StructureType::eTextureLodGatherFormatPropertiesAMD: return "TextureLodGatherFormatPropertiesAMD";
#if defined( VK_USE_PLATFORM_GGP )
      case StructureType::eStreamDescriptorSurfaceCreateInfoGGP: return "StreamDescriptorSurfaceCreateInfoGGP";
#endif /*VK_USE_PLATFORM_GGP*/
      case StructureType::ePhysicalDeviceCornerSampledImageFeaturesNV:
        return "PhysicalDeviceCornerSampledImageFeaturesNV";
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
      case StructureType::ePhysicalDeviceTextureCompressionAstcHdrFeaturesEXT:
        return "PhysicalDeviceTextureCompressionAstcHdrFeaturesEXT";
      case StructureType::eImageViewAstcDecodeModeEXT: return "ImageViewAstcDecodeModeEXT";
      case StructureType::ePhysicalDeviceAstcDecodeFeaturesEXT: return "PhysicalDeviceAstcDecodeFeaturesEXT";
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
      case StructureType::ePhysicalDevicePushDescriptorPropertiesKHR:
        return "PhysicalDevicePushDescriptorPropertiesKHR";
      case StructureType::eCommandBufferInheritanceConditionalRenderingInfoEXT:
        return "CommandBufferInheritanceConditionalRenderingInfoEXT";
      case StructureType::ePhysicalDeviceConditionalRenderingFeaturesEXT:
        return "PhysicalDeviceConditionalRenderingFeaturesEXT";
      case StructureType::eConditionalRenderingBeginInfoEXT: return "ConditionalRenderingBeginInfoEXT";
      case StructureType::ePresentRegionsKHR: return "PresentRegionsKHR";
      case StructureType::ePipelineViewportWScalingStateCreateInfoNV:
        return "PipelineViewportWScalingStateCreateInfoNV";
      case StructureType::eSurfaceCapabilities2EXT: return "SurfaceCapabilities2EXT";
      case StructureType::eDisplayPowerInfoEXT: return "DisplayPowerInfoEXT";
      case StructureType::eDeviceEventInfoEXT: return "DeviceEventInfoEXT";
      case StructureType::eDisplayEventInfoEXT: return "DisplayEventInfoEXT";
      case StructureType::eSwapchainCounterCreateInfoEXT: return "SwapchainCounterCreateInfoEXT";
      case StructureType::ePresentTimesInfoGOOGLE: return "PresentTimesInfoGOOGLE";
      case StructureType::ePhysicalDeviceMultiviewPerViewAttributesPropertiesNVX:
        return "PhysicalDeviceMultiviewPerViewAttributesPropertiesNVX";
      case StructureType::ePipelineViewportSwizzleStateCreateInfoNV: return "PipelineViewportSwizzleStateCreateInfoNV";
      case StructureType::ePhysicalDeviceDiscardRectanglePropertiesEXT:
        return "PhysicalDeviceDiscardRectanglePropertiesEXT";
      case StructureType::ePipelineDiscardRectangleStateCreateInfoEXT:
        return "PipelineDiscardRectangleStateCreateInfoEXT";
      case StructureType::ePhysicalDeviceConservativeRasterizationPropertiesEXT:
        return "PhysicalDeviceConservativeRasterizationPropertiesEXT";
      case StructureType::ePipelineRasterizationConservativeStateCreateInfoEXT:
        return "PipelineRasterizationConservativeStateCreateInfoEXT";
      case StructureType::ePhysicalDeviceDepthClipEnableFeaturesEXT: return "PhysicalDeviceDepthClipEnableFeaturesEXT";
      case StructureType::ePipelineRasterizationDepthClipStateCreateInfoEXT:
        return "PipelineRasterizationDepthClipStateCreateInfoEXT";
      case StructureType::eHdrMetadataEXT: return "HdrMetadataEXT";
      case StructureType::eSharedPresentSurfaceCapabilitiesKHR: return "SharedPresentSurfaceCapabilitiesKHR";
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      case StructureType::eImportFenceWin32HandleInfoKHR: return "ImportFenceWin32HandleInfoKHR";
      case StructureType::eExportFenceWin32HandleInfoKHR: return "ExportFenceWin32HandleInfoKHR";
      case StructureType::eFenceGetWin32HandleInfoKHR: return "FenceGetWin32HandleInfoKHR";
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      case StructureType::eImportFenceFdInfoKHR: return "ImportFenceFdInfoKHR";
      case StructureType::eFenceGetFdInfoKHR: return "FenceGetFdInfoKHR";
      case StructureType::ePhysicalDevicePerformanceQueryFeaturesKHR:
        return "PhysicalDevicePerformanceQueryFeaturesKHR";
      case StructureType::ePhysicalDevicePerformanceQueryPropertiesKHR:
        return "PhysicalDevicePerformanceQueryPropertiesKHR";
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
      case StructureType::eAndroidHardwareBufferFormatPropertiesANDROID:
        return "AndroidHardwareBufferFormatPropertiesANDROID";
      case StructureType::eImportAndroidHardwareBufferInfoANDROID: return "ImportAndroidHardwareBufferInfoANDROID";
      case StructureType::eMemoryGetAndroidHardwareBufferInfoANDROID:
        return "MemoryGetAndroidHardwareBufferInfoANDROID";
      case StructureType::eExternalFormatANDROID: return "ExternalFormatANDROID";
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
      case StructureType::ePhysicalDeviceInlineUniformBlockFeaturesEXT:
        return "PhysicalDeviceInlineUniformBlockFeaturesEXT";
      case StructureType::ePhysicalDeviceInlineUniformBlockPropertiesEXT:
        return "PhysicalDeviceInlineUniformBlockPropertiesEXT";
      case StructureType::eWriteDescriptorSetInlineUniformBlockEXT: return "WriteDescriptorSetInlineUniformBlockEXT";
      case StructureType::eDescriptorPoolInlineUniformBlockCreateInfoEXT:
        return "DescriptorPoolInlineUniformBlockCreateInfoEXT";
      case StructureType::eSampleLocationsInfoEXT: return "SampleLocationsInfoEXT";
      case StructureType::eRenderPassSampleLocationsBeginInfoEXT: return "RenderPassSampleLocationsBeginInfoEXT";
      case StructureType::ePipelineSampleLocationsStateCreateInfoEXT:
        return "PipelineSampleLocationsStateCreateInfoEXT";
      case StructureType::ePhysicalDeviceSampleLocationsPropertiesEXT:
        return "PhysicalDeviceSampleLocationsPropertiesEXT";
      case StructureType::eMultisamplePropertiesEXT: return "MultisamplePropertiesEXT";
      case StructureType::ePhysicalDeviceBlendOperationAdvancedFeaturesEXT:
        return "PhysicalDeviceBlendOperationAdvancedFeaturesEXT";
      case StructureType::ePhysicalDeviceBlendOperationAdvancedPropertiesEXT:
        return "PhysicalDeviceBlendOperationAdvancedPropertiesEXT";
      case StructureType::ePipelineColorBlendAdvancedStateCreateInfoEXT:
        return "PipelineColorBlendAdvancedStateCreateInfoEXT";
      case StructureType::ePipelineCoverageToColorStateCreateInfoNV: return "PipelineCoverageToColorStateCreateInfoNV";
      case StructureType::eWriteDescriptorSetAccelerationStructureKHR:
        return "WriteDescriptorSetAccelerationStructureKHR";
      case StructureType::eAccelerationStructureBuildGeometryInfoKHR:
        return "AccelerationStructureBuildGeometryInfoKHR";
      case StructureType::eAccelerationStructureDeviceAddressInfoKHR:
        return "AccelerationStructureDeviceAddressInfoKHR";
      case StructureType::eAccelerationStructureGeometryAabbsDataKHR:
        return "AccelerationStructureGeometryAabbsDataKHR";
      case StructureType::eAccelerationStructureGeometryInstancesDataKHR:
        return "AccelerationStructureGeometryInstancesDataKHR";
      case StructureType::eAccelerationStructureGeometryTrianglesDataKHR:
        return "AccelerationStructureGeometryTrianglesDataKHR";
      case StructureType::eAccelerationStructureGeometryKHR: return "AccelerationStructureGeometryKHR";
      case StructureType::eAccelerationStructureVersionInfoKHR: return "AccelerationStructureVersionInfoKHR";
      case StructureType::eCopyAccelerationStructureInfoKHR: return "CopyAccelerationStructureInfoKHR";
      case StructureType::eCopyAccelerationStructureToMemoryInfoKHR: return "CopyAccelerationStructureToMemoryInfoKHR";
      case StructureType::eCopyMemoryToAccelerationStructureInfoKHR: return "CopyMemoryToAccelerationStructureInfoKHR";
      case StructureType::ePhysicalDeviceAccelerationStructureFeaturesKHR:
        return "PhysicalDeviceAccelerationStructureFeaturesKHR";
      case StructureType::ePhysicalDeviceAccelerationStructurePropertiesKHR:
        return "PhysicalDeviceAccelerationStructurePropertiesKHR";
      case StructureType::eAccelerationStructureCreateInfoKHR: return "AccelerationStructureCreateInfoKHR";
      case StructureType::eAccelerationStructureBuildSizesInfoKHR: return "AccelerationStructureBuildSizesInfoKHR";
      case StructureType::ePhysicalDeviceRayTracingPipelineFeaturesKHR:
        return "PhysicalDeviceRayTracingPipelineFeaturesKHR";
      case StructureType::ePhysicalDeviceRayTracingPipelinePropertiesKHR:
        return "PhysicalDeviceRayTracingPipelinePropertiesKHR";
      case StructureType::eRayTracingPipelineCreateInfoKHR: return "RayTracingPipelineCreateInfoKHR";
      case StructureType::eRayTracingShaderGroupCreateInfoKHR: return "RayTracingShaderGroupCreateInfoKHR";
      case StructureType::eRayTracingPipelineInterfaceCreateInfoKHR: return "RayTracingPipelineInterfaceCreateInfoKHR";
      case StructureType::ePhysicalDeviceRayQueryFeaturesKHR: return "PhysicalDeviceRayQueryFeaturesKHR";
      case StructureType::ePipelineCoverageModulationStateCreateInfoNV:
        return "PipelineCoverageModulationStateCreateInfoNV";
      case StructureType::ePhysicalDeviceShaderSmBuiltinsFeaturesNV: return "PhysicalDeviceShaderSmBuiltinsFeaturesNV";
      case StructureType::ePhysicalDeviceShaderSmBuiltinsPropertiesNV:
        return "PhysicalDeviceShaderSmBuiltinsPropertiesNV";
      case StructureType::eDrmFormatModifierPropertiesListEXT: return "DrmFormatModifierPropertiesListEXT";
      case StructureType::ePhysicalDeviceImageDrmFormatModifierInfoEXT:
        return "PhysicalDeviceImageDrmFormatModifierInfoEXT";
      case StructureType::eImageDrmFormatModifierListCreateInfoEXT: return "ImageDrmFormatModifierListCreateInfoEXT";
      case StructureType::eImageDrmFormatModifierExplicitCreateInfoEXT:
        return "ImageDrmFormatModifierExplicitCreateInfoEXT";
      case StructureType::eImageDrmFormatModifierPropertiesEXT: return "ImageDrmFormatModifierPropertiesEXT";
      case StructureType::eValidationCacheCreateInfoEXT: return "ValidationCacheCreateInfoEXT";
      case StructureType::eShaderModuleValidationCacheCreateInfoEXT: return "ShaderModuleValidationCacheCreateInfoEXT";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case StructureType::ePhysicalDevicePortabilitySubsetFeaturesKHR:
        return "PhysicalDevicePortabilitySubsetFeaturesKHR";
      case StructureType::ePhysicalDevicePortabilitySubsetPropertiesKHR:
        return "PhysicalDevicePortabilitySubsetPropertiesKHR";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      case StructureType::ePipelineViewportShadingRateImageStateCreateInfoNV:
        return "PipelineViewportShadingRateImageStateCreateInfoNV";
      case StructureType::ePhysicalDeviceShadingRateImageFeaturesNV: return "PhysicalDeviceShadingRateImageFeaturesNV";
      case StructureType::ePhysicalDeviceShadingRateImagePropertiesNV:
        return "PhysicalDeviceShadingRateImagePropertiesNV";
      case StructureType::ePipelineViewportCoarseSampleOrderStateCreateInfoNV:
        return "PipelineViewportCoarseSampleOrderStateCreateInfoNV";
      case StructureType::eRayTracingPipelineCreateInfoNV: return "RayTracingPipelineCreateInfoNV";
      case StructureType::eAccelerationStructureCreateInfoNV: return "AccelerationStructureCreateInfoNV";
      case StructureType::eGeometryNV: return "GeometryNV";
      case StructureType::eGeometryTrianglesNV: return "GeometryTrianglesNV";
      case StructureType::eGeometryAabbNV: return "GeometryAabbNV";
      case StructureType::eBindAccelerationStructureMemoryInfoNV: return "BindAccelerationStructureMemoryInfoNV";
      case StructureType::eWriteDescriptorSetAccelerationStructureNV:
        return "WriteDescriptorSetAccelerationStructureNV";
      case StructureType::eAccelerationStructureMemoryRequirementsInfoNV:
        return "AccelerationStructureMemoryRequirementsInfoNV";
      case StructureType::ePhysicalDeviceRayTracingPropertiesNV: return "PhysicalDeviceRayTracingPropertiesNV";
      case StructureType::eRayTracingShaderGroupCreateInfoNV: return "RayTracingShaderGroupCreateInfoNV";
      case StructureType::eAccelerationStructureInfoNV: return "AccelerationStructureInfoNV";
      case StructureType::ePhysicalDeviceRepresentativeFragmentTestFeaturesNV:
        return "PhysicalDeviceRepresentativeFragmentTestFeaturesNV";
      case StructureType::ePipelineRepresentativeFragmentTestStateCreateInfoNV:
        return "PipelineRepresentativeFragmentTestStateCreateInfoNV";
      case StructureType::ePhysicalDeviceImageViewImageFormatInfoEXT:
        return "PhysicalDeviceImageViewImageFormatInfoEXT";
      case StructureType::eFilterCubicImageViewImageFormatPropertiesEXT:
        return "FilterCubicImageViewImageFormatPropertiesEXT";
      case StructureType::eDeviceQueueGlobalPriorityCreateInfoEXT: return "DeviceQueueGlobalPriorityCreateInfoEXT";
      case StructureType::eImportMemoryHostPointerInfoEXT: return "ImportMemoryHostPointerInfoEXT";
      case StructureType::eMemoryHostPointerPropertiesEXT: return "MemoryHostPointerPropertiesEXT";
      case StructureType::ePhysicalDeviceExternalMemoryHostPropertiesEXT:
        return "PhysicalDeviceExternalMemoryHostPropertiesEXT";
      case StructureType::ePhysicalDeviceShaderClockFeaturesKHR: return "PhysicalDeviceShaderClockFeaturesKHR";
      case StructureType::ePipelineCompilerControlCreateInfoAMD: return "PipelineCompilerControlCreateInfoAMD";
      case StructureType::eCalibratedTimestampInfoEXT: return "CalibratedTimestampInfoEXT";
      case StructureType::ePhysicalDeviceShaderCorePropertiesAMD: return "PhysicalDeviceShaderCorePropertiesAMD";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case StructureType::eVideoDecodeH265CapabilitiesEXT: return "VideoDecodeH265CapabilitiesEXT";
      case StructureType::eVideoDecodeH265SessionCreateInfoEXT: return "VideoDecodeH265SessionCreateInfoEXT";
      case StructureType::eVideoDecodeH265SessionParametersCreateInfoEXT:
        return "VideoDecodeH265SessionParametersCreateInfoEXT";
      case StructureType::eVideoDecodeH265SessionParametersAddInfoEXT:
        return "VideoDecodeH265SessionParametersAddInfoEXT";
      case StructureType::eVideoDecodeH265ProfileEXT: return "VideoDecodeH265ProfileEXT";
      case StructureType::eVideoDecodeH265PictureInfoEXT: return "VideoDecodeH265PictureInfoEXT";
      case StructureType::eVideoDecodeH265DpbSlotInfoEXT: return "VideoDecodeH265DpbSlotInfoEXT";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      case StructureType::eDeviceMemoryOverallocationCreateInfoAMD: return "DeviceMemoryOverallocationCreateInfoAMD";
      case StructureType::ePhysicalDeviceVertexAttributeDivisorPropertiesEXT:
        return "PhysicalDeviceVertexAttributeDivisorPropertiesEXT";
      case StructureType::ePipelineVertexInputDivisorStateCreateInfoEXT:
        return "PipelineVertexInputDivisorStateCreateInfoEXT";
      case StructureType::ePhysicalDeviceVertexAttributeDivisorFeaturesEXT:
        return "PhysicalDeviceVertexAttributeDivisorFeaturesEXT";
#if defined( VK_USE_PLATFORM_GGP )
      case StructureType::ePresentFrameTokenGGP: return "PresentFrameTokenGGP";
#endif /*VK_USE_PLATFORM_GGP*/
      case StructureType::ePipelineCreationFeedbackCreateInfoEXT: return "PipelineCreationFeedbackCreateInfoEXT";
      case StructureType::ePhysicalDeviceComputeShaderDerivativesFeaturesNV:
        return "PhysicalDeviceComputeShaderDerivativesFeaturesNV";
      case StructureType::ePhysicalDeviceMeshShaderFeaturesNV: return "PhysicalDeviceMeshShaderFeaturesNV";
      case StructureType::ePhysicalDeviceMeshShaderPropertiesNV: return "PhysicalDeviceMeshShaderPropertiesNV";
      case StructureType::ePhysicalDeviceFragmentShaderBarycentricFeaturesNV:
        return "PhysicalDeviceFragmentShaderBarycentricFeaturesNV";
      case StructureType::ePhysicalDeviceShaderImageFootprintFeaturesNV:
        return "PhysicalDeviceShaderImageFootprintFeaturesNV";
      case StructureType::ePipelineViewportExclusiveScissorStateCreateInfoNV:
        return "PipelineViewportExclusiveScissorStateCreateInfoNV";
      case StructureType::ePhysicalDeviceExclusiveScissorFeaturesNV: return "PhysicalDeviceExclusiveScissorFeaturesNV";
      case StructureType::eCheckpointDataNV: return "CheckpointDataNV";
      case StructureType::eQueueFamilyCheckpointPropertiesNV: return "QueueFamilyCheckpointPropertiesNV";
      case StructureType::ePhysicalDeviceShaderIntegerFunctions2FeaturesINTEL:
        return "PhysicalDeviceShaderIntegerFunctions2FeaturesINTEL";
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
      case StructureType::ePhysicalDeviceShaderTerminateInvocationFeaturesKHR:
        return "PhysicalDeviceShaderTerminateInvocationFeaturesKHR";
#if defined( VK_USE_PLATFORM_METAL_EXT )
      case StructureType::eMetalSurfaceCreateInfoEXT: return "MetalSurfaceCreateInfoEXT";
#endif /*VK_USE_PLATFORM_METAL_EXT*/
      case StructureType::ePhysicalDeviceFragmentDensityMapFeaturesEXT:
        return "PhysicalDeviceFragmentDensityMapFeaturesEXT";
      case StructureType::ePhysicalDeviceFragmentDensityMapPropertiesEXT:
        return "PhysicalDeviceFragmentDensityMapPropertiesEXT";
      case StructureType::eRenderPassFragmentDensityMapCreateInfoEXT:
        return "RenderPassFragmentDensityMapCreateInfoEXT";
      case StructureType::ePhysicalDeviceSubgroupSizeControlPropertiesEXT:
        return "PhysicalDeviceSubgroupSizeControlPropertiesEXT";
      case StructureType::ePipelineShaderStageRequiredSubgroupSizeCreateInfoEXT:
        return "PipelineShaderStageRequiredSubgroupSizeCreateInfoEXT";
      case StructureType::ePhysicalDeviceSubgroupSizeControlFeaturesEXT:
        return "PhysicalDeviceSubgroupSizeControlFeaturesEXT";
      case StructureType::eFragmentShadingRateAttachmentInfoKHR: return "FragmentShadingRateAttachmentInfoKHR";
      case StructureType::ePipelineFragmentShadingRateStateCreateInfoKHR:
        return "PipelineFragmentShadingRateStateCreateInfoKHR";
      case StructureType::ePhysicalDeviceFragmentShadingRatePropertiesKHR:
        return "PhysicalDeviceFragmentShadingRatePropertiesKHR";
      case StructureType::ePhysicalDeviceFragmentShadingRateFeaturesKHR:
        return "PhysicalDeviceFragmentShadingRateFeaturesKHR";
      case StructureType::ePhysicalDeviceFragmentShadingRateKHR: return "PhysicalDeviceFragmentShadingRateKHR";
      case StructureType::ePhysicalDeviceShaderCoreProperties2AMD: return "PhysicalDeviceShaderCoreProperties2AMD";
      case StructureType::ePhysicalDeviceCoherentMemoryFeaturesAMD: return "PhysicalDeviceCoherentMemoryFeaturesAMD";
      case StructureType::ePhysicalDeviceShaderImageAtomicInt64FeaturesEXT:
        return "PhysicalDeviceShaderImageAtomicInt64FeaturesEXT";
      case StructureType::ePhysicalDeviceMemoryBudgetPropertiesEXT: return "PhysicalDeviceMemoryBudgetPropertiesEXT";
      case StructureType::ePhysicalDeviceMemoryPriorityFeaturesEXT: return "PhysicalDeviceMemoryPriorityFeaturesEXT";
      case StructureType::eMemoryPriorityAllocateInfoEXT: return "MemoryPriorityAllocateInfoEXT";
      case StructureType::eSurfaceProtectedCapabilitiesKHR: return "SurfaceProtectedCapabilitiesKHR";
      case StructureType::ePhysicalDeviceDedicatedAllocationImageAliasingFeaturesNV:
        return "PhysicalDeviceDedicatedAllocationImageAliasingFeaturesNV";
      case StructureType::ePhysicalDeviceBufferDeviceAddressFeaturesEXT:
        return "PhysicalDeviceBufferDeviceAddressFeaturesEXT";
      case StructureType::eBufferDeviceAddressCreateInfoEXT: return "BufferDeviceAddressCreateInfoEXT";
      case StructureType::ePhysicalDeviceToolPropertiesEXT: return "PhysicalDeviceToolPropertiesEXT";
      case StructureType::eValidationFeaturesEXT: return "ValidationFeaturesEXT";
      case StructureType::ePhysicalDeviceCooperativeMatrixFeaturesNV:
        return "PhysicalDeviceCooperativeMatrixFeaturesNV";
      case StructureType::eCooperativeMatrixPropertiesNV: return "CooperativeMatrixPropertiesNV";
      case StructureType::ePhysicalDeviceCooperativeMatrixPropertiesNV:
        return "PhysicalDeviceCooperativeMatrixPropertiesNV";
      case StructureType::ePhysicalDeviceCoverageReductionModeFeaturesNV:
        return "PhysicalDeviceCoverageReductionModeFeaturesNV";
      case StructureType::ePipelineCoverageReductionStateCreateInfoNV:
        return "PipelineCoverageReductionStateCreateInfoNV";
      case StructureType::eFramebufferMixedSamplesCombinationNV: return "FramebufferMixedSamplesCombinationNV";
      case StructureType::ePhysicalDeviceFragmentShaderInterlockFeaturesEXT:
        return "PhysicalDeviceFragmentShaderInterlockFeaturesEXT";
      case StructureType::ePhysicalDeviceYcbcrImageArraysFeaturesEXT:
        return "PhysicalDeviceYcbcrImageArraysFeaturesEXT";
      case StructureType::ePhysicalDeviceProvokingVertexFeaturesEXT: return "PhysicalDeviceProvokingVertexFeaturesEXT";
      case StructureType::ePipelineRasterizationProvokingVertexStateCreateInfoEXT:
        return "PipelineRasterizationProvokingVertexStateCreateInfoEXT";
      case StructureType::ePhysicalDeviceProvokingVertexPropertiesEXT:
        return "PhysicalDeviceProvokingVertexPropertiesEXT";
#if defined( VK_USE_PLATFORM_WIN32_KHR )
      case StructureType::eSurfaceFullScreenExclusiveInfoEXT: return "SurfaceFullScreenExclusiveInfoEXT";
      case StructureType::eSurfaceCapabilitiesFullScreenExclusiveEXT:
        return "SurfaceCapabilitiesFullScreenExclusiveEXT";
      case StructureType::eSurfaceFullScreenExclusiveWin32InfoEXT: return "SurfaceFullScreenExclusiveWin32InfoEXT";
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
      case StructureType::eHeadlessSurfaceCreateInfoEXT: return "HeadlessSurfaceCreateInfoEXT";
      case StructureType::ePhysicalDeviceLineRasterizationFeaturesEXT:
        return "PhysicalDeviceLineRasterizationFeaturesEXT";
      case StructureType::ePipelineRasterizationLineStateCreateInfoEXT:
        return "PipelineRasterizationLineStateCreateInfoEXT";
      case StructureType::ePhysicalDeviceLineRasterizationPropertiesEXT:
        return "PhysicalDeviceLineRasterizationPropertiesEXT";
      case StructureType::ePhysicalDeviceShaderAtomicFloatFeaturesEXT:
        return "PhysicalDeviceShaderAtomicFloatFeaturesEXT";
      case StructureType::ePhysicalDeviceIndexTypeUint8FeaturesEXT: return "PhysicalDeviceIndexTypeUint8FeaturesEXT";
      case StructureType::ePhysicalDeviceExtendedDynamicStateFeaturesEXT:
        return "PhysicalDeviceExtendedDynamicStateFeaturesEXT";
      case StructureType::ePhysicalDevicePipelineExecutablePropertiesFeaturesKHR:
        return "PhysicalDevicePipelineExecutablePropertiesFeaturesKHR";
      case StructureType::ePipelineInfoKHR: return "PipelineInfoKHR";
      case StructureType::ePipelineExecutablePropertiesKHR: return "PipelineExecutablePropertiesKHR";
      case StructureType::ePipelineExecutableInfoKHR: return "PipelineExecutableInfoKHR";
      case StructureType::ePipelineExecutableStatisticKHR: return "PipelineExecutableStatisticKHR";
      case StructureType::ePipelineExecutableInternalRepresentationKHR:
        return "PipelineExecutableInternalRepresentationKHR";
      case StructureType::ePhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT:
        return "PhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT";
      case StructureType::ePhysicalDeviceDeviceGeneratedCommandsPropertiesNV:
        return "PhysicalDeviceDeviceGeneratedCommandsPropertiesNV";
      case StructureType::eGraphicsShaderGroupCreateInfoNV: return "GraphicsShaderGroupCreateInfoNV";
      case StructureType::eGraphicsPipelineShaderGroupsCreateInfoNV: return "GraphicsPipelineShaderGroupsCreateInfoNV";
      case StructureType::eIndirectCommandsLayoutTokenNV: return "IndirectCommandsLayoutTokenNV";
      case StructureType::eIndirectCommandsLayoutCreateInfoNV: return "IndirectCommandsLayoutCreateInfoNV";
      case StructureType::eGeneratedCommandsInfoNV: return "GeneratedCommandsInfoNV";
      case StructureType::eGeneratedCommandsMemoryRequirementsInfoNV:
        return "GeneratedCommandsMemoryRequirementsInfoNV";
      case StructureType::ePhysicalDeviceDeviceGeneratedCommandsFeaturesNV:
        return "PhysicalDeviceDeviceGeneratedCommandsFeaturesNV";
      case StructureType::ePhysicalDeviceInheritedViewportScissorFeaturesNV:
        return "PhysicalDeviceInheritedViewportScissorFeaturesNV";
      case StructureType::eCommandBufferInheritanceViewportScissorInfoNV:
        return "CommandBufferInheritanceViewportScissorInfoNV";
      case StructureType::ePhysicalDeviceTexelBufferAlignmentFeaturesEXT:
        return "PhysicalDeviceTexelBufferAlignmentFeaturesEXT";
      case StructureType::ePhysicalDeviceTexelBufferAlignmentPropertiesEXT:
        return "PhysicalDeviceTexelBufferAlignmentPropertiesEXT";
      case StructureType::eCommandBufferInheritanceRenderPassTransformInfoQCOM:
        return "CommandBufferInheritanceRenderPassTransformInfoQCOM";
      case StructureType::eRenderPassTransformBeginInfoQCOM: return "RenderPassTransformBeginInfoQCOM";
      case StructureType::ePhysicalDeviceDeviceMemoryReportFeaturesEXT:
        return "PhysicalDeviceDeviceMemoryReportFeaturesEXT";
      case StructureType::eDeviceDeviceMemoryReportCreateInfoEXT: return "DeviceDeviceMemoryReportCreateInfoEXT";
      case StructureType::eDeviceMemoryReportCallbackDataEXT: return "DeviceMemoryReportCallbackDataEXT";
      case StructureType::ePhysicalDeviceRobustness2FeaturesEXT: return "PhysicalDeviceRobustness2FeaturesEXT";
      case StructureType::ePhysicalDeviceRobustness2PropertiesEXT: return "PhysicalDeviceRobustness2PropertiesEXT";
      case StructureType::eSamplerCustomBorderColorCreateInfoEXT: return "SamplerCustomBorderColorCreateInfoEXT";
      case StructureType::ePhysicalDeviceCustomBorderColorPropertiesEXT:
        return "PhysicalDeviceCustomBorderColorPropertiesEXT";
      case StructureType::ePhysicalDeviceCustomBorderColorFeaturesEXT:
        return "PhysicalDeviceCustomBorderColorFeaturesEXT";
      case StructureType::ePipelineLibraryCreateInfoKHR: return "PipelineLibraryCreateInfoKHR";
      case StructureType::ePhysicalDevicePrivateDataFeaturesEXT: return "PhysicalDevicePrivateDataFeaturesEXT";
      case StructureType::eDevicePrivateDataCreateInfoEXT: return "DevicePrivateDataCreateInfoEXT";
      case StructureType::ePrivateDataSlotCreateInfoEXT: return "PrivateDataSlotCreateInfoEXT";
      case StructureType::ePhysicalDevicePipelineCreationCacheControlFeaturesEXT:
        return "PhysicalDevicePipelineCreationCacheControlFeaturesEXT";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case StructureType::eVideoEncodeInfoKHR: return "VideoEncodeInfoKHR";
      case StructureType::eVideoEncodeRateControlInfoKHR: return "VideoEncodeRateControlInfoKHR";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      case StructureType::ePhysicalDeviceDiagnosticsConfigFeaturesNV:
        return "PhysicalDeviceDiagnosticsConfigFeaturesNV";
      case StructureType::eDeviceDiagnosticsConfigCreateInfoNV: return "DeviceDiagnosticsConfigCreateInfoNV";
      case StructureType::eMemoryBarrier2KHR: return "MemoryBarrier2KHR";
      case StructureType::eBufferMemoryBarrier2KHR: return "BufferMemoryBarrier2KHR";
      case StructureType::eImageMemoryBarrier2KHR: return "ImageMemoryBarrier2KHR";
      case StructureType::eDependencyInfoKHR: return "DependencyInfoKHR";
      case StructureType::eSubmitInfo2KHR: return "SubmitInfo2KHR";
      case StructureType::eSemaphoreSubmitInfoKHR: return "SemaphoreSubmitInfoKHR";
      case StructureType::eCommandBufferSubmitInfoKHR: return "CommandBufferSubmitInfoKHR";
      case StructureType::ePhysicalDeviceSynchronization2FeaturesKHR:
        return "PhysicalDeviceSynchronization2FeaturesKHR";
      case StructureType::eQueueFamilyCheckpointProperties2NV: return "QueueFamilyCheckpointProperties2NV";
      case StructureType::eCheckpointData2NV: return "CheckpointData2NV";
      case StructureType::ePhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR:
        return "PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR";
      case StructureType::ePhysicalDeviceZeroInitializeWorkgroupMemoryFeaturesKHR:
        return "PhysicalDeviceZeroInitializeWorkgroupMemoryFeaturesKHR";
      case StructureType::ePhysicalDeviceFragmentShadingRateEnumsPropertiesNV:
        return "PhysicalDeviceFragmentShadingRateEnumsPropertiesNV";
      case StructureType::ePhysicalDeviceFragmentShadingRateEnumsFeaturesNV:
        return "PhysicalDeviceFragmentShadingRateEnumsFeaturesNV";
      case StructureType::ePipelineFragmentShadingRateEnumStateCreateInfoNV:
        return "PipelineFragmentShadingRateEnumStateCreateInfoNV";
      case StructureType::eAccelerationStructureGeometryMotionTrianglesDataNV:
        return "AccelerationStructureGeometryMotionTrianglesDataNV";
      case StructureType::ePhysicalDeviceRayTracingMotionBlurFeaturesNV:
        return "PhysicalDeviceRayTracingMotionBlurFeaturesNV";
      case StructureType::eAccelerationStructureMotionInfoNV: return "AccelerationStructureMotionInfoNV";
      case StructureType::ePhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT:
        return "PhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT";
      case StructureType::ePhysicalDeviceFragmentDensityMap2FeaturesEXT:
        return "PhysicalDeviceFragmentDensityMap2FeaturesEXT";
      case StructureType::ePhysicalDeviceFragmentDensityMap2PropertiesEXT:
        return "PhysicalDeviceFragmentDensityMap2PropertiesEXT";
      case StructureType::eCopyCommandTransformInfoQCOM: return "CopyCommandTransformInfoQCOM";
      case StructureType::ePhysicalDeviceImageRobustnessFeaturesEXT: return "PhysicalDeviceImageRobustnessFeaturesEXT";
      case StructureType::ePhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR:
        return "PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR";
      case StructureType::eCopyBufferInfo2KHR: return "CopyBufferInfo2KHR";
      case StructureType::eCopyImageInfo2KHR: return "CopyImageInfo2KHR";
      case StructureType::eCopyBufferToImageInfo2KHR: return "CopyBufferToImageInfo2KHR";
      case StructureType::eCopyImageToBufferInfo2KHR: return "CopyImageToBufferInfo2KHR";
      case StructureType::eBlitImageInfo2KHR: return "BlitImageInfo2KHR";
      case StructureType::eResolveImageInfo2KHR: return "ResolveImageInfo2KHR";
      case StructureType::eBufferCopy2KHR: return "BufferCopy2KHR";
      case StructureType::eImageCopy2KHR: return "ImageCopy2KHR";
      case StructureType::eImageBlit2KHR: return "ImageBlit2KHR";
      case StructureType::eBufferImageCopy2KHR: return "BufferImageCopy2KHR";
      case StructureType::eImageResolve2KHR: return "ImageResolve2KHR";
      case StructureType::ePhysicalDevice4444FormatsFeaturesEXT: return "PhysicalDevice4444FormatsFeaturesEXT";
#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
      case StructureType::eDirectfbSurfaceCreateInfoEXT: return "DirectfbSurfaceCreateInfoEXT";
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/
      case StructureType::ePhysicalDeviceMutableDescriptorTypeFeaturesVALVE:
        return "PhysicalDeviceMutableDescriptorTypeFeaturesVALVE";
      case StructureType::eMutableDescriptorTypeCreateInfoVALVE: return "MutableDescriptorTypeCreateInfoVALVE";
      case StructureType::ePhysicalDeviceVertexInputDynamicStateFeaturesEXT:
        return "PhysicalDeviceVertexInputDynamicStateFeaturesEXT";
      case StructureType::eVertexInputBindingDescription2EXT: return "VertexInputBindingDescription2EXT";
      case StructureType::eVertexInputAttributeDescription2EXT: return "VertexInputAttributeDescription2EXT";
      case StructureType::ePhysicalDeviceDrmPropertiesEXT: return "PhysicalDeviceDrmPropertiesEXT";
#if defined( VK_USE_PLATFORM_FUCHSIA )
      case StructureType::eImportMemoryZirconHandleInfoFUCHSIA: return "ImportMemoryZirconHandleInfoFUCHSIA";
      case StructureType::eMemoryZirconHandlePropertiesFUCHSIA: return "MemoryZirconHandlePropertiesFUCHSIA";
      case StructureType::eMemoryGetZirconHandleInfoFUCHSIA: return "MemoryGetZirconHandleInfoFUCHSIA";
      case StructureType::eImportSemaphoreZirconHandleInfoFUCHSIA: return "ImportSemaphoreZirconHandleInfoFUCHSIA";
      case StructureType::eSemaphoreGetZirconHandleInfoFUCHSIA: return "SemaphoreGetZirconHandleInfoFUCHSIA";
#endif /*VK_USE_PLATFORM_FUCHSIA*/
      case StructureType::eSubpasssShadingPipelineCreateInfoHUAWEI: return "SubpasssShadingPipelineCreateInfoHUAWEI";
      case StructureType::ePhysicalDeviceSubpassShadingFeaturesHUAWEI:
        return "PhysicalDeviceSubpassShadingFeaturesHUAWEI";
      case StructureType::ePhysicalDeviceSubpassShadingPropertiesHUAWEI:
        return "PhysicalDeviceSubpassShadingPropertiesHUAWEI";
      case StructureType::ePhysicalDeviceExtendedDynamicState2FeaturesEXT:
        return "PhysicalDeviceExtendedDynamicState2FeaturesEXT";
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
      case StructureType::eScreenSurfaceCreateInfoQNX: return "ScreenSurfaceCreateInfoQNX";
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/
      case StructureType::ePhysicalDeviceColorWriteEnableFeaturesEXT:
        return "PhysicalDeviceColorWriteEnableFeaturesEXT";
      case StructureType::ePipelineColorWriteCreateInfoEXT: return "PipelineColorWriteCreateInfoEXT";
      case StructureType::ePhysicalDeviceGlobalPriorityQueryFeaturesEXT:
        return "PhysicalDeviceGlobalPriorityQueryFeaturesEXT";
      case StructureType::eQueueFamilyGlobalPriorityPropertiesEXT: return "QueueFamilyGlobalPriorityPropertiesEXT";
      case StructureType::ePhysicalDeviceMultiDrawFeaturesEXT: return "PhysicalDeviceMultiDrawFeaturesEXT";
      case StructureType::ePhysicalDeviceMultiDrawPropertiesEXT: return "PhysicalDeviceMultiDrawPropertiesEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class ObjectType
  {
    eUnknown                  = VK_OBJECT_TYPE_UNKNOWN,
    eInstance                 = VK_OBJECT_TYPE_INSTANCE,
    ePhysicalDevice           = VK_OBJECT_TYPE_PHYSICAL_DEVICE,
    eDevice                   = VK_OBJECT_TYPE_DEVICE,
    eQueue                    = VK_OBJECT_TYPE_QUEUE,
    eSemaphore                = VK_OBJECT_TYPE_SEMAPHORE,
    eCommandBuffer            = VK_OBJECT_TYPE_COMMAND_BUFFER,
    eFence                    = VK_OBJECT_TYPE_FENCE,
    eDeviceMemory             = VK_OBJECT_TYPE_DEVICE_MEMORY,
    eBuffer                   = VK_OBJECT_TYPE_BUFFER,
    eImage                    = VK_OBJECT_TYPE_IMAGE,
    eEvent                    = VK_OBJECT_TYPE_EVENT,
    eQueryPool                = VK_OBJECT_TYPE_QUERY_POOL,
    eBufferView               = VK_OBJECT_TYPE_BUFFER_VIEW,
    eImageView                = VK_OBJECT_TYPE_IMAGE_VIEW,
    eShaderModule             = VK_OBJECT_TYPE_SHADER_MODULE,
    ePipelineCache            = VK_OBJECT_TYPE_PIPELINE_CACHE,
    ePipelineLayout           = VK_OBJECT_TYPE_PIPELINE_LAYOUT,
    eRenderPass               = VK_OBJECT_TYPE_RENDER_PASS,
    ePipeline                 = VK_OBJECT_TYPE_PIPELINE,
    eDescriptorSetLayout      = VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
    eSampler                  = VK_OBJECT_TYPE_SAMPLER,
    eDescriptorPool           = VK_OBJECT_TYPE_DESCRIPTOR_POOL,
    eDescriptorSet            = VK_OBJECT_TYPE_DESCRIPTOR_SET,
    eFramebuffer              = VK_OBJECT_TYPE_FRAMEBUFFER,
    eCommandPool              = VK_OBJECT_TYPE_COMMAND_POOL,
    eSamplerYcbcrConversion   = VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION,
    eDescriptorUpdateTemplate = VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE,
    eSurfaceKHR               = VK_OBJECT_TYPE_SURFACE_KHR,
    eSwapchainKHR             = VK_OBJECT_TYPE_SWAPCHAIN_KHR,
    eDisplayKHR               = VK_OBJECT_TYPE_DISPLAY_KHR,
    eDisplayModeKHR           = VK_OBJECT_TYPE_DISPLAY_MODE_KHR,
    eDebugReportCallbackEXT   = VK_OBJECT_TYPE_DEBUG_REPORT_CALLBACK_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoSessionKHR           = VK_OBJECT_TYPE_VIDEO_SESSION_KHR,
    eVideoSessionParametersKHR = VK_OBJECT_TYPE_VIDEO_SESSION_PARAMETERS_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eCuModuleNVX                   = VK_OBJECT_TYPE_CU_MODULE_NVX,
    eCuFunctionNVX                 = VK_OBJECT_TYPE_CU_FUNCTION_NVX,
    eDebugUtilsMessengerEXT        = VK_OBJECT_TYPE_DEBUG_UTILS_MESSENGER_EXT,
    eAccelerationStructureKHR      = VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR,
    eValidationCacheEXT            = VK_OBJECT_TYPE_VALIDATION_CACHE_EXT,
    eAccelerationStructureNV       = VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV,
    ePerformanceConfigurationINTEL = VK_OBJECT_TYPE_PERFORMANCE_CONFIGURATION_INTEL,
    eDeferredOperationKHR          = VK_OBJECT_TYPE_DEFERRED_OPERATION_KHR,
    eIndirectCommandsLayoutNV      = VK_OBJECT_TYPE_INDIRECT_COMMANDS_LAYOUT_NV,
    ePrivateDataSlotEXT            = VK_OBJECT_TYPE_PRIVATE_DATA_SLOT_EXT,
    eDescriptorUpdateTemplateKHR   = VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_KHR,
    eSamplerYcbcrConversionKHR     = VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION_KHR
  };

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
      case ObjectType::eSurfaceKHR: return "SurfaceKHR";
      case ObjectType::eSwapchainKHR: return "SwapchainKHR";
      case ObjectType::eDisplayKHR: return "DisplayKHR";
      case ObjectType::eDisplayModeKHR: return "DisplayModeKHR";
      case ObjectType::eDebugReportCallbackEXT: return "DebugReportCallbackEXT";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case ObjectType::eVideoSessionKHR: return "VideoSessionKHR";
      case ObjectType::eVideoSessionParametersKHR: return "VideoSessionParametersKHR";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      case ObjectType::eCuModuleNVX: return "CuModuleNVX";
      case ObjectType::eCuFunctionNVX: return "CuFunctionNVX";
      case ObjectType::eDebugUtilsMessengerEXT: return "DebugUtilsMessengerEXT";
      case ObjectType::eAccelerationStructureKHR: return "AccelerationStructureKHR";
      case ObjectType::eValidationCacheEXT: return "ValidationCacheEXT";
      case ObjectType::eAccelerationStructureNV: return "AccelerationStructureNV";
      case ObjectType::ePerformanceConfigurationINTEL: return "PerformanceConfigurationINTEL";
      case ObjectType::eDeferredOperationKHR: return "DeferredOperationKHR";
      case ObjectType::eIndirectCommandsLayoutNV: return "IndirectCommandsLayoutNV";
      case ObjectType::ePrivateDataSlotEXT: return "PrivateDataSlotEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class VendorId
  {
    eVIV      = VK_VENDOR_ID_VIV,
    eVSI      = VK_VENDOR_ID_VSI,
    eKazan    = VK_VENDOR_ID_KAZAN,
    eCodeplay = VK_VENDOR_ID_CODEPLAY,
    eMESA     = VK_VENDOR_ID_MESA,
    ePocl     = VK_VENDOR_ID_POCL
  };

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
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class PipelineCacheHeaderVersion
  {
    eOne = VK_PIPELINE_CACHE_HEADER_VERSION_ONE
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineCacheHeaderVersion value )
  {
    switch ( value )
    {
      case PipelineCacheHeaderVersion::eOne: return "One";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class Format
  {
    eUndefined                               = VK_FORMAT_UNDEFINED,
    eR4G4UnormPack8                          = VK_FORMAT_R4G4_UNORM_PACK8,
    eR4G4B4A4UnormPack16                     = VK_FORMAT_R4G4B4A4_UNORM_PACK16,
    eB4G4R4A4UnormPack16                     = VK_FORMAT_B4G4R4A4_UNORM_PACK16,
    eR5G6B5UnormPack16                       = VK_FORMAT_R5G6B5_UNORM_PACK16,
    eB5G6R5UnormPack16                       = VK_FORMAT_B5G6R5_UNORM_PACK16,
    eR5G5B5A1UnormPack16                     = VK_FORMAT_R5G5B5A1_UNORM_PACK16,
    eB5G5R5A1UnormPack16                     = VK_FORMAT_B5G5R5A1_UNORM_PACK16,
    eA1R5G5B5UnormPack16                     = VK_FORMAT_A1R5G5B5_UNORM_PACK16,
    eR8Unorm                                 = VK_FORMAT_R8_UNORM,
    eR8Snorm                                 = VK_FORMAT_R8_SNORM,
    eR8Uscaled                               = VK_FORMAT_R8_USCALED,
    eR8Sscaled                               = VK_FORMAT_R8_SSCALED,
    eR8Uint                                  = VK_FORMAT_R8_UINT,
    eR8Sint                                  = VK_FORMAT_R8_SINT,
    eR8Srgb                                  = VK_FORMAT_R8_SRGB,
    eR8G8Unorm                               = VK_FORMAT_R8G8_UNORM,
    eR8G8Snorm                               = VK_FORMAT_R8G8_SNORM,
    eR8G8Uscaled                             = VK_FORMAT_R8G8_USCALED,
    eR8G8Sscaled                             = VK_FORMAT_R8G8_SSCALED,
    eR8G8Uint                                = VK_FORMAT_R8G8_UINT,
    eR8G8Sint                                = VK_FORMAT_R8G8_SINT,
    eR8G8Srgb                                = VK_FORMAT_R8G8_SRGB,
    eR8G8B8Unorm                             = VK_FORMAT_R8G8B8_UNORM,
    eR8G8B8Snorm                             = VK_FORMAT_R8G8B8_SNORM,
    eR8G8B8Uscaled                           = VK_FORMAT_R8G8B8_USCALED,
    eR8G8B8Sscaled                           = VK_FORMAT_R8G8B8_SSCALED,
    eR8G8B8Uint                              = VK_FORMAT_R8G8B8_UINT,
    eR8G8B8Sint                              = VK_FORMAT_R8G8B8_SINT,
    eR8G8B8Srgb                              = VK_FORMAT_R8G8B8_SRGB,
    eB8G8R8Unorm                             = VK_FORMAT_B8G8R8_UNORM,
    eB8G8R8Snorm                             = VK_FORMAT_B8G8R8_SNORM,
    eB8G8R8Uscaled                           = VK_FORMAT_B8G8R8_USCALED,
    eB8G8R8Sscaled                           = VK_FORMAT_B8G8R8_SSCALED,
    eB8G8R8Uint                              = VK_FORMAT_B8G8R8_UINT,
    eB8G8R8Sint                              = VK_FORMAT_B8G8R8_SINT,
    eB8G8R8Srgb                              = VK_FORMAT_B8G8R8_SRGB,
    eR8G8B8A8Unorm                           = VK_FORMAT_R8G8B8A8_UNORM,
    eR8G8B8A8Snorm                           = VK_FORMAT_R8G8B8A8_SNORM,
    eR8G8B8A8Uscaled                         = VK_FORMAT_R8G8B8A8_USCALED,
    eR8G8B8A8Sscaled                         = VK_FORMAT_R8G8B8A8_SSCALED,
    eR8G8B8A8Uint                            = VK_FORMAT_R8G8B8A8_UINT,
    eR8G8B8A8Sint                            = VK_FORMAT_R8G8B8A8_SINT,
    eR8G8B8A8Srgb                            = VK_FORMAT_R8G8B8A8_SRGB,
    eB8G8R8A8Unorm                           = VK_FORMAT_B8G8R8A8_UNORM,
    eB8G8R8A8Snorm                           = VK_FORMAT_B8G8R8A8_SNORM,
    eB8G8R8A8Uscaled                         = VK_FORMAT_B8G8R8A8_USCALED,
    eB8G8R8A8Sscaled                         = VK_FORMAT_B8G8R8A8_SSCALED,
    eB8G8R8A8Uint                            = VK_FORMAT_B8G8R8A8_UINT,
    eB8G8R8A8Sint                            = VK_FORMAT_B8G8R8A8_SINT,
    eB8G8R8A8Srgb                            = VK_FORMAT_B8G8R8A8_SRGB,
    eA8B8G8R8UnormPack32                     = VK_FORMAT_A8B8G8R8_UNORM_PACK32,
    eA8B8G8R8SnormPack32                     = VK_FORMAT_A8B8G8R8_SNORM_PACK32,
    eA8B8G8R8UscaledPack32                   = VK_FORMAT_A8B8G8R8_USCALED_PACK32,
    eA8B8G8R8SscaledPack32                   = VK_FORMAT_A8B8G8R8_SSCALED_PACK32,
    eA8B8G8R8UintPack32                      = VK_FORMAT_A8B8G8R8_UINT_PACK32,
    eA8B8G8R8SintPack32                      = VK_FORMAT_A8B8G8R8_SINT_PACK32,
    eA8B8G8R8SrgbPack32                      = VK_FORMAT_A8B8G8R8_SRGB_PACK32,
    eA2R10G10B10UnormPack32                  = VK_FORMAT_A2R10G10B10_UNORM_PACK32,
    eA2R10G10B10SnormPack32                  = VK_FORMAT_A2R10G10B10_SNORM_PACK32,
    eA2R10G10B10UscaledPack32                = VK_FORMAT_A2R10G10B10_USCALED_PACK32,
    eA2R10G10B10SscaledPack32                = VK_FORMAT_A2R10G10B10_SSCALED_PACK32,
    eA2R10G10B10UintPack32                   = VK_FORMAT_A2R10G10B10_UINT_PACK32,
    eA2R10G10B10SintPack32                   = VK_FORMAT_A2R10G10B10_SINT_PACK32,
    eA2B10G10R10UnormPack32                  = VK_FORMAT_A2B10G10R10_UNORM_PACK32,
    eA2B10G10R10SnormPack32                  = VK_FORMAT_A2B10G10R10_SNORM_PACK32,
    eA2B10G10R10UscaledPack32                = VK_FORMAT_A2B10G10R10_USCALED_PACK32,
    eA2B10G10R10SscaledPack32                = VK_FORMAT_A2B10G10R10_SSCALED_PACK32,
    eA2B10G10R10UintPack32                   = VK_FORMAT_A2B10G10R10_UINT_PACK32,
    eA2B10G10R10SintPack32                   = VK_FORMAT_A2B10G10R10_SINT_PACK32,
    eR16Unorm                                = VK_FORMAT_R16_UNORM,
    eR16Snorm                                = VK_FORMAT_R16_SNORM,
    eR16Uscaled                              = VK_FORMAT_R16_USCALED,
    eR16Sscaled                              = VK_FORMAT_R16_SSCALED,
    eR16Uint                                 = VK_FORMAT_R16_UINT,
    eR16Sint                                 = VK_FORMAT_R16_SINT,
    eR16Sfloat                               = VK_FORMAT_R16_SFLOAT,
    eR16G16Unorm                             = VK_FORMAT_R16G16_UNORM,
    eR16G16Snorm                             = VK_FORMAT_R16G16_SNORM,
    eR16G16Uscaled                           = VK_FORMAT_R16G16_USCALED,
    eR16G16Sscaled                           = VK_FORMAT_R16G16_SSCALED,
    eR16G16Uint                              = VK_FORMAT_R16G16_UINT,
    eR16G16Sint                              = VK_FORMAT_R16G16_SINT,
    eR16G16Sfloat                            = VK_FORMAT_R16G16_SFLOAT,
    eR16G16B16Unorm                          = VK_FORMAT_R16G16B16_UNORM,
    eR16G16B16Snorm                          = VK_FORMAT_R16G16B16_SNORM,
    eR16G16B16Uscaled                        = VK_FORMAT_R16G16B16_USCALED,
    eR16G16B16Sscaled                        = VK_FORMAT_R16G16B16_SSCALED,
    eR16G16B16Uint                           = VK_FORMAT_R16G16B16_UINT,
    eR16G16B16Sint                           = VK_FORMAT_R16G16B16_SINT,
    eR16G16B16Sfloat                         = VK_FORMAT_R16G16B16_SFLOAT,
    eR16G16B16A16Unorm                       = VK_FORMAT_R16G16B16A16_UNORM,
    eR16G16B16A16Snorm                       = VK_FORMAT_R16G16B16A16_SNORM,
    eR16G16B16A16Uscaled                     = VK_FORMAT_R16G16B16A16_USCALED,
    eR16G16B16A16Sscaled                     = VK_FORMAT_R16G16B16A16_SSCALED,
    eR16G16B16A16Uint                        = VK_FORMAT_R16G16B16A16_UINT,
    eR16G16B16A16Sint                        = VK_FORMAT_R16G16B16A16_SINT,
    eR16G16B16A16Sfloat                      = VK_FORMAT_R16G16B16A16_SFLOAT,
    eR32Uint                                 = VK_FORMAT_R32_UINT,
    eR32Sint                                 = VK_FORMAT_R32_SINT,
    eR32Sfloat                               = VK_FORMAT_R32_SFLOAT,
    eR32G32Uint                              = VK_FORMAT_R32G32_UINT,
    eR32G32Sint                              = VK_FORMAT_R32G32_SINT,
    eR32G32Sfloat                            = VK_FORMAT_R32G32_SFLOAT,
    eR32G32B32Uint                           = VK_FORMAT_R32G32B32_UINT,
    eR32G32B32Sint                           = VK_FORMAT_R32G32B32_SINT,
    eR32G32B32Sfloat                         = VK_FORMAT_R32G32B32_SFLOAT,
    eR32G32B32A32Uint                        = VK_FORMAT_R32G32B32A32_UINT,
    eR32G32B32A32Sint                        = VK_FORMAT_R32G32B32A32_SINT,
    eR32G32B32A32Sfloat                      = VK_FORMAT_R32G32B32A32_SFLOAT,
    eR64Uint                                 = VK_FORMAT_R64_UINT,
    eR64Sint                                 = VK_FORMAT_R64_SINT,
    eR64Sfloat                               = VK_FORMAT_R64_SFLOAT,
    eR64G64Uint                              = VK_FORMAT_R64G64_UINT,
    eR64G64Sint                              = VK_FORMAT_R64G64_SINT,
    eR64G64Sfloat                            = VK_FORMAT_R64G64_SFLOAT,
    eR64G64B64Uint                           = VK_FORMAT_R64G64B64_UINT,
    eR64G64B64Sint                           = VK_FORMAT_R64G64B64_SINT,
    eR64G64B64Sfloat                         = VK_FORMAT_R64G64B64_SFLOAT,
    eR64G64B64A64Uint                        = VK_FORMAT_R64G64B64A64_UINT,
    eR64G64B64A64Sint                        = VK_FORMAT_R64G64B64A64_SINT,
    eR64G64B64A64Sfloat                      = VK_FORMAT_R64G64B64A64_SFLOAT,
    eB10G11R11UfloatPack32                   = VK_FORMAT_B10G11R11_UFLOAT_PACK32,
    eE5B9G9R9UfloatPack32                    = VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,
    eD16Unorm                                = VK_FORMAT_D16_UNORM,
    eX8D24UnormPack32                        = VK_FORMAT_X8_D24_UNORM_PACK32,
    eD32Sfloat                               = VK_FORMAT_D32_SFLOAT,
    eS8Uint                                  = VK_FORMAT_S8_UINT,
    eD16UnormS8Uint                          = VK_FORMAT_D16_UNORM_S8_UINT,
    eD24UnormS8Uint                          = VK_FORMAT_D24_UNORM_S8_UINT,
    eD32SfloatS8Uint                         = VK_FORMAT_D32_SFLOAT_S8_UINT,
    eBc1RgbUnormBlock                        = VK_FORMAT_BC1_RGB_UNORM_BLOCK,
    eBc1RgbSrgbBlock                         = VK_FORMAT_BC1_RGB_SRGB_BLOCK,
    eBc1RgbaUnormBlock                       = VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
    eBc1RgbaSrgbBlock                        = VK_FORMAT_BC1_RGBA_SRGB_BLOCK,
    eBc2UnormBlock                           = VK_FORMAT_BC2_UNORM_BLOCK,
    eBc2SrgbBlock                            = VK_FORMAT_BC2_SRGB_BLOCK,
    eBc3UnormBlock                           = VK_FORMAT_BC3_UNORM_BLOCK,
    eBc3SrgbBlock                            = VK_FORMAT_BC3_SRGB_BLOCK,
    eBc4UnormBlock                           = VK_FORMAT_BC4_UNORM_BLOCK,
    eBc4SnormBlock                           = VK_FORMAT_BC4_SNORM_BLOCK,
    eBc5UnormBlock                           = VK_FORMAT_BC5_UNORM_BLOCK,
    eBc5SnormBlock                           = VK_FORMAT_BC5_SNORM_BLOCK,
    eBc6HUfloatBlock                         = VK_FORMAT_BC6H_UFLOAT_BLOCK,
    eBc6HSfloatBlock                         = VK_FORMAT_BC6H_SFLOAT_BLOCK,
    eBc7UnormBlock                           = VK_FORMAT_BC7_UNORM_BLOCK,
    eBc7SrgbBlock                            = VK_FORMAT_BC7_SRGB_BLOCK,
    eEtc2R8G8B8UnormBlock                    = VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK,
    eEtc2R8G8B8SrgbBlock                     = VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK,
    eEtc2R8G8B8A1UnormBlock                  = VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK,
    eEtc2R8G8B8A1SrgbBlock                   = VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK,
    eEtc2R8G8B8A8UnormBlock                  = VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK,
    eEtc2R8G8B8A8SrgbBlock                   = VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK,
    eEacR11UnormBlock                        = VK_FORMAT_EAC_R11_UNORM_BLOCK,
    eEacR11SnormBlock                        = VK_FORMAT_EAC_R11_SNORM_BLOCK,
    eEacR11G11UnormBlock                     = VK_FORMAT_EAC_R11G11_UNORM_BLOCK,
    eEacR11G11SnormBlock                     = VK_FORMAT_EAC_R11G11_SNORM_BLOCK,
    eAstc4x4UnormBlock                       = VK_FORMAT_ASTC_4x4_UNORM_BLOCK,
    eAstc4x4SrgbBlock                        = VK_FORMAT_ASTC_4x4_SRGB_BLOCK,
    eAstc5x4UnormBlock                       = VK_FORMAT_ASTC_5x4_UNORM_BLOCK,
    eAstc5x4SrgbBlock                        = VK_FORMAT_ASTC_5x4_SRGB_BLOCK,
    eAstc5x5UnormBlock                       = VK_FORMAT_ASTC_5x5_UNORM_BLOCK,
    eAstc5x5SrgbBlock                        = VK_FORMAT_ASTC_5x5_SRGB_BLOCK,
    eAstc6x5UnormBlock                       = VK_FORMAT_ASTC_6x5_UNORM_BLOCK,
    eAstc6x5SrgbBlock                        = VK_FORMAT_ASTC_6x5_SRGB_BLOCK,
    eAstc6x6UnormBlock                       = VK_FORMAT_ASTC_6x6_UNORM_BLOCK,
    eAstc6x6SrgbBlock                        = VK_FORMAT_ASTC_6x6_SRGB_BLOCK,
    eAstc8x5UnormBlock                       = VK_FORMAT_ASTC_8x5_UNORM_BLOCK,
    eAstc8x5SrgbBlock                        = VK_FORMAT_ASTC_8x5_SRGB_BLOCK,
    eAstc8x6UnormBlock                       = VK_FORMAT_ASTC_8x6_UNORM_BLOCK,
    eAstc8x6SrgbBlock                        = VK_FORMAT_ASTC_8x6_SRGB_BLOCK,
    eAstc8x8UnormBlock                       = VK_FORMAT_ASTC_8x8_UNORM_BLOCK,
    eAstc8x8SrgbBlock                        = VK_FORMAT_ASTC_8x8_SRGB_BLOCK,
    eAstc10x5UnormBlock                      = VK_FORMAT_ASTC_10x5_UNORM_BLOCK,
    eAstc10x5SrgbBlock                       = VK_FORMAT_ASTC_10x5_SRGB_BLOCK,
    eAstc10x6UnormBlock                      = VK_FORMAT_ASTC_10x6_UNORM_BLOCK,
    eAstc10x6SrgbBlock                       = VK_FORMAT_ASTC_10x6_SRGB_BLOCK,
    eAstc10x8UnormBlock                      = VK_FORMAT_ASTC_10x8_UNORM_BLOCK,
    eAstc10x8SrgbBlock                       = VK_FORMAT_ASTC_10x8_SRGB_BLOCK,
    eAstc10x10UnormBlock                     = VK_FORMAT_ASTC_10x10_UNORM_BLOCK,
    eAstc10x10SrgbBlock                      = VK_FORMAT_ASTC_10x10_SRGB_BLOCK,
    eAstc12x10UnormBlock                     = VK_FORMAT_ASTC_12x10_UNORM_BLOCK,
    eAstc12x10SrgbBlock                      = VK_FORMAT_ASTC_12x10_SRGB_BLOCK,
    eAstc12x12UnormBlock                     = VK_FORMAT_ASTC_12x12_UNORM_BLOCK,
    eAstc12x12SrgbBlock                      = VK_FORMAT_ASTC_12x12_SRGB_BLOCK,
    eG8B8G8R8422Unorm                        = VK_FORMAT_G8B8G8R8_422_UNORM,
    eB8G8R8G8422Unorm                        = VK_FORMAT_B8G8R8G8_422_UNORM,
    eG8B8R83Plane420Unorm                    = VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM,
    eG8B8R82Plane420Unorm                    = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
    eG8B8R83Plane422Unorm                    = VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM,
    eG8B8R82Plane422Unorm                    = VK_FORMAT_G8_B8R8_2PLANE_422_UNORM,
    eG8B8R83Plane444Unorm                    = VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM,
    eR10X6UnormPack16                        = VK_FORMAT_R10X6_UNORM_PACK16,
    eR10X6G10X6Unorm2Pack16                  = VK_FORMAT_R10X6G10X6_UNORM_2PACK16,
    eR10X6G10X6B10X6A10X6Unorm4Pack16        = VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16,
    eG10X6B10X6G10X6R10X6422Unorm4Pack16     = VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16,
    eB10X6G10X6R10X6G10X6422Unorm4Pack16     = VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16,
    eG10X6B10X6R10X63Plane420Unorm3Pack16    = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16,
    eG10X6B10X6R10X62Plane420Unorm3Pack16    = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
    eG10X6B10X6R10X63Plane422Unorm3Pack16    = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16,
    eG10X6B10X6R10X62Plane422Unorm3Pack16    = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16,
    eG10X6B10X6R10X63Plane444Unorm3Pack16    = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16,
    eR12X4UnormPack16                        = VK_FORMAT_R12X4_UNORM_PACK16,
    eR12X4G12X4Unorm2Pack16                  = VK_FORMAT_R12X4G12X4_UNORM_2PACK16,
    eR12X4G12X4B12X4A12X4Unorm4Pack16        = VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16,
    eG12X4B12X4G12X4R12X4422Unorm4Pack16     = VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16,
    eB12X4G12X4R12X4G12X4422Unorm4Pack16     = VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16,
    eG12X4B12X4R12X43Plane420Unorm3Pack16    = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16,
    eG12X4B12X4R12X42Plane420Unorm3Pack16    = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16,
    eG12X4B12X4R12X43Plane422Unorm3Pack16    = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16,
    eG12X4B12X4R12X42Plane422Unorm3Pack16    = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16,
    eG12X4B12X4R12X43Plane444Unorm3Pack16    = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16,
    eG16B16G16R16422Unorm                    = VK_FORMAT_G16B16G16R16_422_UNORM,
    eB16G16R16G16422Unorm                    = VK_FORMAT_B16G16R16G16_422_UNORM,
    eG16B16R163Plane420Unorm                 = VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM,
    eG16B16R162Plane420Unorm                 = VK_FORMAT_G16_B16R16_2PLANE_420_UNORM,
    eG16B16R163Plane422Unorm                 = VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM,
    eG16B16R162Plane422Unorm                 = VK_FORMAT_G16_B16R16_2PLANE_422_UNORM,
    eG16B16R163Plane444Unorm                 = VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM,
    ePvrtc12BppUnormBlockIMG                 = VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG,
    ePvrtc14BppUnormBlockIMG                 = VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG,
    ePvrtc22BppUnormBlockIMG                 = VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG,
    ePvrtc24BppUnormBlockIMG                 = VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG,
    ePvrtc12BppSrgbBlockIMG                  = VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG,
    ePvrtc14BppSrgbBlockIMG                  = VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG,
    ePvrtc22BppSrgbBlockIMG                  = VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG,
    ePvrtc24BppSrgbBlockIMG                  = VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG,
    eAstc4x4SfloatBlockEXT                   = VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK_EXT,
    eAstc5x4SfloatBlockEXT                   = VK_FORMAT_ASTC_5x4_SFLOAT_BLOCK_EXT,
    eAstc5x5SfloatBlockEXT                   = VK_FORMAT_ASTC_5x5_SFLOAT_BLOCK_EXT,
    eAstc6x5SfloatBlockEXT                   = VK_FORMAT_ASTC_6x5_SFLOAT_BLOCK_EXT,
    eAstc6x6SfloatBlockEXT                   = VK_FORMAT_ASTC_6x6_SFLOAT_BLOCK_EXT,
    eAstc8x5SfloatBlockEXT                   = VK_FORMAT_ASTC_8x5_SFLOAT_BLOCK_EXT,
    eAstc8x6SfloatBlockEXT                   = VK_FORMAT_ASTC_8x6_SFLOAT_BLOCK_EXT,
    eAstc8x8SfloatBlockEXT                   = VK_FORMAT_ASTC_8x8_SFLOAT_BLOCK_EXT,
    eAstc10x5SfloatBlockEXT                  = VK_FORMAT_ASTC_10x5_SFLOAT_BLOCK_EXT,
    eAstc10x6SfloatBlockEXT                  = VK_FORMAT_ASTC_10x6_SFLOAT_BLOCK_EXT,
    eAstc10x8SfloatBlockEXT                  = VK_FORMAT_ASTC_10x8_SFLOAT_BLOCK_EXT,
    eAstc10x10SfloatBlockEXT                 = VK_FORMAT_ASTC_10x10_SFLOAT_BLOCK_EXT,
    eAstc12x10SfloatBlockEXT                 = VK_FORMAT_ASTC_12x10_SFLOAT_BLOCK_EXT,
    eAstc12x12SfloatBlockEXT                 = VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK_EXT,
    eG8B8R82Plane444UnormEXT                 = VK_FORMAT_G8_B8R8_2PLANE_444_UNORM_EXT,
    eG10X6B10X6R10X62Plane444Unorm3Pack16EXT = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16_EXT,
    eG12X4B12X4R12X42Plane444Unorm3Pack16EXT = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16_EXT,
    eG16B16R162Plane444UnormEXT              = VK_FORMAT_G16_B16R16_2PLANE_444_UNORM_EXT,
    eA4R4G4B4UnormPack16EXT                  = VK_FORMAT_A4R4G4B4_UNORM_PACK16_EXT,
    eA4B4G4R4UnormPack16EXT                  = VK_FORMAT_A4B4G4R4_UNORM_PACK16_EXT,
    eB10X6G10X6R10X6G10X6422Unorm4Pack16KHR  = VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16_KHR,
    eB12X4G12X4R12X4G12X4422Unorm4Pack16KHR  = VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16_KHR,
    eB16G16R16G16422UnormKHR                 = VK_FORMAT_B16G16R16G16_422_UNORM_KHR,
    eB8G8R8G8422UnormKHR                     = VK_FORMAT_B8G8R8G8_422_UNORM_KHR,
    eG10X6B10X6G10X6R10X6422Unorm4Pack16KHR  = VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16_KHR,
    eG10X6B10X6R10X62Plane420Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16_KHR,
    eG10X6B10X6R10X62Plane422Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16_KHR,
    eG10X6B10X6R10X63Plane420Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16_KHR,
    eG10X6B10X6R10X63Plane422Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16_KHR,
    eG10X6B10X6R10X63Plane444Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16_KHR,
    eG12X4B12X4G12X4R12X4422Unorm4Pack16KHR  = VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16_KHR,
    eG12X4B12X4R12X42Plane420Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16_KHR,
    eG12X4B12X4R12X42Plane422Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16_KHR,
    eG12X4B12X4R12X43Plane420Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16_KHR,
    eG12X4B12X4R12X43Plane422Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16_KHR,
    eG12X4B12X4R12X43Plane444Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16_KHR,
    eG16B16G16R16422UnormKHR                 = VK_FORMAT_G16B16G16R16_422_UNORM_KHR,
    eG16B16R162Plane420UnormKHR              = VK_FORMAT_G16_B16R16_2PLANE_420_UNORM_KHR,
    eG16B16R162Plane422UnormKHR              = VK_FORMAT_G16_B16R16_2PLANE_422_UNORM_KHR,
    eG16B16R163Plane420UnormKHR              = VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM_KHR,
    eG16B16R163Plane422UnormKHR              = VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM_KHR,
    eG16B16R163Plane444UnormKHR              = VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM_KHR,
    eG8B8G8R8422UnormKHR                     = VK_FORMAT_G8B8G8R8_422_UNORM_KHR,
    eG8B8R82Plane420UnormKHR                 = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM_KHR,
    eG8B8R82Plane422UnormKHR                 = VK_FORMAT_G8_B8R8_2PLANE_422_UNORM_KHR,
    eG8B8R83Plane420UnormKHR                 = VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM_KHR,
    eG8B8R83Plane422UnormKHR                 = VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM_KHR,
    eG8B8R83Plane444UnormKHR                 = VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM_KHR,
    eR10X6G10X6B10X6A10X6Unorm4Pack16KHR     = VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16_KHR,
    eR10X6G10X6Unorm2Pack16KHR               = VK_FORMAT_R10X6G10X6_UNORM_2PACK16_KHR,
    eR10X6UnormPack16KHR                     = VK_FORMAT_R10X6_UNORM_PACK16_KHR,
    eR12X4G12X4B12X4A12X4Unorm4Pack16KHR     = VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16_KHR,
    eR12X4G12X4Unorm2Pack16KHR               = VK_FORMAT_R12X4G12X4_UNORM_2PACK16_KHR,
    eR12X4UnormPack16KHR                     = VK_FORMAT_R12X4_UNORM_PACK16_KHR
  };

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
      case Format::ePvrtc12BppUnormBlockIMG: return "Pvrtc12BppUnormBlockIMG";
      case Format::ePvrtc14BppUnormBlockIMG: return "Pvrtc14BppUnormBlockIMG";
      case Format::ePvrtc22BppUnormBlockIMG: return "Pvrtc22BppUnormBlockIMG";
      case Format::ePvrtc24BppUnormBlockIMG: return "Pvrtc24BppUnormBlockIMG";
      case Format::ePvrtc12BppSrgbBlockIMG: return "Pvrtc12BppSrgbBlockIMG";
      case Format::ePvrtc14BppSrgbBlockIMG: return "Pvrtc14BppSrgbBlockIMG";
      case Format::ePvrtc22BppSrgbBlockIMG: return "Pvrtc22BppSrgbBlockIMG";
      case Format::ePvrtc24BppSrgbBlockIMG: return "Pvrtc24BppSrgbBlockIMG";
      case Format::eAstc4x4SfloatBlockEXT: return "Astc4x4SfloatBlockEXT";
      case Format::eAstc5x4SfloatBlockEXT: return "Astc5x4SfloatBlockEXT";
      case Format::eAstc5x5SfloatBlockEXT: return "Astc5x5SfloatBlockEXT";
      case Format::eAstc6x5SfloatBlockEXT: return "Astc6x5SfloatBlockEXT";
      case Format::eAstc6x6SfloatBlockEXT: return "Astc6x6SfloatBlockEXT";
      case Format::eAstc8x5SfloatBlockEXT: return "Astc8x5SfloatBlockEXT";
      case Format::eAstc8x6SfloatBlockEXT: return "Astc8x6SfloatBlockEXT";
      case Format::eAstc8x8SfloatBlockEXT: return "Astc8x8SfloatBlockEXT";
      case Format::eAstc10x5SfloatBlockEXT: return "Astc10x5SfloatBlockEXT";
      case Format::eAstc10x6SfloatBlockEXT: return "Astc10x6SfloatBlockEXT";
      case Format::eAstc10x8SfloatBlockEXT: return "Astc10x8SfloatBlockEXT";
      case Format::eAstc10x10SfloatBlockEXT: return "Astc10x10SfloatBlockEXT";
      case Format::eAstc12x10SfloatBlockEXT: return "Astc12x10SfloatBlockEXT";
      case Format::eAstc12x12SfloatBlockEXT: return "Astc12x12SfloatBlockEXT";
      case Format::eG8B8R82Plane444UnormEXT: return "G8B8R82Plane444UnormEXT";
      case Format::eG10X6B10X6R10X62Plane444Unorm3Pack16EXT: return "G10X6B10X6R10X62Plane444Unorm3Pack16EXT";
      case Format::eG12X4B12X4R12X42Plane444Unorm3Pack16EXT: return "G12X4B12X4R12X42Plane444Unorm3Pack16EXT";
      case Format::eG16B16R162Plane444UnormEXT: return "G16B16R162Plane444UnormEXT";
      case Format::eA4R4G4B4UnormPack16EXT: return "A4R4G4B4UnormPack16EXT";
      case Format::eA4B4G4R4UnormPack16EXT: return "A4B4G4R4UnormPack16EXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class FormatFeatureFlagBits : VkFormatFeatureFlags
  {
    eSampledImage                            = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT,
    eStorageImage                            = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT,
    eStorageImageAtomic                      = VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT,
    eUniformTexelBuffer                      = VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT,
    eStorageTexelBuffer                      = VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT,
    eStorageTexelBufferAtomic                = VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT,
    eVertexBuffer                            = VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT,
    eColorAttachment                         = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT,
    eColorAttachmentBlend                    = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT,
    eDepthStencilAttachment                  = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT,
    eBlitSrc                                 = VK_FORMAT_FEATURE_BLIT_SRC_BIT,
    eBlitDst                                 = VK_FORMAT_FEATURE_BLIT_DST_BIT,
    eSampledImageFilterLinear                = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT,
    eTransferSrc                             = VK_FORMAT_FEATURE_TRANSFER_SRC_BIT,
    eTransferDst                             = VK_FORMAT_FEATURE_TRANSFER_DST_BIT,
    eMidpointChromaSamples                   = VK_FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT,
    eSampledImageYcbcrConversionLinearFilter = VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT,
    eSampledImageYcbcrConversionSeparateReconstructionFilter =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT,
    eSampledImageYcbcrConversionChromaReconstructionExplicit =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_BIT,
    eSampledImageYcbcrConversionChromaReconstructionExplicitForceable =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE_BIT,
    eDisjoint                   = VK_FORMAT_FEATURE_DISJOINT_BIT,
    eCositedChromaSamples       = VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT,
    eSampledImageFilterMinmax   = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_MINMAX_BIT,
    eSampledImageFilterCubicIMG = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_CUBIC_BIT_IMG,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoDecodeOutputKHR = VK_FORMAT_FEATURE_VIDEO_DECODE_OUTPUT_BIT_KHR,
    eVideoDecodeDpbKHR    = VK_FORMAT_FEATURE_VIDEO_DECODE_DPB_BIT_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eAccelerationStructureVertexBufferKHR = VK_FORMAT_FEATURE_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR,
    eFragmentDensityMapEXT                = VK_FORMAT_FEATURE_FRAGMENT_DENSITY_MAP_BIT_EXT,
    eFragmentShadingRateAttachmentKHR     = VK_FORMAT_FEATURE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoEncodeInputKHR = VK_FORMAT_FEATURE_VIDEO_ENCODE_INPUT_BIT_KHR,
    eVideoEncodeDpbKHR   = VK_FORMAT_FEATURE_VIDEO_ENCODE_DPB_BIT_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eCositedChromaSamplesKHR     = VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT_KHR,
    eDisjointKHR                 = VK_FORMAT_FEATURE_DISJOINT_BIT_KHR,
    eMidpointChromaSamplesKHR    = VK_FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT_KHR,
    eSampledImageFilterCubicEXT  = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_CUBIC_BIT_EXT,
    eSampledImageFilterMinmaxEXT = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_MINMAX_BIT_EXT,
    eSampledImageYcbcrConversionChromaReconstructionExplicitKHR =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_BIT_KHR,
    eSampledImageYcbcrConversionChromaReconstructionExplicitForceableKHR =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE_BIT_KHR,
    eSampledImageYcbcrConversionLinearFilterKHR =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT_KHR,
    eSampledImageYcbcrConversionSeparateReconstructionFilterKHR =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT_KHR,
    eTransferDstKHR = VK_FORMAT_FEATURE_TRANSFER_DST_BIT_KHR,
    eTransferSrcKHR = VK_FORMAT_FEATURE_TRANSFER_SRC_BIT_KHR
  };

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
      case FormatFeatureFlagBits::eSampledImageYcbcrConversionLinearFilter:
        return "SampledImageYcbcrConversionLinearFilter";
      case FormatFeatureFlagBits::eSampledImageYcbcrConversionSeparateReconstructionFilter:
        return "SampledImageYcbcrConversionSeparateReconstructionFilter";
      case FormatFeatureFlagBits::eSampledImageYcbcrConversionChromaReconstructionExplicit:
        return "SampledImageYcbcrConversionChromaReconstructionExplicit";
      case FormatFeatureFlagBits::eSampledImageYcbcrConversionChromaReconstructionExplicitForceable:
        return "SampledImageYcbcrConversionChromaReconstructionExplicitForceable";
      case FormatFeatureFlagBits::eDisjoint: return "Disjoint";
      case FormatFeatureFlagBits::eCositedChromaSamples: return "CositedChromaSamples";
      case FormatFeatureFlagBits::eSampledImageFilterMinmax: return "SampledImageFilterMinmax";
      case FormatFeatureFlagBits::eSampledImageFilterCubicIMG: return "SampledImageFilterCubicIMG";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case FormatFeatureFlagBits::eVideoDecodeOutputKHR: return "VideoDecodeOutputKHR";
      case FormatFeatureFlagBits::eVideoDecodeDpbKHR: return "VideoDecodeDpbKHR";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      case FormatFeatureFlagBits::eAccelerationStructureVertexBufferKHR: return "AccelerationStructureVertexBufferKHR";
      case FormatFeatureFlagBits::eFragmentDensityMapEXT: return "FragmentDensityMapEXT";
      case FormatFeatureFlagBits::eFragmentShadingRateAttachmentKHR: return "FragmentShadingRateAttachmentKHR";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case FormatFeatureFlagBits::eVideoEncodeInputKHR: return "VideoEncodeInputKHR";
      case FormatFeatureFlagBits::eVideoEncodeDpbKHR: return "VideoEncodeDpbKHR";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class ImageCreateFlagBits : VkImageCreateFlags
  {
    eSparseBinding                     = VK_IMAGE_CREATE_SPARSE_BINDING_BIT,
    eSparseResidency                   = VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
    eSparseAliased                     = VK_IMAGE_CREATE_SPARSE_ALIASED_BIT,
    eMutableFormat                     = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
    eCubeCompatible                    = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
    eAlias                             = VK_IMAGE_CREATE_ALIAS_BIT,
    eSplitInstanceBindRegions          = VK_IMAGE_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT,
    e2DArrayCompatible                 = VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT,
    eBlockTexelViewCompatible          = VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT,
    eExtendedUsage                     = VK_IMAGE_CREATE_EXTENDED_USAGE_BIT,
    eProtected                         = VK_IMAGE_CREATE_PROTECTED_BIT,
    eDisjoint                          = VK_IMAGE_CREATE_DISJOINT_BIT,
    eCornerSampledNV                   = VK_IMAGE_CREATE_CORNER_SAMPLED_BIT_NV,
    eSampleLocationsCompatibleDepthEXT = VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT,
    eSubsampledEXT                     = VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT,
    e2DArrayCompatibleKHR              = VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT_KHR,
    eAliasKHR                          = VK_IMAGE_CREATE_ALIAS_BIT_KHR,
    eBlockTexelViewCompatibleKHR       = VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT_KHR,
    eDisjointKHR                       = VK_IMAGE_CREATE_DISJOINT_BIT_KHR,
    eExtendedUsageKHR                  = VK_IMAGE_CREATE_EXTENDED_USAGE_BIT_KHR,
    eSplitInstanceBindRegionsKHR       = VK_IMAGE_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT_KHR
  };

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
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class ImageTiling
  {
    eOptimal              = VK_IMAGE_TILING_OPTIMAL,
    eLinear               = VK_IMAGE_TILING_LINEAR,
    eDrmFormatModifierEXT = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT
  };

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

  enum class ImageType
  {
    e1D = VK_IMAGE_TYPE_1D,
    e2D = VK_IMAGE_TYPE_2D,
    e3D = VK_IMAGE_TYPE_3D
  };

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

  enum class ImageUsageFlagBits : VkImageUsageFlags
  {
    eTransferSrc            = VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
    eTransferDst            = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
    eSampled                = VK_IMAGE_USAGE_SAMPLED_BIT,
    eStorage                = VK_IMAGE_USAGE_STORAGE_BIT,
    eColorAttachment        = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    eDepthStencilAttachment = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
    eTransientAttachment    = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
    eInputAttachment        = VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoDecodeDstKHR = VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR,
    eVideoDecodeSrcKHR = VK_IMAGE_USAGE_VIDEO_DECODE_SRC_BIT_KHR,
    eVideoDecodeDpbKHR = VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eFragmentDensityMapEXT            = VK_IMAGE_USAGE_FRAGMENT_DENSITY_MAP_BIT_EXT,
    eFragmentShadingRateAttachmentKHR = VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoEncodeDstKHR = VK_IMAGE_USAGE_VIDEO_ENCODE_DST_BIT_KHR,
    eVideoEncodeSrcKHR = VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR,
    eVideoEncodeDpbKHR = VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eShadingRateImageNV = VK_IMAGE_USAGE_SHADING_RATE_IMAGE_BIT_NV
  };

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
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case ImageUsageFlagBits::eVideoDecodeDstKHR: return "VideoDecodeDstKHR";
      case ImageUsageFlagBits::eVideoDecodeSrcKHR: return "VideoDecodeSrcKHR";
      case ImageUsageFlagBits::eVideoDecodeDpbKHR: return "VideoDecodeDpbKHR";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      case ImageUsageFlagBits::eFragmentDensityMapEXT: return "FragmentDensityMapEXT";
      case ImageUsageFlagBits::eFragmentShadingRateAttachmentKHR: return "FragmentShadingRateAttachmentKHR";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case ImageUsageFlagBits::eVideoEncodeDstKHR: return "VideoEncodeDstKHR";
      case ImageUsageFlagBits::eVideoEncodeSrcKHR: return "VideoEncodeSrcKHR";
      case ImageUsageFlagBits::eVideoEncodeDpbKHR: return "VideoEncodeDpbKHR";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class InternalAllocationType
  {
    eExecutable = VK_INTERNAL_ALLOCATION_TYPE_EXECUTABLE
  };

  VULKAN_HPP_INLINE std::string to_string( InternalAllocationType value )
  {
    switch ( value )
    {
      case InternalAllocationType::eExecutable: return "Executable";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class MemoryHeapFlagBits : VkMemoryHeapFlags
  {
    eDeviceLocal      = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT,
    eMultiInstance    = VK_MEMORY_HEAP_MULTI_INSTANCE_BIT,
    eMultiInstanceKHR = VK_MEMORY_HEAP_MULTI_INSTANCE_BIT_KHR
  };

  VULKAN_HPP_INLINE std::string to_string( MemoryHeapFlagBits value )
  {
    switch ( value )
    {
      case MemoryHeapFlagBits::eDeviceLocal: return "DeviceLocal";
      case MemoryHeapFlagBits::eMultiInstance: return "MultiInstance";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class MemoryPropertyFlagBits : VkMemoryPropertyFlags
  {
    eDeviceLocal       = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    eHostVisible       = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
    eHostCoherent      = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    eHostCached        = VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
    eLazilyAllocated   = VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT,
    eProtected         = VK_MEMORY_PROPERTY_PROTECTED_BIT,
    eDeviceCoherentAMD = VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD,
    eDeviceUncachedAMD = VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD
  };

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
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class PhysicalDeviceType
  {
    eOther         = VK_PHYSICAL_DEVICE_TYPE_OTHER,
    eIntegratedGpu = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
    eDiscreteGpu   = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
    eVirtualGpu    = VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU,
    eCpu           = VK_PHYSICAL_DEVICE_TYPE_CPU
  };

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

  enum class QueueFlagBits : VkQueueFlags
  {
    eGraphics      = VK_QUEUE_GRAPHICS_BIT,
    eCompute       = VK_QUEUE_COMPUTE_BIT,
    eTransfer      = VK_QUEUE_TRANSFER_BIT,
    eSparseBinding = VK_QUEUE_SPARSE_BINDING_BIT,
    eProtected     = VK_QUEUE_PROTECTED_BIT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoDecodeKHR = VK_QUEUE_VIDEO_DECODE_BIT_KHR,
    eVideoEncodeKHR = VK_QUEUE_VIDEO_ENCODE_BIT_KHR
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
  };

  VULKAN_HPP_INLINE std::string to_string( QueueFlagBits value )
  {
    switch ( value )
    {
      case QueueFlagBits::eGraphics: return "Graphics";
      case QueueFlagBits::eCompute: return "Compute";
      case QueueFlagBits::eTransfer: return "Transfer";
      case QueueFlagBits::eSparseBinding: return "SparseBinding";
      case QueueFlagBits::eProtected: return "Protected";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case QueueFlagBits::eVideoDecodeKHR: return "VideoDecodeKHR";
      case QueueFlagBits::eVideoEncodeKHR: return "VideoEncodeKHR";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class SampleCountFlagBits : VkSampleCountFlags
  {
    e1  = VK_SAMPLE_COUNT_1_BIT,
    e2  = VK_SAMPLE_COUNT_2_BIT,
    e4  = VK_SAMPLE_COUNT_4_BIT,
    e8  = VK_SAMPLE_COUNT_8_BIT,
    e16 = VK_SAMPLE_COUNT_16_BIT,
    e32 = VK_SAMPLE_COUNT_32_BIT,
    e64 = VK_SAMPLE_COUNT_64_BIT
  };

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

  enum class SystemAllocationScope
  {
    eCommand  = VK_SYSTEM_ALLOCATION_SCOPE_COMMAND,
    eObject   = VK_SYSTEM_ALLOCATION_SCOPE_OBJECT,
    eCache    = VK_SYSTEM_ALLOCATION_SCOPE_CACHE,
    eDevice   = VK_SYSTEM_ALLOCATION_SCOPE_DEVICE,
    eInstance = VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE
  };

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

  enum class DeviceQueueCreateFlagBits : VkDeviceQueueCreateFlags
  {
    eProtected = VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT
  };

  VULKAN_HPP_INLINE std::string to_string( DeviceQueueCreateFlagBits value )
  {
    switch ( value )
    {
      case DeviceQueueCreateFlagBits::eProtected: return "Protected";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class PipelineStageFlagBits : VkPipelineStageFlags
  {
    eTopOfPipe                        = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
    eDrawIndirect                     = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
    eVertexInput                      = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
    eVertexShader                     = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
    eTessellationControlShader        = VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT,
    eTessellationEvaluationShader     = VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT,
    eGeometryShader                   = VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT,
    eFragmentShader                   = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
    eEarlyFragmentTests               = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
    eLateFragmentTests                = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
    eColorAttachmentOutput            = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
    eComputeShader                    = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    eTransfer                         = VK_PIPELINE_STAGE_TRANSFER_BIT,
    eBottomOfPipe                     = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
    eHost                             = VK_PIPELINE_STAGE_HOST_BIT,
    eAllGraphics                      = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
    eAllCommands                      = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
    eTransformFeedbackEXT             = VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
    eConditionalRenderingEXT          = VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT,
    eAccelerationStructureBuildKHR    = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    eRayTracingShaderKHR              = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
    eTaskShaderNV                     = VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV,
    eMeshShaderNV                     = VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV,
    eFragmentDensityProcessEXT        = VK_PIPELINE_STAGE_FRAGMENT_DENSITY_PROCESS_BIT_EXT,
    eFragmentShadingRateAttachmentKHR = VK_PIPELINE_STAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
    eCommandPreprocessNV              = VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_NV,
    eNoneKHR                          = VK_PIPELINE_STAGE_NONE_KHR,
    eAccelerationStructureBuildNV     = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
    eRayTracingShaderNV               = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_NV,
    eShadingRateImageNV               = VK_PIPELINE_STAGE_SHADING_RATE_IMAGE_BIT_NV
  };

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
      case PipelineStageFlagBits::eTransformFeedbackEXT: return "TransformFeedbackEXT";
      case PipelineStageFlagBits::eConditionalRenderingEXT: return "ConditionalRenderingEXT";
      case PipelineStageFlagBits::eAccelerationStructureBuildKHR: return "AccelerationStructureBuildKHR";
      case PipelineStageFlagBits::eRayTracingShaderKHR: return "RayTracingShaderKHR";
      case PipelineStageFlagBits::eTaskShaderNV: return "TaskShaderNV";
      case PipelineStageFlagBits::eMeshShaderNV: return "MeshShaderNV";
      case PipelineStageFlagBits::eFragmentDensityProcessEXT: return "FragmentDensityProcessEXT";
      case PipelineStageFlagBits::eFragmentShadingRateAttachmentKHR: return "FragmentShadingRateAttachmentKHR";
      case PipelineStageFlagBits::eCommandPreprocessNV: return "CommandPreprocessNV";
      case PipelineStageFlagBits::eNoneKHR: return "NoneKHR";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class ImageAspectFlagBits : VkImageAspectFlags
  {
    eColor           = VK_IMAGE_ASPECT_COLOR_BIT,
    eDepth           = VK_IMAGE_ASPECT_DEPTH_BIT,
    eStencil         = VK_IMAGE_ASPECT_STENCIL_BIT,
    eMetadata        = VK_IMAGE_ASPECT_METADATA_BIT,
    ePlane0          = VK_IMAGE_ASPECT_PLANE_0_BIT,
    ePlane1          = VK_IMAGE_ASPECT_PLANE_1_BIT,
    ePlane2          = VK_IMAGE_ASPECT_PLANE_2_BIT,
    eMemoryPlane0EXT = VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT,
    eMemoryPlane1EXT = VK_IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT,
    eMemoryPlane2EXT = VK_IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT,
    eMemoryPlane3EXT = VK_IMAGE_ASPECT_MEMORY_PLANE_3_BIT_EXT,
    ePlane0KHR       = VK_IMAGE_ASPECT_PLANE_0_BIT_KHR,
    ePlane1KHR       = VK_IMAGE_ASPECT_PLANE_1_BIT_KHR,
    ePlane2KHR       = VK_IMAGE_ASPECT_PLANE_2_BIT_KHR
  };

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
      case ImageAspectFlagBits::eMemoryPlane0EXT: return "MemoryPlane0EXT";
      case ImageAspectFlagBits::eMemoryPlane1EXT: return "MemoryPlane1EXT";
      case ImageAspectFlagBits::eMemoryPlane2EXT: return "MemoryPlane2EXT";
      case ImageAspectFlagBits::eMemoryPlane3EXT: return "MemoryPlane3EXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class SparseImageFormatFlagBits : VkSparseImageFormatFlags
  {
    eSingleMiptail        = VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT,
    eAlignedMipSize       = VK_SPARSE_IMAGE_FORMAT_ALIGNED_MIP_SIZE_BIT,
    eNonstandardBlockSize = VK_SPARSE_IMAGE_FORMAT_NONSTANDARD_BLOCK_SIZE_BIT
  };

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

  enum class SparseMemoryBindFlagBits : VkSparseMemoryBindFlags
  {
    eMetadata = VK_SPARSE_MEMORY_BIND_METADATA_BIT
  };

  VULKAN_HPP_INLINE std::string to_string( SparseMemoryBindFlagBits value )
  {
    switch ( value )
    {
      case SparseMemoryBindFlagBits::eMetadata: return "Metadata";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class FenceCreateFlagBits : VkFenceCreateFlags
  {
    eSignaled = VK_FENCE_CREATE_SIGNALED_BIT
  };

  VULKAN_HPP_INLINE std::string to_string( FenceCreateFlagBits value )
  {
    switch ( value )
    {
      case FenceCreateFlagBits::eSignaled: return "Signaled";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class EventCreateFlagBits : VkEventCreateFlags
  {
    eDeviceOnlyKHR = VK_EVENT_CREATE_DEVICE_ONLY_BIT_KHR
  };

  VULKAN_HPP_INLINE std::string to_string( EventCreateFlagBits value )
  {
    switch ( value )
    {
      case EventCreateFlagBits::eDeviceOnlyKHR: return "DeviceOnlyKHR";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class QueryPipelineStatisticFlagBits : VkQueryPipelineStatisticFlags
  {
    eInputAssemblyVertices            = VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT,
    eInputAssemblyPrimitives          = VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT,
    eVertexShaderInvocations          = VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT,
    eGeometryShaderInvocations        = VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT,
    eGeometryShaderPrimitives         = VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_PRIMITIVES_BIT,
    eClippingInvocations              = VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT,
    eClippingPrimitives               = VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT,
    eFragmentShaderInvocations        = VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT,
    eTessellationControlShaderPatches = VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT,
    eTessellationEvaluationShaderInvocations =
      VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_EVALUATION_SHADER_INVOCATIONS_BIT,
    eComputeShaderInvocations = VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT
  };

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
      case QueryPipelineStatisticFlagBits::eTessellationEvaluationShaderInvocations:
        return "TessellationEvaluationShaderInvocations";
      case QueryPipelineStatisticFlagBits::eComputeShaderInvocations: return "ComputeShaderInvocations";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class QueryResultFlagBits : VkQueryResultFlags
  {
    e64               = VK_QUERY_RESULT_64_BIT,
    eWait             = VK_QUERY_RESULT_WAIT_BIT,
    eWithAvailability = VK_QUERY_RESULT_WITH_AVAILABILITY_BIT,
    ePartial          = VK_QUERY_RESULT_PARTIAL_BIT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eWithStatusKHR = VK_QUERY_RESULT_WITH_STATUS_BIT_KHR
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
  };

  VULKAN_HPP_INLINE std::string to_string( QueryResultFlagBits value )
  {
    switch ( value )
    {
      case QueryResultFlagBits::e64: return "64";
      case QueryResultFlagBits::eWait: return "Wait";
      case QueryResultFlagBits::eWithAvailability: return "WithAvailability";
      case QueryResultFlagBits::ePartial: return "Partial";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case QueryResultFlagBits::eWithStatusKHR: return "WithStatusKHR";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class QueryType
  {
    eOcclusion          = VK_QUERY_TYPE_OCCLUSION,
    ePipelineStatistics = VK_QUERY_TYPE_PIPELINE_STATISTICS,
    eTimestamp          = VK_QUERY_TYPE_TIMESTAMP,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eResultStatusOnlyKHR = VK_QUERY_TYPE_RESULT_STATUS_ONLY_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eTransformFeedbackStreamEXT                = VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT,
    ePerformanceQueryKHR                       = VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR,
    eAccelerationStructureCompactedSizeKHR     = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
    eAccelerationStructureSerializationSizeKHR = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR,
    eAccelerationStructureCompactedSizeNV      = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_NV,
    ePerformanceQueryINTEL                     = VK_QUERY_TYPE_PERFORMANCE_QUERY_INTEL,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoEncodeBitstreamBufferRangeKHR = VK_QUERY_TYPE_VIDEO_ENCODE_BITSTREAM_BUFFER_RANGE_KHR
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
  };

  VULKAN_HPP_INLINE std::string to_string( QueryType value )
  {
    switch ( value )
    {
      case QueryType::eOcclusion: return "Occlusion";
      case QueryType::ePipelineStatistics: return "PipelineStatistics";
      case QueryType::eTimestamp: return "Timestamp";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case QueryType::eResultStatusOnlyKHR: return "ResultStatusOnlyKHR";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      case QueryType::eTransformFeedbackStreamEXT: return "TransformFeedbackStreamEXT";
      case QueryType::ePerformanceQueryKHR: return "PerformanceQueryKHR";
      case QueryType::eAccelerationStructureCompactedSizeKHR: return "AccelerationStructureCompactedSizeKHR";
      case QueryType::eAccelerationStructureSerializationSizeKHR: return "AccelerationStructureSerializationSizeKHR";
      case QueryType::eAccelerationStructureCompactedSizeNV: return "AccelerationStructureCompactedSizeNV";
      case QueryType::ePerformanceQueryINTEL: return "PerformanceQueryINTEL";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case QueryType::eVideoEncodeBitstreamBufferRangeKHR: return "VideoEncodeBitstreamBufferRangeKHR";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class BufferCreateFlagBits : VkBufferCreateFlags
  {
    eSparseBinding                 = VK_BUFFER_CREATE_SPARSE_BINDING_BIT,
    eSparseResidency               = VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT,
    eSparseAliased                 = VK_BUFFER_CREATE_SPARSE_ALIASED_BIT,
    eProtected                     = VK_BUFFER_CREATE_PROTECTED_BIT,
    eDeviceAddressCaptureReplay    = VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT,
    eDeviceAddressCaptureReplayEXT = VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT_EXT,
    eDeviceAddressCaptureReplayKHR = VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT_KHR
  };

  VULKAN_HPP_INLINE std::string to_string( BufferCreateFlagBits value )
  {
    switch ( value )
    {
      case BufferCreateFlagBits::eSparseBinding: return "SparseBinding";
      case BufferCreateFlagBits::eSparseResidency: return "SparseResidency";
      case BufferCreateFlagBits::eSparseAliased: return "SparseAliased";
      case BufferCreateFlagBits::eProtected: return "Protected";
      case BufferCreateFlagBits::eDeviceAddressCaptureReplay: return "DeviceAddressCaptureReplay";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class BufferUsageFlagBits : VkBufferUsageFlags
  {
    eTransferSrc         = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    eTransferDst         = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    eUniformTexelBuffer  = VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT,
    eStorageTexelBuffer  = VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT,
    eUniformBuffer       = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
    eStorageBuffer       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    eIndexBuffer         = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
    eVertexBuffer        = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
    eIndirectBuffer      = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
    eShaderDeviceAddress = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoDecodeSrcKHR = VK_BUFFER_USAGE_VIDEO_DECODE_SRC_BIT_KHR,
    eVideoDecodeDstKHR = VK_BUFFER_USAGE_VIDEO_DECODE_DST_BIT_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eTransformFeedbackBufferEXT                 = VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT,
    eTransformFeedbackCounterBufferEXT          = VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT,
    eConditionalRenderingEXT                    = VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT,
    eAccelerationStructureBuildInputReadOnlyKHR = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
    eAccelerationStructureStorageKHR            = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
    eShaderBindingTableKHR                      = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoEncodeDstKHR = VK_BUFFER_USAGE_VIDEO_ENCODE_DST_BIT_KHR,
    eVideoEncodeSrcKHR = VK_BUFFER_USAGE_VIDEO_ENCODE_SRC_BIT_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eRayTracingNV           = VK_BUFFER_USAGE_RAY_TRACING_BIT_NV,
    eShaderDeviceAddressEXT = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_EXT,
    eShaderDeviceAddressKHR = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR
  };

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
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case BufferUsageFlagBits::eVideoDecodeSrcKHR: return "VideoDecodeSrcKHR";
      case BufferUsageFlagBits::eVideoDecodeDstKHR: return "VideoDecodeDstKHR";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      case BufferUsageFlagBits::eTransformFeedbackBufferEXT: return "TransformFeedbackBufferEXT";
      case BufferUsageFlagBits::eTransformFeedbackCounterBufferEXT: return "TransformFeedbackCounterBufferEXT";
      case BufferUsageFlagBits::eConditionalRenderingEXT: return "ConditionalRenderingEXT";
      case BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR:
        return "AccelerationStructureBuildInputReadOnlyKHR";
      case BufferUsageFlagBits::eAccelerationStructureStorageKHR: return "AccelerationStructureStorageKHR";
      case BufferUsageFlagBits::eShaderBindingTableKHR: return "ShaderBindingTableKHR";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case BufferUsageFlagBits::eVideoEncodeDstKHR: return "VideoEncodeDstKHR";
      case BufferUsageFlagBits::eVideoEncodeSrcKHR: return "VideoEncodeSrcKHR";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class SharingMode
  {
    eExclusive  = VK_SHARING_MODE_EXCLUSIVE,
    eConcurrent = VK_SHARING_MODE_CONCURRENT
  };

  VULKAN_HPP_INLINE std::string to_string( SharingMode value )
  {
    switch ( value )
    {
      case SharingMode::eExclusive: return "Exclusive";
      case SharingMode::eConcurrent: return "Concurrent";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class ImageLayout
  {
    eUndefined                             = VK_IMAGE_LAYOUT_UNDEFINED,
    eGeneral                               = VK_IMAGE_LAYOUT_GENERAL,
    eColorAttachmentOptimal                = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    eDepthStencilAttachmentOptimal         = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    eDepthStencilReadOnlyOptimal           = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
    eShaderReadOnlyOptimal                 = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    eTransferSrcOptimal                    = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    eTransferDstOptimal                    = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    ePreinitialized                        = VK_IMAGE_LAYOUT_PREINITIALIZED,
    eDepthReadOnlyStencilAttachmentOptimal = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
    eDepthAttachmentStencilReadOnlyOptimal = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
    eDepthAttachmentOptimal                = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
    eDepthReadOnlyOptimal                  = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
    eStencilAttachmentOptimal              = VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL,
    eStencilReadOnlyOptimal                = VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL,
    ePresentSrcKHR                         = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoDecodeDstKHR = VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR,
    eVideoDecodeSrcKHR = VK_IMAGE_LAYOUT_VIDEO_DECODE_SRC_KHR,
    eVideoDecodeDpbKHR = VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eSharedPresentKHR                        = VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR,
    eFragmentDensityMapOptimalEXT            = VK_IMAGE_LAYOUT_FRAGMENT_DENSITY_MAP_OPTIMAL_EXT,
    eFragmentShadingRateAttachmentOptimalKHR = VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoEncodeDstKHR = VK_IMAGE_LAYOUT_VIDEO_ENCODE_DST_KHR,
    eVideoEncodeSrcKHR = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR,
    eVideoEncodeDpbKHR = VK_IMAGE_LAYOUT_VIDEO_ENCODE_DPB_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eReadOnlyOptimalKHR                       = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL_KHR,
    eAttachmentOptimalKHR                     = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR,
    eDepthAttachmentOptimalKHR                = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL_KHR,
    eDepthAttachmentStencilReadOnlyOptimalKHR = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL_KHR,
    eDepthReadOnlyOptimalKHR                  = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL_KHR,
    eDepthReadOnlyStencilAttachmentOptimalKHR = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL_KHR,
    eShadingRateOptimalNV                     = VK_IMAGE_LAYOUT_SHADING_RATE_OPTIMAL_NV,
    eStencilAttachmentOptimalKHR              = VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL_KHR,
    eStencilReadOnlyOptimalKHR                = VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL_KHR
  };

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
      case ImageLayout::ePresentSrcKHR: return "PresentSrcKHR";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case ImageLayout::eVideoDecodeDstKHR: return "VideoDecodeDstKHR";
      case ImageLayout::eVideoDecodeSrcKHR: return "VideoDecodeSrcKHR";
      case ImageLayout::eVideoDecodeDpbKHR: return "VideoDecodeDpbKHR";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      case ImageLayout::eSharedPresentKHR: return "SharedPresentKHR";
      case ImageLayout::eFragmentDensityMapOptimalEXT: return "FragmentDensityMapOptimalEXT";
      case ImageLayout::eFragmentShadingRateAttachmentOptimalKHR: return "FragmentShadingRateAttachmentOptimalKHR";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case ImageLayout::eVideoEncodeDstKHR: return "VideoEncodeDstKHR";
      case ImageLayout::eVideoEncodeSrcKHR: return "VideoEncodeSrcKHR";
      case ImageLayout::eVideoEncodeDpbKHR: return "VideoEncodeDpbKHR";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      case ImageLayout::eReadOnlyOptimalKHR: return "ReadOnlyOptimalKHR";
      case ImageLayout::eAttachmentOptimalKHR: return "AttachmentOptimalKHR";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class ComponentSwizzle
  {
    eIdentity = VK_COMPONENT_SWIZZLE_IDENTITY,
    eZero     = VK_COMPONENT_SWIZZLE_ZERO,
    eOne      = VK_COMPONENT_SWIZZLE_ONE,
    eR        = VK_COMPONENT_SWIZZLE_R,
    eG        = VK_COMPONENT_SWIZZLE_G,
    eB        = VK_COMPONENT_SWIZZLE_B,
    eA        = VK_COMPONENT_SWIZZLE_A
  };

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

  enum class ImageViewCreateFlagBits : VkImageViewCreateFlags
  {
    eFragmentDensityMapDynamicEXT  = VK_IMAGE_VIEW_CREATE_FRAGMENT_DENSITY_MAP_DYNAMIC_BIT_EXT,
    eFragmentDensityMapDeferredEXT = VK_IMAGE_VIEW_CREATE_FRAGMENT_DENSITY_MAP_DEFERRED_BIT_EXT
  };

  VULKAN_HPP_INLINE std::string to_string( ImageViewCreateFlagBits value )
  {
    switch ( value )
    {
      case ImageViewCreateFlagBits::eFragmentDensityMapDynamicEXT: return "FragmentDensityMapDynamicEXT";
      case ImageViewCreateFlagBits::eFragmentDensityMapDeferredEXT: return "FragmentDensityMapDeferredEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class ImageViewType
  {
    e1D        = VK_IMAGE_VIEW_TYPE_1D,
    e2D        = VK_IMAGE_VIEW_TYPE_2D,
    e3D        = VK_IMAGE_VIEW_TYPE_3D,
    eCube      = VK_IMAGE_VIEW_TYPE_CUBE,
    e1DArray   = VK_IMAGE_VIEW_TYPE_1D_ARRAY,
    e2DArray   = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
    eCubeArray = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY
  };

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

  enum class ShaderModuleCreateFlagBits : VkShaderModuleCreateFlags
  {
  };

  VULKAN_HPP_INLINE std::string to_string( ShaderModuleCreateFlagBits )
  {
    return "(void)";
  }

  enum class PipelineCacheCreateFlagBits : VkPipelineCacheCreateFlags
  {
    eExternallySynchronizedEXT = VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT_EXT
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineCacheCreateFlagBits value )
  {
    switch ( value )
    {
      case PipelineCacheCreateFlagBits::eExternallySynchronizedEXT: return "ExternallySynchronizedEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class BlendFactor
  {
    eZero                  = VK_BLEND_FACTOR_ZERO,
    eOne                   = VK_BLEND_FACTOR_ONE,
    eSrcColor              = VK_BLEND_FACTOR_SRC_COLOR,
    eOneMinusSrcColor      = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
    eDstColor              = VK_BLEND_FACTOR_DST_COLOR,
    eOneMinusDstColor      = VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
    eSrcAlpha              = VK_BLEND_FACTOR_SRC_ALPHA,
    eOneMinusSrcAlpha      = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
    eDstAlpha              = VK_BLEND_FACTOR_DST_ALPHA,
    eOneMinusDstAlpha      = VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
    eConstantColor         = VK_BLEND_FACTOR_CONSTANT_COLOR,
    eOneMinusConstantColor = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
    eConstantAlpha         = VK_BLEND_FACTOR_CONSTANT_ALPHA,
    eOneMinusConstantAlpha = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
    eSrcAlphaSaturate      = VK_BLEND_FACTOR_SRC_ALPHA_SATURATE,
    eSrc1Color             = VK_BLEND_FACTOR_SRC1_COLOR,
    eOneMinusSrc1Color     = VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR,
    eSrc1Alpha             = VK_BLEND_FACTOR_SRC1_ALPHA,
    eOneMinusSrc1Alpha     = VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA
  };

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

  enum class BlendOp
  {
    eAdd                 = VK_BLEND_OP_ADD,
    eSubtract            = VK_BLEND_OP_SUBTRACT,
    eReverseSubtract     = VK_BLEND_OP_REVERSE_SUBTRACT,
    eMin                 = VK_BLEND_OP_MIN,
    eMax                 = VK_BLEND_OP_MAX,
    eZeroEXT             = VK_BLEND_OP_ZERO_EXT,
    eSrcEXT              = VK_BLEND_OP_SRC_EXT,
    eDstEXT              = VK_BLEND_OP_DST_EXT,
    eSrcOverEXT          = VK_BLEND_OP_SRC_OVER_EXT,
    eDstOverEXT          = VK_BLEND_OP_DST_OVER_EXT,
    eSrcInEXT            = VK_BLEND_OP_SRC_IN_EXT,
    eDstInEXT            = VK_BLEND_OP_DST_IN_EXT,
    eSrcOutEXT           = VK_BLEND_OP_SRC_OUT_EXT,
    eDstOutEXT           = VK_BLEND_OP_DST_OUT_EXT,
    eSrcAtopEXT          = VK_BLEND_OP_SRC_ATOP_EXT,
    eDstAtopEXT          = VK_BLEND_OP_DST_ATOP_EXT,
    eXorEXT              = VK_BLEND_OP_XOR_EXT,
    eMultiplyEXT         = VK_BLEND_OP_MULTIPLY_EXT,
    eScreenEXT           = VK_BLEND_OP_SCREEN_EXT,
    eOverlayEXT          = VK_BLEND_OP_OVERLAY_EXT,
    eDarkenEXT           = VK_BLEND_OP_DARKEN_EXT,
    eLightenEXT          = VK_BLEND_OP_LIGHTEN_EXT,
    eColordodgeEXT       = VK_BLEND_OP_COLORDODGE_EXT,
    eColorburnEXT        = VK_BLEND_OP_COLORBURN_EXT,
    eHardlightEXT        = VK_BLEND_OP_HARDLIGHT_EXT,
    eSoftlightEXT        = VK_BLEND_OP_SOFTLIGHT_EXT,
    eDifferenceEXT       = VK_BLEND_OP_DIFFERENCE_EXT,
    eExclusionEXT        = VK_BLEND_OP_EXCLUSION_EXT,
    eInvertEXT           = VK_BLEND_OP_INVERT_EXT,
    eInvertRgbEXT        = VK_BLEND_OP_INVERT_RGB_EXT,
    eLineardodgeEXT      = VK_BLEND_OP_LINEARDODGE_EXT,
    eLinearburnEXT       = VK_BLEND_OP_LINEARBURN_EXT,
    eVividlightEXT       = VK_BLEND_OP_VIVIDLIGHT_EXT,
    eLinearlightEXT      = VK_BLEND_OP_LINEARLIGHT_EXT,
    ePinlightEXT         = VK_BLEND_OP_PINLIGHT_EXT,
    eHardmixEXT          = VK_BLEND_OP_HARDMIX_EXT,
    eHslHueEXT           = VK_BLEND_OP_HSL_HUE_EXT,
    eHslSaturationEXT    = VK_BLEND_OP_HSL_SATURATION_EXT,
    eHslColorEXT         = VK_BLEND_OP_HSL_COLOR_EXT,
    eHslLuminosityEXT    = VK_BLEND_OP_HSL_LUMINOSITY_EXT,
    ePlusEXT             = VK_BLEND_OP_PLUS_EXT,
    ePlusClampedEXT      = VK_BLEND_OP_PLUS_CLAMPED_EXT,
    ePlusClampedAlphaEXT = VK_BLEND_OP_PLUS_CLAMPED_ALPHA_EXT,
    ePlusDarkerEXT       = VK_BLEND_OP_PLUS_DARKER_EXT,
    eMinusEXT            = VK_BLEND_OP_MINUS_EXT,
    eMinusClampedEXT     = VK_BLEND_OP_MINUS_CLAMPED_EXT,
    eContrastEXT         = VK_BLEND_OP_CONTRAST_EXT,
    eInvertOvgEXT        = VK_BLEND_OP_INVERT_OVG_EXT,
    eRedEXT              = VK_BLEND_OP_RED_EXT,
    eGreenEXT            = VK_BLEND_OP_GREEN_EXT,
    eBlueEXT             = VK_BLEND_OP_BLUE_EXT
  };

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

  enum class ColorComponentFlagBits : VkColorComponentFlags
  {
    eR = VK_COLOR_COMPONENT_R_BIT,
    eG = VK_COLOR_COMPONENT_G_BIT,
    eB = VK_COLOR_COMPONENT_B_BIT,
    eA = VK_COLOR_COMPONENT_A_BIT
  };

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

  enum class CompareOp
  {
    eNever          = VK_COMPARE_OP_NEVER,
    eLess           = VK_COMPARE_OP_LESS,
    eEqual          = VK_COMPARE_OP_EQUAL,
    eLessOrEqual    = VK_COMPARE_OP_LESS_OR_EQUAL,
    eGreater        = VK_COMPARE_OP_GREATER,
    eNotEqual       = VK_COMPARE_OP_NOT_EQUAL,
    eGreaterOrEqual = VK_COMPARE_OP_GREATER_OR_EQUAL,
    eAlways         = VK_COMPARE_OP_ALWAYS
  };

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

  enum class CullModeFlagBits : VkCullModeFlags
  {
    eNone         = VK_CULL_MODE_NONE,
    eFront        = VK_CULL_MODE_FRONT_BIT,
    eBack         = VK_CULL_MODE_BACK_BIT,
    eFrontAndBack = VK_CULL_MODE_FRONT_AND_BACK
  };

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

  enum class DynamicState
  {
    eViewport                       = VK_DYNAMIC_STATE_VIEWPORT,
    eScissor                        = VK_DYNAMIC_STATE_SCISSOR,
    eLineWidth                      = VK_DYNAMIC_STATE_LINE_WIDTH,
    eDepthBias                      = VK_DYNAMIC_STATE_DEPTH_BIAS,
    eBlendConstants                 = VK_DYNAMIC_STATE_BLEND_CONSTANTS,
    eDepthBounds                    = VK_DYNAMIC_STATE_DEPTH_BOUNDS,
    eStencilCompareMask             = VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
    eStencilWriteMask               = VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
    eStencilReference               = VK_DYNAMIC_STATE_STENCIL_REFERENCE,
    eViewportWScalingNV             = VK_DYNAMIC_STATE_VIEWPORT_W_SCALING_NV,
    eDiscardRectangleEXT            = VK_DYNAMIC_STATE_DISCARD_RECTANGLE_EXT,
    eSampleLocationsEXT             = VK_DYNAMIC_STATE_SAMPLE_LOCATIONS_EXT,
    eRayTracingPipelineStackSizeKHR = VK_DYNAMIC_STATE_RAY_TRACING_PIPELINE_STACK_SIZE_KHR,
    eViewportShadingRatePaletteNV   = VK_DYNAMIC_STATE_VIEWPORT_SHADING_RATE_PALETTE_NV,
    eViewportCoarseSampleOrderNV    = VK_DYNAMIC_STATE_VIEWPORT_COARSE_SAMPLE_ORDER_NV,
    eExclusiveScissorNV             = VK_DYNAMIC_STATE_EXCLUSIVE_SCISSOR_NV,
    eFragmentShadingRateKHR         = VK_DYNAMIC_STATE_FRAGMENT_SHADING_RATE_KHR,
    eLineStippleEXT                 = VK_DYNAMIC_STATE_LINE_STIPPLE_EXT,
    eCullModeEXT                    = VK_DYNAMIC_STATE_CULL_MODE_EXT,
    eFrontFaceEXT                   = VK_DYNAMIC_STATE_FRONT_FACE_EXT,
    ePrimitiveTopologyEXT           = VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY_EXT,
    eViewportWithCountEXT           = VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT_EXT,
    eScissorWithCountEXT            = VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT_EXT,
    eVertexInputBindingStrideEXT    = VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE_EXT,
    eDepthTestEnableEXT             = VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE_EXT,
    eDepthWriteEnableEXT            = VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE_EXT,
    eDepthCompareOpEXT              = VK_DYNAMIC_STATE_DEPTH_COMPARE_OP_EXT,
    eDepthBoundsTestEnableEXT       = VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE_EXT,
    eStencilTestEnableEXT           = VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE_EXT,
    eStencilOpEXT                   = VK_DYNAMIC_STATE_STENCIL_OP_EXT,
    eVertexInputEXT                 = VK_DYNAMIC_STATE_VERTEX_INPUT_EXT,
    ePatchControlPointsEXT          = VK_DYNAMIC_STATE_PATCH_CONTROL_POINTS_EXT,
    eRasterizerDiscardEnableEXT     = VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE_EXT,
    eDepthBiasEnableEXT             = VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE_EXT,
    eLogicOpEXT                     = VK_DYNAMIC_STATE_LOGIC_OP_EXT,
    ePrimitiveRestartEnableEXT      = VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE_EXT,
    eColorWriteEnableEXT            = VK_DYNAMIC_STATE_COLOR_WRITE_ENABLE_EXT
  };

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
      case DynamicState::eViewportWScalingNV: return "ViewportWScalingNV";
      case DynamicState::eDiscardRectangleEXT: return "DiscardRectangleEXT";
      case DynamicState::eSampleLocationsEXT: return "SampleLocationsEXT";
      case DynamicState::eRayTracingPipelineStackSizeKHR: return "RayTracingPipelineStackSizeKHR";
      case DynamicState::eViewportShadingRatePaletteNV: return "ViewportShadingRatePaletteNV";
      case DynamicState::eViewportCoarseSampleOrderNV: return "ViewportCoarseSampleOrderNV";
      case DynamicState::eExclusiveScissorNV: return "ExclusiveScissorNV";
      case DynamicState::eFragmentShadingRateKHR: return "FragmentShadingRateKHR";
      case DynamicState::eLineStippleEXT: return "LineStippleEXT";
      case DynamicState::eCullModeEXT: return "CullModeEXT";
      case DynamicState::eFrontFaceEXT: return "FrontFaceEXT";
      case DynamicState::ePrimitiveTopologyEXT: return "PrimitiveTopologyEXT";
      case DynamicState::eViewportWithCountEXT: return "ViewportWithCountEXT";
      case DynamicState::eScissorWithCountEXT: return "ScissorWithCountEXT";
      case DynamicState::eVertexInputBindingStrideEXT: return "VertexInputBindingStrideEXT";
      case DynamicState::eDepthTestEnableEXT: return "DepthTestEnableEXT";
      case DynamicState::eDepthWriteEnableEXT: return "DepthWriteEnableEXT";
      case DynamicState::eDepthCompareOpEXT: return "DepthCompareOpEXT";
      case DynamicState::eDepthBoundsTestEnableEXT: return "DepthBoundsTestEnableEXT";
      case DynamicState::eStencilTestEnableEXT: return "StencilTestEnableEXT";
      case DynamicState::eStencilOpEXT: return "StencilOpEXT";
      case DynamicState::eVertexInputEXT: return "VertexInputEXT";
      case DynamicState::ePatchControlPointsEXT: return "PatchControlPointsEXT";
      case DynamicState::eRasterizerDiscardEnableEXT: return "RasterizerDiscardEnableEXT";
      case DynamicState::eDepthBiasEnableEXT: return "DepthBiasEnableEXT";
      case DynamicState::eLogicOpEXT: return "LogicOpEXT";
      case DynamicState::ePrimitiveRestartEnableEXT: return "PrimitiveRestartEnableEXT";
      case DynamicState::eColorWriteEnableEXT: return "ColorWriteEnableEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class FrontFace
  {
    eCounterClockwise = VK_FRONT_FACE_COUNTER_CLOCKWISE,
    eClockwise        = VK_FRONT_FACE_CLOCKWISE
  };

  VULKAN_HPP_INLINE std::string to_string( FrontFace value )
  {
    switch ( value )
    {
      case FrontFace::eCounterClockwise: return "CounterClockwise";
      case FrontFace::eClockwise: return "Clockwise";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class LogicOp
  {
    eClear        = VK_LOGIC_OP_CLEAR,
    eAnd          = VK_LOGIC_OP_AND,
    eAndReverse   = VK_LOGIC_OP_AND_REVERSE,
    eCopy         = VK_LOGIC_OP_COPY,
    eAndInverted  = VK_LOGIC_OP_AND_INVERTED,
    eNoOp         = VK_LOGIC_OP_NO_OP,
    eXor          = VK_LOGIC_OP_XOR,
    eOr           = VK_LOGIC_OP_OR,
    eNor          = VK_LOGIC_OP_NOR,
    eEquivalent   = VK_LOGIC_OP_EQUIVALENT,
    eInvert       = VK_LOGIC_OP_INVERT,
    eOrReverse    = VK_LOGIC_OP_OR_REVERSE,
    eCopyInverted = VK_LOGIC_OP_COPY_INVERTED,
    eOrInverted   = VK_LOGIC_OP_OR_INVERTED,
    eNand         = VK_LOGIC_OP_NAND,
    eSet          = VK_LOGIC_OP_SET
  };

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

  enum class PipelineCreateFlagBits : VkPipelineCreateFlags
  {
    eDisableOptimization                    = VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT,
    eAllowDerivatives                       = VK_PIPELINE_CREATE_ALLOW_DERIVATIVES_BIT,
    eDerivative                             = VK_PIPELINE_CREATE_DERIVATIVE_BIT,
    eViewIndexFromDeviceIndex               = VK_PIPELINE_CREATE_VIEW_INDEX_FROM_DEVICE_INDEX_BIT,
    eDispatchBase                           = VK_PIPELINE_CREATE_DISPATCH_BASE_BIT,
    eRayTracingNoNullAnyHitShadersKHR       = VK_PIPELINE_CREATE_RAY_TRACING_NO_NULL_ANY_HIT_SHADERS_BIT_KHR,
    eRayTracingNoNullClosestHitShadersKHR   = VK_PIPELINE_CREATE_RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS_BIT_KHR,
    eRayTracingNoNullMissShadersKHR         = VK_PIPELINE_CREATE_RAY_TRACING_NO_NULL_MISS_SHADERS_BIT_KHR,
    eRayTracingNoNullIntersectionShadersKHR = VK_PIPELINE_CREATE_RAY_TRACING_NO_NULL_INTERSECTION_SHADERS_BIT_KHR,
    eRayTracingSkipTrianglesKHR             = VK_PIPELINE_CREATE_RAY_TRACING_SKIP_TRIANGLES_BIT_KHR,
    eRayTracingSkipAabbsKHR                 = VK_PIPELINE_CREATE_RAY_TRACING_SKIP_AABBS_BIT_KHR,
    eRayTracingShaderGroupHandleCaptureReplayKHR =
      VK_PIPELINE_CREATE_RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY_BIT_KHR,
    eDeferCompileNV                    = VK_PIPELINE_CREATE_DEFER_COMPILE_BIT_NV,
    eCaptureStatisticsKHR              = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR,
    eCaptureInternalRepresentationsKHR = VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR,
    eIndirectBindableNV                = VK_PIPELINE_CREATE_INDIRECT_BINDABLE_BIT_NV,
    eLibraryKHR                        = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR,
    eFailOnPipelineCompileRequiredEXT  = VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_EXT,
    eEarlyReturnOnFailureEXT           = VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT_EXT,
    eRayTracingAllowMotionNV           = VK_PIPELINE_CREATE_RAY_TRACING_ALLOW_MOTION_BIT_NV,
    eDispatchBaseKHR                   = VK_PIPELINE_CREATE_DISPATCH_BASE_KHR,
    eViewIndexFromDeviceIndexKHR       = VK_PIPELINE_CREATE_VIEW_INDEX_FROM_DEVICE_INDEX_BIT_KHR
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineCreateFlagBits value )
  {
    switch ( value )
    {
      case PipelineCreateFlagBits::eDisableOptimization: return "DisableOptimization";
      case PipelineCreateFlagBits::eAllowDerivatives: return "AllowDerivatives";
      case PipelineCreateFlagBits::eDerivative: return "Derivative";
      case PipelineCreateFlagBits::eViewIndexFromDeviceIndex: return "ViewIndexFromDeviceIndex";
      case PipelineCreateFlagBits::eDispatchBase: return "DispatchBase";
      case PipelineCreateFlagBits::eRayTracingNoNullAnyHitShadersKHR: return "RayTracingNoNullAnyHitShadersKHR";
      case PipelineCreateFlagBits::eRayTracingNoNullClosestHitShadersKHR: return "RayTracingNoNullClosestHitShadersKHR";
      case PipelineCreateFlagBits::eRayTracingNoNullMissShadersKHR: return "RayTracingNoNullMissShadersKHR";
      case PipelineCreateFlagBits::eRayTracingNoNullIntersectionShadersKHR:
        return "RayTracingNoNullIntersectionShadersKHR";
      case PipelineCreateFlagBits::eRayTracingSkipTrianglesKHR: return "RayTracingSkipTrianglesKHR";
      case PipelineCreateFlagBits::eRayTracingSkipAabbsKHR: return "RayTracingSkipAabbsKHR";
      case PipelineCreateFlagBits::eRayTracingShaderGroupHandleCaptureReplayKHR:
        return "RayTracingShaderGroupHandleCaptureReplayKHR";
      case PipelineCreateFlagBits::eDeferCompileNV: return "DeferCompileNV";
      case PipelineCreateFlagBits::eCaptureStatisticsKHR: return "CaptureStatisticsKHR";
      case PipelineCreateFlagBits::eCaptureInternalRepresentationsKHR: return "CaptureInternalRepresentationsKHR";
      case PipelineCreateFlagBits::eIndirectBindableNV: return "IndirectBindableNV";
      case PipelineCreateFlagBits::eLibraryKHR: return "LibraryKHR";
      case PipelineCreateFlagBits::eFailOnPipelineCompileRequiredEXT: return "FailOnPipelineCompileRequiredEXT";
      case PipelineCreateFlagBits::eEarlyReturnOnFailureEXT: return "EarlyReturnOnFailureEXT";
      case PipelineCreateFlagBits::eRayTracingAllowMotionNV: return "RayTracingAllowMotionNV";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class PipelineShaderStageCreateFlagBits : VkPipelineShaderStageCreateFlags
  {
    eAllowVaryingSubgroupSizeEXT = VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT_EXT,
    eRequireFullSubgroupsEXT     = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineShaderStageCreateFlagBits value )
  {
    switch ( value )
    {
      case PipelineShaderStageCreateFlagBits::eAllowVaryingSubgroupSizeEXT: return "AllowVaryingSubgroupSizeEXT";
      case PipelineShaderStageCreateFlagBits::eRequireFullSubgroupsEXT: return "RequireFullSubgroupsEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class PolygonMode
  {
    eFill            = VK_POLYGON_MODE_FILL,
    eLine            = VK_POLYGON_MODE_LINE,
    ePoint           = VK_POLYGON_MODE_POINT,
    eFillRectangleNV = VK_POLYGON_MODE_FILL_RECTANGLE_NV
  };

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

  enum class PrimitiveTopology
  {
    ePointList                  = VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
    eLineList                   = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
    eLineStrip                  = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,
    eTriangleList               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    eTriangleStrip              = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
    eTriangleFan                = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
    eLineListWithAdjacency      = VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY,
    eLineStripWithAdjacency     = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY,
    eTriangleListWithAdjacency  = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY,
    eTriangleStripWithAdjacency = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY,
    ePatchList                  = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST
  };

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

  enum class ShaderStageFlagBits : VkShaderStageFlags
  {
    eVertex                 = VK_SHADER_STAGE_VERTEX_BIT,
    eTessellationControl    = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
    eTessellationEvaluation = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
    eGeometry               = VK_SHADER_STAGE_GEOMETRY_BIT,
    eFragment               = VK_SHADER_STAGE_FRAGMENT_BIT,
    eCompute                = VK_SHADER_STAGE_COMPUTE_BIT,
    eAllGraphics            = VK_SHADER_STAGE_ALL_GRAPHICS,
    eAll                    = VK_SHADER_STAGE_ALL,
    eRaygenKHR              = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
    eAnyHitKHR              = VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
    eClosestHitKHR          = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
    eMissKHR                = VK_SHADER_STAGE_MISS_BIT_KHR,
    eIntersectionKHR        = VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
    eCallableKHR            = VK_SHADER_STAGE_CALLABLE_BIT_KHR,
    eTaskNV                 = VK_SHADER_STAGE_TASK_BIT_NV,
    eMeshNV                 = VK_SHADER_STAGE_MESH_BIT_NV,
    eSubpassShadingHUAWEI   = VK_SHADER_STAGE_SUBPASS_SHADING_BIT_HUAWEI,
    eAnyHitNV               = VK_SHADER_STAGE_ANY_HIT_BIT_NV,
    eCallableNV             = VK_SHADER_STAGE_CALLABLE_BIT_NV,
    eClosestHitNV           = VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV,
    eIntersectionNV         = VK_SHADER_STAGE_INTERSECTION_BIT_NV,
    eMissNV                 = VK_SHADER_STAGE_MISS_BIT_NV,
    eRaygenNV               = VK_SHADER_STAGE_RAYGEN_BIT_NV
  };

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
      case ShaderStageFlagBits::eTaskNV: return "TaskNV";
      case ShaderStageFlagBits::eMeshNV: return "MeshNV";
      case ShaderStageFlagBits::eSubpassShadingHUAWEI: return "SubpassShadingHUAWEI";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class StencilOp
  {
    eKeep              = VK_STENCIL_OP_KEEP,
    eZero              = VK_STENCIL_OP_ZERO,
    eReplace           = VK_STENCIL_OP_REPLACE,
    eIncrementAndClamp = VK_STENCIL_OP_INCREMENT_AND_CLAMP,
    eDecrementAndClamp = VK_STENCIL_OP_DECREMENT_AND_CLAMP,
    eInvert            = VK_STENCIL_OP_INVERT,
    eIncrementAndWrap  = VK_STENCIL_OP_INCREMENT_AND_WRAP,
    eDecrementAndWrap  = VK_STENCIL_OP_DECREMENT_AND_WRAP
  };

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

  enum class VertexInputRate
  {
    eVertex   = VK_VERTEX_INPUT_RATE_VERTEX,
    eInstance = VK_VERTEX_INPUT_RATE_INSTANCE
  };

  VULKAN_HPP_INLINE std::string to_string( VertexInputRate value )
  {
    switch ( value )
    {
      case VertexInputRate::eVertex: return "Vertex";
      case VertexInputRate::eInstance: return "Instance";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class BorderColor
  {
    eFloatTransparentBlack = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
    eIntTransparentBlack   = VK_BORDER_COLOR_INT_TRANSPARENT_BLACK,
    eFloatOpaqueBlack      = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
    eIntOpaqueBlack        = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
    eFloatOpaqueWhite      = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
    eIntOpaqueWhite        = VK_BORDER_COLOR_INT_OPAQUE_WHITE,
    eFloatCustomEXT        = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT,
    eIntCustomEXT          = VK_BORDER_COLOR_INT_CUSTOM_EXT
  };

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

  enum class Filter
  {
    eNearest  = VK_FILTER_NEAREST,
    eLinear   = VK_FILTER_LINEAR,
    eCubicIMG = VK_FILTER_CUBIC_IMG,
    eCubicEXT = VK_FILTER_CUBIC_EXT
  };

  VULKAN_HPP_INLINE std::string to_string( Filter value )
  {
    switch ( value )
    {
      case Filter::eNearest: return "Nearest";
      case Filter::eLinear: return "Linear";
      case Filter::eCubicIMG: return "CubicIMG";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class SamplerAddressMode
  {
    eRepeat               = VK_SAMPLER_ADDRESS_MODE_REPEAT,
    eMirroredRepeat       = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
    eClampToEdge          = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    eClampToBorder        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
    eMirrorClampToEdge    = VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
    eMirrorClampToEdgeKHR = VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE_KHR
  };

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

  enum class SamplerCreateFlagBits : VkSamplerCreateFlags
  {
    eSubsampledEXT                     = VK_SAMPLER_CREATE_SUBSAMPLED_BIT_EXT,
    eSubsampledCoarseReconstructionEXT = VK_SAMPLER_CREATE_SUBSAMPLED_COARSE_RECONSTRUCTION_BIT_EXT
  };

  VULKAN_HPP_INLINE std::string to_string( SamplerCreateFlagBits value )
  {
    switch ( value )
    {
      case SamplerCreateFlagBits::eSubsampledEXT: return "SubsampledEXT";
      case SamplerCreateFlagBits::eSubsampledCoarseReconstructionEXT: return "SubsampledCoarseReconstructionEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class SamplerMipmapMode
  {
    eNearest = VK_SAMPLER_MIPMAP_MODE_NEAREST,
    eLinear  = VK_SAMPLER_MIPMAP_MODE_LINEAR
  };

  VULKAN_HPP_INLINE std::string to_string( SamplerMipmapMode value )
  {
    switch ( value )
    {
      case SamplerMipmapMode::eNearest: return "Nearest";
      case SamplerMipmapMode::eLinear: return "Linear";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class DescriptorPoolCreateFlagBits : VkDescriptorPoolCreateFlags
  {
    eFreeDescriptorSet  = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
    eUpdateAfterBind    = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
    eHostOnlyVALVE      = VK_DESCRIPTOR_POOL_CREATE_HOST_ONLY_BIT_VALVE,
    eUpdateAfterBindEXT = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT_EXT
  };

  VULKAN_HPP_INLINE std::string to_string( DescriptorPoolCreateFlagBits value )
  {
    switch ( value )
    {
      case DescriptorPoolCreateFlagBits::eFreeDescriptorSet: return "FreeDescriptorSet";
      case DescriptorPoolCreateFlagBits::eUpdateAfterBind: return "UpdateAfterBind";
      case DescriptorPoolCreateFlagBits::eHostOnlyVALVE: return "HostOnlyVALVE";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class DescriptorSetLayoutCreateFlagBits : VkDescriptorSetLayoutCreateFlags
  {
    eUpdateAfterBindPool    = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
    ePushDescriptorKHR      = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
    eHostOnlyPoolVALVE      = VK_DESCRIPTOR_SET_LAYOUT_CREATE_HOST_ONLY_POOL_BIT_VALVE,
    eUpdateAfterBindPoolEXT = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT_EXT
  };

  VULKAN_HPP_INLINE std::string to_string( DescriptorSetLayoutCreateFlagBits value )
  {
    switch ( value )
    {
      case DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool: return "UpdateAfterBindPool";
      case DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR: return "PushDescriptorKHR";
      case DescriptorSetLayoutCreateFlagBits::eHostOnlyPoolVALVE: return "HostOnlyPoolVALVE";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class DescriptorType
  {
    eSampler                  = VK_DESCRIPTOR_TYPE_SAMPLER,
    eCombinedImageSampler     = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    eSampledImage             = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
    eStorageImage             = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
    eUniformTexelBuffer       = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
    eStorageTexelBuffer       = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
    eUniformBuffer            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    eStorageBuffer            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    eUniformBufferDynamic     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
    eStorageBufferDynamic     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,
    eInputAttachment          = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
    eInlineUniformBlockEXT    = VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT,
    eAccelerationStructureKHR = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
    eAccelerationStructureNV  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV,
    eMutableVALVE             = VK_DESCRIPTOR_TYPE_MUTABLE_VALVE
  };

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
      case DescriptorType::eInlineUniformBlockEXT: return "InlineUniformBlockEXT";
      case DescriptorType::eAccelerationStructureKHR: return "AccelerationStructureKHR";
      case DescriptorType::eAccelerationStructureNV: return "AccelerationStructureNV";
      case DescriptorType::eMutableVALVE: return "MutableVALVE";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class AccessFlagBits : VkAccessFlags
  {
    eIndirectCommandRead                  = VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
    eIndexRead                            = VK_ACCESS_INDEX_READ_BIT,
    eVertexAttributeRead                  = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
    eUniformRead                          = VK_ACCESS_UNIFORM_READ_BIT,
    eInputAttachmentRead                  = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT,
    eShaderRead                           = VK_ACCESS_SHADER_READ_BIT,
    eShaderWrite                          = VK_ACCESS_SHADER_WRITE_BIT,
    eColorAttachmentRead                  = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
    eColorAttachmentWrite                 = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    eDepthStencilAttachmentRead           = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
    eDepthStencilAttachmentWrite          = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
    eTransferRead                         = VK_ACCESS_TRANSFER_READ_BIT,
    eTransferWrite                        = VK_ACCESS_TRANSFER_WRITE_BIT,
    eHostRead                             = VK_ACCESS_HOST_READ_BIT,
    eHostWrite                            = VK_ACCESS_HOST_WRITE_BIT,
    eMemoryRead                           = VK_ACCESS_MEMORY_READ_BIT,
    eMemoryWrite                          = VK_ACCESS_MEMORY_WRITE_BIT,
    eTransformFeedbackWriteEXT            = VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT,
    eTransformFeedbackCounterReadEXT      = VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT,
    eTransformFeedbackCounterWriteEXT     = VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT,
    eConditionalRenderingReadEXT          = VK_ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT,
    eColorAttachmentReadNoncoherentEXT    = VK_ACCESS_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT,
    eAccelerationStructureReadKHR         = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    eAccelerationStructureWriteKHR        = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
    eFragmentDensityMapReadEXT            = VK_ACCESS_FRAGMENT_DENSITY_MAP_READ_BIT_EXT,
    eFragmentShadingRateAttachmentReadKHR = VK_ACCESS_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR,
    eCommandPreprocessReadNV              = VK_ACCESS_COMMAND_PREPROCESS_READ_BIT_NV,
    eCommandPreprocessWriteNV             = VK_ACCESS_COMMAND_PREPROCESS_WRITE_BIT_NV,
    eNoneKHR                              = VK_ACCESS_NONE_KHR,
    eAccelerationStructureReadNV          = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV,
    eAccelerationStructureWriteNV         = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV,
    eShadingRateImageReadNV               = VK_ACCESS_SHADING_RATE_IMAGE_READ_BIT_NV
  };

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
      case AccessFlagBits::eNoneKHR: return "NoneKHR";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class AttachmentDescriptionFlagBits : VkAttachmentDescriptionFlags
  {
    eMayAlias = VK_ATTACHMENT_DESCRIPTION_MAY_ALIAS_BIT
  };

  VULKAN_HPP_INLINE std::string to_string( AttachmentDescriptionFlagBits value )
  {
    switch ( value )
    {
      case AttachmentDescriptionFlagBits::eMayAlias: return "MayAlias";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class AttachmentLoadOp
  {
    eLoad     = VK_ATTACHMENT_LOAD_OP_LOAD,
    eClear    = VK_ATTACHMENT_LOAD_OP_CLEAR,
    eDontCare = VK_ATTACHMENT_LOAD_OP_DONT_CARE
  };

  VULKAN_HPP_INLINE std::string to_string( AttachmentLoadOp value )
  {
    switch ( value )
    {
      case AttachmentLoadOp::eLoad: return "Load";
      case AttachmentLoadOp::eClear: return "Clear";
      case AttachmentLoadOp::eDontCare: return "DontCare";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class AttachmentStoreOp
  {
    eStore    = VK_ATTACHMENT_STORE_OP_STORE,
    eDontCare = VK_ATTACHMENT_STORE_OP_DONT_CARE,
    eNoneQCOM = VK_ATTACHMENT_STORE_OP_NONE_QCOM
  };

  VULKAN_HPP_INLINE std::string to_string( AttachmentStoreOp value )
  {
    switch ( value )
    {
      case AttachmentStoreOp::eStore: return "Store";
      case AttachmentStoreOp::eDontCare: return "DontCare";
      case AttachmentStoreOp::eNoneQCOM: return "NoneQCOM";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class DependencyFlagBits : VkDependencyFlags
  {
    eByRegion       = VK_DEPENDENCY_BY_REGION_BIT,
    eDeviceGroup    = VK_DEPENDENCY_DEVICE_GROUP_BIT,
    eViewLocal      = VK_DEPENDENCY_VIEW_LOCAL_BIT,
    eDeviceGroupKHR = VK_DEPENDENCY_DEVICE_GROUP_BIT_KHR,
    eViewLocalKHR   = VK_DEPENDENCY_VIEW_LOCAL_BIT_KHR
  };

  VULKAN_HPP_INLINE std::string to_string( DependencyFlagBits value )
  {
    switch ( value )
    {
      case DependencyFlagBits::eByRegion: return "ByRegion";
      case DependencyFlagBits::eDeviceGroup: return "DeviceGroup";
      case DependencyFlagBits::eViewLocal: return "ViewLocal";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class FramebufferCreateFlagBits : VkFramebufferCreateFlags
  {
    eImageless    = VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT,
    eImagelessKHR = VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT_KHR
  };

  VULKAN_HPP_INLINE std::string to_string( FramebufferCreateFlagBits value )
  {
    switch ( value )
    {
      case FramebufferCreateFlagBits::eImageless: return "Imageless";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class PipelineBindPoint
  {
    eGraphics             = VK_PIPELINE_BIND_POINT_GRAPHICS,
    eCompute              = VK_PIPELINE_BIND_POINT_COMPUTE,
    eRayTracingKHR        = VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
    eSubpassShadingHUAWEI = VK_PIPELINE_BIND_POINT_SUBPASS_SHADING_HUAWEI,
    eRayTracingNV         = VK_PIPELINE_BIND_POINT_RAY_TRACING_NV
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineBindPoint value )
  {
    switch ( value )
    {
      case PipelineBindPoint::eGraphics: return "Graphics";
      case PipelineBindPoint::eCompute: return "Compute";
      case PipelineBindPoint::eRayTracingKHR: return "RayTracingKHR";
      case PipelineBindPoint::eSubpassShadingHUAWEI: return "SubpassShadingHUAWEI";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class RenderPassCreateFlagBits : VkRenderPassCreateFlags
  {
    eTransformQCOM = VK_RENDER_PASS_CREATE_TRANSFORM_BIT_QCOM
  };

  VULKAN_HPP_INLINE std::string to_string( RenderPassCreateFlagBits value )
  {
    switch ( value )
    {
      case RenderPassCreateFlagBits::eTransformQCOM: return "TransformQCOM";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class SubpassDescriptionFlagBits : VkSubpassDescriptionFlags
  {
    ePerViewAttributesNVX    = VK_SUBPASS_DESCRIPTION_PER_VIEW_ATTRIBUTES_BIT_NVX,
    ePerViewPositionXOnlyNVX = VK_SUBPASS_DESCRIPTION_PER_VIEW_POSITION_X_ONLY_BIT_NVX,
    eFragmentRegionQCOM      = VK_SUBPASS_DESCRIPTION_FRAGMENT_REGION_BIT_QCOM,
    eShaderResolveQCOM       = VK_SUBPASS_DESCRIPTION_SHADER_RESOLVE_BIT_QCOM
  };

  VULKAN_HPP_INLINE std::string to_string( SubpassDescriptionFlagBits value )
  {
    switch ( value )
    {
      case SubpassDescriptionFlagBits::ePerViewAttributesNVX: return "PerViewAttributesNVX";
      case SubpassDescriptionFlagBits::ePerViewPositionXOnlyNVX: return "PerViewPositionXOnlyNVX";
      case SubpassDescriptionFlagBits::eFragmentRegionQCOM: return "FragmentRegionQCOM";
      case SubpassDescriptionFlagBits::eShaderResolveQCOM: return "ShaderResolveQCOM";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class CommandPoolCreateFlagBits : VkCommandPoolCreateFlags
  {
    eTransient          = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
    eResetCommandBuffer = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    eProtected          = VK_COMMAND_POOL_CREATE_PROTECTED_BIT
  };

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

  enum class CommandPoolResetFlagBits : VkCommandPoolResetFlags
  {
    eReleaseResources = VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT
  };

  VULKAN_HPP_INLINE std::string to_string( CommandPoolResetFlagBits value )
  {
    switch ( value )
    {
      case CommandPoolResetFlagBits::eReleaseResources: return "ReleaseResources";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class CommandBufferLevel
  {
    ePrimary   = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    eSecondary = VK_COMMAND_BUFFER_LEVEL_SECONDARY
  };

  VULKAN_HPP_INLINE std::string to_string( CommandBufferLevel value )
  {
    switch ( value )
    {
      case CommandBufferLevel::ePrimary: return "Primary";
      case CommandBufferLevel::eSecondary: return "Secondary";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class CommandBufferResetFlagBits : VkCommandBufferResetFlags
  {
    eReleaseResources = VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT
  };

  VULKAN_HPP_INLINE std::string to_string( CommandBufferResetFlagBits value )
  {
    switch ( value )
    {
      case CommandBufferResetFlagBits::eReleaseResources: return "ReleaseResources";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class CommandBufferUsageFlagBits : VkCommandBufferUsageFlags
  {
    eOneTimeSubmit      = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    eRenderPassContinue = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
    eSimultaneousUse    = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT
  };

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

  enum class QueryControlFlagBits : VkQueryControlFlags
  {
    ePrecise = VK_QUERY_CONTROL_PRECISE_BIT
  };

  VULKAN_HPP_INLINE std::string to_string( QueryControlFlagBits value )
  {
    switch ( value )
    {
      case QueryControlFlagBits::ePrecise: return "Precise";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class IndexType
  {
    eUint16   = VK_INDEX_TYPE_UINT16,
    eUint32   = VK_INDEX_TYPE_UINT32,
    eNoneKHR  = VK_INDEX_TYPE_NONE_KHR,
    eUint8EXT = VK_INDEX_TYPE_UINT8_EXT,
    eNoneNV   = VK_INDEX_TYPE_NONE_NV
  };

  VULKAN_HPP_INLINE std::string to_string( IndexType value )
  {
    switch ( value )
    {
      case IndexType::eUint16: return "Uint16";
      case IndexType::eUint32: return "Uint32";
      case IndexType::eNoneKHR: return "NoneKHR";
      case IndexType::eUint8EXT: return "Uint8EXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class StencilFaceFlagBits : VkStencilFaceFlags
  {
    eFront                 = VK_STENCIL_FACE_FRONT_BIT,
    eBack                  = VK_STENCIL_FACE_BACK_BIT,
    eFrontAndBack          = VK_STENCIL_FACE_FRONT_AND_BACK,
    eVkStencilFrontAndBack = VK_STENCIL_FRONT_AND_BACK
  };

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

  enum class SubpassContents
  {
    eInline                  = VK_SUBPASS_CONTENTS_INLINE,
    eSecondaryCommandBuffers = VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS
  };

  VULKAN_HPP_INLINE std::string to_string( SubpassContents value )
  {
    switch ( value )
    {
      case SubpassContents::eInline: return "Inline";
      case SubpassContents::eSecondaryCommandBuffers: return "SecondaryCommandBuffers";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class InstanceCreateFlagBits
  {
  };

  VULKAN_HPP_INLINE std::string to_string( InstanceCreateFlagBits )
  {
    return "(void)";
  }

  enum class DeviceCreateFlagBits
  {
  };

  VULKAN_HPP_INLINE std::string to_string( DeviceCreateFlagBits )
  {
    return "(void)";
  }

  enum class MemoryMapFlagBits : VkMemoryMapFlags
  {
  };

  VULKAN_HPP_INLINE std::string to_string( MemoryMapFlagBits )
  {
    return "(void)";
  }

  enum class SemaphoreCreateFlagBits : VkSemaphoreCreateFlags
  {
  };

  VULKAN_HPP_INLINE std::string to_string( SemaphoreCreateFlagBits )
  {
    return "(void)";
  }

  enum class QueryPoolCreateFlagBits
  {
  };

  VULKAN_HPP_INLINE std::string to_string( QueryPoolCreateFlagBits )
  {
    return "(void)";
  }

  enum class BufferViewCreateFlagBits : VkBufferViewCreateFlags
  {
  };

  VULKAN_HPP_INLINE std::string to_string( BufferViewCreateFlagBits )
  {
    return "(void)";
  }

  enum class PipelineColorBlendStateCreateFlagBits : VkPipelineColorBlendStateCreateFlags
  {
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineColorBlendStateCreateFlagBits )
  {
    return "(void)";
  }

  enum class PipelineDepthStencilStateCreateFlagBits : VkPipelineDepthStencilStateCreateFlags
  {
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineDepthStencilStateCreateFlagBits )
  {
    return "(void)";
  }

  enum class PipelineDynamicStateCreateFlagBits : VkPipelineDynamicStateCreateFlags
  {
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineDynamicStateCreateFlagBits )
  {
    return "(void)";
  }

  enum class PipelineInputAssemblyStateCreateFlagBits : VkPipelineInputAssemblyStateCreateFlags
  {
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineInputAssemblyStateCreateFlagBits )
  {
    return "(void)";
  }

  enum class PipelineLayoutCreateFlagBits : VkPipelineLayoutCreateFlags
  {
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineLayoutCreateFlagBits )
  {
    return "(void)";
  }

  enum class PipelineMultisampleStateCreateFlagBits : VkPipelineMultisampleStateCreateFlags
  {
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineMultisampleStateCreateFlagBits )
  {
    return "(void)";
  }

  enum class PipelineRasterizationStateCreateFlagBits : VkPipelineRasterizationStateCreateFlags
  {
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineRasterizationStateCreateFlagBits )
  {
    return "(void)";
  }

  enum class PipelineTessellationStateCreateFlagBits : VkPipelineTessellationStateCreateFlags
  {
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineTessellationStateCreateFlagBits )
  {
    return "(void)";
  }

  enum class PipelineVertexInputStateCreateFlagBits : VkPipelineVertexInputStateCreateFlags
  {
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineVertexInputStateCreateFlagBits )
  {
    return "(void)";
  }

  enum class PipelineViewportStateCreateFlagBits : VkPipelineViewportStateCreateFlags
  {
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineViewportStateCreateFlagBits )
  {
    return "(void)";
  }

  enum class DescriptorPoolResetFlagBits : VkDescriptorPoolResetFlags
  {
  };

  VULKAN_HPP_INLINE std::string to_string( DescriptorPoolResetFlagBits )
  {
    return "(void)";
  }

  //=== VK_VERSION_1_1 ===

  enum class SubgroupFeatureFlagBits : VkSubgroupFeatureFlags
  {
    eBasic           = VK_SUBGROUP_FEATURE_BASIC_BIT,
    eVote            = VK_SUBGROUP_FEATURE_VOTE_BIT,
    eArithmetic      = VK_SUBGROUP_FEATURE_ARITHMETIC_BIT,
    eBallot          = VK_SUBGROUP_FEATURE_BALLOT_BIT,
    eShuffle         = VK_SUBGROUP_FEATURE_SHUFFLE_BIT,
    eShuffleRelative = VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT,
    eClustered       = VK_SUBGROUP_FEATURE_CLUSTERED_BIT,
    eQuad            = VK_SUBGROUP_FEATURE_QUAD_BIT,
    ePartitionedNV   = VK_SUBGROUP_FEATURE_PARTITIONED_BIT_NV
  };

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
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class PeerMemoryFeatureFlagBits : VkPeerMemoryFeatureFlags
  {
    eCopySrc    = VK_PEER_MEMORY_FEATURE_COPY_SRC_BIT,
    eCopyDst    = VK_PEER_MEMORY_FEATURE_COPY_DST_BIT,
    eGenericSrc = VK_PEER_MEMORY_FEATURE_GENERIC_SRC_BIT,
    eGenericDst = VK_PEER_MEMORY_FEATURE_GENERIC_DST_BIT
  };
  using PeerMemoryFeatureFlagBitsKHR = PeerMemoryFeatureFlagBits;

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

  enum class MemoryAllocateFlagBits : VkMemoryAllocateFlags
  {
    eDeviceMask                 = VK_MEMORY_ALLOCATE_DEVICE_MASK_BIT,
    eDeviceAddress              = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    eDeviceAddressCaptureReplay = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT
  };
  using MemoryAllocateFlagBitsKHR = MemoryAllocateFlagBits;

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

  enum class PointClippingBehavior
  {
    eAllClipPlanes      = VK_POINT_CLIPPING_BEHAVIOR_ALL_CLIP_PLANES,
    eUserClipPlanesOnly = VK_POINT_CLIPPING_BEHAVIOR_USER_CLIP_PLANES_ONLY
  };
  using PointClippingBehaviorKHR = PointClippingBehavior;

  VULKAN_HPP_INLINE std::string to_string( PointClippingBehavior value )
  {
    switch ( value )
    {
      case PointClippingBehavior::eAllClipPlanes: return "AllClipPlanes";
      case PointClippingBehavior::eUserClipPlanesOnly: return "UserClipPlanesOnly";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class TessellationDomainOrigin
  {
    eUpperLeft = VK_TESSELLATION_DOMAIN_ORIGIN_UPPER_LEFT,
    eLowerLeft = VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT
  };
  using TessellationDomainOriginKHR = TessellationDomainOrigin;

  VULKAN_HPP_INLINE std::string to_string( TessellationDomainOrigin value )
  {
    switch ( value )
    {
      case TessellationDomainOrigin::eUpperLeft: return "UpperLeft";
      case TessellationDomainOrigin::eLowerLeft: return "LowerLeft";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class SamplerYcbcrModelConversion
  {
    eRgbIdentity   = VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY,
    eYcbcrIdentity = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_IDENTITY,
    eYcbcr709      = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709,
    eYcbcr601      = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601,
    eYcbcr2020     = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_2020
  };
  using SamplerYcbcrModelConversionKHR = SamplerYcbcrModelConversion;

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

  enum class SamplerYcbcrRange
  {
    eItuFull   = VK_SAMPLER_YCBCR_RANGE_ITU_FULL,
    eItuNarrow = VK_SAMPLER_YCBCR_RANGE_ITU_NARROW
  };
  using SamplerYcbcrRangeKHR = SamplerYcbcrRange;

  VULKAN_HPP_INLINE std::string to_string( SamplerYcbcrRange value )
  {
    switch ( value )
    {
      case SamplerYcbcrRange::eItuFull: return "ItuFull";
      case SamplerYcbcrRange::eItuNarrow: return "ItuNarrow";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class ChromaLocation
  {
    eCositedEven = VK_CHROMA_LOCATION_COSITED_EVEN,
    eMidpoint    = VK_CHROMA_LOCATION_MIDPOINT
  };
  using ChromaLocationKHR = ChromaLocation;

  VULKAN_HPP_INLINE std::string to_string( ChromaLocation value )
  {
    switch ( value )
    {
      case ChromaLocation::eCositedEven: return "CositedEven";
      case ChromaLocation::eMidpoint: return "Midpoint";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class DescriptorUpdateTemplateType
  {
    eDescriptorSet      = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET,
    ePushDescriptorsKHR = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR
  };
  using DescriptorUpdateTemplateTypeKHR = DescriptorUpdateTemplateType;

  VULKAN_HPP_INLINE std::string to_string( DescriptorUpdateTemplateType value )
  {
    switch ( value )
    {
      case DescriptorUpdateTemplateType::eDescriptorSet: return "DescriptorSet";
      case DescriptorUpdateTemplateType::ePushDescriptorsKHR: return "PushDescriptorsKHR";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class ExternalMemoryHandleTypeFlagBits : VkExternalMemoryHandleTypeFlags
  {
    eOpaqueFd        = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    eOpaqueWin32     = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
    eOpaqueWin32Kmt  = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT,
    eD3D11Texture    = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT,
    eD3D11TextureKmt = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT,
    eD3D12Heap       = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT,
    eD3D12Resource   = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
    eDmaBufEXT       = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
    eAndroidHardwareBufferANDROID = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID,
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
    eHostAllocationEXT          = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
    eHostMappedForeignMemoryEXT = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT,
#if defined( VK_USE_PLATFORM_FUCHSIA )
    eZirconVmoFUCHSIA = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ZIRCON_VMO_BIT_FUCHSIA
#endif /*VK_USE_PLATFORM_FUCHSIA*/
  };
  using ExternalMemoryHandleTypeFlagBitsKHR = ExternalMemoryHandleTypeFlagBits;

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
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class ExternalMemoryFeatureFlagBits : VkExternalMemoryFeatureFlags
  {
    eDedicatedOnly = VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT,
    eExportable    = VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT,
    eImportable    = VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT
  };
  using ExternalMemoryFeatureFlagBitsKHR = ExternalMemoryFeatureFlagBits;

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

  enum class ExternalFenceHandleTypeFlagBits : VkExternalFenceHandleTypeFlags
  {
    eOpaqueFd       = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT,
    eOpaqueWin32    = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_WIN32_BIT,
    eOpaqueWin32Kmt = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT,
    eSyncFd         = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT
  };
  using ExternalFenceHandleTypeFlagBitsKHR = ExternalFenceHandleTypeFlagBits;

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

  enum class ExternalFenceFeatureFlagBits : VkExternalFenceFeatureFlags
  {
    eExportable = VK_EXTERNAL_FENCE_FEATURE_EXPORTABLE_BIT,
    eImportable = VK_EXTERNAL_FENCE_FEATURE_IMPORTABLE_BIT
  };
  using ExternalFenceFeatureFlagBitsKHR = ExternalFenceFeatureFlagBits;

  VULKAN_HPP_INLINE std::string to_string( ExternalFenceFeatureFlagBits value )
  {
    switch ( value )
    {
      case ExternalFenceFeatureFlagBits::eExportable: return "Exportable";
      case ExternalFenceFeatureFlagBits::eImportable: return "Importable";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class FenceImportFlagBits : VkFenceImportFlags
  {
    eTemporary = VK_FENCE_IMPORT_TEMPORARY_BIT
  };
  using FenceImportFlagBitsKHR = FenceImportFlagBits;

  VULKAN_HPP_INLINE std::string to_string( FenceImportFlagBits value )
  {
    switch ( value )
    {
      case FenceImportFlagBits::eTemporary: return "Temporary";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class SemaphoreImportFlagBits : VkSemaphoreImportFlags
  {
    eTemporary = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT
  };
  using SemaphoreImportFlagBitsKHR = SemaphoreImportFlagBits;

  VULKAN_HPP_INLINE std::string to_string( SemaphoreImportFlagBits value )
  {
    switch ( value )
    {
      case SemaphoreImportFlagBits::eTemporary: return "Temporary";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class ExternalSemaphoreHandleTypeFlagBits : VkExternalSemaphoreHandleTypeFlags
  {
    eOpaqueFd       = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
    eOpaqueWin32    = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT,
    eOpaqueWin32Kmt = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT,
    eD3D12Fence     = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT,
    eSyncFd         = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
#if defined( VK_USE_PLATFORM_FUCHSIA )
    eZirconEventFUCHSIA = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_ZIRCON_EVENT_BIT_FUCHSIA,
#endif /*VK_USE_PLATFORM_FUCHSIA*/
    eD3D11Fence = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE_BIT
  };
  using ExternalSemaphoreHandleTypeFlagBitsKHR = ExternalSemaphoreHandleTypeFlagBits;

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

  enum class ExternalSemaphoreFeatureFlagBits : VkExternalSemaphoreFeatureFlags
  {
    eExportable = VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT,
    eImportable = VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT
  };
  using ExternalSemaphoreFeatureFlagBitsKHR = ExternalSemaphoreFeatureFlagBits;

  VULKAN_HPP_INLINE std::string to_string( ExternalSemaphoreFeatureFlagBits value )
  {
    switch ( value )
    {
      case ExternalSemaphoreFeatureFlagBits::eExportable: return "Exportable";
      case ExternalSemaphoreFeatureFlagBits::eImportable: return "Importable";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class CommandPoolTrimFlagBits : VkCommandPoolTrimFlags
  {
  };

  VULKAN_HPP_INLINE std::string to_string( CommandPoolTrimFlagBits )
  {
    return "(void)";
  }

  enum class DescriptorUpdateTemplateCreateFlagBits : VkDescriptorUpdateTemplateCreateFlags
  {
  };

  VULKAN_HPP_INLINE std::string to_string( DescriptorUpdateTemplateCreateFlagBits )
  {
    return "(void)";
  }

  //=== VK_VERSION_1_2 ===

  enum class DriverId
  {
    eAmdProprietary          = VK_DRIVER_ID_AMD_PROPRIETARY,
    eAmdOpenSource           = VK_DRIVER_ID_AMD_OPEN_SOURCE,
    eMesaRadv                = VK_DRIVER_ID_MESA_RADV,
    eNvidiaProprietary       = VK_DRIVER_ID_NVIDIA_PROPRIETARY,
    eIntelProprietaryWindows = VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS,
    eIntelOpenSourceMESA     = VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA,
    eImaginationProprietary  = VK_DRIVER_ID_IMAGINATION_PROPRIETARY,
    eQualcommProprietary     = VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
    eArmProprietary          = VK_DRIVER_ID_ARM_PROPRIETARY,
    eGoogleSwiftshader       = VK_DRIVER_ID_GOOGLE_SWIFTSHADER,
    eGgpProprietary          = VK_DRIVER_ID_GGP_PROPRIETARY,
    eBroadcomProprietary     = VK_DRIVER_ID_BROADCOM_PROPRIETARY,
    eMesaLlvmpipe            = VK_DRIVER_ID_MESA_LLVMPIPE,
    eMoltenvk                = VK_DRIVER_ID_MOLTENVK,
    eCoreaviProprietary      = VK_DRIVER_ID_COREAVI_PROPRIETARY,
    eJuiceProprietary        = VK_DRIVER_ID_JUICE_PROPRIETARY
  };
  using DriverIdKHR = DriverId;

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
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class ShaderFloatControlsIndependence
  {
    e32BitOnly = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY,
    eAll       = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
    eNone      = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE
  };
  using ShaderFloatControlsIndependenceKHR = ShaderFloatControlsIndependence;

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

  enum class DescriptorBindingFlagBits : VkDescriptorBindingFlags
  {
    eUpdateAfterBind          = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
    eUpdateUnusedWhilePending = VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT,
    ePartiallyBound           = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
    eVariableDescriptorCount  = VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT
  };
  using DescriptorBindingFlagBitsEXT = DescriptorBindingFlagBits;

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

  enum class ResolveModeFlagBits : VkResolveModeFlags
  {
    eNone       = VK_RESOLVE_MODE_NONE,
    eSampleZero = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT,
    eAverage    = VK_RESOLVE_MODE_AVERAGE_BIT,
    eMin        = VK_RESOLVE_MODE_MIN_BIT,
    eMax        = VK_RESOLVE_MODE_MAX_BIT
  };
  using ResolveModeFlagBitsKHR = ResolveModeFlagBits;

  VULKAN_HPP_INLINE std::string to_string( ResolveModeFlagBits value )
  {
    switch ( value )
    {
      case ResolveModeFlagBits::eNone: return "None";
      case ResolveModeFlagBits::eSampleZero: return "SampleZero";
      case ResolveModeFlagBits::eAverage: return "Average";
      case ResolveModeFlagBits::eMin: return "Min";
      case ResolveModeFlagBits::eMax: return "Max";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class SamplerReductionMode
  {
    eWeightedAverage = VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE,
    eMin             = VK_SAMPLER_REDUCTION_MODE_MIN,
    eMax             = VK_SAMPLER_REDUCTION_MODE_MAX
  };
  using SamplerReductionModeEXT = SamplerReductionMode;

  VULKAN_HPP_INLINE std::string to_string( SamplerReductionMode value )
  {
    switch ( value )
    {
      case SamplerReductionMode::eWeightedAverage: return "WeightedAverage";
      case SamplerReductionMode::eMin: return "Min";
      case SamplerReductionMode::eMax: return "Max";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class SemaphoreType
  {
    eBinary   = VK_SEMAPHORE_TYPE_BINARY,
    eTimeline = VK_SEMAPHORE_TYPE_TIMELINE
  };
  using SemaphoreTypeKHR = SemaphoreType;

  VULKAN_HPP_INLINE std::string to_string( SemaphoreType value )
  {
    switch ( value )
    {
      case SemaphoreType::eBinary: return "Binary";
      case SemaphoreType::eTimeline: return "Timeline";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class SemaphoreWaitFlagBits : VkSemaphoreWaitFlags
  {
    eAny = VK_SEMAPHORE_WAIT_ANY_BIT
  };
  using SemaphoreWaitFlagBitsKHR = SemaphoreWaitFlagBits;

  VULKAN_HPP_INLINE std::string to_string( SemaphoreWaitFlagBits value )
  {
    switch ( value )
    {
      case SemaphoreWaitFlagBits::eAny: return "Any";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_KHR_surface ===

  enum class SurfaceTransformFlagBitsKHR : VkSurfaceTransformFlagsKHR
  {
    eIdentity                  = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
    eRotate90                  = VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR,
    eRotate180                 = VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR,
    eRotate270                 = VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR,
    eHorizontalMirror          = VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_BIT_KHR,
    eHorizontalMirrorRotate90  = VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_90_BIT_KHR,
    eHorizontalMirrorRotate180 = VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_180_BIT_KHR,
    eHorizontalMirrorRotate270 = VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_270_BIT_KHR,
    eInherit                   = VK_SURFACE_TRANSFORM_INHERIT_BIT_KHR
  };

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

  enum class PresentModeKHR
  {
    eImmediate               = VK_PRESENT_MODE_IMMEDIATE_KHR,
    eMailbox                 = VK_PRESENT_MODE_MAILBOX_KHR,
    eFifo                    = VK_PRESENT_MODE_FIFO_KHR,
    eFifoRelaxed             = VK_PRESENT_MODE_FIFO_RELAXED_KHR,
    eSharedDemandRefresh     = VK_PRESENT_MODE_SHARED_DEMAND_REFRESH_KHR,
    eSharedContinuousRefresh = VK_PRESENT_MODE_SHARED_CONTINUOUS_REFRESH_KHR
  };

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

  enum class ColorSpaceKHR
  {
    eSrgbNonlinear             = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
    eDisplayP3NonlinearEXT     = VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT,
    eExtendedSrgbLinearEXT     = VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT,
    eDisplayP3LinearEXT        = VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT,
    eDciP3NonlinearEXT         = VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT,
    eBt709LinearEXT            = VK_COLOR_SPACE_BT709_LINEAR_EXT,
    eBt709NonlinearEXT         = VK_COLOR_SPACE_BT709_NONLINEAR_EXT,
    eBt2020LinearEXT           = VK_COLOR_SPACE_BT2020_LINEAR_EXT,
    eHdr10St2084EXT            = VK_COLOR_SPACE_HDR10_ST2084_EXT,
    eDolbyvisionEXT            = VK_COLOR_SPACE_DOLBYVISION_EXT,
    eHdr10HlgEXT               = VK_COLOR_SPACE_HDR10_HLG_EXT,
    eAdobergbLinearEXT         = VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT,
    eAdobergbNonlinearEXT      = VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT,
    ePassThroughEXT            = VK_COLOR_SPACE_PASS_THROUGH_EXT,
    eExtendedSrgbNonlinearEXT  = VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT,
    eDisplayNativeAMD          = VK_COLOR_SPACE_DISPLAY_NATIVE_AMD,
    eVkColorspaceSrgbNonlinear = VK_COLORSPACE_SRGB_NONLINEAR_KHR,
    eDciP3LinearEXT            = VK_COLOR_SPACE_DCI_P3_LINEAR_EXT
  };

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

  enum class CompositeAlphaFlagBitsKHR : VkCompositeAlphaFlagsKHR
  {
    eOpaque         = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
    ePreMultiplied  = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
    ePostMultiplied = VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
    eInherit        = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR
  };

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

  enum class SwapchainCreateFlagBitsKHR : VkSwapchainCreateFlagsKHR
  {
    eSplitInstanceBindRegions = VK_SWAPCHAIN_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT_KHR,
    eProtected                = VK_SWAPCHAIN_CREATE_PROTECTED_BIT_KHR,
    eMutableFormat            = VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR
  };

  VULKAN_HPP_INLINE std::string to_string( SwapchainCreateFlagBitsKHR value )
  {
    switch ( value )
    {
      case SwapchainCreateFlagBitsKHR::eSplitInstanceBindRegions: return "SplitInstanceBindRegions";
      case SwapchainCreateFlagBitsKHR::eProtected: return "Protected";
      case SwapchainCreateFlagBitsKHR::eMutableFormat: return "MutableFormat";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class DeviceGroupPresentModeFlagBitsKHR : VkDeviceGroupPresentModeFlagsKHR
  {
    eLocal            = VK_DEVICE_GROUP_PRESENT_MODE_LOCAL_BIT_KHR,
    eRemote           = VK_DEVICE_GROUP_PRESENT_MODE_REMOTE_BIT_KHR,
    eSum              = VK_DEVICE_GROUP_PRESENT_MODE_SUM_BIT_KHR,
    eLocalMultiDevice = VK_DEVICE_GROUP_PRESENT_MODE_LOCAL_MULTI_DEVICE_BIT_KHR
  };

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

  enum class DisplayPlaneAlphaFlagBitsKHR : VkDisplayPlaneAlphaFlagsKHR
  {
    eOpaque                = VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR,
    eGlobal                = VK_DISPLAY_PLANE_ALPHA_GLOBAL_BIT_KHR,
    ePerPixel              = VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_BIT_KHR,
    ePerPixelPremultiplied = VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_PREMULTIPLIED_BIT_KHR
  };

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

  enum class DisplayModeCreateFlagBitsKHR : VkDisplayModeCreateFlagsKHR
  {
  };

  VULKAN_HPP_INLINE std::string to_string( DisplayModeCreateFlagBitsKHR )
  {
    return "(void)";
  }

  enum class DisplaySurfaceCreateFlagBitsKHR : VkDisplaySurfaceCreateFlagsKHR
  {
  };

  VULKAN_HPP_INLINE std::string to_string( DisplaySurfaceCreateFlagBitsKHR )
  {
    return "(void)";
  }

#if defined( VK_USE_PLATFORM_XLIB_KHR )
  //=== VK_KHR_xlib_surface ===

  enum class XlibSurfaceCreateFlagBitsKHR : VkXlibSurfaceCreateFlagsKHR
  {
  };

  VULKAN_HPP_INLINE std::string to_string( XlibSurfaceCreateFlagBitsKHR )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_XLIB_KHR*/

#if defined( VK_USE_PLATFORM_XCB_KHR )
  //=== VK_KHR_xcb_surface ===

  enum class XcbSurfaceCreateFlagBitsKHR : VkXcbSurfaceCreateFlagsKHR
  {
  };

  VULKAN_HPP_INLINE std::string to_string( XcbSurfaceCreateFlagBitsKHR )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_XCB_KHR*/

#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
  //=== VK_KHR_wayland_surface ===

  enum class WaylandSurfaceCreateFlagBitsKHR : VkWaylandSurfaceCreateFlagsKHR
  {
  };

  VULKAN_HPP_INLINE std::string to_string( WaylandSurfaceCreateFlagBitsKHR )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  //=== VK_KHR_android_surface ===

  enum class AndroidSurfaceCreateFlagBitsKHR : VkAndroidSurfaceCreateFlagsKHR
  {
  };

  VULKAN_HPP_INLINE std::string to_string( AndroidSurfaceCreateFlagBitsKHR )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_win32_surface ===

  enum class Win32SurfaceCreateFlagBitsKHR : VkWin32SurfaceCreateFlagsKHR
  {
  };

  VULKAN_HPP_INLINE std::string to_string( Win32SurfaceCreateFlagBitsKHR )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_EXT_debug_report ===

  enum class DebugReportFlagBitsEXT : VkDebugReportFlagsEXT
  {
    eInformation        = VK_DEBUG_REPORT_INFORMATION_BIT_EXT,
    eWarning            = VK_DEBUG_REPORT_WARNING_BIT_EXT,
    ePerformanceWarning = VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT,
    eError              = VK_DEBUG_REPORT_ERROR_BIT_EXT,
    eDebug              = VK_DEBUG_REPORT_DEBUG_BIT_EXT
  };

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

  enum class DebugReportObjectTypeEXT
  {
    eUnknown                     = VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT,
    eInstance                    = VK_DEBUG_REPORT_OBJECT_TYPE_INSTANCE_EXT,
    ePhysicalDevice              = VK_DEBUG_REPORT_OBJECT_TYPE_PHYSICAL_DEVICE_EXT,
    eDevice                      = VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_EXT,
    eQueue                       = VK_DEBUG_REPORT_OBJECT_TYPE_QUEUE_EXT,
    eSemaphore                   = VK_DEBUG_REPORT_OBJECT_TYPE_SEMAPHORE_EXT,
    eCommandBuffer               = VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
    eFence                       = VK_DEBUG_REPORT_OBJECT_TYPE_FENCE_EXT,
    eDeviceMemory                = VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_MEMORY_EXT,
    eBuffer                      = VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT,
    eImage                       = VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
    eEvent                       = VK_DEBUG_REPORT_OBJECT_TYPE_EVENT_EXT,
    eQueryPool                   = VK_DEBUG_REPORT_OBJECT_TYPE_QUERY_POOL_EXT,
    eBufferView                  = VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_VIEW_EXT,
    eImageView                   = VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_VIEW_EXT,
    eShaderModule                = VK_DEBUG_REPORT_OBJECT_TYPE_SHADER_MODULE_EXT,
    ePipelineCache               = VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_CACHE_EXT,
    ePipelineLayout              = VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_LAYOUT_EXT,
    eRenderPass                  = VK_DEBUG_REPORT_OBJECT_TYPE_RENDER_PASS_EXT,
    ePipeline                    = VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_EXT,
    eDescriptorSetLayout         = VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT_EXT,
    eSampler                     = VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_EXT,
    eDescriptorPool              = VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_POOL_EXT,
    eDescriptorSet               = VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_EXT,
    eFramebuffer                 = VK_DEBUG_REPORT_OBJECT_TYPE_FRAMEBUFFER_EXT,
    eCommandPool                 = VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_POOL_EXT,
    eSurfaceKHR                  = VK_DEBUG_REPORT_OBJECT_TYPE_SURFACE_KHR_EXT,
    eSwapchainKHR                = VK_DEBUG_REPORT_OBJECT_TYPE_SWAPCHAIN_KHR_EXT,
    eDebugReportCallbackEXT      = VK_DEBUG_REPORT_OBJECT_TYPE_DEBUG_REPORT_CALLBACK_EXT_EXT,
    eDisplayKHR                  = VK_DEBUG_REPORT_OBJECT_TYPE_DISPLAY_KHR_EXT,
    eDisplayModeKHR              = VK_DEBUG_REPORT_OBJECT_TYPE_DISPLAY_MODE_KHR_EXT,
    eValidationCacheEXT          = VK_DEBUG_REPORT_OBJECT_TYPE_VALIDATION_CACHE_EXT_EXT,
    eSamplerYcbcrConversion      = VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION_EXT,
    eDescriptorUpdateTemplate    = VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_EXT,
    eCuModuleNVX                 = VK_DEBUG_REPORT_OBJECT_TYPE_CU_MODULE_NVX_EXT,
    eCuFunctionNVX               = VK_DEBUG_REPORT_OBJECT_TYPE_CU_FUNCTION_NVX_EXT,
    eAccelerationStructureKHR    = VK_DEBUG_REPORT_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR_EXT,
    eAccelerationStructureNV     = VK_DEBUG_REPORT_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV_EXT,
    eDebugReport                 = VK_DEBUG_REPORT_OBJECT_TYPE_DEBUG_REPORT_EXT,
    eDescriptorUpdateTemplateKHR = VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_KHR_EXT,
    eSamplerYcbcrConversionKHR   = VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION_KHR_EXT,
    eValidationCache             = VK_DEBUG_REPORT_OBJECT_TYPE_VALIDATION_CACHE_EXT
  };

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
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_AMD_rasterization_order ===

  enum class RasterizationOrderAMD
  {
    eStrict  = VK_RASTERIZATION_ORDER_STRICT_AMD,
    eRelaxed = VK_RASTERIZATION_ORDER_RELAXED_AMD
  };

  VULKAN_HPP_INLINE std::string to_string( RasterizationOrderAMD value )
  {
    switch ( value )
    {
      case RasterizationOrderAMD::eStrict: return "Strict";
      case RasterizationOrderAMD::eRelaxed: return "Relaxed";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_KHR_video_queue ===

  enum class VideoCodecOperationFlagBitsKHR : VkVideoCodecOperationFlagsKHR
  {
    eInvalid = VK_VIDEO_CODEC_OPERATION_INVALID_BIT_KHR,
#  if defined( VK_ENABLE_BETA_EXTENSIONS )
    eEncodeH264EXT = VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_EXT,
    eDecodeH264EXT = VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_EXT,
    eDecodeH265EXT = VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_EXT
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/
  };

  VULKAN_HPP_INLINE std::string to_string( VideoCodecOperationFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoCodecOperationFlagBitsKHR::eInvalid: return "Invalid";
#  if defined( VK_ENABLE_BETA_EXTENSIONS )
      case VideoCodecOperationFlagBitsKHR::eEncodeH264EXT: return "EncodeH264EXT";
      case VideoCodecOperationFlagBitsKHR::eDecodeH264EXT: return "DecodeH264EXT";
      case VideoCodecOperationFlagBitsKHR::eDecodeH265EXT: return "DecodeH265EXT";
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class VideoChromaSubsamplingFlagBitsKHR : VkVideoChromaSubsamplingFlagsKHR
  {
    eInvalid    = VK_VIDEO_CHROMA_SUBSAMPLING_INVALID_BIT_KHR,
    eMonochrome = VK_VIDEO_CHROMA_SUBSAMPLING_MONOCHROME_BIT_KHR,
    e420        = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR,
    e422        = VK_VIDEO_CHROMA_SUBSAMPLING_422_BIT_KHR,
    e444        = VK_VIDEO_CHROMA_SUBSAMPLING_444_BIT_KHR
  };

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

  enum class VideoComponentBitDepthFlagBitsKHR : VkVideoComponentBitDepthFlagsKHR
  {
    eInvalid = VK_VIDEO_COMPONENT_BIT_DEPTH_INVALID_KHR,
    e8       = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
    e10      = VK_VIDEO_COMPONENT_BIT_DEPTH_10_BIT_KHR,
    e12      = VK_VIDEO_COMPONENT_BIT_DEPTH_12_BIT_KHR
  };

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

  enum class VideoCapabilitiesFlagBitsKHR : VkVideoCapabilitiesFlagsKHR
  {
    eProtectedContent        = VK_VIDEO_CAPABILITIES_PROTECTED_CONTENT_BIT_KHR,
    eSeparateReferenceImages = VK_VIDEO_CAPABILITIES_SEPARATE_REFERENCE_IMAGES_BIT_KHR
  };

  VULKAN_HPP_INLINE std::string to_string( VideoCapabilitiesFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoCapabilitiesFlagBitsKHR::eProtectedContent: return "ProtectedContent";
      case VideoCapabilitiesFlagBitsKHR::eSeparateReferenceImages: return "SeparateReferenceImages";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class VideoSessionCreateFlagBitsKHR : VkVideoSessionCreateFlagsKHR
  {
    eDefault          = VK_VIDEO_SESSION_CREATE_DEFAULT_KHR,
    eProtectedContent = VK_VIDEO_SESSION_CREATE_PROTECTED_CONTENT_BIT_KHR
  };

  VULKAN_HPP_INLINE std::string to_string( VideoSessionCreateFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoSessionCreateFlagBitsKHR::eDefault: return "Default";
      case VideoSessionCreateFlagBitsKHR::eProtectedContent: return "ProtectedContent";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class VideoCodingControlFlagBitsKHR : VkVideoCodingControlFlagsKHR
  {
    eDefault = VK_VIDEO_CODING_CONTROL_DEFAULT_KHR,
    eReset   = VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR
  };

  VULKAN_HPP_INLINE std::string to_string( VideoCodingControlFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoCodingControlFlagBitsKHR::eDefault: return "Default";
      case VideoCodingControlFlagBitsKHR::eReset: return "Reset";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class VideoCodingQualityPresetFlagBitsKHR : VkVideoCodingQualityPresetFlagsKHR
  {
    eDefault = VK_VIDEO_CODING_QUALITY_PRESET_DEFAULT_BIT_KHR,
    eNormal  = VK_VIDEO_CODING_QUALITY_PRESET_NORMAL_BIT_KHR,
    ePower   = VK_VIDEO_CODING_QUALITY_PRESET_POWER_BIT_KHR,
    eQuality = VK_VIDEO_CODING_QUALITY_PRESET_QUALITY_BIT_KHR
  };

  VULKAN_HPP_INLINE std::string to_string( VideoCodingQualityPresetFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoCodingQualityPresetFlagBitsKHR::eDefault: return "Default";
      case VideoCodingQualityPresetFlagBitsKHR::eNormal: return "Normal";
      case VideoCodingQualityPresetFlagBitsKHR::ePower: return "Power";
      case VideoCodingQualityPresetFlagBitsKHR::eQuality: return "Quality";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class QueryResultStatusKHR
  {
    eError    = VK_QUERY_RESULT_STATUS_ERROR_KHR,
    eNotReady = VK_QUERY_RESULT_STATUS_NOT_READY_KHR,
    eComplete = VK_QUERY_RESULT_STATUS_COMPLETE_KHR
  };

  VULKAN_HPP_INLINE std::string to_string( QueryResultStatusKHR value )
  {
    switch ( value )
    {
      case QueryResultStatusKHR::eError: return "Error";
      case QueryResultStatusKHR::eNotReady: return "NotReady";
      case QueryResultStatusKHR::eComplete: return "Complete";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class VideoBeginCodingFlagBitsKHR : VkVideoBeginCodingFlagsKHR
  {
  };

  VULKAN_HPP_INLINE std::string to_string( VideoBeginCodingFlagBitsKHR )
  {
    return "(void)";
  }

  enum class VideoEndCodingFlagBitsKHR : VkVideoEndCodingFlagsKHR
  {
  };

  VULKAN_HPP_INLINE std::string to_string( VideoEndCodingFlagBitsKHR )
  {
    return "(void)";
  }
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_KHR_video_decode_queue ===

  enum class VideoDecodeFlagBitsKHR : VkVideoDecodeFlagsKHR
  {
    eDefault   = VK_VIDEO_DECODE_DEFAULT_KHR,
    eReserved0 = VK_VIDEO_DECODE_RESERVED_0_BIT_KHR
  };

  VULKAN_HPP_INLINE std::string to_string( VideoDecodeFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoDecodeFlagBitsKHR::eDefault: return "Default";
      case VideoDecodeFlagBitsKHR::eReserved0: return "Reserved0";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_EXT_transform_feedback ===

  enum class PipelineRasterizationStateStreamCreateFlagBitsEXT : VkPipelineRasterizationStateStreamCreateFlagsEXT
  {
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineRasterizationStateStreamCreateFlagBitsEXT )
  {
    return "(void)";
  }

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_EXT_video_encode_h264 ===

  enum class VideoEncodeH264CapabilitiesFlagBitsEXT : VkVideoEncodeH264CapabilitiesFlagsEXT
  {
    eVkVideoEncodeH264CapabilityCabac = VK_VIDEO_ENCODE_H264_CAPABILITY_CABAC_BIT_EXT,
    eVkVideoEncodeH264CapabilityCavlc = VK_VIDEO_ENCODE_H264_CAPABILITY_CAVLC_BIT_EXT,
    eVkVideoEncodeH264CapabilityWeightedBiPredImplicit =
      VK_VIDEO_ENCODE_H264_CAPABILITY_WEIGHTED_BI_PRED_IMPLICIT_BIT_EXT,
    eVkVideoEncodeH264CapabilityTransform8X8         = VK_VIDEO_ENCODE_H264_CAPABILITY_TRANSFORM_8X8_BIT_EXT,
    eVkVideoEncodeH264CapabilityChromaQpOffset       = VK_VIDEO_ENCODE_H264_CAPABILITY_CHROMA_QP_OFFSET_BIT_EXT,
    eVkVideoEncodeH264CapabilitySecondChromaQpOffset = VK_VIDEO_ENCODE_H264_CAPABILITY_SECOND_CHROMA_QP_OFFSET_BIT_EXT,
    eVkVideoEncodeH264CapabilityDeblockingFilterDisabled =
      VK_VIDEO_ENCODE_H264_CAPABILITY_DEBLOCKING_FILTER_DISABLED_BIT_EXT,
    eVkVideoEncodeH264CapabilityDeblockingFilterEnabled =
      VK_VIDEO_ENCODE_H264_CAPABILITY_DEBLOCKING_FILTER_ENABLED_BIT_EXT,
    eVkVideoEncodeH264CapabilityDeblockingFilterPartial =
      VK_VIDEO_ENCODE_H264_CAPABILITY_DEBLOCKING_FILTER_PARTIAL_BIT_EXT,
    eVkVideoEncodeH264CapabilityMultipleSlicePerFrame =
      VK_VIDEO_ENCODE_H264_CAPABILITY_MULTIPLE_SLICE_PER_FRAME_BIT_EXT,
    eVkVideoEncodeH264CapabilityEvenlyDistributedSliceSize =
      VK_VIDEO_ENCODE_H264_CAPABILITY_EVENLY_DISTRIBUTED_SLICE_SIZE_BIT_EXT
  };

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeH264CapabilitiesFlagBitsEXT value )
  {
    switch ( value )
    {
      case VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityCabac:
        return "VkVideoEncodeH264CapabilityCabac";
      case VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityCavlc:
        return "VkVideoEncodeH264CapabilityCavlc";
      case VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityWeightedBiPredImplicit:
        return "VkVideoEncodeH264CapabilityWeightedBiPredImplicit";
      case VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityTransform8X8:
        return "VkVideoEncodeH264CapabilityTransform8X8";
      case VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityChromaQpOffset:
        return "VkVideoEncodeH264CapabilityChromaQpOffset";
      case VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilitySecondChromaQpOffset:
        return "VkVideoEncodeH264CapabilitySecondChromaQpOffset";
      case VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityDeblockingFilterDisabled:
        return "VkVideoEncodeH264CapabilityDeblockingFilterDisabled";
      case VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityDeblockingFilterEnabled:
        return "VkVideoEncodeH264CapabilityDeblockingFilterEnabled";
      case VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityDeblockingFilterPartial:
        return "VkVideoEncodeH264CapabilityDeblockingFilterPartial";
      case VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityMultipleSlicePerFrame:
        return "VkVideoEncodeH264CapabilityMultipleSlicePerFrame";
      case VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityEvenlyDistributedSliceSize:
        return "VkVideoEncodeH264CapabilityEvenlyDistributedSliceSize";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class VideoEncodeH264InputModeFlagBitsEXT : VkVideoEncodeH264InputModeFlagsEXT
  {
    eFrame  = VK_VIDEO_ENCODE_H264_INPUT_MODE_FRAME_BIT_EXT,
    eSlice  = VK_VIDEO_ENCODE_H264_INPUT_MODE_SLICE_BIT_EXT,
    eNonVcl = VK_VIDEO_ENCODE_H264_INPUT_MODE_NON_VCL_BIT_EXT
  };

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeH264InputModeFlagBitsEXT value )
  {
    switch ( value )
    {
      case VideoEncodeH264InputModeFlagBitsEXT::eFrame: return "Frame";
      case VideoEncodeH264InputModeFlagBitsEXT::eSlice: return "Slice";
      case VideoEncodeH264InputModeFlagBitsEXT::eNonVcl: return "NonVcl";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class VideoEncodeH264OutputModeFlagBitsEXT : VkVideoEncodeH264OutputModeFlagsEXT
  {
    eFrame  = VK_VIDEO_ENCODE_H264_OUTPUT_MODE_FRAME_BIT_EXT,
    eSlice  = VK_VIDEO_ENCODE_H264_OUTPUT_MODE_SLICE_BIT_EXT,
    eNonVcl = VK_VIDEO_ENCODE_H264_OUTPUT_MODE_NON_VCL_BIT_EXT
  };

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeH264OutputModeFlagBitsEXT value )
  {
    switch ( value )
    {
      case VideoEncodeH264OutputModeFlagBitsEXT::eFrame: return "Frame";
      case VideoEncodeH264OutputModeFlagBitsEXT::eSlice: return "Slice";
      case VideoEncodeH264OutputModeFlagBitsEXT::eNonVcl: return "NonVcl";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class VideoEncodeH264CreateFlagBitsEXT : VkVideoEncodeH264CreateFlagsEXT
  {
    eDefault   = VK_VIDEO_ENCODE_H264_CREATE_DEFAULT_EXT,
    eReserved0 = VK_VIDEO_ENCODE_H264_CREATE_RESERVED_0_BIT_EXT
  };

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeH264CreateFlagBitsEXT value )
  {
    switch ( value )
    {
      case VideoEncodeH264CreateFlagBitsEXT::eDefault: return "Default";
      case VideoEncodeH264CreateFlagBitsEXT::eReserved0: return "Reserved0";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_EXT_video_decode_h264 ===

  enum class VideoDecodeH264FieldLayoutFlagBitsEXT : VkVideoDecodeH264FieldLayoutFlagsEXT
  {
    eVkVideoDecodeH264ProgressivePicturesOnly = VK_VIDEO_DECODE_H264_PROGRESSIVE_PICTURES_ONLY_EXT,
    eLineInterlacedPlane                      = VK_VIDEO_DECODE_H264_FIELD_LAYOUT_LINE_INTERLACED_PLANE_BIT_EXT,
    eSeparateInterlacedPlane                  = VK_VIDEO_DECODE_H264_FIELD_LAYOUT_SEPARATE_INTERLACED_PLANE_BIT_EXT
  };

  VULKAN_HPP_INLINE std::string to_string( VideoDecodeH264FieldLayoutFlagBitsEXT value )
  {
    switch ( value )
    {
      case VideoDecodeH264FieldLayoutFlagBitsEXT::eVkVideoDecodeH264ProgressivePicturesOnly:
        return "VkVideoDecodeH264ProgressivePicturesOnly";
      case VideoDecodeH264FieldLayoutFlagBitsEXT::eLineInterlacedPlane: return "LineInterlacedPlane";
      case VideoDecodeH264FieldLayoutFlagBitsEXT::eSeparateInterlacedPlane: return "SeparateInterlacedPlane";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class VideoDecodeH264CreateFlagBitsEXT : VkVideoDecodeH264CreateFlagsEXT
  {
  };

  VULKAN_HPP_INLINE std::string to_string( VideoDecodeH264CreateFlagBitsEXT )
  {
    return "(void)";
  }
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_AMD_shader_info ===

  enum class ShaderInfoTypeAMD
  {
    eStatistics  = VK_SHADER_INFO_TYPE_STATISTICS_AMD,
    eBinary      = VK_SHADER_INFO_TYPE_BINARY_AMD,
    eDisassembly = VK_SHADER_INFO_TYPE_DISASSEMBLY_AMD
  };

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

  enum class StreamDescriptorSurfaceCreateFlagBitsGGP : VkStreamDescriptorSurfaceCreateFlagsGGP
  {
  };

  VULKAN_HPP_INLINE std::string to_string( StreamDescriptorSurfaceCreateFlagBitsGGP )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_GGP*/

  //=== VK_NV_external_memory_capabilities ===

  enum class ExternalMemoryHandleTypeFlagBitsNV : VkExternalMemoryHandleTypeFlagsNV
  {
    eOpaqueWin32    = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_NV,
    eOpaqueWin32Kmt = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_NV,
    eD3D11Image     = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_IMAGE_BIT_NV,
    eD3D11ImageKmt  = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_IMAGE_KMT_BIT_NV
  };

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

  enum class ExternalMemoryFeatureFlagBitsNV : VkExternalMemoryFeatureFlagsNV
  {
    eDedicatedOnly = VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT_NV,
    eExportable    = VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT_NV,
    eImportable    = VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT_NV
  };

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

  enum class ValidationCheckEXT
  {
    eAll     = VK_VALIDATION_CHECK_ALL_EXT,
    eShaders = VK_VALIDATION_CHECK_SHADERS_EXT
  };

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

  enum class ViSurfaceCreateFlagBitsNN : VkViSurfaceCreateFlagsNN
  {
  };

  VULKAN_HPP_INLINE std::string to_string( ViSurfaceCreateFlagBitsNN )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_VI_NN*/

  //=== VK_EXT_conditional_rendering ===

  enum class ConditionalRenderingFlagBitsEXT : VkConditionalRenderingFlagsEXT
  {
    eInverted = VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT
  };

  VULKAN_HPP_INLINE std::string to_string( ConditionalRenderingFlagBitsEXT value )
  {
    switch ( value )
    {
      case ConditionalRenderingFlagBitsEXT::eInverted: return "Inverted";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_EXT_display_surface_counter ===

  enum class SurfaceCounterFlagBitsEXT : VkSurfaceCounterFlagsEXT
  {
    eVblank = VK_SURFACE_COUNTER_VBLANK_BIT_EXT
  };

  VULKAN_HPP_INLINE std::string to_string( SurfaceCounterFlagBitsEXT value )
  {
    switch ( value )
    {
      case SurfaceCounterFlagBitsEXT::eVblank: return "Vblank";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_EXT_display_control ===

  enum class DisplayPowerStateEXT
  {
    eOff     = VK_DISPLAY_POWER_STATE_OFF_EXT,
    eSuspend = VK_DISPLAY_POWER_STATE_SUSPEND_EXT,
    eOn      = VK_DISPLAY_POWER_STATE_ON_EXT
  };

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

  enum class DeviceEventTypeEXT
  {
    eDisplayHotplug = VK_DEVICE_EVENT_TYPE_DISPLAY_HOTPLUG_EXT
  };

  VULKAN_HPP_INLINE std::string to_string( DeviceEventTypeEXT value )
  {
    switch ( value )
    {
      case DeviceEventTypeEXT::eDisplayHotplug: return "DisplayHotplug";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class DisplayEventTypeEXT
  {
    eFirstPixelOut = VK_DISPLAY_EVENT_TYPE_FIRST_PIXEL_OUT_EXT
  };

  VULKAN_HPP_INLINE std::string to_string( DisplayEventTypeEXT value )
  {
    switch ( value )
    {
      case DisplayEventTypeEXT::eFirstPixelOut: return "FirstPixelOut";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_NV_viewport_swizzle ===

  enum class ViewportCoordinateSwizzleNV
  {
    ePositiveX = VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_X_NV,
    eNegativeX = VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_X_NV,
    ePositiveY = VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Y_NV,
    eNegativeY = VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_Y_NV,
    ePositiveZ = VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Z_NV,
    eNegativeZ = VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_Z_NV,
    ePositiveW = VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV,
    eNegativeW = VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_W_NV
  };

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

  enum class PipelineViewportSwizzleStateCreateFlagBitsNV : VkPipelineViewportSwizzleStateCreateFlagsNV
  {
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineViewportSwizzleStateCreateFlagBitsNV )
  {
    return "(void)";
  }

  //=== VK_EXT_discard_rectangles ===

  enum class DiscardRectangleModeEXT
  {
    eInclusive = VK_DISCARD_RECTANGLE_MODE_INCLUSIVE_EXT,
    eExclusive = VK_DISCARD_RECTANGLE_MODE_EXCLUSIVE_EXT
  };

  VULKAN_HPP_INLINE std::string to_string( DiscardRectangleModeEXT value )
  {
    switch ( value )
    {
      case DiscardRectangleModeEXT::eInclusive: return "Inclusive";
      case DiscardRectangleModeEXT::eExclusive: return "Exclusive";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class PipelineDiscardRectangleStateCreateFlagBitsEXT : VkPipelineDiscardRectangleStateCreateFlagsEXT
  {
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineDiscardRectangleStateCreateFlagBitsEXT )
  {
    return "(void)";
  }

  //=== VK_EXT_conservative_rasterization ===

  enum class ConservativeRasterizationModeEXT
  {
    eDisabled      = VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT,
    eOverestimate  = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT,
    eUnderestimate = VK_CONSERVATIVE_RASTERIZATION_MODE_UNDERESTIMATE_EXT
  };

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

  enum class
    PipelineRasterizationConservativeStateCreateFlagBitsEXT : VkPipelineRasterizationConservativeStateCreateFlagsEXT
  {
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineRasterizationConservativeStateCreateFlagBitsEXT )
  {
    return "(void)";
  }

  //=== VK_EXT_depth_clip_enable ===

  enum class PipelineRasterizationDepthClipStateCreateFlagBitsEXT : VkPipelineRasterizationDepthClipStateCreateFlagsEXT
  {
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineRasterizationDepthClipStateCreateFlagBitsEXT )
  {
    return "(void)";
  }

  //=== VK_KHR_performance_query ===

  enum class PerformanceCounterDescriptionFlagBitsKHR : VkPerformanceCounterDescriptionFlagsKHR
  {
    ePerformanceImpacting = VK_PERFORMANCE_COUNTER_DESCRIPTION_PERFORMANCE_IMPACTING_BIT_KHR,
    eConcurrentlyImpacted = VK_PERFORMANCE_COUNTER_DESCRIPTION_CONCURRENTLY_IMPACTED_BIT_KHR
  };

  VULKAN_HPP_INLINE std::string to_string( PerformanceCounterDescriptionFlagBitsKHR value )
  {
    switch ( value )
    {
      case PerformanceCounterDescriptionFlagBitsKHR::ePerformanceImpacting: return "PerformanceImpacting";
      case PerformanceCounterDescriptionFlagBitsKHR::eConcurrentlyImpacted: return "ConcurrentlyImpacted";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class PerformanceCounterScopeKHR
  {
    eCommandBuffer             = VK_PERFORMANCE_COUNTER_SCOPE_COMMAND_BUFFER_KHR,
    eRenderPass                = VK_PERFORMANCE_COUNTER_SCOPE_RENDER_PASS_KHR,
    eCommand                   = VK_PERFORMANCE_COUNTER_SCOPE_COMMAND_KHR,
    eVkQueryScopeCommandBuffer = VK_QUERY_SCOPE_COMMAND_BUFFER_KHR,
    eVkQueryScopeCommand       = VK_QUERY_SCOPE_COMMAND_KHR,
    eVkQueryScopeRenderPass    = VK_QUERY_SCOPE_RENDER_PASS_KHR
  };

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

  enum class PerformanceCounterStorageKHR
  {
    eInt32   = VK_PERFORMANCE_COUNTER_STORAGE_INT32_KHR,
    eInt64   = VK_PERFORMANCE_COUNTER_STORAGE_INT64_KHR,
    eUint32  = VK_PERFORMANCE_COUNTER_STORAGE_UINT32_KHR,
    eUint64  = VK_PERFORMANCE_COUNTER_STORAGE_UINT64_KHR,
    eFloat32 = VK_PERFORMANCE_COUNTER_STORAGE_FLOAT32_KHR,
    eFloat64 = VK_PERFORMANCE_COUNTER_STORAGE_FLOAT64_KHR
  };

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

  enum class PerformanceCounterUnitKHR
  {
    eGeneric        = VK_PERFORMANCE_COUNTER_UNIT_GENERIC_KHR,
    ePercentage     = VK_PERFORMANCE_COUNTER_UNIT_PERCENTAGE_KHR,
    eNanoseconds    = VK_PERFORMANCE_COUNTER_UNIT_NANOSECONDS_KHR,
    eBytes          = VK_PERFORMANCE_COUNTER_UNIT_BYTES_KHR,
    eBytesPerSecond = VK_PERFORMANCE_COUNTER_UNIT_BYTES_PER_SECOND_KHR,
    eKelvin         = VK_PERFORMANCE_COUNTER_UNIT_KELVIN_KHR,
    eWatts          = VK_PERFORMANCE_COUNTER_UNIT_WATTS_KHR,
    eVolts          = VK_PERFORMANCE_COUNTER_UNIT_VOLTS_KHR,
    eAmps           = VK_PERFORMANCE_COUNTER_UNIT_AMPS_KHR,
    eHertz          = VK_PERFORMANCE_COUNTER_UNIT_HERTZ_KHR,
    eCycles         = VK_PERFORMANCE_COUNTER_UNIT_CYCLES_KHR
  };

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

  enum class AcquireProfilingLockFlagBitsKHR : VkAcquireProfilingLockFlagsKHR
  {
  };

  VULKAN_HPP_INLINE std::string to_string( AcquireProfilingLockFlagBitsKHR )
  {
    return "(void)";
  }

#if defined( VK_USE_PLATFORM_IOS_MVK )
  //=== VK_MVK_ios_surface ===

  enum class IOSSurfaceCreateFlagBitsMVK : VkIOSSurfaceCreateFlagsMVK
  {
  };

  VULKAN_HPP_INLINE std::string to_string( IOSSurfaceCreateFlagBitsMVK )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_IOS_MVK*/

#if defined( VK_USE_PLATFORM_MACOS_MVK )
  //=== VK_MVK_macos_surface ===

  enum class MacOSSurfaceCreateFlagBitsMVK : VkMacOSSurfaceCreateFlagsMVK
  {
  };

  VULKAN_HPP_INLINE std::string to_string( MacOSSurfaceCreateFlagBitsMVK )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_MACOS_MVK*/

  //=== VK_EXT_debug_utils ===

  enum class DebugUtilsMessageSeverityFlagBitsEXT : VkDebugUtilsMessageSeverityFlagsEXT
  {
    eVerbose = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT,
    eInfo    = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT,
    eWarning = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
    eError   = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
  };

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

  enum class DebugUtilsMessageTypeFlagBitsEXT : VkDebugUtilsMessageTypeFlagsEXT
  {
    eGeneral     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT,
    eValidation  = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
    ePerformance = VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT
  };

  VULKAN_HPP_INLINE std::string to_string( DebugUtilsMessageTypeFlagBitsEXT value )
  {
    switch ( value )
    {
      case DebugUtilsMessageTypeFlagBitsEXT::eGeneral: return "General";
      case DebugUtilsMessageTypeFlagBitsEXT::eValidation: return "Validation";
      case DebugUtilsMessageTypeFlagBitsEXT::ePerformance: return "Performance";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class DebugUtilsMessengerCallbackDataFlagBitsEXT : VkDebugUtilsMessengerCallbackDataFlagsEXT
  {
  };

  VULKAN_HPP_INLINE std::string to_string( DebugUtilsMessengerCallbackDataFlagBitsEXT )
  {
    return "(void)";
  }

  enum class DebugUtilsMessengerCreateFlagBitsEXT : VkDebugUtilsMessengerCreateFlagsEXT
  {
  };

  VULKAN_HPP_INLINE std::string to_string( DebugUtilsMessengerCreateFlagBitsEXT )
  {
    return "(void)";
  }

  //=== VK_EXT_blend_operation_advanced ===

  enum class BlendOverlapEXT
  {
    eUncorrelated = VK_BLEND_OVERLAP_UNCORRELATED_EXT,
    eDisjoint     = VK_BLEND_OVERLAP_DISJOINT_EXT,
    eConjoint     = VK_BLEND_OVERLAP_CONJOINT_EXT
  };

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

  enum class PipelineCoverageToColorStateCreateFlagBitsNV : VkPipelineCoverageToColorStateCreateFlagsNV
  {
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineCoverageToColorStateCreateFlagBitsNV )
  {
    return "(void)";
  }

  //=== VK_KHR_acceleration_structure ===

  enum class AccelerationStructureTypeKHR
  {
    eTopLevel    = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
    eBottomLevel = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
    eGeneric     = VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR
  };
  using AccelerationStructureTypeNV = AccelerationStructureTypeKHR;

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

  enum class AccelerationStructureBuildTypeKHR
  {
    eHost         = VK_ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_KHR,
    eDevice       = VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
    eHostOrDevice = VK_ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_OR_DEVICE_KHR
  };

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

  enum class GeometryFlagBitsKHR : VkGeometryFlagsKHR
  {
    eOpaque                      = VK_GEOMETRY_OPAQUE_BIT_KHR,
    eNoDuplicateAnyHitInvocation = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR
  };
  using GeometryFlagBitsNV = GeometryFlagBitsKHR;

  VULKAN_HPP_INLINE std::string to_string( GeometryFlagBitsKHR value )
  {
    switch ( value )
    {
      case GeometryFlagBitsKHR::eOpaque: return "Opaque";
      case GeometryFlagBitsKHR::eNoDuplicateAnyHitInvocation: return "NoDuplicateAnyHitInvocation";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class GeometryInstanceFlagBitsKHR : VkGeometryInstanceFlagsKHR
  {
    eTriangleFacingCullDisable     = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
    eTriangleFrontCounterclockwise = VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR,
    eForceOpaque                   = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR,
    eForceNoOpaque                 = VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR,
    eTriangleCullDisable           = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV
  };
  using GeometryInstanceFlagBitsNV = GeometryInstanceFlagBitsKHR;

  VULKAN_HPP_INLINE std::string to_string( GeometryInstanceFlagBitsKHR value )
  {
    switch ( value )
    {
      case GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable: return "TriangleFacingCullDisable";
      case GeometryInstanceFlagBitsKHR::eTriangleFrontCounterclockwise: return "TriangleFrontCounterclockwise";
      case GeometryInstanceFlagBitsKHR::eForceOpaque: return "ForceOpaque";
      case GeometryInstanceFlagBitsKHR::eForceNoOpaque: return "ForceNoOpaque";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class BuildAccelerationStructureFlagBitsKHR : VkBuildAccelerationStructureFlagsKHR
  {
    eAllowUpdate     = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
    eAllowCompaction = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR,
    ePreferFastTrace = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
    ePreferFastBuild = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
    eLowMemory       = VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR,
    eMotionNV        = VK_BUILD_ACCELERATION_STRUCTURE_MOTION_BIT_NV
  };
  using BuildAccelerationStructureFlagBitsNV = BuildAccelerationStructureFlagBitsKHR;

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
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class CopyAccelerationStructureModeKHR
  {
    eClone       = VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR,
    eCompact     = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR,
    eSerialize   = VK_COPY_ACCELERATION_STRUCTURE_MODE_SERIALIZE_KHR,
    eDeserialize = VK_COPY_ACCELERATION_STRUCTURE_MODE_DESERIALIZE_KHR
  };
  using CopyAccelerationStructureModeNV = CopyAccelerationStructureModeKHR;

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

  enum class GeometryTypeKHR
  {
    eTriangles = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
    eAabbs     = VK_GEOMETRY_TYPE_AABBS_KHR,
    eInstances = VK_GEOMETRY_TYPE_INSTANCES_KHR
  };
  using GeometryTypeNV = GeometryTypeKHR;

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

  enum class AccelerationStructureCompatibilityKHR
  {
    eCompatible   = VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR,
    eIncompatible = VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR
  };

  VULKAN_HPP_INLINE std::string to_string( AccelerationStructureCompatibilityKHR value )
  {
    switch ( value )
    {
      case AccelerationStructureCompatibilityKHR::eCompatible: return "Compatible";
      case AccelerationStructureCompatibilityKHR::eIncompatible: return "Incompatible";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class AccelerationStructureCreateFlagBitsKHR : VkAccelerationStructureCreateFlagsKHR
  {
    eDeviceAddressCaptureReplay = VK_ACCELERATION_STRUCTURE_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT_KHR,
    eMotionNV                   = VK_ACCELERATION_STRUCTURE_CREATE_MOTION_BIT_NV
  };

  VULKAN_HPP_INLINE std::string to_string( AccelerationStructureCreateFlagBitsKHR value )
  {
    switch ( value )
    {
      case AccelerationStructureCreateFlagBitsKHR::eDeviceAddressCaptureReplay: return "DeviceAddressCaptureReplay";
      case AccelerationStructureCreateFlagBitsKHR::eMotionNV: return "MotionNV";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class BuildAccelerationStructureModeKHR
  {
    eBuild  = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
    eUpdate = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
  };

  VULKAN_HPP_INLINE std::string to_string( BuildAccelerationStructureModeKHR value )
  {
    switch ( value )
    {
      case BuildAccelerationStructureModeKHR::eBuild: return "Build";
      case BuildAccelerationStructureModeKHR::eUpdate: return "Update";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_NV_framebuffer_mixed_samples ===

  enum class CoverageModulationModeNV
  {
    eNone  = VK_COVERAGE_MODULATION_MODE_NONE_NV,
    eRgb   = VK_COVERAGE_MODULATION_MODE_RGB_NV,
    eAlpha = VK_COVERAGE_MODULATION_MODE_ALPHA_NV,
    eRgba  = VK_COVERAGE_MODULATION_MODE_RGBA_NV
  };

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

  enum class PipelineCoverageModulationStateCreateFlagBitsNV : VkPipelineCoverageModulationStateCreateFlagsNV
  {
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineCoverageModulationStateCreateFlagBitsNV )
  {
    return "(void)";
  }

  //=== VK_EXT_validation_cache ===

  enum class ValidationCacheHeaderVersionEXT
  {
    eOne = VK_VALIDATION_CACHE_HEADER_VERSION_ONE_EXT
  };

  VULKAN_HPP_INLINE std::string to_string( ValidationCacheHeaderVersionEXT value )
  {
    switch ( value )
    {
      case ValidationCacheHeaderVersionEXT::eOne: return "One";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class ValidationCacheCreateFlagBitsEXT : VkValidationCacheCreateFlagsEXT
  {
  };

  VULKAN_HPP_INLINE std::string to_string( ValidationCacheCreateFlagBitsEXT )
  {
    return "(void)";
  }

  //=== VK_NV_shading_rate_image ===

  enum class ShadingRatePaletteEntryNV
  {
    eNoInvocations           = VK_SHADING_RATE_PALETTE_ENTRY_NO_INVOCATIONS_NV,
    e16InvocationsPerPixel   = VK_SHADING_RATE_PALETTE_ENTRY_16_INVOCATIONS_PER_PIXEL_NV,
    e8InvocationsPerPixel    = VK_SHADING_RATE_PALETTE_ENTRY_8_INVOCATIONS_PER_PIXEL_NV,
    e4InvocationsPerPixel    = VK_SHADING_RATE_PALETTE_ENTRY_4_INVOCATIONS_PER_PIXEL_NV,
    e2InvocationsPerPixel    = VK_SHADING_RATE_PALETTE_ENTRY_2_INVOCATIONS_PER_PIXEL_NV,
    e1InvocationPerPixel     = VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_PIXEL_NV,
    e1InvocationPer2X1Pixels = VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_2X1_PIXELS_NV,
    e1InvocationPer1X2Pixels = VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_1X2_PIXELS_NV,
    e1InvocationPer2X2Pixels = VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_2X2_PIXELS_NV,
    e1InvocationPer4X2Pixels = VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_4X2_PIXELS_NV,
    e1InvocationPer2X4Pixels = VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_2X4_PIXELS_NV,
    e1InvocationPer4X4Pixels = VK_SHADING_RATE_PALETTE_ENTRY_1_INVOCATION_PER_4X4_PIXELS_NV
  };

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

  enum class CoarseSampleOrderTypeNV
  {
    eDefault     = VK_COARSE_SAMPLE_ORDER_TYPE_DEFAULT_NV,
    eCustom      = VK_COARSE_SAMPLE_ORDER_TYPE_CUSTOM_NV,
    ePixelMajor  = VK_COARSE_SAMPLE_ORDER_TYPE_PIXEL_MAJOR_NV,
    eSampleMajor = VK_COARSE_SAMPLE_ORDER_TYPE_SAMPLE_MAJOR_NV
  };

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

  enum class AccelerationStructureMemoryRequirementsTypeNV
  {
    eObject        = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV,
    eBuildScratch  = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV,
    eUpdateScratch = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_UPDATE_SCRATCH_NV
  };

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

  //=== VK_EXT_global_priority ===

  enum class QueueGlobalPriorityEXT
  {
    eLow      = VK_QUEUE_GLOBAL_PRIORITY_LOW_EXT,
    eMedium   = VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_EXT,
    eHigh     = VK_QUEUE_GLOBAL_PRIORITY_HIGH_EXT,
    eRealtime = VK_QUEUE_GLOBAL_PRIORITY_REALTIME_EXT
  };

  VULKAN_HPP_INLINE std::string to_string( QueueGlobalPriorityEXT value )
  {
    switch ( value )
    {
      case QueueGlobalPriorityEXT::eLow: return "Low";
      case QueueGlobalPriorityEXT::eMedium: return "Medium";
      case QueueGlobalPriorityEXT::eHigh: return "High";
      case QueueGlobalPriorityEXT::eRealtime: return "Realtime";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_AMD_pipeline_compiler_control ===

  enum class PipelineCompilerControlFlagBitsAMD : VkPipelineCompilerControlFlagsAMD
  {
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineCompilerControlFlagBitsAMD )
  {
    return "(void)";
  }

  //=== VK_EXT_calibrated_timestamps ===

  enum class TimeDomainEXT
  {
    eDevice                  = VK_TIME_DOMAIN_DEVICE_EXT,
    eClockMonotonic          = VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT,
    eClockMonotonicRaw       = VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_EXT,
    eQueryPerformanceCounter = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT
  };

  VULKAN_HPP_INLINE std::string to_string( TimeDomainEXT value )
  {
    switch ( value )
    {
      case TimeDomainEXT::eDevice: return "Device";
      case TimeDomainEXT::eClockMonotonic: return "ClockMonotonic";
      case TimeDomainEXT::eClockMonotonicRaw: return "ClockMonotonicRaw";
      case TimeDomainEXT::eQueryPerformanceCounter: return "QueryPerformanceCounter";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_EXT_video_decode_h265 ===

  enum class VideoDecodeH265CreateFlagBitsEXT : VkVideoDecodeH265CreateFlagsEXT
  {
  };

  VULKAN_HPP_INLINE std::string to_string( VideoDecodeH265CreateFlagBitsEXT )
  {
    return "(void)";
  }
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_AMD_memory_overallocation_behavior ===

  enum class MemoryOverallocationBehaviorAMD
  {
    eDefault    = VK_MEMORY_OVERALLOCATION_BEHAVIOR_DEFAULT_AMD,
    eAllowed    = VK_MEMORY_OVERALLOCATION_BEHAVIOR_ALLOWED_AMD,
    eDisallowed = VK_MEMORY_OVERALLOCATION_BEHAVIOR_DISALLOWED_AMD
  };

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

  //=== VK_EXT_pipeline_creation_feedback ===

  enum class PipelineCreationFeedbackFlagBitsEXT : VkPipelineCreationFeedbackFlagsEXT
  {
    eValid                       = VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT_EXT,
    eApplicationPipelineCacheHit = VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT_EXT,
    eBasePipelineAcceleration    = VK_PIPELINE_CREATION_FEEDBACK_BASE_PIPELINE_ACCELERATION_BIT_EXT
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineCreationFeedbackFlagBitsEXT value )
  {
    switch ( value )
    {
      case PipelineCreationFeedbackFlagBitsEXT::eValid: return "Valid";
      case PipelineCreationFeedbackFlagBitsEXT::eApplicationPipelineCacheHit: return "ApplicationPipelineCacheHit";
      case PipelineCreationFeedbackFlagBitsEXT::eBasePipelineAcceleration: return "BasePipelineAcceleration";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_INTEL_performance_query ===

  enum class PerformanceConfigurationTypeINTEL
  {
    eCommandQueueMetricsDiscoveryActivated =
      VK_PERFORMANCE_CONFIGURATION_TYPE_COMMAND_QUEUE_METRICS_DISCOVERY_ACTIVATED_INTEL
  };

  VULKAN_HPP_INLINE std::string to_string( PerformanceConfigurationTypeINTEL value )
  {
    switch ( value )
    {
      case PerformanceConfigurationTypeINTEL::eCommandQueueMetricsDiscoveryActivated:
        return "CommandQueueMetricsDiscoveryActivated";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class QueryPoolSamplingModeINTEL
  {
    eManual = VK_QUERY_POOL_SAMPLING_MODE_MANUAL_INTEL
  };

  VULKAN_HPP_INLINE std::string to_string( QueryPoolSamplingModeINTEL value )
  {
    switch ( value )
    {
      case QueryPoolSamplingModeINTEL::eManual: return "Manual";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class PerformanceOverrideTypeINTEL
  {
    eNullHardware   = VK_PERFORMANCE_OVERRIDE_TYPE_NULL_HARDWARE_INTEL,
    eFlushGpuCaches = VK_PERFORMANCE_OVERRIDE_TYPE_FLUSH_GPU_CACHES_INTEL
  };

  VULKAN_HPP_INLINE std::string to_string( PerformanceOverrideTypeINTEL value )
  {
    switch ( value )
    {
      case PerformanceOverrideTypeINTEL::eNullHardware: return "NullHardware";
      case PerformanceOverrideTypeINTEL::eFlushGpuCaches: return "FlushGpuCaches";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class PerformanceParameterTypeINTEL
  {
    eHwCountersSupported   = VK_PERFORMANCE_PARAMETER_TYPE_HW_COUNTERS_SUPPORTED_INTEL,
    eStreamMarkerValidBits = VK_PERFORMANCE_PARAMETER_TYPE_STREAM_MARKER_VALID_BITS_INTEL
  };

  VULKAN_HPP_INLINE std::string to_string( PerformanceParameterTypeINTEL value )
  {
    switch ( value )
    {
      case PerformanceParameterTypeINTEL::eHwCountersSupported: return "HwCountersSupported";
      case PerformanceParameterTypeINTEL::eStreamMarkerValidBits: return "StreamMarkerValidBits";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class PerformanceValueTypeINTEL
  {
    eUint32 = VK_PERFORMANCE_VALUE_TYPE_UINT32_INTEL,
    eUint64 = VK_PERFORMANCE_VALUE_TYPE_UINT64_INTEL,
    eFloat  = VK_PERFORMANCE_VALUE_TYPE_FLOAT_INTEL,
    eBool   = VK_PERFORMANCE_VALUE_TYPE_BOOL_INTEL,
    eString = VK_PERFORMANCE_VALUE_TYPE_STRING_INTEL
  };

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

  enum class ImagePipeSurfaceCreateFlagBitsFUCHSIA : VkImagePipeSurfaceCreateFlagsFUCHSIA
  {
  };

  VULKAN_HPP_INLINE std::string to_string( ImagePipeSurfaceCreateFlagBitsFUCHSIA )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_surface ===

  enum class MetalSurfaceCreateFlagBitsEXT : VkMetalSurfaceCreateFlagsEXT
  {
  };

  VULKAN_HPP_INLINE std::string to_string( MetalSurfaceCreateFlagBitsEXT )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_KHR_fragment_shading_rate ===

  enum class FragmentShadingRateCombinerOpKHR
  {
    eKeep    = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR,
    eReplace = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR,
    eMin     = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_MIN_KHR,
    eMax     = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_MAX_KHR,
    eMul     = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_MUL_KHR
  };

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

  enum class ShaderCorePropertiesFlagBitsAMD : VkShaderCorePropertiesFlagsAMD
  {
  };

  VULKAN_HPP_INLINE std::string to_string( ShaderCorePropertiesFlagBitsAMD )
  {
    return "(void)";
  }

  //=== VK_EXT_tooling_info ===

  enum class ToolPurposeFlagBitsEXT : VkToolPurposeFlagsEXT
  {
    eValidation         = VK_TOOL_PURPOSE_VALIDATION_BIT_EXT,
    eProfiling          = VK_TOOL_PURPOSE_PROFILING_BIT_EXT,
    eTracing            = VK_TOOL_PURPOSE_TRACING_BIT_EXT,
    eAdditionalFeatures = VK_TOOL_PURPOSE_ADDITIONAL_FEATURES_BIT_EXT,
    eModifyingFeatures  = VK_TOOL_PURPOSE_MODIFYING_FEATURES_BIT_EXT,
    eDebugReporting     = VK_TOOL_PURPOSE_DEBUG_REPORTING_BIT_EXT,
    eDebugMarkers       = VK_TOOL_PURPOSE_DEBUG_MARKERS_BIT_EXT
  };

  VULKAN_HPP_INLINE std::string to_string( ToolPurposeFlagBitsEXT value )
  {
    switch ( value )
    {
      case ToolPurposeFlagBitsEXT::eValidation: return "Validation";
      case ToolPurposeFlagBitsEXT::eProfiling: return "Profiling";
      case ToolPurposeFlagBitsEXT::eTracing: return "Tracing";
      case ToolPurposeFlagBitsEXT::eAdditionalFeatures: return "AdditionalFeatures";
      case ToolPurposeFlagBitsEXT::eModifyingFeatures: return "ModifyingFeatures";
      case ToolPurposeFlagBitsEXT::eDebugReporting: return "DebugReporting";
      case ToolPurposeFlagBitsEXT::eDebugMarkers: return "DebugMarkers";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_EXT_validation_features ===

  enum class ValidationFeatureEnableEXT
  {
    eGpuAssisted                   = VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT,
    eGpuAssistedReserveBindingSlot = VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT,
    eBestPractices                 = VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT,
    eDebugPrintf                   = VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT,
    eSynchronizationValidation     = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT
  };

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

  enum class ValidationFeatureDisableEXT
  {
    eAll                   = VK_VALIDATION_FEATURE_DISABLE_ALL_EXT,
    eShaders               = VK_VALIDATION_FEATURE_DISABLE_SHADERS_EXT,
    eThreadSafety          = VK_VALIDATION_FEATURE_DISABLE_THREAD_SAFETY_EXT,
    eApiParameters         = VK_VALIDATION_FEATURE_DISABLE_API_PARAMETERS_EXT,
    eObjectLifetimes       = VK_VALIDATION_FEATURE_DISABLE_OBJECT_LIFETIMES_EXT,
    eCoreChecks            = VK_VALIDATION_FEATURE_DISABLE_CORE_CHECKS_EXT,
    eUniqueHandles         = VK_VALIDATION_FEATURE_DISABLE_UNIQUE_HANDLES_EXT,
    eShaderValidationCache = VK_VALIDATION_FEATURE_DISABLE_SHADER_VALIDATION_CACHE_EXT
  };

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

  //=== VK_NV_cooperative_matrix ===

  enum class ScopeNV
  {
    eDevice      = VK_SCOPE_DEVICE_NV,
    eWorkgroup   = VK_SCOPE_WORKGROUP_NV,
    eSubgroup    = VK_SCOPE_SUBGROUP_NV,
    eQueueFamily = VK_SCOPE_QUEUE_FAMILY_NV
  };

  VULKAN_HPP_INLINE std::string to_string( ScopeNV value )
  {
    switch ( value )
    {
      case ScopeNV::eDevice: return "Device";
      case ScopeNV::eWorkgroup: return "Workgroup";
      case ScopeNV::eSubgroup: return "Subgroup";
      case ScopeNV::eQueueFamily: return "QueueFamily";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class ComponentTypeNV
  {
    eFloat16 = VK_COMPONENT_TYPE_FLOAT16_NV,
    eFloat32 = VK_COMPONENT_TYPE_FLOAT32_NV,
    eFloat64 = VK_COMPONENT_TYPE_FLOAT64_NV,
    eSint8   = VK_COMPONENT_TYPE_SINT8_NV,
    eSint16  = VK_COMPONENT_TYPE_SINT16_NV,
    eSint32  = VK_COMPONENT_TYPE_SINT32_NV,
    eSint64  = VK_COMPONENT_TYPE_SINT64_NV,
    eUint8   = VK_COMPONENT_TYPE_UINT8_NV,
    eUint16  = VK_COMPONENT_TYPE_UINT16_NV,
    eUint32  = VK_COMPONENT_TYPE_UINT32_NV,
    eUint64  = VK_COMPONENT_TYPE_UINT64_NV
  };

  VULKAN_HPP_INLINE std::string to_string( ComponentTypeNV value )
  {
    switch ( value )
    {
      case ComponentTypeNV::eFloat16: return "Float16";
      case ComponentTypeNV::eFloat32: return "Float32";
      case ComponentTypeNV::eFloat64: return "Float64";
      case ComponentTypeNV::eSint8: return "Sint8";
      case ComponentTypeNV::eSint16: return "Sint16";
      case ComponentTypeNV::eSint32: return "Sint32";
      case ComponentTypeNV::eSint64: return "Sint64";
      case ComponentTypeNV::eUint8: return "Uint8";
      case ComponentTypeNV::eUint16: return "Uint16";
      case ComponentTypeNV::eUint32: return "Uint32";
      case ComponentTypeNV::eUint64: return "Uint64";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_NV_coverage_reduction_mode ===

  enum class CoverageReductionModeNV
  {
    eMerge    = VK_COVERAGE_REDUCTION_MODE_MERGE_NV,
    eTruncate = VK_COVERAGE_REDUCTION_MODE_TRUNCATE_NV
  };

  VULKAN_HPP_INLINE std::string to_string( CoverageReductionModeNV value )
  {
    switch ( value )
    {
      case CoverageReductionModeNV::eMerge: return "Merge";
      case CoverageReductionModeNV::eTruncate: return "Truncate";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class PipelineCoverageReductionStateCreateFlagBitsNV : VkPipelineCoverageReductionStateCreateFlagsNV
  {
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineCoverageReductionStateCreateFlagBitsNV )
  {
    return "(void)";
  }

  //=== VK_EXT_provoking_vertex ===

  enum class ProvokingVertexModeEXT
  {
    eFirstVertex = VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT,
    eLastVertex  = VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT
  };

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

  enum class FullScreenExclusiveEXT
  {
    eDefault               = VK_FULL_SCREEN_EXCLUSIVE_DEFAULT_EXT,
    eAllowed               = VK_FULL_SCREEN_EXCLUSIVE_ALLOWED_EXT,
    eDisallowed            = VK_FULL_SCREEN_EXCLUSIVE_DISALLOWED_EXT,
    eApplicationControlled = VK_FULL_SCREEN_EXCLUSIVE_APPLICATION_CONTROLLED_EXT
  };

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

  enum class HeadlessSurfaceCreateFlagBitsEXT : VkHeadlessSurfaceCreateFlagsEXT
  {
  };

  VULKAN_HPP_INLINE std::string to_string( HeadlessSurfaceCreateFlagBitsEXT )
  {
    return "(void)";
  }

  //=== VK_EXT_line_rasterization ===

  enum class LineRasterizationModeEXT
  {
    eDefault           = VK_LINE_RASTERIZATION_MODE_DEFAULT_EXT,
    eRectangular       = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_EXT,
    eBresenham         = VK_LINE_RASTERIZATION_MODE_BRESENHAM_EXT,
    eRectangularSmooth = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_EXT
  };

  VULKAN_HPP_INLINE std::string to_string( LineRasterizationModeEXT value )
  {
    switch ( value )
    {
      case LineRasterizationModeEXT::eDefault: return "Default";
      case LineRasterizationModeEXT::eRectangular: return "Rectangular";
      case LineRasterizationModeEXT::eBresenham: return "Bresenham";
      case LineRasterizationModeEXT::eRectangularSmooth: return "RectangularSmooth";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_KHR_pipeline_executable_properties ===

  enum class PipelineExecutableStatisticFormatKHR
  {
    eBool32  = VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR,
    eInt64   = VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR,
    eUint64  = VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR,
    eFloat64 = VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR
  };

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

  //=== VK_NV_device_generated_commands ===

  enum class IndirectStateFlagBitsNV : VkIndirectStateFlagsNV
  {
    eFlagFrontface = VK_INDIRECT_STATE_FLAG_FRONTFACE_BIT_NV
  };

  VULKAN_HPP_INLINE std::string to_string( IndirectStateFlagBitsNV value )
  {
    switch ( value )
    {
      case IndirectStateFlagBitsNV::eFlagFrontface: return "FlagFrontface";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class IndirectCommandsTokenTypeNV
  {
    eShaderGroup  = VK_INDIRECT_COMMANDS_TOKEN_TYPE_SHADER_GROUP_NV,
    eStateFlags   = VK_INDIRECT_COMMANDS_TOKEN_TYPE_STATE_FLAGS_NV,
    eIndexBuffer  = VK_INDIRECT_COMMANDS_TOKEN_TYPE_INDEX_BUFFER_NV,
    eVertexBuffer = VK_INDIRECT_COMMANDS_TOKEN_TYPE_VERTEX_BUFFER_NV,
    ePushConstant = VK_INDIRECT_COMMANDS_TOKEN_TYPE_PUSH_CONSTANT_NV,
    eDrawIndexed  = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_INDEXED_NV,
    eDraw         = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_NV,
    eDrawTasks    = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_TASKS_NV
  };

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
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class IndirectCommandsLayoutUsageFlagBitsNV : VkIndirectCommandsLayoutUsageFlagsNV
  {
    eExplicitPreprocess = VK_INDIRECT_COMMANDS_LAYOUT_USAGE_EXPLICIT_PREPROCESS_BIT_NV,
    eIndexedSequences   = VK_INDIRECT_COMMANDS_LAYOUT_USAGE_INDEXED_SEQUENCES_BIT_NV,
    eUnorderedSequences = VK_INDIRECT_COMMANDS_LAYOUT_USAGE_UNORDERED_SEQUENCES_BIT_NV
  };

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

  //=== VK_EXT_device_memory_report ===

  enum class DeviceMemoryReportEventTypeEXT
  {
    eAllocate         = VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_ALLOCATE_EXT,
    eFree             = VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_FREE_EXT,
    eImport           = VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_IMPORT_EXT,
    eUnimport         = VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_UNIMPORT_EXT,
    eAllocationFailed = VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_ALLOCATION_FAILED_EXT
  };

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

  enum class DeviceMemoryReportFlagBitsEXT : VkDeviceMemoryReportFlagsEXT
  {
  };

  VULKAN_HPP_INLINE std::string to_string( DeviceMemoryReportFlagBitsEXT )
  {
    return "(void)";
  }

  //=== VK_EXT_private_data ===

  enum class PrivateDataSlotCreateFlagBitsEXT : VkPrivateDataSlotCreateFlagsEXT
  {
  };

  VULKAN_HPP_INLINE std::string to_string( PrivateDataSlotCreateFlagBitsEXT )
  {
    return "(void)";
  }

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_KHR_video_encode_queue ===

  enum class VideoEncodeFlagBitsKHR : VkVideoEncodeFlagsKHR
  {
    eDefault   = VK_VIDEO_ENCODE_DEFAULT_KHR,
    eReserved0 = VK_VIDEO_ENCODE_RESERVED_0_BIT_KHR
  };

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoEncodeFlagBitsKHR::eDefault: return "Default";
      case VideoEncodeFlagBitsKHR::eReserved0: return "Reserved0";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class VideoEncodeRateControlFlagBitsKHR : VkVideoEncodeRateControlFlagsKHR
  {
    eDefault = VK_VIDEO_ENCODE_RATE_CONTROL_DEFAULT_KHR,
    eReset   = VK_VIDEO_ENCODE_RATE_CONTROL_RESET_BIT_KHR
  };

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeRateControlFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoEncodeRateControlFlagBitsKHR::eDefault: return "Default";
      case VideoEncodeRateControlFlagBitsKHR::eReset: return "Reset";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class VideoEncodeRateControlModeFlagBitsKHR : VkVideoEncodeRateControlModeFlagsKHR
  {
    eNone = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_NONE_BIT_KHR,
    eCbr  = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR,
    eVbr  = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_VBR_BIT_KHR
  };

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeRateControlModeFlagBitsKHR value )
  {
    switch ( value )
    {
      case VideoEncodeRateControlModeFlagBitsKHR::eNone: return "None";
      case VideoEncodeRateControlModeFlagBitsKHR::eCbr: return "Cbr";
      case VideoEncodeRateControlModeFlagBitsKHR::eVbr: return "Vbr";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_NV_device_diagnostics_config ===

  enum class DeviceDiagnosticsConfigFlagBitsNV : VkDeviceDiagnosticsConfigFlagsNV
  {
    eEnableShaderDebugInfo      = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_DEBUG_INFO_BIT_NV,
    eEnableResourceTracking     = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_RESOURCE_TRACKING_BIT_NV,
    eEnableAutomaticCheckpoints = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV
  };

  VULKAN_HPP_INLINE std::string to_string( DeviceDiagnosticsConfigFlagBitsNV value )
  {
    switch ( value )
    {
      case DeviceDiagnosticsConfigFlagBitsNV::eEnableShaderDebugInfo: return "EnableShaderDebugInfo";
      case DeviceDiagnosticsConfigFlagBitsNV::eEnableResourceTracking: return "EnableResourceTracking";
      case DeviceDiagnosticsConfigFlagBitsNV::eEnableAutomaticCheckpoints: return "EnableAutomaticCheckpoints";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_KHR_synchronization2 ===

  enum class PipelineStageFlagBits2KHR : VkPipelineStageFlags2KHR
  {
    eNone                         = VK_PIPELINE_STAGE_2_NONE_KHR,
    eTopOfPipe                    = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT_KHR,
    eDrawIndirect                 = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT_KHR,
    eVertexInput                  = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT_KHR,
    eVertexShader                 = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT_KHR,
    eTessellationControlShader    = VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT_KHR,
    eTessellationEvaluationShader = VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT_KHR,
    eGeometryShader               = VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT_KHR,
    eFragmentShader               = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT_KHR,
    eEarlyFragmentTests           = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT_KHR,
    eLateFragmentTests            = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT_KHR,
    eColorAttachmentOutput        = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR,
    eComputeShader                = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR,
    eAllTransfer                  = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT_KHR,
    eBottomOfPipe                 = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT_KHR,
    eHost                         = VK_PIPELINE_STAGE_2_HOST_BIT_KHR,
    eAllGraphics                  = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT_KHR,
    eAllCommands                  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR,
    eCopy                         = VK_PIPELINE_STAGE_2_COPY_BIT_KHR,
    eResolve                      = VK_PIPELINE_STAGE_2_RESOLVE_BIT_KHR,
    eBlit                         = VK_PIPELINE_STAGE_2_BLIT_BIT_KHR,
    eClear                        = VK_PIPELINE_STAGE_2_CLEAR_BIT_KHR,
    eIndexInput                   = VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT_KHR,
    eVertexAttributeInput         = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT_KHR,
    ePreRasterizationShaders      = VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT_KHR,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoDecode = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR,
    eVideoEncode = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eTransformFeedbackEXT          = VK_PIPELINE_STAGE_2_TRANSFORM_FEEDBACK_BIT_EXT,
    eConditionalRenderingEXT       = VK_PIPELINE_STAGE_2_CONDITIONAL_RENDERING_BIT_EXT,
    eCommandPreprocessNV           = VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_NV,
    eFragmentShadingRateAttachment = VK_PIPELINE_STAGE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
    eAccelerationStructureBuild    = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    eRayTracingShader              = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
    eFragmentDensityProcessEXT     = VK_PIPELINE_STAGE_2_FRAGMENT_DENSITY_PROCESS_BIT_EXT,
    eTaskShaderNV                  = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_NV,
    eMeshShaderNV                  = VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_NV,
    eSubpassShadingHUAWEI          = VK_PIPELINE_STAGE_2_SUBPASS_SHADING_BIT_HUAWEI,
    eAccelerationStructureBuildNV  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
    eRayTracingShaderNV            = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_NV,
    eShadingRateImageNV            = VK_PIPELINE_STAGE_2_SHADING_RATE_IMAGE_BIT_NV,
    eTransfer                      = VK_PIPELINE_STAGE_2_TRANSFER_BIT_KHR
  };

  VULKAN_HPP_INLINE std::string to_string( PipelineStageFlagBits2KHR value )
  {
    switch ( value )
    {
      case PipelineStageFlagBits2KHR::eNone: return "None";
      case PipelineStageFlagBits2KHR::eTopOfPipe: return "TopOfPipe";
      case PipelineStageFlagBits2KHR::eDrawIndirect: return "DrawIndirect";
      case PipelineStageFlagBits2KHR::eVertexInput: return "VertexInput";
      case PipelineStageFlagBits2KHR::eVertexShader: return "VertexShader";
      case PipelineStageFlagBits2KHR::eTessellationControlShader: return "TessellationControlShader";
      case PipelineStageFlagBits2KHR::eTessellationEvaluationShader: return "TessellationEvaluationShader";
      case PipelineStageFlagBits2KHR::eGeometryShader: return "GeometryShader";
      case PipelineStageFlagBits2KHR::eFragmentShader: return "FragmentShader";
      case PipelineStageFlagBits2KHR::eEarlyFragmentTests: return "EarlyFragmentTests";
      case PipelineStageFlagBits2KHR::eLateFragmentTests: return "LateFragmentTests";
      case PipelineStageFlagBits2KHR::eColorAttachmentOutput: return "ColorAttachmentOutput";
      case PipelineStageFlagBits2KHR::eComputeShader: return "ComputeShader";
      case PipelineStageFlagBits2KHR::eAllTransfer: return "AllTransfer";
      case PipelineStageFlagBits2KHR::eBottomOfPipe: return "BottomOfPipe";
      case PipelineStageFlagBits2KHR::eHost: return "Host";
      case PipelineStageFlagBits2KHR::eAllGraphics: return "AllGraphics";
      case PipelineStageFlagBits2KHR::eAllCommands: return "AllCommands";
      case PipelineStageFlagBits2KHR::eCopy: return "Copy";
      case PipelineStageFlagBits2KHR::eResolve: return "Resolve";
      case PipelineStageFlagBits2KHR::eBlit: return "Blit";
      case PipelineStageFlagBits2KHR::eClear: return "Clear";
      case PipelineStageFlagBits2KHR::eIndexInput: return "IndexInput";
      case PipelineStageFlagBits2KHR::eVertexAttributeInput: return "VertexAttributeInput";
      case PipelineStageFlagBits2KHR::ePreRasterizationShaders: return "PreRasterizationShaders";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case PipelineStageFlagBits2KHR::eVideoDecode: return "VideoDecode";
      case PipelineStageFlagBits2KHR::eVideoEncode: return "VideoEncode";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      case PipelineStageFlagBits2KHR::eTransformFeedbackEXT: return "TransformFeedbackEXT";
      case PipelineStageFlagBits2KHR::eConditionalRenderingEXT: return "ConditionalRenderingEXT";
      case PipelineStageFlagBits2KHR::eCommandPreprocessNV: return "CommandPreprocessNV";
      case PipelineStageFlagBits2KHR::eFragmentShadingRateAttachment: return "FragmentShadingRateAttachment";
      case PipelineStageFlagBits2KHR::eAccelerationStructureBuild: return "AccelerationStructureBuild";
      case PipelineStageFlagBits2KHR::eRayTracingShader: return "RayTracingShader";
      case PipelineStageFlagBits2KHR::eFragmentDensityProcessEXT: return "FragmentDensityProcessEXT";
      case PipelineStageFlagBits2KHR::eTaskShaderNV: return "TaskShaderNV";
      case PipelineStageFlagBits2KHR::eMeshShaderNV: return "MeshShaderNV";
      case PipelineStageFlagBits2KHR::eSubpassShadingHUAWEI: return "SubpassShadingHUAWEI";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class AccessFlagBits2KHR : VkAccessFlags2KHR
  {
    eNone                        = VK_ACCESS_2_NONE_KHR,
    eIndirectCommandRead         = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT_KHR,
    eIndexRead                   = VK_ACCESS_2_INDEX_READ_BIT_KHR,
    eVertexAttributeRead         = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT_KHR,
    eUniformRead                 = VK_ACCESS_2_UNIFORM_READ_BIT_KHR,
    eInputAttachmentRead         = VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT_KHR,
    eShaderRead                  = VK_ACCESS_2_SHADER_READ_BIT_KHR,
    eShaderWrite                 = VK_ACCESS_2_SHADER_WRITE_BIT_KHR,
    eColorAttachmentRead         = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT_KHR,
    eColorAttachmentWrite        = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT_KHR,
    eDepthStencilAttachmentRead  = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT_KHR,
    eDepthStencilAttachmentWrite = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT_KHR,
    eTransferRead                = VK_ACCESS_2_TRANSFER_READ_BIT_KHR,
    eTransferWrite               = VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR,
    eHostRead                    = VK_ACCESS_2_HOST_READ_BIT_KHR,
    eHostWrite                   = VK_ACCESS_2_HOST_WRITE_BIT_KHR,
    eMemoryRead                  = VK_ACCESS_2_MEMORY_READ_BIT_KHR,
    eMemoryWrite                 = VK_ACCESS_2_MEMORY_WRITE_BIT_KHR,
    eShaderSampledRead           = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT_KHR,
    eShaderStorageRead           = VK_ACCESS_2_SHADER_STORAGE_READ_BIT_KHR,
    eShaderStorageWrite          = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT_KHR,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoDecodeRead  = VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR,
    eVideoDecodeWrite = VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR,
    eVideoEncodeRead  = VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR,
    eVideoEncodeWrite = VK_ACCESS_2_VIDEO_ENCODE_WRITE_BIT_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eTransformFeedbackWriteEXT         = VK_ACCESS_2_TRANSFORM_FEEDBACK_WRITE_BIT_EXT,
    eTransformFeedbackCounterReadEXT   = VK_ACCESS_2_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT,
    eTransformFeedbackCounterWriteEXT  = VK_ACCESS_2_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT,
    eConditionalRenderingReadEXT       = VK_ACCESS_2_CONDITIONAL_RENDERING_READ_BIT_EXT,
    eCommandPreprocessReadNV           = VK_ACCESS_2_COMMAND_PREPROCESS_READ_BIT_NV,
    eCommandPreprocessWriteNV          = VK_ACCESS_2_COMMAND_PREPROCESS_WRITE_BIT_NV,
    eFragmentShadingRateAttachmentRead = VK_ACCESS_2_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR,
    eAccelerationStructureRead         = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    eAccelerationStructureWrite        = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
    eFragmentDensityMapReadEXT         = VK_ACCESS_2_FRAGMENT_DENSITY_MAP_READ_BIT_EXT,
    eColorAttachmentReadNoncoherentEXT = VK_ACCESS_2_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT,
    eAccelerationStructureReadNV       = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_NV,
    eAccelerationStructureWriteNV      = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_NV,
    eShadingRateImageReadNV            = VK_ACCESS_2_SHADING_RATE_IMAGE_READ_BIT_NV
  };

  VULKAN_HPP_INLINE std::string to_string( AccessFlagBits2KHR value )
  {
    switch ( value )
    {
      case AccessFlagBits2KHR::eNone: return "None";
      case AccessFlagBits2KHR::eIndirectCommandRead: return "IndirectCommandRead";
      case AccessFlagBits2KHR::eIndexRead: return "IndexRead";
      case AccessFlagBits2KHR::eVertexAttributeRead: return "VertexAttributeRead";
      case AccessFlagBits2KHR::eUniformRead: return "UniformRead";
      case AccessFlagBits2KHR::eInputAttachmentRead: return "InputAttachmentRead";
      case AccessFlagBits2KHR::eShaderRead: return "ShaderRead";
      case AccessFlagBits2KHR::eShaderWrite: return "ShaderWrite";
      case AccessFlagBits2KHR::eColorAttachmentRead: return "ColorAttachmentRead";
      case AccessFlagBits2KHR::eColorAttachmentWrite: return "ColorAttachmentWrite";
      case AccessFlagBits2KHR::eDepthStencilAttachmentRead: return "DepthStencilAttachmentRead";
      case AccessFlagBits2KHR::eDepthStencilAttachmentWrite: return "DepthStencilAttachmentWrite";
      case AccessFlagBits2KHR::eTransferRead: return "TransferRead";
      case AccessFlagBits2KHR::eTransferWrite: return "TransferWrite";
      case AccessFlagBits2KHR::eHostRead: return "HostRead";
      case AccessFlagBits2KHR::eHostWrite: return "HostWrite";
      case AccessFlagBits2KHR::eMemoryRead: return "MemoryRead";
      case AccessFlagBits2KHR::eMemoryWrite: return "MemoryWrite";
      case AccessFlagBits2KHR::eShaderSampledRead: return "ShaderSampledRead";
      case AccessFlagBits2KHR::eShaderStorageRead: return "ShaderStorageRead";
      case AccessFlagBits2KHR::eShaderStorageWrite: return "ShaderStorageWrite";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      case AccessFlagBits2KHR::eVideoDecodeRead: return "VideoDecodeRead";
      case AccessFlagBits2KHR::eVideoDecodeWrite: return "VideoDecodeWrite";
      case AccessFlagBits2KHR::eVideoEncodeRead: return "VideoEncodeRead";
      case AccessFlagBits2KHR::eVideoEncodeWrite: return "VideoEncodeWrite";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      case AccessFlagBits2KHR::eTransformFeedbackWriteEXT: return "TransformFeedbackWriteEXT";
      case AccessFlagBits2KHR::eTransformFeedbackCounterReadEXT: return "TransformFeedbackCounterReadEXT";
      case AccessFlagBits2KHR::eTransformFeedbackCounterWriteEXT: return "TransformFeedbackCounterWriteEXT";
      case AccessFlagBits2KHR::eConditionalRenderingReadEXT: return "ConditionalRenderingReadEXT";
      case AccessFlagBits2KHR::eCommandPreprocessReadNV: return "CommandPreprocessReadNV";
      case AccessFlagBits2KHR::eCommandPreprocessWriteNV: return "CommandPreprocessWriteNV";
      case AccessFlagBits2KHR::eFragmentShadingRateAttachmentRead: return "FragmentShadingRateAttachmentRead";
      case AccessFlagBits2KHR::eAccelerationStructureRead: return "AccelerationStructureRead";
      case AccessFlagBits2KHR::eAccelerationStructureWrite: return "AccelerationStructureWrite";
      case AccessFlagBits2KHR::eFragmentDensityMapReadEXT: return "FragmentDensityMapReadEXT";
      case AccessFlagBits2KHR::eColorAttachmentReadNoncoherentEXT: return "ColorAttachmentReadNoncoherentEXT";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  enum class SubmitFlagBitsKHR : VkSubmitFlagsKHR
  {
    eProtected = VK_SUBMIT_PROTECTED_BIT_KHR
  };

  VULKAN_HPP_INLINE std::string to_string( SubmitFlagBitsKHR value )
  {
    switch ( value )
    {
      case SubmitFlagBitsKHR::eProtected: return "Protected";
      default: return "invalid ( " + VULKAN_HPP_NAMESPACE::toHexString( static_cast<uint32_t>( value ) ) + " )";
    }
  }

  //=== VK_NV_fragment_shading_rate_enums ===

  enum class FragmentShadingRateNV
  {
    e1InvocationPerPixel     = VK_FRAGMENT_SHADING_RATE_1_INVOCATION_PER_PIXEL_NV,
    e1InvocationPer1X2Pixels = VK_FRAGMENT_SHADING_RATE_1_INVOCATION_PER_1X2_PIXELS_NV,
    e1InvocationPer2X1Pixels = VK_FRAGMENT_SHADING_RATE_1_INVOCATION_PER_2X1_PIXELS_NV,
    e1InvocationPer2X2Pixels = VK_FRAGMENT_SHADING_RATE_1_INVOCATION_PER_2X2_PIXELS_NV,
    e1InvocationPer2X4Pixels = VK_FRAGMENT_SHADING_RATE_1_INVOCATION_PER_2X4_PIXELS_NV,
    e1InvocationPer4X2Pixels = VK_FRAGMENT_SHADING_RATE_1_INVOCATION_PER_4X2_PIXELS_NV,
    e1InvocationPer4X4Pixels = VK_FRAGMENT_SHADING_RATE_1_INVOCATION_PER_4X4_PIXELS_NV,
    e2InvocationsPerPixel    = VK_FRAGMENT_SHADING_RATE_2_INVOCATIONS_PER_PIXEL_NV,
    e4InvocationsPerPixel    = VK_FRAGMENT_SHADING_RATE_4_INVOCATIONS_PER_PIXEL_NV,
    e8InvocationsPerPixel    = VK_FRAGMENT_SHADING_RATE_8_INVOCATIONS_PER_PIXEL_NV,
    e16InvocationsPerPixel   = VK_FRAGMENT_SHADING_RATE_16_INVOCATIONS_PER_PIXEL_NV,
    eNoInvocations           = VK_FRAGMENT_SHADING_RATE_NO_INVOCATIONS_NV
  };

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

  enum class FragmentShadingRateTypeNV
  {
    eFragmentSize = VK_FRAGMENT_SHADING_RATE_TYPE_FRAGMENT_SIZE_NV,
    eEnums        = VK_FRAGMENT_SHADING_RATE_TYPE_ENUMS_NV
  };

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

  enum class AccelerationStructureMotionInstanceTypeNV
  {
    eStatic       = VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_STATIC_NV,
    eMatrixMotion = VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_MATRIX_MOTION_NV,
    eSrtMotion    = VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_SRT_MOTION_NV
  };

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

  enum class AccelerationStructureMotionInfoFlagBitsNV : VkAccelerationStructureMotionInfoFlagsNV
  {
  };

  VULKAN_HPP_INLINE std::string to_string( AccelerationStructureMotionInfoFlagBitsNV )
  {
    return "(void)";
  }

  enum class AccelerationStructureMotionInstanceFlagBitsNV : VkAccelerationStructureMotionInstanceFlagsNV
  {
  };

  VULKAN_HPP_INLINE std::string to_string( AccelerationStructureMotionInstanceFlagBitsNV )
  {
    return "(void)";
  }

#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
  //=== VK_EXT_directfb_surface ===

  enum class DirectFBSurfaceCreateFlagBitsEXT : VkDirectFBSurfaceCreateFlagsEXT
  {
  };

  VULKAN_HPP_INLINE std::string to_string( DirectFBSurfaceCreateFlagBitsEXT )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

  //=== VK_KHR_ray_tracing_pipeline ===

  enum class RayTracingShaderGroupTypeKHR
  {
    eGeneral            = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
    eTrianglesHitGroup  = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
    eProceduralHitGroup = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR
  };
  using RayTracingShaderGroupTypeNV = RayTracingShaderGroupTypeKHR;

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

  enum class ShaderGroupShaderKHR
  {
    eGeneral      = VK_SHADER_GROUP_SHADER_GENERAL_KHR,
    eClosestHit   = VK_SHADER_GROUP_SHADER_CLOSEST_HIT_KHR,
    eAnyHit       = VK_SHADER_GROUP_SHADER_ANY_HIT_KHR,
    eIntersection = VK_SHADER_GROUP_SHADER_INTERSECTION_KHR
  };

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

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
  //=== VK_QNX_screen_surface ===

  enum class ScreenSurfaceCreateFlagBitsQNX : VkScreenSurfaceCreateFlagsQNX
  {
  };

  VULKAN_HPP_INLINE std::string to_string( ScreenSurfaceCreateFlagBitsQNX )
  {
    return "(void)";
  }
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

  template <ObjectType value>
  struct cpp_type
  {};

  template <typename T>
  struct IndexTypeValue
  {};

  template <>
  struct IndexTypeValue<uint16_t>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR IndexType value = IndexType::eUint16;
  };

  template <>
  struct CppType<IndexType, IndexType::eUint16>
  {
    using Type = uint16_t;
  };

  template <>
  struct IndexTypeValue<uint32_t>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR IndexType value = IndexType::eUint32;
  };

  template <>
  struct CppType<IndexType, IndexType::eUint32>
  {
    using Type = uint32_t;
  };

  template <>
  struct IndexTypeValue<uint8_t>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR IndexType value = IndexType::eUint8EXT;
  };

  template <>
  struct CppType<IndexType, IndexType::eUint8EXT>
  {
    using Type = uint8_t;
  };

  //================
  //=== BITMASKs ===
  //================

  //=== VK_VERSION_1_0 ===

  using FormatFeatureFlags = Flags<FormatFeatureFlagBits>;

  template <>
  struct FlagTraits<FormatFeatureFlagBits>
  {
    enum : VkFlags
    {
      allFlags =
        VkFlags( FormatFeatureFlagBits::eSampledImage ) | VkFlags( FormatFeatureFlagBits::eStorageImage ) |
        VkFlags( FormatFeatureFlagBits::eStorageImageAtomic ) | VkFlags( FormatFeatureFlagBits::eUniformTexelBuffer ) |
        VkFlags( FormatFeatureFlagBits::eStorageTexelBuffer ) |
        VkFlags( FormatFeatureFlagBits::eStorageTexelBufferAtomic ) | VkFlags( FormatFeatureFlagBits::eVertexBuffer ) |
        VkFlags( FormatFeatureFlagBits::eColorAttachment ) | VkFlags( FormatFeatureFlagBits::eColorAttachmentBlend ) |
        VkFlags( FormatFeatureFlagBits::eDepthStencilAttachment ) | VkFlags( FormatFeatureFlagBits::eBlitSrc ) |
        VkFlags( FormatFeatureFlagBits::eBlitDst ) | VkFlags( FormatFeatureFlagBits::eSampledImageFilterLinear ) |
        VkFlags( FormatFeatureFlagBits::eTransferSrc ) | VkFlags( FormatFeatureFlagBits::eTransferDst ) |
        VkFlags( FormatFeatureFlagBits::eMidpointChromaSamples ) |
        VkFlags( FormatFeatureFlagBits::eSampledImageYcbcrConversionLinearFilter ) |
        VkFlags( FormatFeatureFlagBits::eSampledImageYcbcrConversionSeparateReconstructionFilter ) |
        VkFlags( FormatFeatureFlagBits::eSampledImageYcbcrConversionChromaReconstructionExplicit ) |
        VkFlags( FormatFeatureFlagBits::eSampledImageYcbcrConversionChromaReconstructionExplicitForceable ) |
        VkFlags( FormatFeatureFlagBits::eDisjoint ) | VkFlags( FormatFeatureFlagBits::eCositedChromaSamples ) |
        VkFlags( FormatFeatureFlagBits::eSampledImageFilterMinmax ) |
        VkFlags( FormatFeatureFlagBits::eSampledImageFilterCubicIMG )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
        | VkFlags( FormatFeatureFlagBits::eVideoDecodeOutputKHR ) | VkFlags( FormatFeatureFlagBits::eVideoDecodeDpbKHR )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
        | VkFlags( FormatFeatureFlagBits::eAccelerationStructureVertexBufferKHR ) |
        VkFlags( FormatFeatureFlagBits::eFragmentDensityMapEXT ) |
        VkFlags( FormatFeatureFlagBits::eFragmentShadingRateAttachmentKHR )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
        | VkFlags( FormatFeatureFlagBits::eVideoEncodeInputKHR ) | VkFlags( FormatFeatureFlagBits::eVideoEncodeDpbKHR )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FormatFeatureFlags operator|( FormatFeatureFlagBits bit0,
                                                                       FormatFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return FormatFeatureFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FormatFeatureFlags operator&(FormatFeatureFlagBits bit0,
                                                                      FormatFeatureFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return FormatFeatureFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FormatFeatureFlags operator^( FormatFeatureFlagBits bit0,
                                                                       FormatFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return FormatFeatureFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FormatFeatureFlags operator~( FormatFeatureFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( FormatFeatureFlags( bits ) );
  }

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
    if ( value & FormatFeatureFlagBits::eSampledImageFilterCubicIMG )
      result += "SampledImageFilterCubicIMG | ";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & FormatFeatureFlagBits::eVideoDecodeOutputKHR )
      result += "VideoDecodeOutputKHR | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & FormatFeatureFlagBits::eVideoDecodeDpbKHR )
      result += "VideoDecodeDpbKHR | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    if ( value & FormatFeatureFlagBits::eAccelerationStructureVertexBufferKHR )
      result += "AccelerationStructureVertexBufferKHR | ";
    if ( value & FormatFeatureFlagBits::eFragmentDensityMapEXT )
      result += "FragmentDensityMapEXT | ";
    if ( value & FormatFeatureFlagBits::eFragmentShadingRateAttachmentKHR )
      result += "FragmentShadingRateAttachmentKHR | ";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & FormatFeatureFlagBits::eVideoEncodeInputKHR )
      result += "VideoEncodeInputKHR | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & FormatFeatureFlagBits::eVideoEncodeDpbKHR )
      result += "VideoEncodeDpbKHR | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using ImageCreateFlags = Flags<ImageCreateFlagBits>;

  template <>
  struct FlagTraits<ImageCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ImageCreateFlagBits::eSparseBinding ) | VkFlags( ImageCreateFlagBits::eSparseResidency ) |
                 VkFlags( ImageCreateFlagBits::eSparseAliased ) | VkFlags( ImageCreateFlagBits::eMutableFormat ) |
                 VkFlags( ImageCreateFlagBits::eCubeCompatible ) | VkFlags( ImageCreateFlagBits::eAlias ) |
                 VkFlags( ImageCreateFlagBits::eSplitInstanceBindRegions ) |
                 VkFlags( ImageCreateFlagBits::e2DArrayCompatible ) |
                 VkFlags( ImageCreateFlagBits::eBlockTexelViewCompatible ) |
                 VkFlags( ImageCreateFlagBits::eExtendedUsage ) | VkFlags( ImageCreateFlagBits::eProtected ) |
                 VkFlags( ImageCreateFlagBits::eDisjoint ) | VkFlags( ImageCreateFlagBits::eCornerSampledNV ) |
                 VkFlags( ImageCreateFlagBits::eSampleLocationsCompatibleDepthEXT ) |
                 VkFlags( ImageCreateFlagBits::eSubsampledEXT )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageCreateFlags operator|( ImageCreateFlagBits bit0,
                                                                     ImageCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageCreateFlags operator&(ImageCreateFlagBits bit0,
                                                                    ImageCreateFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return ImageCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageCreateFlags operator^( ImageCreateFlagBits bit0,
                                                                     ImageCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageCreateFlags operator~( ImageCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ImageCreateFlags( bits ) );
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
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using ImageUsageFlags = Flags<ImageUsageFlagBits>;

  template <>
  struct FlagTraits<ImageUsageFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ImageUsageFlagBits::eTransferSrc ) | VkFlags( ImageUsageFlagBits::eTransferDst ) |
                 VkFlags( ImageUsageFlagBits::eSampled ) | VkFlags( ImageUsageFlagBits::eStorage ) |
                 VkFlags( ImageUsageFlagBits::eColorAttachment ) |
                 VkFlags( ImageUsageFlagBits::eDepthStencilAttachment ) |
                 VkFlags( ImageUsageFlagBits::eTransientAttachment ) | VkFlags( ImageUsageFlagBits::eInputAttachment )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
                 | VkFlags( ImageUsageFlagBits::eVideoDecodeDstKHR ) |
                 VkFlags( ImageUsageFlagBits::eVideoDecodeSrcKHR ) | VkFlags( ImageUsageFlagBits::eVideoDecodeDpbKHR )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
                 | VkFlags( ImageUsageFlagBits::eFragmentDensityMapEXT ) |
                 VkFlags( ImageUsageFlagBits::eFragmentShadingRateAttachmentKHR )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
                 | VkFlags( ImageUsageFlagBits::eVideoEncodeDstKHR ) |
                 VkFlags( ImageUsageFlagBits::eVideoEncodeSrcKHR ) | VkFlags( ImageUsageFlagBits::eVideoEncodeDpbKHR )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageUsageFlags operator|( ImageUsageFlagBits bit0,
                                                                    ImageUsageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageUsageFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageUsageFlags operator&(ImageUsageFlagBits bit0,
                                                                   ImageUsageFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return ImageUsageFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageUsageFlags operator^( ImageUsageFlagBits bit0,
                                                                    ImageUsageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageUsageFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageUsageFlags operator~( ImageUsageFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ImageUsageFlags( bits ) );
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
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & ImageUsageFlagBits::eVideoDecodeDstKHR )
      result += "VideoDecodeDstKHR | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & ImageUsageFlagBits::eVideoDecodeSrcKHR )
      result += "VideoDecodeSrcKHR | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & ImageUsageFlagBits::eVideoDecodeDpbKHR )
      result += "VideoDecodeDpbKHR | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    if ( value & ImageUsageFlagBits::eFragmentDensityMapEXT )
      result += "FragmentDensityMapEXT | ";
    if ( value & ImageUsageFlagBits::eFragmentShadingRateAttachmentKHR )
      result += "FragmentShadingRateAttachmentKHR | ";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & ImageUsageFlagBits::eVideoEncodeDstKHR )
      result += "VideoEncodeDstKHR | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & ImageUsageFlagBits::eVideoEncodeSrcKHR )
      result += "VideoEncodeSrcKHR | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & ImageUsageFlagBits::eVideoEncodeDpbKHR )
      result += "VideoEncodeDpbKHR | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using InstanceCreateFlags = Flags<InstanceCreateFlagBits>;

  VULKAN_HPP_INLINE std::string to_string( InstanceCreateFlags )
  {
    return "{}";
  }

  using MemoryHeapFlags = Flags<MemoryHeapFlagBits>;

  template <>
  struct FlagTraits<MemoryHeapFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( MemoryHeapFlagBits::eDeviceLocal ) | VkFlags( MemoryHeapFlagBits::eMultiInstance )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MemoryHeapFlags operator|( MemoryHeapFlagBits bit0,
                                                                    MemoryHeapFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return MemoryHeapFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MemoryHeapFlags operator&(MemoryHeapFlagBits bit0,
                                                                   MemoryHeapFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return MemoryHeapFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MemoryHeapFlags operator^( MemoryHeapFlagBits bit0,
                                                                    MemoryHeapFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return MemoryHeapFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MemoryHeapFlags operator~( MemoryHeapFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( MemoryHeapFlags( bits ) );
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

  using MemoryPropertyFlags = Flags<MemoryPropertyFlagBits>;

  template <>
  struct FlagTraits<MemoryPropertyFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( MemoryPropertyFlagBits::eDeviceLocal ) | VkFlags( MemoryPropertyFlagBits::eHostVisible ) |
                 VkFlags( MemoryPropertyFlagBits::eHostCoherent ) | VkFlags( MemoryPropertyFlagBits::eHostCached ) |
                 VkFlags( MemoryPropertyFlagBits::eLazilyAllocated ) | VkFlags( MemoryPropertyFlagBits::eProtected ) |
                 VkFlags( MemoryPropertyFlagBits::eDeviceCoherentAMD ) |
                 VkFlags( MemoryPropertyFlagBits::eDeviceUncachedAMD )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MemoryPropertyFlags
                                         operator|( MemoryPropertyFlagBits bit0, MemoryPropertyFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return MemoryPropertyFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MemoryPropertyFlags operator&(MemoryPropertyFlagBits bit0,
                                                                       MemoryPropertyFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return MemoryPropertyFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MemoryPropertyFlags
                                         operator^( MemoryPropertyFlagBits bit0, MemoryPropertyFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return MemoryPropertyFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MemoryPropertyFlags operator~( MemoryPropertyFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( MemoryPropertyFlags( bits ) );
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
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using QueueFlags = Flags<QueueFlagBits>;

  template <>
  struct FlagTraits<QueueFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( QueueFlagBits::eGraphics ) | VkFlags( QueueFlagBits::eCompute ) |
                 VkFlags( QueueFlagBits::eTransfer ) | VkFlags( QueueFlagBits::eSparseBinding ) |
                 VkFlags( QueueFlagBits::eProtected )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
                 | VkFlags( QueueFlagBits::eVideoDecodeKHR ) | VkFlags( QueueFlagBits::eVideoEncodeKHR )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueueFlags operator|( QueueFlagBits bit0,
                                                               QueueFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return QueueFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueueFlags operator&(QueueFlagBits bit0, QueueFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return QueueFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueueFlags operator^( QueueFlagBits bit0,
                                                               QueueFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return QueueFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueueFlags operator~( QueueFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( QueueFlags( bits ) );
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
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & QueueFlagBits::eVideoDecodeKHR )
      result += "VideoDecodeKHR | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & QueueFlagBits::eVideoEncodeKHR )
      result += "VideoEncodeKHR | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using SampleCountFlags = Flags<SampleCountFlagBits>;

  template <>
  struct FlagTraits<SampleCountFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( SampleCountFlagBits::e1 ) | VkFlags( SampleCountFlagBits::e2 ) |
                 VkFlags( SampleCountFlagBits::e4 ) | VkFlags( SampleCountFlagBits::e8 ) |
                 VkFlags( SampleCountFlagBits::e16 ) | VkFlags( SampleCountFlagBits::e32 ) |
                 VkFlags( SampleCountFlagBits::e64 )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SampleCountFlags operator|( SampleCountFlagBits bit0,
                                                                     SampleCountFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SampleCountFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SampleCountFlags operator&(SampleCountFlagBits bit0,
                                                                    SampleCountFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return SampleCountFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SampleCountFlags operator^( SampleCountFlagBits bit0,
                                                                     SampleCountFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SampleCountFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SampleCountFlags operator~( SampleCountFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( SampleCountFlags( bits ) );
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

  using DeviceCreateFlags = Flags<DeviceCreateFlagBits>;

  VULKAN_HPP_INLINE std::string to_string( DeviceCreateFlags )
  {
    return "{}";
  }

  using DeviceQueueCreateFlags = Flags<DeviceQueueCreateFlagBits>;

  template <>
  struct FlagTraits<DeviceQueueCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( DeviceQueueCreateFlagBits::eProtected )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceQueueCreateFlags
                                         operator|( DeviceQueueCreateFlagBits bit0, DeviceQueueCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DeviceQueueCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceQueueCreateFlags
                                         operator&(DeviceQueueCreateFlagBits bit0, DeviceQueueCreateFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return DeviceQueueCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceQueueCreateFlags
                                         operator^( DeviceQueueCreateFlagBits bit0, DeviceQueueCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DeviceQueueCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceQueueCreateFlags operator~( DeviceQueueCreateFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( DeviceQueueCreateFlags( bits ) );
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

  using PipelineStageFlags = Flags<PipelineStageFlagBits>;

  template <>
  struct FlagTraits<PipelineStageFlagBits>
  {
    enum : VkFlags
    {
      allFlags =
        VkFlags( PipelineStageFlagBits::eTopOfPipe ) | VkFlags( PipelineStageFlagBits::eDrawIndirect ) |
        VkFlags( PipelineStageFlagBits::eVertexInput ) | VkFlags( PipelineStageFlagBits::eVertexShader ) |
        VkFlags( PipelineStageFlagBits::eTessellationControlShader ) |
        VkFlags( PipelineStageFlagBits::eTessellationEvaluationShader ) |
        VkFlags( PipelineStageFlagBits::eGeometryShader ) | VkFlags( PipelineStageFlagBits::eFragmentShader ) |
        VkFlags( PipelineStageFlagBits::eEarlyFragmentTests ) | VkFlags( PipelineStageFlagBits::eLateFragmentTests ) |
        VkFlags( PipelineStageFlagBits::eColorAttachmentOutput ) | VkFlags( PipelineStageFlagBits::eComputeShader ) |
        VkFlags( PipelineStageFlagBits::eTransfer ) | VkFlags( PipelineStageFlagBits::eBottomOfPipe ) |
        VkFlags( PipelineStageFlagBits::eHost ) | VkFlags( PipelineStageFlagBits::eAllGraphics ) |
        VkFlags( PipelineStageFlagBits::eAllCommands ) | VkFlags( PipelineStageFlagBits::eTransformFeedbackEXT ) |
        VkFlags( PipelineStageFlagBits::eConditionalRenderingEXT ) |
        VkFlags( PipelineStageFlagBits::eAccelerationStructureBuildKHR ) |
        VkFlags( PipelineStageFlagBits::eRayTracingShaderKHR ) | VkFlags( PipelineStageFlagBits::eTaskShaderNV ) |
        VkFlags( PipelineStageFlagBits::eMeshShaderNV ) | VkFlags( PipelineStageFlagBits::eFragmentDensityProcessEXT ) |
        VkFlags( PipelineStageFlagBits::eFragmentShadingRateAttachmentKHR ) |
        VkFlags( PipelineStageFlagBits::eCommandPreprocessNV ) | VkFlags( PipelineStageFlagBits::eNoneKHR )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineStageFlags operator|( PipelineStageFlagBits bit0,
                                                                       PipelineStageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineStageFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineStageFlags operator&(PipelineStageFlagBits bit0,
                                                                      PipelineStageFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return PipelineStageFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineStageFlags operator^( PipelineStageFlagBits bit0,
                                                                       PipelineStageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineStageFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineStageFlags operator~( PipelineStageFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( PipelineStageFlags( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineStageFlags value )
  {
    if ( !value )
      return "{}";

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
    if ( value & PipelineStageFlagBits::eTaskShaderNV )
      result += "TaskShaderNV | ";
    if ( value & PipelineStageFlagBits::eMeshShaderNV )
      result += "MeshShaderNV | ";
    if ( value & PipelineStageFlagBits::eFragmentDensityProcessEXT )
      result += "FragmentDensityProcessEXT | ";
    if ( value & PipelineStageFlagBits::eFragmentShadingRateAttachmentKHR )
      result += "FragmentShadingRateAttachmentKHR | ";
    if ( value & PipelineStageFlagBits::eCommandPreprocessNV )
      result += "CommandPreprocessNV | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using MemoryMapFlags = Flags<MemoryMapFlagBits>;

  VULKAN_HPP_INLINE std::string to_string( MemoryMapFlags )
  {
    return "{}";
  }

  using ImageAspectFlags = Flags<ImageAspectFlagBits>;

  template <>
  struct FlagTraits<ImageAspectFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ImageAspectFlagBits::eColor ) | VkFlags( ImageAspectFlagBits::eDepth ) |
                 VkFlags( ImageAspectFlagBits::eStencil ) | VkFlags( ImageAspectFlagBits::eMetadata ) |
                 VkFlags( ImageAspectFlagBits::ePlane0 ) | VkFlags( ImageAspectFlagBits::ePlane1 ) |
                 VkFlags( ImageAspectFlagBits::ePlane2 ) | VkFlags( ImageAspectFlagBits::eMemoryPlane0EXT ) |
                 VkFlags( ImageAspectFlagBits::eMemoryPlane1EXT ) | VkFlags( ImageAspectFlagBits::eMemoryPlane2EXT ) |
                 VkFlags( ImageAspectFlagBits::eMemoryPlane3EXT )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageAspectFlags operator|( ImageAspectFlagBits bit0,
                                                                     ImageAspectFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageAspectFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageAspectFlags operator&(ImageAspectFlagBits bit0,
                                                                    ImageAspectFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return ImageAspectFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageAspectFlags operator^( ImageAspectFlagBits bit0,
                                                                     ImageAspectFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageAspectFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageAspectFlags operator~( ImageAspectFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ImageAspectFlags( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( ImageAspectFlags value )
  {
    if ( !value )
      return "{}";

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

  using SparseImageFormatFlags = Flags<SparseImageFormatFlagBits>;

  template <>
  struct FlagTraits<SparseImageFormatFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( SparseImageFormatFlagBits::eSingleMiptail ) |
                 VkFlags( SparseImageFormatFlagBits::eAlignedMipSize ) |
                 VkFlags( SparseImageFormatFlagBits::eNonstandardBlockSize )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SparseImageFormatFlags
                                         operator|( SparseImageFormatFlagBits bit0, SparseImageFormatFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SparseImageFormatFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SparseImageFormatFlags
                                         operator&(SparseImageFormatFlagBits bit0, SparseImageFormatFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return SparseImageFormatFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SparseImageFormatFlags
                                         operator^( SparseImageFormatFlagBits bit0, SparseImageFormatFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SparseImageFormatFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SparseImageFormatFlags operator~( SparseImageFormatFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( SparseImageFormatFlags( bits ) );
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

  using SparseMemoryBindFlags = Flags<SparseMemoryBindFlagBits>;

  template <>
  struct FlagTraits<SparseMemoryBindFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( SparseMemoryBindFlagBits::eMetadata )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SparseMemoryBindFlags
                                         operator|( SparseMemoryBindFlagBits bit0, SparseMemoryBindFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SparseMemoryBindFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SparseMemoryBindFlags
                                         operator&(SparseMemoryBindFlagBits bit0, SparseMemoryBindFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return SparseMemoryBindFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SparseMemoryBindFlags
                                         operator^( SparseMemoryBindFlagBits bit0, SparseMemoryBindFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SparseMemoryBindFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SparseMemoryBindFlags operator~( SparseMemoryBindFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( SparseMemoryBindFlags( bits ) );
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

  using FenceCreateFlags = Flags<FenceCreateFlagBits>;

  template <>
  struct FlagTraits<FenceCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( FenceCreateFlagBits::eSignaled )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FenceCreateFlags operator|( FenceCreateFlagBits bit0,
                                                                     FenceCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return FenceCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FenceCreateFlags operator&(FenceCreateFlagBits bit0,
                                                                    FenceCreateFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return FenceCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FenceCreateFlags operator^( FenceCreateFlagBits bit0,
                                                                     FenceCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return FenceCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FenceCreateFlags operator~( FenceCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( FenceCreateFlags( bits ) );
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

  using SemaphoreCreateFlags = Flags<SemaphoreCreateFlagBits>;

  VULKAN_HPP_INLINE std::string to_string( SemaphoreCreateFlags )
  {
    return "{}";
  }

  using EventCreateFlags = Flags<EventCreateFlagBits>;

  template <>
  struct FlagTraits<EventCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( EventCreateFlagBits::eDeviceOnlyKHR )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR EventCreateFlags operator|( EventCreateFlagBits bit0,
                                                                     EventCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return EventCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR EventCreateFlags operator&(EventCreateFlagBits bit0,
                                                                    EventCreateFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return EventCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR EventCreateFlags operator^( EventCreateFlagBits bit0,
                                                                     EventCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return EventCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR EventCreateFlags operator~( EventCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( EventCreateFlags( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( EventCreateFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & EventCreateFlagBits::eDeviceOnlyKHR )
      result += "DeviceOnlyKHR | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using QueryPipelineStatisticFlags = Flags<QueryPipelineStatisticFlagBits>;

  template <>
  struct FlagTraits<QueryPipelineStatisticFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( QueryPipelineStatisticFlagBits::eInputAssemblyVertices ) |
                 VkFlags( QueryPipelineStatisticFlagBits::eInputAssemblyPrimitives ) |
                 VkFlags( QueryPipelineStatisticFlagBits::eVertexShaderInvocations ) |
                 VkFlags( QueryPipelineStatisticFlagBits::eGeometryShaderInvocations ) |
                 VkFlags( QueryPipelineStatisticFlagBits::eGeometryShaderPrimitives ) |
                 VkFlags( QueryPipelineStatisticFlagBits::eClippingInvocations ) |
                 VkFlags( QueryPipelineStatisticFlagBits::eClippingPrimitives ) |
                 VkFlags( QueryPipelineStatisticFlagBits::eFragmentShaderInvocations ) |
                 VkFlags( QueryPipelineStatisticFlagBits::eTessellationControlShaderPatches ) |
                 VkFlags( QueryPipelineStatisticFlagBits::eTessellationEvaluationShaderInvocations ) |
                 VkFlags( QueryPipelineStatisticFlagBits::eComputeShaderInvocations )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueryPipelineStatisticFlags
                                         operator|( QueryPipelineStatisticFlagBits bit0, QueryPipelineStatisticFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return QueryPipelineStatisticFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueryPipelineStatisticFlags
                                         operator&(QueryPipelineStatisticFlagBits bit0, QueryPipelineStatisticFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return QueryPipelineStatisticFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueryPipelineStatisticFlags
                                         operator^( QueryPipelineStatisticFlagBits bit0, QueryPipelineStatisticFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return QueryPipelineStatisticFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueryPipelineStatisticFlags operator~( QueryPipelineStatisticFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( QueryPipelineStatisticFlags( bits ) );
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
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using QueryPoolCreateFlags = Flags<QueryPoolCreateFlagBits>;

  VULKAN_HPP_INLINE std::string to_string( QueryPoolCreateFlags )
  {
    return "{}";
  }

  using QueryResultFlags = Flags<QueryResultFlagBits>;

  template <>
  struct FlagTraits<QueryResultFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( QueryResultFlagBits::e64 ) | VkFlags( QueryResultFlagBits::eWait ) |
                 VkFlags( QueryResultFlagBits::eWithAvailability ) | VkFlags( QueryResultFlagBits::ePartial )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
                 | VkFlags( QueryResultFlagBits::eWithStatusKHR )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueryResultFlags operator|( QueryResultFlagBits bit0,
                                                                     QueryResultFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return QueryResultFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueryResultFlags operator&(QueryResultFlagBits bit0,
                                                                    QueryResultFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return QueryResultFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueryResultFlags operator^( QueryResultFlagBits bit0,
                                                                     QueryResultFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return QueryResultFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueryResultFlags operator~( QueryResultFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( QueryResultFlags( bits ) );
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
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & QueryResultFlagBits::eWithStatusKHR )
      result += "WithStatusKHR | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using BufferCreateFlags = Flags<BufferCreateFlagBits>;

  template <>
  struct FlagTraits<BufferCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( BufferCreateFlagBits::eSparseBinding ) | VkFlags( BufferCreateFlagBits::eSparseResidency ) |
                 VkFlags( BufferCreateFlagBits::eSparseAliased ) | VkFlags( BufferCreateFlagBits::eProtected ) |
                 VkFlags( BufferCreateFlagBits::eDeviceAddressCaptureReplay )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BufferCreateFlags operator|( BufferCreateFlagBits bit0,
                                                                      BufferCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return BufferCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BufferCreateFlags operator&(BufferCreateFlagBits bit0,
                                                                     BufferCreateFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return BufferCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BufferCreateFlags operator^( BufferCreateFlagBits bit0,
                                                                      BufferCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return BufferCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BufferCreateFlags operator~( BufferCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( BufferCreateFlags( bits ) );
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
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using BufferUsageFlags = Flags<BufferUsageFlagBits>;

  template <>
  struct FlagTraits<BufferUsageFlagBits>
  {
    enum : VkFlags
    {
      allFlags =
        VkFlags( BufferUsageFlagBits::eTransferSrc ) | VkFlags( BufferUsageFlagBits::eTransferDst ) |
        VkFlags( BufferUsageFlagBits::eUniformTexelBuffer ) | VkFlags( BufferUsageFlagBits::eStorageTexelBuffer ) |
        VkFlags( BufferUsageFlagBits::eUniformBuffer ) | VkFlags( BufferUsageFlagBits::eStorageBuffer ) |
        VkFlags( BufferUsageFlagBits::eIndexBuffer ) | VkFlags( BufferUsageFlagBits::eVertexBuffer ) |
        VkFlags( BufferUsageFlagBits::eIndirectBuffer ) | VkFlags( BufferUsageFlagBits::eShaderDeviceAddress )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
        | VkFlags( BufferUsageFlagBits::eVideoDecodeSrcKHR ) | VkFlags( BufferUsageFlagBits::eVideoDecodeDstKHR )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
        | VkFlags( BufferUsageFlagBits::eTransformFeedbackBufferEXT ) |
        VkFlags( BufferUsageFlagBits::eTransformFeedbackCounterBufferEXT ) |
        VkFlags( BufferUsageFlagBits::eConditionalRenderingEXT ) |
        VkFlags( BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR ) |
        VkFlags( BufferUsageFlagBits::eAccelerationStructureStorageKHR ) |
        VkFlags( BufferUsageFlagBits::eShaderBindingTableKHR )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
        | VkFlags( BufferUsageFlagBits::eVideoEncodeDstKHR ) | VkFlags( BufferUsageFlagBits::eVideoEncodeSrcKHR )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BufferUsageFlags operator|( BufferUsageFlagBits bit0,
                                                                     BufferUsageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return BufferUsageFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BufferUsageFlags operator&(BufferUsageFlagBits bit0,
                                                                    BufferUsageFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return BufferUsageFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BufferUsageFlags operator^( BufferUsageFlagBits bit0,
                                                                     BufferUsageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return BufferUsageFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BufferUsageFlags operator~( BufferUsageFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( BufferUsageFlags( bits ) );
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
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & BufferUsageFlagBits::eVideoDecodeSrcKHR )
      result += "VideoDecodeSrcKHR | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & BufferUsageFlagBits::eVideoDecodeDstKHR )
      result += "VideoDecodeDstKHR | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    if ( value & BufferUsageFlagBits::eTransformFeedbackBufferEXT )
      result += "TransformFeedbackBufferEXT | ";
    if ( value & BufferUsageFlagBits::eTransformFeedbackCounterBufferEXT )
      result += "TransformFeedbackCounterBufferEXT | ";
    if ( value & BufferUsageFlagBits::eConditionalRenderingEXT )
      result += "ConditionalRenderingEXT | ";
    if ( value & BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR )
      result += "AccelerationStructureBuildInputReadOnlyKHR | ";
    if ( value & BufferUsageFlagBits::eAccelerationStructureStorageKHR )
      result += "AccelerationStructureStorageKHR | ";
    if ( value & BufferUsageFlagBits::eShaderBindingTableKHR )
      result += "ShaderBindingTableKHR | ";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & BufferUsageFlagBits::eVideoEncodeDstKHR )
      result += "VideoEncodeDstKHR | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & BufferUsageFlagBits::eVideoEncodeSrcKHR )
      result += "VideoEncodeSrcKHR | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using BufferViewCreateFlags = Flags<BufferViewCreateFlagBits>;

  VULKAN_HPP_INLINE std::string to_string( BufferViewCreateFlags )
  {
    return "{}";
  }

  using ImageViewCreateFlags = Flags<ImageViewCreateFlagBits>;

  template <>
  struct FlagTraits<ImageViewCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ImageViewCreateFlagBits::eFragmentDensityMapDynamicEXT ) |
                 VkFlags( ImageViewCreateFlagBits::eFragmentDensityMapDeferredEXT )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageViewCreateFlags
                                         operator|( ImageViewCreateFlagBits bit0, ImageViewCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageViewCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageViewCreateFlags operator&(ImageViewCreateFlagBits bit0,
                                                                        ImageViewCreateFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return ImageViewCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageViewCreateFlags
                                         operator^( ImageViewCreateFlagBits bit0, ImageViewCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageViewCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageViewCreateFlags operator~( ImageViewCreateFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( ImageViewCreateFlags( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( ImageViewCreateFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & ImageViewCreateFlagBits::eFragmentDensityMapDynamicEXT )
      result += "FragmentDensityMapDynamicEXT | ";
    if ( value & ImageViewCreateFlagBits::eFragmentDensityMapDeferredEXT )
      result += "FragmentDensityMapDeferredEXT | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using ShaderModuleCreateFlags = Flags<ShaderModuleCreateFlagBits>;

  VULKAN_HPP_INLINE std::string to_string( ShaderModuleCreateFlags )
  {
    return "{}";
  }

  using PipelineCacheCreateFlags = Flags<PipelineCacheCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineCacheCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( PipelineCacheCreateFlagBits::eExternallySynchronizedEXT )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineCacheCreateFlags
                                         operator|( PipelineCacheCreateFlagBits bit0, PipelineCacheCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineCacheCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineCacheCreateFlags
                                         operator&(PipelineCacheCreateFlagBits bit0, PipelineCacheCreateFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return PipelineCacheCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineCacheCreateFlags
                                         operator^( PipelineCacheCreateFlagBits bit0, PipelineCacheCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineCacheCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineCacheCreateFlags operator~( PipelineCacheCreateFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( PipelineCacheCreateFlags( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineCacheCreateFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & PipelineCacheCreateFlagBits::eExternallySynchronizedEXT )
      result += "ExternallySynchronizedEXT | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using ColorComponentFlags = Flags<ColorComponentFlagBits>;

  template <>
  struct FlagTraits<ColorComponentFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ColorComponentFlagBits::eR ) | VkFlags( ColorComponentFlagBits::eG ) |
                 VkFlags( ColorComponentFlagBits::eB ) | VkFlags( ColorComponentFlagBits::eA )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ColorComponentFlags
                                         operator|( ColorComponentFlagBits bit0, ColorComponentFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ColorComponentFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ColorComponentFlags operator&(ColorComponentFlagBits bit0,
                                                                       ColorComponentFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return ColorComponentFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ColorComponentFlags
                                         operator^( ColorComponentFlagBits bit0, ColorComponentFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ColorComponentFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ColorComponentFlags operator~( ColorComponentFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( ColorComponentFlags( bits ) );
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

  using CullModeFlags = Flags<CullModeFlagBits>;

  template <>
  struct FlagTraits<CullModeFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( CullModeFlagBits::eNone ) | VkFlags( CullModeFlagBits::eFront ) |
                 VkFlags( CullModeFlagBits::eBack ) | VkFlags( CullModeFlagBits::eFrontAndBack )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CullModeFlags operator|( CullModeFlagBits bit0,
                                                                  CullModeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CullModeFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CullModeFlags operator&(CullModeFlagBits bit0,
                                                                 CullModeFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return CullModeFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CullModeFlags operator^( CullModeFlagBits bit0,
                                                                  CullModeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CullModeFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CullModeFlags operator~( CullModeFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( CullModeFlags( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( CullModeFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & CullModeFlagBits::eFront )
      result += "Front | ";
    if ( value & CullModeFlagBits::eBack )
      result += "Back | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using PipelineColorBlendStateCreateFlags = Flags<PipelineColorBlendStateCreateFlagBits>;

  VULKAN_HPP_INLINE std::string to_string( PipelineColorBlendStateCreateFlags )
  {
    return "{}";
  }

  using PipelineCreateFlags = Flags<PipelineCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags =
        VkFlags( PipelineCreateFlagBits::eDisableOptimization ) | VkFlags( PipelineCreateFlagBits::eAllowDerivatives ) |
        VkFlags( PipelineCreateFlagBits::eDerivative ) | VkFlags( PipelineCreateFlagBits::eViewIndexFromDeviceIndex ) |
        VkFlags( PipelineCreateFlagBits::eDispatchBase ) |
        VkFlags( PipelineCreateFlagBits::eRayTracingNoNullAnyHitShadersKHR ) |
        VkFlags( PipelineCreateFlagBits::eRayTracingNoNullClosestHitShadersKHR ) |
        VkFlags( PipelineCreateFlagBits::eRayTracingNoNullMissShadersKHR ) |
        VkFlags( PipelineCreateFlagBits::eRayTracingNoNullIntersectionShadersKHR ) |
        VkFlags( PipelineCreateFlagBits::eRayTracingSkipTrianglesKHR ) |
        VkFlags( PipelineCreateFlagBits::eRayTracingSkipAabbsKHR ) |
        VkFlags( PipelineCreateFlagBits::eRayTracingShaderGroupHandleCaptureReplayKHR ) |
        VkFlags( PipelineCreateFlagBits::eDeferCompileNV ) | VkFlags( PipelineCreateFlagBits::eCaptureStatisticsKHR ) |
        VkFlags( PipelineCreateFlagBits::eCaptureInternalRepresentationsKHR ) |
        VkFlags( PipelineCreateFlagBits::eIndirectBindableNV ) | VkFlags( PipelineCreateFlagBits::eLibraryKHR ) |
        VkFlags( PipelineCreateFlagBits::eFailOnPipelineCompileRequiredEXT ) |
        VkFlags( PipelineCreateFlagBits::eEarlyReturnOnFailureEXT ) |
        VkFlags( PipelineCreateFlagBits::eRayTracingAllowMotionNV )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineCreateFlags
                                         operator|( PipelineCreateFlagBits bit0, PipelineCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineCreateFlags operator&(PipelineCreateFlagBits bit0,
                                                                       PipelineCreateFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return PipelineCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineCreateFlags
                                         operator^( PipelineCreateFlagBits bit0, PipelineCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineCreateFlags operator~( PipelineCreateFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( PipelineCreateFlags( bits ) );
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
    if ( value & PipelineCreateFlagBits::eFailOnPipelineCompileRequiredEXT )
      result += "FailOnPipelineCompileRequiredEXT | ";
    if ( value & PipelineCreateFlagBits::eEarlyReturnOnFailureEXT )
      result += "EarlyReturnOnFailureEXT | ";
    if ( value & PipelineCreateFlagBits::eRayTracingAllowMotionNV )
      result += "RayTracingAllowMotionNV | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using PipelineDepthStencilStateCreateFlags = Flags<PipelineDepthStencilStateCreateFlagBits>;

  VULKAN_HPP_INLINE std::string to_string( PipelineDepthStencilStateCreateFlags )
  {
    return "{}";
  }

  using PipelineDynamicStateCreateFlags = Flags<PipelineDynamicStateCreateFlagBits>;

  VULKAN_HPP_INLINE std::string to_string( PipelineDynamicStateCreateFlags )
  {
    return "{}";
  }

  using PipelineInputAssemblyStateCreateFlags = Flags<PipelineInputAssemblyStateCreateFlagBits>;

  VULKAN_HPP_INLINE std::string to_string( PipelineInputAssemblyStateCreateFlags )
  {
    return "{}";
  }

  using PipelineLayoutCreateFlags = Flags<PipelineLayoutCreateFlagBits>;

  VULKAN_HPP_INLINE std::string to_string( PipelineLayoutCreateFlags )
  {
    return "{}";
  }

  using PipelineMultisampleStateCreateFlags = Flags<PipelineMultisampleStateCreateFlagBits>;

  VULKAN_HPP_INLINE std::string to_string( PipelineMultisampleStateCreateFlags )
  {
    return "{}";
  }

  using PipelineRasterizationStateCreateFlags = Flags<PipelineRasterizationStateCreateFlagBits>;

  VULKAN_HPP_INLINE std::string to_string( PipelineRasterizationStateCreateFlags )
  {
    return "{}";
  }

  using PipelineShaderStageCreateFlags = Flags<PipelineShaderStageCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineShaderStageCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( PipelineShaderStageCreateFlagBits::eAllowVaryingSubgroupSizeEXT ) |
                 VkFlags( PipelineShaderStageCreateFlagBits::eRequireFullSubgroupsEXT )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineShaderStageCreateFlags
                                         operator|( PipelineShaderStageCreateFlagBits bit0, PipelineShaderStageCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineShaderStageCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineShaderStageCreateFlags
                                         operator&(PipelineShaderStageCreateFlagBits bit0, PipelineShaderStageCreateFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return PipelineShaderStageCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineShaderStageCreateFlags
                                         operator^( PipelineShaderStageCreateFlagBits bit0, PipelineShaderStageCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineShaderStageCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineShaderStageCreateFlags
                                         operator~( PipelineShaderStageCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( PipelineShaderStageCreateFlags( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineShaderStageCreateFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & PipelineShaderStageCreateFlagBits::eAllowVaryingSubgroupSizeEXT )
      result += "AllowVaryingSubgroupSizeEXT | ";
    if ( value & PipelineShaderStageCreateFlagBits::eRequireFullSubgroupsEXT )
      result += "RequireFullSubgroupsEXT | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using PipelineTessellationStateCreateFlags = Flags<PipelineTessellationStateCreateFlagBits>;

  VULKAN_HPP_INLINE std::string to_string( PipelineTessellationStateCreateFlags )
  {
    return "{}";
  }

  using PipelineVertexInputStateCreateFlags = Flags<PipelineVertexInputStateCreateFlagBits>;

  VULKAN_HPP_INLINE std::string to_string( PipelineVertexInputStateCreateFlags )
  {
    return "{}";
  }

  using PipelineViewportStateCreateFlags = Flags<PipelineViewportStateCreateFlagBits>;

  VULKAN_HPP_INLINE std::string to_string( PipelineViewportStateCreateFlags )
  {
    return "{}";
  }

  using ShaderStageFlags = Flags<ShaderStageFlagBits>;

  template <>
  struct FlagTraits<ShaderStageFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ShaderStageFlagBits::eVertex ) | VkFlags( ShaderStageFlagBits::eTessellationControl ) |
                 VkFlags( ShaderStageFlagBits::eTessellationEvaluation ) | VkFlags( ShaderStageFlagBits::eGeometry ) |
                 VkFlags( ShaderStageFlagBits::eFragment ) | VkFlags( ShaderStageFlagBits::eCompute ) |
                 VkFlags( ShaderStageFlagBits::eAllGraphics ) | VkFlags( ShaderStageFlagBits::eAll ) |
                 VkFlags( ShaderStageFlagBits::eRaygenKHR ) | VkFlags( ShaderStageFlagBits::eAnyHitKHR ) |
                 VkFlags( ShaderStageFlagBits::eClosestHitKHR ) | VkFlags( ShaderStageFlagBits::eMissKHR ) |
                 VkFlags( ShaderStageFlagBits::eIntersectionKHR ) | VkFlags( ShaderStageFlagBits::eCallableKHR ) |
                 VkFlags( ShaderStageFlagBits::eTaskNV ) | VkFlags( ShaderStageFlagBits::eMeshNV ) |
                 VkFlags( ShaderStageFlagBits::eSubpassShadingHUAWEI )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ShaderStageFlags operator|( ShaderStageFlagBits bit0,
                                                                     ShaderStageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ShaderStageFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ShaderStageFlags operator&(ShaderStageFlagBits bit0,
                                                                    ShaderStageFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return ShaderStageFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ShaderStageFlags operator^( ShaderStageFlagBits bit0,
                                                                     ShaderStageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ShaderStageFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ShaderStageFlags operator~( ShaderStageFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ShaderStageFlags( bits ) );
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
    if ( value & ShaderStageFlagBits::eTaskNV )
      result += "TaskNV | ";
    if ( value & ShaderStageFlagBits::eMeshNV )
      result += "MeshNV | ";
    if ( value & ShaderStageFlagBits::eSubpassShadingHUAWEI )
      result += "SubpassShadingHUAWEI | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using SamplerCreateFlags = Flags<SamplerCreateFlagBits>;

  template <>
  struct FlagTraits<SamplerCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( SamplerCreateFlagBits::eSubsampledEXT ) |
                 VkFlags( SamplerCreateFlagBits::eSubsampledCoarseReconstructionEXT )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SamplerCreateFlags operator|( SamplerCreateFlagBits bit0,
                                                                       SamplerCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SamplerCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SamplerCreateFlags operator&(SamplerCreateFlagBits bit0,
                                                                      SamplerCreateFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return SamplerCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SamplerCreateFlags operator^( SamplerCreateFlagBits bit0,
                                                                       SamplerCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SamplerCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SamplerCreateFlags operator~( SamplerCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( SamplerCreateFlags( bits ) );
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
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using DescriptorPoolCreateFlags = Flags<DescriptorPoolCreateFlagBits>;

  template <>
  struct FlagTraits<DescriptorPoolCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( DescriptorPoolCreateFlagBits::eFreeDescriptorSet ) |
                 VkFlags( DescriptorPoolCreateFlagBits::eUpdateAfterBind ) |
                 VkFlags( DescriptorPoolCreateFlagBits::eHostOnlyVALVE )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DescriptorPoolCreateFlags
                                         operator|( DescriptorPoolCreateFlagBits bit0, DescriptorPoolCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DescriptorPoolCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DescriptorPoolCreateFlags
                                         operator&(DescriptorPoolCreateFlagBits bit0, DescriptorPoolCreateFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return DescriptorPoolCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DescriptorPoolCreateFlags
                                         operator^( DescriptorPoolCreateFlagBits bit0, DescriptorPoolCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DescriptorPoolCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DescriptorPoolCreateFlags operator~( DescriptorPoolCreateFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( DescriptorPoolCreateFlags( bits ) );
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
    if ( value & DescriptorPoolCreateFlagBits::eHostOnlyVALVE )
      result += "HostOnlyVALVE | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using DescriptorPoolResetFlags = Flags<DescriptorPoolResetFlagBits>;

  VULKAN_HPP_INLINE std::string to_string( DescriptorPoolResetFlags )
  {
    return "{}";
  }

  using DescriptorSetLayoutCreateFlags = Flags<DescriptorSetLayoutCreateFlagBits>;

  template <>
  struct FlagTraits<DescriptorSetLayoutCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool ) |
                 VkFlags( DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR ) |
                 VkFlags( DescriptorSetLayoutCreateFlagBits::eHostOnlyPoolVALVE )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DescriptorSetLayoutCreateFlags
                                         operator|( DescriptorSetLayoutCreateFlagBits bit0, DescriptorSetLayoutCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DescriptorSetLayoutCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DescriptorSetLayoutCreateFlags
                                         operator&(DescriptorSetLayoutCreateFlagBits bit0, DescriptorSetLayoutCreateFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return DescriptorSetLayoutCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DescriptorSetLayoutCreateFlags
                                         operator^( DescriptorSetLayoutCreateFlagBits bit0, DescriptorSetLayoutCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DescriptorSetLayoutCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DescriptorSetLayoutCreateFlags
                                         operator~( DescriptorSetLayoutCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( DescriptorSetLayoutCreateFlags( bits ) );
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
    if ( value & DescriptorSetLayoutCreateFlagBits::eHostOnlyPoolVALVE )
      result += "HostOnlyPoolVALVE | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using AccessFlags = Flags<AccessFlagBits>;

  template <>
  struct FlagTraits<AccessFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( AccessFlagBits::eIndirectCommandRead ) | VkFlags( AccessFlagBits::eIndexRead ) |
                 VkFlags( AccessFlagBits::eVertexAttributeRead ) | VkFlags( AccessFlagBits::eUniformRead ) |
                 VkFlags( AccessFlagBits::eInputAttachmentRead ) | VkFlags( AccessFlagBits::eShaderRead ) |
                 VkFlags( AccessFlagBits::eShaderWrite ) | VkFlags( AccessFlagBits::eColorAttachmentRead ) |
                 VkFlags( AccessFlagBits::eColorAttachmentWrite ) |
                 VkFlags( AccessFlagBits::eDepthStencilAttachmentRead ) |
                 VkFlags( AccessFlagBits::eDepthStencilAttachmentWrite ) | VkFlags( AccessFlagBits::eTransferRead ) |
                 VkFlags( AccessFlagBits::eTransferWrite ) | VkFlags( AccessFlagBits::eHostRead ) |
                 VkFlags( AccessFlagBits::eHostWrite ) | VkFlags( AccessFlagBits::eMemoryRead ) |
                 VkFlags( AccessFlagBits::eMemoryWrite ) | VkFlags( AccessFlagBits::eTransformFeedbackWriteEXT ) |
                 VkFlags( AccessFlagBits::eTransformFeedbackCounterReadEXT ) |
                 VkFlags( AccessFlagBits::eTransformFeedbackCounterWriteEXT ) |
                 VkFlags( AccessFlagBits::eConditionalRenderingReadEXT ) |
                 VkFlags( AccessFlagBits::eColorAttachmentReadNoncoherentEXT ) |
                 VkFlags( AccessFlagBits::eAccelerationStructureReadKHR ) |
                 VkFlags( AccessFlagBits::eAccelerationStructureWriteKHR ) |
                 VkFlags( AccessFlagBits::eFragmentDensityMapReadEXT ) |
                 VkFlags( AccessFlagBits::eFragmentShadingRateAttachmentReadKHR ) |
                 VkFlags( AccessFlagBits::eCommandPreprocessReadNV ) |
                 VkFlags( AccessFlagBits::eCommandPreprocessWriteNV ) | VkFlags( AccessFlagBits::eNoneKHR )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AccessFlags operator|( AccessFlagBits bit0,
                                                                AccessFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return AccessFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AccessFlags operator&(AccessFlagBits bit0,
                                                               AccessFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return AccessFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AccessFlags operator^( AccessFlagBits bit0,
                                                                AccessFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return AccessFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AccessFlags operator~( AccessFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( AccessFlags( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( AccessFlags value )
  {
    if ( !value )
      return "{}";

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

  using AttachmentDescriptionFlags = Flags<AttachmentDescriptionFlagBits>;

  template <>
  struct FlagTraits<AttachmentDescriptionFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( AttachmentDescriptionFlagBits::eMayAlias )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AttachmentDescriptionFlags
                                         operator|( AttachmentDescriptionFlagBits bit0, AttachmentDescriptionFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return AttachmentDescriptionFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AttachmentDescriptionFlags
                                         operator&(AttachmentDescriptionFlagBits bit0, AttachmentDescriptionFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return AttachmentDescriptionFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AttachmentDescriptionFlags
                                         operator^( AttachmentDescriptionFlagBits bit0, AttachmentDescriptionFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return AttachmentDescriptionFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AttachmentDescriptionFlags operator~( AttachmentDescriptionFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( AttachmentDescriptionFlags( bits ) );
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

  using DependencyFlags = Flags<DependencyFlagBits>;

  template <>
  struct FlagTraits<DependencyFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( DependencyFlagBits::eByRegion ) | VkFlags( DependencyFlagBits::eDeviceGroup ) |
                 VkFlags( DependencyFlagBits::eViewLocal )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DependencyFlags operator|( DependencyFlagBits bit0,
                                                                    DependencyFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DependencyFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DependencyFlags operator&(DependencyFlagBits bit0,
                                                                   DependencyFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return DependencyFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DependencyFlags operator^( DependencyFlagBits bit0,
                                                                    DependencyFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DependencyFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DependencyFlags operator~( DependencyFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( DependencyFlags( bits ) );
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
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using FramebufferCreateFlags = Flags<FramebufferCreateFlagBits>;

  template <>
  struct FlagTraits<FramebufferCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( FramebufferCreateFlagBits::eImageless )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FramebufferCreateFlags
                                         operator|( FramebufferCreateFlagBits bit0, FramebufferCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return FramebufferCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FramebufferCreateFlags
                                         operator&(FramebufferCreateFlagBits bit0, FramebufferCreateFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return FramebufferCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FramebufferCreateFlags
                                         operator^( FramebufferCreateFlagBits bit0, FramebufferCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return FramebufferCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FramebufferCreateFlags operator~( FramebufferCreateFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( FramebufferCreateFlags( bits ) );
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

  using RenderPassCreateFlags = Flags<RenderPassCreateFlagBits>;

  template <>
  struct FlagTraits<RenderPassCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( RenderPassCreateFlagBits::eTransformQCOM )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR RenderPassCreateFlags
                                         operator|( RenderPassCreateFlagBits bit0, RenderPassCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return RenderPassCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR RenderPassCreateFlags
                                         operator&(RenderPassCreateFlagBits bit0, RenderPassCreateFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return RenderPassCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR RenderPassCreateFlags
                                         operator^( RenderPassCreateFlagBits bit0, RenderPassCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return RenderPassCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR RenderPassCreateFlags operator~( RenderPassCreateFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( RenderPassCreateFlags( bits ) );
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

  using SubpassDescriptionFlags = Flags<SubpassDescriptionFlagBits>;

  template <>
  struct FlagTraits<SubpassDescriptionFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( SubpassDescriptionFlagBits::ePerViewAttributesNVX ) |
                 VkFlags( SubpassDescriptionFlagBits::ePerViewPositionXOnlyNVX ) |
                 VkFlags( SubpassDescriptionFlagBits::eFragmentRegionQCOM ) |
                 VkFlags( SubpassDescriptionFlagBits::eShaderResolveQCOM )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SubpassDescriptionFlags
                                         operator|( SubpassDescriptionFlagBits bit0, SubpassDescriptionFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SubpassDescriptionFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SubpassDescriptionFlags
                                         operator&(SubpassDescriptionFlagBits bit0, SubpassDescriptionFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return SubpassDescriptionFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SubpassDescriptionFlags
                                         operator^( SubpassDescriptionFlagBits bit0, SubpassDescriptionFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SubpassDescriptionFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SubpassDescriptionFlags operator~( SubpassDescriptionFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( SubpassDescriptionFlags( bits ) );
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
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using CommandPoolCreateFlags = Flags<CommandPoolCreateFlagBits>;

  template <>
  struct FlagTraits<CommandPoolCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( CommandPoolCreateFlagBits::eTransient ) |
                 VkFlags( CommandPoolCreateFlagBits::eResetCommandBuffer ) |
                 VkFlags( CommandPoolCreateFlagBits::eProtected )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandPoolCreateFlags
                                         operator|( CommandPoolCreateFlagBits bit0, CommandPoolCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CommandPoolCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandPoolCreateFlags
                                         operator&(CommandPoolCreateFlagBits bit0, CommandPoolCreateFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return CommandPoolCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandPoolCreateFlags
                                         operator^( CommandPoolCreateFlagBits bit0, CommandPoolCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CommandPoolCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandPoolCreateFlags operator~( CommandPoolCreateFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( CommandPoolCreateFlags( bits ) );
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

  using CommandPoolResetFlags = Flags<CommandPoolResetFlagBits>;

  template <>
  struct FlagTraits<CommandPoolResetFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( CommandPoolResetFlagBits::eReleaseResources )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandPoolResetFlags
                                         operator|( CommandPoolResetFlagBits bit0, CommandPoolResetFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CommandPoolResetFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandPoolResetFlags
                                         operator&(CommandPoolResetFlagBits bit0, CommandPoolResetFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return CommandPoolResetFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandPoolResetFlags
                                         operator^( CommandPoolResetFlagBits bit0, CommandPoolResetFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CommandPoolResetFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandPoolResetFlags operator~( CommandPoolResetFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( CommandPoolResetFlags( bits ) );
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

  using CommandBufferResetFlags = Flags<CommandBufferResetFlagBits>;

  template <>
  struct FlagTraits<CommandBufferResetFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( CommandBufferResetFlagBits::eReleaseResources )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandBufferResetFlags
                                         operator|( CommandBufferResetFlagBits bit0, CommandBufferResetFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CommandBufferResetFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandBufferResetFlags
                                         operator&(CommandBufferResetFlagBits bit0, CommandBufferResetFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return CommandBufferResetFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandBufferResetFlags
                                         operator^( CommandBufferResetFlagBits bit0, CommandBufferResetFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CommandBufferResetFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandBufferResetFlags operator~( CommandBufferResetFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( CommandBufferResetFlags( bits ) );
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

  using CommandBufferUsageFlags = Flags<CommandBufferUsageFlagBits>;

  template <>
  struct FlagTraits<CommandBufferUsageFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( CommandBufferUsageFlagBits::eOneTimeSubmit ) |
                 VkFlags( CommandBufferUsageFlagBits::eRenderPassContinue ) |
                 VkFlags( CommandBufferUsageFlagBits::eSimultaneousUse )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandBufferUsageFlags
                                         operator|( CommandBufferUsageFlagBits bit0, CommandBufferUsageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CommandBufferUsageFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandBufferUsageFlags
                                         operator&(CommandBufferUsageFlagBits bit0, CommandBufferUsageFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return CommandBufferUsageFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandBufferUsageFlags
                                         operator^( CommandBufferUsageFlagBits bit0, CommandBufferUsageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CommandBufferUsageFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandBufferUsageFlags operator~( CommandBufferUsageFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( CommandBufferUsageFlags( bits ) );
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

  using QueryControlFlags = Flags<QueryControlFlagBits>;

  template <>
  struct FlagTraits<QueryControlFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( QueryControlFlagBits::ePrecise )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueryControlFlags operator|( QueryControlFlagBits bit0,
                                                                      QueryControlFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return QueryControlFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueryControlFlags operator&(QueryControlFlagBits bit0,
                                                                     QueryControlFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return QueryControlFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueryControlFlags operator^( QueryControlFlagBits bit0,
                                                                      QueryControlFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return QueryControlFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueryControlFlags operator~( QueryControlFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( QueryControlFlags( bits ) );
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

  using StencilFaceFlags = Flags<StencilFaceFlagBits>;

  template <>
  struct FlagTraits<StencilFaceFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( StencilFaceFlagBits::eFront ) | VkFlags( StencilFaceFlagBits::eBack ) |
                 VkFlags( StencilFaceFlagBits::eFrontAndBack )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR StencilFaceFlags operator|( StencilFaceFlagBits bit0,
                                                                     StencilFaceFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return StencilFaceFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR StencilFaceFlags operator&(StencilFaceFlagBits bit0,
                                                                    StencilFaceFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return StencilFaceFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR StencilFaceFlags operator^( StencilFaceFlagBits bit0,
                                                                     StencilFaceFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return StencilFaceFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR StencilFaceFlags operator~( StencilFaceFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( StencilFaceFlags( bits ) );
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

  using SubgroupFeatureFlags = Flags<SubgroupFeatureFlagBits>;

  template <>
  struct FlagTraits<SubgroupFeatureFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( SubgroupFeatureFlagBits::eBasic ) | VkFlags( SubgroupFeatureFlagBits::eVote ) |
                 VkFlags( SubgroupFeatureFlagBits::eArithmetic ) | VkFlags( SubgroupFeatureFlagBits::eBallot ) |
                 VkFlags( SubgroupFeatureFlagBits::eShuffle ) | VkFlags( SubgroupFeatureFlagBits::eShuffleRelative ) |
                 VkFlags( SubgroupFeatureFlagBits::eClustered ) | VkFlags( SubgroupFeatureFlagBits::eQuad ) |
                 VkFlags( SubgroupFeatureFlagBits::ePartitionedNV )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SubgroupFeatureFlags
                                         operator|( SubgroupFeatureFlagBits bit0, SubgroupFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SubgroupFeatureFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SubgroupFeatureFlags operator&(SubgroupFeatureFlagBits bit0,
                                                                        SubgroupFeatureFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return SubgroupFeatureFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SubgroupFeatureFlags
                                         operator^( SubgroupFeatureFlagBits bit0, SubgroupFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SubgroupFeatureFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SubgroupFeatureFlags operator~( SubgroupFeatureFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( SubgroupFeatureFlags( bits ) );
  }

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
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using PeerMemoryFeatureFlags = Flags<PeerMemoryFeatureFlagBits>;

  template <>
  struct FlagTraits<PeerMemoryFeatureFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( PeerMemoryFeatureFlagBits::eCopySrc ) | VkFlags( PeerMemoryFeatureFlagBits::eCopyDst ) |
                 VkFlags( PeerMemoryFeatureFlagBits::eGenericSrc ) | VkFlags( PeerMemoryFeatureFlagBits::eGenericDst )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PeerMemoryFeatureFlags
                                         operator|( PeerMemoryFeatureFlagBits bit0, PeerMemoryFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PeerMemoryFeatureFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PeerMemoryFeatureFlags
                                         operator&(PeerMemoryFeatureFlagBits bit0, PeerMemoryFeatureFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return PeerMemoryFeatureFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PeerMemoryFeatureFlags
                                         operator^( PeerMemoryFeatureFlagBits bit0, PeerMemoryFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PeerMemoryFeatureFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PeerMemoryFeatureFlags operator~( PeerMemoryFeatureFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( PeerMemoryFeatureFlags( bits ) );
  }

  using PeerMemoryFeatureFlagsKHR = PeerMemoryFeatureFlags;

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

  using MemoryAllocateFlags = Flags<MemoryAllocateFlagBits>;

  template <>
  struct FlagTraits<MemoryAllocateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( MemoryAllocateFlagBits::eDeviceMask ) | VkFlags( MemoryAllocateFlagBits::eDeviceAddress ) |
                 VkFlags( MemoryAllocateFlagBits::eDeviceAddressCaptureReplay )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MemoryAllocateFlags
                                         operator|( MemoryAllocateFlagBits bit0, MemoryAllocateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return MemoryAllocateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MemoryAllocateFlags operator&(MemoryAllocateFlagBits bit0,
                                                                       MemoryAllocateFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return MemoryAllocateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MemoryAllocateFlags
                                         operator^( MemoryAllocateFlagBits bit0, MemoryAllocateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return MemoryAllocateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MemoryAllocateFlags operator~( MemoryAllocateFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( MemoryAllocateFlags( bits ) );
  }

  using MemoryAllocateFlagsKHR = MemoryAllocateFlags;

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

  using CommandPoolTrimFlags = Flags<CommandPoolTrimFlagBits>;

  using CommandPoolTrimFlagsKHR = CommandPoolTrimFlags;

  VULKAN_HPP_INLINE std::string to_string( CommandPoolTrimFlags )
  {
    return "{}";
  }

  using DescriptorUpdateTemplateCreateFlags = Flags<DescriptorUpdateTemplateCreateFlagBits>;

  using DescriptorUpdateTemplateCreateFlagsKHR = DescriptorUpdateTemplateCreateFlags;

  VULKAN_HPP_INLINE std::string to_string( DescriptorUpdateTemplateCreateFlags )
  {
    return "{}";
  }

  using ExternalMemoryHandleTypeFlags = Flags<ExternalMemoryHandleTypeFlagBits>;

  template <>
  struct FlagTraits<ExternalMemoryHandleTypeFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ExternalMemoryHandleTypeFlagBits::eOpaqueFd ) |
                 VkFlags( ExternalMemoryHandleTypeFlagBits::eOpaqueWin32 ) |
                 VkFlags( ExternalMemoryHandleTypeFlagBits::eOpaqueWin32Kmt ) |
                 VkFlags( ExternalMemoryHandleTypeFlagBits::eD3D11Texture ) |
                 VkFlags( ExternalMemoryHandleTypeFlagBits::eD3D11TextureKmt ) |
                 VkFlags( ExternalMemoryHandleTypeFlagBits::eD3D12Heap ) |
                 VkFlags( ExternalMemoryHandleTypeFlagBits::eD3D12Resource ) |
                 VkFlags( ExternalMemoryHandleTypeFlagBits::eDmaBufEXT )
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
                 | VkFlags( ExternalMemoryHandleTypeFlagBits::eAndroidHardwareBufferANDROID )
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
                 | VkFlags( ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT ) |
                 VkFlags( ExternalMemoryHandleTypeFlagBits::eHostMappedForeignMemoryEXT )
#if defined( VK_USE_PLATFORM_FUCHSIA )
                 | VkFlags( ExternalMemoryHandleTypeFlagBits::eZirconVmoFUCHSIA )
#endif /*VK_USE_PLATFORM_FUCHSIA*/
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryHandleTypeFlags
                                         operator|( ExternalMemoryHandleTypeFlagBits bit0, ExternalMemoryHandleTypeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalMemoryHandleTypeFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryHandleTypeFlags
                                         operator&(ExternalMemoryHandleTypeFlagBits bit0, ExternalMemoryHandleTypeFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return ExternalMemoryHandleTypeFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryHandleTypeFlags
                                         operator^( ExternalMemoryHandleTypeFlagBits bit0, ExternalMemoryHandleTypeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalMemoryHandleTypeFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryHandleTypeFlags
                                         operator~( ExternalMemoryHandleTypeFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ExternalMemoryHandleTypeFlags( bits ) );
  }

  using ExternalMemoryHandleTypeFlagsKHR = ExternalMemoryHandleTypeFlags;

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
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using ExternalMemoryFeatureFlags = Flags<ExternalMemoryFeatureFlagBits>;

  template <>
  struct FlagTraits<ExternalMemoryFeatureFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ExternalMemoryFeatureFlagBits::eDedicatedOnly ) |
                 VkFlags( ExternalMemoryFeatureFlagBits::eExportable ) |
                 VkFlags( ExternalMemoryFeatureFlagBits::eImportable )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryFeatureFlags
                                         operator|( ExternalMemoryFeatureFlagBits bit0, ExternalMemoryFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalMemoryFeatureFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryFeatureFlags
                                         operator&(ExternalMemoryFeatureFlagBits bit0, ExternalMemoryFeatureFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return ExternalMemoryFeatureFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryFeatureFlags
                                         operator^( ExternalMemoryFeatureFlagBits bit0, ExternalMemoryFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalMemoryFeatureFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryFeatureFlags operator~( ExternalMemoryFeatureFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( ExternalMemoryFeatureFlags( bits ) );
  }

  using ExternalMemoryFeatureFlagsKHR = ExternalMemoryFeatureFlags;

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

  using ExternalFenceHandleTypeFlags = Flags<ExternalFenceHandleTypeFlagBits>;

  template <>
  struct FlagTraits<ExternalFenceHandleTypeFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ExternalFenceHandleTypeFlagBits::eOpaqueFd ) |
                 VkFlags( ExternalFenceHandleTypeFlagBits::eOpaqueWin32 ) |
                 VkFlags( ExternalFenceHandleTypeFlagBits::eOpaqueWin32Kmt ) |
                 VkFlags( ExternalFenceHandleTypeFlagBits::eSyncFd )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalFenceHandleTypeFlags
                                         operator|( ExternalFenceHandleTypeFlagBits bit0, ExternalFenceHandleTypeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalFenceHandleTypeFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalFenceHandleTypeFlags
                                         operator&(ExternalFenceHandleTypeFlagBits bit0, ExternalFenceHandleTypeFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return ExternalFenceHandleTypeFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalFenceHandleTypeFlags
                                         operator^( ExternalFenceHandleTypeFlagBits bit0, ExternalFenceHandleTypeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalFenceHandleTypeFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalFenceHandleTypeFlags operator~( ExternalFenceHandleTypeFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( ExternalFenceHandleTypeFlags( bits ) );
  }

  using ExternalFenceHandleTypeFlagsKHR = ExternalFenceHandleTypeFlags;

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

  using ExternalFenceFeatureFlags = Flags<ExternalFenceFeatureFlagBits>;

  template <>
  struct FlagTraits<ExternalFenceFeatureFlagBits>
  {
    enum : VkFlags
    {
      allFlags =
        VkFlags( ExternalFenceFeatureFlagBits::eExportable ) | VkFlags( ExternalFenceFeatureFlagBits::eImportable )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalFenceFeatureFlags
                                         operator|( ExternalFenceFeatureFlagBits bit0, ExternalFenceFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalFenceFeatureFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalFenceFeatureFlags
                                         operator&(ExternalFenceFeatureFlagBits bit0, ExternalFenceFeatureFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return ExternalFenceFeatureFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalFenceFeatureFlags
                                         operator^( ExternalFenceFeatureFlagBits bit0, ExternalFenceFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalFenceFeatureFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalFenceFeatureFlags operator~( ExternalFenceFeatureFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( ExternalFenceFeatureFlags( bits ) );
  }

  using ExternalFenceFeatureFlagsKHR = ExternalFenceFeatureFlags;

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

  using FenceImportFlags = Flags<FenceImportFlagBits>;

  template <>
  struct FlagTraits<FenceImportFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( FenceImportFlagBits::eTemporary )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FenceImportFlags operator|( FenceImportFlagBits bit0,
                                                                     FenceImportFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return FenceImportFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FenceImportFlags operator&(FenceImportFlagBits bit0,
                                                                    FenceImportFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return FenceImportFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FenceImportFlags operator^( FenceImportFlagBits bit0,
                                                                     FenceImportFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return FenceImportFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FenceImportFlags operator~( FenceImportFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( FenceImportFlags( bits ) );
  }

  using FenceImportFlagsKHR = FenceImportFlags;

  VULKAN_HPP_INLINE std::string to_string( FenceImportFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & FenceImportFlagBits::eTemporary )
      result += "Temporary | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using SemaphoreImportFlags = Flags<SemaphoreImportFlagBits>;

  template <>
  struct FlagTraits<SemaphoreImportFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( SemaphoreImportFlagBits::eTemporary )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SemaphoreImportFlags
                                         operator|( SemaphoreImportFlagBits bit0, SemaphoreImportFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SemaphoreImportFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SemaphoreImportFlags operator&(SemaphoreImportFlagBits bit0,
                                                                        SemaphoreImportFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return SemaphoreImportFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SemaphoreImportFlags
                                         operator^( SemaphoreImportFlagBits bit0, SemaphoreImportFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SemaphoreImportFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SemaphoreImportFlags operator~( SemaphoreImportFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( SemaphoreImportFlags( bits ) );
  }

  using SemaphoreImportFlagsKHR = SemaphoreImportFlags;

  VULKAN_HPP_INLINE std::string to_string( SemaphoreImportFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & SemaphoreImportFlagBits::eTemporary )
      result += "Temporary | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using ExternalSemaphoreHandleTypeFlags = Flags<ExternalSemaphoreHandleTypeFlagBits>;

  template <>
  struct FlagTraits<ExternalSemaphoreHandleTypeFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd ) |
                 VkFlags( ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32 ) |
                 VkFlags( ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32Kmt ) |
                 VkFlags( ExternalSemaphoreHandleTypeFlagBits::eD3D12Fence ) |
                 VkFlags( ExternalSemaphoreHandleTypeFlagBits::eSyncFd )
#if defined( VK_USE_PLATFORM_FUCHSIA )
                 | VkFlags( ExternalSemaphoreHandleTypeFlagBits::eZirconEventFUCHSIA )
#endif /*VK_USE_PLATFORM_FUCHSIA*/
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalSemaphoreHandleTypeFlags
                                         operator|( ExternalSemaphoreHandleTypeFlagBits bit0, ExternalSemaphoreHandleTypeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalSemaphoreHandleTypeFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalSemaphoreHandleTypeFlags
                                         operator&(ExternalSemaphoreHandleTypeFlagBits bit0, ExternalSemaphoreHandleTypeFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return ExternalSemaphoreHandleTypeFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalSemaphoreHandleTypeFlags
                                         operator^( ExternalSemaphoreHandleTypeFlagBits bit0, ExternalSemaphoreHandleTypeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalSemaphoreHandleTypeFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalSemaphoreHandleTypeFlags
                                         operator~( ExternalSemaphoreHandleTypeFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ExternalSemaphoreHandleTypeFlags( bits ) );
  }

  using ExternalSemaphoreHandleTypeFlagsKHR = ExternalSemaphoreHandleTypeFlags;

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

  using ExternalSemaphoreFeatureFlags = Flags<ExternalSemaphoreFeatureFlagBits>;

  template <>
  struct FlagTraits<ExternalSemaphoreFeatureFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ExternalSemaphoreFeatureFlagBits::eExportable ) |
                 VkFlags( ExternalSemaphoreFeatureFlagBits::eImportable )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalSemaphoreFeatureFlags
                                         operator|( ExternalSemaphoreFeatureFlagBits bit0, ExternalSemaphoreFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalSemaphoreFeatureFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalSemaphoreFeatureFlags
                                         operator&(ExternalSemaphoreFeatureFlagBits bit0, ExternalSemaphoreFeatureFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return ExternalSemaphoreFeatureFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalSemaphoreFeatureFlags
                                         operator^( ExternalSemaphoreFeatureFlagBits bit0, ExternalSemaphoreFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalSemaphoreFeatureFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalSemaphoreFeatureFlags
                                         operator~( ExternalSemaphoreFeatureFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ExternalSemaphoreFeatureFlags( bits ) );
  }

  using ExternalSemaphoreFeatureFlagsKHR = ExternalSemaphoreFeatureFlags;

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

  using DescriptorBindingFlags = Flags<DescriptorBindingFlagBits>;

  template <>
  struct FlagTraits<DescriptorBindingFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( DescriptorBindingFlagBits::eUpdateAfterBind ) |
                 VkFlags( DescriptorBindingFlagBits::eUpdateUnusedWhilePending ) |
                 VkFlags( DescriptorBindingFlagBits::ePartiallyBound ) |
                 VkFlags( DescriptorBindingFlagBits::eVariableDescriptorCount )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DescriptorBindingFlags
                                         operator|( DescriptorBindingFlagBits bit0, DescriptorBindingFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DescriptorBindingFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DescriptorBindingFlags
                                         operator&(DescriptorBindingFlagBits bit0, DescriptorBindingFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return DescriptorBindingFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DescriptorBindingFlags
                                         operator^( DescriptorBindingFlagBits bit0, DescriptorBindingFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DescriptorBindingFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DescriptorBindingFlags operator~( DescriptorBindingFlagBits bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( DescriptorBindingFlags( bits ) );
  }

  using DescriptorBindingFlagsEXT = DescriptorBindingFlags;

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

  using ResolveModeFlags = Flags<ResolveModeFlagBits>;

  template <>
  struct FlagTraits<ResolveModeFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ResolveModeFlagBits::eNone ) | VkFlags( ResolveModeFlagBits::eSampleZero ) |
                 VkFlags( ResolveModeFlagBits::eAverage ) | VkFlags( ResolveModeFlagBits::eMin ) |
                 VkFlags( ResolveModeFlagBits::eMax )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ResolveModeFlags operator|( ResolveModeFlagBits bit0,
                                                                     ResolveModeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ResolveModeFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ResolveModeFlags operator&(ResolveModeFlagBits bit0,
                                                                    ResolveModeFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return ResolveModeFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ResolveModeFlags operator^( ResolveModeFlagBits bit0,
                                                                     ResolveModeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ResolveModeFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ResolveModeFlags operator~( ResolveModeFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ResolveModeFlags( bits ) );
  }

  using ResolveModeFlagsKHR = ResolveModeFlags;

  VULKAN_HPP_INLINE std::string to_string( ResolveModeFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & ResolveModeFlagBits::eSampleZero )
      result += "SampleZero | ";
    if ( value & ResolveModeFlagBits::eAverage )
      result += "Average | ";
    if ( value & ResolveModeFlagBits::eMin )
      result += "Min | ";
    if ( value & ResolveModeFlagBits::eMax )
      result += "Max | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using SemaphoreWaitFlags = Flags<SemaphoreWaitFlagBits>;

  template <>
  struct FlagTraits<SemaphoreWaitFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( SemaphoreWaitFlagBits::eAny )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SemaphoreWaitFlags operator|( SemaphoreWaitFlagBits bit0,
                                                                       SemaphoreWaitFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SemaphoreWaitFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SemaphoreWaitFlags operator&(SemaphoreWaitFlagBits bit0,
                                                                      SemaphoreWaitFlagBits bit1)VULKAN_HPP_NOEXCEPT
  {
    return SemaphoreWaitFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SemaphoreWaitFlags operator^( SemaphoreWaitFlagBits bit0,
                                                                       SemaphoreWaitFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SemaphoreWaitFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SemaphoreWaitFlags operator~( SemaphoreWaitFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( SemaphoreWaitFlags( bits ) );
  }

  using SemaphoreWaitFlagsKHR = SemaphoreWaitFlags;

  VULKAN_HPP_INLINE std::string to_string( SemaphoreWaitFlags value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & SemaphoreWaitFlagBits::eAny )
      result += "Any | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_KHR_surface ===

  using CompositeAlphaFlagsKHR = Flags<CompositeAlphaFlagBitsKHR>;

  template <>
  struct FlagTraits<CompositeAlphaFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( CompositeAlphaFlagBitsKHR::eOpaque ) | VkFlags( CompositeAlphaFlagBitsKHR::ePreMultiplied ) |
                 VkFlags( CompositeAlphaFlagBitsKHR::ePostMultiplied ) | VkFlags( CompositeAlphaFlagBitsKHR::eInherit )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CompositeAlphaFlagsKHR
                                         operator|( CompositeAlphaFlagBitsKHR bit0, CompositeAlphaFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CompositeAlphaFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CompositeAlphaFlagsKHR
                                         operator&(CompositeAlphaFlagBitsKHR bit0, CompositeAlphaFlagBitsKHR bit1)VULKAN_HPP_NOEXCEPT
  {
    return CompositeAlphaFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CompositeAlphaFlagsKHR
                                         operator^( CompositeAlphaFlagBitsKHR bit0, CompositeAlphaFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CompositeAlphaFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CompositeAlphaFlagsKHR operator~( CompositeAlphaFlagBitsKHR bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( CompositeAlphaFlagsKHR( bits ) );
  }

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

  using SwapchainCreateFlagsKHR = Flags<SwapchainCreateFlagBitsKHR>;

  template <>
  struct FlagTraits<SwapchainCreateFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( SwapchainCreateFlagBitsKHR::eSplitInstanceBindRegions ) |
                 VkFlags( SwapchainCreateFlagBitsKHR::eProtected ) |
                 VkFlags( SwapchainCreateFlagBitsKHR::eMutableFormat )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SwapchainCreateFlagsKHR
                                         operator|( SwapchainCreateFlagBitsKHR bit0, SwapchainCreateFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SwapchainCreateFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SwapchainCreateFlagsKHR
                                         operator&(SwapchainCreateFlagBitsKHR bit0, SwapchainCreateFlagBitsKHR bit1)VULKAN_HPP_NOEXCEPT
  {
    return SwapchainCreateFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SwapchainCreateFlagsKHR
                                         operator^( SwapchainCreateFlagBitsKHR bit0, SwapchainCreateFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SwapchainCreateFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SwapchainCreateFlagsKHR operator~( SwapchainCreateFlagBitsKHR bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( SwapchainCreateFlagsKHR( bits ) );
  }

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
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using DeviceGroupPresentModeFlagsKHR = Flags<DeviceGroupPresentModeFlagBitsKHR>;

  template <>
  struct FlagTraits<DeviceGroupPresentModeFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( DeviceGroupPresentModeFlagBitsKHR::eLocal ) |
                 VkFlags( DeviceGroupPresentModeFlagBitsKHR::eRemote ) |
                 VkFlags( DeviceGroupPresentModeFlagBitsKHR::eSum ) |
                 VkFlags( DeviceGroupPresentModeFlagBitsKHR::eLocalMultiDevice )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceGroupPresentModeFlagsKHR
                                         operator|( DeviceGroupPresentModeFlagBitsKHR bit0, DeviceGroupPresentModeFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DeviceGroupPresentModeFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceGroupPresentModeFlagsKHR
                                         operator&(DeviceGroupPresentModeFlagBitsKHR bit0, DeviceGroupPresentModeFlagBitsKHR bit1)VULKAN_HPP_NOEXCEPT
  {
    return DeviceGroupPresentModeFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceGroupPresentModeFlagsKHR
                                         operator^( DeviceGroupPresentModeFlagBitsKHR bit0, DeviceGroupPresentModeFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DeviceGroupPresentModeFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceGroupPresentModeFlagsKHR
                                         operator~( DeviceGroupPresentModeFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( DeviceGroupPresentModeFlagsKHR( bits ) );
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

  using DisplayModeCreateFlagsKHR = Flags<DisplayModeCreateFlagBitsKHR>;

  VULKAN_HPP_INLINE std::string to_string( DisplayModeCreateFlagsKHR )
  {
    return "{}";
  }

  using DisplayPlaneAlphaFlagsKHR = Flags<DisplayPlaneAlphaFlagBitsKHR>;

  template <>
  struct FlagTraits<DisplayPlaneAlphaFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( DisplayPlaneAlphaFlagBitsKHR::eOpaque ) | VkFlags( DisplayPlaneAlphaFlagBitsKHR::eGlobal ) |
                 VkFlags( DisplayPlaneAlphaFlagBitsKHR::ePerPixel ) |
                 VkFlags( DisplayPlaneAlphaFlagBitsKHR::ePerPixelPremultiplied )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DisplayPlaneAlphaFlagsKHR
                                         operator|( DisplayPlaneAlphaFlagBitsKHR bit0, DisplayPlaneAlphaFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DisplayPlaneAlphaFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DisplayPlaneAlphaFlagsKHR
                                         operator&(DisplayPlaneAlphaFlagBitsKHR bit0, DisplayPlaneAlphaFlagBitsKHR bit1)VULKAN_HPP_NOEXCEPT
  {
    return DisplayPlaneAlphaFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DisplayPlaneAlphaFlagsKHR
                                         operator^( DisplayPlaneAlphaFlagBitsKHR bit0, DisplayPlaneAlphaFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DisplayPlaneAlphaFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DisplayPlaneAlphaFlagsKHR operator~( DisplayPlaneAlphaFlagBitsKHR bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( DisplayPlaneAlphaFlagsKHR( bits ) );
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

  using DisplaySurfaceCreateFlagsKHR = Flags<DisplaySurfaceCreateFlagBitsKHR>;

  VULKAN_HPP_INLINE std::string to_string( DisplaySurfaceCreateFlagsKHR )
  {
    return "{}";
  }

  using SurfaceTransformFlagsKHR = Flags<SurfaceTransformFlagBitsKHR>;

  template <>
  struct FlagTraits<SurfaceTransformFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( SurfaceTransformFlagBitsKHR::eIdentity ) | VkFlags( SurfaceTransformFlagBitsKHR::eRotate90 ) |
                 VkFlags( SurfaceTransformFlagBitsKHR::eRotate180 ) |
                 VkFlags( SurfaceTransformFlagBitsKHR::eRotate270 ) |
                 VkFlags( SurfaceTransformFlagBitsKHR::eHorizontalMirror ) |
                 VkFlags( SurfaceTransformFlagBitsKHR::eHorizontalMirrorRotate90 ) |
                 VkFlags( SurfaceTransformFlagBitsKHR::eHorizontalMirrorRotate180 ) |
                 VkFlags( SurfaceTransformFlagBitsKHR::eHorizontalMirrorRotate270 ) |
                 VkFlags( SurfaceTransformFlagBitsKHR::eInherit )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SurfaceTransformFlagsKHR
                                         operator|( SurfaceTransformFlagBitsKHR bit0, SurfaceTransformFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SurfaceTransformFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SurfaceTransformFlagsKHR
                                         operator&(SurfaceTransformFlagBitsKHR bit0, SurfaceTransformFlagBitsKHR bit1)VULKAN_HPP_NOEXCEPT
  {
    return SurfaceTransformFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SurfaceTransformFlagsKHR
                                         operator^( SurfaceTransformFlagBitsKHR bit0, SurfaceTransformFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SurfaceTransformFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SurfaceTransformFlagsKHR operator~( SurfaceTransformFlagBitsKHR bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( SurfaceTransformFlagsKHR( bits ) );
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

  using XlibSurfaceCreateFlagsKHR = Flags<XlibSurfaceCreateFlagBitsKHR>;

  VULKAN_HPP_INLINE std::string to_string( XlibSurfaceCreateFlagsKHR )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_XLIB_KHR*/

#if defined( VK_USE_PLATFORM_XCB_KHR )
  //=== VK_KHR_xcb_surface ===

  using XcbSurfaceCreateFlagsKHR = Flags<XcbSurfaceCreateFlagBitsKHR>;

  VULKAN_HPP_INLINE std::string to_string( XcbSurfaceCreateFlagsKHR )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_XCB_KHR*/

#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
  //=== VK_KHR_wayland_surface ===

  using WaylandSurfaceCreateFlagsKHR = Flags<WaylandSurfaceCreateFlagBitsKHR>;

  VULKAN_HPP_INLINE std::string to_string( WaylandSurfaceCreateFlagsKHR )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  //=== VK_KHR_android_surface ===

  using AndroidSurfaceCreateFlagsKHR = Flags<AndroidSurfaceCreateFlagBitsKHR>;

  VULKAN_HPP_INLINE std::string to_string( AndroidSurfaceCreateFlagsKHR )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_win32_surface ===

  using Win32SurfaceCreateFlagsKHR = Flags<Win32SurfaceCreateFlagBitsKHR>;

  VULKAN_HPP_INLINE std::string to_string( Win32SurfaceCreateFlagsKHR )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_EXT_debug_report ===

  using DebugReportFlagsEXT = Flags<DebugReportFlagBitsEXT>;

  template <>
  struct FlagTraits<DebugReportFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( DebugReportFlagBitsEXT::eInformation ) | VkFlags( DebugReportFlagBitsEXT::eWarning ) |
                 VkFlags( DebugReportFlagBitsEXT::ePerformanceWarning ) | VkFlags( DebugReportFlagBitsEXT::eError ) |
                 VkFlags( DebugReportFlagBitsEXT::eDebug )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DebugReportFlagsEXT
                                         operator|( DebugReportFlagBitsEXT bit0, DebugReportFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DebugReportFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DebugReportFlagsEXT operator&(DebugReportFlagBitsEXT bit0,
                                                                       DebugReportFlagBitsEXT bit1)VULKAN_HPP_NOEXCEPT
  {
    return DebugReportFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DebugReportFlagsEXT
                                         operator^( DebugReportFlagBitsEXT bit0, DebugReportFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DebugReportFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DebugReportFlagsEXT operator~( DebugReportFlagBitsEXT bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( DebugReportFlagsEXT( bits ) );
  }

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

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_KHR_video_queue ===

  using VideoCodecOperationFlagsKHR = Flags<VideoCodecOperationFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoCodecOperationFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoCodecOperationFlagBitsKHR::eInvalid )
#  if defined( VK_ENABLE_BETA_EXTENSIONS )
                 | VkFlags( VideoCodecOperationFlagBitsKHR::eEncodeH264EXT ) |
                 VkFlags( VideoCodecOperationFlagBitsKHR::eDecodeH264EXT ) |
                 VkFlags( VideoCodecOperationFlagBitsKHR::eDecodeH265EXT )
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCodecOperationFlagsKHR
                                         operator|( VideoCodecOperationFlagBitsKHR bit0, VideoCodecOperationFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoCodecOperationFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCodecOperationFlagsKHR
                                         operator&(VideoCodecOperationFlagBitsKHR bit0, VideoCodecOperationFlagBitsKHR bit1)VULKAN_HPP_NOEXCEPT
  {
    return VideoCodecOperationFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCodecOperationFlagsKHR
                                         operator^( VideoCodecOperationFlagBitsKHR bit0, VideoCodecOperationFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoCodecOperationFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCodecOperationFlagsKHR operator~( VideoCodecOperationFlagBitsKHR bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoCodecOperationFlagsKHR( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( VideoCodecOperationFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
#  if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & VideoCodecOperationFlagBitsKHR::eEncodeH264EXT )
      result += "EncodeH264EXT | ";
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/
#  if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & VideoCodecOperationFlagBitsKHR::eDecodeH264EXT )
      result += "DecodeH264EXT | ";
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/
#  if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & VideoCodecOperationFlagBitsKHR::eDecodeH265EXT )
      result += "DecodeH265EXT | ";
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using VideoChromaSubsamplingFlagsKHR = Flags<VideoChromaSubsamplingFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoChromaSubsamplingFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoChromaSubsamplingFlagBitsKHR::eInvalid ) |
                 VkFlags( VideoChromaSubsamplingFlagBitsKHR::eMonochrome ) |
                 VkFlags( VideoChromaSubsamplingFlagBitsKHR::e420 ) |
                 VkFlags( VideoChromaSubsamplingFlagBitsKHR::e422 ) | VkFlags( VideoChromaSubsamplingFlagBitsKHR::e444 )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoChromaSubsamplingFlagsKHR
                                         operator|( VideoChromaSubsamplingFlagBitsKHR bit0, VideoChromaSubsamplingFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoChromaSubsamplingFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoChromaSubsamplingFlagsKHR
                                         operator&(VideoChromaSubsamplingFlagBitsKHR bit0, VideoChromaSubsamplingFlagBitsKHR bit1)VULKAN_HPP_NOEXCEPT
  {
    return VideoChromaSubsamplingFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoChromaSubsamplingFlagsKHR
                                         operator^( VideoChromaSubsamplingFlagBitsKHR bit0, VideoChromaSubsamplingFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoChromaSubsamplingFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoChromaSubsamplingFlagsKHR
                                         operator~( VideoChromaSubsamplingFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoChromaSubsamplingFlagsKHR( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( VideoChromaSubsamplingFlagsKHR value )
  {
    if ( !value )
      return "{}";

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

  using VideoComponentBitDepthFlagsKHR = Flags<VideoComponentBitDepthFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoComponentBitDepthFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoComponentBitDepthFlagBitsKHR::eInvalid ) |
                 VkFlags( VideoComponentBitDepthFlagBitsKHR::e8 ) | VkFlags( VideoComponentBitDepthFlagBitsKHR::e10 ) |
                 VkFlags( VideoComponentBitDepthFlagBitsKHR::e12 )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoComponentBitDepthFlagsKHR
                                         operator|( VideoComponentBitDepthFlagBitsKHR bit0, VideoComponentBitDepthFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoComponentBitDepthFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoComponentBitDepthFlagsKHR
                                         operator&(VideoComponentBitDepthFlagBitsKHR bit0, VideoComponentBitDepthFlagBitsKHR bit1)VULKAN_HPP_NOEXCEPT
  {
    return VideoComponentBitDepthFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoComponentBitDepthFlagsKHR
                                         operator^( VideoComponentBitDepthFlagBitsKHR bit0, VideoComponentBitDepthFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoComponentBitDepthFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoComponentBitDepthFlagsKHR
                                         operator~( VideoComponentBitDepthFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoComponentBitDepthFlagsKHR( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( VideoComponentBitDepthFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoComponentBitDepthFlagBitsKHR::e8 )
      result += "8 | ";
    if ( value & VideoComponentBitDepthFlagBitsKHR::e10 )
      result += "10 | ";
    if ( value & VideoComponentBitDepthFlagBitsKHR::e12 )
      result += "12 | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using VideoCapabilitiesFlagsKHR = Flags<VideoCapabilitiesFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoCapabilitiesFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoCapabilitiesFlagBitsKHR::eProtectedContent ) |
                 VkFlags( VideoCapabilitiesFlagBitsKHR::eSeparateReferenceImages )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCapabilitiesFlagsKHR
                                         operator|( VideoCapabilitiesFlagBitsKHR bit0, VideoCapabilitiesFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoCapabilitiesFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCapabilitiesFlagsKHR
                                         operator&(VideoCapabilitiesFlagBitsKHR bit0, VideoCapabilitiesFlagBitsKHR bit1)VULKAN_HPP_NOEXCEPT
  {
    return VideoCapabilitiesFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCapabilitiesFlagsKHR
                                         operator^( VideoCapabilitiesFlagBitsKHR bit0, VideoCapabilitiesFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoCapabilitiesFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCapabilitiesFlagsKHR operator~( VideoCapabilitiesFlagBitsKHR bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoCapabilitiesFlagsKHR( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( VideoCapabilitiesFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoCapabilitiesFlagBitsKHR::eProtectedContent )
      result += "ProtectedContent | ";
    if ( value & VideoCapabilitiesFlagBitsKHR::eSeparateReferenceImages )
      result += "SeparateReferenceImages | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using VideoSessionCreateFlagsKHR = Flags<VideoSessionCreateFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoSessionCreateFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags =
        VkFlags( VideoSessionCreateFlagBitsKHR::eDefault ) | VkFlags( VideoSessionCreateFlagBitsKHR::eProtectedContent )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoSessionCreateFlagsKHR
                                         operator|( VideoSessionCreateFlagBitsKHR bit0, VideoSessionCreateFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoSessionCreateFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoSessionCreateFlagsKHR
                                         operator&(VideoSessionCreateFlagBitsKHR bit0, VideoSessionCreateFlagBitsKHR bit1)VULKAN_HPP_NOEXCEPT
  {
    return VideoSessionCreateFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoSessionCreateFlagsKHR
                                         operator^( VideoSessionCreateFlagBitsKHR bit0, VideoSessionCreateFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoSessionCreateFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoSessionCreateFlagsKHR operator~( VideoSessionCreateFlagBitsKHR bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoSessionCreateFlagsKHR( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( VideoSessionCreateFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoSessionCreateFlagBitsKHR::eProtectedContent )
      result += "ProtectedContent | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using VideoBeginCodingFlagsKHR = Flags<VideoBeginCodingFlagBitsKHR>;

  VULKAN_HPP_INLINE std::string to_string( VideoBeginCodingFlagsKHR )
  {
    return "{}";
  }

  using VideoEndCodingFlagsKHR = Flags<VideoEndCodingFlagBitsKHR>;

  VULKAN_HPP_INLINE std::string to_string( VideoEndCodingFlagsKHR )
  {
    return "{}";
  }

  using VideoCodingControlFlagsKHR = Flags<VideoCodingControlFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoCodingControlFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoCodingControlFlagBitsKHR::eDefault ) | VkFlags( VideoCodingControlFlagBitsKHR::eReset )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCodingControlFlagsKHR
                                         operator|( VideoCodingControlFlagBitsKHR bit0, VideoCodingControlFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoCodingControlFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCodingControlFlagsKHR
                                         operator&(VideoCodingControlFlagBitsKHR bit0, VideoCodingControlFlagBitsKHR bit1)VULKAN_HPP_NOEXCEPT
  {
    return VideoCodingControlFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCodingControlFlagsKHR
                                         operator^( VideoCodingControlFlagBitsKHR bit0, VideoCodingControlFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoCodingControlFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCodingControlFlagsKHR operator~( VideoCodingControlFlagBitsKHR bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoCodingControlFlagsKHR( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( VideoCodingControlFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoCodingControlFlagBitsKHR::eReset )
      result += "Reset | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using VideoCodingQualityPresetFlagsKHR = Flags<VideoCodingQualityPresetFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoCodingQualityPresetFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoCodingQualityPresetFlagBitsKHR::eDefault ) |
                 VkFlags( VideoCodingQualityPresetFlagBitsKHR::eNormal ) |
                 VkFlags( VideoCodingQualityPresetFlagBitsKHR::ePower ) |
                 VkFlags( VideoCodingQualityPresetFlagBitsKHR::eQuality )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCodingQualityPresetFlagsKHR
                                         operator|( VideoCodingQualityPresetFlagBitsKHR bit0, VideoCodingQualityPresetFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoCodingQualityPresetFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCodingQualityPresetFlagsKHR
                                         operator&(VideoCodingQualityPresetFlagBitsKHR bit0, VideoCodingQualityPresetFlagBitsKHR bit1)VULKAN_HPP_NOEXCEPT
  {
    return VideoCodingQualityPresetFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCodingQualityPresetFlagsKHR
                                         operator^( VideoCodingQualityPresetFlagBitsKHR bit0, VideoCodingQualityPresetFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoCodingQualityPresetFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCodingQualityPresetFlagsKHR
                                         operator~( VideoCodingQualityPresetFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoCodingQualityPresetFlagsKHR( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( VideoCodingQualityPresetFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoCodingQualityPresetFlagBitsKHR::eNormal )
      result += "Normal | ";
    if ( value & VideoCodingQualityPresetFlagBitsKHR::ePower )
      result += "Power | ";
    if ( value & VideoCodingQualityPresetFlagBitsKHR::eQuality )
      result += "Quality | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_KHR_video_decode_queue ===

  using VideoDecodeFlagsKHR = Flags<VideoDecodeFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoDecodeFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoDecodeFlagBitsKHR::eDefault ) | VkFlags( VideoDecodeFlagBitsKHR::eReserved0 )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoDecodeFlagsKHR
                                         operator|( VideoDecodeFlagBitsKHR bit0, VideoDecodeFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoDecodeFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoDecodeFlagsKHR operator&(VideoDecodeFlagBitsKHR bit0,
                                                                       VideoDecodeFlagBitsKHR bit1)VULKAN_HPP_NOEXCEPT
  {
    return VideoDecodeFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoDecodeFlagsKHR
                                         operator^( VideoDecodeFlagBitsKHR bit0, VideoDecodeFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoDecodeFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoDecodeFlagsKHR operator~( VideoDecodeFlagBitsKHR bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoDecodeFlagsKHR( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( VideoDecodeFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoDecodeFlagBitsKHR::eReserved0 )
      result += "Reserved0 | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_EXT_transform_feedback ===

  using PipelineRasterizationStateStreamCreateFlagsEXT = Flags<PipelineRasterizationStateStreamCreateFlagBitsEXT>;

  VULKAN_HPP_INLINE std::string to_string( PipelineRasterizationStateStreamCreateFlagsEXT )
  {
    return "{}";
  }

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_EXT_video_encode_h264 ===

  using VideoEncodeH264CapabilitiesFlagsEXT = Flags<VideoEncodeH264CapabilitiesFlagBitsEXT>;

  template <>
  struct FlagTraits<VideoEncodeH264CapabilitiesFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags =
        VkFlags( VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityCabac ) |
        VkFlags( VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityCavlc ) |
        VkFlags( VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityWeightedBiPredImplicit ) |
        VkFlags( VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityTransform8X8 ) |
        VkFlags( VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityChromaQpOffset ) |
        VkFlags( VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilitySecondChromaQpOffset ) |
        VkFlags( VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityDeblockingFilterDisabled ) |
        VkFlags( VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityDeblockingFilterEnabled ) |
        VkFlags( VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityDeblockingFilterPartial ) |
        VkFlags( VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityMultipleSlicePerFrame ) |
        VkFlags( VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityEvenlyDistributedSliceSize )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264CapabilitiesFlagsEXT operator|(
    VideoEncodeH264CapabilitiesFlagBitsEXT bit0, VideoEncodeH264CapabilitiesFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH264CapabilitiesFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264CapabilitiesFlagsEXT operator&(
    VideoEncodeH264CapabilitiesFlagBitsEXT bit0, VideoEncodeH264CapabilitiesFlagBitsEXT bit1)VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH264CapabilitiesFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264CapabilitiesFlagsEXT operator^(
    VideoEncodeH264CapabilitiesFlagBitsEXT bit0, VideoEncodeH264CapabilitiesFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH264CapabilitiesFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264CapabilitiesFlagsEXT
                                         operator~( VideoEncodeH264CapabilitiesFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoEncodeH264CapabilitiesFlagsEXT( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeH264CapabilitiesFlagsEXT value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityCabac )
      result += "VkVideoEncodeH264CapabilityCabac | ";
    if ( value & VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityCavlc )
      result += "VkVideoEncodeH264CapabilityCavlc | ";
    if ( value & VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityWeightedBiPredImplicit )
      result += "VkVideoEncodeH264CapabilityWeightedBiPredImplicit | ";
    if ( value & VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityTransform8X8 )
      result += "VkVideoEncodeH264CapabilityTransform8X8 | ";
    if ( value & VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityChromaQpOffset )
      result += "VkVideoEncodeH264CapabilityChromaQpOffset | ";
    if ( value & VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilitySecondChromaQpOffset )
      result += "VkVideoEncodeH264CapabilitySecondChromaQpOffset | ";
    if ( value & VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityDeblockingFilterDisabled )
      result += "VkVideoEncodeH264CapabilityDeblockingFilterDisabled | ";
    if ( value & VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityDeblockingFilterEnabled )
      result += "VkVideoEncodeH264CapabilityDeblockingFilterEnabled | ";
    if ( value & VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityDeblockingFilterPartial )
      result += "VkVideoEncodeH264CapabilityDeblockingFilterPartial | ";
    if ( value & VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityMultipleSlicePerFrame )
      result += "VkVideoEncodeH264CapabilityMultipleSlicePerFrame | ";
    if ( value & VideoEncodeH264CapabilitiesFlagBitsEXT::eVkVideoEncodeH264CapabilityEvenlyDistributedSliceSize )
      result += "VkVideoEncodeH264CapabilityEvenlyDistributedSliceSize | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using VideoEncodeH264InputModeFlagsEXT = Flags<VideoEncodeH264InputModeFlagBitsEXT>;

  template <>
  struct FlagTraits<VideoEncodeH264InputModeFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoEncodeH264InputModeFlagBitsEXT::eFrame ) |
                 VkFlags( VideoEncodeH264InputModeFlagBitsEXT::eSlice ) |
                 VkFlags( VideoEncodeH264InputModeFlagBitsEXT::eNonVcl )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264InputModeFlagsEXT
                                         operator|( VideoEncodeH264InputModeFlagBitsEXT bit0, VideoEncodeH264InputModeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH264InputModeFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264InputModeFlagsEXT
                                         operator&(VideoEncodeH264InputModeFlagBitsEXT bit0, VideoEncodeH264InputModeFlagBitsEXT bit1)VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH264InputModeFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264InputModeFlagsEXT
                                         operator^( VideoEncodeH264InputModeFlagBitsEXT bit0, VideoEncodeH264InputModeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH264InputModeFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264InputModeFlagsEXT
                                         operator~( VideoEncodeH264InputModeFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoEncodeH264InputModeFlagsEXT( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeH264InputModeFlagsEXT value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoEncodeH264InputModeFlagBitsEXT::eFrame )
      result += "Frame | ";
    if ( value & VideoEncodeH264InputModeFlagBitsEXT::eSlice )
      result += "Slice | ";
    if ( value & VideoEncodeH264InputModeFlagBitsEXT::eNonVcl )
      result += "NonVcl | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using VideoEncodeH264OutputModeFlagsEXT = Flags<VideoEncodeH264OutputModeFlagBitsEXT>;

  template <>
  struct FlagTraits<VideoEncodeH264OutputModeFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoEncodeH264OutputModeFlagBitsEXT::eFrame ) |
                 VkFlags( VideoEncodeH264OutputModeFlagBitsEXT::eSlice ) |
                 VkFlags( VideoEncodeH264OutputModeFlagBitsEXT::eNonVcl )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264OutputModeFlagsEXT operator|(
    VideoEncodeH264OutputModeFlagBitsEXT bit0, VideoEncodeH264OutputModeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH264OutputModeFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264OutputModeFlagsEXT
                                         operator&(VideoEncodeH264OutputModeFlagBitsEXT bit0, VideoEncodeH264OutputModeFlagBitsEXT bit1)VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH264OutputModeFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264OutputModeFlagsEXT operator^(
    VideoEncodeH264OutputModeFlagBitsEXT bit0, VideoEncodeH264OutputModeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH264OutputModeFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264OutputModeFlagsEXT
                                         operator~( VideoEncodeH264OutputModeFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoEncodeH264OutputModeFlagsEXT( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeH264OutputModeFlagsEXT value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoEncodeH264OutputModeFlagBitsEXT::eFrame )
      result += "Frame | ";
    if ( value & VideoEncodeH264OutputModeFlagBitsEXT::eSlice )
      result += "Slice | ";
    if ( value & VideoEncodeH264OutputModeFlagBitsEXT::eNonVcl )
      result += "NonVcl | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using VideoEncodeH264CreateFlagsEXT = Flags<VideoEncodeH264CreateFlagBitsEXT>;

  template <>
  struct FlagTraits<VideoEncodeH264CreateFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags =
        VkFlags( VideoEncodeH264CreateFlagBitsEXT::eDefault ) | VkFlags( VideoEncodeH264CreateFlagBitsEXT::eReserved0 )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264CreateFlagsEXT
                                         operator|( VideoEncodeH264CreateFlagBitsEXT bit0, VideoEncodeH264CreateFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH264CreateFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264CreateFlagsEXT
                                         operator&(VideoEncodeH264CreateFlagBitsEXT bit0, VideoEncodeH264CreateFlagBitsEXT bit1)VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH264CreateFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264CreateFlagsEXT
                                         operator^( VideoEncodeH264CreateFlagBitsEXT bit0, VideoEncodeH264CreateFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH264CreateFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264CreateFlagsEXT
                                         operator~( VideoEncodeH264CreateFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoEncodeH264CreateFlagsEXT( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeH264CreateFlagsEXT value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoEncodeH264CreateFlagBitsEXT::eReserved0 )
      result += "Reserved0 | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_EXT_video_decode_h264 ===

  using VideoDecodeH264FieldLayoutFlagsEXT = Flags<VideoDecodeH264FieldLayoutFlagBitsEXT>;

  template <>
  struct FlagTraits<VideoDecodeH264FieldLayoutFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoDecodeH264FieldLayoutFlagBitsEXT::eVkVideoDecodeH264ProgressivePicturesOnly ) |
                 VkFlags( VideoDecodeH264FieldLayoutFlagBitsEXT::eLineInterlacedPlane ) |
                 VkFlags( VideoDecodeH264FieldLayoutFlagBitsEXT::eSeparateInterlacedPlane )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoDecodeH264FieldLayoutFlagsEXT operator|(
    VideoDecodeH264FieldLayoutFlagBitsEXT bit0, VideoDecodeH264FieldLayoutFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoDecodeH264FieldLayoutFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoDecodeH264FieldLayoutFlagsEXT
                                         operator&(VideoDecodeH264FieldLayoutFlagBitsEXT bit0, VideoDecodeH264FieldLayoutFlagBitsEXT bit1)VULKAN_HPP_NOEXCEPT
  {
    return VideoDecodeH264FieldLayoutFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoDecodeH264FieldLayoutFlagsEXT operator^(
    VideoDecodeH264FieldLayoutFlagBitsEXT bit0, VideoDecodeH264FieldLayoutFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoDecodeH264FieldLayoutFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoDecodeH264FieldLayoutFlagsEXT
                                         operator~( VideoDecodeH264FieldLayoutFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoDecodeH264FieldLayoutFlagsEXT( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( VideoDecodeH264FieldLayoutFlagsEXT value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoDecodeH264FieldLayoutFlagBitsEXT::eLineInterlacedPlane )
      result += "LineInterlacedPlane | ";
    if ( value & VideoDecodeH264FieldLayoutFlagBitsEXT::eSeparateInterlacedPlane )
      result += "SeparateInterlacedPlane | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using VideoDecodeH264CreateFlagsEXT = Flags<VideoDecodeH264CreateFlagBitsEXT>;

  VULKAN_HPP_INLINE std::string to_string( VideoDecodeH264CreateFlagsEXT )
  {
    return "{}";
  }
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

#if defined( VK_USE_PLATFORM_GGP )
  //=== VK_GGP_stream_descriptor_surface ===

  using StreamDescriptorSurfaceCreateFlagsGGP = Flags<StreamDescriptorSurfaceCreateFlagBitsGGP>;

  VULKAN_HPP_INLINE std::string to_string( StreamDescriptorSurfaceCreateFlagsGGP )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_GGP*/

  //=== VK_NV_external_memory_capabilities ===

  using ExternalMemoryHandleTypeFlagsNV = Flags<ExternalMemoryHandleTypeFlagBitsNV>;

  template <>
  struct FlagTraits<ExternalMemoryHandleTypeFlagBitsNV>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ExternalMemoryHandleTypeFlagBitsNV::eOpaqueWin32 ) |
                 VkFlags( ExternalMemoryHandleTypeFlagBitsNV::eOpaqueWin32Kmt ) |
                 VkFlags( ExternalMemoryHandleTypeFlagBitsNV::eD3D11Image ) |
                 VkFlags( ExternalMemoryHandleTypeFlagBitsNV::eD3D11ImageKmt )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryHandleTypeFlagsNV
                                         operator|( ExternalMemoryHandleTypeFlagBitsNV bit0, ExternalMemoryHandleTypeFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalMemoryHandleTypeFlagsNV( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryHandleTypeFlagsNV
                                         operator&(ExternalMemoryHandleTypeFlagBitsNV bit0, ExternalMemoryHandleTypeFlagBitsNV bit1)VULKAN_HPP_NOEXCEPT
  {
    return ExternalMemoryHandleTypeFlagsNV( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryHandleTypeFlagsNV
                                         operator^( ExternalMemoryHandleTypeFlagBitsNV bit0, ExternalMemoryHandleTypeFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalMemoryHandleTypeFlagsNV( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryHandleTypeFlagsNV
                                         operator~( ExternalMemoryHandleTypeFlagBitsNV bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ExternalMemoryHandleTypeFlagsNV( bits ) );
  }

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

  using ExternalMemoryFeatureFlagsNV = Flags<ExternalMemoryFeatureFlagBitsNV>;

  template <>
  struct FlagTraits<ExternalMemoryFeatureFlagBitsNV>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ExternalMemoryFeatureFlagBitsNV::eDedicatedOnly ) |
                 VkFlags( ExternalMemoryFeatureFlagBitsNV::eExportable ) |
                 VkFlags( ExternalMemoryFeatureFlagBitsNV::eImportable )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryFeatureFlagsNV
                                         operator|( ExternalMemoryFeatureFlagBitsNV bit0, ExternalMemoryFeatureFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalMemoryFeatureFlagsNV( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryFeatureFlagsNV
                                         operator&(ExternalMemoryFeatureFlagBitsNV bit0, ExternalMemoryFeatureFlagBitsNV bit1)VULKAN_HPP_NOEXCEPT
  {
    return ExternalMemoryFeatureFlagsNV( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryFeatureFlagsNV
                                         operator^( ExternalMemoryFeatureFlagBitsNV bit0, ExternalMemoryFeatureFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalMemoryFeatureFlagsNV( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryFeatureFlagsNV operator~( ExternalMemoryFeatureFlagBitsNV bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( ExternalMemoryFeatureFlagsNV( bits ) );
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

  using ViSurfaceCreateFlagsNN = Flags<ViSurfaceCreateFlagBitsNN>;

  VULKAN_HPP_INLINE std::string to_string( ViSurfaceCreateFlagsNN )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_VI_NN*/

  //=== VK_EXT_conditional_rendering ===

  using ConditionalRenderingFlagsEXT = Flags<ConditionalRenderingFlagBitsEXT>;

  template <>
  struct FlagTraits<ConditionalRenderingFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ConditionalRenderingFlagBitsEXT::eInverted )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ConditionalRenderingFlagsEXT
                                         operator|( ConditionalRenderingFlagBitsEXT bit0, ConditionalRenderingFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ConditionalRenderingFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ConditionalRenderingFlagsEXT
                                         operator&(ConditionalRenderingFlagBitsEXT bit0, ConditionalRenderingFlagBitsEXT bit1)VULKAN_HPP_NOEXCEPT
  {
    return ConditionalRenderingFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ConditionalRenderingFlagsEXT
                                         operator^( ConditionalRenderingFlagBitsEXT bit0, ConditionalRenderingFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ConditionalRenderingFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ConditionalRenderingFlagsEXT operator~( ConditionalRenderingFlagBitsEXT bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( ConditionalRenderingFlagsEXT( bits ) );
  }

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

  using SurfaceCounterFlagsEXT = Flags<SurfaceCounterFlagBitsEXT>;

  template <>
  struct FlagTraits<SurfaceCounterFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( SurfaceCounterFlagBitsEXT::eVblank )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SurfaceCounterFlagsEXT
                                         operator|( SurfaceCounterFlagBitsEXT bit0, SurfaceCounterFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SurfaceCounterFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SurfaceCounterFlagsEXT
                                         operator&(SurfaceCounterFlagBitsEXT bit0, SurfaceCounterFlagBitsEXT bit1)VULKAN_HPP_NOEXCEPT
  {
    return SurfaceCounterFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SurfaceCounterFlagsEXT
                                         operator^( SurfaceCounterFlagBitsEXT bit0, SurfaceCounterFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SurfaceCounterFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SurfaceCounterFlagsEXT operator~( SurfaceCounterFlagBitsEXT bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( SurfaceCounterFlagsEXT( bits ) );
  }

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

  using PipelineViewportSwizzleStateCreateFlagsNV = Flags<PipelineViewportSwizzleStateCreateFlagBitsNV>;

  VULKAN_HPP_INLINE std::string to_string( PipelineViewportSwizzleStateCreateFlagsNV )
  {
    return "{}";
  }

  //=== VK_EXT_discard_rectangles ===

  using PipelineDiscardRectangleStateCreateFlagsEXT = Flags<PipelineDiscardRectangleStateCreateFlagBitsEXT>;

  VULKAN_HPP_INLINE std::string to_string( PipelineDiscardRectangleStateCreateFlagsEXT )
  {
    return "{}";
  }

  //=== VK_EXT_conservative_rasterization ===

  using PipelineRasterizationConservativeStateCreateFlagsEXT =
    Flags<PipelineRasterizationConservativeStateCreateFlagBitsEXT>;

  VULKAN_HPP_INLINE std::string to_string( PipelineRasterizationConservativeStateCreateFlagsEXT )
  {
    return "{}";
  }

  //=== VK_EXT_depth_clip_enable ===

  using PipelineRasterizationDepthClipStateCreateFlagsEXT = Flags<PipelineRasterizationDepthClipStateCreateFlagBitsEXT>;

  VULKAN_HPP_INLINE std::string to_string( PipelineRasterizationDepthClipStateCreateFlagsEXT )
  {
    return "{}";
  }

  //=== VK_KHR_performance_query ===

  using PerformanceCounterDescriptionFlagsKHR = Flags<PerformanceCounterDescriptionFlagBitsKHR>;

  template <>
  struct FlagTraits<PerformanceCounterDescriptionFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( PerformanceCounterDescriptionFlagBitsKHR::ePerformanceImpacting ) |
                 VkFlags( PerformanceCounterDescriptionFlagBitsKHR::eConcurrentlyImpacted )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PerformanceCounterDescriptionFlagsKHR operator|(
    PerformanceCounterDescriptionFlagBitsKHR bit0, PerformanceCounterDescriptionFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PerformanceCounterDescriptionFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PerformanceCounterDescriptionFlagsKHR operator&(
    PerformanceCounterDescriptionFlagBitsKHR bit0, PerformanceCounterDescriptionFlagBitsKHR bit1)VULKAN_HPP_NOEXCEPT
  {
    return PerformanceCounterDescriptionFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PerformanceCounterDescriptionFlagsKHR operator^(
    PerformanceCounterDescriptionFlagBitsKHR bit0, PerformanceCounterDescriptionFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PerformanceCounterDescriptionFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PerformanceCounterDescriptionFlagsKHR
                                         operator~( PerformanceCounterDescriptionFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( PerformanceCounterDescriptionFlagsKHR( bits ) );
  }

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

  using AcquireProfilingLockFlagsKHR = Flags<AcquireProfilingLockFlagBitsKHR>;

  VULKAN_HPP_INLINE std::string to_string( AcquireProfilingLockFlagsKHR )
  {
    return "{}";
  }

#if defined( VK_USE_PLATFORM_IOS_MVK )
  //=== VK_MVK_ios_surface ===

  using IOSSurfaceCreateFlagsMVK = Flags<IOSSurfaceCreateFlagBitsMVK>;

  VULKAN_HPP_INLINE std::string to_string( IOSSurfaceCreateFlagsMVK )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_IOS_MVK*/

#if defined( VK_USE_PLATFORM_MACOS_MVK )
  //=== VK_MVK_macos_surface ===

  using MacOSSurfaceCreateFlagsMVK = Flags<MacOSSurfaceCreateFlagBitsMVK>;

  VULKAN_HPP_INLINE std::string to_string( MacOSSurfaceCreateFlagsMVK )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_MACOS_MVK*/

  //=== VK_EXT_debug_utils ===

  using DebugUtilsMessageSeverityFlagsEXT = Flags<DebugUtilsMessageSeverityFlagBitsEXT>;

  template <>
  struct FlagTraits<DebugUtilsMessageSeverityFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( DebugUtilsMessageSeverityFlagBitsEXT::eVerbose ) |
                 VkFlags( DebugUtilsMessageSeverityFlagBitsEXT::eInfo ) |
                 VkFlags( DebugUtilsMessageSeverityFlagBitsEXT::eWarning ) |
                 VkFlags( DebugUtilsMessageSeverityFlagBitsEXT::eError )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DebugUtilsMessageSeverityFlagsEXT operator|(
    DebugUtilsMessageSeverityFlagBitsEXT bit0, DebugUtilsMessageSeverityFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DebugUtilsMessageSeverityFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DebugUtilsMessageSeverityFlagsEXT
                                         operator&(DebugUtilsMessageSeverityFlagBitsEXT bit0, DebugUtilsMessageSeverityFlagBitsEXT bit1)VULKAN_HPP_NOEXCEPT
  {
    return DebugUtilsMessageSeverityFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DebugUtilsMessageSeverityFlagsEXT operator^(
    DebugUtilsMessageSeverityFlagBitsEXT bit0, DebugUtilsMessageSeverityFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DebugUtilsMessageSeverityFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DebugUtilsMessageSeverityFlagsEXT
                                         operator~( DebugUtilsMessageSeverityFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( DebugUtilsMessageSeverityFlagsEXT( bits ) );
  }

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

  using DebugUtilsMessageTypeFlagsEXT = Flags<DebugUtilsMessageTypeFlagBitsEXT>;

  template <>
  struct FlagTraits<DebugUtilsMessageTypeFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( DebugUtilsMessageTypeFlagBitsEXT::eGeneral ) |
                 VkFlags( DebugUtilsMessageTypeFlagBitsEXT::eValidation ) |
                 VkFlags( DebugUtilsMessageTypeFlagBitsEXT::ePerformance )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DebugUtilsMessageTypeFlagsEXT
                                         operator|( DebugUtilsMessageTypeFlagBitsEXT bit0, DebugUtilsMessageTypeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DebugUtilsMessageTypeFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DebugUtilsMessageTypeFlagsEXT
                                         operator&(DebugUtilsMessageTypeFlagBitsEXT bit0, DebugUtilsMessageTypeFlagBitsEXT bit1)VULKAN_HPP_NOEXCEPT
  {
    return DebugUtilsMessageTypeFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DebugUtilsMessageTypeFlagsEXT
                                         operator^( DebugUtilsMessageTypeFlagBitsEXT bit0, DebugUtilsMessageTypeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DebugUtilsMessageTypeFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DebugUtilsMessageTypeFlagsEXT
                                         operator~( DebugUtilsMessageTypeFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( DebugUtilsMessageTypeFlagsEXT( bits ) );
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
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using DebugUtilsMessengerCallbackDataFlagsEXT = Flags<DebugUtilsMessengerCallbackDataFlagBitsEXT>;

  VULKAN_HPP_INLINE std::string to_string( DebugUtilsMessengerCallbackDataFlagsEXT )
  {
    return "{}";
  }

  using DebugUtilsMessengerCreateFlagsEXT = Flags<DebugUtilsMessengerCreateFlagBitsEXT>;

  VULKAN_HPP_INLINE std::string to_string( DebugUtilsMessengerCreateFlagsEXT )
  {
    return "{}";
  }

  //=== VK_NV_fragment_coverage_to_color ===

  using PipelineCoverageToColorStateCreateFlagsNV = Flags<PipelineCoverageToColorStateCreateFlagBitsNV>;

  VULKAN_HPP_INLINE std::string to_string( PipelineCoverageToColorStateCreateFlagsNV )
  {
    return "{}";
  }

  //=== VK_KHR_acceleration_structure ===

  using GeometryFlagsKHR = Flags<GeometryFlagBitsKHR>;

  template <>
  struct FlagTraits<GeometryFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( GeometryFlagBitsKHR::eOpaque ) | VkFlags( GeometryFlagBitsKHR::eNoDuplicateAnyHitInvocation )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR GeometryFlagsKHR operator|( GeometryFlagBitsKHR bit0,
                                                                     GeometryFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return GeometryFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR GeometryFlagsKHR operator&(GeometryFlagBitsKHR bit0,
                                                                    GeometryFlagBitsKHR bit1)VULKAN_HPP_NOEXCEPT
  {
    return GeometryFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR GeometryFlagsKHR operator^( GeometryFlagBitsKHR bit0,
                                                                     GeometryFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return GeometryFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR GeometryFlagsKHR operator~( GeometryFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( GeometryFlagsKHR( bits ) );
  }

  using GeometryFlagsNV = GeometryFlagsKHR;

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

  using GeometryInstanceFlagsKHR = Flags<GeometryInstanceFlagBitsKHR>;

  template <>
  struct FlagTraits<GeometryInstanceFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable ) |
                 VkFlags( GeometryInstanceFlagBitsKHR::eTriangleFrontCounterclockwise ) |
                 VkFlags( GeometryInstanceFlagBitsKHR::eForceOpaque ) |
                 VkFlags( GeometryInstanceFlagBitsKHR::eForceNoOpaque )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR GeometryInstanceFlagsKHR
                                         operator|( GeometryInstanceFlagBitsKHR bit0, GeometryInstanceFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return GeometryInstanceFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR GeometryInstanceFlagsKHR
                                         operator&(GeometryInstanceFlagBitsKHR bit0, GeometryInstanceFlagBitsKHR bit1)VULKAN_HPP_NOEXCEPT
  {
    return GeometryInstanceFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR GeometryInstanceFlagsKHR
                                         operator^( GeometryInstanceFlagBitsKHR bit0, GeometryInstanceFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return GeometryInstanceFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR GeometryInstanceFlagsKHR operator~( GeometryInstanceFlagBitsKHR bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( GeometryInstanceFlagsKHR( bits ) );
  }

  using GeometryInstanceFlagsNV = GeometryInstanceFlagsKHR;

  VULKAN_HPP_INLINE std::string to_string( GeometryInstanceFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable )
      result += "TriangleFacingCullDisable | ";
    if ( value & GeometryInstanceFlagBitsKHR::eTriangleFrontCounterclockwise )
      result += "TriangleFrontCounterclockwise | ";
    if ( value & GeometryInstanceFlagBitsKHR::eForceOpaque )
      result += "ForceOpaque | ";
    if ( value & GeometryInstanceFlagBitsKHR::eForceNoOpaque )
      result += "ForceNoOpaque | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using BuildAccelerationStructureFlagsKHR = Flags<BuildAccelerationStructureFlagBitsKHR>;

  template <>
  struct FlagTraits<BuildAccelerationStructureFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( BuildAccelerationStructureFlagBitsKHR::eAllowUpdate ) |
                 VkFlags( BuildAccelerationStructureFlagBitsKHR::eAllowCompaction ) |
                 VkFlags( BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace ) |
                 VkFlags( BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild ) |
                 VkFlags( BuildAccelerationStructureFlagBitsKHR::eLowMemory ) |
                 VkFlags( BuildAccelerationStructureFlagBitsKHR::eMotionNV )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BuildAccelerationStructureFlagsKHR operator|(
    BuildAccelerationStructureFlagBitsKHR bit0, BuildAccelerationStructureFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return BuildAccelerationStructureFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BuildAccelerationStructureFlagsKHR
                                         operator&(BuildAccelerationStructureFlagBitsKHR bit0, BuildAccelerationStructureFlagBitsKHR bit1)VULKAN_HPP_NOEXCEPT
  {
    return BuildAccelerationStructureFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BuildAccelerationStructureFlagsKHR operator^(
    BuildAccelerationStructureFlagBitsKHR bit0, BuildAccelerationStructureFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return BuildAccelerationStructureFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BuildAccelerationStructureFlagsKHR
                                         operator~( BuildAccelerationStructureFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( BuildAccelerationStructureFlagsKHR( bits ) );
  }

  using BuildAccelerationStructureFlagsNV = BuildAccelerationStructureFlagsKHR;

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
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using AccelerationStructureCreateFlagsKHR = Flags<AccelerationStructureCreateFlagBitsKHR>;

  template <>
  struct FlagTraits<AccelerationStructureCreateFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( AccelerationStructureCreateFlagBitsKHR::eDeviceAddressCaptureReplay ) |
                 VkFlags( AccelerationStructureCreateFlagBitsKHR::eMotionNV )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AccelerationStructureCreateFlagsKHR operator|(
    AccelerationStructureCreateFlagBitsKHR bit0, AccelerationStructureCreateFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return AccelerationStructureCreateFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AccelerationStructureCreateFlagsKHR operator&(
    AccelerationStructureCreateFlagBitsKHR bit0, AccelerationStructureCreateFlagBitsKHR bit1)VULKAN_HPP_NOEXCEPT
  {
    return AccelerationStructureCreateFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AccelerationStructureCreateFlagsKHR operator^(
    AccelerationStructureCreateFlagBitsKHR bit0, AccelerationStructureCreateFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return AccelerationStructureCreateFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AccelerationStructureCreateFlagsKHR
                                         operator~( AccelerationStructureCreateFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( AccelerationStructureCreateFlagsKHR( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( AccelerationStructureCreateFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & AccelerationStructureCreateFlagBitsKHR::eDeviceAddressCaptureReplay )
      result += "DeviceAddressCaptureReplay | ";
    if ( value & AccelerationStructureCreateFlagBitsKHR::eMotionNV )
      result += "MotionNV | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_NV_framebuffer_mixed_samples ===

  using PipelineCoverageModulationStateCreateFlagsNV = Flags<PipelineCoverageModulationStateCreateFlagBitsNV>;

  VULKAN_HPP_INLINE std::string to_string( PipelineCoverageModulationStateCreateFlagsNV )
  {
    return "{}";
  }

  //=== VK_EXT_validation_cache ===

  using ValidationCacheCreateFlagsEXT = Flags<ValidationCacheCreateFlagBitsEXT>;

  VULKAN_HPP_INLINE std::string to_string( ValidationCacheCreateFlagsEXT )
  {
    return "{}";
  }

  //=== VK_AMD_pipeline_compiler_control ===

  using PipelineCompilerControlFlagsAMD = Flags<PipelineCompilerControlFlagBitsAMD>;

  VULKAN_HPP_INLINE std::string to_string( PipelineCompilerControlFlagsAMD )
  {
    return "{}";
  }

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_EXT_video_decode_h265 ===

  using VideoDecodeH265CreateFlagsEXT = Flags<VideoDecodeH265CreateFlagBitsEXT>;

  VULKAN_HPP_INLINE std::string to_string( VideoDecodeH265CreateFlagsEXT )
  {
    return "{}";
  }
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_EXT_pipeline_creation_feedback ===

  using PipelineCreationFeedbackFlagsEXT = Flags<PipelineCreationFeedbackFlagBitsEXT>;

  template <>
  struct FlagTraits<PipelineCreationFeedbackFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( PipelineCreationFeedbackFlagBitsEXT::eValid ) |
                 VkFlags( PipelineCreationFeedbackFlagBitsEXT::eApplicationPipelineCacheHit ) |
                 VkFlags( PipelineCreationFeedbackFlagBitsEXT::eBasePipelineAcceleration )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineCreationFeedbackFlagsEXT
                                         operator|( PipelineCreationFeedbackFlagBitsEXT bit0, PipelineCreationFeedbackFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineCreationFeedbackFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineCreationFeedbackFlagsEXT
                                         operator&(PipelineCreationFeedbackFlagBitsEXT bit0, PipelineCreationFeedbackFlagBitsEXT bit1)VULKAN_HPP_NOEXCEPT
  {
    return PipelineCreationFeedbackFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineCreationFeedbackFlagsEXT
                                         operator^( PipelineCreationFeedbackFlagBitsEXT bit0, PipelineCreationFeedbackFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineCreationFeedbackFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineCreationFeedbackFlagsEXT
                                         operator~( PipelineCreationFeedbackFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( PipelineCreationFeedbackFlagsEXT( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineCreationFeedbackFlagsEXT value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & PipelineCreationFeedbackFlagBitsEXT::eValid )
      result += "Valid | ";
    if ( value & PipelineCreationFeedbackFlagBitsEXT::eApplicationPipelineCacheHit )
      result += "ApplicationPipelineCacheHit | ";
    if ( value & PipelineCreationFeedbackFlagBitsEXT::eBasePipelineAcceleration )
      result += "BasePipelineAcceleration | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_imagepipe_surface ===

  using ImagePipeSurfaceCreateFlagsFUCHSIA = Flags<ImagePipeSurfaceCreateFlagBitsFUCHSIA>;

  VULKAN_HPP_INLINE std::string to_string( ImagePipeSurfaceCreateFlagsFUCHSIA )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_surface ===

  using MetalSurfaceCreateFlagsEXT = Flags<MetalSurfaceCreateFlagBitsEXT>;

  VULKAN_HPP_INLINE std::string to_string( MetalSurfaceCreateFlagsEXT )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_AMD_shader_core_properties2 ===

  using ShaderCorePropertiesFlagsAMD = Flags<ShaderCorePropertiesFlagBitsAMD>;

  VULKAN_HPP_INLINE std::string to_string( ShaderCorePropertiesFlagsAMD )
  {
    return "{}";
  }

  //=== VK_EXT_tooling_info ===

  using ToolPurposeFlagsEXT = Flags<ToolPurposeFlagBitsEXT>;

  template <>
  struct FlagTraits<ToolPurposeFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ToolPurposeFlagBitsEXT::eValidation ) | VkFlags( ToolPurposeFlagBitsEXT::eProfiling ) |
                 VkFlags( ToolPurposeFlagBitsEXT::eTracing ) | VkFlags( ToolPurposeFlagBitsEXT::eAdditionalFeatures ) |
                 VkFlags( ToolPurposeFlagBitsEXT::eModifyingFeatures ) |
                 VkFlags( ToolPurposeFlagBitsEXT::eDebugReporting ) | VkFlags( ToolPurposeFlagBitsEXT::eDebugMarkers )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ToolPurposeFlagsEXT
                                         operator|( ToolPurposeFlagBitsEXT bit0, ToolPurposeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ToolPurposeFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ToolPurposeFlagsEXT operator&(ToolPurposeFlagBitsEXT bit0,
                                                                       ToolPurposeFlagBitsEXT bit1)VULKAN_HPP_NOEXCEPT
  {
    return ToolPurposeFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ToolPurposeFlagsEXT
                                         operator^( ToolPurposeFlagBitsEXT bit0, ToolPurposeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ToolPurposeFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ToolPurposeFlagsEXT operator~( ToolPurposeFlagBitsEXT bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( ToolPurposeFlagsEXT( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( ToolPurposeFlagsEXT value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & ToolPurposeFlagBitsEXT::eValidation )
      result += "Validation | ";
    if ( value & ToolPurposeFlagBitsEXT::eProfiling )
      result += "Profiling | ";
    if ( value & ToolPurposeFlagBitsEXT::eTracing )
      result += "Tracing | ";
    if ( value & ToolPurposeFlagBitsEXT::eAdditionalFeatures )
      result += "AdditionalFeatures | ";
    if ( value & ToolPurposeFlagBitsEXT::eModifyingFeatures )
      result += "ModifyingFeatures | ";
    if ( value & ToolPurposeFlagBitsEXT::eDebugReporting )
      result += "DebugReporting | ";
    if ( value & ToolPurposeFlagBitsEXT::eDebugMarkers )
      result += "DebugMarkers | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_NV_coverage_reduction_mode ===

  using PipelineCoverageReductionStateCreateFlagsNV = Flags<PipelineCoverageReductionStateCreateFlagBitsNV>;

  VULKAN_HPP_INLINE std::string to_string( PipelineCoverageReductionStateCreateFlagsNV )
  {
    return "{}";
  }

  //=== VK_EXT_headless_surface ===

  using HeadlessSurfaceCreateFlagsEXT = Flags<HeadlessSurfaceCreateFlagBitsEXT>;

  VULKAN_HPP_INLINE std::string to_string( HeadlessSurfaceCreateFlagsEXT )
  {
    return "{}";
  }

  //=== VK_NV_device_generated_commands ===

  using IndirectStateFlagsNV = Flags<IndirectStateFlagBitsNV>;

  template <>
  struct FlagTraits<IndirectStateFlagBitsNV>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( IndirectStateFlagBitsNV::eFlagFrontface )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR IndirectStateFlagsNV
                                         operator|( IndirectStateFlagBitsNV bit0, IndirectStateFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return IndirectStateFlagsNV( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR IndirectStateFlagsNV operator&(IndirectStateFlagBitsNV bit0,
                                                                        IndirectStateFlagBitsNV bit1)VULKAN_HPP_NOEXCEPT
  {
    return IndirectStateFlagsNV( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR IndirectStateFlagsNV
                                         operator^( IndirectStateFlagBitsNV bit0, IndirectStateFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return IndirectStateFlagsNV( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR IndirectStateFlagsNV operator~( IndirectStateFlagBitsNV bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( IndirectStateFlagsNV( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( IndirectStateFlagsNV value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & IndirectStateFlagBitsNV::eFlagFrontface )
      result += "FlagFrontface | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using IndirectCommandsLayoutUsageFlagsNV = Flags<IndirectCommandsLayoutUsageFlagBitsNV>;

  template <>
  struct FlagTraits<IndirectCommandsLayoutUsageFlagBitsNV>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( IndirectCommandsLayoutUsageFlagBitsNV::eExplicitPreprocess ) |
                 VkFlags( IndirectCommandsLayoutUsageFlagBitsNV::eIndexedSequences ) |
                 VkFlags( IndirectCommandsLayoutUsageFlagBitsNV::eUnorderedSequences )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR IndirectCommandsLayoutUsageFlagsNV operator|(
    IndirectCommandsLayoutUsageFlagBitsNV bit0, IndirectCommandsLayoutUsageFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return IndirectCommandsLayoutUsageFlagsNV( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR IndirectCommandsLayoutUsageFlagsNV
                                         operator&(IndirectCommandsLayoutUsageFlagBitsNV bit0, IndirectCommandsLayoutUsageFlagBitsNV bit1)VULKAN_HPP_NOEXCEPT
  {
    return IndirectCommandsLayoutUsageFlagsNV( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR IndirectCommandsLayoutUsageFlagsNV operator^(
    IndirectCommandsLayoutUsageFlagBitsNV bit0, IndirectCommandsLayoutUsageFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return IndirectCommandsLayoutUsageFlagsNV( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR IndirectCommandsLayoutUsageFlagsNV
                                         operator~( IndirectCommandsLayoutUsageFlagBitsNV bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( IndirectCommandsLayoutUsageFlagsNV( bits ) );
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

  using DeviceMemoryReportFlagsEXT = Flags<DeviceMemoryReportFlagBitsEXT>;

  VULKAN_HPP_INLINE std::string to_string( DeviceMemoryReportFlagsEXT )
  {
    return "{}";
  }

  //=== VK_EXT_private_data ===

  using PrivateDataSlotCreateFlagsEXT = Flags<PrivateDataSlotCreateFlagBitsEXT>;

  VULKAN_HPP_INLINE std::string to_string( PrivateDataSlotCreateFlagsEXT )
  {
    return "{}";
  }

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_KHR_video_encode_queue ===

  using VideoEncodeFlagsKHR = Flags<VideoEncodeFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoEncodeFlagBitsKHR::eDefault ) | VkFlags( VideoEncodeFlagBitsKHR::eReserved0 )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeFlagsKHR
                                         operator|( VideoEncodeFlagBitsKHR bit0, VideoEncodeFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeFlagsKHR operator&(VideoEncodeFlagBitsKHR bit0,
                                                                       VideoEncodeFlagBitsKHR bit1)VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeFlagsKHR
                                         operator^( VideoEncodeFlagBitsKHR bit0, VideoEncodeFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeFlagsKHR operator~( VideoEncodeFlagBitsKHR bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoEncodeFlagsKHR( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoEncodeFlagBitsKHR::eReserved0 )
      result += "Reserved0 | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using VideoEncodeRateControlFlagsKHR = Flags<VideoEncodeRateControlFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeRateControlFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags =
        VkFlags( VideoEncodeRateControlFlagBitsKHR::eDefault ) | VkFlags( VideoEncodeRateControlFlagBitsKHR::eReset )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeRateControlFlagsKHR
                                         operator|( VideoEncodeRateControlFlagBitsKHR bit0, VideoEncodeRateControlFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeRateControlFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeRateControlFlagsKHR
                                         operator&(VideoEncodeRateControlFlagBitsKHR bit0, VideoEncodeRateControlFlagBitsKHR bit1)VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeRateControlFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeRateControlFlagsKHR
                                         operator^( VideoEncodeRateControlFlagBitsKHR bit0, VideoEncodeRateControlFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeRateControlFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeRateControlFlagsKHR
                                         operator~( VideoEncodeRateControlFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoEncodeRateControlFlagsKHR( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeRateControlFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & VideoEncodeRateControlFlagBitsKHR::eReset )
      result += "Reset | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using VideoEncodeRateControlModeFlagsKHR = Flags<VideoEncodeRateControlModeFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeRateControlModeFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoEncodeRateControlModeFlagBitsKHR::eNone ) |
                 VkFlags( VideoEncodeRateControlModeFlagBitsKHR::eCbr ) |
                 VkFlags( VideoEncodeRateControlModeFlagBitsKHR::eVbr )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeRateControlModeFlagsKHR operator|(
    VideoEncodeRateControlModeFlagBitsKHR bit0, VideoEncodeRateControlModeFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeRateControlModeFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeRateControlModeFlagsKHR
                                         operator&(VideoEncodeRateControlModeFlagBitsKHR bit0, VideoEncodeRateControlModeFlagBitsKHR bit1)VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeRateControlModeFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeRateControlModeFlagsKHR operator^(
    VideoEncodeRateControlModeFlagBitsKHR bit0, VideoEncodeRateControlModeFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeRateControlModeFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeRateControlModeFlagsKHR
                                         operator~( VideoEncodeRateControlModeFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoEncodeRateControlModeFlagsKHR( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( VideoEncodeRateControlModeFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_NV_device_diagnostics_config ===

  using DeviceDiagnosticsConfigFlagsNV = Flags<DeviceDiagnosticsConfigFlagBitsNV>;

  template <>
  struct FlagTraits<DeviceDiagnosticsConfigFlagBitsNV>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( DeviceDiagnosticsConfigFlagBitsNV::eEnableShaderDebugInfo ) |
                 VkFlags( DeviceDiagnosticsConfigFlagBitsNV::eEnableResourceTracking ) |
                 VkFlags( DeviceDiagnosticsConfigFlagBitsNV::eEnableAutomaticCheckpoints )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceDiagnosticsConfigFlagsNV
                                         operator|( DeviceDiagnosticsConfigFlagBitsNV bit0, DeviceDiagnosticsConfigFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DeviceDiagnosticsConfigFlagsNV( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceDiagnosticsConfigFlagsNV
                                         operator&(DeviceDiagnosticsConfigFlagBitsNV bit0, DeviceDiagnosticsConfigFlagBitsNV bit1)VULKAN_HPP_NOEXCEPT
  {
    return DeviceDiagnosticsConfigFlagsNV( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceDiagnosticsConfigFlagsNV
                                         operator^( DeviceDiagnosticsConfigFlagBitsNV bit0, DeviceDiagnosticsConfigFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DeviceDiagnosticsConfigFlagsNV( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceDiagnosticsConfigFlagsNV
                                         operator~( DeviceDiagnosticsConfigFlagBitsNV bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( DeviceDiagnosticsConfigFlagsNV( bits ) );
  }

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
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_KHR_synchronization2 ===

  using PipelineStageFlags2KHR = Flags<PipelineStageFlagBits2KHR>;

  template <>
  struct FlagTraits<PipelineStageFlagBits2KHR>
  {
    enum : VkFlags64
    {
      allFlags =
        VkFlags64( PipelineStageFlagBits2KHR::eNone ) | VkFlags64( PipelineStageFlagBits2KHR::eTopOfPipe ) |
        VkFlags64( PipelineStageFlagBits2KHR::eDrawIndirect ) | VkFlags64( PipelineStageFlagBits2KHR::eVertexInput ) |
        VkFlags64( PipelineStageFlagBits2KHR::eVertexShader ) |
        VkFlags64( PipelineStageFlagBits2KHR::eTessellationControlShader ) |
        VkFlags64( PipelineStageFlagBits2KHR::eTessellationEvaluationShader ) |
        VkFlags64( PipelineStageFlagBits2KHR::eGeometryShader ) |
        VkFlags64( PipelineStageFlagBits2KHR::eFragmentShader ) |
        VkFlags64( PipelineStageFlagBits2KHR::eEarlyFragmentTests ) |
        VkFlags64( PipelineStageFlagBits2KHR::eLateFragmentTests ) |
        VkFlags64( PipelineStageFlagBits2KHR::eColorAttachmentOutput ) |
        VkFlags64( PipelineStageFlagBits2KHR::eComputeShader ) | VkFlags64( PipelineStageFlagBits2KHR::eAllTransfer ) |
        VkFlags64( PipelineStageFlagBits2KHR::eBottomOfPipe ) | VkFlags64( PipelineStageFlagBits2KHR::eHost ) |
        VkFlags64( PipelineStageFlagBits2KHR::eAllGraphics ) | VkFlags64( PipelineStageFlagBits2KHR::eAllCommands ) |
        VkFlags64( PipelineStageFlagBits2KHR::eCopy ) | VkFlags64( PipelineStageFlagBits2KHR::eResolve ) |
        VkFlags64( PipelineStageFlagBits2KHR::eBlit ) | VkFlags64( PipelineStageFlagBits2KHR::eClear ) |
        VkFlags64( PipelineStageFlagBits2KHR::eIndexInput ) |
        VkFlags64( PipelineStageFlagBits2KHR::eVertexAttributeInput ) |
        VkFlags64( PipelineStageFlagBits2KHR::ePreRasterizationShaders )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
        | VkFlags64( PipelineStageFlagBits2KHR::eVideoDecode ) | VkFlags64( PipelineStageFlagBits2KHR::eVideoEncode )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
        | VkFlags64( PipelineStageFlagBits2KHR::eTransformFeedbackEXT ) |
        VkFlags64( PipelineStageFlagBits2KHR::eConditionalRenderingEXT ) |
        VkFlags64( PipelineStageFlagBits2KHR::eCommandPreprocessNV ) |
        VkFlags64( PipelineStageFlagBits2KHR::eFragmentShadingRateAttachment ) |
        VkFlags64( PipelineStageFlagBits2KHR::eAccelerationStructureBuild ) |
        VkFlags64( PipelineStageFlagBits2KHR::eRayTracingShader ) |
        VkFlags64( PipelineStageFlagBits2KHR::eFragmentDensityProcessEXT ) |
        VkFlags64( PipelineStageFlagBits2KHR::eTaskShaderNV ) | VkFlags64( PipelineStageFlagBits2KHR::eMeshShaderNV ) |
        VkFlags64( PipelineStageFlagBits2KHR::eSubpassShadingHUAWEI )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineStageFlags2KHR
                                         operator|( PipelineStageFlagBits2KHR bit0, PipelineStageFlagBits2KHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineStageFlags2KHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineStageFlags2KHR
                                         operator&(PipelineStageFlagBits2KHR bit0, PipelineStageFlagBits2KHR bit1)VULKAN_HPP_NOEXCEPT
  {
    return PipelineStageFlags2KHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineStageFlags2KHR
                                         operator^( PipelineStageFlagBits2KHR bit0, PipelineStageFlagBits2KHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineStageFlags2KHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineStageFlags2KHR operator~( PipelineStageFlagBits2KHR bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( PipelineStageFlags2KHR( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( PipelineStageFlags2KHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & PipelineStageFlagBits2KHR::eTopOfPipe )
      result += "TopOfPipe | ";
    if ( value & PipelineStageFlagBits2KHR::eDrawIndirect )
      result += "DrawIndirect | ";
    if ( value & PipelineStageFlagBits2KHR::eVertexInput )
      result += "VertexInput | ";
    if ( value & PipelineStageFlagBits2KHR::eVertexShader )
      result += "VertexShader | ";
    if ( value & PipelineStageFlagBits2KHR::eTessellationControlShader )
      result += "TessellationControlShader | ";
    if ( value & PipelineStageFlagBits2KHR::eTessellationEvaluationShader )
      result += "TessellationEvaluationShader | ";
    if ( value & PipelineStageFlagBits2KHR::eGeometryShader )
      result += "GeometryShader | ";
    if ( value & PipelineStageFlagBits2KHR::eFragmentShader )
      result += "FragmentShader | ";
    if ( value & PipelineStageFlagBits2KHR::eEarlyFragmentTests )
      result += "EarlyFragmentTests | ";
    if ( value & PipelineStageFlagBits2KHR::eLateFragmentTests )
      result += "LateFragmentTests | ";
    if ( value & PipelineStageFlagBits2KHR::eColorAttachmentOutput )
      result += "ColorAttachmentOutput | ";
    if ( value & PipelineStageFlagBits2KHR::eComputeShader )
      result += "ComputeShader | ";
    if ( value & PipelineStageFlagBits2KHR::eAllTransfer )
      result += "AllTransfer | ";
    if ( value & PipelineStageFlagBits2KHR::eBottomOfPipe )
      result += "BottomOfPipe | ";
    if ( value & PipelineStageFlagBits2KHR::eHost )
      result += "Host | ";
    if ( value & PipelineStageFlagBits2KHR::eAllGraphics )
      result += "AllGraphics | ";
    if ( value & PipelineStageFlagBits2KHR::eAllCommands )
      result += "AllCommands | ";
    if ( value & PipelineStageFlagBits2KHR::eCopy )
      result += "Copy | ";
    if ( value & PipelineStageFlagBits2KHR::eResolve )
      result += "Resolve | ";
    if ( value & PipelineStageFlagBits2KHR::eBlit )
      result += "Blit | ";
    if ( value & PipelineStageFlagBits2KHR::eClear )
      result += "Clear | ";
    if ( value & PipelineStageFlagBits2KHR::eIndexInput )
      result += "IndexInput | ";
    if ( value & PipelineStageFlagBits2KHR::eVertexAttributeInput )
      result += "VertexAttributeInput | ";
    if ( value & PipelineStageFlagBits2KHR::ePreRasterizationShaders )
      result += "PreRasterizationShaders | ";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & PipelineStageFlagBits2KHR::eVideoDecode )
      result += "VideoDecode | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & PipelineStageFlagBits2KHR::eVideoEncode )
      result += "VideoEncode | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    if ( value & PipelineStageFlagBits2KHR::eTransformFeedbackEXT )
      result += "TransformFeedbackEXT | ";
    if ( value & PipelineStageFlagBits2KHR::eConditionalRenderingEXT )
      result += "ConditionalRenderingEXT | ";
    if ( value & PipelineStageFlagBits2KHR::eCommandPreprocessNV )
      result += "CommandPreprocessNV | ";
    if ( value & PipelineStageFlagBits2KHR::eFragmentShadingRateAttachment )
      result += "FragmentShadingRateAttachment | ";
    if ( value & PipelineStageFlagBits2KHR::eAccelerationStructureBuild )
      result += "AccelerationStructureBuild | ";
    if ( value & PipelineStageFlagBits2KHR::eRayTracingShader )
      result += "RayTracingShader | ";
    if ( value & PipelineStageFlagBits2KHR::eFragmentDensityProcessEXT )
      result += "FragmentDensityProcessEXT | ";
    if ( value & PipelineStageFlagBits2KHR::eTaskShaderNV )
      result += "TaskShaderNV | ";
    if ( value & PipelineStageFlagBits2KHR::eMeshShaderNV )
      result += "MeshShaderNV | ";
    if ( value & PipelineStageFlagBits2KHR::eSubpassShadingHUAWEI )
      result += "SubpassShadingHUAWEI | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using AccessFlags2KHR = Flags<AccessFlagBits2KHR>;

  template <>
  struct FlagTraits<AccessFlagBits2KHR>
  {
    enum : VkFlags64
    {
      allFlags =
        VkFlags64( AccessFlagBits2KHR::eNone ) | VkFlags64( AccessFlagBits2KHR::eIndirectCommandRead ) |
        VkFlags64( AccessFlagBits2KHR::eIndexRead ) | VkFlags64( AccessFlagBits2KHR::eVertexAttributeRead ) |
        VkFlags64( AccessFlagBits2KHR::eUniformRead ) | VkFlags64( AccessFlagBits2KHR::eInputAttachmentRead ) |
        VkFlags64( AccessFlagBits2KHR::eShaderRead ) | VkFlags64( AccessFlagBits2KHR::eShaderWrite ) |
        VkFlags64( AccessFlagBits2KHR::eColorAttachmentRead ) | VkFlags64( AccessFlagBits2KHR::eColorAttachmentWrite ) |
        VkFlags64( AccessFlagBits2KHR::eDepthStencilAttachmentRead ) |
        VkFlags64( AccessFlagBits2KHR::eDepthStencilAttachmentWrite ) | VkFlags64( AccessFlagBits2KHR::eTransferRead ) |
        VkFlags64( AccessFlagBits2KHR::eTransferWrite ) | VkFlags64( AccessFlagBits2KHR::eHostRead ) |
        VkFlags64( AccessFlagBits2KHR::eHostWrite ) | VkFlags64( AccessFlagBits2KHR::eMemoryRead ) |
        VkFlags64( AccessFlagBits2KHR::eMemoryWrite ) | VkFlags64( AccessFlagBits2KHR::eShaderSampledRead ) |
        VkFlags64( AccessFlagBits2KHR::eShaderStorageRead ) | VkFlags64( AccessFlagBits2KHR::eShaderStorageWrite )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
        | VkFlags64( AccessFlagBits2KHR::eVideoDecodeRead ) | VkFlags64( AccessFlagBits2KHR::eVideoDecodeWrite ) |
        VkFlags64( AccessFlagBits2KHR::eVideoEncodeRead ) | VkFlags64( AccessFlagBits2KHR::eVideoEncodeWrite )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
        | VkFlags64( AccessFlagBits2KHR::eTransformFeedbackWriteEXT ) |
        VkFlags64( AccessFlagBits2KHR::eTransformFeedbackCounterReadEXT ) |
        VkFlags64( AccessFlagBits2KHR::eTransformFeedbackCounterWriteEXT ) |
        VkFlags64( AccessFlagBits2KHR::eConditionalRenderingReadEXT ) |
        VkFlags64( AccessFlagBits2KHR::eCommandPreprocessReadNV ) |
        VkFlags64( AccessFlagBits2KHR::eCommandPreprocessWriteNV ) |
        VkFlags64( AccessFlagBits2KHR::eFragmentShadingRateAttachmentRead ) |
        VkFlags64( AccessFlagBits2KHR::eAccelerationStructureRead ) |
        VkFlags64( AccessFlagBits2KHR::eAccelerationStructureWrite ) |
        VkFlags64( AccessFlagBits2KHR::eFragmentDensityMapReadEXT ) |
        VkFlags64( AccessFlagBits2KHR::eColorAttachmentReadNoncoherentEXT )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AccessFlags2KHR operator|( AccessFlagBits2KHR bit0,
                                                                    AccessFlagBits2KHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return AccessFlags2KHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AccessFlags2KHR operator&(AccessFlagBits2KHR bit0,
                                                                   AccessFlagBits2KHR bit1)VULKAN_HPP_NOEXCEPT
  {
    return AccessFlags2KHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AccessFlags2KHR operator^( AccessFlagBits2KHR bit0,
                                                                    AccessFlagBits2KHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return AccessFlags2KHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AccessFlags2KHR operator~( AccessFlagBits2KHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( AccessFlags2KHR( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( AccessFlags2KHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & AccessFlagBits2KHR::eIndirectCommandRead )
      result += "IndirectCommandRead | ";
    if ( value & AccessFlagBits2KHR::eIndexRead )
      result += "IndexRead | ";
    if ( value & AccessFlagBits2KHR::eVertexAttributeRead )
      result += "VertexAttributeRead | ";
    if ( value & AccessFlagBits2KHR::eUniformRead )
      result += "UniformRead | ";
    if ( value & AccessFlagBits2KHR::eInputAttachmentRead )
      result += "InputAttachmentRead | ";
    if ( value & AccessFlagBits2KHR::eShaderRead )
      result += "ShaderRead | ";
    if ( value & AccessFlagBits2KHR::eShaderWrite )
      result += "ShaderWrite | ";
    if ( value & AccessFlagBits2KHR::eColorAttachmentRead )
      result += "ColorAttachmentRead | ";
    if ( value & AccessFlagBits2KHR::eColorAttachmentWrite )
      result += "ColorAttachmentWrite | ";
    if ( value & AccessFlagBits2KHR::eDepthStencilAttachmentRead )
      result += "DepthStencilAttachmentRead | ";
    if ( value & AccessFlagBits2KHR::eDepthStencilAttachmentWrite )
      result += "DepthStencilAttachmentWrite | ";
    if ( value & AccessFlagBits2KHR::eTransferRead )
      result += "TransferRead | ";
    if ( value & AccessFlagBits2KHR::eTransferWrite )
      result += "TransferWrite | ";
    if ( value & AccessFlagBits2KHR::eHostRead )
      result += "HostRead | ";
    if ( value & AccessFlagBits2KHR::eHostWrite )
      result += "HostWrite | ";
    if ( value & AccessFlagBits2KHR::eMemoryRead )
      result += "MemoryRead | ";
    if ( value & AccessFlagBits2KHR::eMemoryWrite )
      result += "MemoryWrite | ";
    if ( value & AccessFlagBits2KHR::eShaderSampledRead )
      result += "ShaderSampledRead | ";
    if ( value & AccessFlagBits2KHR::eShaderStorageRead )
      result += "ShaderStorageRead | ";
    if ( value & AccessFlagBits2KHR::eShaderStorageWrite )
      result += "ShaderStorageWrite | ";
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & AccessFlagBits2KHR::eVideoDecodeRead )
      result += "VideoDecodeRead | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & AccessFlagBits2KHR::eVideoDecodeWrite )
      result += "VideoDecodeWrite | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & AccessFlagBits2KHR::eVideoEncodeRead )
      result += "VideoEncodeRead | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    if ( value & AccessFlagBits2KHR::eVideoEncodeWrite )
      result += "VideoEncodeWrite | ";
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    if ( value & AccessFlagBits2KHR::eTransformFeedbackWriteEXT )
      result += "TransformFeedbackWriteEXT | ";
    if ( value & AccessFlagBits2KHR::eTransformFeedbackCounterReadEXT )
      result += "TransformFeedbackCounterReadEXT | ";
    if ( value & AccessFlagBits2KHR::eTransformFeedbackCounterWriteEXT )
      result += "TransformFeedbackCounterWriteEXT | ";
    if ( value & AccessFlagBits2KHR::eConditionalRenderingReadEXT )
      result += "ConditionalRenderingReadEXT | ";
    if ( value & AccessFlagBits2KHR::eCommandPreprocessReadNV )
      result += "CommandPreprocessReadNV | ";
    if ( value & AccessFlagBits2KHR::eCommandPreprocessWriteNV )
      result += "CommandPreprocessWriteNV | ";
    if ( value & AccessFlagBits2KHR::eFragmentShadingRateAttachmentRead )
      result += "FragmentShadingRateAttachmentRead | ";
    if ( value & AccessFlagBits2KHR::eAccelerationStructureRead )
      result += "AccelerationStructureRead | ";
    if ( value & AccessFlagBits2KHR::eAccelerationStructureWrite )
      result += "AccelerationStructureWrite | ";
    if ( value & AccessFlagBits2KHR::eFragmentDensityMapReadEXT )
      result += "FragmentDensityMapReadEXT | ";
    if ( value & AccessFlagBits2KHR::eColorAttachmentReadNoncoherentEXT )
      result += "ColorAttachmentReadNoncoherentEXT | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  using SubmitFlagsKHR = Flags<SubmitFlagBitsKHR>;

  template <>
  struct FlagTraits<SubmitFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( SubmitFlagBitsKHR::eProtected )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SubmitFlagsKHR operator|( SubmitFlagBitsKHR bit0,
                                                                   SubmitFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SubmitFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SubmitFlagsKHR operator&(SubmitFlagBitsKHR bit0,
                                                                  SubmitFlagBitsKHR bit1)VULKAN_HPP_NOEXCEPT
  {
    return SubmitFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SubmitFlagsKHR operator^( SubmitFlagBitsKHR bit0,
                                                                   SubmitFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SubmitFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SubmitFlagsKHR operator~( SubmitFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( SubmitFlagsKHR( bits ) );
  }

  VULKAN_HPP_INLINE std::string to_string( SubmitFlagsKHR value )
  {
    if ( !value )
      return "{}";

    std::string result;
    if ( value & SubmitFlagBitsKHR::eProtected )
      result += "Protected | ";
    return "{ " + result.substr( 0, result.size() - 3 ) + " }";
  }

  //=== VK_NV_ray_tracing_motion_blur ===

  using AccelerationStructureMotionInfoFlagsNV = Flags<AccelerationStructureMotionInfoFlagBitsNV>;

  VULKAN_HPP_INLINE std::string to_string( AccelerationStructureMotionInfoFlagsNV )
  {
    return "{}";
  }

  using AccelerationStructureMotionInstanceFlagsNV = Flags<AccelerationStructureMotionInstanceFlagBitsNV>;

  VULKAN_HPP_INLINE std::string to_string( AccelerationStructureMotionInstanceFlagsNV )
  {
    return "{}";
  }

#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
  //=== VK_EXT_directfb_surface ===

  using DirectFBSurfaceCreateFlagsEXT = Flags<DirectFBSurfaceCreateFlagBitsEXT>;

  VULKAN_HPP_INLINE std::string to_string( DirectFBSurfaceCreateFlagsEXT )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
  //=== VK_QNX_screen_surface ===

  using ScreenSurfaceCreateFlagsQNX = Flags<ScreenSurfaceCreateFlagBitsQNX>;

  VULKAN_HPP_INLINE std::string to_string( ScreenSurfaceCreateFlagsQNX )
  {
    return "{}";
  }
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

}  // namespace VULKAN_HPP_NAMESPACE
#endif
