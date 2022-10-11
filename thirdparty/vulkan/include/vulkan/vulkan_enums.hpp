// Copyright 2015-2022 The Khronos Group Inc.
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
  {
  };

  template <typename Type>
  struct isVulkanHandleType
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool value = false;
  };

  //=============
  //=== ENUMs ===
  //=============

  //=== VK_VERSION_1_0 ===

  enum class Result
  {
    eSuccess                          = VK_SUCCESS,
    eNotReady                         = VK_NOT_READY,
    eTimeout                          = VK_TIMEOUT,
    eEventSet                         = VK_EVENT_SET,
    eEventReset                       = VK_EVENT_RESET,
    eIncomplete                       = VK_INCOMPLETE,
    eErrorOutOfHostMemory             = VK_ERROR_OUT_OF_HOST_MEMORY,
    eErrorOutOfDeviceMemory           = VK_ERROR_OUT_OF_DEVICE_MEMORY,
    eErrorInitializationFailed        = VK_ERROR_INITIALIZATION_FAILED,
    eErrorDeviceLost                  = VK_ERROR_DEVICE_LOST,
    eErrorMemoryMapFailed             = VK_ERROR_MEMORY_MAP_FAILED,
    eErrorLayerNotPresent             = VK_ERROR_LAYER_NOT_PRESENT,
    eErrorExtensionNotPresent         = VK_ERROR_EXTENSION_NOT_PRESENT,
    eErrorFeatureNotPresent           = VK_ERROR_FEATURE_NOT_PRESENT,
    eErrorIncompatibleDriver          = VK_ERROR_INCOMPATIBLE_DRIVER,
    eErrorTooManyObjects              = VK_ERROR_TOO_MANY_OBJECTS,
    eErrorFormatNotSupported          = VK_ERROR_FORMAT_NOT_SUPPORTED,
    eErrorFragmentedPool              = VK_ERROR_FRAGMENTED_POOL,
    eErrorUnknown                     = VK_ERROR_UNKNOWN,
    eErrorOutOfPoolMemory             = VK_ERROR_OUT_OF_POOL_MEMORY,
    eErrorInvalidExternalHandle       = VK_ERROR_INVALID_EXTERNAL_HANDLE,
    eErrorFragmentation               = VK_ERROR_FRAGMENTATION,
    eErrorInvalidOpaqueCaptureAddress = VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS,
    ePipelineCompileRequired          = VK_PIPELINE_COMPILE_REQUIRED,
    eErrorSurfaceLostKHR              = VK_ERROR_SURFACE_LOST_KHR,
    eErrorNativeWindowInUseKHR        = VK_ERROR_NATIVE_WINDOW_IN_USE_KHR,
    eSuboptimalKHR                    = VK_SUBOPTIMAL_KHR,
    eErrorOutOfDateKHR                = VK_ERROR_OUT_OF_DATE_KHR,
    eErrorIncompatibleDisplayKHR      = VK_ERROR_INCOMPATIBLE_DISPLAY_KHR,
    eErrorValidationFailedEXT         = VK_ERROR_VALIDATION_FAILED_EXT,
    eErrorInvalidShaderNV             = VK_ERROR_INVALID_SHADER_NV,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eErrorImageUsageNotSupportedKHR            = VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR,
    eErrorVideoPictureLayoutNotSupportedKHR    = VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR,
    eErrorVideoProfileOperationNotSupportedKHR = VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR,
    eErrorVideoProfileFormatNotSupportedKHR    = VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR,
    eErrorVideoProfileCodecNotSupportedKHR     = VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR,
    eErrorVideoStdVersionNotSupportedKHR       = VK_ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eErrorInvalidDrmFormatModifierPlaneLayoutEXT = VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT,
    eErrorNotPermittedKHR                        = VK_ERROR_NOT_PERMITTED_KHR,
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    eErrorFullScreenExclusiveModeLostEXT = VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT,
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    eThreadIdleKHR                       = VK_THREAD_IDLE_KHR,
    eThreadDoneKHR                       = VK_THREAD_DONE_KHR,
    eOperationDeferredKHR                = VK_OPERATION_DEFERRED_KHR,
    eOperationNotDeferredKHR             = VK_OPERATION_NOT_DEFERRED_KHR,
    eErrorCompressionExhaustedEXT        = VK_ERROR_COMPRESSION_EXHAUSTED_EXT,
    eErrorFragmentationEXT               = VK_ERROR_FRAGMENTATION_EXT,
    eErrorInvalidDeviceAddressEXT        = VK_ERROR_INVALID_DEVICE_ADDRESS_EXT,
    eErrorInvalidExternalHandleKHR       = VK_ERROR_INVALID_EXTERNAL_HANDLE_KHR,
    eErrorInvalidOpaqueCaptureAddressKHR = VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR,
    eErrorNotPermittedEXT                = VK_ERROR_NOT_PERMITTED_EXT,
    eErrorOutOfPoolMemoryKHR             = VK_ERROR_OUT_OF_POOL_MEMORY_KHR,
    eErrorPipelineCompileRequiredEXT     = VK_ERROR_PIPELINE_COMPILE_REQUIRED_EXT,
    ePipelineCompileRequiredEXT          = VK_PIPELINE_COMPILE_REQUIRED_EXT
  };

  enum class StructureType
  {
    eApplicationInfo                                      = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    eInstanceCreateInfo                                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    eDeviceQueueCreateInfo                                = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    eDeviceCreateInfo                                     = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    eSubmitInfo                                           = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    eMemoryAllocateInfo                                   = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    eMappedMemoryRange                                    = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
    eBindSparseInfo                                       = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
    eFenceCreateInfo                                      = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    eSemaphoreCreateInfo                                  = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    eEventCreateInfo                                      = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO,
    eQueryPoolCreateInfo                                  = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
    eBufferCreateInfo                                     = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    eBufferViewCreateInfo                                 = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
    eImageCreateInfo                                      = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
    eImageViewCreateInfo                                  = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
    eShaderModuleCreateInfo                               = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    ePipelineCacheCreateInfo                              = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
    ePipelineShaderStageCreateInfo                        = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    ePipelineVertexInputStateCreateInfo                   = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    ePipelineInputAssemblyStateCreateInfo                 = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
    ePipelineTessellationStateCreateInfo                  = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
    ePipelineViewportStateCreateInfo                      = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
    ePipelineRasterizationStateCreateInfo                 = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
    ePipelineMultisampleStateCreateInfo                   = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
    ePipelineDepthStencilStateCreateInfo                  = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
    ePipelineColorBlendStateCreateInfo                    = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
    ePipelineDynamicStateCreateInfo                       = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
    eGraphicsPipelineCreateInfo                           = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
    eComputePipelineCreateInfo                            = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    ePipelineLayoutCreateInfo                             = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    eSamplerCreateInfo                                    = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
    eDescriptorSetLayoutCreateInfo                        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    eDescriptorPoolCreateInfo                             = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    eDescriptorSetAllocateInfo                            = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    eWriteDescriptorSet                                   = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    eCopyDescriptorSet                                    = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET,
    eFramebufferCreateInfo                                = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
    eRenderPassCreateInfo                                 = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
    eCommandPoolCreateInfo                                = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    eCommandBufferAllocateInfo                            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    eCommandBufferInheritanceInfo                         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
    eCommandBufferBeginInfo                               = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    eRenderPassBeginInfo                                  = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
    eBufferMemoryBarrier                                  = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
    eImageMemoryBarrier                                   = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
    eMemoryBarrier                                        = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    eLoaderInstanceCreateInfo                             = VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO,
    eLoaderDeviceCreateInfo                               = VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO,
    ePhysicalDeviceSubgroupProperties                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    eBindBufferMemoryInfo                                 = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
    eBindImageMemoryInfo                                  = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,
    ePhysicalDevice16BitStorageFeatures                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
    eMemoryDedicatedRequirements                          = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
    eMemoryDedicatedAllocateInfo                          = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
    eMemoryAllocateFlagsInfo                              = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
    eDeviceGroupRenderPassBeginInfo                       = VK_STRUCTURE_TYPE_DEVICE_GROUP_RENDER_PASS_BEGIN_INFO,
    eDeviceGroupCommandBufferBeginInfo                    = VK_STRUCTURE_TYPE_DEVICE_GROUP_COMMAND_BUFFER_BEGIN_INFO,
    eDeviceGroupSubmitInfo                                = VK_STRUCTURE_TYPE_DEVICE_GROUP_SUBMIT_INFO,
    eDeviceGroupBindSparseInfo                            = VK_STRUCTURE_TYPE_DEVICE_GROUP_BIND_SPARSE_INFO,
    eBindBufferMemoryDeviceGroupInfo                      = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_DEVICE_GROUP_INFO,
    eBindImageMemoryDeviceGroupInfo                       = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_DEVICE_GROUP_INFO,
    ePhysicalDeviceGroupProperties                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES,
    eDeviceGroupDeviceCreateInfo                          = VK_STRUCTURE_TYPE_DEVICE_GROUP_DEVICE_CREATE_INFO,
    eBufferMemoryRequirementsInfo2                        = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
    eImageMemoryRequirementsInfo2                         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
    eImageSparseMemoryRequirementsInfo2                   = VK_STRUCTURE_TYPE_IMAGE_SPARSE_MEMORY_REQUIREMENTS_INFO_2,
    eMemoryRequirements2                                  = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
    eSparseImageMemoryRequirements2                       = VK_STRUCTURE_TYPE_SPARSE_IMAGE_MEMORY_REQUIREMENTS_2,
    ePhysicalDeviceFeatures2                              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
    ePhysicalDeviceProperties2                            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
    eFormatProperties2                                    = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
    eImageFormatProperties2                               = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
    ePhysicalDeviceImageFormatInfo2                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
    eQueueFamilyProperties2                               = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2,
    ePhysicalDeviceMemoryProperties2                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
    eSparseImageFormatProperties2                         = VK_STRUCTURE_TYPE_SPARSE_IMAGE_FORMAT_PROPERTIES_2,
    ePhysicalDeviceSparseImageFormatInfo2                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SPARSE_IMAGE_FORMAT_INFO_2,
    ePhysicalDevicePointClippingProperties                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_POINT_CLIPPING_PROPERTIES,
    eRenderPassInputAttachmentAspectCreateInfo            = VK_STRUCTURE_TYPE_RENDER_PASS_INPUT_ATTACHMENT_ASPECT_CREATE_INFO,
    eImageViewUsageCreateInfo                             = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
    ePipelineTessellationDomainOriginStateCreateInfo      = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_DOMAIN_ORIGIN_STATE_CREATE_INFO,
    eRenderPassMultiviewCreateInfo                        = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO,
    ePhysicalDeviceMultiviewFeatures                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES,
    ePhysicalDeviceMultiviewProperties                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES,
    ePhysicalDeviceVariablePointersFeatures               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES,
    eProtectedSubmitInfo                                  = VK_STRUCTURE_TYPE_PROTECTED_SUBMIT_INFO,
    ePhysicalDeviceProtectedMemoryFeatures                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES,
    ePhysicalDeviceProtectedMemoryProperties              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_PROPERTIES,
    eDeviceQueueInfo2                                     = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2,
    eSamplerYcbcrConversionCreateInfo                     = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
    eSamplerYcbcrConversionInfo                           = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO,
    eBindImagePlaneMemoryInfo                             = VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO,
    eImagePlaneMemoryRequirementsInfo                     = VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO,
    ePhysicalDeviceSamplerYcbcrConversionFeatures         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES,
    eSamplerYcbcrConversionImageFormatProperties          = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES,
    eDescriptorUpdateTemplateCreateInfo                   = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO,
    ePhysicalDeviceExternalImageFormatInfo                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
    eExternalImageFormatProperties                        = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES,
    ePhysicalDeviceExternalBufferInfo                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
    eExternalBufferProperties                             = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES,
    ePhysicalDeviceIdProperties                           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
    eExternalMemoryBufferCreateInfo                       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
    eExternalMemoryImageCreateInfo                        = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
    eExportMemoryAllocateInfo                             = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
    ePhysicalDeviceExternalFenceInfo                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_FENCE_INFO,
    eExternalFenceProperties                              = VK_STRUCTURE_TYPE_EXTERNAL_FENCE_PROPERTIES,
    eExportFenceCreateInfo                                = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO,
    eExportSemaphoreCreateInfo                            = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
    ePhysicalDeviceExternalSemaphoreInfo                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
    eExternalSemaphoreProperties                          = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES,
    ePhysicalDeviceMaintenance3Properties                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES,
    eDescriptorSetLayoutSupport                           = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT,
    ePhysicalDeviceShaderDrawParametersFeatures           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES,
    ePhysicalDeviceVulkan11Features                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
    ePhysicalDeviceVulkan11Properties                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES,
    ePhysicalDeviceVulkan12Features                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
    ePhysicalDeviceVulkan12Properties                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES,
    eImageFormatListCreateInfo                            = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
    eAttachmentDescription2                               = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
    eAttachmentReference2                                 = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2,
    eSubpassDescription2                                  = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2,
    eSubpassDependency2                                   = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2,
    eRenderPassCreateInfo2                                = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2,
    eSubpassBeginInfo                                     = VK_STRUCTURE_TYPE_SUBPASS_BEGIN_INFO,
    eSubpassEndInfo                                       = VK_STRUCTURE_TYPE_SUBPASS_END_INFO,
    ePhysicalDevice8BitStorageFeatures                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
    ePhysicalDeviceDriverProperties                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
    ePhysicalDeviceShaderAtomicInt64Features              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES,
    ePhysicalDeviceShaderFloat16Int8Features              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
    ePhysicalDeviceFloatControlsProperties                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES,
    eDescriptorSetLayoutBindingFlagsCreateInfo            = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
    ePhysicalDeviceDescriptorIndexingFeatures             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
    ePhysicalDeviceDescriptorIndexingProperties           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES,
    eDescriptorSetVariableDescriptorCountAllocateInfo     = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO,
    eDescriptorSetVariableDescriptorCountLayoutSupport    = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_LAYOUT_SUPPORT,
    ePhysicalDeviceDepthStencilResolveProperties          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES,
    eSubpassDescriptionDepthStencilResolve                = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE,
    ePhysicalDeviceScalarBlockLayoutFeatures              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES,
    eImageStencilUsageCreateInfo                          = VK_STRUCTURE_TYPE_IMAGE_STENCIL_USAGE_CREATE_INFO,
    ePhysicalDeviceSamplerFilterMinmaxProperties          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_FILTER_MINMAX_PROPERTIES,
    eSamplerReductionModeCreateInfo                       = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO,
    ePhysicalDeviceVulkanMemoryModelFeatures              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES,
    ePhysicalDeviceImagelessFramebufferFeatures           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES,
    eFramebufferAttachmentsCreateInfo                     = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENTS_CREATE_INFO,
    eFramebufferAttachmentImageInfo                       = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENT_IMAGE_INFO,
    eRenderPassAttachmentBeginInfo                        = VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO,
    ePhysicalDeviceUniformBufferStandardLayoutFeatures    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES,
    ePhysicalDeviceShaderSubgroupExtendedTypesFeatures    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES,
    ePhysicalDeviceSeparateDepthStencilLayoutsFeatures    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES,
    eAttachmentReferenceStencilLayout                     = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_STENCIL_LAYOUT,
    eAttachmentDescriptionStencilLayout                   = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_STENCIL_LAYOUT,
    ePhysicalDeviceHostQueryResetFeatures                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES,
    ePhysicalDeviceTimelineSemaphoreFeatures              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
    ePhysicalDeviceTimelineSemaphoreProperties            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES,
    eSemaphoreTypeCreateInfo                              = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
    eTimelineSemaphoreSubmitInfo                          = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
    eSemaphoreWaitInfo                                    = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
    eSemaphoreSignalInfo                                  = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
    ePhysicalDeviceBufferDeviceAddressFeatures            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
    eBufferDeviceAddressInfo                              = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
    eBufferOpaqueCaptureAddressCreateInfo                 = VK_STRUCTURE_TYPE_BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO,
    eMemoryOpaqueCaptureAddressAllocateInfo               = VK_STRUCTURE_TYPE_MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO,
    eDeviceMemoryOpaqueCaptureAddressInfo                 = VK_STRUCTURE_TYPE_DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO,
    ePhysicalDeviceVulkan13Features                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
    ePhysicalDeviceVulkan13Properties                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES,
    ePipelineCreationFeedbackCreateInfo                   = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO,
    ePhysicalDeviceShaderTerminateInvocationFeatures      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES,
    ePhysicalDeviceToolProperties                         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TOOL_PROPERTIES,
    ePhysicalDeviceShaderDemoteToHelperInvocationFeatures = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES,
    ePhysicalDevicePrivateDataFeatures                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES,
    eDevicePrivateDataCreateInfo                          = VK_STRUCTURE_TYPE_DEVICE_PRIVATE_DATA_CREATE_INFO,
    ePrivateDataSlotCreateInfo                            = VK_STRUCTURE_TYPE_PRIVATE_DATA_SLOT_CREATE_INFO,
    ePhysicalDevicePipelineCreationCacheControlFeatures   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES,
    eMemoryBarrier2                                       = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
    eBufferMemoryBarrier2                                 = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
    eImageMemoryBarrier2                                  = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
    eDependencyInfo                                       = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
    eSubmitInfo2                                          = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
    eSemaphoreSubmitInfo                                  = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
    eCommandBufferSubmitInfo                              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
    ePhysicalDeviceSynchronization2Features               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
    ePhysicalDeviceZeroInitializeWorkgroupMemoryFeatures  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES,
    ePhysicalDeviceImageRobustnessFeatures                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES,
    eCopyBufferInfo2                                      = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
    eCopyImageInfo2                                       = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2,
    eCopyBufferToImageInfo2                               = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
    eCopyImageToBufferInfo2                               = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
    eBlitImageInfo2                                       = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
    eResolveImageInfo2                                    = VK_STRUCTURE_TYPE_RESOLVE_IMAGE_INFO_2,
    eBufferCopy2                                          = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
    eImageCopy2                                           = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
    eImageBlit2                                           = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
    eBufferImageCopy2                                     = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
    eImageResolve2                                        = VK_STRUCTURE_TYPE_IMAGE_RESOLVE_2,
    ePhysicalDeviceSubgroupSizeControlProperties          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES,
    ePipelineShaderStageRequiredSubgroupSizeCreateInfo    = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
    ePhysicalDeviceSubgroupSizeControlFeatures            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
    ePhysicalDeviceInlineUniformBlockFeatures             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES,
    ePhysicalDeviceInlineUniformBlockProperties           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_PROPERTIES,
    eWriteDescriptorSetInlineUniformBlock                 = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK,
    eDescriptorPoolInlineUniformBlockCreateInfo           = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_INLINE_UNIFORM_BLOCK_CREATE_INFO,
    ePhysicalDeviceTextureCompressionAstcHdrFeatures      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES,
    eRenderingInfo                                        = VK_STRUCTURE_TYPE_RENDERING_INFO,
    eRenderingAttachmentInfo                              = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
    ePipelineRenderingCreateInfo                          = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
    ePhysicalDeviceDynamicRenderingFeatures               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
    eCommandBufferInheritanceRenderingInfo                = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO,
    ePhysicalDeviceShaderIntegerDotProductFeatures        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
    ePhysicalDeviceShaderIntegerDotProductProperties      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES,
    ePhysicalDeviceTexelBufferAlignmentProperties         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_PROPERTIES,
    eFormatProperties3                                    = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3,
    ePhysicalDeviceMaintenance4Features                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES,
    ePhysicalDeviceMaintenance4Properties                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES,
    eDeviceBufferMemoryRequirements                       = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS,
    eDeviceImageMemoryRequirements                        = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS,
    eSwapchainCreateInfoKHR                               = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
    ePresentInfoKHR                                       = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
    eDeviceGroupPresentCapabilitiesKHR                    = VK_STRUCTURE_TYPE_DEVICE_GROUP_PRESENT_CAPABILITIES_KHR,
    eImageSwapchainCreateInfoKHR                          = VK_STRUCTURE_TYPE_IMAGE_SWAPCHAIN_CREATE_INFO_KHR,
    eBindImageMemorySwapchainInfoKHR                      = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_SWAPCHAIN_INFO_KHR,
    eAcquireNextImageInfoKHR                              = VK_STRUCTURE_TYPE_ACQUIRE_NEXT_IMAGE_INFO_KHR,
    eDeviceGroupPresentInfoKHR                            = VK_STRUCTURE_TYPE_DEVICE_GROUP_PRESENT_INFO_KHR,
    eDeviceGroupSwapchainCreateInfoKHR                    = VK_STRUCTURE_TYPE_DEVICE_GROUP_SWAPCHAIN_CREATE_INFO_KHR,
    eDisplayModeCreateInfoKHR                             = VK_STRUCTURE_TYPE_DISPLAY_MODE_CREATE_INFO_KHR,
    eDisplaySurfaceCreateInfoKHR                          = VK_STRUCTURE_TYPE_DISPLAY_SURFACE_CREATE_INFO_KHR,
    eDisplayPresentInfoKHR                                = VK_STRUCTURE_TYPE_DISPLAY_PRESENT_INFO_KHR,
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
    eDebugReportCallbackCreateInfoEXT                = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
    ePipelineRasterizationStateRasterizationOrderAMD = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_RASTERIZATION_ORDER_AMD,
    eDebugMarkerObjectNameInfoEXT                    = VK_STRUCTURE_TYPE_DEBUG_MARKER_OBJECT_NAME_INFO_EXT,
    eDebugMarkerObjectTagInfoEXT                     = VK_STRUCTURE_TYPE_DEBUG_MARKER_OBJECT_TAG_INFO_EXT,
    eDebugMarkerMarkerInfoEXT                        = VK_STRUCTURE_TYPE_DEBUG_MARKER_MARKER_INFO_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoProfileInfoKHR                       = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR,
    eVideoCapabilitiesKHR                      = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR,
    eVideoPictureResourceInfoKHR               = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,
    eVideoSessionMemoryRequirementsKHR         = VK_STRUCTURE_TYPE_VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR,
    eBindVideoSessionMemoryInfoKHR             = VK_STRUCTURE_TYPE_BIND_VIDEO_SESSION_MEMORY_INFO_KHR,
    eVideoSessionCreateInfoKHR                 = VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR,
    eVideoSessionParametersCreateInfoKHR       = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
    eVideoSessionParametersUpdateInfoKHR       = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_UPDATE_INFO_KHR,
    eVideoBeginCodingInfoKHR                   = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR,
    eVideoEndCodingInfoKHR                     = VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR,
    eVideoCodingControlInfoKHR                 = VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR,
    eVideoReferenceSlotInfoKHR                 = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
    eQueueFamilyVideoPropertiesKHR             = VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR,
    eVideoProfileListInfoKHR                   = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR,
    ePhysicalDeviceVideoFormatInfoKHR          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_FORMAT_INFO_KHR,
    eVideoFormatPropertiesKHR                  = VK_STRUCTURE_TYPE_VIDEO_FORMAT_PROPERTIES_KHR,
    eQueueFamilyQueryResultStatusPropertiesKHR = VK_STRUCTURE_TYPE_QUEUE_FAMILY_QUERY_RESULT_STATUS_PROPERTIES_KHR,
    eVideoDecodeInfoKHR                        = VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR,
    eVideoDecodeCapabilitiesKHR                = VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR,
    eVideoDecodeUsageInfoKHR                   = VK_STRUCTURE_TYPE_VIDEO_DECODE_USAGE_INFO_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eDedicatedAllocationImageCreateInfoNV          = VK_STRUCTURE_TYPE_DEDICATED_ALLOCATION_IMAGE_CREATE_INFO_NV,
    eDedicatedAllocationBufferCreateInfoNV         = VK_STRUCTURE_TYPE_DEDICATED_ALLOCATION_BUFFER_CREATE_INFO_NV,
    eDedicatedAllocationMemoryAllocateInfoNV       = VK_STRUCTURE_TYPE_DEDICATED_ALLOCATION_MEMORY_ALLOCATE_INFO_NV,
    ePhysicalDeviceTransformFeedbackFeaturesEXT    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT,
    ePhysicalDeviceTransformFeedbackPropertiesEXT  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_PROPERTIES_EXT,
    ePipelineRasterizationStateStreamCreateInfoEXT = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_STREAM_CREATE_INFO_EXT,
    eCuModuleCreateInfoNVX                         = VK_STRUCTURE_TYPE_CU_MODULE_CREATE_INFO_NVX,
    eCuFunctionCreateInfoNVX                       = VK_STRUCTURE_TYPE_CU_FUNCTION_CREATE_INFO_NVX,
    eCuLaunchInfoNVX                               = VK_STRUCTURE_TYPE_CU_LAUNCH_INFO_NVX,
    eImageViewHandleInfoNVX                        = VK_STRUCTURE_TYPE_IMAGE_VIEW_HANDLE_INFO_NVX,
    eImageViewAddressPropertiesNVX                 = VK_STRUCTURE_TYPE_IMAGE_VIEW_ADDRESS_PROPERTIES_NVX,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoEncodeH264CapabilitiesEXT                = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_CAPABILITIES_EXT,
    eVideoEncodeH264SessionParametersCreateInfoEXT = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_CREATE_INFO_EXT,
    eVideoEncodeH264SessionParametersAddInfoEXT    = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_ADD_INFO_EXT,
    eVideoEncodeH264VclFrameInfoEXT                = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_VCL_FRAME_INFO_EXT,
    eVideoEncodeH264DpbSlotInfoEXT                 = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_DPB_SLOT_INFO_EXT,
    eVideoEncodeH264NaluSliceInfoEXT               = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_NALU_SLICE_INFO_EXT,
    eVideoEncodeH264EmitPictureParametersInfoEXT   = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_EMIT_PICTURE_PARAMETERS_INFO_EXT,
    eVideoEncodeH264ProfileInfoEXT                 = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PROFILE_INFO_EXT,
    eVideoEncodeH264RateControlInfoEXT             = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_RATE_CONTROL_INFO_EXT,
    eVideoEncodeH264RateControlLayerInfoEXT        = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_RATE_CONTROL_LAYER_INFO_EXT,
    eVideoEncodeH264ReferenceListsInfoEXT          = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_REFERENCE_LISTS_INFO_EXT,
    eVideoEncodeH265CapabilitiesEXT                = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_CAPABILITIES_EXT,
    eVideoEncodeH265SessionParametersCreateInfoEXT = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_SESSION_PARAMETERS_CREATE_INFO_EXT,
    eVideoEncodeH265SessionParametersAddInfoEXT    = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_SESSION_PARAMETERS_ADD_INFO_EXT,
    eVideoEncodeH265VclFrameInfoEXT                = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_VCL_FRAME_INFO_EXT,
    eVideoEncodeH265DpbSlotInfoEXT                 = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_DPB_SLOT_INFO_EXT,
    eVideoEncodeH265NaluSliceSegmentInfoEXT        = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_NALU_SLICE_SEGMENT_INFO_EXT,
    eVideoEncodeH265EmitPictureParametersInfoEXT   = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_EMIT_PICTURE_PARAMETERS_INFO_EXT,
    eVideoEncodeH265ProfileInfoEXT                 = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_PROFILE_INFO_EXT,
    eVideoEncodeH265ReferenceListsInfoEXT          = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_REFERENCE_LISTS_INFO_EXT,
    eVideoEncodeH265RateControlInfoEXT             = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_RATE_CONTROL_INFO_EXT,
    eVideoEncodeH265RateControlLayerInfoEXT        = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_RATE_CONTROL_LAYER_INFO_EXT,
    eVideoDecodeH264CapabilitiesEXT                = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_CAPABILITIES_EXT,
    eVideoDecodeH264PictureInfoEXT                 = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PICTURE_INFO_EXT,
    eVideoDecodeH264ProfileInfoEXT                 = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PROFILE_INFO_EXT,
    eVideoDecodeH264SessionParametersCreateInfoEXT = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO_EXT,
    eVideoDecodeH264SessionParametersAddInfoEXT    = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_EXT,
    eVideoDecodeH264DpbSlotInfoEXT                 = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_EXT,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eTextureLodGatherFormatPropertiesAMD           = VK_STRUCTURE_TYPE_TEXTURE_LOD_GATHER_FORMAT_PROPERTIES_AMD,
    eRenderingFragmentShadingRateAttachmentInfoKHR = VK_STRUCTURE_TYPE_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_INFO_KHR,
    eRenderingFragmentDensityMapAttachmentInfoEXT  = VK_STRUCTURE_TYPE_RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT_INFO_EXT,
    eAttachmentSampleCountInfoAMD                  = VK_STRUCTURE_TYPE_ATTACHMENT_SAMPLE_COUNT_INFO_AMD,
    eMultiviewPerViewAttributesInfoNVX             = VK_STRUCTURE_TYPE_MULTIVIEW_PER_VIEW_ATTRIBUTES_INFO_NVX,
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
    eImageViewAstcDecodeModeEXT                    = VK_STRUCTURE_TYPE_IMAGE_VIEW_ASTC_DECODE_MODE_EXT,
    ePhysicalDeviceAstcDecodeFeaturesEXT           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ASTC_DECODE_FEATURES_EXT,
    ePipelineRobustnessCreateInfoEXT               = VK_STRUCTURE_TYPE_PIPELINE_ROBUSTNESS_CREATE_INFO_EXT,
    ePhysicalDevicePipelineRobustnessFeaturesEXT   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES_EXT,
    ePhysicalDevicePipelineRobustnessPropertiesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_PROPERTIES_EXT,
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
    eImportSemaphoreFdInfoKHR                              = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
    eSemaphoreGetFdInfoKHR                                 = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
    ePhysicalDevicePushDescriptorPropertiesKHR             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR,
    eCommandBufferInheritanceConditionalRenderingInfoEXT   = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_CONDITIONAL_RENDERING_INFO_EXT,
    ePhysicalDeviceConditionalRenderingFeaturesEXT         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT,
    eConditionalRenderingBeginInfoEXT                      = VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT,
    ePresentRegionsKHR                                     = VK_STRUCTURE_TYPE_PRESENT_REGIONS_KHR,
    ePipelineViewportWScalingStateCreateInfoNV             = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_W_SCALING_STATE_CREATE_INFO_NV,
    eSurfaceCapabilities2EXT                               = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_EXT,
    eDisplayPowerInfoEXT                                   = VK_STRUCTURE_TYPE_DISPLAY_POWER_INFO_EXT,
    eDeviceEventInfoEXT                                    = VK_STRUCTURE_TYPE_DEVICE_EVENT_INFO_EXT,
    eDisplayEventInfoEXT                                   = VK_STRUCTURE_TYPE_DISPLAY_EVENT_INFO_EXT,
    eSwapchainCounterCreateInfoEXT                         = VK_STRUCTURE_TYPE_SWAPCHAIN_COUNTER_CREATE_INFO_EXT,
    ePresentTimesInfoGOOGLE                                = VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE,
    ePhysicalDeviceMultiviewPerViewAttributesPropertiesNVX = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PER_VIEW_ATTRIBUTES_PROPERTIES_NVX,
    ePipelineViewportSwizzleStateCreateInfoNV              = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_SWIZZLE_STATE_CREATE_INFO_NV,
    ePhysicalDeviceDiscardRectanglePropertiesEXT           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DISCARD_RECTANGLE_PROPERTIES_EXT,
    ePipelineDiscardRectangleStateCreateInfoEXT            = VK_STRUCTURE_TYPE_PIPELINE_DISCARD_RECTANGLE_STATE_CREATE_INFO_EXT,
    ePhysicalDeviceConservativeRasterizationPropertiesEXT  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES_EXT,
    ePipelineRasterizationConservativeStateCreateInfoEXT   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT,
    ePhysicalDeviceDepthClipEnableFeaturesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT,
    ePipelineRasterizationDepthClipStateCreateInfoEXT      = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT,
    eHdrMetadataEXT                                        = VK_STRUCTURE_TYPE_HDR_METADATA_EXT,
    eSharedPresentSurfaceCapabilitiesKHR                   = VK_STRUCTURE_TYPE_SHARED_PRESENT_SURFACE_CAPABILITIES_KHR,
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
    eAndroidHardwareBufferUsageANDROID             = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_USAGE_ANDROID,
    eAndroidHardwareBufferPropertiesANDROID        = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID,
    eAndroidHardwareBufferFormatPropertiesANDROID  = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID,
    eImportAndroidHardwareBufferInfoANDROID        = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
    eMemoryGetAndroidHardwareBufferInfoANDROID     = VK_STRUCTURE_TYPE_MEMORY_GET_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
    eExternalFormatANDROID                         = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID,
    eAndroidHardwareBufferFormatProperties2ANDROID = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_2_ANDROID,
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
    eSampleLocationsInfoEXT                            = VK_STRUCTURE_TYPE_SAMPLE_LOCATIONS_INFO_EXT,
    eRenderPassSampleLocationsBeginInfoEXT             = VK_STRUCTURE_TYPE_RENDER_PASS_SAMPLE_LOCATIONS_BEGIN_INFO_EXT,
    ePipelineSampleLocationsStateCreateInfoEXT         = VK_STRUCTURE_TYPE_PIPELINE_SAMPLE_LOCATIONS_STATE_CREATE_INFO_EXT,
    ePhysicalDeviceSampleLocationsPropertiesEXT        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLE_LOCATIONS_PROPERTIES_EXT,
    eMultisamplePropertiesEXT                          = VK_STRUCTURE_TYPE_MULTISAMPLE_PROPERTIES_EXT,
    ePhysicalDeviceBlendOperationAdvancedFeaturesEXT   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BLEND_OPERATION_ADVANCED_FEATURES_EXT,
    ePhysicalDeviceBlendOperationAdvancedPropertiesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BLEND_OPERATION_ADVANCED_PROPERTIES_EXT,
    ePipelineColorBlendAdvancedStateCreateInfoEXT      = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_ADVANCED_STATE_CREATE_INFO_EXT,
    ePipelineCoverageToColorStateCreateInfoNV          = VK_STRUCTURE_TYPE_PIPELINE_COVERAGE_TO_COLOR_STATE_CREATE_INFO_NV,
    eWriteDescriptorSetAccelerationStructureKHR        = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
    eAccelerationStructureBuildGeometryInfoKHR         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
    eAccelerationStructureDeviceAddressInfoKHR         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
    eAccelerationStructureGeometryAabbsDataKHR         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR,
    eAccelerationStructureGeometryInstancesDataKHR     = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
    eAccelerationStructureGeometryTrianglesDataKHR     = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
    eAccelerationStructureGeometryKHR                  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
    eAccelerationStructureVersionInfoKHR               = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_VERSION_INFO_KHR,
    eCopyAccelerationStructureInfoKHR                  = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR,
    eCopyAccelerationStructureToMemoryInfoKHR          = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_TO_MEMORY_INFO_KHR,
    eCopyMemoryToAccelerationStructureInfoKHR          = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_ACCELERATION_STRUCTURE_INFO_KHR,
    ePhysicalDeviceAccelerationStructureFeaturesKHR    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
    ePhysicalDeviceAccelerationStructurePropertiesKHR  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR,
    eAccelerationStructureCreateInfoKHR                = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
    eAccelerationStructureBuildSizesInfoKHR            = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
    ePhysicalDeviceRayTracingPipelineFeaturesKHR       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
    ePhysicalDeviceRayTracingPipelinePropertiesKHR     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
    eRayTracingPipelineCreateInfoKHR                   = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
    eRayTracingShaderGroupCreateInfoKHR                = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
    eRayTracingPipelineInterfaceCreateInfoKHR          = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_INTERFACE_CREATE_INFO_KHR,
    ePhysicalDeviceRayQueryFeaturesKHR                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
    ePipelineCoverageModulationStateCreateInfoNV       = VK_STRUCTURE_TYPE_PIPELINE_COVERAGE_MODULATION_STATE_CREATE_INFO_NV,
    ePhysicalDeviceShaderSmBuiltinsFeaturesNV          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_FEATURES_NV,
    ePhysicalDeviceShaderSmBuiltinsPropertiesNV        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV,
    eDrmFormatModifierPropertiesListEXT                = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT,
    ePhysicalDeviceImageDrmFormatModifierInfoEXT       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT,
    eImageDrmFormatModifierListCreateInfoEXT           = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
    eImageDrmFormatModifierExplicitCreateInfoEXT       = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
    eImageDrmFormatModifierPropertiesEXT               = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT,
    eDrmFormatModifierPropertiesList2EXT               = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_2_EXT,
    eValidationCacheCreateInfoEXT                      = VK_STRUCTURE_TYPE_VALIDATION_CACHE_CREATE_INFO_EXT,
    eShaderModuleValidationCacheCreateInfoEXT          = VK_STRUCTURE_TYPE_SHADER_MODULE_VALIDATION_CACHE_CREATE_INFO_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    ePhysicalDevicePortabilitySubsetFeaturesKHR   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR,
    ePhysicalDevicePortabilitySubsetPropertiesKHR = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_PROPERTIES_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    ePipelineViewportShadingRateImageStateCreateInfoNV   = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_SHADING_RATE_IMAGE_STATE_CREATE_INFO_NV,
    ePhysicalDeviceShadingRateImageFeaturesNV            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADING_RATE_IMAGE_FEATURES_NV,
    ePhysicalDeviceShadingRateImagePropertiesNV          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADING_RATE_IMAGE_PROPERTIES_NV,
    ePipelineViewportCoarseSampleOrderStateCreateInfoNV  = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_COARSE_SAMPLE_ORDER_STATE_CREATE_INFO_NV,
    eRayTracingPipelineCreateInfoNV                      = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_NV,
    eAccelerationStructureCreateInfoNV                   = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV,
    eGeometryNV                                          = VK_STRUCTURE_TYPE_GEOMETRY_NV,
    eGeometryTrianglesNV                                 = VK_STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV,
    eGeometryAabbNV                                      = VK_STRUCTURE_TYPE_GEOMETRY_AABB_NV,
    eBindAccelerationStructureMemoryInfoNV               = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV,
    eWriteDescriptorSetAccelerationStructureNV           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_NV,
    eAccelerationStructureMemoryRequirementsInfoNV       = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV,
    ePhysicalDeviceRayTracingPropertiesNV                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PROPERTIES_NV,
    eRayTracingShaderGroupCreateInfoNV                   = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV,
    eAccelerationStructureInfoNV                         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV,
    ePhysicalDeviceRepresentativeFragmentTestFeaturesNV  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_REPRESENTATIVE_FRAGMENT_TEST_FEATURES_NV,
    ePipelineRepresentativeFragmentTestStateCreateInfoNV = VK_STRUCTURE_TYPE_PIPELINE_REPRESENTATIVE_FRAGMENT_TEST_STATE_CREATE_INFO_NV,
    ePhysicalDeviceImageViewImageFormatInfoEXT           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_VIEW_IMAGE_FORMAT_INFO_EXT,
    eFilterCubicImageViewImageFormatPropertiesEXT        = VK_STRUCTURE_TYPE_FILTER_CUBIC_IMAGE_VIEW_IMAGE_FORMAT_PROPERTIES_EXT,
    eImportMemoryHostPointerInfoEXT                      = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
    eMemoryHostPointerPropertiesEXT                      = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT,
    ePhysicalDeviceExternalMemoryHostPropertiesEXT       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT,
    ePhysicalDeviceShaderClockFeaturesKHR                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR,
    ePipelineCompilerControlCreateInfoAMD                = VK_STRUCTURE_TYPE_PIPELINE_COMPILER_CONTROL_CREATE_INFO_AMD,
    eCalibratedTimestampInfoEXT                          = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT,
    ePhysicalDeviceShaderCorePropertiesAMD               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoDecodeH265CapabilitiesEXT                = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_CAPABILITIES_EXT,
    eVideoDecodeH265SessionParametersCreateInfoEXT = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_CREATE_INFO_EXT,
    eVideoDecodeH265SessionParametersAddInfoEXT    = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_ADD_INFO_EXT,
    eVideoDecodeH265ProfileInfoEXT                 = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PROFILE_INFO_EXT,
    eVideoDecodeH265PictureInfoEXT                 = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PICTURE_INFO_EXT,
    eVideoDecodeH265DpbSlotInfoEXT                 = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_DPB_SLOT_INFO_EXT,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eDeviceQueueGlobalPriorityCreateInfoKHR            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR,
    ePhysicalDeviceGlobalPriorityQueryFeaturesKHR      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GLOBAL_PRIORITY_QUERY_FEATURES_KHR,
    eQueueFamilyGlobalPriorityPropertiesKHR            = VK_STRUCTURE_TYPE_QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES_KHR,
    eDeviceMemoryOverallocationCreateInfoAMD           = VK_STRUCTURE_TYPE_DEVICE_MEMORY_OVERALLOCATION_CREATE_INFO_AMD,
    ePhysicalDeviceVertexAttributeDivisorPropertiesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_PROPERTIES_EXT,
    ePipelineVertexInputDivisorStateCreateInfoEXT      = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT,
    ePhysicalDeviceVertexAttributeDivisorFeaturesEXT   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT,
#if defined( VK_USE_PLATFORM_GGP )
    ePresentFrameTokenGGP = VK_STRUCTURE_TYPE_PRESENT_FRAME_TOKEN_GGP,
#endif /*VK_USE_PLATFORM_GGP*/
    ePhysicalDeviceComputeShaderDerivativesFeaturesNV   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_NV,
    ePhysicalDeviceMeshShaderFeaturesNV                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV,
    ePhysicalDeviceMeshShaderPropertiesNV               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_NV,
    ePhysicalDeviceShaderImageFootprintFeaturesNV       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_FOOTPRINT_FEATURES_NV,
    ePipelineViewportExclusiveScissorStateCreateInfoNV  = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_EXCLUSIVE_SCISSOR_STATE_CREATE_INFO_NV,
    ePhysicalDeviceExclusiveScissorFeaturesNV           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXCLUSIVE_SCISSOR_FEATURES_NV,
    eCheckpointDataNV                                   = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV,
    eQueueFamilyCheckpointPropertiesNV                  = VK_STRUCTURE_TYPE_QUEUE_FAMILY_CHECKPOINT_PROPERTIES_NV,
    ePhysicalDeviceShaderIntegerFunctions2FeaturesINTEL = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_FUNCTIONS_2_FEATURES_INTEL,
    eQueryPoolPerformanceQueryCreateInfoINTEL           = VK_STRUCTURE_TYPE_QUERY_POOL_PERFORMANCE_QUERY_CREATE_INFO_INTEL,
    eInitializePerformanceApiInfoINTEL                  = VK_STRUCTURE_TYPE_INITIALIZE_PERFORMANCE_API_INFO_INTEL,
    ePerformanceMarkerInfoINTEL                         = VK_STRUCTURE_TYPE_PERFORMANCE_MARKER_INFO_INTEL,
    ePerformanceStreamMarkerInfoINTEL                   = VK_STRUCTURE_TYPE_PERFORMANCE_STREAM_MARKER_INFO_INTEL,
    ePerformanceOverrideInfoINTEL                       = VK_STRUCTURE_TYPE_PERFORMANCE_OVERRIDE_INFO_INTEL,
    ePerformanceConfigurationAcquireInfoINTEL           = VK_STRUCTURE_TYPE_PERFORMANCE_CONFIGURATION_ACQUIRE_INFO_INTEL,
    ePhysicalDevicePciBusInfoPropertiesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT,
    eDisplayNativeHdrSurfaceCapabilitiesAMD             = VK_STRUCTURE_TYPE_DISPLAY_NATIVE_HDR_SURFACE_CAPABILITIES_AMD,
    eSwapchainDisplayNativeHdrCreateInfoAMD             = VK_STRUCTURE_TYPE_SWAPCHAIN_DISPLAY_NATIVE_HDR_CREATE_INFO_AMD,
#if defined( VK_USE_PLATFORM_FUCHSIA )
    eImagepipeSurfaceCreateInfoFUCHSIA = VK_STRUCTURE_TYPE_IMAGEPIPE_SURFACE_CREATE_INFO_FUCHSIA,
#endif /*VK_USE_PLATFORM_FUCHSIA*/
#if defined( VK_USE_PLATFORM_METAL_EXT )
    eMetalSurfaceCreateInfoEXT = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT,
#endif /*VK_USE_PLATFORM_METAL_EXT*/
    ePhysicalDeviceFragmentDensityMapFeaturesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_FEATURES_EXT,
    ePhysicalDeviceFragmentDensityMapPropertiesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_PROPERTIES_EXT,
    eRenderPassFragmentDensityMapCreateInfoEXT                = VK_STRUCTURE_TYPE_RENDER_PASS_FRAGMENT_DENSITY_MAP_CREATE_INFO_EXT,
    eFragmentShadingRateAttachmentInfoKHR                     = VK_STRUCTURE_TYPE_FRAGMENT_SHADING_RATE_ATTACHMENT_INFO_KHR,
    ePipelineFragmentShadingRateStateCreateInfoKHR            = VK_STRUCTURE_TYPE_PIPELINE_FRAGMENT_SHADING_RATE_STATE_CREATE_INFO_KHR,
    ePhysicalDeviceFragmentShadingRatePropertiesKHR           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_PROPERTIES_KHR,
    ePhysicalDeviceFragmentShadingRateFeaturesKHR             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR,
    ePhysicalDeviceFragmentShadingRateKHR                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_KHR,
    ePhysicalDeviceShaderCoreProperties2AMD                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_2_AMD,
    ePhysicalDeviceCoherentMemoryFeaturesAMD                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD,
    ePhysicalDeviceShaderImageAtomicInt64FeaturesEXT          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT,
    ePhysicalDeviceMemoryBudgetPropertiesEXT                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT,
    ePhysicalDeviceMemoryPriorityFeaturesEXT                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT,
    eMemoryPriorityAllocateInfoEXT                            = VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT,
    eSurfaceProtectedCapabilitiesKHR                          = VK_STRUCTURE_TYPE_SURFACE_PROTECTED_CAPABILITIES_KHR,
    ePhysicalDeviceDedicatedAllocationImageAliasingFeaturesNV = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEDICATED_ALLOCATION_IMAGE_ALIASING_FEATURES_NV,
    ePhysicalDeviceBufferDeviceAddressFeaturesEXT             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_EXT,
    eBufferDeviceAddressCreateInfoEXT                         = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_CREATE_INFO_EXT,
    eValidationFeaturesEXT                                    = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
    ePhysicalDevicePresentWaitFeaturesKHR                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR,
    ePhysicalDeviceCooperativeMatrixFeaturesNV                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_NV,
    eCooperativeMatrixPropertiesNV                            = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_NV,
    ePhysicalDeviceCooperativeMatrixPropertiesNV              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_NV,
    ePhysicalDeviceCoverageReductionModeFeaturesNV            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COVERAGE_REDUCTION_MODE_FEATURES_NV,
    ePipelineCoverageReductionStateCreateInfoNV               = VK_STRUCTURE_TYPE_PIPELINE_COVERAGE_REDUCTION_STATE_CREATE_INFO_NV,
    eFramebufferMixedSamplesCombinationNV                     = VK_STRUCTURE_TYPE_FRAMEBUFFER_MIXED_SAMPLES_COMBINATION_NV,
    ePhysicalDeviceFragmentShaderInterlockFeaturesEXT         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT,
    ePhysicalDeviceYcbcrImageArraysFeaturesEXT                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_YCBCR_IMAGE_ARRAYS_FEATURES_EXT,
    ePhysicalDeviceProvokingVertexFeaturesEXT                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT,
    ePipelineRasterizationProvokingVertexStateCreateInfoEXT   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT,
    ePhysicalDeviceProvokingVertexPropertiesEXT               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_PROPERTIES_EXT,
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    eSurfaceFullScreenExclusiveInfoEXT         = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_INFO_EXT,
    eSurfaceCapabilitiesFullScreenExclusiveEXT = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_FULL_SCREEN_EXCLUSIVE_EXT,
    eSurfaceFullScreenExclusiveWin32InfoEXT    = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_WIN32_INFO_EXT,
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    eHeadlessSurfaceCreateInfoEXT                          = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT,
    ePhysicalDeviceLineRasterizationFeaturesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_EXT,
    ePipelineRasterizationLineStateCreateInfoEXT           = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO_EXT,
    ePhysicalDeviceLineRasterizationPropertiesEXT          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_PROPERTIES_EXT,
    ePhysicalDeviceShaderAtomicFloatFeaturesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
    ePhysicalDeviceIndexTypeUint8FeaturesEXT               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT,
    ePhysicalDeviceExtendedDynamicStateFeaturesEXT         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT,
    ePhysicalDevicePipelineExecutablePropertiesFeaturesKHR = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR,
    ePipelineInfoKHR                                       = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR,
    ePipelineExecutablePropertiesKHR                       = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR,
    ePipelineExecutableInfoKHR                             = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR,
    ePipelineExecutableStatisticKHR                        = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR,
    ePipelineExecutableInternalRepresentationKHR           = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR,
    ePhysicalDeviceShaderAtomicFloat2FeaturesEXT           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT,
    ePhysicalDeviceDeviceGeneratedCommandsPropertiesNV     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_PROPERTIES_NV,
    eGraphicsShaderGroupCreateInfoNV                       = VK_STRUCTURE_TYPE_GRAPHICS_SHADER_GROUP_CREATE_INFO_NV,
    eGraphicsPipelineShaderGroupsCreateInfoNV              = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_SHADER_GROUPS_CREATE_INFO_NV,
    eIndirectCommandsLayoutTokenNV                         = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_NV,
    eIndirectCommandsLayoutCreateInfoNV                    = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_NV,
    eGeneratedCommandsInfoNV                               = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_NV,
    eGeneratedCommandsMemoryRequirementsInfoNV             = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_NV,
    ePhysicalDeviceDeviceGeneratedCommandsFeaturesNV       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_NV,
    ePhysicalDeviceInheritedViewportScissorFeaturesNV      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INHERITED_VIEWPORT_SCISSOR_FEATURES_NV,
    eCommandBufferInheritanceViewportScissorInfoNV         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_VIEWPORT_SCISSOR_INFO_NV,
    ePhysicalDeviceTexelBufferAlignmentFeaturesEXT         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_FEATURES_EXT,
    eCommandBufferInheritanceRenderPassTransformInfoQCOM   = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDER_PASS_TRANSFORM_INFO_QCOM,
    eRenderPassTransformBeginInfoQCOM                      = VK_STRUCTURE_TYPE_RENDER_PASS_TRANSFORM_BEGIN_INFO_QCOM,
    ePhysicalDeviceDeviceMemoryReportFeaturesEXT           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_MEMORY_REPORT_FEATURES_EXT,
    eDeviceDeviceMemoryReportCreateInfoEXT                 = VK_STRUCTURE_TYPE_DEVICE_DEVICE_MEMORY_REPORT_CREATE_INFO_EXT,
    eDeviceMemoryReportCallbackDataEXT                     = VK_STRUCTURE_TYPE_DEVICE_MEMORY_REPORT_CALLBACK_DATA_EXT,
    ePhysicalDeviceRobustness2FeaturesEXT                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
    ePhysicalDeviceRobustness2PropertiesEXT                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_PROPERTIES_EXT,
    eSamplerCustomBorderColorCreateInfoEXT                 = VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT,
    ePhysicalDeviceCustomBorderColorPropertiesEXT          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_PROPERTIES_EXT,
    ePhysicalDeviceCustomBorderColorFeaturesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT,
    ePipelineLibraryCreateInfoKHR                          = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR,
    ePhysicalDevicePresentBarrierFeaturesNV                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_BARRIER_FEATURES_NV,
    eSurfaceCapabilitiesPresentBarrierNV                   = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_PRESENT_BARRIER_NV,
    eSwapchainPresentBarrierCreateInfoNV                   = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_BARRIER_CREATE_INFO_NV,
    ePresentIdKHR                                          = VK_STRUCTURE_TYPE_PRESENT_ID_KHR,
    ePhysicalDevicePresentIdFeaturesKHR                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoEncodeInfoKHR                 = VK_STRUCTURE_TYPE_VIDEO_ENCODE_INFO_KHR,
    eVideoEncodeRateControlInfoKHR      = VK_STRUCTURE_TYPE_VIDEO_ENCODE_RATE_CONTROL_INFO_KHR,
    eVideoEncodeRateControlLayerInfoKHR = VK_STRUCTURE_TYPE_VIDEO_ENCODE_RATE_CONTROL_LAYER_INFO_KHR,
    eVideoEncodeCapabilitiesKHR         = VK_STRUCTURE_TYPE_VIDEO_ENCODE_CAPABILITIES_KHR,
    eVideoEncodeUsageInfoKHR            = VK_STRUCTURE_TYPE_VIDEO_ENCODE_USAGE_INFO_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    ePhysicalDeviceDiagnosticsConfigFeaturesNV = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DIAGNOSTICS_CONFIG_FEATURES_NV,
    eDeviceDiagnosticsConfigCreateInfoNV       = VK_STRUCTURE_TYPE_DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO_NV,
#if defined( VK_USE_PLATFORM_METAL_EXT )
    eExportMetalObjectCreateInfoEXT = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECT_CREATE_INFO_EXT,
    eExportMetalObjectsInfoEXT      = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT,
    eExportMetalDeviceInfoEXT       = VK_STRUCTURE_TYPE_EXPORT_METAL_DEVICE_INFO_EXT,
    eExportMetalCommandQueueInfoEXT = VK_STRUCTURE_TYPE_EXPORT_METAL_COMMAND_QUEUE_INFO_EXT,
    eExportMetalBufferInfoEXT       = VK_STRUCTURE_TYPE_EXPORT_METAL_BUFFER_INFO_EXT,
    eImportMetalBufferInfoEXT       = VK_STRUCTURE_TYPE_IMPORT_METAL_BUFFER_INFO_EXT,
    eExportMetalTextureInfoEXT      = VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT,
    eImportMetalTextureInfoEXT      = VK_STRUCTURE_TYPE_IMPORT_METAL_TEXTURE_INFO_EXT,
    eExportMetalIoSurfaceInfoEXT    = VK_STRUCTURE_TYPE_EXPORT_METAL_IO_SURFACE_INFO_EXT,
    eImportMetalIoSurfaceInfoEXT    = VK_STRUCTURE_TYPE_IMPORT_METAL_IO_SURFACE_INFO_EXT,
    eExportMetalSharedEventInfoEXT  = VK_STRUCTURE_TYPE_EXPORT_METAL_SHARED_EVENT_INFO_EXT,
    eImportMetalSharedEventInfoEXT  = VK_STRUCTURE_TYPE_IMPORT_METAL_SHARED_EVENT_INFO_EXT,
#endif /*VK_USE_PLATFORM_METAL_EXT*/
    eQueueFamilyCheckpointProperties2NV                        = VK_STRUCTURE_TYPE_QUEUE_FAMILY_CHECKPOINT_PROPERTIES_2_NV,
    eCheckpointData2NV                                         = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_2_NV,
    ePhysicalDeviceGraphicsPipelineLibraryFeaturesEXT          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT,
    ePhysicalDeviceGraphicsPipelineLibraryPropertiesEXT        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT,
    eGraphicsPipelineLibraryCreateInfoEXT                      = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
    ePhysicalDeviceShaderEarlyAndLateFragmentTestsFeaturesAMD  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_EARLY_AND_LATE_FRAGMENT_TESTS_FEATURES_AMD,
    ePhysicalDeviceFragmentShaderBarycentricFeaturesKHR        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR,
    ePhysicalDeviceFragmentShaderBarycentricPropertiesKHR      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_PROPERTIES_KHR,
    ePhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_FEATURES_KHR,
    ePhysicalDeviceFragmentShadingRateEnumsPropertiesNV        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_ENUMS_PROPERTIES_NV,
    ePhysicalDeviceFragmentShadingRateEnumsFeaturesNV          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_ENUMS_FEATURES_NV,
    ePipelineFragmentShadingRateEnumStateCreateInfoNV          = VK_STRUCTURE_TYPE_PIPELINE_FRAGMENT_SHADING_RATE_ENUM_STATE_CREATE_INFO_NV,
    eAccelerationStructureGeometryMotionTrianglesDataNV        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_MOTION_TRIANGLES_DATA_NV,
    ePhysicalDeviceRayTracingMotionBlurFeaturesNV              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MOTION_BLUR_FEATURES_NV,
    eAccelerationStructureMotionInfoNV                         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MOTION_INFO_NV,
    ePhysicalDeviceMeshShaderFeaturesEXT                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
    ePhysicalDeviceMeshShaderPropertiesEXT                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT,
    ePhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_YCBCR_2_PLANE_444_FORMATS_FEATURES_EXT,
    ePhysicalDeviceFragmentDensityMap2FeaturesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_2_FEATURES_EXT,
    ePhysicalDeviceFragmentDensityMap2PropertiesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_2_PROPERTIES_EXT,
    eCopyCommandTransformInfoQCOM                              = VK_STRUCTURE_TYPE_COPY_COMMAND_TRANSFORM_INFO_QCOM,
    ePhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR,
    ePhysicalDeviceImageCompressionControlFeaturesEXT          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_FEATURES_EXT,
    eImageCompressionControlEXT                                = VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT,
    eSubresourceLayout2EXT                                     = VK_STRUCTURE_TYPE_SUBRESOURCE_LAYOUT_2_EXT,
    eImageSubresource2EXT                                      = VK_STRUCTURE_TYPE_IMAGE_SUBRESOURCE_2_EXT,
    eImageCompressionPropertiesEXT                             = VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_PROPERTIES_EXT,
    ePhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_FEATURES_EXT,
    ePhysicalDevice4444FormatsFeaturesEXT                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_4444_FORMATS_FEATURES_EXT,
    ePhysicalDeviceFaultFeaturesEXT                            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT,
    eDeviceFaultCountsEXT                                      = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT,
    eDeviceFaultInfoEXT                                        = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT,
    ePhysicalDeviceRgba10X6FormatsFeaturesEXT                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RGBA10X6_FORMATS_FEATURES_EXT,
#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
    eDirectfbSurfaceCreateInfoEXT = VK_STRUCTURE_TYPE_DIRECTFB_SURFACE_CREATE_INFO_EXT,
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/
    ePhysicalDeviceVertexInputDynamicStateFeaturesEXT      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT,
    eVertexInputBindingDescription2EXT                     = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
    eVertexInputAttributeDescription2EXT                   = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
    ePhysicalDeviceDrmPropertiesEXT                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT,
    ePhysicalDeviceAddressBindingReportFeaturesEXT         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ADDRESS_BINDING_REPORT_FEATURES_EXT,
    eDeviceAddressBindingCallbackDataEXT                   = VK_STRUCTURE_TYPE_DEVICE_ADDRESS_BINDING_CALLBACK_DATA_EXT,
    ePhysicalDeviceDepthClipControlFeaturesEXT             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT,
    ePipelineViewportDepthClipControlCreateInfoEXT         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT,
    ePhysicalDevicePrimitiveTopologyListRestartFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVE_TOPOLOGY_LIST_RESTART_FEATURES_EXT,
#if defined( VK_USE_PLATFORM_FUCHSIA )
    eImportMemoryZirconHandleInfoFUCHSIA     = VK_STRUCTURE_TYPE_IMPORT_MEMORY_ZIRCON_HANDLE_INFO_FUCHSIA,
    eMemoryZirconHandlePropertiesFUCHSIA     = VK_STRUCTURE_TYPE_MEMORY_ZIRCON_HANDLE_PROPERTIES_FUCHSIA,
    eMemoryGetZirconHandleInfoFUCHSIA        = VK_STRUCTURE_TYPE_MEMORY_GET_ZIRCON_HANDLE_INFO_FUCHSIA,
    eImportSemaphoreZirconHandleInfoFUCHSIA  = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_ZIRCON_HANDLE_INFO_FUCHSIA,
    eSemaphoreGetZirconHandleInfoFUCHSIA     = VK_STRUCTURE_TYPE_SEMAPHORE_GET_ZIRCON_HANDLE_INFO_FUCHSIA,
    eBufferCollectionCreateInfoFUCHSIA       = VK_STRUCTURE_TYPE_BUFFER_COLLECTION_CREATE_INFO_FUCHSIA,
    eImportMemoryBufferCollectionFUCHSIA     = VK_STRUCTURE_TYPE_IMPORT_MEMORY_BUFFER_COLLECTION_FUCHSIA,
    eBufferCollectionImageCreateInfoFUCHSIA  = VK_STRUCTURE_TYPE_BUFFER_COLLECTION_IMAGE_CREATE_INFO_FUCHSIA,
    eBufferCollectionPropertiesFUCHSIA       = VK_STRUCTURE_TYPE_BUFFER_COLLECTION_PROPERTIES_FUCHSIA,
    eBufferConstraintsInfoFUCHSIA            = VK_STRUCTURE_TYPE_BUFFER_CONSTRAINTS_INFO_FUCHSIA,
    eBufferCollectionBufferCreateInfoFUCHSIA = VK_STRUCTURE_TYPE_BUFFER_COLLECTION_BUFFER_CREATE_INFO_FUCHSIA,
    eImageConstraintsInfoFUCHSIA             = VK_STRUCTURE_TYPE_IMAGE_CONSTRAINTS_INFO_FUCHSIA,
    eImageFormatConstraintsInfoFUCHSIA       = VK_STRUCTURE_TYPE_IMAGE_FORMAT_CONSTRAINTS_INFO_FUCHSIA,
    eSysmemColorSpaceFUCHSIA                 = VK_STRUCTURE_TYPE_SYSMEM_COLOR_SPACE_FUCHSIA,
    eBufferCollectionConstraintsInfoFUCHSIA  = VK_STRUCTURE_TYPE_BUFFER_COLLECTION_CONSTRAINTS_INFO_FUCHSIA,
#endif /*VK_USE_PLATFORM_FUCHSIA*/
    eSubpassShadingPipelineCreateInfoHUAWEI                     = VK_STRUCTURE_TYPE_SUBPASS_SHADING_PIPELINE_CREATE_INFO_HUAWEI,
    ePhysicalDeviceSubpassShadingFeaturesHUAWEI                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBPASS_SHADING_FEATURES_HUAWEI,
    ePhysicalDeviceSubpassShadingPropertiesHUAWEI               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBPASS_SHADING_PROPERTIES_HUAWEI,
    ePhysicalDeviceInvocationMaskFeaturesHUAWEI                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INVOCATION_MASK_FEATURES_HUAWEI,
    eMemoryGetRemoteAddressInfoNV                               = VK_STRUCTURE_TYPE_MEMORY_GET_REMOTE_ADDRESS_INFO_NV,
    ePhysicalDeviceExternalMemoryRdmaFeaturesNV                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_RDMA_FEATURES_NV,
    ePipelinePropertiesIdentifierEXT                            = VK_STRUCTURE_TYPE_PIPELINE_PROPERTIES_IDENTIFIER_EXT,
    ePhysicalDevicePipelinePropertiesFeaturesEXT                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_PROPERTIES_FEATURES_EXT,
    ePhysicalDeviceMultisampledRenderToSingleSampledFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_FEATURES_EXT,
    eSubpassResolvePerformanceQueryEXT                          = VK_STRUCTURE_TYPE_SUBPASS_RESOLVE_PERFORMANCE_QUERY_EXT,
    eMultisampledRenderToSingleSampledInfoEXT                   = VK_STRUCTURE_TYPE_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_INFO_EXT,
    ePhysicalDeviceExtendedDynamicState2FeaturesEXT             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT,
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
    eScreenSurfaceCreateInfoQNX = VK_STRUCTURE_TYPE_SCREEN_SURFACE_CREATE_INFO_QNX,
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/
    ePhysicalDeviceColorWriteEnableFeaturesEXT                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COLOR_WRITE_ENABLE_FEATURES_EXT,
    ePipelineColorWriteCreateInfoEXT                             = VK_STRUCTURE_TYPE_PIPELINE_COLOR_WRITE_CREATE_INFO_EXT,
    ePhysicalDevicePrimitivesGeneratedQueryFeaturesEXT           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVES_GENERATED_QUERY_FEATURES_EXT,
    ePhysicalDeviceRayTracingMaintenance1FeaturesKHR             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MAINTENANCE_1_FEATURES_KHR,
    ePhysicalDeviceImageViewMinLodFeaturesEXT                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_VIEW_MIN_LOD_FEATURES_EXT,
    eImageViewMinLodCreateInfoEXT                                = VK_STRUCTURE_TYPE_IMAGE_VIEW_MIN_LOD_CREATE_INFO_EXT,
    ePhysicalDeviceMultiDrawFeaturesEXT                          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTI_DRAW_FEATURES_EXT,
    ePhysicalDeviceMultiDrawPropertiesEXT                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTI_DRAW_PROPERTIES_EXT,
    ePhysicalDeviceImage2DViewOf3DFeaturesEXT                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_2D_VIEW_OF_3D_FEATURES_EXT,
    eMicromapBuildInfoEXT                                        = VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT,
    eMicromapVersionInfoEXT                                      = VK_STRUCTURE_TYPE_MICROMAP_VERSION_INFO_EXT,
    eCopyMicromapInfoEXT                                         = VK_STRUCTURE_TYPE_COPY_MICROMAP_INFO_EXT,
    eCopyMicromapToMemoryInfoEXT                                 = VK_STRUCTURE_TYPE_COPY_MICROMAP_TO_MEMORY_INFO_EXT,
    eCopyMemoryToMicromapInfoEXT                                 = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_MICROMAP_INFO_EXT,
    ePhysicalDeviceOpacityMicromapFeaturesEXT                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT,
    ePhysicalDeviceOpacityMicromapPropertiesEXT                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES_EXT,
    eMicromapCreateInfoEXT                                       = VK_STRUCTURE_TYPE_MICROMAP_CREATE_INFO_EXT,
    eMicromapBuildSizesInfoEXT                                   = VK_STRUCTURE_TYPE_MICROMAP_BUILD_SIZES_INFO_EXT,
    eAccelerationStructureTrianglesOpacityMicromapEXT            = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP_EXT,
    ePhysicalDeviceBorderColorSwizzleFeaturesEXT                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BORDER_COLOR_SWIZZLE_FEATURES_EXT,
    eSamplerBorderColorComponentMappingCreateInfoEXT             = VK_STRUCTURE_TYPE_SAMPLER_BORDER_COLOR_COMPONENT_MAPPING_CREATE_INFO_EXT,
    ePhysicalDevicePageableDeviceLocalMemoryFeaturesEXT          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PAGEABLE_DEVICE_LOCAL_MEMORY_FEATURES_EXT,
    ePhysicalDeviceDescriptorSetHostMappingFeaturesVALVE         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_SET_HOST_MAPPING_FEATURES_VALVE,
    eDescriptorSetBindingReferenceVALVE                          = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_BINDING_REFERENCE_VALVE,
    eDescriptorSetLayoutHostMappingInfoVALVE                     = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_HOST_MAPPING_INFO_VALVE,
    ePhysicalDeviceDepthClampZeroOneFeaturesEXT                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLAMP_ZERO_ONE_FEATURES_EXT,
    ePhysicalDeviceNonSeamlessCubeMapFeaturesEXT                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_NON_SEAMLESS_CUBE_MAP_FEATURES_EXT,
    ePhysicalDeviceFragmentDensityMapOffsetFeaturesQCOM          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_OFFSET_FEATURES_QCOM,
    ePhysicalDeviceFragmentDensityMapOffsetPropertiesQCOM        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_OFFSET_PROPERTIES_QCOM,
    eSubpassFragmentDensityMapOffsetEndInfoQCOM                  = VK_STRUCTURE_TYPE_SUBPASS_FRAGMENT_DENSITY_MAP_OFFSET_END_INFO_QCOM,
    ePhysicalDeviceLinearColorAttachmentFeaturesNV               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINEAR_COLOR_ATTACHMENT_FEATURES_NV,
    ePhysicalDeviceImageCompressionControlSwapchainFeaturesEXT   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_SWAPCHAIN_FEATURES_EXT,
    ePhysicalDeviceImageProcessingFeaturesQCOM                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_PROCESSING_FEATURES_QCOM,
    ePhysicalDeviceImageProcessingPropertiesQCOM                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_PROCESSING_PROPERTIES_QCOM,
    eImageViewSampleWeightCreateInfoQCOM                         = VK_STRUCTURE_TYPE_IMAGE_VIEW_SAMPLE_WEIGHT_CREATE_INFO_QCOM,
    ePhysicalDeviceExtendedDynamicState3FeaturesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT,
    ePhysicalDeviceExtendedDynamicState3PropertiesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_PROPERTIES_EXT,
    ePhysicalDeviceSubpassMergeFeedbackFeaturesEXT               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBPASS_MERGE_FEEDBACK_FEATURES_EXT,
    eRenderPassCreationControlEXT                                = VK_STRUCTURE_TYPE_RENDER_PASS_CREATION_CONTROL_EXT,
    eRenderPassCreationFeedbackCreateInfoEXT                     = VK_STRUCTURE_TYPE_RENDER_PASS_CREATION_FEEDBACK_CREATE_INFO_EXT,
    eRenderPassSubpassFeedbackCreateInfoEXT                      = VK_STRUCTURE_TYPE_RENDER_PASS_SUBPASS_FEEDBACK_CREATE_INFO_EXT,
    ePhysicalDeviceShaderModuleIdentifierFeaturesEXT             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MODULE_IDENTIFIER_FEATURES_EXT,
    ePhysicalDeviceShaderModuleIdentifierPropertiesEXT           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MODULE_IDENTIFIER_PROPERTIES_EXT,
    ePipelineShaderStageModuleIdentifierCreateInfoEXT            = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_MODULE_IDENTIFIER_CREATE_INFO_EXT,
    eShaderModuleIdentifierEXT                                   = VK_STRUCTURE_TYPE_SHADER_MODULE_IDENTIFIER_EXT,
    ePhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_FEATURES_EXT,
    ePhysicalDeviceOpticalFlowFeaturesNV                         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV,
    ePhysicalDeviceOpticalFlowPropertiesNV                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_PROPERTIES_NV,
    eOpticalFlowImageFormatInfoNV                                = VK_STRUCTURE_TYPE_OPTICAL_FLOW_IMAGE_FORMAT_INFO_NV,
    eOpticalFlowImageFormatPropertiesNV                          = VK_STRUCTURE_TYPE_OPTICAL_FLOW_IMAGE_FORMAT_PROPERTIES_NV,
    eOpticalFlowSessionCreateInfoNV                              = VK_STRUCTURE_TYPE_OPTICAL_FLOW_SESSION_CREATE_INFO_NV,
    eOpticalFlowExecuteInfoNV                                    = VK_STRUCTURE_TYPE_OPTICAL_FLOW_EXECUTE_INFO_NV,
    eOpticalFlowSessionCreatePrivateDataInfoNV                   = VK_STRUCTURE_TYPE_OPTICAL_FLOW_SESSION_CREATE_PRIVATE_DATA_INFO_NV,
    ePhysicalDeviceLegacyDitheringFeaturesEXT                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LEGACY_DITHERING_FEATURES_EXT,
    ePhysicalDevicePipelineProtectedAccessFeaturesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_PROTECTED_ACCESS_FEATURES_EXT,
    ePhysicalDeviceTilePropertiesFeaturesQCOM                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TILE_PROPERTIES_FEATURES_QCOM,
    eTilePropertiesQCOM                                          = VK_STRUCTURE_TYPE_TILE_PROPERTIES_QCOM,
    ePhysicalDeviceAmigoProfilingFeaturesSEC                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_AMIGO_PROFILING_FEATURES_SEC,
    eAmigoProfilingSubmitInfoSEC                                 = VK_STRUCTURE_TYPE_AMIGO_PROFILING_SUBMIT_INFO_SEC,
    ePhysicalDeviceMutableDescriptorTypeFeaturesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT,
    eMutableDescriptorTypeCreateInfoEXT                          = VK_STRUCTURE_TYPE_MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_EXT,
    eAttachmentDescription2KHR                                   = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2_KHR,
    eAttachmentDescriptionStencilLayoutKHR                       = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_STENCIL_LAYOUT_KHR,
    eAttachmentReference2KHR                                     = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2_KHR,
    eAttachmentReferenceStencilLayoutKHR                         = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_STENCIL_LAYOUT_KHR,
    eAttachmentSampleCountInfoNV                                 = VK_STRUCTURE_TYPE_ATTACHMENT_SAMPLE_COUNT_INFO_NV,
    eBindBufferMemoryDeviceGroupInfoKHR                          = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_DEVICE_GROUP_INFO_KHR,
    eBindBufferMemoryInfoKHR                                     = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO_KHR,
    eBindImageMemoryDeviceGroupInfoKHR                           = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_DEVICE_GROUP_INFO_KHR,
    eBindImageMemoryInfoKHR                                      = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO_KHR,
    eBindImagePlaneMemoryInfoKHR                                 = VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO_KHR,
    eBlitImageInfo2KHR                                           = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2_KHR,
    eBufferCopy2KHR                                              = VK_STRUCTURE_TYPE_BUFFER_COPY_2_KHR,
    eBufferDeviceAddressInfoEXT                                  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_EXT,
    eBufferDeviceAddressInfoKHR                                  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR,
    eBufferImageCopy2KHR                                         = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2_KHR,
    eBufferMemoryBarrier2KHR                                     = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR,
    eBufferMemoryRequirementsInfo2KHR                            = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2_KHR,
    eBufferOpaqueCaptureAddressCreateInfoKHR                     = VK_STRUCTURE_TYPE_BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO_KHR,
    eCommandBufferInheritanceRenderingInfoKHR                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO_KHR,
    eCommandBufferSubmitInfoKHR                                  = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR,
    eCopyBufferInfo2KHR                                          = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2_KHR,
    eCopyBufferToImageInfo2KHR                                   = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2_KHR,
    eCopyImageInfo2KHR                                           = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2_KHR,
    eCopyImageToBufferInfo2KHR                                   = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2_KHR,
    eDebugReportCreateInfoEXT                                    = VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT,
    eDependencyInfoKHR                                           = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
    eDescriptorPoolInlineUniformBlockCreateInfoEXT               = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_INLINE_UNIFORM_BLOCK_CREATE_INFO_EXT,
    eDescriptorSetLayoutBindingFlagsCreateInfoEXT                = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT,
    eDescriptorSetLayoutSupportKHR                               = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT_KHR,
    eDescriptorSetVariableDescriptorCountAllocateInfoEXT         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT,
    eDescriptorSetVariableDescriptorCountLayoutSupportEXT        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_LAYOUT_SUPPORT_EXT,
    eDescriptorUpdateTemplateCreateInfoKHR                       = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO_KHR,
    eDeviceBufferMemoryRequirementsKHR                           = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS_KHR,
    eDeviceGroupBindSparseInfoKHR                                = VK_STRUCTURE_TYPE_DEVICE_GROUP_BIND_SPARSE_INFO_KHR,
    eDeviceGroupCommandBufferBeginInfoKHR                        = VK_STRUCTURE_TYPE_DEVICE_GROUP_COMMAND_BUFFER_BEGIN_INFO_KHR,
    eDeviceGroupDeviceCreateInfoKHR                              = VK_STRUCTURE_TYPE_DEVICE_GROUP_DEVICE_CREATE_INFO_KHR,
    eDeviceGroupRenderPassBeginInfoKHR                           = VK_STRUCTURE_TYPE_DEVICE_GROUP_RENDER_PASS_BEGIN_INFO_KHR,
    eDeviceGroupSubmitInfoKHR                                    = VK_STRUCTURE_TYPE_DEVICE_GROUP_SUBMIT_INFO_KHR,
    eDeviceImageMemoryRequirementsKHR                            = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS_KHR,
    eDeviceMemoryOpaqueCaptureAddressInfoKHR                     = VK_STRUCTURE_TYPE_DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO_KHR,
    eDevicePrivateDataCreateInfoEXT                              = VK_STRUCTURE_TYPE_DEVICE_PRIVATE_DATA_CREATE_INFO_EXT,
    eDeviceQueueGlobalPriorityCreateInfoEXT                      = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_EXT,
    eExportFenceCreateInfoKHR                                    = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO_KHR,
    eExportMemoryAllocateInfoKHR                                 = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR,
    eExportSemaphoreCreateInfoKHR                                = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR,
    eExternalBufferPropertiesKHR                                 = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES_KHR,
    eExternalFencePropertiesKHR                                  = VK_STRUCTURE_TYPE_EXTERNAL_FENCE_PROPERTIES_KHR,
    eExternalImageFormatPropertiesKHR                            = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES_KHR,
    eExternalMemoryBufferCreateInfoKHR                           = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO_KHR,
    eExternalMemoryImageCreateInfoKHR                            = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO_KHR,
    eExternalSemaphorePropertiesKHR                              = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES_KHR,
    eFormatProperties2KHR                                        = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2_KHR,
    eFormatProperties3KHR                                        = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3_KHR,
    eFramebufferAttachmentsCreateInfoKHR                         = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENTS_CREATE_INFO_KHR,
    eFramebufferAttachmentImageInfoKHR                           = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENT_IMAGE_INFO_KHR,
    eImageBlit2KHR                                               = VK_STRUCTURE_TYPE_IMAGE_BLIT_2_KHR,
    eImageCopy2KHR                                               = VK_STRUCTURE_TYPE_IMAGE_COPY_2_KHR,
    eImageFormatListCreateInfoKHR                                = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO_KHR,
    eImageFormatProperties2KHR                                   = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2_KHR,
    eImageMemoryBarrier2KHR                                      = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR,
    eImageMemoryRequirementsInfo2KHR                             = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2_KHR,
    eImagePlaneMemoryRequirementsInfoKHR                         = VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO_KHR,
    eImageResolve2KHR                                            = VK_STRUCTURE_TYPE_IMAGE_RESOLVE_2_KHR,
    eImageSparseMemoryRequirementsInfo2KHR                       = VK_STRUCTURE_TYPE_IMAGE_SPARSE_MEMORY_REQUIREMENTS_INFO_2_KHR,
    eImageStencilUsageCreateInfoEXT                              = VK_STRUCTURE_TYPE_IMAGE_STENCIL_USAGE_CREATE_INFO_EXT,
    eImageViewUsageCreateInfoKHR                                 = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO_KHR,
    eMemoryAllocateFlagsInfoKHR                                  = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_KHR,
    eMemoryBarrier2KHR                                           = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR,
    eMemoryDedicatedAllocateInfoKHR                              = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR,
    eMemoryDedicatedRequirementsKHR                              = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR,
    eMemoryOpaqueCaptureAddressAllocateInfoKHR                   = VK_STRUCTURE_TYPE_MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO_KHR,
    eMemoryRequirements2KHR                                      = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR,
    eMutableDescriptorTypeCreateInfoVALVE                        = VK_STRUCTURE_TYPE_MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_VALVE,
    ePhysicalDevice16BitStorageFeaturesKHR                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR,
    ePhysicalDevice8BitStorageFeaturesKHR                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR,
    ePhysicalDeviceBufferAddressFeaturesEXT                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_ADDRESS_FEATURES_EXT,
    ePhysicalDeviceBufferDeviceAddressFeaturesKHR                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR,
    ePhysicalDeviceDepthStencilResolvePropertiesKHR              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES_KHR,
    ePhysicalDeviceDescriptorIndexingFeaturesEXT                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT,
    ePhysicalDeviceDescriptorIndexingPropertiesEXT               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES_EXT,
    ePhysicalDeviceDriverPropertiesKHR                           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES_KHR,
    ePhysicalDeviceDynamicRenderingFeaturesKHR                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR,
    ePhysicalDeviceExternalBufferInfoKHR                         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO_KHR,
    ePhysicalDeviceExternalFenceInfoKHR                          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_FENCE_INFO_KHR,
    ePhysicalDeviceExternalImageFormatInfoKHR                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO_KHR,
    ePhysicalDeviceExternalSemaphoreInfoKHR                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO_KHR,
    ePhysicalDeviceFeatures2KHR                                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR,
    ePhysicalDeviceFloat16Int8FeaturesKHR                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR,
    ePhysicalDeviceFloatControlsPropertiesKHR                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES_KHR,
    ePhysicalDeviceFragmentShaderBarycentricFeaturesNV           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_NV,
    ePhysicalDeviceGlobalPriorityQueryFeaturesEXT                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GLOBAL_PRIORITY_QUERY_FEATURES_EXT,
    ePhysicalDeviceGroupPropertiesKHR                            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES_KHR,
    ePhysicalDeviceHostQueryResetFeaturesEXT                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES_EXT,
    ePhysicalDeviceIdPropertiesKHR                               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES_KHR,
    ePhysicalDeviceImagelessFramebufferFeaturesKHR               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES_KHR,
    ePhysicalDeviceImageFormatInfo2KHR                           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2_KHR,
    ePhysicalDeviceImageRobustnessFeaturesEXT                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES_EXT,
    ePhysicalDeviceInlineUniformBlockFeaturesEXT                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES_EXT,
    ePhysicalDeviceInlineUniformBlockPropertiesEXT               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_PROPERTIES_EXT,
    ePhysicalDeviceMaintenance3PropertiesKHR                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES_KHR,
    ePhysicalDeviceMaintenance4FeaturesKHR                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES_KHR,
    ePhysicalDeviceMaintenance4PropertiesKHR                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES_KHR,
    ePhysicalDeviceMemoryProperties2KHR                          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR,
    ePhysicalDeviceMultiviewFeaturesKHR                          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES_KHR,
    ePhysicalDeviceMultiviewPropertiesKHR                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES_KHR,
    ePhysicalDeviceMutableDescriptorTypeFeaturesVALVE            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_VALVE,
    ePhysicalDevicePipelineCreationCacheControlFeaturesEXT       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES_EXT,
    ePhysicalDevicePointClippingPropertiesKHR                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_POINT_CLIPPING_PROPERTIES_KHR,
    ePhysicalDevicePrivateDataFeaturesEXT                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES_EXT,
    ePhysicalDeviceProperties2KHR                                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR,
    ePhysicalDeviceRasterizationOrderAttachmentAccessFeaturesARM = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_FEATURES_ARM,
    ePhysicalDeviceSamplerFilterMinmaxPropertiesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_FILTER_MINMAX_PROPERTIES_EXT,
    ePhysicalDeviceSamplerYcbcrConversionFeaturesKHR             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES_KHR,
    ePhysicalDeviceScalarBlockLayoutFeaturesEXT                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES_EXT,
    ePhysicalDeviceSeparateDepthStencilLayoutsFeaturesKHR        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES_KHR,
    ePhysicalDeviceShaderAtomicInt64FeaturesKHR                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES_KHR,
    ePhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES_EXT,
    ePhysicalDeviceShaderDrawParameterFeatures                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETER_FEATURES,
    ePhysicalDeviceShaderFloat16Int8FeaturesKHR                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR,
    ePhysicalDeviceShaderIntegerDotProductFeaturesKHR            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR,
    ePhysicalDeviceShaderIntegerDotProductPropertiesKHR          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES_KHR,
    ePhysicalDeviceShaderSubgroupExtendedTypesFeaturesKHR        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES_KHR,
    ePhysicalDeviceShaderTerminateInvocationFeaturesKHR          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES_KHR,
    ePhysicalDeviceSparseImageFormatInfo2KHR                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SPARSE_IMAGE_FORMAT_INFO_2_KHR,
    ePhysicalDeviceSubgroupSizeControlFeaturesEXT                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT,
    ePhysicalDeviceSubgroupSizeControlPropertiesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT,
    ePhysicalDeviceSynchronization2FeaturesKHR                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR,
    ePhysicalDeviceTexelBufferAlignmentPropertiesEXT             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_PROPERTIES_EXT,
    ePhysicalDeviceTextureCompressionAstcHdrFeaturesEXT          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES_EXT,
    ePhysicalDeviceTimelineSemaphoreFeaturesKHR                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR,
    ePhysicalDeviceTimelineSemaphorePropertiesKHR                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES_KHR,
    ePhysicalDeviceToolPropertiesEXT                             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TOOL_PROPERTIES_EXT,
    ePhysicalDeviceUniformBufferStandardLayoutFeaturesKHR        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES_KHR,
    ePhysicalDeviceVariablePointersFeaturesKHR                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES_KHR,
    ePhysicalDeviceVariablePointerFeatures                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTER_FEATURES,
    ePhysicalDeviceVariablePointerFeaturesKHR                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTER_FEATURES_KHR,
    ePhysicalDeviceVulkanMemoryModelFeaturesKHR                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES_KHR,
    ePhysicalDeviceZeroInitializeWorkgroupMemoryFeaturesKHR      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES_KHR,
    ePipelineCreationFeedbackCreateInfoEXT                       = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO_EXT,
    ePipelineInfoEXT                                             = VK_STRUCTURE_TYPE_PIPELINE_INFO_EXT,
    ePipelineRenderingCreateInfoKHR                              = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
    ePipelineShaderStageRequiredSubgroupSizeCreateInfoEXT        = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
    ePipelineTessellationDomainOriginStateCreateInfoKHR          = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_DOMAIN_ORIGIN_STATE_CREATE_INFO_KHR,
    ePrivateDataSlotCreateInfoEXT                                = VK_STRUCTURE_TYPE_PRIVATE_DATA_SLOT_CREATE_INFO_EXT,
    eQueryPoolCreateInfoINTEL                                    = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO_INTEL,
    eQueueFamilyGlobalPriorityPropertiesEXT                      = VK_STRUCTURE_TYPE_QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES_EXT,
    eQueueFamilyProperties2KHR                                   = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2_KHR,
    eRenderingAttachmentInfoKHR                                  = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
    eRenderingInfoKHR                                            = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
    eRenderPassAttachmentBeginInfoKHR                            = VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO_KHR,
    eRenderPassCreateInfo2KHR                                    = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2_KHR,
    eRenderPassInputAttachmentAspectCreateInfoKHR                = VK_STRUCTURE_TYPE_RENDER_PASS_INPUT_ATTACHMENT_ASPECT_CREATE_INFO_KHR,
    eRenderPassMultiviewCreateInfoKHR                            = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO_KHR,
    eResolveImageInfo2KHR                                        = VK_STRUCTURE_TYPE_RESOLVE_IMAGE_INFO_2_KHR,
    eSamplerReductionModeCreateInfoEXT                           = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO_EXT,
    eSamplerYcbcrConversionCreateInfoKHR                         = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO_KHR,
    eSamplerYcbcrConversionImageFormatPropertiesKHR              = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES_KHR,
    eSamplerYcbcrConversionInfoKHR                               = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO_KHR,
    eSemaphoreSignalInfoKHR                                      = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO_KHR,
    eSemaphoreSubmitInfoKHR                                      = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR,
    eSemaphoreTypeCreateInfoKHR                                  = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR,
    eSemaphoreWaitInfoKHR                                        = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO_KHR,
    eSparseImageFormatProperties2KHR                             = VK_STRUCTURE_TYPE_SPARSE_IMAGE_FORMAT_PROPERTIES_2_KHR,
    eSparseImageMemoryRequirements2KHR                           = VK_STRUCTURE_TYPE_SPARSE_IMAGE_MEMORY_REQUIREMENTS_2_KHR,
    eSubmitInfo2KHR                                              = VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR,
    eSubpassBeginInfoKHR                                         = VK_STRUCTURE_TYPE_SUBPASS_BEGIN_INFO_KHR,
    eSubpassDependency2KHR                                       = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2_KHR,
    eSubpassDescription2KHR                                      = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2_KHR,
    eSubpassDescriptionDepthStencilResolveKHR                    = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE_KHR,
    eSubpassEndInfoKHR                                           = VK_STRUCTURE_TYPE_SUBPASS_END_INFO_KHR,
    eTimelineSemaphoreSubmitInfoKHR                              = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR,
    eWriteDescriptorSetInlineUniformBlockEXT                     = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK_EXT
  };

  enum class PipelineCacheHeaderVersion
  {
    eOne = VK_PIPELINE_CACHE_HEADER_VERSION_ONE
  };

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
    ePrivateDataSlot          = VK_OBJECT_TYPE_PRIVATE_DATA_SLOT,
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
#if defined( VK_USE_PLATFORM_FUCHSIA )
    eBufferCollectionFUCHSIA = VK_OBJECT_TYPE_BUFFER_COLLECTION_FUCHSIA,
#endif /*VK_USE_PLATFORM_FUCHSIA*/
    eMicromapEXT                 = VK_OBJECT_TYPE_MICROMAP_EXT,
    eOpticalFlowSessionNV        = VK_OBJECT_TYPE_OPTICAL_FLOW_SESSION_NV,
    eDescriptorUpdateTemplateKHR = VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_KHR,
    ePrivateDataSlotEXT          = VK_OBJECT_TYPE_PRIVATE_DATA_SLOT_EXT,
    eSamplerYcbcrConversionKHR   = VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION_KHR
  };

  enum class VendorId
  {
    eVIV      = VK_VENDOR_ID_VIV,
    eVSI      = VK_VENDOR_ID_VSI,
    eKazan    = VK_VENDOR_ID_KAZAN,
    eCodeplay = VK_VENDOR_ID_CODEPLAY,
    eMESA     = VK_VENDOR_ID_MESA,
    ePocl     = VK_VENDOR_ID_POCL
  };

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
    eG8B8R82Plane444Unorm                    = VK_FORMAT_G8_B8R8_2PLANE_444_UNORM,
    eG10X6B10X6R10X62Plane444Unorm3Pack16    = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16,
    eG12X4B12X4R12X42Plane444Unorm3Pack16    = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16,
    eG16B16R162Plane444Unorm                 = VK_FORMAT_G16_B16R16_2PLANE_444_UNORM,
    eA4R4G4B4UnormPack16                     = VK_FORMAT_A4R4G4B4_UNORM_PACK16,
    eA4B4G4R4UnormPack16                     = VK_FORMAT_A4B4G4R4_UNORM_PACK16,
    eAstc4x4SfloatBlock                      = VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK,
    eAstc5x4SfloatBlock                      = VK_FORMAT_ASTC_5x4_SFLOAT_BLOCK,
    eAstc5x5SfloatBlock                      = VK_FORMAT_ASTC_5x5_SFLOAT_BLOCK,
    eAstc6x5SfloatBlock                      = VK_FORMAT_ASTC_6x5_SFLOAT_BLOCK,
    eAstc6x6SfloatBlock                      = VK_FORMAT_ASTC_6x6_SFLOAT_BLOCK,
    eAstc8x5SfloatBlock                      = VK_FORMAT_ASTC_8x5_SFLOAT_BLOCK,
    eAstc8x6SfloatBlock                      = VK_FORMAT_ASTC_8x6_SFLOAT_BLOCK,
    eAstc8x8SfloatBlock                      = VK_FORMAT_ASTC_8x8_SFLOAT_BLOCK,
    eAstc10x5SfloatBlock                     = VK_FORMAT_ASTC_10x5_SFLOAT_BLOCK,
    eAstc10x6SfloatBlock                     = VK_FORMAT_ASTC_10x6_SFLOAT_BLOCK,
    eAstc10x8SfloatBlock                     = VK_FORMAT_ASTC_10x8_SFLOAT_BLOCK,
    eAstc10x10SfloatBlock                    = VK_FORMAT_ASTC_10x10_SFLOAT_BLOCK,
    eAstc12x10SfloatBlock                    = VK_FORMAT_ASTC_12x10_SFLOAT_BLOCK,
    eAstc12x12SfloatBlock                    = VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK,
    ePvrtc12BppUnormBlockIMG                 = VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG,
    ePvrtc14BppUnormBlockIMG                 = VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG,
    ePvrtc22BppUnormBlockIMG                 = VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG,
    ePvrtc24BppUnormBlockIMG                 = VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG,
    ePvrtc12BppSrgbBlockIMG                  = VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG,
    ePvrtc14BppSrgbBlockIMG                  = VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG,
    ePvrtc22BppSrgbBlockIMG                  = VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG,
    ePvrtc24BppSrgbBlockIMG                  = VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG,
    eR16G16S105NV                            = VK_FORMAT_R16G16_S10_5_NV,
    eA4B4G4R4UnormPack16EXT                  = VK_FORMAT_A4B4G4R4_UNORM_PACK16_EXT,
    eA4R4G4B4UnormPack16EXT                  = VK_FORMAT_A4R4G4B4_UNORM_PACK16_EXT,
    eAstc10x10SfloatBlockEXT                 = VK_FORMAT_ASTC_10x10_SFLOAT_BLOCK_EXT,
    eAstc10x5SfloatBlockEXT                  = VK_FORMAT_ASTC_10x5_SFLOAT_BLOCK_EXT,
    eAstc10x6SfloatBlockEXT                  = VK_FORMAT_ASTC_10x6_SFLOAT_BLOCK_EXT,
    eAstc10x8SfloatBlockEXT                  = VK_FORMAT_ASTC_10x8_SFLOAT_BLOCK_EXT,
    eAstc12x10SfloatBlockEXT                 = VK_FORMAT_ASTC_12x10_SFLOAT_BLOCK_EXT,
    eAstc12x12SfloatBlockEXT                 = VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK_EXT,
    eAstc4x4SfloatBlockEXT                   = VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK_EXT,
    eAstc5x4SfloatBlockEXT                   = VK_FORMAT_ASTC_5x4_SFLOAT_BLOCK_EXT,
    eAstc5x5SfloatBlockEXT                   = VK_FORMAT_ASTC_5x5_SFLOAT_BLOCK_EXT,
    eAstc6x5SfloatBlockEXT                   = VK_FORMAT_ASTC_6x5_SFLOAT_BLOCK_EXT,
    eAstc6x6SfloatBlockEXT                   = VK_FORMAT_ASTC_6x6_SFLOAT_BLOCK_EXT,
    eAstc8x5SfloatBlockEXT                   = VK_FORMAT_ASTC_8x5_SFLOAT_BLOCK_EXT,
    eAstc8x6SfloatBlockEXT                   = VK_FORMAT_ASTC_8x6_SFLOAT_BLOCK_EXT,
    eAstc8x8SfloatBlockEXT                   = VK_FORMAT_ASTC_8x8_SFLOAT_BLOCK_EXT,
    eB10X6G10X6R10X6G10X6422Unorm4Pack16KHR  = VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16_KHR,
    eB12X4G12X4R12X4G12X4422Unorm4Pack16KHR  = VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16_KHR,
    eB16G16R16G16422UnormKHR                 = VK_FORMAT_B16G16R16G16_422_UNORM_KHR,
    eB8G8R8G8422UnormKHR                     = VK_FORMAT_B8G8R8G8_422_UNORM_KHR,
    eG10X6B10X6G10X6R10X6422Unorm4Pack16KHR  = VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16_KHR,
    eG10X6B10X6R10X62Plane420Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16_KHR,
    eG10X6B10X6R10X62Plane422Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16_KHR,
    eG10X6B10X6R10X62Plane444Unorm3Pack16EXT = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16_EXT,
    eG10X6B10X6R10X63Plane420Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16_KHR,
    eG10X6B10X6R10X63Plane422Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16_KHR,
    eG10X6B10X6R10X63Plane444Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16_KHR,
    eG12X4B12X4G12X4R12X4422Unorm4Pack16KHR  = VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16_KHR,
    eG12X4B12X4R12X42Plane420Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16_KHR,
    eG12X4B12X4R12X42Plane422Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16_KHR,
    eG12X4B12X4R12X42Plane444Unorm3Pack16EXT = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16_EXT,
    eG12X4B12X4R12X43Plane420Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16_KHR,
    eG12X4B12X4R12X43Plane422Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16_KHR,
    eG12X4B12X4R12X43Plane444Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16_KHR,
    eG16B16G16R16422UnormKHR                 = VK_FORMAT_G16B16G16R16_422_UNORM_KHR,
    eG16B16R162Plane420UnormKHR              = VK_FORMAT_G16_B16R16_2PLANE_420_UNORM_KHR,
    eG16B16R162Plane422UnormKHR              = VK_FORMAT_G16_B16R16_2PLANE_422_UNORM_KHR,
    eG16B16R162Plane444UnormEXT              = VK_FORMAT_G16_B16R16_2PLANE_444_UNORM_EXT,
    eG16B16R163Plane420UnormKHR              = VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM_KHR,
    eG16B16R163Plane422UnormKHR              = VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM_KHR,
    eG16B16R163Plane444UnormKHR              = VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM_KHR,
    eG8B8G8R8422UnormKHR                     = VK_FORMAT_G8B8G8R8_422_UNORM_KHR,
    eG8B8R82Plane420UnormKHR                 = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM_KHR,
    eG8B8R82Plane422UnormKHR                 = VK_FORMAT_G8_B8R8_2PLANE_422_UNORM_KHR,
    eG8B8R82Plane444UnormEXT                 = VK_FORMAT_G8_B8R8_2PLANE_444_UNORM_EXT,
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

  enum class FormatFeatureFlagBits : VkFormatFeatureFlags
  {
    eSampledImage                                            = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT,
    eStorageImage                                            = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT,
    eStorageImageAtomic                                      = VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT,
    eUniformTexelBuffer                                      = VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT,
    eStorageTexelBuffer                                      = VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT,
    eStorageTexelBufferAtomic                                = VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT,
    eVertexBuffer                                            = VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT,
    eColorAttachment                                         = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT,
    eColorAttachmentBlend                                    = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT,
    eDepthStencilAttachment                                  = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT,
    eBlitSrc                                                 = VK_FORMAT_FEATURE_BLIT_SRC_BIT,
    eBlitDst                                                 = VK_FORMAT_FEATURE_BLIT_DST_BIT,
    eSampledImageFilterLinear                                = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT,
    eTransferSrc                                             = VK_FORMAT_FEATURE_TRANSFER_SRC_BIT,
    eTransferDst                                             = VK_FORMAT_FEATURE_TRANSFER_DST_BIT,
    eMidpointChromaSamples                                   = VK_FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT,
    eSampledImageYcbcrConversionLinearFilter                 = VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT,
    eSampledImageYcbcrConversionSeparateReconstructionFilter = VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT,
    eSampledImageYcbcrConversionChromaReconstructionExplicit = VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_BIT,
    eSampledImageYcbcrConversionChromaReconstructionExplicitForceable =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE_BIT,
    eDisjoint                 = VK_FORMAT_FEATURE_DISJOINT_BIT,
    eCositedChromaSamples     = VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT,
    eSampledImageFilterMinmax = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_MINMAX_BIT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoDecodeOutputKHR = VK_FORMAT_FEATURE_VIDEO_DECODE_OUTPUT_BIT_KHR,
    eVideoDecodeDpbKHR    = VK_FORMAT_FEATURE_VIDEO_DECODE_DPB_BIT_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eAccelerationStructureVertexBufferKHR = VK_FORMAT_FEATURE_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR,
    eSampledImageFilterCubicEXT           = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_CUBIC_BIT_EXT,
    eFragmentDensityMapEXT                = VK_FORMAT_FEATURE_FRAGMENT_DENSITY_MAP_BIT_EXT,
    eFragmentShadingRateAttachmentKHR     = VK_FORMAT_FEATURE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoEncodeInputKHR = VK_FORMAT_FEATURE_VIDEO_ENCODE_INPUT_BIT_KHR,
    eVideoEncodeDpbKHR   = VK_FORMAT_FEATURE_VIDEO_ENCODE_DPB_BIT_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eCositedChromaSamplesKHR                                    = VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT_KHR,
    eDisjointKHR                                                = VK_FORMAT_FEATURE_DISJOINT_BIT_KHR,
    eMidpointChromaSamplesKHR                                   = VK_FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT_KHR,
    eSampledImageFilterCubicIMG                                 = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_CUBIC_BIT_IMG,
    eSampledImageFilterMinmaxEXT                                = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_MINMAX_BIT_EXT,
    eSampledImageYcbcrConversionChromaReconstructionExplicitKHR = VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_BIT_KHR,
    eSampledImageYcbcrConversionChromaReconstructionExplicitForceableKHR =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE_BIT_KHR,
    eSampledImageYcbcrConversionLinearFilterKHR                 = VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT_KHR,
    eSampledImageYcbcrConversionSeparateReconstructionFilterKHR = VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT_KHR,
    eTransferDstKHR                                             = VK_FORMAT_FEATURE_TRANSFER_DST_BIT_KHR,
    eTransferSrcKHR                                             = VK_FORMAT_FEATURE_TRANSFER_SRC_BIT_KHR
  };

  enum class ImageCreateFlagBits : VkImageCreateFlags
  {
    eSparseBinding                        = VK_IMAGE_CREATE_SPARSE_BINDING_BIT,
    eSparseResidency                      = VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
    eSparseAliased                        = VK_IMAGE_CREATE_SPARSE_ALIASED_BIT,
    eMutableFormat                        = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
    eCubeCompatible                       = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
    eAlias                                = VK_IMAGE_CREATE_ALIAS_BIT,
    eSplitInstanceBindRegions             = VK_IMAGE_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT,
    e2DArrayCompatible                    = VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT,
    eBlockTexelViewCompatible             = VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT,
    eExtendedUsage                        = VK_IMAGE_CREATE_EXTENDED_USAGE_BIT,
    eProtected                            = VK_IMAGE_CREATE_PROTECTED_BIT,
    eDisjoint                             = VK_IMAGE_CREATE_DISJOINT_BIT,
    eCornerSampledNV                      = VK_IMAGE_CREATE_CORNER_SAMPLED_BIT_NV,
    eSampleLocationsCompatibleDepthEXT    = VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT,
    eSubsampledEXT                        = VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT,
    eMultisampledRenderToSingleSampledEXT = VK_IMAGE_CREATE_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_BIT_EXT,
    e2DViewCompatibleEXT                  = VK_IMAGE_CREATE_2D_VIEW_COMPATIBLE_BIT_EXT,
    eFragmentDensityMapOffsetQCOM         = VK_IMAGE_CREATE_FRAGMENT_DENSITY_MAP_OFFSET_BIT_QCOM,
    e2DArrayCompatibleKHR                 = VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT_KHR,
    eAliasKHR                             = VK_IMAGE_CREATE_ALIAS_BIT_KHR,
    eBlockTexelViewCompatibleKHR          = VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT_KHR,
    eDisjointKHR                          = VK_IMAGE_CREATE_DISJOINT_BIT_KHR,
    eExtendedUsageKHR                     = VK_IMAGE_CREATE_EXTENDED_USAGE_BIT_KHR,
    eSplitInstanceBindRegionsKHR          = VK_IMAGE_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT_KHR
  };

  enum class ImageTiling
  {
    eOptimal              = VK_IMAGE_TILING_OPTIMAL,
    eLinear               = VK_IMAGE_TILING_LINEAR,
    eDrmFormatModifierEXT = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT
  };

  enum class ImageType
  {
    e1D = VK_IMAGE_TYPE_1D,
    e2D = VK_IMAGE_TYPE_2D,
    e3D = VK_IMAGE_TYPE_3D
  };

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
    eAttachmentFeedbackLoopEXT = VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT,
    eInvocationMaskHUAWEI      = VK_IMAGE_USAGE_INVOCATION_MASK_BIT_HUAWEI,
    eSampleWeightQCOM          = VK_IMAGE_USAGE_SAMPLE_WEIGHT_BIT_QCOM,
    eSampleBlockMatchQCOM      = VK_IMAGE_USAGE_SAMPLE_BLOCK_MATCH_BIT_QCOM,
    eShadingRateImageNV        = VK_IMAGE_USAGE_SHADING_RATE_IMAGE_BIT_NV
  };

  enum class InstanceCreateFlagBits : VkInstanceCreateFlags
  {
    eEnumeratePortabilityKHR = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
  };

  enum class InternalAllocationType
  {
    eExecutable = VK_INTERNAL_ALLOCATION_TYPE_EXECUTABLE
  };

  enum class MemoryHeapFlagBits : VkMemoryHeapFlags
  {
    eDeviceLocal      = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT,
    eMultiInstance    = VK_MEMORY_HEAP_MULTI_INSTANCE_BIT,
    eMultiInstanceKHR = VK_MEMORY_HEAP_MULTI_INSTANCE_BIT_KHR
  };

  enum class MemoryPropertyFlagBits : VkMemoryPropertyFlags
  {
    eDeviceLocal       = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    eHostVisible       = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
    eHostCoherent      = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    eHostCached        = VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
    eLazilyAllocated   = VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT,
    eProtected         = VK_MEMORY_PROPERTY_PROTECTED_BIT,
    eDeviceCoherentAMD = VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD,
    eDeviceUncachedAMD = VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD,
    eRdmaCapableNV     = VK_MEMORY_PROPERTY_RDMA_CAPABLE_BIT_NV
  };

  enum class PhysicalDeviceType
  {
    eOther         = VK_PHYSICAL_DEVICE_TYPE_OTHER,
    eIntegratedGpu = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
    eDiscreteGpu   = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
    eVirtualGpu    = VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU,
    eCpu           = VK_PHYSICAL_DEVICE_TYPE_CPU
  };

  enum class QueueFlagBits : VkQueueFlags
  {
    eGraphics      = VK_QUEUE_GRAPHICS_BIT,
    eCompute       = VK_QUEUE_COMPUTE_BIT,
    eTransfer      = VK_QUEUE_TRANSFER_BIT,
    eSparseBinding = VK_QUEUE_SPARSE_BINDING_BIT,
    eProtected     = VK_QUEUE_PROTECTED_BIT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoDecodeKHR = VK_QUEUE_VIDEO_DECODE_BIT_KHR,
    eVideoEncodeKHR = VK_QUEUE_VIDEO_ENCODE_BIT_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eOpticalFlowNV = VK_QUEUE_OPTICAL_FLOW_BIT_NV
  };

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

  enum class SystemAllocationScope
  {
    eCommand  = VK_SYSTEM_ALLOCATION_SCOPE_COMMAND,
    eObject   = VK_SYSTEM_ALLOCATION_SCOPE_OBJECT,
    eCache    = VK_SYSTEM_ALLOCATION_SCOPE_CACHE,
    eDevice   = VK_SYSTEM_ALLOCATION_SCOPE_DEVICE,
    eInstance = VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE
  };

  enum class DeviceCreateFlagBits
  {
  };

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
    eNone                             = VK_PIPELINE_STAGE_NONE,
    eTransformFeedbackEXT             = VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
    eConditionalRenderingEXT          = VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT,
    eAccelerationStructureBuildKHR    = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    eRayTracingShaderKHR              = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
    eFragmentDensityProcessEXT        = VK_PIPELINE_STAGE_FRAGMENT_DENSITY_PROCESS_BIT_EXT,
    eFragmentShadingRateAttachmentKHR = VK_PIPELINE_STAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
    eCommandPreprocessNV              = VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_NV,
    eTaskShaderEXT                    = VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT,
    eMeshShaderEXT                    = VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT,
    eAccelerationStructureBuildNV     = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
    eMeshShaderNV                     = VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV,
    eNoneKHR                          = VK_PIPELINE_STAGE_NONE_KHR,
    eRayTracingShaderNV               = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_NV,
    eShadingRateImageNV               = VK_PIPELINE_STAGE_SHADING_RATE_IMAGE_BIT_NV,
    eTaskShaderNV                     = VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV
  };

  enum class MemoryMapFlagBits : VkMemoryMapFlags
  {
  };

  enum class ImageAspectFlagBits : VkImageAspectFlags
  {
    eColor           = VK_IMAGE_ASPECT_COLOR_BIT,
    eDepth           = VK_IMAGE_ASPECT_DEPTH_BIT,
    eStencil         = VK_IMAGE_ASPECT_STENCIL_BIT,
    eMetadata        = VK_IMAGE_ASPECT_METADATA_BIT,
    ePlane0          = VK_IMAGE_ASPECT_PLANE_0_BIT,
    ePlane1          = VK_IMAGE_ASPECT_PLANE_1_BIT,
    ePlane2          = VK_IMAGE_ASPECT_PLANE_2_BIT,
    eNone            = VK_IMAGE_ASPECT_NONE,
    eMemoryPlane0EXT = VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT,
    eMemoryPlane1EXT = VK_IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT,
    eMemoryPlane2EXT = VK_IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT,
    eMemoryPlane3EXT = VK_IMAGE_ASPECT_MEMORY_PLANE_3_BIT_EXT,
    eNoneKHR         = VK_IMAGE_ASPECT_NONE_KHR,
    ePlane0KHR       = VK_IMAGE_ASPECT_PLANE_0_BIT_KHR,
    ePlane1KHR       = VK_IMAGE_ASPECT_PLANE_1_BIT_KHR,
    ePlane2KHR       = VK_IMAGE_ASPECT_PLANE_2_BIT_KHR
  };

  enum class SparseImageFormatFlagBits : VkSparseImageFormatFlags
  {
    eSingleMiptail        = VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT,
    eAlignedMipSize       = VK_SPARSE_IMAGE_FORMAT_ALIGNED_MIP_SIZE_BIT,
    eNonstandardBlockSize = VK_SPARSE_IMAGE_FORMAT_NONSTANDARD_BLOCK_SIZE_BIT
  };

  enum class SparseMemoryBindFlagBits : VkSparseMemoryBindFlags
  {
    eMetadata = VK_SPARSE_MEMORY_BIND_METADATA_BIT
  };

  enum class FenceCreateFlagBits : VkFenceCreateFlags
  {
    eSignaled = VK_FENCE_CREATE_SIGNALED_BIT
  };

  enum class SemaphoreCreateFlagBits : VkSemaphoreCreateFlags
  {
  };

  enum class EventCreateFlagBits : VkEventCreateFlags
  {
    eDeviceOnly    = VK_EVENT_CREATE_DEVICE_ONLY_BIT,
    eDeviceOnlyKHR = VK_EVENT_CREATE_DEVICE_ONLY_BIT_KHR
  };

  enum class QueryPipelineStatisticFlagBits : VkQueryPipelineStatisticFlags
  {
    eInputAssemblyVertices                   = VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT,
    eInputAssemblyPrimitives                 = VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT,
    eVertexShaderInvocations                 = VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT,
    eGeometryShaderInvocations               = VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT,
    eGeometryShaderPrimitives                = VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_PRIMITIVES_BIT,
    eClippingInvocations                     = VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT,
    eClippingPrimitives                      = VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT,
    eFragmentShaderInvocations               = VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT,
    eTessellationControlShaderPatches        = VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT,
    eTessellationEvaluationShaderInvocations = VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_EVALUATION_SHADER_INVOCATIONS_BIT,
    eComputeShaderInvocations                = VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT,
    eTaskShaderInvocationsEXT                = VK_QUERY_PIPELINE_STATISTIC_TASK_SHADER_INVOCATIONS_BIT_EXT,
    eMeshShaderInvocationsEXT                = VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT
  };

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
    eVideoEncodeBitstreamBufferRangeKHR = VK_QUERY_TYPE_VIDEO_ENCODE_BITSTREAM_BUFFER_RANGE_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eMeshPrimitivesGeneratedEXT                               = VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT,
    ePrimitivesGeneratedEXT                                   = VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT,
    eAccelerationStructureSerializationBottomLevelPointersKHR = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR,
    eAccelerationStructureSizeKHR                             = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR,
    eMicromapSerializationSizeEXT                             = VK_QUERY_TYPE_MICROMAP_SERIALIZATION_SIZE_EXT,
    eMicromapCompactedSizeEXT                                 = VK_QUERY_TYPE_MICROMAP_COMPACTED_SIZE_EXT
  };

  enum class QueryPoolCreateFlagBits
  {
  };

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
    eMicromapBuildInputReadOnlyEXT = VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT,
    eMicromapStorageEXT            = VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT,
    eRayTracingNV                  = VK_BUFFER_USAGE_RAY_TRACING_BIT_NV,
    eShaderDeviceAddressEXT        = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_EXT,
    eShaderDeviceAddressKHR        = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR
  };

  enum class SharingMode
  {
    eExclusive  = VK_SHARING_MODE_EXCLUSIVE,
    eConcurrent = VK_SHARING_MODE_CONCURRENT
  };

  enum class BufferViewCreateFlagBits : VkBufferViewCreateFlags
  {
  };

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
    eReadOnlyOptimal                       = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
    eAttachmentOptimal                     = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
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
    eAttachmentFeedbackLoopOptimalEXT         = VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT,
    eAttachmentOptimalKHR                     = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR,
    eDepthAttachmentOptimalKHR                = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL_KHR,
    eDepthAttachmentStencilReadOnlyOptimalKHR = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL_KHR,
    eDepthReadOnlyOptimalKHR                  = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL_KHR,
    eDepthReadOnlyStencilAttachmentOptimalKHR = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL_KHR,
    eReadOnlyOptimalKHR                       = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL_KHR,
    eShadingRateOptimalNV                     = VK_IMAGE_LAYOUT_SHADING_RATE_OPTIMAL_NV,
    eStencilAttachmentOptimalKHR              = VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL_KHR,
    eStencilReadOnlyOptimalKHR                = VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL_KHR
  };

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

  enum class ImageViewCreateFlagBits : VkImageViewCreateFlags
  {
    eFragmentDensityMapDynamicEXT  = VK_IMAGE_VIEW_CREATE_FRAGMENT_DENSITY_MAP_DYNAMIC_BIT_EXT,
    eFragmentDensityMapDeferredEXT = VK_IMAGE_VIEW_CREATE_FRAGMENT_DENSITY_MAP_DEFERRED_BIT_EXT
  };

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

  enum class ShaderModuleCreateFlagBits : VkShaderModuleCreateFlags
  {
  };

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

  enum class ColorComponentFlagBits : VkColorComponentFlags
  {
    eR = VK_COLOR_COMPONENT_R_BIT,
    eG = VK_COLOR_COMPONENT_G_BIT,
    eB = VK_COLOR_COMPONENT_B_BIT,
    eA = VK_COLOR_COMPONENT_A_BIT
  };

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

  enum class CullModeFlagBits : VkCullModeFlags
  {
    eNone         = VK_CULL_MODE_NONE,
    eFront        = VK_CULL_MODE_FRONT_BIT,
    eBack         = VK_CULL_MODE_BACK_BIT,
    eFrontAndBack = VK_CULL_MODE_FRONT_AND_BACK
  };

  enum class DynamicState
  {
    eViewport                            = VK_DYNAMIC_STATE_VIEWPORT,
    eScissor                             = VK_DYNAMIC_STATE_SCISSOR,
    eLineWidth                           = VK_DYNAMIC_STATE_LINE_WIDTH,
    eDepthBias                           = VK_DYNAMIC_STATE_DEPTH_BIAS,
    eBlendConstants                      = VK_DYNAMIC_STATE_BLEND_CONSTANTS,
    eDepthBounds                         = VK_DYNAMIC_STATE_DEPTH_BOUNDS,
    eStencilCompareMask                  = VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
    eStencilWriteMask                    = VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
    eStencilReference                    = VK_DYNAMIC_STATE_STENCIL_REFERENCE,
    eCullMode                            = VK_DYNAMIC_STATE_CULL_MODE,
    eFrontFace                           = VK_DYNAMIC_STATE_FRONT_FACE,
    ePrimitiveTopology                   = VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
    eViewportWithCount                   = VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT,
    eScissorWithCount                    = VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT,
    eVertexInputBindingStride            = VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE,
    eDepthTestEnable                     = VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
    eDepthWriteEnable                    = VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
    eDepthCompareOp                      = VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
    eDepthBoundsTestEnable               = VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE,
    eStencilTestEnable                   = VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
    eStencilOp                           = VK_DYNAMIC_STATE_STENCIL_OP,
    eRasterizerDiscardEnable             = VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE,
    eDepthBiasEnable                     = VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE,
    ePrimitiveRestartEnable              = VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE,
    eViewportWScalingNV                  = VK_DYNAMIC_STATE_VIEWPORT_W_SCALING_NV,
    eDiscardRectangleEXT                 = VK_DYNAMIC_STATE_DISCARD_RECTANGLE_EXT,
    eSampleLocationsEXT                  = VK_DYNAMIC_STATE_SAMPLE_LOCATIONS_EXT,
    eRayTracingPipelineStackSizeKHR      = VK_DYNAMIC_STATE_RAY_TRACING_PIPELINE_STACK_SIZE_KHR,
    eViewportShadingRatePaletteNV        = VK_DYNAMIC_STATE_VIEWPORT_SHADING_RATE_PALETTE_NV,
    eViewportCoarseSampleOrderNV         = VK_DYNAMIC_STATE_VIEWPORT_COARSE_SAMPLE_ORDER_NV,
    eExclusiveScissorNV                  = VK_DYNAMIC_STATE_EXCLUSIVE_SCISSOR_NV,
    eFragmentShadingRateKHR              = VK_DYNAMIC_STATE_FRAGMENT_SHADING_RATE_KHR,
    eLineStippleEXT                      = VK_DYNAMIC_STATE_LINE_STIPPLE_EXT,
    eVertexInputEXT                      = VK_DYNAMIC_STATE_VERTEX_INPUT_EXT,
    ePatchControlPointsEXT               = VK_DYNAMIC_STATE_PATCH_CONTROL_POINTS_EXT,
    eLogicOpEXT                          = VK_DYNAMIC_STATE_LOGIC_OP_EXT,
    eColorWriteEnableEXT                 = VK_DYNAMIC_STATE_COLOR_WRITE_ENABLE_EXT,
    eTessellationDomainOriginEXT         = VK_DYNAMIC_STATE_TESSELLATION_DOMAIN_ORIGIN_EXT,
    eDepthClampEnableEXT                 = VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT,
    ePolygonModeEXT                      = VK_DYNAMIC_STATE_POLYGON_MODE_EXT,
    eRasterizationSamplesEXT             = VK_DYNAMIC_STATE_RASTERIZATION_SAMPLES_EXT,
    eSampleMaskEXT                       = VK_DYNAMIC_STATE_SAMPLE_MASK_EXT,
    eAlphaToCoverageEnableEXT            = VK_DYNAMIC_STATE_ALPHA_TO_COVERAGE_ENABLE_EXT,
    eAlphaToOneEnableEXT                 = VK_DYNAMIC_STATE_ALPHA_TO_ONE_ENABLE_EXT,
    eLogicOpEnableEXT                    = VK_DYNAMIC_STATE_LOGIC_OP_ENABLE_EXT,
    eColorBlendEnableEXT                 = VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT,
    eColorBlendEquationEXT               = VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT,
    eColorWriteMaskEXT                   = VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT,
    eRasterizationStreamEXT              = VK_DYNAMIC_STATE_RASTERIZATION_STREAM_EXT,
    eConservativeRasterizationModeEXT    = VK_DYNAMIC_STATE_CONSERVATIVE_RASTERIZATION_MODE_EXT,
    eExtraPrimitiveOverestimationSizeEXT = VK_DYNAMIC_STATE_EXTRA_PRIMITIVE_OVERESTIMATION_SIZE_EXT,
    eDepthClipEnableEXT                  = VK_DYNAMIC_STATE_DEPTH_CLIP_ENABLE_EXT,
    eSampleLocationsEnableEXT            = VK_DYNAMIC_STATE_SAMPLE_LOCATIONS_ENABLE_EXT,
    eColorBlendAdvancedEXT               = VK_DYNAMIC_STATE_COLOR_BLEND_ADVANCED_EXT,
    eProvokingVertexModeEXT              = VK_DYNAMIC_STATE_PROVOKING_VERTEX_MODE_EXT,
    eLineRasterizationModeEXT            = VK_DYNAMIC_STATE_LINE_RASTERIZATION_MODE_EXT,
    eLineStippleEnableEXT                = VK_DYNAMIC_STATE_LINE_STIPPLE_ENABLE_EXT,
    eDepthClipNegativeOneToOneEXT        = VK_DYNAMIC_STATE_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE_EXT,
    eViewportWScalingEnableNV            = VK_DYNAMIC_STATE_VIEWPORT_W_SCALING_ENABLE_NV,
    eViewportSwizzleNV                   = VK_DYNAMIC_STATE_VIEWPORT_SWIZZLE_NV,
    eCoverageToColorEnableNV             = VK_DYNAMIC_STATE_COVERAGE_TO_COLOR_ENABLE_NV,
    eCoverageToColorLocationNV           = VK_DYNAMIC_STATE_COVERAGE_TO_COLOR_LOCATION_NV,
    eCoverageModulationModeNV            = VK_DYNAMIC_STATE_COVERAGE_MODULATION_MODE_NV,
    eCoverageModulationTableEnableNV     = VK_DYNAMIC_STATE_COVERAGE_MODULATION_TABLE_ENABLE_NV,
    eCoverageModulationTableNV           = VK_DYNAMIC_STATE_COVERAGE_MODULATION_TABLE_NV,
    eShadingRateImageEnableNV            = VK_DYNAMIC_STATE_SHADING_RATE_IMAGE_ENABLE_NV,
    eRepresentativeFragmentTestEnableNV  = VK_DYNAMIC_STATE_REPRESENTATIVE_FRAGMENT_TEST_ENABLE_NV,
    eCoverageReductionModeNV             = VK_DYNAMIC_STATE_COVERAGE_REDUCTION_MODE_NV,
    eCullModeEXT                         = VK_DYNAMIC_STATE_CULL_MODE_EXT,
    eDepthBiasEnableEXT                  = VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE_EXT,
    eDepthBoundsTestEnableEXT            = VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE_EXT,
    eDepthCompareOpEXT                   = VK_DYNAMIC_STATE_DEPTH_COMPARE_OP_EXT,
    eDepthTestEnableEXT                  = VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE_EXT,
    eDepthWriteEnableEXT                 = VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE_EXT,
    eFrontFaceEXT                        = VK_DYNAMIC_STATE_FRONT_FACE_EXT,
    ePrimitiveRestartEnableEXT           = VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE_EXT,
    ePrimitiveTopologyEXT                = VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY_EXT,
    eRasterizerDiscardEnableEXT          = VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE_EXT,
    eScissorWithCountEXT                 = VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT_EXT,
    eStencilOpEXT                        = VK_DYNAMIC_STATE_STENCIL_OP_EXT,
    eStencilTestEnableEXT                = VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE_EXT,
    eVertexInputBindingStrideEXT         = VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE_EXT,
    eViewportWithCountEXT                = VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT_EXT
  };

  enum class FrontFace
  {
    eCounterClockwise = VK_FRONT_FACE_COUNTER_CLOCKWISE,
    eClockwise        = VK_FRONT_FACE_CLOCKWISE
  };

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

  enum class PipelineCreateFlagBits : VkPipelineCreateFlags
  {
    eDisableOptimization                                                = VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT,
    eAllowDerivatives                                                   = VK_PIPELINE_CREATE_ALLOW_DERIVATIVES_BIT,
    eDerivative                                                         = VK_PIPELINE_CREATE_DERIVATIVE_BIT,
    eViewIndexFromDeviceIndex                                           = VK_PIPELINE_CREATE_VIEW_INDEX_FROM_DEVICE_INDEX_BIT,
    eDispatchBase                                                       = VK_PIPELINE_CREATE_DISPATCH_BASE_BIT,
    eFailOnPipelineCompileRequired                                      = VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT,
    eEarlyReturnOnFailure                                               = VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT,
    eRenderingFragmentShadingRateAttachmentKHR                          = VK_PIPELINE_CREATE_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
    eRenderingFragmentDensityMapAttachmentEXT                           = VK_PIPELINE_CREATE_RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT_BIT_EXT,
    eRayTracingNoNullAnyHitShadersKHR                                   = VK_PIPELINE_CREATE_RAY_TRACING_NO_NULL_ANY_HIT_SHADERS_BIT_KHR,
    eRayTracingNoNullClosestHitShadersKHR                               = VK_PIPELINE_CREATE_RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS_BIT_KHR,
    eRayTracingNoNullMissShadersKHR                                     = VK_PIPELINE_CREATE_RAY_TRACING_NO_NULL_MISS_SHADERS_BIT_KHR,
    eRayTracingNoNullIntersectionShadersKHR                             = VK_PIPELINE_CREATE_RAY_TRACING_NO_NULL_INTERSECTION_SHADERS_BIT_KHR,
    eRayTracingSkipTrianglesKHR                                         = VK_PIPELINE_CREATE_RAY_TRACING_SKIP_TRIANGLES_BIT_KHR,
    eRayTracingSkipAabbsKHR                                             = VK_PIPELINE_CREATE_RAY_TRACING_SKIP_AABBS_BIT_KHR,
    eRayTracingShaderGroupHandleCaptureReplayKHR                        = VK_PIPELINE_CREATE_RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY_BIT_KHR,
    eDeferCompileNV                                                     = VK_PIPELINE_CREATE_DEFER_COMPILE_BIT_NV,
    eCaptureStatisticsKHR                                               = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR,
    eCaptureInternalRepresentationsKHR                                  = VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR,
    eIndirectBindableNV                                                 = VK_PIPELINE_CREATE_INDIRECT_BINDABLE_BIT_NV,
    eLibraryKHR                                                         = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR,
    eRetainLinkTimeOptimizationInfoEXT                                  = VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT,
    eLinkTimeOptimizationEXT                                            = VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT,
    eRayTracingAllowMotionNV                                            = VK_PIPELINE_CREATE_RAY_TRACING_ALLOW_MOTION_BIT_NV,
    eColorAttachmentFeedbackLoopEXT                                     = VK_PIPELINE_CREATE_COLOR_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT,
    eDepthStencilAttachmentFeedbackLoopEXT                              = VK_PIPELINE_CREATE_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT,
    eRayTracingOpacityMicromapEXT                                       = VK_PIPELINE_CREATE_RAY_TRACING_OPACITY_MICROMAP_BIT_EXT,
    eNoProtectedAccessEXT                                               = VK_PIPELINE_CREATE_NO_PROTECTED_ACCESS_BIT_EXT,
    eProtectedAccessOnlyEXT                                             = VK_PIPELINE_CREATE_PROTECTED_ACCESS_ONLY_BIT_EXT,
    eDispatchBaseKHR                                                    = VK_PIPELINE_CREATE_DISPATCH_BASE_KHR,
    eEarlyReturnOnFailureEXT                                            = VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT_EXT,
    eFailOnPipelineCompileRequiredEXT                                   = VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_EXT,
    eViewIndexFromDeviceIndexKHR                                        = VK_PIPELINE_CREATE_VIEW_INDEX_FROM_DEVICE_INDEX_BIT_KHR,
    eVkPipelineRasterizationStateCreateFragmentDensityMapAttachmentEXT  = VK_PIPELINE_RASTERIZATION_STATE_CREATE_FRAGMENT_DENSITY_MAP_ATTACHMENT_BIT_EXT,
    eVkPipelineRasterizationStateCreateFragmentShadingRateAttachmentKHR = VK_PIPELINE_RASTERIZATION_STATE_CREATE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR
  };

  enum class PipelineShaderStageCreateFlagBits : VkPipelineShaderStageCreateFlags
  {
    eAllowVaryingSubgroupSize    = VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT,
    eRequireFullSubgroups        = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT,
    eAllowVaryingSubgroupSizeEXT = VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT_EXT,
    eRequireFullSubgroupsEXT     = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT
  };

  enum class PolygonMode
  {
    eFill            = VK_POLYGON_MODE_FILL,
    eLine            = VK_POLYGON_MODE_LINE,
    ePoint           = VK_POLYGON_MODE_POINT,
    eFillRectangleNV = VK_POLYGON_MODE_FILL_RECTANGLE_NV
  };

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
    eTaskEXT                = VK_SHADER_STAGE_TASK_BIT_EXT,
    eMeshEXT                = VK_SHADER_STAGE_MESH_BIT_EXT,
    eSubpassShadingHUAWEI   = VK_SHADER_STAGE_SUBPASS_SHADING_BIT_HUAWEI,
    eAnyHitNV               = VK_SHADER_STAGE_ANY_HIT_BIT_NV,
    eCallableNV             = VK_SHADER_STAGE_CALLABLE_BIT_NV,
    eClosestHitNV           = VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV,
    eIntersectionNV         = VK_SHADER_STAGE_INTERSECTION_BIT_NV,
    eMeshNV                 = VK_SHADER_STAGE_MESH_BIT_NV,
    eMissNV                 = VK_SHADER_STAGE_MISS_BIT_NV,
    eRaygenNV               = VK_SHADER_STAGE_RAYGEN_BIT_NV,
    eTaskNV                 = VK_SHADER_STAGE_TASK_BIT_NV
  };

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

  enum class VertexInputRate
  {
    eVertex   = VK_VERTEX_INPUT_RATE_VERTEX,
    eInstance = VK_VERTEX_INPUT_RATE_INSTANCE
  };

  enum class PipelineDynamicStateCreateFlagBits : VkPipelineDynamicStateCreateFlags
  {
  };

  enum class PipelineInputAssemblyStateCreateFlagBits : VkPipelineInputAssemblyStateCreateFlags
  {
  };

  enum class PipelineMultisampleStateCreateFlagBits : VkPipelineMultisampleStateCreateFlags
  {
  };

  enum class PipelineRasterizationStateCreateFlagBits : VkPipelineRasterizationStateCreateFlags
  {
  };

  enum class PipelineTessellationStateCreateFlagBits : VkPipelineTessellationStateCreateFlags
  {
  };

  enum class PipelineVertexInputStateCreateFlagBits : VkPipelineVertexInputStateCreateFlags
  {
  };

  enum class PipelineViewportStateCreateFlagBits : VkPipelineViewportStateCreateFlags
  {
  };

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

  enum class Filter
  {
    eNearest  = VK_FILTER_NEAREST,
    eLinear   = VK_FILTER_LINEAR,
    eCubicEXT = VK_FILTER_CUBIC_EXT,
    eCubicIMG = VK_FILTER_CUBIC_IMG
  };

  enum class SamplerAddressMode
  {
    eRepeat               = VK_SAMPLER_ADDRESS_MODE_REPEAT,
    eMirroredRepeat       = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
    eClampToEdge          = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    eClampToBorder        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
    eMirrorClampToEdge    = VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
    eMirrorClampToEdgeKHR = VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE_KHR
  };

  enum class SamplerCreateFlagBits : VkSamplerCreateFlags
  {
    eSubsampledEXT                     = VK_SAMPLER_CREATE_SUBSAMPLED_BIT_EXT,
    eSubsampledCoarseReconstructionEXT = VK_SAMPLER_CREATE_SUBSAMPLED_COARSE_RECONSTRUCTION_BIT_EXT,
    eNonSeamlessCubeMapEXT             = VK_SAMPLER_CREATE_NON_SEAMLESS_CUBE_MAP_BIT_EXT,
    eImageProcessingQCOM               = VK_SAMPLER_CREATE_IMAGE_PROCESSING_BIT_QCOM
  };

  enum class SamplerMipmapMode
  {
    eNearest = VK_SAMPLER_MIPMAP_MODE_NEAREST,
    eLinear  = VK_SAMPLER_MIPMAP_MODE_LINEAR
  };

  enum class DescriptorPoolCreateFlagBits : VkDescriptorPoolCreateFlags
  {
    eFreeDescriptorSet  = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
    eUpdateAfterBind    = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
    eHostOnlyEXT        = VK_DESCRIPTOR_POOL_CREATE_HOST_ONLY_BIT_EXT,
    eHostOnlyVALVE      = VK_DESCRIPTOR_POOL_CREATE_HOST_ONLY_BIT_VALVE,
    eUpdateAfterBindEXT = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT_EXT
  };

  enum class DescriptorSetLayoutCreateFlagBits : VkDescriptorSetLayoutCreateFlags
  {
    eUpdateAfterBindPool    = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
    ePushDescriptorKHR      = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
    eHostOnlyPoolEXT        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_HOST_ONLY_POOL_BIT_EXT,
    eHostOnlyPoolVALVE      = VK_DESCRIPTOR_SET_LAYOUT_CREATE_HOST_ONLY_POOL_BIT_VALVE,
    eUpdateAfterBindPoolEXT = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT_EXT
  };

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
    eInlineUniformBlock       = VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK,
    eAccelerationStructureKHR = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
    eAccelerationStructureNV  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV,
    eSampleWeightImageQCOM    = VK_DESCRIPTOR_TYPE_SAMPLE_WEIGHT_IMAGE_QCOM,
    eBlockMatchImageQCOM      = VK_DESCRIPTOR_TYPE_BLOCK_MATCH_IMAGE_QCOM,
    eMutableEXT               = VK_DESCRIPTOR_TYPE_MUTABLE_EXT,
    eInlineUniformBlockEXT    = VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT,
    eMutableVALVE             = VK_DESCRIPTOR_TYPE_MUTABLE_VALVE
  };

  enum class DescriptorPoolResetFlagBits : VkDescriptorPoolResetFlags
  {
  };

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
    eNone                                 = VK_ACCESS_NONE,
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
    eAccelerationStructureReadNV          = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV,
    eAccelerationStructureWriteNV         = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV,
    eNoneKHR                              = VK_ACCESS_NONE_KHR,
    eShadingRateImageReadNV               = VK_ACCESS_SHADING_RATE_IMAGE_READ_BIT_NV
  };

  enum class AttachmentDescriptionFlagBits : VkAttachmentDescriptionFlags
  {
    eMayAlias = VK_ATTACHMENT_DESCRIPTION_MAY_ALIAS_BIT
  };

  enum class AttachmentLoadOp
  {
    eLoad     = VK_ATTACHMENT_LOAD_OP_LOAD,
    eClear    = VK_ATTACHMENT_LOAD_OP_CLEAR,
    eDontCare = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
    eNoneEXT  = VK_ATTACHMENT_LOAD_OP_NONE_EXT
  };

  enum class AttachmentStoreOp
  {
    eStore    = VK_ATTACHMENT_STORE_OP_STORE,
    eDontCare = VK_ATTACHMENT_STORE_OP_DONT_CARE,
    eNone     = VK_ATTACHMENT_STORE_OP_NONE,
    eNoneEXT  = VK_ATTACHMENT_STORE_OP_NONE_EXT,
    eNoneKHR  = VK_ATTACHMENT_STORE_OP_NONE_KHR,
    eNoneQCOM = VK_ATTACHMENT_STORE_OP_NONE_QCOM
  };

  enum class DependencyFlagBits : VkDependencyFlags
  {
    eByRegion        = VK_DEPENDENCY_BY_REGION_BIT,
    eDeviceGroup     = VK_DEPENDENCY_DEVICE_GROUP_BIT,
    eViewLocal       = VK_DEPENDENCY_VIEW_LOCAL_BIT,
    eFeedbackLoopEXT = VK_DEPENDENCY_FEEDBACK_LOOP_BIT_EXT,
    eDeviceGroupKHR  = VK_DEPENDENCY_DEVICE_GROUP_BIT_KHR,
    eViewLocalKHR    = VK_DEPENDENCY_VIEW_LOCAL_BIT_KHR
  };

  enum class FramebufferCreateFlagBits : VkFramebufferCreateFlags
  {
    eImageless    = VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT,
    eImagelessKHR = VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT_KHR
  };

  enum class PipelineBindPoint
  {
    eGraphics             = VK_PIPELINE_BIND_POINT_GRAPHICS,
    eCompute              = VK_PIPELINE_BIND_POINT_COMPUTE,
    eRayTracingKHR        = VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
    eSubpassShadingHUAWEI = VK_PIPELINE_BIND_POINT_SUBPASS_SHADING_HUAWEI,
    eRayTracingNV         = VK_PIPELINE_BIND_POINT_RAY_TRACING_NV
  };

  enum class RenderPassCreateFlagBits : VkRenderPassCreateFlags
  {
    eTransformQCOM = VK_RENDER_PASS_CREATE_TRANSFORM_BIT_QCOM
  };

  enum class SubpassDescriptionFlagBits : VkSubpassDescriptionFlags
  {
    ePerViewAttributesNVX                         = VK_SUBPASS_DESCRIPTION_PER_VIEW_ATTRIBUTES_BIT_NVX,
    ePerViewPositionXOnlyNVX                      = VK_SUBPASS_DESCRIPTION_PER_VIEW_POSITION_X_ONLY_BIT_NVX,
    eFragmentRegionQCOM                           = VK_SUBPASS_DESCRIPTION_FRAGMENT_REGION_BIT_QCOM,
    eShaderResolveQCOM                            = VK_SUBPASS_DESCRIPTION_SHADER_RESOLVE_BIT_QCOM,
    eRasterizationOrderAttachmentColorAccessEXT   = VK_SUBPASS_DESCRIPTION_RASTERIZATION_ORDER_ATTACHMENT_COLOR_ACCESS_BIT_EXT,
    eRasterizationOrderAttachmentDepthAccessEXT   = VK_SUBPASS_DESCRIPTION_RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_BIT_EXT,
    eRasterizationOrderAttachmentStencilAccessEXT = VK_SUBPASS_DESCRIPTION_RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_BIT_EXT,
    eEnableLegacyDitheringEXT                     = VK_SUBPASS_DESCRIPTION_ENABLE_LEGACY_DITHERING_BIT_EXT,
    eRasterizationOrderAttachmentColorAccessARM   = VK_SUBPASS_DESCRIPTION_RASTERIZATION_ORDER_ATTACHMENT_COLOR_ACCESS_BIT_ARM,
    eRasterizationOrderAttachmentDepthAccessARM   = VK_SUBPASS_DESCRIPTION_RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_BIT_ARM,
    eRasterizationOrderAttachmentStencilAccessARM = VK_SUBPASS_DESCRIPTION_RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_BIT_ARM
  };

  enum class CommandPoolCreateFlagBits : VkCommandPoolCreateFlags
  {
    eTransient          = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
    eResetCommandBuffer = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    eProtected          = VK_COMMAND_POOL_CREATE_PROTECTED_BIT
  };

  enum class CommandPoolResetFlagBits : VkCommandPoolResetFlags
  {
    eReleaseResources = VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT
  };

  enum class CommandBufferLevel
  {
    ePrimary   = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    eSecondary = VK_COMMAND_BUFFER_LEVEL_SECONDARY
  };

  enum class CommandBufferResetFlagBits : VkCommandBufferResetFlags
  {
    eReleaseResources = VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT
  };

  enum class CommandBufferUsageFlagBits : VkCommandBufferUsageFlags
  {
    eOneTimeSubmit      = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    eRenderPassContinue = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
    eSimultaneousUse    = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT
  };

  enum class QueryControlFlagBits : VkQueryControlFlags
  {
    ePrecise = VK_QUERY_CONTROL_PRECISE_BIT
  };

  enum class IndexType
  {
    eUint16   = VK_INDEX_TYPE_UINT16,
    eUint32   = VK_INDEX_TYPE_UINT32,
    eNoneKHR  = VK_INDEX_TYPE_NONE_KHR,
    eUint8EXT = VK_INDEX_TYPE_UINT8_EXT,
    eNoneNV   = VK_INDEX_TYPE_NONE_NV
  };

  enum class StencilFaceFlagBits : VkStencilFaceFlags
  {
    eFront                 = VK_STENCIL_FACE_FRONT_BIT,
    eBack                  = VK_STENCIL_FACE_BACK_BIT,
    eFrontAndBack          = VK_STENCIL_FACE_FRONT_AND_BACK,
    eVkStencilFrontAndBack = VK_STENCIL_FRONT_AND_BACK
  };

  enum class SubpassContents
  {
    eInline                  = VK_SUBPASS_CONTENTS_INLINE,
    eSecondaryCommandBuffers = VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS
  };

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

  enum class PeerMemoryFeatureFlagBits : VkPeerMemoryFeatureFlags
  {
    eCopySrc    = VK_PEER_MEMORY_FEATURE_COPY_SRC_BIT,
    eCopyDst    = VK_PEER_MEMORY_FEATURE_COPY_DST_BIT,
    eGenericSrc = VK_PEER_MEMORY_FEATURE_GENERIC_SRC_BIT,
    eGenericDst = VK_PEER_MEMORY_FEATURE_GENERIC_DST_BIT
  };
  using PeerMemoryFeatureFlagBitsKHR = PeerMemoryFeatureFlagBits;

  enum class MemoryAllocateFlagBits : VkMemoryAllocateFlags
  {
    eDeviceMask                 = VK_MEMORY_ALLOCATE_DEVICE_MASK_BIT,
    eDeviceAddress              = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    eDeviceAddressCaptureReplay = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT
  };
  using MemoryAllocateFlagBitsKHR = MemoryAllocateFlagBits;

  enum class CommandPoolTrimFlagBits : VkCommandPoolTrimFlags
  {
  };

  enum class PointClippingBehavior
  {
    eAllClipPlanes      = VK_POINT_CLIPPING_BEHAVIOR_ALL_CLIP_PLANES,
    eUserClipPlanesOnly = VK_POINT_CLIPPING_BEHAVIOR_USER_CLIP_PLANES_ONLY
  };
  using PointClippingBehaviorKHR = PointClippingBehavior;

  enum class TessellationDomainOrigin
  {
    eUpperLeft = VK_TESSELLATION_DOMAIN_ORIGIN_UPPER_LEFT,
    eLowerLeft = VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT
  };
  using TessellationDomainOriginKHR = TessellationDomainOrigin;

  enum class DeviceQueueCreateFlagBits : VkDeviceQueueCreateFlags
  {
    eProtected = VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT
  };

  enum class SamplerYcbcrModelConversion
  {
    eRgbIdentity   = VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY,
    eYcbcrIdentity = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_IDENTITY,
    eYcbcr709      = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709,
    eYcbcr601      = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601,
    eYcbcr2020     = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_2020
  };
  using SamplerYcbcrModelConversionKHR = SamplerYcbcrModelConversion;

  enum class SamplerYcbcrRange
  {
    eItuFull   = VK_SAMPLER_YCBCR_RANGE_ITU_FULL,
    eItuNarrow = VK_SAMPLER_YCBCR_RANGE_ITU_NARROW
  };
  using SamplerYcbcrRangeKHR = SamplerYcbcrRange;

  enum class ChromaLocation
  {
    eCositedEven = VK_CHROMA_LOCATION_COSITED_EVEN,
    eMidpoint    = VK_CHROMA_LOCATION_MIDPOINT
  };
  using ChromaLocationKHR = ChromaLocation;

  enum class DescriptorUpdateTemplateType
  {
    eDescriptorSet      = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET,
    ePushDescriptorsKHR = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR
  };
  using DescriptorUpdateTemplateTypeKHR = DescriptorUpdateTemplateType;

  enum class DescriptorUpdateTemplateCreateFlagBits : VkDescriptorUpdateTemplateCreateFlags
  {
  };

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
    eZirconVmoFUCHSIA = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ZIRCON_VMO_BIT_FUCHSIA,
#endif /*VK_USE_PLATFORM_FUCHSIA*/
    eRdmaAddressNV = VK_EXTERNAL_MEMORY_HANDLE_TYPE_RDMA_ADDRESS_BIT_NV
  };
  using ExternalMemoryHandleTypeFlagBitsKHR = ExternalMemoryHandleTypeFlagBits;

  enum class ExternalMemoryFeatureFlagBits : VkExternalMemoryFeatureFlags
  {
    eDedicatedOnly = VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT,
    eExportable    = VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT,
    eImportable    = VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT
  };
  using ExternalMemoryFeatureFlagBitsKHR = ExternalMemoryFeatureFlagBits;

  enum class ExternalFenceHandleTypeFlagBits : VkExternalFenceHandleTypeFlags
  {
    eOpaqueFd       = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT,
    eOpaqueWin32    = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_WIN32_BIT,
    eOpaqueWin32Kmt = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT,
    eSyncFd         = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT
  };
  using ExternalFenceHandleTypeFlagBitsKHR = ExternalFenceHandleTypeFlagBits;

  enum class ExternalFenceFeatureFlagBits : VkExternalFenceFeatureFlags
  {
    eExportable = VK_EXTERNAL_FENCE_FEATURE_EXPORTABLE_BIT,
    eImportable = VK_EXTERNAL_FENCE_FEATURE_IMPORTABLE_BIT
  };
  using ExternalFenceFeatureFlagBitsKHR = ExternalFenceFeatureFlagBits;

  enum class FenceImportFlagBits : VkFenceImportFlags
  {
    eTemporary = VK_FENCE_IMPORT_TEMPORARY_BIT
  };
  using FenceImportFlagBitsKHR = FenceImportFlagBits;

  enum class SemaphoreImportFlagBits : VkSemaphoreImportFlags
  {
    eTemporary = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT
  };
  using SemaphoreImportFlagBitsKHR = SemaphoreImportFlagBits;

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

  enum class ExternalSemaphoreFeatureFlagBits : VkExternalSemaphoreFeatureFlags
  {
    eExportable = VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT,
    eImportable = VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT
  };
  using ExternalSemaphoreFeatureFlagBitsKHR = ExternalSemaphoreFeatureFlagBits;

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
    eJuiceProprietary        = VK_DRIVER_ID_JUICE_PROPRIETARY,
    eVerisiliconProprietary  = VK_DRIVER_ID_VERISILICON_PROPRIETARY,
    eMesaTurnip              = VK_DRIVER_ID_MESA_TURNIP,
    eMesaV3Dv                = VK_DRIVER_ID_MESA_V3DV,
    eMesaPanvk               = VK_DRIVER_ID_MESA_PANVK,
    eSamsungProprietary      = VK_DRIVER_ID_SAMSUNG_PROPRIETARY,
    eMesaVenus               = VK_DRIVER_ID_MESA_VENUS,
    eMesaDozen               = VK_DRIVER_ID_MESA_DOZEN
  };
  using DriverIdKHR = DriverId;

  enum class ShaderFloatControlsIndependence
  {
    e32BitOnly = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY,
    eAll       = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
    eNone      = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE
  };
  using ShaderFloatControlsIndependenceKHR = ShaderFloatControlsIndependence;

  enum class DescriptorBindingFlagBits : VkDescriptorBindingFlags
  {
    eUpdateAfterBind          = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
    eUpdateUnusedWhilePending = VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT,
    ePartiallyBound           = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
    eVariableDescriptorCount  = VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT
  };
  using DescriptorBindingFlagBitsEXT = DescriptorBindingFlagBits;

  enum class ResolveModeFlagBits : VkResolveModeFlags
  {
    eNone       = VK_RESOLVE_MODE_NONE,
    eSampleZero = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT,
    eAverage    = VK_RESOLVE_MODE_AVERAGE_BIT,
    eMin        = VK_RESOLVE_MODE_MIN_BIT,
    eMax        = VK_RESOLVE_MODE_MAX_BIT
  };
  using ResolveModeFlagBitsKHR = ResolveModeFlagBits;

  enum class SamplerReductionMode
  {
    eWeightedAverage = VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE,
    eMin             = VK_SAMPLER_REDUCTION_MODE_MIN,
    eMax             = VK_SAMPLER_REDUCTION_MODE_MAX
  };
  using SamplerReductionModeEXT = SamplerReductionMode;

  enum class SemaphoreType
  {
    eBinary   = VK_SEMAPHORE_TYPE_BINARY,
    eTimeline = VK_SEMAPHORE_TYPE_TIMELINE
  };
  using SemaphoreTypeKHR = SemaphoreType;

  enum class SemaphoreWaitFlagBits : VkSemaphoreWaitFlags
  {
    eAny = VK_SEMAPHORE_WAIT_ANY_BIT
  };
  using SemaphoreWaitFlagBitsKHR = SemaphoreWaitFlagBits;

  //=== VK_VERSION_1_3 ===

  enum class PipelineCreationFeedbackFlagBits : VkPipelineCreationFeedbackFlags
  {
    eValid                       = VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT,
    eApplicationPipelineCacheHit = VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT,
    eBasePipelineAcceleration    = VK_PIPELINE_CREATION_FEEDBACK_BASE_PIPELINE_ACCELERATION_BIT
  };
  using PipelineCreationFeedbackFlagBitsEXT = PipelineCreationFeedbackFlagBits;

  enum class ToolPurposeFlagBits : VkToolPurposeFlags
  {
    eValidation         = VK_TOOL_PURPOSE_VALIDATION_BIT,
    eProfiling          = VK_TOOL_PURPOSE_PROFILING_BIT,
    eTracing            = VK_TOOL_PURPOSE_TRACING_BIT,
    eAdditionalFeatures = VK_TOOL_PURPOSE_ADDITIONAL_FEATURES_BIT,
    eModifyingFeatures  = VK_TOOL_PURPOSE_MODIFYING_FEATURES_BIT,
    eDebugReportingEXT  = VK_TOOL_PURPOSE_DEBUG_REPORTING_BIT_EXT,
    eDebugMarkersEXT    = VK_TOOL_PURPOSE_DEBUG_MARKERS_BIT_EXT
  };
  using ToolPurposeFlagBitsEXT = ToolPurposeFlagBits;

  enum class PrivateDataSlotCreateFlagBits : VkPrivateDataSlotCreateFlags
  {
  };
  using PrivateDataSlotCreateFlagBitsEXT = PrivateDataSlotCreateFlagBits;

  enum class PipelineStageFlagBits2 : VkPipelineStageFlags2
  {
    eNone                         = VK_PIPELINE_STAGE_2_NONE,
    eTopOfPipe                    = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
    eDrawIndirect                 = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
    eVertexInput                  = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT,
    eVertexShader                 = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
    eTessellationControlShader    = VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT,
    eTessellationEvaluationShader = VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT,
    eGeometryShader               = VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT,
    eFragmentShader               = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
    eEarlyFragmentTests           = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
    eLateFragmentTests            = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
    eColorAttachmentOutput        = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
    eComputeShader                = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    eAllTransfer                  = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
    eBottomOfPipe                 = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
    eHost                         = VK_PIPELINE_STAGE_2_HOST_BIT,
    eAllGraphics                  = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
    eAllCommands                  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
    eCopy                         = VK_PIPELINE_STAGE_2_COPY_BIT,
    eResolve                      = VK_PIPELINE_STAGE_2_RESOLVE_BIT,
    eBlit                         = VK_PIPELINE_STAGE_2_BLIT_BIT,
    eClear                        = VK_PIPELINE_STAGE_2_CLEAR_BIT,
    eIndexInput                   = VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,
    eVertexAttributeInput         = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT,
    ePreRasterizationShaders      = VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoDecodeKHR = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR,
    eVideoEncodeKHR = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eTransformFeedbackEXT             = VK_PIPELINE_STAGE_2_TRANSFORM_FEEDBACK_BIT_EXT,
    eConditionalRenderingEXT          = VK_PIPELINE_STAGE_2_CONDITIONAL_RENDERING_BIT_EXT,
    eCommandPreprocessNV              = VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_NV,
    eFragmentShadingRateAttachmentKHR = VK_PIPELINE_STAGE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
    eAccelerationStructureBuildKHR    = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    eRayTracingShaderKHR              = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
    eFragmentDensityProcessEXT        = VK_PIPELINE_STAGE_2_FRAGMENT_DENSITY_PROCESS_BIT_EXT,
    eTaskShaderEXT                    = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
    eMeshShaderEXT                    = VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
    eSubpassShadingHUAWEI             = VK_PIPELINE_STAGE_2_SUBPASS_SHADING_BIT_HUAWEI,
    eInvocationMaskHUAWEI             = VK_PIPELINE_STAGE_2_INVOCATION_MASK_BIT_HUAWEI,
    eAccelerationStructureCopyKHR     = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR,
    eMicromapBuildEXT                 = VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT,
    eOpticalFlowNV                    = VK_PIPELINE_STAGE_2_OPTICAL_FLOW_BIT_NV,
    eAccelerationStructureBuildNV     = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
    eMeshShaderNV                     = VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_NV,
    eRayTracingShaderNV               = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_NV,
    eShadingRateImageNV               = VK_PIPELINE_STAGE_2_SHADING_RATE_IMAGE_BIT_NV,
    eTaskShaderNV                     = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_NV,
    eTransfer                         = VK_PIPELINE_STAGE_2_TRANSFER_BIT
  };
  using PipelineStageFlagBits2KHR = PipelineStageFlagBits2;

  enum class AccessFlagBits2 : VkAccessFlags2
  {
    eNone                        = VK_ACCESS_2_NONE,
    eIndirectCommandRead         = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
    eIndexRead                   = VK_ACCESS_2_INDEX_READ_BIT,
    eVertexAttributeRead         = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
    eUniformRead                 = VK_ACCESS_2_UNIFORM_READ_BIT,
    eInputAttachmentRead         = VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT,
    eShaderRead                  = VK_ACCESS_2_SHADER_READ_BIT,
    eShaderWrite                 = VK_ACCESS_2_SHADER_WRITE_BIT,
    eColorAttachmentRead         = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
    eColorAttachmentWrite        = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
    eDepthStencilAttachmentRead  = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
    eDepthStencilAttachmentWrite = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
    eTransferRead                = VK_ACCESS_2_TRANSFER_READ_BIT,
    eTransferWrite               = VK_ACCESS_2_TRANSFER_WRITE_BIT,
    eHostRead                    = VK_ACCESS_2_HOST_READ_BIT,
    eHostWrite                   = VK_ACCESS_2_HOST_WRITE_BIT,
    eMemoryRead                  = VK_ACCESS_2_MEMORY_READ_BIT,
    eMemoryWrite                 = VK_ACCESS_2_MEMORY_WRITE_BIT,
    eShaderSampledRead           = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
    eShaderStorageRead           = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
    eShaderStorageWrite          = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoDecodeReadKHR  = VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR,
    eVideoDecodeWriteKHR = VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR,
    eVideoEncodeReadKHR  = VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR,
    eVideoEncodeWriteKHR = VK_ACCESS_2_VIDEO_ENCODE_WRITE_BIT_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eTransformFeedbackWriteEXT            = VK_ACCESS_2_TRANSFORM_FEEDBACK_WRITE_BIT_EXT,
    eTransformFeedbackCounterReadEXT      = VK_ACCESS_2_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT,
    eTransformFeedbackCounterWriteEXT     = VK_ACCESS_2_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT,
    eConditionalRenderingReadEXT          = VK_ACCESS_2_CONDITIONAL_RENDERING_READ_BIT_EXT,
    eCommandPreprocessReadNV              = VK_ACCESS_2_COMMAND_PREPROCESS_READ_BIT_NV,
    eCommandPreprocessWriteNV             = VK_ACCESS_2_COMMAND_PREPROCESS_WRITE_BIT_NV,
    eFragmentShadingRateAttachmentReadKHR = VK_ACCESS_2_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR,
    eAccelerationStructureReadKHR         = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    eAccelerationStructureWriteKHR        = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
    eFragmentDensityMapReadEXT            = VK_ACCESS_2_FRAGMENT_DENSITY_MAP_READ_BIT_EXT,
    eColorAttachmentReadNoncoherentEXT    = VK_ACCESS_2_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT,
    eInvocationMaskReadHUAWEI             = VK_ACCESS_2_INVOCATION_MASK_READ_BIT_HUAWEI,
    eShaderBindingTableReadKHR            = VK_ACCESS_2_SHADER_BINDING_TABLE_READ_BIT_KHR,
    eMicromapReadEXT                      = VK_ACCESS_2_MICROMAP_READ_BIT_EXT,
    eMicromapWriteEXT                     = VK_ACCESS_2_MICROMAP_WRITE_BIT_EXT,
    eOpticalFlowReadNV                    = VK_ACCESS_2_OPTICAL_FLOW_READ_BIT_NV,
    eOpticalFlowWriteNV                   = VK_ACCESS_2_OPTICAL_FLOW_WRITE_BIT_NV,
    eAccelerationStructureReadNV          = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_NV,
    eAccelerationStructureWriteNV         = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_NV,
    eShadingRateImageReadNV               = VK_ACCESS_2_SHADING_RATE_IMAGE_READ_BIT_NV
  };
  using AccessFlagBits2KHR = AccessFlagBits2;

  enum class SubmitFlagBits : VkSubmitFlags
  {
    eProtected = VK_SUBMIT_PROTECTED_BIT
  };
  using SubmitFlagBitsKHR = SubmitFlagBits;

  enum class RenderingFlagBits : VkRenderingFlags
  {
    eContentsSecondaryCommandBuffers = VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT,
    eSuspending                      = VK_RENDERING_SUSPENDING_BIT,
    eResuming                        = VK_RENDERING_RESUMING_BIT,
    eEnableLegacyDitheringEXT        = VK_RENDERING_ENABLE_LEGACY_DITHERING_BIT_EXT
  };
  using RenderingFlagBitsKHR = RenderingFlagBits;

  enum class FormatFeatureFlagBits2 : VkFormatFeatureFlags2
  {
    eSampledImage                                            = VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT,
    eStorageImage                                            = VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT,
    eStorageImageAtomic                                      = VK_FORMAT_FEATURE_2_STORAGE_IMAGE_ATOMIC_BIT,
    eUniformTexelBuffer                                      = VK_FORMAT_FEATURE_2_UNIFORM_TEXEL_BUFFER_BIT,
    eStorageTexelBuffer                                      = VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT,
    eStorageTexelBufferAtomic                                = VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_ATOMIC_BIT,
    eVertexBuffer                                            = VK_FORMAT_FEATURE_2_VERTEX_BUFFER_BIT,
    eColorAttachment                                         = VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT,
    eColorAttachmentBlend                                    = VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BLEND_BIT,
    eDepthStencilAttachment                                  = VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT,
    eBlitSrc                                                 = VK_FORMAT_FEATURE_2_BLIT_SRC_BIT,
    eBlitDst                                                 = VK_FORMAT_FEATURE_2_BLIT_DST_BIT,
    eSampledImageFilterLinear                                = VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT,
    eSampledImageFilterCubic                                 = VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_CUBIC_BIT,
    eTransferSrc                                             = VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT,
    eTransferDst                                             = VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT,
    eSampledImageFilterMinmax                                = VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_MINMAX_BIT,
    eMidpointChromaSamples                                   = VK_FORMAT_FEATURE_2_MIDPOINT_CHROMA_SAMPLES_BIT,
    eSampledImageYcbcrConversionLinearFilter                 = VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT,
    eSampledImageYcbcrConversionSeparateReconstructionFilter = VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT,
    eSampledImageYcbcrConversionChromaReconstructionExplicit = VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_BIT,
    eSampledImageYcbcrConversionChromaReconstructionExplicitForceable =
      VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE_BIT,
    eDisjoint                    = VK_FORMAT_FEATURE_2_DISJOINT_BIT,
    eCositedChromaSamples        = VK_FORMAT_FEATURE_2_COSITED_CHROMA_SAMPLES_BIT,
    eStorageReadWithoutFormat    = VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT,
    eStorageWriteWithoutFormat   = VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT,
    eSampledImageDepthComparison = VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoDecodeOutputKHR = VK_FORMAT_FEATURE_2_VIDEO_DECODE_OUTPUT_BIT_KHR,
    eVideoDecodeDpbKHR    = VK_FORMAT_FEATURE_2_VIDEO_DECODE_DPB_BIT_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eAccelerationStructureVertexBufferKHR = VK_FORMAT_FEATURE_2_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR,
    eFragmentDensityMapEXT                = VK_FORMAT_FEATURE_2_FRAGMENT_DENSITY_MAP_BIT_EXT,
    eFragmentShadingRateAttachmentKHR     = VK_FORMAT_FEATURE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoEncodeInputKHR = VK_FORMAT_FEATURE_2_VIDEO_ENCODE_INPUT_BIT_KHR,
    eVideoEncodeDpbKHR   = VK_FORMAT_FEATURE_2_VIDEO_ENCODE_DPB_BIT_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eLinearColorAttachmentNV    = VK_FORMAT_FEATURE_2_LINEAR_COLOR_ATTACHMENT_BIT_NV,
    eWeightImageQCOM            = VK_FORMAT_FEATURE_2_WEIGHT_IMAGE_BIT_QCOM,
    eWeightSampledImageQCOM     = VK_FORMAT_FEATURE_2_WEIGHT_SAMPLED_IMAGE_BIT_QCOM,
    eBlockMatchingQCOM          = VK_FORMAT_FEATURE_2_BLOCK_MATCHING_BIT_QCOM,
    eBoxFilterSampledQCOM       = VK_FORMAT_FEATURE_2_BOX_FILTER_SAMPLED_BIT_QCOM,
    eOpticalFlowImageNV         = VK_FORMAT_FEATURE_2_OPTICAL_FLOW_IMAGE_BIT_NV,
    eOpticalFlowVectorNV        = VK_FORMAT_FEATURE_2_OPTICAL_FLOW_VECTOR_BIT_NV,
    eOpticalFlowCostNV          = VK_FORMAT_FEATURE_2_OPTICAL_FLOW_COST_BIT_NV,
    eSampledImageFilterCubicEXT = VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_CUBIC_BIT_EXT
  };
  using FormatFeatureFlagBits2KHR = FormatFeatureFlagBits2;

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

  enum class PresentModeKHR
  {
    eImmediate               = VK_PRESENT_MODE_IMMEDIATE_KHR,
    eMailbox                 = VK_PRESENT_MODE_MAILBOX_KHR,
    eFifo                    = VK_PRESENT_MODE_FIFO_KHR,
    eFifoRelaxed             = VK_PRESENT_MODE_FIFO_RELAXED_KHR,
    eSharedDemandRefresh     = VK_PRESENT_MODE_SHARED_DEMAND_REFRESH_KHR,
    eSharedContinuousRefresh = VK_PRESENT_MODE_SHARED_CONTINUOUS_REFRESH_KHR
  };

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

  enum class CompositeAlphaFlagBitsKHR : VkCompositeAlphaFlagsKHR
  {
    eOpaque         = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
    ePreMultiplied  = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
    ePostMultiplied = VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
    eInherit        = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR
  };

  //=== VK_KHR_swapchain ===

  enum class SwapchainCreateFlagBitsKHR : VkSwapchainCreateFlagsKHR
  {
    eSplitInstanceBindRegions = VK_SWAPCHAIN_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT_KHR,
    eProtected                = VK_SWAPCHAIN_CREATE_PROTECTED_BIT_KHR,
    eMutableFormat            = VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR
  };

  enum class DeviceGroupPresentModeFlagBitsKHR : VkDeviceGroupPresentModeFlagsKHR
  {
    eLocal            = VK_DEVICE_GROUP_PRESENT_MODE_LOCAL_BIT_KHR,
    eRemote           = VK_DEVICE_GROUP_PRESENT_MODE_REMOTE_BIT_KHR,
    eSum              = VK_DEVICE_GROUP_PRESENT_MODE_SUM_BIT_KHR,
    eLocalMultiDevice = VK_DEVICE_GROUP_PRESENT_MODE_LOCAL_MULTI_DEVICE_BIT_KHR
  };

  //=== VK_KHR_display ===

  enum class DisplayPlaneAlphaFlagBitsKHR : VkDisplayPlaneAlphaFlagsKHR
  {
    eOpaque                = VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR,
    eGlobal                = VK_DISPLAY_PLANE_ALPHA_GLOBAL_BIT_KHR,
    ePerPixel              = VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_BIT_KHR,
    ePerPixelPremultiplied = VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_PREMULTIPLIED_BIT_KHR
  };

  enum class DisplayModeCreateFlagBitsKHR : VkDisplayModeCreateFlagsKHR
  {
  };

  enum class DisplaySurfaceCreateFlagBitsKHR : VkDisplaySurfaceCreateFlagsKHR
  {
  };

#if defined( VK_USE_PLATFORM_XLIB_KHR )
  //=== VK_KHR_xlib_surface ===

  enum class XlibSurfaceCreateFlagBitsKHR : VkXlibSurfaceCreateFlagsKHR
  {
  };
#endif /*VK_USE_PLATFORM_XLIB_KHR*/

#if defined( VK_USE_PLATFORM_XCB_KHR )
  //=== VK_KHR_xcb_surface ===

  enum class XcbSurfaceCreateFlagBitsKHR : VkXcbSurfaceCreateFlagsKHR
  {
  };
#endif /*VK_USE_PLATFORM_XCB_KHR*/

#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
  //=== VK_KHR_wayland_surface ===

  enum class WaylandSurfaceCreateFlagBitsKHR : VkWaylandSurfaceCreateFlagsKHR
  {
  };
#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  //=== VK_KHR_android_surface ===

  enum class AndroidSurfaceCreateFlagBitsKHR : VkAndroidSurfaceCreateFlagsKHR
  {
  };
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_win32_surface ===

  enum class Win32SurfaceCreateFlagBitsKHR : VkWin32SurfaceCreateFlagsKHR
  {
  };
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

  enum class DebugReportObjectTypeEXT
  {
    eUnknown                  = VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT,
    eInstance                 = VK_DEBUG_REPORT_OBJECT_TYPE_INSTANCE_EXT,
    ePhysicalDevice           = VK_DEBUG_REPORT_OBJECT_TYPE_PHYSICAL_DEVICE_EXT,
    eDevice                   = VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_EXT,
    eQueue                    = VK_DEBUG_REPORT_OBJECT_TYPE_QUEUE_EXT,
    eSemaphore                = VK_DEBUG_REPORT_OBJECT_TYPE_SEMAPHORE_EXT,
    eCommandBuffer            = VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
    eFence                    = VK_DEBUG_REPORT_OBJECT_TYPE_FENCE_EXT,
    eDeviceMemory             = VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_MEMORY_EXT,
    eBuffer                   = VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT,
    eImage                    = VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,
    eEvent                    = VK_DEBUG_REPORT_OBJECT_TYPE_EVENT_EXT,
    eQueryPool                = VK_DEBUG_REPORT_OBJECT_TYPE_QUERY_POOL_EXT,
    eBufferView               = VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_VIEW_EXT,
    eImageView                = VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_VIEW_EXT,
    eShaderModule             = VK_DEBUG_REPORT_OBJECT_TYPE_SHADER_MODULE_EXT,
    ePipelineCache            = VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_CACHE_EXT,
    ePipelineLayout           = VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_LAYOUT_EXT,
    eRenderPass               = VK_DEBUG_REPORT_OBJECT_TYPE_RENDER_PASS_EXT,
    ePipeline                 = VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_EXT,
    eDescriptorSetLayout      = VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT_EXT,
    eSampler                  = VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_EXT,
    eDescriptorPool           = VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_POOL_EXT,
    eDescriptorSet            = VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_EXT,
    eFramebuffer              = VK_DEBUG_REPORT_OBJECT_TYPE_FRAMEBUFFER_EXT,
    eCommandPool              = VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_POOL_EXT,
    eSurfaceKHR               = VK_DEBUG_REPORT_OBJECT_TYPE_SURFACE_KHR_EXT,
    eSwapchainKHR             = VK_DEBUG_REPORT_OBJECT_TYPE_SWAPCHAIN_KHR_EXT,
    eDebugReportCallbackEXT   = VK_DEBUG_REPORT_OBJECT_TYPE_DEBUG_REPORT_CALLBACK_EXT_EXT,
    eDisplayKHR               = VK_DEBUG_REPORT_OBJECT_TYPE_DISPLAY_KHR_EXT,
    eDisplayModeKHR           = VK_DEBUG_REPORT_OBJECT_TYPE_DISPLAY_MODE_KHR_EXT,
    eValidationCacheEXT       = VK_DEBUG_REPORT_OBJECT_TYPE_VALIDATION_CACHE_EXT_EXT,
    eSamplerYcbcrConversion   = VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION_EXT,
    eDescriptorUpdateTemplate = VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_EXT,
    eCuModuleNVX              = VK_DEBUG_REPORT_OBJECT_TYPE_CU_MODULE_NVX_EXT,
    eCuFunctionNVX            = VK_DEBUG_REPORT_OBJECT_TYPE_CU_FUNCTION_NVX_EXT,
    eAccelerationStructureKHR = VK_DEBUG_REPORT_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR_EXT,
    eAccelerationStructureNV  = VK_DEBUG_REPORT_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV_EXT,
#if defined( VK_USE_PLATFORM_FUCHSIA )
    eBufferCollectionFUCHSIA = VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_COLLECTION_FUCHSIA_EXT,
#endif /*VK_USE_PLATFORM_FUCHSIA*/
    eDebugReport                 = VK_DEBUG_REPORT_OBJECT_TYPE_DEBUG_REPORT_EXT,
    eDescriptorUpdateTemplateKHR = VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_KHR_EXT,
    eSamplerYcbcrConversionKHR   = VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION_KHR_EXT,
    eValidationCache             = VK_DEBUG_REPORT_OBJECT_TYPE_VALIDATION_CACHE_EXT
  };

  //=== VK_AMD_rasterization_order ===

  enum class RasterizationOrderAMD
  {
    eStrict  = VK_RASTERIZATION_ORDER_STRICT_AMD,
    eRelaxed = VK_RASTERIZATION_ORDER_RELAXED_AMD
  };

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_KHR_video_queue ===

  enum class VideoCodecOperationFlagBitsKHR : VkVideoCodecOperationFlagsKHR
  {
    eNone = VK_VIDEO_CODEC_OPERATION_NONE_KHR,
#  if defined( VK_ENABLE_BETA_EXTENSIONS )
    eEncodeH264EXT = VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_EXT,
    eEncodeH265EXT = VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_EXT,
    eDecodeH264EXT = VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_EXT,
    eDecodeH265EXT = VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_EXT
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/
  };

  enum class VideoChromaSubsamplingFlagBitsKHR : VkVideoChromaSubsamplingFlagsKHR
  {
    eInvalid    = VK_VIDEO_CHROMA_SUBSAMPLING_INVALID_KHR,
    eMonochrome = VK_VIDEO_CHROMA_SUBSAMPLING_MONOCHROME_BIT_KHR,
    e420        = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR,
    e422        = VK_VIDEO_CHROMA_SUBSAMPLING_422_BIT_KHR,
    e444        = VK_VIDEO_CHROMA_SUBSAMPLING_444_BIT_KHR
  };

  enum class VideoComponentBitDepthFlagBitsKHR : VkVideoComponentBitDepthFlagsKHR
  {
    eInvalid = VK_VIDEO_COMPONENT_BIT_DEPTH_INVALID_KHR,
    e8       = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
    e10      = VK_VIDEO_COMPONENT_BIT_DEPTH_10_BIT_KHR,
    e12      = VK_VIDEO_COMPONENT_BIT_DEPTH_12_BIT_KHR
  };

  enum class VideoCapabilityFlagBitsKHR : VkVideoCapabilityFlagsKHR
  {
    eProtectedContent        = VK_VIDEO_CAPABILITY_PROTECTED_CONTENT_BIT_KHR,
    eSeparateReferenceImages = VK_VIDEO_CAPABILITY_SEPARATE_REFERENCE_IMAGES_BIT_KHR
  };

  enum class VideoSessionCreateFlagBitsKHR : VkVideoSessionCreateFlagsKHR
  {
    eProtectedContent = VK_VIDEO_SESSION_CREATE_PROTECTED_CONTENT_BIT_KHR
  };

  enum class VideoCodingControlFlagBitsKHR : VkVideoCodingControlFlagsKHR
  {
    eReset = VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR,
#  if defined( VK_ENABLE_BETA_EXTENSIONS )
    eEncodeRateControl      = VK_VIDEO_CODING_CONTROL_ENCODE_RATE_CONTROL_BIT_KHR,
    eEncodeRateControlLayer = VK_VIDEO_CODING_CONTROL_ENCODE_RATE_CONTROL_LAYER_BIT_KHR
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/
  };

  enum class QueryResultStatusKHR
  {
    eError    = VK_QUERY_RESULT_STATUS_ERROR_KHR,
    eNotReady = VK_QUERY_RESULT_STATUS_NOT_READY_KHR,
    eComplete = VK_QUERY_RESULT_STATUS_COMPLETE_KHR
  };

  enum class VideoSessionParametersCreateFlagBitsKHR : VkVideoSessionParametersCreateFlagsKHR
  {
  };

  enum class VideoBeginCodingFlagBitsKHR : VkVideoBeginCodingFlagsKHR
  {
  };

  enum class VideoEndCodingFlagBitsKHR : VkVideoEndCodingFlagsKHR
  {
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_KHR_video_decode_queue ===

  enum class VideoDecodeCapabilityFlagBitsKHR : VkVideoDecodeCapabilityFlagsKHR
  {
    eDpbAndOutputCoincide = VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR,
    eDpbAndOutputDistinct = VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_DISTINCT_BIT_KHR
  };

  enum class VideoDecodeUsageFlagBitsKHR : VkVideoDecodeUsageFlagsKHR
  {
    eDefault     = VK_VIDEO_DECODE_USAGE_DEFAULT_KHR,
    eTranscoding = VK_VIDEO_DECODE_USAGE_TRANSCODING_BIT_KHR,
    eOffline     = VK_VIDEO_DECODE_USAGE_OFFLINE_BIT_KHR,
    eStreaming   = VK_VIDEO_DECODE_USAGE_STREAMING_BIT_KHR
  };

  enum class VideoDecodeFlagBitsKHR : VkVideoDecodeFlagsKHR
  {
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_EXT_transform_feedback ===

  enum class PipelineRasterizationStateStreamCreateFlagBitsEXT : VkPipelineRasterizationStateStreamCreateFlagsEXT
  {
  };

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_EXT_video_encode_h264 ===

  enum class VideoEncodeH264CapabilityFlagBitsEXT : VkVideoEncodeH264CapabilityFlagsEXT
  {
    eDirect8X8InferenceEnabled   = VK_VIDEO_ENCODE_H264_CAPABILITY_DIRECT_8X8_INFERENCE_ENABLED_BIT_EXT,
    eDirect8X8InferenceDisabled  = VK_VIDEO_ENCODE_H264_CAPABILITY_DIRECT_8X8_INFERENCE_DISABLED_BIT_EXT,
    eSeparateColourPlane         = VK_VIDEO_ENCODE_H264_CAPABILITY_SEPARATE_COLOUR_PLANE_BIT_EXT,
    eQpprimeYZeroTransformBypass = VK_VIDEO_ENCODE_H264_CAPABILITY_QPPRIME_Y_ZERO_TRANSFORM_BYPASS_BIT_EXT,
    eScalingLists                = VK_VIDEO_ENCODE_H264_CAPABILITY_SCALING_LISTS_BIT_EXT,
    eHrdCompliance               = VK_VIDEO_ENCODE_H264_CAPABILITY_HRD_COMPLIANCE_BIT_EXT,
    eChromaQpOffset              = VK_VIDEO_ENCODE_H264_CAPABILITY_CHROMA_QP_OFFSET_BIT_EXT,
    eSecondChromaQpOffset        = VK_VIDEO_ENCODE_H264_CAPABILITY_SECOND_CHROMA_QP_OFFSET_BIT_EXT,
    ePicInitQpMinus26            = VK_VIDEO_ENCODE_H264_CAPABILITY_PIC_INIT_QP_MINUS26_BIT_EXT,
    eWeightedPred                = VK_VIDEO_ENCODE_H264_CAPABILITY_WEIGHTED_PRED_BIT_EXT,
    eWeightedBipredExplicit      = VK_VIDEO_ENCODE_H264_CAPABILITY_WEIGHTED_BIPRED_EXPLICIT_BIT_EXT,
    eWeightedBipredImplicit      = VK_VIDEO_ENCODE_H264_CAPABILITY_WEIGHTED_BIPRED_IMPLICIT_BIT_EXT,
    eWeightedPredNoTable         = VK_VIDEO_ENCODE_H264_CAPABILITY_WEIGHTED_PRED_NO_TABLE_BIT_EXT,
    eTransform8X8                = VK_VIDEO_ENCODE_H264_CAPABILITY_TRANSFORM_8X8_BIT_EXT,
    eCabac                       = VK_VIDEO_ENCODE_H264_CAPABILITY_CABAC_BIT_EXT,
    eCavlc                       = VK_VIDEO_ENCODE_H264_CAPABILITY_CAVLC_BIT_EXT,
    eDeblockingFilterDisabled    = VK_VIDEO_ENCODE_H264_CAPABILITY_DEBLOCKING_FILTER_DISABLED_BIT_EXT,
    eDeblockingFilterEnabled     = VK_VIDEO_ENCODE_H264_CAPABILITY_DEBLOCKING_FILTER_ENABLED_BIT_EXT,
    eDeblockingFilterPartial     = VK_VIDEO_ENCODE_H264_CAPABILITY_DEBLOCKING_FILTER_PARTIAL_BIT_EXT,
    eDisableDirectSpatialMvPred  = VK_VIDEO_ENCODE_H264_CAPABILITY_DISABLE_DIRECT_SPATIAL_MV_PRED_BIT_EXT,
    eMultipleSlicePerFrame       = VK_VIDEO_ENCODE_H264_CAPABILITY_MULTIPLE_SLICE_PER_FRAME_BIT_EXT,
    eSliceMbCount                = VK_VIDEO_ENCODE_H264_CAPABILITY_SLICE_MB_COUNT_BIT_EXT,
    eRowUnalignedSlice           = VK_VIDEO_ENCODE_H264_CAPABILITY_ROW_UNALIGNED_SLICE_BIT_EXT,
    eDifferentSliceType          = VK_VIDEO_ENCODE_H264_CAPABILITY_DIFFERENT_SLICE_TYPE_BIT_EXT,
    eBFrameInL1List              = VK_VIDEO_ENCODE_H264_CAPABILITY_B_FRAME_IN_L1_LIST_BIT_EXT
  };

  enum class VideoEncodeH264InputModeFlagBitsEXT : VkVideoEncodeH264InputModeFlagsEXT
  {
    eFrame  = VK_VIDEO_ENCODE_H264_INPUT_MODE_FRAME_BIT_EXT,
    eSlice  = VK_VIDEO_ENCODE_H264_INPUT_MODE_SLICE_BIT_EXT,
    eNonVcl = VK_VIDEO_ENCODE_H264_INPUT_MODE_NON_VCL_BIT_EXT
  };

  enum class VideoEncodeH264OutputModeFlagBitsEXT : VkVideoEncodeH264OutputModeFlagsEXT
  {
    eFrame  = VK_VIDEO_ENCODE_H264_OUTPUT_MODE_FRAME_BIT_EXT,
    eSlice  = VK_VIDEO_ENCODE_H264_OUTPUT_MODE_SLICE_BIT_EXT,
    eNonVcl = VK_VIDEO_ENCODE_H264_OUTPUT_MODE_NON_VCL_BIT_EXT
  };

  enum class VideoEncodeH264RateControlStructureEXT
  {
    eUnknown = VK_VIDEO_ENCODE_H264_RATE_CONTROL_STRUCTURE_UNKNOWN_EXT,
    eFlat    = VK_VIDEO_ENCODE_H264_RATE_CONTROL_STRUCTURE_FLAT_EXT,
    eDyadic  = VK_VIDEO_ENCODE_H264_RATE_CONTROL_STRUCTURE_DYADIC_EXT
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_EXT_video_encode_h265 ===

  enum class VideoEncodeH265CapabilityFlagBitsEXT : VkVideoEncodeH265CapabilityFlagsEXT
  {
    eSeparateColourPlane             = VK_VIDEO_ENCODE_H265_CAPABILITY_SEPARATE_COLOUR_PLANE_BIT_EXT,
    eScalingLists                    = VK_VIDEO_ENCODE_H265_CAPABILITY_SCALING_LISTS_BIT_EXT,
    eSampleAdaptiveOffsetEnabled     = VK_VIDEO_ENCODE_H265_CAPABILITY_SAMPLE_ADAPTIVE_OFFSET_ENABLED_BIT_EXT,
    ePcmEnable                       = VK_VIDEO_ENCODE_H265_CAPABILITY_PCM_ENABLE_BIT_EXT,
    eSpsTemporalMvpEnabled           = VK_VIDEO_ENCODE_H265_CAPABILITY_SPS_TEMPORAL_MVP_ENABLED_BIT_EXT,
    eHrdCompliance                   = VK_VIDEO_ENCODE_H265_CAPABILITY_HRD_COMPLIANCE_BIT_EXT,
    eInitQpMinus26                   = VK_VIDEO_ENCODE_H265_CAPABILITY_INIT_QP_MINUS26_BIT_EXT,
    eLog2ParallelMergeLevelMinus2    = VK_VIDEO_ENCODE_H265_CAPABILITY_LOG2_PARALLEL_MERGE_LEVEL_MINUS2_BIT_EXT,
    eSignDataHidingEnabled           = VK_VIDEO_ENCODE_H265_CAPABILITY_SIGN_DATA_HIDING_ENABLED_BIT_EXT,
    eTransformSkipEnabled            = VK_VIDEO_ENCODE_H265_CAPABILITY_TRANSFORM_SKIP_ENABLED_BIT_EXT,
    eTransformSkipDisabled           = VK_VIDEO_ENCODE_H265_CAPABILITY_TRANSFORM_SKIP_DISABLED_BIT_EXT,
    ePpsSliceChromaQpOffsetsPresent  = VK_VIDEO_ENCODE_H265_CAPABILITY_PPS_SLICE_CHROMA_QP_OFFSETS_PRESENT_BIT_EXT,
    eWeightedPred                    = VK_VIDEO_ENCODE_H265_CAPABILITY_WEIGHTED_PRED_BIT_EXT,
    eWeightedBipred                  = VK_VIDEO_ENCODE_H265_CAPABILITY_WEIGHTED_BIPRED_BIT_EXT,
    eWeightedPredNoTable             = VK_VIDEO_ENCODE_H265_CAPABILITY_WEIGHTED_PRED_NO_TABLE_BIT_EXT,
    eTransquantBypassEnabled         = VK_VIDEO_ENCODE_H265_CAPABILITY_TRANSQUANT_BYPASS_ENABLED_BIT_EXT,
    eEntropyCodingSyncEnabled        = VK_VIDEO_ENCODE_H265_CAPABILITY_ENTROPY_CODING_SYNC_ENABLED_BIT_EXT,
    eDeblockingFilterOverrideEnabled = VK_VIDEO_ENCODE_H265_CAPABILITY_DEBLOCKING_FILTER_OVERRIDE_ENABLED_BIT_EXT,
    eMultipleTilePerFrame            = VK_VIDEO_ENCODE_H265_CAPABILITY_MULTIPLE_TILE_PER_FRAME_BIT_EXT,
    eMultipleSlicePerTile            = VK_VIDEO_ENCODE_H265_CAPABILITY_MULTIPLE_SLICE_PER_TILE_BIT_EXT,
    eMultipleTilePerSlice            = VK_VIDEO_ENCODE_H265_CAPABILITY_MULTIPLE_TILE_PER_SLICE_BIT_EXT,
    eSliceSegmentCtbCount            = VK_VIDEO_ENCODE_H265_CAPABILITY_SLICE_SEGMENT_CTB_COUNT_BIT_EXT,
    eRowUnalignedSliceSegment        = VK_VIDEO_ENCODE_H265_CAPABILITY_ROW_UNALIGNED_SLICE_SEGMENT_BIT_EXT,
    eDependentSliceSegment           = VK_VIDEO_ENCODE_H265_CAPABILITY_DEPENDENT_SLICE_SEGMENT_BIT_EXT,
    eDifferentSliceType              = VK_VIDEO_ENCODE_H265_CAPABILITY_DIFFERENT_SLICE_TYPE_BIT_EXT,
    eBFrameInL1List                  = VK_VIDEO_ENCODE_H265_CAPABILITY_B_FRAME_IN_L1_LIST_BIT_EXT
  };

  enum class VideoEncodeH265InputModeFlagBitsEXT : VkVideoEncodeH265InputModeFlagsEXT
  {
    eFrame        = VK_VIDEO_ENCODE_H265_INPUT_MODE_FRAME_BIT_EXT,
    eSliceSegment = VK_VIDEO_ENCODE_H265_INPUT_MODE_SLICE_SEGMENT_BIT_EXT,
    eNonVcl       = VK_VIDEO_ENCODE_H265_INPUT_MODE_NON_VCL_BIT_EXT
  };

  enum class VideoEncodeH265OutputModeFlagBitsEXT : VkVideoEncodeH265OutputModeFlagsEXT
  {
    eFrame        = VK_VIDEO_ENCODE_H265_OUTPUT_MODE_FRAME_BIT_EXT,
    eSliceSegment = VK_VIDEO_ENCODE_H265_OUTPUT_MODE_SLICE_SEGMENT_BIT_EXT,
    eNonVcl       = VK_VIDEO_ENCODE_H265_OUTPUT_MODE_NON_VCL_BIT_EXT
  };

  enum class VideoEncodeH265CtbSizeFlagBitsEXT : VkVideoEncodeH265CtbSizeFlagsEXT
  {
    e16 = VK_VIDEO_ENCODE_H265_CTB_SIZE_16_BIT_EXT,
    e32 = VK_VIDEO_ENCODE_H265_CTB_SIZE_32_BIT_EXT,
    e64 = VK_VIDEO_ENCODE_H265_CTB_SIZE_64_BIT_EXT
  };

  enum class VideoEncodeH265TransformBlockSizeFlagBitsEXT : VkVideoEncodeH265TransformBlockSizeFlagsEXT
  {
    e4  = VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_4_BIT_EXT,
    e8  = VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_8_BIT_EXT,
    e16 = VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_16_BIT_EXT,
    e32 = VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_32_BIT_EXT
  };

  enum class VideoEncodeH265RateControlStructureEXT
  {
    eUnknown = VK_VIDEO_ENCODE_H265_RATE_CONTROL_STRUCTURE_UNKNOWN_EXT,
    eFlat    = VK_VIDEO_ENCODE_H265_RATE_CONTROL_STRUCTURE_FLAT_EXT,
    eDyadic  = VK_VIDEO_ENCODE_H265_RATE_CONTROL_STRUCTURE_DYADIC_EXT
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_EXT_video_decode_h264 ===

  enum class VideoDecodeH264PictureLayoutFlagBitsEXT : VkVideoDecodeH264PictureLayoutFlagsEXT
  {
    eProgressive                = VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_PROGRESSIVE_EXT,
    eInterlacedInterleavedLines = VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_INTERLACED_INTERLEAVED_LINES_BIT_EXT,
    eInterlacedSeparatePlanes   = VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_INTERLACED_SEPARATE_PLANES_BIT_EXT
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_AMD_shader_info ===

  enum class ShaderInfoTypeAMD
  {
    eStatistics  = VK_SHADER_INFO_TYPE_STATISTICS_AMD,
    eBinary      = VK_SHADER_INFO_TYPE_BINARY_AMD,
    eDisassembly = VK_SHADER_INFO_TYPE_DISASSEMBLY_AMD
  };

#if defined( VK_USE_PLATFORM_GGP )
  //=== VK_GGP_stream_descriptor_surface ===

  enum class StreamDescriptorSurfaceCreateFlagBitsGGP : VkStreamDescriptorSurfaceCreateFlagsGGP
  {
  };
#endif /*VK_USE_PLATFORM_GGP*/

  //=== VK_NV_external_memory_capabilities ===

  enum class ExternalMemoryHandleTypeFlagBitsNV : VkExternalMemoryHandleTypeFlagsNV
  {
    eOpaqueWin32    = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_NV,
    eOpaqueWin32Kmt = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_NV,
    eD3D11Image     = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_IMAGE_BIT_NV,
    eD3D11ImageKmt  = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_IMAGE_KMT_BIT_NV
  };

  enum class ExternalMemoryFeatureFlagBitsNV : VkExternalMemoryFeatureFlagsNV
  {
    eDedicatedOnly = VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT_NV,
    eExportable    = VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT_NV,
    eImportable    = VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT_NV
  };

  //=== VK_EXT_validation_flags ===

  enum class ValidationCheckEXT
  {
    eAll     = VK_VALIDATION_CHECK_ALL_EXT,
    eShaders = VK_VALIDATION_CHECK_SHADERS_EXT
  };

#if defined( VK_USE_PLATFORM_VI_NN )
  //=== VK_NN_vi_surface ===

  enum class ViSurfaceCreateFlagBitsNN : VkViSurfaceCreateFlagsNN
  {
  };
#endif /*VK_USE_PLATFORM_VI_NN*/

  //=== VK_EXT_pipeline_robustness ===

  enum class PipelineRobustnessBufferBehaviorEXT
  {
    eDeviceDefault       = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DEVICE_DEFAULT_EXT,
    eDisabled            = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
    eRobustBufferAccess  = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_EXT,
    eRobustBufferAccess2 = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT
  };

  enum class PipelineRobustnessImageBehaviorEXT
  {
    eDeviceDefault      = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DEVICE_DEFAULT_EXT,
    eDisabled           = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
    eRobustImageAccess  = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_ROBUST_IMAGE_ACCESS_EXT,
    eRobustImageAccess2 = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_ROBUST_IMAGE_ACCESS_2_EXT
  };

  //=== VK_EXT_conditional_rendering ===

  enum class ConditionalRenderingFlagBitsEXT : VkConditionalRenderingFlagsEXT
  {
    eInverted = VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT
  };

  //=== VK_EXT_display_surface_counter ===

  enum class SurfaceCounterFlagBitsEXT : VkSurfaceCounterFlagsEXT
  {
    eVblank = VK_SURFACE_COUNTER_VBLANK_BIT_EXT
  };

  //=== VK_EXT_display_control ===

  enum class DisplayPowerStateEXT
  {
    eOff     = VK_DISPLAY_POWER_STATE_OFF_EXT,
    eSuspend = VK_DISPLAY_POWER_STATE_SUSPEND_EXT,
    eOn      = VK_DISPLAY_POWER_STATE_ON_EXT
  };

  enum class DeviceEventTypeEXT
  {
    eDisplayHotplug = VK_DEVICE_EVENT_TYPE_DISPLAY_HOTPLUG_EXT
  };

  enum class DisplayEventTypeEXT
  {
    eFirstPixelOut = VK_DISPLAY_EVENT_TYPE_FIRST_PIXEL_OUT_EXT
  };

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

  enum class PipelineViewportSwizzleStateCreateFlagBitsNV : VkPipelineViewportSwizzleStateCreateFlagsNV
  {
  };

  //=== VK_EXT_discard_rectangles ===

  enum class DiscardRectangleModeEXT
  {
    eInclusive = VK_DISCARD_RECTANGLE_MODE_INCLUSIVE_EXT,
    eExclusive = VK_DISCARD_RECTANGLE_MODE_EXCLUSIVE_EXT
  };

  enum class PipelineDiscardRectangleStateCreateFlagBitsEXT : VkPipelineDiscardRectangleStateCreateFlagsEXT
  {
  };

  //=== VK_EXT_conservative_rasterization ===

  enum class ConservativeRasterizationModeEXT
  {
    eDisabled      = VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT,
    eOverestimate  = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT,
    eUnderestimate = VK_CONSERVATIVE_RASTERIZATION_MODE_UNDERESTIMATE_EXT
  };

  enum class PipelineRasterizationConservativeStateCreateFlagBitsEXT : VkPipelineRasterizationConservativeStateCreateFlagsEXT
  {
  };

  //=== VK_EXT_depth_clip_enable ===

  enum class PipelineRasterizationDepthClipStateCreateFlagBitsEXT : VkPipelineRasterizationDepthClipStateCreateFlagsEXT
  {
  };

  //=== VK_KHR_performance_query ===

  enum class PerformanceCounterDescriptionFlagBitsKHR : VkPerformanceCounterDescriptionFlagsKHR
  {
    ePerformanceImpacting = VK_PERFORMANCE_COUNTER_DESCRIPTION_PERFORMANCE_IMPACTING_BIT_KHR,
    eConcurrentlyImpacted = VK_PERFORMANCE_COUNTER_DESCRIPTION_CONCURRENTLY_IMPACTED_BIT_KHR
  };

  enum class PerformanceCounterScopeKHR
  {
    eCommandBuffer             = VK_PERFORMANCE_COUNTER_SCOPE_COMMAND_BUFFER_KHR,
    eRenderPass                = VK_PERFORMANCE_COUNTER_SCOPE_RENDER_PASS_KHR,
    eCommand                   = VK_PERFORMANCE_COUNTER_SCOPE_COMMAND_KHR,
    eVkQueryScopeCommandBuffer = VK_QUERY_SCOPE_COMMAND_BUFFER_KHR,
    eVkQueryScopeCommand       = VK_QUERY_SCOPE_COMMAND_KHR,
    eVkQueryScopeRenderPass    = VK_QUERY_SCOPE_RENDER_PASS_KHR
  };

  enum class PerformanceCounterStorageKHR
  {
    eInt32   = VK_PERFORMANCE_COUNTER_STORAGE_INT32_KHR,
    eInt64   = VK_PERFORMANCE_COUNTER_STORAGE_INT64_KHR,
    eUint32  = VK_PERFORMANCE_COUNTER_STORAGE_UINT32_KHR,
    eUint64  = VK_PERFORMANCE_COUNTER_STORAGE_UINT64_KHR,
    eFloat32 = VK_PERFORMANCE_COUNTER_STORAGE_FLOAT32_KHR,
    eFloat64 = VK_PERFORMANCE_COUNTER_STORAGE_FLOAT64_KHR
  };

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

  enum class AcquireProfilingLockFlagBitsKHR : VkAcquireProfilingLockFlagsKHR
  {
  };

#if defined( VK_USE_PLATFORM_IOS_MVK )
  //=== VK_MVK_ios_surface ===

  enum class IOSSurfaceCreateFlagBitsMVK : VkIOSSurfaceCreateFlagsMVK
  {
  };
#endif /*VK_USE_PLATFORM_IOS_MVK*/

#if defined( VK_USE_PLATFORM_MACOS_MVK )
  //=== VK_MVK_macos_surface ===

  enum class MacOSSurfaceCreateFlagBitsMVK : VkMacOSSurfaceCreateFlagsMVK
  {
  };
#endif /*VK_USE_PLATFORM_MACOS_MVK*/

  //=== VK_EXT_debug_utils ===

  enum class DebugUtilsMessageSeverityFlagBitsEXT : VkDebugUtilsMessageSeverityFlagsEXT
  {
    eVerbose = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT,
    eInfo    = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT,
    eWarning = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
    eError   = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
  };

  enum class DebugUtilsMessageTypeFlagBitsEXT : VkDebugUtilsMessageTypeFlagsEXT
  {
    eGeneral              = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT,
    eValidation           = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
    ePerformance          = VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
    eDeviceAddressBinding = VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT
  };

  enum class DebugUtilsMessengerCallbackDataFlagBitsEXT : VkDebugUtilsMessengerCallbackDataFlagsEXT
  {
  };

  enum class DebugUtilsMessengerCreateFlagBitsEXT : VkDebugUtilsMessengerCreateFlagsEXT
  {
  };

  //=== VK_EXT_blend_operation_advanced ===

  enum class BlendOverlapEXT
  {
    eUncorrelated = VK_BLEND_OVERLAP_UNCORRELATED_EXT,
    eDisjoint     = VK_BLEND_OVERLAP_DISJOINT_EXT,
    eConjoint     = VK_BLEND_OVERLAP_CONJOINT_EXT
  };

  //=== VK_NV_fragment_coverage_to_color ===

  enum class PipelineCoverageToColorStateCreateFlagBitsNV : VkPipelineCoverageToColorStateCreateFlagsNV
  {
  };

  //=== VK_KHR_acceleration_structure ===

  enum class AccelerationStructureTypeKHR
  {
    eTopLevel    = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
    eBottomLevel = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
    eGeneric     = VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR
  };
  using AccelerationStructureTypeNV = AccelerationStructureTypeKHR;

  enum class AccelerationStructureBuildTypeKHR
  {
    eHost         = VK_ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_KHR,
    eDevice       = VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
    eHostOrDevice = VK_ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_OR_DEVICE_KHR
  };

  enum class GeometryFlagBitsKHR : VkGeometryFlagsKHR
  {
    eOpaque                      = VK_GEOMETRY_OPAQUE_BIT_KHR,
    eNoDuplicateAnyHitInvocation = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR
  };
  using GeometryFlagBitsNV = GeometryFlagBitsKHR;

  enum class GeometryInstanceFlagBitsKHR : VkGeometryInstanceFlagsKHR
  {
    eTriangleFacingCullDisable        = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
    eTriangleFlipFacing               = VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR,
    eForceOpaque                      = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR,
    eForceNoOpaque                    = VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR,
    eForceOpacityMicromap2StateEXT    = VK_GEOMETRY_INSTANCE_FORCE_OPACITY_MICROMAP_2_STATE_EXT,
    eDisableOpacityMicromapsEXT       = VK_GEOMETRY_INSTANCE_DISABLE_OPACITY_MICROMAPS_EXT,
    eTriangleCullDisable              = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV,
    eTriangleFrontCounterclockwiseKHR = VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR,
    eTriangleFrontCounterclockwise    = VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_NV
  };
  using GeometryInstanceFlagBitsNV = GeometryInstanceFlagBitsKHR;

  enum class BuildAccelerationStructureFlagBitsKHR : VkBuildAccelerationStructureFlagsKHR
  {
    eAllowUpdate                       = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
    eAllowCompaction                   = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR,
    ePreferFastTrace                   = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
    ePreferFastBuild                   = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
    eLowMemory                         = VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR,
    eMotionNV                          = VK_BUILD_ACCELERATION_STRUCTURE_MOTION_BIT_NV,
    eAllowOpacityMicromapUpdateEXT     = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_OPACITY_MICROMAP_UPDATE_EXT,
    eAllowDisableOpacityMicromapsEXT   = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DISABLE_OPACITY_MICROMAPS_EXT,
    eAllowOpacityMicromapDataUpdateEXT = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_OPACITY_MICROMAP_DATA_UPDATE_EXT
  };
  using BuildAccelerationStructureFlagBitsNV = BuildAccelerationStructureFlagBitsKHR;

  enum class CopyAccelerationStructureModeKHR
  {
    eClone       = VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR,
    eCompact     = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR,
    eSerialize   = VK_COPY_ACCELERATION_STRUCTURE_MODE_SERIALIZE_KHR,
    eDeserialize = VK_COPY_ACCELERATION_STRUCTURE_MODE_DESERIALIZE_KHR
  };
  using CopyAccelerationStructureModeNV = CopyAccelerationStructureModeKHR;

  enum class GeometryTypeKHR
  {
    eTriangles = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
    eAabbs     = VK_GEOMETRY_TYPE_AABBS_KHR,
    eInstances = VK_GEOMETRY_TYPE_INSTANCES_KHR
  };
  using GeometryTypeNV = GeometryTypeKHR;

  enum class AccelerationStructureCompatibilityKHR
  {
    eCompatible   = VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR,
    eIncompatible = VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR
  };

  enum class AccelerationStructureCreateFlagBitsKHR : VkAccelerationStructureCreateFlagsKHR
  {
    eDeviceAddressCaptureReplay = VK_ACCELERATION_STRUCTURE_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT_KHR,
    eMotionNV                   = VK_ACCELERATION_STRUCTURE_CREATE_MOTION_BIT_NV
  };

  enum class BuildAccelerationStructureModeKHR
  {
    eBuild  = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
    eUpdate = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
  };

  //=== VK_NV_framebuffer_mixed_samples ===

  enum class CoverageModulationModeNV
  {
    eNone  = VK_COVERAGE_MODULATION_MODE_NONE_NV,
    eRgb   = VK_COVERAGE_MODULATION_MODE_RGB_NV,
    eAlpha = VK_COVERAGE_MODULATION_MODE_ALPHA_NV,
    eRgba  = VK_COVERAGE_MODULATION_MODE_RGBA_NV
  };

  enum class PipelineCoverageModulationStateCreateFlagBitsNV : VkPipelineCoverageModulationStateCreateFlagsNV
  {
  };

  //=== VK_EXT_validation_cache ===

  enum class ValidationCacheHeaderVersionEXT
  {
    eOne = VK_VALIDATION_CACHE_HEADER_VERSION_ONE_EXT
  };

  enum class ValidationCacheCreateFlagBitsEXT : VkValidationCacheCreateFlagsEXT
  {
  };

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

  enum class CoarseSampleOrderTypeNV
  {
    eDefault     = VK_COARSE_SAMPLE_ORDER_TYPE_DEFAULT_NV,
    eCustom      = VK_COARSE_SAMPLE_ORDER_TYPE_CUSTOM_NV,
    ePixelMajor  = VK_COARSE_SAMPLE_ORDER_TYPE_PIXEL_MAJOR_NV,
    eSampleMajor = VK_COARSE_SAMPLE_ORDER_TYPE_SAMPLE_MAJOR_NV
  };

  //=== VK_NV_ray_tracing ===

  enum class AccelerationStructureMemoryRequirementsTypeNV
  {
    eObject        = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV,
    eBuildScratch  = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV,
    eUpdateScratch = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_UPDATE_SCRATCH_NV
  };

  //=== VK_AMD_pipeline_compiler_control ===

  enum class PipelineCompilerControlFlagBitsAMD : VkPipelineCompilerControlFlagsAMD
  {
  };

  //=== VK_EXT_calibrated_timestamps ===

  enum class TimeDomainEXT
  {
    eDevice                  = VK_TIME_DOMAIN_DEVICE_EXT,
    eClockMonotonic          = VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT,
    eClockMonotonicRaw       = VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_EXT,
    eQueryPerformanceCounter = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT
  };

  //=== VK_KHR_global_priority ===

  enum class QueueGlobalPriorityKHR
  {
    eLow      = VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR,
    eMedium   = VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR,
    eHigh     = VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR,
    eRealtime = VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR
  };
  using QueueGlobalPriorityEXT = QueueGlobalPriorityKHR;

  //=== VK_AMD_memory_overallocation_behavior ===

  enum class MemoryOverallocationBehaviorAMD
  {
    eDefault    = VK_MEMORY_OVERALLOCATION_BEHAVIOR_DEFAULT_AMD,
    eAllowed    = VK_MEMORY_OVERALLOCATION_BEHAVIOR_ALLOWED_AMD,
    eDisallowed = VK_MEMORY_OVERALLOCATION_BEHAVIOR_DISALLOWED_AMD
  };

  //=== VK_INTEL_performance_query ===

  enum class PerformanceConfigurationTypeINTEL
  {
    eCommandQueueMetricsDiscoveryActivated = VK_PERFORMANCE_CONFIGURATION_TYPE_COMMAND_QUEUE_METRICS_DISCOVERY_ACTIVATED_INTEL
  };

  enum class QueryPoolSamplingModeINTEL
  {
    eManual = VK_QUERY_POOL_SAMPLING_MODE_MANUAL_INTEL
  };

  enum class PerformanceOverrideTypeINTEL
  {
    eNullHardware   = VK_PERFORMANCE_OVERRIDE_TYPE_NULL_HARDWARE_INTEL,
    eFlushGpuCaches = VK_PERFORMANCE_OVERRIDE_TYPE_FLUSH_GPU_CACHES_INTEL
  };

  enum class PerformanceParameterTypeINTEL
  {
    eHwCountersSupported   = VK_PERFORMANCE_PARAMETER_TYPE_HW_COUNTERS_SUPPORTED_INTEL,
    eStreamMarkerValidBits = VK_PERFORMANCE_PARAMETER_TYPE_STREAM_MARKER_VALID_BITS_INTEL
  };

  enum class PerformanceValueTypeINTEL
  {
    eUint32 = VK_PERFORMANCE_VALUE_TYPE_UINT32_INTEL,
    eUint64 = VK_PERFORMANCE_VALUE_TYPE_UINT64_INTEL,
    eFloat  = VK_PERFORMANCE_VALUE_TYPE_FLOAT_INTEL,
    eBool   = VK_PERFORMANCE_VALUE_TYPE_BOOL_INTEL,
    eString = VK_PERFORMANCE_VALUE_TYPE_STRING_INTEL
  };

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_imagepipe_surface ===

  enum class ImagePipeSurfaceCreateFlagBitsFUCHSIA : VkImagePipeSurfaceCreateFlagsFUCHSIA
  {
  };
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_surface ===

  enum class MetalSurfaceCreateFlagBitsEXT : VkMetalSurfaceCreateFlagsEXT
  {
  };
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

  //=== VK_AMD_shader_core_properties2 ===

  enum class ShaderCorePropertiesFlagBitsAMD : VkShaderCorePropertiesFlagsAMD
  {
  };

  //=== VK_EXT_validation_features ===

  enum class ValidationFeatureEnableEXT
  {
    eGpuAssisted                   = VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT,
    eGpuAssistedReserveBindingSlot = VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT,
    eBestPractices                 = VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT,
    eDebugPrintf                   = VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT,
    eSynchronizationValidation     = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT
  };

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

  //=== VK_NV_cooperative_matrix ===

  enum class ScopeNV
  {
    eDevice      = VK_SCOPE_DEVICE_NV,
    eWorkgroup   = VK_SCOPE_WORKGROUP_NV,
    eSubgroup    = VK_SCOPE_SUBGROUP_NV,
    eQueueFamily = VK_SCOPE_QUEUE_FAMILY_NV
  };

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

  //=== VK_NV_coverage_reduction_mode ===

  enum class CoverageReductionModeNV
  {
    eMerge    = VK_COVERAGE_REDUCTION_MODE_MERGE_NV,
    eTruncate = VK_COVERAGE_REDUCTION_MODE_TRUNCATE_NV
  };

  enum class PipelineCoverageReductionStateCreateFlagBitsNV : VkPipelineCoverageReductionStateCreateFlagsNV
  {
  };

  //=== VK_EXT_provoking_vertex ===

  enum class ProvokingVertexModeEXT
  {
    eFirstVertex = VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT,
    eLastVertex  = VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT
  };

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_EXT_full_screen_exclusive ===

  enum class FullScreenExclusiveEXT
  {
    eDefault               = VK_FULL_SCREEN_EXCLUSIVE_DEFAULT_EXT,
    eAllowed               = VK_FULL_SCREEN_EXCLUSIVE_ALLOWED_EXT,
    eDisallowed            = VK_FULL_SCREEN_EXCLUSIVE_DISALLOWED_EXT,
    eApplicationControlled = VK_FULL_SCREEN_EXCLUSIVE_APPLICATION_CONTROLLED_EXT
  };
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_EXT_headless_surface ===

  enum class HeadlessSurfaceCreateFlagBitsEXT : VkHeadlessSurfaceCreateFlagsEXT
  {
  };

  //=== VK_EXT_line_rasterization ===

  enum class LineRasterizationModeEXT
  {
    eDefault           = VK_LINE_RASTERIZATION_MODE_DEFAULT_EXT,
    eRectangular       = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_EXT,
    eBresenham         = VK_LINE_RASTERIZATION_MODE_BRESENHAM_EXT,
    eRectangularSmooth = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_EXT
  };

  //=== VK_KHR_pipeline_executable_properties ===

  enum class PipelineExecutableStatisticFormatKHR
  {
    eBool32  = VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR,
    eInt64   = VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR,
    eUint64  = VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR,
    eFloat64 = VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR
  };

  //=== VK_NV_device_generated_commands ===

  enum class IndirectStateFlagBitsNV : VkIndirectStateFlagsNV
  {
    eFlagFrontface = VK_INDIRECT_STATE_FLAG_FRONTFACE_BIT_NV
  };

  enum class IndirectCommandsTokenTypeNV
  {
    eShaderGroup   = VK_INDIRECT_COMMANDS_TOKEN_TYPE_SHADER_GROUP_NV,
    eStateFlags    = VK_INDIRECT_COMMANDS_TOKEN_TYPE_STATE_FLAGS_NV,
    eIndexBuffer   = VK_INDIRECT_COMMANDS_TOKEN_TYPE_INDEX_BUFFER_NV,
    eVertexBuffer  = VK_INDIRECT_COMMANDS_TOKEN_TYPE_VERTEX_BUFFER_NV,
    ePushConstant  = VK_INDIRECT_COMMANDS_TOKEN_TYPE_PUSH_CONSTANT_NV,
    eDrawIndexed   = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_INDEXED_NV,
    eDraw          = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_NV,
    eDrawTasks     = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_TASKS_NV,
    eDrawMeshTasks = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_MESH_TASKS_NV
  };

  enum class IndirectCommandsLayoutUsageFlagBitsNV : VkIndirectCommandsLayoutUsageFlagsNV
  {
    eExplicitPreprocess = VK_INDIRECT_COMMANDS_LAYOUT_USAGE_EXPLICIT_PREPROCESS_BIT_NV,
    eIndexedSequences   = VK_INDIRECT_COMMANDS_LAYOUT_USAGE_INDEXED_SEQUENCES_BIT_NV,
    eUnorderedSequences = VK_INDIRECT_COMMANDS_LAYOUT_USAGE_UNORDERED_SEQUENCES_BIT_NV
  };

  //=== VK_EXT_device_memory_report ===

  enum class DeviceMemoryReportEventTypeEXT
  {
    eAllocate         = VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_ALLOCATE_EXT,
    eFree             = VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_FREE_EXT,
    eImport           = VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_IMPORT_EXT,
    eUnimport         = VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_UNIMPORT_EXT,
    eAllocationFailed = VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_ALLOCATION_FAILED_EXT
  };

  enum class DeviceMemoryReportFlagBitsEXT : VkDeviceMemoryReportFlagsEXT
  {
  };

  //=== VK_EXT_pipeline_creation_cache_control ===

  enum class PipelineCacheCreateFlagBits : VkPipelineCacheCreateFlags
  {
    eExternallySynchronized    = VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT,
    eExternallySynchronizedEXT = VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT_EXT
  };

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_KHR_video_encode_queue ===

  enum class VideoEncodeCapabilityFlagBitsKHR : VkVideoEncodeCapabilityFlagsKHR
  {
    ePrecedingExternallyEncodedBytes = VK_VIDEO_ENCODE_CAPABILITY_PRECEDING_EXTERNALLY_ENCODED_BYTES_BIT_KHR
  };

  enum class VideoEncodeUsageFlagBitsKHR : VkVideoEncodeUsageFlagsKHR
  {
    eDefault      = VK_VIDEO_ENCODE_USAGE_DEFAULT_KHR,
    eTranscoding  = VK_VIDEO_ENCODE_USAGE_TRANSCODING_BIT_KHR,
    eStreaming    = VK_VIDEO_ENCODE_USAGE_STREAMING_BIT_KHR,
    eRecording    = VK_VIDEO_ENCODE_USAGE_RECORDING_BIT_KHR,
    eConferencing = VK_VIDEO_ENCODE_USAGE_CONFERENCING_BIT_KHR
  };

  enum class VideoEncodeContentFlagBitsKHR : VkVideoEncodeContentFlagsKHR
  {
    eDefault  = VK_VIDEO_ENCODE_CONTENT_DEFAULT_KHR,
    eCamera   = VK_VIDEO_ENCODE_CONTENT_CAMERA_BIT_KHR,
    eDesktop  = VK_VIDEO_ENCODE_CONTENT_DESKTOP_BIT_KHR,
    eRendered = VK_VIDEO_ENCODE_CONTENT_RENDERED_BIT_KHR
  };

  enum class VideoEncodeTuningModeKHR
  {
    eDefault         = VK_VIDEO_ENCODE_TUNING_MODE_DEFAULT_KHR,
    eHighQuality     = VK_VIDEO_ENCODE_TUNING_MODE_HIGH_QUALITY_KHR,
    eLowLatency      = VK_VIDEO_ENCODE_TUNING_MODE_LOW_LATENCY_KHR,
    eUltraLowLatency = VK_VIDEO_ENCODE_TUNING_MODE_ULTRA_LOW_LATENCY_KHR,
    eLossless        = VK_VIDEO_ENCODE_TUNING_MODE_LOSSLESS_KHR
  };

  enum class VideoEncodeRateControlModeFlagBitsKHR : VkVideoEncodeRateControlModeFlagsKHR
  {
    eNone = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_NONE_BIT_KHR,
    eCbr  = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR,
    eVbr  = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_VBR_BIT_KHR
  };

  enum class VideoEncodeFlagBitsKHR : VkVideoEncodeFlagsKHR
  {
  };

  enum class VideoEncodeRateControlFlagBitsKHR : VkVideoEncodeRateControlFlagsKHR
  {
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_NV_device_diagnostics_config ===

  enum class DeviceDiagnosticsConfigFlagBitsNV : VkDeviceDiagnosticsConfigFlagsNV
  {
    eEnableShaderDebugInfo      = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_DEBUG_INFO_BIT_NV,
    eEnableResourceTracking     = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_RESOURCE_TRACKING_BIT_NV,
    eEnableAutomaticCheckpoints = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV,
    eEnableShaderErrorReporting = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_ERROR_REPORTING_BIT_NV
  };

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_objects ===

  enum class ExportMetalObjectTypeFlagBitsEXT : VkExportMetalObjectTypeFlagsEXT
  {
    eMetalDevice       = VK_EXPORT_METAL_OBJECT_TYPE_METAL_DEVICE_BIT_EXT,
    eMetalCommandQueue = VK_EXPORT_METAL_OBJECT_TYPE_METAL_COMMAND_QUEUE_BIT_EXT,
    eMetalBuffer       = VK_EXPORT_METAL_OBJECT_TYPE_METAL_BUFFER_BIT_EXT,
    eMetalTexture      = VK_EXPORT_METAL_OBJECT_TYPE_METAL_TEXTURE_BIT_EXT,
    eMetalIosurface    = VK_EXPORT_METAL_OBJECT_TYPE_METAL_IOSURFACE_BIT_EXT,
    eMetalSharedEvent  = VK_EXPORT_METAL_OBJECT_TYPE_METAL_SHARED_EVENT_BIT_EXT
  };
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_EXT_graphics_pipeline_library ===

  enum class GraphicsPipelineLibraryFlagBitsEXT : VkGraphicsPipelineLibraryFlagsEXT
  {
    eVertexInputInterface    = VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT,
    ePreRasterizationShaders = VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT,
    eFragmentShader          = VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT,
    eFragmentOutputInterface = VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT
  };

  enum class PipelineLayoutCreateFlagBits : VkPipelineLayoutCreateFlags
  {
    eIndependentSetsEXT = VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT
  };

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

  enum class FragmentShadingRateTypeNV
  {
    eFragmentSize = VK_FRAGMENT_SHADING_RATE_TYPE_FRAGMENT_SIZE_NV,
    eEnums        = VK_FRAGMENT_SHADING_RATE_TYPE_ENUMS_NV
  };

  //=== VK_NV_ray_tracing_motion_blur ===

  enum class AccelerationStructureMotionInstanceTypeNV
  {
    eStatic       = VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_STATIC_NV,
    eMatrixMotion = VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_MATRIX_MOTION_NV,
    eSrtMotion    = VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_SRT_MOTION_NV
  };

  enum class AccelerationStructureMotionInfoFlagBitsNV : VkAccelerationStructureMotionInfoFlagsNV
  {
  };

  enum class AccelerationStructureMotionInstanceFlagBitsNV : VkAccelerationStructureMotionInstanceFlagsNV
  {
  };

  //=== VK_EXT_image_compression_control ===

  enum class ImageCompressionFlagBitsEXT : VkImageCompressionFlagsEXT
  {
    eDefault           = VK_IMAGE_COMPRESSION_DEFAULT_EXT,
    eFixedRateDefault  = VK_IMAGE_COMPRESSION_FIXED_RATE_DEFAULT_EXT,
    eFixedRateExplicit = VK_IMAGE_COMPRESSION_FIXED_RATE_EXPLICIT_EXT,
    eDisabled          = VK_IMAGE_COMPRESSION_DISABLED_EXT
  };

  enum class ImageCompressionFixedRateFlagBitsEXT : VkImageCompressionFixedRateFlagsEXT
  {
    eNone  = VK_IMAGE_COMPRESSION_FIXED_RATE_NONE_EXT,
    e1Bpc  = VK_IMAGE_COMPRESSION_FIXED_RATE_1BPC_BIT_EXT,
    e2Bpc  = VK_IMAGE_COMPRESSION_FIXED_RATE_2BPC_BIT_EXT,
    e3Bpc  = VK_IMAGE_COMPRESSION_FIXED_RATE_3BPC_BIT_EXT,
    e4Bpc  = VK_IMAGE_COMPRESSION_FIXED_RATE_4BPC_BIT_EXT,
    e5Bpc  = VK_IMAGE_COMPRESSION_FIXED_RATE_5BPC_BIT_EXT,
    e6Bpc  = VK_IMAGE_COMPRESSION_FIXED_RATE_6BPC_BIT_EXT,
    e7Bpc  = VK_IMAGE_COMPRESSION_FIXED_RATE_7BPC_BIT_EXT,
    e8Bpc  = VK_IMAGE_COMPRESSION_FIXED_RATE_8BPC_BIT_EXT,
    e9Bpc  = VK_IMAGE_COMPRESSION_FIXED_RATE_9BPC_BIT_EXT,
    e10Bpc = VK_IMAGE_COMPRESSION_FIXED_RATE_10BPC_BIT_EXT,
    e11Bpc = VK_IMAGE_COMPRESSION_FIXED_RATE_11BPC_BIT_EXT,
    e12Bpc = VK_IMAGE_COMPRESSION_FIXED_RATE_12BPC_BIT_EXT,
    e13Bpc = VK_IMAGE_COMPRESSION_FIXED_RATE_13BPC_BIT_EXT,
    e14Bpc = VK_IMAGE_COMPRESSION_FIXED_RATE_14BPC_BIT_EXT,
    e15Bpc = VK_IMAGE_COMPRESSION_FIXED_RATE_15BPC_BIT_EXT,
    e16Bpc = VK_IMAGE_COMPRESSION_FIXED_RATE_16BPC_BIT_EXT,
    e17Bpc = VK_IMAGE_COMPRESSION_FIXED_RATE_17BPC_BIT_EXT,
    e18Bpc = VK_IMAGE_COMPRESSION_FIXED_RATE_18BPC_BIT_EXT,
    e19Bpc = VK_IMAGE_COMPRESSION_FIXED_RATE_19BPC_BIT_EXT,
    e20Bpc = VK_IMAGE_COMPRESSION_FIXED_RATE_20BPC_BIT_EXT,
    e21Bpc = VK_IMAGE_COMPRESSION_FIXED_RATE_21BPC_BIT_EXT,
    e22Bpc = VK_IMAGE_COMPRESSION_FIXED_RATE_22BPC_BIT_EXT,
    e23Bpc = VK_IMAGE_COMPRESSION_FIXED_RATE_23BPC_BIT_EXT,
    e24Bpc = VK_IMAGE_COMPRESSION_FIXED_RATE_24BPC_BIT_EXT
  };

  //=== VK_EXT_device_fault ===

  enum class DeviceFaultAddressTypeEXT
  {
    eNone                      = VK_DEVICE_FAULT_ADDRESS_TYPE_NONE_EXT,
    eReadInvalid               = VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT,
    eWriteInvalid              = VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT,
    eExecuteInvalid            = VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT,
    eInstructionPointerUnknown = VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_UNKNOWN_EXT,
    eInstructionPointerInvalid = VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_INVALID_EXT,
    eInstructionPointerFault   = VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_FAULT_EXT
  };

  enum class DeviceFaultVendorBinaryHeaderVersionEXT
  {
    eOne = VK_DEVICE_FAULT_VENDOR_BINARY_HEADER_VERSION_ONE_EXT
  };

#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
  //=== VK_EXT_directfb_surface ===

  enum class DirectFBSurfaceCreateFlagBitsEXT : VkDirectFBSurfaceCreateFlagsEXT
  {
  };
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

  //=== VK_KHR_ray_tracing_pipeline ===

  enum class RayTracingShaderGroupTypeKHR
  {
    eGeneral            = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
    eTrianglesHitGroup  = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
    eProceduralHitGroup = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR
  };
  using RayTracingShaderGroupTypeNV = RayTracingShaderGroupTypeKHR;

  enum class ShaderGroupShaderKHR
  {
    eGeneral      = VK_SHADER_GROUP_SHADER_GENERAL_KHR,
    eClosestHit   = VK_SHADER_GROUP_SHADER_CLOSEST_HIT_KHR,
    eAnyHit       = VK_SHADER_GROUP_SHADER_ANY_HIT_KHR,
    eIntersection = VK_SHADER_GROUP_SHADER_INTERSECTION_KHR
  };

  //=== VK_EXT_device_address_binding_report ===

  enum class DeviceAddressBindingFlagBitsEXT : VkDeviceAddressBindingFlagsEXT
  {
    eInternalObject = VK_DEVICE_ADDRESS_BINDING_INTERNAL_OBJECT_BIT_EXT
  };

  enum class DeviceAddressBindingTypeEXT
  {
    eBind   = VK_DEVICE_ADDRESS_BINDING_TYPE_BIND_EXT,
    eUnbind = VK_DEVICE_ADDRESS_BINDING_TYPE_UNBIND_EXT
  };

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_buffer_collection ===

  enum class ImageConstraintsInfoFlagBitsFUCHSIA : VkImageConstraintsInfoFlagsFUCHSIA
  {
    eCpuReadRarely     = VK_IMAGE_CONSTRAINTS_INFO_CPU_READ_RARELY_FUCHSIA,
    eCpuReadOften      = VK_IMAGE_CONSTRAINTS_INFO_CPU_READ_OFTEN_FUCHSIA,
    eCpuWriteRarely    = VK_IMAGE_CONSTRAINTS_INFO_CPU_WRITE_RARELY_FUCHSIA,
    eCpuWriteOften     = VK_IMAGE_CONSTRAINTS_INFO_CPU_WRITE_OFTEN_FUCHSIA,
    eProtectedOptional = VK_IMAGE_CONSTRAINTS_INFO_PROTECTED_OPTIONAL_FUCHSIA
  };

  enum class ImageFormatConstraintsFlagBitsFUCHSIA : VkImageFormatConstraintsFlagsFUCHSIA
  {
  };
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
  //=== VK_QNX_screen_surface ===

  enum class ScreenSurfaceCreateFlagBitsQNX : VkScreenSurfaceCreateFlagsQNX
  {
  };
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

  //=== VK_EXT_opacity_micromap ===

  enum class MicromapTypeEXT
  {
    eOpacityMicromap = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT
  };

  enum class BuildMicromapFlagBitsEXT : VkBuildMicromapFlagsEXT
  {
    ePreferFastTrace = VK_BUILD_MICROMAP_PREFER_FAST_TRACE_BIT_EXT,
    ePreferFastBuild = VK_BUILD_MICROMAP_PREFER_FAST_BUILD_BIT_EXT,
    eAllowCompaction = VK_BUILD_MICROMAP_ALLOW_COMPACTION_BIT_EXT
  };

  enum class CopyMicromapModeEXT
  {
    eClone       = VK_COPY_MICROMAP_MODE_CLONE_EXT,
    eSerialize   = VK_COPY_MICROMAP_MODE_SERIALIZE_EXT,
    eDeserialize = VK_COPY_MICROMAP_MODE_DESERIALIZE_EXT,
    eCompact     = VK_COPY_MICROMAP_MODE_COMPACT_EXT
  };

  enum class MicromapCreateFlagBitsEXT : VkMicromapCreateFlagsEXT
  {
    eDeviceAddressCaptureReplay = VK_MICROMAP_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT_EXT
  };

  enum class BuildMicromapModeEXT
  {
    eBuild = VK_BUILD_MICROMAP_MODE_BUILD_EXT
  };

  enum class OpacityMicromapFormatEXT
  {
    e2State = VK_OPACITY_MICROMAP_FORMAT_2_STATE_EXT,
    e4State = VK_OPACITY_MICROMAP_FORMAT_4_STATE_EXT
  };

  enum class OpacityMicromapSpecialIndexEXT
  {
    eFullyTransparent        = VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_TRANSPARENT_EXT,
    eFullyOpaque             = VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_OPAQUE_EXT,
    eFullyUnknownTransparent = VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_UNKNOWN_TRANSPARENT_EXT,
    eFullyUnknownOpaque      = VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_UNKNOWN_OPAQUE_EXT
  };

  //=== VK_EXT_subpass_merge_feedback ===

  enum class SubpassMergeStatusEXT
  {
    eMerged                               = VK_SUBPASS_MERGE_STATUS_MERGED_EXT,
    eDisallowed                           = VK_SUBPASS_MERGE_STATUS_DISALLOWED_EXT,
    eNotMergedSideEffects                 = VK_SUBPASS_MERGE_STATUS_NOT_MERGED_SIDE_EFFECTS_EXT,
    eNotMergedSamplesMismatch             = VK_SUBPASS_MERGE_STATUS_NOT_MERGED_SAMPLES_MISMATCH_EXT,
    eNotMergedViewsMismatch               = VK_SUBPASS_MERGE_STATUS_NOT_MERGED_VIEWS_MISMATCH_EXT,
    eNotMergedAliasing                    = VK_SUBPASS_MERGE_STATUS_NOT_MERGED_ALIASING_EXT,
    eNotMergedDependencies                = VK_SUBPASS_MERGE_STATUS_NOT_MERGED_DEPENDENCIES_EXT,
    eNotMergedIncompatibleInputAttachment = VK_SUBPASS_MERGE_STATUS_NOT_MERGED_INCOMPATIBLE_INPUT_ATTACHMENT_EXT,
    eNotMergedTooManyAttachments          = VK_SUBPASS_MERGE_STATUS_NOT_MERGED_TOO_MANY_ATTACHMENTS_EXT,
    eNotMergedInsufficientStorage         = VK_SUBPASS_MERGE_STATUS_NOT_MERGED_INSUFFICIENT_STORAGE_EXT,
    eNotMergedDepthStencilCount           = VK_SUBPASS_MERGE_STATUS_NOT_MERGED_DEPTH_STENCIL_COUNT_EXT,
    eNotMergedResolveAttachmentReuse      = VK_SUBPASS_MERGE_STATUS_NOT_MERGED_RESOLVE_ATTACHMENT_REUSE_EXT,
    eNotMergedSingleSubpass               = VK_SUBPASS_MERGE_STATUS_NOT_MERGED_SINGLE_SUBPASS_EXT,
    eNotMergedUnspecified                 = VK_SUBPASS_MERGE_STATUS_NOT_MERGED_UNSPECIFIED_EXT
  };

  //=== VK_EXT_rasterization_order_attachment_access ===

  enum class PipelineColorBlendStateCreateFlagBits : VkPipelineColorBlendStateCreateFlags
  {
    eRasterizationOrderAttachmentAccessEXT = VK_PIPELINE_COLOR_BLEND_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_BIT_EXT,
    eRasterizationOrderAttachmentAccessARM = VK_PIPELINE_COLOR_BLEND_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_BIT_ARM
  };

  enum class PipelineDepthStencilStateCreateFlagBits : VkPipelineDepthStencilStateCreateFlags
  {
    eRasterizationOrderAttachmentDepthAccessEXT   = VK_PIPELINE_DEPTH_STENCIL_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_BIT_EXT,
    eRasterizationOrderAttachmentStencilAccessEXT = VK_PIPELINE_DEPTH_STENCIL_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_BIT_EXT,
    eRasterizationOrderAttachmentDepthAccessARM   = VK_PIPELINE_DEPTH_STENCIL_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_BIT_ARM,
    eRasterizationOrderAttachmentStencilAccessARM = VK_PIPELINE_DEPTH_STENCIL_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_BIT_ARM
  };

  //=== VK_NV_optical_flow ===

  enum class OpticalFlowUsageFlagBitsNV : VkOpticalFlowUsageFlagsNV
  {
    eUnknown    = VK_OPTICAL_FLOW_USAGE_UNKNOWN_NV,
    eInput      = VK_OPTICAL_FLOW_USAGE_INPUT_BIT_NV,
    eOutput     = VK_OPTICAL_FLOW_USAGE_OUTPUT_BIT_NV,
    eHint       = VK_OPTICAL_FLOW_USAGE_HINT_BIT_NV,
    eCost       = VK_OPTICAL_FLOW_USAGE_COST_BIT_NV,
    eGlobalFlow = VK_OPTICAL_FLOW_USAGE_GLOBAL_FLOW_BIT_NV
  };

  enum class OpticalFlowGridSizeFlagBitsNV : VkOpticalFlowGridSizeFlagsNV
  {
    eUnknown = VK_OPTICAL_FLOW_GRID_SIZE_UNKNOWN_NV,
    e1X1     = VK_OPTICAL_FLOW_GRID_SIZE_1X1_BIT_NV,
    e2X2     = VK_OPTICAL_FLOW_GRID_SIZE_2X2_BIT_NV,
    e4X4     = VK_OPTICAL_FLOW_GRID_SIZE_4X4_BIT_NV,
    e8X8     = VK_OPTICAL_FLOW_GRID_SIZE_8X8_BIT_NV
  };

  enum class OpticalFlowPerformanceLevelNV
  {
    eUnknown = VK_OPTICAL_FLOW_PERFORMANCE_LEVEL_UNKNOWN_NV,
    eSlow    = VK_OPTICAL_FLOW_PERFORMANCE_LEVEL_SLOW_NV,
    eMedium  = VK_OPTICAL_FLOW_PERFORMANCE_LEVEL_MEDIUM_NV,
    eFast    = VK_OPTICAL_FLOW_PERFORMANCE_LEVEL_FAST_NV
  };

  enum class OpticalFlowSessionBindingPointNV
  {
    eUnknown            = VK_OPTICAL_FLOW_SESSION_BINDING_POINT_UNKNOWN_NV,
    eInput              = VK_OPTICAL_FLOW_SESSION_BINDING_POINT_INPUT_NV,
    eReference          = VK_OPTICAL_FLOW_SESSION_BINDING_POINT_REFERENCE_NV,
    eHint               = VK_OPTICAL_FLOW_SESSION_BINDING_POINT_HINT_NV,
    eFlowVector         = VK_OPTICAL_FLOW_SESSION_BINDING_POINT_FLOW_VECTOR_NV,
    eBackwardFlowVector = VK_OPTICAL_FLOW_SESSION_BINDING_POINT_BACKWARD_FLOW_VECTOR_NV,
    eCost               = VK_OPTICAL_FLOW_SESSION_BINDING_POINT_COST_NV,
    eBackwardCost       = VK_OPTICAL_FLOW_SESSION_BINDING_POINT_BACKWARD_COST_NV,
    eGlobalFlow         = VK_OPTICAL_FLOW_SESSION_BINDING_POINT_GLOBAL_FLOW_NV
  };

  enum class OpticalFlowSessionCreateFlagBitsNV : VkOpticalFlowSessionCreateFlagsNV
  {
    eEnableHint       = VK_OPTICAL_FLOW_SESSION_CREATE_ENABLE_HINT_BIT_NV,
    eEnableCost       = VK_OPTICAL_FLOW_SESSION_CREATE_ENABLE_COST_BIT_NV,
    eEnableGlobalFlow = VK_OPTICAL_FLOW_SESSION_CREATE_ENABLE_GLOBAL_FLOW_BIT_NV,
    eAllowRegions     = VK_OPTICAL_FLOW_SESSION_CREATE_ALLOW_REGIONS_BIT_NV,
    eBothDirections   = VK_OPTICAL_FLOW_SESSION_CREATE_BOTH_DIRECTIONS_BIT_NV
  };

  enum class OpticalFlowExecuteFlagBitsNV : VkOpticalFlowExecuteFlagsNV
  {
    eDisableTemporalHints = VK_OPTICAL_FLOW_EXECUTE_DISABLE_TEMPORAL_HINTS_BIT_NV
  };

  template <typename T>
  struct IndexTypeValue
  {
  };

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
        VkFlags( FormatFeatureFlagBits::eStorageTexelBuffer ) | VkFlags( FormatFeatureFlagBits::eStorageTexelBufferAtomic ) |
        VkFlags( FormatFeatureFlagBits::eVertexBuffer ) | VkFlags( FormatFeatureFlagBits::eColorAttachment ) |
        VkFlags( FormatFeatureFlagBits::eColorAttachmentBlend ) | VkFlags( FormatFeatureFlagBits::eDepthStencilAttachment ) |
        VkFlags( FormatFeatureFlagBits::eBlitSrc ) | VkFlags( FormatFeatureFlagBits::eBlitDst ) | VkFlags( FormatFeatureFlagBits::eSampledImageFilterLinear ) |
        VkFlags( FormatFeatureFlagBits::eTransferSrc ) | VkFlags( FormatFeatureFlagBits::eTransferDst ) |
        VkFlags( FormatFeatureFlagBits::eMidpointChromaSamples ) | VkFlags( FormatFeatureFlagBits::eSampledImageYcbcrConversionLinearFilter ) |
        VkFlags( FormatFeatureFlagBits::eSampledImageYcbcrConversionSeparateReconstructionFilter ) |
        VkFlags( FormatFeatureFlagBits::eSampledImageYcbcrConversionChromaReconstructionExplicit ) |
        VkFlags( FormatFeatureFlagBits::eSampledImageYcbcrConversionChromaReconstructionExplicitForceable ) | VkFlags( FormatFeatureFlagBits::eDisjoint ) |
        VkFlags( FormatFeatureFlagBits::eCositedChromaSamples ) | VkFlags( FormatFeatureFlagBits::eSampledImageFilterMinmax )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
        | VkFlags( FormatFeatureFlagBits::eVideoDecodeOutputKHR ) | VkFlags( FormatFeatureFlagBits::eVideoDecodeDpbKHR )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
        | VkFlags( FormatFeatureFlagBits::eAccelerationStructureVertexBufferKHR ) | VkFlags( FormatFeatureFlagBits::eSampledImageFilterCubicEXT ) |
        VkFlags( FormatFeatureFlagBits::eFragmentDensityMapEXT ) | VkFlags( FormatFeatureFlagBits::eFragmentShadingRateAttachmentKHR )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
        | VkFlags( FormatFeatureFlagBits::eVideoEncodeInputKHR ) | VkFlags( FormatFeatureFlagBits::eVideoEncodeDpbKHR )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FormatFeatureFlags operator|( FormatFeatureFlagBits bit0, FormatFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return FormatFeatureFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FormatFeatureFlags operator&( FormatFeatureFlagBits bit0, FormatFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return FormatFeatureFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FormatFeatureFlags operator^( FormatFeatureFlagBits bit0, FormatFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return FormatFeatureFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FormatFeatureFlags operator~( FormatFeatureFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( FormatFeatureFlags( bits ) );
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
                 VkFlags( ImageCreateFlagBits::eSplitInstanceBindRegions ) | VkFlags( ImageCreateFlagBits::e2DArrayCompatible ) |
                 VkFlags( ImageCreateFlagBits::eBlockTexelViewCompatible ) | VkFlags( ImageCreateFlagBits::eExtendedUsage ) |
                 VkFlags( ImageCreateFlagBits::eProtected ) | VkFlags( ImageCreateFlagBits::eDisjoint ) | VkFlags( ImageCreateFlagBits::eCornerSampledNV ) |
                 VkFlags( ImageCreateFlagBits::eSampleLocationsCompatibleDepthEXT ) | VkFlags( ImageCreateFlagBits::eSubsampledEXT ) |
                 VkFlags( ImageCreateFlagBits::eMultisampledRenderToSingleSampledEXT ) | VkFlags( ImageCreateFlagBits::e2DViewCompatibleEXT ) |
                 VkFlags( ImageCreateFlagBits::eFragmentDensityMapOffsetQCOM )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageCreateFlags operator|( ImageCreateFlagBits bit0, ImageCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageCreateFlags operator&( ImageCreateFlagBits bit0, ImageCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageCreateFlags operator^( ImageCreateFlagBits bit0, ImageCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageCreateFlags operator~( ImageCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ImageCreateFlags( bits ) );
  }

  using ImageUsageFlags = Flags<ImageUsageFlagBits>;

  template <>
  struct FlagTraits<ImageUsageFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ImageUsageFlagBits::eTransferSrc ) | VkFlags( ImageUsageFlagBits::eTransferDst ) | VkFlags( ImageUsageFlagBits::eSampled ) |
                 VkFlags( ImageUsageFlagBits::eStorage ) | VkFlags( ImageUsageFlagBits::eColorAttachment ) |
                 VkFlags( ImageUsageFlagBits::eDepthStencilAttachment ) | VkFlags( ImageUsageFlagBits::eTransientAttachment ) |
                 VkFlags( ImageUsageFlagBits::eInputAttachment )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
                 | VkFlags( ImageUsageFlagBits::eVideoDecodeDstKHR ) | VkFlags( ImageUsageFlagBits::eVideoDecodeSrcKHR ) |
                 VkFlags( ImageUsageFlagBits::eVideoDecodeDpbKHR )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
                 | VkFlags( ImageUsageFlagBits::eFragmentDensityMapEXT ) | VkFlags( ImageUsageFlagBits::eFragmentShadingRateAttachmentKHR )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
                 | VkFlags( ImageUsageFlagBits::eVideoEncodeDstKHR ) | VkFlags( ImageUsageFlagBits::eVideoEncodeSrcKHR ) |
                 VkFlags( ImageUsageFlagBits::eVideoEncodeDpbKHR )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
                 | VkFlags( ImageUsageFlagBits::eAttachmentFeedbackLoopEXT ) | VkFlags( ImageUsageFlagBits::eInvocationMaskHUAWEI ) |
                 VkFlags( ImageUsageFlagBits::eSampleWeightQCOM ) | VkFlags( ImageUsageFlagBits::eSampleBlockMatchQCOM )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageUsageFlags operator|( ImageUsageFlagBits bit0, ImageUsageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageUsageFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageUsageFlags operator&( ImageUsageFlagBits bit0, ImageUsageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageUsageFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageUsageFlags operator^( ImageUsageFlagBits bit0, ImageUsageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageUsageFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageUsageFlags operator~( ImageUsageFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ImageUsageFlags( bits ) );
  }

  using InstanceCreateFlags = Flags<InstanceCreateFlagBits>;

  template <>
  struct FlagTraits<InstanceCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( InstanceCreateFlagBits::eEnumeratePortabilityKHR )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR InstanceCreateFlags operator|( InstanceCreateFlagBits bit0, InstanceCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return InstanceCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR InstanceCreateFlags operator&( InstanceCreateFlagBits bit0, InstanceCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return InstanceCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR InstanceCreateFlags operator^( InstanceCreateFlagBits bit0, InstanceCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return InstanceCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR InstanceCreateFlags operator~( InstanceCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( InstanceCreateFlags( bits ) );
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

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MemoryHeapFlags operator|( MemoryHeapFlagBits bit0, MemoryHeapFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return MemoryHeapFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MemoryHeapFlags operator&( MemoryHeapFlagBits bit0, MemoryHeapFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return MemoryHeapFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MemoryHeapFlags operator^( MemoryHeapFlagBits bit0, MemoryHeapFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return MemoryHeapFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MemoryHeapFlags operator~( MemoryHeapFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( MemoryHeapFlags( bits ) );
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
                 VkFlags( MemoryPropertyFlagBits::eDeviceCoherentAMD ) | VkFlags( MemoryPropertyFlagBits::eDeviceUncachedAMD ) |
                 VkFlags( MemoryPropertyFlagBits::eRdmaCapableNV )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MemoryPropertyFlags operator|( MemoryPropertyFlagBits bit0, MemoryPropertyFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return MemoryPropertyFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MemoryPropertyFlags operator&( MemoryPropertyFlagBits bit0, MemoryPropertyFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return MemoryPropertyFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MemoryPropertyFlags operator^( MemoryPropertyFlagBits bit0, MemoryPropertyFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return MemoryPropertyFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MemoryPropertyFlags operator~( MemoryPropertyFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( MemoryPropertyFlags( bits ) );
  }

  using QueueFlags = Flags<QueueFlagBits>;

  template <>
  struct FlagTraits<QueueFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( QueueFlagBits::eGraphics ) | VkFlags( QueueFlagBits::eCompute ) | VkFlags( QueueFlagBits::eTransfer ) |
                 VkFlags( QueueFlagBits::eSparseBinding ) | VkFlags( QueueFlagBits::eProtected )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
                 | VkFlags( QueueFlagBits::eVideoDecodeKHR ) | VkFlags( QueueFlagBits::eVideoEncodeKHR )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
                 | VkFlags( QueueFlagBits::eOpticalFlowNV )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueueFlags operator|( QueueFlagBits bit0, QueueFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return QueueFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueueFlags operator&( QueueFlagBits bit0, QueueFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return QueueFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueueFlags operator^( QueueFlagBits bit0, QueueFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return QueueFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueueFlags operator~( QueueFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( QueueFlags( bits ) );
  }

  using SampleCountFlags = Flags<SampleCountFlagBits>;

  template <>
  struct FlagTraits<SampleCountFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( SampleCountFlagBits::e1 ) | VkFlags( SampleCountFlagBits::e2 ) | VkFlags( SampleCountFlagBits::e4 ) |
                 VkFlags( SampleCountFlagBits::e8 ) | VkFlags( SampleCountFlagBits::e16 ) | VkFlags( SampleCountFlagBits::e32 ) |
                 VkFlags( SampleCountFlagBits::e64 )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SampleCountFlags operator|( SampleCountFlagBits bit0, SampleCountFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SampleCountFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SampleCountFlags operator&( SampleCountFlagBits bit0, SampleCountFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SampleCountFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SampleCountFlags operator^( SampleCountFlagBits bit0, SampleCountFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SampleCountFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SampleCountFlags operator~( SampleCountFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( SampleCountFlags( bits ) );
  }

  using DeviceCreateFlags = Flags<DeviceCreateFlagBits>;

  using DeviceQueueCreateFlags = Flags<DeviceQueueCreateFlagBits>;

  template <>
  struct FlagTraits<DeviceQueueCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( DeviceQueueCreateFlagBits::eProtected )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceQueueCreateFlags operator|( DeviceQueueCreateFlagBits bit0, DeviceQueueCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DeviceQueueCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceQueueCreateFlags operator&( DeviceQueueCreateFlagBits bit0, DeviceQueueCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DeviceQueueCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceQueueCreateFlags operator^( DeviceQueueCreateFlagBits bit0, DeviceQueueCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DeviceQueueCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceQueueCreateFlags operator~( DeviceQueueCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( DeviceQueueCreateFlags( bits ) );
  }

  using PipelineStageFlags = Flags<PipelineStageFlagBits>;

  template <>
  struct FlagTraits<PipelineStageFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( PipelineStageFlagBits::eTopOfPipe ) | VkFlags( PipelineStageFlagBits::eDrawIndirect ) |
                 VkFlags( PipelineStageFlagBits::eVertexInput ) | VkFlags( PipelineStageFlagBits::eVertexShader ) |
                 VkFlags( PipelineStageFlagBits::eTessellationControlShader ) | VkFlags( PipelineStageFlagBits::eTessellationEvaluationShader ) |
                 VkFlags( PipelineStageFlagBits::eGeometryShader ) | VkFlags( PipelineStageFlagBits::eFragmentShader ) |
                 VkFlags( PipelineStageFlagBits::eEarlyFragmentTests ) | VkFlags( PipelineStageFlagBits::eLateFragmentTests ) |
                 VkFlags( PipelineStageFlagBits::eColorAttachmentOutput ) | VkFlags( PipelineStageFlagBits::eComputeShader ) |
                 VkFlags( PipelineStageFlagBits::eTransfer ) | VkFlags( PipelineStageFlagBits::eBottomOfPipe ) | VkFlags( PipelineStageFlagBits::eHost ) |
                 VkFlags( PipelineStageFlagBits::eAllGraphics ) | VkFlags( PipelineStageFlagBits::eAllCommands ) | VkFlags( PipelineStageFlagBits::eNone ) |
                 VkFlags( PipelineStageFlagBits::eTransformFeedbackEXT ) | VkFlags( PipelineStageFlagBits::eConditionalRenderingEXT ) |
                 VkFlags( PipelineStageFlagBits::eAccelerationStructureBuildKHR ) | VkFlags( PipelineStageFlagBits::eRayTracingShaderKHR ) |
                 VkFlags( PipelineStageFlagBits::eFragmentDensityProcessEXT ) | VkFlags( PipelineStageFlagBits::eFragmentShadingRateAttachmentKHR ) |
                 VkFlags( PipelineStageFlagBits::eCommandPreprocessNV ) | VkFlags( PipelineStageFlagBits::eTaskShaderEXT ) |
                 VkFlags( PipelineStageFlagBits::eMeshShaderEXT )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineStageFlags operator|( PipelineStageFlagBits bit0, PipelineStageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineStageFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineStageFlags operator&( PipelineStageFlagBits bit0, PipelineStageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineStageFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineStageFlags operator^( PipelineStageFlagBits bit0, PipelineStageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineStageFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineStageFlags operator~( PipelineStageFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( PipelineStageFlags( bits ) );
  }

  using MemoryMapFlags = Flags<MemoryMapFlagBits>;

  using ImageAspectFlags = Flags<ImageAspectFlagBits>;

  template <>
  struct FlagTraits<ImageAspectFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ImageAspectFlagBits::eColor ) | VkFlags( ImageAspectFlagBits::eDepth ) | VkFlags( ImageAspectFlagBits::eStencil ) |
                 VkFlags( ImageAspectFlagBits::eMetadata ) | VkFlags( ImageAspectFlagBits::ePlane0 ) | VkFlags( ImageAspectFlagBits::ePlane1 ) |
                 VkFlags( ImageAspectFlagBits::ePlane2 ) | VkFlags( ImageAspectFlagBits::eNone ) | VkFlags( ImageAspectFlagBits::eMemoryPlane0EXT ) |
                 VkFlags( ImageAspectFlagBits::eMemoryPlane1EXT ) | VkFlags( ImageAspectFlagBits::eMemoryPlane2EXT ) |
                 VkFlags( ImageAspectFlagBits::eMemoryPlane3EXT )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageAspectFlags operator|( ImageAspectFlagBits bit0, ImageAspectFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageAspectFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageAspectFlags operator&( ImageAspectFlagBits bit0, ImageAspectFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageAspectFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageAspectFlags operator^( ImageAspectFlagBits bit0, ImageAspectFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageAspectFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageAspectFlags operator~( ImageAspectFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ImageAspectFlags( bits ) );
  }

  using SparseImageFormatFlags = Flags<SparseImageFormatFlagBits>;

  template <>
  struct FlagTraits<SparseImageFormatFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( SparseImageFormatFlagBits::eSingleMiptail ) | VkFlags( SparseImageFormatFlagBits::eAlignedMipSize ) |
                 VkFlags( SparseImageFormatFlagBits::eNonstandardBlockSize )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SparseImageFormatFlags operator|( SparseImageFormatFlagBits bit0, SparseImageFormatFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SparseImageFormatFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SparseImageFormatFlags operator&( SparseImageFormatFlagBits bit0, SparseImageFormatFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SparseImageFormatFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SparseImageFormatFlags operator^( SparseImageFormatFlagBits bit0, SparseImageFormatFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SparseImageFormatFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SparseImageFormatFlags operator~( SparseImageFormatFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( SparseImageFormatFlags( bits ) );
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

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SparseMemoryBindFlags operator|( SparseMemoryBindFlagBits bit0, SparseMemoryBindFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SparseMemoryBindFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SparseMemoryBindFlags operator&( SparseMemoryBindFlagBits bit0, SparseMemoryBindFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SparseMemoryBindFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SparseMemoryBindFlags operator^( SparseMemoryBindFlagBits bit0, SparseMemoryBindFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SparseMemoryBindFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SparseMemoryBindFlags operator~( SparseMemoryBindFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( SparseMemoryBindFlags( bits ) );
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

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FenceCreateFlags operator|( FenceCreateFlagBits bit0, FenceCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return FenceCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FenceCreateFlags operator&( FenceCreateFlagBits bit0, FenceCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return FenceCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FenceCreateFlags operator^( FenceCreateFlagBits bit0, FenceCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return FenceCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FenceCreateFlags operator~( FenceCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( FenceCreateFlags( bits ) );
  }

  using SemaphoreCreateFlags = Flags<SemaphoreCreateFlagBits>;

  using EventCreateFlags = Flags<EventCreateFlagBits>;

  template <>
  struct FlagTraits<EventCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( EventCreateFlagBits::eDeviceOnly )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR EventCreateFlags operator|( EventCreateFlagBits bit0, EventCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return EventCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR EventCreateFlags operator&( EventCreateFlagBits bit0, EventCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return EventCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR EventCreateFlags operator^( EventCreateFlagBits bit0, EventCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return EventCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR EventCreateFlags operator~( EventCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( EventCreateFlags( bits ) );
  }

  using QueryPipelineStatisticFlags = Flags<QueryPipelineStatisticFlagBits>;

  template <>
  struct FlagTraits<QueryPipelineStatisticFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( QueryPipelineStatisticFlagBits::eInputAssemblyVertices ) | VkFlags( QueryPipelineStatisticFlagBits::eInputAssemblyPrimitives ) |
                 VkFlags( QueryPipelineStatisticFlagBits::eVertexShaderInvocations ) | VkFlags( QueryPipelineStatisticFlagBits::eGeometryShaderInvocations ) |
                 VkFlags( QueryPipelineStatisticFlagBits::eGeometryShaderPrimitives ) | VkFlags( QueryPipelineStatisticFlagBits::eClippingInvocations ) |
                 VkFlags( QueryPipelineStatisticFlagBits::eClippingPrimitives ) | VkFlags( QueryPipelineStatisticFlagBits::eFragmentShaderInvocations ) |
                 VkFlags( QueryPipelineStatisticFlagBits::eTessellationControlShaderPatches ) |
                 VkFlags( QueryPipelineStatisticFlagBits::eTessellationEvaluationShaderInvocations ) |
                 VkFlags( QueryPipelineStatisticFlagBits::eComputeShaderInvocations ) | VkFlags( QueryPipelineStatisticFlagBits::eTaskShaderInvocationsEXT ) |
                 VkFlags( QueryPipelineStatisticFlagBits::eMeshShaderInvocationsEXT )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueryPipelineStatisticFlags operator|( QueryPipelineStatisticFlagBits bit0,
                                                                                QueryPipelineStatisticFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return QueryPipelineStatisticFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueryPipelineStatisticFlags operator&( QueryPipelineStatisticFlagBits bit0,
                                                                                QueryPipelineStatisticFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return QueryPipelineStatisticFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueryPipelineStatisticFlags operator^( QueryPipelineStatisticFlagBits bit0,
                                                                                QueryPipelineStatisticFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return QueryPipelineStatisticFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueryPipelineStatisticFlags operator~( QueryPipelineStatisticFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( QueryPipelineStatisticFlags( bits ) );
  }

  using QueryPoolCreateFlags = Flags<QueryPoolCreateFlagBits>;

  using QueryResultFlags = Flags<QueryResultFlagBits>;

  template <>
  struct FlagTraits<QueryResultFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( QueryResultFlagBits::e64 ) | VkFlags( QueryResultFlagBits::eWait ) | VkFlags( QueryResultFlagBits::eWithAvailability ) |
                 VkFlags( QueryResultFlagBits::ePartial )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
                 | VkFlags( QueryResultFlagBits::eWithStatusKHR )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueryResultFlags operator|( QueryResultFlagBits bit0, QueryResultFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return QueryResultFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueryResultFlags operator&( QueryResultFlagBits bit0, QueryResultFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return QueryResultFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueryResultFlags operator^( QueryResultFlagBits bit0, QueryResultFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return QueryResultFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueryResultFlags operator~( QueryResultFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( QueryResultFlags( bits ) );
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

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BufferCreateFlags operator|( BufferCreateFlagBits bit0, BufferCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return BufferCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BufferCreateFlags operator&( BufferCreateFlagBits bit0, BufferCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return BufferCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BufferCreateFlags operator^( BufferCreateFlagBits bit0, BufferCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return BufferCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BufferCreateFlags operator~( BufferCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( BufferCreateFlags( bits ) );
  }

  using BufferUsageFlags = Flags<BufferUsageFlagBits>;

  template <>
  struct FlagTraits<BufferUsageFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( BufferUsageFlagBits::eTransferSrc ) | VkFlags( BufferUsageFlagBits::eTransferDst ) |
                 VkFlags( BufferUsageFlagBits::eUniformTexelBuffer ) | VkFlags( BufferUsageFlagBits::eStorageTexelBuffer ) |
                 VkFlags( BufferUsageFlagBits::eUniformBuffer ) | VkFlags( BufferUsageFlagBits::eStorageBuffer ) |
                 VkFlags( BufferUsageFlagBits::eIndexBuffer ) | VkFlags( BufferUsageFlagBits::eVertexBuffer ) |
                 VkFlags( BufferUsageFlagBits::eIndirectBuffer ) | VkFlags( BufferUsageFlagBits::eShaderDeviceAddress )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
                 | VkFlags( BufferUsageFlagBits::eVideoDecodeSrcKHR ) | VkFlags( BufferUsageFlagBits::eVideoDecodeDstKHR )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
                 | VkFlags( BufferUsageFlagBits::eTransformFeedbackBufferEXT ) | VkFlags( BufferUsageFlagBits::eTransformFeedbackCounterBufferEXT ) |
                 VkFlags( BufferUsageFlagBits::eConditionalRenderingEXT ) | VkFlags( BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR ) |
                 VkFlags( BufferUsageFlagBits::eAccelerationStructureStorageKHR ) | VkFlags( BufferUsageFlagBits::eShaderBindingTableKHR )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
                 | VkFlags( BufferUsageFlagBits::eVideoEncodeDstKHR ) | VkFlags( BufferUsageFlagBits::eVideoEncodeSrcKHR )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
                 | VkFlags( BufferUsageFlagBits::eMicromapBuildInputReadOnlyEXT ) | VkFlags( BufferUsageFlagBits::eMicromapStorageEXT )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BufferUsageFlags operator|( BufferUsageFlagBits bit0, BufferUsageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return BufferUsageFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BufferUsageFlags operator&( BufferUsageFlagBits bit0, BufferUsageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return BufferUsageFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BufferUsageFlags operator^( BufferUsageFlagBits bit0, BufferUsageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return BufferUsageFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BufferUsageFlags operator~( BufferUsageFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( BufferUsageFlags( bits ) );
  }

  using BufferViewCreateFlags = Flags<BufferViewCreateFlagBits>;

  using ImageViewCreateFlags = Flags<ImageViewCreateFlagBits>;

  template <>
  struct FlagTraits<ImageViewCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ImageViewCreateFlagBits::eFragmentDensityMapDynamicEXT ) | VkFlags( ImageViewCreateFlagBits::eFragmentDensityMapDeferredEXT )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageViewCreateFlags operator|( ImageViewCreateFlagBits bit0, ImageViewCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageViewCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageViewCreateFlags operator&( ImageViewCreateFlagBits bit0, ImageViewCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageViewCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageViewCreateFlags operator^( ImageViewCreateFlagBits bit0, ImageViewCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageViewCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageViewCreateFlags operator~( ImageViewCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ImageViewCreateFlags( bits ) );
  }

  using ShaderModuleCreateFlags = Flags<ShaderModuleCreateFlagBits>;

  using PipelineCacheCreateFlags = Flags<PipelineCacheCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineCacheCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( PipelineCacheCreateFlagBits::eExternallySynchronized )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineCacheCreateFlags operator|( PipelineCacheCreateFlagBits bit0,
                                                                             PipelineCacheCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineCacheCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineCacheCreateFlags operator&( PipelineCacheCreateFlagBits bit0,
                                                                             PipelineCacheCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineCacheCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineCacheCreateFlags operator^( PipelineCacheCreateFlagBits bit0,
                                                                             PipelineCacheCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineCacheCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineCacheCreateFlags operator~( PipelineCacheCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( PipelineCacheCreateFlags( bits ) );
  }

  using ColorComponentFlags = Flags<ColorComponentFlagBits>;

  template <>
  struct FlagTraits<ColorComponentFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ColorComponentFlagBits::eR ) | VkFlags( ColorComponentFlagBits::eG ) | VkFlags( ColorComponentFlagBits::eB ) |
                 VkFlags( ColorComponentFlagBits::eA )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ColorComponentFlags operator|( ColorComponentFlagBits bit0, ColorComponentFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ColorComponentFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ColorComponentFlags operator&( ColorComponentFlagBits bit0, ColorComponentFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ColorComponentFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ColorComponentFlags operator^( ColorComponentFlagBits bit0, ColorComponentFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ColorComponentFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ColorComponentFlags operator~( ColorComponentFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ColorComponentFlags( bits ) );
  }

  using CullModeFlags = Flags<CullModeFlagBits>;

  template <>
  struct FlagTraits<CullModeFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( CullModeFlagBits::eNone ) | VkFlags( CullModeFlagBits::eFront ) | VkFlags( CullModeFlagBits::eBack ) |
                 VkFlags( CullModeFlagBits::eFrontAndBack )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CullModeFlags operator|( CullModeFlagBits bit0, CullModeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CullModeFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CullModeFlags operator&( CullModeFlagBits bit0, CullModeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CullModeFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CullModeFlags operator^( CullModeFlagBits bit0, CullModeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CullModeFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CullModeFlags operator~( CullModeFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( CullModeFlags( bits ) );
  }

  using PipelineColorBlendStateCreateFlags = Flags<PipelineColorBlendStateCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineColorBlendStateCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( PipelineColorBlendStateCreateFlagBits::eRasterizationOrderAttachmentAccessEXT )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineColorBlendStateCreateFlags operator|( PipelineColorBlendStateCreateFlagBits bit0,
                                                                                       PipelineColorBlendStateCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineColorBlendStateCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineColorBlendStateCreateFlags operator&( PipelineColorBlendStateCreateFlagBits bit0,
                                                                                       PipelineColorBlendStateCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineColorBlendStateCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineColorBlendStateCreateFlags operator^( PipelineColorBlendStateCreateFlagBits bit0,
                                                                                       PipelineColorBlendStateCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineColorBlendStateCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineColorBlendStateCreateFlags operator~( PipelineColorBlendStateCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( PipelineColorBlendStateCreateFlags( bits ) );
  }

  using PipelineCreateFlags = Flags<PipelineCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( PipelineCreateFlagBits::eDisableOptimization ) | VkFlags( PipelineCreateFlagBits::eAllowDerivatives ) |
                 VkFlags( PipelineCreateFlagBits::eDerivative ) | VkFlags( PipelineCreateFlagBits::eViewIndexFromDeviceIndex ) |
                 VkFlags( PipelineCreateFlagBits::eDispatchBase ) | VkFlags( PipelineCreateFlagBits::eFailOnPipelineCompileRequired ) |
                 VkFlags( PipelineCreateFlagBits::eEarlyReturnOnFailure ) | VkFlags( PipelineCreateFlagBits::eRenderingFragmentShadingRateAttachmentKHR ) |
                 VkFlags( PipelineCreateFlagBits::eRenderingFragmentDensityMapAttachmentEXT ) |
                 VkFlags( PipelineCreateFlagBits::eRayTracingNoNullAnyHitShadersKHR ) |
                 VkFlags( PipelineCreateFlagBits::eRayTracingNoNullClosestHitShadersKHR ) | VkFlags( PipelineCreateFlagBits::eRayTracingNoNullMissShadersKHR ) |
                 VkFlags( PipelineCreateFlagBits::eRayTracingNoNullIntersectionShadersKHR ) | VkFlags( PipelineCreateFlagBits::eRayTracingSkipTrianglesKHR ) |
                 VkFlags( PipelineCreateFlagBits::eRayTracingSkipAabbsKHR ) | VkFlags( PipelineCreateFlagBits::eRayTracingShaderGroupHandleCaptureReplayKHR ) |
                 VkFlags( PipelineCreateFlagBits::eDeferCompileNV ) | VkFlags( PipelineCreateFlagBits::eCaptureStatisticsKHR ) |
                 VkFlags( PipelineCreateFlagBits::eCaptureInternalRepresentationsKHR ) | VkFlags( PipelineCreateFlagBits::eIndirectBindableNV ) |
                 VkFlags( PipelineCreateFlagBits::eLibraryKHR ) | VkFlags( PipelineCreateFlagBits::eRetainLinkTimeOptimizationInfoEXT ) |
                 VkFlags( PipelineCreateFlagBits::eLinkTimeOptimizationEXT ) | VkFlags( PipelineCreateFlagBits::eRayTracingAllowMotionNV ) |
                 VkFlags( PipelineCreateFlagBits::eColorAttachmentFeedbackLoopEXT ) |
                 VkFlags( PipelineCreateFlagBits::eDepthStencilAttachmentFeedbackLoopEXT ) | VkFlags( PipelineCreateFlagBits::eRayTracingOpacityMicromapEXT ) |
                 VkFlags( PipelineCreateFlagBits::eNoProtectedAccessEXT ) | VkFlags( PipelineCreateFlagBits::eProtectedAccessOnlyEXT )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineCreateFlags operator|( PipelineCreateFlagBits bit0, PipelineCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineCreateFlags operator&( PipelineCreateFlagBits bit0, PipelineCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineCreateFlags operator^( PipelineCreateFlagBits bit0, PipelineCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineCreateFlags operator~( PipelineCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( PipelineCreateFlags( bits ) );
  }

  using PipelineDepthStencilStateCreateFlags = Flags<PipelineDepthStencilStateCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineDepthStencilStateCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( PipelineDepthStencilStateCreateFlagBits::eRasterizationOrderAttachmentDepthAccessEXT ) |
                 VkFlags( PipelineDepthStencilStateCreateFlagBits::eRasterizationOrderAttachmentStencilAccessEXT )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineDepthStencilStateCreateFlags operator|( PipelineDepthStencilStateCreateFlagBits bit0,
                                                                                         PipelineDepthStencilStateCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineDepthStencilStateCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineDepthStencilStateCreateFlags operator&( PipelineDepthStencilStateCreateFlagBits bit0,
                                                                                         PipelineDepthStencilStateCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineDepthStencilStateCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineDepthStencilStateCreateFlags operator^( PipelineDepthStencilStateCreateFlagBits bit0,
                                                                                         PipelineDepthStencilStateCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineDepthStencilStateCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineDepthStencilStateCreateFlags operator~( PipelineDepthStencilStateCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( PipelineDepthStencilStateCreateFlags( bits ) );
  }

  using PipelineDynamicStateCreateFlags = Flags<PipelineDynamicStateCreateFlagBits>;

  using PipelineInputAssemblyStateCreateFlags = Flags<PipelineInputAssemblyStateCreateFlagBits>;

  using PipelineLayoutCreateFlags = Flags<PipelineLayoutCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineLayoutCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( PipelineLayoutCreateFlagBits::eIndependentSetsEXT )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineLayoutCreateFlags operator|( PipelineLayoutCreateFlagBits bit0,
                                                                              PipelineLayoutCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineLayoutCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineLayoutCreateFlags operator&( PipelineLayoutCreateFlagBits bit0,
                                                                              PipelineLayoutCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineLayoutCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineLayoutCreateFlags operator^( PipelineLayoutCreateFlagBits bit0,
                                                                              PipelineLayoutCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineLayoutCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineLayoutCreateFlags operator~( PipelineLayoutCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( PipelineLayoutCreateFlags( bits ) );
  }

  using PipelineMultisampleStateCreateFlags = Flags<PipelineMultisampleStateCreateFlagBits>;

  using PipelineRasterizationStateCreateFlags = Flags<PipelineRasterizationStateCreateFlagBits>;

  using PipelineShaderStageCreateFlags = Flags<PipelineShaderStageCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineShaderStageCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( PipelineShaderStageCreateFlagBits::eAllowVaryingSubgroupSize ) | VkFlags( PipelineShaderStageCreateFlagBits::eRequireFullSubgroups )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineShaderStageCreateFlags operator|( PipelineShaderStageCreateFlagBits bit0,
                                                                                   PipelineShaderStageCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineShaderStageCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineShaderStageCreateFlags operator&( PipelineShaderStageCreateFlagBits bit0,
                                                                                   PipelineShaderStageCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineShaderStageCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineShaderStageCreateFlags operator^( PipelineShaderStageCreateFlagBits bit0,
                                                                                   PipelineShaderStageCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineShaderStageCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineShaderStageCreateFlags operator~( PipelineShaderStageCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( PipelineShaderStageCreateFlags( bits ) );
  }

  using PipelineTessellationStateCreateFlags = Flags<PipelineTessellationStateCreateFlagBits>;

  using PipelineVertexInputStateCreateFlags = Flags<PipelineVertexInputStateCreateFlagBits>;

  using PipelineViewportStateCreateFlags = Flags<PipelineViewportStateCreateFlagBits>;

  using ShaderStageFlags = Flags<ShaderStageFlagBits>;

  template <>
  struct FlagTraits<ShaderStageFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ShaderStageFlagBits::eVertex ) | VkFlags( ShaderStageFlagBits::eTessellationControl ) |
                 VkFlags( ShaderStageFlagBits::eTessellationEvaluation ) | VkFlags( ShaderStageFlagBits::eGeometry ) |
                 VkFlags( ShaderStageFlagBits::eFragment ) | VkFlags( ShaderStageFlagBits::eCompute ) | VkFlags( ShaderStageFlagBits::eAllGraphics ) |
                 VkFlags( ShaderStageFlagBits::eAll ) | VkFlags( ShaderStageFlagBits::eRaygenKHR ) | VkFlags( ShaderStageFlagBits::eAnyHitKHR ) |
                 VkFlags( ShaderStageFlagBits::eClosestHitKHR ) | VkFlags( ShaderStageFlagBits::eMissKHR ) | VkFlags( ShaderStageFlagBits::eIntersectionKHR ) |
                 VkFlags( ShaderStageFlagBits::eCallableKHR ) | VkFlags( ShaderStageFlagBits::eTaskEXT ) | VkFlags( ShaderStageFlagBits::eMeshEXT ) |
                 VkFlags( ShaderStageFlagBits::eSubpassShadingHUAWEI )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ShaderStageFlags operator|( ShaderStageFlagBits bit0, ShaderStageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ShaderStageFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ShaderStageFlags operator&( ShaderStageFlagBits bit0, ShaderStageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ShaderStageFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ShaderStageFlags operator^( ShaderStageFlagBits bit0, ShaderStageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ShaderStageFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ShaderStageFlags operator~( ShaderStageFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ShaderStageFlags( bits ) );
  }

  using SamplerCreateFlags = Flags<SamplerCreateFlagBits>;

  template <>
  struct FlagTraits<SamplerCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( SamplerCreateFlagBits::eSubsampledEXT ) | VkFlags( SamplerCreateFlagBits::eSubsampledCoarseReconstructionEXT ) |
                 VkFlags( SamplerCreateFlagBits::eNonSeamlessCubeMapEXT ) | VkFlags( SamplerCreateFlagBits::eImageProcessingQCOM )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SamplerCreateFlags operator|( SamplerCreateFlagBits bit0, SamplerCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SamplerCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SamplerCreateFlags operator&( SamplerCreateFlagBits bit0, SamplerCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SamplerCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SamplerCreateFlags operator^( SamplerCreateFlagBits bit0, SamplerCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SamplerCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SamplerCreateFlags operator~( SamplerCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( SamplerCreateFlags( bits ) );
  }

  using DescriptorPoolCreateFlags = Flags<DescriptorPoolCreateFlagBits>;

  template <>
  struct FlagTraits<DescriptorPoolCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( DescriptorPoolCreateFlagBits::eFreeDescriptorSet ) | VkFlags( DescriptorPoolCreateFlagBits::eUpdateAfterBind ) |
                 VkFlags( DescriptorPoolCreateFlagBits::eHostOnlyEXT )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DescriptorPoolCreateFlags operator|( DescriptorPoolCreateFlagBits bit0,
                                                                              DescriptorPoolCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DescriptorPoolCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DescriptorPoolCreateFlags operator&( DescriptorPoolCreateFlagBits bit0,
                                                                              DescriptorPoolCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DescriptorPoolCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DescriptorPoolCreateFlags operator^( DescriptorPoolCreateFlagBits bit0,
                                                                              DescriptorPoolCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DescriptorPoolCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DescriptorPoolCreateFlags operator~( DescriptorPoolCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( DescriptorPoolCreateFlags( bits ) );
  }

  using DescriptorPoolResetFlags = Flags<DescriptorPoolResetFlagBits>;

  using DescriptorSetLayoutCreateFlags = Flags<DescriptorSetLayoutCreateFlagBits>;

  template <>
  struct FlagTraits<DescriptorSetLayoutCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool ) | VkFlags( DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR ) |
                 VkFlags( DescriptorSetLayoutCreateFlagBits::eHostOnlyPoolEXT )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DescriptorSetLayoutCreateFlags operator|( DescriptorSetLayoutCreateFlagBits bit0,
                                                                                   DescriptorSetLayoutCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DescriptorSetLayoutCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DescriptorSetLayoutCreateFlags operator&( DescriptorSetLayoutCreateFlagBits bit0,
                                                                                   DescriptorSetLayoutCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DescriptorSetLayoutCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DescriptorSetLayoutCreateFlags operator^( DescriptorSetLayoutCreateFlagBits bit0,
                                                                                   DescriptorSetLayoutCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DescriptorSetLayoutCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DescriptorSetLayoutCreateFlags operator~( DescriptorSetLayoutCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( DescriptorSetLayoutCreateFlags( bits ) );
  }

  using AccessFlags = Flags<AccessFlagBits>;

  template <>
  struct FlagTraits<AccessFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( AccessFlagBits::eIndirectCommandRead ) | VkFlags( AccessFlagBits::eIndexRead ) | VkFlags( AccessFlagBits::eVertexAttributeRead ) |
                 VkFlags( AccessFlagBits::eUniformRead ) | VkFlags( AccessFlagBits::eInputAttachmentRead ) | VkFlags( AccessFlagBits::eShaderRead ) |
                 VkFlags( AccessFlagBits::eShaderWrite ) | VkFlags( AccessFlagBits::eColorAttachmentRead ) | VkFlags( AccessFlagBits::eColorAttachmentWrite ) |
                 VkFlags( AccessFlagBits::eDepthStencilAttachmentRead ) | VkFlags( AccessFlagBits::eDepthStencilAttachmentWrite ) |
                 VkFlags( AccessFlagBits::eTransferRead ) | VkFlags( AccessFlagBits::eTransferWrite ) | VkFlags( AccessFlagBits::eHostRead ) |
                 VkFlags( AccessFlagBits::eHostWrite ) | VkFlags( AccessFlagBits::eMemoryRead ) | VkFlags( AccessFlagBits::eMemoryWrite ) |
                 VkFlags( AccessFlagBits::eNone ) | VkFlags( AccessFlagBits::eTransformFeedbackWriteEXT ) |
                 VkFlags( AccessFlagBits::eTransformFeedbackCounterReadEXT ) | VkFlags( AccessFlagBits::eTransformFeedbackCounterWriteEXT ) |
                 VkFlags( AccessFlagBits::eConditionalRenderingReadEXT ) | VkFlags( AccessFlagBits::eColorAttachmentReadNoncoherentEXT ) |
                 VkFlags( AccessFlagBits::eAccelerationStructureReadKHR ) | VkFlags( AccessFlagBits::eAccelerationStructureWriteKHR ) |
                 VkFlags( AccessFlagBits::eFragmentDensityMapReadEXT ) | VkFlags( AccessFlagBits::eFragmentShadingRateAttachmentReadKHR ) |
                 VkFlags( AccessFlagBits::eCommandPreprocessReadNV ) | VkFlags( AccessFlagBits::eCommandPreprocessWriteNV )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AccessFlags operator|( AccessFlagBits bit0, AccessFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return AccessFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AccessFlags operator&( AccessFlagBits bit0, AccessFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return AccessFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AccessFlags operator^( AccessFlagBits bit0, AccessFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return AccessFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AccessFlags operator~( AccessFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( AccessFlags( bits ) );
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

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AttachmentDescriptionFlags operator|( AttachmentDescriptionFlagBits bit0,
                                                                               AttachmentDescriptionFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return AttachmentDescriptionFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AttachmentDescriptionFlags operator&( AttachmentDescriptionFlagBits bit0,
                                                                               AttachmentDescriptionFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return AttachmentDescriptionFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AttachmentDescriptionFlags operator^( AttachmentDescriptionFlagBits bit0,
                                                                               AttachmentDescriptionFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return AttachmentDescriptionFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AttachmentDescriptionFlags operator~( AttachmentDescriptionFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( AttachmentDescriptionFlags( bits ) );
  }

  using DependencyFlags = Flags<DependencyFlagBits>;

  template <>
  struct FlagTraits<DependencyFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( DependencyFlagBits::eByRegion ) | VkFlags( DependencyFlagBits::eDeviceGroup ) | VkFlags( DependencyFlagBits::eViewLocal ) |
                 VkFlags( DependencyFlagBits::eFeedbackLoopEXT )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DependencyFlags operator|( DependencyFlagBits bit0, DependencyFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DependencyFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DependencyFlags operator&( DependencyFlagBits bit0, DependencyFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DependencyFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DependencyFlags operator^( DependencyFlagBits bit0, DependencyFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DependencyFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DependencyFlags operator~( DependencyFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( DependencyFlags( bits ) );
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

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FramebufferCreateFlags operator|( FramebufferCreateFlagBits bit0, FramebufferCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return FramebufferCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FramebufferCreateFlags operator&( FramebufferCreateFlagBits bit0, FramebufferCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return FramebufferCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FramebufferCreateFlags operator^( FramebufferCreateFlagBits bit0, FramebufferCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return FramebufferCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FramebufferCreateFlags operator~( FramebufferCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( FramebufferCreateFlags( bits ) );
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

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR RenderPassCreateFlags operator|( RenderPassCreateFlagBits bit0, RenderPassCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return RenderPassCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR RenderPassCreateFlags operator&( RenderPassCreateFlagBits bit0, RenderPassCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return RenderPassCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR RenderPassCreateFlags operator^( RenderPassCreateFlagBits bit0, RenderPassCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return RenderPassCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR RenderPassCreateFlags operator~( RenderPassCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( RenderPassCreateFlags( bits ) );
  }

  using SubpassDescriptionFlags = Flags<SubpassDescriptionFlagBits>;

  template <>
  struct FlagTraits<SubpassDescriptionFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( SubpassDescriptionFlagBits::ePerViewAttributesNVX ) | VkFlags( SubpassDescriptionFlagBits::ePerViewPositionXOnlyNVX ) |
                 VkFlags( SubpassDescriptionFlagBits::eFragmentRegionQCOM ) | VkFlags( SubpassDescriptionFlagBits::eShaderResolveQCOM ) |
                 VkFlags( SubpassDescriptionFlagBits::eRasterizationOrderAttachmentColorAccessEXT ) |
                 VkFlags( SubpassDescriptionFlagBits::eRasterizationOrderAttachmentDepthAccessEXT ) |
                 VkFlags( SubpassDescriptionFlagBits::eRasterizationOrderAttachmentStencilAccessEXT ) |
                 VkFlags( SubpassDescriptionFlagBits::eEnableLegacyDitheringEXT )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SubpassDescriptionFlags operator|( SubpassDescriptionFlagBits bit0,
                                                                            SubpassDescriptionFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SubpassDescriptionFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SubpassDescriptionFlags operator&( SubpassDescriptionFlagBits bit0,
                                                                            SubpassDescriptionFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SubpassDescriptionFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SubpassDescriptionFlags operator^( SubpassDescriptionFlagBits bit0,
                                                                            SubpassDescriptionFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SubpassDescriptionFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SubpassDescriptionFlags operator~( SubpassDescriptionFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( SubpassDescriptionFlags( bits ) );
  }

  using CommandPoolCreateFlags = Flags<CommandPoolCreateFlagBits>;

  template <>
  struct FlagTraits<CommandPoolCreateFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( CommandPoolCreateFlagBits::eTransient ) | VkFlags( CommandPoolCreateFlagBits::eResetCommandBuffer ) |
                 VkFlags( CommandPoolCreateFlagBits::eProtected )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandPoolCreateFlags operator|( CommandPoolCreateFlagBits bit0, CommandPoolCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CommandPoolCreateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandPoolCreateFlags operator&( CommandPoolCreateFlagBits bit0, CommandPoolCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CommandPoolCreateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandPoolCreateFlags operator^( CommandPoolCreateFlagBits bit0, CommandPoolCreateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CommandPoolCreateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandPoolCreateFlags operator~( CommandPoolCreateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( CommandPoolCreateFlags( bits ) );
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

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandPoolResetFlags operator|( CommandPoolResetFlagBits bit0, CommandPoolResetFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CommandPoolResetFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandPoolResetFlags operator&( CommandPoolResetFlagBits bit0, CommandPoolResetFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CommandPoolResetFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandPoolResetFlags operator^( CommandPoolResetFlagBits bit0, CommandPoolResetFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CommandPoolResetFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandPoolResetFlags operator~( CommandPoolResetFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( CommandPoolResetFlags( bits ) );
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

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandBufferResetFlags operator|( CommandBufferResetFlagBits bit0,
                                                                            CommandBufferResetFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CommandBufferResetFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandBufferResetFlags operator&( CommandBufferResetFlagBits bit0,
                                                                            CommandBufferResetFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CommandBufferResetFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandBufferResetFlags operator^( CommandBufferResetFlagBits bit0,
                                                                            CommandBufferResetFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CommandBufferResetFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandBufferResetFlags operator~( CommandBufferResetFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( CommandBufferResetFlags( bits ) );
  }

  using CommandBufferUsageFlags = Flags<CommandBufferUsageFlagBits>;

  template <>
  struct FlagTraits<CommandBufferUsageFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( CommandBufferUsageFlagBits::eOneTimeSubmit ) | VkFlags( CommandBufferUsageFlagBits::eRenderPassContinue ) |
                 VkFlags( CommandBufferUsageFlagBits::eSimultaneousUse )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandBufferUsageFlags operator|( CommandBufferUsageFlagBits bit0,
                                                                            CommandBufferUsageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CommandBufferUsageFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandBufferUsageFlags operator&( CommandBufferUsageFlagBits bit0,
                                                                            CommandBufferUsageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CommandBufferUsageFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandBufferUsageFlags operator^( CommandBufferUsageFlagBits bit0,
                                                                            CommandBufferUsageFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CommandBufferUsageFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CommandBufferUsageFlags operator~( CommandBufferUsageFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( CommandBufferUsageFlags( bits ) );
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

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueryControlFlags operator|( QueryControlFlagBits bit0, QueryControlFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return QueryControlFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueryControlFlags operator&( QueryControlFlagBits bit0, QueryControlFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return QueryControlFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueryControlFlags operator^( QueryControlFlagBits bit0, QueryControlFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return QueryControlFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR QueryControlFlags operator~( QueryControlFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( QueryControlFlags( bits ) );
  }

  using StencilFaceFlags = Flags<StencilFaceFlagBits>;

  template <>
  struct FlagTraits<StencilFaceFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( StencilFaceFlagBits::eFront ) | VkFlags( StencilFaceFlagBits::eBack ) | VkFlags( StencilFaceFlagBits::eFrontAndBack )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR StencilFaceFlags operator|( StencilFaceFlagBits bit0, StencilFaceFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return StencilFaceFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR StencilFaceFlags operator&( StencilFaceFlagBits bit0, StencilFaceFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return StencilFaceFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR StencilFaceFlags operator^( StencilFaceFlagBits bit0, StencilFaceFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return StencilFaceFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR StencilFaceFlags operator~( StencilFaceFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( StencilFaceFlags( bits ) );
  }

  //=== VK_VERSION_1_1 ===

  using SubgroupFeatureFlags = Flags<SubgroupFeatureFlagBits>;

  template <>
  struct FlagTraits<SubgroupFeatureFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( SubgroupFeatureFlagBits::eBasic ) | VkFlags( SubgroupFeatureFlagBits::eVote ) | VkFlags( SubgroupFeatureFlagBits::eArithmetic ) |
                 VkFlags( SubgroupFeatureFlagBits::eBallot ) | VkFlags( SubgroupFeatureFlagBits::eShuffle ) |
                 VkFlags( SubgroupFeatureFlagBits::eShuffleRelative ) | VkFlags( SubgroupFeatureFlagBits::eClustered ) |
                 VkFlags( SubgroupFeatureFlagBits::eQuad ) | VkFlags( SubgroupFeatureFlagBits::ePartitionedNV )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SubgroupFeatureFlags operator|( SubgroupFeatureFlagBits bit0, SubgroupFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SubgroupFeatureFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SubgroupFeatureFlags operator&( SubgroupFeatureFlagBits bit0, SubgroupFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SubgroupFeatureFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SubgroupFeatureFlags operator^( SubgroupFeatureFlagBits bit0, SubgroupFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SubgroupFeatureFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SubgroupFeatureFlags operator~( SubgroupFeatureFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( SubgroupFeatureFlags( bits ) );
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

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PeerMemoryFeatureFlags operator|( PeerMemoryFeatureFlagBits bit0, PeerMemoryFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PeerMemoryFeatureFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PeerMemoryFeatureFlags operator&( PeerMemoryFeatureFlagBits bit0, PeerMemoryFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PeerMemoryFeatureFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PeerMemoryFeatureFlags operator^( PeerMemoryFeatureFlagBits bit0, PeerMemoryFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PeerMemoryFeatureFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PeerMemoryFeatureFlags operator~( PeerMemoryFeatureFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( PeerMemoryFeatureFlags( bits ) );
  }

  using PeerMemoryFeatureFlagsKHR = PeerMemoryFeatureFlags;

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

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MemoryAllocateFlags operator|( MemoryAllocateFlagBits bit0, MemoryAllocateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return MemoryAllocateFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MemoryAllocateFlags operator&( MemoryAllocateFlagBits bit0, MemoryAllocateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return MemoryAllocateFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MemoryAllocateFlags operator^( MemoryAllocateFlagBits bit0, MemoryAllocateFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return MemoryAllocateFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MemoryAllocateFlags operator~( MemoryAllocateFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( MemoryAllocateFlags( bits ) );
  }

  using MemoryAllocateFlagsKHR = MemoryAllocateFlags;

  using CommandPoolTrimFlags = Flags<CommandPoolTrimFlagBits>;

  using CommandPoolTrimFlagsKHR = CommandPoolTrimFlags;

  using DescriptorUpdateTemplateCreateFlags = Flags<DescriptorUpdateTemplateCreateFlagBits>;

  using DescriptorUpdateTemplateCreateFlagsKHR = DescriptorUpdateTemplateCreateFlags;

  using ExternalMemoryHandleTypeFlags = Flags<ExternalMemoryHandleTypeFlagBits>;

  template <>
  struct FlagTraits<ExternalMemoryHandleTypeFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ExternalMemoryHandleTypeFlagBits::eOpaqueFd ) | VkFlags( ExternalMemoryHandleTypeFlagBits::eOpaqueWin32 ) |
                 VkFlags( ExternalMemoryHandleTypeFlagBits::eOpaqueWin32Kmt ) | VkFlags( ExternalMemoryHandleTypeFlagBits::eD3D11Texture ) |
                 VkFlags( ExternalMemoryHandleTypeFlagBits::eD3D11TextureKmt ) | VkFlags( ExternalMemoryHandleTypeFlagBits::eD3D12Heap ) |
                 VkFlags( ExternalMemoryHandleTypeFlagBits::eD3D12Resource ) | VkFlags( ExternalMemoryHandleTypeFlagBits::eDmaBufEXT )
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
                 | VkFlags( ExternalMemoryHandleTypeFlagBits::eAndroidHardwareBufferANDROID )
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
                 | VkFlags( ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT ) | VkFlags( ExternalMemoryHandleTypeFlagBits::eHostMappedForeignMemoryEXT )
#if defined( VK_USE_PLATFORM_FUCHSIA )
                 | VkFlags( ExternalMemoryHandleTypeFlagBits::eZirconVmoFUCHSIA )
#endif /*VK_USE_PLATFORM_FUCHSIA*/
                 | VkFlags( ExternalMemoryHandleTypeFlagBits::eRdmaAddressNV )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryHandleTypeFlags operator|( ExternalMemoryHandleTypeFlagBits bit0,
                                                                                  ExternalMemoryHandleTypeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalMemoryHandleTypeFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryHandleTypeFlags operator&( ExternalMemoryHandleTypeFlagBits bit0,
                                                                                  ExternalMemoryHandleTypeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalMemoryHandleTypeFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryHandleTypeFlags operator^( ExternalMemoryHandleTypeFlagBits bit0,
                                                                                  ExternalMemoryHandleTypeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalMemoryHandleTypeFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryHandleTypeFlags operator~( ExternalMemoryHandleTypeFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ExternalMemoryHandleTypeFlags( bits ) );
  }

  using ExternalMemoryHandleTypeFlagsKHR = ExternalMemoryHandleTypeFlags;

  using ExternalMemoryFeatureFlags = Flags<ExternalMemoryFeatureFlagBits>;

  template <>
  struct FlagTraits<ExternalMemoryFeatureFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ExternalMemoryFeatureFlagBits::eDedicatedOnly ) | VkFlags( ExternalMemoryFeatureFlagBits::eExportable ) |
                 VkFlags( ExternalMemoryFeatureFlagBits::eImportable )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryFeatureFlags operator|( ExternalMemoryFeatureFlagBits bit0,
                                                                               ExternalMemoryFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalMemoryFeatureFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryFeatureFlags operator&( ExternalMemoryFeatureFlagBits bit0,
                                                                               ExternalMemoryFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalMemoryFeatureFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryFeatureFlags operator^( ExternalMemoryFeatureFlagBits bit0,
                                                                               ExternalMemoryFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalMemoryFeatureFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryFeatureFlags operator~( ExternalMemoryFeatureFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ExternalMemoryFeatureFlags( bits ) );
  }

  using ExternalMemoryFeatureFlagsKHR = ExternalMemoryFeatureFlags;

  using ExternalFenceHandleTypeFlags = Flags<ExternalFenceHandleTypeFlagBits>;

  template <>
  struct FlagTraits<ExternalFenceHandleTypeFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ExternalFenceHandleTypeFlagBits::eOpaqueFd ) | VkFlags( ExternalFenceHandleTypeFlagBits::eOpaqueWin32 ) |
                 VkFlags( ExternalFenceHandleTypeFlagBits::eOpaqueWin32Kmt ) | VkFlags( ExternalFenceHandleTypeFlagBits::eSyncFd )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalFenceHandleTypeFlags operator|( ExternalFenceHandleTypeFlagBits bit0,
                                                                                 ExternalFenceHandleTypeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalFenceHandleTypeFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalFenceHandleTypeFlags operator&( ExternalFenceHandleTypeFlagBits bit0,
                                                                                 ExternalFenceHandleTypeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalFenceHandleTypeFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalFenceHandleTypeFlags operator^( ExternalFenceHandleTypeFlagBits bit0,
                                                                                 ExternalFenceHandleTypeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalFenceHandleTypeFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalFenceHandleTypeFlags operator~( ExternalFenceHandleTypeFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ExternalFenceHandleTypeFlags( bits ) );
  }

  using ExternalFenceHandleTypeFlagsKHR = ExternalFenceHandleTypeFlags;

  using ExternalFenceFeatureFlags = Flags<ExternalFenceFeatureFlagBits>;

  template <>
  struct FlagTraits<ExternalFenceFeatureFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ExternalFenceFeatureFlagBits::eExportable ) | VkFlags( ExternalFenceFeatureFlagBits::eImportable )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalFenceFeatureFlags operator|( ExternalFenceFeatureFlagBits bit0,
                                                                              ExternalFenceFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalFenceFeatureFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalFenceFeatureFlags operator&( ExternalFenceFeatureFlagBits bit0,
                                                                              ExternalFenceFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalFenceFeatureFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalFenceFeatureFlags operator^( ExternalFenceFeatureFlagBits bit0,
                                                                              ExternalFenceFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalFenceFeatureFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalFenceFeatureFlags operator~( ExternalFenceFeatureFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ExternalFenceFeatureFlags( bits ) );
  }

  using ExternalFenceFeatureFlagsKHR = ExternalFenceFeatureFlags;

  using FenceImportFlags = Flags<FenceImportFlagBits>;

  template <>
  struct FlagTraits<FenceImportFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( FenceImportFlagBits::eTemporary )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FenceImportFlags operator|( FenceImportFlagBits bit0, FenceImportFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return FenceImportFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FenceImportFlags operator&( FenceImportFlagBits bit0, FenceImportFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return FenceImportFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FenceImportFlags operator^( FenceImportFlagBits bit0, FenceImportFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return FenceImportFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FenceImportFlags operator~( FenceImportFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( FenceImportFlags( bits ) );
  }

  using FenceImportFlagsKHR = FenceImportFlags;

  using SemaphoreImportFlags = Flags<SemaphoreImportFlagBits>;

  template <>
  struct FlagTraits<SemaphoreImportFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( SemaphoreImportFlagBits::eTemporary )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SemaphoreImportFlags operator|( SemaphoreImportFlagBits bit0, SemaphoreImportFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SemaphoreImportFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SemaphoreImportFlags operator&( SemaphoreImportFlagBits bit0, SemaphoreImportFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SemaphoreImportFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SemaphoreImportFlags operator^( SemaphoreImportFlagBits bit0, SemaphoreImportFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SemaphoreImportFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SemaphoreImportFlags operator~( SemaphoreImportFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( SemaphoreImportFlags( bits ) );
  }

  using SemaphoreImportFlagsKHR = SemaphoreImportFlags;

  using ExternalSemaphoreHandleTypeFlags = Flags<ExternalSemaphoreHandleTypeFlagBits>;

  template <>
  struct FlagTraits<ExternalSemaphoreHandleTypeFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd ) | VkFlags( ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32 ) |
                 VkFlags( ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32Kmt ) | VkFlags( ExternalSemaphoreHandleTypeFlagBits::eD3D12Fence ) |
                 VkFlags( ExternalSemaphoreHandleTypeFlagBits::eSyncFd )
#if defined( VK_USE_PLATFORM_FUCHSIA )
                 | VkFlags( ExternalSemaphoreHandleTypeFlagBits::eZirconEventFUCHSIA )
#endif /*VK_USE_PLATFORM_FUCHSIA*/
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalSemaphoreHandleTypeFlags operator|( ExternalSemaphoreHandleTypeFlagBits bit0,
                                                                                     ExternalSemaphoreHandleTypeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalSemaphoreHandleTypeFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalSemaphoreHandleTypeFlags operator&( ExternalSemaphoreHandleTypeFlagBits bit0,
                                                                                     ExternalSemaphoreHandleTypeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalSemaphoreHandleTypeFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalSemaphoreHandleTypeFlags operator^( ExternalSemaphoreHandleTypeFlagBits bit0,
                                                                                     ExternalSemaphoreHandleTypeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalSemaphoreHandleTypeFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalSemaphoreHandleTypeFlags operator~( ExternalSemaphoreHandleTypeFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ExternalSemaphoreHandleTypeFlags( bits ) );
  }

  using ExternalSemaphoreHandleTypeFlagsKHR = ExternalSemaphoreHandleTypeFlags;

  using ExternalSemaphoreFeatureFlags = Flags<ExternalSemaphoreFeatureFlagBits>;

  template <>
  struct FlagTraits<ExternalSemaphoreFeatureFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ExternalSemaphoreFeatureFlagBits::eExportable ) | VkFlags( ExternalSemaphoreFeatureFlagBits::eImportable )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalSemaphoreFeatureFlags operator|( ExternalSemaphoreFeatureFlagBits bit0,
                                                                                  ExternalSemaphoreFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalSemaphoreFeatureFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalSemaphoreFeatureFlags operator&( ExternalSemaphoreFeatureFlagBits bit0,
                                                                                  ExternalSemaphoreFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalSemaphoreFeatureFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalSemaphoreFeatureFlags operator^( ExternalSemaphoreFeatureFlagBits bit0,
                                                                                  ExternalSemaphoreFeatureFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalSemaphoreFeatureFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalSemaphoreFeatureFlags operator~( ExternalSemaphoreFeatureFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ExternalSemaphoreFeatureFlags( bits ) );
  }

  using ExternalSemaphoreFeatureFlagsKHR = ExternalSemaphoreFeatureFlags;

  //=== VK_VERSION_1_2 ===

  using DescriptorBindingFlags = Flags<DescriptorBindingFlagBits>;

  template <>
  struct FlagTraits<DescriptorBindingFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( DescriptorBindingFlagBits::eUpdateAfterBind ) | VkFlags( DescriptorBindingFlagBits::eUpdateUnusedWhilePending ) |
                 VkFlags( DescriptorBindingFlagBits::ePartiallyBound ) | VkFlags( DescriptorBindingFlagBits::eVariableDescriptorCount )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DescriptorBindingFlags operator|( DescriptorBindingFlagBits bit0, DescriptorBindingFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DescriptorBindingFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DescriptorBindingFlags operator&( DescriptorBindingFlagBits bit0, DescriptorBindingFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DescriptorBindingFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DescriptorBindingFlags operator^( DescriptorBindingFlagBits bit0, DescriptorBindingFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DescriptorBindingFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DescriptorBindingFlags operator~( DescriptorBindingFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( DescriptorBindingFlags( bits ) );
  }

  using DescriptorBindingFlagsEXT = DescriptorBindingFlags;

  using ResolveModeFlags = Flags<ResolveModeFlagBits>;

  template <>
  struct FlagTraits<ResolveModeFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ResolveModeFlagBits::eNone ) | VkFlags( ResolveModeFlagBits::eSampleZero ) | VkFlags( ResolveModeFlagBits::eAverage ) |
                 VkFlags( ResolveModeFlagBits::eMin ) | VkFlags( ResolveModeFlagBits::eMax )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ResolveModeFlags operator|( ResolveModeFlagBits bit0, ResolveModeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ResolveModeFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ResolveModeFlags operator&( ResolveModeFlagBits bit0, ResolveModeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ResolveModeFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ResolveModeFlags operator^( ResolveModeFlagBits bit0, ResolveModeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ResolveModeFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ResolveModeFlags operator~( ResolveModeFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ResolveModeFlags( bits ) );
  }

  using ResolveModeFlagsKHR = ResolveModeFlags;

  using SemaphoreWaitFlags = Flags<SemaphoreWaitFlagBits>;

  template <>
  struct FlagTraits<SemaphoreWaitFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( SemaphoreWaitFlagBits::eAny )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SemaphoreWaitFlags operator|( SemaphoreWaitFlagBits bit0, SemaphoreWaitFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SemaphoreWaitFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SemaphoreWaitFlags operator&( SemaphoreWaitFlagBits bit0, SemaphoreWaitFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SemaphoreWaitFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SemaphoreWaitFlags operator^( SemaphoreWaitFlagBits bit0, SemaphoreWaitFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SemaphoreWaitFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SemaphoreWaitFlags operator~( SemaphoreWaitFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( SemaphoreWaitFlags( bits ) );
  }

  using SemaphoreWaitFlagsKHR = SemaphoreWaitFlags;

  //=== VK_VERSION_1_3 ===

  using PipelineCreationFeedbackFlags = Flags<PipelineCreationFeedbackFlagBits>;

  template <>
  struct FlagTraits<PipelineCreationFeedbackFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( PipelineCreationFeedbackFlagBits::eValid ) | VkFlags( PipelineCreationFeedbackFlagBits::eApplicationPipelineCacheHit ) |
                 VkFlags( PipelineCreationFeedbackFlagBits::eBasePipelineAcceleration )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineCreationFeedbackFlags operator|( PipelineCreationFeedbackFlagBits bit0,
                                                                                  PipelineCreationFeedbackFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineCreationFeedbackFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineCreationFeedbackFlags operator&( PipelineCreationFeedbackFlagBits bit0,
                                                                                  PipelineCreationFeedbackFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineCreationFeedbackFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineCreationFeedbackFlags operator^( PipelineCreationFeedbackFlagBits bit0,
                                                                                  PipelineCreationFeedbackFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineCreationFeedbackFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineCreationFeedbackFlags operator~( PipelineCreationFeedbackFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( PipelineCreationFeedbackFlags( bits ) );
  }

  using PipelineCreationFeedbackFlagsEXT = PipelineCreationFeedbackFlags;

  using ToolPurposeFlags = Flags<ToolPurposeFlagBits>;

  template <>
  struct FlagTraits<ToolPurposeFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ToolPurposeFlagBits::eValidation ) | VkFlags( ToolPurposeFlagBits::eProfiling ) | VkFlags( ToolPurposeFlagBits::eTracing ) |
                 VkFlags( ToolPurposeFlagBits::eAdditionalFeatures ) | VkFlags( ToolPurposeFlagBits::eModifyingFeatures ) |
                 VkFlags( ToolPurposeFlagBits::eDebugReportingEXT ) | VkFlags( ToolPurposeFlagBits::eDebugMarkersEXT )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ToolPurposeFlags operator|( ToolPurposeFlagBits bit0, ToolPurposeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ToolPurposeFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ToolPurposeFlags operator&( ToolPurposeFlagBits bit0, ToolPurposeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ToolPurposeFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ToolPurposeFlags operator^( ToolPurposeFlagBits bit0, ToolPurposeFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ToolPurposeFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ToolPurposeFlags operator~( ToolPurposeFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ToolPurposeFlags( bits ) );
  }

  using ToolPurposeFlagsEXT = ToolPurposeFlags;

  using PrivateDataSlotCreateFlags = Flags<PrivateDataSlotCreateFlagBits>;

  using PrivateDataSlotCreateFlagsEXT = PrivateDataSlotCreateFlags;

  using PipelineStageFlags2 = Flags<PipelineStageFlagBits2>;

  template <>
  struct FlagTraits<PipelineStageFlagBits2>
  {
    enum : VkFlags64
    {
      allFlags = VkFlags64( PipelineStageFlagBits2::eNone ) | VkFlags64( PipelineStageFlagBits2::eTopOfPipe ) |
                 VkFlags64( PipelineStageFlagBits2::eDrawIndirect ) | VkFlags64( PipelineStageFlagBits2::eVertexInput ) |
                 VkFlags64( PipelineStageFlagBits2::eVertexShader ) | VkFlags64( PipelineStageFlagBits2::eTessellationControlShader ) |
                 VkFlags64( PipelineStageFlagBits2::eTessellationEvaluationShader ) | VkFlags64( PipelineStageFlagBits2::eGeometryShader ) |
                 VkFlags64( PipelineStageFlagBits2::eFragmentShader ) | VkFlags64( PipelineStageFlagBits2::eEarlyFragmentTests ) |
                 VkFlags64( PipelineStageFlagBits2::eLateFragmentTests ) | VkFlags64( PipelineStageFlagBits2::eColorAttachmentOutput ) |
                 VkFlags64( PipelineStageFlagBits2::eComputeShader ) | VkFlags64( PipelineStageFlagBits2::eAllTransfer ) |
                 VkFlags64( PipelineStageFlagBits2::eBottomOfPipe ) | VkFlags64( PipelineStageFlagBits2::eHost ) |
                 VkFlags64( PipelineStageFlagBits2::eAllGraphics ) | VkFlags64( PipelineStageFlagBits2::eAllCommands ) |
                 VkFlags64( PipelineStageFlagBits2::eCopy ) | VkFlags64( PipelineStageFlagBits2::eResolve ) | VkFlags64( PipelineStageFlagBits2::eBlit ) |
                 VkFlags64( PipelineStageFlagBits2::eClear ) | VkFlags64( PipelineStageFlagBits2::eIndexInput ) |
                 VkFlags64( PipelineStageFlagBits2::eVertexAttributeInput ) | VkFlags64( PipelineStageFlagBits2::ePreRasterizationShaders )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
                 | VkFlags64( PipelineStageFlagBits2::eVideoDecodeKHR ) | VkFlags64( PipelineStageFlagBits2::eVideoEncodeKHR )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
                 | VkFlags64( PipelineStageFlagBits2::eTransformFeedbackEXT ) | VkFlags64( PipelineStageFlagBits2::eConditionalRenderingEXT ) |
                 VkFlags64( PipelineStageFlagBits2::eCommandPreprocessNV ) | VkFlags64( PipelineStageFlagBits2::eFragmentShadingRateAttachmentKHR ) |
                 VkFlags64( PipelineStageFlagBits2::eAccelerationStructureBuildKHR ) | VkFlags64( PipelineStageFlagBits2::eRayTracingShaderKHR ) |
                 VkFlags64( PipelineStageFlagBits2::eFragmentDensityProcessEXT ) | VkFlags64( PipelineStageFlagBits2::eTaskShaderEXT ) |
                 VkFlags64( PipelineStageFlagBits2::eMeshShaderEXT ) | VkFlags64( PipelineStageFlagBits2::eSubpassShadingHUAWEI ) |
                 VkFlags64( PipelineStageFlagBits2::eInvocationMaskHUAWEI ) | VkFlags64( PipelineStageFlagBits2::eAccelerationStructureCopyKHR ) |
                 VkFlags64( PipelineStageFlagBits2::eMicromapBuildEXT ) | VkFlags64( PipelineStageFlagBits2::eOpticalFlowNV )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineStageFlags2 operator|( PipelineStageFlagBits2 bit0, PipelineStageFlagBits2 bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineStageFlags2( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineStageFlags2 operator&( PipelineStageFlagBits2 bit0, PipelineStageFlagBits2 bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineStageFlags2( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineStageFlags2 operator^( PipelineStageFlagBits2 bit0, PipelineStageFlagBits2 bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PipelineStageFlags2( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PipelineStageFlags2 operator~( PipelineStageFlagBits2 bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( PipelineStageFlags2( bits ) );
  }

  using PipelineStageFlags2KHR = PipelineStageFlags2;

  using AccessFlags2 = Flags<AccessFlagBits2>;

  template <>
  struct FlagTraits<AccessFlagBits2>
  {
    enum : VkFlags64
    {
      allFlags =
        VkFlags64( AccessFlagBits2::eNone ) | VkFlags64( AccessFlagBits2::eIndirectCommandRead ) | VkFlags64( AccessFlagBits2::eIndexRead ) |
        VkFlags64( AccessFlagBits2::eVertexAttributeRead ) | VkFlags64( AccessFlagBits2::eUniformRead ) | VkFlags64( AccessFlagBits2::eInputAttachmentRead ) |
        VkFlags64( AccessFlagBits2::eShaderRead ) | VkFlags64( AccessFlagBits2::eShaderWrite ) | VkFlags64( AccessFlagBits2::eColorAttachmentRead ) |
        VkFlags64( AccessFlagBits2::eColorAttachmentWrite ) | VkFlags64( AccessFlagBits2::eDepthStencilAttachmentRead ) |
        VkFlags64( AccessFlagBits2::eDepthStencilAttachmentWrite ) | VkFlags64( AccessFlagBits2::eTransferRead ) |
        VkFlags64( AccessFlagBits2::eTransferWrite ) | VkFlags64( AccessFlagBits2::eHostRead ) | VkFlags64( AccessFlagBits2::eHostWrite ) |
        VkFlags64( AccessFlagBits2::eMemoryRead ) | VkFlags64( AccessFlagBits2::eMemoryWrite ) | VkFlags64( AccessFlagBits2::eShaderSampledRead ) |
        VkFlags64( AccessFlagBits2::eShaderStorageRead ) | VkFlags64( AccessFlagBits2::eShaderStorageWrite )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
        | VkFlags64( AccessFlagBits2::eVideoDecodeReadKHR ) | VkFlags64( AccessFlagBits2::eVideoDecodeWriteKHR ) |
        VkFlags64( AccessFlagBits2::eVideoEncodeReadKHR ) | VkFlags64( AccessFlagBits2::eVideoEncodeWriteKHR )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
        | VkFlags64( AccessFlagBits2::eTransformFeedbackWriteEXT ) | VkFlags64( AccessFlagBits2::eTransformFeedbackCounterReadEXT ) |
        VkFlags64( AccessFlagBits2::eTransformFeedbackCounterWriteEXT ) | VkFlags64( AccessFlagBits2::eConditionalRenderingReadEXT ) |
        VkFlags64( AccessFlagBits2::eCommandPreprocessReadNV ) | VkFlags64( AccessFlagBits2::eCommandPreprocessWriteNV ) |
        VkFlags64( AccessFlagBits2::eFragmentShadingRateAttachmentReadKHR ) | VkFlags64( AccessFlagBits2::eAccelerationStructureReadKHR ) |
        VkFlags64( AccessFlagBits2::eAccelerationStructureWriteKHR ) | VkFlags64( AccessFlagBits2::eFragmentDensityMapReadEXT ) |
        VkFlags64( AccessFlagBits2::eColorAttachmentReadNoncoherentEXT ) | VkFlags64( AccessFlagBits2::eInvocationMaskReadHUAWEI ) |
        VkFlags64( AccessFlagBits2::eShaderBindingTableReadKHR ) | VkFlags64( AccessFlagBits2::eMicromapReadEXT ) |
        VkFlags64( AccessFlagBits2::eMicromapWriteEXT ) | VkFlags64( AccessFlagBits2::eOpticalFlowReadNV ) | VkFlags64( AccessFlagBits2::eOpticalFlowWriteNV )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AccessFlags2 operator|( AccessFlagBits2 bit0, AccessFlagBits2 bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return AccessFlags2( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AccessFlags2 operator&( AccessFlagBits2 bit0, AccessFlagBits2 bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return AccessFlags2( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AccessFlags2 operator^( AccessFlagBits2 bit0, AccessFlagBits2 bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return AccessFlags2( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AccessFlags2 operator~( AccessFlagBits2 bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( AccessFlags2( bits ) );
  }

  using AccessFlags2KHR = AccessFlags2;

  using SubmitFlags = Flags<SubmitFlagBits>;

  template <>
  struct FlagTraits<SubmitFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( SubmitFlagBits::eProtected )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SubmitFlags operator|( SubmitFlagBits bit0, SubmitFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SubmitFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SubmitFlags operator&( SubmitFlagBits bit0, SubmitFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SubmitFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SubmitFlags operator^( SubmitFlagBits bit0, SubmitFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SubmitFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SubmitFlags operator~( SubmitFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( SubmitFlags( bits ) );
  }

  using SubmitFlagsKHR = SubmitFlags;

  using RenderingFlags = Flags<RenderingFlagBits>;

  template <>
  struct FlagTraits<RenderingFlagBits>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( RenderingFlagBits::eContentsSecondaryCommandBuffers ) | VkFlags( RenderingFlagBits::eSuspending ) |
                 VkFlags( RenderingFlagBits::eResuming ) | VkFlags( RenderingFlagBits::eEnableLegacyDitheringEXT )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR RenderingFlags operator|( RenderingFlagBits bit0, RenderingFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return RenderingFlags( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR RenderingFlags operator&( RenderingFlagBits bit0, RenderingFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return RenderingFlags( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR RenderingFlags operator^( RenderingFlagBits bit0, RenderingFlagBits bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return RenderingFlags( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR RenderingFlags operator~( RenderingFlagBits bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( RenderingFlags( bits ) );
  }

  using RenderingFlagsKHR = RenderingFlags;

  using FormatFeatureFlags2 = Flags<FormatFeatureFlagBits2>;

  template <>
  struct FlagTraits<FormatFeatureFlagBits2>
  {
    enum : VkFlags64
    {
      allFlags = VkFlags64( FormatFeatureFlagBits2::eSampledImage ) | VkFlags64( FormatFeatureFlagBits2::eStorageImage ) |
                 VkFlags64( FormatFeatureFlagBits2::eStorageImageAtomic ) | VkFlags64( FormatFeatureFlagBits2::eUniformTexelBuffer ) |
                 VkFlags64( FormatFeatureFlagBits2::eStorageTexelBuffer ) | VkFlags64( FormatFeatureFlagBits2::eStorageTexelBufferAtomic ) |
                 VkFlags64( FormatFeatureFlagBits2::eVertexBuffer ) | VkFlags64( FormatFeatureFlagBits2::eColorAttachment ) |
                 VkFlags64( FormatFeatureFlagBits2::eColorAttachmentBlend ) | VkFlags64( FormatFeatureFlagBits2::eDepthStencilAttachment ) |
                 VkFlags64( FormatFeatureFlagBits2::eBlitSrc ) | VkFlags64( FormatFeatureFlagBits2::eBlitDst ) |
                 VkFlags64( FormatFeatureFlagBits2::eSampledImageFilterLinear ) | VkFlags64( FormatFeatureFlagBits2::eSampledImageFilterCubic ) |
                 VkFlags64( FormatFeatureFlagBits2::eTransferSrc ) | VkFlags64( FormatFeatureFlagBits2::eTransferDst ) |
                 VkFlags64( FormatFeatureFlagBits2::eSampledImageFilterMinmax ) | VkFlags64( FormatFeatureFlagBits2::eMidpointChromaSamples ) |
                 VkFlags64( FormatFeatureFlagBits2::eSampledImageYcbcrConversionLinearFilter ) |
                 VkFlags64( FormatFeatureFlagBits2::eSampledImageYcbcrConversionSeparateReconstructionFilter ) |
                 VkFlags64( FormatFeatureFlagBits2::eSampledImageYcbcrConversionChromaReconstructionExplicit ) |
                 VkFlags64( FormatFeatureFlagBits2::eSampledImageYcbcrConversionChromaReconstructionExplicitForceable ) |
                 VkFlags64( FormatFeatureFlagBits2::eDisjoint ) | VkFlags64( FormatFeatureFlagBits2::eCositedChromaSamples ) |
                 VkFlags64( FormatFeatureFlagBits2::eStorageReadWithoutFormat ) | VkFlags64( FormatFeatureFlagBits2::eStorageWriteWithoutFormat ) |
                 VkFlags64( FormatFeatureFlagBits2::eSampledImageDepthComparison )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
                 | VkFlags64( FormatFeatureFlagBits2::eVideoDecodeOutputKHR ) | VkFlags64( FormatFeatureFlagBits2::eVideoDecodeDpbKHR )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
                 | VkFlags64( FormatFeatureFlagBits2::eAccelerationStructureVertexBufferKHR ) | VkFlags64( FormatFeatureFlagBits2::eFragmentDensityMapEXT ) |
                 VkFlags64( FormatFeatureFlagBits2::eFragmentShadingRateAttachmentKHR )
#if defined( VK_ENABLE_BETA_EXTENSIONS )
                 | VkFlags64( FormatFeatureFlagBits2::eVideoEncodeInputKHR ) | VkFlags64( FormatFeatureFlagBits2::eVideoEncodeDpbKHR )
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
                 | VkFlags64( FormatFeatureFlagBits2::eLinearColorAttachmentNV ) | VkFlags64( FormatFeatureFlagBits2::eWeightImageQCOM ) |
                 VkFlags64( FormatFeatureFlagBits2::eWeightSampledImageQCOM ) | VkFlags64( FormatFeatureFlagBits2::eBlockMatchingQCOM ) |
                 VkFlags64( FormatFeatureFlagBits2::eBoxFilterSampledQCOM ) | VkFlags64( FormatFeatureFlagBits2::eOpticalFlowImageNV ) |
                 VkFlags64( FormatFeatureFlagBits2::eOpticalFlowVectorNV ) | VkFlags64( FormatFeatureFlagBits2::eOpticalFlowCostNV )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FormatFeatureFlags2 operator|( FormatFeatureFlagBits2 bit0, FormatFeatureFlagBits2 bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return FormatFeatureFlags2( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FormatFeatureFlags2 operator&( FormatFeatureFlagBits2 bit0, FormatFeatureFlagBits2 bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return FormatFeatureFlags2( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FormatFeatureFlags2 operator^( FormatFeatureFlagBits2 bit0, FormatFeatureFlagBits2 bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return FormatFeatureFlags2( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR FormatFeatureFlags2 operator~( FormatFeatureFlagBits2 bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( FormatFeatureFlags2( bits ) );
  }

  using FormatFeatureFlags2KHR = FormatFeatureFlags2;

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

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CompositeAlphaFlagsKHR operator|( CompositeAlphaFlagBitsKHR bit0, CompositeAlphaFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CompositeAlphaFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CompositeAlphaFlagsKHR operator&( CompositeAlphaFlagBitsKHR bit0, CompositeAlphaFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CompositeAlphaFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CompositeAlphaFlagsKHR operator^( CompositeAlphaFlagBitsKHR bit0, CompositeAlphaFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return CompositeAlphaFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR CompositeAlphaFlagsKHR operator~( CompositeAlphaFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( CompositeAlphaFlagsKHR( bits ) );
  }

  //=== VK_KHR_swapchain ===

  using SwapchainCreateFlagsKHR = Flags<SwapchainCreateFlagBitsKHR>;

  template <>
  struct FlagTraits<SwapchainCreateFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( SwapchainCreateFlagBitsKHR::eSplitInstanceBindRegions ) | VkFlags( SwapchainCreateFlagBitsKHR::eProtected ) |
                 VkFlags( SwapchainCreateFlagBitsKHR::eMutableFormat )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SwapchainCreateFlagsKHR operator|( SwapchainCreateFlagBitsKHR bit0,
                                                                            SwapchainCreateFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SwapchainCreateFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SwapchainCreateFlagsKHR operator&( SwapchainCreateFlagBitsKHR bit0,
                                                                            SwapchainCreateFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SwapchainCreateFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SwapchainCreateFlagsKHR operator^( SwapchainCreateFlagBitsKHR bit0,
                                                                            SwapchainCreateFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SwapchainCreateFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SwapchainCreateFlagsKHR operator~( SwapchainCreateFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( SwapchainCreateFlagsKHR( bits ) );
  }

  using DeviceGroupPresentModeFlagsKHR = Flags<DeviceGroupPresentModeFlagBitsKHR>;

  template <>
  struct FlagTraits<DeviceGroupPresentModeFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( DeviceGroupPresentModeFlagBitsKHR::eLocal ) | VkFlags( DeviceGroupPresentModeFlagBitsKHR::eRemote ) |
                 VkFlags( DeviceGroupPresentModeFlagBitsKHR::eSum ) | VkFlags( DeviceGroupPresentModeFlagBitsKHR::eLocalMultiDevice )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceGroupPresentModeFlagsKHR operator|( DeviceGroupPresentModeFlagBitsKHR bit0,
                                                                                   DeviceGroupPresentModeFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DeviceGroupPresentModeFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceGroupPresentModeFlagsKHR operator&( DeviceGroupPresentModeFlagBitsKHR bit0,
                                                                                   DeviceGroupPresentModeFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DeviceGroupPresentModeFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceGroupPresentModeFlagsKHR operator^( DeviceGroupPresentModeFlagBitsKHR bit0,
                                                                                   DeviceGroupPresentModeFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DeviceGroupPresentModeFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceGroupPresentModeFlagsKHR operator~( DeviceGroupPresentModeFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( DeviceGroupPresentModeFlagsKHR( bits ) );
  }

  //=== VK_KHR_display ===

  using DisplayModeCreateFlagsKHR = Flags<DisplayModeCreateFlagBitsKHR>;

  using DisplayPlaneAlphaFlagsKHR = Flags<DisplayPlaneAlphaFlagBitsKHR>;

  template <>
  struct FlagTraits<DisplayPlaneAlphaFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( DisplayPlaneAlphaFlagBitsKHR::eOpaque ) | VkFlags( DisplayPlaneAlphaFlagBitsKHR::eGlobal ) |
                 VkFlags( DisplayPlaneAlphaFlagBitsKHR::ePerPixel ) | VkFlags( DisplayPlaneAlphaFlagBitsKHR::ePerPixelPremultiplied )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DisplayPlaneAlphaFlagsKHR operator|( DisplayPlaneAlphaFlagBitsKHR bit0,
                                                                              DisplayPlaneAlphaFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DisplayPlaneAlphaFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DisplayPlaneAlphaFlagsKHR operator&( DisplayPlaneAlphaFlagBitsKHR bit0,
                                                                              DisplayPlaneAlphaFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DisplayPlaneAlphaFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DisplayPlaneAlphaFlagsKHR operator^( DisplayPlaneAlphaFlagBitsKHR bit0,
                                                                              DisplayPlaneAlphaFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DisplayPlaneAlphaFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DisplayPlaneAlphaFlagsKHR operator~( DisplayPlaneAlphaFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( DisplayPlaneAlphaFlagsKHR( bits ) );
  }

  using DisplaySurfaceCreateFlagsKHR = Flags<DisplaySurfaceCreateFlagBitsKHR>;

  using SurfaceTransformFlagsKHR = Flags<SurfaceTransformFlagBitsKHR>;

  template <>
  struct FlagTraits<SurfaceTransformFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( SurfaceTransformFlagBitsKHR::eIdentity ) | VkFlags( SurfaceTransformFlagBitsKHR::eRotate90 ) |
                 VkFlags( SurfaceTransformFlagBitsKHR::eRotate180 ) | VkFlags( SurfaceTransformFlagBitsKHR::eRotate270 ) |
                 VkFlags( SurfaceTransformFlagBitsKHR::eHorizontalMirror ) | VkFlags( SurfaceTransformFlagBitsKHR::eHorizontalMirrorRotate90 ) |
                 VkFlags( SurfaceTransformFlagBitsKHR::eHorizontalMirrorRotate180 ) | VkFlags( SurfaceTransformFlagBitsKHR::eHorizontalMirrorRotate270 ) |
                 VkFlags( SurfaceTransformFlagBitsKHR::eInherit )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SurfaceTransformFlagsKHR operator|( SurfaceTransformFlagBitsKHR bit0,
                                                                             SurfaceTransformFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SurfaceTransformFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SurfaceTransformFlagsKHR operator&( SurfaceTransformFlagBitsKHR bit0,
                                                                             SurfaceTransformFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SurfaceTransformFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SurfaceTransformFlagsKHR operator^( SurfaceTransformFlagBitsKHR bit0,
                                                                             SurfaceTransformFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SurfaceTransformFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SurfaceTransformFlagsKHR operator~( SurfaceTransformFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( SurfaceTransformFlagsKHR( bits ) );
  }

#if defined( VK_USE_PLATFORM_XLIB_KHR )
  //=== VK_KHR_xlib_surface ===

  using XlibSurfaceCreateFlagsKHR = Flags<XlibSurfaceCreateFlagBitsKHR>;

#endif /*VK_USE_PLATFORM_XLIB_KHR*/

#if defined( VK_USE_PLATFORM_XCB_KHR )
  //=== VK_KHR_xcb_surface ===

  using XcbSurfaceCreateFlagsKHR = Flags<XcbSurfaceCreateFlagBitsKHR>;

#endif /*VK_USE_PLATFORM_XCB_KHR*/

#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
  //=== VK_KHR_wayland_surface ===

  using WaylandSurfaceCreateFlagsKHR = Flags<WaylandSurfaceCreateFlagBitsKHR>;

#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  //=== VK_KHR_android_surface ===

  using AndroidSurfaceCreateFlagsKHR = Flags<AndroidSurfaceCreateFlagBitsKHR>;

#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_win32_surface ===

  using Win32SurfaceCreateFlagsKHR = Flags<Win32SurfaceCreateFlagBitsKHR>;

#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_EXT_debug_report ===

  using DebugReportFlagsEXT = Flags<DebugReportFlagBitsEXT>;

  template <>
  struct FlagTraits<DebugReportFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( DebugReportFlagBitsEXT::eInformation ) | VkFlags( DebugReportFlagBitsEXT::eWarning ) |
                 VkFlags( DebugReportFlagBitsEXT::ePerformanceWarning ) | VkFlags( DebugReportFlagBitsEXT::eError ) | VkFlags( DebugReportFlagBitsEXT::eDebug )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DebugReportFlagsEXT operator|( DebugReportFlagBitsEXT bit0, DebugReportFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DebugReportFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DebugReportFlagsEXT operator&( DebugReportFlagBitsEXT bit0, DebugReportFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DebugReportFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DebugReportFlagsEXT operator^( DebugReportFlagBitsEXT bit0, DebugReportFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DebugReportFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DebugReportFlagsEXT operator~( DebugReportFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( DebugReportFlagsEXT( bits ) );
  }

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_KHR_video_queue ===

  using VideoCodecOperationFlagsKHR = Flags<VideoCodecOperationFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoCodecOperationFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoCodecOperationFlagBitsKHR::eNone )
#  if defined( VK_ENABLE_BETA_EXTENSIONS )
                 | VkFlags( VideoCodecOperationFlagBitsKHR::eEncodeH264EXT ) | VkFlags( VideoCodecOperationFlagBitsKHR::eEncodeH265EXT ) |
                 VkFlags( VideoCodecOperationFlagBitsKHR::eDecodeH264EXT ) | VkFlags( VideoCodecOperationFlagBitsKHR::eDecodeH265EXT )
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCodecOperationFlagsKHR operator|( VideoCodecOperationFlagBitsKHR bit0,
                                                                                VideoCodecOperationFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoCodecOperationFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCodecOperationFlagsKHR operator&( VideoCodecOperationFlagBitsKHR bit0,
                                                                                VideoCodecOperationFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoCodecOperationFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCodecOperationFlagsKHR operator^( VideoCodecOperationFlagBitsKHR bit0,
                                                                                VideoCodecOperationFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoCodecOperationFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCodecOperationFlagsKHR operator~( VideoCodecOperationFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoCodecOperationFlagsKHR( bits ) );
  }

  using VideoChromaSubsamplingFlagsKHR = Flags<VideoChromaSubsamplingFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoChromaSubsamplingFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoChromaSubsamplingFlagBitsKHR::eInvalid ) | VkFlags( VideoChromaSubsamplingFlagBitsKHR::eMonochrome ) |
                 VkFlags( VideoChromaSubsamplingFlagBitsKHR::e420 ) | VkFlags( VideoChromaSubsamplingFlagBitsKHR::e422 ) |
                 VkFlags( VideoChromaSubsamplingFlagBitsKHR::e444 )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoChromaSubsamplingFlagsKHR operator|( VideoChromaSubsamplingFlagBitsKHR bit0,
                                                                                   VideoChromaSubsamplingFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoChromaSubsamplingFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoChromaSubsamplingFlagsKHR operator&( VideoChromaSubsamplingFlagBitsKHR bit0,
                                                                                   VideoChromaSubsamplingFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoChromaSubsamplingFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoChromaSubsamplingFlagsKHR operator^( VideoChromaSubsamplingFlagBitsKHR bit0,
                                                                                   VideoChromaSubsamplingFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoChromaSubsamplingFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoChromaSubsamplingFlagsKHR operator~( VideoChromaSubsamplingFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoChromaSubsamplingFlagsKHR( bits ) );
  }

  using VideoComponentBitDepthFlagsKHR = Flags<VideoComponentBitDepthFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoComponentBitDepthFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoComponentBitDepthFlagBitsKHR::eInvalid ) | VkFlags( VideoComponentBitDepthFlagBitsKHR::e8 ) |
                 VkFlags( VideoComponentBitDepthFlagBitsKHR::e10 ) | VkFlags( VideoComponentBitDepthFlagBitsKHR::e12 )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoComponentBitDepthFlagsKHR operator|( VideoComponentBitDepthFlagBitsKHR bit0,
                                                                                   VideoComponentBitDepthFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoComponentBitDepthFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoComponentBitDepthFlagsKHR operator&( VideoComponentBitDepthFlagBitsKHR bit0,
                                                                                   VideoComponentBitDepthFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoComponentBitDepthFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoComponentBitDepthFlagsKHR operator^( VideoComponentBitDepthFlagBitsKHR bit0,
                                                                                   VideoComponentBitDepthFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoComponentBitDepthFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoComponentBitDepthFlagsKHR operator~( VideoComponentBitDepthFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoComponentBitDepthFlagsKHR( bits ) );
  }

  using VideoCapabilityFlagsKHR = Flags<VideoCapabilityFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoCapabilityFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoCapabilityFlagBitsKHR::eProtectedContent ) | VkFlags( VideoCapabilityFlagBitsKHR::eSeparateReferenceImages )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCapabilityFlagsKHR operator|( VideoCapabilityFlagBitsKHR bit0,
                                                                            VideoCapabilityFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoCapabilityFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCapabilityFlagsKHR operator&( VideoCapabilityFlagBitsKHR bit0,
                                                                            VideoCapabilityFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoCapabilityFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCapabilityFlagsKHR operator^( VideoCapabilityFlagBitsKHR bit0,
                                                                            VideoCapabilityFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoCapabilityFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCapabilityFlagsKHR operator~( VideoCapabilityFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoCapabilityFlagsKHR( bits ) );
  }

  using VideoSessionCreateFlagsKHR = Flags<VideoSessionCreateFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoSessionCreateFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoSessionCreateFlagBitsKHR::eProtectedContent )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoSessionCreateFlagsKHR operator|( VideoSessionCreateFlagBitsKHR bit0,
                                                                               VideoSessionCreateFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoSessionCreateFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoSessionCreateFlagsKHR operator&( VideoSessionCreateFlagBitsKHR bit0,
                                                                               VideoSessionCreateFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoSessionCreateFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoSessionCreateFlagsKHR operator^( VideoSessionCreateFlagBitsKHR bit0,
                                                                               VideoSessionCreateFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoSessionCreateFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoSessionCreateFlagsKHR operator~( VideoSessionCreateFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoSessionCreateFlagsKHR( bits ) );
  }

  using VideoSessionParametersCreateFlagsKHR = Flags<VideoSessionParametersCreateFlagBitsKHR>;

  using VideoBeginCodingFlagsKHR = Flags<VideoBeginCodingFlagBitsKHR>;

  using VideoEndCodingFlagsKHR = Flags<VideoEndCodingFlagBitsKHR>;

  using VideoCodingControlFlagsKHR = Flags<VideoCodingControlFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoCodingControlFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoCodingControlFlagBitsKHR::eReset )
#  if defined( VK_ENABLE_BETA_EXTENSIONS )
                 | VkFlags( VideoCodingControlFlagBitsKHR::eEncodeRateControl ) | VkFlags( VideoCodingControlFlagBitsKHR::eEncodeRateControlLayer )
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCodingControlFlagsKHR operator|( VideoCodingControlFlagBitsKHR bit0,
                                                                               VideoCodingControlFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoCodingControlFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCodingControlFlagsKHR operator&( VideoCodingControlFlagBitsKHR bit0,
                                                                               VideoCodingControlFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoCodingControlFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCodingControlFlagsKHR operator^( VideoCodingControlFlagBitsKHR bit0,
                                                                               VideoCodingControlFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoCodingControlFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoCodingControlFlagsKHR operator~( VideoCodingControlFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoCodingControlFlagsKHR( bits ) );
  }

#endif /*VK_ENABLE_BETA_EXTENSIONS*/

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_KHR_video_decode_queue ===

  using VideoDecodeCapabilityFlagsKHR = Flags<VideoDecodeCapabilityFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoDecodeCapabilityFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoDecodeCapabilityFlagBitsKHR::eDpbAndOutputCoincide ) | VkFlags( VideoDecodeCapabilityFlagBitsKHR::eDpbAndOutputDistinct )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoDecodeCapabilityFlagsKHR operator|( VideoDecodeCapabilityFlagBitsKHR bit0,
                                                                                  VideoDecodeCapabilityFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoDecodeCapabilityFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoDecodeCapabilityFlagsKHR operator&( VideoDecodeCapabilityFlagBitsKHR bit0,
                                                                                  VideoDecodeCapabilityFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoDecodeCapabilityFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoDecodeCapabilityFlagsKHR operator^( VideoDecodeCapabilityFlagBitsKHR bit0,
                                                                                  VideoDecodeCapabilityFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoDecodeCapabilityFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoDecodeCapabilityFlagsKHR operator~( VideoDecodeCapabilityFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoDecodeCapabilityFlagsKHR( bits ) );
  }

  using VideoDecodeUsageFlagsKHR = Flags<VideoDecodeUsageFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoDecodeUsageFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoDecodeUsageFlagBitsKHR::eDefault ) | VkFlags( VideoDecodeUsageFlagBitsKHR::eTranscoding ) |
                 VkFlags( VideoDecodeUsageFlagBitsKHR::eOffline ) | VkFlags( VideoDecodeUsageFlagBitsKHR::eStreaming )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoDecodeUsageFlagsKHR operator|( VideoDecodeUsageFlagBitsKHR bit0,
                                                                             VideoDecodeUsageFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoDecodeUsageFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoDecodeUsageFlagsKHR operator&( VideoDecodeUsageFlagBitsKHR bit0,
                                                                             VideoDecodeUsageFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoDecodeUsageFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoDecodeUsageFlagsKHR operator^( VideoDecodeUsageFlagBitsKHR bit0,
                                                                             VideoDecodeUsageFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoDecodeUsageFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoDecodeUsageFlagsKHR operator~( VideoDecodeUsageFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoDecodeUsageFlagsKHR( bits ) );
  }

  using VideoDecodeFlagsKHR = Flags<VideoDecodeFlagBitsKHR>;

#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_EXT_transform_feedback ===

  using PipelineRasterizationStateStreamCreateFlagsEXT = Flags<PipelineRasterizationStateStreamCreateFlagBitsEXT>;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_EXT_video_encode_h264 ===

  using VideoEncodeH264CapabilityFlagsEXT = Flags<VideoEncodeH264CapabilityFlagBitsEXT>;

  template <>
  struct FlagTraits<VideoEncodeH264CapabilityFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags =
        VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::eDirect8X8InferenceEnabled ) |
        VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::eDirect8X8InferenceDisabled ) | VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::eSeparateColourPlane ) |
        VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::eQpprimeYZeroTransformBypass ) | VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::eScalingLists ) |
        VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::eHrdCompliance ) | VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::eChromaQpOffset ) |
        VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::eSecondChromaQpOffset ) | VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::ePicInitQpMinus26 ) |
        VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::eWeightedPred ) | VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::eWeightedBipredExplicit ) |
        VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::eWeightedBipredImplicit ) | VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::eWeightedPredNoTable ) |
        VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::eTransform8X8 ) | VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::eCabac ) |
        VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::eCavlc ) | VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::eDeblockingFilterDisabled ) |
        VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::eDeblockingFilterEnabled ) | VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::eDeblockingFilterPartial ) |
        VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::eDisableDirectSpatialMvPred ) | VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::eMultipleSlicePerFrame ) |
        VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::eSliceMbCount ) | VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::eRowUnalignedSlice ) |
        VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::eDifferentSliceType ) | VkFlags( VideoEncodeH264CapabilityFlagBitsEXT::eBFrameInL1List )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264CapabilityFlagsEXT operator|( VideoEncodeH264CapabilityFlagBitsEXT bit0,
                                                                                      VideoEncodeH264CapabilityFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH264CapabilityFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264CapabilityFlagsEXT operator&( VideoEncodeH264CapabilityFlagBitsEXT bit0,
                                                                                      VideoEncodeH264CapabilityFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH264CapabilityFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264CapabilityFlagsEXT operator^( VideoEncodeH264CapabilityFlagBitsEXT bit0,
                                                                                      VideoEncodeH264CapabilityFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH264CapabilityFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264CapabilityFlagsEXT operator~( VideoEncodeH264CapabilityFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoEncodeH264CapabilityFlagsEXT( bits ) );
  }

  using VideoEncodeH264InputModeFlagsEXT = Flags<VideoEncodeH264InputModeFlagBitsEXT>;

  template <>
  struct FlagTraits<VideoEncodeH264InputModeFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoEncodeH264InputModeFlagBitsEXT::eFrame ) | VkFlags( VideoEncodeH264InputModeFlagBitsEXT::eSlice ) |
                 VkFlags( VideoEncodeH264InputModeFlagBitsEXT::eNonVcl )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264InputModeFlagsEXT operator|( VideoEncodeH264InputModeFlagBitsEXT bit0,
                                                                                     VideoEncodeH264InputModeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH264InputModeFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264InputModeFlagsEXT operator&( VideoEncodeH264InputModeFlagBitsEXT bit0,
                                                                                     VideoEncodeH264InputModeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH264InputModeFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264InputModeFlagsEXT operator^( VideoEncodeH264InputModeFlagBitsEXT bit0,
                                                                                     VideoEncodeH264InputModeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH264InputModeFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264InputModeFlagsEXT operator~( VideoEncodeH264InputModeFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoEncodeH264InputModeFlagsEXT( bits ) );
  }

  using VideoEncodeH264OutputModeFlagsEXT = Flags<VideoEncodeH264OutputModeFlagBitsEXT>;

  template <>
  struct FlagTraits<VideoEncodeH264OutputModeFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoEncodeH264OutputModeFlagBitsEXT::eFrame ) | VkFlags( VideoEncodeH264OutputModeFlagBitsEXT::eSlice ) |
                 VkFlags( VideoEncodeH264OutputModeFlagBitsEXT::eNonVcl )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264OutputModeFlagsEXT operator|( VideoEncodeH264OutputModeFlagBitsEXT bit0,
                                                                                      VideoEncodeH264OutputModeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH264OutputModeFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264OutputModeFlagsEXT operator&( VideoEncodeH264OutputModeFlagBitsEXT bit0,
                                                                                      VideoEncodeH264OutputModeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH264OutputModeFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264OutputModeFlagsEXT operator^( VideoEncodeH264OutputModeFlagBitsEXT bit0,
                                                                                      VideoEncodeH264OutputModeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH264OutputModeFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH264OutputModeFlagsEXT operator~( VideoEncodeH264OutputModeFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoEncodeH264OutputModeFlagsEXT( bits ) );
  }

#endif /*VK_ENABLE_BETA_EXTENSIONS*/

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_EXT_video_encode_h265 ===

  using VideoEncodeH265CapabilityFlagsEXT = Flags<VideoEncodeH265CapabilityFlagBitsEXT>;

  template <>
  struct FlagTraits<VideoEncodeH265CapabilityFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags =
        VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::eSeparateColourPlane ) | VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::eScalingLists ) |
        VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::eSampleAdaptiveOffsetEnabled ) | VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::ePcmEnable ) |
        VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::eSpsTemporalMvpEnabled ) | VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::eHrdCompliance ) |
        VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::eInitQpMinus26 ) | VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::eLog2ParallelMergeLevelMinus2 ) |
        VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::eSignDataHidingEnabled ) | VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::eTransformSkipEnabled ) |
        VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::eTransformSkipDisabled ) |
        VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::ePpsSliceChromaQpOffsetsPresent ) | VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::eWeightedPred ) |
        VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::eWeightedBipred ) | VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::eWeightedPredNoTable ) |
        VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::eTransquantBypassEnabled ) | VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::eEntropyCodingSyncEnabled ) |
        VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::eDeblockingFilterOverrideEnabled ) |
        VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::eMultipleTilePerFrame ) | VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::eMultipleSlicePerTile ) |
        VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::eMultipleTilePerSlice ) | VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::eSliceSegmentCtbCount ) |
        VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::eRowUnalignedSliceSegment ) | VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::eDependentSliceSegment ) |
        VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::eDifferentSliceType ) | VkFlags( VideoEncodeH265CapabilityFlagBitsEXT::eBFrameInL1List )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH265CapabilityFlagsEXT operator|( VideoEncodeH265CapabilityFlagBitsEXT bit0,
                                                                                      VideoEncodeH265CapabilityFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH265CapabilityFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH265CapabilityFlagsEXT operator&( VideoEncodeH265CapabilityFlagBitsEXT bit0,
                                                                                      VideoEncodeH265CapabilityFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH265CapabilityFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH265CapabilityFlagsEXT operator^( VideoEncodeH265CapabilityFlagBitsEXT bit0,
                                                                                      VideoEncodeH265CapabilityFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH265CapabilityFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH265CapabilityFlagsEXT operator~( VideoEncodeH265CapabilityFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoEncodeH265CapabilityFlagsEXT( bits ) );
  }

  using VideoEncodeH265InputModeFlagsEXT = Flags<VideoEncodeH265InputModeFlagBitsEXT>;

  template <>
  struct FlagTraits<VideoEncodeH265InputModeFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoEncodeH265InputModeFlagBitsEXT::eFrame ) | VkFlags( VideoEncodeH265InputModeFlagBitsEXT::eSliceSegment ) |
                 VkFlags( VideoEncodeH265InputModeFlagBitsEXT::eNonVcl )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH265InputModeFlagsEXT operator|( VideoEncodeH265InputModeFlagBitsEXT bit0,
                                                                                     VideoEncodeH265InputModeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH265InputModeFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH265InputModeFlagsEXT operator&( VideoEncodeH265InputModeFlagBitsEXT bit0,
                                                                                     VideoEncodeH265InputModeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH265InputModeFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH265InputModeFlagsEXT operator^( VideoEncodeH265InputModeFlagBitsEXT bit0,
                                                                                     VideoEncodeH265InputModeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH265InputModeFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH265InputModeFlagsEXT operator~( VideoEncodeH265InputModeFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoEncodeH265InputModeFlagsEXT( bits ) );
  }

  using VideoEncodeH265OutputModeFlagsEXT = Flags<VideoEncodeH265OutputModeFlagBitsEXT>;

  template <>
  struct FlagTraits<VideoEncodeH265OutputModeFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoEncodeH265OutputModeFlagBitsEXT::eFrame ) | VkFlags( VideoEncodeH265OutputModeFlagBitsEXT::eSliceSegment ) |
                 VkFlags( VideoEncodeH265OutputModeFlagBitsEXT::eNonVcl )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH265OutputModeFlagsEXT operator|( VideoEncodeH265OutputModeFlagBitsEXT bit0,
                                                                                      VideoEncodeH265OutputModeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH265OutputModeFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH265OutputModeFlagsEXT operator&( VideoEncodeH265OutputModeFlagBitsEXT bit0,
                                                                                      VideoEncodeH265OutputModeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH265OutputModeFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH265OutputModeFlagsEXT operator^( VideoEncodeH265OutputModeFlagBitsEXT bit0,
                                                                                      VideoEncodeH265OutputModeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH265OutputModeFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH265OutputModeFlagsEXT operator~( VideoEncodeH265OutputModeFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoEncodeH265OutputModeFlagsEXT( bits ) );
  }

  using VideoEncodeH265CtbSizeFlagsEXT = Flags<VideoEncodeH265CtbSizeFlagBitsEXT>;

  template <>
  struct FlagTraits<VideoEncodeH265CtbSizeFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoEncodeH265CtbSizeFlagBitsEXT::e16 ) | VkFlags( VideoEncodeH265CtbSizeFlagBitsEXT::e32 ) |
                 VkFlags( VideoEncodeH265CtbSizeFlagBitsEXT::e64 )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH265CtbSizeFlagsEXT operator|( VideoEncodeH265CtbSizeFlagBitsEXT bit0,
                                                                                   VideoEncodeH265CtbSizeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH265CtbSizeFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH265CtbSizeFlagsEXT operator&( VideoEncodeH265CtbSizeFlagBitsEXT bit0,
                                                                                   VideoEncodeH265CtbSizeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH265CtbSizeFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH265CtbSizeFlagsEXT operator^( VideoEncodeH265CtbSizeFlagBitsEXT bit0,
                                                                                   VideoEncodeH265CtbSizeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH265CtbSizeFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH265CtbSizeFlagsEXT operator~( VideoEncodeH265CtbSizeFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoEncodeH265CtbSizeFlagsEXT( bits ) );
  }

  using VideoEncodeH265TransformBlockSizeFlagsEXT = Flags<VideoEncodeH265TransformBlockSizeFlagBitsEXT>;

  template <>
  struct FlagTraits<VideoEncodeH265TransformBlockSizeFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoEncodeH265TransformBlockSizeFlagBitsEXT::e4 ) | VkFlags( VideoEncodeH265TransformBlockSizeFlagBitsEXT::e8 ) |
                 VkFlags( VideoEncodeH265TransformBlockSizeFlagBitsEXT::e16 ) | VkFlags( VideoEncodeH265TransformBlockSizeFlagBitsEXT::e32 )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH265TransformBlockSizeFlagsEXT
    operator|( VideoEncodeH265TransformBlockSizeFlagBitsEXT bit0, VideoEncodeH265TransformBlockSizeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH265TransformBlockSizeFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH265TransformBlockSizeFlagsEXT
    operator&( VideoEncodeH265TransformBlockSizeFlagBitsEXT bit0, VideoEncodeH265TransformBlockSizeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH265TransformBlockSizeFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH265TransformBlockSizeFlagsEXT
    operator^( VideoEncodeH265TransformBlockSizeFlagBitsEXT bit0, VideoEncodeH265TransformBlockSizeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeH265TransformBlockSizeFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeH265TransformBlockSizeFlagsEXT operator~( VideoEncodeH265TransformBlockSizeFlagBitsEXT bits )
    VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoEncodeH265TransformBlockSizeFlagsEXT( bits ) );
  }

#endif /*VK_ENABLE_BETA_EXTENSIONS*/

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_EXT_video_decode_h264 ===

  using VideoDecodeH264PictureLayoutFlagsEXT = Flags<VideoDecodeH264PictureLayoutFlagBitsEXT>;

  template <>
  struct FlagTraits<VideoDecodeH264PictureLayoutFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoDecodeH264PictureLayoutFlagBitsEXT::eProgressive ) |
                 VkFlags( VideoDecodeH264PictureLayoutFlagBitsEXT::eInterlacedInterleavedLines ) |
                 VkFlags( VideoDecodeH264PictureLayoutFlagBitsEXT::eInterlacedSeparatePlanes )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoDecodeH264PictureLayoutFlagsEXT operator|( VideoDecodeH264PictureLayoutFlagBitsEXT bit0,
                                                                                         VideoDecodeH264PictureLayoutFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoDecodeH264PictureLayoutFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoDecodeH264PictureLayoutFlagsEXT operator&( VideoDecodeH264PictureLayoutFlagBitsEXT bit0,
                                                                                         VideoDecodeH264PictureLayoutFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoDecodeH264PictureLayoutFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoDecodeH264PictureLayoutFlagsEXT operator^( VideoDecodeH264PictureLayoutFlagBitsEXT bit0,
                                                                                         VideoDecodeH264PictureLayoutFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoDecodeH264PictureLayoutFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoDecodeH264PictureLayoutFlagsEXT operator~( VideoDecodeH264PictureLayoutFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoDecodeH264PictureLayoutFlagsEXT( bits ) );
  }

#endif /*VK_ENABLE_BETA_EXTENSIONS*/

#if defined( VK_USE_PLATFORM_GGP )
  //=== VK_GGP_stream_descriptor_surface ===

  using StreamDescriptorSurfaceCreateFlagsGGP = Flags<StreamDescriptorSurfaceCreateFlagBitsGGP>;

#endif /*VK_USE_PLATFORM_GGP*/

  //=== VK_NV_external_memory_capabilities ===

  using ExternalMemoryHandleTypeFlagsNV = Flags<ExternalMemoryHandleTypeFlagBitsNV>;

  template <>
  struct FlagTraits<ExternalMemoryHandleTypeFlagBitsNV>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ExternalMemoryHandleTypeFlagBitsNV::eOpaqueWin32 ) | VkFlags( ExternalMemoryHandleTypeFlagBitsNV::eOpaqueWin32Kmt ) |
                 VkFlags( ExternalMemoryHandleTypeFlagBitsNV::eD3D11Image ) | VkFlags( ExternalMemoryHandleTypeFlagBitsNV::eD3D11ImageKmt )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryHandleTypeFlagsNV operator|( ExternalMemoryHandleTypeFlagBitsNV bit0,
                                                                                    ExternalMemoryHandleTypeFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalMemoryHandleTypeFlagsNV( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryHandleTypeFlagsNV operator&( ExternalMemoryHandleTypeFlagBitsNV bit0,
                                                                                    ExternalMemoryHandleTypeFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalMemoryHandleTypeFlagsNV( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryHandleTypeFlagsNV operator^( ExternalMemoryHandleTypeFlagBitsNV bit0,
                                                                                    ExternalMemoryHandleTypeFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalMemoryHandleTypeFlagsNV( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryHandleTypeFlagsNV operator~( ExternalMemoryHandleTypeFlagBitsNV bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ExternalMemoryHandleTypeFlagsNV( bits ) );
  }

  using ExternalMemoryFeatureFlagsNV = Flags<ExternalMemoryFeatureFlagBitsNV>;

  template <>
  struct FlagTraits<ExternalMemoryFeatureFlagBitsNV>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ExternalMemoryFeatureFlagBitsNV::eDedicatedOnly ) | VkFlags( ExternalMemoryFeatureFlagBitsNV::eExportable ) |
                 VkFlags( ExternalMemoryFeatureFlagBitsNV::eImportable )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryFeatureFlagsNV operator|( ExternalMemoryFeatureFlagBitsNV bit0,
                                                                                 ExternalMemoryFeatureFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalMemoryFeatureFlagsNV( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryFeatureFlagsNV operator&( ExternalMemoryFeatureFlagBitsNV bit0,
                                                                                 ExternalMemoryFeatureFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalMemoryFeatureFlagsNV( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryFeatureFlagsNV operator^( ExternalMemoryFeatureFlagBitsNV bit0,
                                                                                 ExternalMemoryFeatureFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExternalMemoryFeatureFlagsNV( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExternalMemoryFeatureFlagsNV operator~( ExternalMemoryFeatureFlagBitsNV bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ExternalMemoryFeatureFlagsNV( bits ) );
  }

#if defined( VK_USE_PLATFORM_VI_NN )
  //=== VK_NN_vi_surface ===

  using ViSurfaceCreateFlagsNN = Flags<ViSurfaceCreateFlagBitsNN>;

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

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ConditionalRenderingFlagsEXT operator|( ConditionalRenderingFlagBitsEXT bit0,
                                                                                 ConditionalRenderingFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ConditionalRenderingFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ConditionalRenderingFlagsEXT operator&( ConditionalRenderingFlagBitsEXT bit0,
                                                                                 ConditionalRenderingFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ConditionalRenderingFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ConditionalRenderingFlagsEXT operator^( ConditionalRenderingFlagBitsEXT bit0,
                                                                                 ConditionalRenderingFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ConditionalRenderingFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ConditionalRenderingFlagsEXT operator~( ConditionalRenderingFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ConditionalRenderingFlagsEXT( bits ) );
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

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SurfaceCounterFlagsEXT operator|( SurfaceCounterFlagBitsEXT bit0, SurfaceCounterFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SurfaceCounterFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SurfaceCounterFlagsEXT operator&( SurfaceCounterFlagBitsEXT bit0, SurfaceCounterFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SurfaceCounterFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SurfaceCounterFlagsEXT operator^( SurfaceCounterFlagBitsEXT bit0, SurfaceCounterFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return SurfaceCounterFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR SurfaceCounterFlagsEXT operator~( SurfaceCounterFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( SurfaceCounterFlagsEXT( bits ) );
  }

  //=== VK_NV_viewport_swizzle ===

  using PipelineViewportSwizzleStateCreateFlagsNV = Flags<PipelineViewportSwizzleStateCreateFlagBitsNV>;

  //=== VK_EXT_discard_rectangles ===

  using PipelineDiscardRectangleStateCreateFlagsEXT = Flags<PipelineDiscardRectangleStateCreateFlagBitsEXT>;

  //=== VK_EXT_conservative_rasterization ===

  using PipelineRasterizationConservativeStateCreateFlagsEXT = Flags<PipelineRasterizationConservativeStateCreateFlagBitsEXT>;

  //=== VK_EXT_depth_clip_enable ===

  using PipelineRasterizationDepthClipStateCreateFlagsEXT = Flags<PipelineRasterizationDepthClipStateCreateFlagBitsEXT>;

  //=== VK_KHR_performance_query ===

  using PerformanceCounterDescriptionFlagsKHR = Flags<PerformanceCounterDescriptionFlagBitsKHR>;

  template <>
  struct FlagTraits<PerformanceCounterDescriptionFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags =
        VkFlags( PerformanceCounterDescriptionFlagBitsKHR::ePerformanceImpacting ) | VkFlags( PerformanceCounterDescriptionFlagBitsKHR::eConcurrentlyImpacted )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PerformanceCounterDescriptionFlagsKHR operator|( PerformanceCounterDescriptionFlagBitsKHR bit0,
                                                                                          PerformanceCounterDescriptionFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PerformanceCounterDescriptionFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PerformanceCounterDescriptionFlagsKHR operator&( PerformanceCounterDescriptionFlagBitsKHR bit0,
                                                                                          PerformanceCounterDescriptionFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PerformanceCounterDescriptionFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PerformanceCounterDescriptionFlagsKHR operator^( PerformanceCounterDescriptionFlagBitsKHR bit0,
                                                                                          PerformanceCounterDescriptionFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return PerformanceCounterDescriptionFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR PerformanceCounterDescriptionFlagsKHR operator~( PerformanceCounterDescriptionFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( PerformanceCounterDescriptionFlagsKHR( bits ) );
  }

  using AcquireProfilingLockFlagsKHR = Flags<AcquireProfilingLockFlagBitsKHR>;

#if defined( VK_USE_PLATFORM_IOS_MVK )
  //=== VK_MVK_ios_surface ===

  using IOSSurfaceCreateFlagsMVK = Flags<IOSSurfaceCreateFlagBitsMVK>;

#endif /*VK_USE_PLATFORM_IOS_MVK*/

#if defined( VK_USE_PLATFORM_MACOS_MVK )
  //=== VK_MVK_macos_surface ===

  using MacOSSurfaceCreateFlagsMVK = Flags<MacOSSurfaceCreateFlagBitsMVK>;

#endif /*VK_USE_PLATFORM_MACOS_MVK*/

  //=== VK_EXT_debug_utils ===

  using DebugUtilsMessageSeverityFlagsEXT = Flags<DebugUtilsMessageSeverityFlagBitsEXT>;

  template <>
  struct FlagTraits<DebugUtilsMessageSeverityFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( DebugUtilsMessageSeverityFlagBitsEXT::eVerbose ) | VkFlags( DebugUtilsMessageSeverityFlagBitsEXT::eInfo ) |
                 VkFlags( DebugUtilsMessageSeverityFlagBitsEXT::eWarning ) | VkFlags( DebugUtilsMessageSeverityFlagBitsEXT::eError )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DebugUtilsMessageSeverityFlagsEXT operator|( DebugUtilsMessageSeverityFlagBitsEXT bit0,
                                                                                      DebugUtilsMessageSeverityFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DebugUtilsMessageSeverityFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DebugUtilsMessageSeverityFlagsEXT operator&( DebugUtilsMessageSeverityFlagBitsEXT bit0,
                                                                                      DebugUtilsMessageSeverityFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DebugUtilsMessageSeverityFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DebugUtilsMessageSeverityFlagsEXT operator^( DebugUtilsMessageSeverityFlagBitsEXT bit0,
                                                                                      DebugUtilsMessageSeverityFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DebugUtilsMessageSeverityFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DebugUtilsMessageSeverityFlagsEXT operator~( DebugUtilsMessageSeverityFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( DebugUtilsMessageSeverityFlagsEXT( bits ) );
  }

  using DebugUtilsMessageTypeFlagsEXT = Flags<DebugUtilsMessageTypeFlagBitsEXT>;

  template <>
  struct FlagTraits<DebugUtilsMessageTypeFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( DebugUtilsMessageTypeFlagBitsEXT::eGeneral ) | VkFlags( DebugUtilsMessageTypeFlagBitsEXT::eValidation ) |
                 VkFlags( DebugUtilsMessageTypeFlagBitsEXT::ePerformance ) | VkFlags( DebugUtilsMessageTypeFlagBitsEXT::eDeviceAddressBinding )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DebugUtilsMessageTypeFlagsEXT operator|( DebugUtilsMessageTypeFlagBitsEXT bit0,
                                                                                  DebugUtilsMessageTypeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DebugUtilsMessageTypeFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DebugUtilsMessageTypeFlagsEXT operator&( DebugUtilsMessageTypeFlagBitsEXT bit0,
                                                                                  DebugUtilsMessageTypeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DebugUtilsMessageTypeFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DebugUtilsMessageTypeFlagsEXT operator^( DebugUtilsMessageTypeFlagBitsEXT bit0,
                                                                                  DebugUtilsMessageTypeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DebugUtilsMessageTypeFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DebugUtilsMessageTypeFlagsEXT operator~( DebugUtilsMessageTypeFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( DebugUtilsMessageTypeFlagsEXT( bits ) );
  }

  using DebugUtilsMessengerCallbackDataFlagsEXT = Flags<DebugUtilsMessengerCallbackDataFlagBitsEXT>;

  using DebugUtilsMessengerCreateFlagsEXT = Flags<DebugUtilsMessengerCreateFlagBitsEXT>;

  //=== VK_NV_fragment_coverage_to_color ===

  using PipelineCoverageToColorStateCreateFlagsNV = Flags<PipelineCoverageToColorStateCreateFlagBitsNV>;

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

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR GeometryFlagsKHR operator|( GeometryFlagBitsKHR bit0, GeometryFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return GeometryFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR GeometryFlagsKHR operator&( GeometryFlagBitsKHR bit0, GeometryFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return GeometryFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR GeometryFlagsKHR operator^( GeometryFlagBitsKHR bit0, GeometryFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return GeometryFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR GeometryFlagsKHR operator~( GeometryFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( GeometryFlagsKHR( bits ) );
  }

  using GeometryFlagsNV = GeometryFlagsKHR;

  using GeometryInstanceFlagsKHR = Flags<GeometryInstanceFlagBitsKHR>;

  template <>
  struct FlagTraits<GeometryInstanceFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable ) | VkFlags( GeometryInstanceFlagBitsKHR::eTriangleFlipFacing ) |
                 VkFlags( GeometryInstanceFlagBitsKHR::eForceOpaque ) | VkFlags( GeometryInstanceFlagBitsKHR::eForceNoOpaque ) |
                 VkFlags( GeometryInstanceFlagBitsKHR::eForceOpacityMicromap2StateEXT ) | VkFlags( GeometryInstanceFlagBitsKHR::eDisableOpacityMicromapsEXT )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR GeometryInstanceFlagsKHR operator|( GeometryInstanceFlagBitsKHR bit0,
                                                                             GeometryInstanceFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return GeometryInstanceFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR GeometryInstanceFlagsKHR operator&( GeometryInstanceFlagBitsKHR bit0,
                                                                             GeometryInstanceFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return GeometryInstanceFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR GeometryInstanceFlagsKHR operator^( GeometryInstanceFlagBitsKHR bit0,
                                                                             GeometryInstanceFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return GeometryInstanceFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR GeometryInstanceFlagsKHR operator~( GeometryInstanceFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( GeometryInstanceFlagsKHR( bits ) );
  }

  using GeometryInstanceFlagsNV = GeometryInstanceFlagsKHR;

  using BuildAccelerationStructureFlagsKHR = Flags<BuildAccelerationStructureFlagBitsKHR>;

  template <>
  struct FlagTraits<BuildAccelerationStructureFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( BuildAccelerationStructureFlagBitsKHR::eAllowUpdate ) | VkFlags( BuildAccelerationStructureFlagBitsKHR::eAllowCompaction ) |
                 VkFlags( BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace ) | VkFlags( BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild ) |
                 VkFlags( BuildAccelerationStructureFlagBitsKHR::eLowMemory ) | VkFlags( BuildAccelerationStructureFlagBitsKHR::eMotionNV ) |
                 VkFlags( BuildAccelerationStructureFlagBitsKHR::eAllowOpacityMicromapUpdateEXT ) |
                 VkFlags( BuildAccelerationStructureFlagBitsKHR::eAllowDisableOpacityMicromapsEXT ) |
                 VkFlags( BuildAccelerationStructureFlagBitsKHR::eAllowOpacityMicromapDataUpdateEXT )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BuildAccelerationStructureFlagsKHR operator|( BuildAccelerationStructureFlagBitsKHR bit0,
                                                                                       BuildAccelerationStructureFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return BuildAccelerationStructureFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BuildAccelerationStructureFlagsKHR operator&( BuildAccelerationStructureFlagBitsKHR bit0,
                                                                                       BuildAccelerationStructureFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return BuildAccelerationStructureFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BuildAccelerationStructureFlagsKHR operator^( BuildAccelerationStructureFlagBitsKHR bit0,
                                                                                       BuildAccelerationStructureFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return BuildAccelerationStructureFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BuildAccelerationStructureFlagsKHR operator~( BuildAccelerationStructureFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( BuildAccelerationStructureFlagsKHR( bits ) );
  }

  using BuildAccelerationStructureFlagsNV = BuildAccelerationStructureFlagsKHR;

  using AccelerationStructureCreateFlagsKHR = Flags<AccelerationStructureCreateFlagBitsKHR>;

  template <>
  struct FlagTraits<AccelerationStructureCreateFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( AccelerationStructureCreateFlagBitsKHR::eDeviceAddressCaptureReplay ) | VkFlags( AccelerationStructureCreateFlagBitsKHR::eMotionNV )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AccelerationStructureCreateFlagsKHR operator|( AccelerationStructureCreateFlagBitsKHR bit0,
                                                                                        AccelerationStructureCreateFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return AccelerationStructureCreateFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AccelerationStructureCreateFlagsKHR operator&( AccelerationStructureCreateFlagBitsKHR bit0,
                                                                                        AccelerationStructureCreateFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return AccelerationStructureCreateFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AccelerationStructureCreateFlagsKHR operator^( AccelerationStructureCreateFlagBitsKHR bit0,
                                                                                        AccelerationStructureCreateFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return AccelerationStructureCreateFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR AccelerationStructureCreateFlagsKHR operator~( AccelerationStructureCreateFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( AccelerationStructureCreateFlagsKHR( bits ) );
  }

  //=== VK_NV_framebuffer_mixed_samples ===

  using PipelineCoverageModulationStateCreateFlagsNV = Flags<PipelineCoverageModulationStateCreateFlagBitsNV>;

  //=== VK_EXT_validation_cache ===

  using ValidationCacheCreateFlagsEXT = Flags<ValidationCacheCreateFlagBitsEXT>;

  //=== VK_AMD_pipeline_compiler_control ===

  using PipelineCompilerControlFlagsAMD = Flags<PipelineCompilerControlFlagBitsAMD>;

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_imagepipe_surface ===

  using ImagePipeSurfaceCreateFlagsFUCHSIA = Flags<ImagePipeSurfaceCreateFlagBitsFUCHSIA>;

#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_surface ===

  using MetalSurfaceCreateFlagsEXT = Flags<MetalSurfaceCreateFlagBitsEXT>;

#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_AMD_shader_core_properties2 ===

  using ShaderCorePropertiesFlagsAMD = Flags<ShaderCorePropertiesFlagBitsAMD>;

  //=== VK_NV_coverage_reduction_mode ===

  using PipelineCoverageReductionStateCreateFlagsNV = Flags<PipelineCoverageReductionStateCreateFlagBitsNV>;

  //=== VK_EXT_headless_surface ===

  using HeadlessSurfaceCreateFlagsEXT = Flags<HeadlessSurfaceCreateFlagBitsEXT>;

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

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR IndirectStateFlagsNV operator|( IndirectStateFlagBitsNV bit0, IndirectStateFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return IndirectStateFlagsNV( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR IndirectStateFlagsNV operator&( IndirectStateFlagBitsNV bit0, IndirectStateFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return IndirectStateFlagsNV( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR IndirectStateFlagsNV operator^( IndirectStateFlagBitsNV bit0, IndirectStateFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return IndirectStateFlagsNV( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR IndirectStateFlagsNV operator~( IndirectStateFlagBitsNV bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( IndirectStateFlagsNV( bits ) );
  }

  using IndirectCommandsLayoutUsageFlagsNV = Flags<IndirectCommandsLayoutUsageFlagBitsNV>;

  template <>
  struct FlagTraits<IndirectCommandsLayoutUsageFlagBitsNV>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( IndirectCommandsLayoutUsageFlagBitsNV::eExplicitPreprocess ) | VkFlags( IndirectCommandsLayoutUsageFlagBitsNV::eIndexedSequences ) |
                 VkFlags( IndirectCommandsLayoutUsageFlagBitsNV::eUnorderedSequences )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR IndirectCommandsLayoutUsageFlagsNV operator|( IndirectCommandsLayoutUsageFlagBitsNV bit0,
                                                                                       IndirectCommandsLayoutUsageFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return IndirectCommandsLayoutUsageFlagsNV( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR IndirectCommandsLayoutUsageFlagsNV operator&( IndirectCommandsLayoutUsageFlagBitsNV bit0,
                                                                                       IndirectCommandsLayoutUsageFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return IndirectCommandsLayoutUsageFlagsNV( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR IndirectCommandsLayoutUsageFlagsNV operator^( IndirectCommandsLayoutUsageFlagBitsNV bit0,
                                                                                       IndirectCommandsLayoutUsageFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return IndirectCommandsLayoutUsageFlagsNV( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR IndirectCommandsLayoutUsageFlagsNV operator~( IndirectCommandsLayoutUsageFlagBitsNV bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( IndirectCommandsLayoutUsageFlagsNV( bits ) );
  }

  //=== VK_EXT_device_memory_report ===

  using DeviceMemoryReportFlagsEXT = Flags<DeviceMemoryReportFlagBitsEXT>;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_KHR_video_encode_queue ===

  using VideoEncodeFlagsKHR = Flags<VideoEncodeFlagBitsKHR>;

  using VideoEncodeCapabilityFlagsKHR = Flags<VideoEncodeCapabilityFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeCapabilityFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoEncodeCapabilityFlagBitsKHR::ePrecedingExternallyEncodedBytes )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeCapabilityFlagsKHR operator|( VideoEncodeCapabilityFlagBitsKHR bit0,
                                                                                  VideoEncodeCapabilityFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeCapabilityFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeCapabilityFlagsKHR operator&( VideoEncodeCapabilityFlagBitsKHR bit0,
                                                                                  VideoEncodeCapabilityFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeCapabilityFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeCapabilityFlagsKHR operator^( VideoEncodeCapabilityFlagBitsKHR bit0,
                                                                                  VideoEncodeCapabilityFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeCapabilityFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeCapabilityFlagsKHR operator~( VideoEncodeCapabilityFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoEncodeCapabilityFlagsKHR( bits ) );
  }

  using VideoEncodeUsageFlagsKHR = Flags<VideoEncodeUsageFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeUsageFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoEncodeUsageFlagBitsKHR::eDefault ) | VkFlags( VideoEncodeUsageFlagBitsKHR::eTranscoding ) |
                 VkFlags( VideoEncodeUsageFlagBitsKHR::eStreaming ) | VkFlags( VideoEncodeUsageFlagBitsKHR::eRecording ) |
                 VkFlags( VideoEncodeUsageFlagBitsKHR::eConferencing )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeUsageFlagsKHR operator|( VideoEncodeUsageFlagBitsKHR bit0,
                                                                             VideoEncodeUsageFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeUsageFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeUsageFlagsKHR operator&( VideoEncodeUsageFlagBitsKHR bit0,
                                                                             VideoEncodeUsageFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeUsageFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeUsageFlagsKHR operator^( VideoEncodeUsageFlagBitsKHR bit0,
                                                                             VideoEncodeUsageFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeUsageFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeUsageFlagsKHR operator~( VideoEncodeUsageFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoEncodeUsageFlagsKHR( bits ) );
  }

  using VideoEncodeContentFlagsKHR = Flags<VideoEncodeContentFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeContentFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoEncodeContentFlagBitsKHR::eDefault ) | VkFlags( VideoEncodeContentFlagBitsKHR::eCamera ) |
                 VkFlags( VideoEncodeContentFlagBitsKHR::eDesktop ) | VkFlags( VideoEncodeContentFlagBitsKHR::eRendered )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeContentFlagsKHR operator|( VideoEncodeContentFlagBitsKHR bit0,
                                                                               VideoEncodeContentFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeContentFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeContentFlagsKHR operator&( VideoEncodeContentFlagBitsKHR bit0,
                                                                               VideoEncodeContentFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeContentFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeContentFlagsKHR operator^( VideoEncodeContentFlagBitsKHR bit0,
                                                                               VideoEncodeContentFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeContentFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeContentFlagsKHR operator~( VideoEncodeContentFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoEncodeContentFlagsKHR( bits ) );
  }

  using VideoEncodeRateControlFlagsKHR = Flags<VideoEncodeRateControlFlagBitsKHR>;

  using VideoEncodeRateControlModeFlagsKHR = Flags<VideoEncodeRateControlModeFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeRateControlModeFlagBitsKHR>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( VideoEncodeRateControlModeFlagBitsKHR::eNone ) | VkFlags( VideoEncodeRateControlModeFlagBitsKHR::eCbr ) |
                 VkFlags( VideoEncodeRateControlModeFlagBitsKHR::eVbr )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeRateControlModeFlagsKHR operator|( VideoEncodeRateControlModeFlagBitsKHR bit0,
                                                                                       VideoEncodeRateControlModeFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeRateControlModeFlagsKHR( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeRateControlModeFlagsKHR operator&( VideoEncodeRateControlModeFlagBitsKHR bit0,
                                                                                       VideoEncodeRateControlModeFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeRateControlModeFlagsKHR( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeRateControlModeFlagsKHR operator^( VideoEncodeRateControlModeFlagBitsKHR bit0,
                                                                                       VideoEncodeRateControlModeFlagBitsKHR bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return VideoEncodeRateControlModeFlagsKHR( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR VideoEncodeRateControlModeFlagsKHR operator~( VideoEncodeRateControlModeFlagBitsKHR bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( VideoEncodeRateControlModeFlagsKHR( bits ) );
  }

#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_NV_device_diagnostics_config ===

  using DeviceDiagnosticsConfigFlagsNV = Flags<DeviceDiagnosticsConfigFlagBitsNV>;

  template <>
  struct FlagTraits<DeviceDiagnosticsConfigFlagBitsNV>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( DeviceDiagnosticsConfigFlagBitsNV::eEnableShaderDebugInfo ) | VkFlags( DeviceDiagnosticsConfigFlagBitsNV::eEnableResourceTracking ) |
                 VkFlags( DeviceDiagnosticsConfigFlagBitsNV::eEnableAutomaticCheckpoints ) |
                 VkFlags( DeviceDiagnosticsConfigFlagBitsNV::eEnableShaderErrorReporting )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceDiagnosticsConfigFlagsNV operator|( DeviceDiagnosticsConfigFlagBitsNV bit0,
                                                                                   DeviceDiagnosticsConfigFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DeviceDiagnosticsConfigFlagsNV( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceDiagnosticsConfigFlagsNV operator&( DeviceDiagnosticsConfigFlagBitsNV bit0,
                                                                                   DeviceDiagnosticsConfigFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DeviceDiagnosticsConfigFlagsNV( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceDiagnosticsConfigFlagsNV operator^( DeviceDiagnosticsConfigFlagBitsNV bit0,
                                                                                   DeviceDiagnosticsConfigFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DeviceDiagnosticsConfigFlagsNV( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceDiagnosticsConfigFlagsNV operator~( DeviceDiagnosticsConfigFlagBitsNV bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( DeviceDiagnosticsConfigFlagsNV( bits ) );
  }

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_objects ===

  using ExportMetalObjectTypeFlagsEXT = Flags<ExportMetalObjectTypeFlagBitsEXT>;

  template <>
  struct FlagTraits<ExportMetalObjectTypeFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ExportMetalObjectTypeFlagBitsEXT::eMetalDevice ) | VkFlags( ExportMetalObjectTypeFlagBitsEXT::eMetalCommandQueue ) |
                 VkFlags( ExportMetalObjectTypeFlagBitsEXT::eMetalBuffer ) | VkFlags( ExportMetalObjectTypeFlagBitsEXT::eMetalTexture ) |
                 VkFlags( ExportMetalObjectTypeFlagBitsEXT::eMetalIosurface ) | VkFlags( ExportMetalObjectTypeFlagBitsEXT::eMetalSharedEvent )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExportMetalObjectTypeFlagsEXT operator|( ExportMetalObjectTypeFlagBitsEXT bit0,
                                                                                  ExportMetalObjectTypeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExportMetalObjectTypeFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExportMetalObjectTypeFlagsEXT operator&( ExportMetalObjectTypeFlagBitsEXT bit0,
                                                                                  ExportMetalObjectTypeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExportMetalObjectTypeFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExportMetalObjectTypeFlagsEXT operator^( ExportMetalObjectTypeFlagBitsEXT bit0,
                                                                                  ExportMetalObjectTypeFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ExportMetalObjectTypeFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ExportMetalObjectTypeFlagsEXT operator~( ExportMetalObjectTypeFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ExportMetalObjectTypeFlagsEXT( bits ) );
  }

#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_EXT_graphics_pipeline_library ===

  using GraphicsPipelineLibraryFlagsEXT = Flags<GraphicsPipelineLibraryFlagBitsEXT>;

  template <>
  struct FlagTraits<GraphicsPipelineLibraryFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( GraphicsPipelineLibraryFlagBitsEXT::eVertexInputInterface ) |
                 VkFlags( GraphicsPipelineLibraryFlagBitsEXT::ePreRasterizationShaders ) | VkFlags( GraphicsPipelineLibraryFlagBitsEXT::eFragmentShader ) |
                 VkFlags( GraphicsPipelineLibraryFlagBitsEXT::eFragmentOutputInterface )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR GraphicsPipelineLibraryFlagsEXT operator|( GraphicsPipelineLibraryFlagBitsEXT bit0,
                                                                                    GraphicsPipelineLibraryFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return GraphicsPipelineLibraryFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR GraphicsPipelineLibraryFlagsEXT operator&( GraphicsPipelineLibraryFlagBitsEXT bit0,
                                                                                    GraphicsPipelineLibraryFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return GraphicsPipelineLibraryFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR GraphicsPipelineLibraryFlagsEXT operator^( GraphicsPipelineLibraryFlagBitsEXT bit0,
                                                                                    GraphicsPipelineLibraryFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return GraphicsPipelineLibraryFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR GraphicsPipelineLibraryFlagsEXT operator~( GraphicsPipelineLibraryFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( GraphicsPipelineLibraryFlagsEXT( bits ) );
  }

  //=== VK_NV_ray_tracing_motion_blur ===

  using AccelerationStructureMotionInfoFlagsNV = Flags<AccelerationStructureMotionInfoFlagBitsNV>;

  using AccelerationStructureMotionInstanceFlagsNV = Flags<AccelerationStructureMotionInstanceFlagBitsNV>;

  //=== VK_EXT_image_compression_control ===

  using ImageCompressionFlagsEXT = Flags<ImageCompressionFlagBitsEXT>;

  template <>
  struct FlagTraits<ImageCompressionFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ImageCompressionFlagBitsEXT::eDefault ) | VkFlags( ImageCompressionFlagBitsEXT::eFixedRateDefault ) |
                 VkFlags( ImageCompressionFlagBitsEXT::eFixedRateExplicit ) | VkFlags( ImageCompressionFlagBitsEXT::eDisabled )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageCompressionFlagsEXT operator|( ImageCompressionFlagBitsEXT bit0,
                                                                             ImageCompressionFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageCompressionFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageCompressionFlagsEXT operator&( ImageCompressionFlagBitsEXT bit0,
                                                                             ImageCompressionFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageCompressionFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageCompressionFlagsEXT operator^( ImageCompressionFlagBitsEXT bit0,
                                                                             ImageCompressionFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageCompressionFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageCompressionFlagsEXT operator~( ImageCompressionFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ImageCompressionFlagsEXT( bits ) );
  }

  using ImageCompressionFixedRateFlagsEXT = Flags<ImageCompressionFixedRateFlagBitsEXT>;

  template <>
  struct FlagTraits<ImageCompressionFixedRateFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ImageCompressionFixedRateFlagBitsEXT::eNone ) | VkFlags( ImageCompressionFixedRateFlagBitsEXT::e1Bpc ) |
                 VkFlags( ImageCompressionFixedRateFlagBitsEXT::e2Bpc ) | VkFlags( ImageCompressionFixedRateFlagBitsEXT::e3Bpc ) |
                 VkFlags( ImageCompressionFixedRateFlagBitsEXT::e4Bpc ) | VkFlags( ImageCompressionFixedRateFlagBitsEXT::e5Bpc ) |
                 VkFlags( ImageCompressionFixedRateFlagBitsEXT::e6Bpc ) | VkFlags( ImageCompressionFixedRateFlagBitsEXT::e7Bpc ) |
                 VkFlags( ImageCompressionFixedRateFlagBitsEXT::e8Bpc ) | VkFlags( ImageCompressionFixedRateFlagBitsEXT::e9Bpc ) |
                 VkFlags( ImageCompressionFixedRateFlagBitsEXT::e10Bpc ) | VkFlags( ImageCompressionFixedRateFlagBitsEXT::e11Bpc ) |
                 VkFlags( ImageCompressionFixedRateFlagBitsEXT::e12Bpc ) | VkFlags( ImageCompressionFixedRateFlagBitsEXT::e13Bpc ) |
                 VkFlags( ImageCompressionFixedRateFlagBitsEXT::e14Bpc ) | VkFlags( ImageCompressionFixedRateFlagBitsEXT::e15Bpc ) |
                 VkFlags( ImageCompressionFixedRateFlagBitsEXT::e16Bpc ) | VkFlags( ImageCompressionFixedRateFlagBitsEXT::e17Bpc ) |
                 VkFlags( ImageCompressionFixedRateFlagBitsEXT::e18Bpc ) | VkFlags( ImageCompressionFixedRateFlagBitsEXT::e19Bpc ) |
                 VkFlags( ImageCompressionFixedRateFlagBitsEXT::e20Bpc ) | VkFlags( ImageCompressionFixedRateFlagBitsEXT::e21Bpc ) |
                 VkFlags( ImageCompressionFixedRateFlagBitsEXT::e22Bpc ) | VkFlags( ImageCompressionFixedRateFlagBitsEXT::e23Bpc ) |
                 VkFlags( ImageCompressionFixedRateFlagBitsEXT::e24Bpc )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageCompressionFixedRateFlagsEXT operator|( ImageCompressionFixedRateFlagBitsEXT bit0,
                                                                                      ImageCompressionFixedRateFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageCompressionFixedRateFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageCompressionFixedRateFlagsEXT operator&( ImageCompressionFixedRateFlagBitsEXT bit0,
                                                                                      ImageCompressionFixedRateFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageCompressionFixedRateFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageCompressionFixedRateFlagsEXT operator^( ImageCompressionFixedRateFlagBitsEXT bit0,
                                                                                      ImageCompressionFixedRateFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageCompressionFixedRateFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageCompressionFixedRateFlagsEXT operator~( ImageCompressionFixedRateFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ImageCompressionFixedRateFlagsEXT( bits ) );
  }

#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
  //=== VK_EXT_directfb_surface ===

  using DirectFBSurfaceCreateFlagsEXT = Flags<DirectFBSurfaceCreateFlagBitsEXT>;

#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

  //=== VK_EXT_device_address_binding_report ===

  using DeviceAddressBindingFlagsEXT = Flags<DeviceAddressBindingFlagBitsEXT>;

  template <>
  struct FlagTraits<DeviceAddressBindingFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( DeviceAddressBindingFlagBitsEXT::eInternalObject )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceAddressBindingFlagsEXT operator|( DeviceAddressBindingFlagBitsEXT bit0,
                                                                                 DeviceAddressBindingFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DeviceAddressBindingFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceAddressBindingFlagsEXT operator&( DeviceAddressBindingFlagBitsEXT bit0,
                                                                                 DeviceAddressBindingFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DeviceAddressBindingFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceAddressBindingFlagsEXT operator^( DeviceAddressBindingFlagBitsEXT bit0,
                                                                                 DeviceAddressBindingFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return DeviceAddressBindingFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR DeviceAddressBindingFlagsEXT operator~( DeviceAddressBindingFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( DeviceAddressBindingFlagsEXT( bits ) );
  }

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_buffer_collection ===

  using ImageFormatConstraintsFlagsFUCHSIA = Flags<ImageFormatConstraintsFlagBitsFUCHSIA>;

  using ImageConstraintsInfoFlagsFUCHSIA = Flags<ImageConstraintsInfoFlagBitsFUCHSIA>;

  template <>
  struct FlagTraits<ImageConstraintsInfoFlagBitsFUCHSIA>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( ImageConstraintsInfoFlagBitsFUCHSIA::eCpuReadRarely ) | VkFlags( ImageConstraintsInfoFlagBitsFUCHSIA::eCpuReadOften ) |
                 VkFlags( ImageConstraintsInfoFlagBitsFUCHSIA::eCpuWriteRarely ) | VkFlags( ImageConstraintsInfoFlagBitsFUCHSIA::eCpuWriteOften ) |
                 VkFlags( ImageConstraintsInfoFlagBitsFUCHSIA::eProtectedOptional )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageConstraintsInfoFlagsFUCHSIA operator|( ImageConstraintsInfoFlagBitsFUCHSIA bit0,
                                                                                     ImageConstraintsInfoFlagBitsFUCHSIA bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageConstraintsInfoFlagsFUCHSIA( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageConstraintsInfoFlagsFUCHSIA operator&( ImageConstraintsInfoFlagBitsFUCHSIA bit0,
                                                                                     ImageConstraintsInfoFlagBitsFUCHSIA bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageConstraintsInfoFlagsFUCHSIA( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageConstraintsInfoFlagsFUCHSIA operator^( ImageConstraintsInfoFlagBitsFUCHSIA bit0,
                                                                                     ImageConstraintsInfoFlagBitsFUCHSIA bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return ImageConstraintsInfoFlagsFUCHSIA( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR ImageConstraintsInfoFlagsFUCHSIA operator~( ImageConstraintsInfoFlagBitsFUCHSIA bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( ImageConstraintsInfoFlagsFUCHSIA( bits ) );
  }

#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
  //=== VK_QNX_screen_surface ===

  using ScreenSurfaceCreateFlagsQNX = Flags<ScreenSurfaceCreateFlagBitsQNX>;

#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

  //=== VK_EXT_opacity_micromap ===

  using BuildMicromapFlagsEXT = Flags<BuildMicromapFlagBitsEXT>;

  template <>
  struct FlagTraits<BuildMicromapFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( BuildMicromapFlagBitsEXT::ePreferFastTrace ) | VkFlags( BuildMicromapFlagBitsEXT::ePreferFastBuild ) |
                 VkFlags( BuildMicromapFlagBitsEXT::eAllowCompaction )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BuildMicromapFlagsEXT operator|( BuildMicromapFlagBitsEXT bit0, BuildMicromapFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return BuildMicromapFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BuildMicromapFlagsEXT operator&( BuildMicromapFlagBitsEXT bit0, BuildMicromapFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return BuildMicromapFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BuildMicromapFlagsEXT operator^( BuildMicromapFlagBitsEXT bit0, BuildMicromapFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return BuildMicromapFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR BuildMicromapFlagsEXT operator~( BuildMicromapFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( BuildMicromapFlagsEXT( bits ) );
  }

  using MicromapCreateFlagsEXT = Flags<MicromapCreateFlagBitsEXT>;

  template <>
  struct FlagTraits<MicromapCreateFlagBitsEXT>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( MicromapCreateFlagBitsEXT::eDeviceAddressCaptureReplay )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MicromapCreateFlagsEXT operator|( MicromapCreateFlagBitsEXT bit0, MicromapCreateFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return MicromapCreateFlagsEXT( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MicromapCreateFlagsEXT operator&( MicromapCreateFlagBitsEXT bit0, MicromapCreateFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return MicromapCreateFlagsEXT( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MicromapCreateFlagsEXT operator^( MicromapCreateFlagBitsEXT bit0, MicromapCreateFlagBitsEXT bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return MicromapCreateFlagsEXT( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR MicromapCreateFlagsEXT operator~( MicromapCreateFlagBitsEXT bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( MicromapCreateFlagsEXT( bits ) );
  }

  //=== VK_NV_optical_flow ===

  using OpticalFlowUsageFlagsNV = Flags<OpticalFlowUsageFlagBitsNV>;

  template <>
  struct FlagTraits<OpticalFlowUsageFlagBitsNV>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( OpticalFlowUsageFlagBitsNV::eUnknown ) | VkFlags( OpticalFlowUsageFlagBitsNV::eInput ) |
                 VkFlags( OpticalFlowUsageFlagBitsNV::eOutput ) | VkFlags( OpticalFlowUsageFlagBitsNV::eHint ) | VkFlags( OpticalFlowUsageFlagBitsNV::eCost ) |
                 VkFlags( OpticalFlowUsageFlagBitsNV::eGlobalFlow )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR OpticalFlowUsageFlagsNV operator|( OpticalFlowUsageFlagBitsNV bit0,
                                                                            OpticalFlowUsageFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return OpticalFlowUsageFlagsNV( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR OpticalFlowUsageFlagsNV operator&( OpticalFlowUsageFlagBitsNV bit0,
                                                                            OpticalFlowUsageFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return OpticalFlowUsageFlagsNV( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR OpticalFlowUsageFlagsNV operator^( OpticalFlowUsageFlagBitsNV bit0,
                                                                            OpticalFlowUsageFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return OpticalFlowUsageFlagsNV( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR OpticalFlowUsageFlagsNV operator~( OpticalFlowUsageFlagBitsNV bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( OpticalFlowUsageFlagsNV( bits ) );
  }

  using OpticalFlowGridSizeFlagsNV = Flags<OpticalFlowGridSizeFlagBitsNV>;

  template <>
  struct FlagTraits<OpticalFlowGridSizeFlagBitsNV>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( OpticalFlowGridSizeFlagBitsNV::eUnknown ) | VkFlags( OpticalFlowGridSizeFlagBitsNV::e1X1 ) |
                 VkFlags( OpticalFlowGridSizeFlagBitsNV::e2X2 ) | VkFlags( OpticalFlowGridSizeFlagBitsNV::e4X4 ) |
                 VkFlags( OpticalFlowGridSizeFlagBitsNV::e8X8 )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR OpticalFlowGridSizeFlagsNV operator|( OpticalFlowGridSizeFlagBitsNV bit0,
                                                                               OpticalFlowGridSizeFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return OpticalFlowGridSizeFlagsNV( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR OpticalFlowGridSizeFlagsNV operator&( OpticalFlowGridSizeFlagBitsNV bit0,
                                                                               OpticalFlowGridSizeFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return OpticalFlowGridSizeFlagsNV( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR OpticalFlowGridSizeFlagsNV operator^( OpticalFlowGridSizeFlagBitsNV bit0,
                                                                               OpticalFlowGridSizeFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return OpticalFlowGridSizeFlagsNV( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR OpticalFlowGridSizeFlagsNV operator~( OpticalFlowGridSizeFlagBitsNV bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( OpticalFlowGridSizeFlagsNV( bits ) );
  }

  using OpticalFlowSessionCreateFlagsNV = Flags<OpticalFlowSessionCreateFlagBitsNV>;

  template <>
  struct FlagTraits<OpticalFlowSessionCreateFlagBitsNV>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( OpticalFlowSessionCreateFlagBitsNV::eEnableHint ) | VkFlags( OpticalFlowSessionCreateFlagBitsNV::eEnableCost ) |
                 VkFlags( OpticalFlowSessionCreateFlagBitsNV::eEnableGlobalFlow ) | VkFlags( OpticalFlowSessionCreateFlagBitsNV::eAllowRegions ) |
                 VkFlags( OpticalFlowSessionCreateFlagBitsNV::eBothDirections )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR OpticalFlowSessionCreateFlagsNV operator|( OpticalFlowSessionCreateFlagBitsNV bit0,
                                                                                    OpticalFlowSessionCreateFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return OpticalFlowSessionCreateFlagsNV( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR OpticalFlowSessionCreateFlagsNV operator&( OpticalFlowSessionCreateFlagBitsNV bit0,
                                                                                    OpticalFlowSessionCreateFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return OpticalFlowSessionCreateFlagsNV( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR OpticalFlowSessionCreateFlagsNV operator^( OpticalFlowSessionCreateFlagBitsNV bit0,
                                                                                    OpticalFlowSessionCreateFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return OpticalFlowSessionCreateFlagsNV( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR OpticalFlowSessionCreateFlagsNV operator~( OpticalFlowSessionCreateFlagBitsNV bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( OpticalFlowSessionCreateFlagsNV( bits ) );
  }

  using OpticalFlowExecuteFlagsNV = Flags<OpticalFlowExecuteFlagBitsNV>;

  template <>
  struct FlagTraits<OpticalFlowExecuteFlagBitsNV>
  {
    enum : VkFlags
    {
      allFlags = VkFlags( OpticalFlowExecuteFlagBitsNV::eDisableTemporalHints )
    };
  };

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR OpticalFlowExecuteFlagsNV operator|( OpticalFlowExecuteFlagBitsNV bit0,
                                                                              OpticalFlowExecuteFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return OpticalFlowExecuteFlagsNV( bit0 ) | bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR OpticalFlowExecuteFlagsNV operator&( OpticalFlowExecuteFlagBitsNV bit0,
                                                                              OpticalFlowExecuteFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return OpticalFlowExecuteFlagsNV( bit0 ) & bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR OpticalFlowExecuteFlagsNV operator^( OpticalFlowExecuteFlagBitsNV bit0,
                                                                              OpticalFlowExecuteFlagBitsNV bit1 ) VULKAN_HPP_NOEXCEPT
  {
    return OpticalFlowExecuteFlagsNV( bit0 ) ^ bit1;
  }

  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR OpticalFlowExecuteFlagsNV operator~( OpticalFlowExecuteFlagBitsNV bits ) VULKAN_HPP_NOEXCEPT
  {
    return ~( OpticalFlowExecuteFlagsNV( bits ) );
  }

}  // namespace VULKAN_HPP_NAMESPACE
#endif
