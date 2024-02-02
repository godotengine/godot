// Copyright 2015-2023 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//

// This header is generated from the Khronos Vulkan XML API Registry.

#ifndef VULKAN_ENUMS_HPP
#define VULKAN_ENUMS_HPP

namespace VULKAN_HPP_NAMESPACE
{
  template <typename FlagBitsType>
  struct FlagTraits
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool isBitmask = false;
  };

  template <typename BitType>
  class Flags
  {
  public:
    using MaskType = typename std::underlying_type<BitType>::type;

    // constructors
    VULKAN_HPP_CONSTEXPR Flags() VULKAN_HPP_NOEXCEPT : m_mask( 0 ) {}

    VULKAN_HPP_CONSTEXPR Flags( BitType bit ) VULKAN_HPP_NOEXCEPT : m_mask( static_cast<MaskType>( bit ) ) {}

    VULKAN_HPP_CONSTEXPR Flags( Flags<BitType> const & rhs ) VULKAN_HPP_NOEXCEPT = default;

    VULKAN_HPP_CONSTEXPR explicit Flags( MaskType flags ) VULKAN_HPP_NOEXCEPT : m_mask( flags ) {}

    // relational operators
#if defined( VULKAN_HPP_HAS_SPACESHIP_OPERATOR )
    auto operator<=>( Flags<BitType> const & ) const = default;
#else
    VULKAN_HPP_CONSTEXPR bool operator<( Flags<BitType> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return m_mask < rhs.m_mask;
    }

    VULKAN_HPP_CONSTEXPR bool operator<=( Flags<BitType> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return m_mask <= rhs.m_mask;
    }

    VULKAN_HPP_CONSTEXPR bool operator>( Flags<BitType> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return m_mask > rhs.m_mask;
    }

    VULKAN_HPP_CONSTEXPR bool operator>=( Flags<BitType> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return m_mask >= rhs.m_mask;
    }

    VULKAN_HPP_CONSTEXPR bool operator==( Flags<BitType> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return m_mask == rhs.m_mask;
    }

    VULKAN_HPP_CONSTEXPR bool operator!=( Flags<BitType> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return m_mask != rhs.m_mask;
    }
#endif

    // logical operator
    VULKAN_HPP_CONSTEXPR bool operator!() const VULKAN_HPP_NOEXCEPT
    {
      return !m_mask;
    }

    // bitwise operators
    VULKAN_HPP_CONSTEXPR Flags<BitType> operator&( Flags<BitType> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return Flags<BitType>( m_mask & rhs.m_mask );
    }

    VULKAN_HPP_CONSTEXPR Flags<BitType> operator|( Flags<BitType> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return Flags<BitType>( m_mask | rhs.m_mask );
    }

    VULKAN_HPP_CONSTEXPR Flags<BitType> operator^( Flags<BitType> const & rhs ) const VULKAN_HPP_NOEXCEPT
    {
      return Flags<BitType>( m_mask ^ rhs.m_mask );
    }

    VULKAN_HPP_CONSTEXPR Flags<BitType> operator~() const VULKAN_HPP_NOEXCEPT
    {
      return Flags<BitType>( m_mask ^ FlagTraits<BitType>::allFlags.m_mask );
    }

    // assignment operators
    VULKAN_HPP_CONSTEXPR_14 Flags<BitType> & operator=( Flags<BitType> const & rhs ) VULKAN_HPP_NOEXCEPT = default;

    VULKAN_HPP_CONSTEXPR_14 Flags<BitType> & operator|=( Flags<BitType> const & rhs ) VULKAN_HPP_NOEXCEPT
    {
      m_mask |= rhs.m_mask;
      return *this;
    }

    VULKAN_HPP_CONSTEXPR_14 Flags<BitType> & operator&=( Flags<BitType> const & rhs ) VULKAN_HPP_NOEXCEPT
    {
      m_mask &= rhs.m_mask;
      return *this;
    }

    VULKAN_HPP_CONSTEXPR_14 Flags<BitType> & operator^=( Flags<BitType> const & rhs ) VULKAN_HPP_NOEXCEPT
    {
      m_mask ^= rhs.m_mask;
      return *this;
    }

    // cast operators
    explicit VULKAN_HPP_CONSTEXPR operator bool() const VULKAN_HPP_NOEXCEPT
    {
      return !!m_mask;
    }

    explicit VULKAN_HPP_CONSTEXPR operator MaskType() const VULKAN_HPP_NOEXCEPT
    {
      return m_mask;
    }

#if defined( VULKAN_HPP_FLAGS_MASK_TYPE_AS_PUBLIC )
  public:
#else
  private:
#endif
    MaskType m_mask;
  };

#if !defined( VULKAN_HPP_HAS_SPACESHIP_OPERATOR )
  // relational operators only needed for pre C++20
  template <typename BitType>
  VULKAN_HPP_CONSTEXPR bool operator<( BitType bit, Flags<BitType> const & flags ) VULKAN_HPP_NOEXCEPT
  {
    return flags.operator>( bit );
  }

  template <typename BitType>
  VULKAN_HPP_CONSTEXPR bool operator<=( BitType bit, Flags<BitType> const & flags ) VULKAN_HPP_NOEXCEPT
  {
    return flags.operator>=( bit );
  }

  template <typename BitType>
  VULKAN_HPP_CONSTEXPR bool operator>( BitType bit, Flags<BitType> const & flags ) VULKAN_HPP_NOEXCEPT
  {
    return flags.operator<( bit );
  }

  template <typename BitType>
  VULKAN_HPP_CONSTEXPR bool operator>=( BitType bit, Flags<BitType> const & flags ) VULKAN_HPP_NOEXCEPT
  {
    return flags.operator<=( bit );
  }

  template <typename BitType>
  VULKAN_HPP_CONSTEXPR bool operator==( BitType bit, Flags<BitType> const & flags ) VULKAN_HPP_NOEXCEPT
  {
    return flags.operator==( bit );
  }

  template <typename BitType>
  VULKAN_HPP_CONSTEXPR bool operator!=( BitType bit, Flags<BitType> const & flags ) VULKAN_HPP_NOEXCEPT
  {
    return flags.operator!=( bit );
  }
#endif

  // bitwise operators
  template <typename BitType>
  VULKAN_HPP_CONSTEXPR Flags<BitType> operator&( BitType bit, Flags<BitType> const & flags ) VULKAN_HPP_NOEXCEPT
  {
    return flags.operator&( bit );
  }

  template <typename BitType>
  VULKAN_HPP_CONSTEXPR Flags<BitType> operator|( BitType bit, Flags<BitType> const & flags ) VULKAN_HPP_NOEXCEPT
  {
    return flags.operator|( bit );
  }

  template <typename BitType>
  VULKAN_HPP_CONSTEXPR Flags<BitType> operator^( BitType bit, Flags<BitType> const & flags ) VULKAN_HPP_NOEXCEPT
  {
    return flags.operator^( bit );
  }

  // bitwise operators on BitType
  template <typename BitType, typename std::enable_if<FlagTraits<BitType>::isBitmask, bool>::type = true>
  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR Flags<BitType> operator&( BitType lhs, BitType rhs ) VULKAN_HPP_NOEXCEPT
  {
    return Flags<BitType>( lhs ) & rhs;
  }

  template <typename BitType, typename std::enable_if<FlagTraits<BitType>::isBitmask, bool>::type = true>
  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR Flags<BitType> operator|( BitType lhs, BitType rhs ) VULKAN_HPP_NOEXCEPT
  {
    return Flags<BitType>( lhs ) | rhs;
  }

  template <typename BitType, typename std::enable_if<FlagTraits<BitType>::isBitmask, bool>::type = true>
  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR Flags<BitType> operator^( BitType lhs, BitType rhs ) VULKAN_HPP_NOEXCEPT
  {
    return Flags<BitType>( lhs ) ^ rhs;
  }

  template <typename BitType, typename std::enable_if<FlagTraits<BitType>::isBitmask, bool>::type = true>
  VULKAN_HPP_INLINE VULKAN_HPP_CONSTEXPR Flags<BitType> operator~( BitType bit ) VULKAN_HPP_NOEXCEPT
  {
    return ~( Flags<BitType>( bit ) );
  }

  template <typename EnumType, EnumType value>
  struct CppType
  {
  };

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
    ePipelineCompileRequired                     = VK_PIPELINE_COMPILE_REQUIRED,
    eErrorSurfaceLostKHR                         = VK_ERROR_SURFACE_LOST_KHR,
    eErrorNativeWindowInUseKHR                   = VK_ERROR_NATIVE_WINDOW_IN_USE_KHR,
    eSuboptimalKHR                               = VK_SUBOPTIMAL_KHR,
    eErrorOutOfDateKHR                           = VK_ERROR_OUT_OF_DATE_KHR,
    eErrorIncompatibleDisplayKHR                 = VK_ERROR_INCOMPATIBLE_DISPLAY_KHR,
    eErrorValidationFailedEXT                    = VK_ERROR_VALIDATION_FAILED_EXT,
    eErrorInvalidShaderNV                        = VK_ERROR_INVALID_SHADER_NV,
    eErrorImageUsageNotSupportedKHR              = VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR,
    eErrorVideoPictureLayoutNotSupportedKHR      = VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR,
    eErrorVideoProfileOperationNotSupportedKHR   = VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR,
    eErrorVideoProfileFormatNotSupportedKHR      = VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR,
    eErrorVideoProfileCodecNotSupportedKHR       = VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR,
    eErrorVideoStdVersionNotSupportedKHR         = VK_ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR,
    eErrorOutOfPoolMemoryKHR                     = VK_ERROR_OUT_OF_POOL_MEMORY_KHR,
    eErrorInvalidExternalHandleKHR               = VK_ERROR_INVALID_EXTERNAL_HANDLE_KHR,
    eErrorInvalidDrmFormatModifierPlaneLayoutEXT = VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT,
    eErrorFragmentationEXT                       = VK_ERROR_FRAGMENTATION_EXT,
    eErrorNotPermittedEXT                        = VK_ERROR_NOT_PERMITTED_EXT,
    eErrorNotPermittedKHR                        = VK_ERROR_NOT_PERMITTED_KHR,
    eErrorInvalidDeviceAddressEXT                = VK_ERROR_INVALID_DEVICE_ADDRESS_EXT,
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    eErrorFullScreenExclusiveModeLostEXT = VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT,
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    eErrorInvalidOpaqueCaptureAddressKHR = VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR,
    eThreadIdleKHR                       = VK_THREAD_IDLE_KHR,
    eThreadDoneKHR                       = VK_THREAD_DONE_KHR,
    eOperationDeferredKHR                = VK_OPERATION_DEFERRED_KHR,
    eOperationNotDeferredKHR             = VK_OPERATION_NOT_DEFERRED_KHR,
    ePipelineCompileRequiredEXT          = VK_PIPELINE_COMPILE_REQUIRED_EXT,
    eErrorPipelineCompileRequiredEXT     = VK_ERROR_PIPELINE_COMPILE_REQUIRED_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eErrorInvalidVideoStdParametersKHR = VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eErrorCompressionExhaustedEXT     = VK_ERROR_COMPRESSION_EXHAUSTED_EXT,
    eErrorIncompatibleShaderBinaryEXT = VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT
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
    ePhysicalDeviceVariablePointerFeatures                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTER_FEATURES,
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
    ePhysicalDeviceShaderDrawParameterFeatures            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETER_FEATURES,
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
    eDebugReportCreateInfoEXT                        = VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT,
    ePipelineRasterizationStateRasterizationOrderAMD = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_RASTERIZATION_ORDER_AMD,
    eDebugMarkerObjectNameInfoEXT                    = VK_STRUCTURE_TYPE_DEBUG_MARKER_OBJECT_NAME_INFO_EXT,
    eDebugMarkerObjectTagInfoEXT                     = VK_STRUCTURE_TYPE_DEBUG_MARKER_OBJECT_TAG_INFO_EXT,
    eDebugMarkerMarkerInfoEXT                        = VK_STRUCTURE_TYPE_DEBUG_MARKER_MARKER_INFO_EXT,
    eVideoProfileInfoKHR                             = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR,
    eVideoCapabilitiesKHR                            = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR,
    eVideoPictureResourceInfoKHR                     = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,
    eVideoSessionMemoryRequirementsKHR               = VK_STRUCTURE_TYPE_VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR,
    eBindVideoSessionMemoryInfoKHR                   = VK_STRUCTURE_TYPE_BIND_VIDEO_SESSION_MEMORY_INFO_KHR,
    eVideoSessionCreateInfoKHR                       = VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR,
    eVideoSessionParametersCreateInfoKHR             = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
    eVideoSessionParametersUpdateInfoKHR             = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_UPDATE_INFO_KHR,
    eVideoBeginCodingInfoKHR                         = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR,
    eVideoEndCodingInfoKHR                           = VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR,
    eVideoCodingControlInfoKHR                       = VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR,
    eVideoReferenceSlotInfoKHR                       = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
    eQueueFamilyVideoPropertiesKHR                   = VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR,
    eVideoProfileListInfoKHR                         = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR,
    ePhysicalDeviceVideoFormatInfoKHR                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_FORMAT_INFO_KHR,
    eVideoFormatPropertiesKHR                        = VK_STRUCTURE_TYPE_VIDEO_FORMAT_PROPERTIES_KHR,
    eQueueFamilyQueryResultStatusPropertiesKHR       = VK_STRUCTURE_TYPE_QUEUE_FAMILY_QUERY_RESULT_STATUS_PROPERTIES_KHR,
    eVideoDecodeInfoKHR                              = VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR,
    eVideoDecodeCapabilitiesKHR                      = VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR,
    eVideoDecodeUsageInfoKHR                         = VK_STRUCTURE_TYPE_VIDEO_DECODE_USAGE_INFO_KHR,
    eDedicatedAllocationImageCreateInfoNV            = VK_STRUCTURE_TYPE_DEDICATED_ALLOCATION_IMAGE_CREATE_INFO_NV,
    eDedicatedAllocationBufferCreateInfoNV           = VK_STRUCTURE_TYPE_DEDICATED_ALLOCATION_BUFFER_CREATE_INFO_NV,
    eDedicatedAllocationMemoryAllocateInfoNV         = VK_STRUCTURE_TYPE_DEDICATED_ALLOCATION_MEMORY_ALLOCATE_INFO_NV,
    ePhysicalDeviceTransformFeedbackFeaturesEXT      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT,
    ePhysicalDeviceTransformFeedbackPropertiesEXT    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_PROPERTIES_EXT,
    ePipelineRasterizationStateStreamCreateInfoEXT   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_STREAM_CREATE_INFO_EXT,
    eCuModuleCreateInfoNVX                           = VK_STRUCTURE_TYPE_CU_MODULE_CREATE_INFO_NVX,
    eCuFunctionCreateInfoNVX                         = VK_STRUCTURE_TYPE_CU_FUNCTION_CREATE_INFO_NVX,
    eCuLaunchInfoNVX                                 = VK_STRUCTURE_TYPE_CU_LAUNCH_INFO_NVX,
    eImageViewHandleInfoNVX                          = VK_STRUCTURE_TYPE_IMAGE_VIEW_HANDLE_INFO_NVX,
    eImageViewAddressPropertiesNVX                   = VK_STRUCTURE_TYPE_IMAGE_VIEW_ADDRESS_PROPERTIES_NVX,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoEncodeH264CapabilitiesEXT                  = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_CAPABILITIES_EXT,
    eVideoEncodeH264SessionParametersCreateInfoEXT   = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_CREATE_INFO_EXT,
    eVideoEncodeH264SessionParametersAddInfoEXT      = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_ADD_INFO_EXT,
    eVideoEncodeH264PictureInfoEXT                   = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PICTURE_INFO_EXT,
    eVideoEncodeH264DpbSlotInfoEXT                   = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_DPB_SLOT_INFO_EXT,
    eVideoEncodeH264NaluSliceInfoEXT                 = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_NALU_SLICE_INFO_EXT,
    eVideoEncodeH264GopRemainingFrameInfoEXT         = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_GOP_REMAINING_FRAME_INFO_EXT,
    eVideoEncodeH264ProfileInfoEXT                   = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PROFILE_INFO_EXT,
    eVideoEncodeH264RateControlInfoEXT               = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_RATE_CONTROL_INFO_EXT,
    eVideoEncodeH264RateControlLayerInfoEXT          = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_RATE_CONTROL_LAYER_INFO_EXT,
    eVideoEncodeH264SessionCreateInfoEXT             = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_CREATE_INFO_EXT,
    eVideoEncodeH264QualityLevelPropertiesEXT        = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_QUALITY_LEVEL_PROPERTIES_EXT,
    eVideoEncodeH264SessionParametersGetInfoEXT      = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_GET_INFO_EXT,
    eVideoEncodeH264SessionParametersFeedbackInfoEXT = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_FEEDBACK_INFO_EXT,
    eVideoEncodeH265CapabilitiesEXT                  = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_CAPABILITIES_EXT,
    eVideoEncodeH265SessionParametersCreateInfoEXT   = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_SESSION_PARAMETERS_CREATE_INFO_EXT,
    eVideoEncodeH265SessionParametersAddInfoEXT      = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_SESSION_PARAMETERS_ADD_INFO_EXT,
    eVideoEncodeH265PictureInfoEXT                   = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_PICTURE_INFO_EXT,
    eVideoEncodeH265DpbSlotInfoEXT                   = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_DPB_SLOT_INFO_EXT,
    eVideoEncodeH265NaluSliceSegmentInfoEXT          = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_NALU_SLICE_SEGMENT_INFO_EXT,
    eVideoEncodeH265GopRemainingFrameInfoEXT         = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_GOP_REMAINING_FRAME_INFO_EXT,
    eVideoEncodeH265ProfileInfoEXT                   = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_PROFILE_INFO_EXT,
    eVideoEncodeH265RateControlInfoEXT               = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_RATE_CONTROL_INFO_EXT,
    eVideoEncodeH265RateControlLayerInfoEXT          = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_RATE_CONTROL_LAYER_INFO_EXT,
    eVideoEncodeH265SessionCreateInfoEXT             = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_SESSION_CREATE_INFO_EXT,
    eVideoEncodeH265QualityLevelPropertiesEXT        = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_QUALITY_LEVEL_PROPERTIES_EXT,
    eVideoEncodeH265SessionParametersGetInfoEXT      = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_SESSION_PARAMETERS_GET_INFO_EXT,
    eVideoEncodeH265SessionParametersFeedbackInfoEXT = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_SESSION_PARAMETERS_FEEDBACK_INFO_EXT,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eVideoDecodeH264CapabilitiesKHR                = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_CAPABILITIES_KHR,
    eVideoDecodeH264PictureInfoKHR                 = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PICTURE_INFO_KHR,
    eVideoDecodeH264ProfileInfoKHR                 = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PROFILE_INFO_KHR,
    eVideoDecodeH264SessionParametersCreateInfoKHR = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR,
    eVideoDecodeH264SessionParametersAddInfoKHR    = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR,
    eVideoDecodeH264DpbSlotInfoKHR                 = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,
    eTextureLodGatherFormatPropertiesAMD           = VK_STRUCTURE_TYPE_TEXTURE_LOD_GATHER_FORMAT_PROPERTIES_AMD,
    eRenderingInfoKHR                              = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
    eRenderingAttachmentInfoKHR                    = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
    ePipelineRenderingCreateInfoKHR                = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
    ePhysicalDeviceDynamicRenderingFeaturesKHR     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR,
    eCommandBufferInheritanceRenderingInfoKHR      = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO_KHR,
    eRenderingFragmentShadingRateAttachmentInfoKHR = VK_STRUCTURE_TYPE_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_INFO_KHR,
    eRenderingFragmentDensityMapAttachmentInfoEXT  = VK_STRUCTURE_TYPE_RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT_INFO_EXT,
    eAttachmentSampleCountInfoAMD                  = VK_STRUCTURE_TYPE_ATTACHMENT_SAMPLE_COUNT_INFO_AMD,
    eAttachmentSampleCountInfoNV                   = VK_STRUCTURE_TYPE_ATTACHMENT_SAMPLE_COUNT_INFO_NV,
    eMultiviewPerViewAttributesInfoNVX             = VK_STRUCTURE_TYPE_MULTIVIEW_PER_VIEW_ATTRIBUTES_INFO_NVX,
#if defined( VK_USE_PLATFORM_GGP )
    eStreamDescriptorSurfaceCreateInfoGGP = VK_STRUCTURE_TYPE_STREAM_DESCRIPTOR_SURFACE_CREATE_INFO_GGP,
#endif /*VK_USE_PLATFORM_GGP*/
    ePhysicalDeviceCornerSampledImageFeaturesNV = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CORNER_SAMPLED_IMAGE_FEATURES_NV,
    eRenderPassMultiviewCreateInfoKHR           = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO_KHR,
    ePhysicalDeviceMultiviewFeaturesKHR         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES_KHR,
    ePhysicalDeviceMultiviewPropertiesKHR       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES_KHR,
    eExternalMemoryImageCreateInfoNV            = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO_NV,
    eExportMemoryAllocateInfoNV                 = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_NV,
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    eImportMemoryWin32HandleInfoNV       = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_NV,
    eExportMemoryWin32HandleInfoNV       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_NV,
    eWin32KeyedMutexAcquireReleaseInfoNV = VK_STRUCTURE_TYPE_WIN32_KEYED_MUTEX_ACQUIRE_RELEASE_INFO_NV,
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    ePhysicalDeviceFeatures2KHR              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR,
    ePhysicalDeviceProperties2KHR            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR,
    eFormatProperties2KHR                    = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2_KHR,
    eImageFormatProperties2KHR               = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2_KHR,
    ePhysicalDeviceImageFormatInfo2KHR       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2_KHR,
    eQueueFamilyProperties2KHR               = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2_KHR,
    ePhysicalDeviceMemoryProperties2KHR      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR,
    eSparseImageFormatProperties2KHR         = VK_STRUCTURE_TYPE_SPARSE_IMAGE_FORMAT_PROPERTIES_2_KHR,
    ePhysicalDeviceSparseImageFormatInfo2KHR = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SPARSE_IMAGE_FORMAT_INFO_2_KHR,
    eMemoryAllocateFlagsInfoKHR              = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_KHR,
    eDeviceGroupRenderPassBeginInfoKHR       = VK_STRUCTURE_TYPE_DEVICE_GROUP_RENDER_PASS_BEGIN_INFO_KHR,
    eDeviceGroupCommandBufferBeginInfoKHR    = VK_STRUCTURE_TYPE_DEVICE_GROUP_COMMAND_BUFFER_BEGIN_INFO_KHR,
    eDeviceGroupSubmitInfoKHR                = VK_STRUCTURE_TYPE_DEVICE_GROUP_SUBMIT_INFO_KHR,
    eDeviceGroupBindSparseInfoKHR            = VK_STRUCTURE_TYPE_DEVICE_GROUP_BIND_SPARSE_INFO_KHR,
    eBindBufferMemoryDeviceGroupInfoKHR      = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_DEVICE_GROUP_INFO_KHR,
    eBindImageMemoryDeviceGroupInfoKHR       = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_DEVICE_GROUP_INFO_KHR,
    eValidationFlagsEXT                      = VK_STRUCTURE_TYPE_VALIDATION_FLAGS_EXT,
#if defined( VK_USE_PLATFORM_VI_NN )
    eViSurfaceCreateInfoNN = VK_STRUCTURE_TYPE_VI_SURFACE_CREATE_INFO_NN,
#endif /*VK_USE_PLATFORM_VI_NN*/
    ePhysicalDeviceTextureCompressionAstcHdrFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES_EXT,
    eImageViewAstcDecodeModeEXT                         = VK_STRUCTURE_TYPE_IMAGE_VIEW_ASTC_DECODE_MODE_EXT,
    ePhysicalDeviceAstcDecodeFeaturesEXT                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ASTC_DECODE_FEATURES_EXT,
    ePipelineRobustnessCreateInfoEXT                    = VK_STRUCTURE_TYPE_PIPELINE_ROBUSTNESS_CREATE_INFO_EXT,
    ePhysicalDevicePipelineRobustnessFeaturesEXT        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES_EXT,
    ePhysicalDevicePipelineRobustnessPropertiesEXT      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_PROPERTIES_EXT,
    ePhysicalDeviceGroupPropertiesKHR                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES_KHR,
    eDeviceGroupDeviceCreateInfoKHR                     = VK_STRUCTURE_TYPE_DEVICE_GROUP_DEVICE_CREATE_INFO_KHR,
    ePhysicalDeviceExternalImageFormatInfoKHR           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO_KHR,
    eExternalImageFormatPropertiesKHR                   = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES_KHR,
    ePhysicalDeviceExternalBufferInfoKHR                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO_KHR,
    eExternalBufferPropertiesKHR                        = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES_KHR,
    ePhysicalDeviceIdPropertiesKHR                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES_KHR,
    eExternalMemoryBufferCreateInfoKHR                  = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO_KHR,
    eExternalMemoryImageCreateInfoKHR                   = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO_KHR,
    eExportMemoryAllocateInfoKHR                        = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR,
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
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    ePhysicalDeviceExternalSemaphoreInfoKHR = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO_KHR,
    eExternalSemaphorePropertiesKHR         = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES_KHR,
    eExportSemaphoreCreateInfoKHR           = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR,
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    eImportSemaphoreWin32HandleInfoKHR = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR,
    eExportSemaphoreWin32HandleInfoKHR = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR,
    eD3D12FenceSubmitInfoKHR           = VK_STRUCTURE_TYPE_D3D12_FENCE_SUBMIT_INFO_KHR,
    eSemaphoreGetWin32HandleInfoKHR    = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR,
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    eImportSemaphoreFdInfoKHR                              = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
    eSemaphoreGetFdInfoKHR                                 = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
    ePhysicalDevicePushDescriptorPropertiesKHR             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR,
    eCommandBufferInheritanceConditionalRenderingInfoEXT   = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_CONDITIONAL_RENDERING_INFO_EXT,
    ePhysicalDeviceConditionalRenderingFeaturesEXT         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT,
    eConditionalRenderingBeginInfoEXT                      = VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT,
    ePhysicalDeviceShaderFloat16Int8FeaturesKHR            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR,
    ePhysicalDeviceFloat16Int8FeaturesKHR                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR,
    ePhysicalDevice16BitStorageFeaturesKHR                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR,
    ePresentRegionsKHR                                     = VK_STRUCTURE_TYPE_PRESENT_REGIONS_KHR,
    eDescriptorUpdateTemplateCreateInfoKHR                 = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO_KHR,
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
    ePhysicalDeviceImagelessFramebufferFeaturesKHR         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES_KHR,
    eFramebufferAttachmentsCreateInfoKHR                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENTS_CREATE_INFO_KHR,
    eFramebufferAttachmentImageInfoKHR                     = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENT_IMAGE_INFO_KHR,
    eRenderPassAttachmentBeginInfoKHR                      = VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO_KHR,
    eAttachmentDescription2KHR                             = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2_KHR,
    eAttachmentReference2KHR                               = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2_KHR,
    eSubpassDescription2KHR                                = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2_KHR,
    eSubpassDependency2KHR                                 = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2_KHR,
    eRenderPassCreateInfo2KHR                              = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2_KHR,
    eSubpassBeginInfoKHR                                   = VK_STRUCTURE_TYPE_SUBPASS_BEGIN_INFO_KHR,
    eSubpassEndInfoKHR                                     = VK_STRUCTURE_TYPE_SUBPASS_END_INFO_KHR,
    eSharedPresentSurfaceCapabilitiesKHR                   = VK_STRUCTURE_TYPE_SHARED_PRESENT_SURFACE_CAPABILITIES_KHR,
    ePhysicalDeviceExternalFenceInfoKHR                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_FENCE_INFO_KHR,
    eExternalFencePropertiesKHR                            = VK_STRUCTURE_TYPE_EXTERNAL_FENCE_PROPERTIES_KHR,
    eExportFenceCreateInfoKHR                              = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO_KHR,
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    eImportFenceWin32HandleInfoKHR = VK_STRUCTURE_TYPE_IMPORT_FENCE_WIN32_HANDLE_INFO_KHR,
    eExportFenceWin32HandleInfoKHR = VK_STRUCTURE_TYPE_EXPORT_FENCE_WIN32_HANDLE_INFO_KHR,
    eFenceGetWin32HandleInfoKHR    = VK_STRUCTURE_TYPE_FENCE_GET_WIN32_HANDLE_INFO_KHR,
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    eImportFenceFdInfoKHR                               = VK_STRUCTURE_TYPE_IMPORT_FENCE_FD_INFO_KHR,
    eFenceGetFdInfoKHR                                  = VK_STRUCTURE_TYPE_FENCE_GET_FD_INFO_KHR,
    ePhysicalDevicePerformanceQueryFeaturesKHR          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PERFORMANCE_QUERY_FEATURES_KHR,
    ePhysicalDevicePerformanceQueryPropertiesKHR        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PERFORMANCE_QUERY_PROPERTIES_KHR,
    eQueryPoolPerformanceCreateInfoKHR                  = VK_STRUCTURE_TYPE_QUERY_POOL_PERFORMANCE_CREATE_INFO_KHR,
    ePerformanceQuerySubmitInfoKHR                      = VK_STRUCTURE_TYPE_PERFORMANCE_QUERY_SUBMIT_INFO_KHR,
    eAcquireProfilingLockInfoKHR                        = VK_STRUCTURE_TYPE_ACQUIRE_PROFILING_LOCK_INFO_KHR,
    ePerformanceCounterKHR                              = VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_KHR,
    ePerformanceCounterDescriptionKHR                   = VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_DESCRIPTION_KHR,
    ePhysicalDevicePointClippingPropertiesKHR           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_POINT_CLIPPING_PROPERTIES_KHR,
    eRenderPassInputAttachmentAspectCreateInfoKHR       = VK_STRUCTURE_TYPE_RENDER_PASS_INPUT_ATTACHMENT_ASPECT_CREATE_INFO_KHR,
    eImageViewUsageCreateInfoKHR                        = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO_KHR,
    ePipelineTessellationDomainOriginStateCreateInfoKHR = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_DOMAIN_ORIGIN_STATE_CREATE_INFO_KHR,
    ePhysicalDeviceSurfaceInfo2KHR                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR,
    eSurfaceCapabilities2KHR                            = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR,
    eSurfaceFormat2KHR                                  = VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR,
    ePhysicalDeviceVariablePointersFeaturesKHR          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES_KHR,
    ePhysicalDeviceVariablePointerFeaturesKHR           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTER_FEATURES_KHR,
    eDisplayProperties2KHR                              = VK_STRUCTURE_TYPE_DISPLAY_PROPERTIES_2_KHR,
    eDisplayPlaneProperties2KHR                         = VK_STRUCTURE_TYPE_DISPLAY_PLANE_PROPERTIES_2_KHR,
    eDisplayModeProperties2KHR                          = VK_STRUCTURE_TYPE_DISPLAY_MODE_PROPERTIES_2_KHR,
    eDisplayPlaneInfo2KHR                               = VK_STRUCTURE_TYPE_DISPLAY_PLANE_INFO_2_KHR,
    eDisplayPlaneCapabilities2KHR                       = VK_STRUCTURE_TYPE_DISPLAY_PLANE_CAPABILITIES_2_KHR,
#if defined( VK_USE_PLATFORM_IOS_MVK )
    eIosSurfaceCreateInfoMVK = VK_STRUCTURE_TYPE_IOS_SURFACE_CREATE_INFO_MVK,
#endif /*VK_USE_PLATFORM_IOS_MVK*/
#if defined( VK_USE_PLATFORM_MACOS_MVK )
    eMacosSurfaceCreateInfoMVK = VK_STRUCTURE_TYPE_MACOS_SURFACE_CREATE_INFO_MVK,
#endif /*VK_USE_PLATFORM_MACOS_MVK*/
    eMemoryDedicatedRequirementsKHR     = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR,
    eMemoryDedicatedAllocateInfoKHR     = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR,
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
    ePhysicalDeviceSamplerFilterMinmaxPropertiesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_FILTER_MINMAX_PROPERTIES_EXT,
    eSamplerReductionModeCreateInfoEXT              = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    ePhysicalDeviceShaderEnqueueFeaturesAMDX   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ENQUEUE_FEATURES_AMDX,
    ePhysicalDeviceShaderEnqueuePropertiesAMDX = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ENQUEUE_PROPERTIES_AMDX,
    eExecutionGraphPipelineScratchSizeAMDX     = VK_STRUCTURE_TYPE_EXECUTION_GRAPH_PIPELINE_SCRATCH_SIZE_AMDX,
    eExecutionGraphPipelineCreateInfoAMDX      = VK_STRUCTURE_TYPE_EXECUTION_GRAPH_PIPELINE_CREATE_INFO_AMDX,
    ePipelineShaderStageNodeCreateInfoAMDX     = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NODE_CREATE_INFO_AMDX,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    ePhysicalDeviceInlineUniformBlockFeaturesEXT          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES_EXT,
    ePhysicalDeviceInlineUniformBlockPropertiesEXT        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_PROPERTIES_EXT,
    eWriteDescriptorSetInlineUniformBlockEXT              = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK_EXT,
    eDescriptorPoolInlineUniformBlockCreateInfoEXT        = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_INLINE_UNIFORM_BLOCK_CREATE_INFO_EXT,
    eSampleLocationsInfoEXT                               = VK_STRUCTURE_TYPE_SAMPLE_LOCATIONS_INFO_EXT,
    eRenderPassSampleLocationsBeginInfoEXT                = VK_STRUCTURE_TYPE_RENDER_PASS_SAMPLE_LOCATIONS_BEGIN_INFO_EXT,
    ePipelineSampleLocationsStateCreateInfoEXT            = VK_STRUCTURE_TYPE_PIPELINE_SAMPLE_LOCATIONS_STATE_CREATE_INFO_EXT,
    ePhysicalDeviceSampleLocationsPropertiesEXT           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLE_LOCATIONS_PROPERTIES_EXT,
    eMultisamplePropertiesEXT                             = VK_STRUCTURE_TYPE_MULTISAMPLE_PROPERTIES_EXT,
    eBufferMemoryRequirementsInfo2KHR                     = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2_KHR,
    eImageMemoryRequirementsInfo2KHR                      = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2_KHR,
    eImageSparseMemoryRequirementsInfo2KHR                = VK_STRUCTURE_TYPE_IMAGE_SPARSE_MEMORY_REQUIREMENTS_INFO_2_KHR,
    eMemoryRequirements2KHR                               = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR,
    eSparseImageMemoryRequirements2KHR                    = VK_STRUCTURE_TYPE_SPARSE_IMAGE_MEMORY_REQUIREMENTS_2_KHR,
    eImageFormatListCreateInfoKHR                         = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO_KHR,
    ePhysicalDeviceBlendOperationAdvancedFeaturesEXT      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BLEND_OPERATION_ADVANCED_FEATURES_EXT,
    ePhysicalDeviceBlendOperationAdvancedPropertiesEXT    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BLEND_OPERATION_ADVANCED_PROPERTIES_EXT,
    ePipelineColorBlendAdvancedStateCreateInfoEXT         = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_ADVANCED_STATE_CREATE_INFO_EXT,
    ePipelineCoverageToColorStateCreateInfoNV             = VK_STRUCTURE_TYPE_PIPELINE_COVERAGE_TO_COLOR_STATE_CREATE_INFO_NV,
    eWriteDescriptorSetAccelerationStructureKHR           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
    eAccelerationStructureBuildGeometryInfoKHR            = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
    eAccelerationStructureDeviceAddressInfoKHR            = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
    eAccelerationStructureGeometryAabbsDataKHR            = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR,
    eAccelerationStructureGeometryInstancesDataKHR        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
    eAccelerationStructureGeometryTrianglesDataKHR        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
    eAccelerationStructureGeometryKHR                     = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
    eAccelerationStructureVersionInfoKHR                  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_VERSION_INFO_KHR,
    eCopyAccelerationStructureInfoKHR                     = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR,
    eCopyAccelerationStructureToMemoryInfoKHR             = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_TO_MEMORY_INFO_KHR,
    eCopyMemoryToAccelerationStructureInfoKHR             = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_ACCELERATION_STRUCTURE_INFO_KHR,
    ePhysicalDeviceAccelerationStructureFeaturesKHR       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
    ePhysicalDeviceAccelerationStructurePropertiesKHR     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR,
    eAccelerationStructureCreateInfoKHR                   = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
    eAccelerationStructureBuildSizesInfoKHR               = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
    ePhysicalDeviceRayTracingPipelineFeaturesKHR          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
    ePhysicalDeviceRayTracingPipelinePropertiesKHR        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
    eRayTracingPipelineCreateInfoKHR                      = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
    eRayTracingShaderGroupCreateInfoKHR                   = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
    eRayTracingPipelineInterfaceCreateInfoKHR             = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_INTERFACE_CREATE_INFO_KHR,
    ePhysicalDeviceRayQueryFeaturesKHR                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
    ePipelineCoverageModulationStateCreateInfoNV          = VK_STRUCTURE_TYPE_PIPELINE_COVERAGE_MODULATION_STATE_CREATE_INFO_NV,
    ePhysicalDeviceShaderSmBuiltinsFeaturesNV             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_FEATURES_NV,
    ePhysicalDeviceShaderSmBuiltinsPropertiesNV           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV,
    eSamplerYcbcrConversionCreateInfoKHR                  = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO_KHR,
    eSamplerYcbcrConversionInfoKHR                        = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO_KHR,
    eBindImagePlaneMemoryInfoKHR                          = VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO_KHR,
    eImagePlaneMemoryRequirementsInfoKHR                  = VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO_KHR,
    ePhysicalDeviceSamplerYcbcrConversionFeaturesKHR      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES_KHR,
    eSamplerYcbcrConversionImageFormatPropertiesKHR       = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES_KHR,
    eBindBufferMemoryInfoKHR                              = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO_KHR,
    eBindImageMemoryInfoKHR                               = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO_KHR,
    eDrmFormatModifierPropertiesListEXT                   = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT,
    ePhysicalDeviceImageDrmFormatModifierInfoEXT          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT,
    eImageDrmFormatModifierListCreateInfoEXT              = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
    eImageDrmFormatModifierExplicitCreateInfoEXT          = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
    eImageDrmFormatModifierPropertiesEXT                  = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT,
    eDrmFormatModifierPropertiesList2EXT                  = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_2_EXT,
    eValidationCacheCreateInfoEXT                         = VK_STRUCTURE_TYPE_VALIDATION_CACHE_CREATE_INFO_EXT,
    eShaderModuleValidationCacheCreateInfoEXT             = VK_STRUCTURE_TYPE_SHADER_MODULE_VALIDATION_CACHE_CREATE_INFO_EXT,
    eDescriptorSetLayoutBindingFlagsCreateInfoEXT         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT,
    ePhysicalDeviceDescriptorIndexingFeaturesEXT          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT,
    ePhysicalDeviceDescriptorIndexingPropertiesEXT        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES_EXT,
    eDescriptorSetVariableDescriptorCountAllocateInfoEXT  = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT,
    eDescriptorSetVariableDescriptorCountLayoutSupportEXT = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_LAYOUT_SUPPORT_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    ePhysicalDevicePortabilitySubsetFeaturesKHR   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR,
    ePhysicalDevicePortabilitySubsetPropertiesKHR = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_PROPERTIES_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    ePipelineViewportShadingRateImageStateCreateInfoNV    = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_SHADING_RATE_IMAGE_STATE_CREATE_INFO_NV,
    ePhysicalDeviceShadingRateImageFeaturesNV             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADING_RATE_IMAGE_FEATURES_NV,
    ePhysicalDeviceShadingRateImagePropertiesNV           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADING_RATE_IMAGE_PROPERTIES_NV,
    ePipelineViewportCoarseSampleOrderStateCreateInfoNV   = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_COARSE_SAMPLE_ORDER_STATE_CREATE_INFO_NV,
    eRayTracingPipelineCreateInfoNV                       = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_NV,
    eAccelerationStructureCreateInfoNV                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV,
    eGeometryNV                                           = VK_STRUCTURE_TYPE_GEOMETRY_NV,
    eGeometryTrianglesNV                                  = VK_STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV,
    eGeometryAabbNV                                       = VK_STRUCTURE_TYPE_GEOMETRY_AABB_NV,
    eBindAccelerationStructureMemoryInfoNV                = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV,
    eWriteDescriptorSetAccelerationStructureNV            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_NV,
    eAccelerationStructureMemoryRequirementsInfoNV        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV,
    ePhysicalDeviceRayTracingPropertiesNV                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PROPERTIES_NV,
    eRayTracingShaderGroupCreateInfoNV                    = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV,
    eAccelerationStructureInfoNV                          = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV,
    ePhysicalDeviceRepresentativeFragmentTestFeaturesNV   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_REPRESENTATIVE_FRAGMENT_TEST_FEATURES_NV,
    ePipelineRepresentativeFragmentTestStateCreateInfoNV  = VK_STRUCTURE_TYPE_PIPELINE_REPRESENTATIVE_FRAGMENT_TEST_STATE_CREATE_INFO_NV,
    ePhysicalDeviceMaintenance3PropertiesKHR              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES_KHR,
    eDescriptorSetLayoutSupportKHR                        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT_KHR,
    ePhysicalDeviceImageViewImageFormatInfoEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_VIEW_IMAGE_FORMAT_INFO_EXT,
    eFilterCubicImageViewImageFormatPropertiesEXT         = VK_STRUCTURE_TYPE_FILTER_CUBIC_IMAGE_VIEW_IMAGE_FORMAT_PROPERTIES_EXT,
    eDeviceQueueGlobalPriorityCreateInfoEXT               = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_EXT,
    ePhysicalDeviceShaderSubgroupExtendedTypesFeaturesKHR = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES_KHR,
    ePhysicalDevice8BitStorageFeaturesKHR                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR,
    eImportMemoryHostPointerInfoEXT                       = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
    eMemoryHostPointerPropertiesEXT                       = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT,
    ePhysicalDeviceExternalMemoryHostPropertiesEXT        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT,
    ePhysicalDeviceShaderAtomicInt64FeaturesKHR           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES_KHR,
    ePhysicalDeviceShaderClockFeaturesKHR                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR,
    ePipelineCompilerControlCreateInfoAMD                 = VK_STRUCTURE_TYPE_PIPELINE_COMPILER_CONTROL_CREATE_INFO_AMD,
    eCalibratedTimestampInfoEXT                           = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT,
    ePhysicalDeviceShaderCorePropertiesAMD                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD,
    eVideoDecodeH265CapabilitiesKHR                       = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_CAPABILITIES_KHR,
    eVideoDecodeH265SessionParametersCreateInfoKHR        = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_CREATE_INFO_KHR,
    eVideoDecodeH265SessionParametersAddInfoKHR           = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_ADD_INFO_KHR,
    eVideoDecodeH265ProfileInfoKHR                        = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PROFILE_INFO_KHR,
    eVideoDecodeH265PictureInfoKHR                        = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PICTURE_INFO_KHR,
    eVideoDecodeH265DpbSlotInfoKHR                        = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR,
    eDeviceQueueGlobalPriorityCreateInfoKHR               = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR,
    ePhysicalDeviceGlobalPriorityQueryFeaturesKHR         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GLOBAL_PRIORITY_QUERY_FEATURES_KHR,
    eQueueFamilyGlobalPriorityPropertiesKHR               = VK_STRUCTURE_TYPE_QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES_KHR,
    eDeviceMemoryOverallocationCreateInfoAMD              = VK_STRUCTURE_TYPE_DEVICE_MEMORY_OVERALLOCATION_CREATE_INFO_AMD,
    ePhysicalDeviceVertexAttributeDivisorPropertiesEXT    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_PROPERTIES_EXT,
    ePipelineVertexInputDivisorStateCreateInfoEXT         = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT,
    ePhysicalDeviceVertexAttributeDivisorFeaturesEXT      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT,
#if defined( VK_USE_PLATFORM_GGP )
    ePresentFrameTokenGGP = VK_STRUCTURE_TYPE_PRESENT_FRAME_TOKEN_GGP,
#endif /*VK_USE_PLATFORM_GGP*/
    ePipelineCreationFeedbackCreateInfoEXT              = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO_EXT,
    ePhysicalDeviceDriverPropertiesKHR                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES_KHR,
    ePhysicalDeviceFloatControlsPropertiesKHR           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES_KHR,
    ePhysicalDeviceDepthStencilResolvePropertiesKHR     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES_KHR,
    eSubpassDescriptionDepthStencilResolveKHR           = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE_KHR,
    ePhysicalDeviceComputeShaderDerivativesFeaturesNV   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_NV,
    ePhysicalDeviceMeshShaderFeaturesNV                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV,
    ePhysicalDeviceMeshShaderPropertiesNV               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_NV,
    ePhysicalDeviceFragmentShaderBarycentricFeaturesNV  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_NV,
    ePhysicalDeviceShaderImageFootprintFeaturesNV       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_FOOTPRINT_FEATURES_NV,
    ePipelineViewportExclusiveScissorStateCreateInfoNV  = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_EXCLUSIVE_SCISSOR_STATE_CREATE_INFO_NV,
    ePhysicalDeviceExclusiveScissorFeaturesNV           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXCLUSIVE_SCISSOR_FEATURES_NV,
    eCheckpointDataNV                                   = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV,
    eQueueFamilyCheckpointPropertiesNV                  = VK_STRUCTURE_TYPE_QUEUE_FAMILY_CHECKPOINT_PROPERTIES_NV,
    ePhysicalDeviceTimelineSemaphoreFeaturesKHR         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR,
    ePhysicalDeviceTimelineSemaphorePropertiesKHR       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES_KHR,
    eSemaphoreTypeCreateInfoKHR                         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR,
    eTimelineSemaphoreSubmitInfoKHR                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR,
    eSemaphoreWaitInfoKHR                               = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO_KHR,
    eSemaphoreSignalInfoKHR                             = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO_KHR,
    ePhysicalDeviceShaderIntegerFunctions2FeaturesINTEL = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_FUNCTIONS_2_FEATURES_INTEL,
    eQueryPoolPerformanceQueryCreateInfoINTEL           = VK_STRUCTURE_TYPE_QUERY_POOL_PERFORMANCE_QUERY_CREATE_INFO_INTEL,
    eQueryPoolCreateInfoINTEL                           = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO_INTEL,
    eInitializePerformanceApiInfoINTEL                  = VK_STRUCTURE_TYPE_INITIALIZE_PERFORMANCE_API_INFO_INTEL,
    ePerformanceMarkerInfoINTEL                         = VK_STRUCTURE_TYPE_PERFORMANCE_MARKER_INFO_INTEL,
    ePerformanceStreamMarkerInfoINTEL                   = VK_STRUCTURE_TYPE_PERFORMANCE_STREAM_MARKER_INFO_INTEL,
    ePerformanceOverrideInfoINTEL                       = VK_STRUCTURE_TYPE_PERFORMANCE_OVERRIDE_INFO_INTEL,
    ePerformanceConfigurationAcquireInfoINTEL           = VK_STRUCTURE_TYPE_PERFORMANCE_CONFIGURATION_ACQUIRE_INFO_INTEL,
    ePhysicalDeviceVulkanMemoryModelFeaturesKHR         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES_KHR,
    ePhysicalDevicePciBusInfoPropertiesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT,
    eDisplayNativeHdrSurfaceCapabilitiesAMD             = VK_STRUCTURE_TYPE_DISPLAY_NATIVE_HDR_SURFACE_CAPABILITIES_AMD,
    eSwapchainDisplayNativeHdrCreateInfoAMD             = VK_STRUCTURE_TYPE_SWAPCHAIN_DISPLAY_NATIVE_HDR_CREATE_INFO_AMD,
#if defined( VK_USE_PLATFORM_FUCHSIA )
    eImagepipeSurfaceCreateInfoFUCHSIA = VK_STRUCTURE_TYPE_IMAGEPIPE_SURFACE_CREATE_INFO_FUCHSIA,
#endif /*VK_USE_PLATFORM_FUCHSIA*/
    ePhysicalDeviceShaderTerminateInvocationFeaturesKHR = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES_KHR,
#if defined( VK_USE_PLATFORM_METAL_EXT )
    eMetalSurfaceCreateInfoEXT = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT,
#endif /*VK_USE_PLATFORM_METAL_EXT*/
    ePhysicalDeviceFragmentDensityMapFeaturesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_FEATURES_EXT,
    ePhysicalDeviceFragmentDensityMapPropertiesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_PROPERTIES_EXT,
    eRenderPassFragmentDensityMapCreateInfoEXT                = VK_STRUCTURE_TYPE_RENDER_PASS_FRAGMENT_DENSITY_MAP_CREATE_INFO_EXT,
    ePhysicalDeviceScalarBlockLayoutFeaturesEXT               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES_EXT,
    ePhysicalDeviceSubgroupSizeControlPropertiesEXT           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT,
    ePipelineShaderStageRequiredSubgroupSizeCreateInfoEXT     = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
    ePhysicalDeviceSubgroupSizeControlFeaturesEXT             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT,
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
    ePhysicalDeviceSeparateDepthStencilLayoutsFeaturesKHR     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES_KHR,
    eAttachmentReferenceStencilLayoutKHR                      = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_STENCIL_LAYOUT_KHR,
    eAttachmentDescriptionStencilLayoutKHR                    = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_STENCIL_LAYOUT_KHR,
    ePhysicalDeviceBufferDeviceAddressFeaturesEXT             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_EXT,
    ePhysicalDeviceBufferAddressFeaturesEXT                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_ADDRESS_FEATURES_EXT,
    eBufferDeviceAddressInfoEXT                               = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_EXT,
    eBufferDeviceAddressCreateInfoEXT                         = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_CREATE_INFO_EXT,
    ePhysicalDeviceToolPropertiesEXT                          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TOOL_PROPERTIES_EXT,
    eImageStencilUsageCreateInfoEXT                           = VK_STRUCTURE_TYPE_IMAGE_STENCIL_USAGE_CREATE_INFO_EXT,
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
    ePhysicalDeviceUniformBufferStandardLayoutFeaturesKHR     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES_KHR,
    ePhysicalDeviceProvokingVertexFeaturesEXT                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT,
    ePipelineRasterizationProvokingVertexStateCreateInfoEXT   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT,
    ePhysicalDeviceProvokingVertexPropertiesEXT               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_PROPERTIES_EXT,
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    eSurfaceFullScreenExclusiveInfoEXT         = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_INFO_EXT,
    eSurfaceCapabilitiesFullScreenExclusiveEXT = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_FULL_SCREEN_EXCLUSIVE_EXT,
    eSurfaceFullScreenExclusiveWin32InfoEXT    = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_WIN32_INFO_EXT,
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    eHeadlessSurfaceCreateInfoEXT                            = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT,
    ePhysicalDeviceBufferDeviceAddressFeaturesKHR            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR,
    eBufferDeviceAddressInfoKHR                              = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR,
    eBufferOpaqueCaptureAddressCreateInfoKHR                 = VK_STRUCTURE_TYPE_BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO_KHR,
    eMemoryOpaqueCaptureAddressAllocateInfoKHR               = VK_STRUCTURE_TYPE_MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO_KHR,
    eDeviceMemoryOpaqueCaptureAddressInfoKHR                 = VK_STRUCTURE_TYPE_DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO_KHR,
    ePhysicalDeviceLineRasterizationFeaturesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_EXT,
    ePipelineRasterizationLineStateCreateInfoEXT             = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO_EXT,
    ePhysicalDeviceLineRasterizationPropertiesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_PROPERTIES_EXT,
    ePhysicalDeviceShaderAtomicFloatFeaturesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
    ePhysicalDeviceHostQueryResetFeaturesEXT                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES_EXT,
    ePhysicalDeviceIndexTypeUint8FeaturesEXT                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT,
    ePhysicalDeviceExtendedDynamicStateFeaturesEXT           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT,
    ePhysicalDevicePipelineExecutablePropertiesFeaturesKHR   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR,
    ePipelineInfoKHR                                         = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR,
    ePipelineExecutablePropertiesKHR                         = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR,
    ePipelineExecutableInfoKHR                               = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR,
    ePipelineExecutableStatisticKHR                          = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR,
    ePipelineExecutableInternalRepresentationKHR             = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR,
    ePhysicalDeviceHostImageCopyFeaturesEXT                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT,
    ePhysicalDeviceHostImageCopyPropertiesEXT                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_PROPERTIES_EXT,
    eMemoryToImageCopyEXT                                    = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY_EXT,
    eImageToMemoryCopyEXT                                    = VK_STRUCTURE_TYPE_IMAGE_TO_MEMORY_COPY_EXT,
    eCopyImageToMemoryInfoEXT                                = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_MEMORY_INFO_EXT,
    eCopyMemoryToImageInfoEXT                                = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO_EXT,
    eHostImageLayoutTransitionInfoEXT                        = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO_EXT,
    eCopyImageToImageInfoEXT                                 = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_IMAGE_INFO_EXT,
    eSubresourceHostMemcpySizeEXT                            = VK_STRUCTURE_TYPE_SUBRESOURCE_HOST_MEMCPY_SIZE_EXT,
    eHostImageCopyDevicePerformanceQueryEXT                  = VK_STRUCTURE_TYPE_HOST_IMAGE_COPY_DEVICE_PERFORMANCE_QUERY_EXT,
    eMemoryMapInfoKHR                                        = VK_STRUCTURE_TYPE_MEMORY_MAP_INFO_KHR,
    eMemoryUnmapInfoKHR                                      = VK_STRUCTURE_TYPE_MEMORY_UNMAP_INFO_KHR,
    ePhysicalDeviceShaderAtomicFloat2FeaturesEXT             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT,
    eSurfacePresentModeEXT                                   = VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_EXT,
    eSurfacePresentScalingCapabilitiesEXT                    = VK_STRUCTURE_TYPE_SURFACE_PRESENT_SCALING_CAPABILITIES_EXT,
    eSurfacePresentModeCompatibilityEXT                      = VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_COMPATIBILITY_EXT,
    ePhysicalDeviceSwapchainMaintenance1FeaturesEXT          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT,
    eSwapchainPresentFenceInfoEXT                            = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT,
    eSwapchainPresentModesCreateInfoEXT                      = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT,
    eSwapchainPresentModeInfoEXT                             = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_EXT,
    eSwapchainPresentScalingCreateInfoEXT                    = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_SCALING_CREATE_INFO_EXT,
    eReleaseSwapchainImagesInfoEXT                           = VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_EXT,
    ePhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES_EXT,
    ePhysicalDeviceDeviceGeneratedCommandsPropertiesNV       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_PROPERTIES_NV,
    eGraphicsShaderGroupCreateInfoNV                         = VK_STRUCTURE_TYPE_GRAPHICS_SHADER_GROUP_CREATE_INFO_NV,
    eGraphicsPipelineShaderGroupsCreateInfoNV                = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_SHADER_GROUPS_CREATE_INFO_NV,
    eIndirectCommandsLayoutTokenNV                           = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_NV,
    eIndirectCommandsLayoutCreateInfoNV                      = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_NV,
    eGeneratedCommandsInfoNV                                 = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_NV,
    eGeneratedCommandsMemoryRequirementsInfoNV               = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_NV,
    ePhysicalDeviceDeviceGeneratedCommandsFeaturesNV         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_NV,
    ePhysicalDeviceInheritedViewportScissorFeaturesNV        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INHERITED_VIEWPORT_SCISSOR_FEATURES_NV,
    eCommandBufferInheritanceViewportScissorInfoNV           = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_VIEWPORT_SCISSOR_INFO_NV,
    ePhysicalDeviceShaderIntegerDotProductFeaturesKHR        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR,
    ePhysicalDeviceShaderIntegerDotProductPropertiesKHR      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES_KHR,
    ePhysicalDeviceTexelBufferAlignmentFeaturesEXT           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_FEATURES_EXT,
    ePhysicalDeviceTexelBufferAlignmentPropertiesEXT         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_PROPERTIES_EXT,
    eCommandBufferInheritanceRenderPassTransformInfoQCOM     = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDER_PASS_TRANSFORM_INFO_QCOM,
    eRenderPassTransformBeginInfoQCOM                        = VK_STRUCTURE_TYPE_RENDER_PASS_TRANSFORM_BEGIN_INFO_QCOM,
    ePhysicalDeviceDepthBiasControlFeaturesEXT               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_BIAS_CONTROL_FEATURES_EXT,
    eDepthBiasInfoEXT                                        = VK_STRUCTURE_TYPE_DEPTH_BIAS_INFO_EXT,
    eDepthBiasRepresentationInfoEXT                          = VK_STRUCTURE_TYPE_DEPTH_BIAS_REPRESENTATION_INFO_EXT,
    ePhysicalDeviceDeviceMemoryReportFeaturesEXT             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_MEMORY_REPORT_FEATURES_EXT,
    eDeviceDeviceMemoryReportCreateInfoEXT                   = VK_STRUCTURE_TYPE_DEVICE_DEVICE_MEMORY_REPORT_CREATE_INFO_EXT,
    eDeviceMemoryReportCallbackDataEXT                       = VK_STRUCTURE_TYPE_DEVICE_MEMORY_REPORT_CALLBACK_DATA_EXT,
    ePhysicalDeviceRobustness2FeaturesEXT                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
    ePhysicalDeviceRobustness2PropertiesEXT                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_PROPERTIES_EXT,
    eSamplerCustomBorderColorCreateInfoEXT                   = VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT,
    ePhysicalDeviceCustomBorderColorPropertiesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_PROPERTIES_EXT,
    ePhysicalDeviceCustomBorderColorFeaturesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT,
    ePipelineLibraryCreateInfoKHR                            = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR,
    ePhysicalDevicePresentBarrierFeaturesNV                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_BARRIER_FEATURES_NV,
    eSurfaceCapabilitiesPresentBarrierNV                     = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_PRESENT_BARRIER_NV,
    eSwapchainPresentBarrierCreateInfoNV                     = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_BARRIER_CREATE_INFO_NV,
    ePresentIdKHR                                            = VK_STRUCTURE_TYPE_PRESENT_ID_KHR,
    ePhysicalDevicePresentIdFeaturesKHR                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR,
    ePhysicalDevicePrivateDataFeaturesEXT                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES_EXT,
    eDevicePrivateDataCreateInfoEXT                          = VK_STRUCTURE_TYPE_DEVICE_PRIVATE_DATA_CREATE_INFO_EXT,
    ePrivateDataSlotCreateInfoEXT                            = VK_STRUCTURE_TYPE_PRIVATE_DATA_SLOT_CREATE_INFO_EXT,
    ePhysicalDevicePipelineCreationCacheControlFeaturesEXT   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoEncodeInfoKHR                           = VK_STRUCTURE_TYPE_VIDEO_ENCODE_INFO_KHR,
    eVideoEncodeRateControlInfoKHR                = VK_STRUCTURE_TYPE_VIDEO_ENCODE_RATE_CONTROL_INFO_KHR,
    eVideoEncodeRateControlLayerInfoKHR           = VK_STRUCTURE_TYPE_VIDEO_ENCODE_RATE_CONTROL_LAYER_INFO_KHR,
    eVideoEncodeCapabilitiesKHR                   = VK_STRUCTURE_TYPE_VIDEO_ENCODE_CAPABILITIES_KHR,
    eVideoEncodeUsageInfoKHR                      = VK_STRUCTURE_TYPE_VIDEO_ENCODE_USAGE_INFO_KHR,
    eQueryPoolVideoEncodeFeedbackCreateInfoKHR    = VK_STRUCTURE_TYPE_QUERY_POOL_VIDEO_ENCODE_FEEDBACK_CREATE_INFO_KHR,
    ePhysicalDeviceVideoEncodeQualityLevelInfoKHR = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_ENCODE_QUALITY_LEVEL_INFO_KHR,
    eVideoEncodeQualityLevelPropertiesKHR         = VK_STRUCTURE_TYPE_VIDEO_ENCODE_QUALITY_LEVEL_PROPERTIES_KHR,
    eVideoEncodeQualityLevelInfoKHR               = VK_STRUCTURE_TYPE_VIDEO_ENCODE_QUALITY_LEVEL_INFO_KHR,
    eVideoEncodeSessionParametersGetInfoKHR       = VK_STRUCTURE_TYPE_VIDEO_ENCODE_SESSION_PARAMETERS_GET_INFO_KHR,
    eVideoEncodeSessionParametersFeedbackInfoKHR  = VK_STRUCTURE_TYPE_VIDEO_ENCODE_SESSION_PARAMETERS_FEEDBACK_INFO_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    ePhysicalDeviceDiagnosticsConfigFeaturesNV = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DIAGNOSTICS_CONFIG_FEATURES_NV,
    eDeviceDiagnosticsConfigCreateInfoNV       = VK_STRUCTURE_TYPE_DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO_NV,
    eQueryLowLatencySupportNV                  = VK_STRUCTURE_TYPE_QUERY_LOW_LATENCY_SUPPORT_NV,
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
    eMemoryBarrier2KHR                                           = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR,
    eBufferMemoryBarrier2KHR                                     = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR,
    eImageMemoryBarrier2KHR                                      = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR,
    eDependencyInfoKHR                                           = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
    eSubmitInfo2KHR                                              = VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR,
    eSemaphoreSubmitInfoKHR                                      = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR,
    eCommandBufferSubmitInfoKHR                                  = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR,
    ePhysicalDeviceSynchronization2FeaturesKHR                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR,
    eQueueFamilyCheckpointProperties2NV                          = VK_STRUCTURE_TYPE_QUEUE_FAMILY_CHECKPOINT_PROPERTIES_2_NV,
    eCheckpointData2NV                                           = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_2_NV,
    ePhysicalDeviceDescriptorBufferPropertiesEXT                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT,
    ePhysicalDeviceDescriptorBufferDensityMapPropertiesEXT       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_DENSITY_MAP_PROPERTIES_EXT,
    ePhysicalDeviceDescriptorBufferFeaturesEXT                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT,
    eDescriptorAddressInfoEXT                                    = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
    eDescriptorGetInfoEXT                                        = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
    eBufferCaptureDescriptorDataInfoEXT                          = VK_STRUCTURE_TYPE_BUFFER_CAPTURE_DESCRIPTOR_DATA_INFO_EXT,
    eImageCaptureDescriptorDataInfoEXT                           = VK_STRUCTURE_TYPE_IMAGE_CAPTURE_DESCRIPTOR_DATA_INFO_EXT,
    eImageViewCaptureDescriptorDataInfoEXT                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CAPTURE_DESCRIPTOR_DATA_INFO_EXT,
    eSamplerCaptureDescriptorDataInfoEXT                         = VK_STRUCTURE_TYPE_SAMPLER_CAPTURE_DESCRIPTOR_DATA_INFO_EXT,
    eOpaqueCaptureDescriptorDataCreateInfoEXT                    = VK_STRUCTURE_TYPE_OPAQUE_CAPTURE_DESCRIPTOR_DATA_CREATE_INFO_EXT,
    eDescriptorBufferBindingInfoEXT                              = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
    eDescriptorBufferBindingPushDescriptorBufferHandleEXT        = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_PUSH_DESCRIPTOR_BUFFER_HANDLE_EXT,
    eAccelerationStructureCaptureDescriptorDataInfoEXT           = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CAPTURE_DESCRIPTOR_DATA_INFO_EXT,
    ePhysicalDeviceGraphicsPipelineLibraryFeaturesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT,
    ePhysicalDeviceGraphicsPipelineLibraryPropertiesEXT          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT,
    eGraphicsPipelineLibraryCreateInfoEXT                        = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
    ePhysicalDeviceShaderEarlyAndLateFragmentTestsFeaturesAMD    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_EARLY_AND_LATE_FRAGMENT_TESTS_FEATURES_AMD,
    ePhysicalDeviceFragmentShaderBarycentricFeaturesKHR          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR,
    ePhysicalDeviceFragmentShaderBarycentricPropertiesKHR        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_PROPERTIES_KHR,
    ePhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_FEATURES_KHR,
    ePhysicalDeviceZeroInitializeWorkgroupMemoryFeaturesKHR      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES_KHR,
    ePhysicalDeviceFragmentShadingRateEnumsPropertiesNV          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_ENUMS_PROPERTIES_NV,
    ePhysicalDeviceFragmentShadingRateEnumsFeaturesNV            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_ENUMS_FEATURES_NV,
    ePipelineFragmentShadingRateEnumStateCreateInfoNV            = VK_STRUCTURE_TYPE_PIPELINE_FRAGMENT_SHADING_RATE_ENUM_STATE_CREATE_INFO_NV,
    eAccelerationStructureGeometryMotionTrianglesDataNV          = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_MOTION_TRIANGLES_DATA_NV,
    ePhysicalDeviceRayTracingMotionBlurFeaturesNV                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MOTION_BLUR_FEATURES_NV,
    eAccelerationStructureMotionInfoNV                           = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MOTION_INFO_NV,
    ePhysicalDeviceMeshShaderFeaturesEXT                         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
    ePhysicalDeviceMeshShaderPropertiesEXT                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT,
    ePhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_YCBCR_2_PLANE_444_FORMATS_FEATURES_EXT,
    ePhysicalDeviceFragmentDensityMap2FeaturesEXT                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_2_FEATURES_EXT,
    ePhysicalDeviceFragmentDensityMap2PropertiesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_2_PROPERTIES_EXT,
    eCopyCommandTransformInfoQCOM                                = VK_STRUCTURE_TYPE_COPY_COMMAND_TRANSFORM_INFO_QCOM,
    ePhysicalDeviceImageRobustnessFeaturesEXT                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES_EXT,
    ePhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR,
    eCopyBufferInfo2KHR                                          = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2_KHR,
    eCopyImageInfo2KHR                                           = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2_KHR,
    eCopyBufferToImageInfo2KHR                                   = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2_KHR,
    eCopyImageToBufferInfo2KHR                                   = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2_KHR,
    eBlitImageInfo2KHR                                           = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2_KHR,
    eResolveImageInfo2KHR                                        = VK_STRUCTURE_TYPE_RESOLVE_IMAGE_INFO_2_KHR,
    eBufferCopy2KHR                                              = VK_STRUCTURE_TYPE_BUFFER_COPY_2_KHR,
    eImageCopy2KHR                                               = VK_STRUCTURE_TYPE_IMAGE_COPY_2_KHR,
    eImageBlit2KHR                                               = VK_STRUCTURE_TYPE_IMAGE_BLIT_2_KHR,
    eBufferImageCopy2KHR                                         = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2_KHR,
    eImageResolve2KHR                                            = VK_STRUCTURE_TYPE_IMAGE_RESOLVE_2_KHR,
    ePhysicalDeviceImageCompressionControlFeaturesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_FEATURES_EXT,
    eImageCompressionControlEXT                                  = VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_CONTROL_EXT,
    eSubresourceLayout2EXT                                       = VK_STRUCTURE_TYPE_SUBRESOURCE_LAYOUT_2_EXT,
    eImageSubresource2EXT                                        = VK_STRUCTURE_TYPE_IMAGE_SUBRESOURCE_2_EXT,
    eImageCompressionPropertiesEXT                               = VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_PROPERTIES_EXT,
    ePhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_FEATURES_EXT,
    ePhysicalDevice4444FormatsFeaturesEXT                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_4444_FORMATS_FEATURES_EXT,
    ePhysicalDeviceFaultFeaturesEXT                              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT,
    eDeviceFaultCountsEXT                                        = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT,
    eDeviceFaultInfoEXT                                          = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT,
    ePhysicalDeviceRasterizationOrderAttachmentAccessFeaturesARM = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_FEATURES_ARM,
    ePhysicalDeviceRgba10X6FormatsFeaturesEXT                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RGBA10X6_FORMATS_FEATURES_EXT,
#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
    eDirectfbSurfaceCreateInfoEXT = VK_STRUCTURE_TYPE_DIRECTFB_SURFACE_CREATE_INFO_EXT,
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/
    ePhysicalDeviceMutableDescriptorTypeFeaturesVALVE      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_VALVE,
    eMutableDescriptorTypeCreateInfoVALVE                  = VK_STRUCTURE_TYPE_MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_VALVE,
    ePhysicalDeviceVertexInputDynamicStateFeaturesEXT      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT,
    eVertexInputBindingDescription2EXT                     = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
    eVertexInputAttributeDescription2EXT                   = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
    ePhysicalDeviceDrmPropertiesEXT                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT,
    ePhysicalDeviceAddressBindingReportFeaturesEXT         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ADDRESS_BINDING_REPORT_FEATURES_EXT,
    eDeviceAddressBindingCallbackDataEXT                   = VK_STRUCTURE_TYPE_DEVICE_ADDRESS_BINDING_CALLBACK_DATA_EXT,
    ePhysicalDeviceDepthClipControlFeaturesEXT             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT,
    ePipelineViewportDepthClipControlCreateInfoEXT         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT,
    ePhysicalDevicePrimitiveTopologyListRestartFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVE_TOPOLOGY_LIST_RESTART_FEATURES_EXT,
    eFormatProperties3KHR                                  = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3_KHR,
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
    ePipelineInfoEXT                                            = VK_STRUCTURE_TYPE_PIPELINE_INFO_EXT,
    ePhysicalDeviceFrameBoundaryFeaturesEXT                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAME_BOUNDARY_FEATURES_EXT,
    eFrameBoundaryEXT                                           = VK_STRUCTURE_TYPE_FRAME_BOUNDARY_EXT,
    ePhysicalDeviceMultisampledRenderToSingleSampledFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_FEATURES_EXT,
    eSubpassResolvePerformanceQueryEXT                          = VK_STRUCTURE_TYPE_SUBPASS_RESOLVE_PERFORMANCE_QUERY_EXT,
    eMultisampledRenderToSingleSampledInfoEXT                   = VK_STRUCTURE_TYPE_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_INFO_EXT,
    ePhysicalDeviceExtendedDynamicState2FeaturesEXT             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT,
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
    eScreenSurfaceCreateInfoQNX = VK_STRUCTURE_TYPE_SCREEN_SURFACE_CREATE_INFO_QNX,
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/
    ePhysicalDeviceColorWriteEnableFeaturesEXT         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COLOR_WRITE_ENABLE_FEATURES_EXT,
    ePipelineColorWriteCreateInfoEXT                   = VK_STRUCTURE_TYPE_PIPELINE_COLOR_WRITE_CREATE_INFO_EXT,
    ePhysicalDevicePrimitivesGeneratedQueryFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVES_GENERATED_QUERY_FEATURES_EXT,
    ePhysicalDeviceRayTracingMaintenance1FeaturesKHR   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MAINTENANCE_1_FEATURES_KHR,
    ePhysicalDeviceGlobalPriorityQueryFeaturesEXT      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GLOBAL_PRIORITY_QUERY_FEATURES_EXT,
    eQueueFamilyGlobalPriorityPropertiesEXT            = VK_STRUCTURE_TYPE_QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES_EXT,
    ePhysicalDeviceImageViewMinLodFeaturesEXT          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_VIEW_MIN_LOD_FEATURES_EXT,
    eImageViewMinLodCreateInfoEXT                      = VK_STRUCTURE_TYPE_IMAGE_VIEW_MIN_LOD_CREATE_INFO_EXT,
    ePhysicalDeviceMultiDrawFeaturesEXT                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTI_DRAW_FEATURES_EXT,
    ePhysicalDeviceMultiDrawPropertiesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTI_DRAW_PROPERTIES_EXT,
    ePhysicalDeviceImage2DViewOf3DFeaturesEXT          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_2D_VIEW_OF_3D_FEATURES_EXT,
    ePhysicalDeviceShaderTileImageFeaturesEXT          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TILE_IMAGE_FEATURES_EXT,
    ePhysicalDeviceShaderTileImagePropertiesEXT        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TILE_IMAGE_PROPERTIES_EXT,
    eMicromapBuildInfoEXT                              = VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT,
    eMicromapVersionInfoEXT                            = VK_STRUCTURE_TYPE_MICROMAP_VERSION_INFO_EXT,
    eCopyMicromapInfoEXT                               = VK_STRUCTURE_TYPE_COPY_MICROMAP_INFO_EXT,
    eCopyMicromapToMemoryInfoEXT                       = VK_STRUCTURE_TYPE_COPY_MICROMAP_TO_MEMORY_INFO_EXT,
    eCopyMemoryToMicromapInfoEXT                       = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_MICROMAP_INFO_EXT,
    ePhysicalDeviceOpacityMicromapFeaturesEXT          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT,
    ePhysicalDeviceOpacityMicromapPropertiesEXT        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES_EXT,
    eMicromapCreateInfoEXT                             = VK_STRUCTURE_TYPE_MICROMAP_CREATE_INFO_EXT,
    eMicromapBuildSizesInfoEXT                         = VK_STRUCTURE_TYPE_MICROMAP_BUILD_SIZES_INFO_EXT,
    eAccelerationStructureTrianglesOpacityMicromapEXT  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    ePhysicalDeviceDisplacementMicromapFeaturesNV         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DISPLACEMENT_MICROMAP_FEATURES_NV,
    ePhysicalDeviceDisplacementMicromapPropertiesNV       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DISPLACEMENT_MICROMAP_PROPERTIES_NV,
    eAccelerationStructureTrianglesDisplacementMicromapNV = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_DISPLACEMENT_MICROMAP_NV,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    ePhysicalDeviceClusterCullingShaderFeaturesHUAWEI            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_CULLING_SHADER_FEATURES_HUAWEI,
    ePhysicalDeviceClusterCullingShaderPropertiesHUAWEI          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_CULLING_SHADER_PROPERTIES_HUAWEI,
    ePhysicalDeviceBorderColorSwizzleFeaturesEXT                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BORDER_COLOR_SWIZZLE_FEATURES_EXT,
    eSamplerBorderColorComponentMappingCreateInfoEXT             = VK_STRUCTURE_TYPE_SAMPLER_BORDER_COLOR_COMPONENT_MAPPING_CREATE_INFO_EXT,
    ePhysicalDevicePageableDeviceLocalMemoryFeaturesEXT          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PAGEABLE_DEVICE_LOCAL_MEMORY_FEATURES_EXT,
    ePhysicalDeviceMaintenance4FeaturesKHR                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES_KHR,
    ePhysicalDeviceMaintenance4PropertiesKHR                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES_KHR,
    eDeviceBufferMemoryRequirementsKHR                           = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS_KHR,
    eDeviceImageMemoryRequirementsKHR                            = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS_KHR,
    ePhysicalDeviceShaderCorePropertiesARM                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_ARM,
    ePhysicalDeviceImageSlicedViewOf3DFeaturesEXT                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_SLICED_VIEW_OF_3D_FEATURES_EXT,
    eImageViewSlicedCreateInfoEXT                                = VK_STRUCTURE_TYPE_IMAGE_VIEW_SLICED_CREATE_INFO_EXT,
    ePhysicalDeviceDescriptorSetHostMappingFeaturesVALVE         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_SET_HOST_MAPPING_FEATURES_VALVE,
    eDescriptorSetBindingReferenceVALVE                          = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_BINDING_REFERENCE_VALVE,
    eDescriptorSetLayoutHostMappingInfoVALVE                     = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_HOST_MAPPING_INFO_VALVE,
    ePhysicalDeviceDepthClampZeroOneFeaturesEXT                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLAMP_ZERO_ONE_FEATURES_EXT,
    ePhysicalDeviceNonSeamlessCubeMapFeaturesEXT                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_NON_SEAMLESS_CUBE_MAP_FEATURES_EXT,
    ePhysicalDeviceFragmentDensityMapOffsetFeaturesQCOM          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_OFFSET_FEATURES_QCOM,
    ePhysicalDeviceFragmentDensityMapOffsetPropertiesQCOM        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_OFFSET_PROPERTIES_QCOM,
    eSubpassFragmentDensityMapOffsetEndInfoQCOM                  = VK_STRUCTURE_TYPE_SUBPASS_FRAGMENT_DENSITY_MAP_OFFSET_END_INFO_QCOM,
    ePhysicalDeviceCopyMemoryIndirectFeaturesNV                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COPY_MEMORY_INDIRECT_FEATURES_NV,
    ePhysicalDeviceCopyMemoryIndirectPropertiesNV                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COPY_MEMORY_INDIRECT_PROPERTIES_NV,
    ePhysicalDeviceMemoryDecompressionFeaturesNV                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_DECOMPRESSION_FEATURES_NV,
    ePhysicalDeviceMemoryDecompressionPropertiesNV               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_DECOMPRESSION_PROPERTIES_NV,
    ePhysicalDeviceDeviceGeneratedCommandsComputeFeaturesNV      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_COMPUTE_FEATURES_NV,
    eComputePipelineIndirectBufferInfoNV                         = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_INDIRECT_BUFFER_INFO_NV,
    ePipelineIndirectDeviceAddressInfoNV                         = VK_STRUCTURE_TYPE_PIPELINE_INDIRECT_DEVICE_ADDRESS_INFO_NV,
    ePhysicalDeviceLinearColorAttachmentFeaturesNV               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINEAR_COLOR_ATTACHMENT_FEATURES_NV,
    ePhysicalDeviceImageCompressionControlSwapchainFeaturesEXT   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_SWAPCHAIN_FEATURES_EXT,
    ePhysicalDeviceImageProcessingFeaturesQCOM                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_PROCESSING_FEATURES_QCOM,
    ePhysicalDeviceImageProcessingPropertiesQCOM                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_PROCESSING_PROPERTIES_QCOM,
    eImageViewSampleWeightCreateInfoQCOM                         = VK_STRUCTURE_TYPE_IMAGE_VIEW_SAMPLE_WEIGHT_CREATE_INFO_QCOM,
    ePhysicalDeviceNestedCommandBufferFeaturesEXT                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_NESTED_COMMAND_BUFFER_FEATURES_EXT,
    ePhysicalDeviceNestedCommandBufferPropertiesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_NESTED_COMMAND_BUFFER_PROPERTIES_EXT,
    eExternalMemoryAcquireUnmodifiedEXT                          = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_ACQUIRE_UNMODIFIED_EXT,
    ePhysicalDeviceExtendedDynamicState3FeaturesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT,
    ePhysicalDeviceExtendedDynamicState3PropertiesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_PROPERTIES_EXT,
    ePhysicalDeviceSubpassMergeFeedbackFeaturesEXT               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBPASS_MERGE_FEEDBACK_FEATURES_EXT,
    eRenderPassCreationControlEXT                                = VK_STRUCTURE_TYPE_RENDER_PASS_CREATION_CONTROL_EXT,
    eRenderPassCreationFeedbackCreateInfoEXT                     = VK_STRUCTURE_TYPE_RENDER_PASS_CREATION_FEEDBACK_CREATE_INFO_EXT,
    eRenderPassSubpassFeedbackCreateInfoEXT                      = VK_STRUCTURE_TYPE_RENDER_PASS_SUBPASS_FEEDBACK_CREATE_INFO_EXT,
    eDirectDriverLoadingInfoLUNARG                               = VK_STRUCTURE_TYPE_DIRECT_DRIVER_LOADING_INFO_LUNARG,
    eDirectDriverLoadingListLUNARG                               = VK_STRUCTURE_TYPE_DIRECT_DRIVER_LOADING_LIST_LUNARG,
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
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
    ePhysicalDeviceExternalFormatResolveFeaturesANDROID   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_FORMAT_RESOLVE_FEATURES_ANDROID,
    ePhysicalDeviceExternalFormatResolvePropertiesANDROID = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_FORMAT_RESOLVE_PROPERTIES_ANDROID,
    eAndroidHardwareBufferFormatResolvePropertiesANDROID  = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_RESOLVE_PROPERTIES_ANDROID,
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
    ePhysicalDeviceMaintenance5FeaturesKHR                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR,
    ePhysicalDeviceMaintenance5PropertiesKHR                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_PROPERTIES_KHR,
    eRenderingAreaInfoKHR                                        = VK_STRUCTURE_TYPE_RENDERING_AREA_INFO_KHR,
    eDeviceImageSubresourceInfoKHR                               = VK_STRUCTURE_TYPE_DEVICE_IMAGE_SUBRESOURCE_INFO_KHR,
    eSubresourceLayout2KHR                                       = VK_STRUCTURE_TYPE_SUBRESOURCE_LAYOUT_2_KHR,
    eImageSubresource2KHR                                        = VK_STRUCTURE_TYPE_IMAGE_SUBRESOURCE_2_KHR,
    ePipelineCreateFlags2CreateInfoKHR                           = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR,
    eBufferUsageFlags2CreateInfoKHR                              = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR,
    ePhysicalDeviceRayTracingPositionFetchFeaturesKHR            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR,
    ePhysicalDeviceShaderObjectFeaturesEXT                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT,
    ePhysicalDeviceShaderObjectPropertiesEXT                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_PROPERTIES_EXT,
    eShaderCreateInfoEXT                                         = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
    eShaderRequiredSubgroupSizeCreateInfoEXT                     = VK_STRUCTURE_TYPE_SHADER_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
    ePhysicalDeviceTilePropertiesFeaturesQCOM                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TILE_PROPERTIES_FEATURES_QCOM,
    eTilePropertiesQCOM                                          = VK_STRUCTURE_TYPE_TILE_PROPERTIES_QCOM,
    ePhysicalDeviceAmigoProfilingFeaturesSEC                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_AMIGO_PROFILING_FEATURES_SEC,
    eAmigoProfilingSubmitInfoSEC                                 = VK_STRUCTURE_TYPE_AMIGO_PROFILING_SUBMIT_INFO_SEC,
    ePhysicalDeviceMultiviewPerViewViewportsFeaturesQCOM         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PER_VIEW_VIEWPORTS_FEATURES_QCOM,
    ePhysicalDeviceRayTracingInvocationReorderFeaturesNV         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV,
    ePhysicalDeviceRayTracingInvocationReorderPropertiesNV       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_NV,
    ePhysicalDeviceExtendedSparseAddressSpaceFeaturesNV          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_SPARSE_ADDRESS_SPACE_FEATURES_NV,
    ePhysicalDeviceExtendedSparseAddressSpacePropertiesNV        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_SPARSE_ADDRESS_SPACE_PROPERTIES_NV,
    ePhysicalDeviceMutableDescriptorTypeFeaturesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT,
    eMutableDescriptorTypeCreateInfoEXT                          = VK_STRUCTURE_TYPE_MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_EXT,
    ePhysicalDeviceShaderCoreBuiltinsFeaturesARM                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_BUILTINS_FEATURES_ARM,
    ePhysicalDeviceShaderCoreBuiltinsPropertiesARM               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_BUILTINS_PROPERTIES_ARM,
    ePhysicalDevicePipelineLibraryGroupHandlesFeaturesEXT        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_LIBRARY_GROUP_HANDLES_FEATURES_EXT,
    ePhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_FEATURES_EXT,
    eLatencySleepModeInfoNV                                      = VK_STRUCTURE_TYPE_LATENCY_SLEEP_MODE_INFO_NV,
    eLatencySleepInfoNV                                          = VK_STRUCTURE_TYPE_LATENCY_SLEEP_INFO_NV,
    eSetLatencyMarkerInfoNV                                      = VK_STRUCTURE_TYPE_SET_LATENCY_MARKER_INFO_NV,
    eGetLatencyMarkerInfoNV                                      = VK_STRUCTURE_TYPE_GET_LATENCY_MARKER_INFO_NV,
    eLatencyTimingsFrameReportNV                                 = VK_STRUCTURE_TYPE_LATENCY_TIMINGS_FRAME_REPORT_NV,
    eLatencySubmissionPresentIdNV                                = VK_STRUCTURE_TYPE_LATENCY_SUBMISSION_PRESENT_ID_NV,
    eOutOfBandQueueTypeInfoNV                                    = VK_STRUCTURE_TYPE_OUT_OF_BAND_QUEUE_TYPE_INFO_NV,
    eSwapchainLatencyCreateInfoNV                                = VK_STRUCTURE_TYPE_SWAPCHAIN_LATENCY_CREATE_INFO_NV,
    eLatencySurfaceCapabilitiesNV                                = VK_STRUCTURE_TYPE_LATENCY_SURFACE_CAPABILITIES_NV,
    ePhysicalDeviceCooperativeMatrixFeaturesKHR                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
    eCooperativeMatrixPropertiesKHR                              = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR,
    ePhysicalDeviceCooperativeMatrixPropertiesKHR                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR,
    ePhysicalDeviceMultiviewPerViewRenderAreasFeaturesQCOM       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PER_VIEW_RENDER_AREAS_FEATURES_QCOM,
    eMultiviewPerViewRenderAreasRenderPassBeginInfoQCOM          = VK_STRUCTURE_TYPE_MULTIVIEW_PER_VIEW_RENDER_AREAS_RENDER_PASS_BEGIN_INFO_QCOM,
    ePhysicalDeviceImageProcessing2FeaturesQCOM                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_PROCESSING_2_FEATURES_QCOM,
    ePhysicalDeviceImageProcessing2PropertiesQCOM                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_PROCESSING_2_PROPERTIES_QCOM,
    eSamplerBlockMatchWindowCreateInfoQCOM                       = VK_STRUCTURE_TYPE_SAMPLER_BLOCK_MATCH_WINDOW_CREATE_INFO_QCOM,
    eSamplerCubicWeightsCreateInfoQCOM                           = VK_STRUCTURE_TYPE_SAMPLER_CUBIC_WEIGHTS_CREATE_INFO_QCOM,
    ePhysicalDeviceCubicWeightsFeaturesQCOM                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUBIC_WEIGHTS_FEATURES_QCOM,
    eBlitImageCubicWeightsInfoQCOM                               = VK_STRUCTURE_TYPE_BLIT_IMAGE_CUBIC_WEIGHTS_INFO_QCOM,
    ePhysicalDeviceYcbcrDegammaFeaturesQCOM                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_YCBCR_DEGAMMA_FEATURES_QCOM,
    eSamplerYcbcrConversionYcbcrDegammaCreateInfoQCOM            = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_YCBCR_DEGAMMA_CREATE_INFO_QCOM,
    ePhysicalDeviceCubicClampFeaturesQCOM                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUBIC_CLAMP_FEATURES_QCOM,
    ePhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_FEATURES_EXT,
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
    eScreenBufferPropertiesQNX                           = VK_STRUCTURE_TYPE_SCREEN_BUFFER_PROPERTIES_QNX,
    eScreenBufferFormatPropertiesQNX                     = VK_STRUCTURE_TYPE_SCREEN_BUFFER_FORMAT_PROPERTIES_QNX,
    eImportScreenBufferInfoQNX                           = VK_STRUCTURE_TYPE_IMPORT_SCREEN_BUFFER_INFO_QNX,
    eExternalFormatQNX                                   = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_QNX,
    ePhysicalDeviceExternalMemoryScreenBufferFeaturesQNX = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_SCREEN_BUFFER_FEATURES_QNX,
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/
    ePhysicalDeviceLayeredDriverPropertiesMSFT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LAYERED_DRIVER_PROPERTIES_MSFT,
    ePhysicalDeviceDescriptorPoolOverallocationFeaturesNV = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_POOL_OVERALLOCATION_FEATURES_NV
  };

  enum class PipelineCacheHeaderVersion
  {
    eOne = VK_PIPELINE_CACHE_HEADER_VERSION_ONE
  };

  enum class ObjectType
  {
    eUnknown                       = VK_OBJECT_TYPE_UNKNOWN,
    eInstance                      = VK_OBJECT_TYPE_INSTANCE,
    ePhysicalDevice                = VK_OBJECT_TYPE_PHYSICAL_DEVICE,
    eDevice                        = VK_OBJECT_TYPE_DEVICE,
    eQueue                         = VK_OBJECT_TYPE_QUEUE,
    eSemaphore                     = VK_OBJECT_TYPE_SEMAPHORE,
    eCommandBuffer                 = VK_OBJECT_TYPE_COMMAND_BUFFER,
    eFence                         = VK_OBJECT_TYPE_FENCE,
    eDeviceMemory                  = VK_OBJECT_TYPE_DEVICE_MEMORY,
    eBuffer                        = VK_OBJECT_TYPE_BUFFER,
    eImage                         = VK_OBJECT_TYPE_IMAGE,
    eEvent                         = VK_OBJECT_TYPE_EVENT,
    eQueryPool                     = VK_OBJECT_TYPE_QUERY_POOL,
    eBufferView                    = VK_OBJECT_TYPE_BUFFER_VIEW,
    eImageView                     = VK_OBJECT_TYPE_IMAGE_VIEW,
    eShaderModule                  = VK_OBJECT_TYPE_SHADER_MODULE,
    ePipelineCache                 = VK_OBJECT_TYPE_PIPELINE_CACHE,
    ePipelineLayout                = VK_OBJECT_TYPE_PIPELINE_LAYOUT,
    eRenderPass                    = VK_OBJECT_TYPE_RENDER_PASS,
    ePipeline                      = VK_OBJECT_TYPE_PIPELINE,
    eDescriptorSetLayout           = VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
    eSampler                       = VK_OBJECT_TYPE_SAMPLER,
    eDescriptorPool                = VK_OBJECT_TYPE_DESCRIPTOR_POOL,
    eDescriptorSet                 = VK_OBJECT_TYPE_DESCRIPTOR_SET,
    eFramebuffer                   = VK_OBJECT_TYPE_FRAMEBUFFER,
    eCommandPool                   = VK_OBJECT_TYPE_COMMAND_POOL,
    eSamplerYcbcrConversion        = VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION,
    eDescriptorUpdateTemplate      = VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE,
    ePrivateDataSlot               = VK_OBJECT_TYPE_PRIVATE_DATA_SLOT,
    eSurfaceKHR                    = VK_OBJECT_TYPE_SURFACE_KHR,
    eSwapchainKHR                  = VK_OBJECT_TYPE_SWAPCHAIN_KHR,
    eDisplayKHR                    = VK_OBJECT_TYPE_DISPLAY_KHR,
    eDisplayModeKHR                = VK_OBJECT_TYPE_DISPLAY_MODE_KHR,
    eDebugReportCallbackEXT        = VK_OBJECT_TYPE_DEBUG_REPORT_CALLBACK_EXT,
    eVideoSessionKHR               = VK_OBJECT_TYPE_VIDEO_SESSION_KHR,
    eVideoSessionParametersKHR     = VK_OBJECT_TYPE_VIDEO_SESSION_PARAMETERS_KHR,
    eCuModuleNVX                   = VK_OBJECT_TYPE_CU_MODULE_NVX,
    eCuFunctionNVX                 = VK_OBJECT_TYPE_CU_FUNCTION_NVX,
    eDescriptorUpdateTemplateKHR   = VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_KHR,
    eDebugUtilsMessengerEXT        = VK_OBJECT_TYPE_DEBUG_UTILS_MESSENGER_EXT,
    eAccelerationStructureKHR      = VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR,
    eSamplerYcbcrConversionKHR     = VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION_KHR,
    eValidationCacheEXT            = VK_OBJECT_TYPE_VALIDATION_CACHE_EXT,
    eAccelerationStructureNV       = VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV,
    ePerformanceConfigurationINTEL = VK_OBJECT_TYPE_PERFORMANCE_CONFIGURATION_INTEL,
    eDeferredOperationKHR          = VK_OBJECT_TYPE_DEFERRED_OPERATION_KHR,
    eIndirectCommandsLayoutNV      = VK_OBJECT_TYPE_INDIRECT_COMMANDS_LAYOUT_NV,
    ePrivateDataSlotEXT            = VK_OBJECT_TYPE_PRIVATE_DATA_SLOT_EXT,
#if defined( VK_USE_PLATFORM_FUCHSIA )
    eBufferCollectionFUCHSIA = VK_OBJECT_TYPE_BUFFER_COLLECTION_FUCHSIA,
#endif /*VK_USE_PLATFORM_FUCHSIA*/
    eMicromapEXT          = VK_OBJECT_TYPE_MICROMAP_EXT,
    eOpticalFlowSessionNV = VK_OBJECT_TYPE_OPTICAL_FLOW_SESSION_NV,
    eShaderEXT            = VK_OBJECT_TYPE_SHADER_EXT
  };

  enum class VendorId
  {
    eVIV      = VK_VENDOR_ID_VIV,
    eVSI      = VK_VENDOR_ID_VSI,
    eKazan    = VK_VENDOR_ID_KAZAN,
    eCodeplay = VK_VENDOR_ID_CODEPLAY,
    eMESA     = VK_VENDOR_ID_MESA,
    ePocl     = VK_VENDOR_ID_POCL,
    eMobileye = VK_VENDOR_ID_MOBILEYE
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
    eG8B8G8R8422UnormKHR                     = VK_FORMAT_G8B8G8R8_422_UNORM_KHR,
    eB8G8R8G8422UnormKHR                     = VK_FORMAT_B8G8R8G8_422_UNORM_KHR,
    eG8B8R83Plane420UnormKHR                 = VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM_KHR,
    eG8B8R82Plane420UnormKHR                 = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM_KHR,
    eG8B8R83Plane422UnormKHR                 = VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM_KHR,
    eG8B8R82Plane422UnormKHR                 = VK_FORMAT_G8_B8R8_2PLANE_422_UNORM_KHR,
    eG8B8R83Plane444UnormKHR                 = VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM_KHR,
    eR10X6UnormPack16KHR                     = VK_FORMAT_R10X6_UNORM_PACK16_KHR,
    eR10X6G10X6Unorm2Pack16KHR               = VK_FORMAT_R10X6G10X6_UNORM_2PACK16_KHR,
    eR10X6G10X6B10X6A10X6Unorm4Pack16KHR     = VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16_KHR,
    eG10X6B10X6G10X6R10X6422Unorm4Pack16KHR  = VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16_KHR,
    eB10X6G10X6R10X6G10X6422Unorm4Pack16KHR  = VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16_KHR,
    eG10X6B10X6R10X63Plane420Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16_KHR,
    eG10X6B10X6R10X62Plane420Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16_KHR,
    eG10X6B10X6R10X63Plane422Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16_KHR,
    eG10X6B10X6R10X62Plane422Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16_KHR,
    eG10X6B10X6R10X63Plane444Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16_KHR,
    eR12X4UnormPack16KHR                     = VK_FORMAT_R12X4_UNORM_PACK16_KHR,
    eR12X4G12X4Unorm2Pack16KHR               = VK_FORMAT_R12X4G12X4_UNORM_2PACK16_KHR,
    eR12X4G12X4B12X4A12X4Unorm4Pack16KHR     = VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16_KHR,
    eG12X4B12X4G12X4R12X4422Unorm4Pack16KHR  = VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16_KHR,
    eB12X4G12X4R12X4G12X4422Unorm4Pack16KHR  = VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16_KHR,
    eG12X4B12X4R12X43Plane420Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16_KHR,
    eG12X4B12X4R12X42Plane420Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16_KHR,
    eG12X4B12X4R12X43Plane422Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16_KHR,
    eG12X4B12X4R12X42Plane422Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16_KHR,
    eG12X4B12X4R12X43Plane444Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16_KHR,
    eG16B16G16R16422UnormKHR                 = VK_FORMAT_G16B16G16R16_422_UNORM_KHR,
    eB16G16R16G16422UnormKHR                 = VK_FORMAT_B16G16R16G16_422_UNORM_KHR,
    eG16B16R163Plane420UnormKHR              = VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM_KHR,
    eG16B16R162Plane420UnormKHR              = VK_FORMAT_G16_B16R16_2PLANE_420_UNORM_KHR,
    eG16B16R163Plane422UnormKHR              = VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM_KHR,
    eG16B16R162Plane422UnormKHR              = VK_FORMAT_G16_B16R16_2PLANE_422_UNORM_KHR,
    eG16B16R163Plane444UnormKHR              = VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM_KHR,
    eG8B8R82Plane444UnormEXT                 = VK_FORMAT_G8_B8R8_2PLANE_444_UNORM_EXT,
    eG10X6B10X6R10X62Plane444Unorm3Pack16EXT = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16_EXT,
    eG12X4B12X4R12X42Plane444Unorm3Pack16EXT = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16_EXT,
    eG16B16R162Plane444UnormEXT              = VK_FORMAT_G16_B16R16_2PLANE_444_UNORM_EXT,
    eA4R4G4B4UnormPack16EXT                  = VK_FORMAT_A4R4G4B4_UNORM_PACK16_EXT,
    eA4B4G4R4UnormPack16EXT                  = VK_FORMAT_A4B4G4R4_UNORM_PACK16_EXT,
    eR16G16S105NV                            = VK_FORMAT_R16G16_S10_5_NV,
    eA1B5G5R5UnormPack16KHR                  = VK_FORMAT_A1B5G5R5_UNORM_PACK16_KHR,
    eA8UnormKHR                              = VK_FORMAT_A8_UNORM_KHR
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
    eDisjoint                                                   = VK_FORMAT_FEATURE_DISJOINT_BIT,
    eCositedChromaSamples                                       = VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT,
    eSampledImageFilterMinmax                                   = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_MINMAX_BIT,
    eSampledImageFilterCubicIMG                                 = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_CUBIC_BIT_IMG,
    eVideoDecodeOutputKHR                                       = VK_FORMAT_FEATURE_VIDEO_DECODE_OUTPUT_BIT_KHR,
    eVideoDecodeDpbKHR                                          = VK_FORMAT_FEATURE_VIDEO_DECODE_DPB_BIT_KHR,
    eTransferSrcKHR                                             = VK_FORMAT_FEATURE_TRANSFER_SRC_BIT_KHR,
    eTransferDstKHR                                             = VK_FORMAT_FEATURE_TRANSFER_DST_BIT_KHR,
    eSampledImageFilterMinmaxEXT                                = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_MINMAX_BIT_EXT,
    eAccelerationStructureVertexBufferKHR                       = VK_FORMAT_FEATURE_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR,
    eMidpointChromaSamplesKHR                                   = VK_FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT_KHR,
    eSampledImageYcbcrConversionLinearFilterKHR                 = VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT_KHR,
    eSampledImageYcbcrConversionSeparateReconstructionFilterKHR = VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT_KHR,
    eSampledImageYcbcrConversionChromaReconstructionExplicitKHR = VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_BIT_KHR,
    eSampledImageYcbcrConversionChromaReconstructionExplicitForceableKHR =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE_BIT_KHR,
    eDisjointKHR                      = VK_FORMAT_FEATURE_DISJOINT_BIT_KHR,
    eCositedChromaSamplesKHR          = VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT_KHR,
    eSampledImageFilterCubicEXT       = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_CUBIC_BIT_EXT,
    eFragmentDensityMapEXT            = VK_FORMAT_FEATURE_FRAGMENT_DENSITY_MAP_BIT_EXT,
    eFragmentShadingRateAttachmentKHR = VK_FORMAT_FEATURE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoEncodeInputKHR = VK_FORMAT_FEATURE_VIDEO_ENCODE_INPUT_BIT_KHR,
    eVideoEncodeDpbKHR   = VK_FORMAT_FEATURE_VIDEO_ENCODE_DPB_BIT_KHR
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
  };

  using FormatFeatureFlags = Flags<FormatFeatureFlagBits>;

  template <>
  struct FlagTraits<FormatFeatureFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool               isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR FormatFeatureFlags allFlags =
      FormatFeatureFlagBits::eSampledImage | FormatFeatureFlagBits::eStorageImage | FormatFeatureFlagBits::eStorageImageAtomic |
      FormatFeatureFlagBits::eUniformTexelBuffer | FormatFeatureFlagBits::eStorageTexelBuffer | FormatFeatureFlagBits::eStorageTexelBufferAtomic |
      FormatFeatureFlagBits::eVertexBuffer | FormatFeatureFlagBits::eColorAttachment | FormatFeatureFlagBits::eColorAttachmentBlend |
      FormatFeatureFlagBits::eDepthStencilAttachment | FormatFeatureFlagBits::eBlitSrc | FormatFeatureFlagBits::eBlitDst |
      FormatFeatureFlagBits::eSampledImageFilterLinear | FormatFeatureFlagBits::eTransferSrc | FormatFeatureFlagBits::eTransferDst |
      FormatFeatureFlagBits::eMidpointChromaSamples | FormatFeatureFlagBits::eSampledImageYcbcrConversionLinearFilter |
      FormatFeatureFlagBits::eSampledImageYcbcrConversionSeparateReconstructionFilter |
      FormatFeatureFlagBits::eSampledImageYcbcrConversionChromaReconstructionExplicit |
      FormatFeatureFlagBits::eSampledImageYcbcrConversionChromaReconstructionExplicitForceable | FormatFeatureFlagBits::eDisjoint |
      FormatFeatureFlagBits::eCositedChromaSamples | FormatFeatureFlagBits::eSampledImageFilterMinmax | FormatFeatureFlagBits::eVideoDecodeOutputKHR |
      FormatFeatureFlagBits::eVideoDecodeDpbKHR | FormatFeatureFlagBits::eAccelerationStructureVertexBufferKHR |
      FormatFeatureFlagBits::eSampledImageFilterCubicEXT | FormatFeatureFlagBits::eFragmentDensityMapEXT |
      FormatFeatureFlagBits::eFragmentShadingRateAttachmentKHR
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      | FormatFeatureFlagBits::eVideoEncodeInputKHR | FormatFeatureFlagBits::eVideoEncodeDpbKHR
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      ;
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
    eSplitInstanceBindRegionsKHR          = VK_IMAGE_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT_KHR,
    e2DArrayCompatibleKHR                 = VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT_KHR,
    eBlockTexelViewCompatibleKHR          = VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT_KHR,
    eExtendedUsageKHR                     = VK_IMAGE_CREATE_EXTENDED_USAGE_BIT_KHR,
    eSampleLocationsCompatibleDepthEXT    = VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT,
    eDisjointKHR                          = VK_IMAGE_CREATE_DISJOINT_BIT_KHR,
    eAliasKHR                             = VK_IMAGE_CREATE_ALIAS_BIT_KHR,
    eSubsampledEXT                        = VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT,
    eDescriptorBufferCaptureReplayEXT     = VK_IMAGE_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_EXT,
    eMultisampledRenderToSingleSampledEXT = VK_IMAGE_CREATE_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_BIT_EXT,
    e2DViewCompatibleEXT                  = VK_IMAGE_CREATE_2D_VIEW_COMPATIBLE_BIT_EXT,
    eFragmentDensityMapOffsetQCOM         = VK_IMAGE_CREATE_FRAGMENT_DENSITY_MAP_OFFSET_BIT_QCOM
  };

  using ImageCreateFlags = Flags<ImageCreateFlagBits>;

  template <>
  struct FlagTraits<ImageCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ImageCreateFlags allFlags =
      ImageCreateFlagBits::eSparseBinding | ImageCreateFlagBits::eSparseResidency | ImageCreateFlagBits::eSparseAliased | ImageCreateFlagBits::eMutableFormat |
      ImageCreateFlagBits::eCubeCompatible | ImageCreateFlagBits::eAlias | ImageCreateFlagBits::eSplitInstanceBindRegions |
      ImageCreateFlagBits::e2DArrayCompatible | ImageCreateFlagBits::eBlockTexelViewCompatible | ImageCreateFlagBits::eExtendedUsage |
      ImageCreateFlagBits::eProtected | ImageCreateFlagBits::eDisjoint | ImageCreateFlagBits::eCornerSampledNV |
      ImageCreateFlagBits::eSampleLocationsCompatibleDepthEXT | ImageCreateFlagBits::eSubsampledEXT | ImageCreateFlagBits::eDescriptorBufferCaptureReplayEXT |
      ImageCreateFlagBits::eMultisampledRenderToSingleSampledEXT | ImageCreateFlagBits::e2DViewCompatibleEXT |
      ImageCreateFlagBits::eFragmentDensityMapOffsetQCOM;
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
    eTransferSrc                      = VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
    eTransferDst                      = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
    eSampled                          = VK_IMAGE_USAGE_SAMPLED_BIT,
    eStorage                          = VK_IMAGE_USAGE_STORAGE_BIT,
    eColorAttachment                  = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    eDepthStencilAttachment           = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
    eTransientAttachment              = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
    eInputAttachment                  = VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
    eVideoDecodeDstKHR                = VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR,
    eVideoDecodeSrcKHR                = VK_IMAGE_USAGE_VIDEO_DECODE_SRC_BIT_KHR,
    eVideoDecodeDpbKHR                = VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR,
    eShadingRateImageNV               = VK_IMAGE_USAGE_SHADING_RATE_IMAGE_BIT_NV,
    eFragmentDensityMapEXT            = VK_IMAGE_USAGE_FRAGMENT_DENSITY_MAP_BIT_EXT,
    eFragmentShadingRateAttachmentKHR = VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
    eHostTransferEXT                  = VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoEncodeDstKHR = VK_IMAGE_USAGE_VIDEO_ENCODE_DST_BIT_KHR,
    eVideoEncodeSrcKHR = VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR,
    eVideoEncodeDpbKHR = VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eAttachmentFeedbackLoopEXT = VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT,
    eInvocationMaskHUAWEI      = VK_IMAGE_USAGE_INVOCATION_MASK_BIT_HUAWEI,
    eSampleWeightQCOM          = VK_IMAGE_USAGE_SAMPLE_WEIGHT_BIT_QCOM,
    eSampleBlockMatchQCOM      = VK_IMAGE_USAGE_SAMPLE_BLOCK_MATCH_BIT_QCOM
  };

  using ImageUsageFlags = Flags<ImageUsageFlagBits>;

  template <>
  struct FlagTraits<ImageUsageFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool            isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ImageUsageFlags allFlags =
      ImageUsageFlagBits::eTransferSrc | ImageUsageFlagBits::eTransferDst | ImageUsageFlagBits::eSampled | ImageUsageFlagBits::eStorage |
      ImageUsageFlagBits::eColorAttachment | ImageUsageFlagBits::eDepthStencilAttachment | ImageUsageFlagBits::eTransientAttachment |
      ImageUsageFlagBits::eInputAttachment | ImageUsageFlagBits::eVideoDecodeDstKHR | ImageUsageFlagBits::eVideoDecodeSrcKHR |
      ImageUsageFlagBits::eVideoDecodeDpbKHR | ImageUsageFlagBits::eFragmentDensityMapEXT | ImageUsageFlagBits::eFragmentShadingRateAttachmentKHR |
      ImageUsageFlagBits::eHostTransferEXT
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      | ImageUsageFlagBits::eVideoEncodeDstKHR | ImageUsageFlagBits::eVideoEncodeSrcKHR | ImageUsageFlagBits::eVideoEncodeDpbKHR
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      | ImageUsageFlagBits::eAttachmentFeedbackLoopEXT | ImageUsageFlagBits::eInvocationMaskHUAWEI | ImageUsageFlagBits::eSampleWeightQCOM |
      ImageUsageFlagBits::eSampleBlockMatchQCOM;
  };

  enum class InstanceCreateFlagBits : VkInstanceCreateFlags
  {
    eEnumeratePortabilityKHR = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
  };

  using InstanceCreateFlags = Flags<InstanceCreateFlagBits>;

  template <>
  struct FlagTraits<InstanceCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR InstanceCreateFlags allFlags  = InstanceCreateFlagBits::eEnumeratePortabilityKHR;
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

  using MemoryHeapFlags = Flags<MemoryHeapFlagBits>;

  template <>
  struct FlagTraits<MemoryHeapFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool            isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR MemoryHeapFlags allFlags  = MemoryHeapFlagBits::eDeviceLocal | MemoryHeapFlagBits::eMultiInstance;
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

  using MemoryPropertyFlags = Flags<MemoryPropertyFlagBits>;

  template <>
  struct FlagTraits<MemoryPropertyFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR MemoryPropertyFlags allFlags =
      MemoryPropertyFlagBits::eDeviceLocal | MemoryPropertyFlagBits::eHostVisible | MemoryPropertyFlagBits::eHostCoherent |
      MemoryPropertyFlagBits::eHostCached | MemoryPropertyFlagBits::eLazilyAllocated | MemoryPropertyFlagBits::eProtected |
      MemoryPropertyFlagBits::eDeviceCoherentAMD | MemoryPropertyFlagBits::eDeviceUncachedAMD | MemoryPropertyFlagBits::eRdmaCapableNV;
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
    eGraphics       = VK_QUEUE_GRAPHICS_BIT,
    eCompute        = VK_QUEUE_COMPUTE_BIT,
    eTransfer       = VK_QUEUE_TRANSFER_BIT,
    eSparseBinding  = VK_QUEUE_SPARSE_BINDING_BIT,
    eProtected      = VK_QUEUE_PROTECTED_BIT,
    eVideoDecodeKHR = VK_QUEUE_VIDEO_DECODE_BIT_KHR,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoEncodeKHR = VK_QUEUE_VIDEO_ENCODE_BIT_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eOpticalFlowNV = VK_QUEUE_OPTICAL_FLOW_BIT_NV
  };

  using QueueFlags = Flags<QueueFlagBits>;

  template <>
  struct FlagTraits<QueueFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR QueueFlags allFlags  = QueueFlagBits::eGraphics | QueueFlagBits::eCompute | QueueFlagBits::eTransfer |
                                                               QueueFlagBits::eSparseBinding | QueueFlagBits::eProtected | QueueFlagBits::eVideoDecodeKHR
#if defined( VK_ENABLE_BETA_EXTENSIONS )
                                                               | QueueFlagBits::eVideoEncodeKHR
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
                                                               | QueueFlagBits::eOpticalFlowNV;
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

  using SampleCountFlags = Flags<SampleCountFlagBits>;

  template <>
  struct FlagTraits<SampleCountFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SampleCountFlags allFlags  = SampleCountFlagBits::e1 | SampleCountFlagBits::e2 | SampleCountFlagBits::e4 |
                                                                     SampleCountFlagBits::e8 | SampleCountFlagBits::e16 | SampleCountFlagBits::e32 |
                                                                     SampleCountFlagBits::e64;
  };

  enum class SystemAllocationScope
  {
    eCommand  = VK_SYSTEM_ALLOCATION_SCOPE_COMMAND,
    eObject   = VK_SYSTEM_ALLOCATION_SCOPE_OBJECT,
    eCache    = VK_SYSTEM_ALLOCATION_SCOPE_CACHE,
    eDevice   = VK_SYSTEM_ALLOCATION_SCOPE_DEVICE,
    eInstance = VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE
  };

  enum class DeviceCreateFlagBits : VkDeviceCreateFlags
  {
  };

  using DeviceCreateFlags = Flags<DeviceCreateFlagBits>;

  template <>
  struct FlagTraits<DeviceCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool              isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DeviceCreateFlags allFlags  = {};
  };

  enum class DeviceQueueCreateFlagBits : VkDeviceQueueCreateFlags
  {
    eProtected = VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT
  };

  using DeviceQueueCreateFlags = Flags<DeviceQueueCreateFlagBits>;

  template <>
  struct FlagTraits<DeviceQueueCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DeviceQueueCreateFlags allFlags  = DeviceQueueCreateFlagBits::eProtected;
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
    eShadingRateImageNV               = VK_PIPELINE_STAGE_SHADING_RATE_IMAGE_BIT_NV,
    eRayTracingShaderNV               = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_NV,
    eAccelerationStructureBuildNV     = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
    eTaskShaderNV                     = VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV,
    eMeshShaderNV                     = VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV,
    eFragmentDensityProcessEXT        = VK_PIPELINE_STAGE_FRAGMENT_DENSITY_PROCESS_BIT_EXT,
    eFragmentShadingRateAttachmentKHR = VK_PIPELINE_STAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
    eCommandPreprocessNV              = VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_NV,
    eNoneKHR                          = VK_PIPELINE_STAGE_NONE_KHR,
    eTaskShaderEXT                    = VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT,
    eMeshShaderEXT                    = VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT
  };

  using PipelineStageFlags = Flags<PipelineStageFlagBits>;

  template <>
  struct FlagTraits<PipelineStageFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool               isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineStageFlags allFlags =
      PipelineStageFlagBits::eTopOfPipe | PipelineStageFlagBits::eDrawIndirect | PipelineStageFlagBits::eVertexInput | PipelineStageFlagBits::eVertexShader |
      PipelineStageFlagBits::eTessellationControlShader | PipelineStageFlagBits::eTessellationEvaluationShader | PipelineStageFlagBits::eGeometryShader |
      PipelineStageFlagBits::eFragmentShader | PipelineStageFlagBits::eEarlyFragmentTests | PipelineStageFlagBits::eLateFragmentTests |
      PipelineStageFlagBits::eColorAttachmentOutput | PipelineStageFlagBits::eComputeShader | PipelineStageFlagBits::eTransfer |
      PipelineStageFlagBits::eBottomOfPipe | PipelineStageFlagBits::eHost | PipelineStageFlagBits::eAllGraphics | PipelineStageFlagBits::eAllCommands |
      PipelineStageFlagBits::eNone | PipelineStageFlagBits::eTransformFeedbackEXT | PipelineStageFlagBits::eConditionalRenderingEXT |
      PipelineStageFlagBits::eAccelerationStructureBuildKHR | PipelineStageFlagBits::eRayTracingShaderKHR | PipelineStageFlagBits::eFragmentDensityProcessEXT |
      PipelineStageFlagBits::eFragmentShadingRateAttachmentKHR | PipelineStageFlagBits::eCommandPreprocessNV | PipelineStageFlagBits::eTaskShaderEXT |
      PipelineStageFlagBits::eMeshShaderEXT;
  };

  enum class MemoryMapFlagBits : VkMemoryMapFlags
  {
  };

  using MemoryMapFlags = Flags<MemoryMapFlagBits>;

  template <>
  struct FlagTraits<MemoryMapFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR MemoryMapFlags allFlags  = {};
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
    ePlane0KHR       = VK_IMAGE_ASPECT_PLANE_0_BIT_KHR,
    ePlane1KHR       = VK_IMAGE_ASPECT_PLANE_1_BIT_KHR,
    ePlane2KHR       = VK_IMAGE_ASPECT_PLANE_2_BIT_KHR,
    eMemoryPlane0EXT = VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT,
    eMemoryPlane1EXT = VK_IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT,
    eMemoryPlane2EXT = VK_IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT,
    eMemoryPlane3EXT = VK_IMAGE_ASPECT_MEMORY_PLANE_3_BIT_EXT,
    eNoneKHR         = VK_IMAGE_ASPECT_NONE_KHR
  };

  using ImageAspectFlags = Flags<ImageAspectFlagBits>;

  template <>
  struct FlagTraits<ImageAspectFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ImageAspectFlags allFlags = ImageAspectFlagBits::eColor | ImageAspectFlagBits::eDepth | ImageAspectFlagBits::eStencil |
                                                                     ImageAspectFlagBits::eMetadata | ImageAspectFlagBits::ePlane0 |
                                                                     ImageAspectFlagBits::ePlane1 | ImageAspectFlagBits::ePlane2 | ImageAspectFlagBits::eNone |
                                                                     ImageAspectFlagBits::eMemoryPlane0EXT | ImageAspectFlagBits::eMemoryPlane1EXT |
                                                                     ImageAspectFlagBits::eMemoryPlane2EXT | ImageAspectFlagBits::eMemoryPlane3EXT;
  };

  enum class SparseImageFormatFlagBits : VkSparseImageFormatFlags
  {
    eSingleMiptail        = VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT,
    eAlignedMipSize       = VK_SPARSE_IMAGE_FORMAT_ALIGNED_MIP_SIZE_BIT,
    eNonstandardBlockSize = VK_SPARSE_IMAGE_FORMAT_NONSTANDARD_BLOCK_SIZE_BIT
  };

  using SparseImageFormatFlags = Flags<SparseImageFormatFlagBits>;

  template <>
  struct FlagTraits<SparseImageFormatFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SparseImageFormatFlags allFlags =
      SparseImageFormatFlagBits::eSingleMiptail | SparseImageFormatFlagBits::eAlignedMipSize | SparseImageFormatFlagBits::eNonstandardBlockSize;
  };

  enum class SparseMemoryBindFlagBits : VkSparseMemoryBindFlags
  {
    eMetadata = VK_SPARSE_MEMORY_BIND_METADATA_BIT
  };

  using SparseMemoryBindFlags = Flags<SparseMemoryBindFlagBits>;

  template <>
  struct FlagTraits<SparseMemoryBindFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                  isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SparseMemoryBindFlags allFlags  = SparseMemoryBindFlagBits::eMetadata;
  };

  enum class FenceCreateFlagBits : VkFenceCreateFlags
  {
    eSignaled = VK_FENCE_CREATE_SIGNALED_BIT
  };

  using FenceCreateFlags = Flags<FenceCreateFlagBits>;

  template <>
  struct FlagTraits<FenceCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR FenceCreateFlags allFlags  = FenceCreateFlagBits::eSignaled;
  };

  enum class SemaphoreCreateFlagBits : VkSemaphoreCreateFlags
  {
  };

  using SemaphoreCreateFlags = Flags<SemaphoreCreateFlagBits>;

  template <>
  struct FlagTraits<SemaphoreCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SemaphoreCreateFlags allFlags  = {};
  };

  enum class EventCreateFlagBits : VkEventCreateFlags
  {
    eDeviceOnly    = VK_EVENT_CREATE_DEVICE_ONLY_BIT,
    eDeviceOnlyKHR = VK_EVENT_CREATE_DEVICE_ONLY_BIT_KHR
  };

  using EventCreateFlags = Flags<EventCreateFlagBits>;

  template <>
  struct FlagTraits<EventCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR EventCreateFlags allFlags  = EventCreateFlagBits::eDeviceOnly;
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
    eMeshShaderInvocationsEXT                = VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT,
    eClusterCullingShaderInvocationsHUAWEI   = VK_QUERY_PIPELINE_STATISTIC_CLUSTER_CULLING_SHADER_INVOCATIONS_BIT_HUAWEI
  };

  using QueryPipelineStatisticFlags = Flags<QueryPipelineStatisticFlagBits>;

  template <>
  struct FlagTraits<QueryPipelineStatisticFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                        isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR QueryPipelineStatisticFlags allFlags =
      QueryPipelineStatisticFlagBits::eInputAssemblyVertices | QueryPipelineStatisticFlagBits::eInputAssemblyPrimitives |
      QueryPipelineStatisticFlagBits::eVertexShaderInvocations | QueryPipelineStatisticFlagBits::eGeometryShaderInvocations |
      QueryPipelineStatisticFlagBits::eGeometryShaderPrimitives | QueryPipelineStatisticFlagBits::eClippingInvocations |
      QueryPipelineStatisticFlagBits::eClippingPrimitives | QueryPipelineStatisticFlagBits::eFragmentShaderInvocations |
      QueryPipelineStatisticFlagBits::eTessellationControlShaderPatches | QueryPipelineStatisticFlagBits::eTessellationEvaluationShaderInvocations |
      QueryPipelineStatisticFlagBits::eComputeShaderInvocations | QueryPipelineStatisticFlagBits::eTaskShaderInvocationsEXT |
      QueryPipelineStatisticFlagBits::eMeshShaderInvocationsEXT | QueryPipelineStatisticFlagBits::eClusterCullingShaderInvocationsHUAWEI;
  };

  enum class QueryResultFlagBits : VkQueryResultFlags
  {
    e64               = VK_QUERY_RESULT_64_BIT,
    eWait             = VK_QUERY_RESULT_WAIT_BIT,
    eWithAvailability = VK_QUERY_RESULT_WITH_AVAILABILITY_BIT,
    ePartial          = VK_QUERY_RESULT_PARTIAL_BIT,
    eWithStatusKHR    = VK_QUERY_RESULT_WITH_STATUS_BIT_KHR
  };

  using QueryResultFlags = Flags<QueryResultFlagBits>;

  template <>
  struct FlagTraits<QueryResultFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR QueryResultFlags allFlags  = QueryResultFlagBits::e64 | QueryResultFlagBits::eWait |
                                                                     QueryResultFlagBits::eWithAvailability | QueryResultFlagBits::ePartial |
                                                                     QueryResultFlagBits::eWithStatusKHR;
  };

  enum class QueryType
  {
    eOcclusion                                 = VK_QUERY_TYPE_OCCLUSION,
    ePipelineStatistics                        = VK_QUERY_TYPE_PIPELINE_STATISTICS,
    eTimestamp                                 = VK_QUERY_TYPE_TIMESTAMP,
    eResultStatusOnlyKHR                       = VK_QUERY_TYPE_RESULT_STATUS_ONLY_KHR,
    eTransformFeedbackStreamEXT                = VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT,
    ePerformanceQueryKHR                       = VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR,
    eAccelerationStructureCompactedSizeKHR     = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
    eAccelerationStructureSerializationSizeKHR = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR,
    eAccelerationStructureCompactedSizeNV      = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_NV,
    ePerformanceQueryINTEL                     = VK_QUERY_TYPE_PERFORMANCE_QUERY_INTEL,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoEncodeFeedbackKHR = VK_QUERY_TYPE_VIDEO_ENCODE_FEEDBACK_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eMeshPrimitivesGeneratedEXT                               = VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT,
    ePrimitivesGeneratedEXT                                   = VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT,
    eAccelerationStructureSerializationBottomLevelPointersKHR = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR,
    eAccelerationStructureSizeKHR                             = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR,
    eMicromapSerializationSizeEXT                             = VK_QUERY_TYPE_MICROMAP_SERIALIZATION_SIZE_EXT,
    eMicromapCompactedSizeEXT                                 = VK_QUERY_TYPE_MICROMAP_COMPACTED_SIZE_EXT
  };

  enum class QueryPoolCreateFlagBits : VkQueryPoolCreateFlags
  {
  };

  using QueryPoolCreateFlags = Flags<QueryPoolCreateFlagBits>;

  template <>
  struct FlagTraits<QueryPoolCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR QueryPoolCreateFlags allFlags  = {};
  };

  enum class BufferCreateFlagBits : VkBufferCreateFlags
  {
    eSparseBinding                    = VK_BUFFER_CREATE_SPARSE_BINDING_BIT,
    eSparseResidency                  = VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT,
    eSparseAliased                    = VK_BUFFER_CREATE_SPARSE_ALIASED_BIT,
    eProtected                        = VK_BUFFER_CREATE_PROTECTED_BIT,
    eDeviceAddressCaptureReplay       = VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT,
    eDeviceAddressCaptureReplayEXT    = VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT_EXT,
    eDeviceAddressCaptureReplayKHR    = VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT_KHR,
    eDescriptorBufferCaptureReplayEXT = VK_BUFFER_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_EXT
  };

  using BufferCreateFlags = Flags<BufferCreateFlagBits>;

  template <>
  struct FlagTraits<BufferCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool              isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR BufferCreateFlags allFlags =
      BufferCreateFlagBits::eSparseBinding | BufferCreateFlagBits::eSparseResidency | BufferCreateFlagBits::eSparseAliased | BufferCreateFlagBits::eProtected |
      BufferCreateFlagBits::eDeviceAddressCaptureReplay | BufferCreateFlagBits::eDescriptorBufferCaptureReplayEXT;
  };

  enum class BufferUsageFlagBits : VkBufferUsageFlags
  {
    eTransferSrc                       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    eTransferDst                       = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    eUniformTexelBuffer                = VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT,
    eStorageTexelBuffer                = VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT,
    eUniformBuffer                     = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
    eStorageBuffer                     = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    eIndexBuffer                       = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
    eVertexBuffer                      = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
    eIndirectBuffer                    = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
    eShaderDeviceAddress               = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    eVideoDecodeSrcKHR                 = VK_BUFFER_USAGE_VIDEO_DECODE_SRC_BIT_KHR,
    eVideoDecodeDstKHR                 = VK_BUFFER_USAGE_VIDEO_DECODE_DST_BIT_KHR,
    eTransformFeedbackBufferEXT        = VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT,
    eTransformFeedbackCounterBufferEXT = VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT,
    eConditionalRenderingEXT           = VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eExecutionGraphScratchAMDX = VK_BUFFER_USAGE_EXECUTION_GRAPH_SCRATCH_BIT_AMDX,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eAccelerationStructureBuildInputReadOnlyKHR = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
    eAccelerationStructureStorageKHR            = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
    eShaderBindingTableKHR                      = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
    eRayTracingNV                               = VK_BUFFER_USAGE_RAY_TRACING_BIT_NV,
    eShaderDeviceAddressEXT                     = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_EXT,
    eShaderDeviceAddressKHR                     = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoEncodeDstKHR = VK_BUFFER_USAGE_VIDEO_ENCODE_DST_BIT_KHR,
    eVideoEncodeSrcKHR = VK_BUFFER_USAGE_VIDEO_ENCODE_SRC_BIT_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eSamplerDescriptorBufferEXT         = VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT,
    eResourceDescriptorBufferEXT        = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT,
    ePushDescriptorsDescriptorBufferEXT = VK_BUFFER_USAGE_PUSH_DESCRIPTORS_DESCRIPTOR_BUFFER_BIT_EXT,
    eMicromapBuildInputReadOnlyEXT      = VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT,
    eMicromapStorageEXT                 = VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT
  };

  using BufferUsageFlags = Flags<BufferUsageFlagBits>;

  template <>
  struct FlagTraits<BufferUsageFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR BufferUsageFlags allFlags =
      BufferUsageFlagBits::eTransferSrc | BufferUsageFlagBits::eTransferDst | BufferUsageFlagBits::eUniformTexelBuffer |
      BufferUsageFlagBits::eStorageTexelBuffer | BufferUsageFlagBits::eUniformBuffer | BufferUsageFlagBits::eStorageBuffer | BufferUsageFlagBits::eIndexBuffer |
      BufferUsageFlagBits::eVertexBuffer | BufferUsageFlagBits::eIndirectBuffer | BufferUsageFlagBits::eShaderDeviceAddress |
      BufferUsageFlagBits::eVideoDecodeSrcKHR | BufferUsageFlagBits::eVideoDecodeDstKHR | BufferUsageFlagBits::eTransformFeedbackBufferEXT |
      BufferUsageFlagBits::eTransformFeedbackCounterBufferEXT | BufferUsageFlagBits::eConditionalRenderingEXT
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      | BufferUsageFlagBits::eExecutionGraphScratchAMDX
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      | BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | BufferUsageFlagBits::eAccelerationStructureStorageKHR |
      BufferUsageFlagBits::eShaderBindingTableKHR
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      | BufferUsageFlagBits::eVideoEncodeDstKHR | BufferUsageFlagBits::eVideoEncodeSrcKHR
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      | BufferUsageFlagBits::eSamplerDescriptorBufferEXT | BufferUsageFlagBits::eResourceDescriptorBufferEXT |
      BufferUsageFlagBits::ePushDescriptorsDescriptorBufferEXT | BufferUsageFlagBits::eMicromapBuildInputReadOnlyEXT | BufferUsageFlagBits::eMicromapStorageEXT;
  };

  enum class SharingMode
  {
    eExclusive  = VK_SHARING_MODE_EXCLUSIVE,
    eConcurrent = VK_SHARING_MODE_CONCURRENT
  };

  enum class BufferViewCreateFlagBits : VkBufferViewCreateFlags
  {
  };

  using BufferViewCreateFlags = Flags<BufferViewCreateFlagBits>;

  template <>
  struct FlagTraits<BufferViewCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                  isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR BufferViewCreateFlags allFlags  = {};
  };

  enum class ImageLayout
  {
    eUndefined                                = VK_IMAGE_LAYOUT_UNDEFINED,
    eGeneral                                  = VK_IMAGE_LAYOUT_GENERAL,
    eColorAttachmentOptimal                   = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    eDepthStencilAttachmentOptimal            = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    eDepthStencilReadOnlyOptimal              = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
    eShaderReadOnlyOptimal                    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    eTransferSrcOptimal                       = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    eTransferDstOptimal                       = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    ePreinitialized                           = VK_IMAGE_LAYOUT_PREINITIALIZED,
    eDepthReadOnlyStencilAttachmentOptimal    = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
    eDepthAttachmentStencilReadOnlyOptimal    = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
    eDepthAttachmentOptimal                   = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
    eDepthReadOnlyOptimal                     = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
    eStencilAttachmentOptimal                 = VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL,
    eStencilReadOnlyOptimal                   = VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL,
    eReadOnlyOptimal                          = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
    eAttachmentOptimal                        = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
    ePresentSrcKHR                            = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    eVideoDecodeDstKHR                        = VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR,
    eVideoDecodeSrcKHR                        = VK_IMAGE_LAYOUT_VIDEO_DECODE_SRC_KHR,
    eVideoDecodeDpbKHR                        = VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR,
    eSharedPresentKHR                         = VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR,
    eDepthReadOnlyStencilAttachmentOptimalKHR = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL_KHR,
    eDepthAttachmentStencilReadOnlyOptimalKHR = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL_KHR,
    eShadingRateOptimalNV                     = VK_IMAGE_LAYOUT_SHADING_RATE_OPTIMAL_NV,
    eFragmentDensityMapOptimalEXT             = VK_IMAGE_LAYOUT_FRAGMENT_DENSITY_MAP_OPTIMAL_EXT,
    eFragmentShadingRateAttachmentOptimalKHR  = VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR,
    eDepthAttachmentOptimalKHR                = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL_KHR,
    eDepthReadOnlyOptimalKHR                  = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL_KHR,
    eStencilAttachmentOptimalKHR              = VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL_KHR,
    eStencilReadOnlyOptimalKHR                = VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL_KHR,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoEncodeDstKHR = VK_IMAGE_LAYOUT_VIDEO_ENCODE_DST_KHR,
    eVideoEncodeSrcKHR = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR,
    eVideoEncodeDpbKHR = VK_IMAGE_LAYOUT_VIDEO_ENCODE_DPB_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eReadOnlyOptimalKHR               = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL_KHR,
    eAttachmentOptimalKHR             = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR,
    eAttachmentFeedbackLoopOptimalEXT = VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
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
    eFragmentDensityMapDynamicEXT     = VK_IMAGE_VIEW_CREATE_FRAGMENT_DENSITY_MAP_DYNAMIC_BIT_EXT,
    eDescriptorBufferCaptureReplayEXT = VK_IMAGE_VIEW_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_EXT,
    eFragmentDensityMapDeferredEXT    = VK_IMAGE_VIEW_CREATE_FRAGMENT_DENSITY_MAP_DEFERRED_BIT_EXT
  };

  using ImageViewCreateFlags = Flags<ImageViewCreateFlagBits>;

  template <>
  struct FlagTraits<ImageViewCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ImageViewCreateFlags allFlags  = ImageViewCreateFlagBits::eFragmentDensityMapDynamicEXT |
                                                                         ImageViewCreateFlagBits::eDescriptorBufferCaptureReplayEXT |
                                                                         ImageViewCreateFlagBits::eFragmentDensityMapDeferredEXT;
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

  using ShaderModuleCreateFlags = Flags<ShaderModuleCreateFlagBits>;

  template <>
  struct FlagTraits<ShaderModuleCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                    isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ShaderModuleCreateFlags allFlags  = {};
  };

  enum class PipelineCacheCreateFlagBits : VkPipelineCacheCreateFlags
  {
    eExternallySynchronized    = VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT,
    eExternallySynchronizedEXT = VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT_EXT
  };

  using PipelineCacheCreateFlags = Flags<PipelineCacheCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineCacheCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                     isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineCacheCreateFlags allFlags  = PipelineCacheCreateFlagBits::eExternallySynchronized;
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

  using ColorComponentFlags = Flags<ColorComponentFlagBits>;

  template <>
  struct FlagTraits<ColorComponentFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ColorComponentFlags allFlags =
      ColorComponentFlagBits::eR | ColorComponentFlagBits::eG | ColorComponentFlagBits::eB | ColorComponentFlagBits::eA;
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

  using CullModeFlags = Flags<CullModeFlagBits>;

  template <>
  struct FlagTraits<CullModeFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool          isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR CullModeFlags allFlags =
      CullModeFlagBits::eNone | CullModeFlagBits::eFront | CullModeFlagBits::eBack | CullModeFlagBits::eFrontAndBack;
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
    eDiscardRectangleEnableEXT           = VK_DYNAMIC_STATE_DISCARD_RECTANGLE_ENABLE_EXT,
    eDiscardRectangleModeEXT             = VK_DYNAMIC_STATE_DISCARD_RECTANGLE_MODE_EXT,
    eSampleLocationsEXT                  = VK_DYNAMIC_STATE_SAMPLE_LOCATIONS_EXT,
    eRayTracingPipelineStackSizeKHR      = VK_DYNAMIC_STATE_RAY_TRACING_PIPELINE_STACK_SIZE_KHR,
    eViewportShadingRatePaletteNV        = VK_DYNAMIC_STATE_VIEWPORT_SHADING_RATE_PALETTE_NV,
    eViewportCoarseSampleOrderNV         = VK_DYNAMIC_STATE_VIEWPORT_COARSE_SAMPLE_ORDER_NV,
    eExclusiveScissorEnableNV            = VK_DYNAMIC_STATE_EXCLUSIVE_SCISSOR_ENABLE_NV,
    eExclusiveScissorNV                  = VK_DYNAMIC_STATE_EXCLUSIVE_SCISSOR_NV,
    eFragmentShadingRateKHR              = VK_DYNAMIC_STATE_FRAGMENT_SHADING_RATE_KHR,
    eLineStippleEXT                      = VK_DYNAMIC_STATE_LINE_STIPPLE_EXT,
    eCullModeEXT                         = VK_DYNAMIC_STATE_CULL_MODE_EXT,
    eFrontFaceEXT                        = VK_DYNAMIC_STATE_FRONT_FACE_EXT,
    ePrimitiveTopologyEXT                = VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY_EXT,
    eViewportWithCountEXT                = VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT_EXT,
    eScissorWithCountEXT                 = VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT_EXT,
    eVertexInputBindingStrideEXT         = VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE_EXT,
    eDepthTestEnableEXT                  = VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE_EXT,
    eDepthWriteEnableEXT                 = VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE_EXT,
    eDepthCompareOpEXT                   = VK_DYNAMIC_STATE_DEPTH_COMPARE_OP_EXT,
    eDepthBoundsTestEnableEXT            = VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE_EXT,
    eStencilTestEnableEXT                = VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE_EXT,
    eStencilOpEXT                        = VK_DYNAMIC_STATE_STENCIL_OP_EXT,
    eVertexInputEXT                      = VK_DYNAMIC_STATE_VERTEX_INPUT_EXT,
    ePatchControlPointsEXT               = VK_DYNAMIC_STATE_PATCH_CONTROL_POINTS_EXT,
    eRasterizerDiscardEnableEXT          = VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE_EXT,
    eDepthBiasEnableEXT                  = VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE_EXT,
    eLogicOpEXT                          = VK_DYNAMIC_STATE_LOGIC_OP_EXT,
    ePrimitiveRestartEnableEXT           = VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE_EXT,
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
    eAttachmentFeedbackLoopEnableEXT     = VK_DYNAMIC_STATE_ATTACHMENT_FEEDBACK_LOOP_ENABLE_EXT
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
    eVkPipelineRasterizationStateCreateFragmentShadingRateAttachmentKHR = VK_PIPELINE_RASTERIZATION_STATE_CREATE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
    eRenderingFragmentDensityMapAttachmentEXT                           = VK_PIPELINE_CREATE_RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT_BIT_EXT,
    eVkPipelineRasterizationStateCreateFragmentDensityMapAttachmentEXT  = VK_PIPELINE_RASTERIZATION_STATE_CREATE_FRAGMENT_DENSITY_MAP_ATTACHMENT_BIT_EXT,
    eViewIndexFromDeviceIndexKHR                                        = VK_PIPELINE_CREATE_VIEW_INDEX_FROM_DEVICE_INDEX_BIT_KHR,
    eDispatchBaseKHR                                                    = VK_PIPELINE_CREATE_DISPATCH_BASE_KHR,
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
    eFailOnPipelineCompileRequiredEXT                                   = VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_EXT,
    eEarlyReturnOnFailureEXT                                            = VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT_EXT,
    eDescriptorBufferEXT                                                = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT,
    eRetainLinkTimeOptimizationInfoEXT                                  = VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT,
    eLinkTimeOptimizationEXT                                            = VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT,
    eRayTracingAllowMotionNV                                            = VK_PIPELINE_CREATE_RAY_TRACING_ALLOW_MOTION_BIT_NV,
    eColorAttachmentFeedbackLoopEXT                                     = VK_PIPELINE_CREATE_COLOR_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT,
    eDepthStencilAttachmentFeedbackLoopEXT                              = VK_PIPELINE_CREATE_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT,
    eRayTracingOpacityMicromapEXT                                       = VK_PIPELINE_CREATE_RAY_TRACING_OPACITY_MICROMAP_BIT_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eRayTracingDisplacementMicromapNV = VK_PIPELINE_CREATE_RAY_TRACING_DISPLACEMENT_MICROMAP_BIT_NV,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eNoProtectedAccessEXT   = VK_PIPELINE_CREATE_NO_PROTECTED_ACCESS_BIT_EXT,
    eProtectedAccessOnlyEXT = VK_PIPELINE_CREATE_PROTECTED_ACCESS_ONLY_BIT_EXT
  };

  using PipelineCreateFlags = Flags<PipelineCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineCreateFlags allFlags =
      PipelineCreateFlagBits::eDisableOptimization | PipelineCreateFlagBits::eAllowDerivatives | PipelineCreateFlagBits::eDerivative |
      PipelineCreateFlagBits::eViewIndexFromDeviceIndex | PipelineCreateFlagBits::eDispatchBase | PipelineCreateFlagBits::eFailOnPipelineCompileRequired |
      PipelineCreateFlagBits::eEarlyReturnOnFailure | PipelineCreateFlagBits::eRenderingFragmentShadingRateAttachmentKHR |
      PipelineCreateFlagBits::eRenderingFragmentDensityMapAttachmentEXT | PipelineCreateFlagBits::eRayTracingNoNullAnyHitShadersKHR |
      PipelineCreateFlagBits::eRayTracingNoNullClosestHitShadersKHR | PipelineCreateFlagBits::eRayTracingNoNullMissShadersKHR |
      PipelineCreateFlagBits::eRayTracingNoNullIntersectionShadersKHR | PipelineCreateFlagBits::eRayTracingSkipTrianglesKHR |
      PipelineCreateFlagBits::eRayTracingSkipAabbsKHR | PipelineCreateFlagBits::eRayTracingShaderGroupHandleCaptureReplayKHR |
      PipelineCreateFlagBits::eDeferCompileNV | PipelineCreateFlagBits::eCaptureStatisticsKHR | PipelineCreateFlagBits::eCaptureInternalRepresentationsKHR |
      PipelineCreateFlagBits::eIndirectBindableNV | PipelineCreateFlagBits::eLibraryKHR | PipelineCreateFlagBits::eDescriptorBufferEXT |
      PipelineCreateFlagBits::eRetainLinkTimeOptimizationInfoEXT | PipelineCreateFlagBits::eLinkTimeOptimizationEXT |
      PipelineCreateFlagBits::eRayTracingAllowMotionNV | PipelineCreateFlagBits::eColorAttachmentFeedbackLoopEXT |
      PipelineCreateFlagBits::eDepthStencilAttachmentFeedbackLoopEXT | PipelineCreateFlagBits::eRayTracingOpacityMicromapEXT
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      | PipelineCreateFlagBits::eRayTracingDisplacementMicromapNV
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      | PipelineCreateFlagBits::eNoProtectedAccessEXT | PipelineCreateFlagBits::eProtectedAccessOnlyEXT;
  };

  enum class PipelineShaderStageCreateFlagBits : VkPipelineShaderStageCreateFlags
  {
    eAllowVaryingSubgroupSize    = VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT,
    eRequireFullSubgroups        = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT,
    eAllowVaryingSubgroupSizeEXT = VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT_EXT,
    eRequireFullSubgroupsEXT     = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT
  };

  using PipelineShaderStageCreateFlags = Flags<PipelineShaderStageCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineShaderStageCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineShaderStageCreateFlags allFlags =
      PipelineShaderStageCreateFlagBits::eAllowVaryingSubgroupSize | PipelineShaderStageCreateFlagBits::eRequireFullSubgroups;
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
    eRaygenNV               = VK_SHADER_STAGE_RAYGEN_BIT_NV,
    eAnyHitNV               = VK_SHADER_STAGE_ANY_HIT_BIT_NV,
    eClosestHitNV           = VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV,
    eMissNV                 = VK_SHADER_STAGE_MISS_BIT_NV,
    eIntersectionNV         = VK_SHADER_STAGE_INTERSECTION_BIT_NV,
    eCallableNV             = VK_SHADER_STAGE_CALLABLE_BIT_NV,
    eTaskNV                 = VK_SHADER_STAGE_TASK_BIT_NV,
    eMeshNV                 = VK_SHADER_STAGE_MESH_BIT_NV,
    eTaskEXT                = VK_SHADER_STAGE_TASK_BIT_EXT,
    eMeshEXT                = VK_SHADER_STAGE_MESH_BIT_EXT,
    eSubpassShadingHUAWEI   = VK_SHADER_STAGE_SUBPASS_SHADING_BIT_HUAWEI,
    eClusterCullingHUAWEI   = VK_SHADER_STAGE_CLUSTER_CULLING_BIT_HUAWEI
  };

  using ShaderStageFlags = Flags<ShaderStageFlagBits>;

  template <>
  struct FlagTraits<ShaderStageFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ShaderStageFlags allFlags =
      ShaderStageFlagBits::eVertex | ShaderStageFlagBits::eTessellationControl | ShaderStageFlagBits::eTessellationEvaluation | ShaderStageFlagBits::eGeometry |
      ShaderStageFlagBits::eFragment | ShaderStageFlagBits::eCompute | ShaderStageFlagBits::eAllGraphics | ShaderStageFlagBits::eAll |
      ShaderStageFlagBits::eRaygenKHR | ShaderStageFlagBits::eAnyHitKHR | ShaderStageFlagBits::eClosestHitKHR | ShaderStageFlagBits::eMissKHR |
      ShaderStageFlagBits::eIntersectionKHR | ShaderStageFlagBits::eCallableKHR | ShaderStageFlagBits::eTaskEXT | ShaderStageFlagBits::eMeshEXT |
      ShaderStageFlagBits::eSubpassShadingHUAWEI | ShaderStageFlagBits::eClusterCullingHUAWEI;
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

  enum class PipelineColorBlendStateCreateFlagBits : VkPipelineColorBlendStateCreateFlags
  {
    eRasterizationOrderAttachmentAccessARM = VK_PIPELINE_COLOR_BLEND_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_BIT_ARM,
    eRasterizationOrderAttachmentAccessEXT = VK_PIPELINE_COLOR_BLEND_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_BIT_EXT
  };

  using PipelineColorBlendStateCreateFlags = Flags<PipelineColorBlendStateCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineColorBlendStateCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                               isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineColorBlendStateCreateFlags allFlags =
      PipelineColorBlendStateCreateFlagBits::eRasterizationOrderAttachmentAccessEXT;
  };

  enum class PipelineDepthStencilStateCreateFlagBits : VkPipelineDepthStencilStateCreateFlags
  {
    eRasterizationOrderAttachmentDepthAccessARM   = VK_PIPELINE_DEPTH_STENCIL_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_BIT_ARM,
    eRasterizationOrderAttachmentStencilAccessARM = VK_PIPELINE_DEPTH_STENCIL_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_BIT_ARM,
    eRasterizationOrderAttachmentDepthAccessEXT   = VK_PIPELINE_DEPTH_STENCIL_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_BIT_EXT,
    eRasterizationOrderAttachmentStencilAccessEXT = VK_PIPELINE_DEPTH_STENCIL_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_BIT_EXT
  };

  using PipelineDepthStencilStateCreateFlags = Flags<PipelineDepthStencilStateCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineDepthStencilStateCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineDepthStencilStateCreateFlags allFlags =
      PipelineDepthStencilStateCreateFlagBits::eRasterizationOrderAttachmentDepthAccessEXT |
      PipelineDepthStencilStateCreateFlagBits::eRasterizationOrderAttachmentStencilAccessEXT;
  };

  enum class PipelineDynamicStateCreateFlagBits : VkPipelineDynamicStateCreateFlags
  {
  };

  using PipelineDynamicStateCreateFlags = Flags<PipelineDynamicStateCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineDynamicStateCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                            isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineDynamicStateCreateFlags allFlags  = {};
  };

  enum class PipelineInputAssemblyStateCreateFlagBits : VkPipelineInputAssemblyStateCreateFlags
  {
  };

  using PipelineInputAssemblyStateCreateFlags = Flags<PipelineInputAssemblyStateCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineInputAssemblyStateCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                  isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineInputAssemblyStateCreateFlags allFlags  = {};
  };

  enum class PipelineLayoutCreateFlagBits : VkPipelineLayoutCreateFlags
  {
    eIndependentSetsEXT = VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT
  };

  using PipelineLayoutCreateFlags = Flags<PipelineLayoutCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineLayoutCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                      isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineLayoutCreateFlags allFlags  = PipelineLayoutCreateFlagBits::eIndependentSetsEXT;
  };

  enum class PipelineMultisampleStateCreateFlagBits : VkPipelineMultisampleStateCreateFlags
  {
  };

  using PipelineMultisampleStateCreateFlags = Flags<PipelineMultisampleStateCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineMultisampleStateCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineMultisampleStateCreateFlags allFlags  = {};
  };

  enum class PipelineRasterizationStateCreateFlagBits : VkPipelineRasterizationStateCreateFlags
  {
  };

  using PipelineRasterizationStateCreateFlags = Flags<PipelineRasterizationStateCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineRasterizationStateCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                  isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineRasterizationStateCreateFlags allFlags  = {};
  };

  enum class PipelineTessellationStateCreateFlagBits : VkPipelineTessellationStateCreateFlags
  {
  };

  using PipelineTessellationStateCreateFlags = Flags<PipelineTessellationStateCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineTessellationStateCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineTessellationStateCreateFlags allFlags  = {};
  };

  enum class PipelineVertexInputStateCreateFlagBits : VkPipelineVertexInputStateCreateFlags
  {
  };

  using PipelineVertexInputStateCreateFlags = Flags<PipelineVertexInputStateCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineVertexInputStateCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineVertexInputStateCreateFlags allFlags  = {};
  };

  enum class PipelineViewportStateCreateFlagBits : VkPipelineViewportStateCreateFlags
  {
  };

  using PipelineViewportStateCreateFlags = Flags<PipelineViewportStateCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineViewportStateCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineViewportStateCreateFlags allFlags  = {};
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
    eCubicIMG = VK_FILTER_CUBIC_IMG,
    eCubicEXT = VK_FILTER_CUBIC_EXT
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
    eDescriptorBufferCaptureReplayEXT  = VK_SAMPLER_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_EXT,
    eNonSeamlessCubeMapEXT             = VK_SAMPLER_CREATE_NON_SEAMLESS_CUBE_MAP_BIT_EXT,
    eImageProcessingQCOM               = VK_SAMPLER_CREATE_IMAGE_PROCESSING_BIT_QCOM
  };

  using SamplerCreateFlags = Flags<SamplerCreateFlagBits>;

  template <>
  struct FlagTraits<SamplerCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool               isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SamplerCreateFlags allFlags =
      SamplerCreateFlagBits::eSubsampledEXT | SamplerCreateFlagBits::eSubsampledCoarseReconstructionEXT |
      SamplerCreateFlagBits::eDescriptorBufferCaptureReplayEXT | SamplerCreateFlagBits::eNonSeamlessCubeMapEXT | SamplerCreateFlagBits::eImageProcessingQCOM;
  };

  enum class SamplerMipmapMode
  {
    eNearest = VK_SAMPLER_MIPMAP_MODE_NEAREST,
    eLinear  = VK_SAMPLER_MIPMAP_MODE_LINEAR
  };

  enum class DescriptorPoolCreateFlagBits : VkDescriptorPoolCreateFlags
  {
    eFreeDescriptorSet          = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
    eUpdateAfterBind            = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
    eUpdateAfterBindEXT         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT_EXT,
    eHostOnlyVALVE              = VK_DESCRIPTOR_POOL_CREATE_HOST_ONLY_BIT_VALVE,
    eHostOnlyEXT                = VK_DESCRIPTOR_POOL_CREATE_HOST_ONLY_BIT_EXT,
    eAllowOverallocationSetsNV  = VK_DESCRIPTOR_POOL_CREATE_ALLOW_OVERALLOCATION_SETS_BIT_NV,
    eAllowOverallocationPoolsNV = VK_DESCRIPTOR_POOL_CREATE_ALLOW_OVERALLOCATION_POOLS_BIT_NV
  };

  using DescriptorPoolCreateFlags = Flags<DescriptorPoolCreateFlagBits>;

  template <>
  struct FlagTraits<DescriptorPoolCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                      isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DescriptorPoolCreateFlags allFlags =
      DescriptorPoolCreateFlagBits::eFreeDescriptorSet | DescriptorPoolCreateFlagBits::eUpdateAfterBind | DescriptorPoolCreateFlagBits::eHostOnlyEXT |
      DescriptorPoolCreateFlagBits::eAllowOverallocationSetsNV | DescriptorPoolCreateFlagBits::eAllowOverallocationPoolsNV;
  };

  enum class DescriptorSetLayoutCreateFlagBits : VkDescriptorSetLayoutCreateFlags
  {
    eUpdateAfterBindPool          = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
    ePushDescriptorKHR            = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
    eUpdateAfterBindPoolEXT       = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT_EXT,
    eDescriptorBufferEXT          = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT,
    eEmbeddedImmutableSamplersEXT = VK_DESCRIPTOR_SET_LAYOUT_CREATE_EMBEDDED_IMMUTABLE_SAMPLERS_BIT_EXT,
    eHostOnlyPoolVALVE            = VK_DESCRIPTOR_SET_LAYOUT_CREATE_HOST_ONLY_POOL_BIT_VALVE,
    eIndirectBindableNV           = VK_DESCRIPTOR_SET_LAYOUT_CREATE_INDIRECT_BINDABLE_BIT_NV,
    eHostOnlyPoolEXT              = VK_DESCRIPTOR_SET_LAYOUT_CREATE_HOST_ONLY_POOL_BIT_EXT
  };

  using DescriptorSetLayoutCreateFlags = Flags<DescriptorSetLayoutCreateFlagBits>;

  template <>
  struct FlagTraits<DescriptorSetLayoutCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DescriptorSetLayoutCreateFlags allFlags =
      DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool | DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR |
      DescriptorSetLayoutCreateFlagBits::eDescriptorBufferEXT | DescriptorSetLayoutCreateFlagBits::eEmbeddedImmutableSamplersEXT |
      DescriptorSetLayoutCreateFlagBits::eIndirectBindableNV | DescriptorSetLayoutCreateFlagBits::eHostOnlyPoolEXT;
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
    eInlineUniformBlockEXT    = VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT,
    eAccelerationStructureKHR = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
    eAccelerationStructureNV  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV,
    eMutableVALVE             = VK_DESCRIPTOR_TYPE_MUTABLE_VALVE,
    eSampleWeightImageQCOM    = VK_DESCRIPTOR_TYPE_SAMPLE_WEIGHT_IMAGE_QCOM,
    eBlockMatchImageQCOM      = VK_DESCRIPTOR_TYPE_BLOCK_MATCH_IMAGE_QCOM,
    eMutableEXT               = VK_DESCRIPTOR_TYPE_MUTABLE_EXT
  };

  enum class DescriptorPoolResetFlagBits : VkDescriptorPoolResetFlags
  {
  };

  using DescriptorPoolResetFlags = Flags<DescriptorPoolResetFlagBits>;

  template <>
  struct FlagTraits<DescriptorPoolResetFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                     isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DescriptorPoolResetFlags allFlags  = {};
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
    eShadingRateImageReadNV               = VK_ACCESS_SHADING_RATE_IMAGE_READ_BIT_NV,
    eAccelerationStructureReadNV          = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV,
    eAccelerationStructureWriteNV         = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV,
    eFragmentDensityMapReadEXT            = VK_ACCESS_FRAGMENT_DENSITY_MAP_READ_BIT_EXT,
    eFragmentShadingRateAttachmentReadKHR = VK_ACCESS_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR,
    eCommandPreprocessReadNV              = VK_ACCESS_COMMAND_PREPROCESS_READ_BIT_NV,
    eCommandPreprocessWriteNV             = VK_ACCESS_COMMAND_PREPROCESS_WRITE_BIT_NV,
    eNoneKHR                              = VK_ACCESS_NONE_KHR
  };

  using AccessFlags = Flags<AccessFlagBits>;

  template <>
  struct FlagTraits<AccessFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool        isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR AccessFlags allFlags =
      AccessFlagBits::eIndirectCommandRead | AccessFlagBits::eIndexRead | AccessFlagBits::eVertexAttributeRead | AccessFlagBits::eUniformRead |
      AccessFlagBits::eInputAttachmentRead | AccessFlagBits::eShaderRead | AccessFlagBits::eShaderWrite | AccessFlagBits::eColorAttachmentRead |
      AccessFlagBits::eColorAttachmentWrite | AccessFlagBits::eDepthStencilAttachmentRead | AccessFlagBits::eDepthStencilAttachmentWrite |
      AccessFlagBits::eTransferRead | AccessFlagBits::eTransferWrite | AccessFlagBits::eHostRead | AccessFlagBits::eHostWrite | AccessFlagBits::eMemoryRead |
      AccessFlagBits::eMemoryWrite | AccessFlagBits::eNone | AccessFlagBits::eTransformFeedbackWriteEXT | AccessFlagBits::eTransformFeedbackCounterReadEXT |
      AccessFlagBits::eTransformFeedbackCounterWriteEXT | AccessFlagBits::eConditionalRenderingReadEXT | AccessFlagBits::eColorAttachmentReadNoncoherentEXT |
      AccessFlagBits::eAccelerationStructureReadKHR | AccessFlagBits::eAccelerationStructureWriteKHR | AccessFlagBits::eFragmentDensityMapReadEXT |
      AccessFlagBits::eFragmentShadingRateAttachmentReadKHR | AccessFlagBits::eCommandPreprocessReadNV | AccessFlagBits::eCommandPreprocessWriteNV;
  };

  enum class AttachmentDescriptionFlagBits : VkAttachmentDescriptionFlags
  {
    eMayAlias = VK_ATTACHMENT_DESCRIPTION_MAY_ALIAS_BIT
  };

  using AttachmentDescriptionFlags = Flags<AttachmentDescriptionFlagBits>;

  template <>
  struct FlagTraits<AttachmentDescriptionFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR AttachmentDescriptionFlags allFlags  = AttachmentDescriptionFlagBits::eMayAlias;
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
    eNoneKHR  = VK_ATTACHMENT_STORE_OP_NONE_KHR,
    eNoneQCOM = VK_ATTACHMENT_STORE_OP_NONE_QCOM,
    eNoneEXT  = VK_ATTACHMENT_STORE_OP_NONE_EXT
  };

  enum class DependencyFlagBits : VkDependencyFlags
  {
    eByRegion        = VK_DEPENDENCY_BY_REGION_BIT,
    eDeviceGroup     = VK_DEPENDENCY_DEVICE_GROUP_BIT,
    eViewLocal       = VK_DEPENDENCY_VIEW_LOCAL_BIT,
    eViewLocalKHR    = VK_DEPENDENCY_VIEW_LOCAL_BIT_KHR,
    eDeviceGroupKHR  = VK_DEPENDENCY_DEVICE_GROUP_BIT_KHR,
    eFeedbackLoopEXT = VK_DEPENDENCY_FEEDBACK_LOOP_BIT_EXT
  };

  using DependencyFlags = Flags<DependencyFlagBits>;

  template <>
  struct FlagTraits<DependencyFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool            isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DependencyFlags allFlags =
      DependencyFlagBits::eByRegion | DependencyFlagBits::eDeviceGroup | DependencyFlagBits::eViewLocal | DependencyFlagBits::eFeedbackLoopEXT;
  };

  enum class FramebufferCreateFlagBits : VkFramebufferCreateFlags
  {
    eImageless    = VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT,
    eImagelessKHR = VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT_KHR
  };

  using FramebufferCreateFlags = Flags<FramebufferCreateFlagBits>;

  template <>
  struct FlagTraits<FramebufferCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR FramebufferCreateFlags allFlags  = FramebufferCreateFlagBits::eImageless;
  };

  enum class PipelineBindPoint
  {
    eGraphics = VK_PIPELINE_BIND_POINT_GRAPHICS,
    eCompute  = VK_PIPELINE_BIND_POINT_COMPUTE,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eExecutionGraphAMDX = VK_PIPELINE_BIND_POINT_EXECUTION_GRAPH_AMDX,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eRayTracingKHR        = VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
    eRayTracingNV         = VK_PIPELINE_BIND_POINT_RAY_TRACING_NV,
    eSubpassShadingHUAWEI = VK_PIPELINE_BIND_POINT_SUBPASS_SHADING_HUAWEI
  };

  enum class RenderPassCreateFlagBits : VkRenderPassCreateFlags
  {
    eTransformQCOM = VK_RENDER_PASS_CREATE_TRANSFORM_BIT_QCOM
  };

  using RenderPassCreateFlags = Flags<RenderPassCreateFlagBits>;

  template <>
  struct FlagTraits<RenderPassCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                  isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR RenderPassCreateFlags allFlags  = RenderPassCreateFlagBits::eTransformQCOM;
  };

  enum class SubpassDescriptionFlagBits : VkSubpassDescriptionFlags
  {
    ePerViewAttributesNVX                         = VK_SUBPASS_DESCRIPTION_PER_VIEW_ATTRIBUTES_BIT_NVX,
    ePerViewPositionXOnlyNVX                      = VK_SUBPASS_DESCRIPTION_PER_VIEW_POSITION_X_ONLY_BIT_NVX,
    eFragmentRegionQCOM                           = VK_SUBPASS_DESCRIPTION_FRAGMENT_REGION_BIT_QCOM,
    eShaderResolveQCOM                            = VK_SUBPASS_DESCRIPTION_SHADER_RESOLVE_BIT_QCOM,
    eRasterizationOrderAttachmentColorAccessARM   = VK_SUBPASS_DESCRIPTION_RASTERIZATION_ORDER_ATTACHMENT_COLOR_ACCESS_BIT_ARM,
    eRasterizationOrderAttachmentDepthAccessARM   = VK_SUBPASS_DESCRIPTION_RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_BIT_ARM,
    eRasterizationOrderAttachmentStencilAccessARM = VK_SUBPASS_DESCRIPTION_RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_BIT_ARM,
    eRasterizationOrderAttachmentColorAccessEXT   = VK_SUBPASS_DESCRIPTION_RASTERIZATION_ORDER_ATTACHMENT_COLOR_ACCESS_BIT_EXT,
    eRasterizationOrderAttachmentDepthAccessEXT   = VK_SUBPASS_DESCRIPTION_RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_BIT_EXT,
    eRasterizationOrderAttachmentStencilAccessEXT = VK_SUBPASS_DESCRIPTION_RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_BIT_EXT,
    eEnableLegacyDitheringEXT                     = VK_SUBPASS_DESCRIPTION_ENABLE_LEGACY_DITHERING_BIT_EXT
  };

  using SubpassDescriptionFlags = Flags<SubpassDescriptionFlagBits>;

  template <>
  struct FlagTraits<SubpassDescriptionFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                    isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SubpassDescriptionFlags allFlags =
      SubpassDescriptionFlagBits::ePerViewAttributesNVX | SubpassDescriptionFlagBits::ePerViewPositionXOnlyNVX |
      SubpassDescriptionFlagBits::eFragmentRegionQCOM | SubpassDescriptionFlagBits::eShaderResolveQCOM |
      SubpassDescriptionFlagBits::eRasterizationOrderAttachmentColorAccessEXT | SubpassDescriptionFlagBits::eRasterizationOrderAttachmentDepthAccessEXT |
      SubpassDescriptionFlagBits::eRasterizationOrderAttachmentStencilAccessEXT | SubpassDescriptionFlagBits::eEnableLegacyDitheringEXT;
  };

  enum class CommandPoolCreateFlagBits : VkCommandPoolCreateFlags
  {
    eTransient          = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
    eResetCommandBuffer = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    eProtected          = VK_COMMAND_POOL_CREATE_PROTECTED_BIT
  };

  using CommandPoolCreateFlags = Flags<CommandPoolCreateFlagBits>;

  template <>
  struct FlagTraits<CommandPoolCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR CommandPoolCreateFlags allFlags =
      CommandPoolCreateFlagBits::eTransient | CommandPoolCreateFlagBits::eResetCommandBuffer | CommandPoolCreateFlagBits::eProtected;
  };

  enum class CommandPoolResetFlagBits : VkCommandPoolResetFlags
  {
    eReleaseResources = VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT
  };

  using CommandPoolResetFlags = Flags<CommandPoolResetFlagBits>;

  template <>
  struct FlagTraits<CommandPoolResetFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                  isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR CommandPoolResetFlags allFlags  = CommandPoolResetFlagBits::eReleaseResources;
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

  using CommandBufferResetFlags = Flags<CommandBufferResetFlagBits>;

  template <>
  struct FlagTraits<CommandBufferResetFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                    isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR CommandBufferResetFlags allFlags  = CommandBufferResetFlagBits::eReleaseResources;
  };

  enum class CommandBufferUsageFlagBits : VkCommandBufferUsageFlags
  {
    eOneTimeSubmit      = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    eRenderPassContinue = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
    eSimultaneousUse    = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT
  };

  using CommandBufferUsageFlags = Flags<CommandBufferUsageFlagBits>;

  template <>
  struct FlagTraits<CommandBufferUsageFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                    isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR CommandBufferUsageFlags allFlags =
      CommandBufferUsageFlagBits::eOneTimeSubmit | CommandBufferUsageFlagBits::eRenderPassContinue | CommandBufferUsageFlagBits::eSimultaneousUse;
  };

  enum class QueryControlFlagBits : VkQueryControlFlags
  {
    ePrecise = VK_QUERY_CONTROL_PRECISE_BIT
  };

  using QueryControlFlags = Flags<QueryControlFlagBits>;

  template <>
  struct FlagTraits<QueryControlFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool              isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR QueryControlFlags allFlags  = QueryControlFlagBits::ePrecise;
  };

  enum class IndexType
  {
    eUint16   = VK_INDEX_TYPE_UINT16,
    eUint32   = VK_INDEX_TYPE_UINT32,
    eNoneKHR  = VK_INDEX_TYPE_NONE_KHR,
    eNoneNV   = VK_INDEX_TYPE_NONE_NV,
    eUint8EXT = VK_INDEX_TYPE_UINT8_EXT
  };

  enum class StencilFaceFlagBits : VkStencilFaceFlags
  {
    eFront                 = VK_STENCIL_FACE_FRONT_BIT,
    eBack                  = VK_STENCIL_FACE_BACK_BIT,
    eFrontAndBack          = VK_STENCIL_FACE_FRONT_AND_BACK,
    eVkStencilFrontAndBack = VK_STENCIL_FRONT_AND_BACK
  };

  using StencilFaceFlags = Flags<StencilFaceFlagBits>;

  template <>
  struct FlagTraits<StencilFaceFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR StencilFaceFlags allFlags =
      StencilFaceFlagBits::eFront | StencilFaceFlagBits::eBack | StencilFaceFlagBits::eFrontAndBack;
  };

  enum class SubpassContents
  {
    eInline                              = VK_SUBPASS_CONTENTS_INLINE,
    eSecondaryCommandBuffers             = VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS,
    eInlineAndSecondaryCommandBuffersEXT = VK_SUBPASS_CONTENTS_INLINE_AND_SECONDARY_COMMAND_BUFFERS_EXT
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

  using SubgroupFeatureFlags = Flags<SubgroupFeatureFlagBits>;

  template <>
  struct FlagTraits<SubgroupFeatureFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SubgroupFeatureFlags allFlags =
      SubgroupFeatureFlagBits::eBasic | SubgroupFeatureFlagBits::eVote | SubgroupFeatureFlagBits::eArithmetic | SubgroupFeatureFlagBits::eBallot |
      SubgroupFeatureFlagBits::eShuffle | SubgroupFeatureFlagBits::eShuffleRelative | SubgroupFeatureFlagBits::eClustered | SubgroupFeatureFlagBits::eQuad |
      SubgroupFeatureFlagBits::ePartitionedNV;
  };

  enum class PeerMemoryFeatureFlagBits : VkPeerMemoryFeatureFlags
  {
    eCopySrc    = VK_PEER_MEMORY_FEATURE_COPY_SRC_BIT,
    eCopyDst    = VK_PEER_MEMORY_FEATURE_COPY_DST_BIT,
    eGenericSrc = VK_PEER_MEMORY_FEATURE_GENERIC_SRC_BIT,
    eGenericDst = VK_PEER_MEMORY_FEATURE_GENERIC_DST_BIT
  };
  using PeerMemoryFeatureFlagBitsKHR = PeerMemoryFeatureFlagBits;

  using PeerMemoryFeatureFlags    = Flags<PeerMemoryFeatureFlagBits>;
  using PeerMemoryFeatureFlagsKHR = PeerMemoryFeatureFlags;

  template <>
  struct FlagTraits<PeerMemoryFeatureFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PeerMemoryFeatureFlags allFlags  = PeerMemoryFeatureFlagBits::eCopySrc | PeerMemoryFeatureFlagBits::eCopyDst |
                                                                           PeerMemoryFeatureFlagBits::eGenericSrc | PeerMemoryFeatureFlagBits::eGenericDst;
  };

  enum class MemoryAllocateFlagBits : VkMemoryAllocateFlags
  {
    eDeviceMask                 = VK_MEMORY_ALLOCATE_DEVICE_MASK_BIT,
    eDeviceAddress              = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    eDeviceAddressCaptureReplay = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT
  };
  using MemoryAllocateFlagBitsKHR = MemoryAllocateFlagBits;

  using MemoryAllocateFlags    = Flags<MemoryAllocateFlagBits>;
  using MemoryAllocateFlagsKHR = MemoryAllocateFlags;

  template <>
  struct FlagTraits<MemoryAllocateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR MemoryAllocateFlags allFlags =
      MemoryAllocateFlagBits::eDeviceMask | MemoryAllocateFlagBits::eDeviceAddress | MemoryAllocateFlagBits::eDeviceAddressCaptureReplay;
  };

  enum class CommandPoolTrimFlagBits : VkCommandPoolTrimFlags
  {
  };

  using CommandPoolTrimFlags    = Flags<CommandPoolTrimFlagBits>;
  using CommandPoolTrimFlagsKHR = CommandPoolTrimFlags;

  template <>
  struct FlagTraits<CommandPoolTrimFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR CommandPoolTrimFlags allFlags  = {};
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

  using DescriptorUpdateTemplateCreateFlags    = Flags<DescriptorUpdateTemplateCreateFlagBits>;
  using DescriptorUpdateTemplateCreateFlagsKHR = DescriptorUpdateTemplateCreateFlags;

  template <>
  struct FlagTraits<DescriptorUpdateTemplateCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DescriptorUpdateTemplateCreateFlags allFlags  = {};
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
    eRdmaAddressNV = VK_EXTERNAL_MEMORY_HANDLE_TYPE_RDMA_ADDRESS_BIT_NV,
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
    eScreenBufferQNX = VK_EXTERNAL_MEMORY_HANDLE_TYPE_SCREEN_BUFFER_BIT_QNX
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/
  };
  using ExternalMemoryHandleTypeFlagBitsKHR = ExternalMemoryHandleTypeFlagBits;

  using ExternalMemoryHandleTypeFlags    = Flags<ExternalMemoryHandleTypeFlagBits>;
  using ExternalMemoryHandleTypeFlagsKHR = ExternalMemoryHandleTypeFlags;

  template <>
  struct FlagTraits<ExternalMemoryHandleTypeFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                          isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ExternalMemoryHandleTypeFlags allFlags =
      ExternalMemoryHandleTypeFlagBits::eOpaqueFd | ExternalMemoryHandleTypeFlagBits::eOpaqueWin32 | ExternalMemoryHandleTypeFlagBits::eOpaqueWin32Kmt |
      ExternalMemoryHandleTypeFlagBits::eD3D11Texture | ExternalMemoryHandleTypeFlagBits::eD3D11TextureKmt | ExternalMemoryHandleTypeFlagBits::eD3D12Heap |
      ExternalMemoryHandleTypeFlagBits::eD3D12Resource | ExternalMemoryHandleTypeFlagBits::eDmaBufEXT
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
      | ExternalMemoryHandleTypeFlagBits::eAndroidHardwareBufferANDROID
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
      | ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT | ExternalMemoryHandleTypeFlagBits::eHostMappedForeignMemoryEXT
#if defined( VK_USE_PLATFORM_FUCHSIA )
      | ExternalMemoryHandleTypeFlagBits::eZirconVmoFUCHSIA
#endif /*VK_USE_PLATFORM_FUCHSIA*/
      | ExternalMemoryHandleTypeFlagBits::eRdmaAddressNV
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
      | ExternalMemoryHandleTypeFlagBits::eScreenBufferQNX
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/
      ;
  };

  enum class ExternalMemoryFeatureFlagBits : VkExternalMemoryFeatureFlags
  {
    eDedicatedOnly = VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT,
    eExportable    = VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT,
    eImportable    = VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT
  };
  using ExternalMemoryFeatureFlagBitsKHR = ExternalMemoryFeatureFlagBits;

  using ExternalMemoryFeatureFlags    = Flags<ExternalMemoryFeatureFlagBits>;
  using ExternalMemoryFeatureFlagsKHR = ExternalMemoryFeatureFlags;

  template <>
  struct FlagTraits<ExternalMemoryFeatureFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ExternalMemoryFeatureFlags allFlags =
      ExternalMemoryFeatureFlagBits::eDedicatedOnly | ExternalMemoryFeatureFlagBits::eExportable | ExternalMemoryFeatureFlagBits::eImportable;
  };

  enum class ExternalFenceHandleTypeFlagBits : VkExternalFenceHandleTypeFlags
  {
    eOpaqueFd       = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT,
    eOpaqueWin32    = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_WIN32_BIT,
    eOpaqueWin32Kmt = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT,
    eSyncFd         = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT
  };
  using ExternalFenceHandleTypeFlagBitsKHR = ExternalFenceHandleTypeFlagBits;

  using ExternalFenceHandleTypeFlags    = Flags<ExternalFenceHandleTypeFlagBits>;
  using ExternalFenceHandleTypeFlagsKHR = ExternalFenceHandleTypeFlags;

  template <>
  struct FlagTraits<ExternalFenceHandleTypeFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                         isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ExternalFenceHandleTypeFlags allFlags =
      ExternalFenceHandleTypeFlagBits::eOpaqueFd | ExternalFenceHandleTypeFlagBits::eOpaqueWin32 | ExternalFenceHandleTypeFlagBits::eOpaqueWin32Kmt |
      ExternalFenceHandleTypeFlagBits::eSyncFd;
  };

  enum class ExternalFenceFeatureFlagBits : VkExternalFenceFeatureFlags
  {
    eExportable = VK_EXTERNAL_FENCE_FEATURE_EXPORTABLE_BIT,
    eImportable = VK_EXTERNAL_FENCE_FEATURE_IMPORTABLE_BIT
  };
  using ExternalFenceFeatureFlagBitsKHR = ExternalFenceFeatureFlagBits;

  using ExternalFenceFeatureFlags    = Flags<ExternalFenceFeatureFlagBits>;
  using ExternalFenceFeatureFlagsKHR = ExternalFenceFeatureFlags;

  template <>
  struct FlagTraits<ExternalFenceFeatureFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                      isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ExternalFenceFeatureFlags allFlags =
      ExternalFenceFeatureFlagBits::eExportable | ExternalFenceFeatureFlagBits::eImportable;
  };

  enum class FenceImportFlagBits : VkFenceImportFlags
  {
    eTemporary = VK_FENCE_IMPORT_TEMPORARY_BIT
  };
  using FenceImportFlagBitsKHR = FenceImportFlagBits;

  using FenceImportFlags    = Flags<FenceImportFlagBits>;
  using FenceImportFlagsKHR = FenceImportFlags;

  template <>
  struct FlagTraits<FenceImportFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR FenceImportFlags allFlags  = FenceImportFlagBits::eTemporary;
  };

  enum class SemaphoreImportFlagBits : VkSemaphoreImportFlags
  {
    eTemporary = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT
  };
  using SemaphoreImportFlagBitsKHR = SemaphoreImportFlagBits;

  using SemaphoreImportFlags    = Flags<SemaphoreImportFlagBits>;
  using SemaphoreImportFlagsKHR = SemaphoreImportFlags;

  template <>
  struct FlagTraits<SemaphoreImportFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SemaphoreImportFlags allFlags  = SemaphoreImportFlagBits::eTemporary;
  };

  enum class ExternalSemaphoreHandleTypeFlagBits : VkExternalSemaphoreHandleTypeFlags
  {
    eOpaqueFd       = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
    eOpaqueWin32    = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT,
    eOpaqueWin32Kmt = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT,
    eD3D12Fence     = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT,
    eD3D11Fence     = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE_BIT,
    eSyncFd         = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
#if defined( VK_USE_PLATFORM_FUCHSIA )
    eZirconEventFUCHSIA = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_ZIRCON_EVENT_BIT_FUCHSIA
#endif /*VK_USE_PLATFORM_FUCHSIA*/
  };
  using ExternalSemaphoreHandleTypeFlagBitsKHR = ExternalSemaphoreHandleTypeFlagBits;

  using ExternalSemaphoreHandleTypeFlags    = Flags<ExternalSemaphoreHandleTypeFlagBits>;
  using ExternalSemaphoreHandleTypeFlagsKHR = ExternalSemaphoreHandleTypeFlags;

  template <>
  struct FlagTraits<ExternalSemaphoreHandleTypeFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ExternalSemaphoreHandleTypeFlags allFlags =
      ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd | ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32 |
      ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32Kmt | ExternalSemaphoreHandleTypeFlagBits::eD3D12Fence | ExternalSemaphoreHandleTypeFlagBits::eSyncFd
#if defined( VK_USE_PLATFORM_FUCHSIA )
      | ExternalSemaphoreHandleTypeFlagBits::eZirconEventFUCHSIA
#endif /*VK_USE_PLATFORM_FUCHSIA*/
      ;
  };

  enum class ExternalSemaphoreFeatureFlagBits : VkExternalSemaphoreFeatureFlags
  {
    eExportable = VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT,
    eImportable = VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT
  };
  using ExternalSemaphoreFeatureFlagBitsKHR = ExternalSemaphoreFeatureFlagBits;

  using ExternalSemaphoreFeatureFlags    = Flags<ExternalSemaphoreFeatureFlagBits>;
  using ExternalSemaphoreFeatureFlagsKHR = ExternalSemaphoreFeatureFlags;

  template <>
  struct FlagTraits<ExternalSemaphoreFeatureFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                          isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ExternalSemaphoreFeatureFlags allFlags =
      ExternalSemaphoreFeatureFlagBits::eExportable | ExternalSemaphoreFeatureFlagBits::eImportable;
  };

  //=== VK_VERSION_1_2 ===

  enum class DriverId
  {
    eAmdProprietary            = VK_DRIVER_ID_AMD_PROPRIETARY,
    eAmdOpenSource             = VK_DRIVER_ID_AMD_OPEN_SOURCE,
    eMesaRadv                  = VK_DRIVER_ID_MESA_RADV,
    eNvidiaProprietary         = VK_DRIVER_ID_NVIDIA_PROPRIETARY,
    eIntelProprietaryWindows   = VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS,
    eIntelOpenSourceMESA       = VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA,
    eImaginationProprietary    = VK_DRIVER_ID_IMAGINATION_PROPRIETARY,
    eQualcommProprietary       = VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
    eArmProprietary            = VK_DRIVER_ID_ARM_PROPRIETARY,
    eGoogleSwiftshader         = VK_DRIVER_ID_GOOGLE_SWIFTSHADER,
    eGgpProprietary            = VK_DRIVER_ID_GGP_PROPRIETARY,
    eBroadcomProprietary       = VK_DRIVER_ID_BROADCOM_PROPRIETARY,
    eMesaLlvmpipe              = VK_DRIVER_ID_MESA_LLVMPIPE,
    eMoltenvk                  = VK_DRIVER_ID_MOLTENVK,
    eCoreaviProprietary        = VK_DRIVER_ID_COREAVI_PROPRIETARY,
    eJuiceProprietary          = VK_DRIVER_ID_JUICE_PROPRIETARY,
    eVerisiliconProprietary    = VK_DRIVER_ID_VERISILICON_PROPRIETARY,
    eMesaTurnip                = VK_DRIVER_ID_MESA_TURNIP,
    eMesaV3Dv                  = VK_DRIVER_ID_MESA_V3DV,
    eMesaPanvk                 = VK_DRIVER_ID_MESA_PANVK,
    eSamsungProprietary        = VK_DRIVER_ID_SAMSUNG_PROPRIETARY,
    eMesaVenus                 = VK_DRIVER_ID_MESA_VENUS,
    eMesaDozen                 = VK_DRIVER_ID_MESA_DOZEN,
    eMesaNvk                   = VK_DRIVER_ID_MESA_NVK,
    eImaginationOpenSourceMESA = VK_DRIVER_ID_IMAGINATION_OPEN_SOURCE_MESA,
    eMesaAgxv                  = VK_DRIVER_ID_MESA_AGXV
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

  using DescriptorBindingFlags    = Flags<DescriptorBindingFlagBits>;
  using DescriptorBindingFlagsEXT = DescriptorBindingFlags;

  template <>
  struct FlagTraits<DescriptorBindingFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DescriptorBindingFlags allFlags =
      DescriptorBindingFlagBits::eUpdateAfterBind | DescriptorBindingFlagBits::eUpdateUnusedWhilePending | DescriptorBindingFlagBits::ePartiallyBound |
      DescriptorBindingFlagBits::eVariableDescriptorCount;
  };

  enum class ResolveModeFlagBits : VkResolveModeFlags
  {
    eNone       = VK_RESOLVE_MODE_NONE,
    eSampleZero = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT,
    eAverage    = VK_RESOLVE_MODE_AVERAGE_BIT,
    eMin        = VK_RESOLVE_MODE_MIN_BIT,
    eMax        = VK_RESOLVE_MODE_MAX_BIT,
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
    eExternalFormatDownsampleANDROID = VK_RESOLVE_MODE_EXTERNAL_FORMAT_DOWNSAMPLE_ANDROID
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
  };
  using ResolveModeFlagBitsKHR = ResolveModeFlagBits;

  using ResolveModeFlags    = Flags<ResolveModeFlagBits>;
  using ResolveModeFlagsKHR = ResolveModeFlags;

  template <>
  struct FlagTraits<ResolveModeFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ResolveModeFlags allFlags  = ResolveModeFlagBits::eNone | ResolveModeFlagBits::eSampleZero |
                                                                     ResolveModeFlagBits::eAverage | ResolveModeFlagBits::eMin | ResolveModeFlagBits::eMax
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
                                                                     | ResolveModeFlagBits::eExternalFormatDownsampleANDROID
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
      ;
  };

  enum class SamplerReductionMode
  {
    eWeightedAverage               = VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE,
    eMin                           = VK_SAMPLER_REDUCTION_MODE_MIN,
    eMax                           = VK_SAMPLER_REDUCTION_MODE_MAX,
    eWeightedAverageRangeclampQCOM = VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE_RANGECLAMP_QCOM
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

  using SemaphoreWaitFlags    = Flags<SemaphoreWaitFlagBits>;
  using SemaphoreWaitFlagsKHR = SemaphoreWaitFlags;

  template <>
  struct FlagTraits<SemaphoreWaitFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool               isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SemaphoreWaitFlags allFlags  = SemaphoreWaitFlagBits::eAny;
  };

  //=== VK_VERSION_1_3 ===

  enum class PipelineCreationFeedbackFlagBits : VkPipelineCreationFeedbackFlags
  {
    eValid                       = VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT,
    eApplicationPipelineCacheHit = VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT,
    eBasePipelineAcceleration    = VK_PIPELINE_CREATION_FEEDBACK_BASE_PIPELINE_ACCELERATION_BIT
  };
  using PipelineCreationFeedbackFlagBitsEXT = PipelineCreationFeedbackFlagBits;

  using PipelineCreationFeedbackFlags    = Flags<PipelineCreationFeedbackFlagBits>;
  using PipelineCreationFeedbackFlagsEXT = PipelineCreationFeedbackFlags;

  template <>
  struct FlagTraits<PipelineCreationFeedbackFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                          isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineCreationFeedbackFlags allFlags  = PipelineCreationFeedbackFlagBits::eValid |
                                                                                  PipelineCreationFeedbackFlagBits::eApplicationPipelineCacheHit |
                                                                                  PipelineCreationFeedbackFlagBits::eBasePipelineAcceleration;
  };

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

  using ToolPurposeFlags    = Flags<ToolPurposeFlagBits>;
  using ToolPurposeFlagsEXT = ToolPurposeFlags;

  template <>
  struct FlagTraits<ToolPurposeFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ToolPurposeFlags allFlags =
      ToolPurposeFlagBits::eValidation | ToolPurposeFlagBits::eProfiling | ToolPurposeFlagBits::eTracing | ToolPurposeFlagBits::eAdditionalFeatures |
      ToolPurposeFlagBits::eModifyingFeatures | ToolPurposeFlagBits::eDebugReportingEXT | ToolPurposeFlagBits::eDebugMarkersEXT;
  };

  enum class PrivateDataSlotCreateFlagBits : VkPrivateDataSlotCreateFlags
  {
  };
  using PrivateDataSlotCreateFlagBitsEXT = PrivateDataSlotCreateFlagBits;

  using PrivateDataSlotCreateFlags    = Flags<PrivateDataSlotCreateFlagBits>;
  using PrivateDataSlotCreateFlagsEXT = PrivateDataSlotCreateFlags;

  template <>
  struct FlagTraits<PrivateDataSlotCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PrivateDataSlotCreateFlags allFlags  = {};
  };

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
    eTransfer                     = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
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
    eVideoDecodeKHR               = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoEncodeKHR = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eTransformFeedbackEXT             = VK_PIPELINE_STAGE_2_TRANSFORM_FEEDBACK_BIT_EXT,
    eConditionalRenderingEXT          = VK_PIPELINE_STAGE_2_CONDITIONAL_RENDERING_BIT_EXT,
    eCommandPreprocessNV              = VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_NV,
    eFragmentShadingRateAttachmentKHR = VK_PIPELINE_STAGE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
    eShadingRateImageNV               = VK_PIPELINE_STAGE_2_SHADING_RATE_IMAGE_BIT_NV,
    eAccelerationStructureBuildKHR    = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    eRayTracingShaderKHR              = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
    eRayTracingShaderNV               = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_NV,
    eAccelerationStructureBuildNV     = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
    eFragmentDensityProcessEXT        = VK_PIPELINE_STAGE_2_FRAGMENT_DENSITY_PROCESS_BIT_EXT,
    eTaskShaderNV                     = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_NV,
    eMeshShaderNV                     = VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_NV,
    eTaskShaderEXT                    = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
    eMeshShaderEXT                    = VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
    eSubpassShaderHUAWEI              = VK_PIPELINE_STAGE_2_SUBPASS_SHADER_BIT_HUAWEI,
    eSubpassShadingHUAWEI             = VK_PIPELINE_STAGE_2_SUBPASS_SHADING_BIT_HUAWEI,
    eInvocationMaskHUAWEI             = VK_PIPELINE_STAGE_2_INVOCATION_MASK_BIT_HUAWEI,
    eAccelerationStructureCopyKHR     = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR,
    eMicromapBuildEXT                 = VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT,
    eClusterCullingShaderHUAWEI       = VK_PIPELINE_STAGE_2_CLUSTER_CULLING_SHADER_BIT_HUAWEI,
    eOpticalFlowNV                    = VK_PIPELINE_STAGE_2_OPTICAL_FLOW_BIT_NV
  };
  using PipelineStageFlagBits2KHR = PipelineStageFlagBits2;

  using PipelineStageFlags2    = Flags<PipelineStageFlagBits2>;
  using PipelineStageFlags2KHR = PipelineStageFlags2;

  template <>
  struct FlagTraits<PipelineStageFlagBits2>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineStageFlags2 allFlags =
      PipelineStageFlagBits2::eNone | PipelineStageFlagBits2::eTopOfPipe | PipelineStageFlagBits2::eDrawIndirect | PipelineStageFlagBits2::eVertexInput |
      PipelineStageFlagBits2::eVertexShader | PipelineStageFlagBits2::eTessellationControlShader | PipelineStageFlagBits2::eTessellationEvaluationShader |
      PipelineStageFlagBits2::eGeometryShader | PipelineStageFlagBits2::eFragmentShader | PipelineStageFlagBits2::eEarlyFragmentTests |
      PipelineStageFlagBits2::eLateFragmentTests | PipelineStageFlagBits2::eColorAttachmentOutput | PipelineStageFlagBits2::eComputeShader |
      PipelineStageFlagBits2::eAllTransfer | PipelineStageFlagBits2::eBottomOfPipe | PipelineStageFlagBits2::eHost | PipelineStageFlagBits2::eAllGraphics |
      PipelineStageFlagBits2::eAllCommands | PipelineStageFlagBits2::eCopy | PipelineStageFlagBits2::eResolve | PipelineStageFlagBits2::eBlit |
      PipelineStageFlagBits2::eClear | PipelineStageFlagBits2::eIndexInput | PipelineStageFlagBits2::eVertexAttributeInput |
      PipelineStageFlagBits2::ePreRasterizationShaders | PipelineStageFlagBits2::eVideoDecodeKHR
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      | PipelineStageFlagBits2::eVideoEncodeKHR
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      | PipelineStageFlagBits2::eTransformFeedbackEXT | PipelineStageFlagBits2::eConditionalRenderingEXT | PipelineStageFlagBits2::eCommandPreprocessNV |
      PipelineStageFlagBits2::eFragmentShadingRateAttachmentKHR | PipelineStageFlagBits2::eAccelerationStructureBuildKHR |
      PipelineStageFlagBits2::eRayTracingShaderKHR | PipelineStageFlagBits2::eFragmentDensityProcessEXT | PipelineStageFlagBits2::eTaskShaderEXT |
      PipelineStageFlagBits2::eMeshShaderEXT | PipelineStageFlagBits2::eSubpassShaderHUAWEI | PipelineStageFlagBits2::eInvocationMaskHUAWEI |
      PipelineStageFlagBits2::eAccelerationStructureCopyKHR | PipelineStageFlagBits2::eMicromapBuildEXT | PipelineStageFlagBits2::eClusterCullingShaderHUAWEI |
      PipelineStageFlagBits2::eOpticalFlowNV;
  };

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
    eVideoDecodeReadKHR          = VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR,
    eVideoDecodeWriteKHR         = VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
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
    eShadingRateImageReadNV               = VK_ACCESS_2_SHADING_RATE_IMAGE_READ_BIT_NV,
    eAccelerationStructureReadKHR         = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    eAccelerationStructureWriteKHR        = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
    eAccelerationStructureReadNV          = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_NV,
    eAccelerationStructureWriteNV         = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_NV,
    eFragmentDensityMapReadEXT            = VK_ACCESS_2_FRAGMENT_DENSITY_MAP_READ_BIT_EXT,
    eColorAttachmentReadNoncoherentEXT    = VK_ACCESS_2_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT,
    eDescriptorBufferReadEXT              = VK_ACCESS_2_DESCRIPTOR_BUFFER_READ_BIT_EXT,
    eInvocationMaskReadHUAWEI             = VK_ACCESS_2_INVOCATION_MASK_READ_BIT_HUAWEI,
    eShaderBindingTableReadKHR            = VK_ACCESS_2_SHADER_BINDING_TABLE_READ_BIT_KHR,
    eMicromapReadEXT                      = VK_ACCESS_2_MICROMAP_READ_BIT_EXT,
    eMicromapWriteEXT                     = VK_ACCESS_2_MICROMAP_WRITE_BIT_EXT,
    eOpticalFlowReadNV                    = VK_ACCESS_2_OPTICAL_FLOW_READ_BIT_NV,
    eOpticalFlowWriteNV                   = VK_ACCESS_2_OPTICAL_FLOW_WRITE_BIT_NV
  };
  using AccessFlagBits2KHR = AccessFlagBits2;

  using AccessFlags2    = Flags<AccessFlagBits2>;
  using AccessFlags2KHR = AccessFlags2;

  template <>
  struct FlagTraits<AccessFlagBits2>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool         isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR AccessFlags2 allFlags =
      AccessFlagBits2::eNone | AccessFlagBits2::eIndirectCommandRead | AccessFlagBits2::eIndexRead | AccessFlagBits2::eVertexAttributeRead |
      AccessFlagBits2::eUniformRead | AccessFlagBits2::eInputAttachmentRead | AccessFlagBits2::eShaderRead | AccessFlagBits2::eShaderWrite |
      AccessFlagBits2::eColorAttachmentRead | AccessFlagBits2::eColorAttachmentWrite | AccessFlagBits2::eDepthStencilAttachmentRead |
      AccessFlagBits2::eDepthStencilAttachmentWrite | AccessFlagBits2::eTransferRead | AccessFlagBits2::eTransferWrite | AccessFlagBits2::eHostRead |
      AccessFlagBits2::eHostWrite | AccessFlagBits2::eMemoryRead | AccessFlagBits2::eMemoryWrite | AccessFlagBits2::eShaderSampledRead |
      AccessFlagBits2::eShaderStorageRead | AccessFlagBits2::eShaderStorageWrite | AccessFlagBits2::eVideoDecodeReadKHR | AccessFlagBits2::eVideoDecodeWriteKHR
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      | AccessFlagBits2::eVideoEncodeReadKHR | AccessFlagBits2::eVideoEncodeWriteKHR
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      | AccessFlagBits2::eTransformFeedbackWriteEXT | AccessFlagBits2::eTransformFeedbackCounterReadEXT | AccessFlagBits2::eTransformFeedbackCounterWriteEXT |
      AccessFlagBits2::eConditionalRenderingReadEXT | AccessFlagBits2::eCommandPreprocessReadNV | AccessFlagBits2::eCommandPreprocessWriteNV |
      AccessFlagBits2::eFragmentShadingRateAttachmentReadKHR | AccessFlagBits2::eAccelerationStructureReadKHR |
      AccessFlagBits2::eAccelerationStructureWriteKHR | AccessFlagBits2::eFragmentDensityMapReadEXT | AccessFlagBits2::eColorAttachmentReadNoncoherentEXT |
      AccessFlagBits2::eDescriptorBufferReadEXT | AccessFlagBits2::eInvocationMaskReadHUAWEI | AccessFlagBits2::eShaderBindingTableReadKHR |
      AccessFlagBits2::eMicromapReadEXT | AccessFlagBits2::eMicromapWriteEXT | AccessFlagBits2::eOpticalFlowReadNV | AccessFlagBits2::eOpticalFlowWriteNV;
  };

  enum class SubmitFlagBits : VkSubmitFlags
  {
    eProtected = VK_SUBMIT_PROTECTED_BIT
  };
  using SubmitFlagBitsKHR = SubmitFlagBits;

  using SubmitFlags    = Flags<SubmitFlagBits>;
  using SubmitFlagsKHR = SubmitFlags;

  template <>
  struct FlagTraits<SubmitFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool        isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SubmitFlags allFlags  = SubmitFlagBits::eProtected;
  };

  enum class RenderingFlagBits : VkRenderingFlags
  {
    eContentsSecondaryCommandBuffers = VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT,
    eSuspending                      = VK_RENDERING_SUSPENDING_BIT,
    eResuming                        = VK_RENDERING_RESUMING_BIT,
    eContentsInlineEXT               = VK_RENDERING_CONTENTS_INLINE_BIT_EXT,
    eEnableLegacyDitheringEXT        = VK_RENDERING_ENABLE_LEGACY_DITHERING_BIT_EXT
  };
  using RenderingFlagBitsKHR = RenderingFlagBits;

  using RenderingFlags    = Flags<RenderingFlagBits>;
  using RenderingFlagsKHR = RenderingFlags;

  template <>
  struct FlagTraits<RenderingFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR RenderingFlags allFlags  = RenderingFlagBits::eContentsSecondaryCommandBuffers | RenderingFlagBits::eSuspending |
                                                                   RenderingFlagBits::eResuming | RenderingFlagBits::eContentsInlineEXT |
                                                                   RenderingFlagBits::eEnableLegacyDitheringEXT;
  };

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
    eSampledImageFilterCubicEXT                              = VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_CUBIC_BIT_EXT,
    eTransferSrc                                             = VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT,
    eTransferDst                                             = VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT,
    eSampledImageFilterMinmax                                = VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_MINMAX_BIT,
    eMidpointChromaSamples                                   = VK_FORMAT_FEATURE_2_MIDPOINT_CHROMA_SAMPLES_BIT,
    eSampledImageYcbcrConversionLinearFilter                 = VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT,
    eSampledImageYcbcrConversionSeparateReconstructionFilter = VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT,
    eSampledImageYcbcrConversionChromaReconstructionExplicit = VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_BIT,
    eSampledImageYcbcrConversionChromaReconstructionExplicitForceable =
      VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE_BIT,
    eDisjoint                             = VK_FORMAT_FEATURE_2_DISJOINT_BIT,
    eCositedChromaSamples                 = VK_FORMAT_FEATURE_2_COSITED_CHROMA_SAMPLES_BIT,
    eStorageReadWithoutFormat             = VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT,
    eStorageWriteWithoutFormat            = VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT,
    eSampledImageDepthComparison          = VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT,
    eVideoDecodeOutputKHR                 = VK_FORMAT_FEATURE_2_VIDEO_DECODE_OUTPUT_BIT_KHR,
    eVideoDecodeDpbKHR                    = VK_FORMAT_FEATURE_2_VIDEO_DECODE_DPB_BIT_KHR,
    eAccelerationStructureVertexBufferKHR = VK_FORMAT_FEATURE_2_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR,
    eFragmentDensityMapEXT                = VK_FORMAT_FEATURE_2_FRAGMENT_DENSITY_MAP_BIT_EXT,
    eFragmentShadingRateAttachmentKHR     = VK_FORMAT_FEATURE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
    eHostImageTransferEXT                 = VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoEncodeInputKHR = VK_FORMAT_FEATURE_2_VIDEO_ENCODE_INPUT_BIT_KHR,
    eVideoEncodeDpbKHR   = VK_FORMAT_FEATURE_2_VIDEO_ENCODE_DPB_BIT_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eLinearColorAttachmentNV = VK_FORMAT_FEATURE_2_LINEAR_COLOR_ATTACHMENT_BIT_NV,
    eWeightImageQCOM         = VK_FORMAT_FEATURE_2_WEIGHT_IMAGE_BIT_QCOM,
    eWeightSampledImageQCOM  = VK_FORMAT_FEATURE_2_WEIGHT_SAMPLED_IMAGE_BIT_QCOM,
    eBlockMatchingQCOM       = VK_FORMAT_FEATURE_2_BLOCK_MATCHING_BIT_QCOM,
    eBoxFilterSampledQCOM    = VK_FORMAT_FEATURE_2_BOX_FILTER_SAMPLED_BIT_QCOM,
    eOpticalFlowImageNV      = VK_FORMAT_FEATURE_2_OPTICAL_FLOW_IMAGE_BIT_NV,
    eOpticalFlowVectorNV     = VK_FORMAT_FEATURE_2_OPTICAL_FLOW_VECTOR_BIT_NV,
    eOpticalFlowCostNV       = VK_FORMAT_FEATURE_2_OPTICAL_FLOW_COST_BIT_NV
  };
  using FormatFeatureFlagBits2KHR = FormatFeatureFlagBits2;

  using FormatFeatureFlags2    = Flags<FormatFeatureFlagBits2>;
  using FormatFeatureFlags2KHR = FormatFeatureFlags2;

  template <>
  struct FlagTraits<FormatFeatureFlagBits2>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR FormatFeatureFlags2 allFlags =
      FormatFeatureFlagBits2::eSampledImage | FormatFeatureFlagBits2::eStorageImage | FormatFeatureFlagBits2::eStorageImageAtomic |
      FormatFeatureFlagBits2::eUniformTexelBuffer | FormatFeatureFlagBits2::eStorageTexelBuffer | FormatFeatureFlagBits2::eStorageTexelBufferAtomic |
      FormatFeatureFlagBits2::eVertexBuffer | FormatFeatureFlagBits2::eColorAttachment | FormatFeatureFlagBits2::eColorAttachmentBlend |
      FormatFeatureFlagBits2::eDepthStencilAttachment | FormatFeatureFlagBits2::eBlitSrc | FormatFeatureFlagBits2::eBlitDst |
      FormatFeatureFlagBits2::eSampledImageFilterLinear | FormatFeatureFlagBits2::eSampledImageFilterCubic | FormatFeatureFlagBits2::eTransferSrc |
      FormatFeatureFlagBits2::eTransferDst | FormatFeatureFlagBits2::eSampledImageFilterMinmax | FormatFeatureFlagBits2::eMidpointChromaSamples |
      FormatFeatureFlagBits2::eSampledImageYcbcrConversionLinearFilter | FormatFeatureFlagBits2::eSampledImageYcbcrConversionSeparateReconstructionFilter |
      FormatFeatureFlagBits2::eSampledImageYcbcrConversionChromaReconstructionExplicit |
      FormatFeatureFlagBits2::eSampledImageYcbcrConversionChromaReconstructionExplicitForceable | FormatFeatureFlagBits2::eDisjoint |
      FormatFeatureFlagBits2::eCositedChromaSamples | FormatFeatureFlagBits2::eStorageReadWithoutFormat | FormatFeatureFlagBits2::eStorageWriteWithoutFormat |
      FormatFeatureFlagBits2::eSampledImageDepthComparison | FormatFeatureFlagBits2::eVideoDecodeOutputKHR | FormatFeatureFlagBits2::eVideoDecodeDpbKHR |
      FormatFeatureFlagBits2::eAccelerationStructureVertexBufferKHR | FormatFeatureFlagBits2::eFragmentDensityMapEXT |
      FormatFeatureFlagBits2::eFragmentShadingRateAttachmentKHR | FormatFeatureFlagBits2::eHostImageTransferEXT
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      | FormatFeatureFlagBits2::eVideoEncodeInputKHR | FormatFeatureFlagBits2::eVideoEncodeDpbKHR
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      | FormatFeatureFlagBits2::eLinearColorAttachmentNV | FormatFeatureFlagBits2::eWeightImageQCOM | FormatFeatureFlagBits2::eWeightSampledImageQCOM |
      FormatFeatureFlagBits2::eBlockMatchingQCOM | FormatFeatureFlagBits2::eBoxFilterSampledQCOM | FormatFeatureFlagBits2::eOpticalFlowImageNV |
      FormatFeatureFlagBits2::eOpticalFlowVectorNV | FormatFeatureFlagBits2::eOpticalFlowCostNV;
  };

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

  using SurfaceTransformFlagsKHR = Flags<SurfaceTransformFlagBitsKHR>;

  template <>
  struct FlagTraits<SurfaceTransformFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                     isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SurfaceTransformFlagsKHR allFlags =
      SurfaceTransformFlagBitsKHR::eIdentity | SurfaceTransformFlagBitsKHR::eRotate90 | SurfaceTransformFlagBitsKHR::eRotate180 |
      SurfaceTransformFlagBitsKHR::eRotate270 | SurfaceTransformFlagBitsKHR::eHorizontalMirror | SurfaceTransformFlagBitsKHR::eHorizontalMirrorRotate90 |
      SurfaceTransformFlagBitsKHR::eHorizontalMirrorRotate180 | SurfaceTransformFlagBitsKHR::eHorizontalMirrorRotate270 | SurfaceTransformFlagBitsKHR::eInherit;
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
    eVkColorspaceSrgbNonlinear = VK_COLORSPACE_SRGB_NONLINEAR_KHR,
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
    eDciP3LinearEXT            = VK_COLOR_SPACE_DCI_P3_LINEAR_EXT,
    eDisplayNativeAMD          = VK_COLOR_SPACE_DISPLAY_NATIVE_AMD
  };

  enum class CompositeAlphaFlagBitsKHR : VkCompositeAlphaFlagsKHR
  {
    eOpaque         = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
    ePreMultiplied  = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
    ePostMultiplied = VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
    eInherit        = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR
  };

  using CompositeAlphaFlagsKHR = Flags<CompositeAlphaFlagBitsKHR>;

  template <>
  struct FlagTraits<CompositeAlphaFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR CompositeAlphaFlagsKHR allFlags  = CompositeAlphaFlagBitsKHR::eOpaque | CompositeAlphaFlagBitsKHR::ePreMultiplied |
                                                                           CompositeAlphaFlagBitsKHR::ePostMultiplied | CompositeAlphaFlagBitsKHR::eInherit;
  };

  //=== VK_KHR_swapchain ===

  enum class SwapchainCreateFlagBitsKHR : VkSwapchainCreateFlagsKHR
  {
    eSplitInstanceBindRegions    = VK_SWAPCHAIN_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT_KHR,
    eProtected                   = VK_SWAPCHAIN_CREATE_PROTECTED_BIT_KHR,
    eMutableFormat               = VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR,
    eDeferredMemoryAllocationEXT = VK_SWAPCHAIN_CREATE_DEFERRED_MEMORY_ALLOCATION_BIT_EXT
  };

  using SwapchainCreateFlagsKHR = Flags<SwapchainCreateFlagBitsKHR>;

  template <>
  struct FlagTraits<SwapchainCreateFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                    isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SwapchainCreateFlagsKHR allFlags =
      SwapchainCreateFlagBitsKHR::eSplitInstanceBindRegions | SwapchainCreateFlagBitsKHR::eProtected | SwapchainCreateFlagBitsKHR::eMutableFormat |
      SwapchainCreateFlagBitsKHR::eDeferredMemoryAllocationEXT;
  };

  enum class DeviceGroupPresentModeFlagBitsKHR : VkDeviceGroupPresentModeFlagsKHR
  {
    eLocal            = VK_DEVICE_GROUP_PRESENT_MODE_LOCAL_BIT_KHR,
    eRemote           = VK_DEVICE_GROUP_PRESENT_MODE_REMOTE_BIT_KHR,
    eSum              = VK_DEVICE_GROUP_PRESENT_MODE_SUM_BIT_KHR,
    eLocalMultiDevice = VK_DEVICE_GROUP_PRESENT_MODE_LOCAL_MULTI_DEVICE_BIT_KHR
  };

  using DeviceGroupPresentModeFlagsKHR = Flags<DeviceGroupPresentModeFlagBitsKHR>;

  template <>
  struct FlagTraits<DeviceGroupPresentModeFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DeviceGroupPresentModeFlagsKHR allFlags =
      DeviceGroupPresentModeFlagBitsKHR::eLocal | DeviceGroupPresentModeFlagBitsKHR::eRemote | DeviceGroupPresentModeFlagBitsKHR::eSum |
      DeviceGroupPresentModeFlagBitsKHR::eLocalMultiDevice;
  };

  //=== VK_KHR_display ===

  enum class DisplayPlaneAlphaFlagBitsKHR : VkDisplayPlaneAlphaFlagsKHR
  {
    eOpaque                = VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR,
    eGlobal                = VK_DISPLAY_PLANE_ALPHA_GLOBAL_BIT_KHR,
    ePerPixel              = VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_BIT_KHR,
    ePerPixelPremultiplied = VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_PREMULTIPLIED_BIT_KHR
  };

  using DisplayPlaneAlphaFlagsKHR = Flags<DisplayPlaneAlphaFlagBitsKHR>;

  template <>
  struct FlagTraits<DisplayPlaneAlphaFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                      isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DisplayPlaneAlphaFlagsKHR allFlags  = DisplayPlaneAlphaFlagBitsKHR::eOpaque | DisplayPlaneAlphaFlagBitsKHR::eGlobal |
                                                                              DisplayPlaneAlphaFlagBitsKHR::ePerPixel |
                                                                              DisplayPlaneAlphaFlagBitsKHR::ePerPixelPremultiplied;
  };

  enum class DisplayModeCreateFlagBitsKHR : VkDisplayModeCreateFlagsKHR
  {
  };

  using DisplayModeCreateFlagsKHR = Flags<DisplayModeCreateFlagBitsKHR>;

  template <>
  struct FlagTraits<DisplayModeCreateFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                      isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DisplayModeCreateFlagsKHR allFlags  = {};
  };

  enum class DisplaySurfaceCreateFlagBitsKHR : VkDisplaySurfaceCreateFlagsKHR
  {
  };

  using DisplaySurfaceCreateFlagsKHR = Flags<DisplaySurfaceCreateFlagBitsKHR>;

  template <>
  struct FlagTraits<DisplaySurfaceCreateFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                         isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DisplaySurfaceCreateFlagsKHR allFlags  = {};
  };

#if defined( VK_USE_PLATFORM_XLIB_KHR )
  //=== VK_KHR_xlib_surface ===

  enum class XlibSurfaceCreateFlagBitsKHR : VkXlibSurfaceCreateFlagsKHR
  {
  };

  using XlibSurfaceCreateFlagsKHR = Flags<XlibSurfaceCreateFlagBitsKHR>;

  template <>
  struct FlagTraits<XlibSurfaceCreateFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                      isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR XlibSurfaceCreateFlagsKHR allFlags  = {};
  };
#endif /*VK_USE_PLATFORM_XLIB_KHR*/

#if defined( VK_USE_PLATFORM_XCB_KHR )
  //=== VK_KHR_xcb_surface ===

  enum class XcbSurfaceCreateFlagBitsKHR : VkXcbSurfaceCreateFlagsKHR
  {
  };

  using XcbSurfaceCreateFlagsKHR = Flags<XcbSurfaceCreateFlagBitsKHR>;

  template <>
  struct FlagTraits<XcbSurfaceCreateFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                     isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR XcbSurfaceCreateFlagsKHR allFlags  = {};
  };
#endif /*VK_USE_PLATFORM_XCB_KHR*/

#if defined( VK_USE_PLATFORM_WAYLAND_KHR )
  //=== VK_KHR_wayland_surface ===

  enum class WaylandSurfaceCreateFlagBitsKHR : VkWaylandSurfaceCreateFlagsKHR
  {
  };

  using WaylandSurfaceCreateFlagsKHR = Flags<WaylandSurfaceCreateFlagBitsKHR>;

  template <>
  struct FlagTraits<WaylandSurfaceCreateFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                         isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR WaylandSurfaceCreateFlagsKHR allFlags  = {};
  };
#endif /*VK_USE_PLATFORM_WAYLAND_KHR*/

#if defined( VK_USE_PLATFORM_ANDROID_KHR )
  //=== VK_KHR_android_surface ===

  enum class AndroidSurfaceCreateFlagBitsKHR : VkAndroidSurfaceCreateFlagsKHR
  {
  };

  using AndroidSurfaceCreateFlagsKHR = Flags<AndroidSurfaceCreateFlagBitsKHR>;

  template <>
  struct FlagTraits<AndroidSurfaceCreateFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                         isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR AndroidSurfaceCreateFlagsKHR allFlags  = {};
  };
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_KHR_win32_surface ===

  enum class Win32SurfaceCreateFlagBitsKHR : VkWin32SurfaceCreateFlagsKHR
  {
  };

  using Win32SurfaceCreateFlagsKHR = Flags<Win32SurfaceCreateFlagBitsKHR>;

  template <>
  struct FlagTraits<Win32SurfaceCreateFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR Win32SurfaceCreateFlagsKHR allFlags  = {};
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

  using DebugReportFlagsEXT = Flags<DebugReportFlagBitsEXT>;

  template <>
  struct FlagTraits<DebugReportFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DebugReportFlagsEXT allFlags  = DebugReportFlagBitsEXT::eInformation | DebugReportFlagBitsEXT::eWarning |
                                                                        DebugReportFlagBitsEXT::ePerformanceWarning | DebugReportFlagBitsEXT::eError |
                                                                        DebugReportFlagBitsEXT::eDebug;
  };

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
    eDebugReport                 = VK_DEBUG_REPORT_OBJECT_TYPE_DEBUG_REPORT_EXT,
    eDisplayKHR                  = VK_DEBUG_REPORT_OBJECT_TYPE_DISPLAY_KHR_EXT,
    eDisplayModeKHR              = VK_DEBUG_REPORT_OBJECT_TYPE_DISPLAY_MODE_KHR_EXT,
    eValidationCacheEXT          = VK_DEBUG_REPORT_OBJECT_TYPE_VALIDATION_CACHE_EXT_EXT,
    eValidationCache             = VK_DEBUG_REPORT_OBJECT_TYPE_VALIDATION_CACHE_EXT,
    eSamplerYcbcrConversion      = VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION_EXT,
    eDescriptorUpdateTemplate    = VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_EXT,
    eCuModuleNVX                 = VK_DEBUG_REPORT_OBJECT_TYPE_CU_MODULE_NVX_EXT,
    eCuFunctionNVX               = VK_DEBUG_REPORT_OBJECT_TYPE_CU_FUNCTION_NVX_EXT,
    eDescriptorUpdateTemplateKHR = VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_KHR_EXT,
    eAccelerationStructureKHR    = VK_DEBUG_REPORT_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR_EXT,
    eSamplerYcbcrConversionKHR   = VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION_KHR_EXT,
    eAccelerationStructureNV     = VK_DEBUG_REPORT_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV_EXT,
#if defined( VK_USE_PLATFORM_FUCHSIA )
    eBufferCollectionFUCHSIA = VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_COLLECTION_FUCHSIA_EXT
#endif /*VK_USE_PLATFORM_FUCHSIA*/
  };

  //=== VK_AMD_rasterization_order ===

  enum class RasterizationOrderAMD
  {
    eStrict  = VK_RASTERIZATION_ORDER_STRICT_AMD,
    eRelaxed = VK_RASTERIZATION_ORDER_RELAXED_AMD
  };

  //=== VK_KHR_video_queue ===

  enum class VideoCodecOperationFlagBitsKHR : VkVideoCodecOperationFlagsKHR
  {
    eNone = VK_VIDEO_CODEC_OPERATION_NONE_KHR,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eEncodeH264EXT = VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_EXT,
    eEncodeH265EXT = VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_EXT,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eDecodeH264 = VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR,
    eDecodeH265 = VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR
  };

  using VideoCodecOperationFlagsKHR = Flags<VideoCodecOperationFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoCodecOperationFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                        isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoCodecOperationFlagsKHR allFlags =
      VideoCodecOperationFlagBitsKHR::eNone
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      | VideoCodecOperationFlagBitsKHR::eEncodeH264EXT | VideoCodecOperationFlagBitsKHR::eEncodeH265EXT
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      | VideoCodecOperationFlagBitsKHR::eDecodeH264 | VideoCodecOperationFlagBitsKHR::eDecodeH265;
  };

  enum class VideoChromaSubsamplingFlagBitsKHR : VkVideoChromaSubsamplingFlagsKHR
  {
    eInvalid    = VK_VIDEO_CHROMA_SUBSAMPLING_INVALID_KHR,
    eMonochrome = VK_VIDEO_CHROMA_SUBSAMPLING_MONOCHROME_BIT_KHR,
    e420        = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR,
    e422        = VK_VIDEO_CHROMA_SUBSAMPLING_422_BIT_KHR,
    e444        = VK_VIDEO_CHROMA_SUBSAMPLING_444_BIT_KHR
  };

  using VideoChromaSubsamplingFlagsKHR = Flags<VideoChromaSubsamplingFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoChromaSubsamplingFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoChromaSubsamplingFlagsKHR allFlags =
      VideoChromaSubsamplingFlagBitsKHR::eInvalid | VideoChromaSubsamplingFlagBitsKHR::eMonochrome | VideoChromaSubsamplingFlagBitsKHR::e420 |
      VideoChromaSubsamplingFlagBitsKHR::e422 | VideoChromaSubsamplingFlagBitsKHR::e444;
  };

  enum class VideoComponentBitDepthFlagBitsKHR : VkVideoComponentBitDepthFlagsKHR
  {
    eInvalid = VK_VIDEO_COMPONENT_BIT_DEPTH_INVALID_KHR,
    e8       = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
    e10      = VK_VIDEO_COMPONENT_BIT_DEPTH_10_BIT_KHR,
    e12      = VK_VIDEO_COMPONENT_BIT_DEPTH_12_BIT_KHR
  };

  using VideoComponentBitDepthFlagsKHR = Flags<VideoComponentBitDepthFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoComponentBitDepthFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoComponentBitDepthFlagsKHR allFlags =
      VideoComponentBitDepthFlagBitsKHR::eInvalid | VideoComponentBitDepthFlagBitsKHR::e8 | VideoComponentBitDepthFlagBitsKHR::e10 |
      VideoComponentBitDepthFlagBitsKHR::e12;
  };

  enum class VideoCapabilityFlagBitsKHR : VkVideoCapabilityFlagsKHR
  {
    eProtectedContent        = VK_VIDEO_CAPABILITY_PROTECTED_CONTENT_BIT_KHR,
    eSeparateReferenceImages = VK_VIDEO_CAPABILITY_SEPARATE_REFERENCE_IMAGES_BIT_KHR
  };

  using VideoCapabilityFlagsKHR = Flags<VideoCapabilityFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoCapabilityFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                    isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoCapabilityFlagsKHR allFlags =
      VideoCapabilityFlagBitsKHR::eProtectedContent | VideoCapabilityFlagBitsKHR::eSeparateReferenceImages;
  };

  enum class VideoSessionCreateFlagBitsKHR : VkVideoSessionCreateFlagsKHR
  {
    eProtectedContent = VK_VIDEO_SESSION_CREATE_PROTECTED_CONTENT_BIT_KHR,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eAllowEncodeParameterOptimizations = VK_VIDEO_SESSION_CREATE_ALLOW_ENCODE_PARAMETER_OPTIMIZATIONS_BIT_KHR
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
  };

  using VideoSessionCreateFlagsKHR = Flags<VideoSessionCreateFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoSessionCreateFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoSessionCreateFlagsKHR allFlags  = VideoSessionCreateFlagBitsKHR::eProtectedContent
#if defined( VK_ENABLE_BETA_EXTENSIONS )
                                                                               | VideoSessionCreateFlagBitsKHR::eAllowEncodeParameterOptimizations
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      ;
  };

  enum class VideoCodingControlFlagBitsKHR : VkVideoCodingControlFlagsKHR
  {
    eReset = VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eEncodeRateControl  = VK_VIDEO_CODING_CONTROL_ENCODE_RATE_CONTROL_BIT_KHR,
    eEncodeQualityLevel = VK_VIDEO_CODING_CONTROL_ENCODE_QUALITY_LEVEL_BIT_KHR
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
  };

  using VideoCodingControlFlagsKHR = Flags<VideoCodingControlFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoCodingControlFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoCodingControlFlagsKHR allFlags  = VideoCodingControlFlagBitsKHR::eReset
#if defined( VK_ENABLE_BETA_EXTENSIONS )
                                                                               | VideoCodingControlFlagBitsKHR::eEncodeRateControl |
                                                                               VideoCodingControlFlagBitsKHR::eEncodeQualityLevel
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      ;
  };

  enum class QueryResultStatusKHR
  {
    eError    = VK_QUERY_RESULT_STATUS_ERROR_KHR,
    eNotReady = VK_QUERY_RESULT_STATUS_NOT_READY_KHR,
    eComplete = VK_QUERY_RESULT_STATUS_COMPLETE_KHR,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eInsufficientBitstreamBufferRange = VK_QUERY_RESULT_STATUS_INSUFFICIENT_BITSTREAM_BUFFER_RANGE_KHR
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
  };

  enum class VideoSessionParametersCreateFlagBitsKHR : VkVideoSessionParametersCreateFlagsKHR
  {
  };

  using VideoSessionParametersCreateFlagsKHR = Flags<VideoSessionParametersCreateFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoSessionParametersCreateFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoSessionParametersCreateFlagsKHR allFlags  = {};
  };

  enum class VideoBeginCodingFlagBitsKHR : VkVideoBeginCodingFlagsKHR
  {
  };

  using VideoBeginCodingFlagsKHR = Flags<VideoBeginCodingFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoBeginCodingFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                     isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoBeginCodingFlagsKHR allFlags  = {};
  };

  enum class VideoEndCodingFlagBitsKHR : VkVideoEndCodingFlagsKHR
  {
  };

  using VideoEndCodingFlagsKHR = Flags<VideoEndCodingFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEndCodingFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEndCodingFlagsKHR allFlags  = {};
  };

  //=== VK_KHR_video_decode_queue ===

  enum class VideoDecodeCapabilityFlagBitsKHR : VkVideoDecodeCapabilityFlagsKHR
  {
    eDpbAndOutputCoincide = VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR,
    eDpbAndOutputDistinct = VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_DISTINCT_BIT_KHR
  };

  using VideoDecodeCapabilityFlagsKHR = Flags<VideoDecodeCapabilityFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoDecodeCapabilityFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                          isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoDecodeCapabilityFlagsKHR allFlags =
      VideoDecodeCapabilityFlagBitsKHR::eDpbAndOutputCoincide | VideoDecodeCapabilityFlagBitsKHR::eDpbAndOutputDistinct;
  };

  enum class VideoDecodeUsageFlagBitsKHR : VkVideoDecodeUsageFlagsKHR
  {
    eDefault     = VK_VIDEO_DECODE_USAGE_DEFAULT_KHR,
    eTranscoding = VK_VIDEO_DECODE_USAGE_TRANSCODING_BIT_KHR,
    eOffline     = VK_VIDEO_DECODE_USAGE_OFFLINE_BIT_KHR,
    eStreaming   = VK_VIDEO_DECODE_USAGE_STREAMING_BIT_KHR
  };

  using VideoDecodeUsageFlagsKHR = Flags<VideoDecodeUsageFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoDecodeUsageFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                     isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoDecodeUsageFlagsKHR allFlags = VideoDecodeUsageFlagBitsKHR::eDefault | VideoDecodeUsageFlagBitsKHR::eTranscoding |
                                                                             VideoDecodeUsageFlagBitsKHR::eOffline | VideoDecodeUsageFlagBitsKHR::eStreaming;
  };

  enum class VideoDecodeFlagBitsKHR : VkVideoDecodeFlagsKHR
  {
  };

  using VideoDecodeFlagsKHR = Flags<VideoDecodeFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoDecodeFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoDecodeFlagsKHR allFlags  = {};
  };

  //=== VK_EXT_transform_feedback ===

  enum class PipelineRasterizationStateStreamCreateFlagBitsEXT : VkPipelineRasterizationStateStreamCreateFlagsEXT
  {
  };

  using PipelineRasterizationStateStreamCreateFlagsEXT = Flags<PipelineRasterizationStateStreamCreateFlagBitsEXT>;

  template <>
  struct FlagTraits<PipelineRasterizationStateStreamCreateFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineRasterizationStateStreamCreateFlagsEXT allFlags  = {};
  };

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_EXT_video_encode_h264 ===

  enum class VideoEncodeH264CapabilityFlagBitsEXT : VkVideoEncodeH264CapabilityFlagsEXT
  {
    eHrdCompliance                  = VK_VIDEO_ENCODE_H264_CAPABILITY_HRD_COMPLIANCE_BIT_EXT,
    ePredictionWeightTableGenerated = VK_VIDEO_ENCODE_H264_CAPABILITY_PREDICTION_WEIGHT_TABLE_GENERATED_BIT_EXT,
    eRowUnalignedSlice              = VK_VIDEO_ENCODE_H264_CAPABILITY_ROW_UNALIGNED_SLICE_BIT_EXT,
    eDifferentSliceType             = VK_VIDEO_ENCODE_H264_CAPABILITY_DIFFERENT_SLICE_TYPE_BIT_EXT,
    eBFrameInL0List                 = VK_VIDEO_ENCODE_H264_CAPABILITY_B_FRAME_IN_L0_LIST_BIT_EXT,
    eBFrameInL1List                 = VK_VIDEO_ENCODE_H264_CAPABILITY_B_FRAME_IN_L1_LIST_BIT_EXT,
    ePerPictureTypeMinMaxQp         = VK_VIDEO_ENCODE_H264_CAPABILITY_PER_PICTURE_TYPE_MIN_MAX_QP_BIT_EXT,
    ePerSliceConstantQp             = VK_VIDEO_ENCODE_H264_CAPABILITY_PER_SLICE_CONSTANT_QP_BIT_EXT,
    eGeneratePrefixNalu             = VK_VIDEO_ENCODE_H264_CAPABILITY_GENERATE_PREFIX_NALU_BIT_EXT
  };

  using VideoEncodeH264CapabilityFlagsEXT = Flags<VideoEncodeH264CapabilityFlagBitsEXT>;

  template <>
  struct FlagTraits<VideoEncodeH264CapabilityFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                              isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeH264CapabilityFlagsEXT allFlags =
      VideoEncodeH264CapabilityFlagBitsEXT::eHrdCompliance | VideoEncodeH264CapabilityFlagBitsEXT::ePredictionWeightTableGenerated |
      VideoEncodeH264CapabilityFlagBitsEXT::eRowUnalignedSlice | VideoEncodeH264CapabilityFlagBitsEXT::eDifferentSliceType |
      VideoEncodeH264CapabilityFlagBitsEXT::eBFrameInL0List | VideoEncodeH264CapabilityFlagBitsEXT::eBFrameInL1List |
      VideoEncodeH264CapabilityFlagBitsEXT::ePerPictureTypeMinMaxQp | VideoEncodeH264CapabilityFlagBitsEXT::ePerSliceConstantQp |
      VideoEncodeH264CapabilityFlagBitsEXT::eGeneratePrefixNalu;
  };

  enum class VideoEncodeH264StdFlagBitsEXT : VkVideoEncodeH264StdFlagsEXT
  {
    eSeparateColorPlaneFlagSet          = VK_VIDEO_ENCODE_H264_STD_SEPARATE_COLOR_PLANE_FLAG_SET_BIT_EXT,
    eQpprimeYZeroTransformBypassFlagSet = VK_VIDEO_ENCODE_H264_STD_QPPRIME_Y_ZERO_TRANSFORM_BYPASS_FLAG_SET_BIT_EXT,
    eScalingMatrixPresentFlagSet        = VK_VIDEO_ENCODE_H264_STD_SCALING_MATRIX_PRESENT_FLAG_SET_BIT_EXT,
    eChromaQpIndexOffset                = VK_VIDEO_ENCODE_H264_STD_CHROMA_QP_INDEX_OFFSET_BIT_EXT,
    eSecondChromaQpIndexOffset          = VK_VIDEO_ENCODE_H264_STD_SECOND_CHROMA_QP_INDEX_OFFSET_BIT_EXT,
    ePicInitQpMinus26                   = VK_VIDEO_ENCODE_H264_STD_PIC_INIT_QP_MINUS26_BIT_EXT,
    eWeightedPredFlagSet                = VK_VIDEO_ENCODE_H264_STD_WEIGHTED_PRED_FLAG_SET_BIT_EXT,
    eWeightedBipredIdcExplicit          = VK_VIDEO_ENCODE_H264_STD_WEIGHTED_BIPRED_IDC_EXPLICIT_BIT_EXT,
    eWeightedBipredIdcImplicit          = VK_VIDEO_ENCODE_H264_STD_WEIGHTED_BIPRED_IDC_IMPLICIT_BIT_EXT,
    eTransform8X8ModeFlagSet            = VK_VIDEO_ENCODE_H264_STD_TRANSFORM_8X8_MODE_FLAG_SET_BIT_EXT,
    eDirectSpatialMvPredFlagUnset       = VK_VIDEO_ENCODE_H264_STD_DIRECT_SPATIAL_MV_PRED_FLAG_UNSET_BIT_EXT,
    eEntropyCodingModeFlagUnset         = VK_VIDEO_ENCODE_H264_STD_ENTROPY_CODING_MODE_FLAG_UNSET_BIT_EXT,
    eEntropyCodingModeFlagSet           = VK_VIDEO_ENCODE_H264_STD_ENTROPY_CODING_MODE_FLAG_SET_BIT_EXT,
    eDirect8X8InferenceFlagUnset        = VK_VIDEO_ENCODE_H264_STD_DIRECT_8X8_INFERENCE_FLAG_UNSET_BIT_EXT,
    eConstrainedIntraPredFlagSet        = VK_VIDEO_ENCODE_H264_STD_CONSTRAINED_INTRA_PRED_FLAG_SET_BIT_EXT,
    eDeblockingFilterDisabled           = VK_VIDEO_ENCODE_H264_STD_DEBLOCKING_FILTER_DISABLED_BIT_EXT,
    eDeblockingFilterEnabled            = VK_VIDEO_ENCODE_H264_STD_DEBLOCKING_FILTER_ENABLED_BIT_EXT,
    eDeblockingFilterPartial            = VK_VIDEO_ENCODE_H264_STD_DEBLOCKING_FILTER_PARTIAL_BIT_EXT,
    eSliceQpDelta                       = VK_VIDEO_ENCODE_H264_STD_SLICE_QP_DELTA_BIT_EXT,
    eDifferentSliceQpDelta              = VK_VIDEO_ENCODE_H264_STD_DIFFERENT_SLICE_QP_DELTA_BIT_EXT
  };

  using VideoEncodeH264StdFlagsEXT = Flags<VideoEncodeH264StdFlagBitsEXT>;

  template <>
  struct FlagTraits<VideoEncodeH264StdFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeH264StdFlagsEXT allFlags =
      VideoEncodeH264StdFlagBitsEXT::eSeparateColorPlaneFlagSet | VideoEncodeH264StdFlagBitsEXT::eQpprimeYZeroTransformBypassFlagSet |
      VideoEncodeH264StdFlagBitsEXT::eScalingMatrixPresentFlagSet | VideoEncodeH264StdFlagBitsEXT::eChromaQpIndexOffset |
      VideoEncodeH264StdFlagBitsEXT::eSecondChromaQpIndexOffset | VideoEncodeH264StdFlagBitsEXT::ePicInitQpMinus26 |
      VideoEncodeH264StdFlagBitsEXT::eWeightedPredFlagSet | VideoEncodeH264StdFlagBitsEXT::eWeightedBipredIdcExplicit |
      VideoEncodeH264StdFlagBitsEXT::eWeightedBipredIdcImplicit | VideoEncodeH264StdFlagBitsEXT::eTransform8X8ModeFlagSet |
      VideoEncodeH264StdFlagBitsEXT::eDirectSpatialMvPredFlagUnset | VideoEncodeH264StdFlagBitsEXT::eEntropyCodingModeFlagUnset |
      VideoEncodeH264StdFlagBitsEXT::eEntropyCodingModeFlagSet | VideoEncodeH264StdFlagBitsEXT::eDirect8X8InferenceFlagUnset |
      VideoEncodeH264StdFlagBitsEXT::eConstrainedIntraPredFlagSet | VideoEncodeH264StdFlagBitsEXT::eDeblockingFilterDisabled |
      VideoEncodeH264StdFlagBitsEXT::eDeblockingFilterEnabled | VideoEncodeH264StdFlagBitsEXT::eDeblockingFilterPartial |
      VideoEncodeH264StdFlagBitsEXT::eSliceQpDelta | VideoEncodeH264StdFlagBitsEXT::eDifferentSliceQpDelta;
  };

  enum class VideoEncodeH264RateControlFlagBitsEXT : VkVideoEncodeH264RateControlFlagsEXT
  {
    eAttemptHrdCompliance       = VK_VIDEO_ENCODE_H264_RATE_CONTROL_ATTEMPT_HRD_COMPLIANCE_BIT_EXT,
    eRegularGop                 = VK_VIDEO_ENCODE_H264_RATE_CONTROL_REGULAR_GOP_BIT_EXT,
    eReferencePatternFlat       = VK_VIDEO_ENCODE_H264_RATE_CONTROL_REFERENCE_PATTERN_FLAT_BIT_EXT,
    eReferencePatternDyadic     = VK_VIDEO_ENCODE_H264_RATE_CONTROL_REFERENCE_PATTERN_DYADIC_BIT_EXT,
    eTemporalLayerPatternDyadic = VK_VIDEO_ENCODE_H264_RATE_CONTROL_TEMPORAL_LAYER_PATTERN_DYADIC_BIT_EXT
  };

  using VideoEncodeH264RateControlFlagsEXT = Flags<VideoEncodeH264RateControlFlagBitsEXT>;

  template <>
  struct FlagTraits<VideoEncodeH264RateControlFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                               isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeH264RateControlFlagsEXT allFlags =
      VideoEncodeH264RateControlFlagBitsEXT::eAttemptHrdCompliance | VideoEncodeH264RateControlFlagBitsEXT::eRegularGop |
      VideoEncodeH264RateControlFlagBitsEXT::eReferencePatternFlat | VideoEncodeH264RateControlFlagBitsEXT::eReferencePatternDyadic |
      VideoEncodeH264RateControlFlagBitsEXT::eTemporalLayerPatternDyadic;
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_EXT_video_encode_h265 ===

  enum class VideoEncodeH265CapabilityFlagBitsEXT : VkVideoEncodeH265CapabilityFlagsEXT
  {
    eHrdCompliance                  = VK_VIDEO_ENCODE_H265_CAPABILITY_HRD_COMPLIANCE_BIT_EXT,
    ePredictionWeightTableGenerated = VK_VIDEO_ENCODE_H265_CAPABILITY_PREDICTION_WEIGHT_TABLE_GENERATED_BIT_EXT,
    eRowUnalignedSliceSegment       = VK_VIDEO_ENCODE_H265_CAPABILITY_ROW_UNALIGNED_SLICE_SEGMENT_BIT_EXT,
    eDifferentSliceSegmentType      = VK_VIDEO_ENCODE_H265_CAPABILITY_DIFFERENT_SLICE_SEGMENT_TYPE_BIT_EXT,
    eBFrameInL0List                 = VK_VIDEO_ENCODE_H265_CAPABILITY_B_FRAME_IN_L0_LIST_BIT_EXT,
    eBFrameInL1List                 = VK_VIDEO_ENCODE_H265_CAPABILITY_B_FRAME_IN_L1_LIST_BIT_EXT,
    ePerPictureTypeMinMaxQp         = VK_VIDEO_ENCODE_H265_CAPABILITY_PER_PICTURE_TYPE_MIN_MAX_QP_BIT_EXT,
    ePerSliceSegmentConstantQp      = VK_VIDEO_ENCODE_H265_CAPABILITY_PER_SLICE_SEGMENT_CONSTANT_QP_BIT_EXT,
    eMultipleTilesPerSliceSegment   = VK_VIDEO_ENCODE_H265_CAPABILITY_MULTIPLE_TILES_PER_SLICE_SEGMENT_BIT_EXT,
    eMultipleSliceSegmentsPerTile   = VK_VIDEO_ENCODE_H265_CAPABILITY_MULTIPLE_SLICE_SEGMENTS_PER_TILE_BIT_EXT
  };

  using VideoEncodeH265CapabilityFlagsEXT = Flags<VideoEncodeH265CapabilityFlagBitsEXT>;

  template <>
  struct FlagTraits<VideoEncodeH265CapabilityFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                              isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeH265CapabilityFlagsEXT allFlags =
      VideoEncodeH265CapabilityFlagBitsEXT::eHrdCompliance | VideoEncodeH265CapabilityFlagBitsEXT::ePredictionWeightTableGenerated |
      VideoEncodeH265CapabilityFlagBitsEXT::eRowUnalignedSliceSegment | VideoEncodeH265CapabilityFlagBitsEXT::eDifferentSliceSegmentType |
      VideoEncodeH265CapabilityFlagBitsEXT::eBFrameInL0List | VideoEncodeH265CapabilityFlagBitsEXT::eBFrameInL1List |
      VideoEncodeH265CapabilityFlagBitsEXT::ePerPictureTypeMinMaxQp | VideoEncodeH265CapabilityFlagBitsEXT::ePerSliceSegmentConstantQp |
      VideoEncodeH265CapabilityFlagBitsEXT::eMultipleTilesPerSliceSegment | VideoEncodeH265CapabilityFlagBitsEXT::eMultipleSliceSegmentsPerTile;
  };

  enum class VideoEncodeH265StdFlagBitsEXT : VkVideoEncodeH265StdFlagsEXT
  {
    eSeparateColorPlaneFlagSet              = VK_VIDEO_ENCODE_H265_STD_SEPARATE_COLOR_PLANE_FLAG_SET_BIT_EXT,
    eSampleAdaptiveOffsetEnabledFlagSet     = VK_VIDEO_ENCODE_H265_STD_SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG_SET_BIT_EXT,
    eScalingListDataPresentFlagSet          = VK_VIDEO_ENCODE_H265_STD_SCALING_LIST_DATA_PRESENT_FLAG_SET_BIT_EXT,
    ePcmEnabledFlagSet                      = VK_VIDEO_ENCODE_H265_STD_PCM_ENABLED_FLAG_SET_BIT_EXT,
    eSpsTemporalMvpEnabledFlagSet           = VK_VIDEO_ENCODE_H265_STD_SPS_TEMPORAL_MVP_ENABLED_FLAG_SET_BIT_EXT,
    eInitQpMinus26                          = VK_VIDEO_ENCODE_H265_STD_INIT_QP_MINUS26_BIT_EXT,
    eWeightedPredFlagSet                    = VK_VIDEO_ENCODE_H265_STD_WEIGHTED_PRED_FLAG_SET_BIT_EXT,
    eWeightedBipredFlagSet                  = VK_VIDEO_ENCODE_H265_STD_WEIGHTED_BIPRED_FLAG_SET_BIT_EXT,
    eLog2ParallelMergeLevelMinus2           = VK_VIDEO_ENCODE_H265_STD_LOG2_PARALLEL_MERGE_LEVEL_MINUS2_BIT_EXT,
    eSignDataHidingEnabledFlagSet           = VK_VIDEO_ENCODE_H265_STD_SIGN_DATA_HIDING_ENABLED_FLAG_SET_BIT_EXT,
    eTransformSkipEnabledFlagSet            = VK_VIDEO_ENCODE_H265_STD_TRANSFORM_SKIP_ENABLED_FLAG_SET_BIT_EXT,
    eTransformSkipEnabledFlagUnset          = VK_VIDEO_ENCODE_H265_STD_TRANSFORM_SKIP_ENABLED_FLAG_UNSET_BIT_EXT,
    ePpsSliceChromaQpOffsetsPresentFlagSet  = VK_VIDEO_ENCODE_H265_STD_PPS_SLICE_CHROMA_QP_OFFSETS_PRESENT_FLAG_SET_BIT_EXT,
    eTransquantBypassEnabledFlagSet         = VK_VIDEO_ENCODE_H265_STD_TRANSQUANT_BYPASS_ENABLED_FLAG_SET_BIT_EXT,
    eConstrainedIntraPredFlagSet            = VK_VIDEO_ENCODE_H265_STD_CONSTRAINED_INTRA_PRED_FLAG_SET_BIT_EXT,
    eEntropyCodingSyncEnabledFlagSet        = VK_VIDEO_ENCODE_H265_STD_ENTROPY_CODING_SYNC_ENABLED_FLAG_SET_BIT_EXT,
    eDeblockingFilterOverrideEnabledFlagSet = VK_VIDEO_ENCODE_H265_STD_DEBLOCKING_FILTER_OVERRIDE_ENABLED_FLAG_SET_BIT_EXT,
    eDependentSliceSegmentsEnabledFlagSet   = VK_VIDEO_ENCODE_H265_STD_DEPENDENT_SLICE_SEGMENTS_ENABLED_FLAG_SET_BIT_EXT,
    eDependentSliceSegmentFlagSet           = VK_VIDEO_ENCODE_H265_STD_DEPENDENT_SLICE_SEGMENT_FLAG_SET_BIT_EXT,
    eSliceQpDelta                           = VK_VIDEO_ENCODE_H265_STD_SLICE_QP_DELTA_BIT_EXT,
    eDifferentSliceQpDelta                  = VK_VIDEO_ENCODE_H265_STD_DIFFERENT_SLICE_QP_DELTA_BIT_EXT
  };

  using VideoEncodeH265StdFlagsEXT = Flags<VideoEncodeH265StdFlagBitsEXT>;

  template <>
  struct FlagTraits<VideoEncodeH265StdFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeH265StdFlagsEXT allFlags =
      VideoEncodeH265StdFlagBitsEXT::eSeparateColorPlaneFlagSet | VideoEncodeH265StdFlagBitsEXT::eSampleAdaptiveOffsetEnabledFlagSet |
      VideoEncodeH265StdFlagBitsEXT::eScalingListDataPresentFlagSet | VideoEncodeH265StdFlagBitsEXT::ePcmEnabledFlagSet |
      VideoEncodeH265StdFlagBitsEXT::eSpsTemporalMvpEnabledFlagSet | VideoEncodeH265StdFlagBitsEXT::eInitQpMinus26 |
      VideoEncodeH265StdFlagBitsEXT::eWeightedPredFlagSet | VideoEncodeH265StdFlagBitsEXT::eWeightedBipredFlagSet |
      VideoEncodeH265StdFlagBitsEXT::eLog2ParallelMergeLevelMinus2 | VideoEncodeH265StdFlagBitsEXT::eSignDataHidingEnabledFlagSet |
      VideoEncodeH265StdFlagBitsEXT::eTransformSkipEnabledFlagSet | VideoEncodeH265StdFlagBitsEXT::eTransformSkipEnabledFlagUnset |
      VideoEncodeH265StdFlagBitsEXT::ePpsSliceChromaQpOffsetsPresentFlagSet | VideoEncodeH265StdFlagBitsEXT::eTransquantBypassEnabledFlagSet |
      VideoEncodeH265StdFlagBitsEXT::eConstrainedIntraPredFlagSet | VideoEncodeH265StdFlagBitsEXT::eEntropyCodingSyncEnabledFlagSet |
      VideoEncodeH265StdFlagBitsEXT::eDeblockingFilterOverrideEnabledFlagSet | VideoEncodeH265StdFlagBitsEXT::eDependentSliceSegmentsEnabledFlagSet |
      VideoEncodeH265StdFlagBitsEXT::eDependentSliceSegmentFlagSet | VideoEncodeH265StdFlagBitsEXT::eSliceQpDelta |
      VideoEncodeH265StdFlagBitsEXT::eDifferentSliceQpDelta;
  };

  enum class VideoEncodeH265CtbSizeFlagBitsEXT : VkVideoEncodeH265CtbSizeFlagsEXT
  {
    e16 = VK_VIDEO_ENCODE_H265_CTB_SIZE_16_BIT_EXT,
    e32 = VK_VIDEO_ENCODE_H265_CTB_SIZE_32_BIT_EXT,
    e64 = VK_VIDEO_ENCODE_H265_CTB_SIZE_64_BIT_EXT
  };

  using VideoEncodeH265CtbSizeFlagsEXT = Flags<VideoEncodeH265CtbSizeFlagBitsEXT>;

  template <>
  struct FlagTraits<VideoEncodeH265CtbSizeFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeH265CtbSizeFlagsEXT allFlags =
      VideoEncodeH265CtbSizeFlagBitsEXT::e16 | VideoEncodeH265CtbSizeFlagBitsEXT::e32 | VideoEncodeH265CtbSizeFlagBitsEXT::e64;
  };

  enum class VideoEncodeH265TransformBlockSizeFlagBitsEXT : VkVideoEncodeH265TransformBlockSizeFlagsEXT
  {
    e4  = VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_4_BIT_EXT,
    e8  = VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_8_BIT_EXT,
    e16 = VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_16_BIT_EXT,
    e32 = VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_32_BIT_EXT
  };

  using VideoEncodeH265TransformBlockSizeFlagsEXT = Flags<VideoEncodeH265TransformBlockSizeFlagBitsEXT>;

  template <>
  struct FlagTraits<VideoEncodeH265TransformBlockSizeFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                      isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeH265TransformBlockSizeFlagsEXT allFlags =
      VideoEncodeH265TransformBlockSizeFlagBitsEXT::e4 | VideoEncodeH265TransformBlockSizeFlagBitsEXT::e8 | VideoEncodeH265TransformBlockSizeFlagBitsEXT::e16 |
      VideoEncodeH265TransformBlockSizeFlagBitsEXT::e32;
  };

  enum class VideoEncodeH265RateControlFlagBitsEXT : VkVideoEncodeH265RateControlFlagsEXT
  {
    eAttemptHrdCompliance          = VK_VIDEO_ENCODE_H265_RATE_CONTROL_ATTEMPT_HRD_COMPLIANCE_BIT_EXT,
    eRegularGop                    = VK_VIDEO_ENCODE_H265_RATE_CONTROL_REGULAR_GOP_BIT_EXT,
    eReferencePatternFlat          = VK_VIDEO_ENCODE_H265_RATE_CONTROL_REFERENCE_PATTERN_FLAT_BIT_EXT,
    eReferencePatternDyadic        = VK_VIDEO_ENCODE_H265_RATE_CONTROL_REFERENCE_PATTERN_DYADIC_BIT_EXT,
    eTemporalSubLayerPatternDyadic = VK_VIDEO_ENCODE_H265_RATE_CONTROL_TEMPORAL_SUB_LAYER_PATTERN_DYADIC_BIT_EXT
  };

  using VideoEncodeH265RateControlFlagsEXT = Flags<VideoEncodeH265RateControlFlagBitsEXT>;

  template <>
  struct FlagTraits<VideoEncodeH265RateControlFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                               isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeH265RateControlFlagsEXT allFlags =
      VideoEncodeH265RateControlFlagBitsEXT::eAttemptHrdCompliance | VideoEncodeH265RateControlFlagBitsEXT::eRegularGop |
      VideoEncodeH265RateControlFlagBitsEXT::eReferencePatternFlat | VideoEncodeH265RateControlFlagBitsEXT::eReferencePatternDyadic |
      VideoEncodeH265RateControlFlagBitsEXT::eTemporalSubLayerPatternDyadic;
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_KHR_video_decode_h264 ===

  enum class VideoDecodeH264PictureLayoutFlagBitsKHR : VkVideoDecodeH264PictureLayoutFlagsKHR
  {
    eProgressive                = VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_PROGRESSIVE_KHR,
    eInterlacedInterleavedLines = VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_INTERLACED_INTERLEAVED_LINES_BIT_KHR,
    eInterlacedSeparatePlanes   = VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_INTERLACED_SEPARATE_PLANES_BIT_KHR
  };

  using VideoDecodeH264PictureLayoutFlagsKHR = Flags<VideoDecodeH264PictureLayoutFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoDecodeH264PictureLayoutFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoDecodeH264PictureLayoutFlagsKHR allFlags  = VideoDecodeH264PictureLayoutFlagBitsKHR::eProgressive |
                                                                                         VideoDecodeH264PictureLayoutFlagBitsKHR::eInterlacedInterleavedLines |
                                                                                         VideoDecodeH264PictureLayoutFlagBitsKHR::eInterlacedSeparatePlanes;
  };

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

  using StreamDescriptorSurfaceCreateFlagsGGP = Flags<StreamDescriptorSurfaceCreateFlagBitsGGP>;

  template <>
  struct FlagTraits<StreamDescriptorSurfaceCreateFlagBitsGGP>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                  isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR StreamDescriptorSurfaceCreateFlagsGGP allFlags  = {};
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

  using ExternalMemoryHandleTypeFlagsNV = Flags<ExternalMemoryHandleTypeFlagBitsNV>;

  template <>
  struct FlagTraits<ExternalMemoryHandleTypeFlagBitsNV>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                            isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ExternalMemoryHandleTypeFlagsNV allFlags =
      ExternalMemoryHandleTypeFlagBitsNV::eOpaqueWin32 | ExternalMemoryHandleTypeFlagBitsNV::eOpaqueWin32Kmt | ExternalMemoryHandleTypeFlagBitsNV::eD3D11Image |
      ExternalMemoryHandleTypeFlagBitsNV::eD3D11ImageKmt;
  };

  enum class ExternalMemoryFeatureFlagBitsNV : VkExternalMemoryFeatureFlagsNV
  {
    eDedicatedOnly = VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT_NV,
    eExportable    = VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT_NV,
    eImportable    = VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT_NV
  };

  using ExternalMemoryFeatureFlagsNV = Flags<ExternalMemoryFeatureFlagBitsNV>;

  template <>
  struct FlagTraits<ExternalMemoryFeatureFlagBitsNV>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                         isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ExternalMemoryFeatureFlagsNV allFlags =
      ExternalMemoryFeatureFlagBitsNV::eDedicatedOnly | ExternalMemoryFeatureFlagBitsNV::eExportable | ExternalMemoryFeatureFlagBitsNV::eImportable;
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

  using ViSurfaceCreateFlagsNN = Flags<ViSurfaceCreateFlagBitsNN>;

  template <>
  struct FlagTraits<ViSurfaceCreateFlagBitsNN>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ViSurfaceCreateFlagsNN allFlags  = {};
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

  using ConditionalRenderingFlagsEXT = Flags<ConditionalRenderingFlagBitsEXT>;

  template <>
  struct FlagTraits<ConditionalRenderingFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                         isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ConditionalRenderingFlagsEXT allFlags  = ConditionalRenderingFlagBitsEXT::eInverted;
  };

  //=== VK_EXT_display_surface_counter ===

  enum class SurfaceCounterFlagBitsEXT : VkSurfaceCounterFlagsEXT
  {
    eVblank = VK_SURFACE_COUNTER_VBLANK_BIT_EXT
  };

  using SurfaceCounterFlagsEXT = Flags<SurfaceCounterFlagBitsEXT>;

  template <>
  struct FlagTraits<SurfaceCounterFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SurfaceCounterFlagsEXT allFlags  = SurfaceCounterFlagBitsEXT::eVblank;
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

  using PipelineViewportSwizzleStateCreateFlagsNV = Flags<PipelineViewportSwizzleStateCreateFlagBitsNV>;

  template <>
  struct FlagTraits<PipelineViewportSwizzleStateCreateFlagBitsNV>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                      isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineViewportSwizzleStateCreateFlagsNV allFlags  = {};
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

  using PipelineDiscardRectangleStateCreateFlagsEXT = Flags<PipelineDiscardRectangleStateCreateFlagBitsEXT>;

  template <>
  struct FlagTraits<PipelineDiscardRectangleStateCreateFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                        isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineDiscardRectangleStateCreateFlagsEXT allFlags  = {};
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

  using PipelineRasterizationConservativeStateCreateFlagsEXT = Flags<PipelineRasterizationConservativeStateCreateFlagBitsEXT>;

  template <>
  struct FlagTraits<PipelineRasterizationConservativeStateCreateFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineRasterizationConservativeStateCreateFlagsEXT allFlags  = {};
  };

  //=== VK_EXT_depth_clip_enable ===

  enum class PipelineRasterizationDepthClipStateCreateFlagBitsEXT : VkPipelineRasterizationDepthClipStateCreateFlagsEXT
  {
  };

  using PipelineRasterizationDepthClipStateCreateFlagsEXT = Flags<PipelineRasterizationDepthClipStateCreateFlagBitsEXT>;

  template <>
  struct FlagTraits<PipelineRasterizationDepthClipStateCreateFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                              isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineRasterizationDepthClipStateCreateFlagsEXT allFlags  = {};
  };

  //=== VK_KHR_performance_query ===

  enum class PerformanceCounterDescriptionFlagBitsKHR : VkPerformanceCounterDescriptionFlagsKHR
  {
    ePerformanceImpacting = VK_PERFORMANCE_COUNTER_DESCRIPTION_PERFORMANCE_IMPACTING_BIT_KHR,
    eConcurrentlyImpacted = VK_PERFORMANCE_COUNTER_DESCRIPTION_CONCURRENTLY_IMPACTED_BIT_KHR
  };

  using PerformanceCounterDescriptionFlagsKHR = Flags<PerformanceCounterDescriptionFlagBitsKHR>;

  template <>
  struct FlagTraits<PerformanceCounterDescriptionFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                  isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PerformanceCounterDescriptionFlagsKHR allFlags =
      PerformanceCounterDescriptionFlagBitsKHR::ePerformanceImpacting | PerformanceCounterDescriptionFlagBitsKHR::eConcurrentlyImpacted;
  };

  enum class PerformanceCounterScopeKHR
  {
    eCommandBuffer             = VK_PERFORMANCE_COUNTER_SCOPE_COMMAND_BUFFER_KHR,
    eRenderPass                = VK_PERFORMANCE_COUNTER_SCOPE_RENDER_PASS_KHR,
    eCommand                   = VK_PERFORMANCE_COUNTER_SCOPE_COMMAND_KHR,
    eVkQueryScopeCommandBuffer = VK_QUERY_SCOPE_COMMAND_BUFFER_KHR,
    eVkQueryScopeRenderPass    = VK_QUERY_SCOPE_RENDER_PASS_KHR,
    eVkQueryScopeCommand       = VK_QUERY_SCOPE_COMMAND_KHR
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

  using AcquireProfilingLockFlagsKHR = Flags<AcquireProfilingLockFlagBitsKHR>;

  template <>
  struct FlagTraits<AcquireProfilingLockFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                         isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR AcquireProfilingLockFlagsKHR allFlags  = {};
  };

#if defined( VK_USE_PLATFORM_IOS_MVK )
  //=== VK_MVK_ios_surface ===

  enum class IOSSurfaceCreateFlagBitsMVK : VkIOSSurfaceCreateFlagsMVK
  {
  };

  using IOSSurfaceCreateFlagsMVK = Flags<IOSSurfaceCreateFlagBitsMVK>;

  template <>
  struct FlagTraits<IOSSurfaceCreateFlagBitsMVK>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                     isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR IOSSurfaceCreateFlagsMVK allFlags  = {};
  };
#endif /*VK_USE_PLATFORM_IOS_MVK*/

#if defined( VK_USE_PLATFORM_MACOS_MVK )
  //=== VK_MVK_macos_surface ===

  enum class MacOSSurfaceCreateFlagBitsMVK : VkMacOSSurfaceCreateFlagsMVK
  {
  };

  using MacOSSurfaceCreateFlagsMVK = Flags<MacOSSurfaceCreateFlagBitsMVK>;

  template <>
  struct FlagTraits<MacOSSurfaceCreateFlagBitsMVK>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR MacOSSurfaceCreateFlagsMVK allFlags  = {};
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

  using DebugUtilsMessageSeverityFlagsEXT = Flags<DebugUtilsMessageSeverityFlagBitsEXT>;

  template <>
  struct FlagTraits<DebugUtilsMessageSeverityFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                              isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DebugUtilsMessageSeverityFlagsEXT allFlags =
      DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | DebugUtilsMessageSeverityFlagBitsEXT::eInfo | DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
      DebugUtilsMessageSeverityFlagBitsEXT::eError;
  };

  enum class DebugUtilsMessageTypeFlagBitsEXT : VkDebugUtilsMessageTypeFlagsEXT
  {
    eGeneral              = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT,
    eValidation           = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
    ePerformance          = VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
    eDeviceAddressBinding = VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT
  };

  using DebugUtilsMessageTypeFlagsEXT = Flags<DebugUtilsMessageTypeFlagBitsEXT>;

  template <>
  struct FlagTraits<DebugUtilsMessageTypeFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                          isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DebugUtilsMessageTypeFlagsEXT allFlags =
      DebugUtilsMessageTypeFlagBitsEXT::eGeneral | DebugUtilsMessageTypeFlagBitsEXT::eValidation | DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
      DebugUtilsMessageTypeFlagBitsEXT::eDeviceAddressBinding;
  };

  enum class DebugUtilsMessengerCallbackDataFlagBitsEXT : VkDebugUtilsMessengerCallbackDataFlagsEXT
  {
  };

  using DebugUtilsMessengerCallbackDataFlagsEXT = Flags<DebugUtilsMessengerCallbackDataFlagBitsEXT>;

  template <>
  struct FlagTraits<DebugUtilsMessengerCallbackDataFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                    isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DebugUtilsMessengerCallbackDataFlagsEXT allFlags  = {};
  };

  enum class DebugUtilsMessengerCreateFlagBitsEXT : VkDebugUtilsMessengerCreateFlagsEXT
  {
  };

  using DebugUtilsMessengerCreateFlagsEXT = Flags<DebugUtilsMessengerCreateFlagBitsEXT>;

  template <>
  struct FlagTraits<DebugUtilsMessengerCreateFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                              isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DebugUtilsMessengerCreateFlagsEXT allFlags  = {};
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

  using PipelineCoverageToColorStateCreateFlagsNV = Flags<PipelineCoverageToColorStateCreateFlagBitsNV>;

  template <>
  struct FlagTraits<PipelineCoverageToColorStateCreateFlagBitsNV>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                      isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineCoverageToColorStateCreateFlagsNV allFlags  = {};
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

  using GeometryFlagsKHR = Flags<GeometryFlagBitsKHR>;
  using GeometryFlagsNV  = GeometryFlagsKHR;

  template <>
  struct FlagTraits<GeometryFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR GeometryFlagsKHR allFlags  = GeometryFlagBitsKHR::eOpaque | GeometryFlagBitsKHR::eNoDuplicateAnyHitInvocation;
  };

  enum class GeometryInstanceFlagBitsKHR : VkGeometryInstanceFlagsKHR
  {
    eTriangleFacingCullDisable        = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
    eTriangleFlipFacing               = VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR,
    eForceOpaque                      = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR,
    eForceNoOpaque                    = VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR,
    eTriangleFrontCounterclockwiseKHR = VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR,
    eTriangleCullDisable              = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV,
    eTriangleFrontCounterclockwise    = VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_NV,
    eForceOpacityMicromap2StateEXT    = VK_GEOMETRY_INSTANCE_FORCE_OPACITY_MICROMAP_2_STATE_EXT,
    eDisableOpacityMicromapsEXT       = VK_GEOMETRY_INSTANCE_DISABLE_OPACITY_MICROMAPS_EXT
  };
  using GeometryInstanceFlagBitsNV = GeometryInstanceFlagBitsKHR;

  using GeometryInstanceFlagsKHR = Flags<GeometryInstanceFlagBitsKHR>;
  using GeometryInstanceFlagsNV  = GeometryInstanceFlagsKHR;

  template <>
  struct FlagTraits<GeometryInstanceFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                     isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR GeometryInstanceFlagsKHR allFlags =
      GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable | GeometryInstanceFlagBitsKHR::eTriangleFlipFacing | GeometryInstanceFlagBitsKHR::eForceOpaque |
      GeometryInstanceFlagBitsKHR::eForceNoOpaque | GeometryInstanceFlagBitsKHR::eForceOpacityMicromap2StateEXT |
      GeometryInstanceFlagBitsKHR::eDisableOpacityMicromapsEXT;
  };

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
    eAllowOpacityMicromapDataUpdateEXT = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_OPACITY_MICROMAP_DATA_UPDATE_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eAllowDisplacementMicromapUpdateNV = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DISPLACEMENT_MICROMAP_UPDATE_NV,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eAllowDataAccess = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_KHR
  };
  using BuildAccelerationStructureFlagBitsNV = BuildAccelerationStructureFlagBitsKHR;

  using BuildAccelerationStructureFlagsKHR = Flags<BuildAccelerationStructureFlagBitsKHR>;
  using BuildAccelerationStructureFlagsNV  = BuildAccelerationStructureFlagsKHR;

  template <>
  struct FlagTraits<BuildAccelerationStructureFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                               isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR BuildAccelerationStructureFlagsKHR allFlags =
      BuildAccelerationStructureFlagBitsKHR::eAllowUpdate | BuildAccelerationStructureFlagBitsKHR::eAllowCompaction |
      BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace | BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild |
      BuildAccelerationStructureFlagBitsKHR::eLowMemory | BuildAccelerationStructureFlagBitsKHR::eMotionNV |
      BuildAccelerationStructureFlagBitsKHR::eAllowOpacityMicromapUpdateEXT | BuildAccelerationStructureFlagBitsKHR::eAllowDisableOpacityMicromapsEXT |
      BuildAccelerationStructureFlagBitsKHR::eAllowOpacityMicromapDataUpdateEXT
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      | BuildAccelerationStructureFlagBitsKHR::eAllowDisplacementMicromapUpdateNV
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      | BuildAccelerationStructureFlagBitsKHR::eAllowDataAccess;
  };

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
    eDeviceAddressCaptureReplay       = VK_ACCELERATION_STRUCTURE_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT_KHR,
    eDescriptorBufferCaptureReplayEXT = VK_ACCELERATION_STRUCTURE_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_EXT,
    eMotionNV                         = VK_ACCELERATION_STRUCTURE_CREATE_MOTION_BIT_NV
  };

  using AccelerationStructureCreateFlagsKHR = Flags<AccelerationStructureCreateFlagBitsKHR>;

  template <>
  struct FlagTraits<AccelerationStructureCreateFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR AccelerationStructureCreateFlagsKHR allFlags =
      AccelerationStructureCreateFlagBitsKHR::eDeviceAddressCaptureReplay | AccelerationStructureCreateFlagBitsKHR::eDescriptorBufferCaptureReplayEXT |
      AccelerationStructureCreateFlagBitsKHR::eMotionNV;
  };

  enum class BuildAccelerationStructureModeKHR
  {
    eBuild  = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
    eUpdate = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
  };

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

  using PipelineCoverageModulationStateCreateFlagsNV = Flags<PipelineCoverageModulationStateCreateFlagBitsNV>;

  template <>
  struct FlagTraits<PipelineCoverageModulationStateCreateFlagBitsNV>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                         isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineCoverageModulationStateCreateFlagsNV allFlags  = {};
  };

  //=== VK_EXT_validation_cache ===

  enum class ValidationCacheHeaderVersionEXT
  {
    eOne = VK_VALIDATION_CACHE_HEADER_VERSION_ONE_EXT
  };

  enum class ValidationCacheCreateFlagBitsEXT : VkValidationCacheCreateFlagsEXT
  {
  };

  using ValidationCacheCreateFlagsEXT = Flags<ValidationCacheCreateFlagBitsEXT>;

  template <>
  struct FlagTraits<ValidationCacheCreateFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                          isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ValidationCacheCreateFlagsEXT allFlags  = {};
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

  using PipelineCompilerControlFlagsAMD = Flags<PipelineCompilerControlFlagBitsAMD>;

  template <>
  struct FlagTraits<PipelineCompilerControlFlagBitsAMD>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                            isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineCompilerControlFlagsAMD allFlags  = {};
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

  using ImagePipeSurfaceCreateFlagsFUCHSIA = Flags<ImagePipeSurfaceCreateFlagBitsFUCHSIA>;

  template <>
  struct FlagTraits<ImagePipeSurfaceCreateFlagBitsFUCHSIA>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                               isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ImagePipeSurfaceCreateFlagsFUCHSIA allFlags  = {};
  };
#endif /*VK_USE_PLATFORM_FUCHSIA*/

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_surface ===

  enum class MetalSurfaceCreateFlagBitsEXT : VkMetalSurfaceCreateFlagsEXT
  {
  };

  using MetalSurfaceCreateFlagsEXT = Flags<MetalSurfaceCreateFlagBitsEXT>;

  template <>
  struct FlagTraits<MetalSurfaceCreateFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR MetalSurfaceCreateFlagsEXT allFlags  = {};
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

  using ShaderCorePropertiesFlagsAMD = Flags<ShaderCorePropertiesFlagBitsAMD>;

  template <>
  struct FlagTraits<ShaderCorePropertiesFlagBitsAMD>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                         isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ShaderCorePropertiesFlagsAMD allFlags  = {};
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

  //=== VK_NV_coverage_reduction_mode ===

  enum class CoverageReductionModeNV
  {
    eMerge    = VK_COVERAGE_REDUCTION_MODE_MERGE_NV,
    eTruncate = VK_COVERAGE_REDUCTION_MODE_TRUNCATE_NV
  };

  enum class PipelineCoverageReductionStateCreateFlagBitsNV : VkPipelineCoverageReductionStateCreateFlagsNV
  {
  };

  using PipelineCoverageReductionStateCreateFlagsNV = Flags<PipelineCoverageReductionStateCreateFlagBitsNV>;

  template <>
  struct FlagTraits<PipelineCoverageReductionStateCreateFlagBitsNV>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                        isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineCoverageReductionStateCreateFlagsNV allFlags  = {};
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

  using HeadlessSurfaceCreateFlagsEXT = Flags<HeadlessSurfaceCreateFlagBitsEXT>;

  template <>
  struct FlagTraits<HeadlessSurfaceCreateFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                          isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR HeadlessSurfaceCreateFlagsEXT allFlags  = {};
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

  //=== VK_EXT_host_image_copy ===

  enum class HostImageCopyFlagBitsEXT : VkHostImageCopyFlagsEXT
  {
    eMemcpy = VK_HOST_IMAGE_COPY_MEMCPY_EXT
  };

  using HostImageCopyFlagsEXT = Flags<HostImageCopyFlagBitsEXT>;

  template <>
  struct FlagTraits<HostImageCopyFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                  isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR HostImageCopyFlagsEXT allFlags  = HostImageCopyFlagBitsEXT::eMemcpy;
  };

  //=== VK_KHR_map_memory2 ===

  enum class MemoryUnmapFlagBitsKHR : VkMemoryUnmapFlagsKHR
  {
  };

  using MemoryUnmapFlagsKHR = Flags<MemoryUnmapFlagBitsKHR>;

  template <>
  struct FlagTraits<MemoryUnmapFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR MemoryUnmapFlagsKHR allFlags  = {};
  };

  //=== VK_EXT_surface_maintenance1 ===

  enum class PresentScalingFlagBitsEXT : VkPresentScalingFlagsEXT
  {
    eOneToOne           = VK_PRESENT_SCALING_ONE_TO_ONE_BIT_EXT,
    eAspectRatioStretch = VK_PRESENT_SCALING_ASPECT_RATIO_STRETCH_BIT_EXT,
    eStretch            = VK_PRESENT_SCALING_STRETCH_BIT_EXT
  };

  using PresentScalingFlagsEXT = Flags<PresentScalingFlagBitsEXT>;

  template <>
  struct FlagTraits<PresentScalingFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PresentScalingFlagsEXT allFlags =
      PresentScalingFlagBitsEXT::eOneToOne | PresentScalingFlagBitsEXT::eAspectRatioStretch | PresentScalingFlagBitsEXT::eStretch;
  };

  enum class PresentGravityFlagBitsEXT : VkPresentGravityFlagsEXT
  {
    eMin      = VK_PRESENT_GRAVITY_MIN_BIT_EXT,
    eMax      = VK_PRESENT_GRAVITY_MAX_BIT_EXT,
    eCentered = VK_PRESENT_GRAVITY_CENTERED_BIT_EXT
  };

  using PresentGravityFlagsEXT = Flags<PresentGravityFlagBitsEXT>;

  template <>
  struct FlagTraits<PresentGravityFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PresentGravityFlagsEXT allFlags =
      PresentGravityFlagBitsEXT::eMin | PresentGravityFlagBitsEXT::eMax | PresentGravityFlagBitsEXT::eCentered;
  };

  //=== VK_NV_device_generated_commands ===

  enum class IndirectStateFlagBitsNV : VkIndirectStateFlagsNV
  {
    eFlagFrontface = VK_INDIRECT_STATE_FLAG_FRONTFACE_BIT_NV
  };

  using IndirectStateFlagsNV = Flags<IndirectStateFlagBitsNV>;

  template <>
  struct FlagTraits<IndirectStateFlagBitsNV>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR IndirectStateFlagsNV allFlags  = IndirectStateFlagBitsNV::eFlagFrontface;
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
    eDrawMeshTasks = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_MESH_TASKS_NV,
    ePipeline      = VK_INDIRECT_COMMANDS_TOKEN_TYPE_PIPELINE_NV,
    eDispatch      = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_NV
  };

  enum class IndirectCommandsLayoutUsageFlagBitsNV : VkIndirectCommandsLayoutUsageFlagsNV
  {
    eExplicitPreprocess = VK_INDIRECT_COMMANDS_LAYOUT_USAGE_EXPLICIT_PREPROCESS_BIT_NV,
    eIndexedSequences   = VK_INDIRECT_COMMANDS_LAYOUT_USAGE_INDEXED_SEQUENCES_BIT_NV,
    eUnorderedSequences = VK_INDIRECT_COMMANDS_LAYOUT_USAGE_UNORDERED_SEQUENCES_BIT_NV
  };

  using IndirectCommandsLayoutUsageFlagsNV = Flags<IndirectCommandsLayoutUsageFlagBitsNV>;

  template <>
  struct FlagTraits<IndirectCommandsLayoutUsageFlagBitsNV>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                               isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR IndirectCommandsLayoutUsageFlagsNV allFlags  = IndirectCommandsLayoutUsageFlagBitsNV::eExplicitPreprocess |
                                                                                       IndirectCommandsLayoutUsageFlagBitsNV::eIndexedSequences |
                                                                                       IndirectCommandsLayoutUsageFlagBitsNV::eUnorderedSequences;
  };

  //=== VK_EXT_depth_bias_control ===

  enum class DepthBiasRepresentationEXT
  {
    eLeastRepresentableValueFormat     = VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORMAT_EXT,
    eLeastRepresentableValueForceUnorm = VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT,
    eFloat                             = VK_DEPTH_BIAS_REPRESENTATION_FLOAT_EXT
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

  using DeviceMemoryReportFlagsEXT = Flags<DeviceMemoryReportFlagBitsEXT>;

  template <>
  struct FlagTraits<DeviceMemoryReportFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DeviceMemoryReportFlagsEXT allFlags  = {};
  };

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_KHR_video_encode_queue ===

  enum class VideoEncodeCapabilityFlagBitsKHR : VkVideoEncodeCapabilityFlagsKHR
  {
    ePrecedingExternallyEncodedBytes           = VK_VIDEO_ENCODE_CAPABILITY_PRECEDING_EXTERNALLY_ENCODED_BYTES_BIT_KHR,
    eInsufficientstreamBufferRangeDetectionBit = VK_VIDEO_ENCODE_CAPABILITY_INSUFFICIENT_BITSTREAM_BUFFER_RANGE_DETECTION_BIT_KHR
  };

  using VideoEncodeCapabilityFlagsKHR = Flags<VideoEncodeCapabilityFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeCapabilityFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                          isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeCapabilityFlagsKHR allFlags =
      VideoEncodeCapabilityFlagBitsKHR::ePrecedingExternallyEncodedBytes | VideoEncodeCapabilityFlagBitsKHR::eInsufficientstreamBufferRangeDetectionBit;
  };

  enum class VideoEncodeFeedbackFlagBitsKHR : VkVideoEncodeFeedbackFlagsKHR
  {
    estreamBufferOffsetBit = VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BUFFER_OFFSET_BIT_KHR,
    estreamBytesWrittenBit = VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BYTES_WRITTEN_BIT_KHR,
    estreamHasOverridesBit = VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_HAS_OVERRIDES_BIT_KHR
  };

  using VideoEncodeFeedbackFlagsKHR = Flags<VideoEncodeFeedbackFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeFeedbackFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                        isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeFeedbackFlagsKHR allFlags  = VideoEncodeFeedbackFlagBitsKHR::estreamBufferOffsetBit |
                                                                                VideoEncodeFeedbackFlagBitsKHR::estreamBytesWrittenBit |
                                                                                VideoEncodeFeedbackFlagBitsKHR::estreamHasOverridesBit;
  };

  enum class VideoEncodeUsageFlagBitsKHR : VkVideoEncodeUsageFlagsKHR
  {
    eDefault      = VK_VIDEO_ENCODE_USAGE_DEFAULT_KHR,
    eTranscoding  = VK_VIDEO_ENCODE_USAGE_TRANSCODING_BIT_KHR,
    eStreaming    = VK_VIDEO_ENCODE_USAGE_STREAMING_BIT_KHR,
    eRecording    = VK_VIDEO_ENCODE_USAGE_RECORDING_BIT_KHR,
    eConferencing = VK_VIDEO_ENCODE_USAGE_CONFERENCING_BIT_KHR
  };

  using VideoEncodeUsageFlagsKHR = Flags<VideoEncodeUsageFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeUsageFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                     isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeUsageFlagsKHR allFlags = VideoEncodeUsageFlagBitsKHR::eDefault | VideoEncodeUsageFlagBitsKHR::eTranscoding |
                                                                             VideoEncodeUsageFlagBitsKHR::eStreaming | VideoEncodeUsageFlagBitsKHR::eRecording |
                                                                             VideoEncodeUsageFlagBitsKHR::eConferencing;
  };

  enum class VideoEncodeContentFlagBitsKHR : VkVideoEncodeContentFlagsKHR
  {
    eDefault  = VK_VIDEO_ENCODE_CONTENT_DEFAULT_KHR,
    eCamera   = VK_VIDEO_ENCODE_CONTENT_CAMERA_BIT_KHR,
    eDesktop  = VK_VIDEO_ENCODE_CONTENT_DESKTOP_BIT_KHR,
    eRendered = VK_VIDEO_ENCODE_CONTENT_RENDERED_BIT_KHR
  };

  using VideoEncodeContentFlagsKHR = Flags<VideoEncodeContentFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeContentFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeContentFlagsKHR allFlags =
      VideoEncodeContentFlagBitsKHR::eDefault | VideoEncodeContentFlagBitsKHR::eCamera | VideoEncodeContentFlagBitsKHR::eDesktop |
      VideoEncodeContentFlagBitsKHR::eRendered;
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
    eDefault  = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DEFAULT_KHR,
    eDisabled = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR,
    eCbr      = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR,
    eVbr      = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_VBR_BIT_KHR
  };

  using VideoEncodeRateControlModeFlagsKHR = Flags<VideoEncodeRateControlModeFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeRateControlModeFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                               isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeRateControlModeFlagsKHR allFlags =
      VideoEncodeRateControlModeFlagBitsKHR::eDefault | VideoEncodeRateControlModeFlagBitsKHR::eDisabled | VideoEncodeRateControlModeFlagBitsKHR::eCbr |
      VideoEncodeRateControlModeFlagBitsKHR::eVbr;
  };

  enum class VideoEncodeFlagBitsKHR : VkVideoEncodeFlagsKHR
  {
  };

  using VideoEncodeFlagsKHR = Flags<VideoEncodeFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeFlagsKHR allFlags  = {};
  };

  enum class VideoEncodeRateControlFlagBitsKHR : VkVideoEncodeRateControlFlagsKHR
  {
  };

  using VideoEncodeRateControlFlagsKHR = Flags<VideoEncodeRateControlFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeRateControlFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeRateControlFlagsKHR allFlags  = {};
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

  using DeviceDiagnosticsConfigFlagsNV = Flags<DeviceDiagnosticsConfigFlagBitsNV>;

  template <>
  struct FlagTraits<DeviceDiagnosticsConfigFlagBitsNV>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DeviceDiagnosticsConfigFlagsNV allFlags =
      DeviceDiagnosticsConfigFlagBitsNV::eEnableShaderDebugInfo | DeviceDiagnosticsConfigFlagBitsNV::eEnableResourceTracking |
      DeviceDiagnosticsConfigFlagBitsNV::eEnableAutomaticCheckpoints | DeviceDiagnosticsConfigFlagBitsNV::eEnableShaderErrorReporting;
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

  using ExportMetalObjectTypeFlagsEXT = Flags<ExportMetalObjectTypeFlagBitsEXT>;

  template <>
  struct FlagTraits<ExportMetalObjectTypeFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                          isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ExportMetalObjectTypeFlagsEXT allFlags =
      ExportMetalObjectTypeFlagBitsEXT::eMetalDevice | ExportMetalObjectTypeFlagBitsEXT::eMetalCommandQueue | ExportMetalObjectTypeFlagBitsEXT::eMetalBuffer |
      ExportMetalObjectTypeFlagBitsEXT::eMetalTexture | ExportMetalObjectTypeFlagBitsEXT::eMetalIosurface | ExportMetalObjectTypeFlagBitsEXT::eMetalSharedEvent;
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

  using GraphicsPipelineLibraryFlagsEXT = Flags<GraphicsPipelineLibraryFlagBitsEXT>;

  template <>
  struct FlagTraits<GraphicsPipelineLibraryFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                            isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR GraphicsPipelineLibraryFlagsEXT allFlags =
      GraphicsPipelineLibraryFlagBitsEXT::eVertexInputInterface | GraphicsPipelineLibraryFlagBitsEXT::ePreRasterizationShaders |
      GraphicsPipelineLibraryFlagBitsEXT::eFragmentShader | GraphicsPipelineLibraryFlagBitsEXT::eFragmentOutputInterface;
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

  using AccelerationStructureMotionInfoFlagsNV = Flags<AccelerationStructureMotionInfoFlagBitsNV>;

  template <>
  struct FlagTraits<AccelerationStructureMotionInfoFlagBitsNV>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR AccelerationStructureMotionInfoFlagsNV allFlags  = {};
  };

  enum class AccelerationStructureMotionInstanceFlagBitsNV : VkAccelerationStructureMotionInstanceFlagsNV
  {
  };

  using AccelerationStructureMotionInstanceFlagsNV = Flags<AccelerationStructureMotionInstanceFlagBitsNV>;

  template <>
  struct FlagTraits<AccelerationStructureMotionInstanceFlagBitsNV>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR AccelerationStructureMotionInstanceFlagsNV allFlags  = {};
  };

  //=== VK_EXT_image_compression_control ===

  enum class ImageCompressionFlagBitsEXT : VkImageCompressionFlagsEXT
  {
    eDefault           = VK_IMAGE_COMPRESSION_DEFAULT_EXT,
    eFixedRateDefault  = VK_IMAGE_COMPRESSION_FIXED_RATE_DEFAULT_EXT,
    eFixedRateExplicit = VK_IMAGE_COMPRESSION_FIXED_RATE_EXPLICIT_EXT,
    eDisabled          = VK_IMAGE_COMPRESSION_DISABLED_EXT
  };

  using ImageCompressionFlagsEXT = Flags<ImageCompressionFlagBitsEXT>;

  template <>
  struct FlagTraits<ImageCompressionFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                     isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ImageCompressionFlagsEXT allFlags =
      ImageCompressionFlagBitsEXT::eDefault | ImageCompressionFlagBitsEXT::eFixedRateDefault | ImageCompressionFlagBitsEXT::eFixedRateExplicit |
      ImageCompressionFlagBitsEXT::eDisabled;
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

  using ImageCompressionFixedRateFlagsEXT = Flags<ImageCompressionFixedRateFlagBitsEXT>;

  template <>
  struct FlagTraits<ImageCompressionFixedRateFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                              isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ImageCompressionFixedRateFlagsEXT allFlags =
      ImageCompressionFixedRateFlagBitsEXT::eNone | ImageCompressionFixedRateFlagBitsEXT::e1Bpc | ImageCompressionFixedRateFlagBitsEXT::e2Bpc |
      ImageCompressionFixedRateFlagBitsEXT::e3Bpc | ImageCompressionFixedRateFlagBitsEXT::e4Bpc | ImageCompressionFixedRateFlagBitsEXT::e5Bpc |
      ImageCompressionFixedRateFlagBitsEXT::e6Bpc | ImageCompressionFixedRateFlagBitsEXT::e7Bpc | ImageCompressionFixedRateFlagBitsEXT::e8Bpc |
      ImageCompressionFixedRateFlagBitsEXT::e9Bpc | ImageCompressionFixedRateFlagBitsEXT::e10Bpc | ImageCompressionFixedRateFlagBitsEXT::e11Bpc |
      ImageCompressionFixedRateFlagBitsEXT::e12Bpc | ImageCompressionFixedRateFlagBitsEXT::e13Bpc | ImageCompressionFixedRateFlagBitsEXT::e14Bpc |
      ImageCompressionFixedRateFlagBitsEXT::e15Bpc | ImageCompressionFixedRateFlagBitsEXT::e16Bpc | ImageCompressionFixedRateFlagBitsEXT::e17Bpc |
      ImageCompressionFixedRateFlagBitsEXT::e18Bpc | ImageCompressionFixedRateFlagBitsEXT::e19Bpc | ImageCompressionFixedRateFlagBitsEXT::e20Bpc |
      ImageCompressionFixedRateFlagBitsEXT::e21Bpc | ImageCompressionFixedRateFlagBitsEXT::e22Bpc | ImageCompressionFixedRateFlagBitsEXT::e23Bpc |
      ImageCompressionFixedRateFlagBitsEXT::e24Bpc;
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

  using DirectFBSurfaceCreateFlagsEXT = Flags<DirectFBSurfaceCreateFlagBitsEXT>;

  template <>
  struct FlagTraits<DirectFBSurfaceCreateFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                          isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DirectFBSurfaceCreateFlagsEXT allFlags  = {};
  };
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

  //=== VK_EXT_device_address_binding_report ===

  enum class DeviceAddressBindingFlagBitsEXT : VkDeviceAddressBindingFlagsEXT
  {
    eInternalObject = VK_DEVICE_ADDRESS_BINDING_INTERNAL_OBJECT_BIT_EXT
  };

  using DeviceAddressBindingFlagsEXT = Flags<DeviceAddressBindingFlagBitsEXT>;

  template <>
  struct FlagTraits<DeviceAddressBindingFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                         isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DeviceAddressBindingFlagsEXT allFlags  = DeviceAddressBindingFlagBitsEXT::eInternalObject;
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

  using ImageConstraintsInfoFlagsFUCHSIA = Flags<ImageConstraintsInfoFlagBitsFUCHSIA>;

  template <>
  struct FlagTraits<ImageConstraintsInfoFlagBitsFUCHSIA>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ImageConstraintsInfoFlagsFUCHSIA allFlags =
      ImageConstraintsInfoFlagBitsFUCHSIA::eCpuReadRarely | ImageConstraintsInfoFlagBitsFUCHSIA::eCpuReadOften |
      ImageConstraintsInfoFlagBitsFUCHSIA::eCpuWriteRarely | ImageConstraintsInfoFlagBitsFUCHSIA::eCpuWriteOften |
      ImageConstraintsInfoFlagBitsFUCHSIA::eProtectedOptional;
  };

  enum class ImageFormatConstraintsFlagBitsFUCHSIA : VkImageFormatConstraintsFlagsFUCHSIA
  {
  };

  using ImageFormatConstraintsFlagsFUCHSIA = Flags<ImageFormatConstraintsFlagBitsFUCHSIA>;

  template <>
  struct FlagTraits<ImageFormatConstraintsFlagBitsFUCHSIA>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                               isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ImageFormatConstraintsFlagsFUCHSIA allFlags  = {};
  };
#endif /*VK_USE_PLATFORM_FUCHSIA*/

  //=== VK_EXT_frame_boundary ===

  enum class FrameBoundaryFlagBitsEXT : VkFrameBoundaryFlagsEXT
  {
    eFrameEnd = VK_FRAME_BOUNDARY_FRAME_END_BIT_EXT
  };

  using FrameBoundaryFlagsEXT = Flags<FrameBoundaryFlagBitsEXT>;

  template <>
  struct FlagTraits<FrameBoundaryFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                  isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR FrameBoundaryFlagsEXT allFlags  = FrameBoundaryFlagBitsEXT::eFrameEnd;
  };

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
  //=== VK_QNX_screen_surface ===

  enum class ScreenSurfaceCreateFlagBitsQNX : VkScreenSurfaceCreateFlagsQNX
  {
  };

  using ScreenSurfaceCreateFlagsQNX = Flags<ScreenSurfaceCreateFlagBitsQNX>;

  template <>
  struct FlagTraits<ScreenSurfaceCreateFlagBitsQNX>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                        isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ScreenSurfaceCreateFlagsQNX allFlags  = {};
  };
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

  //=== VK_EXT_opacity_micromap ===

  enum class MicromapTypeEXT
  {
    eOpacityMicromap = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eDisplacementMicromapNV = VK_MICROMAP_TYPE_DISPLACEMENT_MICROMAP_NV
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
  };

  enum class BuildMicromapFlagBitsEXT : VkBuildMicromapFlagsEXT
  {
    ePreferFastTrace = VK_BUILD_MICROMAP_PREFER_FAST_TRACE_BIT_EXT,
    ePreferFastBuild = VK_BUILD_MICROMAP_PREFER_FAST_BUILD_BIT_EXT,
    eAllowCompaction = VK_BUILD_MICROMAP_ALLOW_COMPACTION_BIT_EXT
  };

  using BuildMicromapFlagsEXT = Flags<BuildMicromapFlagBitsEXT>;

  template <>
  struct FlagTraits<BuildMicromapFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                  isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR BuildMicromapFlagsEXT allFlags =
      BuildMicromapFlagBitsEXT::ePreferFastTrace | BuildMicromapFlagBitsEXT::ePreferFastBuild | BuildMicromapFlagBitsEXT::eAllowCompaction;
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

  using MicromapCreateFlagsEXT = Flags<MicromapCreateFlagBitsEXT>;

  template <>
  struct FlagTraits<MicromapCreateFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR MicromapCreateFlagsEXT allFlags  = MicromapCreateFlagBitsEXT::eDeviceAddressCaptureReplay;
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

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_NV_displacement_micromap ===

  enum class DisplacementMicromapFormatNV
  {
    e64Triangles64Bytes    = VK_DISPLACEMENT_MICROMAP_FORMAT_64_TRIANGLES_64_BYTES_NV,
    e256Triangles128Bytes  = VK_DISPLACEMENT_MICROMAP_FORMAT_256_TRIANGLES_128_BYTES_NV,
    e1024Triangles128Bytes = VK_DISPLACEMENT_MICROMAP_FORMAT_1024_TRIANGLES_128_BYTES_NV
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_NV_memory_decompression ===

  enum class MemoryDecompressionMethodFlagBitsNV : VkMemoryDecompressionMethodFlagsNV
  {
    eGdeflate10 = VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_NV
  };

  using MemoryDecompressionMethodFlagsNV = Flags<MemoryDecompressionMethodFlagBitsNV>;

  template <>
  struct FlagTraits<MemoryDecompressionMethodFlagBitsNV>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR MemoryDecompressionMethodFlagsNV allFlags  = MemoryDecompressionMethodFlagBitsNV::eGdeflate10;
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

  //=== VK_LUNARG_direct_driver_loading ===

  enum class DirectDriverLoadingModeLUNARG
  {
    eExclusive = VK_DIRECT_DRIVER_LOADING_MODE_EXCLUSIVE_LUNARG,
    eInclusive = VK_DIRECT_DRIVER_LOADING_MODE_INCLUSIVE_LUNARG
  };

  enum class DirectDriverLoadingFlagBitsLUNARG : VkDirectDriverLoadingFlagsLUNARG
  {
  };

  using DirectDriverLoadingFlagsLUNARG = Flags<DirectDriverLoadingFlagBitsLUNARG>;

  template <>
  struct FlagTraits<DirectDriverLoadingFlagBitsLUNARG>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DirectDriverLoadingFlagsLUNARG allFlags  = {};
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

  using OpticalFlowUsageFlagsNV = Flags<OpticalFlowUsageFlagBitsNV>;

  template <>
  struct FlagTraits<OpticalFlowUsageFlagBitsNV>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                    isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR OpticalFlowUsageFlagsNV allFlags  = OpticalFlowUsageFlagBitsNV::eUnknown | OpticalFlowUsageFlagBitsNV::eInput |
                                                                            OpticalFlowUsageFlagBitsNV::eOutput | OpticalFlowUsageFlagBitsNV::eHint |
                                                                            OpticalFlowUsageFlagBitsNV::eCost | OpticalFlowUsageFlagBitsNV::eGlobalFlow;
  };

  enum class OpticalFlowGridSizeFlagBitsNV : VkOpticalFlowGridSizeFlagsNV
  {
    eUnknown = VK_OPTICAL_FLOW_GRID_SIZE_UNKNOWN_NV,
    e1X1     = VK_OPTICAL_FLOW_GRID_SIZE_1X1_BIT_NV,
    e2X2     = VK_OPTICAL_FLOW_GRID_SIZE_2X2_BIT_NV,
    e4X4     = VK_OPTICAL_FLOW_GRID_SIZE_4X4_BIT_NV,
    e8X8     = VK_OPTICAL_FLOW_GRID_SIZE_8X8_BIT_NV
  };

  using OpticalFlowGridSizeFlagsNV = Flags<OpticalFlowGridSizeFlagBitsNV>;

  template <>
  struct FlagTraits<OpticalFlowGridSizeFlagBitsNV>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR OpticalFlowGridSizeFlagsNV allFlags  = OpticalFlowGridSizeFlagBitsNV::eUnknown | OpticalFlowGridSizeFlagBitsNV::e1X1 |
                                                                               OpticalFlowGridSizeFlagBitsNV::e2X2 | OpticalFlowGridSizeFlagBitsNV::e4X4 |
                                                                               OpticalFlowGridSizeFlagBitsNV::e8X8;
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

  using OpticalFlowSessionCreateFlagsNV = Flags<OpticalFlowSessionCreateFlagBitsNV>;

  template <>
  struct FlagTraits<OpticalFlowSessionCreateFlagBitsNV>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                            isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR OpticalFlowSessionCreateFlagsNV allFlags =
      OpticalFlowSessionCreateFlagBitsNV::eEnableHint | OpticalFlowSessionCreateFlagBitsNV::eEnableCost |
      OpticalFlowSessionCreateFlagBitsNV::eEnableGlobalFlow | OpticalFlowSessionCreateFlagBitsNV::eAllowRegions |
      OpticalFlowSessionCreateFlagBitsNV::eBothDirections;
  };

  enum class OpticalFlowExecuteFlagBitsNV : VkOpticalFlowExecuteFlagsNV
  {
    eDisableTemporalHints = VK_OPTICAL_FLOW_EXECUTE_DISABLE_TEMPORAL_HINTS_BIT_NV
  };

  using OpticalFlowExecuteFlagsNV = Flags<OpticalFlowExecuteFlagBitsNV>;

  template <>
  struct FlagTraits<OpticalFlowExecuteFlagBitsNV>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                      isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR OpticalFlowExecuteFlagsNV allFlags  = OpticalFlowExecuteFlagBitsNV::eDisableTemporalHints;
  };

  //=== VK_KHR_maintenance5 ===

  enum class PipelineCreateFlagBits2KHR : VkPipelineCreateFlags2KHR
  {
    eDisableOptimization                      = VK_PIPELINE_CREATE_2_DISABLE_OPTIMIZATION_BIT_KHR,
    eAllowDerivatives                         = VK_PIPELINE_CREATE_2_ALLOW_DERIVATIVES_BIT_KHR,
    eDerivative                               = VK_PIPELINE_CREATE_2_DERIVATIVE_BIT_KHR,
    eViewIndexFromDeviceIndex                 = VK_PIPELINE_CREATE_2_VIEW_INDEX_FROM_DEVICE_INDEX_BIT_KHR,
    eDispatchBase                             = VK_PIPELINE_CREATE_2_DISPATCH_BASE_BIT_KHR,
    eDeferCompileNV                           = VK_PIPELINE_CREATE_2_DEFER_COMPILE_BIT_NV,
    eCaptureStatistics                        = VK_PIPELINE_CREATE_2_CAPTURE_STATISTICS_BIT_KHR,
    eCaptureInternalRepresentations           = VK_PIPELINE_CREATE_2_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR,
    eFailOnPipelineCompileRequired            = VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_KHR,
    eEarlyReturnOnFailure                     = VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT_KHR,
    eLinkTimeOptimizationEXT                  = VK_PIPELINE_CREATE_2_LINK_TIME_OPTIMIZATION_BIT_EXT,
    eRetainLinkTimeOptimizationInfoEXT        = VK_PIPELINE_CREATE_2_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT,
    eLibrary                                  = VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR,
    eRayTracingSkipTriangles                  = VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_TRIANGLES_BIT_KHR,
    eRayTracingSkipAabbs                      = VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_AABBS_BIT_KHR,
    eRayTracingNoNullAnyHitShaders            = VK_PIPELINE_CREATE_2_RAY_TRACING_NO_NULL_ANY_HIT_SHADERS_BIT_KHR,
    eRayTracingNoNullClosestHitShaders        = VK_PIPELINE_CREATE_2_RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS_BIT_KHR,
    eRayTracingNoNullMissShaders              = VK_PIPELINE_CREATE_2_RAY_TRACING_NO_NULL_MISS_SHADERS_BIT_KHR,
    eRayTracingNoNullIntersectionShaders      = VK_PIPELINE_CREATE_2_RAY_TRACING_NO_NULL_INTERSECTION_SHADERS_BIT_KHR,
    eRayTracingShaderGroupHandleCaptureReplay = VK_PIPELINE_CREATE_2_RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY_BIT_KHR,
    eIndirectBindableNV                       = VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_NV,
    eRayTracingAllowMotionNV                  = VK_PIPELINE_CREATE_2_RAY_TRACING_ALLOW_MOTION_BIT_NV,
    eRenderingFragmentShadingRateAttachment   = VK_PIPELINE_CREATE_2_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
    eRenderingFragmentDensityMapAttachmentEXT = VK_PIPELINE_CREATE_2_RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT_BIT_EXT,
    eRayTracingOpacityMicromapEXT             = VK_PIPELINE_CREATE_2_RAY_TRACING_OPACITY_MICROMAP_BIT_EXT,
    eColorAttachmentFeedbackLoopEXT           = VK_PIPELINE_CREATE_2_COLOR_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT,
    eDepthStencilAttachmentFeedbackLoopEXT    = VK_PIPELINE_CREATE_2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT,
    eNoProtectedAccessEXT                     = VK_PIPELINE_CREATE_2_NO_PROTECTED_ACCESS_BIT_EXT,
    eProtectedAccessOnlyEXT                   = VK_PIPELINE_CREATE_2_PROTECTED_ACCESS_ONLY_BIT_EXT,
    eRayTracingDisplacementMicromapNV         = VK_PIPELINE_CREATE_2_RAY_TRACING_DISPLACEMENT_MICROMAP_BIT_NV,
    eDescriptorBufferEXT                      = VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT
  };

  using PipelineCreateFlags2KHR = Flags<PipelineCreateFlagBits2KHR>;

  template <>
  struct FlagTraits<PipelineCreateFlagBits2KHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                    isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineCreateFlags2KHR allFlags =
      PipelineCreateFlagBits2KHR::eDisableOptimization | PipelineCreateFlagBits2KHR::eAllowDerivatives | PipelineCreateFlagBits2KHR::eDerivative |
      PipelineCreateFlagBits2KHR::eViewIndexFromDeviceIndex | PipelineCreateFlagBits2KHR::eDispatchBase | PipelineCreateFlagBits2KHR::eDeferCompileNV |
      PipelineCreateFlagBits2KHR::eCaptureStatistics | PipelineCreateFlagBits2KHR::eCaptureInternalRepresentations |
      PipelineCreateFlagBits2KHR::eFailOnPipelineCompileRequired | PipelineCreateFlagBits2KHR::eEarlyReturnOnFailure |
      PipelineCreateFlagBits2KHR::eLinkTimeOptimizationEXT | PipelineCreateFlagBits2KHR::eRetainLinkTimeOptimizationInfoEXT |
      PipelineCreateFlagBits2KHR::eLibrary | PipelineCreateFlagBits2KHR::eRayTracingSkipTriangles | PipelineCreateFlagBits2KHR::eRayTracingSkipAabbs |
      PipelineCreateFlagBits2KHR::eRayTracingNoNullAnyHitShaders | PipelineCreateFlagBits2KHR::eRayTracingNoNullClosestHitShaders |
      PipelineCreateFlagBits2KHR::eRayTracingNoNullMissShaders | PipelineCreateFlagBits2KHR::eRayTracingNoNullIntersectionShaders |
      PipelineCreateFlagBits2KHR::eRayTracingShaderGroupHandleCaptureReplay | PipelineCreateFlagBits2KHR::eIndirectBindableNV |
      PipelineCreateFlagBits2KHR::eRayTracingAllowMotionNV | PipelineCreateFlagBits2KHR::eRenderingFragmentShadingRateAttachment |
      PipelineCreateFlagBits2KHR::eRenderingFragmentDensityMapAttachmentEXT | PipelineCreateFlagBits2KHR::eRayTracingOpacityMicromapEXT |
      PipelineCreateFlagBits2KHR::eColorAttachmentFeedbackLoopEXT | PipelineCreateFlagBits2KHR::eDepthStencilAttachmentFeedbackLoopEXT |
      PipelineCreateFlagBits2KHR::eNoProtectedAccessEXT | PipelineCreateFlagBits2KHR::eProtectedAccessOnlyEXT |
      PipelineCreateFlagBits2KHR::eRayTracingDisplacementMicromapNV | PipelineCreateFlagBits2KHR::eDescriptorBufferEXT;
  };

  enum class BufferUsageFlagBits2KHR : VkBufferUsageFlags2KHR
  {
    eTransferSrc        = VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT_KHR,
    eTransferDst        = VK_BUFFER_USAGE_2_TRANSFER_DST_BIT_KHR,
    eUniformTexelBuffer = VK_BUFFER_USAGE_2_UNIFORM_TEXEL_BUFFER_BIT_KHR,
    eStorageTexelBuffer = VK_BUFFER_USAGE_2_STORAGE_TEXEL_BUFFER_BIT_KHR,
    eUniformBuffer      = VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT_KHR,
    eStorageBuffer      = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR,
    eIndexBuffer        = VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT_KHR,
    eVertexBuffer       = VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT_KHR,
    eIndirectBuffer     = VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT_KHR,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eExecutionGraphScratchAMDX = VK_BUFFER_USAGE_2_EXECUTION_GRAPH_SCRATCH_BIT_AMDX,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eConditionalRenderingEXT           = VK_BUFFER_USAGE_2_CONDITIONAL_RENDERING_BIT_EXT,
    eShaderBindingTable                = VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR,
    eRayTracingNV                      = VK_BUFFER_USAGE_2_RAY_TRACING_BIT_NV,
    eTransformFeedbackBufferEXT        = VK_BUFFER_USAGE_2_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT,
    eTransformFeedbackCounterBufferEXT = VK_BUFFER_USAGE_2_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT,
    eVideoDecodeSrc                    = VK_BUFFER_USAGE_2_VIDEO_DECODE_SRC_BIT_KHR,
    eVideoDecodeDst                    = VK_BUFFER_USAGE_2_VIDEO_DECODE_DST_BIT_KHR,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eVideoEncodeDst = VK_BUFFER_USAGE_2_VIDEO_ENCODE_DST_BIT_KHR,
    eVideoEncodeSrc = VK_BUFFER_USAGE_2_VIDEO_ENCODE_SRC_BIT_KHR,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eShaderDeviceAddress                     = VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT_KHR,
    eAccelerationStructureBuildInputReadOnly = VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
    eAccelerationStructureStorage            = VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
    eSamplerDescriptorBufferEXT              = VK_BUFFER_USAGE_2_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT,
    eResourceDescriptorBufferEXT             = VK_BUFFER_USAGE_2_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT,
    ePushDescriptorsDescriptorBufferEXT      = VK_BUFFER_USAGE_2_PUSH_DESCRIPTORS_DESCRIPTOR_BUFFER_BIT_EXT,
    eMicromapBuildInputReadOnlyEXT           = VK_BUFFER_USAGE_2_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT,
    eMicromapStorageEXT                      = VK_BUFFER_USAGE_2_MICROMAP_STORAGE_BIT_EXT
  };

  using BufferUsageFlags2KHR = Flags<BufferUsageFlagBits2KHR>;

  template <>
  struct FlagTraits<BufferUsageFlagBits2KHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR BufferUsageFlags2KHR allFlags =
      BufferUsageFlagBits2KHR::eTransferSrc | BufferUsageFlagBits2KHR::eTransferDst | BufferUsageFlagBits2KHR::eUniformTexelBuffer |
      BufferUsageFlagBits2KHR::eStorageTexelBuffer | BufferUsageFlagBits2KHR::eUniformBuffer | BufferUsageFlagBits2KHR::eStorageBuffer |
      BufferUsageFlagBits2KHR::eIndexBuffer | BufferUsageFlagBits2KHR::eVertexBuffer | BufferUsageFlagBits2KHR::eIndirectBuffer
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      | BufferUsageFlagBits2KHR::eExecutionGraphScratchAMDX
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      | BufferUsageFlagBits2KHR::eConditionalRenderingEXT | BufferUsageFlagBits2KHR::eShaderBindingTable |
      BufferUsageFlagBits2KHR::eTransformFeedbackBufferEXT | BufferUsageFlagBits2KHR::eTransformFeedbackCounterBufferEXT |
      BufferUsageFlagBits2KHR::eVideoDecodeSrc | BufferUsageFlagBits2KHR::eVideoDecodeDst
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      | BufferUsageFlagBits2KHR::eVideoEncodeDst | BufferUsageFlagBits2KHR::eVideoEncodeSrc
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      | BufferUsageFlagBits2KHR::eShaderDeviceAddress | BufferUsageFlagBits2KHR::eAccelerationStructureBuildInputReadOnly |
      BufferUsageFlagBits2KHR::eAccelerationStructureStorage | BufferUsageFlagBits2KHR::eSamplerDescriptorBufferEXT |
      BufferUsageFlagBits2KHR::eResourceDescriptorBufferEXT | BufferUsageFlagBits2KHR::ePushDescriptorsDescriptorBufferEXT |
      BufferUsageFlagBits2KHR::eMicromapBuildInputReadOnlyEXT | BufferUsageFlagBits2KHR::eMicromapStorageEXT;
  };

  //=== VK_EXT_shader_object ===

  enum class ShaderCreateFlagBitsEXT : VkShaderCreateFlagsEXT
  {
    eLinkStage                     = VK_SHADER_CREATE_LINK_STAGE_BIT_EXT,
    eAllowVaryingSubgroupSize      = VK_SHADER_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT_EXT,
    eRequireFullSubgroups          = VK_SHADER_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT,
    eNoTaskShader                  = VK_SHADER_CREATE_NO_TASK_SHADER_BIT_EXT,
    eDispatchBase                  = VK_SHADER_CREATE_DISPATCH_BASE_BIT_EXT,
    eFragmentShadingRateAttachment = VK_SHADER_CREATE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_EXT,
    eFragmentDensityMapAttachment  = VK_SHADER_CREATE_FRAGMENT_DENSITY_MAP_ATTACHMENT_BIT_EXT
  };

  using ShaderCreateFlagsEXT = Flags<ShaderCreateFlagBitsEXT>;

  template <>
  struct FlagTraits<ShaderCreateFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ShaderCreateFlagsEXT allFlags =
      ShaderCreateFlagBitsEXT::eLinkStage | ShaderCreateFlagBitsEXT::eAllowVaryingSubgroupSize | ShaderCreateFlagBitsEXT::eRequireFullSubgroups |
      ShaderCreateFlagBitsEXT::eNoTaskShader | ShaderCreateFlagBitsEXT::eDispatchBase | ShaderCreateFlagBitsEXT::eFragmentShadingRateAttachment |
      ShaderCreateFlagBitsEXT::eFragmentDensityMapAttachment;
  };

  enum class ShaderCodeTypeEXT
  {
    eBinary = VK_SHADER_CODE_TYPE_BINARY_EXT,
    eSpirv  = VK_SHADER_CODE_TYPE_SPIRV_EXT
  };

  //=== VK_NV_ray_tracing_invocation_reorder ===

  enum class RayTracingInvocationReorderModeNV
  {
    eNone    = VK_RAY_TRACING_INVOCATION_REORDER_MODE_NONE_NV,
    eReorder = VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER_NV
  };

  //=== VK_NV_low_latency2 ===

  enum class LatencyMarkerNV
  {
    eSimulationStart            = VK_LATENCY_MARKER_SIMULATION_START_NV,
    eSimulationEnd              = VK_LATENCY_MARKER_SIMULATION_END_NV,
    eRendersubmitStart          = VK_LATENCY_MARKER_RENDERSUBMIT_START_NV,
    eRendersubmitEnd            = VK_LATENCY_MARKER_RENDERSUBMIT_END_NV,
    ePresentStart               = VK_LATENCY_MARKER_PRESENT_START_NV,
    ePresentEnd                 = VK_LATENCY_MARKER_PRESENT_END_NV,
    eInputSample                = VK_LATENCY_MARKER_INPUT_SAMPLE_NV,
    eTriggerFlash               = VK_LATENCY_MARKER_TRIGGER_FLASH_NV,
    eOutOfBandRendersubmitStart = VK_LATENCY_MARKER_OUT_OF_BAND_RENDERSUBMIT_START_NV,
    eOutOfBandRendersubmitEnd   = VK_LATENCY_MARKER_OUT_OF_BAND_RENDERSUBMIT_END_NV,
    eOutOfBandPresentStart      = VK_LATENCY_MARKER_OUT_OF_BAND_PRESENT_START_NV,
    eOutOfBandPresentEnd        = VK_LATENCY_MARKER_OUT_OF_BAND_PRESENT_END_NV
  };

  enum class OutOfBandQueueTypeNV
  {
    eRender  = VK_OUT_OF_BAND_QUEUE_TYPE_RENDER_NV,
    ePresent = VK_OUT_OF_BAND_QUEUE_TYPE_PRESENT_NV
  };

  //=== VK_KHR_cooperative_matrix ===

  enum class ScopeKHR
  {
    eDevice      = VK_SCOPE_DEVICE_KHR,
    eWorkgroup   = VK_SCOPE_WORKGROUP_KHR,
    eSubgroup    = VK_SCOPE_SUBGROUP_KHR,
    eQueueFamily = VK_SCOPE_QUEUE_FAMILY_KHR
  };
  using ScopeNV = ScopeKHR;

  enum class ComponentTypeKHR
  {
    eFloat16 = VK_COMPONENT_TYPE_FLOAT16_KHR,
    eFloat32 = VK_COMPONENT_TYPE_FLOAT32_KHR,
    eFloat64 = VK_COMPONENT_TYPE_FLOAT64_KHR,
    eSint8   = VK_COMPONENT_TYPE_SINT8_KHR,
    eSint16  = VK_COMPONENT_TYPE_SINT16_KHR,
    eSint32  = VK_COMPONENT_TYPE_SINT32_KHR,
    eSint64  = VK_COMPONENT_TYPE_SINT64_KHR,
    eUint8   = VK_COMPONENT_TYPE_UINT8_KHR,
    eUint16  = VK_COMPONENT_TYPE_UINT16_KHR,
    eUint32  = VK_COMPONENT_TYPE_UINT32_KHR,
    eUint64  = VK_COMPONENT_TYPE_UINT64_KHR
  };
  using ComponentTypeNV = ComponentTypeKHR;

  //=== VK_QCOM_image_processing2 ===

  enum class BlockMatchWindowCompareModeQCOM
  {
    eMin = VK_BLOCK_MATCH_WINDOW_COMPARE_MODE_MIN_QCOM,
    eMax = VK_BLOCK_MATCH_WINDOW_COMPARE_MODE_MAX_QCOM
  };

  //=== VK_QCOM_filter_cubic_weights ===

  enum class CubicFilterWeightsQCOM
  {
    eCatmullRom          = VK_CUBIC_FILTER_WEIGHTS_CATMULL_ROM_QCOM,
    eZeroTangentCardinal = VK_CUBIC_FILTER_WEIGHTS_ZERO_TANGENT_CARDINAL_QCOM,
    eBSpline             = VK_CUBIC_FILTER_WEIGHTS_B_SPLINE_QCOM,
    eMitchellNetravali   = VK_CUBIC_FILTER_WEIGHTS_MITCHELL_NETRAVALI_QCOM
  };

  //=== VK_MSFT_layered_driver ===

  enum class LayeredDriverUnderlyingApiMSFT
  {
    eNone  = VK_LAYERED_DRIVER_UNDERLYING_API_NONE_MSFT,
    eD3D12 = VK_LAYERED_DRIVER_UNDERLYING_API_D3D12_MSFT
  };

  //=========================
  //=== Index Type Traits ===
  //=========================

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

  //===========================================================
  //=== Mapping from ObjectType to DebugReportObjectTypeEXT ===
  //===========================================================

  VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType( VULKAN_HPP_NAMESPACE::ObjectType objectType )
  {
    switch ( objectType )
    {
        //=== VK_VERSION_1_0 ===
      case VULKAN_HPP_NAMESPACE::ObjectType::eInstance: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eInstance;
      case VULKAN_HPP_NAMESPACE::ObjectType::ePhysicalDevice: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::ePhysicalDevice;
      case VULKAN_HPP_NAMESPACE::ObjectType::eDevice: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eDevice;
      case VULKAN_HPP_NAMESPACE::ObjectType::eQueue: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eQueue;
      case VULKAN_HPP_NAMESPACE::ObjectType::eDeviceMemory: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eDeviceMemory;
      case VULKAN_HPP_NAMESPACE::ObjectType::eFence: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eFence;
      case VULKAN_HPP_NAMESPACE::ObjectType::eSemaphore: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eSemaphore;
      case VULKAN_HPP_NAMESPACE::ObjectType::eEvent: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eEvent;
      case VULKAN_HPP_NAMESPACE::ObjectType::eQueryPool: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eQueryPool;
      case VULKAN_HPP_NAMESPACE::ObjectType::eBuffer: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eBuffer;
      case VULKAN_HPP_NAMESPACE::ObjectType::eBufferView: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eBufferView;
      case VULKAN_HPP_NAMESPACE::ObjectType::eImage: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eImage;
      case VULKAN_HPP_NAMESPACE::ObjectType::eImageView: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eImageView;
      case VULKAN_HPP_NAMESPACE::ObjectType::eShaderModule: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eShaderModule;
      case VULKAN_HPP_NAMESPACE::ObjectType::ePipelineCache: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::ePipelineCache;
      case VULKAN_HPP_NAMESPACE::ObjectType::ePipeline: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::ePipeline;
      case VULKAN_HPP_NAMESPACE::ObjectType::ePipelineLayout: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::ePipelineLayout;
      case VULKAN_HPP_NAMESPACE::ObjectType::eSampler: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eSampler;
      case VULKAN_HPP_NAMESPACE::ObjectType::eDescriptorPool: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eDescriptorPool;
      case VULKAN_HPP_NAMESPACE::ObjectType::eDescriptorSet: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eDescriptorSet;
      case VULKAN_HPP_NAMESPACE::ObjectType::eDescriptorSetLayout: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eDescriptorSetLayout;
      case VULKAN_HPP_NAMESPACE::ObjectType::eFramebuffer: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eFramebuffer;
      case VULKAN_HPP_NAMESPACE::ObjectType::eRenderPass: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eRenderPass;
      case VULKAN_HPP_NAMESPACE::ObjectType::eCommandPool: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eCommandPool;
      case VULKAN_HPP_NAMESPACE::ObjectType::eCommandBuffer:
        return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eCommandBuffer;

        //=== VK_VERSION_1_1 ===
      case VULKAN_HPP_NAMESPACE::ObjectType::eSamplerYcbcrConversion: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eSamplerYcbcrConversion;
      case VULKAN_HPP_NAMESPACE::ObjectType::eDescriptorUpdateTemplate:
        return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eDescriptorUpdateTemplate;

        //=== VK_VERSION_1_3 ===
      case VULKAN_HPP_NAMESPACE::ObjectType::ePrivateDataSlot:
        return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eUnknown;

        //=== VK_KHR_surface ===
      case VULKAN_HPP_NAMESPACE::ObjectType::eSurfaceKHR:
        return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eSurfaceKHR;

        //=== VK_KHR_swapchain ===
      case VULKAN_HPP_NAMESPACE::ObjectType::eSwapchainKHR:
        return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eSwapchainKHR;

        //=== VK_KHR_display ===
      case VULKAN_HPP_NAMESPACE::ObjectType::eDisplayKHR: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eDisplayKHR;
      case VULKAN_HPP_NAMESPACE::ObjectType::eDisplayModeKHR:
        return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eDisplayModeKHR;

        //=== VK_EXT_debug_report ===
      case VULKAN_HPP_NAMESPACE::ObjectType::eDebugReportCallbackEXT:
        return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eDebugReportCallbackEXT;

        //=== VK_KHR_video_queue ===
      case VULKAN_HPP_NAMESPACE::ObjectType::eVideoSessionKHR: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eUnknown;
      case VULKAN_HPP_NAMESPACE::ObjectType::eVideoSessionParametersKHR:
        return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eUnknown;

        //=== VK_NVX_binary_import ===
      case VULKAN_HPP_NAMESPACE::ObjectType::eCuModuleNVX: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eCuModuleNVX;
      case VULKAN_HPP_NAMESPACE::ObjectType::eCuFunctionNVX:
        return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eCuFunctionNVX;

        //=== VK_EXT_debug_utils ===
      case VULKAN_HPP_NAMESPACE::ObjectType::eDebugUtilsMessengerEXT:
        return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eUnknown;

        //=== VK_KHR_acceleration_structure ===
      case VULKAN_HPP_NAMESPACE::ObjectType::eAccelerationStructureKHR:
        return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eAccelerationStructureKHR;

        //=== VK_EXT_validation_cache ===
      case VULKAN_HPP_NAMESPACE::ObjectType::eValidationCacheEXT:
        return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eValidationCacheEXT;

        //=== VK_NV_ray_tracing ===
      case VULKAN_HPP_NAMESPACE::ObjectType::eAccelerationStructureNV:
        return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eAccelerationStructureNV;

        //=== VK_INTEL_performance_query ===
      case VULKAN_HPP_NAMESPACE::ObjectType::ePerformanceConfigurationINTEL:
        return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eUnknown;

        //=== VK_KHR_deferred_host_operations ===
      case VULKAN_HPP_NAMESPACE::ObjectType::eDeferredOperationKHR:
        return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eUnknown;

        //=== VK_NV_device_generated_commands ===
      case VULKAN_HPP_NAMESPACE::ObjectType::eIndirectCommandsLayoutNV: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eUnknown;

#if defined( VK_USE_PLATFORM_FUCHSIA )
        //=== VK_FUCHSIA_buffer_collection ===
      case VULKAN_HPP_NAMESPACE::ObjectType::eBufferCollectionFUCHSIA: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eBufferCollectionFUCHSIA;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

        //=== VK_EXT_opacity_micromap ===
      case VULKAN_HPP_NAMESPACE::ObjectType::eMicromapEXT:
        return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eUnknown;

        //=== VK_NV_optical_flow ===
      case VULKAN_HPP_NAMESPACE::ObjectType::eOpticalFlowSessionNV:
        return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eUnknown;

        //=== VK_EXT_shader_object ===
      case VULKAN_HPP_NAMESPACE::ObjectType::eShaderEXT: return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eUnknown;

      default: VULKAN_HPP_ASSERT( false && "unknown ObjectType" ); return VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eUnknown;
    }
  }

}  // namespace VULKAN_HPP_NAMESPACE
#endif
