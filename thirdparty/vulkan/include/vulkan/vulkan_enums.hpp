// Copyright 2015-2025 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//

// This header is generated from the Khronos Vulkan XML API Registry.

#ifndef VULKAN_ENUMS_HPP
#define VULKAN_ENUMS_HPP

// include-what-you-use: make sure, vulkan.hpp is used by code-completers
// IWYU pragma: private, include "vulkan/vulkan.hpp"

#if !defined( VULKAN_HPP_CXX_MODULE )
#  include <type_traits>  // for std::underlying_type
#endif

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
    using BitsType = BitType;
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

  //=============
  //=== ENUMs ===
  //=============

  //=== VK_VERSION_1_0 ===

  // wrapper class for enum VkResult, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkResult.html
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
    eErrorValidationFailed                       = VK_ERROR_VALIDATION_FAILED,
    eErrorValidationFailedEXT                    = VK_ERROR_VALIDATION_FAILED_EXT,
    eErrorOutOfPoolMemory                        = VK_ERROR_OUT_OF_POOL_MEMORY,
    eErrorOutOfPoolMemoryKHR                     = VK_ERROR_OUT_OF_POOL_MEMORY_KHR,
    eErrorInvalidExternalHandle                  = VK_ERROR_INVALID_EXTERNAL_HANDLE,
    eErrorInvalidExternalHandleKHR               = VK_ERROR_INVALID_EXTERNAL_HANDLE_KHR,
    eErrorInvalidOpaqueCaptureAddress            = VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS,
    eErrorInvalidDeviceAddressEXT                = VK_ERROR_INVALID_DEVICE_ADDRESS_EXT,
    eErrorInvalidOpaqueCaptureAddressKHR         = VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR,
    eErrorFragmentation                          = VK_ERROR_FRAGMENTATION,
    eErrorFragmentationEXT                       = VK_ERROR_FRAGMENTATION_EXT,
    ePipelineCompileRequired                     = VK_PIPELINE_COMPILE_REQUIRED,
    ePipelineCompileRequiredEXT                  = VK_PIPELINE_COMPILE_REQUIRED_EXT,
    eErrorPipelineCompileRequiredEXT             = VK_ERROR_PIPELINE_COMPILE_REQUIRED_EXT,
    eErrorNotPermitted                           = VK_ERROR_NOT_PERMITTED,
    eErrorNotPermittedEXT                        = VK_ERROR_NOT_PERMITTED_EXT,
    eErrorNotPermittedKHR                        = VK_ERROR_NOT_PERMITTED_KHR,
    eErrorSurfaceLostKHR                         = VK_ERROR_SURFACE_LOST_KHR,
    eErrorNativeWindowInUseKHR                   = VK_ERROR_NATIVE_WINDOW_IN_USE_KHR,
    eSuboptimalKHR                               = VK_SUBOPTIMAL_KHR,
    eErrorOutOfDateKHR                           = VK_ERROR_OUT_OF_DATE_KHR,
    eErrorIncompatibleDisplayKHR                 = VK_ERROR_INCOMPATIBLE_DISPLAY_KHR,
    eErrorInvalidShaderNV                        = VK_ERROR_INVALID_SHADER_NV,
    eErrorImageUsageNotSupportedKHR              = VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR,
    eErrorVideoPictureLayoutNotSupportedKHR      = VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR,
    eErrorVideoProfileOperationNotSupportedKHR   = VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR,
    eErrorVideoProfileFormatNotSupportedKHR      = VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR,
    eErrorVideoProfileCodecNotSupportedKHR       = VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR,
    eErrorVideoStdVersionNotSupportedKHR         = VK_ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR,
    eErrorInvalidDrmFormatModifierPlaneLayoutEXT = VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT,
    eErrorPresentTimingQueueFullEXT              = VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT,
#if defined( VK_USE_PLATFORM_WIN32_KHR )
    eErrorFullScreenExclusiveModeLostEXT = VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT,
#endif /*VK_USE_PLATFORM_WIN32_KHR*/
    eThreadIdleKHR                     = VK_THREAD_IDLE_KHR,
    eThreadDoneKHR                     = VK_THREAD_DONE_KHR,
    eOperationDeferredKHR              = VK_OPERATION_DEFERRED_KHR,
    eOperationNotDeferredKHR           = VK_OPERATION_NOT_DEFERRED_KHR,
    eErrorInvalidVideoStdParametersKHR = VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR,
    eErrorCompressionExhaustedEXT      = VK_ERROR_COMPRESSION_EXHAUSTED_EXT,
    eIncompatibleShaderBinaryEXT       = VK_INCOMPATIBLE_SHADER_BINARY_EXT,
    eErrorIncompatibleShaderBinaryEXT  = VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT,
    ePipelineBinaryMissingKHR          = VK_PIPELINE_BINARY_MISSING_KHR,
    eErrorNotEnoughSpaceKHR            = VK_ERROR_NOT_ENOUGH_SPACE_KHR
  };

  // wrapper class for enum VkStructureType, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkStructureType.html
  enum class StructureType
  {
    eApplicationInfo                                         = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    eInstanceCreateInfo                                      = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    eDeviceQueueCreateInfo                                   = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    eDeviceCreateInfo                                        = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    eSubmitInfo                                              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    eMemoryAllocateInfo                                      = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    eMappedMemoryRange                                       = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
    eBindSparseInfo                                          = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
    eFenceCreateInfo                                         = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    eSemaphoreCreateInfo                                     = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    eEventCreateInfo                                         = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO,
    eQueryPoolCreateInfo                                     = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
    eBufferCreateInfo                                        = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    eBufferViewCreateInfo                                    = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
    eImageCreateInfo                                         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
    eImageViewCreateInfo                                     = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
    eShaderModuleCreateInfo                                  = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    ePipelineCacheCreateInfo                                 = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
    ePipelineShaderStageCreateInfo                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    ePipelineVertexInputStateCreateInfo                      = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    ePipelineInputAssemblyStateCreateInfo                    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
    ePipelineTessellationStateCreateInfo                     = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
    ePipelineViewportStateCreateInfo                         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
    ePipelineRasterizationStateCreateInfo                    = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
    ePipelineMultisampleStateCreateInfo                      = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
    ePipelineDepthStencilStateCreateInfo                     = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
    ePipelineColorBlendStateCreateInfo                       = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
    ePipelineDynamicStateCreateInfo                          = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
    eGraphicsPipelineCreateInfo                              = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
    eComputePipelineCreateInfo                               = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    ePipelineLayoutCreateInfo                                = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    eSamplerCreateInfo                                       = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
    eDescriptorSetLayoutCreateInfo                           = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    eDescriptorPoolCreateInfo                                = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    eDescriptorSetAllocateInfo                               = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    eWriteDescriptorSet                                      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    eCopyDescriptorSet                                       = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET,
    eFramebufferCreateInfo                                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
    eRenderPassCreateInfo                                    = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
    eCommandPoolCreateInfo                                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    eCommandBufferAllocateInfo                               = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    eCommandBufferInheritanceInfo                            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
    eCommandBufferBeginInfo                                  = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    eRenderPassBeginInfo                                     = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
    eBufferMemoryBarrier                                     = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
    eImageMemoryBarrier                                      = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
    eMemoryBarrier                                           = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    eLoaderInstanceCreateInfo                                = VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO,
    eLoaderDeviceCreateInfo                                  = VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO,
    eBindBufferMemoryInfo                                    = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
    eBindBufferMemoryInfoKHR                                 = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO_KHR,
    eBindImageMemoryInfo                                     = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,
    eBindImageMemoryInfoKHR                                  = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO_KHR,
    eMemoryDedicatedRequirements                             = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
    eMemoryDedicatedRequirementsKHR                          = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR,
    eMemoryDedicatedAllocateInfo                             = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
    eMemoryDedicatedAllocateInfoKHR                          = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR,
    eMemoryAllocateFlagsInfo                                 = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
    eMemoryAllocateFlagsInfoKHR                              = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_KHR,
    eDeviceGroupCommandBufferBeginInfo                       = VK_STRUCTURE_TYPE_DEVICE_GROUP_COMMAND_BUFFER_BEGIN_INFO,
    eDeviceGroupCommandBufferBeginInfoKHR                    = VK_STRUCTURE_TYPE_DEVICE_GROUP_COMMAND_BUFFER_BEGIN_INFO_KHR,
    eDeviceGroupSubmitInfo                                   = VK_STRUCTURE_TYPE_DEVICE_GROUP_SUBMIT_INFO,
    eDeviceGroupSubmitInfoKHR                                = VK_STRUCTURE_TYPE_DEVICE_GROUP_SUBMIT_INFO_KHR,
    eDeviceGroupBindSparseInfo                               = VK_STRUCTURE_TYPE_DEVICE_GROUP_BIND_SPARSE_INFO,
    eDeviceGroupBindSparseInfoKHR                            = VK_STRUCTURE_TYPE_DEVICE_GROUP_BIND_SPARSE_INFO_KHR,
    eBindBufferMemoryDeviceGroupInfo                         = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_DEVICE_GROUP_INFO,
    eBindBufferMemoryDeviceGroupInfoKHR                      = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_DEVICE_GROUP_INFO_KHR,
    eBindImageMemoryDeviceGroupInfo                          = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_DEVICE_GROUP_INFO,
    eBindImageMemoryDeviceGroupInfoKHR                       = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_DEVICE_GROUP_INFO_KHR,
    ePhysicalDeviceGroupProperties                           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES,
    ePhysicalDeviceGroupPropertiesKHR                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES_KHR,
    eDeviceGroupDeviceCreateInfo                             = VK_STRUCTURE_TYPE_DEVICE_GROUP_DEVICE_CREATE_INFO,
    eDeviceGroupDeviceCreateInfoKHR                          = VK_STRUCTURE_TYPE_DEVICE_GROUP_DEVICE_CREATE_INFO_KHR,
    eBufferMemoryRequirementsInfo2                           = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
    eBufferMemoryRequirementsInfo2KHR                        = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2_KHR,
    eImageMemoryRequirementsInfo2                            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
    eImageMemoryRequirementsInfo2KHR                         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2_KHR,
    eImageSparseMemoryRequirementsInfo2                      = VK_STRUCTURE_TYPE_IMAGE_SPARSE_MEMORY_REQUIREMENTS_INFO_2,
    eImageSparseMemoryRequirementsInfo2KHR                   = VK_STRUCTURE_TYPE_IMAGE_SPARSE_MEMORY_REQUIREMENTS_INFO_2_KHR,
    eMemoryRequirements2                                     = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
    eMemoryRequirements2KHR                                  = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR,
    eSparseImageMemoryRequirements2                          = VK_STRUCTURE_TYPE_SPARSE_IMAGE_MEMORY_REQUIREMENTS_2,
    eSparseImageMemoryRequirements2KHR                       = VK_STRUCTURE_TYPE_SPARSE_IMAGE_MEMORY_REQUIREMENTS_2_KHR,
    ePhysicalDeviceFeatures2                                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
    ePhysicalDeviceFeatures2KHR                              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR,
    ePhysicalDeviceProperties2                               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
    ePhysicalDeviceProperties2KHR                            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR,
    eFormatProperties2                                       = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
    eFormatProperties2KHR                                    = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2_KHR,
    eImageFormatProperties2                                  = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
    eImageFormatProperties2KHR                               = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2_KHR,
    ePhysicalDeviceImageFormatInfo2                          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
    ePhysicalDeviceImageFormatInfo2KHR                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2_KHR,
    eQueueFamilyProperties2                                  = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2,
    eQueueFamilyProperties2KHR                               = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2_KHR,
    ePhysicalDeviceMemoryProperties2                         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
    ePhysicalDeviceMemoryProperties2KHR                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR,
    eSparseImageFormatProperties2                            = VK_STRUCTURE_TYPE_SPARSE_IMAGE_FORMAT_PROPERTIES_2,
    eSparseImageFormatProperties2KHR                         = VK_STRUCTURE_TYPE_SPARSE_IMAGE_FORMAT_PROPERTIES_2_KHR,
    ePhysicalDeviceSparseImageFormatInfo2                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SPARSE_IMAGE_FORMAT_INFO_2,
    ePhysicalDeviceSparseImageFormatInfo2KHR                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SPARSE_IMAGE_FORMAT_INFO_2_KHR,
    eImageViewUsageCreateInfo                                = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
    eImageViewUsageCreateInfoKHR                             = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO_KHR,
    eProtectedSubmitInfo                                     = VK_STRUCTURE_TYPE_PROTECTED_SUBMIT_INFO,
    ePhysicalDeviceProtectedMemoryFeatures                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES,
    ePhysicalDeviceProtectedMemoryProperties                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_PROPERTIES,
    eDeviceQueueInfo2                                        = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2,
    ePhysicalDeviceExternalImageFormatInfo                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
    ePhysicalDeviceExternalImageFormatInfoKHR                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO_KHR,
    eExternalImageFormatProperties                           = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES,
    eExternalImageFormatPropertiesKHR                        = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES_KHR,
    ePhysicalDeviceExternalBufferInfo                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
    ePhysicalDeviceExternalBufferInfoKHR                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO_KHR,
    eExternalBufferProperties                                = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES,
    eExternalBufferPropertiesKHR                             = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES_KHR,
    ePhysicalDeviceIdProperties                              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
    ePhysicalDeviceIdPropertiesKHR                           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES_KHR,
    eExternalMemoryBufferCreateInfo                          = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
    eExternalMemoryBufferCreateInfoKHR                       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO_KHR,
    eExternalMemoryImageCreateInfo                           = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
    eExternalMemoryImageCreateInfoKHR                        = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO_KHR,
    eExportMemoryAllocateInfo                                = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
    eExportMemoryAllocateInfoKHR                             = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR,
    ePhysicalDeviceExternalFenceInfo                         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_FENCE_INFO,
    ePhysicalDeviceExternalFenceInfoKHR                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_FENCE_INFO_KHR,
    eExternalFenceProperties                                 = VK_STRUCTURE_TYPE_EXTERNAL_FENCE_PROPERTIES,
    eExternalFencePropertiesKHR                              = VK_STRUCTURE_TYPE_EXTERNAL_FENCE_PROPERTIES_KHR,
    eExportFenceCreateInfo                                   = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO,
    eExportFenceCreateInfoKHR                                = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO_KHR,
    eExportSemaphoreCreateInfo                               = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
    eExportSemaphoreCreateInfoKHR                            = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR,
    ePhysicalDeviceExternalSemaphoreInfo                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
    ePhysicalDeviceExternalSemaphoreInfoKHR                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO_KHR,
    eExternalSemaphoreProperties                             = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES,
    eExternalSemaphorePropertiesKHR                          = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES_KHR,
    ePhysicalDeviceSubgroupProperties                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    ePhysicalDevice16BitStorageFeatures                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
    ePhysicalDevice16BitStorageFeaturesKHR                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR,
    ePhysicalDeviceVariablePointersFeatures                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES,
    ePhysicalDeviceVariablePointerFeatures                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTER_FEATURES,
    ePhysicalDeviceVariablePointersFeaturesKHR               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES_KHR,
    ePhysicalDeviceVariablePointerFeaturesKHR                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTER_FEATURES_KHR,
    eDescriptorUpdateTemplateCreateInfo                      = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO,
    eDescriptorUpdateTemplateCreateInfoKHR                   = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO_KHR,
    ePhysicalDeviceMaintenance3Properties                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES,
    ePhysicalDeviceMaintenance3PropertiesKHR                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES_KHR,
    eDescriptorSetLayoutSupport                              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT,
    eDescriptorSetLayoutSupportKHR                           = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT_KHR,
    eSamplerYcbcrConversionCreateInfo                        = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
    eSamplerYcbcrConversionCreateInfoKHR                     = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO_KHR,
    eSamplerYcbcrConversionInfo                              = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO,
    eSamplerYcbcrConversionInfoKHR                           = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO_KHR,
    eBindImagePlaneMemoryInfo                                = VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO,
    eBindImagePlaneMemoryInfoKHR                             = VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO_KHR,
    eImagePlaneMemoryRequirementsInfo                        = VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO,
    eImagePlaneMemoryRequirementsInfoKHR                     = VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO_KHR,
    ePhysicalDeviceSamplerYcbcrConversionFeatures            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES,
    ePhysicalDeviceSamplerYcbcrConversionFeaturesKHR         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES_KHR,
    eSamplerYcbcrConversionImageFormatProperties             = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES,
    eSamplerYcbcrConversionImageFormatPropertiesKHR          = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES_KHR,
    eDeviceGroupRenderPassBeginInfo                          = VK_STRUCTURE_TYPE_DEVICE_GROUP_RENDER_PASS_BEGIN_INFO,
    eDeviceGroupRenderPassBeginInfoKHR                       = VK_STRUCTURE_TYPE_DEVICE_GROUP_RENDER_PASS_BEGIN_INFO_KHR,
    ePhysicalDevicePointClippingProperties                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_POINT_CLIPPING_PROPERTIES,
    ePhysicalDevicePointClippingPropertiesKHR                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_POINT_CLIPPING_PROPERTIES_KHR,
    eRenderPassInputAttachmentAspectCreateInfo               = VK_STRUCTURE_TYPE_RENDER_PASS_INPUT_ATTACHMENT_ASPECT_CREATE_INFO,
    eRenderPassInputAttachmentAspectCreateInfoKHR            = VK_STRUCTURE_TYPE_RENDER_PASS_INPUT_ATTACHMENT_ASPECT_CREATE_INFO_KHR,
    ePipelineTessellationDomainOriginStateCreateInfo         = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_DOMAIN_ORIGIN_STATE_CREATE_INFO,
    ePipelineTessellationDomainOriginStateCreateInfoKHR      = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_DOMAIN_ORIGIN_STATE_CREATE_INFO_KHR,
    eRenderPassMultiviewCreateInfo                           = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO,
    eRenderPassMultiviewCreateInfoKHR                        = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO_KHR,
    ePhysicalDeviceMultiviewFeatures                         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES,
    ePhysicalDeviceMultiviewFeaturesKHR                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES_KHR,
    ePhysicalDeviceMultiviewProperties                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES,
    ePhysicalDeviceMultiviewPropertiesKHR                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES_KHR,
    ePhysicalDeviceShaderDrawParametersFeatures              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES,
    ePhysicalDeviceShaderDrawParameterFeatures               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETER_FEATURES,
    ePhysicalDeviceVulkan11Features                          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
    ePhysicalDeviceVulkan11Properties                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES,
    ePhysicalDeviceVulkan12Features                          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
    ePhysicalDeviceVulkan12Properties                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES,
    eImageFormatListCreateInfo                               = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
    eImageFormatListCreateInfoKHR                            = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO_KHR,
    ePhysicalDeviceDriverProperties                          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
    ePhysicalDeviceDriverPropertiesKHR                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES_KHR,
    ePhysicalDeviceVulkanMemoryModelFeatures                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES,
    ePhysicalDeviceVulkanMemoryModelFeaturesKHR              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES_KHR,
    ePhysicalDeviceHostQueryResetFeatures                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES,
    ePhysicalDeviceHostQueryResetFeaturesEXT                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES_EXT,
    ePhysicalDeviceTimelineSemaphoreFeatures                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
    ePhysicalDeviceTimelineSemaphoreFeaturesKHR              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR,
    ePhysicalDeviceTimelineSemaphoreProperties               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES,
    ePhysicalDeviceTimelineSemaphorePropertiesKHR            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES_KHR,
    eSemaphoreTypeCreateInfo                                 = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
    eSemaphoreTypeCreateInfoKHR                              = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR,
    eTimelineSemaphoreSubmitInfo                             = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
    eTimelineSemaphoreSubmitInfoKHR                          = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR,
    eSemaphoreWaitInfo                                       = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
    eSemaphoreWaitInfoKHR                                    = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO_KHR,
    eSemaphoreSignalInfo                                     = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
    eSemaphoreSignalInfoKHR                                  = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO_KHR,
    ePhysicalDeviceBufferDeviceAddressFeatures               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
    ePhysicalDeviceBufferDeviceAddressFeaturesKHR            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR,
    eBufferDeviceAddressInfo                                 = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
    eBufferDeviceAddressInfoEXT                              = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_EXT,
    eBufferDeviceAddressInfoKHR                              = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR,
    eBufferOpaqueCaptureAddressCreateInfo                    = VK_STRUCTURE_TYPE_BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO,
    eBufferOpaqueCaptureAddressCreateInfoKHR                 = VK_STRUCTURE_TYPE_BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO_KHR,
    eMemoryOpaqueCaptureAddressAllocateInfo                  = VK_STRUCTURE_TYPE_MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO,
    eMemoryOpaqueCaptureAddressAllocateInfoKHR               = VK_STRUCTURE_TYPE_MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO_KHR,
    eDeviceMemoryOpaqueCaptureAddressInfo                    = VK_STRUCTURE_TYPE_DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO,
    eDeviceMemoryOpaqueCaptureAddressInfoKHR                 = VK_STRUCTURE_TYPE_DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO_KHR,
    ePhysicalDevice8BitStorageFeatures                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
    ePhysicalDevice8BitStorageFeaturesKHR                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR,
    ePhysicalDeviceShaderAtomicInt64Features                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES,
    ePhysicalDeviceShaderAtomicInt64FeaturesKHR              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES_KHR,
    ePhysicalDeviceShaderFloat16Int8Features                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
    ePhysicalDeviceShaderFloat16Int8FeaturesKHR              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR,
    ePhysicalDeviceFloat16Int8FeaturesKHR                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR,
    ePhysicalDeviceFloatControlsProperties                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES,
    ePhysicalDeviceFloatControlsPropertiesKHR                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES_KHR,
    eDescriptorSetLayoutBindingFlagsCreateInfo               = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
    eDescriptorSetLayoutBindingFlagsCreateInfoEXT            = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT,
    ePhysicalDeviceDescriptorIndexingFeatures                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
    ePhysicalDeviceDescriptorIndexingFeaturesEXT             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT,
    ePhysicalDeviceDescriptorIndexingProperties              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES,
    ePhysicalDeviceDescriptorIndexingPropertiesEXT           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES_EXT,
    eDescriptorSetVariableDescriptorCountAllocateInfo        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO,
    eDescriptorSetVariableDescriptorCountAllocateInfoEXT     = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT,
    eDescriptorSetVariableDescriptorCountLayoutSupport       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_LAYOUT_SUPPORT,
    eDescriptorSetVariableDescriptorCountLayoutSupportEXT    = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_LAYOUT_SUPPORT_EXT,
    ePhysicalDeviceScalarBlockLayoutFeatures                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES,
    ePhysicalDeviceScalarBlockLayoutFeaturesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES_EXT,
    ePhysicalDeviceSamplerFilterMinmaxProperties             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_FILTER_MINMAX_PROPERTIES,
    ePhysicalDeviceSamplerFilterMinmaxPropertiesEXT          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_FILTER_MINMAX_PROPERTIES_EXT,
    eSamplerReductionModeCreateInfo                          = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO,
    eSamplerReductionModeCreateInfoEXT                       = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO_EXT,
    ePhysicalDeviceUniformBufferStandardLayoutFeatures       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES,
    ePhysicalDeviceUniformBufferStandardLayoutFeaturesKHR    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES_KHR,
    ePhysicalDeviceShaderSubgroupExtendedTypesFeatures       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES,
    ePhysicalDeviceShaderSubgroupExtendedTypesFeaturesKHR    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES_KHR,
    eAttachmentDescription2                                  = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
    eAttachmentDescription2KHR                               = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2_KHR,
    eAttachmentReference2                                    = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2,
    eAttachmentReference2KHR                                 = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2_KHR,
    eSubpassDescription2                                     = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2,
    eSubpassDescription2KHR                                  = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2_KHR,
    eSubpassDependency2                                      = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2,
    eSubpassDependency2KHR                                   = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2_KHR,
    eRenderPassCreateInfo2                                   = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2,
    eRenderPassCreateInfo2KHR                                = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2_KHR,
    eSubpassBeginInfo                                        = VK_STRUCTURE_TYPE_SUBPASS_BEGIN_INFO,
    eSubpassBeginInfoKHR                                     = VK_STRUCTURE_TYPE_SUBPASS_BEGIN_INFO_KHR,
    eSubpassEndInfo                                          = VK_STRUCTURE_TYPE_SUBPASS_END_INFO,
    eSubpassEndInfoKHR                                       = VK_STRUCTURE_TYPE_SUBPASS_END_INFO_KHR,
    ePhysicalDeviceDepthStencilResolveProperties             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES,
    ePhysicalDeviceDepthStencilResolvePropertiesKHR          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES_KHR,
    eSubpassDescriptionDepthStencilResolve                   = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE,
    eSubpassDescriptionDepthStencilResolveKHR                = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE_KHR,
    eImageStencilUsageCreateInfo                             = VK_STRUCTURE_TYPE_IMAGE_STENCIL_USAGE_CREATE_INFO,
    eImageStencilUsageCreateInfoEXT                          = VK_STRUCTURE_TYPE_IMAGE_STENCIL_USAGE_CREATE_INFO_EXT,
    ePhysicalDeviceImagelessFramebufferFeatures              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES,
    ePhysicalDeviceImagelessFramebufferFeaturesKHR           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES_KHR,
    eFramebufferAttachmentsCreateInfo                        = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENTS_CREATE_INFO,
    eFramebufferAttachmentsCreateInfoKHR                     = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENTS_CREATE_INFO_KHR,
    eFramebufferAttachmentImageInfo                          = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENT_IMAGE_INFO,
    eFramebufferAttachmentImageInfoKHR                       = VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENT_IMAGE_INFO_KHR,
    eRenderPassAttachmentBeginInfo                           = VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO,
    eRenderPassAttachmentBeginInfoKHR                        = VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO_KHR,
    ePhysicalDeviceSeparateDepthStencilLayoutsFeatures       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES,
    ePhysicalDeviceSeparateDepthStencilLayoutsFeaturesKHR    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES_KHR,
    eAttachmentReferenceStencilLayout                        = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_STENCIL_LAYOUT,
    eAttachmentReferenceStencilLayoutKHR                     = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_STENCIL_LAYOUT_KHR,
    eAttachmentDescriptionStencilLayout                      = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_STENCIL_LAYOUT,
    eAttachmentDescriptionStencilLayoutKHR                   = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_STENCIL_LAYOUT_KHR,
    ePhysicalDeviceVulkan13Features                          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
    ePhysicalDeviceVulkan13Properties                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES,
    ePhysicalDeviceToolProperties                            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TOOL_PROPERTIES,
    ePhysicalDeviceToolPropertiesEXT                         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TOOL_PROPERTIES_EXT,
    ePhysicalDevicePrivateDataFeatures                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES,
    ePhysicalDevicePrivateDataFeaturesEXT                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES_EXT,
    eDevicePrivateDataCreateInfo                             = VK_STRUCTURE_TYPE_DEVICE_PRIVATE_DATA_CREATE_INFO,
    eDevicePrivateDataCreateInfoEXT                          = VK_STRUCTURE_TYPE_DEVICE_PRIVATE_DATA_CREATE_INFO_EXT,
    ePrivateDataSlotCreateInfo                               = VK_STRUCTURE_TYPE_PRIVATE_DATA_SLOT_CREATE_INFO,
    ePrivateDataSlotCreateInfoEXT                            = VK_STRUCTURE_TYPE_PRIVATE_DATA_SLOT_CREATE_INFO_EXT,
    eMemoryBarrier2                                          = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
    eMemoryBarrier2KHR                                       = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR,
    eBufferMemoryBarrier2                                    = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
    eBufferMemoryBarrier2KHR                                 = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR,
    eImageMemoryBarrier2                                     = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
    eImageMemoryBarrier2KHR                                  = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR,
    eDependencyInfo                                          = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
    eDependencyInfoKHR                                       = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
    eSubmitInfo2                                             = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
    eSubmitInfo2KHR                                          = VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR,
    eSemaphoreSubmitInfo                                     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
    eSemaphoreSubmitInfoKHR                                  = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR,
    eCommandBufferSubmitInfo                                 = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
    eCommandBufferSubmitInfoKHR                              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR,
    ePhysicalDeviceSynchronization2Features                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
    ePhysicalDeviceSynchronization2FeaturesKHR               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR,
    eCopyBufferInfo2                                         = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
    eCopyBufferInfo2KHR                                      = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2_KHR,
    eCopyImageInfo2                                          = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2,
    eCopyImageInfo2KHR                                       = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2_KHR,
    eCopyBufferToImageInfo2                                  = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
    eCopyBufferToImageInfo2KHR                               = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2_KHR,
    eCopyImageToBufferInfo2                                  = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
    eCopyImageToBufferInfo2KHR                               = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2_KHR,
    eBufferCopy2                                             = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
    eBufferCopy2KHR                                          = VK_STRUCTURE_TYPE_BUFFER_COPY_2_KHR,
    eImageCopy2                                              = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
    eImageCopy2KHR                                           = VK_STRUCTURE_TYPE_IMAGE_COPY_2_KHR,
    eBufferImageCopy2                                        = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
    eBufferImageCopy2KHR                                     = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2_KHR,
    ePhysicalDeviceTextureCompressionAstcHdrFeatures         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES,
    ePhysicalDeviceTextureCompressionAstcHdrFeaturesEXT      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES_EXT,
    eFormatProperties3                                       = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3,
    eFormatProperties3KHR                                    = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3_KHR,
    ePhysicalDeviceMaintenance4Features                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES,
    ePhysicalDeviceMaintenance4FeaturesKHR                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES_KHR,
    ePhysicalDeviceMaintenance4Properties                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES,
    ePhysicalDeviceMaintenance4PropertiesKHR                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES_KHR,
    eDeviceBufferMemoryRequirements                          = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS,
    eDeviceBufferMemoryRequirementsKHR                       = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS_KHR,
    eDeviceImageMemoryRequirements                           = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS,
    eDeviceImageMemoryRequirementsKHR                        = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS_KHR,
    ePipelineCreationFeedbackCreateInfo                      = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO,
    ePipelineCreationFeedbackCreateInfoEXT                   = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO_EXT,
    ePhysicalDeviceShaderTerminateInvocationFeatures         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES,
    ePhysicalDeviceShaderTerminateInvocationFeaturesKHR      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES_KHR,
    ePhysicalDeviceShaderDemoteToHelperInvocationFeatures    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES,
    ePhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES_EXT,
    ePhysicalDevicePipelineCreationCacheControlFeatures      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES,
    ePhysicalDevicePipelineCreationCacheControlFeaturesEXT   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES_EXT,
    ePhysicalDeviceZeroInitializeWorkgroupMemoryFeatures     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES,
    ePhysicalDeviceZeroInitializeWorkgroupMemoryFeaturesKHR  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES_KHR,
    ePhysicalDeviceImageRobustnessFeatures                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES,
    ePhysicalDeviceImageRobustnessFeaturesEXT                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES_EXT,
    ePhysicalDeviceSubgroupSizeControlProperties             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES,
    ePhysicalDeviceSubgroupSizeControlPropertiesEXT          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT,
    ePipelineShaderStageRequiredSubgroupSizeCreateInfo       = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
    ePipelineShaderStageRequiredSubgroupSizeCreateInfoEXT    = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
    eShaderRequiredSubgroupSizeCreateInfoEXT                 = VK_STRUCTURE_TYPE_SHADER_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
    ePhysicalDeviceSubgroupSizeControlFeatures               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
    ePhysicalDeviceSubgroupSizeControlFeaturesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT,
    ePhysicalDeviceInlineUniformBlockFeatures                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES,
    ePhysicalDeviceInlineUniformBlockFeaturesEXT             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES_EXT,
    ePhysicalDeviceInlineUniformBlockProperties              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_PROPERTIES,
    ePhysicalDeviceInlineUniformBlockPropertiesEXT           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_PROPERTIES_EXT,
    eWriteDescriptorSetInlineUniformBlock                    = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK,
    eWriteDescriptorSetInlineUniformBlockEXT                 = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK_EXT,
    eDescriptorPoolInlineUniformBlockCreateInfo              = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_INLINE_UNIFORM_BLOCK_CREATE_INFO,
    eDescriptorPoolInlineUniformBlockCreateInfoEXT           = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_INLINE_UNIFORM_BLOCK_CREATE_INFO_EXT,
    ePhysicalDeviceShaderIntegerDotProductFeatures           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
    ePhysicalDeviceShaderIntegerDotProductFeaturesKHR        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR,
    ePhysicalDeviceShaderIntegerDotProductProperties         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES,
    ePhysicalDeviceShaderIntegerDotProductPropertiesKHR      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES_KHR,
    ePhysicalDeviceTexelBufferAlignmentProperties            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_PROPERTIES,
    ePhysicalDeviceTexelBufferAlignmentPropertiesEXT         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXEL_BUFFER_ALIGNMENT_PROPERTIES_EXT,
    eBlitImageInfo2                                          = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
    eBlitImageInfo2KHR                                       = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2_KHR,
    eResolveImageInfo2                                       = VK_STRUCTURE_TYPE_RESOLVE_IMAGE_INFO_2,
    eResolveImageInfo2KHR                                    = VK_STRUCTURE_TYPE_RESOLVE_IMAGE_INFO_2_KHR,
    eImageBlit2                                              = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
    eImageBlit2KHR                                           = VK_STRUCTURE_TYPE_IMAGE_BLIT_2_KHR,
    eImageResolve2                                           = VK_STRUCTURE_TYPE_IMAGE_RESOLVE_2,
    eImageResolve2KHR                                        = VK_STRUCTURE_TYPE_IMAGE_RESOLVE_2_KHR,
    eRenderingInfo                                           = VK_STRUCTURE_TYPE_RENDERING_INFO,
    eRenderingInfoKHR                                        = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
    eRenderingAttachmentInfo                                 = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
    eRenderingAttachmentInfoKHR                              = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
    ePipelineRenderingCreateInfo                             = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
    ePipelineRenderingCreateInfoKHR                          = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
    ePhysicalDeviceDynamicRenderingFeatures                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
    ePhysicalDeviceDynamicRenderingFeaturesKHR               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR,
    eCommandBufferInheritanceRenderingInfo                   = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO,
    eCommandBufferInheritanceRenderingInfoKHR                = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO_KHR,
    ePhysicalDeviceVulkan14Features                          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES,
    ePhysicalDeviceVulkan14Properties                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_PROPERTIES,
    eDeviceQueueGlobalPriorityCreateInfo                     = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO,
    eDeviceQueueGlobalPriorityCreateInfoEXT                  = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_EXT,
    eDeviceQueueGlobalPriorityCreateInfoKHR                  = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR,
    ePhysicalDeviceGlobalPriorityQueryFeatures               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GLOBAL_PRIORITY_QUERY_FEATURES,
    ePhysicalDeviceGlobalPriorityQueryFeaturesKHR            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GLOBAL_PRIORITY_QUERY_FEATURES_KHR,
    ePhysicalDeviceGlobalPriorityQueryFeaturesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GLOBAL_PRIORITY_QUERY_FEATURES_EXT,
    eQueueFamilyGlobalPriorityProperties                     = VK_STRUCTURE_TYPE_QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES,
    eQueueFamilyGlobalPriorityPropertiesKHR                  = VK_STRUCTURE_TYPE_QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES_KHR,
    eQueueFamilyGlobalPriorityPropertiesEXT                  = VK_STRUCTURE_TYPE_QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES_EXT,
    ePhysicalDeviceIndexTypeUint8Features                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES,
    ePhysicalDeviceIndexTypeUint8FeaturesEXT                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT,
    ePhysicalDeviceIndexTypeUint8FeaturesKHR                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_KHR,
    eMemoryMapInfo                                           = VK_STRUCTURE_TYPE_MEMORY_MAP_INFO,
    eMemoryMapInfoKHR                                        = VK_STRUCTURE_TYPE_MEMORY_MAP_INFO_KHR,
    eMemoryUnmapInfo                                         = VK_STRUCTURE_TYPE_MEMORY_UNMAP_INFO,
    eMemoryUnmapInfoKHR                                      = VK_STRUCTURE_TYPE_MEMORY_UNMAP_INFO_KHR,
    ePhysicalDeviceMaintenance5Features                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES,
    ePhysicalDeviceMaintenance5FeaturesKHR                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR,
    ePhysicalDeviceMaintenance5Properties                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_PROPERTIES,
    ePhysicalDeviceMaintenance5PropertiesKHR                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_PROPERTIES_KHR,
    eDeviceImageSubresourceInfo                              = VK_STRUCTURE_TYPE_DEVICE_IMAGE_SUBRESOURCE_INFO,
    eDeviceImageSubresourceInfoKHR                           = VK_STRUCTURE_TYPE_DEVICE_IMAGE_SUBRESOURCE_INFO_KHR,
    eSubresourceLayout2                                      = VK_STRUCTURE_TYPE_SUBRESOURCE_LAYOUT_2,
    eSubresourceLayout2EXT                                   = VK_STRUCTURE_TYPE_SUBRESOURCE_LAYOUT_2_EXT,
    eSubresourceLayout2KHR                                   = VK_STRUCTURE_TYPE_SUBRESOURCE_LAYOUT_2_KHR,
    eImageSubresource2                                       = VK_STRUCTURE_TYPE_IMAGE_SUBRESOURCE_2,
    eImageSubresource2EXT                                    = VK_STRUCTURE_TYPE_IMAGE_SUBRESOURCE_2_EXT,
    eImageSubresource2KHR                                    = VK_STRUCTURE_TYPE_IMAGE_SUBRESOURCE_2_KHR,
    eBufferUsageFlags2CreateInfo                             = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO,
    eBufferUsageFlags2CreateInfoKHR                          = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR,
    ePhysicalDeviceMaintenance6Features                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_6_FEATURES,
    ePhysicalDeviceMaintenance6FeaturesKHR                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_6_FEATURES_KHR,
    ePhysicalDeviceMaintenance6Properties                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_6_PROPERTIES,
    ePhysicalDeviceMaintenance6PropertiesKHR                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_6_PROPERTIES_KHR,
    eBindMemoryStatus                                        = VK_STRUCTURE_TYPE_BIND_MEMORY_STATUS,
    eBindMemoryStatusKHR                                     = VK_STRUCTURE_TYPE_BIND_MEMORY_STATUS_KHR,
    ePhysicalDeviceHostImageCopyFeatures                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES,
    ePhysicalDeviceHostImageCopyFeaturesEXT                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT,
    ePhysicalDeviceHostImageCopyProperties                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_PROPERTIES,
    ePhysicalDeviceHostImageCopyPropertiesEXT                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_PROPERTIES_EXT,
    eMemoryToImageCopy                                       = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY,
    eMemoryToImageCopyEXT                                    = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY_EXT,
    eImageToMemoryCopy                                       = VK_STRUCTURE_TYPE_IMAGE_TO_MEMORY_COPY,
    eImageToMemoryCopyEXT                                    = VK_STRUCTURE_TYPE_IMAGE_TO_MEMORY_COPY_EXT,
    eCopyImageToMemoryInfo                                   = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_MEMORY_INFO,
    eCopyImageToMemoryInfoEXT                                = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_MEMORY_INFO_EXT,
    eCopyMemoryToImageInfo                                   = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO,
    eCopyMemoryToImageInfoEXT                                = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO_EXT,
    eHostImageLayoutTransitionInfo                           = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO,
    eHostImageLayoutTransitionInfoEXT                        = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO_EXT,
    eCopyImageToImageInfo                                    = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_IMAGE_INFO,
    eCopyImageToImageInfoEXT                                 = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_IMAGE_INFO_EXT,
    eSubresourceHostMemcpySize                               = VK_STRUCTURE_TYPE_SUBRESOURCE_HOST_MEMCPY_SIZE,
    eSubresourceHostMemcpySizeEXT                            = VK_STRUCTURE_TYPE_SUBRESOURCE_HOST_MEMCPY_SIZE_EXT,
    eHostImageCopyDevicePerformanceQuery                     = VK_STRUCTURE_TYPE_HOST_IMAGE_COPY_DEVICE_PERFORMANCE_QUERY,
    eHostImageCopyDevicePerformanceQueryEXT                  = VK_STRUCTURE_TYPE_HOST_IMAGE_COPY_DEVICE_PERFORMANCE_QUERY_EXT,
    ePhysicalDeviceShaderSubgroupRotateFeatures              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_ROTATE_FEATURES,
    ePhysicalDeviceShaderSubgroupRotateFeaturesKHR           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_ROTATE_FEATURES_KHR,
    ePhysicalDeviceShaderFloatControls2Features              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES,
    ePhysicalDeviceShaderFloatControls2FeaturesKHR           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES_KHR,
    ePhysicalDeviceShaderExpectAssumeFeatures                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_EXPECT_ASSUME_FEATURES,
    ePhysicalDeviceShaderExpectAssumeFeaturesKHR             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_EXPECT_ASSUME_FEATURES_KHR,
    ePipelineCreateFlags2CreateInfo                          = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
    ePipelineCreateFlags2CreateInfoKHR                       = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR,
    ePhysicalDevicePushDescriptorProperties                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES,
    ePhysicalDevicePushDescriptorPropertiesKHR               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR,
    eBindDescriptorSetsInfo                                  = VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO,
    eBindDescriptorSetsInfoKHR                               = VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO_KHR,
    ePushConstantsInfo                                       = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
    ePushConstantsInfoKHR                                    = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO_KHR,
    ePushDescriptorSetInfo                                   = VK_STRUCTURE_TYPE_PUSH_DESCRIPTOR_SET_INFO,
    ePushDescriptorSetInfoKHR                                = VK_STRUCTURE_TYPE_PUSH_DESCRIPTOR_SET_INFO_KHR,
    ePushDescriptorSetWithTemplateInfo                       = VK_STRUCTURE_TYPE_PUSH_DESCRIPTOR_SET_WITH_TEMPLATE_INFO,
    ePushDescriptorSetWithTemplateInfoKHR                    = VK_STRUCTURE_TYPE_PUSH_DESCRIPTOR_SET_WITH_TEMPLATE_INFO_KHR,
    ePhysicalDevicePipelineProtectedAccessFeatures           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_PROTECTED_ACCESS_FEATURES,
    ePhysicalDevicePipelineProtectedAccessFeaturesEXT        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_PROTECTED_ACCESS_FEATURES_EXT,
    ePipelineRobustnessCreateInfo                            = VK_STRUCTURE_TYPE_PIPELINE_ROBUSTNESS_CREATE_INFO,
    ePipelineRobustnessCreateInfoEXT                         = VK_STRUCTURE_TYPE_PIPELINE_ROBUSTNESS_CREATE_INFO_EXT,
    ePhysicalDevicePipelineRobustnessFeatures                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES,
    ePhysicalDevicePipelineRobustnessFeaturesEXT             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES_EXT,
    ePhysicalDevicePipelineRobustnessProperties              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_PROPERTIES,
    ePhysicalDevicePipelineRobustnessPropertiesEXT           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_PROPERTIES_EXT,
    ePhysicalDeviceLineRasterizationFeatures                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES,
    ePhysicalDeviceLineRasterizationFeaturesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_EXT,
    ePhysicalDeviceLineRasterizationFeaturesKHR              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_KHR,
    ePipelineRasterizationLineStateCreateInfo                = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO,
    ePipelineRasterizationLineStateCreateInfoEXT             = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO_EXT,
    ePipelineRasterizationLineStateCreateInfoKHR             = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO_KHR,
    ePhysicalDeviceLineRasterizationProperties               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_PROPERTIES,
    ePhysicalDeviceLineRasterizationPropertiesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_PROPERTIES_EXT,
    ePhysicalDeviceLineRasterizationPropertiesKHR            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_PROPERTIES_KHR,
    ePhysicalDeviceVertexAttributeDivisorProperties          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_PROPERTIES,
    ePhysicalDeviceVertexAttributeDivisorPropertiesKHR       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_PROPERTIES_KHR,
    ePipelineVertexInputDivisorStateCreateInfo               = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO,
    ePipelineVertexInputDivisorStateCreateInfoEXT            = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT,
    ePipelineVertexInputDivisorStateCreateInfoKHR            = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_KHR,
    ePhysicalDeviceVertexAttributeDivisorFeatures            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES,
    ePhysicalDeviceVertexAttributeDivisorFeaturesEXT         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT,
    ePhysicalDeviceVertexAttributeDivisorFeaturesKHR         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_KHR,
    eRenderingAreaInfo                                       = VK_STRUCTURE_TYPE_RENDERING_AREA_INFO,
    eRenderingAreaInfoKHR                                    = VK_STRUCTURE_TYPE_RENDERING_AREA_INFO_KHR,
    ePhysicalDeviceDynamicRenderingLocalReadFeatures         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES,
    ePhysicalDeviceDynamicRenderingLocalReadFeaturesKHR      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES_KHR,
    eRenderingAttachmentLocationInfo                         = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO,
    eRenderingAttachmentLocationInfoKHR                      = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO_KHR,
    eRenderingInputAttachmentIndexInfo                       = VK_STRUCTURE_TYPE_RENDERING_INPUT_ATTACHMENT_INDEX_INFO,
    eRenderingInputAttachmentIndexInfoKHR                    = VK_STRUCTURE_TYPE_RENDERING_INPUT_ATTACHMENT_INDEX_INFO_KHR,
    eSwapchainCreateInfoKHR                                  = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
    ePresentInfoKHR                                          = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
    eDeviceGroupPresentCapabilitiesKHR                       = VK_STRUCTURE_TYPE_DEVICE_GROUP_PRESENT_CAPABILITIES_KHR,
    eImageSwapchainCreateInfoKHR                             = VK_STRUCTURE_TYPE_IMAGE_SWAPCHAIN_CREATE_INFO_KHR,
    eBindImageMemorySwapchainInfoKHR                         = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_SWAPCHAIN_INFO_KHR,
    eAcquireNextImageInfoKHR                                 = VK_STRUCTURE_TYPE_ACQUIRE_NEXT_IMAGE_INFO_KHR,
    eDeviceGroupPresentInfoKHR                               = VK_STRUCTURE_TYPE_DEVICE_GROUP_PRESENT_INFO_KHR,
    eDeviceGroupSwapchainCreateInfoKHR                       = VK_STRUCTURE_TYPE_DEVICE_GROUP_SWAPCHAIN_CREATE_INFO_KHR,
    eDisplayModeCreateInfoKHR                                = VK_STRUCTURE_TYPE_DISPLAY_MODE_CREATE_INFO_KHR,
    eDisplaySurfaceCreateInfoKHR                             = VK_STRUCTURE_TYPE_DISPLAY_SURFACE_CREATE_INFO_KHR,
    eDisplayPresentInfoKHR                                   = VK_STRUCTURE_TYPE_DISPLAY_PRESENT_INFO_KHR,
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
    eCuModuleTexturingModeCreateInfoNVX              = VK_STRUCTURE_TYPE_CU_MODULE_TEXTURING_MODE_CREATE_INFO_NVX,
    eImageViewHandleInfoNVX                          = VK_STRUCTURE_TYPE_IMAGE_VIEW_HANDLE_INFO_NVX,
    eImageViewAddressPropertiesNVX                   = VK_STRUCTURE_TYPE_IMAGE_VIEW_ADDRESS_PROPERTIES_NVX,
    eVideoEncodeH264CapabilitiesKHR                  = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_CAPABILITIES_KHR,
    eVideoEncodeH264SessionParametersCreateInfoKHR   = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR,
    eVideoEncodeH264SessionParametersAddInfoKHR      = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR,
    eVideoEncodeH264PictureInfoKHR                   = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PICTURE_INFO_KHR,
    eVideoEncodeH264DpbSlotInfoKHR                   = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_DPB_SLOT_INFO_KHR,
    eVideoEncodeH264NaluSliceInfoKHR                 = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_NALU_SLICE_INFO_KHR,
    eVideoEncodeH264GopRemainingFrameInfoKHR         = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_GOP_REMAINING_FRAME_INFO_KHR,
    eVideoEncodeH264ProfileInfoKHR                   = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PROFILE_INFO_KHR,
    eVideoEncodeH264RateControlInfoKHR               = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_RATE_CONTROL_INFO_KHR,
    eVideoEncodeH264RateControlLayerInfoKHR          = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_RATE_CONTROL_LAYER_INFO_KHR,
    eVideoEncodeH264SessionCreateInfoKHR             = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_CREATE_INFO_KHR,
    eVideoEncodeH264QualityLevelPropertiesKHR        = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_QUALITY_LEVEL_PROPERTIES_KHR,
    eVideoEncodeH264SessionParametersGetInfoKHR      = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_GET_INFO_KHR,
    eVideoEncodeH264SessionParametersFeedbackInfoKHR = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_FEEDBACK_INFO_KHR,
    eVideoEncodeH265CapabilitiesKHR                  = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_CAPABILITIES_KHR,
    eVideoEncodeH265SessionParametersCreateInfoKHR   = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_SESSION_PARAMETERS_CREATE_INFO_KHR,
    eVideoEncodeH265SessionParametersAddInfoKHR      = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_SESSION_PARAMETERS_ADD_INFO_KHR,
    eVideoEncodeH265PictureInfoKHR                   = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_PICTURE_INFO_KHR,
    eVideoEncodeH265DpbSlotInfoKHR                   = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_DPB_SLOT_INFO_KHR,
    eVideoEncodeH265NaluSliceSegmentInfoKHR          = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_NALU_SLICE_SEGMENT_INFO_KHR,
    eVideoEncodeH265GopRemainingFrameInfoKHR         = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_GOP_REMAINING_FRAME_INFO_KHR,
    eVideoEncodeH265ProfileInfoKHR                   = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_PROFILE_INFO_KHR,
    eVideoEncodeH265RateControlInfoKHR               = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_RATE_CONTROL_INFO_KHR,
    eVideoEncodeH265RateControlLayerInfoKHR          = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_RATE_CONTROL_LAYER_INFO_KHR,
    eVideoEncodeH265SessionCreateInfoKHR             = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_SESSION_CREATE_INFO_KHR,
    eVideoEncodeH265QualityLevelPropertiesKHR        = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_QUALITY_LEVEL_PROPERTIES_KHR,
    eVideoEncodeH265SessionParametersGetInfoKHR      = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_SESSION_PARAMETERS_GET_INFO_KHR,
    eVideoEncodeH265SessionParametersFeedbackInfoKHR = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_SESSION_PARAMETERS_FEEDBACK_INFO_KHR,
    eVideoDecodeH264CapabilitiesKHR                  = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_CAPABILITIES_KHR,
    eVideoDecodeH264PictureInfoKHR                   = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PICTURE_INFO_KHR,
    eVideoDecodeH264ProfileInfoKHR                   = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PROFILE_INFO_KHR,
    eVideoDecodeH264SessionParametersCreateInfoKHR   = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR,
    eVideoDecodeH264SessionParametersAddInfoKHR      = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR,
    eVideoDecodeH264DpbSlotInfoKHR                   = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,
    eTextureLodGatherFormatPropertiesAMD             = VK_STRUCTURE_TYPE_TEXTURE_LOD_GATHER_FORMAT_PROPERTIES_AMD,
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
    eImportSemaphoreFdInfoKHR                              = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
    eSemaphoreGetFdInfoKHR                                 = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
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
    eMultiviewPerViewAttributesInfoNVX                     = VK_STRUCTURE_TYPE_MULTIVIEW_PER_VIEW_ATTRIBUTES_INFO_NVX,
    ePipelineViewportSwizzleStateCreateInfoNV              = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_SWIZZLE_STATE_CREATE_INFO_NV,
    ePhysicalDeviceDiscardRectanglePropertiesEXT           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DISCARD_RECTANGLE_PROPERTIES_EXT,
    ePipelineDiscardRectangleStateCreateInfoEXT            = VK_STRUCTURE_TYPE_PIPELINE_DISCARD_RECTANGLE_STATE_CREATE_INFO_EXT,
    ePhysicalDeviceConservativeRasterizationPropertiesEXT  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES_EXT,
    ePipelineRasterizationConservativeStateCreateInfoEXT   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT,
    ePhysicalDeviceDepthClipEnableFeaturesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT,
    ePipelineRasterizationDepthClipStateCreateInfoEXT      = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT,
    eHdrMetadataEXT                                        = VK_STRUCTURE_TYPE_HDR_METADATA_EXT,
    ePhysicalDeviceRelaxedLineRasterizationFeaturesIMG     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RELAXED_LINE_RASTERIZATION_FEATURES_IMG,
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
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    ePhysicalDeviceShaderEnqueueFeaturesAMDX   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ENQUEUE_FEATURES_AMDX,
    ePhysicalDeviceShaderEnqueuePropertiesAMDX = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ENQUEUE_PROPERTIES_AMDX,
    eExecutionGraphPipelineScratchSizeAMDX     = VK_STRUCTURE_TYPE_EXECUTION_GRAPH_PIPELINE_SCRATCH_SIZE_AMDX,
    eExecutionGraphPipelineCreateInfoAMDX      = VK_STRUCTURE_TYPE_EXECUTION_GRAPH_PIPELINE_CREATE_INFO_AMDX,
    ePipelineShaderStageNodeCreateInfoAMDX     = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NODE_CREATE_INFO_AMDX,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eAttachmentSampleCountInfoAMD                      = VK_STRUCTURE_TYPE_ATTACHMENT_SAMPLE_COUNT_INFO_AMD,
    eAttachmentSampleCountInfoNV                       = VK_STRUCTURE_TYPE_ATTACHMENT_SAMPLE_COUNT_INFO_NV,
    ePhysicalDeviceShaderBfloat16FeaturesKHR           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
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
    ePhysicalDeviceShaderCorePropertiesAMD               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD,
    eVideoDecodeH265CapabilitiesKHR                      = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_CAPABILITIES_KHR,
    eVideoDecodeH265SessionParametersCreateInfoKHR       = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_CREATE_INFO_KHR,
    eVideoDecodeH265SessionParametersAddInfoKHR          = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_ADD_INFO_KHR,
    eVideoDecodeH265ProfileInfoKHR                       = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PROFILE_INFO_KHR,
    eVideoDecodeH265PictureInfoKHR                       = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PICTURE_INFO_KHR,
    eVideoDecodeH265DpbSlotInfoKHR                       = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR,
    eDeviceMemoryOverallocationCreateInfoAMD             = VK_STRUCTURE_TYPE_DEVICE_MEMORY_OVERALLOCATION_CREATE_INFO_AMD,
    ePhysicalDeviceVertexAttributeDivisorPropertiesEXT   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_PROPERTIES_EXT,
#if defined( VK_USE_PLATFORM_GGP )
    ePresentFrameTokenGGP = VK_STRUCTURE_TYPE_PRESENT_FRAME_TOKEN_GGP,
#endif /*VK_USE_PLATFORM_GGP*/
    ePhysicalDeviceMeshShaderFeaturesNV                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV,
    ePhysicalDeviceMeshShaderPropertiesNV               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_NV,
    ePhysicalDeviceShaderImageFootprintFeaturesNV       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_FOOTPRINT_FEATURES_NV,
    ePipelineViewportExclusiveScissorStateCreateInfoNV  = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_EXCLUSIVE_SCISSOR_STATE_CREATE_INFO_NV,
    ePhysicalDeviceExclusiveScissorFeaturesNV           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXCLUSIVE_SCISSOR_FEATURES_NV,
    eCheckpointDataNV                                   = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV,
    eQueueFamilyCheckpointPropertiesNV                  = VK_STRUCTURE_TYPE_QUEUE_FAMILY_CHECKPOINT_PROPERTIES_NV,
    eQueueFamilyCheckpointProperties2NV                 = VK_STRUCTURE_TYPE_QUEUE_FAMILY_CHECKPOINT_PROPERTIES_2_NV,
    eCheckpointData2NV                                  = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_2_NV,
    ePhysicalDevicePresentTimingFeaturesEXT             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES_EXT,
    eSwapchainTimingPropertiesEXT                       = VK_STRUCTURE_TYPE_SWAPCHAIN_TIMING_PROPERTIES_EXT,
    eSwapchainTimeDomainPropertiesEXT                   = VK_STRUCTURE_TYPE_SWAPCHAIN_TIME_DOMAIN_PROPERTIES_EXT,
    ePresentTimingsInfoEXT                              = VK_STRUCTURE_TYPE_PRESENT_TIMINGS_INFO_EXT,
    ePresentTimingInfoEXT                               = VK_STRUCTURE_TYPE_PRESENT_TIMING_INFO_EXT,
    ePastPresentationTimingInfoEXT                      = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_INFO_EXT,
    ePastPresentationTimingPropertiesEXT                = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_PROPERTIES_EXT,
    ePastPresentationTimingEXT                          = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_EXT,
    ePresentTimingSurfaceCapabilitiesEXT                = VK_STRUCTURE_TYPE_PRESENT_TIMING_SURFACE_CAPABILITIES_EXT,
    eSwapchainCalibratedTimestampInfoEXT                = VK_STRUCTURE_TYPE_SWAPCHAIN_CALIBRATED_TIMESTAMP_INFO_EXT,
    ePhysicalDeviceShaderIntegerFunctions2FeaturesINTEL = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_FUNCTIONS_2_FEATURES_INTEL,
    eQueryPoolPerformanceQueryCreateInfoINTEL           = VK_STRUCTURE_TYPE_QUERY_POOL_PERFORMANCE_QUERY_CREATE_INFO_INTEL,
    eQueryPoolCreateInfoINTEL                           = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO_INTEL,
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
    eRenderingFragmentDensityMapAttachmentInfoEXT             = VK_STRUCTURE_TYPE_RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT_INFO_EXT,
    eFragmentShadingRateAttachmentInfoKHR                     = VK_STRUCTURE_TYPE_FRAGMENT_SHADING_RATE_ATTACHMENT_INFO_KHR,
    ePipelineFragmentShadingRateStateCreateInfoKHR            = VK_STRUCTURE_TYPE_PIPELINE_FRAGMENT_SHADING_RATE_STATE_CREATE_INFO_KHR,
    ePhysicalDeviceFragmentShadingRatePropertiesKHR           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_PROPERTIES_KHR,
    ePhysicalDeviceFragmentShadingRateFeaturesKHR             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR,
    ePhysicalDeviceFragmentShadingRateKHR                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_KHR,
    eRenderingFragmentShadingRateAttachmentInfoKHR            = VK_STRUCTURE_TYPE_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_INFO_KHR,
    ePhysicalDeviceShaderCoreProperties2AMD                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_2_AMD,
    ePhysicalDeviceCoherentMemoryFeaturesAMD                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD,
    ePhysicalDeviceShaderImageAtomicInt64FeaturesEXT          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT,
    ePhysicalDeviceShaderQuadControlFeaturesKHR               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_QUAD_CONTROL_FEATURES_KHR,
    ePhysicalDeviceMemoryBudgetPropertiesEXT                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT,
    ePhysicalDeviceMemoryPriorityFeaturesEXT                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT,
    eMemoryPriorityAllocateInfoEXT                            = VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT,
    eSurfaceProtectedCapabilitiesKHR                          = VK_STRUCTURE_TYPE_SURFACE_PROTECTED_CAPABILITIES_KHR,
    ePhysicalDeviceDedicatedAllocationImageAliasingFeaturesNV = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEDICATED_ALLOCATION_IMAGE_ALIASING_FEATURES_NV,
    ePhysicalDeviceBufferDeviceAddressFeaturesEXT             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_EXT,
    ePhysicalDeviceBufferAddressFeaturesEXT                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_ADDRESS_FEATURES_EXT,
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
    ePhysicalDeviceShaderAtomicFloatFeaturesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
    ePhysicalDeviceExtendedDynamicStateFeaturesEXT         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT,
    ePhysicalDevicePipelineExecutablePropertiesFeaturesKHR = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR,
    ePipelineInfoKHR                                       = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR,
    ePipelineInfoEXT                                       = VK_STRUCTURE_TYPE_PIPELINE_INFO_EXT,
    ePipelineExecutablePropertiesKHR                       = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR,
    ePipelineExecutableInfoKHR                             = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR,
    ePipelineExecutableStatisticKHR                        = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR,
    ePipelineExecutableInternalRepresentationKHR           = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR,
    ePhysicalDeviceMapMemoryPlacedFeaturesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_FEATURES_EXT,
    ePhysicalDeviceMapMemoryPlacedPropertiesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_PROPERTIES_EXT,
    eMemoryMapPlacedInfoEXT                                = VK_STRUCTURE_TYPE_MEMORY_MAP_PLACED_INFO_EXT,
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
    ePhysicalDeviceDepthBiasControlFeaturesEXT             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_BIAS_CONTROL_FEATURES_EXT,
    eDepthBiasInfoEXT                                      = VK_STRUCTURE_TYPE_DEPTH_BIAS_INFO_EXT,
    eDepthBiasRepresentationInfoEXT                        = VK_STRUCTURE_TYPE_DEPTH_BIAS_REPRESENTATION_INFO_EXT,
    ePhysicalDeviceDeviceMemoryReportFeaturesEXT           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_MEMORY_REPORT_FEATURES_EXT,
    eDeviceDeviceMemoryReportCreateInfoEXT                 = VK_STRUCTURE_TYPE_DEVICE_DEVICE_MEMORY_REPORT_CREATE_INFO_EXT,
    eDeviceMemoryReportCallbackDataEXT                     = VK_STRUCTURE_TYPE_DEVICE_MEMORY_REPORT_CALLBACK_DATA_EXT,
    eSamplerCustomBorderColorCreateInfoEXT                 = VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT,
    ePhysicalDeviceCustomBorderColorPropertiesEXT          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_PROPERTIES_EXT,
    ePhysicalDeviceCustomBorderColorFeaturesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT,
    ePipelineLibraryCreateInfoKHR                          = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR,
    ePhysicalDevicePresentBarrierFeaturesNV                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_BARRIER_FEATURES_NV,
    eSurfaceCapabilitiesPresentBarrierNV                   = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_PRESENT_BARRIER_NV,
    eSwapchainPresentBarrierCreateInfoNV                   = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_BARRIER_CREATE_INFO_NV,
    ePresentIdKHR                                          = VK_STRUCTURE_TYPE_PRESENT_ID_KHR,
    ePhysicalDevicePresentIdFeaturesKHR                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR,
    eVideoEncodeInfoKHR                                    = VK_STRUCTURE_TYPE_VIDEO_ENCODE_INFO_KHR,
    eVideoEncodeRateControlInfoKHR                         = VK_STRUCTURE_TYPE_VIDEO_ENCODE_RATE_CONTROL_INFO_KHR,
    eVideoEncodeRateControlLayerInfoKHR                    = VK_STRUCTURE_TYPE_VIDEO_ENCODE_RATE_CONTROL_LAYER_INFO_KHR,
    eVideoEncodeCapabilitiesKHR                            = VK_STRUCTURE_TYPE_VIDEO_ENCODE_CAPABILITIES_KHR,
    eVideoEncodeUsageInfoKHR                               = VK_STRUCTURE_TYPE_VIDEO_ENCODE_USAGE_INFO_KHR,
    eQueryPoolVideoEncodeFeedbackCreateInfoKHR             = VK_STRUCTURE_TYPE_QUERY_POOL_VIDEO_ENCODE_FEEDBACK_CREATE_INFO_KHR,
    ePhysicalDeviceVideoEncodeQualityLevelInfoKHR          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_ENCODE_QUALITY_LEVEL_INFO_KHR,
    eVideoEncodeQualityLevelPropertiesKHR                  = VK_STRUCTURE_TYPE_VIDEO_ENCODE_QUALITY_LEVEL_PROPERTIES_KHR,
    eVideoEncodeQualityLevelInfoKHR                        = VK_STRUCTURE_TYPE_VIDEO_ENCODE_QUALITY_LEVEL_INFO_KHR,
    eVideoEncodeSessionParametersGetInfoKHR                = VK_STRUCTURE_TYPE_VIDEO_ENCODE_SESSION_PARAMETERS_GET_INFO_KHR,
    eVideoEncodeSessionParametersFeedbackInfoKHR           = VK_STRUCTURE_TYPE_VIDEO_ENCODE_SESSION_PARAMETERS_FEEDBACK_INFO_KHR,
    ePhysicalDeviceDiagnosticsConfigFeaturesNV             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DIAGNOSTICS_CONFIG_FEATURES_NV,
    eDeviceDiagnosticsConfigCreateInfoNV                   = VK_STRUCTURE_TYPE_DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO_NV,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eCudaModuleCreateInfoNV                     = VK_STRUCTURE_TYPE_CUDA_MODULE_CREATE_INFO_NV,
    eCudaFunctionCreateInfoNV                   = VK_STRUCTURE_TYPE_CUDA_FUNCTION_CREATE_INFO_NV,
    eCudaLaunchInfoNV                           = VK_STRUCTURE_TYPE_CUDA_LAUNCH_INFO_NV,
    ePhysicalDeviceCudaKernelLaunchFeaturesNV   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES_NV,
    ePhysicalDeviceCudaKernelLaunchPropertiesNV = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_PROPERTIES_NV,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    ePhysicalDeviceTileShadingFeaturesQCOM   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TILE_SHADING_FEATURES_QCOM,
    ePhysicalDeviceTileShadingPropertiesQCOM = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TILE_SHADING_PROPERTIES_QCOM,
    eRenderPassTileShadingCreateInfoQCOM     = VK_STRUCTURE_TYPE_RENDER_PASS_TILE_SHADING_CREATE_INFO_QCOM,
    ePerTileBeginInfoQCOM                    = VK_STRUCTURE_TYPE_PER_TILE_BEGIN_INFO_QCOM,
    ePerTileEndInfoQCOM                      = VK_STRUCTURE_TYPE_PER_TILE_END_INFO_QCOM,
    eDispatchTileInfoQCOM                    = VK_STRUCTURE_TYPE_DISPATCH_TILE_INFO_QCOM,
    eQueryLowLatencySupportNV                = VK_STRUCTURE_TYPE_QUERY_LOW_LATENCY_SUPPORT_NV,
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
    ePhysicalDeviceDescriptorBufferPropertiesEXT               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT,
    ePhysicalDeviceDescriptorBufferDensityMapPropertiesEXT     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_DENSITY_MAP_PROPERTIES_EXT,
    ePhysicalDeviceDescriptorBufferFeaturesEXT                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT,
    eDescriptorAddressInfoEXT                                  = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
    eDescriptorGetInfoEXT                                      = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
    eBufferCaptureDescriptorDataInfoEXT                        = VK_STRUCTURE_TYPE_BUFFER_CAPTURE_DESCRIPTOR_DATA_INFO_EXT,
    eImageCaptureDescriptorDataInfoEXT                         = VK_STRUCTURE_TYPE_IMAGE_CAPTURE_DESCRIPTOR_DATA_INFO_EXT,
    eImageViewCaptureDescriptorDataInfoEXT                     = VK_STRUCTURE_TYPE_IMAGE_VIEW_CAPTURE_DESCRIPTOR_DATA_INFO_EXT,
    eSamplerCaptureDescriptorDataInfoEXT                       = VK_STRUCTURE_TYPE_SAMPLER_CAPTURE_DESCRIPTOR_DATA_INFO_EXT,
    eOpaqueCaptureDescriptorDataCreateInfoEXT                  = VK_STRUCTURE_TYPE_OPAQUE_CAPTURE_DESCRIPTOR_DATA_CREATE_INFO_EXT,
    eDescriptorBufferBindingInfoEXT                            = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
    eDescriptorBufferBindingPushDescriptorBufferHandleEXT      = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_PUSH_DESCRIPTOR_BUFFER_HANDLE_EXT,
    eAccelerationStructureCaptureDescriptorDataInfoEXT         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CAPTURE_DESCRIPTOR_DATA_INFO_EXT,
    ePhysicalDeviceGraphicsPipelineLibraryFeaturesEXT          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT,
    ePhysicalDeviceGraphicsPipelineLibraryPropertiesEXT        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT,
    eGraphicsPipelineLibraryCreateInfoEXT                      = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
    ePhysicalDeviceShaderEarlyAndLateFragmentTestsFeaturesAMD  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_EARLY_AND_LATE_FRAGMENT_TESTS_FEATURES_AMD,
    ePhysicalDeviceFragmentShaderBarycentricFeaturesKHR        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR,
    ePhysicalDeviceFragmentShaderBarycentricFeaturesNV         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_NV,
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
    ePhysicalDeviceFrameBoundaryFeaturesEXT                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAME_BOUNDARY_FEATURES_EXT,
    eFrameBoundaryEXT                                           = VK_STRUCTURE_TYPE_FRAME_BOUNDARY_EXT,
    ePhysicalDeviceMultisampledRenderToSingleSampledFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_FEATURES_EXT,
    eSubpassResolvePerformanceQueryEXT                          = VK_STRUCTURE_TYPE_SUBPASS_RESOLVE_PERFORMANCE_QUERY_EXT,
    eMultisampledRenderToSingleSampledInfoEXT                   = VK_STRUCTURE_TYPE_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_INFO_EXT,
    ePhysicalDeviceExtendedDynamicState2FeaturesEXT             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT,
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
    eScreenSurfaceCreateInfoQNX = VK_STRUCTURE_TYPE_SCREEN_SURFACE_CREATE_INFO_QNX,
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/
    ePhysicalDeviceColorWriteEnableFeaturesEXT           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COLOR_WRITE_ENABLE_FEATURES_EXT,
    ePipelineColorWriteCreateInfoEXT                     = VK_STRUCTURE_TYPE_PIPELINE_COLOR_WRITE_CREATE_INFO_EXT,
    ePhysicalDevicePrimitivesGeneratedQueryFeaturesEXT   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVES_GENERATED_QUERY_FEATURES_EXT,
    ePhysicalDeviceRayTracingMaintenance1FeaturesKHR     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MAINTENANCE_1_FEATURES_KHR,
    ePhysicalDeviceShaderUntypedPointersFeaturesKHR      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR,
    ePhysicalDeviceVideoEncodeRgbConversionFeaturesVALVE = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_ENCODE_RGB_CONVERSION_FEATURES_VALVE,
    eVideoEncodeRgbConversionCapabilitiesVALVE           = VK_STRUCTURE_TYPE_VIDEO_ENCODE_RGB_CONVERSION_CAPABILITIES_VALVE,
    eVideoEncodeProfileRgbConversionInfoVALVE            = VK_STRUCTURE_TYPE_VIDEO_ENCODE_PROFILE_RGB_CONVERSION_INFO_VALVE,
    eVideoEncodeSessionRgbConversionCreateInfoVALVE      = VK_STRUCTURE_TYPE_VIDEO_ENCODE_SESSION_RGB_CONVERSION_CREATE_INFO_VALVE,
    ePhysicalDeviceImageViewMinLodFeaturesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_VIEW_MIN_LOD_FEATURES_EXT,
    eImageViewMinLodCreateInfoEXT                        = VK_STRUCTURE_TYPE_IMAGE_VIEW_MIN_LOD_CREATE_INFO_EXT,
    ePhysicalDeviceMultiDrawFeaturesEXT                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTI_DRAW_FEATURES_EXT,
    ePhysicalDeviceMultiDrawPropertiesEXT                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTI_DRAW_PROPERTIES_EXT,
    ePhysicalDeviceImage2DViewOf3DFeaturesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_2D_VIEW_OF_3D_FEATURES_EXT,
    ePhysicalDeviceShaderTileImageFeaturesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TILE_IMAGE_FEATURES_EXT,
    ePhysicalDeviceShaderTileImagePropertiesEXT          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TILE_IMAGE_PROPERTIES_EXT,
    eMicromapBuildInfoEXT                                = VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT,
    eMicromapVersionInfoEXT                              = VK_STRUCTURE_TYPE_MICROMAP_VERSION_INFO_EXT,
    eCopyMicromapInfoEXT                                 = VK_STRUCTURE_TYPE_COPY_MICROMAP_INFO_EXT,
    eCopyMicromapToMemoryInfoEXT                         = VK_STRUCTURE_TYPE_COPY_MICROMAP_TO_MEMORY_INFO_EXT,
    eCopyMemoryToMicromapInfoEXT                         = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_MICROMAP_INFO_EXT,
    ePhysicalDeviceOpacityMicromapFeaturesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT,
    ePhysicalDeviceOpacityMicromapPropertiesEXT          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES_EXT,
    eMicromapCreateInfoEXT                               = VK_STRUCTURE_TYPE_MICROMAP_CREATE_INFO_EXT,
    eMicromapBuildSizesInfoEXT                           = VK_STRUCTURE_TYPE_MICROMAP_BUILD_SIZES_INFO_EXT,
    eAccelerationStructureTrianglesOpacityMicromapEXT    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    ePhysicalDeviceDisplacementMicromapFeaturesNV         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DISPLACEMENT_MICROMAP_FEATURES_NV,
    ePhysicalDeviceDisplacementMicromapPropertiesNV       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DISPLACEMENT_MICROMAP_PROPERTIES_NV,
    eAccelerationStructureTrianglesDisplacementMicromapNV = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_DISPLACEMENT_MICROMAP_NV,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    ePhysicalDeviceClusterCullingShaderFeaturesHUAWEI          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_CULLING_SHADER_FEATURES_HUAWEI,
    ePhysicalDeviceClusterCullingShaderPropertiesHUAWEI        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_CULLING_SHADER_PROPERTIES_HUAWEI,
    ePhysicalDeviceClusterCullingShaderVrsFeaturesHUAWEI       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_CULLING_SHADER_VRS_FEATURES_HUAWEI,
    ePhysicalDeviceBorderColorSwizzleFeaturesEXT               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BORDER_COLOR_SWIZZLE_FEATURES_EXT,
    eSamplerBorderColorComponentMappingCreateInfoEXT           = VK_STRUCTURE_TYPE_SAMPLER_BORDER_COLOR_COMPONENT_MAPPING_CREATE_INFO_EXT,
    ePhysicalDevicePageableDeviceLocalMemoryFeaturesEXT        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PAGEABLE_DEVICE_LOCAL_MEMORY_FEATURES_EXT,
    ePhysicalDeviceShaderCorePropertiesARM                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_ARM,
    eDeviceQueueShaderCoreControlCreateInfoARM                 = VK_STRUCTURE_TYPE_DEVICE_QUEUE_SHADER_CORE_CONTROL_CREATE_INFO_ARM,
    ePhysicalDeviceSchedulingControlsFeaturesARM               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCHEDULING_CONTROLS_FEATURES_ARM,
    ePhysicalDeviceSchedulingControlsPropertiesARM             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCHEDULING_CONTROLS_PROPERTIES_ARM,
    ePhysicalDeviceImageSlicedViewOf3DFeaturesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_SLICED_VIEW_OF_3D_FEATURES_EXT,
    eImageViewSlicedCreateInfoEXT                              = VK_STRUCTURE_TYPE_IMAGE_VIEW_SLICED_CREATE_INFO_EXT,
    ePhysicalDeviceDescriptorSetHostMappingFeaturesVALVE       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_SET_HOST_MAPPING_FEATURES_VALVE,
    eDescriptorSetBindingReferenceVALVE                        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_BINDING_REFERENCE_VALVE,
    eDescriptorSetLayoutHostMappingInfoVALVE                   = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_HOST_MAPPING_INFO_VALVE,
    ePhysicalDeviceNonSeamlessCubeMapFeaturesEXT               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_NON_SEAMLESS_CUBE_MAP_FEATURES_EXT,
    ePhysicalDeviceRenderPassStripedFeaturesARM                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RENDER_PASS_STRIPED_FEATURES_ARM,
    ePhysicalDeviceRenderPassStripedPropertiesARM              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RENDER_PASS_STRIPED_PROPERTIES_ARM,
    eRenderPassStripeBeginInfoARM                              = VK_STRUCTURE_TYPE_RENDER_PASS_STRIPE_BEGIN_INFO_ARM,
    eRenderPassStripeInfoARM                                   = VK_STRUCTURE_TYPE_RENDER_PASS_STRIPE_INFO_ARM,
    eRenderPassStripeSubmitInfoARM                             = VK_STRUCTURE_TYPE_RENDER_PASS_STRIPE_SUBMIT_INFO_ARM,
    ePhysicalDeviceCopyMemoryIndirectFeaturesNV                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COPY_MEMORY_INDIRECT_FEATURES_NV,
    ePhysicalDeviceDeviceGeneratedCommandsComputeFeaturesNV    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_COMPUTE_FEATURES_NV,
    eComputePipelineIndirectBufferInfoNV                       = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_INDIRECT_BUFFER_INFO_NV,
    ePipelineIndirectDeviceAddressInfoNV                       = VK_STRUCTURE_TYPE_PIPELINE_INDIRECT_DEVICE_ADDRESS_INFO_NV,
    ePhysicalDeviceRayTracingLinearSweptSpheresFeaturesNV      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_LINEAR_SWEPT_SPHERES_FEATURES_NV,
    eAccelerationStructureGeometryLinearSweptSpheresDataNV     = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_LINEAR_SWEPT_SPHERES_DATA_NV,
    eAccelerationStructureGeometrySpheresDataNV                = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_SPHERES_DATA_NV,
    ePhysicalDeviceLinearColorAttachmentFeaturesNV             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINEAR_COLOR_ATTACHMENT_FEATURES_NV,
    ePhysicalDeviceShaderMaximalReconvergenceFeaturesKHR       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES_KHR,
    ePhysicalDeviceImageCompressionControlSwapchainFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_COMPRESSION_CONTROL_SWAPCHAIN_FEATURES_EXT,
    ePhysicalDeviceImageProcessingFeaturesQCOM                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_PROCESSING_FEATURES_QCOM,
    ePhysicalDeviceImageProcessingPropertiesQCOM               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_PROCESSING_PROPERTIES_QCOM,
    eImageViewSampleWeightCreateInfoQCOM                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_SAMPLE_WEIGHT_CREATE_INFO_QCOM,
    ePhysicalDeviceNestedCommandBufferFeaturesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_NESTED_COMMAND_BUFFER_FEATURES_EXT,
    ePhysicalDeviceNestedCommandBufferPropertiesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_NESTED_COMMAND_BUFFER_PROPERTIES_EXT,
#if defined( VK_USE_PLATFORM_OHOS )
    eNativeBufferUsageOHOS            = VK_STRUCTURE_TYPE_NATIVE_BUFFER_USAGE_OHOS,
    eNativeBufferPropertiesOHOS       = VK_STRUCTURE_TYPE_NATIVE_BUFFER_PROPERTIES_OHOS,
    eNativeBufferFormatPropertiesOHOS = VK_STRUCTURE_TYPE_NATIVE_BUFFER_FORMAT_PROPERTIES_OHOS,
    eImportNativeBufferInfoOHOS       = VK_STRUCTURE_TYPE_IMPORT_NATIVE_BUFFER_INFO_OHOS,
    eMemoryGetNativeBufferInfoOHOS    = VK_STRUCTURE_TYPE_MEMORY_GET_NATIVE_BUFFER_INFO_OHOS,
    eExternalFormatOHOS               = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_OHOS,
#endif /*VK_USE_PLATFORM_OHOS*/
    eExternalMemoryAcquireUnmodifiedEXT                          = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_ACQUIRE_UNMODIFIED_EXT,
    ePhysicalDeviceExtendedDynamicState3FeaturesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT,
    ePhysicalDeviceExtendedDynamicState3PropertiesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_PROPERTIES_EXT,
    ePhysicalDeviceSubpassMergeFeedbackFeaturesEXT               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBPASS_MERGE_FEEDBACK_FEATURES_EXT,
    eRenderPassCreationControlEXT                                = VK_STRUCTURE_TYPE_RENDER_PASS_CREATION_CONTROL_EXT,
    eRenderPassCreationFeedbackCreateInfoEXT                     = VK_STRUCTURE_TYPE_RENDER_PASS_CREATION_FEEDBACK_CREATE_INFO_EXT,
    eRenderPassSubpassFeedbackCreateInfoEXT                      = VK_STRUCTURE_TYPE_RENDER_PASS_SUBPASS_FEEDBACK_CREATE_INFO_EXT,
    eDirectDriverLoadingInfoLUNARG                               = VK_STRUCTURE_TYPE_DIRECT_DRIVER_LOADING_INFO_LUNARG,
    eDirectDriverLoadingListLUNARG                               = VK_STRUCTURE_TYPE_DIRECT_DRIVER_LOADING_LIST_LUNARG,
    eTensorCreateInfoARM                                         = VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_ARM,
    eTensorViewCreateInfoARM                                     = VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_ARM,
    eBindTensorMemoryInfoARM                                     = VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_ARM,
    eWriteDescriptorSetTensorARM                                 = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM,
    ePhysicalDeviceTensorPropertiesARM                           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_PROPERTIES_ARM,
    eTensorFormatPropertiesARM                                   = VK_STRUCTURE_TYPE_TENSOR_FORMAT_PROPERTIES_ARM,
    eTensorDescriptionARM                                        = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
    eTensorMemoryRequirementsInfoARM                             = VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_ARM,
    eTensorMemoryBarrierARM                                      = VK_STRUCTURE_TYPE_TENSOR_MEMORY_BARRIER_ARM,
    ePhysicalDeviceTensorFeaturesARM                             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM,
    eDeviceTensorMemoryRequirementsARM                           = VK_STRUCTURE_TYPE_DEVICE_TENSOR_MEMORY_REQUIREMENTS_ARM,
    eCopyTensorInfoARM                                           = VK_STRUCTURE_TYPE_COPY_TENSOR_INFO_ARM,
    eTensorCopyARM                                               = VK_STRUCTURE_TYPE_TENSOR_COPY_ARM,
    eTensorDependencyInfoARM                                     = VK_STRUCTURE_TYPE_TENSOR_DEPENDENCY_INFO_ARM,
    eMemoryDedicatedAllocateInfoTensorARM                        = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_TENSOR_ARM,
    ePhysicalDeviceExternalTensorInfoARM                         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_TENSOR_INFO_ARM,
    eExternalTensorPropertiesARM                                 = VK_STRUCTURE_TYPE_EXTERNAL_TENSOR_PROPERTIES_ARM,
    eExternalMemoryTensorCreateInfoARM                           = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_TENSOR_CREATE_INFO_ARM,
    ePhysicalDeviceDescriptorBufferTensorFeaturesARM             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_TENSOR_FEATURES_ARM,
    ePhysicalDeviceDescriptorBufferTensorPropertiesARM           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_TENSOR_PROPERTIES_ARM,
    eDescriptorGetTensorInfoARM                                  = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_TENSOR_INFO_ARM,
    eTensorCaptureDescriptorDataInfoARM                          = VK_STRUCTURE_TYPE_TENSOR_CAPTURE_DESCRIPTOR_DATA_INFO_ARM,
    eTensorViewCaptureDescriptorDataInfoARM                      = VK_STRUCTURE_TYPE_TENSOR_VIEW_CAPTURE_DESCRIPTOR_DATA_INFO_ARM,
    eFrameBoundaryTensorsARM                                     = VK_STRUCTURE_TYPE_FRAME_BOUNDARY_TENSORS_ARM,
    ePhysicalDeviceShaderModuleIdentifierFeaturesEXT             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MODULE_IDENTIFIER_FEATURES_EXT,
    ePhysicalDeviceShaderModuleIdentifierPropertiesEXT           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MODULE_IDENTIFIER_PROPERTIES_EXT,
    ePipelineShaderStageModuleIdentifierCreateInfoEXT            = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_MODULE_IDENTIFIER_CREATE_INFO_EXT,
    eShaderModuleIdentifierEXT                                   = VK_STRUCTURE_TYPE_SHADER_MODULE_IDENTIFIER_EXT,
    ePhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_FEATURES_EXT,
    ePhysicalDeviceRasterizationOrderAttachmentAccessFeaturesARM = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_FEATURES_ARM,
    ePhysicalDeviceOpticalFlowFeaturesNV                         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV,
    ePhysicalDeviceOpticalFlowPropertiesNV                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_PROPERTIES_NV,
    eOpticalFlowImageFormatInfoNV                                = VK_STRUCTURE_TYPE_OPTICAL_FLOW_IMAGE_FORMAT_INFO_NV,
    eOpticalFlowImageFormatPropertiesNV                          = VK_STRUCTURE_TYPE_OPTICAL_FLOW_IMAGE_FORMAT_PROPERTIES_NV,
    eOpticalFlowSessionCreateInfoNV                              = VK_STRUCTURE_TYPE_OPTICAL_FLOW_SESSION_CREATE_INFO_NV,
    eOpticalFlowExecuteInfoNV                                    = VK_STRUCTURE_TYPE_OPTICAL_FLOW_EXECUTE_INFO_NV,
    eOpticalFlowSessionCreatePrivateDataInfoNV                   = VK_STRUCTURE_TYPE_OPTICAL_FLOW_SESSION_CREATE_PRIVATE_DATA_INFO_NV,
    ePhysicalDeviceLegacyDitheringFeaturesEXT                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LEGACY_DITHERING_FEATURES_EXT,
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
    ePhysicalDeviceExternalFormatResolveFeaturesANDROID   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_FORMAT_RESOLVE_FEATURES_ANDROID,
    ePhysicalDeviceExternalFormatResolvePropertiesANDROID = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_FORMAT_RESOLVE_PROPERTIES_ANDROID,
    eAndroidHardwareBufferFormatResolvePropertiesANDROID  = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_RESOLVE_PROPERTIES_ANDROID,
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
    ePhysicalDeviceAntiLagFeaturesAMD = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ANTI_LAG_FEATURES_AMD,
    eAntiLagDataAMD                   = VK_STRUCTURE_TYPE_ANTI_LAG_DATA_AMD,
    eAntiLagPresentationInfoAMD       = VK_STRUCTURE_TYPE_ANTI_LAG_PRESENTATION_INFO_AMD,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    ePhysicalDeviceDenseGeometryFormatFeaturesAMDX             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DENSE_GEOMETRY_FORMAT_FEATURES_AMDX,
    eAccelerationStructureDenseGeometryFormatTrianglesDataAMDX = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DENSE_GEOMETRY_FORMAT_TRIANGLES_DATA_AMDX,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eSurfaceCapabilitiesPresentId2KHR                             = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_PRESENT_ID_2_KHR,
    ePresentId2KHR                                                = VK_STRUCTURE_TYPE_PRESENT_ID_2_KHR,
    ePhysicalDevicePresentId2FeaturesKHR                          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_2_FEATURES_KHR,
    eSurfaceCapabilitiesPresentWait2KHR                           = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_PRESENT_WAIT_2_KHR,
    ePhysicalDevicePresentWait2FeaturesKHR                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_2_FEATURES_KHR,
    ePresentWait2InfoKHR                                          = VK_STRUCTURE_TYPE_PRESENT_WAIT_2_INFO_KHR,
    ePhysicalDeviceRayTracingPositionFetchFeaturesKHR             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR,
    ePhysicalDeviceShaderObjectFeaturesEXT                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT,
    ePhysicalDeviceShaderObjectPropertiesEXT                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_PROPERTIES_EXT,
    eShaderCreateInfoEXT                                          = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
    ePhysicalDevicePipelineBinaryFeaturesKHR                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR,
    ePipelineBinaryCreateInfoKHR                                  = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR,
    ePipelineBinaryInfoKHR                                        = VK_STRUCTURE_TYPE_PIPELINE_BINARY_INFO_KHR,
    ePipelineBinaryKeyKHR                                         = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR,
    ePhysicalDevicePipelineBinaryPropertiesKHR                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_PROPERTIES_KHR,
    eReleaseCapturedPipelineDataInfoKHR                           = VK_STRUCTURE_TYPE_RELEASE_CAPTURED_PIPELINE_DATA_INFO_KHR,
    ePipelineBinaryDataInfoKHR                                    = VK_STRUCTURE_TYPE_PIPELINE_BINARY_DATA_INFO_KHR,
    ePipelineCreateInfoKHR                                        = VK_STRUCTURE_TYPE_PIPELINE_CREATE_INFO_KHR,
    eDevicePipelineBinaryInternalCacheControlKHR                  = VK_STRUCTURE_TYPE_DEVICE_PIPELINE_BINARY_INTERNAL_CACHE_CONTROL_KHR,
    ePipelineBinaryHandlesInfoKHR                                 = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR,
    ePhysicalDeviceTilePropertiesFeaturesQCOM                     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TILE_PROPERTIES_FEATURES_QCOM,
    eTilePropertiesQCOM                                           = VK_STRUCTURE_TYPE_TILE_PROPERTIES_QCOM,
    ePhysicalDeviceAmigoProfilingFeaturesSEC                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_AMIGO_PROFILING_FEATURES_SEC,
    eAmigoProfilingSubmitInfoSEC                                  = VK_STRUCTURE_TYPE_AMIGO_PROFILING_SUBMIT_INFO_SEC,
    eSurfacePresentModeKHR                                        = VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_KHR,
    eSurfacePresentModeEXT                                        = VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_EXT,
    eSurfacePresentScalingCapabilitiesKHR                         = VK_STRUCTURE_TYPE_SURFACE_PRESENT_SCALING_CAPABILITIES_KHR,
    eSurfacePresentScalingCapabilitiesEXT                         = VK_STRUCTURE_TYPE_SURFACE_PRESENT_SCALING_CAPABILITIES_EXT,
    eSurfacePresentModeCompatibilityKHR                           = VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_COMPATIBILITY_KHR,
    eSurfacePresentModeCompatibilityEXT                           = VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_COMPATIBILITY_EXT,
    ePhysicalDeviceSwapchainMaintenance1FeaturesKHR               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR,
    ePhysicalDeviceSwapchainMaintenance1FeaturesEXT               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT,
    eSwapchainPresentFenceInfoKHR                                 = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR,
    eSwapchainPresentFenceInfoEXT                                 = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT,
    eSwapchainPresentModesCreateInfoKHR                           = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_KHR,
    eSwapchainPresentModesCreateInfoEXT                           = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT,
    eSwapchainPresentModeInfoKHR                                  = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_KHR,
    eSwapchainPresentModeInfoEXT                                  = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_EXT,
    eSwapchainPresentScalingCreateInfoKHR                         = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_SCALING_CREATE_INFO_KHR,
    eSwapchainPresentScalingCreateInfoEXT                         = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_SCALING_CREATE_INFO_EXT,
    eReleaseSwapchainImagesInfoKHR                                = VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_KHR,
    eReleaseSwapchainImagesInfoEXT                                = VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_EXT,
    ePhysicalDeviceMultiviewPerViewViewportsFeaturesQCOM          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PER_VIEW_VIEWPORTS_FEATURES_QCOM,
    ePhysicalDeviceRayTracingInvocationReorderFeaturesNV          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV,
    ePhysicalDeviceRayTracingInvocationReorderPropertiesNV        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_NV,
    ePhysicalDeviceCooperativeVectorFeaturesNV                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV,
    ePhysicalDeviceCooperativeVectorPropertiesNV                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_PROPERTIES_NV,
    eCooperativeVectorPropertiesNV                                = VK_STRUCTURE_TYPE_COOPERATIVE_VECTOR_PROPERTIES_NV,
    eConvertCooperativeVectorMatrixInfoNV                         = VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV,
    ePhysicalDeviceExtendedSparseAddressSpaceFeaturesNV           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_SPARSE_ADDRESS_SPACE_FEATURES_NV,
    ePhysicalDeviceExtendedSparseAddressSpacePropertiesNV         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_SPARSE_ADDRESS_SPACE_PROPERTIES_NV,
    ePhysicalDeviceMutableDescriptorTypeFeaturesEXT               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT,
    ePhysicalDeviceMutableDescriptorTypeFeaturesVALVE             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_VALVE,
    eMutableDescriptorTypeCreateInfoEXT                           = VK_STRUCTURE_TYPE_MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_EXT,
    eMutableDescriptorTypeCreateInfoVALVE                         = VK_STRUCTURE_TYPE_MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_VALVE,
    ePhysicalDeviceLegacyVertexAttributesFeaturesEXT              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LEGACY_VERTEX_ATTRIBUTES_FEATURES_EXT,
    ePhysicalDeviceLegacyVertexAttributesPropertiesEXT            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LEGACY_VERTEX_ATTRIBUTES_PROPERTIES_EXT,
    eLayerSettingsCreateInfoEXT                                   = VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT,
    ePhysicalDeviceShaderCoreBuiltinsFeaturesARM                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_BUILTINS_FEATURES_ARM,
    ePhysicalDeviceShaderCoreBuiltinsPropertiesARM                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_BUILTINS_PROPERTIES_ARM,
    ePhysicalDevicePipelineLibraryGroupHandlesFeaturesEXT         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_LIBRARY_GROUP_HANDLES_FEATURES_EXT,
    ePhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_FEATURES_EXT,
    eLatencySleepModeInfoNV                                       = VK_STRUCTURE_TYPE_LATENCY_SLEEP_MODE_INFO_NV,
    eLatencySleepInfoNV                                           = VK_STRUCTURE_TYPE_LATENCY_SLEEP_INFO_NV,
    eSetLatencyMarkerInfoNV                                       = VK_STRUCTURE_TYPE_SET_LATENCY_MARKER_INFO_NV,
    eGetLatencyMarkerInfoNV                                       = VK_STRUCTURE_TYPE_GET_LATENCY_MARKER_INFO_NV,
    eLatencyTimingsFrameReportNV                                  = VK_STRUCTURE_TYPE_LATENCY_TIMINGS_FRAME_REPORT_NV,
    eLatencySubmissionPresentIdNV                                 = VK_STRUCTURE_TYPE_LATENCY_SUBMISSION_PRESENT_ID_NV,
    eOutOfBandQueueTypeInfoNV                                     = VK_STRUCTURE_TYPE_OUT_OF_BAND_QUEUE_TYPE_INFO_NV,
    eSwapchainLatencyCreateInfoNV                                 = VK_STRUCTURE_TYPE_SWAPCHAIN_LATENCY_CREATE_INFO_NV,
    eLatencySurfaceCapabilitiesNV                                 = VK_STRUCTURE_TYPE_LATENCY_SURFACE_CAPABILITIES_NV,
    ePhysicalDeviceCooperativeMatrixFeaturesKHR                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
    eCooperativeMatrixPropertiesKHR                               = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR,
    ePhysicalDeviceCooperativeMatrixPropertiesKHR                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR,
    eDataGraphPipelineCreateInfoARM                               = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CREATE_INFO_ARM,
    eDataGraphPipelineSessionCreateInfoARM                        = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM,
    eDataGraphPipelineResourceInfoARM                             = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM,
    eDataGraphPipelineConstantARM                                 = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CONSTANT_ARM,
    eDataGraphPipelineSessionMemoryRequirementsInfoARM            = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_MEMORY_REQUIREMENTS_INFO_ARM,
    eBindDataGraphPipelineSessionMemoryInfoARM                    = VK_STRUCTURE_TYPE_BIND_DATA_GRAPH_PIPELINE_SESSION_MEMORY_INFO_ARM,
    ePhysicalDeviceDataGraphFeaturesARM                           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM,
    eDataGraphPipelineShaderModuleCreateInfoARM                   = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO_ARM,
    eDataGraphPipelinePropertyQueryResultARM                      = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_PROPERTY_QUERY_RESULT_ARM,
    eDataGraphPipelineInfoARM                                     = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_INFO_ARM,
    eDataGraphPipelineCompilerControlCreateInfoARM                = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_COMPILER_CONTROL_CREATE_INFO_ARM,
    eDataGraphPipelineSessionBindPointRequirementsInfoARM         = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENTS_INFO_ARM,
    eDataGraphPipelineSessionBindPointRequirementARM              = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENT_ARM,
    eDataGraphPipelineIdentifierCreateInfoARM                     = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_IDENTIFIER_CREATE_INFO_ARM,
    eDataGraphPipelineDispatchInfoARM                             = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_DISPATCH_INFO_ARM,
    eDataGraphProcessingEngineCreateInfoARM                       = VK_STRUCTURE_TYPE_DATA_GRAPH_PROCESSING_ENGINE_CREATE_INFO_ARM,
    eQueueFamilyDataGraphProcessingEnginePropertiesARM            = VK_STRUCTURE_TYPE_QUEUE_FAMILY_DATA_GRAPH_PROCESSING_ENGINE_PROPERTIES_ARM,
    eQueueFamilyDataGraphPropertiesARM                            = VK_STRUCTURE_TYPE_QUEUE_FAMILY_DATA_GRAPH_PROPERTIES_ARM,
    ePhysicalDeviceQueueFamilyDataGraphProcessingEngineInfoARM    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_QUEUE_FAMILY_DATA_GRAPH_PROCESSING_ENGINE_INFO_ARM,
    eDataGraphPipelineConstantTensorSemiStructuredSparsityInfoARM = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CONSTANT_TENSOR_SEMI_STRUCTURED_SPARSITY_INFO_ARM,
    ePhysicalDeviceMultiviewPerViewRenderAreasFeaturesQCOM        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PER_VIEW_RENDER_AREAS_FEATURES_QCOM,
    eMultiviewPerViewRenderAreasRenderPassBeginInfoQCOM           = VK_STRUCTURE_TYPE_MULTIVIEW_PER_VIEW_RENDER_AREAS_RENDER_PASS_BEGIN_INFO_QCOM,
    ePhysicalDeviceComputeShaderDerivativesFeaturesKHR            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR,
    ePhysicalDeviceComputeShaderDerivativesFeaturesNV             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_NV,
    ePhysicalDeviceComputeShaderDerivativesPropertiesKHR          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_PROPERTIES_KHR,
    eVideoDecodeAv1CapabilitiesKHR                                = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_CAPABILITIES_KHR,
    eVideoDecodeAv1PictureInfoKHR                                 = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PICTURE_INFO_KHR,
    eVideoDecodeAv1ProfileInfoKHR                                 = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PROFILE_INFO_KHR,
    eVideoDecodeAv1SessionParametersCreateInfoKHR                 = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_SESSION_PARAMETERS_CREATE_INFO_KHR,
    eVideoDecodeAv1DpbSlotInfoKHR                                 = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_DPB_SLOT_INFO_KHR,
    eVideoEncodeAv1CapabilitiesKHR                                = VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_CAPABILITIES_KHR,
    eVideoEncodeAv1SessionParametersCreateInfoKHR                 = VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_SESSION_PARAMETERS_CREATE_INFO_KHR,
    eVideoEncodeAv1PictureInfoKHR                                 = VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_PICTURE_INFO_KHR,
    eVideoEncodeAv1DpbSlotInfoKHR                                 = VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_DPB_SLOT_INFO_KHR,
    ePhysicalDeviceVideoEncodeAv1FeaturesKHR                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_ENCODE_AV1_FEATURES_KHR,
    eVideoEncodeAv1ProfileInfoKHR                                 = VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_PROFILE_INFO_KHR,
    eVideoEncodeAv1RateControlInfoKHR                             = VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_RATE_CONTROL_INFO_KHR,
    eVideoEncodeAv1RateControlLayerInfoKHR                        = VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_RATE_CONTROL_LAYER_INFO_KHR,
    eVideoEncodeAv1QualityLevelPropertiesKHR                      = VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_QUALITY_LEVEL_PROPERTIES_KHR,
    eVideoEncodeAv1SessionCreateInfoKHR                           = VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_SESSION_CREATE_INFO_KHR,
    eVideoEncodeAv1GopRemainingFrameInfoKHR                       = VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_GOP_REMAINING_FRAME_INFO_KHR,
    ePhysicalDeviceVideoDecodeVp9FeaturesKHR                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_DECODE_VP9_FEATURES_KHR,
    eVideoDecodeVp9CapabilitiesKHR                                = VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_CAPABILITIES_KHR,
    eVideoDecodeVp9PictureInfoKHR                                 = VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_PICTURE_INFO_KHR,
    eVideoDecodeVp9ProfileInfoKHR                                 = VK_STRUCTURE_TYPE_VIDEO_DECODE_VP9_PROFILE_INFO_KHR,
    ePhysicalDeviceVideoMaintenance1FeaturesKHR                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_MAINTENANCE_1_FEATURES_KHR,
    eVideoInlineQueryInfoKHR                                      = VK_STRUCTURE_TYPE_VIDEO_INLINE_QUERY_INFO_KHR,
    ePhysicalDevicePerStageDescriptorSetFeaturesNV                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PER_STAGE_DESCRIPTOR_SET_FEATURES_NV,
    ePhysicalDeviceImageProcessing2FeaturesQCOM                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_PROCESSING_2_FEATURES_QCOM,
    ePhysicalDeviceImageProcessing2PropertiesQCOM                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_PROCESSING_2_PROPERTIES_QCOM,
    eSamplerBlockMatchWindowCreateInfoQCOM                        = VK_STRUCTURE_TYPE_SAMPLER_BLOCK_MATCH_WINDOW_CREATE_INFO_QCOM,
    eSamplerCubicWeightsCreateInfoQCOM                            = VK_STRUCTURE_TYPE_SAMPLER_CUBIC_WEIGHTS_CREATE_INFO_QCOM,
    ePhysicalDeviceCubicWeightsFeaturesQCOM                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUBIC_WEIGHTS_FEATURES_QCOM,
    eBlitImageCubicWeightsInfoQCOM                                = VK_STRUCTURE_TYPE_BLIT_IMAGE_CUBIC_WEIGHTS_INFO_QCOM,
    ePhysicalDeviceYcbcrDegammaFeaturesQCOM                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_YCBCR_DEGAMMA_FEATURES_QCOM,
    eSamplerYcbcrConversionYcbcrDegammaCreateInfoQCOM             = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_YCBCR_DEGAMMA_CREATE_INFO_QCOM,
    ePhysicalDeviceCubicClampFeaturesQCOM                         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUBIC_CLAMP_FEATURES_QCOM,
    ePhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_FEATURES_EXT,
    ePhysicalDeviceUnifiedImageLayoutsFeaturesKHR                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR,
    eAttachmentFeedbackLoopInfoEXT                                = VK_STRUCTURE_TYPE_ATTACHMENT_FEEDBACK_LOOP_INFO_EXT,
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
    eScreenBufferPropertiesQNX                           = VK_STRUCTURE_TYPE_SCREEN_BUFFER_PROPERTIES_QNX,
    eScreenBufferFormatPropertiesQNX                     = VK_STRUCTURE_TYPE_SCREEN_BUFFER_FORMAT_PROPERTIES_QNX,
    eImportScreenBufferInfoQNX                           = VK_STRUCTURE_TYPE_IMPORT_SCREEN_BUFFER_INFO_QNX,
    eExternalFormatQNX                                   = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_QNX,
    ePhysicalDeviceExternalMemoryScreenBufferFeaturesQNX = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_SCREEN_BUFFER_FEATURES_QNX,
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/
    ePhysicalDeviceLayeredDriverPropertiesMSFT                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LAYERED_DRIVER_PROPERTIES_MSFT,
    eCalibratedTimestampInfoKHR                                 = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR,
    eCalibratedTimestampInfoEXT                                 = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT,
    eSetDescriptorBufferOffsetsInfoEXT                          = VK_STRUCTURE_TYPE_SET_DESCRIPTOR_BUFFER_OFFSETS_INFO_EXT,
    eBindDescriptorBufferEmbeddedSamplersInfoEXT                = VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_BUFFER_EMBEDDED_SAMPLERS_INFO_EXT,
    ePhysicalDeviceDescriptorPoolOverallocationFeaturesNV       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_POOL_OVERALLOCATION_FEATURES_NV,
    ePhysicalDeviceTileMemoryHeapFeaturesQCOM                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TILE_MEMORY_HEAP_FEATURES_QCOM,
    ePhysicalDeviceTileMemoryHeapPropertiesQCOM                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TILE_MEMORY_HEAP_PROPERTIES_QCOM,
    eTileMemoryRequirementsQCOM                                 = VK_STRUCTURE_TYPE_TILE_MEMORY_REQUIREMENTS_QCOM,
    eTileMemoryBindInfoQCOM                                     = VK_STRUCTURE_TYPE_TILE_MEMORY_BIND_INFO_QCOM,
    eTileMemorySizeInfoQCOM                                     = VK_STRUCTURE_TYPE_TILE_MEMORY_SIZE_INFO_QCOM,
    ePhysicalDeviceCopyMemoryIndirectFeaturesKHR                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COPY_MEMORY_INDIRECT_FEATURES_KHR,
    ePhysicalDeviceCopyMemoryIndirectPropertiesKHR              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COPY_MEMORY_INDIRECT_PROPERTIES_KHR,
    ePhysicalDeviceCopyMemoryIndirectPropertiesNV               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COPY_MEMORY_INDIRECT_PROPERTIES_NV,
    eCopyMemoryIndirectInfoKHR                                  = VK_STRUCTURE_TYPE_COPY_MEMORY_INDIRECT_INFO_KHR,
    eCopyMemoryToImageIndirectInfoKHR                           = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INDIRECT_INFO_KHR,
    ePhysicalDeviceMemoryDecompressionFeaturesEXT               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_DECOMPRESSION_FEATURES_EXT,
    ePhysicalDeviceMemoryDecompressionFeaturesNV                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_DECOMPRESSION_FEATURES_NV,
    ePhysicalDeviceMemoryDecompressionPropertiesEXT             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_DECOMPRESSION_PROPERTIES_EXT,
    ePhysicalDeviceMemoryDecompressionPropertiesNV              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_DECOMPRESSION_PROPERTIES_NV,
    eDecompressMemoryInfoEXT                                    = VK_STRUCTURE_TYPE_DECOMPRESS_MEMORY_INFO_EXT,
    eDisplaySurfaceStereoCreateInfoNV                           = VK_STRUCTURE_TYPE_DISPLAY_SURFACE_STEREO_CREATE_INFO_NV,
    eDisplayModeStereoPropertiesNV                              = VK_STRUCTURE_TYPE_DISPLAY_MODE_STEREO_PROPERTIES_NV,
    eVideoEncodeIntraRefreshCapabilitiesKHR                     = VK_STRUCTURE_TYPE_VIDEO_ENCODE_INTRA_REFRESH_CAPABILITIES_KHR,
    eVideoEncodeSessionIntraRefreshCreateInfoKHR                = VK_STRUCTURE_TYPE_VIDEO_ENCODE_SESSION_INTRA_REFRESH_CREATE_INFO_KHR,
    eVideoEncodeIntraRefreshInfoKHR                             = VK_STRUCTURE_TYPE_VIDEO_ENCODE_INTRA_REFRESH_INFO_KHR,
    eVideoReferenceIntraRefreshInfoKHR                          = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_INTRA_REFRESH_INFO_KHR,
    ePhysicalDeviceVideoEncodeIntraRefreshFeaturesKHR           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_ENCODE_INTRA_REFRESH_FEATURES_KHR,
    eVideoEncodeQuantizationMapCapabilitiesKHR                  = VK_STRUCTURE_TYPE_VIDEO_ENCODE_QUANTIZATION_MAP_CAPABILITIES_KHR,
    eVideoFormatQuantizationMapPropertiesKHR                    = VK_STRUCTURE_TYPE_VIDEO_FORMAT_QUANTIZATION_MAP_PROPERTIES_KHR,
    eVideoEncodeQuantizationMapInfoKHR                          = VK_STRUCTURE_TYPE_VIDEO_ENCODE_QUANTIZATION_MAP_INFO_KHR,
    eVideoEncodeQuantizationMapSessionParametersCreateInfoKHR   = VK_STRUCTURE_TYPE_VIDEO_ENCODE_QUANTIZATION_MAP_SESSION_PARAMETERS_CREATE_INFO_KHR,
    ePhysicalDeviceVideoEncodeQuantizationMapFeaturesKHR        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_ENCODE_QUANTIZATION_MAP_FEATURES_KHR,
    eVideoEncodeH264QuantizationMapCapabilitiesKHR              = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_QUANTIZATION_MAP_CAPABILITIES_KHR,
    eVideoEncodeH265QuantizationMapCapabilitiesKHR              = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_QUANTIZATION_MAP_CAPABILITIES_KHR,
    eVideoFormatH265QuantizationMapPropertiesKHR                = VK_STRUCTURE_TYPE_VIDEO_FORMAT_H265_QUANTIZATION_MAP_PROPERTIES_KHR,
    eVideoEncodeAv1QuantizationMapCapabilitiesKHR               = VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_QUANTIZATION_MAP_CAPABILITIES_KHR,
    eVideoFormatAv1QuantizationMapPropertiesKHR                 = VK_STRUCTURE_TYPE_VIDEO_FORMAT_AV1_QUANTIZATION_MAP_PROPERTIES_KHR,
    ePhysicalDeviceRawAccessChainsFeaturesNV                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAW_ACCESS_CHAINS_FEATURES_NV,
    eExternalComputeQueueDeviceCreateInfoNV                     = VK_STRUCTURE_TYPE_EXTERNAL_COMPUTE_QUEUE_DEVICE_CREATE_INFO_NV,
    eExternalComputeQueueCreateInfoNV                           = VK_STRUCTURE_TYPE_EXTERNAL_COMPUTE_QUEUE_CREATE_INFO_NV,
    eExternalComputeQueueDataParamsNV                           = VK_STRUCTURE_TYPE_EXTERNAL_COMPUTE_QUEUE_DATA_PARAMS_NV,
    ePhysicalDeviceExternalComputeQueuePropertiesNV             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_COMPUTE_QUEUE_PROPERTIES_NV,
    ePhysicalDeviceShaderRelaxedExtendedInstructionFeaturesKHR  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_RELAXED_EXTENDED_INSTRUCTION_FEATURES_KHR,
    ePhysicalDeviceCommandBufferInheritanceFeaturesNV           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMMAND_BUFFER_INHERITANCE_FEATURES_NV,
    ePhysicalDeviceMaintenance7FeaturesKHR                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_7_FEATURES_KHR,
    ePhysicalDeviceMaintenance7PropertiesKHR                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_7_PROPERTIES_KHR,
    ePhysicalDeviceLayeredApiPropertiesListKHR                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LAYERED_API_PROPERTIES_LIST_KHR,
    ePhysicalDeviceLayeredApiPropertiesKHR                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LAYERED_API_PROPERTIES_KHR,
    ePhysicalDeviceLayeredApiVulkanPropertiesKHR                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LAYERED_API_VULKAN_PROPERTIES_KHR,
    ePhysicalDeviceShaderAtomicFloat16VectorFeaturesNV          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT16_VECTOR_FEATURES_NV,
    ePhysicalDeviceShaderReplicatedCompositesFeaturesEXT        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_REPLICATED_COMPOSITES_FEATURES_EXT,
    ePhysicalDeviceShaderFloat8FeaturesEXT                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT,
    ePhysicalDeviceRayTracingValidationFeaturesNV               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_VALIDATION_FEATURES_NV,
    ePhysicalDeviceClusterAccelerationStructureFeaturesNV       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV,
    ePhysicalDeviceClusterAccelerationStructurePropertiesNV     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV,
    eClusterAccelerationStructureClustersBottomLevelInputNV     = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV,
    eClusterAccelerationStructureTriangleClusterInputNV         = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV,
    eClusterAccelerationStructureMoveObjectsInputNV             = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_MOVE_OBJECTS_INPUT_NV,
    eClusterAccelerationStructureInputInfoNV                    = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV,
    eClusterAccelerationStructureCommandsInfoNV                 = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV,
    eRayTracingPipelineClusterAccelerationStructureCreateInfoNV = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CLUSTER_ACCELERATION_STRUCTURE_CREATE_INFO_NV,
    ePhysicalDevicePartitionedAccelerationStructureFeaturesNV   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_FEATURES_NV,
    ePhysicalDevicePartitionedAccelerationStructurePropertiesNV = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_PROPERTIES_NV,
    eWriteDescriptorSetPartitionedAccelerationStructureNV       = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_PARTITIONED_ACCELERATION_STRUCTURE_NV,
    ePartitionedAccelerationStructureInstancesInputNV           = VK_STRUCTURE_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCES_INPUT_NV,
    eBuildPartitionedAccelerationStructureInfoNV                = VK_STRUCTURE_TYPE_BUILD_PARTITIONED_ACCELERATION_STRUCTURE_INFO_NV,
    ePartitionedAccelerationStructureFlagsNV                    = VK_STRUCTURE_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_FLAGS_NV,
    ePhysicalDeviceDeviceGeneratedCommandsFeaturesEXT           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_EXT,
    ePhysicalDeviceDeviceGeneratedCommandsPropertiesEXT         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_PROPERTIES_EXT,
    eGeneratedCommandsMemoryRequirementsInfoEXT                 = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_EXT,
    eIndirectExecutionSetCreateInfoEXT                          = VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_CREATE_INFO_EXT,
    eGeneratedCommandsInfoEXT                                   = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_EXT,
    eIndirectCommandsLayoutCreateInfoEXT                        = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_EXT,
    eIndirectCommandsLayoutTokenEXT                             = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT,
    eWriteIndirectExecutionSetPipelineEXT                       = VK_STRUCTURE_TYPE_WRITE_INDIRECT_EXECUTION_SET_PIPELINE_EXT,
    eWriteIndirectExecutionSetShaderEXT                         = VK_STRUCTURE_TYPE_WRITE_INDIRECT_EXECUTION_SET_SHADER_EXT,
    eIndirectExecutionSetPipelineInfoEXT                        = VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_PIPELINE_INFO_EXT,
    eIndirectExecutionSetShaderInfoEXT                          = VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_SHADER_INFO_EXT,
    eIndirectExecutionSetShaderLayoutInfoEXT                    = VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_SHADER_LAYOUT_INFO_EXT,
    eGeneratedCommandsPipelineInfoEXT                           = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_PIPELINE_INFO_EXT,
    eGeneratedCommandsShaderInfoEXT                             = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_SHADER_INFO_EXT,
    ePhysicalDeviceMaintenance8FeaturesKHR                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR,
    eMemoryBarrierAccessFlags3KHR                               = VK_STRUCTURE_TYPE_MEMORY_BARRIER_ACCESS_FLAGS_3_KHR,
    ePhysicalDeviceImageAlignmentControlFeaturesMESA            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ALIGNMENT_CONTROL_FEATURES_MESA,
    ePhysicalDeviceImageAlignmentControlPropertiesMESA          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ALIGNMENT_CONTROL_PROPERTIES_MESA,
    eImageAlignmentControlCreateInfoMESA                        = VK_STRUCTURE_TYPE_IMAGE_ALIGNMENT_CONTROL_CREATE_INFO_MESA,
    ePhysicalDeviceShaderFmaFeaturesKHR                         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR,
    ePhysicalDeviceRayTracingInvocationReorderFeaturesEXT       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT,
    ePhysicalDeviceRayTracingInvocationReorderPropertiesEXT     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_EXT,
    ePhysicalDeviceDepthClampControlFeaturesEXT                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLAMP_CONTROL_FEATURES_EXT,
    ePipelineViewportDepthClampControlCreateInfoEXT             = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLAMP_CONTROL_CREATE_INFO_EXT,
    ePhysicalDeviceMaintenance9FeaturesKHR                      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_9_FEATURES_KHR,
    ePhysicalDeviceMaintenance9PropertiesKHR                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_9_PROPERTIES_KHR,
    eQueueFamilyOwnershipTransferPropertiesKHR                  = VK_STRUCTURE_TYPE_QUEUE_FAMILY_OWNERSHIP_TRANSFER_PROPERTIES_KHR,
    ePhysicalDeviceVideoMaintenance2FeaturesKHR                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_MAINTENANCE_2_FEATURES_KHR,
    eVideoDecodeH264InlineSessionParametersInfoKHR              = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_INLINE_SESSION_PARAMETERS_INFO_KHR,
    eVideoDecodeH265InlineSessionParametersInfoKHR              = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_INLINE_SESSION_PARAMETERS_INFO_KHR,
    eVideoDecodeAv1InlineSessionParametersInfoKHR               = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_INLINE_SESSION_PARAMETERS_INFO_KHR,
#if defined( VK_USE_PLATFORM_OHOS )
    eSurfaceCreateInfoOHOS                    = VK_STRUCTURE_TYPE_SURFACE_CREATE_INFO_OHOS,
    eNativeBufferOHOS                         = VK_STRUCTURE_TYPE_NATIVE_BUFFER_OHOS,
    eSwapchainImageCreateInfoOHOS             = VK_STRUCTURE_TYPE_SWAPCHAIN_IMAGE_CREATE_INFO_OHOS,
    ePhysicalDevicePresentationPropertiesOHOS = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENTATION_PROPERTIES_OHOS,
#endif /*VK_USE_PLATFORM_OHOS*/
    ePhysicalDeviceHdrVividFeaturesHUAWEI             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HDR_VIVID_FEATURES_HUAWEI,
    eHdrVividDynamicMetadataHUAWEI                    = VK_STRUCTURE_TYPE_HDR_VIVID_DYNAMIC_METADATA_HUAWEI,
    ePhysicalDeviceCooperativeMatrix2FeaturesNV       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV,
    eCooperativeMatrixFlexibleDimensionsPropertiesNV  = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_FLEXIBLE_DIMENSIONS_PROPERTIES_NV,
    ePhysicalDeviceCooperativeMatrix2PropertiesNV     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_PROPERTIES_NV,
    ePhysicalDevicePipelineOpacityMicromapFeaturesARM = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_OPACITY_MICROMAP_FEATURES_ARM,
#if defined( VK_USE_PLATFORM_METAL_EXT )
    eImportMemoryMetalHandleInfoEXT = VK_STRUCTURE_TYPE_IMPORT_MEMORY_METAL_HANDLE_INFO_EXT,
    eMemoryMetalHandlePropertiesEXT = VK_STRUCTURE_TYPE_MEMORY_METAL_HANDLE_PROPERTIES_EXT,
    eMemoryGetMetalHandleInfoEXT    = VK_STRUCTURE_TYPE_MEMORY_GET_METAL_HANDLE_INFO_EXT,
#endif /*VK_USE_PLATFORM_METAL_EXT*/
    ePhysicalDeviceDepthClampZeroOneFeaturesKHR             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLAMP_ZERO_ONE_FEATURES_KHR,
    ePhysicalDeviceDepthClampZeroOneFeaturesEXT             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLAMP_ZERO_ONE_FEATURES_EXT,
    ePhysicalDevicePerformanceCountersByRegionFeaturesARM   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PERFORMANCE_COUNTERS_BY_REGION_FEATURES_ARM,
    ePhysicalDevicePerformanceCountersByRegionPropertiesARM = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PERFORMANCE_COUNTERS_BY_REGION_PROPERTIES_ARM,
    ePerformanceCounterARM                                  = VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_ARM,
    ePerformanceCounterDescriptionARM                       = VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_DESCRIPTION_ARM,
    eRenderPassPerformanceCountersByRegionBeginInfoARM      = VK_STRUCTURE_TYPE_RENDER_PASS_PERFORMANCE_COUNTERS_BY_REGION_BEGIN_INFO_ARM,
    ePhysicalDeviceVertexAttributeRobustnessFeaturesEXT     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_ROBUSTNESS_FEATURES_EXT,
    ePhysicalDeviceFormatPackFeaturesARM                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FORMAT_PACK_FEATURES_ARM,
    ePhysicalDeviceFragmentDensityMapLayeredFeaturesVALVE   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_LAYERED_FEATURES_VALVE,
    ePhysicalDeviceFragmentDensityMapLayeredPropertiesVALVE = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_LAYERED_PROPERTIES_VALVE,
    ePipelineFragmentDensityMapLayeredCreateInfoVALVE       = VK_STRUCTURE_TYPE_PIPELINE_FRAGMENT_DENSITY_MAP_LAYERED_CREATE_INFO_VALVE,
    ePhysicalDeviceRobustness2FeaturesKHR                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_KHR,
    ePhysicalDeviceRobustness2FeaturesEXT                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
    ePhysicalDeviceRobustness2PropertiesKHR                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_PROPERTIES_KHR,
    ePhysicalDeviceRobustness2PropertiesEXT                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_PROPERTIES_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eSetPresentConfigNV                      = VK_STRUCTURE_TYPE_SET_PRESENT_CONFIG_NV,
    ePhysicalDevicePresentMeteringFeaturesNV = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_METERING_FEATURES_NV,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    ePhysicalDeviceFragmentDensityMapOffsetFeaturesEXT        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_OFFSET_FEATURES_EXT,
    ePhysicalDeviceFragmentDensityMapOffsetFeaturesQCOM       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_OFFSET_FEATURES_QCOM,
    ePhysicalDeviceFragmentDensityMapOffsetPropertiesEXT      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_OFFSET_PROPERTIES_EXT,
    ePhysicalDeviceFragmentDensityMapOffsetPropertiesQCOM     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_OFFSET_PROPERTIES_QCOM,
    eRenderPassFragmentDensityMapOffsetEndInfoEXT             = VK_STRUCTURE_TYPE_RENDER_PASS_FRAGMENT_DENSITY_MAP_OFFSET_END_INFO_EXT,
    eSubpassFragmentDensityMapOffsetEndInfoQCOM               = VK_STRUCTURE_TYPE_SUBPASS_FRAGMENT_DENSITY_MAP_OFFSET_END_INFO_QCOM,
    ePhysicalDeviceZeroInitializeDeviceMemoryFeaturesEXT      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_DEVICE_MEMORY_FEATURES_EXT,
    ePhysicalDevicePresentModeFifoLatestReadyFeaturesKHR      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_MODE_FIFO_LATEST_READY_FEATURES_KHR,
    ePhysicalDevicePresentModeFifoLatestReadyFeaturesEXT      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_MODE_FIFO_LATEST_READY_FEATURES_EXT,
    ePhysicalDeviceShader64BitIndexingFeaturesEXT             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_64_BIT_INDEXING_FEATURES_EXT,
    ePhysicalDeviceCustomResolveFeaturesEXT                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_RESOLVE_FEATURES_EXT,
    eBeginCustomResolveInfoEXT                                = VK_STRUCTURE_TYPE_BEGIN_CUSTOM_RESOLVE_INFO_EXT,
    eCustomResolveCreateInfoEXT                               = VK_STRUCTURE_TYPE_CUSTOM_RESOLVE_CREATE_INFO_EXT,
    ePhysicalDeviceDataGraphModelFeaturesQCOM                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_MODEL_FEATURES_QCOM,
    eDataGraphPipelineBuiltinModelCreateInfoQCOM              = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_BUILTIN_MODEL_CREATE_INFO_QCOM,
    ePhysicalDeviceMaintenance10FeaturesKHR                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_10_FEATURES_KHR,
    ePhysicalDeviceMaintenance10PropertiesKHR                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_10_PROPERTIES_KHR,
    eRenderingAttachmentFlagsInfoKHR                          = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_FLAGS_INFO_KHR,
    eRenderingEndInfoKHR                                      = VK_STRUCTURE_TYPE_RENDERING_END_INFO_KHR,
    eRenderingEndInfoEXT                                      = VK_STRUCTURE_TYPE_RENDERING_END_INFO_EXT,
    eResolveImageModeInfoKHR                                  = VK_STRUCTURE_TYPE_RESOLVE_IMAGE_MODE_INFO_KHR,
    ePhysicalDevicePipelineCacheIncrementalModeFeaturesSEC    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CACHE_INCREMENTAL_MODE_FEATURES_SEC,
    ePhysicalDeviceShaderUniformBufferUnsizedArrayFeaturesEXT = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNIFORM_BUFFER_UNSIZED_ARRAY_FEATURES_EXT
  };

  // wrapper class for enum VkObjectType, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkObjectType.html
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
    eDescriptorUpdateTemplate      = VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE,
    eDescriptorUpdateTemplateKHR   = VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_KHR,
    eSamplerYcbcrConversion        = VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION,
    eSamplerYcbcrConversionKHR     = VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION_KHR,
    ePrivateDataSlot               = VK_OBJECT_TYPE_PRIVATE_DATA_SLOT,
    ePrivateDataSlotEXT            = VK_OBJECT_TYPE_PRIVATE_DATA_SLOT_EXT,
    eSurfaceKHR                    = VK_OBJECT_TYPE_SURFACE_KHR,
    eSwapchainKHR                  = VK_OBJECT_TYPE_SWAPCHAIN_KHR,
    eDisplayKHR                    = VK_OBJECT_TYPE_DISPLAY_KHR,
    eDisplayModeKHR                = VK_OBJECT_TYPE_DISPLAY_MODE_KHR,
    eDebugReportCallbackEXT        = VK_OBJECT_TYPE_DEBUG_REPORT_CALLBACK_EXT,
    eVideoSessionKHR               = VK_OBJECT_TYPE_VIDEO_SESSION_KHR,
    eVideoSessionParametersKHR     = VK_OBJECT_TYPE_VIDEO_SESSION_PARAMETERS_KHR,
    eCuModuleNVX                   = VK_OBJECT_TYPE_CU_MODULE_NVX,
    eCuFunctionNVX                 = VK_OBJECT_TYPE_CU_FUNCTION_NVX,
    eDebugUtilsMessengerEXT        = VK_OBJECT_TYPE_DEBUG_UTILS_MESSENGER_EXT,
    eAccelerationStructureKHR      = VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR,
    eValidationCacheEXT            = VK_OBJECT_TYPE_VALIDATION_CACHE_EXT,
    eAccelerationStructureNV       = VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV,
    ePerformanceConfigurationINTEL = VK_OBJECT_TYPE_PERFORMANCE_CONFIGURATION_INTEL,
    eDeferredOperationKHR          = VK_OBJECT_TYPE_DEFERRED_OPERATION_KHR,
    eIndirectCommandsLayoutNV      = VK_OBJECT_TYPE_INDIRECT_COMMANDS_LAYOUT_NV,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eCudaModuleNV   = VK_OBJECT_TYPE_CUDA_MODULE_NV,
    eCudaFunctionNV = VK_OBJECT_TYPE_CUDA_FUNCTION_NV,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_USE_PLATFORM_FUCHSIA )
    eBufferCollectionFUCHSIA = VK_OBJECT_TYPE_BUFFER_COLLECTION_FUCHSIA,
#endif /*VK_USE_PLATFORM_FUCHSIA*/
    eMicromapEXT                 = VK_OBJECT_TYPE_MICROMAP_EXT,
    eTensorARM                   = VK_OBJECT_TYPE_TENSOR_ARM,
    eTensorViewARM               = VK_OBJECT_TYPE_TENSOR_VIEW_ARM,
    eOpticalFlowSessionNV        = VK_OBJECT_TYPE_OPTICAL_FLOW_SESSION_NV,
    eShaderEXT                   = VK_OBJECT_TYPE_SHADER_EXT,
    ePipelineBinaryKHR           = VK_OBJECT_TYPE_PIPELINE_BINARY_KHR,
    eDataGraphPipelineSessionARM = VK_OBJECT_TYPE_DATA_GRAPH_PIPELINE_SESSION_ARM,
    eExternalComputeQueueNV      = VK_OBJECT_TYPE_EXTERNAL_COMPUTE_QUEUE_NV,
    eIndirectCommandsLayoutEXT   = VK_OBJECT_TYPE_INDIRECT_COMMANDS_LAYOUT_EXT,
    eIndirectExecutionSetEXT     = VK_OBJECT_TYPE_INDIRECT_EXECUTION_SET_EXT
  };

  // wrapper class for enum VkVendorId, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVendorId.html
  enum class VendorId
  {
    eKhronos  = VK_VENDOR_ID_KHRONOS,
    eVIV      = VK_VENDOR_ID_VIV,
    eVSI      = VK_VENDOR_ID_VSI,
    eKazan    = VK_VENDOR_ID_KAZAN,
    eCodeplay = VK_VENDOR_ID_CODEPLAY,
    eMESA     = VK_VENDOR_ID_MESA,
    ePocl     = VK_VENDOR_ID_POCL,
    eMobileye = VK_VENDOR_ID_MOBILEYE
  };

  // wrapper class for enum VkFormat, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkFormat.html
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
    eG8B8G8R8422UnormKHR                     = VK_FORMAT_G8B8G8R8_422_UNORM_KHR,
    eB8G8R8G8422Unorm                        = VK_FORMAT_B8G8R8G8_422_UNORM,
    eB8G8R8G8422UnormKHR                     = VK_FORMAT_B8G8R8G8_422_UNORM_KHR,
    eG8B8R83Plane420Unorm                    = VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM,
    eG8B8R83Plane420UnormKHR                 = VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM_KHR,
    eG8B8R82Plane420Unorm                    = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
    eG8B8R82Plane420UnormKHR                 = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM_KHR,
    eG8B8R83Plane422Unorm                    = VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM,
    eG8B8R83Plane422UnormKHR                 = VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM_KHR,
    eG8B8R82Plane422Unorm                    = VK_FORMAT_G8_B8R8_2PLANE_422_UNORM,
    eG8B8R82Plane422UnormKHR                 = VK_FORMAT_G8_B8R8_2PLANE_422_UNORM_KHR,
    eG8B8R83Plane444Unorm                    = VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM,
    eG8B8R83Plane444UnormKHR                 = VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM_KHR,
    eR10X6UnormPack16                        = VK_FORMAT_R10X6_UNORM_PACK16,
    eR10X6UnormPack16KHR                     = VK_FORMAT_R10X6_UNORM_PACK16_KHR,
    eR10X6G10X6Unorm2Pack16                  = VK_FORMAT_R10X6G10X6_UNORM_2PACK16,
    eR10X6G10X6Unorm2Pack16KHR               = VK_FORMAT_R10X6G10X6_UNORM_2PACK16_KHR,
    eR10X6G10X6B10X6A10X6Unorm4Pack16        = VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16,
    eR10X6G10X6B10X6A10X6Unorm4Pack16KHR     = VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16_KHR,
    eG10X6B10X6G10X6R10X6422Unorm4Pack16     = VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16,
    eG10X6B10X6G10X6R10X6422Unorm4Pack16KHR  = VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16_KHR,
    eB10X6G10X6R10X6G10X6422Unorm4Pack16     = VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16,
    eB10X6G10X6R10X6G10X6422Unorm4Pack16KHR  = VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16_KHR,
    eG10X6B10X6R10X63Plane420Unorm3Pack16    = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16,
    eG10X6B10X6R10X63Plane420Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16_KHR,
    eG10X6B10X6R10X62Plane420Unorm3Pack16    = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
    eG10X6B10X6R10X62Plane420Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16_KHR,
    eG10X6B10X6R10X63Plane422Unorm3Pack16    = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16,
    eG10X6B10X6R10X63Plane422Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16_KHR,
    eG10X6B10X6R10X62Plane422Unorm3Pack16    = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16,
    eG10X6B10X6R10X62Plane422Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16_KHR,
    eG10X6B10X6R10X63Plane444Unorm3Pack16    = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16,
    eG10X6B10X6R10X63Plane444Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16_KHR,
    eR12X4UnormPack16                        = VK_FORMAT_R12X4_UNORM_PACK16,
    eR12X4UnormPack16KHR                     = VK_FORMAT_R12X4_UNORM_PACK16_KHR,
    eR12X4G12X4Unorm2Pack16                  = VK_FORMAT_R12X4G12X4_UNORM_2PACK16,
    eR12X4G12X4Unorm2Pack16KHR               = VK_FORMAT_R12X4G12X4_UNORM_2PACK16_KHR,
    eR12X4G12X4B12X4A12X4Unorm4Pack16        = VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16,
    eR12X4G12X4B12X4A12X4Unorm4Pack16KHR     = VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16_KHR,
    eG12X4B12X4G12X4R12X4422Unorm4Pack16     = VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16,
    eG12X4B12X4G12X4R12X4422Unorm4Pack16KHR  = VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16_KHR,
    eB12X4G12X4R12X4G12X4422Unorm4Pack16     = VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16,
    eB12X4G12X4R12X4G12X4422Unorm4Pack16KHR  = VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16_KHR,
    eG12X4B12X4R12X43Plane420Unorm3Pack16    = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16,
    eG12X4B12X4R12X43Plane420Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16_KHR,
    eG12X4B12X4R12X42Plane420Unorm3Pack16    = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16,
    eG12X4B12X4R12X42Plane420Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16_KHR,
    eG12X4B12X4R12X43Plane422Unorm3Pack16    = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16,
    eG12X4B12X4R12X43Plane422Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16_KHR,
    eG12X4B12X4R12X42Plane422Unorm3Pack16    = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16,
    eG12X4B12X4R12X42Plane422Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16_KHR,
    eG12X4B12X4R12X43Plane444Unorm3Pack16    = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16,
    eG12X4B12X4R12X43Plane444Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16_KHR,
    eG16B16G16R16422Unorm                    = VK_FORMAT_G16B16G16R16_422_UNORM,
    eG16B16G16R16422UnormKHR                 = VK_FORMAT_G16B16G16R16_422_UNORM_KHR,
    eB16G16R16G16422Unorm                    = VK_FORMAT_B16G16R16G16_422_UNORM,
    eB16G16R16G16422UnormKHR                 = VK_FORMAT_B16G16R16G16_422_UNORM_KHR,
    eG16B16R163Plane420Unorm                 = VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM,
    eG16B16R163Plane420UnormKHR              = VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM_KHR,
    eG16B16R162Plane420Unorm                 = VK_FORMAT_G16_B16R16_2PLANE_420_UNORM,
    eG16B16R162Plane420UnormKHR              = VK_FORMAT_G16_B16R16_2PLANE_420_UNORM_KHR,
    eG16B16R163Plane422Unorm                 = VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM,
    eG16B16R163Plane422UnormKHR              = VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM_KHR,
    eG16B16R162Plane422Unorm                 = VK_FORMAT_G16_B16R16_2PLANE_422_UNORM,
    eG16B16R162Plane422UnormKHR              = VK_FORMAT_G16_B16R16_2PLANE_422_UNORM_KHR,
    eG16B16R163Plane444Unorm                 = VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM,
    eG16B16R163Plane444UnormKHR              = VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM_KHR,
    eG8B8R82Plane444Unorm                    = VK_FORMAT_G8_B8R8_2PLANE_444_UNORM,
    eG8B8R82Plane444UnormEXT                 = VK_FORMAT_G8_B8R8_2PLANE_444_UNORM_EXT,
    eG10X6B10X6R10X62Plane444Unorm3Pack16    = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16,
    eG10X6B10X6R10X62Plane444Unorm3Pack16EXT = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16_EXT,
    eG12X4B12X4R12X42Plane444Unorm3Pack16    = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16,
    eG12X4B12X4R12X42Plane444Unorm3Pack16EXT = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16_EXT,
    eG16B16R162Plane444Unorm                 = VK_FORMAT_G16_B16R16_2PLANE_444_UNORM,
    eG16B16R162Plane444UnormEXT              = VK_FORMAT_G16_B16R16_2PLANE_444_UNORM_EXT,
    eA4R4G4B4UnormPack16                     = VK_FORMAT_A4R4G4B4_UNORM_PACK16,
    eA4R4G4B4UnormPack16EXT                  = VK_FORMAT_A4R4G4B4_UNORM_PACK16_EXT,
    eA4B4G4R4UnormPack16                     = VK_FORMAT_A4B4G4R4_UNORM_PACK16,
    eA4B4G4R4UnormPack16EXT                  = VK_FORMAT_A4B4G4R4_UNORM_PACK16_EXT,
    eAstc4x4SfloatBlock                      = VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK,
    eAstc4x4SfloatBlockEXT                   = VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK_EXT,
    eAstc5x4SfloatBlock                      = VK_FORMAT_ASTC_5x4_SFLOAT_BLOCK,
    eAstc5x4SfloatBlockEXT                   = VK_FORMAT_ASTC_5x4_SFLOAT_BLOCK_EXT,
    eAstc5x5SfloatBlock                      = VK_FORMAT_ASTC_5x5_SFLOAT_BLOCK,
    eAstc5x5SfloatBlockEXT                   = VK_FORMAT_ASTC_5x5_SFLOAT_BLOCK_EXT,
    eAstc6x5SfloatBlock                      = VK_FORMAT_ASTC_6x5_SFLOAT_BLOCK,
    eAstc6x5SfloatBlockEXT                   = VK_FORMAT_ASTC_6x5_SFLOAT_BLOCK_EXT,
    eAstc6x6SfloatBlock                      = VK_FORMAT_ASTC_6x6_SFLOAT_BLOCK,
    eAstc6x6SfloatBlockEXT                   = VK_FORMAT_ASTC_6x6_SFLOAT_BLOCK_EXT,
    eAstc8x5SfloatBlock                      = VK_FORMAT_ASTC_8x5_SFLOAT_BLOCK,
    eAstc8x5SfloatBlockEXT                   = VK_FORMAT_ASTC_8x5_SFLOAT_BLOCK_EXT,
    eAstc8x6SfloatBlock                      = VK_FORMAT_ASTC_8x6_SFLOAT_BLOCK,
    eAstc8x6SfloatBlockEXT                   = VK_FORMAT_ASTC_8x6_SFLOAT_BLOCK_EXT,
    eAstc8x8SfloatBlock                      = VK_FORMAT_ASTC_8x8_SFLOAT_BLOCK,
    eAstc8x8SfloatBlockEXT                   = VK_FORMAT_ASTC_8x8_SFLOAT_BLOCK_EXT,
    eAstc10x5SfloatBlock                     = VK_FORMAT_ASTC_10x5_SFLOAT_BLOCK,
    eAstc10x5SfloatBlockEXT                  = VK_FORMAT_ASTC_10x5_SFLOAT_BLOCK_EXT,
    eAstc10x6SfloatBlock                     = VK_FORMAT_ASTC_10x6_SFLOAT_BLOCK,
    eAstc10x6SfloatBlockEXT                  = VK_FORMAT_ASTC_10x6_SFLOAT_BLOCK_EXT,
    eAstc10x8SfloatBlock                     = VK_FORMAT_ASTC_10x8_SFLOAT_BLOCK,
    eAstc10x8SfloatBlockEXT                  = VK_FORMAT_ASTC_10x8_SFLOAT_BLOCK_EXT,
    eAstc10x10SfloatBlock                    = VK_FORMAT_ASTC_10x10_SFLOAT_BLOCK,
    eAstc10x10SfloatBlockEXT                 = VK_FORMAT_ASTC_10x10_SFLOAT_BLOCK_EXT,
    eAstc12x10SfloatBlock                    = VK_FORMAT_ASTC_12x10_SFLOAT_BLOCK,
    eAstc12x10SfloatBlockEXT                 = VK_FORMAT_ASTC_12x10_SFLOAT_BLOCK_EXT,
    eAstc12x12SfloatBlock                    = VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK,
    eAstc12x12SfloatBlockEXT                 = VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK_EXT,
    eA1B5G5R5UnormPack16                     = VK_FORMAT_A1B5G5R5_UNORM_PACK16,
    eA1B5G5R5UnormPack16KHR                  = VK_FORMAT_A1B5G5R5_UNORM_PACK16_KHR,
    eA8Unorm                                 = VK_FORMAT_A8_UNORM,
    eA8UnormKHR                              = VK_FORMAT_A8_UNORM_KHR,
    ePvrtc12BppUnormBlockIMG                 = VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG,
    ePvrtc14BppUnormBlockIMG                 = VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG,
    ePvrtc22BppUnormBlockIMG                 = VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG,
    ePvrtc24BppUnormBlockIMG                 = VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG,
    ePvrtc12BppSrgbBlockIMG                  = VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG,
    ePvrtc14BppSrgbBlockIMG                  = VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG,
    ePvrtc22BppSrgbBlockIMG                  = VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG,
    ePvrtc24BppSrgbBlockIMG                  = VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG,
    eR8BoolARM                               = VK_FORMAT_R8_BOOL_ARM,
    eR16G16Sfixed5NV                         = VK_FORMAT_R16G16_SFIXED5_NV,
    eR16G16S105NV                            = VK_FORMAT_R16G16_S10_5_NV,
    eR10X6UintPack16ARM                      = VK_FORMAT_R10X6_UINT_PACK16_ARM,
    eR10X6G10X6Uint2Pack16ARM                = VK_FORMAT_R10X6G10X6_UINT_2PACK16_ARM,
    eR10X6G10X6B10X6A10X6Uint4Pack16ARM      = VK_FORMAT_R10X6G10X6B10X6A10X6_UINT_4PACK16_ARM,
    eR12X4UintPack16ARM                      = VK_FORMAT_R12X4_UINT_PACK16_ARM,
    eR12X4G12X4Uint2Pack16ARM                = VK_FORMAT_R12X4G12X4_UINT_2PACK16_ARM,
    eR12X4G12X4B12X4A12X4Uint4Pack16ARM      = VK_FORMAT_R12X4G12X4B12X4A12X4_UINT_4PACK16_ARM,
    eR14X2UintPack16ARM                      = VK_FORMAT_R14X2_UINT_PACK16_ARM,
    eR14X2G14X2Uint2Pack16ARM                = VK_FORMAT_R14X2G14X2_UINT_2PACK16_ARM,
    eR14X2G14X2B14X2A14X2Uint4Pack16ARM      = VK_FORMAT_R14X2G14X2B14X2A14X2_UINT_4PACK16_ARM,
    eR14X2UnormPack16ARM                     = VK_FORMAT_R14X2_UNORM_PACK16_ARM,
    eR14X2G14X2Unorm2Pack16ARM               = VK_FORMAT_R14X2G14X2_UNORM_2PACK16_ARM,
    eR14X2G14X2B14X2A14X2Unorm4Pack16ARM     = VK_FORMAT_R14X2G14X2B14X2A14X2_UNORM_4PACK16_ARM,
    eG14X2B14X2R14X22Plane420Unorm3Pack16ARM = VK_FORMAT_G14X2_B14X2R14X2_2PLANE_420_UNORM_3PACK16_ARM,
    eG14X2B14X2R14X22Plane422Unorm3Pack16ARM = VK_FORMAT_G14X2_B14X2R14X2_2PLANE_422_UNORM_3PACK16_ARM
  };

  // wrapper class for enum VkFormatFeatureFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkFormatFeatureFlagBits.html
  enum class FormatFeatureFlagBits : VkFormatFeatureFlags
  {
    eSampledImage                                               = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT,
    eStorageImage                                               = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT,
    eStorageImageAtomic                                         = VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT,
    eUniformTexelBuffer                                         = VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT,
    eStorageTexelBuffer                                         = VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT,
    eStorageTexelBufferAtomic                                   = VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT,
    eVertexBuffer                                               = VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT,
    eColorAttachment                                            = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT,
    eColorAttachmentBlend                                       = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT,
    eDepthStencilAttachment                                     = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT,
    eBlitSrc                                                    = VK_FORMAT_FEATURE_BLIT_SRC_BIT,
    eBlitDst                                                    = VK_FORMAT_FEATURE_BLIT_DST_BIT,
    eSampledImageFilterLinear                                   = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT,
    eTransferSrc                                                = VK_FORMAT_FEATURE_TRANSFER_SRC_BIT,
    eTransferSrcKHR                                             = VK_FORMAT_FEATURE_TRANSFER_SRC_BIT_KHR,
    eTransferDst                                                = VK_FORMAT_FEATURE_TRANSFER_DST_BIT,
    eTransferDstKHR                                             = VK_FORMAT_FEATURE_TRANSFER_DST_BIT_KHR,
    eMidpointChromaSamples                                      = VK_FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT,
    eMidpointChromaSamplesKHR                                   = VK_FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT_KHR,
    eSampledImageYcbcrConversionLinearFilter                    = VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT,
    eSampledImageYcbcrConversionLinearFilterKHR                 = VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT_KHR,
    eSampledImageYcbcrConversionSeparateReconstructionFilter    = VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT,
    eSampledImageYcbcrConversionSeparateReconstructionFilterKHR = VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT_KHR,
    eSampledImageYcbcrConversionChromaReconstructionExplicit    = VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_BIT,
    eSampledImageYcbcrConversionChromaReconstructionExplicitKHR = VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_BIT_KHR,
    eSampledImageYcbcrConversionChromaReconstructionExplicitForceable =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE_BIT,
    eSampledImageYcbcrConversionChromaReconstructionExplicitForceableKHR =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE_BIT_KHR,
    eDisjoint                             = VK_FORMAT_FEATURE_DISJOINT_BIT,
    eDisjointKHR                          = VK_FORMAT_FEATURE_DISJOINT_BIT_KHR,
    eCositedChromaSamples                 = VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT,
    eCositedChromaSamplesKHR              = VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT_KHR,
    eSampledImageFilterMinmax             = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_MINMAX_BIT,
    eSampledImageFilterMinmaxEXT          = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_MINMAX_BIT_EXT,
    eVideoDecodeOutputKHR                 = VK_FORMAT_FEATURE_VIDEO_DECODE_OUTPUT_BIT_KHR,
    eVideoDecodeDpbKHR                    = VK_FORMAT_FEATURE_VIDEO_DECODE_DPB_BIT_KHR,
    eAccelerationStructureVertexBufferKHR = VK_FORMAT_FEATURE_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR,
    eSampledImageFilterCubicEXT           = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_CUBIC_BIT_EXT,
    eSampledImageFilterCubicIMG           = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_CUBIC_BIT_IMG,
    eFragmentDensityMapEXT                = VK_FORMAT_FEATURE_FRAGMENT_DENSITY_MAP_BIT_EXT,
    eFragmentShadingRateAttachmentKHR     = VK_FORMAT_FEATURE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
    eVideoEncodeInputKHR                  = VK_FORMAT_FEATURE_VIDEO_ENCODE_INPUT_BIT_KHR,
    eVideoEncodeDpbKHR                    = VK_FORMAT_FEATURE_VIDEO_ENCODE_DPB_BIT_KHR
  };

  // wrapper using for bitmask VkFormatFeatureFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkFormatFeatureFlags.html
  using FormatFeatureFlags = Flags<FormatFeatureFlagBits>;

  template <>
  struct FlagTraits<FormatFeatureFlagBits>
  {
    using WrappedType                                                 = VkFormatFeatureFlagBits;
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
      FormatFeatureFlagBits::eFragmentShadingRateAttachmentKHR | FormatFeatureFlagBits::eVideoEncodeInputKHR | FormatFeatureFlagBits::eVideoEncodeDpbKHR;
  };

  // wrapper class for enum VkImageCreateFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageCreateFlagBits.html
  enum class ImageCreateFlagBits : VkImageCreateFlags
  {
    eSparseBinding                        = VK_IMAGE_CREATE_SPARSE_BINDING_BIT,
    eSparseResidency                      = VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
    eSparseAliased                        = VK_IMAGE_CREATE_SPARSE_ALIASED_BIT,
    eMutableFormat                        = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
    eCubeCompatible                       = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
    eAlias                                = VK_IMAGE_CREATE_ALIAS_BIT,
    eAliasKHR                             = VK_IMAGE_CREATE_ALIAS_BIT_KHR,
    eSplitInstanceBindRegions             = VK_IMAGE_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT,
    eSplitInstanceBindRegionsKHR          = VK_IMAGE_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT_KHR,
    e2DArrayCompatible                    = VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT,
    e2DArrayCompatibleKHR                 = VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT_KHR,
    eBlockTexelViewCompatible             = VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT,
    eBlockTexelViewCompatibleKHR          = VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT_KHR,
    eExtendedUsage                        = VK_IMAGE_CREATE_EXTENDED_USAGE_BIT,
    eExtendedUsageKHR                     = VK_IMAGE_CREATE_EXTENDED_USAGE_BIT_KHR,
    eProtected                            = VK_IMAGE_CREATE_PROTECTED_BIT,
    eDisjoint                             = VK_IMAGE_CREATE_DISJOINT_BIT,
    eDisjointKHR                          = VK_IMAGE_CREATE_DISJOINT_BIT_KHR,
    eCornerSampledNV                      = VK_IMAGE_CREATE_CORNER_SAMPLED_BIT_NV,
    eSampleLocationsCompatibleDepthEXT    = VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT,
    eSubsampledEXT                        = VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT,
    eDescriptorBufferCaptureReplayEXT     = VK_IMAGE_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_EXT,
    eMultisampledRenderToSingleSampledEXT = VK_IMAGE_CREATE_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_BIT_EXT,
    e2DViewCompatibleEXT                  = VK_IMAGE_CREATE_2D_VIEW_COMPATIBLE_BIT_EXT,
    eVideoProfileIndependentKHR           = VK_IMAGE_CREATE_VIDEO_PROFILE_INDEPENDENT_BIT_KHR,
    eFragmentDensityMapOffsetEXT          = VK_IMAGE_CREATE_FRAGMENT_DENSITY_MAP_OFFSET_BIT_EXT,
    eFragmentDensityMapOffsetQCOM         = VK_IMAGE_CREATE_FRAGMENT_DENSITY_MAP_OFFSET_BIT_QCOM
  };

  // wrapper using for bitmask VkImageCreateFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageCreateFlags.html
  using ImageCreateFlags = Flags<ImageCreateFlagBits>;

  template <>
  struct FlagTraits<ImageCreateFlagBits>
  {
    using WrappedType                                               = VkImageCreateFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ImageCreateFlags allFlags =
      ImageCreateFlagBits::eSparseBinding | ImageCreateFlagBits::eSparseResidency | ImageCreateFlagBits::eSparseAliased | ImageCreateFlagBits::eMutableFormat |
      ImageCreateFlagBits::eCubeCompatible | ImageCreateFlagBits::eAlias | ImageCreateFlagBits::eSplitInstanceBindRegions |
      ImageCreateFlagBits::e2DArrayCompatible | ImageCreateFlagBits::eBlockTexelViewCompatible | ImageCreateFlagBits::eExtendedUsage |
      ImageCreateFlagBits::eProtected | ImageCreateFlagBits::eDisjoint | ImageCreateFlagBits::eCornerSampledNV |
      ImageCreateFlagBits::eSampleLocationsCompatibleDepthEXT | ImageCreateFlagBits::eSubsampledEXT | ImageCreateFlagBits::eDescriptorBufferCaptureReplayEXT |
      ImageCreateFlagBits::eMultisampledRenderToSingleSampledEXT | ImageCreateFlagBits::e2DViewCompatibleEXT |
      ImageCreateFlagBits::eVideoProfileIndependentKHR | ImageCreateFlagBits::eFragmentDensityMapOffsetEXT;
  };

  // wrapper class for enum VkImageTiling, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageTiling.html
  enum class ImageTiling
  {
    eOptimal              = VK_IMAGE_TILING_OPTIMAL,
    eLinear               = VK_IMAGE_TILING_LINEAR,
    eDrmFormatModifierEXT = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT
  };

  // wrapper class for enum VkImageType, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageType.html
  enum class ImageType
  {
    e1D = VK_IMAGE_TYPE_1D,
    e2D = VK_IMAGE_TYPE_2D,
    e3D = VK_IMAGE_TYPE_3D
  };

  // wrapper class for enum VkImageUsageFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageUsageFlagBits.html
  enum class ImageUsageFlagBits : VkImageUsageFlags
  {
    eTransferSrc                        = VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
    eTransferDst                        = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
    eSampled                            = VK_IMAGE_USAGE_SAMPLED_BIT,
    eStorage                            = VK_IMAGE_USAGE_STORAGE_BIT,
    eColorAttachment                    = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    eDepthStencilAttachment             = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
    eTransientAttachment                = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
    eInputAttachment                    = VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
    eHostTransfer                       = VK_IMAGE_USAGE_HOST_TRANSFER_BIT,
    eHostTransferEXT                    = VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT,
    eVideoDecodeDstKHR                  = VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR,
    eVideoDecodeSrcKHR                  = VK_IMAGE_USAGE_VIDEO_DECODE_SRC_BIT_KHR,
    eVideoDecodeDpbKHR                  = VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR,
    eFragmentDensityMapEXT              = VK_IMAGE_USAGE_FRAGMENT_DENSITY_MAP_BIT_EXT,
    eFragmentShadingRateAttachmentKHR   = VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
    eShadingRateImageNV                 = VK_IMAGE_USAGE_SHADING_RATE_IMAGE_BIT_NV,
    eVideoEncodeDstKHR                  = VK_IMAGE_USAGE_VIDEO_ENCODE_DST_BIT_KHR,
    eVideoEncodeSrcKHR                  = VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR,
    eVideoEncodeDpbKHR                  = VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR,
    eAttachmentFeedbackLoopEXT          = VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT,
    eInvocationMaskHUAWEI               = VK_IMAGE_USAGE_INVOCATION_MASK_BIT_HUAWEI,
    eSampleWeightQCOM                   = VK_IMAGE_USAGE_SAMPLE_WEIGHT_BIT_QCOM,
    eSampleBlockMatchQCOM               = VK_IMAGE_USAGE_SAMPLE_BLOCK_MATCH_BIT_QCOM,
    eTensorAliasingARM                  = VK_IMAGE_USAGE_TENSOR_ALIASING_BIT_ARM,
    eTileMemoryQCOM                     = VK_IMAGE_USAGE_TILE_MEMORY_BIT_QCOM,
    eVideoEncodeQuantizationDeltaMapKHR = VK_IMAGE_USAGE_VIDEO_ENCODE_QUANTIZATION_DELTA_MAP_BIT_KHR,
    eVideoEncodeEmphasisMapKHR          = VK_IMAGE_USAGE_VIDEO_ENCODE_EMPHASIS_MAP_BIT_KHR
  };

  // wrapper using for bitmask VkImageUsageFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageUsageFlags.html
  using ImageUsageFlags = Flags<ImageUsageFlagBits>;

  template <>
  struct FlagTraits<ImageUsageFlagBits>
  {
    using WrappedType                                              = VkImageUsageFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool            isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ImageUsageFlags allFlags =
      ImageUsageFlagBits::eTransferSrc | ImageUsageFlagBits::eTransferDst | ImageUsageFlagBits::eSampled | ImageUsageFlagBits::eStorage |
      ImageUsageFlagBits::eColorAttachment | ImageUsageFlagBits::eDepthStencilAttachment | ImageUsageFlagBits::eTransientAttachment |
      ImageUsageFlagBits::eInputAttachment | ImageUsageFlagBits::eHostTransfer | ImageUsageFlagBits::eVideoDecodeDstKHR |
      ImageUsageFlagBits::eVideoDecodeSrcKHR | ImageUsageFlagBits::eVideoDecodeDpbKHR | ImageUsageFlagBits::eFragmentDensityMapEXT |
      ImageUsageFlagBits::eFragmentShadingRateAttachmentKHR | ImageUsageFlagBits::eVideoEncodeDstKHR | ImageUsageFlagBits::eVideoEncodeSrcKHR |
      ImageUsageFlagBits::eVideoEncodeDpbKHR | ImageUsageFlagBits::eAttachmentFeedbackLoopEXT | ImageUsageFlagBits::eInvocationMaskHUAWEI |
      ImageUsageFlagBits::eSampleWeightQCOM | ImageUsageFlagBits::eSampleBlockMatchQCOM | ImageUsageFlagBits::eTensorAliasingARM |
      ImageUsageFlagBits::eTileMemoryQCOM | ImageUsageFlagBits::eVideoEncodeQuantizationDeltaMapKHR | ImageUsageFlagBits::eVideoEncodeEmphasisMapKHR;
  };

  // wrapper class for enum VkInstanceCreateFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkInstanceCreateFlagBits.html
  enum class InstanceCreateFlagBits : VkInstanceCreateFlags
  {
    eEnumeratePortabilityKHR = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
  };

  // wrapper using for bitmask VkInstanceCreateFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkInstanceCreateFlags.html
  using InstanceCreateFlags = Flags<InstanceCreateFlagBits>;

  template <>
  struct FlagTraits<InstanceCreateFlagBits>
  {
    using WrappedType                                                  = VkInstanceCreateFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR InstanceCreateFlags allFlags  = InstanceCreateFlagBits::eEnumeratePortabilityKHR;
  };

  // wrapper class for enum VkInternalAllocationType, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkInternalAllocationType.html
  enum class InternalAllocationType
  {
    eExecutable = VK_INTERNAL_ALLOCATION_TYPE_EXECUTABLE
  };

  // wrapper class for enum VkMemoryHeapFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryHeapFlagBits.html
  enum class MemoryHeapFlagBits : VkMemoryHeapFlags
  {
    eDeviceLocal      = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT,
    eMultiInstance    = VK_MEMORY_HEAP_MULTI_INSTANCE_BIT,
    eMultiInstanceKHR = VK_MEMORY_HEAP_MULTI_INSTANCE_BIT_KHR,
    eTileMemoryQCOM   = VK_MEMORY_HEAP_TILE_MEMORY_BIT_QCOM
  };

  // wrapper using for bitmask VkMemoryHeapFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryHeapFlags.html
  using MemoryHeapFlags = Flags<MemoryHeapFlagBits>;

  template <>
  struct FlagTraits<MemoryHeapFlagBits>
  {
    using WrappedType                                              = VkMemoryHeapFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool            isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR MemoryHeapFlags allFlags =
      MemoryHeapFlagBits::eDeviceLocal | MemoryHeapFlagBits::eMultiInstance | MemoryHeapFlagBits::eTileMemoryQCOM;
  };

  // wrapper class for enum VkMemoryPropertyFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryPropertyFlagBits.html
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

  // wrapper using for bitmask VkMemoryPropertyFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryPropertyFlags.html
  using MemoryPropertyFlags = Flags<MemoryPropertyFlagBits>;

  template <>
  struct FlagTraits<MemoryPropertyFlagBits>
  {
    using WrappedType                                                  = VkMemoryPropertyFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR MemoryPropertyFlags allFlags =
      MemoryPropertyFlagBits::eDeviceLocal | MemoryPropertyFlagBits::eHostVisible | MemoryPropertyFlagBits::eHostCoherent |
      MemoryPropertyFlagBits::eHostCached | MemoryPropertyFlagBits::eLazilyAllocated | MemoryPropertyFlagBits::eProtected |
      MemoryPropertyFlagBits::eDeviceCoherentAMD | MemoryPropertyFlagBits::eDeviceUncachedAMD | MemoryPropertyFlagBits::eRdmaCapableNV;
  };

  // wrapper class for enum VkPhysicalDeviceType, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceType.html
  enum class PhysicalDeviceType
  {
    eOther         = VK_PHYSICAL_DEVICE_TYPE_OTHER,
    eIntegratedGpu = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
    eDiscreteGpu   = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
    eVirtualGpu    = VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU,
    eCpu           = VK_PHYSICAL_DEVICE_TYPE_CPU
  };

  // wrapper class for enum VkQueueFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkQueueFlagBits.html
  enum class QueueFlagBits : VkQueueFlags
  {
    eGraphics       = VK_QUEUE_GRAPHICS_BIT,
    eCompute        = VK_QUEUE_COMPUTE_BIT,
    eTransfer       = VK_QUEUE_TRANSFER_BIT,
    eSparseBinding  = VK_QUEUE_SPARSE_BINDING_BIT,
    eProtected      = VK_QUEUE_PROTECTED_BIT,
    eVideoDecodeKHR = VK_QUEUE_VIDEO_DECODE_BIT_KHR,
    eVideoEncodeKHR = VK_QUEUE_VIDEO_ENCODE_BIT_KHR,
    eOpticalFlowNV  = VK_QUEUE_OPTICAL_FLOW_BIT_NV,
    eDataGraphARM   = VK_QUEUE_DATA_GRAPH_BIT_ARM
  };

  // wrapper using for bitmask VkQueueFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkQueueFlags.html
  using QueueFlags = Flags<QueueFlagBits>;

  template <>
  struct FlagTraits<QueueFlagBits>
  {
    using WrappedType                                         = VkQueueFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR QueueFlags allFlags  = QueueFlagBits::eGraphics | QueueFlagBits::eCompute | QueueFlagBits::eTransfer |
                                                               QueueFlagBits::eSparseBinding | QueueFlagBits::eProtected | QueueFlagBits::eVideoDecodeKHR |
                                                               QueueFlagBits::eVideoEncodeKHR | QueueFlagBits::eOpticalFlowNV | QueueFlagBits::eDataGraphARM;
  };

  // wrapper class for enum VkSampleCountFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSampleCountFlagBits.html
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

  // wrapper using for bitmask VkSampleCountFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSampleCountFlags.html
  using SampleCountFlags = Flags<SampleCountFlagBits>;

  template <>
  struct FlagTraits<SampleCountFlagBits>
  {
    using WrappedType                                               = VkSampleCountFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SampleCountFlags allFlags  = SampleCountFlagBits::e1 | SampleCountFlagBits::e2 | SampleCountFlagBits::e4 |
                                                                     SampleCountFlagBits::e8 | SampleCountFlagBits::e16 | SampleCountFlagBits::e32 |
                                                                     SampleCountFlagBits::e64;
  };

  // wrapper class for enum VkSystemAllocationScope, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSystemAllocationScope.html
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

  // wrapper using for bitmask VkDeviceCreateFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceCreateFlags.html
  using DeviceCreateFlags = Flags<DeviceCreateFlagBits>;

  template <>
  struct FlagTraits<DeviceCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool              isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DeviceCreateFlags allFlags  = {};
  };

  // wrapper class for enum VkDeviceQueueCreateFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceQueueCreateFlagBits.html
  enum class DeviceQueueCreateFlagBits : VkDeviceQueueCreateFlags
  {
    eProtected = VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT
  };

  // wrapper using for bitmask VkDeviceQueueCreateFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceQueueCreateFlags.html
  using DeviceQueueCreateFlags = Flags<DeviceQueueCreateFlagBits>;

  template <>
  struct FlagTraits<DeviceQueueCreateFlagBits>
  {
    using WrappedType                                                     = VkDeviceQueueCreateFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DeviceQueueCreateFlags allFlags  = DeviceQueueCreateFlagBits::eProtected;
  };

  // wrapper class for enum VkPipelineStageFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineStageFlagBits.html
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
    eNoneKHR                          = VK_PIPELINE_STAGE_NONE_KHR,
    eTransformFeedbackEXT             = VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
    eConditionalRenderingEXT          = VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT,
    eAccelerationStructureBuildKHR    = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    eAccelerationStructureBuildNV     = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
    eRayTracingShaderKHR              = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
    eRayTracingShaderNV               = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_NV,
    eFragmentDensityProcessEXT        = VK_PIPELINE_STAGE_FRAGMENT_DENSITY_PROCESS_BIT_EXT,
    eFragmentShadingRateAttachmentKHR = VK_PIPELINE_STAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
    eShadingRateImageNV               = VK_PIPELINE_STAGE_SHADING_RATE_IMAGE_BIT_NV,
    eTaskShaderEXT                    = VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT,
    eTaskShaderNV                     = VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV,
    eMeshShaderEXT                    = VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT,
    eMeshShaderNV                     = VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV,
    eCommandPreprocessEXT             = VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_EXT,
    eCommandPreprocessNV              = VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_NV
  };

  // wrapper using for bitmask VkPipelineStageFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineStageFlags.html
  using PipelineStageFlags = Flags<PipelineStageFlagBits>;

  template <>
  struct FlagTraits<PipelineStageFlagBits>
  {
    using WrappedType                                                 = VkPipelineStageFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool               isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineStageFlags allFlags =
      PipelineStageFlagBits::eTopOfPipe | PipelineStageFlagBits::eDrawIndirect | PipelineStageFlagBits::eVertexInput | PipelineStageFlagBits::eVertexShader |
      PipelineStageFlagBits::eTessellationControlShader | PipelineStageFlagBits::eTessellationEvaluationShader | PipelineStageFlagBits::eGeometryShader |
      PipelineStageFlagBits::eFragmentShader | PipelineStageFlagBits::eEarlyFragmentTests | PipelineStageFlagBits::eLateFragmentTests |
      PipelineStageFlagBits::eColorAttachmentOutput | PipelineStageFlagBits::eComputeShader | PipelineStageFlagBits::eTransfer |
      PipelineStageFlagBits::eBottomOfPipe | PipelineStageFlagBits::eHost | PipelineStageFlagBits::eAllGraphics | PipelineStageFlagBits::eAllCommands |
      PipelineStageFlagBits::eNone | PipelineStageFlagBits::eTransformFeedbackEXT | PipelineStageFlagBits::eConditionalRenderingEXT |
      PipelineStageFlagBits::eAccelerationStructureBuildKHR | PipelineStageFlagBits::eRayTracingShaderKHR | PipelineStageFlagBits::eFragmentDensityProcessEXT |
      PipelineStageFlagBits::eFragmentShadingRateAttachmentKHR | PipelineStageFlagBits::eTaskShaderEXT | PipelineStageFlagBits::eMeshShaderEXT |
      PipelineStageFlagBits::eCommandPreprocessEXT;
  };

  // wrapper class for enum VkMemoryMapFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryMapFlagBits.html
  enum class MemoryMapFlagBits : VkMemoryMapFlags
  {
    ePlacedEXT = VK_MEMORY_MAP_PLACED_BIT_EXT
  };

  // wrapper using for bitmask VkMemoryMapFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryMapFlags.html
  using MemoryMapFlags = Flags<MemoryMapFlagBits>;

  template <>
  struct FlagTraits<MemoryMapFlagBits>
  {
    using WrappedType                                             = VkMemoryMapFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR MemoryMapFlags allFlags  = MemoryMapFlagBits::ePlacedEXT;
  };

  // wrapper class for enum VkImageAspectFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageAspectFlagBits.html
  enum class ImageAspectFlagBits : VkImageAspectFlags
  {
    eColor           = VK_IMAGE_ASPECT_COLOR_BIT,
    eDepth           = VK_IMAGE_ASPECT_DEPTH_BIT,
    eStencil         = VK_IMAGE_ASPECT_STENCIL_BIT,
    eMetadata        = VK_IMAGE_ASPECT_METADATA_BIT,
    ePlane0          = VK_IMAGE_ASPECT_PLANE_0_BIT,
    ePlane0KHR       = VK_IMAGE_ASPECT_PLANE_0_BIT_KHR,
    ePlane1          = VK_IMAGE_ASPECT_PLANE_1_BIT,
    ePlane1KHR       = VK_IMAGE_ASPECT_PLANE_1_BIT_KHR,
    ePlane2          = VK_IMAGE_ASPECT_PLANE_2_BIT,
    ePlane2KHR       = VK_IMAGE_ASPECT_PLANE_2_BIT_KHR,
    eNone            = VK_IMAGE_ASPECT_NONE,
    eNoneKHR         = VK_IMAGE_ASPECT_NONE_KHR,
    eMemoryPlane0EXT = VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT,
    eMemoryPlane1EXT = VK_IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT,
    eMemoryPlane2EXT = VK_IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT,
    eMemoryPlane3EXT = VK_IMAGE_ASPECT_MEMORY_PLANE_3_BIT_EXT
  };

  // wrapper using for bitmask VkImageAspectFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageAspectFlags.html
  using ImageAspectFlags = Flags<ImageAspectFlagBits>;

  template <>
  struct FlagTraits<ImageAspectFlagBits>
  {
    using WrappedType                                               = VkImageAspectFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ImageAspectFlags allFlags = ImageAspectFlagBits::eColor | ImageAspectFlagBits::eDepth | ImageAspectFlagBits::eStencil |
                                                                     ImageAspectFlagBits::eMetadata | ImageAspectFlagBits::ePlane0 |
                                                                     ImageAspectFlagBits::ePlane1 | ImageAspectFlagBits::ePlane2 | ImageAspectFlagBits::eNone |
                                                                     ImageAspectFlagBits::eMemoryPlane0EXT | ImageAspectFlagBits::eMemoryPlane1EXT |
                                                                     ImageAspectFlagBits::eMemoryPlane2EXT | ImageAspectFlagBits::eMemoryPlane3EXT;
  };

  // wrapper class for enum VkSparseImageFormatFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSparseImageFormatFlagBits.html
  enum class SparseImageFormatFlagBits : VkSparseImageFormatFlags
  {
    eSingleMiptail        = VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT,
    eAlignedMipSize       = VK_SPARSE_IMAGE_FORMAT_ALIGNED_MIP_SIZE_BIT,
    eNonstandardBlockSize = VK_SPARSE_IMAGE_FORMAT_NONSTANDARD_BLOCK_SIZE_BIT
  };

  // wrapper using for bitmask VkSparseImageFormatFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSparseImageFormatFlags.html
  using SparseImageFormatFlags = Flags<SparseImageFormatFlagBits>;

  template <>
  struct FlagTraits<SparseImageFormatFlagBits>
  {
    using WrappedType                                                     = VkSparseImageFormatFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SparseImageFormatFlags allFlags =
      SparseImageFormatFlagBits::eSingleMiptail | SparseImageFormatFlagBits::eAlignedMipSize | SparseImageFormatFlagBits::eNonstandardBlockSize;
  };

  // wrapper class for enum VkSparseMemoryBindFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSparseMemoryBindFlagBits.html
  enum class SparseMemoryBindFlagBits : VkSparseMemoryBindFlags
  {
    eMetadata = VK_SPARSE_MEMORY_BIND_METADATA_BIT
  };

  // wrapper using for bitmask VkSparseMemoryBindFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSparseMemoryBindFlags.html
  using SparseMemoryBindFlags = Flags<SparseMemoryBindFlagBits>;

  template <>
  struct FlagTraits<SparseMemoryBindFlagBits>
  {
    using WrappedType                                                    = VkSparseMemoryBindFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                  isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SparseMemoryBindFlags allFlags  = SparseMemoryBindFlagBits::eMetadata;
  };

  // wrapper class for enum VkFenceCreateFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkFenceCreateFlagBits.html
  enum class FenceCreateFlagBits : VkFenceCreateFlags
  {
    eSignaled = VK_FENCE_CREATE_SIGNALED_BIT
  };

  // wrapper using for bitmask VkFenceCreateFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkFenceCreateFlags.html
  using FenceCreateFlags = Flags<FenceCreateFlagBits>;

  template <>
  struct FlagTraits<FenceCreateFlagBits>
  {
    using WrappedType                                               = VkFenceCreateFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR FenceCreateFlags allFlags  = FenceCreateFlagBits::eSignaled;
  };

  enum class SemaphoreCreateFlagBits : VkSemaphoreCreateFlags
  {
  };

  // wrapper using for bitmask VkSemaphoreCreateFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSemaphoreCreateFlags.html
  using SemaphoreCreateFlags = Flags<SemaphoreCreateFlagBits>;

  template <>
  struct FlagTraits<SemaphoreCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SemaphoreCreateFlags allFlags  = {};
  };

  // wrapper class for enum VkQueryPoolCreateFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkQueryPoolCreateFlagBits.html
  enum class QueryPoolCreateFlagBits : VkQueryPoolCreateFlags
  {
    eResetKHR = VK_QUERY_POOL_CREATE_RESET_BIT_KHR
  };

  // wrapper using for bitmask VkQueryPoolCreateFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkQueryPoolCreateFlags.html
  using QueryPoolCreateFlags = Flags<QueryPoolCreateFlagBits>;

  template <>
  struct FlagTraits<QueryPoolCreateFlagBits>
  {
    using WrappedType                                                   = VkQueryPoolCreateFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR QueryPoolCreateFlags allFlags  = QueryPoolCreateFlagBits::eResetKHR;
  };

  // wrapper class for enum VkQueryResultFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkQueryResultFlagBits.html
  enum class QueryResultFlagBits : VkQueryResultFlags
  {
    e64               = VK_QUERY_RESULT_64_BIT,
    eWait             = VK_QUERY_RESULT_WAIT_BIT,
    eWithAvailability = VK_QUERY_RESULT_WITH_AVAILABILITY_BIT,
    ePartial          = VK_QUERY_RESULT_PARTIAL_BIT,
    eWithStatusKHR    = VK_QUERY_RESULT_WITH_STATUS_BIT_KHR
  };

  // wrapper using for bitmask VkQueryResultFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkQueryResultFlags.html
  using QueryResultFlags = Flags<QueryResultFlagBits>;

  template <>
  struct FlagTraits<QueryResultFlagBits>
  {
    using WrappedType                                               = VkQueryResultFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR QueryResultFlags allFlags  = QueryResultFlagBits::e64 | QueryResultFlagBits::eWait |
                                                                     QueryResultFlagBits::eWithAvailability | QueryResultFlagBits::ePartial |
                                                                     QueryResultFlagBits::eWithStatusKHR;
  };

  // wrapper class for enum VkQueryType, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkQueryType.html
  enum class QueryType
  {
    eOcclusion                                                = VK_QUERY_TYPE_OCCLUSION,
    ePipelineStatistics                                       = VK_QUERY_TYPE_PIPELINE_STATISTICS,
    eTimestamp                                                = VK_QUERY_TYPE_TIMESTAMP,
    eResultStatusOnlyKHR                                      = VK_QUERY_TYPE_RESULT_STATUS_ONLY_KHR,
    eTransformFeedbackStreamEXT                               = VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT,
    ePerformanceQueryKHR                                      = VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR,
    eAccelerationStructureCompactedSizeKHR                    = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
    eAccelerationStructureSerializationSizeKHR                = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR,
    eAccelerationStructureCompactedSizeNV                     = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_NV,
    ePerformanceQueryINTEL                                    = VK_QUERY_TYPE_PERFORMANCE_QUERY_INTEL,
    eVideoEncodeFeedbackKHR                                   = VK_QUERY_TYPE_VIDEO_ENCODE_FEEDBACK_KHR,
    eMeshPrimitivesGeneratedEXT                               = VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT,
    ePrimitivesGeneratedEXT                                   = VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT,
    eAccelerationStructureSerializationBottomLevelPointersKHR = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR,
    eAccelerationStructureSizeKHR                             = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR,
    eMicromapSerializationSizeEXT                             = VK_QUERY_TYPE_MICROMAP_SERIALIZATION_SIZE_EXT,
    eMicromapCompactedSizeEXT                                 = VK_QUERY_TYPE_MICROMAP_COMPACTED_SIZE_EXT
  };

  // wrapper class for enum VkBufferCreateFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkBufferCreateFlagBits.html
  enum class BufferCreateFlagBits : VkBufferCreateFlags
  {
    eSparseBinding                    = VK_BUFFER_CREATE_SPARSE_BINDING_BIT,
    eSparseResidency                  = VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT,
    eSparseAliased                    = VK_BUFFER_CREATE_SPARSE_ALIASED_BIT,
    eProtected                        = VK_BUFFER_CREATE_PROTECTED_BIT,
    eDeviceAddressCaptureReplay       = VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT,
    eDeviceAddressCaptureReplayEXT    = VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT_EXT,
    eDeviceAddressCaptureReplayKHR    = VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT_KHR,
    eDescriptorBufferCaptureReplayEXT = VK_BUFFER_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_EXT,
    eVideoProfileIndependentKHR       = VK_BUFFER_CREATE_VIDEO_PROFILE_INDEPENDENT_BIT_KHR
  };

  // wrapper using for bitmask VkBufferCreateFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkBufferCreateFlags.html
  using BufferCreateFlags = Flags<BufferCreateFlagBits>;

  template <>
  struct FlagTraits<BufferCreateFlagBits>
  {
    using WrappedType                                                = VkBufferCreateFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool              isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR BufferCreateFlags allFlags =
      BufferCreateFlagBits::eSparseBinding | BufferCreateFlagBits::eSparseResidency | BufferCreateFlagBits::eSparseAliased | BufferCreateFlagBits::eProtected |
      BufferCreateFlagBits::eDeviceAddressCaptureReplay | BufferCreateFlagBits::eDescriptorBufferCaptureReplayEXT |
      BufferCreateFlagBits::eVideoProfileIndependentKHR;
  };

  // wrapper class for enum VkBufferUsageFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkBufferUsageFlagBits.html
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
    eShaderDeviceAddressEXT            = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_EXT,
    eShaderDeviceAddressKHR            = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR,
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
    eVideoEncodeDstKHR                          = VK_BUFFER_USAGE_VIDEO_ENCODE_DST_BIT_KHR,
    eVideoEncodeSrcKHR                          = VK_BUFFER_USAGE_VIDEO_ENCODE_SRC_BIT_KHR,
    eSamplerDescriptorBufferEXT                 = VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT,
    eResourceDescriptorBufferEXT                = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT,
    ePushDescriptorsDescriptorBufferEXT         = VK_BUFFER_USAGE_PUSH_DESCRIPTORS_DESCRIPTOR_BUFFER_BIT_EXT,
    eMicromapBuildInputReadOnlyEXT              = VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT,
    eMicromapStorageEXT                         = VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT,
    eTileMemoryQCOM                             = VK_BUFFER_USAGE_TILE_MEMORY_BIT_QCOM
  };

  // wrapper using for bitmask VkBufferUsageFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkBufferUsageFlags.html
  using BufferUsageFlags = Flags<BufferUsageFlagBits>;

  template <>
  struct FlagTraits<BufferUsageFlagBits>
  {
    using WrappedType                                               = VkBufferUsageFlagBits;
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
      BufferUsageFlagBits::eShaderBindingTableKHR | BufferUsageFlagBits::eVideoEncodeDstKHR | BufferUsageFlagBits::eVideoEncodeSrcKHR |
      BufferUsageFlagBits::eSamplerDescriptorBufferEXT | BufferUsageFlagBits::eResourceDescriptorBufferEXT |
      BufferUsageFlagBits::ePushDescriptorsDescriptorBufferEXT | BufferUsageFlagBits::eMicromapBuildInputReadOnlyEXT |
      BufferUsageFlagBits::eMicromapStorageEXT | BufferUsageFlagBits::eTileMemoryQCOM;
  };

  // wrapper class for enum VkSharingMode, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSharingMode.html
  enum class SharingMode
  {
    eExclusive  = VK_SHARING_MODE_EXCLUSIVE,
    eConcurrent = VK_SHARING_MODE_CONCURRENT
  };

  // wrapper class for enum VkImageLayout, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageLayout.html
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
    eDepthReadOnlyStencilAttachmentOptimalKHR = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL_KHR,
    eDepthAttachmentStencilReadOnlyOptimal    = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
    eDepthAttachmentStencilReadOnlyOptimalKHR = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL_KHR,
    eDepthAttachmentOptimal                   = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
    eDepthAttachmentOptimalKHR                = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL_KHR,
    eDepthReadOnlyOptimal                     = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
    eDepthReadOnlyOptimalKHR                  = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL_KHR,
    eStencilAttachmentOptimal                 = VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL,
    eStencilAttachmentOptimalKHR              = VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL_KHR,
    eStencilReadOnlyOptimal                   = VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL,
    eStencilReadOnlyOptimalKHR                = VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL_KHR,
    eReadOnlyOptimal                          = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
    eReadOnlyOptimalKHR                       = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL_KHR,
    eAttachmentOptimal                        = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
    eAttachmentOptimalKHR                     = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR,
    eRenderingLocalRead                       = VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ,
    eRenderingLocalReadKHR                    = VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ_KHR,
    ePresentSrcKHR                            = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    eVideoDecodeDstKHR                        = VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR,
    eVideoDecodeSrcKHR                        = VK_IMAGE_LAYOUT_VIDEO_DECODE_SRC_KHR,
    eVideoDecodeDpbKHR                        = VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR,
    eSharedPresentKHR                         = VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR,
    eFragmentDensityMapOptimalEXT             = VK_IMAGE_LAYOUT_FRAGMENT_DENSITY_MAP_OPTIMAL_EXT,
    eFragmentShadingRateAttachmentOptimalKHR  = VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR,
    eShadingRateOptimalNV                     = VK_IMAGE_LAYOUT_SHADING_RATE_OPTIMAL_NV,
    eVideoEncodeDstKHR                        = VK_IMAGE_LAYOUT_VIDEO_ENCODE_DST_KHR,
    eVideoEncodeSrcKHR                        = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR,
    eVideoEncodeDpbKHR                        = VK_IMAGE_LAYOUT_VIDEO_ENCODE_DPB_KHR,
    eAttachmentFeedbackLoopOptimalEXT         = VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT,
    eTensorAliasingARM                        = VK_IMAGE_LAYOUT_TENSOR_ALIASING_ARM,
    eVideoEncodeQuantizationMapKHR            = VK_IMAGE_LAYOUT_VIDEO_ENCODE_QUANTIZATION_MAP_KHR,
    eZeroInitializedEXT                       = VK_IMAGE_LAYOUT_ZERO_INITIALIZED_EXT
  };

  // wrapper class for enum VkComponentSwizzle, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkComponentSwizzle.html
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

  // wrapper class for enum VkImageViewCreateFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageViewCreateFlagBits.html
  enum class ImageViewCreateFlagBits : VkImageViewCreateFlags
  {
    eFragmentDensityMapDynamicEXT     = VK_IMAGE_VIEW_CREATE_FRAGMENT_DENSITY_MAP_DYNAMIC_BIT_EXT,
    eDescriptorBufferCaptureReplayEXT = VK_IMAGE_VIEW_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_EXT,
    eFragmentDensityMapDeferredEXT    = VK_IMAGE_VIEW_CREATE_FRAGMENT_DENSITY_MAP_DEFERRED_BIT_EXT
  };

  // wrapper using for bitmask VkImageViewCreateFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageViewCreateFlags.html
  using ImageViewCreateFlags = Flags<ImageViewCreateFlagBits>;

  template <>
  struct FlagTraits<ImageViewCreateFlagBits>
  {
    using WrappedType                                                   = VkImageViewCreateFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ImageViewCreateFlags allFlags  = ImageViewCreateFlagBits::eFragmentDensityMapDynamicEXT |
                                                                         ImageViewCreateFlagBits::eDescriptorBufferCaptureReplayEXT |
                                                                         ImageViewCreateFlagBits::eFragmentDensityMapDeferredEXT;
  };

  // wrapper class for enum VkImageViewType, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageViewType.html
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

  // wrapper class for enum VkAccessFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccessFlagBits.html
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
    eNoneKHR                              = VK_ACCESS_NONE_KHR,
    eTransformFeedbackWriteEXT            = VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT,
    eTransformFeedbackCounterReadEXT      = VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT,
    eTransformFeedbackCounterWriteEXT     = VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT,
    eConditionalRenderingReadEXT          = VK_ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT,
    eColorAttachmentReadNoncoherentEXT    = VK_ACCESS_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT,
    eAccelerationStructureReadKHR         = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    eAccelerationStructureReadNV          = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV,
    eAccelerationStructureWriteKHR        = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
    eAccelerationStructureWriteNV         = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV,
    eFragmentDensityMapReadEXT            = VK_ACCESS_FRAGMENT_DENSITY_MAP_READ_BIT_EXT,
    eFragmentShadingRateAttachmentReadKHR = VK_ACCESS_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR,
    eShadingRateImageReadNV               = VK_ACCESS_SHADING_RATE_IMAGE_READ_BIT_NV,
    eCommandPreprocessReadEXT             = VK_ACCESS_COMMAND_PREPROCESS_READ_BIT_EXT,
    eCommandPreprocessReadNV              = VK_ACCESS_COMMAND_PREPROCESS_READ_BIT_NV,
    eCommandPreprocessWriteEXT            = VK_ACCESS_COMMAND_PREPROCESS_WRITE_BIT_EXT,
    eCommandPreprocessWriteNV             = VK_ACCESS_COMMAND_PREPROCESS_WRITE_BIT_NV
  };

  // wrapper using for bitmask VkAccessFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccessFlags.html
  using AccessFlags = Flags<AccessFlagBits>;

  template <>
  struct FlagTraits<AccessFlagBits>
  {
    using WrappedType                                          = VkAccessFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool        isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR AccessFlags allFlags =
      AccessFlagBits::eIndirectCommandRead | AccessFlagBits::eIndexRead | AccessFlagBits::eVertexAttributeRead | AccessFlagBits::eUniformRead |
      AccessFlagBits::eInputAttachmentRead | AccessFlagBits::eShaderRead | AccessFlagBits::eShaderWrite | AccessFlagBits::eColorAttachmentRead |
      AccessFlagBits::eColorAttachmentWrite | AccessFlagBits::eDepthStencilAttachmentRead | AccessFlagBits::eDepthStencilAttachmentWrite |
      AccessFlagBits::eTransferRead | AccessFlagBits::eTransferWrite | AccessFlagBits::eHostRead | AccessFlagBits::eHostWrite | AccessFlagBits::eMemoryRead |
      AccessFlagBits::eMemoryWrite | AccessFlagBits::eNone | AccessFlagBits::eTransformFeedbackWriteEXT | AccessFlagBits::eTransformFeedbackCounterReadEXT |
      AccessFlagBits::eTransformFeedbackCounterWriteEXT | AccessFlagBits::eConditionalRenderingReadEXT | AccessFlagBits::eColorAttachmentReadNoncoherentEXT |
      AccessFlagBits::eAccelerationStructureReadKHR | AccessFlagBits::eAccelerationStructureWriteKHR | AccessFlagBits::eFragmentDensityMapReadEXT |
      AccessFlagBits::eFragmentShadingRateAttachmentReadKHR | AccessFlagBits::eCommandPreprocessReadEXT | AccessFlagBits::eCommandPreprocessWriteEXT;
  };

  // wrapper class for enum VkDependencyFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDependencyFlagBits.html
  enum class DependencyFlagBits : VkDependencyFlags
  {
    eByRegion                                    = VK_DEPENDENCY_BY_REGION_BIT,
    eDeviceGroup                                 = VK_DEPENDENCY_DEVICE_GROUP_BIT,
    eDeviceGroupKHR                              = VK_DEPENDENCY_DEVICE_GROUP_BIT_KHR,
    eViewLocal                                   = VK_DEPENDENCY_VIEW_LOCAL_BIT,
    eViewLocalKHR                                = VK_DEPENDENCY_VIEW_LOCAL_BIT_KHR,
    eFeedbackLoopEXT                             = VK_DEPENDENCY_FEEDBACK_LOOP_BIT_EXT,
    eQueueFamilyOwnershipTransferUseAllStagesKHR = VK_DEPENDENCY_QUEUE_FAMILY_OWNERSHIP_TRANSFER_USE_ALL_STAGES_BIT_KHR,
    eAsymmetricEventKHR                          = VK_DEPENDENCY_ASYMMETRIC_EVENT_BIT_KHR
  };

  // wrapper using for bitmask VkDependencyFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDependencyFlags.html
  using DependencyFlags = Flags<DependencyFlagBits>;

  template <>
  struct FlagTraits<DependencyFlagBits>
  {
    using WrappedType                                              = VkDependencyFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool            isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DependencyFlags allFlags =
      DependencyFlagBits::eByRegion | DependencyFlagBits::eDeviceGroup | DependencyFlagBits::eViewLocal | DependencyFlagBits::eFeedbackLoopEXT |
      DependencyFlagBits::eQueueFamilyOwnershipTransferUseAllStagesKHR | DependencyFlagBits::eAsymmetricEventKHR;
  };

  // wrapper class for enum VkCommandPoolCreateFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkCommandPoolCreateFlagBits.html
  enum class CommandPoolCreateFlagBits : VkCommandPoolCreateFlags
  {
    eTransient          = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
    eResetCommandBuffer = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    eProtected          = VK_COMMAND_POOL_CREATE_PROTECTED_BIT
  };

  // wrapper using for bitmask VkCommandPoolCreateFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkCommandPoolCreateFlags.html
  using CommandPoolCreateFlags = Flags<CommandPoolCreateFlagBits>;

  template <>
  struct FlagTraits<CommandPoolCreateFlagBits>
  {
    using WrappedType                                                     = VkCommandPoolCreateFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR CommandPoolCreateFlags allFlags =
      CommandPoolCreateFlagBits::eTransient | CommandPoolCreateFlagBits::eResetCommandBuffer | CommandPoolCreateFlagBits::eProtected;
  };

  // wrapper class for enum VkCommandPoolResetFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkCommandPoolResetFlagBits.html
  enum class CommandPoolResetFlagBits : VkCommandPoolResetFlags
  {
    eReleaseResources = VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT
  };

  // wrapper using for bitmask VkCommandPoolResetFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkCommandPoolResetFlags.html
  using CommandPoolResetFlags = Flags<CommandPoolResetFlagBits>;

  template <>
  struct FlagTraits<CommandPoolResetFlagBits>
  {
    using WrappedType                                                    = VkCommandPoolResetFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                  isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR CommandPoolResetFlags allFlags  = CommandPoolResetFlagBits::eReleaseResources;
  };

  // wrapper class for enum VkCommandBufferLevel, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkCommandBufferLevel.html
  enum class CommandBufferLevel
  {
    ePrimary   = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    eSecondary = VK_COMMAND_BUFFER_LEVEL_SECONDARY
  };

  // wrapper class for enum VkCommandBufferResetFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkCommandBufferResetFlagBits.html
  enum class CommandBufferResetFlagBits : VkCommandBufferResetFlags
  {
    eReleaseResources = VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT
  };

  // wrapper using for bitmask VkCommandBufferResetFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkCommandBufferResetFlags.html
  using CommandBufferResetFlags = Flags<CommandBufferResetFlagBits>;

  template <>
  struct FlagTraits<CommandBufferResetFlagBits>
  {
    using WrappedType                                                      = VkCommandBufferResetFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                    isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR CommandBufferResetFlags allFlags  = CommandBufferResetFlagBits::eReleaseResources;
  };

  // wrapper class for enum VkCommandBufferUsageFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkCommandBufferUsageFlagBits.html
  enum class CommandBufferUsageFlagBits : VkCommandBufferUsageFlags
  {
    eOneTimeSubmit      = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    eRenderPassContinue = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
    eSimultaneousUse    = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT
  };

  // wrapper using for bitmask VkCommandBufferUsageFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkCommandBufferUsageFlags.html
  using CommandBufferUsageFlags = Flags<CommandBufferUsageFlagBits>;

  template <>
  struct FlagTraits<CommandBufferUsageFlagBits>
  {
    using WrappedType                                                      = VkCommandBufferUsageFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                    isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR CommandBufferUsageFlags allFlags =
      CommandBufferUsageFlagBits::eOneTimeSubmit | CommandBufferUsageFlagBits::eRenderPassContinue | CommandBufferUsageFlagBits::eSimultaneousUse;
  };

  // wrapper class for enum VkQueryControlFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkQueryControlFlagBits.html
  enum class QueryControlFlagBits : VkQueryControlFlags
  {
    ePrecise = VK_QUERY_CONTROL_PRECISE_BIT
  };

  // wrapper using for bitmask VkQueryControlFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkQueryControlFlags.html
  using QueryControlFlags = Flags<QueryControlFlagBits>;

  template <>
  struct FlagTraits<QueryControlFlagBits>
  {
    using WrappedType                                                = VkQueryControlFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool              isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR QueryControlFlags allFlags  = QueryControlFlagBits::ePrecise;
  };

  // wrapper class for enum VkIndexType, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkIndexType.html
  enum class IndexType
  {
    eUint16   = VK_INDEX_TYPE_UINT16,
    eUint32   = VK_INDEX_TYPE_UINT32,
    eUint8    = VK_INDEX_TYPE_UINT8,
    eUint8EXT = VK_INDEX_TYPE_UINT8_EXT,
    eUint8KHR = VK_INDEX_TYPE_UINT8_KHR,
    eNoneKHR  = VK_INDEX_TYPE_NONE_KHR,
    eNoneNV   = VK_INDEX_TYPE_NONE_NV
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
    static VULKAN_HPP_CONST_OR_CONSTEXPR IndexType value = IndexType::eUint8;
  };

  template <>
  struct CppType<IndexType, IndexType::eUint8>
  {
    using Type = uint8_t;
  };

  // wrapper class for enum VkPipelineCacheHeaderVersion, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineCacheHeaderVersion.html
  enum class PipelineCacheHeaderVersion
  {
    eOne           = VK_PIPELINE_CACHE_HEADER_VERSION_ONE,
    eDataGraphQCOM = VK_PIPELINE_CACHE_HEADER_VERSION_DATA_GRAPH_QCOM
  };

  // wrapper class for enum VkEventCreateFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkEventCreateFlagBits.html
  enum class EventCreateFlagBits : VkEventCreateFlags
  {
    eDeviceOnly    = VK_EVENT_CREATE_DEVICE_ONLY_BIT,
    eDeviceOnlyKHR = VK_EVENT_CREATE_DEVICE_ONLY_BIT_KHR
  };

  // wrapper using for bitmask VkEventCreateFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkEventCreateFlags.html
  using EventCreateFlags = Flags<EventCreateFlagBits>;

  template <>
  struct FlagTraits<EventCreateFlagBits>
  {
    using WrappedType                                               = VkEventCreateFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR EventCreateFlags allFlags  = EventCreateFlagBits::eDeviceOnly;
  };

  enum class BufferViewCreateFlagBits : VkBufferViewCreateFlags
  {
  };

  // wrapper using for bitmask VkBufferViewCreateFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkBufferViewCreateFlags.html
  using BufferViewCreateFlags = Flags<BufferViewCreateFlagBits>;

  template <>
  struct FlagTraits<BufferViewCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                  isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR BufferViewCreateFlags allFlags  = {};
  };

  enum class ShaderModuleCreateFlagBits : VkShaderModuleCreateFlags
  {
  };

  // wrapper using for bitmask VkShaderModuleCreateFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkShaderModuleCreateFlags.html
  using ShaderModuleCreateFlags = Flags<ShaderModuleCreateFlagBits>;

  template <>
  struct FlagTraits<ShaderModuleCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                    isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ShaderModuleCreateFlags allFlags  = {};
  };

  // wrapper class for enum VkPipelineCacheCreateFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineCacheCreateFlagBits.html
  enum class PipelineCacheCreateFlagBits : VkPipelineCacheCreateFlags
  {
    eExternallySynchronized         = VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT,
    eExternallySynchronizedEXT      = VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT_EXT,
    eInternallySynchronizedMergeKHR = VK_PIPELINE_CACHE_CREATE_INTERNALLY_SYNCHRONIZED_MERGE_BIT_KHR
  };

  // wrapper using for bitmask VkPipelineCacheCreateFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineCacheCreateFlags.html
  using PipelineCacheCreateFlags = Flags<PipelineCacheCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineCacheCreateFlagBits>
  {
    using WrappedType                                                       = VkPipelineCacheCreateFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                     isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineCacheCreateFlags allFlags =
      PipelineCacheCreateFlagBits::eExternallySynchronized | PipelineCacheCreateFlagBits::eInternallySynchronizedMergeKHR;
  };

  // wrapper class for enum VkPipelineCreateFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineCreateFlagBits.html
  enum class PipelineCreateFlagBits : VkPipelineCreateFlags
  {
    eDisableOptimization                                                = VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT,
    eAllowDerivatives                                                   = VK_PIPELINE_CREATE_ALLOW_DERIVATIVES_BIT,
    eDerivative                                                         = VK_PIPELINE_CREATE_DERIVATIVE_BIT,
    eDispatchBase                                                       = VK_PIPELINE_CREATE_DISPATCH_BASE_BIT,
    eDispatchBaseKHR                                                    = VK_PIPELINE_CREATE_DISPATCH_BASE_BIT_KHR,
    eViewIndexFromDeviceIndex                                           = VK_PIPELINE_CREATE_VIEW_INDEX_FROM_DEVICE_INDEX_BIT,
    eViewIndexFromDeviceIndexKHR                                        = VK_PIPELINE_CREATE_VIEW_INDEX_FROM_DEVICE_INDEX_BIT_KHR,
    eFailOnPipelineCompileRequired                                      = VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT,
    eFailOnPipelineCompileRequiredEXT                                   = VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_EXT,
    eEarlyReturnOnFailure                                               = VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT,
    eEarlyReturnOnFailureEXT                                            = VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT_EXT,
    eNoProtectedAccess                                                  = VK_PIPELINE_CREATE_NO_PROTECTED_ACCESS_BIT,
    eNoProtectedAccessEXT                                               = VK_PIPELINE_CREATE_NO_PROTECTED_ACCESS_BIT_EXT,
    eProtectedAccessOnly                                                = VK_PIPELINE_CREATE_PROTECTED_ACCESS_ONLY_BIT,
    eProtectedAccessOnlyEXT                                             = VK_PIPELINE_CREATE_PROTECTED_ACCESS_ONLY_BIT_EXT,
    eRayTracingNoNullAnyHitShadersKHR                                   = VK_PIPELINE_CREATE_RAY_TRACING_NO_NULL_ANY_HIT_SHADERS_BIT_KHR,
    eRayTracingNoNullClosestHitShadersKHR                               = VK_PIPELINE_CREATE_RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS_BIT_KHR,
    eRayTracingNoNullMissShadersKHR                                     = VK_PIPELINE_CREATE_RAY_TRACING_NO_NULL_MISS_SHADERS_BIT_KHR,
    eRayTracingNoNullIntersectionShadersKHR                             = VK_PIPELINE_CREATE_RAY_TRACING_NO_NULL_INTERSECTION_SHADERS_BIT_KHR,
    eRayTracingSkipTrianglesKHR                                         = VK_PIPELINE_CREATE_RAY_TRACING_SKIP_TRIANGLES_BIT_KHR,
    eRayTracingSkipAabbsKHR                                             = VK_PIPELINE_CREATE_RAY_TRACING_SKIP_AABBS_BIT_KHR,
    eRayTracingShaderGroupHandleCaptureReplayKHR                        = VK_PIPELINE_CREATE_RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY_BIT_KHR,
    eDeferCompileNV                                                     = VK_PIPELINE_CREATE_DEFER_COMPILE_BIT_NV,
    eRenderingFragmentDensityMapAttachmentEXT                           = VK_PIPELINE_CREATE_RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT_BIT_EXT,
    eVkPipelineRasterizationStateCreateFragmentDensityMapAttachmentEXT  = VK_PIPELINE_RASTERIZATION_STATE_CREATE_FRAGMENT_DENSITY_MAP_ATTACHMENT_BIT_EXT,
    eRenderingFragmentShadingRateAttachmentKHR                          = VK_PIPELINE_CREATE_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
    eVkPipelineRasterizationStateCreateFragmentShadingRateAttachmentKHR = VK_PIPELINE_RASTERIZATION_STATE_CREATE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
    eCaptureStatisticsKHR                                               = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR,
    eCaptureInternalRepresentationsKHR                                  = VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR,
    eIndirectBindableNV                                                 = VK_PIPELINE_CREATE_INDIRECT_BINDABLE_BIT_NV,
    eLibraryKHR                                                         = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR,
    eDescriptorBufferEXT                                                = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT,
    eRetainLinkTimeOptimizationInfoEXT                                  = VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT,
    eLinkTimeOptimizationEXT                                            = VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT,
    eRayTracingAllowMotionNV                                            = VK_PIPELINE_CREATE_RAY_TRACING_ALLOW_MOTION_BIT_NV,
    eColorAttachmentFeedbackLoopEXT                                     = VK_PIPELINE_CREATE_COLOR_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT,
    eDepthStencilAttachmentFeedbackLoopEXT                              = VK_PIPELINE_CREATE_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT,
    eRayTracingOpacityMicromapEXT                                       = VK_PIPELINE_CREATE_RAY_TRACING_OPACITY_MICROMAP_BIT_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eRayTracingDisplacementMicromapNV = VK_PIPELINE_CREATE_RAY_TRACING_DISPLACEMENT_MICROMAP_BIT_NV
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
  };

  // wrapper using for bitmask VkPipelineCreateFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineCreateFlags.html
  using PipelineCreateFlags = Flags<PipelineCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineCreateFlagBits>
  {
    using WrappedType                                                  = VkPipelineCreateFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineCreateFlags allFlags =
      PipelineCreateFlagBits::eDisableOptimization | PipelineCreateFlagBits::eAllowDerivatives | PipelineCreateFlagBits::eDerivative |
      PipelineCreateFlagBits::eDispatchBase | PipelineCreateFlagBits::eViewIndexFromDeviceIndex | PipelineCreateFlagBits::eFailOnPipelineCompileRequired |
      PipelineCreateFlagBits::eEarlyReturnOnFailure | PipelineCreateFlagBits::eNoProtectedAccess | PipelineCreateFlagBits::eProtectedAccessOnly |
      PipelineCreateFlagBits::eRayTracingNoNullAnyHitShadersKHR | PipelineCreateFlagBits::eRayTracingNoNullClosestHitShadersKHR |
      PipelineCreateFlagBits::eRayTracingNoNullMissShadersKHR | PipelineCreateFlagBits::eRayTracingNoNullIntersectionShadersKHR |
      PipelineCreateFlagBits::eRayTracingSkipTrianglesKHR | PipelineCreateFlagBits::eRayTracingSkipAabbsKHR |
      PipelineCreateFlagBits::eRayTracingShaderGroupHandleCaptureReplayKHR | PipelineCreateFlagBits::eDeferCompileNV |
      PipelineCreateFlagBits::eRenderingFragmentDensityMapAttachmentEXT | PipelineCreateFlagBits::eRenderingFragmentShadingRateAttachmentKHR |
      PipelineCreateFlagBits::eCaptureStatisticsKHR | PipelineCreateFlagBits::eCaptureInternalRepresentationsKHR | PipelineCreateFlagBits::eIndirectBindableNV |
      PipelineCreateFlagBits::eLibraryKHR | PipelineCreateFlagBits::eDescriptorBufferEXT | PipelineCreateFlagBits::eRetainLinkTimeOptimizationInfoEXT |
      PipelineCreateFlagBits::eLinkTimeOptimizationEXT | PipelineCreateFlagBits::eRayTracingAllowMotionNV |
      PipelineCreateFlagBits::eColorAttachmentFeedbackLoopEXT | PipelineCreateFlagBits::eDepthStencilAttachmentFeedbackLoopEXT |
      PipelineCreateFlagBits::eRayTracingOpacityMicromapEXT
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      | PipelineCreateFlagBits::eRayTracingDisplacementMicromapNV
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      ;
  };

  // wrapper class for enum VkPipelineShaderStageCreateFlagBits, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineShaderStageCreateFlagBits.html
  enum class PipelineShaderStageCreateFlagBits : VkPipelineShaderStageCreateFlags
  {
    eAllowVaryingSubgroupSize    = VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT,
    eAllowVaryingSubgroupSizeEXT = VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT_EXT,
    eRequireFullSubgroups        = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT,
    eRequireFullSubgroupsEXT     = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT
  };

  // wrapper using for bitmask VkPipelineShaderStageCreateFlags, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineShaderStageCreateFlags.html
  using PipelineShaderStageCreateFlags = Flags<PipelineShaderStageCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineShaderStageCreateFlagBits>
  {
    using WrappedType                                                             = VkPipelineShaderStageCreateFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineShaderStageCreateFlags allFlags =
      PipelineShaderStageCreateFlagBits::eAllowVaryingSubgroupSize | PipelineShaderStageCreateFlagBits::eRequireFullSubgroups;
  };

  // wrapper class for enum VkShaderStageFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkShaderStageFlagBits.html
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
    eRaygenNV               = VK_SHADER_STAGE_RAYGEN_BIT_NV,
    eAnyHitKHR              = VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
    eAnyHitNV               = VK_SHADER_STAGE_ANY_HIT_BIT_NV,
    eClosestHitKHR          = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
    eClosestHitNV           = VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV,
    eMissKHR                = VK_SHADER_STAGE_MISS_BIT_KHR,
    eMissNV                 = VK_SHADER_STAGE_MISS_BIT_NV,
    eIntersectionKHR        = VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
    eIntersectionNV         = VK_SHADER_STAGE_INTERSECTION_BIT_NV,
    eCallableKHR            = VK_SHADER_STAGE_CALLABLE_BIT_KHR,
    eCallableNV             = VK_SHADER_STAGE_CALLABLE_BIT_NV,
    eTaskEXT                = VK_SHADER_STAGE_TASK_BIT_EXT,
    eTaskNV                 = VK_SHADER_STAGE_TASK_BIT_NV,
    eMeshEXT                = VK_SHADER_STAGE_MESH_BIT_EXT,
    eMeshNV                 = VK_SHADER_STAGE_MESH_BIT_NV,
    eSubpassShadingHUAWEI   = VK_SHADER_STAGE_SUBPASS_SHADING_BIT_HUAWEI,
    eClusterCullingHUAWEI   = VK_SHADER_STAGE_CLUSTER_CULLING_BIT_HUAWEI
  };

  // wrapper using for bitmask VkShaderStageFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkShaderStageFlags.html
  using ShaderStageFlags = Flags<ShaderStageFlagBits>;

  template <>
  struct FlagTraits<ShaderStageFlagBits>
  {
    using WrappedType                                               = VkShaderStageFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ShaderStageFlags allFlags =
      ShaderStageFlagBits::eVertex | ShaderStageFlagBits::eTessellationControl | ShaderStageFlagBits::eTessellationEvaluation | ShaderStageFlagBits::eGeometry |
      ShaderStageFlagBits::eFragment | ShaderStageFlagBits::eCompute | ShaderStageFlagBits::eAllGraphics | ShaderStageFlagBits::eAll |
      ShaderStageFlagBits::eRaygenKHR | ShaderStageFlagBits::eAnyHitKHR | ShaderStageFlagBits::eClosestHitKHR | ShaderStageFlagBits::eMissKHR |
      ShaderStageFlagBits::eIntersectionKHR | ShaderStageFlagBits::eCallableKHR | ShaderStageFlagBits::eTaskEXT | ShaderStageFlagBits::eMeshEXT |
      ShaderStageFlagBits::eSubpassShadingHUAWEI | ShaderStageFlagBits::eClusterCullingHUAWEI;
  };

  // wrapper class for enum VkPipelineLayoutCreateFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineLayoutCreateFlagBits.html
  enum class PipelineLayoutCreateFlagBits : VkPipelineLayoutCreateFlags
  {
    eIndependentSetsEXT = VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT
  };

  // wrapper using for bitmask VkPipelineLayoutCreateFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineLayoutCreateFlags.html
  using PipelineLayoutCreateFlags = Flags<PipelineLayoutCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineLayoutCreateFlagBits>
  {
    using WrappedType                                                        = VkPipelineLayoutCreateFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                      isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineLayoutCreateFlags allFlags  = PipelineLayoutCreateFlagBits::eIndependentSetsEXT;
  };

  // wrapper class for enum VkBorderColor, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkBorderColor.html
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

  // wrapper class for enum VkFilter, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkFilter.html
  enum class Filter
  {
    eNearest  = VK_FILTER_NEAREST,
    eLinear   = VK_FILTER_LINEAR,
    eCubicEXT = VK_FILTER_CUBIC_EXT,
    eCubicIMG = VK_FILTER_CUBIC_IMG
  };

  // wrapper class for enum VkSamplerAddressMode, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSamplerAddressMode.html
  enum class SamplerAddressMode
  {
    eRepeat               = VK_SAMPLER_ADDRESS_MODE_REPEAT,
    eMirroredRepeat       = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
    eClampToEdge          = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    eClampToBorder        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
    eMirrorClampToEdge    = VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
    eMirrorClampToEdgeKHR = VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE_KHR
  };

  // wrapper class for enum VkSamplerCreateFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSamplerCreateFlagBits.html
  enum class SamplerCreateFlagBits : VkSamplerCreateFlags
  {
    eSubsampledEXT                     = VK_SAMPLER_CREATE_SUBSAMPLED_BIT_EXT,
    eSubsampledCoarseReconstructionEXT = VK_SAMPLER_CREATE_SUBSAMPLED_COARSE_RECONSTRUCTION_BIT_EXT,
    eDescriptorBufferCaptureReplayEXT  = VK_SAMPLER_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_EXT,
    eNonSeamlessCubeMapEXT             = VK_SAMPLER_CREATE_NON_SEAMLESS_CUBE_MAP_BIT_EXT,
    eImageProcessingQCOM               = VK_SAMPLER_CREATE_IMAGE_PROCESSING_BIT_QCOM
  };

  // wrapper using for bitmask VkSamplerCreateFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSamplerCreateFlags.html
  using SamplerCreateFlags = Flags<SamplerCreateFlagBits>;

  template <>
  struct FlagTraits<SamplerCreateFlagBits>
  {
    using WrappedType                                                 = VkSamplerCreateFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool               isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SamplerCreateFlags allFlags =
      SamplerCreateFlagBits::eSubsampledEXT | SamplerCreateFlagBits::eSubsampledCoarseReconstructionEXT |
      SamplerCreateFlagBits::eDescriptorBufferCaptureReplayEXT | SamplerCreateFlagBits::eNonSeamlessCubeMapEXT | SamplerCreateFlagBits::eImageProcessingQCOM;
  };

  // wrapper class for enum VkSamplerMipmapMode, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSamplerMipmapMode.html
  enum class SamplerMipmapMode
  {
    eNearest = VK_SAMPLER_MIPMAP_MODE_NEAREST,
    eLinear  = VK_SAMPLER_MIPMAP_MODE_LINEAR
  };

  // wrapper class for enum VkDescriptorPoolCreateFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDescriptorPoolCreateFlagBits.html
  enum class DescriptorPoolCreateFlagBits : VkDescriptorPoolCreateFlags
  {
    eFreeDescriptorSet          = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
    eUpdateAfterBind            = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
    eUpdateAfterBindEXT         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT_EXT,
    eHostOnlyEXT                = VK_DESCRIPTOR_POOL_CREATE_HOST_ONLY_BIT_EXT,
    eHostOnlyVALVE              = VK_DESCRIPTOR_POOL_CREATE_HOST_ONLY_BIT_VALVE,
    eAllowOverallocationSetsNV  = VK_DESCRIPTOR_POOL_CREATE_ALLOW_OVERALLOCATION_SETS_BIT_NV,
    eAllowOverallocationPoolsNV = VK_DESCRIPTOR_POOL_CREATE_ALLOW_OVERALLOCATION_POOLS_BIT_NV
  };

  // wrapper using for bitmask VkDescriptorPoolCreateFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDescriptorPoolCreateFlags.html
  using DescriptorPoolCreateFlags = Flags<DescriptorPoolCreateFlagBits>;

  template <>
  struct FlagTraits<DescriptorPoolCreateFlagBits>
  {
    using WrappedType                                                        = VkDescriptorPoolCreateFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                      isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DescriptorPoolCreateFlags allFlags =
      DescriptorPoolCreateFlagBits::eFreeDescriptorSet | DescriptorPoolCreateFlagBits::eUpdateAfterBind | DescriptorPoolCreateFlagBits::eHostOnlyEXT |
      DescriptorPoolCreateFlagBits::eAllowOverallocationSetsNV | DescriptorPoolCreateFlagBits::eAllowOverallocationPoolsNV;
  };

  // wrapper class for enum VkDescriptorSetLayoutCreateFlagBits, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkDescriptorSetLayoutCreateFlagBits.html
  enum class DescriptorSetLayoutCreateFlagBits : VkDescriptorSetLayoutCreateFlags
  {
    eUpdateAfterBindPool          = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
    eUpdateAfterBindPoolEXT       = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT_EXT,
    ePushDescriptor               = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
    ePushDescriptorKHR            = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
    eDescriptorBufferEXT          = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT,
    eEmbeddedImmutableSamplersEXT = VK_DESCRIPTOR_SET_LAYOUT_CREATE_EMBEDDED_IMMUTABLE_SAMPLERS_BIT_EXT,
    eIndirectBindableNV           = VK_DESCRIPTOR_SET_LAYOUT_CREATE_INDIRECT_BINDABLE_BIT_NV,
    eHostOnlyPoolEXT              = VK_DESCRIPTOR_SET_LAYOUT_CREATE_HOST_ONLY_POOL_BIT_EXT,
    eHostOnlyPoolVALVE            = VK_DESCRIPTOR_SET_LAYOUT_CREATE_HOST_ONLY_POOL_BIT_VALVE,
    ePerStageNV                   = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PER_STAGE_BIT_NV
  };

  // wrapper using for bitmask VkDescriptorSetLayoutCreateFlags, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkDescriptorSetLayoutCreateFlags.html
  using DescriptorSetLayoutCreateFlags = Flags<DescriptorSetLayoutCreateFlagBits>;

  template <>
  struct FlagTraits<DescriptorSetLayoutCreateFlagBits>
  {
    using WrappedType                                                             = VkDescriptorSetLayoutCreateFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DescriptorSetLayoutCreateFlags allFlags =
      DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool | DescriptorSetLayoutCreateFlagBits::ePushDescriptor |
      DescriptorSetLayoutCreateFlagBits::eDescriptorBufferEXT | DescriptorSetLayoutCreateFlagBits::eEmbeddedImmutableSamplersEXT |
      DescriptorSetLayoutCreateFlagBits::eIndirectBindableNV | DescriptorSetLayoutCreateFlagBits::eHostOnlyPoolEXT |
      DescriptorSetLayoutCreateFlagBits::ePerStageNV;
  };

  // wrapper class for enum VkDescriptorType, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDescriptorType.html
  enum class DescriptorType
  {
    eSampler                            = VK_DESCRIPTOR_TYPE_SAMPLER,
    eCombinedImageSampler               = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    eSampledImage                       = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
    eStorageImage                       = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
    eUniformTexelBuffer                 = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
    eStorageTexelBuffer                 = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
    eUniformBuffer                      = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    eStorageBuffer                      = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    eUniformBufferDynamic               = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
    eStorageBufferDynamic               = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,
    eInputAttachment                    = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
    eInlineUniformBlock                 = VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK,
    eInlineUniformBlockEXT              = VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT,
    eAccelerationStructureKHR           = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
    eAccelerationStructureNV            = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV,
    eSampleWeightImageQCOM              = VK_DESCRIPTOR_TYPE_SAMPLE_WEIGHT_IMAGE_QCOM,
    eBlockMatchImageQCOM                = VK_DESCRIPTOR_TYPE_BLOCK_MATCH_IMAGE_QCOM,
    eTensorARM                          = VK_DESCRIPTOR_TYPE_TENSOR_ARM,
    eMutableEXT                         = VK_DESCRIPTOR_TYPE_MUTABLE_EXT,
    eMutableVALVE                       = VK_DESCRIPTOR_TYPE_MUTABLE_VALVE,
    ePartitionedAccelerationStructureNV = VK_DESCRIPTOR_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_NV
  };

  enum class DescriptorPoolResetFlagBits : VkDescriptorPoolResetFlags
  {
  };

  // wrapper using for bitmask VkDescriptorPoolResetFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDescriptorPoolResetFlags.html
  using DescriptorPoolResetFlags = Flags<DescriptorPoolResetFlagBits>;

  template <>
  struct FlagTraits<DescriptorPoolResetFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                     isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DescriptorPoolResetFlags allFlags  = {};
  };

  // wrapper class for enum VkQueryPipelineStatisticFlagBits, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkQueryPipelineStatisticFlagBits.html
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

  // wrapper using for bitmask VkQueryPipelineStatisticFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkQueryPipelineStatisticFlags.html
  using QueryPipelineStatisticFlags = Flags<QueryPipelineStatisticFlagBits>;

  template <>
  struct FlagTraits<QueryPipelineStatisticFlagBits>
  {
    using WrappedType                                                          = VkQueryPipelineStatisticFlagBits;
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

  // wrapper class for enum VkPipelineBindPoint, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineBindPoint.html
  enum class PipelineBindPoint
  {
    eGraphics = VK_PIPELINE_BIND_POINT_GRAPHICS,
    eCompute  = VK_PIPELINE_BIND_POINT_COMPUTE,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eExecutionGraphAMDX = VK_PIPELINE_BIND_POINT_EXECUTION_GRAPH_AMDX,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eRayTracingKHR        = VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
    eRayTracingNV         = VK_PIPELINE_BIND_POINT_RAY_TRACING_NV,
    eSubpassShadingHUAWEI = VK_PIPELINE_BIND_POINT_SUBPASS_SHADING_HUAWEI,
    eDataGraphARM         = VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM
  };

  // wrapper class for enum VkBlendFactor, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkBlendFactor.html
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

  // wrapper class for enum VkBlendOp, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkBlendOp.html
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

  // wrapper class for enum VkColorComponentFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkColorComponentFlagBits.html
  enum class ColorComponentFlagBits : VkColorComponentFlags
  {
    eR = VK_COLOR_COMPONENT_R_BIT,
    eG = VK_COLOR_COMPONENT_G_BIT,
    eB = VK_COLOR_COMPONENT_B_BIT,
    eA = VK_COLOR_COMPONENT_A_BIT
  };

  // wrapper using for bitmask VkColorComponentFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkColorComponentFlags.html
  using ColorComponentFlags = Flags<ColorComponentFlagBits>;

  template <>
  struct FlagTraits<ColorComponentFlagBits>
  {
    using WrappedType                                                  = VkColorComponentFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ColorComponentFlags allFlags =
      ColorComponentFlagBits::eR | ColorComponentFlagBits::eG | ColorComponentFlagBits::eB | ColorComponentFlagBits::eA;
  };

  // wrapper class for enum VkCompareOp, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkCompareOp.html
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

  // wrapper class for enum VkCullModeFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkCullModeFlagBits.html
  enum class CullModeFlagBits : VkCullModeFlags
  {
    eNone         = VK_CULL_MODE_NONE,
    eFront        = VK_CULL_MODE_FRONT_BIT,
    eBack         = VK_CULL_MODE_BACK_BIT,
    eFrontAndBack = VK_CULL_MODE_FRONT_AND_BACK
  };

  // wrapper using for bitmask VkCullModeFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkCullModeFlags.html
  using CullModeFlags = Flags<CullModeFlagBits>;

  template <>
  struct FlagTraits<CullModeFlagBits>
  {
    using WrappedType                                            = VkCullModeFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool          isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR CullModeFlags allFlags =
      CullModeFlagBits::eNone | CullModeFlagBits::eFront | CullModeFlagBits::eBack | CullModeFlagBits::eFrontAndBack;
  };

  // wrapper class for enum VkDynamicState, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDynamicState.html
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
    eCullModeEXT                         = VK_DYNAMIC_STATE_CULL_MODE_EXT,
    eFrontFace                           = VK_DYNAMIC_STATE_FRONT_FACE,
    eFrontFaceEXT                        = VK_DYNAMIC_STATE_FRONT_FACE_EXT,
    ePrimitiveTopology                   = VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
    ePrimitiveTopologyEXT                = VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY_EXT,
    eViewportWithCount                   = VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT,
    eViewportWithCountEXT                = VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT_EXT,
    eScissorWithCount                    = VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT,
    eScissorWithCountEXT                 = VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT_EXT,
    eVertexInputBindingStride            = VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE,
    eVertexInputBindingStrideEXT         = VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE_EXT,
    eDepthTestEnable                     = VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
    eDepthTestEnableEXT                  = VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE_EXT,
    eDepthWriteEnable                    = VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
    eDepthWriteEnableEXT                 = VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE_EXT,
    eDepthCompareOp                      = VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
    eDepthCompareOpEXT                   = VK_DYNAMIC_STATE_DEPTH_COMPARE_OP_EXT,
    eDepthBoundsTestEnable               = VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE,
    eDepthBoundsTestEnableEXT            = VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE_EXT,
    eStencilTestEnable                   = VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
    eStencilTestEnableEXT                = VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE_EXT,
    eStencilOp                           = VK_DYNAMIC_STATE_STENCIL_OP,
    eStencilOpEXT                        = VK_DYNAMIC_STATE_STENCIL_OP_EXT,
    eRasterizerDiscardEnable             = VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE,
    eRasterizerDiscardEnableEXT          = VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE_EXT,
    eDepthBiasEnable                     = VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE,
    eDepthBiasEnableEXT                  = VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE_EXT,
    ePrimitiveRestartEnable              = VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE,
    ePrimitiveRestartEnableEXT           = VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE_EXT,
    eLineStipple                         = VK_DYNAMIC_STATE_LINE_STIPPLE,
    eLineStippleEXT                      = VK_DYNAMIC_STATE_LINE_STIPPLE_EXT,
    eLineStippleKHR                      = VK_DYNAMIC_STATE_LINE_STIPPLE_KHR,
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
    eVertexInputEXT                      = VK_DYNAMIC_STATE_VERTEX_INPUT_EXT,
    ePatchControlPointsEXT               = VK_DYNAMIC_STATE_PATCH_CONTROL_POINTS_EXT,
    eLogicOpEXT                          = VK_DYNAMIC_STATE_LOGIC_OP_EXT,
    eColorWriteEnableEXT                 = VK_DYNAMIC_STATE_COLOR_WRITE_ENABLE_EXT,
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
    eTessellationDomainOriginEXT         = VK_DYNAMIC_STATE_TESSELLATION_DOMAIN_ORIGIN_EXT,
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
    eAttachmentFeedbackLoopEnableEXT     = VK_DYNAMIC_STATE_ATTACHMENT_FEEDBACK_LOOP_ENABLE_EXT,
    eDepthClampRangeEXT                  = VK_DYNAMIC_STATE_DEPTH_CLAMP_RANGE_EXT
  };

  // wrapper class for enum VkFrontFace, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkFrontFace.html
  enum class FrontFace
  {
    eCounterClockwise = VK_FRONT_FACE_COUNTER_CLOCKWISE,
    eClockwise        = VK_FRONT_FACE_CLOCKWISE
  };

  // wrapper class for enum VkLogicOp, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkLogicOp.html
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

  // wrapper class for enum VkPolygonMode, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPolygonMode.html
  enum class PolygonMode
  {
    eFill            = VK_POLYGON_MODE_FILL,
    eLine            = VK_POLYGON_MODE_LINE,
    ePoint           = VK_POLYGON_MODE_POINT,
    eFillRectangleNV = VK_POLYGON_MODE_FILL_RECTANGLE_NV
  };

  // wrapper class for enum VkPrimitiveTopology, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPrimitiveTopology.html
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

  // wrapper class for enum VkStencilOp, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkStencilOp.html
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

  // wrapper class for enum VkVertexInputRate, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVertexInputRate.html
  enum class VertexInputRate
  {
    eVertex   = VK_VERTEX_INPUT_RATE_VERTEX,
    eInstance = VK_VERTEX_INPUT_RATE_INSTANCE
  };

  // wrapper class for enum VkPipelineColorBlendStateCreateFlagBits, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineColorBlendStateCreateFlagBits.html
  enum class PipelineColorBlendStateCreateFlagBits : VkPipelineColorBlendStateCreateFlags
  {
    eRasterizationOrderAttachmentAccessEXT = VK_PIPELINE_COLOR_BLEND_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_BIT_EXT,
    eRasterizationOrderAttachmentAccessARM = VK_PIPELINE_COLOR_BLEND_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_BIT_ARM
  };

  // wrapper using for bitmask VkPipelineColorBlendStateCreateFlags, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineColorBlendStateCreateFlags.html
  using PipelineColorBlendStateCreateFlags = Flags<PipelineColorBlendStateCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineColorBlendStateCreateFlagBits>
  {
    using WrappedType                                                                 = VkPipelineColorBlendStateCreateFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                               isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineColorBlendStateCreateFlags allFlags =
      PipelineColorBlendStateCreateFlagBits::eRasterizationOrderAttachmentAccessEXT;
  };

  // wrapper class for enum VkPipelineDepthStencilStateCreateFlagBits, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineDepthStencilStateCreateFlagBits.html
  enum class PipelineDepthStencilStateCreateFlagBits : VkPipelineDepthStencilStateCreateFlags
  {
    eRasterizationOrderAttachmentDepthAccessEXT   = VK_PIPELINE_DEPTH_STENCIL_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_BIT_EXT,
    eRasterizationOrderAttachmentDepthAccessARM   = VK_PIPELINE_DEPTH_STENCIL_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_BIT_ARM,
    eRasterizationOrderAttachmentStencilAccessEXT = VK_PIPELINE_DEPTH_STENCIL_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_BIT_EXT,
    eRasterizationOrderAttachmentStencilAccessARM = VK_PIPELINE_DEPTH_STENCIL_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_BIT_ARM
  };

  // wrapper using for bitmask VkPipelineDepthStencilStateCreateFlags, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineDepthStencilStateCreateFlags.html
  using PipelineDepthStencilStateCreateFlags = Flags<PipelineDepthStencilStateCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineDepthStencilStateCreateFlagBits>
  {
    using WrappedType                                                                   = VkPipelineDepthStencilStateCreateFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineDepthStencilStateCreateFlags allFlags =
      PipelineDepthStencilStateCreateFlagBits::eRasterizationOrderAttachmentDepthAccessEXT |
      PipelineDepthStencilStateCreateFlagBits::eRasterizationOrderAttachmentStencilAccessEXT;
  };

  enum class PipelineDynamicStateCreateFlagBits : VkPipelineDynamicStateCreateFlags
  {
  };

  // wrapper using for bitmask VkPipelineDynamicStateCreateFlags, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineDynamicStateCreateFlags.html
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

  // wrapper using for bitmask VkPipelineInputAssemblyStateCreateFlags, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineInputAssemblyStateCreateFlags.html
  using PipelineInputAssemblyStateCreateFlags = Flags<PipelineInputAssemblyStateCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineInputAssemblyStateCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                  isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineInputAssemblyStateCreateFlags allFlags  = {};
  };

  enum class PipelineMultisampleStateCreateFlagBits : VkPipelineMultisampleStateCreateFlags
  {
  };

  // wrapper using for bitmask VkPipelineMultisampleStateCreateFlags, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineMultisampleStateCreateFlags.html
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

  // wrapper using for bitmask VkPipelineRasterizationStateCreateFlags, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineRasterizationStateCreateFlags.html
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

  // wrapper using for bitmask VkPipelineTessellationStateCreateFlags, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineTessellationStateCreateFlags.html
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

  // wrapper using for bitmask VkPipelineVertexInputStateCreateFlags, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineVertexInputStateCreateFlags.html
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

  // wrapper using for bitmask VkPipelineViewportStateCreateFlags, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineViewportStateCreateFlags.html
  using PipelineViewportStateCreateFlags = Flags<PipelineViewportStateCreateFlagBits>;

  template <>
  struct FlagTraits<PipelineViewportStateCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineViewportStateCreateFlags allFlags  = {};
  };

  // wrapper class for enum VkAttachmentDescriptionFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkAttachmentDescriptionFlagBits.html
  enum class AttachmentDescriptionFlagBits : VkAttachmentDescriptionFlags
  {
    eMayAlias                         = VK_ATTACHMENT_DESCRIPTION_MAY_ALIAS_BIT,
    eResolveSkipTransferFunctionKHR   = VK_ATTACHMENT_DESCRIPTION_RESOLVE_SKIP_TRANSFER_FUNCTION_BIT_KHR,
    eResolveEnableTransferFunctionKHR = VK_ATTACHMENT_DESCRIPTION_RESOLVE_ENABLE_TRANSFER_FUNCTION_BIT_KHR
  };

  // wrapper using for bitmask VkAttachmentDescriptionFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkAttachmentDescriptionFlags.html
  using AttachmentDescriptionFlags = Flags<AttachmentDescriptionFlagBits>;

  template <>
  struct FlagTraits<AttachmentDescriptionFlagBits>
  {
    using WrappedType                                                         = VkAttachmentDescriptionFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR AttachmentDescriptionFlags allFlags  = AttachmentDescriptionFlagBits::eMayAlias |
                                                                               AttachmentDescriptionFlagBits::eResolveSkipTransferFunctionKHR |
                                                                               AttachmentDescriptionFlagBits::eResolveEnableTransferFunctionKHR;
  };

  // wrapper class for enum VkAttachmentLoadOp, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkAttachmentLoadOp.html
  enum class AttachmentLoadOp
  {
    eLoad     = VK_ATTACHMENT_LOAD_OP_LOAD,
    eClear    = VK_ATTACHMENT_LOAD_OP_CLEAR,
    eDontCare = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
    eNone     = VK_ATTACHMENT_LOAD_OP_NONE,
    eNoneEXT  = VK_ATTACHMENT_LOAD_OP_NONE_EXT,
    eNoneKHR  = VK_ATTACHMENT_LOAD_OP_NONE_KHR
  };

  // wrapper class for enum VkAttachmentStoreOp, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkAttachmentStoreOp.html
  enum class AttachmentStoreOp
  {
    eStore    = VK_ATTACHMENT_STORE_OP_STORE,
    eDontCare = VK_ATTACHMENT_STORE_OP_DONT_CARE,
    eNone     = VK_ATTACHMENT_STORE_OP_NONE,
    eNoneKHR  = VK_ATTACHMENT_STORE_OP_NONE_KHR,
    eNoneQCOM = VK_ATTACHMENT_STORE_OP_NONE_QCOM,
    eNoneEXT  = VK_ATTACHMENT_STORE_OP_NONE_EXT
  };

  // wrapper class for enum VkFramebufferCreateFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkFramebufferCreateFlagBits.html
  enum class FramebufferCreateFlagBits : VkFramebufferCreateFlags
  {
    eImageless    = VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT,
    eImagelessKHR = VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT_KHR
  };

  // wrapper using for bitmask VkFramebufferCreateFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkFramebufferCreateFlags.html
  using FramebufferCreateFlags = Flags<FramebufferCreateFlagBits>;

  template <>
  struct FlagTraits<FramebufferCreateFlagBits>
  {
    using WrappedType                                                     = VkFramebufferCreateFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR FramebufferCreateFlags allFlags  = FramebufferCreateFlagBits::eImageless;
  };

  // wrapper class for enum VkRenderPassCreateFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkRenderPassCreateFlagBits.html
  enum class RenderPassCreateFlagBits : VkRenderPassCreateFlags
  {
    eTransformQCOM                = VK_RENDER_PASS_CREATE_TRANSFORM_BIT_QCOM,
    ePerLayerFragmentDensityVALVE = VK_RENDER_PASS_CREATE_PER_LAYER_FRAGMENT_DENSITY_BIT_VALVE
  };

  // wrapper using for bitmask VkRenderPassCreateFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkRenderPassCreateFlags.html
  using RenderPassCreateFlags = Flags<RenderPassCreateFlagBits>;

  template <>
  struct FlagTraits<RenderPassCreateFlagBits>
  {
    using WrappedType                                                    = VkRenderPassCreateFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                  isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR RenderPassCreateFlags allFlags =
      RenderPassCreateFlagBits::eTransformQCOM | RenderPassCreateFlagBits::ePerLayerFragmentDensityVALVE;
  };

  // wrapper class for enum VkSubpassDescriptionFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSubpassDescriptionFlagBits.html
  enum class SubpassDescriptionFlagBits : VkSubpassDescriptionFlags
  {
    ePerViewAttributesNVX                         = VK_SUBPASS_DESCRIPTION_PER_VIEW_ATTRIBUTES_BIT_NVX,
    ePerViewPositionXOnlyNVX                      = VK_SUBPASS_DESCRIPTION_PER_VIEW_POSITION_X_ONLY_BIT_NVX,
    eTileShadingApronQCOM                         = VK_SUBPASS_DESCRIPTION_TILE_SHADING_APRON_BIT_QCOM,
    eRasterizationOrderAttachmentColorAccessEXT   = VK_SUBPASS_DESCRIPTION_RASTERIZATION_ORDER_ATTACHMENT_COLOR_ACCESS_BIT_EXT,
    eRasterizationOrderAttachmentColorAccessARM   = VK_SUBPASS_DESCRIPTION_RASTERIZATION_ORDER_ATTACHMENT_COLOR_ACCESS_BIT_ARM,
    eRasterizationOrderAttachmentDepthAccessEXT   = VK_SUBPASS_DESCRIPTION_RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_BIT_EXT,
    eRasterizationOrderAttachmentDepthAccessARM   = VK_SUBPASS_DESCRIPTION_RASTERIZATION_ORDER_ATTACHMENT_DEPTH_ACCESS_BIT_ARM,
    eRasterizationOrderAttachmentStencilAccessEXT = VK_SUBPASS_DESCRIPTION_RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_BIT_EXT,
    eRasterizationOrderAttachmentStencilAccessARM = VK_SUBPASS_DESCRIPTION_RASTERIZATION_ORDER_ATTACHMENT_STENCIL_ACCESS_BIT_ARM,
    eEnableLegacyDitheringEXT                     = VK_SUBPASS_DESCRIPTION_ENABLE_LEGACY_DITHERING_BIT_EXT,
    eFragmentRegionEXT                            = VK_SUBPASS_DESCRIPTION_FRAGMENT_REGION_BIT_EXT,
    eFragmentRegionQCOM                           = VK_SUBPASS_DESCRIPTION_FRAGMENT_REGION_BIT_QCOM,
    eCustomResolveEXT                             = VK_SUBPASS_DESCRIPTION_CUSTOM_RESOLVE_BIT_EXT,
    eShaderResolveQCOM                            = VK_SUBPASS_DESCRIPTION_SHADER_RESOLVE_BIT_QCOM
  };

  // wrapper using for bitmask VkSubpassDescriptionFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSubpassDescriptionFlags.html
  using SubpassDescriptionFlags = Flags<SubpassDescriptionFlagBits>;

  template <>
  struct FlagTraits<SubpassDescriptionFlagBits>
  {
    using WrappedType                                                      = VkSubpassDescriptionFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                    isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SubpassDescriptionFlags allFlags =
      SubpassDescriptionFlagBits::ePerViewAttributesNVX | SubpassDescriptionFlagBits::ePerViewPositionXOnlyNVX |
      SubpassDescriptionFlagBits::eTileShadingApronQCOM | SubpassDescriptionFlagBits::eRasterizationOrderAttachmentColorAccessEXT |
      SubpassDescriptionFlagBits::eRasterizationOrderAttachmentDepthAccessEXT | SubpassDescriptionFlagBits::eRasterizationOrderAttachmentStencilAccessEXT |
      SubpassDescriptionFlagBits::eEnableLegacyDitheringEXT | SubpassDescriptionFlagBits::eFragmentRegionEXT | SubpassDescriptionFlagBits::eCustomResolveEXT;
  };

  // wrapper class for enum VkStencilFaceFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkStencilFaceFlagBits.html
  enum class StencilFaceFlagBits : VkStencilFaceFlags
  {
    eFront                 = VK_STENCIL_FACE_FRONT_BIT,
    eBack                  = VK_STENCIL_FACE_BACK_BIT,
    eFrontAndBack          = VK_STENCIL_FACE_FRONT_AND_BACK,
    eVkStencilFrontAndBack = VK_STENCIL_FRONT_AND_BACK
  };

  // wrapper using for bitmask VkStencilFaceFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkStencilFaceFlags.html
  using StencilFaceFlags = Flags<StencilFaceFlagBits>;

  template <>
  struct FlagTraits<StencilFaceFlagBits>
  {
    using WrappedType                                               = VkStencilFaceFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR StencilFaceFlags allFlags =
      StencilFaceFlagBits::eFront | StencilFaceFlagBits::eBack | StencilFaceFlagBits::eFrontAndBack;
  };

  // wrapper class for enum VkSubpassContents, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSubpassContents.html
  enum class SubpassContents
  {
    eInline                              = VK_SUBPASS_CONTENTS_INLINE,
    eSecondaryCommandBuffers             = VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS,
    eInlineAndSecondaryCommandBuffersKHR = VK_SUBPASS_CONTENTS_INLINE_AND_SECONDARY_COMMAND_BUFFERS_KHR,
    eInlineAndSecondaryCommandBuffersEXT = VK_SUBPASS_CONTENTS_INLINE_AND_SECONDARY_COMMAND_BUFFERS_EXT
  };

  //=== VK_VERSION_1_1 ===

  // wrapper class for enum VkPeerMemoryFeatureFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPeerMemoryFeatureFlagBits.html
  enum class PeerMemoryFeatureFlagBits : VkPeerMemoryFeatureFlags
  {
    eCopySrc    = VK_PEER_MEMORY_FEATURE_COPY_SRC_BIT,
    eCopyDst    = VK_PEER_MEMORY_FEATURE_COPY_DST_BIT,
    eGenericSrc = VK_PEER_MEMORY_FEATURE_GENERIC_SRC_BIT,
    eGenericDst = VK_PEER_MEMORY_FEATURE_GENERIC_DST_BIT
  };

  using PeerMemoryFeatureFlagBitsKHR = PeerMemoryFeatureFlagBits;

  // wrapper using for bitmask VkPeerMemoryFeatureFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPeerMemoryFeatureFlags.html
  using PeerMemoryFeatureFlags    = Flags<PeerMemoryFeatureFlagBits>;
  using PeerMemoryFeatureFlagsKHR = PeerMemoryFeatureFlags;

  template <>
  struct FlagTraits<PeerMemoryFeatureFlagBits>
  {
    using WrappedType                                                     = VkPeerMemoryFeatureFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PeerMemoryFeatureFlags allFlags  = PeerMemoryFeatureFlagBits::eCopySrc | PeerMemoryFeatureFlagBits::eCopyDst |
                                                                           PeerMemoryFeatureFlagBits::eGenericSrc | PeerMemoryFeatureFlagBits::eGenericDst;
  };

  // wrapper class for enum VkMemoryAllocateFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryAllocateFlagBits.html
  enum class MemoryAllocateFlagBits : VkMemoryAllocateFlags
  {
    eDeviceMask                 = VK_MEMORY_ALLOCATE_DEVICE_MASK_BIT,
    eDeviceAddress              = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    eDeviceAddressCaptureReplay = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT,
    eZeroInitializeEXT          = VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT
  };

  using MemoryAllocateFlagBitsKHR = MemoryAllocateFlagBits;

  // wrapper using for bitmask VkMemoryAllocateFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryAllocateFlags.html
  using MemoryAllocateFlags    = Flags<MemoryAllocateFlagBits>;
  using MemoryAllocateFlagsKHR = MemoryAllocateFlags;

  template <>
  struct FlagTraits<MemoryAllocateFlagBits>
  {
    using WrappedType                                                  = VkMemoryAllocateFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR MemoryAllocateFlags allFlags  = MemoryAllocateFlagBits::eDeviceMask | MemoryAllocateFlagBits::eDeviceAddress |
                                                                        MemoryAllocateFlagBits::eDeviceAddressCaptureReplay |
                                                                        MemoryAllocateFlagBits::eZeroInitializeEXT;
  };

  enum class CommandPoolTrimFlagBits : VkCommandPoolTrimFlags
  {
  };

  // wrapper using for bitmask VkCommandPoolTrimFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkCommandPoolTrimFlags.html
  using CommandPoolTrimFlags    = Flags<CommandPoolTrimFlagBits>;
  using CommandPoolTrimFlagsKHR = CommandPoolTrimFlags;

  template <>
  struct FlagTraits<CommandPoolTrimFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR CommandPoolTrimFlags allFlags  = {};
  };

  // wrapper class for enum VkExternalMemoryHandleTypeFlagBits, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkExternalMemoryHandleTypeFlagBits.html
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
#if defined( VK_USE_PLATFORM_OHOS )
    eOhNativeBufferOHOS = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OH_NATIVE_BUFFER_BIT_OHOS,
#endif /*VK_USE_PLATFORM_OHOS*/
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
    eScreenBufferQNX = VK_EXTERNAL_MEMORY_HANDLE_TYPE_SCREEN_BUFFER_BIT_QNX,
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/
#if defined( VK_USE_PLATFORM_METAL_EXT )
    eMtlbufferEXT  = VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLBUFFER_BIT_EXT,
    eMtltextureEXT = VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLTEXTURE_BIT_EXT,
    eMtlheapEXT    = VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLHEAP_BIT_EXT
#endif /*VK_USE_PLATFORM_METAL_EXT*/
  };

  using ExternalMemoryHandleTypeFlagBitsKHR = ExternalMemoryHandleTypeFlagBits;

  // wrapper using for bitmask VkExternalMemoryHandleTypeFlags, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkExternalMemoryHandleTypeFlags.html
  using ExternalMemoryHandleTypeFlags    = Flags<ExternalMemoryHandleTypeFlagBits>;
  using ExternalMemoryHandleTypeFlagsKHR = ExternalMemoryHandleTypeFlags;

  template <>
  struct FlagTraits<ExternalMemoryHandleTypeFlagBits>
  {
    using WrappedType                                                            = VkExternalMemoryHandleTypeFlagBits;
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
#if defined( VK_USE_PLATFORM_OHOS )
      | ExternalMemoryHandleTypeFlagBits::eOhNativeBufferOHOS
#endif /*VK_USE_PLATFORM_OHOS*/
#if defined( VK_USE_PLATFORM_SCREEN_QNX )
      | ExternalMemoryHandleTypeFlagBits::eScreenBufferQNX
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/
#if defined( VK_USE_PLATFORM_METAL_EXT )
      | ExternalMemoryHandleTypeFlagBits::eMtlbufferEXT | ExternalMemoryHandleTypeFlagBits::eMtltextureEXT | ExternalMemoryHandleTypeFlagBits::eMtlheapEXT
#endif /*VK_USE_PLATFORM_METAL_EXT*/
      ;
  };

  // wrapper class for enum VkExternalMemoryFeatureFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkExternalMemoryFeatureFlagBits.html
  enum class ExternalMemoryFeatureFlagBits : VkExternalMemoryFeatureFlags
  {
    eDedicatedOnly = VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT,
    eExportable    = VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT,
    eImportable    = VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT
  };

  using ExternalMemoryFeatureFlagBitsKHR = ExternalMemoryFeatureFlagBits;

  // wrapper using for bitmask VkExternalMemoryFeatureFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkExternalMemoryFeatureFlags.html
  using ExternalMemoryFeatureFlags    = Flags<ExternalMemoryFeatureFlagBits>;
  using ExternalMemoryFeatureFlagsKHR = ExternalMemoryFeatureFlags;

  template <>
  struct FlagTraits<ExternalMemoryFeatureFlagBits>
  {
    using WrappedType                                                         = VkExternalMemoryFeatureFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ExternalMemoryFeatureFlags allFlags =
      ExternalMemoryFeatureFlagBits::eDedicatedOnly | ExternalMemoryFeatureFlagBits::eExportable | ExternalMemoryFeatureFlagBits::eImportable;
  };

  // wrapper class for enum VkExternalFenceHandleTypeFlagBits, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkExternalFenceHandleTypeFlagBits.html
  enum class ExternalFenceHandleTypeFlagBits : VkExternalFenceHandleTypeFlags
  {
    eOpaqueFd       = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT,
    eOpaqueWin32    = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_WIN32_BIT,
    eOpaqueWin32Kmt = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT,
    eSyncFd         = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT
  };

  using ExternalFenceHandleTypeFlagBitsKHR = ExternalFenceHandleTypeFlagBits;

  // wrapper using for bitmask VkExternalFenceHandleTypeFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkExternalFenceHandleTypeFlags.html
  using ExternalFenceHandleTypeFlags    = Flags<ExternalFenceHandleTypeFlagBits>;
  using ExternalFenceHandleTypeFlagsKHR = ExternalFenceHandleTypeFlags;

  template <>
  struct FlagTraits<ExternalFenceHandleTypeFlagBits>
  {
    using WrappedType                                                           = VkExternalFenceHandleTypeFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                         isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ExternalFenceHandleTypeFlags allFlags =
      ExternalFenceHandleTypeFlagBits::eOpaqueFd | ExternalFenceHandleTypeFlagBits::eOpaqueWin32 | ExternalFenceHandleTypeFlagBits::eOpaqueWin32Kmt |
      ExternalFenceHandleTypeFlagBits::eSyncFd;
  };

  // wrapper class for enum VkExternalFenceFeatureFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkExternalFenceFeatureFlagBits.html
  enum class ExternalFenceFeatureFlagBits : VkExternalFenceFeatureFlags
  {
    eExportable = VK_EXTERNAL_FENCE_FEATURE_EXPORTABLE_BIT,
    eImportable = VK_EXTERNAL_FENCE_FEATURE_IMPORTABLE_BIT
  };

  using ExternalFenceFeatureFlagBitsKHR = ExternalFenceFeatureFlagBits;

  // wrapper using for bitmask VkExternalFenceFeatureFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkExternalFenceFeatureFlags.html
  using ExternalFenceFeatureFlags    = Flags<ExternalFenceFeatureFlagBits>;
  using ExternalFenceFeatureFlagsKHR = ExternalFenceFeatureFlags;

  template <>
  struct FlagTraits<ExternalFenceFeatureFlagBits>
  {
    using WrappedType                                                        = VkExternalFenceFeatureFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                      isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ExternalFenceFeatureFlags allFlags =
      ExternalFenceFeatureFlagBits::eExportable | ExternalFenceFeatureFlagBits::eImportable;
  };

  // wrapper class for enum VkFenceImportFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkFenceImportFlagBits.html
  enum class FenceImportFlagBits : VkFenceImportFlags
  {
    eTemporary = VK_FENCE_IMPORT_TEMPORARY_BIT
  };

  using FenceImportFlagBitsKHR = FenceImportFlagBits;

  // wrapper using for bitmask VkFenceImportFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkFenceImportFlags.html
  using FenceImportFlags    = Flags<FenceImportFlagBits>;
  using FenceImportFlagsKHR = FenceImportFlags;

  template <>
  struct FlagTraits<FenceImportFlagBits>
  {
    using WrappedType                                               = VkFenceImportFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR FenceImportFlags allFlags  = FenceImportFlagBits::eTemporary;
  };

  // wrapper class for enum VkSemaphoreImportFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSemaphoreImportFlagBits.html
  enum class SemaphoreImportFlagBits : VkSemaphoreImportFlags
  {
    eTemporary = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT
  };

  using SemaphoreImportFlagBitsKHR = SemaphoreImportFlagBits;

  // wrapper using for bitmask VkSemaphoreImportFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSemaphoreImportFlags.html
  using SemaphoreImportFlags    = Flags<SemaphoreImportFlagBits>;
  using SemaphoreImportFlagsKHR = SemaphoreImportFlags;

  template <>
  struct FlagTraits<SemaphoreImportFlagBits>
  {
    using WrappedType                                                   = VkSemaphoreImportFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SemaphoreImportFlags allFlags  = SemaphoreImportFlagBits::eTemporary;
  };

  // wrapper class for enum VkExternalSemaphoreHandleTypeFlagBits, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkExternalSemaphoreHandleTypeFlagBits.html
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

  // wrapper using for bitmask VkExternalSemaphoreHandleTypeFlags, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkExternalSemaphoreHandleTypeFlags.html
  using ExternalSemaphoreHandleTypeFlags    = Flags<ExternalSemaphoreHandleTypeFlagBits>;
  using ExternalSemaphoreHandleTypeFlagsKHR = ExternalSemaphoreHandleTypeFlags;

  template <>
  struct FlagTraits<ExternalSemaphoreHandleTypeFlagBits>
  {
    using WrappedType                                                               = VkExternalSemaphoreHandleTypeFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ExternalSemaphoreHandleTypeFlags allFlags =
      ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd | ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32 |
      ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32Kmt | ExternalSemaphoreHandleTypeFlagBits::eD3D12Fence | ExternalSemaphoreHandleTypeFlagBits::eSyncFd
#if defined( VK_USE_PLATFORM_FUCHSIA )
      | ExternalSemaphoreHandleTypeFlagBits::eZirconEventFUCHSIA
#endif /*VK_USE_PLATFORM_FUCHSIA*/
      ;
  };

  // wrapper class for enum VkExternalSemaphoreFeatureFlagBits, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkExternalSemaphoreFeatureFlagBits.html
  enum class ExternalSemaphoreFeatureFlagBits : VkExternalSemaphoreFeatureFlags
  {
    eExportable = VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT,
    eImportable = VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT
  };

  using ExternalSemaphoreFeatureFlagBitsKHR = ExternalSemaphoreFeatureFlagBits;

  // wrapper using for bitmask VkExternalSemaphoreFeatureFlags, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkExternalSemaphoreFeatureFlags.html
  using ExternalSemaphoreFeatureFlags    = Flags<ExternalSemaphoreFeatureFlagBits>;
  using ExternalSemaphoreFeatureFlagsKHR = ExternalSemaphoreFeatureFlags;

  template <>
  struct FlagTraits<ExternalSemaphoreFeatureFlagBits>
  {
    using WrappedType                                                            = VkExternalSemaphoreFeatureFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                          isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ExternalSemaphoreFeatureFlags allFlags =
      ExternalSemaphoreFeatureFlagBits::eExportable | ExternalSemaphoreFeatureFlagBits::eImportable;
  };

  // wrapper class for enum VkSubgroupFeatureFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSubgroupFeatureFlagBits.html
  enum class SubgroupFeatureFlagBits : VkSubgroupFeatureFlags
  {
    eBasic              = VK_SUBGROUP_FEATURE_BASIC_BIT,
    eVote               = VK_SUBGROUP_FEATURE_VOTE_BIT,
    eArithmetic         = VK_SUBGROUP_FEATURE_ARITHMETIC_BIT,
    eBallot             = VK_SUBGROUP_FEATURE_BALLOT_BIT,
    eShuffle            = VK_SUBGROUP_FEATURE_SHUFFLE_BIT,
    eShuffleRelative    = VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT,
    eClustered          = VK_SUBGROUP_FEATURE_CLUSTERED_BIT,
    eQuad               = VK_SUBGROUP_FEATURE_QUAD_BIT,
    eRotate             = VK_SUBGROUP_FEATURE_ROTATE_BIT,
    eRotateKHR          = VK_SUBGROUP_FEATURE_ROTATE_BIT_KHR,
    eRotateClustered    = VK_SUBGROUP_FEATURE_ROTATE_CLUSTERED_BIT,
    eRotateClusteredKHR = VK_SUBGROUP_FEATURE_ROTATE_CLUSTERED_BIT_KHR,
    ePartitionedNV      = VK_SUBGROUP_FEATURE_PARTITIONED_BIT_NV
  };

  // wrapper using for bitmask VkSubgroupFeatureFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSubgroupFeatureFlags.html
  using SubgroupFeatureFlags = Flags<SubgroupFeatureFlagBits>;

  template <>
  struct FlagTraits<SubgroupFeatureFlagBits>
  {
    using WrappedType                                                   = VkSubgroupFeatureFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SubgroupFeatureFlags allFlags =
      SubgroupFeatureFlagBits::eBasic | SubgroupFeatureFlagBits::eVote | SubgroupFeatureFlagBits::eArithmetic | SubgroupFeatureFlagBits::eBallot |
      SubgroupFeatureFlagBits::eShuffle | SubgroupFeatureFlagBits::eShuffleRelative | SubgroupFeatureFlagBits::eClustered | SubgroupFeatureFlagBits::eQuad |
      SubgroupFeatureFlagBits::eRotate | SubgroupFeatureFlagBits::eRotateClustered | SubgroupFeatureFlagBits::ePartitionedNV;
  };

  // wrapper class for enum VkDescriptorUpdateTemplateType, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDescriptorUpdateTemplateType.html
  enum class DescriptorUpdateTemplateType
  {
    eDescriptorSet   = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET,
    ePushDescriptors = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS
  };

  using DescriptorUpdateTemplateTypeKHR = DescriptorUpdateTemplateType;

  enum class DescriptorUpdateTemplateCreateFlagBits : VkDescriptorUpdateTemplateCreateFlags
  {
  };

  // wrapper using for bitmask VkDescriptorUpdateTemplateCreateFlags, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkDescriptorUpdateTemplateCreateFlags.html
  using DescriptorUpdateTemplateCreateFlags    = Flags<DescriptorUpdateTemplateCreateFlagBits>;
  using DescriptorUpdateTemplateCreateFlagsKHR = DescriptorUpdateTemplateCreateFlags;

  template <>
  struct FlagTraits<DescriptorUpdateTemplateCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DescriptorUpdateTemplateCreateFlags allFlags  = {};
  };

  // wrapper class for enum VkSamplerYcbcrModelConversion, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSamplerYcbcrModelConversion.html
  enum class SamplerYcbcrModelConversion
  {
    eRgbIdentity   = VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY,
    eYcbcrIdentity = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_IDENTITY,
    eYcbcr709      = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709,
    eYcbcr601      = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601,
    eYcbcr2020     = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_2020
  };

  using SamplerYcbcrModelConversionKHR = SamplerYcbcrModelConversion;

  // wrapper class for enum VkSamplerYcbcrRange, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSamplerYcbcrRange.html
  enum class SamplerYcbcrRange
  {
    eItuFull   = VK_SAMPLER_YCBCR_RANGE_ITU_FULL,
    eItuNarrow = VK_SAMPLER_YCBCR_RANGE_ITU_NARROW
  };

  using SamplerYcbcrRangeKHR = SamplerYcbcrRange;

  // wrapper class for enum VkChromaLocation, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkChromaLocation.html
  enum class ChromaLocation
  {
    eCositedEven = VK_CHROMA_LOCATION_COSITED_EVEN,
    eMidpoint    = VK_CHROMA_LOCATION_MIDPOINT
  };

  using ChromaLocationKHR = ChromaLocation;

  // wrapper class for enum VkPointClippingBehavior, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPointClippingBehavior.html
  enum class PointClippingBehavior
  {
    eAllClipPlanes      = VK_POINT_CLIPPING_BEHAVIOR_ALL_CLIP_PLANES,
    eUserClipPlanesOnly = VK_POINT_CLIPPING_BEHAVIOR_USER_CLIP_PLANES_ONLY
  };

  using PointClippingBehaviorKHR = PointClippingBehavior;

  // wrapper class for enum VkTessellationDomainOrigin, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkTessellationDomainOrigin.html
  enum class TessellationDomainOrigin
  {
    eUpperLeft = VK_TESSELLATION_DOMAIN_ORIGIN_UPPER_LEFT,
    eLowerLeft = VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT
  };

  using TessellationDomainOriginKHR = TessellationDomainOrigin;

  //=== VK_VERSION_1_2 ===

  // wrapper class for enum VkDriverId, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDriverId.html
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
    eMesaHoneykrisp            = VK_DRIVER_ID_MESA_HONEYKRISP,
    eVulkanScEmulationOnVulkan = VK_DRIVER_ID_VULKAN_SC_EMULATION_ON_VULKAN,
    eMesaKosmickrisp           = VK_DRIVER_ID_MESA_KOSMICKRISP
  };

  using DriverIdKHR = DriverId;

  // wrapper class for enum VkSemaphoreType, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSemaphoreType.html
  enum class SemaphoreType
  {
    eBinary   = VK_SEMAPHORE_TYPE_BINARY,
    eTimeline = VK_SEMAPHORE_TYPE_TIMELINE
  };

  using SemaphoreTypeKHR = SemaphoreType;

  // wrapper class for enum VkSemaphoreWaitFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSemaphoreWaitFlagBits.html
  enum class SemaphoreWaitFlagBits : VkSemaphoreWaitFlags
  {
    eAny = VK_SEMAPHORE_WAIT_ANY_BIT
  };

  using SemaphoreWaitFlagBitsKHR = SemaphoreWaitFlagBits;

  // wrapper using for bitmask VkSemaphoreWaitFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSemaphoreWaitFlags.html
  using SemaphoreWaitFlags    = Flags<SemaphoreWaitFlagBits>;
  using SemaphoreWaitFlagsKHR = SemaphoreWaitFlags;

  template <>
  struct FlagTraits<SemaphoreWaitFlagBits>
  {
    using WrappedType                                                 = VkSemaphoreWaitFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool               isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SemaphoreWaitFlags allFlags  = SemaphoreWaitFlagBits::eAny;
  };

  // wrapper class for enum VkShaderFloatControlsIndependence, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkShaderFloatControlsIndependence.html
  enum class ShaderFloatControlsIndependence
  {
    e32BitOnly = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY,
    eAll       = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
    eNone      = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE
  };

  using ShaderFloatControlsIndependenceKHR = ShaderFloatControlsIndependence;

  // wrapper class for enum VkDescriptorBindingFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDescriptorBindingFlagBits.html
  enum class DescriptorBindingFlagBits : VkDescriptorBindingFlags
  {
    eUpdateAfterBind          = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
    eUpdateUnusedWhilePending = VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT,
    ePartiallyBound           = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
    eVariableDescriptorCount  = VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT
  };

  using DescriptorBindingFlagBitsEXT = DescriptorBindingFlagBits;

  // wrapper using for bitmask VkDescriptorBindingFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDescriptorBindingFlags.html
  using DescriptorBindingFlags    = Flags<DescriptorBindingFlagBits>;
  using DescriptorBindingFlagsEXT = DescriptorBindingFlags;

  template <>
  struct FlagTraits<DescriptorBindingFlagBits>
  {
    using WrappedType                                                     = VkDescriptorBindingFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DescriptorBindingFlags allFlags =
      DescriptorBindingFlagBits::eUpdateAfterBind | DescriptorBindingFlagBits::eUpdateUnusedWhilePending | DescriptorBindingFlagBits::ePartiallyBound |
      DescriptorBindingFlagBits::eVariableDescriptorCount;
  };

  // wrapper class for enum VkSamplerReductionMode, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSamplerReductionMode.html
  enum class SamplerReductionMode
  {
    eWeightedAverage               = VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE,
    eMin                           = VK_SAMPLER_REDUCTION_MODE_MIN,
    eMax                           = VK_SAMPLER_REDUCTION_MODE_MAX,
    eWeightedAverageRangeclampQCOM = VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE_RANGECLAMP_QCOM
  };

  using SamplerReductionModeEXT = SamplerReductionMode;

  // wrapper class for enum VkResolveModeFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkResolveModeFlagBits.html
  enum class ResolveModeFlagBits : VkResolveModeFlags
  {
    eNone       = VK_RESOLVE_MODE_NONE,
    eSampleZero = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT,
    eAverage    = VK_RESOLVE_MODE_AVERAGE_BIT,
    eMin        = VK_RESOLVE_MODE_MIN_BIT,
    eMax        = VK_RESOLVE_MODE_MAX_BIT,
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
    eExternalFormatDownsampleANDROID = VK_RESOLVE_MODE_EXTERNAL_FORMAT_DOWNSAMPLE_BIT_ANDROID,
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
    eCustomEXT = VK_RESOLVE_MODE_CUSTOM_BIT_EXT
  };

  using ResolveModeFlagBitsKHR = ResolveModeFlagBits;

  // wrapper using for bitmask VkResolveModeFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkResolveModeFlags.html
  using ResolveModeFlags    = Flags<ResolveModeFlagBits>;
  using ResolveModeFlagsKHR = ResolveModeFlags;

  template <>
  struct FlagTraits<ResolveModeFlagBits>
  {
    using WrappedType                                               = VkResolveModeFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ResolveModeFlags allFlags  = ResolveModeFlagBits::eNone | ResolveModeFlagBits::eSampleZero |
                                                                     ResolveModeFlagBits::eAverage | ResolveModeFlagBits::eMin | ResolveModeFlagBits::eMax
#if defined( VK_USE_PLATFORM_ANDROID_KHR )
                                                                   | ResolveModeFlagBits::eExternalFormatDownsampleANDROID
#endif /*VK_USE_PLATFORM_ANDROID_KHR*/
                                                                   | ResolveModeFlagBits::eCustomEXT;
  };

  //=== VK_VERSION_1_3 ===

  // wrapper class for enum VkToolPurposeFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkToolPurposeFlagBits.html
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

  // wrapper using for bitmask VkToolPurposeFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkToolPurposeFlags.html
  using ToolPurposeFlags    = Flags<ToolPurposeFlagBits>;
  using ToolPurposeFlagsEXT = ToolPurposeFlags;

  template <>
  struct FlagTraits<ToolPurposeFlagBits>
  {
    using WrappedType                                               = VkToolPurposeFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ToolPurposeFlags allFlags =
      ToolPurposeFlagBits::eValidation | ToolPurposeFlagBits::eProfiling | ToolPurposeFlagBits::eTracing | ToolPurposeFlagBits::eAdditionalFeatures |
      ToolPurposeFlagBits::eModifyingFeatures | ToolPurposeFlagBits::eDebugReportingEXT | ToolPurposeFlagBits::eDebugMarkersEXT;
  };

  enum class PrivateDataSlotCreateFlagBits : VkPrivateDataSlotCreateFlags
  {
  };

  using PrivateDataSlotCreateFlagBitsEXT = PrivateDataSlotCreateFlagBits;

  // wrapper using for bitmask VkPrivateDataSlotCreateFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPrivateDataSlotCreateFlags.html
  using PrivateDataSlotCreateFlags    = Flags<PrivateDataSlotCreateFlagBits>;
  using PrivateDataSlotCreateFlagsEXT = PrivateDataSlotCreateFlags;

  template <>
  struct FlagTraits<PrivateDataSlotCreateFlagBits>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PrivateDataSlotCreateFlags allFlags  = {};
  };

  // wrapper class for enum VkPipelineStageFlagBits2, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineStageFlagBits2.html
  enum class PipelineStageFlagBits2 : VkPipelineStageFlags2
  {
    eNone                             = VK_PIPELINE_STAGE_2_NONE,
    eTopOfPipe                        = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
    eDrawIndirect                     = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
    eVertexInput                      = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT,
    eVertexShader                     = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
    eTessellationControlShader        = VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT,
    eTessellationEvaluationShader     = VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT,
    eGeometryShader                   = VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT,
    eFragmentShader                   = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
    eEarlyFragmentTests               = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
    eLateFragmentTests                = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
    eColorAttachmentOutput            = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
    eComputeShader                    = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    eAllTransfer                      = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
    eTransfer                         = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
    eBottomOfPipe                     = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
    eHost                             = VK_PIPELINE_STAGE_2_HOST_BIT,
    eAllGraphics                      = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
    eAllCommands                      = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
    eCopy                             = VK_PIPELINE_STAGE_2_COPY_BIT,
    eResolve                          = VK_PIPELINE_STAGE_2_RESOLVE_BIT,
    eBlit                             = VK_PIPELINE_STAGE_2_BLIT_BIT,
    eClear                            = VK_PIPELINE_STAGE_2_CLEAR_BIT,
    eIndexInput                       = VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,
    eVertexAttributeInput             = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT,
    ePreRasterizationShaders          = VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT,
    eVideoDecodeKHR                   = VK_PIPELINE_STAGE_2_VIDEO_DECODE_BIT_KHR,
    eVideoEncodeKHR                   = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR,
    eTransformFeedbackEXT             = VK_PIPELINE_STAGE_2_TRANSFORM_FEEDBACK_BIT_EXT,
    eConditionalRenderingEXT          = VK_PIPELINE_STAGE_2_CONDITIONAL_RENDERING_BIT_EXT,
    eCommandPreprocessEXT             = VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT,
    eCommandPreprocessNV              = VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_NV,
    eFragmentShadingRateAttachmentKHR = VK_PIPELINE_STAGE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
    eShadingRateImageNV               = VK_PIPELINE_STAGE_2_SHADING_RATE_IMAGE_BIT_NV,
    eAccelerationStructureBuildKHR    = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
    eAccelerationStructureBuildNV     = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
    eRayTracingShaderKHR              = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
    eRayTracingShaderNV               = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_NV,
    eFragmentDensityProcessEXT        = VK_PIPELINE_STAGE_2_FRAGMENT_DENSITY_PROCESS_BIT_EXT,
    eTaskShaderEXT                    = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT,
    eTaskShaderNV                     = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_NV,
    eMeshShaderEXT                    = VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
    eMeshShaderNV                     = VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_NV,
    eSubpassShaderHUAWEI              = VK_PIPELINE_STAGE_2_SUBPASS_SHADER_BIT_HUAWEI,
    eSubpassShadingHUAWEI             = VK_PIPELINE_STAGE_2_SUBPASS_SHADING_BIT_HUAWEI,
    eInvocationMaskHUAWEI             = VK_PIPELINE_STAGE_2_INVOCATION_MASK_BIT_HUAWEI,
    eAccelerationStructureCopyKHR     = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR,
    eMicromapBuildEXT                 = VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT,
    eClusterCullingShaderHUAWEI       = VK_PIPELINE_STAGE_2_CLUSTER_CULLING_SHADER_BIT_HUAWEI,
    eOpticalFlowNV                    = VK_PIPELINE_STAGE_2_OPTICAL_FLOW_BIT_NV,
    eConvertCooperativeVectorMatrixNV = VK_PIPELINE_STAGE_2_CONVERT_COOPERATIVE_VECTOR_MATRIX_BIT_NV,
    eDataGraphARM                     = VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM,
    eCopyIndirectKHR                  = VK_PIPELINE_STAGE_2_COPY_INDIRECT_BIT_KHR,
    eMemoryDecompressionEXT           = VK_PIPELINE_STAGE_2_MEMORY_DECOMPRESSION_BIT_EXT
  };

  using PipelineStageFlagBits2KHR = PipelineStageFlagBits2;

  // wrapper using for bitmask VkPipelineStageFlags2, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineStageFlags2.html
  using PipelineStageFlags2    = Flags<PipelineStageFlagBits2>;
  using PipelineStageFlags2KHR = PipelineStageFlags2;

  template <>
  struct FlagTraits<PipelineStageFlagBits2>
  {
    using WrappedType                                                  = VkPipelineStageFlagBits2;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineStageFlags2 allFlags =
      PipelineStageFlagBits2::eNone | PipelineStageFlagBits2::eTopOfPipe | PipelineStageFlagBits2::eDrawIndirect | PipelineStageFlagBits2::eVertexInput |
      PipelineStageFlagBits2::eVertexShader | PipelineStageFlagBits2::eTessellationControlShader | PipelineStageFlagBits2::eTessellationEvaluationShader |
      PipelineStageFlagBits2::eGeometryShader | PipelineStageFlagBits2::eFragmentShader | PipelineStageFlagBits2::eEarlyFragmentTests |
      PipelineStageFlagBits2::eLateFragmentTests | PipelineStageFlagBits2::eColorAttachmentOutput | PipelineStageFlagBits2::eComputeShader |
      PipelineStageFlagBits2::eAllTransfer | PipelineStageFlagBits2::eBottomOfPipe | PipelineStageFlagBits2::eHost | PipelineStageFlagBits2::eAllGraphics |
      PipelineStageFlagBits2::eAllCommands | PipelineStageFlagBits2::eCopy | PipelineStageFlagBits2::eResolve | PipelineStageFlagBits2::eBlit |
      PipelineStageFlagBits2::eClear | PipelineStageFlagBits2::eIndexInput | PipelineStageFlagBits2::eVertexAttributeInput |
      PipelineStageFlagBits2::ePreRasterizationShaders | PipelineStageFlagBits2::eVideoDecodeKHR | PipelineStageFlagBits2::eVideoEncodeKHR |
      PipelineStageFlagBits2::eTransformFeedbackEXT | PipelineStageFlagBits2::eConditionalRenderingEXT | PipelineStageFlagBits2::eCommandPreprocessEXT |
      PipelineStageFlagBits2::eFragmentShadingRateAttachmentKHR | PipelineStageFlagBits2::eAccelerationStructureBuildKHR |
      PipelineStageFlagBits2::eRayTracingShaderKHR | PipelineStageFlagBits2::eFragmentDensityProcessEXT | PipelineStageFlagBits2::eTaskShaderEXT |
      PipelineStageFlagBits2::eMeshShaderEXT | PipelineStageFlagBits2::eSubpassShaderHUAWEI | PipelineStageFlagBits2::eInvocationMaskHUAWEI |
      PipelineStageFlagBits2::eAccelerationStructureCopyKHR | PipelineStageFlagBits2::eMicromapBuildEXT | PipelineStageFlagBits2::eClusterCullingShaderHUAWEI |
      PipelineStageFlagBits2::eOpticalFlowNV | PipelineStageFlagBits2::eConvertCooperativeVectorMatrixNV | PipelineStageFlagBits2::eDataGraphARM |
      PipelineStageFlagBits2::eCopyIndirectKHR | PipelineStageFlagBits2::eMemoryDecompressionEXT;
  };

  // wrapper class for enum VkAccessFlagBits2, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccessFlagBits2.html
  enum class AccessFlagBits2 : VkAccessFlags2
  {
    eNone                                 = VK_ACCESS_2_NONE,
    eIndirectCommandRead                  = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
    eIndexRead                            = VK_ACCESS_2_INDEX_READ_BIT,
    eVertexAttributeRead                  = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
    eUniformRead                          = VK_ACCESS_2_UNIFORM_READ_BIT,
    eInputAttachmentRead                  = VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT,
    eShaderRead                           = VK_ACCESS_2_SHADER_READ_BIT,
    eShaderWrite                          = VK_ACCESS_2_SHADER_WRITE_BIT,
    eColorAttachmentRead                  = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
    eColorAttachmentWrite                 = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
    eDepthStencilAttachmentRead           = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
    eDepthStencilAttachmentWrite          = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
    eTransferRead                         = VK_ACCESS_2_TRANSFER_READ_BIT,
    eTransferWrite                        = VK_ACCESS_2_TRANSFER_WRITE_BIT,
    eHostRead                             = VK_ACCESS_2_HOST_READ_BIT,
    eHostWrite                            = VK_ACCESS_2_HOST_WRITE_BIT,
    eMemoryRead                           = VK_ACCESS_2_MEMORY_READ_BIT,
    eMemoryWrite                          = VK_ACCESS_2_MEMORY_WRITE_BIT,
    eShaderSampledRead                    = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
    eShaderStorageRead                    = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
    eShaderStorageWrite                   = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
    eVideoDecodeReadKHR                   = VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR,
    eVideoDecodeWriteKHR                  = VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR,
    eVideoEncodeReadKHR                   = VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR,
    eVideoEncodeWriteKHR                  = VK_ACCESS_2_VIDEO_ENCODE_WRITE_BIT_KHR,
    eShaderTileAttachmentReadQCOM         = VK_ACCESS_2_SHADER_TILE_ATTACHMENT_READ_BIT_QCOM,
    eShaderTileAttachmentWriteQCOM        = VK_ACCESS_2_SHADER_TILE_ATTACHMENT_WRITE_BIT_QCOM,
    eTransformFeedbackWriteEXT            = VK_ACCESS_2_TRANSFORM_FEEDBACK_WRITE_BIT_EXT,
    eTransformFeedbackCounterReadEXT      = VK_ACCESS_2_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT,
    eTransformFeedbackCounterWriteEXT     = VK_ACCESS_2_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT,
    eConditionalRenderingReadEXT          = VK_ACCESS_2_CONDITIONAL_RENDERING_READ_BIT_EXT,
    eCommandPreprocessReadEXT             = VK_ACCESS_2_COMMAND_PREPROCESS_READ_BIT_EXT,
    eCommandPreprocessReadNV              = VK_ACCESS_2_COMMAND_PREPROCESS_READ_BIT_NV,
    eCommandPreprocessWriteEXT            = VK_ACCESS_2_COMMAND_PREPROCESS_WRITE_BIT_EXT,
    eCommandPreprocessWriteNV             = VK_ACCESS_2_COMMAND_PREPROCESS_WRITE_BIT_NV,
    eFragmentShadingRateAttachmentReadKHR = VK_ACCESS_2_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR,
    eShadingRateImageReadNV               = VK_ACCESS_2_SHADING_RATE_IMAGE_READ_BIT_NV,
    eAccelerationStructureReadKHR         = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    eAccelerationStructureReadNV          = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_NV,
    eAccelerationStructureWriteKHR        = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
    eAccelerationStructureWriteNV         = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_NV,
    eFragmentDensityMapReadEXT            = VK_ACCESS_2_FRAGMENT_DENSITY_MAP_READ_BIT_EXT,
    eColorAttachmentReadNoncoherentEXT    = VK_ACCESS_2_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT,
    eDescriptorBufferReadEXT              = VK_ACCESS_2_DESCRIPTOR_BUFFER_READ_BIT_EXT,
    eInvocationMaskReadHUAWEI             = VK_ACCESS_2_INVOCATION_MASK_READ_BIT_HUAWEI,
    eShaderBindingTableReadKHR            = VK_ACCESS_2_SHADER_BINDING_TABLE_READ_BIT_KHR,
    eMicromapReadEXT                      = VK_ACCESS_2_MICROMAP_READ_BIT_EXT,
    eMicromapWriteEXT                     = VK_ACCESS_2_MICROMAP_WRITE_BIT_EXT,
    eOpticalFlowReadNV                    = VK_ACCESS_2_OPTICAL_FLOW_READ_BIT_NV,
    eOpticalFlowWriteNV                   = VK_ACCESS_2_OPTICAL_FLOW_WRITE_BIT_NV,
    eDataGraphReadARM                     = VK_ACCESS_2_DATA_GRAPH_READ_BIT_ARM,
    eDataGraphWriteARM                    = VK_ACCESS_2_DATA_GRAPH_WRITE_BIT_ARM,
    eMemoryDecompressionReadEXT           = VK_ACCESS_2_MEMORY_DECOMPRESSION_READ_BIT_EXT,
    eMemoryDecompressionWriteEXT          = VK_ACCESS_2_MEMORY_DECOMPRESSION_WRITE_BIT_EXT
  };

  using AccessFlagBits2KHR = AccessFlagBits2;

  // wrapper using for bitmask VkAccessFlags2, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccessFlags2.html
  using AccessFlags2    = Flags<AccessFlagBits2>;
  using AccessFlags2KHR = AccessFlags2;

  template <>
  struct FlagTraits<AccessFlagBits2>
  {
    using WrappedType                                           = VkAccessFlagBits2;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool         isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR AccessFlags2 allFlags =
      AccessFlagBits2::eNone | AccessFlagBits2::eIndirectCommandRead | AccessFlagBits2::eIndexRead | AccessFlagBits2::eVertexAttributeRead |
      AccessFlagBits2::eUniformRead | AccessFlagBits2::eInputAttachmentRead | AccessFlagBits2::eShaderRead | AccessFlagBits2::eShaderWrite |
      AccessFlagBits2::eColorAttachmentRead | AccessFlagBits2::eColorAttachmentWrite | AccessFlagBits2::eDepthStencilAttachmentRead |
      AccessFlagBits2::eDepthStencilAttachmentWrite | AccessFlagBits2::eTransferRead | AccessFlagBits2::eTransferWrite | AccessFlagBits2::eHostRead |
      AccessFlagBits2::eHostWrite | AccessFlagBits2::eMemoryRead | AccessFlagBits2::eMemoryWrite | AccessFlagBits2::eShaderSampledRead |
      AccessFlagBits2::eShaderStorageRead | AccessFlagBits2::eShaderStorageWrite | AccessFlagBits2::eVideoDecodeReadKHR |
      AccessFlagBits2::eVideoDecodeWriteKHR | AccessFlagBits2::eVideoEncodeReadKHR | AccessFlagBits2::eVideoEncodeWriteKHR |
      AccessFlagBits2::eShaderTileAttachmentReadQCOM | AccessFlagBits2::eShaderTileAttachmentWriteQCOM | AccessFlagBits2::eTransformFeedbackWriteEXT |
      AccessFlagBits2::eTransformFeedbackCounterReadEXT | AccessFlagBits2::eTransformFeedbackCounterWriteEXT | AccessFlagBits2::eConditionalRenderingReadEXT |
      AccessFlagBits2::eCommandPreprocessReadEXT | AccessFlagBits2::eCommandPreprocessWriteEXT | AccessFlagBits2::eFragmentShadingRateAttachmentReadKHR |
      AccessFlagBits2::eAccelerationStructureReadKHR | AccessFlagBits2::eAccelerationStructureWriteKHR | AccessFlagBits2::eFragmentDensityMapReadEXT |
      AccessFlagBits2::eColorAttachmentReadNoncoherentEXT | AccessFlagBits2::eDescriptorBufferReadEXT | AccessFlagBits2::eInvocationMaskReadHUAWEI |
      AccessFlagBits2::eShaderBindingTableReadKHR | AccessFlagBits2::eMicromapReadEXT | AccessFlagBits2::eMicromapWriteEXT |
      AccessFlagBits2::eOpticalFlowReadNV | AccessFlagBits2::eOpticalFlowWriteNV | AccessFlagBits2::eDataGraphReadARM | AccessFlagBits2::eDataGraphWriteARM |
      AccessFlagBits2::eMemoryDecompressionReadEXT | AccessFlagBits2::eMemoryDecompressionWriteEXT;
  };

  // wrapper class for enum VkSubmitFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSubmitFlagBits.html
  enum class SubmitFlagBits : VkSubmitFlags
  {
    eProtected = VK_SUBMIT_PROTECTED_BIT
  };

  using SubmitFlagBitsKHR = SubmitFlagBits;

  // wrapper using for bitmask VkSubmitFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSubmitFlags.html
  using SubmitFlags    = Flags<SubmitFlagBits>;
  using SubmitFlagsKHR = SubmitFlags;

  template <>
  struct FlagTraits<SubmitFlagBits>
  {
    using WrappedType                                          = VkSubmitFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool        isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SubmitFlags allFlags  = SubmitFlagBits::eProtected;
  };

  // wrapper class for enum VkFormatFeatureFlagBits2, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkFormatFeatureFlagBits2.html
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
    eSampledImageFilterCubic              = VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_CUBIC_BIT,
    eSampledImageFilterCubicEXT           = VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_CUBIC_BIT_EXT,
    eHostImageTransfer                    = VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT,
    eHostImageTransferEXT                 = VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT_EXT,
    eVideoDecodeOutputKHR                 = VK_FORMAT_FEATURE_2_VIDEO_DECODE_OUTPUT_BIT_KHR,
    eVideoDecodeDpbKHR                    = VK_FORMAT_FEATURE_2_VIDEO_DECODE_DPB_BIT_KHR,
    eAccelerationStructureVertexBufferKHR = VK_FORMAT_FEATURE_2_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR,
    eFragmentDensityMapEXT                = VK_FORMAT_FEATURE_2_FRAGMENT_DENSITY_MAP_BIT_EXT,
    eFragmentShadingRateAttachmentKHR     = VK_FORMAT_FEATURE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
    eVideoEncodeInputKHR                  = VK_FORMAT_FEATURE_2_VIDEO_ENCODE_INPUT_BIT_KHR,
    eVideoEncodeDpbKHR                    = VK_FORMAT_FEATURE_2_VIDEO_ENCODE_DPB_BIT_KHR,
    eAccelerationStructureRadiusBufferNV  = VK_FORMAT_FEATURE_2_ACCELERATION_STRUCTURE_RADIUS_BUFFER_BIT_NV,
    eLinearColorAttachmentNV              = VK_FORMAT_FEATURE_2_LINEAR_COLOR_ATTACHMENT_BIT_NV,
    eWeightImageQCOM                      = VK_FORMAT_FEATURE_2_WEIGHT_IMAGE_BIT_QCOM,
    eWeightSampledImageQCOM               = VK_FORMAT_FEATURE_2_WEIGHT_SAMPLED_IMAGE_BIT_QCOM,
    eBlockMatchingQCOM                    = VK_FORMAT_FEATURE_2_BLOCK_MATCHING_BIT_QCOM,
    eBoxFilterSampledQCOM                 = VK_FORMAT_FEATURE_2_BOX_FILTER_SAMPLED_BIT_QCOM,
    eTensorShaderARM                      = VK_FORMAT_FEATURE_2_TENSOR_SHADER_BIT_ARM,
    eTensorImageAliasingARM               = VK_FORMAT_FEATURE_2_TENSOR_IMAGE_ALIASING_BIT_ARM,
    eOpticalFlowImageNV                   = VK_FORMAT_FEATURE_2_OPTICAL_FLOW_IMAGE_BIT_NV,
    eOpticalFlowVectorNV                  = VK_FORMAT_FEATURE_2_OPTICAL_FLOW_VECTOR_BIT_NV,
    eOpticalFlowCostNV                    = VK_FORMAT_FEATURE_2_OPTICAL_FLOW_COST_BIT_NV,
    eTensorDataGraphARM                   = VK_FORMAT_FEATURE_2_TENSOR_DATA_GRAPH_BIT_ARM,
    eCopyImageIndirectDstKHR              = VK_FORMAT_FEATURE_2_COPY_IMAGE_INDIRECT_DST_BIT_KHR,
    eVideoEncodeQuantizationDeltaMapKHR   = VK_FORMAT_FEATURE_2_VIDEO_ENCODE_QUANTIZATION_DELTA_MAP_BIT_KHR,
    eVideoEncodeEmphasisMapKHR            = VK_FORMAT_FEATURE_2_VIDEO_ENCODE_EMPHASIS_MAP_BIT_KHR,
    eDepthCopyOnComputeQueueKHR           = VK_FORMAT_FEATURE_2_DEPTH_COPY_ON_COMPUTE_QUEUE_BIT_KHR,
    eDepthCopyOnTransferQueueKHR          = VK_FORMAT_FEATURE_2_DEPTH_COPY_ON_TRANSFER_QUEUE_BIT_KHR,
    eStencilCopyOnComputeQueueKHR         = VK_FORMAT_FEATURE_2_STENCIL_COPY_ON_COMPUTE_QUEUE_BIT_KHR,
    eStencilCopyOnTransferQueueKHR        = VK_FORMAT_FEATURE_2_STENCIL_COPY_ON_TRANSFER_QUEUE_BIT_KHR
  };

  using FormatFeatureFlagBits2KHR = FormatFeatureFlagBits2;

  // wrapper using for bitmask VkFormatFeatureFlags2, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkFormatFeatureFlags2.html
  using FormatFeatureFlags2    = Flags<FormatFeatureFlagBits2>;
  using FormatFeatureFlags2KHR = FormatFeatureFlags2;

  template <>
  struct FlagTraits<FormatFeatureFlagBits2>
  {
    using WrappedType                                                  = VkFormatFeatureFlagBits2;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR FormatFeatureFlags2 allFlags =
      FormatFeatureFlagBits2::eSampledImage | FormatFeatureFlagBits2::eStorageImage | FormatFeatureFlagBits2::eStorageImageAtomic |
      FormatFeatureFlagBits2::eUniformTexelBuffer | FormatFeatureFlagBits2::eStorageTexelBuffer | FormatFeatureFlagBits2::eStorageTexelBufferAtomic |
      FormatFeatureFlagBits2::eVertexBuffer | FormatFeatureFlagBits2::eColorAttachment | FormatFeatureFlagBits2::eColorAttachmentBlend |
      FormatFeatureFlagBits2::eDepthStencilAttachment | FormatFeatureFlagBits2::eBlitSrc | FormatFeatureFlagBits2::eBlitDst |
      FormatFeatureFlagBits2::eSampledImageFilterLinear | FormatFeatureFlagBits2::eTransferSrc | FormatFeatureFlagBits2::eTransferDst |
      FormatFeatureFlagBits2::eSampledImageFilterMinmax | FormatFeatureFlagBits2::eMidpointChromaSamples |
      FormatFeatureFlagBits2::eSampledImageYcbcrConversionLinearFilter | FormatFeatureFlagBits2::eSampledImageYcbcrConversionSeparateReconstructionFilter |
      FormatFeatureFlagBits2::eSampledImageYcbcrConversionChromaReconstructionExplicit |
      FormatFeatureFlagBits2::eSampledImageYcbcrConversionChromaReconstructionExplicitForceable | FormatFeatureFlagBits2::eDisjoint |
      FormatFeatureFlagBits2::eCositedChromaSamples | FormatFeatureFlagBits2::eStorageReadWithoutFormat | FormatFeatureFlagBits2::eStorageWriteWithoutFormat |
      FormatFeatureFlagBits2::eSampledImageDepthComparison | FormatFeatureFlagBits2::eSampledImageFilterCubic | FormatFeatureFlagBits2::eHostImageTransfer |
      FormatFeatureFlagBits2::eVideoDecodeOutputKHR | FormatFeatureFlagBits2::eVideoDecodeDpbKHR |
      FormatFeatureFlagBits2::eAccelerationStructureVertexBufferKHR | FormatFeatureFlagBits2::eFragmentDensityMapEXT |
      FormatFeatureFlagBits2::eFragmentShadingRateAttachmentKHR | FormatFeatureFlagBits2::eVideoEncodeInputKHR | FormatFeatureFlagBits2::eVideoEncodeDpbKHR |
      FormatFeatureFlagBits2::eAccelerationStructureRadiusBufferNV | FormatFeatureFlagBits2::eLinearColorAttachmentNV |
      FormatFeatureFlagBits2::eWeightImageQCOM | FormatFeatureFlagBits2::eWeightSampledImageQCOM | FormatFeatureFlagBits2::eBlockMatchingQCOM |
      FormatFeatureFlagBits2::eBoxFilterSampledQCOM | FormatFeatureFlagBits2::eTensorShaderARM | FormatFeatureFlagBits2::eTensorImageAliasingARM |
      FormatFeatureFlagBits2::eOpticalFlowImageNV | FormatFeatureFlagBits2::eOpticalFlowVectorNV | FormatFeatureFlagBits2::eOpticalFlowCostNV |
      FormatFeatureFlagBits2::eTensorDataGraphARM | FormatFeatureFlagBits2::eCopyImageIndirectDstKHR |
      FormatFeatureFlagBits2::eVideoEncodeQuantizationDeltaMapKHR | FormatFeatureFlagBits2::eVideoEncodeEmphasisMapKHR |
      FormatFeatureFlagBits2::eDepthCopyOnComputeQueueKHR | FormatFeatureFlagBits2::eDepthCopyOnTransferQueueKHR |
      FormatFeatureFlagBits2::eStencilCopyOnComputeQueueKHR | FormatFeatureFlagBits2::eStencilCopyOnTransferQueueKHR;
  };

  // wrapper class for enum VkPipelineCreationFeedbackFlagBits, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineCreationFeedbackFlagBits.html
  enum class PipelineCreationFeedbackFlagBits : VkPipelineCreationFeedbackFlags
  {
    eValid                       = VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT,
    eApplicationPipelineCacheHit = VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT,
    eBasePipelineAcceleration    = VK_PIPELINE_CREATION_FEEDBACK_BASE_PIPELINE_ACCELERATION_BIT
  };

  using PipelineCreationFeedbackFlagBitsEXT = PipelineCreationFeedbackFlagBits;

  // wrapper using for bitmask VkPipelineCreationFeedbackFlags, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineCreationFeedbackFlags.html
  using PipelineCreationFeedbackFlags    = Flags<PipelineCreationFeedbackFlagBits>;
  using PipelineCreationFeedbackFlagsEXT = PipelineCreationFeedbackFlags;

  template <>
  struct FlagTraits<PipelineCreationFeedbackFlagBits>
  {
    using WrappedType                                                            = VkPipelineCreationFeedbackFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                          isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineCreationFeedbackFlags allFlags  = PipelineCreationFeedbackFlagBits::eValid |
                                                                                  PipelineCreationFeedbackFlagBits::eApplicationPipelineCacheHit |
                                                                                  PipelineCreationFeedbackFlagBits::eBasePipelineAcceleration;
  };

  // wrapper class for enum VkRenderingFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkRenderingFlagBits.html
  enum class RenderingFlagBits : VkRenderingFlags
  {
    eContentsSecondaryCommandBuffers     = VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT,
    eSuspending                          = VK_RENDERING_SUSPENDING_BIT,
    eResuming                            = VK_RENDERING_RESUMING_BIT,
    eEnableLegacyDitheringEXT            = VK_RENDERING_ENABLE_LEGACY_DITHERING_BIT_EXT,
    eContentsInlineKHR                   = VK_RENDERING_CONTENTS_INLINE_BIT_KHR,
    eContentsInlineEXT                   = VK_RENDERING_CONTENTS_INLINE_BIT_EXT,
    ePerLayerFragmentDensityVALVE        = VK_RENDERING_PER_LAYER_FRAGMENT_DENSITY_BIT_VALVE,
    eFragmentRegionEXT                   = VK_RENDERING_FRAGMENT_REGION_BIT_EXT,
    eCustomResolveEXT                    = VK_RENDERING_CUSTOM_RESOLVE_BIT_EXT,
    eLocalReadConcurrentAccessControlKHR = VK_RENDERING_LOCAL_READ_CONCURRENT_ACCESS_CONTROL_BIT_KHR
  };

  using RenderingFlagBitsKHR = RenderingFlagBits;

  // wrapper using for bitmask VkRenderingFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkRenderingFlags.html
  using RenderingFlags    = Flags<RenderingFlagBits>;
  using RenderingFlagsKHR = RenderingFlags;

  template <>
  struct FlagTraits<RenderingFlagBits>
  {
    using WrappedType                                             = VkRenderingFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR RenderingFlags allFlags =
      RenderingFlagBits::eContentsSecondaryCommandBuffers | RenderingFlagBits::eSuspending | RenderingFlagBits::eResuming |
      RenderingFlagBits::eEnableLegacyDitheringEXT | RenderingFlagBits::eContentsInlineKHR | RenderingFlagBits::ePerLayerFragmentDensityVALVE |
      RenderingFlagBits::eFragmentRegionEXT | RenderingFlagBits::eCustomResolveEXT | RenderingFlagBits::eLocalReadConcurrentAccessControlKHR;
  };

  //=== VK_VERSION_1_4 ===

  // wrapper class for enum VkQueueGlobalPriority, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkQueueGlobalPriority.html
  enum class QueueGlobalPriority
  {
    eLow         = VK_QUEUE_GLOBAL_PRIORITY_LOW,
    eLowKHR      = VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR,
    eMedium      = VK_QUEUE_GLOBAL_PRIORITY_MEDIUM,
    eMediumKHR   = VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR,
    eHigh        = VK_QUEUE_GLOBAL_PRIORITY_HIGH,
    eHighKHR     = VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR,
    eRealtime    = VK_QUEUE_GLOBAL_PRIORITY_REALTIME,
    eRealtimeKHR = VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR
  };

  using QueueGlobalPriorityEXT = QueueGlobalPriority;
  using QueueGlobalPriorityKHR = QueueGlobalPriority;

  // wrapper class for enum VkMemoryUnmapFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryUnmapFlagBits.html
  enum class MemoryUnmapFlagBits : VkMemoryUnmapFlags
  {
    eReserveEXT = VK_MEMORY_UNMAP_RESERVE_BIT_EXT
  };

  using MemoryUnmapFlagBitsKHR = MemoryUnmapFlagBits;

  // wrapper using for bitmask VkMemoryUnmapFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryUnmapFlags.html
  using MemoryUnmapFlags    = Flags<MemoryUnmapFlagBits>;
  using MemoryUnmapFlagsKHR = MemoryUnmapFlags;

  template <>
  struct FlagTraits<MemoryUnmapFlagBits>
  {
    using WrappedType                                               = VkMemoryUnmapFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR MemoryUnmapFlags allFlags  = MemoryUnmapFlagBits::eReserveEXT;
  };

  // wrapper class for enum VkBufferUsageFlagBits2, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkBufferUsageFlagBits2.html
  enum class BufferUsageFlagBits2 : VkBufferUsageFlags2
  {
    eTransferSrc         = VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
    eTransferDst         = VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
    eUniformTexelBuffer  = VK_BUFFER_USAGE_2_UNIFORM_TEXEL_BUFFER_BIT,
    eStorageTexelBuffer  = VK_BUFFER_USAGE_2_STORAGE_TEXEL_BUFFER_BIT,
    eUniformBuffer       = VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT,
    eStorageBuffer       = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT,
    eIndexBuffer         = VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT,
    eVertexBuffer        = VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT,
    eIndirectBuffer      = VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT,
    eShaderDeviceAddress = VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eExecutionGraphScratchAMDX = VK_BUFFER_USAGE_2_EXECUTION_GRAPH_SCRATCH_BIT_AMDX,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eConditionalRenderingEXT                    = VK_BUFFER_USAGE_2_CONDITIONAL_RENDERING_BIT_EXT,
    eShaderBindingTableKHR                      = VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR,
    eRayTracingNV                               = VK_BUFFER_USAGE_2_RAY_TRACING_BIT_NV,
    eTransformFeedbackBufferEXT                 = VK_BUFFER_USAGE_2_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT,
    eTransformFeedbackCounterBufferEXT          = VK_BUFFER_USAGE_2_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT,
    eVideoDecodeSrcKHR                          = VK_BUFFER_USAGE_2_VIDEO_DECODE_SRC_BIT_KHR,
    eVideoDecodeDstKHR                          = VK_BUFFER_USAGE_2_VIDEO_DECODE_DST_BIT_KHR,
    eVideoEncodeDstKHR                          = VK_BUFFER_USAGE_2_VIDEO_ENCODE_DST_BIT_KHR,
    eVideoEncodeSrcKHR                          = VK_BUFFER_USAGE_2_VIDEO_ENCODE_SRC_BIT_KHR,
    eAccelerationStructureBuildInputReadOnlyKHR = VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
    eAccelerationStructureStorageKHR            = VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
    eSamplerDescriptorBufferEXT                 = VK_BUFFER_USAGE_2_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT,
    eResourceDescriptorBufferEXT                = VK_BUFFER_USAGE_2_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT,
    ePushDescriptorsDescriptorBufferEXT         = VK_BUFFER_USAGE_2_PUSH_DESCRIPTORS_DESCRIPTOR_BUFFER_BIT_EXT,
    eMicromapBuildInputReadOnlyEXT              = VK_BUFFER_USAGE_2_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT,
    eMicromapStorageEXT                         = VK_BUFFER_USAGE_2_MICROMAP_STORAGE_BIT_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eCompressedDataDgf1AMDX = VK_BUFFER_USAGE_2_COMPRESSED_DATA_DGF1_BIT_AMDX,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eDataGraphForeignDescriptorARM = VK_BUFFER_USAGE_2_DATA_GRAPH_FOREIGN_DESCRIPTOR_BIT_ARM,
    eTileMemoryQCOM                = VK_BUFFER_USAGE_2_TILE_MEMORY_BIT_QCOM,
    eMemoryDecompressionEXT        = VK_BUFFER_USAGE_2_MEMORY_DECOMPRESSION_BIT_EXT,
    ePreprocessBufferEXT           = VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT
  };

  using BufferUsageFlagBits2KHR = BufferUsageFlagBits2;

  // wrapper using for bitmask VkBufferUsageFlags2, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkBufferUsageFlags2.html
  using BufferUsageFlags2    = Flags<BufferUsageFlagBits2>;
  using BufferUsageFlags2KHR = BufferUsageFlags2;

  template <>
  struct FlagTraits<BufferUsageFlagBits2>
  {
    using WrappedType                                                = VkBufferUsageFlagBits2;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool              isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR BufferUsageFlags2 allFlags =
      BufferUsageFlagBits2::eTransferSrc | BufferUsageFlagBits2::eTransferDst | BufferUsageFlagBits2::eUniformTexelBuffer |
      BufferUsageFlagBits2::eStorageTexelBuffer | BufferUsageFlagBits2::eUniformBuffer | BufferUsageFlagBits2::eStorageBuffer |
      BufferUsageFlagBits2::eIndexBuffer | BufferUsageFlagBits2::eVertexBuffer | BufferUsageFlagBits2::eIndirectBuffer |
      BufferUsageFlagBits2::eShaderDeviceAddress
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      | BufferUsageFlagBits2::eExecutionGraphScratchAMDX
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      | BufferUsageFlagBits2::eConditionalRenderingEXT | BufferUsageFlagBits2::eShaderBindingTableKHR | BufferUsageFlagBits2::eTransformFeedbackBufferEXT |
      BufferUsageFlagBits2::eTransformFeedbackCounterBufferEXT | BufferUsageFlagBits2::eVideoDecodeSrcKHR | BufferUsageFlagBits2::eVideoDecodeDstKHR |
      BufferUsageFlagBits2::eVideoEncodeDstKHR | BufferUsageFlagBits2::eVideoEncodeSrcKHR | BufferUsageFlagBits2::eAccelerationStructureBuildInputReadOnlyKHR |
      BufferUsageFlagBits2::eAccelerationStructureStorageKHR | BufferUsageFlagBits2::eSamplerDescriptorBufferEXT |
      BufferUsageFlagBits2::eResourceDescriptorBufferEXT | BufferUsageFlagBits2::ePushDescriptorsDescriptorBufferEXT |
      BufferUsageFlagBits2::eMicromapBuildInputReadOnlyEXT | BufferUsageFlagBits2::eMicromapStorageEXT
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      | BufferUsageFlagBits2::eCompressedDataDgf1AMDX
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      | BufferUsageFlagBits2::eDataGraphForeignDescriptorARM | BufferUsageFlagBits2::eTileMemoryQCOM | BufferUsageFlagBits2::eMemoryDecompressionEXT |
      BufferUsageFlagBits2::ePreprocessBufferEXT;
  };

  // wrapper class for enum VkHostImageCopyFlagBits, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkHostImageCopyFlagBits.html
  enum class HostImageCopyFlagBits : VkHostImageCopyFlags
  {
    eMemcpy = VK_HOST_IMAGE_COPY_MEMCPY_BIT
  };

  using HostImageCopyFlagBitsEXT = HostImageCopyFlagBits;

  // wrapper using for bitmask VkHostImageCopyFlags, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkHostImageCopyFlags.html
  using HostImageCopyFlags    = Flags<HostImageCopyFlagBits>;
  using HostImageCopyFlagsEXT = HostImageCopyFlags;

  template <>
  struct FlagTraits<HostImageCopyFlagBits>
  {
    using WrappedType                                                 = VkHostImageCopyFlagBits;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool               isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR HostImageCopyFlags allFlags  = HostImageCopyFlagBits::eMemcpy;
  };

  // wrapper class for enum VkPipelineCreateFlagBits2, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineCreateFlagBits2.html
  enum class PipelineCreateFlagBits2 : VkPipelineCreateFlags2
  {
    eDisableOptimization           = VK_PIPELINE_CREATE_2_DISABLE_OPTIMIZATION_BIT,
    eAllowDerivatives              = VK_PIPELINE_CREATE_2_ALLOW_DERIVATIVES_BIT,
    eDerivative                    = VK_PIPELINE_CREATE_2_DERIVATIVE_BIT,
    eViewIndexFromDeviceIndex      = VK_PIPELINE_CREATE_2_VIEW_INDEX_FROM_DEVICE_INDEX_BIT,
    eDispatchBase                  = VK_PIPELINE_CREATE_2_DISPATCH_BASE_BIT,
    eFailOnPipelineCompileRequired = VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT,
    eEarlyReturnOnFailure          = VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT,
    eNoProtectedAccess             = VK_PIPELINE_CREATE_2_NO_PROTECTED_ACCESS_BIT,
    eNoProtectedAccessEXT          = VK_PIPELINE_CREATE_2_NO_PROTECTED_ACCESS_BIT_EXT,
    eProtectedAccessOnly           = VK_PIPELINE_CREATE_2_PROTECTED_ACCESS_ONLY_BIT,
    eProtectedAccessOnlyEXT        = VK_PIPELINE_CREATE_2_PROTECTED_ACCESS_ONLY_BIT_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eExecutionGraphAMDX = VK_PIPELINE_CREATE_2_EXECUTION_GRAPH_BIT_AMDX,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eRayTracingAllowSpheresAndLinearSweptSpheresNV = VK_PIPELINE_CREATE_2_RAY_TRACING_ALLOW_SPHERES_AND_LINEAR_SWEPT_SPHERES_BIT_NV,
    eEnableLegacyDitheringEXT                      = VK_PIPELINE_CREATE_2_ENABLE_LEGACY_DITHERING_BIT_EXT,
    eDeferCompileNV                                = VK_PIPELINE_CREATE_2_DEFER_COMPILE_BIT_NV,
    eCaptureStatisticsKHR                          = VK_PIPELINE_CREATE_2_CAPTURE_STATISTICS_BIT_KHR,
    eCaptureInternalRepresentationsKHR             = VK_PIPELINE_CREATE_2_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR,
    eLinkTimeOptimizationEXT                       = VK_PIPELINE_CREATE_2_LINK_TIME_OPTIMIZATION_BIT_EXT,
    eRetainLinkTimeOptimizationInfoEXT             = VK_PIPELINE_CREATE_2_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT,
    eLibraryKHR                                    = VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR,
    eRayTracingSkipTrianglesKHR                    = VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_TRIANGLES_BIT_KHR,
    eRayTracingSkipBuiltInPrimitives               = VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_BUILT_IN_PRIMITIVES_BIT_KHR,
    eRayTracingSkipAabbsKHR                        = VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_AABBS_BIT_KHR,
    eRayTracingNoNullAnyHitShadersKHR              = VK_PIPELINE_CREATE_2_RAY_TRACING_NO_NULL_ANY_HIT_SHADERS_BIT_KHR,
    eRayTracingNoNullClosestHitShadersKHR          = VK_PIPELINE_CREATE_2_RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS_BIT_KHR,
    eRayTracingNoNullMissShadersKHR                = VK_PIPELINE_CREATE_2_RAY_TRACING_NO_NULL_MISS_SHADERS_BIT_KHR,
    eRayTracingNoNullIntersectionShadersKHR        = VK_PIPELINE_CREATE_2_RAY_TRACING_NO_NULL_INTERSECTION_SHADERS_BIT_KHR,
    eRayTracingShaderGroupHandleCaptureReplayKHR   = VK_PIPELINE_CREATE_2_RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY_BIT_KHR,
    eIndirectBindableNV                            = VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_NV,
    eRayTracingAllowMotionNV                       = VK_PIPELINE_CREATE_2_RAY_TRACING_ALLOW_MOTION_BIT_NV,
    eRenderingFragmentShadingRateAttachmentKHR     = VK_PIPELINE_CREATE_2_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
    eRenderingFragmentDensityMapAttachmentEXT      = VK_PIPELINE_CREATE_2_RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT_BIT_EXT,
    eRayTracingOpacityMicromapEXT                  = VK_PIPELINE_CREATE_2_RAY_TRACING_OPACITY_MICROMAP_BIT_EXT,
    eColorAttachmentFeedbackLoopEXT                = VK_PIPELINE_CREATE_2_COLOR_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT,
    eDepthStencilAttachmentFeedbackLoopEXT         = VK_PIPELINE_CREATE_2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT,
    eRayTracingDisplacementMicromapNV              = VK_PIPELINE_CREATE_2_RAY_TRACING_DISPLACEMENT_MICROMAP_BIT_NV,
    eDescriptorBufferEXT                           = VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT,
    eDisallowOpacityMicromapARM                    = VK_PIPELINE_CREATE_2_DISALLOW_OPACITY_MICROMAP_BIT_ARM,
    eCaptureDataKHR                                = VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR,
    eIndirectBindableEXT                           = VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT,
    ePerLayerFragmentDensityVALVE                  = VK_PIPELINE_CREATE_2_PER_LAYER_FRAGMENT_DENSITY_BIT_VALVE,
    e64BitIndexingEXT                              = VK_PIPELINE_CREATE_2_64_BIT_INDEXING_BIT_EXT
  };

  using PipelineCreateFlagBits2KHR = PipelineCreateFlagBits2;

  // wrapper using for bitmask VkPipelineCreateFlags2, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineCreateFlags2.html
  using PipelineCreateFlags2    = Flags<PipelineCreateFlagBits2>;
  using PipelineCreateFlags2KHR = PipelineCreateFlags2;

  template <>
  struct FlagTraits<PipelineCreateFlagBits2>
  {
    using WrappedType                                                   = VkPipelineCreateFlagBits2;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineCreateFlags2 allFlags =
      PipelineCreateFlagBits2::eDisableOptimization | PipelineCreateFlagBits2::eAllowDerivatives | PipelineCreateFlagBits2::eDerivative |
      PipelineCreateFlagBits2::eViewIndexFromDeviceIndex | PipelineCreateFlagBits2::eDispatchBase | PipelineCreateFlagBits2::eFailOnPipelineCompileRequired |
      PipelineCreateFlagBits2::eEarlyReturnOnFailure | PipelineCreateFlagBits2::eNoProtectedAccess | PipelineCreateFlagBits2::eProtectedAccessOnly
#if defined( VK_ENABLE_BETA_EXTENSIONS )
      | PipelineCreateFlagBits2::eExecutionGraphAMDX
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
      | PipelineCreateFlagBits2::eRayTracingAllowSpheresAndLinearSweptSpheresNV | PipelineCreateFlagBits2::eEnableLegacyDitheringEXT |
      PipelineCreateFlagBits2::eDeferCompileNV | PipelineCreateFlagBits2::eCaptureStatisticsKHR | PipelineCreateFlagBits2::eCaptureInternalRepresentationsKHR |
      PipelineCreateFlagBits2::eLinkTimeOptimizationEXT | PipelineCreateFlagBits2::eRetainLinkTimeOptimizationInfoEXT | PipelineCreateFlagBits2::eLibraryKHR |
      PipelineCreateFlagBits2::eRayTracingSkipTrianglesKHR | PipelineCreateFlagBits2::eRayTracingSkipAabbsKHR |
      PipelineCreateFlagBits2::eRayTracingNoNullAnyHitShadersKHR | PipelineCreateFlagBits2::eRayTracingNoNullClosestHitShadersKHR |
      PipelineCreateFlagBits2::eRayTracingNoNullMissShadersKHR | PipelineCreateFlagBits2::eRayTracingNoNullIntersectionShadersKHR |
      PipelineCreateFlagBits2::eRayTracingShaderGroupHandleCaptureReplayKHR | PipelineCreateFlagBits2::eIndirectBindableNV |
      PipelineCreateFlagBits2::eRayTracingAllowMotionNV | PipelineCreateFlagBits2::eRenderingFragmentShadingRateAttachmentKHR |
      PipelineCreateFlagBits2::eRenderingFragmentDensityMapAttachmentEXT | PipelineCreateFlagBits2::eRayTracingOpacityMicromapEXT |
      PipelineCreateFlagBits2::eColorAttachmentFeedbackLoopEXT | PipelineCreateFlagBits2::eDepthStencilAttachmentFeedbackLoopEXT |
      PipelineCreateFlagBits2::eRayTracingDisplacementMicromapNV | PipelineCreateFlagBits2::eDescriptorBufferEXT |
      PipelineCreateFlagBits2::eDisallowOpacityMicromapARM | PipelineCreateFlagBits2::eCaptureDataKHR | PipelineCreateFlagBits2::eIndirectBindableEXT |
      PipelineCreateFlagBits2::ePerLayerFragmentDensityVALVE | PipelineCreateFlagBits2::e64BitIndexingEXT;
  };

  // wrapper class for enum VkPipelineRobustnessBufferBehavior, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineRobustnessBufferBehavior.html
  enum class PipelineRobustnessBufferBehavior
  {
    eDeviceDefault       = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DEVICE_DEFAULT,
    eDisabled            = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED,
    eRobustBufferAccess  = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS,
    eRobustBufferAccess2 = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2
  };

  using PipelineRobustnessBufferBehaviorEXT = PipelineRobustnessBufferBehavior;

  // wrapper class for enum VkPipelineRobustnessImageBehavior, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineRobustnessImageBehavior.html
  enum class PipelineRobustnessImageBehavior
  {
    eDeviceDefault      = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DEVICE_DEFAULT,
    eDisabled           = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED,
    eRobustImageAccess  = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_ROBUST_IMAGE_ACCESS,
    eRobustImageAccess2 = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_ROBUST_IMAGE_ACCESS_2
  };

  using PipelineRobustnessImageBehaviorEXT = PipelineRobustnessImageBehavior;

  // wrapper class for enum VkLineRasterizationMode, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkLineRasterizationMode.html
  enum class LineRasterizationMode
  {
    eDefault              = VK_LINE_RASTERIZATION_MODE_DEFAULT,
    eDefaultKHR           = VK_LINE_RASTERIZATION_MODE_DEFAULT_KHR,
    eRectangular          = VK_LINE_RASTERIZATION_MODE_RECTANGULAR,
    eRectangularKHR       = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_KHR,
    eBresenham            = VK_LINE_RASTERIZATION_MODE_BRESENHAM,
    eBresenhamKHR         = VK_LINE_RASTERIZATION_MODE_BRESENHAM_KHR,
    eRectangularSmooth    = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH,
    eRectangularSmoothKHR = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_KHR
  };

  using LineRasterizationModeEXT = LineRasterizationMode;
  using LineRasterizationModeKHR = LineRasterizationMode;

  //=== VK_KHR_surface ===

  // wrapper class for enum VkSurfaceTransformFlagBitsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSurfaceTransformFlagBitsKHR.html
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

  // wrapper using for bitmask VkSurfaceTransformFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSurfaceTransformFlagsKHR.html
  using SurfaceTransformFlagsKHR = Flags<SurfaceTransformFlagBitsKHR>;

  template <>
  struct FlagTraits<SurfaceTransformFlagBitsKHR>
  {
    using WrappedType                                                       = VkSurfaceTransformFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                     isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SurfaceTransformFlagsKHR allFlags =
      SurfaceTransformFlagBitsKHR::eIdentity | SurfaceTransformFlagBitsKHR::eRotate90 | SurfaceTransformFlagBitsKHR::eRotate180 |
      SurfaceTransformFlagBitsKHR::eRotate270 | SurfaceTransformFlagBitsKHR::eHorizontalMirror | SurfaceTransformFlagBitsKHR::eHorizontalMirrorRotate90 |
      SurfaceTransformFlagBitsKHR::eHorizontalMirrorRotate180 | SurfaceTransformFlagBitsKHR::eHorizontalMirrorRotate270 | SurfaceTransformFlagBitsKHR::eInherit;
  };

  // wrapper class for enum VkPresentModeKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPresentModeKHR.html
  enum class PresentModeKHR
  {
    eImmediate               = VK_PRESENT_MODE_IMMEDIATE_KHR,
    eMailbox                 = VK_PRESENT_MODE_MAILBOX_KHR,
    eFifo                    = VK_PRESENT_MODE_FIFO_KHR,
    eFifoRelaxed             = VK_PRESENT_MODE_FIFO_RELAXED_KHR,
    eSharedDemandRefresh     = VK_PRESENT_MODE_SHARED_DEMAND_REFRESH_KHR,
    eSharedContinuousRefresh = VK_PRESENT_MODE_SHARED_CONTINUOUS_REFRESH_KHR,
    eFifoLatestReady         = VK_PRESENT_MODE_FIFO_LATEST_READY_KHR,
    eFifoLatestReadyEXT      = VK_PRESENT_MODE_FIFO_LATEST_READY_EXT
  };

  // wrapper class for enum VkColorSpaceKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkColorSpaceKHR.html
  enum class ColorSpaceKHR
  {
    eSrgbNonlinear             = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
    eVkColorspaceSrgbNonlinear = VK_COLORSPACE_SRGB_NONLINEAR_KHR,
    eDisplayP3NonlinearEXT     = VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT,
    eExtendedSrgbLinearEXT     = VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT,
    eDisplayP3LinearEXT        = VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT,
    eDciP3LinearEXT            = VK_COLOR_SPACE_DCI_P3_LINEAR_EXT,
    eDciP3NonlinearEXT         = VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT,
    eBt709LinearEXT            = VK_COLOR_SPACE_BT709_LINEAR_EXT,
    eBt709NonlinearEXT         = VK_COLOR_SPACE_BT709_NONLINEAR_EXT,
    eBt2020LinearEXT           = VK_COLOR_SPACE_BT2020_LINEAR_EXT,
    eHdr10St2084EXT            = VK_COLOR_SPACE_HDR10_ST2084_EXT,
    eDolbyvisionEXT VULKAN_HPP_DEPRECATED_17( "eDolbyvisionEXT is deprecated, but no reason was given in the API XML" ) = VK_COLOR_SPACE_DOLBYVISION_EXT,
    eHdr10HlgEXT                                                                                                        = VK_COLOR_SPACE_HDR10_HLG_EXT,
    eAdobergbLinearEXT                                                                                                  = VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT,
    eAdobergbNonlinearEXT                                                                                               = VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT,
    ePassThroughEXT                                                                                                     = VK_COLOR_SPACE_PASS_THROUGH_EXT,
    eExtendedSrgbNonlinearEXT = VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT,
    eDisplayNativeAMD         = VK_COLOR_SPACE_DISPLAY_NATIVE_AMD
  };

  // wrapper class for enum VkCompositeAlphaFlagBitsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkCompositeAlphaFlagBitsKHR.html
  enum class CompositeAlphaFlagBitsKHR : VkCompositeAlphaFlagsKHR
  {
    eOpaque         = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
    ePreMultiplied  = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
    ePostMultiplied = VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
    eInherit        = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR
  };

  // wrapper using for bitmask VkCompositeAlphaFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkCompositeAlphaFlagsKHR.html
  using CompositeAlphaFlagsKHR = Flags<CompositeAlphaFlagBitsKHR>;

  template <>
  struct FlagTraits<CompositeAlphaFlagBitsKHR>
  {
    using WrappedType                                                     = VkCompositeAlphaFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR CompositeAlphaFlagsKHR allFlags  = CompositeAlphaFlagBitsKHR::eOpaque | CompositeAlphaFlagBitsKHR::ePreMultiplied |
                                                                           CompositeAlphaFlagBitsKHR::ePostMultiplied | CompositeAlphaFlagBitsKHR::eInherit;
  };

  //=== VK_KHR_swapchain ===

  // wrapper class for enum VkSwapchainCreateFlagBitsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSwapchainCreateFlagBitsKHR.html
  enum class SwapchainCreateFlagBitsKHR : VkSwapchainCreateFlagsKHR
  {
    eSplitInstanceBindRegions    = VK_SWAPCHAIN_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT_KHR,
    eProtected                   = VK_SWAPCHAIN_CREATE_PROTECTED_BIT_KHR,
    eMutableFormat               = VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR,
    ePresentTimingEXT            = VK_SWAPCHAIN_CREATE_PRESENT_TIMING_BIT_EXT,
    ePresentId2                  = VK_SWAPCHAIN_CREATE_PRESENT_ID_2_BIT_KHR,
    ePresentWait2                = VK_SWAPCHAIN_CREATE_PRESENT_WAIT_2_BIT_KHR,
    eDeferredMemoryAllocation    = VK_SWAPCHAIN_CREATE_DEFERRED_MEMORY_ALLOCATION_BIT_KHR,
    eDeferredMemoryAllocationEXT = VK_SWAPCHAIN_CREATE_DEFERRED_MEMORY_ALLOCATION_BIT_EXT
  };

  // wrapper using for bitmask VkSwapchainCreateFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSwapchainCreateFlagsKHR.html
  using SwapchainCreateFlagsKHR = Flags<SwapchainCreateFlagBitsKHR>;

  template <>
  struct FlagTraits<SwapchainCreateFlagBitsKHR>
  {
    using WrappedType                                                      = VkSwapchainCreateFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                    isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SwapchainCreateFlagsKHR allFlags =
      SwapchainCreateFlagBitsKHR::eSplitInstanceBindRegions | SwapchainCreateFlagBitsKHR::eProtected | SwapchainCreateFlagBitsKHR::eMutableFormat |
      SwapchainCreateFlagBitsKHR::ePresentTimingEXT | SwapchainCreateFlagBitsKHR::ePresentId2 | SwapchainCreateFlagBitsKHR::ePresentWait2 |
      SwapchainCreateFlagBitsKHR::eDeferredMemoryAllocation;
  };

  // wrapper class for enum VkDeviceGroupPresentModeFlagBitsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceGroupPresentModeFlagBitsKHR.html
  enum class DeviceGroupPresentModeFlagBitsKHR : VkDeviceGroupPresentModeFlagsKHR
  {
    eLocal            = VK_DEVICE_GROUP_PRESENT_MODE_LOCAL_BIT_KHR,
    eRemote           = VK_DEVICE_GROUP_PRESENT_MODE_REMOTE_BIT_KHR,
    eSum              = VK_DEVICE_GROUP_PRESENT_MODE_SUM_BIT_KHR,
    eLocalMultiDevice = VK_DEVICE_GROUP_PRESENT_MODE_LOCAL_MULTI_DEVICE_BIT_KHR
  };

  // wrapper using for bitmask VkDeviceGroupPresentModeFlagsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceGroupPresentModeFlagsKHR.html
  using DeviceGroupPresentModeFlagsKHR = Flags<DeviceGroupPresentModeFlagBitsKHR>;

  template <>
  struct FlagTraits<DeviceGroupPresentModeFlagBitsKHR>
  {
    using WrappedType                                                             = VkDeviceGroupPresentModeFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DeviceGroupPresentModeFlagsKHR allFlags =
      DeviceGroupPresentModeFlagBitsKHR::eLocal | DeviceGroupPresentModeFlagBitsKHR::eRemote | DeviceGroupPresentModeFlagBitsKHR::eSum |
      DeviceGroupPresentModeFlagBitsKHR::eLocalMultiDevice;
  };

  //=== VK_KHR_display ===

  // wrapper class for enum VkDisplayPlaneAlphaFlagBitsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDisplayPlaneAlphaFlagBitsKHR.html
  enum class DisplayPlaneAlphaFlagBitsKHR : VkDisplayPlaneAlphaFlagsKHR
  {
    eOpaque                = VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR,
    eGlobal                = VK_DISPLAY_PLANE_ALPHA_GLOBAL_BIT_KHR,
    ePerPixel              = VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_BIT_KHR,
    ePerPixelPremultiplied = VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_PREMULTIPLIED_BIT_KHR
  };

  // wrapper using for bitmask VkDisplayPlaneAlphaFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDisplayPlaneAlphaFlagsKHR.html
  using DisplayPlaneAlphaFlagsKHR = Flags<DisplayPlaneAlphaFlagBitsKHR>;

  template <>
  struct FlagTraits<DisplayPlaneAlphaFlagBitsKHR>
  {
    using WrappedType                                                        = VkDisplayPlaneAlphaFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                      isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DisplayPlaneAlphaFlagsKHR allFlags  = DisplayPlaneAlphaFlagBitsKHR::eOpaque | DisplayPlaneAlphaFlagBitsKHR::eGlobal |
                                                                              DisplayPlaneAlphaFlagBitsKHR::ePerPixel |
                                                                              DisplayPlaneAlphaFlagBitsKHR::ePerPixelPremultiplied;
  };

  enum class DisplayModeCreateFlagBitsKHR : VkDisplayModeCreateFlagsKHR
  {
  };

  // wrapper using for bitmask VkDisplayModeCreateFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDisplayModeCreateFlagsKHR.html
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

  // wrapper using for bitmask VkDisplaySurfaceCreateFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDisplaySurfaceCreateFlagsKHR.html
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

  // wrapper using for bitmask VkXlibSurfaceCreateFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkXlibSurfaceCreateFlagsKHR.html
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

  // wrapper using for bitmask VkXcbSurfaceCreateFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkXcbSurfaceCreateFlagsKHR.html
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

  // wrapper using for bitmask VkWaylandSurfaceCreateFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkWaylandSurfaceCreateFlagsKHR.html
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

  // wrapper using for bitmask VkAndroidSurfaceCreateFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkAndroidSurfaceCreateFlagsKHR.html
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

  // wrapper using for bitmask VkWin32SurfaceCreateFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkWin32SurfaceCreateFlagsKHR.html
  using Win32SurfaceCreateFlagsKHR = Flags<Win32SurfaceCreateFlagBitsKHR>;

  template <>
  struct FlagTraits<Win32SurfaceCreateFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR Win32SurfaceCreateFlagsKHR allFlags  = {};
  };
#endif /*VK_USE_PLATFORM_WIN32_KHR*/

  //=== VK_EXT_debug_report ===

  // wrapper class for enum VkDebugReportFlagBitsEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDebugReportFlagBitsEXT.html
  enum class DebugReportFlagBitsEXT : VkDebugReportFlagsEXT
  {
    eInformation        = VK_DEBUG_REPORT_INFORMATION_BIT_EXT,
    eWarning            = VK_DEBUG_REPORT_WARNING_BIT_EXT,
    ePerformanceWarning = VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT,
    eError              = VK_DEBUG_REPORT_ERROR_BIT_EXT,
    eDebug              = VK_DEBUG_REPORT_DEBUG_BIT_EXT
  };

  // wrapper using for bitmask VkDebugReportFlagsEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDebugReportFlagsEXT.html
  using DebugReportFlagsEXT = Flags<DebugReportFlagBitsEXT>;

  template <>
  struct FlagTraits<DebugReportFlagBitsEXT>
  {
    using WrappedType                                                  = VkDebugReportFlagBitsEXT;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DebugReportFlagsEXT allFlags  = DebugReportFlagBitsEXT::eInformation | DebugReportFlagBitsEXT::eWarning |
                                                                        DebugReportFlagBitsEXT::ePerformanceWarning | DebugReportFlagBitsEXT::eError |
                                                                        DebugReportFlagBitsEXT::eDebug;
  };

  // wrapper class for enum VkDebugReportObjectTypeEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDebugReportObjectTypeEXT.html
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
    eSamplerYcbcrConversionKHR   = VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION_KHR_EXT,
    eDescriptorUpdateTemplate    = VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_EXT,
    eDescriptorUpdateTemplateKHR = VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_KHR_EXT,
    eCuModuleNVX                 = VK_DEBUG_REPORT_OBJECT_TYPE_CU_MODULE_NVX_EXT,
    eCuFunctionNVX               = VK_DEBUG_REPORT_OBJECT_TYPE_CU_FUNCTION_NVX_EXT,
    eAccelerationStructureKHR    = VK_DEBUG_REPORT_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR_EXT,
    eAccelerationStructureNV     = VK_DEBUG_REPORT_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eCudaModuleNV   = VK_DEBUG_REPORT_OBJECT_TYPE_CUDA_MODULE_NV_EXT,
    eCudaFunctionNV = VK_DEBUG_REPORT_OBJECT_TYPE_CUDA_FUNCTION_NV_EXT,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
#if defined( VK_USE_PLATFORM_FUCHSIA )
    eBufferCollectionFUCHSIA = VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_COLLECTION_FUCHSIA_EXT
#endif /*VK_USE_PLATFORM_FUCHSIA*/
  };

  //=== VK_AMD_rasterization_order ===

  // wrapper class for enum VkRasterizationOrderAMD, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkRasterizationOrderAMD.html
  enum class RasterizationOrderAMD
  {
    eStrict  = VK_RASTERIZATION_ORDER_STRICT_AMD,
    eRelaxed = VK_RASTERIZATION_ORDER_RELAXED_AMD
  };

  //=== VK_KHR_video_queue ===

  // wrapper class for enum VkVideoCodecOperationFlagBitsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoCodecOperationFlagBitsKHR.html
  enum class VideoCodecOperationFlagBitsKHR : VkVideoCodecOperationFlagsKHR
  {
    eNone       = VK_VIDEO_CODEC_OPERATION_NONE_KHR,
    eEncodeH264 = VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR,
    eEncodeH265 = VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR,
    eDecodeH264 = VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR,
    eDecodeH265 = VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR,
    eDecodeAv1  = VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR,
    eEncodeAv1  = VK_VIDEO_CODEC_OPERATION_ENCODE_AV1_BIT_KHR,
    eDecodeVp9  = VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR
  };

  // wrapper using for bitmask VkVideoCodecOperationFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoCodecOperationFlagsKHR.html
  using VideoCodecOperationFlagsKHR = Flags<VideoCodecOperationFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoCodecOperationFlagBitsKHR>
  {
    using WrappedType                                                          = VkVideoCodecOperationFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                        isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoCodecOperationFlagsKHR allFlags =
      VideoCodecOperationFlagBitsKHR::eNone | VideoCodecOperationFlagBitsKHR::eEncodeH264 | VideoCodecOperationFlagBitsKHR::eEncodeH265 |
      VideoCodecOperationFlagBitsKHR::eDecodeH264 | VideoCodecOperationFlagBitsKHR::eDecodeH265 | VideoCodecOperationFlagBitsKHR::eDecodeAv1 |
      VideoCodecOperationFlagBitsKHR::eEncodeAv1 | VideoCodecOperationFlagBitsKHR::eDecodeVp9;
  };

  // wrapper class for enum VkVideoChromaSubsamplingFlagBitsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoChromaSubsamplingFlagBitsKHR.html
  enum class VideoChromaSubsamplingFlagBitsKHR : VkVideoChromaSubsamplingFlagsKHR
  {
    eInvalid    = VK_VIDEO_CHROMA_SUBSAMPLING_INVALID_KHR,
    eMonochrome = VK_VIDEO_CHROMA_SUBSAMPLING_MONOCHROME_BIT_KHR,
    e420        = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR,
    e422        = VK_VIDEO_CHROMA_SUBSAMPLING_422_BIT_KHR,
    e444        = VK_VIDEO_CHROMA_SUBSAMPLING_444_BIT_KHR
  };

  // wrapper using for bitmask VkVideoChromaSubsamplingFlagsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoChromaSubsamplingFlagsKHR.html
  using VideoChromaSubsamplingFlagsKHR = Flags<VideoChromaSubsamplingFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoChromaSubsamplingFlagBitsKHR>
  {
    using WrappedType                                                             = VkVideoChromaSubsamplingFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoChromaSubsamplingFlagsKHR allFlags =
      VideoChromaSubsamplingFlagBitsKHR::eInvalid | VideoChromaSubsamplingFlagBitsKHR::eMonochrome | VideoChromaSubsamplingFlagBitsKHR::e420 |
      VideoChromaSubsamplingFlagBitsKHR::e422 | VideoChromaSubsamplingFlagBitsKHR::e444;
  };

  // wrapper class for enum VkVideoComponentBitDepthFlagBitsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoComponentBitDepthFlagBitsKHR.html
  enum class VideoComponentBitDepthFlagBitsKHR : VkVideoComponentBitDepthFlagsKHR
  {
    eInvalid = VK_VIDEO_COMPONENT_BIT_DEPTH_INVALID_KHR,
    e8       = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
    e10      = VK_VIDEO_COMPONENT_BIT_DEPTH_10_BIT_KHR,
    e12      = VK_VIDEO_COMPONENT_BIT_DEPTH_12_BIT_KHR
  };

  // wrapper using for bitmask VkVideoComponentBitDepthFlagsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoComponentBitDepthFlagsKHR.html
  using VideoComponentBitDepthFlagsKHR = Flags<VideoComponentBitDepthFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoComponentBitDepthFlagBitsKHR>
  {
    using WrappedType                                                             = VkVideoComponentBitDepthFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoComponentBitDepthFlagsKHR allFlags =
      VideoComponentBitDepthFlagBitsKHR::eInvalid | VideoComponentBitDepthFlagBitsKHR::e8 | VideoComponentBitDepthFlagBitsKHR::e10 |
      VideoComponentBitDepthFlagBitsKHR::e12;
  };

  // wrapper class for enum VkVideoCapabilityFlagBitsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoCapabilityFlagBitsKHR.html
  enum class VideoCapabilityFlagBitsKHR : VkVideoCapabilityFlagsKHR
  {
    eProtectedContent        = VK_VIDEO_CAPABILITY_PROTECTED_CONTENT_BIT_KHR,
    eSeparateReferenceImages = VK_VIDEO_CAPABILITY_SEPARATE_REFERENCE_IMAGES_BIT_KHR
  };

  // wrapper using for bitmask VkVideoCapabilityFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoCapabilityFlagsKHR.html
  using VideoCapabilityFlagsKHR = Flags<VideoCapabilityFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoCapabilityFlagBitsKHR>
  {
    using WrappedType                                                      = VkVideoCapabilityFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                    isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoCapabilityFlagsKHR allFlags =
      VideoCapabilityFlagBitsKHR::eProtectedContent | VideoCapabilityFlagBitsKHR::eSeparateReferenceImages;
  };

  // wrapper class for enum VkVideoSessionCreateFlagBitsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoSessionCreateFlagBitsKHR.html
  enum class VideoSessionCreateFlagBitsKHR : VkVideoSessionCreateFlagsKHR
  {
    eProtectedContent                  = VK_VIDEO_SESSION_CREATE_PROTECTED_CONTENT_BIT_KHR,
    eAllowEncodeParameterOptimizations = VK_VIDEO_SESSION_CREATE_ALLOW_ENCODE_PARAMETER_OPTIMIZATIONS_BIT_KHR,
    eInlineQueries                     = VK_VIDEO_SESSION_CREATE_INLINE_QUERIES_BIT_KHR,
    eAllowEncodeQuantizationDeltaMap   = VK_VIDEO_SESSION_CREATE_ALLOW_ENCODE_QUANTIZATION_DELTA_MAP_BIT_KHR,
    eAllowEncodeEmphasisMap            = VK_VIDEO_SESSION_CREATE_ALLOW_ENCODE_EMPHASIS_MAP_BIT_KHR,
    eInlineSessionParameters           = VK_VIDEO_SESSION_CREATE_INLINE_SESSION_PARAMETERS_BIT_KHR
  };

  // wrapper using for bitmask VkVideoSessionCreateFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoSessionCreateFlagsKHR.html
  using VideoSessionCreateFlagsKHR = Flags<VideoSessionCreateFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoSessionCreateFlagBitsKHR>
  {
    using WrappedType                                                         = VkVideoSessionCreateFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoSessionCreateFlagsKHR allFlags =
      VideoSessionCreateFlagBitsKHR::eProtectedContent | VideoSessionCreateFlagBitsKHR::eAllowEncodeParameterOptimizations |
      VideoSessionCreateFlagBitsKHR::eInlineQueries | VideoSessionCreateFlagBitsKHR::eAllowEncodeQuantizationDeltaMap |
      VideoSessionCreateFlagBitsKHR::eAllowEncodeEmphasisMap | VideoSessionCreateFlagBitsKHR::eInlineSessionParameters;
  };

  // wrapper class for enum VkVideoCodingControlFlagBitsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoCodingControlFlagBitsKHR.html
  enum class VideoCodingControlFlagBitsKHR : VkVideoCodingControlFlagsKHR
  {
    eReset              = VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR,
    eEncodeRateControl  = VK_VIDEO_CODING_CONTROL_ENCODE_RATE_CONTROL_BIT_KHR,
    eEncodeQualityLevel = VK_VIDEO_CODING_CONTROL_ENCODE_QUALITY_LEVEL_BIT_KHR
  };

  // wrapper using for bitmask VkVideoCodingControlFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoCodingControlFlagsKHR.html
  using VideoCodingControlFlagsKHR = Flags<VideoCodingControlFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoCodingControlFlagBitsKHR>
  {
    using WrappedType                                                         = VkVideoCodingControlFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoCodingControlFlagsKHR allFlags =
      VideoCodingControlFlagBitsKHR::eReset | VideoCodingControlFlagBitsKHR::eEncodeRateControl | VideoCodingControlFlagBitsKHR::eEncodeQualityLevel;
  };

  // wrapper class for enum VkQueryResultStatusKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkQueryResultStatusKHR.html
  enum class QueryResultStatusKHR
  {
    eError                            = VK_QUERY_RESULT_STATUS_ERROR_KHR,
    eNotReady                         = VK_QUERY_RESULT_STATUS_NOT_READY_KHR,
    eComplete                         = VK_QUERY_RESULT_STATUS_COMPLETE_KHR,
    eInsufficientBitstreamBufferRange = VK_QUERY_RESULT_STATUS_INSUFFICIENT_BITSTREAM_BUFFER_RANGE_KHR
  };

  // wrapper class for enum VkVideoSessionParametersCreateFlagBitsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoSessionParametersCreateFlagBitsKHR.html
  enum class VideoSessionParametersCreateFlagBitsKHR : VkVideoSessionParametersCreateFlagsKHR
  {
    eQuantizationMapCompatible = VK_VIDEO_SESSION_PARAMETERS_CREATE_QUANTIZATION_MAP_COMPATIBLE_BIT_KHR
  };

  // wrapper using for bitmask VkVideoSessionParametersCreateFlagsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoSessionParametersCreateFlagsKHR.html
  using VideoSessionParametersCreateFlagsKHR = Flags<VideoSessionParametersCreateFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoSessionParametersCreateFlagBitsKHR>
  {
    using WrappedType                                                                   = VkVideoSessionParametersCreateFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoSessionParametersCreateFlagsKHR allFlags  = VideoSessionParametersCreateFlagBitsKHR::eQuantizationMapCompatible;
  };

  enum class VideoBeginCodingFlagBitsKHR : VkVideoBeginCodingFlagsKHR
  {
  };

  // wrapper using for bitmask VkVideoBeginCodingFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoBeginCodingFlagsKHR.html
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

  // wrapper using for bitmask VkVideoEndCodingFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEndCodingFlagsKHR.html
  using VideoEndCodingFlagsKHR = Flags<VideoEndCodingFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEndCodingFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEndCodingFlagsKHR allFlags  = {};
  };

  //=== VK_KHR_video_decode_queue ===

  // wrapper class for enum VkVideoDecodeCapabilityFlagBitsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoDecodeCapabilityFlagBitsKHR.html
  enum class VideoDecodeCapabilityFlagBitsKHR : VkVideoDecodeCapabilityFlagsKHR
  {
    eDpbAndOutputCoincide = VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR,
    eDpbAndOutputDistinct = VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_DISTINCT_BIT_KHR
  };

  // wrapper using for bitmask VkVideoDecodeCapabilityFlagsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoDecodeCapabilityFlagsKHR.html
  using VideoDecodeCapabilityFlagsKHR = Flags<VideoDecodeCapabilityFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoDecodeCapabilityFlagBitsKHR>
  {
    using WrappedType                                                            = VkVideoDecodeCapabilityFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                          isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoDecodeCapabilityFlagsKHR allFlags =
      VideoDecodeCapabilityFlagBitsKHR::eDpbAndOutputCoincide | VideoDecodeCapabilityFlagBitsKHR::eDpbAndOutputDistinct;
  };

  // wrapper class for enum VkVideoDecodeUsageFlagBitsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoDecodeUsageFlagBitsKHR.html
  enum class VideoDecodeUsageFlagBitsKHR : VkVideoDecodeUsageFlagsKHR
  {
    eDefault     = VK_VIDEO_DECODE_USAGE_DEFAULT_KHR,
    eTranscoding = VK_VIDEO_DECODE_USAGE_TRANSCODING_BIT_KHR,
    eOffline     = VK_VIDEO_DECODE_USAGE_OFFLINE_BIT_KHR,
    eStreaming   = VK_VIDEO_DECODE_USAGE_STREAMING_BIT_KHR
  };

  // wrapper using for bitmask VkVideoDecodeUsageFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoDecodeUsageFlagsKHR.html
  using VideoDecodeUsageFlagsKHR = Flags<VideoDecodeUsageFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoDecodeUsageFlagBitsKHR>
  {
    using WrappedType                                                       = VkVideoDecodeUsageFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                     isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoDecodeUsageFlagsKHR allFlags = VideoDecodeUsageFlagBitsKHR::eDefault | VideoDecodeUsageFlagBitsKHR::eTranscoding |
                                                                             VideoDecodeUsageFlagBitsKHR::eOffline | VideoDecodeUsageFlagBitsKHR::eStreaming;
  };

  enum class VideoDecodeFlagBitsKHR : VkVideoDecodeFlagsKHR
  {
  };

  // wrapper using for bitmask VkVideoDecodeFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoDecodeFlagsKHR.html
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

  // wrapper using for bitmask VkPipelineRasterizationStateStreamCreateFlagsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineRasterizationStateStreamCreateFlagsEXT.html
  using PipelineRasterizationStateStreamCreateFlagsEXT = Flags<PipelineRasterizationStateStreamCreateFlagBitsEXT>;

  template <>
  struct FlagTraits<PipelineRasterizationStateStreamCreateFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineRasterizationStateStreamCreateFlagsEXT allFlags  = {};
  };

  //=== VK_KHR_video_encode_h264 ===

  // wrapper class for enum VkVideoEncodeH264CapabilityFlagBitsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeH264CapabilityFlagBitsKHR.html
  enum class VideoEncodeH264CapabilityFlagBitsKHR : VkVideoEncodeH264CapabilityFlagsKHR
  {
    eHrdCompliance                  = VK_VIDEO_ENCODE_H264_CAPABILITY_HRD_COMPLIANCE_BIT_KHR,
    ePredictionWeightTableGenerated = VK_VIDEO_ENCODE_H264_CAPABILITY_PREDICTION_WEIGHT_TABLE_GENERATED_BIT_KHR,
    eRowUnalignedSlice              = VK_VIDEO_ENCODE_H264_CAPABILITY_ROW_UNALIGNED_SLICE_BIT_KHR,
    eDifferentSliceType             = VK_VIDEO_ENCODE_H264_CAPABILITY_DIFFERENT_SLICE_TYPE_BIT_KHR,
    eBFrameInL0List                 = VK_VIDEO_ENCODE_H264_CAPABILITY_B_FRAME_IN_L0_LIST_BIT_KHR,
    eBFrameInL1List                 = VK_VIDEO_ENCODE_H264_CAPABILITY_B_FRAME_IN_L1_LIST_BIT_KHR,
    ePerPictureTypeMinMaxQp         = VK_VIDEO_ENCODE_H264_CAPABILITY_PER_PICTURE_TYPE_MIN_MAX_QP_BIT_KHR,
    ePerSliceConstantQp             = VK_VIDEO_ENCODE_H264_CAPABILITY_PER_SLICE_CONSTANT_QP_BIT_KHR,
    eGeneratePrefixNalu             = VK_VIDEO_ENCODE_H264_CAPABILITY_GENERATE_PREFIX_NALU_BIT_KHR,
    eBPictureIntraRefresh           = VK_VIDEO_ENCODE_H264_CAPABILITY_B_PICTURE_INTRA_REFRESH_BIT_KHR,
    eMbQpDiffWraparound             = VK_VIDEO_ENCODE_H264_CAPABILITY_MB_QP_DIFF_WRAPAROUND_BIT_KHR
  };

  // wrapper using for bitmask VkVideoEncodeH264CapabilityFlagsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeH264CapabilityFlagsKHR.html
  using VideoEncodeH264CapabilityFlagsKHR = Flags<VideoEncodeH264CapabilityFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeH264CapabilityFlagBitsKHR>
  {
    using WrappedType                                                                = VkVideoEncodeH264CapabilityFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                              isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeH264CapabilityFlagsKHR allFlags =
      VideoEncodeH264CapabilityFlagBitsKHR::eHrdCompliance | VideoEncodeH264CapabilityFlagBitsKHR::ePredictionWeightTableGenerated |
      VideoEncodeH264CapabilityFlagBitsKHR::eRowUnalignedSlice | VideoEncodeH264CapabilityFlagBitsKHR::eDifferentSliceType |
      VideoEncodeH264CapabilityFlagBitsKHR::eBFrameInL0List | VideoEncodeH264CapabilityFlagBitsKHR::eBFrameInL1List |
      VideoEncodeH264CapabilityFlagBitsKHR::ePerPictureTypeMinMaxQp | VideoEncodeH264CapabilityFlagBitsKHR::ePerSliceConstantQp |
      VideoEncodeH264CapabilityFlagBitsKHR::eGeneratePrefixNalu | VideoEncodeH264CapabilityFlagBitsKHR::eBPictureIntraRefresh |
      VideoEncodeH264CapabilityFlagBitsKHR::eMbQpDiffWraparound;
  };

  // wrapper class for enum VkVideoEncodeH264StdFlagBitsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeH264StdFlagBitsKHR.html
  enum class VideoEncodeH264StdFlagBitsKHR : VkVideoEncodeH264StdFlagsKHR
  {
    eSeparateColorPlaneFlagSet          = VK_VIDEO_ENCODE_H264_STD_SEPARATE_COLOR_PLANE_FLAG_SET_BIT_KHR,
    eQpprimeYZeroTransformBypassFlagSet = VK_VIDEO_ENCODE_H264_STD_QPPRIME_Y_ZERO_TRANSFORM_BYPASS_FLAG_SET_BIT_KHR,
    eScalingMatrixPresentFlagSet        = VK_VIDEO_ENCODE_H264_STD_SCALING_MATRIX_PRESENT_FLAG_SET_BIT_KHR,
    eChromaQpIndexOffset                = VK_VIDEO_ENCODE_H264_STD_CHROMA_QP_INDEX_OFFSET_BIT_KHR,
    eSecondChromaQpIndexOffset          = VK_VIDEO_ENCODE_H264_STD_SECOND_CHROMA_QP_INDEX_OFFSET_BIT_KHR,
    ePicInitQpMinus26                   = VK_VIDEO_ENCODE_H264_STD_PIC_INIT_QP_MINUS26_BIT_KHR,
    eWeightedPredFlagSet                = VK_VIDEO_ENCODE_H264_STD_WEIGHTED_PRED_FLAG_SET_BIT_KHR,
    eWeightedBipredIdcExplicit          = VK_VIDEO_ENCODE_H264_STD_WEIGHTED_BIPRED_IDC_EXPLICIT_BIT_KHR,
    eWeightedBipredIdcImplicit          = VK_VIDEO_ENCODE_H264_STD_WEIGHTED_BIPRED_IDC_IMPLICIT_BIT_KHR,
    eTransform8X8ModeFlagSet            = VK_VIDEO_ENCODE_H264_STD_TRANSFORM_8X8_MODE_FLAG_SET_BIT_KHR,
    eDirectSpatialMvPredFlagUnset       = VK_VIDEO_ENCODE_H264_STD_DIRECT_SPATIAL_MV_PRED_FLAG_UNSET_BIT_KHR,
    eEntropyCodingModeFlagUnset         = VK_VIDEO_ENCODE_H264_STD_ENTROPY_CODING_MODE_FLAG_UNSET_BIT_KHR,
    eEntropyCodingModeFlagSet           = VK_VIDEO_ENCODE_H264_STD_ENTROPY_CODING_MODE_FLAG_SET_BIT_KHR,
    eDirect8X8InferenceFlagUnset        = VK_VIDEO_ENCODE_H264_STD_DIRECT_8X8_INFERENCE_FLAG_UNSET_BIT_KHR,
    eConstrainedIntraPredFlagSet        = VK_VIDEO_ENCODE_H264_STD_CONSTRAINED_INTRA_PRED_FLAG_SET_BIT_KHR,
    eDeblockingFilterDisabled           = VK_VIDEO_ENCODE_H264_STD_DEBLOCKING_FILTER_DISABLED_BIT_KHR,
    eDeblockingFilterEnabled            = VK_VIDEO_ENCODE_H264_STD_DEBLOCKING_FILTER_ENABLED_BIT_KHR,
    eDeblockingFilterPartial            = VK_VIDEO_ENCODE_H264_STD_DEBLOCKING_FILTER_PARTIAL_BIT_KHR,
    eSliceQpDelta                       = VK_VIDEO_ENCODE_H264_STD_SLICE_QP_DELTA_BIT_KHR,
    eDifferentSliceQpDelta              = VK_VIDEO_ENCODE_H264_STD_DIFFERENT_SLICE_QP_DELTA_BIT_KHR
  };

  // wrapper using for bitmask VkVideoEncodeH264StdFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeH264StdFlagsKHR.html
  using VideoEncodeH264StdFlagsKHR = Flags<VideoEncodeH264StdFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeH264StdFlagBitsKHR>
  {
    using WrappedType                                                         = VkVideoEncodeH264StdFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeH264StdFlagsKHR allFlags =
      VideoEncodeH264StdFlagBitsKHR::eSeparateColorPlaneFlagSet | VideoEncodeH264StdFlagBitsKHR::eQpprimeYZeroTransformBypassFlagSet |
      VideoEncodeH264StdFlagBitsKHR::eScalingMatrixPresentFlagSet | VideoEncodeH264StdFlagBitsKHR::eChromaQpIndexOffset |
      VideoEncodeH264StdFlagBitsKHR::eSecondChromaQpIndexOffset | VideoEncodeH264StdFlagBitsKHR::ePicInitQpMinus26 |
      VideoEncodeH264StdFlagBitsKHR::eWeightedPredFlagSet | VideoEncodeH264StdFlagBitsKHR::eWeightedBipredIdcExplicit |
      VideoEncodeH264StdFlagBitsKHR::eWeightedBipredIdcImplicit | VideoEncodeH264StdFlagBitsKHR::eTransform8X8ModeFlagSet |
      VideoEncodeH264StdFlagBitsKHR::eDirectSpatialMvPredFlagUnset | VideoEncodeH264StdFlagBitsKHR::eEntropyCodingModeFlagUnset |
      VideoEncodeH264StdFlagBitsKHR::eEntropyCodingModeFlagSet | VideoEncodeH264StdFlagBitsKHR::eDirect8X8InferenceFlagUnset |
      VideoEncodeH264StdFlagBitsKHR::eConstrainedIntraPredFlagSet | VideoEncodeH264StdFlagBitsKHR::eDeblockingFilterDisabled |
      VideoEncodeH264StdFlagBitsKHR::eDeblockingFilterEnabled | VideoEncodeH264StdFlagBitsKHR::eDeblockingFilterPartial |
      VideoEncodeH264StdFlagBitsKHR::eSliceQpDelta | VideoEncodeH264StdFlagBitsKHR::eDifferentSliceQpDelta;
  };

  // wrapper class for enum VkVideoEncodeH264RateControlFlagBitsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeH264RateControlFlagBitsKHR.html
  enum class VideoEncodeH264RateControlFlagBitsKHR : VkVideoEncodeH264RateControlFlagsKHR
  {
    eAttemptHrdCompliance       = VK_VIDEO_ENCODE_H264_RATE_CONTROL_ATTEMPT_HRD_COMPLIANCE_BIT_KHR,
    eRegularGop                 = VK_VIDEO_ENCODE_H264_RATE_CONTROL_REGULAR_GOP_BIT_KHR,
    eReferencePatternFlat       = VK_VIDEO_ENCODE_H264_RATE_CONTROL_REFERENCE_PATTERN_FLAT_BIT_KHR,
    eReferencePatternDyadic     = VK_VIDEO_ENCODE_H264_RATE_CONTROL_REFERENCE_PATTERN_DYADIC_BIT_KHR,
    eTemporalLayerPatternDyadic = VK_VIDEO_ENCODE_H264_RATE_CONTROL_TEMPORAL_LAYER_PATTERN_DYADIC_BIT_KHR
  };

  // wrapper using for bitmask VkVideoEncodeH264RateControlFlagsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeH264RateControlFlagsKHR.html
  using VideoEncodeH264RateControlFlagsKHR = Flags<VideoEncodeH264RateControlFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeH264RateControlFlagBitsKHR>
  {
    using WrappedType                                                                 = VkVideoEncodeH264RateControlFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                               isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeH264RateControlFlagsKHR allFlags =
      VideoEncodeH264RateControlFlagBitsKHR::eAttemptHrdCompliance | VideoEncodeH264RateControlFlagBitsKHR::eRegularGop |
      VideoEncodeH264RateControlFlagBitsKHR::eReferencePatternFlat | VideoEncodeH264RateControlFlagBitsKHR::eReferencePatternDyadic |
      VideoEncodeH264RateControlFlagBitsKHR::eTemporalLayerPatternDyadic;
  };

  //=== VK_KHR_video_encode_h265 ===

  // wrapper class for enum VkVideoEncodeH265CapabilityFlagBitsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeH265CapabilityFlagBitsKHR.html
  enum class VideoEncodeH265CapabilityFlagBitsKHR : VkVideoEncodeH265CapabilityFlagsKHR
  {
    eHrdCompliance                  = VK_VIDEO_ENCODE_H265_CAPABILITY_HRD_COMPLIANCE_BIT_KHR,
    ePredictionWeightTableGenerated = VK_VIDEO_ENCODE_H265_CAPABILITY_PREDICTION_WEIGHT_TABLE_GENERATED_BIT_KHR,
    eRowUnalignedSliceSegment       = VK_VIDEO_ENCODE_H265_CAPABILITY_ROW_UNALIGNED_SLICE_SEGMENT_BIT_KHR,
    eDifferentSliceSegmentType      = VK_VIDEO_ENCODE_H265_CAPABILITY_DIFFERENT_SLICE_SEGMENT_TYPE_BIT_KHR,
    eBFrameInL0List                 = VK_VIDEO_ENCODE_H265_CAPABILITY_B_FRAME_IN_L0_LIST_BIT_KHR,
    eBFrameInL1List                 = VK_VIDEO_ENCODE_H265_CAPABILITY_B_FRAME_IN_L1_LIST_BIT_KHR,
    ePerPictureTypeMinMaxQp         = VK_VIDEO_ENCODE_H265_CAPABILITY_PER_PICTURE_TYPE_MIN_MAX_QP_BIT_KHR,
    ePerSliceSegmentConstantQp      = VK_VIDEO_ENCODE_H265_CAPABILITY_PER_SLICE_SEGMENT_CONSTANT_QP_BIT_KHR,
    eMultipleTilesPerSliceSegment   = VK_VIDEO_ENCODE_H265_CAPABILITY_MULTIPLE_TILES_PER_SLICE_SEGMENT_BIT_KHR,
    eMultipleSliceSegmentsPerTile   = VK_VIDEO_ENCODE_H265_CAPABILITY_MULTIPLE_SLICE_SEGMENTS_PER_TILE_BIT_KHR,
    eBPictureIntraRefresh           = VK_VIDEO_ENCODE_H265_CAPABILITY_B_PICTURE_INTRA_REFRESH_BIT_KHR,
    eCuQpDiffWraparound             = VK_VIDEO_ENCODE_H265_CAPABILITY_CU_QP_DIFF_WRAPAROUND_BIT_KHR
  };

  // wrapper using for bitmask VkVideoEncodeH265CapabilityFlagsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeH265CapabilityFlagsKHR.html
  using VideoEncodeH265CapabilityFlagsKHR = Flags<VideoEncodeH265CapabilityFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeH265CapabilityFlagBitsKHR>
  {
    using WrappedType                                                                = VkVideoEncodeH265CapabilityFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                              isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeH265CapabilityFlagsKHR allFlags =
      VideoEncodeH265CapabilityFlagBitsKHR::eHrdCompliance | VideoEncodeH265CapabilityFlagBitsKHR::ePredictionWeightTableGenerated |
      VideoEncodeH265CapabilityFlagBitsKHR::eRowUnalignedSliceSegment | VideoEncodeH265CapabilityFlagBitsKHR::eDifferentSliceSegmentType |
      VideoEncodeH265CapabilityFlagBitsKHR::eBFrameInL0List | VideoEncodeH265CapabilityFlagBitsKHR::eBFrameInL1List |
      VideoEncodeH265CapabilityFlagBitsKHR::ePerPictureTypeMinMaxQp | VideoEncodeH265CapabilityFlagBitsKHR::ePerSliceSegmentConstantQp |
      VideoEncodeH265CapabilityFlagBitsKHR::eMultipleTilesPerSliceSegment | VideoEncodeH265CapabilityFlagBitsKHR::eMultipleSliceSegmentsPerTile |
      VideoEncodeH265CapabilityFlagBitsKHR::eBPictureIntraRefresh | VideoEncodeH265CapabilityFlagBitsKHR::eCuQpDiffWraparound;
  };

  // wrapper class for enum VkVideoEncodeH265StdFlagBitsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeH265StdFlagBitsKHR.html
  enum class VideoEncodeH265StdFlagBitsKHR : VkVideoEncodeH265StdFlagsKHR
  {
    eSeparateColorPlaneFlagSet              = VK_VIDEO_ENCODE_H265_STD_SEPARATE_COLOR_PLANE_FLAG_SET_BIT_KHR,
    eSampleAdaptiveOffsetEnabledFlagSet     = VK_VIDEO_ENCODE_H265_STD_SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG_SET_BIT_KHR,
    eScalingListDataPresentFlagSet          = VK_VIDEO_ENCODE_H265_STD_SCALING_LIST_DATA_PRESENT_FLAG_SET_BIT_KHR,
    ePcmEnabledFlagSet                      = VK_VIDEO_ENCODE_H265_STD_PCM_ENABLED_FLAG_SET_BIT_KHR,
    eSpsTemporalMvpEnabledFlagSet           = VK_VIDEO_ENCODE_H265_STD_SPS_TEMPORAL_MVP_ENABLED_FLAG_SET_BIT_KHR,
    eInitQpMinus26                          = VK_VIDEO_ENCODE_H265_STD_INIT_QP_MINUS26_BIT_KHR,
    eWeightedPredFlagSet                    = VK_VIDEO_ENCODE_H265_STD_WEIGHTED_PRED_FLAG_SET_BIT_KHR,
    eWeightedBipredFlagSet                  = VK_VIDEO_ENCODE_H265_STD_WEIGHTED_BIPRED_FLAG_SET_BIT_KHR,
    eLog2ParallelMergeLevelMinus2           = VK_VIDEO_ENCODE_H265_STD_LOG2_PARALLEL_MERGE_LEVEL_MINUS2_BIT_KHR,
    eSignDataHidingEnabledFlagSet           = VK_VIDEO_ENCODE_H265_STD_SIGN_DATA_HIDING_ENABLED_FLAG_SET_BIT_KHR,
    eTransformSkipEnabledFlagSet            = VK_VIDEO_ENCODE_H265_STD_TRANSFORM_SKIP_ENABLED_FLAG_SET_BIT_KHR,
    eTransformSkipEnabledFlagUnset          = VK_VIDEO_ENCODE_H265_STD_TRANSFORM_SKIP_ENABLED_FLAG_UNSET_BIT_KHR,
    ePpsSliceChromaQpOffsetsPresentFlagSet  = VK_VIDEO_ENCODE_H265_STD_PPS_SLICE_CHROMA_QP_OFFSETS_PRESENT_FLAG_SET_BIT_KHR,
    eTransquantBypassEnabledFlagSet         = VK_VIDEO_ENCODE_H265_STD_TRANSQUANT_BYPASS_ENABLED_FLAG_SET_BIT_KHR,
    eConstrainedIntraPredFlagSet            = VK_VIDEO_ENCODE_H265_STD_CONSTRAINED_INTRA_PRED_FLAG_SET_BIT_KHR,
    eEntropyCodingSyncEnabledFlagSet        = VK_VIDEO_ENCODE_H265_STD_ENTROPY_CODING_SYNC_ENABLED_FLAG_SET_BIT_KHR,
    eDeblockingFilterOverrideEnabledFlagSet = VK_VIDEO_ENCODE_H265_STD_DEBLOCKING_FILTER_OVERRIDE_ENABLED_FLAG_SET_BIT_KHR,
    eDependentSliceSegmentsEnabledFlagSet   = VK_VIDEO_ENCODE_H265_STD_DEPENDENT_SLICE_SEGMENTS_ENABLED_FLAG_SET_BIT_KHR,
    eDependentSliceSegmentFlagSet           = VK_VIDEO_ENCODE_H265_STD_DEPENDENT_SLICE_SEGMENT_FLAG_SET_BIT_KHR,
    eSliceQpDelta                           = VK_VIDEO_ENCODE_H265_STD_SLICE_QP_DELTA_BIT_KHR,
    eDifferentSliceQpDelta                  = VK_VIDEO_ENCODE_H265_STD_DIFFERENT_SLICE_QP_DELTA_BIT_KHR
  };

  // wrapper using for bitmask VkVideoEncodeH265StdFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeH265StdFlagsKHR.html
  using VideoEncodeH265StdFlagsKHR = Flags<VideoEncodeH265StdFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeH265StdFlagBitsKHR>
  {
    using WrappedType                                                         = VkVideoEncodeH265StdFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeH265StdFlagsKHR allFlags =
      VideoEncodeH265StdFlagBitsKHR::eSeparateColorPlaneFlagSet | VideoEncodeH265StdFlagBitsKHR::eSampleAdaptiveOffsetEnabledFlagSet |
      VideoEncodeH265StdFlagBitsKHR::eScalingListDataPresentFlagSet | VideoEncodeH265StdFlagBitsKHR::ePcmEnabledFlagSet |
      VideoEncodeH265StdFlagBitsKHR::eSpsTemporalMvpEnabledFlagSet | VideoEncodeH265StdFlagBitsKHR::eInitQpMinus26 |
      VideoEncodeH265StdFlagBitsKHR::eWeightedPredFlagSet | VideoEncodeH265StdFlagBitsKHR::eWeightedBipredFlagSet |
      VideoEncodeH265StdFlagBitsKHR::eLog2ParallelMergeLevelMinus2 | VideoEncodeH265StdFlagBitsKHR::eSignDataHidingEnabledFlagSet |
      VideoEncodeH265StdFlagBitsKHR::eTransformSkipEnabledFlagSet | VideoEncodeH265StdFlagBitsKHR::eTransformSkipEnabledFlagUnset |
      VideoEncodeH265StdFlagBitsKHR::ePpsSliceChromaQpOffsetsPresentFlagSet | VideoEncodeH265StdFlagBitsKHR::eTransquantBypassEnabledFlagSet |
      VideoEncodeH265StdFlagBitsKHR::eConstrainedIntraPredFlagSet | VideoEncodeH265StdFlagBitsKHR::eEntropyCodingSyncEnabledFlagSet |
      VideoEncodeH265StdFlagBitsKHR::eDeblockingFilterOverrideEnabledFlagSet | VideoEncodeH265StdFlagBitsKHR::eDependentSliceSegmentsEnabledFlagSet |
      VideoEncodeH265StdFlagBitsKHR::eDependentSliceSegmentFlagSet | VideoEncodeH265StdFlagBitsKHR::eSliceQpDelta |
      VideoEncodeH265StdFlagBitsKHR::eDifferentSliceQpDelta;
  };

  // wrapper class for enum VkVideoEncodeH265CtbSizeFlagBitsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeH265CtbSizeFlagBitsKHR.html
  enum class VideoEncodeH265CtbSizeFlagBitsKHR : VkVideoEncodeH265CtbSizeFlagsKHR
  {
    e16 = VK_VIDEO_ENCODE_H265_CTB_SIZE_16_BIT_KHR,
    e32 = VK_VIDEO_ENCODE_H265_CTB_SIZE_32_BIT_KHR,
    e64 = VK_VIDEO_ENCODE_H265_CTB_SIZE_64_BIT_KHR
  };

  // wrapper using for bitmask VkVideoEncodeH265CtbSizeFlagsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeH265CtbSizeFlagsKHR.html
  using VideoEncodeH265CtbSizeFlagsKHR = Flags<VideoEncodeH265CtbSizeFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeH265CtbSizeFlagBitsKHR>
  {
    using WrappedType                                                             = VkVideoEncodeH265CtbSizeFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeH265CtbSizeFlagsKHR allFlags =
      VideoEncodeH265CtbSizeFlagBitsKHR::e16 | VideoEncodeH265CtbSizeFlagBitsKHR::e32 | VideoEncodeH265CtbSizeFlagBitsKHR::e64;
  };

  // wrapper class for enum VkVideoEncodeH265TransformBlockSizeFlagBitsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeH265TransformBlockSizeFlagBitsKHR.html
  enum class VideoEncodeH265TransformBlockSizeFlagBitsKHR : VkVideoEncodeH265TransformBlockSizeFlagsKHR
  {
    e4  = VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_4_BIT_KHR,
    e8  = VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_8_BIT_KHR,
    e16 = VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_16_BIT_KHR,
    e32 = VK_VIDEO_ENCODE_H265_TRANSFORM_BLOCK_SIZE_32_BIT_KHR
  };

  // wrapper using for bitmask VkVideoEncodeH265TransformBlockSizeFlagsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeH265TransformBlockSizeFlagsKHR.html
  using VideoEncodeH265TransformBlockSizeFlagsKHR = Flags<VideoEncodeH265TransformBlockSizeFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeH265TransformBlockSizeFlagBitsKHR>
  {
    using WrappedType                                                                        = VkVideoEncodeH265TransformBlockSizeFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                      isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeH265TransformBlockSizeFlagsKHR allFlags =
      VideoEncodeH265TransformBlockSizeFlagBitsKHR::e4 | VideoEncodeH265TransformBlockSizeFlagBitsKHR::e8 | VideoEncodeH265TransformBlockSizeFlagBitsKHR::e16 |
      VideoEncodeH265TransformBlockSizeFlagBitsKHR::e32;
  };

  // wrapper class for enum VkVideoEncodeH265RateControlFlagBitsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeH265RateControlFlagBitsKHR.html
  enum class VideoEncodeH265RateControlFlagBitsKHR : VkVideoEncodeH265RateControlFlagsKHR
  {
    eAttemptHrdCompliance          = VK_VIDEO_ENCODE_H265_RATE_CONTROL_ATTEMPT_HRD_COMPLIANCE_BIT_KHR,
    eRegularGop                    = VK_VIDEO_ENCODE_H265_RATE_CONTROL_REGULAR_GOP_BIT_KHR,
    eReferencePatternFlat          = VK_VIDEO_ENCODE_H265_RATE_CONTROL_REFERENCE_PATTERN_FLAT_BIT_KHR,
    eReferencePatternDyadic        = VK_VIDEO_ENCODE_H265_RATE_CONTROL_REFERENCE_PATTERN_DYADIC_BIT_KHR,
    eTemporalSubLayerPatternDyadic = VK_VIDEO_ENCODE_H265_RATE_CONTROL_TEMPORAL_SUB_LAYER_PATTERN_DYADIC_BIT_KHR
  };

  // wrapper using for bitmask VkVideoEncodeH265RateControlFlagsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeH265RateControlFlagsKHR.html
  using VideoEncodeH265RateControlFlagsKHR = Flags<VideoEncodeH265RateControlFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeH265RateControlFlagBitsKHR>
  {
    using WrappedType                                                                 = VkVideoEncodeH265RateControlFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                               isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeH265RateControlFlagsKHR allFlags =
      VideoEncodeH265RateControlFlagBitsKHR::eAttemptHrdCompliance | VideoEncodeH265RateControlFlagBitsKHR::eRegularGop |
      VideoEncodeH265RateControlFlagBitsKHR::eReferencePatternFlat | VideoEncodeH265RateControlFlagBitsKHR::eReferencePatternDyadic |
      VideoEncodeH265RateControlFlagBitsKHR::eTemporalSubLayerPatternDyadic;
  };

  //=== VK_KHR_video_decode_h264 ===

  // wrapper class for enum VkVideoDecodeH264PictureLayoutFlagBitsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoDecodeH264PictureLayoutFlagBitsKHR.html
  enum class VideoDecodeH264PictureLayoutFlagBitsKHR : VkVideoDecodeH264PictureLayoutFlagsKHR
  {
    eProgressive                = VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_PROGRESSIVE_KHR,
    eInterlacedInterleavedLines = VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_INTERLACED_INTERLEAVED_LINES_BIT_KHR,
    eInterlacedSeparatePlanes   = VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_INTERLACED_SEPARATE_PLANES_BIT_KHR
  };

  // wrapper using for bitmask VkVideoDecodeH264PictureLayoutFlagsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoDecodeH264PictureLayoutFlagsKHR.html
  using VideoDecodeH264PictureLayoutFlagsKHR = Flags<VideoDecodeH264PictureLayoutFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoDecodeH264PictureLayoutFlagBitsKHR>
  {
    using WrappedType                                                                   = VkVideoDecodeH264PictureLayoutFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoDecodeH264PictureLayoutFlagsKHR allFlags  = VideoDecodeH264PictureLayoutFlagBitsKHR::eProgressive |
                                                                                         VideoDecodeH264PictureLayoutFlagBitsKHR::eInterlacedInterleavedLines |
                                                                                         VideoDecodeH264PictureLayoutFlagBitsKHR::eInterlacedSeparatePlanes;
  };

  //=== VK_AMD_shader_info ===

  // wrapper class for enum VkShaderInfoTypeAMD, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkShaderInfoTypeAMD.html
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

  // wrapper using for bitmask VkStreamDescriptorSurfaceCreateFlagsGGP, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkStreamDescriptorSurfaceCreateFlagsGGP.html
  using StreamDescriptorSurfaceCreateFlagsGGP = Flags<StreamDescriptorSurfaceCreateFlagBitsGGP>;

  template <>
  struct FlagTraits<StreamDescriptorSurfaceCreateFlagBitsGGP>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                  isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR StreamDescriptorSurfaceCreateFlagsGGP allFlags  = {};
  };
#endif /*VK_USE_PLATFORM_GGP*/

  //=== VK_NV_external_memory_capabilities ===

  // wrapper class for enum VkExternalMemoryHandleTypeFlagBitsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkExternalMemoryHandleTypeFlagBitsNV.html
  enum class ExternalMemoryHandleTypeFlagBitsNV : VkExternalMemoryHandleTypeFlagsNV
  {
    eOpaqueWin32    = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_NV,
    eOpaqueWin32Kmt = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_NV,
    eD3D11Image     = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_IMAGE_BIT_NV,
    eD3D11ImageKmt  = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_IMAGE_KMT_BIT_NV
  };

  // wrapper using for bitmask VkExternalMemoryHandleTypeFlagsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkExternalMemoryHandleTypeFlagsNV.html
  using ExternalMemoryHandleTypeFlagsNV = Flags<ExternalMemoryHandleTypeFlagBitsNV>;

  template <>
  struct FlagTraits<ExternalMemoryHandleTypeFlagBitsNV>
  {
    using WrappedType                                                              = VkExternalMemoryHandleTypeFlagBitsNV;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                            isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ExternalMemoryHandleTypeFlagsNV allFlags =
      ExternalMemoryHandleTypeFlagBitsNV::eOpaqueWin32 | ExternalMemoryHandleTypeFlagBitsNV::eOpaqueWin32Kmt | ExternalMemoryHandleTypeFlagBitsNV::eD3D11Image |
      ExternalMemoryHandleTypeFlagBitsNV::eD3D11ImageKmt;
  };

  // wrapper class for enum VkExternalMemoryFeatureFlagBitsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkExternalMemoryFeatureFlagBitsNV.html
  enum class ExternalMemoryFeatureFlagBitsNV : VkExternalMemoryFeatureFlagsNV
  {
    eDedicatedOnly = VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT_NV,
    eExportable    = VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT_NV,
    eImportable    = VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT_NV
  };

  // wrapper using for bitmask VkExternalMemoryFeatureFlagsNV, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkExternalMemoryFeatureFlagsNV.html
  using ExternalMemoryFeatureFlagsNV = Flags<ExternalMemoryFeatureFlagBitsNV>;

  template <>
  struct FlagTraits<ExternalMemoryFeatureFlagBitsNV>
  {
    using WrappedType                                                           = VkExternalMemoryFeatureFlagBitsNV;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                         isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ExternalMemoryFeatureFlagsNV allFlags =
      ExternalMemoryFeatureFlagBitsNV::eDedicatedOnly | ExternalMemoryFeatureFlagBitsNV::eExportable | ExternalMemoryFeatureFlagBitsNV::eImportable;
  };

  //=== VK_EXT_validation_flags ===

  // wrapper class for enum VkValidationCheckEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkValidationCheckEXT.html
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

  // wrapper using for bitmask VkViSurfaceCreateFlagsNN, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkViSurfaceCreateFlagsNN.html
  using ViSurfaceCreateFlagsNN = Flags<ViSurfaceCreateFlagBitsNN>;

  template <>
  struct FlagTraits<ViSurfaceCreateFlagBitsNN>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ViSurfaceCreateFlagsNN allFlags  = {};
  };
#endif /*VK_USE_PLATFORM_VI_NN*/

  //=== VK_EXT_conditional_rendering ===

  // wrapper class for enum VkConditionalRenderingFlagBitsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkConditionalRenderingFlagBitsEXT.html
  enum class ConditionalRenderingFlagBitsEXT : VkConditionalRenderingFlagsEXT
  {
    eInverted = VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT
  };

  // wrapper using for bitmask VkConditionalRenderingFlagsEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkConditionalRenderingFlagsEXT.html
  using ConditionalRenderingFlagsEXT = Flags<ConditionalRenderingFlagBitsEXT>;

  template <>
  struct FlagTraits<ConditionalRenderingFlagBitsEXT>
  {
    using WrappedType                                                           = VkConditionalRenderingFlagBitsEXT;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                         isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ConditionalRenderingFlagsEXT allFlags  = ConditionalRenderingFlagBitsEXT::eInverted;
  };

  //=== VK_EXT_display_surface_counter ===

  // wrapper class for enum VkSurfaceCounterFlagBitsEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSurfaceCounterFlagBitsEXT.html
  enum class SurfaceCounterFlagBitsEXT : VkSurfaceCounterFlagsEXT
  {
    eVblank = VK_SURFACE_COUNTER_VBLANK_BIT_EXT
  };

  // wrapper using for bitmask VkSurfaceCounterFlagsEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSurfaceCounterFlagsEXT.html
  using SurfaceCounterFlagsEXT = Flags<SurfaceCounterFlagBitsEXT>;

  template <>
  struct FlagTraits<SurfaceCounterFlagBitsEXT>
  {
    using WrappedType                                                     = VkSurfaceCounterFlagBitsEXT;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SurfaceCounterFlagsEXT allFlags  = SurfaceCounterFlagBitsEXT::eVblank;
  };

  //=== VK_EXT_display_control ===

  // wrapper class for enum VkDisplayPowerStateEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDisplayPowerStateEXT.html
  enum class DisplayPowerStateEXT
  {
    eOff     = VK_DISPLAY_POWER_STATE_OFF_EXT,
    eSuspend = VK_DISPLAY_POWER_STATE_SUSPEND_EXT,
    eOn      = VK_DISPLAY_POWER_STATE_ON_EXT
  };

  // wrapper class for enum VkDeviceEventTypeEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceEventTypeEXT.html
  enum class DeviceEventTypeEXT
  {
    eDisplayHotplug = VK_DEVICE_EVENT_TYPE_DISPLAY_HOTPLUG_EXT
  };

  // wrapper class for enum VkDisplayEventTypeEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDisplayEventTypeEXT.html
  enum class DisplayEventTypeEXT
  {
    eFirstPixelOut = VK_DISPLAY_EVENT_TYPE_FIRST_PIXEL_OUT_EXT
  };

  //=== VK_NV_viewport_swizzle ===

  // wrapper class for enum VkViewportCoordinateSwizzleNV, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkViewportCoordinateSwizzleNV.html
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

  // wrapper using for bitmask VkPipelineViewportSwizzleStateCreateFlagsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineViewportSwizzleStateCreateFlagsNV.html
  using PipelineViewportSwizzleStateCreateFlagsNV = Flags<PipelineViewportSwizzleStateCreateFlagBitsNV>;

  template <>
  struct FlagTraits<PipelineViewportSwizzleStateCreateFlagBitsNV>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                      isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineViewportSwizzleStateCreateFlagsNV allFlags  = {};
  };

  //=== VK_EXT_discard_rectangles ===

  // wrapper class for enum VkDiscardRectangleModeEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDiscardRectangleModeEXT.html
  enum class DiscardRectangleModeEXT
  {
    eInclusive = VK_DISCARD_RECTANGLE_MODE_INCLUSIVE_EXT,
    eExclusive = VK_DISCARD_RECTANGLE_MODE_EXCLUSIVE_EXT
  };

  enum class PipelineDiscardRectangleStateCreateFlagBitsEXT : VkPipelineDiscardRectangleStateCreateFlagsEXT
  {
  };

  // wrapper using for bitmask VkPipelineDiscardRectangleStateCreateFlagsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineDiscardRectangleStateCreateFlagsEXT.html
  using PipelineDiscardRectangleStateCreateFlagsEXT = Flags<PipelineDiscardRectangleStateCreateFlagBitsEXT>;

  template <>
  struct FlagTraits<PipelineDiscardRectangleStateCreateFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                        isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineDiscardRectangleStateCreateFlagsEXT allFlags  = {};
  };

  //=== VK_EXT_conservative_rasterization ===

  // wrapper class for enum VkConservativeRasterizationModeEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkConservativeRasterizationModeEXT.html
  enum class ConservativeRasterizationModeEXT
  {
    eDisabled      = VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT,
    eOverestimate  = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT,
    eUnderestimate = VK_CONSERVATIVE_RASTERIZATION_MODE_UNDERESTIMATE_EXT
  };

  enum class PipelineRasterizationConservativeStateCreateFlagBitsEXT : VkPipelineRasterizationConservativeStateCreateFlagsEXT
  {
  };

  // wrapper using for bitmask VkPipelineRasterizationConservativeStateCreateFlagsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineRasterizationConservativeStateCreateFlagsEXT.html
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

  // wrapper using for bitmask VkPipelineRasterizationDepthClipStateCreateFlagsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineRasterizationDepthClipStateCreateFlagsEXT.html
  using PipelineRasterizationDepthClipStateCreateFlagsEXT = Flags<PipelineRasterizationDepthClipStateCreateFlagBitsEXT>;

  template <>
  struct FlagTraits<PipelineRasterizationDepthClipStateCreateFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                              isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineRasterizationDepthClipStateCreateFlagsEXT allFlags  = {};
  };

  //=== VK_KHR_performance_query ===

  // wrapper class for enum VkPerformanceCounterDescriptionFlagBitsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPerformanceCounterDescriptionFlagBitsKHR.html
  enum class PerformanceCounterDescriptionFlagBitsKHR : VkPerformanceCounterDescriptionFlagsKHR
  {
    ePerformanceImpacting = VK_PERFORMANCE_COUNTER_DESCRIPTION_PERFORMANCE_IMPACTING_BIT_KHR,
    eConcurrentlyImpacted = VK_PERFORMANCE_COUNTER_DESCRIPTION_CONCURRENTLY_IMPACTED_BIT_KHR
  };

  // wrapper using for bitmask VkPerformanceCounterDescriptionFlagsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPerformanceCounterDescriptionFlagsKHR.html
  using PerformanceCounterDescriptionFlagsKHR = Flags<PerformanceCounterDescriptionFlagBitsKHR>;

  template <>
  struct FlagTraits<PerformanceCounterDescriptionFlagBitsKHR>
  {
    using WrappedType                                                                    = VkPerformanceCounterDescriptionFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                  isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PerformanceCounterDescriptionFlagsKHR allFlags =
      PerformanceCounterDescriptionFlagBitsKHR::ePerformanceImpacting | PerformanceCounterDescriptionFlagBitsKHR::eConcurrentlyImpacted;
  };

  // wrapper class for enum VkPerformanceCounterScopeKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPerformanceCounterScopeKHR.html
  enum class PerformanceCounterScopeKHR
  {
    eCommandBuffer             = VK_PERFORMANCE_COUNTER_SCOPE_COMMAND_BUFFER_KHR,
    eVkQueryScopeCommandBuffer = VK_QUERY_SCOPE_COMMAND_BUFFER_KHR,
    eRenderPass                = VK_PERFORMANCE_COUNTER_SCOPE_RENDER_PASS_KHR,
    eVkQueryScopeRenderPass    = VK_QUERY_SCOPE_RENDER_PASS_KHR,
    eCommand                   = VK_PERFORMANCE_COUNTER_SCOPE_COMMAND_KHR,
    eVkQueryScopeCommand       = VK_QUERY_SCOPE_COMMAND_KHR
  };

  // wrapper class for enum VkPerformanceCounterStorageKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPerformanceCounterStorageKHR.html
  enum class PerformanceCounterStorageKHR
  {
    eInt32   = VK_PERFORMANCE_COUNTER_STORAGE_INT32_KHR,
    eInt64   = VK_PERFORMANCE_COUNTER_STORAGE_INT64_KHR,
    eUint32  = VK_PERFORMANCE_COUNTER_STORAGE_UINT32_KHR,
    eUint64  = VK_PERFORMANCE_COUNTER_STORAGE_UINT64_KHR,
    eFloat32 = VK_PERFORMANCE_COUNTER_STORAGE_FLOAT32_KHR,
    eFloat64 = VK_PERFORMANCE_COUNTER_STORAGE_FLOAT64_KHR
  };

  // wrapper class for enum VkPerformanceCounterUnitKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPerformanceCounterUnitKHR.html
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

  // wrapper using for bitmask VkAcquireProfilingLockFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkAcquireProfilingLockFlagsKHR.html
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

  // wrapper using for bitmask VkIOSSurfaceCreateFlagsMVK, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkIOSSurfaceCreateFlagsMVK.html
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

  // wrapper using for bitmask VkMacOSSurfaceCreateFlagsMVK, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkMacOSSurfaceCreateFlagsMVK.html
  using MacOSSurfaceCreateFlagsMVK = Flags<MacOSSurfaceCreateFlagBitsMVK>;

  template <>
  struct FlagTraits<MacOSSurfaceCreateFlagBitsMVK>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR MacOSSurfaceCreateFlagsMVK allFlags  = {};
  };
#endif /*VK_USE_PLATFORM_MACOS_MVK*/

  //=== VK_EXT_debug_utils ===

  // wrapper class for enum VkDebugUtilsMessageSeverityFlagBitsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkDebugUtilsMessageSeverityFlagBitsEXT.html
  enum class DebugUtilsMessageSeverityFlagBitsEXT : VkDebugUtilsMessageSeverityFlagsEXT
  {
    eVerbose = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT,
    eInfo    = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT,
    eWarning = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
    eError   = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
  };

  // wrapper using for bitmask VkDebugUtilsMessageSeverityFlagsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkDebugUtilsMessageSeverityFlagsEXT.html
  using DebugUtilsMessageSeverityFlagsEXT = Flags<DebugUtilsMessageSeverityFlagBitsEXT>;

  template <>
  struct FlagTraits<DebugUtilsMessageSeverityFlagBitsEXT>
  {
    using WrappedType                                                                = VkDebugUtilsMessageSeverityFlagBitsEXT;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                              isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DebugUtilsMessageSeverityFlagsEXT allFlags =
      DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | DebugUtilsMessageSeverityFlagBitsEXT::eInfo | DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
      DebugUtilsMessageSeverityFlagBitsEXT::eError;
  };

  // wrapper class for enum VkDebugUtilsMessageTypeFlagBitsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkDebugUtilsMessageTypeFlagBitsEXT.html
  enum class DebugUtilsMessageTypeFlagBitsEXT : VkDebugUtilsMessageTypeFlagsEXT
  {
    eGeneral              = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT,
    eValidation           = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
    ePerformance          = VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
    eDeviceAddressBinding = VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT
  };

  // wrapper using for bitmask VkDebugUtilsMessageTypeFlagsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkDebugUtilsMessageTypeFlagsEXT.html
  using DebugUtilsMessageTypeFlagsEXT = Flags<DebugUtilsMessageTypeFlagBitsEXT>;

  template <>
  struct FlagTraits<DebugUtilsMessageTypeFlagBitsEXT>
  {
    using WrappedType                                                            = VkDebugUtilsMessageTypeFlagBitsEXT;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                          isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DebugUtilsMessageTypeFlagsEXT allFlags =
      DebugUtilsMessageTypeFlagBitsEXT::eGeneral | DebugUtilsMessageTypeFlagBitsEXT::eValidation | DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
      DebugUtilsMessageTypeFlagBitsEXT::eDeviceAddressBinding;
  };

  enum class DebugUtilsMessengerCallbackDataFlagBitsEXT : VkDebugUtilsMessengerCallbackDataFlagsEXT
  {
  };

  // wrapper using for bitmask VkDebugUtilsMessengerCallbackDataFlagsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkDebugUtilsMessengerCallbackDataFlagsEXT.html
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

  // wrapper using for bitmask VkDebugUtilsMessengerCreateFlagsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkDebugUtilsMessengerCreateFlagsEXT.html
  using DebugUtilsMessengerCreateFlagsEXT = Flags<DebugUtilsMessengerCreateFlagBitsEXT>;

  template <>
  struct FlagTraits<DebugUtilsMessengerCreateFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                              isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DebugUtilsMessengerCreateFlagsEXT allFlags  = {};
  };

  //=== VK_EXT_blend_operation_advanced ===

  // wrapper class for enum VkBlendOverlapEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkBlendOverlapEXT.html
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

  // wrapper using for bitmask VkPipelineCoverageToColorStateCreateFlagsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineCoverageToColorStateCreateFlagsNV.html
  using PipelineCoverageToColorStateCreateFlagsNV = Flags<PipelineCoverageToColorStateCreateFlagBitsNV>;

  template <>
  struct FlagTraits<PipelineCoverageToColorStateCreateFlagBitsNV>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                      isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineCoverageToColorStateCreateFlagsNV allFlags  = {};
  };

  //=== VK_KHR_acceleration_structure ===

  // wrapper class for enum VkAccelerationStructureTypeKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureTypeKHR.html
  enum class AccelerationStructureTypeKHR
  {
    eTopLevel    = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
    eBottomLevel = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
    eGeneric     = VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR
  };

  using AccelerationStructureTypeNV = AccelerationStructureTypeKHR;

  // wrapper class for enum VkAccelerationStructureBuildTypeKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureBuildTypeKHR.html
  enum class AccelerationStructureBuildTypeKHR
  {
    eHost         = VK_ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_KHR,
    eDevice       = VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
    eHostOrDevice = VK_ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_OR_DEVICE_KHR
  };

  // wrapper class for enum VkGeometryFlagBitsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkGeometryFlagBitsKHR.html
  enum class GeometryFlagBitsKHR : VkGeometryFlagsKHR
  {
    eOpaque                      = VK_GEOMETRY_OPAQUE_BIT_KHR,
    eNoDuplicateAnyHitInvocation = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR
  };

  using GeometryFlagBitsNV = GeometryFlagBitsKHR;

  // wrapper using for bitmask VkGeometryFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkGeometryFlagsKHR.html
  using GeometryFlagsKHR = Flags<GeometryFlagBitsKHR>;
  using GeometryFlagsNV  = GeometryFlagsKHR;

  template <>
  struct FlagTraits<GeometryFlagBitsKHR>
  {
    using WrappedType                                               = VkGeometryFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR GeometryFlagsKHR allFlags  = GeometryFlagBitsKHR::eOpaque | GeometryFlagBitsKHR::eNoDuplicateAnyHitInvocation;
  };

  // wrapper class for enum VkGeometryInstanceFlagBitsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkGeometryInstanceFlagBitsKHR.html
  enum class GeometryInstanceFlagBitsKHR : VkGeometryInstanceFlagsKHR
  {
    eTriangleFacingCullDisable     = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
    eTriangleCullDisable           = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV,
    eTriangleFlipFacing            = VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR,
    eTriangleFrontCounterclockwise = VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR,
    eForceOpaque                   = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR,
    eForceNoOpaque                 = VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR,
    eForceOpacityMicromap2StateEXT = VK_GEOMETRY_INSTANCE_FORCE_OPACITY_MICROMAP_2_STATE_BIT_EXT,
    eDisableOpacityMicromapsEXT    = VK_GEOMETRY_INSTANCE_DISABLE_OPACITY_MICROMAPS_BIT_EXT
  };

  using GeometryInstanceFlagBitsNV = GeometryInstanceFlagBitsKHR;

  // wrapper using for bitmask VkGeometryInstanceFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkGeometryInstanceFlagsKHR.html
  using GeometryInstanceFlagsKHR = Flags<GeometryInstanceFlagBitsKHR>;
  using GeometryInstanceFlagsNV  = GeometryInstanceFlagsKHR;

  template <>
  struct FlagTraits<GeometryInstanceFlagBitsKHR>
  {
    using WrappedType                                                       = VkGeometryInstanceFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                     isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR GeometryInstanceFlagsKHR allFlags =
      GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable | GeometryInstanceFlagBitsKHR::eTriangleFlipFacing | GeometryInstanceFlagBitsKHR::eForceOpaque |
      GeometryInstanceFlagBitsKHR::eForceNoOpaque | GeometryInstanceFlagBitsKHR::eForceOpacityMicromap2StateEXT |
      GeometryInstanceFlagBitsKHR::eDisableOpacityMicromapsEXT;
  };

  // wrapper class for enum VkBuildAccelerationStructureFlagBitsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkBuildAccelerationStructureFlagBitsKHR.html
  enum class BuildAccelerationStructureFlagBitsKHR : VkBuildAccelerationStructureFlagsKHR
  {
    eAllowUpdate                       = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
    eAllowCompaction                   = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR,
    ePreferFastTrace                   = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
    ePreferFastBuild                   = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
    eLowMemory                         = VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR,
    eMotionNV                          = VK_BUILD_ACCELERATION_STRUCTURE_MOTION_BIT_NV,
    eAllowOpacityMicromapUpdateEXT     = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_OPACITY_MICROMAP_UPDATE_BIT_EXT,
    eAllowDisableOpacityMicromapsEXT   = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DISABLE_OPACITY_MICROMAPS_BIT_EXT,
    eAllowOpacityMicromapDataUpdateEXT = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_OPACITY_MICROMAP_DATA_UPDATE_BIT_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eAllowDisplacementMicromapUpdateNV = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DISPLACEMENT_MICROMAP_UPDATE_BIT_NV,
    eAllowDisplacementMicromapUpdate   = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DISPLACEMENT_MICROMAP_UPDATE_NV,
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
    eAllowDataAccess                = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_BIT_KHR,
    eAllowClusterOpacityMicromapsNV = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_CLUSTER_OPACITY_MICROMAPS_BIT_NV
  };

  using BuildAccelerationStructureFlagBitsNV = BuildAccelerationStructureFlagBitsKHR;

  // wrapper using for bitmask VkBuildAccelerationStructureFlagsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkBuildAccelerationStructureFlagsKHR.html
  using BuildAccelerationStructureFlagsKHR = Flags<BuildAccelerationStructureFlagBitsKHR>;
  using BuildAccelerationStructureFlagsNV  = BuildAccelerationStructureFlagsKHR;

  template <>
  struct FlagTraits<BuildAccelerationStructureFlagBitsKHR>
  {
    using WrappedType                                                                 = VkBuildAccelerationStructureFlagBitsKHR;
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
      | BuildAccelerationStructureFlagBitsKHR::eAllowDataAccess | BuildAccelerationStructureFlagBitsKHR::eAllowClusterOpacityMicromapsNV;
  };

  // wrapper class for enum VkCopyAccelerationStructureModeKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkCopyAccelerationStructureModeKHR.html
  enum class CopyAccelerationStructureModeKHR
  {
    eClone       = VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR,
    eCompact     = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR,
    eSerialize   = VK_COPY_ACCELERATION_STRUCTURE_MODE_SERIALIZE_KHR,
    eDeserialize = VK_COPY_ACCELERATION_STRUCTURE_MODE_DESERIALIZE_KHR
  };

  using CopyAccelerationStructureModeNV = CopyAccelerationStructureModeKHR;

  // wrapper class for enum VkGeometryTypeKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkGeometryTypeKHR.html
  enum class GeometryTypeKHR
  {
    eTriangles            = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
    eAabbs                = VK_GEOMETRY_TYPE_AABBS_KHR,
    eInstances            = VK_GEOMETRY_TYPE_INSTANCES_KHR,
    eSpheresNV            = VK_GEOMETRY_TYPE_SPHERES_NV,
    eLinearSweptSpheresNV = VK_GEOMETRY_TYPE_LINEAR_SWEPT_SPHERES_NV,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eDenseGeometryFormatTrianglesAMDX = VK_GEOMETRY_TYPE_DENSE_GEOMETRY_FORMAT_TRIANGLES_AMDX
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
  };

  using GeometryTypeNV = GeometryTypeKHR;

  // wrapper class for enum VkAccelerationStructureCompatibilityKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureCompatibilityKHR.html
  enum class AccelerationStructureCompatibilityKHR
  {
    eCompatible   = VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR,
    eIncompatible = VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR
  };

  // wrapper class for enum VkAccelerationStructureCreateFlagBitsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureCreateFlagBitsKHR.html
  enum class AccelerationStructureCreateFlagBitsKHR : VkAccelerationStructureCreateFlagsKHR
  {
    eDeviceAddressCaptureReplay       = VK_ACCELERATION_STRUCTURE_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT_KHR,
    eDescriptorBufferCaptureReplayEXT = VK_ACCELERATION_STRUCTURE_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_EXT,
    eMotionNV                         = VK_ACCELERATION_STRUCTURE_CREATE_MOTION_BIT_NV
  };

  // wrapper using for bitmask VkAccelerationStructureCreateFlagsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureCreateFlagsKHR.html
  using AccelerationStructureCreateFlagsKHR = Flags<AccelerationStructureCreateFlagBitsKHR>;

  template <>
  struct FlagTraits<AccelerationStructureCreateFlagBitsKHR>
  {
    using WrappedType                                                                  = VkAccelerationStructureCreateFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR AccelerationStructureCreateFlagsKHR allFlags =
      AccelerationStructureCreateFlagBitsKHR::eDeviceAddressCaptureReplay | AccelerationStructureCreateFlagBitsKHR::eDescriptorBufferCaptureReplayEXT |
      AccelerationStructureCreateFlagBitsKHR::eMotionNV;
  };

  // wrapper class for enum VkBuildAccelerationStructureModeKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkBuildAccelerationStructureModeKHR.html
  enum class BuildAccelerationStructureModeKHR
  {
    eBuild  = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
    eUpdate = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
  };

  //=== VK_KHR_ray_tracing_pipeline ===

  // wrapper class for enum VkRayTracingShaderGroupTypeKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkRayTracingShaderGroupTypeKHR.html
  enum class RayTracingShaderGroupTypeKHR
  {
    eGeneral            = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
    eTrianglesHitGroup  = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
    eProceduralHitGroup = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR
  };

  using RayTracingShaderGroupTypeNV = RayTracingShaderGroupTypeKHR;

  // wrapper class for enum VkShaderGroupShaderKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkShaderGroupShaderKHR.html
  enum class ShaderGroupShaderKHR
  {
    eGeneral      = VK_SHADER_GROUP_SHADER_GENERAL_KHR,
    eClosestHit   = VK_SHADER_GROUP_SHADER_CLOSEST_HIT_KHR,
    eAnyHit       = VK_SHADER_GROUP_SHADER_ANY_HIT_KHR,
    eIntersection = VK_SHADER_GROUP_SHADER_INTERSECTION_KHR
  };

  //=== VK_NV_framebuffer_mixed_samples ===

  // wrapper class for enum VkCoverageModulationModeNV, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkCoverageModulationModeNV.html
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

  // wrapper using for bitmask VkPipelineCoverageModulationStateCreateFlagsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineCoverageModulationStateCreateFlagsNV.html
  using PipelineCoverageModulationStateCreateFlagsNV = Flags<PipelineCoverageModulationStateCreateFlagBitsNV>;

  template <>
  struct FlagTraits<PipelineCoverageModulationStateCreateFlagBitsNV>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                         isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineCoverageModulationStateCreateFlagsNV allFlags  = {};
  };

  //=== VK_EXT_validation_cache ===

  // wrapper class for enum VkValidationCacheHeaderVersionEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkValidationCacheHeaderVersionEXT.html
  enum class ValidationCacheHeaderVersionEXT
  {
    eOne = VK_VALIDATION_CACHE_HEADER_VERSION_ONE_EXT
  };

  enum class ValidationCacheCreateFlagBitsEXT : VkValidationCacheCreateFlagsEXT
  {
  };

  // wrapper using for bitmask VkValidationCacheCreateFlagsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkValidationCacheCreateFlagsEXT.html
  using ValidationCacheCreateFlagsEXT = Flags<ValidationCacheCreateFlagBitsEXT>;

  template <>
  struct FlagTraits<ValidationCacheCreateFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                          isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ValidationCacheCreateFlagsEXT allFlags  = {};
  };

  //=== VK_NV_shading_rate_image ===

  // wrapper class for enum VkShadingRatePaletteEntryNV, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkShadingRatePaletteEntryNV.html
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

  // wrapper class for enum VkCoarseSampleOrderTypeNV, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkCoarseSampleOrderTypeNV.html
  enum class CoarseSampleOrderTypeNV
  {
    eDefault     = VK_COARSE_SAMPLE_ORDER_TYPE_DEFAULT_NV,
    eCustom      = VK_COARSE_SAMPLE_ORDER_TYPE_CUSTOM_NV,
    ePixelMajor  = VK_COARSE_SAMPLE_ORDER_TYPE_PIXEL_MAJOR_NV,
    eSampleMajor = VK_COARSE_SAMPLE_ORDER_TYPE_SAMPLE_MAJOR_NV
  };

  //=== VK_NV_ray_tracing ===

  // wrapper class for enum VkAccelerationStructureMemoryRequirementsTypeNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureMemoryRequirementsTypeNV.html
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

  // wrapper using for bitmask VkPipelineCompilerControlFlagsAMD, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineCompilerControlFlagsAMD.html
  using PipelineCompilerControlFlagsAMD = Flags<PipelineCompilerControlFlagBitsAMD>;

  template <>
  struct FlagTraits<PipelineCompilerControlFlagBitsAMD>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                            isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineCompilerControlFlagsAMD allFlags  = {};
  };

  //=== VK_AMD_memory_overallocation_behavior ===

  // wrapper class for enum VkMemoryOverallocationBehaviorAMD, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryOverallocationBehaviorAMD.html
  enum class MemoryOverallocationBehaviorAMD
  {
    eDefault    = VK_MEMORY_OVERALLOCATION_BEHAVIOR_DEFAULT_AMD,
    eAllowed    = VK_MEMORY_OVERALLOCATION_BEHAVIOR_ALLOWED_AMD,
    eDisallowed = VK_MEMORY_OVERALLOCATION_BEHAVIOR_DISALLOWED_AMD
  };

  //=== VK_EXT_present_timing ===

  // wrapper class for enum VkPresentStageFlagBitsEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPresentStageFlagBitsEXT.html
  enum class PresentStageFlagBitsEXT : VkPresentStageFlagsEXT
  {
    eQueueOperationsEnd     = VK_PRESENT_STAGE_QUEUE_OPERATIONS_END_BIT_EXT,
    eRequestDequeued        = VK_PRESENT_STAGE_REQUEST_DEQUEUED_BIT_EXT,
    eImageFirstPixelOut     = VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT,
    eImageFirstPixelVisible = VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_VISIBLE_BIT_EXT
  };

  // wrapper using for bitmask VkPresentStageFlagsEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPresentStageFlagsEXT.html
  using PresentStageFlagsEXT = Flags<PresentStageFlagBitsEXT>;

  template <>
  struct FlagTraits<PresentStageFlagBitsEXT>
  {
    using WrappedType                                                   = VkPresentStageFlagBitsEXT;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PresentStageFlagsEXT allFlags =
      PresentStageFlagBitsEXT::eQueueOperationsEnd | PresentStageFlagBitsEXT::eRequestDequeued | PresentStageFlagBitsEXT::eImageFirstPixelOut |
      PresentStageFlagBitsEXT::eImageFirstPixelVisible;
  };

  // wrapper class for enum VkPresentTimingInfoFlagBitsEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPresentTimingInfoFlagBitsEXT.html
  enum class PresentTimingInfoFlagBitsEXT : VkPresentTimingInfoFlagsEXT
  {
    ePresentAtRelativeTime        = VK_PRESENT_TIMING_INFO_PRESENT_AT_RELATIVE_TIME_BIT_EXT,
    ePresentAtNearestRefreshCycle = VK_PRESENT_TIMING_INFO_PRESENT_AT_NEAREST_REFRESH_CYCLE_BIT_EXT
  };

  // wrapper using for bitmask VkPresentTimingInfoFlagsEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPresentTimingInfoFlagsEXT.html
  using PresentTimingInfoFlagsEXT = Flags<PresentTimingInfoFlagBitsEXT>;

  template <>
  struct FlagTraits<PresentTimingInfoFlagBitsEXT>
  {
    using WrappedType                                                        = VkPresentTimingInfoFlagBitsEXT;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                      isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PresentTimingInfoFlagsEXT allFlags =
      PresentTimingInfoFlagBitsEXT::ePresentAtRelativeTime | PresentTimingInfoFlagBitsEXT::ePresentAtNearestRefreshCycle;
  };

  // wrapper class for enum VkPastPresentationTimingFlagBitsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPastPresentationTimingFlagBitsEXT.html
  enum class PastPresentationTimingFlagBitsEXT : VkPastPresentationTimingFlagsEXT
  {
    eAllowPartialResults    = VK_PAST_PRESENTATION_TIMING_ALLOW_PARTIAL_RESULTS_BIT_EXT,
    eAllowOutOfOrderResults = VK_PAST_PRESENTATION_TIMING_ALLOW_OUT_OF_ORDER_RESULTS_BIT_EXT
  };

  // wrapper using for bitmask VkPastPresentationTimingFlagsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPastPresentationTimingFlagsEXT.html
  using PastPresentationTimingFlagsEXT = Flags<PastPresentationTimingFlagBitsEXT>;

  template <>
  struct FlagTraits<PastPresentationTimingFlagBitsEXT>
  {
    using WrappedType                                                             = VkPastPresentationTimingFlagBitsEXT;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PastPresentationTimingFlagsEXT allFlags =
      PastPresentationTimingFlagBitsEXT::eAllowPartialResults | PastPresentationTimingFlagBitsEXT::eAllowOutOfOrderResults;
  };

  //=== VK_INTEL_performance_query ===

  // wrapper class for enum VkPerformanceConfigurationTypeINTEL, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPerformanceConfigurationTypeINTEL.html
  enum class PerformanceConfigurationTypeINTEL
  {
    eCommandQueueMetricsDiscoveryActivated = VK_PERFORMANCE_CONFIGURATION_TYPE_COMMAND_QUEUE_METRICS_DISCOVERY_ACTIVATED_INTEL
  };

  // wrapper class for enum VkQueryPoolSamplingModeINTEL, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkQueryPoolSamplingModeINTEL.html
  enum class QueryPoolSamplingModeINTEL
  {
    eManual = VK_QUERY_POOL_SAMPLING_MODE_MANUAL_INTEL
  };

  // wrapper class for enum VkPerformanceOverrideTypeINTEL, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPerformanceOverrideTypeINTEL.html
  enum class PerformanceOverrideTypeINTEL
  {
    eNullHardware   = VK_PERFORMANCE_OVERRIDE_TYPE_NULL_HARDWARE_INTEL,
    eFlushGpuCaches = VK_PERFORMANCE_OVERRIDE_TYPE_FLUSH_GPU_CACHES_INTEL
  };

  // wrapper class for enum VkPerformanceParameterTypeINTEL, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPerformanceParameterTypeINTEL.html
  enum class PerformanceParameterTypeINTEL
  {
    eHwCountersSupported   = VK_PERFORMANCE_PARAMETER_TYPE_HW_COUNTERS_SUPPORTED_INTEL,
    eStreamMarkerValidBits = VK_PERFORMANCE_PARAMETER_TYPE_STREAM_MARKER_VALID_BITS_INTEL
  };

  // wrapper class for enum VkPerformanceValueTypeINTEL, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPerformanceValueTypeINTEL.html
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

  // wrapper using for bitmask VkImagePipeSurfaceCreateFlagsFUCHSIA, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkImagePipeSurfaceCreateFlagsFUCHSIA.html
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

  // wrapper using for bitmask VkMetalSurfaceCreateFlagsEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkMetalSurfaceCreateFlagsEXT.html
  using MetalSurfaceCreateFlagsEXT = Flags<MetalSurfaceCreateFlagBitsEXT>;

  template <>
  struct FlagTraits<MetalSurfaceCreateFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR MetalSurfaceCreateFlagsEXT allFlags  = {};
  };
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_KHR_fragment_shading_rate ===

  // wrapper class for enum VkFragmentShadingRateCombinerOpKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkFragmentShadingRateCombinerOpKHR.html
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

  // wrapper using for bitmask VkShaderCorePropertiesFlagsAMD, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkShaderCorePropertiesFlagsAMD.html
  using ShaderCorePropertiesFlagsAMD = Flags<ShaderCorePropertiesFlagBitsAMD>;

  template <>
  struct FlagTraits<ShaderCorePropertiesFlagBitsAMD>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                         isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ShaderCorePropertiesFlagsAMD allFlags  = {};
  };

  //=== VK_EXT_validation_features ===

  // wrapper class for enum VkValidationFeatureEnableEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkValidationFeatureEnableEXT.html
  enum class ValidationFeatureEnableEXT
  {
    eGpuAssisted                   = VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT,
    eGpuAssistedReserveBindingSlot = VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT,
    eBestPractices                 = VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT,
    eDebugPrintf                   = VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT,
    eSynchronizationValidation     = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT
  };

  // wrapper class for enum VkValidationFeatureDisableEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkValidationFeatureDisableEXT.html
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

  // wrapper class for enum VkCoverageReductionModeNV, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkCoverageReductionModeNV.html
  enum class CoverageReductionModeNV
  {
    eMerge    = VK_COVERAGE_REDUCTION_MODE_MERGE_NV,
    eTruncate = VK_COVERAGE_REDUCTION_MODE_TRUNCATE_NV
  };

  enum class PipelineCoverageReductionStateCreateFlagBitsNV : VkPipelineCoverageReductionStateCreateFlagsNV
  {
  };

  // wrapper using for bitmask VkPipelineCoverageReductionStateCreateFlagsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineCoverageReductionStateCreateFlagsNV.html
  using PipelineCoverageReductionStateCreateFlagsNV = Flags<PipelineCoverageReductionStateCreateFlagBitsNV>;

  template <>
  struct FlagTraits<PipelineCoverageReductionStateCreateFlagBitsNV>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                        isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PipelineCoverageReductionStateCreateFlagsNV allFlags  = {};
  };

  //=== VK_EXT_provoking_vertex ===

  // wrapper class for enum VkProvokingVertexModeEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkProvokingVertexModeEXT.html
  enum class ProvokingVertexModeEXT
  {
    eFirstVertex = VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT,
    eLastVertex  = VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT
  };

#if defined( VK_USE_PLATFORM_WIN32_KHR )
  //=== VK_EXT_full_screen_exclusive ===

  // wrapper class for enum VkFullScreenExclusiveEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkFullScreenExclusiveEXT.html
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

  // wrapper using for bitmask VkHeadlessSurfaceCreateFlagsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkHeadlessSurfaceCreateFlagsEXT.html
  using HeadlessSurfaceCreateFlagsEXT = Flags<HeadlessSurfaceCreateFlagBitsEXT>;

  template <>
  struct FlagTraits<HeadlessSurfaceCreateFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                          isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR HeadlessSurfaceCreateFlagsEXT allFlags  = {};
  };

  //=== VK_KHR_pipeline_executable_properties ===

  // wrapper class for enum VkPipelineExecutableStatisticFormatKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipelineExecutableStatisticFormatKHR.html
  enum class PipelineExecutableStatisticFormatKHR
  {
    eBool32  = VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR,
    eInt64   = VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR,
    eUint64  = VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR,
    eFloat64 = VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR
  };

  //=== VK_NV_device_generated_commands ===

  // wrapper class for enum VkIndirectStateFlagBitsNV, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkIndirectStateFlagBitsNV.html
  enum class IndirectStateFlagBitsNV : VkIndirectStateFlagsNV
  {
    eFlagFrontface = VK_INDIRECT_STATE_FLAG_FRONTFACE_BIT_NV
  };

  // wrapper using for bitmask VkIndirectStateFlagsNV, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkIndirectStateFlagsNV.html
  using IndirectStateFlagsNV = Flags<IndirectStateFlagBitsNV>;

  template <>
  struct FlagTraits<IndirectStateFlagBitsNV>
  {
    using WrappedType                                                   = VkIndirectStateFlagBitsNV;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR IndirectStateFlagsNV allFlags  = IndirectStateFlagBitsNV::eFlagFrontface;
  };

  // wrapper class for enum VkIndirectCommandsTokenTypeNV, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkIndirectCommandsTokenTypeNV.html
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

  // wrapper class for enum VkIndirectCommandsLayoutUsageFlagBitsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkIndirectCommandsLayoutUsageFlagBitsNV.html
  enum class IndirectCommandsLayoutUsageFlagBitsNV : VkIndirectCommandsLayoutUsageFlagsNV
  {
    eExplicitPreprocess = VK_INDIRECT_COMMANDS_LAYOUT_USAGE_EXPLICIT_PREPROCESS_BIT_NV,
    eIndexedSequences   = VK_INDIRECT_COMMANDS_LAYOUT_USAGE_INDEXED_SEQUENCES_BIT_NV,
    eUnorderedSequences = VK_INDIRECT_COMMANDS_LAYOUT_USAGE_UNORDERED_SEQUENCES_BIT_NV
  };

  // wrapper using for bitmask VkIndirectCommandsLayoutUsageFlagsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkIndirectCommandsLayoutUsageFlagsNV.html
  using IndirectCommandsLayoutUsageFlagsNV = Flags<IndirectCommandsLayoutUsageFlagBitsNV>;

  template <>
  struct FlagTraits<IndirectCommandsLayoutUsageFlagBitsNV>
  {
    using WrappedType                                                                 = VkIndirectCommandsLayoutUsageFlagBitsNV;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                               isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR IndirectCommandsLayoutUsageFlagsNV allFlags  = IndirectCommandsLayoutUsageFlagBitsNV::eExplicitPreprocess |
                                                                                       IndirectCommandsLayoutUsageFlagBitsNV::eIndexedSequences |
                                                                                       IndirectCommandsLayoutUsageFlagBitsNV::eUnorderedSequences;
  };

  //=== VK_EXT_depth_bias_control ===

  // wrapper class for enum VkDepthBiasRepresentationEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDepthBiasRepresentationEXT.html
  enum class DepthBiasRepresentationEXT
  {
    eLeastRepresentableValueFormat     = VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORMAT_EXT,
    eLeastRepresentableValueForceUnorm = VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT,
    eFloat                             = VK_DEPTH_BIAS_REPRESENTATION_FLOAT_EXT
  };

  //=== VK_EXT_device_memory_report ===

  // wrapper class for enum VkDeviceMemoryReportEventTypeEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceMemoryReportEventTypeEXT.html
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

  // wrapper using for bitmask VkDeviceMemoryReportFlagsEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceMemoryReportFlagsEXT.html
  using DeviceMemoryReportFlagsEXT = Flags<DeviceMemoryReportFlagBitsEXT>;

  template <>
  struct FlagTraits<DeviceMemoryReportFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DeviceMemoryReportFlagsEXT allFlags  = {};
  };

  //=== VK_KHR_video_encode_queue ===

  // wrapper class for enum VkVideoEncodeCapabilityFlagBitsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeCapabilityFlagBitsKHR.html
  enum class VideoEncodeCapabilityFlagBitsKHR : VkVideoEncodeCapabilityFlagsKHR
  {
    ePrecedingExternallyEncodedBytes           = VK_VIDEO_ENCODE_CAPABILITY_PRECEDING_EXTERNALLY_ENCODED_BYTES_BIT_KHR,
    eInsufficientBitstreamBufferRangeDetection = VK_VIDEO_ENCODE_CAPABILITY_INSUFFICIENT_BITSTREAM_BUFFER_RANGE_DETECTION_BIT_KHR,
    eQuantizationDeltaMap                      = VK_VIDEO_ENCODE_CAPABILITY_QUANTIZATION_DELTA_MAP_BIT_KHR,
    eEmphasisMap                               = VK_VIDEO_ENCODE_CAPABILITY_EMPHASIS_MAP_BIT_KHR
  };

  // wrapper using for bitmask VkVideoEncodeCapabilityFlagsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeCapabilityFlagsKHR.html
  using VideoEncodeCapabilityFlagsKHR = Flags<VideoEncodeCapabilityFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeCapabilityFlagBitsKHR>
  {
    using WrappedType                                                            = VkVideoEncodeCapabilityFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                          isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeCapabilityFlagsKHR allFlags =
      VideoEncodeCapabilityFlagBitsKHR::ePrecedingExternallyEncodedBytes | VideoEncodeCapabilityFlagBitsKHR::eInsufficientBitstreamBufferRangeDetection |
      VideoEncodeCapabilityFlagBitsKHR::eQuantizationDeltaMap | VideoEncodeCapabilityFlagBitsKHR::eEmphasisMap;
  };

  // wrapper class for enum VkVideoEncodeFeedbackFlagBitsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeFeedbackFlagBitsKHR.html
  enum class VideoEncodeFeedbackFlagBitsKHR : VkVideoEncodeFeedbackFlagsKHR
  {
    eBitstreamBufferOffset = VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BUFFER_OFFSET_BIT_KHR,
    eBitstreamBytesWritten = VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BYTES_WRITTEN_BIT_KHR,
    eBitstreamHasOverrides = VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_HAS_OVERRIDES_BIT_KHR
  };

  // wrapper using for bitmask VkVideoEncodeFeedbackFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeFeedbackFlagsKHR.html
  using VideoEncodeFeedbackFlagsKHR = Flags<VideoEncodeFeedbackFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeFeedbackFlagBitsKHR>
  {
    using WrappedType                                                          = VkVideoEncodeFeedbackFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                        isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeFeedbackFlagsKHR allFlags  = VideoEncodeFeedbackFlagBitsKHR::eBitstreamBufferOffset |
                                                                                VideoEncodeFeedbackFlagBitsKHR::eBitstreamBytesWritten |
                                                                                VideoEncodeFeedbackFlagBitsKHR::eBitstreamHasOverrides;
  };

  // wrapper class for enum VkVideoEncodeUsageFlagBitsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeUsageFlagBitsKHR.html
  enum class VideoEncodeUsageFlagBitsKHR : VkVideoEncodeUsageFlagsKHR
  {
    eDefault      = VK_VIDEO_ENCODE_USAGE_DEFAULT_KHR,
    eTranscoding  = VK_VIDEO_ENCODE_USAGE_TRANSCODING_BIT_KHR,
    eStreaming    = VK_VIDEO_ENCODE_USAGE_STREAMING_BIT_KHR,
    eRecording    = VK_VIDEO_ENCODE_USAGE_RECORDING_BIT_KHR,
    eConferencing = VK_VIDEO_ENCODE_USAGE_CONFERENCING_BIT_KHR
  };

  // wrapper using for bitmask VkVideoEncodeUsageFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeUsageFlagsKHR.html
  using VideoEncodeUsageFlagsKHR = Flags<VideoEncodeUsageFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeUsageFlagBitsKHR>
  {
    using WrappedType                                                       = VkVideoEncodeUsageFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                     isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeUsageFlagsKHR allFlags = VideoEncodeUsageFlagBitsKHR::eDefault | VideoEncodeUsageFlagBitsKHR::eTranscoding |
                                                                             VideoEncodeUsageFlagBitsKHR::eStreaming | VideoEncodeUsageFlagBitsKHR::eRecording |
                                                                             VideoEncodeUsageFlagBitsKHR::eConferencing;
  };

  // wrapper class for enum VkVideoEncodeContentFlagBitsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeContentFlagBitsKHR.html
  enum class VideoEncodeContentFlagBitsKHR : VkVideoEncodeContentFlagsKHR
  {
    eDefault  = VK_VIDEO_ENCODE_CONTENT_DEFAULT_KHR,
    eCamera   = VK_VIDEO_ENCODE_CONTENT_CAMERA_BIT_KHR,
    eDesktop  = VK_VIDEO_ENCODE_CONTENT_DESKTOP_BIT_KHR,
    eRendered = VK_VIDEO_ENCODE_CONTENT_RENDERED_BIT_KHR
  };

  // wrapper using for bitmask VkVideoEncodeContentFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeContentFlagsKHR.html
  using VideoEncodeContentFlagsKHR = Flags<VideoEncodeContentFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeContentFlagBitsKHR>
  {
    using WrappedType                                                         = VkVideoEncodeContentFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeContentFlagsKHR allFlags =
      VideoEncodeContentFlagBitsKHR::eDefault | VideoEncodeContentFlagBitsKHR::eCamera | VideoEncodeContentFlagBitsKHR::eDesktop |
      VideoEncodeContentFlagBitsKHR::eRendered;
  };

  // wrapper class for enum VkVideoEncodeTuningModeKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeTuningModeKHR.html
  enum class VideoEncodeTuningModeKHR
  {
    eDefault         = VK_VIDEO_ENCODE_TUNING_MODE_DEFAULT_KHR,
    eHighQuality     = VK_VIDEO_ENCODE_TUNING_MODE_HIGH_QUALITY_KHR,
    eLowLatency      = VK_VIDEO_ENCODE_TUNING_MODE_LOW_LATENCY_KHR,
    eUltraLowLatency = VK_VIDEO_ENCODE_TUNING_MODE_ULTRA_LOW_LATENCY_KHR,
    eLossless        = VK_VIDEO_ENCODE_TUNING_MODE_LOSSLESS_KHR
  };

  // wrapper class for enum VkVideoEncodeRateControlModeFlagBitsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeRateControlModeFlagBitsKHR.html
  enum class VideoEncodeRateControlModeFlagBitsKHR : VkVideoEncodeRateControlModeFlagsKHR
  {
    eDefault  = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DEFAULT_KHR,
    eDisabled = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR,
    eCbr      = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR,
    eVbr      = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_VBR_BIT_KHR
  };

  // wrapper using for bitmask VkVideoEncodeRateControlModeFlagsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeRateControlModeFlagsKHR.html
  using VideoEncodeRateControlModeFlagsKHR = Flags<VideoEncodeRateControlModeFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeRateControlModeFlagBitsKHR>
  {
    using WrappedType                                                                 = VkVideoEncodeRateControlModeFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                               isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeRateControlModeFlagsKHR allFlags =
      VideoEncodeRateControlModeFlagBitsKHR::eDefault | VideoEncodeRateControlModeFlagBitsKHR::eDisabled | VideoEncodeRateControlModeFlagBitsKHR::eCbr |
      VideoEncodeRateControlModeFlagBitsKHR::eVbr;
  };

  // wrapper class for enum VkVideoEncodeFlagBitsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeFlagBitsKHR.html
  enum class VideoEncodeFlagBitsKHR : VkVideoEncodeFlagsKHR
  {
    eIntraRefresh             = VK_VIDEO_ENCODE_INTRA_REFRESH_BIT_KHR,
    eWithQuantizationDeltaMap = VK_VIDEO_ENCODE_WITH_QUANTIZATION_DELTA_MAP_BIT_KHR,
    eWithEmphasisMap          = VK_VIDEO_ENCODE_WITH_EMPHASIS_MAP_BIT_KHR
  };

  // wrapper using for bitmask VkVideoEncodeFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeFlagsKHR.html
  using VideoEncodeFlagsKHR = Flags<VideoEncodeFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeFlagBitsKHR>
  {
    using WrappedType                                                  = VkVideoEncodeFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeFlagsKHR allFlags =
      VideoEncodeFlagBitsKHR::eIntraRefresh | VideoEncodeFlagBitsKHR::eWithQuantizationDeltaMap | VideoEncodeFlagBitsKHR::eWithEmphasisMap;
  };

  enum class VideoEncodeRateControlFlagBitsKHR : VkVideoEncodeRateControlFlagsKHR
  {
  };

  // wrapper using for bitmask VkVideoEncodeRateControlFlagsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeRateControlFlagsKHR.html
  using VideoEncodeRateControlFlagsKHR = Flags<VideoEncodeRateControlFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeRateControlFlagBitsKHR>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeRateControlFlagsKHR allFlags  = {};
  };

  //=== VK_NV_device_diagnostics_config ===

  // wrapper class for enum VkDeviceDiagnosticsConfigFlagBitsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceDiagnosticsConfigFlagBitsNV.html
  enum class DeviceDiagnosticsConfigFlagBitsNV : VkDeviceDiagnosticsConfigFlagsNV
  {
    eEnableShaderDebugInfo      = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_DEBUG_INFO_BIT_NV,
    eEnableResourceTracking     = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_RESOURCE_TRACKING_BIT_NV,
    eEnableAutomaticCheckpoints = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV,
    eEnableShaderErrorReporting = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_ERROR_REPORTING_BIT_NV
  };

  // wrapper using for bitmask VkDeviceDiagnosticsConfigFlagsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceDiagnosticsConfigFlagsNV.html
  using DeviceDiagnosticsConfigFlagsNV = Flags<DeviceDiagnosticsConfigFlagBitsNV>;

  template <>
  struct FlagTraits<DeviceDiagnosticsConfigFlagBitsNV>
  {
    using WrappedType                                                             = VkDeviceDiagnosticsConfigFlagBitsNV;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DeviceDiagnosticsConfigFlagsNV allFlags =
      DeviceDiagnosticsConfigFlagBitsNV::eEnableShaderDebugInfo | DeviceDiagnosticsConfigFlagBitsNV::eEnableResourceTracking |
      DeviceDiagnosticsConfigFlagBitsNV::eEnableAutomaticCheckpoints | DeviceDiagnosticsConfigFlagBitsNV::eEnableShaderErrorReporting;
  };

  //=== VK_QCOM_tile_shading ===

  // wrapper class for enum VkTileShadingRenderPassFlagBitsQCOM, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkTileShadingRenderPassFlagBitsQCOM.html
  enum class TileShadingRenderPassFlagBitsQCOM : VkTileShadingRenderPassFlagsQCOM
  {
    eEnable           = VK_TILE_SHADING_RENDER_PASS_ENABLE_BIT_QCOM,
    ePerTileExecution = VK_TILE_SHADING_RENDER_PASS_PER_TILE_EXECUTION_BIT_QCOM
  };

  // wrapper using for bitmask VkTileShadingRenderPassFlagsQCOM, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkTileShadingRenderPassFlagsQCOM.html
  using TileShadingRenderPassFlagsQCOM = Flags<TileShadingRenderPassFlagBitsQCOM>;

  template <>
  struct FlagTraits<TileShadingRenderPassFlagBitsQCOM>
  {
    using WrappedType                                                             = VkTileShadingRenderPassFlagBitsQCOM;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR TileShadingRenderPassFlagsQCOM allFlags =
      TileShadingRenderPassFlagBitsQCOM::eEnable | TileShadingRenderPassFlagBitsQCOM::ePerTileExecution;
  };

#if defined( VK_USE_PLATFORM_METAL_EXT )
  //=== VK_EXT_metal_objects ===

  // wrapper class for enum VkExportMetalObjectTypeFlagBitsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkExportMetalObjectTypeFlagBitsEXT.html
  enum class ExportMetalObjectTypeFlagBitsEXT : VkExportMetalObjectTypeFlagsEXT
  {
    eMetalDevice       = VK_EXPORT_METAL_OBJECT_TYPE_METAL_DEVICE_BIT_EXT,
    eMetalCommandQueue = VK_EXPORT_METAL_OBJECT_TYPE_METAL_COMMAND_QUEUE_BIT_EXT,
    eMetalBuffer       = VK_EXPORT_METAL_OBJECT_TYPE_METAL_BUFFER_BIT_EXT,
    eMetalTexture      = VK_EXPORT_METAL_OBJECT_TYPE_METAL_TEXTURE_BIT_EXT,
    eMetalIosurface    = VK_EXPORT_METAL_OBJECT_TYPE_METAL_IOSURFACE_BIT_EXT,
    eMetalSharedEvent  = VK_EXPORT_METAL_OBJECT_TYPE_METAL_SHARED_EVENT_BIT_EXT
  };

  // wrapper using for bitmask VkExportMetalObjectTypeFlagsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkExportMetalObjectTypeFlagsEXT.html
  using ExportMetalObjectTypeFlagsEXT = Flags<ExportMetalObjectTypeFlagBitsEXT>;

  template <>
  struct FlagTraits<ExportMetalObjectTypeFlagBitsEXT>
  {
    using WrappedType                                                            = VkExportMetalObjectTypeFlagBitsEXT;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                          isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ExportMetalObjectTypeFlagsEXT allFlags =
      ExportMetalObjectTypeFlagBitsEXT::eMetalDevice | ExportMetalObjectTypeFlagBitsEXT::eMetalCommandQueue | ExportMetalObjectTypeFlagBitsEXT::eMetalBuffer |
      ExportMetalObjectTypeFlagBitsEXT::eMetalTexture | ExportMetalObjectTypeFlagBitsEXT::eMetalIosurface | ExportMetalObjectTypeFlagBitsEXT::eMetalSharedEvent;
  };
#endif /*VK_USE_PLATFORM_METAL_EXT*/

  //=== VK_EXT_graphics_pipeline_library ===

  // wrapper class for enum VkGraphicsPipelineLibraryFlagBitsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkGraphicsPipelineLibraryFlagBitsEXT.html
  enum class GraphicsPipelineLibraryFlagBitsEXT : VkGraphicsPipelineLibraryFlagsEXT
  {
    eVertexInputInterface    = VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT,
    ePreRasterizationShaders = VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT,
    eFragmentShader          = VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT,
    eFragmentOutputInterface = VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT
  };

  // wrapper using for bitmask VkGraphicsPipelineLibraryFlagsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkGraphicsPipelineLibraryFlagsEXT.html
  using GraphicsPipelineLibraryFlagsEXT = Flags<GraphicsPipelineLibraryFlagBitsEXT>;

  template <>
  struct FlagTraits<GraphicsPipelineLibraryFlagBitsEXT>
  {
    using WrappedType                                                              = VkGraphicsPipelineLibraryFlagBitsEXT;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                            isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR GraphicsPipelineLibraryFlagsEXT allFlags =
      GraphicsPipelineLibraryFlagBitsEXT::eVertexInputInterface | GraphicsPipelineLibraryFlagBitsEXT::ePreRasterizationShaders |
      GraphicsPipelineLibraryFlagBitsEXT::eFragmentShader | GraphicsPipelineLibraryFlagBitsEXT::eFragmentOutputInterface;
  };

  //=== VK_NV_fragment_shading_rate_enums ===

  // wrapper class for enum VkFragmentShadingRateNV, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkFragmentShadingRateNV.html
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

  // wrapper class for enum VkFragmentShadingRateTypeNV, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkFragmentShadingRateTypeNV.html
  enum class FragmentShadingRateTypeNV
  {
    eFragmentSize = VK_FRAGMENT_SHADING_RATE_TYPE_FRAGMENT_SIZE_NV,
    eEnums        = VK_FRAGMENT_SHADING_RATE_TYPE_ENUMS_NV
  };

  //=== VK_NV_ray_tracing_motion_blur ===

  // wrapper class for enum VkAccelerationStructureMotionInstanceTypeNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureMotionInstanceTypeNV.html
  enum class AccelerationStructureMotionInstanceTypeNV
  {
    eStatic       = VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_STATIC_NV,
    eMatrixMotion = VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_MATRIX_MOTION_NV,
    eSrtMotion    = VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_SRT_MOTION_NV
  };

  enum class AccelerationStructureMotionInfoFlagBitsNV : VkAccelerationStructureMotionInfoFlagsNV
  {
  };

  // wrapper using for bitmask VkAccelerationStructureMotionInfoFlagsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureMotionInfoFlagsNV.html
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

  // wrapper using for bitmask VkAccelerationStructureMotionInstanceFlagsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccelerationStructureMotionInstanceFlagsNV.html
  using AccelerationStructureMotionInstanceFlagsNV = Flags<AccelerationStructureMotionInstanceFlagBitsNV>;

  template <>
  struct FlagTraits<AccelerationStructureMotionInstanceFlagBitsNV>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR AccelerationStructureMotionInstanceFlagsNV allFlags  = {};
  };

  //=== VK_EXT_image_compression_control ===

  // wrapper class for enum VkImageCompressionFlagBitsEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageCompressionFlagBitsEXT.html
  enum class ImageCompressionFlagBitsEXT : VkImageCompressionFlagsEXT
  {
    eDefault           = VK_IMAGE_COMPRESSION_DEFAULT_EXT,
    eFixedRateDefault  = VK_IMAGE_COMPRESSION_FIXED_RATE_DEFAULT_EXT,
    eFixedRateExplicit = VK_IMAGE_COMPRESSION_FIXED_RATE_EXPLICIT_EXT,
    eDisabled          = VK_IMAGE_COMPRESSION_DISABLED_EXT
  };

  // wrapper using for bitmask VkImageCompressionFlagsEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageCompressionFlagsEXT.html
  using ImageCompressionFlagsEXT = Flags<ImageCompressionFlagBitsEXT>;

  template <>
  struct FlagTraits<ImageCompressionFlagBitsEXT>
  {
    using WrappedType                                                       = VkImageCompressionFlagBitsEXT;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                     isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ImageCompressionFlagsEXT allFlags =
      ImageCompressionFlagBitsEXT::eDefault | ImageCompressionFlagBitsEXT::eFixedRateDefault | ImageCompressionFlagBitsEXT::eFixedRateExplicit |
      ImageCompressionFlagBitsEXT::eDisabled;
  };

  // wrapper class for enum VkImageCompressionFixedRateFlagBitsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageCompressionFixedRateFlagBitsEXT.html
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

  // wrapper using for bitmask VkImageCompressionFixedRateFlagsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageCompressionFixedRateFlagsEXT.html
  using ImageCompressionFixedRateFlagsEXT = Flags<ImageCompressionFixedRateFlagBitsEXT>;

  template <>
  struct FlagTraits<ImageCompressionFixedRateFlagBitsEXT>
  {
    using WrappedType                                                                = VkImageCompressionFixedRateFlagBitsEXT;
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

  // wrapper class for enum VkDeviceFaultAddressTypeEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceFaultAddressTypeEXT.html
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

  // wrapper class for enum VkDeviceFaultVendorBinaryHeaderVersionEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceFaultVendorBinaryHeaderVersionEXT.html
  enum class DeviceFaultVendorBinaryHeaderVersionEXT
  {
    eOne = VK_DEVICE_FAULT_VENDOR_BINARY_HEADER_VERSION_ONE_EXT
  };

#if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
  //=== VK_EXT_directfb_surface ===

  enum class DirectFBSurfaceCreateFlagBitsEXT : VkDirectFBSurfaceCreateFlagsEXT
  {
  };

  // wrapper using for bitmask VkDirectFBSurfaceCreateFlagsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkDirectFBSurfaceCreateFlagsEXT.html
  using DirectFBSurfaceCreateFlagsEXT = Flags<DirectFBSurfaceCreateFlagBitsEXT>;

  template <>
  struct FlagTraits<DirectFBSurfaceCreateFlagBitsEXT>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                          isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DirectFBSurfaceCreateFlagsEXT allFlags  = {};
  };
#endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

  //=== VK_EXT_device_address_binding_report ===

  // wrapper class for enum VkDeviceAddressBindingFlagBitsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceAddressBindingFlagBitsEXT.html
  enum class DeviceAddressBindingFlagBitsEXT : VkDeviceAddressBindingFlagsEXT
  {
    eInternalObject = VK_DEVICE_ADDRESS_BINDING_INTERNAL_OBJECT_BIT_EXT
  };

  // wrapper using for bitmask VkDeviceAddressBindingFlagsEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceAddressBindingFlagsEXT.html
  using DeviceAddressBindingFlagsEXT = Flags<DeviceAddressBindingFlagBitsEXT>;

  template <>
  struct FlagTraits<DeviceAddressBindingFlagBitsEXT>
  {
    using WrappedType                                                           = VkDeviceAddressBindingFlagBitsEXT;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                         isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DeviceAddressBindingFlagsEXT allFlags  = DeviceAddressBindingFlagBitsEXT::eInternalObject;
  };

  // wrapper class for enum VkDeviceAddressBindingTypeEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDeviceAddressBindingTypeEXT.html
  enum class DeviceAddressBindingTypeEXT
  {
    eBind   = VK_DEVICE_ADDRESS_BINDING_TYPE_BIND_EXT,
    eUnbind = VK_DEVICE_ADDRESS_BINDING_TYPE_UNBIND_EXT
  };

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_buffer_collection ===

  // wrapper class for enum VkImageConstraintsInfoFlagBitsFUCHSIA, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageConstraintsInfoFlagBitsFUCHSIA.html
  enum class ImageConstraintsInfoFlagBitsFUCHSIA : VkImageConstraintsInfoFlagsFUCHSIA
  {
    eCpuReadRarely     = VK_IMAGE_CONSTRAINTS_INFO_CPU_READ_RARELY_FUCHSIA,
    eCpuReadOften      = VK_IMAGE_CONSTRAINTS_INFO_CPU_READ_OFTEN_FUCHSIA,
    eCpuWriteRarely    = VK_IMAGE_CONSTRAINTS_INFO_CPU_WRITE_RARELY_FUCHSIA,
    eCpuWriteOften     = VK_IMAGE_CONSTRAINTS_INFO_CPU_WRITE_OFTEN_FUCHSIA,
    eProtectedOptional = VK_IMAGE_CONSTRAINTS_INFO_PROTECTED_OPTIONAL_FUCHSIA
  };

  // wrapper using for bitmask VkImageConstraintsInfoFlagsFUCHSIA, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageConstraintsInfoFlagsFUCHSIA.html
  using ImageConstraintsInfoFlagsFUCHSIA = Flags<ImageConstraintsInfoFlagBitsFUCHSIA>;

  template <>
  struct FlagTraits<ImageConstraintsInfoFlagBitsFUCHSIA>
  {
    using WrappedType                                                               = VkImageConstraintsInfoFlagBitsFUCHSIA;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ImageConstraintsInfoFlagsFUCHSIA allFlags =
      ImageConstraintsInfoFlagBitsFUCHSIA::eCpuReadRarely | ImageConstraintsInfoFlagBitsFUCHSIA::eCpuReadOften |
      ImageConstraintsInfoFlagBitsFUCHSIA::eCpuWriteRarely | ImageConstraintsInfoFlagBitsFUCHSIA::eCpuWriteOften |
      ImageConstraintsInfoFlagBitsFUCHSIA::eProtectedOptional;
  };

  enum class ImageFormatConstraintsFlagBitsFUCHSIA : VkImageFormatConstraintsFlagsFUCHSIA
  {
  };

  // wrapper using for bitmask VkImageFormatConstraintsFlagsFUCHSIA, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkImageFormatConstraintsFlagsFUCHSIA.html
  using ImageFormatConstraintsFlagsFUCHSIA = Flags<ImageFormatConstraintsFlagBitsFUCHSIA>;

  template <>
  struct FlagTraits<ImageFormatConstraintsFlagBitsFUCHSIA>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                               isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ImageFormatConstraintsFlagsFUCHSIA allFlags  = {};
  };
#endif /*VK_USE_PLATFORM_FUCHSIA*/

  //=== VK_EXT_frame_boundary ===

  // wrapper class for enum VkFrameBoundaryFlagBitsEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkFrameBoundaryFlagBitsEXT.html
  enum class FrameBoundaryFlagBitsEXT : VkFrameBoundaryFlagsEXT
  {
    eFrameEnd = VK_FRAME_BOUNDARY_FRAME_END_BIT_EXT
  };

  // wrapper using for bitmask VkFrameBoundaryFlagsEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkFrameBoundaryFlagsEXT.html
  using FrameBoundaryFlagsEXT = Flags<FrameBoundaryFlagBitsEXT>;

  template <>
  struct FlagTraits<FrameBoundaryFlagBitsEXT>
  {
    using WrappedType                                                    = VkFrameBoundaryFlagBitsEXT;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                  isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR FrameBoundaryFlagsEXT allFlags  = FrameBoundaryFlagBitsEXT::eFrameEnd;
  };

#if defined( VK_USE_PLATFORM_SCREEN_QNX )
  //=== VK_QNX_screen_surface ===

  enum class ScreenSurfaceCreateFlagBitsQNX : VkScreenSurfaceCreateFlagsQNX
  {
  };

  // wrapper using for bitmask VkScreenSurfaceCreateFlagsQNX, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkScreenSurfaceCreateFlagsQNX.html
  using ScreenSurfaceCreateFlagsQNX = Flags<ScreenSurfaceCreateFlagBitsQNX>;

  template <>
  struct FlagTraits<ScreenSurfaceCreateFlagBitsQNX>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                        isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ScreenSurfaceCreateFlagsQNX allFlags  = {};
  };
#endif /*VK_USE_PLATFORM_SCREEN_QNX*/

  //=== VK_VALVE_video_encode_rgb_conversion ===

  // wrapper class for enum VkVideoEncodeRgbModelConversionFlagBitsVALVE, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeRgbModelConversionFlagBitsVALVE.html
  enum class VideoEncodeRgbModelConversionFlagBitsVALVE : VkVideoEncodeRgbModelConversionFlagsVALVE
  {
    eRgbIdentity   = VK_VIDEO_ENCODE_RGB_MODEL_CONVERSION_RGB_IDENTITY_BIT_VALVE,
    eYcbcrIdentity = VK_VIDEO_ENCODE_RGB_MODEL_CONVERSION_YCBCR_IDENTITY_BIT_VALVE,
    eYcbcr709      = VK_VIDEO_ENCODE_RGB_MODEL_CONVERSION_YCBCR_709_BIT_VALVE,
    eYcbcr601      = VK_VIDEO_ENCODE_RGB_MODEL_CONVERSION_YCBCR_601_BIT_VALVE,
    eYcbcr2020     = VK_VIDEO_ENCODE_RGB_MODEL_CONVERSION_YCBCR_2020_BIT_VALVE
  };

  // wrapper using for bitmask VkVideoEncodeRgbModelConversionFlagsVALVE, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeRgbModelConversionFlagsVALVE.html
  using VideoEncodeRgbModelConversionFlagsVALVE = Flags<VideoEncodeRgbModelConversionFlagBitsVALVE>;

  template <>
  struct FlagTraits<VideoEncodeRgbModelConversionFlagBitsVALVE>
  {
    using WrappedType                                                                      = VkVideoEncodeRgbModelConversionFlagBitsVALVE;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                    isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeRgbModelConversionFlagsVALVE allFlags =
      VideoEncodeRgbModelConversionFlagBitsVALVE::eRgbIdentity | VideoEncodeRgbModelConversionFlagBitsVALVE::eYcbcrIdentity |
      VideoEncodeRgbModelConversionFlagBitsVALVE::eYcbcr709 | VideoEncodeRgbModelConversionFlagBitsVALVE::eYcbcr601 |
      VideoEncodeRgbModelConversionFlagBitsVALVE::eYcbcr2020;
  };

  // wrapper class for enum VkVideoEncodeRgbRangeCompressionFlagBitsVALVE, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeRgbRangeCompressionFlagBitsVALVE.html
  enum class VideoEncodeRgbRangeCompressionFlagBitsVALVE : VkVideoEncodeRgbRangeCompressionFlagsVALVE
  {
    eFullRange   = VK_VIDEO_ENCODE_RGB_RANGE_COMPRESSION_FULL_RANGE_BIT_VALVE,
    eNarrowRange = VK_VIDEO_ENCODE_RGB_RANGE_COMPRESSION_NARROW_RANGE_BIT_VALVE
  };

  // wrapper using for bitmask VkVideoEncodeRgbRangeCompressionFlagsVALVE, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeRgbRangeCompressionFlagsVALVE.html
  using VideoEncodeRgbRangeCompressionFlagsVALVE = Flags<VideoEncodeRgbRangeCompressionFlagBitsVALVE>;

  template <>
  struct FlagTraits<VideoEncodeRgbRangeCompressionFlagBitsVALVE>
  {
    using WrappedType                                                                       = VkVideoEncodeRgbRangeCompressionFlagBitsVALVE;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                     isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeRgbRangeCompressionFlagsVALVE allFlags =
      VideoEncodeRgbRangeCompressionFlagBitsVALVE::eFullRange | VideoEncodeRgbRangeCompressionFlagBitsVALVE::eNarrowRange;
  };

  // wrapper class for enum VkVideoEncodeRgbChromaOffsetFlagBitsVALVE, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeRgbChromaOffsetFlagBitsVALVE.html
  enum class VideoEncodeRgbChromaOffsetFlagBitsVALVE : VkVideoEncodeRgbChromaOffsetFlagsVALVE
  {
    eCositedEven = VK_VIDEO_ENCODE_RGB_CHROMA_OFFSET_COSITED_EVEN_BIT_VALVE,
    eMidpoint    = VK_VIDEO_ENCODE_RGB_CHROMA_OFFSET_MIDPOINT_BIT_VALVE
  };

  // wrapper using for bitmask VkVideoEncodeRgbChromaOffsetFlagsVALVE, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeRgbChromaOffsetFlagsVALVE.html
  using VideoEncodeRgbChromaOffsetFlagsVALVE = Flags<VideoEncodeRgbChromaOffsetFlagBitsVALVE>;

  template <>
  struct FlagTraits<VideoEncodeRgbChromaOffsetFlagBitsVALVE>
  {
    using WrappedType                                                                   = VkVideoEncodeRgbChromaOffsetFlagBitsVALVE;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeRgbChromaOffsetFlagsVALVE allFlags =
      VideoEncodeRgbChromaOffsetFlagBitsVALVE::eCositedEven | VideoEncodeRgbChromaOffsetFlagBitsVALVE::eMidpoint;
  };

  //=== VK_EXT_opacity_micromap ===

  // wrapper class for enum VkMicromapTypeEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkMicromapTypeEXT.html
  enum class MicromapTypeEXT
  {
    eOpacityMicromap = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT,
#if defined( VK_ENABLE_BETA_EXTENSIONS )
    eDisplacementMicromapNV = VK_MICROMAP_TYPE_DISPLACEMENT_MICROMAP_NV
#endif /*VK_ENABLE_BETA_EXTENSIONS*/
  };

  // wrapper class for enum VkBuildMicromapFlagBitsEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkBuildMicromapFlagBitsEXT.html
  enum class BuildMicromapFlagBitsEXT : VkBuildMicromapFlagsEXT
  {
    ePreferFastTrace = VK_BUILD_MICROMAP_PREFER_FAST_TRACE_BIT_EXT,
    ePreferFastBuild = VK_BUILD_MICROMAP_PREFER_FAST_BUILD_BIT_EXT,
    eAllowCompaction = VK_BUILD_MICROMAP_ALLOW_COMPACTION_BIT_EXT
  };

  // wrapper using for bitmask VkBuildMicromapFlagsEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkBuildMicromapFlagsEXT.html
  using BuildMicromapFlagsEXT = Flags<BuildMicromapFlagBitsEXT>;

  template <>
  struct FlagTraits<BuildMicromapFlagBitsEXT>
  {
    using WrappedType                                                    = VkBuildMicromapFlagBitsEXT;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                  isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR BuildMicromapFlagsEXT allFlags =
      BuildMicromapFlagBitsEXT::ePreferFastTrace | BuildMicromapFlagBitsEXT::ePreferFastBuild | BuildMicromapFlagBitsEXT::eAllowCompaction;
  };

  // wrapper class for enum VkCopyMicromapModeEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkCopyMicromapModeEXT.html
  enum class CopyMicromapModeEXT
  {
    eClone       = VK_COPY_MICROMAP_MODE_CLONE_EXT,
    eSerialize   = VK_COPY_MICROMAP_MODE_SERIALIZE_EXT,
    eDeserialize = VK_COPY_MICROMAP_MODE_DESERIALIZE_EXT,
    eCompact     = VK_COPY_MICROMAP_MODE_COMPACT_EXT
  };

  // wrapper class for enum VkMicromapCreateFlagBitsEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkMicromapCreateFlagBitsEXT.html
  enum class MicromapCreateFlagBitsEXT : VkMicromapCreateFlagsEXT
  {
    eDeviceAddressCaptureReplay = VK_MICROMAP_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT_EXT
  };

  // wrapper using for bitmask VkMicromapCreateFlagsEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkMicromapCreateFlagsEXT.html
  using MicromapCreateFlagsEXT = Flags<MicromapCreateFlagBitsEXT>;

  template <>
  struct FlagTraits<MicromapCreateFlagBitsEXT>
  {
    using WrappedType                                                     = VkMicromapCreateFlagBitsEXT;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR MicromapCreateFlagsEXT allFlags  = MicromapCreateFlagBitsEXT::eDeviceAddressCaptureReplay;
  };

  // wrapper class for enum VkBuildMicromapModeEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkBuildMicromapModeEXT.html
  enum class BuildMicromapModeEXT
  {
    eBuild = VK_BUILD_MICROMAP_MODE_BUILD_EXT
  };

  // wrapper class for enum VkOpacityMicromapFormatEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkOpacityMicromapFormatEXT.html
  enum class OpacityMicromapFormatEXT
  {
    e2State = VK_OPACITY_MICROMAP_FORMAT_2_STATE_EXT,
    e4State = VK_OPACITY_MICROMAP_FORMAT_4_STATE_EXT
  };

  // wrapper class for enum VkOpacityMicromapSpecialIndexEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkOpacityMicromapSpecialIndexEXT.html
  enum class OpacityMicromapSpecialIndexEXT
  {
    eFullyTransparent                        = VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_TRANSPARENT_EXT,
    eFullyOpaque                             = VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_OPAQUE_EXT,
    eFullyUnknownTransparent                 = VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_UNKNOWN_TRANSPARENT_EXT,
    eFullyUnknownOpaque                      = VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_UNKNOWN_OPAQUE_EXT,
    eClusterGeometryDisableOpacityMicromapNV = VK_OPACITY_MICROMAP_SPECIAL_INDEX_CLUSTER_GEOMETRY_DISABLE_OPACITY_MICROMAP_NV
  };

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_NV_displacement_micromap ===

  // wrapper class for enum VkDisplacementMicromapFormatNV, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDisplacementMicromapFormatNV.html
  enum class DisplacementMicromapFormatNV
  {
    e64Triangles64Bytes    = VK_DISPLACEMENT_MICROMAP_FORMAT_64_TRIANGLES_64_BYTES_NV,
    e256Triangles128Bytes  = VK_DISPLACEMENT_MICROMAP_FORMAT_256_TRIANGLES_128_BYTES_NV,
    e1024Triangles128Bytes = VK_DISPLACEMENT_MICROMAP_FORMAT_1024_TRIANGLES_128_BYTES_NV
  };

#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_ARM_scheduling_controls ===

  // wrapper class for enum VkPhysicalDeviceSchedulingControlsFlagBitsARM, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceSchedulingControlsFlagBitsARM.html
  enum class PhysicalDeviceSchedulingControlsFlagBitsARM : VkPhysicalDeviceSchedulingControlsFlagsARM
  {
    eShaderCoreCount = VK_PHYSICAL_DEVICE_SCHEDULING_CONTROLS_SHADER_CORE_COUNT_ARM
  };

  // wrapper using for bitmask VkPhysicalDeviceSchedulingControlsFlagsARM, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceSchedulingControlsFlagsARM.html
  using PhysicalDeviceSchedulingControlsFlagsARM = Flags<PhysicalDeviceSchedulingControlsFlagBitsARM>;

  template <>
  struct FlagTraits<PhysicalDeviceSchedulingControlsFlagBitsARM>
  {
    using WrappedType                                                                       = VkPhysicalDeviceSchedulingControlsFlagBitsARM;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                     isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PhysicalDeviceSchedulingControlsFlagsARM allFlags  = PhysicalDeviceSchedulingControlsFlagBitsARM::eShaderCoreCount;
  };

  //=== VK_NV_ray_tracing_linear_swept_spheres ===

  // wrapper class for enum VkRayTracingLssIndexingModeNV, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkRayTracingLssIndexingModeNV.html
  enum class RayTracingLssIndexingModeNV
  {
    eList       = VK_RAY_TRACING_LSS_INDEXING_MODE_LIST_NV,
    eSuccessive = VK_RAY_TRACING_LSS_INDEXING_MODE_SUCCESSIVE_NV
  };

  // wrapper class for enum VkRayTracingLssPrimitiveEndCapsModeNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkRayTracingLssPrimitiveEndCapsModeNV.html
  enum class RayTracingLssPrimitiveEndCapsModeNV
  {
    eNone    = VK_RAY_TRACING_LSS_PRIMITIVE_END_CAPS_MODE_NONE_NV,
    eChained = VK_RAY_TRACING_LSS_PRIMITIVE_END_CAPS_MODE_CHAINED_NV
  };

  //=== VK_EXT_subpass_merge_feedback ===

  // wrapper class for enum VkSubpassMergeStatusEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSubpassMergeStatusEXT.html
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

  // wrapper class for enum VkDirectDriverLoadingModeLUNARG, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDirectDriverLoadingModeLUNARG.html
  enum class DirectDriverLoadingModeLUNARG
  {
    eExclusive = VK_DIRECT_DRIVER_LOADING_MODE_EXCLUSIVE_LUNARG,
    eInclusive = VK_DIRECT_DRIVER_LOADING_MODE_INCLUSIVE_LUNARG
  };

  enum class DirectDriverLoadingFlagBitsLUNARG : VkDirectDriverLoadingFlagsLUNARG
  {
  };

  // wrapper using for bitmask VkDirectDriverLoadingFlagsLUNARG, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkDirectDriverLoadingFlagsLUNARG.html
  using DirectDriverLoadingFlagsLUNARG = Flags<DirectDriverLoadingFlagBitsLUNARG>;

  template <>
  struct FlagTraits<DirectDriverLoadingFlagBitsLUNARG>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DirectDriverLoadingFlagsLUNARG allFlags  = {};
  };

  //=== VK_ARM_tensors ===

  // wrapper class for enum VkTensorCreateFlagBitsARM, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkTensorCreateFlagBitsARM.html
  enum class TensorCreateFlagBitsARM : VkTensorCreateFlagsARM
  {
    eMutableFormat                 = VK_TENSOR_CREATE_MUTABLE_FORMAT_BIT_ARM,
    eProtected                     = VK_TENSOR_CREATE_PROTECTED_BIT_ARM,
    eDescriptorBufferCaptureReplay = VK_TENSOR_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_ARM
  };

  // wrapper using for bitmask VkTensorCreateFlagsARM, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkTensorCreateFlagsARM.html
  using TensorCreateFlagsARM = Flags<TensorCreateFlagBitsARM>;

  template <>
  struct FlagTraits<TensorCreateFlagBitsARM>
  {
    using WrappedType                                                   = VkTensorCreateFlagBitsARM;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR TensorCreateFlagsARM allFlags =
      TensorCreateFlagBitsARM::eMutableFormat | TensorCreateFlagBitsARM::eProtected | TensorCreateFlagBitsARM::eDescriptorBufferCaptureReplay;
  };

  // wrapper class for enum VkTensorViewCreateFlagBitsARM, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkTensorViewCreateFlagBitsARM.html
  enum class TensorViewCreateFlagBitsARM : VkTensorViewCreateFlagsARM
  {
    eDescriptorBufferCaptureReplay = VK_TENSOR_VIEW_CREATE_DESCRIPTOR_BUFFER_CAPTURE_REPLAY_BIT_ARM
  };

  // wrapper using for bitmask VkTensorViewCreateFlagsARM, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkTensorViewCreateFlagsARM.html
  using TensorViewCreateFlagsARM = Flags<TensorViewCreateFlagBitsARM>;

  template <>
  struct FlagTraits<TensorViewCreateFlagBitsARM>
  {
    using WrappedType                                                       = VkTensorViewCreateFlagBitsARM;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                     isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR TensorViewCreateFlagsARM allFlags  = TensorViewCreateFlagBitsARM::eDescriptorBufferCaptureReplay;
  };

  // wrapper class for enum VkTensorUsageFlagBitsARM, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkTensorUsageFlagBitsARM.html
  enum class TensorUsageFlagBitsARM : VkTensorUsageFlagsARM
  {
    eShader        = VK_TENSOR_USAGE_SHADER_BIT_ARM,
    eTransferSrc   = VK_TENSOR_USAGE_TRANSFER_SRC_BIT_ARM,
    eTransferDst   = VK_TENSOR_USAGE_TRANSFER_DST_BIT_ARM,
    eImageAliasing = VK_TENSOR_USAGE_IMAGE_ALIASING_BIT_ARM,
    eDataGraph     = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM
  };

  // wrapper using for bitmask VkTensorUsageFlagsARM, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkTensorUsageFlagsARM.html
  using TensorUsageFlagsARM = Flags<TensorUsageFlagBitsARM>;

  template <>
  struct FlagTraits<TensorUsageFlagBitsARM>
  {
    using WrappedType                                                  = VkTensorUsageFlagBitsARM;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR TensorUsageFlagsARM allFlags  = TensorUsageFlagBitsARM::eShader | TensorUsageFlagBitsARM::eTransferSrc |
                                                                        TensorUsageFlagBitsARM::eTransferDst | TensorUsageFlagBitsARM::eImageAliasing |
                                                                        TensorUsageFlagBitsARM::eDataGraph;
  };

  // wrapper class for enum VkTensorTilingARM, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkTensorTilingARM.html
  enum class TensorTilingARM
  {
    eOptimal = VK_TENSOR_TILING_OPTIMAL_ARM,
    eLinear  = VK_TENSOR_TILING_LINEAR_ARM
  };

  //=== VK_NV_optical_flow ===

  // wrapper class for enum VkOpticalFlowUsageFlagBitsNV, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkOpticalFlowUsageFlagBitsNV.html
  enum class OpticalFlowUsageFlagBitsNV : VkOpticalFlowUsageFlagsNV
  {
    eUnknown    = VK_OPTICAL_FLOW_USAGE_UNKNOWN_NV,
    eInput      = VK_OPTICAL_FLOW_USAGE_INPUT_BIT_NV,
    eOutput     = VK_OPTICAL_FLOW_USAGE_OUTPUT_BIT_NV,
    eHint       = VK_OPTICAL_FLOW_USAGE_HINT_BIT_NV,
    eCost       = VK_OPTICAL_FLOW_USAGE_COST_BIT_NV,
    eGlobalFlow = VK_OPTICAL_FLOW_USAGE_GLOBAL_FLOW_BIT_NV
  };

  // wrapper using for bitmask VkOpticalFlowUsageFlagsNV, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkOpticalFlowUsageFlagsNV.html
  using OpticalFlowUsageFlagsNV = Flags<OpticalFlowUsageFlagBitsNV>;

  template <>
  struct FlagTraits<OpticalFlowUsageFlagBitsNV>
  {
    using WrappedType                                                      = VkOpticalFlowUsageFlagBitsNV;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                    isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR OpticalFlowUsageFlagsNV allFlags  = OpticalFlowUsageFlagBitsNV::eUnknown | OpticalFlowUsageFlagBitsNV::eInput |
                                                                            OpticalFlowUsageFlagBitsNV::eOutput | OpticalFlowUsageFlagBitsNV::eHint |
                                                                            OpticalFlowUsageFlagBitsNV::eCost | OpticalFlowUsageFlagBitsNV::eGlobalFlow;
  };

  // wrapper class for enum VkOpticalFlowGridSizeFlagBitsNV, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkOpticalFlowGridSizeFlagBitsNV.html
  enum class OpticalFlowGridSizeFlagBitsNV : VkOpticalFlowGridSizeFlagsNV
  {
    eUnknown = VK_OPTICAL_FLOW_GRID_SIZE_UNKNOWN_NV,
    e1X1     = VK_OPTICAL_FLOW_GRID_SIZE_1X1_BIT_NV,
    e2X2     = VK_OPTICAL_FLOW_GRID_SIZE_2X2_BIT_NV,
    e4X4     = VK_OPTICAL_FLOW_GRID_SIZE_4X4_BIT_NV,
    e8X8     = VK_OPTICAL_FLOW_GRID_SIZE_8X8_BIT_NV
  };

  // wrapper using for bitmask VkOpticalFlowGridSizeFlagsNV, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkOpticalFlowGridSizeFlagsNV.html
  using OpticalFlowGridSizeFlagsNV = Flags<OpticalFlowGridSizeFlagBitsNV>;

  template <>
  struct FlagTraits<OpticalFlowGridSizeFlagBitsNV>
  {
    using WrappedType                                                         = VkOpticalFlowGridSizeFlagBitsNV;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR OpticalFlowGridSizeFlagsNV allFlags  = OpticalFlowGridSizeFlagBitsNV::eUnknown | OpticalFlowGridSizeFlagBitsNV::e1X1 |
                                                                               OpticalFlowGridSizeFlagBitsNV::e2X2 | OpticalFlowGridSizeFlagBitsNV::e4X4 |
                                                                               OpticalFlowGridSizeFlagBitsNV::e8X8;
  };

  // wrapper class for enum VkOpticalFlowPerformanceLevelNV, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkOpticalFlowPerformanceLevelNV.html
  enum class OpticalFlowPerformanceLevelNV
  {
    eUnknown = VK_OPTICAL_FLOW_PERFORMANCE_LEVEL_UNKNOWN_NV,
    eSlow    = VK_OPTICAL_FLOW_PERFORMANCE_LEVEL_SLOW_NV,
    eMedium  = VK_OPTICAL_FLOW_PERFORMANCE_LEVEL_MEDIUM_NV,
    eFast    = VK_OPTICAL_FLOW_PERFORMANCE_LEVEL_FAST_NV
  };

  // wrapper class for enum VkOpticalFlowSessionBindingPointNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkOpticalFlowSessionBindingPointNV.html
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

  // wrapper class for enum VkOpticalFlowSessionCreateFlagBitsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkOpticalFlowSessionCreateFlagBitsNV.html
  enum class OpticalFlowSessionCreateFlagBitsNV : VkOpticalFlowSessionCreateFlagsNV
  {
    eEnableHint       = VK_OPTICAL_FLOW_SESSION_CREATE_ENABLE_HINT_BIT_NV,
    eEnableCost       = VK_OPTICAL_FLOW_SESSION_CREATE_ENABLE_COST_BIT_NV,
    eEnableGlobalFlow = VK_OPTICAL_FLOW_SESSION_CREATE_ENABLE_GLOBAL_FLOW_BIT_NV,
    eAllowRegions     = VK_OPTICAL_FLOW_SESSION_CREATE_ALLOW_REGIONS_BIT_NV,
    eBothDirections   = VK_OPTICAL_FLOW_SESSION_CREATE_BOTH_DIRECTIONS_BIT_NV
  };

  // wrapper using for bitmask VkOpticalFlowSessionCreateFlagsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkOpticalFlowSessionCreateFlagsNV.html
  using OpticalFlowSessionCreateFlagsNV = Flags<OpticalFlowSessionCreateFlagBitsNV>;

  template <>
  struct FlagTraits<OpticalFlowSessionCreateFlagBitsNV>
  {
    using WrappedType                                                              = VkOpticalFlowSessionCreateFlagBitsNV;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                            isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR OpticalFlowSessionCreateFlagsNV allFlags =
      OpticalFlowSessionCreateFlagBitsNV::eEnableHint | OpticalFlowSessionCreateFlagBitsNV::eEnableCost |
      OpticalFlowSessionCreateFlagBitsNV::eEnableGlobalFlow | OpticalFlowSessionCreateFlagBitsNV::eAllowRegions |
      OpticalFlowSessionCreateFlagBitsNV::eBothDirections;
  };

  // wrapper class for enum VkOpticalFlowExecuteFlagBitsNV, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkOpticalFlowExecuteFlagBitsNV.html
  enum class OpticalFlowExecuteFlagBitsNV : VkOpticalFlowExecuteFlagsNV
  {
    eDisableTemporalHints = VK_OPTICAL_FLOW_EXECUTE_DISABLE_TEMPORAL_HINTS_BIT_NV
  };

  // wrapper using for bitmask VkOpticalFlowExecuteFlagsNV, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkOpticalFlowExecuteFlagsNV.html
  using OpticalFlowExecuteFlagsNV = Flags<OpticalFlowExecuteFlagBitsNV>;

  template <>
  struct FlagTraits<OpticalFlowExecuteFlagBitsNV>
  {
    using WrappedType                                                        = VkOpticalFlowExecuteFlagBitsNV;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                      isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR OpticalFlowExecuteFlagsNV allFlags  = OpticalFlowExecuteFlagBitsNV::eDisableTemporalHints;
  };

  //=== VK_AMD_anti_lag ===

  // wrapper class for enum VkAntiLagModeAMD, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkAntiLagModeAMD.html
  enum class AntiLagModeAMD
  {
    eDriverControl = VK_ANTI_LAG_MODE_DRIVER_CONTROL_AMD,
    eOn            = VK_ANTI_LAG_MODE_ON_AMD,
    eOff           = VK_ANTI_LAG_MODE_OFF_AMD
  };

  // wrapper class for enum VkAntiLagStageAMD, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkAntiLagStageAMD.html
  enum class AntiLagStageAMD
  {
    eInput   = VK_ANTI_LAG_STAGE_INPUT_AMD,
    ePresent = VK_ANTI_LAG_STAGE_PRESENT_AMD
  };

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_AMDX_dense_geometry_format ===

  // wrapper class for enum VkCompressedTriangleFormatAMDX, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkCompressedTriangleFormatAMDX.html
  enum class CompressedTriangleFormatAMDX
  {
    eDgf1 = VK_COMPRESSED_TRIANGLE_FORMAT_DGF1_AMDX
  };

#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_EXT_shader_object ===

  // wrapper class for enum VkShaderCreateFlagBitsEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkShaderCreateFlagBitsEXT.html
  enum class ShaderCreateFlagBitsEXT : VkShaderCreateFlagsEXT
  {
    eLinkStage                     = VK_SHADER_CREATE_LINK_STAGE_BIT_EXT,
    eAllowVaryingSubgroupSize      = VK_SHADER_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT_EXT,
    eRequireFullSubgroups          = VK_SHADER_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT,
    eNoTaskShader                  = VK_SHADER_CREATE_NO_TASK_SHADER_BIT_EXT,
    eDispatchBase                  = VK_SHADER_CREATE_DISPATCH_BASE_BIT_EXT,
    eFragmentShadingRateAttachment = VK_SHADER_CREATE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_EXT,
    eFragmentDensityMapAttachment  = VK_SHADER_CREATE_FRAGMENT_DENSITY_MAP_ATTACHMENT_BIT_EXT,
    eIndirectBindable              = VK_SHADER_CREATE_INDIRECT_BINDABLE_BIT_EXT,
    e64BitIndexing                 = VK_SHADER_CREATE_64_BIT_INDEXING_BIT_EXT
  };

  // wrapper using for bitmask VkShaderCreateFlagsEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkShaderCreateFlagsEXT.html
  using ShaderCreateFlagsEXT = Flags<ShaderCreateFlagBitsEXT>;

  template <>
  struct FlagTraits<ShaderCreateFlagBitsEXT>
  {
    using WrappedType                                                   = VkShaderCreateFlagBitsEXT;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ShaderCreateFlagsEXT allFlags =
      ShaderCreateFlagBitsEXT::eLinkStage | ShaderCreateFlagBitsEXT::eAllowVaryingSubgroupSize | ShaderCreateFlagBitsEXT::eRequireFullSubgroups |
      ShaderCreateFlagBitsEXT::eNoTaskShader | ShaderCreateFlagBitsEXT::eDispatchBase | ShaderCreateFlagBitsEXT::eFragmentShadingRateAttachment |
      ShaderCreateFlagBitsEXT::eFragmentDensityMapAttachment | ShaderCreateFlagBitsEXT::eIndirectBindable | ShaderCreateFlagBitsEXT::e64BitIndexing;
  };

  // wrapper class for enum VkShaderCodeTypeEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkShaderCodeTypeEXT.html
  enum class ShaderCodeTypeEXT
  {
    eBinary = VK_SHADER_CODE_TYPE_BINARY_EXT,
    eSpirv  = VK_SHADER_CODE_TYPE_SPIRV_EXT
  };

  //=== VK_KHR_surface_maintenance1 ===

  // wrapper class for enum VkPresentScalingFlagBitsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPresentScalingFlagBitsKHR.html
  enum class PresentScalingFlagBitsKHR : VkPresentScalingFlagsKHR
  {
    eOneToOne           = VK_PRESENT_SCALING_ONE_TO_ONE_BIT_KHR,
    eAspectRatioStretch = VK_PRESENT_SCALING_ASPECT_RATIO_STRETCH_BIT_KHR,
    eStretch            = VK_PRESENT_SCALING_STRETCH_BIT_KHR
  };

  using PresentScalingFlagBitsEXT = PresentScalingFlagBitsKHR;

  // wrapper using for bitmask VkPresentScalingFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPresentScalingFlagsKHR.html
  using PresentScalingFlagsKHR = Flags<PresentScalingFlagBitsKHR>;
  using PresentScalingFlagsEXT = PresentScalingFlagsKHR;

  template <>
  struct FlagTraits<PresentScalingFlagBitsKHR>
  {
    using WrappedType                                                     = VkPresentScalingFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PresentScalingFlagsKHR allFlags =
      PresentScalingFlagBitsKHR::eOneToOne | PresentScalingFlagBitsKHR::eAspectRatioStretch | PresentScalingFlagBitsKHR::eStretch;
  };

  // wrapper class for enum VkPresentGravityFlagBitsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPresentGravityFlagBitsKHR.html
  enum class PresentGravityFlagBitsKHR : VkPresentGravityFlagsKHR
  {
    eMin      = VK_PRESENT_GRAVITY_MIN_BIT_KHR,
    eMax      = VK_PRESENT_GRAVITY_MAX_BIT_KHR,
    eCentered = VK_PRESENT_GRAVITY_CENTERED_BIT_KHR
  };

  using PresentGravityFlagBitsEXT = PresentGravityFlagBitsKHR;

  // wrapper using for bitmask VkPresentGravityFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPresentGravityFlagsKHR.html
  using PresentGravityFlagsKHR = Flags<PresentGravityFlagBitsKHR>;
  using PresentGravityFlagsEXT = PresentGravityFlagsKHR;

  template <>
  struct FlagTraits<PresentGravityFlagBitsKHR>
  {
    using WrappedType                                                     = VkPresentGravityFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PresentGravityFlagsKHR allFlags =
      PresentGravityFlagBitsKHR::eMin | PresentGravityFlagBitsKHR::eMax | PresentGravityFlagBitsKHR::eCentered;
  };

  //=== VK_NV_cooperative_vector ===

  // wrapper class for enum VkCooperativeVectorMatrixLayoutNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkCooperativeVectorMatrixLayoutNV.html
  enum class CooperativeVectorMatrixLayoutNV
  {
    eRowMajor           = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV,
    eColumnMajor        = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_COLUMN_MAJOR_NV,
    eInferencingOptimal = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_NV,
    eTrainingOptimal    = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_TRAINING_OPTIMAL_NV
  };

  // wrapper class for enum VkComponentTypeKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkComponentTypeKHR.html
  enum class ComponentTypeKHR
  {
    eFloat16       = VK_COMPONENT_TYPE_FLOAT16_KHR,
    eFloat32       = VK_COMPONENT_TYPE_FLOAT32_KHR,
    eFloat64       = VK_COMPONENT_TYPE_FLOAT64_KHR,
    eSint8         = VK_COMPONENT_TYPE_SINT8_KHR,
    eSint16        = VK_COMPONENT_TYPE_SINT16_KHR,
    eSint32        = VK_COMPONENT_TYPE_SINT32_KHR,
    eSint64        = VK_COMPONENT_TYPE_SINT64_KHR,
    eUint8         = VK_COMPONENT_TYPE_UINT8_KHR,
    eUint16        = VK_COMPONENT_TYPE_UINT16_KHR,
    eUint32        = VK_COMPONENT_TYPE_UINT32_KHR,
    eUint64        = VK_COMPONENT_TYPE_UINT64_KHR,
    eBfloat16      = VK_COMPONENT_TYPE_BFLOAT16_KHR,
    eSint8PackedNV = VK_COMPONENT_TYPE_SINT8_PACKED_NV,
    eUint8PackedNV = VK_COMPONENT_TYPE_UINT8_PACKED_NV,
    eFloat8E4M3EXT = VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT,
    eFloatE4M3     = VK_COMPONENT_TYPE_FLOAT_E4M3_NV,
    eFloat8E5M2EXT = VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT,
    eFloatE5M2     = VK_COMPONENT_TYPE_FLOAT_E5M2_NV
  };

  using ComponentTypeNV = ComponentTypeKHR;

  //=== VK_EXT_layer_settings ===

  // wrapper class for enum VkLayerSettingTypeEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkLayerSettingTypeEXT.html
  enum class LayerSettingTypeEXT
  {
    eBool32  = VK_LAYER_SETTING_TYPE_BOOL32_EXT,
    eInt32   = VK_LAYER_SETTING_TYPE_INT32_EXT,
    eInt64   = VK_LAYER_SETTING_TYPE_INT64_EXT,
    eUint32  = VK_LAYER_SETTING_TYPE_UINT32_EXT,
    eUint64  = VK_LAYER_SETTING_TYPE_UINT64_EXT,
    eFloat32 = VK_LAYER_SETTING_TYPE_FLOAT32_EXT,
    eFloat64 = VK_LAYER_SETTING_TYPE_FLOAT64_EXT,
    eString  = VK_LAYER_SETTING_TYPE_STRING_EXT
  };

  //=================================
  //=== Layer Setting Type Traits ===
  //=================================

  template <>
  struct CppType<LayerSettingTypeEXT, LayerSettingTypeEXT::eBool32>
  {
    using Type = Bool32;
  };

  template <>
  struct CppType<LayerSettingTypeEXT, LayerSettingTypeEXT::eInt32>
  {
    using Type = int32_t;
  };

  template <>
  struct CppType<LayerSettingTypeEXT, LayerSettingTypeEXT::eInt64>
  {
    using Type = int64_t;
  };

  template <>
  struct CppType<LayerSettingTypeEXT, LayerSettingTypeEXT::eUint32>
  {
    using Type = uint32_t;
  };

  template <>
  struct CppType<LayerSettingTypeEXT, LayerSettingTypeEXT::eUint64>
  {
    using Type = uint64_t;
  };

  template <>
  struct CppType<LayerSettingTypeEXT, LayerSettingTypeEXT::eFloat32>
  {
    using Type = float;
  };

  template <>
  struct CppType<LayerSettingTypeEXT, LayerSettingTypeEXT::eFloat64>
  {
    using Type = double;
  };

  template <>
  struct CppType<LayerSettingTypeEXT, LayerSettingTypeEXT::eString>
  {
    using Type = char *;
  };

  template <typename T>
  bool isSameType( LayerSettingTypeEXT layerSettingType )
  {
    switch ( layerSettingType )
    {
      case LayerSettingTypeEXT::eBool32 : return std::is_same<T, Bool32>::value;
      case LayerSettingTypeEXT::eInt32  : return std::is_same<T, int32_t>::value;
      case LayerSettingTypeEXT::eInt64  : return std::is_same<T, int64_t>::value;
      case LayerSettingTypeEXT::eUint32 : return std::is_same<T, uint32_t>::value;
      case LayerSettingTypeEXT::eUint64 : return std::is_same<T, uint64_t>::value;
      case LayerSettingTypeEXT::eFloat32: return std::is_same<T, float>::value;
      case LayerSettingTypeEXT::eFloat64: return std::is_same<T, double>::value;
      case LayerSettingTypeEXT::eString : return std::is_same<T, char *>::value;
      default                           : return false;
    }
  }

  //=== VK_NV_low_latency2 ===

  // wrapper class for enum VkLatencyMarkerNV, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkLatencyMarkerNV.html
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

  // wrapper class for enum VkOutOfBandQueueTypeNV, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkOutOfBandQueueTypeNV.html
  enum class OutOfBandQueueTypeNV
  {
    eRender  = VK_OUT_OF_BAND_QUEUE_TYPE_RENDER_NV,
    ePresent = VK_OUT_OF_BAND_QUEUE_TYPE_PRESENT_NV
  };

  //=== VK_KHR_cooperative_matrix ===

  // wrapper class for enum VkScopeKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkScopeKHR.html
  enum class ScopeKHR
  {
    eDevice      = VK_SCOPE_DEVICE_KHR,
    eWorkgroup   = VK_SCOPE_WORKGROUP_KHR,
    eSubgroup    = VK_SCOPE_SUBGROUP_KHR,
    eQueueFamily = VK_SCOPE_QUEUE_FAMILY_KHR
  };

  using ScopeNV = ScopeKHR;

  //=== VK_ARM_data_graph ===

  // wrapper class for enum VkDataGraphPipelineSessionBindPointARM, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkDataGraphPipelineSessionBindPointARM.html
  enum class DataGraphPipelineSessionBindPointARM
  {
    eTransient = VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM
  };

  // wrapper class for enum VkDataGraphPipelineSessionBindPointTypeARM, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkDataGraphPipelineSessionBindPointTypeARM.html
  enum class DataGraphPipelineSessionBindPointTypeARM
  {
    eMemory = VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TYPE_MEMORY_ARM
  };

  // wrapper class for enum VkDataGraphPipelineSessionCreateFlagBitsARM, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkDataGraphPipelineSessionCreateFlagBitsARM.html
  enum class DataGraphPipelineSessionCreateFlagBitsARM : VkDataGraphPipelineSessionCreateFlagsARM
  {
    eProtected = VK_DATA_GRAPH_PIPELINE_SESSION_CREATE_PROTECTED_BIT_ARM
  };

  // wrapper using for bitmask VkDataGraphPipelineSessionCreateFlagsARM, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkDataGraphPipelineSessionCreateFlagsARM.html
  using DataGraphPipelineSessionCreateFlagsARM = Flags<DataGraphPipelineSessionCreateFlagBitsARM>;

  template <>
  struct FlagTraits<DataGraphPipelineSessionCreateFlagBitsARM>
  {
    using WrappedType                                                                     = VkDataGraphPipelineSessionCreateFlagBitsARM;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DataGraphPipelineSessionCreateFlagsARM allFlags  = DataGraphPipelineSessionCreateFlagBitsARM::eProtected;
  };

  // wrapper class for enum VkDataGraphPipelinePropertyARM, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDataGraphPipelinePropertyARM.html
  enum class DataGraphPipelinePropertyARM
  {
    eCreationLog = VK_DATA_GRAPH_PIPELINE_PROPERTY_CREATION_LOG_ARM,
    eIdentifier  = VK_DATA_GRAPH_PIPELINE_PROPERTY_IDENTIFIER_ARM
  };

  enum class DataGraphPipelineDispatchFlagBitsARM : VkDataGraphPipelineDispatchFlagsARM
  {
  };

  // wrapper using for bitmask VkDataGraphPipelineDispatchFlagsARM, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkDataGraphPipelineDispatchFlagsARM.html
  using DataGraphPipelineDispatchFlagsARM = Flags<DataGraphPipelineDispatchFlagBitsARM>;

  template <>
  struct FlagTraits<DataGraphPipelineDispatchFlagBitsARM>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                              isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR DataGraphPipelineDispatchFlagsARM allFlags  = {};
  };

  // wrapper class for enum VkPhysicalDeviceDataGraphProcessingEngineTypeARM, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceDataGraphProcessingEngineTypeARM.html
  enum class PhysicalDeviceDataGraphProcessingEngineTypeARM
  {
    eDefault     = VK_PHYSICAL_DEVICE_DATA_GRAPH_PROCESSING_ENGINE_TYPE_DEFAULT_ARM,
    eNeuralQCOM  = VK_PHYSICAL_DEVICE_DATA_GRAPH_PROCESSING_ENGINE_TYPE_NEURAL_QCOM,
    eComputeQCOM = VK_PHYSICAL_DEVICE_DATA_GRAPH_PROCESSING_ENGINE_TYPE_COMPUTE_QCOM
  };

  // wrapper class for enum VkPhysicalDeviceDataGraphOperationTypeARM, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceDataGraphOperationTypeARM.html
  enum class PhysicalDeviceDataGraphOperationTypeARM
  {
    eSpirvExtendedInstructionSet = VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_SPIRV_EXTENDED_INSTRUCTION_SET_ARM,
    eNeuralModelQCOM             = VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_NEURAL_MODEL_QCOM,
    eBuiltinModelQCOM            = VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_BUILTIN_MODEL_QCOM
  };

  //=== VK_KHR_video_encode_av1 ===

  // wrapper class for enum VkVideoEncodeAV1PredictionModeKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeAV1PredictionModeKHR.html
  enum class VideoEncodeAV1PredictionModeKHR
  {
    eIntraOnly              = VK_VIDEO_ENCODE_AV1_PREDICTION_MODE_INTRA_ONLY_KHR,
    eSingleReference        = VK_VIDEO_ENCODE_AV1_PREDICTION_MODE_SINGLE_REFERENCE_KHR,
    eUnidirectionalCompound = VK_VIDEO_ENCODE_AV1_PREDICTION_MODE_UNIDIRECTIONAL_COMPOUND_KHR,
    eBidirectionalCompound  = VK_VIDEO_ENCODE_AV1_PREDICTION_MODE_BIDIRECTIONAL_COMPOUND_KHR
  };

  // wrapper class for enum VkVideoEncodeAV1RateControlGroupKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeAV1RateControlGroupKHR.html
  enum class VideoEncodeAV1RateControlGroupKHR
  {
    eIntra        = VK_VIDEO_ENCODE_AV1_RATE_CONTROL_GROUP_INTRA_KHR,
    ePredictive   = VK_VIDEO_ENCODE_AV1_RATE_CONTROL_GROUP_PREDICTIVE_KHR,
    eBipredictive = VK_VIDEO_ENCODE_AV1_RATE_CONTROL_GROUP_BIPREDICTIVE_KHR
  };

  // wrapper class for enum VkVideoEncodeAV1CapabilityFlagBitsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeAV1CapabilityFlagBitsKHR.html
  enum class VideoEncodeAV1CapabilityFlagBitsKHR : VkVideoEncodeAV1CapabilityFlagsKHR
  {
    ePerRateControlGroupMinMaxQIndex = VK_VIDEO_ENCODE_AV1_CAPABILITY_PER_RATE_CONTROL_GROUP_MIN_MAX_Q_INDEX_BIT_KHR,
    eGenerateObuExtensionHeader      = VK_VIDEO_ENCODE_AV1_CAPABILITY_GENERATE_OBU_EXTENSION_HEADER_BIT_KHR,
    ePrimaryReferenceCdfOnly         = VK_VIDEO_ENCODE_AV1_CAPABILITY_PRIMARY_REFERENCE_CDF_ONLY_BIT_KHR,
    eFrameSizeOverride               = VK_VIDEO_ENCODE_AV1_CAPABILITY_FRAME_SIZE_OVERRIDE_BIT_KHR,
    eMotionVectorScaling             = VK_VIDEO_ENCODE_AV1_CAPABILITY_MOTION_VECTOR_SCALING_BIT_KHR,
    eCompoundPredictionIntraRefresh  = VK_VIDEO_ENCODE_AV1_CAPABILITY_COMPOUND_PREDICTION_INTRA_REFRESH_BIT_KHR
  };

  // wrapper using for bitmask VkVideoEncodeAV1CapabilityFlagsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeAV1CapabilityFlagsKHR.html
  using VideoEncodeAV1CapabilityFlagsKHR = Flags<VideoEncodeAV1CapabilityFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeAV1CapabilityFlagBitsKHR>
  {
    using WrappedType                                                               = VkVideoEncodeAV1CapabilityFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                             isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeAV1CapabilityFlagsKHR allFlags =
      VideoEncodeAV1CapabilityFlagBitsKHR::ePerRateControlGroupMinMaxQIndex | VideoEncodeAV1CapabilityFlagBitsKHR::eGenerateObuExtensionHeader |
      VideoEncodeAV1CapabilityFlagBitsKHR::ePrimaryReferenceCdfOnly | VideoEncodeAV1CapabilityFlagBitsKHR::eFrameSizeOverride |
      VideoEncodeAV1CapabilityFlagBitsKHR::eMotionVectorScaling | VideoEncodeAV1CapabilityFlagBitsKHR::eCompoundPredictionIntraRefresh;
  };

  // wrapper class for enum VkVideoEncodeAV1StdFlagBitsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeAV1StdFlagBitsKHR.html
  enum class VideoEncodeAV1StdFlagBitsKHR : VkVideoEncodeAV1StdFlagsKHR
  {
    eUniformTileSpacingFlagSet = VK_VIDEO_ENCODE_AV1_STD_UNIFORM_TILE_SPACING_FLAG_SET_BIT_KHR,
    eSkipModePresentUnset      = VK_VIDEO_ENCODE_AV1_STD_SKIP_MODE_PRESENT_UNSET_BIT_KHR,
    ePrimaryRefFrame           = VK_VIDEO_ENCODE_AV1_STD_PRIMARY_REF_FRAME_BIT_KHR,
    eDeltaQ                    = VK_VIDEO_ENCODE_AV1_STD_DELTA_Q_BIT_KHR
  };

  // wrapper using for bitmask VkVideoEncodeAV1StdFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeAV1StdFlagsKHR.html
  using VideoEncodeAV1StdFlagsKHR = Flags<VideoEncodeAV1StdFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeAV1StdFlagBitsKHR>
  {
    using WrappedType                                                        = VkVideoEncodeAV1StdFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                      isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeAV1StdFlagsKHR allFlags =
      VideoEncodeAV1StdFlagBitsKHR::eUniformTileSpacingFlagSet | VideoEncodeAV1StdFlagBitsKHR::eSkipModePresentUnset |
      VideoEncodeAV1StdFlagBitsKHR::ePrimaryRefFrame | VideoEncodeAV1StdFlagBitsKHR::eDeltaQ;
  };

  // wrapper class for enum VkVideoEncodeAV1SuperblockSizeFlagBitsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeAV1SuperblockSizeFlagBitsKHR.html
  enum class VideoEncodeAV1SuperblockSizeFlagBitsKHR : VkVideoEncodeAV1SuperblockSizeFlagsKHR
  {
    e64  = VK_VIDEO_ENCODE_AV1_SUPERBLOCK_SIZE_64_BIT_KHR,
    e128 = VK_VIDEO_ENCODE_AV1_SUPERBLOCK_SIZE_128_BIT_KHR
  };

  // wrapper using for bitmask VkVideoEncodeAV1SuperblockSizeFlagsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeAV1SuperblockSizeFlagsKHR.html
  using VideoEncodeAV1SuperblockSizeFlagsKHR = Flags<VideoEncodeAV1SuperblockSizeFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeAV1SuperblockSizeFlagBitsKHR>
  {
    using WrappedType                                                                   = VkVideoEncodeAV1SuperblockSizeFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeAV1SuperblockSizeFlagsKHR allFlags =
      VideoEncodeAV1SuperblockSizeFlagBitsKHR::e64 | VideoEncodeAV1SuperblockSizeFlagBitsKHR::e128;
  };

  // wrapper class for enum VkVideoEncodeAV1RateControlFlagBitsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeAV1RateControlFlagBitsKHR.html
  enum class VideoEncodeAV1RateControlFlagBitsKHR : VkVideoEncodeAV1RateControlFlagsKHR
  {
    eRegularGop                 = VK_VIDEO_ENCODE_AV1_RATE_CONTROL_REGULAR_GOP_BIT_KHR,
    eTemporalLayerPatternDyadic = VK_VIDEO_ENCODE_AV1_RATE_CONTROL_TEMPORAL_LAYER_PATTERN_DYADIC_BIT_KHR,
    eReferencePatternFlat       = VK_VIDEO_ENCODE_AV1_RATE_CONTROL_REFERENCE_PATTERN_FLAT_BIT_KHR,
    eReferencePatternDyadic     = VK_VIDEO_ENCODE_AV1_RATE_CONTROL_REFERENCE_PATTERN_DYADIC_BIT_KHR
  };

  // wrapper using for bitmask VkVideoEncodeAV1RateControlFlagsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeAV1RateControlFlagsKHR.html
  using VideoEncodeAV1RateControlFlagsKHR = Flags<VideoEncodeAV1RateControlFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeAV1RateControlFlagBitsKHR>
  {
    using WrappedType                                                                = VkVideoEncodeAV1RateControlFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                              isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeAV1RateControlFlagsKHR allFlags =
      VideoEncodeAV1RateControlFlagBitsKHR::eRegularGop | VideoEncodeAV1RateControlFlagBitsKHR::eTemporalLayerPatternDyadic |
      VideoEncodeAV1RateControlFlagBitsKHR::eReferencePatternFlat | VideoEncodeAV1RateControlFlagBitsKHR::eReferencePatternDyadic;
  };

  //=== VK_QCOM_image_processing2 ===

  // wrapper class for enum VkBlockMatchWindowCompareModeQCOM, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkBlockMatchWindowCompareModeQCOM.html
  enum class BlockMatchWindowCompareModeQCOM
  {
    eMin = VK_BLOCK_MATCH_WINDOW_COMPARE_MODE_MIN_QCOM,
    eMax = VK_BLOCK_MATCH_WINDOW_COMPARE_MODE_MAX_QCOM
  };

  //=== VK_QCOM_filter_cubic_weights ===

  // wrapper class for enum VkCubicFilterWeightsQCOM, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkCubicFilterWeightsQCOM.html
  enum class CubicFilterWeightsQCOM
  {
    eCatmullRom          = VK_CUBIC_FILTER_WEIGHTS_CATMULL_ROM_QCOM,
    eZeroTangentCardinal = VK_CUBIC_FILTER_WEIGHTS_ZERO_TANGENT_CARDINAL_QCOM,
    eBSpline             = VK_CUBIC_FILTER_WEIGHTS_B_SPLINE_QCOM,
    eMitchellNetravali   = VK_CUBIC_FILTER_WEIGHTS_MITCHELL_NETRAVALI_QCOM
  };

  //=== VK_MSFT_layered_driver ===

  // wrapper class for enum VkLayeredDriverUnderlyingApiMSFT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkLayeredDriverUnderlyingApiMSFT.html
  enum class LayeredDriverUnderlyingApiMSFT
  {
    eNone  = VK_LAYERED_DRIVER_UNDERLYING_API_NONE_MSFT,
    eD3D12 = VK_LAYERED_DRIVER_UNDERLYING_API_D3D12_MSFT
  };

  //=== VK_KHR_calibrated_timestamps ===

  // wrapper class for enum VkTimeDomainKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkTimeDomainKHR.html
  enum class TimeDomainKHR
  {
    eDevice                  = VK_TIME_DOMAIN_DEVICE_KHR,
    eClockMonotonic          = VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR,
    eClockMonotonicRaw       = VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_KHR,
    eQueryPerformanceCounter = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR,
    ePresentStageLocalEXT    = VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT,
    eSwapchainLocalEXT       = VK_TIME_DOMAIN_SWAPCHAIN_LOCAL_EXT
  };

  using TimeDomainEXT = TimeDomainKHR;

  //=== VK_KHR_copy_memory_indirect ===

  // wrapper class for enum VkAddressCopyFlagBitsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkAddressCopyFlagBitsKHR.html
  enum class AddressCopyFlagBitsKHR : VkAddressCopyFlagsKHR
  {
    eDeviceLocal = VK_ADDRESS_COPY_DEVICE_LOCAL_BIT_KHR,
    eSparse      = VK_ADDRESS_COPY_SPARSE_BIT_KHR,
    eProtected   = VK_ADDRESS_COPY_PROTECTED_BIT_KHR
  };

  // wrapper using for bitmask VkAddressCopyFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkAddressCopyFlagsKHR.html
  using AddressCopyFlagsKHR = Flags<AddressCopyFlagBitsKHR>;

  template <>
  struct FlagTraits<AddressCopyFlagBitsKHR>
  {
    using WrappedType                                                  = VkAddressCopyFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR AddressCopyFlagsKHR allFlags =
      AddressCopyFlagBitsKHR::eDeviceLocal | AddressCopyFlagBitsKHR::eSparse | AddressCopyFlagBitsKHR::eProtected;
  };

  //=== VK_EXT_memory_decompression ===

  // wrapper class for enum VkMemoryDecompressionMethodFlagBitsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryDecompressionMethodFlagBitsEXT.html
  enum class MemoryDecompressionMethodFlagBitsEXT : VkMemoryDecompressionMethodFlagsEXT
  {
    eGdeflate10 = VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_EXT
  };

  using MemoryDecompressionMethodFlagBitsNV = MemoryDecompressionMethodFlagBitsEXT;

  // wrapper using for bitmask VkMemoryDecompressionMethodFlagsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkMemoryDecompressionMethodFlagsEXT.html
  using MemoryDecompressionMethodFlagsEXT = Flags<MemoryDecompressionMethodFlagBitsEXT>;
  using MemoryDecompressionMethodFlagsNV  = MemoryDecompressionMethodFlagsEXT;

  template <>
  struct FlagTraits<MemoryDecompressionMethodFlagBitsEXT>
  {
    using WrappedType                                                                = VkMemoryDecompressionMethodFlagBitsEXT;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                              isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR MemoryDecompressionMethodFlagsEXT allFlags  = MemoryDecompressionMethodFlagBitsEXT::eGdeflate10;
  };

  //=== VK_NV_display_stereo ===

  // wrapper class for enum VkDisplaySurfaceStereoTypeNV, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDisplaySurfaceStereoTypeNV.html
  enum class DisplaySurfaceStereoTypeNV
  {
    eNone              = VK_DISPLAY_SURFACE_STEREO_TYPE_NONE_NV,
    eOnboardDin        = VK_DISPLAY_SURFACE_STEREO_TYPE_ONBOARD_DIN_NV,
    eHdmi3D            = VK_DISPLAY_SURFACE_STEREO_TYPE_HDMI_3D_NV,
    eInbandDisplayport = VK_DISPLAY_SURFACE_STEREO_TYPE_INBAND_DISPLAYPORT_NV
  };

  //=== VK_KHR_video_encode_intra_refresh ===

  // wrapper class for enum VkVideoEncodeIntraRefreshModeFlagBitsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeIntraRefreshModeFlagBitsKHR.html
  enum class VideoEncodeIntraRefreshModeFlagBitsKHR : VkVideoEncodeIntraRefreshModeFlagsKHR
  {
    eNone                = VK_VIDEO_ENCODE_INTRA_REFRESH_MODE_NONE_KHR,
    ePerPicturePartition = VK_VIDEO_ENCODE_INTRA_REFRESH_MODE_PER_PICTURE_PARTITION_BIT_KHR,
    eBlockBased          = VK_VIDEO_ENCODE_INTRA_REFRESH_MODE_BLOCK_BASED_BIT_KHR,
    eBlockRowBased       = VK_VIDEO_ENCODE_INTRA_REFRESH_MODE_BLOCK_ROW_BASED_BIT_KHR,
    eBlockColumnBased    = VK_VIDEO_ENCODE_INTRA_REFRESH_MODE_BLOCK_COLUMN_BASED_BIT_KHR
  };

  // wrapper using for bitmask VkVideoEncodeIntraRefreshModeFlagsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkVideoEncodeIntraRefreshModeFlagsKHR.html
  using VideoEncodeIntraRefreshModeFlagsKHR = Flags<VideoEncodeIntraRefreshModeFlagBitsKHR>;

  template <>
  struct FlagTraits<VideoEncodeIntraRefreshModeFlagBitsKHR>
  {
    using WrappedType                                                                  = VkVideoEncodeIntraRefreshModeFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR VideoEncodeIntraRefreshModeFlagsKHR allFlags =
      VideoEncodeIntraRefreshModeFlagBitsKHR::eNone | VideoEncodeIntraRefreshModeFlagBitsKHR::ePerPicturePartition |
      VideoEncodeIntraRefreshModeFlagBitsKHR::eBlockBased | VideoEncodeIntraRefreshModeFlagBitsKHR::eBlockRowBased |
      VideoEncodeIntraRefreshModeFlagBitsKHR::eBlockColumnBased;
  };

  //=== VK_KHR_maintenance7 ===

  // wrapper class for enum VkPhysicalDeviceLayeredApiKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceLayeredApiKHR.html
  enum class PhysicalDeviceLayeredApiKHR
  {
    eVulkan   = VK_PHYSICAL_DEVICE_LAYERED_API_VULKAN_KHR,
    eD3D12    = VK_PHYSICAL_DEVICE_LAYERED_API_D3D12_KHR,
    eMetal    = VK_PHYSICAL_DEVICE_LAYERED_API_METAL_KHR,
    eOpengl   = VK_PHYSICAL_DEVICE_LAYERED_API_OPENGL_KHR,
    eOpengles = VK_PHYSICAL_DEVICE_LAYERED_API_OPENGLES_KHR
  };

  //=== VK_NV_cluster_acceleration_structure ===

  // wrapper class for enum VkClusterAccelerationStructureClusterFlagBitsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkClusterAccelerationStructureClusterFlagBitsNV.html
  enum class ClusterAccelerationStructureClusterFlagBitsNV : VkClusterAccelerationStructureClusterFlagsNV
  {
    eAllowDisableOpacityMicromaps = VK_CLUSTER_ACCELERATION_STRUCTURE_CLUSTER_ALLOW_DISABLE_OPACITY_MICROMAPS_NV
  };

  // wrapper using for bitmask VkClusterAccelerationStructureClusterFlagsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkClusterAccelerationStructureClusterFlagsNV.html
  using ClusterAccelerationStructureClusterFlagsNV = Flags<ClusterAccelerationStructureClusterFlagBitsNV>;

  template <>
  struct FlagTraits<ClusterAccelerationStructureClusterFlagBitsNV>
  {
    using WrappedType                                                                         = VkClusterAccelerationStructureClusterFlagBitsNV;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                       isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ClusterAccelerationStructureClusterFlagsNV allFlags =
      ClusterAccelerationStructureClusterFlagBitsNV::eAllowDisableOpacityMicromaps;
  };

  // wrapper class for enum VkClusterAccelerationStructureGeometryFlagBitsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkClusterAccelerationStructureGeometryFlagBitsNV.html
  enum class ClusterAccelerationStructureGeometryFlagBitsNV : VkClusterAccelerationStructureGeometryFlagsNV
  {
    eCullDisable                 = VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_CULL_DISABLE_BIT_NV,
    eNoDuplicateAnyhitInvocation = VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_NO_DUPLICATE_ANYHIT_INVOCATION_BIT_NV,
    eOpaque                      = VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV
  };

  // wrapper using for bitmask VkClusterAccelerationStructureGeometryFlagsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkClusterAccelerationStructureGeometryFlagsNV.html
  using ClusterAccelerationStructureGeometryFlagsNV = Flags<ClusterAccelerationStructureGeometryFlagBitsNV>;

  template <>
  struct FlagTraits<ClusterAccelerationStructureGeometryFlagBitsNV>
  {
    using WrappedType                                                                          = VkClusterAccelerationStructureGeometryFlagBitsNV;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                        isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ClusterAccelerationStructureGeometryFlagsNV allFlags =
      ClusterAccelerationStructureGeometryFlagBitsNV::eCullDisable | ClusterAccelerationStructureGeometryFlagBitsNV::eNoDuplicateAnyhitInvocation |
      ClusterAccelerationStructureGeometryFlagBitsNV::eOpaque;
  };

  // wrapper class for enum VkClusterAccelerationStructureAddressResolutionFlagBitsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkClusterAccelerationStructureAddressResolutionFlagBitsNV.html
  enum class ClusterAccelerationStructureAddressResolutionFlagBitsNV : VkClusterAccelerationStructureAddressResolutionFlagsNV
  {
    eNone                      = VK_CLUSTER_ACCELERATION_STRUCTURE_ADDRESS_RESOLUTION_NONE_NV,
    eIndirectedDstImplicitData = VK_CLUSTER_ACCELERATION_STRUCTURE_ADDRESS_RESOLUTION_INDIRECTED_DST_IMPLICIT_DATA_BIT_NV,
    eIndirectedScratchData     = VK_CLUSTER_ACCELERATION_STRUCTURE_ADDRESS_RESOLUTION_INDIRECTED_SCRATCH_DATA_BIT_NV,
    eIndirectedDstAddressArray = VK_CLUSTER_ACCELERATION_STRUCTURE_ADDRESS_RESOLUTION_INDIRECTED_DST_ADDRESS_ARRAY_BIT_NV,
    eIndirectedDstSizesArray   = VK_CLUSTER_ACCELERATION_STRUCTURE_ADDRESS_RESOLUTION_INDIRECTED_DST_SIZES_ARRAY_BIT_NV,
    eIndirectedSrcInfosArray   = VK_CLUSTER_ACCELERATION_STRUCTURE_ADDRESS_RESOLUTION_INDIRECTED_SRC_INFOS_ARRAY_BIT_NV,
    eIndirectedSrcInfosCount   = VK_CLUSTER_ACCELERATION_STRUCTURE_ADDRESS_RESOLUTION_INDIRECTED_SRC_INFOS_COUNT_BIT_NV
  };

  // wrapper using for bitmask VkClusterAccelerationStructureAddressResolutionFlagsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkClusterAccelerationStructureAddressResolutionFlagsNV.html
  using ClusterAccelerationStructureAddressResolutionFlagsNV = Flags<ClusterAccelerationStructureAddressResolutionFlagBitsNV>;

  template <>
  struct FlagTraits<ClusterAccelerationStructureAddressResolutionFlagBitsNV>
  {
    using WrappedType = VkClusterAccelerationStructureAddressResolutionFlagBitsNV;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ClusterAccelerationStructureAddressResolutionFlagsNV allFlags =
      ClusterAccelerationStructureAddressResolutionFlagBitsNV::eNone | ClusterAccelerationStructureAddressResolutionFlagBitsNV::eIndirectedDstImplicitData |
      ClusterAccelerationStructureAddressResolutionFlagBitsNV::eIndirectedScratchData |
      ClusterAccelerationStructureAddressResolutionFlagBitsNV::eIndirectedDstAddressArray |
      ClusterAccelerationStructureAddressResolutionFlagBitsNV::eIndirectedDstSizesArray |
      ClusterAccelerationStructureAddressResolutionFlagBitsNV::eIndirectedSrcInfosArray |
      ClusterAccelerationStructureAddressResolutionFlagBitsNV::eIndirectedSrcInfosCount;
  };

  // wrapper class for enum VkClusterAccelerationStructureIndexFormatFlagBitsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkClusterAccelerationStructureIndexFormatFlagBitsNV.html
  enum class ClusterAccelerationStructureIndexFormatFlagBitsNV : VkClusterAccelerationStructureIndexFormatFlagsNV
  {
    e8  = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV,
    e16 = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_16BIT_NV,
    e32 = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_32BIT_NV
  };

  // wrapper using for bitmask VkClusterAccelerationStructureIndexFormatFlagsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkClusterAccelerationStructureIndexFormatFlagsNV.html
  using ClusterAccelerationStructureIndexFormatFlagsNV = Flags<ClusterAccelerationStructureIndexFormatFlagBitsNV>;

  template <>
  struct FlagTraits<ClusterAccelerationStructureIndexFormatFlagBitsNV>
  {
    using WrappedType                                                                             = VkClusterAccelerationStructureIndexFormatFlagBitsNV;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                           isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ClusterAccelerationStructureIndexFormatFlagsNV allFlags  = ClusterAccelerationStructureIndexFormatFlagBitsNV::e8 |
                                                                                                   ClusterAccelerationStructureIndexFormatFlagBitsNV::e16 |
                                                                                                   ClusterAccelerationStructureIndexFormatFlagBitsNV::e32;
  };

  // wrapper class for enum VkClusterAccelerationStructureTypeNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkClusterAccelerationStructureTypeNV.html
  enum class ClusterAccelerationStructureTypeNV
  {
    eClustersBottomLevel     = VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_CLUSTERS_BOTTOM_LEVEL_NV,
    eTriangleCluster         = VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_TRIANGLE_CLUSTER_NV,
    eTriangleClusterTemplate = VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_TRIANGLE_CLUSTER_TEMPLATE_NV
  };

  // wrapper class for enum VkClusterAccelerationStructureOpTypeNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkClusterAccelerationStructureOpTypeNV.html
  enum class ClusterAccelerationStructureOpTypeNV
  {
    eMoveObjects                  = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_MOVE_OBJECTS_NV,
    eBuildClustersBottomLevel     = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV,
    eBuildTriangleCluster         = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV,
    eBuildTriangleClusterTemplate = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV,
    eInstantiateTriangleCluster   = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV,
    eGetClusterTemplateIndices    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_GET_CLUSTER_TEMPLATE_INDICES_NV
  };

  // wrapper class for enum VkClusterAccelerationStructureOpModeNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkClusterAccelerationStructureOpModeNV.html
  enum class ClusterAccelerationStructureOpModeNV
  {
    eImplicitDestinations = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV,
    eExplicitDestinations = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV,
    eComputeSizes         = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV
  };

  //=== VK_NV_partitioned_acceleration_structure ===

  // wrapper class for enum VkPartitionedAccelerationStructureOpTypeNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPartitionedAccelerationStructureOpTypeNV.html
  enum class PartitionedAccelerationStructureOpTypeNV
  {
    eWriteInstance             = VK_PARTITIONED_ACCELERATION_STRUCTURE_OP_TYPE_WRITE_INSTANCE_NV,
    eUpdateInstance            = VK_PARTITIONED_ACCELERATION_STRUCTURE_OP_TYPE_UPDATE_INSTANCE_NV,
    eWritePartitionTranslation = VK_PARTITIONED_ACCELERATION_STRUCTURE_OP_TYPE_WRITE_PARTITION_TRANSLATION_NV
  };

  // wrapper class for enum VkPartitionedAccelerationStructureInstanceFlagBitsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPartitionedAccelerationStructureInstanceFlagBitsNV.html
  enum class PartitionedAccelerationStructureInstanceFlagBitsNV : VkPartitionedAccelerationStructureInstanceFlagsNV
  {
    eFlagTriangleFacingCullDisable = VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_TRIANGLE_FACING_CULL_DISABLE_BIT_NV,
    eFlagTriangleFlipFacing        = VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_TRIANGLE_FLIP_FACING_BIT_NV,
    eFlagForceOpaque               = VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_FORCE_OPAQUE_BIT_NV,
    eFlagForceNoOpaque             = VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_FORCE_NO_OPAQUE_BIT_NV,
    eFlagEnableExplicitBoundingBox = VK_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCE_FLAG_ENABLE_EXPLICIT_BOUNDING_BOX_NV
  };

  // wrapper using for bitmask VkPartitionedAccelerationStructureInstanceFlagsNV, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPartitionedAccelerationStructureInstanceFlagsNV.html
  using PartitionedAccelerationStructureInstanceFlagsNV = Flags<PartitionedAccelerationStructureInstanceFlagBitsNV>;

  template <>
  struct FlagTraits<PartitionedAccelerationStructureInstanceFlagBitsNV>
  {
    using WrappedType                                                                              = VkPartitionedAccelerationStructureInstanceFlagBitsNV;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                            isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PartitionedAccelerationStructureInstanceFlagsNV allFlags =
      PartitionedAccelerationStructureInstanceFlagBitsNV::eFlagTriangleFacingCullDisable |
      PartitionedAccelerationStructureInstanceFlagBitsNV::eFlagTriangleFlipFacing | PartitionedAccelerationStructureInstanceFlagBitsNV::eFlagForceOpaque |
      PartitionedAccelerationStructureInstanceFlagBitsNV::eFlagForceNoOpaque |
      PartitionedAccelerationStructureInstanceFlagBitsNV::eFlagEnableExplicitBoundingBox;
  };

  //=== VK_EXT_device_generated_commands ===

  // wrapper class for enum VkIndirectCommandsTokenTypeEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkIndirectCommandsTokenTypeEXT.html
  enum class IndirectCommandsTokenTypeEXT
  {
    eExecutionSet         = VK_INDIRECT_COMMANDS_TOKEN_TYPE_EXECUTION_SET_EXT,
    ePushConstant         = VK_INDIRECT_COMMANDS_TOKEN_TYPE_PUSH_CONSTANT_EXT,
    eSequenceIndex        = VK_INDIRECT_COMMANDS_TOKEN_TYPE_SEQUENCE_INDEX_EXT,
    eIndexBuffer          = VK_INDIRECT_COMMANDS_TOKEN_TYPE_INDEX_BUFFER_EXT,
    eVertexBuffer         = VK_INDIRECT_COMMANDS_TOKEN_TYPE_VERTEX_BUFFER_EXT,
    eDrawIndexed          = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_INDEXED_EXT,
    eDraw                 = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_EXT,
    eDrawIndexedCount     = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_INDEXED_COUNT_EXT,
    eDrawCount            = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_COUNT_EXT,
    eDispatch             = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_EXT,
    eDrawMeshTasksNV      = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_MESH_TASKS_NV_EXT,
    eDrawMeshTasksCountNV = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_MESH_TASKS_COUNT_NV_EXT,
    eDrawMeshTasks        = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_MESH_TASKS_EXT,
    eDrawMeshTasksCount   = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_MESH_TASKS_COUNT_EXT,
    eTraceRays2           = VK_INDIRECT_COMMANDS_TOKEN_TYPE_TRACE_RAYS2_EXT
  };

  // wrapper class for enum VkIndirectExecutionSetInfoTypeEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkIndirectExecutionSetInfoTypeEXT.html
  enum class IndirectExecutionSetInfoTypeEXT
  {
    ePipelines     = VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT,
    eShaderObjects = VK_INDIRECT_EXECUTION_SET_INFO_TYPE_SHADER_OBJECTS_EXT
  };

  // wrapper class for enum VkIndirectCommandsLayoutUsageFlagBitsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkIndirectCommandsLayoutUsageFlagBitsEXT.html
  enum class IndirectCommandsLayoutUsageFlagBitsEXT : VkIndirectCommandsLayoutUsageFlagsEXT
  {
    eExplicitPreprocess = VK_INDIRECT_COMMANDS_LAYOUT_USAGE_EXPLICIT_PREPROCESS_BIT_EXT,
    eUnorderedSequences = VK_INDIRECT_COMMANDS_LAYOUT_USAGE_UNORDERED_SEQUENCES_BIT_EXT
  };

  // wrapper using for bitmask VkIndirectCommandsLayoutUsageFlagsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkIndirectCommandsLayoutUsageFlagsEXT.html
  using IndirectCommandsLayoutUsageFlagsEXT = Flags<IndirectCommandsLayoutUsageFlagBitsEXT>;

  template <>
  struct FlagTraits<IndirectCommandsLayoutUsageFlagBitsEXT>
  {
    using WrappedType                                                                  = VkIndirectCommandsLayoutUsageFlagBitsEXT;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR IndirectCommandsLayoutUsageFlagsEXT allFlags =
      IndirectCommandsLayoutUsageFlagBitsEXT::eExplicitPreprocess | IndirectCommandsLayoutUsageFlagBitsEXT::eUnorderedSequences;
  };

  // wrapper class for enum VkIndirectCommandsInputModeFlagBitsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkIndirectCommandsInputModeFlagBitsEXT.html
  enum class IndirectCommandsInputModeFlagBitsEXT : VkIndirectCommandsInputModeFlagsEXT
  {
    eVulkanIndexBuffer = VK_INDIRECT_COMMANDS_INPUT_MODE_VULKAN_INDEX_BUFFER_EXT,
    eDxgiIndexBuffer   = VK_INDIRECT_COMMANDS_INPUT_MODE_DXGI_INDEX_BUFFER_EXT
  };

  // wrapper using for bitmask VkIndirectCommandsInputModeFlagsEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkIndirectCommandsInputModeFlagsEXT.html
  using IndirectCommandsInputModeFlagsEXT = Flags<IndirectCommandsInputModeFlagBitsEXT>;

  template <>
  struct FlagTraits<IndirectCommandsInputModeFlagBitsEXT>
  {
    using WrappedType                                                                = VkIndirectCommandsInputModeFlagBitsEXT;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                              isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR IndirectCommandsInputModeFlagsEXT allFlags =
      IndirectCommandsInputModeFlagBitsEXT::eVulkanIndexBuffer | IndirectCommandsInputModeFlagBitsEXT::eDxgiIndexBuffer;
  };

  //=== VK_KHR_maintenance8 ===

  // wrapper class for enum VkAccessFlagBits3KHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccessFlagBits3KHR.html
  enum class AccessFlagBits3KHR : VkAccessFlags3KHR
  {
    eNone = VK_ACCESS_3_NONE_KHR
  };

  // wrapper using for bitmask VkAccessFlags3KHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkAccessFlags3KHR.html
  using AccessFlags3KHR = Flags<AccessFlagBits3KHR>;

  template <>
  struct FlagTraits<AccessFlagBits3KHR>
  {
    using WrappedType                                              = VkAccessFlagBits3KHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool            isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR AccessFlags3KHR allFlags  = AccessFlagBits3KHR::eNone;
  };

  //=== VK_EXT_ray_tracing_invocation_reorder ===

  // wrapper class for enum VkRayTracingInvocationReorderModeEXT, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkRayTracingInvocationReorderModeEXT.html
  enum class RayTracingInvocationReorderModeEXT
  {
    eNone    = VK_RAY_TRACING_INVOCATION_REORDER_MODE_NONE_EXT,
    eReorder = VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER_EXT
  };

  using RayTracingInvocationReorderModeNV = RayTracingInvocationReorderModeEXT;

  //=== VK_EXT_depth_clamp_control ===

  // wrapper class for enum VkDepthClampModeEXT, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDepthClampModeEXT.html
  enum class DepthClampModeEXT
  {
    eViewportRange    = VK_DEPTH_CLAMP_MODE_VIEWPORT_RANGE_EXT,
    eUserDefinedRange = VK_DEPTH_CLAMP_MODE_USER_DEFINED_RANGE_EXT
  };

  //=== VK_KHR_maintenance9 ===

  // wrapper class for enum VkDefaultVertexAttributeValueKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkDefaultVertexAttributeValueKHR.html
  enum class DefaultVertexAttributeValueKHR
  {
    eZeroZeroZeroZero = VK_DEFAULT_VERTEX_ATTRIBUTE_VALUE_ZERO_ZERO_ZERO_ZERO_KHR,
    eZeroZeroZeroOne  = VK_DEFAULT_VERTEX_ATTRIBUTE_VALUE_ZERO_ZERO_ZERO_ONE_KHR
  };

#if defined( VK_USE_PLATFORM_OHOS )
  //=== VK_OHOS_surface ===

  enum class SurfaceCreateFlagBitsOHOS : VkSurfaceCreateFlagsOHOS
  {
  };

  // wrapper using for bitmask VkSurfaceCreateFlagsOHOS, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSurfaceCreateFlagsOHOS.html
  using SurfaceCreateFlagsOHOS = Flags<SurfaceCreateFlagBitsOHOS>;

  template <>
  struct FlagTraits<SurfaceCreateFlagBitsOHOS>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                   isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SurfaceCreateFlagsOHOS allFlags  = {};
  };
#endif /*VK_USE_PLATFORM_OHOS*/

#if defined( VK_USE_PLATFORM_OHOS )
  //=== VK_OHOS_native_buffer ===

  // wrapper class for enum VkSwapchainImageUsageFlagBitsOHOS, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkSwapchainImageUsageFlagBitsOHOS.html
  enum class SwapchainImageUsageFlagBitsOHOS : VkSwapchainImageUsageFlagsOHOS
  {
    eShared = VK_SWAPCHAIN_IMAGE_USAGE_SHARED_BIT_OHOS
  };

  // wrapper using for bitmask VkSwapchainImageUsageFlagsOHOS, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkSwapchainImageUsageFlagsOHOS.html
  using SwapchainImageUsageFlagsOHOS = Flags<SwapchainImageUsageFlagBitsOHOS>;

  template <>
  struct FlagTraits<SwapchainImageUsageFlagBitsOHOS>
  {
    using WrappedType                                                           = VkSwapchainImageUsageFlagBitsOHOS;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                         isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR SwapchainImageUsageFlagsOHOS allFlags  = SwapchainImageUsageFlagBitsOHOS::eShared;
  };
#endif /*VK_USE_PLATFORM_OHOS*/

  //=== VK_ARM_performance_counters_by_region ===

  enum class PerformanceCounterDescriptionFlagBitsARM : VkPerformanceCounterDescriptionFlagsARM
  {
  };

  // wrapper using for bitmask VkPerformanceCounterDescriptionFlagsARM, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkPerformanceCounterDescriptionFlagsARM.html
  using PerformanceCounterDescriptionFlagsARM = Flags<PerformanceCounterDescriptionFlagBitsARM>;

  template <>
  struct FlagTraits<PerformanceCounterDescriptionFlagBitsARM>
  {
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                                  isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR PerformanceCounterDescriptionFlagsARM allFlags  = {};
  };

  //=== VK_QCOM_data_graph_model ===

  // wrapper class for enum VkDataGraphModelCacheTypeQCOM, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkDataGraphModelCacheTypeQCOM.html
  enum class DataGraphModelCacheTypeQCOM
  {
    eGenericBinary = VK_DATA_GRAPH_MODEL_CACHE_TYPE_GENERIC_BINARY_QCOM
  };

  //=== VK_KHR_maintenance10 ===

  // wrapper class for enum VkRenderingAttachmentFlagBitsKHR, see
  // https://registry.khronos.org/vulkan/specs/latest/man/html/VkRenderingAttachmentFlagBitsKHR.html
  enum class RenderingAttachmentFlagBitsKHR : VkRenderingAttachmentFlagsKHR
  {
    eInputAttachmentFeedback       = VK_RENDERING_ATTACHMENT_INPUT_ATTACHMENT_FEEDBACK_BIT_KHR,
    eResolveSkipTransferFunction   = VK_RENDERING_ATTACHMENT_RESOLVE_SKIP_TRANSFER_FUNCTION_BIT_KHR,
    eResolveEnableTransferFunction = VK_RENDERING_ATTACHMENT_RESOLVE_ENABLE_TRANSFER_FUNCTION_BIT_KHR
  };

  // wrapper using for bitmask VkRenderingAttachmentFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkRenderingAttachmentFlagsKHR.html
  using RenderingAttachmentFlagsKHR = Flags<RenderingAttachmentFlagBitsKHR>;

  template <>
  struct FlagTraits<RenderingAttachmentFlagBitsKHR>
  {
    using WrappedType                                                          = VkRenderingAttachmentFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                        isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR RenderingAttachmentFlagsKHR allFlags  = RenderingAttachmentFlagBitsKHR::eInputAttachmentFeedback |
                                                                                RenderingAttachmentFlagBitsKHR::eResolveSkipTransferFunction |
                                                                                RenderingAttachmentFlagBitsKHR::eResolveEnableTransferFunction;
  };

  // wrapper class for enum VkResolveImageFlagBitsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkResolveImageFlagBitsKHR.html
  enum class ResolveImageFlagBitsKHR : VkResolveImageFlagsKHR
  {
    eSkipTransferFunction   = VK_RESOLVE_IMAGE_SKIP_TRANSFER_FUNCTION_BIT_KHR,
    eEnableTransferFunction = VK_RESOLVE_IMAGE_ENABLE_TRANSFER_FUNCTION_BIT_KHR
  };

  // wrapper using for bitmask VkResolveImageFlagsKHR, see https://registry.khronos.org/vulkan/specs/latest/man/html/VkResolveImageFlagsKHR.html
  using ResolveImageFlagsKHR = Flags<ResolveImageFlagBitsKHR>;

  template <>
  struct FlagTraits<ResolveImageFlagBitsKHR>
  {
    using WrappedType                                                   = VkResolveImageFlagBitsKHR;
    static VULKAN_HPP_CONST_OR_CONSTEXPR bool                 isBitmask = true;
    static VULKAN_HPP_CONST_OR_CONSTEXPR ResolveImageFlagsKHR allFlags =
      ResolveImageFlagBitsKHR::eSkipTransferFunction | ResolveImageFlagBitsKHR::eEnableTransferFunction;
  };

  //===========================================================
  //=== Mapping from ObjectType to DebugReportObjectTypeEXT ===
  //===========================================================

  VULKAN_HPP_INLINE DebugReportObjectTypeEXT debugReportObjectType( ObjectType objectType )
  {
    switch ( objectType )
    {
        //=== VK_VERSION_1_0 ===
      case ObjectType::eInstance           : return DebugReportObjectTypeEXT::eInstance;
      case ObjectType::ePhysicalDevice     : return DebugReportObjectTypeEXT::ePhysicalDevice;
      case ObjectType::eDevice             : return DebugReportObjectTypeEXT::eDevice;
      case ObjectType::eQueue              : return DebugReportObjectTypeEXT::eQueue;
      case ObjectType::eDeviceMemory       : return DebugReportObjectTypeEXT::eDeviceMemory;
      case ObjectType::eFence              : return DebugReportObjectTypeEXT::eFence;
      case ObjectType::eSemaphore          : return DebugReportObjectTypeEXT::eSemaphore;
      case ObjectType::eQueryPool          : return DebugReportObjectTypeEXT::eQueryPool;
      case ObjectType::eBuffer             : return DebugReportObjectTypeEXT::eBuffer;
      case ObjectType::eImage              : return DebugReportObjectTypeEXT::eImage;
      case ObjectType::eImageView          : return DebugReportObjectTypeEXT::eImageView;
      case ObjectType::eCommandPool        : return DebugReportObjectTypeEXT::eCommandPool;
      case ObjectType::eCommandBuffer      : return DebugReportObjectTypeEXT::eCommandBuffer;
      case ObjectType::eEvent              : return DebugReportObjectTypeEXT::eEvent;
      case ObjectType::eBufferView         : return DebugReportObjectTypeEXT::eBufferView;
      case ObjectType::eShaderModule       : return DebugReportObjectTypeEXT::eShaderModule;
      case ObjectType::ePipelineCache      : return DebugReportObjectTypeEXT::ePipelineCache;
      case ObjectType::ePipeline           : return DebugReportObjectTypeEXT::ePipeline;
      case ObjectType::ePipelineLayout     : return DebugReportObjectTypeEXT::ePipelineLayout;
      case ObjectType::eSampler            : return DebugReportObjectTypeEXT::eSampler;
      case ObjectType::eDescriptorPool     : return DebugReportObjectTypeEXT::eDescriptorPool;
      case ObjectType::eDescriptorSet      : return DebugReportObjectTypeEXT::eDescriptorSet;
      case ObjectType::eDescriptorSetLayout: return DebugReportObjectTypeEXT::eDescriptorSetLayout;
      case ObjectType::eFramebuffer        : return DebugReportObjectTypeEXT::eFramebuffer;
      case ObjectType::eRenderPass:
        return DebugReportObjectTypeEXT::eRenderPass;

        //=== VK_VERSION_1_1 ===
      case ObjectType::eDescriptorUpdateTemplate: return DebugReportObjectTypeEXT::eDescriptorUpdateTemplate;
      case ObjectType::eSamplerYcbcrConversion:
        return DebugReportObjectTypeEXT::eSamplerYcbcrConversion;

        //=== VK_VERSION_1_3 ===
      case ObjectType::ePrivateDataSlot:
        return DebugReportObjectTypeEXT::eUnknown;

        //=== VK_KHR_surface ===
      case ObjectType::eSurfaceKHR:
        return DebugReportObjectTypeEXT::eSurfaceKHR;

        //=== VK_KHR_swapchain ===
      case ObjectType::eSwapchainKHR:
        return DebugReportObjectTypeEXT::eSwapchainKHR;

        //=== VK_KHR_display ===
      case ObjectType::eDisplayKHR: return DebugReportObjectTypeEXT::eDisplayKHR;
      case ObjectType::eDisplayModeKHR:
        return DebugReportObjectTypeEXT::eDisplayModeKHR;

        //=== VK_EXT_debug_report ===
      case ObjectType::eDebugReportCallbackEXT:
        return DebugReportObjectTypeEXT::eDebugReportCallbackEXT;

        //=== VK_KHR_video_queue ===
      case ObjectType::eVideoSessionKHR: return DebugReportObjectTypeEXT::eUnknown;
      case ObjectType::eVideoSessionParametersKHR:
        return DebugReportObjectTypeEXT::eUnknown;

        //=== VK_NVX_binary_import ===
      case ObjectType::eCuModuleNVX: return DebugReportObjectTypeEXT::eCuModuleNVX;
      case ObjectType::eCuFunctionNVX:
        return DebugReportObjectTypeEXT::eCuFunctionNVX;

        //=== VK_EXT_debug_utils ===
      case ObjectType::eDebugUtilsMessengerEXT:
        return DebugReportObjectTypeEXT::eUnknown;

        //=== VK_KHR_acceleration_structure ===
      case ObjectType::eAccelerationStructureKHR:
        return DebugReportObjectTypeEXT::eAccelerationStructureKHR;

        //=== VK_EXT_validation_cache ===
      case ObjectType::eValidationCacheEXT:
        return DebugReportObjectTypeEXT::eValidationCacheEXT;

        //=== VK_NV_ray_tracing ===
      case ObjectType::eAccelerationStructureNV:
        return DebugReportObjectTypeEXT::eAccelerationStructureNV;

        //=== VK_INTEL_performance_query ===
      case ObjectType::ePerformanceConfigurationINTEL:
        return DebugReportObjectTypeEXT::eUnknown;

        //=== VK_KHR_deferred_host_operations ===
      case ObjectType::eDeferredOperationKHR:
        return DebugReportObjectTypeEXT::eUnknown;

        //=== VK_NV_device_generated_commands ===
      case ObjectType::eIndirectCommandsLayoutNV: return DebugReportObjectTypeEXT::eUnknown;

#if defined( VK_ENABLE_BETA_EXTENSIONS )
        //=== VK_NV_cuda_kernel_launch ===
      case ObjectType::eCudaModuleNV  : return DebugReportObjectTypeEXT::eCudaModuleNV;
      case ObjectType::eCudaFunctionNV: return DebugReportObjectTypeEXT::eCudaFunctionNV;
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

#if defined( VK_USE_PLATFORM_FUCHSIA )
        //=== VK_FUCHSIA_buffer_collection ===
      case ObjectType::eBufferCollectionFUCHSIA: return DebugReportObjectTypeEXT::eBufferCollectionFUCHSIA;
#endif /*VK_USE_PLATFORM_FUCHSIA*/

        //=== VK_EXT_opacity_micromap ===
      case ObjectType::eMicromapEXT:
        return DebugReportObjectTypeEXT::eUnknown;

        //=== VK_ARM_tensors ===
      case ObjectType::eTensorARM: return DebugReportObjectTypeEXT::eUnknown;
      case ObjectType::eTensorViewARM:
        return DebugReportObjectTypeEXT::eUnknown;

        //=== VK_NV_optical_flow ===
      case ObjectType::eOpticalFlowSessionNV:
        return DebugReportObjectTypeEXT::eUnknown;

        //=== VK_EXT_shader_object ===
      case ObjectType::eShaderEXT:
        return DebugReportObjectTypeEXT::eUnknown;

        //=== VK_KHR_pipeline_binary ===
      case ObjectType::ePipelineBinaryKHR:
        return DebugReportObjectTypeEXT::eUnknown;

        //=== VK_ARM_data_graph ===
      case ObjectType::eDataGraphPipelineSessionARM:
        return DebugReportObjectTypeEXT::eUnknown;

        //=== VK_NV_external_compute_queue ===
      case ObjectType::eExternalComputeQueueNV:
        return DebugReportObjectTypeEXT::eUnknown;

        //=== VK_EXT_device_generated_commands ===
      case ObjectType::eIndirectCommandsLayoutEXT: return DebugReportObjectTypeEXT::eUnknown;
      case ObjectType::eIndirectExecutionSetEXT  : return DebugReportObjectTypeEXT::eUnknown;

      default: VULKAN_HPP_ASSERT( false && "unknown ObjectType" ); return DebugReportObjectTypeEXT::eUnknown;
    }
  }

}  // namespace VULKAN_HPP_NAMESPACE
#endif
