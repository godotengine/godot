// Copyright 2015-2022 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//

// This header is generated from the Khronos Vulkan XML API Registry.

#ifndef VULKAN_HASH_HPP
#define VULKAN_HASH_HPP

#include <vulkan/vulkan.hpp>

namespace std
{
  //=======================================
  //=== HASH structures for Flags types ===
  //=======================================

  template <typename BitType>
  struct hash<VULKAN_HPP_NAMESPACE::Flags<BitType>>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Flags<BitType> const & flags ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<typename std::underlying_type<BitType>::type>{}( static_cast<typename std::underlying_type<BitType>::type>( flags ) );
    }
  };

  //===================================
  //=== HASH structures for handles ===
  //===================================

  //=== VK_VERSION_1_0 ===

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Instance>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Instance const & instance ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkInstance>{}( static_cast<VkInstance>( instance ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevice>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDevice const & physicalDevice ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkPhysicalDevice>{}( static_cast<VkPhysicalDevice>( physicalDevice ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Device>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Device const & device ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkDevice>{}( static_cast<VkDevice>( device ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Queue>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Queue const & queue ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkQueue>{}( static_cast<VkQueue>( queue ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceMemory>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceMemory const & deviceMemory ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkDeviceMemory>{}( static_cast<VkDeviceMemory>( deviceMemory ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Fence>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Fence const & fence ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkFence>{}( static_cast<VkFence>( fence ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Semaphore>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Semaphore const & semaphore ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkSemaphore>{}( static_cast<VkSemaphore>( semaphore ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Event>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Event const & event ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkEvent>{}( static_cast<VkEvent>( event ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueryPool>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::QueryPool const & queryPool ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkQueryPool>{}( static_cast<VkQueryPool>( queryPool ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Buffer>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Buffer const & buffer ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkBuffer>{}( static_cast<VkBuffer>( buffer ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferView>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BufferView const & bufferView ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkBufferView>{}( static_cast<VkBufferView>( bufferView ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Image>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Image const & image ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkImage>{}( static_cast<VkImage>( image ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageView>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageView const & imageView ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkImageView>{}( static_cast<VkImageView>( imageView ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ShaderModule>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ShaderModule const & shaderModule ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkShaderModule>{}( static_cast<VkShaderModule>( shaderModule ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineCache>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineCache const & pipelineCache ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkPipelineCache>{}( static_cast<VkPipelineCache>( pipelineCache ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Pipeline>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Pipeline const & pipeline ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkPipeline>{}( static_cast<VkPipeline>( pipeline ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineLayout>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineLayout const & pipelineLayout ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkPipelineLayout>{}( static_cast<VkPipelineLayout>( pipelineLayout ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Sampler>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Sampler const & sampler ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkSampler>{}( static_cast<VkSampler>( sampler ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorPool>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DescriptorPool const & descriptorPool ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkDescriptorPool>{}( static_cast<VkDescriptorPool>( descriptorPool ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorSet>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DescriptorSet const & descriptorSet ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkDescriptorSet>{}( static_cast<VkDescriptorSet>( descriptorSet ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorSetLayout>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DescriptorSetLayout const & descriptorSetLayout ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkDescriptorSetLayout>{}( static_cast<VkDescriptorSetLayout>( descriptorSetLayout ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Framebuffer>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Framebuffer const & framebuffer ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkFramebuffer>{}( static_cast<VkFramebuffer>( framebuffer ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPass>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::RenderPass const & renderPass ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkRenderPass>{}( static_cast<VkRenderPass>( renderPass ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CommandPool>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CommandPool const & commandPool ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkCommandPool>{}( static_cast<VkCommandPool>( commandPool ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CommandBuffer>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CommandBuffer const & commandBuffer ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkCommandBuffer>{}( static_cast<VkCommandBuffer>( commandBuffer ) );
    }
  };

  //=== VK_VERSION_1_1 ===

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SamplerYcbcrConversion>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SamplerYcbcrConversion const & samplerYcbcrConversion ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkSamplerYcbcrConversion>{}( static_cast<VkSamplerYcbcrConversion>( samplerYcbcrConversion ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplate>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplate const & descriptorUpdateTemplate ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkDescriptorUpdateTemplate>{}( static_cast<VkDescriptorUpdateTemplate>( descriptorUpdateTemplate ) );
    }
  };

  //=== VK_VERSION_1_3 ===

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PrivateDataSlot>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PrivateDataSlot const & privateDataSlot ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkPrivateDataSlot>{}( static_cast<VkPrivateDataSlot>( privateDataSlot ) );
    }
  };

  //=== VK_KHR_surface ===

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SurfaceKHR const & surfaceKHR ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkSurfaceKHR>{}( static_cast<VkSurfaceKHR>( surfaceKHR ) );
    }
  };

  //=== VK_KHR_swapchain ===

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SwapchainKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SwapchainKHR const & swapchainKHR ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkSwapchainKHR>{}( static_cast<VkSwapchainKHR>( swapchainKHR ) );
    }
  };

  //=== VK_KHR_display ===

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DisplayKHR const & displayKHR ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkDisplayKHR>{}( static_cast<VkDisplayKHR>( displayKHR ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayModeKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DisplayModeKHR const & displayModeKHR ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkDisplayModeKHR>{}( static_cast<VkDisplayModeKHR>( displayModeKHR ) );
    }
  };

  //=== VK_EXT_debug_report ===

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DebugReportCallbackEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DebugReportCallbackEXT const & debugReportCallbackEXT ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkDebugReportCallbackEXT>{}( static_cast<VkDebugReportCallbackEXT>( debugReportCallbackEXT ) );
    }
  };

#if defined( VK_ENABLE_BETA_EXTENSIONS )
  //=== VK_KHR_video_queue ===

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoSessionKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoSessionKHR const & videoSessionKHR ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkVideoSessionKHR>{}( static_cast<VkVideoSessionKHR>( videoSessionKHR ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoSessionParametersKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoSessionParametersKHR const & videoSessionParametersKHR ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkVideoSessionParametersKHR>{}( static_cast<VkVideoSessionParametersKHR>( videoSessionParametersKHR ) );
    }
  };
#endif /*VK_ENABLE_BETA_EXTENSIONS*/

  //=== VK_NVX_binary_import ===

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CuModuleNVX>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CuModuleNVX const & cuModuleNVX ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkCuModuleNVX>{}( static_cast<VkCuModuleNVX>( cuModuleNVX ) );
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CuFunctionNVX>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CuFunctionNVX const & cuFunctionNVX ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkCuFunctionNVX>{}( static_cast<VkCuFunctionNVX>( cuFunctionNVX ) );
    }
  };

  //=== VK_EXT_debug_utils ===

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DebugUtilsMessengerEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DebugUtilsMessengerEXT const & debugUtilsMessengerEXT ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkDebugUtilsMessengerEXT>{}( static_cast<VkDebugUtilsMessengerEXT>( debugUtilsMessengerEXT ) );
    }
  };

  //=== VK_KHR_acceleration_structure ===

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AccelerationStructureKHR const & accelerationStructureKHR ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkAccelerationStructureKHR>{}( static_cast<VkAccelerationStructureKHR>( accelerationStructureKHR ) );
    }
  };

  //=== VK_EXT_validation_cache ===

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ValidationCacheEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ValidationCacheEXT const & validationCacheEXT ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkValidationCacheEXT>{}( static_cast<VkValidationCacheEXT>( validationCacheEXT ) );
    }
  };

  //=== VK_NV_ray_tracing ===

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AccelerationStructureNV const & accelerationStructureNV ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkAccelerationStructureNV>{}( static_cast<VkAccelerationStructureNV>( accelerationStructureNV ) );
    }
  };

  //=== VK_INTEL_performance_query ===

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PerformanceConfigurationINTEL>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PerformanceConfigurationINTEL const & performanceConfigurationINTEL ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkPerformanceConfigurationINTEL>{}( static_cast<VkPerformanceConfigurationINTEL>( performanceConfigurationINTEL ) );
    }
  };

  //=== VK_KHR_deferred_host_operations ===

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeferredOperationKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeferredOperationKHR const & deferredOperationKHR ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkDeferredOperationKHR>{}( static_cast<VkDeferredOperationKHR>( deferredOperationKHR ) );
    }
  };

  //=== VK_NV_device_generated_commands ===

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutNV const & indirectCommandsLayoutNV ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkIndirectCommandsLayoutNV>{}( static_cast<VkIndirectCommandsLayoutNV>( indirectCommandsLayoutNV ) );
    }
  };

#if defined( VK_USE_PLATFORM_FUCHSIA )
  //=== VK_FUCHSIA_buffer_collection ===

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferCollectionFUCHSIA>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BufferCollectionFUCHSIA const & bufferCollectionFUCHSIA ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkBufferCollectionFUCHSIA>{}( static_cast<VkBufferCollectionFUCHSIA>( bufferCollectionFUCHSIA ) );
    }
  };
#endif /*VK_USE_PLATFORM_FUCHSIA*/

  //=== VK_EXT_opacity_micromap ===

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MicromapEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MicromapEXT const & micromapEXT ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkMicromapEXT>{}( static_cast<VkMicromapEXT>( micromapEXT ) );
    }
  };

  //=== VK_NV_optical_flow ===

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::OpticalFlowSessionNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::OpticalFlowSessionNV const & opticalFlowSessionNV ) const VULKAN_HPP_NOEXCEPT
    {
      return std::hash<VkOpticalFlowSessionNV>{}( static_cast<VkOpticalFlowSessionNV>( opticalFlowSessionNV ) );
    }
  };

#if 14 <= VULKAN_HPP_CPP_VERSION
  //======================================
  //=== HASH structures for structures ===
  //======================================

#  if !defined( VULKAN_HPP_HASH_COMBINE )
#    define VULKAN_HPP_HASH_COMBINE( seed, value ) \
      seed ^= std::hash<std::decay<decltype( value )>::type>{}( value ) + 0x9e3779b9 + ( seed << 6 ) + ( seed >> 2 )
#  endif

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AabbPositionsKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AabbPositionsKHR const & aabbPositionsKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, aabbPositionsKHR.minX );
      VULKAN_HPP_HASH_COMBINE( seed, aabbPositionsKHR.minY );
      VULKAN_HPP_HASH_COMBINE( seed, aabbPositionsKHR.minZ );
      VULKAN_HPP_HASH_COMBINE( seed, aabbPositionsKHR.maxX );
      VULKAN_HPP_HASH_COMBINE( seed, aabbPositionsKHR.maxY );
      VULKAN_HPP_HASH_COMBINE( seed, aabbPositionsKHR.maxZ );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureBuildRangeInfoKHR>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::AccelerationStructureBuildRangeInfoKHR const & accelerationStructureBuildRangeInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureBuildRangeInfoKHR.primitiveCount );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureBuildRangeInfoKHR.primitiveOffset );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureBuildRangeInfoKHR.firstVertex );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureBuildRangeInfoKHR.transformOffset );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureBuildSizesInfoKHR>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::AccelerationStructureBuildSizesInfoKHR const & accelerationStructureBuildSizesInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureBuildSizesInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureBuildSizesInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureBuildSizesInfoKHR.accelerationStructureSize );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureBuildSizesInfoKHR.updateScratchSize );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureBuildSizesInfoKHR.buildScratchSize );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureCreateInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AccelerationStructureCreateInfoKHR const & accelerationStructureCreateInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureCreateInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureCreateInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureCreateInfoKHR.createFlags );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureCreateInfoKHR.buffer );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureCreateInfoKHR.offset );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureCreateInfoKHR.size );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureCreateInfoKHR.type );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureCreateInfoKHR.deviceAddress );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GeometryTrianglesNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::GeometryTrianglesNV const & geometryTrianglesNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, geometryTrianglesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, geometryTrianglesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, geometryTrianglesNV.vertexData );
      VULKAN_HPP_HASH_COMBINE( seed, geometryTrianglesNV.vertexOffset );
      VULKAN_HPP_HASH_COMBINE( seed, geometryTrianglesNV.vertexCount );
      VULKAN_HPP_HASH_COMBINE( seed, geometryTrianglesNV.vertexStride );
      VULKAN_HPP_HASH_COMBINE( seed, geometryTrianglesNV.vertexFormat );
      VULKAN_HPP_HASH_COMBINE( seed, geometryTrianglesNV.indexData );
      VULKAN_HPP_HASH_COMBINE( seed, geometryTrianglesNV.indexOffset );
      VULKAN_HPP_HASH_COMBINE( seed, geometryTrianglesNV.indexCount );
      VULKAN_HPP_HASH_COMBINE( seed, geometryTrianglesNV.indexType );
      VULKAN_HPP_HASH_COMBINE( seed, geometryTrianglesNV.transformData );
      VULKAN_HPP_HASH_COMBINE( seed, geometryTrianglesNV.transformOffset );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GeometryAABBNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::GeometryAABBNV const & geometryAABBNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, geometryAABBNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, geometryAABBNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, geometryAABBNV.aabbData );
      VULKAN_HPP_HASH_COMBINE( seed, geometryAABBNV.numAABBs );
      VULKAN_HPP_HASH_COMBINE( seed, geometryAABBNV.stride );
      VULKAN_HPP_HASH_COMBINE( seed, geometryAABBNV.offset );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GeometryDataNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::GeometryDataNV const & geometryDataNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, geometryDataNV.triangles );
      VULKAN_HPP_HASH_COMBINE( seed, geometryDataNV.aabbs );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GeometryNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::GeometryNV const & geometryNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, geometryNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, geometryNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, geometryNV.geometryType );
      VULKAN_HPP_HASH_COMBINE( seed, geometryNV.geometry );
      VULKAN_HPP_HASH_COMBINE( seed, geometryNV.flags );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AccelerationStructureInfoNV const & accelerationStructureInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureInfoNV.type );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureInfoNV.flags );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureInfoNV.instanceCount );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureInfoNV.geometryCount );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureInfoNV.pGeometries );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureCreateInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AccelerationStructureCreateInfoNV const & accelerationStructureCreateInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureCreateInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureCreateInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureCreateInfoNV.compactedSize );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureCreateInfoNV.info );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureDeviceAddressInfoKHR>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::AccelerationStructureDeviceAddressInfoKHR const & accelerationStructureDeviceAddressInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureDeviceAddressInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureDeviceAddressInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureDeviceAddressInfoKHR.accelerationStructure );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TransformMatrixKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::TransformMatrixKHR const & transformMatrixKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      for ( size_t i = 0; i < 3; ++i )
      {
        for ( size_t j = 0; j < 4; ++j )
        {
          VULKAN_HPP_HASH_COMBINE( seed, transformMatrixKHR.matrix[i][j] );
        }
      }
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureInstanceKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AccelerationStructureInstanceKHR const & accelerationStructureInstanceKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureInstanceKHR.transform );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureInstanceKHR.instanceCustomIndex );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureInstanceKHR.mask );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureInstanceKHR.instanceShaderBindingTableRecordOffset );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureInstanceKHR.flags );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureInstanceKHR.accelerationStructureReference );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureMatrixMotionInstanceNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AccelerationStructureMatrixMotionInstanceNV const & accelerationStructureMatrixMotionInstanceNV ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureMatrixMotionInstanceNV.transformT0 );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureMatrixMotionInstanceNV.transformT1 );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureMatrixMotionInstanceNV.instanceCustomIndex );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureMatrixMotionInstanceNV.mask );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureMatrixMotionInstanceNV.instanceShaderBindingTableRecordOffset );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureMatrixMotionInstanceNV.flags );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureMatrixMotionInstanceNV.accelerationStructureReference );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureMemoryRequirementsInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AccelerationStructureMemoryRequirementsInfoNV const & accelerationStructureMemoryRequirementsInfoNV ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureMemoryRequirementsInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureMemoryRequirementsInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureMemoryRequirementsInfoNV.type );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureMemoryRequirementsInfoNV.accelerationStructure );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureMotionInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AccelerationStructureMotionInfoNV const & accelerationStructureMotionInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureMotionInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureMotionInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureMotionInfoNV.maxInstances );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureMotionInfoNV.flags );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SRTDataNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SRTDataNV const & sRTDataNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, sRTDataNV.sx );
      VULKAN_HPP_HASH_COMBINE( seed, sRTDataNV.a );
      VULKAN_HPP_HASH_COMBINE( seed, sRTDataNV.b );
      VULKAN_HPP_HASH_COMBINE( seed, sRTDataNV.pvx );
      VULKAN_HPP_HASH_COMBINE( seed, sRTDataNV.sy );
      VULKAN_HPP_HASH_COMBINE( seed, sRTDataNV.c );
      VULKAN_HPP_HASH_COMBINE( seed, sRTDataNV.pvy );
      VULKAN_HPP_HASH_COMBINE( seed, sRTDataNV.sz );
      VULKAN_HPP_HASH_COMBINE( seed, sRTDataNV.pvz );
      VULKAN_HPP_HASH_COMBINE( seed, sRTDataNV.qx );
      VULKAN_HPP_HASH_COMBINE( seed, sRTDataNV.qy );
      VULKAN_HPP_HASH_COMBINE( seed, sRTDataNV.qz );
      VULKAN_HPP_HASH_COMBINE( seed, sRTDataNV.qw );
      VULKAN_HPP_HASH_COMBINE( seed, sRTDataNV.tx );
      VULKAN_HPP_HASH_COMBINE( seed, sRTDataNV.ty );
      VULKAN_HPP_HASH_COMBINE( seed, sRTDataNV.tz );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureSRTMotionInstanceNV>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::AccelerationStructureSRTMotionInstanceNV const & accelerationStructureSRTMotionInstanceNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureSRTMotionInstanceNV.transformT0 );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureSRTMotionInstanceNV.transformT1 );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureSRTMotionInstanceNV.instanceCustomIndex );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureSRTMotionInstanceNV.mask );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureSRTMotionInstanceNV.instanceShaderBindingTableRecordOffset );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureSRTMotionInstanceNV.flags );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureSRTMotionInstanceNV.accelerationStructureReference );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MicromapUsageEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MicromapUsageEXT const & micromapUsageEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, micromapUsageEXT.count );
      VULKAN_HPP_HASH_COMBINE( seed, micromapUsageEXT.subdivisionLevel );
      VULKAN_HPP_HASH_COMBINE( seed, micromapUsageEXT.format );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AccelerationStructureVersionInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AccelerationStructureVersionInfoKHR const & accelerationStructureVersionInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureVersionInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureVersionInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, accelerationStructureVersionInfoKHR.pVersionData );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AcquireNextImageInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AcquireNextImageInfoKHR const & acquireNextImageInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, acquireNextImageInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, acquireNextImageInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, acquireNextImageInfoKHR.swapchain );
      VULKAN_HPP_HASH_COMBINE( seed, acquireNextImageInfoKHR.timeout );
      VULKAN_HPP_HASH_COMBINE( seed, acquireNextImageInfoKHR.semaphore );
      VULKAN_HPP_HASH_COMBINE( seed, acquireNextImageInfoKHR.fence );
      VULKAN_HPP_HASH_COMBINE( seed, acquireNextImageInfoKHR.deviceMask );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AcquireProfilingLockInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AcquireProfilingLockInfoKHR const & acquireProfilingLockInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, acquireProfilingLockInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, acquireProfilingLockInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, acquireProfilingLockInfoKHR.flags );
      VULKAN_HPP_HASH_COMBINE( seed, acquireProfilingLockInfoKHR.timeout );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AllocationCallbacks>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AllocationCallbacks const & allocationCallbacks ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, allocationCallbacks.pUserData );
      VULKAN_HPP_HASH_COMBINE( seed, allocationCallbacks.pfnAllocation );
      VULKAN_HPP_HASH_COMBINE( seed, allocationCallbacks.pfnReallocation );
      VULKAN_HPP_HASH_COMBINE( seed, allocationCallbacks.pfnFree );
      VULKAN_HPP_HASH_COMBINE( seed, allocationCallbacks.pfnInternalAllocation );
      VULKAN_HPP_HASH_COMBINE( seed, allocationCallbacks.pfnInternalFree );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AmigoProfilingSubmitInfoSEC>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AmigoProfilingSubmitInfoSEC const & amigoProfilingSubmitInfoSEC ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, amigoProfilingSubmitInfoSEC.sType );
      VULKAN_HPP_HASH_COMBINE( seed, amigoProfilingSubmitInfoSEC.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, amigoProfilingSubmitInfoSEC.firstDrawTimestamp );
      VULKAN_HPP_HASH_COMBINE( seed, amigoProfilingSubmitInfoSEC.swapBufferTimestamp );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ComponentMapping>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ComponentMapping const & componentMapping ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, componentMapping.r );
      VULKAN_HPP_HASH_COMBINE( seed, componentMapping.g );
      VULKAN_HPP_HASH_COMBINE( seed, componentMapping.b );
      VULKAN_HPP_HASH_COMBINE( seed, componentMapping.a );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_ANDROID_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AndroidHardwareBufferFormatProperties2ANDROID>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AndroidHardwareBufferFormatProperties2ANDROID const & androidHardwareBufferFormatProperties2ANDROID ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferFormatProperties2ANDROID.sType );
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferFormatProperties2ANDROID.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferFormatProperties2ANDROID.format );
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferFormatProperties2ANDROID.externalFormat );
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferFormatProperties2ANDROID.formatFeatures );
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferFormatProperties2ANDROID.samplerYcbcrConversionComponents );
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferFormatProperties2ANDROID.suggestedYcbcrModel );
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferFormatProperties2ANDROID.suggestedYcbcrRange );
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferFormatProperties2ANDROID.suggestedXChromaOffset );
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferFormatProperties2ANDROID.suggestedYChromaOffset );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#  if defined( VK_USE_PLATFORM_ANDROID_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AndroidHardwareBufferFormatPropertiesANDROID>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AndroidHardwareBufferFormatPropertiesANDROID const & androidHardwareBufferFormatPropertiesANDROID ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferFormatPropertiesANDROID.sType );
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferFormatPropertiesANDROID.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferFormatPropertiesANDROID.format );
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferFormatPropertiesANDROID.externalFormat );
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferFormatPropertiesANDROID.formatFeatures );
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferFormatPropertiesANDROID.samplerYcbcrConversionComponents );
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferFormatPropertiesANDROID.suggestedYcbcrModel );
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferFormatPropertiesANDROID.suggestedYcbcrRange );
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferFormatPropertiesANDROID.suggestedXChromaOffset );
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferFormatPropertiesANDROID.suggestedYChromaOffset );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#  if defined( VK_USE_PLATFORM_ANDROID_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AndroidHardwareBufferPropertiesANDROID>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::AndroidHardwareBufferPropertiesANDROID const & androidHardwareBufferPropertiesANDROID ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferPropertiesANDROID.sType );
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferPropertiesANDROID.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferPropertiesANDROID.allocationSize );
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferPropertiesANDROID.memoryTypeBits );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#  if defined( VK_USE_PLATFORM_ANDROID_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AndroidHardwareBufferUsageANDROID>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AndroidHardwareBufferUsageANDROID const & androidHardwareBufferUsageANDROID ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferUsageANDROID.sType );
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferUsageANDROID.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, androidHardwareBufferUsageANDROID.androidHardwareBufferUsage );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#  if defined( VK_USE_PLATFORM_ANDROID_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AndroidSurfaceCreateInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AndroidSurfaceCreateInfoKHR const & androidSurfaceCreateInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, androidSurfaceCreateInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, androidSurfaceCreateInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, androidSurfaceCreateInfoKHR.flags );
      VULKAN_HPP_HASH_COMBINE( seed, androidSurfaceCreateInfoKHR.window );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_ANDROID_KHR*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ApplicationInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ApplicationInfo const & applicationInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, applicationInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, applicationInfo.pNext );
      for ( const char * p = applicationInfo.pApplicationName; *p != '\0'; ++p )
      {
        VULKAN_HPP_HASH_COMBINE( seed, *p );
      }
      VULKAN_HPP_HASH_COMBINE( seed, applicationInfo.applicationVersion );
      for ( const char * p = applicationInfo.pEngineName; *p != '\0'; ++p )
      {
        VULKAN_HPP_HASH_COMBINE( seed, *p );
      }
      VULKAN_HPP_HASH_COMBINE( seed, applicationInfo.engineVersion );
      VULKAN_HPP_HASH_COMBINE( seed, applicationInfo.apiVersion );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AttachmentDescription>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AttachmentDescription const & attachmentDescription ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, attachmentDescription.flags );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentDescription.format );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentDescription.samples );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentDescription.loadOp );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentDescription.storeOp );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentDescription.stencilLoadOp );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentDescription.stencilStoreOp );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentDescription.initialLayout );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentDescription.finalLayout );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AttachmentDescription2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AttachmentDescription2 const & attachmentDescription2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, attachmentDescription2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentDescription2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentDescription2.flags );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentDescription2.format );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentDescription2.samples );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentDescription2.loadOp );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentDescription2.storeOp );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentDescription2.stencilLoadOp );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentDescription2.stencilStoreOp );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentDescription2.initialLayout );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentDescription2.finalLayout );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AttachmentDescriptionStencilLayout>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AttachmentDescriptionStencilLayout const & attachmentDescriptionStencilLayout ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, attachmentDescriptionStencilLayout.sType );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentDescriptionStencilLayout.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentDescriptionStencilLayout.stencilInitialLayout );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentDescriptionStencilLayout.stencilFinalLayout );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AttachmentReference>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AttachmentReference const & attachmentReference ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, attachmentReference.attachment );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentReference.layout );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AttachmentReference2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AttachmentReference2 const & attachmentReference2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, attachmentReference2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentReference2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentReference2.attachment );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentReference2.layout );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentReference2.aspectMask );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AttachmentReferenceStencilLayout>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AttachmentReferenceStencilLayout const & attachmentReferenceStencilLayout ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, attachmentReferenceStencilLayout.sType );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentReferenceStencilLayout.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentReferenceStencilLayout.stencilLayout );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AttachmentSampleCountInfoAMD>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AttachmentSampleCountInfoAMD const & attachmentSampleCountInfoAMD ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, attachmentSampleCountInfoAMD.sType );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentSampleCountInfoAMD.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentSampleCountInfoAMD.colorAttachmentCount );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentSampleCountInfoAMD.pColorAttachmentSamples );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentSampleCountInfoAMD.depthStencilAttachmentSamples );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Extent2D>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Extent2D const & extent2D ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, extent2D.width );
      VULKAN_HPP_HASH_COMBINE( seed, extent2D.height );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SampleLocationEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SampleLocationEXT const & sampleLocationEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, sampleLocationEXT.x );
      VULKAN_HPP_HASH_COMBINE( seed, sampleLocationEXT.y );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SampleLocationsInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SampleLocationsInfoEXT const & sampleLocationsInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, sampleLocationsInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, sampleLocationsInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, sampleLocationsInfoEXT.sampleLocationsPerPixel );
      VULKAN_HPP_HASH_COMBINE( seed, sampleLocationsInfoEXT.sampleLocationGridSize );
      VULKAN_HPP_HASH_COMBINE( seed, sampleLocationsInfoEXT.sampleLocationsCount );
      VULKAN_HPP_HASH_COMBINE( seed, sampleLocationsInfoEXT.pSampleLocations );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::AttachmentSampleLocationsEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::AttachmentSampleLocationsEXT const & attachmentSampleLocationsEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, attachmentSampleLocationsEXT.attachmentIndex );
      VULKAN_HPP_HASH_COMBINE( seed, attachmentSampleLocationsEXT.sampleLocationsInfo );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BaseInStructure>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BaseInStructure const & baseInStructure ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, baseInStructure.sType );
      VULKAN_HPP_HASH_COMBINE( seed, baseInStructure.pNext );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BaseOutStructure>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BaseOutStructure const & baseOutStructure ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, baseOutStructure.sType );
      VULKAN_HPP_HASH_COMBINE( seed, baseOutStructure.pNext );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindAccelerationStructureMemoryInfoNV>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::BindAccelerationStructureMemoryInfoNV const & bindAccelerationStructureMemoryInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bindAccelerationStructureMemoryInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bindAccelerationStructureMemoryInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bindAccelerationStructureMemoryInfoNV.accelerationStructure );
      VULKAN_HPP_HASH_COMBINE( seed, bindAccelerationStructureMemoryInfoNV.memory );
      VULKAN_HPP_HASH_COMBINE( seed, bindAccelerationStructureMemoryInfoNV.memoryOffset );
      VULKAN_HPP_HASH_COMBINE( seed, bindAccelerationStructureMemoryInfoNV.deviceIndexCount );
      VULKAN_HPP_HASH_COMBINE( seed, bindAccelerationStructureMemoryInfoNV.pDeviceIndices );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindBufferMemoryDeviceGroupInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BindBufferMemoryDeviceGroupInfo const & bindBufferMemoryDeviceGroupInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bindBufferMemoryDeviceGroupInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bindBufferMemoryDeviceGroupInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bindBufferMemoryDeviceGroupInfo.deviceIndexCount );
      VULKAN_HPP_HASH_COMBINE( seed, bindBufferMemoryDeviceGroupInfo.pDeviceIndices );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindBufferMemoryInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BindBufferMemoryInfo const & bindBufferMemoryInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bindBufferMemoryInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bindBufferMemoryInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bindBufferMemoryInfo.buffer );
      VULKAN_HPP_HASH_COMBINE( seed, bindBufferMemoryInfo.memory );
      VULKAN_HPP_HASH_COMBINE( seed, bindBufferMemoryInfo.memoryOffset );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Offset2D>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Offset2D const & offset2D ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, offset2D.x );
      VULKAN_HPP_HASH_COMBINE( seed, offset2D.y );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Rect2D>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Rect2D const & rect2D ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, rect2D.offset );
      VULKAN_HPP_HASH_COMBINE( seed, rect2D.extent );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindImageMemoryDeviceGroupInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BindImageMemoryDeviceGroupInfo const & bindImageMemoryDeviceGroupInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bindImageMemoryDeviceGroupInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bindImageMemoryDeviceGroupInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bindImageMemoryDeviceGroupInfo.deviceIndexCount );
      VULKAN_HPP_HASH_COMBINE( seed, bindImageMemoryDeviceGroupInfo.pDeviceIndices );
      VULKAN_HPP_HASH_COMBINE( seed, bindImageMemoryDeviceGroupInfo.splitInstanceBindRegionCount );
      VULKAN_HPP_HASH_COMBINE( seed, bindImageMemoryDeviceGroupInfo.pSplitInstanceBindRegions );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindImageMemoryInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BindImageMemoryInfo const & bindImageMemoryInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bindImageMemoryInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bindImageMemoryInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bindImageMemoryInfo.image );
      VULKAN_HPP_HASH_COMBINE( seed, bindImageMemoryInfo.memory );
      VULKAN_HPP_HASH_COMBINE( seed, bindImageMemoryInfo.memoryOffset );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindImageMemorySwapchainInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BindImageMemorySwapchainInfoKHR const & bindImageMemorySwapchainInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bindImageMemorySwapchainInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bindImageMemorySwapchainInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bindImageMemorySwapchainInfoKHR.swapchain );
      VULKAN_HPP_HASH_COMBINE( seed, bindImageMemorySwapchainInfoKHR.imageIndex );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindImagePlaneMemoryInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BindImagePlaneMemoryInfo const & bindImagePlaneMemoryInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bindImagePlaneMemoryInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bindImagePlaneMemoryInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bindImagePlaneMemoryInfo.planeAspect );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindIndexBufferIndirectCommandNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BindIndexBufferIndirectCommandNV const & bindIndexBufferIndirectCommandNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bindIndexBufferIndirectCommandNV.bufferAddress );
      VULKAN_HPP_HASH_COMBINE( seed, bindIndexBufferIndirectCommandNV.size );
      VULKAN_HPP_HASH_COMBINE( seed, bindIndexBufferIndirectCommandNV.indexType );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindShaderGroupIndirectCommandNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BindShaderGroupIndirectCommandNV const & bindShaderGroupIndirectCommandNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bindShaderGroupIndirectCommandNV.groupIndex );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SparseMemoryBind>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SparseMemoryBind const & sparseMemoryBind ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, sparseMemoryBind.resourceOffset );
      VULKAN_HPP_HASH_COMBINE( seed, sparseMemoryBind.size );
      VULKAN_HPP_HASH_COMBINE( seed, sparseMemoryBind.memory );
      VULKAN_HPP_HASH_COMBINE( seed, sparseMemoryBind.memoryOffset );
      VULKAN_HPP_HASH_COMBINE( seed, sparseMemoryBind.flags );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SparseBufferMemoryBindInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SparseBufferMemoryBindInfo const & sparseBufferMemoryBindInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, sparseBufferMemoryBindInfo.buffer );
      VULKAN_HPP_HASH_COMBINE( seed, sparseBufferMemoryBindInfo.bindCount );
      VULKAN_HPP_HASH_COMBINE( seed, sparseBufferMemoryBindInfo.pBinds );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SparseImageOpaqueMemoryBindInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SparseImageOpaqueMemoryBindInfo const & sparseImageOpaqueMemoryBindInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageOpaqueMemoryBindInfo.image );
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageOpaqueMemoryBindInfo.bindCount );
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageOpaqueMemoryBindInfo.pBinds );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageSubresource>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageSubresource const & imageSubresource ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageSubresource.aspectMask );
      VULKAN_HPP_HASH_COMBINE( seed, imageSubresource.mipLevel );
      VULKAN_HPP_HASH_COMBINE( seed, imageSubresource.arrayLayer );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Offset3D>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Offset3D const & offset3D ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, offset3D.x );
      VULKAN_HPP_HASH_COMBINE( seed, offset3D.y );
      VULKAN_HPP_HASH_COMBINE( seed, offset3D.z );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Extent3D>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Extent3D const & extent3D ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, extent3D.width );
      VULKAN_HPP_HASH_COMBINE( seed, extent3D.height );
      VULKAN_HPP_HASH_COMBINE( seed, extent3D.depth );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SparseImageMemoryBind>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SparseImageMemoryBind const & sparseImageMemoryBind ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageMemoryBind.subresource );
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageMemoryBind.offset );
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageMemoryBind.extent );
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageMemoryBind.memory );
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageMemoryBind.memoryOffset );
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageMemoryBind.flags );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SparseImageMemoryBindInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SparseImageMemoryBindInfo const & sparseImageMemoryBindInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageMemoryBindInfo.image );
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageMemoryBindInfo.bindCount );
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageMemoryBindInfo.pBinds );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindSparseInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BindSparseInfo const & bindSparseInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bindSparseInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bindSparseInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bindSparseInfo.waitSemaphoreCount );
      VULKAN_HPP_HASH_COMBINE( seed, bindSparseInfo.pWaitSemaphores );
      VULKAN_HPP_HASH_COMBINE( seed, bindSparseInfo.bufferBindCount );
      VULKAN_HPP_HASH_COMBINE( seed, bindSparseInfo.pBufferBinds );
      VULKAN_HPP_HASH_COMBINE( seed, bindSparseInfo.imageOpaqueBindCount );
      VULKAN_HPP_HASH_COMBINE( seed, bindSparseInfo.pImageOpaqueBinds );
      VULKAN_HPP_HASH_COMBINE( seed, bindSparseInfo.imageBindCount );
      VULKAN_HPP_HASH_COMBINE( seed, bindSparseInfo.pImageBinds );
      VULKAN_HPP_HASH_COMBINE( seed, bindSparseInfo.signalSemaphoreCount );
      VULKAN_HPP_HASH_COMBINE( seed, bindSparseInfo.pSignalSemaphores );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindVertexBufferIndirectCommandNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BindVertexBufferIndirectCommandNV const & bindVertexBufferIndirectCommandNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bindVertexBufferIndirectCommandNV.bufferAddress );
      VULKAN_HPP_HASH_COMBINE( seed, bindVertexBufferIndirectCommandNV.size );
      VULKAN_HPP_HASH_COMBINE( seed, bindVertexBufferIndirectCommandNV.stride );
      return seed;
    }
  };

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BindVideoSessionMemoryInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BindVideoSessionMemoryInfoKHR const & bindVideoSessionMemoryInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bindVideoSessionMemoryInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bindVideoSessionMemoryInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bindVideoSessionMemoryInfoKHR.memoryBindIndex );
      VULKAN_HPP_HASH_COMBINE( seed, bindVideoSessionMemoryInfoKHR.memory );
      VULKAN_HPP_HASH_COMBINE( seed, bindVideoSessionMemoryInfoKHR.memoryOffset );
      VULKAN_HPP_HASH_COMBINE( seed, bindVideoSessionMemoryInfoKHR.memorySize );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageSubresourceLayers>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageSubresourceLayers const & imageSubresourceLayers ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageSubresourceLayers.aspectMask );
      VULKAN_HPP_HASH_COMBINE( seed, imageSubresourceLayers.mipLevel );
      VULKAN_HPP_HASH_COMBINE( seed, imageSubresourceLayers.baseArrayLayer );
      VULKAN_HPP_HASH_COMBINE( seed, imageSubresourceLayers.layerCount );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageBlit2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageBlit2 const & imageBlit2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageBlit2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageBlit2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageBlit2.srcSubresource );
      for ( size_t i = 0; i < 2; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, imageBlit2.srcOffsets[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, imageBlit2.dstSubresource );
      for ( size_t i = 0; i < 2; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, imageBlit2.dstOffsets[i] );
      }
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BlitImageInfo2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BlitImageInfo2 const & blitImageInfo2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, blitImageInfo2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, blitImageInfo2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, blitImageInfo2.srcImage );
      VULKAN_HPP_HASH_COMBINE( seed, blitImageInfo2.srcImageLayout );
      VULKAN_HPP_HASH_COMBINE( seed, blitImageInfo2.dstImage );
      VULKAN_HPP_HASH_COMBINE( seed, blitImageInfo2.dstImageLayout );
      VULKAN_HPP_HASH_COMBINE( seed, blitImageInfo2.regionCount );
      VULKAN_HPP_HASH_COMBINE( seed, blitImageInfo2.pRegions );
      VULKAN_HPP_HASH_COMBINE( seed, blitImageInfo2.filter );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_FUCHSIA )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferCollectionBufferCreateInfoFUCHSIA>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::BufferCollectionBufferCreateInfoFUCHSIA const & bufferCollectionBufferCreateInfoFUCHSIA ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionBufferCreateInfoFUCHSIA.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionBufferCreateInfoFUCHSIA.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionBufferCreateInfoFUCHSIA.collection );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionBufferCreateInfoFUCHSIA.index );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferCollectionConstraintsInfoFUCHSIA>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::BufferCollectionConstraintsInfoFUCHSIA const & bufferCollectionConstraintsInfoFUCHSIA ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionConstraintsInfoFUCHSIA.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionConstraintsInfoFUCHSIA.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionConstraintsInfoFUCHSIA.minBufferCount );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionConstraintsInfoFUCHSIA.maxBufferCount );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionConstraintsInfoFUCHSIA.minBufferCountForCamping );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionConstraintsInfoFUCHSIA.minBufferCountForDedicatedSlack );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionConstraintsInfoFUCHSIA.minBufferCountForSharedSlack );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferCollectionCreateInfoFUCHSIA>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BufferCollectionCreateInfoFUCHSIA const & bufferCollectionCreateInfoFUCHSIA ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionCreateInfoFUCHSIA.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionCreateInfoFUCHSIA.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionCreateInfoFUCHSIA.collectionToken );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferCollectionImageCreateInfoFUCHSIA>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::BufferCollectionImageCreateInfoFUCHSIA const & bufferCollectionImageCreateInfoFUCHSIA ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionImageCreateInfoFUCHSIA.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionImageCreateInfoFUCHSIA.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionImageCreateInfoFUCHSIA.collection );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionImageCreateInfoFUCHSIA.index );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SysmemColorSpaceFUCHSIA>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SysmemColorSpaceFUCHSIA const & sysmemColorSpaceFUCHSIA ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, sysmemColorSpaceFUCHSIA.sType );
      VULKAN_HPP_HASH_COMBINE( seed, sysmemColorSpaceFUCHSIA.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, sysmemColorSpaceFUCHSIA.colorSpace );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferCollectionPropertiesFUCHSIA>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BufferCollectionPropertiesFUCHSIA const & bufferCollectionPropertiesFUCHSIA ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionPropertiesFUCHSIA.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionPropertiesFUCHSIA.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionPropertiesFUCHSIA.memoryTypeBits );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionPropertiesFUCHSIA.bufferCount );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionPropertiesFUCHSIA.createInfoIndex );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionPropertiesFUCHSIA.sysmemPixelFormat );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionPropertiesFUCHSIA.formatFeatures );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionPropertiesFUCHSIA.sysmemColorSpaceIndex );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionPropertiesFUCHSIA.samplerYcbcrConversionComponents );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionPropertiesFUCHSIA.suggestedYcbcrModel );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionPropertiesFUCHSIA.suggestedYcbcrRange );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionPropertiesFUCHSIA.suggestedXChromaOffset );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCollectionPropertiesFUCHSIA.suggestedYChromaOffset );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BufferCreateInfo const & bufferCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bufferCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCreateInfo.size );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCreateInfo.usage );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCreateInfo.sharingMode );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCreateInfo.queueFamilyIndexCount );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCreateInfo.pQueueFamilyIndices );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_FUCHSIA )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferConstraintsInfoFUCHSIA>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BufferConstraintsInfoFUCHSIA const & bufferConstraintsInfoFUCHSIA ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bufferConstraintsInfoFUCHSIA.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bufferConstraintsInfoFUCHSIA.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bufferConstraintsInfoFUCHSIA.createInfo );
      VULKAN_HPP_HASH_COMBINE( seed, bufferConstraintsInfoFUCHSIA.requiredFormatFeatures );
      VULKAN_HPP_HASH_COMBINE( seed, bufferConstraintsInfoFUCHSIA.bufferCollectionConstraints );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferCopy>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BufferCopy const & bufferCopy ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bufferCopy.srcOffset );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCopy.dstOffset );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCopy.size );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferCopy2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BufferCopy2 const & bufferCopy2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bufferCopy2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCopy2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCopy2.srcOffset );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCopy2.dstOffset );
      VULKAN_HPP_HASH_COMBINE( seed, bufferCopy2.size );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferDeviceAddressCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BufferDeviceAddressCreateInfoEXT const & bufferDeviceAddressCreateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bufferDeviceAddressCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bufferDeviceAddressCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bufferDeviceAddressCreateInfoEXT.deviceAddress );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferDeviceAddressInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BufferDeviceAddressInfo const & bufferDeviceAddressInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bufferDeviceAddressInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bufferDeviceAddressInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bufferDeviceAddressInfo.buffer );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferImageCopy>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BufferImageCopy const & bufferImageCopy ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bufferImageCopy.bufferOffset );
      VULKAN_HPP_HASH_COMBINE( seed, bufferImageCopy.bufferRowLength );
      VULKAN_HPP_HASH_COMBINE( seed, bufferImageCopy.bufferImageHeight );
      VULKAN_HPP_HASH_COMBINE( seed, bufferImageCopy.imageSubresource );
      VULKAN_HPP_HASH_COMBINE( seed, bufferImageCopy.imageOffset );
      VULKAN_HPP_HASH_COMBINE( seed, bufferImageCopy.imageExtent );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferImageCopy2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BufferImageCopy2 const & bufferImageCopy2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bufferImageCopy2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bufferImageCopy2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bufferImageCopy2.bufferOffset );
      VULKAN_HPP_HASH_COMBINE( seed, bufferImageCopy2.bufferRowLength );
      VULKAN_HPP_HASH_COMBINE( seed, bufferImageCopy2.bufferImageHeight );
      VULKAN_HPP_HASH_COMBINE( seed, bufferImageCopy2.imageSubresource );
      VULKAN_HPP_HASH_COMBINE( seed, bufferImageCopy2.imageOffset );
      VULKAN_HPP_HASH_COMBINE( seed, bufferImageCopy2.imageExtent );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferMemoryBarrier>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BufferMemoryBarrier const & bufferMemoryBarrier ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bufferMemoryBarrier.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bufferMemoryBarrier.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bufferMemoryBarrier.srcAccessMask );
      VULKAN_HPP_HASH_COMBINE( seed, bufferMemoryBarrier.dstAccessMask );
      VULKAN_HPP_HASH_COMBINE( seed, bufferMemoryBarrier.srcQueueFamilyIndex );
      VULKAN_HPP_HASH_COMBINE( seed, bufferMemoryBarrier.dstQueueFamilyIndex );
      VULKAN_HPP_HASH_COMBINE( seed, bufferMemoryBarrier.buffer );
      VULKAN_HPP_HASH_COMBINE( seed, bufferMemoryBarrier.offset );
      VULKAN_HPP_HASH_COMBINE( seed, bufferMemoryBarrier.size );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferMemoryBarrier2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BufferMemoryBarrier2 const & bufferMemoryBarrier2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bufferMemoryBarrier2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bufferMemoryBarrier2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bufferMemoryBarrier2.srcStageMask );
      VULKAN_HPP_HASH_COMBINE( seed, bufferMemoryBarrier2.srcAccessMask );
      VULKAN_HPP_HASH_COMBINE( seed, bufferMemoryBarrier2.dstStageMask );
      VULKAN_HPP_HASH_COMBINE( seed, bufferMemoryBarrier2.dstAccessMask );
      VULKAN_HPP_HASH_COMBINE( seed, bufferMemoryBarrier2.srcQueueFamilyIndex );
      VULKAN_HPP_HASH_COMBINE( seed, bufferMemoryBarrier2.dstQueueFamilyIndex );
      VULKAN_HPP_HASH_COMBINE( seed, bufferMemoryBarrier2.buffer );
      VULKAN_HPP_HASH_COMBINE( seed, bufferMemoryBarrier2.offset );
      VULKAN_HPP_HASH_COMBINE( seed, bufferMemoryBarrier2.size );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferMemoryRequirementsInfo2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BufferMemoryRequirementsInfo2 const & bufferMemoryRequirementsInfo2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bufferMemoryRequirementsInfo2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bufferMemoryRequirementsInfo2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bufferMemoryRequirementsInfo2.buffer );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferOpaqueCaptureAddressCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BufferOpaqueCaptureAddressCreateInfo const & bufferOpaqueCaptureAddressCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bufferOpaqueCaptureAddressCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bufferOpaqueCaptureAddressCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bufferOpaqueCaptureAddressCreateInfo.opaqueCaptureAddress );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::BufferViewCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::BufferViewCreateInfo const & bufferViewCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, bufferViewCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, bufferViewCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, bufferViewCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, bufferViewCreateInfo.buffer );
      VULKAN_HPP_HASH_COMBINE( seed, bufferViewCreateInfo.format );
      VULKAN_HPP_HASH_COMBINE( seed, bufferViewCreateInfo.offset );
      VULKAN_HPP_HASH_COMBINE( seed, bufferViewCreateInfo.range );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CalibratedTimestampInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CalibratedTimestampInfoEXT const & calibratedTimestampInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, calibratedTimestampInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, calibratedTimestampInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, calibratedTimestampInfoEXT.timeDomain );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CheckpointData2NV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CheckpointData2NV const & checkpointData2NV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, checkpointData2NV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, checkpointData2NV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, checkpointData2NV.stage );
      VULKAN_HPP_HASH_COMBINE( seed, checkpointData2NV.pCheckpointMarker );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CheckpointDataNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CheckpointDataNV const & checkpointDataNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, checkpointDataNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, checkpointDataNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, checkpointDataNV.stage );
      VULKAN_HPP_HASH_COMBINE( seed, checkpointDataNV.pCheckpointMarker );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ClearDepthStencilValue>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ClearDepthStencilValue const & clearDepthStencilValue ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, clearDepthStencilValue.depth );
      VULKAN_HPP_HASH_COMBINE( seed, clearDepthStencilValue.stencil );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ClearRect>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ClearRect const & clearRect ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, clearRect.rect );
      VULKAN_HPP_HASH_COMBINE( seed, clearRect.baseArrayLayer );
      VULKAN_HPP_HASH_COMBINE( seed, clearRect.layerCount );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CoarseSampleLocationNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CoarseSampleLocationNV const & coarseSampleLocationNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, coarseSampleLocationNV.pixelX );
      VULKAN_HPP_HASH_COMBINE( seed, coarseSampleLocationNV.pixelY );
      VULKAN_HPP_HASH_COMBINE( seed, coarseSampleLocationNV.sample );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CoarseSampleOrderCustomNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CoarseSampleOrderCustomNV const & coarseSampleOrderCustomNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, coarseSampleOrderCustomNV.shadingRate );
      VULKAN_HPP_HASH_COMBINE( seed, coarseSampleOrderCustomNV.sampleCount );
      VULKAN_HPP_HASH_COMBINE( seed, coarseSampleOrderCustomNV.sampleLocationCount );
      VULKAN_HPP_HASH_COMBINE( seed, coarseSampleOrderCustomNV.pSampleLocations );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ColorBlendAdvancedEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ColorBlendAdvancedEXT const & colorBlendAdvancedEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, colorBlendAdvancedEXT.advancedBlendOp );
      VULKAN_HPP_HASH_COMBINE( seed, colorBlendAdvancedEXT.srcPremultiplied );
      VULKAN_HPP_HASH_COMBINE( seed, colorBlendAdvancedEXT.dstPremultiplied );
      VULKAN_HPP_HASH_COMBINE( seed, colorBlendAdvancedEXT.blendOverlap );
      VULKAN_HPP_HASH_COMBINE( seed, colorBlendAdvancedEXT.clampResults );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ColorBlendEquationEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ColorBlendEquationEXT const & colorBlendEquationEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, colorBlendEquationEXT.srcColorBlendFactor );
      VULKAN_HPP_HASH_COMBINE( seed, colorBlendEquationEXT.dstColorBlendFactor );
      VULKAN_HPP_HASH_COMBINE( seed, colorBlendEquationEXT.colorBlendOp );
      VULKAN_HPP_HASH_COMBINE( seed, colorBlendEquationEXT.srcAlphaBlendFactor );
      VULKAN_HPP_HASH_COMBINE( seed, colorBlendEquationEXT.dstAlphaBlendFactor );
      VULKAN_HPP_HASH_COMBINE( seed, colorBlendEquationEXT.alphaBlendOp );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CommandBufferAllocateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CommandBufferAllocateInfo const & commandBufferAllocateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferAllocateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferAllocateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferAllocateInfo.commandPool );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferAllocateInfo.level );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferAllocateInfo.commandBufferCount );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CommandBufferInheritanceInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CommandBufferInheritanceInfo const & commandBufferInheritanceInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceInfo.renderPass );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceInfo.subpass );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceInfo.framebuffer );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceInfo.occlusionQueryEnable );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceInfo.queryFlags );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceInfo.pipelineStatistics );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CommandBufferBeginInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CommandBufferBeginInfo const & commandBufferBeginInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferBeginInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferBeginInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferBeginInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferBeginInfo.pInheritanceInfo );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CommandBufferInheritanceConditionalRenderingInfoEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::CommandBufferInheritanceConditionalRenderingInfoEXT const & commandBufferInheritanceConditionalRenderingInfoEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceConditionalRenderingInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceConditionalRenderingInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceConditionalRenderingInfoEXT.conditionalRenderingEnable );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CommandBufferInheritanceRenderPassTransformInfoQCOM>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::CommandBufferInheritanceRenderPassTransformInfoQCOM const & commandBufferInheritanceRenderPassTransformInfoQCOM ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceRenderPassTransformInfoQCOM.sType );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceRenderPassTransformInfoQCOM.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceRenderPassTransformInfoQCOM.transform );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceRenderPassTransformInfoQCOM.renderArea );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CommandBufferInheritanceRenderingInfo>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::CommandBufferInheritanceRenderingInfo const & commandBufferInheritanceRenderingInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceRenderingInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceRenderingInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceRenderingInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceRenderingInfo.viewMask );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceRenderingInfo.colorAttachmentCount );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceRenderingInfo.pColorAttachmentFormats );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceRenderingInfo.depthAttachmentFormat );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceRenderingInfo.stencilAttachmentFormat );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceRenderingInfo.rasterizationSamples );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Viewport>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Viewport const & viewport ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, viewport.x );
      VULKAN_HPP_HASH_COMBINE( seed, viewport.y );
      VULKAN_HPP_HASH_COMBINE( seed, viewport.width );
      VULKAN_HPP_HASH_COMBINE( seed, viewport.height );
      VULKAN_HPP_HASH_COMBINE( seed, viewport.minDepth );
      VULKAN_HPP_HASH_COMBINE( seed, viewport.maxDepth );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CommandBufferInheritanceViewportScissorInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CommandBufferInheritanceViewportScissorInfoNV const & commandBufferInheritanceViewportScissorInfoNV ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceViewportScissorInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceViewportScissorInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceViewportScissorInfoNV.viewportScissor2D );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceViewportScissorInfoNV.viewportDepthCount );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferInheritanceViewportScissorInfoNV.pViewportDepths );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CommandBufferSubmitInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CommandBufferSubmitInfo const & commandBufferSubmitInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferSubmitInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferSubmitInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferSubmitInfo.commandBuffer );
      VULKAN_HPP_HASH_COMBINE( seed, commandBufferSubmitInfo.deviceMask );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CommandPoolCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CommandPoolCreateInfo const & commandPoolCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, commandPoolCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, commandPoolCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, commandPoolCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, commandPoolCreateInfo.queueFamilyIndex );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SpecializationMapEntry>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SpecializationMapEntry const & specializationMapEntry ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, specializationMapEntry.constantID );
      VULKAN_HPP_HASH_COMBINE( seed, specializationMapEntry.offset );
      VULKAN_HPP_HASH_COMBINE( seed, specializationMapEntry.size );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SpecializationInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SpecializationInfo const & specializationInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, specializationInfo.mapEntryCount );
      VULKAN_HPP_HASH_COMBINE( seed, specializationInfo.pMapEntries );
      VULKAN_HPP_HASH_COMBINE( seed, specializationInfo.dataSize );
      VULKAN_HPP_HASH_COMBINE( seed, specializationInfo.pData );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineShaderStageCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineShaderStageCreateInfo const & pipelineShaderStageCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineShaderStageCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineShaderStageCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineShaderStageCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineShaderStageCreateInfo.stage );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineShaderStageCreateInfo.module );
      for ( const char * p = pipelineShaderStageCreateInfo.pName; *p != '\0'; ++p )
      {
        VULKAN_HPP_HASH_COMBINE( seed, *p );
      }
      VULKAN_HPP_HASH_COMBINE( seed, pipelineShaderStageCreateInfo.pSpecializationInfo );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ComputePipelineCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ComputePipelineCreateInfo const & computePipelineCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, computePipelineCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, computePipelineCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, computePipelineCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, computePipelineCreateInfo.stage );
      VULKAN_HPP_HASH_COMBINE( seed, computePipelineCreateInfo.layout );
      VULKAN_HPP_HASH_COMBINE( seed, computePipelineCreateInfo.basePipelineHandle );
      VULKAN_HPP_HASH_COMBINE( seed, computePipelineCreateInfo.basePipelineIndex );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ConditionalRenderingBeginInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ConditionalRenderingBeginInfoEXT const & conditionalRenderingBeginInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, conditionalRenderingBeginInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, conditionalRenderingBeginInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, conditionalRenderingBeginInfoEXT.buffer );
      VULKAN_HPP_HASH_COMBINE( seed, conditionalRenderingBeginInfoEXT.offset );
      VULKAN_HPP_HASH_COMBINE( seed, conditionalRenderingBeginInfoEXT.flags );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ConformanceVersion>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ConformanceVersion const & conformanceVersion ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, conformanceVersion.major );
      VULKAN_HPP_HASH_COMBINE( seed, conformanceVersion.minor );
      VULKAN_HPP_HASH_COMBINE( seed, conformanceVersion.subminor );
      VULKAN_HPP_HASH_COMBINE( seed, conformanceVersion.patch );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CooperativeMatrixPropertiesNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CooperativeMatrixPropertiesNV const & cooperativeMatrixPropertiesNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, cooperativeMatrixPropertiesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, cooperativeMatrixPropertiesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, cooperativeMatrixPropertiesNV.MSize );
      VULKAN_HPP_HASH_COMBINE( seed, cooperativeMatrixPropertiesNV.NSize );
      VULKAN_HPP_HASH_COMBINE( seed, cooperativeMatrixPropertiesNV.KSize );
      VULKAN_HPP_HASH_COMBINE( seed, cooperativeMatrixPropertiesNV.AType );
      VULKAN_HPP_HASH_COMBINE( seed, cooperativeMatrixPropertiesNV.BType );
      VULKAN_HPP_HASH_COMBINE( seed, cooperativeMatrixPropertiesNV.CType );
      VULKAN_HPP_HASH_COMBINE( seed, cooperativeMatrixPropertiesNV.DType );
      VULKAN_HPP_HASH_COMBINE( seed, cooperativeMatrixPropertiesNV.scope );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyAccelerationStructureInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CopyAccelerationStructureInfoKHR const & copyAccelerationStructureInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, copyAccelerationStructureInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, copyAccelerationStructureInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, copyAccelerationStructureInfoKHR.src );
      VULKAN_HPP_HASH_COMBINE( seed, copyAccelerationStructureInfoKHR.dst );
      VULKAN_HPP_HASH_COMBINE( seed, copyAccelerationStructureInfoKHR.mode );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyBufferInfo2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CopyBufferInfo2 const & copyBufferInfo2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, copyBufferInfo2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, copyBufferInfo2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, copyBufferInfo2.srcBuffer );
      VULKAN_HPP_HASH_COMBINE( seed, copyBufferInfo2.dstBuffer );
      VULKAN_HPP_HASH_COMBINE( seed, copyBufferInfo2.regionCount );
      VULKAN_HPP_HASH_COMBINE( seed, copyBufferInfo2.pRegions );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyBufferToImageInfo2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CopyBufferToImageInfo2 const & copyBufferToImageInfo2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, copyBufferToImageInfo2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, copyBufferToImageInfo2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, copyBufferToImageInfo2.srcBuffer );
      VULKAN_HPP_HASH_COMBINE( seed, copyBufferToImageInfo2.dstImage );
      VULKAN_HPP_HASH_COMBINE( seed, copyBufferToImageInfo2.dstImageLayout );
      VULKAN_HPP_HASH_COMBINE( seed, copyBufferToImageInfo2.regionCount );
      VULKAN_HPP_HASH_COMBINE( seed, copyBufferToImageInfo2.pRegions );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyCommandTransformInfoQCOM>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CopyCommandTransformInfoQCOM const & copyCommandTransformInfoQCOM ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, copyCommandTransformInfoQCOM.sType );
      VULKAN_HPP_HASH_COMBINE( seed, copyCommandTransformInfoQCOM.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, copyCommandTransformInfoQCOM.transform );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyDescriptorSet>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CopyDescriptorSet const & copyDescriptorSet ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, copyDescriptorSet.sType );
      VULKAN_HPP_HASH_COMBINE( seed, copyDescriptorSet.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, copyDescriptorSet.srcSet );
      VULKAN_HPP_HASH_COMBINE( seed, copyDescriptorSet.srcBinding );
      VULKAN_HPP_HASH_COMBINE( seed, copyDescriptorSet.srcArrayElement );
      VULKAN_HPP_HASH_COMBINE( seed, copyDescriptorSet.dstSet );
      VULKAN_HPP_HASH_COMBINE( seed, copyDescriptorSet.dstBinding );
      VULKAN_HPP_HASH_COMBINE( seed, copyDescriptorSet.dstArrayElement );
      VULKAN_HPP_HASH_COMBINE( seed, copyDescriptorSet.descriptorCount );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageCopy2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageCopy2 const & imageCopy2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageCopy2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageCopy2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageCopy2.srcSubresource );
      VULKAN_HPP_HASH_COMBINE( seed, imageCopy2.srcOffset );
      VULKAN_HPP_HASH_COMBINE( seed, imageCopy2.dstSubresource );
      VULKAN_HPP_HASH_COMBINE( seed, imageCopy2.dstOffset );
      VULKAN_HPP_HASH_COMBINE( seed, imageCopy2.extent );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyImageInfo2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CopyImageInfo2 const & copyImageInfo2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, copyImageInfo2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, copyImageInfo2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, copyImageInfo2.srcImage );
      VULKAN_HPP_HASH_COMBINE( seed, copyImageInfo2.srcImageLayout );
      VULKAN_HPP_HASH_COMBINE( seed, copyImageInfo2.dstImage );
      VULKAN_HPP_HASH_COMBINE( seed, copyImageInfo2.dstImageLayout );
      VULKAN_HPP_HASH_COMBINE( seed, copyImageInfo2.regionCount );
      VULKAN_HPP_HASH_COMBINE( seed, copyImageInfo2.pRegions );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyImageToBufferInfo2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CopyImageToBufferInfo2 const & copyImageToBufferInfo2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, copyImageToBufferInfo2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, copyImageToBufferInfo2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, copyImageToBufferInfo2.srcImage );
      VULKAN_HPP_HASH_COMBINE( seed, copyImageToBufferInfo2.srcImageLayout );
      VULKAN_HPP_HASH_COMBINE( seed, copyImageToBufferInfo2.dstBuffer );
      VULKAN_HPP_HASH_COMBINE( seed, copyImageToBufferInfo2.regionCount );
      VULKAN_HPP_HASH_COMBINE( seed, copyImageToBufferInfo2.pRegions );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CopyMicromapInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CopyMicromapInfoEXT const & copyMicromapInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, copyMicromapInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, copyMicromapInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, copyMicromapInfoEXT.src );
      VULKAN_HPP_HASH_COMBINE( seed, copyMicromapInfoEXT.dst );
      VULKAN_HPP_HASH_COMBINE( seed, copyMicromapInfoEXT.mode );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CuFunctionCreateInfoNVX>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CuFunctionCreateInfoNVX const & cuFunctionCreateInfoNVX ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, cuFunctionCreateInfoNVX.sType );
      VULKAN_HPP_HASH_COMBINE( seed, cuFunctionCreateInfoNVX.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, cuFunctionCreateInfoNVX.module );
      for ( const char * p = cuFunctionCreateInfoNVX.pName; *p != '\0'; ++p )
      {
        VULKAN_HPP_HASH_COMBINE( seed, *p );
      }
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CuLaunchInfoNVX>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CuLaunchInfoNVX const & cuLaunchInfoNVX ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, cuLaunchInfoNVX.sType );
      VULKAN_HPP_HASH_COMBINE( seed, cuLaunchInfoNVX.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, cuLaunchInfoNVX.function );
      VULKAN_HPP_HASH_COMBINE( seed, cuLaunchInfoNVX.gridDimX );
      VULKAN_HPP_HASH_COMBINE( seed, cuLaunchInfoNVX.gridDimY );
      VULKAN_HPP_HASH_COMBINE( seed, cuLaunchInfoNVX.gridDimZ );
      VULKAN_HPP_HASH_COMBINE( seed, cuLaunchInfoNVX.blockDimX );
      VULKAN_HPP_HASH_COMBINE( seed, cuLaunchInfoNVX.blockDimY );
      VULKAN_HPP_HASH_COMBINE( seed, cuLaunchInfoNVX.blockDimZ );
      VULKAN_HPP_HASH_COMBINE( seed, cuLaunchInfoNVX.sharedMemBytes );
      VULKAN_HPP_HASH_COMBINE( seed, cuLaunchInfoNVX.paramCount );
      VULKAN_HPP_HASH_COMBINE( seed, cuLaunchInfoNVX.pParams );
      VULKAN_HPP_HASH_COMBINE( seed, cuLaunchInfoNVX.extraCount );
      VULKAN_HPP_HASH_COMBINE( seed, cuLaunchInfoNVX.pExtras );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::CuModuleCreateInfoNVX>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::CuModuleCreateInfoNVX const & cuModuleCreateInfoNVX ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, cuModuleCreateInfoNVX.sType );
      VULKAN_HPP_HASH_COMBINE( seed, cuModuleCreateInfoNVX.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, cuModuleCreateInfoNVX.dataSize );
      VULKAN_HPP_HASH_COMBINE( seed, cuModuleCreateInfoNVX.pData );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::D3D12FenceSubmitInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::D3D12FenceSubmitInfoKHR const & d3D12FenceSubmitInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, d3D12FenceSubmitInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, d3D12FenceSubmitInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, d3D12FenceSubmitInfoKHR.waitSemaphoreValuesCount );
      VULKAN_HPP_HASH_COMBINE( seed, d3D12FenceSubmitInfoKHR.pWaitSemaphoreValues );
      VULKAN_HPP_HASH_COMBINE( seed, d3D12FenceSubmitInfoKHR.signalSemaphoreValuesCount );
      VULKAN_HPP_HASH_COMBINE( seed, d3D12FenceSubmitInfoKHR.pSignalSemaphoreValues );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DebugMarkerMarkerInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DebugMarkerMarkerInfoEXT const & debugMarkerMarkerInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, debugMarkerMarkerInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, debugMarkerMarkerInfoEXT.pNext );
      for ( const char * p = debugMarkerMarkerInfoEXT.pMarkerName; *p != '\0'; ++p )
      {
        VULKAN_HPP_HASH_COMBINE( seed, *p );
      }
      for ( size_t i = 0; i < 4; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, debugMarkerMarkerInfoEXT.color[i] );
      }
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DebugMarkerObjectNameInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DebugMarkerObjectNameInfoEXT const & debugMarkerObjectNameInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, debugMarkerObjectNameInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, debugMarkerObjectNameInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, debugMarkerObjectNameInfoEXT.objectType );
      VULKAN_HPP_HASH_COMBINE( seed, debugMarkerObjectNameInfoEXT.object );
      for ( const char * p = debugMarkerObjectNameInfoEXT.pObjectName; *p != '\0'; ++p )
      {
        VULKAN_HPP_HASH_COMBINE( seed, *p );
      }
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DebugMarkerObjectTagInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DebugMarkerObjectTagInfoEXT const & debugMarkerObjectTagInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, debugMarkerObjectTagInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, debugMarkerObjectTagInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, debugMarkerObjectTagInfoEXT.objectType );
      VULKAN_HPP_HASH_COMBINE( seed, debugMarkerObjectTagInfoEXT.object );
      VULKAN_HPP_HASH_COMBINE( seed, debugMarkerObjectTagInfoEXT.tagName );
      VULKAN_HPP_HASH_COMBINE( seed, debugMarkerObjectTagInfoEXT.tagSize );
      VULKAN_HPP_HASH_COMBINE( seed, debugMarkerObjectTagInfoEXT.pTag );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DebugReportCallbackCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DebugReportCallbackCreateInfoEXT const & debugReportCallbackCreateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, debugReportCallbackCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, debugReportCallbackCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, debugReportCallbackCreateInfoEXT.flags );
      VULKAN_HPP_HASH_COMBINE( seed, debugReportCallbackCreateInfoEXT.pfnCallback );
      VULKAN_HPP_HASH_COMBINE( seed, debugReportCallbackCreateInfoEXT.pUserData );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DebugUtilsLabelEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DebugUtilsLabelEXT const & debugUtilsLabelEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsLabelEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsLabelEXT.pNext );
      for ( const char * p = debugUtilsLabelEXT.pLabelName; *p != '\0'; ++p )
      {
        VULKAN_HPP_HASH_COMBINE( seed, *p );
      }
      for ( size_t i = 0; i < 4; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, debugUtilsLabelEXT.color[i] );
      }
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DebugUtilsObjectNameInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DebugUtilsObjectNameInfoEXT const & debugUtilsObjectNameInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsObjectNameInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsObjectNameInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsObjectNameInfoEXT.objectType );
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsObjectNameInfoEXT.objectHandle );
      for ( const char * p = debugUtilsObjectNameInfoEXT.pObjectName; *p != '\0'; ++p )
      {
        VULKAN_HPP_HASH_COMBINE( seed, *p );
      }
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DebugUtilsMessengerCallbackDataEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DebugUtilsMessengerCallbackDataEXT const & debugUtilsMessengerCallbackDataEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsMessengerCallbackDataEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsMessengerCallbackDataEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsMessengerCallbackDataEXT.flags );
      for ( const char * p = debugUtilsMessengerCallbackDataEXT.pMessageIdName; *p != '\0'; ++p )
      {
        VULKAN_HPP_HASH_COMBINE( seed, *p );
      }
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsMessengerCallbackDataEXT.messageIdNumber );
      for ( const char * p = debugUtilsMessengerCallbackDataEXT.pMessage; *p != '\0'; ++p )
      {
        VULKAN_HPP_HASH_COMBINE( seed, *p );
      }
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsMessengerCallbackDataEXT.queueLabelCount );
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsMessengerCallbackDataEXT.pQueueLabels );
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsMessengerCallbackDataEXT.cmdBufLabelCount );
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsMessengerCallbackDataEXT.pCmdBufLabels );
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsMessengerCallbackDataEXT.objectCount );
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsMessengerCallbackDataEXT.pObjects );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DebugUtilsMessengerCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DebugUtilsMessengerCreateInfoEXT const & debugUtilsMessengerCreateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsMessengerCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsMessengerCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsMessengerCreateInfoEXT.flags );
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsMessengerCreateInfoEXT.messageSeverity );
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsMessengerCreateInfoEXT.messageType );
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsMessengerCreateInfoEXT.pfnUserCallback );
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsMessengerCreateInfoEXT.pUserData );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DebugUtilsObjectTagInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DebugUtilsObjectTagInfoEXT const & debugUtilsObjectTagInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsObjectTagInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsObjectTagInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsObjectTagInfoEXT.objectType );
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsObjectTagInfoEXT.objectHandle );
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsObjectTagInfoEXT.tagName );
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsObjectTagInfoEXT.tagSize );
      VULKAN_HPP_HASH_COMBINE( seed, debugUtilsObjectTagInfoEXT.pTag );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DedicatedAllocationBufferCreateInfoNV>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::DedicatedAllocationBufferCreateInfoNV const & dedicatedAllocationBufferCreateInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, dedicatedAllocationBufferCreateInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, dedicatedAllocationBufferCreateInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, dedicatedAllocationBufferCreateInfoNV.dedicatedAllocation );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DedicatedAllocationImageCreateInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DedicatedAllocationImageCreateInfoNV const & dedicatedAllocationImageCreateInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, dedicatedAllocationImageCreateInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, dedicatedAllocationImageCreateInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, dedicatedAllocationImageCreateInfoNV.dedicatedAllocation );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DedicatedAllocationMemoryAllocateInfoNV>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::DedicatedAllocationMemoryAllocateInfoNV const & dedicatedAllocationMemoryAllocateInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, dedicatedAllocationMemoryAllocateInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, dedicatedAllocationMemoryAllocateInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, dedicatedAllocationMemoryAllocateInfoNV.image );
      VULKAN_HPP_HASH_COMBINE( seed, dedicatedAllocationMemoryAllocateInfoNV.buffer );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryBarrier2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MemoryBarrier2 const & memoryBarrier2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, memoryBarrier2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, memoryBarrier2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, memoryBarrier2.srcStageMask );
      VULKAN_HPP_HASH_COMBINE( seed, memoryBarrier2.srcAccessMask );
      VULKAN_HPP_HASH_COMBINE( seed, memoryBarrier2.dstStageMask );
      VULKAN_HPP_HASH_COMBINE( seed, memoryBarrier2.dstAccessMask );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageSubresourceRange>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageSubresourceRange const & imageSubresourceRange ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageSubresourceRange.aspectMask );
      VULKAN_HPP_HASH_COMBINE( seed, imageSubresourceRange.baseMipLevel );
      VULKAN_HPP_HASH_COMBINE( seed, imageSubresourceRange.levelCount );
      VULKAN_HPP_HASH_COMBINE( seed, imageSubresourceRange.baseArrayLayer );
      VULKAN_HPP_HASH_COMBINE( seed, imageSubresourceRange.layerCount );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageMemoryBarrier2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageMemoryBarrier2 const & imageMemoryBarrier2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryBarrier2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryBarrier2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryBarrier2.srcStageMask );
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryBarrier2.srcAccessMask );
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryBarrier2.dstStageMask );
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryBarrier2.dstAccessMask );
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryBarrier2.oldLayout );
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryBarrier2.newLayout );
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryBarrier2.srcQueueFamilyIndex );
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryBarrier2.dstQueueFamilyIndex );
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryBarrier2.image );
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryBarrier2.subresourceRange );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DependencyInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DependencyInfo const & dependencyInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, dependencyInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, dependencyInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, dependencyInfo.dependencyFlags );
      VULKAN_HPP_HASH_COMBINE( seed, dependencyInfo.memoryBarrierCount );
      VULKAN_HPP_HASH_COMBINE( seed, dependencyInfo.pMemoryBarriers );
      VULKAN_HPP_HASH_COMBINE( seed, dependencyInfo.bufferMemoryBarrierCount );
      VULKAN_HPP_HASH_COMBINE( seed, dependencyInfo.pBufferMemoryBarriers );
      VULKAN_HPP_HASH_COMBINE( seed, dependencyInfo.imageMemoryBarrierCount );
      VULKAN_HPP_HASH_COMBINE( seed, dependencyInfo.pImageMemoryBarriers );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorBufferInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DescriptorBufferInfo const & descriptorBufferInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, descriptorBufferInfo.buffer );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorBufferInfo.offset );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorBufferInfo.range );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorImageInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DescriptorImageInfo const & descriptorImageInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, descriptorImageInfo.sampler );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorImageInfo.imageView );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorImageInfo.imageLayout );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorPoolSize>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DescriptorPoolSize const & descriptorPoolSize ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, descriptorPoolSize.type );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorPoolSize.descriptorCount );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorPoolCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DescriptorPoolCreateInfo const & descriptorPoolCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, descriptorPoolCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorPoolCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorPoolCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorPoolCreateInfo.maxSets );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorPoolCreateInfo.poolSizeCount );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorPoolCreateInfo.pPoolSizes );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorPoolInlineUniformBlockCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DescriptorPoolInlineUniformBlockCreateInfo const & descriptorPoolInlineUniformBlockCreateInfo ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, descriptorPoolInlineUniformBlockCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorPoolInlineUniformBlockCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorPoolInlineUniformBlockCreateInfo.maxInlineUniformBlockBindings );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorSetAllocateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DescriptorSetAllocateInfo const & descriptorSetAllocateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetAllocateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetAllocateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetAllocateInfo.descriptorPool );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetAllocateInfo.descriptorSetCount );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetAllocateInfo.pSetLayouts );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorSetBindingReferenceVALVE>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DescriptorSetBindingReferenceVALVE const & descriptorSetBindingReferenceVALVE ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetBindingReferenceVALVE.sType );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetBindingReferenceVALVE.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetBindingReferenceVALVE.descriptorSetLayout );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetBindingReferenceVALVE.binding );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorSetLayoutBinding>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DescriptorSetLayoutBinding const & descriptorSetLayoutBinding ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetLayoutBinding.binding );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetLayoutBinding.descriptorType );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetLayoutBinding.descriptorCount );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetLayoutBinding.stageFlags );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetLayoutBinding.pImmutableSamplers );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorSetLayoutBindingFlagsCreateInfo>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::DescriptorSetLayoutBindingFlagsCreateInfo const & descriptorSetLayoutBindingFlagsCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetLayoutBindingFlagsCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetLayoutBindingFlagsCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetLayoutBindingFlagsCreateInfo.bindingCount );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetLayoutBindingFlagsCreateInfo.pBindingFlags );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorSetLayoutCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DescriptorSetLayoutCreateInfo const & descriptorSetLayoutCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetLayoutCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetLayoutCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetLayoutCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetLayoutCreateInfo.bindingCount );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetLayoutCreateInfo.pBindings );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorSetLayoutHostMappingInfoVALVE>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::DescriptorSetLayoutHostMappingInfoVALVE const & descriptorSetLayoutHostMappingInfoVALVE ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetLayoutHostMappingInfoVALVE.sType );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetLayoutHostMappingInfoVALVE.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetLayoutHostMappingInfoVALVE.descriptorOffset );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetLayoutHostMappingInfoVALVE.descriptorSize );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorSetLayoutSupport>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DescriptorSetLayoutSupport const & descriptorSetLayoutSupport ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetLayoutSupport.sType );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetLayoutSupport.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetLayoutSupport.supported );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorSetVariableDescriptorCountAllocateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DescriptorSetVariableDescriptorCountAllocateInfo const & descriptorSetVariableDescriptorCountAllocateInfo )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetVariableDescriptorCountAllocateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetVariableDescriptorCountAllocateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetVariableDescriptorCountAllocateInfo.descriptorSetCount );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetVariableDescriptorCountAllocateInfo.pDescriptorCounts );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorSetVariableDescriptorCountLayoutSupport>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DescriptorSetVariableDescriptorCountLayoutSupport const & descriptorSetVariableDescriptorCountLayoutSupport )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetVariableDescriptorCountLayoutSupport.sType );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetVariableDescriptorCountLayoutSupport.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorSetVariableDescriptorCountLayoutSupport.maxVariableDescriptorCount );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateEntry>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateEntry const & descriptorUpdateTemplateEntry ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, descriptorUpdateTemplateEntry.dstBinding );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorUpdateTemplateEntry.dstArrayElement );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorUpdateTemplateEntry.descriptorCount );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorUpdateTemplateEntry.descriptorType );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorUpdateTemplateEntry.offset );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorUpdateTemplateEntry.stride );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateCreateInfo const & descriptorUpdateTemplateCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, descriptorUpdateTemplateCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorUpdateTemplateCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorUpdateTemplateCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorUpdateTemplateCreateInfo.descriptorUpdateEntryCount );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorUpdateTemplateCreateInfo.pDescriptorUpdateEntries );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorUpdateTemplateCreateInfo.templateType );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorUpdateTemplateCreateInfo.descriptorSetLayout );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorUpdateTemplateCreateInfo.pipelineBindPoint );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorUpdateTemplateCreateInfo.pipelineLayout );
      VULKAN_HPP_HASH_COMBINE( seed, descriptorUpdateTemplateCreateInfo.set );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceAddressBindingCallbackDataEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceAddressBindingCallbackDataEXT const & deviceAddressBindingCallbackDataEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceAddressBindingCallbackDataEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceAddressBindingCallbackDataEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, deviceAddressBindingCallbackDataEXT.flags );
      VULKAN_HPP_HASH_COMBINE( seed, deviceAddressBindingCallbackDataEXT.baseAddress );
      VULKAN_HPP_HASH_COMBINE( seed, deviceAddressBindingCallbackDataEXT.size );
      VULKAN_HPP_HASH_COMBINE( seed, deviceAddressBindingCallbackDataEXT.bindingType );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceBufferMemoryRequirements>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceBufferMemoryRequirements const & deviceBufferMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceBufferMemoryRequirements.sType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceBufferMemoryRequirements.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, deviceBufferMemoryRequirements.pCreateInfo );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceQueueCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceQueueCreateInfo const & deviceQueueCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceQueueCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceQueueCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, deviceQueueCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, deviceQueueCreateInfo.queueFamilyIndex );
      VULKAN_HPP_HASH_COMBINE( seed, deviceQueueCreateInfo.queueCount );
      VULKAN_HPP_HASH_COMBINE( seed, deviceQueueCreateInfo.pQueuePriorities );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures const & physicalDeviceFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.robustBufferAccess );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.fullDrawIndexUint32 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.imageCubeArray );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.independentBlend );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.geometryShader );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.tessellationShader );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.sampleRateShading );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.dualSrcBlend );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.logicOp );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.multiDrawIndirect );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.drawIndirectFirstInstance );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.depthClamp );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.depthBiasClamp );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.fillModeNonSolid );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.depthBounds );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.wideLines );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.largePoints );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.alphaToOne );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.multiViewport );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.samplerAnisotropy );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.textureCompressionETC2 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.textureCompressionASTC_LDR );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.textureCompressionBC );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.occlusionQueryPrecise );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.pipelineStatisticsQuery );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.vertexPipelineStoresAndAtomics );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.fragmentStoresAndAtomics );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.shaderTessellationAndGeometryPointSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.shaderImageGatherExtended );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.shaderStorageImageExtendedFormats );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.shaderStorageImageMultisample );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.shaderStorageImageReadWithoutFormat );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.shaderStorageImageWriteWithoutFormat );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.shaderUniformBufferArrayDynamicIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.shaderSampledImageArrayDynamicIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.shaderStorageBufferArrayDynamicIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.shaderStorageImageArrayDynamicIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.shaderClipDistance );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.shaderCullDistance );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.shaderFloat64 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.shaderInt64 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.shaderInt16 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.shaderResourceResidency );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.shaderResourceMinLod );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.sparseBinding );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.sparseResidencyBuffer );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.sparseResidencyImage2D );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.sparseResidencyImage3D );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.sparseResidency2Samples );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.sparseResidency4Samples );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.sparseResidency8Samples );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.sparseResidency16Samples );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.sparseResidencyAliased );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.variableMultisampleRate );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures.inheritedQueries );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceCreateInfo const & deviceCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, deviceCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, deviceCreateInfo.queueCreateInfoCount );
      VULKAN_HPP_HASH_COMBINE( seed, deviceCreateInfo.pQueueCreateInfos );
      VULKAN_HPP_HASH_COMBINE( seed, deviceCreateInfo.enabledLayerCount );
      for ( size_t i = 0; i < deviceCreateInfo.enabledLayerCount; ++i )
      {
        for ( const char * p = deviceCreateInfo.ppEnabledLayerNames[i]; *p != '\0'; ++p )
        {
          VULKAN_HPP_HASH_COMBINE( seed, *p );
        }
      }
      VULKAN_HPP_HASH_COMBINE( seed, deviceCreateInfo.enabledExtensionCount );
      for ( size_t i = 0; i < deviceCreateInfo.enabledExtensionCount; ++i )
      {
        for ( const char * p = deviceCreateInfo.ppEnabledExtensionNames[i]; *p != '\0'; ++p )
        {
          VULKAN_HPP_HASH_COMBINE( seed, *p );
        }
      }
      VULKAN_HPP_HASH_COMBINE( seed, deviceCreateInfo.pEnabledFeatures );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceDeviceMemoryReportCreateInfoEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::DeviceDeviceMemoryReportCreateInfoEXT const & deviceDeviceMemoryReportCreateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceDeviceMemoryReportCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceDeviceMemoryReportCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, deviceDeviceMemoryReportCreateInfoEXT.flags );
      VULKAN_HPP_HASH_COMBINE( seed, deviceDeviceMemoryReportCreateInfoEXT.pfnUserCallback );
      VULKAN_HPP_HASH_COMBINE( seed, deviceDeviceMemoryReportCreateInfoEXT.pUserData );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceDiagnosticsConfigCreateInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceDiagnosticsConfigCreateInfoNV const & deviceDiagnosticsConfigCreateInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceDiagnosticsConfigCreateInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceDiagnosticsConfigCreateInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, deviceDiagnosticsConfigCreateInfoNV.flags );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceEventInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceEventInfoEXT const & deviceEventInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceEventInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceEventInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, deviceEventInfoEXT.deviceEvent );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceFaultAddressInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceFaultAddressInfoEXT const & deviceFaultAddressInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceFaultAddressInfoEXT.addressType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceFaultAddressInfoEXT.reportedAddress );
      VULKAN_HPP_HASH_COMBINE( seed, deviceFaultAddressInfoEXT.addressPrecision );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceFaultCountsEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceFaultCountsEXT const & deviceFaultCountsEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceFaultCountsEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceFaultCountsEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, deviceFaultCountsEXT.addressInfoCount );
      VULKAN_HPP_HASH_COMBINE( seed, deviceFaultCountsEXT.vendorInfoCount );
      VULKAN_HPP_HASH_COMBINE( seed, deviceFaultCountsEXT.vendorBinarySize );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceFaultVendorInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceFaultVendorInfoEXT const & deviceFaultVendorInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      for ( size_t i = 0; i < VK_MAX_DESCRIPTION_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, deviceFaultVendorInfoEXT.description[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, deviceFaultVendorInfoEXT.vendorFaultCode );
      VULKAN_HPP_HASH_COMBINE( seed, deviceFaultVendorInfoEXT.vendorFaultData );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceFaultInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceFaultInfoEXT const & deviceFaultInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceFaultInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceFaultInfoEXT.pNext );
      for ( size_t i = 0; i < VK_MAX_DESCRIPTION_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, deviceFaultInfoEXT.description[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, deviceFaultInfoEXT.pAddressInfos );
      VULKAN_HPP_HASH_COMBINE( seed, deviceFaultInfoEXT.pVendorInfos );
      VULKAN_HPP_HASH_COMBINE( seed, deviceFaultInfoEXT.pVendorBinaryData );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceFaultVendorBinaryHeaderVersionOneEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceFaultVendorBinaryHeaderVersionOneEXT const & deviceFaultVendorBinaryHeaderVersionOneEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceFaultVendorBinaryHeaderVersionOneEXT.headerSize );
      VULKAN_HPP_HASH_COMBINE( seed, deviceFaultVendorBinaryHeaderVersionOneEXT.headerVersion );
      VULKAN_HPP_HASH_COMBINE( seed, deviceFaultVendorBinaryHeaderVersionOneEXT.vendorID );
      VULKAN_HPP_HASH_COMBINE( seed, deviceFaultVendorBinaryHeaderVersionOneEXT.deviceID );
      VULKAN_HPP_HASH_COMBINE( seed, deviceFaultVendorBinaryHeaderVersionOneEXT.driverVersion );
      for ( size_t i = 0; i < VK_UUID_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, deviceFaultVendorBinaryHeaderVersionOneEXT.pipelineCacheUUID[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, deviceFaultVendorBinaryHeaderVersionOneEXT.applicationNameOffset );
      VULKAN_HPP_HASH_COMBINE( seed, deviceFaultVendorBinaryHeaderVersionOneEXT.applicationVersion );
      VULKAN_HPP_HASH_COMBINE( seed, deviceFaultVendorBinaryHeaderVersionOneEXT.engineNameOffset );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceGroupBindSparseInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceGroupBindSparseInfo const & deviceGroupBindSparseInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupBindSparseInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupBindSparseInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupBindSparseInfo.resourceDeviceIndex );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupBindSparseInfo.memoryDeviceIndex );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceGroupCommandBufferBeginInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceGroupCommandBufferBeginInfo const & deviceGroupCommandBufferBeginInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupCommandBufferBeginInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupCommandBufferBeginInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupCommandBufferBeginInfo.deviceMask );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceGroupDeviceCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceGroupDeviceCreateInfo const & deviceGroupDeviceCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupDeviceCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupDeviceCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupDeviceCreateInfo.physicalDeviceCount );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupDeviceCreateInfo.pPhysicalDevices );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceGroupPresentCapabilitiesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceGroupPresentCapabilitiesKHR const & deviceGroupPresentCapabilitiesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupPresentCapabilitiesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupPresentCapabilitiesKHR.pNext );
      for ( size_t i = 0; i < VK_MAX_DEVICE_GROUP_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, deviceGroupPresentCapabilitiesKHR.presentMask[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupPresentCapabilitiesKHR.modes );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceGroupPresentInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceGroupPresentInfoKHR const & deviceGroupPresentInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupPresentInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupPresentInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupPresentInfoKHR.swapchainCount );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupPresentInfoKHR.pDeviceMasks );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupPresentInfoKHR.mode );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceGroupRenderPassBeginInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceGroupRenderPassBeginInfo const & deviceGroupRenderPassBeginInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupRenderPassBeginInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupRenderPassBeginInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupRenderPassBeginInfo.deviceMask );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupRenderPassBeginInfo.deviceRenderAreaCount );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupRenderPassBeginInfo.pDeviceRenderAreas );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceGroupSubmitInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceGroupSubmitInfo const & deviceGroupSubmitInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupSubmitInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupSubmitInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupSubmitInfo.waitSemaphoreCount );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupSubmitInfo.pWaitSemaphoreDeviceIndices );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupSubmitInfo.commandBufferCount );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupSubmitInfo.pCommandBufferDeviceMasks );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupSubmitInfo.signalSemaphoreCount );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupSubmitInfo.pSignalSemaphoreDeviceIndices );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceGroupSwapchainCreateInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceGroupSwapchainCreateInfoKHR const & deviceGroupSwapchainCreateInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupSwapchainCreateInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupSwapchainCreateInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, deviceGroupSwapchainCreateInfoKHR.modes );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageCreateInfo const & imageCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, imageCreateInfo.imageType );
      VULKAN_HPP_HASH_COMBINE( seed, imageCreateInfo.format );
      VULKAN_HPP_HASH_COMBINE( seed, imageCreateInfo.extent );
      VULKAN_HPP_HASH_COMBINE( seed, imageCreateInfo.mipLevels );
      VULKAN_HPP_HASH_COMBINE( seed, imageCreateInfo.arrayLayers );
      VULKAN_HPP_HASH_COMBINE( seed, imageCreateInfo.samples );
      VULKAN_HPP_HASH_COMBINE( seed, imageCreateInfo.tiling );
      VULKAN_HPP_HASH_COMBINE( seed, imageCreateInfo.usage );
      VULKAN_HPP_HASH_COMBINE( seed, imageCreateInfo.sharingMode );
      VULKAN_HPP_HASH_COMBINE( seed, imageCreateInfo.queueFamilyIndexCount );
      VULKAN_HPP_HASH_COMBINE( seed, imageCreateInfo.pQueueFamilyIndices );
      VULKAN_HPP_HASH_COMBINE( seed, imageCreateInfo.initialLayout );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceImageMemoryRequirements>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceImageMemoryRequirements const & deviceImageMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceImageMemoryRequirements.sType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceImageMemoryRequirements.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, deviceImageMemoryRequirements.pCreateInfo );
      VULKAN_HPP_HASH_COMBINE( seed, deviceImageMemoryRequirements.planeAspect );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceMemoryOpaqueCaptureAddressInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceMemoryOpaqueCaptureAddressInfo const & deviceMemoryOpaqueCaptureAddressInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceMemoryOpaqueCaptureAddressInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceMemoryOpaqueCaptureAddressInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, deviceMemoryOpaqueCaptureAddressInfo.memory );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceMemoryOverallocationCreateInfoAMD>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::DeviceMemoryOverallocationCreateInfoAMD const & deviceMemoryOverallocationCreateInfoAMD ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceMemoryOverallocationCreateInfoAMD.sType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceMemoryOverallocationCreateInfoAMD.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, deviceMemoryOverallocationCreateInfoAMD.overallocationBehavior );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceMemoryReportCallbackDataEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceMemoryReportCallbackDataEXT const & deviceMemoryReportCallbackDataEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceMemoryReportCallbackDataEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceMemoryReportCallbackDataEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, deviceMemoryReportCallbackDataEXT.flags );
      VULKAN_HPP_HASH_COMBINE( seed, deviceMemoryReportCallbackDataEXT.type );
      VULKAN_HPP_HASH_COMBINE( seed, deviceMemoryReportCallbackDataEXT.memoryObjectId );
      VULKAN_HPP_HASH_COMBINE( seed, deviceMemoryReportCallbackDataEXT.size );
      VULKAN_HPP_HASH_COMBINE( seed, deviceMemoryReportCallbackDataEXT.objectType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceMemoryReportCallbackDataEXT.objectHandle );
      VULKAN_HPP_HASH_COMBINE( seed, deviceMemoryReportCallbackDataEXT.heapIndex );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DevicePrivateDataCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DevicePrivateDataCreateInfo const & devicePrivateDataCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, devicePrivateDataCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, devicePrivateDataCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, devicePrivateDataCreateInfo.privateDataSlotRequestCount );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceQueueGlobalPriorityCreateInfoKHR>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::DeviceQueueGlobalPriorityCreateInfoKHR const & deviceQueueGlobalPriorityCreateInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceQueueGlobalPriorityCreateInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceQueueGlobalPriorityCreateInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, deviceQueueGlobalPriorityCreateInfoKHR.globalPriority );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DeviceQueueInfo2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DeviceQueueInfo2 const & deviceQueueInfo2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, deviceQueueInfo2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, deviceQueueInfo2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, deviceQueueInfo2.flags );
      VULKAN_HPP_HASH_COMBINE( seed, deviceQueueInfo2.queueFamilyIndex );
      VULKAN_HPP_HASH_COMBINE( seed, deviceQueueInfo2.queueIndex );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DirectFBSurfaceCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DirectFBSurfaceCreateInfoEXT const & directFBSurfaceCreateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, directFBSurfaceCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, directFBSurfaceCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, directFBSurfaceCreateInfoEXT.flags );
      VULKAN_HPP_HASH_COMBINE( seed, directFBSurfaceCreateInfoEXT.dfb );
      VULKAN_HPP_HASH_COMBINE( seed, directFBSurfaceCreateInfoEXT.surface );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DispatchIndirectCommand>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DispatchIndirectCommand const & dispatchIndirectCommand ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, dispatchIndirectCommand.x );
      VULKAN_HPP_HASH_COMBINE( seed, dispatchIndirectCommand.y );
      VULKAN_HPP_HASH_COMBINE( seed, dispatchIndirectCommand.z );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayEventInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DisplayEventInfoEXT const & displayEventInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, displayEventInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, displayEventInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, displayEventInfoEXT.displayEvent );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayModeParametersKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DisplayModeParametersKHR const & displayModeParametersKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, displayModeParametersKHR.visibleRegion );
      VULKAN_HPP_HASH_COMBINE( seed, displayModeParametersKHR.refreshRate );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayModeCreateInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DisplayModeCreateInfoKHR const & displayModeCreateInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, displayModeCreateInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, displayModeCreateInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, displayModeCreateInfoKHR.flags );
      VULKAN_HPP_HASH_COMBINE( seed, displayModeCreateInfoKHR.parameters );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayModePropertiesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DisplayModePropertiesKHR const & displayModePropertiesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, displayModePropertiesKHR.displayMode );
      VULKAN_HPP_HASH_COMBINE( seed, displayModePropertiesKHR.parameters );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayModeProperties2KHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DisplayModeProperties2KHR const & displayModeProperties2KHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, displayModeProperties2KHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, displayModeProperties2KHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, displayModeProperties2KHR.displayModeProperties );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayNativeHdrSurfaceCapabilitiesAMD>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::DisplayNativeHdrSurfaceCapabilitiesAMD const & displayNativeHdrSurfaceCapabilitiesAMD ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, displayNativeHdrSurfaceCapabilitiesAMD.sType );
      VULKAN_HPP_HASH_COMBINE( seed, displayNativeHdrSurfaceCapabilitiesAMD.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, displayNativeHdrSurfaceCapabilitiesAMD.localDimmingSupport );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayPlaneCapabilitiesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DisplayPlaneCapabilitiesKHR const & displayPlaneCapabilitiesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, displayPlaneCapabilitiesKHR.supportedAlpha );
      VULKAN_HPP_HASH_COMBINE( seed, displayPlaneCapabilitiesKHR.minSrcPosition );
      VULKAN_HPP_HASH_COMBINE( seed, displayPlaneCapabilitiesKHR.maxSrcPosition );
      VULKAN_HPP_HASH_COMBINE( seed, displayPlaneCapabilitiesKHR.minSrcExtent );
      VULKAN_HPP_HASH_COMBINE( seed, displayPlaneCapabilitiesKHR.maxSrcExtent );
      VULKAN_HPP_HASH_COMBINE( seed, displayPlaneCapabilitiesKHR.minDstPosition );
      VULKAN_HPP_HASH_COMBINE( seed, displayPlaneCapabilitiesKHR.maxDstPosition );
      VULKAN_HPP_HASH_COMBINE( seed, displayPlaneCapabilitiesKHR.minDstExtent );
      VULKAN_HPP_HASH_COMBINE( seed, displayPlaneCapabilitiesKHR.maxDstExtent );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayPlaneCapabilities2KHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DisplayPlaneCapabilities2KHR const & displayPlaneCapabilities2KHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, displayPlaneCapabilities2KHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, displayPlaneCapabilities2KHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, displayPlaneCapabilities2KHR.capabilities );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayPlaneInfo2KHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DisplayPlaneInfo2KHR const & displayPlaneInfo2KHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, displayPlaneInfo2KHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, displayPlaneInfo2KHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, displayPlaneInfo2KHR.mode );
      VULKAN_HPP_HASH_COMBINE( seed, displayPlaneInfo2KHR.planeIndex );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayPlanePropertiesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DisplayPlanePropertiesKHR const & displayPlanePropertiesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, displayPlanePropertiesKHR.currentDisplay );
      VULKAN_HPP_HASH_COMBINE( seed, displayPlanePropertiesKHR.currentStackIndex );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayPlaneProperties2KHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DisplayPlaneProperties2KHR const & displayPlaneProperties2KHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, displayPlaneProperties2KHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, displayPlaneProperties2KHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, displayPlaneProperties2KHR.displayPlaneProperties );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayPowerInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DisplayPowerInfoEXT const & displayPowerInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, displayPowerInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, displayPowerInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, displayPowerInfoEXT.powerState );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayPresentInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DisplayPresentInfoKHR const & displayPresentInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, displayPresentInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, displayPresentInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, displayPresentInfoKHR.srcRect );
      VULKAN_HPP_HASH_COMBINE( seed, displayPresentInfoKHR.dstRect );
      VULKAN_HPP_HASH_COMBINE( seed, displayPresentInfoKHR.persistent );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayPropertiesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DisplayPropertiesKHR const & displayPropertiesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, displayPropertiesKHR.display );
      for ( const char * p = displayPropertiesKHR.displayName; *p != '\0'; ++p )
      {
        VULKAN_HPP_HASH_COMBINE( seed, *p );
      }
      VULKAN_HPP_HASH_COMBINE( seed, displayPropertiesKHR.physicalDimensions );
      VULKAN_HPP_HASH_COMBINE( seed, displayPropertiesKHR.physicalResolution );
      VULKAN_HPP_HASH_COMBINE( seed, displayPropertiesKHR.supportedTransforms );
      VULKAN_HPP_HASH_COMBINE( seed, displayPropertiesKHR.planeReorderPossible );
      VULKAN_HPP_HASH_COMBINE( seed, displayPropertiesKHR.persistentContent );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplayProperties2KHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DisplayProperties2KHR const & displayProperties2KHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, displayProperties2KHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, displayProperties2KHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, displayProperties2KHR.displayProperties );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DisplaySurfaceCreateInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DisplaySurfaceCreateInfoKHR const & displaySurfaceCreateInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, displaySurfaceCreateInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, displaySurfaceCreateInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, displaySurfaceCreateInfoKHR.flags );
      VULKAN_HPP_HASH_COMBINE( seed, displaySurfaceCreateInfoKHR.displayMode );
      VULKAN_HPP_HASH_COMBINE( seed, displaySurfaceCreateInfoKHR.planeIndex );
      VULKAN_HPP_HASH_COMBINE( seed, displaySurfaceCreateInfoKHR.planeStackIndex );
      VULKAN_HPP_HASH_COMBINE( seed, displaySurfaceCreateInfoKHR.transform );
      VULKAN_HPP_HASH_COMBINE( seed, displaySurfaceCreateInfoKHR.globalAlpha );
      VULKAN_HPP_HASH_COMBINE( seed, displaySurfaceCreateInfoKHR.alphaMode );
      VULKAN_HPP_HASH_COMBINE( seed, displaySurfaceCreateInfoKHR.imageExtent );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DrawIndexedIndirectCommand>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DrawIndexedIndirectCommand const & drawIndexedIndirectCommand ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, drawIndexedIndirectCommand.indexCount );
      VULKAN_HPP_HASH_COMBINE( seed, drawIndexedIndirectCommand.instanceCount );
      VULKAN_HPP_HASH_COMBINE( seed, drawIndexedIndirectCommand.firstIndex );
      VULKAN_HPP_HASH_COMBINE( seed, drawIndexedIndirectCommand.vertexOffset );
      VULKAN_HPP_HASH_COMBINE( seed, drawIndexedIndirectCommand.firstInstance );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DrawIndirectCommand>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DrawIndirectCommand const & drawIndirectCommand ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, drawIndirectCommand.vertexCount );
      VULKAN_HPP_HASH_COMBINE( seed, drawIndirectCommand.instanceCount );
      VULKAN_HPP_HASH_COMBINE( seed, drawIndirectCommand.firstVertex );
      VULKAN_HPP_HASH_COMBINE( seed, drawIndirectCommand.firstInstance );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DrawMeshTasksIndirectCommandEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DrawMeshTasksIndirectCommandEXT const & drawMeshTasksIndirectCommandEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, drawMeshTasksIndirectCommandEXT.groupCountX );
      VULKAN_HPP_HASH_COMBINE( seed, drawMeshTasksIndirectCommandEXT.groupCountY );
      VULKAN_HPP_HASH_COMBINE( seed, drawMeshTasksIndirectCommandEXT.groupCountZ );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DrawMeshTasksIndirectCommandNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DrawMeshTasksIndirectCommandNV const & drawMeshTasksIndirectCommandNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, drawMeshTasksIndirectCommandNV.taskCount );
      VULKAN_HPP_HASH_COMBINE( seed, drawMeshTasksIndirectCommandNV.firstTask );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DrmFormatModifierProperties2EXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DrmFormatModifierProperties2EXT const & drmFormatModifierProperties2EXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, drmFormatModifierProperties2EXT.drmFormatModifier );
      VULKAN_HPP_HASH_COMBINE( seed, drmFormatModifierProperties2EXT.drmFormatModifierPlaneCount );
      VULKAN_HPP_HASH_COMBINE( seed, drmFormatModifierProperties2EXT.drmFormatModifierTilingFeatures );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DrmFormatModifierPropertiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DrmFormatModifierPropertiesEXT const & drmFormatModifierPropertiesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, drmFormatModifierPropertiesEXT.drmFormatModifier );
      VULKAN_HPP_HASH_COMBINE( seed, drmFormatModifierPropertiesEXT.drmFormatModifierPlaneCount );
      VULKAN_HPP_HASH_COMBINE( seed, drmFormatModifierPropertiesEXT.drmFormatModifierTilingFeatures );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DrmFormatModifierPropertiesList2EXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DrmFormatModifierPropertiesList2EXT const & drmFormatModifierPropertiesList2EXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, drmFormatModifierPropertiesList2EXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, drmFormatModifierPropertiesList2EXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, drmFormatModifierPropertiesList2EXT.drmFormatModifierCount );
      VULKAN_HPP_HASH_COMBINE( seed, drmFormatModifierPropertiesList2EXT.pDrmFormatModifierProperties );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::DrmFormatModifierPropertiesListEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::DrmFormatModifierPropertiesListEXT const & drmFormatModifierPropertiesListEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, drmFormatModifierPropertiesListEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, drmFormatModifierPropertiesListEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, drmFormatModifierPropertiesListEXT.drmFormatModifierCount );
      VULKAN_HPP_HASH_COMBINE( seed, drmFormatModifierPropertiesListEXT.pDrmFormatModifierProperties );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::EventCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::EventCreateInfo const & eventCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, eventCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, eventCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, eventCreateInfo.flags );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportFenceCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExportFenceCreateInfo const & exportFenceCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, exportFenceCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, exportFenceCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, exportFenceCreateInfo.handleTypes );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportFenceWin32HandleInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExportFenceWin32HandleInfoKHR const & exportFenceWin32HandleInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, exportFenceWin32HandleInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, exportFenceWin32HandleInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, exportFenceWin32HandleInfoKHR.pAttributes );
      VULKAN_HPP_HASH_COMBINE( seed, exportFenceWin32HandleInfoKHR.dwAccess );
      VULKAN_HPP_HASH_COMBINE( seed, exportFenceWin32HandleInfoKHR.name );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportMemoryAllocateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExportMemoryAllocateInfo const & exportMemoryAllocateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, exportMemoryAllocateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, exportMemoryAllocateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, exportMemoryAllocateInfo.handleTypes );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportMemoryAllocateInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExportMemoryAllocateInfoNV const & exportMemoryAllocateInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, exportMemoryAllocateInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, exportMemoryAllocateInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, exportMemoryAllocateInfoNV.handleTypes );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportMemoryWin32HandleInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExportMemoryWin32HandleInfoKHR const & exportMemoryWin32HandleInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, exportMemoryWin32HandleInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, exportMemoryWin32HandleInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, exportMemoryWin32HandleInfoKHR.pAttributes );
      VULKAN_HPP_HASH_COMBINE( seed, exportMemoryWin32HandleInfoKHR.dwAccess );
      VULKAN_HPP_HASH_COMBINE( seed, exportMemoryWin32HandleInfoKHR.name );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportMemoryWin32HandleInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExportMemoryWin32HandleInfoNV const & exportMemoryWin32HandleInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, exportMemoryWin32HandleInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, exportMemoryWin32HandleInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, exportMemoryWin32HandleInfoNV.pAttributes );
      VULKAN_HPP_HASH_COMBINE( seed, exportMemoryWin32HandleInfoNV.dwAccess );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

#  if defined( VK_USE_PLATFORM_METAL_EXT )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportMetalBufferInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExportMetalBufferInfoEXT const & exportMetalBufferInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalBufferInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalBufferInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalBufferInfoEXT.memory );
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalBufferInfoEXT.mtlBuffer );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_METAL_EXT*/

#  if defined( VK_USE_PLATFORM_METAL_EXT )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportMetalCommandQueueInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExportMetalCommandQueueInfoEXT const & exportMetalCommandQueueInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalCommandQueueInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalCommandQueueInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalCommandQueueInfoEXT.queue );
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalCommandQueueInfoEXT.mtlCommandQueue );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_METAL_EXT*/

#  if defined( VK_USE_PLATFORM_METAL_EXT )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportMetalDeviceInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExportMetalDeviceInfoEXT const & exportMetalDeviceInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalDeviceInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalDeviceInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalDeviceInfoEXT.mtlDevice );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_METAL_EXT*/

#  if defined( VK_USE_PLATFORM_METAL_EXT )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportMetalIOSurfaceInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExportMetalIOSurfaceInfoEXT const & exportMetalIOSurfaceInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalIOSurfaceInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalIOSurfaceInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalIOSurfaceInfoEXT.image );
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalIOSurfaceInfoEXT.ioSurface );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_METAL_EXT*/

#  if defined( VK_USE_PLATFORM_METAL_EXT )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportMetalObjectCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExportMetalObjectCreateInfoEXT const & exportMetalObjectCreateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalObjectCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalObjectCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalObjectCreateInfoEXT.exportObjectType );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_METAL_EXT*/

#  if defined( VK_USE_PLATFORM_METAL_EXT )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportMetalObjectsInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExportMetalObjectsInfoEXT const & exportMetalObjectsInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalObjectsInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalObjectsInfoEXT.pNext );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_METAL_EXT*/

#  if defined( VK_USE_PLATFORM_METAL_EXT )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportMetalSharedEventInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExportMetalSharedEventInfoEXT const & exportMetalSharedEventInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalSharedEventInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalSharedEventInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalSharedEventInfoEXT.semaphore );
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalSharedEventInfoEXT.event );
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalSharedEventInfoEXT.mtlSharedEvent );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_METAL_EXT*/

#  if defined( VK_USE_PLATFORM_METAL_EXT )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportMetalTextureInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExportMetalTextureInfoEXT const & exportMetalTextureInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalTextureInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalTextureInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalTextureInfoEXT.image );
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalTextureInfoEXT.imageView );
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalTextureInfoEXT.bufferView );
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalTextureInfoEXT.plane );
      VULKAN_HPP_HASH_COMBINE( seed, exportMetalTextureInfoEXT.mtlTexture );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_METAL_EXT*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportSemaphoreCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExportSemaphoreCreateInfo const & exportSemaphoreCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, exportSemaphoreCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, exportSemaphoreCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, exportSemaphoreCreateInfo.handleTypes );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExportSemaphoreWin32HandleInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExportSemaphoreWin32HandleInfoKHR const & exportSemaphoreWin32HandleInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, exportSemaphoreWin32HandleInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, exportSemaphoreWin32HandleInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, exportSemaphoreWin32HandleInfoKHR.pAttributes );
      VULKAN_HPP_HASH_COMBINE( seed, exportSemaphoreWin32HandleInfoKHR.dwAccess );
      VULKAN_HPP_HASH_COMBINE( seed, exportSemaphoreWin32HandleInfoKHR.name );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExtensionProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExtensionProperties const & extensionProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      for ( size_t i = 0; i < VK_MAX_EXTENSION_NAME_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, extensionProperties.extensionName[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, extensionProperties.specVersion );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalMemoryProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExternalMemoryProperties const & externalMemoryProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, externalMemoryProperties.externalMemoryFeatures );
      VULKAN_HPP_HASH_COMBINE( seed, externalMemoryProperties.exportFromImportedHandleTypes );
      VULKAN_HPP_HASH_COMBINE( seed, externalMemoryProperties.compatibleHandleTypes );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalBufferProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExternalBufferProperties const & externalBufferProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, externalBufferProperties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, externalBufferProperties.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, externalBufferProperties.externalMemoryProperties );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalFenceProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExternalFenceProperties const & externalFenceProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, externalFenceProperties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, externalFenceProperties.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, externalFenceProperties.exportFromImportedHandleTypes );
      VULKAN_HPP_HASH_COMBINE( seed, externalFenceProperties.compatibleHandleTypes );
      VULKAN_HPP_HASH_COMBINE( seed, externalFenceProperties.externalFenceFeatures );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_ANDROID_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalFormatANDROID>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExternalFormatANDROID const & externalFormatANDROID ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, externalFormatANDROID.sType );
      VULKAN_HPP_HASH_COMBINE( seed, externalFormatANDROID.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, externalFormatANDROID.externalFormat );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_ANDROID_KHR*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalImageFormatProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExternalImageFormatProperties const & externalImageFormatProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, externalImageFormatProperties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, externalImageFormatProperties.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, externalImageFormatProperties.externalMemoryProperties );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageFormatProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageFormatProperties const & imageFormatProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageFormatProperties.maxExtent );
      VULKAN_HPP_HASH_COMBINE( seed, imageFormatProperties.maxMipLevels );
      VULKAN_HPP_HASH_COMBINE( seed, imageFormatProperties.maxArrayLayers );
      VULKAN_HPP_HASH_COMBINE( seed, imageFormatProperties.sampleCounts );
      VULKAN_HPP_HASH_COMBINE( seed, imageFormatProperties.maxResourceSize );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalImageFormatPropertiesNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExternalImageFormatPropertiesNV const & externalImageFormatPropertiesNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, externalImageFormatPropertiesNV.imageFormatProperties );
      VULKAN_HPP_HASH_COMBINE( seed, externalImageFormatPropertiesNV.externalMemoryFeatures );
      VULKAN_HPP_HASH_COMBINE( seed, externalImageFormatPropertiesNV.exportFromImportedHandleTypes );
      VULKAN_HPP_HASH_COMBINE( seed, externalImageFormatPropertiesNV.compatibleHandleTypes );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalMemoryBufferCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExternalMemoryBufferCreateInfo const & externalMemoryBufferCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, externalMemoryBufferCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, externalMemoryBufferCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, externalMemoryBufferCreateInfo.handleTypes );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalMemoryImageCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExternalMemoryImageCreateInfo const & externalMemoryImageCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, externalMemoryImageCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, externalMemoryImageCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, externalMemoryImageCreateInfo.handleTypes );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalMemoryImageCreateInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExternalMemoryImageCreateInfoNV const & externalMemoryImageCreateInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, externalMemoryImageCreateInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, externalMemoryImageCreateInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, externalMemoryImageCreateInfoNV.handleTypes );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ExternalSemaphoreProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ExternalSemaphoreProperties const & externalSemaphoreProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, externalSemaphoreProperties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, externalSemaphoreProperties.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, externalSemaphoreProperties.exportFromImportedHandleTypes );
      VULKAN_HPP_HASH_COMBINE( seed, externalSemaphoreProperties.compatibleHandleTypes );
      VULKAN_HPP_HASH_COMBINE( seed, externalSemaphoreProperties.externalSemaphoreFeatures );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FenceCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::FenceCreateInfo const & fenceCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, fenceCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, fenceCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, fenceCreateInfo.flags );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FenceGetFdInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::FenceGetFdInfoKHR const & fenceGetFdInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, fenceGetFdInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, fenceGetFdInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, fenceGetFdInfoKHR.fence );
      VULKAN_HPP_HASH_COMBINE( seed, fenceGetFdInfoKHR.handleType );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FenceGetWin32HandleInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::FenceGetWin32HandleInfoKHR const & fenceGetWin32HandleInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, fenceGetWin32HandleInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, fenceGetWin32HandleInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, fenceGetWin32HandleInfoKHR.fence );
      VULKAN_HPP_HASH_COMBINE( seed, fenceGetWin32HandleInfoKHR.handleType );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FilterCubicImageViewImageFormatPropertiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::FilterCubicImageViewImageFormatPropertiesEXT const & filterCubicImageViewImageFormatPropertiesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, filterCubicImageViewImageFormatPropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, filterCubicImageViewImageFormatPropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, filterCubicImageViewImageFormatPropertiesEXT.filterCubic );
      VULKAN_HPP_HASH_COMBINE( seed, filterCubicImageViewImageFormatPropertiesEXT.filterCubicMinmax );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FormatProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::FormatProperties const & formatProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, formatProperties.linearTilingFeatures );
      VULKAN_HPP_HASH_COMBINE( seed, formatProperties.optimalTilingFeatures );
      VULKAN_HPP_HASH_COMBINE( seed, formatProperties.bufferFeatures );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FormatProperties2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::FormatProperties2 const & formatProperties2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, formatProperties2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, formatProperties2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, formatProperties2.formatProperties );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FormatProperties3>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::FormatProperties3 const & formatProperties3 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, formatProperties3.sType );
      VULKAN_HPP_HASH_COMBINE( seed, formatProperties3.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, formatProperties3.linearTilingFeatures );
      VULKAN_HPP_HASH_COMBINE( seed, formatProperties3.optimalTilingFeatures );
      VULKAN_HPP_HASH_COMBINE( seed, formatProperties3.bufferFeatures );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FragmentShadingRateAttachmentInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::FragmentShadingRateAttachmentInfoKHR const & fragmentShadingRateAttachmentInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, fragmentShadingRateAttachmentInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, fragmentShadingRateAttachmentInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, fragmentShadingRateAttachmentInfoKHR.pFragmentShadingRateAttachment );
      VULKAN_HPP_HASH_COMBINE( seed, fragmentShadingRateAttachmentInfoKHR.shadingRateAttachmentTexelSize );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FramebufferAttachmentImageInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::FramebufferAttachmentImageInfo const & framebufferAttachmentImageInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, framebufferAttachmentImageInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, framebufferAttachmentImageInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, framebufferAttachmentImageInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, framebufferAttachmentImageInfo.usage );
      VULKAN_HPP_HASH_COMBINE( seed, framebufferAttachmentImageInfo.width );
      VULKAN_HPP_HASH_COMBINE( seed, framebufferAttachmentImageInfo.height );
      VULKAN_HPP_HASH_COMBINE( seed, framebufferAttachmentImageInfo.layerCount );
      VULKAN_HPP_HASH_COMBINE( seed, framebufferAttachmentImageInfo.viewFormatCount );
      VULKAN_HPP_HASH_COMBINE( seed, framebufferAttachmentImageInfo.pViewFormats );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FramebufferAttachmentsCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::FramebufferAttachmentsCreateInfo const & framebufferAttachmentsCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, framebufferAttachmentsCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, framebufferAttachmentsCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, framebufferAttachmentsCreateInfo.attachmentImageInfoCount );
      VULKAN_HPP_HASH_COMBINE( seed, framebufferAttachmentsCreateInfo.pAttachmentImageInfos );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FramebufferCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::FramebufferCreateInfo const & framebufferCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, framebufferCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, framebufferCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, framebufferCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, framebufferCreateInfo.renderPass );
      VULKAN_HPP_HASH_COMBINE( seed, framebufferCreateInfo.attachmentCount );
      VULKAN_HPP_HASH_COMBINE( seed, framebufferCreateInfo.pAttachments );
      VULKAN_HPP_HASH_COMBINE( seed, framebufferCreateInfo.width );
      VULKAN_HPP_HASH_COMBINE( seed, framebufferCreateInfo.height );
      VULKAN_HPP_HASH_COMBINE( seed, framebufferCreateInfo.layers );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::FramebufferMixedSamplesCombinationNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::FramebufferMixedSamplesCombinationNV const & framebufferMixedSamplesCombinationNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, framebufferMixedSamplesCombinationNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, framebufferMixedSamplesCombinationNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, framebufferMixedSamplesCombinationNV.coverageReductionMode );
      VULKAN_HPP_HASH_COMBINE( seed, framebufferMixedSamplesCombinationNV.rasterizationSamples );
      VULKAN_HPP_HASH_COMBINE( seed, framebufferMixedSamplesCombinationNV.depthStencilSamples );
      VULKAN_HPP_HASH_COMBINE( seed, framebufferMixedSamplesCombinationNV.colorSamples );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IndirectCommandsStreamNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::IndirectCommandsStreamNV const & indirectCommandsStreamNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsStreamNV.buffer );
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsStreamNV.offset );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GeneratedCommandsInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::GeneratedCommandsInfoNV const & generatedCommandsInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, generatedCommandsInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, generatedCommandsInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, generatedCommandsInfoNV.pipelineBindPoint );
      VULKAN_HPP_HASH_COMBINE( seed, generatedCommandsInfoNV.pipeline );
      VULKAN_HPP_HASH_COMBINE( seed, generatedCommandsInfoNV.indirectCommandsLayout );
      VULKAN_HPP_HASH_COMBINE( seed, generatedCommandsInfoNV.streamCount );
      VULKAN_HPP_HASH_COMBINE( seed, generatedCommandsInfoNV.pStreams );
      VULKAN_HPP_HASH_COMBINE( seed, generatedCommandsInfoNV.sequencesCount );
      VULKAN_HPP_HASH_COMBINE( seed, generatedCommandsInfoNV.preprocessBuffer );
      VULKAN_HPP_HASH_COMBINE( seed, generatedCommandsInfoNV.preprocessOffset );
      VULKAN_HPP_HASH_COMBINE( seed, generatedCommandsInfoNV.preprocessSize );
      VULKAN_HPP_HASH_COMBINE( seed, generatedCommandsInfoNV.sequencesCountBuffer );
      VULKAN_HPP_HASH_COMBINE( seed, generatedCommandsInfoNV.sequencesCountOffset );
      VULKAN_HPP_HASH_COMBINE( seed, generatedCommandsInfoNV.sequencesIndexBuffer );
      VULKAN_HPP_HASH_COMBINE( seed, generatedCommandsInfoNV.sequencesIndexOffset );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GeneratedCommandsMemoryRequirementsInfoNV>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::GeneratedCommandsMemoryRequirementsInfoNV const & generatedCommandsMemoryRequirementsInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, generatedCommandsMemoryRequirementsInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, generatedCommandsMemoryRequirementsInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, generatedCommandsMemoryRequirementsInfoNV.pipelineBindPoint );
      VULKAN_HPP_HASH_COMBINE( seed, generatedCommandsMemoryRequirementsInfoNV.pipeline );
      VULKAN_HPP_HASH_COMBINE( seed, generatedCommandsMemoryRequirementsInfoNV.indirectCommandsLayout );
      VULKAN_HPP_HASH_COMBINE( seed, generatedCommandsMemoryRequirementsInfoNV.maxSequencesCount );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VertexInputBindingDescription>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VertexInputBindingDescription const & vertexInputBindingDescription ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, vertexInputBindingDescription.binding );
      VULKAN_HPP_HASH_COMBINE( seed, vertexInputBindingDescription.stride );
      VULKAN_HPP_HASH_COMBINE( seed, vertexInputBindingDescription.inputRate );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VertexInputAttributeDescription>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VertexInputAttributeDescription const & vertexInputAttributeDescription ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, vertexInputAttributeDescription.location );
      VULKAN_HPP_HASH_COMBINE( seed, vertexInputAttributeDescription.binding );
      VULKAN_HPP_HASH_COMBINE( seed, vertexInputAttributeDescription.format );
      VULKAN_HPP_HASH_COMBINE( seed, vertexInputAttributeDescription.offset );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineVertexInputStateCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineVertexInputStateCreateInfo const & pipelineVertexInputStateCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineVertexInputStateCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineVertexInputStateCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineVertexInputStateCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineVertexInputStateCreateInfo.vertexBindingDescriptionCount );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineVertexInputStateCreateInfo.pVertexBindingDescriptions );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineVertexInputStateCreateInfo.vertexAttributeDescriptionCount );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineVertexInputStateCreateInfo.pVertexAttributeDescriptions );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineInputAssemblyStateCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineInputAssemblyStateCreateInfo const & pipelineInputAssemblyStateCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineInputAssemblyStateCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineInputAssemblyStateCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineInputAssemblyStateCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineInputAssemblyStateCreateInfo.topology );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineInputAssemblyStateCreateInfo.primitiveRestartEnable );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineTessellationStateCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineTessellationStateCreateInfo const & pipelineTessellationStateCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineTessellationStateCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineTessellationStateCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineTessellationStateCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineTessellationStateCreateInfo.patchControlPoints );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineViewportStateCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineViewportStateCreateInfo const & pipelineViewportStateCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportStateCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportStateCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportStateCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportStateCreateInfo.viewportCount );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportStateCreateInfo.pViewports );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportStateCreateInfo.scissorCount );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportStateCreateInfo.pScissors );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineRasterizationStateCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineRasterizationStateCreateInfo const & pipelineRasterizationStateCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationStateCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationStateCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationStateCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationStateCreateInfo.depthClampEnable );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationStateCreateInfo.rasterizerDiscardEnable );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationStateCreateInfo.polygonMode );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationStateCreateInfo.cullMode );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationStateCreateInfo.frontFace );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationStateCreateInfo.depthBiasEnable );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationStateCreateInfo.depthBiasConstantFactor );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationStateCreateInfo.depthBiasClamp );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationStateCreateInfo.depthBiasSlopeFactor );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationStateCreateInfo.lineWidth );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineMultisampleStateCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineMultisampleStateCreateInfo const & pipelineMultisampleStateCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineMultisampleStateCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineMultisampleStateCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineMultisampleStateCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineMultisampleStateCreateInfo.rasterizationSamples );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineMultisampleStateCreateInfo.sampleShadingEnable );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineMultisampleStateCreateInfo.minSampleShading );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineMultisampleStateCreateInfo.pSampleMask );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineMultisampleStateCreateInfo.alphaToCoverageEnable );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineMultisampleStateCreateInfo.alphaToOneEnable );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::StencilOpState>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::StencilOpState const & stencilOpState ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, stencilOpState.failOp );
      VULKAN_HPP_HASH_COMBINE( seed, stencilOpState.passOp );
      VULKAN_HPP_HASH_COMBINE( seed, stencilOpState.depthFailOp );
      VULKAN_HPP_HASH_COMBINE( seed, stencilOpState.compareOp );
      VULKAN_HPP_HASH_COMBINE( seed, stencilOpState.compareMask );
      VULKAN_HPP_HASH_COMBINE( seed, stencilOpState.writeMask );
      VULKAN_HPP_HASH_COMBINE( seed, stencilOpState.reference );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineDepthStencilStateCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineDepthStencilStateCreateInfo const & pipelineDepthStencilStateCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineDepthStencilStateCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineDepthStencilStateCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineDepthStencilStateCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineDepthStencilStateCreateInfo.depthTestEnable );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineDepthStencilStateCreateInfo.depthWriteEnable );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineDepthStencilStateCreateInfo.depthCompareOp );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineDepthStencilStateCreateInfo.depthBoundsTestEnable );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineDepthStencilStateCreateInfo.stencilTestEnable );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineDepthStencilStateCreateInfo.front );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineDepthStencilStateCreateInfo.back );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineDepthStencilStateCreateInfo.minDepthBounds );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineDepthStencilStateCreateInfo.maxDepthBounds );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineColorBlendAttachmentState>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineColorBlendAttachmentState const & pipelineColorBlendAttachmentState ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineColorBlendAttachmentState.blendEnable );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineColorBlendAttachmentState.srcColorBlendFactor );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineColorBlendAttachmentState.dstColorBlendFactor );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineColorBlendAttachmentState.colorBlendOp );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineColorBlendAttachmentState.srcAlphaBlendFactor );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineColorBlendAttachmentState.dstAlphaBlendFactor );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineColorBlendAttachmentState.alphaBlendOp );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineColorBlendAttachmentState.colorWriteMask );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineColorBlendStateCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineColorBlendStateCreateInfo const & pipelineColorBlendStateCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineColorBlendStateCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineColorBlendStateCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineColorBlendStateCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineColorBlendStateCreateInfo.logicOpEnable );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineColorBlendStateCreateInfo.logicOp );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineColorBlendStateCreateInfo.attachmentCount );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineColorBlendStateCreateInfo.pAttachments );
      for ( size_t i = 0; i < 4; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, pipelineColorBlendStateCreateInfo.blendConstants[i] );
      }
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineDynamicStateCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineDynamicStateCreateInfo const & pipelineDynamicStateCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineDynamicStateCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineDynamicStateCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineDynamicStateCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineDynamicStateCreateInfo.dynamicStateCount );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineDynamicStateCreateInfo.pDynamicStates );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GraphicsPipelineCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::GraphicsPipelineCreateInfo const & graphicsPipelineCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineCreateInfo.stageCount );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineCreateInfo.pStages );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineCreateInfo.pVertexInputState );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineCreateInfo.pInputAssemblyState );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineCreateInfo.pTessellationState );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineCreateInfo.pViewportState );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineCreateInfo.pRasterizationState );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineCreateInfo.pMultisampleState );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineCreateInfo.pDepthStencilState );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineCreateInfo.pColorBlendState );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineCreateInfo.pDynamicState );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineCreateInfo.layout );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineCreateInfo.renderPass );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineCreateInfo.subpass );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineCreateInfo.basePipelineHandle );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineCreateInfo.basePipelineIndex );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GraphicsPipelineLibraryCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::GraphicsPipelineLibraryCreateInfoEXT const & graphicsPipelineLibraryCreateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineLibraryCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineLibraryCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineLibraryCreateInfoEXT.flags );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GraphicsShaderGroupCreateInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::GraphicsShaderGroupCreateInfoNV const & graphicsShaderGroupCreateInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, graphicsShaderGroupCreateInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsShaderGroupCreateInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsShaderGroupCreateInfoNV.stageCount );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsShaderGroupCreateInfoNV.pStages );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsShaderGroupCreateInfoNV.pVertexInputState );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsShaderGroupCreateInfoNV.pTessellationState );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::GraphicsPipelineShaderGroupsCreateInfoNV>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::GraphicsPipelineShaderGroupsCreateInfoNV const & graphicsPipelineShaderGroupsCreateInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineShaderGroupsCreateInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineShaderGroupsCreateInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineShaderGroupsCreateInfoNV.groupCount );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineShaderGroupsCreateInfoNV.pGroups );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineShaderGroupsCreateInfoNV.pipelineCount );
      VULKAN_HPP_HASH_COMBINE( seed, graphicsPipelineShaderGroupsCreateInfoNV.pPipelines );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::XYColorEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::XYColorEXT const & xYColorEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, xYColorEXT.x );
      VULKAN_HPP_HASH_COMBINE( seed, xYColorEXT.y );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::HdrMetadataEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::HdrMetadataEXT const & hdrMetadataEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, hdrMetadataEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, hdrMetadataEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, hdrMetadataEXT.displayPrimaryRed );
      VULKAN_HPP_HASH_COMBINE( seed, hdrMetadataEXT.displayPrimaryGreen );
      VULKAN_HPP_HASH_COMBINE( seed, hdrMetadataEXT.displayPrimaryBlue );
      VULKAN_HPP_HASH_COMBINE( seed, hdrMetadataEXT.whitePoint );
      VULKAN_HPP_HASH_COMBINE( seed, hdrMetadataEXT.maxLuminance );
      VULKAN_HPP_HASH_COMBINE( seed, hdrMetadataEXT.minLuminance );
      VULKAN_HPP_HASH_COMBINE( seed, hdrMetadataEXT.maxContentLightLevel );
      VULKAN_HPP_HASH_COMBINE( seed, hdrMetadataEXT.maxFrameAverageLightLevel );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::HeadlessSurfaceCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::HeadlessSurfaceCreateInfoEXT const & headlessSurfaceCreateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, headlessSurfaceCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, headlessSurfaceCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, headlessSurfaceCreateInfoEXT.flags );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_IOS_MVK )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IOSSurfaceCreateInfoMVK>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::IOSSurfaceCreateInfoMVK const & iOSSurfaceCreateInfoMVK ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, iOSSurfaceCreateInfoMVK.sType );
      VULKAN_HPP_HASH_COMBINE( seed, iOSSurfaceCreateInfoMVK.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, iOSSurfaceCreateInfoMVK.flags );
      VULKAN_HPP_HASH_COMBINE( seed, iOSSurfaceCreateInfoMVK.pView );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_IOS_MVK*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageBlit>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageBlit const & imageBlit ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageBlit.srcSubresource );
      for ( size_t i = 0; i < 2; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, imageBlit.srcOffsets[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, imageBlit.dstSubresource );
      for ( size_t i = 0; i < 2; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, imageBlit.dstOffsets[i] );
      }
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageCompressionControlEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageCompressionControlEXT const & imageCompressionControlEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageCompressionControlEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageCompressionControlEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageCompressionControlEXT.flags );
      VULKAN_HPP_HASH_COMBINE( seed, imageCompressionControlEXT.compressionControlPlaneCount );
      VULKAN_HPP_HASH_COMBINE( seed, imageCompressionControlEXT.pFixedRateFlags );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageCompressionPropertiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageCompressionPropertiesEXT const & imageCompressionPropertiesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageCompressionPropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageCompressionPropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageCompressionPropertiesEXT.imageCompressionFlags );
      VULKAN_HPP_HASH_COMBINE( seed, imageCompressionPropertiesEXT.imageCompressionFixedRateFlags );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_FUCHSIA )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageFormatConstraintsInfoFUCHSIA>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageFormatConstraintsInfoFUCHSIA const & imageFormatConstraintsInfoFUCHSIA ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageFormatConstraintsInfoFUCHSIA.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageFormatConstraintsInfoFUCHSIA.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageFormatConstraintsInfoFUCHSIA.imageCreateInfo );
      VULKAN_HPP_HASH_COMBINE( seed, imageFormatConstraintsInfoFUCHSIA.requiredFormatFeatures );
      VULKAN_HPP_HASH_COMBINE( seed, imageFormatConstraintsInfoFUCHSIA.flags );
      VULKAN_HPP_HASH_COMBINE( seed, imageFormatConstraintsInfoFUCHSIA.sysmemPixelFormat );
      VULKAN_HPP_HASH_COMBINE( seed, imageFormatConstraintsInfoFUCHSIA.colorSpaceCount );
      VULKAN_HPP_HASH_COMBINE( seed, imageFormatConstraintsInfoFUCHSIA.pColorSpaces );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageConstraintsInfoFUCHSIA>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageConstraintsInfoFUCHSIA const & imageConstraintsInfoFUCHSIA ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageConstraintsInfoFUCHSIA.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageConstraintsInfoFUCHSIA.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageConstraintsInfoFUCHSIA.formatConstraintsCount );
      VULKAN_HPP_HASH_COMBINE( seed, imageConstraintsInfoFUCHSIA.pFormatConstraints );
      VULKAN_HPP_HASH_COMBINE( seed, imageConstraintsInfoFUCHSIA.bufferCollectionConstraints );
      VULKAN_HPP_HASH_COMBINE( seed, imageConstraintsInfoFUCHSIA.flags );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageCopy>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageCopy const & imageCopy ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageCopy.srcSubresource );
      VULKAN_HPP_HASH_COMBINE( seed, imageCopy.srcOffset );
      VULKAN_HPP_HASH_COMBINE( seed, imageCopy.dstSubresource );
      VULKAN_HPP_HASH_COMBINE( seed, imageCopy.dstOffset );
      VULKAN_HPP_HASH_COMBINE( seed, imageCopy.extent );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubresourceLayout>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SubresourceLayout const & subresourceLayout ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, subresourceLayout.offset );
      VULKAN_HPP_HASH_COMBINE( seed, subresourceLayout.size );
      VULKAN_HPP_HASH_COMBINE( seed, subresourceLayout.rowPitch );
      VULKAN_HPP_HASH_COMBINE( seed, subresourceLayout.arrayPitch );
      VULKAN_HPP_HASH_COMBINE( seed, subresourceLayout.depthPitch );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageDrmFormatModifierExplicitCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageDrmFormatModifierExplicitCreateInfoEXT const & imageDrmFormatModifierExplicitCreateInfoEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageDrmFormatModifierExplicitCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageDrmFormatModifierExplicitCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageDrmFormatModifierExplicitCreateInfoEXT.drmFormatModifier );
      VULKAN_HPP_HASH_COMBINE( seed, imageDrmFormatModifierExplicitCreateInfoEXT.drmFormatModifierPlaneCount );
      VULKAN_HPP_HASH_COMBINE( seed, imageDrmFormatModifierExplicitCreateInfoEXT.pPlaneLayouts );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageDrmFormatModifierListCreateInfoEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::ImageDrmFormatModifierListCreateInfoEXT const & imageDrmFormatModifierListCreateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageDrmFormatModifierListCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageDrmFormatModifierListCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageDrmFormatModifierListCreateInfoEXT.drmFormatModifierCount );
      VULKAN_HPP_HASH_COMBINE( seed, imageDrmFormatModifierListCreateInfoEXT.pDrmFormatModifiers );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageDrmFormatModifierPropertiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageDrmFormatModifierPropertiesEXT const & imageDrmFormatModifierPropertiesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageDrmFormatModifierPropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageDrmFormatModifierPropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageDrmFormatModifierPropertiesEXT.drmFormatModifier );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageFormatListCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageFormatListCreateInfo const & imageFormatListCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageFormatListCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageFormatListCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageFormatListCreateInfo.viewFormatCount );
      VULKAN_HPP_HASH_COMBINE( seed, imageFormatListCreateInfo.pViewFormats );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageFormatProperties2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageFormatProperties2 const & imageFormatProperties2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageFormatProperties2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageFormatProperties2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageFormatProperties2.imageFormatProperties );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageMemoryBarrier>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageMemoryBarrier const & imageMemoryBarrier ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryBarrier.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryBarrier.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryBarrier.srcAccessMask );
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryBarrier.dstAccessMask );
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryBarrier.oldLayout );
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryBarrier.newLayout );
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryBarrier.srcQueueFamilyIndex );
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryBarrier.dstQueueFamilyIndex );
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryBarrier.image );
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryBarrier.subresourceRange );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageMemoryRequirementsInfo2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageMemoryRequirementsInfo2 const & imageMemoryRequirementsInfo2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryRequirementsInfo2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryRequirementsInfo2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageMemoryRequirementsInfo2.image );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_FUCHSIA )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImagePipeSurfaceCreateInfoFUCHSIA>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImagePipeSurfaceCreateInfoFUCHSIA const & imagePipeSurfaceCreateInfoFUCHSIA ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imagePipeSurfaceCreateInfoFUCHSIA.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imagePipeSurfaceCreateInfoFUCHSIA.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imagePipeSurfaceCreateInfoFUCHSIA.flags );
      VULKAN_HPP_HASH_COMBINE( seed, imagePipeSurfaceCreateInfoFUCHSIA.imagePipeHandle );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImagePlaneMemoryRequirementsInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImagePlaneMemoryRequirementsInfo const & imagePlaneMemoryRequirementsInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imagePlaneMemoryRequirementsInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imagePlaneMemoryRequirementsInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imagePlaneMemoryRequirementsInfo.planeAspect );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageResolve>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageResolve const & imageResolve ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageResolve.srcSubresource );
      VULKAN_HPP_HASH_COMBINE( seed, imageResolve.srcOffset );
      VULKAN_HPP_HASH_COMBINE( seed, imageResolve.dstSubresource );
      VULKAN_HPP_HASH_COMBINE( seed, imageResolve.dstOffset );
      VULKAN_HPP_HASH_COMBINE( seed, imageResolve.extent );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageResolve2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageResolve2 const & imageResolve2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageResolve2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageResolve2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageResolve2.srcSubresource );
      VULKAN_HPP_HASH_COMBINE( seed, imageResolve2.srcOffset );
      VULKAN_HPP_HASH_COMBINE( seed, imageResolve2.dstSubresource );
      VULKAN_HPP_HASH_COMBINE( seed, imageResolve2.dstOffset );
      VULKAN_HPP_HASH_COMBINE( seed, imageResolve2.extent );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageSparseMemoryRequirementsInfo2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageSparseMemoryRequirementsInfo2 const & imageSparseMemoryRequirementsInfo2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageSparseMemoryRequirementsInfo2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageSparseMemoryRequirementsInfo2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageSparseMemoryRequirementsInfo2.image );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageStencilUsageCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageStencilUsageCreateInfo const & imageStencilUsageCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageStencilUsageCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageStencilUsageCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageStencilUsageCreateInfo.stencilUsage );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageSubresource2EXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageSubresource2EXT const & imageSubresource2EXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageSubresource2EXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageSubresource2EXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageSubresource2EXT.imageSubresource );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageSwapchainCreateInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageSwapchainCreateInfoKHR const & imageSwapchainCreateInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageSwapchainCreateInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageSwapchainCreateInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageSwapchainCreateInfoKHR.swapchain );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageViewASTCDecodeModeEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageViewASTCDecodeModeEXT const & imageViewASTCDecodeModeEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageViewASTCDecodeModeEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageViewASTCDecodeModeEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageViewASTCDecodeModeEXT.decodeMode );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageViewAddressPropertiesNVX>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageViewAddressPropertiesNVX const & imageViewAddressPropertiesNVX ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageViewAddressPropertiesNVX.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageViewAddressPropertiesNVX.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageViewAddressPropertiesNVX.deviceAddress );
      VULKAN_HPP_HASH_COMBINE( seed, imageViewAddressPropertiesNVX.size );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageViewCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageViewCreateInfo const & imageViewCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageViewCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageViewCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageViewCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, imageViewCreateInfo.image );
      VULKAN_HPP_HASH_COMBINE( seed, imageViewCreateInfo.viewType );
      VULKAN_HPP_HASH_COMBINE( seed, imageViewCreateInfo.format );
      VULKAN_HPP_HASH_COMBINE( seed, imageViewCreateInfo.components );
      VULKAN_HPP_HASH_COMBINE( seed, imageViewCreateInfo.subresourceRange );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageViewHandleInfoNVX>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageViewHandleInfoNVX const & imageViewHandleInfoNVX ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageViewHandleInfoNVX.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageViewHandleInfoNVX.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageViewHandleInfoNVX.imageView );
      VULKAN_HPP_HASH_COMBINE( seed, imageViewHandleInfoNVX.descriptorType );
      VULKAN_HPP_HASH_COMBINE( seed, imageViewHandleInfoNVX.sampler );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageViewMinLodCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageViewMinLodCreateInfoEXT const & imageViewMinLodCreateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageViewMinLodCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageViewMinLodCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageViewMinLodCreateInfoEXT.minLod );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageViewSampleWeightCreateInfoQCOM>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageViewSampleWeightCreateInfoQCOM const & imageViewSampleWeightCreateInfoQCOM ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageViewSampleWeightCreateInfoQCOM.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageViewSampleWeightCreateInfoQCOM.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageViewSampleWeightCreateInfoQCOM.filterCenter );
      VULKAN_HPP_HASH_COMBINE( seed, imageViewSampleWeightCreateInfoQCOM.filterSize );
      VULKAN_HPP_HASH_COMBINE( seed, imageViewSampleWeightCreateInfoQCOM.numPhases );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImageViewUsageCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImageViewUsageCreateInfo const & imageViewUsageCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, imageViewUsageCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, imageViewUsageCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, imageViewUsageCreateInfo.usage );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_ANDROID_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportAndroidHardwareBufferInfoANDROID>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::ImportAndroidHardwareBufferInfoANDROID const & importAndroidHardwareBufferInfoANDROID ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, importAndroidHardwareBufferInfoANDROID.sType );
      VULKAN_HPP_HASH_COMBINE( seed, importAndroidHardwareBufferInfoANDROID.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, importAndroidHardwareBufferInfoANDROID.buffer );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_ANDROID_KHR*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportFenceFdInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImportFenceFdInfoKHR const & importFenceFdInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, importFenceFdInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, importFenceFdInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, importFenceFdInfoKHR.fence );
      VULKAN_HPP_HASH_COMBINE( seed, importFenceFdInfoKHR.flags );
      VULKAN_HPP_HASH_COMBINE( seed, importFenceFdInfoKHR.handleType );
      VULKAN_HPP_HASH_COMBINE( seed, importFenceFdInfoKHR.fd );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportFenceWin32HandleInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImportFenceWin32HandleInfoKHR const & importFenceWin32HandleInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, importFenceWin32HandleInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, importFenceWin32HandleInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, importFenceWin32HandleInfoKHR.fence );
      VULKAN_HPP_HASH_COMBINE( seed, importFenceWin32HandleInfoKHR.flags );
      VULKAN_HPP_HASH_COMBINE( seed, importFenceWin32HandleInfoKHR.handleType );
      VULKAN_HPP_HASH_COMBINE( seed, importFenceWin32HandleInfoKHR.handle );
      VULKAN_HPP_HASH_COMBINE( seed, importFenceWin32HandleInfoKHR.name );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportMemoryBufferCollectionFUCHSIA>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImportMemoryBufferCollectionFUCHSIA const & importMemoryBufferCollectionFUCHSIA ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryBufferCollectionFUCHSIA.sType );
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryBufferCollectionFUCHSIA.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryBufferCollectionFUCHSIA.collection );
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryBufferCollectionFUCHSIA.index );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportMemoryFdInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImportMemoryFdInfoKHR const & importMemoryFdInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryFdInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryFdInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryFdInfoKHR.handleType );
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryFdInfoKHR.fd );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportMemoryHostPointerInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImportMemoryHostPointerInfoEXT const & importMemoryHostPointerInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryHostPointerInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryHostPointerInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryHostPointerInfoEXT.handleType );
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryHostPointerInfoEXT.pHostPointer );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportMemoryWin32HandleInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImportMemoryWin32HandleInfoKHR const & importMemoryWin32HandleInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryWin32HandleInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryWin32HandleInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryWin32HandleInfoKHR.handleType );
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryWin32HandleInfoKHR.handle );
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryWin32HandleInfoKHR.name );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportMemoryWin32HandleInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImportMemoryWin32HandleInfoNV const & importMemoryWin32HandleInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryWin32HandleInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryWin32HandleInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryWin32HandleInfoNV.handleType );
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryWin32HandleInfoNV.handle );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportMemoryZirconHandleInfoFUCHSIA>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImportMemoryZirconHandleInfoFUCHSIA const & importMemoryZirconHandleInfoFUCHSIA ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryZirconHandleInfoFUCHSIA.sType );
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryZirconHandleInfoFUCHSIA.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryZirconHandleInfoFUCHSIA.handleType );
      VULKAN_HPP_HASH_COMBINE( seed, importMemoryZirconHandleInfoFUCHSIA.handle );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

#  if defined( VK_USE_PLATFORM_METAL_EXT )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportMetalBufferInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImportMetalBufferInfoEXT const & importMetalBufferInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, importMetalBufferInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, importMetalBufferInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, importMetalBufferInfoEXT.mtlBuffer );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_METAL_EXT*/

#  if defined( VK_USE_PLATFORM_METAL_EXT )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportMetalIOSurfaceInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImportMetalIOSurfaceInfoEXT const & importMetalIOSurfaceInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, importMetalIOSurfaceInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, importMetalIOSurfaceInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, importMetalIOSurfaceInfoEXT.ioSurface );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_METAL_EXT*/

#  if defined( VK_USE_PLATFORM_METAL_EXT )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportMetalSharedEventInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImportMetalSharedEventInfoEXT const & importMetalSharedEventInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, importMetalSharedEventInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, importMetalSharedEventInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, importMetalSharedEventInfoEXT.mtlSharedEvent );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_METAL_EXT*/

#  if defined( VK_USE_PLATFORM_METAL_EXT )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportMetalTextureInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImportMetalTextureInfoEXT const & importMetalTextureInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, importMetalTextureInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, importMetalTextureInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, importMetalTextureInfoEXT.plane );
      VULKAN_HPP_HASH_COMBINE( seed, importMetalTextureInfoEXT.mtlTexture );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_METAL_EXT*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportSemaphoreFdInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImportSemaphoreFdInfoKHR const & importSemaphoreFdInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, importSemaphoreFdInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, importSemaphoreFdInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, importSemaphoreFdInfoKHR.semaphore );
      VULKAN_HPP_HASH_COMBINE( seed, importSemaphoreFdInfoKHR.flags );
      VULKAN_HPP_HASH_COMBINE( seed, importSemaphoreFdInfoKHR.handleType );
      VULKAN_HPP_HASH_COMBINE( seed, importSemaphoreFdInfoKHR.fd );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportSemaphoreWin32HandleInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ImportSemaphoreWin32HandleInfoKHR const & importSemaphoreWin32HandleInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, importSemaphoreWin32HandleInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, importSemaphoreWin32HandleInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, importSemaphoreWin32HandleInfoKHR.semaphore );
      VULKAN_HPP_HASH_COMBINE( seed, importSemaphoreWin32HandleInfoKHR.flags );
      VULKAN_HPP_HASH_COMBINE( seed, importSemaphoreWin32HandleInfoKHR.handleType );
      VULKAN_HPP_HASH_COMBINE( seed, importSemaphoreWin32HandleInfoKHR.handle );
      VULKAN_HPP_HASH_COMBINE( seed, importSemaphoreWin32HandleInfoKHR.name );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ImportSemaphoreZirconHandleInfoFUCHSIA>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::ImportSemaphoreZirconHandleInfoFUCHSIA const & importSemaphoreZirconHandleInfoFUCHSIA ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, importSemaphoreZirconHandleInfoFUCHSIA.sType );
      VULKAN_HPP_HASH_COMBINE( seed, importSemaphoreZirconHandleInfoFUCHSIA.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, importSemaphoreZirconHandleInfoFUCHSIA.semaphore );
      VULKAN_HPP_HASH_COMBINE( seed, importSemaphoreZirconHandleInfoFUCHSIA.flags );
      VULKAN_HPP_HASH_COMBINE( seed, importSemaphoreZirconHandleInfoFUCHSIA.handleType );
      VULKAN_HPP_HASH_COMBINE( seed, importSemaphoreZirconHandleInfoFUCHSIA.zirconHandle );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutTokenNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutTokenNV const & indirectCommandsLayoutTokenNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsLayoutTokenNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsLayoutTokenNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsLayoutTokenNV.tokenType );
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsLayoutTokenNV.stream );
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsLayoutTokenNV.offset );
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsLayoutTokenNV.vertexBindingUnit );
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsLayoutTokenNV.vertexDynamicStride );
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsLayoutTokenNV.pushconstantPipelineLayout );
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsLayoutTokenNV.pushconstantShaderStageFlags );
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsLayoutTokenNV.pushconstantOffset );
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsLayoutTokenNV.pushconstantSize );
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsLayoutTokenNV.indirectStateFlags );
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsLayoutTokenNV.indexTypeCount );
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsLayoutTokenNV.pIndexTypes );
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsLayoutTokenNV.pIndexTypeValues );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutCreateInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutCreateInfoNV const & indirectCommandsLayoutCreateInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsLayoutCreateInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsLayoutCreateInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsLayoutCreateInfoNV.flags );
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsLayoutCreateInfoNV.pipelineBindPoint );
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsLayoutCreateInfoNV.tokenCount );
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsLayoutCreateInfoNV.pTokens );
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsLayoutCreateInfoNV.streamCount );
      VULKAN_HPP_HASH_COMBINE( seed, indirectCommandsLayoutCreateInfoNV.pStreamStrides );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::InitializePerformanceApiInfoINTEL>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::InitializePerformanceApiInfoINTEL const & initializePerformanceApiInfoINTEL ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, initializePerformanceApiInfoINTEL.sType );
      VULKAN_HPP_HASH_COMBINE( seed, initializePerformanceApiInfoINTEL.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, initializePerformanceApiInfoINTEL.pUserData );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::InputAttachmentAspectReference>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::InputAttachmentAspectReference const & inputAttachmentAspectReference ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, inputAttachmentAspectReference.subpass );
      VULKAN_HPP_HASH_COMBINE( seed, inputAttachmentAspectReference.inputAttachmentIndex );
      VULKAN_HPP_HASH_COMBINE( seed, inputAttachmentAspectReference.aspectMask );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::InstanceCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::InstanceCreateInfo const & instanceCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, instanceCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, instanceCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, instanceCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, instanceCreateInfo.pApplicationInfo );
      VULKAN_HPP_HASH_COMBINE( seed, instanceCreateInfo.enabledLayerCount );
      for ( size_t i = 0; i < instanceCreateInfo.enabledLayerCount; ++i )
      {
        for ( const char * p = instanceCreateInfo.ppEnabledLayerNames[i]; *p != '\0'; ++p )
        {
          VULKAN_HPP_HASH_COMBINE( seed, *p );
        }
      }
      VULKAN_HPP_HASH_COMBINE( seed, instanceCreateInfo.enabledExtensionCount );
      for ( size_t i = 0; i < instanceCreateInfo.enabledExtensionCount; ++i )
      {
        for ( const char * p = instanceCreateInfo.ppEnabledExtensionNames[i]; *p != '\0'; ++p )
        {
          VULKAN_HPP_HASH_COMBINE( seed, *p );
        }
      }
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::LayerProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::LayerProperties const & layerProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      for ( size_t i = 0; i < VK_MAX_EXTENSION_NAME_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, layerProperties.layerName[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, layerProperties.specVersion );
      VULKAN_HPP_HASH_COMBINE( seed, layerProperties.implementationVersion );
      for ( size_t i = 0; i < VK_MAX_DESCRIPTION_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, layerProperties.description[i] );
      }
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_MACOS_MVK )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MacOSSurfaceCreateInfoMVK>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MacOSSurfaceCreateInfoMVK const & macOSSurfaceCreateInfoMVK ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, macOSSurfaceCreateInfoMVK.sType );
      VULKAN_HPP_HASH_COMBINE( seed, macOSSurfaceCreateInfoMVK.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, macOSSurfaceCreateInfoMVK.flags );
      VULKAN_HPP_HASH_COMBINE( seed, macOSSurfaceCreateInfoMVK.pView );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_MACOS_MVK*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MappedMemoryRange>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MappedMemoryRange const & mappedMemoryRange ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, mappedMemoryRange.sType );
      VULKAN_HPP_HASH_COMBINE( seed, mappedMemoryRange.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, mappedMemoryRange.memory );
      VULKAN_HPP_HASH_COMBINE( seed, mappedMemoryRange.offset );
      VULKAN_HPP_HASH_COMBINE( seed, mappedMemoryRange.size );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryAllocateFlagsInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MemoryAllocateFlagsInfo const & memoryAllocateFlagsInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, memoryAllocateFlagsInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, memoryAllocateFlagsInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, memoryAllocateFlagsInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, memoryAllocateFlagsInfo.deviceMask );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryAllocateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MemoryAllocateInfo const & memoryAllocateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, memoryAllocateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, memoryAllocateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, memoryAllocateInfo.allocationSize );
      VULKAN_HPP_HASH_COMBINE( seed, memoryAllocateInfo.memoryTypeIndex );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryBarrier>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MemoryBarrier const & memoryBarrier ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, memoryBarrier.sType );
      VULKAN_HPP_HASH_COMBINE( seed, memoryBarrier.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, memoryBarrier.srcAccessMask );
      VULKAN_HPP_HASH_COMBINE( seed, memoryBarrier.dstAccessMask );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryDedicatedAllocateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MemoryDedicatedAllocateInfo const & memoryDedicatedAllocateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, memoryDedicatedAllocateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, memoryDedicatedAllocateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, memoryDedicatedAllocateInfo.image );
      VULKAN_HPP_HASH_COMBINE( seed, memoryDedicatedAllocateInfo.buffer );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryDedicatedRequirements>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MemoryDedicatedRequirements const & memoryDedicatedRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, memoryDedicatedRequirements.sType );
      VULKAN_HPP_HASH_COMBINE( seed, memoryDedicatedRequirements.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, memoryDedicatedRequirements.prefersDedicatedAllocation );
      VULKAN_HPP_HASH_COMBINE( seed, memoryDedicatedRequirements.requiresDedicatedAllocation );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryFdPropertiesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MemoryFdPropertiesKHR const & memoryFdPropertiesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, memoryFdPropertiesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, memoryFdPropertiesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, memoryFdPropertiesKHR.memoryTypeBits );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_ANDROID_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryGetAndroidHardwareBufferInfoANDROID>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::MemoryGetAndroidHardwareBufferInfoANDROID const & memoryGetAndroidHardwareBufferInfoANDROID ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, memoryGetAndroidHardwareBufferInfoANDROID.sType );
      VULKAN_HPP_HASH_COMBINE( seed, memoryGetAndroidHardwareBufferInfoANDROID.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, memoryGetAndroidHardwareBufferInfoANDROID.memory );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_ANDROID_KHR*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryGetFdInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MemoryGetFdInfoKHR const & memoryGetFdInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, memoryGetFdInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, memoryGetFdInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, memoryGetFdInfoKHR.memory );
      VULKAN_HPP_HASH_COMBINE( seed, memoryGetFdInfoKHR.handleType );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryGetRemoteAddressInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MemoryGetRemoteAddressInfoNV const & memoryGetRemoteAddressInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, memoryGetRemoteAddressInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, memoryGetRemoteAddressInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, memoryGetRemoteAddressInfoNV.memory );
      VULKAN_HPP_HASH_COMBINE( seed, memoryGetRemoteAddressInfoNV.handleType );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryGetWin32HandleInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MemoryGetWin32HandleInfoKHR const & memoryGetWin32HandleInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, memoryGetWin32HandleInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, memoryGetWin32HandleInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, memoryGetWin32HandleInfoKHR.memory );
      VULKAN_HPP_HASH_COMBINE( seed, memoryGetWin32HandleInfoKHR.handleType );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryGetZirconHandleInfoFUCHSIA>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MemoryGetZirconHandleInfoFUCHSIA const & memoryGetZirconHandleInfoFUCHSIA ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, memoryGetZirconHandleInfoFUCHSIA.sType );
      VULKAN_HPP_HASH_COMBINE( seed, memoryGetZirconHandleInfoFUCHSIA.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, memoryGetZirconHandleInfoFUCHSIA.memory );
      VULKAN_HPP_HASH_COMBINE( seed, memoryGetZirconHandleInfoFUCHSIA.handleType );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryHeap>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MemoryHeap const & memoryHeap ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, memoryHeap.size );
      VULKAN_HPP_HASH_COMBINE( seed, memoryHeap.flags );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryHostPointerPropertiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MemoryHostPointerPropertiesEXT const & memoryHostPointerPropertiesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, memoryHostPointerPropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, memoryHostPointerPropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, memoryHostPointerPropertiesEXT.memoryTypeBits );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryOpaqueCaptureAddressAllocateInfo>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::MemoryOpaqueCaptureAddressAllocateInfo const & memoryOpaqueCaptureAddressAllocateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, memoryOpaqueCaptureAddressAllocateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, memoryOpaqueCaptureAddressAllocateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, memoryOpaqueCaptureAddressAllocateInfo.opaqueCaptureAddress );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryPriorityAllocateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MemoryPriorityAllocateInfoEXT const & memoryPriorityAllocateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, memoryPriorityAllocateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, memoryPriorityAllocateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, memoryPriorityAllocateInfoEXT.priority );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryRequirements>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MemoryRequirements const & memoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, memoryRequirements.size );
      VULKAN_HPP_HASH_COMBINE( seed, memoryRequirements.alignment );
      VULKAN_HPP_HASH_COMBINE( seed, memoryRequirements.memoryTypeBits );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryRequirements2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MemoryRequirements2 const & memoryRequirements2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, memoryRequirements2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, memoryRequirements2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, memoryRequirements2.memoryRequirements );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryType>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MemoryType const & memoryType ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, memoryType.propertyFlags );
      VULKAN_HPP_HASH_COMBINE( seed, memoryType.heapIndex );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryWin32HandlePropertiesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MemoryWin32HandlePropertiesKHR const & memoryWin32HandlePropertiesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, memoryWin32HandlePropertiesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, memoryWin32HandlePropertiesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, memoryWin32HandlePropertiesKHR.memoryTypeBits );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MemoryZirconHandlePropertiesFUCHSIA>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MemoryZirconHandlePropertiesFUCHSIA const & memoryZirconHandlePropertiesFUCHSIA ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, memoryZirconHandlePropertiesFUCHSIA.sType );
      VULKAN_HPP_HASH_COMBINE( seed, memoryZirconHandlePropertiesFUCHSIA.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, memoryZirconHandlePropertiesFUCHSIA.memoryTypeBits );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

#  if defined( VK_USE_PLATFORM_METAL_EXT )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MetalSurfaceCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MetalSurfaceCreateInfoEXT const & metalSurfaceCreateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, metalSurfaceCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, metalSurfaceCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, metalSurfaceCreateInfoEXT.flags );
      VULKAN_HPP_HASH_COMBINE( seed, metalSurfaceCreateInfoEXT.pLayer );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_METAL_EXT*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MicromapBuildSizesInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MicromapBuildSizesInfoEXT const & micromapBuildSizesInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, micromapBuildSizesInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, micromapBuildSizesInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, micromapBuildSizesInfoEXT.micromapSize );
      VULKAN_HPP_HASH_COMBINE( seed, micromapBuildSizesInfoEXT.buildScratchSize );
      VULKAN_HPP_HASH_COMBINE( seed, micromapBuildSizesInfoEXT.discardable );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MicromapCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MicromapCreateInfoEXT const & micromapCreateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, micromapCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, micromapCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, micromapCreateInfoEXT.createFlags );
      VULKAN_HPP_HASH_COMBINE( seed, micromapCreateInfoEXT.buffer );
      VULKAN_HPP_HASH_COMBINE( seed, micromapCreateInfoEXT.offset );
      VULKAN_HPP_HASH_COMBINE( seed, micromapCreateInfoEXT.size );
      VULKAN_HPP_HASH_COMBINE( seed, micromapCreateInfoEXT.type );
      VULKAN_HPP_HASH_COMBINE( seed, micromapCreateInfoEXT.deviceAddress );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MicromapTriangleEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MicromapTriangleEXT const & micromapTriangleEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, micromapTriangleEXT.dataOffset );
      VULKAN_HPP_HASH_COMBINE( seed, micromapTriangleEXT.subdivisionLevel );
      VULKAN_HPP_HASH_COMBINE( seed, micromapTriangleEXT.format );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MicromapVersionInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MicromapVersionInfoEXT const & micromapVersionInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, micromapVersionInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, micromapVersionInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, micromapVersionInfoEXT.pVersionData );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MultiDrawIndexedInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MultiDrawIndexedInfoEXT const & multiDrawIndexedInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, multiDrawIndexedInfoEXT.firstIndex );
      VULKAN_HPP_HASH_COMBINE( seed, multiDrawIndexedInfoEXT.indexCount );
      VULKAN_HPP_HASH_COMBINE( seed, multiDrawIndexedInfoEXT.vertexOffset );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MultiDrawInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MultiDrawInfoEXT const & multiDrawInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, multiDrawInfoEXT.firstVertex );
      VULKAN_HPP_HASH_COMBINE( seed, multiDrawInfoEXT.vertexCount );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MultisamplePropertiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MultisamplePropertiesEXT const & multisamplePropertiesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, multisamplePropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, multisamplePropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, multisamplePropertiesEXT.maxSampleLocationGridSize );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MultisampledRenderToSingleSampledInfoEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::MultisampledRenderToSingleSampledInfoEXT const & multisampledRenderToSingleSampledInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, multisampledRenderToSingleSampledInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, multisampledRenderToSingleSampledInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, multisampledRenderToSingleSampledInfoEXT.multisampledRenderToSingleSampledEnable );
      VULKAN_HPP_HASH_COMBINE( seed, multisampledRenderToSingleSampledInfoEXT.rasterizationSamples );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MultiviewPerViewAttributesInfoNVX>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MultiviewPerViewAttributesInfoNVX const & multiviewPerViewAttributesInfoNVX ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, multiviewPerViewAttributesInfoNVX.sType );
      VULKAN_HPP_HASH_COMBINE( seed, multiviewPerViewAttributesInfoNVX.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, multiviewPerViewAttributesInfoNVX.perViewAttributes );
      VULKAN_HPP_HASH_COMBINE( seed, multiviewPerViewAttributesInfoNVX.perViewAttributesPositionXOnly );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MutableDescriptorTypeListEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MutableDescriptorTypeListEXT const & mutableDescriptorTypeListEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, mutableDescriptorTypeListEXT.descriptorTypeCount );
      VULKAN_HPP_HASH_COMBINE( seed, mutableDescriptorTypeListEXT.pDescriptorTypes );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::MutableDescriptorTypeCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::MutableDescriptorTypeCreateInfoEXT const & mutableDescriptorTypeCreateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, mutableDescriptorTypeCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, mutableDescriptorTypeCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, mutableDescriptorTypeCreateInfoEXT.mutableDescriptorTypeListCount );
      VULKAN_HPP_HASH_COMBINE( seed, mutableDescriptorTypeCreateInfoEXT.pMutableDescriptorTypeLists );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::OpticalFlowExecuteInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::OpticalFlowExecuteInfoNV const & opticalFlowExecuteInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowExecuteInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowExecuteInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowExecuteInfoNV.flags );
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowExecuteInfoNV.regionCount );
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowExecuteInfoNV.pRegions );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::OpticalFlowImageFormatInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::OpticalFlowImageFormatInfoNV const & opticalFlowImageFormatInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowImageFormatInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowImageFormatInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowImageFormatInfoNV.usage );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::OpticalFlowImageFormatPropertiesNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::OpticalFlowImageFormatPropertiesNV const & opticalFlowImageFormatPropertiesNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowImageFormatPropertiesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowImageFormatPropertiesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowImageFormatPropertiesNV.format );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::OpticalFlowSessionCreateInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::OpticalFlowSessionCreateInfoNV const & opticalFlowSessionCreateInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowSessionCreateInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowSessionCreateInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowSessionCreateInfoNV.width );
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowSessionCreateInfoNV.height );
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowSessionCreateInfoNV.imageFormat );
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowSessionCreateInfoNV.flowVectorFormat );
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowSessionCreateInfoNV.costFormat );
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowSessionCreateInfoNV.outputGridSize );
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowSessionCreateInfoNV.hintGridSize );
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowSessionCreateInfoNV.performanceLevel );
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowSessionCreateInfoNV.flags );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::OpticalFlowSessionCreatePrivateDataInfoNV>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::OpticalFlowSessionCreatePrivateDataInfoNV const & opticalFlowSessionCreatePrivateDataInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowSessionCreatePrivateDataInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowSessionCreatePrivateDataInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowSessionCreatePrivateDataInfoNV.id );
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowSessionCreatePrivateDataInfoNV.size );
      VULKAN_HPP_HASH_COMBINE( seed, opticalFlowSessionCreatePrivateDataInfoNV.pPrivateData );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PastPresentationTimingGOOGLE>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PastPresentationTimingGOOGLE const & pastPresentationTimingGOOGLE ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pastPresentationTimingGOOGLE.presentID );
      VULKAN_HPP_HASH_COMBINE( seed, pastPresentationTimingGOOGLE.desiredPresentTime );
      VULKAN_HPP_HASH_COMBINE( seed, pastPresentationTimingGOOGLE.actualPresentTime );
      VULKAN_HPP_HASH_COMBINE( seed, pastPresentationTimingGOOGLE.earliestPresentTime );
      VULKAN_HPP_HASH_COMBINE( seed, pastPresentationTimingGOOGLE.presentMargin );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PerformanceConfigurationAcquireInfoINTEL>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PerformanceConfigurationAcquireInfoINTEL const & performanceConfigurationAcquireInfoINTEL ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, performanceConfigurationAcquireInfoINTEL.sType );
      VULKAN_HPP_HASH_COMBINE( seed, performanceConfigurationAcquireInfoINTEL.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, performanceConfigurationAcquireInfoINTEL.type );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PerformanceCounterDescriptionKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PerformanceCounterDescriptionKHR const & performanceCounterDescriptionKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, performanceCounterDescriptionKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, performanceCounterDescriptionKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, performanceCounterDescriptionKHR.flags );
      for ( size_t i = 0; i < VK_MAX_DESCRIPTION_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, performanceCounterDescriptionKHR.name[i] );
      }
      for ( size_t i = 0; i < VK_MAX_DESCRIPTION_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, performanceCounterDescriptionKHR.category[i] );
      }
      for ( size_t i = 0; i < VK_MAX_DESCRIPTION_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, performanceCounterDescriptionKHR.description[i] );
      }
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PerformanceCounterKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PerformanceCounterKHR const & performanceCounterKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, performanceCounterKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, performanceCounterKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, performanceCounterKHR.unit );
      VULKAN_HPP_HASH_COMBINE( seed, performanceCounterKHR.scope );
      VULKAN_HPP_HASH_COMBINE( seed, performanceCounterKHR.storage );
      for ( size_t i = 0; i < VK_UUID_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, performanceCounterKHR.uuid[i] );
      }
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PerformanceMarkerInfoINTEL>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PerformanceMarkerInfoINTEL const & performanceMarkerInfoINTEL ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, performanceMarkerInfoINTEL.sType );
      VULKAN_HPP_HASH_COMBINE( seed, performanceMarkerInfoINTEL.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, performanceMarkerInfoINTEL.marker );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PerformanceOverrideInfoINTEL>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PerformanceOverrideInfoINTEL const & performanceOverrideInfoINTEL ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, performanceOverrideInfoINTEL.sType );
      VULKAN_HPP_HASH_COMBINE( seed, performanceOverrideInfoINTEL.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, performanceOverrideInfoINTEL.type );
      VULKAN_HPP_HASH_COMBINE( seed, performanceOverrideInfoINTEL.enable );
      VULKAN_HPP_HASH_COMBINE( seed, performanceOverrideInfoINTEL.parameter );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PerformanceQuerySubmitInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PerformanceQuerySubmitInfoKHR const & performanceQuerySubmitInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, performanceQuerySubmitInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, performanceQuerySubmitInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, performanceQuerySubmitInfoKHR.counterPassIndex );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PerformanceStreamMarkerInfoINTEL>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PerformanceStreamMarkerInfoINTEL const & performanceStreamMarkerInfoINTEL ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, performanceStreamMarkerInfoINTEL.sType );
      VULKAN_HPP_HASH_COMBINE( seed, performanceStreamMarkerInfoINTEL.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, performanceStreamMarkerInfoINTEL.marker );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevice16BitStorageFeatures>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDevice16BitStorageFeatures const & physicalDevice16BitStorageFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevice16BitStorageFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevice16BitStorageFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevice16BitStorageFeatures.storageBuffer16BitAccess );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevice16BitStorageFeatures.uniformAndStorageBuffer16BitAccess );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevice16BitStorageFeatures.storagePushConstant16 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevice16BitStorageFeatures.storageInputOutput16 );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevice4444FormatsFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDevice4444FormatsFeaturesEXT const & physicalDevice4444FormatsFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevice4444FormatsFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevice4444FormatsFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevice4444FormatsFeaturesEXT.formatA4R4G4B4 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevice4444FormatsFeaturesEXT.formatA4B4G4R4 );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevice8BitStorageFeatures>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDevice8BitStorageFeatures const & physicalDevice8BitStorageFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevice8BitStorageFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevice8BitStorageFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevice8BitStorageFeatures.storageBuffer8BitAccess );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevice8BitStorageFeatures.uniformAndStorageBuffer8BitAccess );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevice8BitStorageFeatures.storagePushConstant8 );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceASTCDecodeFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceASTCDecodeFeaturesEXT const & physicalDeviceASTCDecodeFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceASTCDecodeFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceASTCDecodeFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceASTCDecodeFeaturesEXT.decodeModeSharedExponent );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceAccelerationStructureFeaturesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceAccelerationStructureFeaturesKHR const & physicalDeviceAccelerationStructureFeaturesKHR ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAccelerationStructureFeaturesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAccelerationStructureFeaturesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAccelerationStructureFeaturesKHR.accelerationStructure );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAccelerationStructureFeaturesKHR.accelerationStructureCaptureReplay );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAccelerationStructureFeaturesKHR.accelerationStructureIndirectBuild );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAccelerationStructureFeaturesKHR.accelerationStructureHostCommands );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAccelerationStructureFeaturesKHR.descriptorBindingAccelerationStructureUpdateAfterBind );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceAccelerationStructurePropertiesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceAccelerationStructurePropertiesKHR const & physicalDeviceAccelerationStructurePropertiesKHR )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAccelerationStructurePropertiesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAccelerationStructurePropertiesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAccelerationStructurePropertiesKHR.maxGeometryCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAccelerationStructurePropertiesKHR.maxInstanceCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAccelerationStructurePropertiesKHR.maxPrimitiveCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAccelerationStructurePropertiesKHR.maxPerStageDescriptorAccelerationStructures );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAccelerationStructurePropertiesKHR.maxPerStageDescriptorUpdateAfterBindAccelerationStructures );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAccelerationStructurePropertiesKHR.maxDescriptorSetAccelerationStructures );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAccelerationStructurePropertiesKHR.maxDescriptorSetUpdateAfterBindAccelerationStructures );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAccelerationStructurePropertiesKHR.minAccelerationStructureScratchOffsetAlignment );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceAddressBindingReportFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceAddressBindingReportFeaturesEXT const & physicalDeviceAddressBindingReportFeaturesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAddressBindingReportFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAddressBindingReportFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAddressBindingReportFeaturesEXT.reportAddressBinding );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceAmigoProfilingFeaturesSEC>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceAmigoProfilingFeaturesSEC const & physicalDeviceAmigoProfilingFeaturesSEC ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAmigoProfilingFeaturesSEC.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAmigoProfilingFeaturesSEC.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAmigoProfilingFeaturesSEC.amigoProfiling );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT const &
                              physicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT.attachmentFeedbackLoopLayout );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceBlendOperationAdvancedFeaturesEXT>
  {
    std::size_t operator()(
      VULKAN_HPP_NAMESPACE::PhysicalDeviceBlendOperationAdvancedFeaturesEXT const & physicalDeviceBlendOperationAdvancedFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBlendOperationAdvancedFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBlendOperationAdvancedFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBlendOperationAdvancedFeaturesEXT.advancedBlendCoherentOperations );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceBlendOperationAdvancedPropertiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceBlendOperationAdvancedPropertiesEXT const & physicalDeviceBlendOperationAdvancedPropertiesEXT )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBlendOperationAdvancedPropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBlendOperationAdvancedPropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBlendOperationAdvancedPropertiesEXT.advancedBlendMaxColorAttachments );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBlendOperationAdvancedPropertiesEXT.advancedBlendIndependentBlend );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBlendOperationAdvancedPropertiesEXT.advancedBlendNonPremultipliedSrcColor );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBlendOperationAdvancedPropertiesEXT.advancedBlendNonPremultipliedDstColor );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBlendOperationAdvancedPropertiesEXT.advancedBlendCorrelatedOverlap );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBlendOperationAdvancedPropertiesEXT.advancedBlendAllOperations );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceBorderColorSwizzleFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceBorderColorSwizzleFeaturesEXT const & physicalDeviceBorderColorSwizzleFeaturesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBorderColorSwizzleFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBorderColorSwizzleFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBorderColorSwizzleFeaturesEXT.borderColorSwizzle );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBorderColorSwizzleFeaturesEXT.borderColorSwizzleFromImage );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceBufferDeviceAddressFeatures>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceBufferDeviceAddressFeatures const & physicalDeviceBufferDeviceAddressFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBufferDeviceAddressFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBufferDeviceAddressFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBufferDeviceAddressFeatures.bufferDeviceAddress );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBufferDeviceAddressFeatures.bufferDeviceAddressCaptureReplay );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBufferDeviceAddressFeatures.bufferDeviceAddressMultiDevice );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceBufferDeviceAddressFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceBufferDeviceAddressFeaturesEXT const & physicalDeviceBufferDeviceAddressFeaturesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBufferDeviceAddressFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBufferDeviceAddressFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBufferDeviceAddressFeaturesEXT.bufferDeviceAddress );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBufferDeviceAddressFeaturesEXT.bufferDeviceAddressCaptureReplay );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceBufferDeviceAddressFeaturesEXT.bufferDeviceAddressMultiDevice );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCoherentMemoryFeaturesAMD>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceCoherentMemoryFeaturesAMD const & physicalDeviceCoherentMemoryFeaturesAMD ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceCoherentMemoryFeaturesAMD.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceCoherentMemoryFeaturesAMD.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceCoherentMemoryFeaturesAMD.deviceCoherentMemory );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceColorWriteEnableFeaturesEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceColorWriteEnableFeaturesEXT const & physicalDeviceColorWriteEnableFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceColorWriteEnableFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceColorWriteEnableFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceColorWriteEnableFeaturesEXT.colorWriteEnable );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceComputeShaderDerivativesFeaturesNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceComputeShaderDerivativesFeaturesNV const & physicalDeviceComputeShaderDerivativesFeaturesNV )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceComputeShaderDerivativesFeaturesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceComputeShaderDerivativesFeaturesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceComputeShaderDerivativesFeaturesNV.computeDerivativeGroupQuads );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceComputeShaderDerivativesFeaturesNV.computeDerivativeGroupLinear );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceConditionalRenderingFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceConditionalRenderingFeaturesEXT const & physicalDeviceConditionalRenderingFeaturesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceConditionalRenderingFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceConditionalRenderingFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceConditionalRenderingFeaturesEXT.conditionalRendering );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceConditionalRenderingFeaturesEXT.inheritedConditionalRendering );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceConservativeRasterizationPropertiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceConservativeRasterizationPropertiesEXT const &
                              physicalDeviceConservativeRasterizationPropertiesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceConservativeRasterizationPropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceConservativeRasterizationPropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceConservativeRasterizationPropertiesEXT.primitiveOverestimationSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceConservativeRasterizationPropertiesEXT.maxExtraPrimitiveOverestimationSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceConservativeRasterizationPropertiesEXT.extraPrimitiveOverestimationSizeGranularity );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceConservativeRasterizationPropertiesEXT.primitiveUnderestimation );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceConservativeRasterizationPropertiesEXT.conservativePointAndLineRasterization );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceConservativeRasterizationPropertiesEXT.degenerateTrianglesRasterized );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceConservativeRasterizationPropertiesEXT.degenerateLinesRasterized );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceConservativeRasterizationPropertiesEXT.fullyCoveredFragmentShaderInputVariable );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceConservativeRasterizationPropertiesEXT.conservativeRasterizationPostDepthCoverage );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixFeaturesNV>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixFeaturesNV const & physicalDeviceCooperativeMatrixFeaturesNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceCooperativeMatrixFeaturesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceCooperativeMatrixFeaturesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceCooperativeMatrixFeaturesNV.cooperativeMatrix );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceCooperativeMatrixFeaturesNV.cooperativeMatrixRobustBufferAccess );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixPropertiesNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceCooperativeMatrixPropertiesNV const & physicalDeviceCooperativeMatrixPropertiesNV ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceCooperativeMatrixPropertiesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceCooperativeMatrixPropertiesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceCooperativeMatrixPropertiesNV.cooperativeMatrixSupportedStages );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCornerSampledImageFeaturesNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceCornerSampledImageFeaturesNV const & physicalDeviceCornerSampledImageFeaturesNV ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceCornerSampledImageFeaturesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceCornerSampledImageFeaturesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceCornerSampledImageFeaturesNV.cornerSampledImage );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCoverageReductionModeFeaturesNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceCoverageReductionModeFeaturesNV const & physicalDeviceCoverageReductionModeFeaturesNV ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceCoverageReductionModeFeaturesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceCoverageReductionModeFeaturesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceCoverageReductionModeFeaturesNV.coverageReductionMode );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCustomBorderColorFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceCustomBorderColorFeaturesEXT const & physicalDeviceCustomBorderColorFeaturesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceCustomBorderColorFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceCustomBorderColorFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceCustomBorderColorFeaturesEXT.customBorderColors );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceCustomBorderColorFeaturesEXT.customBorderColorWithoutFormat );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceCustomBorderColorPropertiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceCustomBorderColorPropertiesEXT const & physicalDeviceCustomBorderColorPropertiesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceCustomBorderColorPropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceCustomBorderColorPropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceCustomBorderColorPropertiesEXT.maxCustomBorderColorSamplers );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDedicatedAllocationImageAliasingFeaturesNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceDedicatedAllocationImageAliasingFeaturesNV const &
                              physicalDeviceDedicatedAllocationImageAliasingFeaturesNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDedicatedAllocationImageAliasingFeaturesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDedicatedAllocationImageAliasingFeaturesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDedicatedAllocationImageAliasingFeaturesNV.dedicatedAllocationImageAliasing );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClampZeroOneFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClampZeroOneFeaturesEXT const & physicalDeviceDepthClampZeroOneFeaturesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDepthClampZeroOneFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDepthClampZeroOneFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDepthClampZeroOneFeaturesEXT.depthClampZeroOne );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClipControlFeaturesEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClipControlFeaturesEXT const & physicalDeviceDepthClipControlFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDepthClipControlFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDepthClipControlFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDepthClipControlFeaturesEXT.depthClipControl );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClipEnableFeaturesEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthClipEnableFeaturesEXT const & physicalDeviceDepthClipEnableFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDepthClipEnableFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDepthClipEnableFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDepthClipEnableFeaturesEXT.depthClipEnable );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthStencilResolveProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceDepthStencilResolveProperties const & physicalDeviceDepthStencilResolveProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDepthStencilResolveProperties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDepthStencilResolveProperties.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDepthStencilResolveProperties.supportedDepthResolveModes );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDepthStencilResolveProperties.supportedStencilResolveModes );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDepthStencilResolveProperties.independentResolveNone );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDepthStencilResolveProperties.independentResolve );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorIndexingFeatures>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorIndexingFeatures const & physicalDeviceDescriptorIndexingFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingFeatures.shaderInputAttachmentArrayDynamicIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingFeatures.shaderUniformTexelBufferArrayDynamicIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingFeatures.shaderStorageTexelBufferArrayDynamicIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingFeatures.shaderUniformBufferArrayNonUniformIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingFeatures.shaderSampledImageArrayNonUniformIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingFeatures.shaderStorageBufferArrayNonUniformIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingFeatures.shaderStorageImageArrayNonUniformIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingFeatures.shaderInputAttachmentArrayNonUniformIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingFeatures.shaderUniformTexelBufferArrayNonUniformIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingFeatures.shaderStorageTexelBufferArrayNonUniformIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingFeatures.descriptorBindingUniformBufferUpdateAfterBind );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingFeatures.descriptorBindingSampledImageUpdateAfterBind );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingFeatures.descriptorBindingStorageImageUpdateAfterBind );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingFeatures.descriptorBindingStorageBufferUpdateAfterBind );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingFeatures.descriptorBindingUniformTexelBufferUpdateAfterBind );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingFeatures.descriptorBindingStorageTexelBufferUpdateAfterBind );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingFeatures.descriptorBindingUpdateUnusedWhilePending );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingFeatures.descriptorBindingPartiallyBound );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingFeatures.descriptorBindingVariableDescriptorCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingFeatures.runtimeDescriptorArray );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorIndexingProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorIndexingProperties const & physicalDeviceDescriptorIndexingProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.maxUpdateAfterBindDescriptorsInAllPools );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.shaderUniformBufferArrayNonUniformIndexingNative );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.shaderSampledImageArrayNonUniformIndexingNative );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.shaderStorageBufferArrayNonUniformIndexingNative );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.shaderStorageImageArrayNonUniformIndexingNative );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.shaderInputAttachmentArrayNonUniformIndexingNative );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.robustBufferAccessUpdateAfterBind );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.quadDivergentImplicitLod );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.maxPerStageDescriptorUpdateAfterBindSamplers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.maxPerStageDescriptorUpdateAfterBindUniformBuffers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.maxPerStageDescriptorUpdateAfterBindStorageBuffers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.maxPerStageDescriptorUpdateAfterBindSampledImages );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.maxPerStageDescriptorUpdateAfterBindStorageImages );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.maxPerStageDescriptorUpdateAfterBindInputAttachments );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.maxPerStageUpdateAfterBindResources );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.maxDescriptorSetUpdateAfterBindSamplers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.maxDescriptorSetUpdateAfterBindUniformBuffers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.maxDescriptorSetUpdateAfterBindUniformBuffersDynamic );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.maxDescriptorSetUpdateAfterBindStorageBuffers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.maxDescriptorSetUpdateAfterBindStorageBuffersDynamic );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.maxDescriptorSetUpdateAfterBindSampledImages );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.maxDescriptorSetUpdateAfterBindStorageImages );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorIndexingProperties.maxDescriptorSetUpdateAfterBindInputAttachments );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorSetHostMappingFeaturesVALVE>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceDescriptorSetHostMappingFeaturesVALVE const & physicalDeviceDescriptorSetHostMappingFeaturesVALVE ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorSetHostMappingFeaturesVALVE.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorSetHostMappingFeaturesVALVE.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDescriptorSetHostMappingFeaturesVALVE.descriptorSetHostMapping );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceGeneratedCommandsFeaturesNV>
  {
    std::size_t operator()(
      VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceGeneratedCommandsFeaturesNV const & physicalDeviceDeviceGeneratedCommandsFeaturesNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDeviceGeneratedCommandsFeaturesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDeviceGeneratedCommandsFeaturesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDeviceGeneratedCommandsFeaturesNV.deviceGeneratedCommands );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceGeneratedCommandsPropertiesNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceGeneratedCommandsPropertiesNV const & physicalDeviceDeviceGeneratedCommandsPropertiesNV )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDeviceGeneratedCommandsPropertiesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDeviceGeneratedCommandsPropertiesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDeviceGeneratedCommandsPropertiesNV.maxGraphicsShaderGroupCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDeviceGeneratedCommandsPropertiesNV.maxIndirectSequenceCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDeviceGeneratedCommandsPropertiesNV.maxIndirectCommandsTokenCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDeviceGeneratedCommandsPropertiesNV.maxIndirectCommandsStreamCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDeviceGeneratedCommandsPropertiesNV.maxIndirectCommandsTokenOffset );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDeviceGeneratedCommandsPropertiesNV.maxIndirectCommandsStreamStride );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDeviceGeneratedCommandsPropertiesNV.minSequencesCountBufferOffsetAlignment );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDeviceGeneratedCommandsPropertiesNV.minSequencesIndexBufferOffsetAlignment );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDeviceGeneratedCommandsPropertiesNV.minIndirectCommandsBufferOffsetAlignment );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceMemoryReportFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceDeviceMemoryReportFeaturesEXT const & physicalDeviceDeviceMemoryReportFeaturesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDeviceMemoryReportFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDeviceMemoryReportFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDeviceMemoryReportFeaturesEXT.deviceMemoryReport );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDiagnosticsConfigFeaturesNV>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceDiagnosticsConfigFeaturesNV const & physicalDeviceDiagnosticsConfigFeaturesNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDiagnosticsConfigFeaturesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDiagnosticsConfigFeaturesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDiagnosticsConfigFeaturesNV.diagnosticsConfig );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDiscardRectanglePropertiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceDiscardRectanglePropertiesEXT const & physicalDeviceDiscardRectanglePropertiesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDiscardRectanglePropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDiscardRectanglePropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDiscardRectanglePropertiesEXT.maxDiscardRectangles );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDriverProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceDriverProperties const & physicalDeviceDriverProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDriverProperties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDriverProperties.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDriverProperties.driverID );
      for ( size_t i = 0; i < VK_MAX_DRIVER_NAME_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDriverProperties.driverName[i] );
      }
      for ( size_t i = 0; i < VK_MAX_DRIVER_INFO_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDriverProperties.driverInfo[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDriverProperties.conformanceVersion );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDrmPropertiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceDrmPropertiesEXT const & physicalDeviceDrmPropertiesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDrmPropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDrmPropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDrmPropertiesEXT.hasPrimary );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDrmPropertiesEXT.hasRender );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDrmPropertiesEXT.primaryMajor );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDrmPropertiesEXT.primaryMinor );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDrmPropertiesEXT.renderMajor );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDrmPropertiesEXT.renderMinor );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceDynamicRenderingFeatures>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceDynamicRenderingFeatures const & physicalDeviceDynamicRenderingFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDynamicRenderingFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDynamicRenderingFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceDynamicRenderingFeatures.dynamicRendering );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExclusiveScissorFeaturesNV>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceExclusiveScissorFeaturesNV const & physicalDeviceExclusiveScissorFeaturesNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExclusiveScissorFeaturesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExclusiveScissorFeaturesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExclusiveScissorFeaturesNV.exclusiveScissor );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicState2FeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicState2FeaturesEXT const & physicalDeviceExtendedDynamicState2FeaturesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState2FeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState2FeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState2FeaturesEXT.extendedDynamicState2 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState2FeaturesEXT.extendedDynamicState2LogicOp );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState2FeaturesEXT.extendedDynamicState2PatchControlPoints );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicState3FeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicState3FeaturesEXT const & physicalDeviceExtendedDynamicState3FeaturesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3TessellationDomainOrigin );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3DepthClampEnable );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3PolygonMode );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3RasterizationSamples );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3SampleMask );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3AlphaToCoverageEnable );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3AlphaToOneEnable );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3LogicOpEnable );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3ColorBlendEnable );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3ColorBlendEquation );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3ColorWriteMask );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3RasterizationStream );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3ConservativeRasterizationMode );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3ExtraPrimitiveOverestimationSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3DepthClipEnable );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3SampleLocationsEnable );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3ColorBlendAdvanced );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3ProvokingVertexMode );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3LineRasterizationMode );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3LineStippleEnable );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3DepthClipNegativeOneToOne );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3ViewportWScalingEnable );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3ViewportSwizzle );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3CoverageToColorEnable );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3CoverageToColorLocation );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3CoverageModulationMode );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3CoverageModulationTableEnable );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3CoverageModulationTable );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3CoverageReductionMode );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3RepresentativeFragmentTestEnable );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3FeaturesEXT.extendedDynamicState3ShadingRateImageEnable );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicState3PropertiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicState3PropertiesEXT const & physicalDeviceExtendedDynamicState3PropertiesEXT )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3PropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3PropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicState3PropertiesEXT.dynamicPrimitiveTopologyUnrestricted );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceExtendedDynamicStateFeaturesEXT const & physicalDeviceExtendedDynamicStateFeaturesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicStateFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicStateFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExtendedDynamicStateFeaturesEXT.extendedDynamicState );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalBufferInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalBufferInfo const & physicalDeviceExternalBufferInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExternalBufferInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExternalBufferInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExternalBufferInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExternalBufferInfo.usage );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExternalBufferInfo.handleType );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalFenceInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalFenceInfo const & physicalDeviceExternalFenceInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExternalFenceInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExternalFenceInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExternalFenceInfo.handleType );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalImageFormatInfo>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalImageFormatInfo const & physicalDeviceExternalImageFormatInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExternalImageFormatInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExternalImageFormatInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExternalImageFormatInfo.handleType );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalMemoryHostPropertiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalMemoryHostPropertiesEXT const & physicalDeviceExternalMemoryHostPropertiesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExternalMemoryHostPropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExternalMemoryHostPropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExternalMemoryHostPropertiesEXT.minImportedHostPointerAlignment );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalMemoryRDMAFeaturesNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalMemoryRDMAFeaturesNV const & physicalDeviceExternalMemoryRDMAFeaturesNV ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExternalMemoryRDMAFeaturesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExternalMemoryRDMAFeaturesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExternalMemoryRDMAFeaturesNV.externalMemoryRDMA );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalSemaphoreInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceExternalSemaphoreInfo const & physicalDeviceExternalSemaphoreInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExternalSemaphoreInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExternalSemaphoreInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceExternalSemaphoreInfo.handleType );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFaultFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceFaultFeaturesEXT const & physicalDeviceFaultFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFaultFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFaultFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFaultFeaturesEXT.deviceFault );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFaultFeaturesEXT.deviceFaultVendorBinary );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures2 const & physicalDeviceFeatures2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFeatures2.features );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFloatControlsProperties>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceFloatControlsProperties const & physicalDeviceFloatControlsProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFloatControlsProperties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFloatControlsProperties.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFloatControlsProperties.denormBehaviorIndependence );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFloatControlsProperties.roundingModeIndependence );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFloatControlsProperties.shaderSignedZeroInfNanPreserveFloat16 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFloatControlsProperties.shaderSignedZeroInfNanPreserveFloat32 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFloatControlsProperties.shaderSignedZeroInfNanPreserveFloat64 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFloatControlsProperties.shaderDenormPreserveFloat16 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFloatControlsProperties.shaderDenormPreserveFloat32 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFloatControlsProperties.shaderDenormPreserveFloat64 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFloatControlsProperties.shaderDenormFlushToZeroFloat16 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFloatControlsProperties.shaderDenormFlushToZeroFloat32 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFloatControlsProperties.shaderDenormFlushToZeroFloat64 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFloatControlsProperties.shaderRoundingModeRTEFloat16 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFloatControlsProperties.shaderRoundingModeRTEFloat32 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFloatControlsProperties.shaderRoundingModeRTEFloat64 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFloatControlsProperties.shaderRoundingModeRTZFloat16 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFloatControlsProperties.shaderRoundingModeRTZFloat32 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFloatControlsProperties.shaderRoundingModeRTZFloat64 );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMap2FeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMap2FeaturesEXT const & physicalDeviceFragmentDensityMap2FeaturesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMap2FeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMap2FeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMap2FeaturesEXT.fragmentDensityMapDeferred );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMap2PropertiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMap2PropertiesEXT const & physicalDeviceFragmentDensityMap2PropertiesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMap2PropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMap2PropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMap2PropertiesEXT.subsampledLoads );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMap2PropertiesEXT.subsampledCoarseReconstructionEarlyAccess );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMap2PropertiesEXT.maxSubsampledArrayLayers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMap2PropertiesEXT.maxDescriptorSetSubsampledSamplers );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapFeaturesEXT const & physicalDeviceFragmentDensityMapFeaturesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMapFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMapFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMapFeaturesEXT.fragmentDensityMap );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMapFeaturesEXT.fragmentDensityMapDynamic );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMapFeaturesEXT.fragmentDensityMapNonSubsampledImages );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapOffsetFeaturesQCOM>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapOffsetFeaturesQCOM const & physicalDeviceFragmentDensityMapOffsetFeaturesQCOM ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMapOffsetFeaturesQCOM.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMapOffsetFeaturesQCOM.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMapOffsetFeaturesQCOM.fragmentDensityMapOffset );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapOffsetPropertiesQCOM>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapOffsetPropertiesQCOM const &
                              physicalDeviceFragmentDensityMapOffsetPropertiesQCOM ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMapOffsetPropertiesQCOM.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMapOffsetPropertiesQCOM.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMapOffsetPropertiesQCOM.fragmentDensityOffsetGranularity );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapPropertiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentDensityMapPropertiesEXT const & physicalDeviceFragmentDensityMapPropertiesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMapPropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMapPropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMapPropertiesEXT.minFragmentDensityTexelSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMapPropertiesEXT.maxFragmentDensityTexelSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentDensityMapPropertiesEXT.fragmentDensityInvocations );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR const & physicalDeviceFragmentShaderBarycentricFeaturesKHR ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShaderBarycentricFeaturesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShaderBarycentricFeaturesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShaderBarycentricFeaturesKHR.fragmentShaderBarycentric );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShaderBarycentricPropertiesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShaderBarycentricPropertiesKHR const &
                              physicalDeviceFragmentShaderBarycentricPropertiesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShaderBarycentricPropertiesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShaderBarycentricPropertiesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShaderBarycentricPropertiesKHR.triStripVertexOrderIndependentOfProvokingVertex );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShaderInterlockFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShaderInterlockFeaturesEXT const & physicalDeviceFragmentShaderInterlockFeaturesEXT )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShaderInterlockFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShaderInterlockFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShaderInterlockFeaturesEXT.fragmentShaderSampleInterlock );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShaderInterlockFeaturesEXT.fragmentShaderPixelInterlock );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShaderInterlockFeaturesEXT.fragmentShaderShadingRateInterlock );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateEnumsFeaturesNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateEnumsFeaturesNV const & physicalDeviceFragmentShadingRateEnumsFeaturesNV )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRateEnumsFeaturesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRateEnumsFeaturesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRateEnumsFeaturesNV.fragmentShadingRateEnums );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRateEnumsFeaturesNV.supersampleFragmentShadingRates );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRateEnumsFeaturesNV.noInvocationFragmentShadingRates );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateEnumsPropertiesNV>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateEnumsPropertiesNV const & physicalDeviceFragmentShadingRateEnumsPropertiesNV ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRateEnumsPropertiesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRateEnumsPropertiesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRateEnumsPropertiesNV.maxFragmentShadingRateInvocationCount );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateFeaturesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateFeaturesKHR const & physicalDeviceFragmentShadingRateFeaturesKHR ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRateFeaturesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRateFeaturesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRateFeaturesKHR.pipelineFragmentShadingRate );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRateFeaturesKHR.primitiveFragmentShadingRate );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRateFeaturesKHR.attachmentFragmentShadingRate );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateKHR const & physicalDeviceFragmentShadingRateKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRateKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRateKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRateKHR.sampleCounts );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRateKHR.fragmentSize );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRatePropertiesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRatePropertiesKHR const & physicalDeviceFragmentShadingRatePropertiesKHR ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRatePropertiesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRatePropertiesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRatePropertiesKHR.minFragmentShadingRateAttachmentTexelSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRatePropertiesKHR.maxFragmentShadingRateAttachmentTexelSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRatePropertiesKHR.maxFragmentShadingRateAttachmentTexelSizeAspectRatio );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRatePropertiesKHR.primitiveFragmentShadingRateWithMultipleViewports );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRatePropertiesKHR.layeredShadingRateAttachments );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRatePropertiesKHR.fragmentShadingRateNonTrivialCombinerOps );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRatePropertiesKHR.maxFragmentSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRatePropertiesKHR.maxFragmentSizeAspectRatio );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRatePropertiesKHR.maxFragmentShadingRateCoverageSamples );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRatePropertiesKHR.maxFragmentShadingRateRasterizationSamples );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRatePropertiesKHR.fragmentShadingRateWithShaderDepthStencilWrites );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRatePropertiesKHR.fragmentShadingRateWithSampleMask );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRatePropertiesKHR.fragmentShadingRateWithShaderSampleMask );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRatePropertiesKHR.fragmentShadingRateWithConservativeRasterization );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRatePropertiesKHR.fragmentShadingRateWithFragmentShaderInterlock );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRatePropertiesKHR.fragmentShadingRateWithCustomSampleLocations );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceFragmentShadingRatePropertiesKHR.fragmentShadingRateStrictMultiplyCombiner );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceGlobalPriorityQueryFeaturesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceGlobalPriorityQueryFeaturesKHR const & physicalDeviceGlobalPriorityQueryFeaturesKHR ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceGlobalPriorityQueryFeaturesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceGlobalPriorityQueryFeaturesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceGlobalPriorityQueryFeaturesKHR.globalPriorityQuery );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT const & physicalDeviceGraphicsPipelineLibraryFeaturesEXT )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceGraphicsPipelineLibraryFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceGraphicsPipelineLibraryFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceGraphicsPipelineLibraryFeaturesEXT.graphicsPipelineLibrary );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT const & physicalDeviceGraphicsPipelineLibraryPropertiesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceGraphicsPipelineLibraryPropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceGraphicsPipelineLibraryPropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceGraphicsPipelineLibraryPropertiesEXT.graphicsPipelineLibraryFastLinking );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceGraphicsPipelineLibraryPropertiesEXT.graphicsPipelineLibraryIndependentInterpolationDecoration );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceGroupProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceGroupProperties const & physicalDeviceGroupProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceGroupProperties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceGroupProperties.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceGroupProperties.physicalDeviceCount );
      for ( size_t i = 0; i < VK_MAX_DEVICE_GROUP_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceGroupProperties.physicalDevices[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceGroupProperties.subsetAllocation );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceHostQueryResetFeatures>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceHostQueryResetFeatures const & physicalDeviceHostQueryResetFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceHostQueryResetFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceHostQueryResetFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceHostQueryResetFeatures.hostQueryReset );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceIDProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceIDProperties const & physicalDeviceIDProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceIDProperties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceIDProperties.pNext );
      for ( size_t i = 0; i < VK_UUID_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceIDProperties.deviceUUID[i] );
      }
      for ( size_t i = 0; i < VK_UUID_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceIDProperties.driverUUID[i] );
      }
      for ( size_t i = 0; i < VK_LUID_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceIDProperties.deviceLUID[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceIDProperties.deviceNodeMask );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceIDProperties.deviceLUIDValid );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImage2DViewOf3DFeaturesEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceImage2DViewOf3DFeaturesEXT const & physicalDeviceImage2DViewOf3DFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImage2DViewOf3DFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImage2DViewOf3DFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImage2DViewOf3DFeaturesEXT.image2DViewOf3D );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImage2DViewOf3DFeaturesEXT.sampler2DViewOf3D );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageCompressionControlFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceImageCompressionControlFeaturesEXT const & physicalDeviceImageCompressionControlFeaturesEXT )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageCompressionControlFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageCompressionControlFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageCompressionControlFeaturesEXT.imageCompressionControl );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageCompressionControlSwapchainFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceImageCompressionControlSwapchainFeaturesEXT const &
                              physicalDeviceImageCompressionControlSwapchainFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageCompressionControlSwapchainFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageCompressionControlSwapchainFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageCompressionControlSwapchainFeaturesEXT.imageCompressionControlSwapchain );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageDrmFormatModifierInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceImageDrmFormatModifierInfoEXT const & physicalDeviceImageDrmFormatModifierInfoEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageDrmFormatModifierInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageDrmFormatModifierInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageDrmFormatModifierInfoEXT.drmFormatModifier );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageDrmFormatModifierInfoEXT.sharingMode );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageDrmFormatModifierInfoEXT.queueFamilyIndexCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageDrmFormatModifierInfoEXT.pQueueFamilyIndices );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageFormatInfo2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceImageFormatInfo2 const & physicalDeviceImageFormatInfo2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageFormatInfo2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageFormatInfo2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageFormatInfo2.format );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageFormatInfo2.type );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageFormatInfo2.tiling );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageFormatInfo2.usage );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageFormatInfo2.flags );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessingFeaturesQCOM>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessingFeaturesQCOM const & physicalDeviceImageProcessingFeaturesQCOM ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageProcessingFeaturesQCOM.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageProcessingFeaturesQCOM.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageProcessingFeaturesQCOM.textureSampleWeighted );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageProcessingFeaturesQCOM.textureBoxFilter );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageProcessingFeaturesQCOM.textureBlockMatch );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessingPropertiesQCOM>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceImageProcessingPropertiesQCOM const & physicalDeviceImageProcessingPropertiesQCOM ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageProcessingPropertiesQCOM.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageProcessingPropertiesQCOM.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageProcessingPropertiesQCOM.maxWeightFilterPhases );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageProcessingPropertiesQCOM.maxWeightFilterDimension );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageProcessingPropertiesQCOM.maxBlockMatchRegion );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageProcessingPropertiesQCOM.maxBoxFilterBlockSize );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageRobustnessFeatures>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceImageRobustnessFeatures const & physicalDeviceImageRobustnessFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageRobustnessFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageRobustnessFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageRobustnessFeatures.robustImageAccess );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageViewImageFormatInfoEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceImageViewImageFormatInfoEXT const & physicalDeviceImageViewImageFormatInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageViewImageFormatInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageViewImageFormatInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageViewImageFormatInfoEXT.imageViewType );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImageViewMinLodFeaturesEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceImageViewMinLodFeaturesEXT const & physicalDeviceImageViewMinLodFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageViewMinLodFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageViewMinLodFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImageViewMinLodFeaturesEXT.minLod );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceImagelessFramebufferFeatures>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceImagelessFramebufferFeatures const & physicalDeviceImagelessFramebufferFeatures ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImagelessFramebufferFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImagelessFramebufferFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceImagelessFramebufferFeatures.imagelessFramebuffer );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceIndexTypeUint8FeaturesEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceIndexTypeUint8FeaturesEXT const & physicalDeviceIndexTypeUint8FeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceIndexTypeUint8FeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceIndexTypeUint8FeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceIndexTypeUint8FeaturesEXT.indexTypeUint8 );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceInheritedViewportScissorFeaturesNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceInheritedViewportScissorFeaturesNV const & physicalDeviceInheritedViewportScissorFeaturesNV )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceInheritedViewportScissorFeaturesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceInheritedViewportScissorFeaturesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceInheritedViewportScissorFeaturesNV.inheritedViewportScissor2D );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceInlineUniformBlockFeatures>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceInlineUniformBlockFeatures const & physicalDeviceInlineUniformBlockFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceInlineUniformBlockFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceInlineUniformBlockFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceInlineUniformBlockFeatures.inlineUniformBlock );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceInlineUniformBlockFeatures.descriptorBindingInlineUniformBlockUpdateAfterBind );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceInlineUniformBlockProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceInlineUniformBlockProperties const & physicalDeviceInlineUniformBlockProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceInlineUniformBlockProperties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceInlineUniformBlockProperties.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceInlineUniformBlockProperties.maxInlineUniformBlockSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceInlineUniformBlockProperties.maxPerStageDescriptorInlineUniformBlocks );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceInlineUniformBlockProperties.maxPerStageDescriptorUpdateAfterBindInlineUniformBlocks );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceInlineUniformBlockProperties.maxDescriptorSetInlineUniformBlocks );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceInlineUniformBlockProperties.maxDescriptorSetUpdateAfterBindInlineUniformBlocks );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceInvocationMaskFeaturesHUAWEI>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceInvocationMaskFeaturesHUAWEI const & physicalDeviceInvocationMaskFeaturesHUAWEI ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceInvocationMaskFeaturesHUAWEI.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceInvocationMaskFeaturesHUAWEI.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceInvocationMaskFeaturesHUAWEI.invocationMask );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceLegacyDitheringFeaturesEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceLegacyDitheringFeaturesEXT const & physicalDeviceLegacyDitheringFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLegacyDitheringFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLegacyDitheringFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLegacyDitheringFeaturesEXT.legacyDithering );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceLimits>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceLimits const & physicalDeviceLimits ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxImageDimension1D );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxImageDimension2D );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxImageDimension3D );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxImageDimensionCube );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxImageArrayLayers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxTexelBufferElements );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxUniformBufferRange );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxStorageBufferRange );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxPushConstantsSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxMemoryAllocationCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxSamplerAllocationCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.bufferImageGranularity );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.sparseAddressSpaceSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxBoundDescriptorSets );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxPerStageDescriptorSamplers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxPerStageDescriptorUniformBuffers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxPerStageDescriptorStorageBuffers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxPerStageDescriptorSampledImages );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxPerStageDescriptorStorageImages );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxPerStageDescriptorInputAttachments );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxPerStageResources );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxDescriptorSetSamplers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxDescriptorSetUniformBuffers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxDescriptorSetUniformBuffersDynamic );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxDescriptorSetStorageBuffers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxDescriptorSetStorageBuffersDynamic );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxDescriptorSetSampledImages );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxDescriptorSetStorageImages );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxDescriptorSetInputAttachments );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxVertexInputAttributes );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxVertexInputBindings );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxVertexInputAttributeOffset );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxVertexInputBindingStride );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxVertexOutputComponents );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxTessellationGenerationLevel );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxTessellationPatchSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxTessellationControlPerVertexInputComponents );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxTessellationControlPerVertexOutputComponents );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxTessellationControlPerPatchOutputComponents );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxTessellationControlTotalOutputComponents );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxTessellationEvaluationInputComponents );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxTessellationEvaluationOutputComponents );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxGeometryShaderInvocations );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxGeometryInputComponents );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxGeometryOutputComponents );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxGeometryOutputVertices );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxGeometryTotalOutputComponents );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxFragmentInputComponents );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxFragmentOutputAttachments );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxFragmentDualSrcAttachments );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxFragmentCombinedOutputResources );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxComputeSharedMemorySize );
      for ( size_t i = 0; i < 3; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxComputeWorkGroupCount[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxComputeWorkGroupInvocations );
      for ( size_t i = 0; i < 3; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxComputeWorkGroupSize[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.subPixelPrecisionBits );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.subTexelPrecisionBits );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.mipmapPrecisionBits );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxDrawIndexedIndexValue );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxDrawIndirectCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxSamplerLodBias );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxSamplerAnisotropy );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxViewports );
      for ( size_t i = 0; i < 2; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxViewportDimensions[i] );
      }
      for ( size_t i = 0; i < 2; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.viewportBoundsRange[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.viewportSubPixelBits );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.minMemoryMapAlignment );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.minTexelBufferOffsetAlignment );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.minUniformBufferOffsetAlignment );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.minStorageBufferOffsetAlignment );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.minTexelOffset );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxTexelOffset );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.minTexelGatherOffset );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxTexelGatherOffset );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.minInterpolationOffset );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxInterpolationOffset );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.subPixelInterpolationOffsetBits );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxFramebufferWidth );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxFramebufferHeight );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxFramebufferLayers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.framebufferColorSampleCounts );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.framebufferDepthSampleCounts );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.framebufferStencilSampleCounts );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.framebufferNoAttachmentsSampleCounts );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxColorAttachments );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.sampledImageColorSampleCounts );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.sampledImageIntegerSampleCounts );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.sampledImageDepthSampleCounts );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.sampledImageStencilSampleCounts );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.storageImageSampleCounts );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxSampleMaskWords );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.timestampComputeAndGraphics );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.timestampPeriod );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxClipDistances );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxCullDistances );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.maxCombinedClipAndCullDistances );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.discreteQueuePriorities );
      for ( size_t i = 0; i < 2; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.pointSizeRange[i] );
      }
      for ( size_t i = 0; i < 2; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.lineWidthRange[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.pointSizeGranularity );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.lineWidthGranularity );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.strictLines );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.standardSampleLocations );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.optimalBufferCopyOffsetAlignment );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.optimalBufferCopyRowPitchAlignment );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLimits.nonCoherentAtomSize );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceLineRasterizationFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceLineRasterizationFeaturesEXT const & physicalDeviceLineRasterizationFeaturesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLineRasterizationFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLineRasterizationFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLineRasterizationFeaturesEXT.rectangularLines );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLineRasterizationFeaturesEXT.bresenhamLines );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLineRasterizationFeaturesEXT.smoothLines );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLineRasterizationFeaturesEXT.stippledRectangularLines );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLineRasterizationFeaturesEXT.stippledBresenhamLines );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLineRasterizationFeaturesEXT.stippledSmoothLines );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceLineRasterizationPropertiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceLineRasterizationPropertiesEXT const & physicalDeviceLineRasterizationPropertiesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLineRasterizationPropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLineRasterizationPropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLineRasterizationPropertiesEXT.lineSubPixelPrecisionBits );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceLinearColorAttachmentFeaturesNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceLinearColorAttachmentFeaturesNV const & physicalDeviceLinearColorAttachmentFeaturesNV ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLinearColorAttachmentFeaturesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLinearColorAttachmentFeaturesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceLinearColorAttachmentFeaturesNV.linearColorAttachment );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance3Properties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance3Properties const & physicalDeviceMaintenance3Properties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMaintenance3Properties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMaintenance3Properties.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMaintenance3Properties.maxPerSetDescriptors );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMaintenance3Properties.maxMemoryAllocationSize );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance4Features>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance4Features const & physicalDeviceMaintenance4Features ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMaintenance4Features.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMaintenance4Features.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMaintenance4Features.maintenance4 );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance4Properties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceMaintenance4Properties const & physicalDeviceMaintenance4Properties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMaintenance4Properties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMaintenance4Properties.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMaintenance4Properties.maxBufferSize );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryBudgetPropertiesEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryBudgetPropertiesEXT const & physicalDeviceMemoryBudgetPropertiesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMemoryBudgetPropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMemoryBudgetPropertiesEXT.pNext );
      for ( size_t i = 0; i < VK_MAX_MEMORY_HEAPS; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMemoryBudgetPropertiesEXT.heapBudget[i] );
      }
      for ( size_t i = 0; i < VK_MAX_MEMORY_HEAPS; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMemoryBudgetPropertiesEXT.heapUsage[i] );
      }
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryPriorityFeaturesEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryPriorityFeaturesEXT const & physicalDeviceMemoryPriorityFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMemoryPriorityFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMemoryPriorityFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMemoryPriorityFeaturesEXT.memoryPriority );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties const & physicalDeviceMemoryProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMemoryProperties.memoryTypeCount );
      for ( size_t i = 0; i < VK_MAX_MEMORY_TYPES; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMemoryProperties.memoryTypes[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMemoryProperties.memoryHeapCount );
      for ( size_t i = 0; i < VK_MAX_MEMORY_HEAPS; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMemoryProperties.memoryHeaps[i] );
      }
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties2 const & physicalDeviceMemoryProperties2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMemoryProperties2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMemoryProperties2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMemoryProperties2.memoryProperties );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderFeaturesEXT const & physicalDeviceMeshShaderFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderFeaturesEXT.taskShader );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderFeaturesEXT.meshShader );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderFeaturesEXT.multiviewMeshShader );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderFeaturesEXT.primitiveFragmentShadingRateMeshShader );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderFeaturesEXT.meshShaderQueries );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderFeaturesNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderFeaturesNV const & physicalDeviceMeshShaderFeaturesNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderFeaturesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderFeaturesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderFeaturesNV.taskShader );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderFeaturesNV.meshShader );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderPropertiesEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderPropertiesEXT const & physicalDeviceMeshShaderPropertiesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.maxTaskWorkGroupTotalCount );
      for ( size_t i = 0; i < 3; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.maxTaskWorkGroupCount[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.maxTaskWorkGroupInvocations );
      for ( size_t i = 0; i < 3; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.maxTaskWorkGroupSize[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.maxTaskPayloadSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.maxTaskSharedMemorySize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.maxTaskPayloadAndSharedMemorySize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.maxMeshWorkGroupTotalCount );
      for ( size_t i = 0; i < 3; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.maxMeshWorkGroupCount[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.maxMeshWorkGroupInvocations );
      for ( size_t i = 0; i < 3; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.maxMeshWorkGroupSize[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.maxMeshSharedMemorySize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.maxMeshPayloadAndSharedMemorySize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.maxMeshOutputMemorySize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.maxMeshPayloadAndOutputMemorySize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.maxMeshOutputComponents );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.maxMeshOutputVertices );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.maxMeshOutputPrimitives );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.maxMeshOutputLayers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.maxMeshMultiviewViewCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.meshOutputPerVertexGranularity );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.meshOutputPerPrimitiveGranularity );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.maxPreferredTaskWorkGroupInvocations );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.maxPreferredMeshWorkGroupInvocations );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.prefersLocalInvocationVertexOutput );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.prefersLocalInvocationPrimitiveOutput );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.prefersCompactVertexOutput );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesEXT.prefersCompactPrimitiveOutput );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderPropertiesNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceMeshShaderPropertiesNV const & physicalDeviceMeshShaderPropertiesNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesNV.maxDrawMeshTasksCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesNV.maxTaskWorkGroupInvocations );
      for ( size_t i = 0; i < 3; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesNV.maxTaskWorkGroupSize[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesNV.maxTaskTotalMemorySize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesNV.maxTaskOutputCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesNV.maxMeshWorkGroupInvocations );
      for ( size_t i = 0; i < 3; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesNV.maxMeshWorkGroupSize[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesNV.maxMeshTotalMemorySize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesNV.maxMeshOutputVertices );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesNV.maxMeshOutputPrimitives );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesNV.maxMeshMultiviewViewCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesNV.meshOutputPerVertexGranularity );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMeshShaderPropertiesNV.meshOutputPerPrimitiveGranularity );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiDrawFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiDrawFeaturesEXT const & physicalDeviceMultiDrawFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMultiDrawFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMultiDrawFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMultiDrawFeaturesEXT.multiDraw );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiDrawPropertiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiDrawPropertiesEXT const & physicalDeviceMultiDrawPropertiesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMultiDrawPropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMultiDrawPropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMultiDrawPropertiesEXT.maxMultiDrawCount );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultisampledRenderToSingleSampledFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceMultisampledRenderToSingleSampledFeaturesEXT const &
                              physicalDeviceMultisampledRenderToSingleSampledFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMultisampledRenderToSingleSampledFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMultisampledRenderToSingleSampledFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMultisampledRenderToSingleSampledFeaturesEXT.multisampledRenderToSingleSampled );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewFeatures>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewFeatures const & physicalDeviceMultiviewFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMultiviewFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMultiviewFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMultiviewFeatures.multiview );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMultiviewFeatures.multiviewGeometryShader );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMultiviewFeatures.multiviewTessellationShader );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewPerViewAttributesPropertiesNVX>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewPerViewAttributesPropertiesNVX const &
                              physicalDeviceMultiviewPerViewAttributesPropertiesNVX ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMultiviewPerViewAttributesPropertiesNVX.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMultiviewPerViewAttributesPropertiesNVX.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMultiviewPerViewAttributesPropertiesNVX.perViewPositionAllComponents );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceMultiviewProperties const & physicalDeviceMultiviewProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMultiviewProperties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMultiviewProperties.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMultiviewProperties.maxMultiviewViewCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMultiviewProperties.maxMultiviewInstanceIndex );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceMutableDescriptorTypeFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceMutableDescriptorTypeFeaturesEXT const & physicalDeviceMutableDescriptorTypeFeaturesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMutableDescriptorTypeFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMutableDescriptorTypeFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceMutableDescriptorTypeFeaturesEXT.mutableDescriptorType );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceNonSeamlessCubeMapFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceNonSeamlessCubeMapFeaturesEXT const & physicalDeviceNonSeamlessCubeMapFeaturesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceNonSeamlessCubeMapFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceNonSeamlessCubeMapFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceNonSeamlessCubeMapFeaturesEXT.nonSeamlessCubeMap );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceOpacityMicromapFeaturesEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceOpacityMicromapFeaturesEXT const & physicalDeviceOpacityMicromapFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpacityMicromapFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpacityMicromapFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpacityMicromapFeaturesEXT.micromap );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpacityMicromapFeaturesEXT.micromapCaptureReplay );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpacityMicromapFeaturesEXT.micromapHostCommands );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceOpacityMicromapPropertiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceOpacityMicromapPropertiesEXT const & physicalDeviceOpacityMicromapPropertiesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpacityMicromapPropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpacityMicromapPropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpacityMicromapPropertiesEXT.maxOpacity2StateSubdivisionLevel );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpacityMicromapPropertiesEXT.maxOpacity4StateSubdivisionLevel );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceOpticalFlowFeaturesNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceOpticalFlowFeaturesNV const & physicalDeviceOpticalFlowFeaturesNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpticalFlowFeaturesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpticalFlowFeaturesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpticalFlowFeaturesNV.opticalFlow );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceOpticalFlowPropertiesNV>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceOpticalFlowPropertiesNV const & physicalDeviceOpticalFlowPropertiesNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpticalFlowPropertiesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpticalFlowPropertiesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpticalFlowPropertiesNV.supportedOutputGridSizes );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpticalFlowPropertiesNV.supportedHintGridSizes );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpticalFlowPropertiesNV.hintSupported );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpticalFlowPropertiesNV.costSupported );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpticalFlowPropertiesNV.bidirectionalFlowSupported );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpticalFlowPropertiesNV.globalFlowSupported );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpticalFlowPropertiesNV.minWidth );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpticalFlowPropertiesNV.minHeight );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpticalFlowPropertiesNV.maxWidth );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpticalFlowPropertiesNV.maxHeight );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceOpticalFlowPropertiesNV.maxNumRegionsOfInterest );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePCIBusInfoPropertiesEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDevicePCIBusInfoPropertiesEXT const & physicalDevicePCIBusInfoPropertiesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePCIBusInfoPropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePCIBusInfoPropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePCIBusInfoPropertiesEXT.pciDomain );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePCIBusInfoPropertiesEXT.pciBus );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePCIBusInfoPropertiesEXT.pciDevice );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePCIBusInfoPropertiesEXT.pciFunction );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT const & physicalDevicePageableDeviceLocalMemoryFeaturesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePageableDeviceLocalMemoryFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePageableDeviceLocalMemoryFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePageableDeviceLocalMemoryFeaturesEXT.pageableDeviceLocalMemory );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePerformanceQueryFeaturesKHR>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDevicePerformanceQueryFeaturesKHR const & physicalDevicePerformanceQueryFeaturesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePerformanceQueryFeaturesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePerformanceQueryFeaturesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePerformanceQueryFeaturesKHR.performanceCounterQueryPools );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePerformanceQueryFeaturesKHR.performanceCounterMultipleQueryPools );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePerformanceQueryPropertiesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDevicePerformanceQueryPropertiesKHR const & physicalDevicePerformanceQueryPropertiesKHR ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePerformanceQueryPropertiesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePerformanceQueryPropertiesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePerformanceQueryPropertiesKHR.allowCommandBufferQueryCopies );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineCreationCacheControlFeatures>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineCreationCacheControlFeatures const & physicalDevicePipelineCreationCacheControlFeatures ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePipelineCreationCacheControlFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePipelineCreationCacheControlFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePipelineCreationCacheControlFeatures.pipelineCreationCacheControl );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR const &
                              physicalDevicePipelineExecutablePropertiesFeaturesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePipelineExecutablePropertiesFeaturesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePipelineExecutablePropertiesFeaturesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePipelineExecutablePropertiesFeaturesKHR.pipelineExecutableInfo );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelinePropertiesFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDevicePipelinePropertiesFeaturesEXT const & physicalDevicePipelinePropertiesFeaturesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePipelinePropertiesFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePipelinePropertiesFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePipelinePropertiesFeaturesEXT.pipelinePropertiesIdentifier );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineProtectedAccessFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineProtectedAccessFeaturesEXT const & physicalDevicePipelineProtectedAccessFeaturesEXT )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePipelineProtectedAccessFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePipelineProtectedAccessFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePipelineProtectedAccessFeaturesEXT.pipelineProtectedAccess );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineRobustnessFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineRobustnessFeaturesEXT const & physicalDevicePipelineRobustnessFeaturesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePipelineRobustnessFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePipelineRobustnessFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePipelineRobustnessFeaturesEXT.pipelineRobustness );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineRobustnessPropertiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDevicePipelineRobustnessPropertiesEXT const & physicalDevicePipelineRobustnessPropertiesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePipelineRobustnessPropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePipelineRobustnessPropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePipelineRobustnessPropertiesEXT.defaultRobustnessStorageBuffers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePipelineRobustnessPropertiesEXT.defaultRobustnessUniformBuffers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePipelineRobustnessPropertiesEXT.defaultRobustnessVertexInputs );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePipelineRobustnessPropertiesEXT.defaultRobustnessImages );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePointClippingProperties>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDevicePointClippingProperties const & physicalDevicePointClippingProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePointClippingProperties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePointClippingProperties.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePointClippingProperties.pointClippingBehavior );
      return seed;
    }
  };

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePortabilitySubsetFeaturesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDevicePortabilitySubsetFeaturesKHR const & physicalDevicePortabilitySubsetFeaturesKHR ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePortabilitySubsetFeaturesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePortabilitySubsetFeaturesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePortabilitySubsetFeaturesKHR.constantAlphaColorBlendFactors );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePortabilitySubsetFeaturesKHR.events );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePortabilitySubsetFeaturesKHR.imageViewFormatReinterpretation );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePortabilitySubsetFeaturesKHR.imageViewFormatSwizzle );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePortabilitySubsetFeaturesKHR.imageView2DOn3DImage );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePortabilitySubsetFeaturesKHR.multisampleArrayImage );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePortabilitySubsetFeaturesKHR.mutableComparisonSamplers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePortabilitySubsetFeaturesKHR.pointPolygons );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePortabilitySubsetFeaturesKHR.samplerMipLodBias );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePortabilitySubsetFeaturesKHR.separateStencilMaskRef );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePortabilitySubsetFeaturesKHR.shaderSampleRateInterpolationFunctions );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePortabilitySubsetFeaturesKHR.tessellationIsolines );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePortabilitySubsetFeaturesKHR.tessellationPointMode );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePortabilitySubsetFeaturesKHR.triangleFans );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePortabilitySubsetFeaturesKHR.vertexAttributeAccessBeyondStride );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePortabilitySubsetPropertiesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDevicePortabilitySubsetPropertiesKHR const & physicalDevicePortabilitySubsetPropertiesKHR ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePortabilitySubsetPropertiesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePortabilitySubsetPropertiesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePortabilitySubsetPropertiesKHR.minVertexInputBindingStrideAlignment );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePresentBarrierFeaturesNV>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDevicePresentBarrierFeaturesNV const & physicalDevicePresentBarrierFeaturesNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePresentBarrierFeaturesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePresentBarrierFeaturesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePresentBarrierFeaturesNV.presentBarrier );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePresentIdFeaturesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDevicePresentIdFeaturesKHR const & physicalDevicePresentIdFeaturesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePresentIdFeaturesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePresentIdFeaturesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePresentIdFeaturesKHR.presentId );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePresentWaitFeaturesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDevicePresentWaitFeaturesKHR const & physicalDevicePresentWaitFeaturesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePresentWaitFeaturesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePresentWaitFeaturesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePresentWaitFeaturesKHR.presentWait );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT const &
                              physicalDevicePrimitiveTopologyListRestartFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePrimitiveTopologyListRestartFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePrimitiveTopologyListRestartFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePrimitiveTopologyListRestartFeaturesEXT.primitiveTopologyListRestart );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePrimitiveTopologyListRestartFeaturesEXT.primitiveTopologyPatchListRestart );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePrimitivesGeneratedQueryFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDevicePrimitivesGeneratedQueryFeaturesEXT const & physicalDevicePrimitivesGeneratedQueryFeaturesEXT )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePrimitivesGeneratedQueryFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePrimitivesGeneratedQueryFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePrimitivesGeneratedQueryFeaturesEXT.primitivesGeneratedQuery );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePrimitivesGeneratedQueryFeaturesEXT.primitivesGeneratedQueryWithRasterizerDiscard );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePrimitivesGeneratedQueryFeaturesEXT.primitivesGeneratedQueryWithNonZeroStreams );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePrivateDataFeatures>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDevicePrivateDataFeatures const & physicalDevicePrivateDataFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePrivateDataFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePrivateDataFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePrivateDataFeatures.privateData );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSparseProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceSparseProperties const & physicalDeviceSparseProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSparseProperties.residencyStandard2DBlockShape );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSparseProperties.residencyStandard2DMultisampleBlockShape );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSparseProperties.residencyStandard3DBlockShape );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSparseProperties.residencyAlignedMipSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSparseProperties.residencyNonResidentStrict );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties const & physicalDeviceProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProperties.apiVersion );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProperties.driverVersion );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProperties.vendorID );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProperties.deviceID );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProperties.deviceType );
      for ( size_t i = 0; i < VK_MAX_PHYSICAL_DEVICE_NAME_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProperties.deviceName[i] );
      }
      for ( size_t i = 0; i < VK_UUID_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProperties.pipelineCacheUUID[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProperties.limits );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProperties.sparseProperties );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties2 const & physicalDeviceProperties2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProperties2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProperties2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProperties2.properties );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceProtectedMemoryFeatures>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceProtectedMemoryFeatures const & physicalDeviceProtectedMemoryFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProtectedMemoryFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProtectedMemoryFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProtectedMemoryFeatures.protectedMemory );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceProtectedMemoryProperties>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceProtectedMemoryProperties const & physicalDeviceProtectedMemoryProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProtectedMemoryProperties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProtectedMemoryProperties.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProtectedMemoryProperties.protectedNoFault );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceProvokingVertexFeaturesEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceProvokingVertexFeaturesEXT const & physicalDeviceProvokingVertexFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProvokingVertexFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProvokingVertexFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProvokingVertexFeaturesEXT.provokingVertexLast );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProvokingVertexFeaturesEXT.transformFeedbackPreservesProvokingVertex );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceProvokingVertexPropertiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceProvokingVertexPropertiesEXT const & physicalDeviceProvokingVertexPropertiesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProvokingVertexPropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProvokingVertexPropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProvokingVertexPropertiesEXT.provokingVertexModePerPipeline );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceProvokingVertexPropertiesEXT.transformFeedbackPreservesTriangleFanProvokingVertex );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDevicePushDescriptorPropertiesKHR>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDevicePushDescriptorPropertiesKHR const & physicalDevicePushDescriptorPropertiesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePushDescriptorPropertiesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePushDescriptorPropertiesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDevicePushDescriptorPropertiesKHR.maxPushDescriptors );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRGBA10X6FormatsFeaturesEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceRGBA10X6FormatsFeaturesEXT const & physicalDeviceRGBA10X6FormatsFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRGBA10X6FormatsFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRGBA10X6FormatsFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRGBA10X6FormatsFeaturesEXT.formatRgba10x6WithoutYCbCrSampler );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT const &
                              physicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT.rasterizationOrderColorAttachmentAccess );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT.rasterizationOrderDepthAttachmentAccess );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT.rasterizationOrderStencilAttachmentAccess );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayQueryFeaturesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceRayQueryFeaturesKHR const & physicalDeviceRayQueryFeaturesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayQueryFeaturesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayQueryFeaturesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayQueryFeaturesKHR.rayQuery );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingMaintenance1FeaturesKHR>
  {
    std::size_t operator()(
      VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingMaintenance1FeaturesKHR const & physicalDeviceRayTracingMaintenance1FeaturesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingMaintenance1FeaturesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingMaintenance1FeaturesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingMaintenance1FeaturesKHR.rayTracingMaintenance1 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingMaintenance1FeaturesKHR.rayTracingPipelineTraceRaysIndirect2 );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingMotionBlurFeaturesNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingMotionBlurFeaturesNV const & physicalDeviceRayTracingMotionBlurFeaturesNV ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingMotionBlurFeaturesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingMotionBlurFeaturesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingMotionBlurFeaturesNV.rayTracingMotionBlur );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingMotionBlurFeaturesNV.rayTracingMotionBlurPipelineTraceRaysIndirect );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPipelineFeaturesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPipelineFeaturesKHR const & physicalDeviceRayTracingPipelineFeaturesKHR ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPipelineFeaturesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPipelineFeaturesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPipelineFeaturesKHR.rayTracingPipeline );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPipelineFeaturesKHR.rayTracingPipelineShaderGroupHandleCaptureReplay );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPipelineFeaturesKHR.rayTracingPipelineShaderGroupHandleCaptureReplayMixed );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPipelineFeaturesKHR.rayTracingPipelineTraceRaysIndirect );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPipelineFeaturesKHR.rayTraversalPrimitiveCulling );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPipelinePropertiesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPipelinePropertiesKHR const & physicalDeviceRayTracingPipelinePropertiesKHR ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPipelinePropertiesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPipelinePropertiesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPipelinePropertiesKHR.shaderGroupHandleSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPipelinePropertiesKHR.maxRayRecursionDepth );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPipelinePropertiesKHR.maxShaderGroupStride );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPipelinePropertiesKHR.shaderGroupBaseAlignment );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPipelinePropertiesKHR.shaderGroupHandleCaptureReplaySize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPipelinePropertiesKHR.maxRayDispatchInvocationCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPipelinePropertiesKHR.shaderGroupHandleAlignment );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPipelinePropertiesKHR.maxRayHitAttributeSize );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPropertiesNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceRayTracingPropertiesNV const & physicalDeviceRayTracingPropertiesNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPropertiesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPropertiesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPropertiesNV.shaderGroupHandleSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPropertiesNV.maxRecursionDepth );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPropertiesNV.maxShaderGroupStride );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPropertiesNV.shaderGroupBaseAlignment );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPropertiesNV.maxGeometryCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPropertiesNV.maxInstanceCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPropertiesNV.maxTriangleCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRayTracingPropertiesNV.maxDescriptorSetAccelerationStructures );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRepresentativeFragmentTestFeaturesNV>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceRepresentativeFragmentTestFeaturesNV const & physicalDeviceRepresentativeFragmentTestFeaturesNV ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRepresentativeFragmentTestFeaturesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRepresentativeFragmentTestFeaturesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRepresentativeFragmentTestFeaturesNV.representativeFragmentTest );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRobustness2FeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceRobustness2FeaturesEXT const & physicalDeviceRobustness2FeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRobustness2FeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRobustness2FeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRobustness2FeaturesEXT.robustBufferAccess2 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRobustness2FeaturesEXT.robustImageAccess2 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRobustness2FeaturesEXT.nullDescriptor );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceRobustness2PropertiesEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceRobustness2PropertiesEXT const & physicalDeviceRobustness2PropertiesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRobustness2PropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRobustness2PropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRobustness2PropertiesEXT.robustStorageBufferAccessSizeAlignment );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceRobustness2PropertiesEXT.robustUniformBufferAccessSizeAlignment );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSampleLocationsPropertiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceSampleLocationsPropertiesEXT const & physicalDeviceSampleLocationsPropertiesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSampleLocationsPropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSampleLocationsPropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSampleLocationsPropertiesEXT.sampleLocationSampleCounts );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSampleLocationsPropertiesEXT.maxSampleLocationGridSize );
      for ( size_t i = 0; i < 2; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSampleLocationsPropertiesEXT.sampleLocationCoordinateRange[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSampleLocationsPropertiesEXT.sampleLocationSubPixelBits );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSampleLocationsPropertiesEXT.variableSampleLocations );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSamplerFilterMinmaxProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceSamplerFilterMinmaxProperties const & physicalDeviceSamplerFilterMinmaxProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSamplerFilterMinmaxProperties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSamplerFilterMinmaxProperties.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSamplerFilterMinmaxProperties.filterMinmaxSingleComponentFormats );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSamplerFilterMinmaxProperties.filterMinmaxImageComponentMapping );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSamplerYcbcrConversionFeatures>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceSamplerYcbcrConversionFeatures const & physicalDeviceSamplerYcbcrConversionFeatures ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSamplerYcbcrConversionFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSamplerYcbcrConversionFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSamplerYcbcrConversionFeatures.samplerYcbcrConversion );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceScalarBlockLayoutFeatures>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceScalarBlockLayoutFeatures const & physicalDeviceScalarBlockLayoutFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceScalarBlockLayoutFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceScalarBlockLayoutFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceScalarBlockLayoutFeatures.scalarBlockLayout );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSeparateDepthStencilLayoutsFeatures>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceSeparateDepthStencilLayoutsFeatures const & physicalDeviceSeparateDepthStencilLayoutsFeatures )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSeparateDepthStencilLayoutsFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSeparateDepthStencilLayoutsFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSeparateDepthStencilLayoutsFeatures.separateDepthStencilLayouts );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicFloat2FeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicFloat2FeaturesEXT const & physicalDeviceShaderAtomicFloat2FeaturesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloat2FeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloat2FeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloat2FeaturesEXT.shaderBufferFloat16Atomics );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloat2FeaturesEXT.shaderBufferFloat16AtomicAdd );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloat2FeaturesEXT.shaderBufferFloat16AtomicMinMax );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloat2FeaturesEXT.shaderBufferFloat32AtomicMinMax );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloat2FeaturesEXT.shaderBufferFloat64AtomicMinMax );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloat2FeaturesEXT.shaderSharedFloat16Atomics );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloat2FeaturesEXT.shaderSharedFloat16AtomicAdd );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloat2FeaturesEXT.shaderSharedFloat16AtomicMinMax );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloat2FeaturesEXT.shaderSharedFloat32AtomicMinMax );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloat2FeaturesEXT.shaderSharedFloat64AtomicMinMax );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloat2FeaturesEXT.shaderImageFloat32AtomicMinMax );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloat2FeaturesEXT.sparseImageFloat32AtomicMinMax );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicFloatFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicFloatFeaturesEXT const & physicalDeviceShaderAtomicFloatFeaturesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloatFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloatFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloatFeaturesEXT.shaderBufferFloat32Atomics );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloatFeaturesEXT.shaderBufferFloat32AtomicAdd );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloatFeaturesEXT.shaderBufferFloat64Atomics );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloatFeaturesEXT.shaderBufferFloat64AtomicAdd );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloatFeaturesEXT.shaderSharedFloat32Atomics );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloatFeaturesEXT.shaderSharedFloat32AtomicAdd );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloatFeaturesEXT.shaderSharedFloat64Atomics );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloatFeaturesEXT.shaderSharedFloat64AtomicAdd );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloatFeaturesEXT.shaderImageFloat32Atomics );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloatFeaturesEXT.shaderImageFloat32AtomicAdd );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloatFeaturesEXT.sparseImageFloat32Atomics );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicFloatFeaturesEXT.sparseImageFloat32AtomicAdd );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicInt64Features>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderAtomicInt64Features const & physicalDeviceShaderAtomicInt64Features ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicInt64Features.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicInt64Features.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicInt64Features.shaderBufferInt64Atomics );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderAtomicInt64Features.shaderSharedInt64Atomics );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderClockFeaturesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderClockFeaturesKHR const & physicalDeviceShaderClockFeaturesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderClockFeaturesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderClockFeaturesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderClockFeaturesKHR.shaderSubgroupClock );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderClockFeaturesKHR.shaderDeviceClock );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCoreProperties2AMD>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCoreProperties2AMD const & physicalDeviceShaderCoreProperties2AMD ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderCoreProperties2AMD.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderCoreProperties2AMD.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderCoreProperties2AMD.shaderCoreFeatures );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderCoreProperties2AMD.activeComputeUnitCount );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCorePropertiesAMD>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderCorePropertiesAMD const & physicalDeviceShaderCorePropertiesAMD ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderCorePropertiesAMD.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderCorePropertiesAMD.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderCorePropertiesAMD.shaderEngineCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderCorePropertiesAMD.shaderArraysPerEngineCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderCorePropertiesAMD.computeUnitsPerShaderArray );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderCorePropertiesAMD.simdPerComputeUnit );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderCorePropertiesAMD.wavefrontsPerSimd );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderCorePropertiesAMD.wavefrontSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderCorePropertiesAMD.sgprsPerSimd );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderCorePropertiesAMD.minSgprAllocation );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderCorePropertiesAMD.maxSgprAllocation );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderCorePropertiesAMD.sgprAllocationGranularity );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderCorePropertiesAMD.vgprsPerSimd );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderCorePropertiesAMD.minVgprAllocation );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderCorePropertiesAMD.maxVgprAllocation );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderCorePropertiesAMD.vgprAllocationGranularity );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderDemoteToHelperInvocationFeatures>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderDemoteToHelperInvocationFeatures const &
                              physicalDeviceShaderDemoteToHelperInvocationFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderDemoteToHelperInvocationFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderDemoteToHelperInvocationFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderDemoteToHelperInvocationFeatures.shaderDemoteToHelperInvocation );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderDrawParametersFeatures>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderDrawParametersFeatures const & physicalDeviceShaderDrawParametersFeatures ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderDrawParametersFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderDrawParametersFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderDrawParametersFeatures.shaderDrawParameters );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderEarlyAndLateFragmentTestsFeaturesAMD>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderEarlyAndLateFragmentTestsFeaturesAMD const &
                              physicalDeviceShaderEarlyAndLateFragmentTestsFeaturesAMD ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderEarlyAndLateFragmentTestsFeaturesAMD.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderEarlyAndLateFragmentTestsFeaturesAMD.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderEarlyAndLateFragmentTestsFeaturesAMD.shaderEarlyAndLateFragmentTests );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderFloat16Int8Features>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderFloat16Int8Features const & physicalDeviceShaderFloat16Int8Features ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderFloat16Int8Features.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderFloat16Int8Features.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderFloat16Int8Features.shaderFloat16 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderFloat16Int8Features.shaderInt8 );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT>
  {
    std::size_t operator()(
      VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT const & physicalDeviceShaderImageAtomicInt64FeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderImageAtomicInt64FeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderImageAtomicInt64FeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderImageAtomicInt64FeaturesEXT.shaderImageInt64Atomics );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderImageAtomicInt64FeaturesEXT.sparseImageInt64Atomics );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderImageFootprintFeaturesNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderImageFootprintFeaturesNV const & physicalDeviceShaderImageFootprintFeaturesNV ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderImageFootprintFeaturesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderImageFootprintFeaturesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderImageFootprintFeaturesNV.imageFootprint );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderIntegerDotProductFeatures>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderIntegerDotProductFeatures const & physicalDeviceShaderIntegerDotProductFeatures ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductFeatures.shaderIntegerDotProduct );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderIntegerDotProductProperties>
  {
    std::size_t operator()(
      VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderIntegerDotProductProperties const & physicalDeviceShaderIntegerDotProductProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProduct8BitUnsignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProduct8BitSignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProduct8BitMixedSignednessAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProduct4x8BitPackedUnsignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProduct4x8BitPackedSignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProduct4x8BitPackedMixedSignednessAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProduct16BitUnsignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProduct16BitSignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProduct16BitMixedSignednessAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProduct32BitUnsignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProduct32BitSignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProduct32BitMixedSignednessAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProduct64BitUnsignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProduct64BitSignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProduct64BitMixedSignednessAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating8BitUnsignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating8BitSignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed,
                               physicalDeviceShaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating16BitUnsignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating16BitSignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating32BitUnsignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating32BitSignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating64BitUnsignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating64BitSignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerDotProductProperties.integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderIntegerFunctions2FeaturesINTEL>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderIntegerFunctions2FeaturesINTEL const & physicalDeviceShaderIntegerFunctions2FeaturesINTEL ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerFunctions2FeaturesINTEL.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerFunctions2FeaturesINTEL.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderIntegerFunctions2FeaturesINTEL.shaderIntegerFunctions2 );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderModuleIdentifierFeaturesEXT>
  {
    std::size_t operator()(
      VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderModuleIdentifierFeaturesEXT const & physicalDeviceShaderModuleIdentifierFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderModuleIdentifierFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderModuleIdentifierFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderModuleIdentifierFeaturesEXT.shaderModuleIdentifier );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderModuleIdentifierPropertiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderModuleIdentifierPropertiesEXT const & physicalDeviceShaderModuleIdentifierPropertiesEXT )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderModuleIdentifierPropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderModuleIdentifierPropertiesEXT.pNext );
      for ( size_t i = 0; i < VK_UUID_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderModuleIdentifierPropertiesEXT.shaderModuleIdentifierAlgorithmUUID[i] );
      }
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSMBuiltinsFeaturesNV>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSMBuiltinsFeaturesNV const & physicalDeviceShaderSMBuiltinsFeaturesNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderSMBuiltinsFeaturesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderSMBuiltinsFeaturesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderSMBuiltinsFeaturesNV.shaderSMBuiltins );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSMBuiltinsPropertiesNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSMBuiltinsPropertiesNV const & physicalDeviceShaderSMBuiltinsPropertiesNV ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderSMBuiltinsPropertiesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderSMBuiltinsPropertiesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderSMBuiltinsPropertiesNV.shaderSMCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderSMBuiltinsPropertiesNV.shaderWarpsPerSM );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSubgroupExtendedTypesFeatures>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSubgroupExtendedTypesFeatures const & physicalDeviceShaderSubgroupExtendedTypesFeatures )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderSubgroupExtendedTypesFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderSubgroupExtendedTypesFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderSubgroupExtendedTypesFeatures.shaderSubgroupExtendedTypes );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR const &
                              physicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR.shaderSubgroupUniformControlFlow );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderTerminateInvocationFeatures>
  {
    std::size_t operator()(
      VULKAN_HPP_NAMESPACE::PhysicalDeviceShaderTerminateInvocationFeatures const & physicalDeviceShaderTerminateInvocationFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderTerminateInvocationFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderTerminateInvocationFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShaderTerminateInvocationFeatures.shaderTerminateInvocation );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShadingRateImageFeaturesNV>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceShadingRateImageFeaturesNV const & physicalDeviceShadingRateImageFeaturesNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShadingRateImageFeaturesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShadingRateImageFeaturesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShadingRateImageFeaturesNV.shadingRateImage );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShadingRateImageFeaturesNV.shadingRateCoarseSampleOrder );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceShadingRateImagePropertiesNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceShadingRateImagePropertiesNV const & physicalDeviceShadingRateImagePropertiesNV ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShadingRateImagePropertiesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShadingRateImagePropertiesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShadingRateImagePropertiesNV.shadingRateTexelSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShadingRateImagePropertiesNV.shadingRatePaletteSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceShadingRateImagePropertiesNV.shadingRateMaxCoarseSamples );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSparseImageFormatInfo2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceSparseImageFormatInfo2 const & physicalDeviceSparseImageFormatInfo2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSparseImageFormatInfo2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSparseImageFormatInfo2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSparseImageFormatInfo2.format );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSparseImageFormatInfo2.type );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSparseImageFormatInfo2.samples );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSparseImageFormatInfo2.usage );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSparseImageFormatInfo2.tiling );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSubgroupProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceSubgroupProperties const & physicalDeviceSubgroupProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubgroupProperties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubgroupProperties.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubgroupProperties.subgroupSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubgroupProperties.supportedStages );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubgroupProperties.supportedOperations );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubgroupProperties.quadOperationsInAllStages );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSubgroupSizeControlFeatures>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceSubgroupSizeControlFeatures const & physicalDeviceSubgroupSizeControlFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubgroupSizeControlFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubgroupSizeControlFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubgroupSizeControlFeatures.subgroupSizeControl );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubgroupSizeControlFeatures.computeFullSubgroups );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSubgroupSizeControlProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceSubgroupSizeControlProperties const & physicalDeviceSubgroupSizeControlProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubgroupSizeControlProperties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubgroupSizeControlProperties.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubgroupSizeControlProperties.minSubgroupSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubgroupSizeControlProperties.maxSubgroupSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubgroupSizeControlProperties.maxComputeWorkgroupSubgroups );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubgroupSizeControlProperties.requiredSubgroupSizeStages );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSubpassMergeFeedbackFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceSubpassMergeFeedbackFeaturesEXT const & physicalDeviceSubpassMergeFeedbackFeaturesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubpassMergeFeedbackFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubpassMergeFeedbackFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubpassMergeFeedbackFeaturesEXT.subpassMergeFeedback );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSubpassShadingFeaturesHUAWEI>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceSubpassShadingFeaturesHUAWEI const & physicalDeviceSubpassShadingFeaturesHUAWEI ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubpassShadingFeaturesHUAWEI.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubpassShadingFeaturesHUAWEI.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubpassShadingFeaturesHUAWEI.subpassShading );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSubpassShadingPropertiesHUAWEI>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceSubpassShadingPropertiesHUAWEI const & physicalDeviceSubpassShadingPropertiesHUAWEI ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubpassShadingPropertiesHUAWEI.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubpassShadingPropertiesHUAWEI.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSubpassShadingPropertiesHUAWEI.maxSubpassShadingWorkgroupSizeAspectRatio );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSurfaceInfo2KHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceSurfaceInfo2KHR const & physicalDeviceSurfaceInfo2KHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSurfaceInfo2KHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSurfaceInfo2KHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSurfaceInfo2KHR.surface );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceSynchronization2Features>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceSynchronization2Features const & physicalDeviceSynchronization2Features ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSynchronization2Features.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSynchronization2Features.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceSynchronization2Features.synchronization2 );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceTexelBufferAlignmentFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceTexelBufferAlignmentFeaturesEXT const & physicalDeviceTexelBufferAlignmentFeaturesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTexelBufferAlignmentFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTexelBufferAlignmentFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTexelBufferAlignmentFeaturesEXT.texelBufferAlignment );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceTexelBufferAlignmentProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceTexelBufferAlignmentProperties const & physicalDeviceTexelBufferAlignmentProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTexelBufferAlignmentProperties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTexelBufferAlignmentProperties.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTexelBufferAlignmentProperties.storageTexelBufferOffsetAlignmentBytes );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTexelBufferAlignmentProperties.storageTexelBufferOffsetSingleTexelAlignment );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTexelBufferAlignmentProperties.uniformTexelBufferOffsetAlignmentBytes );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTexelBufferAlignmentProperties.uniformTexelBufferOffsetSingleTexelAlignment );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceTextureCompressionASTCHDRFeatures>
  {
    std::size_t operator()(
      VULKAN_HPP_NAMESPACE::PhysicalDeviceTextureCompressionASTCHDRFeatures const & physicalDeviceTextureCompressionASTCHDRFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTextureCompressionASTCHDRFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTextureCompressionASTCHDRFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTextureCompressionASTCHDRFeatures.textureCompressionASTC_HDR );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceTilePropertiesFeaturesQCOM>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceTilePropertiesFeaturesQCOM const & physicalDeviceTilePropertiesFeaturesQCOM ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTilePropertiesFeaturesQCOM.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTilePropertiesFeaturesQCOM.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTilePropertiesFeaturesQCOM.tileProperties );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceTimelineSemaphoreFeatures>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceTimelineSemaphoreFeatures const & physicalDeviceTimelineSemaphoreFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTimelineSemaphoreFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTimelineSemaphoreFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTimelineSemaphoreFeatures.timelineSemaphore );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceTimelineSemaphoreProperties>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceTimelineSemaphoreProperties const & physicalDeviceTimelineSemaphoreProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTimelineSemaphoreProperties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTimelineSemaphoreProperties.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTimelineSemaphoreProperties.maxTimelineSemaphoreValueDifference );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceToolProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceToolProperties const & physicalDeviceToolProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceToolProperties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceToolProperties.pNext );
      for ( size_t i = 0; i < VK_MAX_EXTENSION_NAME_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceToolProperties.name[i] );
      }
      for ( size_t i = 0; i < VK_MAX_EXTENSION_NAME_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceToolProperties.version[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceToolProperties.purposes );
      for ( size_t i = 0; i < VK_MAX_DESCRIPTION_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceToolProperties.description[i] );
      }
      for ( size_t i = 0; i < VK_MAX_EXTENSION_NAME_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceToolProperties.layer[i] );
      }
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceTransformFeedbackFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceTransformFeedbackFeaturesEXT const & physicalDeviceTransformFeedbackFeaturesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTransformFeedbackFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTransformFeedbackFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTransformFeedbackFeaturesEXT.transformFeedback );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTransformFeedbackFeaturesEXT.geometryStreams );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceTransformFeedbackPropertiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceTransformFeedbackPropertiesEXT const & physicalDeviceTransformFeedbackPropertiesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTransformFeedbackPropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTransformFeedbackPropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTransformFeedbackPropertiesEXT.maxTransformFeedbackStreams );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTransformFeedbackPropertiesEXT.maxTransformFeedbackBuffers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTransformFeedbackPropertiesEXT.maxTransformFeedbackBufferSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTransformFeedbackPropertiesEXT.maxTransformFeedbackStreamDataSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTransformFeedbackPropertiesEXT.maxTransformFeedbackBufferDataSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTransformFeedbackPropertiesEXT.maxTransformFeedbackBufferDataStride );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTransformFeedbackPropertiesEXT.transformFeedbackQueries );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTransformFeedbackPropertiesEXT.transformFeedbackStreamsLinesTriangles );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTransformFeedbackPropertiesEXT.transformFeedbackRasterizationStreamSelect );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceTransformFeedbackPropertiesEXT.transformFeedbackDraw );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceUniformBufferStandardLayoutFeatures>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceUniformBufferStandardLayoutFeatures const & physicalDeviceUniformBufferStandardLayoutFeatures )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceUniformBufferStandardLayoutFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceUniformBufferStandardLayoutFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceUniformBufferStandardLayoutFeatures.uniformBufferStandardLayout );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVariablePointersFeatures>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceVariablePointersFeatures const & physicalDeviceVariablePointersFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVariablePointersFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVariablePointersFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVariablePointersFeatures.variablePointersStorageBuffer );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVariablePointersFeatures.variablePointers );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeDivisorFeaturesEXT>
  {
    std::size_t operator()(
      VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeDivisorFeaturesEXT const & physicalDeviceVertexAttributeDivisorFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVertexAttributeDivisorFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVertexAttributeDivisorFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVertexAttributeDivisorFeaturesEXT.vertexAttributeInstanceRateDivisor );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVertexAttributeDivisorFeaturesEXT.vertexAttributeInstanceRateZeroDivisor );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeDivisorPropertiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexAttributeDivisorPropertiesEXT const & physicalDeviceVertexAttributeDivisorPropertiesEXT )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVertexAttributeDivisorPropertiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVertexAttributeDivisorPropertiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVertexAttributeDivisorPropertiesEXT.maxVertexAttribDivisor );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexInputDynamicStateFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceVertexInputDynamicStateFeaturesEXT const & physicalDeviceVertexInputDynamicStateFeaturesEXT )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVertexInputDynamicStateFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVertexInputDynamicStateFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVertexInputDynamicStateFeaturesEXT.vertexInputDynamicState );
      return seed;
    }
  };

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoFormatInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceVideoFormatInfoKHR const & physicalDeviceVideoFormatInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVideoFormatInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVideoFormatInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVideoFormatInfoKHR.imageUsage );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan11Features>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan11Features const & physicalDeviceVulkan11Features ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Features.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Features.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Features.storageBuffer16BitAccess );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Features.uniformAndStorageBuffer16BitAccess );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Features.storagePushConstant16 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Features.storageInputOutput16 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Features.multiview );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Features.multiviewGeometryShader );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Features.multiviewTessellationShader );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Features.variablePointersStorageBuffer );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Features.variablePointers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Features.protectedMemory );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Features.samplerYcbcrConversion );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Features.shaderDrawParameters );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan11Properties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan11Properties const & physicalDeviceVulkan11Properties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Properties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Properties.pNext );
      for ( size_t i = 0; i < VK_UUID_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Properties.deviceUUID[i] );
      }
      for ( size_t i = 0; i < VK_UUID_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Properties.driverUUID[i] );
      }
      for ( size_t i = 0; i < VK_LUID_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Properties.deviceLUID[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Properties.deviceNodeMask );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Properties.deviceLUIDValid );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Properties.subgroupSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Properties.subgroupSupportedStages );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Properties.subgroupSupportedOperations );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Properties.subgroupQuadOperationsInAllStages );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Properties.pointClippingBehavior );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Properties.maxMultiviewViewCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Properties.maxMultiviewInstanceIndex );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Properties.protectedNoFault );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Properties.maxPerSetDescriptors );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan11Properties.maxMemoryAllocationSize );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan12Features>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan12Features const & physicalDeviceVulkan12Features ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.samplerMirrorClampToEdge );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.drawIndirectCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.storageBuffer8BitAccess );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.uniformAndStorageBuffer8BitAccess );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.storagePushConstant8 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.shaderBufferInt64Atomics );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.shaderSharedInt64Atomics );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.shaderFloat16 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.shaderInt8 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.descriptorIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.shaderInputAttachmentArrayDynamicIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.shaderUniformTexelBufferArrayDynamicIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.shaderStorageTexelBufferArrayDynamicIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.shaderUniformBufferArrayNonUniformIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.shaderSampledImageArrayNonUniformIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.shaderStorageBufferArrayNonUniformIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.shaderStorageImageArrayNonUniformIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.shaderInputAttachmentArrayNonUniformIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.shaderUniformTexelBufferArrayNonUniformIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.shaderStorageTexelBufferArrayNonUniformIndexing );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.descriptorBindingUniformBufferUpdateAfterBind );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.descriptorBindingSampledImageUpdateAfterBind );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.descriptorBindingStorageImageUpdateAfterBind );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.descriptorBindingStorageBufferUpdateAfterBind );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.descriptorBindingUniformTexelBufferUpdateAfterBind );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.descriptorBindingStorageTexelBufferUpdateAfterBind );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.descriptorBindingUpdateUnusedWhilePending );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.descriptorBindingPartiallyBound );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.descriptorBindingVariableDescriptorCount );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.runtimeDescriptorArray );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.samplerFilterMinmax );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.scalarBlockLayout );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.imagelessFramebuffer );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.uniformBufferStandardLayout );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.shaderSubgroupExtendedTypes );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.separateDepthStencilLayouts );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.hostQueryReset );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.timelineSemaphore );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.bufferDeviceAddress );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.bufferDeviceAddressCaptureReplay );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.bufferDeviceAddressMultiDevice );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.vulkanMemoryModel );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.vulkanMemoryModelDeviceScope );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.vulkanMemoryModelAvailabilityVisibilityChains );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.shaderOutputViewportIndex );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.shaderOutputLayer );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Features.subgroupBroadcastDynamicId );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan12Properties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan12Properties const & physicalDeviceVulkan12Properties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.driverID );
      for ( size_t i = 0; i < VK_MAX_DRIVER_NAME_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.driverName[i] );
      }
      for ( size_t i = 0; i < VK_MAX_DRIVER_INFO_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.driverInfo[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.conformanceVersion );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.denormBehaviorIndependence );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.roundingModeIndependence );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.shaderSignedZeroInfNanPreserveFloat16 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.shaderSignedZeroInfNanPreserveFloat32 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.shaderSignedZeroInfNanPreserveFloat64 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.shaderDenormPreserveFloat16 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.shaderDenormPreserveFloat32 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.shaderDenormPreserveFloat64 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.shaderDenormFlushToZeroFloat16 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.shaderDenormFlushToZeroFloat32 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.shaderDenormFlushToZeroFloat64 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.shaderRoundingModeRTEFloat16 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.shaderRoundingModeRTEFloat32 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.shaderRoundingModeRTEFloat64 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.shaderRoundingModeRTZFloat16 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.shaderRoundingModeRTZFloat32 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.shaderRoundingModeRTZFloat64 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.maxUpdateAfterBindDescriptorsInAllPools );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.shaderUniformBufferArrayNonUniformIndexingNative );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.shaderSampledImageArrayNonUniformIndexingNative );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.shaderStorageBufferArrayNonUniformIndexingNative );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.shaderStorageImageArrayNonUniformIndexingNative );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.shaderInputAttachmentArrayNonUniformIndexingNative );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.robustBufferAccessUpdateAfterBind );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.quadDivergentImplicitLod );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.maxPerStageDescriptorUpdateAfterBindSamplers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.maxPerStageDescriptorUpdateAfterBindUniformBuffers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.maxPerStageDescriptorUpdateAfterBindStorageBuffers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.maxPerStageDescriptorUpdateAfterBindSampledImages );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.maxPerStageDescriptorUpdateAfterBindStorageImages );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.maxPerStageDescriptorUpdateAfterBindInputAttachments );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.maxPerStageUpdateAfterBindResources );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.maxDescriptorSetUpdateAfterBindSamplers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.maxDescriptorSetUpdateAfterBindUniformBuffers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.maxDescriptorSetUpdateAfterBindUniformBuffersDynamic );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.maxDescriptorSetUpdateAfterBindStorageBuffers );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.maxDescriptorSetUpdateAfterBindStorageBuffersDynamic );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.maxDescriptorSetUpdateAfterBindSampledImages );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.maxDescriptorSetUpdateAfterBindStorageImages );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.maxDescriptorSetUpdateAfterBindInputAttachments );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.supportedDepthResolveModes );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.supportedStencilResolveModes );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.independentResolveNone );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.independentResolve );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.filterMinmaxSingleComponentFormats );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.filterMinmaxImageComponentMapping );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.maxTimelineSemaphoreValueDifference );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan12Properties.framebufferIntegerColorSampleCounts );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan13Features>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan13Features const & physicalDeviceVulkan13Features ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Features.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Features.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Features.robustImageAccess );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Features.inlineUniformBlock );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Features.descriptorBindingInlineUniformBlockUpdateAfterBind );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Features.pipelineCreationCacheControl );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Features.privateData );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Features.shaderDemoteToHelperInvocation );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Features.shaderTerminateInvocation );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Features.subgroupSizeControl );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Features.computeFullSubgroups );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Features.synchronization2 );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Features.textureCompressionASTC_HDR );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Features.shaderZeroInitializeWorkgroupMemory );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Features.dynamicRendering );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Features.shaderIntegerDotProduct );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Features.maintenance4 );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan13Properties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkan13Properties const & physicalDeviceVulkan13Properties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.minSubgroupSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.maxSubgroupSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.maxComputeWorkgroupSubgroups );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.requiredSubgroupSizeStages );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.maxInlineUniformBlockSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.maxPerStageDescriptorInlineUniformBlocks );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.maxPerStageDescriptorUpdateAfterBindInlineUniformBlocks );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.maxDescriptorSetInlineUniformBlocks );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.maxDescriptorSetUpdateAfterBindInlineUniformBlocks );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.maxInlineUniformTotalSize );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProduct8BitUnsignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProduct8BitSignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProduct8BitMixedSignednessAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProduct4x8BitPackedUnsignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProduct4x8BitPackedSignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProduct4x8BitPackedMixedSignednessAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProduct16BitUnsignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProduct16BitSignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProduct16BitMixedSignednessAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProduct32BitUnsignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProduct32BitSignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProduct32BitMixedSignednessAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProduct64BitUnsignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProduct64BitSignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProduct64BitMixedSignednessAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProductAccumulatingSaturating8BitUnsignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProductAccumulatingSaturating8BitSignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProductAccumulatingSaturating16BitUnsignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProductAccumulatingSaturating16BitSignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProductAccumulatingSaturating32BitUnsignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProductAccumulatingSaturating32BitSignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProductAccumulatingSaturating64BitUnsignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProductAccumulatingSaturating64BitSignedAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.storageTexelBufferOffsetAlignmentBytes );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.storageTexelBufferOffsetSingleTexelAlignment );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.uniformTexelBufferOffsetAlignmentBytes );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.uniformTexelBufferOffsetSingleTexelAlignment );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkan13Properties.maxBufferSize );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkanMemoryModelFeatures>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceVulkanMemoryModelFeatures const & physicalDeviceVulkanMemoryModelFeatures ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkanMemoryModelFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkanMemoryModelFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkanMemoryModelFeatures.vulkanMemoryModel );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkanMemoryModelFeatures.vulkanMemoryModelDeviceScope );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceVulkanMemoryModelFeatures.vulkanMemoryModelAvailabilityVisibilityChains );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR const &
                              physicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR.workgroupMemoryExplicitLayout );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR.workgroupMemoryExplicitLayoutScalarBlockLayout );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR.workgroupMemoryExplicitLayout8BitAccess );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR.workgroupMemoryExplicitLayout16BitAccess );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT const & physicalDeviceYcbcr2Plane444FormatsFeaturesEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceYcbcr2Plane444FormatsFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceYcbcr2Plane444FormatsFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceYcbcr2Plane444FormatsFeaturesEXT.ycbcr2plane444Formats );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceYcbcrImageArraysFeaturesEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceYcbcrImageArraysFeaturesEXT const & physicalDeviceYcbcrImageArraysFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceYcbcrImageArraysFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceYcbcrImageArraysFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceYcbcrImageArraysFeaturesEXT.ycbcrImageArrays );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures const & physicalDeviceZeroInitializeWorkgroupMemoryFeatures ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceZeroInitializeWorkgroupMemoryFeatures.sType );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceZeroInitializeWorkgroupMemoryFeatures.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, physicalDeviceZeroInitializeWorkgroupMemoryFeatures.shaderZeroInitializeWorkgroupMemory );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineCacheCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineCacheCreateInfo const & pipelineCacheCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCacheCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCacheCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCacheCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCacheCreateInfo.initialDataSize );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCacheCreateInfo.pInitialData );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineCacheHeaderVersionOne>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineCacheHeaderVersionOne const & pipelineCacheHeaderVersionOne ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCacheHeaderVersionOne.headerSize );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCacheHeaderVersionOne.headerVersion );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCacheHeaderVersionOne.vendorID );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCacheHeaderVersionOne.deviceID );
      for ( size_t i = 0; i < VK_UUID_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, pipelineCacheHeaderVersionOne.pipelineCacheUUID[i] );
      }
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineColorBlendAdvancedStateCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineColorBlendAdvancedStateCreateInfoEXT const & pipelineColorBlendAdvancedStateCreateInfoEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineColorBlendAdvancedStateCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineColorBlendAdvancedStateCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineColorBlendAdvancedStateCreateInfoEXT.srcPremultiplied );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineColorBlendAdvancedStateCreateInfoEXT.dstPremultiplied );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineColorBlendAdvancedStateCreateInfoEXT.blendOverlap );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineColorWriteCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineColorWriteCreateInfoEXT const & pipelineColorWriteCreateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineColorWriteCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineColorWriteCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineColorWriteCreateInfoEXT.attachmentCount );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineColorWriteCreateInfoEXT.pColorWriteEnables );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineCompilerControlCreateInfoAMD>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineCompilerControlCreateInfoAMD const & pipelineCompilerControlCreateInfoAMD ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCompilerControlCreateInfoAMD.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCompilerControlCreateInfoAMD.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCompilerControlCreateInfoAMD.compilerControlFlags );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineCoverageModulationStateCreateInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineCoverageModulationStateCreateInfoNV const & pipelineCoverageModulationStateCreateInfoNV ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCoverageModulationStateCreateInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCoverageModulationStateCreateInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCoverageModulationStateCreateInfoNV.flags );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCoverageModulationStateCreateInfoNV.coverageModulationMode );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCoverageModulationStateCreateInfoNV.coverageModulationTableEnable );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCoverageModulationStateCreateInfoNV.coverageModulationTableCount );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCoverageModulationStateCreateInfoNV.pCoverageModulationTable );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineCoverageReductionStateCreateInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineCoverageReductionStateCreateInfoNV const & pipelineCoverageReductionStateCreateInfoNV ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCoverageReductionStateCreateInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCoverageReductionStateCreateInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCoverageReductionStateCreateInfoNV.flags );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCoverageReductionStateCreateInfoNV.coverageReductionMode );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineCoverageToColorStateCreateInfoNV>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PipelineCoverageToColorStateCreateInfoNV const & pipelineCoverageToColorStateCreateInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCoverageToColorStateCreateInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCoverageToColorStateCreateInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCoverageToColorStateCreateInfoNV.flags );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCoverageToColorStateCreateInfoNV.coverageToColorEnable );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCoverageToColorStateCreateInfoNV.coverageToColorLocation );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineCreationFeedback>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineCreationFeedback const & pipelineCreationFeedback ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCreationFeedback.flags );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCreationFeedback.duration );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineCreationFeedbackCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineCreationFeedbackCreateInfo const & pipelineCreationFeedbackCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCreationFeedbackCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCreationFeedbackCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCreationFeedbackCreateInfo.pPipelineCreationFeedback );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCreationFeedbackCreateInfo.pipelineStageCreationFeedbackCount );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineCreationFeedbackCreateInfo.pPipelineStageCreationFeedbacks );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineDiscardRectangleStateCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineDiscardRectangleStateCreateInfoEXT const & pipelineDiscardRectangleStateCreateInfoEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineDiscardRectangleStateCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineDiscardRectangleStateCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineDiscardRectangleStateCreateInfoEXT.flags );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineDiscardRectangleStateCreateInfoEXT.discardRectangleMode );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineDiscardRectangleStateCreateInfoEXT.discardRectangleCount );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineDiscardRectangleStateCreateInfoEXT.pDiscardRectangles );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineExecutableInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineExecutableInfoKHR const & pipelineExecutableInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineExecutableInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineExecutableInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineExecutableInfoKHR.pipeline );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineExecutableInfoKHR.executableIndex );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineExecutableInternalRepresentationKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineExecutableInternalRepresentationKHR const & pipelineExecutableInternalRepresentationKHR ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineExecutableInternalRepresentationKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineExecutableInternalRepresentationKHR.pNext );
      for ( size_t i = 0; i < VK_MAX_DESCRIPTION_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, pipelineExecutableInternalRepresentationKHR.name[i] );
      }
      for ( size_t i = 0; i < VK_MAX_DESCRIPTION_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, pipelineExecutableInternalRepresentationKHR.description[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, pipelineExecutableInternalRepresentationKHR.isText );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineExecutableInternalRepresentationKHR.dataSize );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineExecutableInternalRepresentationKHR.pData );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineExecutablePropertiesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineExecutablePropertiesKHR const & pipelineExecutablePropertiesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineExecutablePropertiesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineExecutablePropertiesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineExecutablePropertiesKHR.stages );
      for ( size_t i = 0; i < VK_MAX_DESCRIPTION_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, pipelineExecutablePropertiesKHR.name[i] );
      }
      for ( size_t i = 0; i < VK_MAX_DESCRIPTION_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, pipelineExecutablePropertiesKHR.description[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, pipelineExecutablePropertiesKHR.subgroupSize );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineFragmentShadingRateEnumStateCreateInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineFragmentShadingRateEnumStateCreateInfoNV const & pipelineFragmentShadingRateEnumStateCreateInfoNV )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineFragmentShadingRateEnumStateCreateInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineFragmentShadingRateEnumStateCreateInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineFragmentShadingRateEnumStateCreateInfoNV.shadingRateType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineFragmentShadingRateEnumStateCreateInfoNV.shadingRate );
      for ( size_t i = 0; i < 2; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, pipelineFragmentShadingRateEnumStateCreateInfoNV.combinerOps[i] );
      }
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineFragmentShadingRateStateCreateInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineFragmentShadingRateStateCreateInfoKHR const & pipelineFragmentShadingRateStateCreateInfoKHR ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineFragmentShadingRateStateCreateInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineFragmentShadingRateStateCreateInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineFragmentShadingRateStateCreateInfoKHR.fragmentSize );
      for ( size_t i = 0; i < 2; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, pipelineFragmentShadingRateStateCreateInfoKHR.combinerOps[i] );
      }
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineInfoKHR const & pipelineInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineInfoKHR.pipeline );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PushConstantRange>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PushConstantRange const & pushConstantRange ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pushConstantRange.stageFlags );
      VULKAN_HPP_HASH_COMBINE( seed, pushConstantRange.offset );
      VULKAN_HPP_HASH_COMBINE( seed, pushConstantRange.size );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineLayoutCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineLayoutCreateInfo const & pipelineLayoutCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineLayoutCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineLayoutCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineLayoutCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineLayoutCreateInfo.setLayoutCount );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineLayoutCreateInfo.pSetLayouts );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineLayoutCreateInfo.pushConstantRangeCount );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineLayoutCreateInfo.pPushConstantRanges );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineLibraryCreateInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineLibraryCreateInfoKHR const & pipelineLibraryCreateInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineLibraryCreateInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineLibraryCreateInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineLibraryCreateInfoKHR.libraryCount );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineLibraryCreateInfoKHR.pLibraries );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelinePropertiesIdentifierEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelinePropertiesIdentifierEXT const & pipelinePropertiesIdentifierEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelinePropertiesIdentifierEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelinePropertiesIdentifierEXT.pNext );
      for ( size_t i = 0; i < VK_UUID_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, pipelinePropertiesIdentifierEXT.pipelineIdentifier[i] );
      }
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineRasterizationConservativeStateCreateInfoEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PipelineRasterizationConservativeStateCreateInfoEXT const & pipelineRasterizationConservativeStateCreateInfoEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationConservativeStateCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationConservativeStateCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationConservativeStateCreateInfoEXT.flags );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationConservativeStateCreateInfoEXT.conservativeRasterizationMode );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationConservativeStateCreateInfoEXT.extraPrimitiveOverestimationSize );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineRasterizationDepthClipStateCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineRasterizationDepthClipStateCreateInfoEXT const & pipelineRasterizationDepthClipStateCreateInfoEXT )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationDepthClipStateCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationDepthClipStateCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationDepthClipStateCreateInfoEXT.flags );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationDepthClipStateCreateInfoEXT.depthClipEnable );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineRasterizationLineStateCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineRasterizationLineStateCreateInfoEXT const & pipelineRasterizationLineStateCreateInfoEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationLineStateCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationLineStateCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationLineStateCreateInfoEXT.lineRasterizationMode );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationLineStateCreateInfoEXT.stippledLineEnable );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationLineStateCreateInfoEXT.lineStippleFactor );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationLineStateCreateInfoEXT.lineStipplePattern );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineRasterizationProvokingVertexStateCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineRasterizationProvokingVertexStateCreateInfoEXT const &
                              pipelineRasterizationProvokingVertexStateCreateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationProvokingVertexStateCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationProvokingVertexStateCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationProvokingVertexStateCreateInfoEXT.provokingVertexMode );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineRasterizationStateRasterizationOrderAMD>
  {
    std::size_t operator()(
      VULKAN_HPP_NAMESPACE::PipelineRasterizationStateRasterizationOrderAMD const & pipelineRasterizationStateRasterizationOrderAMD ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationStateRasterizationOrderAMD.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationStateRasterizationOrderAMD.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationStateRasterizationOrderAMD.rasterizationOrder );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineRasterizationStateStreamCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineRasterizationStateStreamCreateInfoEXT const & pipelineRasterizationStateStreamCreateInfoEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationStateStreamCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationStateStreamCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationStateStreamCreateInfoEXT.flags );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRasterizationStateStreamCreateInfoEXT.rasterizationStream );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineRenderingCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineRenderingCreateInfo const & pipelineRenderingCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRenderingCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRenderingCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRenderingCreateInfo.viewMask );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRenderingCreateInfo.colorAttachmentCount );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRenderingCreateInfo.pColorAttachmentFormats );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRenderingCreateInfo.depthAttachmentFormat );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRenderingCreateInfo.stencilAttachmentFormat );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineRepresentativeFragmentTestStateCreateInfoNV>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PipelineRepresentativeFragmentTestStateCreateInfoNV const & pipelineRepresentativeFragmentTestStateCreateInfoNV ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRepresentativeFragmentTestStateCreateInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRepresentativeFragmentTestStateCreateInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRepresentativeFragmentTestStateCreateInfoNV.representativeFragmentTestEnable );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineRobustnessCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineRobustnessCreateInfoEXT const & pipelineRobustnessCreateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRobustnessCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRobustnessCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRobustnessCreateInfoEXT.storageBuffers );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRobustnessCreateInfoEXT.uniformBuffers );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRobustnessCreateInfoEXT.vertexInputs );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineRobustnessCreateInfoEXT.images );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineSampleLocationsStateCreateInfoEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PipelineSampleLocationsStateCreateInfoEXT const & pipelineSampleLocationsStateCreateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineSampleLocationsStateCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineSampleLocationsStateCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineSampleLocationsStateCreateInfoEXT.sampleLocationsEnable );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineSampleLocationsStateCreateInfoEXT.sampleLocationsInfo );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineShaderStageModuleIdentifierCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineShaderStageModuleIdentifierCreateInfoEXT const & pipelineShaderStageModuleIdentifierCreateInfoEXT )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineShaderStageModuleIdentifierCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineShaderStageModuleIdentifierCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineShaderStageModuleIdentifierCreateInfoEXT.identifierSize );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineShaderStageModuleIdentifierCreateInfoEXT.pIdentifier );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineShaderStageRequiredSubgroupSizeCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineShaderStageRequiredSubgroupSizeCreateInfo const & pipelineShaderStageRequiredSubgroupSizeCreateInfo )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineShaderStageRequiredSubgroupSizeCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineShaderStageRequiredSubgroupSizeCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineShaderStageRequiredSubgroupSizeCreateInfo.requiredSubgroupSize );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineTessellationDomainOriginStateCreateInfo>
  {
    std::size_t operator()(
      VULKAN_HPP_NAMESPACE::PipelineTessellationDomainOriginStateCreateInfo const & pipelineTessellationDomainOriginStateCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineTessellationDomainOriginStateCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineTessellationDomainOriginStateCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineTessellationDomainOriginStateCreateInfo.domainOrigin );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VertexInputBindingDivisorDescriptionEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::VertexInputBindingDivisorDescriptionEXT const & vertexInputBindingDivisorDescriptionEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, vertexInputBindingDivisorDescriptionEXT.binding );
      VULKAN_HPP_HASH_COMBINE( seed, vertexInputBindingDivisorDescriptionEXT.divisor );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineVertexInputDivisorStateCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineVertexInputDivisorStateCreateInfoEXT const & pipelineVertexInputDivisorStateCreateInfoEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineVertexInputDivisorStateCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineVertexInputDivisorStateCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineVertexInputDivisorStateCreateInfoEXT.vertexBindingDivisorCount );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineVertexInputDivisorStateCreateInfoEXT.pVertexBindingDivisors );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineViewportCoarseSampleOrderStateCreateInfoNV>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PipelineViewportCoarseSampleOrderStateCreateInfoNV const & pipelineViewportCoarseSampleOrderStateCreateInfoNV ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportCoarseSampleOrderStateCreateInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportCoarseSampleOrderStateCreateInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportCoarseSampleOrderStateCreateInfoNV.sampleOrderType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportCoarseSampleOrderStateCreateInfoNV.customSampleOrderCount );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportCoarseSampleOrderStateCreateInfoNV.pCustomSampleOrders );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineViewportDepthClipControlCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineViewportDepthClipControlCreateInfoEXT const & pipelineViewportDepthClipControlCreateInfoEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportDepthClipControlCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportDepthClipControlCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportDepthClipControlCreateInfoEXT.negativeOneToOne );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineViewportExclusiveScissorStateCreateInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineViewportExclusiveScissorStateCreateInfoNV const & pipelineViewportExclusiveScissorStateCreateInfoNV )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportExclusiveScissorStateCreateInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportExclusiveScissorStateCreateInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportExclusiveScissorStateCreateInfoNV.exclusiveScissorCount );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportExclusiveScissorStateCreateInfoNV.pExclusiveScissors );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ShadingRatePaletteNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ShadingRatePaletteNV const & shadingRatePaletteNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, shadingRatePaletteNV.shadingRatePaletteEntryCount );
      VULKAN_HPP_HASH_COMBINE( seed, shadingRatePaletteNV.pShadingRatePaletteEntries );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineViewportShadingRateImageStateCreateInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PipelineViewportShadingRateImageStateCreateInfoNV const & pipelineViewportShadingRateImageStateCreateInfoNV )
      const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportShadingRateImageStateCreateInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportShadingRateImageStateCreateInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportShadingRateImageStateCreateInfoNV.shadingRateImageEnable );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportShadingRateImageStateCreateInfoNV.viewportCount );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportShadingRateImageStateCreateInfoNV.pShadingRatePalettes );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ViewportSwizzleNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ViewportSwizzleNV const & viewportSwizzleNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, viewportSwizzleNV.x );
      VULKAN_HPP_HASH_COMBINE( seed, viewportSwizzleNV.y );
      VULKAN_HPP_HASH_COMBINE( seed, viewportSwizzleNV.z );
      VULKAN_HPP_HASH_COMBINE( seed, viewportSwizzleNV.w );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineViewportSwizzleStateCreateInfoNV>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PipelineViewportSwizzleStateCreateInfoNV const & pipelineViewportSwizzleStateCreateInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportSwizzleStateCreateInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportSwizzleStateCreateInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportSwizzleStateCreateInfoNV.flags );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportSwizzleStateCreateInfoNV.viewportCount );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportSwizzleStateCreateInfoNV.pViewportSwizzles );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ViewportWScalingNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ViewportWScalingNV const & viewportWScalingNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, viewportWScalingNV.xcoeff );
      VULKAN_HPP_HASH_COMBINE( seed, viewportWScalingNV.ycoeff );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PipelineViewportWScalingStateCreateInfoNV>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::PipelineViewportWScalingStateCreateInfoNV const & pipelineViewportWScalingStateCreateInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportWScalingStateCreateInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportWScalingStateCreateInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportWScalingStateCreateInfoNV.viewportWScalingEnable );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportWScalingStateCreateInfoNV.viewportCount );
      VULKAN_HPP_HASH_COMBINE( seed, pipelineViewportWScalingStateCreateInfoNV.pViewportWScalings );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_GGP )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PresentFrameTokenGGP>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PresentFrameTokenGGP const & presentFrameTokenGGP ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, presentFrameTokenGGP.sType );
      VULKAN_HPP_HASH_COMBINE( seed, presentFrameTokenGGP.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, presentFrameTokenGGP.frameToken );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_GGP*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PresentIdKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PresentIdKHR const & presentIdKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, presentIdKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, presentIdKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, presentIdKHR.swapchainCount );
      VULKAN_HPP_HASH_COMBINE( seed, presentIdKHR.pPresentIds );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PresentInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PresentInfoKHR const & presentInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, presentInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, presentInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, presentInfoKHR.waitSemaphoreCount );
      VULKAN_HPP_HASH_COMBINE( seed, presentInfoKHR.pWaitSemaphores );
      VULKAN_HPP_HASH_COMBINE( seed, presentInfoKHR.swapchainCount );
      VULKAN_HPP_HASH_COMBINE( seed, presentInfoKHR.pSwapchains );
      VULKAN_HPP_HASH_COMBINE( seed, presentInfoKHR.pImageIndices );
      VULKAN_HPP_HASH_COMBINE( seed, presentInfoKHR.pResults );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RectLayerKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::RectLayerKHR const & rectLayerKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, rectLayerKHR.offset );
      VULKAN_HPP_HASH_COMBINE( seed, rectLayerKHR.extent );
      VULKAN_HPP_HASH_COMBINE( seed, rectLayerKHR.layer );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PresentRegionKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PresentRegionKHR const & presentRegionKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, presentRegionKHR.rectangleCount );
      VULKAN_HPP_HASH_COMBINE( seed, presentRegionKHR.pRectangles );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PresentRegionsKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PresentRegionsKHR const & presentRegionsKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, presentRegionsKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, presentRegionsKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, presentRegionsKHR.swapchainCount );
      VULKAN_HPP_HASH_COMBINE( seed, presentRegionsKHR.pRegions );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PresentTimeGOOGLE>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PresentTimeGOOGLE const & presentTimeGOOGLE ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, presentTimeGOOGLE.presentID );
      VULKAN_HPP_HASH_COMBINE( seed, presentTimeGOOGLE.desiredPresentTime );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PresentTimesInfoGOOGLE>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PresentTimesInfoGOOGLE const & presentTimesInfoGOOGLE ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, presentTimesInfoGOOGLE.sType );
      VULKAN_HPP_HASH_COMBINE( seed, presentTimesInfoGOOGLE.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, presentTimesInfoGOOGLE.swapchainCount );
      VULKAN_HPP_HASH_COMBINE( seed, presentTimesInfoGOOGLE.pTimes );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::PrivateDataSlotCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::PrivateDataSlotCreateInfo const & privateDataSlotCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, privateDataSlotCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, privateDataSlotCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, privateDataSlotCreateInfo.flags );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ProtectedSubmitInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ProtectedSubmitInfo const & protectedSubmitInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, protectedSubmitInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, protectedSubmitInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, protectedSubmitInfo.protectedSubmit );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueryPoolCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::QueryPoolCreateInfo const & queryPoolCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, queryPoolCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, queryPoolCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, queryPoolCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, queryPoolCreateInfo.queryType );
      VULKAN_HPP_HASH_COMBINE( seed, queryPoolCreateInfo.queryCount );
      VULKAN_HPP_HASH_COMBINE( seed, queryPoolCreateInfo.pipelineStatistics );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueryPoolPerformanceCreateInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::QueryPoolPerformanceCreateInfoKHR const & queryPoolPerformanceCreateInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, queryPoolPerformanceCreateInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, queryPoolPerformanceCreateInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, queryPoolPerformanceCreateInfoKHR.queueFamilyIndex );
      VULKAN_HPP_HASH_COMBINE( seed, queryPoolPerformanceCreateInfoKHR.counterIndexCount );
      VULKAN_HPP_HASH_COMBINE( seed, queryPoolPerformanceCreateInfoKHR.pCounterIndices );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueryPoolPerformanceQueryCreateInfoINTEL>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::QueryPoolPerformanceQueryCreateInfoINTEL const & queryPoolPerformanceQueryCreateInfoINTEL ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, queryPoolPerformanceQueryCreateInfoINTEL.sType );
      VULKAN_HPP_HASH_COMBINE( seed, queryPoolPerformanceQueryCreateInfoINTEL.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, queryPoolPerformanceQueryCreateInfoINTEL.performanceCountersSampling );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueueFamilyCheckpointProperties2NV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::QueueFamilyCheckpointProperties2NV const & queueFamilyCheckpointProperties2NV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, queueFamilyCheckpointProperties2NV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, queueFamilyCheckpointProperties2NV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, queueFamilyCheckpointProperties2NV.checkpointExecutionStageMask );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueueFamilyCheckpointPropertiesNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::QueueFamilyCheckpointPropertiesNV const & queueFamilyCheckpointPropertiesNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, queueFamilyCheckpointPropertiesNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, queueFamilyCheckpointPropertiesNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, queueFamilyCheckpointPropertiesNV.checkpointExecutionStageMask );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueueFamilyGlobalPriorityPropertiesKHR>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::QueueFamilyGlobalPriorityPropertiesKHR const & queueFamilyGlobalPriorityPropertiesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, queueFamilyGlobalPriorityPropertiesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, queueFamilyGlobalPriorityPropertiesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, queueFamilyGlobalPriorityPropertiesKHR.priorityCount );
      for ( size_t i = 0; i < VK_MAX_GLOBAL_PRIORITY_SIZE_KHR; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, queueFamilyGlobalPriorityPropertiesKHR.priorities[i] );
      }
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueueFamilyProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::QueueFamilyProperties const & queueFamilyProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, queueFamilyProperties.queueFlags );
      VULKAN_HPP_HASH_COMBINE( seed, queueFamilyProperties.queueCount );
      VULKAN_HPP_HASH_COMBINE( seed, queueFamilyProperties.timestampValidBits );
      VULKAN_HPP_HASH_COMBINE( seed, queueFamilyProperties.minImageTransferGranularity );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueueFamilyProperties2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::QueueFamilyProperties2 const & queueFamilyProperties2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, queueFamilyProperties2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, queueFamilyProperties2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, queueFamilyProperties2.queueFamilyProperties );
      return seed;
    }
  };

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueueFamilyQueryResultStatusPropertiesKHR>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::QueueFamilyQueryResultStatusPropertiesKHR const & queueFamilyQueryResultStatusPropertiesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, queueFamilyQueryResultStatusPropertiesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, queueFamilyQueryResultStatusPropertiesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, queueFamilyQueryResultStatusPropertiesKHR.queryResultStatusSupport );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::QueueFamilyVideoPropertiesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::QueueFamilyVideoPropertiesKHR const & queueFamilyVideoPropertiesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, queueFamilyVideoPropertiesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, queueFamilyVideoPropertiesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, queueFamilyVideoPropertiesKHR.videoCodecOperations );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RayTracingShaderGroupCreateInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::RayTracingShaderGroupCreateInfoKHR const & rayTracingShaderGroupCreateInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingShaderGroupCreateInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingShaderGroupCreateInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingShaderGroupCreateInfoKHR.type );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingShaderGroupCreateInfoKHR.generalShader );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingShaderGroupCreateInfoKHR.closestHitShader );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingShaderGroupCreateInfoKHR.anyHitShader );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingShaderGroupCreateInfoKHR.intersectionShader );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingShaderGroupCreateInfoKHR.pShaderGroupCaptureReplayHandle );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RayTracingPipelineInterfaceCreateInfoKHR>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::RayTracingPipelineInterfaceCreateInfoKHR const & rayTracingPipelineInterfaceCreateInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineInterfaceCreateInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineInterfaceCreateInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineInterfaceCreateInfoKHR.maxPipelineRayPayloadSize );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineInterfaceCreateInfoKHR.maxPipelineRayHitAttributeSize );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RayTracingPipelineCreateInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::RayTracingPipelineCreateInfoKHR const & rayTracingPipelineCreateInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoKHR.flags );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoKHR.stageCount );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoKHR.pStages );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoKHR.groupCount );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoKHR.pGroups );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoKHR.maxPipelineRayRecursionDepth );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoKHR.pLibraryInfo );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoKHR.pLibraryInterface );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoKHR.pDynamicState );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoKHR.layout );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoKHR.basePipelineHandle );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoKHR.basePipelineIndex );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RayTracingShaderGroupCreateInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::RayTracingShaderGroupCreateInfoNV const & rayTracingShaderGroupCreateInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingShaderGroupCreateInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingShaderGroupCreateInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingShaderGroupCreateInfoNV.type );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingShaderGroupCreateInfoNV.generalShader );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingShaderGroupCreateInfoNV.closestHitShader );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingShaderGroupCreateInfoNV.anyHitShader );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingShaderGroupCreateInfoNV.intersectionShader );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RayTracingPipelineCreateInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::RayTracingPipelineCreateInfoNV const & rayTracingPipelineCreateInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoNV.flags );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoNV.stageCount );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoNV.pStages );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoNV.groupCount );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoNV.pGroups );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoNV.maxRecursionDepth );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoNV.layout );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoNV.basePipelineHandle );
      VULKAN_HPP_HASH_COMBINE( seed, rayTracingPipelineCreateInfoNV.basePipelineIndex );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RefreshCycleDurationGOOGLE>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::RefreshCycleDurationGOOGLE const & refreshCycleDurationGOOGLE ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, refreshCycleDurationGOOGLE.refreshDuration );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassAttachmentBeginInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::RenderPassAttachmentBeginInfo const & renderPassAttachmentBeginInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, renderPassAttachmentBeginInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassAttachmentBeginInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassAttachmentBeginInfo.attachmentCount );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassAttachmentBeginInfo.pAttachments );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassBeginInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::RenderPassBeginInfo const & renderPassBeginInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, renderPassBeginInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassBeginInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassBeginInfo.renderPass );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassBeginInfo.framebuffer );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassBeginInfo.renderArea );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassBeginInfo.clearValueCount );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassBeginInfo.pClearValues );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubpassDescription>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SubpassDescription const & subpassDescription ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescription.flags );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescription.pipelineBindPoint );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescription.inputAttachmentCount );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescription.pInputAttachments );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescription.colorAttachmentCount );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescription.pColorAttachments );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescription.pResolveAttachments );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescription.pDepthStencilAttachment );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescription.preserveAttachmentCount );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescription.pPreserveAttachments );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubpassDependency>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SubpassDependency const & subpassDependency ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, subpassDependency.srcSubpass );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDependency.dstSubpass );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDependency.srcStageMask );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDependency.dstStageMask );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDependency.srcAccessMask );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDependency.dstAccessMask );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDependency.dependencyFlags );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::RenderPassCreateInfo const & renderPassCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreateInfo.attachmentCount );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreateInfo.pAttachments );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreateInfo.subpassCount );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreateInfo.pSubpasses );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreateInfo.dependencyCount );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreateInfo.pDependencies );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubpassDescription2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SubpassDescription2 const & subpassDescription2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescription2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescription2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescription2.flags );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescription2.pipelineBindPoint );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescription2.viewMask );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescription2.inputAttachmentCount );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescription2.pInputAttachments );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescription2.colorAttachmentCount );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescription2.pColorAttachments );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescription2.pResolveAttachments );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescription2.pDepthStencilAttachment );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescription2.preserveAttachmentCount );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescription2.pPreserveAttachments );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubpassDependency2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SubpassDependency2 const & subpassDependency2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, subpassDependency2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDependency2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDependency2.srcSubpass );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDependency2.dstSubpass );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDependency2.srcStageMask );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDependency2.dstStageMask );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDependency2.srcAccessMask );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDependency2.dstAccessMask );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDependency2.dependencyFlags );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDependency2.viewOffset );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassCreateInfo2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::RenderPassCreateInfo2 const & renderPassCreateInfo2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreateInfo2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreateInfo2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreateInfo2.flags );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreateInfo2.attachmentCount );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreateInfo2.pAttachments );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreateInfo2.subpassCount );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreateInfo2.pSubpasses );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreateInfo2.dependencyCount );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreateInfo2.pDependencies );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreateInfo2.correlatedViewMaskCount );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreateInfo2.pCorrelatedViewMasks );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassCreationControlEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::RenderPassCreationControlEXT const & renderPassCreationControlEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreationControlEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreationControlEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreationControlEXT.disallowMerging );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassCreationFeedbackInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::RenderPassCreationFeedbackInfoEXT const & renderPassCreationFeedbackInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreationFeedbackInfoEXT.postMergeSubpassCount );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassCreationFeedbackCreateInfoEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::RenderPassCreationFeedbackCreateInfoEXT const & renderPassCreationFeedbackCreateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreationFeedbackCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreationFeedbackCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassCreationFeedbackCreateInfoEXT.pRenderPassFeedback );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassFragmentDensityMapCreateInfoEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::RenderPassFragmentDensityMapCreateInfoEXT const & renderPassFragmentDensityMapCreateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, renderPassFragmentDensityMapCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassFragmentDensityMapCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassFragmentDensityMapCreateInfoEXT.fragmentDensityMapAttachment );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassInputAttachmentAspectCreateInfo>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::RenderPassInputAttachmentAspectCreateInfo const & renderPassInputAttachmentAspectCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, renderPassInputAttachmentAspectCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassInputAttachmentAspectCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassInputAttachmentAspectCreateInfo.aspectReferenceCount );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassInputAttachmentAspectCreateInfo.pAspectReferences );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassMultiviewCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::RenderPassMultiviewCreateInfo const & renderPassMultiviewCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, renderPassMultiviewCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassMultiviewCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassMultiviewCreateInfo.subpassCount );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassMultiviewCreateInfo.pViewMasks );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassMultiviewCreateInfo.dependencyCount );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassMultiviewCreateInfo.pViewOffsets );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassMultiviewCreateInfo.correlationMaskCount );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassMultiviewCreateInfo.pCorrelationMasks );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubpassSampleLocationsEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SubpassSampleLocationsEXT const & subpassSampleLocationsEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, subpassSampleLocationsEXT.subpassIndex );
      VULKAN_HPP_HASH_COMBINE( seed, subpassSampleLocationsEXT.sampleLocationsInfo );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassSampleLocationsBeginInfoEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::RenderPassSampleLocationsBeginInfoEXT const & renderPassSampleLocationsBeginInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, renderPassSampleLocationsBeginInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassSampleLocationsBeginInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassSampleLocationsBeginInfoEXT.attachmentInitialSampleLocationsCount );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassSampleLocationsBeginInfoEXT.pAttachmentInitialSampleLocations );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassSampleLocationsBeginInfoEXT.postSubpassSampleLocationsCount );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassSampleLocationsBeginInfoEXT.pPostSubpassSampleLocations );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassSubpassFeedbackInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::RenderPassSubpassFeedbackInfoEXT const & renderPassSubpassFeedbackInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, renderPassSubpassFeedbackInfoEXT.subpassMergeStatus );
      for ( size_t i = 0; i < VK_MAX_DESCRIPTION_SIZE; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, renderPassSubpassFeedbackInfoEXT.description[i] );
      }
      VULKAN_HPP_HASH_COMBINE( seed, renderPassSubpassFeedbackInfoEXT.postMergeIndex );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassSubpassFeedbackCreateInfoEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::RenderPassSubpassFeedbackCreateInfoEXT const & renderPassSubpassFeedbackCreateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, renderPassSubpassFeedbackCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassSubpassFeedbackCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassSubpassFeedbackCreateInfoEXT.pSubpassFeedback );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderPassTransformBeginInfoQCOM>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::RenderPassTransformBeginInfoQCOM const & renderPassTransformBeginInfoQCOM ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, renderPassTransformBeginInfoQCOM.sType );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassTransformBeginInfoQCOM.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, renderPassTransformBeginInfoQCOM.transform );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderingFragmentDensityMapAttachmentInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::RenderingFragmentDensityMapAttachmentInfoEXT const & renderingFragmentDensityMapAttachmentInfoEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, renderingFragmentDensityMapAttachmentInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, renderingFragmentDensityMapAttachmentInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, renderingFragmentDensityMapAttachmentInfoEXT.imageView );
      VULKAN_HPP_HASH_COMBINE( seed, renderingFragmentDensityMapAttachmentInfoEXT.imageLayout );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderingFragmentShadingRateAttachmentInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::RenderingFragmentShadingRateAttachmentInfoKHR const & renderingFragmentShadingRateAttachmentInfoKHR ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, renderingFragmentShadingRateAttachmentInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, renderingFragmentShadingRateAttachmentInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, renderingFragmentShadingRateAttachmentInfoKHR.imageView );
      VULKAN_HPP_HASH_COMBINE( seed, renderingFragmentShadingRateAttachmentInfoKHR.imageLayout );
      VULKAN_HPP_HASH_COMBINE( seed, renderingFragmentShadingRateAttachmentInfoKHR.shadingRateAttachmentTexelSize );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::RenderingInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::RenderingInfo const & renderingInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, renderingInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, renderingInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, renderingInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, renderingInfo.renderArea );
      VULKAN_HPP_HASH_COMBINE( seed, renderingInfo.layerCount );
      VULKAN_HPP_HASH_COMBINE( seed, renderingInfo.viewMask );
      VULKAN_HPP_HASH_COMBINE( seed, renderingInfo.colorAttachmentCount );
      VULKAN_HPP_HASH_COMBINE( seed, renderingInfo.pColorAttachments );
      VULKAN_HPP_HASH_COMBINE( seed, renderingInfo.pDepthAttachment );
      VULKAN_HPP_HASH_COMBINE( seed, renderingInfo.pStencilAttachment );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ResolveImageInfo2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ResolveImageInfo2 const & resolveImageInfo2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, resolveImageInfo2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, resolveImageInfo2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, resolveImageInfo2.srcImage );
      VULKAN_HPP_HASH_COMBINE( seed, resolveImageInfo2.srcImageLayout );
      VULKAN_HPP_HASH_COMBINE( seed, resolveImageInfo2.dstImage );
      VULKAN_HPP_HASH_COMBINE( seed, resolveImageInfo2.dstImageLayout );
      VULKAN_HPP_HASH_COMBINE( seed, resolveImageInfo2.regionCount );
      VULKAN_HPP_HASH_COMBINE( seed, resolveImageInfo2.pRegions );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SamplerBorderColorComponentMappingCreateInfoEXT>
  {
    std::size_t operator()(
      VULKAN_HPP_NAMESPACE::SamplerBorderColorComponentMappingCreateInfoEXT const & samplerBorderColorComponentMappingCreateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, samplerBorderColorComponentMappingCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, samplerBorderColorComponentMappingCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, samplerBorderColorComponentMappingCreateInfoEXT.components );
      VULKAN_HPP_HASH_COMBINE( seed, samplerBorderColorComponentMappingCreateInfoEXT.srgb );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SamplerCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SamplerCreateInfo const & samplerCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, samplerCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, samplerCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, samplerCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, samplerCreateInfo.magFilter );
      VULKAN_HPP_HASH_COMBINE( seed, samplerCreateInfo.minFilter );
      VULKAN_HPP_HASH_COMBINE( seed, samplerCreateInfo.mipmapMode );
      VULKAN_HPP_HASH_COMBINE( seed, samplerCreateInfo.addressModeU );
      VULKAN_HPP_HASH_COMBINE( seed, samplerCreateInfo.addressModeV );
      VULKAN_HPP_HASH_COMBINE( seed, samplerCreateInfo.addressModeW );
      VULKAN_HPP_HASH_COMBINE( seed, samplerCreateInfo.mipLodBias );
      VULKAN_HPP_HASH_COMBINE( seed, samplerCreateInfo.anisotropyEnable );
      VULKAN_HPP_HASH_COMBINE( seed, samplerCreateInfo.maxAnisotropy );
      VULKAN_HPP_HASH_COMBINE( seed, samplerCreateInfo.compareEnable );
      VULKAN_HPP_HASH_COMBINE( seed, samplerCreateInfo.compareOp );
      VULKAN_HPP_HASH_COMBINE( seed, samplerCreateInfo.minLod );
      VULKAN_HPP_HASH_COMBINE( seed, samplerCreateInfo.maxLod );
      VULKAN_HPP_HASH_COMBINE( seed, samplerCreateInfo.borderColor );
      VULKAN_HPP_HASH_COMBINE( seed, samplerCreateInfo.unnormalizedCoordinates );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SamplerReductionModeCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SamplerReductionModeCreateInfo const & samplerReductionModeCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, samplerReductionModeCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, samplerReductionModeCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, samplerReductionModeCreateInfo.reductionMode );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionCreateInfo const & samplerYcbcrConversionCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, samplerYcbcrConversionCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, samplerYcbcrConversionCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, samplerYcbcrConversionCreateInfo.format );
      VULKAN_HPP_HASH_COMBINE( seed, samplerYcbcrConversionCreateInfo.ycbcrModel );
      VULKAN_HPP_HASH_COMBINE( seed, samplerYcbcrConversionCreateInfo.ycbcrRange );
      VULKAN_HPP_HASH_COMBINE( seed, samplerYcbcrConversionCreateInfo.components );
      VULKAN_HPP_HASH_COMBINE( seed, samplerYcbcrConversionCreateInfo.xChromaOffset );
      VULKAN_HPP_HASH_COMBINE( seed, samplerYcbcrConversionCreateInfo.yChromaOffset );
      VULKAN_HPP_HASH_COMBINE( seed, samplerYcbcrConversionCreateInfo.chromaFilter );
      VULKAN_HPP_HASH_COMBINE( seed, samplerYcbcrConversionCreateInfo.forceExplicitReconstruction );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionImageFormatProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionImageFormatProperties const & samplerYcbcrConversionImageFormatProperties ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, samplerYcbcrConversionImageFormatProperties.sType );
      VULKAN_HPP_HASH_COMBINE( seed, samplerYcbcrConversionImageFormatProperties.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, samplerYcbcrConversionImageFormatProperties.combinedImageSamplerDescriptorCount );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionInfo const & samplerYcbcrConversionInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, samplerYcbcrConversionInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, samplerYcbcrConversionInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, samplerYcbcrConversionInfo.conversion );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_SCREEN_QNX )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ScreenSurfaceCreateInfoQNX>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ScreenSurfaceCreateInfoQNX const & screenSurfaceCreateInfoQNX ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, screenSurfaceCreateInfoQNX.sType );
      VULKAN_HPP_HASH_COMBINE( seed, screenSurfaceCreateInfoQNX.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, screenSurfaceCreateInfoQNX.flags );
      VULKAN_HPP_HASH_COMBINE( seed, screenSurfaceCreateInfoQNX.context );
      VULKAN_HPP_HASH_COMBINE( seed, screenSurfaceCreateInfoQNX.window );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_SCREEN_QNX*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SemaphoreCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SemaphoreCreateInfo const & semaphoreCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreCreateInfo.flags );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SemaphoreGetFdInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SemaphoreGetFdInfoKHR const & semaphoreGetFdInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreGetFdInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreGetFdInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreGetFdInfoKHR.semaphore );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreGetFdInfoKHR.handleType );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SemaphoreGetWin32HandleInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SemaphoreGetWin32HandleInfoKHR const & semaphoreGetWin32HandleInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreGetWin32HandleInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreGetWin32HandleInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreGetWin32HandleInfoKHR.semaphore );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreGetWin32HandleInfoKHR.handleType );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SemaphoreGetZirconHandleInfoFUCHSIA>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SemaphoreGetZirconHandleInfoFUCHSIA const & semaphoreGetZirconHandleInfoFUCHSIA ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreGetZirconHandleInfoFUCHSIA.sType );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreGetZirconHandleInfoFUCHSIA.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreGetZirconHandleInfoFUCHSIA.semaphore );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreGetZirconHandleInfoFUCHSIA.handleType );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SemaphoreSignalInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SemaphoreSignalInfo const & semaphoreSignalInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreSignalInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreSignalInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreSignalInfo.semaphore );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreSignalInfo.value );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SemaphoreSubmitInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SemaphoreSubmitInfo const & semaphoreSubmitInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreSubmitInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreSubmitInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreSubmitInfo.semaphore );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreSubmitInfo.value );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreSubmitInfo.stageMask );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreSubmitInfo.deviceIndex );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SemaphoreTypeCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SemaphoreTypeCreateInfo const & semaphoreTypeCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreTypeCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreTypeCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreTypeCreateInfo.semaphoreType );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreTypeCreateInfo.initialValue );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SemaphoreWaitInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SemaphoreWaitInfo const & semaphoreWaitInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreWaitInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreWaitInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreWaitInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreWaitInfo.semaphoreCount );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreWaitInfo.pSemaphores );
      VULKAN_HPP_HASH_COMBINE( seed, semaphoreWaitInfo.pValues );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SetStateFlagsIndirectCommandNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SetStateFlagsIndirectCommandNV const & setStateFlagsIndirectCommandNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, setStateFlagsIndirectCommandNV.data );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ShaderModuleCreateInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ShaderModuleCreateInfo const & shaderModuleCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, shaderModuleCreateInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, shaderModuleCreateInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, shaderModuleCreateInfo.flags );
      VULKAN_HPP_HASH_COMBINE( seed, shaderModuleCreateInfo.codeSize );
      VULKAN_HPP_HASH_COMBINE( seed, shaderModuleCreateInfo.pCode );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ShaderModuleIdentifierEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ShaderModuleIdentifierEXT const & shaderModuleIdentifierEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, shaderModuleIdentifierEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, shaderModuleIdentifierEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, shaderModuleIdentifierEXT.identifierSize );
      for ( size_t i = 0; i < VK_MAX_SHADER_MODULE_IDENTIFIER_SIZE_EXT; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, shaderModuleIdentifierEXT.identifier[i] );
      }
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ShaderModuleValidationCacheCreateInfoEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::ShaderModuleValidationCacheCreateInfoEXT const & shaderModuleValidationCacheCreateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, shaderModuleValidationCacheCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, shaderModuleValidationCacheCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, shaderModuleValidationCacheCreateInfoEXT.validationCache );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ShaderResourceUsageAMD>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ShaderResourceUsageAMD const & shaderResourceUsageAMD ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, shaderResourceUsageAMD.numUsedVgprs );
      VULKAN_HPP_HASH_COMBINE( seed, shaderResourceUsageAMD.numUsedSgprs );
      VULKAN_HPP_HASH_COMBINE( seed, shaderResourceUsageAMD.ldsSizePerLocalWorkGroup );
      VULKAN_HPP_HASH_COMBINE( seed, shaderResourceUsageAMD.ldsUsageSizeInBytes );
      VULKAN_HPP_HASH_COMBINE( seed, shaderResourceUsageAMD.scratchMemUsageInBytes );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ShaderStatisticsInfoAMD>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ShaderStatisticsInfoAMD const & shaderStatisticsInfoAMD ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, shaderStatisticsInfoAMD.shaderStageMask );
      VULKAN_HPP_HASH_COMBINE( seed, shaderStatisticsInfoAMD.resourceUsage );
      VULKAN_HPP_HASH_COMBINE( seed, shaderStatisticsInfoAMD.numPhysicalVgprs );
      VULKAN_HPP_HASH_COMBINE( seed, shaderStatisticsInfoAMD.numPhysicalSgprs );
      VULKAN_HPP_HASH_COMBINE( seed, shaderStatisticsInfoAMD.numAvailableVgprs );
      VULKAN_HPP_HASH_COMBINE( seed, shaderStatisticsInfoAMD.numAvailableSgprs );
      for ( size_t i = 0; i < 3; ++i )
      {
        VULKAN_HPP_HASH_COMBINE( seed, shaderStatisticsInfoAMD.computeWorkGroupSize[i] );
      }
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SharedPresentSurfaceCapabilitiesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SharedPresentSurfaceCapabilitiesKHR const & sharedPresentSurfaceCapabilitiesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, sharedPresentSurfaceCapabilitiesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, sharedPresentSurfaceCapabilitiesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, sharedPresentSurfaceCapabilitiesKHR.sharedPresentSupportedUsageFlags );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SparseImageFormatProperties>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SparseImageFormatProperties const & sparseImageFormatProperties ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageFormatProperties.aspectMask );
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageFormatProperties.imageGranularity );
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageFormatProperties.flags );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SparseImageFormatProperties2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SparseImageFormatProperties2 const & sparseImageFormatProperties2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageFormatProperties2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageFormatProperties2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageFormatProperties2.properties );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SparseImageMemoryRequirements>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SparseImageMemoryRequirements const & sparseImageMemoryRequirements ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageMemoryRequirements.formatProperties );
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageMemoryRequirements.imageMipTailFirstLod );
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageMemoryRequirements.imageMipTailSize );
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageMemoryRequirements.imageMipTailOffset );
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageMemoryRequirements.imageMipTailStride );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SparseImageMemoryRequirements2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SparseImageMemoryRequirements2 const & sparseImageMemoryRequirements2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageMemoryRequirements2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageMemoryRequirements2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, sparseImageMemoryRequirements2.memoryRequirements );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_GGP )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::StreamDescriptorSurfaceCreateInfoGGP>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::StreamDescriptorSurfaceCreateInfoGGP const & streamDescriptorSurfaceCreateInfoGGP ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, streamDescriptorSurfaceCreateInfoGGP.sType );
      VULKAN_HPP_HASH_COMBINE( seed, streamDescriptorSurfaceCreateInfoGGP.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, streamDescriptorSurfaceCreateInfoGGP.flags );
      VULKAN_HPP_HASH_COMBINE( seed, streamDescriptorSurfaceCreateInfoGGP.streamDescriptor );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_GGP*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::StridedDeviceAddressRegionKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::StridedDeviceAddressRegionKHR const & stridedDeviceAddressRegionKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, stridedDeviceAddressRegionKHR.deviceAddress );
      VULKAN_HPP_HASH_COMBINE( seed, stridedDeviceAddressRegionKHR.stride );
      VULKAN_HPP_HASH_COMBINE( seed, stridedDeviceAddressRegionKHR.size );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubmitInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SubmitInfo const & submitInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, submitInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, submitInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, submitInfo.waitSemaphoreCount );
      VULKAN_HPP_HASH_COMBINE( seed, submitInfo.pWaitSemaphores );
      VULKAN_HPP_HASH_COMBINE( seed, submitInfo.pWaitDstStageMask );
      VULKAN_HPP_HASH_COMBINE( seed, submitInfo.commandBufferCount );
      VULKAN_HPP_HASH_COMBINE( seed, submitInfo.pCommandBuffers );
      VULKAN_HPP_HASH_COMBINE( seed, submitInfo.signalSemaphoreCount );
      VULKAN_HPP_HASH_COMBINE( seed, submitInfo.pSignalSemaphores );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubmitInfo2>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SubmitInfo2 const & submitInfo2 ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, submitInfo2.sType );
      VULKAN_HPP_HASH_COMBINE( seed, submitInfo2.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, submitInfo2.flags );
      VULKAN_HPP_HASH_COMBINE( seed, submitInfo2.waitSemaphoreInfoCount );
      VULKAN_HPP_HASH_COMBINE( seed, submitInfo2.pWaitSemaphoreInfos );
      VULKAN_HPP_HASH_COMBINE( seed, submitInfo2.commandBufferInfoCount );
      VULKAN_HPP_HASH_COMBINE( seed, submitInfo2.pCommandBufferInfos );
      VULKAN_HPP_HASH_COMBINE( seed, submitInfo2.signalSemaphoreInfoCount );
      VULKAN_HPP_HASH_COMBINE( seed, submitInfo2.pSignalSemaphoreInfos );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubpassBeginInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SubpassBeginInfo const & subpassBeginInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, subpassBeginInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, subpassBeginInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, subpassBeginInfo.contents );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubpassDescriptionDepthStencilResolve>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::SubpassDescriptionDepthStencilResolve const & subpassDescriptionDepthStencilResolve ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescriptionDepthStencilResolve.sType );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescriptionDepthStencilResolve.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescriptionDepthStencilResolve.depthResolveMode );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescriptionDepthStencilResolve.stencilResolveMode );
      VULKAN_HPP_HASH_COMBINE( seed, subpassDescriptionDepthStencilResolve.pDepthStencilResolveAttachment );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubpassEndInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SubpassEndInfo const & subpassEndInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, subpassEndInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, subpassEndInfo.pNext );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubpassFragmentDensityMapOffsetEndInfoQCOM>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SubpassFragmentDensityMapOffsetEndInfoQCOM const & subpassFragmentDensityMapOffsetEndInfoQCOM ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, subpassFragmentDensityMapOffsetEndInfoQCOM.sType );
      VULKAN_HPP_HASH_COMBINE( seed, subpassFragmentDensityMapOffsetEndInfoQCOM.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, subpassFragmentDensityMapOffsetEndInfoQCOM.fragmentDensityOffsetCount );
      VULKAN_HPP_HASH_COMBINE( seed, subpassFragmentDensityMapOffsetEndInfoQCOM.pFragmentDensityOffsets );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubpassResolvePerformanceQueryEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SubpassResolvePerformanceQueryEXT const & subpassResolvePerformanceQueryEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, subpassResolvePerformanceQueryEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, subpassResolvePerformanceQueryEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, subpassResolvePerformanceQueryEXT.optimal );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubpassShadingPipelineCreateInfoHUAWEI>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::SubpassShadingPipelineCreateInfoHUAWEI const & subpassShadingPipelineCreateInfoHUAWEI ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, subpassShadingPipelineCreateInfoHUAWEI.sType );
      VULKAN_HPP_HASH_COMBINE( seed, subpassShadingPipelineCreateInfoHUAWEI.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, subpassShadingPipelineCreateInfoHUAWEI.renderPass );
      VULKAN_HPP_HASH_COMBINE( seed, subpassShadingPipelineCreateInfoHUAWEI.subpass );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SubresourceLayout2EXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SubresourceLayout2EXT const & subresourceLayout2EXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, subresourceLayout2EXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, subresourceLayout2EXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, subresourceLayout2EXT.subresourceLayout );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceCapabilities2EXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SurfaceCapabilities2EXT const & surfaceCapabilities2EXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilities2EXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilities2EXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilities2EXT.minImageCount );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilities2EXT.maxImageCount );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilities2EXT.currentExtent );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilities2EXT.minImageExtent );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilities2EXT.maxImageExtent );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilities2EXT.maxImageArrayLayers );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilities2EXT.supportedTransforms );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilities2EXT.currentTransform );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilities2EXT.supportedCompositeAlpha );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilities2EXT.supportedUsageFlags );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilities2EXT.supportedSurfaceCounters );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesKHR const & surfaceCapabilitiesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilitiesKHR.minImageCount );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilitiesKHR.maxImageCount );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilitiesKHR.currentExtent );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilitiesKHR.minImageExtent );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilitiesKHR.maxImageExtent );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilitiesKHR.maxImageArrayLayers );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilitiesKHR.supportedTransforms );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilitiesKHR.currentTransform );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilitiesKHR.supportedCompositeAlpha );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilitiesKHR.supportedUsageFlags );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceCapabilities2KHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SurfaceCapabilities2KHR const & surfaceCapabilities2KHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilities2KHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilities2KHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilities2KHR.surfaceCapabilities );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesFullScreenExclusiveEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesFullScreenExclusiveEXT const & surfaceCapabilitiesFullScreenExclusiveEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilitiesFullScreenExclusiveEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilitiesFullScreenExclusiveEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilitiesFullScreenExclusiveEXT.fullScreenExclusiveSupported );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesPresentBarrierNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesPresentBarrierNV const & surfaceCapabilitiesPresentBarrierNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilitiesPresentBarrierNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilitiesPresentBarrierNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceCapabilitiesPresentBarrierNV.presentBarrierSupported );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceFormatKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SurfaceFormatKHR const & surfaceFormatKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, surfaceFormatKHR.format );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceFormatKHR.colorSpace );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceFormat2KHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SurfaceFormat2KHR const & surfaceFormat2KHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, surfaceFormat2KHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceFormat2KHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceFormat2KHR.surfaceFormat );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceFullScreenExclusiveInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SurfaceFullScreenExclusiveInfoEXT const & surfaceFullScreenExclusiveInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, surfaceFullScreenExclusiveInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceFullScreenExclusiveInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceFullScreenExclusiveInfoEXT.fullScreenExclusive );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceFullScreenExclusiveWin32InfoEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::SurfaceFullScreenExclusiveWin32InfoEXT const & surfaceFullScreenExclusiveWin32InfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, surfaceFullScreenExclusiveWin32InfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceFullScreenExclusiveWin32InfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceFullScreenExclusiveWin32InfoEXT.hmonitor );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SurfaceProtectedCapabilitiesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SurfaceProtectedCapabilitiesKHR const & surfaceProtectedCapabilitiesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, surfaceProtectedCapabilitiesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceProtectedCapabilitiesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, surfaceProtectedCapabilitiesKHR.supportsProtected );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SwapchainCounterCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SwapchainCounterCreateInfoEXT const & swapchainCounterCreateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, swapchainCounterCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, swapchainCounterCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, swapchainCounterCreateInfoEXT.surfaceCounters );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SwapchainCreateInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SwapchainCreateInfoKHR const & swapchainCreateInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, swapchainCreateInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, swapchainCreateInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, swapchainCreateInfoKHR.flags );
      VULKAN_HPP_HASH_COMBINE( seed, swapchainCreateInfoKHR.surface );
      VULKAN_HPP_HASH_COMBINE( seed, swapchainCreateInfoKHR.minImageCount );
      VULKAN_HPP_HASH_COMBINE( seed, swapchainCreateInfoKHR.imageFormat );
      VULKAN_HPP_HASH_COMBINE( seed, swapchainCreateInfoKHR.imageColorSpace );
      VULKAN_HPP_HASH_COMBINE( seed, swapchainCreateInfoKHR.imageExtent );
      VULKAN_HPP_HASH_COMBINE( seed, swapchainCreateInfoKHR.imageArrayLayers );
      VULKAN_HPP_HASH_COMBINE( seed, swapchainCreateInfoKHR.imageUsage );
      VULKAN_HPP_HASH_COMBINE( seed, swapchainCreateInfoKHR.imageSharingMode );
      VULKAN_HPP_HASH_COMBINE( seed, swapchainCreateInfoKHR.queueFamilyIndexCount );
      VULKAN_HPP_HASH_COMBINE( seed, swapchainCreateInfoKHR.pQueueFamilyIndices );
      VULKAN_HPP_HASH_COMBINE( seed, swapchainCreateInfoKHR.preTransform );
      VULKAN_HPP_HASH_COMBINE( seed, swapchainCreateInfoKHR.compositeAlpha );
      VULKAN_HPP_HASH_COMBINE( seed, swapchainCreateInfoKHR.presentMode );
      VULKAN_HPP_HASH_COMBINE( seed, swapchainCreateInfoKHR.clipped );
      VULKAN_HPP_HASH_COMBINE( seed, swapchainCreateInfoKHR.oldSwapchain );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SwapchainDisplayNativeHdrCreateInfoAMD>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::SwapchainDisplayNativeHdrCreateInfoAMD const & swapchainDisplayNativeHdrCreateInfoAMD ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, swapchainDisplayNativeHdrCreateInfoAMD.sType );
      VULKAN_HPP_HASH_COMBINE( seed, swapchainDisplayNativeHdrCreateInfoAMD.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, swapchainDisplayNativeHdrCreateInfoAMD.localDimmingEnable );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::SwapchainPresentBarrierCreateInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::SwapchainPresentBarrierCreateInfoNV const & swapchainPresentBarrierCreateInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, swapchainPresentBarrierCreateInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, swapchainPresentBarrierCreateInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, swapchainPresentBarrierCreateInfoNV.presentBarrierEnable );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TextureLODGatherFormatPropertiesAMD>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::TextureLODGatherFormatPropertiesAMD const & textureLODGatherFormatPropertiesAMD ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, textureLODGatherFormatPropertiesAMD.sType );
      VULKAN_HPP_HASH_COMBINE( seed, textureLODGatherFormatPropertiesAMD.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, textureLODGatherFormatPropertiesAMD.supportsTextureGatherLODBiasAMD );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TilePropertiesQCOM>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::TilePropertiesQCOM const & tilePropertiesQCOM ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, tilePropertiesQCOM.sType );
      VULKAN_HPP_HASH_COMBINE( seed, tilePropertiesQCOM.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, tilePropertiesQCOM.tileSize );
      VULKAN_HPP_HASH_COMBINE( seed, tilePropertiesQCOM.apronSize );
      VULKAN_HPP_HASH_COMBINE( seed, tilePropertiesQCOM.origin );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TimelineSemaphoreSubmitInfo>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::TimelineSemaphoreSubmitInfo const & timelineSemaphoreSubmitInfo ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, timelineSemaphoreSubmitInfo.sType );
      VULKAN_HPP_HASH_COMBINE( seed, timelineSemaphoreSubmitInfo.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, timelineSemaphoreSubmitInfo.waitSemaphoreValueCount );
      VULKAN_HPP_HASH_COMBINE( seed, timelineSemaphoreSubmitInfo.pWaitSemaphoreValues );
      VULKAN_HPP_HASH_COMBINE( seed, timelineSemaphoreSubmitInfo.signalSemaphoreValueCount );
      VULKAN_HPP_HASH_COMBINE( seed, timelineSemaphoreSubmitInfo.pSignalSemaphoreValues );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TraceRaysIndirectCommand2KHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::TraceRaysIndirectCommand2KHR const & traceRaysIndirectCommand2KHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, traceRaysIndirectCommand2KHR.raygenShaderRecordAddress );
      VULKAN_HPP_HASH_COMBINE( seed, traceRaysIndirectCommand2KHR.raygenShaderRecordSize );
      VULKAN_HPP_HASH_COMBINE( seed, traceRaysIndirectCommand2KHR.missShaderBindingTableAddress );
      VULKAN_HPP_HASH_COMBINE( seed, traceRaysIndirectCommand2KHR.missShaderBindingTableSize );
      VULKAN_HPP_HASH_COMBINE( seed, traceRaysIndirectCommand2KHR.missShaderBindingTableStride );
      VULKAN_HPP_HASH_COMBINE( seed, traceRaysIndirectCommand2KHR.hitShaderBindingTableAddress );
      VULKAN_HPP_HASH_COMBINE( seed, traceRaysIndirectCommand2KHR.hitShaderBindingTableSize );
      VULKAN_HPP_HASH_COMBINE( seed, traceRaysIndirectCommand2KHR.hitShaderBindingTableStride );
      VULKAN_HPP_HASH_COMBINE( seed, traceRaysIndirectCommand2KHR.callableShaderBindingTableAddress );
      VULKAN_HPP_HASH_COMBINE( seed, traceRaysIndirectCommand2KHR.callableShaderBindingTableSize );
      VULKAN_HPP_HASH_COMBINE( seed, traceRaysIndirectCommand2KHR.callableShaderBindingTableStride );
      VULKAN_HPP_HASH_COMBINE( seed, traceRaysIndirectCommand2KHR.width );
      VULKAN_HPP_HASH_COMBINE( seed, traceRaysIndirectCommand2KHR.height );
      VULKAN_HPP_HASH_COMBINE( seed, traceRaysIndirectCommand2KHR.depth );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::TraceRaysIndirectCommandKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::TraceRaysIndirectCommandKHR const & traceRaysIndirectCommandKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, traceRaysIndirectCommandKHR.width );
      VULKAN_HPP_HASH_COMBINE( seed, traceRaysIndirectCommandKHR.height );
      VULKAN_HPP_HASH_COMBINE( seed, traceRaysIndirectCommandKHR.depth );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ValidationCacheCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ValidationCacheCreateInfoEXT const & validationCacheCreateInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, validationCacheCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, validationCacheCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, validationCacheCreateInfoEXT.flags );
      VULKAN_HPP_HASH_COMBINE( seed, validationCacheCreateInfoEXT.initialDataSize );
      VULKAN_HPP_HASH_COMBINE( seed, validationCacheCreateInfoEXT.pInitialData );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ValidationFeaturesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ValidationFeaturesEXT const & validationFeaturesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, validationFeaturesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, validationFeaturesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, validationFeaturesEXT.enabledValidationFeatureCount );
      VULKAN_HPP_HASH_COMBINE( seed, validationFeaturesEXT.pEnabledValidationFeatures );
      VULKAN_HPP_HASH_COMBINE( seed, validationFeaturesEXT.disabledValidationFeatureCount );
      VULKAN_HPP_HASH_COMBINE( seed, validationFeaturesEXT.pDisabledValidationFeatures );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ValidationFlagsEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ValidationFlagsEXT const & validationFlagsEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, validationFlagsEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, validationFlagsEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, validationFlagsEXT.disabledValidationCheckCount );
      VULKAN_HPP_HASH_COMBINE( seed, validationFlagsEXT.pDisabledValidationChecks );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VertexInputAttributeDescription2EXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VertexInputAttributeDescription2EXT const & vertexInputAttributeDescription2EXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, vertexInputAttributeDescription2EXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, vertexInputAttributeDescription2EXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, vertexInputAttributeDescription2EXT.location );
      VULKAN_HPP_HASH_COMBINE( seed, vertexInputAttributeDescription2EXT.binding );
      VULKAN_HPP_HASH_COMBINE( seed, vertexInputAttributeDescription2EXT.format );
      VULKAN_HPP_HASH_COMBINE( seed, vertexInputAttributeDescription2EXT.offset );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VertexInputBindingDescription2EXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VertexInputBindingDescription2EXT const & vertexInputBindingDescription2EXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, vertexInputBindingDescription2EXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, vertexInputBindingDescription2EXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, vertexInputBindingDescription2EXT.binding );
      VULKAN_HPP_HASH_COMBINE( seed, vertexInputBindingDescription2EXT.stride );
      VULKAN_HPP_HASH_COMBINE( seed, vertexInputBindingDescription2EXT.inputRate );
      VULKAN_HPP_HASH_COMBINE( seed, vertexInputBindingDescription2EXT.divisor );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_VI_NN )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::ViSurfaceCreateInfoNN>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::ViSurfaceCreateInfoNN const & viSurfaceCreateInfoNN ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, viSurfaceCreateInfoNN.sType );
      VULKAN_HPP_HASH_COMBINE( seed, viSurfaceCreateInfoNN.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, viSurfaceCreateInfoNN.flags );
      VULKAN_HPP_HASH_COMBINE( seed, viSurfaceCreateInfoNN.window );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_VI_NN*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoPictureResourceInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoPictureResourceInfoKHR const & videoPictureResourceInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoPictureResourceInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoPictureResourceInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoPictureResourceInfoKHR.codedOffset );
      VULKAN_HPP_HASH_COMBINE( seed, videoPictureResourceInfoKHR.codedExtent );
      VULKAN_HPP_HASH_COMBINE( seed, videoPictureResourceInfoKHR.baseArrayLayer );
      VULKAN_HPP_HASH_COMBINE( seed, videoPictureResourceInfoKHR.imageViewBinding );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoReferenceSlotInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoReferenceSlotInfoKHR const & videoReferenceSlotInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoReferenceSlotInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoReferenceSlotInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoReferenceSlotInfoKHR.slotIndex );
      VULKAN_HPP_HASH_COMBINE( seed, videoReferenceSlotInfoKHR.pPictureResource );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoBeginCodingInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoBeginCodingInfoKHR const & videoBeginCodingInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoBeginCodingInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoBeginCodingInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoBeginCodingInfoKHR.flags );
      VULKAN_HPP_HASH_COMBINE( seed, videoBeginCodingInfoKHR.videoSession );
      VULKAN_HPP_HASH_COMBINE( seed, videoBeginCodingInfoKHR.videoSessionParameters );
      VULKAN_HPP_HASH_COMBINE( seed, videoBeginCodingInfoKHR.referenceSlotCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoBeginCodingInfoKHR.pReferenceSlots );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoCapabilitiesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoCapabilitiesKHR const & videoCapabilitiesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoCapabilitiesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoCapabilitiesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoCapabilitiesKHR.flags );
      VULKAN_HPP_HASH_COMBINE( seed, videoCapabilitiesKHR.minBitstreamBufferOffsetAlignment );
      VULKAN_HPP_HASH_COMBINE( seed, videoCapabilitiesKHR.minBitstreamBufferSizeAlignment );
      VULKAN_HPP_HASH_COMBINE( seed, videoCapabilitiesKHR.pictureAccessGranularity );
      VULKAN_HPP_HASH_COMBINE( seed, videoCapabilitiesKHR.minCodedExtent );
      VULKAN_HPP_HASH_COMBINE( seed, videoCapabilitiesKHR.maxCodedExtent );
      VULKAN_HPP_HASH_COMBINE( seed, videoCapabilitiesKHR.maxDpbSlots );
      VULKAN_HPP_HASH_COMBINE( seed, videoCapabilitiesKHR.maxActiveReferencePictures );
      VULKAN_HPP_HASH_COMBINE( seed, videoCapabilitiesKHR.stdHeaderVersion );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoCodingControlInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoCodingControlInfoKHR const & videoCodingControlInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoCodingControlInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoCodingControlInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoCodingControlInfoKHR.flags );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeCapabilitiesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoDecodeCapabilitiesKHR const & videoDecodeCapabilitiesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeCapabilitiesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeCapabilitiesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeCapabilitiesKHR.flags );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH264CapabilitiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoDecodeH264CapabilitiesEXT const & videoDecodeH264CapabilitiesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264CapabilitiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264CapabilitiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264CapabilitiesEXT.maxLevelIdc );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264CapabilitiesEXT.fieldOffsetGranularity );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH264DpbSlotInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoDecodeH264DpbSlotInfoEXT const & videoDecodeH264DpbSlotInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264DpbSlotInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264DpbSlotInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264DpbSlotInfoEXT.pStdReferenceInfo );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH264PictureInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoDecodeH264PictureInfoEXT const & videoDecodeH264PictureInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264PictureInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264PictureInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264PictureInfoEXT.pStdPictureInfo );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264PictureInfoEXT.sliceCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264PictureInfoEXT.pSliceOffsets );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH264ProfileInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoDecodeH264ProfileInfoEXT const & videoDecodeH264ProfileInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264ProfileInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264ProfileInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264ProfileInfoEXT.stdProfileIdc );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264ProfileInfoEXT.pictureLayout );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH264SessionParametersAddInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoDecodeH264SessionParametersAddInfoEXT const & videoDecodeH264SessionParametersAddInfoEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264SessionParametersAddInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264SessionParametersAddInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264SessionParametersAddInfoEXT.stdSPSCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264SessionParametersAddInfoEXT.pStdSPSs );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264SessionParametersAddInfoEXT.stdPPSCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264SessionParametersAddInfoEXT.pStdPPSs );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH264SessionParametersCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoDecodeH264SessionParametersCreateInfoEXT const & videoDecodeH264SessionParametersCreateInfoEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264SessionParametersCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264SessionParametersCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264SessionParametersCreateInfoEXT.maxStdSPSCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264SessionParametersCreateInfoEXT.maxStdPPSCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH264SessionParametersCreateInfoEXT.pParametersAddInfo );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH265CapabilitiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoDecodeH265CapabilitiesEXT const & videoDecodeH265CapabilitiesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265CapabilitiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265CapabilitiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265CapabilitiesEXT.maxLevelIdc );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH265DpbSlotInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoDecodeH265DpbSlotInfoEXT const & videoDecodeH265DpbSlotInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265DpbSlotInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265DpbSlotInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265DpbSlotInfoEXT.pStdReferenceInfo );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH265PictureInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoDecodeH265PictureInfoEXT const & videoDecodeH265PictureInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265PictureInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265PictureInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265PictureInfoEXT.pStdPictureInfo );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265PictureInfoEXT.sliceCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265PictureInfoEXT.pSliceOffsets );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH265ProfileInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoDecodeH265ProfileInfoEXT const & videoDecodeH265ProfileInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265ProfileInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265ProfileInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265ProfileInfoEXT.stdProfileIdc );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH265SessionParametersAddInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoDecodeH265SessionParametersAddInfoEXT const & videoDecodeH265SessionParametersAddInfoEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265SessionParametersAddInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265SessionParametersAddInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265SessionParametersAddInfoEXT.stdVPSCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265SessionParametersAddInfoEXT.pStdVPSs );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265SessionParametersAddInfoEXT.stdSPSCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265SessionParametersAddInfoEXT.pStdSPSs );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265SessionParametersAddInfoEXT.stdPPSCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265SessionParametersAddInfoEXT.pStdPPSs );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeH265SessionParametersCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoDecodeH265SessionParametersCreateInfoEXT const & videoDecodeH265SessionParametersCreateInfoEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265SessionParametersCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265SessionParametersCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265SessionParametersCreateInfoEXT.maxStdVPSCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265SessionParametersCreateInfoEXT.maxStdSPSCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265SessionParametersCreateInfoEXT.maxStdPPSCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeH265SessionParametersCreateInfoEXT.pParametersAddInfo );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoDecodeInfoKHR const & videoDecodeInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeInfoKHR.flags );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeInfoKHR.srcBuffer );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeInfoKHR.srcBufferOffset );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeInfoKHR.srcBufferRange );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeInfoKHR.dstPictureResource );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeInfoKHR.pSetupReferenceSlot );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeInfoKHR.referenceSlotCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeInfoKHR.pReferenceSlots );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoDecodeUsageInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoDecodeUsageInfoKHR const & videoDecodeUsageInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeUsageInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeUsageInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoDecodeUsageInfoKHR.videoUsageHints );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeCapabilitiesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeCapabilitiesKHR const & videoEncodeCapabilitiesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeCapabilitiesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeCapabilitiesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeCapabilitiesKHR.flags );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeCapabilitiesKHR.rateControlModes );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeCapabilitiesKHR.rateControlLayerCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeCapabilitiesKHR.qualityLevelCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeCapabilitiesKHR.inputImageDataFillAlignment );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264CapabilitiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH264CapabilitiesEXT const & videoEncodeH264CapabilitiesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264CapabilitiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264CapabilitiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264CapabilitiesEXT.flags );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264CapabilitiesEXT.inputModeFlags );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264CapabilitiesEXT.outputModeFlags );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264CapabilitiesEXT.maxPPictureL0ReferenceCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264CapabilitiesEXT.maxBPictureL0ReferenceCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264CapabilitiesEXT.maxL1ReferenceCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264CapabilitiesEXT.motionVectorsOverPicBoundariesFlag );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264CapabilitiesEXT.maxBytesPerPicDenom );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264CapabilitiesEXT.maxBitsPerMbDenom );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264CapabilitiesEXT.log2MaxMvLengthHorizontal );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264CapabilitiesEXT.log2MaxMvLengthVertical );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264DpbSlotInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH264DpbSlotInfoEXT const & videoEncodeH264DpbSlotInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264DpbSlotInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264DpbSlotInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264DpbSlotInfoEXT.slotIndex );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264DpbSlotInfoEXT.pStdReferenceInfo );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264EmitPictureParametersInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH264EmitPictureParametersInfoEXT const & videoEncodeH264EmitPictureParametersInfoEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264EmitPictureParametersInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264EmitPictureParametersInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264EmitPictureParametersInfoEXT.spsId );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264EmitPictureParametersInfoEXT.emitSpsEnable );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264EmitPictureParametersInfoEXT.ppsIdEntryCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264EmitPictureParametersInfoEXT.ppsIdEntries );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264FrameSizeEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH264FrameSizeEXT const & videoEncodeH264FrameSizeEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264FrameSizeEXT.frameISize );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264FrameSizeEXT.framePSize );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264FrameSizeEXT.frameBSize );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264ReferenceListsInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH264ReferenceListsInfoEXT const & videoEncodeH264ReferenceListsInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264ReferenceListsInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264ReferenceListsInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264ReferenceListsInfoEXT.referenceList0EntryCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264ReferenceListsInfoEXT.pReferenceList0Entries );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264ReferenceListsInfoEXT.referenceList1EntryCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264ReferenceListsInfoEXT.pReferenceList1Entries );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264ReferenceListsInfoEXT.pMemMgmtCtrlOperations );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264NaluSliceInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH264NaluSliceInfoEXT const & videoEncodeH264NaluSliceInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264NaluSliceInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264NaluSliceInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264NaluSliceInfoEXT.mbCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264NaluSliceInfoEXT.pReferenceFinalLists );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264NaluSliceInfoEXT.pSliceHeaderStd );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264ProfileInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH264ProfileInfoEXT const & videoEncodeH264ProfileInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264ProfileInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264ProfileInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264ProfileInfoEXT.stdProfileIdc );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264QpEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH264QpEXT const & videoEncodeH264QpEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264QpEXT.qpI );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264QpEXT.qpP );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264QpEXT.qpB );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264RateControlInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH264RateControlInfoEXT const & videoEncodeH264RateControlInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264RateControlInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264RateControlInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264RateControlInfoEXT.gopFrameCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264RateControlInfoEXT.idrPeriod );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264RateControlInfoEXT.consecutiveBFrameCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264RateControlInfoEXT.rateControlStructure );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264RateControlInfoEXT.temporalLayerCount );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264RateControlLayerInfoEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH264RateControlLayerInfoEXT const & videoEncodeH264RateControlLayerInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264RateControlLayerInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264RateControlLayerInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264RateControlLayerInfoEXT.temporalLayerId );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264RateControlLayerInfoEXT.useInitialRcQp );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264RateControlLayerInfoEXT.initialRcQp );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264RateControlLayerInfoEXT.useMinQp );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264RateControlLayerInfoEXT.minQp );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264RateControlLayerInfoEXT.useMaxQp );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264RateControlLayerInfoEXT.maxQp );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264RateControlLayerInfoEXT.useMaxFrameSize );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264RateControlLayerInfoEXT.maxFrameSize );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersAddInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersAddInfoEXT const & videoEncodeH264SessionParametersAddInfoEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264SessionParametersAddInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264SessionParametersAddInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264SessionParametersAddInfoEXT.stdSPSCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264SessionParametersAddInfoEXT.pStdSPSs );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264SessionParametersAddInfoEXT.stdPPSCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264SessionParametersAddInfoEXT.pStdPPSs );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH264SessionParametersCreateInfoEXT const & videoEncodeH264SessionParametersCreateInfoEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264SessionParametersCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264SessionParametersCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264SessionParametersCreateInfoEXT.maxStdSPSCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264SessionParametersCreateInfoEXT.maxStdPPSCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264SessionParametersCreateInfoEXT.pParametersAddInfo );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH264VclFrameInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH264VclFrameInfoEXT const & videoEncodeH264VclFrameInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264VclFrameInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264VclFrameInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264VclFrameInfoEXT.pReferenceFinalLists );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264VclFrameInfoEXT.naluSliceEntryCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264VclFrameInfoEXT.pNaluSliceEntries );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH264VclFrameInfoEXT.pCurrentPictureInfo );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265CapabilitiesEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH265CapabilitiesEXT const & videoEncodeH265CapabilitiesEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265CapabilitiesEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265CapabilitiesEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265CapabilitiesEXT.flags );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265CapabilitiesEXT.inputModeFlags );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265CapabilitiesEXT.outputModeFlags );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265CapabilitiesEXT.ctbSizes );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265CapabilitiesEXT.transformBlockSizes );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265CapabilitiesEXT.maxPPictureL0ReferenceCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265CapabilitiesEXT.maxBPictureL0ReferenceCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265CapabilitiesEXT.maxL1ReferenceCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265CapabilitiesEXT.maxSubLayersCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265CapabilitiesEXT.minLog2MinLumaCodingBlockSizeMinus3 );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265CapabilitiesEXT.maxLog2MinLumaCodingBlockSizeMinus3 );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265CapabilitiesEXT.minLog2MinLumaTransformBlockSizeMinus2 );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265CapabilitiesEXT.maxLog2MinLumaTransformBlockSizeMinus2 );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265CapabilitiesEXT.minMaxTransformHierarchyDepthInter );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265CapabilitiesEXT.maxMaxTransformHierarchyDepthInter );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265CapabilitiesEXT.minMaxTransformHierarchyDepthIntra );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265CapabilitiesEXT.maxMaxTransformHierarchyDepthIntra );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265CapabilitiesEXT.maxDiffCuQpDeltaDepth );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265CapabilitiesEXT.minMaxNumMergeCand );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265CapabilitiesEXT.maxMaxNumMergeCand );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265DpbSlotInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH265DpbSlotInfoEXT const & videoEncodeH265DpbSlotInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265DpbSlotInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265DpbSlotInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265DpbSlotInfoEXT.slotIndex );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265DpbSlotInfoEXT.pStdReferenceInfo );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265EmitPictureParametersInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH265EmitPictureParametersInfoEXT const & videoEncodeH265EmitPictureParametersInfoEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265EmitPictureParametersInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265EmitPictureParametersInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265EmitPictureParametersInfoEXT.vpsId );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265EmitPictureParametersInfoEXT.spsId );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265EmitPictureParametersInfoEXT.emitVpsEnable );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265EmitPictureParametersInfoEXT.emitSpsEnable );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265EmitPictureParametersInfoEXT.ppsIdEntryCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265EmitPictureParametersInfoEXT.ppsIdEntries );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265FrameSizeEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH265FrameSizeEXT const & videoEncodeH265FrameSizeEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265FrameSizeEXT.frameISize );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265FrameSizeEXT.framePSize );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265FrameSizeEXT.frameBSize );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265ReferenceListsInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH265ReferenceListsInfoEXT const & videoEncodeH265ReferenceListsInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265ReferenceListsInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265ReferenceListsInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265ReferenceListsInfoEXT.referenceList0EntryCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265ReferenceListsInfoEXT.pReferenceList0Entries );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265ReferenceListsInfoEXT.referenceList1EntryCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265ReferenceListsInfoEXT.pReferenceList1Entries );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265ReferenceListsInfoEXT.pReferenceModifications );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265NaluSliceSegmentInfoEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH265NaluSliceSegmentInfoEXT const & videoEncodeH265NaluSliceSegmentInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265NaluSliceSegmentInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265NaluSliceSegmentInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265NaluSliceSegmentInfoEXT.ctbCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265NaluSliceSegmentInfoEXT.pReferenceFinalLists );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265NaluSliceSegmentInfoEXT.pSliceSegmentHeaderStd );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265ProfileInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH265ProfileInfoEXT const & videoEncodeH265ProfileInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265ProfileInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265ProfileInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265ProfileInfoEXT.stdProfileIdc );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265QpEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH265QpEXT const & videoEncodeH265QpEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265QpEXT.qpI );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265QpEXT.qpP );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265QpEXT.qpB );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265RateControlInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH265RateControlInfoEXT const & videoEncodeH265RateControlInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265RateControlInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265RateControlInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265RateControlInfoEXT.gopFrameCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265RateControlInfoEXT.idrPeriod );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265RateControlInfoEXT.consecutiveBFrameCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265RateControlInfoEXT.rateControlStructure );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265RateControlInfoEXT.subLayerCount );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265RateControlLayerInfoEXT>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH265RateControlLayerInfoEXT const & videoEncodeH265RateControlLayerInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265RateControlLayerInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265RateControlLayerInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265RateControlLayerInfoEXT.temporalId );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265RateControlLayerInfoEXT.useInitialRcQp );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265RateControlLayerInfoEXT.initialRcQp );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265RateControlLayerInfoEXT.useMinQp );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265RateControlLayerInfoEXT.minQp );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265RateControlLayerInfoEXT.useMaxQp );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265RateControlLayerInfoEXT.maxQp );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265RateControlLayerInfoEXT.useMaxFrameSize );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265RateControlLayerInfoEXT.maxFrameSize );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersAddInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersAddInfoEXT const & videoEncodeH265SessionParametersAddInfoEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265SessionParametersAddInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265SessionParametersAddInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265SessionParametersAddInfoEXT.stdVPSCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265SessionParametersAddInfoEXT.pStdVPSs );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265SessionParametersAddInfoEXT.stdSPSCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265SessionParametersAddInfoEXT.pStdSPSs );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265SessionParametersAddInfoEXT.stdPPSCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265SessionParametersAddInfoEXT.pStdPPSs );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersCreateInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH265SessionParametersCreateInfoEXT const & videoEncodeH265SessionParametersCreateInfoEXT ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265SessionParametersCreateInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265SessionParametersCreateInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265SessionParametersCreateInfoEXT.maxStdVPSCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265SessionParametersCreateInfoEXT.maxStdSPSCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265SessionParametersCreateInfoEXT.maxStdPPSCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265SessionParametersCreateInfoEXT.pParametersAddInfo );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeH265VclFrameInfoEXT>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeH265VclFrameInfoEXT const & videoEncodeH265VclFrameInfoEXT ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265VclFrameInfoEXT.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265VclFrameInfoEXT.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265VclFrameInfoEXT.pReferenceFinalLists );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265VclFrameInfoEXT.naluSliceSegmentEntryCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265VclFrameInfoEXT.pNaluSliceSegmentEntries );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeH265VclFrameInfoEXT.pCurrentPictureInfo );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeInfoKHR const & videoEncodeInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeInfoKHR.flags );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeInfoKHR.qualityLevel );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeInfoKHR.dstBitstreamBuffer );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeInfoKHR.dstBitstreamBufferOffset );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeInfoKHR.dstBitstreamBufferMaxRange );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeInfoKHR.srcPictureResource );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeInfoKHR.pSetupReferenceSlot );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeInfoKHR.referenceSlotCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeInfoKHR.pReferenceSlots );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeInfoKHR.precedingExternallyEncodedBytes );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeRateControlLayerInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeRateControlLayerInfoKHR const & videoEncodeRateControlLayerInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeRateControlLayerInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeRateControlLayerInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeRateControlLayerInfoKHR.averageBitrate );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeRateControlLayerInfoKHR.maxBitrate );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeRateControlLayerInfoKHR.frameRateNumerator );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeRateControlLayerInfoKHR.frameRateDenominator );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeRateControlLayerInfoKHR.virtualBufferSizeInMs );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeRateControlLayerInfoKHR.initialVirtualBufferSizeInMs );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeRateControlInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeRateControlInfoKHR const & videoEncodeRateControlInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeRateControlInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeRateControlInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeRateControlInfoKHR.flags );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeRateControlInfoKHR.rateControlMode );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeRateControlInfoKHR.layerCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeRateControlInfoKHR.pLayerConfigs );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEncodeUsageInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEncodeUsageInfoKHR const & videoEncodeUsageInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeUsageInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeUsageInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeUsageInfoKHR.videoUsageHints );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeUsageInfoKHR.videoContentHints );
      VULKAN_HPP_HASH_COMBINE( seed, videoEncodeUsageInfoKHR.tuningMode );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoEndCodingInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoEndCodingInfoKHR const & videoEndCodingInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoEndCodingInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoEndCodingInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoEndCodingInfoKHR.flags );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoFormatPropertiesKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoFormatPropertiesKHR const & videoFormatPropertiesKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoFormatPropertiesKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoFormatPropertiesKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoFormatPropertiesKHR.format );
      VULKAN_HPP_HASH_COMBINE( seed, videoFormatPropertiesKHR.componentMapping );
      VULKAN_HPP_HASH_COMBINE( seed, videoFormatPropertiesKHR.imageCreateFlags );
      VULKAN_HPP_HASH_COMBINE( seed, videoFormatPropertiesKHR.imageType );
      VULKAN_HPP_HASH_COMBINE( seed, videoFormatPropertiesKHR.imageTiling );
      VULKAN_HPP_HASH_COMBINE( seed, videoFormatPropertiesKHR.imageUsageFlags );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoProfileInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoProfileInfoKHR const & videoProfileInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoProfileInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoProfileInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoProfileInfoKHR.videoCodecOperation );
      VULKAN_HPP_HASH_COMBINE( seed, videoProfileInfoKHR.chromaSubsampling );
      VULKAN_HPP_HASH_COMBINE( seed, videoProfileInfoKHR.lumaBitDepth );
      VULKAN_HPP_HASH_COMBINE( seed, videoProfileInfoKHR.chromaBitDepth );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoProfileListInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoProfileListInfoKHR const & videoProfileListInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoProfileListInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoProfileListInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoProfileListInfoKHR.profileCount );
      VULKAN_HPP_HASH_COMBINE( seed, videoProfileListInfoKHR.pProfiles );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoSessionCreateInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoSessionCreateInfoKHR const & videoSessionCreateInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoSessionCreateInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoSessionCreateInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoSessionCreateInfoKHR.queueFamilyIndex );
      VULKAN_HPP_HASH_COMBINE( seed, videoSessionCreateInfoKHR.flags );
      VULKAN_HPP_HASH_COMBINE( seed, videoSessionCreateInfoKHR.pVideoProfile );
      VULKAN_HPP_HASH_COMBINE( seed, videoSessionCreateInfoKHR.pictureFormat );
      VULKAN_HPP_HASH_COMBINE( seed, videoSessionCreateInfoKHR.maxCodedExtent );
      VULKAN_HPP_HASH_COMBINE( seed, videoSessionCreateInfoKHR.referencePictureFormat );
      VULKAN_HPP_HASH_COMBINE( seed, videoSessionCreateInfoKHR.maxDpbSlots );
      VULKAN_HPP_HASH_COMBINE( seed, videoSessionCreateInfoKHR.maxActiveReferencePictures );
      VULKAN_HPP_HASH_COMBINE( seed, videoSessionCreateInfoKHR.pStdHeaderVersion );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoSessionMemoryRequirementsKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoSessionMemoryRequirementsKHR const & videoSessionMemoryRequirementsKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoSessionMemoryRequirementsKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoSessionMemoryRequirementsKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoSessionMemoryRequirementsKHR.memoryBindIndex );
      VULKAN_HPP_HASH_COMBINE( seed, videoSessionMemoryRequirementsKHR.memoryRequirements );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoSessionParametersCreateInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoSessionParametersCreateInfoKHR const & videoSessionParametersCreateInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoSessionParametersCreateInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoSessionParametersCreateInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoSessionParametersCreateInfoKHR.flags );
      VULKAN_HPP_HASH_COMBINE( seed, videoSessionParametersCreateInfoKHR.videoSessionParametersTemplate );
      VULKAN_HPP_HASH_COMBINE( seed, videoSessionParametersCreateInfoKHR.videoSession );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::VideoSessionParametersUpdateInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::VideoSessionParametersUpdateInfoKHR const & videoSessionParametersUpdateInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, videoSessionParametersUpdateInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, videoSessionParametersUpdateInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, videoSessionParametersUpdateInfoKHR.updateSequenceCount );
      return seed;
    }
  };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_USE_PLATFORM_WAYLAND_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::WaylandSurfaceCreateInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::WaylandSurfaceCreateInfoKHR const & waylandSurfaceCreateInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, waylandSurfaceCreateInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, waylandSurfaceCreateInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, waylandSurfaceCreateInfoKHR.flags );
      VULKAN_HPP_HASH_COMBINE( seed, waylandSurfaceCreateInfoKHR.display );
      VULKAN_HPP_HASH_COMBINE( seed, waylandSurfaceCreateInfoKHR.surface );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_WAYLAND_KHR*/

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Win32KeyedMutexAcquireReleaseInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Win32KeyedMutexAcquireReleaseInfoKHR const & win32KeyedMutexAcquireReleaseInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, win32KeyedMutexAcquireReleaseInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, win32KeyedMutexAcquireReleaseInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, win32KeyedMutexAcquireReleaseInfoKHR.acquireCount );
      VULKAN_HPP_HASH_COMBINE( seed, win32KeyedMutexAcquireReleaseInfoKHR.pAcquireSyncs );
      VULKAN_HPP_HASH_COMBINE( seed, win32KeyedMutexAcquireReleaseInfoKHR.pAcquireKeys );
      VULKAN_HPP_HASH_COMBINE( seed, win32KeyedMutexAcquireReleaseInfoKHR.pAcquireTimeouts );
      VULKAN_HPP_HASH_COMBINE( seed, win32KeyedMutexAcquireReleaseInfoKHR.releaseCount );
      VULKAN_HPP_HASH_COMBINE( seed, win32KeyedMutexAcquireReleaseInfoKHR.pReleaseSyncs );
      VULKAN_HPP_HASH_COMBINE( seed, win32KeyedMutexAcquireReleaseInfoKHR.pReleaseKeys );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Win32KeyedMutexAcquireReleaseInfoNV>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Win32KeyedMutexAcquireReleaseInfoNV const & win32KeyedMutexAcquireReleaseInfoNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, win32KeyedMutexAcquireReleaseInfoNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, win32KeyedMutexAcquireReleaseInfoNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, win32KeyedMutexAcquireReleaseInfoNV.acquireCount );
      VULKAN_HPP_HASH_COMBINE( seed, win32KeyedMutexAcquireReleaseInfoNV.pAcquireSyncs );
      VULKAN_HPP_HASH_COMBINE( seed, win32KeyedMutexAcquireReleaseInfoNV.pAcquireKeys );
      VULKAN_HPP_HASH_COMBINE( seed, win32KeyedMutexAcquireReleaseInfoNV.pAcquireTimeoutMilliseconds );
      VULKAN_HPP_HASH_COMBINE( seed, win32KeyedMutexAcquireReleaseInfoNV.releaseCount );
      VULKAN_HPP_HASH_COMBINE( seed, win32KeyedMutexAcquireReleaseInfoNV.pReleaseSyncs );
      VULKAN_HPP_HASH_COMBINE( seed, win32KeyedMutexAcquireReleaseInfoNV.pReleaseKeys );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::Win32SurfaceCreateInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::Win32SurfaceCreateInfoKHR const & win32SurfaceCreateInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, win32SurfaceCreateInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, win32SurfaceCreateInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, win32SurfaceCreateInfoKHR.flags );
      VULKAN_HPP_HASH_COMBINE( seed, win32SurfaceCreateInfoKHR.hinstance );
      VULKAN_HPP_HASH_COMBINE( seed, win32SurfaceCreateInfoKHR.hwnd );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::WriteDescriptorSet>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::WriteDescriptorSet const & writeDescriptorSet ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, writeDescriptorSet.sType );
      VULKAN_HPP_HASH_COMBINE( seed, writeDescriptorSet.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, writeDescriptorSet.dstSet );
      VULKAN_HPP_HASH_COMBINE( seed, writeDescriptorSet.dstBinding );
      VULKAN_HPP_HASH_COMBINE( seed, writeDescriptorSet.dstArrayElement );
      VULKAN_HPP_HASH_COMBINE( seed, writeDescriptorSet.descriptorCount );
      VULKAN_HPP_HASH_COMBINE( seed, writeDescriptorSet.descriptorType );
      VULKAN_HPP_HASH_COMBINE( seed, writeDescriptorSet.pImageInfo );
      VULKAN_HPP_HASH_COMBINE( seed, writeDescriptorSet.pBufferInfo );
      VULKAN_HPP_HASH_COMBINE( seed, writeDescriptorSet.pTexelBufferView );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::WriteDescriptorSetAccelerationStructureKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::WriteDescriptorSetAccelerationStructureKHR const & writeDescriptorSetAccelerationStructureKHR ) const
      VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, writeDescriptorSetAccelerationStructureKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, writeDescriptorSetAccelerationStructureKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, writeDescriptorSetAccelerationStructureKHR.accelerationStructureCount );
      VULKAN_HPP_HASH_COMBINE( seed, writeDescriptorSetAccelerationStructureKHR.pAccelerationStructures );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::WriteDescriptorSetAccelerationStructureNV>
  {
    std::size_t
      operator()( VULKAN_HPP_NAMESPACE::WriteDescriptorSetAccelerationStructureNV const & writeDescriptorSetAccelerationStructureNV ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, writeDescriptorSetAccelerationStructureNV.sType );
      VULKAN_HPP_HASH_COMBINE( seed, writeDescriptorSetAccelerationStructureNV.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, writeDescriptorSetAccelerationStructureNV.accelerationStructureCount );
      VULKAN_HPP_HASH_COMBINE( seed, writeDescriptorSetAccelerationStructureNV.pAccelerationStructures );
      return seed;
    }
  };

  template <>
  struct hash<VULKAN_HPP_NAMESPACE::WriteDescriptorSetInlineUniformBlock>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::WriteDescriptorSetInlineUniformBlock const & writeDescriptorSetInlineUniformBlock ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, writeDescriptorSetInlineUniformBlock.sType );
      VULKAN_HPP_HASH_COMBINE( seed, writeDescriptorSetInlineUniformBlock.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, writeDescriptorSetInlineUniformBlock.dataSize );
      VULKAN_HPP_HASH_COMBINE( seed, writeDescriptorSetInlineUniformBlock.pData );
      return seed;
    }
  };

#  if defined( VK_USE_PLATFORM_XCB_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::XcbSurfaceCreateInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::XcbSurfaceCreateInfoKHR const & xcbSurfaceCreateInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, xcbSurfaceCreateInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, xcbSurfaceCreateInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, xcbSurfaceCreateInfoKHR.flags );
      VULKAN_HPP_HASH_COMBINE( seed, xcbSurfaceCreateInfoKHR.connection );
      VULKAN_HPP_HASH_COMBINE( seed, xcbSurfaceCreateInfoKHR.window );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_XCB_KHR*/

#  if defined( VK_USE_PLATFORM_XLIB_KHR )
  template <>
  struct hash<VULKAN_HPP_NAMESPACE::XlibSurfaceCreateInfoKHR>
  {
    std::size_t operator()( VULKAN_HPP_NAMESPACE::XlibSurfaceCreateInfoKHR const & xlibSurfaceCreateInfoKHR ) const VULKAN_HPP_NOEXCEPT
    {
      std::size_t seed = 0;
      VULKAN_HPP_HASH_COMBINE( seed, xlibSurfaceCreateInfoKHR.sType );
      VULKAN_HPP_HASH_COMBINE( seed, xlibSurfaceCreateInfoKHR.pNext );
      VULKAN_HPP_HASH_COMBINE( seed, xlibSurfaceCreateInfoKHR.flags );
      VULKAN_HPP_HASH_COMBINE( seed, xlibSurfaceCreateInfoKHR.dpy );
      VULKAN_HPP_HASH_COMBINE( seed, xlibSurfaceCreateInfoKHR.window );
      return seed;
    }
  };
#  endif /*VK_USE_PLATFORM_XLIB_KHR*/

#endif  // 14 <= VULKAN_HPP_CPP_VERSION

}  // namespace std
#endif
