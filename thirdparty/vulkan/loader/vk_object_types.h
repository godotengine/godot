// *** THIS FILE IS GENERATED - DO NOT EDIT ***
// See helper_file_generator.py for modifications


/***************************************************************************
 *
 * Copyright (c) 2015-2017 The Khronos Group Inc.
 * Copyright (c) 2015-2017 Valve Corporation
 * Copyright (c) 2015-2017 LunarG, Inc.
 * Copyright (c) 2015-2017 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Mark Lobodzinski <mark@lunarg.com>
 * Author: Courtney Goeltzenleuchter <courtneygo@google.com>
 * Author: Tobin Ehlis <tobine@google.com>
 * Author: Chris Forbes <chrisforbes@google.com>
 * Author: John Zulauf<jzulauf@lunarg.com>
 *
 ****************************************************************************/


#pragma once

#include <vulkan/vulkan.h>

// Object Type enum for validation layer internal object handling
typedef enum VulkanObjectType {
    kVulkanObjectTypeUnknown = 0,
    kVulkanObjectTypeBuffer = 1,
    kVulkanObjectTypeImage = 2,
    kVulkanObjectTypeInstance = 3,
    kVulkanObjectTypePhysicalDevice = 4,
    kVulkanObjectTypeDevice = 5,
    kVulkanObjectTypeQueue = 6,
    kVulkanObjectTypeSemaphore = 7,
    kVulkanObjectTypeCommandBuffer = 8,
    kVulkanObjectTypeFence = 9,
    kVulkanObjectTypeDeviceMemory = 10,
    kVulkanObjectTypeEvent = 11,
    kVulkanObjectTypeQueryPool = 12,
    kVulkanObjectTypeBufferView = 13,
    kVulkanObjectTypeImageView = 14,
    kVulkanObjectTypeShaderModule = 15,
    kVulkanObjectTypePipelineCache = 16,
    kVulkanObjectTypePipelineLayout = 17,
    kVulkanObjectTypePipeline = 18,
    kVulkanObjectTypeRenderPass = 19,
    kVulkanObjectTypeDescriptorSetLayout = 20,
    kVulkanObjectTypeSampler = 21,
    kVulkanObjectTypeDescriptorSet = 22,
    kVulkanObjectTypeDescriptorPool = 23,
    kVulkanObjectTypeFramebuffer = 24,
    kVulkanObjectTypeCommandPool = 25,
    kVulkanObjectTypeSamplerYcbcrConversion = 26,
    kVulkanObjectTypeDescriptorUpdateTemplate = 27,
    kVulkanObjectTypeSurfaceKHR = 28,
    kVulkanObjectTypeSwapchainKHR = 29,
    kVulkanObjectTypeDisplayKHR = 30,
    kVulkanObjectTypeDisplayModeKHR = 31,
    kVulkanObjectTypeDeferredOperationKHR = 32,
    kVulkanObjectTypeDebugReportCallbackEXT = 33,
    kVulkanObjectTypeDebugUtilsMessengerEXT = 34,
    kVulkanObjectTypeValidationCacheEXT = 35,
    kVulkanObjectTypeAccelerationStructureNV = 36,
    kVulkanObjectTypePerformanceConfigurationINTEL = 37,
    kVulkanObjectTypeIndirectCommandsLayoutNV = 38,
    kVulkanObjectTypePrivateDataSlotEXT = 39,
    kVulkanObjectTypeAccelerationStructureKHR = 40,
    kVulkanObjectTypeMax = 41,
    // Aliases for backwards compatibilty of "promoted" types
    kVulkanObjectTypeDescriptorUpdateTemplateKHR = kVulkanObjectTypeDescriptorUpdateTemplate,
    kVulkanObjectTypeSamplerYcbcrConversionKHR = kVulkanObjectTypeSamplerYcbcrConversion,
} VulkanObjectType;

// Array of object name strings for OBJECT_TYPE enum conversion
static const char * const object_string[kVulkanObjectTypeMax] = {
    "Unknown",
    "Buffer",
    "Image",
    "Instance",
    "PhysicalDevice",
    "Device",
    "Queue",
    "Semaphore",
    "CommandBuffer",
    "Fence",
    "DeviceMemory",
    "Event",
    "QueryPool",
    "BufferView",
    "ImageView",
    "ShaderModule",
    "PipelineCache",
    "PipelineLayout",
    "Pipeline",
    "RenderPass",
    "DescriptorSetLayout",
    "Sampler",
    "DescriptorSet",
    "DescriptorPool",
    "Framebuffer",
    "CommandPool",
    "SamplerYcbcrConversion",
    "DescriptorUpdateTemplate",
    "SurfaceKHR",
    "SwapchainKHR",
    "DisplayKHR",
    "DisplayModeKHR",
    "DeferredOperationKHR",
    "DebugReportCallbackEXT",
    "DebugUtilsMessengerEXT",
    "ValidationCacheEXT",
    "AccelerationStructureNV",
    "PerformanceConfigurationINTEL",
    "IndirectCommandsLayoutNV",
    "PrivateDataSlotEXT",
    "AccelerationStructureKHR",
};

// Helper array to get Vulkan VK_EXT_debug_report object type enum from the internal layers version
const VkDebugReportObjectTypeEXT get_debug_report_enum[] = {
    VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT, // kVulkanObjectTypeUnknown
    VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT,   // kVulkanObjectTypeBuffer
    VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,   // kVulkanObjectTypeImage
    VK_DEBUG_REPORT_OBJECT_TYPE_INSTANCE_EXT,   // kVulkanObjectTypeInstance
    VK_DEBUG_REPORT_OBJECT_TYPE_PHYSICAL_DEVICE_EXT,   // kVulkanObjectTypePhysicalDevice
    VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_EXT,   // kVulkanObjectTypeDevice
    VK_DEBUG_REPORT_OBJECT_TYPE_QUEUE_EXT,   // kVulkanObjectTypeQueue
    VK_DEBUG_REPORT_OBJECT_TYPE_SEMAPHORE_EXT,   // kVulkanObjectTypeSemaphore
    VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,   // kVulkanObjectTypeCommandBuffer
    VK_DEBUG_REPORT_OBJECT_TYPE_FENCE_EXT,   // kVulkanObjectTypeFence
    VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_MEMORY_EXT,   // kVulkanObjectTypeDeviceMemory
    VK_DEBUG_REPORT_OBJECT_TYPE_EVENT_EXT,   // kVulkanObjectTypeEvent
    VK_DEBUG_REPORT_OBJECT_TYPE_QUERY_POOL_EXT,   // kVulkanObjectTypeQueryPool
    VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_VIEW_EXT,   // kVulkanObjectTypeBufferView
    VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_VIEW_EXT,   // kVulkanObjectTypeImageView
    VK_DEBUG_REPORT_OBJECT_TYPE_SHADER_MODULE_EXT,   // kVulkanObjectTypeShaderModule
    VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_CACHE_EXT,   // kVulkanObjectTypePipelineCache
    VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_LAYOUT_EXT,   // kVulkanObjectTypePipelineLayout
    VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_EXT,   // kVulkanObjectTypePipeline
    VK_DEBUG_REPORT_OBJECT_TYPE_RENDER_PASS_EXT,   // kVulkanObjectTypeRenderPass
    VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT_EXT,   // kVulkanObjectTypeDescriptorSetLayout
    VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_EXT,   // kVulkanObjectTypeSampler
    VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_EXT,   // kVulkanObjectTypeDescriptorSet
    VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_POOL_EXT,   // kVulkanObjectTypeDescriptorPool
    VK_DEBUG_REPORT_OBJECT_TYPE_FRAMEBUFFER_EXT,   // kVulkanObjectTypeFramebuffer
    VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_POOL_EXT,   // kVulkanObjectTypeCommandPool
    VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION_EXT,   // kVulkanObjectTypeSamplerYcbcrConversion
    VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_EXT,   // kVulkanObjectTypeDescriptorUpdateTemplate
    VK_DEBUG_REPORT_OBJECT_TYPE_SURFACE_KHR_EXT,   // kVulkanObjectTypeSurfaceKHR
    VK_DEBUG_REPORT_OBJECT_TYPE_SWAPCHAIN_KHR_EXT,   // kVulkanObjectTypeSwapchainKHR
    VK_DEBUG_REPORT_OBJECT_TYPE_DISPLAY_KHR_EXT,   // kVulkanObjectTypeDisplayKHR
    VK_DEBUG_REPORT_OBJECT_TYPE_DISPLAY_MODE_KHR_EXT,   // kVulkanObjectTypeDisplayModeKHR
    VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT,   // kVulkanObjectTypeDeferredOperationKHR
    VK_DEBUG_REPORT_OBJECT_TYPE_DEBUG_REPORT_CALLBACK_EXT_EXT,   // kVulkanObjectTypeDebugReportCallbackEXT
    VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT,   // kVulkanObjectTypeDebugUtilsMessengerEXT
    VK_DEBUG_REPORT_OBJECT_TYPE_VALIDATION_CACHE_EXT_EXT,   // kVulkanObjectTypeValidationCacheEXT
    VK_DEBUG_REPORT_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV_EXT,   // kVulkanObjectTypeAccelerationStructureNV
    VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT,   // kVulkanObjectTypePerformanceConfigurationINTEL
    VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT,   // kVulkanObjectTypeIndirectCommandsLayoutNV
    VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT,   // kVulkanObjectTypePrivateDataSlotEXT
    VK_DEBUG_REPORT_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR_EXT,   // kVulkanObjectTypeAccelerationStructureKHR
};

// Helper array to get Official Vulkan VkObjectType enum from the internal layers version
const VkObjectType get_object_type_enum[] = {
    VK_OBJECT_TYPE_UNKNOWN, // kVulkanObjectTypeUnknown
    VK_OBJECT_TYPE_BUFFER,   // kVulkanObjectTypeBuffer
    VK_OBJECT_TYPE_IMAGE,   // kVulkanObjectTypeImage
    VK_OBJECT_TYPE_INSTANCE,   // kVulkanObjectTypeInstance
    VK_OBJECT_TYPE_PHYSICAL_DEVICE,   // kVulkanObjectTypePhysicalDevice
    VK_OBJECT_TYPE_DEVICE,   // kVulkanObjectTypeDevice
    VK_OBJECT_TYPE_QUEUE,   // kVulkanObjectTypeQueue
    VK_OBJECT_TYPE_SEMAPHORE,   // kVulkanObjectTypeSemaphore
    VK_OBJECT_TYPE_COMMAND_BUFFER,   // kVulkanObjectTypeCommandBuffer
    VK_OBJECT_TYPE_FENCE,   // kVulkanObjectTypeFence
    VK_OBJECT_TYPE_DEVICE_MEMORY,   // kVulkanObjectTypeDeviceMemory
    VK_OBJECT_TYPE_EVENT,   // kVulkanObjectTypeEvent
    VK_OBJECT_TYPE_QUERY_POOL,   // kVulkanObjectTypeQueryPool
    VK_OBJECT_TYPE_BUFFER_VIEW,   // kVulkanObjectTypeBufferView
    VK_OBJECT_TYPE_IMAGE_VIEW,   // kVulkanObjectTypeImageView
    VK_OBJECT_TYPE_SHADER_MODULE,   // kVulkanObjectTypeShaderModule
    VK_OBJECT_TYPE_PIPELINE_CACHE,   // kVulkanObjectTypePipelineCache
    VK_OBJECT_TYPE_PIPELINE_LAYOUT,   // kVulkanObjectTypePipelineLayout
    VK_OBJECT_TYPE_PIPELINE,   // kVulkanObjectTypePipeline
    VK_OBJECT_TYPE_RENDER_PASS,   // kVulkanObjectTypeRenderPass
    VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,   // kVulkanObjectTypeDescriptorSetLayout
    VK_OBJECT_TYPE_SAMPLER,   // kVulkanObjectTypeSampler
    VK_OBJECT_TYPE_DESCRIPTOR_SET,   // kVulkanObjectTypeDescriptorSet
    VK_OBJECT_TYPE_DESCRIPTOR_POOL,   // kVulkanObjectTypeDescriptorPool
    VK_OBJECT_TYPE_FRAMEBUFFER,   // kVulkanObjectTypeFramebuffer
    VK_OBJECT_TYPE_COMMAND_POOL,   // kVulkanObjectTypeCommandPool
    VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION,   // kVulkanObjectTypeSamplerYcbcrConversion
    VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE,   // kVulkanObjectTypeDescriptorUpdateTemplate
    VK_OBJECT_TYPE_SURFACE_KHR,   // kVulkanObjectTypeSurfaceKHR
    VK_OBJECT_TYPE_SWAPCHAIN_KHR,   // kVulkanObjectTypeSwapchainKHR
    VK_OBJECT_TYPE_DISPLAY_KHR,   // kVulkanObjectTypeDisplayKHR
    VK_OBJECT_TYPE_DISPLAY_MODE_KHR,   // kVulkanObjectTypeDisplayModeKHR
    VK_OBJECT_TYPE_DEFERRED_OPERATION_KHR,   // kVulkanObjectTypeDeferredOperationKHR
    VK_OBJECT_TYPE_DEBUG_REPORT_CALLBACK_EXT,   // kVulkanObjectTypeDebugReportCallbackEXT
    VK_OBJECT_TYPE_DEBUG_UTILS_MESSENGER_EXT,   // kVulkanObjectTypeDebugUtilsMessengerEXT
    VK_OBJECT_TYPE_VALIDATION_CACHE_EXT,   // kVulkanObjectTypeValidationCacheEXT
    VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV,   // kVulkanObjectTypeAccelerationStructureNV
    VK_OBJECT_TYPE_PERFORMANCE_CONFIGURATION_INTEL,   // kVulkanObjectTypePerformanceConfigurationINTEL
    VK_OBJECT_TYPE_INDIRECT_COMMANDS_LAYOUT_NV,   // kVulkanObjectTypeIndirectCommandsLayoutNV
    VK_OBJECT_TYPE_PRIVATE_DATA_SLOT_EXT,   // kVulkanObjectTypePrivateDataSlotEXT
    VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR,   // kVulkanObjectTypeAccelerationStructureKHR
};

// Helper function to convert from VkDebugReportObjectTypeEXT to VkObjectType
static inline VkObjectType convertDebugReportObjectToCoreObject(VkDebugReportObjectTypeEXT debug_report_obj){
    if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT) {
        return VK_OBJECT_TYPE_UNKNOWN;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT) {
        return VK_OBJECT_TYPE_UNKNOWN;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_INSTANCE_EXT) {
        return VK_OBJECT_TYPE_INSTANCE;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_PHYSICAL_DEVICE_EXT) {
        return VK_OBJECT_TYPE_PHYSICAL_DEVICE;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_EXT) {
        return VK_OBJECT_TYPE_DEVICE;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_QUEUE_EXT) {
        return VK_OBJECT_TYPE_QUEUE;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_SEMAPHORE_EXT) {
        return VK_OBJECT_TYPE_SEMAPHORE;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT) {
        return VK_OBJECT_TYPE_COMMAND_BUFFER;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_FENCE_EXT) {
        return VK_OBJECT_TYPE_FENCE;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_MEMORY_EXT) {
        return VK_OBJECT_TYPE_DEVICE_MEMORY;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT) {
        return VK_OBJECT_TYPE_BUFFER;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT) {
        return VK_OBJECT_TYPE_IMAGE;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_EVENT_EXT) {
        return VK_OBJECT_TYPE_EVENT;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_QUERY_POOL_EXT) {
        return VK_OBJECT_TYPE_QUERY_POOL;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_VIEW_EXT) {
        return VK_OBJECT_TYPE_BUFFER_VIEW;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_VIEW_EXT) {
        return VK_OBJECT_TYPE_IMAGE_VIEW;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_SHADER_MODULE_EXT) {
        return VK_OBJECT_TYPE_SHADER_MODULE;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_CACHE_EXT) {
        return VK_OBJECT_TYPE_PIPELINE_CACHE;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_LAYOUT_EXT) {
        return VK_OBJECT_TYPE_PIPELINE_LAYOUT;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_RENDER_PASS_EXT) {
        return VK_OBJECT_TYPE_RENDER_PASS;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_EXT) {
        return VK_OBJECT_TYPE_PIPELINE;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT_EXT) {
        return VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_EXT) {
        return VK_OBJECT_TYPE_SAMPLER;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_POOL_EXT) {
        return VK_OBJECT_TYPE_DESCRIPTOR_POOL;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_EXT) {
        return VK_OBJECT_TYPE_DESCRIPTOR_SET;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_FRAMEBUFFER_EXT) {
        return VK_OBJECT_TYPE_FRAMEBUFFER;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_POOL_EXT) {
        return VK_OBJECT_TYPE_COMMAND_POOL;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION_EXT) {
        return VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_EXT) {
        return VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_SURFACE_KHR_EXT) {
        return VK_OBJECT_TYPE_SURFACE_KHR;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_SWAPCHAIN_KHR_EXT) {
        return VK_OBJECT_TYPE_SWAPCHAIN_KHR;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_DISPLAY_KHR_EXT) {
        return VK_OBJECT_TYPE_DISPLAY_KHR;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_DISPLAY_MODE_KHR_EXT) {
        return VK_OBJECT_TYPE_DISPLAY_MODE_KHR;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_DEBUG_REPORT_CALLBACK_EXT_EXT) {
        return VK_OBJECT_TYPE_DEBUG_REPORT_CALLBACK_EXT;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_KHR_EXT) {
        return VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_KHR;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR_EXT) {
        return VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION_KHR_EXT) {
        return VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION_KHR;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_VALIDATION_CACHE_EXT_EXT) {
        return VK_OBJECT_TYPE_VALIDATION_CACHE_EXT;
    } else if (debug_report_obj == VK_DEBUG_REPORT_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV_EXT) {
        return VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV;
    }
    return VK_OBJECT_TYPE_UNKNOWN;
}

// Helper function to convert from VkDebugReportObjectTypeEXT to VkObjectType
static inline VkDebugReportObjectTypeEXT convertCoreObjectToDebugReportObject(VkObjectType core_report_obj){
    if (core_report_obj == VK_OBJECT_TYPE_UNKNOWN) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_UNKNOWN) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_INSTANCE) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_INSTANCE_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_PHYSICAL_DEVICE) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_PHYSICAL_DEVICE_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_DEVICE) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_QUEUE) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_QUEUE_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_SEMAPHORE) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_SEMAPHORE_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_COMMAND_BUFFER) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_FENCE) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_FENCE_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_DEVICE_MEMORY) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_MEMORY_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_BUFFER) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_IMAGE) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_EVENT) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_EVENT_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_QUERY_POOL) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_QUERY_POOL_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_BUFFER_VIEW) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_VIEW_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_IMAGE_VIEW) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_VIEW_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_SHADER_MODULE) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_SHADER_MODULE_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_PIPELINE_CACHE) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_CACHE_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_PIPELINE_LAYOUT) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_LAYOUT_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_RENDER_PASS) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_RENDER_PASS_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_PIPELINE) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_SAMPLER) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_DESCRIPTOR_POOL) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_POOL_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_DESCRIPTOR_SET) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_FRAMEBUFFER) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_FRAMEBUFFER_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_COMMAND_POOL) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_POOL_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_SURFACE_KHR) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_SURFACE_KHR_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_SWAPCHAIN_KHR) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_SWAPCHAIN_KHR_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_DISPLAY_KHR) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_DISPLAY_KHR_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_DISPLAY_MODE_KHR) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_DISPLAY_MODE_KHR_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_DEBUG_REPORT_CALLBACK_EXT) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_DEBUG_REPORT_CALLBACK_EXT_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_KHR) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_KHR_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION_KHR) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION_KHR_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_VALIDATION_CACHE_EXT) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_VALIDATION_CACHE_EXT_EXT;
    } else if (core_report_obj == VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV) {
        return VK_DEBUG_REPORT_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV_EXT;
    }
    return VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT;
}
