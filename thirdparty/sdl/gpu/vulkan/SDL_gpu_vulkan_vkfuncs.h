/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

/*
 * Global functions from the Vulkan Loader
 */

#ifndef VULKAN_GLOBAL_FUNCTION
#define VULKAN_GLOBAL_FUNCTION(name)
#endif
VULKAN_GLOBAL_FUNCTION(vkCreateInstance)
VULKAN_GLOBAL_FUNCTION(vkEnumerateInstanceExtensionProperties)
VULKAN_GLOBAL_FUNCTION(vkEnumerateInstanceLayerProperties)

/*
 * vkInstance, created by global vkCreateInstance function
 */

#ifndef VULKAN_INSTANCE_FUNCTION
#define VULKAN_INSTANCE_FUNCTION(name)
#endif

// Vulkan 1.0
VULKAN_INSTANCE_FUNCTION(vkGetDeviceProcAddr)
VULKAN_INSTANCE_FUNCTION(vkCreateDevice)
VULKAN_INSTANCE_FUNCTION(vkDestroyInstance)
VULKAN_INSTANCE_FUNCTION(vkEnumerateDeviceExtensionProperties)
VULKAN_INSTANCE_FUNCTION(vkEnumeratePhysicalDevices)
VULKAN_INSTANCE_FUNCTION(vkGetPhysicalDeviceFeatures)
VULKAN_INSTANCE_FUNCTION(vkGetPhysicalDeviceQueueFamilyProperties)
VULKAN_INSTANCE_FUNCTION(vkGetPhysicalDeviceFormatProperties)
VULKAN_INSTANCE_FUNCTION(vkGetPhysicalDeviceImageFormatProperties)
VULKAN_INSTANCE_FUNCTION(vkGetPhysicalDeviceMemoryProperties)
VULKAN_INSTANCE_FUNCTION(vkGetPhysicalDeviceProperties)

// VK_KHR_get_physical_device_properties2, needed for KHR_driver_properties
VULKAN_INSTANCE_FUNCTION(vkGetPhysicalDeviceProperties2KHR)

// VK_KHR_surface
VULKAN_INSTANCE_FUNCTION(vkDestroySurfaceKHR)
VULKAN_INSTANCE_FUNCTION(vkGetPhysicalDeviceSurfaceCapabilitiesKHR)
VULKAN_INSTANCE_FUNCTION(vkGetPhysicalDeviceSurfaceFormatsKHR)
VULKAN_INSTANCE_FUNCTION(vkGetPhysicalDeviceSurfacePresentModesKHR)
VULKAN_INSTANCE_FUNCTION(vkGetPhysicalDeviceSurfaceSupportKHR)

// VK_EXT_debug_utils
VULKAN_INSTANCE_FUNCTION(vkCmdBeginDebugUtilsLabelEXT)
VULKAN_INSTANCE_FUNCTION(vkSetDebugUtilsObjectNameEXT)
VULKAN_INSTANCE_FUNCTION(vkCmdEndDebugUtilsLabelEXT)
VULKAN_INSTANCE_FUNCTION(vkCmdInsertDebugUtilsLabelEXT)

/*
 * vkDevice, created by a vkInstance
 */

#ifndef VULKAN_DEVICE_FUNCTION
#define VULKAN_DEVICE_FUNCTION(name)
#endif

// Vulkan 1.0
VULKAN_DEVICE_FUNCTION(vkAllocateCommandBuffers)
VULKAN_DEVICE_FUNCTION(vkAllocateDescriptorSets)
VULKAN_DEVICE_FUNCTION(vkAllocateMemory)
VULKAN_DEVICE_FUNCTION(vkBeginCommandBuffer)
VULKAN_DEVICE_FUNCTION(vkBindBufferMemory)
VULKAN_DEVICE_FUNCTION(vkBindImageMemory)
VULKAN_DEVICE_FUNCTION(vkCmdBeginRenderPass)
VULKAN_DEVICE_FUNCTION(vkCmdBindDescriptorSets)
VULKAN_DEVICE_FUNCTION(vkCmdBindIndexBuffer)
VULKAN_DEVICE_FUNCTION(vkCmdBindPipeline)
VULKAN_DEVICE_FUNCTION(vkCmdBindVertexBuffers)
VULKAN_DEVICE_FUNCTION(vkCmdBlitImage)
VULKAN_DEVICE_FUNCTION(vkCmdClearAttachments)
VULKAN_DEVICE_FUNCTION(vkCmdClearColorImage)
VULKAN_DEVICE_FUNCTION(vkCmdClearDepthStencilImage)
VULKAN_DEVICE_FUNCTION(vkCmdCopyBuffer)
VULKAN_DEVICE_FUNCTION(vkCmdCopyImage)
VULKAN_DEVICE_FUNCTION(vkCmdCopyBufferToImage)
VULKAN_DEVICE_FUNCTION(vkCmdCopyImageToBuffer)
VULKAN_DEVICE_FUNCTION(vkCmdDispatch)
VULKAN_DEVICE_FUNCTION(vkCmdDispatchIndirect)
VULKAN_DEVICE_FUNCTION(vkCmdDraw)
VULKAN_DEVICE_FUNCTION(vkCmdDrawIndexed)
VULKAN_DEVICE_FUNCTION(vkCmdDrawIndexedIndirect)
VULKAN_DEVICE_FUNCTION(vkCmdDrawIndirect)
VULKAN_DEVICE_FUNCTION(vkCmdEndRenderPass)
VULKAN_DEVICE_FUNCTION(vkCmdPipelineBarrier)
VULKAN_DEVICE_FUNCTION(vkCmdResolveImage)
VULKAN_DEVICE_FUNCTION(vkCmdSetBlendConstants)
VULKAN_DEVICE_FUNCTION(vkCmdSetDepthBias)
VULKAN_DEVICE_FUNCTION(vkCmdSetScissor)
VULKAN_DEVICE_FUNCTION(vkCmdSetStencilReference)
VULKAN_DEVICE_FUNCTION(vkCmdSetViewport)
VULKAN_DEVICE_FUNCTION(vkCreateBuffer)
VULKAN_DEVICE_FUNCTION(vkCreateCommandPool)
VULKAN_DEVICE_FUNCTION(vkCreateDescriptorPool)
VULKAN_DEVICE_FUNCTION(vkCreateDescriptorSetLayout)
VULKAN_DEVICE_FUNCTION(vkCreateFence)
VULKAN_DEVICE_FUNCTION(vkCreateFramebuffer)
VULKAN_DEVICE_FUNCTION(vkCreateComputePipelines)
VULKAN_DEVICE_FUNCTION(vkCreateGraphicsPipelines)
VULKAN_DEVICE_FUNCTION(vkCreateImage)
VULKAN_DEVICE_FUNCTION(vkCreateImageView)
VULKAN_DEVICE_FUNCTION(vkCreatePipelineCache)
VULKAN_DEVICE_FUNCTION(vkCreatePipelineLayout)
VULKAN_DEVICE_FUNCTION(vkCreateRenderPass)
VULKAN_DEVICE_FUNCTION(vkCreateSampler)
VULKAN_DEVICE_FUNCTION(vkCreateSemaphore)
VULKAN_DEVICE_FUNCTION(vkCreateShaderModule)
VULKAN_DEVICE_FUNCTION(vkDestroyBuffer)
VULKAN_DEVICE_FUNCTION(vkDestroyCommandPool)
VULKAN_DEVICE_FUNCTION(vkDestroyDescriptorPool)
VULKAN_DEVICE_FUNCTION(vkDestroyDescriptorSetLayout)
VULKAN_DEVICE_FUNCTION(vkDestroyDevice)
VULKAN_DEVICE_FUNCTION(vkDestroyFence)
VULKAN_DEVICE_FUNCTION(vkDestroyFramebuffer)
VULKAN_DEVICE_FUNCTION(vkDestroyImage)
VULKAN_DEVICE_FUNCTION(vkDestroyImageView)
VULKAN_DEVICE_FUNCTION(vkDestroyPipeline)
VULKAN_DEVICE_FUNCTION(vkDestroyPipelineCache)
VULKAN_DEVICE_FUNCTION(vkDestroyPipelineLayout)
VULKAN_DEVICE_FUNCTION(vkDestroyRenderPass)
VULKAN_DEVICE_FUNCTION(vkDestroySampler)
VULKAN_DEVICE_FUNCTION(vkDestroySemaphore)
VULKAN_DEVICE_FUNCTION(vkDestroyShaderModule)
VULKAN_DEVICE_FUNCTION(vkDeviceWaitIdle)
VULKAN_DEVICE_FUNCTION(vkEndCommandBuffer)
VULKAN_DEVICE_FUNCTION(vkFreeCommandBuffers)
VULKAN_DEVICE_FUNCTION(vkFreeMemory)
VULKAN_DEVICE_FUNCTION(vkGetDeviceQueue)
VULKAN_DEVICE_FUNCTION(vkGetPipelineCacheData)
VULKAN_DEVICE_FUNCTION(vkGetFenceStatus)
VULKAN_DEVICE_FUNCTION(vkGetBufferMemoryRequirements)
VULKAN_DEVICE_FUNCTION(vkGetImageMemoryRequirements)
VULKAN_DEVICE_FUNCTION(vkMapMemory)
VULKAN_DEVICE_FUNCTION(vkQueueSubmit)
VULKAN_DEVICE_FUNCTION(vkQueueWaitIdle)
VULKAN_DEVICE_FUNCTION(vkResetCommandBuffer)
VULKAN_DEVICE_FUNCTION(vkResetCommandPool)
VULKAN_DEVICE_FUNCTION(vkResetDescriptorPool)
VULKAN_DEVICE_FUNCTION(vkResetFences)
VULKAN_DEVICE_FUNCTION(vkUnmapMemory)
VULKAN_DEVICE_FUNCTION(vkUpdateDescriptorSets)
VULKAN_DEVICE_FUNCTION(vkWaitForFences)

// VK_KHR_swapchain
VULKAN_DEVICE_FUNCTION(vkAcquireNextImageKHR)
VULKAN_DEVICE_FUNCTION(vkCreateSwapchainKHR)
VULKAN_DEVICE_FUNCTION(vkDestroySwapchainKHR)
VULKAN_DEVICE_FUNCTION(vkQueuePresentKHR)
VULKAN_DEVICE_FUNCTION(vkGetSwapchainImagesKHR)

/*
 * Redefine these every time you include this header!
 */
#undef VULKAN_GLOBAL_FUNCTION
#undef VULKAN_INSTANCE_FUNCTION
#undef VULKAN_DEVICE_FUNCTION
