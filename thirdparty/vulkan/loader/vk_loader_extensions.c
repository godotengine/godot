// *** THIS FILE IS GENERATED - DO NOT EDIT ***
// See loader_extension_generator.py for modifications

/*
 * Copyright (c) 2015-2017 The Khronos Group Inc.
 * Copyright (c) 2015-2017 Valve Corporation
 * Copyright (c) 2015-2017 LunarG, Inc.
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
 * Author: Mark Young <marky@lunarg.com>
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk_loader_platform.h"
#include "loader.h"
#include "vk_loader_extensions.h"
#include <vulkan/vk_icd.h>
#include "wsi.h"
#include "debug_utils.h"
#include "extension_manual.h"

// Device extension error function
VKAPI_ATTR VkResult VKAPI_CALL vkDevExtError(VkDevice dev) {
    struct loader_device *found_dev;
    // The device going in is a trampoline device
    struct loader_icd_term *icd_term = loader_get_icd_and_device(dev, &found_dev, NULL);

    if (icd_term)
        loader_log(icd_term->this_instance, VK_DEBUG_REPORT_ERROR_BIT_EXT, 0,
                   "Bad destination in loader trampoline dispatch,"
                   "Are layers and extensions that you are calling enabled?");
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

VKAPI_ATTR bool VKAPI_CALL loader_icd_init_entries(struct loader_icd_term *icd_term, VkInstance inst,
                                                   const PFN_vkGetInstanceProcAddr fp_gipa) {

#define LOOKUP_GIPA(func, required)                                                        \
    do {                                                                                   \
        icd_term->dispatch.func = (PFN_vk##func)fp_gipa(inst, "vk" #func);                 \
        if (!icd_term->dispatch.func && required) {                                        \
            loader_log((struct loader_instance *)inst, VK_DEBUG_REPORT_WARNING_BIT_EXT, 0, \
                       loader_platform_get_proc_address_error("vk" #func));                \
            return false;                                                                  \
        }                                                                                  \
    } while (0)


    // ---- Core 1_0
    LOOKUP_GIPA(DestroyInstance, true);
    LOOKUP_GIPA(EnumeratePhysicalDevices, true);
    LOOKUP_GIPA(GetPhysicalDeviceFeatures, true);
    LOOKUP_GIPA(GetPhysicalDeviceFormatProperties, true);
    LOOKUP_GIPA(GetPhysicalDeviceImageFormatProperties, true);
    LOOKUP_GIPA(GetPhysicalDeviceProperties, true);
    LOOKUP_GIPA(GetPhysicalDeviceQueueFamilyProperties, true);
    LOOKUP_GIPA(GetPhysicalDeviceMemoryProperties, true);
    LOOKUP_GIPA(GetDeviceProcAddr, true);
    LOOKUP_GIPA(CreateDevice, true);
    LOOKUP_GIPA(EnumerateDeviceExtensionProperties, true);
    LOOKUP_GIPA(GetPhysicalDeviceSparseImageFormatProperties, true);

    // ---- Core 1_1
    LOOKUP_GIPA(EnumeratePhysicalDeviceGroups, false);
    LOOKUP_GIPA(GetPhysicalDeviceFeatures2, false);
    LOOKUP_GIPA(GetPhysicalDeviceProperties2, false);
    LOOKUP_GIPA(GetPhysicalDeviceFormatProperties2, false);
    LOOKUP_GIPA(GetPhysicalDeviceImageFormatProperties2, false);
    LOOKUP_GIPA(GetPhysicalDeviceQueueFamilyProperties2, false);
    LOOKUP_GIPA(GetPhysicalDeviceMemoryProperties2, false);
    LOOKUP_GIPA(GetPhysicalDeviceSparseImageFormatProperties2, false);
    LOOKUP_GIPA(GetPhysicalDeviceExternalBufferProperties, false);
    LOOKUP_GIPA(GetPhysicalDeviceExternalFenceProperties, false);
    LOOKUP_GIPA(GetPhysicalDeviceExternalSemaphoreProperties, false);

    // ---- VK_KHR_surface extension commands
    LOOKUP_GIPA(DestroySurfaceKHR, false);
    LOOKUP_GIPA(GetPhysicalDeviceSurfaceSupportKHR, false);
    LOOKUP_GIPA(GetPhysicalDeviceSurfaceCapabilitiesKHR, false);
    LOOKUP_GIPA(GetPhysicalDeviceSurfaceFormatsKHR, false);
    LOOKUP_GIPA(GetPhysicalDeviceSurfacePresentModesKHR, false);

    // ---- VK_KHR_swapchain extension commands
    LOOKUP_GIPA(CreateSwapchainKHR, false);
    LOOKUP_GIPA(GetDeviceGroupSurfacePresentModesKHR, false);
    LOOKUP_GIPA(GetPhysicalDevicePresentRectanglesKHR, false);

    // ---- VK_KHR_display extension commands
    LOOKUP_GIPA(GetPhysicalDeviceDisplayPropertiesKHR, false);
    LOOKUP_GIPA(GetPhysicalDeviceDisplayPlanePropertiesKHR, false);
    LOOKUP_GIPA(GetDisplayPlaneSupportedDisplaysKHR, false);
    LOOKUP_GIPA(GetDisplayModePropertiesKHR, false);
    LOOKUP_GIPA(CreateDisplayModeKHR, false);
    LOOKUP_GIPA(GetDisplayPlaneCapabilitiesKHR, false);
    LOOKUP_GIPA(CreateDisplayPlaneSurfaceKHR, false);

    // ---- VK_KHR_display_swapchain extension commands
    LOOKUP_GIPA(CreateSharedSwapchainsKHR, false);

    // ---- VK_KHR_xlib_surface extension commands
#ifdef VK_USE_PLATFORM_XLIB_KHR
    LOOKUP_GIPA(CreateXlibSurfaceKHR, false);
#endif // VK_USE_PLATFORM_XLIB_KHR
#ifdef VK_USE_PLATFORM_XLIB_KHR
    LOOKUP_GIPA(GetPhysicalDeviceXlibPresentationSupportKHR, false);
#endif // VK_USE_PLATFORM_XLIB_KHR

    // ---- VK_KHR_xcb_surface extension commands
#ifdef VK_USE_PLATFORM_XCB_KHR
    LOOKUP_GIPA(CreateXcbSurfaceKHR, false);
#endif // VK_USE_PLATFORM_XCB_KHR
#ifdef VK_USE_PLATFORM_XCB_KHR
    LOOKUP_GIPA(GetPhysicalDeviceXcbPresentationSupportKHR, false);
#endif // VK_USE_PLATFORM_XCB_KHR

    // ---- VK_KHR_wayland_surface extension commands
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    LOOKUP_GIPA(CreateWaylandSurfaceKHR, false);
#endif // VK_USE_PLATFORM_WAYLAND_KHR
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    LOOKUP_GIPA(GetPhysicalDeviceWaylandPresentationSupportKHR, false);
#endif // VK_USE_PLATFORM_WAYLAND_KHR

    // ---- VK_KHR_android_surface extension commands
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    LOOKUP_GIPA(CreateAndroidSurfaceKHR, false);
#endif // VK_USE_PLATFORM_ANDROID_KHR

    // ---- VK_KHR_win32_surface extension commands
#ifdef VK_USE_PLATFORM_WIN32_KHR
    LOOKUP_GIPA(CreateWin32SurfaceKHR, false);
#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
    LOOKUP_GIPA(GetPhysicalDeviceWin32PresentationSupportKHR, false);
#endif // VK_USE_PLATFORM_WIN32_KHR

    // ---- VK_KHR_get_physical_device_properties2 extension commands
    LOOKUP_GIPA(GetPhysicalDeviceFeatures2KHR, false);
    LOOKUP_GIPA(GetPhysicalDeviceProperties2KHR, false);
    LOOKUP_GIPA(GetPhysicalDeviceFormatProperties2KHR, false);
    LOOKUP_GIPA(GetPhysicalDeviceImageFormatProperties2KHR, false);
    LOOKUP_GIPA(GetPhysicalDeviceQueueFamilyProperties2KHR, false);
    LOOKUP_GIPA(GetPhysicalDeviceMemoryProperties2KHR, false);
    LOOKUP_GIPA(GetPhysicalDeviceSparseImageFormatProperties2KHR, false);

    // ---- VK_KHR_device_group_creation extension commands
    LOOKUP_GIPA(EnumeratePhysicalDeviceGroupsKHR, false);

    // ---- VK_KHR_external_memory_capabilities extension commands
    LOOKUP_GIPA(GetPhysicalDeviceExternalBufferPropertiesKHR, false);

    // ---- VK_KHR_external_semaphore_capabilities extension commands
    LOOKUP_GIPA(GetPhysicalDeviceExternalSemaphorePropertiesKHR, false);

    // ---- VK_KHR_external_fence_capabilities extension commands
    LOOKUP_GIPA(GetPhysicalDeviceExternalFencePropertiesKHR, false);

    // ---- VK_KHR_performance_query extension commands
    LOOKUP_GIPA(EnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR, false);
    LOOKUP_GIPA(GetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR, false);

    // ---- VK_KHR_get_surface_capabilities2 extension commands
    LOOKUP_GIPA(GetPhysicalDeviceSurfaceCapabilities2KHR, false);
    LOOKUP_GIPA(GetPhysicalDeviceSurfaceFormats2KHR, false);

    // ---- VK_KHR_get_display_properties2 extension commands
    LOOKUP_GIPA(GetPhysicalDeviceDisplayProperties2KHR, false);
    LOOKUP_GIPA(GetPhysicalDeviceDisplayPlaneProperties2KHR, false);
    LOOKUP_GIPA(GetDisplayModeProperties2KHR, false);
    LOOKUP_GIPA(GetDisplayPlaneCapabilities2KHR, false);

    // ---- VK_EXT_debug_report extension commands
    LOOKUP_GIPA(CreateDebugReportCallbackEXT, false);
    LOOKUP_GIPA(DestroyDebugReportCallbackEXT, false);
    LOOKUP_GIPA(DebugReportMessageEXT, false);

    // ---- VK_EXT_debug_marker extension commands
    LOOKUP_GIPA(DebugMarkerSetObjectTagEXT, false);
    LOOKUP_GIPA(DebugMarkerSetObjectNameEXT, false);

    // ---- VK_GGP_stream_descriptor_surface extension commands
#ifdef VK_USE_PLATFORM_GGP
    LOOKUP_GIPA(CreateStreamDescriptorSurfaceGGP, false);
#endif // VK_USE_PLATFORM_GGP

    // ---- VK_NV_external_memory_capabilities extension commands
    LOOKUP_GIPA(GetPhysicalDeviceExternalImageFormatPropertiesNV, false);

    // ---- VK_NN_vi_surface extension commands
#ifdef VK_USE_PLATFORM_VI_NN
    LOOKUP_GIPA(CreateViSurfaceNN, false);
#endif // VK_USE_PLATFORM_VI_NN

    // ---- VK_EXT_direct_mode_display extension commands
    LOOKUP_GIPA(ReleaseDisplayEXT, false);

    // ---- VK_EXT_acquire_xlib_display extension commands
#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT
    LOOKUP_GIPA(AcquireXlibDisplayEXT, false);
#endif // VK_USE_PLATFORM_XLIB_XRANDR_EXT
#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT
    LOOKUP_GIPA(GetRandROutputDisplayEXT, false);
#endif // VK_USE_PLATFORM_XLIB_XRANDR_EXT

    // ---- VK_EXT_display_surface_counter extension commands
    LOOKUP_GIPA(GetPhysicalDeviceSurfaceCapabilities2EXT, false);

    // ---- VK_MVK_ios_surface extension commands
#ifdef VK_USE_PLATFORM_IOS_MVK
    LOOKUP_GIPA(CreateIOSSurfaceMVK, false);
#endif // VK_USE_PLATFORM_IOS_MVK

    // ---- VK_MVK_macos_surface extension commands
#ifdef VK_USE_PLATFORM_MACOS_MVK
    LOOKUP_GIPA(CreateMacOSSurfaceMVK, false);
#endif // VK_USE_PLATFORM_MACOS_MVK

    // ---- VK_EXT_debug_utils extension commands
    LOOKUP_GIPA(SetDebugUtilsObjectNameEXT, false);
    LOOKUP_GIPA(SetDebugUtilsObjectTagEXT, false);
    LOOKUP_GIPA(QueueBeginDebugUtilsLabelEXT, false);
    LOOKUP_GIPA(QueueEndDebugUtilsLabelEXT, false);
    LOOKUP_GIPA(QueueInsertDebugUtilsLabelEXT, false);
    LOOKUP_GIPA(CmdBeginDebugUtilsLabelEXT, false);
    LOOKUP_GIPA(CmdEndDebugUtilsLabelEXT, false);
    LOOKUP_GIPA(CmdInsertDebugUtilsLabelEXT, false);
    LOOKUP_GIPA(CreateDebugUtilsMessengerEXT, false);
    LOOKUP_GIPA(DestroyDebugUtilsMessengerEXT, false);
    LOOKUP_GIPA(SubmitDebugUtilsMessageEXT, false);

    // ---- VK_EXT_sample_locations extension commands
    LOOKUP_GIPA(GetPhysicalDeviceMultisamplePropertiesEXT, false);

    // ---- VK_EXT_calibrated_timestamps extension commands
    LOOKUP_GIPA(GetPhysicalDeviceCalibrateableTimeDomainsEXT, false);

    // ---- VK_FUCHSIA_imagepipe_surface extension commands
#ifdef VK_USE_PLATFORM_FUCHSIA
    LOOKUP_GIPA(CreateImagePipeSurfaceFUCHSIA, false);
#endif // VK_USE_PLATFORM_FUCHSIA

    // ---- VK_EXT_metal_surface extension commands
#ifdef VK_USE_PLATFORM_METAL_EXT
    LOOKUP_GIPA(CreateMetalSurfaceEXT, false);
#endif // VK_USE_PLATFORM_METAL_EXT

    // ---- VK_EXT_tooling_info extension commands
    LOOKUP_GIPA(GetPhysicalDeviceToolPropertiesEXT, false);

    // ---- VK_NV_cooperative_matrix extension commands
    LOOKUP_GIPA(GetPhysicalDeviceCooperativeMatrixPropertiesNV, false);

    // ---- VK_NV_coverage_reduction_mode extension commands
    LOOKUP_GIPA(GetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV, false);

    // ---- VK_EXT_full_screen_exclusive extension commands
#ifdef VK_USE_PLATFORM_WIN32_KHR
    LOOKUP_GIPA(GetPhysicalDeviceSurfacePresentModes2EXT, false);
#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
    LOOKUP_GIPA(GetDeviceGroupSurfacePresentModes2EXT, false);
#endif // VK_USE_PLATFORM_WIN32_KHR

    // ---- VK_EXT_headless_surface extension commands
    LOOKUP_GIPA(CreateHeadlessSurfaceEXT, false);

    // ---- VK_EXT_directfb_surface extension commands
#ifdef VK_USE_PLATFORM_DIRECTFB_EXT
    LOOKUP_GIPA(CreateDirectFBSurfaceEXT, false);
#endif // VK_USE_PLATFORM_DIRECTFB_EXT
#ifdef VK_USE_PLATFORM_DIRECTFB_EXT
    LOOKUP_GIPA(GetPhysicalDeviceDirectFBPresentationSupportEXT, false);
#endif // VK_USE_PLATFORM_DIRECTFB_EXT

#undef LOOKUP_GIPA

    return true;
};

// Init Device function pointer dispatch table with core commands
VKAPI_ATTR void VKAPI_CALL loader_init_device_dispatch_table(struct loader_dev_dispatch_table *dev_table, PFN_vkGetDeviceProcAddr gpa,
                                                             VkDevice dev) {
    VkLayerDispatchTable *table = &dev_table->core_dispatch;
    for (uint32_t i = 0; i < MAX_NUM_UNKNOWN_EXTS; i++) dev_table->ext_dispatch.dev_ext[i] = (PFN_vkDevExt)vkDevExtError;

    // ---- Core 1_0 commands
    table->GetDeviceProcAddr = gpa;
    table->DestroyDevice = (PFN_vkDestroyDevice)gpa(dev, "vkDestroyDevice");
    table->GetDeviceQueue = (PFN_vkGetDeviceQueue)gpa(dev, "vkGetDeviceQueue");
    table->QueueSubmit = (PFN_vkQueueSubmit)gpa(dev, "vkQueueSubmit");
    table->QueueWaitIdle = (PFN_vkQueueWaitIdle)gpa(dev, "vkQueueWaitIdle");
    table->DeviceWaitIdle = (PFN_vkDeviceWaitIdle)gpa(dev, "vkDeviceWaitIdle");
    table->AllocateMemory = (PFN_vkAllocateMemory)gpa(dev, "vkAllocateMemory");
    table->FreeMemory = (PFN_vkFreeMemory)gpa(dev, "vkFreeMemory");
    table->MapMemory = (PFN_vkMapMemory)gpa(dev, "vkMapMemory");
    table->UnmapMemory = (PFN_vkUnmapMemory)gpa(dev, "vkUnmapMemory");
    table->FlushMappedMemoryRanges = (PFN_vkFlushMappedMemoryRanges)gpa(dev, "vkFlushMappedMemoryRanges");
    table->InvalidateMappedMemoryRanges = (PFN_vkInvalidateMappedMemoryRanges)gpa(dev, "vkInvalidateMappedMemoryRanges");
    table->GetDeviceMemoryCommitment = (PFN_vkGetDeviceMemoryCommitment)gpa(dev, "vkGetDeviceMemoryCommitment");
    table->BindBufferMemory = (PFN_vkBindBufferMemory)gpa(dev, "vkBindBufferMemory");
    table->BindImageMemory = (PFN_vkBindImageMemory)gpa(dev, "vkBindImageMemory");
    table->GetBufferMemoryRequirements = (PFN_vkGetBufferMemoryRequirements)gpa(dev, "vkGetBufferMemoryRequirements");
    table->GetImageMemoryRequirements = (PFN_vkGetImageMemoryRequirements)gpa(dev, "vkGetImageMemoryRequirements");
    table->GetImageSparseMemoryRequirements = (PFN_vkGetImageSparseMemoryRequirements)gpa(dev, "vkGetImageSparseMemoryRequirements");
    table->QueueBindSparse = (PFN_vkQueueBindSparse)gpa(dev, "vkQueueBindSparse");
    table->CreateFence = (PFN_vkCreateFence)gpa(dev, "vkCreateFence");
    table->DestroyFence = (PFN_vkDestroyFence)gpa(dev, "vkDestroyFence");
    table->ResetFences = (PFN_vkResetFences)gpa(dev, "vkResetFences");
    table->GetFenceStatus = (PFN_vkGetFenceStatus)gpa(dev, "vkGetFenceStatus");
    table->WaitForFences = (PFN_vkWaitForFences)gpa(dev, "vkWaitForFences");
    table->CreateSemaphore = (PFN_vkCreateSemaphore)gpa(dev, "vkCreateSemaphore");
    table->DestroySemaphore = (PFN_vkDestroySemaphore)gpa(dev, "vkDestroySemaphore");
    table->CreateEvent = (PFN_vkCreateEvent)gpa(dev, "vkCreateEvent");
    table->DestroyEvent = (PFN_vkDestroyEvent)gpa(dev, "vkDestroyEvent");
    table->GetEventStatus = (PFN_vkGetEventStatus)gpa(dev, "vkGetEventStatus");
    table->SetEvent = (PFN_vkSetEvent)gpa(dev, "vkSetEvent");
    table->ResetEvent = (PFN_vkResetEvent)gpa(dev, "vkResetEvent");
    table->CreateQueryPool = (PFN_vkCreateQueryPool)gpa(dev, "vkCreateQueryPool");
    table->DestroyQueryPool = (PFN_vkDestroyQueryPool)gpa(dev, "vkDestroyQueryPool");
    table->GetQueryPoolResults = (PFN_vkGetQueryPoolResults)gpa(dev, "vkGetQueryPoolResults");
    table->CreateBuffer = (PFN_vkCreateBuffer)gpa(dev, "vkCreateBuffer");
    table->DestroyBuffer = (PFN_vkDestroyBuffer)gpa(dev, "vkDestroyBuffer");
    table->CreateBufferView = (PFN_vkCreateBufferView)gpa(dev, "vkCreateBufferView");
    table->DestroyBufferView = (PFN_vkDestroyBufferView)gpa(dev, "vkDestroyBufferView");
    table->CreateImage = (PFN_vkCreateImage)gpa(dev, "vkCreateImage");
    table->DestroyImage = (PFN_vkDestroyImage)gpa(dev, "vkDestroyImage");
    table->GetImageSubresourceLayout = (PFN_vkGetImageSubresourceLayout)gpa(dev, "vkGetImageSubresourceLayout");
    table->CreateImageView = (PFN_vkCreateImageView)gpa(dev, "vkCreateImageView");
    table->DestroyImageView = (PFN_vkDestroyImageView)gpa(dev, "vkDestroyImageView");
    table->CreateShaderModule = (PFN_vkCreateShaderModule)gpa(dev, "vkCreateShaderModule");
    table->DestroyShaderModule = (PFN_vkDestroyShaderModule)gpa(dev, "vkDestroyShaderModule");
    table->CreatePipelineCache = (PFN_vkCreatePipelineCache)gpa(dev, "vkCreatePipelineCache");
    table->DestroyPipelineCache = (PFN_vkDestroyPipelineCache)gpa(dev, "vkDestroyPipelineCache");
    table->GetPipelineCacheData = (PFN_vkGetPipelineCacheData)gpa(dev, "vkGetPipelineCacheData");
    table->MergePipelineCaches = (PFN_vkMergePipelineCaches)gpa(dev, "vkMergePipelineCaches");
    table->CreateGraphicsPipelines = (PFN_vkCreateGraphicsPipelines)gpa(dev, "vkCreateGraphicsPipelines");
    table->CreateComputePipelines = (PFN_vkCreateComputePipelines)gpa(dev, "vkCreateComputePipelines");
    table->DestroyPipeline = (PFN_vkDestroyPipeline)gpa(dev, "vkDestroyPipeline");
    table->CreatePipelineLayout = (PFN_vkCreatePipelineLayout)gpa(dev, "vkCreatePipelineLayout");
    table->DestroyPipelineLayout = (PFN_vkDestroyPipelineLayout)gpa(dev, "vkDestroyPipelineLayout");
    table->CreateSampler = (PFN_vkCreateSampler)gpa(dev, "vkCreateSampler");
    table->DestroySampler = (PFN_vkDestroySampler)gpa(dev, "vkDestroySampler");
    table->CreateDescriptorSetLayout = (PFN_vkCreateDescriptorSetLayout)gpa(dev, "vkCreateDescriptorSetLayout");
    table->DestroyDescriptorSetLayout = (PFN_vkDestroyDescriptorSetLayout)gpa(dev, "vkDestroyDescriptorSetLayout");
    table->CreateDescriptorPool = (PFN_vkCreateDescriptorPool)gpa(dev, "vkCreateDescriptorPool");
    table->DestroyDescriptorPool = (PFN_vkDestroyDescriptorPool)gpa(dev, "vkDestroyDescriptorPool");
    table->ResetDescriptorPool = (PFN_vkResetDescriptorPool)gpa(dev, "vkResetDescriptorPool");
    table->AllocateDescriptorSets = (PFN_vkAllocateDescriptorSets)gpa(dev, "vkAllocateDescriptorSets");
    table->FreeDescriptorSets = (PFN_vkFreeDescriptorSets)gpa(dev, "vkFreeDescriptorSets");
    table->UpdateDescriptorSets = (PFN_vkUpdateDescriptorSets)gpa(dev, "vkUpdateDescriptorSets");
    table->CreateFramebuffer = (PFN_vkCreateFramebuffer)gpa(dev, "vkCreateFramebuffer");
    table->DestroyFramebuffer = (PFN_vkDestroyFramebuffer)gpa(dev, "vkDestroyFramebuffer");
    table->CreateRenderPass = (PFN_vkCreateRenderPass)gpa(dev, "vkCreateRenderPass");
    table->DestroyRenderPass = (PFN_vkDestroyRenderPass)gpa(dev, "vkDestroyRenderPass");
    table->GetRenderAreaGranularity = (PFN_vkGetRenderAreaGranularity)gpa(dev, "vkGetRenderAreaGranularity");
    table->CreateCommandPool = (PFN_vkCreateCommandPool)gpa(dev, "vkCreateCommandPool");
    table->DestroyCommandPool = (PFN_vkDestroyCommandPool)gpa(dev, "vkDestroyCommandPool");
    table->ResetCommandPool = (PFN_vkResetCommandPool)gpa(dev, "vkResetCommandPool");
    table->AllocateCommandBuffers = (PFN_vkAllocateCommandBuffers)gpa(dev, "vkAllocateCommandBuffers");
    table->FreeCommandBuffers = (PFN_vkFreeCommandBuffers)gpa(dev, "vkFreeCommandBuffers");
    table->BeginCommandBuffer = (PFN_vkBeginCommandBuffer)gpa(dev, "vkBeginCommandBuffer");
    table->EndCommandBuffer = (PFN_vkEndCommandBuffer)gpa(dev, "vkEndCommandBuffer");
    table->ResetCommandBuffer = (PFN_vkResetCommandBuffer)gpa(dev, "vkResetCommandBuffer");
    table->CmdBindPipeline = (PFN_vkCmdBindPipeline)gpa(dev, "vkCmdBindPipeline");
    table->CmdSetViewport = (PFN_vkCmdSetViewport)gpa(dev, "vkCmdSetViewport");
    table->CmdSetScissor = (PFN_vkCmdSetScissor)gpa(dev, "vkCmdSetScissor");
    table->CmdSetLineWidth = (PFN_vkCmdSetLineWidth)gpa(dev, "vkCmdSetLineWidth");
    table->CmdSetDepthBias = (PFN_vkCmdSetDepthBias)gpa(dev, "vkCmdSetDepthBias");
    table->CmdSetBlendConstants = (PFN_vkCmdSetBlendConstants)gpa(dev, "vkCmdSetBlendConstants");
    table->CmdSetDepthBounds = (PFN_vkCmdSetDepthBounds)gpa(dev, "vkCmdSetDepthBounds");
    table->CmdSetStencilCompareMask = (PFN_vkCmdSetStencilCompareMask)gpa(dev, "vkCmdSetStencilCompareMask");
    table->CmdSetStencilWriteMask = (PFN_vkCmdSetStencilWriteMask)gpa(dev, "vkCmdSetStencilWriteMask");
    table->CmdSetStencilReference = (PFN_vkCmdSetStencilReference)gpa(dev, "vkCmdSetStencilReference");
    table->CmdBindDescriptorSets = (PFN_vkCmdBindDescriptorSets)gpa(dev, "vkCmdBindDescriptorSets");
    table->CmdBindIndexBuffer = (PFN_vkCmdBindIndexBuffer)gpa(dev, "vkCmdBindIndexBuffer");
    table->CmdBindVertexBuffers = (PFN_vkCmdBindVertexBuffers)gpa(dev, "vkCmdBindVertexBuffers");
    table->CmdDraw = (PFN_vkCmdDraw)gpa(dev, "vkCmdDraw");
    table->CmdDrawIndexed = (PFN_vkCmdDrawIndexed)gpa(dev, "vkCmdDrawIndexed");
    table->CmdDrawIndirect = (PFN_vkCmdDrawIndirect)gpa(dev, "vkCmdDrawIndirect");
    table->CmdDrawIndexedIndirect = (PFN_vkCmdDrawIndexedIndirect)gpa(dev, "vkCmdDrawIndexedIndirect");
    table->CmdDispatch = (PFN_vkCmdDispatch)gpa(dev, "vkCmdDispatch");
    table->CmdDispatchIndirect = (PFN_vkCmdDispatchIndirect)gpa(dev, "vkCmdDispatchIndirect");
    table->CmdCopyBuffer = (PFN_vkCmdCopyBuffer)gpa(dev, "vkCmdCopyBuffer");
    table->CmdCopyImage = (PFN_vkCmdCopyImage)gpa(dev, "vkCmdCopyImage");
    table->CmdBlitImage = (PFN_vkCmdBlitImage)gpa(dev, "vkCmdBlitImage");
    table->CmdCopyBufferToImage = (PFN_vkCmdCopyBufferToImage)gpa(dev, "vkCmdCopyBufferToImage");
    table->CmdCopyImageToBuffer = (PFN_vkCmdCopyImageToBuffer)gpa(dev, "vkCmdCopyImageToBuffer");
    table->CmdUpdateBuffer = (PFN_vkCmdUpdateBuffer)gpa(dev, "vkCmdUpdateBuffer");
    table->CmdFillBuffer = (PFN_vkCmdFillBuffer)gpa(dev, "vkCmdFillBuffer");
    table->CmdClearColorImage = (PFN_vkCmdClearColorImage)gpa(dev, "vkCmdClearColorImage");
    table->CmdClearDepthStencilImage = (PFN_vkCmdClearDepthStencilImage)gpa(dev, "vkCmdClearDepthStencilImage");
    table->CmdClearAttachments = (PFN_vkCmdClearAttachments)gpa(dev, "vkCmdClearAttachments");
    table->CmdResolveImage = (PFN_vkCmdResolveImage)gpa(dev, "vkCmdResolveImage");
    table->CmdSetEvent = (PFN_vkCmdSetEvent)gpa(dev, "vkCmdSetEvent");
    table->CmdResetEvent = (PFN_vkCmdResetEvent)gpa(dev, "vkCmdResetEvent");
    table->CmdWaitEvents = (PFN_vkCmdWaitEvents)gpa(dev, "vkCmdWaitEvents");
    table->CmdPipelineBarrier = (PFN_vkCmdPipelineBarrier)gpa(dev, "vkCmdPipelineBarrier");
    table->CmdBeginQuery = (PFN_vkCmdBeginQuery)gpa(dev, "vkCmdBeginQuery");
    table->CmdEndQuery = (PFN_vkCmdEndQuery)gpa(dev, "vkCmdEndQuery");
    table->CmdResetQueryPool = (PFN_vkCmdResetQueryPool)gpa(dev, "vkCmdResetQueryPool");
    table->CmdWriteTimestamp = (PFN_vkCmdWriteTimestamp)gpa(dev, "vkCmdWriteTimestamp");
    table->CmdCopyQueryPoolResults = (PFN_vkCmdCopyQueryPoolResults)gpa(dev, "vkCmdCopyQueryPoolResults");
    table->CmdPushConstants = (PFN_vkCmdPushConstants)gpa(dev, "vkCmdPushConstants");
    table->CmdBeginRenderPass = (PFN_vkCmdBeginRenderPass)gpa(dev, "vkCmdBeginRenderPass");
    table->CmdNextSubpass = (PFN_vkCmdNextSubpass)gpa(dev, "vkCmdNextSubpass");
    table->CmdEndRenderPass = (PFN_vkCmdEndRenderPass)gpa(dev, "vkCmdEndRenderPass");
    table->CmdExecuteCommands = (PFN_vkCmdExecuteCommands)gpa(dev, "vkCmdExecuteCommands");

    // ---- Core 1_1 commands
    table->BindBufferMemory2 = (PFN_vkBindBufferMemory2)gpa(dev, "vkBindBufferMemory2");
    table->BindImageMemory2 = (PFN_vkBindImageMemory2)gpa(dev, "vkBindImageMemory2");
    table->GetDeviceGroupPeerMemoryFeatures = (PFN_vkGetDeviceGroupPeerMemoryFeatures)gpa(dev, "vkGetDeviceGroupPeerMemoryFeatures");
    table->CmdSetDeviceMask = (PFN_vkCmdSetDeviceMask)gpa(dev, "vkCmdSetDeviceMask");
    table->CmdDispatchBase = (PFN_vkCmdDispatchBase)gpa(dev, "vkCmdDispatchBase");
    table->GetImageMemoryRequirements2 = (PFN_vkGetImageMemoryRequirements2)gpa(dev, "vkGetImageMemoryRequirements2");
    table->GetBufferMemoryRequirements2 = (PFN_vkGetBufferMemoryRequirements2)gpa(dev, "vkGetBufferMemoryRequirements2");
    table->GetImageSparseMemoryRequirements2 = (PFN_vkGetImageSparseMemoryRequirements2)gpa(dev, "vkGetImageSparseMemoryRequirements2");
    table->TrimCommandPool = (PFN_vkTrimCommandPool)gpa(dev, "vkTrimCommandPool");
    table->GetDeviceQueue2 = (PFN_vkGetDeviceQueue2)gpa(dev, "vkGetDeviceQueue2");
    table->CreateSamplerYcbcrConversion = (PFN_vkCreateSamplerYcbcrConversion)gpa(dev, "vkCreateSamplerYcbcrConversion");
    table->DestroySamplerYcbcrConversion = (PFN_vkDestroySamplerYcbcrConversion)gpa(dev, "vkDestroySamplerYcbcrConversion");
    table->CreateDescriptorUpdateTemplate = (PFN_vkCreateDescriptorUpdateTemplate)gpa(dev, "vkCreateDescriptorUpdateTemplate");
    table->DestroyDescriptorUpdateTemplate = (PFN_vkDestroyDescriptorUpdateTemplate)gpa(dev, "vkDestroyDescriptorUpdateTemplate");
    table->UpdateDescriptorSetWithTemplate = (PFN_vkUpdateDescriptorSetWithTemplate)gpa(dev, "vkUpdateDescriptorSetWithTemplate");
    table->GetDescriptorSetLayoutSupport = (PFN_vkGetDescriptorSetLayoutSupport)gpa(dev, "vkGetDescriptorSetLayoutSupport");

    // ---- Core 1_2 commands
    table->CmdDrawIndirectCount = (PFN_vkCmdDrawIndirectCount)gpa(dev, "vkCmdDrawIndirectCount");
    table->CmdDrawIndexedIndirectCount = (PFN_vkCmdDrawIndexedIndirectCount)gpa(dev, "vkCmdDrawIndexedIndirectCount");
    table->CreateRenderPass2 = (PFN_vkCreateRenderPass2)gpa(dev, "vkCreateRenderPass2");
    table->CmdBeginRenderPass2 = (PFN_vkCmdBeginRenderPass2)gpa(dev, "vkCmdBeginRenderPass2");
    table->CmdNextSubpass2 = (PFN_vkCmdNextSubpass2)gpa(dev, "vkCmdNextSubpass2");
    table->CmdEndRenderPass2 = (PFN_vkCmdEndRenderPass2)gpa(dev, "vkCmdEndRenderPass2");
    table->ResetQueryPool = (PFN_vkResetQueryPool)gpa(dev, "vkResetQueryPool");
    table->GetSemaphoreCounterValue = (PFN_vkGetSemaphoreCounterValue)gpa(dev, "vkGetSemaphoreCounterValue");
    table->WaitSemaphores = (PFN_vkWaitSemaphores)gpa(dev, "vkWaitSemaphores");
    table->SignalSemaphore = (PFN_vkSignalSemaphore)gpa(dev, "vkSignalSemaphore");
    table->GetBufferDeviceAddress = (PFN_vkGetBufferDeviceAddress)gpa(dev, "vkGetBufferDeviceAddress");
    table->GetBufferOpaqueCaptureAddress = (PFN_vkGetBufferOpaqueCaptureAddress)gpa(dev, "vkGetBufferOpaqueCaptureAddress");
    table->GetDeviceMemoryOpaqueCaptureAddress = (PFN_vkGetDeviceMemoryOpaqueCaptureAddress)gpa(dev, "vkGetDeviceMemoryOpaqueCaptureAddress");
}

// Init Device function pointer dispatch table with extension commands
VKAPI_ATTR void VKAPI_CALL loader_init_device_extension_dispatch_table(struct loader_dev_dispatch_table *dev_table,
                                                                       PFN_vkGetInstanceProcAddr gipa,
                                                                       PFN_vkGetDeviceProcAddr gdpa,
                                                                       VkInstance inst,
                                                                       VkDevice dev) {
    VkLayerDispatchTable *table = &dev_table->core_dispatch;

    // ---- VK_KHR_swapchain extension commands
    table->CreateSwapchainKHR = (PFN_vkCreateSwapchainKHR)gdpa(dev, "vkCreateSwapchainKHR");
    table->DestroySwapchainKHR = (PFN_vkDestroySwapchainKHR)gdpa(dev, "vkDestroySwapchainKHR");
    table->GetSwapchainImagesKHR = (PFN_vkGetSwapchainImagesKHR)gdpa(dev, "vkGetSwapchainImagesKHR");
    table->AcquireNextImageKHR = (PFN_vkAcquireNextImageKHR)gdpa(dev, "vkAcquireNextImageKHR");
    table->QueuePresentKHR = (PFN_vkQueuePresentKHR)gdpa(dev, "vkQueuePresentKHR");
    table->GetDeviceGroupPresentCapabilitiesKHR = (PFN_vkGetDeviceGroupPresentCapabilitiesKHR)gdpa(dev, "vkGetDeviceGroupPresentCapabilitiesKHR");
    table->GetDeviceGroupSurfacePresentModesKHR = (PFN_vkGetDeviceGroupSurfacePresentModesKHR)gdpa(dev, "vkGetDeviceGroupSurfacePresentModesKHR");
    table->AcquireNextImage2KHR = (PFN_vkAcquireNextImage2KHR)gdpa(dev, "vkAcquireNextImage2KHR");

    // ---- VK_KHR_display_swapchain extension commands
    table->CreateSharedSwapchainsKHR = (PFN_vkCreateSharedSwapchainsKHR)gdpa(dev, "vkCreateSharedSwapchainsKHR");

    // ---- VK_KHR_device_group extension commands
    table->GetDeviceGroupPeerMemoryFeaturesKHR = (PFN_vkGetDeviceGroupPeerMemoryFeaturesKHR)gdpa(dev, "vkGetDeviceGroupPeerMemoryFeaturesKHR");
    table->CmdSetDeviceMaskKHR = (PFN_vkCmdSetDeviceMaskKHR)gdpa(dev, "vkCmdSetDeviceMaskKHR");
    table->CmdDispatchBaseKHR = (PFN_vkCmdDispatchBaseKHR)gdpa(dev, "vkCmdDispatchBaseKHR");

    // ---- VK_KHR_maintenance1 extension commands
    table->TrimCommandPoolKHR = (PFN_vkTrimCommandPoolKHR)gdpa(dev, "vkTrimCommandPoolKHR");

    // ---- VK_KHR_external_memory_win32 extension commands
#ifdef VK_USE_PLATFORM_WIN32_KHR
    table->GetMemoryWin32HandleKHR = (PFN_vkGetMemoryWin32HandleKHR)gdpa(dev, "vkGetMemoryWin32HandleKHR");
#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
    table->GetMemoryWin32HandlePropertiesKHR = (PFN_vkGetMemoryWin32HandlePropertiesKHR)gdpa(dev, "vkGetMemoryWin32HandlePropertiesKHR");
#endif // VK_USE_PLATFORM_WIN32_KHR

    // ---- VK_KHR_external_memory_fd extension commands
    table->GetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)gdpa(dev, "vkGetMemoryFdKHR");
    table->GetMemoryFdPropertiesKHR = (PFN_vkGetMemoryFdPropertiesKHR)gdpa(dev, "vkGetMemoryFdPropertiesKHR");

    // ---- VK_KHR_external_semaphore_win32 extension commands
#ifdef VK_USE_PLATFORM_WIN32_KHR
    table->ImportSemaphoreWin32HandleKHR = (PFN_vkImportSemaphoreWin32HandleKHR)gdpa(dev, "vkImportSemaphoreWin32HandleKHR");
#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
    table->GetSemaphoreWin32HandleKHR = (PFN_vkGetSemaphoreWin32HandleKHR)gdpa(dev, "vkGetSemaphoreWin32HandleKHR");
#endif // VK_USE_PLATFORM_WIN32_KHR

    // ---- VK_KHR_external_semaphore_fd extension commands
    table->ImportSemaphoreFdKHR = (PFN_vkImportSemaphoreFdKHR)gdpa(dev, "vkImportSemaphoreFdKHR");
    table->GetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)gdpa(dev, "vkGetSemaphoreFdKHR");

    // ---- VK_KHR_push_descriptor extension commands
    table->CmdPushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)gdpa(dev, "vkCmdPushDescriptorSetKHR");
    table->CmdPushDescriptorSetWithTemplateKHR = (PFN_vkCmdPushDescriptorSetWithTemplateKHR)gdpa(dev, "vkCmdPushDescriptorSetWithTemplateKHR");

    // ---- VK_KHR_descriptor_update_template extension commands
    table->CreateDescriptorUpdateTemplateKHR = (PFN_vkCreateDescriptorUpdateTemplateKHR)gdpa(dev, "vkCreateDescriptorUpdateTemplateKHR");
    table->DestroyDescriptorUpdateTemplateKHR = (PFN_vkDestroyDescriptorUpdateTemplateKHR)gdpa(dev, "vkDestroyDescriptorUpdateTemplateKHR");
    table->UpdateDescriptorSetWithTemplateKHR = (PFN_vkUpdateDescriptorSetWithTemplateKHR)gdpa(dev, "vkUpdateDescriptorSetWithTemplateKHR");

    // ---- VK_KHR_create_renderpass2 extension commands
    table->CreateRenderPass2KHR = (PFN_vkCreateRenderPass2KHR)gdpa(dev, "vkCreateRenderPass2KHR");
    table->CmdBeginRenderPass2KHR = (PFN_vkCmdBeginRenderPass2KHR)gdpa(dev, "vkCmdBeginRenderPass2KHR");
    table->CmdNextSubpass2KHR = (PFN_vkCmdNextSubpass2KHR)gdpa(dev, "vkCmdNextSubpass2KHR");
    table->CmdEndRenderPass2KHR = (PFN_vkCmdEndRenderPass2KHR)gdpa(dev, "vkCmdEndRenderPass2KHR");

    // ---- VK_KHR_shared_presentable_image extension commands
    table->GetSwapchainStatusKHR = (PFN_vkGetSwapchainStatusKHR)gdpa(dev, "vkGetSwapchainStatusKHR");

    // ---- VK_KHR_external_fence_win32 extension commands
#ifdef VK_USE_PLATFORM_WIN32_KHR
    table->ImportFenceWin32HandleKHR = (PFN_vkImportFenceWin32HandleKHR)gdpa(dev, "vkImportFenceWin32HandleKHR");
#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
    table->GetFenceWin32HandleKHR = (PFN_vkGetFenceWin32HandleKHR)gdpa(dev, "vkGetFenceWin32HandleKHR");
#endif // VK_USE_PLATFORM_WIN32_KHR

    // ---- VK_KHR_external_fence_fd extension commands
    table->ImportFenceFdKHR = (PFN_vkImportFenceFdKHR)gdpa(dev, "vkImportFenceFdKHR");
    table->GetFenceFdKHR = (PFN_vkGetFenceFdKHR)gdpa(dev, "vkGetFenceFdKHR");

    // ---- VK_KHR_performance_query extension commands
    table->AcquireProfilingLockKHR = (PFN_vkAcquireProfilingLockKHR)gdpa(dev, "vkAcquireProfilingLockKHR");
    table->ReleaseProfilingLockKHR = (PFN_vkReleaseProfilingLockKHR)gdpa(dev, "vkReleaseProfilingLockKHR");

    // ---- VK_KHR_get_memory_requirements2 extension commands
    table->GetImageMemoryRequirements2KHR = (PFN_vkGetImageMemoryRequirements2KHR)gdpa(dev, "vkGetImageMemoryRequirements2KHR");
    table->GetBufferMemoryRequirements2KHR = (PFN_vkGetBufferMemoryRequirements2KHR)gdpa(dev, "vkGetBufferMemoryRequirements2KHR");
    table->GetImageSparseMemoryRequirements2KHR = (PFN_vkGetImageSparseMemoryRequirements2KHR)gdpa(dev, "vkGetImageSparseMemoryRequirements2KHR");

    // ---- VK_KHR_sampler_ycbcr_conversion extension commands
    table->CreateSamplerYcbcrConversionKHR = (PFN_vkCreateSamplerYcbcrConversionKHR)gdpa(dev, "vkCreateSamplerYcbcrConversionKHR");
    table->DestroySamplerYcbcrConversionKHR = (PFN_vkDestroySamplerYcbcrConversionKHR)gdpa(dev, "vkDestroySamplerYcbcrConversionKHR");

    // ---- VK_KHR_bind_memory2 extension commands
    table->BindBufferMemory2KHR = (PFN_vkBindBufferMemory2KHR)gdpa(dev, "vkBindBufferMemory2KHR");
    table->BindImageMemory2KHR = (PFN_vkBindImageMemory2KHR)gdpa(dev, "vkBindImageMemory2KHR");

    // ---- VK_KHR_maintenance3 extension commands
    table->GetDescriptorSetLayoutSupportKHR = (PFN_vkGetDescriptorSetLayoutSupportKHR)gdpa(dev, "vkGetDescriptorSetLayoutSupportKHR");

    // ---- VK_KHR_draw_indirect_count extension commands
    table->CmdDrawIndirectCountKHR = (PFN_vkCmdDrawIndirectCountKHR)gdpa(dev, "vkCmdDrawIndirectCountKHR");
    table->CmdDrawIndexedIndirectCountKHR = (PFN_vkCmdDrawIndexedIndirectCountKHR)gdpa(dev, "vkCmdDrawIndexedIndirectCountKHR");

    // ---- VK_KHR_timeline_semaphore extension commands
    table->GetSemaphoreCounterValueKHR = (PFN_vkGetSemaphoreCounterValueKHR)gdpa(dev, "vkGetSemaphoreCounterValueKHR");
    table->WaitSemaphoresKHR = (PFN_vkWaitSemaphoresKHR)gdpa(dev, "vkWaitSemaphoresKHR");
    table->SignalSemaphoreKHR = (PFN_vkSignalSemaphoreKHR)gdpa(dev, "vkSignalSemaphoreKHR");

    // ---- VK_KHR_buffer_device_address extension commands
    table->GetBufferDeviceAddressKHR = (PFN_vkGetBufferDeviceAddressKHR)gdpa(dev, "vkGetBufferDeviceAddressKHR");
    table->GetBufferOpaqueCaptureAddressKHR = (PFN_vkGetBufferOpaqueCaptureAddressKHR)gdpa(dev, "vkGetBufferOpaqueCaptureAddressKHR");
    table->GetDeviceMemoryOpaqueCaptureAddressKHR = (PFN_vkGetDeviceMemoryOpaqueCaptureAddressKHR)gdpa(dev, "vkGetDeviceMemoryOpaqueCaptureAddressKHR");

    // ---- VK_KHR_deferred_host_operations extension commands
#ifdef VK_ENABLE_BETA_EXTENSIONS
    table->CreateDeferredOperationKHR = (PFN_vkCreateDeferredOperationKHR)gdpa(dev, "vkCreateDeferredOperationKHR");
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    table->DestroyDeferredOperationKHR = (PFN_vkDestroyDeferredOperationKHR)gdpa(dev, "vkDestroyDeferredOperationKHR");
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    table->GetDeferredOperationMaxConcurrencyKHR = (PFN_vkGetDeferredOperationMaxConcurrencyKHR)gdpa(dev, "vkGetDeferredOperationMaxConcurrencyKHR");
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    table->GetDeferredOperationResultKHR = (PFN_vkGetDeferredOperationResultKHR)gdpa(dev, "vkGetDeferredOperationResultKHR");
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    table->DeferredOperationJoinKHR = (PFN_vkDeferredOperationJoinKHR)gdpa(dev, "vkDeferredOperationJoinKHR");
#endif // VK_ENABLE_BETA_EXTENSIONS

    // ---- VK_KHR_pipeline_executable_properties extension commands
    table->GetPipelineExecutablePropertiesKHR = (PFN_vkGetPipelineExecutablePropertiesKHR)gdpa(dev, "vkGetPipelineExecutablePropertiesKHR");
    table->GetPipelineExecutableStatisticsKHR = (PFN_vkGetPipelineExecutableStatisticsKHR)gdpa(dev, "vkGetPipelineExecutableStatisticsKHR");
    table->GetPipelineExecutableInternalRepresentationsKHR = (PFN_vkGetPipelineExecutableInternalRepresentationsKHR)gdpa(dev, "vkGetPipelineExecutableInternalRepresentationsKHR");

    // ---- VK_KHR_copy_commands2 extension commands
    table->CmdCopyBuffer2KHR = (PFN_vkCmdCopyBuffer2KHR)gdpa(dev, "vkCmdCopyBuffer2KHR");
    table->CmdCopyImage2KHR = (PFN_vkCmdCopyImage2KHR)gdpa(dev, "vkCmdCopyImage2KHR");
    table->CmdCopyBufferToImage2KHR = (PFN_vkCmdCopyBufferToImage2KHR)gdpa(dev, "vkCmdCopyBufferToImage2KHR");
    table->CmdCopyImageToBuffer2KHR = (PFN_vkCmdCopyImageToBuffer2KHR)gdpa(dev, "vkCmdCopyImageToBuffer2KHR");
    table->CmdBlitImage2KHR = (PFN_vkCmdBlitImage2KHR)gdpa(dev, "vkCmdBlitImage2KHR");
    table->CmdResolveImage2KHR = (PFN_vkCmdResolveImage2KHR)gdpa(dev, "vkCmdResolveImage2KHR");

    // ---- VK_EXT_debug_marker extension commands
    table->DebugMarkerSetObjectTagEXT = (PFN_vkDebugMarkerSetObjectTagEXT)gdpa(dev, "vkDebugMarkerSetObjectTagEXT");
    table->DebugMarkerSetObjectNameEXT = (PFN_vkDebugMarkerSetObjectNameEXT)gdpa(dev, "vkDebugMarkerSetObjectNameEXT");
    table->CmdDebugMarkerBeginEXT = (PFN_vkCmdDebugMarkerBeginEXT)gdpa(dev, "vkCmdDebugMarkerBeginEXT");
    table->CmdDebugMarkerEndEXT = (PFN_vkCmdDebugMarkerEndEXT)gdpa(dev, "vkCmdDebugMarkerEndEXT");
    table->CmdDebugMarkerInsertEXT = (PFN_vkCmdDebugMarkerInsertEXT)gdpa(dev, "vkCmdDebugMarkerInsertEXT");

    // ---- VK_EXT_transform_feedback extension commands
    table->CmdBindTransformFeedbackBuffersEXT = (PFN_vkCmdBindTransformFeedbackBuffersEXT)gdpa(dev, "vkCmdBindTransformFeedbackBuffersEXT");
    table->CmdBeginTransformFeedbackEXT = (PFN_vkCmdBeginTransformFeedbackEXT)gdpa(dev, "vkCmdBeginTransformFeedbackEXT");
    table->CmdEndTransformFeedbackEXT = (PFN_vkCmdEndTransformFeedbackEXT)gdpa(dev, "vkCmdEndTransformFeedbackEXT");
    table->CmdBeginQueryIndexedEXT = (PFN_vkCmdBeginQueryIndexedEXT)gdpa(dev, "vkCmdBeginQueryIndexedEXT");
    table->CmdEndQueryIndexedEXT = (PFN_vkCmdEndQueryIndexedEXT)gdpa(dev, "vkCmdEndQueryIndexedEXT");
    table->CmdDrawIndirectByteCountEXT = (PFN_vkCmdDrawIndirectByteCountEXT)gdpa(dev, "vkCmdDrawIndirectByteCountEXT");

    // ---- VK_NVX_image_view_handle extension commands
    table->GetImageViewHandleNVX = (PFN_vkGetImageViewHandleNVX)gdpa(dev, "vkGetImageViewHandleNVX");
    table->GetImageViewAddressNVX = (PFN_vkGetImageViewAddressNVX)gdpa(dev, "vkGetImageViewAddressNVX");

    // ---- VK_AMD_draw_indirect_count extension commands
    table->CmdDrawIndirectCountAMD = (PFN_vkCmdDrawIndirectCountAMD)gdpa(dev, "vkCmdDrawIndirectCountAMD");
    table->CmdDrawIndexedIndirectCountAMD = (PFN_vkCmdDrawIndexedIndirectCountAMD)gdpa(dev, "vkCmdDrawIndexedIndirectCountAMD");

    // ---- VK_AMD_shader_info extension commands
    table->GetShaderInfoAMD = (PFN_vkGetShaderInfoAMD)gdpa(dev, "vkGetShaderInfoAMD");

    // ---- VK_NV_external_memory_win32 extension commands
#ifdef VK_USE_PLATFORM_WIN32_KHR
    table->GetMemoryWin32HandleNV = (PFN_vkGetMemoryWin32HandleNV)gdpa(dev, "vkGetMemoryWin32HandleNV");
#endif // VK_USE_PLATFORM_WIN32_KHR

    // ---- VK_EXT_conditional_rendering extension commands
    table->CmdBeginConditionalRenderingEXT = (PFN_vkCmdBeginConditionalRenderingEXT)gdpa(dev, "vkCmdBeginConditionalRenderingEXT");
    table->CmdEndConditionalRenderingEXT = (PFN_vkCmdEndConditionalRenderingEXT)gdpa(dev, "vkCmdEndConditionalRenderingEXT");

    // ---- VK_NV_clip_space_w_scaling extension commands
    table->CmdSetViewportWScalingNV = (PFN_vkCmdSetViewportWScalingNV)gdpa(dev, "vkCmdSetViewportWScalingNV");

    // ---- VK_EXT_display_control extension commands
    table->DisplayPowerControlEXT = (PFN_vkDisplayPowerControlEXT)gdpa(dev, "vkDisplayPowerControlEXT");
    table->RegisterDeviceEventEXT = (PFN_vkRegisterDeviceEventEXT)gdpa(dev, "vkRegisterDeviceEventEXT");
    table->RegisterDisplayEventEXT = (PFN_vkRegisterDisplayEventEXT)gdpa(dev, "vkRegisterDisplayEventEXT");
    table->GetSwapchainCounterEXT = (PFN_vkGetSwapchainCounterEXT)gdpa(dev, "vkGetSwapchainCounterEXT");

    // ---- VK_GOOGLE_display_timing extension commands
    table->GetRefreshCycleDurationGOOGLE = (PFN_vkGetRefreshCycleDurationGOOGLE)gdpa(dev, "vkGetRefreshCycleDurationGOOGLE");
    table->GetPastPresentationTimingGOOGLE = (PFN_vkGetPastPresentationTimingGOOGLE)gdpa(dev, "vkGetPastPresentationTimingGOOGLE");

    // ---- VK_EXT_discard_rectangles extension commands
    table->CmdSetDiscardRectangleEXT = (PFN_vkCmdSetDiscardRectangleEXT)gdpa(dev, "vkCmdSetDiscardRectangleEXT");

    // ---- VK_EXT_hdr_metadata extension commands
    table->SetHdrMetadataEXT = (PFN_vkSetHdrMetadataEXT)gdpa(dev, "vkSetHdrMetadataEXT");

    // ---- VK_EXT_debug_utils extension commands
    table->SetDebugUtilsObjectNameEXT = (PFN_vkSetDebugUtilsObjectNameEXT)gipa(inst, "vkSetDebugUtilsObjectNameEXT");
    table->SetDebugUtilsObjectTagEXT = (PFN_vkSetDebugUtilsObjectTagEXT)gipa(inst, "vkSetDebugUtilsObjectTagEXT");
    table->QueueBeginDebugUtilsLabelEXT = (PFN_vkQueueBeginDebugUtilsLabelEXT)gipa(inst, "vkQueueBeginDebugUtilsLabelEXT");
    table->QueueEndDebugUtilsLabelEXT = (PFN_vkQueueEndDebugUtilsLabelEXT)gipa(inst, "vkQueueEndDebugUtilsLabelEXT");
    table->QueueInsertDebugUtilsLabelEXT = (PFN_vkQueueInsertDebugUtilsLabelEXT)gipa(inst, "vkQueueInsertDebugUtilsLabelEXT");
    table->CmdBeginDebugUtilsLabelEXT = (PFN_vkCmdBeginDebugUtilsLabelEXT)gipa(inst, "vkCmdBeginDebugUtilsLabelEXT");
    table->CmdEndDebugUtilsLabelEXT = (PFN_vkCmdEndDebugUtilsLabelEXT)gipa(inst, "vkCmdEndDebugUtilsLabelEXT");
    table->CmdInsertDebugUtilsLabelEXT = (PFN_vkCmdInsertDebugUtilsLabelEXT)gipa(inst, "vkCmdInsertDebugUtilsLabelEXT");

    // ---- VK_ANDROID_external_memory_android_hardware_buffer extension commands
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    table->GetAndroidHardwareBufferPropertiesANDROID = (PFN_vkGetAndroidHardwareBufferPropertiesANDROID)gdpa(dev, "vkGetAndroidHardwareBufferPropertiesANDROID");
#endif // VK_USE_PLATFORM_ANDROID_KHR
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    table->GetMemoryAndroidHardwareBufferANDROID = (PFN_vkGetMemoryAndroidHardwareBufferANDROID)gdpa(dev, "vkGetMemoryAndroidHardwareBufferANDROID");
#endif // VK_USE_PLATFORM_ANDROID_KHR

    // ---- VK_EXT_sample_locations extension commands
    table->CmdSetSampleLocationsEXT = (PFN_vkCmdSetSampleLocationsEXT)gdpa(dev, "vkCmdSetSampleLocationsEXT");

    // ---- VK_EXT_image_drm_format_modifier extension commands
    table->GetImageDrmFormatModifierPropertiesEXT = (PFN_vkGetImageDrmFormatModifierPropertiesEXT)gdpa(dev, "vkGetImageDrmFormatModifierPropertiesEXT");

    // ---- VK_EXT_validation_cache extension commands
    table->CreateValidationCacheEXT = (PFN_vkCreateValidationCacheEXT)gdpa(dev, "vkCreateValidationCacheEXT");
    table->DestroyValidationCacheEXT = (PFN_vkDestroyValidationCacheEXT)gdpa(dev, "vkDestroyValidationCacheEXT");
    table->MergeValidationCachesEXT = (PFN_vkMergeValidationCachesEXT)gdpa(dev, "vkMergeValidationCachesEXT");
    table->GetValidationCacheDataEXT = (PFN_vkGetValidationCacheDataEXT)gdpa(dev, "vkGetValidationCacheDataEXT");

    // ---- VK_NV_shading_rate_image extension commands
    table->CmdBindShadingRateImageNV = (PFN_vkCmdBindShadingRateImageNV)gdpa(dev, "vkCmdBindShadingRateImageNV");
    table->CmdSetViewportShadingRatePaletteNV = (PFN_vkCmdSetViewportShadingRatePaletteNV)gdpa(dev, "vkCmdSetViewportShadingRatePaletteNV");
    table->CmdSetCoarseSampleOrderNV = (PFN_vkCmdSetCoarseSampleOrderNV)gdpa(dev, "vkCmdSetCoarseSampleOrderNV");

    // ---- VK_NV_ray_tracing extension commands
    table->CreateAccelerationStructureNV = (PFN_vkCreateAccelerationStructureNV)gdpa(dev, "vkCreateAccelerationStructureNV");
    table->DestroyAccelerationStructureKHR = (PFN_vkDestroyAccelerationStructureKHR)gdpa(dev, "vkDestroyAccelerationStructureKHR");
    table->DestroyAccelerationStructureNV = (PFN_vkDestroyAccelerationStructureNV)gdpa(dev, "vkDestroyAccelerationStructureNV");
    table->GetAccelerationStructureMemoryRequirementsNV = (PFN_vkGetAccelerationStructureMemoryRequirementsNV)gdpa(dev, "vkGetAccelerationStructureMemoryRequirementsNV");
    table->BindAccelerationStructureMemoryKHR = (PFN_vkBindAccelerationStructureMemoryKHR)gdpa(dev, "vkBindAccelerationStructureMemoryKHR");
    table->BindAccelerationStructureMemoryNV = (PFN_vkBindAccelerationStructureMemoryNV)gdpa(dev, "vkBindAccelerationStructureMemoryNV");
    table->CmdBuildAccelerationStructureNV = (PFN_vkCmdBuildAccelerationStructureNV)gdpa(dev, "vkCmdBuildAccelerationStructureNV");
    table->CmdCopyAccelerationStructureNV = (PFN_vkCmdCopyAccelerationStructureNV)gdpa(dev, "vkCmdCopyAccelerationStructureNV");
    table->CmdTraceRaysNV = (PFN_vkCmdTraceRaysNV)gdpa(dev, "vkCmdTraceRaysNV");
    table->CreateRayTracingPipelinesNV = (PFN_vkCreateRayTracingPipelinesNV)gdpa(dev, "vkCreateRayTracingPipelinesNV");
    table->GetRayTracingShaderGroupHandlesKHR = (PFN_vkGetRayTracingShaderGroupHandlesKHR)gdpa(dev, "vkGetRayTracingShaderGroupHandlesKHR");
    table->GetRayTracingShaderGroupHandlesNV = (PFN_vkGetRayTracingShaderGroupHandlesNV)gdpa(dev, "vkGetRayTracingShaderGroupHandlesNV");
    table->GetAccelerationStructureHandleNV = (PFN_vkGetAccelerationStructureHandleNV)gdpa(dev, "vkGetAccelerationStructureHandleNV");
    table->CmdWriteAccelerationStructuresPropertiesKHR = (PFN_vkCmdWriteAccelerationStructuresPropertiesKHR)gdpa(dev, "vkCmdWriteAccelerationStructuresPropertiesKHR");
    table->CmdWriteAccelerationStructuresPropertiesNV = (PFN_vkCmdWriteAccelerationStructuresPropertiesNV)gdpa(dev, "vkCmdWriteAccelerationStructuresPropertiesNV");
    table->CompileDeferredNV = (PFN_vkCompileDeferredNV)gdpa(dev, "vkCompileDeferredNV");

    // ---- VK_EXT_external_memory_host extension commands
    table->GetMemoryHostPointerPropertiesEXT = (PFN_vkGetMemoryHostPointerPropertiesEXT)gdpa(dev, "vkGetMemoryHostPointerPropertiesEXT");

    // ---- VK_AMD_buffer_marker extension commands
    table->CmdWriteBufferMarkerAMD = (PFN_vkCmdWriteBufferMarkerAMD)gdpa(dev, "vkCmdWriteBufferMarkerAMD");

    // ---- VK_EXT_calibrated_timestamps extension commands
    table->GetCalibratedTimestampsEXT = (PFN_vkGetCalibratedTimestampsEXT)gdpa(dev, "vkGetCalibratedTimestampsEXT");

    // ---- VK_NV_mesh_shader extension commands
    table->CmdDrawMeshTasksNV = (PFN_vkCmdDrawMeshTasksNV)gdpa(dev, "vkCmdDrawMeshTasksNV");
    table->CmdDrawMeshTasksIndirectNV = (PFN_vkCmdDrawMeshTasksIndirectNV)gdpa(dev, "vkCmdDrawMeshTasksIndirectNV");
    table->CmdDrawMeshTasksIndirectCountNV = (PFN_vkCmdDrawMeshTasksIndirectCountNV)gdpa(dev, "vkCmdDrawMeshTasksIndirectCountNV");

    // ---- VK_NV_scissor_exclusive extension commands
    table->CmdSetExclusiveScissorNV = (PFN_vkCmdSetExclusiveScissorNV)gdpa(dev, "vkCmdSetExclusiveScissorNV");

    // ---- VK_NV_device_diagnostic_checkpoints extension commands
    table->CmdSetCheckpointNV = (PFN_vkCmdSetCheckpointNV)gdpa(dev, "vkCmdSetCheckpointNV");
    table->GetQueueCheckpointDataNV = (PFN_vkGetQueueCheckpointDataNV)gdpa(dev, "vkGetQueueCheckpointDataNV");

    // ---- VK_INTEL_performance_query extension commands
    table->InitializePerformanceApiINTEL = (PFN_vkInitializePerformanceApiINTEL)gdpa(dev, "vkInitializePerformanceApiINTEL");
    table->UninitializePerformanceApiINTEL = (PFN_vkUninitializePerformanceApiINTEL)gdpa(dev, "vkUninitializePerformanceApiINTEL");
    table->CmdSetPerformanceMarkerINTEL = (PFN_vkCmdSetPerformanceMarkerINTEL)gdpa(dev, "vkCmdSetPerformanceMarkerINTEL");
    table->CmdSetPerformanceStreamMarkerINTEL = (PFN_vkCmdSetPerformanceStreamMarkerINTEL)gdpa(dev, "vkCmdSetPerformanceStreamMarkerINTEL");
    table->CmdSetPerformanceOverrideINTEL = (PFN_vkCmdSetPerformanceOverrideINTEL)gdpa(dev, "vkCmdSetPerformanceOverrideINTEL");
    table->AcquirePerformanceConfigurationINTEL = (PFN_vkAcquirePerformanceConfigurationINTEL)gdpa(dev, "vkAcquirePerformanceConfigurationINTEL");
    table->ReleasePerformanceConfigurationINTEL = (PFN_vkReleasePerformanceConfigurationINTEL)gdpa(dev, "vkReleasePerformanceConfigurationINTEL");
    table->QueueSetPerformanceConfigurationINTEL = (PFN_vkQueueSetPerformanceConfigurationINTEL)gdpa(dev, "vkQueueSetPerformanceConfigurationINTEL");
    table->GetPerformanceParameterINTEL = (PFN_vkGetPerformanceParameterINTEL)gdpa(dev, "vkGetPerformanceParameterINTEL");

    // ---- VK_AMD_display_native_hdr extension commands
    table->SetLocalDimmingAMD = (PFN_vkSetLocalDimmingAMD)gdpa(dev, "vkSetLocalDimmingAMD");

    // ---- VK_EXT_buffer_device_address extension commands
    table->GetBufferDeviceAddressEXT = (PFN_vkGetBufferDeviceAddressEXT)gdpa(dev, "vkGetBufferDeviceAddressEXT");

    // ---- VK_EXT_full_screen_exclusive extension commands
#ifdef VK_USE_PLATFORM_WIN32_KHR
    table->AcquireFullScreenExclusiveModeEXT = (PFN_vkAcquireFullScreenExclusiveModeEXT)gdpa(dev, "vkAcquireFullScreenExclusiveModeEXT");
#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
    table->ReleaseFullScreenExclusiveModeEXT = (PFN_vkReleaseFullScreenExclusiveModeEXT)gdpa(dev, "vkReleaseFullScreenExclusiveModeEXT");
#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
    table->GetDeviceGroupSurfacePresentModes2EXT = (PFN_vkGetDeviceGroupSurfacePresentModes2EXT)gdpa(dev, "vkGetDeviceGroupSurfacePresentModes2EXT");
#endif // VK_USE_PLATFORM_WIN32_KHR

    // ---- VK_EXT_line_rasterization extension commands
    table->CmdSetLineStippleEXT = (PFN_vkCmdSetLineStippleEXT)gdpa(dev, "vkCmdSetLineStippleEXT");

    // ---- VK_EXT_host_query_reset extension commands
    table->ResetQueryPoolEXT = (PFN_vkResetQueryPoolEXT)gdpa(dev, "vkResetQueryPoolEXT");

    // ---- VK_EXT_extended_dynamic_state extension commands
    table->CmdSetCullModeEXT = (PFN_vkCmdSetCullModeEXT)gdpa(dev, "vkCmdSetCullModeEXT");
    table->CmdSetFrontFaceEXT = (PFN_vkCmdSetFrontFaceEXT)gdpa(dev, "vkCmdSetFrontFaceEXT");
    table->CmdSetPrimitiveTopologyEXT = (PFN_vkCmdSetPrimitiveTopologyEXT)gdpa(dev, "vkCmdSetPrimitiveTopologyEXT");
    table->CmdSetViewportWithCountEXT = (PFN_vkCmdSetViewportWithCountEXT)gdpa(dev, "vkCmdSetViewportWithCountEXT");
    table->CmdSetScissorWithCountEXT = (PFN_vkCmdSetScissorWithCountEXT)gdpa(dev, "vkCmdSetScissorWithCountEXT");
    table->CmdBindVertexBuffers2EXT = (PFN_vkCmdBindVertexBuffers2EXT)gdpa(dev, "vkCmdBindVertexBuffers2EXT");
    table->CmdSetDepthTestEnableEXT = (PFN_vkCmdSetDepthTestEnableEXT)gdpa(dev, "vkCmdSetDepthTestEnableEXT");
    table->CmdSetDepthWriteEnableEXT = (PFN_vkCmdSetDepthWriteEnableEXT)gdpa(dev, "vkCmdSetDepthWriteEnableEXT");
    table->CmdSetDepthCompareOpEXT = (PFN_vkCmdSetDepthCompareOpEXT)gdpa(dev, "vkCmdSetDepthCompareOpEXT");
    table->CmdSetDepthBoundsTestEnableEXT = (PFN_vkCmdSetDepthBoundsTestEnableEXT)gdpa(dev, "vkCmdSetDepthBoundsTestEnableEXT");
    table->CmdSetStencilTestEnableEXT = (PFN_vkCmdSetStencilTestEnableEXT)gdpa(dev, "vkCmdSetStencilTestEnableEXT");
    table->CmdSetStencilOpEXT = (PFN_vkCmdSetStencilOpEXT)gdpa(dev, "vkCmdSetStencilOpEXT");

    // ---- VK_NV_device_generated_commands extension commands
    table->GetGeneratedCommandsMemoryRequirementsNV = (PFN_vkGetGeneratedCommandsMemoryRequirementsNV)gdpa(dev, "vkGetGeneratedCommandsMemoryRequirementsNV");
    table->CmdPreprocessGeneratedCommandsNV = (PFN_vkCmdPreprocessGeneratedCommandsNV)gdpa(dev, "vkCmdPreprocessGeneratedCommandsNV");
    table->CmdExecuteGeneratedCommandsNV = (PFN_vkCmdExecuteGeneratedCommandsNV)gdpa(dev, "vkCmdExecuteGeneratedCommandsNV");
    table->CmdBindPipelineShaderGroupNV = (PFN_vkCmdBindPipelineShaderGroupNV)gdpa(dev, "vkCmdBindPipelineShaderGroupNV");
    table->CreateIndirectCommandsLayoutNV = (PFN_vkCreateIndirectCommandsLayoutNV)gdpa(dev, "vkCreateIndirectCommandsLayoutNV");
    table->DestroyIndirectCommandsLayoutNV = (PFN_vkDestroyIndirectCommandsLayoutNV)gdpa(dev, "vkDestroyIndirectCommandsLayoutNV");

    // ---- VK_EXT_private_data extension commands
    table->CreatePrivateDataSlotEXT = (PFN_vkCreatePrivateDataSlotEXT)gdpa(dev, "vkCreatePrivateDataSlotEXT");
    table->DestroyPrivateDataSlotEXT = (PFN_vkDestroyPrivateDataSlotEXT)gdpa(dev, "vkDestroyPrivateDataSlotEXT");
    table->SetPrivateDataEXT = (PFN_vkSetPrivateDataEXT)gdpa(dev, "vkSetPrivateDataEXT");
    table->GetPrivateDataEXT = (PFN_vkGetPrivateDataEXT)gdpa(dev, "vkGetPrivateDataEXT");

    // ---- VK_KHR_ray_tracing extension commands
#ifdef VK_ENABLE_BETA_EXTENSIONS
    table->CreateAccelerationStructureKHR = (PFN_vkCreateAccelerationStructureKHR)gdpa(dev, "vkCreateAccelerationStructureKHR");
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    table->GetAccelerationStructureMemoryRequirementsKHR = (PFN_vkGetAccelerationStructureMemoryRequirementsKHR)gdpa(dev, "vkGetAccelerationStructureMemoryRequirementsKHR");
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    table->CmdBuildAccelerationStructureKHR = (PFN_vkCmdBuildAccelerationStructureKHR)gdpa(dev, "vkCmdBuildAccelerationStructureKHR");
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    table->CmdBuildAccelerationStructureIndirectKHR = (PFN_vkCmdBuildAccelerationStructureIndirectKHR)gdpa(dev, "vkCmdBuildAccelerationStructureIndirectKHR");
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    table->BuildAccelerationStructureKHR = (PFN_vkBuildAccelerationStructureKHR)gdpa(dev, "vkBuildAccelerationStructureKHR");
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    table->CopyAccelerationStructureKHR = (PFN_vkCopyAccelerationStructureKHR)gdpa(dev, "vkCopyAccelerationStructureKHR");
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    table->CopyAccelerationStructureToMemoryKHR = (PFN_vkCopyAccelerationStructureToMemoryKHR)gdpa(dev, "vkCopyAccelerationStructureToMemoryKHR");
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    table->CopyMemoryToAccelerationStructureKHR = (PFN_vkCopyMemoryToAccelerationStructureKHR)gdpa(dev, "vkCopyMemoryToAccelerationStructureKHR");
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    table->WriteAccelerationStructuresPropertiesKHR = (PFN_vkWriteAccelerationStructuresPropertiesKHR)gdpa(dev, "vkWriteAccelerationStructuresPropertiesKHR");
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    table->CmdCopyAccelerationStructureKHR = (PFN_vkCmdCopyAccelerationStructureKHR)gdpa(dev, "vkCmdCopyAccelerationStructureKHR");
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    table->CmdCopyAccelerationStructureToMemoryKHR = (PFN_vkCmdCopyAccelerationStructureToMemoryKHR)gdpa(dev, "vkCmdCopyAccelerationStructureToMemoryKHR");
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    table->CmdCopyMemoryToAccelerationStructureKHR = (PFN_vkCmdCopyMemoryToAccelerationStructureKHR)gdpa(dev, "vkCmdCopyMemoryToAccelerationStructureKHR");
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    table->CmdTraceRaysKHR = (PFN_vkCmdTraceRaysKHR)gdpa(dev, "vkCmdTraceRaysKHR");
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    table->CreateRayTracingPipelinesKHR = (PFN_vkCreateRayTracingPipelinesKHR)gdpa(dev, "vkCreateRayTracingPipelinesKHR");
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    table->GetAccelerationStructureDeviceAddressKHR = (PFN_vkGetAccelerationStructureDeviceAddressKHR)gdpa(dev, "vkGetAccelerationStructureDeviceAddressKHR");
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    table->GetRayTracingCaptureReplayShaderGroupHandlesKHR = (PFN_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR)gdpa(dev, "vkGetRayTracingCaptureReplayShaderGroupHandlesKHR");
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    table->CmdTraceRaysIndirectKHR = (PFN_vkCmdTraceRaysIndirectKHR)gdpa(dev, "vkCmdTraceRaysIndirectKHR");
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    table->GetDeviceAccelerationStructureCompatibilityKHR = (PFN_vkGetDeviceAccelerationStructureCompatibilityKHR)gdpa(dev, "vkGetDeviceAccelerationStructureCompatibilityKHR");
#endif // VK_ENABLE_BETA_EXTENSIONS
}

// Init Instance function pointer dispatch table with core commands
VKAPI_ATTR void VKAPI_CALL loader_init_instance_core_dispatch_table(VkLayerInstanceDispatchTable *table, PFN_vkGetInstanceProcAddr gpa,
                                                                    VkInstance inst) {

    // ---- Core 1_0 commands
    table->DestroyInstance = (PFN_vkDestroyInstance)gpa(inst, "vkDestroyInstance");
    table->EnumeratePhysicalDevices = (PFN_vkEnumeratePhysicalDevices)gpa(inst, "vkEnumeratePhysicalDevices");
    table->GetPhysicalDeviceFeatures = (PFN_vkGetPhysicalDeviceFeatures)gpa(inst, "vkGetPhysicalDeviceFeatures");
    table->GetPhysicalDeviceFormatProperties = (PFN_vkGetPhysicalDeviceFormatProperties)gpa(inst, "vkGetPhysicalDeviceFormatProperties");
    table->GetPhysicalDeviceImageFormatProperties = (PFN_vkGetPhysicalDeviceImageFormatProperties)gpa(inst, "vkGetPhysicalDeviceImageFormatProperties");
    table->GetPhysicalDeviceProperties = (PFN_vkGetPhysicalDeviceProperties)gpa(inst, "vkGetPhysicalDeviceProperties");
    table->GetPhysicalDeviceQueueFamilyProperties = (PFN_vkGetPhysicalDeviceQueueFamilyProperties)gpa(inst, "vkGetPhysicalDeviceQueueFamilyProperties");
    table->GetPhysicalDeviceMemoryProperties = (PFN_vkGetPhysicalDeviceMemoryProperties)gpa(inst, "vkGetPhysicalDeviceMemoryProperties");
    table->GetInstanceProcAddr = gpa;
    table->EnumerateDeviceExtensionProperties = (PFN_vkEnumerateDeviceExtensionProperties)gpa(inst, "vkEnumerateDeviceExtensionProperties");
    table->EnumerateDeviceLayerProperties = (PFN_vkEnumerateDeviceLayerProperties)gpa(inst, "vkEnumerateDeviceLayerProperties");
    table->GetPhysicalDeviceSparseImageFormatProperties = (PFN_vkGetPhysicalDeviceSparseImageFormatProperties)gpa(inst, "vkGetPhysicalDeviceSparseImageFormatProperties");

    // ---- Core 1_1 commands
    table->EnumeratePhysicalDeviceGroups = (PFN_vkEnumeratePhysicalDeviceGroups)gpa(inst, "vkEnumeratePhysicalDeviceGroups");
    table->GetPhysicalDeviceFeatures2 = (PFN_vkGetPhysicalDeviceFeatures2)gpa(inst, "vkGetPhysicalDeviceFeatures2");
    table->GetPhysicalDeviceProperties2 = (PFN_vkGetPhysicalDeviceProperties2)gpa(inst, "vkGetPhysicalDeviceProperties2");
    table->GetPhysicalDeviceFormatProperties2 = (PFN_vkGetPhysicalDeviceFormatProperties2)gpa(inst, "vkGetPhysicalDeviceFormatProperties2");
    table->GetPhysicalDeviceImageFormatProperties2 = (PFN_vkGetPhysicalDeviceImageFormatProperties2)gpa(inst, "vkGetPhysicalDeviceImageFormatProperties2");
    table->GetPhysicalDeviceQueueFamilyProperties2 = (PFN_vkGetPhysicalDeviceQueueFamilyProperties2)gpa(inst, "vkGetPhysicalDeviceQueueFamilyProperties2");
    table->GetPhysicalDeviceMemoryProperties2 = (PFN_vkGetPhysicalDeviceMemoryProperties2)gpa(inst, "vkGetPhysicalDeviceMemoryProperties2");
    table->GetPhysicalDeviceSparseImageFormatProperties2 = (PFN_vkGetPhysicalDeviceSparseImageFormatProperties2)gpa(inst, "vkGetPhysicalDeviceSparseImageFormatProperties2");
    table->GetPhysicalDeviceExternalBufferProperties = (PFN_vkGetPhysicalDeviceExternalBufferProperties)gpa(inst, "vkGetPhysicalDeviceExternalBufferProperties");
    table->GetPhysicalDeviceExternalFenceProperties = (PFN_vkGetPhysicalDeviceExternalFenceProperties)gpa(inst, "vkGetPhysicalDeviceExternalFenceProperties");
    table->GetPhysicalDeviceExternalSemaphoreProperties = (PFN_vkGetPhysicalDeviceExternalSemaphoreProperties)gpa(inst, "vkGetPhysicalDeviceExternalSemaphoreProperties");
}

// Init Instance function pointer dispatch table with core commands
VKAPI_ATTR void VKAPI_CALL loader_init_instance_extension_dispatch_table(VkLayerInstanceDispatchTable *table, PFN_vkGetInstanceProcAddr gpa,
                                                                        VkInstance inst) {

    // ---- VK_KHR_surface extension commands
    table->DestroySurfaceKHR = (PFN_vkDestroySurfaceKHR)gpa(inst, "vkDestroySurfaceKHR");
    table->GetPhysicalDeviceSurfaceSupportKHR = (PFN_vkGetPhysicalDeviceSurfaceSupportKHR)gpa(inst, "vkGetPhysicalDeviceSurfaceSupportKHR");
    table->GetPhysicalDeviceSurfaceCapabilitiesKHR = (PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR)gpa(inst, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    table->GetPhysicalDeviceSurfaceFormatsKHR = (PFN_vkGetPhysicalDeviceSurfaceFormatsKHR)gpa(inst, "vkGetPhysicalDeviceSurfaceFormatsKHR");
    table->GetPhysicalDeviceSurfacePresentModesKHR = (PFN_vkGetPhysicalDeviceSurfacePresentModesKHR)gpa(inst, "vkGetPhysicalDeviceSurfacePresentModesKHR");

    // ---- VK_KHR_swapchain extension commands
    table->GetPhysicalDevicePresentRectanglesKHR = (PFN_vkGetPhysicalDevicePresentRectanglesKHR)gpa(inst, "vkGetPhysicalDevicePresentRectanglesKHR");

    // ---- VK_KHR_display extension commands
    table->GetPhysicalDeviceDisplayPropertiesKHR = (PFN_vkGetPhysicalDeviceDisplayPropertiesKHR)gpa(inst, "vkGetPhysicalDeviceDisplayPropertiesKHR");
    table->GetPhysicalDeviceDisplayPlanePropertiesKHR = (PFN_vkGetPhysicalDeviceDisplayPlanePropertiesKHR)gpa(inst, "vkGetPhysicalDeviceDisplayPlanePropertiesKHR");
    table->GetDisplayPlaneSupportedDisplaysKHR = (PFN_vkGetDisplayPlaneSupportedDisplaysKHR)gpa(inst, "vkGetDisplayPlaneSupportedDisplaysKHR");
    table->GetDisplayModePropertiesKHR = (PFN_vkGetDisplayModePropertiesKHR)gpa(inst, "vkGetDisplayModePropertiesKHR");
    table->CreateDisplayModeKHR = (PFN_vkCreateDisplayModeKHR)gpa(inst, "vkCreateDisplayModeKHR");
    table->GetDisplayPlaneCapabilitiesKHR = (PFN_vkGetDisplayPlaneCapabilitiesKHR)gpa(inst, "vkGetDisplayPlaneCapabilitiesKHR");
    table->CreateDisplayPlaneSurfaceKHR = (PFN_vkCreateDisplayPlaneSurfaceKHR)gpa(inst, "vkCreateDisplayPlaneSurfaceKHR");

    // ---- VK_KHR_xlib_surface extension commands
#ifdef VK_USE_PLATFORM_XLIB_KHR
    table->CreateXlibSurfaceKHR = (PFN_vkCreateXlibSurfaceKHR)gpa(inst, "vkCreateXlibSurfaceKHR");
#endif // VK_USE_PLATFORM_XLIB_KHR
#ifdef VK_USE_PLATFORM_XLIB_KHR
    table->GetPhysicalDeviceXlibPresentationSupportKHR = (PFN_vkGetPhysicalDeviceXlibPresentationSupportKHR)gpa(inst, "vkGetPhysicalDeviceXlibPresentationSupportKHR");
#endif // VK_USE_PLATFORM_XLIB_KHR

    // ---- VK_KHR_xcb_surface extension commands
#ifdef VK_USE_PLATFORM_XCB_KHR
    table->CreateXcbSurfaceKHR = (PFN_vkCreateXcbSurfaceKHR)gpa(inst, "vkCreateXcbSurfaceKHR");
#endif // VK_USE_PLATFORM_XCB_KHR
#ifdef VK_USE_PLATFORM_XCB_KHR
    table->GetPhysicalDeviceXcbPresentationSupportKHR = (PFN_vkGetPhysicalDeviceXcbPresentationSupportKHR)gpa(inst, "vkGetPhysicalDeviceXcbPresentationSupportKHR");
#endif // VK_USE_PLATFORM_XCB_KHR

    // ---- VK_KHR_wayland_surface extension commands
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    table->CreateWaylandSurfaceKHR = (PFN_vkCreateWaylandSurfaceKHR)gpa(inst, "vkCreateWaylandSurfaceKHR");
#endif // VK_USE_PLATFORM_WAYLAND_KHR
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    table->GetPhysicalDeviceWaylandPresentationSupportKHR = (PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR)gpa(inst, "vkGetPhysicalDeviceWaylandPresentationSupportKHR");
#endif // VK_USE_PLATFORM_WAYLAND_KHR

    // ---- VK_KHR_android_surface extension commands
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    table->CreateAndroidSurfaceKHR = (PFN_vkCreateAndroidSurfaceKHR)gpa(inst, "vkCreateAndroidSurfaceKHR");
#endif // VK_USE_PLATFORM_ANDROID_KHR

    // ---- VK_KHR_win32_surface extension commands
#ifdef VK_USE_PLATFORM_WIN32_KHR
    table->CreateWin32SurfaceKHR = (PFN_vkCreateWin32SurfaceKHR)gpa(inst, "vkCreateWin32SurfaceKHR");
#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
    table->GetPhysicalDeviceWin32PresentationSupportKHR = (PFN_vkGetPhysicalDeviceWin32PresentationSupportKHR)gpa(inst, "vkGetPhysicalDeviceWin32PresentationSupportKHR");
#endif // VK_USE_PLATFORM_WIN32_KHR

    // ---- VK_KHR_get_physical_device_properties2 extension commands
    table->GetPhysicalDeviceFeatures2KHR = (PFN_vkGetPhysicalDeviceFeatures2KHR)gpa(inst, "vkGetPhysicalDeviceFeatures2KHR");
    table->GetPhysicalDeviceProperties2KHR = (PFN_vkGetPhysicalDeviceProperties2KHR)gpa(inst, "vkGetPhysicalDeviceProperties2KHR");
    table->GetPhysicalDeviceFormatProperties2KHR = (PFN_vkGetPhysicalDeviceFormatProperties2KHR)gpa(inst, "vkGetPhysicalDeviceFormatProperties2KHR");
    table->GetPhysicalDeviceImageFormatProperties2KHR = (PFN_vkGetPhysicalDeviceImageFormatProperties2KHR)gpa(inst, "vkGetPhysicalDeviceImageFormatProperties2KHR");
    table->GetPhysicalDeviceQueueFamilyProperties2KHR = (PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR)gpa(inst, "vkGetPhysicalDeviceQueueFamilyProperties2KHR");
    table->GetPhysicalDeviceMemoryProperties2KHR = (PFN_vkGetPhysicalDeviceMemoryProperties2KHR)gpa(inst, "vkGetPhysicalDeviceMemoryProperties2KHR");
    table->GetPhysicalDeviceSparseImageFormatProperties2KHR = (PFN_vkGetPhysicalDeviceSparseImageFormatProperties2KHR)gpa(inst, "vkGetPhysicalDeviceSparseImageFormatProperties2KHR");

    // ---- VK_KHR_device_group_creation extension commands
    table->EnumeratePhysicalDeviceGroupsKHR = (PFN_vkEnumeratePhysicalDeviceGroupsKHR)gpa(inst, "vkEnumeratePhysicalDeviceGroupsKHR");

    // ---- VK_KHR_external_memory_capabilities extension commands
    table->GetPhysicalDeviceExternalBufferPropertiesKHR = (PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR)gpa(inst, "vkGetPhysicalDeviceExternalBufferPropertiesKHR");

    // ---- VK_KHR_external_semaphore_capabilities extension commands
    table->GetPhysicalDeviceExternalSemaphorePropertiesKHR = (PFN_vkGetPhysicalDeviceExternalSemaphorePropertiesKHR)gpa(inst, "vkGetPhysicalDeviceExternalSemaphorePropertiesKHR");

    // ---- VK_KHR_external_fence_capabilities extension commands
    table->GetPhysicalDeviceExternalFencePropertiesKHR = (PFN_vkGetPhysicalDeviceExternalFencePropertiesKHR)gpa(inst, "vkGetPhysicalDeviceExternalFencePropertiesKHR");

    // ---- VK_KHR_performance_query extension commands
    table->EnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR = (PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR)gpa(inst, "vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR");
    table->GetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR = (PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR)gpa(inst, "vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR");

    // ---- VK_KHR_get_surface_capabilities2 extension commands
    table->GetPhysicalDeviceSurfaceCapabilities2KHR = (PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR)gpa(inst, "vkGetPhysicalDeviceSurfaceCapabilities2KHR");
    table->GetPhysicalDeviceSurfaceFormats2KHR = (PFN_vkGetPhysicalDeviceSurfaceFormats2KHR)gpa(inst, "vkGetPhysicalDeviceSurfaceFormats2KHR");

    // ---- VK_KHR_get_display_properties2 extension commands
    table->GetPhysicalDeviceDisplayProperties2KHR = (PFN_vkGetPhysicalDeviceDisplayProperties2KHR)gpa(inst, "vkGetPhysicalDeviceDisplayProperties2KHR");
    table->GetPhysicalDeviceDisplayPlaneProperties2KHR = (PFN_vkGetPhysicalDeviceDisplayPlaneProperties2KHR)gpa(inst, "vkGetPhysicalDeviceDisplayPlaneProperties2KHR");
    table->GetDisplayModeProperties2KHR = (PFN_vkGetDisplayModeProperties2KHR)gpa(inst, "vkGetDisplayModeProperties2KHR");
    table->GetDisplayPlaneCapabilities2KHR = (PFN_vkGetDisplayPlaneCapabilities2KHR)gpa(inst, "vkGetDisplayPlaneCapabilities2KHR");

    // ---- VK_EXT_debug_report extension commands
    table->CreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)gpa(inst, "vkCreateDebugReportCallbackEXT");
    table->DestroyDebugReportCallbackEXT = (PFN_vkDestroyDebugReportCallbackEXT)gpa(inst, "vkDestroyDebugReportCallbackEXT");
    table->DebugReportMessageEXT = (PFN_vkDebugReportMessageEXT)gpa(inst, "vkDebugReportMessageEXT");

    // ---- VK_GGP_stream_descriptor_surface extension commands
#ifdef VK_USE_PLATFORM_GGP
    table->CreateStreamDescriptorSurfaceGGP = (PFN_vkCreateStreamDescriptorSurfaceGGP)gpa(inst, "vkCreateStreamDescriptorSurfaceGGP");
#endif // VK_USE_PLATFORM_GGP

    // ---- VK_NV_external_memory_capabilities extension commands
    table->GetPhysicalDeviceExternalImageFormatPropertiesNV = (PFN_vkGetPhysicalDeviceExternalImageFormatPropertiesNV)gpa(inst, "vkGetPhysicalDeviceExternalImageFormatPropertiesNV");

    // ---- VK_NN_vi_surface extension commands
#ifdef VK_USE_PLATFORM_VI_NN
    table->CreateViSurfaceNN = (PFN_vkCreateViSurfaceNN)gpa(inst, "vkCreateViSurfaceNN");
#endif // VK_USE_PLATFORM_VI_NN

    // ---- VK_EXT_direct_mode_display extension commands
    table->ReleaseDisplayEXT = (PFN_vkReleaseDisplayEXT)gpa(inst, "vkReleaseDisplayEXT");

    // ---- VK_EXT_acquire_xlib_display extension commands
#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT
    table->AcquireXlibDisplayEXT = (PFN_vkAcquireXlibDisplayEXT)gpa(inst, "vkAcquireXlibDisplayEXT");
#endif // VK_USE_PLATFORM_XLIB_XRANDR_EXT
#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT
    table->GetRandROutputDisplayEXT = (PFN_vkGetRandROutputDisplayEXT)gpa(inst, "vkGetRandROutputDisplayEXT");
#endif // VK_USE_PLATFORM_XLIB_XRANDR_EXT

    // ---- VK_EXT_display_surface_counter extension commands
    table->GetPhysicalDeviceSurfaceCapabilities2EXT = (PFN_vkGetPhysicalDeviceSurfaceCapabilities2EXT)gpa(inst, "vkGetPhysicalDeviceSurfaceCapabilities2EXT");

    // ---- VK_MVK_ios_surface extension commands
#ifdef VK_USE_PLATFORM_IOS_MVK
    table->CreateIOSSurfaceMVK = (PFN_vkCreateIOSSurfaceMVK)gpa(inst, "vkCreateIOSSurfaceMVK");
#endif // VK_USE_PLATFORM_IOS_MVK

    // ---- VK_MVK_macos_surface extension commands
#ifdef VK_USE_PLATFORM_MACOS_MVK
    table->CreateMacOSSurfaceMVK = (PFN_vkCreateMacOSSurfaceMVK)gpa(inst, "vkCreateMacOSSurfaceMVK");
#endif // VK_USE_PLATFORM_MACOS_MVK

    // ---- VK_EXT_debug_utils extension commands
    table->CreateDebugUtilsMessengerEXT = (PFN_vkCreateDebugUtilsMessengerEXT)gpa(inst, "vkCreateDebugUtilsMessengerEXT");
    table->DestroyDebugUtilsMessengerEXT = (PFN_vkDestroyDebugUtilsMessengerEXT)gpa(inst, "vkDestroyDebugUtilsMessengerEXT");
    table->SubmitDebugUtilsMessageEXT = (PFN_vkSubmitDebugUtilsMessageEXT)gpa(inst, "vkSubmitDebugUtilsMessageEXT");

    // ---- VK_EXT_sample_locations extension commands
    table->GetPhysicalDeviceMultisamplePropertiesEXT = (PFN_vkGetPhysicalDeviceMultisamplePropertiesEXT)gpa(inst, "vkGetPhysicalDeviceMultisamplePropertiesEXT");

    // ---- VK_EXT_calibrated_timestamps extension commands
    table->GetPhysicalDeviceCalibrateableTimeDomainsEXT = (PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT)gpa(inst, "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT");

    // ---- VK_FUCHSIA_imagepipe_surface extension commands
#ifdef VK_USE_PLATFORM_FUCHSIA
    table->CreateImagePipeSurfaceFUCHSIA = (PFN_vkCreateImagePipeSurfaceFUCHSIA)gpa(inst, "vkCreateImagePipeSurfaceFUCHSIA");
#endif // VK_USE_PLATFORM_FUCHSIA

    // ---- VK_EXT_metal_surface extension commands
#ifdef VK_USE_PLATFORM_METAL_EXT
    table->CreateMetalSurfaceEXT = (PFN_vkCreateMetalSurfaceEXT)gpa(inst, "vkCreateMetalSurfaceEXT");
#endif // VK_USE_PLATFORM_METAL_EXT

    // ---- VK_EXT_tooling_info extension commands
    table->GetPhysicalDeviceToolPropertiesEXT = (PFN_vkGetPhysicalDeviceToolPropertiesEXT)gpa(inst, "vkGetPhysicalDeviceToolPropertiesEXT");

    // ---- VK_NV_cooperative_matrix extension commands
    table->GetPhysicalDeviceCooperativeMatrixPropertiesNV = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV)gpa(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesNV");

    // ---- VK_NV_coverage_reduction_mode extension commands
    table->GetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV = (PFN_vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV)gpa(inst, "vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV");

    // ---- VK_EXT_full_screen_exclusive extension commands
#ifdef VK_USE_PLATFORM_WIN32_KHR
    table->GetPhysicalDeviceSurfacePresentModes2EXT = (PFN_vkGetPhysicalDeviceSurfacePresentModes2EXT)gpa(inst, "vkGetPhysicalDeviceSurfacePresentModes2EXT");
#endif // VK_USE_PLATFORM_WIN32_KHR

    // ---- VK_EXT_headless_surface extension commands
    table->CreateHeadlessSurfaceEXT = (PFN_vkCreateHeadlessSurfaceEXT)gpa(inst, "vkCreateHeadlessSurfaceEXT");

    // ---- VK_EXT_directfb_surface extension commands
#ifdef VK_USE_PLATFORM_DIRECTFB_EXT
    table->CreateDirectFBSurfaceEXT = (PFN_vkCreateDirectFBSurfaceEXT)gpa(inst, "vkCreateDirectFBSurfaceEXT");
#endif // VK_USE_PLATFORM_DIRECTFB_EXT
#ifdef VK_USE_PLATFORM_DIRECTFB_EXT
    table->GetPhysicalDeviceDirectFBPresentationSupportEXT = (PFN_vkGetPhysicalDeviceDirectFBPresentationSupportEXT)gpa(inst, "vkGetPhysicalDeviceDirectFBPresentationSupportEXT");
#endif // VK_USE_PLATFORM_DIRECTFB_EXT
}

// Device command lookup function
VKAPI_ATTR void* VKAPI_CALL loader_lookup_device_dispatch_table(const VkLayerDispatchTable *table, const char *name) {
    if (!name || name[0] != 'v' || name[1] != 'k') return NULL;

    name += 2;

    // ---- Core 1_0 commands
    if (!strcmp(name, "GetDeviceProcAddr")) return (void *)table->GetDeviceProcAddr;
    if (!strcmp(name, "DestroyDevice")) return (void *)table->DestroyDevice;
    if (!strcmp(name, "GetDeviceQueue")) return (void *)table->GetDeviceQueue;
    if (!strcmp(name, "QueueSubmit")) return (void *)table->QueueSubmit;
    if (!strcmp(name, "QueueWaitIdle")) return (void *)table->QueueWaitIdle;
    if (!strcmp(name, "DeviceWaitIdle")) return (void *)table->DeviceWaitIdle;
    if (!strcmp(name, "AllocateMemory")) return (void *)table->AllocateMemory;
    if (!strcmp(name, "FreeMemory")) return (void *)table->FreeMemory;
    if (!strcmp(name, "MapMemory")) return (void *)table->MapMemory;
    if (!strcmp(name, "UnmapMemory")) return (void *)table->UnmapMemory;
    if (!strcmp(name, "FlushMappedMemoryRanges")) return (void *)table->FlushMappedMemoryRanges;
    if (!strcmp(name, "InvalidateMappedMemoryRanges")) return (void *)table->InvalidateMappedMemoryRanges;
    if (!strcmp(name, "GetDeviceMemoryCommitment")) return (void *)table->GetDeviceMemoryCommitment;
    if (!strcmp(name, "BindBufferMemory")) return (void *)table->BindBufferMemory;
    if (!strcmp(name, "BindImageMemory")) return (void *)table->BindImageMemory;
    if (!strcmp(name, "GetBufferMemoryRequirements")) return (void *)table->GetBufferMemoryRequirements;
    if (!strcmp(name, "GetImageMemoryRequirements")) return (void *)table->GetImageMemoryRequirements;
    if (!strcmp(name, "GetImageSparseMemoryRequirements")) return (void *)table->GetImageSparseMemoryRequirements;
    if (!strcmp(name, "QueueBindSparse")) return (void *)table->QueueBindSparse;
    if (!strcmp(name, "CreateFence")) return (void *)table->CreateFence;
    if (!strcmp(name, "DestroyFence")) return (void *)table->DestroyFence;
    if (!strcmp(name, "ResetFences")) return (void *)table->ResetFences;
    if (!strcmp(name, "GetFenceStatus")) return (void *)table->GetFenceStatus;
    if (!strcmp(name, "WaitForFences")) return (void *)table->WaitForFences;
    if (!strcmp(name, "CreateSemaphore")) return (void *)table->CreateSemaphore;
    if (!strcmp(name, "DestroySemaphore")) return (void *)table->DestroySemaphore;
    if (!strcmp(name, "CreateEvent")) return (void *)table->CreateEvent;
    if (!strcmp(name, "DestroyEvent")) return (void *)table->DestroyEvent;
    if (!strcmp(name, "GetEventStatus")) return (void *)table->GetEventStatus;
    if (!strcmp(name, "SetEvent")) return (void *)table->SetEvent;
    if (!strcmp(name, "ResetEvent")) return (void *)table->ResetEvent;
    if (!strcmp(name, "CreateQueryPool")) return (void *)table->CreateQueryPool;
    if (!strcmp(name, "DestroyQueryPool")) return (void *)table->DestroyQueryPool;
    if (!strcmp(name, "GetQueryPoolResults")) return (void *)table->GetQueryPoolResults;
    if (!strcmp(name, "CreateBuffer")) return (void *)table->CreateBuffer;
    if (!strcmp(name, "DestroyBuffer")) return (void *)table->DestroyBuffer;
    if (!strcmp(name, "CreateBufferView")) return (void *)table->CreateBufferView;
    if (!strcmp(name, "DestroyBufferView")) return (void *)table->DestroyBufferView;
    if (!strcmp(name, "CreateImage")) return (void *)table->CreateImage;
    if (!strcmp(name, "DestroyImage")) return (void *)table->DestroyImage;
    if (!strcmp(name, "GetImageSubresourceLayout")) return (void *)table->GetImageSubresourceLayout;
    if (!strcmp(name, "CreateImageView")) return (void *)table->CreateImageView;
    if (!strcmp(name, "DestroyImageView")) return (void *)table->DestroyImageView;
    if (!strcmp(name, "CreateShaderModule")) return (void *)table->CreateShaderModule;
    if (!strcmp(name, "DestroyShaderModule")) return (void *)table->DestroyShaderModule;
    if (!strcmp(name, "CreatePipelineCache")) return (void *)table->CreatePipelineCache;
    if (!strcmp(name, "DestroyPipelineCache")) return (void *)table->DestroyPipelineCache;
    if (!strcmp(name, "GetPipelineCacheData")) return (void *)table->GetPipelineCacheData;
    if (!strcmp(name, "MergePipelineCaches")) return (void *)table->MergePipelineCaches;
    if (!strcmp(name, "CreateGraphicsPipelines")) return (void *)table->CreateGraphicsPipelines;
    if (!strcmp(name, "CreateComputePipelines")) return (void *)table->CreateComputePipelines;
    if (!strcmp(name, "DestroyPipeline")) return (void *)table->DestroyPipeline;
    if (!strcmp(name, "CreatePipelineLayout")) return (void *)table->CreatePipelineLayout;
    if (!strcmp(name, "DestroyPipelineLayout")) return (void *)table->DestroyPipelineLayout;
    if (!strcmp(name, "CreateSampler")) return (void *)table->CreateSampler;
    if (!strcmp(name, "DestroySampler")) return (void *)table->DestroySampler;
    if (!strcmp(name, "CreateDescriptorSetLayout")) return (void *)table->CreateDescriptorSetLayout;
    if (!strcmp(name, "DestroyDescriptorSetLayout")) return (void *)table->DestroyDescriptorSetLayout;
    if (!strcmp(name, "CreateDescriptorPool")) return (void *)table->CreateDescriptorPool;
    if (!strcmp(name, "DestroyDescriptorPool")) return (void *)table->DestroyDescriptorPool;
    if (!strcmp(name, "ResetDescriptorPool")) return (void *)table->ResetDescriptorPool;
    if (!strcmp(name, "AllocateDescriptorSets")) return (void *)table->AllocateDescriptorSets;
    if (!strcmp(name, "FreeDescriptorSets")) return (void *)table->FreeDescriptorSets;
    if (!strcmp(name, "UpdateDescriptorSets")) return (void *)table->UpdateDescriptorSets;
    if (!strcmp(name, "CreateFramebuffer")) return (void *)table->CreateFramebuffer;
    if (!strcmp(name, "DestroyFramebuffer")) return (void *)table->DestroyFramebuffer;
    if (!strcmp(name, "CreateRenderPass")) return (void *)table->CreateRenderPass;
    if (!strcmp(name, "DestroyRenderPass")) return (void *)table->DestroyRenderPass;
    if (!strcmp(name, "GetRenderAreaGranularity")) return (void *)table->GetRenderAreaGranularity;
    if (!strcmp(name, "CreateCommandPool")) return (void *)table->CreateCommandPool;
    if (!strcmp(name, "DestroyCommandPool")) return (void *)table->DestroyCommandPool;
    if (!strcmp(name, "ResetCommandPool")) return (void *)table->ResetCommandPool;
    if (!strcmp(name, "AllocateCommandBuffers")) return (void *)table->AllocateCommandBuffers;
    if (!strcmp(name, "FreeCommandBuffers")) return (void *)table->FreeCommandBuffers;
    if (!strcmp(name, "BeginCommandBuffer")) return (void *)table->BeginCommandBuffer;
    if (!strcmp(name, "EndCommandBuffer")) return (void *)table->EndCommandBuffer;
    if (!strcmp(name, "ResetCommandBuffer")) return (void *)table->ResetCommandBuffer;
    if (!strcmp(name, "CmdBindPipeline")) return (void *)table->CmdBindPipeline;
    if (!strcmp(name, "CmdSetViewport")) return (void *)table->CmdSetViewport;
    if (!strcmp(name, "CmdSetScissor")) return (void *)table->CmdSetScissor;
    if (!strcmp(name, "CmdSetLineWidth")) return (void *)table->CmdSetLineWidth;
    if (!strcmp(name, "CmdSetDepthBias")) return (void *)table->CmdSetDepthBias;
    if (!strcmp(name, "CmdSetBlendConstants")) return (void *)table->CmdSetBlendConstants;
    if (!strcmp(name, "CmdSetDepthBounds")) return (void *)table->CmdSetDepthBounds;
    if (!strcmp(name, "CmdSetStencilCompareMask")) return (void *)table->CmdSetStencilCompareMask;
    if (!strcmp(name, "CmdSetStencilWriteMask")) return (void *)table->CmdSetStencilWriteMask;
    if (!strcmp(name, "CmdSetStencilReference")) return (void *)table->CmdSetStencilReference;
    if (!strcmp(name, "CmdBindDescriptorSets")) return (void *)table->CmdBindDescriptorSets;
    if (!strcmp(name, "CmdBindIndexBuffer")) return (void *)table->CmdBindIndexBuffer;
    if (!strcmp(name, "CmdBindVertexBuffers")) return (void *)table->CmdBindVertexBuffers;
    if (!strcmp(name, "CmdDraw")) return (void *)table->CmdDraw;
    if (!strcmp(name, "CmdDrawIndexed")) return (void *)table->CmdDrawIndexed;
    if (!strcmp(name, "CmdDrawIndirect")) return (void *)table->CmdDrawIndirect;
    if (!strcmp(name, "CmdDrawIndexedIndirect")) return (void *)table->CmdDrawIndexedIndirect;
    if (!strcmp(name, "CmdDispatch")) return (void *)table->CmdDispatch;
    if (!strcmp(name, "CmdDispatchIndirect")) return (void *)table->CmdDispatchIndirect;
    if (!strcmp(name, "CmdCopyBuffer")) return (void *)table->CmdCopyBuffer;
    if (!strcmp(name, "CmdCopyImage")) return (void *)table->CmdCopyImage;
    if (!strcmp(name, "CmdBlitImage")) return (void *)table->CmdBlitImage;
    if (!strcmp(name, "CmdCopyBufferToImage")) return (void *)table->CmdCopyBufferToImage;
    if (!strcmp(name, "CmdCopyImageToBuffer")) return (void *)table->CmdCopyImageToBuffer;
    if (!strcmp(name, "CmdUpdateBuffer")) return (void *)table->CmdUpdateBuffer;
    if (!strcmp(name, "CmdFillBuffer")) return (void *)table->CmdFillBuffer;
    if (!strcmp(name, "CmdClearColorImage")) return (void *)table->CmdClearColorImage;
    if (!strcmp(name, "CmdClearDepthStencilImage")) return (void *)table->CmdClearDepthStencilImage;
    if (!strcmp(name, "CmdClearAttachments")) return (void *)table->CmdClearAttachments;
    if (!strcmp(name, "CmdResolveImage")) return (void *)table->CmdResolveImage;
    if (!strcmp(name, "CmdSetEvent")) return (void *)table->CmdSetEvent;
    if (!strcmp(name, "CmdResetEvent")) return (void *)table->CmdResetEvent;
    if (!strcmp(name, "CmdWaitEvents")) return (void *)table->CmdWaitEvents;
    if (!strcmp(name, "CmdPipelineBarrier")) return (void *)table->CmdPipelineBarrier;
    if (!strcmp(name, "CmdBeginQuery")) return (void *)table->CmdBeginQuery;
    if (!strcmp(name, "CmdEndQuery")) return (void *)table->CmdEndQuery;
    if (!strcmp(name, "CmdResetQueryPool")) return (void *)table->CmdResetQueryPool;
    if (!strcmp(name, "CmdWriteTimestamp")) return (void *)table->CmdWriteTimestamp;
    if (!strcmp(name, "CmdCopyQueryPoolResults")) return (void *)table->CmdCopyQueryPoolResults;
    if (!strcmp(name, "CmdPushConstants")) return (void *)table->CmdPushConstants;
    if (!strcmp(name, "CmdBeginRenderPass")) return (void *)table->CmdBeginRenderPass;
    if (!strcmp(name, "CmdNextSubpass")) return (void *)table->CmdNextSubpass;
    if (!strcmp(name, "CmdEndRenderPass")) return (void *)table->CmdEndRenderPass;
    if (!strcmp(name, "CmdExecuteCommands")) return (void *)table->CmdExecuteCommands;

    // ---- Core 1_1 commands
    if (!strcmp(name, "BindBufferMemory2")) return (void *)table->BindBufferMemory2;
    if (!strcmp(name, "BindImageMemory2")) return (void *)table->BindImageMemory2;
    if (!strcmp(name, "GetDeviceGroupPeerMemoryFeatures")) return (void *)table->GetDeviceGroupPeerMemoryFeatures;
    if (!strcmp(name, "CmdSetDeviceMask")) return (void *)table->CmdSetDeviceMask;
    if (!strcmp(name, "CmdDispatchBase")) return (void *)table->CmdDispatchBase;
    if (!strcmp(name, "GetImageMemoryRequirements2")) return (void *)table->GetImageMemoryRequirements2;
    if (!strcmp(name, "GetBufferMemoryRequirements2")) return (void *)table->GetBufferMemoryRequirements2;
    if (!strcmp(name, "GetImageSparseMemoryRequirements2")) return (void *)table->GetImageSparseMemoryRequirements2;
    if (!strcmp(name, "TrimCommandPool")) return (void *)table->TrimCommandPool;
    if (!strcmp(name, "GetDeviceQueue2")) return (void *)table->GetDeviceQueue2;
    if (!strcmp(name, "CreateSamplerYcbcrConversion")) return (void *)table->CreateSamplerYcbcrConversion;
    if (!strcmp(name, "DestroySamplerYcbcrConversion")) return (void *)table->DestroySamplerYcbcrConversion;
    if (!strcmp(name, "CreateDescriptorUpdateTemplate")) return (void *)table->CreateDescriptorUpdateTemplate;
    if (!strcmp(name, "DestroyDescriptorUpdateTemplate")) return (void *)table->DestroyDescriptorUpdateTemplate;
    if (!strcmp(name, "UpdateDescriptorSetWithTemplate")) return (void *)table->UpdateDescriptorSetWithTemplate;
    if (!strcmp(name, "GetDescriptorSetLayoutSupport")) return (void *)table->GetDescriptorSetLayoutSupport;

    // ---- Core 1_2 commands
    if (!strcmp(name, "CmdDrawIndirectCount")) return (void *)table->CmdDrawIndirectCount;
    if (!strcmp(name, "CmdDrawIndexedIndirectCount")) return (void *)table->CmdDrawIndexedIndirectCount;
    if (!strcmp(name, "CreateRenderPass2")) return (void *)table->CreateRenderPass2;
    if (!strcmp(name, "CmdBeginRenderPass2")) return (void *)table->CmdBeginRenderPass2;
    if (!strcmp(name, "CmdNextSubpass2")) return (void *)table->CmdNextSubpass2;
    if (!strcmp(name, "CmdEndRenderPass2")) return (void *)table->CmdEndRenderPass2;
    if (!strcmp(name, "ResetQueryPool")) return (void *)table->ResetQueryPool;
    if (!strcmp(name, "GetSemaphoreCounterValue")) return (void *)table->GetSemaphoreCounterValue;
    if (!strcmp(name, "WaitSemaphores")) return (void *)table->WaitSemaphores;
    if (!strcmp(name, "SignalSemaphore")) return (void *)table->SignalSemaphore;
    if (!strcmp(name, "GetBufferDeviceAddress")) return (void *)table->GetBufferDeviceAddress;
    if (!strcmp(name, "GetBufferOpaqueCaptureAddress")) return (void *)table->GetBufferOpaqueCaptureAddress;
    if (!strcmp(name, "GetDeviceMemoryOpaqueCaptureAddress")) return (void *)table->GetDeviceMemoryOpaqueCaptureAddress;

    // ---- VK_KHR_swapchain extension commands
    if (!strcmp(name, "CreateSwapchainKHR")) return (void *)table->CreateSwapchainKHR;
    if (!strcmp(name, "DestroySwapchainKHR")) return (void *)table->DestroySwapchainKHR;
    if (!strcmp(name, "GetSwapchainImagesKHR")) return (void *)table->GetSwapchainImagesKHR;
    if (!strcmp(name, "AcquireNextImageKHR")) return (void *)table->AcquireNextImageKHR;
    if (!strcmp(name, "QueuePresentKHR")) return (void *)table->QueuePresentKHR;
    if (!strcmp(name, "GetDeviceGroupPresentCapabilitiesKHR")) return (void *)table->GetDeviceGroupPresentCapabilitiesKHR;
    if (!strcmp(name, "GetDeviceGroupSurfacePresentModesKHR")) return (void *)table->GetDeviceGroupSurfacePresentModesKHR;
    if (!strcmp(name, "AcquireNextImage2KHR")) return (void *)table->AcquireNextImage2KHR;

    // ---- VK_KHR_display_swapchain extension commands
    if (!strcmp(name, "CreateSharedSwapchainsKHR")) return (void *)table->CreateSharedSwapchainsKHR;

    // ---- VK_KHR_device_group extension commands
    if (!strcmp(name, "GetDeviceGroupPeerMemoryFeaturesKHR")) return (void *)table->GetDeviceGroupPeerMemoryFeaturesKHR;
    if (!strcmp(name, "CmdSetDeviceMaskKHR")) return (void *)table->CmdSetDeviceMaskKHR;
    if (!strcmp(name, "CmdDispatchBaseKHR")) return (void *)table->CmdDispatchBaseKHR;

    // ---- VK_KHR_maintenance1 extension commands
    if (!strcmp(name, "TrimCommandPoolKHR")) return (void *)table->TrimCommandPoolKHR;

    // ---- VK_KHR_external_memory_win32 extension commands
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (!strcmp(name, "GetMemoryWin32HandleKHR")) return (void *)table->GetMemoryWin32HandleKHR;
#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (!strcmp(name, "GetMemoryWin32HandlePropertiesKHR")) return (void *)table->GetMemoryWin32HandlePropertiesKHR;
#endif // VK_USE_PLATFORM_WIN32_KHR

    // ---- VK_KHR_external_memory_fd extension commands
    if (!strcmp(name, "GetMemoryFdKHR")) return (void *)table->GetMemoryFdKHR;
    if (!strcmp(name, "GetMemoryFdPropertiesKHR")) return (void *)table->GetMemoryFdPropertiesKHR;

    // ---- VK_KHR_external_semaphore_win32 extension commands
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (!strcmp(name, "ImportSemaphoreWin32HandleKHR")) return (void *)table->ImportSemaphoreWin32HandleKHR;
#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (!strcmp(name, "GetSemaphoreWin32HandleKHR")) return (void *)table->GetSemaphoreWin32HandleKHR;
#endif // VK_USE_PLATFORM_WIN32_KHR

    // ---- VK_KHR_external_semaphore_fd extension commands
    if (!strcmp(name, "ImportSemaphoreFdKHR")) return (void *)table->ImportSemaphoreFdKHR;
    if (!strcmp(name, "GetSemaphoreFdKHR")) return (void *)table->GetSemaphoreFdKHR;

    // ---- VK_KHR_push_descriptor extension commands
    if (!strcmp(name, "CmdPushDescriptorSetKHR")) return (void *)table->CmdPushDescriptorSetKHR;
    if (!strcmp(name, "CmdPushDescriptorSetWithTemplateKHR")) return (void *)table->CmdPushDescriptorSetWithTemplateKHR;

    // ---- VK_KHR_descriptor_update_template extension commands
    if (!strcmp(name, "CreateDescriptorUpdateTemplateKHR")) return (void *)table->CreateDescriptorUpdateTemplateKHR;
    if (!strcmp(name, "DestroyDescriptorUpdateTemplateKHR")) return (void *)table->DestroyDescriptorUpdateTemplateKHR;
    if (!strcmp(name, "UpdateDescriptorSetWithTemplateKHR")) return (void *)table->UpdateDescriptorSetWithTemplateKHR;

    // ---- VK_KHR_create_renderpass2 extension commands
    if (!strcmp(name, "CreateRenderPass2KHR")) return (void *)table->CreateRenderPass2KHR;
    if (!strcmp(name, "CmdBeginRenderPass2KHR")) return (void *)table->CmdBeginRenderPass2KHR;
    if (!strcmp(name, "CmdNextSubpass2KHR")) return (void *)table->CmdNextSubpass2KHR;
    if (!strcmp(name, "CmdEndRenderPass2KHR")) return (void *)table->CmdEndRenderPass2KHR;

    // ---- VK_KHR_shared_presentable_image extension commands
    if (!strcmp(name, "GetSwapchainStatusKHR")) return (void *)table->GetSwapchainStatusKHR;

    // ---- VK_KHR_external_fence_win32 extension commands
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (!strcmp(name, "ImportFenceWin32HandleKHR")) return (void *)table->ImportFenceWin32HandleKHR;
#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (!strcmp(name, "GetFenceWin32HandleKHR")) return (void *)table->GetFenceWin32HandleKHR;
#endif // VK_USE_PLATFORM_WIN32_KHR

    // ---- VK_KHR_external_fence_fd extension commands
    if (!strcmp(name, "ImportFenceFdKHR")) return (void *)table->ImportFenceFdKHR;
    if (!strcmp(name, "GetFenceFdKHR")) return (void *)table->GetFenceFdKHR;

    // ---- VK_KHR_performance_query extension commands
    if (!strcmp(name, "AcquireProfilingLockKHR")) return (void *)table->AcquireProfilingLockKHR;
    if (!strcmp(name, "ReleaseProfilingLockKHR")) return (void *)table->ReleaseProfilingLockKHR;

    // ---- VK_KHR_get_memory_requirements2 extension commands
    if (!strcmp(name, "GetImageMemoryRequirements2KHR")) return (void *)table->GetImageMemoryRequirements2KHR;
    if (!strcmp(name, "GetBufferMemoryRequirements2KHR")) return (void *)table->GetBufferMemoryRequirements2KHR;
    if (!strcmp(name, "GetImageSparseMemoryRequirements2KHR")) return (void *)table->GetImageSparseMemoryRequirements2KHR;

    // ---- VK_KHR_sampler_ycbcr_conversion extension commands
    if (!strcmp(name, "CreateSamplerYcbcrConversionKHR")) return (void *)table->CreateSamplerYcbcrConversionKHR;
    if (!strcmp(name, "DestroySamplerYcbcrConversionKHR")) return (void *)table->DestroySamplerYcbcrConversionKHR;

    // ---- VK_KHR_bind_memory2 extension commands
    if (!strcmp(name, "BindBufferMemory2KHR")) return (void *)table->BindBufferMemory2KHR;
    if (!strcmp(name, "BindImageMemory2KHR")) return (void *)table->BindImageMemory2KHR;

    // ---- VK_KHR_maintenance3 extension commands
    if (!strcmp(name, "GetDescriptorSetLayoutSupportKHR")) return (void *)table->GetDescriptorSetLayoutSupportKHR;

    // ---- VK_KHR_draw_indirect_count extension commands
    if (!strcmp(name, "CmdDrawIndirectCountKHR")) return (void *)table->CmdDrawIndirectCountKHR;
    if (!strcmp(name, "CmdDrawIndexedIndirectCountKHR")) return (void *)table->CmdDrawIndexedIndirectCountKHR;

    // ---- VK_KHR_timeline_semaphore extension commands
    if (!strcmp(name, "GetSemaphoreCounterValueKHR")) return (void *)table->GetSemaphoreCounterValueKHR;
    if (!strcmp(name, "WaitSemaphoresKHR")) return (void *)table->WaitSemaphoresKHR;
    if (!strcmp(name, "SignalSemaphoreKHR")) return (void *)table->SignalSemaphoreKHR;

    // ---- VK_KHR_buffer_device_address extension commands
    if (!strcmp(name, "GetBufferDeviceAddressKHR")) return (void *)table->GetBufferDeviceAddressKHR;
    if (!strcmp(name, "GetBufferOpaqueCaptureAddressKHR")) return (void *)table->GetBufferOpaqueCaptureAddressKHR;
    if (!strcmp(name, "GetDeviceMemoryOpaqueCaptureAddressKHR")) return (void *)table->GetDeviceMemoryOpaqueCaptureAddressKHR;

    // ---- VK_KHR_deferred_host_operations extension commands
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp(name, "CreateDeferredOperationKHR")) return (void *)table->CreateDeferredOperationKHR;
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp(name, "DestroyDeferredOperationKHR")) return (void *)table->DestroyDeferredOperationKHR;
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp(name, "GetDeferredOperationMaxConcurrencyKHR")) return (void *)table->GetDeferredOperationMaxConcurrencyKHR;
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp(name, "GetDeferredOperationResultKHR")) return (void *)table->GetDeferredOperationResultKHR;
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp(name, "DeferredOperationJoinKHR")) return (void *)table->DeferredOperationJoinKHR;
#endif // VK_ENABLE_BETA_EXTENSIONS

    // ---- VK_KHR_pipeline_executable_properties extension commands
    if (!strcmp(name, "GetPipelineExecutablePropertiesKHR")) return (void *)table->GetPipelineExecutablePropertiesKHR;
    if (!strcmp(name, "GetPipelineExecutableStatisticsKHR")) return (void *)table->GetPipelineExecutableStatisticsKHR;
    if (!strcmp(name, "GetPipelineExecutableInternalRepresentationsKHR")) return (void *)table->GetPipelineExecutableInternalRepresentationsKHR;

    // ---- VK_KHR_copy_commands2 extension commands
    if (!strcmp(name, "CmdCopyBuffer2KHR")) return (void *)table->CmdCopyBuffer2KHR;
    if (!strcmp(name, "CmdCopyImage2KHR")) return (void *)table->CmdCopyImage2KHR;
    if (!strcmp(name, "CmdCopyBufferToImage2KHR")) return (void *)table->CmdCopyBufferToImage2KHR;
    if (!strcmp(name, "CmdCopyImageToBuffer2KHR")) return (void *)table->CmdCopyImageToBuffer2KHR;
    if (!strcmp(name, "CmdBlitImage2KHR")) return (void *)table->CmdBlitImage2KHR;
    if (!strcmp(name, "CmdResolveImage2KHR")) return (void *)table->CmdResolveImage2KHR;

    // ---- VK_EXT_debug_marker extension commands
    if (!strcmp(name, "DebugMarkerSetObjectTagEXT")) return (void *)table->DebugMarkerSetObjectTagEXT;
    if (!strcmp(name, "DebugMarkerSetObjectNameEXT")) return (void *)table->DebugMarkerSetObjectNameEXT;
    if (!strcmp(name, "CmdDebugMarkerBeginEXT")) return (void *)table->CmdDebugMarkerBeginEXT;
    if (!strcmp(name, "CmdDebugMarkerEndEXT")) return (void *)table->CmdDebugMarkerEndEXT;
    if (!strcmp(name, "CmdDebugMarkerInsertEXT")) return (void *)table->CmdDebugMarkerInsertEXT;

    // ---- VK_EXT_transform_feedback extension commands
    if (!strcmp(name, "CmdBindTransformFeedbackBuffersEXT")) return (void *)table->CmdBindTransformFeedbackBuffersEXT;
    if (!strcmp(name, "CmdBeginTransformFeedbackEXT")) return (void *)table->CmdBeginTransformFeedbackEXT;
    if (!strcmp(name, "CmdEndTransformFeedbackEXT")) return (void *)table->CmdEndTransformFeedbackEXT;
    if (!strcmp(name, "CmdBeginQueryIndexedEXT")) return (void *)table->CmdBeginQueryIndexedEXT;
    if (!strcmp(name, "CmdEndQueryIndexedEXT")) return (void *)table->CmdEndQueryIndexedEXT;
    if (!strcmp(name, "CmdDrawIndirectByteCountEXT")) return (void *)table->CmdDrawIndirectByteCountEXT;

    // ---- VK_NVX_image_view_handle extension commands
    if (!strcmp(name, "GetImageViewHandleNVX")) return (void *)table->GetImageViewHandleNVX;
    if (!strcmp(name, "GetImageViewAddressNVX")) return (void *)table->GetImageViewAddressNVX;

    // ---- VK_AMD_draw_indirect_count extension commands
    if (!strcmp(name, "CmdDrawIndirectCountAMD")) return (void *)table->CmdDrawIndirectCountAMD;
    if (!strcmp(name, "CmdDrawIndexedIndirectCountAMD")) return (void *)table->CmdDrawIndexedIndirectCountAMD;

    // ---- VK_AMD_shader_info extension commands
    if (!strcmp(name, "GetShaderInfoAMD")) return (void *)table->GetShaderInfoAMD;

    // ---- VK_NV_external_memory_win32 extension commands
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (!strcmp(name, "GetMemoryWin32HandleNV")) return (void *)table->GetMemoryWin32HandleNV;
#endif // VK_USE_PLATFORM_WIN32_KHR

    // ---- VK_EXT_conditional_rendering extension commands
    if (!strcmp(name, "CmdBeginConditionalRenderingEXT")) return (void *)table->CmdBeginConditionalRenderingEXT;
    if (!strcmp(name, "CmdEndConditionalRenderingEXT")) return (void *)table->CmdEndConditionalRenderingEXT;

    // ---- VK_NV_clip_space_w_scaling extension commands
    if (!strcmp(name, "CmdSetViewportWScalingNV")) return (void *)table->CmdSetViewportWScalingNV;

    // ---- VK_EXT_display_control extension commands
    if (!strcmp(name, "DisplayPowerControlEXT")) return (void *)table->DisplayPowerControlEXT;
    if (!strcmp(name, "RegisterDeviceEventEXT")) return (void *)table->RegisterDeviceEventEXT;
    if (!strcmp(name, "RegisterDisplayEventEXT")) return (void *)table->RegisterDisplayEventEXT;
    if (!strcmp(name, "GetSwapchainCounterEXT")) return (void *)table->GetSwapchainCounterEXT;

    // ---- VK_GOOGLE_display_timing extension commands
    if (!strcmp(name, "GetRefreshCycleDurationGOOGLE")) return (void *)table->GetRefreshCycleDurationGOOGLE;
    if (!strcmp(name, "GetPastPresentationTimingGOOGLE")) return (void *)table->GetPastPresentationTimingGOOGLE;

    // ---- VK_EXT_discard_rectangles extension commands
    if (!strcmp(name, "CmdSetDiscardRectangleEXT")) return (void *)table->CmdSetDiscardRectangleEXT;

    // ---- VK_EXT_hdr_metadata extension commands
    if (!strcmp(name, "SetHdrMetadataEXT")) return (void *)table->SetHdrMetadataEXT;

    // ---- VK_EXT_debug_utils extension commands
    if (!strcmp(name, "SetDebugUtilsObjectNameEXT")) return (void *)table->SetDebugUtilsObjectNameEXT;
    if (!strcmp(name, "SetDebugUtilsObjectTagEXT")) return (void *)table->SetDebugUtilsObjectTagEXT;
    if (!strcmp(name, "QueueBeginDebugUtilsLabelEXT")) return (void *)table->QueueBeginDebugUtilsLabelEXT;
    if (!strcmp(name, "QueueEndDebugUtilsLabelEXT")) return (void *)table->QueueEndDebugUtilsLabelEXT;
    if (!strcmp(name, "QueueInsertDebugUtilsLabelEXT")) return (void *)table->QueueInsertDebugUtilsLabelEXT;
    if (!strcmp(name, "CmdBeginDebugUtilsLabelEXT")) return (void *)table->CmdBeginDebugUtilsLabelEXT;
    if (!strcmp(name, "CmdEndDebugUtilsLabelEXT")) return (void *)table->CmdEndDebugUtilsLabelEXT;
    if (!strcmp(name, "CmdInsertDebugUtilsLabelEXT")) return (void *)table->CmdInsertDebugUtilsLabelEXT;

    // ---- VK_ANDROID_external_memory_android_hardware_buffer extension commands
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    if (!strcmp(name, "GetAndroidHardwareBufferPropertiesANDROID")) return (void *)table->GetAndroidHardwareBufferPropertiesANDROID;
#endif // VK_USE_PLATFORM_ANDROID_KHR
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    if (!strcmp(name, "GetMemoryAndroidHardwareBufferANDROID")) return (void *)table->GetMemoryAndroidHardwareBufferANDROID;
#endif // VK_USE_PLATFORM_ANDROID_KHR

    // ---- VK_EXT_sample_locations extension commands
    if (!strcmp(name, "CmdSetSampleLocationsEXT")) return (void *)table->CmdSetSampleLocationsEXT;

    // ---- VK_EXT_image_drm_format_modifier extension commands
    if (!strcmp(name, "GetImageDrmFormatModifierPropertiesEXT")) return (void *)table->GetImageDrmFormatModifierPropertiesEXT;

    // ---- VK_EXT_validation_cache extension commands
    if (!strcmp(name, "CreateValidationCacheEXT")) return (void *)table->CreateValidationCacheEXT;
    if (!strcmp(name, "DestroyValidationCacheEXT")) return (void *)table->DestroyValidationCacheEXT;
    if (!strcmp(name, "MergeValidationCachesEXT")) return (void *)table->MergeValidationCachesEXT;
    if (!strcmp(name, "GetValidationCacheDataEXT")) return (void *)table->GetValidationCacheDataEXT;

    // ---- VK_NV_shading_rate_image extension commands
    if (!strcmp(name, "CmdBindShadingRateImageNV")) return (void *)table->CmdBindShadingRateImageNV;
    if (!strcmp(name, "CmdSetViewportShadingRatePaletteNV")) return (void *)table->CmdSetViewportShadingRatePaletteNV;
    if (!strcmp(name, "CmdSetCoarseSampleOrderNV")) return (void *)table->CmdSetCoarseSampleOrderNV;

    // ---- VK_NV_ray_tracing extension commands
    if (!strcmp(name, "CreateAccelerationStructureNV")) return (void *)table->CreateAccelerationStructureNV;
    if (!strcmp(name, "DestroyAccelerationStructureKHR")) return (void *)table->DestroyAccelerationStructureKHR;
    if (!strcmp(name, "DestroyAccelerationStructureNV")) return (void *)table->DestroyAccelerationStructureNV;
    if (!strcmp(name, "GetAccelerationStructureMemoryRequirementsNV")) return (void *)table->GetAccelerationStructureMemoryRequirementsNV;
    if (!strcmp(name, "BindAccelerationStructureMemoryKHR")) return (void *)table->BindAccelerationStructureMemoryKHR;
    if (!strcmp(name, "BindAccelerationStructureMemoryNV")) return (void *)table->BindAccelerationStructureMemoryNV;
    if (!strcmp(name, "CmdBuildAccelerationStructureNV")) return (void *)table->CmdBuildAccelerationStructureNV;
    if (!strcmp(name, "CmdCopyAccelerationStructureNV")) return (void *)table->CmdCopyAccelerationStructureNV;
    if (!strcmp(name, "CmdTraceRaysNV")) return (void *)table->CmdTraceRaysNV;
    if (!strcmp(name, "CreateRayTracingPipelinesNV")) return (void *)table->CreateRayTracingPipelinesNV;
    if (!strcmp(name, "GetRayTracingShaderGroupHandlesKHR")) return (void *)table->GetRayTracingShaderGroupHandlesKHR;
    if (!strcmp(name, "GetRayTracingShaderGroupHandlesNV")) return (void *)table->GetRayTracingShaderGroupHandlesNV;
    if (!strcmp(name, "GetAccelerationStructureHandleNV")) return (void *)table->GetAccelerationStructureHandleNV;
    if (!strcmp(name, "CmdWriteAccelerationStructuresPropertiesKHR")) return (void *)table->CmdWriteAccelerationStructuresPropertiesKHR;
    if (!strcmp(name, "CmdWriteAccelerationStructuresPropertiesNV")) return (void *)table->CmdWriteAccelerationStructuresPropertiesNV;
    if (!strcmp(name, "CompileDeferredNV")) return (void *)table->CompileDeferredNV;

    // ---- VK_EXT_external_memory_host extension commands
    if (!strcmp(name, "GetMemoryHostPointerPropertiesEXT")) return (void *)table->GetMemoryHostPointerPropertiesEXT;

    // ---- VK_AMD_buffer_marker extension commands
    if (!strcmp(name, "CmdWriteBufferMarkerAMD")) return (void *)table->CmdWriteBufferMarkerAMD;

    // ---- VK_EXT_calibrated_timestamps extension commands
    if (!strcmp(name, "GetCalibratedTimestampsEXT")) return (void *)table->GetCalibratedTimestampsEXT;

    // ---- VK_NV_mesh_shader extension commands
    if (!strcmp(name, "CmdDrawMeshTasksNV")) return (void *)table->CmdDrawMeshTasksNV;
    if (!strcmp(name, "CmdDrawMeshTasksIndirectNV")) return (void *)table->CmdDrawMeshTasksIndirectNV;
    if (!strcmp(name, "CmdDrawMeshTasksIndirectCountNV")) return (void *)table->CmdDrawMeshTasksIndirectCountNV;

    // ---- VK_NV_scissor_exclusive extension commands
    if (!strcmp(name, "CmdSetExclusiveScissorNV")) return (void *)table->CmdSetExclusiveScissorNV;

    // ---- VK_NV_device_diagnostic_checkpoints extension commands
    if (!strcmp(name, "CmdSetCheckpointNV")) return (void *)table->CmdSetCheckpointNV;
    if (!strcmp(name, "GetQueueCheckpointDataNV")) return (void *)table->GetQueueCheckpointDataNV;

    // ---- VK_INTEL_performance_query extension commands
    if (!strcmp(name, "InitializePerformanceApiINTEL")) return (void *)table->InitializePerformanceApiINTEL;
    if (!strcmp(name, "UninitializePerformanceApiINTEL")) return (void *)table->UninitializePerformanceApiINTEL;
    if (!strcmp(name, "CmdSetPerformanceMarkerINTEL")) return (void *)table->CmdSetPerformanceMarkerINTEL;
    if (!strcmp(name, "CmdSetPerformanceStreamMarkerINTEL")) return (void *)table->CmdSetPerformanceStreamMarkerINTEL;
    if (!strcmp(name, "CmdSetPerformanceOverrideINTEL")) return (void *)table->CmdSetPerformanceOverrideINTEL;
    if (!strcmp(name, "AcquirePerformanceConfigurationINTEL")) return (void *)table->AcquirePerformanceConfigurationINTEL;
    if (!strcmp(name, "ReleasePerformanceConfigurationINTEL")) return (void *)table->ReleasePerformanceConfigurationINTEL;
    if (!strcmp(name, "QueueSetPerformanceConfigurationINTEL")) return (void *)table->QueueSetPerformanceConfigurationINTEL;
    if (!strcmp(name, "GetPerformanceParameterINTEL")) return (void *)table->GetPerformanceParameterINTEL;

    // ---- VK_AMD_display_native_hdr extension commands
    if (!strcmp(name, "SetLocalDimmingAMD")) return (void *)table->SetLocalDimmingAMD;

    // ---- VK_EXT_buffer_device_address extension commands
    if (!strcmp(name, "GetBufferDeviceAddressEXT")) return (void *)table->GetBufferDeviceAddressEXT;

    // ---- VK_EXT_full_screen_exclusive extension commands
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (!strcmp(name, "AcquireFullScreenExclusiveModeEXT")) return (void *)table->AcquireFullScreenExclusiveModeEXT;
#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (!strcmp(name, "ReleaseFullScreenExclusiveModeEXT")) return (void *)table->ReleaseFullScreenExclusiveModeEXT;
#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (!strcmp(name, "GetDeviceGroupSurfacePresentModes2EXT")) return (void *)table->GetDeviceGroupSurfacePresentModes2EXT;
#endif // VK_USE_PLATFORM_WIN32_KHR

    // ---- VK_EXT_line_rasterization extension commands
    if (!strcmp(name, "CmdSetLineStippleEXT")) return (void *)table->CmdSetLineStippleEXT;

    // ---- VK_EXT_host_query_reset extension commands
    if (!strcmp(name, "ResetQueryPoolEXT")) return (void *)table->ResetQueryPoolEXT;

    // ---- VK_EXT_extended_dynamic_state extension commands
    if (!strcmp(name, "CmdSetCullModeEXT")) return (void *)table->CmdSetCullModeEXT;
    if (!strcmp(name, "CmdSetFrontFaceEXT")) return (void *)table->CmdSetFrontFaceEXT;
    if (!strcmp(name, "CmdSetPrimitiveTopologyEXT")) return (void *)table->CmdSetPrimitiveTopologyEXT;
    if (!strcmp(name, "CmdSetViewportWithCountEXT")) return (void *)table->CmdSetViewportWithCountEXT;
    if (!strcmp(name, "CmdSetScissorWithCountEXT")) return (void *)table->CmdSetScissorWithCountEXT;
    if (!strcmp(name, "CmdBindVertexBuffers2EXT")) return (void *)table->CmdBindVertexBuffers2EXT;
    if (!strcmp(name, "CmdSetDepthTestEnableEXT")) return (void *)table->CmdSetDepthTestEnableEXT;
    if (!strcmp(name, "CmdSetDepthWriteEnableEXT")) return (void *)table->CmdSetDepthWriteEnableEXT;
    if (!strcmp(name, "CmdSetDepthCompareOpEXT")) return (void *)table->CmdSetDepthCompareOpEXT;
    if (!strcmp(name, "CmdSetDepthBoundsTestEnableEXT")) return (void *)table->CmdSetDepthBoundsTestEnableEXT;
    if (!strcmp(name, "CmdSetStencilTestEnableEXT")) return (void *)table->CmdSetStencilTestEnableEXT;
    if (!strcmp(name, "CmdSetStencilOpEXT")) return (void *)table->CmdSetStencilOpEXT;

    // ---- VK_NV_device_generated_commands extension commands
    if (!strcmp(name, "GetGeneratedCommandsMemoryRequirementsNV")) return (void *)table->GetGeneratedCommandsMemoryRequirementsNV;
    if (!strcmp(name, "CmdPreprocessGeneratedCommandsNV")) return (void *)table->CmdPreprocessGeneratedCommandsNV;
    if (!strcmp(name, "CmdExecuteGeneratedCommandsNV")) return (void *)table->CmdExecuteGeneratedCommandsNV;
    if (!strcmp(name, "CmdBindPipelineShaderGroupNV")) return (void *)table->CmdBindPipelineShaderGroupNV;
    if (!strcmp(name, "CreateIndirectCommandsLayoutNV")) return (void *)table->CreateIndirectCommandsLayoutNV;
    if (!strcmp(name, "DestroyIndirectCommandsLayoutNV")) return (void *)table->DestroyIndirectCommandsLayoutNV;

    // ---- VK_EXT_private_data extension commands
    if (!strcmp(name, "CreatePrivateDataSlotEXT")) return (void *)table->CreatePrivateDataSlotEXT;
    if (!strcmp(name, "DestroyPrivateDataSlotEXT")) return (void *)table->DestroyPrivateDataSlotEXT;
    if (!strcmp(name, "SetPrivateDataEXT")) return (void *)table->SetPrivateDataEXT;
    if (!strcmp(name, "GetPrivateDataEXT")) return (void *)table->GetPrivateDataEXT;

    // ---- VK_KHR_ray_tracing extension commands
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp(name, "CreateAccelerationStructureKHR")) return (void *)table->CreateAccelerationStructureKHR;
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp(name, "GetAccelerationStructureMemoryRequirementsKHR")) return (void *)table->GetAccelerationStructureMemoryRequirementsKHR;
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp(name, "CmdBuildAccelerationStructureKHR")) return (void *)table->CmdBuildAccelerationStructureKHR;
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp(name, "CmdBuildAccelerationStructureIndirectKHR")) return (void *)table->CmdBuildAccelerationStructureIndirectKHR;
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp(name, "BuildAccelerationStructureKHR")) return (void *)table->BuildAccelerationStructureKHR;
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp(name, "CopyAccelerationStructureKHR")) return (void *)table->CopyAccelerationStructureKHR;
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp(name, "CopyAccelerationStructureToMemoryKHR")) return (void *)table->CopyAccelerationStructureToMemoryKHR;
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp(name, "CopyMemoryToAccelerationStructureKHR")) return (void *)table->CopyMemoryToAccelerationStructureKHR;
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp(name, "WriteAccelerationStructuresPropertiesKHR")) return (void *)table->WriteAccelerationStructuresPropertiesKHR;
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp(name, "CmdCopyAccelerationStructureKHR")) return (void *)table->CmdCopyAccelerationStructureKHR;
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp(name, "CmdCopyAccelerationStructureToMemoryKHR")) return (void *)table->CmdCopyAccelerationStructureToMemoryKHR;
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp(name, "CmdCopyMemoryToAccelerationStructureKHR")) return (void *)table->CmdCopyMemoryToAccelerationStructureKHR;
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp(name, "CmdTraceRaysKHR")) return (void *)table->CmdTraceRaysKHR;
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp(name, "CreateRayTracingPipelinesKHR")) return (void *)table->CreateRayTracingPipelinesKHR;
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp(name, "GetAccelerationStructureDeviceAddressKHR")) return (void *)table->GetAccelerationStructureDeviceAddressKHR;
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp(name, "GetRayTracingCaptureReplayShaderGroupHandlesKHR")) return (void *)table->GetRayTracingCaptureReplayShaderGroupHandlesKHR;
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp(name, "CmdTraceRaysIndirectKHR")) return (void *)table->CmdTraceRaysIndirectKHR;
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp(name, "GetDeviceAccelerationStructureCompatibilityKHR")) return (void *)table->GetDeviceAccelerationStructureCompatibilityKHR;
#endif // VK_ENABLE_BETA_EXTENSIONS

    return NULL;
}

// Instance command lookup function
VKAPI_ATTR void* VKAPI_CALL loader_lookup_instance_dispatch_table(const VkLayerInstanceDispatchTable *table, const char *name,
                                                                 bool *found_name) {
    if (!name || name[0] != 'v' || name[1] != 'k') {
        *found_name = false;
        return NULL;
    }

    *found_name = true;
    name += 2;

    // ---- Core 1_0 commands
    if (!strcmp(name, "DestroyInstance")) return (void *)table->DestroyInstance;
    if (!strcmp(name, "EnumeratePhysicalDevices")) return (void *)table->EnumeratePhysicalDevices;
    if (!strcmp(name, "GetPhysicalDeviceFeatures")) return (void *)table->GetPhysicalDeviceFeatures;
    if (!strcmp(name, "GetPhysicalDeviceFormatProperties")) return (void *)table->GetPhysicalDeviceFormatProperties;
    if (!strcmp(name, "GetPhysicalDeviceImageFormatProperties")) return (void *)table->GetPhysicalDeviceImageFormatProperties;
    if (!strcmp(name, "GetPhysicalDeviceProperties")) return (void *)table->GetPhysicalDeviceProperties;
    if (!strcmp(name, "GetPhysicalDeviceQueueFamilyProperties")) return (void *)table->GetPhysicalDeviceQueueFamilyProperties;
    if (!strcmp(name, "GetPhysicalDeviceMemoryProperties")) return (void *)table->GetPhysicalDeviceMemoryProperties;
    if (!strcmp(name, "GetInstanceProcAddr")) return (void *)table->GetInstanceProcAddr;
    if (!strcmp(name, "EnumerateDeviceExtensionProperties")) return (void *)table->EnumerateDeviceExtensionProperties;
    if (!strcmp(name, "EnumerateDeviceLayerProperties")) return (void *)table->EnumerateDeviceLayerProperties;
    if (!strcmp(name, "GetPhysicalDeviceSparseImageFormatProperties")) return (void *)table->GetPhysicalDeviceSparseImageFormatProperties;

    // ---- Core 1_1 commands
    if (!strcmp(name, "EnumeratePhysicalDeviceGroups")) return (void *)table->EnumeratePhysicalDeviceGroups;
    if (!strcmp(name, "GetPhysicalDeviceFeatures2")) return (void *)table->GetPhysicalDeviceFeatures2;
    if (!strcmp(name, "GetPhysicalDeviceProperties2")) return (void *)table->GetPhysicalDeviceProperties2;
    if (!strcmp(name, "GetPhysicalDeviceFormatProperties2")) return (void *)table->GetPhysicalDeviceFormatProperties2;
    if (!strcmp(name, "GetPhysicalDeviceImageFormatProperties2")) return (void *)table->GetPhysicalDeviceImageFormatProperties2;
    if (!strcmp(name, "GetPhysicalDeviceQueueFamilyProperties2")) return (void *)table->GetPhysicalDeviceQueueFamilyProperties2;
    if (!strcmp(name, "GetPhysicalDeviceMemoryProperties2")) return (void *)table->GetPhysicalDeviceMemoryProperties2;
    if (!strcmp(name, "GetPhysicalDeviceSparseImageFormatProperties2")) return (void *)table->GetPhysicalDeviceSparseImageFormatProperties2;
    if (!strcmp(name, "GetPhysicalDeviceExternalBufferProperties")) return (void *)table->GetPhysicalDeviceExternalBufferProperties;
    if (!strcmp(name, "GetPhysicalDeviceExternalFenceProperties")) return (void *)table->GetPhysicalDeviceExternalFenceProperties;
    if (!strcmp(name, "GetPhysicalDeviceExternalSemaphoreProperties")) return (void *)table->GetPhysicalDeviceExternalSemaphoreProperties;

    // ---- VK_KHR_surface extension commands
    if (!strcmp(name, "DestroySurfaceKHR")) return (void *)table->DestroySurfaceKHR;
    if (!strcmp(name, "GetPhysicalDeviceSurfaceSupportKHR")) return (void *)table->GetPhysicalDeviceSurfaceSupportKHR;
    if (!strcmp(name, "GetPhysicalDeviceSurfaceCapabilitiesKHR")) return (void *)table->GetPhysicalDeviceSurfaceCapabilitiesKHR;
    if (!strcmp(name, "GetPhysicalDeviceSurfaceFormatsKHR")) return (void *)table->GetPhysicalDeviceSurfaceFormatsKHR;
    if (!strcmp(name, "GetPhysicalDeviceSurfacePresentModesKHR")) return (void *)table->GetPhysicalDeviceSurfacePresentModesKHR;

    // ---- VK_KHR_swapchain extension commands
    if (!strcmp(name, "GetPhysicalDevicePresentRectanglesKHR")) return (void *)table->GetPhysicalDevicePresentRectanglesKHR;

    // ---- VK_KHR_display extension commands
    if (!strcmp(name, "GetPhysicalDeviceDisplayPropertiesKHR")) return (void *)table->GetPhysicalDeviceDisplayPropertiesKHR;
    if (!strcmp(name, "GetPhysicalDeviceDisplayPlanePropertiesKHR")) return (void *)table->GetPhysicalDeviceDisplayPlanePropertiesKHR;
    if (!strcmp(name, "GetDisplayPlaneSupportedDisplaysKHR")) return (void *)table->GetDisplayPlaneSupportedDisplaysKHR;
    if (!strcmp(name, "GetDisplayModePropertiesKHR")) return (void *)table->GetDisplayModePropertiesKHR;
    if (!strcmp(name, "CreateDisplayModeKHR")) return (void *)table->CreateDisplayModeKHR;
    if (!strcmp(name, "GetDisplayPlaneCapabilitiesKHR")) return (void *)table->GetDisplayPlaneCapabilitiesKHR;
    if (!strcmp(name, "CreateDisplayPlaneSurfaceKHR")) return (void *)table->CreateDisplayPlaneSurfaceKHR;

    // ---- VK_KHR_xlib_surface extension commands
#ifdef VK_USE_PLATFORM_XLIB_KHR
    if (!strcmp(name, "CreateXlibSurfaceKHR")) return (void *)table->CreateXlibSurfaceKHR;
#endif // VK_USE_PLATFORM_XLIB_KHR
#ifdef VK_USE_PLATFORM_XLIB_KHR
    if (!strcmp(name, "GetPhysicalDeviceXlibPresentationSupportKHR")) return (void *)table->GetPhysicalDeviceXlibPresentationSupportKHR;
#endif // VK_USE_PLATFORM_XLIB_KHR

    // ---- VK_KHR_xcb_surface extension commands
#ifdef VK_USE_PLATFORM_XCB_KHR
    if (!strcmp(name, "CreateXcbSurfaceKHR")) return (void *)table->CreateXcbSurfaceKHR;
#endif // VK_USE_PLATFORM_XCB_KHR
#ifdef VK_USE_PLATFORM_XCB_KHR
    if (!strcmp(name, "GetPhysicalDeviceXcbPresentationSupportKHR")) return (void *)table->GetPhysicalDeviceXcbPresentationSupportKHR;
#endif // VK_USE_PLATFORM_XCB_KHR

    // ---- VK_KHR_wayland_surface extension commands
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    if (!strcmp(name, "CreateWaylandSurfaceKHR")) return (void *)table->CreateWaylandSurfaceKHR;
#endif // VK_USE_PLATFORM_WAYLAND_KHR
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    if (!strcmp(name, "GetPhysicalDeviceWaylandPresentationSupportKHR")) return (void *)table->GetPhysicalDeviceWaylandPresentationSupportKHR;
#endif // VK_USE_PLATFORM_WAYLAND_KHR

    // ---- VK_KHR_android_surface extension commands
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    if (!strcmp(name, "CreateAndroidSurfaceKHR")) return (void *)table->CreateAndroidSurfaceKHR;
#endif // VK_USE_PLATFORM_ANDROID_KHR

    // ---- VK_KHR_win32_surface extension commands
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (!strcmp(name, "CreateWin32SurfaceKHR")) return (void *)table->CreateWin32SurfaceKHR;
#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (!strcmp(name, "GetPhysicalDeviceWin32PresentationSupportKHR")) return (void *)table->GetPhysicalDeviceWin32PresentationSupportKHR;
#endif // VK_USE_PLATFORM_WIN32_KHR

    // ---- VK_KHR_get_physical_device_properties2 extension commands
    if (!strcmp(name, "GetPhysicalDeviceFeatures2KHR")) return (void *)table->GetPhysicalDeviceFeatures2KHR;
    if (!strcmp(name, "GetPhysicalDeviceProperties2KHR")) return (void *)table->GetPhysicalDeviceProperties2KHR;
    if (!strcmp(name, "GetPhysicalDeviceFormatProperties2KHR")) return (void *)table->GetPhysicalDeviceFormatProperties2KHR;
    if (!strcmp(name, "GetPhysicalDeviceImageFormatProperties2KHR")) return (void *)table->GetPhysicalDeviceImageFormatProperties2KHR;
    if (!strcmp(name, "GetPhysicalDeviceQueueFamilyProperties2KHR")) return (void *)table->GetPhysicalDeviceQueueFamilyProperties2KHR;
    if (!strcmp(name, "GetPhysicalDeviceMemoryProperties2KHR")) return (void *)table->GetPhysicalDeviceMemoryProperties2KHR;
    if (!strcmp(name, "GetPhysicalDeviceSparseImageFormatProperties2KHR")) return (void *)table->GetPhysicalDeviceSparseImageFormatProperties2KHR;

    // ---- VK_KHR_device_group_creation extension commands
    if (!strcmp(name, "EnumeratePhysicalDeviceGroupsKHR")) return (void *)table->EnumeratePhysicalDeviceGroupsKHR;

    // ---- VK_KHR_external_memory_capabilities extension commands
    if (!strcmp(name, "GetPhysicalDeviceExternalBufferPropertiesKHR")) return (void *)table->GetPhysicalDeviceExternalBufferPropertiesKHR;

    // ---- VK_KHR_external_semaphore_capabilities extension commands
    if (!strcmp(name, "GetPhysicalDeviceExternalSemaphorePropertiesKHR")) return (void *)table->GetPhysicalDeviceExternalSemaphorePropertiesKHR;

    // ---- VK_KHR_external_fence_capabilities extension commands
    if (!strcmp(name, "GetPhysicalDeviceExternalFencePropertiesKHR")) return (void *)table->GetPhysicalDeviceExternalFencePropertiesKHR;

    // ---- VK_KHR_performance_query extension commands
    if (!strcmp(name, "EnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR")) return (void *)table->EnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR;
    if (!strcmp(name, "GetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR")) return (void *)table->GetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR;

    // ---- VK_KHR_get_surface_capabilities2 extension commands
    if (!strcmp(name, "GetPhysicalDeviceSurfaceCapabilities2KHR")) return (void *)table->GetPhysicalDeviceSurfaceCapabilities2KHR;
    if (!strcmp(name, "GetPhysicalDeviceSurfaceFormats2KHR")) return (void *)table->GetPhysicalDeviceSurfaceFormats2KHR;

    // ---- VK_KHR_get_display_properties2 extension commands
    if (!strcmp(name, "GetPhysicalDeviceDisplayProperties2KHR")) return (void *)table->GetPhysicalDeviceDisplayProperties2KHR;
    if (!strcmp(name, "GetPhysicalDeviceDisplayPlaneProperties2KHR")) return (void *)table->GetPhysicalDeviceDisplayPlaneProperties2KHR;
    if (!strcmp(name, "GetDisplayModeProperties2KHR")) return (void *)table->GetDisplayModeProperties2KHR;
    if (!strcmp(name, "GetDisplayPlaneCapabilities2KHR")) return (void *)table->GetDisplayPlaneCapabilities2KHR;

    // ---- VK_EXT_debug_report extension commands
    if (!strcmp(name, "CreateDebugReportCallbackEXT")) return (void *)table->CreateDebugReportCallbackEXT;
    if (!strcmp(name, "DestroyDebugReportCallbackEXT")) return (void *)table->DestroyDebugReportCallbackEXT;
    if (!strcmp(name, "DebugReportMessageEXT")) return (void *)table->DebugReportMessageEXT;

    // ---- VK_GGP_stream_descriptor_surface extension commands
#ifdef VK_USE_PLATFORM_GGP
    if (!strcmp(name, "CreateStreamDescriptorSurfaceGGP")) return (void *)table->CreateStreamDescriptorSurfaceGGP;
#endif // VK_USE_PLATFORM_GGP

    // ---- VK_NV_external_memory_capabilities extension commands
    if (!strcmp(name, "GetPhysicalDeviceExternalImageFormatPropertiesNV")) return (void *)table->GetPhysicalDeviceExternalImageFormatPropertiesNV;

    // ---- VK_NN_vi_surface extension commands
#ifdef VK_USE_PLATFORM_VI_NN
    if (!strcmp(name, "CreateViSurfaceNN")) return (void *)table->CreateViSurfaceNN;
#endif // VK_USE_PLATFORM_VI_NN

    // ---- VK_EXT_direct_mode_display extension commands
    if (!strcmp(name, "ReleaseDisplayEXT")) return (void *)table->ReleaseDisplayEXT;

    // ---- VK_EXT_acquire_xlib_display extension commands
#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT
    if (!strcmp(name, "AcquireXlibDisplayEXT")) return (void *)table->AcquireXlibDisplayEXT;
#endif // VK_USE_PLATFORM_XLIB_XRANDR_EXT
#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT
    if (!strcmp(name, "GetRandROutputDisplayEXT")) return (void *)table->GetRandROutputDisplayEXT;
#endif // VK_USE_PLATFORM_XLIB_XRANDR_EXT

    // ---- VK_EXT_display_surface_counter extension commands
    if (!strcmp(name, "GetPhysicalDeviceSurfaceCapabilities2EXT")) return (void *)table->GetPhysicalDeviceSurfaceCapabilities2EXT;

    // ---- VK_MVK_ios_surface extension commands
#ifdef VK_USE_PLATFORM_IOS_MVK
    if (!strcmp(name, "CreateIOSSurfaceMVK")) return (void *)table->CreateIOSSurfaceMVK;
#endif // VK_USE_PLATFORM_IOS_MVK

    // ---- VK_MVK_macos_surface extension commands
#ifdef VK_USE_PLATFORM_MACOS_MVK
    if (!strcmp(name, "CreateMacOSSurfaceMVK")) return (void *)table->CreateMacOSSurfaceMVK;
#endif // VK_USE_PLATFORM_MACOS_MVK

    // ---- VK_EXT_debug_utils extension commands
    if (!strcmp(name, "CreateDebugUtilsMessengerEXT")) return (void *)table->CreateDebugUtilsMessengerEXT;
    if (!strcmp(name, "DestroyDebugUtilsMessengerEXT")) return (void *)table->DestroyDebugUtilsMessengerEXT;
    if (!strcmp(name, "SubmitDebugUtilsMessageEXT")) return (void *)table->SubmitDebugUtilsMessageEXT;

    // ---- VK_EXT_sample_locations extension commands
    if (!strcmp(name, "GetPhysicalDeviceMultisamplePropertiesEXT")) return (void *)table->GetPhysicalDeviceMultisamplePropertiesEXT;

    // ---- VK_EXT_calibrated_timestamps extension commands
    if (!strcmp(name, "GetPhysicalDeviceCalibrateableTimeDomainsEXT")) return (void *)table->GetPhysicalDeviceCalibrateableTimeDomainsEXT;

    // ---- VK_FUCHSIA_imagepipe_surface extension commands
#ifdef VK_USE_PLATFORM_FUCHSIA
    if (!strcmp(name, "CreateImagePipeSurfaceFUCHSIA")) return (void *)table->CreateImagePipeSurfaceFUCHSIA;
#endif // VK_USE_PLATFORM_FUCHSIA

    // ---- VK_EXT_metal_surface extension commands
#ifdef VK_USE_PLATFORM_METAL_EXT
    if (!strcmp(name, "CreateMetalSurfaceEXT")) return (void *)table->CreateMetalSurfaceEXT;
#endif // VK_USE_PLATFORM_METAL_EXT

    // ---- VK_EXT_tooling_info extension commands
    if (!strcmp(name, "GetPhysicalDeviceToolPropertiesEXT")) return (void *)table->GetPhysicalDeviceToolPropertiesEXT;

    // ---- VK_NV_cooperative_matrix extension commands
    if (!strcmp(name, "GetPhysicalDeviceCooperativeMatrixPropertiesNV")) return (void *)table->GetPhysicalDeviceCooperativeMatrixPropertiesNV;

    // ---- VK_NV_coverage_reduction_mode extension commands
    if (!strcmp(name, "GetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV")) return (void *)table->GetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV;

    // ---- VK_EXT_full_screen_exclusive extension commands
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (!strcmp(name, "GetPhysicalDeviceSurfacePresentModes2EXT")) return (void *)table->GetPhysicalDeviceSurfacePresentModes2EXT;
#endif // VK_USE_PLATFORM_WIN32_KHR

    // ---- VK_EXT_headless_surface extension commands
    if (!strcmp(name, "CreateHeadlessSurfaceEXT")) return (void *)table->CreateHeadlessSurfaceEXT;

    // ---- VK_EXT_directfb_surface extension commands
#ifdef VK_USE_PLATFORM_DIRECTFB_EXT
    if (!strcmp(name, "CreateDirectFBSurfaceEXT")) return (void *)table->CreateDirectFBSurfaceEXT;
#endif // VK_USE_PLATFORM_DIRECTFB_EXT
#ifdef VK_USE_PLATFORM_DIRECTFB_EXT
    if (!strcmp(name, "GetPhysicalDeviceDirectFBPresentationSupportEXT")) return (void *)table->GetPhysicalDeviceDirectFBPresentationSupportEXT;
#endif // VK_USE_PLATFORM_DIRECTFB_EXT

    *found_name = false;
    return NULL;
}


// ---- VK_KHR_device_group extension trampoline/terminators

VKAPI_ATTR void VKAPI_CALL GetDeviceGroupPeerMemoryFeaturesKHR(
    VkDevice                                    device,
    uint32_t                                    heapIndex,
    uint32_t                                    localDeviceIndex,
    uint32_t                                    remoteDeviceIndex,
    VkPeerMemoryFeatureFlags*                   pPeerMemoryFeatures) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->GetDeviceGroupPeerMemoryFeaturesKHR(device, heapIndex, localDeviceIndex, remoteDeviceIndex, pPeerMemoryFeatures);
}

VKAPI_ATTR void VKAPI_CALL CmdSetDeviceMaskKHR(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    deviceMask) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdSetDeviceMaskKHR(commandBuffer, deviceMask);
}

VKAPI_ATTR void VKAPI_CALL CmdDispatchBaseKHR(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    baseGroupX,
    uint32_t                                    baseGroupY,
    uint32_t                                    baseGroupZ,
    uint32_t                                    groupCountX,
    uint32_t                                    groupCountY,
    uint32_t                                    groupCountZ) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdDispatchBaseKHR(commandBuffer, baseGroupX, baseGroupY, baseGroupZ, groupCountX, groupCountY, groupCountZ);
}


// ---- VK_KHR_maintenance1 extension trampoline/terminators

VKAPI_ATTR void VKAPI_CALL TrimCommandPoolKHR(
    VkDevice                                    device,
    VkCommandPool                               commandPool,
    VkCommandPoolTrimFlags                      flags) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->TrimCommandPoolKHR(device, commandPool, flags);
}


// ---- VK_KHR_external_memory_win32 extension trampoline/terminators

#ifdef VK_USE_PLATFORM_WIN32_KHR
VKAPI_ATTR VkResult VKAPI_CALL GetMemoryWin32HandleKHR(
    VkDevice                                    device,
    const VkMemoryGetWin32HandleInfoKHR*        pGetWin32HandleInfo,
    HANDLE*                                     pHandle) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetMemoryWin32HandleKHR(device, pGetWin32HandleInfo, pHandle);
}

#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
VKAPI_ATTR VkResult VKAPI_CALL GetMemoryWin32HandlePropertiesKHR(
    VkDevice                                    device,
    VkExternalMemoryHandleTypeFlagBits          handleType,
    HANDLE                                      handle,
    VkMemoryWin32HandlePropertiesKHR*           pMemoryWin32HandleProperties) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetMemoryWin32HandlePropertiesKHR(device, handleType, handle, pMemoryWin32HandleProperties);
}

#endif // VK_USE_PLATFORM_WIN32_KHR

// ---- VK_KHR_external_memory_fd extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL GetMemoryFdKHR(
    VkDevice                                    device,
    const VkMemoryGetFdInfoKHR*                 pGetFdInfo,
    int*                                        pFd) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetMemoryFdKHR(device, pGetFdInfo, pFd);
}

VKAPI_ATTR VkResult VKAPI_CALL GetMemoryFdPropertiesKHR(
    VkDevice                                    device,
    VkExternalMemoryHandleTypeFlagBits          handleType,
    int                                         fd,
    VkMemoryFdPropertiesKHR*                    pMemoryFdProperties) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetMemoryFdPropertiesKHR(device, handleType, fd, pMemoryFdProperties);
}


// ---- VK_KHR_external_semaphore_win32 extension trampoline/terminators

#ifdef VK_USE_PLATFORM_WIN32_KHR
VKAPI_ATTR VkResult VKAPI_CALL ImportSemaphoreWin32HandleKHR(
    VkDevice                                    device,
    const VkImportSemaphoreWin32HandleInfoKHR*  pImportSemaphoreWin32HandleInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->ImportSemaphoreWin32HandleKHR(device, pImportSemaphoreWin32HandleInfo);
}

#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
VKAPI_ATTR VkResult VKAPI_CALL GetSemaphoreWin32HandleKHR(
    VkDevice                                    device,
    const VkSemaphoreGetWin32HandleInfoKHR*     pGetWin32HandleInfo,
    HANDLE*                                     pHandle) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetSemaphoreWin32HandleKHR(device, pGetWin32HandleInfo, pHandle);
}

#endif // VK_USE_PLATFORM_WIN32_KHR

// ---- VK_KHR_external_semaphore_fd extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL ImportSemaphoreFdKHR(
    VkDevice                                    device,
    const VkImportSemaphoreFdInfoKHR*           pImportSemaphoreFdInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->ImportSemaphoreFdKHR(device, pImportSemaphoreFdInfo);
}

VKAPI_ATTR VkResult VKAPI_CALL GetSemaphoreFdKHR(
    VkDevice                                    device,
    const VkSemaphoreGetFdInfoKHR*              pGetFdInfo,
    int*                                        pFd) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetSemaphoreFdKHR(device, pGetFdInfo, pFd);
}


// ---- VK_KHR_push_descriptor extension trampoline/terminators

VKAPI_ATTR void VKAPI_CALL CmdPushDescriptorSetKHR(
    VkCommandBuffer                             commandBuffer,
    VkPipelineBindPoint                         pipelineBindPoint,
    VkPipelineLayout                            layout,
    uint32_t                                    set,
    uint32_t                                    descriptorWriteCount,
    const VkWriteDescriptorSet*                 pDescriptorWrites) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdPushDescriptorSetKHR(commandBuffer, pipelineBindPoint, layout, set, descriptorWriteCount, pDescriptorWrites);
}

VKAPI_ATTR void VKAPI_CALL CmdPushDescriptorSetWithTemplateKHR(
    VkCommandBuffer                             commandBuffer,
    VkDescriptorUpdateTemplate                  descriptorUpdateTemplate,
    VkPipelineLayout                            layout,
    uint32_t                                    set,
    const void*                                 pData) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdPushDescriptorSetWithTemplateKHR(commandBuffer, descriptorUpdateTemplate, layout, set, pData);
}


// ---- VK_KHR_descriptor_update_template extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorUpdateTemplateKHR(
    VkDevice                                    device,
    const VkDescriptorUpdateTemplateCreateInfo* pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkDescriptorUpdateTemplate*                 pDescriptorUpdateTemplate) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->CreateDescriptorUpdateTemplateKHR(device, pCreateInfo, pAllocator, pDescriptorUpdateTemplate);
}

VKAPI_ATTR void VKAPI_CALL DestroyDescriptorUpdateTemplateKHR(
    VkDevice                                    device,
    VkDescriptorUpdateTemplate                  descriptorUpdateTemplate,
    const VkAllocationCallbacks*                pAllocator) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->DestroyDescriptorUpdateTemplateKHR(device, descriptorUpdateTemplate, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL UpdateDescriptorSetWithTemplateKHR(
    VkDevice                                    device,
    VkDescriptorSet                             descriptorSet,
    VkDescriptorUpdateTemplate                  descriptorUpdateTemplate,
    const void*                                 pData) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->UpdateDescriptorSetWithTemplateKHR(device, descriptorSet, descriptorUpdateTemplate, pData);
}


// ---- VK_KHR_create_renderpass2 extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL CreateRenderPass2KHR(
    VkDevice                                    device,
    const VkRenderPassCreateInfo2*              pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkRenderPass*                               pRenderPass) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->CreateRenderPass2KHR(device, pCreateInfo, pAllocator, pRenderPass);
}

VKAPI_ATTR void VKAPI_CALL CmdBeginRenderPass2KHR(
    VkCommandBuffer                             commandBuffer,
    const VkRenderPassBeginInfo*                pRenderPassBegin,
    const VkSubpassBeginInfo*                   pSubpassBeginInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdBeginRenderPass2KHR(commandBuffer, pRenderPassBegin, pSubpassBeginInfo);
}

VKAPI_ATTR void VKAPI_CALL CmdNextSubpass2KHR(
    VkCommandBuffer                             commandBuffer,
    const VkSubpassBeginInfo*                   pSubpassBeginInfo,
    const VkSubpassEndInfo*                     pSubpassEndInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdNextSubpass2KHR(commandBuffer, pSubpassBeginInfo, pSubpassEndInfo);
}

VKAPI_ATTR void VKAPI_CALL CmdEndRenderPass2KHR(
    VkCommandBuffer                             commandBuffer,
    const VkSubpassEndInfo*                     pSubpassEndInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdEndRenderPass2KHR(commandBuffer, pSubpassEndInfo);
}


// ---- VK_KHR_shared_presentable_image extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL GetSwapchainStatusKHR(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetSwapchainStatusKHR(device, swapchain);
}


// ---- VK_KHR_external_fence_win32 extension trampoline/terminators

#ifdef VK_USE_PLATFORM_WIN32_KHR
VKAPI_ATTR VkResult VKAPI_CALL ImportFenceWin32HandleKHR(
    VkDevice                                    device,
    const VkImportFenceWin32HandleInfoKHR*      pImportFenceWin32HandleInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->ImportFenceWin32HandleKHR(device, pImportFenceWin32HandleInfo);
}

#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
VKAPI_ATTR VkResult VKAPI_CALL GetFenceWin32HandleKHR(
    VkDevice                                    device,
    const VkFenceGetWin32HandleInfoKHR*         pGetWin32HandleInfo,
    HANDLE*                                     pHandle) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetFenceWin32HandleKHR(device, pGetWin32HandleInfo, pHandle);
}

#endif // VK_USE_PLATFORM_WIN32_KHR

// ---- VK_KHR_external_fence_fd extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL ImportFenceFdKHR(
    VkDevice                                    device,
    const VkImportFenceFdInfoKHR*               pImportFenceFdInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->ImportFenceFdKHR(device, pImportFenceFdInfo);
}

VKAPI_ATTR VkResult VKAPI_CALL GetFenceFdKHR(
    VkDevice                                    device,
    const VkFenceGetFdInfoKHR*                  pGetFdInfo,
    int*                                        pFd) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetFenceFdKHR(device, pGetFdInfo, pFd);
}


// ---- VK_KHR_performance_query extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL EnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR(
    VkPhysicalDevice                            physicalDevice,
    uint32_t                                    queueFamilyIndex,
    uint32_t*                                   pCounterCount,
    VkPerformanceCounterKHR*                    pCounters,
    VkPerformanceCounterDescriptionKHR*         pCounterDescriptions) {
    const VkLayerInstanceDispatchTable *disp;
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    disp = loader_get_instance_layer_dispatch(physicalDevice);
    return disp->EnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR(unwrapped_phys_dev, queueFamilyIndex, pCounterCount, pCounters, pCounterDescriptions);
}

VKAPI_ATTR VkResult VKAPI_CALL terminator_EnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR(
    VkPhysicalDevice                            physicalDevice,
    uint32_t                                    queueFamilyIndex,
    uint32_t*                                   pCounterCount,
    VkPerformanceCounterKHR*                    pCounters,
    VkPerformanceCounterDescriptionKHR*         pCounterDescriptions) {
    struct loader_physical_device_term *phys_dev_term = (struct loader_physical_device_term *)physicalDevice;
    struct loader_icd_term *icd_term = phys_dev_term->this_icd_term;
    if (NULL == icd_term->dispatch.EnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR) {
        loader_log(icd_term->this_instance, VK_DEBUG_REPORT_ERROR_BIT_EXT, 0,
                   "ICD associated with VkPhysicalDevice does not support EnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR");
    }
    return icd_term->dispatch.EnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR(phys_dev_term->phys_dev, queueFamilyIndex, pCounterCount, pCounters, pCounterDescriptions);
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR(
    VkPhysicalDevice                            physicalDevice,
    const VkQueryPoolPerformanceCreateInfoKHR*  pPerformanceQueryCreateInfo,
    uint32_t*                                   pNumPasses) {
    const VkLayerInstanceDispatchTable *disp;
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    disp = loader_get_instance_layer_dispatch(physicalDevice);
    disp->GetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR(unwrapped_phys_dev, pPerformanceQueryCreateInfo, pNumPasses);
}

VKAPI_ATTR void VKAPI_CALL terminator_GetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR(
    VkPhysicalDevice                            physicalDevice,
    const VkQueryPoolPerformanceCreateInfoKHR*  pPerformanceQueryCreateInfo,
    uint32_t*                                   pNumPasses) {
    struct loader_physical_device_term *phys_dev_term = (struct loader_physical_device_term *)physicalDevice;
    struct loader_icd_term *icd_term = phys_dev_term->this_icd_term;
    if (NULL == icd_term->dispatch.GetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR) {
        loader_log(icd_term->this_instance, VK_DEBUG_REPORT_ERROR_BIT_EXT, 0,
                   "ICD associated with VkPhysicalDevice does not support GetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR");
    }
    icd_term->dispatch.GetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR(phys_dev_term->phys_dev, pPerformanceQueryCreateInfo, pNumPasses);
}

VKAPI_ATTR VkResult VKAPI_CALL AcquireProfilingLockKHR(
    VkDevice                                    device,
    const VkAcquireProfilingLockInfoKHR*        pInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->AcquireProfilingLockKHR(device, pInfo);
}

VKAPI_ATTR void VKAPI_CALL ReleaseProfilingLockKHR(
    VkDevice                                    device) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->ReleaseProfilingLockKHR(device);
}


// ---- VK_KHR_get_memory_requirements2 extension trampoline/terminators

VKAPI_ATTR void VKAPI_CALL GetImageMemoryRequirements2KHR(
    VkDevice                                    device,
    const VkImageMemoryRequirementsInfo2*       pInfo,
    VkMemoryRequirements2*                      pMemoryRequirements) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->GetImageMemoryRequirements2KHR(device, pInfo, pMemoryRequirements);
}

VKAPI_ATTR void VKAPI_CALL GetBufferMemoryRequirements2KHR(
    VkDevice                                    device,
    const VkBufferMemoryRequirementsInfo2*      pInfo,
    VkMemoryRequirements2*                      pMemoryRequirements) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->GetBufferMemoryRequirements2KHR(device, pInfo, pMemoryRequirements);
}

VKAPI_ATTR void VKAPI_CALL GetImageSparseMemoryRequirements2KHR(
    VkDevice                                    device,
    const VkImageSparseMemoryRequirementsInfo2* pInfo,
    uint32_t*                                   pSparseMemoryRequirementCount,
    VkSparseImageMemoryRequirements2*           pSparseMemoryRequirements) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->GetImageSparseMemoryRequirements2KHR(device, pInfo, pSparseMemoryRequirementCount, pSparseMemoryRequirements);
}


// ---- VK_KHR_sampler_ycbcr_conversion extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL CreateSamplerYcbcrConversionKHR(
    VkDevice                                    device,
    const VkSamplerYcbcrConversionCreateInfo*   pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSamplerYcbcrConversion*                   pYcbcrConversion) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->CreateSamplerYcbcrConversionKHR(device, pCreateInfo, pAllocator, pYcbcrConversion);
}

VKAPI_ATTR void VKAPI_CALL DestroySamplerYcbcrConversionKHR(
    VkDevice                                    device,
    VkSamplerYcbcrConversion                    ycbcrConversion,
    const VkAllocationCallbacks*                pAllocator) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->DestroySamplerYcbcrConversionKHR(device, ycbcrConversion, pAllocator);
}


// ---- VK_KHR_bind_memory2 extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL BindBufferMemory2KHR(
    VkDevice                                    device,
    uint32_t                                    bindInfoCount,
    const VkBindBufferMemoryInfo*               pBindInfos) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->BindBufferMemory2KHR(device, bindInfoCount, pBindInfos);
}

VKAPI_ATTR VkResult VKAPI_CALL BindImageMemory2KHR(
    VkDevice                                    device,
    uint32_t                                    bindInfoCount,
    const VkBindImageMemoryInfo*                pBindInfos) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->BindImageMemory2KHR(device, bindInfoCount, pBindInfos);
}


// ---- VK_KHR_maintenance3 extension trampoline/terminators

VKAPI_ATTR void VKAPI_CALL GetDescriptorSetLayoutSupportKHR(
    VkDevice                                    device,
    const VkDescriptorSetLayoutCreateInfo*      pCreateInfo,
    VkDescriptorSetLayoutSupport*               pSupport) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->GetDescriptorSetLayoutSupportKHR(device, pCreateInfo, pSupport);
}


// ---- VK_KHR_draw_indirect_count extension trampoline/terminators

VKAPI_ATTR void VKAPI_CALL CmdDrawIndirectCountKHR(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    buffer,
    VkDeviceSize                                offset,
    VkBuffer                                    countBuffer,
    VkDeviceSize                                countBufferOffset,
    uint32_t                                    maxDrawCount,
    uint32_t                                    stride) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdDrawIndirectCountKHR(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndexedIndirectCountKHR(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    buffer,
    VkDeviceSize                                offset,
    VkBuffer                                    countBuffer,
    VkDeviceSize                                countBufferOffset,
    uint32_t                                    maxDrawCount,
    uint32_t                                    stride) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdDrawIndexedIndirectCountKHR(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
}


// ---- VK_KHR_timeline_semaphore extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL GetSemaphoreCounterValueKHR(
    VkDevice                                    device,
    VkSemaphore                                 semaphore,
    uint64_t*                                   pValue) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetSemaphoreCounterValueKHR(device, semaphore, pValue);
}

VKAPI_ATTR VkResult VKAPI_CALL WaitSemaphoresKHR(
    VkDevice                                    device,
    const VkSemaphoreWaitInfo*                  pWaitInfo,
    uint64_t                                    timeout) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->WaitSemaphoresKHR(device, pWaitInfo, timeout);
}

VKAPI_ATTR VkResult VKAPI_CALL SignalSemaphoreKHR(
    VkDevice                                    device,
    const VkSemaphoreSignalInfo*                pSignalInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->SignalSemaphoreKHR(device, pSignalInfo);
}


// ---- VK_KHR_buffer_device_address extension trampoline/terminators

VKAPI_ATTR VkDeviceAddress VKAPI_CALL GetBufferDeviceAddressKHR(
    VkDevice                                    device,
    const VkBufferDeviceAddressInfo*            pInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetBufferDeviceAddressKHR(device, pInfo);
}

VKAPI_ATTR uint64_t VKAPI_CALL GetBufferOpaqueCaptureAddressKHR(
    VkDevice                                    device,
    const VkBufferDeviceAddressInfo*            pInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetBufferOpaqueCaptureAddressKHR(device, pInfo);
}

VKAPI_ATTR uint64_t VKAPI_CALL GetDeviceMemoryOpaqueCaptureAddressKHR(
    VkDevice                                    device,
    const VkDeviceMemoryOpaqueCaptureAddressInfo* pInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetDeviceMemoryOpaqueCaptureAddressKHR(device, pInfo);
}


// ---- VK_KHR_deferred_host_operations extension trampoline/terminators

#ifdef VK_ENABLE_BETA_EXTENSIONS
VKAPI_ATTR VkResult VKAPI_CALL CreateDeferredOperationKHR(
    VkDevice                                    device,
    const VkAllocationCallbacks*                pAllocator,
    VkDeferredOperationKHR*                     pDeferredOperation) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->CreateDeferredOperationKHR(device, pAllocator, pDeferredOperation);
}

#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
VKAPI_ATTR void VKAPI_CALL DestroyDeferredOperationKHR(
    VkDevice                                    device,
    VkDeferredOperationKHR                      operation,
    const VkAllocationCallbacks*                pAllocator) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->DestroyDeferredOperationKHR(device, operation, pAllocator);
}

#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
VKAPI_ATTR uint32_t VKAPI_CALL GetDeferredOperationMaxConcurrencyKHR(
    VkDevice                                    device,
    VkDeferredOperationKHR                      operation) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetDeferredOperationMaxConcurrencyKHR(device, operation);
}

#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
VKAPI_ATTR VkResult VKAPI_CALL GetDeferredOperationResultKHR(
    VkDevice                                    device,
    VkDeferredOperationKHR                      operation) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetDeferredOperationResultKHR(device, operation);
}

#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
VKAPI_ATTR VkResult VKAPI_CALL DeferredOperationJoinKHR(
    VkDevice                                    device,
    VkDeferredOperationKHR                      operation) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->DeferredOperationJoinKHR(device, operation);
}

#endif // VK_ENABLE_BETA_EXTENSIONS

// ---- VK_KHR_pipeline_executable_properties extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL GetPipelineExecutablePropertiesKHR(
    VkDevice                                    device,
    const VkPipelineInfoKHR*                    pPipelineInfo,
    uint32_t*                                   pExecutableCount,
    VkPipelineExecutablePropertiesKHR*          pProperties) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetPipelineExecutablePropertiesKHR(device, pPipelineInfo, pExecutableCount, pProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL GetPipelineExecutableStatisticsKHR(
    VkDevice                                    device,
    const VkPipelineExecutableInfoKHR*          pExecutableInfo,
    uint32_t*                                   pStatisticCount,
    VkPipelineExecutableStatisticKHR*           pStatistics) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetPipelineExecutableStatisticsKHR(device, pExecutableInfo, pStatisticCount, pStatistics);
}

VKAPI_ATTR VkResult VKAPI_CALL GetPipelineExecutableInternalRepresentationsKHR(
    VkDevice                                    device,
    const VkPipelineExecutableInfoKHR*          pExecutableInfo,
    uint32_t*                                   pInternalRepresentationCount,
    VkPipelineExecutableInternalRepresentationKHR* pInternalRepresentations) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetPipelineExecutableInternalRepresentationsKHR(device, pExecutableInfo, pInternalRepresentationCount, pInternalRepresentations);
}


// ---- VK_KHR_copy_commands2 extension trampoline/terminators

VKAPI_ATTR void VKAPI_CALL CmdCopyBuffer2KHR(
    VkCommandBuffer                             commandBuffer,
    const VkCopyBufferInfo2KHR*                 pCopyBufferInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdCopyBuffer2KHR(commandBuffer, pCopyBufferInfo);
}

VKAPI_ATTR void VKAPI_CALL CmdCopyImage2KHR(
    VkCommandBuffer                             commandBuffer,
    const VkCopyImageInfo2KHR*                  pCopyImageInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdCopyImage2KHR(commandBuffer, pCopyImageInfo);
}

VKAPI_ATTR void VKAPI_CALL CmdCopyBufferToImage2KHR(
    VkCommandBuffer                             commandBuffer,
    const VkCopyBufferToImageInfo2KHR*          pCopyBufferToImageInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdCopyBufferToImage2KHR(commandBuffer, pCopyBufferToImageInfo);
}

VKAPI_ATTR void VKAPI_CALL CmdCopyImageToBuffer2KHR(
    VkCommandBuffer                             commandBuffer,
    const VkCopyImageToBufferInfo2KHR*          pCopyImageToBufferInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdCopyImageToBuffer2KHR(commandBuffer, pCopyImageToBufferInfo);
}

VKAPI_ATTR void VKAPI_CALL CmdBlitImage2KHR(
    VkCommandBuffer                             commandBuffer,
    const VkBlitImageInfo2KHR*                  pBlitImageInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdBlitImage2KHR(commandBuffer, pBlitImageInfo);
}

VKAPI_ATTR void VKAPI_CALL CmdResolveImage2KHR(
    VkCommandBuffer                             commandBuffer,
    const VkResolveImageInfo2KHR*               pResolveImageInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdResolveImage2KHR(commandBuffer, pResolveImageInfo);
}


// ---- VK_EXT_debug_marker extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL DebugMarkerSetObjectTagEXT(
    VkDevice                                    device,
    const VkDebugMarkerObjectTagInfoEXT*        pTagInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    VkDebugMarkerObjectTagInfoEXT local_tag_info;
    memcpy(&local_tag_info, pTagInfo, sizeof(VkDebugMarkerObjectTagInfoEXT));
    // If this is a physical device, we have to replace it with the proper one for the next call.
    if (pTagInfo->objectType == VK_DEBUG_REPORT_OBJECT_TYPE_PHYSICAL_DEVICE_EXT) {
        struct loader_physical_device_tramp *phys_dev_tramp = (struct loader_physical_device_tramp *)(uintptr_t)pTagInfo->object;
        local_tag_info.object = (uint64_t)(uintptr_t)phys_dev_tramp->phys_dev;
    }
    return disp->DebugMarkerSetObjectTagEXT(device, &local_tag_info);
}

VKAPI_ATTR VkResult VKAPI_CALL terminator_DebugMarkerSetObjectTagEXT(
    VkDevice                                    device,
    const VkDebugMarkerObjectTagInfoEXT*        pTagInfo) {
    uint32_t icd_index = 0;
    struct loader_device *dev;
    struct loader_icd_term *icd_term = loader_get_icd_and_device(device, &dev, &icd_index);
    if (NULL != icd_term && NULL != icd_term->dispatch.DebugMarkerSetObjectTagEXT) {
        VkDebugMarkerObjectTagInfoEXT local_tag_info;
        memcpy(&local_tag_info, pTagInfo, sizeof(VkDebugMarkerObjectTagInfoEXT));
        // If this is a physical device, we have to replace it with the proper one for the next call.
        if (pTagInfo->objectType == VK_DEBUG_REPORT_OBJECT_TYPE_PHYSICAL_DEVICE_EXT) {
            struct loader_physical_device_term *phys_dev_term = (struct loader_physical_device_term *)(uintptr_t)pTagInfo->object;
            local_tag_info.object = (uint64_t)(uintptr_t)phys_dev_term->phys_dev;
        // If this is a KHR_surface, and the ICD has created its own, we have to replace it with the proper one for the next call.
        } else if (pTagInfo->objectType == VK_DEBUG_REPORT_OBJECT_TYPE_SURFACE_KHR_EXT) {
            if (NULL != icd_term && NULL != icd_term->dispatch.CreateSwapchainKHR) {
                VkIcdSurface *icd_surface = (VkIcdSurface *)(uintptr_t)pTagInfo->object;
                if (NULL != icd_surface->real_icd_surfaces) {
                    local_tag_info.object = (uint64_t)icd_surface->real_icd_surfaces[icd_index];
                }
            }
        }
        return icd_term->dispatch.DebugMarkerSetObjectTagEXT(device, &local_tag_info);
    } else {
        return VK_SUCCESS;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL DebugMarkerSetObjectNameEXT(
    VkDevice                                    device,
    const VkDebugMarkerObjectNameInfoEXT*       pNameInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    VkDebugMarkerObjectNameInfoEXT local_name_info;
    memcpy(&local_name_info, pNameInfo, sizeof(VkDebugMarkerObjectNameInfoEXT));
    // If this is a physical device, we have to replace it with the proper one for the next call.
    if (pNameInfo->objectType == VK_DEBUG_REPORT_OBJECT_TYPE_PHYSICAL_DEVICE_EXT) {
        struct loader_physical_device_tramp *phys_dev_tramp = (struct loader_physical_device_tramp *)(uintptr_t)pNameInfo->object;
        local_name_info.object = (uint64_t)(uintptr_t)phys_dev_tramp->phys_dev;
    }
    return disp->DebugMarkerSetObjectNameEXT(device, &local_name_info);
}

VKAPI_ATTR VkResult VKAPI_CALL terminator_DebugMarkerSetObjectNameEXT(
    VkDevice                                    device,
    const VkDebugMarkerObjectNameInfoEXT*       pNameInfo) {
    uint32_t icd_index = 0;
    struct loader_device *dev;
    struct loader_icd_term *icd_term = loader_get_icd_and_device(device, &dev, &icd_index);
    if (NULL != icd_term && NULL != icd_term->dispatch.DebugMarkerSetObjectNameEXT) {
        VkDebugMarkerObjectNameInfoEXT local_name_info;
        memcpy(&local_name_info, pNameInfo, sizeof(VkDebugMarkerObjectNameInfoEXT));
        // If this is a physical device, we have to replace it with the proper one for the next call.
        if (pNameInfo->objectType == VK_DEBUG_REPORT_OBJECT_TYPE_PHYSICAL_DEVICE_EXT) {
            struct loader_physical_device_term *phys_dev_term = (struct loader_physical_device_term *)(uintptr_t)pNameInfo->object;
            local_name_info.object = (uint64_t)(uintptr_t)phys_dev_term->phys_dev;
        // If this is a KHR_surface, and the ICD has created its own, we have to replace it with the proper one for the next call.
        } else if (pNameInfo->objectType == VK_DEBUG_REPORT_OBJECT_TYPE_SURFACE_KHR_EXT) {
            if (NULL != icd_term && NULL != icd_term->dispatch.CreateSwapchainKHR) {
                VkIcdSurface *icd_surface = (VkIcdSurface *)(uintptr_t)pNameInfo->object;
                if (NULL != icd_surface->real_icd_surfaces) {
                    local_name_info.object = (uint64_t)icd_surface->real_icd_surfaces[icd_index];
                }
            }
        }
        return icd_term->dispatch.DebugMarkerSetObjectNameEXT(device, &local_name_info);
    } else {
        return VK_SUCCESS;
    }
}

VKAPI_ATTR void VKAPI_CALL CmdDebugMarkerBeginEXT(
    VkCommandBuffer                             commandBuffer,
    const VkDebugMarkerMarkerInfoEXT*           pMarkerInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdDebugMarkerBeginEXT(commandBuffer, pMarkerInfo);
}

VKAPI_ATTR void VKAPI_CALL CmdDebugMarkerEndEXT(
    VkCommandBuffer                             commandBuffer) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdDebugMarkerEndEXT(commandBuffer);
}

VKAPI_ATTR void VKAPI_CALL CmdDebugMarkerInsertEXT(
    VkCommandBuffer                             commandBuffer,
    const VkDebugMarkerMarkerInfoEXT*           pMarkerInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdDebugMarkerInsertEXT(commandBuffer, pMarkerInfo);
}


// ---- VK_EXT_transform_feedback extension trampoline/terminators

VKAPI_ATTR void VKAPI_CALL CmdBindTransformFeedbackBuffersEXT(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    firstBinding,
    uint32_t                                    bindingCount,
    const VkBuffer*                             pBuffers,
    const VkDeviceSize*                         pOffsets,
    const VkDeviceSize*                         pSizes) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdBindTransformFeedbackBuffersEXT(commandBuffer, firstBinding, bindingCount, pBuffers, pOffsets, pSizes);
}

VKAPI_ATTR void VKAPI_CALL CmdBeginTransformFeedbackEXT(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    firstCounterBuffer,
    uint32_t                                    counterBufferCount,
    const VkBuffer*                             pCounterBuffers,
    const VkDeviceSize*                         pCounterBufferOffsets) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdBeginTransformFeedbackEXT(commandBuffer, firstCounterBuffer, counterBufferCount, pCounterBuffers, pCounterBufferOffsets);
}

VKAPI_ATTR void VKAPI_CALL CmdEndTransformFeedbackEXT(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    firstCounterBuffer,
    uint32_t                                    counterBufferCount,
    const VkBuffer*                             pCounterBuffers,
    const VkDeviceSize*                         pCounterBufferOffsets) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdEndTransformFeedbackEXT(commandBuffer, firstCounterBuffer, counterBufferCount, pCounterBuffers, pCounterBufferOffsets);
}

VKAPI_ATTR void VKAPI_CALL CmdBeginQueryIndexedEXT(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    query,
    VkQueryControlFlags                         flags,
    uint32_t                                    index) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdBeginQueryIndexedEXT(commandBuffer, queryPool, query, flags, index);
}

VKAPI_ATTR void VKAPI_CALL CmdEndQueryIndexedEXT(
    VkCommandBuffer                             commandBuffer,
    VkQueryPool                                 queryPool,
    uint32_t                                    query,
    uint32_t                                    index) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdEndQueryIndexedEXT(commandBuffer, queryPool, query, index);
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndirectByteCountEXT(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    instanceCount,
    uint32_t                                    firstInstance,
    VkBuffer                                    counterBuffer,
    VkDeviceSize                                counterBufferOffset,
    uint32_t                                    counterOffset,
    uint32_t                                    vertexStride) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdDrawIndirectByteCountEXT(commandBuffer, instanceCount, firstInstance, counterBuffer, counterBufferOffset, counterOffset, vertexStride);
}


// ---- VK_NVX_image_view_handle extension trampoline/terminators

VKAPI_ATTR uint32_t VKAPI_CALL GetImageViewHandleNVX(
    VkDevice                                    device,
    const VkImageViewHandleInfoNVX*             pInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetImageViewHandleNVX(device, pInfo);
}

VKAPI_ATTR VkResult VKAPI_CALL GetImageViewAddressNVX(
    VkDevice                                    device,
    VkImageView                                 imageView,
    VkImageViewAddressPropertiesNVX*            pProperties) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetImageViewAddressNVX(device, imageView, pProperties);
}


// ---- VK_AMD_draw_indirect_count extension trampoline/terminators

VKAPI_ATTR void VKAPI_CALL CmdDrawIndirectCountAMD(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    buffer,
    VkDeviceSize                                offset,
    VkBuffer                                    countBuffer,
    VkDeviceSize                                countBufferOffset,
    uint32_t                                    maxDrawCount,
    uint32_t                                    stride) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdDrawIndirectCountAMD(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndexedIndirectCountAMD(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    buffer,
    VkDeviceSize                                offset,
    VkBuffer                                    countBuffer,
    VkDeviceSize                                countBufferOffset,
    uint32_t                                    maxDrawCount,
    uint32_t                                    stride) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdDrawIndexedIndirectCountAMD(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
}


// ---- VK_AMD_shader_info extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL GetShaderInfoAMD(
    VkDevice                                    device,
    VkPipeline                                  pipeline,
    VkShaderStageFlagBits                       shaderStage,
    VkShaderInfoTypeAMD                         infoType,
    size_t*                                     pInfoSize,
    void*                                       pInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetShaderInfoAMD(device, pipeline, shaderStage, infoType, pInfoSize, pInfo);
}


// ---- VK_GGP_stream_descriptor_surface extension trampoline/terminators

#ifdef VK_USE_PLATFORM_GGP
VKAPI_ATTR VkResult VKAPI_CALL CreateStreamDescriptorSurfaceGGP(
    VkInstance                                  instance,
    const VkStreamDescriptorSurfaceCreateInfoGGP* pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface) {
#error("Not implemented. Likely needs to be manually generated!");
    return disp->CreateStreamDescriptorSurfaceGGP(instance, pCreateInfo, pAllocator, pSurface);
}

VKAPI_ATTR VkResult VKAPI_CALL terminator_CreateStreamDescriptorSurfaceGGP(
    VkInstance                                  instance,
    const VkStreamDescriptorSurfaceCreateInfoGGP* pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface) {
#error("Not implemented. Likely needs to be manually generated!");
}

#endif // VK_USE_PLATFORM_GGP

// ---- VK_NV_external_memory_win32 extension trampoline/terminators

#ifdef VK_USE_PLATFORM_WIN32_KHR
VKAPI_ATTR VkResult VKAPI_CALL GetMemoryWin32HandleNV(
    VkDevice                                    device,
    VkDeviceMemory                              memory,
    VkExternalMemoryHandleTypeFlagsNV           handleType,
    HANDLE*                                     pHandle) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetMemoryWin32HandleNV(device, memory, handleType, pHandle);
}

#endif // VK_USE_PLATFORM_WIN32_KHR

// ---- VK_NN_vi_surface extension trampoline/terminators

#ifdef VK_USE_PLATFORM_VI_NN
VKAPI_ATTR VkResult VKAPI_CALL CreateViSurfaceNN(
    VkInstance                                  instance,
    const VkViSurfaceCreateInfoNN*              pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface) {
#error("Not implemented. Likely needs to be manually generated!");
    return disp->CreateViSurfaceNN(instance, pCreateInfo, pAllocator, pSurface);
}

VKAPI_ATTR VkResult VKAPI_CALL terminator_CreateViSurfaceNN(
    VkInstance                                  instance,
    const VkViSurfaceCreateInfoNN*              pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface) {
#error("Not implemented. Likely needs to be manually generated!");
}

#endif // VK_USE_PLATFORM_VI_NN

// ---- VK_EXT_conditional_rendering extension trampoline/terminators

VKAPI_ATTR void VKAPI_CALL CmdBeginConditionalRenderingEXT(
    VkCommandBuffer                             commandBuffer,
    const VkConditionalRenderingBeginInfoEXT*   pConditionalRenderingBegin) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdBeginConditionalRenderingEXT(commandBuffer, pConditionalRenderingBegin);
}

VKAPI_ATTR void VKAPI_CALL CmdEndConditionalRenderingEXT(
    VkCommandBuffer                             commandBuffer) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdEndConditionalRenderingEXT(commandBuffer);
}


// ---- VK_NV_clip_space_w_scaling extension trampoline/terminators

VKAPI_ATTR void VKAPI_CALL CmdSetViewportWScalingNV(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    firstViewport,
    uint32_t                                    viewportCount,
    const VkViewportWScalingNV*                 pViewportWScalings) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdSetViewportWScalingNV(commandBuffer, firstViewport, viewportCount, pViewportWScalings);
}


// ---- VK_EXT_display_control extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL DisplayPowerControlEXT(
    VkDevice                                    device,
    VkDisplayKHR                                display,
    const VkDisplayPowerInfoEXT*                pDisplayPowerInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->DisplayPowerControlEXT(device, display, pDisplayPowerInfo);
}

VKAPI_ATTR VkResult VKAPI_CALL RegisterDeviceEventEXT(
    VkDevice                                    device,
    const VkDeviceEventInfoEXT*                 pDeviceEventInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkFence*                                    pFence) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->RegisterDeviceEventEXT(device, pDeviceEventInfo, pAllocator, pFence);
}

VKAPI_ATTR VkResult VKAPI_CALL RegisterDisplayEventEXT(
    VkDevice                                    device,
    VkDisplayKHR                                display,
    const VkDisplayEventInfoEXT*                pDisplayEventInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkFence*                                    pFence) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->RegisterDisplayEventEXT(device, display, pDisplayEventInfo, pAllocator, pFence);
}

VKAPI_ATTR VkResult VKAPI_CALL GetSwapchainCounterEXT(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain,
    VkSurfaceCounterFlagBitsEXT                 counter,
    uint64_t*                                   pCounterValue) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetSwapchainCounterEXT(device, swapchain, counter, pCounterValue);
}


// ---- VK_GOOGLE_display_timing extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL GetRefreshCycleDurationGOOGLE(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain,
    VkRefreshCycleDurationGOOGLE*               pDisplayTimingProperties) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetRefreshCycleDurationGOOGLE(device, swapchain, pDisplayTimingProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL GetPastPresentationTimingGOOGLE(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain,
    uint32_t*                                   pPresentationTimingCount,
    VkPastPresentationTimingGOOGLE*             pPresentationTimings) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetPastPresentationTimingGOOGLE(device, swapchain, pPresentationTimingCount, pPresentationTimings);
}


// ---- VK_EXT_discard_rectangles extension trampoline/terminators

VKAPI_ATTR void VKAPI_CALL CmdSetDiscardRectangleEXT(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    firstDiscardRectangle,
    uint32_t                                    discardRectangleCount,
    const VkRect2D*                             pDiscardRectangles) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdSetDiscardRectangleEXT(commandBuffer, firstDiscardRectangle, discardRectangleCount, pDiscardRectangles);
}


// ---- VK_EXT_hdr_metadata extension trampoline/terminators

VKAPI_ATTR void VKAPI_CALL SetHdrMetadataEXT(
    VkDevice                                    device,
    uint32_t                                    swapchainCount,
    const VkSwapchainKHR*                       pSwapchains,
    const VkHdrMetadataEXT*                     pMetadata) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->SetHdrMetadataEXT(device, swapchainCount, pSwapchains, pMetadata);
}


// ---- VK_EXT_debug_utils extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL SetDebugUtilsObjectNameEXT(
    VkDevice                                    device,
    const VkDebugUtilsObjectNameInfoEXT*        pNameInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    VkDebugUtilsObjectNameInfoEXT local_name_info;
    memcpy(&local_name_info, pNameInfo, sizeof(VkDebugUtilsObjectNameInfoEXT));
    // If this is a physical device, we have to replace it with the proper one for the next call.
    if (pNameInfo->objectType == VK_OBJECT_TYPE_PHYSICAL_DEVICE) {
        struct loader_physical_device_tramp *phys_dev_tramp = (struct loader_physical_device_tramp *)(uintptr_t)pNameInfo->objectHandle;
        local_name_info.objectHandle = (uint64_t)(uintptr_t)phys_dev_tramp->phys_dev;
    }
    if (disp->SetDebugUtilsObjectNameEXT != NULL) {
        return disp->SetDebugUtilsObjectNameEXT(device, &local_name_info);
    } else {
        return VK_SUCCESS;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL terminator_SetDebugUtilsObjectNameEXT(
    VkDevice                                    device,
    const VkDebugUtilsObjectNameInfoEXT*        pNameInfo) {
    uint32_t icd_index = 0;
    struct loader_device *dev;
    struct loader_icd_term *icd_term = loader_get_icd_and_device(device, &dev, &icd_index);
    if (NULL != icd_term && NULL != icd_term->dispatch.SetDebugUtilsObjectNameEXT) {
        VkDebugUtilsObjectNameInfoEXT local_name_info;
        memcpy(&local_name_info, pNameInfo, sizeof(VkDebugUtilsObjectNameInfoEXT));
        // If this is a physical device, we have to replace it with the proper one for the next call.
        if (pNameInfo->objectType == VK_OBJECT_TYPE_PHYSICAL_DEVICE) {
            struct loader_physical_device_term *phys_dev_term = (struct loader_physical_device_term *)(uintptr_t)pNameInfo->objectHandle;
            local_name_info.objectHandle = (uint64_t)(uintptr_t)phys_dev_term->phys_dev;
        // If this is a KHR_surface, and the ICD has created its own, we have to replace it with the proper one for the next call.
        } else if (pNameInfo->objectType == VK_OBJECT_TYPE_SURFACE_KHR) {
            if (NULL != icd_term && NULL != icd_term->dispatch.CreateSwapchainKHR) {
                VkIcdSurface *icd_surface = (VkIcdSurface *)(uintptr_t)pNameInfo->objectHandle;
                if (NULL != icd_surface->real_icd_surfaces) {
                    local_name_info.objectHandle = (uint64_t)icd_surface->real_icd_surfaces[icd_index];
                }
            }
        }
        return icd_term->dispatch.SetDebugUtilsObjectNameEXT(device, &local_name_info);
    } else {
        return VK_SUCCESS;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL SetDebugUtilsObjectTagEXT(
    VkDevice                                    device,
    const VkDebugUtilsObjectTagInfoEXT*         pTagInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    VkDebugUtilsObjectTagInfoEXT local_tag_info;
    memcpy(&local_tag_info, pTagInfo, sizeof(VkDebugUtilsObjectTagInfoEXT));
    // If this is a physical device, we have to replace it with the proper one for the next call.
    if (pTagInfo->objectType == VK_OBJECT_TYPE_PHYSICAL_DEVICE) {
        struct loader_physical_device_tramp *phys_dev_tramp = (struct loader_physical_device_tramp *)(uintptr_t)pTagInfo->objectHandle;
        local_tag_info.objectHandle = (uint64_t)(uintptr_t)phys_dev_tramp->phys_dev;
    }
    if (disp->SetDebugUtilsObjectTagEXT != NULL) {
        return disp->SetDebugUtilsObjectTagEXT(device, &local_tag_info);
    } else {
        return VK_SUCCESS;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL terminator_SetDebugUtilsObjectTagEXT(
    VkDevice                                    device,
    const VkDebugUtilsObjectTagInfoEXT*         pTagInfo) {
    uint32_t icd_index = 0;
    struct loader_device *dev;
    struct loader_icd_term *icd_term = loader_get_icd_and_device(device, &dev, &icd_index);
    if (NULL != icd_term && NULL != icd_term->dispatch.SetDebugUtilsObjectTagEXT) {
        VkDebugUtilsObjectTagInfoEXT local_tag_info;
        memcpy(&local_tag_info, pTagInfo, sizeof(VkDebugUtilsObjectTagInfoEXT));
        // If this is a physical device, we have to replace it with the proper one for the next call.
        if (pTagInfo->objectType == VK_OBJECT_TYPE_PHYSICAL_DEVICE) {
            struct loader_physical_device_term *phys_dev_term = (struct loader_physical_device_term *)(uintptr_t)pTagInfo->objectHandle;
            local_tag_info.objectHandle = (uint64_t)(uintptr_t)phys_dev_term->phys_dev;
        // If this is a KHR_surface, and the ICD has created its own, we have to replace it with the proper one for the next call.
        } else if (pTagInfo->objectType == VK_OBJECT_TYPE_SURFACE_KHR) {
            if (NULL != icd_term && NULL != icd_term->dispatch.CreateSwapchainKHR) {
                VkIcdSurface *icd_surface = (VkIcdSurface *)(uintptr_t)pTagInfo->objectHandle;
                if (NULL != icd_surface->real_icd_surfaces) {
                    local_tag_info.objectHandle = (uint64_t)icd_surface->real_icd_surfaces[icd_index];
                }
            }
        }
        return icd_term->dispatch.SetDebugUtilsObjectTagEXT(device, &local_tag_info);
    } else {
        return VK_SUCCESS;
    }
}

VKAPI_ATTR void VKAPI_CALL QueueBeginDebugUtilsLabelEXT(
    VkQueue                                     queue,
    const VkDebugUtilsLabelEXT*                 pLabelInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(queue);
    if (disp->QueueBeginDebugUtilsLabelEXT != NULL) {
        disp->QueueBeginDebugUtilsLabelEXT(queue, pLabelInfo);
    }
}

VKAPI_ATTR void VKAPI_CALL terminator_QueueBeginDebugUtilsLabelEXT(
    VkQueue                                     queue,
    const VkDebugUtilsLabelEXT*                 pLabelInfo) {
    uint32_t icd_index = 0;
    struct loader_device *dev;
    struct loader_icd_term *icd_term = loader_get_icd_and_device(queue, &dev, &icd_index);
    if (NULL != icd_term && NULL != icd_term->dispatch.QueueBeginDebugUtilsLabelEXT) {
        icd_term->dispatch.QueueBeginDebugUtilsLabelEXT(queue, pLabelInfo);
    }
}

VKAPI_ATTR void VKAPI_CALL QueueEndDebugUtilsLabelEXT(
    VkQueue                                     queue) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(queue);
    if (disp->QueueEndDebugUtilsLabelEXT != NULL) {
        disp->QueueEndDebugUtilsLabelEXT(queue);
    }
}

VKAPI_ATTR void VKAPI_CALL terminator_QueueEndDebugUtilsLabelEXT(
    VkQueue                                     queue) {
    uint32_t icd_index = 0;
    struct loader_device *dev;
    struct loader_icd_term *icd_term = loader_get_icd_and_device(queue, &dev, &icd_index);
    if (NULL != icd_term && NULL != icd_term->dispatch.QueueEndDebugUtilsLabelEXT) {
        icd_term->dispatch.QueueEndDebugUtilsLabelEXT(queue);
    }
}

VKAPI_ATTR void VKAPI_CALL QueueInsertDebugUtilsLabelEXT(
    VkQueue                                     queue,
    const VkDebugUtilsLabelEXT*                 pLabelInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(queue);
    if (disp->QueueInsertDebugUtilsLabelEXT != NULL) {
        disp->QueueInsertDebugUtilsLabelEXT(queue, pLabelInfo);
    }
}

VKAPI_ATTR void VKAPI_CALL terminator_QueueInsertDebugUtilsLabelEXT(
    VkQueue                                     queue,
    const VkDebugUtilsLabelEXT*                 pLabelInfo) {
    uint32_t icd_index = 0;
    struct loader_device *dev;
    struct loader_icd_term *icd_term = loader_get_icd_and_device(queue, &dev, &icd_index);
    if (NULL != icd_term && NULL != icd_term->dispatch.QueueInsertDebugUtilsLabelEXT) {
        icd_term->dispatch.QueueInsertDebugUtilsLabelEXT(queue, pLabelInfo);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdBeginDebugUtilsLabelEXT(
    VkCommandBuffer                             commandBuffer,
    const VkDebugUtilsLabelEXT*                 pLabelInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    if (disp->CmdBeginDebugUtilsLabelEXT != NULL) {
        disp->CmdBeginDebugUtilsLabelEXT(commandBuffer, pLabelInfo);
    }
}

VKAPI_ATTR void VKAPI_CALL terminator_CmdBeginDebugUtilsLabelEXT(
    VkCommandBuffer                             commandBuffer,
    const VkDebugUtilsLabelEXT*                 pLabelInfo) {
    uint32_t icd_index = 0;
    struct loader_device *dev;
    struct loader_icd_term *icd_term = loader_get_icd_and_device(commandBuffer, &dev, &icd_index);
    if (NULL != icd_term && NULL != icd_term->dispatch.CmdBeginDebugUtilsLabelEXT) {
        icd_term->dispatch.CmdBeginDebugUtilsLabelEXT(commandBuffer, pLabelInfo);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdEndDebugUtilsLabelEXT(
    VkCommandBuffer                             commandBuffer) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    if (disp->CmdEndDebugUtilsLabelEXT != NULL) {
        disp->CmdEndDebugUtilsLabelEXT(commandBuffer);
    }
}

VKAPI_ATTR void VKAPI_CALL terminator_CmdEndDebugUtilsLabelEXT(
    VkCommandBuffer                             commandBuffer) {
    uint32_t icd_index = 0;
    struct loader_device *dev;
    struct loader_icd_term *icd_term = loader_get_icd_and_device(commandBuffer, &dev, &icd_index);
    if (NULL != icd_term && NULL != icd_term->dispatch.CmdEndDebugUtilsLabelEXT) {
        icd_term->dispatch.CmdEndDebugUtilsLabelEXT(commandBuffer);
    }
}

VKAPI_ATTR void VKAPI_CALL CmdInsertDebugUtilsLabelEXT(
    VkCommandBuffer                             commandBuffer,
    const VkDebugUtilsLabelEXT*                 pLabelInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    if (disp->CmdInsertDebugUtilsLabelEXT != NULL) {
        disp->CmdInsertDebugUtilsLabelEXT(commandBuffer, pLabelInfo);
    }
}

VKAPI_ATTR void VKAPI_CALL terminator_CmdInsertDebugUtilsLabelEXT(
    VkCommandBuffer                             commandBuffer,
    const VkDebugUtilsLabelEXT*                 pLabelInfo) {
    uint32_t icd_index = 0;
    struct loader_device *dev;
    struct loader_icd_term *icd_term = loader_get_icd_and_device(commandBuffer, &dev, &icd_index);
    if (NULL != icd_term && NULL != icd_term->dispatch.CmdInsertDebugUtilsLabelEXT) {
        icd_term->dispatch.CmdInsertDebugUtilsLabelEXT(commandBuffer, pLabelInfo);
    }
}


// ---- VK_ANDROID_external_memory_android_hardware_buffer extension trampoline/terminators

#ifdef VK_USE_PLATFORM_ANDROID_KHR
VKAPI_ATTR VkResult VKAPI_CALL GetAndroidHardwareBufferPropertiesANDROID(
    VkDevice                                    device,
    const struct AHardwareBuffer*               buffer,
    VkAndroidHardwareBufferPropertiesANDROID*   pProperties) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetAndroidHardwareBufferPropertiesANDROID(device, buffer, pProperties);
}

#endif // VK_USE_PLATFORM_ANDROID_KHR
#ifdef VK_USE_PLATFORM_ANDROID_KHR
VKAPI_ATTR VkResult VKAPI_CALL GetMemoryAndroidHardwareBufferANDROID(
    VkDevice                                    device,
    const VkMemoryGetAndroidHardwareBufferInfoANDROID* pInfo,
    struct AHardwareBuffer**                    pBuffer) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetMemoryAndroidHardwareBufferANDROID(device, pInfo, pBuffer);
}

#endif // VK_USE_PLATFORM_ANDROID_KHR

// ---- VK_EXT_sample_locations extension trampoline/terminators

VKAPI_ATTR void VKAPI_CALL CmdSetSampleLocationsEXT(
    VkCommandBuffer                             commandBuffer,
    const VkSampleLocationsInfoEXT*             pSampleLocationsInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdSetSampleLocationsEXT(commandBuffer, pSampleLocationsInfo);
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceMultisamplePropertiesEXT(
    VkPhysicalDevice                            physicalDevice,
    VkSampleCountFlagBits                       samples,
    VkMultisamplePropertiesEXT*                 pMultisampleProperties) {
    const VkLayerInstanceDispatchTable *disp;
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    disp = loader_get_instance_layer_dispatch(physicalDevice);
    disp->GetPhysicalDeviceMultisamplePropertiesEXT(unwrapped_phys_dev, samples, pMultisampleProperties);
}

VKAPI_ATTR void VKAPI_CALL terminator_GetPhysicalDeviceMultisamplePropertiesEXT(
    VkPhysicalDevice                            physicalDevice,
    VkSampleCountFlagBits                       samples,
    VkMultisamplePropertiesEXT*                 pMultisampleProperties) {
    struct loader_physical_device_term *phys_dev_term = (struct loader_physical_device_term *)physicalDevice;
    struct loader_icd_term *icd_term = phys_dev_term->this_icd_term;
    if (NULL == icd_term->dispatch.GetPhysicalDeviceMultisamplePropertiesEXT) {
        loader_log(icd_term->this_instance, VK_DEBUG_REPORT_ERROR_BIT_EXT, 0,
                   "ICD associated with VkPhysicalDevice does not support GetPhysicalDeviceMultisamplePropertiesEXT");
    }
    icd_term->dispatch.GetPhysicalDeviceMultisamplePropertiesEXT(phys_dev_term->phys_dev, samples, pMultisampleProperties);
}


// ---- VK_EXT_image_drm_format_modifier extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL GetImageDrmFormatModifierPropertiesEXT(
    VkDevice                                    device,
    VkImage                                     image,
    VkImageDrmFormatModifierPropertiesEXT*      pProperties) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetImageDrmFormatModifierPropertiesEXT(device, image, pProperties);
}


// ---- VK_EXT_validation_cache extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL CreateValidationCacheEXT(
    VkDevice                                    device,
    const VkValidationCacheCreateInfoEXT*       pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkValidationCacheEXT*                       pValidationCache) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->CreateValidationCacheEXT(device, pCreateInfo, pAllocator, pValidationCache);
}

VKAPI_ATTR void VKAPI_CALL DestroyValidationCacheEXT(
    VkDevice                                    device,
    VkValidationCacheEXT                        validationCache,
    const VkAllocationCallbacks*                pAllocator) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->DestroyValidationCacheEXT(device, validationCache, pAllocator);
}

VKAPI_ATTR VkResult VKAPI_CALL MergeValidationCachesEXT(
    VkDevice                                    device,
    VkValidationCacheEXT                        dstCache,
    uint32_t                                    srcCacheCount,
    const VkValidationCacheEXT*                 pSrcCaches) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->MergeValidationCachesEXT(device, dstCache, srcCacheCount, pSrcCaches);
}

VKAPI_ATTR VkResult VKAPI_CALL GetValidationCacheDataEXT(
    VkDevice                                    device,
    VkValidationCacheEXT                        validationCache,
    size_t*                                     pDataSize,
    void*                                       pData) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetValidationCacheDataEXT(device, validationCache, pDataSize, pData);
}


// ---- VK_NV_shading_rate_image extension trampoline/terminators

VKAPI_ATTR void VKAPI_CALL CmdBindShadingRateImageNV(
    VkCommandBuffer                             commandBuffer,
    VkImageView                                 imageView,
    VkImageLayout                               imageLayout) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdBindShadingRateImageNV(commandBuffer, imageView, imageLayout);
}

VKAPI_ATTR void VKAPI_CALL CmdSetViewportShadingRatePaletteNV(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    firstViewport,
    uint32_t                                    viewportCount,
    const VkShadingRatePaletteNV*               pShadingRatePalettes) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdSetViewportShadingRatePaletteNV(commandBuffer, firstViewport, viewportCount, pShadingRatePalettes);
}

VKAPI_ATTR void VKAPI_CALL CmdSetCoarseSampleOrderNV(
    VkCommandBuffer                             commandBuffer,
    VkCoarseSampleOrderTypeNV                   sampleOrderType,
    uint32_t                                    customSampleOrderCount,
    const VkCoarseSampleOrderCustomNV*          pCustomSampleOrders) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdSetCoarseSampleOrderNV(commandBuffer, sampleOrderType, customSampleOrderCount, pCustomSampleOrders);
}


// ---- VK_NV_ray_tracing extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL CreateAccelerationStructureNV(
    VkDevice                                    device,
    const VkAccelerationStructureCreateInfoNV*  pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkAccelerationStructureNV*                  pAccelerationStructure) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->CreateAccelerationStructureNV(device, pCreateInfo, pAllocator, pAccelerationStructure);
}

VKAPI_ATTR void VKAPI_CALL DestroyAccelerationStructureKHR(
    VkDevice                                    device,
    VkAccelerationStructureKHR                  accelerationStructure,
    const VkAllocationCallbacks*                pAllocator) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->DestroyAccelerationStructureKHR(device, accelerationStructure, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL DestroyAccelerationStructureNV(
    VkDevice                                    device,
    VkAccelerationStructureKHR                  accelerationStructure,
    const VkAllocationCallbacks*                pAllocator) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->DestroyAccelerationStructureNV(device, accelerationStructure, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL GetAccelerationStructureMemoryRequirementsNV(
    VkDevice                                    device,
    const VkAccelerationStructureMemoryRequirementsInfoNV* pInfo,
    VkMemoryRequirements2KHR*                   pMemoryRequirements) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->GetAccelerationStructureMemoryRequirementsNV(device, pInfo, pMemoryRequirements);
}

VKAPI_ATTR VkResult VKAPI_CALL BindAccelerationStructureMemoryKHR(
    VkDevice                                    device,
    uint32_t                                    bindInfoCount,
    const VkBindAccelerationStructureMemoryInfoKHR* pBindInfos) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->BindAccelerationStructureMemoryKHR(device, bindInfoCount, pBindInfos);
}

VKAPI_ATTR VkResult VKAPI_CALL BindAccelerationStructureMemoryNV(
    VkDevice                                    device,
    uint32_t                                    bindInfoCount,
    const VkBindAccelerationStructureMemoryInfoKHR* pBindInfos) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->BindAccelerationStructureMemoryNV(device, bindInfoCount, pBindInfos);
}

VKAPI_ATTR void VKAPI_CALL CmdBuildAccelerationStructureNV(
    VkCommandBuffer                             commandBuffer,
    const VkAccelerationStructureInfoNV*        pInfo,
    VkBuffer                                    instanceData,
    VkDeviceSize                                instanceOffset,
    VkBool32                                    update,
    VkAccelerationStructureKHR                  dst,
    VkAccelerationStructureKHR                  src,
    VkBuffer                                    scratch,
    VkDeviceSize                                scratchOffset) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdBuildAccelerationStructureNV(commandBuffer, pInfo, instanceData, instanceOffset, update, dst, src, scratch, scratchOffset);
}

VKAPI_ATTR void VKAPI_CALL CmdCopyAccelerationStructureNV(
    VkCommandBuffer                             commandBuffer,
    VkAccelerationStructureKHR                  dst,
    VkAccelerationStructureKHR                  src,
    VkCopyAccelerationStructureModeKHR          mode) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdCopyAccelerationStructureNV(commandBuffer, dst, src, mode);
}

VKAPI_ATTR void VKAPI_CALL CmdTraceRaysNV(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    raygenShaderBindingTableBuffer,
    VkDeviceSize                                raygenShaderBindingOffset,
    VkBuffer                                    missShaderBindingTableBuffer,
    VkDeviceSize                                missShaderBindingOffset,
    VkDeviceSize                                missShaderBindingStride,
    VkBuffer                                    hitShaderBindingTableBuffer,
    VkDeviceSize                                hitShaderBindingOffset,
    VkDeviceSize                                hitShaderBindingStride,
    VkBuffer                                    callableShaderBindingTableBuffer,
    VkDeviceSize                                callableShaderBindingOffset,
    VkDeviceSize                                callableShaderBindingStride,
    uint32_t                                    width,
    uint32_t                                    height,
    uint32_t                                    depth) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdTraceRaysNV(commandBuffer, raygenShaderBindingTableBuffer, raygenShaderBindingOffset, missShaderBindingTableBuffer, missShaderBindingOffset, missShaderBindingStride, hitShaderBindingTableBuffer, hitShaderBindingOffset, hitShaderBindingStride, callableShaderBindingTableBuffer, callableShaderBindingOffset, callableShaderBindingStride, width, height, depth);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateRayTracingPipelinesNV(
    VkDevice                                    device,
    VkPipelineCache                             pipelineCache,
    uint32_t                                    createInfoCount,
    const VkRayTracingPipelineCreateInfoNV*     pCreateInfos,
    const VkAllocationCallbacks*                pAllocator,
    VkPipeline*                                 pPipelines) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->CreateRayTracingPipelinesNV(device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
}

VKAPI_ATTR VkResult VKAPI_CALL GetRayTracingShaderGroupHandlesKHR(
    VkDevice                                    device,
    VkPipeline                                  pipeline,
    uint32_t                                    firstGroup,
    uint32_t                                    groupCount,
    size_t                                      dataSize,
    void*                                       pData) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetRayTracingShaderGroupHandlesKHR(device, pipeline, firstGroup, groupCount, dataSize, pData);
}

VKAPI_ATTR VkResult VKAPI_CALL GetRayTracingShaderGroupHandlesNV(
    VkDevice                                    device,
    VkPipeline                                  pipeline,
    uint32_t                                    firstGroup,
    uint32_t                                    groupCount,
    size_t                                      dataSize,
    void*                                       pData) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetRayTracingShaderGroupHandlesNV(device, pipeline, firstGroup, groupCount, dataSize, pData);
}

VKAPI_ATTR VkResult VKAPI_CALL GetAccelerationStructureHandleNV(
    VkDevice                                    device,
    VkAccelerationStructureKHR                  accelerationStructure,
    size_t                                      dataSize,
    void*                                       pData) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetAccelerationStructureHandleNV(device, accelerationStructure, dataSize, pData);
}

VKAPI_ATTR void VKAPI_CALL CmdWriteAccelerationStructuresPropertiesKHR(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    accelerationStructureCount,
    const VkAccelerationStructureKHR*           pAccelerationStructures,
    VkQueryType                                 queryType,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdWriteAccelerationStructuresPropertiesKHR(commandBuffer, accelerationStructureCount, pAccelerationStructures, queryType, queryPool, firstQuery);
}

VKAPI_ATTR void VKAPI_CALL CmdWriteAccelerationStructuresPropertiesNV(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    accelerationStructureCount,
    const VkAccelerationStructureKHR*           pAccelerationStructures,
    VkQueryType                                 queryType,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdWriteAccelerationStructuresPropertiesNV(commandBuffer, accelerationStructureCount, pAccelerationStructures, queryType, queryPool, firstQuery);
}

VKAPI_ATTR VkResult VKAPI_CALL CompileDeferredNV(
    VkDevice                                    device,
    VkPipeline                                  pipeline,
    uint32_t                                    shader) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->CompileDeferredNV(device, pipeline, shader);
}


// ---- VK_EXT_external_memory_host extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL GetMemoryHostPointerPropertiesEXT(
    VkDevice                                    device,
    VkExternalMemoryHandleTypeFlagBits          handleType,
    const void*                                 pHostPointer,
    VkMemoryHostPointerPropertiesEXT*           pMemoryHostPointerProperties) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetMemoryHostPointerPropertiesEXT(device, handleType, pHostPointer, pMemoryHostPointerProperties);
}


// ---- VK_AMD_buffer_marker extension trampoline/terminators

VKAPI_ATTR void VKAPI_CALL CmdWriteBufferMarkerAMD(
    VkCommandBuffer                             commandBuffer,
    VkPipelineStageFlagBits                     pipelineStage,
    VkBuffer                                    dstBuffer,
    VkDeviceSize                                dstOffset,
    uint32_t                                    marker) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdWriteBufferMarkerAMD(commandBuffer, pipelineStage, dstBuffer, dstOffset, marker);
}


// ---- VK_EXT_calibrated_timestamps extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceCalibrateableTimeDomainsEXT(
    VkPhysicalDevice                            physicalDevice,
    uint32_t*                                   pTimeDomainCount,
    VkTimeDomainEXT*                            pTimeDomains) {
    const VkLayerInstanceDispatchTable *disp;
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    disp = loader_get_instance_layer_dispatch(physicalDevice);
    return disp->GetPhysicalDeviceCalibrateableTimeDomainsEXT(unwrapped_phys_dev, pTimeDomainCount, pTimeDomains);
}

VKAPI_ATTR VkResult VKAPI_CALL terminator_GetPhysicalDeviceCalibrateableTimeDomainsEXT(
    VkPhysicalDevice                            physicalDevice,
    uint32_t*                                   pTimeDomainCount,
    VkTimeDomainEXT*                            pTimeDomains) {
    struct loader_physical_device_term *phys_dev_term = (struct loader_physical_device_term *)physicalDevice;
    struct loader_icd_term *icd_term = phys_dev_term->this_icd_term;
    if (NULL == icd_term->dispatch.GetPhysicalDeviceCalibrateableTimeDomainsEXT) {
        loader_log(icd_term->this_instance, VK_DEBUG_REPORT_ERROR_BIT_EXT, 0,
                   "ICD associated with VkPhysicalDevice does not support GetPhysicalDeviceCalibrateableTimeDomainsEXT");
    }
    return icd_term->dispatch.GetPhysicalDeviceCalibrateableTimeDomainsEXT(phys_dev_term->phys_dev, pTimeDomainCount, pTimeDomains);
}

VKAPI_ATTR VkResult VKAPI_CALL GetCalibratedTimestampsEXT(
    VkDevice                                    device,
    uint32_t                                    timestampCount,
    const VkCalibratedTimestampInfoEXT*         pTimestampInfos,
    uint64_t*                                   pTimestamps,
    uint64_t*                                   pMaxDeviation) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetCalibratedTimestampsEXT(device, timestampCount, pTimestampInfos, pTimestamps, pMaxDeviation);
}


// ---- VK_NV_mesh_shader extension trampoline/terminators

VKAPI_ATTR void VKAPI_CALL CmdDrawMeshTasksNV(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    taskCount,
    uint32_t                                    firstTask) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdDrawMeshTasksNV(commandBuffer, taskCount, firstTask);
}

VKAPI_ATTR void VKAPI_CALL CmdDrawMeshTasksIndirectNV(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    buffer,
    VkDeviceSize                                offset,
    uint32_t                                    drawCount,
    uint32_t                                    stride) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdDrawMeshTasksIndirectNV(commandBuffer, buffer, offset, drawCount, stride);
}

VKAPI_ATTR void VKAPI_CALL CmdDrawMeshTasksIndirectCountNV(
    VkCommandBuffer                             commandBuffer,
    VkBuffer                                    buffer,
    VkDeviceSize                                offset,
    VkBuffer                                    countBuffer,
    VkDeviceSize                                countBufferOffset,
    uint32_t                                    maxDrawCount,
    uint32_t                                    stride) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdDrawMeshTasksIndirectCountNV(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
}


// ---- VK_NV_scissor_exclusive extension trampoline/terminators

VKAPI_ATTR void VKAPI_CALL CmdSetExclusiveScissorNV(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    firstExclusiveScissor,
    uint32_t                                    exclusiveScissorCount,
    const VkRect2D*                             pExclusiveScissors) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdSetExclusiveScissorNV(commandBuffer, firstExclusiveScissor, exclusiveScissorCount, pExclusiveScissors);
}


// ---- VK_NV_device_diagnostic_checkpoints extension trampoline/terminators

VKAPI_ATTR void VKAPI_CALL CmdSetCheckpointNV(
    VkCommandBuffer                             commandBuffer,
    const void*                                 pCheckpointMarker) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdSetCheckpointNV(commandBuffer, pCheckpointMarker);
}

VKAPI_ATTR void VKAPI_CALL GetQueueCheckpointDataNV(
    VkQueue                                     queue,
    uint32_t*                                   pCheckpointDataCount,
    VkCheckpointDataNV*                         pCheckpointData) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(queue);
    disp->GetQueueCheckpointDataNV(queue, pCheckpointDataCount, pCheckpointData);
}


// ---- VK_INTEL_performance_query extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL InitializePerformanceApiINTEL(
    VkDevice                                    device,
    const VkInitializePerformanceApiInfoINTEL*  pInitializeInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->InitializePerformanceApiINTEL(device, pInitializeInfo);
}

VKAPI_ATTR void VKAPI_CALL UninitializePerformanceApiINTEL(
    VkDevice                                    device) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->UninitializePerformanceApiINTEL(device);
}

VKAPI_ATTR VkResult VKAPI_CALL CmdSetPerformanceMarkerINTEL(
    VkCommandBuffer                             commandBuffer,
    const VkPerformanceMarkerInfoINTEL*         pMarkerInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    return disp->CmdSetPerformanceMarkerINTEL(commandBuffer, pMarkerInfo);
}

VKAPI_ATTR VkResult VKAPI_CALL CmdSetPerformanceStreamMarkerINTEL(
    VkCommandBuffer                             commandBuffer,
    const VkPerformanceStreamMarkerInfoINTEL*   pMarkerInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    return disp->CmdSetPerformanceStreamMarkerINTEL(commandBuffer, pMarkerInfo);
}

VKAPI_ATTR VkResult VKAPI_CALL CmdSetPerformanceOverrideINTEL(
    VkCommandBuffer                             commandBuffer,
    const VkPerformanceOverrideInfoINTEL*       pOverrideInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    return disp->CmdSetPerformanceOverrideINTEL(commandBuffer, pOverrideInfo);
}

VKAPI_ATTR VkResult VKAPI_CALL AcquirePerformanceConfigurationINTEL(
    VkDevice                                    device,
    const VkPerformanceConfigurationAcquireInfoINTEL* pAcquireInfo,
    VkPerformanceConfigurationINTEL*            pConfiguration) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->AcquirePerformanceConfigurationINTEL(device, pAcquireInfo, pConfiguration);
}

VKAPI_ATTR VkResult VKAPI_CALL ReleasePerformanceConfigurationINTEL(
    VkDevice                                    device,
    VkPerformanceConfigurationINTEL             configuration) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->ReleasePerformanceConfigurationINTEL(device, configuration);
}

VKAPI_ATTR VkResult VKAPI_CALL QueueSetPerformanceConfigurationINTEL(
    VkQueue                                     queue,
    VkPerformanceConfigurationINTEL             configuration) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(queue);
    return disp->QueueSetPerformanceConfigurationINTEL(queue, configuration);
}

VKAPI_ATTR VkResult VKAPI_CALL GetPerformanceParameterINTEL(
    VkDevice                                    device,
    VkPerformanceParameterTypeINTEL             parameter,
    VkPerformanceValueINTEL*                    pValue) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetPerformanceParameterINTEL(device, parameter, pValue);
}


// ---- VK_AMD_display_native_hdr extension trampoline/terminators

VKAPI_ATTR void VKAPI_CALL SetLocalDimmingAMD(
    VkDevice                                    device,
    VkSwapchainKHR                              swapChain,
    VkBool32                                    localDimmingEnable) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->SetLocalDimmingAMD(device, swapChain, localDimmingEnable);
}


// ---- VK_FUCHSIA_imagepipe_surface extension trampoline/terminators

#ifdef VK_USE_PLATFORM_FUCHSIA
VKAPI_ATTR VkResult VKAPI_CALL CreateImagePipeSurfaceFUCHSIA(
    VkInstance                                  instance,
    const VkImagePipeSurfaceCreateInfoFUCHSIA*  pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface) {
#error("Not implemented. Likely needs to be manually generated!");
    return disp->CreateImagePipeSurfaceFUCHSIA(instance, pCreateInfo, pAllocator, pSurface);
}

VKAPI_ATTR VkResult VKAPI_CALL terminator_CreateImagePipeSurfaceFUCHSIA(
    VkInstance                                  instance,
    const VkImagePipeSurfaceCreateInfoFUCHSIA*  pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface) {
#error("Not implemented. Likely needs to be manually generated!");
}

#endif // VK_USE_PLATFORM_FUCHSIA

// ---- VK_EXT_buffer_device_address extension trampoline/terminators

VKAPI_ATTR VkDeviceAddress VKAPI_CALL GetBufferDeviceAddressEXT(
    VkDevice                                    device,
    const VkBufferDeviceAddressInfo*            pInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetBufferDeviceAddressEXT(device, pInfo);
}


// ---- VK_NV_cooperative_matrix extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceCooperativeMatrixPropertiesNV(
    VkPhysicalDevice                            physicalDevice,
    uint32_t*                                   pPropertyCount,
    VkCooperativeMatrixPropertiesNV*            pProperties) {
    const VkLayerInstanceDispatchTable *disp;
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    disp = loader_get_instance_layer_dispatch(physicalDevice);
    return disp->GetPhysicalDeviceCooperativeMatrixPropertiesNV(unwrapped_phys_dev, pPropertyCount, pProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL terminator_GetPhysicalDeviceCooperativeMatrixPropertiesNV(
    VkPhysicalDevice                            physicalDevice,
    uint32_t*                                   pPropertyCount,
    VkCooperativeMatrixPropertiesNV*            pProperties) {
    struct loader_physical_device_term *phys_dev_term = (struct loader_physical_device_term *)physicalDevice;
    struct loader_icd_term *icd_term = phys_dev_term->this_icd_term;
    if (NULL == icd_term->dispatch.GetPhysicalDeviceCooperativeMatrixPropertiesNV) {
        loader_log(icd_term->this_instance, VK_DEBUG_REPORT_ERROR_BIT_EXT, 0,
                   "ICD associated with VkPhysicalDevice does not support GetPhysicalDeviceCooperativeMatrixPropertiesNV");
    }
    return icd_term->dispatch.GetPhysicalDeviceCooperativeMatrixPropertiesNV(phys_dev_term->phys_dev, pPropertyCount, pProperties);
}


// ---- VK_NV_coverage_reduction_mode extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV(
    VkPhysicalDevice                            physicalDevice,
    uint32_t*                                   pCombinationCount,
    VkFramebufferMixedSamplesCombinationNV*     pCombinations) {
    const VkLayerInstanceDispatchTable *disp;
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    disp = loader_get_instance_layer_dispatch(physicalDevice);
    return disp->GetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV(unwrapped_phys_dev, pCombinationCount, pCombinations);
}

VKAPI_ATTR VkResult VKAPI_CALL terminator_GetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV(
    VkPhysicalDevice                            physicalDevice,
    uint32_t*                                   pCombinationCount,
    VkFramebufferMixedSamplesCombinationNV*     pCombinations) {
    struct loader_physical_device_term *phys_dev_term = (struct loader_physical_device_term *)physicalDevice;
    struct loader_icd_term *icd_term = phys_dev_term->this_icd_term;
    if (NULL == icd_term->dispatch.GetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV) {
        loader_log(icd_term->this_instance, VK_DEBUG_REPORT_ERROR_BIT_EXT, 0,
                   "ICD associated with VkPhysicalDevice does not support GetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV");
    }
    return icd_term->dispatch.GetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV(phys_dev_term->phys_dev, pCombinationCount, pCombinations);
}


// ---- VK_EXT_full_screen_exclusive extension trampoline/terminators

#ifdef VK_USE_PLATFORM_WIN32_KHR
VKAPI_ATTR VkResult VKAPI_CALL AcquireFullScreenExclusiveModeEXT(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->AcquireFullScreenExclusiveModeEXT(device, swapchain);
}

#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
VKAPI_ATTR VkResult VKAPI_CALL ReleaseFullScreenExclusiveModeEXT(
    VkDevice                                    device,
    VkSwapchainKHR                              swapchain) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->ReleaseFullScreenExclusiveModeEXT(device, swapchain);
}

#endif // VK_USE_PLATFORM_WIN32_KHR

// ---- VK_EXT_line_rasterization extension trampoline/terminators

VKAPI_ATTR void VKAPI_CALL CmdSetLineStippleEXT(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    lineStippleFactor,
    uint16_t                                    lineStipplePattern) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdSetLineStippleEXT(commandBuffer, lineStippleFactor, lineStipplePattern);
}


// ---- VK_EXT_host_query_reset extension trampoline/terminators

VKAPI_ATTR void VKAPI_CALL ResetQueryPoolEXT(
    VkDevice                                    device,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery,
    uint32_t                                    queryCount) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->ResetQueryPoolEXT(device, queryPool, firstQuery, queryCount);
}


// ---- VK_EXT_extended_dynamic_state extension trampoline/terminators

VKAPI_ATTR void VKAPI_CALL CmdSetCullModeEXT(
    VkCommandBuffer                             commandBuffer,
    VkCullModeFlags                             cullMode) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdSetCullModeEXT(commandBuffer, cullMode);
}

VKAPI_ATTR void VKAPI_CALL CmdSetFrontFaceEXT(
    VkCommandBuffer                             commandBuffer,
    VkFrontFace                                 frontFace) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdSetFrontFaceEXT(commandBuffer, frontFace);
}

VKAPI_ATTR void VKAPI_CALL CmdSetPrimitiveTopologyEXT(
    VkCommandBuffer                             commandBuffer,
    VkPrimitiveTopology                         primitiveTopology) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdSetPrimitiveTopologyEXT(commandBuffer, primitiveTopology);
}

VKAPI_ATTR void VKAPI_CALL CmdSetViewportWithCountEXT(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    viewportCount,
    const VkViewport*                           pViewports) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdSetViewportWithCountEXT(commandBuffer, viewportCount, pViewports);
}

VKAPI_ATTR void VKAPI_CALL CmdSetScissorWithCountEXT(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    scissorCount,
    const VkRect2D*                             pScissors) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdSetScissorWithCountEXT(commandBuffer, scissorCount, pScissors);
}

VKAPI_ATTR void VKAPI_CALL CmdBindVertexBuffers2EXT(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    firstBinding,
    uint32_t                                    bindingCount,
    const VkBuffer*                             pBuffers,
    const VkDeviceSize*                         pOffsets,
    const VkDeviceSize*                         pSizes,
    const VkDeviceSize*                         pStrides) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdBindVertexBuffers2EXT(commandBuffer, firstBinding, bindingCount, pBuffers, pOffsets, pSizes, pStrides);
}

VKAPI_ATTR void VKAPI_CALL CmdSetDepthTestEnableEXT(
    VkCommandBuffer                             commandBuffer,
    VkBool32                                    depthTestEnable) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdSetDepthTestEnableEXT(commandBuffer, depthTestEnable);
}

VKAPI_ATTR void VKAPI_CALL CmdSetDepthWriteEnableEXT(
    VkCommandBuffer                             commandBuffer,
    VkBool32                                    depthWriteEnable) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdSetDepthWriteEnableEXT(commandBuffer, depthWriteEnable);
}

VKAPI_ATTR void VKAPI_CALL CmdSetDepthCompareOpEXT(
    VkCommandBuffer                             commandBuffer,
    VkCompareOp                                 depthCompareOp) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdSetDepthCompareOpEXT(commandBuffer, depthCompareOp);
}

VKAPI_ATTR void VKAPI_CALL CmdSetDepthBoundsTestEnableEXT(
    VkCommandBuffer                             commandBuffer,
    VkBool32                                    depthBoundsTestEnable) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdSetDepthBoundsTestEnableEXT(commandBuffer, depthBoundsTestEnable);
}

VKAPI_ATTR void VKAPI_CALL CmdSetStencilTestEnableEXT(
    VkCommandBuffer                             commandBuffer,
    VkBool32                                    stencilTestEnable) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdSetStencilTestEnableEXT(commandBuffer, stencilTestEnable);
}

VKAPI_ATTR void VKAPI_CALL CmdSetStencilOpEXT(
    VkCommandBuffer                             commandBuffer,
    VkStencilFaceFlags                          faceMask,
    VkStencilOp                                 failOp,
    VkStencilOp                                 passOp,
    VkStencilOp                                 depthFailOp,
    VkCompareOp                                 compareOp) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdSetStencilOpEXT(commandBuffer, faceMask, failOp, passOp, depthFailOp, compareOp);
}


// ---- VK_NV_device_generated_commands extension trampoline/terminators

VKAPI_ATTR void VKAPI_CALL GetGeneratedCommandsMemoryRequirementsNV(
    VkDevice                                    device,
    const VkGeneratedCommandsMemoryRequirementsInfoNV* pInfo,
    VkMemoryRequirements2*                      pMemoryRequirements) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->GetGeneratedCommandsMemoryRequirementsNV(device, pInfo, pMemoryRequirements);
}

VKAPI_ATTR void VKAPI_CALL CmdPreprocessGeneratedCommandsNV(
    VkCommandBuffer                             commandBuffer,
    const VkGeneratedCommandsInfoNV*            pGeneratedCommandsInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdPreprocessGeneratedCommandsNV(commandBuffer, pGeneratedCommandsInfo);
}

VKAPI_ATTR void VKAPI_CALL CmdExecuteGeneratedCommandsNV(
    VkCommandBuffer                             commandBuffer,
    VkBool32                                    isPreprocessed,
    const VkGeneratedCommandsInfoNV*            pGeneratedCommandsInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdExecuteGeneratedCommandsNV(commandBuffer, isPreprocessed, pGeneratedCommandsInfo);
}

VKAPI_ATTR void VKAPI_CALL CmdBindPipelineShaderGroupNV(
    VkCommandBuffer                             commandBuffer,
    VkPipelineBindPoint                         pipelineBindPoint,
    VkPipeline                                  pipeline,
    uint32_t                                    groupIndex) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdBindPipelineShaderGroupNV(commandBuffer, pipelineBindPoint, pipeline, groupIndex);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateIndirectCommandsLayoutNV(
    VkDevice                                    device,
    const VkIndirectCommandsLayoutCreateInfoNV* pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkIndirectCommandsLayoutNV*                 pIndirectCommandsLayout) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->CreateIndirectCommandsLayoutNV(device, pCreateInfo, pAllocator, pIndirectCommandsLayout);
}

VKAPI_ATTR void VKAPI_CALL DestroyIndirectCommandsLayoutNV(
    VkDevice                                    device,
    VkIndirectCommandsLayoutNV                  indirectCommandsLayout,
    const VkAllocationCallbacks*                pAllocator) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->DestroyIndirectCommandsLayoutNV(device, indirectCommandsLayout, pAllocator);
}


// ---- VK_EXT_private_data extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL CreatePrivateDataSlotEXT(
    VkDevice                                    device,
    const VkPrivateDataSlotCreateInfoEXT*       pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkPrivateDataSlotEXT*                       pPrivateDataSlot) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->CreatePrivateDataSlotEXT(device, pCreateInfo, pAllocator, pPrivateDataSlot);
}

VKAPI_ATTR void VKAPI_CALL DestroyPrivateDataSlotEXT(
    VkDevice                                    device,
    VkPrivateDataSlotEXT                        privateDataSlot,
    const VkAllocationCallbacks*                pAllocator) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->DestroyPrivateDataSlotEXT(device, privateDataSlot, pAllocator);
}

VKAPI_ATTR VkResult VKAPI_CALL SetPrivateDataEXT(
    VkDevice                                    device,
    VkObjectType                                objectType,
    uint64_t                                    objectHandle,
    VkPrivateDataSlotEXT                        privateDataSlot,
    uint64_t                                    data) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->SetPrivateDataEXT(device, objectType, objectHandle, privateDataSlot, data);
}

VKAPI_ATTR void VKAPI_CALL GetPrivateDataEXT(
    VkDevice                                    device,
    VkObjectType                                objectType,
    uint64_t                                    objectHandle,
    VkPrivateDataSlotEXT                        privateDataSlot,
    uint64_t*                                   pData) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->GetPrivateDataEXT(device, objectType, objectHandle, privateDataSlot, pData);
}


// ---- VK_KHR_ray_tracing extension trampoline/terminators

#ifdef VK_ENABLE_BETA_EXTENSIONS
VKAPI_ATTR VkResult VKAPI_CALL CreateAccelerationStructureKHR(
    VkDevice                                    device,
    const VkAccelerationStructureCreateInfoKHR* pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkAccelerationStructureKHR*                 pAccelerationStructure) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->CreateAccelerationStructureKHR(device, pCreateInfo, pAllocator, pAccelerationStructure);
}

#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
VKAPI_ATTR void VKAPI_CALL GetAccelerationStructureMemoryRequirementsKHR(
    VkDevice                                    device,
    const VkAccelerationStructureMemoryRequirementsInfoKHR* pInfo,
    VkMemoryRequirements2*                      pMemoryRequirements) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->GetAccelerationStructureMemoryRequirementsKHR(device, pInfo, pMemoryRequirements);
}

#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
VKAPI_ATTR void VKAPI_CALL CmdBuildAccelerationStructureKHR(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    infoCount,
    const VkAccelerationStructureBuildGeometryInfoKHR* pInfos,
    const VkAccelerationStructureBuildOffsetInfoKHR* const* ppOffsetInfos) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdBuildAccelerationStructureKHR(commandBuffer, infoCount, pInfos, ppOffsetInfos);
}

#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
VKAPI_ATTR void VKAPI_CALL CmdBuildAccelerationStructureIndirectKHR(
    VkCommandBuffer                             commandBuffer,
    const VkAccelerationStructureBuildGeometryInfoKHR* pInfo,
    VkBuffer                                    indirectBuffer,
    VkDeviceSize                                indirectOffset,
    uint32_t                                    indirectStride) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdBuildAccelerationStructureIndirectKHR(commandBuffer, pInfo, indirectBuffer, indirectOffset, indirectStride);
}

#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
VKAPI_ATTR VkResult VKAPI_CALL BuildAccelerationStructureKHR(
    VkDevice                                    device,
    uint32_t                                    infoCount,
    const VkAccelerationStructureBuildGeometryInfoKHR* pInfos,
    const VkAccelerationStructureBuildOffsetInfoKHR* const* ppOffsetInfos) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->BuildAccelerationStructureKHR(device, infoCount, pInfos, ppOffsetInfos);
}

#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
VKAPI_ATTR VkResult VKAPI_CALL CopyAccelerationStructureKHR(
    VkDevice                                    device,
    const VkCopyAccelerationStructureInfoKHR*   pInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->CopyAccelerationStructureKHR(device, pInfo);
}

#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
VKAPI_ATTR VkResult VKAPI_CALL CopyAccelerationStructureToMemoryKHR(
    VkDevice                                    device,
    const VkCopyAccelerationStructureToMemoryInfoKHR* pInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->CopyAccelerationStructureToMemoryKHR(device, pInfo);
}

#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
VKAPI_ATTR VkResult VKAPI_CALL CopyMemoryToAccelerationStructureKHR(
    VkDevice                                    device,
    const VkCopyMemoryToAccelerationStructureInfoKHR* pInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->CopyMemoryToAccelerationStructureKHR(device, pInfo);
}

#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
VKAPI_ATTR VkResult VKAPI_CALL WriteAccelerationStructuresPropertiesKHR(
    VkDevice                                    device,
    uint32_t                                    accelerationStructureCount,
    const VkAccelerationStructureKHR*           pAccelerationStructures,
    VkQueryType                                 queryType,
    size_t                                      dataSize,
    void*                                       pData,
    size_t                                      stride) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->WriteAccelerationStructuresPropertiesKHR(device, accelerationStructureCount, pAccelerationStructures, queryType, dataSize, pData, stride);
}

#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
VKAPI_ATTR void VKAPI_CALL CmdCopyAccelerationStructureKHR(
    VkCommandBuffer                             commandBuffer,
    const VkCopyAccelerationStructureInfoKHR*   pInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdCopyAccelerationStructureKHR(commandBuffer, pInfo);
}

#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
VKAPI_ATTR void VKAPI_CALL CmdCopyAccelerationStructureToMemoryKHR(
    VkCommandBuffer                             commandBuffer,
    const VkCopyAccelerationStructureToMemoryInfoKHR* pInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdCopyAccelerationStructureToMemoryKHR(commandBuffer, pInfo);
}

#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
VKAPI_ATTR void VKAPI_CALL CmdCopyMemoryToAccelerationStructureKHR(
    VkCommandBuffer                             commandBuffer,
    const VkCopyMemoryToAccelerationStructureInfoKHR* pInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdCopyMemoryToAccelerationStructureKHR(commandBuffer, pInfo);
}

#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
VKAPI_ATTR void VKAPI_CALL CmdTraceRaysKHR(
    VkCommandBuffer                             commandBuffer,
    const VkStridedBufferRegionKHR*             pRaygenShaderBindingTable,
    const VkStridedBufferRegionKHR*             pMissShaderBindingTable,
    const VkStridedBufferRegionKHR*             pHitShaderBindingTable,
    const VkStridedBufferRegionKHR*             pCallableShaderBindingTable,
    uint32_t                                    width,
    uint32_t                                    height,
    uint32_t                                    depth) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdTraceRaysKHR(commandBuffer, pRaygenShaderBindingTable, pMissShaderBindingTable, pHitShaderBindingTable, pCallableShaderBindingTable, width, height, depth);
}

#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
VKAPI_ATTR VkResult VKAPI_CALL CreateRayTracingPipelinesKHR(
    VkDevice                                    device,
    VkPipelineCache                             pipelineCache,
    uint32_t                                    createInfoCount,
    const VkRayTracingPipelineCreateInfoKHR*    pCreateInfos,
    const VkAllocationCallbacks*                pAllocator,
    VkPipeline*                                 pPipelines) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->CreateRayTracingPipelinesKHR(device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
}

#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
VKAPI_ATTR VkDeviceAddress VKAPI_CALL GetAccelerationStructureDeviceAddressKHR(
    VkDevice                                    device,
    const VkAccelerationStructureDeviceAddressInfoKHR* pInfo) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetAccelerationStructureDeviceAddressKHR(device, pInfo);
}

#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
VKAPI_ATTR VkResult VKAPI_CALL GetRayTracingCaptureReplayShaderGroupHandlesKHR(
    VkDevice                                    device,
    VkPipeline                                  pipeline,
    uint32_t                                    firstGroup,
    uint32_t                                    groupCount,
    size_t                                      dataSize,
    void*                                       pData) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetRayTracingCaptureReplayShaderGroupHandlesKHR(device, pipeline, firstGroup, groupCount, dataSize, pData);
}

#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
VKAPI_ATTR void VKAPI_CALL CmdTraceRaysIndirectKHR(
    VkCommandBuffer                             commandBuffer,
    const VkStridedBufferRegionKHR*             pRaygenShaderBindingTable,
    const VkStridedBufferRegionKHR*             pMissShaderBindingTable,
    const VkStridedBufferRegionKHR*             pHitShaderBindingTable,
    const VkStridedBufferRegionKHR*             pCallableShaderBindingTable,
    VkBuffer                                    buffer,
    VkDeviceSize                                offset) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdTraceRaysIndirectKHR(commandBuffer, pRaygenShaderBindingTable, pMissShaderBindingTable, pHitShaderBindingTable, pCallableShaderBindingTable, buffer, offset);
}

#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
VKAPI_ATTR VkResult VKAPI_CALL GetDeviceAccelerationStructureCompatibilityKHR(
    VkDevice                                    device,
    const VkAccelerationStructureVersionKHR*    version) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetDeviceAccelerationStructureCompatibilityKHR(device, version);
}

#endif // VK_ENABLE_BETA_EXTENSIONS
// GPA helpers for extensions
bool extension_instance_gpa(struct loader_instance *ptr_instance, const char *name, void **addr) {
    *addr = NULL;


    // ---- VK_KHR_get_physical_device_properties2 extension commands
    if (!strcmp("vkGetPhysicalDeviceFeatures2KHR", name)) {
        *addr = (ptr_instance->enabled_known_extensions.khr_get_physical_device_properties2 == 1)
                     ? (void *)vkGetPhysicalDeviceFeatures2
                     : NULL;
        return true;
    }
    if (!strcmp("vkGetPhysicalDeviceProperties2KHR", name)) {
        *addr = (ptr_instance->enabled_known_extensions.khr_get_physical_device_properties2 == 1)
                     ? (void *)vkGetPhysicalDeviceProperties2
                     : NULL;
        return true;
    }
    if (!strcmp("vkGetPhysicalDeviceFormatProperties2KHR", name)) {
        *addr = (ptr_instance->enabled_known_extensions.khr_get_physical_device_properties2 == 1)
                     ? (void *)vkGetPhysicalDeviceFormatProperties2
                     : NULL;
        return true;
    }
    if (!strcmp("vkGetPhysicalDeviceImageFormatProperties2KHR", name)) {
        *addr = (ptr_instance->enabled_known_extensions.khr_get_physical_device_properties2 == 1)
                     ? (void *)vkGetPhysicalDeviceImageFormatProperties2
                     : NULL;
        return true;
    }
    if (!strcmp("vkGetPhysicalDeviceQueueFamilyProperties2KHR", name)) {
        *addr = (ptr_instance->enabled_known_extensions.khr_get_physical_device_properties2 == 1)
                     ? (void *)vkGetPhysicalDeviceQueueFamilyProperties2
                     : NULL;
        return true;
    }
    if (!strcmp("vkGetPhysicalDeviceMemoryProperties2KHR", name)) {
        *addr = (ptr_instance->enabled_known_extensions.khr_get_physical_device_properties2 == 1)
                     ? (void *)vkGetPhysicalDeviceMemoryProperties2
                     : NULL;
        return true;
    }
    if (!strcmp("vkGetPhysicalDeviceSparseImageFormatProperties2KHR", name)) {
        *addr = (ptr_instance->enabled_known_extensions.khr_get_physical_device_properties2 == 1)
                     ? (void *)vkGetPhysicalDeviceSparseImageFormatProperties2
                     : NULL;
        return true;
    }

    // ---- VK_KHR_device_group extension commands
    if (!strcmp("vkGetDeviceGroupPeerMemoryFeaturesKHR", name)) {
        *addr = (void *)GetDeviceGroupPeerMemoryFeaturesKHR;
        return true;
    }
    if (!strcmp("vkCmdSetDeviceMaskKHR", name)) {
        *addr = (void *)CmdSetDeviceMaskKHR;
        return true;
    }
    if (!strcmp("vkCmdDispatchBaseKHR", name)) {
        *addr = (void *)CmdDispatchBaseKHR;
        return true;
    }

    // ---- VK_KHR_maintenance1 extension commands
    if (!strcmp("vkTrimCommandPoolKHR", name)) {
        *addr = (void *)TrimCommandPoolKHR;
        return true;
    }

    // ---- VK_KHR_device_group_creation extension commands
    if (!strcmp("vkEnumeratePhysicalDeviceGroupsKHR", name)) {
        *addr = (ptr_instance->enabled_known_extensions.khr_device_group_creation == 1)
                     ? (void *)vkEnumeratePhysicalDeviceGroups
                     : NULL;
        return true;
    }

    // ---- VK_KHR_external_memory_capabilities extension commands
    if (!strcmp("vkGetPhysicalDeviceExternalBufferPropertiesKHR", name)) {
        *addr = (ptr_instance->enabled_known_extensions.khr_external_memory_capabilities == 1)
                     ? (void *)vkGetPhysicalDeviceExternalBufferProperties
                     : NULL;
        return true;
    }

    // ---- VK_KHR_external_memory_win32 extension commands
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (!strcmp("vkGetMemoryWin32HandleKHR", name)) {
        *addr = (void *)GetMemoryWin32HandleKHR;
        return true;
    }
#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (!strcmp("vkGetMemoryWin32HandlePropertiesKHR", name)) {
        *addr = (void *)GetMemoryWin32HandlePropertiesKHR;
        return true;
    }
#endif // VK_USE_PLATFORM_WIN32_KHR

    // ---- VK_KHR_external_memory_fd extension commands
    if (!strcmp("vkGetMemoryFdKHR", name)) {
        *addr = (void *)GetMemoryFdKHR;
        return true;
    }
    if (!strcmp("vkGetMemoryFdPropertiesKHR", name)) {
        *addr = (void *)GetMemoryFdPropertiesKHR;
        return true;
    }

    // ---- VK_KHR_external_semaphore_capabilities extension commands
    if (!strcmp("vkGetPhysicalDeviceExternalSemaphorePropertiesKHR", name)) {
        *addr = (ptr_instance->enabled_known_extensions.khr_external_semaphore_capabilities == 1)
                     ? (void *)vkGetPhysicalDeviceExternalSemaphoreProperties
                     : NULL;
        return true;
    }

    // ---- VK_KHR_external_semaphore_win32 extension commands
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (!strcmp("vkImportSemaphoreWin32HandleKHR", name)) {
        *addr = (void *)ImportSemaphoreWin32HandleKHR;
        return true;
    }
#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (!strcmp("vkGetSemaphoreWin32HandleKHR", name)) {
        *addr = (void *)GetSemaphoreWin32HandleKHR;
        return true;
    }
#endif // VK_USE_PLATFORM_WIN32_KHR

    // ---- VK_KHR_external_semaphore_fd extension commands
    if (!strcmp("vkImportSemaphoreFdKHR", name)) {
        *addr = (void *)ImportSemaphoreFdKHR;
        return true;
    }
    if (!strcmp("vkGetSemaphoreFdKHR", name)) {
        *addr = (void *)GetSemaphoreFdKHR;
        return true;
    }

    // ---- VK_KHR_push_descriptor extension commands
    if (!strcmp("vkCmdPushDescriptorSetKHR", name)) {
        *addr = (void *)CmdPushDescriptorSetKHR;
        return true;
    }
    if (!strcmp("vkCmdPushDescriptorSetWithTemplateKHR", name)) {
        *addr = (void *)CmdPushDescriptorSetWithTemplateKHR;
        return true;
    }

    // ---- VK_KHR_descriptor_update_template extension commands
    if (!strcmp("vkCreateDescriptorUpdateTemplateKHR", name)) {
        *addr = (void *)CreateDescriptorUpdateTemplateKHR;
        return true;
    }
    if (!strcmp("vkDestroyDescriptorUpdateTemplateKHR", name)) {
        *addr = (void *)DestroyDescriptorUpdateTemplateKHR;
        return true;
    }
    if (!strcmp("vkUpdateDescriptorSetWithTemplateKHR", name)) {
        *addr = (void *)UpdateDescriptorSetWithTemplateKHR;
        return true;
    }

    // ---- VK_KHR_create_renderpass2 extension commands
    if (!strcmp("vkCreateRenderPass2KHR", name)) {
        *addr = (void *)CreateRenderPass2KHR;
        return true;
    }
    if (!strcmp("vkCmdBeginRenderPass2KHR", name)) {
        *addr = (void *)CmdBeginRenderPass2KHR;
        return true;
    }
    if (!strcmp("vkCmdNextSubpass2KHR", name)) {
        *addr = (void *)CmdNextSubpass2KHR;
        return true;
    }
    if (!strcmp("vkCmdEndRenderPass2KHR", name)) {
        *addr = (void *)CmdEndRenderPass2KHR;
        return true;
    }

    // ---- VK_KHR_shared_presentable_image extension commands
    if (!strcmp("vkGetSwapchainStatusKHR", name)) {
        *addr = (void *)GetSwapchainStatusKHR;
        return true;
    }

    // ---- VK_KHR_external_fence_capabilities extension commands
    if (!strcmp("vkGetPhysicalDeviceExternalFencePropertiesKHR", name)) {
        *addr = (ptr_instance->enabled_known_extensions.khr_external_fence_capabilities == 1)
                     ? (void *)vkGetPhysicalDeviceExternalFenceProperties
                     : NULL;
        return true;
    }

    // ---- VK_KHR_external_fence_win32 extension commands
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (!strcmp("vkImportFenceWin32HandleKHR", name)) {
        *addr = (void *)ImportFenceWin32HandleKHR;
        return true;
    }
#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (!strcmp("vkGetFenceWin32HandleKHR", name)) {
        *addr = (void *)GetFenceWin32HandleKHR;
        return true;
    }
#endif // VK_USE_PLATFORM_WIN32_KHR

    // ---- VK_KHR_external_fence_fd extension commands
    if (!strcmp("vkImportFenceFdKHR", name)) {
        *addr = (void *)ImportFenceFdKHR;
        return true;
    }
    if (!strcmp("vkGetFenceFdKHR", name)) {
        *addr = (void *)GetFenceFdKHR;
        return true;
    }

    // ---- VK_KHR_performance_query extension commands
    if (!strcmp("vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR", name)) {
        *addr = (void *)EnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR;
        return true;
    }
    if (!strcmp("vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR", name)) {
        *addr = (void *)GetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR;
        return true;
    }
    if (!strcmp("vkAcquireProfilingLockKHR", name)) {
        *addr = (void *)AcquireProfilingLockKHR;
        return true;
    }
    if (!strcmp("vkReleaseProfilingLockKHR", name)) {
        *addr = (void *)ReleaseProfilingLockKHR;
        return true;
    }

    // ---- VK_KHR_get_memory_requirements2 extension commands
    if (!strcmp("vkGetImageMemoryRequirements2KHR", name)) {
        *addr = (void *)GetImageMemoryRequirements2KHR;
        return true;
    }
    if (!strcmp("vkGetBufferMemoryRequirements2KHR", name)) {
        *addr = (void *)GetBufferMemoryRequirements2KHR;
        return true;
    }
    if (!strcmp("vkGetImageSparseMemoryRequirements2KHR", name)) {
        *addr = (void *)GetImageSparseMemoryRequirements2KHR;
        return true;
    }

    // ---- VK_KHR_sampler_ycbcr_conversion extension commands
    if (!strcmp("vkCreateSamplerYcbcrConversionKHR", name)) {
        *addr = (void *)CreateSamplerYcbcrConversionKHR;
        return true;
    }
    if (!strcmp("vkDestroySamplerYcbcrConversionKHR", name)) {
        *addr = (void *)DestroySamplerYcbcrConversionKHR;
        return true;
    }

    // ---- VK_KHR_bind_memory2 extension commands
    if (!strcmp("vkBindBufferMemory2KHR", name)) {
        *addr = (void *)BindBufferMemory2KHR;
        return true;
    }
    if (!strcmp("vkBindImageMemory2KHR", name)) {
        *addr = (void *)BindImageMemory2KHR;
        return true;
    }

    // ---- VK_KHR_maintenance3 extension commands
    if (!strcmp("vkGetDescriptorSetLayoutSupportKHR", name)) {
        *addr = (void *)GetDescriptorSetLayoutSupportKHR;
        return true;
    }

    // ---- VK_KHR_draw_indirect_count extension commands
    if (!strcmp("vkCmdDrawIndirectCountKHR", name)) {
        *addr = (void *)CmdDrawIndirectCountKHR;
        return true;
    }
    if (!strcmp("vkCmdDrawIndexedIndirectCountKHR", name)) {
        *addr = (void *)CmdDrawIndexedIndirectCountKHR;
        return true;
    }

    // ---- VK_KHR_timeline_semaphore extension commands
    if (!strcmp("vkGetSemaphoreCounterValueKHR", name)) {
        *addr = (void *)GetSemaphoreCounterValueKHR;
        return true;
    }
    if (!strcmp("vkWaitSemaphoresKHR", name)) {
        *addr = (void *)WaitSemaphoresKHR;
        return true;
    }
    if (!strcmp("vkSignalSemaphoreKHR", name)) {
        *addr = (void *)SignalSemaphoreKHR;
        return true;
    }

    // ---- VK_KHR_buffer_device_address extension commands
    if (!strcmp("vkGetBufferDeviceAddressKHR", name)) {
        *addr = (void *)GetBufferDeviceAddressKHR;
        return true;
    }
    if (!strcmp("vkGetBufferOpaqueCaptureAddressKHR", name)) {
        *addr = (void *)GetBufferOpaqueCaptureAddressKHR;
        return true;
    }
    if (!strcmp("vkGetDeviceMemoryOpaqueCaptureAddressKHR", name)) {
        *addr = (void *)GetDeviceMemoryOpaqueCaptureAddressKHR;
        return true;
    }

    // ---- VK_KHR_deferred_host_operations extension commands
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp("vkCreateDeferredOperationKHR", name)) {
        *addr = (void *)CreateDeferredOperationKHR;
        return true;
    }
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp("vkDestroyDeferredOperationKHR", name)) {
        *addr = (void *)DestroyDeferredOperationKHR;
        return true;
    }
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp("vkGetDeferredOperationMaxConcurrencyKHR", name)) {
        *addr = (void *)GetDeferredOperationMaxConcurrencyKHR;
        return true;
    }
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp("vkGetDeferredOperationResultKHR", name)) {
        *addr = (void *)GetDeferredOperationResultKHR;
        return true;
    }
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp("vkDeferredOperationJoinKHR", name)) {
        *addr = (void *)DeferredOperationJoinKHR;
        return true;
    }
#endif // VK_ENABLE_BETA_EXTENSIONS

    // ---- VK_KHR_pipeline_executable_properties extension commands
    if (!strcmp("vkGetPipelineExecutablePropertiesKHR", name)) {
        *addr = (void *)GetPipelineExecutablePropertiesKHR;
        return true;
    }
    if (!strcmp("vkGetPipelineExecutableStatisticsKHR", name)) {
        *addr = (void *)GetPipelineExecutableStatisticsKHR;
        return true;
    }
    if (!strcmp("vkGetPipelineExecutableInternalRepresentationsKHR", name)) {
        *addr = (void *)GetPipelineExecutableInternalRepresentationsKHR;
        return true;
    }

    // ---- VK_KHR_copy_commands2 extension commands
    if (!strcmp("vkCmdCopyBuffer2KHR", name)) {
        *addr = (void *)CmdCopyBuffer2KHR;
        return true;
    }
    if (!strcmp("vkCmdCopyImage2KHR", name)) {
        *addr = (void *)CmdCopyImage2KHR;
        return true;
    }
    if (!strcmp("vkCmdCopyBufferToImage2KHR", name)) {
        *addr = (void *)CmdCopyBufferToImage2KHR;
        return true;
    }
    if (!strcmp("vkCmdCopyImageToBuffer2KHR", name)) {
        *addr = (void *)CmdCopyImageToBuffer2KHR;
        return true;
    }
    if (!strcmp("vkCmdBlitImage2KHR", name)) {
        *addr = (void *)CmdBlitImage2KHR;
        return true;
    }
    if (!strcmp("vkCmdResolveImage2KHR", name)) {
        *addr = (void *)CmdResolveImage2KHR;
        return true;
    }

    // ---- VK_EXT_debug_marker extension commands
    if (!strcmp("vkDebugMarkerSetObjectTagEXT", name)) {
        *addr = (void *)DebugMarkerSetObjectTagEXT;
        return true;
    }
    if (!strcmp("vkDebugMarkerSetObjectNameEXT", name)) {
        *addr = (void *)DebugMarkerSetObjectNameEXT;
        return true;
    }
    if (!strcmp("vkCmdDebugMarkerBeginEXT", name)) {
        *addr = (void *)CmdDebugMarkerBeginEXT;
        return true;
    }
    if (!strcmp("vkCmdDebugMarkerEndEXT", name)) {
        *addr = (void *)CmdDebugMarkerEndEXT;
        return true;
    }
    if (!strcmp("vkCmdDebugMarkerInsertEXT", name)) {
        *addr = (void *)CmdDebugMarkerInsertEXT;
        return true;
    }

    // ---- VK_EXT_transform_feedback extension commands
    if (!strcmp("vkCmdBindTransformFeedbackBuffersEXT", name)) {
        *addr = (void *)CmdBindTransformFeedbackBuffersEXT;
        return true;
    }
    if (!strcmp("vkCmdBeginTransformFeedbackEXT", name)) {
        *addr = (void *)CmdBeginTransformFeedbackEXT;
        return true;
    }
    if (!strcmp("vkCmdEndTransformFeedbackEXT", name)) {
        *addr = (void *)CmdEndTransformFeedbackEXT;
        return true;
    }
    if (!strcmp("vkCmdBeginQueryIndexedEXT", name)) {
        *addr = (void *)CmdBeginQueryIndexedEXT;
        return true;
    }
    if (!strcmp("vkCmdEndQueryIndexedEXT", name)) {
        *addr = (void *)CmdEndQueryIndexedEXT;
        return true;
    }
    if (!strcmp("vkCmdDrawIndirectByteCountEXT", name)) {
        *addr = (void *)CmdDrawIndirectByteCountEXT;
        return true;
    }

    // ---- VK_NVX_image_view_handle extension commands
    if (!strcmp("vkGetImageViewHandleNVX", name)) {
        *addr = (void *)GetImageViewHandleNVX;
        return true;
    }
    if (!strcmp("vkGetImageViewAddressNVX", name)) {
        *addr = (void *)GetImageViewAddressNVX;
        return true;
    }

    // ---- VK_AMD_draw_indirect_count extension commands
    if (!strcmp("vkCmdDrawIndirectCountAMD", name)) {
        *addr = (void *)CmdDrawIndirectCountAMD;
        return true;
    }
    if (!strcmp("vkCmdDrawIndexedIndirectCountAMD", name)) {
        *addr = (void *)CmdDrawIndexedIndirectCountAMD;
        return true;
    }

    // ---- VK_AMD_shader_info extension commands
    if (!strcmp("vkGetShaderInfoAMD", name)) {
        *addr = (void *)GetShaderInfoAMD;
        return true;
    }

    // ---- VK_GGP_stream_descriptor_surface extension commands
#ifdef VK_USE_PLATFORM_GGP
    if (!strcmp("vkCreateStreamDescriptorSurfaceGGP", name)) {
        *addr = (ptr_instance->enabled_known_extensions.ggp_stream_descriptor_surface == 1)
                     ? (void *)CreateStreamDescriptorSurfaceGGP
                     : NULL;
        return true;
    }
#endif // VK_USE_PLATFORM_GGP

    // ---- VK_NV_external_memory_capabilities extension commands
    if (!strcmp("vkGetPhysicalDeviceExternalImageFormatPropertiesNV", name)) {
        *addr = (ptr_instance->enabled_known_extensions.nv_external_memory_capabilities == 1)
                     ? (void *)GetPhysicalDeviceExternalImageFormatPropertiesNV
                     : NULL;
        return true;
    }

    // ---- VK_NV_external_memory_win32 extension commands
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (!strcmp("vkGetMemoryWin32HandleNV", name)) {
        *addr = (void *)GetMemoryWin32HandleNV;
        return true;
    }
#endif // VK_USE_PLATFORM_WIN32_KHR

    // ---- VK_NN_vi_surface extension commands
#ifdef VK_USE_PLATFORM_VI_NN
    if (!strcmp("vkCreateViSurfaceNN", name)) {
        *addr = (ptr_instance->enabled_known_extensions.nn_vi_surface == 1)
                     ? (void *)CreateViSurfaceNN
                     : NULL;
        return true;
    }
#endif // VK_USE_PLATFORM_VI_NN

    // ---- VK_EXT_conditional_rendering extension commands
    if (!strcmp("vkCmdBeginConditionalRenderingEXT", name)) {
        *addr = (void *)CmdBeginConditionalRenderingEXT;
        return true;
    }
    if (!strcmp("vkCmdEndConditionalRenderingEXT", name)) {
        *addr = (void *)CmdEndConditionalRenderingEXT;
        return true;
    }

    // ---- VK_NV_clip_space_w_scaling extension commands
    if (!strcmp("vkCmdSetViewportWScalingNV", name)) {
        *addr = (void *)CmdSetViewportWScalingNV;
        return true;
    }

    // ---- VK_EXT_direct_mode_display extension commands
    if (!strcmp("vkReleaseDisplayEXT", name)) {
        *addr = (ptr_instance->enabled_known_extensions.ext_direct_mode_display == 1)
                     ? (void *)ReleaseDisplayEXT
                     : NULL;
        return true;
    }

    // ---- VK_EXT_acquire_xlib_display extension commands
#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT
    if (!strcmp("vkAcquireXlibDisplayEXT", name)) {
        *addr = (ptr_instance->enabled_known_extensions.ext_acquire_xlib_display == 1)
                     ? (void *)AcquireXlibDisplayEXT
                     : NULL;
        return true;
    }
#endif // VK_USE_PLATFORM_XLIB_XRANDR_EXT
#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT
    if (!strcmp("vkGetRandROutputDisplayEXT", name)) {
        *addr = (ptr_instance->enabled_known_extensions.ext_acquire_xlib_display == 1)
                     ? (void *)GetRandROutputDisplayEXT
                     : NULL;
        return true;
    }
#endif // VK_USE_PLATFORM_XLIB_XRANDR_EXT

    // ---- VK_EXT_display_surface_counter extension commands
    if (!strcmp("vkGetPhysicalDeviceSurfaceCapabilities2EXT", name)) {
        *addr = (ptr_instance->enabled_known_extensions.ext_display_surface_counter == 1)
                     ? (void *)GetPhysicalDeviceSurfaceCapabilities2EXT
                     : NULL;
        return true;
    }

    // ---- VK_EXT_display_control extension commands
    if (!strcmp("vkDisplayPowerControlEXT", name)) {
        *addr = (void *)DisplayPowerControlEXT;
        return true;
    }
    if (!strcmp("vkRegisterDeviceEventEXT", name)) {
        *addr = (void *)RegisterDeviceEventEXT;
        return true;
    }
    if (!strcmp("vkRegisterDisplayEventEXT", name)) {
        *addr = (void *)RegisterDisplayEventEXT;
        return true;
    }
    if (!strcmp("vkGetSwapchainCounterEXT", name)) {
        *addr = (void *)GetSwapchainCounterEXT;
        return true;
    }

    // ---- VK_GOOGLE_display_timing extension commands
    if (!strcmp("vkGetRefreshCycleDurationGOOGLE", name)) {
        *addr = (void *)GetRefreshCycleDurationGOOGLE;
        return true;
    }
    if (!strcmp("vkGetPastPresentationTimingGOOGLE", name)) {
        *addr = (void *)GetPastPresentationTimingGOOGLE;
        return true;
    }

    // ---- VK_EXT_discard_rectangles extension commands
    if (!strcmp("vkCmdSetDiscardRectangleEXT", name)) {
        *addr = (void *)CmdSetDiscardRectangleEXT;
        return true;
    }

    // ---- VK_EXT_hdr_metadata extension commands
    if (!strcmp("vkSetHdrMetadataEXT", name)) {
        *addr = (void *)SetHdrMetadataEXT;
        return true;
    }

    // ---- VK_EXT_debug_utils extension commands
    if (!strcmp("vkSetDebugUtilsObjectNameEXT", name)) {
        *addr = (ptr_instance->enabled_known_extensions.ext_debug_utils == 1)
                     ? (void *)SetDebugUtilsObjectNameEXT
                     : NULL;
        return true;
    }
    if (!strcmp("vkSetDebugUtilsObjectTagEXT", name)) {
        *addr = (ptr_instance->enabled_known_extensions.ext_debug_utils == 1)
                     ? (void *)SetDebugUtilsObjectTagEXT
                     : NULL;
        return true;
    }
    if (!strcmp("vkQueueBeginDebugUtilsLabelEXT", name)) {
        *addr = (ptr_instance->enabled_known_extensions.ext_debug_utils == 1)
                     ? (void *)QueueBeginDebugUtilsLabelEXT
                     : NULL;
        return true;
    }
    if (!strcmp("vkQueueEndDebugUtilsLabelEXT", name)) {
        *addr = (ptr_instance->enabled_known_extensions.ext_debug_utils == 1)
                     ? (void *)QueueEndDebugUtilsLabelEXT
                     : NULL;
        return true;
    }
    if (!strcmp("vkQueueInsertDebugUtilsLabelEXT", name)) {
        *addr = (ptr_instance->enabled_known_extensions.ext_debug_utils == 1)
                     ? (void *)QueueInsertDebugUtilsLabelEXT
                     : NULL;
        return true;
    }
    if (!strcmp("vkCmdBeginDebugUtilsLabelEXT", name)) {
        *addr = (ptr_instance->enabled_known_extensions.ext_debug_utils == 1)
                     ? (void *)CmdBeginDebugUtilsLabelEXT
                     : NULL;
        return true;
    }
    if (!strcmp("vkCmdEndDebugUtilsLabelEXT", name)) {
        *addr = (ptr_instance->enabled_known_extensions.ext_debug_utils == 1)
                     ? (void *)CmdEndDebugUtilsLabelEXT
                     : NULL;
        return true;
    }
    if (!strcmp("vkCmdInsertDebugUtilsLabelEXT", name)) {
        *addr = (ptr_instance->enabled_known_extensions.ext_debug_utils == 1)
                     ? (void *)CmdInsertDebugUtilsLabelEXT
                     : NULL;
        return true;
    }

    // ---- VK_ANDROID_external_memory_android_hardware_buffer extension commands
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    if (!strcmp("vkGetAndroidHardwareBufferPropertiesANDROID", name)) {
        *addr = (void *)GetAndroidHardwareBufferPropertiesANDROID;
        return true;
    }
#endif // VK_USE_PLATFORM_ANDROID_KHR
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    if (!strcmp("vkGetMemoryAndroidHardwareBufferANDROID", name)) {
        *addr = (void *)GetMemoryAndroidHardwareBufferANDROID;
        return true;
    }
#endif // VK_USE_PLATFORM_ANDROID_KHR

    // ---- VK_EXT_sample_locations extension commands
    if (!strcmp("vkCmdSetSampleLocationsEXT", name)) {
        *addr = (void *)CmdSetSampleLocationsEXT;
        return true;
    }
    if (!strcmp("vkGetPhysicalDeviceMultisamplePropertiesEXT", name)) {
        *addr = (void *)GetPhysicalDeviceMultisamplePropertiesEXT;
        return true;
    }

    // ---- VK_EXT_image_drm_format_modifier extension commands
    if (!strcmp("vkGetImageDrmFormatModifierPropertiesEXT", name)) {
        *addr = (void *)GetImageDrmFormatModifierPropertiesEXT;
        return true;
    }

    // ---- VK_EXT_validation_cache extension commands
    if (!strcmp("vkCreateValidationCacheEXT", name)) {
        *addr = (void *)CreateValidationCacheEXT;
        return true;
    }
    if (!strcmp("vkDestroyValidationCacheEXT", name)) {
        *addr = (void *)DestroyValidationCacheEXT;
        return true;
    }
    if (!strcmp("vkMergeValidationCachesEXT", name)) {
        *addr = (void *)MergeValidationCachesEXT;
        return true;
    }
    if (!strcmp("vkGetValidationCacheDataEXT", name)) {
        *addr = (void *)GetValidationCacheDataEXT;
        return true;
    }

    // ---- VK_NV_shading_rate_image extension commands
    if (!strcmp("vkCmdBindShadingRateImageNV", name)) {
        *addr = (void *)CmdBindShadingRateImageNV;
        return true;
    }
    if (!strcmp("vkCmdSetViewportShadingRatePaletteNV", name)) {
        *addr = (void *)CmdSetViewportShadingRatePaletteNV;
        return true;
    }
    if (!strcmp("vkCmdSetCoarseSampleOrderNV", name)) {
        *addr = (void *)CmdSetCoarseSampleOrderNV;
        return true;
    }

    // ---- VK_NV_ray_tracing extension commands
    if (!strcmp("vkCreateAccelerationStructureNV", name)) {
        *addr = (void *)CreateAccelerationStructureNV;
        return true;
    }
    if (!strcmp("vkDestroyAccelerationStructureKHR", name)) {
        *addr = (void *)DestroyAccelerationStructureKHR;
        return true;
    }
    if (!strcmp("vkDestroyAccelerationStructureNV", name)) {
        *addr = (void *)DestroyAccelerationStructureNV;
        return true;
    }
    if (!strcmp("vkGetAccelerationStructureMemoryRequirementsNV", name)) {
        *addr = (void *)GetAccelerationStructureMemoryRequirementsNV;
        return true;
    }
    if (!strcmp("vkBindAccelerationStructureMemoryKHR", name)) {
        *addr = (void *)BindAccelerationStructureMemoryKHR;
        return true;
    }
    if (!strcmp("vkBindAccelerationStructureMemoryNV", name)) {
        *addr = (void *)BindAccelerationStructureMemoryNV;
        return true;
    }
    if (!strcmp("vkCmdBuildAccelerationStructureNV", name)) {
        *addr = (void *)CmdBuildAccelerationStructureNV;
        return true;
    }
    if (!strcmp("vkCmdCopyAccelerationStructureNV", name)) {
        *addr = (void *)CmdCopyAccelerationStructureNV;
        return true;
    }
    if (!strcmp("vkCmdTraceRaysNV", name)) {
        *addr = (void *)CmdTraceRaysNV;
        return true;
    }
    if (!strcmp("vkCreateRayTracingPipelinesNV", name)) {
        *addr = (void *)CreateRayTracingPipelinesNV;
        return true;
    }
    if (!strcmp("vkGetRayTracingShaderGroupHandlesKHR", name)) {
        *addr = (void *)GetRayTracingShaderGroupHandlesKHR;
        return true;
    }
    if (!strcmp("vkGetRayTracingShaderGroupHandlesNV", name)) {
        *addr = (void *)GetRayTracingShaderGroupHandlesNV;
        return true;
    }
    if (!strcmp("vkGetAccelerationStructureHandleNV", name)) {
        *addr = (void *)GetAccelerationStructureHandleNV;
        return true;
    }
    if (!strcmp("vkCmdWriteAccelerationStructuresPropertiesKHR", name)) {
        *addr = (void *)CmdWriteAccelerationStructuresPropertiesKHR;
        return true;
    }
    if (!strcmp("vkCmdWriteAccelerationStructuresPropertiesNV", name)) {
        *addr = (void *)CmdWriteAccelerationStructuresPropertiesNV;
        return true;
    }
    if (!strcmp("vkCompileDeferredNV", name)) {
        *addr = (void *)CompileDeferredNV;
        return true;
    }

    // ---- VK_EXT_external_memory_host extension commands
    if (!strcmp("vkGetMemoryHostPointerPropertiesEXT", name)) {
        *addr = (void *)GetMemoryHostPointerPropertiesEXT;
        return true;
    }

    // ---- VK_AMD_buffer_marker extension commands
    if (!strcmp("vkCmdWriteBufferMarkerAMD", name)) {
        *addr = (void *)CmdWriteBufferMarkerAMD;
        return true;
    }

    // ---- VK_EXT_calibrated_timestamps extension commands
    if (!strcmp("vkGetPhysicalDeviceCalibrateableTimeDomainsEXT", name)) {
        *addr = (void *)GetPhysicalDeviceCalibrateableTimeDomainsEXT;
        return true;
    }
    if (!strcmp("vkGetCalibratedTimestampsEXT", name)) {
        *addr = (void *)GetCalibratedTimestampsEXT;
        return true;
    }

    // ---- VK_NV_mesh_shader extension commands
    if (!strcmp("vkCmdDrawMeshTasksNV", name)) {
        *addr = (void *)CmdDrawMeshTasksNV;
        return true;
    }
    if (!strcmp("vkCmdDrawMeshTasksIndirectNV", name)) {
        *addr = (void *)CmdDrawMeshTasksIndirectNV;
        return true;
    }
    if (!strcmp("vkCmdDrawMeshTasksIndirectCountNV", name)) {
        *addr = (void *)CmdDrawMeshTasksIndirectCountNV;
        return true;
    }

    // ---- VK_NV_scissor_exclusive extension commands
    if (!strcmp("vkCmdSetExclusiveScissorNV", name)) {
        *addr = (void *)CmdSetExclusiveScissorNV;
        return true;
    }

    // ---- VK_NV_device_diagnostic_checkpoints extension commands
    if (!strcmp("vkCmdSetCheckpointNV", name)) {
        *addr = (void *)CmdSetCheckpointNV;
        return true;
    }
    if (!strcmp("vkGetQueueCheckpointDataNV", name)) {
        *addr = (void *)GetQueueCheckpointDataNV;
        return true;
    }

    // ---- VK_INTEL_performance_query extension commands
    if (!strcmp("vkInitializePerformanceApiINTEL", name)) {
        *addr = (void *)InitializePerformanceApiINTEL;
        return true;
    }
    if (!strcmp("vkUninitializePerformanceApiINTEL", name)) {
        *addr = (void *)UninitializePerformanceApiINTEL;
        return true;
    }
    if (!strcmp("vkCmdSetPerformanceMarkerINTEL", name)) {
        *addr = (void *)CmdSetPerformanceMarkerINTEL;
        return true;
    }
    if (!strcmp("vkCmdSetPerformanceStreamMarkerINTEL", name)) {
        *addr = (void *)CmdSetPerformanceStreamMarkerINTEL;
        return true;
    }
    if (!strcmp("vkCmdSetPerformanceOverrideINTEL", name)) {
        *addr = (void *)CmdSetPerformanceOverrideINTEL;
        return true;
    }
    if (!strcmp("vkAcquirePerformanceConfigurationINTEL", name)) {
        *addr = (void *)AcquirePerformanceConfigurationINTEL;
        return true;
    }
    if (!strcmp("vkReleasePerformanceConfigurationINTEL", name)) {
        *addr = (void *)ReleasePerformanceConfigurationINTEL;
        return true;
    }
    if (!strcmp("vkQueueSetPerformanceConfigurationINTEL", name)) {
        *addr = (void *)QueueSetPerformanceConfigurationINTEL;
        return true;
    }
    if (!strcmp("vkGetPerformanceParameterINTEL", name)) {
        *addr = (void *)GetPerformanceParameterINTEL;
        return true;
    }

    // ---- VK_AMD_display_native_hdr extension commands
    if (!strcmp("vkSetLocalDimmingAMD", name)) {
        *addr = (void *)SetLocalDimmingAMD;
        return true;
    }

    // ---- VK_FUCHSIA_imagepipe_surface extension commands
#ifdef VK_USE_PLATFORM_FUCHSIA
    if (!strcmp("vkCreateImagePipeSurfaceFUCHSIA", name)) {
        *addr = (ptr_instance->enabled_known_extensions.fuchsia_imagepipe_surface == 1)
                     ? (void *)CreateImagePipeSurfaceFUCHSIA
                     : NULL;
        return true;
    }
#endif // VK_USE_PLATFORM_FUCHSIA

    // ---- VK_EXT_buffer_device_address extension commands
    if (!strcmp("vkGetBufferDeviceAddressEXT", name)) {
        *addr = (void *)GetBufferDeviceAddressEXT;
        return true;
    }

    // ---- VK_EXT_tooling_info extension commands
    if (!strcmp("vkGetPhysicalDeviceToolPropertiesEXT", name)) {
        *addr = (void *)GetPhysicalDeviceToolPropertiesEXT;
        return true;
    }

    // ---- VK_NV_cooperative_matrix extension commands
    if (!strcmp("vkGetPhysicalDeviceCooperativeMatrixPropertiesNV", name)) {
        *addr = (void *)GetPhysicalDeviceCooperativeMatrixPropertiesNV;
        return true;
    }

    // ---- VK_NV_coverage_reduction_mode extension commands
    if (!strcmp("vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV", name)) {
        *addr = (void *)GetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV;
        return true;
    }

    // ---- VK_EXT_full_screen_exclusive extension commands
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (!strcmp("vkGetPhysicalDeviceSurfacePresentModes2EXT", name)) {
        *addr = (void *)GetPhysicalDeviceSurfacePresentModes2EXT;
        return true;
    }
#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (!strcmp("vkAcquireFullScreenExclusiveModeEXT", name)) {
        *addr = (void *)AcquireFullScreenExclusiveModeEXT;
        return true;
    }
#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (!strcmp("vkReleaseFullScreenExclusiveModeEXT", name)) {
        *addr = (void *)ReleaseFullScreenExclusiveModeEXT;
        return true;
    }
#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
    if (!strcmp("vkGetDeviceGroupSurfacePresentModes2EXT", name)) {
        *addr = (void *)GetDeviceGroupSurfacePresentModes2EXT;
        return true;
    }
#endif // VK_USE_PLATFORM_WIN32_KHR

    // ---- VK_EXT_line_rasterization extension commands
    if (!strcmp("vkCmdSetLineStippleEXT", name)) {
        *addr = (void *)CmdSetLineStippleEXT;
        return true;
    }

    // ---- VK_EXT_host_query_reset extension commands
    if (!strcmp("vkResetQueryPoolEXT", name)) {
        *addr = (void *)ResetQueryPoolEXT;
        return true;
    }

    // ---- VK_EXT_extended_dynamic_state extension commands
    if (!strcmp("vkCmdSetCullModeEXT", name)) {
        *addr = (void *)CmdSetCullModeEXT;
        return true;
    }
    if (!strcmp("vkCmdSetFrontFaceEXT", name)) {
        *addr = (void *)CmdSetFrontFaceEXT;
        return true;
    }
    if (!strcmp("vkCmdSetPrimitiveTopologyEXT", name)) {
        *addr = (void *)CmdSetPrimitiveTopologyEXT;
        return true;
    }
    if (!strcmp("vkCmdSetViewportWithCountEXT", name)) {
        *addr = (void *)CmdSetViewportWithCountEXT;
        return true;
    }
    if (!strcmp("vkCmdSetScissorWithCountEXT", name)) {
        *addr = (void *)CmdSetScissorWithCountEXT;
        return true;
    }
    if (!strcmp("vkCmdBindVertexBuffers2EXT", name)) {
        *addr = (void *)CmdBindVertexBuffers2EXT;
        return true;
    }
    if (!strcmp("vkCmdSetDepthTestEnableEXT", name)) {
        *addr = (void *)CmdSetDepthTestEnableEXT;
        return true;
    }
    if (!strcmp("vkCmdSetDepthWriteEnableEXT", name)) {
        *addr = (void *)CmdSetDepthWriteEnableEXT;
        return true;
    }
    if (!strcmp("vkCmdSetDepthCompareOpEXT", name)) {
        *addr = (void *)CmdSetDepthCompareOpEXT;
        return true;
    }
    if (!strcmp("vkCmdSetDepthBoundsTestEnableEXT", name)) {
        *addr = (void *)CmdSetDepthBoundsTestEnableEXT;
        return true;
    }
    if (!strcmp("vkCmdSetStencilTestEnableEXT", name)) {
        *addr = (void *)CmdSetStencilTestEnableEXT;
        return true;
    }
    if (!strcmp("vkCmdSetStencilOpEXT", name)) {
        *addr = (void *)CmdSetStencilOpEXT;
        return true;
    }

    // ---- VK_NV_device_generated_commands extension commands
    if (!strcmp("vkGetGeneratedCommandsMemoryRequirementsNV", name)) {
        *addr = (void *)GetGeneratedCommandsMemoryRequirementsNV;
        return true;
    }
    if (!strcmp("vkCmdPreprocessGeneratedCommandsNV", name)) {
        *addr = (void *)CmdPreprocessGeneratedCommandsNV;
        return true;
    }
    if (!strcmp("vkCmdExecuteGeneratedCommandsNV", name)) {
        *addr = (void *)CmdExecuteGeneratedCommandsNV;
        return true;
    }
    if (!strcmp("vkCmdBindPipelineShaderGroupNV", name)) {
        *addr = (void *)CmdBindPipelineShaderGroupNV;
        return true;
    }
    if (!strcmp("vkCreateIndirectCommandsLayoutNV", name)) {
        *addr = (void *)CreateIndirectCommandsLayoutNV;
        return true;
    }
    if (!strcmp("vkDestroyIndirectCommandsLayoutNV", name)) {
        *addr = (void *)DestroyIndirectCommandsLayoutNV;
        return true;
    }

    // ---- VK_EXT_private_data extension commands
    if (!strcmp("vkCreatePrivateDataSlotEXT", name)) {
        *addr = (void *)CreatePrivateDataSlotEXT;
        return true;
    }
    if (!strcmp("vkDestroyPrivateDataSlotEXT", name)) {
        *addr = (void *)DestroyPrivateDataSlotEXT;
        return true;
    }
    if (!strcmp("vkSetPrivateDataEXT", name)) {
        *addr = (void *)SetPrivateDataEXT;
        return true;
    }
    if (!strcmp("vkGetPrivateDataEXT", name)) {
        *addr = (void *)GetPrivateDataEXT;
        return true;
    }

    // ---- VK_KHR_ray_tracing extension commands
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp("vkCreateAccelerationStructureKHR", name)) {
        *addr = (void *)CreateAccelerationStructureKHR;
        return true;
    }
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp("vkGetAccelerationStructureMemoryRequirementsKHR", name)) {
        *addr = (void *)GetAccelerationStructureMemoryRequirementsKHR;
        return true;
    }
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp("vkCmdBuildAccelerationStructureKHR", name)) {
        *addr = (void *)CmdBuildAccelerationStructureKHR;
        return true;
    }
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp("vkCmdBuildAccelerationStructureIndirectKHR", name)) {
        *addr = (void *)CmdBuildAccelerationStructureIndirectKHR;
        return true;
    }
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp("vkBuildAccelerationStructureKHR", name)) {
        *addr = (void *)BuildAccelerationStructureKHR;
        return true;
    }
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp("vkCopyAccelerationStructureKHR", name)) {
        *addr = (void *)CopyAccelerationStructureKHR;
        return true;
    }
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp("vkCopyAccelerationStructureToMemoryKHR", name)) {
        *addr = (void *)CopyAccelerationStructureToMemoryKHR;
        return true;
    }
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp("vkCopyMemoryToAccelerationStructureKHR", name)) {
        *addr = (void *)CopyMemoryToAccelerationStructureKHR;
        return true;
    }
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp("vkWriteAccelerationStructuresPropertiesKHR", name)) {
        *addr = (void *)WriteAccelerationStructuresPropertiesKHR;
        return true;
    }
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp("vkCmdCopyAccelerationStructureKHR", name)) {
        *addr = (void *)CmdCopyAccelerationStructureKHR;
        return true;
    }
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp("vkCmdCopyAccelerationStructureToMemoryKHR", name)) {
        *addr = (void *)CmdCopyAccelerationStructureToMemoryKHR;
        return true;
    }
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp("vkCmdCopyMemoryToAccelerationStructureKHR", name)) {
        *addr = (void *)CmdCopyMemoryToAccelerationStructureKHR;
        return true;
    }
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp("vkCmdTraceRaysKHR", name)) {
        *addr = (void *)CmdTraceRaysKHR;
        return true;
    }
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp("vkCreateRayTracingPipelinesKHR", name)) {
        *addr = (void *)CreateRayTracingPipelinesKHR;
        return true;
    }
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp("vkGetAccelerationStructureDeviceAddressKHR", name)) {
        *addr = (void *)GetAccelerationStructureDeviceAddressKHR;
        return true;
    }
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp("vkGetRayTracingCaptureReplayShaderGroupHandlesKHR", name)) {
        *addr = (void *)GetRayTracingCaptureReplayShaderGroupHandlesKHR;
        return true;
    }
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp("vkCmdTraceRaysIndirectKHR", name)) {
        *addr = (void *)CmdTraceRaysIndirectKHR;
        return true;
    }
#endif // VK_ENABLE_BETA_EXTENSIONS
#ifdef VK_ENABLE_BETA_EXTENSIONS
    if (!strcmp("vkGetDeviceAccelerationStructureCompatibilityKHR", name)) {
        *addr = (void *)GetDeviceAccelerationStructureCompatibilityKHR;
        return true;
    }
#endif // VK_ENABLE_BETA_EXTENSIONS
    return false;
}

// A function that can be used to query enabled extensions during a vkCreateInstance call
void extensions_create_instance(struct loader_instance *ptr_instance, const VkInstanceCreateInfo *pCreateInfo) {
    for (uint32_t i = 0; i < pCreateInfo->enabledExtensionCount; i++) {

    // ---- VK_KHR_get_physical_device_properties2 extension commands
        if (0 == strcmp(pCreateInfo->ppEnabledExtensionNames[i], VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
            ptr_instance->enabled_known_extensions.khr_get_physical_device_properties2 = 1;

    // ---- VK_KHR_device_group_creation extension commands
        } else if (0 == strcmp(pCreateInfo->ppEnabledExtensionNames[i], VK_KHR_DEVICE_GROUP_CREATION_EXTENSION_NAME)) {
            ptr_instance->enabled_known_extensions.khr_device_group_creation = 1;

    // ---- VK_KHR_external_memory_capabilities extension commands
        } else if (0 == strcmp(pCreateInfo->ppEnabledExtensionNames[i], VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME)) {
            ptr_instance->enabled_known_extensions.khr_external_memory_capabilities = 1;

    // ---- VK_KHR_external_semaphore_capabilities extension commands
        } else if (0 == strcmp(pCreateInfo->ppEnabledExtensionNames[i], VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME)) {
            ptr_instance->enabled_known_extensions.khr_external_semaphore_capabilities = 1;

    // ---- VK_KHR_external_fence_capabilities extension commands
        } else if (0 == strcmp(pCreateInfo->ppEnabledExtensionNames[i], VK_KHR_EXTERNAL_FENCE_CAPABILITIES_EXTENSION_NAME)) {
            ptr_instance->enabled_known_extensions.khr_external_fence_capabilities = 1;

    // ---- VK_GGP_stream_descriptor_surface extension commands
#ifdef VK_USE_PLATFORM_GGP
        } else if (0 == strcmp(pCreateInfo->ppEnabledExtensionNames[i], VK_GGP_STREAM_DESCRIPTOR_SURFACE_EXTENSION_NAME)) {
            ptr_instance->enabled_known_extensions.ggp_stream_descriptor_surface = 1;
#endif // VK_USE_PLATFORM_GGP

    // ---- VK_NV_external_memory_capabilities extension commands
        } else if (0 == strcmp(pCreateInfo->ppEnabledExtensionNames[i], VK_NV_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME)) {
            ptr_instance->enabled_known_extensions.nv_external_memory_capabilities = 1;

    // ---- VK_NN_vi_surface extension commands
#ifdef VK_USE_PLATFORM_VI_NN
        } else if (0 == strcmp(pCreateInfo->ppEnabledExtensionNames[i], VK_NN_VI_SURFACE_EXTENSION_NAME)) {
            ptr_instance->enabled_known_extensions.nn_vi_surface = 1;
#endif // VK_USE_PLATFORM_VI_NN

    // ---- VK_EXT_direct_mode_display extension commands
        } else if (0 == strcmp(pCreateInfo->ppEnabledExtensionNames[i], VK_EXT_DIRECT_MODE_DISPLAY_EXTENSION_NAME)) {
            ptr_instance->enabled_known_extensions.ext_direct_mode_display = 1;

    // ---- VK_EXT_acquire_xlib_display extension commands
#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT
        } else if (0 == strcmp(pCreateInfo->ppEnabledExtensionNames[i], VK_EXT_ACQUIRE_XLIB_DISPLAY_EXTENSION_NAME)) {
            ptr_instance->enabled_known_extensions.ext_acquire_xlib_display = 1;
#endif // VK_USE_PLATFORM_XLIB_XRANDR_EXT

    // ---- VK_EXT_display_surface_counter extension commands
        } else if (0 == strcmp(pCreateInfo->ppEnabledExtensionNames[i], VK_EXT_DISPLAY_SURFACE_COUNTER_EXTENSION_NAME)) {
            ptr_instance->enabled_known_extensions.ext_display_surface_counter = 1;

    // ---- VK_EXT_debug_utils extension commands
        } else if (0 == strcmp(pCreateInfo->ppEnabledExtensionNames[i], VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
            ptr_instance->enabled_known_extensions.ext_debug_utils = 1;

    // ---- VK_FUCHSIA_imagepipe_surface extension commands
#ifdef VK_USE_PLATFORM_FUCHSIA
        } else if (0 == strcmp(pCreateInfo->ppEnabledExtensionNames[i], VK_FUCHSIA_IMAGEPIPE_SURFACE_EXTENSION_NAME)) {
            ptr_instance->enabled_known_extensions.fuchsia_imagepipe_surface = 1;
#endif // VK_USE_PLATFORM_FUCHSIA
        }
    }
}

// Some device commands still need a terminator because the loader needs to unwrap something about them.
// In many cases, the item needing unwrapping is a VkPhysicalDevice or VkSurfaceKHR object.  But there may be other items
// in the future.
PFN_vkVoidFunction get_extension_device_proc_terminator(struct loader_device *dev, const char *pName) {
    PFN_vkVoidFunction addr = NULL;

    // ---- VK_KHR_swapchain extension commands
    if (dev->extensions.khr_swapchain_enabled) {
        if(!strcmp(pName, "vkCreateSwapchainKHR")) {
            addr = (PFN_vkVoidFunction)terminator_CreateSwapchainKHR;
        } else if(!strcmp(pName, "vkGetDeviceGroupSurfacePresentModesKHR")) {
            addr = (PFN_vkVoidFunction)terminator_GetDeviceGroupSurfacePresentModesKHR;
        }
    }

    // ---- VK_KHR_display_swapchain extension commands
    if (dev->extensions.khr_display_swapchain_enabled) {
        if(!strcmp(pName, "vkCreateSharedSwapchainsKHR")) {
            addr = (PFN_vkVoidFunction)terminator_CreateSharedSwapchainsKHR;
        }
    }

    // ---- VK_EXT_debug_marker extension commands
    if (dev->extensions.ext_debug_marker_enabled) {
        if(!strcmp(pName, "vkDebugMarkerSetObjectTagEXT")) {
            addr = (PFN_vkVoidFunction)terminator_DebugMarkerSetObjectTagEXT;
        } else if(!strcmp(pName, "vkDebugMarkerSetObjectNameEXT")) {
            addr = (PFN_vkVoidFunction)terminator_DebugMarkerSetObjectNameEXT;
        }
    }

    // ---- VK_EXT_debug_utils extension commands
    if (dev->extensions.ext_debug_utils_enabled) {
        if(!strcmp(pName, "vkSetDebugUtilsObjectNameEXT")) {
            addr = (PFN_vkVoidFunction)terminator_SetDebugUtilsObjectNameEXT;
        } else if(!strcmp(pName, "vkSetDebugUtilsObjectTagEXT")) {
            addr = (PFN_vkVoidFunction)terminator_SetDebugUtilsObjectTagEXT;
        } else if(!strcmp(pName, "vkQueueBeginDebugUtilsLabelEXT")) {
            addr = (PFN_vkVoidFunction)terminator_QueueBeginDebugUtilsLabelEXT;
        } else if(!strcmp(pName, "vkQueueEndDebugUtilsLabelEXT")) {
            addr = (PFN_vkVoidFunction)terminator_QueueEndDebugUtilsLabelEXT;
        } else if(!strcmp(pName, "vkQueueInsertDebugUtilsLabelEXT")) {
            addr = (PFN_vkVoidFunction)terminator_QueueInsertDebugUtilsLabelEXT;
        } else if(!strcmp(pName, "vkCmdBeginDebugUtilsLabelEXT")) {
            addr = (PFN_vkVoidFunction)terminator_CmdBeginDebugUtilsLabelEXT;
        } else if(!strcmp(pName, "vkCmdEndDebugUtilsLabelEXT")) {
            addr = (PFN_vkVoidFunction)terminator_CmdEndDebugUtilsLabelEXT;
        } else if(!strcmp(pName, "vkCmdInsertDebugUtilsLabelEXT")) {
            addr = (PFN_vkVoidFunction)terminator_CmdInsertDebugUtilsLabelEXT;
        }
    }
#ifdef VK_USE_PLATFORM_WIN32_KHR

    // ---- VK_EXT_full_screen_exclusive extension commands
    if (dev->extensions.ext_full_screen_exclusive_enabled && dev->extensions.khr_device_group_enabled) {
        if(!strcmp(pName, "vkGetDeviceGroupSurfacePresentModes2EXT")) {
            addr = (PFN_vkVoidFunction)terminator_GetDeviceGroupSurfacePresentModes2EXT;
        }
    }
#endif // VK_ENABLE_BETA_EXTENSIONS
    return addr;
}

// This table contains the loader's instance dispatch table, which contains
// default functions if no instance layers are activated.  This contains
// pointers to "terminator functions".
const VkLayerInstanceDispatchTable instance_disp = {

    // ---- Core 1_0 commands
    .DestroyInstance = terminator_DestroyInstance,
    .EnumeratePhysicalDevices = terminator_EnumeratePhysicalDevices,
    .GetPhysicalDeviceFeatures = terminator_GetPhysicalDeviceFeatures,
    .GetPhysicalDeviceFormatProperties = terminator_GetPhysicalDeviceFormatProperties,
    .GetPhysicalDeviceImageFormatProperties = terminator_GetPhysicalDeviceImageFormatProperties,
    .GetPhysicalDeviceProperties = terminator_GetPhysicalDeviceProperties,
    .GetPhysicalDeviceQueueFamilyProperties = terminator_GetPhysicalDeviceQueueFamilyProperties,
    .GetPhysicalDeviceMemoryProperties = terminator_GetPhysicalDeviceMemoryProperties,
    .GetInstanceProcAddr = vkGetInstanceProcAddr,
    .EnumerateDeviceExtensionProperties = terminator_EnumerateDeviceExtensionProperties,
    .EnumerateDeviceLayerProperties = terminator_EnumerateDeviceLayerProperties,
    .GetPhysicalDeviceSparseImageFormatProperties = terminator_GetPhysicalDeviceSparseImageFormatProperties,

    // ---- Core 1_1 commands
    .EnumeratePhysicalDeviceGroups = terminator_EnumeratePhysicalDeviceGroups,
    .GetPhysicalDeviceFeatures2 = terminator_GetPhysicalDeviceFeatures2,
    .GetPhysicalDeviceProperties2 = terminator_GetPhysicalDeviceProperties2,
    .GetPhysicalDeviceFormatProperties2 = terminator_GetPhysicalDeviceFormatProperties2,
    .GetPhysicalDeviceImageFormatProperties2 = terminator_GetPhysicalDeviceImageFormatProperties2,
    .GetPhysicalDeviceQueueFamilyProperties2 = terminator_GetPhysicalDeviceQueueFamilyProperties2,
    .GetPhysicalDeviceMemoryProperties2 = terminator_GetPhysicalDeviceMemoryProperties2,
    .GetPhysicalDeviceSparseImageFormatProperties2 = terminator_GetPhysicalDeviceSparseImageFormatProperties2,
    .GetPhysicalDeviceExternalBufferProperties = terminator_GetPhysicalDeviceExternalBufferProperties,
    .GetPhysicalDeviceExternalFenceProperties = terminator_GetPhysicalDeviceExternalFenceProperties,
    .GetPhysicalDeviceExternalSemaphoreProperties = terminator_GetPhysicalDeviceExternalSemaphoreProperties,

    // ---- VK_KHR_surface extension commands
    .DestroySurfaceKHR = terminator_DestroySurfaceKHR,
    .GetPhysicalDeviceSurfaceSupportKHR = terminator_GetPhysicalDeviceSurfaceSupportKHR,
    .GetPhysicalDeviceSurfaceCapabilitiesKHR = terminator_GetPhysicalDeviceSurfaceCapabilitiesKHR,
    .GetPhysicalDeviceSurfaceFormatsKHR = terminator_GetPhysicalDeviceSurfaceFormatsKHR,
    .GetPhysicalDeviceSurfacePresentModesKHR = terminator_GetPhysicalDeviceSurfacePresentModesKHR,

    // ---- VK_KHR_swapchain extension commands
    .GetPhysicalDevicePresentRectanglesKHR = terminator_GetPhysicalDevicePresentRectanglesKHR,

    // ---- VK_KHR_display extension commands
    .GetPhysicalDeviceDisplayPropertiesKHR = terminator_GetPhysicalDeviceDisplayPropertiesKHR,
    .GetPhysicalDeviceDisplayPlanePropertiesKHR = terminator_GetPhysicalDeviceDisplayPlanePropertiesKHR,
    .GetDisplayPlaneSupportedDisplaysKHR = terminator_GetDisplayPlaneSupportedDisplaysKHR,
    .GetDisplayModePropertiesKHR = terminator_GetDisplayModePropertiesKHR,
    .CreateDisplayModeKHR = terminator_CreateDisplayModeKHR,
    .GetDisplayPlaneCapabilitiesKHR = terminator_GetDisplayPlaneCapabilitiesKHR,
    .CreateDisplayPlaneSurfaceKHR = terminator_CreateDisplayPlaneSurfaceKHR,

    // ---- VK_KHR_xlib_surface extension commands
#ifdef VK_USE_PLATFORM_XLIB_KHR
    .CreateXlibSurfaceKHR = terminator_CreateXlibSurfaceKHR,
#endif // VK_USE_PLATFORM_XLIB_KHR
#ifdef VK_USE_PLATFORM_XLIB_KHR
    .GetPhysicalDeviceXlibPresentationSupportKHR = terminator_GetPhysicalDeviceXlibPresentationSupportKHR,
#endif // VK_USE_PLATFORM_XLIB_KHR

    // ---- VK_KHR_xcb_surface extension commands
#ifdef VK_USE_PLATFORM_XCB_KHR
    .CreateXcbSurfaceKHR = terminator_CreateXcbSurfaceKHR,
#endif // VK_USE_PLATFORM_XCB_KHR
#ifdef VK_USE_PLATFORM_XCB_KHR
    .GetPhysicalDeviceXcbPresentationSupportKHR = terminator_GetPhysicalDeviceXcbPresentationSupportKHR,
#endif // VK_USE_PLATFORM_XCB_KHR

    // ---- VK_KHR_wayland_surface extension commands
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    .CreateWaylandSurfaceKHR = terminator_CreateWaylandSurfaceKHR,
#endif // VK_USE_PLATFORM_WAYLAND_KHR
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    .GetPhysicalDeviceWaylandPresentationSupportKHR = terminator_GetPhysicalDeviceWaylandPresentationSupportKHR,
#endif // VK_USE_PLATFORM_WAYLAND_KHR

    // ---- VK_KHR_android_surface extension commands
#ifdef VK_USE_PLATFORM_ANDROID_KHR
    .CreateAndroidSurfaceKHR = terminator_CreateAndroidSurfaceKHR,
#endif // VK_USE_PLATFORM_ANDROID_KHR

    // ---- VK_KHR_win32_surface extension commands
#ifdef VK_USE_PLATFORM_WIN32_KHR
    .CreateWin32SurfaceKHR = terminator_CreateWin32SurfaceKHR,
#endif // VK_USE_PLATFORM_WIN32_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
    .GetPhysicalDeviceWin32PresentationSupportKHR = terminator_GetPhysicalDeviceWin32PresentationSupportKHR,
#endif // VK_USE_PLATFORM_WIN32_KHR

    // ---- VK_KHR_get_physical_device_properties2 extension commands
    .GetPhysicalDeviceFeatures2KHR = terminator_GetPhysicalDeviceFeatures2,
    .GetPhysicalDeviceProperties2KHR = terminator_GetPhysicalDeviceProperties2,
    .GetPhysicalDeviceFormatProperties2KHR = terminator_GetPhysicalDeviceFormatProperties2,
    .GetPhysicalDeviceImageFormatProperties2KHR = terminator_GetPhysicalDeviceImageFormatProperties2,
    .GetPhysicalDeviceQueueFamilyProperties2KHR = terminator_GetPhysicalDeviceQueueFamilyProperties2,
    .GetPhysicalDeviceMemoryProperties2KHR = terminator_GetPhysicalDeviceMemoryProperties2,
    .GetPhysicalDeviceSparseImageFormatProperties2KHR = terminator_GetPhysicalDeviceSparseImageFormatProperties2,

    // ---- VK_KHR_device_group_creation extension commands
    .EnumeratePhysicalDeviceGroupsKHR = terminator_EnumeratePhysicalDeviceGroups,

    // ---- VK_KHR_external_memory_capabilities extension commands
    .GetPhysicalDeviceExternalBufferPropertiesKHR = terminator_GetPhysicalDeviceExternalBufferProperties,

    // ---- VK_KHR_external_semaphore_capabilities extension commands
    .GetPhysicalDeviceExternalSemaphorePropertiesKHR = terminator_GetPhysicalDeviceExternalSemaphoreProperties,

    // ---- VK_KHR_external_fence_capabilities extension commands
    .GetPhysicalDeviceExternalFencePropertiesKHR = terminator_GetPhysicalDeviceExternalFenceProperties,

    // ---- VK_KHR_performance_query extension commands
    .EnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR = terminator_EnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR,
    .GetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR = terminator_GetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR,

    // ---- VK_KHR_get_surface_capabilities2 extension commands
    .GetPhysicalDeviceSurfaceCapabilities2KHR = terminator_GetPhysicalDeviceSurfaceCapabilities2KHR,
    .GetPhysicalDeviceSurfaceFormats2KHR = terminator_GetPhysicalDeviceSurfaceFormats2KHR,

    // ---- VK_KHR_get_display_properties2 extension commands
    .GetPhysicalDeviceDisplayProperties2KHR = terminator_GetPhysicalDeviceDisplayProperties2KHR,
    .GetPhysicalDeviceDisplayPlaneProperties2KHR = terminator_GetPhysicalDeviceDisplayPlaneProperties2KHR,
    .GetDisplayModeProperties2KHR = terminator_GetDisplayModeProperties2KHR,
    .GetDisplayPlaneCapabilities2KHR = terminator_GetDisplayPlaneCapabilities2KHR,

    // ---- VK_EXT_debug_report extension commands
    .CreateDebugReportCallbackEXT = terminator_CreateDebugReportCallbackEXT,
    .DestroyDebugReportCallbackEXT = terminator_DestroyDebugReportCallbackEXT,
    .DebugReportMessageEXT = terminator_DebugReportMessageEXT,

    // ---- VK_GGP_stream_descriptor_surface extension commands
#ifdef VK_USE_PLATFORM_GGP
    .CreateStreamDescriptorSurfaceGGP = terminator_CreateStreamDescriptorSurfaceGGP,
#endif // VK_USE_PLATFORM_GGP

    // ---- VK_NV_external_memory_capabilities extension commands
    .GetPhysicalDeviceExternalImageFormatPropertiesNV = terminator_GetPhysicalDeviceExternalImageFormatPropertiesNV,

    // ---- VK_NN_vi_surface extension commands
#ifdef VK_USE_PLATFORM_VI_NN
    .CreateViSurfaceNN = terminator_CreateViSurfaceNN,
#endif // VK_USE_PLATFORM_VI_NN

    // ---- VK_EXT_direct_mode_display extension commands
    .ReleaseDisplayEXT = terminator_ReleaseDisplayEXT,

    // ---- VK_EXT_acquire_xlib_display extension commands
#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT
    .AcquireXlibDisplayEXT = terminator_AcquireXlibDisplayEXT,
#endif // VK_USE_PLATFORM_XLIB_XRANDR_EXT
#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT
    .GetRandROutputDisplayEXT = terminator_GetRandROutputDisplayEXT,
#endif // VK_USE_PLATFORM_XLIB_XRANDR_EXT

    // ---- VK_EXT_display_surface_counter extension commands
    .GetPhysicalDeviceSurfaceCapabilities2EXT = terminator_GetPhysicalDeviceSurfaceCapabilities2EXT,

    // ---- VK_MVK_ios_surface extension commands
#ifdef VK_USE_PLATFORM_IOS_MVK
    .CreateIOSSurfaceMVK = terminator_CreateIOSSurfaceMVK,
#endif // VK_USE_PLATFORM_IOS_MVK

    // ---- VK_MVK_macos_surface extension commands
#ifdef VK_USE_PLATFORM_MACOS_MVK
    .CreateMacOSSurfaceMVK = terminator_CreateMacOSSurfaceMVK,
#endif // VK_USE_PLATFORM_MACOS_MVK

    // ---- VK_EXT_debug_utils extension commands
    .CreateDebugUtilsMessengerEXT = terminator_CreateDebugUtilsMessengerEXT,
    .DestroyDebugUtilsMessengerEXT = terminator_DestroyDebugUtilsMessengerEXT,
    .SubmitDebugUtilsMessageEXT = terminator_SubmitDebugUtilsMessageEXT,

    // ---- VK_EXT_sample_locations extension commands
    .GetPhysicalDeviceMultisamplePropertiesEXT = terminator_GetPhysicalDeviceMultisamplePropertiesEXT,

    // ---- VK_EXT_calibrated_timestamps extension commands
    .GetPhysicalDeviceCalibrateableTimeDomainsEXT = terminator_GetPhysicalDeviceCalibrateableTimeDomainsEXT,

    // ---- VK_FUCHSIA_imagepipe_surface extension commands
#ifdef VK_USE_PLATFORM_FUCHSIA
    .CreateImagePipeSurfaceFUCHSIA = terminator_CreateImagePipeSurfaceFUCHSIA,
#endif // VK_USE_PLATFORM_FUCHSIA

    // ---- VK_EXT_metal_surface extension commands
#ifdef VK_USE_PLATFORM_METAL_EXT
    .CreateMetalSurfaceEXT = terminator_CreateMetalSurfaceEXT,
#endif // VK_USE_PLATFORM_METAL_EXT

    // ---- VK_EXT_tooling_info extension commands
    .GetPhysicalDeviceToolPropertiesEXT = terminator_GetPhysicalDeviceToolPropertiesEXT,

    // ---- VK_NV_cooperative_matrix extension commands
    .GetPhysicalDeviceCooperativeMatrixPropertiesNV = terminator_GetPhysicalDeviceCooperativeMatrixPropertiesNV,

    // ---- VK_NV_coverage_reduction_mode extension commands
    .GetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV = terminator_GetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV,

    // ---- VK_EXT_full_screen_exclusive extension commands
#ifdef VK_USE_PLATFORM_WIN32_KHR
    .GetPhysicalDeviceSurfacePresentModes2EXT = terminator_GetPhysicalDeviceSurfacePresentModes2EXT,
#endif // VK_USE_PLATFORM_WIN32_KHR

    // ---- VK_EXT_headless_surface extension commands
    .CreateHeadlessSurfaceEXT = terminator_CreateHeadlessSurfaceEXT,

    // ---- VK_EXT_directfb_surface extension commands
#ifdef VK_USE_PLATFORM_DIRECTFB_EXT
    .CreateDirectFBSurfaceEXT = terminator_CreateDirectFBSurfaceEXT,
#endif // VK_USE_PLATFORM_DIRECTFB_EXT
#ifdef VK_USE_PLATFORM_DIRECTFB_EXT
    .GetPhysicalDeviceDirectFBPresentationSupportEXT = terminator_GetPhysicalDeviceDirectFBPresentationSupportEXT,
#endif // VK_USE_PLATFORM_DIRECTFB_EXT
};

// A null-terminated list of all of the instance extensions supported by the loader.
// If an instance extension name is not in this list, but it is exported by one or more of the
// ICDs detected by the loader, then the extension name not in the list will be filtered out
// before passing the list of extensions to the application.
const char *const LOADER_INSTANCE_EXTENSIONS[] = {
                                                  VK_KHR_SURFACE_EXTENSION_NAME,
                                                  VK_KHR_DISPLAY_EXTENSION_NAME,
#ifdef VK_USE_PLATFORM_XLIB_KHR
                                                  VK_KHR_XLIB_SURFACE_EXTENSION_NAME,
#endif // VK_USE_PLATFORM_XLIB_KHR
#ifdef VK_USE_PLATFORM_XCB_KHR
                                                  VK_KHR_XCB_SURFACE_EXTENSION_NAME,
#endif // VK_USE_PLATFORM_XCB_KHR
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
                                                  VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME,
#endif // VK_USE_PLATFORM_WAYLAND_KHR
#ifdef VK_USE_PLATFORM_WIN32_KHR
                                                  VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
#endif // VK_USE_PLATFORM_WIN32_KHR
                                                  VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
                                                  VK_KHR_DEVICE_GROUP_CREATION_EXTENSION_NAME,
                                                  VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
                                                  VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
                                                  VK_KHR_EXTERNAL_FENCE_CAPABILITIES_EXTENSION_NAME,
                                                  VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME,
                                                  VK_KHR_GET_DISPLAY_PROPERTIES_2_EXTENSION_NAME,
                                                  VK_KHR_SURFACE_PROTECTED_CAPABILITIES_EXTENSION_NAME,
                                                  VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
#ifdef VK_USE_PLATFORM_GGP
                                                  VK_GGP_STREAM_DESCRIPTOR_SURFACE_EXTENSION_NAME,
#endif // VK_USE_PLATFORM_GGP
                                                  VK_NV_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
                                                  VK_EXT_VALIDATION_FLAGS_EXTENSION_NAME,
#ifdef VK_USE_PLATFORM_VI_NN
                                                  VK_NN_VI_SURFACE_EXTENSION_NAME,
#endif // VK_USE_PLATFORM_VI_NN
                                                  VK_EXT_DIRECT_MODE_DISPLAY_EXTENSION_NAME,
#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT
                                                  VK_EXT_ACQUIRE_XLIB_DISPLAY_EXTENSION_NAME,
#endif // VK_USE_PLATFORM_XLIB_XRANDR_EXT
                                                  VK_EXT_DISPLAY_SURFACE_COUNTER_EXTENSION_NAME,
                                                  VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME,
#ifdef VK_USE_PLATFORM_IOS_MVK
                                                  VK_MVK_IOS_SURFACE_EXTENSION_NAME,
#endif // VK_USE_PLATFORM_IOS_MVK
#ifdef VK_USE_PLATFORM_MACOS_MVK
                                                  VK_MVK_MACOS_SURFACE_EXTENSION_NAME,
#endif // VK_USE_PLATFORM_MACOS_MVK
                                                  VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
#ifdef VK_USE_PLATFORM_FUCHSIA
                                                  VK_FUCHSIA_IMAGEPIPE_SURFACE_EXTENSION_NAME,
#endif // VK_USE_PLATFORM_FUCHSIA
#ifdef VK_USE_PLATFORM_METAL_EXT
                                                  VK_EXT_METAL_SURFACE_EXTENSION_NAME,
#endif // VK_USE_PLATFORM_METAL_EXT
                                                  VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME,
                                                  VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME,
#ifdef VK_USE_PLATFORM_DIRECTFB_EXT
                                                  VK_EXT_DIRECTFB_SURFACE_EXTENSION_NAME,
#endif // VK_USE_PLATFORM_DIRECTFB_EXT
                                                  NULL };

