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
 * Author: Mark Young <marky@lunarg.com>
 * Author: Lenny Komow <lenny@lunarg.com>
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

// ---- Manually added trampoline/terminator functions

// These functions, for whatever reason, require more complex changes than
// can easily be automatically generated.

// ---- VK_NV_external_memory_capabilities extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL
GetPhysicalDeviceExternalImageFormatPropertiesNV(
    VkPhysicalDevice physicalDevice, VkFormat format, VkImageType type,
    VkImageTiling tiling, VkImageUsageFlags usage, VkImageCreateFlags flags,
    VkExternalMemoryHandleTypeFlagsNV externalHandleType,
    VkExternalImageFormatPropertiesNV *pExternalImageFormatProperties) {
    const VkLayerInstanceDispatchTable *disp;
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    disp = loader_get_instance_layer_dispatch(physicalDevice);

    return disp->GetPhysicalDeviceExternalImageFormatPropertiesNV(
        unwrapped_phys_dev, format, type, tiling, usage, flags,
        externalHandleType, pExternalImageFormatProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL
terminator_GetPhysicalDeviceExternalImageFormatPropertiesNV(
    VkPhysicalDevice physicalDevice, VkFormat format, VkImageType type,
    VkImageTiling tiling, VkImageUsageFlags usage, VkImageCreateFlags flags,
    VkExternalMemoryHandleTypeFlagsNV externalHandleType,
    VkExternalImageFormatPropertiesNV *pExternalImageFormatProperties) {
    struct loader_physical_device_term *phys_dev_term =
        (struct loader_physical_device_term *)physicalDevice;
    struct loader_icd_term *icd_term = phys_dev_term->this_icd_term;

    if (!icd_term->dispatch.GetPhysicalDeviceExternalImageFormatPropertiesNV) {
        if (externalHandleType) {
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        }

        if (!icd_term->dispatch.GetPhysicalDeviceImageFormatProperties) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        pExternalImageFormatProperties->externalMemoryFeatures = 0;
        pExternalImageFormatProperties->exportFromImportedHandleTypes = 0;
        pExternalImageFormatProperties->compatibleHandleTypes = 0;

        return icd_term->dispatch.GetPhysicalDeviceImageFormatProperties(
            phys_dev_term->phys_dev, format, type, tiling, usage, flags,
            &pExternalImageFormatProperties->imageFormatProperties);
    }

    return icd_term->dispatch.GetPhysicalDeviceExternalImageFormatPropertiesNV(
        phys_dev_term->phys_dev, format, type, tiling, usage, flags,
        externalHandleType, pExternalImageFormatProperties);
}

// ---- VK_EXT_display_surface_counter extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfaceCapabilities2EXT(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface,
                                                                        VkSurfaceCapabilities2EXT *pSurfaceCapabilities) {
    const VkLayerInstanceDispatchTable *disp;
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    disp = loader_get_instance_layer_dispatch(physicalDevice);
    return disp->GetPhysicalDeviceSurfaceCapabilities2EXT(unwrapped_phys_dev, surface, pSurfaceCapabilities);
}

VKAPI_ATTR VkResult VKAPI_CALL terminator_GetPhysicalDeviceSurfaceCapabilities2EXT(
    VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, VkSurfaceCapabilities2EXT *pSurfaceCapabilities) {
    struct loader_physical_device_term *phys_dev_term = (struct loader_physical_device_term *)physicalDevice;
    struct loader_icd_term *icd_term = phys_dev_term->this_icd_term;

    VkIcdSurface *icd_surface = (VkIcdSurface *)(surface);
    uint8_t icd_index = phys_dev_term->icd_index;

    // Unwrap the surface if needed
    VkSurfaceKHR unwrapped_surface = surface;
    if (icd_surface->real_icd_surfaces != NULL && (void *)icd_surface->real_icd_surfaces[icd_index] != NULL) {
        unwrapped_surface = icd_surface->real_icd_surfaces[icd_index];
    }

    if (icd_term->dispatch.GetPhysicalDeviceSurfaceCapabilities2EXT != NULL) {
        // Pass the call to the driver
        return icd_term->dispatch.GetPhysicalDeviceSurfaceCapabilities2EXT(phys_dev_term->phys_dev, unwrapped_surface,
                                                                           pSurfaceCapabilities);
    } else {
        // Emulate the call
        loader_log(icd_term->this_instance, VK_DEBUG_REPORT_INFORMATION_BIT_EXT, 0,
                   "vkGetPhysicalDeviceSurfaceCapabilities2EXT: Emulating call in ICD \"%s\" using "
                   "vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
                   icd_term->scanned_icd->lib_name);

        VkSurfaceCapabilitiesKHR surface_caps;
        VkResult res =
            icd_term->dispatch.GetPhysicalDeviceSurfaceCapabilitiesKHR(phys_dev_term->phys_dev, unwrapped_surface, &surface_caps);
        pSurfaceCapabilities->minImageCount = surface_caps.minImageCount;
        pSurfaceCapabilities->maxImageCount = surface_caps.maxImageCount;
        pSurfaceCapabilities->currentExtent = surface_caps.currentExtent;
        pSurfaceCapabilities->minImageExtent = surface_caps.minImageExtent;
        pSurfaceCapabilities->maxImageExtent = surface_caps.maxImageExtent;
        pSurfaceCapabilities->maxImageArrayLayers = surface_caps.maxImageArrayLayers;
        pSurfaceCapabilities->supportedTransforms = surface_caps.supportedTransforms;
        pSurfaceCapabilities->currentTransform = surface_caps.currentTransform;
        pSurfaceCapabilities->supportedCompositeAlpha = surface_caps.supportedCompositeAlpha;
        pSurfaceCapabilities->supportedUsageFlags = surface_caps.supportedUsageFlags;
        pSurfaceCapabilities->supportedSurfaceCounters = 0;

        if (pSurfaceCapabilities->pNext != NULL) {
            loader_log(icd_term->this_instance, VK_DEBUG_REPORT_WARNING_BIT_EXT, 0,
                       "vkGetPhysicalDeviceSurfaceCapabilities2EXT: Emulation found unrecognized structure type in "
                       "pSurfaceCapabilities->pNext - this struct will be ignored");
        }

        return res;
    }
}

// ---- VK_EXT_direct_mode_display extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL ReleaseDisplayEXT(VkPhysicalDevice physicalDevice, VkDisplayKHR display) {
    const VkLayerInstanceDispatchTable *disp;
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    disp = loader_get_instance_layer_dispatch(physicalDevice);
    return disp->ReleaseDisplayEXT(unwrapped_phys_dev, display);
}

VKAPI_ATTR VkResult VKAPI_CALL terminator_ReleaseDisplayEXT(VkPhysicalDevice physicalDevice, VkDisplayKHR display) {
    struct loader_physical_device_term *phys_dev_term = (struct loader_physical_device_term *)physicalDevice;
    struct loader_icd_term *icd_term = phys_dev_term->this_icd_term;

    if (icd_term->dispatch.ReleaseDisplayEXT == NULL) {
        loader_log(icd_term->this_instance, VK_DEBUG_REPORT_ERROR_BIT_EXT, 0,
                   "ICD \"%s\" associated with VkPhysicalDevice does not support vkReleaseDisplayEXT - Consequently, the call is "
                   "invalid because it should not be possible to acquire a display on this device",
                   icd_term->scanned_icd->lib_name);
    }
    return icd_term->dispatch.ReleaseDisplayEXT(phys_dev_term->phys_dev, display);
}

// ---- VK_EXT_acquire_xlib_display extension trampoline/terminators

#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT
VKAPI_ATTR VkResult VKAPI_CALL AcquireXlibDisplayEXT(VkPhysicalDevice physicalDevice, Display *dpy, VkDisplayKHR display) {
    const VkLayerInstanceDispatchTable *disp;
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    disp = loader_get_instance_layer_dispatch(physicalDevice);
    return disp->AcquireXlibDisplayEXT(unwrapped_phys_dev, dpy, display);
}

VKAPI_ATTR VkResult VKAPI_CALL terminator_AcquireXlibDisplayEXT(VkPhysicalDevice physicalDevice, Display *dpy,
                                                                VkDisplayKHR display) {
    struct loader_physical_device_term *phys_dev_term = (struct loader_physical_device_term *)physicalDevice;
    struct loader_icd_term *icd_term = phys_dev_term->this_icd_term;

    if (icd_term->dispatch.AcquireXlibDisplayEXT != NULL) {
        // Pass the call to the driver
        return icd_term->dispatch.AcquireXlibDisplayEXT(phys_dev_term->phys_dev, dpy, display);
    } else {
        // Emulate the call
        loader_log(icd_term->this_instance, VK_DEBUG_REPORT_INFORMATION_BIT_EXT, 0,
                   "vkAcquireXLibDisplayEXT: Emulating call in ICD \"%s\" by returning error", icd_term->scanned_icd->lib_name);

        // Fail for the unsupported command
        return VK_ERROR_INITIALIZATION_FAILED;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL GetRandROutputDisplayEXT(VkPhysicalDevice physicalDevice, Display *dpy, RROutput rrOutput,
                                                        VkDisplayKHR *pDisplay) {
    const VkLayerInstanceDispatchTable *disp;
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    disp = loader_get_instance_layer_dispatch(physicalDevice);
    return disp->GetRandROutputDisplayEXT(unwrapped_phys_dev, dpy, rrOutput, pDisplay);
}

VKAPI_ATTR VkResult VKAPI_CALL terminator_GetRandROutputDisplayEXT(VkPhysicalDevice physicalDevice, Display *dpy, RROutput rrOutput,
                                                                   VkDisplayKHR *pDisplay) {
    struct loader_physical_device_term *phys_dev_term = (struct loader_physical_device_term *)physicalDevice;
    struct loader_icd_term *icd_term = phys_dev_term->this_icd_term;

    if (icd_term->dispatch.GetRandROutputDisplayEXT != NULL) {
        // Pass the call to the driver
        return icd_term->dispatch.GetRandROutputDisplayEXT(phys_dev_term->phys_dev, dpy, rrOutput, pDisplay);
    } else {
        // Emulate the call
        loader_log(icd_term->this_instance, VK_DEBUG_REPORT_INFORMATION_BIT_EXT, 0,
                   "vkGetRandROutputDisplayEXT: Emulating call in ICD \"%s\" by returning null display",
                   icd_term->scanned_icd->lib_name);

        // Return a null handle to indicate this can't be done
        *pDisplay = VK_NULL_HANDLE;
        return VK_SUCCESS;
    }
}

#endif  // VK_USE_PLATFORM_XLIB_XRANDR_EXT

#ifdef VK_USE_PLATFORM_WIN32_KHR
VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceSurfacePresentModes2EXT(
    VkPhysicalDevice                            physicalDevice,
    const VkPhysicalDeviceSurfaceInfo2KHR*      pSurfaceInfo,
    uint32_t*                                   pPresentModeCount,
    VkPresentModeKHR*                           pPresentModes) {
    const VkLayerInstanceDispatchTable *disp;
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    disp = loader_get_instance_layer_dispatch(physicalDevice);
    return disp->GetPhysicalDeviceSurfacePresentModes2EXT(unwrapped_phys_dev, pSurfaceInfo, pPresentModeCount, pPresentModes);
}

VKAPI_ATTR VkResult VKAPI_CALL terminator_GetPhysicalDeviceSurfacePresentModes2EXT(
    VkPhysicalDevice                            physicalDevice,
    const VkPhysicalDeviceSurfaceInfo2KHR*      pSurfaceInfo,
    uint32_t*                                   pPresentModeCount,
    VkPresentModeKHR*                           pPresentModes) {
    struct loader_physical_device_term *phys_dev_term = (struct loader_physical_device_term *)physicalDevice;
    struct loader_icd_term *icd_term = phys_dev_term->this_icd_term;
    if (NULL == icd_term->dispatch.GetPhysicalDeviceSurfacePresentModes2EXT) {
        loader_log(icd_term->this_instance, VK_DEBUG_REPORT_ERROR_BIT_EXT, 0,
                   "ICD associated with VkPhysicalDevice does not support GetPhysicalDeviceSurfacePresentModes2EXT");
    }
    VkIcdSurface *icd_surface = (VkIcdSurface *)(pSurfaceInfo->surface);
    uint8_t icd_index = phys_dev_term->icd_index;
    if (NULL != icd_surface->real_icd_surfaces && NULL != (void *)icd_surface->real_icd_surfaces[icd_index]) {
        const VkPhysicalDeviceSurfaceInfo2KHR surface_info_copy = {
            .sType = pSurfaceInfo->sType,
            .pNext = pSurfaceInfo->pNext,
            .surface = icd_surface->real_icd_surfaces[icd_index],
        };
        return icd_term->dispatch.GetPhysicalDeviceSurfacePresentModes2EXT(phys_dev_term->phys_dev, &surface_info_copy, pPresentModeCount, pPresentModes);
    }
    return icd_term->dispatch.GetPhysicalDeviceSurfacePresentModes2EXT(phys_dev_term->phys_dev, pSurfaceInfo, pPresentModeCount, pPresentModes);
}

VKAPI_ATTR VkResult VKAPI_CALL GetDeviceGroupSurfacePresentModes2EXT(
    VkDevice                                    device,
    const VkPhysicalDeviceSurfaceInfo2KHR*      pSurfaceInfo,
    VkDeviceGroupPresentModeFlagsKHR*           pModes) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetDeviceGroupSurfacePresentModes2EXT(device, pSurfaceInfo, pModes);
}

VKAPI_ATTR VkResult VKAPI_CALL terminator_GetDeviceGroupSurfacePresentModes2EXT(
    VkDevice                                    device,
    const VkPhysicalDeviceSurfaceInfo2KHR*      pSurfaceInfo,
    VkDeviceGroupPresentModeFlagsKHR*           pModes) {
    uint32_t icd_index = 0;
    struct loader_device *dev;
    struct loader_icd_term *icd_term = loader_get_icd_and_device(device, &dev, &icd_index);
    if (NULL != icd_term && NULL != icd_term->dispatch.GetDeviceGroupSurfacePresentModes2EXT) {
        VkIcdSurface *icd_surface = (VkIcdSurface *)(uintptr_t)pSurfaceInfo->surface;
        if (NULL != icd_surface->real_icd_surfaces && (VkSurfaceKHR)NULL != icd_surface->real_icd_surfaces[icd_index]) {
            const VkPhysicalDeviceSurfaceInfo2KHR surface_info_copy = {
                .sType = pSurfaceInfo->sType,
                .pNext = pSurfaceInfo->pNext,
                .surface = icd_surface->real_icd_surfaces[icd_index],
            };
            return icd_term->dispatch.GetDeviceGroupSurfacePresentModes2EXT(device, &surface_info_copy, pModes);
        }
        return icd_term->dispatch.GetDeviceGroupSurfacePresentModes2EXT(device, pSurfaceInfo, pModes);
    }
    return VK_SUCCESS;
}

#endif  // VK_USE_PLATFORM_WIN32_KHR

// ---- VK_EXT_tooling_info extension trampoline/terminators

VKAPI_ATTR VkResult VKAPI_CALL GetPhysicalDeviceToolPropertiesEXT(
    VkPhysicalDevice                            physicalDevice,
    uint32_t*                                   pToolCount,
    VkPhysicalDeviceToolPropertiesEXT*          pToolProperties) {
    const VkLayerInstanceDispatchTable *disp;
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    disp = loader_get_instance_layer_dispatch(physicalDevice);
    return disp->GetPhysicalDeviceToolPropertiesEXT(unwrapped_phys_dev, pToolCount, pToolProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL terminator_GetPhysicalDeviceToolPropertiesEXT(
    VkPhysicalDevice                            physicalDevice,
    uint32_t*                                   pToolCount,
    VkPhysicalDeviceToolPropertiesEXT*          pToolProperties) {
    return VK_SUCCESS;
}
