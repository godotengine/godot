/*
 *
 * Copyright (c) 2015-2016 The Khronos Group Inc.
 * Copyright (c) 2015-2016 Valve Corporation
 * Copyright (c) 2015-2016 LunarG, Inc.
 * Copyright (C) 2015 Google Inc.
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
 * Author: Courtney Goeltzenleuchter <courtney@lunarg.com>
 * Author: Jon Ashburn <jon@lunarg.com>
 * Author: Tony Barbour <tony@LunarG.com>
 * Author: Chia-I Wu <olv@lunarg.com>
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdlib.h>
#include <string.h>

#include "vk_loader_platform.h"
#include "loader.h"
#include "debug_utils.h"
#include "wsi.h"
#include "vk_loader_extensions.h"
#include "gpa_helper.h"


// Trampoline entrypoints are in this file for core Vulkan commands

// Get an instance level or global level entry point address.
// @param instance
// @param pName
// @return
//    If instance == NULL returns a global level functions only
//    If instance is valid returns a trampoline entry point for all dispatchable Vulkan
//    functions both core and extensions.
LOADER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *pName) {
    void *addr;

    addr = globalGetProcAddr(pName);
    if (instance == VK_NULL_HANDLE || addr != NULL) {
        return addr;
    }

    struct loader_instance *ptr_instance = loader_get_instance(instance);
    if (ptr_instance == NULL) return NULL;
    // Return trampoline code for non-global entrypoints including any extensions.
    // Device extensions are returned if a layer or ICD supports the extension.
    // Instance extensions are returned if the extension is enabled and the
    // loader or someone else supports the extension
    return trampolineGetProcAddr(ptr_instance, pName);
}

// Get a device level or global level entry point address.
// @param device
// @param pName
// @return
//    If device is valid, returns a device relative entry point for device level
//    entry points both core and extensions.
//    Device relative means call down the device chain.
LOADER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *pName) {
    void *addr;

    // For entrypoints that loader must handle (ie non-dispatchable or create object)
    // make sure the loader entrypoint is returned
    addr = loader_non_passthrough_gdpa(pName);
    if (addr) {
        return addr;
    }

    // Although CreateDevice is on device chain it's dispatchable object isn't
    // a VkDevice or child of VkDevice so return NULL.
    if (!strcmp(pName, "CreateDevice")) return NULL;

    // Return the dispatch table entrypoint for the fastest case
    const VkLayerDispatchTable *disp_table = *(VkLayerDispatchTable **)device;
    if (disp_table == NULL) return NULL;

    addr = loader_lookup_device_dispatch_table(disp_table, pName);
    if (addr) return addr;

    if (disp_table->GetDeviceProcAddr == NULL) return NULL;
    return disp_table->GetDeviceProcAddr(device, pName);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char *pLayerName,
                                                                                    uint32_t *pPropertyCount,
                                                                                    VkExtensionProperties *pProperties) {
    tls_instance = NULL;
    LOADER_PLATFORM_THREAD_ONCE(&once_init, loader_initialize);

    // We know we need to call at least the terminator
    VkResult res = VK_SUCCESS;
    VkEnumerateInstanceExtensionPropertiesChain chain_tail = {
        .header =
            {
                .type = VK_CHAIN_TYPE_ENUMERATE_INSTANCE_EXTENSION_PROPERTIES,
                .version = VK_CURRENT_CHAIN_VERSION,
                .size = sizeof(chain_tail),
            },
        .pfnNextLayer = &terminator_EnumerateInstanceExtensionProperties,
        .pNextLink = NULL,
    };
    VkEnumerateInstanceExtensionPropertiesChain *chain_head = &chain_tail;

    // Get the implicit layers
    struct loader_layer_list layers;
    memset(&layers, 0, sizeof(layers));
    loaderScanForImplicitLayers(NULL, &layers);

    // We'll need to save the dl handles so we can close them later
    loader_platform_dl_handle *libs = malloc(sizeof(loader_platform_dl_handle) * layers.count);
    if (libs == NULL) {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    size_t lib_count = 0;

    // Prepend layers onto the chain if they implement this entry point
    for (uint32_t i = 0; i < layers.count; ++i) {
        if (!loaderImplicitLayerIsEnabled(NULL, layers.list + i) ||
            layers.list[i].pre_instance_functions.enumerate_instance_extension_properties[0] == '\0') {
            continue;
        }

        loader_platform_dl_handle layer_lib = loader_platform_open_library(layers.list[i].lib_name);
        if (layer_lib == NULL) {
            loader_log(NULL, VK_DEBUG_REPORT_WARNING_BIT_EXT, 0, "%s: Unable to load implicit layer library \"%s\"", __FUNCTION__,
                       layers.list[i].lib_name);
            continue;
        }

        libs[lib_count++] = layer_lib;
        void *pfn = loader_platform_get_proc_address(layer_lib,
                                                     layers.list[i].pre_instance_functions.enumerate_instance_extension_properties);
        if (pfn == NULL) {
            loader_log(NULL, VK_DEBUG_REPORT_WARNING_BIT_EXT, 0,
                       "%s: Unable to resolve symbol \"%s\" in implicit layer library \"%s\"", __FUNCTION__,
                       layers.list[i].pre_instance_functions.enumerate_instance_extension_properties, layers.list[i].lib_name);
            continue;
        }

        VkEnumerateInstanceExtensionPropertiesChain *chain_link = malloc(sizeof(VkEnumerateInstanceExtensionPropertiesChain));
        if (chain_link == NULL) {
            res = VK_ERROR_OUT_OF_HOST_MEMORY;
            break;
        }

        chain_link->header.type = VK_CHAIN_TYPE_ENUMERATE_INSTANCE_EXTENSION_PROPERTIES;
        chain_link->header.version = VK_CURRENT_CHAIN_VERSION;
        chain_link->header.size = sizeof(*chain_link);
        chain_link->pfnNextLayer = pfn;
        chain_link->pNextLink = chain_head;

        chain_head = chain_link;
    }

    // Call down the chain
    if (res == VK_SUCCESS) {
        res = chain_head->pfnNextLayer(chain_head->pNextLink, pLayerName, pPropertyCount, pProperties);
    }

    // Free up the layers
    loaderDeleteLayerListAndProperties(NULL, &layers);

    // Tear down the chain
    while (chain_head != &chain_tail) {
        VkEnumerateInstanceExtensionPropertiesChain *holder = chain_head;
        chain_head = (VkEnumerateInstanceExtensionPropertiesChain *)chain_head->pNextLink;
        free(holder);
    }

    // Close the dl handles
    for (size_t i = 0; i < lib_count; ++i) {
        loader_platform_close_library(libs[i]);
    }
    free(libs);

    return res;
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t *pPropertyCount,
                                                                                VkLayerProperties *pProperties) {
    tls_instance = NULL;
    LOADER_PLATFORM_THREAD_ONCE(&once_init, loader_initialize);

    // We know we need to call at least the terminator
    VkResult res = VK_SUCCESS;
    VkEnumerateInstanceLayerPropertiesChain chain_tail = {
        .header =
            {
                .type = VK_CHAIN_TYPE_ENUMERATE_INSTANCE_LAYER_PROPERTIES,
                .version = VK_CURRENT_CHAIN_VERSION,
                .size = sizeof(chain_tail),
            },
        .pfnNextLayer = &terminator_EnumerateInstanceLayerProperties,
        .pNextLink = NULL,
    };
    VkEnumerateInstanceLayerPropertiesChain *chain_head = &chain_tail;

    // Get the implicit layers
    struct loader_layer_list layers;
    memset(&layers, 0, sizeof(layers));
    loaderScanForImplicitLayers(NULL, &layers);

    // We'll need to save the dl handles so we can close them later
    loader_platform_dl_handle *libs = malloc(sizeof(loader_platform_dl_handle) * layers.count);
    if (libs == NULL) {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    size_t lib_count = 0;

    // Prepend layers onto the chain if they implement this entry point
    for (uint32_t i = 0; i < layers.count; ++i) {
        if (!loaderImplicitLayerIsEnabled(NULL, layers.list + i) ||
            layers.list[i].pre_instance_functions.enumerate_instance_layer_properties[0] == '\0') {
            continue;
        }

        loader_platform_dl_handle layer_lib = loader_platform_open_library(layers.list[i].lib_name);
        if (layer_lib == NULL) {
            loader_log(NULL, VK_DEBUG_REPORT_WARNING_BIT_EXT, 0, "%s: Unable to load implicit layer library \"%s\"", __FUNCTION__,
                       layers.list[i].lib_name);
            continue;
        }

        libs[lib_count++] = layer_lib;
        void *pfn =
            loader_platform_get_proc_address(layer_lib, layers.list[i].pre_instance_functions.enumerate_instance_layer_properties);
        if (pfn == NULL) {
            loader_log(NULL, VK_DEBUG_REPORT_WARNING_BIT_EXT, 0,
                       "%s: Unable to resolve symbol \"%s\" in implicit layer library \"%s\"", __FUNCTION__,
                       layers.list[i].pre_instance_functions.enumerate_instance_layer_properties, layers.list[i].lib_name);
            continue;
        }

        VkEnumerateInstanceLayerPropertiesChain *chain_link = malloc(sizeof(VkEnumerateInstanceLayerPropertiesChain));
        if (chain_link == NULL) {
            res = VK_ERROR_OUT_OF_HOST_MEMORY;
            break;
        }

        chain_link->header.type = VK_CHAIN_TYPE_ENUMERATE_INSTANCE_LAYER_PROPERTIES;
        chain_link->header.version = VK_CURRENT_CHAIN_VERSION;
        chain_link->header.size = sizeof(*chain_link);
        chain_link->pfnNextLayer = pfn;
        chain_link->pNextLink = chain_head;

        chain_head = chain_link;
    }

    // Call down the chain
    if (res == VK_SUCCESS) {
        res = chain_head->pfnNextLayer(chain_head->pNextLink, pPropertyCount, pProperties);
    }

    // Free up the layers
    loaderDeleteLayerListAndProperties(NULL, &layers);

    // Tear down the chain
    while (chain_head != &chain_tail) {
        VkEnumerateInstanceLayerPropertiesChain *holder = chain_head;
        chain_head = (VkEnumerateInstanceLayerPropertiesChain *)chain_head->pNextLink;
        free(holder);
    }

    // Close the dl handles
    for (size_t i = 0; i < lib_count; ++i) {
        loader_platform_close_library(libs[i]);
    }
    free(libs);

    return res;
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceVersion(uint32_t* pApiVersion) {

    tls_instance = NULL;
    LOADER_PLATFORM_THREAD_ONCE(&once_init, loader_initialize);

    // We know we need to call at least the terminator
    VkResult res = VK_SUCCESS;
    VkEnumerateInstanceVersionChain chain_tail = {
        .header =
            {
                .type = VK_CHAIN_TYPE_ENUMERATE_INSTANCE_VERSION,
                .version = VK_CURRENT_CHAIN_VERSION,
                .size = sizeof(chain_tail),
            },
        .pfnNextLayer = &terminator_EnumerateInstanceVersion,
        .pNextLink = NULL,
    };
    VkEnumerateInstanceVersionChain *chain_head = &chain_tail;

    // Get the implicit layers
    struct loader_layer_list layers;
    memset(&layers, 0, sizeof(layers));
    loaderScanForImplicitLayers(NULL, &layers);

    // We'll need to save the dl handles so we can close them later
    loader_platform_dl_handle *libs = malloc(sizeof(loader_platform_dl_handle) * layers.count);
    if (libs == NULL) {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    size_t lib_count = 0;

    // Prepend layers onto the chain if they implement this entry point
    for (uint32_t i = 0; i < layers.count; ++i) {
        if (!loaderImplicitLayerIsEnabled(NULL, layers.list + i) ||
            layers.list[i].pre_instance_functions.enumerate_instance_version[0] == '\0') {
            continue;
        }

        loader_platform_dl_handle layer_lib = loader_platform_open_library(layers.list[i].lib_name);
        if (layer_lib == NULL) {
            loader_log(NULL, VK_DEBUG_REPORT_WARNING_BIT_EXT, 0, "%s: Unable to load implicit layer library \"%s\"", __FUNCTION__,
                       layers.list[i].lib_name);
            continue;
        }

        libs[lib_count++] = layer_lib;
        void *pfn = loader_platform_get_proc_address(layer_lib,
                                                     layers.list[i].pre_instance_functions.enumerate_instance_version);
        if (pfn == NULL) {
            loader_log(NULL, VK_DEBUG_REPORT_WARNING_BIT_EXT, 0,
                       "%s: Unable to resolve symbol \"%s\" in implicit layer library \"%s\"", __FUNCTION__,
                       layers.list[i].pre_instance_functions.enumerate_instance_version, layers.list[i].lib_name);
            continue;
        }

        VkEnumerateInstanceVersionChain *chain_link = malloc(sizeof(VkEnumerateInstanceVersionChain));
        if (chain_link == NULL) {
            res = VK_ERROR_OUT_OF_HOST_MEMORY;
            break;
        }

        chain_link->header.type = VK_CHAIN_TYPE_ENUMERATE_INSTANCE_VERSION;
        chain_link->header.version = VK_CURRENT_CHAIN_VERSION;
        chain_link->header.size = sizeof(*chain_link);
        chain_link->pfnNextLayer = pfn;
        chain_link->pNextLink = chain_head;

        chain_head = chain_link;
    }

    // Call down the chain
    if (res == VK_SUCCESS) {
        res = chain_head->pfnNextLayer(chain_head->pNextLink, pApiVersion);
    }

    // Free up the layers
    loaderDeleteLayerListAndProperties(NULL, &layers);

    // Tear down the chain
    while (chain_head != &chain_tail) {
        VkEnumerateInstanceVersionChain *holder = chain_head;
        chain_head = (VkEnumerateInstanceVersionChain *)chain_head->pNextLink;
        free(holder);
    }

    // Close the dl handles
    for (size_t i = 0; i < lib_count; ++i) {
        loader_platform_close_library(libs[i]);
    }
    free(libs);

    return res;
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(const VkInstanceCreateInfo *pCreateInfo,
                                                              const VkAllocationCallbacks *pAllocator, VkInstance *pInstance) {
    struct loader_instance *ptr_instance = NULL;
    VkInstance created_instance = VK_NULL_HANDLE;
    bool loaderLocked = false;
    VkResult res = VK_ERROR_INITIALIZATION_FAILED;

    LOADER_PLATFORM_THREAD_ONCE(&once_init, loader_initialize);

#if (DEBUG_DISABLE_APP_ALLOCATORS == 1)
    {
#else
    if (pAllocator) {
        ptr_instance = (struct loader_instance *)pAllocator->pfnAllocation(pAllocator->pUserData, sizeof(struct loader_instance),
                                                                           sizeof(int *), VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
    } else {
#endif
        ptr_instance = (struct loader_instance *)malloc(sizeof(struct loader_instance));
    }

    VkInstanceCreateInfo ici = *pCreateInfo;

    if (ptr_instance == NULL) {
        res = VK_ERROR_OUT_OF_HOST_MEMORY;
        goto out;
    }

    tls_instance = ptr_instance;
    loader_platform_thread_lock_mutex(&loader_lock);
    loaderLocked = true;
    memset(ptr_instance, 0, sizeof(struct loader_instance));
    if (pAllocator) {
        ptr_instance->alloc_callbacks = *pAllocator;
    }

    // Save the application version
    if (NULL == pCreateInfo || NULL == pCreateInfo->pApplicationInfo || 0 == pCreateInfo->pApplicationInfo->apiVersion)
{
        ptr_instance->app_api_major_version = 1;
        ptr_instance->app_api_minor_version = 0;
    } else {
        ptr_instance->app_api_major_version = VK_VERSION_MAJOR(pCreateInfo->pApplicationInfo->apiVersion);
        ptr_instance->app_api_minor_version = VK_VERSION_MINOR(pCreateInfo->pApplicationInfo->apiVersion);
    }

    // Look for one or more VK_EXT_debug_report or VK_EXT_debug_utils create info structures
    // and setup a callback(s) for each one found.
    ptr_instance->num_tmp_report_callbacks = 0;
    ptr_instance->tmp_report_create_infos = NULL;
    ptr_instance->tmp_report_callbacks = NULL;
    ptr_instance->num_tmp_messengers = 0;
    ptr_instance->tmp_messenger_create_infos = NULL;
    ptr_instance->tmp_messengers = NULL;

    // Handle cases of VK_EXT_debug_utils
    if (util_CopyDebugUtilsMessengerCreateInfos(pCreateInfo->pNext, pAllocator, &ptr_instance->num_tmp_messengers,
                                                &ptr_instance->tmp_messenger_create_infos, &ptr_instance->tmp_messengers)) {
        // One or more were found, but allocation failed.  Therefore, clean up and fail this function:
        res = VK_ERROR_OUT_OF_HOST_MEMORY;
        goto out;
    } else if (ptr_instance->num_tmp_messengers > 0) {
        // Setup the temporary messenger(s) here to catch early issues:
        if (util_CreateDebugUtilsMessengers(ptr_instance, pAllocator, ptr_instance->num_tmp_messengers,
                                            ptr_instance->tmp_messenger_create_infos, ptr_instance->tmp_messengers)) {
            // Failure of setting up one or more of the messenger.  Therefore, clean up and fail this function:
            res = VK_ERROR_OUT_OF_HOST_MEMORY;
            goto out;
        }
    }

    // Handle cases of VK_EXT_debug_report
    if (util_CopyDebugReportCreateInfos(pCreateInfo->pNext, pAllocator, &ptr_instance->num_tmp_report_callbacks,
                                        &ptr_instance->tmp_report_create_infos, &ptr_instance->tmp_report_callbacks)) {
        // One or more were found, but allocation failed.  Therefore, clean up and fail this function:
        res = VK_ERROR_OUT_OF_HOST_MEMORY;
        goto out;
    } else if (ptr_instance->num_tmp_report_callbacks > 0) {
        // Setup the temporary callback(s) here to catch early issues:
        if (util_CreateDebugReportCallbacks(ptr_instance, pAllocator, ptr_instance->num_tmp_report_callbacks,
                                            ptr_instance->tmp_report_create_infos, ptr_instance->tmp_report_callbacks)) {
            // Failure of setting up one or more of the callback.  Therefore, clean up and fail this function:
            res = VK_ERROR_OUT_OF_HOST_MEMORY;
            goto out;
        }
    }

    // Due to implicit layers need to get layer list even if
    // enabledLayerCount == 0 and VK_INSTANCE_LAYERS is unset. For now always
    // get layer list via loaderScanForLayers().
    memset(&ptr_instance->instance_layer_list, 0, sizeof(ptr_instance->instance_layer_list));
    loaderScanForLayers(ptr_instance, &ptr_instance->instance_layer_list);

    // Validate the app requested layers to be enabled
    if (pCreateInfo->enabledLayerCount > 0) {
        res = loaderValidateLayers(ptr_instance, pCreateInfo->enabledLayerCount, pCreateInfo->ppEnabledLayerNames,
                                   &ptr_instance->instance_layer_list);
        if (res != VK_SUCCESS) {
            goto out;
        }
    }

    // Scan/discover all ICD libraries
    memset(&ptr_instance->icd_tramp_list, 0, sizeof(ptr_instance->icd_tramp_list));
    res = loader_icd_scan(ptr_instance, &ptr_instance->icd_tramp_list);
    if (res != VK_SUCCESS) {
        goto out;
    }

    // Get extensions from all ICD's, merge so no duplicates, then validate
    res = loader_get_icd_loader_instance_extensions(ptr_instance, &ptr_instance->icd_tramp_list, &ptr_instance->ext_list);
    if (res != VK_SUCCESS) {
        goto out;
    }
    res = loader_validate_instance_extensions(ptr_instance, &ptr_instance->ext_list, &ptr_instance->instance_layer_list, &ici);
    if (res != VK_SUCCESS) {
        goto out;
    }

    ptr_instance->disp = loader_instance_heap_alloc(ptr_instance, sizeof(struct loader_instance_dispatch_table),
                                                    VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
    if (ptr_instance->disp == NULL) {
        loader_log(ptr_instance, VK_DEBUG_REPORT_ERROR_BIT_EXT, 0,
                   "vkCreateInstance:  Failed to allocate Loader's full Instance dispatch table.");
        res = VK_ERROR_OUT_OF_HOST_MEMORY;
        goto out;
    }
    memcpy(&ptr_instance->disp->layer_inst_disp, &instance_disp, sizeof(instance_disp));

    ptr_instance->next = loader.instances;
    loader.instances = ptr_instance;

    // Activate any layers on instance chain
    res = loaderEnableInstanceLayers(ptr_instance, &ici, &ptr_instance->instance_layer_list);
    if (res != VK_SUCCESS) {
        goto out;
    }

    created_instance = (VkInstance)ptr_instance;
    res = loader_create_instance_chain(&ici, pAllocator, ptr_instance, &created_instance);

    if (res == VK_SUCCESS) {
        memset(ptr_instance->enabled_known_extensions.padding, 0, sizeof(uint64_t) * 4);

        wsi_create_instance(ptr_instance, &ici);
        debug_utils_CreateInstance(ptr_instance, &ici);
        extensions_create_instance(ptr_instance, &ici);

        *pInstance = created_instance;

        // Finally have the layers in place and everyone has seen
        // the CreateInstance command go by. This allows the layer's
        // GetInstanceProcAddr functions to return valid extension functions
        // if enabled.
        loaderActivateInstanceLayerExtensions(ptr_instance, *pInstance);
    }

out:

    if (NULL != ptr_instance) {
        if (res != VK_SUCCESS) {
            if (loader.instances == ptr_instance) {
                loader.instances = ptr_instance->next;
            }
            if (NULL != ptr_instance->disp) {
                loader_instance_heap_free(ptr_instance, ptr_instance->disp);
            }
            if (ptr_instance->num_tmp_report_callbacks > 0) {
                // Remove temporary VK_EXT_debug_report items
                util_DestroyDebugReportCallbacks(ptr_instance, pAllocator, ptr_instance->num_tmp_report_callbacks,
                                                 ptr_instance->tmp_report_callbacks);
                util_FreeDebugReportCreateInfos(pAllocator, ptr_instance->tmp_report_create_infos,
                                                ptr_instance->tmp_report_callbacks);
            }
            if (ptr_instance->num_tmp_messengers > 0) {
                // Remove temporary VK_EXT_debug_utils items
                util_DestroyDebugUtilsMessengers(ptr_instance, pAllocator, ptr_instance->num_tmp_messengers,
                                                 ptr_instance->tmp_messengers);
                util_FreeDebugUtilsMessengerCreateInfos(pAllocator, ptr_instance->tmp_messenger_create_infos,
                                                        ptr_instance->tmp_messengers);
            }

            if (NULL != ptr_instance->expanded_activated_layer_list.list) {
                loaderDeactivateLayers(ptr_instance, NULL, &ptr_instance->expanded_activated_layer_list);
            }
            if (NULL != ptr_instance->app_activated_layer_list.list) {
                loaderDestroyLayerList(ptr_instance, NULL, &ptr_instance->app_activated_layer_list);
            }

            loaderDeleteLayerListAndProperties(ptr_instance, &ptr_instance->instance_layer_list);
            loader_scanned_icd_clear(ptr_instance, &ptr_instance->icd_tramp_list);
            loader_destroy_generic_list(ptr_instance, (struct loader_generic_list *)&ptr_instance->ext_list);

            loader_instance_heap_free(ptr_instance, ptr_instance);
        } else {
            // Remove temporary VK_EXT_debug_report or VK_EXT_debug_utils items
            util_DestroyDebugUtilsMessengers(ptr_instance, pAllocator, ptr_instance->num_tmp_messengers,
                                             ptr_instance->tmp_messengers);
            util_DestroyDebugReportCallbacks(ptr_instance, pAllocator, ptr_instance->num_tmp_report_callbacks,
                                             ptr_instance->tmp_report_callbacks);
        }

        if (loaderLocked) {
            loader_platform_thread_unlock_mutex(&loader_lock);
        }
    }

    return res;
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyInstance(VkInstance instance, const VkAllocationCallbacks *pAllocator) {
    const VkLayerInstanceDispatchTable *disp;
    struct loader_instance *ptr_instance = NULL;
    bool callback_setup = false;
    bool messenger_setup = false;

    if (instance == VK_NULL_HANDLE) {
        return;
    }

    disp = loader_get_instance_layer_dispatch(instance);

    loader_platform_thread_lock_mutex(&loader_lock);

    ptr_instance = loader_get_instance(instance);

    if (pAllocator) {
        ptr_instance->alloc_callbacks = *pAllocator;
    }

    if (ptr_instance->num_tmp_messengers > 0) {
        // Setup the temporary VK_EXT_debug_utils messenger(s) here to catch cleanup issues:
        if (!util_CreateDebugUtilsMessengers(ptr_instance, pAllocator, ptr_instance->num_tmp_messengers,
                                             ptr_instance->tmp_messenger_create_infos, ptr_instance->tmp_messengers)) {
            messenger_setup = true;
        }
    }

    if (ptr_instance->num_tmp_report_callbacks > 0) {
        // Setup the temporary VK_EXT_debug_report callback(s) here to catch cleanup issues:
        if (!util_CreateDebugReportCallbacks(ptr_instance, pAllocator, ptr_instance->num_tmp_report_callbacks,
                                             ptr_instance->tmp_report_create_infos, ptr_instance->tmp_report_callbacks)) {
            callback_setup = true;
        }
    }

    disp->DestroyInstance(instance, pAllocator);

    if (NULL != ptr_instance->expanded_activated_layer_list.list) {
        loaderDeactivateLayers(ptr_instance, NULL, &ptr_instance->expanded_activated_layer_list);
    }
    if (NULL != ptr_instance->app_activated_layer_list.list) {
        loaderDestroyLayerList(ptr_instance, NULL, &ptr_instance->app_activated_layer_list);
    }

    if (ptr_instance->phys_devs_tramp) {
        for (uint32_t i = 0; i < ptr_instance->phys_dev_count_tramp; i++) {
            loader_instance_heap_free(ptr_instance, ptr_instance->phys_devs_tramp[i]);
        }
        loader_instance_heap_free(ptr_instance, ptr_instance->phys_devs_tramp);
    }

    if (ptr_instance->phys_dev_groups_tramp) {
        for (uint32_t i = 0; i < ptr_instance->phys_dev_group_count_tramp; i++) {
            loader_instance_heap_free(ptr_instance, ptr_instance->phys_dev_groups_tramp[i]);
        }
        loader_instance_heap_free(ptr_instance, ptr_instance->phys_dev_groups_tramp);
    }

    if (messenger_setup) {
        util_DestroyDebugUtilsMessengers(ptr_instance, pAllocator, ptr_instance->num_tmp_messengers, ptr_instance->tmp_messengers);
        util_FreeDebugUtilsMessengerCreateInfos(pAllocator, ptr_instance->tmp_messenger_create_infos, ptr_instance->tmp_messengers);
    }

    if (callback_setup) {
        util_DestroyDebugReportCallbacks(ptr_instance, pAllocator, ptr_instance->num_tmp_report_callbacks,
                                         ptr_instance->tmp_report_callbacks);
        util_FreeDebugReportCreateInfos(pAllocator, ptr_instance->tmp_report_create_infos, ptr_instance->tmp_report_callbacks);
    }

    loader_instance_heap_free(ptr_instance, ptr_instance->disp);
    loader_instance_heap_free(ptr_instance, ptr_instance);
    loader_platform_thread_unlock_mutex(&loader_lock);

    // Unload preloaded layers, so if vkEnumerateInstanceExtensionProperties or vkCreateInstance is called again, the ICD's are up
    // to date
    loader_unload_preloaded_icds();
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumeratePhysicalDevices(VkInstance instance, uint32_t *pPhysicalDeviceCount,
                                                                        VkPhysicalDevice *pPhysicalDevices) {
    VkResult res = VK_SUCCESS;
    uint32_t count;
    uint32_t i;
    struct loader_instance *inst;

    loader_platform_thread_lock_mutex(&loader_lock);

    inst = loader_get_instance(instance);
    if (NULL == inst) {
        res = VK_ERROR_INITIALIZATION_FAILED;
        goto out;
    }

    if (NULL == pPhysicalDeviceCount) {
        loader_log(inst, VK_DEBUG_REPORT_ERROR_BIT_EXT, 0,
                   "vkEnumeratePhysicalDevices: Received NULL pointer for physical device count return value.");
        res = VK_ERROR_INITIALIZATION_FAILED;
        goto out;
    }

    // Setup the trampoline loader physical devices.  This will actually
    // call down and setup the terminator loader physical devices during the
    // process.
    VkResult setup_res = setupLoaderTrampPhysDevs(instance);
    if (setup_res != VK_SUCCESS && setup_res != VK_INCOMPLETE) {
        res = setup_res;
        goto out;
    }

    count = inst->phys_dev_count_tramp;

    // Wrap the PhysDev object for loader usage, return wrapped objects
    if (NULL != pPhysicalDevices) {
        if (inst->phys_dev_count_tramp > *pPhysicalDeviceCount) {
            loader_log(inst, VK_DEBUG_REPORT_INFORMATION_BIT_EXT, 0,
                       "vkEnumeratePhysicalDevices: Trimming device count down"
                       " by application request from %d to %d physical devices",
                       inst->phys_dev_count_tramp, *pPhysicalDeviceCount);
            count = *pPhysicalDeviceCount;
            res = VK_INCOMPLETE;
        }
        for (i = 0; i < count; i++) {
            pPhysicalDevices[i] = (VkPhysicalDevice)inst->phys_devs_tramp[i];
        }
    }

    *pPhysicalDeviceCount = count;

out:

    loader_platform_thread_unlock_mutex(&loader_lock);
    return res;
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures(VkPhysicalDevice physicalDevice,
                                                                     VkPhysicalDeviceFeatures *pFeatures) {
    const VkLayerInstanceDispatchTable *disp;
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    disp = loader_get_instance_layer_dispatch(physicalDevice);
    disp->GetPhysicalDeviceFeatures(unwrapped_phys_dev, pFeatures);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties(VkPhysicalDevice physicalDevice, VkFormat format,
                                                                             VkFormatProperties *pFormatInfo) {
    const VkLayerInstanceDispatchTable *disp;
    VkPhysicalDevice unwrapped_pd = loader_unwrap_physical_device(physicalDevice);
    disp = loader_get_instance_layer_dispatch(physicalDevice);
    disp->GetPhysicalDeviceFormatProperties(unwrapped_pd, format, pFormatInfo);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceImageFormatProperties(
    VkPhysicalDevice physicalDevice, VkFormat format, VkImageType type, VkImageTiling tiling, VkImageUsageFlags usage,
    VkImageCreateFlags flags, VkImageFormatProperties *pImageFormatProperties) {
    const VkLayerInstanceDispatchTable *disp;
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    disp = loader_get_instance_layer_dispatch(physicalDevice);
    return disp->GetPhysicalDeviceImageFormatProperties(unwrapped_phys_dev, format, type, tiling, usage, flags,
                                                        pImageFormatProperties);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties(VkPhysicalDevice physicalDevice,
                                                                       VkPhysicalDeviceProperties *pProperties) {
    const VkLayerInstanceDispatchTable *disp;
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    disp = loader_get_instance_layer_dispatch(physicalDevice);
    disp->GetPhysicalDeviceProperties(unwrapped_phys_dev, pProperties);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice physicalDevice,
                                                                                  uint32_t *pQueueFamilyPropertyCount,
                                                                                  VkQueueFamilyProperties *pQueueProperties) {
    const VkLayerInstanceDispatchTable *disp;
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    disp = loader_get_instance_layer_dispatch(physicalDevice);
    disp->GetPhysicalDeviceQueueFamilyProperties(unwrapped_phys_dev, pQueueFamilyPropertyCount, pQueueProperties);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice physicalDevice,
                                                                             VkPhysicalDeviceMemoryProperties *pMemoryProperties) {
    const VkLayerInstanceDispatchTable *disp;
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    disp = loader_get_instance_layer_dispatch(physicalDevice);
    disp->GetPhysicalDeviceMemoryProperties(unwrapped_phys_dev, pMemoryProperties);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateDevice(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo *pCreateInfo,
                                                            const VkAllocationCallbacks *pAllocator, VkDevice *pDevice) {
    loader_platform_thread_lock_mutex(&loader_lock);
    VkResult res = loader_layer_create_device(NULL, physicalDevice, pCreateInfo, pAllocator, pDevice, NULL, NULL);
    loader_platform_thread_unlock_mutex(&loader_lock);
    return res;
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyDevice(VkDevice device, const VkAllocationCallbacks *pAllocator) {
    const VkLayerDispatchTable *disp;

    if (device == VK_NULL_HANDLE) {
        return;
    }
    disp = loader_get_dispatch(device);

    loader_platform_thread_lock_mutex(&loader_lock);

    loader_layer_destroy_device(device, pAllocator, disp->DestroyDevice);

    loader_platform_thread_unlock_mutex(&loader_lock);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice,
                                                                                  const char *pLayerName, uint32_t *pPropertyCount,
                                                                                  VkExtensionProperties *pProperties) {
    VkResult res = VK_SUCCESS;
    struct loader_physical_device_tramp *phys_dev;
    const VkLayerInstanceDispatchTable *disp;
    phys_dev = (struct loader_physical_device_tramp *)physicalDevice;

    loader_platform_thread_lock_mutex(&loader_lock);

    // always pass this call down the instance chain which will terminate
    // in the ICD. This allows layers to filter the extensions coming back
    // up the chain. In the terminator we look up layer extensions from the
    // manifest file if it wasn't provided by the layer itself.
    disp = loader_get_instance_layer_dispatch(physicalDevice);
    res = disp->EnumerateDeviceExtensionProperties(phys_dev->phys_dev, pLayerName, pPropertyCount, pProperties);

    loader_platform_thread_unlock_mutex(&loader_lock);
    return res;
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(VkPhysicalDevice physicalDevice,
                                                                              uint32_t *pPropertyCount,
                                                                              VkLayerProperties *pProperties) {
    uint32_t copy_size;
    struct loader_physical_device_tramp *phys_dev;
    struct loader_layer_list *enabled_layers, layers_list;
    memset(&layers_list, 0, sizeof(layers_list));
    loader_platform_thread_lock_mutex(&loader_lock);

    // Don't dispatch this call down the instance chain, want all device layers
    // enumerated and instance chain may not contain all device layers
    // TODO re-evaluate the above statement we maybe able to start calling
    // down the chain

    phys_dev = (struct loader_physical_device_tramp *)physicalDevice;
    const struct loader_instance *inst = phys_dev->this_instance;

    uint32_t count = inst->app_activated_layer_list.count;
    if (count == 0 || pProperties == NULL) {
        *pPropertyCount = count;
        loader_platform_thread_unlock_mutex(&loader_lock);
        return VK_SUCCESS;
    }
    enabled_layers = (struct loader_layer_list *)&inst->app_activated_layer_list;

    copy_size = (*pPropertyCount < count) ? *pPropertyCount : count;
    for (uint32_t i = 0; i < copy_size; i++) {
        memcpy(&pProperties[i], &(enabled_layers->list[i].info), sizeof(VkLayerProperties));
    }
    *pPropertyCount = copy_size;

    if (copy_size < count) {
        loader_platform_thread_unlock_mutex(&loader_lock);
        return VK_INCOMPLETE;
    }

    loader_platform_thread_unlock_mutex(&loader_lock);
    return VK_SUCCESS;
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetDeviceQueue(VkDevice device, uint32_t queueNodeIndex, uint32_t queueIndex,
                                                          VkQueue *pQueue) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->GetDeviceQueue(device, queueNodeIndex, queueIndex, pQueue);
    if (pQueue != NULL) {
        loader_set_dispatch(*pQueue, disp);
    }
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo *pSubmits,
                                                           VkFence fence) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(queue);

    return disp->QueueSubmit(queue, submitCount, pSubmits, fence);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkQueueWaitIdle(VkQueue queue) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(queue);

    return disp->QueueWaitIdle(queue);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkDeviceWaitIdle(VkDevice device) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->DeviceWaitIdle(device);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkAllocateMemory(VkDevice device, const VkMemoryAllocateInfo *pAllocateInfo,
                                                              const VkAllocationCallbacks *pAllocator, VkDeviceMemory *pMemory) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->AllocateMemory(device, pAllocateInfo, pAllocator, pMemory);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkFreeMemory(VkDevice device, VkDeviceMemory mem,
                                                      const VkAllocationCallbacks *pAllocator) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->FreeMemory(device, mem, pAllocator);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkMapMemory(VkDevice device, VkDeviceMemory mem, VkDeviceSize offset,
                                                         VkDeviceSize size, VkFlags flags, void **ppData) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->MapMemory(device, mem, offset, size, flags, ppData);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkUnmapMemory(VkDevice device, VkDeviceMemory mem) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->UnmapMemory(device, mem);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkFlushMappedMemoryRanges(VkDevice device, uint32_t memoryRangeCount,
                                                                       const VkMappedMemoryRange *pMemoryRanges) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->FlushMappedMemoryRanges(device, memoryRangeCount, pMemoryRanges);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkInvalidateMappedMemoryRanges(VkDevice device, uint32_t memoryRangeCount,
                                                                            const VkMappedMemoryRange *pMemoryRanges) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->InvalidateMappedMemoryRanges(device, memoryRangeCount, pMemoryRanges);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetDeviceMemoryCommitment(VkDevice device, VkDeviceMemory memory,
                                                                     VkDeviceSize *pCommittedMemoryInBytes) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->GetDeviceMemoryCommitment(device, memory, pCommittedMemoryInBytes);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkBindBufferMemory(VkDevice device, VkBuffer buffer, VkDeviceMemory mem,
                                                                VkDeviceSize offset) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->BindBufferMemory(device, buffer, mem, offset);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkBindImageMemory(VkDevice device, VkImage image, VkDeviceMemory mem,
                                                               VkDeviceSize offset) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->BindImageMemory(device, image, mem, offset);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetBufferMemoryRequirements(VkDevice device, VkBuffer buffer,
                                                                       VkMemoryRequirements *pMemoryRequirements) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->GetBufferMemoryRequirements(device, buffer, pMemoryRequirements);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetImageMemoryRequirements(VkDevice device, VkImage image,
                                                                      VkMemoryRequirements *pMemoryRequirements) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->GetImageMemoryRequirements(device, image, pMemoryRequirements);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL
vkGetImageSparseMemoryRequirements(VkDevice device, VkImage image, uint32_t *pSparseMemoryRequirementCount,
                                   VkSparseImageMemoryRequirements *pSparseMemoryRequirements) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->GetImageSparseMemoryRequirements(device, image, pSparseMemoryRequirementCount, pSparseMemoryRequirements);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceSparseImageFormatProperties(
    VkPhysicalDevice physicalDevice, VkFormat format, VkImageType type, VkSampleCountFlagBits samples, VkImageUsageFlags usage,
    VkImageTiling tiling, uint32_t *pPropertyCount, VkSparseImageFormatProperties *pProperties) {
    const VkLayerInstanceDispatchTable *disp;
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    disp = loader_get_instance_layer_dispatch(physicalDevice);

    disp->GetPhysicalDeviceSparseImageFormatProperties(unwrapped_phys_dev, format, type, samples, usage, tiling, pPropertyCount,
                                                       pProperties);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkQueueBindSparse(VkQueue queue, uint32_t bindInfoCount,
                                                               const VkBindSparseInfo *pBindInfo, VkFence fence) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(queue);

    return disp->QueueBindSparse(queue, bindInfoCount, pBindInfo, fence);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateFence(VkDevice device, const VkFenceCreateInfo *pCreateInfo,
                                                           const VkAllocationCallbacks *pAllocator, VkFence *pFence) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->CreateFence(device, pCreateInfo, pAllocator, pFence);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyFence(VkDevice device, VkFence fence, const VkAllocationCallbacks *pAllocator) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->DestroyFence(device, fence, pAllocator);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkResetFences(VkDevice device, uint32_t fenceCount, const VkFence *pFences) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->ResetFences(device, fenceCount, pFences);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetFenceStatus(VkDevice device, VkFence fence) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->GetFenceStatus(device, fence);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkWaitForFences(VkDevice device, uint32_t fenceCount, const VkFence *pFences,
                                                             VkBool32 waitAll, uint64_t timeout) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->WaitForFences(device, fenceCount, pFences, waitAll, timeout);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateSemaphore(VkDevice device, const VkSemaphoreCreateInfo *pCreateInfo,
                                                               const VkAllocationCallbacks *pAllocator, VkSemaphore *pSemaphore) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->CreateSemaphore(device, pCreateInfo, pAllocator, pSemaphore);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroySemaphore(VkDevice device, VkSemaphore semaphore,
                                                            const VkAllocationCallbacks *pAllocator) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->DestroySemaphore(device, semaphore, pAllocator);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateEvent(VkDevice device, const VkEventCreateInfo *pCreateInfo,
                                                           const VkAllocationCallbacks *pAllocator, VkEvent *pEvent) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->CreateEvent(device, pCreateInfo, pAllocator, pEvent);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyEvent(VkDevice device, VkEvent event, const VkAllocationCallbacks *pAllocator) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->DestroyEvent(device, event, pAllocator);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetEventStatus(VkDevice device, VkEvent event) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->GetEventStatus(device, event);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkSetEvent(VkDevice device, VkEvent event) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->SetEvent(device, event);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkResetEvent(VkDevice device, VkEvent event) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->ResetEvent(device, event);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateQueryPool(VkDevice device, const VkQueryPoolCreateInfo *pCreateInfo,
                                                               const VkAllocationCallbacks *pAllocator, VkQueryPool *pQueryPool) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->CreateQueryPool(device, pCreateInfo, pAllocator, pQueryPool);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyQueryPool(VkDevice device, VkQueryPool queryPool,
                                                            const VkAllocationCallbacks *pAllocator) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->DestroyQueryPool(device, queryPool, pAllocator);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetQueryPoolResults(VkDevice device, VkQueryPool queryPool, uint32_t firstQuery,
                                                                   uint32_t queryCount, size_t dataSize, void *pData,
                                                                   VkDeviceSize stride, VkQueryResultFlags flags) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->GetQueryPoolResults(device, queryPool, firstQuery, queryCount, dataSize, pData, stride, flags);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateBuffer(VkDevice device, const VkBufferCreateInfo *pCreateInfo,
                                                            const VkAllocationCallbacks *pAllocator, VkBuffer *pBuffer) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->CreateBuffer(device, pCreateInfo, pAllocator, pBuffer);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyBuffer(VkDevice device, VkBuffer buffer,
                                                         const VkAllocationCallbacks *pAllocator) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->DestroyBuffer(device, buffer, pAllocator);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateBufferView(VkDevice device, const VkBufferViewCreateInfo *pCreateInfo,
                                                                const VkAllocationCallbacks *pAllocator, VkBufferView *pView) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->CreateBufferView(device, pCreateInfo, pAllocator, pView);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyBufferView(VkDevice device, VkBufferView bufferView,
                                                             const VkAllocationCallbacks *pAllocator) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->DestroyBufferView(device, bufferView, pAllocator);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateImage(VkDevice device, const VkImageCreateInfo *pCreateInfo,
                                                           const VkAllocationCallbacks *pAllocator, VkImage *pImage) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->CreateImage(device, pCreateInfo, pAllocator, pImage);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks *pAllocator) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->DestroyImage(device, image, pAllocator);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetImageSubresourceLayout(VkDevice device, VkImage image,
                                                                     const VkImageSubresource *pSubresource,
                                                                     VkSubresourceLayout *pLayout) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->GetImageSubresourceLayout(device, image, pSubresource, pLayout);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateImageView(VkDevice device, const VkImageViewCreateInfo *pCreateInfo,
                                                               const VkAllocationCallbacks *pAllocator, VkImageView *pView) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->CreateImageView(device, pCreateInfo, pAllocator, pView);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyImageView(VkDevice device, VkImageView imageView,
                                                            const VkAllocationCallbacks *pAllocator) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->DestroyImageView(device, imageView, pAllocator);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo *pCreateInfo,
                                                                  const VkAllocationCallbacks *pAllocator,
                                                                  VkShaderModule *pShader) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->CreateShaderModule(device, pCreateInfo, pAllocator, pShader);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyShaderModule(VkDevice device, VkShaderModule shaderModule,
                                                               const VkAllocationCallbacks *pAllocator) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->DestroyShaderModule(device, shaderModule, pAllocator);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreatePipelineCache(VkDevice device, const VkPipelineCacheCreateInfo *pCreateInfo,
                                                                   const VkAllocationCallbacks *pAllocator,
                                                                   VkPipelineCache *pPipelineCache) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->CreatePipelineCache(device, pCreateInfo, pAllocator, pPipelineCache);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyPipelineCache(VkDevice device, VkPipelineCache pipelineCache,
                                                                const VkAllocationCallbacks *pAllocator) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->DestroyPipelineCache(device, pipelineCache, pAllocator);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPipelineCacheData(VkDevice device, VkPipelineCache pipelineCache,
                                                                    size_t *pDataSize, void *pData) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->GetPipelineCacheData(device, pipelineCache, pDataSize, pData);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkMergePipelineCaches(VkDevice device, VkPipelineCache dstCache,
                                                                   uint32_t srcCacheCount, const VkPipelineCache *pSrcCaches) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->MergePipelineCaches(device, dstCache, srcCacheCount, pSrcCaches);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateGraphicsPipelines(VkDevice device, VkPipelineCache pipelineCache,
                                                                       uint32_t createInfoCount,
                                                                       const VkGraphicsPipelineCreateInfo *pCreateInfos,
                                                                       const VkAllocationCallbacks *pAllocator,
                                                                       VkPipeline *pPipelines) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->CreateGraphicsPipelines(device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateComputePipelines(VkDevice device, VkPipelineCache pipelineCache,
                                                                      uint32_t createInfoCount,
                                                                      const VkComputePipelineCreateInfo *pCreateInfos,
                                                                      const VkAllocationCallbacks *pAllocator,
                                                                      VkPipeline *pPipelines) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->CreateComputePipelines(device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyPipeline(VkDevice device, VkPipeline pipeline,
                                                           const VkAllocationCallbacks *pAllocator) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->DestroyPipeline(device, pipeline, pAllocator);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreatePipelineLayout(VkDevice device, const VkPipelineLayoutCreateInfo *pCreateInfo,
                                                                    const VkAllocationCallbacks *pAllocator,
                                                                    VkPipelineLayout *pPipelineLayout) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->CreatePipelineLayout(device, pCreateInfo, pAllocator, pPipelineLayout);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyPipelineLayout(VkDevice device, VkPipelineLayout pipelineLayout,
                                                                 const VkAllocationCallbacks *pAllocator) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->DestroyPipelineLayout(device, pipelineLayout, pAllocator);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateSampler(VkDevice device, const VkSamplerCreateInfo *pCreateInfo,
                                                             const VkAllocationCallbacks *pAllocator, VkSampler *pSampler) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->CreateSampler(device, pCreateInfo, pAllocator, pSampler);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroySampler(VkDevice device, VkSampler sampler,
                                                          const VkAllocationCallbacks *pAllocator) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->DestroySampler(device, sampler, pAllocator);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorSetLayout(VkDevice device,
                                                                         const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
                                                                         const VkAllocationCallbacks *pAllocator,
                                                                         VkDescriptorSetLayout *pSetLayout) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->CreateDescriptorSetLayout(device, pCreateInfo, pAllocator, pSetLayout);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout,
                                                                      const VkAllocationCallbacks *pAllocator) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->DestroyDescriptorSetLayout(device, descriptorSetLayout, pAllocator);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorPool(VkDevice device, const VkDescriptorPoolCreateInfo *pCreateInfo,
                                                                    const VkAllocationCallbacks *pAllocator,
                                                                    VkDescriptorPool *pDescriptorPool) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->CreateDescriptorPool(device, pCreateInfo, pAllocator, pDescriptorPool);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool,
                                                                 const VkAllocationCallbacks *pAllocator) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->DestroyDescriptorPool(device, descriptorPool, pAllocator);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkResetDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool,
                                                                   VkDescriptorPoolResetFlags flags) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->ResetDescriptorPool(device, descriptorPool, flags);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkAllocateDescriptorSets(VkDevice device,
                                                                      const VkDescriptorSetAllocateInfo *pAllocateInfo,
                                                                      VkDescriptorSet *pDescriptorSets) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->AllocateDescriptorSets(device, pAllocateInfo, pDescriptorSets);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkFreeDescriptorSets(VkDevice device, VkDescriptorPool descriptorPool,
                                                                  uint32_t descriptorSetCount,
                                                                  const VkDescriptorSet *pDescriptorSets) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->FreeDescriptorSets(device, descriptorPool, descriptorSetCount, pDescriptorSets);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkUpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount,
                                                                const VkWriteDescriptorSet *pDescriptorWrites,
                                                                uint32_t descriptorCopyCount,
                                                                const VkCopyDescriptorSet *pDescriptorCopies) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->UpdateDescriptorSets(device, descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateFramebuffer(VkDevice device, const VkFramebufferCreateInfo *pCreateInfo,
                                                                 const VkAllocationCallbacks *pAllocator,
                                                                 VkFramebuffer *pFramebuffer) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->CreateFramebuffer(device, pCreateInfo, pAllocator, pFramebuffer);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyFramebuffer(VkDevice device, VkFramebuffer framebuffer,
                                                              const VkAllocationCallbacks *pAllocator) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->DestroyFramebuffer(device, framebuffer, pAllocator);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateRenderPass(VkDevice device, const VkRenderPassCreateInfo *pCreateInfo,
                                                                const VkAllocationCallbacks *pAllocator,
                                                                VkRenderPass *pRenderPass) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->CreateRenderPass(device, pCreateInfo, pAllocator, pRenderPass);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyRenderPass(VkDevice device, VkRenderPass renderPass,
                                                             const VkAllocationCallbacks *pAllocator) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->DestroyRenderPass(device, renderPass, pAllocator);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetRenderAreaGranularity(VkDevice device, VkRenderPass renderPass,
                                                                    VkExtent2D *pGranularity) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->GetRenderAreaGranularity(device, renderPass, pGranularity);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateCommandPool(VkDevice device, const VkCommandPoolCreateInfo *pCreateInfo,
                                                                 const VkAllocationCallbacks *pAllocator,
                                                                 VkCommandPool *pCommandPool) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->CreateCommandPool(device, pCreateInfo, pAllocator, pCommandPool);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyCommandPool(VkDevice device, VkCommandPool commandPool,
                                                              const VkAllocationCallbacks *pAllocator) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->DestroyCommandPool(device, commandPool, pAllocator);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkResetCommandPool(VkDevice device, VkCommandPool commandPool,
                                                                VkCommandPoolResetFlags flags) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    return disp->ResetCommandPool(device, commandPool, flags);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkAllocateCommandBuffers(VkDevice device,
                                                                      const VkCommandBufferAllocateInfo *pAllocateInfo,
                                                                      VkCommandBuffer *pCommandBuffers) {
    const VkLayerDispatchTable *disp;
    VkResult res;

    disp = loader_get_dispatch(device);

    res = disp->AllocateCommandBuffers(device, pAllocateInfo, pCommandBuffers);
    if (res == VK_SUCCESS) {
        for (uint32_t i = 0; i < pAllocateInfo->commandBufferCount; i++) {
            if (pCommandBuffers[i]) {
                loader_init_dispatch(pCommandBuffers[i], disp);
            }
        }
    }

    return res;
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkFreeCommandBuffers(VkDevice device, VkCommandPool commandPool,
                                                              uint32_t commandBufferCount, const VkCommandBuffer *pCommandBuffers) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(device);

    disp->FreeCommandBuffers(device, commandPool, commandBufferCount, pCommandBuffers);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkBeginCommandBuffer(VkCommandBuffer commandBuffer,
                                                                  const VkCommandBufferBeginInfo *pBeginInfo) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    return disp->BeginCommandBuffer(commandBuffer, pBeginInfo);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEndCommandBuffer(VkCommandBuffer commandBuffer) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    return disp->EndCommandBuffer(commandBuffer);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkResetCommandBuffer(VkCommandBuffer commandBuffer, VkCommandBufferResetFlags flags) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    return disp->ResetCommandBuffer(commandBuffer, flags);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBindPipeline(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                                                           VkPipeline pipeline) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdBindPipeline(commandBuffer, pipelineBindPoint, pipeline);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetViewport(VkCommandBuffer commandBuffer, uint32_t firstViewport,
                                                          uint32_t viewportCount, const VkViewport *pViewports) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdSetViewport(commandBuffer, firstViewport, viewportCount, pViewports);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetScissor(VkCommandBuffer commandBuffer, uint32_t firstScissor,
                                                         uint32_t scissorCount, const VkRect2D *pScissors) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdSetScissor(commandBuffer, firstScissor, scissorCount, pScissors);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetLineWidth(VkCommandBuffer commandBuffer, float lineWidth) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdSetLineWidth(commandBuffer, lineWidth);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthBias(VkCommandBuffer commandBuffer, float depthBiasConstantFactor,
                                                           float depthBiasClamp, float depthBiasSlopeFactor) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdSetDepthBias(commandBuffer, depthBiasConstantFactor, depthBiasClamp, depthBiasSlopeFactor);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetBlendConstants(VkCommandBuffer commandBuffer, const float blendConstants[4]) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdSetBlendConstants(commandBuffer, blendConstants);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthBounds(VkCommandBuffer commandBuffer, float minDepthBounds,
                                                             float maxDepthBounds) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdSetDepthBounds(commandBuffer, minDepthBounds, maxDepthBounds);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetStencilCompareMask(VkCommandBuffer commandBuffer, VkStencilFaceFlags faceMask,
                                                                    uint32_t compareMask) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdSetStencilCompareMask(commandBuffer, faceMask, compareMask);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetStencilWriteMask(VkCommandBuffer commandBuffer, VkStencilFaceFlags faceMask,
                                                                  uint32_t writeMask) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdSetStencilWriteMask(commandBuffer, faceMask, writeMask);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetStencilReference(VkCommandBuffer commandBuffer, VkStencilFaceFlags faceMask,
                                                                  uint32_t reference) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdSetStencilReference(commandBuffer, faceMask, reference);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBindDescriptorSets(VkCommandBuffer commandBuffer,
                                                                 VkPipelineBindPoint pipelineBindPoint, VkPipelineLayout layout,
                                                                 uint32_t firstSet, uint32_t descriptorSetCount,
                                                                 const VkDescriptorSet *pDescriptorSets,
                                                                 uint32_t dynamicOffsetCount, const uint32_t *pDynamicOffsets) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdBindDescriptorSets(commandBuffer, pipelineBindPoint, layout, firstSet, descriptorSetCount, pDescriptorSets,
                                dynamicOffsetCount, pDynamicOffsets);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBindIndexBuffer(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                              VkIndexType indexType) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdBindIndexBuffer(commandBuffer, buffer, offset, indexType);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBindVertexBuffers(VkCommandBuffer commandBuffer, uint32_t firstBinding,
                                                                uint32_t bindingCount, const VkBuffer *pBuffers,
                                                                const VkDeviceSize *pOffsets) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdBindVertexBuffers(commandBuffer, firstBinding, bindingCount, pBuffers, pOffsets);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount,
                                                   uint32_t firstVertex, uint32_t firstInstance) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdDraw(commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount,
                                                          uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset,
                                                          uint32_t firstInstance) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdDrawIndexed(commandBuffer, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                           uint32_t drawCount, uint32_t stride) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdDrawIndirect(commandBuffer, buffer, offset, drawCount, stride);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer,
                                                                  VkDeviceSize offset, uint32_t drawCount, uint32_t stride) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdDrawIndexedIndirect(commandBuffer, buffer, offset, drawCount, stride);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDispatch(VkCommandBuffer commandBuffer, uint32_t x, uint32_t y, uint32_t z) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdDispatch(commandBuffer, x, y, z);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer,
                                                               VkDeviceSize offset) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdDispatchIndirect(commandBuffer, buffer, offset);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer,
                                                         uint32_t regionCount, const VkBufferCopy *pRegions) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, regionCount, pRegions);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyImage(VkCommandBuffer commandBuffer, VkImage srcImage,
                                                        VkImageLayout srcImageLayout, VkImage dstImage,
                                                        VkImageLayout dstImageLayout, uint32_t regionCount,
                                                        const VkImageCopy *pRegions) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdCopyImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBlitImage(VkCommandBuffer commandBuffer, VkImage srcImage,
                                                        VkImageLayout srcImageLayout, VkImage dstImage,
                                                        VkImageLayout dstImageLayout, uint32_t regionCount,
                                                        const VkImageBlit *pRegions, VkFilter filter) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdBlitImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions, filter);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyBufferToImage(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkImage dstImage,
                                                                VkImageLayout dstImageLayout, uint32_t regionCount,
                                                                const VkBufferImageCopy *pRegions) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdCopyBufferToImage(commandBuffer, srcBuffer, dstImage, dstImageLayout, regionCount, pRegions);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyImageToBuffer(VkCommandBuffer commandBuffer, VkImage srcImage,
                                                                VkImageLayout srcImageLayout, VkBuffer dstBuffer,
                                                                uint32_t regionCount, const VkBufferImageCopy *pRegions) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdCopyImageToBuffer(commandBuffer, srcImage, srcImageLayout, dstBuffer, regionCount, pRegions);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdUpdateBuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer,
                                                           VkDeviceSize dstOffset, VkDeviceSize dataSize, const void *pData) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdUpdateBuffer(commandBuffer, dstBuffer, dstOffset, dataSize, pData);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdFillBuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset,
                                                         VkDeviceSize size, uint32_t data) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdFillBuffer(commandBuffer, dstBuffer, dstOffset, size, data);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdClearColorImage(VkCommandBuffer commandBuffer, VkImage image,
                                                              VkImageLayout imageLayout, const VkClearColorValue *pColor,
                                                              uint32_t rangeCount, const VkImageSubresourceRange *pRanges) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdClearColorImage(commandBuffer, image, imageLayout, pColor, rangeCount, pRanges);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdClearDepthStencilImage(VkCommandBuffer commandBuffer, VkImage image,
                                                                     VkImageLayout imageLayout,
                                                                     const VkClearDepthStencilValue *pDepthStencil,
                                                                     uint32_t rangeCount, const VkImageSubresourceRange *pRanges) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdClearDepthStencilImage(commandBuffer, image, imageLayout, pDepthStencil, rangeCount, pRanges);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdClearAttachments(VkCommandBuffer commandBuffer, uint32_t attachmentCount,
                                                               const VkClearAttachment *pAttachments, uint32_t rectCount,
                                                               const VkClearRect *pRects) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdClearAttachments(commandBuffer, attachmentCount, pAttachments, rectCount, pRects);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdResolveImage(VkCommandBuffer commandBuffer, VkImage srcImage,
                                                           VkImageLayout srcImageLayout, VkImage dstImage,
                                                           VkImageLayout dstImageLayout, uint32_t regionCount,
                                                           const VkImageResolve *pRegions) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdResolveImage(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetEvent(VkCommandBuffer commandBuffer, VkEvent event,
                                                       VkPipelineStageFlags stageMask) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdSetEvent(commandBuffer, event, stageMask);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdResetEvent(VkCommandBuffer commandBuffer, VkEvent event,
                                                         VkPipelineStageFlags stageMask) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdResetEvent(commandBuffer, event, stageMask);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdWaitEvents(VkCommandBuffer commandBuffer, uint32_t eventCount, const VkEvent *pEvents,
                                                         VkPipelineStageFlags sourceStageMask, VkPipelineStageFlags dstStageMask,
                                                         uint32_t memoryBarrierCount, const VkMemoryBarrier *pMemoryBarriers,
                                                         uint32_t bufferMemoryBarrierCount,
                                                         const VkBufferMemoryBarrier *pBufferMemoryBarriers,
                                                         uint32_t imageMemoryBarrierCount,
                                                         const VkImageMemoryBarrier *pImageMemoryBarriers) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdWaitEvents(commandBuffer, eventCount, pEvents, sourceStageMask, dstStageMask, memoryBarrierCount, pMemoryBarriers,
                        bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier(VkCommandBuffer commandBuffer, VkPipelineStageFlags srcStageMask,
                                                              VkPipelineStageFlags dstStageMask, VkDependencyFlags dependencyFlags,
                                                              uint32_t memoryBarrierCount, const VkMemoryBarrier *pMemoryBarriers,
                                                              uint32_t bufferMemoryBarrierCount,
                                                              const VkBufferMemoryBarrier *pBufferMemoryBarriers,
                                                              uint32_t imageMemoryBarrierCount,
                                                              const VkImageMemoryBarrier *pImageMemoryBarriers) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask, dependencyFlags, memoryBarrierCount, pMemoryBarriers,
                             bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBeginQuery(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t slot,
                                                         VkFlags flags) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdBeginQuery(commandBuffer, queryPool, slot, flags);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdEndQuery(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t slot) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdEndQuery(commandBuffer, queryPool, slot);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdResetQueryPool(VkCommandBuffer commandBuffer, VkQueryPool queryPool,
                                                             uint32_t firstQuery, uint32_t queryCount) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdResetQueryPool(commandBuffer, queryPool, firstQuery, queryCount);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdWriteTimestamp(VkCommandBuffer commandBuffer, VkPipelineStageFlagBits pipelineStage,
                                                             VkQueryPool queryPool, uint32_t slot) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdWriteTimestamp(commandBuffer, pipelineStage, queryPool, slot);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdCopyQueryPoolResults(VkCommandBuffer commandBuffer, VkQueryPool queryPool,
                                                                   uint32_t firstQuery, uint32_t queryCount, VkBuffer dstBuffer,
                                                                   VkDeviceSize dstOffset, VkDeviceSize stride, VkFlags flags) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdCopyQueryPoolResults(commandBuffer, queryPool, firstQuery, queryCount, dstBuffer, dstOffset, stride, flags);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdPushConstants(VkCommandBuffer commandBuffer, VkPipelineLayout layout,
                                                            VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size,
                                                            const void *pValues) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdPushConstants(commandBuffer, layout, stageFlags, offset, size, pValues);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBeginRenderPass(VkCommandBuffer commandBuffer,
                                                              const VkRenderPassBeginInfo *pRenderPassBegin,
                                                              VkSubpassContents contents) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdBeginRenderPass(commandBuffer, pRenderPassBegin, contents);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdNextSubpass(VkCommandBuffer commandBuffer, VkSubpassContents contents) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdNextSubpass(commandBuffer, contents);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdEndRenderPass(VkCommandBuffer commandBuffer) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdEndRenderPass(commandBuffer);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdExecuteCommands(VkCommandBuffer commandBuffer, uint32_t commandBuffersCount,
                                                              const VkCommandBuffer *pCommandBuffers) {
    const VkLayerDispatchTable *disp;

    disp = loader_get_dispatch(commandBuffer);

    disp->CmdExecuteCommands(commandBuffer, commandBuffersCount, pCommandBuffers);
}

// ---- Vulkan core 1.1 trampolines

VkResult setupLoaderTrampPhysDevGroups(VkInstance instance) {
    VkResult res = VK_SUCCESS;
    struct loader_instance *inst;
    uint32_t total_count = 0;
    VkPhysicalDeviceGroupPropertiesKHR **new_phys_dev_groups = NULL;
    VkPhysicalDeviceGroupPropertiesKHR *local_phys_dev_groups = NULL;
    PFN_vkEnumeratePhysicalDeviceGroups fpEnumeratePhysicalDeviceGroups = NULL;

    inst = loader_get_instance(instance);
    if (NULL == inst) {
        res = VK_ERROR_INITIALIZATION_FAILED;
        goto out;
    }

    // Get the function pointer to use to call into the ICD. This could be the core or KHR version
    if (inst->enabled_known_extensions.khr_device_group_creation) {
        fpEnumeratePhysicalDeviceGroups = inst->disp->layer_inst_disp.EnumeratePhysicalDeviceGroupsKHR;
    } else {
        fpEnumeratePhysicalDeviceGroups = inst->disp->layer_inst_disp.EnumeratePhysicalDeviceGroups;
    }

    // Setup the trampoline loader physical devices.  This will actually
    // call down and setup the terminator loader physical devices during the
    // process.
    VkResult setup_res = setupLoaderTrampPhysDevs(instance);
    if (setup_res != VK_SUCCESS && setup_res != VK_INCOMPLETE) {
        res = setup_res;
        goto out;
    }

    // Query how many physical device groups there
    res = fpEnumeratePhysicalDeviceGroups(instance, &total_count, NULL);
    if (res != VK_SUCCESS) {
        loader_log(inst, VK_DEBUG_REPORT_ERROR_BIT_EXT, 0,
            "setupLoaderTrampPhysDevGroups:  Failed during dispatch call of "
            "\'EnumeratePhysicalDeviceGroupsKHR\' to lower layers or "
            "loader to get count.");
        goto out;
    }

    // Create an array for the new physical device groups, which will be stored
    // in the instance for the trampoline code.
    new_phys_dev_groups = (VkPhysicalDeviceGroupPropertiesKHR **)loader_instance_heap_alloc(
        inst, total_count * sizeof(VkPhysicalDeviceGroupPropertiesKHR *), VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
    if (NULL == new_phys_dev_groups) {
        loader_log(inst, VK_DEBUG_REPORT_ERROR_BIT_EXT, 0,
            "setupLoaderTrampPhysDevGroups:  Failed to allocate new physical device"
            " group array of size %d",
            total_count);
        res = VK_ERROR_OUT_OF_HOST_MEMORY;
        goto out;
    }
    memset(new_phys_dev_groups, 0, total_count * sizeof(VkPhysicalDeviceGroupPropertiesKHR *));

    // Create a temporary array (on the stack) to keep track of the
    // returned VkPhysicalDevice values.
    local_phys_dev_groups = loader_stack_alloc(sizeof(VkPhysicalDeviceGroupPropertiesKHR) * total_count);
    if (NULL == local_phys_dev_groups) {
        loader_log(inst, VK_DEBUG_REPORT_ERROR_BIT_EXT, 0,
            "setupLoaderTrampPhysDevGroups:  Failed to allocate local "
            "physical device group array of size %d",
            total_count);
        res = VK_ERROR_OUT_OF_HOST_MEMORY;
        goto out;
    }
    // Initialize the memory to something valid
    memset(local_phys_dev_groups, 0, sizeof(VkPhysicalDeviceGroupPropertiesKHR) * total_count);
    for (uint32_t group = 0; group < total_count; group++) {
        local_phys_dev_groups[group].sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES_KHR;
        local_phys_dev_groups[group].pNext = NULL;
        local_phys_dev_groups[group].subsetAllocation = false;
    }

    // Call down and get the content
    fpEnumeratePhysicalDeviceGroups(instance, &total_count, local_phys_dev_groups);
    if (VK_SUCCESS != res) {
        loader_log(inst, VK_DEBUG_REPORT_ERROR_BIT_EXT, 0,
            "setupLoaderTrampPhysDevGroups:  Failed during dispatch call of "
            "\'EnumeratePhysicalDeviceGroupsKHR\' to lower layers or "
            "loader to get content.");
        goto out;
    }

    // Replace all the physical device IDs with the proper loader values
    for (uint32_t group = 0; group < total_count; group++) {
        for (uint32_t group_gpu = 0; group_gpu < local_phys_dev_groups[group].physicalDeviceCount; group_gpu++) {
            bool found = false;
            for (uint32_t tramp_gpu = 0; tramp_gpu < inst->phys_dev_count_tramp; tramp_gpu++) {
                if (local_phys_dev_groups[group].physicalDevices[group_gpu] == inst->phys_devs_tramp[tramp_gpu]->phys_dev) {
                    local_phys_dev_groups[group].physicalDevices[group_gpu] = (VkPhysicalDevice)inst->phys_devs_tramp[tramp_gpu];
                    found = true;
                    break;
                }
            }
            if (!found) {
                loader_log(inst, VK_DEBUG_REPORT_ERROR_BIT_EXT, 0,
                    "setupLoaderTrampPhysDevGroups:  Failed to find GPU %d in group %d"
                    " returned by \'EnumeratePhysicalDeviceGroupsKHR\' in list returned"
                    " by \'EnumeratePhysicalDevices\'", group_gpu, group);
                res = VK_ERROR_INITIALIZATION_FAILED;
                goto out;
            }
        }
    }

    // Copy or create everything to fill the new array of physical device groups
    for (uint32_t new_idx = 0; new_idx < total_count; new_idx++) {
        // Check if this physical device group with the same contents is already in the old buffer
        for (uint32_t old_idx = 0; old_idx < inst->phys_dev_group_count_tramp; old_idx++) {
            if (local_phys_dev_groups[new_idx].physicalDeviceCount == inst->phys_dev_groups_tramp[old_idx]->physicalDeviceCount) {
                bool found_all_gpus = true;
                for (uint32_t old_gpu = 0; old_gpu < inst->phys_dev_groups_tramp[old_idx]->physicalDeviceCount; old_gpu++) {
                    bool found_gpu = false;
                    for (uint32_t new_gpu = 0; new_gpu < local_phys_dev_groups[new_idx].physicalDeviceCount; new_gpu++) {
                        if (local_phys_dev_groups[new_idx].physicalDevices[new_gpu] == inst->phys_dev_groups_tramp[old_idx]->physicalDevices[old_gpu]) {
                            found_gpu = true;
                            break;
                        }
                    }

                    if (!found_gpu) {
                        found_all_gpus = false;
                        break;
                    }
                }
                if (!found_all_gpus) {
                    continue;
                } else {
                    new_phys_dev_groups[new_idx] = inst->phys_dev_groups_tramp[old_idx];
                    break;
                }
            }
        }

        // If this physical device group isn't in the old buffer, create it
        if (NULL == new_phys_dev_groups[new_idx]) {
            new_phys_dev_groups[new_idx] = (VkPhysicalDeviceGroupPropertiesKHR *)loader_instance_heap_alloc(
                inst, sizeof(VkPhysicalDeviceGroupPropertiesKHR), VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
            if (NULL == new_phys_dev_groups[new_idx]) {
                loader_log(inst, VK_DEBUG_REPORT_ERROR_BIT_EXT, 0,
                    "setupLoaderTrampPhysDevGroups:  Failed to allocate "
                    "physical device group trampoline object %d",
                    new_idx);
                total_count = new_idx;
                res = VK_ERROR_OUT_OF_HOST_MEMORY;
                goto out;
            }
            memcpy(new_phys_dev_groups[new_idx], &local_phys_dev_groups[new_idx],
                sizeof(VkPhysicalDeviceGroupPropertiesKHR));
        }
    }

out:

    if (VK_SUCCESS != res) {
        if (NULL != new_phys_dev_groups) {
            for (uint32_t i = 0; i < total_count; i++) {
                loader_instance_heap_free(inst, new_phys_dev_groups[i]);
            }
            loader_instance_heap_free(inst, new_phys_dev_groups);
        }
        total_count = 0;
    } else {
        // Free everything that didn't carry over to the new array of
        // physical device groups
        if (NULL != inst->phys_dev_groups_tramp) {
            for (uint32_t i = 0; i < inst->phys_dev_group_count_tramp; i++) {
                bool found = false;
                for (uint32_t j = 0; j < total_count; j++) {
                    if (inst->phys_dev_groups_tramp[i] == new_phys_dev_groups[j]) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    loader_instance_heap_free(inst, inst->phys_dev_groups_tramp[i]);
                }
            }
            loader_instance_heap_free(inst, inst->phys_dev_groups_tramp);
        }

        // Swap in the new physical device group list
        inst->phys_dev_group_count_tramp = total_count;
        inst->phys_dev_groups_tramp = new_phys_dev_groups;
    }

    return res;
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumeratePhysicalDeviceGroups(
    VkInstance instance, uint32_t *pPhysicalDeviceGroupCount,
    VkPhysicalDeviceGroupProperties *pPhysicalDeviceGroupProperties) {
    VkResult res = VK_SUCCESS;
    uint32_t count;
    uint32_t i;
    struct loader_instance *inst = NULL;

    loader_platform_thread_lock_mutex(&loader_lock);

    inst = loader_get_instance(instance);
    if (NULL == inst) {
        res = VK_ERROR_INITIALIZATION_FAILED;
        goto out;
    }

    if (NULL == pPhysicalDeviceGroupCount) {
        loader_log(inst, VK_DEBUG_REPORT_ERROR_BIT_EXT, 0,
                   "vkEnumeratePhysicalDeviceGroupsKHR: Received NULL pointer for physical "
                   "device group count return value.");
        res = VK_ERROR_INITIALIZATION_FAILED;
        goto out;
    }

    VkResult setup_res = setupLoaderTrampPhysDevGroups(instance);
    if (VK_SUCCESS != setup_res) {
        res = setup_res;
        goto out;
    }

    count = inst->phys_dev_group_count_tramp;

    // Wrap the PhysDev object for loader usage, return wrapped objects
    if (NULL != pPhysicalDeviceGroupProperties) {
        if (inst->phys_dev_group_count_tramp > *pPhysicalDeviceGroupCount) {
            loader_log(inst, VK_DEBUG_REPORT_INFORMATION_BIT_EXT, 0,
                       "vkEnumeratePhysicalDeviceGroupsKHR: Trimming device group count down"
                       " by application request from %d to %d physical device groups",
                       inst->phys_dev_group_count_tramp, *pPhysicalDeviceGroupCount);
            count = *pPhysicalDeviceGroupCount;
            res = VK_INCOMPLETE;
        }
        for (i = 0; i < count; i++) {
            memcpy(&pPhysicalDeviceGroupProperties[i], inst->phys_dev_groups_tramp[i],
                   sizeof(VkPhysicalDeviceGroupPropertiesKHR));
        }
    }

    *pPhysicalDeviceGroupCount = count;

out:

    loader_platform_thread_unlock_mutex(&loader_lock);
    return res;
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2 *pFeatures) {
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    const VkLayerInstanceDispatchTable *disp = loader_get_instance_layer_dispatch(physicalDevice);
    const struct loader_instance *inst = ((struct loader_physical_device_tramp*) physicalDevice)->this_instance;

    if (inst != NULL && inst->enabled_known_extensions.khr_get_physical_device_properties2) {
        disp->GetPhysicalDeviceFeatures2KHR(unwrapped_phys_dev, pFeatures);
    } else {
        disp->GetPhysicalDeviceFeatures2(unwrapped_phys_dev, pFeatures);
    }
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties2(VkPhysicalDevice physicalDevice,
                                                           VkPhysicalDeviceProperties2 *pProperties) {
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    const VkLayerInstanceDispatchTable *disp = loader_get_instance_layer_dispatch(physicalDevice);
    const struct loader_instance *inst = ((struct loader_physical_device_tramp*) physicalDevice)->this_instance;

    if (inst != NULL && inst->enabled_known_extensions.khr_get_physical_device_properties2) {
        disp->GetPhysicalDeviceProperties2KHR(unwrapped_phys_dev, pProperties);
    } else {
        disp->GetPhysicalDeviceProperties2(unwrapped_phys_dev, pProperties);
    }
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties2(VkPhysicalDevice physicalDevice, VkFormat format,
                                                                 VkFormatProperties2 *pFormatProperties) {
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    const VkLayerInstanceDispatchTable *disp = loader_get_instance_layer_dispatch(physicalDevice);
    const struct loader_instance *inst = ((struct loader_physical_device_tramp*) physicalDevice)->this_instance;

    if (inst != NULL && inst->enabled_known_extensions.khr_get_physical_device_properties2) {
        disp->GetPhysicalDeviceFormatProperties2KHR(unwrapped_phys_dev, format, pFormatProperties);
    } else {
        disp->GetPhysicalDeviceFormatProperties2(unwrapped_phys_dev, format, pFormatProperties);
    }
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceImageFormatProperties2(
    VkPhysicalDevice physicalDevice, const VkPhysicalDeviceImageFormatInfo2 *pImageFormatInfo,
    VkImageFormatProperties2 *pImageFormatProperties) {
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    const VkLayerInstanceDispatchTable *disp = loader_get_instance_layer_dispatch(physicalDevice);
    const struct loader_instance *inst = ((struct loader_physical_device_tramp*) physicalDevice)->this_instance;
    
    if (inst != NULL && inst->enabled_known_extensions.khr_get_physical_device_properties2) {
        return disp->GetPhysicalDeviceImageFormatProperties2KHR(unwrapped_phys_dev, pImageFormatInfo, pImageFormatProperties);
    } else {
        return disp->GetPhysicalDeviceImageFormatProperties2(unwrapped_phys_dev, pImageFormatInfo, pImageFormatProperties);
    }
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties2(VkPhysicalDevice physicalDevice,
                                                                      uint32_t *pQueueFamilyPropertyCount,
                                                                      VkQueueFamilyProperties2 *pQueueFamilyProperties) {
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    const VkLayerInstanceDispatchTable *disp = loader_get_instance_layer_dispatch(physicalDevice);
    const struct loader_instance *inst = ((struct loader_physical_device_tramp*) physicalDevice)->this_instance;

    if (inst != NULL && inst->enabled_known_extensions.khr_get_physical_device_properties2) {
        disp->GetPhysicalDeviceQueueFamilyProperties2KHR(unwrapped_phys_dev, pQueueFamilyPropertyCount, pQueueFamilyProperties);
    } else {
        disp->GetPhysicalDeviceQueueFamilyProperties2(unwrapped_phys_dev, pQueueFamilyPropertyCount, pQueueFamilyProperties);
    }
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties2(VkPhysicalDevice physicalDevice,
                                                                 VkPhysicalDeviceMemoryProperties2 *pMemoryProperties) {
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    const VkLayerInstanceDispatchTable *disp = loader_get_instance_layer_dispatch(physicalDevice);
    const struct loader_instance *inst = ((struct loader_physical_device_tramp*) physicalDevice)->this_instance;

    if (inst != NULL && inst->enabled_known_extensions.khr_get_physical_device_properties2) {
        disp->GetPhysicalDeviceMemoryProperties2KHR(unwrapped_phys_dev, pMemoryProperties);
    } else {
        disp->GetPhysicalDeviceMemoryProperties2(unwrapped_phys_dev, pMemoryProperties);
    }
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceSparseImageFormatProperties2(
    VkPhysicalDevice physicalDevice, const VkPhysicalDeviceSparseImageFormatInfo2 *pFormatInfo, uint32_t *pPropertyCount,
    VkSparseImageFormatProperties2 *pProperties) {
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    const VkLayerInstanceDispatchTable *disp = loader_get_instance_layer_dispatch(physicalDevice);
    const struct loader_instance *inst = ((struct loader_physical_device_tramp*) physicalDevice)->this_instance;

    if (inst != NULL && inst->enabled_known_extensions.khr_get_physical_device_properties2) {
        disp->GetPhysicalDeviceSparseImageFormatProperties2KHR(unwrapped_phys_dev, pFormatInfo, pPropertyCount, pProperties);
    } else {
        disp->GetPhysicalDeviceSparseImageFormatProperties2(unwrapped_phys_dev, pFormatInfo, pPropertyCount, pProperties);
    }
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceExternalBufferProperties(
    VkPhysicalDevice physicalDevice, const VkPhysicalDeviceExternalBufferInfo *pExternalBufferInfo,
    VkExternalBufferProperties *pExternalBufferProperties) {
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    const VkLayerInstanceDispatchTable *disp = loader_get_instance_layer_dispatch(physicalDevice);
    const struct loader_instance *inst = ((struct loader_physical_device_tramp*) physicalDevice)->this_instance;

    if (inst != NULL && inst->enabled_known_extensions.khr_external_memory_capabilities){
        disp->GetPhysicalDeviceExternalBufferPropertiesKHR(unwrapped_phys_dev, pExternalBufferInfo, pExternalBufferProperties);
    } else {
        disp->GetPhysicalDeviceExternalBufferProperties(unwrapped_phys_dev, pExternalBufferInfo, pExternalBufferProperties);
    }
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceExternalSemaphoreProperties(
    VkPhysicalDevice physicalDevice, const VkPhysicalDeviceExternalSemaphoreInfoKHR *pExternalSemaphoreInfo,
    VkExternalSemaphoreProperties *pExternalSemaphoreProperties) {
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    const VkLayerInstanceDispatchTable *disp = loader_get_instance_layer_dispatch(physicalDevice);
    const struct loader_instance *inst = ((struct loader_physical_device_tramp*) physicalDevice)->this_instance;

    if (inst != NULL && inst->enabled_known_extensions.khr_external_semaphore_capabilities) {
        disp->GetPhysicalDeviceExternalSemaphorePropertiesKHR(unwrapped_phys_dev, pExternalSemaphoreInfo, pExternalSemaphoreProperties);
    } else {
        disp->GetPhysicalDeviceExternalSemaphoreProperties(unwrapped_phys_dev, pExternalSemaphoreInfo, pExternalSemaphoreProperties);
    }
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceExternalFenceProperties(
    VkPhysicalDevice physicalDevice, const VkPhysicalDeviceExternalFenceInfo *pExternalFenceInfo,
    VkExternalFenceProperties *pExternalFenceProperties) {
    VkPhysicalDevice unwrapped_phys_dev = loader_unwrap_physical_device(physicalDevice);
    const VkLayerInstanceDispatchTable *disp = loader_get_instance_layer_dispatch(physicalDevice);
    const struct loader_instance *inst = ((struct loader_physical_device_tramp*) physicalDevice)->this_instance;

    if (inst != NULL && inst->enabled_known_extensions.khr_external_fence_capabilities) {
        disp->GetPhysicalDeviceExternalFencePropertiesKHR(unwrapped_phys_dev, pExternalFenceInfo, pExternalFenceProperties);
    } else {
        disp->GetPhysicalDeviceExternalFenceProperties(unwrapped_phys_dev, pExternalFenceInfo, pExternalFenceProperties);
    }
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkBindBufferMemory2(
    VkDevice                                    device,
    uint32_t                                    bindInfoCount,
    const VkBindBufferMemoryInfo*               pBindInfos) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->BindBufferMemory2(device, bindInfoCount, pBindInfos);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkBindImageMemory2(
    VkDevice                                    device,
    uint32_t                                    bindInfoCount,
    const VkBindImageMemoryInfo*                pBindInfos) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->BindImageMemory2(device, bindInfoCount, pBindInfos);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetDeviceGroupPeerMemoryFeatures(
    VkDevice                                    device,
    uint32_t                                    heapIndex,
    uint32_t                                    localDeviceIndex,
    uint32_t                                    remoteDeviceIndex,
    VkPeerMemoryFeatureFlags*                   pPeerMemoryFeatures) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->GetDeviceGroupPeerMemoryFeatures(device, heapIndex, localDeviceIndex, remoteDeviceIndex, pPeerMemoryFeatures);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdSetDeviceMask(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    deviceMask) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdSetDeviceMask(commandBuffer, deviceMask);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDispatchBase(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    baseGroupX,
    uint32_t                                    baseGroupY,
    uint32_t                                    baseGroupZ,
    uint32_t                                    groupCountX,
    uint32_t                                    groupCountY,
    uint32_t                                    groupCountZ) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdDispatchBase(commandBuffer, baseGroupX, baseGroupY, baseGroupZ, groupCountX, groupCountY, groupCountZ);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetImageMemoryRequirements2(
    VkDevice                                    device,
    const VkImageMemoryRequirementsInfo2*       pInfo,
    VkMemoryRequirements2*                      pMemoryRequirements) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->GetImageMemoryRequirements2(device, pInfo, pMemoryRequirements);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetBufferMemoryRequirements2(
    VkDevice                                    device,
    const VkBufferMemoryRequirementsInfo2*      pInfo,
    VkMemoryRequirements2*                      pMemoryRequirements) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->GetBufferMemoryRequirements2(device, pInfo, pMemoryRequirements);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetImageSparseMemoryRequirements2(
    VkDevice                                    device,
    const VkImageSparseMemoryRequirementsInfo2* pInfo,
    uint32_t*                                   pSparseMemoryRequirementCount,
    VkSparseImageMemoryRequirements2*           pSparseMemoryRequirements) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->GetImageSparseMemoryRequirements2(device, pInfo, pSparseMemoryRequirementCount, pSparseMemoryRequirements);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkTrimCommandPool(
    VkDevice                                    device,
    VkCommandPool                               commandPool,
    VkCommandPoolTrimFlags                      flags) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->TrimCommandPool(device, commandPool, flags);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetDeviceQueue2(VkDevice device, const VkDeviceQueueInfo2 *pQueueInfo, VkQueue *pQueue) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->GetDeviceQueue2(device, pQueueInfo, pQueue);
    if (pQueue != NULL && *pQueue != NULL) {
        loader_set_dispatch(*pQueue, disp);
    }
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateSamplerYcbcrConversion(
    VkDevice                                    device,
    const VkSamplerYcbcrConversionCreateInfo*   pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSamplerYcbcrConversion*                   pYcbcrConversion) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->CreateSamplerYcbcrConversion(device, pCreateInfo, pAllocator, pYcbcrConversion);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroySamplerYcbcrConversion(
    VkDevice                                    device,
    VkSamplerYcbcrConversion                    ycbcrConversion,
    const VkAllocationCallbacks*                pAllocator) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->DestroySamplerYcbcrConversion(device, ycbcrConversion, pAllocator);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetDescriptorSetLayoutSupport(
    VkDevice                                    device,
    const VkDescriptorSetLayoutCreateInfo*      pCreateInfo,
    VkDescriptorSetLayoutSupport*               pSupport) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->GetDescriptorSetLayoutSupport(device, pCreateInfo, pSupport);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
vkCreateDescriptorUpdateTemplate(VkDevice device, const VkDescriptorUpdateTemplateCreateInfo *pCreateInfo,
                                 const VkAllocationCallbacks *pAllocator, VkDescriptorUpdateTemplate *pDescriptorUpdateTemplate) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->CreateDescriptorUpdateTemplate(device, pCreateInfo, pAllocator, pDescriptorUpdateTemplate);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorUpdateTemplate(VkDevice device,
                                                                           VkDescriptorUpdateTemplate descriptorUpdateTemplate,
                                                                           const VkAllocationCallbacks *pAllocator) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->DestroyDescriptorUpdateTemplate(device, descriptorUpdateTemplate, pAllocator);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkUpdateDescriptorSetWithTemplate(VkDevice device, VkDescriptorSet descriptorSet,
                                                                           VkDescriptorUpdateTemplate descriptorUpdateTemplate,
                                                                           const void *pData) {
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->UpdateDescriptorSetWithTemplate(device, descriptorSet, descriptorUpdateTemplate, pData);
}

// ---- Vulkan core 1.2 trampolines

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateRenderPass2(VkDevice device, const VkRenderPassCreateInfo2* pCreateInfo,
                                                                 const VkAllocationCallbacks* pAllocator, VkRenderPass* pRenderPass)
{
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->CreateRenderPass2(device, pCreateInfo, pAllocator, pRenderPass);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdBeginRenderPass2(VkCommandBuffer commandBuffer,
                                                               const VkRenderPassBeginInfo* pRenderPassBegin,
                                                               const VkSubpassBeginInfo* pSubpassBeginInfo)
{
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdBeginRenderPass2(commandBuffer, pRenderPassBegin, pSubpassBeginInfo);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdNextSubpass2(VkCommandBuffer commandBuffer,
                                                           const VkSubpassBeginInfo* pSubpassBeginInfo,
                                                           const VkSubpassEndInfo* pSubpassEndInfo)
{
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdNextSubpass2(commandBuffer, pSubpassBeginInfo, pSubpassEndInfo);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdEndRenderPass2(VkCommandBuffer commandBuffer, const VkSubpassEndInfo* pSubpassEndInfo)
{
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdEndRenderPass2(commandBuffer, pSubpassEndInfo);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndirectCount(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                                VkBuffer countBuffer, VkDeviceSize countBufferOffset,
                                                                uint32_t maxDrawCount, uint32_t stride)
{
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdDrawIndirectCount(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexedIndirectCount(VkCommandBuffer commandBuffer, VkBuffer buffer,
                                                                       VkDeviceSize offset, VkBuffer countBuffer,
                                                                       VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                                                                       uint32_t stride)
{
    const VkLayerDispatchTable *disp = loader_get_dispatch(commandBuffer);
    disp->CmdDrawIndexedIndirectCount(commandBuffer, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetSemaphoreCounterValue(VkDevice device, VkSemaphore semaphore, uint64_t* pValue)
{
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetSemaphoreCounterValue(device, semaphore, pValue);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkWaitSemaphores(VkDevice device, const VkSemaphoreWaitInfo* pWaitInfo,
                                                              uint64_t timeout)
{
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->WaitSemaphores(device, pWaitInfo, timeout);
}

LOADER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkSignalSemaphore(VkDevice device, const VkSemaphoreSignalInfo* pSignalInfo)
{
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->SignalSemaphore(device, pSignalInfo);
}

LOADER_EXPORT VKAPI_ATTR VkDeviceAddress VKAPI_CALL vkGetBufferDeviceAddress(VkDevice device,
                                                                             const VkBufferDeviceAddressInfo* pInfo)
{
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetBufferDeviceAddress(device, pInfo);
}

LOADER_EXPORT VKAPI_ATTR uint64_t VKAPI_CALL vkGetBufferOpaqueCaptureAddress(VkDevice device,
                                                                             const VkBufferDeviceAddressInfo* pInfo)
{
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetBufferOpaqueCaptureAddress(device, pInfo);
}

LOADER_EXPORT VKAPI_ATTR uint64_t VKAPI_CALL vkGetDeviceMemoryOpaqueCaptureAddress(VkDevice device,
    const VkDeviceMemoryOpaqueCaptureAddressInfo* pInfo)
{
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    return disp->GetDeviceMemoryOpaqueCaptureAddress(device, pInfo);
}

LOADER_EXPORT VKAPI_ATTR void VKAPI_CALL vkResetQueryPool(VkDevice device, VkQueryPool queryPool, uint32_t firstQuery,
                                                          uint32_t queryCount)
{
    const VkLayerDispatchTable *disp = loader_get_dispatch(device);
    disp->ResetQueryPool(device, queryPool, firstQuery, queryCount);
}
