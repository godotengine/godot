/*
 * Copyright (c) 2015-2017 The Khronos Group Inc.
 * Copyright (c) 2015-2017 Valve Corporation
 * Copyright (c) 2015-2017 LunarG, Inc.
 * Copyright (C) 2015-2016 Google Inc.
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
 * Author: Courtney Goeltzenleuchter <courtney@LunarG.com>
 * Author: Jon Ashburn <jon@LunarG.com>
 * Author: Mark Young <marky@lunarg.com>
 *
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#ifndef WIN32
#include <signal.h>
#else
#endif
#include "vk_loader_platform.h"
#include "debug_utils.h"
#include "vulkan/vk_layer.h"
#include "vk_object_types.h"

// VK_EXT_debug_report related items

VkResult util_CreateDebugUtilsMessenger(struct loader_instance *inst, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
                                        const VkAllocationCallbacks *pAllocator, VkDebugUtilsMessengerEXT messenger) {
    VkLayerDbgFunctionNode *pNewDbgFuncNode = NULL;

#if (DEBUG_DISABLE_APP_ALLOCATORS == 1)
    {
#else
    if (pAllocator != NULL) {
        pNewDbgFuncNode = (VkLayerDbgFunctionNode *)pAllocator->pfnAllocation(pAllocator->pUserData, sizeof(VkLayerDbgFunctionNode),
                                                                              sizeof(int *), VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
    } else {
#endif
        pNewDbgFuncNode = (VkLayerDbgFunctionNode *)loader_instance_heap_alloc(inst, sizeof(VkLayerDbgFunctionNode),
                                                                               VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
    }
    if (!pNewDbgFuncNode) {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    memset(pNewDbgFuncNode, 0, sizeof(VkLayerDbgFunctionNode));

    pNewDbgFuncNode->is_messenger = true;
    pNewDbgFuncNode->messenger.messenger = messenger;
    pNewDbgFuncNode->messenger.pfnUserCallback = pCreateInfo->pfnUserCallback;
    pNewDbgFuncNode->messenger.messageSeverity = pCreateInfo->messageSeverity;
    pNewDbgFuncNode->messenger.messageType = pCreateInfo->messageType;
    pNewDbgFuncNode->pUserData = pCreateInfo->pUserData;
    pNewDbgFuncNode->pNext = inst->DbgFunctionHead;
    inst->DbgFunctionHead = pNewDbgFuncNode;

    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
debug_utils_CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
                                         const VkAllocationCallbacks *pAllocator, VkDebugUtilsMessengerEXT *pMessenger) {
    struct loader_instance *inst = loader_get_instance(instance);
    loader_platform_thread_lock_mutex(&loader_lock);
    VkResult result = inst->disp->layer_inst_disp.CreateDebugUtilsMessengerEXT(instance, pCreateInfo, pAllocator, pMessenger);
    loader_platform_thread_unlock_mutex(&loader_lock);
    return result;
}

VkBool32 util_SubmitDebugUtilsMessageEXT(const struct loader_instance *inst, VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                         VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                                         const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData) {
    VkBool32 bail = false;

    if (NULL != pCallbackData) {
        VkLayerDbgFunctionNode *pTrav = inst->DbgFunctionHead;
        VkDebugReportObjectTypeEXT object_type = VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT;
        VkDebugReportFlagsEXT object_flags = 0;
        uint64_t object_handle = 0;

        debug_utils_AnnotFlagsToReportFlags(messageSeverity, messageTypes, &object_flags);
        if (0 < pCallbackData->objectCount) {
            debug_utils_AnnotObjectToDebugReportObject(pCallbackData->pObjects, &object_type, &object_handle);
        }

        while (pTrav) {
            if (pTrav->is_messenger && (pTrav->messenger.messageSeverity & messageSeverity) &&
                (pTrav->messenger.messageType & messageTypes)) {
                if (pTrav->messenger.pfnUserCallback(messageSeverity, messageTypes, pCallbackData, pTrav->pUserData)) {
                    bail = true;
                }
            }
            if (!pTrav->is_messenger && pTrav->report.msgFlags & object_flags) {
                if (pTrav->report.pfnMsgCallback(object_flags, object_type, object_handle, 0, pCallbackData->messageIdNumber,
                                                 pCallbackData->pMessageIdName, pCallbackData->pMessage, pTrav->pUserData)) {
                    bail = true;
                }
            }

            pTrav = pTrav->pNext;
        }
    }

    return bail;
}

void util_DestroyDebugUtilsMessenger(struct loader_instance *inst, VkDebugUtilsMessengerEXT messenger,
                                     const VkAllocationCallbacks *pAllocator) {
    VkLayerDbgFunctionNode *pTrav = inst->DbgFunctionHead;
    VkLayerDbgFunctionNode *pPrev = pTrav;

    while (pTrav) {
        if (pTrav->is_messenger && pTrav->messenger.messenger == messenger) {
            pPrev->pNext = pTrav->pNext;
            if (inst->DbgFunctionHead == pTrav) inst->DbgFunctionHead = pTrav->pNext;
#if (DEBUG_DISABLE_APP_ALLOCATORS == 1)
            {
#else
            if (pAllocator != NULL) {
                pAllocator->pfnFree(pAllocator->pUserData, pTrav);
            } else {
#endif
                loader_instance_heap_free(inst, pTrav);
            }
            break;
        }
        pPrev = pTrav;
        pTrav = pTrav->pNext;
    }
}

// This utility (used by vkInstanceCreateInfo(), looks at a pNext chain.  It
// counts any VkDebugUtilsMessengerCreateInfoEXT structs that it finds.  It
// then allocates array that can hold that many structs, as well as that many
// VkDebugUtilsMessengerEXT handles.  It then copies each
// VkDebugUtilsMessengerCreateInfoEXT, and initializes each handle.
VkResult util_CopyDebugUtilsMessengerCreateInfos(const void *pChain, const VkAllocationCallbacks *pAllocator,
                                                 uint32_t *num_messengers, VkDebugUtilsMessengerCreateInfoEXT **infos,
                                                 VkDebugUtilsMessengerEXT **messengers) {
    uint32_t n = *num_messengers = 0;
    VkDebugUtilsMessengerCreateInfoEXT *pInfos = NULL;
    VkDebugUtilsMessengerEXT *pMessengers = NULL;

    const void *pNext = pChain;
    while (pNext) {
        // 1st, count the number VkDebugUtilsMessengerCreateInfoEXT:
        if (((VkDebugUtilsMessengerCreateInfoEXT *)pNext)->sType == VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT) {
            n++;
        }
        pNext = (void *)((VkDebugUtilsMessengerCreateInfoEXT *)pNext)->pNext;
    }
    if (n == 0) {
        return VK_SUCCESS;
    }

// 2nd, allocate memory for each VkDebugUtilsMessengerCreateInfoEXT:
#if (DEBUG_DISABLE_APP_ALLOCATORS == 1)
    {
#else
    if (pAllocator != NULL) {
        pInfos = *infos = ((VkDebugUtilsMessengerCreateInfoEXT *)pAllocator->pfnAllocation(
            pAllocator->pUserData, n * sizeof(VkDebugUtilsMessengerCreateInfoEXT), sizeof(void *),
            VK_SYSTEM_ALLOCATION_SCOPE_OBJECT));
    } else {
#endif
        pInfos = *infos = ((VkDebugUtilsMessengerCreateInfoEXT *)malloc(n * sizeof(VkDebugUtilsMessengerCreateInfoEXT)));
    }
    if (!pInfos) {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
// 3rd, allocate memory for a unique handle for each callback:
#if (DEBUG_DISABLE_APP_ALLOCATORS == 1)
    {
#else
    if (pAllocator != NULL) {
        pMessengers = *messengers = ((VkDebugUtilsMessengerEXT *)pAllocator->pfnAllocation(
            pAllocator->pUserData, n * sizeof(VkDebugUtilsMessengerEXT), sizeof(void *), VK_SYSTEM_ALLOCATION_SCOPE_OBJECT));
        if (NULL == pMessengers) {
            pAllocator->pfnFree(pAllocator->pUserData, pInfos);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
    } else {
#endif
        pMessengers = *messengers = ((VkDebugUtilsMessengerEXT *)malloc(n * sizeof(VkDebugUtilsMessengerEXT)));
        if (NULL == pMessengers) {
            free(pInfos);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
    }
    // 4th, copy each VkDebugUtilsMessengerCreateInfoEXT for use by
    // vkDestroyInstance, and assign a unique handle to each messenger (just
    // use the address of the copied VkDebugUtilsMessengerCreateInfoEXT):
    pNext = pChain;
    while (pNext) {
        if (((VkInstanceCreateInfo *)pNext)->sType == VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT) {
            memcpy(pInfos, pNext, sizeof(VkDebugUtilsMessengerCreateInfoEXT));
            *pMessengers++ = (VkDebugUtilsMessengerEXT)(uintptr_t)pInfos++;
        }
        pNext = (void *)((VkInstanceCreateInfo *)pNext)->pNext;
    }

    *num_messengers = n;
    return VK_SUCCESS;
}

void util_FreeDebugUtilsMessengerCreateInfos(const VkAllocationCallbacks *pAllocator, VkDebugUtilsMessengerCreateInfoEXT *infos,
                                             VkDebugUtilsMessengerEXT *messengers) {
#if (DEBUG_DISABLE_APP_ALLOCATORS == 1)
    {
#else
    if (pAllocator != NULL) {
        pAllocator->pfnFree(pAllocator->pUserData, infos);
        pAllocator->pfnFree(pAllocator->pUserData, messengers);
    } else {
#endif
        free(infos);
        free(messengers);
    }
}

VkResult util_CreateDebugUtilsMessengers(struct loader_instance *inst, const VkAllocationCallbacks *pAllocator,
                                         uint32_t num_messengers, VkDebugUtilsMessengerCreateInfoEXT *infos,
                                         VkDebugUtilsMessengerEXT *messengers) {
    VkResult rtn = VK_SUCCESS;
    for (uint32_t i = 0; i < num_messengers; i++) {
        rtn = util_CreateDebugUtilsMessenger(inst, &infos[i], pAllocator, messengers[i]);
        if (rtn != VK_SUCCESS) {
            for (uint32_t j = 0; j < i; j++) {
                util_DestroyDebugUtilsMessenger(inst, messengers[j], pAllocator);
            }
            return rtn;
        }
    }
    return rtn;
}

void util_DestroyDebugUtilsMessengers(struct loader_instance *inst, const VkAllocationCallbacks *pAllocator,
                                      uint32_t num_messengers, VkDebugUtilsMessengerEXT *messengers) {
    for (uint32_t i = 0; i < num_messengers; i++) {
        util_DestroyDebugUtilsMessenger(inst, messengers[i], pAllocator);
    }
}

static VKAPI_ATTR void VKAPI_CALL debug_utils_SubmitDebugUtilsMessageEXT(
    VkInstance instance, VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageTypes,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData) {
    struct loader_instance *inst = loader_get_instance(instance);

    inst->disp->layer_inst_disp.SubmitDebugUtilsMessageEXT(instance, messageSeverity, messageTypes, pCallbackData);
}

static VKAPI_ATTR void VKAPI_CALL debug_utils_DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT messenger,
                                                                            const VkAllocationCallbacks *pAllocator) {
    struct loader_instance *inst = loader_get_instance(instance);
    loader_platform_thread_lock_mutex(&loader_lock);

    inst->disp->layer_inst_disp.DestroyDebugUtilsMessengerEXT(instance, messenger, pAllocator);

    loader_platform_thread_unlock_mutex(&loader_lock);
}

// This is the instance chain terminator function for CreateDebugUtilsMessenger
VKAPI_ATTR VkResult VKAPI_CALL terminator_CreateDebugUtilsMessengerEXT(VkInstance instance,
                                                                       const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
                                                                       const VkAllocationCallbacks *pAllocator,
                                                                       VkDebugUtilsMessengerEXT *pMessenger) {
    VkDebugUtilsMessengerEXT *icd_info = NULL;
    const struct loader_icd_term *icd_term;
    struct loader_instance *inst = (struct loader_instance *)instance;
    VkResult res = VK_SUCCESS;
    uint32_t storage_idx;
    VkLayerDbgFunctionNode *pNewDbgFuncNode = NULL;

#if (DEBUG_DISABLE_APP_ALLOCATORS == 1)
    {
#else
    if (pAllocator != NULL) {
        icd_info = ((VkDebugUtilsMessengerEXT *)pAllocator->pfnAllocation(pAllocator->pUserData,
                                                                          inst->total_icd_count * sizeof(VkDebugUtilsMessengerEXT),
                                                                          sizeof(void *), VK_SYSTEM_ALLOCATION_SCOPE_OBJECT));
        if (icd_info) {
            memset(icd_info, 0, inst->total_icd_count * sizeof(VkDebugUtilsMessengerEXT));
        }
    } else {
#endif
        icd_info = calloc(sizeof(VkDebugUtilsMessengerEXT), inst->total_icd_count);
    }
    if (!icd_info) {
        res = VK_ERROR_OUT_OF_HOST_MEMORY;
        goto out;
    }

    storage_idx = 0;
    for (icd_term = inst->icd_terms; icd_term; icd_term = icd_term->next) {
        if (!icd_term->dispatch.CreateDebugUtilsMessengerEXT) {
            continue;
        }

        res = icd_term->dispatch.CreateDebugUtilsMessengerEXT(icd_term->instance, pCreateInfo, pAllocator, &icd_info[storage_idx]);

        if (res != VK_SUCCESS) {
            goto out;
        }
        storage_idx++;
    }

// Setup the debug report callback in the terminator since a layer may want
// to grab the information itself (RenderDoc) and then return back to the
// user callback a sub-set of the messages.
#if (DEBUG_DISABLE_APP_ALLOCATORS == 0)
    if (pAllocator != NULL) {
        pNewDbgFuncNode = (VkLayerDbgFunctionNode *)pAllocator->pfnAllocation(pAllocator->pUserData, sizeof(VkLayerDbgFunctionNode),
                                                                              sizeof(int *), VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
    } else {
#else
    {
#endif
        pNewDbgFuncNode = (VkLayerDbgFunctionNode *)loader_instance_heap_alloc(inst, sizeof(VkLayerDbgFunctionNode),
                                                                               VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
    }
    if (!pNewDbgFuncNode) {
        res = VK_ERROR_OUT_OF_HOST_MEMORY;
        goto out;
    }
    memset(pNewDbgFuncNode, 0, sizeof(VkLayerDbgFunctionNode));

    pNewDbgFuncNode->is_messenger = true;
    pNewDbgFuncNode->messenger.pfnUserCallback = pCreateInfo->pfnUserCallback;
    pNewDbgFuncNode->messenger.messageSeverity = pCreateInfo->messageSeverity;
    pNewDbgFuncNode->messenger.messageType = pCreateInfo->messageType;
    pNewDbgFuncNode->pUserData = pCreateInfo->pUserData;
    pNewDbgFuncNode->pNext = inst->DbgFunctionHead;
    inst->DbgFunctionHead = pNewDbgFuncNode;

    *(VkDebugUtilsMessengerEXT **)pMessenger = icd_info;
    pNewDbgFuncNode->messenger.messenger = *pMessenger;

out:

    // Roll back on errors
    if (VK_SUCCESS != res) {
        storage_idx = 0;
        for (icd_term = inst->icd_terms; icd_term; icd_term = icd_term->next) {
            if (NULL == icd_term->dispatch.DestroyDebugUtilsMessengerEXT) {
                continue;
            }

            if (icd_info && icd_info[storage_idx]) {
                icd_term->dispatch.DestroyDebugUtilsMessengerEXT(icd_term->instance, icd_info[storage_idx], pAllocator);
            }
            storage_idx++;
        }

#if (DEBUG_DISABLE_APP_ALLOCATORS == 1)
        {
#else
        if (pAllocator != NULL) {
            if (NULL != pNewDbgFuncNode) {
                pAllocator->pfnFree(pAllocator->pUserData, pNewDbgFuncNode);
            }
            if (NULL != icd_info) {
                pAllocator->pfnFree(pAllocator->pUserData, icd_info);
            }
        } else {
#endif
            if (NULL != pNewDbgFuncNode) {
                free(pNewDbgFuncNode);
            }
            if (NULL != icd_info) {
                free(icd_info);
            }
        }
    }

    return res;
}

// This is the instance chain terminator function for DestroyDebugUtilsMessenger
VKAPI_ATTR void VKAPI_CALL terminator_DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT messenger,
                                                                    const VkAllocationCallbacks *pAllocator) {
    uint32_t storage_idx;
    VkDebugUtilsMessengerEXT *icd_info;
    const struct loader_icd_term *icd_term;

    struct loader_instance *inst = (struct loader_instance *)instance;
    icd_info = *(VkDebugUtilsMessengerEXT **)&messenger;
    storage_idx = 0;
    for (icd_term = inst->icd_terms; icd_term; icd_term = icd_term->next) {
        if (NULL == icd_term->dispatch.DestroyDebugUtilsMessengerEXT) {
            continue;
        }

        if (icd_info[storage_idx]) {
            icd_term->dispatch.DestroyDebugUtilsMessengerEXT(icd_term->instance, icd_info[storage_idx], pAllocator);
        }
        storage_idx++;
    }

    util_DestroyDebugUtilsMessenger(inst, messenger, pAllocator);

#if (DEBUG_DISABLE_APP_ALLOCATORS == 1)
    {
#else
    if (pAllocator != NULL) {
        pAllocator->pfnFree(pAllocator->pUserData, icd_info);
    } else {
#endif
        free(icd_info);
    }
}

// This is the instance chain terminator function for SubmitDebugUtilsMessageEXT
VKAPI_ATTR void VKAPI_CALL terminator_SubmitDebugUtilsMessageEXT(VkInstance instance,
                                                                 VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                                 VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                                                                 const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData) {
    loader_platform_thread_lock_mutex(&loader_lock);
    // NOTE: Just make the callback ourselves because there could be one or more ICDs that support this extension
    //       and each one will trigger the callback to the user.  This would result in multiple callback triggers
    //       per message.  Instead, if we get a messaged up to here, then just trigger the message ourselves and
    //       return.  This would still allow the ICDs to trigger their own messages, but won't get any external ones.
    struct loader_instance *inst = (struct loader_instance *)instance;
    util_SubmitDebugUtilsMessageEXT(inst, messageSeverity, messageTypes, pCallbackData);
    loader_platform_thread_unlock_mutex(&loader_lock);
}

// VK_EXT_debug_report related items

VkResult util_CreateDebugReportCallback(struct loader_instance *inst, VkDebugReportCallbackCreateInfoEXT *pCreateInfo,
                                        const VkAllocationCallbacks *pAllocator, VkDebugReportCallbackEXT callback) {
    VkLayerDbgFunctionNode *pNewDbgFuncNode = NULL;

#if (DEBUG_DISABLE_APP_ALLOCATORS == 1)
    {
#else
    if (pAllocator != NULL) {
        pNewDbgFuncNode = (VkLayerDbgFunctionNode *)pAllocator->pfnAllocation(pAllocator->pUserData, sizeof(VkLayerDbgFunctionNode),
                                                                              sizeof(int *), VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
    } else {
#endif
        pNewDbgFuncNode = (VkLayerDbgFunctionNode *)loader_instance_heap_alloc(inst, sizeof(VkLayerDbgFunctionNode),
                                                                               VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
    }
    if (!pNewDbgFuncNode) {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    memset(pNewDbgFuncNode, 0, sizeof(VkLayerDbgFunctionNode));

    pNewDbgFuncNode->is_messenger = false;
    pNewDbgFuncNode->report.msgCallback = callback;
    pNewDbgFuncNode->report.pfnMsgCallback = pCreateInfo->pfnCallback;
    pNewDbgFuncNode->report.msgFlags = pCreateInfo->flags;
    pNewDbgFuncNode->pUserData = pCreateInfo->pUserData;
    pNewDbgFuncNode->pNext = inst->DbgFunctionHead;
    inst->DbgFunctionHead = pNewDbgFuncNode;

    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
debug_utils_CreateDebugReportCallbackEXT(VkInstance instance, const VkDebugReportCallbackCreateInfoEXT *pCreateInfo,
                                         const VkAllocationCallbacks *pAllocator, VkDebugReportCallbackEXT *pCallback) {
    struct loader_instance *inst = loader_get_instance(instance);
    loader_platform_thread_lock_mutex(&loader_lock);
    VkResult result = inst->disp->layer_inst_disp.CreateDebugReportCallbackEXT(instance, pCreateInfo, pAllocator, pCallback);
    loader_platform_thread_unlock_mutex(&loader_lock);
    return result;
}

// Utility function to handle reporting
VkBool32 util_DebugReportMessage(const struct loader_instance *inst, VkFlags msgFlags, VkDebugReportObjectTypeEXT objectType,
                                 uint64_t srcObject, size_t location, int32_t msgCode, const char *pLayerPrefix, const char *pMsg) {
    VkBool32 bail = false;
    VkLayerDbgFunctionNode *pTrav = inst->DbgFunctionHead;
    VkDebugUtilsMessageSeverityFlagBitsEXT severity;
    VkDebugUtilsMessageTypeFlagsEXT types;
    VkDebugUtilsMessengerCallbackDataEXT callback_data;
    VkDebugUtilsObjectNameInfoEXT object_name;

    debug_utils_ReportFlagsToAnnotFlags(msgFlags, false, &severity, &types);
    debug_utils_ReportObjectToAnnotObject(objectType, srcObject, &object_name);

    callback_data.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT;
    callback_data.pNext = NULL;
    callback_data.flags = 0;
    callback_data.pMessageIdName = pLayerPrefix;
    callback_data.messageIdNumber = msgCode;
    callback_data.pMessage = pMsg;
    callback_data.cmdBufLabelCount = 0;
    callback_data.pCmdBufLabels = NULL;
    callback_data.queueLabelCount = 0;
    callback_data.pQueueLabels = NULL;
    callback_data.objectCount = 1;
    callback_data.pObjects = &object_name;

    while (pTrav) {
        if (!pTrav->is_messenger && pTrav->report.msgFlags & msgFlags) {
            if (pTrav->report.pfnMsgCallback(msgFlags, objectType, srcObject, location, msgCode, pLayerPrefix, pMsg,
                                             pTrav->pUserData)) {
                bail = true;
            }
        }
        if (pTrav->is_messenger && (pTrav->messenger.messageSeverity & severity) && (pTrav->messenger.messageType & types)) {
            if (pTrav->messenger.pfnUserCallback(severity, types, &callback_data, pTrav->pUserData)) {
                bail = true;
            }
        }

        pTrav = pTrav->pNext;
    }

    return bail;
}

void util_DestroyDebugReportCallback(struct loader_instance *inst, VkDebugReportCallbackEXT callback,
                                     const VkAllocationCallbacks *pAllocator) {
    VkLayerDbgFunctionNode *pTrav = inst->DbgFunctionHead;
    VkLayerDbgFunctionNode *pPrev = pTrav;

    while (pTrav) {
        if (!pTrav->is_messenger && pTrav->report.msgCallback == callback) {
            pPrev->pNext = pTrav->pNext;
            if (inst->DbgFunctionHead == pTrav) inst->DbgFunctionHead = pTrav->pNext;
#if (DEBUG_DISABLE_APP_ALLOCATORS == 1)
            {
#else
            if (pAllocator != NULL) {
                pAllocator->pfnFree(pAllocator->pUserData, pTrav);
            } else {
#endif
                loader_instance_heap_free(inst, pTrav);
            }
            break;
        }
        pPrev = pTrav;
        pTrav = pTrav->pNext;
    }
}

// This utility (used by vkInstanceCreateInfo(), looks at a pNext chain.  It
// counts any VkDebugReportCallbackCreateInfoEXT structs that it finds.  It
// then allocates array that can hold that many structs, as well as that many
// VkDebugReportCallbackEXT handles.  It then copies each
// VkDebugReportCallbackCreateInfoEXT, and initializes each handle.
VkResult util_CopyDebugReportCreateInfos(const void *pChain, const VkAllocationCallbacks *pAllocator, uint32_t *num_callbacks,
                                         VkDebugReportCallbackCreateInfoEXT **infos, VkDebugReportCallbackEXT **callbacks) {
    uint32_t n = *num_callbacks = 0;
    VkDebugReportCallbackCreateInfoEXT *pInfos = NULL;
    VkDebugReportCallbackEXT *pCallbacks = NULL;

    const void *pNext = pChain;
    while (pNext) {
        // 1st, count the number VkDebugReportCallbackCreateInfoEXT:
        if (((VkDebugReportCallbackCreateInfoEXT *)pNext)->sType == VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT) {
            n++;
        }
        pNext = (void *)((VkDebugReportCallbackCreateInfoEXT *)pNext)->pNext;
    }
    if (n == 0) {
        return VK_SUCCESS;
    }

// 2nd, allocate memory for each VkDebugReportCallbackCreateInfoEXT:
#if (DEBUG_DISABLE_APP_ALLOCATORS == 1)
    {
#else
    if (pAllocator != NULL) {
        pInfos = *infos = ((VkDebugReportCallbackCreateInfoEXT *)pAllocator->pfnAllocation(
            pAllocator->pUserData, n * sizeof(VkDebugReportCallbackCreateInfoEXT), sizeof(void *),
            VK_SYSTEM_ALLOCATION_SCOPE_OBJECT));
    } else {
#endif
        pInfos = *infos = ((VkDebugReportCallbackCreateInfoEXT *)malloc(n * sizeof(VkDebugReportCallbackCreateInfoEXT)));
    }
    if (!pInfos) {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
// 3rd, allocate memory for a unique handle for each callback:
#if (DEBUG_DISABLE_APP_ALLOCATORS == 1)
    {
#else
    if (pAllocator != NULL) {
        pCallbacks = *callbacks = ((VkDebugReportCallbackEXT *)pAllocator->pfnAllocation(
            pAllocator->pUserData, n * sizeof(VkDebugReportCallbackEXT), sizeof(void *), VK_SYSTEM_ALLOCATION_SCOPE_OBJECT));
        if (!pCallbacks) {
            pAllocator->pfnFree(pAllocator->pUserData, pInfos);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
    } else {
#endif
        pCallbacks = *callbacks = ((VkDebugReportCallbackEXT *)malloc(n * sizeof(VkDebugReportCallbackEXT)));
        if (!pCallbacks) {
            free(pInfos);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
    }
    // 4th, copy each VkDebugReportCallbackCreateInfoEXT for use by
    // vkDestroyInstance, and assign a unique handle to each callback (just
    // use the address of the copied VkDebugReportCallbackCreateInfoEXT):
    pNext = pChain;
    while (pNext) {
        if (((VkInstanceCreateInfo *)pNext)->sType == VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT) {
            memcpy(pInfos, pNext, sizeof(VkDebugReportCallbackCreateInfoEXT));
            *pCallbacks++ = (VkDebugReportCallbackEXT)(uintptr_t)pInfos++;
        }
        pNext = (void *)((VkInstanceCreateInfo *)pNext)->pNext;
    }

    *num_callbacks = n;
    return VK_SUCCESS;
}

void util_FreeDebugReportCreateInfos(const VkAllocationCallbacks *pAllocator, VkDebugReportCallbackCreateInfoEXT *infos,
                                     VkDebugReportCallbackEXT *callbacks) {
#if (DEBUG_DISABLE_APP_ALLOCATORS == 1)
    {
#else
    if (pAllocator != NULL) {
        pAllocator->pfnFree(pAllocator->pUserData, infos);
        pAllocator->pfnFree(pAllocator->pUserData, callbacks);
    } else {
#endif
        free(infos);
        free(callbacks);
    }
}

VkResult util_CreateDebugReportCallbacks(struct loader_instance *inst, const VkAllocationCallbacks *pAllocator,
                                         uint32_t num_callbacks, VkDebugReportCallbackCreateInfoEXT *infos,
                                         VkDebugReportCallbackEXT *callbacks) {
    VkResult rtn = VK_SUCCESS;
    for (uint32_t i = 0; i < num_callbacks; i++) {
        rtn = util_CreateDebugReportCallback(inst, &infos[i], pAllocator, callbacks[i]);
        if (rtn != VK_SUCCESS) {
            for (uint32_t j = 0; j < i; j++) {
                util_DestroyDebugReportCallback(inst, callbacks[j], pAllocator);
            }
            return rtn;
        }
    }
    return rtn;
}

void util_DestroyDebugReportCallbacks(struct loader_instance *inst, const VkAllocationCallbacks *pAllocator, uint32_t num_callbacks,
                                      VkDebugReportCallbackEXT *callbacks) {
    for (uint32_t i = 0; i < num_callbacks; i++) {
        util_DestroyDebugReportCallback(inst, callbacks[i], pAllocator);
    }
}

static VKAPI_ATTR void VKAPI_CALL debug_utils_DestroyDebugReportCallbackEXT(VkInstance instance, VkDebugReportCallbackEXT callback,
                                                                            const VkAllocationCallbacks *pAllocator) {
    struct loader_instance *inst = loader_get_instance(instance);
    loader_platform_thread_lock_mutex(&loader_lock);

    inst->disp->layer_inst_disp.DestroyDebugReportCallbackEXT(instance, callback, pAllocator);

    loader_platform_thread_unlock_mutex(&loader_lock);
}

static VKAPI_ATTR void VKAPI_CALL debug_utils_DebugReportMessageEXT(VkInstance instance, VkDebugReportFlagsEXT flags,
                                                                    VkDebugReportObjectTypeEXT objType, uint64_t object,
                                                                    size_t location, int32_t msgCode, const char *pLayerPrefix,
                                                                    const char *pMsg) {
    struct loader_instance *inst = loader_get_instance(instance);

    inst->disp->layer_inst_disp.DebugReportMessageEXT(instance, flags, objType, object, location, msgCode, pLayerPrefix, pMsg);
}

// This is the instance chain terminator function
// for CreateDebugReportCallback
VKAPI_ATTR VkResult VKAPI_CALL terminator_CreateDebugReportCallbackEXT(VkInstance instance,
                                                                       const VkDebugReportCallbackCreateInfoEXT *pCreateInfo,
                                                                       const VkAllocationCallbacks *pAllocator,
                                                                       VkDebugReportCallbackEXT *pCallback) {
    VkDebugReportCallbackEXT *icd_info = NULL;
    const struct loader_icd_term *icd_term;
    struct loader_instance *inst = (struct loader_instance *)instance;
    VkResult res = VK_SUCCESS;
    uint32_t storage_idx;
    VkLayerDbgFunctionNode *pNewDbgFuncNode = NULL;

#if (DEBUG_DISABLE_APP_ALLOCATORS == 1)
    {
#else
    if (pAllocator != NULL) {
        icd_info = ((VkDebugReportCallbackEXT *)pAllocator->pfnAllocation(pAllocator->pUserData,
                                                                          inst->total_icd_count * sizeof(VkDebugReportCallbackEXT),
                                                                          sizeof(void *), VK_SYSTEM_ALLOCATION_SCOPE_OBJECT));
        if (icd_info) {
            memset(icd_info, 0, inst->total_icd_count * sizeof(VkDebugReportCallbackEXT));
        }
    } else {
#endif
        icd_info = calloc(sizeof(VkDebugReportCallbackEXT), inst->total_icd_count);
    }
    if (!icd_info) {
        res = VK_ERROR_OUT_OF_HOST_MEMORY;
        goto out;
    }

    storage_idx = 0;
    for (icd_term = inst->icd_terms; icd_term; icd_term = icd_term->next) {
        if (!icd_term->dispatch.CreateDebugReportCallbackEXT) {
            continue;
        }

        res = icd_term->dispatch.CreateDebugReportCallbackEXT(icd_term->instance, pCreateInfo, pAllocator, &icd_info[storage_idx]);

        if (res != VK_SUCCESS) {
            goto out;
        }
        storage_idx++;
    }

// Setup the debug report callback in the terminator since a layer may want
// to grab the information itself (RenderDoc) and then return back to the
// user callback a sub-set of the messages.
#if (DEBUG_DISABLE_APP_ALLOCATORS == 0)
    if (pAllocator != NULL) {
        pNewDbgFuncNode = (VkLayerDbgFunctionNode *)pAllocator->pfnAllocation(pAllocator->pUserData, sizeof(VkLayerDbgFunctionNode),
                                                                              sizeof(int *), VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
    } else {
#else
    {
#endif
        pNewDbgFuncNode = (VkLayerDbgFunctionNode *)loader_instance_heap_alloc(inst, sizeof(VkLayerDbgFunctionNode),
                                                                               VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
    }
    if (!pNewDbgFuncNode) {
        res = VK_ERROR_OUT_OF_HOST_MEMORY;
        goto out;
    }
    memset(pNewDbgFuncNode, 0, sizeof(VkLayerDbgFunctionNode));

    pNewDbgFuncNode->is_messenger = false;
    pNewDbgFuncNode->report.pfnMsgCallback = pCreateInfo->pfnCallback;
    pNewDbgFuncNode->report.msgFlags = pCreateInfo->flags;
    pNewDbgFuncNode->pUserData = pCreateInfo->pUserData;
    pNewDbgFuncNode->pNext = inst->DbgFunctionHead;
    inst->DbgFunctionHead = pNewDbgFuncNode;

    *(VkDebugReportCallbackEXT **)pCallback = icd_info;
    pNewDbgFuncNode->report.msgCallback = *pCallback;

out:

    // Roll back on errors
    if (VK_SUCCESS != res) {
        storage_idx = 0;
        for (icd_term = inst->icd_terms; icd_term; icd_term = icd_term->next) {
            if (NULL == icd_term->dispatch.DestroyDebugReportCallbackEXT) {
                continue;
            }

            if (icd_info && icd_info[storage_idx]) {
                icd_term->dispatch.DestroyDebugReportCallbackEXT(icd_term->instance, icd_info[storage_idx], pAllocator);
            }
            storage_idx++;
        }

#if (DEBUG_DISABLE_APP_ALLOCATORS == 1)
        {
#else
        if (pAllocator != NULL) {
            if (NULL != pNewDbgFuncNode) {
                pAllocator->pfnFree(pAllocator->pUserData, pNewDbgFuncNode);
            }
            if (NULL != icd_info) {
                pAllocator->pfnFree(pAllocator->pUserData, icd_info);
            }
        } else {
#endif
            if (NULL != pNewDbgFuncNode) {
                free(pNewDbgFuncNode);
            }
            if (NULL != icd_info) {
                free(icd_info);
            }
        }
    }

    return res;
}

// This is the instance chain terminator function for DestroyDebugReportCallback
VKAPI_ATTR void VKAPI_CALL terminator_DestroyDebugReportCallbackEXT(VkInstance instance, VkDebugReportCallbackEXT callback,
                                                                    const VkAllocationCallbacks *pAllocator) {
    uint32_t storage_idx;
    VkDebugReportCallbackEXT *icd_info;
    const struct loader_icd_term *icd_term;

    struct loader_instance *inst = (struct loader_instance *)instance;
    icd_info = *(VkDebugReportCallbackEXT **)&callback;
    storage_idx = 0;
    for (icd_term = inst->icd_terms; icd_term; icd_term = icd_term->next) {
        if (NULL == icd_term->dispatch.DestroyDebugReportCallbackEXT) {
            continue;
        }

        if (icd_info[storage_idx]) {
            icd_term->dispatch.DestroyDebugReportCallbackEXT(icd_term->instance, icd_info[storage_idx], pAllocator);
        }
        storage_idx++;
    }

    util_DestroyDebugReportCallback(inst, callback, pAllocator);

#if (DEBUG_DISABLE_APP_ALLOCATORS == 1)
    {
#else
    if (pAllocator != NULL) {
        pAllocator->pfnFree(pAllocator->pUserData, icd_info);
    } else {
#endif
        free(icd_info);
    }
}

// This is the instance chain terminator function for DebugReportMessage
VKAPI_ATTR void VKAPI_CALL terminator_DebugReportMessageEXT(VkInstance instance, VkDebugReportFlagsEXT flags,
                                                            VkDebugReportObjectTypeEXT objType, uint64_t object, size_t location,
                                                            int32_t msgCode, const char *pLayerPrefix, const char *pMsg) {
    const struct loader_icd_term *icd_term;

    struct loader_instance *inst = (struct loader_instance *)instance;

    loader_platform_thread_lock_mutex(&loader_lock);
    for (icd_term = inst->icd_terms; icd_term; icd_term = icd_term->next) {
        if (icd_term->dispatch.DebugReportMessageEXT != NULL) {
            icd_term->dispatch.DebugReportMessageEXT(icd_term->instance, flags, objType, object, location, msgCode, pLayerPrefix,
                                                     pMsg);
        }
    }

    // Now that all ICDs have seen the message, call the necessary callbacks.  Ignoring "bail" return value
    // as there is nothing to bail from at this point.

    util_DebugReportMessage(inst, flags, objType, object, location, msgCode, pLayerPrefix, pMsg);

    loader_platform_thread_unlock_mutex(&loader_lock);
}

// General utilities

static const VkExtensionProperties debug_utils_extension_info[] = {
    {VK_EXT_DEBUG_REPORT_EXTENSION_NAME, VK_EXT_DEBUG_REPORT_SPEC_VERSION},
    {VK_EXT_DEBUG_UTILS_EXTENSION_NAME, VK_EXT_DEBUG_UTILS_SPEC_VERSION},
};

void debug_utils_AddInstanceExtensions(const struct loader_instance *inst, struct loader_extension_list *ext_list) {
    loader_add_to_ext_list(inst, ext_list, sizeof(debug_utils_extension_info) / sizeof(VkExtensionProperties),
                           debug_utils_extension_info);
}

void debug_utils_CreateInstance(struct loader_instance *ptr_instance, const VkInstanceCreateInfo *pCreateInfo) {
    ptr_instance->enabled_known_extensions.ext_debug_report = 0;
    ptr_instance->enabled_known_extensions.ext_debug_utils = 0;

    for (uint32_t i = 0; i < pCreateInfo->enabledExtensionCount; i++) {
        if (strcmp(pCreateInfo->ppEnabledExtensionNames[i], VK_EXT_DEBUG_REPORT_EXTENSION_NAME) == 0) {
            ptr_instance->enabled_known_extensions.ext_debug_report = 1;
        } else if (strcmp(pCreateInfo->ppEnabledExtensionNames[i], VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
            ptr_instance->enabled_known_extensions.ext_debug_utils = 1;
        }
    }
}

bool debug_utils_InstanceGpa(struct loader_instance *ptr_instance, const char *name, void **addr) {
    bool ret_type = false;

    *addr = NULL;

    if (!strcmp("vkCreateDebugReportCallbackEXT", name)) {
        *addr = ptr_instance->enabled_known_extensions.ext_debug_report == 1 ? (void *)debug_utils_CreateDebugReportCallbackEXT : NULL;
        ret_type = true;
    } else if (!strcmp("vkDestroyDebugReportCallbackEXT", name)) {
        *addr = ptr_instance->enabled_known_extensions.ext_debug_report == 1 ? (void *)debug_utils_DestroyDebugReportCallbackEXT : NULL;
        ret_type = true;
    } else if (!strcmp("vkDebugReportMessageEXT", name)) {
        *addr = ptr_instance->enabled_known_extensions.ext_debug_report == 1 ? (void *)debug_utils_DebugReportMessageEXT : NULL;
        return true;
    }
    if (!strcmp("vkCreateDebugUtilsMessengerEXT", name)) {
        *addr = ptr_instance->enabled_known_extensions.ext_debug_utils == 1 ? (void *)debug_utils_CreateDebugUtilsMessengerEXT : NULL;
        ret_type = true;
    } else if (!strcmp("vkDestroyDebugUtilsMessengerEXT", name)) {
        *addr = ptr_instance->enabled_known_extensions.ext_debug_utils == 1 ? (void *)debug_utils_DestroyDebugUtilsMessengerEXT : NULL;
        ret_type = true;
    } else if (!strcmp("vkSubmitDebugUtilsMessageEXT", name)) {
        *addr = ptr_instance->enabled_known_extensions.ext_debug_utils == 1 ? (void *)debug_utils_SubmitDebugUtilsMessageEXT : NULL;
        ret_type = true;
    }

    return ret_type;
}

bool debug_utils_ReportFlagsToAnnotFlags(VkDebugReportFlagsEXT dr_flags, bool default_flag_is_spec,
                                         VkDebugUtilsMessageSeverityFlagBitsEXT *da_severity,
                                         VkDebugUtilsMessageTypeFlagsEXT *da_type) {
    bool type_set = false;
    if (NULL == da_severity || NULL == da_type) {
        return false;
    }
    *da_type = 0;
    *da_severity = 0;

    if ((dr_flags & VK_DEBUG_REPORT_INFORMATION_BIT_EXT) != 0) {
        *da_severity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
        *da_type |= VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
        type_set = true;
    } else if ((dr_flags & (VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT)) != 0) {
        *da_severity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
    } else if ((dr_flags & VK_DEBUG_REPORT_ERROR_BIT_EXT) != 0) {
        *da_severity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    } else if ((dr_flags & VK_DEBUG_REPORT_DEBUG_BIT_EXT) != 0) {
        *da_severity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT;
        *da_type |= VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
        type_set = true;
    }

    if ((dr_flags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT) != 0) {
        *da_type |= VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    } else if (!type_set) {
        if (default_flag_is_spec) {
            *da_type |= VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
        } else {
            *da_type |= VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
        }
    }

    return true;
}

bool debug_utils_AnnotFlagsToReportFlags(VkDebugUtilsMessageSeverityFlagBitsEXT da_severity,
                                         VkDebugUtilsMessageTypeFlagsEXT da_type, VkDebugReportFlagsEXT *dr_flags) {
    if (NULL == dr_flags) {
        return false;
    }

    *dr_flags = 0;

    if ((da_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
        *dr_flags |= VK_DEBUG_REPORT_ERROR_BIT_EXT;
    } else if ((da_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0) {
        if ((da_type & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT) != 0) {
            *dr_flags |= VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
        } else {
            *dr_flags |= VK_DEBUG_REPORT_WARNING_BIT_EXT;
        }
    } else if ((da_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) != 0) {
        *dr_flags |= VK_DEBUG_REPORT_INFORMATION_BIT_EXT;
    } else if ((da_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) != 0) {
        *dr_flags |= VK_DEBUG_REPORT_DEBUG_BIT_EXT;
    }

    return true;
}

bool debug_utils_ReportObjectToAnnotObject(VkDebugReportObjectTypeEXT dr_object_type, uint64_t object_handle,
                                           VkDebugUtilsObjectNameInfoEXT *da_object_name_info) {
    if (NULL == da_object_name_info) {
        return false;
    }
    da_object_name_info->sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    da_object_name_info->pNext = NULL;
    da_object_name_info->objectHandle = (uint64_t)(uintptr_t)object_handle;
    da_object_name_info->pObjectName = NULL;
    da_object_name_info->objectType = convertDebugReportObjectToCoreObject(dr_object_type);
    return true;
}

bool debug_utils_AnnotObjectToDebugReportObject(const VkDebugUtilsObjectNameInfoEXT *da_object_name_info,
                                                VkDebugReportObjectTypeEXT *dr_object_type, uint64_t *dr_object_handle) {
    if (NULL == da_object_name_info || NULL == dr_object_type || NULL == dr_object_handle) {
        return false;
    }
    *dr_object_type = convertCoreObjectToDebugReportObject(da_object_name_info->objectType);
    *dr_object_handle = da_object_name_info->objectHandle;
    return true;
}
