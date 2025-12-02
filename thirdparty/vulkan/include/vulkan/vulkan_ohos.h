#ifndef VULKAN_OHOS_H_
#define VULKAN_OHOS_H_ 1

/*
** Copyright 2015-2022 The Khronos Group Inc.
**
** SPDX-License-Identifier: Apache-2.0
*/

/*
** This header is generated from the Khronos Vulkan XML API Registry.
**
*/


#ifdef __cplusplus
extern "C" {
#endif



#define VK_OHOS_surface 1
typedef struct NativeWindow OHNativeWindow;
#define VK_OHOS_SURFACE_SPEC_VERSION      1
#define VK_OHOS_SURFACE_EXTENSION_NAME    "VK_OHOS_surface"
typedef VkFlags VkSurfaceCreateFlagsOHOS;
typedef struct VkSurfaceCreateInfoOHOS {
    VkStructureType             sType;
    const void*                 pNext;
    VkSurfaceCreateFlagsOHOS    flags;
    OHNativeWindow*             window;
} VkSurfaceCreateInfoOHOS;

typedef VkResult (VKAPI_PTR *PFN_vkCreateSurfaceOHOS)(VkInstance instance, const VkSurfaceCreateInfoOHOS* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface);

#ifndef VK_NO_PROTOTYPES
VKAPI_ATTR VkResult VKAPI_CALL vkCreateSurfaceOHOS(
    VkInstance                                  instance,
    const VkSurfaceCreateInfoOHOS*              pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface);
#endif


#define VK_OHOS_native_buffer 1
struct OHBufferHandle;
#define VK_OHOS_NATIVE_BUFFER_SPEC_VERSION 1
#define VK_OHOS_NATIVE_BUFFER_EXTENSION_NAME "VK_OHOS_native_buffer"

typedef enum VkSwapchainImageUsageFlagBitsOHOS {
    VK_SWAPCHAIN_IMAGE_USAGE_SHARED_BIT_OHOS = 0x00000001,
    VK_SWAPCHAIN_IMAGE_USAGE_FLAG_BITS_MAX_ENUM_OHOS = 0x7FFFFFFF
} VkSwapchainImageUsageFlagBitsOHOS;
typedef VkFlags VkSwapchainImageUsageFlagsOHOS;
typedef struct VkNativeBufferOHOS {
    VkStructureType    sType;
    const void*        pNext;
    struct OHBufferHandle*      handle;
} VkNativeBufferOHOS;

typedef struct VkSwapchainImageCreateInfoOHOS {
    VkStructureType                   sType;
    const void*                       pNext;
    VkSwapchainImageUsageFlagsOHOS    usage;
} VkSwapchainImageCreateInfoOHOS;

typedef struct VkPhysicalDevicePresentationPropertiesOHOS {
    VkStructureType    sType;
    const void*        pNext;
    VkBool32           sharedImage;
} VkPhysicalDevicePresentationPropertiesOHOS;

/**
 * @brief this type is deprecated, please use PFN_vkAcquireImageOHOS instead
 * @deprecated
 */
typedef VkResult (VKAPI_PTR *PFN_vkSetNativeFenceFdOpenHarmony)(VkDevice device, int32_t nativeFenceFd, VkSemaphore semaphore, VkFence fence);

/**
 * @brief this type is deprecated, please use PFN_vkQueueSignalReleaseImageOHOS instead
 * @deprecated
 */
typedef VkResult (VKAPI_PTR *PFN_vkGetNativeFenceFdOpenHarmony)(VkQueue queue, uint32_t waitSemaphoreCount, const VkSemaphore* pWaitSemaphores, VkImage image, int32_t* pNativeFenceFd);
typedef VkResult (VKAPI_PTR *PFN_vkGetSwapchainGrallocUsageOHOS)(VkDevice device, VkFormat format, VkImageUsageFlags imageUsage, uint64_t* grallocUsage);
typedef VkResult (VKAPI_PTR *PFN_vkAcquireImageOHOS)(VkDevice device, VkImage image, int32_t nativeFenceFd, VkSemaphore semaphore, VkFence fence);
typedef VkResult (VKAPI_PTR *PFN_vkQueueSignalReleaseImageOHOS)(VkQueue queue, uint32_t waitSemaphoreCount, const VkSemaphore* pWaitSemaphores, VkImage image, int32_t* pNativeFenceFd);

#ifndef VK_NO_PROTOTYPES
/**
 * @brief this interface is deprecated, please use vkAcquireImageOHOS instead
 * @deprecated
 */
VKAPI_ATTR VkResult VKAPI_CALL vkSetNativeFenceFdOpenHarmony(
    VkDevice                                    device,
    int32_t                                     nativeFenceFd,
    VkSemaphore                                 semaphore,
    VkFence                                     fence);

/**
 * @brief this interface is deprecated, please use vkQueueSignalReleaseImageOHOS instead
 * @deprecated
 */
VKAPI_ATTR VkResult VKAPI_CALL vkGetNativeFenceFdOpenHarmony(
    VkQueue                                     queue,
    uint32_t                                    waitSemaphoreCount,
    const VkSemaphore*                          pWaitSemaphores,
    VkImage                                     image,
    int32_t*                                    pNativeFenceFd);

VKAPI_ATTR VkResult VKAPI_CALL vkGetSwapchainGrallocUsageOHOS(
    VkDevice                                    device,
    VkFormat                                    format,
    VkImageUsageFlags                           imageUsage,
    uint64_t*                                   grallocUsage);

VKAPI_ATTR VkResult VKAPI_CALL vkAcquireImageOHOS(
    VkDevice                                    device,
    VkImage                                     image,
    int32_t                                     nativeFenceFd,
    VkSemaphore                                 semaphore,
    VkFence                                     fence);

VKAPI_ATTR VkResult VKAPI_CALL vkQueueSignalReleaseImageOHOS(
    VkQueue                                     queue,
    uint32_t                                    waitSemaphoreCount,
    const VkSemaphore*                          pWaitSemaphores,
    VkImage                                     image,
    int32_t*                                    pNativeFenceFd);
#endif


#define VK_OHOS_external_memory 1
struct OH_NativeBuffer;
#define VK_OHOS_EXTERNAL_MEMORY_SPEC_VERSION 1
#define VK_OHOS_EXTERNAL_MEMORY_EXTENSION_NAME "VK_OHOS_external_memory"
typedef struct VkNativeBufferUsageOHOS {
    VkStructureType    sType;
    void*              pNext;
    uint64_t           OHOSNativeBufferUsage;
} VkNativeBufferUsageOHOS;

typedef struct VkNativeBufferPropertiesOHOS {
    VkStructureType    sType;
    void*              pNext;
    VkDeviceSize       allocationSize;
    uint32_t           memoryTypeBits;
} VkNativeBufferPropertiesOHOS;

typedef struct VkNativeBufferFormatPropertiesOHOS {
    VkStructureType                  sType;
    void*                            pNext;
    VkFormat                         format;
    uint64_t                         externalFormat;
    VkFormatFeatureFlags             formatFeatures;
    VkComponentMapping               samplerYcbcrConversionComponents;
    VkSamplerYcbcrModelConversion    suggestedYcbcrModel;
    VkSamplerYcbcrRange              suggestedYcbcrRange;
    VkChromaLocation                 suggestedXChromaOffset;
    VkChromaLocation                 suggestedYChromaOffset;
} VkNativeBufferFormatPropertiesOHOS;

typedef struct VkImportNativeBufferInfoOHOS {
    VkStructureType            sType;
    const void*                pNext;
    struct OH_NativeBuffer*    buffer;
} VkImportNativeBufferInfoOHOS;

typedef struct VkMemoryGetNativeBufferInfoOHOS {
    VkStructureType    sType;
    const void*        pNext;
    VkDeviceMemory     memory;
} VkMemoryGetNativeBufferInfoOHOS;

typedef struct VkExternalFormatOHOS {
    VkStructureType    sType;
    void*              pNext;
    uint64_t           externalFormat;
} VkExternalFormatOHOS;

typedef VkResult (VKAPI_PTR *PFN_vkGetNativeBufferPropertiesOHOS)(VkDevice device, const struct OH_NativeBuffer* buffer, VkNativeBufferPropertiesOHOS* pProperties);
typedef VkResult (VKAPI_PTR *PFN_vkGetMemoryNativeBufferOHOS)(VkDevice device, const VkMemoryGetNativeBufferInfoOHOS* pInfo, struct OH_NativeBuffer** pBuffer);

#ifndef VK_NO_PROTOTYPES
VKAPI_ATTR VkResult VKAPI_CALL vkGetNativeBufferPropertiesOHOS(
    VkDevice                                    device,
    const struct OH_NativeBuffer*               buffer,
    VkNativeBufferPropertiesOHOS*               pProperties);

VKAPI_ATTR VkResult VKAPI_CALL vkGetMemoryNativeBufferOHOS(
    VkDevice                                    device,
    const VkMemoryGetNativeBufferInfoOHOS*      pInfo,
    struct OH_NativeBuffer**                    pBuffer);
#endif

#ifdef __cplusplus
}
#endif

#endif
