/* -*- tab-width: 4; -*- */
/* vi: set sw=2 ts=4 expandtab: */

#ifndef KTX_H_C54B42AEE39611E68E1E4FF8C51D1C66
#define KTX_H_C54B42AEE39611E68E1E4FF8C51D1C66

/*
 * Copyright 2017-2020 The Khronos Group, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @internal
 * @file
 * @~English
 *
 * @brief Declares the public functions and structures of the
 *        KTX Vulkan texture loading API.
 *
 * A separate header file is used to avoid extra dependencies for those not
 * using Vulkan. The nature of the Vulkan API, rampant structures and enums,
 * means that vulkan.h must be included @e before including this file. The
 * alternative is duplicating unattractively large parts of it.
 *
 * @author Mark Callow, Edgewise Consulting
 *
 * $Date$
 */

#include <ktx.h>

#if 0
/* Avoid Vulkan include file */
#define VK_DEFINE_HANDLE(object) typedef struct object##_T* object;

#if defined(__LP64__) || defined(_WIN64) || defined(__x86_64__) || defined(_M_X64) || defined(__ia64) || defined (_M_IA64) || defined(__aarch64__) || defined(__powerpc64__)
        #define VK_DEFINE_NON_DISPATCHABLE_HANDLE(object) typedef struct object##_T *object;
#else
        #define VK_DEFINE_NON_DISPATCHABLE_HANDLE(object) typedef uint64_t object;
#endif

VK_DEFINE_HANDLE(VkPhysicalDevice)
VK_DEFINE_HANDLE(VkDevice)
VK_DEFINE_HANDLE(VkQueue)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkCommandPool)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkDeviceMemory)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkImage)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkImageView)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkSampler)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @struct ktxVulkanFunctions
 * @~English
 * @brief Struct for applications to pass Vulkan function pointers to the
 *        ktxTexture_VkUpload functions via a ktxVulkanDeviceInfo struct.
 *
 * @c vkGetInstanceProcAddr and @c vkGetDeviceProcAddr should be set, others
 * are optional.
 */
typedef struct ktxVulkanFunctions {
    // These are functions pointers we need to perform our vulkan duties.
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr;

    // These we optionally specify
    PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers;
    PFN_vkAllocateMemory vkAllocateMemory;
    PFN_vkBeginCommandBuffer vkBeginCommandBuffer;
    PFN_vkBindBufferMemory vkBindBufferMemory;
    PFN_vkBindImageMemory vkBindImageMemory;
    PFN_vkCmdBlitImage vkCmdBlitImage;
    PFN_vkCmdCopyBufferToImage vkCmdCopyBufferToImage;
    PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier;
    PFN_vkCreateImage vkCreateImage;
    PFN_vkDestroyImage vkDestroyImage;
    PFN_vkCreateBuffer vkCreateBuffer;
    PFN_vkDestroyBuffer vkDestroyBuffer;
    PFN_vkCreateFence vkCreateFence;
    PFN_vkDestroyFence vkDestroyFence;
    PFN_vkEndCommandBuffer vkEndCommandBuffer;
    PFN_vkFreeCommandBuffers vkFreeCommandBuffers;
    PFN_vkFreeMemory vkFreeMemory;
    PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements;
    PFN_vkGetImageMemoryRequirements vkGetImageMemoryRequirements;
    PFN_vkGetImageSubresourceLayout vkGetImageSubresourceLayout;
    PFN_vkGetPhysicalDeviceImageFormatProperties vkGetPhysicalDeviceImageFormatProperties;
    PFN_vkGetPhysicalDeviceFormatProperties vkGetPhysicalDeviceFormatProperties;
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties;
    PFN_vkMapMemory vkMapMemory;
    PFN_vkQueueSubmit vkQueueSubmit;
    PFN_vkQueueWaitIdle vkQueueWaitIdle;
    PFN_vkUnmapMemory vkUnmapMemory;
    PFN_vkWaitForFences vkWaitForFences;
} ktxVulkanFunctions;

/**
 * @class ktxVulkanTexture
 * @~English
 * @brief Struct for returning information about the Vulkan texture image
 *        created by the ktxTexture_VkUpload* functions.
 *
 * Creation of these objects is internal to the upload functions.
 */
typedef struct ktxVulkanTexture
{
    PFN_vkDestroyImage vkDestroyImage; /*!< Pointer to vkDestroyImage function */
    PFN_vkFreeMemory vkFreeMemory; /*!< Pointer to vkFreeMemory function */

    VkImage image; /*!< Handle to the Vulkan image created by the loader. */
    VkFormat imageFormat;     /*!< Format of the image data. */
    VkImageLayout imageLayout; /*!< Layout of the created image. Has the same
                                    value as @p layout parameter passed to the
                                    loader. */
    VkDeviceMemory deviceMemory; /*!< The memory allocated for the image on
                                  the Vulkan device. */
    VkImageViewType viewType; /*!< ViewType corresponding to @p image. Reflects
                                   the dimensionality, cubeness and arrayness
                                   of the image. */
    uint32_t width; /*!< The width of the image. */
    uint32_t height; /*!< The height of the image. */
    uint32_t depth; /*!< The depth of the image. */
    uint32_t levelCount; /*!< The number of MIP levels in the image. */
    uint32_t layerCount; /*!< The number of array layers in the image. */
} ktxVulkanTexture;

KTX_API void KTX_APIENTRY
ktxVulkanTexture_Destruct(ktxVulkanTexture* This, VkDevice device,
                          const VkAllocationCallbacks* pAllocator);




/**
 * @class ktxVulkanDeviceInfo
 * @~English
 * @brief Struct for passing information about the Vulkan device on which
 *        to create images to the texture image loading functions.
 *
 * Avoids passing a large number of parameters to each loading function.
 * Use of ktxVulkanDeviceInfo_create() or ktxVulkanDeviceInfo_construct() to
 * populate this structure is highly recommended.
 *
 * @code
    ktxVulkanDeviceInfo vdi;
    ktxVulkanTexture texture;
 
    vdi = ktxVulkanDeviceInfo_create(physicalDevice,
                                     device,
                                     queue,
                                     cmdPool,
                                     &allocator);
    ktxLoadVkTextureN("texture_1.ktx", vdi, &texture, NULL, NULL);
    // ...
    ktxLoadVkTextureN("texture_n.ktx", vdi, &texture, NULL, NULL);
    ktxVulkanDeviceInfo_destroy(vdi);
 * @endcode
 */
typedef struct ktxVulkanDeviceInfo {
    VkInstance instance; /*!< Instance used to communicate with vulkan. */
    VkPhysicalDevice physicalDevice; /*!< Handle of the physical device. */
    VkDevice device; /*!< Handle of the logical device. */
    VkQueue queue; /*!< Handle to the queue to which to submit commands. */
    VkCommandBuffer cmdBuffer; /*!< Handle of the cmdBuffer to use. */
    /** Handle of the command pool from which to allocate the command buffer. */
    VkCommandPool cmdPool;
    /** Pointer to the allocator to use for the command buffer and created
     * images.
     */
    const VkAllocationCallbacks* pAllocator;
    /** Memory properties of the Vulkan physical device. */
    VkPhysicalDeviceMemoryProperties deviceMemoryProperties;

    /** The functions needed to operate functions */
    ktxVulkanFunctions vkFuncs;
} ktxVulkanDeviceInfo;


KTX_API ktxVulkanDeviceInfo* KTX_APIENTRY
ktxVulkanDeviceInfo_CreateEx(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device,
                           VkQueue queue, VkCommandPool cmdPool,
                           const VkAllocationCallbacks* pAllocator,
                           const ktxVulkanFunctions* pFunctions);

KTX_API ktxVulkanDeviceInfo* KTX_APIENTRY
ktxVulkanDeviceInfo_Create(VkPhysicalDevice physicalDevice, VkDevice device,
                           VkQueue queue, VkCommandPool cmdPool,
                           const VkAllocationCallbacks* pAllocator);

KTX_API KTX_error_code KTX_APIENTRY
ktxVulkanDeviceInfo_Construct(ktxVulkanDeviceInfo* This,
                         VkPhysicalDevice physicalDevice, VkDevice device,
                         VkQueue queue, VkCommandPool cmdPool,
                         const VkAllocationCallbacks* pAllocator);

KTX_API KTX_error_code KTX_APIENTRY
ktxVulkanDeviceInfo_ConstructEx(ktxVulkanDeviceInfo* This,
                              VkInstance instance,
                              VkPhysicalDevice physicalDevice, VkDevice device,
                              VkQueue queue, VkCommandPool cmdPool,
                              const VkAllocationCallbacks* pAllocator,
                              const ktxVulkanFunctions* pFunctions);

KTX_API void KTX_APIENTRY
ktxVulkanDeviceInfo_Destruct(ktxVulkanDeviceInfo* This);
KTX_API void KTX_APIENTRY
ktxVulkanDeviceInfo_Destroy(ktxVulkanDeviceInfo* This);
KTX_API KTX_error_code KTX_APIENTRY
ktxTexture_VkUploadEx(ktxTexture* This, ktxVulkanDeviceInfo* vdi,
                      ktxVulkanTexture* vkTexture,
                      VkImageTiling tiling,
                      VkImageUsageFlags usageFlags,
                      VkImageLayout finalLayout);
KTX_API KTX_error_code KTX_APIENTRY
ktxTexture_VkUpload(ktxTexture* texture, ktxVulkanDeviceInfo* vdi,
                    ktxVulkanTexture *vkTexture);
KTX_API KTX_error_code KTX_APIENTRY
ktxTexture1_VkUploadEx(ktxTexture1* This, ktxVulkanDeviceInfo* vdi,
                       ktxVulkanTexture* vkTexture,
                       VkImageTiling tiling,
                       VkImageUsageFlags usageFlags,
                       VkImageLayout finalLayout);
KTX_API KTX_error_code KTX_APIENTRY
ktxTexture1_VkUpload(ktxTexture1* texture, ktxVulkanDeviceInfo* vdi,
                    ktxVulkanTexture *vkTexture);
KTX_API KTX_error_code KTX_APIENTRY
ktxTexture2_VkUploadEx(ktxTexture2* This, ktxVulkanDeviceInfo* vdi,
                       ktxVulkanTexture* vkTexture,
                       VkImageTiling tiling,
                       VkImageUsageFlags usageFlags,
                       VkImageLayout finalLayout);
KTX_API KTX_error_code KTX_APIENTRY
ktxTexture2_VkUpload(ktxTexture2* texture, ktxVulkanDeviceInfo* vdi,
                     ktxVulkanTexture *vkTexture);

KTX_API VkFormat KTX_APIENTRY
ktxTexture_GetVkFormat(ktxTexture* This);

KTX_API VkFormat KTX_APIENTRY
ktxTexture1_GetVkFormat(ktxTexture1* This);

KTX_API VkFormat KTX_APIENTRY
ktxTexture2_GetVkFormat(ktxTexture2* This);

#ifdef __cplusplus
}
#endif

#endif /* KTX_H_A55A6F00956F42F3A137C11929827FE1 */
