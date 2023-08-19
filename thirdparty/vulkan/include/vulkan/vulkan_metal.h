#ifndef VULKAN_METAL_H_
#define VULKAN_METAL_H_ 1

/*
** Copyright 2015-2023 The Khronos Group Inc.
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



#define VK_EXT_metal_surface 1
#ifdef __OBJC__
@class CAMetalLayer;
#else
typedef void CAMetalLayer;
#endif

#define VK_EXT_METAL_SURFACE_SPEC_VERSION 1
#define VK_EXT_METAL_SURFACE_EXTENSION_NAME "VK_EXT_metal_surface"
typedef VkFlags VkMetalSurfaceCreateFlagsEXT;
typedef struct VkMetalSurfaceCreateInfoEXT {
    VkStructureType                 sType;
    const void*                     pNext;
    VkMetalSurfaceCreateFlagsEXT    flags;
    const CAMetalLayer*             pLayer;
} VkMetalSurfaceCreateInfoEXT;

typedef VkResult (VKAPI_PTR *PFN_vkCreateMetalSurfaceEXT)(VkInstance instance, const VkMetalSurfaceCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSurfaceKHR* pSurface);

#ifndef VK_NO_PROTOTYPES
VKAPI_ATTR VkResult VKAPI_CALL vkCreateMetalSurfaceEXT(
    VkInstance                                  instance,
    const VkMetalSurfaceCreateInfoEXT*          pCreateInfo,
    const VkAllocationCallbacks*                pAllocator,
    VkSurfaceKHR*                               pSurface);
#endif


#define VK_EXT_metal_objects 1
#ifdef __OBJC__
@protocol MTLDevice;
typedef id<MTLDevice> MTLDevice_id;
#else
typedef void* MTLDevice_id;
#endif

#ifdef __OBJC__
@protocol MTLCommandQueue;
typedef id<MTLCommandQueue> MTLCommandQueue_id;
#else
typedef void* MTLCommandQueue_id;
#endif

#ifdef __OBJC__
@protocol MTLBuffer;
typedef id<MTLBuffer> MTLBuffer_id;
#else
typedef void* MTLBuffer_id;
#endif

#ifdef __OBJC__
@protocol MTLTexture;
typedef id<MTLTexture> MTLTexture_id;
#else
typedef void* MTLTexture_id;
#endif

typedef struct __IOSurface* IOSurfaceRef;
#ifdef __OBJC__
@protocol MTLSharedEvent;
typedef id<MTLSharedEvent> MTLSharedEvent_id;
#else
typedef void* MTLSharedEvent_id;
#endif

#define VK_EXT_METAL_OBJECTS_SPEC_VERSION 1
#define VK_EXT_METAL_OBJECTS_EXTENSION_NAME "VK_EXT_metal_objects"

typedef enum VkExportMetalObjectTypeFlagBitsEXT {
    VK_EXPORT_METAL_OBJECT_TYPE_METAL_DEVICE_BIT_EXT = 0x00000001,
    VK_EXPORT_METAL_OBJECT_TYPE_METAL_COMMAND_QUEUE_BIT_EXT = 0x00000002,
    VK_EXPORT_METAL_OBJECT_TYPE_METAL_BUFFER_BIT_EXT = 0x00000004,
    VK_EXPORT_METAL_OBJECT_TYPE_METAL_TEXTURE_BIT_EXT = 0x00000008,
    VK_EXPORT_METAL_OBJECT_TYPE_METAL_IOSURFACE_BIT_EXT = 0x00000010,
    VK_EXPORT_METAL_OBJECT_TYPE_METAL_SHARED_EVENT_BIT_EXT = 0x00000020,
    VK_EXPORT_METAL_OBJECT_TYPE_FLAG_BITS_MAX_ENUM_EXT = 0x7FFFFFFF
} VkExportMetalObjectTypeFlagBitsEXT;
typedef VkFlags VkExportMetalObjectTypeFlagsEXT;
typedef struct VkExportMetalObjectCreateInfoEXT {
    VkStructureType                       sType;
    const void*                           pNext;
    VkExportMetalObjectTypeFlagBitsEXT    exportObjectType;
} VkExportMetalObjectCreateInfoEXT;

typedef struct VkExportMetalObjectsInfoEXT {
    VkStructureType    sType;
    const void*        pNext;
} VkExportMetalObjectsInfoEXT;

typedef struct VkExportMetalDeviceInfoEXT {
    VkStructureType    sType;
    const void*        pNext;
    MTLDevice_id       mtlDevice;
} VkExportMetalDeviceInfoEXT;

typedef struct VkExportMetalCommandQueueInfoEXT {
    VkStructureType       sType;
    const void*           pNext;
    VkQueue               queue;
    MTLCommandQueue_id    mtlCommandQueue;
} VkExportMetalCommandQueueInfoEXT;

typedef struct VkExportMetalBufferInfoEXT {
    VkStructureType    sType;
    const void*        pNext;
    VkDeviceMemory     memory;
    MTLBuffer_id       mtlBuffer;
} VkExportMetalBufferInfoEXT;

typedef struct VkImportMetalBufferInfoEXT {
    VkStructureType    sType;
    const void*        pNext;
    MTLBuffer_id       mtlBuffer;
} VkImportMetalBufferInfoEXT;

typedef struct VkExportMetalTextureInfoEXT {
    VkStructureType          sType;
    const void*              pNext;
    VkImage                  image;
    VkImageView              imageView;
    VkBufferView             bufferView;
    VkImageAspectFlagBits    plane;
    MTLTexture_id            mtlTexture;
} VkExportMetalTextureInfoEXT;

typedef struct VkImportMetalTextureInfoEXT {
    VkStructureType          sType;
    const void*              pNext;
    VkImageAspectFlagBits    plane;
    MTLTexture_id            mtlTexture;
} VkImportMetalTextureInfoEXT;

typedef struct VkExportMetalIOSurfaceInfoEXT {
    VkStructureType    sType;
    const void*        pNext;
    VkImage            image;
    IOSurfaceRef       ioSurface;
} VkExportMetalIOSurfaceInfoEXT;

typedef struct VkImportMetalIOSurfaceInfoEXT {
    VkStructureType    sType;
    const void*        pNext;
    IOSurfaceRef       ioSurface;
} VkImportMetalIOSurfaceInfoEXT;

typedef struct VkExportMetalSharedEventInfoEXT {
    VkStructureType      sType;
    const void*          pNext;
    VkSemaphore          semaphore;
    VkEvent              event;
    MTLSharedEvent_id    mtlSharedEvent;
} VkExportMetalSharedEventInfoEXT;

typedef struct VkImportMetalSharedEventInfoEXT {
    VkStructureType      sType;
    const void*          pNext;
    MTLSharedEvent_id    mtlSharedEvent;
} VkImportMetalSharedEventInfoEXT;

typedef void (VKAPI_PTR *PFN_vkExportMetalObjectsEXT)(VkDevice device, VkExportMetalObjectsInfoEXT* pMetalObjectsInfo);

#ifndef VK_NO_PROTOTYPES
VKAPI_ATTR void VKAPI_CALL vkExportMetalObjectsEXT(
    VkDevice                                    device,
    VkExportMetalObjectsInfoEXT*                pMetalObjectsInfo);
#endif

#ifdef __cplusplus
}
#endif

#endif
