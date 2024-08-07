#ifndef OPENXR_PLATFORM_H_
#define OPENXR_PLATFORM_H_ 1

/*
** Copyright 2017-2024, The Khronos Group Inc.
**
** SPDX-License-Identifier: Apache-2.0 OR MIT
*/

/*
** This header is generated from the Khronos OpenXR XML API Registry.
**
*/

#include "openxr.h"

#ifdef __cplusplus
extern "C" {
#endif


#ifdef XR_USE_PLATFORM_ANDROID

// XR_KHR_android_thread_settings is a preprocessor guard. Do not pass it to API calls.
#define XR_KHR_android_thread_settings 1
#define XR_KHR_android_thread_settings_SPEC_VERSION 6
#define XR_KHR_ANDROID_THREAD_SETTINGS_EXTENSION_NAME "XR_KHR_android_thread_settings"

typedef enum XrAndroidThreadTypeKHR {
    XR_ANDROID_THREAD_TYPE_APPLICATION_MAIN_KHR = 1,
    XR_ANDROID_THREAD_TYPE_APPLICATION_WORKER_KHR = 2,
    XR_ANDROID_THREAD_TYPE_RENDERER_MAIN_KHR = 3,
    XR_ANDROID_THREAD_TYPE_RENDERER_WORKER_KHR = 4,
    XR_ANDROID_THREAD_TYPE_MAX_ENUM_KHR = 0x7FFFFFFF
} XrAndroidThreadTypeKHR;
typedef XrResult (XRAPI_PTR *PFN_xrSetAndroidApplicationThreadKHR)(XrSession session, XrAndroidThreadTypeKHR threadType, uint32_t threadId);

#ifndef XR_NO_PROTOTYPES
#ifdef XR_EXTENSION_PROTOTYPES
XRAPI_ATTR XrResult XRAPI_CALL xrSetAndroidApplicationThreadKHR(
    XrSession                                   session,
    XrAndroidThreadTypeKHR                      threadType,
    uint32_t                                    threadId);
#endif /* XR_EXTENSION_PROTOTYPES */
#endif /* !XR_NO_PROTOTYPES */
#endif /* XR_USE_PLATFORM_ANDROID */

#ifdef XR_USE_PLATFORM_ANDROID

// XR_KHR_android_surface_swapchain is a preprocessor guard. Do not pass it to API calls.
#define XR_KHR_android_surface_swapchain 1
#define XR_KHR_android_surface_swapchain_SPEC_VERSION 4
#define XR_KHR_ANDROID_SURFACE_SWAPCHAIN_EXTENSION_NAME "XR_KHR_android_surface_swapchain"
typedef XrResult (XRAPI_PTR *PFN_xrCreateSwapchainAndroidSurfaceKHR)(XrSession session, const XrSwapchainCreateInfo* info, XrSwapchain* swapchain, jobject* surface);

#ifndef XR_NO_PROTOTYPES
#ifdef XR_EXTENSION_PROTOTYPES
XRAPI_ATTR XrResult XRAPI_CALL xrCreateSwapchainAndroidSurfaceKHR(
    XrSession                                   session,
    const XrSwapchainCreateInfo*                info,
    XrSwapchain*                                swapchain,
    jobject*                                    surface);
#endif /* XR_EXTENSION_PROTOTYPES */
#endif /* !XR_NO_PROTOTYPES */
#endif /* XR_USE_PLATFORM_ANDROID */

#ifdef XR_USE_PLATFORM_ANDROID

// XR_KHR_android_create_instance is a preprocessor guard. Do not pass it to API calls.
#define XR_KHR_android_create_instance 1
#define XR_KHR_android_create_instance_SPEC_VERSION 3
#define XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME "XR_KHR_android_create_instance"
// XrInstanceCreateInfoAndroidKHR extends XrInstanceCreateInfo
typedef struct XrInstanceCreateInfoAndroidKHR {
    XrStructureType             type;
    const void* XR_MAY_ALIAS    next;
    void* XR_MAY_ALIAS          applicationVM;
    void* XR_MAY_ALIAS          applicationActivity;
} XrInstanceCreateInfoAndroidKHR;

#endif /* XR_USE_PLATFORM_ANDROID */

#ifdef XR_USE_GRAPHICS_API_VULKAN

// XR_KHR_vulkan_swapchain_format_list is a preprocessor guard. Do not pass it to API calls.
#define XR_KHR_vulkan_swapchain_format_list 1
#define XR_KHR_vulkan_swapchain_format_list_SPEC_VERSION 4
#define XR_KHR_VULKAN_SWAPCHAIN_FORMAT_LIST_EXTENSION_NAME "XR_KHR_vulkan_swapchain_format_list"
typedef struct XrVulkanSwapchainFormatListCreateInfoKHR {
    XrStructureType             type;
    const void* XR_MAY_ALIAS    next;
    uint32_t                    viewFormatCount;
    const VkFormat*             viewFormats;
} XrVulkanSwapchainFormatListCreateInfoKHR;

#endif /* XR_USE_GRAPHICS_API_VULKAN */

#ifdef XR_USE_GRAPHICS_API_OPENGL

// XR_KHR_opengl_enable is a preprocessor guard. Do not pass it to API calls.
#define XR_KHR_opengl_enable 1
#define XR_KHR_opengl_enable_SPEC_VERSION 10
#define XR_KHR_OPENGL_ENABLE_EXTENSION_NAME "XR_KHR_opengl_enable"
#ifdef XR_USE_PLATFORM_WIN32
// XrGraphicsBindingOpenGLWin32KHR extends XrSessionCreateInfo
typedef struct XrGraphicsBindingOpenGLWin32KHR {
    XrStructureType             type;
    const void* XR_MAY_ALIAS    next;
    HDC                         hDC;
    HGLRC                       hGLRC;
} XrGraphicsBindingOpenGLWin32KHR;
#endif // XR_USE_PLATFORM_WIN32

#ifdef XR_USE_PLATFORM_XLIB
// XrGraphicsBindingOpenGLXlibKHR extends XrSessionCreateInfo
typedef struct XrGraphicsBindingOpenGLXlibKHR {
    XrStructureType             type;
    const void* XR_MAY_ALIAS    next;
    Display*                    xDisplay;
    uint32_t                    visualid;
    GLXFBConfig                 glxFBConfig;
    GLXDrawable                 glxDrawable;
    GLXContext                  glxContext;
} XrGraphicsBindingOpenGLXlibKHR;
#endif // XR_USE_PLATFORM_XLIB

#ifdef XR_USE_PLATFORM_XCB
// XrGraphicsBindingOpenGLXcbKHR extends XrSessionCreateInfo
typedef struct XrGraphicsBindingOpenGLXcbKHR {
    XrStructureType             type;
    const void* XR_MAY_ALIAS    next;
    xcb_connection_t*           connection;
    uint32_t                    screenNumber;
    xcb_glx_fbconfig_t          fbconfigid;
    xcb_visualid_t              visualid;
    xcb_glx_drawable_t          glxDrawable;
    xcb_glx_context_t           glxContext;
} XrGraphicsBindingOpenGLXcbKHR;
#endif // XR_USE_PLATFORM_XCB

#ifdef XR_USE_PLATFORM_WAYLAND
// XrGraphicsBindingOpenGLWaylandKHR extends XrSessionCreateInfo
typedef struct XrGraphicsBindingOpenGLWaylandKHR {
    XrStructureType             type;
    const void* XR_MAY_ALIAS    next;
    struct wl_display*          display;
} XrGraphicsBindingOpenGLWaylandKHR;
#endif // XR_USE_PLATFORM_WAYLAND

typedef struct XrSwapchainImageOpenGLKHR {
    XrStructureType       type;
    void* XR_MAY_ALIAS    next;
    uint32_t              image;
} XrSwapchainImageOpenGLKHR;

typedef struct XrGraphicsRequirementsOpenGLKHR {
    XrStructureType       type;
    void* XR_MAY_ALIAS    next;
    XrVersion             minApiVersionSupported;
    XrVersion             maxApiVersionSupported;
} XrGraphicsRequirementsOpenGLKHR;

typedef XrResult (XRAPI_PTR *PFN_xrGetOpenGLGraphicsRequirementsKHR)(XrInstance instance, XrSystemId systemId, XrGraphicsRequirementsOpenGLKHR* graphicsRequirements);

#ifndef XR_NO_PROTOTYPES
#ifdef XR_EXTENSION_PROTOTYPES
XRAPI_ATTR XrResult XRAPI_CALL xrGetOpenGLGraphicsRequirementsKHR(
    XrInstance                                  instance,
    XrSystemId                                  systemId,
    XrGraphicsRequirementsOpenGLKHR*            graphicsRequirements);
#endif /* XR_EXTENSION_PROTOTYPES */
#endif /* !XR_NO_PROTOTYPES */
#endif /* XR_USE_GRAPHICS_API_OPENGL */

#ifdef XR_USE_GRAPHICS_API_OPENGL_ES

// XR_KHR_opengl_es_enable is a preprocessor guard. Do not pass it to API calls.
#define XR_KHR_opengl_es_enable 1
#define XR_KHR_opengl_es_enable_SPEC_VERSION 8
#define XR_KHR_OPENGL_ES_ENABLE_EXTENSION_NAME "XR_KHR_opengl_es_enable"
#ifdef XR_USE_PLATFORM_ANDROID
// XrGraphicsBindingOpenGLESAndroidKHR extends XrSessionCreateInfo
typedef struct XrGraphicsBindingOpenGLESAndroidKHR {
    XrStructureType             type;
    const void* XR_MAY_ALIAS    next;
    EGLDisplay                  display;
    EGLConfig                   config;
    EGLContext                  context;
} XrGraphicsBindingOpenGLESAndroidKHR;
#endif // XR_USE_PLATFORM_ANDROID

typedef struct XrSwapchainImageOpenGLESKHR {
    XrStructureType       type;
    void* XR_MAY_ALIAS    next;
    uint32_t              image;
} XrSwapchainImageOpenGLESKHR;

typedef struct XrGraphicsRequirementsOpenGLESKHR {
    XrStructureType       type;
    void* XR_MAY_ALIAS    next;
    XrVersion             minApiVersionSupported;
    XrVersion             maxApiVersionSupported;
} XrGraphicsRequirementsOpenGLESKHR;

typedef XrResult (XRAPI_PTR *PFN_xrGetOpenGLESGraphicsRequirementsKHR)(XrInstance instance, XrSystemId systemId, XrGraphicsRequirementsOpenGLESKHR* graphicsRequirements);

#ifndef XR_NO_PROTOTYPES
#ifdef XR_EXTENSION_PROTOTYPES
XRAPI_ATTR XrResult XRAPI_CALL xrGetOpenGLESGraphicsRequirementsKHR(
    XrInstance                                  instance,
    XrSystemId                                  systemId,
    XrGraphicsRequirementsOpenGLESKHR*          graphicsRequirements);
#endif /* XR_EXTENSION_PROTOTYPES */
#endif /* !XR_NO_PROTOTYPES */
#endif /* XR_USE_GRAPHICS_API_OPENGL_ES */

#ifdef XR_USE_GRAPHICS_API_VULKAN

// XR_KHR_vulkan_enable is a preprocessor guard. Do not pass it to API calls.
#define XR_KHR_vulkan_enable 1
#define XR_KHR_vulkan_enable_SPEC_VERSION 8
#define XR_KHR_VULKAN_ENABLE_EXTENSION_NAME "XR_KHR_vulkan_enable"
// XrGraphicsBindingVulkanKHR extends XrSessionCreateInfo
typedef struct XrGraphicsBindingVulkanKHR {
    XrStructureType             type;
    const void* XR_MAY_ALIAS    next;
    VkInstance                  instance;
    VkPhysicalDevice            physicalDevice;
    VkDevice                    device;
    uint32_t                    queueFamilyIndex;
    uint32_t                    queueIndex;
} XrGraphicsBindingVulkanKHR;

typedef struct XrSwapchainImageVulkanKHR {
    XrStructureType       type;
    void* XR_MAY_ALIAS    next;
    VkImage               image;
} XrSwapchainImageVulkanKHR;

typedef struct XrGraphicsRequirementsVulkanKHR {
    XrStructureType       type;
    void* XR_MAY_ALIAS    next;
    XrVersion             minApiVersionSupported;
    XrVersion             maxApiVersionSupported;
} XrGraphicsRequirementsVulkanKHR;

typedef XrResult (XRAPI_PTR *PFN_xrGetVulkanInstanceExtensionsKHR)(XrInstance instance, XrSystemId systemId, uint32_t bufferCapacityInput, uint32_t* bufferCountOutput, char* buffer);
typedef XrResult (XRAPI_PTR *PFN_xrGetVulkanDeviceExtensionsKHR)(XrInstance instance, XrSystemId systemId, uint32_t bufferCapacityInput, uint32_t* bufferCountOutput, char* buffer);
typedef XrResult (XRAPI_PTR *PFN_xrGetVulkanGraphicsDeviceKHR)(XrInstance instance, XrSystemId systemId, VkInstance vkInstance, VkPhysicalDevice* vkPhysicalDevice);
typedef XrResult (XRAPI_PTR *PFN_xrGetVulkanGraphicsRequirementsKHR)(XrInstance instance, XrSystemId systemId, XrGraphicsRequirementsVulkanKHR* graphicsRequirements);

#ifndef XR_NO_PROTOTYPES
#ifdef XR_EXTENSION_PROTOTYPES
XRAPI_ATTR XrResult XRAPI_CALL xrGetVulkanInstanceExtensionsKHR(
    XrInstance                                  instance,
    XrSystemId                                  systemId,
    uint32_t                                    bufferCapacityInput,
    uint32_t*                                   bufferCountOutput,
    char*                                       buffer);

XRAPI_ATTR XrResult XRAPI_CALL xrGetVulkanDeviceExtensionsKHR(
    XrInstance                                  instance,
    XrSystemId                                  systemId,
    uint32_t                                    bufferCapacityInput,
    uint32_t*                                   bufferCountOutput,
    char*                                       buffer);

XRAPI_ATTR XrResult XRAPI_CALL xrGetVulkanGraphicsDeviceKHR(
    XrInstance                                  instance,
    XrSystemId                                  systemId,
    VkInstance                                  vkInstance,
    VkPhysicalDevice*                           vkPhysicalDevice);

XRAPI_ATTR XrResult XRAPI_CALL xrGetVulkanGraphicsRequirementsKHR(
    XrInstance                                  instance,
    XrSystemId                                  systemId,
    XrGraphicsRequirementsVulkanKHR*            graphicsRequirements);
#endif /* XR_EXTENSION_PROTOTYPES */
#endif /* !XR_NO_PROTOTYPES */
#endif /* XR_USE_GRAPHICS_API_VULKAN */

#ifdef XR_USE_GRAPHICS_API_D3D11

// XR_KHR_D3D11_enable is a preprocessor guard. Do not pass it to API calls.
#define XR_KHR_D3D11_enable 1
#define XR_KHR_D3D11_enable_SPEC_VERSION  9
#define XR_KHR_D3D11_ENABLE_EXTENSION_NAME "XR_KHR_D3D11_enable"
// XrGraphicsBindingD3D11KHR extends XrSessionCreateInfo
typedef struct XrGraphicsBindingD3D11KHR {
    XrStructureType             type;
    const void* XR_MAY_ALIAS    next;
    ID3D11Device*               device;
} XrGraphicsBindingD3D11KHR;

typedef struct XrSwapchainImageD3D11KHR {
     XrStructureType      type;
    void* XR_MAY_ALIAS    next;
    ID3D11Texture2D*      texture;
} XrSwapchainImageD3D11KHR;

typedef struct XrGraphicsRequirementsD3D11KHR {
    XrStructureType       type;
    void* XR_MAY_ALIAS    next;
    LUID                  adapterLuid;
    D3D_FEATURE_LEVEL     minFeatureLevel;
} XrGraphicsRequirementsD3D11KHR;

typedef XrResult (XRAPI_PTR *PFN_xrGetD3D11GraphicsRequirementsKHR)(XrInstance instance, XrSystemId systemId, XrGraphicsRequirementsD3D11KHR* graphicsRequirements);

#ifndef XR_NO_PROTOTYPES
#ifdef XR_EXTENSION_PROTOTYPES
XRAPI_ATTR XrResult XRAPI_CALL xrGetD3D11GraphicsRequirementsKHR(
    XrInstance                                  instance,
    XrSystemId                                  systemId,
    XrGraphicsRequirementsD3D11KHR*             graphicsRequirements);
#endif /* XR_EXTENSION_PROTOTYPES */
#endif /* !XR_NO_PROTOTYPES */
#endif /* XR_USE_GRAPHICS_API_D3D11 */

#ifdef XR_USE_GRAPHICS_API_D3D12

// XR_KHR_D3D12_enable is a preprocessor guard. Do not pass it to API calls.
#define XR_KHR_D3D12_enable 1
#define XR_KHR_D3D12_enable_SPEC_VERSION  9
#define XR_KHR_D3D12_ENABLE_EXTENSION_NAME "XR_KHR_D3D12_enable"
// XrGraphicsBindingD3D12KHR extends XrSessionCreateInfo
typedef struct XrGraphicsBindingD3D12KHR {
    XrStructureType             type;
    const void* XR_MAY_ALIAS    next;
    ID3D12Device*               device;
    ID3D12CommandQueue*         queue;
} XrGraphicsBindingD3D12KHR;

typedef struct XrSwapchainImageD3D12KHR {
     XrStructureType      type;
    void* XR_MAY_ALIAS    next;
    ID3D12Resource*       texture;
} XrSwapchainImageD3D12KHR;

typedef struct XrGraphicsRequirementsD3D12KHR {
    XrStructureType       type;
    void* XR_MAY_ALIAS    next;
    LUID                  adapterLuid;
    D3D_FEATURE_LEVEL     minFeatureLevel;
} XrGraphicsRequirementsD3D12KHR;

typedef XrResult (XRAPI_PTR *PFN_xrGetD3D12GraphicsRequirementsKHR)(XrInstance instance, XrSystemId systemId, XrGraphicsRequirementsD3D12KHR* graphicsRequirements);

#ifndef XR_NO_PROTOTYPES
#ifdef XR_EXTENSION_PROTOTYPES
XRAPI_ATTR XrResult XRAPI_CALL xrGetD3D12GraphicsRequirementsKHR(
    XrInstance                                  instance,
    XrSystemId                                  systemId,
    XrGraphicsRequirementsD3D12KHR*             graphicsRequirements);
#endif /* XR_EXTENSION_PROTOTYPES */
#endif /* !XR_NO_PROTOTYPES */
#endif /* XR_USE_GRAPHICS_API_D3D12 */

#ifdef XR_USE_PLATFORM_WIN32

// XR_KHR_win32_convert_performance_counter_time is a preprocessor guard. Do not pass it to API calls.
#define XR_KHR_win32_convert_performance_counter_time 1
#define XR_KHR_win32_convert_performance_counter_time_SPEC_VERSION 1
#define XR_KHR_WIN32_CONVERT_PERFORMANCE_COUNTER_TIME_EXTENSION_NAME "XR_KHR_win32_convert_performance_counter_time"
typedef XrResult (XRAPI_PTR *PFN_xrConvertWin32PerformanceCounterToTimeKHR)(XrInstance instance, const LARGE_INTEGER* performanceCounter, XrTime* time);
typedef XrResult (XRAPI_PTR *PFN_xrConvertTimeToWin32PerformanceCounterKHR)(XrInstance instance, XrTime   time, LARGE_INTEGER* performanceCounter);

#ifndef XR_NO_PROTOTYPES
#ifdef XR_EXTENSION_PROTOTYPES
XRAPI_ATTR XrResult XRAPI_CALL xrConvertWin32PerformanceCounterToTimeKHR(
    XrInstance                                  instance,
    const LARGE_INTEGER*                        performanceCounter,
    XrTime*                                     time);

XRAPI_ATTR XrResult XRAPI_CALL xrConvertTimeToWin32PerformanceCounterKHR(
    XrInstance                                  instance,
    XrTime                                      time,
    LARGE_INTEGER*                              performanceCounter);
#endif /* XR_EXTENSION_PROTOTYPES */
#endif /* !XR_NO_PROTOTYPES */
#endif /* XR_USE_PLATFORM_WIN32 */

#ifdef XR_USE_TIMESPEC

// XR_KHR_convert_timespec_time is a preprocessor guard. Do not pass it to API calls.
#define XR_KHR_convert_timespec_time 1
#define XR_KHR_convert_timespec_time_SPEC_VERSION 1
#define XR_KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME "XR_KHR_convert_timespec_time"
typedef XrResult (XRAPI_PTR *PFN_xrConvertTimespecTimeToTimeKHR)(XrInstance instance, const struct timespec* timespecTime, XrTime* time);
typedef XrResult (XRAPI_PTR *PFN_xrConvertTimeToTimespecTimeKHR)(XrInstance instance, XrTime   time, struct timespec* timespecTime);

#ifndef XR_NO_PROTOTYPES
#ifdef XR_EXTENSION_PROTOTYPES
XRAPI_ATTR XrResult XRAPI_CALL xrConvertTimespecTimeToTimeKHR(
    XrInstance                                  instance,
    const struct timespec*                      timespecTime,
    XrTime*                                     time);

XRAPI_ATTR XrResult XRAPI_CALL xrConvertTimeToTimespecTimeKHR(
    XrInstance                                  instance,
    XrTime                                      time,
    struct timespec*                            timespecTime);
#endif /* XR_EXTENSION_PROTOTYPES */
#endif /* !XR_NO_PROTOTYPES */
#endif /* XR_USE_TIMESPEC */

#ifdef XR_USE_PLATFORM_ANDROID

// XR_KHR_loader_init_android is a preprocessor guard. Do not pass it to API calls.
#define XR_KHR_loader_init_android 1
#define XR_KHR_loader_init_android_SPEC_VERSION 1
#define XR_KHR_LOADER_INIT_ANDROID_EXTENSION_NAME "XR_KHR_loader_init_android"
typedef struct XrLoaderInitInfoAndroidKHR {
    XrStructureType             type;
    const void* XR_MAY_ALIAS    next;
    void* XR_MAY_ALIAS          applicationVM;
    void* XR_MAY_ALIAS          applicationContext;
} XrLoaderInitInfoAndroidKHR;

#endif /* XR_USE_PLATFORM_ANDROID */

#ifdef XR_USE_GRAPHICS_API_VULKAN

// XR_KHR_vulkan_enable2 is a preprocessor guard. Do not pass it to API calls.
#define XR_KHR_vulkan_enable2 1
#define XR_KHR_vulkan_enable2_SPEC_VERSION 2
#define XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME "XR_KHR_vulkan_enable2"
typedef XrFlags64 XrVulkanInstanceCreateFlagsKHR;

// Flag bits for XrVulkanInstanceCreateFlagsKHR

typedef XrFlags64 XrVulkanDeviceCreateFlagsKHR;

// Flag bits for XrVulkanDeviceCreateFlagsKHR

typedef struct XrVulkanInstanceCreateInfoKHR {
    XrStructureType                   type;
    const void* XR_MAY_ALIAS          next;
    XrSystemId                        systemId;
    XrVulkanInstanceCreateFlagsKHR    createFlags;
    PFN_vkGetInstanceProcAddr         pfnGetInstanceProcAddr;
    const VkInstanceCreateInfo*       vulkanCreateInfo;
    const VkAllocationCallbacks*      vulkanAllocator;
} XrVulkanInstanceCreateInfoKHR;

typedef struct XrVulkanDeviceCreateInfoKHR {
    XrStructureType                 type;
    const void* XR_MAY_ALIAS        next;
    XrSystemId                      systemId;
    XrVulkanDeviceCreateFlagsKHR    createFlags;
    PFN_vkGetInstanceProcAddr       pfnGetInstanceProcAddr;
    VkPhysicalDevice                vulkanPhysicalDevice;
    const VkDeviceCreateInfo*       vulkanCreateInfo;
    const VkAllocationCallbacks*    vulkanAllocator;
} XrVulkanDeviceCreateInfoKHR;

typedef XrGraphicsBindingVulkanKHR XrGraphicsBindingVulkan2KHR;

typedef struct XrVulkanGraphicsDeviceGetInfoKHR {
    XrStructureType             type;
    const void* XR_MAY_ALIAS    next;
    XrSystemId                  systemId;
    VkInstance                  vulkanInstance;
} XrVulkanGraphicsDeviceGetInfoKHR;

typedef XrSwapchainImageVulkanKHR XrSwapchainImageVulkan2KHR;

typedef XrGraphicsRequirementsVulkanKHR XrGraphicsRequirementsVulkan2KHR;

typedef XrResult (XRAPI_PTR *PFN_xrCreateVulkanInstanceKHR)(XrInstance instance, const XrVulkanInstanceCreateInfoKHR* createInfo, VkInstance* vulkanInstance, VkResult* vulkanResult);
typedef XrResult (XRAPI_PTR *PFN_xrCreateVulkanDeviceKHR)(XrInstance instance, const XrVulkanDeviceCreateInfoKHR* createInfo, VkDevice* vulkanDevice, VkResult* vulkanResult);
typedef XrResult (XRAPI_PTR *PFN_xrGetVulkanGraphicsDevice2KHR)(XrInstance instance, const XrVulkanGraphicsDeviceGetInfoKHR* getInfo, VkPhysicalDevice* vulkanPhysicalDevice);
typedef XrResult (XRAPI_PTR *PFN_xrGetVulkanGraphicsRequirements2KHR)(XrInstance instance, XrSystemId systemId, XrGraphicsRequirementsVulkanKHR* graphicsRequirements);

#ifndef XR_NO_PROTOTYPES
#ifdef XR_EXTENSION_PROTOTYPES
XRAPI_ATTR XrResult XRAPI_CALL xrCreateVulkanInstanceKHR(
    XrInstance                                  instance,
    const XrVulkanInstanceCreateInfoKHR*        createInfo,
    VkInstance*                                 vulkanInstance,
    VkResult*                                   vulkanResult);

XRAPI_ATTR XrResult XRAPI_CALL xrCreateVulkanDeviceKHR(
    XrInstance                                  instance,
    const XrVulkanDeviceCreateInfoKHR*          createInfo,
    VkDevice*                                   vulkanDevice,
    VkResult*                                   vulkanResult);

XRAPI_ATTR XrResult XRAPI_CALL xrGetVulkanGraphicsDevice2KHR(
    XrInstance                                  instance,
    const XrVulkanGraphicsDeviceGetInfoKHR*     getInfo,
    VkPhysicalDevice*                           vulkanPhysicalDevice);

XRAPI_ATTR XrResult XRAPI_CALL xrGetVulkanGraphicsRequirements2KHR(
    XrInstance                                  instance,
    XrSystemId                                  systemId,
    XrGraphicsRequirementsVulkanKHR*            graphicsRequirements);
#endif /* XR_EXTENSION_PROTOTYPES */
#endif /* !XR_NO_PROTOTYPES */
#endif /* XR_USE_GRAPHICS_API_VULKAN */

#ifdef XR_USE_PLATFORM_EGL

// XR_MNDX_egl_enable is a preprocessor guard. Do not pass it to API calls.
#define XR_MNDX_egl_enable 1
#define XR_MNDX_egl_enable_SPEC_VERSION   2
#define XR_MNDX_EGL_ENABLE_EXTENSION_NAME "XR_MNDX_egl_enable"
typedef PFN_xrVoidFunction (*PFN_xrEglGetProcAddressMNDX)(const char *name);
// XrGraphicsBindingEGLMNDX extends XrSessionCreateInfo
typedef struct XrGraphicsBindingEGLMNDX {
    XrStructureType                type;
    const void* XR_MAY_ALIAS       next;
    PFN_xrEglGetProcAddressMNDX    getProcAddress;
    EGLDisplay                     display;
    EGLConfig                      config;
    EGLContext                     context;
} XrGraphicsBindingEGLMNDX;

#endif /* XR_USE_PLATFORM_EGL */

#ifdef XR_USE_PLATFORM_WIN32

// XR_MSFT_perception_anchor_interop is a preprocessor guard. Do not pass it to API calls.
#define XR_MSFT_perception_anchor_interop 1
#define XR_MSFT_perception_anchor_interop_SPEC_VERSION 1
#define XR_MSFT_PERCEPTION_ANCHOR_INTEROP_EXTENSION_NAME "XR_MSFT_perception_anchor_interop"
typedef XrResult (XRAPI_PTR *PFN_xrCreateSpatialAnchorFromPerceptionAnchorMSFT)(XrSession session, IUnknown* perceptionAnchor, XrSpatialAnchorMSFT* anchor);
typedef XrResult (XRAPI_PTR *PFN_xrTryGetPerceptionAnchorFromSpatialAnchorMSFT)(XrSession session, XrSpatialAnchorMSFT anchor, IUnknown** perceptionAnchor);

#ifndef XR_NO_PROTOTYPES
#ifdef XR_EXTENSION_PROTOTYPES
XRAPI_ATTR XrResult XRAPI_CALL xrCreateSpatialAnchorFromPerceptionAnchorMSFT(
    XrSession                                   session,
    IUnknown*                                   perceptionAnchor,
    XrSpatialAnchorMSFT*                        anchor);

XRAPI_ATTR XrResult XRAPI_CALL xrTryGetPerceptionAnchorFromSpatialAnchorMSFT(
    XrSession                                   session,
    XrSpatialAnchorMSFT                         anchor,
    IUnknown**                                  perceptionAnchor);
#endif /* XR_EXTENSION_PROTOTYPES */
#endif /* !XR_NO_PROTOTYPES */
#endif /* XR_USE_PLATFORM_WIN32 */

#ifdef XR_USE_PLATFORM_WIN32

// XR_MSFT_holographic_window_attachment is a preprocessor guard. Do not pass it to API calls.
#define XR_MSFT_holographic_window_attachment 1
#define XR_MSFT_holographic_window_attachment_SPEC_VERSION 1
#define XR_MSFT_HOLOGRAPHIC_WINDOW_ATTACHMENT_EXTENSION_NAME "XR_MSFT_holographic_window_attachment"
#ifdef XR_USE_PLATFORM_WIN32
// XrHolographicWindowAttachmentMSFT extends XrSessionCreateInfo
typedef struct XrHolographicWindowAttachmentMSFT {
    XrStructureType             type;
    const void* XR_MAY_ALIAS    next;
    IUnknown*                   holographicSpace;
    IUnknown*                   coreWindow;
} XrHolographicWindowAttachmentMSFT;
#endif // XR_USE_PLATFORM_WIN32

#endif /* XR_USE_PLATFORM_WIN32 */

#ifdef XR_USE_PLATFORM_ANDROID

// XR_FB_android_surface_swapchain_create is a preprocessor guard. Do not pass it to API calls.
#define XR_FB_android_surface_swapchain_create 1
#define XR_FB_android_surface_swapchain_create_SPEC_VERSION 1
#define XR_FB_ANDROID_SURFACE_SWAPCHAIN_CREATE_EXTENSION_NAME "XR_FB_android_surface_swapchain_create"
typedef XrFlags64 XrAndroidSurfaceSwapchainFlagsFB;

// Flag bits for XrAndroidSurfaceSwapchainFlagsFB
static const XrAndroidSurfaceSwapchainFlagsFB XR_ANDROID_SURFACE_SWAPCHAIN_SYNCHRONOUS_BIT_FB = 0x00000001;
static const XrAndroidSurfaceSwapchainFlagsFB XR_ANDROID_SURFACE_SWAPCHAIN_USE_TIMESTAMPS_BIT_FB = 0x00000002;

#ifdef XR_USE_PLATFORM_ANDROID
// XrAndroidSurfaceSwapchainCreateInfoFB extends XrSwapchainCreateInfo
typedef struct XrAndroidSurfaceSwapchainCreateInfoFB {
    XrStructureType                     type;
    const void* XR_MAY_ALIAS            next;
    XrAndroidSurfaceSwapchainFlagsFB    createFlags;
} XrAndroidSurfaceSwapchainCreateInfoFB;
#endif // XR_USE_PLATFORM_ANDROID

#endif /* XR_USE_PLATFORM_ANDROID */

#ifdef XR_USE_PLATFORM_ML

// XR_ML_compat is a preprocessor guard. Do not pass it to API calls.
#define XR_ML_compat 1
#define XR_ML_compat_SPEC_VERSION         1
#define XR_ML_COMPAT_EXTENSION_NAME       "XR_ML_compat"
typedef struct XrCoordinateSpaceCreateInfoML {
    XrStructureType             type;
    const void* XR_MAY_ALIAS    next;
    MLCoordinateFrameUID        cfuid;
    XrPosef                     poseInCoordinateSpace;
} XrCoordinateSpaceCreateInfoML;

typedef XrResult (XRAPI_PTR *PFN_xrCreateSpaceFromCoordinateFrameUIDML)(XrSession session, const XrCoordinateSpaceCreateInfoML *createInfo, XrSpace* space);

#ifndef XR_NO_PROTOTYPES
#ifdef XR_EXTENSION_PROTOTYPES
XRAPI_ATTR XrResult XRAPI_CALL xrCreateSpaceFromCoordinateFrameUIDML(
    XrSession                                   session,
    const XrCoordinateSpaceCreateInfoML *       createInfo,
    XrSpace*                                    space);
#endif /* XR_EXTENSION_PROTOTYPES */
#endif /* !XR_NO_PROTOTYPES */
#endif /* XR_USE_PLATFORM_ML */

#ifdef XR_USE_PLATFORM_WIN32

// XR_OCULUS_audio_device_guid is a preprocessor guard. Do not pass it to API calls.
#define XR_OCULUS_audio_device_guid 1
#define XR_OCULUS_audio_device_guid_SPEC_VERSION 1
#define XR_OCULUS_AUDIO_DEVICE_GUID_EXTENSION_NAME "XR_OCULUS_audio_device_guid"
#define XR_MAX_AUDIO_DEVICE_STR_SIZE_OCULUS 128
typedef XrResult (XRAPI_PTR *PFN_xrGetAudioOutputDeviceGuidOculus)(XrInstance instance, wchar_t buffer[XR_MAX_AUDIO_DEVICE_STR_SIZE_OCULUS]);
typedef XrResult (XRAPI_PTR *PFN_xrGetAudioInputDeviceGuidOculus)(XrInstance instance, wchar_t buffer[XR_MAX_AUDIO_DEVICE_STR_SIZE_OCULUS]);

#ifndef XR_NO_PROTOTYPES
#ifdef XR_EXTENSION_PROTOTYPES
XRAPI_ATTR XrResult XRAPI_CALL xrGetAudioOutputDeviceGuidOculus(
    XrInstance                                  instance,
    wchar_t                                     buffer[XR_MAX_AUDIO_DEVICE_STR_SIZE_OCULUS]);

XRAPI_ATTR XrResult XRAPI_CALL xrGetAudioInputDeviceGuidOculus(
    XrInstance                                  instance,
    wchar_t                                     buffer[XR_MAX_AUDIO_DEVICE_STR_SIZE_OCULUS]);
#endif /* XR_EXTENSION_PROTOTYPES */
#endif /* !XR_NO_PROTOTYPES */
#endif /* XR_USE_PLATFORM_WIN32 */

#ifdef XR_USE_GRAPHICS_API_VULKAN

// XR_FB_foveation_vulkan is a preprocessor guard. Do not pass it to API calls.
#define XR_FB_foveation_vulkan 1
#define XR_FB_foveation_vulkan_SPEC_VERSION 1
#define XR_FB_FOVEATION_VULKAN_EXTENSION_NAME "XR_FB_foveation_vulkan"
// XrSwapchainImageFoveationVulkanFB extends XrSwapchainImageVulkanKHR
typedef struct XrSwapchainImageFoveationVulkanFB {
    XrStructureType       type;
    void* XR_MAY_ALIAS    next;
    VkImage               image;
    uint32_t              width;
    uint32_t              height;
} XrSwapchainImageFoveationVulkanFB;

#endif /* XR_USE_GRAPHICS_API_VULKAN */

#ifdef XR_USE_PLATFORM_ANDROID

// XR_FB_swapchain_update_state_android_surface is a preprocessor guard. Do not pass it to API calls.
#define XR_FB_swapchain_update_state_android_surface 1
#define XR_FB_swapchain_update_state_android_surface_SPEC_VERSION 1
#define XR_FB_SWAPCHAIN_UPDATE_STATE_ANDROID_SURFACE_EXTENSION_NAME "XR_FB_swapchain_update_state_android_surface"
#ifdef XR_USE_PLATFORM_ANDROID
typedef struct XrSwapchainStateAndroidSurfaceDimensionsFB {
    XrStructureType       type;
    void* XR_MAY_ALIAS    next;
    uint32_t              width;
    uint32_t              height;
} XrSwapchainStateAndroidSurfaceDimensionsFB;
#endif // XR_USE_PLATFORM_ANDROID

#endif /* XR_USE_PLATFORM_ANDROID */

#ifdef XR_USE_GRAPHICS_API_OPENGL_ES

// XR_FB_swapchain_update_state_opengl_es is a preprocessor guard. Do not pass it to API calls.
#define XR_FB_swapchain_update_state_opengl_es 1
#define XR_FB_swapchain_update_state_opengl_es_SPEC_VERSION 1
#define XR_FB_SWAPCHAIN_UPDATE_STATE_OPENGL_ES_EXTENSION_NAME "XR_FB_swapchain_update_state_opengl_es"
#ifdef XR_USE_GRAPHICS_API_OPENGL_ES
typedef struct XrSwapchainStateSamplerOpenGLESFB {
    XrStructureType       type;
    void* XR_MAY_ALIAS    next;
    EGLenum               minFilter;
    EGLenum               magFilter;
    EGLenum               wrapModeS;
    EGLenum               wrapModeT;
    EGLenum               swizzleRed;
    EGLenum               swizzleGreen;
    EGLenum               swizzleBlue;
    EGLenum               swizzleAlpha;
    float                 maxAnisotropy;
    XrColor4f             borderColor;
} XrSwapchainStateSamplerOpenGLESFB;
#endif // XR_USE_GRAPHICS_API_OPENGL_ES

#endif /* XR_USE_GRAPHICS_API_OPENGL_ES */

#ifdef XR_USE_GRAPHICS_API_VULKAN

// XR_FB_swapchain_update_state_vulkan is a preprocessor guard. Do not pass it to API calls.
#define XR_FB_swapchain_update_state_vulkan 1
#define XR_FB_swapchain_update_state_vulkan_SPEC_VERSION 1
#define XR_FB_SWAPCHAIN_UPDATE_STATE_VULKAN_EXTENSION_NAME "XR_FB_swapchain_update_state_vulkan"
#ifdef XR_USE_GRAPHICS_API_VULKAN
typedef struct XrSwapchainStateSamplerVulkanFB {
    XrStructureType         type;
    void* XR_MAY_ALIAS      next;
    VkFilter                minFilter;
    VkFilter                magFilter;
    VkSamplerMipmapMode     mipmapMode;
    VkSamplerAddressMode    wrapModeS;
    VkSamplerAddressMode    wrapModeT;
    VkComponentSwizzle      swizzleRed;
    VkComponentSwizzle      swizzleGreen;
    VkComponentSwizzle      swizzleBlue;
    VkComponentSwizzle      swizzleAlpha;
    float                   maxAnisotropy;
    XrColor4f               borderColor;
} XrSwapchainStateSamplerVulkanFB;
#endif // XR_USE_GRAPHICS_API_VULKAN

#endif /* XR_USE_GRAPHICS_API_VULKAN */

#ifdef XR_USE_GRAPHICS_API_VULKAN

// XR_META_vulkan_swapchain_create_info is a preprocessor guard. Do not pass it to API calls.
#define XR_META_vulkan_swapchain_create_info 1
#define XR_META_vulkan_swapchain_create_info_SPEC_VERSION 1
#define XR_META_VULKAN_SWAPCHAIN_CREATE_INFO_EXTENSION_NAME "XR_META_vulkan_swapchain_create_info"
// XrVulkanSwapchainCreateInfoMETA extends XrSwapchainCreateInfo
typedef struct XrVulkanSwapchainCreateInfoMETA {
    XrStructureType             type;
    const void* XR_MAY_ALIAS    next;
    VkImageCreateFlags          additionalCreateFlags;
    VkImageUsageFlags           additionalUsageFlags;
} XrVulkanSwapchainCreateInfoMETA;

#endif /* XR_USE_GRAPHICS_API_VULKAN */

#ifdef __cplusplus
}
#endif

#endif
