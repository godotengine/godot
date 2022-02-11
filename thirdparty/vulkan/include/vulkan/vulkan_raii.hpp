// Copyright 2015-2021 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//

// This header is generated from the Khronos Vulkan XML API Registry.

#ifndef VULKAN_RAII_HPP
#define VULKAN_RAII_HPP

#include <vulkan/vulkan.hpp>

#if !defined( VULKAN_HPP_RAII_NAMESPACE )
#  define VULKAN_HPP_RAII_NAMESPACE raii
#endif

namespace VULKAN_HPP_NAMESPACE
{
  namespace VULKAN_HPP_RAII_NAMESPACE
  {
#if !defined( VULKAN_HPP_DISABLE_ENHANCED_MODE ) && !defined( VULKAN_HPP_NO_EXCEPTIONS )

    template <class T, class U = T>
    VULKAN_HPP_CONSTEXPR_14 VULKAN_HPP_INLINE T exchange( T & obj, U && newValue )
    {
#  if ( 14 <= VULKAN_HPP_CPP_VERSION )
      return std::exchange<T>( obj, std::forward<U>( newValue ) );
#  else
      T oldValue = std::move( obj );
      obj        = std::forward<U>( newValue );
      return oldValue;
#  endif
    }

    class ContextDispatcher : public DispatchLoaderBase
    {
    public:
      ContextDispatcher( PFN_vkGetInstanceProcAddr getProcAddr )
        : vkGetInstanceProcAddr( getProcAddr )
        //=== VK_VERSION_1_0 ===
        , vkCreateInstance( PFN_vkCreateInstance( getProcAddr( NULL, "vkCreateInstance" ) ) )
        , vkEnumerateInstanceExtensionProperties( PFN_vkEnumerateInstanceExtensionProperties(
            getProcAddr( NULL, "vkEnumerateInstanceExtensionProperties" ) ) )
        , vkEnumerateInstanceLayerProperties(
            PFN_vkEnumerateInstanceLayerProperties( getProcAddr( NULL, "vkEnumerateInstanceLayerProperties" ) ) )
        //=== VK_VERSION_1_1 ===
        , vkEnumerateInstanceVersion(
            PFN_vkEnumerateInstanceVersion( getProcAddr( NULL, "vkEnumerateInstanceVersion" ) ) )
      {}

    public:
      PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = 0;

      //=== VK_VERSION_1_0 ===
      PFN_vkCreateInstance                       vkCreateInstance                       = 0;
      PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties = 0;
      PFN_vkEnumerateInstanceLayerProperties     vkEnumerateInstanceLayerProperties     = 0;

      //=== VK_VERSION_1_1 ===
      PFN_vkEnumerateInstanceVersion vkEnumerateInstanceVersion = 0;
    };

    class InstanceDispatcher : public DispatchLoaderBase
    {
    public:
      InstanceDispatcher( PFN_vkGetInstanceProcAddr getProcAddr ) : vkGetInstanceProcAddr( getProcAddr ) {}

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      InstanceDispatcher() = default;
#  endif

      void init( VkInstance instance )
      {
        //=== VK_VERSION_1_0 ===
        vkDestroyInstance = PFN_vkDestroyInstance( vkGetInstanceProcAddr( instance, "vkDestroyInstance" ) );
        vkEnumeratePhysicalDevices =
          PFN_vkEnumeratePhysicalDevices( vkGetInstanceProcAddr( instance, "vkEnumeratePhysicalDevices" ) );
        vkGetPhysicalDeviceFeatures =
          PFN_vkGetPhysicalDeviceFeatures( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceFeatures" ) );
        vkGetPhysicalDeviceFormatProperties = PFN_vkGetPhysicalDeviceFormatProperties(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceFormatProperties" ) );
        vkGetPhysicalDeviceImageFormatProperties = PFN_vkGetPhysicalDeviceImageFormatProperties(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceImageFormatProperties" ) );
        vkGetPhysicalDeviceProperties =
          PFN_vkGetPhysicalDeviceProperties( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceProperties" ) );
        vkGetPhysicalDeviceQueueFamilyProperties = PFN_vkGetPhysicalDeviceQueueFamilyProperties(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceQueueFamilyProperties" ) );
        vkGetPhysicalDeviceMemoryProperties = PFN_vkGetPhysicalDeviceMemoryProperties(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceMemoryProperties" ) );
        vkGetInstanceProcAddr = PFN_vkGetInstanceProcAddr( vkGetInstanceProcAddr( instance, "vkGetInstanceProcAddr" ) );
        vkCreateDevice        = PFN_vkCreateDevice( vkGetInstanceProcAddr( instance, "vkCreateDevice" ) );
        vkEnumerateDeviceExtensionProperties = PFN_vkEnumerateDeviceExtensionProperties(
          vkGetInstanceProcAddr( instance, "vkEnumerateDeviceExtensionProperties" ) );
        vkEnumerateDeviceLayerProperties =
          PFN_vkEnumerateDeviceLayerProperties( vkGetInstanceProcAddr( instance, "vkEnumerateDeviceLayerProperties" ) );
        vkGetPhysicalDeviceSparseImageFormatProperties = PFN_vkGetPhysicalDeviceSparseImageFormatProperties(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSparseImageFormatProperties" ) );

        //=== VK_VERSION_1_1 ===
        vkEnumeratePhysicalDeviceGroups =
          PFN_vkEnumeratePhysicalDeviceGroups( vkGetInstanceProcAddr( instance, "vkEnumeratePhysicalDeviceGroups" ) );
        vkGetPhysicalDeviceFeatures2 =
          PFN_vkGetPhysicalDeviceFeatures2( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceFeatures2" ) );
        vkGetPhysicalDeviceProperties2 =
          PFN_vkGetPhysicalDeviceProperties2( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceProperties2" ) );
        vkGetPhysicalDeviceFormatProperties2 = PFN_vkGetPhysicalDeviceFormatProperties2(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceFormatProperties2" ) );
        vkGetPhysicalDeviceImageFormatProperties2 = PFN_vkGetPhysicalDeviceImageFormatProperties2(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceImageFormatProperties2" ) );
        vkGetPhysicalDeviceQueueFamilyProperties2 = PFN_vkGetPhysicalDeviceQueueFamilyProperties2(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceQueueFamilyProperties2" ) );
        vkGetPhysicalDeviceMemoryProperties2 = PFN_vkGetPhysicalDeviceMemoryProperties2(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceMemoryProperties2" ) );
        vkGetPhysicalDeviceSparseImageFormatProperties2 = PFN_vkGetPhysicalDeviceSparseImageFormatProperties2(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSparseImageFormatProperties2" ) );
        vkGetPhysicalDeviceExternalBufferProperties = PFN_vkGetPhysicalDeviceExternalBufferProperties(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceExternalBufferProperties" ) );
        vkGetPhysicalDeviceExternalFenceProperties = PFN_vkGetPhysicalDeviceExternalFenceProperties(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceExternalFenceProperties" ) );
        vkGetPhysicalDeviceExternalSemaphoreProperties = PFN_vkGetPhysicalDeviceExternalSemaphoreProperties(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceExternalSemaphoreProperties" ) );

        //=== VK_EXT_acquire_drm_display ===
        vkAcquireDrmDisplayEXT =
          PFN_vkAcquireDrmDisplayEXT( vkGetInstanceProcAddr( instance, "vkAcquireDrmDisplayEXT" ) );
        vkGetDrmDisplayEXT = PFN_vkGetDrmDisplayEXT( vkGetInstanceProcAddr( instance, "vkGetDrmDisplayEXT" ) );

#  if defined( VK_USE_PLATFORM_XLIB_XRANDR_EXT )
        //=== VK_EXT_acquire_xlib_display ===
        vkAcquireXlibDisplayEXT =
          PFN_vkAcquireXlibDisplayEXT( vkGetInstanceProcAddr( instance, "vkAcquireXlibDisplayEXT" ) );
        vkGetRandROutputDisplayEXT =
          PFN_vkGetRandROutputDisplayEXT( vkGetInstanceProcAddr( instance, "vkGetRandROutputDisplayEXT" ) );
#  endif /*VK_USE_PLATFORM_XLIB_XRANDR_EXT*/

        //=== VK_EXT_calibrated_timestamps ===
        vkGetPhysicalDeviceCalibrateableTimeDomainsEXT = PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT" ) );

        //=== VK_EXT_debug_report ===
        vkCreateDebugReportCallbackEXT =
          PFN_vkCreateDebugReportCallbackEXT( vkGetInstanceProcAddr( instance, "vkCreateDebugReportCallbackEXT" ) );
        vkDestroyDebugReportCallbackEXT =
          PFN_vkDestroyDebugReportCallbackEXT( vkGetInstanceProcAddr( instance, "vkDestroyDebugReportCallbackEXT" ) );
        vkDebugReportMessageEXT =
          PFN_vkDebugReportMessageEXT( vkGetInstanceProcAddr( instance, "vkDebugReportMessageEXT" ) );

        //=== VK_EXT_debug_utils ===
        vkCreateDebugUtilsMessengerEXT =
          PFN_vkCreateDebugUtilsMessengerEXT( vkGetInstanceProcAddr( instance, "vkCreateDebugUtilsMessengerEXT" ) );
        vkDestroyDebugUtilsMessengerEXT =
          PFN_vkDestroyDebugUtilsMessengerEXT( vkGetInstanceProcAddr( instance, "vkDestroyDebugUtilsMessengerEXT" ) );
        vkSubmitDebugUtilsMessageEXT =
          PFN_vkSubmitDebugUtilsMessageEXT( vkGetInstanceProcAddr( instance, "vkSubmitDebugUtilsMessageEXT" ) );

        //=== VK_EXT_direct_mode_display ===
        vkReleaseDisplayEXT = PFN_vkReleaseDisplayEXT( vkGetInstanceProcAddr( instance, "vkReleaseDisplayEXT" ) );

#  if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
        //=== VK_EXT_directfb_surface ===
        vkCreateDirectFBSurfaceEXT =
          PFN_vkCreateDirectFBSurfaceEXT( vkGetInstanceProcAddr( instance, "vkCreateDirectFBSurfaceEXT" ) );
        vkGetPhysicalDeviceDirectFBPresentationSupportEXT = PFN_vkGetPhysicalDeviceDirectFBPresentationSupportEXT(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceDirectFBPresentationSupportEXT" ) );
#  endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

        //=== VK_EXT_display_surface_counter ===
        vkGetPhysicalDeviceSurfaceCapabilities2EXT = PFN_vkGetPhysicalDeviceSurfaceCapabilities2EXT(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSurfaceCapabilities2EXT" ) );

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
        //=== VK_EXT_full_screen_exclusive ===
        vkGetPhysicalDeviceSurfacePresentModes2EXT = PFN_vkGetPhysicalDeviceSurfacePresentModes2EXT(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSurfacePresentModes2EXT" ) );
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

        //=== VK_EXT_headless_surface ===
        vkCreateHeadlessSurfaceEXT =
          PFN_vkCreateHeadlessSurfaceEXT( vkGetInstanceProcAddr( instance, "vkCreateHeadlessSurfaceEXT" ) );

#  if defined( VK_USE_PLATFORM_METAL_EXT )
        //=== VK_EXT_metal_surface ===
        vkCreateMetalSurfaceEXT =
          PFN_vkCreateMetalSurfaceEXT( vkGetInstanceProcAddr( instance, "vkCreateMetalSurfaceEXT" ) );
#  endif /*VK_USE_PLATFORM_METAL_EXT*/

        //=== VK_EXT_sample_locations ===
        vkGetPhysicalDeviceMultisamplePropertiesEXT = PFN_vkGetPhysicalDeviceMultisamplePropertiesEXT(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceMultisamplePropertiesEXT" ) );

        //=== VK_EXT_tooling_info ===
        vkGetPhysicalDeviceToolPropertiesEXT = PFN_vkGetPhysicalDeviceToolPropertiesEXT(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceToolPropertiesEXT" ) );

#  if defined( VK_USE_PLATFORM_FUCHSIA )
        //=== VK_FUCHSIA_imagepipe_surface ===
        vkCreateImagePipeSurfaceFUCHSIA =
          PFN_vkCreateImagePipeSurfaceFUCHSIA( vkGetInstanceProcAddr( instance, "vkCreateImagePipeSurfaceFUCHSIA" ) );
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

#  if defined( VK_USE_PLATFORM_GGP )
        //=== VK_GGP_stream_descriptor_surface ===
        vkCreateStreamDescriptorSurfaceGGP = PFN_vkCreateStreamDescriptorSurfaceGGP(
          vkGetInstanceProcAddr( instance, "vkCreateStreamDescriptorSurfaceGGP" ) );
#  endif /*VK_USE_PLATFORM_GGP*/

#  if defined( VK_USE_PLATFORM_ANDROID_KHR )
        //=== VK_KHR_android_surface ===
        vkCreateAndroidSurfaceKHR =
          PFN_vkCreateAndroidSurfaceKHR( vkGetInstanceProcAddr( instance, "vkCreateAndroidSurfaceKHR" ) );
#  endif /*VK_USE_PLATFORM_ANDROID_KHR*/

        //=== VK_KHR_device_group ===
        vkGetPhysicalDevicePresentRectanglesKHR = PFN_vkGetPhysicalDevicePresentRectanglesKHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDevicePresentRectanglesKHR" ) );

        //=== VK_KHR_device_group_creation ===
        vkEnumeratePhysicalDeviceGroupsKHR = PFN_vkEnumeratePhysicalDeviceGroupsKHR(
          vkGetInstanceProcAddr( instance, "vkEnumeratePhysicalDeviceGroupsKHR" ) );
        if ( !vkEnumeratePhysicalDeviceGroups )
          vkEnumeratePhysicalDeviceGroups = vkEnumeratePhysicalDeviceGroupsKHR;

        //=== VK_KHR_display ===
        vkGetPhysicalDeviceDisplayPropertiesKHR = PFN_vkGetPhysicalDeviceDisplayPropertiesKHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceDisplayPropertiesKHR" ) );
        vkGetPhysicalDeviceDisplayPlanePropertiesKHR = PFN_vkGetPhysicalDeviceDisplayPlanePropertiesKHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceDisplayPlanePropertiesKHR" ) );
        vkGetDisplayPlaneSupportedDisplaysKHR = PFN_vkGetDisplayPlaneSupportedDisplaysKHR(
          vkGetInstanceProcAddr( instance, "vkGetDisplayPlaneSupportedDisplaysKHR" ) );
        vkGetDisplayModePropertiesKHR =
          PFN_vkGetDisplayModePropertiesKHR( vkGetInstanceProcAddr( instance, "vkGetDisplayModePropertiesKHR" ) );
        vkCreateDisplayModeKHR =
          PFN_vkCreateDisplayModeKHR( vkGetInstanceProcAddr( instance, "vkCreateDisplayModeKHR" ) );
        vkGetDisplayPlaneCapabilitiesKHR =
          PFN_vkGetDisplayPlaneCapabilitiesKHR( vkGetInstanceProcAddr( instance, "vkGetDisplayPlaneCapabilitiesKHR" ) );
        vkCreateDisplayPlaneSurfaceKHR =
          PFN_vkCreateDisplayPlaneSurfaceKHR( vkGetInstanceProcAddr( instance, "vkCreateDisplayPlaneSurfaceKHR" ) );

        //=== VK_KHR_external_fence_capabilities ===
        vkGetPhysicalDeviceExternalFencePropertiesKHR = PFN_vkGetPhysicalDeviceExternalFencePropertiesKHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceExternalFencePropertiesKHR" ) );
        if ( !vkGetPhysicalDeviceExternalFenceProperties )
          vkGetPhysicalDeviceExternalFenceProperties = vkGetPhysicalDeviceExternalFencePropertiesKHR;

        //=== VK_KHR_external_memory_capabilities ===
        vkGetPhysicalDeviceExternalBufferPropertiesKHR = PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceExternalBufferPropertiesKHR" ) );
        if ( !vkGetPhysicalDeviceExternalBufferProperties )
          vkGetPhysicalDeviceExternalBufferProperties = vkGetPhysicalDeviceExternalBufferPropertiesKHR;

        //=== VK_KHR_external_semaphore_capabilities ===
        vkGetPhysicalDeviceExternalSemaphorePropertiesKHR = PFN_vkGetPhysicalDeviceExternalSemaphorePropertiesKHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceExternalSemaphorePropertiesKHR" ) );
        if ( !vkGetPhysicalDeviceExternalSemaphoreProperties )
          vkGetPhysicalDeviceExternalSemaphoreProperties = vkGetPhysicalDeviceExternalSemaphorePropertiesKHR;

        //=== VK_KHR_fragment_shading_rate ===
        vkGetPhysicalDeviceFragmentShadingRatesKHR = PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceFragmentShadingRatesKHR" ) );

        //=== VK_KHR_get_display_properties2 ===
        vkGetPhysicalDeviceDisplayProperties2KHR = PFN_vkGetPhysicalDeviceDisplayProperties2KHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceDisplayProperties2KHR" ) );
        vkGetPhysicalDeviceDisplayPlaneProperties2KHR = PFN_vkGetPhysicalDeviceDisplayPlaneProperties2KHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceDisplayPlaneProperties2KHR" ) );
        vkGetDisplayModeProperties2KHR =
          PFN_vkGetDisplayModeProperties2KHR( vkGetInstanceProcAddr( instance, "vkGetDisplayModeProperties2KHR" ) );
        vkGetDisplayPlaneCapabilities2KHR = PFN_vkGetDisplayPlaneCapabilities2KHR(
          vkGetInstanceProcAddr( instance, "vkGetDisplayPlaneCapabilities2KHR" ) );

        //=== VK_KHR_get_physical_device_properties2 ===
        vkGetPhysicalDeviceFeatures2KHR =
          PFN_vkGetPhysicalDeviceFeatures2KHR( vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceFeatures2KHR" ) );
        if ( !vkGetPhysicalDeviceFeatures2 )
          vkGetPhysicalDeviceFeatures2 = vkGetPhysicalDeviceFeatures2KHR;
        vkGetPhysicalDeviceProperties2KHR = PFN_vkGetPhysicalDeviceProperties2KHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceProperties2KHR" ) );
        if ( !vkGetPhysicalDeviceProperties2 )
          vkGetPhysicalDeviceProperties2 = vkGetPhysicalDeviceProperties2KHR;
        vkGetPhysicalDeviceFormatProperties2KHR = PFN_vkGetPhysicalDeviceFormatProperties2KHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceFormatProperties2KHR" ) );
        if ( !vkGetPhysicalDeviceFormatProperties2 )
          vkGetPhysicalDeviceFormatProperties2 = vkGetPhysicalDeviceFormatProperties2KHR;
        vkGetPhysicalDeviceImageFormatProperties2KHR = PFN_vkGetPhysicalDeviceImageFormatProperties2KHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceImageFormatProperties2KHR" ) );
        if ( !vkGetPhysicalDeviceImageFormatProperties2 )
          vkGetPhysicalDeviceImageFormatProperties2 = vkGetPhysicalDeviceImageFormatProperties2KHR;
        vkGetPhysicalDeviceQueueFamilyProperties2KHR = PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceQueueFamilyProperties2KHR" ) );
        if ( !vkGetPhysicalDeviceQueueFamilyProperties2 )
          vkGetPhysicalDeviceQueueFamilyProperties2 = vkGetPhysicalDeviceQueueFamilyProperties2KHR;
        vkGetPhysicalDeviceMemoryProperties2KHR = PFN_vkGetPhysicalDeviceMemoryProperties2KHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceMemoryProperties2KHR" ) );
        if ( !vkGetPhysicalDeviceMemoryProperties2 )
          vkGetPhysicalDeviceMemoryProperties2 = vkGetPhysicalDeviceMemoryProperties2KHR;
        vkGetPhysicalDeviceSparseImageFormatProperties2KHR = PFN_vkGetPhysicalDeviceSparseImageFormatProperties2KHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSparseImageFormatProperties2KHR" ) );
        if ( !vkGetPhysicalDeviceSparseImageFormatProperties2 )
          vkGetPhysicalDeviceSparseImageFormatProperties2 = vkGetPhysicalDeviceSparseImageFormatProperties2KHR;

        //=== VK_KHR_get_surface_capabilities2 ===
        vkGetPhysicalDeviceSurfaceCapabilities2KHR = PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSurfaceCapabilities2KHR" ) );
        vkGetPhysicalDeviceSurfaceFormats2KHR = PFN_vkGetPhysicalDeviceSurfaceFormats2KHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSurfaceFormats2KHR" ) );

        //=== VK_KHR_performance_query ===
        vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR =
          PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR(
            vkGetInstanceProcAddr( instance, "vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR" ) );
        vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR =
          PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR(
            vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR" ) );

        //=== VK_KHR_surface ===
        vkDestroySurfaceKHR = PFN_vkDestroySurfaceKHR( vkGetInstanceProcAddr( instance, "vkDestroySurfaceKHR" ) );
        vkGetPhysicalDeviceSurfaceSupportKHR = PFN_vkGetPhysicalDeviceSurfaceSupportKHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSurfaceSupportKHR" ) );
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR = PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR" ) );
        vkGetPhysicalDeviceSurfaceFormatsKHR = PFN_vkGetPhysicalDeviceSurfaceFormatsKHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSurfaceFormatsKHR" ) );
        vkGetPhysicalDeviceSurfacePresentModesKHR = PFN_vkGetPhysicalDeviceSurfacePresentModesKHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSurfacePresentModesKHR" ) );

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
        //=== VK_KHR_video_queue ===
        vkGetPhysicalDeviceVideoCapabilitiesKHR = PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceVideoCapabilitiesKHR" ) );
        vkGetPhysicalDeviceVideoFormatPropertiesKHR = PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceVideoFormatPropertiesKHR" ) );
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_USE_PLATFORM_WAYLAND_KHR )
        //=== VK_KHR_wayland_surface ===
        vkCreateWaylandSurfaceKHR =
          PFN_vkCreateWaylandSurfaceKHR( vkGetInstanceProcAddr( instance, "vkCreateWaylandSurfaceKHR" ) );
        vkGetPhysicalDeviceWaylandPresentationSupportKHR = PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceWaylandPresentationSupportKHR" ) );
#  endif /*VK_USE_PLATFORM_WAYLAND_KHR*/

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
        //=== VK_KHR_win32_surface ===
        vkCreateWin32SurfaceKHR =
          PFN_vkCreateWin32SurfaceKHR( vkGetInstanceProcAddr( instance, "vkCreateWin32SurfaceKHR" ) );
        vkGetPhysicalDeviceWin32PresentationSupportKHR = PFN_vkGetPhysicalDeviceWin32PresentationSupportKHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceWin32PresentationSupportKHR" ) );
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

#  if defined( VK_USE_PLATFORM_XCB_KHR )
        //=== VK_KHR_xcb_surface ===
        vkCreateXcbSurfaceKHR = PFN_vkCreateXcbSurfaceKHR( vkGetInstanceProcAddr( instance, "vkCreateXcbSurfaceKHR" ) );
        vkGetPhysicalDeviceXcbPresentationSupportKHR = PFN_vkGetPhysicalDeviceXcbPresentationSupportKHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceXcbPresentationSupportKHR" ) );
#  endif /*VK_USE_PLATFORM_XCB_KHR*/

#  if defined( VK_USE_PLATFORM_XLIB_KHR )
        //=== VK_KHR_xlib_surface ===
        vkCreateXlibSurfaceKHR =
          PFN_vkCreateXlibSurfaceKHR( vkGetInstanceProcAddr( instance, "vkCreateXlibSurfaceKHR" ) );
        vkGetPhysicalDeviceXlibPresentationSupportKHR = PFN_vkGetPhysicalDeviceXlibPresentationSupportKHR(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceXlibPresentationSupportKHR" ) );
#  endif /*VK_USE_PLATFORM_XLIB_KHR*/

#  if defined( VK_USE_PLATFORM_IOS_MVK )
        //=== VK_MVK_ios_surface ===
        vkCreateIOSSurfaceMVK = PFN_vkCreateIOSSurfaceMVK( vkGetInstanceProcAddr( instance, "vkCreateIOSSurfaceMVK" ) );
#  endif /*VK_USE_PLATFORM_IOS_MVK*/

#  if defined( VK_USE_PLATFORM_MACOS_MVK )
        //=== VK_MVK_macos_surface ===
        vkCreateMacOSSurfaceMVK =
          PFN_vkCreateMacOSSurfaceMVK( vkGetInstanceProcAddr( instance, "vkCreateMacOSSurfaceMVK" ) );
#  endif /*VK_USE_PLATFORM_MACOS_MVK*/

#  if defined( VK_USE_PLATFORM_VI_NN )
        //=== VK_NN_vi_surface ===
        vkCreateViSurfaceNN = PFN_vkCreateViSurfaceNN( vkGetInstanceProcAddr( instance, "vkCreateViSurfaceNN" ) );
#  endif /*VK_USE_PLATFORM_VI_NN*/

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
        //=== VK_NV_acquire_winrt_display ===
        vkAcquireWinrtDisplayNV =
          PFN_vkAcquireWinrtDisplayNV( vkGetInstanceProcAddr( instance, "vkAcquireWinrtDisplayNV" ) );
        vkGetWinrtDisplayNV = PFN_vkGetWinrtDisplayNV( vkGetInstanceProcAddr( instance, "vkGetWinrtDisplayNV" ) );
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

        //=== VK_NV_cooperative_matrix ===
        vkGetPhysicalDeviceCooperativeMatrixPropertiesNV = PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesNV" ) );

        //=== VK_NV_coverage_reduction_mode ===
        vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV =
          PFN_vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV(
            vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV" ) );

        //=== VK_NV_external_memory_capabilities ===
        vkGetPhysicalDeviceExternalImageFormatPropertiesNV = PFN_vkGetPhysicalDeviceExternalImageFormatPropertiesNV(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceExternalImageFormatPropertiesNV" ) );

#  if defined( VK_USE_PLATFORM_SCREEN_QNX )
        //=== VK_QNX_screen_surface ===
        vkCreateScreenSurfaceQNX =
          PFN_vkCreateScreenSurfaceQNX( vkGetInstanceProcAddr( instance, "vkCreateScreenSurfaceQNX" ) );
        vkGetPhysicalDeviceScreenPresentationSupportQNX = PFN_vkGetPhysicalDeviceScreenPresentationSupportQNX(
          vkGetInstanceProcAddr( instance, "vkGetPhysicalDeviceScreenPresentationSupportQNX" ) );
#  endif /*VK_USE_PLATFORM_SCREEN_QNX*/

        vkGetDeviceProcAddr = PFN_vkGetDeviceProcAddr( vkGetInstanceProcAddr( instance, "vkGetDeviceProcAddr" ) );
      }

    public:
      //=== VK_VERSION_1_0 ===
      PFN_vkDestroyInstance                              vkDestroyInstance                              = 0;
      PFN_vkEnumeratePhysicalDevices                     vkEnumeratePhysicalDevices                     = 0;
      PFN_vkGetPhysicalDeviceFeatures                    vkGetPhysicalDeviceFeatures                    = 0;
      PFN_vkGetPhysicalDeviceFormatProperties            vkGetPhysicalDeviceFormatProperties            = 0;
      PFN_vkGetPhysicalDeviceImageFormatProperties       vkGetPhysicalDeviceImageFormatProperties       = 0;
      PFN_vkGetPhysicalDeviceProperties                  vkGetPhysicalDeviceProperties                  = 0;
      PFN_vkGetPhysicalDeviceQueueFamilyProperties       vkGetPhysicalDeviceQueueFamilyProperties       = 0;
      PFN_vkGetPhysicalDeviceMemoryProperties            vkGetPhysicalDeviceMemoryProperties            = 0;
      PFN_vkGetInstanceProcAddr                          vkGetInstanceProcAddr                          = 0;
      PFN_vkCreateDevice                                 vkCreateDevice                                 = 0;
      PFN_vkEnumerateDeviceExtensionProperties           vkEnumerateDeviceExtensionProperties           = 0;
      PFN_vkEnumerateDeviceLayerProperties               vkEnumerateDeviceLayerProperties               = 0;
      PFN_vkGetPhysicalDeviceSparseImageFormatProperties vkGetPhysicalDeviceSparseImageFormatProperties = 0;

      //=== VK_VERSION_1_1 ===
      PFN_vkEnumeratePhysicalDeviceGroups                 vkEnumeratePhysicalDeviceGroups                 = 0;
      PFN_vkGetPhysicalDeviceFeatures2                    vkGetPhysicalDeviceFeatures2                    = 0;
      PFN_vkGetPhysicalDeviceProperties2                  vkGetPhysicalDeviceProperties2                  = 0;
      PFN_vkGetPhysicalDeviceFormatProperties2            vkGetPhysicalDeviceFormatProperties2            = 0;
      PFN_vkGetPhysicalDeviceImageFormatProperties2       vkGetPhysicalDeviceImageFormatProperties2       = 0;
      PFN_vkGetPhysicalDeviceQueueFamilyProperties2       vkGetPhysicalDeviceQueueFamilyProperties2       = 0;
      PFN_vkGetPhysicalDeviceMemoryProperties2            vkGetPhysicalDeviceMemoryProperties2            = 0;
      PFN_vkGetPhysicalDeviceSparseImageFormatProperties2 vkGetPhysicalDeviceSparseImageFormatProperties2 = 0;
      PFN_vkGetPhysicalDeviceExternalBufferProperties     vkGetPhysicalDeviceExternalBufferProperties     = 0;
      PFN_vkGetPhysicalDeviceExternalFenceProperties      vkGetPhysicalDeviceExternalFenceProperties      = 0;
      PFN_vkGetPhysicalDeviceExternalSemaphoreProperties  vkGetPhysicalDeviceExternalSemaphoreProperties  = 0;

      //=== VK_EXT_acquire_drm_display ===
      PFN_vkAcquireDrmDisplayEXT vkAcquireDrmDisplayEXT = 0;
      PFN_vkGetDrmDisplayEXT     vkGetDrmDisplayEXT     = 0;

#  if defined( VK_USE_PLATFORM_XLIB_XRANDR_EXT )
      //=== VK_EXT_acquire_xlib_display ===
      PFN_vkAcquireXlibDisplayEXT    vkAcquireXlibDisplayEXT    = 0;
      PFN_vkGetRandROutputDisplayEXT vkGetRandROutputDisplayEXT = 0;
#  else
      PFN_dummy vkAcquireXlibDisplayEXT_placeholder                           = 0;
      PFN_dummy vkGetRandROutputDisplayEXT_placeholder                        = 0;
#  endif /*VK_USE_PLATFORM_XLIB_XRANDR_EXT*/

      //=== VK_EXT_calibrated_timestamps ===
      PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT vkGetPhysicalDeviceCalibrateableTimeDomainsEXT = 0;

      //=== VK_EXT_debug_report ===
      PFN_vkCreateDebugReportCallbackEXT  vkCreateDebugReportCallbackEXT  = 0;
      PFN_vkDestroyDebugReportCallbackEXT vkDestroyDebugReportCallbackEXT = 0;
      PFN_vkDebugReportMessageEXT         vkDebugReportMessageEXT         = 0;

      //=== VK_EXT_debug_utils ===
      PFN_vkCreateDebugUtilsMessengerEXT  vkCreateDebugUtilsMessengerEXT  = 0;
      PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT = 0;
      PFN_vkSubmitDebugUtilsMessageEXT    vkSubmitDebugUtilsMessageEXT    = 0;

      //=== VK_EXT_direct_mode_display ===
      PFN_vkReleaseDisplayEXT vkReleaseDisplayEXT = 0;

#  if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
      //=== VK_EXT_directfb_surface ===
      PFN_vkCreateDirectFBSurfaceEXT                        vkCreateDirectFBSurfaceEXT                        = 0;
      PFN_vkGetPhysicalDeviceDirectFBPresentationSupportEXT vkGetPhysicalDeviceDirectFBPresentationSupportEXT = 0;
#  else
      PFN_dummy vkCreateDirectFBSurfaceEXT_placeholder                        = 0;
      PFN_dummy vkGetPhysicalDeviceDirectFBPresentationSupportEXT_placeholder = 0;
#  endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

      //=== VK_EXT_display_surface_counter ===
      PFN_vkGetPhysicalDeviceSurfaceCapabilities2EXT vkGetPhysicalDeviceSurfaceCapabilities2EXT = 0;

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_EXT_full_screen_exclusive ===
      PFN_vkGetPhysicalDeviceSurfacePresentModes2EXT vkGetPhysicalDeviceSurfacePresentModes2EXT = 0;
#  else
      PFN_dummy vkGetPhysicalDeviceSurfacePresentModes2EXT_placeholder        = 0;
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

      //=== VK_EXT_headless_surface ===
      PFN_vkCreateHeadlessSurfaceEXT vkCreateHeadlessSurfaceEXT = 0;

#  if defined( VK_USE_PLATFORM_METAL_EXT )
      //=== VK_EXT_metal_surface ===
      PFN_vkCreateMetalSurfaceEXT vkCreateMetalSurfaceEXT = 0;
#  else
      PFN_dummy vkCreateMetalSurfaceEXT_placeholder                           = 0;
#  endif /*VK_USE_PLATFORM_METAL_EXT*/

      //=== VK_EXT_sample_locations ===
      PFN_vkGetPhysicalDeviceMultisamplePropertiesEXT vkGetPhysicalDeviceMultisamplePropertiesEXT = 0;

      //=== VK_EXT_tooling_info ===
      PFN_vkGetPhysicalDeviceToolPropertiesEXT vkGetPhysicalDeviceToolPropertiesEXT = 0;

#  if defined( VK_USE_PLATFORM_FUCHSIA )
      //=== VK_FUCHSIA_imagepipe_surface ===
      PFN_vkCreateImagePipeSurfaceFUCHSIA vkCreateImagePipeSurfaceFUCHSIA = 0;
#  else
      PFN_dummy vkCreateImagePipeSurfaceFUCHSIA_placeholder                   = 0;
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

#  if defined( VK_USE_PLATFORM_GGP )
      //=== VK_GGP_stream_descriptor_surface ===
      PFN_vkCreateStreamDescriptorSurfaceGGP vkCreateStreamDescriptorSurfaceGGP = 0;
#  else
      PFN_dummy vkCreateStreamDescriptorSurfaceGGP_placeholder                = 0;
#  endif /*VK_USE_PLATFORM_GGP*/

#  if defined( VK_USE_PLATFORM_ANDROID_KHR )
      //=== VK_KHR_android_surface ===
      PFN_vkCreateAndroidSurfaceKHR vkCreateAndroidSurfaceKHR = 0;
#  else
      PFN_dummy vkCreateAndroidSurfaceKHR_placeholder                         = 0;
#  endif /*VK_USE_PLATFORM_ANDROID_KHR*/

      //=== VK_KHR_device_group ===
      PFN_vkGetPhysicalDevicePresentRectanglesKHR vkGetPhysicalDevicePresentRectanglesKHR = 0;

      //=== VK_KHR_device_group_creation ===
      PFN_vkEnumeratePhysicalDeviceGroupsKHR vkEnumeratePhysicalDeviceGroupsKHR = 0;

      //=== VK_KHR_display ===
      PFN_vkGetPhysicalDeviceDisplayPropertiesKHR      vkGetPhysicalDeviceDisplayPropertiesKHR      = 0;
      PFN_vkGetPhysicalDeviceDisplayPlanePropertiesKHR vkGetPhysicalDeviceDisplayPlanePropertiesKHR = 0;
      PFN_vkGetDisplayPlaneSupportedDisplaysKHR        vkGetDisplayPlaneSupportedDisplaysKHR        = 0;
      PFN_vkGetDisplayModePropertiesKHR                vkGetDisplayModePropertiesKHR                = 0;
      PFN_vkCreateDisplayModeKHR                       vkCreateDisplayModeKHR                       = 0;
      PFN_vkGetDisplayPlaneCapabilitiesKHR             vkGetDisplayPlaneCapabilitiesKHR             = 0;
      PFN_vkCreateDisplayPlaneSurfaceKHR               vkCreateDisplayPlaneSurfaceKHR               = 0;

      //=== VK_KHR_external_fence_capabilities ===
      PFN_vkGetPhysicalDeviceExternalFencePropertiesKHR vkGetPhysicalDeviceExternalFencePropertiesKHR = 0;

      //=== VK_KHR_external_memory_capabilities ===
      PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR vkGetPhysicalDeviceExternalBufferPropertiesKHR = 0;

      //=== VK_KHR_external_semaphore_capabilities ===
      PFN_vkGetPhysicalDeviceExternalSemaphorePropertiesKHR vkGetPhysicalDeviceExternalSemaphorePropertiesKHR = 0;

      //=== VK_KHR_fragment_shading_rate ===
      PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR vkGetPhysicalDeviceFragmentShadingRatesKHR = 0;

      //=== VK_KHR_get_display_properties2 ===
      PFN_vkGetPhysicalDeviceDisplayProperties2KHR      vkGetPhysicalDeviceDisplayProperties2KHR      = 0;
      PFN_vkGetPhysicalDeviceDisplayPlaneProperties2KHR vkGetPhysicalDeviceDisplayPlaneProperties2KHR = 0;
      PFN_vkGetDisplayModeProperties2KHR                vkGetDisplayModeProperties2KHR                = 0;
      PFN_vkGetDisplayPlaneCapabilities2KHR             vkGetDisplayPlaneCapabilities2KHR             = 0;

      //=== VK_KHR_get_physical_device_properties2 ===
      PFN_vkGetPhysicalDeviceFeatures2KHR                    vkGetPhysicalDeviceFeatures2KHR                    = 0;
      PFN_vkGetPhysicalDeviceProperties2KHR                  vkGetPhysicalDeviceProperties2KHR                  = 0;
      PFN_vkGetPhysicalDeviceFormatProperties2KHR            vkGetPhysicalDeviceFormatProperties2KHR            = 0;
      PFN_vkGetPhysicalDeviceImageFormatProperties2KHR       vkGetPhysicalDeviceImageFormatProperties2KHR       = 0;
      PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR       vkGetPhysicalDeviceQueueFamilyProperties2KHR       = 0;
      PFN_vkGetPhysicalDeviceMemoryProperties2KHR            vkGetPhysicalDeviceMemoryProperties2KHR            = 0;
      PFN_vkGetPhysicalDeviceSparseImageFormatProperties2KHR vkGetPhysicalDeviceSparseImageFormatProperties2KHR = 0;

      //=== VK_KHR_get_surface_capabilities2 ===
      PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR vkGetPhysicalDeviceSurfaceCapabilities2KHR = 0;
      PFN_vkGetPhysicalDeviceSurfaceFormats2KHR      vkGetPhysicalDeviceSurfaceFormats2KHR      = 0;

      //=== VK_KHR_performance_query ===
      PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR
        vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR = 0;
      PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR
        vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR = 0;

      //=== VK_KHR_surface ===
      PFN_vkDestroySurfaceKHR                       vkDestroySurfaceKHR                       = 0;
      PFN_vkGetPhysicalDeviceSurfaceSupportKHR      vkGetPhysicalDeviceSurfaceSupportKHR      = 0;
      PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR vkGetPhysicalDeviceSurfaceCapabilitiesKHR = 0;
      PFN_vkGetPhysicalDeviceSurfaceFormatsKHR      vkGetPhysicalDeviceSurfaceFormatsKHR      = 0;
      PFN_vkGetPhysicalDeviceSurfacePresentModesKHR vkGetPhysicalDeviceSurfacePresentModesKHR = 0;

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
      //=== VK_KHR_video_queue ===
      PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR     vkGetPhysicalDeviceVideoCapabilitiesKHR     = 0;
      PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR vkGetPhysicalDeviceVideoFormatPropertiesKHR = 0;
#  else
      PFN_dummy vkGetPhysicalDeviceVideoCapabilitiesKHR_placeholder           = 0;
      PFN_dummy vkGetPhysicalDeviceVideoFormatPropertiesKHR_placeholder       = 0;
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_USE_PLATFORM_WAYLAND_KHR )
      //=== VK_KHR_wayland_surface ===
      PFN_vkCreateWaylandSurfaceKHR                        vkCreateWaylandSurfaceKHR                        = 0;
      PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR vkGetPhysicalDeviceWaylandPresentationSupportKHR = 0;
#  else
      PFN_dummy vkCreateWaylandSurfaceKHR_placeholder                         = 0;
      PFN_dummy vkGetPhysicalDeviceWaylandPresentationSupportKHR_placeholder  = 0;
#  endif /*VK_USE_PLATFORM_WAYLAND_KHR*/

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_KHR_win32_surface ===
      PFN_vkCreateWin32SurfaceKHR                        vkCreateWin32SurfaceKHR                        = 0;
      PFN_vkGetPhysicalDeviceWin32PresentationSupportKHR vkGetPhysicalDeviceWin32PresentationSupportKHR = 0;
#  else
      PFN_dummy vkCreateWin32SurfaceKHR_placeholder                           = 0;
      PFN_dummy vkGetPhysicalDeviceWin32PresentationSupportKHR_placeholder    = 0;
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

#  if defined( VK_USE_PLATFORM_XCB_KHR )
      //=== VK_KHR_xcb_surface ===
      PFN_vkCreateXcbSurfaceKHR                        vkCreateXcbSurfaceKHR                        = 0;
      PFN_vkGetPhysicalDeviceXcbPresentationSupportKHR vkGetPhysicalDeviceXcbPresentationSupportKHR = 0;
#  else
      PFN_dummy vkCreateXcbSurfaceKHR_placeholder                             = 0;
      PFN_dummy vkGetPhysicalDeviceXcbPresentationSupportKHR_placeholder      = 0;
#  endif /*VK_USE_PLATFORM_XCB_KHR*/

#  if defined( VK_USE_PLATFORM_XLIB_KHR )
      //=== VK_KHR_xlib_surface ===
      PFN_vkCreateXlibSurfaceKHR                        vkCreateXlibSurfaceKHR                        = 0;
      PFN_vkGetPhysicalDeviceXlibPresentationSupportKHR vkGetPhysicalDeviceXlibPresentationSupportKHR = 0;
#  else
      PFN_dummy vkCreateXlibSurfaceKHR_placeholder                            = 0;
      PFN_dummy vkGetPhysicalDeviceXlibPresentationSupportKHR_placeholder     = 0;
#  endif /*VK_USE_PLATFORM_XLIB_KHR*/

#  if defined( VK_USE_PLATFORM_IOS_MVK )
      //=== VK_MVK_ios_surface ===
      PFN_vkCreateIOSSurfaceMVK vkCreateIOSSurfaceMVK = 0;
#  else
      PFN_dummy vkCreateIOSSurfaceMVK_placeholder                             = 0;
#  endif /*VK_USE_PLATFORM_IOS_MVK*/

#  if defined( VK_USE_PLATFORM_MACOS_MVK )
      //=== VK_MVK_macos_surface ===
      PFN_vkCreateMacOSSurfaceMVK vkCreateMacOSSurfaceMVK = 0;
#  else
      PFN_dummy vkCreateMacOSSurfaceMVK_placeholder                           = 0;
#  endif /*VK_USE_PLATFORM_MACOS_MVK*/

#  if defined( VK_USE_PLATFORM_VI_NN )
      //=== VK_NN_vi_surface ===
      PFN_vkCreateViSurfaceNN vkCreateViSurfaceNN = 0;
#  else
      PFN_dummy vkCreateViSurfaceNN_placeholder                               = 0;
#  endif /*VK_USE_PLATFORM_VI_NN*/

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_NV_acquire_winrt_display ===
      PFN_vkAcquireWinrtDisplayNV vkAcquireWinrtDisplayNV = 0;
      PFN_vkGetWinrtDisplayNV     vkGetWinrtDisplayNV     = 0;
#  else
      PFN_dummy vkAcquireWinrtDisplayNV_placeholder                           = 0;
      PFN_dummy vkGetWinrtDisplayNV_placeholder                               = 0;
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

      //=== VK_NV_cooperative_matrix ===
      PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV vkGetPhysicalDeviceCooperativeMatrixPropertiesNV = 0;

      //=== VK_NV_coverage_reduction_mode ===
      PFN_vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV
        vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV = 0;

      //=== VK_NV_external_memory_capabilities ===
      PFN_vkGetPhysicalDeviceExternalImageFormatPropertiesNV vkGetPhysicalDeviceExternalImageFormatPropertiesNV = 0;

#  if defined( VK_USE_PLATFORM_SCREEN_QNX )
      //=== VK_QNX_screen_surface ===
      PFN_vkCreateScreenSurfaceQNX                        vkCreateScreenSurfaceQNX                        = 0;
      PFN_vkGetPhysicalDeviceScreenPresentationSupportQNX vkGetPhysicalDeviceScreenPresentationSupportQNX = 0;
#  else
      PFN_dummy vkCreateScreenSurfaceQNX_placeholder                          = 0;
      PFN_dummy vkGetPhysicalDeviceScreenPresentationSupportQNX_placeholder   = 0;
#  endif /*VK_USE_PLATFORM_SCREEN_QNX*/

      PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr = 0;
    };

    class DeviceDispatcher : public DispatchLoaderBase
    {
    public:
      DeviceDispatcher( PFN_vkGetDeviceProcAddr getProcAddr ) : vkGetDeviceProcAddr( getProcAddr ) {}

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      DeviceDispatcher() = default;
#  endif

      void init( VkDevice device )
      {
        //=== VK_VERSION_1_0 ===
        vkGetDeviceProcAddr = PFN_vkGetDeviceProcAddr( vkGetDeviceProcAddr( device, "vkGetDeviceProcAddr" ) );
        vkDestroyDevice     = PFN_vkDestroyDevice( vkGetDeviceProcAddr( device, "vkDestroyDevice" ) );
        vkGetDeviceQueue    = PFN_vkGetDeviceQueue( vkGetDeviceProcAddr( device, "vkGetDeviceQueue" ) );
        vkQueueSubmit       = PFN_vkQueueSubmit( vkGetDeviceProcAddr( device, "vkQueueSubmit" ) );
        vkQueueWaitIdle     = PFN_vkQueueWaitIdle( vkGetDeviceProcAddr( device, "vkQueueWaitIdle" ) );
        vkDeviceWaitIdle    = PFN_vkDeviceWaitIdle( vkGetDeviceProcAddr( device, "vkDeviceWaitIdle" ) );
        vkAllocateMemory    = PFN_vkAllocateMemory( vkGetDeviceProcAddr( device, "vkAllocateMemory" ) );
        vkFreeMemory        = PFN_vkFreeMemory( vkGetDeviceProcAddr( device, "vkFreeMemory" ) );
        vkMapMemory         = PFN_vkMapMemory( vkGetDeviceProcAddr( device, "vkMapMemory" ) );
        vkUnmapMemory       = PFN_vkUnmapMemory( vkGetDeviceProcAddr( device, "vkUnmapMemory" ) );
        vkFlushMappedMemoryRanges =
          PFN_vkFlushMappedMemoryRanges( vkGetDeviceProcAddr( device, "vkFlushMappedMemoryRanges" ) );
        vkInvalidateMappedMemoryRanges =
          PFN_vkInvalidateMappedMemoryRanges( vkGetDeviceProcAddr( device, "vkInvalidateMappedMemoryRanges" ) );
        vkGetDeviceMemoryCommitment =
          PFN_vkGetDeviceMemoryCommitment( vkGetDeviceProcAddr( device, "vkGetDeviceMemoryCommitment" ) );
        vkBindBufferMemory = PFN_vkBindBufferMemory( vkGetDeviceProcAddr( device, "vkBindBufferMemory" ) );
        vkBindImageMemory  = PFN_vkBindImageMemory( vkGetDeviceProcAddr( device, "vkBindImageMemory" ) );
        vkGetBufferMemoryRequirements =
          PFN_vkGetBufferMemoryRequirements( vkGetDeviceProcAddr( device, "vkGetBufferMemoryRequirements" ) );
        vkGetImageMemoryRequirements =
          PFN_vkGetImageMemoryRequirements( vkGetDeviceProcAddr( device, "vkGetImageMemoryRequirements" ) );
        vkGetImageSparseMemoryRequirements =
          PFN_vkGetImageSparseMemoryRequirements( vkGetDeviceProcAddr( device, "vkGetImageSparseMemoryRequirements" ) );
        vkQueueBindSparse     = PFN_vkQueueBindSparse( vkGetDeviceProcAddr( device, "vkQueueBindSparse" ) );
        vkCreateFence         = PFN_vkCreateFence( vkGetDeviceProcAddr( device, "vkCreateFence" ) );
        vkDestroyFence        = PFN_vkDestroyFence( vkGetDeviceProcAddr( device, "vkDestroyFence" ) );
        vkResetFences         = PFN_vkResetFences( vkGetDeviceProcAddr( device, "vkResetFences" ) );
        vkGetFenceStatus      = PFN_vkGetFenceStatus( vkGetDeviceProcAddr( device, "vkGetFenceStatus" ) );
        vkWaitForFences       = PFN_vkWaitForFences( vkGetDeviceProcAddr( device, "vkWaitForFences" ) );
        vkCreateSemaphore     = PFN_vkCreateSemaphore( vkGetDeviceProcAddr( device, "vkCreateSemaphore" ) );
        vkDestroySemaphore    = PFN_vkDestroySemaphore( vkGetDeviceProcAddr( device, "vkDestroySemaphore" ) );
        vkCreateEvent         = PFN_vkCreateEvent( vkGetDeviceProcAddr( device, "vkCreateEvent" ) );
        vkDestroyEvent        = PFN_vkDestroyEvent( vkGetDeviceProcAddr( device, "vkDestroyEvent" ) );
        vkGetEventStatus      = PFN_vkGetEventStatus( vkGetDeviceProcAddr( device, "vkGetEventStatus" ) );
        vkSetEvent            = PFN_vkSetEvent( vkGetDeviceProcAddr( device, "vkSetEvent" ) );
        vkResetEvent          = PFN_vkResetEvent( vkGetDeviceProcAddr( device, "vkResetEvent" ) );
        vkCreateQueryPool     = PFN_vkCreateQueryPool( vkGetDeviceProcAddr( device, "vkCreateQueryPool" ) );
        vkDestroyQueryPool    = PFN_vkDestroyQueryPool( vkGetDeviceProcAddr( device, "vkDestroyQueryPool" ) );
        vkGetQueryPoolResults = PFN_vkGetQueryPoolResults( vkGetDeviceProcAddr( device, "vkGetQueryPoolResults" ) );
        vkCreateBuffer        = PFN_vkCreateBuffer( vkGetDeviceProcAddr( device, "vkCreateBuffer" ) );
        vkDestroyBuffer       = PFN_vkDestroyBuffer( vkGetDeviceProcAddr( device, "vkDestroyBuffer" ) );
        vkCreateBufferView    = PFN_vkCreateBufferView( vkGetDeviceProcAddr( device, "vkCreateBufferView" ) );
        vkDestroyBufferView   = PFN_vkDestroyBufferView( vkGetDeviceProcAddr( device, "vkDestroyBufferView" ) );
        vkCreateImage         = PFN_vkCreateImage( vkGetDeviceProcAddr( device, "vkCreateImage" ) );
        vkDestroyImage        = PFN_vkDestroyImage( vkGetDeviceProcAddr( device, "vkDestroyImage" ) );
        vkGetImageSubresourceLayout =
          PFN_vkGetImageSubresourceLayout( vkGetDeviceProcAddr( device, "vkGetImageSubresourceLayout" ) );
        vkCreateImageView      = PFN_vkCreateImageView( vkGetDeviceProcAddr( device, "vkCreateImageView" ) );
        vkDestroyImageView     = PFN_vkDestroyImageView( vkGetDeviceProcAddr( device, "vkDestroyImageView" ) );
        vkCreateShaderModule   = PFN_vkCreateShaderModule( vkGetDeviceProcAddr( device, "vkCreateShaderModule" ) );
        vkDestroyShaderModule  = PFN_vkDestroyShaderModule( vkGetDeviceProcAddr( device, "vkDestroyShaderModule" ) );
        vkCreatePipelineCache  = PFN_vkCreatePipelineCache( vkGetDeviceProcAddr( device, "vkCreatePipelineCache" ) );
        vkDestroyPipelineCache = PFN_vkDestroyPipelineCache( vkGetDeviceProcAddr( device, "vkDestroyPipelineCache" ) );
        vkGetPipelineCacheData = PFN_vkGetPipelineCacheData( vkGetDeviceProcAddr( device, "vkGetPipelineCacheData" ) );
        vkMergePipelineCaches  = PFN_vkMergePipelineCaches( vkGetDeviceProcAddr( device, "vkMergePipelineCaches" ) );
        vkCreateGraphicsPipelines =
          PFN_vkCreateGraphicsPipelines( vkGetDeviceProcAddr( device, "vkCreateGraphicsPipelines" ) );
        vkCreateComputePipelines =
          PFN_vkCreateComputePipelines( vkGetDeviceProcAddr( device, "vkCreateComputePipelines" ) );
        vkDestroyPipeline      = PFN_vkDestroyPipeline( vkGetDeviceProcAddr( device, "vkDestroyPipeline" ) );
        vkCreatePipelineLayout = PFN_vkCreatePipelineLayout( vkGetDeviceProcAddr( device, "vkCreatePipelineLayout" ) );
        vkDestroyPipelineLayout =
          PFN_vkDestroyPipelineLayout( vkGetDeviceProcAddr( device, "vkDestroyPipelineLayout" ) );
        vkCreateSampler  = PFN_vkCreateSampler( vkGetDeviceProcAddr( device, "vkCreateSampler" ) );
        vkDestroySampler = PFN_vkDestroySampler( vkGetDeviceProcAddr( device, "vkDestroySampler" ) );
        vkCreateDescriptorSetLayout =
          PFN_vkCreateDescriptorSetLayout( vkGetDeviceProcAddr( device, "vkCreateDescriptorSetLayout" ) );
        vkDestroyDescriptorSetLayout =
          PFN_vkDestroyDescriptorSetLayout( vkGetDeviceProcAddr( device, "vkDestroyDescriptorSetLayout" ) );
        vkCreateDescriptorPool = PFN_vkCreateDescriptorPool( vkGetDeviceProcAddr( device, "vkCreateDescriptorPool" ) );
        vkDestroyDescriptorPool =
          PFN_vkDestroyDescriptorPool( vkGetDeviceProcAddr( device, "vkDestroyDescriptorPool" ) );
        vkResetDescriptorPool = PFN_vkResetDescriptorPool( vkGetDeviceProcAddr( device, "vkResetDescriptorPool" ) );
        vkAllocateDescriptorSets =
          PFN_vkAllocateDescriptorSets( vkGetDeviceProcAddr( device, "vkAllocateDescriptorSets" ) );
        vkFreeDescriptorSets   = PFN_vkFreeDescriptorSets( vkGetDeviceProcAddr( device, "vkFreeDescriptorSets" ) );
        vkUpdateDescriptorSets = PFN_vkUpdateDescriptorSets( vkGetDeviceProcAddr( device, "vkUpdateDescriptorSets" ) );
        vkCreateFramebuffer    = PFN_vkCreateFramebuffer( vkGetDeviceProcAddr( device, "vkCreateFramebuffer" ) );
        vkDestroyFramebuffer   = PFN_vkDestroyFramebuffer( vkGetDeviceProcAddr( device, "vkDestroyFramebuffer" ) );
        vkCreateRenderPass     = PFN_vkCreateRenderPass( vkGetDeviceProcAddr( device, "vkCreateRenderPass" ) );
        vkDestroyRenderPass    = PFN_vkDestroyRenderPass( vkGetDeviceProcAddr( device, "vkDestroyRenderPass" ) );
        vkGetRenderAreaGranularity =
          PFN_vkGetRenderAreaGranularity( vkGetDeviceProcAddr( device, "vkGetRenderAreaGranularity" ) );
        vkCreateCommandPool  = PFN_vkCreateCommandPool( vkGetDeviceProcAddr( device, "vkCreateCommandPool" ) );
        vkDestroyCommandPool = PFN_vkDestroyCommandPool( vkGetDeviceProcAddr( device, "vkDestroyCommandPool" ) );
        vkResetCommandPool   = PFN_vkResetCommandPool( vkGetDeviceProcAddr( device, "vkResetCommandPool" ) );
        vkAllocateCommandBuffers =
          PFN_vkAllocateCommandBuffers( vkGetDeviceProcAddr( device, "vkAllocateCommandBuffers" ) );
        vkFreeCommandBuffers   = PFN_vkFreeCommandBuffers( vkGetDeviceProcAddr( device, "vkFreeCommandBuffers" ) );
        vkBeginCommandBuffer   = PFN_vkBeginCommandBuffer( vkGetDeviceProcAddr( device, "vkBeginCommandBuffer" ) );
        vkEndCommandBuffer     = PFN_vkEndCommandBuffer( vkGetDeviceProcAddr( device, "vkEndCommandBuffer" ) );
        vkResetCommandBuffer   = PFN_vkResetCommandBuffer( vkGetDeviceProcAddr( device, "vkResetCommandBuffer" ) );
        vkCmdBindPipeline      = PFN_vkCmdBindPipeline( vkGetDeviceProcAddr( device, "vkCmdBindPipeline" ) );
        vkCmdSetViewport       = PFN_vkCmdSetViewport( vkGetDeviceProcAddr( device, "vkCmdSetViewport" ) );
        vkCmdSetScissor        = PFN_vkCmdSetScissor( vkGetDeviceProcAddr( device, "vkCmdSetScissor" ) );
        vkCmdSetLineWidth      = PFN_vkCmdSetLineWidth( vkGetDeviceProcAddr( device, "vkCmdSetLineWidth" ) );
        vkCmdSetDepthBias      = PFN_vkCmdSetDepthBias( vkGetDeviceProcAddr( device, "vkCmdSetDepthBias" ) );
        vkCmdSetBlendConstants = PFN_vkCmdSetBlendConstants( vkGetDeviceProcAddr( device, "vkCmdSetBlendConstants" ) );
        vkCmdSetDepthBounds    = PFN_vkCmdSetDepthBounds( vkGetDeviceProcAddr( device, "vkCmdSetDepthBounds" ) );
        vkCmdSetStencilCompareMask =
          PFN_vkCmdSetStencilCompareMask( vkGetDeviceProcAddr( device, "vkCmdSetStencilCompareMask" ) );
        vkCmdSetStencilWriteMask =
          PFN_vkCmdSetStencilWriteMask( vkGetDeviceProcAddr( device, "vkCmdSetStencilWriteMask" ) );
        vkCmdSetStencilReference =
          PFN_vkCmdSetStencilReference( vkGetDeviceProcAddr( device, "vkCmdSetStencilReference" ) );
        vkCmdBindDescriptorSets =
          PFN_vkCmdBindDescriptorSets( vkGetDeviceProcAddr( device, "vkCmdBindDescriptorSets" ) );
        vkCmdBindIndexBuffer   = PFN_vkCmdBindIndexBuffer( vkGetDeviceProcAddr( device, "vkCmdBindIndexBuffer" ) );
        vkCmdBindVertexBuffers = PFN_vkCmdBindVertexBuffers( vkGetDeviceProcAddr( device, "vkCmdBindVertexBuffers" ) );
        vkCmdDraw              = PFN_vkCmdDraw( vkGetDeviceProcAddr( device, "vkCmdDraw" ) );
        vkCmdDrawIndexed       = PFN_vkCmdDrawIndexed( vkGetDeviceProcAddr( device, "vkCmdDrawIndexed" ) );
        vkCmdDrawIndirect      = PFN_vkCmdDrawIndirect( vkGetDeviceProcAddr( device, "vkCmdDrawIndirect" ) );
        vkCmdDrawIndexedIndirect =
          PFN_vkCmdDrawIndexedIndirect( vkGetDeviceProcAddr( device, "vkCmdDrawIndexedIndirect" ) );
        vkCmdDispatch          = PFN_vkCmdDispatch( vkGetDeviceProcAddr( device, "vkCmdDispatch" ) );
        vkCmdDispatchIndirect  = PFN_vkCmdDispatchIndirect( vkGetDeviceProcAddr( device, "vkCmdDispatchIndirect" ) );
        vkCmdCopyBuffer        = PFN_vkCmdCopyBuffer( vkGetDeviceProcAddr( device, "vkCmdCopyBuffer" ) );
        vkCmdCopyImage         = PFN_vkCmdCopyImage( vkGetDeviceProcAddr( device, "vkCmdCopyImage" ) );
        vkCmdBlitImage         = PFN_vkCmdBlitImage( vkGetDeviceProcAddr( device, "vkCmdBlitImage" ) );
        vkCmdCopyBufferToImage = PFN_vkCmdCopyBufferToImage( vkGetDeviceProcAddr( device, "vkCmdCopyBufferToImage" ) );
        vkCmdCopyImageToBuffer = PFN_vkCmdCopyImageToBuffer( vkGetDeviceProcAddr( device, "vkCmdCopyImageToBuffer" ) );
        vkCmdUpdateBuffer      = PFN_vkCmdUpdateBuffer( vkGetDeviceProcAddr( device, "vkCmdUpdateBuffer" ) );
        vkCmdFillBuffer        = PFN_vkCmdFillBuffer( vkGetDeviceProcAddr( device, "vkCmdFillBuffer" ) );
        vkCmdClearColorImage   = PFN_vkCmdClearColorImage( vkGetDeviceProcAddr( device, "vkCmdClearColorImage" ) );
        vkCmdClearDepthStencilImage =
          PFN_vkCmdClearDepthStencilImage( vkGetDeviceProcAddr( device, "vkCmdClearDepthStencilImage" ) );
        vkCmdClearAttachments = PFN_vkCmdClearAttachments( vkGetDeviceProcAddr( device, "vkCmdClearAttachments" ) );
        vkCmdResolveImage     = PFN_vkCmdResolveImage( vkGetDeviceProcAddr( device, "vkCmdResolveImage" ) );
        vkCmdSetEvent         = PFN_vkCmdSetEvent( vkGetDeviceProcAddr( device, "vkCmdSetEvent" ) );
        vkCmdResetEvent       = PFN_vkCmdResetEvent( vkGetDeviceProcAddr( device, "vkCmdResetEvent" ) );
        vkCmdWaitEvents       = PFN_vkCmdWaitEvents( vkGetDeviceProcAddr( device, "vkCmdWaitEvents" ) );
        vkCmdPipelineBarrier  = PFN_vkCmdPipelineBarrier( vkGetDeviceProcAddr( device, "vkCmdPipelineBarrier" ) );
        vkCmdBeginQuery       = PFN_vkCmdBeginQuery( vkGetDeviceProcAddr( device, "vkCmdBeginQuery" ) );
        vkCmdEndQuery         = PFN_vkCmdEndQuery( vkGetDeviceProcAddr( device, "vkCmdEndQuery" ) );
        vkCmdResetQueryPool   = PFN_vkCmdResetQueryPool( vkGetDeviceProcAddr( device, "vkCmdResetQueryPool" ) );
        vkCmdWriteTimestamp   = PFN_vkCmdWriteTimestamp( vkGetDeviceProcAddr( device, "vkCmdWriteTimestamp" ) );
        vkCmdCopyQueryPoolResults =
          PFN_vkCmdCopyQueryPoolResults( vkGetDeviceProcAddr( device, "vkCmdCopyQueryPoolResults" ) );
        vkCmdPushConstants   = PFN_vkCmdPushConstants( vkGetDeviceProcAddr( device, "vkCmdPushConstants" ) );
        vkCmdBeginRenderPass = PFN_vkCmdBeginRenderPass( vkGetDeviceProcAddr( device, "vkCmdBeginRenderPass" ) );
        vkCmdNextSubpass     = PFN_vkCmdNextSubpass( vkGetDeviceProcAddr( device, "vkCmdNextSubpass" ) );
        vkCmdEndRenderPass   = PFN_vkCmdEndRenderPass( vkGetDeviceProcAddr( device, "vkCmdEndRenderPass" ) );
        vkCmdExecuteCommands = PFN_vkCmdExecuteCommands( vkGetDeviceProcAddr( device, "vkCmdExecuteCommands" ) );

        //=== VK_VERSION_1_1 ===
        vkBindBufferMemory2 = PFN_vkBindBufferMemory2( vkGetDeviceProcAddr( device, "vkBindBufferMemory2" ) );
        vkBindImageMemory2  = PFN_vkBindImageMemory2( vkGetDeviceProcAddr( device, "vkBindImageMemory2" ) );
        vkGetDeviceGroupPeerMemoryFeatures =
          PFN_vkGetDeviceGroupPeerMemoryFeatures( vkGetDeviceProcAddr( device, "vkGetDeviceGroupPeerMemoryFeatures" ) );
        vkCmdSetDeviceMask = PFN_vkCmdSetDeviceMask( vkGetDeviceProcAddr( device, "vkCmdSetDeviceMask" ) );
        vkCmdDispatchBase  = PFN_vkCmdDispatchBase( vkGetDeviceProcAddr( device, "vkCmdDispatchBase" ) );
        vkGetImageMemoryRequirements2 =
          PFN_vkGetImageMemoryRequirements2( vkGetDeviceProcAddr( device, "vkGetImageMemoryRequirements2" ) );
        vkGetBufferMemoryRequirements2 =
          PFN_vkGetBufferMemoryRequirements2( vkGetDeviceProcAddr( device, "vkGetBufferMemoryRequirements2" ) );
        vkGetImageSparseMemoryRequirements2 = PFN_vkGetImageSparseMemoryRequirements2(
          vkGetDeviceProcAddr( device, "vkGetImageSparseMemoryRequirements2" ) );
        vkTrimCommandPool = PFN_vkTrimCommandPool( vkGetDeviceProcAddr( device, "vkTrimCommandPool" ) );
        vkGetDeviceQueue2 = PFN_vkGetDeviceQueue2( vkGetDeviceProcAddr( device, "vkGetDeviceQueue2" ) );
        vkCreateSamplerYcbcrConversion =
          PFN_vkCreateSamplerYcbcrConversion( vkGetDeviceProcAddr( device, "vkCreateSamplerYcbcrConversion" ) );
        vkDestroySamplerYcbcrConversion =
          PFN_vkDestroySamplerYcbcrConversion( vkGetDeviceProcAddr( device, "vkDestroySamplerYcbcrConversion" ) );
        vkCreateDescriptorUpdateTemplate =
          PFN_vkCreateDescriptorUpdateTemplate( vkGetDeviceProcAddr( device, "vkCreateDescriptorUpdateTemplate" ) );
        vkDestroyDescriptorUpdateTemplate =
          PFN_vkDestroyDescriptorUpdateTemplate( vkGetDeviceProcAddr( device, "vkDestroyDescriptorUpdateTemplate" ) );
        vkUpdateDescriptorSetWithTemplate =
          PFN_vkUpdateDescriptorSetWithTemplate( vkGetDeviceProcAddr( device, "vkUpdateDescriptorSetWithTemplate" ) );
        vkGetDescriptorSetLayoutSupport =
          PFN_vkGetDescriptorSetLayoutSupport( vkGetDeviceProcAddr( device, "vkGetDescriptorSetLayoutSupport" ) );

        //=== VK_VERSION_1_2 ===
        vkCmdDrawIndirectCount = PFN_vkCmdDrawIndirectCount( vkGetDeviceProcAddr( device, "vkCmdDrawIndirectCount" ) );
        vkCmdDrawIndexedIndirectCount =
          PFN_vkCmdDrawIndexedIndirectCount( vkGetDeviceProcAddr( device, "vkCmdDrawIndexedIndirectCount" ) );
        vkCreateRenderPass2   = PFN_vkCreateRenderPass2( vkGetDeviceProcAddr( device, "vkCreateRenderPass2" ) );
        vkCmdBeginRenderPass2 = PFN_vkCmdBeginRenderPass2( vkGetDeviceProcAddr( device, "vkCmdBeginRenderPass2" ) );
        vkCmdNextSubpass2     = PFN_vkCmdNextSubpass2( vkGetDeviceProcAddr( device, "vkCmdNextSubpass2" ) );
        vkCmdEndRenderPass2   = PFN_vkCmdEndRenderPass2( vkGetDeviceProcAddr( device, "vkCmdEndRenderPass2" ) );
        vkResetQueryPool      = PFN_vkResetQueryPool( vkGetDeviceProcAddr( device, "vkResetQueryPool" ) );
        vkGetSemaphoreCounterValue =
          PFN_vkGetSemaphoreCounterValue( vkGetDeviceProcAddr( device, "vkGetSemaphoreCounterValue" ) );
        vkWaitSemaphores  = PFN_vkWaitSemaphores( vkGetDeviceProcAddr( device, "vkWaitSemaphores" ) );
        vkSignalSemaphore = PFN_vkSignalSemaphore( vkGetDeviceProcAddr( device, "vkSignalSemaphore" ) );
        vkGetBufferDeviceAddress =
          PFN_vkGetBufferDeviceAddress( vkGetDeviceProcAddr( device, "vkGetBufferDeviceAddress" ) );
        vkGetBufferOpaqueCaptureAddress =
          PFN_vkGetBufferOpaqueCaptureAddress( vkGetDeviceProcAddr( device, "vkGetBufferOpaqueCaptureAddress" ) );
        vkGetDeviceMemoryOpaqueCaptureAddress = PFN_vkGetDeviceMemoryOpaqueCaptureAddress(
          vkGetDeviceProcAddr( device, "vkGetDeviceMemoryOpaqueCaptureAddress" ) );

        //=== VK_AMD_buffer_marker ===
        vkCmdWriteBufferMarkerAMD =
          PFN_vkCmdWriteBufferMarkerAMD( vkGetDeviceProcAddr( device, "vkCmdWriteBufferMarkerAMD" ) );

        //=== VK_AMD_display_native_hdr ===
        vkSetLocalDimmingAMD = PFN_vkSetLocalDimmingAMD( vkGetDeviceProcAddr( device, "vkSetLocalDimmingAMD" ) );

        //=== VK_AMD_draw_indirect_count ===
        vkCmdDrawIndirectCountAMD =
          PFN_vkCmdDrawIndirectCountAMD( vkGetDeviceProcAddr( device, "vkCmdDrawIndirectCountAMD" ) );
        if ( !vkCmdDrawIndirectCount )
          vkCmdDrawIndirectCount = vkCmdDrawIndirectCountAMD;
        vkCmdDrawIndexedIndirectCountAMD =
          PFN_vkCmdDrawIndexedIndirectCountAMD( vkGetDeviceProcAddr( device, "vkCmdDrawIndexedIndirectCountAMD" ) );
        if ( !vkCmdDrawIndexedIndirectCount )
          vkCmdDrawIndexedIndirectCount = vkCmdDrawIndexedIndirectCountAMD;

        //=== VK_AMD_shader_info ===
        vkGetShaderInfoAMD = PFN_vkGetShaderInfoAMD( vkGetDeviceProcAddr( device, "vkGetShaderInfoAMD" ) );

#  if defined( VK_USE_PLATFORM_ANDROID_KHR )
        //=== VK_ANDROID_external_memory_android_hardware_buffer ===
        vkGetAndroidHardwareBufferPropertiesANDROID = PFN_vkGetAndroidHardwareBufferPropertiesANDROID(
          vkGetDeviceProcAddr( device, "vkGetAndroidHardwareBufferPropertiesANDROID" ) );
        vkGetMemoryAndroidHardwareBufferANDROID = PFN_vkGetMemoryAndroidHardwareBufferANDROID(
          vkGetDeviceProcAddr( device, "vkGetMemoryAndroidHardwareBufferANDROID" ) );
#  endif /*VK_USE_PLATFORM_ANDROID_KHR*/

        //=== VK_EXT_buffer_device_address ===
        vkGetBufferDeviceAddressEXT =
          PFN_vkGetBufferDeviceAddressEXT( vkGetDeviceProcAddr( device, "vkGetBufferDeviceAddressEXT" ) );
        if ( !vkGetBufferDeviceAddress )
          vkGetBufferDeviceAddress = vkGetBufferDeviceAddressEXT;

        //=== VK_EXT_calibrated_timestamps ===
        vkGetCalibratedTimestampsEXT =
          PFN_vkGetCalibratedTimestampsEXT( vkGetDeviceProcAddr( device, "vkGetCalibratedTimestampsEXT" ) );

        //=== VK_EXT_color_write_enable ===
        vkCmdSetColorWriteEnableEXT =
          PFN_vkCmdSetColorWriteEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetColorWriteEnableEXT" ) );

        //=== VK_EXT_conditional_rendering ===
        vkCmdBeginConditionalRenderingEXT =
          PFN_vkCmdBeginConditionalRenderingEXT( vkGetDeviceProcAddr( device, "vkCmdBeginConditionalRenderingEXT" ) );
        vkCmdEndConditionalRenderingEXT =
          PFN_vkCmdEndConditionalRenderingEXT( vkGetDeviceProcAddr( device, "vkCmdEndConditionalRenderingEXT" ) );

        //=== VK_EXT_debug_marker ===
        vkDebugMarkerSetObjectTagEXT =
          PFN_vkDebugMarkerSetObjectTagEXT( vkGetDeviceProcAddr( device, "vkDebugMarkerSetObjectTagEXT" ) );
        vkDebugMarkerSetObjectNameEXT =
          PFN_vkDebugMarkerSetObjectNameEXT( vkGetDeviceProcAddr( device, "vkDebugMarkerSetObjectNameEXT" ) );
        vkCmdDebugMarkerBeginEXT =
          PFN_vkCmdDebugMarkerBeginEXT( vkGetDeviceProcAddr( device, "vkCmdDebugMarkerBeginEXT" ) );
        vkCmdDebugMarkerEndEXT = PFN_vkCmdDebugMarkerEndEXT( vkGetDeviceProcAddr( device, "vkCmdDebugMarkerEndEXT" ) );
        vkCmdDebugMarkerInsertEXT =
          PFN_vkCmdDebugMarkerInsertEXT( vkGetDeviceProcAddr( device, "vkCmdDebugMarkerInsertEXT" ) );

        //=== VK_EXT_debug_utils ===
        vkSetDebugUtilsObjectNameEXT =
          PFN_vkSetDebugUtilsObjectNameEXT( vkGetDeviceProcAddr( device, "vkSetDebugUtilsObjectNameEXT" ) );
        vkSetDebugUtilsObjectTagEXT =
          PFN_vkSetDebugUtilsObjectTagEXT( vkGetDeviceProcAddr( device, "vkSetDebugUtilsObjectTagEXT" ) );
        vkQueueBeginDebugUtilsLabelEXT =
          PFN_vkQueueBeginDebugUtilsLabelEXT( vkGetDeviceProcAddr( device, "vkQueueBeginDebugUtilsLabelEXT" ) );
        vkQueueEndDebugUtilsLabelEXT =
          PFN_vkQueueEndDebugUtilsLabelEXT( vkGetDeviceProcAddr( device, "vkQueueEndDebugUtilsLabelEXT" ) );
        vkQueueInsertDebugUtilsLabelEXT =
          PFN_vkQueueInsertDebugUtilsLabelEXT( vkGetDeviceProcAddr( device, "vkQueueInsertDebugUtilsLabelEXT" ) );
        vkCmdBeginDebugUtilsLabelEXT =
          PFN_vkCmdBeginDebugUtilsLabelEXT( vkGetDeviceProcAddr( device, "vkCmdBeginDebugUtilsLabelEXT" ) );
        vkCmdEndDebugUtilsLabelEXT =
          PFN_vkCmdEndDebugUtilsLabelEXT( vkGetDeviceProcAddr( device, "vkCmdEndDebugUtilsLabelEXT" ) );
        vkCmdInsertDebugUtilsLabelEXT =
          PFN_vkCmdInsertDebugUtilsLabelEXT( vkGetDeviceProcAddr( device, "vkCmdInsertDebugUtilsLabelEXT" ) );

        //=== VK_EXT_discard_rectangles ===
        vkCmdSetDiscardRectangleEXT =
          PFN_vkCmdSetDiscardRectangleEXT( vkGetDeviceProcAddr( device, "vkCmdSetDiscardRectangleEXT" ) );

        //=== VK_EXT_display_control ===
        vkDisplayPowerControlEXT =
          PFN_vkDisplayPowerControlEXT( vkGetDeviceProcAddr( device, "vkDisplayPowerControlEXT" ) );
        vkRegisterDeviceEventEXT =
          PFN_vkRegisterDeviceEventEXT( vkGetDeviceProcAddr( device, "vkRegisterDeviceEventEXT" ) );
        vkRegisterDisplayEventEXT =
          PFN_vkRegisterDisplayEventEXT( vkGetDeviceProcAddr( device, "vkRegisterDisplayEventEXT" ) );
        vkGetSwapchainCounterEXT =
          PFN_vkGetSwapchainCounterEXT( vkGetDeviceProcAddr( device, "vkGetSwapchainCounterEXT" ) );

        //=== VK_EXT_extended_dynamic_state ===
        vkCmdSetCullModeEXT  = PFN_vkCmdSetCullModeEXT( vkGetDeviceProcAddr( device, "vkCmdSetCullModeEXT" ) );
        vkCmdSetFrontFaceEXT = PFN_vkCmdSetFrontFaceEXT( vkGetDeviceProcAddr( device, "vkCmdSetFrontFaceEXT" ) );
        vkCmdSetPrimitiveTopologyEXT =
          PFN_vkCmdSetPrimitiveTopologyEXT( vkGetDeviceProcAddr( device, "vkCmdSetPrimitiveTopologyEXT" ) );
        vkCmdSetViewportWithCountEXT =
          PFN_vkCmdSetViewportWithCountEXT( vkGetDeviceProcAddr( device, "vkCmdSetViewportWithCountEXT" ) );
        vkCmdSetScissorWithCountEXT =
          PFN_vkCmdSetScissorWithCountEXT( vkGetDeviceProcAddr( device, "vkCmdSetScissorWithCountEXT" ) );
        vkCmdBindVertexBuffers2EXT =
          PFN_vkCmdBindVertexBuffers2EXT( vkGetDeviceProcAddr( device, "vkCmdBindVertexBuffers2EXT" ) );
        vkCmdSetDepthTestEnableEXT =
          PFN_vkCmdSetDepthTestEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetDepthTestEnableEXT" ) );
        vkCmdSetDepthWriteEnableEXT =
          PFN_vkCmdSetDepthWriteEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetDepthWriteEnableEXT" ) );
        vkCmdSetDepthCompareOpEXT =
          PFN_vkCmdSetDepthCompareOpEXT( vkGetDeviceProcAddr( device, "vkCmdSetDepthCompareOpEXT" ) );
        vkCmdSetDepthBoundsTestEnableEXT =
          PFN_vkCmdSetDepthBoundsTestEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetDepthBoundsTestEnableEXT" ) );
        vkCmdSetStencilTestEnableEXT =
          PFN_vkCmdSetStencilTestEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetStencilTestEnableEXT" ) );
        vkCmdSetStencilOpEXT = PFN_vkCmdSetStencilOpEXT( vkGetDeviceProcAddr( device, "vkCmdSetStencilOpEXT" ) );

        //=== VK_EXT_extended_dynamic_state2 ===
        vkCmdSetPatchControlPointsEXT =
          PFN_vkCmdSetPatchControlPointsEXT( vkGetDeviceProcAddr( device, "vkCmdSetPatchControlPointsEXT" ) );
        vkCmdSetRasterizerDiscardEnableEXT =
          PFN_vkCmdSetRasterizerDiscardEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetRasterizerDiscardEnableEXT" ) );
        vkCmdSetDepthBiasEnableEXT =
          PFN_vkCmdSetDepthBiasEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetDepthBiasEnableEXT" ) );
        vkCmdSetLogicOpEXT = PFN_vkCmdSetLogicOpEXT( vkGetDeviceProcAddr( device, "vkCmdSetLogicOpEXT" ) );
        vkCmdSetPrimitiveRestartEnableEXT =
          PFN_vkCmdSetPrimitiveRestartEnableEXT( vkGetDeviceProcAddr( device, "vkCmdSetPrimitiveRestartEnableEXT" ) );

        //=== VK_EXT_external_memory_host ===
        vkGetMemoryHostPointerPropertiesEXT = PFN_vkGetMemoryHostPointerPropertiesEXT(
          vkGetDeviceProcAddr( device, "vkGetMemoryHostPointerPropertiesEXT" ) );

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
        //=== VK_EXT_full_screen_exclusive ===
        vkAcquireFullScreenExclusiveModeEXT = PFN_vkAcquireFullScreenExclusiveModeEXT(
          vkGetDeviceProcAddr( device, "vkAcquireFullScreenExclusiveModeEXT" ) );
        vkReleaseFullScreenExclusiveModeEXT = PFN_vkReleaseFullScreenExclusiveModeEXT(
          vkGetDeviceProcAddr( device, "vkReleaseFullScreenExclusiveModeEXT" ) );
        vkGetDeviceGroupSurfacePresentModes2EXT = PFN_vkGetDeviceGroupSurfacePresentModes2EXT(
          vkGetDeviceProcAddr( device, "vkGetDeviceGroupSurfacePresentModes2EXT" ) );
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

        //=== VK_EXT_hdr_metadata ===
        vkSetHdrMetadataEXT = PFN_vkSetHdrMetadataEXT( vkGetDeviceProcAddr( device, "vkSetHdrMetadataEXT" ) );

        //=== VK_EXT_host_query_reset ===
        vkResetQueryPoolEXT = PFN_vkResetQueryPoolEXT( vkGetDeviceProcAddr( device, "vkResetQueryPoolEXT" ) );
        if ( !vkResetQueryPool )
          vkResetQueryPool = vkResetQueryPoolEXT;

        //=== VK_EXT_image_drm_format_modifier ===
        vkGetImageDrmFormatModifierPropertiesEXT = PFN_vkGetImageDrmFormatModifierPropertiesEXT(
          vkGetDeviceProcAddr( device, "vkGetImageDrmFormatModifierPropertiesEXT" ) );

        //=== VK_EXT_line_rasterization ===
        vkCmdSetLineStippleEXT = PFN_vkCmdSetLineStippleEXT( vkGetDeviceProcAddr( device, "vkCmdSetLineStippleEXT" ) );

        //=== VK_EXT_multi_draw ===
        vkCmdDrawMultiEXT = PFN_vkCmdDrawMultiEXT( vkGetDeviceProcAddr( device, "vkCmdDrawMultiEXT" ) );
        vkCmdDrawMultiIndexedEXT =
          PFN_vkCmdDrawMultiIndexedEXT( vkGetDeviceProcAddr( device, "vkCmdDrawMultiIndexedEXT" ) );

        //=== VK_EXT_private_data ===
        vkCreatePrivateDataSlotEXT =
          PFN_vkCreatePrivateDataSlotEXT( vkGetDeviceProcAddr( device, "vkCreatePrivateDataSlotEXT" ) );
        vkDestroyPrivateDataSlotEXT =
          PFN_vkDestroyPrivateDataSlotEXT( vkGetDeviceProcAddr( device, "vkDestroyPrivateDataSlotEXT" ) );
        vkSetPrivateDataEXT = PFN_vkSetPrivateDataEXT( vkGetDeviceProcAddr( device, "vkSetPrivateDataEXT" ) );
        vkGetPrivateDataEXT = PFN_vkGetPrivateDataEXT( vkGetDeviceProcAddr( device, "vkGetPrivateDataEXT" ) );

        //=== VK_EXT_sample_locations ===
        vkCmdSetSampleLocationsEXT =
          PFN_vkCmdSetSampleLocationsEXT( vkGetDeviceProcAddr( device, "vkCmdSetSampleLocationsEXT" ) );

        //=== VK_EXT_transform_feedback ===
        vkCmdBindTransformFeedbackBuffersEXT = PFN_vkCmdBindTransformFeedbackBuffersEXT(
          vkGetDeviceProcAddr( device, "vkCmdBindTransformFeedbackBuffersEXT" ) );
        vkCmdBeginTransformFeedbackEXT =
          PFN_vkCmdBeginTransformFeedbackEXT( vkGetDeviceProcAddr( device, "vkCmdBeginTransformFeedbackEXT" ) );
        vkCmdEndTransformFeedbackEXT =
          PFN_vkCmdEndTransformFeedbackEXT( vkGetDeviceProcAddr( device, "vkCmdEndTransformFeedbackEXT" ) );
        vkCmdBeginQueryIndexedEXT =
          PFN_vkCmdBeginQueryIndexedEXT( vkGetDeviceProcAddr( device, "vkCmdBeginQueryIndexedEXT" ) );
        vkCmdEndQueryIndexedEXT =
          PFN_vkCmdEndQueryIndexedEXT( vkGetDeviceProcAddr( device, "vkCmdEndQueryIndexedEXT" ) );
        vkCmdDrawIndirectByteCountEXT =
          PFN_vkCmdDrawIndirectByteCountEXT( vkGetDeviceProcAddr( device, "vkCmdDrawIndirectByteCountEXT" ) );

        //=== VK_EXT_validation_cache ===
        vkCreateValidationCacheEXT =
          PFN_vkCreateValidationCacheEXT( vkGetDeviceProcAddr( device, "vkCreateValidationCacheEXT" ) );
        vkDestroyValidationCacheEXT =
          PFN_vkDestroyValidationCacheEXT( vkGetDeviceProcAddr( device, "vkDestroyValidationCacheEXT" ) );
        vkMergeValidationCachesEXT =
          PFN_vkMergeValidationCachesEXT( vkGetDeviceProcAddr( device, "vkMergeValidationCachesEXT" ) );
        vkGetValidationCacheDataEXT =
          PFN_vkGetValidationCacheDataEXT( vkGetDeviceProcAddr( device, "vkGetValidationCacheDataEXT" ) );

        //=== VK_EXT_vertex_input_dynamic_state ===
        vkCmdSetVertexInputEXT = PFN_vkCmdSetVertexInputEXT( vkGetDeviceProcAddr( device, "vkCmdSetVertexInputEXT" ) );

#  if defined( VK_USE_PLATFORM_FUCHSIA )
        //=== VK_FUCHSIA_external_memory ===
        vkGetMemoryZirconHandleFUCHSIA =
          PFN_vkGetMemoryZirconHandleFUCHSIA( vkGetDeviceProcAddr( device, "vkGetMemoryZirconHandleFUCHSIA" ) );
        vkGetMemoryZirconHandlePropertiesFUCHSIA = PFN_vkGetMemoryZirconHandlePropertiesFUCHSIA(
          vkGetDeviceProcAddr( device, "vkGetMemoryZirconHandlePropertiesFUCHSIA" ) );
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
        //=== VK_FUCHSIA_external_semaphore ===
        vkImportSemaphoreZirconHandleFUCHSIA = PFN_vkImportSemaphoreZirconHandleFUCHSIA(
          vkGetDeviceProcAddr( device, "vkImportSemaphoreZirconHandleFUCHSIA" ) );
        vkGetSemaphoreZirconHandleFUCHSIA =
          PFN_vkGetSemaphoreZirconHandleFUCHSIA( vkGetDeviceProcAddr( device, "vkGetSemaphoreZirconHandleFUCHSIA" ) );
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

        //=== VK_GOOGLE_display_timing ===
        vkGetRefreshCycleDurationGOOGLE =
          PFN_vkGetRefreshCycleDurationGOOGLE( vkGetDeviceProcAddr( device, "vkGetRefreshCycleDurationGOOGLE" ) );
        vkGetPastPresentationTimingGOOGLE =
          PFN_vkGetPastPresentationTimingGOOGLE( vkGetDeviceProcAddr( device, "vkGetPastPresentationTimingGOOGLE" ) );

        //=== VK_HUAWEI_invocation_mask ===
        vkCmdBindInvocationMaskHUAWEI =
          PFN_vkCmdBindInvocationMaskHUAWEI( vkGetDeviceProcAddr( device, "vkCmdBindInvocationMaskHUAWEI" ) );

        //=== VK_HUAWEI_subpass_shading ===
        vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI = PFN_vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI(
          vkGetDeviceProcAddr( device, "vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI" ) );
        vkCmdSubpassShadingHUAWEI =
          PFN_vkCmdSubpassShadingHUAWEI( vkGetDeviceProcAddr( device, "vkCmdSubpassShadingHUAWEI" ) );

        //=== VK_INTEL_performance_query ===
        vkInitializePerformanceApiINTEL =
          PFN_vkInitializePerformanceApiINTEL( vkGetDeviceProcAddr( device, "vkInitializePerformanceApiINTEL" ) );
        vkUninitializePerformanceApiINTEL =
          PFN_vkUninitializePerformanceApiINTEL( vkGetDeviceProcAddr( device, "vkUninitializePerformanceApiINTEL" ) );
        vkCmdSetPerformanceMarkerINTEL =
          PFN_vkCmdSetPerformanceMarkerINTEL( vkGetDeviceProcAddr( device, "vkCmdSetPerformanceMarkerINTEL" ) );
        vkCmdSetPerformanceStreamMarkerINTEL = PFN_vkCmdSetPerformanceStreamMarkerINTEL(
          vkGetDeviceProcAddr( device, "vkCmdSetPerformanceStreamMarkerINTEL" ) );
        vkCmdSetPerformanceOverrideINTEL =
          PFN_vkCmdSetPerformanceOverrideINTEL( vkGetDeviceProcAddr( device, "vkCmdSetPerformanceOverrideINTEL" ) );
        vkAcquirePerformanceConfigurationINTEL = PFN_vkAcquirePerformanceConfigurationINTEL(
          vkGetDeviceProcAddr( device, "vkAcquirePerformanceConfigurationINTEL" ) );
        vkReleasePerformanceConfigurationINTEL = PFN_vkReleasePerformanceConfigurationINTEL(
          vkGetDeviceProcAddr( device, "vkReleasePerformanceConfigurationINTEL" ) );
        vkQueueSetPerformanceConfigurationINTEL = PFN_vkQueueSetPerformanceConfigurationINTEL(
          vkGetDeviceProcAddr( device, "vkQueueSetPerformanceConfigurationINTEL" ) );
        vkGetPerformanceParameterINTEL =
          PFN_vkGetPerformanceParameterINTEL( vkGetDeviceProcAddr( device, "vkGetPerformanceParameterINTEL" ) );

        //=== VK_KHR_acceleration_structure ===
        vkCreateAccelerationStructureKHR =
          PFN_vkCreateAccelerationStructureKHR( vkGetDeviceProcAddr( device, "vkCreateAccelerationStructureKHR" ) );
        vkDestroyAccelerationStructureKHR =
          PFN_vkDestroyAccelerationStructureKHR( vkGetDeviceProcAddr( device, "vkDestroyAccelerationStructureKHR" ) );
        vkCmdBuildAccelerationStructuresKHR = PFN_vkCmdBuildAccelerationStructuresKHR(
          vkGetDeviceProcAddr( device, "vkCmdBuildAccelerationStructuresKHR" ) );
        vkCmdBuildAccelerationStructuresIndirectKHR = PFN_vkCmdBuildAccelerationStructuresIndirectKHR(
          vkGetDeviceProcAddr( device, "vkCmdBuildAccelerationStructuresIndirectKHR" ) );
        vkBuildAccelerationStructuresKHR =
          PFN_vkBuildAccelerationStructuresKHR( vkGetDeviceProcAddr( device, "vkBuildAccelerationStructuresKHR" ) );
        vkCopyAccelerationStructureKHR =
          PFN_vkCopyAccelerationStructureKHR( vkGetDeviceProcAddr( device, "vkCopyAccelerationStructureKHR" ) );
        vkCopyAccelerationStructureToMemoryKHR = PFN_vkCopyAccelerationStructureToMemoryKHR(
          vkGetDeviceProcAddr( device, "vkCopyAccelerationStructureToMemoryKHR" ) );
        vkCopyMemoryToAccelerationStructureKHR = PFN_vkCopyMemoryToAccelerationStructureKHR(
          vkGetDeviceProcAddr( device, "vkCopyMemoryToAccelerationStructureKHR" ) );
        vkWriteAccelerationStructuresPropertiesKHR = PFN_vkWriteAccelerationStructuresPropertiesKHR(
          vkGetDeviceProcAddr( device, "vkWriteAccelerationStructuresPropertiesKHR" ) );
        vkCmdCopyAccelerationStructureKHR =
          PFN_vkCmdCopyAccelerationStructureKHR( vkGetDeviceProcAddr( device, "vkCmdCopyAccelerationStructureKHR" ) );
        vkCmdCopyAccelerationStructureToMemoryKHR = PFN_vkCmdCopyAccelerationStructureToMemoryKHR(
          vkGetDeviceProcAddr( device, "vkCmdCopyAccelerationStructureToMemoryKHR" ) );
        vkCmdCopyMemoryToAccelerationStructureKHR = PFN_vkCmdCopyMemoryToAccelerationStructureKHR(
          vkGetDeviceProcAddr( device, "vkCmdCopyMemoryToAccelerationStructureKHR" ) );
        vkGetAccelerationStructureDeviceAddressKHR = PFN_vkGetAccelerationStructureDeviceAddressKHR(
          vkGetDeviceProcAddr( device, "vkGetAccelerationStructureDeviceAddressKHR" ) );
        vkCmdWriteAccelerationStructuresPropertiesKHR = PFN_vkCmdWriteAccelerationStructuresPropertiesKHR(
          vkGetDeviceProcAddr( device, "vkCmdWriteAccelerationStructuresPropertiesKHR" ) );
        vkGetDeviceAccelerationStructureCompatibilityKHR = PFN_vkGetDeviceAccelerationStructureCompatibilityKHR(
          vkGetDeviceProcAddr( device, "vkGetDeviceAccelerationStructureCompatibilityKHR" ) );
        vkGetAccelerationStructureBuildSizesKHR = PFN_vkGetAccelerationStructureBuildSizesKHR(
          vkGetDeviceProcAddr( device, "vkGetAccelerationStructureBuildSizesKHR" ) );

        //=== VK_KHR_bind_memory2 ===
        vkBindBufferMemory2KHR = PFN_vkBindBufferMemory2KHR( vkGetDeviceProcAddr( device, "vkBindBufferMemory2KHR" ) );
        if ( !vkBindBufferMemory2 )
          vkBindBufferMemory2 = vkBindBufferMemory2KHR;
        vkBindImageMemory2KHR = PFN_vkBindImageMemory2KHR( vkGetDeviceProcAddr( device, "vkBindImageMemory2KHR" ) );
        if ( !vkBindImageMemory2 )
          vkBindImageMemory2 = vkBindImageMemory2KHR;

        //=== VK_KHR_buffer_device_address ===
        vkGetBufferDeviceAddressKHR =
          PFN_vkGetBufferDeviceAddressKHR( vkGetDeviceProcAddr( device, "vkGetBufferDeviceAddressKHR" ) );
        if ( !vkGetBufferDeviceAddress )
          vkGetBufferDeviceAddress = vkGetBufferDeviceAddressKHR;
        vkGetBufferOpaqueCaptureAddressKHR =
          PFN_vkGetBufferOpaqueCaptureAddressKHR( vkGetDeviceProcAddr( device, "vkGetBufferOpaqueCaptureAddressKHR" ) );
        if ( !vkGetBufferOpaqueCaptureAddress )
          vkGetBufferOpaqueCaptureAddress = vkGetBufferOpaqueCaptureAddressKHR;
        vkGetDeviceMemoryOpaqueCaptureAddressKHR = PFN_vkGetDeviceMemoryOpaqueCaptureAddressKHR(
          vkGetDeviceProcAddr( device, "vkGetDeviceMemoryOpaqueCaptureAddressKHR" ) );
        if ( !vkGetDeviceMemoryOpaqueCaptureAddress )
          vkGetDeviceMemoryOpaqueCaptureAddress = vkGetDeviceMemoryOpaqueCaptureAddressKHR;

        //=== VK_KHR_copy_commands2 ===
        vkCmdCopyBuffer2KHR = PFN_vkCmdCopyBuffer2KHR( vkGetDeviceProcAddr( device, "vkCmdCopyBuffer2KHR" ) );
        vkCmdCopyImage2KHR  = PFN_vkCmdCopyImage2KHR( vkGetDeviceProcAddr( device, "vkCmdCopyImage2KHR" ) );
        vkCmdCopyBufferToImage2KHR =
          PFN_vkCmdCopyBufferToImage2KHR( vkGetDeviceProcAddr( device, "vkCmdCopyBufferToImage2KHR" ) );
        vkCmdCopyImageToBuffer2KHR =
          PFN_vkCmdCopyImageToBuffer2KHR( vkGetDeviceProcAddr( device, "vkCmdCopyImageToBuffer2KHR" ) );
        vkCmdBlitImage2KHR    = PFN_vkCmdBlitImage2KHR( vkGetDeviceProcAddr( device, "vkCmdBlitImage2KHR" ) );
        vkCmdResolveImage2KHR = PFN_vkCmdResolveImage2KHR( vkGetDeviceProcAddr( device, "vkCmdResolveImage2KHR" ) );

        //=== VK_KHR_create_renderpass2 ===
        vkCreateRenderPass2KHR = PFN_vkCreateRenderPass2KHR( vkGetDeviceProcAddr( device, "vkCreateRenderPass2KHR" ) );
        if ( !vkCreateRenderPass2 )
          vkCreateRenderPass2 = vkCreateRenderPass2KHR;
        vkCmdBeginRenderPass2KHR =
          PFN_vkCmdBeginRenderPass2KHR( vkGetDeviceProcAddr( device, "vkCmdBeginRenderPass2KHR" ) );
        if ( !vkCmdBeginRenderPass2 )
          vkCmdBeginRenderPass2 = vkCmdBeginRenderPass2KHR;
        vkCmdNextSubpass2KHR = PFN_vkCmdNextSubpass2KHR( vkGetDeviceProcAddr( device, "vkCmdNextSubpass2KHR" ) );
        if ( !vkCmdNextSubpass2 )
          vkCmdNextSubpass2 = vkCmdNextSubpass2KHR;
        vkCmdEndRenderPass2KHR = PFN_vkCmdEndRenderPass2KHR( vkGetDeviceProcAddr( device, "vkCmdEndRenderPass2KHR" ) );
        if ( !vkCmdEndRenderPass2 )
          vkCmdEndRenderPass2 = vkCmdEndRenderPass2KHR;

        //=== VK_KHR_deferred_host_operations ===
        vkCreateDeferredOperationKHR =
          PFN_vkCreateDeferredOperationKHR( vkGetDeviceProcAddr( device, "vkCreateDeferredOperationKHR" ) );
        vkDestroyDeferredOperationKHR =
          PFN_vkDestroyDeferredOperationKHR( vkGetDeviceProcAddr( device, "vkDestroyDeferredOperationKHR" ) );
        vkGetDeferredOperationMaxConcurrencyKHR = PFN_vkGetDeferredOperationMaxConcurrencyKHR(
          vkGetDeviceProcAddr( device, "vkGetDeferredOperationMaxConcurrencyKHR" ) );
        vkGetDeferredOperationResultKHR =
          PFN_vkGetDeferredOperationResultKHR( vkGetDeviceProcAddr( device, "vkGetDeferredOperationResultKHR" ) );
        vkDeferredOperationJoinKHR =
          PFN_vkDeferredOperationJoinKHR( vkGetDeviceProcAddr( device, "vkDeferredOperationJoinKHR" ) );

        //=== VK_KHR_descriptor_update_template ===
        vkCreateDescriptorUpdateTemplateKHR = PFN_vkCreateDescriptorUpdateTemplateKHR(
          vkGetDeviceProcAddr( device, "vkCreateDescriptorUpdateTemplateKHR" ) );
        if ( !vkCreateDescriptorUpdateTemplate )
          vkCreateDescriptorUpdateTemplate = vkCreateDescriptorUpdateTemplateKHR;
        vkDestroyDescriptorUpdateTemplateKHR = PFN_vkDestroyDescriptorUpdateTemplateKHR(
          vkGetDeviceProcAddr( device, "vkDestroyDescriptorUpdateTemplateKHR" ) );
        if ( !vkDestroyDescriptorUpdateTemplate )
          vkDestroyDescriptorUpdateTemplate = vkDestroyDescriptorUpdateTemplateKHR;
        vkUpdateDescriptorSetWithTemplateKHR = PFN_vkUpdateDescriptorSetWithTemplateKHR(
          vkGetDeviceProcAddr( device, "vkUpdateDescriptorSetWithTemplateKHR" ) );
        if ( !vkUpdateDescriptorSetWithTemplate )
          vkUpdateDescriptorSetWithTemplate = vkUpdateDescriptorSetWithTemplateKHR;
        vkCmdPushDescriptorSetWithTemplateKHR = PFN_vkCmdPushDescriptorSetWithTemplateKHR(
          vkGetDeviceProcAddr( device, "vkCmdPushDescriptorSetWithTemplateKHR" ) );

        //=== VK_KHR_device_group ===
        vkGetDeviceGroupPeerMemoryFeaturesKHR = PFN_vkGetDeviceGroupPeerMemoryFeaturesKHR(
          vkGetDeviceProcAddr( device, "vkGetDeviceGroupPeerMemoryFeaturesKHR" ) );
        if ( !vkGetDeviceGroupPeerMemoryFeatures )
          vkGetDeviceGroupPeerMemoryFeatures = vkGetDeviceGroupPeerMemoryFeaturesKHR;
        vkCmdSetDeviceMaskKHR = PFN_vkCmdSetDeviceMaskKHR( vkGetDeviceProcAddr( device, "vkCmdSetDeviceMaskKHR" ) );
        if ( !vkCmdSetDeviceMask )
          vkCmdSetDeviceMask = vkCmdSetDeviceMaskKHR;
        vkCmdDispatchBaseKHR = PFN_vkCmdDispatchBaseKHR( vkGetDeviceProcAddr( device, "vkCmdDispatchBaseKHR" ) );
        if ( !vkCmdDispatchBase )
          vkCmdDispatchBase = vkCmdDispatchBaseKHR;
        vkGetDeviceGroupPresentCapabilitiesKHR = PFN_vkGetDeviceGroupPresentCapabilitiesKHR(
          vkGetDeviceProcAddr( device, "vkGetDeviceGroupPresentCapabilitiesKHR" ) );
        vkGetDeviceGroupSurfacePresentModesKHR = PFN_vkGetDeviceGroupSurfacePresentModesKHR(
          vkGetDeviceProcAddr( device, "vkGetDeviceGroupSurfacePresentModesKHR" ) );
        vkAcquireNextImage2KHR = PFN_vkAcquireNextImage2KHR( vkGetDeviceProcAddr( device, "vkAcquireNextImage2KHR" ) );

        //=== VK_KHR_display_swapchain ===
        vkCreateSharedSwapchainsKHR =
          PFN_vkCreateSharedSwapchainsKHR( vkGetDeviceProcAddr( device, "vkCreateSharedSwapchainsKHR" ) );

        //=== VK_KHR_draw_indirect_count ===
        vkCmdDrawIndirectCountKHR =
          PFN_vkCmdDrawIndirectCountKHR( vkGetDeviceProcAddr( device, "vkCmdDrawIndirectCountKHR" ) );
        if ( !vkCmdDrawIndirectCount )
          vkCmdDrawIndirectCount = vkCmdDrawIndirectCountKHR;
        vkCmdDrawIndexedIndirectCountKHR =
          PFN_vkCmdDrawIndexedIndirectCountKHR( vkGetDeviceProcAddr( device, "vkCmdDrawIndexedIndirectCountKHR" ) );
        if ( !vkCmdDrawIndexedIndirectCount )
          vkCmdDrawIndexedIndirectCount = vkCmdDrawIndexedIndirectCountKHR;

        //=== VK_KHR_external_fence_fd ===
        vkImportFenceFdKHR = PFN_vkImportFenceFdKHR( vkGetDeviceProcAddr( device, "vkImportFenceFdKHR" ) );
        vkGetFenceFdKHR    = PFN_vkGetFenceFdKHR( vkGetDeviceProcAddr( device, "vkGetFenceFdKHR" ) );

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
        //=== VK_KHR_external_fence_win32 ===
        vkImportFenceWin32HandleKHR =
          PFN_vkImportFenceWin32HandleKHR( vkGetDeviceProcAddr( device, "vkImportFenceWin32HandleKHR" ) );
        vkGetFenceWin32HandleKHR =
          PFN_vkGetFenceWin32HandleKHR( vkGetDeviceProcAddr( device, "vkGetFenceWin32HandleKHR" ) );
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

        //=== VK_KHR_external_memory_fd ===
        vkGetMemoryFdKHR = PFN_vkGetMemoryFdKHR( vkGetDeviceProcAddr( device, "vkGetMemoryFdKHR" ) );
        vkGetMemoryFdPropertiesKHR =
          PFN_vkGetMemoryFdPropertiesKHR( vkGetDeviceProcAddr( device, "vkGetMemoryFdPropertiesKHR" ) );

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
        //=== VK_KHR_external_memory_win32 ===
        vkGetMemoryWin32HandleKHR =
          PFN_vkGetMemoryWin32HandleKHR( vkGetDeviceProcAddr( device, "vkGetMemoryWin32HandleKHR" ) );
        vkGetMemoryWin32HandlePropertiesKHR = PFN_vkGetMemoryWin32HandlePropertiesKHR(
          vkGetDeviceProcAddr( device, "vkGetMemoryWin32HandlePropertiesKHR" ) );
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

        //=== VK_KHR_external_semaphore_fd ===
        vkImportSemaphoreFdKHR = PFN_vkImportSemaphoreFdKHR( vkGetDeviceProcAddr( device, "vkImportSemaphoreFdKHR" ) );
        vkGetSemaphoreFdKHR    = PFN_vkGetSemaphoreFdKHR( vkGetDeviceProcAddr( device, "vkGetSemaphoreFdKHR" ) );

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
        //=== VK_KHR_external_semaphore_win32 ===
        vkImportSemaphoreWin32HandleKHR =
          PFN_vkImportSemaphoreWin32HandleKHR( vkGetDeviceProcAddr( device, "vkImportSemaphoreWin32HandleKHR" ) );
        vkGetSemaphoreWin32HandleKHR =
          PFN_vkGetSemaphoreWin32HandleKHR( vkGetDeviceProcAddr( device, "vkGetSemaphoreWin32HandleKHR" ) );
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

        //=== VK_KHR_fragment_shading_rate ===
        vkCmdSetFragmentShadingRateKHR =
          PFN_vkCmdSetFragmentShadingRateKHR( vkGetDeviceProcAddr( device, "vkCmdSetFragmentShadingRateKHR" ) );

        //=== VK_KHR_get_memory_requirements2 ===
        vkGetImageMemoryRequirements2KHR =
          PFN_vkGetImageMemoryRequirements2KHR( vkGetDeviceProcAddr( device, "vkGetImageMemoryRequirements2KHR" ) );
        if ( !vkGetImageMemoryRequirements2 )
          vkGetImageMemoryRequirements2 = vkGetImageMemoryRequirements2KHR;
        vkGetBufferMemoryRequirements2KHR =
          PFN_vkGetBufferMemoryRequirements2KHR( vkGetDeviceProcAddr( device, "vkGetBufferMemoryRequirements2KHR" ) );
        if ( !vkGetBufferMemoryRequirements2 )
          vkGetBufferMemoryRequirements2 = vkGetBufferMemoryRequirements2KHR;
        vkGetImageSparseMemoryRequirements2KHR = PFN_vkGetImageSparseMemoryRequirements2KHR(
          vkGetDeviceProcAddr( device, "vkGetImageSparseMemoryRequirements2KHR" ) );
        if ( !vkGetImageSparseMemoryRequirements2 )
          vkGetImageSparseMemoryRequirements2 = vkGetImageSparseMemoryRequirements2KHR;

        //=== VK_KHR_maintenance1 ===
        vkTrimCommandPoolKHR = PFN_vkTrimCommandPoolKHR( vkGetDeviceProcAddr( device, "vkTrimCommandPoolKHR" ) );
        if ( !vkTrimCommandPool )
          vkTrimCommandPool = vkTrimCommandPoolKHR;

        //=== VK_KHR_maintenance3 ===
        vkGetDescriptorSetLayoutSupportKHR =
          PFN_vkGetDescriptorSetLayoutSupportKHR( vkGetDeviceProcAddr( device, "vkGetDescriptorSetLayoutSupportKHR" ) );
        if ( !vkGetDescriptorSetLayoutSupport )
          vkGetDescriptorSetLayoutSupport = vkGetDescriptorSetLayoutSupportKHR;

        //=== VK_KHR_performance_query ===
        vkAcquireProfilingLockKHR =
          PFN_vkAcquireProfilingLockKHR( vkGetDeviceProcAddr( device, "vkAcquireProfilingLockKHR" ) );
        vkReleaseProfilingLockKHR =
          PFN_vkReleaseProfilingLockKHR( vkGetDeviceProcAddr( device, "vkReleaseProfilingLockKHR" ) );

        //=== VK_KHR_pipeline_executable_properties ===
        vkGetPipelineExecutablePropertiesKHR = PFN_vkGetPipelineExecutablePropertiesKHR(
          vkGetDeviceProcAddr( device, "vkGetPipelineExecutablePropertiesKHR" ) );
        vkGetPipelineExecutableStatisticsKHR = PFN_vkGetPipelineExecutableStatisticsKHR(
          vkGetDeviceProcAddr( device, "vkGetPipelineExecutableStatisticsKHR" ) );
        vkGetPipelineExecutableInternalRepresentationsKHR = PFN_vkGetPipelineExecutableInternalRepresentationsKHR(
          vkGetDeviceProcAddr( device, "vkGetPipelineExecutableInternalRepresentationsKHR" ) );

        //=== VK_KHR_present_wait ===
        vkWaitForPresentKHR = PFN_vkWaitForPresentKHR( vkGetDeviceProcAddr( device, "vkWaitForPresentKHR" ) );

        //=== VK_KHR_push_descriptor ===
        vkCmdPushDescriptorSetKHR =
          PFN_vkCmdPushDescriptorSetKHR( vkGetDeviceProcAddr( device, "vkCmdPushDescriptorSetKHR" ) );

        //=== VK_KHR_ray_tracing_pipeline ===
        vkCmdTraceRaysKHR = PFN_vkCmdTraceRaysKHR( vkGetDeviceProcAddr( device, "vkCmdTraceRaysKHR" ) );
        vkCreateRayTracingPipelinesKHR =
          PFN_vkCreateRayTracingPipelinesKHR( vkGetDeviceProcAddr( device, "vkCreateRayTracingPipelinesKHR" ) );
        vkGetRayTracingShaderGroupHandlesKHR = PFN_vkGetRayTracingShaderGroupHandlesKHR(
          vkGetDeviceProcAddr( device, "vkGetRayTracingShaderGroupHandlesKHR" ) );
        vkGetRayTracingCaptureReplayShaderGroupHandlesKHR = PFN_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR(
          vkGetDeviceProcAddr( device, "vkGetRayTracingCaptureReplayShaderGroupHandlesKHR" ) );
        vkCmdTraceRaysIndirectKHR =
          PFN_vkCmdTraceRaysIndirectKHR( vkGetDeviceProcAddr( device, "vkCmdTraceRaysIndirectKHR" ) );
        vkGetRayTracingShaderGroupStackSizeKHR = PFN_vkGetRayTracingShaderGroupStackSizeKHR(
          vkGetDeviceProcAddr( device, "vkGetRayTracingShaderGroupStackSizeKHR" ) );
        vkCmdSetRayTracingPipelineStackSizeKHR = PFN_vkCmdSetRayTracingPipelineStackSizeKHR(
          vkGetDeviceProcAddr( device, "vkCmdSetRayTracingPipelineStackSizeKHR" ) );

        //=== VK_KHR_sampler_ycbcr_conversion ===
        vkCreateSamplerYcbcrConversionKHR =
          PFN_vkCreateSamplerYcbcrConversionKHR( vkGetDeviceProcAddr( device, "vkCreateSamplerYcbcrConversionKHR" ) );
        if ( !vkCreateSamplerYcbcrConversion )
          vkCreateSamplerYcbcrConversion = vkCreateSamplerYcbcrConversionKHR;
        vkDestroySamplerYcbcrConversionKHR =
          PFN_vkDestroySamplerYcbcrConversionKHR( vkGetDeviceProcAddr( device, "vkDestroySamplerYcbcrConversionKHR" ) );
        if ( !vkDestroySamplerYcbcrConversion )
          vkDestroySamplerYcbcrConversion = vkDestroySamplerYcbcrConversionKHR;

        //=== VK_KHR_shared_presentable_image ===
        vkGetSwapchainStatusKHR =
          PFN_vkGetSwapchainStatusKHR( vkGetDeviceProcAddr( device, "vkGetSwapchainStatusKHR" ) );

        //=== VK_KHR_swapchain ===
        vkCreateSwapchainKHR  = PFN_vkCreateSwapchainKHR( vkGetDeviceProcAddr( device, "vkCreateSwapchainKHR" ) );
        vkDestroySwapchainKHR = PFN_vkDestroySwapchainKHR( vkGetDeviceProcAddr( device, "vkDestroySwapchainKHR" ) );
        vkGetSwapchainImagesKHR =
          PFN_vkGetSwapchainImagesKHR( vkGetDeviceProcAddr( device, "vkGetSwapchainImagesKHR" ) );
        vkAcquireNextImageKHR = PFN_vkAcquireNextImageKHR( vkGetDeviceProcAddr( device, "vkAcquireNextImageKHR" ) );
        vkQueuePresentKHR     = PFN_vkQueuePresentKHR( vkGetDeviceProcAddr( device, "vkQueuePresentKHR" ) );

        //=== VK_KHR_synchronization2 ===
        vkCmdSetEvent2KHR   = PFN_vkCmdSetEvent2KHR( vkGetDeviceProcAddr( device, "vkCmdSetEvent2KHR" ) );
        vkCmdResetEvent2KHR = PFN_vkCmdResetEvent2KHR( vkGetDeviceProcAddr( device, "vkCmdResetEvent2KHR" ) );
        vkCmdWaitEvents2KHR = PFN_vkCmdWaitEvents2KHR( vkGetDeviceProcAddr( device, "vkCmdWaitEvents2KHR" ) );
        vkCmdPipelineBarrier2KHR =
          PFN_vkCmdPipelineBarrier2KHR( vkGetDeviceProcAddr( device, "vkCmdPipelineBarrier2KHR" ) );
        vkCmdWriteTimestamp2KHR =
          PFN_vkCmdWriteTimestamp2KHR( vkGetDeviceProcAddr( device, "vkCmdWriteTimestamp2KHR" ) );
        vkQueueSubmit2KHR = PFN_vkQueueSubmit2KHR( vkGetDeviceProcAddr( device, "vkQueueSubmit2KHR" ) );
        vkCmdWriteBufferMarker2AMD =
          PFN_vkCmdWriteBufferMarker2AMD( vkGetDeviceProcAddr( device, "vkCmdWriteBufferMarker2AMD" ) );
        vkGetQueueCheckpointData2NV =
          PFN_vkGetQueueCheckpointData2NV( vkGetDeviceProcAddr( device, "vkGetQueueCheckpointData2NV" ) );

        //=== VK_KHR_timeline_semaphore ===
        vkGetSemaphoreCounterValueKHR =
          PFN_vkGetSemaphoreCounterValueKHR( vkGetDeviceProcAddr( device, "vkGetSemaphoreCounterValueKHR" ) );
        if ( !vkGetSemaphoreCounterValue )
          vkGetSemaphoreCounterValue = vkGetSemaphoreCounterValueKHR;
        vkWaitSemaphoresKHR = PFN_vkWaitSemaphoresKHR( vkGetDeviceProcAddr( device, "vkWaitSemaphoresKHR" ) );
        if ( !vkWaitSemaphores )
          vkWaitSemaphores = vkWaitSemaphoresKHR;
        vkSignalSemaphoreKHR = PFN_vkSignalSemaphoreKHR( vkGetDeviceProcAddr( device, "vkSignalSemaphoreKHR" ) );
        if ( !vkSignalSemaphore )
          vkSignalSemaphore = vkSignalSemaphoreKHR;

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
        //=== VK_KHR_video_decode_queue ===
        vkCmdDecodeVideoKHR = PFN_vkCmdDecodeVideoKHR( vkGetDeviceProcAddr( device, "vkCmdDecodeVideoKHR" ) );
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
        //=== VK_KHR_video_encode_queue ===
        vkCmdEncodeVideoKHR = PFN_vkCmdEncodeVideoKHR( vkGetDeviceProcAddr( device, "vkCmdEncodeVideoKHR" ) );
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
        //=== VK_KHR_video_queue ===
        vkCreateVideoSessionKHR =
          PFN_vkCreateVideoSessionKHR( vkGetDeviceProcAddr( device, "vkCreateVideoSessionKHR" ) );
        vkDestroyVideoSessionKHR =
          PFN_vkDestroyVideoSessionKHR( vkGetDeviceProcAddr( device, "vkDestroyVideoSessionKHR" ) );
        vkGetVideoSessionMemoryRequirementsKHR = PFN_vkGetVideoSessionMemoryRequirementsKHR(
          vkGetDeviceProcAddr( device, "vkGetVideoSessionMemoryRequirementsKHR" ) );
        vkBindVideoSessionMemoryKHR =
          PFN_vkBindVideoSessionMemoryKHR( vkGetDeviceProcAddr( device, "vkBindVideoSessionMemoryKHR" ) );
        vkCreateVideoSessionParametersKHR =
          PFN_vkCreateVideoSessionParametersKHR( vkGetDeviceProcAddr( device, "vkCreateVideoSessionParametersKHR" ) );
        vkUpdateVideoSessionParametersKHR =
          PFN_vkUpdateVideoSessionParametersKHR( vkGetDeviceProcAddr( device, "vkUpdateVideoSessionParametersKHR" ) );
        vkDestroyVideoSessionParametersKHR =
          PFN_vkDestroyVideoSessionParametersKHR( vkGetDeviceProcAddr( device, "vkDestroyVideoSessionParametersKHR" ) );
        vkCmdBeginVideoCodingKHR =
          PFN_vkCmdBeginVideoCodingKHR( vkGetDeviceProcAddr( device, "vkCmdBeginVideoCodingKHR" ) );
        vkCmdEndVideoCodingKHR = PFN_vkCmdEndVideoCodingKHR( vkGetDeviceProcAddr( device, "vkCmdEndVideoCodingKHR" ) );
        vkCmdControlVideoCodingKHR =
          PFN_vkCmdControlVideoCodingKHR( vkGetDeviceProcAddr( device, "vkCmdControlVideoCodingKHR" ) );
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

        //=== VK_NVX_binary_import ===
        vkCreateCuModuleNVX    = PFN_vkCreateCuModuleNVX( vkGetDeviceProcAddr( device, "vkCreateCuModuleNVX" ) );
        vkCreateCuFunctionNVX  = PFN_vkCreateCuFunctionNVX( vkGetDeviceProcAddr( device, "vkCreateCuFunctionNVX" ) );
        vkDestroyCuModuleNVX   = PFN_vkDestroyCuModuleNVX( vkGetDeviceProcAddr( device, "vkDestroyCuModuleNVX" ) );
        vkDestroyCuFunctionNVX = PFN_vkDestroyCuFunctionNVX( vkGetDeviceProcAddr( device, "vkDestroyCuFunctionNVX" ) );
        vkCmdCuLaunchKernelNVX = PFN_vkCmdCuLaunchKernelNVX( vkGetDeviceProcAddr( device, "vkCmdCuLaunchKernelNVX" ) );

        //=== VK_NVX_image_view_handle ===
        vkGetImageViewHandleNVX =
          PFN_vkGetImageViewHandleNVX( vkGetDeviceProcAddr( device, "vkGetImageViewHandleNVX" ) );
        vkGetImageViewAddressNVX =
          PFN_vkGetImageViewAddressNVX( vkGetDeviceProcAddr( device, "vkGetImageViewAddressNVX" ) );

        //=== VK_NV_clip_space_w_scaling ===
        vkCmdSetViewportWScalingNV =
          PFN_vkCmdSetViewportWScalingNV( vkGetDeviceProcAddr( device, "vkCmdSetViewportWScalingNV" ) );

        //=== VK_NV_device_diagnostic_checkpoints ===
        vkCmdSetCheckpointNV = PFN_vkCmdSetCheckpointNV( vkGetDeviceProcAddr( device, "vkCmdSetCheckpointNV" ) );
        vkGetQueueCheckpointDataNV =
          PFN_vkGetQueueCheckpointDataNV( vkGetDeviceProcAddr( device, "vkGetQueueCheckpointDataNV" ) );

        //=== VK_NV_device_generated_commands ===
        vkGetGeneratedCommandsMemoryRequirementsNV = PFN_vkGetGeneratedCommandsMemoryRequirementsNV(
          vkGetDeviceProcAddr( device, "vkGetGeneratedCommandsMemoryRequirementsNV" ) );
        vkCmdPreprocessGeneratedCommandsNV =
          PFN_vkCmdPreprocessGeneratedCommandsNV( vkGetDeviceProcAddr( device, "vkCmdPreprocessGeneratedCommandsNV" ) );
        vkCmdExecuteGeneratedCommandsNV =
          PFN_vkCmdExecuteGeneratedCommandsNV( vkGetDeviceProcAddr( device, "vkCmdExecuteGeneratedCommandsNV" ) );
        vkCmdBindPipelineShaderGroupNV =
          PFN_vkCmdBindPipelineShaderGroupNV( vkGetDeviceProcAddr( device, "vkCmdBindPipelineShaderGroupNV" ) );
        vkCreateIndirectCommandsLayoutNV =
          PFN_vkCreateIndirectCommandsLayoutNV( vkGetDeviceProcAddr( device, "vkCreateIndirectCommandsLayoutNV" ) );
        vkDestroyIndirectCommandsLayoutNV =
          PFN_vkDestroyIndirectCommandsLayoutNV( vkGetDeviceProcAddr( device, "vkDestroyIndirectCommandsLayoutNV" ) );

        //=== VK_NV_external_memory_rdma ===
        vkGetMemoryRemoteAddressNV =
          PFN_vkGetMemoryRemoteAddressNV( vkGetDeviceProcAddr( device, "vkGetMemoryRemoteAddressNV" ) );

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
        //=== VK_NV_external_memory_win32 ===
        vkGetMemoryWin32HandleNV =
          PFN_vkGetMemoryWin32HandleNV( vkGetDeviceProcAddr( device, "vkGetMemoryWin32HandleNV" ) );
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

        //=== VK_NV_fragment_shading_rate_enums ===
        vkCmdSetFragmentShadingRateEnumNV =
          PFN_vkCmdSetFragmentShadingRateEnumNV( vkGetDeviceProcAddr( device, "vkCmdSetFragmentShadingRateEnumNV" ) );

        //=== VK_NV_mesh_shader ===
        vkCmdDrawMeshTasksNV = PFN_vkCmdDrawMeshTasksNV( vkGetDeviceProcAddr( device, "vkCmdDrawMeshTasksNV" ) );
        vkCmdDrawMeshTasksIndirectNV =
          PFN_vkCmdDrawMeshTasksIndirectNV( vkGetDeviceProcAddr( device, "vkCmdDrawMeshTasksIndirectNV" ) );
        vkCmdDrawMeshTasksIndirectCountNV =
          PFN_vkCmdDrawMeshTasksIndirectCountNV( vkGetDeviceProcAddr( device, "vkCmdDrawMeshTasksIndirectCountNV" ) );

        //=== VK_NV_ray_tracing ===
        vkCreateAccelerationStructureNV =
          PFN_vkCreateAccelerationStructureNV( vkGetDeviceProcAddr( device, "vkCreateAccelerationStructureNV" ) );
        vkDestroyAccelerationStructureNV =
          PFN_vkDestroyAccelerationStructureNV( vkGetDeviceProcAddr( device, "vkDestroyAccelerationStructureNV" ) );
        vkGetAccelerationStructureMemoryRequirementsNV = PFN_vkGetAccelerationStructureMemoryRequirementsNV(
          vkGetDeviceProcAddr( device, "vkGetAccelerationStructureMemoryRequirementsNV" ) );
        vkBindAccelerationStructureMemoryNV = PFN_vkBindAccelerationStructureMemoryNV(
          vkGetDeviceProcAddr( device, "vkBindAccelerationStructureMemoryNV" ) );
        vkCmdBuildAccelerationStructureNV =
          PFN_vkCmdBuildAccelerationStructureNV( vkGetDeviceProcAddr( device, "vkCmdBuildAccelerationStructureNV" ) );
        vkCmdCopyAccelerationStructureNV =
          PFN_vkCmdCopyAccelerationStructureNV( vkGetDeviceProcAddr( device, "vkCmdCopyAccelerationStructureNV" ) );
        vkCmdTraceRaysNV = PFN_vkCmdTraceRaysNV( vkGetDeviceProcAddr( device, "vkCmdTraceRaysNV" ) );
        vkCreateRayTracingPipelinesNV =
          PFN_vkCreateRayTracingPipelinesNV( vkGetDeviceProcAddr( device, "vkCreateRayTracingPipelinesNV" ) );
        vkGetRayTracingShaderGroupHandlesNV = PFN_vkGetRayTracingShaderGroupHandlesNV(
          vkGetDeviceProcAddr( device, "vkGetRayTracingShaderGroupHandlesNV" ) );
        if ( !vkGetRayTracingShaderGroupHandlesKHR )
          vkGetRayTracingShaderGroupHandlesKHR = vkGetRayTracingShaderGroupHandlesNV;
        vkGetAccelerationStructureHandleNV =
          PFN_vkGetAccelerationStructureHandleNV( vkGetDeviceProcAddr( device, "vkGetAccelerationStructureHandleNV" ) );
        vkCmdWriteAccelerationStructuresPropertiesNV = PFN_vkCmdWriteAccelerationStructuresPropertiesNV(
          vkGetDeviceProcAddr( device, "vkCmdWriteAccelerationStructuresPropertiesNV" ) );
        vkCompileDeferredNV = PFN_vkCompileDeferredNV( vkGetDeviceProcAddr( device, "vkCompileDeferredNV" ) );

        //=== VK_NV_scissor_exclusive ===
        vkCmdSetExclusiveScissorNV =
          PFN_vkCmdSetExclusiveScissorNV( vkGetDeviceProcAddr( device, "vkCmdSetExclusiveScissorNV" ) );

        //=== VK_NV_shading_rate_image ===
        vkCmdBindShadingRateImageNV =
          PFN_vkCmdBindShadingRateImageNV( vkGetDeviceProcAddr( device, "vkCmdBindShadingRateImageNV" ) );
        vkCmdSetViewportShadingRatePaletteNV = PFN_vkCmdSetViewportShadingRatePaletteNV(
          vkGetDeviceProcAddr( device, "vkCmdSetViewportShadingRatePaletteNV" ) );
        vkCmdSetCoarseSampleOrderNV =
          PFN_vkCmdSetCoarseSampleOrderNV( vkGetDeviceProcAddr( device, "vkCmdSetCoarseSampleOrderNV" ) );
      }

    public:
      //=== VK_VERSION_1_0 ===
      PFN_vkGetDeviceProcAddr                vkGetDeviceProcAddr                = 0;
      PFN_vkDestroyDevice                    vkDestroyDevice                    = 0;
      PFN_vkGetDeviceQueue                   vkGetDeviceQueue                   = 0;
      PFN_vkQueueSubmit                      vkQueueSubmit                      = 0;
      PFN_vkQueueWaitIdle                    vkQueueWaitIdle                    = 0;
      PFN_vkDeviceWaitIdle                   vkDeviceWaitIdle                   = 0;
      PFN_vkAllocateMemory                   vkAllocateMemory                   = 0;
      PFN_vkFreeMemory                       vkFreeMemory                       = 0;
      PFN_vkMapMemory                        vkMapMemory                        = 0;
      PFN_vkUnmapMemory                      vkUnmapMemory                      = 0;
      PFN_vkFlushMappedMemoryRanges          vkFlushMappedMemoryRanges          = 0;
      PFN_vkInvalidateMappedMemoryRanges     vkInvalidateMappedMemoryRanges     = 0;
      PFN_vkGetDeviceMemoryCommitment        vkGetDeviceMemoryCommitment        = 0;
      PFN_vkBindBufferMemory                 vkBindBufferMemory                 = 0;
      PFN_vkBindImageMemory                  vkBindImageMemory                  = 0;
      PFN_vkGetBufferMemoryRequirements      vkGetBufferMemoryRequirements      = 0;
      PFN_vkGetImageMemoryRequirements       vkGetImageMemoryRequirements       = 0;
      PFN_vkGetImageSparseMemoryRequirements vkGetImageSparseMemoryRequirements = 0;
      PFN_vkQueueBindSparse                  vkQueueBindSparse                  = 0;
      PFN_vkCreateFence                      vkCreateFence                      = 0;
      PFN_vkDestroyFence                     vkDestroyFence                     = 0;
      PFN_vkResetFences                      vkResetFences                      = 0;
      PFN_vkGetFenceStatus                   vkGetFenceStatus                   = 0;
      PFN_vkWaitForFences                    vkWaitForFences                    = 0;
      PFN_vkCreateSemaphore                  vkCreateSemaphore                  = 0;
      PFN_vkDestroySemaphore                 vkDestroySemaphore                 = 0;
      PFN_vkCreateEvent                      vkCreateEvent                      = 0;
      PFN_vkDestroyEvent                     vkDestroyEvent                     = 0;
      PFN_vkGetEventStatus                   vkGetEventStatus                   = 0;
      PFN_vkSetEvent                         vkSetEvent                         = 0;
      PFN_vkResetEvent                       vkResetEvent                       = 0;
      PFN_vkCreateQueryPool                  vkCreateQueryPool                  = 0;
      PFN_vkDestroyQueryPool                 vkDestroyQueryPool                 = 0;
      PFN_vkGetQueryPoolResults              vkGetQueryPoolResults              = 0;
      PFN_vkCreateBuffer                     vkCreateBuffer                     = 0;
      PFN_vkDestroyBuffer                    vkDestroyBuffer                    = 0;
      PFN_vkCreateBufferView                 vkCreateBufferView                 = 0;
      PFN_vkDestroyBufferView                vkDestroyBufferView                = 0;
      PFN_vkCreateImage                      vkCreateImage                      = 0;
      PFN_vkDestroyImage                     vkDestroyImage                     = 0;
      PFN_vkGetImageSubresourceLayout        vkGetImageSubresourceLayout        = 0;
      PFN_vkCreateImageView                  vkCreateImageView                  = 0;
      PFN_vkDestroyImageView                 vkDestroyImageView                 = 0;
      PFN_vkCreateShaderModule               vkCreateShaderModule               = 0;
      PFN_vkDestroyShaderModule              vkDestroyShaderModule              = 0;
      PFN_vkCreatePipelineCache              vkCreatePipelineCache              = 0;
      PFN_vkDestroyPipelineCache             vkDestroyPipelineCache             = 0;
      PFN_vkGetPipelineCacheData             vkGetPipelineCacheData             = 0;
      PFN_vkMergePipelineCaches              vkMergePipelineCaches              = 0;
      PFN_vkCreateGraphicsPipelines          vkCreateGraphicsPipelines          = 0;
      PFN_vkCreateComputePipelines           vkCreateComputePipelines           = 0;
      PFN_vkDestroyPipeline                  vkDestroyPipeline                  = 0;
      PFN_vkCreatePipelineLayout             vkCreatePipelineLayout             = 0;
      PFN_vkDestroyPipelineLayout            vkDestroyPipelineLayout            = 0;
      PFN_vkCreateSampler                    vkCreateSampler                    = 0;
      PFN_vkDestroySampler                   vkDestroySampler                   = 0;
      PFN_vkCreateDescriptorSetLayout        vkCreateDescriptorSetLayout        = 0;
      PFN_vkDestroyDescriptorSetLayout       vkDestroyDescriptorSetLayout       = 0;
      PFN_vkCreateDescriptorPool             vkCreateDescriptorPool             = 0;
      PFN_vkDestroyDescriptorPool            vkDestroyDescriptorPool            = 0;
      PFN_vkResetDescriptorPool              vkResetDescriptorPool              = 0;
      PFN_vkAllocateDescriptorSets           vkAllocateDescriptorSets           = 0;
      PFN_vkFreeDescriptorSets               vkFreeDescriptorSets               = 0;
      PFN_vkUpdateDescriptorSets             vkUpdateDescriptorSets             = 0;
      PFN_vkCreateFramebuffer                vkCreateFramebuffer                = 0;
      PFN_vkDestroyFramebuffer               vkDestroyFramebuffer               = 0;
      PFN_vkCreateRenderPass                 vkCreateRenderPass                 = 0;
      PFN_vkDestroyRenderPass                vkDestroyRenderPass                = 0;
      PFN_vkGetRenderAreaGranularity         vkGetRenderAreaGranularity         = 0;
      PFN_vkCreateCommandPool                vkCreateCommandPool                = 0;
      PFN_vkDestroyCommandPool               vkDestroyCommandPool               = 0;
      PFN_vkResetCommandPool                 vkResetCommandPool                 = 0;
      PFN_vkAllocateCommandBuffers           vkAllocateCommandBuffers           = 0;
      PFN_vkFreeCommandBuffers               vkFreeCommandBuffers               = 0;
      PFN_vkBeginCommandBuffer               vkBeginCommandBuffer               = 0;
      PFN_vkEndCommandBuffer                 vkEndCommandBuffer                 = 0;
      PFN_vkResetCommandBuffer               vkResetCommandBuffer               = 0;
      PFN_vkCmdBindPipeline                  vkCmdBindPipeline                  = 0;
      PFN_vkCmdSetViewport                   vkCmdSetViewport                   = 0;
      PFN_vkCmdSetScissor                    vkCmdSetScissor                    = 0;
      PFN_vkCmdSetLineWidth                  vkCmdSetLineWidth                  = 0;
      PFN_vkCmdSetDepthBias                  vkCmdSetDepthBias                  = 0;
      PFN_vkCmdSetBlendConstants             vkCmdSetBlendConstants             = 0;
      PFN_vkCmdSetDepthBounds                vkCmdSetDepthBounds                = 0;
      PFN_vkCmdSetStencilCompareMask         vkCmdSetStencilCompareMask         = 0;
      PFN_vkCmdSetStencilWriteMask           vkCmdSetStencilWriteMask           = 0;
      PFN_vkCmdSetStencilReference           vkCmdSetStencilReference           = 0;
      PFN_vkCmdBindDescriptorSets            vkCmdBindDescriptorSets            = 0;
      PFN_vkCmdBindIndexBuffer               vkCmdBindIndexBuffer               = 0;
      PFN_vkCmdBindVertexBuffers             vkCmdBindVertexBuffers             = 0;
      PFN_vkCmdDraw                          vkCmdDraw                          = 0;
      PFN_vkCmdDrawIndexed                   vkCmdDrawIndexed                   = 0;
      PFN_vkCmdDrawIndirect                  vkCmdDrawIndirect                  = 0;
      PFN_vkCmdDrawIndexedIndirect           vkCmdDrawIndexedIndirect           = 0;
      PFN_vkCmdDispatch                      vkCmdDispatch                      = 0;
      PFN_vkCmdDispatchIndirect              vkCmdDispatchIndirect              = 0;
      PFN_vkCmdCopyBuffer                    vkCmdCopyBuffer                    = 0;
      PFN_vkCmdCopyImage                     vkCmdCopyImage                     = 0;
      PFN_vkCmdBlitImage                     vkCmdBlitImage                     = 0;
      PFN_vkCmdCopyBufferToImage             vkCmdCopyBufferToImage             = 0;
      PFN_vkCmdCopyImageToBuffer             vkCmdCopyImageToBuffer             = 0;
      PFN_vkCmdUpdateBuffer                  vkCmdUpdateBuffer                  = 0;
      PFN_vkCmdFillBuffer                    vkCmdFillBuffer                    = 0;
      PFN_vkCmdClearColorImage               vkCmdClearColorImage               = 0;
      PFN_vkCmdClearDepthStencilImage        vkCmdClearDepthStencilImage        = 0;
      PFN_vkCmdClearAttachments              vkCmdClearAttachments              = 0;
      PFN_vkCmdResolveImage                  vkCmdResolveImage                  = 0;
      PFN_vkCmdSetEvent                      vkCmdSetEvent                      = 0;
      PFN_vkCmdResetEvent                    vkCmdResetEvent                    = 0;
      PFN_vkCmdWaitEvents                    vkCmdWaitEvents                    = 0;
      PFN_vkCmdPipelineBarrier               vkCmdPipelineBarrier               = 0;
      PFN_vkCmdBeginQuery                    vkCmdBeginQuery                    = 0;
      PFN_vkCmdEndQuery                      vkCmdEndQuery                      = 0;
      PFN_vkCmdResetQueryPool                vkCmdResetQueryPool                = 0;
      PFN_vkCmdWriteTimestamp                vkCmdWriteTimestamp                = 0;
      PFN_vkCmdCopyQueryPoolResults          vkCmdCopyQueryPoolResults          = 0;
      PFN_vkCmdPushConstants                 vkCmdPushConstants                 = 0;
      PFN_vkCmdBeginRenderPass               vkCmdBeginRenderPass               = 0;
      PFN_vkCmdNextSubpass                   vkCmdNextSubpass                   = 0;
      PFN_vkCmdEndRenderPass                 vkCmdEndRenderPass                 = 0;
      PFN_vkCmdExecuteCommands               vkCmdExecuteCommands               = 0;

      //=== VK_VERSION_1_1 ===
      PFN_vkBindBufferMemory2                 vkBindBufferMemory2                 = 0;
      PFN_vkBindImageMemory2                  vkBindImageMemory2                  = 0;
      PFN_vkGetDeviceGroupPeerMemoryFeatures  vkGetDeviceGroupPeerMemoryFeatures  = 0;
      PFN_vkCmdSetDeviceMask                  vkCmdSetDeviceMask                  = 0;
      PFN_vkCmdDispatchBase                   vkCmdDispatchBase                   = 0;
      PFN_vkGetImageMemoryRequirements2       vkGetImageMemoryRequirements2       = 0;
      PFN_vkGetBufferMemoryRequirements2      vkGetBufferMemoryRequirements2      = 0;
      PFN_vkGetImageSparseMemoryRequirements2 vkGetImageSparseMemoryRequirements2 = 0;
      PFN_vkTrimCommandPool                   vkTrimCommandPool                   = 0;
      PFN_vkGetDeviceQueue2                   vkGetDeviceQueue2                   = 0;
      PFN_vkCreateSamplerYcbcrConversion      vkCreateSamplerYcbcrConversion      = 0;
      PFN_vkDestroySamplerYcbcrConversion     vkDestroySamplerYcbcrConversion     = 0;
      PFN_vkCreateDescriptorUpdateTemplate    vkCreateDescriptorUpdateTemplate    = 0;
      PFN_vkDestroyDescriptorUpdateTemplate   vkDestroyDescriptorUpdateTemplate   = 0;
      PFN_vkUpdateDescriptorSetWithTemplate   vkUpdateDescriptorSetWithTemplate   = 0;
      PFN_vkGetDescriptorSetLayoutSupport     vkGetDescriptorSetLayoutSupport     = 0;

      //=== VK_VERSION_1_2 ===
      PFN_vkCmdDrawIndirectCount                vkCmdDrawIndirectCount                = 0;
      PFN_vkCmdDrawIndexedIndirectCount         vkCmdDrawIndexedIndirectCount         = 0;
      PFN_vkCreateRenderPass2                   vkCreateRenderPass2                   = 0;
      PFN_vkCmdBeginRenderPass2                 vkCmdBeginRenderPass2                 = 0;
      PFN_vkCmdNextSubpass2                     vkCmdNextSubpass2                     = 0;
      PFN_vkCmdEndRenderPass2                   vkCmdEndRenderPass2                   = 0;
      PFN_vkResetQueryPool                      vkResetQueryPool                      = 0;
      PFN_vkGetSemaphoreCounterValue            vkGetSemaphoreCounterValue            = 0;
      PFN_vkWaitSemaphores                      vkWaitSemaphores                      = 0;
      PFN_vkSignalSemaphore                     vkSignalSemaphore                     = 0;
      PFN_vkGetBufferDeviceAddress              vkGetBufferDeviceAddress              = 0;
      PFN_vkGetBufferOpaqueCaptureAddress       vkGetBufferOpaqueCaptureAddress       = 0;
      PFN_vkGetDeviceMemoryOpaqueCaptureAddress vkGetDeviceMemoryOpaqueCaptureAddress = 0;

      //=== VK_AMD_buffer_marker ===
      PFN_vkCmdWriteBufferMarkerAMD vkCmdWriteBufferMarkerAMD = 0;

      //=== VK_AMD_display_native_hdr ===
      PFN_vkSetLocalDimmingAMD vkSetLocalDimmingAMD = 0;

      //=== VK_AMD_draw_indirect_count ===
      PFN_vkCmdDrawIndirectCountAMD        vkCmdDrawIndirectCountAMD        = 0;
      PFN_vkCmdDrawIndexedIndirectCountAMD vkCmdDrawIndexedIndirectCountAMD = 0;

      //=== VK_AMD_shader_info ===
      PFN_vkGetShaderInfoAMD vkGetShaderInfoAMD = 0;

#  if defined( VK_USE_PLATFORM_ANDROID_KHR )
      //=== VK_ANDROID_external_memory_android_hardware_buffer ===
      PFN_vkGetAndroidHardwareBufferPropertiesANDROID vkGetAndroidHardwareBufferPropertiesANDROID = 0;
      PFN_vkGetMemoryAndroidHardwareBufferANDROID     vkGetMemoryAndroidHardwareBufferANDROID     = 0;
#  else
      PFN_dummy vkGetAndroidHardwareBufferPropertiesANDROID_placeholder       = 0;
      PFN_dummy vkGetMemoryAndroidHardwareBufferANDROID_placeholder           = 0;
#  endif /*VK_USE_PLATFORM_ANDROID_KHR*/

      //=== VK_EXT_buffer_device_address ===
      PFN_vkGetBufferDeviceAddressEXT vkGetBufferDeviceAddressEXT = 0;

      //=== VK_EXT_calibrated_timestamps ===
      PFN_vkGetCalibratedTimestampsEXT vkGetCalibratedTimestampsEXT = 0;

      //=== VK_EXT_color_write_enable ===
      PFN_vkCmdSetColorWriteEnableEXT vkCmdSetColorWriteEnableEXT = 0;

      //=== VK_EXT_conditional_rendering ===
      PFN_vkCmdBeginConditionalRenderingEXT vkCmdBeginConditionalRenderingEXT = 0;
      PFN_vkCmdEndConditionalRenderingEXT   vkCmdEndConditionalRenderingEXT   = 0;

      //=== VK_EXT_debug_marker ===
      PFN_vkDebugMarkerSetObjectTagEXT  vkDebugMarkerSetObjectTagEXT  = 0;
      PFN_vkDebugMarkerSetObjectNameEXT vkDebugMarkerSetObjectNameEXT = 0;
      PFN_vkCmdDebugMarkerBeginEXT      vkCmdDebugMarkerBeginEXT      = 0;
      PFN_vkCmdDebugMarkerEndEXT        vkCmdDebugMarkerEndEXT        = 0;
      PFN_vkCmdDebugMarkerInsertEXT     vkCmdDebugMarkerInsertEXT     = 0;

      //=== VK_EXT_debug_utils ===
      PFN_vkSetDebugUtilsObjectNameEXT    vkSetDebugUtilsObjectNameEXT    = 0;
      PFN_vkSetDebugUtilsObjectTagEXT     vkSetDebugUtilsObjectTagEXT     = 0;
      PFN_vkQueueBeginDebugUtilsLabelEXT  vkQueueBeginDebugUtilsLabelEXT  = 0;
      PFN_vkQueueEndDebugUtilsLabelEXT    vkQueueEndDebugUtilsLabelEXT    = 0;
      PFN_vkQueueInsertDebugUtilsLabelEXT vkQueueInsertDebugUtilsLabelEXT = 0;
      PFN_vkCmdBeginDebugUtilsLabelEXT    vkCmdBeginDebugUtilsLabelEXT    = 0;
      PFN_vkCmdEndDebugUtilsLabelEXT      vkCmdEndDebugUtilsLabelEXT      = 0;
      PFN_vkCmdInsertDebugUtilsLabelEXT   vkCmdInsertDebugUtilsLabelEXT   = 0;

      //=== VK_EXT_discard_rectangles ===
      PFN_vkCmdSetDiscardRectangleEXT vkCmdSetDiscardRectangleEXT = 0;

      //=== VK_EXT_display_control ===
      PFN_vkDisplayPowerControlEXT  vkDisplayPowerControlEXT  = 0;
      PFN_vkRegisterDeviceEventEXT  vkRegisterDeviceEventEXT  = 0;
      PFN_vkRegisterDisplayEventEXT vkRegisterDisplayEventEXT = 0;
      PFN_vkGetSwapchainCounterEXT  vkGetSwapchainCounterEXT  = 0;

      //=== VK_EXT_extended_dynamic_state ===
      PFN_vkCmdSetCullModeEXT              vkCmdSetCullModeEXT              = 0;
      PFN_vkCmdSetFrontFaceEXT             vkCmdSetFrontFaceEXT             = 0;
      PFN_vkCmdSetPrimitiveTopologyEXT     vkCmdSetPrimitiveTopologyEXT     = 0;
      PFN_vkCmdSetViewportWithCountEXT     vkCmdSetViewportWithCountEXT     = 0;
      PFN_vkCmdSetScissorWithCountEXT      vkCmdSetScissorWithCountEXT      = 0;
      PFN_vkCmdBindVertexBuffers2EXT       vkCmdBindVertexBuffers2EXT       = 0;
      PFN_vkCmdSetDepthTestEnableEXT       vkCmdSetDepthTestEnableEXT       = 0;
      PFN_vkCmdSetDepthWriteEnableEXT      vkCmdSetDepthWriteEnableEXT      = 0;
      PFN_vkCmdSetDepthCompareOpEXT        vkCmdSetDepthCompareOpEXT        = 0;
      PFN_vkCmdSetDepthBoundsTestEnableEXT vkCmdSetDepthBoundsTestEnableEXT = 0;
      PFN_vkCmdSetStencilTestEnableEXT     vkCmdSetStencilTestEnableEXT     = 0;
      PFN_vkCmdSetStencilOpEXT             vkCmdSetStencilOpEXT             = 0;

      //=== VK_EXT_extended_dynamic_state2 ===
      PFN_vkCmdSetPatchControlPointsEXT      vkCmdSetPatchControlPointsEXT      = 0;
      PFN_vkCmdSetRasterizerDiscardEnableEXT vkCmdSetRasterizerDiscardEnableEXT = 0;
      PFN_vkCmdSetDepthBiasEnableEXT         vkCmdSetDepthBiasEnableEXT         = 0;
      PFN_vkCmdSetLogicOpEXT                 vkCmdSetLogicOpEXT                 = 0;
      PFN_vkCmdSetPrimitiveRestartEnableEXT  vkCmdSetPrimitiveRestartEnableEXT  = 0;

      //=== VK_EXT_external_memory_host ===
      PFN_vkGetMemoryHostPointerPropertiesEXT vkGetMemoryHostPointerPropertiesEXT = 0;

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_EXT_full_screen_exclusive ===
      PFN_vkAcquireFullScreenExclusiveModeEXT     vkAcquireFullScreenExclusiveModeEXT     = 0;
      PFN_vkReleaseFullScreenExclusiveModeEXT     vkReleaseFullScreenExclusiveModeEXT     = 0;
      PFN_vkGetDeviceGroupSurfacePresentModes2EXT vkGetDeviceGroupSurfacePresentModes2EXT = 0;
#  else
      PFN_dummy vkAcquireFullScreenExclusiveModeEXT_placeholder               = 0;
      PFN_dummy vkReleaseFullScreenExclusiveModeEXT_placeholder               = 0;
      PFN_dummy vkGetDeviceGroupSurfacePresentModes2EXT_placeholder           = 0;
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

      //=== VK_EXT_hdr_metadata ===
      PFN_vkSetHdrMetadataEXT vkSetHdrMetadataEXT = 0;

      //=== VK_EXT_host_query_reset ===
      PFN_vkResetQueryPoolEXT vkResetQueryPoolEXT = 0;

      //=== VK_EXT_image_drm_format_modifier ===
      PFN_vkGetImageDrmFormatModifierPropertiesEXT vkGetImageDrmFormatModifierPropertiesEXT = 0;

      //=== VK_EXT_line_rasterization ===
      PFN_vkCmdSetLineStippleEXT vkCmdSetLineStippleEXT = 0;

      //=== VK_EXT_multi_draw ===
      PFN_vkCmdDrawMultiEXT        vkCmdDrawMultiEXT        = 0;
      PFN_vkCmdDrawMultiIndexedEXT vkCmdDrawMultiIndexedEXT = 0;

      //=== VK_EXT_private_data ===
      PFN_vkCreatePrivateDataSlotEXT  vkCreatePrivateDataSlotEXT  = 0;
      PFN_vkDestroyPrivateDataSlotEXT vkDestroyPrivateDataSlotEXT = 0;
      PFN_vkSetPrivateDataEXT         vkSetPrivateDataEXT         = 0;
      PFN_vkGetPrivateDataEXT         vkGetPrivateDataEXT         = 0;

      //=== VK_EXT_sample_locations ===
      PFN_vkCmdSetSampleLocationsEXT vkCmdSetSampleLocationsEXT = 0;

      //=== VK_EXT_transform_feedback ===
      PFN_vkCmdBindTransformFeedbackBuffersEXT vkCmdBindTransformFeedbackBuffersEXT = 0;
      PFN_vkCmdBeginTransformFeedbackEXT       vkCmdBeginTransformFeedbackEXT       = 0;
      PFN_vkCmdEndTransformFeedbackEXT         vkCmdEndTransformFeedbackEXT         = 0;
      PFN_vkCmdBeginQueryIndexedEXT            vkCmdBeginQueryIndexedEXT            = 0;
      PFN_vkCmdEndQueryIndexedEXT              vkCmdEndQueryIndexedEXT              = 0;
      PFN_vkCmdDrawIndirectByteCountEXT        vkCmdDrawIndirectByteCountEXT        = 0;

      //=== VK_EXT_validation_cache ===
      PFN_vkCreateValidationCacheEXT  vkCreateValidationCacheEXT  = 0;
      PFN_vkDestroyValidationCacheEXT vkDestroyValidationCacheEXT = 0;
      PFN_vkMergeValidationCachesEXT  vkMergeValidationCachesEXT  = 0;
      PFN_vkGetValidationCacheDataEXT vkGetValidationCacheDataEXT = 0;

      //=== VK_EXT_vertex_input_dynamic_state ===
      PFN_vkCmdSetVertexInputEXT vkCmdSetVertexInputEXT = 0;

#  if defined( VK_USE_PLATFORM_FUCHSIA )
      //=== VK_FUCHSIA_external_memory ===
      PFN_vkGetMemoryZirconHandleFUCHSIA           vkGetMemoryZirconHandleFUCHSIA           = 0;
      PFN_vkGetMemoryZirconHandlePropertiesFUCHSIA vkGetMemoryZirconHandlePropertiesFUCHSIA = 0;
#  else
      PFN_dummy vkGetMemoryZirconHandleFUCHSIA_placeholder                    = 0;
      PFN_dummy vkGetMemoryZirconHandlePropertiesFUCHSIA_placeholder          = 0;
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
      //=== VK_FUCHSIA_external_semaphore ===
      PFN_vkImportSemaphoreZirconHandleFUCHSIA vkImportSemaphoreZirconHandleFUCHSIA = 0;
      PFN_vkGetSemaphoreZirconHandleFUCHSIA    vkGetSemaphoreZirconHandleFUCHSIA    = 0;
#  else
      PFN_dummy vkImportSemaphoreZirconHandleFUCHSIA_placeholder              = 0;
      PFN_dummy vkGetSemaphoreZirconHandleFUCHSIA_placeholder                 = 0;
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

      //=== VK_GOOGLE_display_timing ===
      PFN_vkGetRefreshCycleDurationGOOGLE   vkGetRefreshCycleDurationGOOGLE   = 0;
      PFN_vkGetPastPresentationTimingGOOGLE vkGetPastPresentationTimingGOOGLE = 0;

      //=== VK_HUAWEI_invocation_mask ===
      PFN_vkCmdBindInvocationMaskHUAWEI vkCmdBindInvocationMaskHUAWEI = 0;

      //=== VK_HUAWEI_subpass_shading ===
      PFN_vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI = 0;
      PFN_vkCmdSubpassShadingHUAWEI                       vkCmdSubpassShadingHUAWEI                       = 0;

      //=== VK_INTEL_performance_query ===
      PFN_vkInitializePerformanceApiINTEL         vkInitializePerformanceApiINTEL         = 0;
      PFN_vkUninitializePerformanceApiINTEL       vkUninitializePerformanceApiINTEL       = 0;
      PFN_vkCmdSetPerformanceMarkerINTEL          vkCmdSetPerformanceMarkerINTEL          = 0;
      PFN_vkCmdSetPerformanceStreamMarkerINTEL    vkCmdSetPerformanceStreamMarkerINTEL    = 0;
      PFN_vkCmdSetPerformanceOverrideINTEL        vkCmdSetPerformanceOverrideINTEL        = 0;
      PFN_vkAcquirePerformanceConfigurationINTEL  vkAcquirePerformanceConfigurationINTEL  = 0;
      PFN_vkReleasePerformanceConfigurationINTEL  vkReleasePerformanceConfigurationINTEL  = 0;
      PFN_vkQueueSetPerformanceConfigurationINTEL vkQueueSetPerformanceConfigurationINTEL = 0;
      PFN_vkGetPerformanceParameterINTEL          vkGetPerformanceParameterINTEL          = 0;

      //=== VK_KHR_acceleration_structure ===
      PFN_vkCreateAccelerationStructureKHR                 vkCreateAccelerationStructureKHR                 = 0;
      PFN_vkDestroyAccelerationStructureKHR                vkDestroyAccelerationStructureKHR                = 0;
      PFN_vkCmdBuildAccelerationStructuresKHR              vkCmdBuildAccelerationStructuresKHR              = 0;
      PFN_vkCmdBuildAccelerationStructuresIndirectKHR      vkCmdBuildAccelerationStructuresIndirectKHR      = 0;
      PFN_vkBuildAccelerationStructuresKHR                 vkBuildAccelerationStructuresKHR                 = 0;
      PFN_vkCopyAccelerationStructureKHR                   vkCopyAccelerationStructureKHR                   = 0;
      PFN_vkCopyAccelerationStructureToMemoryKHR           vkCopyAccelerationStructureToMemoryKHR           = 0;
      PFN_vkCopyMemoryToAccelerationStructureKHR           vkCopyMemoryToAccelerationStructureKHR           = 0;
      PFN_vkWriteAccelerationStructuresPropertiesKHR       vkWriteAccelerationStructuresPropertiesKHR       = 0;
      PFN_vkCmdCopyAccelerationStructureKHR                vkCmdCopyAccelerationStructureKHR                = 0;
      PFN_vkCmdCopyAccelerationStructureToMemoryKHR        vkCmdCopyAccelerationStructureToMemoryKHR        = 0;
      PFN_vkCmdCopyMemoryToAccelerationStructureKHR        vkCmdCopyMemoryToAccelerationStructureKHR        = 0;
      PFN_vkGetAccelerationStructureDeviceAddressKHR       vkGetAccelerationStructureDeviceAddressKHR       = 0;
      PFN_vkCmdWriteAccelerationStructuresPropertiesKHR    vkCmdWriteAccelerationStructuresPropertiesKHR    = 0;
      PFN_vkGetDeviceAccelerationStructureCompatibilityKHR vkGetDeviceAccelerationStructureCompatibilityKHR = 0;
      PFN_vkGetAccelerationStructureBuildSizesKHR          vkGetAccelerationStructureBuildSizesKHR          = 0;

      //=== VK_KHR_bind_memory2 ===
      PFN_vkBindBufferMemory2KHR vkBindBufferMemory2KHR = 0;
      PFN_vkBindImageMemory2KHR  vkBindImageMemory2KHR  = 0;

      //=== VK_KHR_buffer_device_address ===
      PFN_vkGetBufferDeviceAddressKHR              vkGetBufferDeviceAddressKHR              = 0;
      PFN_vkGetBufferOpaqueCaptureAddressKHR       vkGetBufferOpaqueCaptureAddressKHR       = 0;
      PFN_vkGetDeviceMemoryOpaqueCaptureAddressKHR vkGetDeviceMemoryOpaqueCaptureAddressKHR = 0;

      //=== VK_KHR_copy_commands2 ===
      PFN_vkCmdCopyBuffer2KHR        vkCmdCopyBuffer2KHR        = 0;
      PFN_vkCmdCopyImage2KHR         vkCmdCopyImage2KHR         = 0;
      PFN_vkCmdCopyBufferToImage2KHR vkCmdCopyBufferToImage2KHR = 0;
      PFN_vkCmdCopyImageToBuffer2KHR vkCmdCopyImageToBuffer2KHR = 0;
      PFN_vkCmdBlitImage2KHR         vkCmdBlitImage2KHR         = 0;
      PFN_vkCmdResolveImage2KHR      vkCmdResolveImage2KHR      = 0;

      //=== VK_KHR_create_renderpass2 ===
      PFN_vkCreateRenderPass2KHR   vkCreateRenderPass2KHR   = 0;
      PFN_vkCmdBeginRenderPass2KHR vkCmdBeginRenderPass2KHR = 0;
      PFN_vkCmdNextSubpass2KHR     vkCmdNextSubpass2KHR     = 0;
      PFN_vkCmdEndRenderPass2KHR   vkCmdEndRenderPass2KHR   = 0;

      //=== VK_KHR_deferred_host_operations ===
      PFN_vkCreateDeferredOperationKHR            vkCreateDeferredOperationKHR            = 0;
      PFN_vkDestroyDeferredOperationKHR           vkDestroyDeferredOperationKHR           = 0;
      PFN_vkGetDeferredOperationMaxConcurrencyKHR vkGetDeferredOperationMaxConcurrencyKHR = 0;
      PFN_vkGetDeferredOperationResultKHR         vkGetDeferredOperationResultKHR         = 0;
      PFN_vkDeferredOperationJoinKHR              vkDeferredOperationJoinKHR              = 0;

      //=== VK_KHR_descriptor_update_template ===
      PFN_vkCreateDescriptorUpdateTemplateKHR   vkCreateDescriptorUpdateTemplateKHR   = 0;
      PFN_vkDestroyDescriptorUpdateTemplateKHR  vkDestroyDescriptorUpdateTemplateKHR  = 0;
      PFN_vkUpdateDescriptorSetWithTemplateKHR  vkUpdateDescriptorSetWithTemplateKHR  = 0;
      PFN_vkCmdPushDescriptorSetWithTemplateKHR vkCmdPushDescriptorSetWithTemplateKHR = 0;

      //=== VK_KHR_device_group ===
      PFN_vkGetDeviceGroupPeerMemoryFeaturesKHR  vkGetDeviceGroupPeerMemoryFeaturesKHR  = 0;
      PFN_vkCmdSetDeviceMaskKHR                  vkCmdSetDeviceMaskKHR                  = 0;
      PFN_vkCmdDispatchBaseKHR                   vkCmdDispatchBaseKHR                   = 0;
      PFN_vkGetDeviceGroupPresentCapabilitiesKHR vkGetDeviceGroupPresentCapabilitiesKHR = 0;
      PFN_vkGetDeviceGroupSurfacePresentModesKHR vkGetDeviceGroupSurfacePresentModesKHR = 0;
      PFN_vkAcquireNextImage2KHR                 vkAcquireNextImage2KHR                 = 0;

      //=== VK_KHR_display_swapchain ===
      PFN_vkCreateSharedSwapchainsKHR vkCreateSharedSwapchainsKHR = 0;

      //=== VK_KHR_draw_indirect_count ===
      PFN_vkCmdDrawIndirectCountKHR        vkCmdDrawIndirectCountKHR        = 0;
      PFN_vkCmdDrawIndexedIndirectCountKHR vkCmdDrawIndexedIndirectCountKHR = 0;

      //=== VK_KHR_external_fence_fd ===
      PFN_vkImportFenceFdKHR vkImportFenceFdKHR = 0;
      PFN_vkGetFenceFdKHR    vkGetFenceFdKHR    = 0;

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_KHR_external_fence_win32 ===
      PFN_vkImportFenceWin32HandleKHR vkImportFenceWin32HandleKHR = 0;
      PFN_vkGetFenceWin32HandleKHR    vkGetFenceWin32HandleKHR    = 0;
#  else
      PFN_dummy vkImportFenceWin32HandleKHR_placeholder                       = 0;
      PFN_dummy vkGetFenceWin32HandleKHR_placeholder                          = 0;
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

      //=== VK_KHR_external_memory_fd ===
      PFN_vkGetMemoryFdKHR           vkGetMemoryFdKHR           = 0;
      PFN_vkGetMemoryFdPropertiesKHR vkGetMemoryFdPropertiesKHR = 0;

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_KHR_external_memory_win32 ===
      PFN_vkGetMemoryWin32HandleKHR           vkGetMemoryWin32HandleKHR           = 0;
      PFN_vkGetMemoryWin32HandlePropertiesKHR vkGetMemoryWin32HandlePropertiesKHR = 0;
#  else
      PFN_dummy vkGetMemoryWin32HandleKHR_placeholder                         = 0;
      PFN_dummy vkGetMemoryWin32HandlePropertiesKHR_placeholder               = 0;
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

      //=== VK_KHR_external_semaphore_fd ===
      PFN_vkImportSemaphoreFdKHR vkImportSemaphoreFdKHR = 0;
      PFN_vkGetSemaphoreFdKHR    vkGetSemaphoreFdKHR    = 0;

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_KHR_external_semaphore_win32 ===
      PFN_vkImportSemaphoreWin32HandleKHR vkImportSemaphoreWin32HandleKHR = 0;
      PFN_vkGetSemaphoreWin32HandleKHR    vkGetSemaphoreWin32HandleKHR    = 0;
#  else
      PFN_dummy vkImportSemaphoreWin32HandleKHR_placeholder                   = 0;
      PFN_dummy vkGetSemaphoreWin32HandleKHR_placeholder                      = 0;
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

      //=== VK_KHR_fragment_shading_rate ===
      PFN_vkCmdSetFragmentShadingRateKHR vkCmdSetFragmentShadingRateKHR = 0;

      //=== VK_KHR_get_memory_requirements2 ===
      PFN_vkGetImageMemoryRequirements2KHR       vkGetImageMemoryRequirements2KHR       = 0;
      PFN_vkGetBufferMemoryRequirements2KHR      vkGetBufferMemoryRequirements2KHR      = 0;
      PFN_vkGetImageSparseMemoryRequirements2KHR vkGetImageSparseMemoryRequirements2KHR = 0;

      //=== VK_KHR_maintenance1 ===
      PFN_vkTrimCommandPoolKHR vkTrimCommandPoolKHR = 0;

      //=== VK_KHR_maintenance3 ===
      PFN_vkGetDescriptorSetLayoutSupportKHR vkGetDescriptorSetLayoutSupportKHR = 0;

      //=== VK_KHR_performance_query ===
      PFN_vkAcquireProfilingLockKHR vkAcquireProfilingLockKHR = 0;
      PFN_vkReleaseProfilingLockKHR vkReleaseProfilingLockKHR = 0;

      //=== VK_KHR_pipeline_executable_properties ===
      PFN_vkGetPipelineExecutablePropertiesKHR              vkGetPipelineExecutablePropertiesKHR              = 0;
      PFN_vkGetPipelineExecutableStatisticsKHR              vkGetPipelineExecutableStatisticsKHR              = 0;
      PFN_vkGetPipelineExecutableInternalRepresentationsKHR vkGetPipelineExecutableInternalRepresentationsKHR = 0;

      //=== VK_KHR_present_wait ===
      PFN_vkWaitForPresentKHR vkWaitForPresentKHR = 0;

      //=== VK_KHR_push_descriptor ===
      PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR = 0;

      //=== VK_KHR_ray_tracing_pipeline ===
      PFN_vkCmdTraceRaysKHR                                 vkCmdTraceRaysKHR                                 = 0;
      PFN_vkCreateRayTracingPipelinesKHR                    vkCreateRayTracingPipelinesKHR                    = 0;
      PFN_vkGetRayTracingShaderGroupHandlesKHR              vkGetRayTracingShaderGroupHandlesKHR              = 0;
      PFN_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR vkGetRayTracingCaptureReplayShaderGroupHandlesKHR = 0;
      PFN_vkCmdTraceRaysIndirectKHR                         vkCmdTraceRaysIndirectKHR                         = 0;
      PFN_vkGetRayTracingShaderGroupStackSizeKHR            vkGetRayTracingShaderGroupStackSizeKHR            = 0;
      PFN_vkCmdSetRayTracingPipelineStackSizeKHR            vkCmdSetRayTracingPipelineStackSizeKHR            = 0;

      //=== VK_KHR_sampler_ycbcr_conversion ===
      PFN_vkCreateSamplerYcbcrConversionKHR  vkCreateSamplerYcbcrConversionKHR  = 0;
      PFN_vkDestroySamplerYcbcrConversionKHR vkDestroySamplerYcbcrConversionKHR = 0;

      //=== VK_KHR_shared_presentable_image ===
      PFN_vkGetSwapchainStatusKHR vkGetSwapchainStatusKHR = 0;

      //=== VK_KHR_swapchain ===
      PFN_vkCreateSwapchainKHR    vkCreateSwapchainKHR    = 0;
      PFN_vkDestroySwapchainKHR   vkDestroySwapchainKHR   = 0;
      PFN_vkGetSwapchainImagesKHR vkGetSwapchainImagesKHR = 0;
      PFN_vkAcquireNextImageKHR   vkAcquireNextImageKHR   = 0;
      PFN_vkQueuePresentKHR       vkQueuePresentKHR       = 0;

      //=== VK_KHR_synchronization2 ===
      PFN_vkCmdSetEvent2KHR           vkCmdSetEvent2KHR           = 0;
      PFN_vkCmdResetEvent2KHR         vkCmdResetEvent2KHR         = 0;
      PFN_vkCmdWaitEvents2KHR         vkCmdWaitEvents2KHR         = 0;
      PFN_vkCmdPipelineBarrier2KHR    vkCmdPipelineBarrier2KHR    = 0;
      PFN_vkCmdWriteTimestamp2KHR     vkCmdWriteTimestamp2KHR     = 0;
      PFN_vkQueueSubmit2KHR           vkQueueSubmit2KHR           = 0;
      PFN_vkCmdWriteBufferMarker2AMD  vkCmdWriteBufferMarker2AMD  = 0;
      PFN_vkGetQueueCheckpointData2NV vkGetQueueCheckpointData2NV = 0;

      //=== VK_KHR_timeline_semaphore ===
      PFN_vkGetSemaphoreCounterValueKHR vkGetSemaphoreCounterValueKHR = 0;
      PFN_vkWaitSemaphoresKHR           vkWaitSemaphoresKHR           = 0;
      PFN_vkSignalSemaphoreKHR          vkSignalSemaphoreKHR          = 0;

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
      //=== VK_KHR_video_decode_queue ===
      PFN_vkCmdDecodeVideoKHR vkCmdDecodeVideoKHR = 0;
#  else
      PFN_dummy vkCmdDecodeVideoKHR_placeholder                               = 0;
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
      //=== VK_KHR_video_encode_queue ===
      PFN_vkCmdEncodeVideoKHR vkCmdEncodeVideoKHR = 0;
#  else
      PFN_dummy vkCmdEncodeVideoKHR_placeholder                               = 0;
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
      //=== VK_KHR_video_queue ===
      PFN_vkCreateVideoSessionKHR                vkCreateVideoSessionKHR                = 0;
      PFN_vkDestroyVideoSessionKHR               vkDestroyVideoSessionKHR               = 0;
      PFN_vkGetVideoSessionMemoryRequirementsKHR vkGetVideoSessionMemoryRequirementsKHR = 0;
      PFN_vkBindVideoSessionMemoryKHR            vkBindVideoSessionMemoryKHR            = 0;
      PFN_vkCreateVideoSessionParametersKHR      vkCreateVideoSessionParametersKHR      = 0;
      PFN_vkUpdateVideoSessionParametersKHR      vkUpdateVideoSessionParametersKHR      = 0;
      PFN_vkDestroyVideoSessionParametersKHR     vkDestroyVideoSessionParametersKHR     = 0;
      PFN_vkCmdBeginVideoCodingKHR               vkCmdBeginVideoCodingKHR               = 0;
      PFN_vkCmdEndVideoCodingKHR                 vkCmdEndVideoCodingKHR                 = 0;
      PFN_vkCmdControlVideoCodingKHR             vkCmdControlVideoCodingKHR             = 0;
#  else
      PFN_dummy vkCreateVideoSessionKHR_placeholder                           = 0;
      PFN_dummy vkDestroyVideoSessionKHR_placeholder                          = 0;
      PFN_dummy vkGetVideoSessionMemoryRequirementsKHR_placeholder            = 0;
      PFN_dummy vkBindVideoSessionMemoryKHR_placeholder                       = 0;
      PFN_dummy vkCreateVideoSessionParametersKHR_placeholder                 = 0;
      PFN_dummy vkUpdateVideoSessionParametersKHR_placeholder                 = 0;
      PFN_dummy vkDestroyVideoSessionParametersKHR_placeholder                = 0;
      PFN_dummy vkCmdBeginVideoCodingKHR_placeholder                          = 0;
      PFN_dummy vkCmdEndVideoCodingKHR_placeholder                            = 0;
      PFN_dummy vkCmdControlVideoCodingKHR_placeholder                        = 0;
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

      //=== VK_NVX_binary_import ===
      PFN_vkCreateCuModuleNVX    vkCreateCuModuleNVX    = 0;
      PFN_vkCreateCuFunctionNVX  vkCreateCuFunctionNVX  = 0;
      PFN_vkDestroyCuModuleNVX   vkDestroyCuModuleNVX   = 0;
      PFN_vkDestroyCuFunctionNVX vkDestroyCuFunctionNVX = 0;
      PFN_vkCmdCuLaunchKernelNVX vkCmdCuLaunchKernelNVX = 0;

      //=== VK_NVX_image_view_handle ===
      PFN_vkGetImageViewHandleNVX  vkGetImageViewHandleNVX  = 0;
      PFN_vkGetImageViewAddressNVX vkGetImageViewAddressNVX = 0;

      //=== VK_NV_clip_space_w_scaling ===
      PFN_vkCmdSetViewportWScalingNV vkCmdSetViewportWScalingNV = 0;

      //=== VK_NV_device_diagnostic_checkpoints ===
      PFN_vkCmdSetCheckpointNV       vkCmdSetCheckpointNV       = 0;
      PFN_vkGetQueueCheckpointDataNV vkGetQueueCheckpointDataNV = 0;

      //=== VK_NV_device_generated_commands ===
      PFN_vkGetGeneratedCommandsMemoryRequirementsNV vkGetGeneratedCommandsMemoryRequirementsNV = 0;
      PFN_vkCmdPreprocessGeneratedCommandsNV         vkCmdPreprocessGeneratedCommandsNV         = 0;
      PFN_vkCmdExecuteGeneratedCommandsNV            vkCmdExecuteGeneratedCommandsNV            = 0;
      PFN_vkCmdBindPipelineShaderGroupNV             vkCmdBindPipelineShaderGroupNV             = 0;
      PFN_vkCreateIndirectCommandsLayoutNV           vkCreateIndirectCommandsLayoutNV           = 0;
      PFN_vkDestroyIndirectCommandsLayoutNV          vkDestroyIndirectCommandsLayoutNV          = 0;

      //=== VK_NV_external_memory_rdma ===
      PFN_vkGetMemoryRemoteAddressNV vkGetMemoryRemoteAddressNV = 0;

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_NV_external_memory_win32 ===
      PFN_vkGetMemoryWin32HandleNV vkGetMemoryWin32HandleNV = 0;
#  else
      PFN_dummy vkGetMemoryWin32HandleNV_placeholder                          = 0;
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

      //=== VK_NV_fragment_shading_rate_enums ===
      PFN_vkCmdSetFragmentShadingRateEnumNV vkCmdSetFragmentShadingRateEnumNV = 0;

      //=== VK_NV_mesh_shader ===
      PFN_vkCmdDrawMeshTasksNV              vkCmdDrawMeshTasksNV              = 0;
      PFN_vkCmdDrawMeshTasksIndirectNV      vkCmdDrawMeshTasksIndirectNV      = 0;
      PFN_vkCmdDrawMeshTasksIndirectCountNV vkCmdDrawMeshTasksIndirectCountNV = 0;

      //=== VK_NV_ray_tracing ===
      PFN_vkCreateAccelerationStructureNV                vkCreateAccelerationStructureNV                = 0;
      PFN_vkDestroyAccelerationStructureNV               vkDestroyAccelerationStructureNV               = 0;
      PFN_vkGetAccelerationStructureMemoryRequirementsNV vkGetAccelerationStructureMemoryRequirementsNV = 0;
      PFN_vkBindAccelerationStructureMemoryNV            vkBindAccelerationStructureMemoryNV            = 0;
      PFN_vkCmdBuildAccelerationStructureNV              vkCmdBuildAccelerationStructureNV              = 0;
      PFN_vkCmdCopyAccelerationStructureNV               vkCmdCopyAccelerationStructureNV               = 0;
      PFN_vkCmdTraceRaysNV                               vkCmdTraceRaysNV                               = 0;
      PFN_vkCreateRayTracingPipelinesNV                  vkCreateRayTracingPipelinesNV                  = 0;
      PFN_vkGetRayTracingShaderGroupHandlesNV            vkGetRayTracingShaderGroupHandlesNV            = 0;
      PFN_vkGetAccelerationStructureHandleNV             vkGetAccelerationStructureHandleNV             = 0;
      PFN_vkCmdWriteAccelerationStructuresPropertiesNV   vkCmdWriteAccelerationStructuresPropertiesNV   = 0;
      PFN_vkCompileDeferredNV                            vkCompileDeferredNV                            = 0;

      //=== VK_NV_scissor_exclusive ===
      PFN_vkCmdSetExclusiveScissorNV vkCmdSetExclusiveScissorNV = 0;

      //=== VK_NV_shading_rate_image ===
      PFN_vkCmdBindShadingRateImageNV          vkCmdBindShadingRateImageNV          = 0;
      PFN_vkCmdSetViewportShadingRatePaletteNV vkCmdSetViewportShadingRatePaletteNV = 0;
      PFN_vkCmdSetCoarseSampleOrderNV          vkCmdSetCoarseSampleOrderNV          = 0;
    };

    //====================
    //=== RAII HANDLES ===
    //====================

    class Context
    {
    public:
      Context() : m_dispatcher( m_dynamicLoader.getProcAddress<PFN_vkGetInstanceProcAddr>( "vkGetInstanceProcAddr" ) )
      {}

      ~Context() = default;

      Context( Context const & ) = delete;
      Context( Context && rhs ) VULKAN_HPP_NOEXCEPT
        : m_dynamicLoader( std::move( rhs.m_dynamicLoader ) )
        , m_dispatcher( std::move( rhs.m_dispatcher ) )
      {}
      Context & operator=( Context const & ) = delete;
      Context & operator                     =( Context && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          m_dynamicLoader = std::move( rhs.m_dynamicLoader );
          m_dispatcher    = std::move( rhs.m_dispatcher );
        }
        return *this;
      }

      //=== VK_VERSION_1_0 ===

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::ExtensionProperties> enumerateInstanceExtensionProperties(
        Optional<const std::string> layerName VULKAN_HPP_DEFAULT_ARGUMENT_NULLPTR_ASSIGNMENT ) const;

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::LayerProperties> enumerateInstanceLayerProperties() const;

      //=== VK_VERSION_1_1 ===

      VULKAN_HPP_NODISCARD uint32_t enumerateInstanceVersion() const;

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::ContextDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher.getVkHeaderVersion() == VK_HEADER_VERSION );
        return &m_dispatcher;
      }

    private:
      VULKAN_HPP_NAMESPACE::DynamicLoader                                m_dynamicLoader;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::ContextDispatcher m_dispatcher;
    };

    class Instance
    {
    public:
      using CType = VkInstance;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eInstance;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eInstance;

    public:
      Instance( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Context const &                context,
                VULKAN_HPP_NAMESPACE::InstanceCreateInfo const &                                createInfo,
                VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( context.getDispatcher()->vkGetInstanceProcAddr )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          context.getDispatcher()->vkCreateInstance( reinterpret_cast<const VkInstanceCreateInfo *>( &createInfo ),
                                                     m_allocator,
                                                     reinterpret_cast<VkInstance *>( &m_instance ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateInstance" );
        }
        m_dispatcher.init( static_cast<VkInstance>( m_instance ) );
      }

      Instance( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Context const &                context,
                VkInstance                                                                      instance,
                VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_instance( instance )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( context.getDispatcher()->vkGetInstanceProcAddr )
      {
        m_dispatcher.init( static_cast<VkInstance>( m_instance ) );
      }

      ~Instance()
      {
        if ( m_instance )
        {
          getDispatcher()->vkDestroyInstance( static_cast<VkInstance>( m_instance ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      Instance() = default;
#  else
      Instance()                                                              = delete;
#  endif
      Instance( Instance const & ) = delete;
      Instance( Instance && rhs ) VULKAN_HPP_NOEXCEPT
        : m_instance( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_instance, {} ) )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      Instance & operator=( Instance const & ) = delete;
      Instance & operator                      =( Instance && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_instance )
          {
            getDispatcher()->vkDestroyInstance( static_cast<VkInstance>( m_instance ), m_allocator );
          }
          m_instance   = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_instance, {} );
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::Instance const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_instance;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::InstanceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher.getVkHeaderVersion() == VK_HEADER_VERSION );
        return &m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_instance.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_instance.operator!();
      }
#  endif

      //=== VK_VERSION_1_0 ===

      VULKAN_HPP_NODISCARD PFN_vkVoidFunction getProcAddr( const std::string & name ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_VERSION_1_1 ===

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::PhysicalDeviceGroupProperties>
                           enumeratePhysicalDeviceGroups() const;

      //=== VK_EXT_debug_report ===

      void debugReportMessageEXT( VULKAN_HPP_NAMESPACE::DebugReportFlagsEXT      flags,
                                  VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT objectType_,
                                  uint64_t                                       object,
                                  size_t                                         location,
                                  int32_t                                        messageCode,
                                  const std::string &                            layerPrefix,
                                  const std::string &                            message ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_KHR_device_group_creation ===

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::PhysicalDeviceGroupProperties>
                           enumeratePhysicalDeviceGroupsKHR() const;

      //=== VK_EXT_debug_utils ===

      void
        submitDebugUtilsMessageEXT( VULKAN_HPP_NAMESPACE::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                    VULKAN_HPP_NAMESPACE::DebugUtilsMessageTypeFlagsEXT        messageTypes,
                                    const DebugUtilsMessengerCallbackDataEXT & callbackData ) const VULKAN_HPP_NOEXCEPT;

    private:
      VULKAN_HPP_NAMESPACE::Instance                                      m_instance;
      const VkAllocationCallbacks *                                       m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::InstanceDispatcher m_dispatcher;
    };

    class PhysicalDevice
    {
    public:
      using CType = VkPhysicalDevice;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::ePhysicalDevice;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::ePhysicalDevice;

    public:
      PhysicalDevice( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Instance const & instance,
                      VkPhysicalDevice                                                  physicalDevice )
        : m_physicalDevice( physicalDevice ), m_dispatcher( instance.getDispatcher() )
      {}

      PhysicalDevice( VkPhysicalDevice                                                            physicalDevice,
                      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::InstanceDispatcher const * dispatcher )
        : m_physicalDevice( physicalDevice ), m_dispatcher( dispatcher )
      {}

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      PhysicalDevice() = default;
#  else
      PhysicalDevice()                                                        = delete;
#  endif
      PhysicalDevice( PhysicalDevice const & ) = delete;
      PhysicalDevice( PhysicalDevice && rhs ) VULKAN_HPP_NOEXCEPT
        : m_physicalDevice( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_physicalDevice, {} ) )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      PhysicalDevice & operator=( PhysicalDevice const & ) = delete;
      PhysicalDevice & operator                            =( PhysicalDevice && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          m_physicalDevice = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_physicalDevice, {} );
          m_dispatcher     = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::PhysicalDevice const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_physicalDevice;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::InstanceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_physicalDevice.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_physicalDevice.operator!();
      }
#  endif

      //=== VK_VERSION_1_0 ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures getFeatures() const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::FormatProperties
                           getFormatProperties( VULKAN_HPP_NAMESPACE::Format format ) const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::ImageFormatProperties getImageFormatProperties(
        VULKAN_HPP_NAMESPACE::Format           format,
        VULKAN_HPP_NAMESPACE::ImageType        type,
        VULKAN_HPP_NAMESPACE::ImageTiling      tiling,
        VULKAN_HPP_NAMESPACE::ImageUsageFlags  usage,
        VULKAN_HPP_NAMESPACE::ImageCreateFlags flags VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT ) const;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties getProperties() const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::QueueFamilyProperties>
                           getQueueFamilyProperties() const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties
                           getMemoryProperties() const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::ExtensionProperties> enumerateDeviceExtensionProperties(
        Optional<const std::string> layerName VULKAN_HPP_DEFAULT_ARGUMENT_NULLPTR_ASSIGNMENT ) const;

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::LayerProperties> enumerateDeviceLayerProperties() const;

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::SparseImageFormatProperties>
                           getSparseImageFormatProperties( VULKAN_HPP_NAMESPACE::Format              format,
                                                           VULKAN_HPP_NAMESPACE::ImageType           type,
                                                           VULKAN_HPP_NAMESPACE::SampleCountFlagBits samples,
                                                           VULKAN_HPP_NAMESPACE::ImageUsageFlags     usage,
                                                           VULKAN_HPP_NAMESPACE::ImageTiling         tiling ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_VERSION_1_1 ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures2 getFeatures2() const VULKAN_HPP_NOEXCEPT;

      template <typename X, typename Y, typename... Z>
      VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...> getFeatures2() const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties2 getProperties2() const VULKAN_HPP_NOEXCEPT;

      template <typename X, typename Y, typename... Z>
      VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...> getProperties2() const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::FormatProperties2
                           getFormatProperties2( VULKAN_HPP_NAMESPACE::Format format ) const VULKAN_HPP_NOEXCEPT;

      template <typename X, typename Y, typename... Z>
      VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...>
                           getFormatProperties2( VULKAN_HPP_NAMESPACE::Format format ) const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::ImageFormatProperties2
                           getImageFormatProperties2( const PhysicalDeviceImageFormatInfo2 & imageFormatInfo ) const;

      template <typename X, typename Y, typename... Z>
      VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...>
                           getImageFormatProperties2( const PhysicalDeviceImageFormatInfo2 & imageFormatInfo ) const;

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::QueueFamilyProperties2>
                           getQueueFamilyProperties2() const VULKAN_HPP_NOEXCEPT;

      template <typename StructureChain>
      VULKAN_HPP_NODISCARD std::vector<StructureChain> getQueueFamilyProperties2() const;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties2
                           getMemoryProperties2() const VULKAN_HPP_NOEXCEPT;

      template <typename X, typename Y, typename... Z>
      VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...> getMemoryProperties2() const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::SparseImageFormatProperties2>
                           getSparseImageFormatProperties2( const PhysicalDeviceSparseImageFormatInfo2 & formatInfo ) const
        VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::ExternalBufferProperties getExternalBufferProperties(
        const PhysicalDeviceExternalBufferInfo & externalBufferInfo ) const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::ExternalFenceProperties getExternalFenceProperties(
        const PhysicalDeviceExternalFenceInfo & externalFenceInfo ) const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::ExternalSemaphoreProperties getExternalSemaphoreProperties(
        const PhysicalDeviceExternalSemaphoreInfo & externalSemaphoreInfo ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_KHR_surface ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::Bool32
                           getSurfaceSupportKHR( uint32_t queueFamilyIndex, VULKAN_HPP_NAMESPACE::SurfaceKHR surface ) const;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesKHR
                           getSurfaceCapabilitiesKHR( VULKAN_HPP_NAMESPACE::SurfaceKHR surface ) const;

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::SurfaceFormatKHR>
                           getSurfaceFormatsKHR( VULKAN_HPP_NAMESPACE::SurfaceKHR surface ) const;

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::PresentModeKHR>
                           getSurfacePresentModesKHR( VULKAN_HPP_NAMESPACE::SurfaceKHR surface ) const;

      //=== VK_KHR_swapchain ===

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::Rect2D>
                           getPresentRectanglesKHR( VULKAN_HPP_NAMESPACE::SurfaceKHR surface ) const;

      //=== VK_KHR_display ===

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::DisplayPropertiesKHR> getDisplayPropertiesKHR() const;

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::DisplayPlanePropertiesKHR>
                           getDisplayPlanePropertiesKHR() const;

#  if defined( VK_USE_PLATFORM_XLIB_KHR )
      //=== VK_KHR_xlib_surface ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::Bool32 getXlibPresentationSupportKHR(
        uint32_t queueFamilyIndex, Display & dpy, VisualID visualID ) const VULKAN_HPP_NOEXCEPT;
#  endif /*VK_USE_PLATFORM_XLIB_KHR*/

#  if defined( VK_USE_PLATFORM_XCB_KHR )
      //=== VK_KHR_xcb_surface ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::Bool32 getXcbPresentationSupportKHR(
        uint32_t queueFamilyIndex, xcb_connection_t & connection, xcb_visualid_t visual_id ) const VULKAN_HPP_NOEXCEPT;
#  endif /*VK_USE_PLATFORM_XCB_KHR*/

#  if defined( VK_USE_PLATFORM_WAYLAND_KHR )
      //=== VK_KHR_wayland_surface ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::Bool32
                           getWaylandPresentationSupportKHR( uint32_t            queueFamilyIndex,
                                                             struct wl_display & display ) const VULKAN_HPP_NOEXCEPT;
#  endif /*VK_USE_PLATFORM_WAYLAND_KHR*/

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_KHR_win32_surface ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::Bool32
                           getWin32PresentationSupportKHR( uint32_t queueFamilyIndex ) const VULKAN_HPP_NOEXCEPT;
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
      //=== VK_KHR_video_queue ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::VideoCapabilitiesKHR
                           getVideoCapabilitiesKHR( const VideoProfileKHR & videoProfile ) const;

      template <typename X, typename Y, typename... Z>
      VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...>
                           getVideoCapabilitiesKHR( const VideoProfileKHR & videoProfile ) const;

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::VideoFormatPropertiesKHR>
                           getVideoFormatPropertiesKHR( const PhysicalDeviceVideoFormatInfoKHR & videoFormatInfo ) const;
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

      //=== VK_NV_external_memory_capabilities ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::ExternalImageFormatPropertiesNV getExternalImageFormatPropertiesNV(
        VULKAN_HPP_NAMESPACE::Format           format,
        VULKAN_HPP_NAMESPACE::ImageType        type,
        VULKAN_HPP_NAMESPACE::ImageTiling      tiling,
        VULKAN_HPP_NAMESPACE::ImageUsageFlags  usage,
        VULKAN_HPP_NAMESPACE::ImageCreateFlags flags          VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT,
        VULKAN_HPP_NAMESPACE::ExternalMemoryHandleTypeFlagsNV externalHandleType
                                                              VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT ) const;

      //=== VK_KHR_get_physical_device_properties2 ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures2 getFeatures2KHR() const VULKAN_HPP_NOEXCEPT;

      template <typename X, typename Y, typename... Z>
      VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...> getFeatures2KHR() const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties2
                           getProperties2KHR() const VULKAN_HPP_NOEXCEPT;

      template <typename X, typename Y, typename... Z>
      VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...> getProperties2KHR() const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::FormatProperties2
                           getFormatProperties2KHR( VULKAN_HPP_NAMESPACE::Format format ) const VULKAN_HPP_NOEXCEPT;

      template <typename X, typename Y, typename... Z>
      VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...>
                           getFormatProperties2KHR( VULKAN_HPP_NAMESPACE::Format format ) const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::ImageFormatProperties2
                           getImageFormatProperties2KHR( const PhysicalDeviceImageFormatInfo2 & imageFormatInfo ) const;

      template <typename X, typename Y, typename... Z>
      VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...>
                           getImageFormatProperties2KHR( const PhysicalDeviceImageFormatInfo2 & imageFormatInfo ) const;

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::QueueFamilyProperties2>
                           getQueueFamilyProperties2KHR() const VULKAN_HPP_NOEXCEPT;

      template <typename StructureChain>
      VULKAN_HPP_NODISCARD std::vector<StructureChain> getQueueFamilyProperties2KHR() const;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties2
                           getMemoryProperties2KHR() const VULKAN_HPP_NOEXCEPT;

      template <typename X, typename Y, typename... Z>
      VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...> getMemoryProperties2KHR() const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::SparseImageFormatProperties2>
                           getSparseImageFormatProperties2KHR( const PhysicalDeviceSparseImageFormatInfo2 & formatInfo ) const
        VULKAN_HPP_NOEXCEPT;

      //=== VK_KHR_external_memory_capabilities ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::ExternalBufferProperties getExternalBufferPropertiesKHR(
        const PhysicalDeviceExternalBufferInfo & externalBufferInfo ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_KHR_external_semaphore_capabilities ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::ExternalSemaphoreProperties getExternalSemaphorePropertiesKHR(
        const PhysicalDeviceExternalSemaphoreInfo & externalSemaphoreInfo ) const VULKAN_HPP_NOEXCEPT;

#  if defined( VK_USE_PLATFORM_XLIB_XRANDR_EXT )
      //=== VK_EXT_acquire_xlib_display ===

      void acquireXlibDisplayEXT( Display & dpy, VULKAN_HPP_NAMESPACE::DisplayKHR display ) const;
#  endif /*VK_USE_PLATFORM_XLIB_XRANDR_EXT*/

      //=== VK_EXT_display_surface_counter ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::SurfaceCapabilities2EXT
                           getSurfaceCapabilities2EXT( VULKAN_HPP_NAMESPACE::SurfaceKHR surface ) const;

      //=== VK_KHR_external_fence_capabilities ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::ExternalFenceProperties getExternalFencePropertiesKHR(
        const PhysicalDeviceExternalFenceInfo & externalFenceInfo ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_KHR_performance_query ===

      VULKAN_HPP_NODISCARD std::pair<std::vector<PerformanceCounterKHR>, std::vector<PerformanceCounterDescriptionKHR>>
                           enumerateQueueFamilyPerformanceQueryCountersKHR( uint32_t queueFamilyIndex ) const;

      VULKAN_HPP_NODISCARD uint32_t getQueueFamilyPerformanceQueryPassesKHR(
        const QueryPoolPerformanceCreateInfoKHR & performanceQueryCreateInfo ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_KHR_get_surface_capabilities2 ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::SurfaceCapabilities2KHR
                           getSurfaceCapabilities2KHR( const PhysicalDeviceSurfaceInfo2KHR & surfaceInfo ) const;

      template <typename X, typename Y, typename... Z>
      VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...>
                           getSurfaceCapabilities2KHR( const PhysicalDeviceSurfaceInfo2KHR & surfaceInfo ) const;

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::SurfaceFormat2KHR>
                           getSurfaceFormats2KHR( const PhysicalDeviceSurfaceInfo2KHR & surfaceInfo ) const;

      //=== VK_KHR_get_display_properties2 ===

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::DisplayProperties2KHR> getDisplayProperties2KHR() const;

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::DisplayPlaneProperties2KHR>
                           getDisplayPlaneProperties2KHR() const;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::DisplayPlaneCapabilities2KHR
                           getDisplayPlaneCapabilities2KHR( const DisplayPlaneInfo2KHR & displayPlaneInfo ) const;

      //=== VK_EXT_sample_locations ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::MultisamplePropertiesEXT
                           getMultisamplePropertiesEXT( VULKAN_HPP_NAMESPACE::SampleCountFlagBits samples ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_EXT_calibrated_timestamps ===

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::TimeDomainEXT> getCalibrateableTimeDomainsEXT() const;

      //=== VK_KHR_fragment_shading_rate ===

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateKHR>
                           getFragmentShadingRatesKHR() const;

      //=== VK_EXT_tooling_info ===

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::PhysicalDeviceToolPropertiesEXT>
                           getToolPropertiesEXT() const;

      //=== VK_NV_cooperative_matrix ===

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::CooperativeMatrixPropertiesNV>
                           getCooperativeMatrixPropertiesNV() const;

      //=== VK_NV_coverage_reduction_mode ===

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::FramebufferMixedSamplesCombinationNV>
                           getSupportedFramebufferMixedSamplesCombinationsNV() const;

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_EXT_full_screen_exclusive ===

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::PresentModeKHR>
                           getSurfacePresentModes2EXT( const PhysicalDeviceSurfaceInfo2KHR & surfaceInfo ) const;
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

      //=== VK_EXT_acquire_drm_display ===

      void acquireDrmDisplayEXT( int32_t drmFd, VULKAN_HPP_NAMESPACE::DisplayKHR display ) const;

#  if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
      //=== VK_EXT_directfb_surface ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::Bool32
                           getDirectFBPresentationSupportEXT( uint32_t queueFamilyIndex, IDirectFB & dfb ) const VULKAN_HPP_NOEXCEPT;
#  endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

#  if defined( VK_USE_PLATFORM_SCREEN_QNX )
      //=== VK_QNX_screen_surface ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::Bool32
                           getScreenPresentationSupportQNX( uint32_t                queueFamilyIndex,
                                                            struct _screen_window & window ) const VULKAN_HPP_NOEXCEPT;
#  endif /*VK_USE_PLATFORM_SCREEN_QNX*/

    private:
      VULKAN_HPP_NAMESPACE::PhysicalDevice                                        m_physicalDevice;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::InstanceDispatcher const * m_dispatcher;
    };

    class PhysicalDevices : public std::vector<VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::PhysicalDevice>
    {
    public:
      PhysicalDevices( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Instance const & instance )
      {
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::InstanceDispatcher const * dispatcher =
          instance.getDispatcher();
        std::vector<VkPhysicalDevice> physicalDevices;
        uint32_t                      physicalDeviceCount;
        VULKAN_HPP_NAMESPACE::Result  result;
        do
        {
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>( dispatcher->vkEnumeratePhysicalDevices(
            static_cast<VkInstance>( *instance ), &physicalDeviceCount, nullptr ) );
          if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && physicalDeviceCount )
          {
            physicalDevices.resize( physicalDeviceCount );
            result = static_cast<VULKAN_HPP_NAMESPACE::Result>( dispatcher->vkEnumeratePhysicalDevices(
              static_cast<VkInstance>( *instance ), &physicalDeviceCount, physicalDevices.data() ) );
            VULKAN_HPP_ASSERT( physicalDeviceCount <= physicalDevices.size() );
          }
        } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
        if ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          this->reserve( physicalDeviceCount );
          for ( auto const & physicalDevice : physicalDevices )
          {
            this->emplace_back( physicalDevice, dispatcher );
          }
        }
        else
        {
          throwResultException( result, "vkEnumeratePhysicalDevices" );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      PhysicalDevices() = default;
#  else
      PhysicalDevices()                                                       = delete;
#  endif
      PhysicalDevices( PhysicalDevices const & ) = delete;
      PhysicalDevices( PhysicalDevices && rhs )  = default;
      PhysicalDevices & operator=( PhysicalDevices const & ) = delete;
      PhysicalDevices & operator=( PhysicalDevices && rhs ) = default;
    };

    class Device
    {
    public:
      using CType = VkDevice;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eDevice;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eDevice;

    public:
      Device( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::PhysicalDevice const &         physicalDevice,
              VULKAN_HPP_NAMESPACE::DeviceCreateInfo const &                                  createInfo,
              VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( physicalDevice.getDispatcher()->vkGetDeviceProcAddr )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          physicalDevice.getDispatcher()->vkCreateDevice( static_cast<VkPhysicalDevice>( *physicalDevice ),
                                                          reinterpret_cast<const VkDeviceCreateInfo *>( &createInfo ),
                                                          m_allocator,
                                                          reinterpret_cast<VkDevice *>( &m_device ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateDevice" );
        }
        m_dispatcher.init( static_cast<VkDevice>( m_device ) );
      }

      Device( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::PhysicalDevice const &         physicalDevice,
              VkDevice                                                                        device,
              VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( physicalDevice.getDispatcher()->vkGetDeviceProcAddr )
      {
        m_dispatcher.init( static_cast<VkDevice>( m_device ) );
      }

      ~Device()
      {
        if ( m_device )
        {
          getDispatcher()->vkDestroyDevice( static_cast<VkDevice>( m_device ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      Device() = default;
#  else
      Device()                                                                = delete;
#  endif
      Device( Device const & ) = delete;
      Device( Device && rhs ) VULKAN_HPP_NOEXCEPT
        : m_device( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_device, {} ) )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      Device & operator=( Device const & ) = delete;
      Device & operator                    =( Device && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_device )
          {
            getDispatcher()->vkDestroyDevice( static_cast<VkDevice>( m_device ), m_allocator );
          }
          m_device     = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_device, {} );
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::Device const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_device;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher.getVkHeaderVersion() == VK_HEADER_VERSION );
        return &m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_device.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_device.operator!();
      }
#  endif

      //=== VK_VERSION_1_0 ===

      VULKAN_HPP_NODISCARD PFN_vkVoidFunction getProcAddr( const std::string & name ) const VULKAN_HPP_NOEXCEPT;

      void waitIdle() const;

      void
        flushMappedMemoryRanges( ArrayProxy<const VULKAN_HPP_NAMESPACE::MappedMemoryRange> const & memoryRanges ) const;

      void invalidateMappedMemoryRanges(
        ArrayProxy<const VULKAN_HPP_NAMESPACE::MappedMemoryRange> const & memoryRanges ) const;

      void resetFences( ArrayProxy<const VULKAN_HPP_NAMESPACE::Fence> const & fences ) const;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::Result
                           waitForFences( ArrayProxy<const VULKAN_HPP_NAMESPACE::Fence> const & fences,
                                          VULKAN_HPP_NAMESPACE::Bool32                          waitAll,
                                          uint64_t                                              timeout ) const;

      void updateDescriptorSets(
        ArrayProxy<const VULKAN_HPP_NAMESPACE::WriteDescriptorSet> const & descriptorWrites,
        ArrayProxy<const VULKAN_HPP_NAMESPACE::CopyDescriptorSet> const &  descriptorCopies ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_VERSION_1_1 ===

      void bindBufferMemory2( ArrayProxy<const VULKAN_HPP_NAMESPACE::BindBufferMemoryInfo> const & bindInfos ) const;

      void bindImageMemory2( ArrayProxy<const VULKAN_HPP_NAMESPACE::BindImageMemoryInfo> const & bindInfos ) const;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::PeerMemoryFeatureFlags getGroupPeerMemoryFeatures(
        uint32_t heapIndex, uint32_t localDeviceIndex, uint32_t remoteDeviceIndex ) const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::MemoryRequirements2
                           getImageMemoryRequirements2( const ImageMemoryRequirementsInfo2 & info ) const VULKAN_HPP_NOEXCEPT;

      template <typename X, typename Y, typename... Z>
      VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...>
                           getImageMemoryRequirements2( const ImageMemoryRequirementsInfo2 & info ) const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::MemoryRequirements2
                           getBufferMemoryRequirements2( const BufferMemoryRequirementsInfo2 & info ) const VULKAN_HPP_NOEXCEPT;

      template <typename X, typename Y, typename... Z>
      VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...>
                           getBufferMemoryRequirements2( const BufferMemoryRequirementsInfo2 & info ) const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::SparseImageMemoryRequirements2>
                           getImageSparseMemoryRequirements2( const ImageSparseMemoryRequirementsInfo2 & info ) const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::DescriptorSetLayoutSupport
                           getDescriptorSetLayoutSupport( const DescriptorSetLayoutCreateInfo & createInfo ) const VULKAN_HPP_NOEXCEPT;

      template <typename X, typename Y, typename... Z>
      VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...>
                           getDescriptorSetLayoutSupport( const DescriptorSetLayoutCreateInfo & createInfo ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_VERSION_1_2 ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::Result waitSemaphores( const SemaphoreWaitInfo & waitInfo,
                                                                        uint64_t                  timeout ) const;

      void signalSemaphore( const SemaphoreSignalInfo & signalInfo ) const;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::DeviceAddress
                           getBufferAddress( const BufferDeviceAddressInfo & info ) const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD uint64_t
                           getBufferOpaqueCaptureAddress( const BufferDeviceAddressInfo & info ) const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD uint64_t
                           getMemoryOpaqueCaptureAddress( const DeviceMemoryOpaqueCaptureAddressInfo & info ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_KHR_swapchain ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::DeviceGroupPresentCapabilitiesKHR
                           getGroupPresentCapabilitiesKHR() const;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::DeviceGroupPresentModeFlagsKHR
                           getGroupSurfacePresentModesKHR( VULKAN_HPP_NAMESPACE::SurfaceKHR surface ) const;

      VULKAN_HPP_NODISCARD std::pair<VULKAN_HPP_NAMESPACE::Result, uint32_t>
                           acquireNextImage2KHR( const AcquireNextImageInfoKHR & acquireInfo ) const;

      //=== VK_EXT_debug_marker ===

      void debugMarkerSetObjectTagEXT( const DebugMarkerObjectTagInfoEXT & tagInfo ) const;

      void debugMarkerSetObjectNameEXT( const DebugMarkerObjectNameInfoEXT & nameInfo ) const;

      //=== VK_NVX_image_view_handle ===

      VULKAN_HPP_NODISCARD uint32_t
                           getImageViewHandleNVX( const ImageViewHandleInfoNVX & info ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_KHR_device_group ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::PeerMemoryFeatureFlags getGroupPeerMemoryFeaturesKHR(
        uint32_t heapIndex, uint32_t localDeviceIndex, uint32_t remoteDeviceIndex ) const VULKAN_HPP_NOEXCEPT;

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_KHR_external_memory_win32 ===

      VULKAN_HPP_NODISCARD HANDLE
                           getMemoryWin32HandleKHR( const MemoryGetWin32HandleInfoKHR & getWin32HandleInfo ) const;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::MemoryWin32HandlePropertiesKHR
                           getMemoryWin32HandlePropertiesKHR( VULKAN_HPP_NAMESPACE::ExternalMemoryHandleTypeFlagBits handleType,
                                                              HANDLE                                                 handle ) const;
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

      //=== VK_KHR_external_memory_fd ===

      VULKAN_HPP_NODISCARD int getMemoryFdKHR( const MemoryGetFdInfoKHR & getFdInfo ) const;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::MemoryFdPropertiesKHR
                           getMemoryFdPropertiesKHR( VULKAN_HPP_NAMESPACE::ExternalMemoryHandleTypeFlagBits handleType, int fd ) const;

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_KHR_external_semaphore_win32 ===

      void
        importSemaphoreWin32HandleKHR( const ImportSemaphoreWin32HandleInfoKHR & importSemaphoreWin32HandleInfo ) const;

      VULKAN_HPP_NODISCARD HANDLE
                           getSemaphoreWin32HandleKHR( const SemaphoreGetWin32HandleInfoKHR & getWin32HandleInfo ) const;
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

      //=== VK_KHR_external_semaphore_fd ===

      void importSemaphoreFdKHR( const ImportSemaphoreFdInfoKHR & importSemaphoreFdInfo ) const;

      VULKAN_HPP_NODISCARD int getSemaphoreFdKHR( const SemaphoreGetFdInfoKHR & getFdInfo ) const;

      //=== VK_KHR_descriptor_update_template ===

      void destroyDescriptorUpdateTemplateKHR(
        VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplate descriptorUpdateTemplate VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT,
        Optional<const AllocationCallbacks>                                     allocator
                                                                                VULKAN_HPP_DEFAULT_ARGUMENT_NULLPTR_ASSIGNMENT ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_EXT_display_control ===

      void displayPowerControlEXT( VULKAN_HPP_NAMESPACE::DisplayKHR display,
                                   const DisplayPowerInfoEXT &      displayPowerInfo ) const;

      //=== VK_EXT_hdr_metadata ===

      void setHdrMetadataEXT( ArrayProxy<const VULKAN_HPP_NAMESPACE::SwapchainKHR> const &   swapchains,
                              ArrayProxy<const VULKAN_HPP_NAMESPACE::HdrMetadataEXT> const & metadata ) const
        VULKAN_HPP_NOEXCEPT_WHEN_NO_EXCEPTIONS;

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_KHR_external_fence_win32 ===

      void importFenceWin32HandleKHR( const ImportFenceWin32HandleInfoKHR & importFenceWin32HandleInfo ) const;

      VULKAN_HPP_NODISCARD HANDLE getFenceWin32HandleKHR( const FenceGetWin32HandleInfoKHR & getWin32HandleInfo ) const;
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

      //=== VK_KHR_external_fence_fd ===

      void importFenceFdKHR( const ImportFenceFdInfoKHR & importFenceFdInfo ) const;

      VULKAN_HPP_NODISCARD int getFenceFdKHR( const FenceGetFdInfoKHR & getFdInfo ) const;

      //=== VK_KHR_performance_query ===

      void acquireProfilingLockKHR( const AcquireProfilingLockInfoKHR & info ) const;

      void releaseProfilingLockKHR() const VULKAN_HPP_NOEXCEPT;

      //=== VK_EXT_debug_utils ===

      void setDebugUtilsObjectNameEXT( const DebugUtilsObjectNameInfoEXT & nameInfo ) const;

      void setDebugUtilsObjectTagEXT( const DebugUtilsObjectTagInfoEXT & tagInfo ) const;

#  if defined( VK_USE_PLATFORM_ANDROID_KHR )
      //=== VK_ANDROID_external_memory_android_hardware_buffer ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::AndroidHardwareBufferPropertiesANDROID
                           getAndroidHardwareBufferPropertiesANDROID( const struct AHardwareBuffer & buffer ) const;

      template <typename X, typename Y, typename... Z>
      VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...>
                           getAndroidHardwareBufferPropertiesANDROID( const struct AHardwareBuffer & buffer ) const;

      VULKAN_HPP_NODISCARD struct AHardwareBuffer *
        getMemoryAndroidHardwareBufferANDROID( const MemoryGetAndroidHardwareBufferInfoANDROID & info ) const;
#  endif /*VK_USE_PLATFORM_ANDROID_KHR*/

      //=== VK_KHR_get_memory_requirements2 ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::MemoryRequirements2
                           getImageMemoryRequirements2KHR( const ImageMemoryRequirementsInfo2 & info ) const VULKAN_HPP_NOEXCEPT;

      template <typename X, typename Y, typename... Z>
      VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...>
                           getImageMemoryRequirements2KHR( const ImageMemoryRequirementsInfo2 & info ) const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::MemoryRequirements2
                           getBufferMemoryRequirements2KHR( const BufferMemoryRequirementsInfo2 & info ) const VULKAN_HPP_NOEXCEPT;

      template <typename X, typename Y, typename... Z>
      VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...>
                           getBufferMemoryRequirements2KHR( const BufferMemoryRequirementsInfo2 & info ) const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::SparseImageMemoryRequirements2>
                           getImageSparseMemoryRequirements2KHR( const ImageSparseMemoryRequirementsInfo2 & info ) const
        VULKAN_HPP_NOEXCEPT;

      //=== VK_KHR_acceleration_structure ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::Result buildAccelerationStructuresKHR(
        VULKAN_HPP_NAMESPACE::DeferredOperationKHR                                                deferredOperation,
        ArrayProxy<const VULKAN_HPP_NAMESPACE::AccelerationStructureBuildGeometryInfoKHR> const & infos,
        ArrayProxy<const VULKAN_HPP_NAMESPACE::AccelerationStructureBuildRangeInfoKHR * const> const &
          pBuildRangeInfos ) const;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::Result
                           copyAccelerationStructureKHR( VULKAN_HPP_NAMESPACE::DeferredOperationKHR deferredOperation,
                                                         const CopyAccelerationStructureInfoKHR &   info ) const;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::Result
                           copyAccelerationStructureToMemoryKHR( VULKAN_HPP_NAMESPACE::DeferredOperationKHR       deferredOperation,
                                                                 const CopyAccelerationStructureToMemoryInfoKHR & info ) const;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::Result
                           copyMemoryToAccelerationStructureKHR( VULKAN_HPP_NAMESPACE::DeferredOperationKHR       deferredOperation,
                                                                 const CopyMemoryToAccelerationStructureInfoKHR & info ) const;

      template <typename T>
      VULKAN_HPP_NODISCARD std::vector<T> writeAccelerationStructuresPropertiesKHR(
        ArrayProxy<const VULKAN_HPP_NAMESPACE::AccelerationStructureKHR> const & accelerationStructures,
        VULKAN_HPP_NAMESPACE::QueryType                                          queryType,
        size_t                                                                   dataSize,
        size_t                                                                   stride ) const;

      template <typename T>
      VULKAN_HPP_NODISCARD T writeAccelerationStructuresPropertyKHR(
        ArrayProxy<const VULKAN_HPP_NAMESPACE::AccelerationStructureKHR> const & accelerationStructures,
        VULKAN_HPP_NAMESPACE::QueryType                                          queryType,
        size_t                                                                   stride ) const;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::DeviceAddress getAccelerationStructureAddressKHR(
        const AccelerationStructureDeviceAddressInfoKHR & info ) const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::AccelerationStructureCompatibilityKHR
                           getAccelerationStructureCompatibilityKHR( const AccelerationStructureVersionInfoKHR & versionInfo ) const
        VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::AccelerationStructureBuildSizesInfoKHR
                           getAccelerationStructureBuildSizesKHR( VULKAN_HPP_NAMESPACE::AccelerationStructureBuildTypeKHR buildType,
                                                                  const AccelerationStructureBuildGeometryInfoKHR &       buildInfo,
                                                                  ArrayProxy<const uint32_t> const & maxPrimitiveCounts
                                                                                                     VULKAN_HPP_DEFAULT_ARGUMENT_NULLPTR_ASSIGNMENT ) const
        VULKAN_HPP_NOEXCEPT;

      //=== VK_KHR_sampler_ycbcr_conversion ===

      void destroySamplerYcbcrConversionKHR(
        VULKAN_HPP_NAMESPACE::SamplerYcbcrConversion ycbcrConversion VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT,
        Optional<const AllocationCallbacks>                          allocator
                                                                     VULKAN_HPP_DEFAULT_ARGUMENT_NULLPTR_ASSIGNMENT ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_KHR_bind_memory2 ===

      void bindBufferMemory2KHR( ArrayProxy<const VULKAN_HPP_NAMESPACE::BindBufferMemoryInfo> const & bindInfos ) const;

      void bindImageMemory2KHR( ArrayProxy<const VULKAN_HPP_NAMESPACE::BindImageMemoryInfo> const & bindInfos ) const;

      //=== VK_NV_ray_tracing ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::MemoryRequirements2KHR getAccelerationStructureMemoryRequirementsNV(
        const AccelerationStructureMemoryRequirementsInfoNV & info ) const VULKAN_HPP_NOEXCEPT;

      template <typename X, typename Y, typename... Z>
      VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...> getAccelerationStructureMemoryRequirementsNV(
        const AccelerationStructureMemoryRequirementsInfoNV & info ) const VULKAN_HPP_NOEXCEPT;

      void bindAccelerationStructureMemoryNV(
        ArrayProxy<const VULKAN_HPP_NAMESPACE::BindAccelerationStructureMemoryInfoNV> const & bindInfos ) const;

      //=== VK_KHR_maintenance3 ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::DescriptorSetLayoutSupport
                           getDescriptorSetLayoutSupportKHR( const DescriptorSetLayoutCreateInfo & createInfo ) const VULKAN_HPP_NOEXCEPT;

      template <typename X, typename Y, typename... Z>
      VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...>
                           getDescriptorSetLayoutSupportKHR( const DescriptorSetLayoutCreateInfo & createInfo ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_EXT_external_memory_host ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::MemoryHostPointerPropertiesEXT
                           getMemoryHostPointerPropertiesEXT( VULKAN_HPP_NAMESPACE::ExternalMemoryHandleTypeFlagBits handleType,
                                                              const void *                                           pHostPointer ) const;

      //=== VK_EXT_calibrated_timestamps ===

      VULKAN_HPP_NODISCARD std::pair<std::vector<uint64_t>, uint64_t> getCalibratedTimestampsEXT(
        ArrayProxy<const VULKAN_HPP_NAMESPACE::CalibratedTimestampInfoEXT> const & timestampInfos ) const;

      //=== VK_KHR_timeline_semaphore ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::Result waitSemaphoresKHR( const SemaphoreWaitInfo & waitInfo,
                                                                           uint64_t                  timeout ) const;

      void signalSemaphoreKHR( const SemaphoreSignalInfo & signalInfo ) const;

      //=== VK_INTEL_performance_query ===

      void initializePerformanceApiINTEL( const InitializePerformanceApiInfoINTEL & initializeInfo ) const;

      void uninitializePerformanceApiINTEL() const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::PerformanceValueINTEL
                           getPerformanceParameterINTEL( VULKAN_HPP_NAMESPACE::PerformanceParameterTypeINTEL parameter ) const;

      //=== VK_EXT_buffer_device_address ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::DeviceAddress
                           getBufferAddressEXT( const BufferDeviceAddressInfo & info ) const VULKAN_HPP_NOEXCEPT;

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_EXT_full_screen_exclusive ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::DeviceGroupPresentModeFlagsKHR
                           getGroupSurfacePresentModes2EXT( const PhysicalDeviceSurfaceInfo2KHR & surfaceInfo ) const;
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

      //=== VK_KHR_buffer_device_address ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::DeviceAddress
                           getBufferAddressKHR( const BufferDeviceAddressInfo & info ) const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD uint64_t
                           getBufferOpaqueCaptureAddressKHR( const BufferDeviceAddressInfo & info ) const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD uint64_t
                           getMemoryOpaqueCaptureAddressKHR( const DeviceMemoryOpaqueCaptureAddressInfo & info ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_KHR_pipeline_executable_properties ===

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::PipelineExecutablePropertiesKHR>
                           getPipelineExecutablePropertiesKHR( const PipelineInfoKHR & pipelineInfo ) const;

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::PipelineExecutableStatisticKHR>
                           getPipelineExecutableStatisticsKHR( const PipelineExecutableInfoKHR & executableInfo ) const;

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::PipelineExecutableInternalRepresentationKHR>
                           getPipelineExecutableInternalRepresentationsKHR( const PipelineExecutableInfoKHR & executableInfo ) const;

      //=== VK_NV_device_generated_commands ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::MemoryRequirements2 getGeneratedCommandsMemoryRequirementsNV(
        const GeneratedCommandsMemoryRequirementsInfoNV & info ) const VULKAN_HPP_NOEXCEPT;

      template <typename X, typename Y, typename... Z>
      VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...> getGeneratedCommandsMemoryRequirementsNV(
        const GeneratedCommandsMemoryRequirementsInfoNV & info ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_EXT_private_data ===

      void setPrivateDataEXT( VULKAN_HPP_NAMESPACE::ObjectType         objectType_,
                              uint64_t                                 objectHandle,
                              VULKAN_HPP_NAMESPACE::PrivateDataSlotEXT privateDataSlot,
                              uint64_t                                 data ) const;

      VULKAN_HPP_NODISCARD uint64_t
                           getPrivateDataEXT( VULKAN_HPP_NAMESPACE::ObjectType         objectType_,
                                              uint64_t                                 objectHandle,
                                              VULKAN_HPP_NAMESPACE::PrivateDataSlotEXT privateDataSlot ) const VULKAN_HPP_NOEXCEPT;

#  if defined( VK_USE_PLATFORM_FUCHSIA )
      //=== VK_FUCHSIA_external_memory ===

      VULKAN_HPP_NODISCARD zx_handle_t
                           getMemoryZirconHandleFUCHSIA( const MemoryGetZirconHandleInfoFUCHSIA & getZirconHandleInfo ) const;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::MemoryZirconHandlePropertiesFUCHSIA
                           getMemoryZirconHandlePropertiesFUCHSIA( VULKAN_HPP_NAMESPACE::ExternalMemoryHandleTypeFlagBits handleType,
                                                                   zx_handle_t zirconHandle ) const;
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
      //=== VK_FUCHSIA_external_semaphore ===

      void importSemaphoreZirconHandleFUCHSIA(
        const ImportSemaphoreZirconHandleInfoFUCHSIA & importSemaphoreZirconHandleInfo ) const;

      VULKAN_HPP_NODISCARD zx_handle_t
                           getSemaphoreZirconHandleFUCHSIA( const SemaphoreGetZirconHandleInfoFUCHSIA & getZirconHandleInfo ) const;
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

      //=== VK_NV_external_memory_rdma ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::RemoteAddressNV
                           getMemoryRemoteAddressNV( const MemoryGetRemoteAddressInfoNV & memoryGetRemoteAddressInfo ) const;

    private:
      VULKAN_HPP_NAMESPACE::Device                                      m_device;
      const VkAllocationCallbacks *                                     m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher m_dispatcher;
    };

    class AccelerationStructureKHR
    {
    public:
      using CType = VkAccelerationStructureKHR;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eAccelerationStructureKHR;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eAccelerationStructureKHR;

    public:
      AccelerationStructureKHR(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VULKAN_HPP_NAMESPACE::AccelerationStructureCreateInfoKHR const &                createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateAccelerationStructureKHR(
            static_cast<VkDevice>( *device ),
            reinterpret_cast<const VkAccelerationStructureCreateInfoKHR *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkAccelerationStructureKHR *>( &m_accelerationStructureKHR ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateAccelerationStructureKHR" );
        }
      }

      AccelerationStructureKHR(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VkAccelerationStructureKHR                                                      accelerationStructureKHR,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_accelerationStructureKHR( accelerationStructureKHR )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~AccelerationStructureKHR()
      {
        if ( m_accelerationStructureKHR )
        {
          getDispatcher()->vkDestroyAccelerationStructureKHR(
            m_device, static_cast<VkAccelerationStructureKHR>( m_accelerationStructureKHR ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      AccelerationStructureKHR() = default;
#  else
      AccelerationStructureKHR()                                              = delete;
#  endif
      AccelerationStructureKHR( AccelerationStructureKHR const & ) = delete;
      AccelerationStructureKHR( AccelerationStructureKHR && rhs ) VULKAN_HPP_NOEXCEPT
        : m_accelerationStructureKHR(
            VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_accelerationStructureKHR, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      AccelerationStructureKHR & operator=( AccelerationStructureKHR const & ) = delete;
      AccelerationStructureKHR & operator=( AccelerationStructureKHR && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_accelerationStructureKHR )
          {
            getDispatcher()->vkDestroyAccelerationStructureKHR(
              m_device, static_cast<VkAccelerationStructureKHR>( m_accelerationStructureKHR ), m_allocator );
          }
          m_accelerationStructureKHR =
            VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_accelerationStructureKHR, {} );
          m_device     = rhs.m_device;
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::AccelerationStructureKHR const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_accelerationStructureKHR;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_accelerationStructureKHR.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_accelerationStructureKHR.operator!();
      }
#  endif

    private:
      VULKAN_HPP_NAMESPACE::AccelerationStructureKHR                            m_accelerationStructureKHR;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class AccelerationStructureNV
    {
    public:
      using CType = VkAccelerationStructureNV;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eAccelerationStructureNV;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eAccelerationStructureNV;

    public:
      AccelerationStructureNV(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VULKAN_HPP_NAMESPACE::AccelerationStructureCreateInfoNV const &                 createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateAccelerationStructureNV(
            static_cast<VkDevice>( *device ),
            reinterpret_cast<const VkAccelerationStructureCreateInfoNV *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkAccelerationStructureNV *>( &m_accelerationStructureNV ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateAccelerationStructureNV" );
        }
      }

      AccelerationStructureNV(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VkAccelerationStructureNV                                                       accelerationStructureNV,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_accelerationStructureNV( accelerationStructureNV )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~AccelerationStructureNV()
      {
        if ( m_accelerationStructureNV )
        {
          getDispatcher()->vkDestroyAccelerationStructureNV(
            m_device, static_cast<VkAccelerationStructureNV>( m_accelerationStructureNV ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      AccelerationStructureNV() = default;
#  else
      AccelerationStructureNV()                                               = delete;
#  endif
      AccelerationStructureNV( AccelerationStructureNV const & ) = delete;
      AccelerationStructureNV( AccelerationStructureNV && rhs ) VULKAN_HPP_NOEXCEPT
        : m_accelerationStructureNV(
            VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_accelerationStructureNV, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      AccelerationStructureNV & operator=( AccelerationStructureNV const & ) = delete;
      AccelerationStructureNV & operator=( AccelerationStructureNV && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_accelerationStructureNV )
          {
            getDispatcher()->vkDestroyAccelerationStructureNV(
              m_device, static_cast<VkAccelerationStructureNV>( m_accelerationStructureNV ), m_allocator );
          }
          m_accelerationStructureNV =
            VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_accelerationStructureNV, {} );
          m_device     = rhs.m_device;
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::AccelerationStructureNV const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_accelerationStructureNV;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_accelerationStructureNV.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_accelerationStructureNV.operator!();
      }
#  endif

      //=== VK_NV_ray_tracing ===

      template <typename T>
      VULKAN_HPP_NODISCARD std::vector<T> getHandle( size_t dataSize ) const;

      template <typename T>
      VULKAN_HPP_NODISCARD T getHandle() const;

    private:
      VULKAN_HPP_NAMESPACE::AccelerationStructureNV                             m_accelerationStructureNV;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class Buffer
    {
    public:
      using CType = VkBuffer;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eBuffer;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eBuffer;

    public:
      Buffer( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
              VULKAN_HPP_NAMESPACE::BufferCreateInfo const &                                  createInfo,
              VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkCreateBuffer( static_cast<VkDevice>( *device ),
                                           reinterpret_cast<const VkBufferCreateInfo *>( &createInfo ),
                                           m_allocator,
                                           reinterpret_cast<VkBuffer *>( &m_buffer ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateBuffer" );
        }
      }

      Buffer( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
              VkBuffer                                                                        buffer,
              VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_buffer( buffer )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~Buffer()
      {
        if ( m_buffer )
        {
          getDispatcher()->vkDestroyBuffer( m_device, static_cast<VkBuffer>( m_buffer ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      Buffer() = default;
#  else
      Buffer()                                                                = delete;
#  endif
      Buffer( Buffer const & ) = delete;
      Buffer( Buffer && rhs ) VULKAN_HPP_NOEXCEPT
        : m_buffer( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_buffer, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      Buffer & operator=( Buffer const & ) = delete;
      Buffer & operator                    =( Buffer && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_buffer )
          {
            getDispatcher()->vkDestroyBuffer( m_device, static_cast<VkBuffer>( m_buffer ), m_allocator );
          }
          m_buffer     = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_buffer, {} );
          m_device     = rhs.m_device;
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::Buffer const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_buffer;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_buffer.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_buffer.operator!();
      }
#  endif

      //=== VK_VERSION_1_0 ===

      void bindMemory( VULKAN_HPP_NAMESPACE::DeviceMemory memory, VULKAN_HPP_NAMESPACE::DeviceSize memoryOffset ) const;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::MemoryRequirements getMemoryRequirements() const VULKAN_HPP_NOEXCEPT;

    private:
      VULKAN_HPP_NAMESPACE::Buffer                                              m_buffer;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class BufferView
    {
    public:
      using CType = VkBufferView;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eBufferView;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eBufferView;

    public:
      BufferView( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
                  VULKAN_HPP_NAMESPACE::BufferViewCreateInfo const &                              createInfo,
                  VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkCreateBufferView( static_cast<VkDevice>( *device ),
                                               reinterpret_cast<const VkBufferViewCreateInfo *>( &createInfo ),
                                               m_allocator,
                                               reinterpret_cast<VkBufferView *>( &m_bufferView ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateBufferView" );
        }
      }

      BufferView( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
                  VkBufferView                                                                    bufferView,
                  VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_bufferView( bufferView )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~BufferView()
      {
        if ( m_bufferView )
        {
          getDispatcher()->vkDestroyBufferView( m_device, static_cast<VkBufferView>( m_bufferView ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      BufferView() = default;
#  else
      BufferView()                                                            = delete;
#  endif
      BufferView( BufferView const & ) = delete;
      BufferView( BufferView && rhs ) VULKAN_HPP_NOEXCEPT
        : m_bufferView( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_bufferView, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      BufferView & operator=( BufferView const & ) = delete;
      BufferView & operator                        =( BufferView && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_bufferView )
          {
            getDispatcher()->vkDestroyBufferView( m_device, static_cast<VkBufferView>( m_bufferView ), m_allocator );
          }
          m_bufferView = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_bufferView, {} );
          m_device     = rhs.m_device;
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::BufferView const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_bufferView;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_bufferView.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_bufferView.operator!();
      }
#  endif

    private:
      VULKAN_HPP_NAMESPACE::BufferView                                          m_bufferView;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class CommandPool
    {
    public:
      using CType = VkCommandPool;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eCommandPool;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eCommandPool;

    public:
      CommandPool( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
                   VULKAN_HPP_NAMESPACE::CommandPoolCreateInfo const &                             createInfo,
                   VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkCreateCommandPool( static_cast<VkDevice>( *device ),
                                                reinterpret_cast<const VkCommandPoolCreateInfo *>( &createInfo ),
                                                m_allocator,
                                                reinterpret_cast<VkCommandPool *>( &m_commandPool ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateCommandPool" );
        }
      }

      CommandPool( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
                   VkCommandPool                                                                   commandPool,
                   VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_commandPool( commandPool )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~CommandPool()
      {
        if ( m_commandPool )
        {
          getDispatcher()->vkDestroyCommandPool( m_device, static_cast<VkCommandPool>( m_commandPool ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      CommandPool() = default;
#  else
      CommandPool()                                                           = delete;
#  endif
      CommandPool( CommandPool const & ) = delete;
      CommandPool( CommandPool && rhs ) VULKAN_HPP_NOEXCEPT
        : m_commandPool( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_commandPool, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      CommandPool & operator=( CommandPool const & ) = delete;
      CommandPool & operator                         =( CommandPool && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_commandPool )
          {
            getDispatcher()->vkDestroyCommandPool( m_device, static_cast<VkCommandPool>( m_commandPool ), m_allocator );
          }
          m_commandPool = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_commandPool, {} );
          m_device      = rhs.m_device;
          m_allocator   = rhs.m_allocator;
          m_dispatcher  = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::CommandPool const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_commandPool;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_commandPool.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_commandPool.operator!();
      }
#  endif

      //=== VK_VERSION_1_0 ===

      void reset( VULKAN_HPP_NAMESPACE::CommandPoolResetFlags flags VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT ) const;

      //=== VK_VERSION_1_1 ===

      void trim( VULKAN_HPP_NAMESPACE::CommandPoolTrimFlags flags VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT ) const
        VULKAN_HPP_NOEXCEPT;

      //=== VK_KHR_maintenance1 ===

      void trimKHR( VULKAN_HPP_NAMESPACE::CommandPoolTrimFlags flags VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT ) const
        VULKAN_HPP_NOEXCEPT;

    private:
      VULKAN_HPP_NAMESPACE::CommandPool                                         m_commandPool;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class CommandBuffer
    {
    public:
      using CType = VkCommandBuffer;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eCommandBuffer;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eCommandBuffer;

    public:
      CommandBuffer( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &      device,
                     VkCommandBuffer                                                      commandBuffer,
                     VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::CommandPool const & commandPool )
        : m_commandBuffer( commandBuffer )
        , m_device( *device )
        , m_commandPool( *commandPool )
        , m_dispatcher( device.getDispatcher() )
      {}

      CommandBuffer( VkCommandBuffer                                                           commandBuffer,
                     VkDevice                                                                  device,
                     VkCommandPool                                                             commandPool,
                     VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * dispatcher )
        : m_commandBuffer( commandBuffer ), m_device( device ), m_commandPool( commandPool ), m_dispatcher( dispatcher )
      {}

      ~CommandBuffer()
      {
        if ( m_commandBuffer )
        {
          getDispatcher()->vkFreeCommandBuffers(
            m_device, m_commandPool, 1, reinterpret_cast<VkCommandBuffer const *>( &m_commandBuffer ) );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      CommandBuffer() = default;
#  else
      CommandBuffer()                                                         = delete;
#  endif
      CommandBuffer( CommandBuffer const & ) = delete;
      CommandBuffer( CommandBuffer && rhs ) VULKAN_HPP_NOEXCEPT
        : m_commandBuffer( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_commandBuffer, {} ) )
        , m_device( rhs.m_device )
        , m_commandPool( rhs.m_commandPool )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      CommandBuffer & operator=( CommandBuffer const & ) = delete;
      CommandBuffer & operator                           =( CommandBuffer && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_commandBuffer )
          {
            getDispatcher()->vkFreeCommandBuffers(
              m_device, m_commandPool, 1, reinterpret_cast<VkCommandBuffer const *>( &m_commandBuffer ) );
          }
          m_commandBuffer = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_commandBuffer, {} );
          m_device        = rhs.m_device;
          m_commandPool   = rhs.m_commandPool;
          m_dispatcher    = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::CommandBuffer const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_commandBuffer;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_commandBuffer.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_commandBuffer.operator!();
      }
#  endif

      //=== VK_VERSION_1_0 ===

      void begin( const CommandBufferBeginInfo & beginInfo ) const;

      void end() const;

      void reset( VULKAN_HPP_NAMESPACE::CommandBufferResetFlags flags VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT ) const;

      void bindPipeline( VULKAN_HPP_NAMESPACE::PipelineBindPoint pipelineBindPoint,
                         VULKAN_HPP_NAMESPACE::Pipeline          pipeline ) const VULKAN_HPP_NOEXCEPT;

      void setViewport( uint32_t                                                 firstViewport,
                        ArrayProxy<const VULKAN_HPP_NAMESPACE::Viewport> const & viewports ) const VULKAN_HPP_NOEXCEPT;

      void setScissor( uint32_t                                               firstScissor,
                       ArrayProxy<const VULKAN_HPP_NAMESPACE::Rect2D> const & scissors ) const VULKAN_HPP_NOEXCEPT;

      void setLineWidth( float lineWidth ) const VULKAN_HPP_NOEXCEPT;

      void setDepthBias( float depthBiasConstantFactor,
                         float depthBiasClamp,
                         float depthBiasSlopeFactor ) const VULKAN_HPP_NOEXCEPT;

      void setBlendConstants( const float blendConstants[4] ) const VULKAN_HPP_NOEXCEPT;

      void setDepthBounds( float minDepthBounds, float maxDepthBounds ) const VULKAN_HPP_NOEXCEPT;

      void setStencilCompareMask( VULKAN_HPP_NAMESPACE::StencilFaceFlags faceMask,
                                  uint32_t                               compareMask ) const VULKAN_HPP_NOEXCEPT;

      void setStencilWriteMask( VULKAN_HPP_NAMESPACE::StencilFaceFlags faceMask,
                                uint32_t                               writeMask ) const VULKAN_HPP_NOEXCEPT;

      void setStencilReference( VULKAN_HPP_NAMESPACE::StencilFaceFlags faceMask,
                                uint32_t                               reference ) const VULKAN_HPP_NOEXCEPT;

      void bindDescriptorSets( VULKAN_HPP_NAMESPACE::PipelineBindPoint                       pipelineBindPoint,
                               VULKAN_HPP_NAMESPACE::PipelineLayout                          layout,
                               uint32_t                                                      firstSet,
                               ArrayProxy<const VULKAN_HPP_NAMESPACE::DescriptorSet> const & descriptorSets,
                               ArrayProxy<const uint32_t> const & dynamicOffsets ) const VULKAN_HPP_NOEXCEPT;

      void bindIndexBuffer( VULKAN_HPP_NAMESPACE::Buffer     buffer,
                            VULKAN_HPP_NAMESPACE::DeviceSize offset,
                            VULKAN_HPP_NAMESPACE::IndexType  indexType ) const VULKAN_HPP_NOEXCEPT;

      void bindVertexBuffers( uint32_t                                                   firstBinding,
                              ArrayProxy<const VULKAN_HPP_NAMESPACE::Buffer> const &     buffers,
                              ArrayProxy<const VULKAN_HPP_NAMESPACE::DeviceSize> const & offsets ) const
        VULKAN_HPP_NOEXCEPT_WHEN_NO_EXCEPTIONS;

      void draw( uint32_t vertexCount,
                 uint32_t instanceCount,
                 uint32_t firstVertex,
                 uint32_t firstInstance ) const VULKAN_HPP_NOEXCEPT;

      void drawIndexed( uint32_t indexCount,
                        uint32_t instanceCount,
                        uint32_t firstIndex,
                        int32_t  vertexOffset,
                        uint32_t firstInstance ) const VULKAN_HPP_NOEXCEPT;

      void drawIndirect( VULKAN_HPP_NAMESPACE::Buffer     buffer,
                         VULKAN_HPP_NAMESPACE::DeviceSize offset,
                         uint32_t                         drawCount,
                         uint32_t                         stride ) const VULKAN_HPP_NOEXCEPT;

      void drawIndexedIndirect( VULKAN_HPP_NAMESPACE::Buffer     buffer,
                                VULKAN_HPP_NAMESPACE::DeviceSize offset,
                                uint32_t                         drawCount,
                                uint32_t                         stride ) const VULKAN_HPP_NOEXCEPT;

      void dispatch( uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ ) const VULKAN_HPP_NOEXCEPT;

      void dispatchIndirect( VULKAN_HPP_NAMESPACE::Buffer     buffer,
                             VULKAN_HPP_NAMESPACE::DeviceSize offset ) const VULKAN_HPP_NOEXCEPT;

      void copyBuffer( VULKAN_HPP_NAMESPACE::Buffer                               srcBuffer,
                       VULKAN_HPP_NAMESPACE::Buffer                               dstBuffer,
                       ArrayProxy<const VULKAN_HPP_NAMESPACE::BufferCopy> const & regions ) const VULKAN_HPP_NOEXCEPT;

      void copyImage( VULKAN_HPP_NAMESPACE::Image                               srcImage,
                      VULKAN_HPP_NAMESPACE::ImageLayout                         srcImageLayout,
                      VULKAN_HPP_NAMESPACE::Image                               dstImage,
                      VULKAN_HPP_NAMESPACE::ImageLayout                         dstImageLayout,
                      ArrayProxy<const VULKAN_HPP_NAMESPACE::ImageCopy> const & regions ) const VULKAN_HPP_NOEXCEPT;

      void blitImage( VULKAN_HPP_NAMESPACE::Image                               srcImage,
                      VULKAN_HPP_NAMESPACE::ImageLayout                         srcImageLayout,
                      VULKAN_HPP_NAMESPACE::Image                               dstImage,
                      VULKAN_HPP_NAMESPACE::ImageLayout                         dstImageLayout,
                      ArrayProxy<const VULKAN_HPP_NAMESPACE::ImageBlit> const & regions,
                      VULKAN_HPP_NAMESPACE::Filter                              filter ) const VULKAN_HPP_NOEXCEPT;

      void copyBufferToImage( VULKAN_HPP_NAMESPACE::Buffer                                    srcBuffer,
                              VULKAN_HPP_NAMESPACE::Image                                     dstImage,
                              VULKAN_HPP_NAMESPACE::ImageLayout                               dstImageLayout,
                              ArrayProxy<const VULKAN_HPP_NAMESPACE::BufferImageCopy> const & regions ) const
        VULKAN_HPP_NOEXCEPT;

      void copyImageToBuffer( VULKAN_HPP_NAMESPACE::Image                                     srcImage,
                              VULKAN_HPP_NAMESPACE::ImageLayout                               srcImageLayout,
                              VULKAN_HPP_NAMESPACE::Buffer                                    dstBuffer,
                              ArrayProxy<const VULKAN_HPP_NAMESPACE::BufferImageCopy> const & regions ) const
        VULKAN_HPP_NOEXCEPT;

      template <typename T>
      void updateBuffer( VULKAN_HPP_NAMESPACE::Buffer     dstBuffer,
                         VULKAN_HPP_NAMESPACE::DeviceSize dstOffset,
                         ArrayProxy<const T> const &      data ) const VULKAN_HPP_NOEXCEPT;

      void fillBuffer( VULKAN_HPP_NAMESPACE::Buffer     dstBuffer,
                       VULKAN_HPP_NAMESPACE::DeviceSize dstOffset,
                       VULKAN_HPP_NAMESPACE::DeviceSize size,
                       uint32_t                         data ) const VULKAN_HPP_NOEXCEPT;

      void clearColorImage( VULKAN_HPP_NAMESPACE::Image                                           image,
                            VULKAN_HPP_NAMESPACE::ImageLayout                                     imageLayout,
                            const ClearColorValue &                                               color,
                            ArrayProxy<const VULKAN_HPP_NAMESPACE::ImageSubresourceRange> const & ranges ) const
        VULKAN_HPP_NOEXCEPT;

      void clearDepthStencilImage( VULKAN_HPP_NAMESPACE::Image                                           image,
                                   VULKAN_HPP_NAMESPACE::ImageLayout                                     imageLayout,
                                   const ClearDepthStencilValue &                                        depthStencil,
                                   ArrayProxy<const VULKAN_HPP_NAMESPACE::ImageSubresourceRange> const & ranges ) const
        VULKAN_HPP_NOEXCEPT;

      void
        clearAttachments( ArrayProxy<const VULKAN_HPP_NAMESPACE::ClearAttachment> const & attachments,
                          ArrayProxy<const VULKAN_HPP_NAMESPACE::ClearRect> const & rects ) const VULKAN_HPP_NOEXCEPT;

      void
        resolveImage( VULKAN_HPP_NAMESPACE::Image                                  srcImage,
                      VULKAN_HPP_NAMESPACE::ImageLayout                            srcImageLayout,
                      VULKAN_HPP_NAMESPACE::Image                                  dstImage,
                      VULKAN_HPP_NAMESPACE::ImageLayout                            dstImageLayout,
                      ArrayProxy<const VULKAN_HPP_NAMESPACE::ImageResolve> const & regions ) const VULKAN_HPP_NOEXCEPT;

      void setEvent( VULKAN_HPP_NAMESPACE::Event              event,
                     VULKAN_HPP_NAMESPACE::PipelineStageFlags stageMask
                                                              VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT ) const VULKAN_HPP_NOEXCEPT;

      void resetEvent( VULKAN_HPP_NAMESPACE::Event              event,
                       VULKAN_HPP_NAMESPACE::PipelineStageFlags stageMask
                                                                VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT ) const VULKAN_HPP_NOEXCEPT;

      void waitEvents( ArrayProxy<const VULKAN_HPP_NAMESPACE::Event> const &               events,
                       VULKAN_HPP_NAMESPACE::PipelineStageFlags                            srcStageMask,
                       VULKAN_HPP_NAMESPACE::PipelineStageFlags                            dstStageMask,
                       ArrayProxy<const VULKAN_HPP_NAMESPACE::MemoryBarrier> const &       memoryBarriers,
                       ArrayProxy<const VULKAN_HPP_NAMESPACE::BufferMemoryBarrier> const & bufferMemoryBarriers,
                       ArrayProxy<const VULKAN_HPP_NAMESPACE::ImageMemoryBarrier> const &  imageMemoryBarriers ) const
        VULKAN_HPP_NOEXCEPT;

      void pipelineBarrier( VULKAN_HPP_NAMESPACE::PipelineStageFlags                            srcStageMask,
                            VULKAN_HPP_NAMESPACE::PipelineStageFlags                            dstStageMask,
                            VULKAN_HPP_NAMESPACE::DependencyFlags                               dependencyFlags,
                            ArrayProxy<const VULKAN_HPP_NAMESPACE::MemoryBarrier> const &       memoryBarriers,
                            ArrayProxy<const VULKAN_HPP_NAMESPACE::BufferMemoryBarrier> const & bufferMemoryBarriers,
                            ArrayProxy<const VULKAN_HPP_NAMESPACE::ImageMemoryBarrier> const &  imageMemoryBarriers )
        const VULKAN_HPP_NOEXCEPT;

      void beginQuery( VULKAN_HPP_NAMESPACE::QueryPool         queryPool,
                       uint32_t                                query,
                       VULKAN_HPP_NAMESPACE::QueryControlFlags flags
                                                               VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT ) const VULKAN_HPP_NOEXCEPT;

      void endQuery( VULKAN_HPP_NAMESPACE::QueryPool queryPool, uint32_t query ) const VULKAN_HPP_NOEXCEPT;

      void resetQueryPool( VULKAN_HPP_NAMESPACE::QueryPool queryPool,
                           uint32_t                        firstQuery,
                           uint32_t                        queryCount ) const VULKAN_HPP_NOEXCEPT;

      void writeTimestamp( VULKAN_HPP_NAMESPACE::PipelineStageFlagBits pipelineStage,
                           VULKAN_HPP_NAMESPACE::QueryPool             queryPool,
                           uint32_t                                    query ) const VULKAN_HPP_NOEXCEPT;

      void copyQueryPoolResults( VULKAN_HPP_NAMESPACE::QueryPool        queryPool,
                                 uint32_t                               firstQuery,
                                 uint32_t                               queryCount,
                                 VULKAN_HPP_NAMESPACE::Buffer           dstBuffer,
                                 VULKAN_HPP_NAMESPACE::DeviceSize       dstOffset,
                                 VULKAN_HPP_NAMESPACE::DeviceSize       stride,
                                 VULKAN_HPP_NAMESPACE::QueryResultFlags flags
                                                                        VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT ) const VULKAN_HPP_NOEXCEPT;

      template <typename T>
      void pushConstants( VULKAN_HPP_NAMESPACE::PipelineLayout   layout,
                          VULKAN_HPP_NAMESPACE::ShaderStageFlags stageFlags,
                          uint32_t                               offset,
                          ArrayProxy<const T> const &            values ) const VULKAN_HPP_NOEXCEPT;

      void beginRenderPass( const RenderPassBeginInfo &           renderPassBegin,
                            VULKAN_HPP_NAMESPACE::SubpassContents contents ) const VULKAN_HPP_NOEXCEPT;

      void nextSubpass( VULKAN_HPP_NAMESPACE::SubpassContents contents ) const VULKAN_HPP_NOEXCEPT;

      void endRenderPass() const VULKAN_HPP_NOEXCEPT;

      void executeCommands( ArrayProxy<const VULKAN_HPP_NAMESPACE::CommandBuffer> const & commandBuffers ) const
        VULKAN_HPP_NOEXCEPT;

      //=== VK_VERSION_1_1 ===

      void setDeviceMask( uint32_t deviceMask ) const VULKAN_HPP_NOEXCEPT;

      void dispatchBase( uint32_t baseGroupX,
                         uint32_t baseGroupY,
                         uint32_t baseGroupZ,
                         uint32_t groupCountX,
                         uint32_t groupCountY,
                         uint32_t groupCountZ ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_VERSION_1_2 ===

      void drawIndirectCount( VULKAN_HPP_NAMESPACE::Buffer     buffer,
                              VULKAN_HPP_NAMESPACE::DeviceSize offset,
                              VULKAN_HPP_NAMESPACE::Buffer     countBuffer,
                              VULKAN_HPP_NAMESPACE::DeviceSize countBufferOffset,
                              uint32_t                         maxDrawCount,
                              uint32_t                         stride ) const VULKAN_HPP_NOEXCEPT;

      void drawIndexedIndirectCount( VULKAN_HPP_NAMESPACE::Buffer     buffer,
                                     VULKAN_HPP_NAMESPACE::DeviceSize offset,
                                     VULKAN_HPP_NAMESPACE::Buffer     countBuffer,
                                     VULKAN_HPP_NAMESPACE::DeviceSize countBufferOffset,
                                     uint32_t                         maxDrawCount,
                                     uint32_t                         stride ) const VULKAN_HPP_NOEXCEPT;

      void beginRenderPass2( const RenderPassBeginInfo & renderPassBegin,
                             const SubpassBeginInfo &    subpassBeginInfo ) const VULKAN_HPP_NOEXCEPT;

      void nextSubpass2( const SubpassBeginInfo & subpassBeginInfo,
                         const SubpassEndInfo &   subpassEndInfo ) const VULKAN_HPP_NOEXCEPT;

      void endRenderPass2( const SubpassEndInfo & subpassEndInfo ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_EXT_debug_marker ===

      void debugMarkerBeginEXT( const DebugMarkerMarkerInfoEXT & markerInfo ) const VULKAN_HPP_NOEXCEPT;

      void debugMarkerEndEXT() const VULKAN_HPP_NOEXCEPT;

      void debugMarkerInsertEXT( const DebugMarkerMarkerInfoEXT & markerInfo ) const VULKAN_HPP_NOEXCEPT;

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
      //=== VK_KHR_video_queue ===

      void beginVideoCodingKHR( const VideoBeginCodingInfoKHR & beginInfo ) const VULKAN_HPP_NOEXCEPT;

      void endVideoCodingKHR( const VideoEndCodingInfoKHR & endCodingInfo ) const VULKAN_HPP_NOEXCEPT;

      void controlVideoCodingKHR( const VideoCodingControlInfoKHR & codingControlInfo ) const VULKAN_HPP_NOEXCEPT;
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
      //=== VK_KHR_video_decode_queue ===

      void decodeVideoKHR( const VideoDecodeInfoKHR & frameInfo ) const VULKAN_HPP_NOEXCEPT;
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

      //=== VK_EXT_transform_feedback ===

      void bindTransformFeedbackBuffersEXT( uint32_t                                                   firstBinding,
                                            ArrayProxy<const VULKAN_HPP_NAMESPACE::Buffer> const &     buffers,
                                            ArrayProxy<const VULKAN_HPP_NAMESPACE::DeviceSize> const & offsets,
                                            ArrayProxy<const VULKAN_HPP_NAMESPACE::DeviceSize> const & sizes
                                                                                                       VULKAN_HPP_DEFAULT_ARGUMENT_NULLPTR_ASSIGNMENT ) const
        VULKAN_HPP_NOEXCEPT_WHEN_NO_EXCEPTIONS;

      void beginTransformFeedbackEXT( uint32_t                                                   firstCounterBuffer,
                                      ArrayProxy<const VULKAN_HPP_NAMESPACE::Buffer> const &     counterBuffers,
                                      ArrayProxy<const VULKAN_HPP_NAMESPACE::DeviceSize> const & counterBufferOffsets
                                                                                                 VULKAN_HPP_DEFAULT_ARGUMENT_NULLPTR_ASSIGNMENT ) const
        VULKAN_HPP_NOEXCEPT_WHEN_NO_EXCEPTIONS;

      void endTransformFeedbackEXT( uint32_t                                                   firstCounterBuffer,
                                    ArrayProxy<const VULKAN_HPP_NAMESPACE::Buffer> const &     counterBuffers,
                                    ArrayProxy<const VULKAN_HPP_NAMESPACE::DeviceSize> const & counterBufferOffsets
                                                                                               VULKAN_HPP_DEFAULT_ARGUMENT_NULLPTR_ASSIGNMENT ) const
        VULKAN_HPP_NOEXCEPT_WHEN_NO_EXCEPTIONS;

      void beginQueryIndexedEXT( VULKAN_HPP_NAMESPACE::QueryPool         queryPool,
                                 uint32_t                                query,
                                 VULKAN_HPP_NAMESPACE::QueryControlFlags flags,
                                 uint32_t                                index ) const VULKAN_HPP_NOEXCEPT;

      void endQueryIndexedEXT( VULKAN_HPP_NAMESPACE::QueryPool queryPool,
                               uint32_t                        query,
                               uint32_t                        index ) const VULKAN_HPP_NOEXCEPT;

      void drawIndirectByteCountEXT( uint32_t                         instanceCount,
                                     uint32_t                         firstInstance,
                                     VULKAN_HPP_NAMESPACE::Buffer     counterBuffer,
                                     VULKAN_HPP_NAMESPACE::DeviceSize counterBufferOffset,
                                     uint32_t                         counterOffset,
                                     uint32_t                         vertexStride ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_NVX_binary_import ===

      void cuLaunchKernelNVX( const CuLaunchInfoNVX & launchInfo ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_AMD_draw_indirect_count ===

      void drawIndirectCountAMD( VULKAN_HPP_NAMESPACE::Buffer     buffer,
                                 VULKAN_HPP_NAMESPACE::DeviceSize offset,
                                 VULKAN_HPP_NAMESPACE::Buffer     countBuffer,
                                 VULKAN_HPP_NAMESPACE::DeviceSize countBufferOffset,
                                 uint32_t                         maxDrawCount,
                                 uint32_t                         stride ) const VULKAN_HPP_NOEXCEPT;

      void drawIndexedIndirectCountAMD( VULKAN_HPP_NAMESPACE::Buffer     buffer,
                                        VULKAN_HPP_NAMESPACE::DeviceSize offset,
                                        VULKAN_HPP_NAMESPACE::Buffer     countBuffer,
                                        VULKAN_HPP_NAMESPACE::DeviceSize countBufferOffset,
                                        uint32_t                         maxDrawCount,
                                        uint32_t                         stride ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_KHR_device_group ===

      void setDeviceMaskKHR( uint32_t deviceMask ) const VULKAN_HPP_NOEXCEPT;

      void dispatchBaseKHR( uint32_t baseGroupX,
                            uint32_t baseGroupY,
                            uint32_t baseGroupZ,
                            uint32_t groupCountX,
                            uint32_t groupCountY,
                            uint32_t groupCountZ ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_KHR_push_descriptor ===

      void pushDescriptorSetKHR(
        VULKAN_HPP_NAMESPACE::PipelineBindPoint                            pipelineBindPoint,
        VULKAN_HPP_NAMESPACE::PipelineLayout                               layout,
        uint32_t                                                           set,
        ArrayProxy<const VULKAN_HPP_NAMESPACE::WriteDescriptorSet> const & descriptorWrites ) const VULKAN_HPP_NOEXCEPT;

      void pushDescriptorSetWithTemplateKHR( VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplate descriptorUpdateTemplate,
                                             VULKAN_HPP_NAMESPACE::PipelineLayout           layout,
                                             uint32_t                                       set,
                                             const void * pData ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_EXT_conditional_rendering ===

      void beginConditionalRenderingEXT( const ConditionalRenderingBeginInfoEXT & conditionalRenderingBegin ) const
        VULKAN_HPP_NOEXCEPT;

      void endConditionalRenderingEXT() const VULKAN_HPP_NOEXCEPT;

      //=== VK_NV_clip_space_w_scaling ===

      void setViewportWScalingNV( uint32_t                                                           firstViewport,
                                  ArrayProxy<const VULKAN_HPP_NAMESPACE::ViewportWScalingNV> const & viewportWScalings )
        const VULKAN_HPP_NOEXCEPT;

      //=== VK_EXT_discard_rectangles ===

      void setDiscardRectangleEXT( uint32_t                                               firstDiscardRectangle,
                                   ArrayProxy<const VULKAN_HPP_NAMESPACE::Rect2D> const & discardRectangles ) const
        VULKAN_HPP_NOEXCEPT;

      //=== VK_KHR_create_renderpass2 ===

      void beginRenderPass2KHR( const RenderPassBeginInfo & renderPassBegin,
                                const SubpassBeginInfo &    subpassBeginInfo ) const VULKAN_HPP_NOEXCEPT;

      void nextSubpass2KHR( const SubpassBeginInfo & subpassBeginInfo,
                            const SubpassEndInfo &   subpassEndInfo ) const VULKAN_HPP_NOEXCEPT;

      void endRenderPass2KHR( const SubpassEndInfo & subpassEndInfo ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_EXT_debug_utils ===

      void beginDebugUtilsLabelEXT( const DebugUtilsLabelEXT & labelInfo ) const VULKAN_HPP_NOEXCEPT;

      void endDebugUtilsLabelEXT() const VULKAN_HPP_NOEXCEPT;

      void insertDebugUtilsLabelEXT( const DebugUtilsLabelEXT & labelInfo ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_EXT_sample_locations ===

      void setSampleLocationsEXT( const SampleLocationsInfoEXT & sampleLocationsInfo ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_KHR_acceleration_structure ===

      void buildAccelerationStructuresKHR(
        ArrayProxy<const VULKAN_HPP_NAMESPACE::AccelerationStructureBuildGeometryInfoKHR> const & infos,
        ArrayProxy<const VULKAN_HPP_NAMESPACE::AccelerationStructureBuildRangeInfoKHR * const> const &
          pBuildRangeInfos ) const VULKAN_HPP_NOEXCEPT_WHEN_NO_EXCEPTIONS;

      void buildAccelerationStructuresIndirectKHR(
        ArrayProxy<const VULKAN_HPP_NAMESPACE::AccelerationStructureBuildGeometryInfoKHR> const & infos,
        ArrayProxy<const VULKAN_HPP_NAMESPACE::DeviceAddress> const & indirectDeviceAddresses,
        ArrayProxy<const uint32_t> const &                            indirectStrides,
        ArrayProxy<const uint32_t * const> const & pMaxPrimitiveCounts ) const VULKAN_HPP_NOEXCEPT_WHEN_NO_EXCEPTIONS;

      void copyAccelerationStructureKHR( const CopyAccelerationStructureInfoKHR & info ) const VULKAN_HPP_NOEXCEPT;

      void copyAccelerationStructureToMemoryKHR( const CopyAccelerationStructureToMemoryInfoKHR & info ) const
        VULKAN_HPP_NOEXCEPT;

      void copyMemoryToAccelerationStructureKHR( const CopyMemoryToAccelerationStructureInfoKHR & info ) const
        VULKAN_HPP_NOEXCEPT;

      void writeAccelerationStructuresPropertiesKHR(
        ArrayProxy<const VULKAN_HPP_NAMESPACE::AccelerationStructureKHR> const & accelerationStructures,
        VULKAN_HPP_NAMESPACE::QueryType                                          queryType,
        VULKAN_HPP_NAMESPACE::QueryPool                                          queryPool,
        uint32_t                                                                 firstQuery ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_NV_shading_rate_image ===

      void bindShadingRateImageNV( VULKAN_HPP_NAMESPACE::ImageView   imageView,
                                   VULKAN_HPP_NAMESPACE::ImageLayout imageLayout ) const VULKAN_HPP_NOEXCEPT;

      void setViewportShadingRatePaletteNV( uint32_t firstViewport,
                                            ArrayProxy<const VULKAN_HPP_NAMESPACE::ShadingRatePaletteNV> const &
                                              shadingRatePalettes ) const VULKAN_HPP_NOEXCEPT;

      void setCoarseSampleOrderNV( VULKAN_HPP_NAMESPACE::CoarseSampleOrderTypeNV sampleOrderType,
                                   ArrayProxy<const VULKAN_HPP_NAMESPACE::CoarseSampleOrderCustomNV> const &
                                     customSampleOrders ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_NV_ray_tracing ===

      void buildAccelerationStructureNV( const AccelerationStructureInfoNV &           info,
                                         VULKAN_HPP_NAMESPACE::Buffer                  instanceData,
                                         VULKAN_HPP_NAMESPACE::DeviceSize              instanceOffset,
                                         VULKAN_HPP_NAMESPACE::Bool32                  update,
                                         VULKAN_HPP_NAMESPACE::AccelerationStructureNV dst,
                                         VULKAN_HPP_NAMESPACE::AccelerationStructureNV src,
                                         VULKAN_HPP_NAMESPACE::Buffer                  scratch,
                                         VULKAN_HPP_NAMESPACE::DeviceSize scratchOffset ) const VULKAN_HPP_NOEXCEPT;

      void copyAccelerationStructureNV( VULKAN_HPP_NAMESPACE::AccelerationStructureNV          dst,
                                        VULKAN_HPP_NAMESPACE::AccelerationStructureNV          src,
                                        VULKAN_HPP_NAMESPACE::CopyAccelerationStructureModeKHR mode ) const
        VULKAN_HPP_NOEXCEPT;

      void traceRaysNV( VULKAN_HPP_NAMESPACE::Buffer     raygenShaderBindingTableBuffer,
                        VULKAN_HPP_NAMESPACE::DeviceSize raygenShaderBindingOffset,
                        VULKAN_HPP_NAMESPACE::Buffer     missShaderBindingTableBuffer,
                        VULKAN_HPP_NAMESPACE::DeviceSize missShaderBindingOffset,
                        VULKAN_HPP_NAMESPACE::DeviceSize missShaderBindingStride,
                        VULKAN_HPP_NAMESPACE::Buffer     hitShaderBindingTableBuffer,
                        VULKAN_HPP_NAMESPACE::DeviceSize hitShaderBindingOffset,
                        VULKAN_HPP_NAMESPACE::DeviceSize hitShaderBindingStride,
                        VULKAN_HPP_NAMESPACE::Buffer     callableShaderBindingTableBuffer,
                        VULKAN_HPP_NAMESPACE::DeviceSize callableShaderBindingOffset,
                        VULKAN_HPP_NAMESPACE::DeviceSize callableShaderBindingStride,
                        uint32_t                         width,
                        uint32_t                         height,
                        uint32_t                         depth ) const VULKAN_HPP_NOEXCEPT;

      void writeAccelerationStructuresPropertiesNV(
        ArrayProxy<const VULKAN_HPP_NAMESPACE::AccelerationStructureNV> const & accelerationStructures,
        VULKAN_HPP_NAMESPACE::QueryType                                         queryType,
        VULKAN_HPP_NAMESPACE::QueryPool                                         queryPool,
        uint32_t                                                                firstQuery ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_KHR_draw_indirect_count ===

      void drawIndirectCountKHR( VULKAN_HPP_NAMESPACE::Buffer     buffer,
                                 VULKAN_HPP_NAMESPACE::DeviceSize offset,
                                 VULKAN_HPP_NAMESPACE::Buffer     countBuffer,
                                 VULKAN_HPP_NAMESPACE::DeviceSize countBufferOffset,
                                 uint32_t                         maxDrawCount,
                                 uint32_t                         stride ) const VULKAN_HPP_NOEXCEPT;

      void drawIndexedIndirectCountKHR( VULKAN_HPP_NAMESPACE::Buffer     buffer,
                                        VULKAN_HPP_NAMESPACE::DeviceSize offset,
                                        VULKAN_HPP_NAMESPACE::Buffer     countBuffer,
                                        VULKAN_HPP_NAMESPACE::DeviceSize countBufferOffset,
                                        uint32_t                         maxDrawCount,
                                        uint32_t                         stride ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_AMD_buffer_marker ===

      void writeBufferMarkerAMD( VULKAN_HPP_NAMESPACE::PipelineStageFlagBits pipelineStage,
                                 VULKAN_HPP_NAMESPACE::Buffer                dstBuffer,
                                 VULKAN_HPP_NAMESPACE::DeviceSize            dstOffset,
                                 uint32_t                                    marker ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_NV_mesh_shader ===

      void drawMeshTasksNV( uint32_t taskCount, uint32_t firstTask ) const VULKAN_HPP_NOEXCEPT;

      void drawMeshTasksIndirectNV( VULKAN_HPP_NAMESPACE::Buffer     buffer,
                                    VULKAN_HPP_NAMESPACE::DeviceSize offset,
                                    uint32_t                         drawCount,
                                    uint32_t                         stride ) const VULKAN_HPP_NOEXCEPT;

      void drawMeshTasksIndirectCountNV( VULKAN_HPP_NAMESPACE::Buffer     buffer,
                                         VULKAN_HPP_NAMESPACE::DeviceSize offset,
                                         VULKAN_HPP_NAMESPACE::Buffer     countBuffer,
                                         VULKAN_HPP_NAMESPACE::DeviceSize countBufferOffset,
                                         uint32_t                         maxDrawCount,
                                         uint32_t                         stride ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_NV_scissor_exclusive ===

      void setExclusiveScissorNV( uint32_t                                               firstExclusiveScissor,
                                  ArrayProxy<const VULKAN_HPP_NAMESPACE::Rect2D> const & exclusiveScissors ) const
        VULKAN_HPP_NOEXCEPT;

      //=== VK_NV_device_diagnostic_checkpoints ===

      void setCheckpointNV( const void * pCheckpointMarker ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_INTEL_performance_query ===

      void setPerformanceMarkerINTEL( const PerformanceMarkerInfoINTEL & markerInfo ) const;

      void setPerformanceStreamMarkerINTEL( const PerformanceStreamMarkerInfoINTEL & markerInfo ) const;

      void setPerformanceOverrideINTEL( const PerformanceOverrideInfoINTEL & overrideInfo ) const;

      //=== VK_KHR_fragment_shading_rate ===

      void setFragmentShadingRateKHR(
        const Extent2D &                                             fragmentSize,
        const VULKAN_HPP_NAMESPACE::FragmentShadingRateCombinerOpKHR combinerOps[2] ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_EXT_line_rasterization ===

      void setLineStippleEXT( uint32_t lineStippleFactor, uint16_t lineStipplePattern ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_EXT_extended_dynamic_state ===

      void setCullModeEXT( VULKAN_HPP_NAMESPACE::CullModeFlags cullMode VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT ) const
        VULKAN_HPP_NOEXCEPT;

      void setFrontFaceEXT( VULKAN_HPP_NAMESPACE::FrontFace frontFace ) const VULKAN_HPP_NOEXCEPT;

      void
        setPrimitiveTopologyEXT( VULKAN_HPP_NAMESPACE::PrimitiveTopology primitiveTopology ) const VULKAN_HPP_NOEXCEPT;

      void setViewportWithCountEXT( ArrayProxy<const VULKAN_HPP_NAMESPACE::Viewport> const & viewports ) const
        VULKAN_HPP_NOEXCEPT;

      void setScissorWithCountEXT( ArrayProxy<const VULKAN_HPP_NAMESPACE::Rect2D> const & scissors ) const
        VULKAN_HPP_NOEXCEPT;

      void bindVertexBuffers2EXT(
        uint32_t                                                   firstBinding,
        ArrayProxy<const VULKAN_HPP_NAMESPACE::Buffer> const &     buffers,
        ArrayProxy<const VULKAN_HPP_NAMESPACE::DeviceSize> const & offsets,
        ArrayProxy<const VULKAN_HPP_NAMESPACE::DeviceSize> const & sizes VULKAN_HPP_DEFAULT_ARGUMENT_NULLPTR_ASSIGNMENT,
        ArrayProxy<const VULKAN_HPP_NAMESPACE::DeviceSize> const &       strides
                                                                         VULKAN_HPP_DEFAULT_ARGUMENT_NULLPTR_ASSIGNMENT ) const VULKAN_HPP_NOEXCEPT_WHEN_NO_EXCEPTIONS;

      void setDepthTestEnableEXT( VULKAN_HPP_NAMESPACE::Bool32 depthTestEnable ) const VULKAN_HPP_NOEXCEPT;

      void setDepthWriteEnableEXT( VULKAN_HPP_NAMESPACE::Bool32 depthWriteEnable ) const VULKAN_HPP_NOEXCEPT;

      void setDepthCompareOpEXT( VULKAN_HPP_NAMESPACE::CompareOp depthCompareOp ) const VULKAN_HPP_NOEXCEPT;

      void setDepthBoundsTestEnableEXT( VULKAN_HPP_NAMESPACE::Bool32 depthBoundsTestEnable ) const VULKAN_HPP_NOEXCEPT;

      void setStencilTestEnableEXT( VULKAN_HPP_NAMESPACE::Bool32 stencilTestEnable ) const VULKAN_HPP_NOEXCEPT;

      void setStencilOpEXT( VULKAN_HPP_NAMESPACE::StencilFaceFlags faceMask,
                            VULKAN_HPP_NAMESPACE::StencilOp        failOp,
                            VULKAN_HPP_NAMESPACE::StencilOp        passOp,
                            VULKAN_HPP_NAMESPACE::StencilOp        depthFailOp,
                            VULKAN_HPP_NAMESPACE::CompareOp        compareOp ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_NV_device_generated_commands ===

      void preprocessGeneratedCommandsNV( const GeneratedCommandsInfoNV & generatedCommandsInfo ) const
        VULKAN_HPP_NOEXCEPT;

      void
        executeGeneratedCommandsNV( VULKAN_HPP_NAMESPACE::Bool32    isPreprocessed,
                                    const GeneratedCommandsInfoNV & generatedCommandsInfo ) const VULKAN_HPP_NOEXCEPT;

      void bindPipelineShaderGroupNV( VULKAN_HPP_NAMESPACE::PipelineBindPoint pipelineBindPoint,
                                      VULKAN_HPP_NAMESPACE::Pipeline          pipeline,
                                      uint32_t                                groupIndex ) const VULKAN_HPP_NOEXCEPT;

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
      //=== VK_KHR_video_encode_queue ===

      void encodeVideoKHR( const VideoEncodeInfoKHR & encodeInfo ) const VULKAN_HPP_NOEXCEPT;
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

      //=== VK_KHR_synchronization2 ===

      void setEvent2KHR( VULKAN_HPP_NAMESPACE::Event event,
                         const DependencyInfoKHR &   dependencyInfo ) const VULKAN_HPP_NOEXCEPT;

      void resetEvent2KHR( VULKAN_HPP_NAMESPACE::Event                  event,
                           VULKAN_HPP_NAMESPACE::PipelineStageFlags2KHR stageMask ) const VULKAN_HPP_NOEXCEPT;

      void waitEvents2KHR( ArrayProxy<const VULKAN_HPP_NAMESPACE::Event> const &             events,
                           ArrayProxy<const VULKAN_HPP_NAMESPACE::DependencyInfoKHR> const & dependencyInfos ) const
        VULKAN_HPP_NOEXCEPT_WHEN_NO_EXCEPTIONS;

      void pipelineBarrier2KHR( const DependencyInfoKHR & dependencyInfo ) const VULKAN_HPP_NOEXCEPT;

      void writeTimestamp2KHR( VULKAN_HPP_NAMESPACE::PipelineStageFlags2KHR stage,
                               VULKAN_HPP_NAMESPACE::QueryPool              queryPool,
                               uint32_t                                     query ) const VULKAN_HPP_NOEXCEPT;

      void writeBufferMarker2AMD( VULKAN_HPP_NAMESPACE::PipelineStageFlags2KHR stage,
                                  VULKAN_HPP_NAMESPACE::Buffer                 dstBuffer,
                                  VULKAN_HPP_NAMESPACE::DeviceSize             dstOffset,
                                  uint32_t                                     marker ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_NV_fragment_shading_rate_enums ===

      void setFragmentShadingRateEnumNV(
        VULKAN_HPP_NAMESPACE::FragmentShadingRateNV                  shadingRate,
        const VULKAN_HPP_NAMESPACE::FragmentShadingRateCombinerOpKHR combinerOps[2] ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_KHR_copy_commands2 ===

      void copyBuffer2KHR( const CopyBufferInfo2KHR & copyBufferInfo ) const VULKAN_HPP_NOEXCEPT;

      void copyImage2KHR( const CopyImageInfo2KHR & copyImageInfo ) const VULKAN_HPP_NOEXCEPT;

      void copyBufferToImage2KHR( const CopyBufferToImageInfo2KHR & copyBufferToImageInfo ) const VULKAN_HPP_NOEXCEPT;

      void copyImageToBuffer2KHR( const CopyImageToBufferInfo2KHR & copyImageToBufferInfo ) const VULKAN_HPP_NOEXCEPT;

      void blitImage2KHR( const BlitImageInfo2KHR & blitImageInfo ) const VULKAN_HPP_NOEXCEPT;

      void resolveImage2KHR( const ResolveImageInfo2KHR & resolveImageInfo ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_KHR_ray_tracing_pipeline ===

      void traceRaysKHR( const StridedDeviceAddressRegionKHR & raygenShaderBindingTable,
                         const StridedDeviceAddressRegionKHR & missShaderBindingTable,
                         const StridedDeviceAddressRegionKHR & hitShaderBindingTable,
                         const StridedDeviceAddressRegionKHR & callableShaderBindingTable,
                         uint32_t                              width,
                         uint32_t                              height,
                         uint32_t                              depth ) const VULKAN_HPP_NOEXCEPT;

      void traceRaysIndirectKHR( const StridedDeviceAddressRegionKHR & raygenShaderBindingTable,
                                 const StridedDeviceAddressRegionKHR & missShaderBindingTable,
                                 const StridedDeviceAddressRegionKHR & hitShaderBindingTable,
                                 const StridedDeviceAddressRegionKHR & callableShaderBindingTable,
                                 VULKAN_HPP_NAMESPACE::DeviceAddress indirectDeviceAddress ) const VULKAN_HPP_NOEXCEPT;

      void setRayTracingPipelineStackSizeKHR( uint32_t pipelineStackSize ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_EXT_vertex_input_dynamic_state ===

      void setVertexInputEXT(
        ArrayProxy<const VULKAN_HPP_NAMESPACE::VertexInputBindingDescription2EXT> const & vertexBindingDescriptions,
        ArrayProxy<const VULKAN_HPP_NAMESPACE::VertexInputAttributeDescription2EXT> const &
          vertexAttributeDescriptions ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_HUAWEI_subpass_shading ===

      void subpassShadingHUAWEI() const VULKAN_HPP_NOEXCEPT;

      //=== VK_HUAWEI_invocation_mask ===

      void bindInvocationMaskHUAWEI( VULKAN_HPP_NAMESPACE::ImageView   imageView,
                                     VULKAN_HPP_NAMESPACE::ImageLayout imageLayout ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_EXT_extended_dynamic_state2 ===

      void setPatchControlPointsEXT( uint32_t patchControlPoints ) const VULKAN_HPP_NOEXCEPT;

      void
        setRasterizerDiscardEnableEXT( VULKAN_HPP_NAMESPACE::Bool32 rasterizerDiscardEnable ) const VULKAN_HPP_NOEXCEPT;

      void setDepthBiasEnableEXT( VULKAN_HPP_NAMESPACE::Bool32 depthBiasEnable ) const VULKAN_HPP_NOEXCEPT;

      void setLogicOpEXT( VULKAN_HPP_NAMESPACE::LogicOp logicOp ) const VULKAN_HPP_NOEXCEPT;

      void
        setPrimitiveRestartEnableEXT( VULKAN_HPP_NAMESPACE::Bool32 primitiveRestartEnable ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_EXT_color_write_enable ===

      void setColorWriteEnableEXT( ArrayProxy<const VULKAN_HPP_NAMESPACE::Bool32> const & colorWriteEnables ) const
        VULKAN_HPP_NOEXCEPT;

      //=== VK_EXT_multi_draw ===

      void drawMultiEXT( ArrayProxy<const VULKAN_HPP_NAMESPACE::MultiDrawInfoEXT> const & vertexInfo,
                         uint32_t                                                         instanceCount,
                         uint32_t                                                         firstInstance,
                         uint32_t stride ) const VULKAN_HPP_NOEXCEPT;

      void drawMultiIndexedEXT( ArrayProxy<const VULKAN_HPP_NAMESPACE::MultiDrawIndexedInfoEXT> const & indexInfo,
                                uint32_t                                                                instanceCount,
                                uint32_t                                                                firstInstance,
                                uint32_t                                                                stride,
                                Optional<const int32_t>                                                 vertexOffset
                                                                                                        VULKAN_HPP_DEFAULT_ARGUMENT_NULLPTR_ASSIGNMENT ) const VULKAN_HPP_NOEXCEPT;

    private:
      VULKAN_HPP_NAMESPACE::CommandBuffer                                       m_commandBuffer;
      VkDevice                                                                  m_device;
      VkCommandPool                                                             m_commandPool;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class CommandBuffers : public std::vector<VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::CommandBuffer>
    {
    public:
      CommandBuffers( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const & device,
                      VULKAN_HPP_NAMESPACE::CommandBufferAllocateInfo const &         allocateInfo )
      {
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * dispatcher = device.getDispatcher();
        std::vector<VkCommandBuffer> commandBuffers( allocateInfo.commandBufferCount );
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          dispatcher->vkAllocateCommandBuffers( static_cast<VkDevice>( *device ),
                                                reinterpret_cast<const VkCommandBufferAllocateInfo *>( &allocateInfo ),
                                                commandBuffers.data() ) );
        if ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          this->reserve( allocateInfo.commandBufferCount );
          for ( auto const & commandBuffer : commandBuffers )
          {
            this->emplace_back( commandBuffer,
                                static_cast<VkDevice>( *device ),
                                static_cast<VkCommandPool>( allocateInfo.commandPool ),
                                dispatcher );
          }
        }
        else
        {
          throwResultException( result, "vkAllocateCommandBuffers" );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      CommandBuffers() = default;
#  else
      CommandBuffers()                                                        = delete;
#  endif
      CommandBuffers( CommandBuffers const & ) = delete;
      CommandBuffers( CommandBuffers && rhs )  = default;
      CommandBuffers & operator=( CommandBuffers const & ) = delete;
      CommandBuffers & operator=( CommandBuffers && rhs ) = default;
    };

    class CuFunctionNVX
    {
    public:
      using CType = VkCuFunctionNVX;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eCuFunctionNVX;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eCuFunctionNVX;

    public:
      CuFunctionNVX(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VULKAN_HPP_NAMESPACE::CuFunctionCreateInfoNVX const &                           createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkCreateCuFunctionNVX( static_cast<VkDevice>( *device ),
                                                  reinterpret_cast<const VkCuFunctionCreateInfoNVX *>( &createInfo ),
                                                  m_allocator,
                                                  reinterpret_cast<VkCuFunctionNVX *>( &m_cuFunctionNVX ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateCuFunctionNVX" );
        }
      }

      CuFunctionNVX(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VkCuFunctionNVX                                                                 cuFunctionNVX,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_cuFunctionNVX( cuFunctionNVX )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~CuFunctionNVX()
      {
        if ( m_cuFunctionNVX )
        {
          getDispatcher()->vkDestroyCuFunctionNVX(
            m_device, static_cast<VkCuFunctionNVX>( m_cuFunctionNVX ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      CuFunctionNVX() = default;
#  else
      CuFunctionNVX()                                                         = delete;
#  endif
      CuFunctionNVX( CuFunctionNVX const & ) = delete;
      CuFunctionNVX( CuFunctionNVX && rhs ) VULKAN_HPP_NOEXCEPT
        : m_cuFunctionNVX( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_cuFunctionNVX, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      CuFunctionNVX & operator=( CuFunctionNVX const & ) = delete;
      CuFunctionNVX & operator                           =( CuFunctionNVX && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_cuFunctionNVX )
          {
            getDispatcher()->vkDestroyCuFunctionNVX(
              m_device, static_cast<VkCuFunctionNVX>( m_cuFunctionNVX ), m_allocator );
          }
          m_cuFunctionNVX = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_cuFunctionNVX, {} );
          m_device        = rhs.m_device;
          m_allocator     = rhs.m_allocator;
          m_dispatcher    = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::CuFunctionNVX const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_cuFunctionNVX;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_cuFunctionNVX.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_cuFunctionNVX.operator!();
      }
#  endif

    private:
      VULKAN_HPP_NAMESPACE::CuFunctionNVX                                       m_cuFunctionNVX;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class CuModuleNVX
    {
    public:
      using CType = VkCuModuleNVX;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eCuModuleNVX;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eCuModuleNVX;

    public:
      CuModuleNVX( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
                   VULKAN_HPP_NAMESPACE::CuModuleCreateInfoNVX const &                             createInfo,
                   VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkCreateCuModuleNVX( static_cast<VkDevice>( *device ),
                                                reinterpret_cast<const VkCuModuleCreateInfoNVX *>( &createInfo ),
                                                m_allocator,
                                                reinterpret_cast<VkCuModuleNVX *>( &m_cuModuleNVX ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateCuModuleNVX" );
        }
      }

      CuModuleNVX( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
                   VkCuModuleNVX                                                                   cuModuleNVX,
                   VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_cuModuleNVX( cuModuleNVX )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~CuModuleNVX()
      {
        if ( m_cuModuleNVX )
        {
          getDispatcher()->vkDestroyCuModuleNVX( m_device, static_cast<VkCuModuleNVX>( m_cuModuleNVX ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      CuModuleNVX() = default;
#  else
      CuModuleNVX()                                                           = delete;
#  endif
      CuModuleNVX( CuModuleNVX const & ) = delete;
      CuModuleNVX( CuModuleNVX && rhs ) VULKAN_HPP_NOEXCEPT
        : m_cuModuleNVX( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_cuModuleNVX, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      CuModuleNVX & operator=( CuModuleNVX const & ) = delete;
      CuModuleNVX & operator                         =( CuModuleNVX && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_cuModuleNVX )
          {
            getDispatcher()->vkDestroyCuModuleNVX( m_device, static_cast<VkCuModuleNVX>( m_cuModuleNVX ), m_allocator );
          }
          m_cuModuleNVX = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_cuModuleNVX, {} );
          m_device      = rhs.m_device;
          m_allocator   = rhs.m_allocator;
          m_dispatcher  = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::CuModuleNVX const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_cuModuleNVX;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_cuModuleNVX.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_cuModuleNVX.operator!();
      }
#  endif

    private:
      VULKAN_HPP_NAMESPACE::CuModuleNVX                                         m_cuModuleNVX;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class DebugReportCallbackEXT
    {
    public:
      using CType = VkDebugReportCallbackEXT;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eDebugReportCallbackEXT;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eDebugReportCallbackEXT;

    public:
      DebugReportCallbackEXT(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Instance const &               instance,
        VULKAN_HPP_NAMESPACE::DebugReportCallbackCreateInfoEXT const &                  createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_instance( *instance )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( instance.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateDebugReportCallbackEXT(
            static_cast<VkInstance>( *instance ),
            reinterpret_cast<const VkDebugReportCallbackCreateInfoEXT *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkDebugReportCallbackEXT *>( &m_debugReportCallbackEXT ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateDebugReportCallbackEXT" );
        }
      }

      DebugReportCallbackEXT(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Instance const &               instance,
        VkDebugReportCallbackEXT                                                        debugReportCallbackEXT,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_debugReportCallbackEXT( debugReportCallbackEXT )
        , m_instance( *instance )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( instance.getDispatcher() )
      {}

      ~DebugReportCallbackEXT()
      {
        if ( m_debugReportCallbackEXT )
        {
          getDispatcher()->vkDestroyDebugReportCallbackEXT(
            m_instance, static_cast<VkDebugReportCallbackEXT>( m_debugReportCallbackEXT ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      DebugReportCallbackEXT() = default;
#  else
      DebugReportCallbackEXT()                                                = delete;
#  endif
      DebugReportCallbackEXT( DebugReportCallbackEXT const & ) = delete;
      DebugReportCallbackEXT( DebugReportCallbackEXT && rhs ) VULKAN_HPP_NOEXCEPT
        : m_debugReportCallbackEXT(
            VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_debugReportCallbackEXT, {} ) )
        , m_instance( rhs.m_instance )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      DebugReportCallbackEXT & operator=( DebugReportCallbackEXT const & ) = delete;
      DebugReportCallbackEXT & operator=( DebugReportCallbackEXT && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_debugReportCallbackEXT )
          {
            getDispatcher()->vkDestroyDebugReportCallbackEXT(
              m_instance, static_cast<VkDebugReportCallbackEXT>( m_debugReportCallbackEXT ), m_allocator );
          }
          m_debugReportCallbackEXT =
            VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_debugReportCallbackEXT, {} );
          m_instance   = rhs.m_instance;
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::DebugReportCallbackEXT const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_debugReportCallbackEXT;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::InstanceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_debugReportCallbackEXT.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_debugReportCallbackEXT.operator!();
      }
#  endif

    private:
      VULKAN_HPP_NAMESPACE::DebugReportCallbackEXT                                m_debugReportCallbackEXT;
      VkInstance                                                                  m_instance;
      const VkAllocationCallbacks *                                               m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::InstanceDispatcher const * m_dispatcher;
    };

    class DebugUtilsMessengerEXT
    {
    public:
      using CType = VkDebugUtilsMessengerEXT;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eDebugUtilsMessengerEXT;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eUnknown;

    public:
      DebugUtilsMessengerEXT(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Instance const &               instance,
        VULKAN_HPP_NAMESPACE::DebugUtilsMessengerCreateInfoEXT const &                  createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_instance( *instance )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( instance.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateDebugUtilsMessengerEXT(
            static_cast<VkInstance>( *instance ),
            reinterpret_cast<const VkDebugUtilsMessengerCreateInfoEXT *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkDebugUtilsMessengerEXT *>( &m_debugUtilsMessengerEXT ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateDebugUtilsMessengerEXT" );
        }
      }

      DebugUtilsMessengerEXT(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Instance const &               instance,
        VkDebugUtilsMessengerEXT                                                        debugUtilsMessengerEXT,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_debugUtilsMessengerEXT( debugUtilsMessengerEXT )
        , m_instance( *instance )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( instance.getDispatcher() )
      {}

      ~DebugUtilsMessengerEXT()
      {
        if ( m_debugUtilsMessengerEXT )
        {
          getDispatcher()->vkDestroyDebugUtilsMessengerEXT(
            m_instance, static_cast<VkDebugUtilsMessengerEXT>( m_debugUtilsMessengerEXT ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      DebugUtilsMessengerEXT() = default;
#  else
      DebugUtilsMessengerEXT()                                                = delete;
#  endif
      DebugUtilsMessengerEXT( DebugUtilsMessengerEXT const & ) = delete;
      DebugUtilsMessengerEXT( DebugUtilsMessengerEXT && rhs ) VULKAN_HPP_NOEXCEPT
        : m_debugUtilsMessengerEXT(
            VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_debugUtilsMessengerEXT, {} ) )
        , m_instance( rhs.m_instance )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      DebugUtilsMessengerEXT & operator=( DebugUtilsMessengerEXT const & ) = delete;
      DebugUtilsMessengerEXT & operator=( DebugUtilsMessengerEXT && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_debugUtilsMessengerEXT )
          {
            getDispatcher()->vkDestroyDebugUtilsMessengerEXT(
              m_instance, static_cast<VkDebugUtilsMessengerEXT>( m_debugUtilsMessengerEXT ), m_allocator );
          }
          m_debugUtilsMessengerEXT =
            VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_debugUtilsMessengerEXT, {} );
          m_instance   = rhs.m_instance;
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::DebugUtilsMessengerEXT const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_debugUtilsMessengerEXT;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::InstanceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_debugUtilsMessengerEXT.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_debugUtilsMessengerEXT.operator!();
      }
#  endif

    private:
      VULKAN_HPP_NAMESPACE::DebugUtilsMessengerEXT                                m_debugUtilsMessengerEXT;
      VkInstance                                                                  m_instance;
      const VkAllocationCallbacks *                                               m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::InstanceDispatcher const * m_dispatcher;
    };

    class DeferredOperationKHR
    {
    public:
      using CType = VkDeferredOperationKHR;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eDeferredOperationKHR;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eUnknown;

    public:
      DeferredOperationKHR(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateDeferredOperationKHR(
            static_cast<VkDevice>( *device ),
            m_allocator,
            reinterpret_cast<VkDeferredOperationKHR *>( &m_deferredOperationKHR ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateDeferredOperationKHR" );
        }
      }

      DeferredOperationKHR(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VkDeferredOperationKHR                                                          deferredOperationKHR,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_deferredOperationKHR( deferredOperationKHR )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~DeferredOperationKHR()
      {
        if ( m_deferredOperationKHR )
        {
          getDispatcher()->vkDestroyDeferredOperationKHR(
            m_device, static_cast<VkDeferredOperationKHR>( m_deferredOperationKHR ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      DeferredOperationKHR() = default;
#  else
      DeferredOperationKHR()                                                  = delete;
#  endif
      DeferredOperationKHR( DeferredOperationKHR const & ) = delete;
      DeferredOperationKHR( DeferredOperationKHR && rhs ) VULKAN_HPP_NOEXCEPT
        : m_deferredOperationKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_deferredOperationKHR,
                                                                                             {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      DeferredOperationKHR & operator=( DeferredOperationKHR const & ) = delete;
      DeferredOperationKHR & operator=( DeferredOperationKHR && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_deferredOperationKHR )
          {
            getDispatcher()->vkDestroyDeferredOperationKHR(
              m_device, static_cast<VkDeferredOperationKHR>( m_deferredOperationKHR ), m_allocator );
          }
          m_deferredOperationKHR =
            VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_deferredOperationKHR, {} );
          m_device     = rhs.m_device;
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::DeferredOperationKHR const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_deferredOperationKHR;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_deferredOperationKHR.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_deferredOperationKHR.operator!();
      }
#  endif

      //=== VK_KHR_deferred_host_operations ===

      VULKAN_HPP_NODISCARD uint32_t getMaxConcurrency() const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::Result getResult() const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::Result join() const;

    private:
      VULKAN_HPP_NAMESPACE::DeferredOperationKHR                                m_deferredOperationKHR;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class DescriptorPool
    {
    public:
      using CType = VkDescriptorPool;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eDescriptorPool;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eDescriptorPool;

    public:
      DescriptorPool(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VULKAN_HPP_NAMESPACE::DescriptorPoolCreateInfo const &                          createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkCreateDescriptorPool( static_cast<VkDevice>( *device ),
                                                   reinterpret_cast<const VkDescriptorPoolCreateInfo *>( &createInfo ),
                                                   m_allocator,
                                                   reinterpret_cast<VkDescriptorPool *>( &m_descriptorPool ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateDescriptorPool" );
        }
      }

      DescriptorPool(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VkDescriptorPool                                                                descriptorPool,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_descriptorPool( descriptorPool )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~DescriptorPool()
      {
        if ( m_descriptorPool )
        {
          getDispatcher()->vkDestroyDescriptorPool(
            m_device, static_cast<VkDescriptorPool>( m_descriptorPool ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      DescriptorPool() = default;
#  else
      DescriptorPool()                                                        = delete;
#  endif
      DescriptorPool( DescriptorPool const & ) = delete;
      DescriptorPool( DescriptorPool && rhs ) VULKAN_HPP_NOEXCEPT
        : m_descriptorPool( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_descriptorPool, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      DescriptorPool & operator=( DescriptorPool const & ) = delete;
      DescriptorPool & operator                            =( DescriptorPool && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_descriptorPool )
          {
            getDispatcher()->vkDestroyDescriptorPool(
              m_device, static_cast<VkDescriptorPool>( m_descriptorPool ), m_allocator );
          }
          m_descriptorPool = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_descriptorPool, {} );
          m_device         = rhs.m_device;
          m_allocator      = rhs.m_allocator;
          m_dispatcher     = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::DescriptorPool const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_descriptorPool;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_descriptorPool.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_descriptorPool.operator!();
      }
#  endif

      //=== VK_VERSION_1_0 ===

      void reset( VULKAN_HPP_NAMESPACE::DescriptorPoolResetFlags flags VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT ) const
        VULKAN_HPP_NOEXCEPT;

    private:
      VULKAN_HPP_NAMESPACE::DescriptorPool                                      m_descriptorPool;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class DescriptorSet
    {
    public:
      using CType = VkDescriptorSet;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eDescriptorSet;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eDescriptorSet;

    public:
      DescriptorSet( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &         device,
                     VkDescriptorSet                                                         descriptorSet,
                     VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DescriptorPool const & descriptorPool )
        : m_descriptorSet( descriptorSet )
        , m_device( *device )
        , m_descriptorPool( *descriptorPool )
        , m_dispatcher( device.getDispatcher() )
      {}

      DescriptorSet( VkDescriptorSet                                                           descriptorSet,
                     VkDevice                                                                  device,
                     VkDescriptorPool                                                          descriptorPool,
                     VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * dispatcher )
        : m_descriptorSet( descriptorSet )
        , m_device( device )
        , m_descriptorPool( descriptorPool )
        , m_dispatcher( dispatcher )
      {}

      ~DescriptorSet()
      {
        if ( m_descriptorSet )
        {
          getDispatcher()->vkFreeDescriptorSets(
            m_device, m_descriptorPool, 1, reinterpret_cast<VkDescriptorSet const *>( &m_descriptorSet ) );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      DescriptorSet() = default;
#  else
      DescriptorSet()                                                         = delete;
#  endif
      DescriptorSet( DescriptorSet const & ) = delete;
      DescriptorSet( DescriptorSet && rhs ) VULKAN_HPP_NOEXCEPT
        : m_descriptorSet( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_descriptorSet, {} ) )
        , m_device( rhs.m_device )
        , m_descriptorPool( rhs.m_descriptorPool )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      DescriptorSet & operator=( DescriptorSet const & ) = delete;
      DescriptorSet & operator                           =( DescriptorSet && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_descriptorSet )
          {
            getDispatcher()->vkFreeDescriptorSets(
              m_device, m_descriptorPool, 1, reinterpret_cast<VkDescriptorSet const *>( &m_descriptorSet ) );
          }
          m_descriptorSet  = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_descriptorSet, {} );
          m_device         = rhs.m_device;
          m_descriptorPool = rhs.m_descriptorPool;
          m_dispatcher     = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::DescriptorSet const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_descriptorSet;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_descriptorSet.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_descriptorSet.operator!();
      }
#  endif

      //=== VK_VERSION_1_1 ===

      void updateWithTemplate( VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplate descriptorUpdateTemplate,
                               const void *                                   pData ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_KHR_descriptor_update_template ===

      void updateWithTemplateKHR( VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplate descriptorUpdateTemplate,
                                  const void *                                   pData ) const VULKAN_HPP_NOEXCEPT;

    private:
      VULKAN_HPP_NAMESPACE::DescriptorSet                                       m_descriptorSet;
      VkDevice                                                                  m_device;
      VkDescriptorPool                                                          m_descriptorPool;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class DescriptorSets : public std::vector<VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DescriptorSet>
    {
    public:
      DescriptorSets( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const & device,
                      VULKAN_HPP_NAMESPACE::DescriptorSetAllocateInfo const &         allocateInfo )
      {
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * dispatcher = device.getDispatcher();
        std::vector<VkDescriptorSet> descriptorSets( allocateInfo.descriptorSetCount );
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          dispatcher->vkAllocateDescriptorSets( static_cast<VkDevice>( *device ),
                                                reinterpret_cast<const VkDescriptorSetAllocateInfo *>( &allocateInfo ),
                                                descriptorSets.data() ) );
        if ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          this->reserve( allocateInfo.descriptorSetCount );
          for ( auto const & descriptorSet : descriptorSets )
          {
            this->emplace_back( descriptorSet,
                                static_cast<VkDevice>( *device ),
                                static_cast<VkDescriptorPool>( allocateInfo.descriptorPool ),
                                dispatcher );
          }
        }
        else
        {
          throwResultException( result, "vkAllocateDescriptorSets" );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      DescriptorSets() = default;
#  else
      DescriptorSets()                                                        = delete;
#  endif
      DescriptorSets( DescriptorSets const & ) = delete;
      DescriptorSets( DescriptorSets && rhs )  = default;
      DescriptorSets & operator=( DescriptorSets const & ) = delete;
      DescriptorSets & operator=( DescriptorSets && rhs ) = default;
    };

    class DescriptorSetLayout
    {
    public:
      using CType = VkDescriptorSetLayout;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eDescriptorSetLayout;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eDescriptorSetLayout;

    public:
      DescriptorSetLayout(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VULKAN_HPP_NAMESPACE::DescriptorSetLayoutCreateInfo const &                     createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateDescriptorSetLayout(
            static_cast<VkDevice>( *device ),
            reinterpret_cast<const VkDescriptorSetLayoutCreateInfo *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkDescriptorSetLayout *>( &m_descriptorSetLayout ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateDescriptorSetLayout" );
        }
      }

      DescriptorSetLayout(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VkDescriptorSetLayout                                                           descriptorSetLayout,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_descriptorSetLayout( descriptorSetLayout )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~DescriptorSetLayout()
      {
        if ( m_descriptorSetLayout )
        {
          getDispatcher()->vkDestroyDescriptorSetLayout(
            m_device, static_cast<VkDescriptorSetLayout>( m_descriptorSetLayout ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      DescriptorSetLayout() = default;
#  else
      DescriptorSetLayout()                                                   = delete;
#  endif
      DescriptorSetLayout( DescriptorSetLayout const & ) = delete;
      DescriptorSetLayout( DescriptorSetLayout && rhs ) VULKAN_HPP_NOEXCEPT
        : m_descriptorSetLayout( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_descriptorSetLayout,
                                                                                            {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      DescriptorSetLayout & operator=( DescriptorSetLayout const & ) = delete;
      DescriptorSetLayout & operator                                 =( DescriptorSetLayout && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_descriptorSetLayout )
          {
            getDispatcher()->vkDestroyDescriptorSetLayout(
              m_device, static_cast<VkDescriptorSetLayout>( m_descriptorSetLayout ), m_allocator );
          }
          m_descriptorSetLayout =
            VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_descriptorSetLayout, {} );
          m_device     = rhs.m_device;
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::DescriptorSetLayout const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_descriptorSetLayout;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_descriptorSetLayout.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_descriptorSetLayout.operator!();
      }
#  endif

    private:
      VULKAN_HPP_NAMESPACE::DescriptorSetLayout                                 m_descriptorSetLayout;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class DescriptorUpdateTemplate
    {
    public:
      using CType = VkDescriptorUpdateTemplate;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eDescriptorUpdateTemplate;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eDescriptorUpdateTemplate;

    public:
      DescriptorUpdateTemplate(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplateCreateInfo const &                createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateDescriptorUpdateTemplate(
            static_cast<VkDevice>( *device ),
            reinterpret_cast<const VkDescriptorUpdateTemplateCreateInfo *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkDescriptorUpdateTemplate *>( &m_descriptorUpdateTemplate ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateDescriptorUpdateTemplate" );
        }
      }

      DescriptorUpdateTemplate(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VkDescriptorUpdateTemplate                                                      descriptorUpdateTemplate,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_descriptorUpdateTemplate( descriptorUpdateTemplate )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~DescriptorUpdateTemplate()
      {
        if ( m_descriptorUpdateTemplate )
        {
          getDispatcher()->vkDestroyDescriptorUpdateTemplate(
            m_device, static_cast<VkDescriptorUpdateTemplate>( m_descriptorUpdateTemplate ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      DescriptorUpdateTemplate() = default;
#  else
      DescriptorUpdateTemplate()                                              = delete;
#  endif
      DescriptorUpdateTemplate( DescriptorUpdateTemplate const & ) = delete;
      DescriptorUpdateTemplate( DescriptorUpdateTemplate && rhs ) VULKAN_HPP_NOEXCEPT
        : m_descriptorUpdateTemplate(
            VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_descriptorUpdateTemplate, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      DescriptorUpdateTemplate & operator=( DescriptorUpdateTemplate const & ) = delete;
      DescriptorUpdateTemplate & operator=( DescriptorUpdateTemplate && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_descriptorUpdateTemplate )
          {
            getDispatcher()->vkDestroyDescriptorUpdateTemplate(
              m_device, static_cast<VkDescriptorUpdateTemplate>( m_descriptorUpdateTemplate ), m_allocator );
          }
          m_descriptorUpdateTemplate =
            VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_descriptorUpdateTemplate, {} );
          m_device     = rhs.m_device;
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplate const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_descriptorUpdateTemplate;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_descriptorUpdateTemplate.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_descriptorUpdateTemplate.operator!();
      }
#  endif

    private:
      VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplate                            m_descriptorUpdateTemplate;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class DeviceMemory
    {
    public:
      using CType = VkDeviceMemory;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eDeviceMemory;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eDeviceMemory;

    public:
      DeviceMemory(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VULKAN_HPP_NAMESPACE::MemoryAllocateInfo const &                                allocateInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkAllocateMemory( static_cast<VkDevice>( *device ),
                                             reinterpret_cast<const VkMemoryAllocateInfo *>( &allocateInfo ),
                                             m_allocator,
                                             reinterpret_cast<VkDeviceMemory *>( &m_deviceMemory ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkAllocateMemory" );
        }
      }

      DeviceMemory(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VkDeviceMemory                                                                  deviceMemory,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_deviceMemory( deviceMemory )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~DeviceMemory()
      {
        if ( m_deviceMemory )
        {
          getDispatcher()->vkFreeMemory( m_device, static_cast<VkDeviceMemory>( m_deviceMemory ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      DeviceMemory() = default;
#  else
      DeviceMemory()                                                          = delete;
#  endif
      DeviceMemory( DeviceMemory const & ) = delete;
      DeviceMemory( DeviceMemory && rhs ) VULKAN_HPP_NOEXCEPT
        : m_deviceMemory( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_deviceMemory, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      DeviceMemory & operator=( DeviceMemory const & ) = delete;
      DeviceMemory & operator                          =( DeviceMemory && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_deviceMemory )
          {
            getDispatcher()->vkFreeMemory( m_device, static_cast<VkDeviceMemory>( m_deviceMemory ), m_allocator );
          }
          m_deviceMemory = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_deviceMemory, {} );
          m_device       = rhs.m_device;
          m_allocator    = rhs.m_allocator;
          m_dispatcher   = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::DeviceMemory const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_deviceMemory;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_deviceMemory.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_deviceMemory.operator!();
      }
#  endif

      //=== VK_VERSION_1_0 ===

      VULKAN_HPP_NODISCARD void *
        mapMemory( VULKAN_HPP_NAMESPACE::DeviceSize     offset,
                   VULKAN_HPP_NAMESPACE::DeviceSize     size,
                   VULKAN_HPP_NAMESPACE::MemoryMapFlags flags VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT ) const;

      void unmapMemory() const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::DeviceSize getCommitment() const VULKAN_HPP_NOEXCEPT;

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_NV_external_memory_win32 ===

      VULKAN_HPP_NODISCARD HANDLE
                           getMemoryWin32HandleNV( VULKAN_HPP_NAMESPACE::ExternalMemoryHandleTypeFlagsNV handleType ) const;
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

    private:
      VULKAN_HPP_NAMESPACE::DeviceMemory                                        m_deviceMemory;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class DisplayKHR
    {
    public:
      using CType = VkDisplayKHR;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eDisplayKHR;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eDisplayKHR;

    public:
      DisplayKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::PhysicalDevice const & physicalDevice,
                  int32_t                                                                 drmFd,
                  uint32_t                                                                connectorId )
        : m_physicalDevice( *physicalDevice ), m_dispatcher( physicalDevice.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkGetDrmDisplayEXT( static_cast<VkPhysicalDevice>( *physicalDevice ),
                                               drmFd,
                                               connectorId,
                                               reinterpret_cast<VkDisplayKHR *>( &m_displayKHR ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkGetDrmDisplayEXT" );
        }
      }

#  if defined( VK_USE_PLATFORM_XLIB_XRANDR_EXT )
      DisplayKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::PhysicalDevice const & physicalDevice,
                  Display &                                                               dpy,
                  RROutput                                                                rrOutput )
        : m_physicalDevice( *physicalDevice ), m_dispatcher( physicalDevice.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkGetRandROutputDisplayEXT( static_cast<VkPhysicalDevice>( *physicalDevice ),
                                                       &dpy,
                                                       rrOutput,
                                                       reinterpret_cast<VkDisplayKHR *>( &m_displayKHR ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkGetRandROutputDisplayEXT" );
        }
      }
#  endif /*VK_USE_PLATFORM_XLIB_XRANDR_EXT*/

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
      DisplayKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::PhysicalDevice const & physicalDevice,
                  uint32_t                                                                deviceRelativeId )
        : m_physicalDevice( *physicalDevice ), m_dispatcher( physicalDevice.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkGetWinrtDisplayNV( static_cast<VkPhysicalDevice>( *physicalDevice ),
                                                deviceRelativeId,
                                                reinterpret_cast<VkDisplayKHR *>( &m_displayKHR ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkGetWinrtDisplayNV" );
        }
      }
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

      DisplayKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::PhysicalDevice const & physicalDevice,
                  VkDisplayKHR                                                            displayKHR )
        : m_displayKHR( displayKHR )
        , m_physicalDevice( *physicalDevice )
        , m_dispatcher( physicalDevice.getDispatcher() )
      {}

      DisplayKHR( VkDisplayKHR                                                                displayKHR,
                  VkPhysicalDevice                                                            physicalDevice,
                  VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::InstanceDispatcher const * dispatcher )
        : m_displayKHR( displayKHR ), m_physicalDevice( physicalDevice ), m_dispatcher( dispatcher )
      {}

      ~DisplayKHR()
      {
        if ( m_displayKHR )
        {
          getDispatcher()->vkReleaseDisplayEXT( m_physicalDevice, static_cast<VkDisplayKHR>( m_displayKHR ) );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      DisplayKHR() = default;
#  else
      DisplayKHR()                                                            = delete;
#  endif
      DisplayKHR( DisplayKHR const & ) = delete;
      DisplayKHR( DisplayKHR && rhs ) VULKAN_HPP_NOEXCEPT
        : m_displayKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_displayKHR, {} ) )
        , m_physicalDevice( rhs.m_physicalDevice )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      DisplayKHR & operator=( DisplayKHR const & ) = delete;
      DisplayKHR & operator                        =( DisplayKHR && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_displayKHR )
          {
            getDispatcher()->vkReleaseDisplayEXT( m_physicalDevice, static_cast<VkDisplayKHR>( m_displayKHR ) );
          }
          m_displayKHR     = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_displayKHR, {} );
          m_physicalDevice = rhs.m_physicalDevice;
          m_dispatcher     = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::DisplayKHR const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_displayKHR;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::InstanceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_displayKHR.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_displayKHR.operator!();
      }
#  endif

      //=== VK_KHR_display ===

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::DisplayModePropertiesKHR> getModeProperties() const;

      //=== VK_KHR_get_display_properties2 ===

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::DisplayModeProperties2KHR> getModeProperties2() const;

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_NV_acquire_winrt_display ===

      void acquireWinrtNV() const;
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

    private:
      VULKAN_HPP_NAMESPACE::DisplayKHR                                            m_displayKHR;
      VkPhysicalDevice                                                            m_physicalDevice;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::InstanceDispatcher const * m_dispatcher;
    };

    class DisplayKHRs : public std::vector<VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DisplayKHR>
    {
    public:
      DisplayKHRs( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::PhysicalDevice const & physicalDevice,
                   uint32_t                                                                planeIndex )
      {
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::InstanceDispatcher const * dispatcher =
          physicalDevice.getDispatcher();
        std::vector<VkDisplayKHR>    displays;
        uint32_t                     displayCount;
        VULKAN_HPP_NAMESPACE::Result result;
        do
        {
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>( dispatcher->vkGetDisplayPlaneSupportedDisplaysKHR(
            static_cast<VkPhysicalDevice>( *physicalDevice ), planeIndex, &displayCount, nullptr ) );
          if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && displayCount )
          {
            displays.resize( displayCount );
            result = static_cast<VULKAN_HPP_NAMESPACE::Result>( dispatcher->vkGetDisplayPlaneSupportedDisplaysKHR(
              static_cast<VkPhysicalDevice>( *physicalDevice ), planeIndex, &displayCount, displays.data() ) );
            VULKAN_HPP_ASSERT( displayCount <= displays.size() );
          }
        } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
        if ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          this->reserve( displayCount );
          for ( auto const & displayKHR : displays )
          {
            this->emplace_back( displayKHR, static_cast<VkPhysicalDevice>( *physicalDevice ), dispatcher );
          }
        }
        else
        {
          throwResultException( result, "vkGetDisplayPlaneSupportedDisplaysKHR" );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      DisplayKHRs() = default;
#  else
      DisplayKHRs()                                                           = delete;
#  endif
      DisplayKHRs( DisplayKHRs const & ) = delete;
      DisplayKHRs( DisplayKHRs && rhs )  = default;
      DisplayKHRs & operator=( DisplayKHRs const & ) = delete;
      DisplayKHRs & operator=( DisplayKHRs && rhs ) = default;
    };

    class DisplayModeKHR
    {
    public:
      using CType = VkDisplayModeKHR;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eDisplayModeKHR;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eDisplayModeKHR;

    public:
      DisplayModeKHR(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::PhysicalDevice const &         physicalDevice,
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DisplayKHR const &             display,
        VULKAN_HPP_NAMESPACE::DisplayModeCreateInfoKHR const &                          createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_physicalDevice( *physicalDevice ), m_dispatcher( physicalDevice.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateDisplayModeKHR(
            static_cast<VkPhysicalDevice>( *physicalDevice ),
            static_cast<VkDisplayKHR>( *display ),
            reinterpret_cast<const VkDisplayModeCreateInfoKHR *>( &createInfo ),
            reinterpret_cast<const VkAllocationCallbacks *>(
              static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ),
            reinterpret_cast<VkDisplayModeKHR *>( &m_displayModeKHR ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateDisplayModeKHR" );
        }
      }

      DisplayModeKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::PhysicalDevice const & physicalDevice,
                      VkDisplayModeKHR                                                        displayModeKHR )
        : m_displayModeKHR( displayModeKHR )
        , m_physicalDevice( *physicalDevice )
        , m_dispatcher( physicalDevice.getDispatcher() )
      {}

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      DisplayModeKHR() = default;
#  else
      DisplayModeKHR()                                                        = delete;
#  endif
      DisplayModeKHR( DisplayModeKHR const & ) = delete;
      DisplayModeKHR( DisplayModeKHR && rhs ) VULKAN_HPP_NOEXCEPT
        : m_displayModeKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_displayModeKHR, {} ) )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      DisplayModeKHR & operator=( DisplayModeKHR const & ) = delete;
      DisplayModeKHR & operator                            =( DisplayModeKHR && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          m_displayModeKHR = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_displayModeKHR, {} );
          m_dispatcher     = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::DisplayModeKHR const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_displayModeKHR;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::InstanceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_displayModeKHR.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_displayModeKHR.operator!();
      }
#  endif

      //=== VK_KHR_display ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::DisplayPlaneCapabilitiesKHR
                           getDisplayPlaneCapabilities( uint32_t planeIndex ) const;

    private:
      VULKAN_HPP_NAMESPACE::DisplayModeKHR                                        m_displayModeKHR;
      VULKAN_HPP_NAMESPACE::PhysicalDevice                                        m_physicalDevice;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::InstanceDispatcher const * m_dispatcher;
    };

    class Event
    {
    public:
      using CType = VkEvent;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eEvent;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eEvent;

    public:
      Event( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
             VULKAN_HPP_NAMESPACE::EventCreateInfo const &                                   createInfo,
             VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkCreateEvent( static_cast<VkDevice>( *device ),
                                          reinterpret_cast<const VkEventCreateInfo *>( &createInfo ),
                                          m_allocator,
                                          reinterpret_cast<VkEvent *>( &m_event ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateEvent" );
        }
      }

      Event( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
             VkEvent                                                                         event,
             VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_event( event )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~Event()
      {
        if ( m_event )
        {
          getDispatcher()->vkDestroyEvent( m_device, static_cast<VkEvent>( m_event ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      Event() = default;
#  else
      Event()                                                                 = delete;
#  endif
      Event( Event const & ) = delete;
      Event( Event && rhs ) VULKAN_HPP_NOEXCEPT
        : m_event( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_event, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      Event & operator=( Event const & ) = delete;
      Event & operator                   =( Event && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_event )
          {
            getDispatcher()->vkDestroyEvent( m_device, static_cast<VkEvent>( m_event ), m_allocator );
          }
          m_event      = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_event, {} );
          m_device     = rhs.m_device;
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::Event const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_event;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_event.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_event.operator!();
      }
#  endif

      //=== VK_VERSION_1_0 ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::Result getStatus() const;

      void set() const;

      void reset() const;

    private:
      VULKAN_HPP_NAMESPACE::Event                                               m_event;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class Fence
    {
    public:
      using CType = VkFence;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eFence;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eFence;

    public:
      Fence( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
             VULKAN_HPP_NAMESPACE::FenceCreateInfo const &                                   createInfo,
             VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkCreateFence( static_cast<VkDevice>( *device ),
                                          reinterpret_cast<const VkFenceCreateInfo *>( &createInfo ),
                                          m_allocator,
                                          reinterpret_cast<VkFence *>( &m_fence ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateFence" );
        }
      }

      Fence( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
             VULKAN_HPP_NAMESPACE::DeviceEventInfoEXT const &                                deviceEventInfo,
             VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkRegisterDeviceEventEXT( static_cast<VkDevice>( *device ),
                                                     reinterpret_cast<const VkDeviceEventInfoEXT *>( &deviceEventInfo ),
                                                     m_allocator,
                                                     reinterpret_cast<VkFence *>( &m_fence ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkRegisterDeviceEventEXT" );
        }
      }

      Fence( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
             VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DisplayKHR const &             display,
             VULKAN_HPP_NAMESPACE::DisplayEventInfoEXT const &                               displayEventInfo,
             VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkRegisterDisplayEventEXT(
            static_cast<VkDevice>( *device ),
            static_cast<VkDisplayKHR>( *display ),
            reinterpret_cast<const VkDisplayEventInfoEXT *>( &displayEventInfo ),
            m_allocator,
            reinterpret_cast<VkFence *>( &m_fence ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkRegisterDisplayEventEXT" );
        }
      }

      Fence( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
             VkFence                                                                         fence,
             VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_fence( fence )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~Fence()
      {
        if ( m_fence )
        {
          getDispatcher()->vkDestroyFence( m_device, static_cast<VkFence>( m_fence ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      Fence() = default;
#  else
      Fence()                                                                 = delete;
#  endif
      Fence( Fence const & ) = delete;
      Fence( Fence && rhs ) VULKAN_HPP_NOEXCEPT
        : m_fence( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_fence, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      Fence & operator=( Fence const & ) = delete;
      Fence & operator                   =( Fence && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_fence )
          {
            getDispatcher()->vkDestroyFence( m_device, static_cast<VkFence>( m_fence ), m_allocator );
          }
          m_fence      = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_fence, {} );
          m_device     = rhs.m_device;
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::Fence const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_fence;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_fence.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_fence.operator!();
      }
#  endif

      //=== VK_VERSION_1_0 ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::Result getStatus() const;

    private:
      VULKAN_HPP_NAMESPACE::Fence                                               m_fence;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class Framebuffer
    {
    public:
      using CType = VkFramebuffer;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eFramebuffer;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eFramebuffer;

    public:
      Framebuffer( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
                   VULKAN_HPP_NAMESPACE::FramebufferCreateInfo const &                             createInfo,
                   VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkCreateFramebuffer( static_cast<VkDevice>( *device ),
                                                reinterpret_cast<const VkFramebufferCreateInfo *>( &createInfo ),
                                                m_allocator,
                                                reinterpret_cast<VkFramebuffer *>( &m_framebuffer ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateFramebuffer" );
        }
      }

      Framebuffer( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
                   VkFramebuffer                                                                   framebuffer,
                   VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_framebuffer( framebuffer )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~Framebuffer()
      {
        if ( m_framebuffer )
        {
          getDispatcher()->vkDestroyFramebuffer( m_device, static_cast<VkFramebuffer>( m_framebuffer ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      Framebuffer() = default;
#  else
      Framebuffer()                                                           = delete;
#  endif
      Framebuffer( Framebuffer const & ) = delete;
      Framebuffer( Framebuffer && rhs ) VULKAN_HPP_NOEXCEPT
        : m_framebuffer( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_framebuffer, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      Framebuffer & operator=( Framebuffer const & ) = delete;
      Framebuffer & operator                         =( Framebuffer && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_framebuffer )
          {
            getDispatcher()->vkDestroyFramebuffer( m_device, static_cast<VkFramebuffer>( m_framebuffer ), m_allocator );
          }
          m_framebuffer = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_framebuffer, {} );
          m_device      = rhs.m_device;
          m_allocator   = rhs.m_allocator;
          m_dispatcher  = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::Framebuffer const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_framebuffer;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_framebuffer.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_framebuffer.operator!();
      }
#  endif

    private:
      VULKAN_HPP_NAMESPACE::Framebuffer                                         m_framebuffer;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class Image
    {
    public:
      using CType = VkImage;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eImage;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eImage;

    public:
      Image( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
             VULKAN_HPP_NAMESPACE::ImageCreateInfo const &                                   createInfo,
             VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkCreateImage( static_cast<VkDevice>( *device ),
                                          reinterpret_cast<const VkImageCreateInfo *>( &createInfo ),
                                          m_allocator,
                                          reinterpret_cast<VkImage *>( &m_image ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateImage" );
        }
      }

      Image( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
             VkImage                                                                         image,
             VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_image( image )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~Image()
      {
        if ( m_image )
        {
          getDispatcher()->vkDestroyImage( m_device, static_cast<VkImage>( m_image ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      Image() = default;
#  else
      Image()                                                                 = delete;
#  endif
      Image( Image const & ) = delete;
      Image( Image && rhs ) VULKAN_HPP_NOEXCEPT
        : m_image( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_image, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      Image & operator=( Image const & ) = delete;
      Image & operator                   =( Image && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_image )
          {
            getDispatcher()->vkDestroyImage( m_device, static_cast<VkImage>( m_image ), m_allocator );
          }
          m_image      = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_image, {} );
          m_device     = rhs.m_device;
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::Image const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_image;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_image.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_image.operator!();
      }
#  endif

      //=== VK_VERSION_1_0 ===

      void bindMemory( VULKAN_HPP_NAMESPACE::DeviceMemory memory, VULKAN_HPP_NAMESPACE::DeviceSize memoryOffset ) const;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::MemoryRequirements getMemoryRequirements() const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::SparseImageMemoryRequirements>
                           getSparseMemoryRequirements() const VULKAN_HPP_NOEXCEPT;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::SubresourceLayout
                           getSubresourceLayout( const ImageSubresource & subresource ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_EXT_image_drm_format_modifier ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::ImageDrmFormatModifierPropertiesEXT
                           getDrmFormatModifierPropertiesEXT() const;

    private:
      VULKAN_HPP_NAMESPACE::Image                                               m_image;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class ImageView
    {
    public:
      using CType = VkImageView;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eImageView;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eImageView;

    public:
      ImageView( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
                 VULKAN_HPP_NAMESPACE::ImageViewCreateInfo const &                               createInfo,
                 VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkCreateImageView( static_cast<VkDevice>( *device ),
                                              reinterpret_cast<const VkImageViewCreateInfo *>( &createInfo ),
                                              m_allocator,
                                              reinterpret_cast<VkImageView *>( &m_imageView ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateImageView" );
        }
      }

      ImageView( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
                 VkImageView                                                                     imageView,
                 VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_imageView( imageView )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~ImageView()
      {
        if ( m_imageView )
        {
          getDispatcher()->vkDestroyImageView( m_device, static_cast<VkImageView>( m_imageView ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      ImageView() = default;
#  else
      ImageView()                                                             = delete;
#  endif
      ImageView( ImageView const & ) = delete;
      ImageView( ImageView && rhs ) VULKAN_HPP_NOEXCEPT
        : m_imageView( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_imageView, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      ImageView & operator=( ImageView const & ) = delete;
      ImageView & operator                       =( ImageView && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_imageView )
          {
            getDispatcher()->vkDestroyImageView( m_device, static_cast<VkImageView>( m_imageView ), m_allocator );
          }
          m_imageView  = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_imageView, {} );
          m_device     = rhs.m_device;
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::ImageView const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_imageView;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_imageView.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_imageView.operator!();
      }
#  endif

      //=== VK_NVX_image_view_handle ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::ImageViewAddressPropertiesNVX getAddressNVX() const;

    private:
      VULKAN_HPP_NAMESPACE::ImageView                                           m_imageView;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class IndirectCommandsLayoutNV
    {
    public:
      using CType = VkIndirectCommandsLayoutNV;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eIndirectCommandsLayoutNV;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eUnknown;

    public:
      IndirectCommandsLayoutNV(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutCreateInfoNV const &                createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateIndirectCommandsLayoutNV(
            static_cast<VkDevice>( *device ),
            reinterpret_cast<const VkIndirectCommandsLayoutCreateInfoNV *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkIndirectCommandsLayoutNV *>( &m_indirectCommandsLayoutNV ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateIndirectCommandsLayoutNV" );
        }
      }

      IndirectCommandsLayoutNV(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VkIndirectCommandsLayoutNV                                                      indirectCommandsLayoutNV,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_indirectCommandsLayoutNV( indirectCommandsLayoutNV )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~IndirectCommandsLayoutNV()
      {
        if ( m_indirectCommandsLayoutNV )
        {
          getDispatcher()->vkDestroyIndirectCommandsLayoutNV(
            m_device, static_cast<VkIndirectCommandsLayoutNV>( m_indirectCommandsLayoutNV ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      IndirectCommandsLayoutNV() = default;
#  else
      IndirectCommandsLayoutNV()                                              = delete;
#  endif
      IndirectCommandsLayoutNV( IndirectCommandsLayoutNV const & ) = delete;
      IndirectCommandsLayoutNV( IndirectCommandsLayoutNV && rhs ) VULKAN_HPP_NOEXCEPT
        : m_indirectCommandsLayoutNV(
            VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_indirectCommandsLayoutNV, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      IndirectCommandsLayoutNV & operator=( IndirectCommandsLayoutNV const & ) = delete;
      IndirectCommandsLayoutNV & operator=( IndirectCommandsLayoutNV && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_indirectCommandsLayoutNV )
          {
            getDispatcher()->vkDestroyIndirectCommandsLayoutNV(
              m_device, static_cast<VkIndirectCommandsLayoutNV>( m_indirectCommandsLayoutNV ), m_allocator );
          }
          m_indirectCommandsLayoutNV =
            VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_indirectCommandsLayoutNV, {} );
          m_device     = rhs.m_device;
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutNV const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_indirectCommandsLayoutNV;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_indirectCommandsLayoutNV.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_indirectCommandsLayoutNV.operator!();
      }
#  endif

    private:
      VULKAN_HPP_NAMESPACE::IndirectCommandsLayoutNV                            m_indirectCommandsLayoutNV;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class PerformanceConfigurationINTEL
    {
    public:
      using CType = VkPerformanceConfigurationINTEL;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::ePerformanceConfigurationINTEL;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eUnknown;

    public:
      PerformanceConfigurationINTEL(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &        device,
        VULKAN_HPP_NAMESPACE::PerformanceConfigurationAcquireInfoINTEL const & acquireInfo )
        : m_device( *device ), m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkAcquirePerformanceConfigurationINTEL(
            static_cast<VkDevice>( *device ),
            reinterpret_cast<const VkPerformanceConfigurationAcquireInfoINTEL *>( &acquireInfo ),
            reinterpret_cast<VkPerformanceConfigurationINTEL *>( &m_performanceConfigurationINTEL ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkAcquirePerformanceConfigurationINTEL" );
        }
      }

      PerformanceConfigurationINTEL( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const & device,
                                     VkPerformanceConfigurationINTEL performanceConfigurationINTEL )
        : m_performanceConfigurationINTEL( performanceConfigurationINTEL )
        , m_device( *device )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~PerformanceConfigurationINTEL()
      {
        if ( m_performanceConfigurationINTEL )
        {
          getDispatcher()->vkReleasePerformanceConfigurationINTEL(
            m_device, static_cast<VkPerformanceConfigurationINTEL>( m_performanceConfigurationINTEL ) );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      PerformanceConfigurationINTEL() = default;
#  else
      PerformanceConfigurationINTEL()                                         = delete;
#  endif
      PerformanceConfigurationINTEL( PerformanceConfigurationINTEL const & ) = delete;
      PerformanceConfigurationINTEL( PerformanceConfigurationINTEL && rhs ) VULKAN_HPP_NOEXCEPT
        : m_performanceConfigurationINTEL(
            VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_performanceConfigurationINTEL, {} ) )
        , m_device( rhs.m_device )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      PerformanceConfigurationINTEL & operator=( PerformanceConfigurationINTEL const & ) = delete;
      PerformanceConfigurationINTEL & operator=( PerformanceConfigurationINTEL && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_performanceConfigurationINTEL )
          {
            getDispatcher()->vkReleasePerformanceConfigurationINTEL(
              m_device, static_cast<VkPerformanceConfigurationINTEL>( m_performanceConfigurationINTEL ) );
          }
          m_performanceConfigurationINTEL =
            VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_performanceConfigurationINTEL, {} );
          m_device     = rhs.m_device;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::PerformanceConfigurationINTEL const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_performanceConfigurationINTEL;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_performanceConfigurationINTEL.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_performanceConfigurationINTEL.operator!();
      }
#  endif

    private:
      VULKAN_HPP_NAMESPACE::PerformanceConfigurationINTEL                       m_performanceConfigurationINTEL;
      VkDevice                                                                  m_device;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class PipelineCache
    {
    public:
      using CType = VkPipelineCache;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::ePipelineCache;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::ePipelineCache;

    public:
      PipelineCache(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VULKAN_HPP_NAMESPACE::PipelineCacheCreateInfo const &                           createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkCreatePipelineCache( static_cast<VkDevice>( *device ),
                                                  reinterpret_cast<const VkPipelineCacheCreateInfo *>( &createInfo ),
                                                  m_allocator,
                                                  reinterpret_cast<VkPipelineCache *>( &m_pipelineCache ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreatePipelineCache" );
        }
      }

      PipelineCache(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VkPipelineCache                                                                 pipelineCache,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_pipelineCache( pipelineCache )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~PipelineCache()
      {
        if ( m_pipelineCache )
        {
          getDispatcher()->vkDestroyPipelineCache(
            m_device, static_cast<VkPipelineCache>( m_pipelineCache ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      PipelineCache() = default;
#  else
      PipelineCache()                                                         = delete;
#  endif
      PipelineCache( PipelineCache const & ) = delete;
      PipelineCache( PipelineCache && rhs ) VULKAN_HPP_NOEXCEPT
        : m_pipelineCache( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_pipelineCache, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      PipelineCache & operator=( PipelineCache const & ) = delete;
      PipelineCache & operator                           =( PipelineCache && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_pipelineCache )
          {
            getDispatcher()->vkDestroyPipelineCache(
              m_device, static_cast<VkPipelineCache>( m_pipelineCache ), m_allocator );
          }
          m_pipelineCache = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_pipelineCache, {} );
          m_device        = rhs.m_device;
          m_allocator     = rhs.m_allocator;
          m_dispatcher    = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::PipelineCache const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_pipelineCache;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_pipelineCache.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_pipelineCache.operator!();
      }
#  endif

      //=== VK_VERSION_1_0 ===

      VULKAN_HPP_NODISCARD std::vector<uint8_t> getData() const;

      void merge( ArrayProxy<const VULKAN_HPP_NAMESPACE::PipelineCache> const & srcCaches ) const;

    private:
      VULKAN_HPP_NAMESPACE::PipelineCache                                       m_pipelineCache;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class Pipeline
    {
    public:
      using CType = VkPipeline;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::ePipeline;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::ePipeline;

    public:
      Pipeline(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const & device,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::PipelineCache> const &
                                                                                        pipelineCache,
        VULKAN_HPP_NAMESPACE::ComputePipelineCreateInfo const &                         createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        m_constructorSuccessCode = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateComputePipelines(
          static_cast<VkDevice>( *device ),
          pipelineCache ? static_cast<VkPipelineCache>( **pipelineCache ) : 0,
          1,
          reinterpret_cast<const VkComputePipelineCreateInfo *>( &createInfo ),
          m_allocator,
          reinterpret_cast<VkPipeline *>( &m_pipeline ) ) );
        if ( ( m_constructorSuccessCode != VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
             ( m_constructorSuccessCode != VULKAN_HPP_NAMESPACE::Result::ePipelineCompileRequiredEXT ) )
        {
          throwResultException( m_constructorSuccessCode, "vkCreateComputePipelines" );
        }
      }

      Pipeline(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const & device,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::PipelineCache> const &
                                                                                        pipelineCache,
        VULKAN_HPP_NAMESPACE::GraphicsPipelineCreateInfo const &                        createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        m_constructorSuccessCode =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateGraphicsPipelines(
            static_cast<VkDevice>( *device ),
            pipelineCache ? static_cast<VkPipelineCache>( **pipelineCache ) : 0,
            1,
            reinterpret_cast<const VkGraphicsPipelineCreateInfo *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkPipeline *>( &m_pipeline ) ) );
        if ( ( m_constructorSuccessCode != VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
             ( m_constructorSuccessCode != VULKAN_HPP_NAMESPACE::Result::ePipelineCompileRequiredEXT ) )
        {
          throwResultException( m_constructorSuccessCode, "vkCreateGraphicsPipelines" );
        }
      }

      Pipeline(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const & device,
        VULKAN_HPP_NAMESPACE::Optional<
          const VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeferredOperationKHR> const & deferredOperation,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::PipelineCache> const &
                                                                                        pipelineCache,
        VULKAN_HPP_NAMESPACE::RayTracingPipelineCreateInfoKHR const &                   createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        m_constructorSuccessCode =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateRayTracingPipelinesKHR(
            static_cast<VkDevice>( *device ),
            deferredOperation ? static_cast<VkDeferredOperationKHR>( **deferredOperation ) : 0,
            pipelineCache ? static_cast<VkPipelineCache>( **pipelineCache ) : 0,
            1,
            reinterpret_cast<const VkRayTracingPipelineCreateInfoKHR *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkPipeline *>( &m_pipeline ) ) );
        if ( ( m_constructorSuccessCode != VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
             ( m_constructorSuccessCode != VULKAN_HPP_NAMESPACE::Result::eOperationDeferredKHR ) &&
             ( m_constructorSuccessCode != VULKAN_HPP_NAMESPACE::Result::eOperationNotDeferredKHR ) &&
             ( m_constructorSuccessCode != VULKAN_HPP_NAMESPACE::Result::ePipelineCompileRequiredEXT ) )
        {
          throwResultException( m_constructorSuccessCode, "vkCreateRayTracingPipelinesKHR" );
        }
      }

      Pipeline(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const & device,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::PipelineCache> const &
                                                                                        pipelineCache,
        VULKAN_HPP_NAMESPACE::RayTracingPipelineCreateInfoNV const &                    createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        m_constructorSuccessCode =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateRayTracingPipelinesNV(
            static_cast<VkDevice>( *device ),
            pipelineCache ? static_cast<VkPipelineCache>( **pipelineCache ) : 0,
            1,
            reinterpret_cast<const VkRayTracingPipelineCreateInfoNV *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkPipeline *>( &m_pipeline ) ) );
        if ( ( m_constructorSuccessCode != VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
             ( m_constructorSuccessCode != VULKAN_HPP_NAMESPACE::Result::ePipelineCompileRequiredEXT ) )
        {
          throwResultException( m_constructorSuccessCode, "vkCreateRayTracingPipelinesNV" );
        }
      }

      Pipeline( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
                VkPipeline                                                                      pipeline,
                VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_pipeline( pipeline )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      Pipeline( VkPipeline                                                                pipeline,
                VkDevice                                                                  device,
                VkAllocationCallbacks const *                                             allocator,
                VULKAN_HPP_NAMESPACE::Result                                              successCode,
                VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * dispatcher )
        : m_pipeline( pipeline )
        , m_device( device )
        , m_allocator( allocator )
        , m_constructorSuccessCode( successCode )
        , m_dispatcher( dispatcher )
      {}

      ~Pipeline()
      {
        if ( m_pipeline )
        {
          getDispatcher()->vkDestroyPipeline( m_device, static_cast<VkPipeline>( m_pipeline ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      Pipeline() = default;
#  else
      Pipeline()                                                              = delete;
#  endif
      Pipeline( Pipeline const & ) = delete;
      Pipeline( Pipeline && rhs ) VULKAN_HPP_NOEXCEPT
        : m_pipeline( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_pipeline, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      Pipeline & operator=( Pipeline const & ) = delete;
      Pipeline & operator                      =( Pipeline && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_pipeline )
          {
            getDispatcher()->vkDestroyPipeline( m_device, static_cast<VkPipeline>( m_pipeline ), m_allocator );
          }
          m_pipeline   = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_pipeline, {} );
          m_device     = rhs.m_device;
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::Pipeline const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_pipeline;
      }

      VULKAN_HPP_NAMESPACE::Result getConstructorSuccessCode() const
      {
        return m_constructorSuccessCode;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_pipeline.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_pipeline.operator!();
      }
#  endif

      //=== VK_AMD_shader_info ===

      VULKAN_HPP_NODISCARD std::vector<uint8_t>
                           getShaderInfoAMD( VULKAN_HPP_NAMESPACE::ShaderStageFlagBits shaderStage,
                                             VULKAN_HPP_NAMESPACE::ShaderInfoTypeAMD   infoType ) const;

      //=== VK_NV_ray_tracing ===

      template <typename T>
      VULKAN_HPP_NODISCARD std::vector<T>
                           getRayTracingShaderGroupHandlesNV( uint32_t firstGroup, uint32_t groupCount, size_t dataSize ) const;

      template <typename T>
      VULKAN_HPP_NODISCARD T getRayTracingShaderGroupHandleNV( uint32_t firstGroup, uint32_t groupCount ) const;

      void compileDeferredNV( uint32_t shader ) const;

      //=== VK_KHR_ray_tracing_pipeline ===

      template <typename T>
      VULKAN_HPP_NODISCARD std::vector<T>
                           getRayTracingShaderGroupHandlesKHR( uint32_t firstGroup, uint32_t groupCount, size_t dataSize ) const;

      template <typename T>
      VULKAN_HPP_NODISCARD T getRayTracingShaderGroupHandleKHR( uint32_t firstGroup, uint32_t groupCount ) const;

      template <typename T>
      VULKAN_HPP_NODISCARD std::vector<T> getRayTracingCaptureReplayShaderGroupHandlesKHR( uint32_t firstGroup,
                                                                                           uint32_t groupCount,
                                                                                           size_t   dataSize ) const;

      template <typename T>
      VULKAN_HPP_NODISCARD T getRayTracingCaptureReplayShaderGroupHandleKHR( uint32_t firstGroup,
                                                                             uint32_t groupCount ) const;

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::DeviceSize getRayTracingShaderGroupStackSizeKHR(
        uint32_t group, VULKAN_HPP_NAMESPACE::ShaderGroupShaderKHR groupShader ) const VULKAN_HPP_NOEXCEPT;

    private:
      VULKAN_HPP_NAMESPACE::Pipeline                                            m_pipeline;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::Result                                              m_constructorSuccessCode;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class Pipelines : public std::vector<VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Pipeline>
    {
    public:
      Pipelines(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const & device,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::PipelineCache> const &
                                                                                                  pipelineCache,
        VULKAN_HPP_NAMESPACE::ArrayProxy<VULKAN_HPP_NAMESPACE::ComputePipelineCreateInfo> const & createInfos,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks>           allocator = nullptr )
      {
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * dispatcher = device.getDispatcher();
        std::vector<VkPipeline>                                                   pipelines( createInfos.size() );
        VULKAN_HPP_NAMESPACE::Result                                              result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( dispatcher->vkCreateComputePipelines(
            static_cast<VkDevice>( *device ),
            pipelineCache ? static_cast<VkPipelineCache>( **pipelineCache ) : 0,
            createInfos.size(),
            reinterpret_cast<const VkComputePipelineCreateInfo *>( createInfos.data() ),
            reinterpret_cast<const VkAllocationCallbacks *>(
              static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ),
            pipelines.data() ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) ||
             ( result == VULKAN_HPP_NAMESPACE::Result::ePipelineCompileRequiredEXT ) )
        {
          this->reserve( createInfos.size() );
          for ( auto const & pipeline : pipelines )
          {
            this->emplace_back( pipeline,
                                static_cast<VkDevice>( *device ),
                                reinterpret_cast<const VkAllocationCallbacks *>(
                                  static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ),
                                result,
                                dispatcher );
          }
        }
        else
        {
          throwResultException( result, "vkCreateComputePipelines" );
        }
      }

      Pipelines(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const & device,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::PipelineCache> const &
                                                                                                   pipelineCache,
        VULKAN_HPP_NAMESPACE::ArrayProxy<VULKAN_HPP_NAMESPACE::GraphicsPipelineCreateInfo> const & createInfos,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks>            allocator = nullptr )
      {
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * dispatcher = device.getDispatcher();
        std::vector<VkPipeline>                                                   pipelines( createInfos.size() );
        VULKAN_HPP_NAMESPACE::Result                                              result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( dispatcher->vkCreateGraphicsPipelines(
            static_cast<VkDevice>( *device ),
            pipelineCache ? static_cast<VkPipelineCache>( **pipelineCache ) : 0,
            createInfos.size(),
            reinterpret_cast<const VkGraphicsPipelineCreateInfo *>( createInfos.data() ),
            reinterpret_cast<const VkAllocationCallbacks *>(
              static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ),
            pipelines.data() ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) ||
             ( result == VULKAN_HPP_NAMESPACE::Result::ePipelineCompileRequiredEXT ) )
        {
          this->reserve( createInfos.size() );
          for ( auto const & pipeline : pipelines )
          {
            this->emplace_back( pipeline,
                                static_cast<VkDevice>( *device ),
                                reinterpret_cast<const VkAllocationCallbacks *>(
                                  static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ),
                                result,
                                dispatcher );
          }
        }
        else
        {
          throwResultException( result, "vkCreateGraphicsPipelines" );
        }
      }

      Pipelines(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const & device,
        VULKAN_HPP_NAMESPACE::Optional<
          const VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeferredOperationKHR> const & deferredOperation,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::PipelineCache> const &
                                                                                                        pipelineCache,
        VULKAN_HPP_NAMESPACE::ArrayProxy<VULKAN_HPP_NAMESPACE::RayTracingPipelineCreateInfoKHR> const & createInfos,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
      {
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * dispatcher = device.getDispatcher();
        std::vector<VkPipeline>                                                   pipelines( createInfos.size() );
        VULKAN_HPP_NAMESPACE::Result                                              result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( dispatcher->vkCreateRayTracingPipelinesKHR(
            static_cast<VkDevice>( *device ),
            deferredOperation ? static_cast<VkDeferredOperationKHR>( **deferredOperation ) : 0,
            pipelineCache ? static_cast<VkPipelineCache>( **pipelineCache ) : 0,
            createInfos.size(),
            reinterpret_cast<const VkRayTracingPipelineCreateInfoKHR *>( createInfos.data() ),
            reinterpret_cast<const VkAllocationCallbacks *>(
              static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ),
            pipelines.data() ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) ||
             ( result == VULKAN_HPP_NAMESPACE::Result::eOperationDeferredKHR ) ||
             ( result == VULKAN_HPP_NAMESPACE::Result::eOperationNotDeferredKHR ) ||
             ( result == VULKAN_HPP_NAMESPACE::Result::ePipelineCompileRequiredEXT ) )
        {
          this->reserve( createInfos.size() );
          for ( auto const & pipeline : pipelines )
          {
            this->emplace_back( pipeline,
                                static_cast<VkDevice>( *device ),
                                reinterpret_cast<const VkAllocationCallbacks *>(
                                  static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ),
                                result,
                                dispatcher );
          }
        }
        else
        {
          throwResultException( result, "vkCreateRayTracingPipelinesKHR" );
        }
      }

      Pipelines(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const & device,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::PipelineCache> const &
                                                                                                       pipelineCache,
        VULKAN_HPP_NAMESPACE::ArrayProxy<VULKAN_HPP_NAMESPACE::RayTracingPipelineCreateInfoNV> const & createInfos,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
      {
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * dispatcher = device.getDispatcher();
        std::vector<VkPipeline>                                                   pipelines( createInfos.size() );
        VULKAN_HPP_NAMESPACE::Result                                              result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( dispatcher->vkCreateRayTracingPipelinesNV(
            static_cast<VkDevice>( *device ),
            pipelineCache ? static_cast<VkPipelineCache>( **pipelineCache ) : 0,
            createInfos.size(),
            reinterpret_cast<const VkRayTracingPipelineCreateInfoNV *>( createInfos.data() ),
            reinterpret_cast<const VkAllocationCallbacks *>(
              static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ),
            pipelines.data() ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) ||
             ( result == VULKAN_HPP_NAMESPACE::Result::ePipelineCompileRequiredEXT ) )
        {
          this->reserve( createInfos.size() );
          for ( auto const & pipeline : pipelines )
          {
            this->emplace_back( pipeline,
                                static_cast<VkDevice>( *device ),
                                reinterpret_cast<const VkAllocationCallbacks *>(
                                  static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ),
                                result,
                                dispatcher );
          }
        }
        else
        {
          throwResultException( result, "vkCreateRayTracingPipelinesNV" );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      Pipelines() = default;
#  else
      Pipelines()                                                             = delete;
#  endif
      Pipelines( Pipelines const & ) = delete;
      Pipelines( Pipelines && rhs )  = default;
      Pipelines & operator=( Pipelines const & ) = delete;
      Pipelines & operator=( Pipelines && rhs ) = default;
    };

    class PipelineLayout
    {
    public:
      using CType = VkPipelineLayout;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::ePipelineLayout;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::ePipelineLayout;

    public:
      PipelineLayout(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VULKAN_HPP_NAMESPACE::PipelineLayoutCreateInfo const &                          createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkCreatePipelineLayout( static_cast<VkDevice>( *device ),
                                                   reinterpret_cast<const VkPipelineLayoutCreateInfo *>( &createInfo ),
                                                   m_allocator,
                                                   reinterpret_cast<VkPipelineLayout *>( &m_pipelineLayout ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreatePipelineLayout" );
        }
      }

      PipelineLayout(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VkPipelineLayout                                                                pipelineLayout,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_pipelineLayout( pipelineLayout )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~PipelineLayout()
      {
        if ( m_pipelineLayout )
        {
          getDispatcher()->vkDestroyPipelineLayout(
            m_device, static_cast<VkPipelineLayout>( m_pipelineLayout ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      PipelineLayout() = default;
#  else
      PipelineLayout()                                                        = delete;
#  endif
      PipelineLayout( PipelineLayout const & ) = delete;
      PipelineLayout( PipelineLayout && rhs ) VULKAN_HPP_NOEXCEPT
        : m_pipelineLayout( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_pipelineLayout, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      PipelineLayout & operator=( PipelineLayout const & ) = delete;
      PipelineLayout & operator                            =( PipelineLayout && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_pipelineLayout )
          {
            getDispatcher()->vkDestroyPipelineLayout(
              m_device, static_cast<VkPipelineLayout>( m_pipelineLayout ), m_allocator );
          }
          m_pipelineLayout = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_pipelineLayout, {} );
          m_device         = rhs.m_device;
          m_allocator      = rhs.m_allocator;
          m_dispatcher     = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::PipelineLayout const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_pipelineLayout;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_pipelineLayout.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_pipelineLayout.operator!();
      }
#  endif

    private:
      VULKAN_HPP_NAMESPACE::PipelineLayout                                      m_pipelineLayout;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class PrivateDataSlotEXT
    {
    public:
      using CType = VkPrivateDataSlotEXT;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::ePrivateDataSlotEXT;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eUnknown;

    public:
      PrivateDataSlotEXT(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VULKAN_HPP_NAMESPACE::PrivateDataSlotCreateInfoEXT const &                      createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreatePrivateDataSlotEXT(
            static_cast<VkDevice>( *device ),
            reinterpret_cast<const VkPrivateDataSlotCreateInfoEXT *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkPrivateDataSlotEXT *>( &m_privateDataSlotEXT ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreatePrivateDataSlotEXT" );
        }
      }

      PrivateDataSlotEXT(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VkPrivateDataSlotEXT                                                            privateDataSlotEXT,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_privateDataSlotEXT( privateDataSlotEXT )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~PrivateDataSlotEXT()
      {
        if ( m_privateDataSlotEXT )
        {
          getDispatcher()->vkDestroyPrivateDataSlotEXT(
            m_device, static_cast<VkPrivateDataSlotEXT>( m_privateDataSlotEXT ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      PrivateDataSlotEXT() = default;
#  else
      PrivateDataSlotEXT()                                                    = delete;
#  endif
      PrivateDataSlotEXT( PrivateDataSlotEXT const & ) = delete;
      PrivateDataSlotEXT( PrivateDataSlotEXT && rhs ) VULKAN_HPP_NOEXCEPT
        : m_privateDataSlotEXT( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_privateDataSlotEXT,
                                                                                           {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      PrivateDataSlotEXT & operator=( PrivateDataSlotEXT const & ) = delete;
      PrivateDataSlotEXT & operator                                =( PrivateDataSlotEXT && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_privateDataSlotEXT )
          {
            getDispatcher()->vkDestroyPrivateDataSlotEXT(
              m_device, static_cast<VkPrivateDataSlotEXT>( m_privateDataSlotEXT ), m_allocator );
          }
          m_privateDataSlotEXT =
            VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_privateDataSlotEXT, {} );
          m_device     = rhs.m_device;
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::PrivateDataSlotEXT const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_privateDataSlotEXT;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_privateDataSlotEXT.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_privateDataSlotEXT.operator!();
      }
#  endif

    private:
      VULKAN_HPP_NAMESPACE::PrivateDataSlotEXT                                  m_privateDataSlotEXT;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class QueryPool
    {
    public:
      using CType = VkQueryPool;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eQueryPool;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eQueryPool;

    public:
      QueryPool( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
                 VULKAN_HPP_NAMESPACE::QueryPoolCreateInfo const &                               createInfo,
                 VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkCreateQueryPool( static_cast<VkDevice>( *device ),
                                              reinterpret_cast<const VkQueryPoolCreateInfo *>( &createInfo ),
                                              m_allocator,
                                              reinterpret_cast<VkQueryPool *>( &m_queryPool ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateQueryPool" );
        }
      }

      QueryPool( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
                 VkQueryPool                                                                     queryPool,
                 VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_queryPool( queryPool )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~QueryPool()
      {
        if ( m_queryPool )
        {
          getDispatcher()->vkDestroyQueryPool( m_device, static_cast<VkQueryPool>( m_queryPool ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      QueryPool() = default;
#  else
      QueryPool()                                                             = delete;
#  endif
      QueryPool( QueryPool const & ) = delete;
      QueryPool( QueryPool && rhs ) VULKAN_HPP_NOEXCEPT
        : m_queryPool( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_queryPool, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      QueryPool & operator=( QueryPool const & ) = delete;
      QueryPool & operator                       =( QueryPool && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_queryPool )
          {
            getDispatcher()->vkDestroyQueryPool( m_device, static_cast<VkQueryPool>( m_queryPool ), m_allocator );
          }
          m_queryPool  = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_queryPool, {} );
          m_device     = rhs.m_device;
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::QueryPool const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_queryPool;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_queryPool.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_queryPool.operator!();
      }
#  endif

      //=== VK_VERSION_1_0 ===

      template <typename T>
      VULKAN_HPP_NODISCARD std::pair<VULKAN_HPP_NAMESPACE::Result, std::vector<T>>
                           getResults( uint32_t                               firstQuery,
                                       uint32_t                               queryCount,
                                       size_t                                 dataSize,
                                       VULKAN_HPP_NAMESPACE::DeviceSize       stride,
                                       VULKAN_HPP_NAMESPACE::QueryResultFlags flags VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT ) const;

      template <typename T>
      VULKAN_HPP_NODISCARD std::pair<VULKAN_HPP_NAMESPACE::Result, T>
                           getResult( uint32_t                               firstQuery,
                                      uint32_t                               queryCount,
                                      VULKAN_HPP_NAMESPACE::DeviceSize       stride,
                                      VULKAN_HPP_NAMESPACE::QueryResultFlags flags VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT ) const;

      //=== VK_VERSION_1_2 ===

      void reset( uint32_t firstQuery, uint32_t queryCount ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_EXT_host_query_reset ===

      void resetEXT( uint32_t firstQuery, uint32_t queryCount ) const VULKAN_HPP_NOEXCEPT;

    private:
      VULKAN_HPP_NAMESPACE::QueryPool                                           m_queryPool;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class Queue
    {
    public:
      using CType = VkQueue;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eQueue;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eQueue;

    public:
      Queue( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const & device,
             uint32_t                                                        queueFamilyIndex,
             uint32_t                                                        queueIndex )
        : m_dispatcher( device.getDispatcher() )
      {
        getDispatcher()->vkGetDeviceQueue(
          static_cast<VkDevice>( *device ), queueFamilyIndex, queueIndex, reinterpret_cast<VkQueue *>( &m_queue ) );
      }

      Queue( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const & device,
             VULKAN_HPP_NAMESPACE::DeviceQueueInfo2 const &                  queueInfo )
        : m_dispatcher( device.getDispatcher() )
      {
        getDispatcher()->vkGetDeviceQueue2( static_cast<VkDevice>( *device ),
                                            reinterpret_cast<const VkDeviceQueueInfo2 *>( &queueInfo ),
                                            reinterpret_cast<VkQueue *>( &m_queue ) );
      }

      Queue( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const & device, VkQueue queue )
        : m_queue( queue ), m_dispatcher( device.getDispatcher() )
      {}

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      Queue() = default;
#  else
      Queue()                                                                 = delete;
#  endif
      Queue( Queue const & ) = delete;
      Queue( Queue && rhs ) VULKAN_HPP_NOEXCEPT
        : m_queue( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_queue, {} ) )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      Queue & operator=( Queue const & ) = delete;
      Queue & operator                   =( Queue && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          m_queue      = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_queue, {} );
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::Queue const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_queue;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_queue.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_queue.operator!();
      }
#  endif

      //=== VK_VERSION_1_0 ===

      void submit( ArrayProxy<const VULKAN_HPP_NAMESPACE::SubmitInfo> const & submits,
                   VULKAN_HPP_NAMESPACE::Fence fence VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT ) const;

      void waitIdle() const;

      void bindSparse( ArrayProxy<const VULKAN_HPP_NAMESPACE::BindSparseInfo> const & bindInfo,
                       VULKAN_HPP_NAMESPACE::Fence fence VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT ) const;

      //=== VK_KHR_swapchain ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::Result presentKHR( const PresentInfoKHR & presentInfo ) const;

      //=== VK_EXT_debug_utils ===

      void beginDebugUtilsLabelEXT( const DebugUtilsLabelEXT & labelInfo ) const VULKAN_HPP_NOEXCEPT;

      void endDebugUtilsLabelEXT() const VULKAN_HPP_NOEXCEPT;

      void insertDebugUtilsLabelEXT( const DebugUtilsLabelEXT & labelInfo ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_NV_device_diagnostic_checkpoints ===

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::CheckpointDataNV>
                           getCheckpointDataNV() const VULKAN_HPP_NOEXCEPT;

      //=== VK_INTEL_performance_query ===

      void setPerformanceConfigurationINTEL( VULKAN_HPP_NAMESPACE::PerformanceConfigurationINTEL configuration ) const;

      //=== VK_KHR_synchronization2 ===

      void submit2KHR( ArrayProxy<const VULKAN_HPP_NAMESPACE::SubmitInfo2KHR> const & submits,
                       VULKAN_HPP_NAMESPACE::Fence fence VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT ) const;

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::CheckpointData2NV>
                           getCheckpointData2NV() const VULKAN_HPP_NOEXCEPT;

    private:
      VULKAN_HPP_NAMESPACE::Queue                                               m_queue;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class RenderPass
    {
    public:
      using CType = VkRenderPass;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eRenderPass;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eRenderPass;

    public:
      RenderPass( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
                  VULKAN_HPP_NAMESPACE::RenderPassCreateInfo const &                              createInfo,
                  VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkCreateRenderPass( static_cast<VkDevice>( *device ),
                                               reinterpret_cast<const VkRenderPassCreateInfo *>( &createInfo ),
                                               m_allocator,
                                               reinterpret_cast<VkRenderPass *>( &m_renderPass ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateRenderPass" );
        }
      }

      RenderPass( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
                  VULKAN_HPP_NAMESPACE::RenderPassCreateInfo2 const &                             createInfo,
                  VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkCreateRenderPass2( static_cast<VkDevice>( *device ),
                                                reinterpret_cast<const VkRenderPassCreateInfo2 *>( &createInfo ),
                                                m_allocator,
                                                reinterpret_cast<VkRenderPass *>( &m_renderPass ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateRenderPass2" );
        }
      }

      RenderPass( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
                  VkRenderPass                                                                    renderPass,
                  VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_renderPass( renderPass )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~RenderPass()
      {
        if ( m_renderPass )
        {
          getDispatcher()->vkDestroyRenderPass( m_device, static_cast<VkRenderPass>( m_renderPass ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      RenderPass() = default;
#  else
      RenderPass()                                                            = delete;
#  endif
      RenderPass( RenderPass const & ) = delete;
      RenderPass( RenderPass && rhs ) VULKAN_HPP_NOEXCEPT
        : m_renderPass( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_renderPass, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      RenderPass & operator=( RenderPass const & ) = delete;
      RenderPass & operator                        =( RenderPass && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_renderPass )
          {
            getDispatcher()->vkDestroyRenderPass( m_device, static_cast<VkRenderPass>( m_renderPass ), m_allocator );
          }
          m_renderPass = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_renderPass, {} );
          m_device     = rhs.m_device;
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::RenderPass const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_renderPass;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_renderPass.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_renderPass.operator!();
      }
#  endif

      //=== VK_VERSION_1_0 ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::Extent2D getRenderAreaGranularity() const VULKAN_HPP_NOEXCEPT;

      //=== VK_HUAWEI_subpass_shading ===

      VULKAN_HPP_NODISCARD std::pair<VULKAN_HPP_NAMESPACE::Result, VULKAN_HPP_NAMESPACE::Extent2D>
                           getSubpassShadingMaxWorkgroupSizeHUAWEI() const;

    private:
      VULKAN_HPP_NAMESPACE::RenderPass                                          m_renderPass;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class Sampler
    {
    public:
      using CType = VkSampler;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eSampler;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eSampler;

    public:
      Sampler( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
               VULKAN_HPP_NAMESPACE::SamplerCreateInfo const &                                 createInfo,
               VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkCreateSampler( static_cast<VkDevice>( *device ),
                                            reinterpret_cast<const VkSamplerCreateInfo *>( &createInfo ),
                                            m_allocator,
                                            reinterpret_cast<VkSampler *>( &m_sampler ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateSampler" );
        }
      }

      Sampler( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
               VkSampler                                                                       sampler,
               VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_sampler( sampler )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~Sampler()
      {
        if ( m_sampler )
        {
          getDispatcher()->vkDestroySampler( m_device, static_cast<VkSampler>( m_sampler ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      Sampler() = default;
#  else
      Sampler()                                                               = delete;
#  endif
      Sampler( Sampler const & ) = delete;
      Sampler( Sampler && rhs ) VULKAN_HPP_NOEXCEPT
        : m_sampler( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_sampler, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      Sampler & operator=( Sampler const & ) = delete;
      Sampler & operator                     =( Sampler && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_sampler )
          {
            getDispatcher()->vkDestroySampler( m_device, static_cast<VkSampler>( m_sampler ), m_allocator );
          }
          m_sampler    = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_sampler, {} );
          m_device     = rhs.m_device;
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::Sampler const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_sampler;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_sampler.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_sampler.operator!();
      }
#  endif

    private:
      VULKAN_HPP_NAMESPACE::Sampler                                             m_sampler;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class SamplerYcbcrConversion
    {
    public:
      using CType = VkSamplerYcbcrConversion;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eSamplerYcbcrConversion;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eSamplerYcbcrConversion;

    public:
      SamplerYcbcrConversion(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VULKAN_HPP_NAMESPACE::SamplerYcbcrConversionCreateInfo const &                  createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateSamplerYcbcrConversion(
            static_cast<VkDevice>( *device ),
            reinterpret_cast<const VkSamplerYcbcrConversionCreateInfo *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkSamplerYcbcrConversion *>( &m_samplerYcbcrConversion ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateSamplerYcbcrConversion" );
        }
      }

      SamplerYcbcrConversion(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VkSamplerYcbcrConversion                                                        samplerYcbcrConversion,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_samplerYcbcrConversion( samplerYcbcrConversion )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~SamplerYcbcrConversion()
      {
        if ( m_samplerYcbcrConversion )
        {
          getDispatcher()->vkDestroySamplerYcbcrConversion(
            m_device, static_cast<VkSamplerYcbcrConversion>( m_samplerYcbcrConversion ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      SamplerYcbcrConversion() = default;
#  else
      SamplerYcbcrConversion()                                                = delete;
#  endif
      SamplerYcbcrConversion( SamplerYcbcrConversion const & ) = delete;
      SamplerYcbcrConversion( SamplerYcbcrConversion && rhs ) VULKAN_HPP_NOEXCEPT
        : m_samplerYcbcrConversion(
            VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_samplerYcbcrConversion, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      SamplerYcbcrConversion & operator=( SamplerYcbcrConversion const & ) = delete;
      SamplerYcbcrConversion & operator=( SamplerYcbcrConversion && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_samplerYcbcrConversion )
          {
            getDispatcher()->vkDestroySamplerYcbcrConversion(
              m_device, static_cast<VkSamplerYcbcrConversion>( m_samplerYcbcrConversion ), m_allocator );
          }
          m_samplerYcbcrConversion =
            VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_samplerYcbcrConversion, {} );
          m_device     = rhs.m_device;
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::SamplerYcbcrConversion const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_samplerYcbcrConversion;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_samplerYcbcrConversion.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_samplerYcbcrConversion.operator!();
      }
#  endif

    private:
      VULKAN_HPP_NAMESPACE::SamplerYcbcrConversion                              m_samplerYcbcrConversion;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class Semaphore
    {
    public:
      using CType = VkSemaphore;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eSemaphore;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eSemaphore;

    public:
      Semaphore( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
                 VULKAN_HPP_NAMESPACE::SemaphoreCreateInfo const &                               createInfo,
                 VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkCreateSemaphore( static_cast<VkDevice>( *device ),
                                              reinterpret_cast<const VkSemaphoreCreateInfo *>( &createInfo ),
                                              m_allocator,
                                              reinterpret_cast<VkSemaphore *>( &m_semaphore ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateSemaphore" );
        }
      }

      Semaphore( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
                 VkSemaphore                                                                     semaphore,
                 VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_semaphore( semaphore )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~Semaphore()
      {
        if ( m_semaphore )
        {
          getDispatcher()->vkDestroySemaphore( m_device, static_cast<VkSemaphore>( m_semaphore ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      Semaphore() = default;
#  else
      Semaphore()                                                             = delete;
#  endif
      Semaphore( Semaphore const & ) = delete;
      Semaphore( Semaphore && rhs ) VULKAN_HPP_NOEXCEPT
        : m_semaphore( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_semaphore, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      Semaphore & operator=( Semaphore const & ) = delete;
      Semaphore & operator                       =( Semaphore && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_semaphore )
          {
            getDispatcher()->vkDestroySemaphore( m_device, static_cast<VkSemaphore>( m_semaphore ), m_allocator );
          }
          m_semaphore  = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_semaphore, {} );
          m_device     = rhs.m_device;
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::Semaphore const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_semaphore;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_semaphore.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_semaphore.operator!();
      }
#  endif

      //=== VK_VERSION_1_2 ===

      VULKAN_HPP_NODISCARD uint64_t getCounterValue() const;

      //=== VK_KHR_timeline_semaphore ===

      VULKAN_HPP_NODISCARD uint64_t getCounterValueKHR() const;

    private:
      VULKAN_HPP_NAMESPACE::Semaphore                                           m_semaphore;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class ShaderModule
    {
    public:
      using CType = VkShaderModule;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eShaderModule;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eShaderModule;

    public:
      ShaderModule(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VULKAN_HPP_NAMESPACE::ShaderModuleCreateInfo const &                            createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkCreateShaderModule( static_cast<VkDevice>( *device ),
                                                 reinterpret_cast<const VkShaderModuleCreateInfo *>( &createInfo ),
                                                 m_allocator,
                                                 reinterpret_cast<VkShaderModule *>( &m_shaderModule ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateShaderModule" );
        }
      }

      ShaderModule(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VkShaderModule                                                                  shaderModule,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_shaderModule( shaderModule )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~ShaderModule()
      {
        if ( m_shaderModule )
        {
          getDispatcher()->vkDestroyShaderModule(
            m_device, static_cast<VkShaderModule>( m_shaderModule ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      ShaderModule() = default;
#  else
      ShaderModule()                                                          = delete;
#  endif
      ShaderModule( ShaderModule const & ) = delete;
      ShaderModule( ShaderModule && rhs ) VULKAN_HPP_NOEXCEPT
        : m_shaderModule( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_shaderModule, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      ShaderModule & operator=( ShaderModule const & ) = delete;
      ShaderModule & operator                          =( ShaderModule && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_shaderModule )
          {
            getDispatcher()->vkDestroyShaderModule(
              m_device, static_cast<VkShaderModule>( m_shaderModule ), m_allocator );
          }
          m_shaderModule = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_shaderModule, {} );
          m_device       = rhs.m_device;
          m_allocator    = rhs.m_allocator;
          m_dispatcher   = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::ShaderModule const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_shaderModule;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_shaderModule.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_shaderModule.operator!();
      }
#  endif

    private:
      VULKAN_HPP_NAMESPACE::ShaderModule                                        m_shaderModule;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class SurfaceKHR
    {
    public:
      using CType = VkSurfaceKHR;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eSurfaceKHR;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eSurfaceKHR;

    public:
#  if defined( VK_USE_PLATFORM_ANDROID_KHR )
      SurfaceKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Instance const &               instance,
                  VULKAN_HPP_NAMESPACE::AndroidSurfaceCreateInfoKHR const &                       createInfo,
                  VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_instance( *instance )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( instance.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateAndroidSurfaceKHR(
            static_cast<VkInstance>( *instance ),
            reinterpret_cast<const VkAndroidSurfaceCreateInfoKHR *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkSurfaceKHR *>( &m_surfaceKHR ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateAndroidSurfaceKHR" );
        }
      }
#  endif /*VK_USE_PLATFORM_ANDROID_KHR*/

#  if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
      SurfaceKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Instance const &               instance,
                  VULKAN_HPP_NAMESPACE::DirectFBSurfaceCreateInfoEXT const &                      createInfo,
                  VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_instance( *instance )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( instance.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateDirectFBSurfaceEXT(
            static_cast<VkInstance>( *instance ),
            reinterpret_cast<const VkDirectFBSurfaceCreateInfoEXT *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkSurfaceKHR *>( &m_surfaceKHR ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateDirectFBSurfaceEXT" );
        }
      }
#  endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

      SurfaceKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Instance const &               instance,
                  VULKAN_HPP_NAMESPACE::DisplaySurfaceCreateInfoKHR const &                       createInfo,
                  VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_instance( *instance )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( instance.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateDisplayPlaneSurfaceKHR(
            static_cast<VkInstance>( *instance ),
            reinterpret_cast<const VkDisplaySurfaceCreateInfoKHR *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkSurfaceKHR *>( &m_surfaceKHR ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateDisplayPlaneSurfaceKHR" );
        }
      }

      SurfaceKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Instance const &               instance,
                  VULKAN_HPP_NAMESPACE::HeadlessSurfaceCreateInfoEXT const &                      createInfo,
                  VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_instance( *instance )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( instance.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateHeadlessSurfaceEXT(
            static_cast<VkInstance>( *instance ),
            reinterpret_cast<const VkHeadlessSurfaceCreateInfoEXT *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkSurfaceKHR *>( &m_surfaceKHR ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateHeadlessSurfaceEXT" );
        }
      }

#  if defined( VK_USE_PLATFORM_IOS_MVK )
      SurfaceKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Instance const &               instance,
                  VULKAN_HPP_NAMESPACE::IOSSurfaceCreateInfoMVK const &                           createInfo,
                  VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_instance( *instance )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( instance.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkCreateIOSSurfaceMVK( static_cast<VkInstance>( *instance ),
                                                  reinterpret_cast<const VkIOSSurfaceCreateInfoMVK *>( &createInfo ),
                                                  m_allocator,
                                                  reinterpret_cast<VkSurfaceKHR *>( &m_surfaceKHR ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateIOSSurfaceMVK" );
        }
      }
#  endif /*VK_USE_PLATFORM_IOS_MVK*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
      SurfaceKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Instance const &               instance,
                  VULKAN_HPP_NAMESPACE::ImagePipeSurfaceCreateInfoFUCHSIA const &                 createInfo,
                  VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_instance( *instance )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( instance.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateImagePipeSurfaceFUCHSIA(
            static_cast<VkInstance>( *instance ),
            reinterpret_cast<const VkImagePipeSurfaceCreateInfoFUCHSIA *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkSurfaceKHR *>( &m_surfaceKHR ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateImagePipeSurfaceFUCHSIA" );
        }
      }
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

#  if defined( VK_USE_PLATFORM_MACOS_MVK )
      SurfaceKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Instance const &               instance,
                  VULKAN_HPP_NAMESPACE::MacOSSurfaceCreateInfoMVK const &                         createInfo,
                  VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_instance( *instance )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( instance.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateMacOSSurfaceMVK(
            static_cast<VkInstance>( *instance ),
            reinterpret_cast<const VkMacOSSurfaceCreateInfoMVK *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkSurfaceKHR *>( &m_surfaceKHR ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateMacOSSurfaceMVK" );
        }
      }
#  endif /*VK_USE_PLATFORM_MACOS_MVK*/

#  if defined( VK_USE_PLATFORM_METAL_EXT )
      SurfaceKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Instance const &               instance,
                  VULKAN_HPP_NAMESPACE::MetalSurfaceCreateInfoEXT const &                         createInfo,
                  VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_instance( *instance )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( instance.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateMetalSurfaceEXT(
            static_cast<VkInstance>( *instance ),
            reinterpret_cast<const VkMetalSurfaceCreateInfoEXT *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkSurfaceKHR *>( &m_surfaceKHR ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateMetalSurfaceEXT" );
        }
      }
#  endif /*VK_USE_PLATFORM_METAL_EXT*/

#  if defined( VK_USE_PLATFORM_SCREEN_QNX )
      SurfaceKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Instance const &               instance,
                  VULKAN_HPP_NAMESPACE::ScreenSurfaceCreateInfoQNX const &                        createInfo,
                  VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_instance( *instance )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( instance.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateScreenSurfaceQNX(
            static_cast<VkInstance>( *instance ),
            reinterpret_cast<const VkScreenSurfaceCreateInfoQNX *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkSurfaceKHR *>( &m_surfaceKHR ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateScreenSurfaceQNX" );
        }
      }
#  endif /*VK_USE_PLATFORM_SCREEN_QNX*/

#  if defined( VK_USE_PLATFORM_GGP )
      SurfaceKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Instance const &               instance,
                  VULKAN_HPP_NAMESPACE::StreamDescriptorSurfaceCreateInfoGGP const &              createInfo,
                  VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_instance( *instance )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( instance.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateStreamDescriptorSurfaceGGP(
            static_cast<VkInstance>( *instance ),
            reinterpret_cast<const VkStreamDescriptorSurfaceCreateInfoGGP *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkSurfaceKHR *>( &m_surfaceKHR ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateStreamDescriptorSurfaceGGP" );
        }
      }
#  endif /*VK_USE_PLATFORM_GGP*/

#  if defined( VK_USE_PLATFORM_VI_NN )
      SurfaceKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Instance const &               instance,
                  VULKAN_HPP_NAMESPACE::ViSurfaceCreateInfoNN const &                             createInfo,
                  VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_instance( *instance )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( instance.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkCreateViSurfaceNN( static_cast<VkInstance>( *instance ),
                                                reinterpret_cast<const VkViSurfaceCreateInfoNN *>( &createInfo ),
                                                m_allocator,
                                                reinterpret_cast<VkSurfaceKHR *>( &m_surfaceKHR ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateViSurfaceNN" );
        }
      }
#  endif /*VK_USE_PLATFORM_VI_NN*/

#  if defined( VK_USE_PLATFORM_WAYLAND_KHR )
      SurfaceKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Instance const &               instance,
                  VULKAN_HPP_NAMESPACE::WaylandSurfaceCreateInfoKHR const &                       createInfo,
                  VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_instance( *instance )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( instance.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateWaylandSurfaceKHR(
            static_cast<VkInstance>( *instance ),
            reinterpret_cast<const VkWaylandSurfaceCreateInfoKHR *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkSurfaceKHR *>( &m_surfaceKHR ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateWaylandSurfaceKHR" );
        }
      }
#  endif /*VK_USE_PLATFORM_WAYLAND_KHR*/

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
      SurfaceKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Instance const &               instance,
                  VULKAN_HPP_NAMESPACE::Win32SurfaceCreateInfoKHR const &                         createInfo,
                  VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_instance( *instance )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( instance.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateWin32SurfaceKHR(
            static_cast<VkInstance>( *instance ),
            reinterpret_cast<const VkWin32SurfaceCreateInfoKHR *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkSurfaceKHR *>( &m_surfaceKHR ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateWin32SurfaceKHR" );
        }
      }
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

#  if defined( VK_USE_PLATFORM_XCB_KHR )
      SurfaceKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Instance const &               instance,
                  VULKAN_HPP_NAMESPACE::XcbSurfaceCreateInfoKHR const &                           createInfo,
                  VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_instance( *instance )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( instance.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkCreateXcbSurfaceKHR( static_cast<VkInstance>( *instance ),
                                                  reinterpret_cast<const VkXcbSurfaceCreateInfoKHR *>( &createInfo ),
                                                  m_allocator,
                                                  reinterpret_cast<VkSurfaceKHR *>( &m_surfaceKHR ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateXcbSurfaceKHR" );
        }
      }
#  endif /*VK_USE_PLATFORM_XCB_KHR*/

#  if defined( VK_USE_PLATFORM_XLIB_KHR )
      SurfaceKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Instance const &               instance,
                  VULKAN_HPP_NAMESPACE::XlibSurfaceCreateInfoKHR const &                          createInfo,
                  VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_instance( *instance )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( instance.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkCreateXlibSurfaceKHR( static_cast<VkInstance>( *instance ),
                                                   reinterpret_cast<const VkXlibSurfaceCreateInfoKHR *>( &createInfo ),
                                                   m_allocator,
                                                   reinterpret_cast<VkSurfaceKHR *>( &m_surfaceKHR ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateXlibSurfaceKHR" );
        }
      }
#  endif /*VK_USE_PLATFORM_XLIB_KHR*/

      SurfaceKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Instance const &               instance,
                  VkSurfaceKHR                                                                    surfaceKHR,
                  VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_surfaceKHR( surfaceKHR )
        , m_instance( *instance )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( instance.getDispatcher() )
      {}

      ~SurfaceKHR()
      {
        if ( m_surfaceKHR )
        {
          getDispatcher()->vkDestroySurfaceKHR( m_instance, static_cast<VkSurfaceKHR>( m_surfaceKHR ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      SurfaceKHR() = default;
#  else
      SurfaceKHR()                                                            = delete;
#  endif
      SurfaceKHR( SurfaceKHR const & ) = delete;
      SurfaceKHR( SurfaceKHR && rhs ) VULKAN_HPP_NOEXCEPT
        : m_surfaceKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_surfaceKHR, {} ) )
        , m_instance( rhs.m_instance )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      SurfaceKHR & operator=( SurfaceKHR const & ) = delete;
      SurfaceKHR & operator                        =( SurfaceKHR && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_surfaceKHR )
          {
            getDispatcher()->vkDestroySurfaceKHR( m_instance, static_cast<VkSurfaceKHR>( m_surfaceKHR ), m_allocator );
          }
          m_surfaceKHR = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_surfaceKHR, {} );
          m_instance   = rhs.m_instance;
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::SurfaceKHR const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_surfaceKHR;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::InstanceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_surfaceKHR.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_surfaceKHR.operator!();
      }
#  endif

    private:
      VULKAN_HPP_NAMESPACE::SurfaceKHR                                            m_surfaceKHR;
      VkInstance                                                                  m_instance;
      const VkAllocationCallbacks *                                               m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::InstanceDispatcher const * m_dispatcher;
    };

    class SwapchainKHR
    {
    public:
      using CType = VkSwapchainKHR;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eSwapchainKHR;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eSwapchainKHR;

    public:
      SwapchainKHR(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VULKAN_HPP_NAMESPACE::SwapchainCreateInfoKHR const &                            createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkCreateSwapchainKHR( static_cast<VkDevice>( *device ),
                                                 reinterpret_cast<const VkSwapchainCreateInfoKHR *>( &createInfo ),
                                                 m_allocator,
                                                 reinterpret_cast<VkSwapchainKHR *>( &m_swapchainKHR ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateSwapchainKHR" );
        }
      }

      SwapchainKHR(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VkSwapchainKHR                                                                  swapchainKHR,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_swapchainKHR( swapchainKHR )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      SwapchainKHR( VkSwapchainKHR                                                            swapchainKHR,
                    VkDevice                                                                  device,
                    VkAllocationCallbacks const *                                             allocator,
                    VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * dispatcher )
        : m_swapchainKHR( swapchainKHR ), m_device( device ), m_allocator( allocator ), m_dispatcher( dispatcher )
      {}

      ~SwapchainKHR()
      {
        if ( m_swapchainKHR )
        {
          getDispatcher()->vkDestroySwapchainKHR(
            m_device, static_cast<VkSwapchainKHR>( m_swapchainKHR ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      SwapchainKHR() = default;
#  else
      SwapchainKHR()                                                          = delete;
#  endif
      SwapchainKHR( SwapchainKHR const & ) = delete;
      SwapchainKHR( SwapchainKHR && rhs ) VULKAN_HPP_NOEXCEPT
        : m_swapchainKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_swapchainKHR, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      SwapchainKHR & operator=( SwapchainKHR const & ) = delete;
      SwapchainKHR & operator                          =( SwapchainKHR && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_swapchainKHR )
          {
            getDispatcher()->vkDestroySwapchainKHR(
              m_device, static_cast<VkSwapchainKHR>( m_swapchainKHR ), m_allocator );
          }
          m_swapchainKHR = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_swapchainKHR, {} );
          m_device       = rhs.m_device;
          m_allocator    = rhs.m_allocator;
          m_dispatcher   = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::SwapchainKHR const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_swapchainKHR;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_swapchainKHR.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_swapchainKHR.operator!();
      }
#  endif

      //=== VK_KHR_swapchain ===

      VULKAN_HPP_NODISCARD std::vector<VkImage> getImages() const;

      VULKAN_HPP_NODISCARD std::pair<VULKAN_HPP_NAMESPACE::Result, uint32_t>
                           acquireNextImage( uint64_t                        timeout,
                                             VULKAN_HPP_NAMESPACE::Semaphore semaphore VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT,
                                             VULKAN_HPP_NAMESPACE::Fence fence VULKAN_HPP_DEFAULT_ARGUMENT_ASSIGNMENT ) const;

      //=== VK_EXT_display_control ===

      VULKAN_HPP_NODISCARD uint64_t getCounterEXT( VULKAN_HPP_NAMESPACE::SurfaceCounterFlagBitsEXT counter ) const;

      //=== VK_GOOGLE_display_timing ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::RefreshCycleDurationGOOGLE getRefreshCycleDurationGOOGLE() const;

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::PastPresentationTimingGOOGLE>
                           getPastPresentationTimingGOOGLE() const;

      //=== VK_KHR_shared_presentable_image ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::Result getStatus() const;

      //=== VK_AMD_display_native_hdr ===

      void setLocalDimmingAMD( VULKAN_HPP_NAMESPACE::Bool32 localDimmingEnable ) const VULKAN_HPP_NOEXCEPT;

      //=== VK_KHR_present_wait ===

      VULKAN_HPP_NODISCARD VULKAN_HPP_NAMESPACE::Result waitForPresent( uint64_t presentId, uint64_t timeout ) const;

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
      //=== VK_EXT_full_screen_exclusive ===

      void acquireFullScreenExclusiveModeEXT() const;

      void releaseFullScreenExclusiveModeEXT() const;
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

    private:
      VULKAN_HPP_NAMESPACE::SwapchainKHR                                        m_swapchainKHR;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

    class SwapchainKHRs : public std::vector<VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::SwapchainKHR>
    {
    public:
      SwapchainKHRs(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                        device,
        VULKAN_HPP_NAMESPACE::ArrayProxy<VULKAN_HPP_NAMESPACE::SwapchainCreateInfoKHR> const & createInfos,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks>        allocator = nullptr )
      {
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * dispatcher = device.getDispatcher();
        std::vector<VkSwapchainKHR>                                               swapchains( createInfos.size() );
        VULKAN_HPP_NAMESPACE::Result                                              result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( dispatcher->vkCreateSharedSwapchainsKHR(
            static_cast<VkDevice>( *device ),
            createInfos.size(),
            reinterpret_cast<const VkSwapchainCreateInfoKHR *>( createInfos.data() ),
            reinterpret_cast<const VkAllocationCallbacks *>(
              static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ),
            swapchains.data() ) );
        if ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          this->reserve( createInfos.size() );
          for ( auto const & swapchainKHR : swapchains )
          {
            this->emplace_back( swapchainKHR,
                                static_cast<VkDevice>( *device ),
                                reinterpret_cast<const VkAllocationCallbacks *>(
                                  static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ),
                                dispatcher );
          }
        }
        else
        {
          throwResultException( result, "vkCreateSharedSwapchainsKHR" );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      SwapchainKHRs() = default;
#  else
      SwapchainKHRs()                                                         = delete;
#  endif
      SwapchainKHRs( SwapchainKHRs const & ) = delete;
      SwapchainKHRs( SwapchainKHRs && rhs )  = default;
      SwapchainKHRs & operator=( SwapchainKHRs const & ) = delete;
      SwapchainKHRs & operator=( SwapchainKHRs && rhs ) = default;
    };

    class ValidationCacheEXT
    {
    public:
      using CType = VkValidationCacheEXT;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eValidationCacheEXT;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eValidationCacheEXT;

    public:
      ValidationCacheEXT(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VULKAN_HPP_NAMESPACE::ValidationCacheCreateInfoEXT const &                      createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateValidationCacheEXT(
            static_cast<VkDevice>( *device ),
            reinterpret_cast<const VkValidationCacheCreateInfoEXT *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkValidationCacheEXT *>( &m_validationCacheEXT ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateValidationCacheEXT" );
        }
      }

      ValidationCacheEXT(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VkValidationCacheEXT                                                            validationCacheEXT,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_validationCacheEXT( validationCacheEXT )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~ValidationCacheEXT()
      {
        if ( m_validationCacheEXT )
        {
          getDispatcher()->vkDestroyValidationCacheEXT(
            m_device, static_cast<VkValidationCacheEXT>( m_validationCacheEXT ), m_allocator );
        }
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      ValidationCacheEXT() = default;
#  else
      ValidationCacheEXT()                                                    = delete;
#  endif
      ValidationCacheEXT( ValidationCacheEXT const & ) = delete;
      ValidationCacheEXT( ValidationCacheEXT && rhs ) VULKAN_HPP_NOEXCEPT
        : m_validationCacheEXT( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_validationCacheEXT,
                                                                                           {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      ValidationCacheEXT & operator=( ValidationCacheEXT const & ) = delete;
      ValidationCacheEXT & operator                                =( ValidationCacheEXT && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_validationCacheEXT )
          {
            getDispatcher()->vkDestroyValidationCacheEXT(
              m_device, static_cast<VkValidationCacheEXT>( m_validationCacheEXT ), m_allocator );
          }
          m_validationCacheEXT =
            VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_validationCacheEXT, {} );
          m_device     = rhs.m_device;
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::ValidationCacheEXT const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_validationCacheEXT;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#  if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_validationCacheEXT.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_validationCacheEXT.operator!();
      }
#  endif

      //=== VK_EXT_validation_cache ===

      void merge( ArrayProxy<const VULKAN_HPP_NAMESPACE::ValidationCacheEXT> const & srcCaches ) const;

      VULKAN_HPP_NODISCARD std::vector<uint8_t> getData() const;

    private:
      VULKAN_HPP_NAMESPACE::ValidationCacheEXT                                  m_validationCacheEXT;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
    class VideoSessionKHR
    {
    public:
      using CType = VkVideoSessionKHR;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eVideoSessionKHR;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eUnknown;

    public:
      VideoSessionKHR(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VULKAN_HPP_NAMESPACE::VideoSessionCreateInfoKHR const &                         createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateVideoSessionKHR(
            static_cast<VkDevice>( *device ),
            reinterpret_cast<const VkVideoSessionCreateInfoKHR *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkVideoSessionKHR *>( &m_videoSessionKHR ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateVideoSessionKHR" );
        }
      }

      VideoSessionKHR(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VkVideoSessionKHR                                                               videoSessionKHR,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_videoSessionKHR( videoSessionKHR )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~VideoSessionKHR()
      {
        if ( m_videoSessionKHR )
        {
          getDispatcher()->vkDestroyVideoSessionKHR(
            m_device, static_cast<VkVideoSessionKHR>( m_videoSessionKHR ), m_allocator );
        }
      }

#    if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      VideoSessionKHR() = default;
#    else
      VideoSessionKHR()           = delete;
#    endif
      VideoSessionKHR( VideoSessionKHR const & ) = delete;
      VideoSessionKHR( VideoSessionKHR && rhs ) VULKAN_HPP_NOEXCEPT
        : m_videoSessionKHR( VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_videoSessionKHR, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      VideoSessionKHR & operator=( VideoSessionKHR const & ) = delete;
      VideoSessionKHR & operator                             =( VideoSessionKHR && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_videoSessionKHR )
          {
            getDispatcher()->vkDestroyVideoSessionKHR(
              m_device, static_cast<VkVideoSessionKHR>( m_videoSessionKHR ), m_allocator );
          }
          m_videoSessionKHR = VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_videoSessionKHR, {} );
          m_device          = rhs.m_device;
          m_allocator       = rhs.m_allocator;
          m_dispatcher      = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::VideoSessionKHR const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_videoSessionKHR;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#    if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_videoSessionKHR.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_videoSessionKHR.operator!();
      }
#    endif

      //=== VK_KHR_video_queue ===

      VULKAN_HPP_NODISCARD std::vector<VULKAN_HPP_NAMESPACE::VideoGetMemoryPropertiesKHR> getMemoryRequirements() const;

      void
        bindMemory( ArrayProxy<const VULKAN_HPP_NAMESPACE::VideoBindMemoryKHR> const & videoSessionBindMemories ) const;

    private:
      VULKAN_HPP_NAMESPACE::VideoSessionKHR                                     m_videoSessionKHR;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
    class VideoSessionParametersKHR
    {
    public:
      using CType = VkVideoSessionParametersKHR;

      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::ObjectType objectType =
        VULKAN_HPP_NAMESPACE::ObjectType::eVideoSessionParametersKHR;
      static VULKAN_HPP_CONST_OR_CONSTEXPR VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT debugReportObjectType =
        VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT::eUnknown;

    public:
      VideoSessionParametersKHR(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VULKAN_HPP_NAMESPACE::VideoSessionParametersCreateInfoKHR const &               createInfo,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {
        VULKAN_HPP_NAMESPACE::Result result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCreateVideoSessionParametersKHR(
            static_cast<VkDevice>( *device ),
            reinterpret_cast<const VkVideoSessionParametersCreateInfoKHR *>( &createInfo ),
            m_allocator,
            reinterpret_cast<VkVideoSessionParametersKHR *>( &m_videoSessionParametersKHR ) ) );
        if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
        {
          throwResultException( result, "vkCreateVideoSessionParametersKHR" );
        }
      }

      VideoSessionParametersKHR(
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::Device const &                 device,
        VkVideoSessionParametersKHR                                                     videoSessionParametersKHR,
        VULKAN_HPP_NAMESPACE::Optional<const VULKAN_HPP_NAMESPACE::AllocationCallbacks> allocator = nullptr )
        : m_videoSessionParametersKHR( videoSessionParametersKHR )
        , m_device( *device )
        , m_allocator( reinterpret_cast<const VkAllocationCallbacks *>(
            static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) )
        , m_dispatcher( device.getDispatcher() )
      {}

      ~VideoSessionParametersKHR()
      {
        if ( m_videoSessionParametersKHR )
        {
          getDispatcher()->vkDestroyVideoSessionParametersKHR(
            m_device, static_cast<VkVideoSessionParametersKHR>( m_videoSessionParametersKHR ), m_allocator );
        }
      }

#    if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      VideoSessionParametersKHR() = default;
#    else
      VideoSessionParametersKHR() = delete;
#    endif
      VideoSessionParametersKHR( VideoSessionParametersKHR const & ) = delete;
      VideoSessionParametersKHR( VideoSessionParametersKHR && rhs ) VULKAN_HPP_NOEXCEPT
        : m_videoSessionParametersKHR(
            VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_videoSessionParametersKHR, {} ) )
        , m_device( rhs.m_device )
        , m_allocator( rhs.m_allocator )
        , m_dispatcher( rhs.m_dispatcher )
      {}
      VideoSessionParametersKHR & operator=( VideoSessionParametersKHR const & ) = delete;
      VideoSessionParametersKHR & operator=( VideoSessionParametersKHR && rhs ) VULKAN_HPP_NOEXCEPT
      {
        if ( this != &rhs )
        {
          if ( m_videoSessionParametersKHR )
          {
            getDispatcher()->vkDestroyVideoSessionParametersKHR(
              m_device, static_cast<VkVideoSessionParametersKHR>( m_videoSessionParametersKHR ), m_allocator );
          }
          m_videoSessionParametersKHR =
            VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::exchange( rhs.m_videoSessionParametersKHR, {} );
          m_device     = rhs.m_device;
          m_allocator  = rhs.m_allocator;
          m_dispatcher = rhs.m_dispatcher;
        }
        return *this;
      }

      VULKAN_HPP_NAMESPACE::VideoSessionParametersKHR const & operator*() const VULKAN_HPP_NOEXCEPT
      {
        return m_videoSessionParametersKHR;
      }

      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * getDispatcher() const
      {
        VULKAN_HPP_ASSERT( m_dispatcher->getVkHeaderVersion() == VK_HEADER_VERSION );
        return m_dispatcher;
      }

#    if defined( VULKAN_HPP_RAII_ENABLE_DEFAULT_CONSTRUCTORS )
      explicit operator bool() const VULKAN_HPP_NOEXCEPT
      {
        return m_videoSessionParametersKHR.operator bool();
      }

      bool operator!() const VULKAN_HPP_NOEXCEPT
      {
        return m_videoSessionParametersKHR.operator!();
      }
#    endif

      //=== VK_KHR_video_queue ===

      void update( const VideoSessionParametersUpdateInfoKHR & updateInfo ) const;

    private:
      VULKAN_HPP_NAMESPACE::VideoSessionParametersKHR                           m_videoSessionParametersKHR;
      VkDevice                                                                  m_device;
      const VkAllocationCallbacks *                                             m_allocator;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_RAII_NAMESPACE::DeviceDispatcher const * m_dispatcher;
    };
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

    //===========================
    //=== COMMAND Definitions ===
    //===========================

    //=== VK_VERSION_1_0 ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures
                                           PhysicalDevice::getFeatures() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures features;
      getDispatcher()->vkGetPhysicalDeviceFeatures( static_cast<VkPhysicalDevice>( m_physicalDevice ),
                                                    reinterpret_cast<VkPhysicalDeviceFeatures *>( &features ) );
      return features;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::FormatProperties
                                           PhysicalDevice::getFormatProperties( VULKAN_HPP_NAMESPACE::Format format ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_NAMESPACE::FormatProperties formatProperties;
      getDispatcher()->vkGetPhysicalDeviceFormatProperties(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        static_cast<VkFormat>( format ),
        reinterpret_cast<VkFormatProperties *>( &formatProperties ) );
      return formatProperties;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::ImageFormatProperties
                                           PhysicalDevice::getImageFormatProperties( VULKAN_HPP_NAMESPACE::Format           format,
                                                VULKAN_HPP_NAMESPACE::ImageType        type,
                                                VULKAN_HPP_NAMESPACE::ImageTiling      tiling,
                                                VULKAN_HPP_NAMESPACE::ImageUsageFlags  usage,
                                                VULKAN_HPP_NAMESPACE::ImageCreateFlags flags ) const
    {
      VULKAN_HPP_NAMESPACE::ImageFormatProperties imageFormatProperties;
      VULKAN_HPP_NAMESPACE::Result                result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceImageFormatProperties(
          static_cast<VkPhysicalDevice>( m_physicalDevice ),
          static_cast<VkFormat>( format ),
          static_cast<VkImageType>( type ),
          static_cast<VkImageTiling>( tiling ),
          static_cast<VkImageUsageFlags>( usage ),
          static_cast<VkImageCreateFlags>( flags ),
          reinterpret_cast<VkImageFormatProperties *>( &imageFormatProperties ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getImageFormatProperties" );
      }
      return imageFormatProperties;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties
                                           PhysicalDevice::getProperties() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties properties;
      getDispatcher()->vkGetPhysicalDeviceProperties( static_cast<VkPhysicalDevice>( m_physicalDevice ),
                                                      reinterpret_cast<VkPhysicalDeviceProperties *>( &properties ) );
      return properties;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::QueueFamilyProperties>
                                           PhysicalDevice::getQueueFamilyProperties() const VULKAN_HPP_NOEXCEPT
    {
      uint32_t queueFamilyPropertyCount;
      getDispatcher()->vkGetPhysicalDeviceQueueFamilyProperties(
        static_cast<VkPhysicalDevice>( m_physicalDevice ), &queueFamilyPropertyCount, nullptr );
      std::vector<VULKAN_HPP_NAMESPACE::QueueFamilyProperties> queueFamilyProperties( queueFamilyPropertyCount );
      getDispatcher()->vkGetPhysicalDeviceQueueFamilyProperties(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        &queueFamilyPropertyCount,
        reinterpret_cast<VkQueueFamilyProperties *>( queueFamilyProperties.data() ) );
      VULKAN_HPP_ASSERT( queueFamilyPropertyCount <= queueFamilyProperties.size() );
      return queueFamilyProperties;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties
                                           PhysicalDevice::getMemoryProperties() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties memoryProperties;
      getDispatcher()->vkGetPhysicalDeviceMemoryProperties(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        reinterpret_cast<VkPhysicalDeviceMemoryProperties *>( &memoryProperties ) );
      return memoryProperties;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE PFN_vkVoidFunction
                                           Instance::getProcAddr( const std::string & name ) const VULKAN_HPP_NOEXCEPT
    {
      return getDispatcher()->vkGetInstanceProcAddr( static_cast<VkInstance>( m_instance ), name.c_str() );
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE PFN_vkVoidFunction
                                           Device::getProcAddr( const std::string & name ) const VULKAN_HPP_NOEXCEPT
    {
      return getDispatcher()->vkGetDeviceProcAddr( static_cast<VkDevice>( m_device ), name.c_str() );
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::ExtensionProperties>
                                           Context::enumerateInstanceExtensionProperties( Optional<const std::string> layerName ) const
    {
      std::vector<VULKAN_HPP_NAMESPACE::ExtensionProperties> properties;
      uint32_t                                               propertyCount;
      VULKAN_HPP_NAMESPACE::Result                           result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkEnumerateInstanceExtensionProperties(
          layerName ? layerName->c_str() : nullptr, &propertyCount, nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && propertyCount )
        {
          properties.resize( propertyCount );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkEnumerateInstanceExtensionProperties(
            layerName ? layerName->c_str() : nullptr,
            &propertyCount,
            reinterpret_cast<VkExtensionProperties *>( properties.data() ) ) );
          VULKAN_HPP_ASSERT( propertyCount <= properties.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( propertyCount < properties.size() ) )
      {
        properties.resize( propertyCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Context::enumerateInstanceExtensionProperties" );
      }
      return properties;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::ExtensionProperties>
                                           PhysicalDevice::enumerateDeviceExtensionProperties( Optional<const std::string> layerName ) const
    {
      std::vector<VULKAN_HPP_NAMESPACE::ExtensionProperties> properties;
      uint32_t                                               propertyCount;
      VULKAN_HPP_NAMESPACE::Result                           result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkEnumerateDeviceExtensionProperties( static_cast<VkPhysicalDevice>( m_physicalDevice ),
                                                                 layerName ? layerName->c_str() : nullptr,
                                                                 &propertyCount,
                                                                 nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && propertyCount )
        {
          properties.resize( propertyCount );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkEnumerateDeviceExtensionProperties(
            static_cast<VkPhysicalDevice>( m_physicalDevice ),
            layerName ? layerName->c_str() : nullptr,
            &propertyCount,
            reinterpret_cast<VkExtensionProperties *>( properties.data() ) ) );
          VULKAN_HPP_ASSERT( propertyCount <= properties.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( propertyCount < properties.size() ) )
      {
        properties.resize( propertyCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result,
                              VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::enumerateDeviceExtensionProperties" );
      }
      return properties;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::LayerProperties>
                                           Context::enumerateInstanceLayerProperties() const
    {
      std::vector<VULKAN_HPP_NAMESPACE::LayerProperties> properties;
      uint32_t                                           propertyCount;
      VULKAN_HPP_NAMESPACE::Result                       result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkEnumerateInstanceLayerProperties( &propertyCount, nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && propertyCount )
        {
          properties.resize( propertyCount );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkEnumerateInstanceLayerProperties(
            &propertyCount, reinterpret_cast<VkLayerProperties *>( properties.data() ) ) );
          VULKAN_HPP_ASSERT( propertyCount <= properties.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( propertyCount < properties.size() ) )
      {
        properties.resize( propertyCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Context::enumerateInstanceLayerProperties" );
      }
      return properties;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::LayerProperties>
                                           PhysicalDevice::enumerateDeviceLayerProperties() const
    {
      std::vector<VULKAN_HPP_NAMESPACE::LayerProperties> properties;
      uint32_t                                           propertyCount;
      VULKAN_HPP_NAMESPACE::Result                       result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkEnumerateDeviceLayerProperties(
          static_cast<VkPhysicalDevice>( m_physicalDevice ), &propertyCount, nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && propertyCount )
        {
          properties.resize( propertyCount );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkEnumerateDeviceLayerProperties(
            static_cast<VkPhysicalDevice>( m_physicalDevice ),
            &propertyCount,
            reinterpret_cast<VkLayerProperties *>( properties.data() ) ) );
          VULKAN_HPP_ASSERT( propertyCount <= properties.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( propertyCount < properties.size() ) )
      {
        properties.resize( propertyCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::enumerateDeviceLayerProperties" );
      }
      return properties;
    }

    VULKAN_HPP_INLINE void Queue::submit( ArrayProxy<const VULKAN_HPP_NAMESPACE::SubmitInfo> const & submits,
                                          VULKAN_HPP_NAMESPACE::Fence                                fence ) const
    {
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkQueueSubmit( static_cast<VkQueue>( m_queue ),
                                        submits.size(),
                                        reinterpret_cast<const VkSubmitInfo *>( submits.data() ),
                                        static_cast<VkFence>( fence ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Queue::submit" );
      }
    }

    VULKAN_HPP_INLINE void Queue::waitIdle() const
    {
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkQueueWaitIdle( static_cast<VkQueue>( m_queue ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Queue::waitIdle" );
      }
    }

    VULKAN_HPP_INLINE void Device::waitIdle() const
    {
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkDeviceWaitIdle( static_cast<VkDevice>( m_device ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::waitIdle" );
      }
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE void *
                         DeviceMemory::mapMemory( VULKAN_HPP_NAMESPACE::DeviceSize     offset,
                               VULKAN_HPP_NAMESPACE::DeviceSize     size,
                               VULKAN_HPP_NAMESPACE::MemoryMapFlags flags ) const
    {
      void *                       pData;
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkMapMemory( static_cast<VkDevice>( m_device ),
                                      static_cast<VkDeviceMemory>( m_deviceMemory ),
                                      static_cast<VkDeviceSize>( offset ),
                                      static_cast<VkDeviceSize>( size ),
                                      static_cast<VkMemoryMapFlags>( flags ),
                                      &pData ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::DeviceMemory::mapMemory" );
      }
      return pData;
    }

    VULKAN_HPP_INLINE void DeviceMemory::unmapMemory() const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkUnmapMemory( static_cast<VkDevice>( m_device ),
                                      static_cast<VkDeviceMemory>( m_deviceMemory ) );
    }

    VULKAN_HPP_INLINE void Device::flushMappedMemoryRanges(
      ArrayProxy<const VULKAN_HPP_NAMESPACE::MappedMemoryRange> const & memoryRanges ) const
    {
      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkFlushMappedMemoryRanges(
          static_cast<VkDevice>( m_device ),
          memoryRanges.size(),
          reinterpret_cast<const VkMappedMemoryRange *>( memoryRanges.data() ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::flushMappedMemoryRanges" );
      }
    }

    VULKAN_HPP_INLINE void Device::invalidateMappedMemoryRanges(
      ArrayProxy<const VULKAN_HPP_NAMESPACE::MappedMemoryRange> const & memoryRanges ) const
    {
      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkInvalidateMappedMemoryRanges(
          static_cast<VkDevice>( m_device ),
          memoryRanges.size(),
          reinterpret_cast<const VkMappedMemoryRange *>( memoryRanges.data() ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::invalidateMappedMemoryRanges" );
      }
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::DeviceSize
                                           DeviceMemory::getCommitment() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_NAMESPACE::DeviceSize committedMemoryInBytes;
      getDispatcher()->vkGetDeviceMemoryCommitment( static_cast<VkDevice>( m_device ),
                                                    static_cast<VkDeviceMemory>( m_deviceMemory ),
                                                    reinterpret_cast<VkDeviceSize *>( &committedMemoryInBytes ) );
      return committedMemoryInBytes;
    }

    VULKAN_HPP_INLINE void Buffer::bindMemory( VULKAN_HPP_NAMESPACE::DeviceMemory memory,
                                               VULKAN_HPP_NAMESPACE::DeviceSize   memoryOffset ) const
    {
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkBindBufferMemory( static_cast<VkDevice>( m_device ),
                                             static_cast<VkBuffer>( m_buffer ),
                                             static_cast<VkDeviceMemory>( memory ),
                                             static_cast<VkDeviceSize>( memoryOffset ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Buffer::bindMemory" );
      }
    }

    VULKAN_HPP_INLINE void Image::bindMemory( VULKAN_HPP_NAMESPACE::DeviceMemory memory,
                                              VULKAN_HPP_NAMESPACE::DeviceSize   memoryOffset ) const
    {
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkBindImageMemory( static_cast<VkDevice>( m_device ),
                                            static_cast<VkImage>( m_image ),
                                            static_cast<VkDeviceMemory>( memory ),
                                            static_cast<VkDeviceSize>( memoryOffset ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Image::bindMemory" );
      }
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::MemoryRequirements
                                           Buffer::getMemoryRequirements() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_NAMESPACE::MemoryRequirements memoryRequirements;
      getDispatcher()->vkGetBufferMemoryRequirements( static_cast<VkDevice>( m_device ),
                                                      static_cast<VkBuffer>( m_buffer ),
                                                      reinterpret_cast<VkMemoryRequirements *>( &memoryRequirements ) );
      return memoryRequirements;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::MemoryRequirements
                                           Image::getMemoryRequirements() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_NAMESPACE::MemoryRequirements memoryRequirements;
      getDispatcher()->vkGetImageMemoryRequirements( static_cast<VkDevice>( m_device ),
                                                     static_cast<VkImage>( m_image ),
                                                     reinterpret_cast<VkMemoryRequirements *>( &memoryRequirements ) );
      return memoryRequirements;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::SparseImageMemoryRequirements>
                                           Image::getSparseMemoryRequirements() const VULKAN_HPP_NOEXCEPT
    {
      uint32_t sparseMemoryRequirementCount;
      getDispatcher()->vkGetImageSparseMemoryRequirements(
        static_cast<VkDevice>( m_device ), static_cast<VkImage>( m_image ), &sparseMemoryRequirementCount, nullptr );
      std::vector<VULKAN_HPP_NAMESPACE::SparseImageMemoryRequirements> sparseMemoryRequirements(
        sparseMemoryRequirementCount );
      getDispatcher()->vkGetImageSparseMemoryRequirements(
        static_cast<VkDevice>( m_device ),
        static_cast<VkImage>( m_image ),
        &sparseMemoryRequirementCount,
        reinterpret_cast<VkSparseImageMemoryRequirements *>( sparseMemoryRequirements.data() ) );
      VULKAN_HPP_ASSERT( sparseMemoryRequirementCount <= sparseMemoryRequirements.size() );
      return sparseMemoryRequirements;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::SparseImageFormatProperties>
                                           PhysicalDevice::getSparseImageFormatProperties( VULKAN_HPP_NAMESPACE::Format              format,
                                                      VULKAN_HPP_NAMESPACE::ImageType           type,
                                                      VULKAN_HPP_NAMESPACE::SampleCountFlagBits samples,
                                                      VULKAN_HPP_NAMESPACE::ImageUsageFlags     usage,
                                                      VULKAN_HPP_NAMESPACE::ImageTiling         tiling ) const
      VULKAN_HPP_NOEXCEPT
    {
      uint32_t propertyCount;
      getDispatcher()->vkGetPhysicalDeviceSparseImageFormatProperties(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        static_cast<VkFormat>( format ),
        static_cast<VkImageType>( type ),
        static_cast<VkSampleCountFlagBits>( samples ),
        static_cast<VkImageUsageFlags>( usage ),
        static_cast<VkImageTiling>( tiling ),
        &propertyCount,
        nullptr );
      std::vector<VULKAN_HPP_NAMESPACE::SparseImageFormatProperties> properties( propertyCount );
      getDispatcher()->vkGetPhysicalDeviceSparseImageFormatProperties(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        static_cast<VkFormat>( format ),
        static_cast<VkImageType>( type ),
        static_cast<VkSampleCountFlagBits>( samples ),
        static_cast<VkImageUsageFlags>( usage ),
        static_cast<VkImageTiling>( tiling ),
        &propertyCount,
        reinterpret_cast<VkSparseImageFormatProperties *>( properties.data() ) );
      VULKAN_HPP_ASSERT( propertyCount <= properties.size() );
      return properties;
    }

    VULKAN_HPP_INLINE void Queue::bindSparse( ArrayProxy<const VULKAN_HPP_NAMESPACE::BindSparseInfo> const & bindInfo,
                                              VULKAN_HPP_NAMESPACE::Fence fence ) const
    {
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkQueueBindSparse( static_cast<VkQueue>( m_queue ),
                                            bindInfo.size(),
                                            reinterpret_cast<const VkBindSparseInfo *>( bindInfo.data() ),
                                            static_cast<VkFence>( fence ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Queue::bindSparse" );
      }
    }

    VULKAN_HPP_INLINE void Device::resetFences( ArrayProxy<const VULKAN_HPP_NAMESPACE::Fence> const & fences ) const
    {
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkResetFences(
        static_cast<VkDevice>( m_device ), fences.size(), reinterpret_cast<const VkFence *>( fences.data() ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::resetFences" );
      }
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::Result Fence::getStatus() const
    {
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkGetFenceStatus( static_cast<VkDevice>( m_device ), static_cast<VkFence>( m_fence ) ) );
      if ( ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eNotReady ) )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Fence::getStatus" );
      }
      return result;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::Result
                                           Device::waitForFences( ArrayProxy<const VULKAN_HPP_NAMESPACE::Fence> const & fences,
                             VULKAN_HPP_NAMESPACE::Bool32                          waitAll,
                             uint64_t                                              timeout ) const
    {
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkWaitForFences( static_cast<VkDevice>( m_device ),
                                          fences.size(),
                                          reinterpret_cast<const VkFence *>( fences.data() ),
                                          static_cast<VkBool32>( waitAll ),
                                          timeout ) );
      if ( ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eTimeout ) )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::waitForFences" );
      }
      return result;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::Result Event::getStatus() const
    {
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkGetEventStatus( static_cast<VkDevice>( m_device ), static_cast<VkEvent>( m_event ) ) );
      if ( ( result != VULKAN_HPP_NAMESPACE::Result::eEventSet ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eEventReset ) )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Event::getStatus" );
      }
      return result;
    }

    VULKAN_HPP_INLINE void Event::set() const
    {
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkSetEvent( static_cast<VkDevice>( m_device ), static_cast<VkEvent>( m_event ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Event::set" );
      }
    }

    VULKAN_HPP_INLINE void Event::reset() const
    {
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkResetEvent( static_cast<VkDevice>( m_device ), static_cast<VkEvent>( m_event ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Event::reset" );
      }
    }

    template <typename T>
    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::pair<VULKAN_HPP_NAMESPACE::Result, std::vector<T>>
                                           QueryPool::getResults( uint32_t                               firstQuery,
                             uint32_t                               queryCount,
                             size_t                                 dataSize,
                             VULKAN_HPP_NAMESPACE::DeviceSize       stride,
                             VULKAN_HPP_NAMESPACE::QueryResultFlags flags ) const
    {
      VULKAN_HPP_ASSERT( dataSize % sizeof( T ) == 0 );
      std::vector<T> data( dataSize / sizeof( T ) );
      Result         result =
        static_cast<Result>( getDispatcher()->vkGetQueryPoolResults( static_cast<VkDevice>( m_device ),
                                                                     static_cast<VkQueryPool>( m_queryPool ),
                                                                     firstQuery,
                                                                     queryCount,
                                                                     data.size() * sizeof( T ),
                                                                     reinterpret_cast<void *>( data.data() ),
                                                                     static_cast<VkDeviceSize>( stride ),
                                                                     static_cast<VkQueryResultFlags>( flags ) ) );
      if ( ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eNotReady ) )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::QueryPool::getResults" );
      }
      return std::make_pair( result, data );
    }

    template <typename T>
    VULKAN_HPP_NODISCARD std::pair<VULKAN_HPP_NAMESPACE::Result, T>
                         QueryPool::getResult( uint32_t                               firstQuery,
                            uint32_t                               queryCount,
                            VULKAN_HPP_NAMESPACE::DeviceSize       stride,
                            VULKAN_HPP_NAMESPACE::QueryResultFlags flags ) const
    {
      T      data;
      Result result =
        static_cast<Result>( getDispatcher()->vkGetQueryPoolResults( static_cast<VkDevice>( m_device ),
                                                                     static_cast<VkQueryPool>( m_queryPool ),
                                                                     firstQuery,
                                                                     queryCount,
                                                                     sizeof( T ),
                                                                     reinterpret_cast<void *>( &data ),
                                                                     static_cast<VkDeviceSize>( stride ),
                                                                     static_cast<VkQueryResultFlags>( flags ) ) );
      if ( ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eNotReady ) )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::QueryPool::getResult" );
      }
      return std::make_pair( result, data );
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::SubresourceLayout
                                           Image::getSubresourceLayout( const ImageSubresource & subresource ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_NAMESPACE::SubresourceLayout layout;
      getDispatcher()->vkGetImageSubresourceLayout( static_cast<VkDevice>( m_device ),
                                                    static_cast<VkImage>( m_image ),
                                                    reinterpret_cast<const VkImageSubresource *>( &subresource ),
                                                    reinterpret_cast<VkSubresourceLayout *>( &layout ) );
      return layout;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<uint8_t> PipelineCache::getData() const
    {
      std::vector<uint8_t>         data;
      size_t                       dataSize;
      VULKAN_HPP_NAMESPACE::Result result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPipelineCacheData(
          static_cast<VkDevice>( m_device ), static_cast<VkPipelineCache>( m_pipelineCache ), &dataSize, nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && dataSize )
        {
          data.resize( dataSize );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
            getDispatcher()->vkGetPipelineCacheData( static_cast<VkDevice>( m_device ),
                                                     static_cast<VkPipelineCache>( m_pipelineCache ),
                                                     &dataSize,
                                                     reinterpret_cast<void *>( data.data() ) ) );
          VULKAN_HPP_ASSERT( dataSize <= data.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( dataSize < data.size() ) )
      {
        data.resize( dataSize );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PipelineCache::getData" );
      }
      return data;
    }

    VULKAN_HPP_INLINE void
      PipelineCache::merge( ArrayProxy<const VULKAN_HPP_NAMESPACE::PipelineCache> const & srcCaches ) const
    {
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkMergePipelineCaches( static_cast<VkDevice>( m_device ),
                                                static_cast<VkPipelineCache>( m_pipelineCache ),
                                                srcCaches.size(),
                                                reinterpret_cast<const VkPipelineCache *>( srcCaches.data() ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PipelineCache::merge" );
      }
    }

    VULKAN_HPP_INLINE void
      DescriptorPool::reset( VULKAN_HPP_NAMESPACE::DescriptorPoolResetFlags flags ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkResetDescriptorPool( static_cast<VkDevice>( m_device ),
                                              static_cast<VkDescriptorPool>( m_descriptorPool ),
                                              static_cast<VkDescriptorPoolResetFlags>( flags ) );
    }

    VULKAN_HPP_INLINE void Device::updateDescriptorSets(
      ArrayProxy<const VULKAN_HPP_NAMESPACE::WriteDescriptorSet> const & descriptorWrites,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::CopyDescriptorSet> const &  descriptorCopies ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkUpdateDescriptorSets(
        static_cast<VkDevice>( m_device ),
        descriptorWrites.size(),
        reinterpret_cast<const VkWriteDescriptorSet *>( descriptorWrites.data() ),
        descriptorCopies.size(),
        reinterpret_cast<const VkCopyDescriptorSet *>( descriptorCopies.data() ) );
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::Extent2D
                                           RenderPass::getRenderAreaGranularity() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_NAMESPACE::Extent2D granularity;
      getDispatcher()->vkGetRenderAreaGranularity( static_cast<VkDevice>( m_device ),
                                                   static_cast<VkRenderPass>( m_renderPass ),
                                                   reinterpret_cast<VkExtent2D *>( &granularity ) );
      return granularity;
    }

    VULKAN_HPP_INLINE void CommandPool::reset( VULKAN_HPP_NAMESPACE::CommandPoolResetFlags flags ) const
    {
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkResetCommandPool( static_cast<VkDevice>( m_device ),
                                             static_cast<VkCommandPool>( m_commandPool ),
                                             static_cast<VkCommandPoolResetFlags>( flags ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::CommandPool::reset" );
      }
    }

    VULKAN_HPP_INLINE void CommandBuffer::begin( const CommandBufferBeginInfo & beginInfo ) const
    {
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkBeginCommandBuffer( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                               reinterpret_cast<const VkCommandBufferBeginInfo *>( &beginInfo ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::CommandBuffer::begin" );
      }
    }

    VULKAN_HPP_INLINE void CommandBuffer::end() const
    {
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkEndCommandBuffer( static_cast<VkCommandBuffer>( m_commandBuffer ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::CommandBuffer::end" );
      }
    }

    VULKAN_HPP_INLINE void CommandBuffer::reset( VULKAN_HPP_NAMESPACE::CommandBufferResetFlags flags ) const
    {
      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkResetCommandBuffer(
          static_cast<VkCommandBuffer>( m_commandBuffer ), static_cast<VkCommandBufferResetFlags>( flags ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::CommandBuffer::reset" );
      }
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::bindPipeline( VULKAN_HPP_NAMESPACE::PipelineBindPoint pipelineBindPoint,
                                   VULKAN_HPP_NAMESPACE::Pipeline          pipeline ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdBindPipeline( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                          static_cast<VkPipelineBindPoint>( pipelineBindPoint ),
                                          static_cast<VkPipeline>( pipeline ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::setViewport(
      uint32_t                                                 firstViewport,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::Viewport> const & viewports ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdSetViewport( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                         firstViewport,
                                         viewports.size(),
                                         reinterpret_cast<const VkViewport *>( viewports.data() ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::setScissor(
      uint32_t firstScissor, ArrayProxy<const VULKAN_HPP_NAMESPACE::Rect2D> const & scissors ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdSetScissor( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                        firstScissor,
                                        scissors.size(),
                                        reinterpret_cast<const VkRect2D *>( scissors.data() ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::setLineWidth( float lineWidth ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdSetLineWidth( static_cast<VkCommandBuffer>( m_commandBuffer ), lineWidth );
    }

    VULKAN_HPP_INLINE void CommandBuffer::setDepthBias( float depthBiasConstantFactor,
                                                        float depthBiasClamp,
                                                        float depthBiasSlopeFactor ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdSetDepthBias( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                          depthBiasConstantFactor,
                                          depthBiasClamp,
                                          depthBiasSlopeFactor );
    }

    VULKAN_HPP_INLINE void CommandBuffer::setBlendConstants( const float blendConstants[4] ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdSetBlendConstants( static_cast<VkCommandBuffer>( m_commandBuffer ), blendConstants );
    }

    VULKAN_HPP_INLINE void CommandBuffer::setDepthBounds( float minDepthBounds,
                                                          float maxDepthBounds ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdSetDepthBounds(
        static_cast<VkCommandBuffer>( m_commandBuffer ), minDepthBounds, maxDepthBounds );
    }

    VULKAN_HPP_INLINE void CommandBuffer::setStencilCompareMask( VULKAN_HPP_NAMESPACE::StencilFaceFlags faceMask,
                                                                 uint32_t compareMask ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdSetStencilCompareMask(
        static_cast<VkCommandBuffer>( m_commandBuffer ), static_cast<VkStencilFaceFlags>( faceMask ), compareMask );
    }

    VULKAN_HPP_INLINE void CommandBuffer::setStencilWriteMask( VULKAN_HPP_NAMESPACE::StencilFaceFlags faceMask,
                                                               uint32_t writeMask ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdSetStencilWriteMask(
        static_cast<VkCommandBuffer>( m_commandBuffer ), static_cast<VkStencilFaceFlags>( faceMask ), writeMask );
    }

    VULKAN_HPP_INLINE void CommandBuffer::setStencilReference( VULKAN_HPP_NAMESPACE::StencilFaceFlags faceMask,
                                                               uint32_t reference ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdSetStencilReference(
        static_cast<VkCommandBuffer>( m_commandBuffer ), static_cast<VkStencilFaceFlags>( faceMask ), reference );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::bindDescriptorSets( VULKAN_HPP_NAMESPACE::PipelineBindPoint pipelineBindPoint,
                                         VULKAN_HPP_NAMESPACE::PipelineLayout    layout,
                                         uint32_t                                firstSet,
                                         ArrayProxy<const VULKAN_HPP_NAMESPACE::DescriptorSet> const & descriptorSets,
                                         ArrayProxy<const uint32_t> const & dynamicOffsets ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdBindDescriptorSets( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                static_cast<VkPipelineBindPoint>( pipelineBindPoint ),
                                                static_cast<VkPipelineLayout>( layout ),
                                                firstSet,
                                                descriptorSets.size(),
                                                reinterpret_cast<const VkDescriptorSet *>( descriptorSets.data() ),
                                                dynamicOffsets.size(),
                                                dynamicOffsets.data() );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::bindIndexBuffer( VULKAN_HPP_NAMESPACE::Buffer     buffer,
                                      VULKAN_HPP_NAMESPACE::DeviceSize offset,
                                      VULKAN_HPP_NAMESPACE::IndexType  indexType ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdBindIndexBuffer( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                             static_cast<VkBuffer>( buffer ),
                                             static_cast<VkDeviceSize>( offset ),
                                             static_cast<VkIndexType>( indexType ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::bindVertexBuffers(
      uint32_t                                                   firstBinding,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::Buffer> const &     buffers,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::DeviceSize> const & offsets ) const VULKAN_HPP_NOEXCEPT_WHEN_NO_EXCEPTIONS
    {
#  ifdef VULKAN_HPP_NO_EXCEPTIONS
      VULKAN_HPP_ASSERT( buffers.size() == offsets.size() );
#  else
      if ( buffers.size() != offsets.size() )
      {
        throw LogicError( VULKAN_HPP_NAMESPACE_STRING
                          "::CommandBuffer::bindVertexBuffers: buffers.size() != offsets.size()" );
      }
#  endif /*VULKAN_HPP_NO_EXCEPTIONS*/

      getDispatcher()->vkCmdBindVertexBuffers( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                               firstBinding,
                                               buffers.size(),
                                               reinterpret_cast<const VkBuffer *>( buffers.data() ),
                                               reinterpret_cast<const VkDeviceSize *>( offsets.data() ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::draw( uint32_t vertexCount,
                                                uint32_t instanceCount,
                                                uint32_t firstVertex,
                                                uint32_t firstInstance ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdDraw(
        static_cast<VkCommandBuffer>( m_commandBuffer ), vertexCount, instanceCount, firstVertex, firstInstance );
    }

    VULKAN_HPP_INLINE void CommandBuffer::drawIndexed( uint32_t indexCount,
                                                       uint32_t instanceCount,
                                                       uint32_t firstIndex,
                                                       int32_t  vertexOffset,
                                                       uint32_t firstInstance ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdDrawIndexed( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                         indexCount,
                                         instanceCount,
                                         firstIndex,
                                         vertexOffset,
                                         firstInstance );
    }

    VULKAN_HPP_INLINE void CommandBuffer::drawIndirect( VULKAN_HPP_NAMESPACE::Buffer     buffer,
                                                        VULKAN_HPP_NAMESPACE::DeviceSize offset,
                                                        uint32_t                         drawCount,
                                                        uint32_t stride ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdDrawIndirect( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                          static_cast<VkBuffer>( buffer ),
                                          static_cast<VkDeviceSize>( offset ),
                                          drawCount,
                                          stride );
    }

    VULKAN_HPP_INLINE void CommandBuffer::drawIndexedIndirect( VULKAN_HPP_NAMESPACE::Buffer     buffer,
                                                               VULKAN_HPP_NAMESPACE::DeviceSize offset,
                                                               uint32_t                         drawCount,
                                                               uint32_t stride ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdDrawIndexedIndirect( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                 static_cast<VkBuffer>( buffer ),
                                                 static_cast<VkDeviceSize>( offset ),
                                                 drawCount,
                                                 stride );
    }

    VULKAN_HPP_INLINE void CommandBuffer::dispatch( uint32_t groupCountX,
                                                    uint32_t groupCountY,
                                                    uint32_t groupCountZ ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdDispatch(
        static_cast<VkCommandBuffer>( m_commandBuffer ), groupCountX, groupCountY, groupCountZ );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::dispatchIndirect( VULKAN_HPP_NAMESPACE::Buffer     buffer,
                                       VULKAN_HPP_NAMESPACE::DeviceSize offset ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdDispatchIndirect( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                              static_cast<VkBuffer>( buffer ),
                                              static_cast<VkDeviceSize>( offset ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::copyBuffer(
      VULKAN_HPP_NAMESPACE::Buffer                               srcBuffer,
      VULKAN_HPP_NAMESPACE::Buffer                               dstBuffer,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::BufferCopy> const & regions ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdCopyBuffer( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                        static_cast<VkBuffer>( srcBuffer ),
                                        static_cast<VkBuffer>( dstBuffer ),
                                        regions.size(),
                                        reinterpret_cast<const VkBufferCopy *>( regions.data() ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::copyImage(
      VULKAN_HPP_NAMESPACE::Image                               srcImage,
      VULKAN_HPP_NAMESPACE::ImageLayout                         srcImageLayout,
      VULKAN_HPP_NAMESPACE::Image                               dstImage,
      VULKAN_HPP_NAMESPACE::ImageLayout                         dstImageLayout,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::ImageCopy> const & regions ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdCopyImage( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                       static_cast<VkImage>( srcImage ),
                                       static_cast<VkImageLayout>( srcImageLayout ),
                                       static_cast<VkImage>( dstImage ),
                                       static_cast<VkImageLayout>( dstImageLayout ),
                                       regions.size(),
                                       reinterpret_cast<const VkImageCopy *>( regions.data() ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::blitImage( VULKAN_HPP_NAMESPACE::Image       srcImage,
                                                     VULKAN_HPP_NAMESPACE::ImageLayout srcImageLayout,
                                                     VULKAN_HPP_NAMESPACE::Image       dstImage,
                                                     VULKAN_HPP_NAMESPACE::ImageLayout dstImageLayout,
                                                     ArrayProxy<const VULKAN_HPP_NAMESPACE::ImageBlit> const & regions,
                                                     VULKAN_HPP_NAMESPACE::Filter filter ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdBlitImage( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                       static_cast<VkImage>( srcImage ),
                                       static_cast<VkImageLayout>( srcImageLayout ),
                                       static_cast<VkImage>( dstImage ),
                                       static_cast<VkImageLayout>( dstImageLayout ),
                                       regions.size(),
                                       reinterpret_cast<const VkImageBlit *>( regions.data() ),
                                       static_cast<VkFilter>( filter ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::copyBufferToImage(
      VULKAN_HPP_NAMESPACE::Buffer                                    srcBuffer,
      VULKAN_HPP_NAMESPACE::Image                                     dstImage,
      VULKAN_HPP_NAMESPACE::ImageLayout                               dstImageLayout,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::BufferImageCopy> const & regions ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdCopyBufferToImage( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                               static_cast<VkBuffer>( srcBuffer ),
                                               static_cast<VkImage>( dstImage ),
                                               static_cast<VkImageLayout>( dstImageLayout ),
                                               regions.size(),
                                               reinterpret_cast<const VkBufferImageCopy *>( regions.data() ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::copyImageToBuffer(
      VULKAN_HPP_NAMESPACE::Image                                     srcImage,
      VULKAN_HPP_NAMESPACE::ImageLayout                               srcImageLayout,
      VULKAN_HPP_NAMESPACE::Buffer                                    dstBuffer,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::BufferImageCopy> const & regions ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdCopyImageToBuffer( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                               static_cast<VkImage>( srcImage ),
                                               static_cast<VkImageLayout>( srcImageLayout ),
                                               static_cast<VkBuffer>( dstBuffer ),
                                               regions.size(),
                                               reinterpret_cast<const VkBufferImageCopy *>( regions.data() ) );
    }

    template <typename T>
    VULKAN_HPP_INLINE void CommandBuffer::updateBuffer( VULKAN_HPP_NAMESPACE::Buffer     dstBuffer,
                                                        VULKAN_HPP_NAMESPACE::DeviceSize dstOffset,
                                                        ArrayProxy<const T> const & data ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdUpdateBuffer( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                          static_cast<VkBuffer>( dstBuffer ),
                                          static_cast<VkDeviceSize>( dstOffset ),
                                          data.size() * sizeof( T ),
                                          reinterpret_cast<const void *>( data.data() ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::fillBuffer( VULKAN_HPP_NAMESPACE::Buffer     dstBuffer,
                                                      VULKAN_HPP_NAMESPACE::DeviceSize dstOffset,
                                                      VULKAN_HPP_NAMESPACE::DeviceSize size,
                                                      uint32_t                         data ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdFillBuffer( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                        static_cast<VkBuffer>( dstBuffer ),
                                        static_cast<VkDeviceSize>( dstOffset ),
                                        static_cast<VkDeviceSize>( size ),
                                        data );
    }

    VULKAN_HPP_INLINE void CommandBuffer::clearColorImage(
      VULKAN_HPP_NAMESPACE::Image                                           image,
      VULKAN_HPP_NAMESPACE::ImageLayout                                     imageLayout,
      const ClearColorValue &                                               color,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::ImageSubresourceRange> const & ranges ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdClearColorImage( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                             static_cast<VkImage>( image ),
                                             static_cast<VkImageLayout>( imageLayout ),
                                             reinterpret_cast<const VkClearColorValue *>( &color ),
                                             ranges.size(),
                                             reinterpret_cast<const VkImageSubresourceRange *>( ranges.data() ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::clearDepthStencilImage(
      VULKAN_HPP_NAMESPACE::Image                                           image,
      VULKAN_HPP_NAMESPACE::ImageLayout                                     imageLayout,
      const ClearDepthStencilValue &                                        depthStencil,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::ImageSubresourceRange> const & ranges ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdClearDepthStencilImage(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        static_cast<VkImage>( image ),
        static_cast<VkImageLayout>( imageLayout ),
        reinterpret_cast<const VkClearDepthStencilValue *>( &depthStencil ),
        ranges.size(),
        reinterpret_cast<const VkImageSubresourceRange *>( ranges.data() ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::clearAttachments(
      ArrayProxy<const VULKAN_HPP_NAMESPACE::ClearAttachment> const & attachments,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::ClearRect> const &       rects ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdClearAttachments( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                              attachments.size(),
                                              reinterpret_cast<const VkClearAttachment *>( attachments.data() ),
                                              rects.size(),
                                              reinterpret_cast<const VkClearRect *>( rects.data() ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::resolveImage(
      VULKAN_HPP_NAMESPACE::Image                                  srcImage,
      VULKAN_HPP_NAMESPACE::ImageLayout                            srcImageLayout,
      VULKAN_HPP_NAMESPACE::Image                                  dstImage,
      VULKAN_HPP_NAMESPACE::ImageLayout                            dstImageLayout,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::ImageResolve> const & regions ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdResolveImage( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                          static_cast<VkImage>( srcImage ),
                                          static_cast<VkImageLayout>( srcImageLayout ),
                                          static_cast<VkImage>( dstImage ),
                                          static_cast<VkImageLayout>( dstImageLayout ),
                                          regions.size(),
                                          reinterpret_cast<const VkImageResolve *>( regions.data() ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::setEvent( VULKAN_HPP_NAMESPACE::Event              event,
                               VULKAN_HPP_NAMESPACE::PipelineStageFlags stageMask ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdSetEvent( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                      static_cast<VkEvent>( event ),
                                      static_cast<VkPipelineStageFlags>( stageMask ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::resetEvent( VULKAN_HPP_NAMESPACE::Event              event,
                                 VULKAN_HPP_NAMESPACE::PipelineStageFlags stageMask ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdResetEvent( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                        static_cast<VkEvent>( event ),
                                        static_cast<VkPipelineStageFlags>( stageMask ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::waitEvents(
      ArrayProxy<const VULKAN_HPP_NAMESPACE::Event> const &               events,
      VULKAN_HPP_NAMESPACE::PipelineStageFlags                            srcStageMask,
      VULKAN_HPP_NAMESPACE::PipelineStageFlags                            dstStageMask,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::MemoryBarrier> const &       memoryBarriers,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::BufferMemoryBarrier> const & bufferMemoryBarriers,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::ImageMemoryBarrier> const & imageMemoryBarriers ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdWaitEvents( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                        events.size(),
                                        reinterpret_cast<const VkEvent *>( events.data() ),
                                        static_cast<VkPipelineStageFlags>( srcStageMask ),
                                        static_cast<VkPipelineStageFlags>( dstStageMask ),
                                        memoryBarriers.size(),
                                        reinterpret_cast<const VkMemoryBarrier *>( memoryBarriers.data() ),
                                        bufferMemoryBarriers.size(),
                                        reinterpret_cast<const VkBufferMemoryBarrier *>( bufferMemoryBarriers.data() ),
                                        imageMemoryBarriers.size(),
                                        reinterpret_cast<const VkImageMemoryBarrier *>( imageMemoryBarriers.data() ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::pipelineBarrier(
      VULKAN_HPP_NAMESPACE::PipelineStageFlags                            srcStageMask,
      VULKAN_HPP_NAMESPACE::PipelineStageFlags                            dstStageMask,
      VULKAN_HPP_NAMESPACE::DependencyFlags                               dependencyFlags,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::MemoryBarrier> const &       memoryBarriers,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::BufferMemoryBarrier> const & bufferMemoryBarriers,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::ImageMemoryBarrier> const & imageMemoryBarriers ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdPipelineBarrier(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        static_cast<VkPipelineStageFlags>( srcStageMask ),
        static_cast<VkPipelineStageFlags>( dstStageMask ),
        static_cast<VkDependencyFlags>( dependencyFlags ),
        memoryBarriers.size(),
        reinterpret_cast<const VkMemoryBarrier *>( memoryBarriers.data() ),
        bufferMemoryBarriers.size(),
        reinterpret_cast<const VkBufferMemoryBarrier *>( bufferMemoryBarriers.data() ),
        imageMemoryBarriers.size(),
        reinterpret_cast<const VkImageMemoryBarrier *>( imageMemoryBarriers.data() ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::beginQuery( VULKAN_HPP_NAMESPACE::QueryPool         queryPool,
                                 uint32_t                                query,
                                 VULKAN_HPP_NAMESPACE::QueryControlFlags flags ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdBeginQuery( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                        static_cast<VkQueryPool>( queryPool ),
                                        query,
                                        static_cast<VkQueryControlFlags>( flags ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::endQuery( VULKAN_HPP_NAMESPACE::QueryPool queryPool,
                                                    uint32_t                        query ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdEndQuery(
        static_cast<VkCommandBuffer>( m_commandBuffer ), static_cast<VkQueryPool>( queryPool ), query );
    }

    VULKAN_HPP_INLINE void CommandBuffer::resetQueryPool( VULKAN_HPP_NAMESPACE::QueryPool queryPool,
                                                          uint32_t                        firstQuery,
                                                          uint32_t queryCount ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdResetQueryPool( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                            static_cast<VkQueryPool>( queryPool ),
                                            firstQuery,
                                            queryCount );
    }

    VULKAN_HPP_INLINE void CommandBuffer::writeTimestamp( VULKAN_HPP_NAMESPACE::PipelineStageFlagBits pipelineStage,
                                                          VULKAN_HPP_NAMESPACE::QueryPool             queryPool,
                                                          uint32_t query ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdWriteTimestamp( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                            static_cast<VkPipelineStageFlagBits>( pipelineStage ),
                                            static_cast<VkQueryPool>( queryPool ),
                                            query );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::copyQueryPoolResults( VULKAN_HPP_NAMESPACE::QueryPool        queryPool,
                                           uint32_t                               firstQuery,
                                           uint32_t                               queryCount,
                                           VULKAN_HPP_NAMESPACE::Buffer           dstBuffer,
                                           VULKAN_HPP_NAMESPACE::DeviceSize       dstOffset,
                                           VULKAN_HPP_NAMESPACE::DeviceSize       stride,
                                           VULKAN_HPP_NAMESPACE::QueryResultFlags flags ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdCopyQueryPoolResults( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                  static_cast<VkQueryPool>( queryPool ),
                                                  firstQuery,
                                                  queryCount,
                                                  static_cast<VkBuffer>( dstBuffer ),
                                                  static_cast<VkDeviceSize>( dstOffset ),
                                                  static_cast<VkDeviceSize>( stride ),
                                                  static_cast<VkQueryResultFlags>( flags ) );
    }

    template <typename T>
    VULKAN_HPP_INLINE void CommandBuffer::pushConstants( VULKAN_HPP_NAMESPACE::PipelineLayout   layout,
                                                         VULKAN_HPP_NAMESPACE::ShaderStageFlags stageFlags,
                                                         uint32_t                               offset,
                                                         ArrayProxy<const T> const & values ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdPushConstants( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                           static_cast<VkPipelineLayout>( layout ),
                                           static_cast<VkShaderStageFlags>( stageFlags ),
                                           offset,
                                           values.size() * sizeof( T ),
                                           reinterpret_cast<const void *>( values.data() ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::beginRenderPass( const RenderPassBeginInfo &           renderPassBegin,
                                      VULKAN_HPP_NAMESPACE::SubpassContents contents ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdBeginRenderPass( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                             reinterpret_cast<const VkRenderPassBeginInfo *>( &renderPassBegin ),
                                             static_cast<VkSubpassContents>( contents ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::nextSubpass( VULKAN_HPP_NAMESPACE::SubpassContents contents ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdNextSubpass( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                         static_cast<VkSubpassContents>( contents ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::endRenderPass() const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdEndRenderPass( static_cast<VkCommandBuffer>( m_commandBuffer ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::executeCommands(
      ArrayProxy<const VULKAN_HPP_NAMESPACE::CommandBuffer> const & commandBuffers ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdExecuteCommands( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                             commandBuffers.size(),
                                             reinterpret_cast<const VkCommandBuffer *>( commandBuffers.data() ) );
    }

    //=== VK_VERSION_1_1 ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE uint32_t Context::enumerateInstanceVersion() const
    {
      uint32_t                     apiVersion;
      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkEnumerateInstanceVersion( &apiVersion ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Context::enumerateInstanceVersion" );
      }
      return apiVersion;
    }

    VULKAN_HPP_INLINE void
      Device::bindBufferMemory2( ArrayProxy<const VULKAN_HPP_NAMESPACE::BindBufferMemoryInfo> const & bindInfos ) const
    {
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkBindBufferMemory2( static_cast<VkDevice>( m_device ),
                                              bindInfos.size(),
                                              reinterpret_cast<const VkBindBufferMemoryInfo *>( bindInfos.data() ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::bindBufferMemory2" );
      }
    }

    VULKAN_HPP_INLINE void
      Device::bindImageMemory2( ArrayProxy<const VULKAN_HPP_NAMESPACE::BindImageMemoryInfo> const & bindInfos ) const
    {
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkBindImageMemory2( static_cast<VkDevice>( m_device ),
                                             bindInfos.size(),
                                             reinterpret_cast<const VkBindImageMemoryInfo *>( bindInfos.data() ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::bindImageMemory2" );
      }
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::PeerMemoryFeatureFlags
                                           Device::getGroupPeerMemoryFeatures( uint32_t heapIndex,
                                          uint32_t localDeviceIndex,
                                          uint32_t remoteDeviceIndex ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_NAMESPACE::PeerMemoryFeatureFlags peerMemoryFeatures;
      getDispatcher()->vkGetDeviceGroupPeerMemoryFeatures(
        static_cast<VkDevice>( m_device ),
        heapIndex,
        localDeviceIndex,
        remoteDeviceIndex,
        reinterpret_cast<VkPeerMemoryFeatureFlags *>( &peerMemoryFeatures ) );
      return peerMemoryFeatures;
    }

    VULKAN_HPP_INLINE void CommandBuffer::setDeviceMask( uint32_t deviceMask ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdSetDeviceMask( static_cast<VkCommandBuffer>( m_commandBuffer ), deviceMask );
    }

    VULKAN_HPP_INLINE void CommandBuffer::dispatchBase( uint32_t baseGroupX,
                                                        uint32_t baseGroupY,
                                                        uint32_t baseGroupZ,
                                                        uint32_t groupCountX,
                                                        uint32_t groupCountY,
                                                        uint32_t groupCountZ ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdDispatchBase( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                          baseGroupX,
                                          baseGroupY,
                                          baseGroupZ,
                                          groupCountX,
                                          groupCountY,
                                          groupCountZ );
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::PhysicalDeviceGroupProperties>
                                           Instance::enumeratePhysicalDeviceGroups() const
    {
      std::vector<VULKAN_HPP_NAMESPACE::PhysicalDeviceGroupProperties> physicalDeviceGroupProperties;
      uint32_t                                                         physicalDeviceGroupCount;
      VULKAN_HPP_NAMESPACE::Result                                     result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkEnumeratePhysicalDeviceGroups(
          static_cast<VkInstance>( m_instance ), &physicalDeviceGroupCount, nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && physicalDeviceGroupCount )
        {
          physicalDeviceGroupProperties.resize( physicalDeviceGroupCount );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkEnumeratePhysicalDeviceGroups(
            static_cast<VkInstance>( m_instance ),
            &physicalDeviceGroupCount,
            reinterpret_cast<VkPhysicalDeviceGroupProperties *>( physicalDeviceGroupProperties.data() ) ) );
          VULKAN_HPP_ASSERT( physicalDeviceGroupCount <= physicalDeviceGroupProperties.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
           ( physicalDeviceGroupCount < physicalDeviceGroupProperties.size() ) )
      {
        physicalDeviceGroupProperties.resize( physicalDeviceGroupCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Instance::enumeratePhysicalDeviceGroups" );
      }
      return physicalDeviceGroupProperties;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::MemoryRequirements2
                                           Device::getImageMemoryRequirements2( const ImageMemoryRequirementsInfo2 & info ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_NAMESPACE::MemoryRequirements2 memoryRequirements;
      getDispatcher()->vkGetImageMemoryRequirements2(
        static_cast<VkDevice>( m_device ),
        reinterpret_cast<const VkImageMemoryRequirementsInfo2 *>( &info ),
        reinterpret_cast<VkMemoryRequirements2 *>( &memoryRequirements ) );
      return memoryRequirements;
    }

    template <typename X, typename Y, typename... Z>
    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE StructureChain<X, Y, Z...>
                                           Device::getImageMemoryRequirements2( const ImageMemoryRequirementsInfo2 & info ) const VULKAN_HPP_NOEXCEPT
    {
      StructureChain<X, Y, Z...>                  structureChain;
      VULKAN_HPP_NAMESPACE::MemoryRequirements2 & memoryRequirements =
        structureChain.template get<VULKAN_HPP_NAMESPACE::MemoryRequirements2>();
      getDispatcher()->vkGetImageMemoryRequirements2(
        static_cast<VkDevice>( m_device ),
        reinterpret_cast<const VkImageMemoryRequirementsInfo2 *>( &info ),
        reinterpret_cast<VkMemoryRequirements2 *>( &memoryRequirements ) );
      return structureChain;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::MemoryRequirements2
                                           Device::getBufferMemoryRequirements2( const BufferMemoryRequirementsInfo2 & info ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_NAMESPACE::MemoryRequirements2 memoryRequirements;
      getDispatcher()->vkGetBufferMemoryRequirements2(
        static_cast<VkDevice>( m_device ),
        reinterpret_cast<const VkBufferMemoryRequirementsInfo2 *>( &info ),
        reinterpret_cast<VkMemoryRequirements2 *>( &memoryRequirements ) );
      return memoryRequirements;
    }

    template <typename X, typename Y, typename... Z>
    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE StructureChain<X, Y, Z...>
                                           Device::getBufferMemoryRequirements2( const BufferMemoryRequirementsInfo2 & info ) const VULKAN_HPP_NOEXCEPT
    {
      StructureChain<X, Y, Z...>                  structureChain;
      VULKAN_HPP_NAMESPACE::MemoryRequirements2 & memoryRequirements =
        structureChain.template get<VULKAN_HPP_NAMESPACE::MemoryRequirements2>();
      getDispatcher()->vkGetBufferMemoryRequirements2(
        static_cast<VkDevice>( m_device ),
        reinterpret_cast<const VkBufferMemoryRequirementsInfo2 *>( &info ),
        reinterpret_cast<VkMemoryRequirements2 *>( &memoryRequirements ) );
      return structureChain;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::SparseImageMemoryRequirements2>
                                           Device::getImageSparseMemoryRequirements2( const ImageSparseMemoryRequirementsInfo2 & info ) const
      VULKAN_HPP_NOEXCEPT
    {
      uint32_t sparseMemoryRequirementCount;
      getDispatcher()->vkGetImageSparseMemoryRequirements2(
        static_cast<VkDevice>( m_device ),
        reinterpret_cast<const VkImageSparseMemoryRequirementsInfo2 *>( &info ),
        &sparseMemoryRequirementCount,
        nullptr );
      std::vector<VULKAN_HPP_NAMESPACE::SparseImageMemoryRequirements2> sparseMemoryRequirements(
        sparseMemoryRequirementCount );
      getDispatcher()->vkGetImageSparseMemoryRequirements2(
        static_cast<VkDevice>( m_device ),
        reinterpret_cast<const VkImageSparseMemoryRequirementsInfo2 *>( &info ),
        &sparseMemoryRequirementCount,
        reinterpret_cast<VkSparseImageMemoryRequirements2 *>( sparseMemoryRequirements.data() ) );
      VULKAN_HPP_ASSERT( sparseMemoryRequirementCount <= sparseMemoryRequirements.size() );
      return sparseMemoryRequirements;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures2
                                           PhysicalDevice::getFeatures2() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures2 features;
      getDispatcher()->vkGetPhysicalDeviceFeatures2( static_cast<VkPhysicalDevice>( m_physicalDevice ),
                                                     reinterpret_cast<VkPhysicalDeviceFeatures2 *>( &features ) );
      return features;
    }

    template <typename X, typename Y, typename... Z>
    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE StructureChain<X, Y, Z...>
                                           PhysicalDevice::getFeatures2() const VULKAN_HPP_NOEXCEPT
    {
      StructureChain<X, Y, Z...>                      structureChain;
      VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures2 & features =
        structureChain.template get<VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures2>();
      getDispatcher()->vkGetPhysicalDeviceFeatures2( static_cast<VkPhysicalDevice>( m_physicalDevice ),
                                                     reinterpret_cast<VkPhysicalDeviceFeatures2 *>( &features ) );
      return structureChain;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties2
                                           PhysicalDevice::getProperties2() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties2 properties;
      getDispatcher()->vkGetPhysicalDeviceProperties2( static_cast<VkPhysicalDevice>( m_physicalDevice ),
                                                       reinterpret_cast<VkPhysicalDeviceProperties2 *>( &properties ) );
      return properties;
    }

    template <typename X, typename Y, typename... Z>
    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE StructureChain<X, Y, Z...>
                                           PhysicalDevice::getProperties2() const VULKAN_HPP_NOEXCEPT
    {
      StructureChain<X, Y, Z...>                        structureChain;
      VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties2 & properties =
        structureChain.template get<VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties2>();
      getDispatcher()->vkGetPhysicalDeviceProperties2( static_cast<VkPhysicalDevice>( m_physicalDevice ),
                                                       reinterpret_cast<VkPhysicalDeviceProperties2 *>( &properties ) );
      return structureChain;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::FormatProperties2
                                           PhysicalDevice::getFormatProperties2( VULKAN_HPP_NAMESPACE::Format format ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_NAMESPACE::FormatProperties2 formatProperties;
      getDispatcher()->vkGetPhysicalDeviceFormatProperties2(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        static_cast<VkFormat>( format ),
        reinterpret_cast<VkFormatProperties2 *>( &formatProperties ) );
      return formatProperties;
    }

    template <typename X, typename Y, typename... Z>
    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE StructureChain<X, Y, Z...>
                                           PhysicalDevice::getFormatProperties2( VULKAN_HPP_NAMESPACE::Format format ) const VULKAN_HPP_NOEXCEPT
    {
      StructureChain<X, Y, Z...>                structureChain;
      VULKAN_HPP_NAMESPACE::FormatProperties2 & formatProperties =
        structureChain.template get<VULKAN_HPP_NAMESPACE::FormatProperties2>();
      getDispatcher()->vkGetPhysicalDeviceFormatProperties2(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        static_cast<VkFormat>( format ),
        reinterpret_cast<VkFormatProperties2 *>( &formatProperties ) );
      return structureChain;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::ImageFormatProperties2
                                           PhysicalDevice::getImageFormatProperties2( const PhysicalDeviceImageFormatInfo2 & imageFormatInfo ) const
    {
      VULKAN_HPP_NAMESPACE::ImageFormatProperties2 imageFormatProperties;
      VULKAN_HPP_NAMESPACE::Result                 result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceImageFormatProperties2(
          static_cast<VkPhysicalDevice>( m_physicalDevice ),
          reinterpret_cast<const VkPhysicalDeviceImageFormatInfo2 *>( &imageFormatInfo ),
          reinterpret_cast<VkImageFormatProperties2 *>( &imageFormatProperties ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getImageFormatProperties2" );
      }
      return imageFormatProperties;
    }

    template <typename X, typename Y, typename... Z>
    VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...>
                         PhysicalDevice::getImageFormatProperties2( const PhysicalDeviceImageFormatInfo2 & imageFormatInfo ) const
    {
      StructureChain<X, Y, Z...>                     structureChain;
      VULKAN_HPP_NAMESPACE::ImageFormatProperties2 & imageFormatProperties =
        structureChain.template get<VULKAN_HPP_NAMESPACE::ImageFormatProperties2>();
      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceImageFormatProperties2(
          static_cast<VkPhysicalDevice>( m_physicalDevice ),
          reinterpret_cast<const VkPhysicalDeviceImageFormatInfo2 *>( &imageFormatInfo ),
          reinterpret_cast<VkImageFormatProperties2 *>( &imageFormatProperties ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getImageFormatProperties2" );
      }
      return structureChain;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::QueueFamilyProperties2>
                                           PhysicalDevice::getQueueFamilyProperties2() const VULKAN_HPP_NOEXCEPT
    {
      uint32_t queueFamilyPropertyCount;
      getDispatcher()->vkGetPhysicalDeviceQueueFamilyProperties2(
        static_cast<VkPhysicalDevice>( m_physicalDevice ), &queueFamilyPropertyCount, nullptr );
      std::vector<VULKAN_HPP_NAMESPACE::QueueFamilyProperties2> queueFamilyProperties( queueFamilyPropertyCount );
      getDispatcher()->vkGetPhysicalDeviceQueueFamilyProperties2(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        &queueFamilyPropertyCount,
        reinterpret_cast<VkQueueFamilyProperties2 *>( queueFamilyProperties.data() ) );
      VULKAN_HPP_ASSERT( queueFamilyPropertyCount <= queueFamilyProperties.size() );
      return queueFamilyProperties;
    }

    template <typename StructureChain>
    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<StructureChain> PhysicalDevice::getQueueFamilyProperties2() const
    {
      uint32_t queueFamilyPropertyCount;
      getDispatcher()->vkGetPhysicalDeviceQueueFamilyProperties2(
        static_cast<VkPhysicalDevice>( m_physicalDevice ), &queueFamilyPropertyCount, nullptr );
      std::vector<StructureChain>                               returnVector( queueFamilyPropertyCount );
      std::vector<VULKAN_HPP_NAMESPACE::QueueFamilyProperties2> queueFamilyProperties( queueFamilyPropertyCount );
      for ( uint32_t i = 0; i < queueFamilyPropertyCount; i++ )
      {
        queueFamilyProperties[i].pNext =
          returnVector[i].template get<VULKAN_HPP_NAMESPACE::QueueFamilyProperties2>().pNext;
      }
      getDispatcher()->vkGetPhysicalDeviceQueueFamilyProperties2(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        &queueFamilyPropertyCount,
        reinterpret_cast<VkQueueFamilyProperties2 *>( queueFamilyProperties.data() ) );
      VULKAN_HPP_ASSERT( queueFamilyPropertyCount <= queueFamilyProperties.size() );
      for ( uint32_t i = 0; i < queueFamilyPropertyCount; i++ )
      {
        returnVector[i].template get<VULKAN_HPP_NAMESPACE::QueueFamilyProperties2>() = queueFamilyProperties[i];
      }
      return returnVector;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties2
                                           PhysicalDevice::getMemoryProperties2() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties2 memoryProperties;
      getDispatcher()->vkGetPhysicalDeviceMemoryProperties2(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        reinterpret_cast<VkPhysicalDeviceMemoryProperties2 *>( &memoryProperties ) );
      return memoryProperties;
    }

    template <typename X, typename Y, typename... Z>
    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE StructureChain<X, Y, Z...>
                                           PhysicalDevice::getMemoryProperties2() const VULKAN_HPP_NOEXCEPT
    {
      StructureChain<X, Y, Z...>                              structureChain;
      VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties2 & memoryProperties =
        structureChain.template get<VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties2>();
      getDispatcher()->vkGetPhysicalDeviceMemoryProperties2(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        reinterpret_cast<VkPhysicalDeviceMemoryProperties2 *>( &memoryProperties ) );
      return structureChain;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::SparseImageFormatProperties2>
                                           PhysicalDevice::getSparseImageFormatProperties2( const PhysicalDeviceSparseImageFormatInfo2 & formatInfo ) const
      VULKAN_HPP_NOEXCEPT
    {
      uint32_t propertyCount;
      getDispatcher()->vkGetPhysicalDeviceSparseImageFormatProperties2(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        reinterpret_cast<const VkPhysicalDeviceSparseImageFormatInfo2 *>( &formatInfo ),
        &propertyCount,
        nullptr );
      std::vector<VULKAN_HPP_NAMESPACE::SparseImageFormatProperties2> properties( propertyCount );
      getDispatcher()->vkGetPhysicalDeviceSparseImageFormatProperties2(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        reinterpret_cast<const VkPhysicalDeviceSparseImageFormatInfo2 *>( &formatInfo ),
        &propertyCount,
        reinterpret_cast<VkSparseImageFormatProperties2 *>( properties.data() ) );
      VULKAN_HPP_ASSERT( propertyCount <= properties.size() );
      return properties;
    }

    VULKAN_HPP_INLINE void
      CommandPool::trim( VULKAN_HPP_NAMESPACE::CommandPoolTrimFlags flags ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkTrimCommandPool( static_cast<VkDevice>( m_device ),
                                          static_cast<VkCommandPool>( m_commandPool ),
                                          static_cast<VkCommandPoolTrimFlags>( flags ) );
    }

    VULKAN_HPP_INLINE void
      DescriptorSet::updateWithTemplate( VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplate descriptorUpdateTemplate,
                                         const void * pData ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkUpdateDescriptorSetWithTemplate(
        static_cast<VkDevice>( m_device ),
        static_cast<VkDescriptorSet>( m_descriptorSet ),
        static_cast<VkDescriptorUpdateTemplate>( descriptorUpdateTemplate ),
        pData );
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::ExternalBufferProperties
                                           PhysicalDevice::getExternalBufferProperties( const PhysicalDeviceExternalBufferInfo & externalBufferInfo ) const
      VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_NAMESPACE::ExternalBufferProperties externalBufferProperties;
      getDispatcher()->vkGetPhysicalDeviceExternalBufferProperties(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        reinterpret_cast<const VkPhysicalDeviceExternalBufferInfo *>( &externalBufferInfo ),
        reinterpret_cast<VkExternalBufferProperties *>( &externalBufferProperties ) );
      return externalBufferProperties;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::ExternalFenceProperties
                                           PhysicalDevice::getExternalFenceProperties( const PhysicalDeviceExternalFenceInfo & externalFenceInfo ) const
      VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_NAMESPACE::ExternalFenceProperties externalFenceProperties;
      getDispatcher()->vkGetPhysicalDeviceExternalFenceProperties(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        reinterpret_cast<const VkPhysicalDeviceExternalFenceInfo *>( &externalFenceInfo ),
        reinterpret_cast<VkExternalFenceProperties *>( &externalFenceProperties ) );
      return externalFenceProperties;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::ExternalSemaphoreProperties
                                           PhysicalDevice::getExternalSemaphoreProperties(
        const PhysicalDeviceExternalSemaphoreInfo & externalSemaphoreInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_NAMESPACE::ExternalSemaphoreProperties externalSemaphoreProperties;
      getDispatcher()->vkGetPhysicalDeviceExternalSemaphoreProperties(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        reinterpret_cast<const VkPhysicalDeviceExternalSemaphoreInfo *>( &externalSemaphoreInfo ),
        reinterpret_cast<VkExternalSemaphoreProperties *>( &externalSemaphoreProperties ) );
      return externalSemaphoreProperties;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::DescriptorSetLayoutSupport
                                           Device::getDescriptorSetLayoutSupport( const DescriptorSetLayoutCreateInfo & createInfo ) const
      VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_NAMESPACE::DescriptorSetLayoutSupport support;
      getDispatcher()->vkGetDescriptorSetLayoutSupport(
        static_cast<VkDevice>( m_device ),
        reinterpret_cast<const VkDescriptorSetLayoutCreateInfo *>( &createInfo ),
        reinterpret_cast<VkDescriptorSetLayoutSupport *>( &support ) );
      return support;
    }

    template <typename X, typename Y, typename... Z>
    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE StructureChain<X, Y, Z...> Device::getDescriptorSetLayoutSupport(
      const DescriptorSetLayoutCreateInfo & createInfo ) const VULKAN_HPP_NOEXCEPT
    {
      StructureChain<X, Y, Z...>                         structureChain;
      VULKAN_HPP_NAMESPACE::DescriptorSetLayoutSupport & support =
        structureChain.template get<VULKAN_HPP_NAMESPACE::DescriptorSetLayoutSupport>();
      getDispatcher()->vkGetDescriptorSetLayoutSupport(
        static_cast<VkDevice>( m_device ),
        reinterpret_cast<const VkDescriptorSetLayoutCreateInfo *>( &createInfo ),
        reinterpret_cast<VkDescriptorSetLayoutSupport *>( &support ) );
      return structureChain;
    }

    //=== VK_VERSION_1_2 ===

    VULKAN_HPP_INLINE void CommandBuffer::drawIndirectCount( VULKAN_HPP_NAMESPACE::Buffer     buffer,
                                                             VULKAN_HPP_NAMESPACE::DeviceSize offset,
                                                             VULKAN_HPP_NAMESPACE::Buffer     countBuffer,
                                                             VULKAN_HPP_NAMESPACE::DeviceSize countBufferOffset,
                                                             uint32_t                         maxDrawCount,
                                                             uint32_t stride ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdDrawIndirectCount( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                               static_cast<VkBuffer>( buffer ),
                                               static_cast<VkDeviceSize>( offset ),
                                               static_cast<VkBuffer>( countBuffer ),
                                               static_cast<VkDeviceSize>( countBufferOffset ),
                                               maxDrawCount,
                                               stride );
    }

    VULKAN_HPP_INLINE void CommandBuffer::drawIndexedIndirectCount( VULKAN_HPP_NAMESPACE::Buffer     buffer,
                                                                    VULKAN_HPP_NAMESPACE::DeviceSize offset,
                                                                    VULKAN_HPP_NAMESPACE::Buffer     countBuffer,
                                                                    VULKAN_HPP_NAMESPACE::DeviceSize countBufferOffset,
                                                                    uint32_t                         maxDrawCount,
                                                                    uint32_t stride ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdDrawIndexedIndirectCount( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                      static_cast<VkBuffer>( buffer ),
                                                      static_cast<VkDeviceSize>( offset ),
                                                      static_cast<VkBuffer>( countBuffer ),
                                                      static_cast<VkDeviceSize>( countBufferOffset ),
                                                      maxDrawCount,
                                                      stride );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::beginRenderPass2( const RenderPassBeginInfo & renderPassBegin,
                                       const SubpassBeginInfo &    subpassBeginInfo ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdBeginRenderPass2( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                              reinterpret_cast<const VkRenderPassBeginInfo *>( &renderPassBegin ),
                                              reinterpret_cast<const VkSubpassBeginInfo *>( &subpassBeginInfo ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::nextSubpass2( const SubpassBeginInfo & subpassBeginInfo,
                                   const SubpassEndInfo &   subpassEndInfo ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdNextSubpass2( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                          reinterpret_cast<const VkSubpassBeginInfo *>( &subpassBeginInfo ),
                                          reinterpret_cast<const VkSubpassEndInfo *>( &subpassEndInfo ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::endRenderPass2( const SubpassEndInfo & subpassEndInfo ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkCmdEndRenderPass2( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                            reinterpret_cast<const VkSubpassEndInfo *>( &subpassEndInfo ) );
    }

    VULKAN_HPP_INLINE void QueryPool::reset( uint32_t firstQuery, uint32_t queryCount ) const VULKAN_HPP_NOEXCEPT
    {
      getDispatcher()->vkResetQueryPool(
        static_cast<VkDevice>( m_device ), static_cast<VkQueryPool>( m_queryPool ), firstQuery, queryCount );
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE uint64_t Semaphore::getCounterValue() const
    {
      uint64_t                     value;
      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetSemaphoreCounterValue(
          static_cast<VkDevice>( m_device ), static_cast<VkSemaphore>( m_semaphore ), &value ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Semaphore::getCounterValue" );
      }
      return value;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::Result
                                           Device::waitSemaphores( const SemaphoreWaitInfo & waitInfo, uint64_t timeout ) const
    {
      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkWaitSemaphores(
          static_cast<VkDevice>( m_device ), reinterpret_cast<const VkSemaphoreWaitInfo *>( &waitInfo ), timeout ) );
      if ( ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eTimeout ) )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::waitSemaphores" );
      }
      return result;
    }

    VULKAN_HPP_INLINE void Device::signalSemaphore( const SemaphoreSignalInfo & signalInfo ) const
    {
      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkSignalSemaphore(
          static_cast<VkDevice>( m_device ), reinterpret_cast<const VkSemaphoreSignalInfo *>( &signalInfo ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::signalSemaphore" );
      }
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::DeviceAddress
                                           Device::getBufferAddress( const BufferDeviceAddressInfo & info ) const VULKAN_HPP_NOEXCEPT
    {
      return static_cast<VULKAN_HPP_NAMESPACE::DeviceAddress>( getDispatcher()->vkGetBufferDeviceAddress(
        static_cast<VkDevice>( m_device ), reinterpret_cast<const VkBufferDeviceAddressInfo *>( &info ) ) );
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE uint64_t
                                           Device::getBufferOpaqueCaptureAddress( const BufferDeviceAddressInfo & info ) const VULKAN_HPP_NOEXCEPT
    {
      return getDispatcher()->vkGetBufferOpaqueCaptureAddress(
        static_cast<VkDevice>( m_device ), reinterpret_cast<const VkBufferDeviceAddressInfo *>( &info ) );
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE uint64_t Device::getMemoryOpaqueCaptureAddress(
      const DeviceMemoryOpaqueCaptureAddressInfo & info ) const VULKAN_HPP_NOEXCEPT
    {
      return getDispatcher()->vkGetDeviceMemoryOpaqueCaptureAddress(
        static_cast<VkDevice>( m_device ), reinterpret_cast<const VkDeviceMemoryOpaqueCaptureAddressInfo *>( &info ) );
    }

    //=== VK_KHR_surface ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::Bool32
                                           PhysicalDevice::getSurfaceSupportKHR( uint32_t queueFamilyIndex, VULKAN_HPP_NAMESPACE::SurfaceKHR surface ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkGetPhysicalDeviceSurfaceSupportKHR &&
                         "Function <vkGetPhysicalDeviceSurfaceSupportKHR> needs extension <VK_KHR_surface> enabled!" );

      VULKAN_HPP_NAMESPACE::Bool32 supported;
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkGetPhysicalDeviceSurfaceSupportKHR( static_cast<VkPhysicalDevice>( m_physicalDevice ),
                                                               queueFamilyIndex,
                                                               static_cast<VkSurfaceKHR>( surface ),
                                                               reinterpret_cast<VkBool32 *>( &supported ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getSurfaceSupportKHR" );
      }
      return supported;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesKHR
                                           PhysicalDevice::getSurfaceCapabilitiesKHR( VULKAN_HPP_NAMESPACE::SurfaceKHR surface ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceSurfaceCapabilitiesKHR &&
        "Function <vkGetPhysicalDeviceSurfaceCapabilitiesKHR> needs extension <VK_KHR_surface> enabled!" );

      VULKAN_HPP_NAMESPACE::SurfaceCapabilitiesKHR surfaceCapabilities;
      VULKAN_HPP_NAMESPACE::Result                 result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
          static_cast<VkPhysicalDevice>( m_physicalDevice ),
          static_cast<VkSurfaceKHR>( surface ),
          reinterpret_cast<VkSurfaceCapabilitiesKHR *>( &surfaceCapabilities ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getSurfaceCapabilitiesKHR" );
      }
      return surfaceCapabilities;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::SurfaceFormatKHR>
                                           PhysicalDevice::getSurfaceFormatsKHR( VULKAN_HPP_NAMESPACE::SurfaceKHR surface ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkGetPhysicalDeviceSurfaceFormatsKHR &&
                         "Function <vkGetPhysicalDeviceSurfaceFormatsKHR> needs extension <VK_KHR_surface> enabled!" );

      std::vector<VULKAN_HPP_NAMESPACE::SurfaceFormatKHR> surfaceFormats;
      uint32_t                                            surfaceFormatCount;
      VULKAN_HPP_NAMESPACE::Result                        result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkGetPhysicalDeviceSurfaceFormatsKHR( static_cast<VkPhysicalDevice>( m_physicalDevice ),
                                                                 static_cast<VkSurfaceKHR>( surface ),
                                                                 &surfaceFormatCount,
                                                                 nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && surfaceFormatCount )
        {
          surfaceFormats.resize( surfaceFormatCount );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceSurfaceFormatsKHR(
            static_cast<VkPhysicalDevice>( m_physicalDevice ),
            static_cast<VkSurfaceKHR>( surface ),
            &surfaceFormatCount,
            reinterpret_cast<VkSurfaceFormatKHR *>( surfaceFormats.data() ) ) );
          VULKAN_HPP_ASSERT( surfaceFormatCount <= surfaceFormats.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( surfaceFormatCount < surfaceFormats.size() ) )
      {
        surfaceFormats.resize( surfaceFormatCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getSurfaceFormatsKHR" );
      }
      return surfaceFormats;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::PresentModeKHR>
                                           PhysicalDevice::getSurfacePresentModesKHR( VULKAN_HPP_NAMESPACE::SurfaceKHR surface ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceSurfacePresentModesKHR &&
        "Function <vkGetPhysicalDeviceSurfacePresentModesKHR> needs extension <VK_KHR_surface> enabled!" );

      std::vector<VULKAN_HPP_NAMESPACE::PresentModeKHR> presentModes;
      uint32_t                                          presentModeCount;
      VULKAN_HPP_NAMESPACE::Result                      result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkGetPhysicalDeviceSurfacePresentModesKHR( static_cast<VkPhysicalDevice>( m_physicalDevice ),
                                                                      static_cast<VkSurfaceKHR>( surface ),
                                                                      &presentModeCount,
                                                                      nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && presentModeCount )
        {
          presentModes.resize( presentModeCount );
          result =
            static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceSurfacePresentModesKHR(
              static_cast<VkPhysicalDevice>( m_physicalDevice ),
              static_cast<VkSurfaceKHR>( surface ),
              &presentModeCount,
              reinterpret_cast<VkPresentModeKHR *>( presentModes.data() ) ) );
          VULKAN_HPP_ASSERT( presentModeCount <= presentModes.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( presentModeCount < presentModes.size() ) )
      {
        presentModes.resize( presentModeCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getSurfacePresentModesKHR" );
      }
      return presentModes;
    }

    //=== VK_KHR_swapchain ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VkImage> SwapchainKHR::getImages() const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkGetSwapchainImagesKHR &&
                         "Function <vkGetSwapchainImagesKHR> needs extension <VK_KHR_swapchain> enabled!" );

      std::vector<VkImage>         swapchainImages;
      uint32_t                     swapchainImageCount;
      VULKAN_HPP_NAMESPACE::Result result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkGetSwapchainImagesKHR( static_cast<VkDevice>( m_device ),
                                                    static_cast<VkSwapchainKHR>( m_swapchainKHR ),
                                                    &swapchainImageCount,
                                                    nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && swapchainImageCount )
        {
          swapchainImages.resize( swapchainImageCount );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
            getDispatcher()->vkGetSwapchainImagesKHR( static_cast<VkDevice>( m_device ),
                                                      static_cast<VkSwapchainKHR>( m_swapchainKHR ),
                                                      &swapchainImageCount,
                                                      swapchainImages.data() ) );
          VULKAN_HPP_ASSERT( swapchainImageCount <= swapchainImages.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( swapchainImageCount < swapchainImages.size() ) )
      {
        swapchainImages.resize( swapchainImageCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::SwapchainKHR::getImages" );
      }
      return swapchainImages;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::pair<VULKAN_HPP_NAMESPACE::Result, uint32_t>
                                           SwapchainKHR::acquireNextImage( uint64_t                        timeout,
                                      VULKAN_HPP_NAMESPACE::Semaphore semaphore,
                                      VULKAN_HPP_NAMESPACE::Fence     fence ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkAcquireNextImageKHR &&
                         "Function <vkAcquireNextImageKHR> needs extension <VK_KHR_swapchain> enabled!" );

      uint32_t                     imageIndex;
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkAcquireNextImageKHR( static_cast<VkDevice>( m_device ),
                                                static_cast<VkSwapchainKHR>( m_swapchainKHR ),
                                                timeout,
                                                static_cast<VkSemaphore>( semaphore ),
                                                static_cast<VkFence>( fence ),
                                                &imageIndex ) );
      if ( ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eTimeout ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eNotReady ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eSuboptimalKHR ) )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::SwapchainKHR::acquireNextImage" );
      }
      return std::make_pair( result, imageIndex );
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::Result
                                           Queue::presentKHR( const PresentInfoKHR & presentInfo ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkQueuePresentKHR &&
                         "Function <vkQueuePresentKHR> needs extension <VK_KHR_swapchain> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkQueuePresentKHR(
          static_cast<VkQueue>( m_queue ), reinterpret_cast<const VkPresentInfoKHR *>( &presentInfo ) ) );
      if ( ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eSuboptimalKHR ) )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Queue::presentKHR" );
      }
      return result;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::DeviceGroupPresentCapabilitiesKHR
                                           Device::getGroupPresentCapabilitiesKHR() const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetDeviceGroupPresentCapabilitiesKHR &&
        "Function <vkGetDeviceGroupPresentCapabilitiesKHR> needs extension <VK_KHR_swapchain> enabled!" );

      VULKAN_HPP_NAMESPACE::DeviceGroupPresentCapabilitiesKHR deviceGroupPresentCapabilities;
      VULKAN_HPP_NAMESPACE::Result                            result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetDeviceGroupPresentCapabilitiesKHR(
          static_cast<VkDevice>( m_device ),
          reinterpret_cast<VkDeviceGroupPresentCapabilitiesKHR *>( &deviceGroupPresentCapabilities ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::getGroupPresentCapabilitiesKHR" );
      }
      return deviceGroupPresentCapabilities;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::DeviceGroupPresentModeFlagsKHR
                                           Device::getGroupSurfacePresentModesKHR( VULKAN_HPP_NAMESPACE::SurfaceKHR surface ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetDeviceGroupSurfacePresentModesKHR &&
        "Function <vkGetDeviceGroupSurfacePresentModesKHR> needs extension <VK_KHR_swapchain> enabled!" );

      VULKAN_HPP_NAMESPACE::DeviceGroupPresentModeFlagsKHR modes;
      VULKAN_HPP_NAMESPACE::Result                         result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetDeviceGroupSurfacePresentModesKHR(
          static_cast<VkDevice>( m_device ),
          static_cast<VkSurfaceKHR>( surface ),
          reinterpret_cast<VkDeviceGroupPresentModeFlagsKHR *>( &modes ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::getGroupSurfacePresentModesKHR" );
      }
      return modes;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::Rect2D>
                                           PhysicalDevice::getPresentRectanglesKHR( VULKAN_HPP_NAMESPACE::SurfaceKHR surface ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDevicePresentRectanglesKHR &&
        "Function <vkGetPhysicalDevicePresentRectanglesKHR> needs extension <VK_KHR_swapchain> enabled!" );

      std::vector<VULKAN_HPP_NAMESPACE::Rect2D> rects;
      uint32_t                                  rectCount;
      VULKAN_HPP_NAMESPACE::Result              result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkGetPhysicalDevicePresentRectanglesKHR( static_cast<VkPhysicalDevice>( m_physicalDevice ),
                                                                    static_cast<VkSurfaceKHR>( surface ),
                                                                    &rectCount,
                                                                    nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && rectCount )
        {
          rects.resize( rectCount );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
            getDispatcher()->vkGetPhysicalDevicePresentRectanglesKHR( static_cast<VkPhysicalDevice>( m_physicalDevice ),
                                                                      static_cast<VkSurfaceKHR>( surface ),
                                                                      &rectCount,
                                                                      reinterpret_cast<VkRect2D *>( rects.data() ) ) );
          VULKAN_HPP_ASSERT( rectCount <= rects.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( rectCount < rects.size() ) )
      {
        rects.resize( rectCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getPresentRectanglesKHR" );
      }
      return rects;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::pair<VULKAN_HPP_NAMESPACE::Result, uint32_t>
                                           Device::acquireNextImage2KHR( const AcquireNextImageInfoKHR & acquireInfo ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkAcquireNextImage2KHR &&
                         "Function <vkAcquireNextImage2KHR> needs extension <VK_KHR_swapchain> enabled!" );

      uint32_t                     imageIndex;
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkAcquireNextImage2KHR( static_cast<VkDevice>( m_device ),
                                                 reinterpret_cast<const VkAcquireNextImageInfoKHR *>( &acquireInfo ),
                                                 &imageIndex ) );
      if ( ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eTimeout ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eNotReady ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eSuboptimalKHR ) )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::acquireNextImage2KHR" );
      }
      return std::make_pair( result, imageIndex );
    }

    //=== VK_KHR_display ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::DisplayPropertiesKHR>
                                           PhysicalDevice::getDisplayPropertiesKHR() const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceDisplayPropertiesKHR &&
        "Function <vkGetPhysicalDeviceDisplayPropertiesKHR> needs extension <VK_KHR_display> enabled!" );

      std::vector<VULKAN_HPP_NAMESPACE::DisplayPropertiesKHR> properties;
      uint32_t                                                propertyCount;
      VULKAN_HPP_NAMESPACE::Result                            result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceDisplayPropertiesKHR(
          static_cast<VkPhysicalDevice>( m_physicalDevice ), &propertyCount, nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && propertyCount )
        {
          properties.resize( propertyCount );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceDisplayPropertiesKHR(
            static_cast<VkPhysicalDevice>( m_physicalDevice ),
            &propertyCount,
            reinterpret_cast<VkDisplayPropertiesKHR *>( properties.data() ) ) );
          VULKAN_HPP_ASSERT( propertyCount <= properties.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( propertyCount < properties.size() ) )
      {
        properties.resize( propertyCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getDisplayPropertiesKHR" );
      }
      return properties;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::DisplayPlanePropertiesKHR>
                                           PhysicalDevice::getDisplayPlanePropertiesKHR() const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceDisplayPlanePropertiesKHR &&
        "Function <vkGetPhysicalDeviceDisplayPlanePropertiesKHR> needs extension <VK_KHR_display> enabled!" );

      std::vector<VULKAN_HPP_NAMESPACE::DisplayPlanePropertiesKHR> properties;
      uint32_t                                                     propertyCount;
      VULKAN_HPP_NAMESPACE::Result                                 result;
      do
      {
        result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceDisplayPlanePropertiesKHR(
            static_cast<VkPhysicalDevice>( m_physicalDevice ), &propertyCount, nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && propertyCount )
        {
          properties.resize( propertyCount );
          result =
            static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceDisplayPlanePropertiesKHR(
              static_cast<VkPhysicalDevice>( m_physicalDevice ),
              &propertyCount,
              reinterpret_cast<VkDisplayPlanePropertiesKHR *>( properties.data() ) ) );
          VULKAN_HPP_ASSERT( propertyCount <= properties.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( propertyCount < properties.size() ) )
      {
        properties.resize( propertyCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getDisplayPlanePropertiesKHR" );
      }
      return properties;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::DisplayModePropertiesKHR>
                                           DisplayKHR::getModeProperties() const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkGetDisplayModePropertiesKHR &&
                         "Function <vkGetDisplayModePropertiesKHR> needs extension <VK_KHR_display> enabled!" );

      std::vector<VULKAN_HPP_NAMESPACE::DisplayModePropertiesKHR> properties;
      uint32_t                                                    propertyCount;
      VULKAN_HPP_NAMESPACE::Result                                result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkGetDisplayModePropertiesKHR( static_cast<VkPhysicalDevice>( m_physicalDevice ),
                                                          static_cast<VkDisplayKHR>( m_displayKHR ),
                                                          &propertyCount,
                                                          nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && propertyCount )
        {
          properties.resize( propertyCount );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetDisplayModePropertiesKHR(
            static_cast<VkPhysicalDevice>( m_physicalDevice ),
            static_cast<VkDisplayKHR>( m_displayKHR ),
            &propertyCount,
            reinterpret_cast<VkDisplayModePropertiesKHR *>( properties.data() ) ) );
          VULKAN_HPP_ASSERT( propertyCount <= properties.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( propertyCount < properties.size() ) )
      {
        properties.resize( propertyCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::DisplayKHR::getModeProperties" );
      }
      return properties;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::DisplayPlaneCapabilitiesKHR
                                           DisplayModeKHR::getDisplayPlaneCapabilities( uint32_t planeIndex ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkGetDisplayPlaneCapabilitiesKHR &&
                         "Function <vkGetDisplayPlaneCapabilitiesKHR> needs extension <VK_KHR_display> enabled!" );

      VULKAN_HPP_NAMESPACE::DisplayPlaneCapabilitiesKHR capabilities;
      VULKAN_HPP_NAMESPACE::Result                      result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetDisplayPlaneCapabilitiesKHR(
          static_cast<VkPhysicalDevice>( m_physicalDevice ),
          static_cast<VkDisplayModeKHR>( m_displayModeKHR ),
          planeIndex,
          reinterpret_cast<VkDisplayPlaneCapabilitiesKHR *>( &capabilities ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::DisplayModeKHR::getDisplayPlaneCapabilities" );
      }
      return capabilities;
    }

#  if defined( VK_USE_PLATFORM_XLIB_KHR )
    //=== VK_KHR_xlib_surface ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::Bool32 PhysicalDevice::getXlibPresentationSupportKHR(
      uint32_t queueFamilyIndex, Display & dpy, VisualID visualID ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceXlibPresentationSupportKHR &&
        "Function <vkGetPhysicalDeviceXlibPresentationSupportKHR> needs extension <VK_KHR_xlib_surface> enabled!" );

      return static_cast<VULKAN_HPP_NAMESPACE::Bool32>( getDispatcher()->vkGetPhysicalDeviceXlibPresentationSupportKHR(
        static_cast<VkPhysicalDevice>( m_physicalDevice ), queueFamilyIndex, &dpy, visualID ) );
    }
#  endif /*VK_USE_PLATFORM_XLIB_KHR*/

#  if defined( VK_USE_PLATFORM_XCB_KHR )
    //=== VK_KHR_xcb_surface ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::Bool32 PhysicalDevice::getXcbPresentationSupportKHR(
      uint32_t queueFamilyIndex, xcb_connection_t & connection, xcb_visualid_t visual_id ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceXcbPresentationSupportKHR &&
        "Function <vkGetPhysicalDeviceXcbPresentationSupportKHR> needs extension <VK_KHR_xcb_surface> enabled!" );

      return static_cast<VULKAN_HPP_NAMESPACE::Bool32>( getDispatcher()->vkGetPhysicalDeviceXcbPresentationSupportKHR(
        static_cast<VkPhysicalDevice>( m_physicalDevice ), queueFamilyIndex, &connection, visual_id ) );
    }
#  endif /*VK_USE_PLATFORM_XCB_KHR*/

#  if defined( VK_USE_PLATFORM_WAYLAND_KHR )
    //=== VK_KHR_wayland_surface ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::Bool32
                                           PhysicalDevice::getWaylandPresentationSupportKHR( uint32_t            queueFamilyIndex,
                                                        struct wl_display & display ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceWaylandPresentationSupportKHR &&
        "Function <vkGetPhysicalDeviceWaylandPresentationSupportKHR> needs extension <VK_KHR_wayland_surface> enabled!" );

      return static_cast<VULKAN_HPP_NAMESPACE::Bool32>(
        getDispatcher()->vkGetPhysicalDeviceWaylandPresentationSupportKHR(
          static_cast<VkPhysicalDevice>( m_physicalDevice ), queueFamilyIndex, &display ) );
    }
#  endif /*VK_USE_PLATFORM_WAYLAND_KHR*/

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_KHR_win32_surface ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::Bool32
                                           PhysicalDevice::getWin32PresentationSupportKHR( uint32_t queueFamilyIndex ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceWin32PresentationSupportKHR &&
        "Function <vkGetPhysicalDeviceWin32PresentationSupportKHR> needs extension <VK_KHR_win32_surface> enabled!" );

      return static_cast<VULKAN_HPP_NAMESPACE::Bool32>( getDispatcher()->vkGetPhysicalDeviceWin32PresentationSupportKHR(
        static_cast<VkPhysicalDevice>( m_physicalDevice ), queueFamilyIndex ) );
    }
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

    //=== VK_EXT_debug_report ===

    VULKAN_HPP_INLINE void Instance::debugReportMessageEXT( VULKAN_HPP_NAMESPACE::DebugReportFlagsEXT      flags,
                                                            VULKAN_HPP_NAMESPACE::DebugReportObjectTypeEXT objectType_,
                                                            uint64_t                                       object,
                                                            size_t                                         location,
                                                            int32_t                                        messageCode,
                                                            const std::string &                            layerPrefix,
                                                            const std::string & message ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkDebugReportMessageEXT &&
                         "Function <vkDebugReportMessageEXT> needs extension <VK_EXT_debug_report> enabled!" );

      getDispatcher()->vkDebugReportMessageEXT( static_cast<VkInstance>( m_instance ),
                                                static_cast<VkDebugReportFlagsEXT>( flags ),
                                                static_cast<VkDebugReportObjectTypeEXT>( objectType_ ),
                                                object,
                                                location,
                                                messageCode,
                                                layerPrefix.c_str(),
                                                message.c_str() );
    }

    //=== VK_EXT_debug_marker ===

    VULKAN_HPP_INLINE void Device::debugMarkerSetObjectTagEXT( const DebugMarkerObjectTagInfoEXT & tagInfo ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkDebugMarkerSetObjectTagEXT &&
                         "Function <vkDebugMarkerSetObjectTagEXT> needs extension <VK_EXT_debug_marker> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkDebugMarkerSetObjectTagEXT(
          static_cast<VkDevice>( m_device ), reinterpret_cast<const VkDebugMarkerObjectTagInfoEXT *>( &tagInfo ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::debugMarkerSetObjectTagEXT" );
      }
    }

    VULKAN_HPP_INLINE void Device::debugMarkerSetObjectNameEXT( const DebugMarkerObjectNameInfoEXT & nameInfo ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkDebugMarkerSetObjectNameEXT &&
                         "Function <vkDebugMarkerSetObjectNameEXT> needs extension <VK_EXT_debug_marker> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkDebugMarkerSetObjectNameEXT(
          static_cast<VkDevice>( m_device ), reinterpret_cast<const VkDebugMarkerObjectNameInfoEXT *>( &nameInfo ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::debugMarkerSetObjectNameEXT" );
      }
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::debugMarkerBeginEXT( const DebugMarkerMarkerInfoEXT & markerInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdDebugMarkerBeginEXT &&
                         "Function <vkCmdDebugMarkerBeginEXT> needs extension <VK_EXT_debug_marker> enabled!" );

      getDispatcher()->vkCmdDebugMarkerBeginEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                 reinterpret_cast<const VkDebugMarkerMarkerInfoEXT *>( &markerInfo ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::debugMarkerEndEXT() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdDebugMarkerEndEXT &&
                         "Function <vkCmdDebugMarkerEndEXT> needs extension <VK_EXT_debug_marker> enabled!" );

      getDispatcher()->vkCmdDebugMarkerEndEXT( static_cast<VkCommandBuffer>( m_commandBuffer ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::debugMarkerInsertEXT( const DebugMarkerMarkerInfoEXT & markerInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdDebugMarkerInsertEXT &&
                         "Function <vkCmdDebugMarkerInsertEXT> needs extension <VK_EXT_debug_marker> enabled!" );

      getDispatcher()->vkCmdDebugMarkerInsertEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                  reinterpret_cast<const VkDebugMarkerMarkerInfoEXT *>( &markerInfo ) );
    }

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
    //=== VK_KHR_video_queue ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::VideoCapabilitiesKHR
                                           PhysicalDevice::getVideoCapabilitiesKHR( const VideoProfileKHR & videoProfile ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceVideoCapabilitiesKHR &&
        "Function <vkGetPhysicalDeviceVideoCapabilitiesKHR> needs extension <VK_KHR_video_queue> enabled!" );

      VULKAN_HPP_NAMESPACE::VideoCapabilitiesKHR capabilities;
      VULKAN_HPP_NAMESPACE::Result               result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceVideoCapabilitiesKHR(
          static_cast<VkPhysicalDevice>( m_physicalDevice ),
          reinterpret_cast<const VkVideoProfileKHR *>( &videoProfile ),
          reinterpret_cast<VkVideoCapabilitiesKHR *>( &capabilities ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getVideoCapabilitiesKHR" );
      }
      return capabilities;
    }

    template <typename X, typename Y, typename... Z>
    VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...>
                         PhysicalDevice::getVideoCapabilitiesKHR( const VideoProfileKHR & videoProfile ) const
    {
      StructureChain<X, Y, Z...>                   structureChain;
      VULKAN_HPP_NAMESPACE::VideoCapabilitiesKHR & capabilities =
        structureChain.template get<VULKAN_HPP_NAMESPACE::VideoCapabilitiesKHR>();
      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceVideoCapabilitiesKHR(
          static_cast<VkPhysicalDevice>( m_physicalDevice ),
          reinterpret_cast<const VkVideoProfileKHR *>( &videoProfile ),
          reinterpret_cast<VkVideoCapabilitiesKHR *>( &capabilities ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getVideoCapabilitiesKHR" );
      }
      return structureChain;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::VideoFormatPropertiesKHR>
                                           PhysicalDevice::getVideoFormatPropertiesKHR( const PhysicalDeviceVideoFormatInfoKHR & videoFormatInfo ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceVideoFormatPropertiesKHR &&
        "Function <vkGetPhysicalDeviceVideoFormatPropertiesKHR> needs extension <VK_KHR_video_queue> enabled!" );

      std::vector<VULKAN_HPP_NAMESPACE::VideoFormatPropertiesKHR> videoFormatProperties;
      uint32_t                                                    videoFormatPropertyCount;
      VULKAN_HPP_NAMESPACE::Result                                result;
      do
      {
        result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceVideoFormatPropertiesKHR(
            static_cast<VkPhysicalDevice>( m_physicalDevice ),
            reinterpret_cast<const VkPhysicalDeviceVideoFormatInfoKHR *>( &videoFormatInfo ),
            &videoFormatPropertyCount,
            nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && videoFormatPropertyCount )
        {
          videoFormatProperties.resize( videoFormatPropertyCount );
          result =
            static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceVideoFormatPropertiesKHR(
              static_cast<VkPhysicalDevice>( m_physicalDevice ),
              reinterpret_cast<const VkPhysicalDeviceVideoFormatInfoKHR *>( &videoFormatInfo ),
              &videoFormatPropertyCount,
              reinterpret_cast<VkVideoFormatPropertiesKHR *>( videoFormatProperties.data() ) ) );
          VULKAN_HPP_ASSERT( videoFormatPropertyCount <= videoFormatProperties.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
           ( videoFormatPropertyCount < videoFormatProperties.size() ) )
      {
        videoFormatProperties.resize( videoFormatPropertyCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getVideoFormatPropertiesKHR" );
      }
      return videoFormatProperties;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::VideoGetMemoryPropertiesKHR>
                                           VideoSessionKHR::getMemoryRequirements() const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetVideoSessionMemoryRequirementsKHR &&
        "Function <vkGetVideoSessionMemoryRequirementsKHR> needs extension <VK_KHR_video_queue> enabled!" );

      std::vector<VULKAN_HPP_NAMESPACE::VideoGetMemoryPropertiesKHR> videoSessionMemoryRequirements;
      uint32_t                                                       videoSessionMemoryRequirementsCount;
      VULKAN_HPP_NAMESPACE::Result                                   result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkGetVideoSessionMemoryRequirementsKHR( static_cast<VkDevice>( m_device ),
                                                                   static_cast<VkVideoSessionKHR>( m_videoSessionKHR ),
                                                                   &videoSessionMemoryRequirementsCount,
                                                                   nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && videoSessionMemoryRequirementsCount )
        {
          videoSessionMemoryRequirements.resize( videoSessionMemoryRequirementsCount );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetVideoSessionMemoryRequirementsKHR(
            static_cast<VkDevice>( m_device ),
            static_cast<VkVideoSessionKHR>( m_videoSessionKHR ),
            &videoSessionMemoryRequirementsCount,
            reinterpret_cast<VkVideoGetMemoryPropertiesKHR *>( videoSessionMemoryRequirements.data() ) ) );
          VULKAN_HPP_ASSERT( videoSessionMemoryRequirementsCount <= videoSessionMemoryRequirements.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
           ( videoSessionMemoryRequirementsCount < videoSessionMemoryRequirements.size() ) )
      {
        videoSessionMemoryRequirements.resize( videoSessionMemoryRequirementsCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::VideoSessionKHR::getMemoryRequirements" );
      }
      return videoSessionMemoryRequirements;
    }

    VULKAN_HPP_INLINE void VideoSessionKHR::bindMemory(
      ArrayProxy<const VULKAN_HPP_NAMESPACE::VideoBindMemoryKHR> const & videoSessionBindMemories ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkBindVideoSessionMemoryKHR &&
                         "Function <vkBindVideoSessionMemoryKHR> needs extension <VK_KHR_video_queue> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkBindVideoSessionMemoryKHR(
          static_cast<VkDevice>( m_device ),
          static_cast<VkVideoSessionKHR>( m_videoSessionKHR ),
          videoSessionBindMemories.size(),
          reinterpret_cast<const VkVideoBindMemoryKHR *>( videoSessionBindMemories.data() ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::VideoSessionKHR::bindMemory" );
      }
    }

    VULKAN_HPP_INLINE void
      VideoSessionParametersKHR::update( const VideoSessionParametersUpdateInfoKHR & updateInfo ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkUpdateVideoSessionParametersKHR &&
                         "Function <vkUpdateVideoSessionParametersKHR> needs extension <VK_KHR_video_queue> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkUpdateVideoSessionParametersKHR(
          static_cast<VkDevice>( m_device ),
          static_cast<VkVideoSessionParametersKHR>( m_videoSessionParametersKHR ),
          reinterpret_cast<const VkVideoSessionParametersUpdateInfoKHR *>( &updateInfo ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::VideoSessionParametersKHR::update" );
      }
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::beginVideoCodingKHR( const VideoBeginCodingInfoKHR & beginInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdBeginVideoCodingKHR &&
                         "Function <vkCmdBeginVideoCodingKHR> needs extension <VK_KHR_video_queue> enabled!" );

      getDispatcher()->vkCmdBeginVideoCodingKHR( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                 reinterpret_cast<const VkVideoBeginCodingInfoKHR *>( &beginInfo ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::endVideoCodingKHR( const VideoEndCodingInfoKHR & endCodingInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdEndVideoCodingKHR &&
                         "Function <vkCmdEndVideoCodingKHR> needs extension <VK_KHR_video_queue> enabled!" );

      getDispatcher()->vkCmdEndVideoCodingKHR( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                               reinterpret_cast<const VkVideoEndCodingInfoKHR *>( &endCodingInfo ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::controlVideoCodingKHR(
      const VideoCodingControlInfoKHR & codingControlInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdControlVideoCodingKHR &&
                         "Function <vkCmdControlVideoCodingKHR> needs extension <VK_KHR_video_queue> enabled!" );

      getDispatcher()->vkCmdControlVideoCodingKHR(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        reinterpret_cast<const VkVideoCodingControlInfoKHR *>( &codingControlInfo ) );
    }
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
    //=== VK_KHR_video_decode_queue ===

    VULKAN_HPP_INLINE void
      CommandBuffer::decodeVideoKHR( const VideoDecodeInfoKHR & frameInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdDecodeVideoKHR &&
                         "Function <vkCmdDecodeVideoKHR> needs extension <VK_KHR_video_decode_queue> enabled!" );

      getDispatcher()->vkCmdDecodeVideoKHR( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                            reinterpret_cast<const VkVideoDecodeInfoKHR *>( &frameInfo ) );
    }
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

    //=== VK_EXT_transform_feedback ===

    VULKAN_HPP_INLINE void CommandBuffer::bindTransformFeedbackBuffersEXT(
      uint32_t                                                   firstBinding,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::Buffer> const &     buffers,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::DeviceSize> const & offsets,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::DeviceSize> const & sizes ) const VULKAN_HPP_NOEXCEPT_WHEN_NO_EXCEPTIONS
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdBindTransformFeedbackBuffersEXT &&
        "Function <vkCmdBindTransformFeedbackBuffersEXT> needs extension <VK_EXT_transform_feedback> enabled!" );

#  ifdef VULKAN_HPP_NO_EXCEPTIONS
      VULKAN_HPP_ASSERT( buffers.size() == offsets.size() );
      VULKAN_HPP_ASSERT( sizes.empty() || buffers.size() == sizes.size() );
#  else
      if ( buffers.size() != offsets.size() )
      {
        throw LogicError( VULKAN_HPP_NAMESPACE_STRING
                          "::CommandBuffer::bindTransformFeedbackBuffersEXT: buffers.size() != offsets.size()" );
      }
      if ( !sizes.empty() && buffers.size() != sizes.size() )
      {
        throw LogicError( VULKAN_HPP_NAMESPACE_STRING
                          "::CommandBuffer::bindTransformFeedbackBuffersEXT: buffers.size() != sizes.size()" );
      }
#  endif /*VULKAN_HPP_NO_EXCEPTIONS*/

      getDispatcher()->vkCmdBindTransformFeedbackBuffersEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                             firstBinding,
                                                             buffers.size(),
                                                             reinterpret_cast<const VkBuffer *>( buffers.data() ),
                                                             reinterpret_cast<const VkDeviceSize *>( offsets.data() ),
                                                             reinterpret_cast<const VkDeviceSize *>( sizes.data() ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::beginTransformFeedbackEXT(
      uint32_t                                                   firstCounterBuffer,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::Buffer> const &     counterBuffers,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::DeviceSize> const & counterBufferOffsets ) const
      VULKAN_HPP_NOEXCEPT_WHEN_NO_EXCEPTIONS
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdBeginTransformFeedbackEXT &&
        "Function <vkCmdBeginTransformFeedbackEXT> needs extension <VK_EXT_transform_feedback> enabled!" );

#  ifdef VULKAN_HPP_NO_EXCEPTIONS
      VULKAN_HPP_ASSERT( counterBufferOffsets.empty() || counterBuffers.size() == counterBufferOffsets.size() );
#  else
      if ( !counterBufferOffsets.empty() && counterBuffers.size() != counterBufferOffsets.size() )
      {
        throw LogicError(
          VULKAN_HPP_NAMESPACE_STRING
          "::CommandBuffer::beginTransformFeedbackEXT: counterBuffers.size() != counterBufferOffsets.size()" );
      }
#  endif /*VULKAN_HPP_NO_EXCEPTIONS*/

      getDispatcher()->vkCmdBeginTransformFeedbackEXT(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        firstCounterBuffer,
        counterBuffers.size(),
        reinterpret_cast<const VkBuffer *>( counterBuffers.data() ),
        reinterpret_cast<const VkDeviceSize *>( counterBufferOffsets.data() ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::endTransformFeedbackEXT(
      uint32_t                                                   firstCounterBuffer,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::Buffer> const &     counterBuffers,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::DeviceSize> const & counterBufferOffsets ) const
      VULKAN_HPP_NOEXCEPT_WHEN_NO_EXCEPTIONS
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdEndTransformFeedbackEXT &&
        "Function <vkCmdEndTransformFeedbackEXT> needs extension <VK_EXT_transform_feedback> enabled!" );

#  ifdef VULKAN_HPP_NO_EXCEPTIONS
      VULKAN_HPP_ASSERT( counterBufferOffsets.empty() || counterBuffers.size() == counterBufferOffsets.size() );
#  else
      if ( !counterBufferOffsets.empty() && counterBuffers.size() != counterBufferOffsets.size() )
      {
        throw LogicError(
          VULKAN_HPP_NAMESPACE_STRING
          "::CommandBuffer::endTransformFeedbackEXT: counterBuffers.size() != counterBufferOffsets.size()" );
      }
#  endif /*VULKAN_HPP_NO_EXCEPTIONS*/

      getDispatcher()->vkCmdEndTransformFeedbackEXT(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        firstCounterBuffer,
        counterBuffers.size(),
        reinterpret_cast<const VkBuffer *>( counterBuffers.data() ),
        reinterpret_cast<const VkDeviceSize *>( counterBufferOffsets.data() ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::beginQueryIndexedEXT( VULKAN_HPP_NAMESPACE::QueryPool         queryPool,
                                                                uint32_t                                query,
                                                                VULKAN_HPP_NAMESPACE::QueryControlFlags flags,
                                                                uint32_t index ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdBeginQueryIndexedEXT &&
                         "Function <vkCmdBeginQueryIndexedEXT> needs extension <VK_EXT_transform_feedback> enabled!" );

      getDispatcher()->vkCmdBeginQueryIndexedEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                  static_cast<VkQueryPool>( queryPool ),
                                                  query,
                                                  static_cast<VkQueryControlFlags>( flags ),
                                                  index );
    }

    VULKAN_HPP_INLINE void CommandBuffer::endQueryIndexedEXT( VULKAN_HPP_NAMESPACE::QueryPool queryPool,
                                                              uint32_t                        query,
                                                              uint32_t index ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdEndQueryIndexedEXT &&
                         "Function <vkCmdEndQueryIndexedEXT> needs extension <VK_EXT_transform_feedback> enabled!" );

      getDispatcher()->vkCmdEndQueryIndexedEXT(
        static_cast<VkCommandBuffer>( m_commandBuffer ), static_cast<VkQueryPool>( queryPool ), query, index );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::drawIndirectByteCountEXT( uint32_t                         instanceCount,
                                               uint32_t                         firstInstance,
                                               VULKAN_HPP_NAMESPACE::Buffer     counterBuffer,
                                               VULKAN_HPP_NAMESPACE::DeviceSize counterBufferOffset,
                                               uint32_t                         counterOffset,
                                               uint32_t                         vertexStride ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdDrawIndirectByteCountEXT &&
        "Function <vkCmdDrawIndirectByteCountEXT> needs extension <VK_EXT_transform_feedback> enabled!" );

      getDispatcher()->vkCmdDrawIndirectByteCountEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                      instanceCount,
                                                      firstInstance,
                                                      static_cast<VkBuffer>( counterBuffer ),
                                                      static_cast<VkDeviceSize>( counterBufferOffset ),
                                                      counterOffset,
                                                      vertexStride );
    }

    //=== VK_NVX_binary_import ===

    VULKAN_HPP_INLINE void
      CommandBuffer::cuLaunchKernelNVX( const CuLaunchInfoNVX & launchInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdCuLaunchKernelNVX &&
                         "Function <vkCmdCuLaunchKernelNVX> needs extension <VK_NVX_binary_import> enabled!" );

      getDispatcher()->vkCmdCuLaunchKernelNVX( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                               reinterpret_cast<const VkCuLaunchInfoNVX *>( &launchInfo ) );
    }

    //=== VK_NVX_image_view_handle ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE uint32_t
                                           Device::getImageViewHandleNVX( const ImageViewHandleInfoNVX & info ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkGetImageViewHandleNVX &&
                         "Function <vkGetImageViewHandleNVX> needs extension <VK_NVX_image_view_handle> enabled!" );

      return getDispatcher()->vkGetImageViewHandleNVX( static_cast<VkDevice>( m_device ),
                                                       reinterpret_cast<const VkImageViewHandleInfoNVX *>( &info ) );
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::ImageViewAddressPropertiesNVX
                                           ImageView::getAddressNVX() const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkGetImageViewAddressNVX &&
                         "Function <vkGetImageViewAddressNVX> needs extension <VK_NVX_image_view_handle> enabled!" );

      VULKAN_HPP_NAMESPACE::ImageViewAddressPropertiesNVX properties;
      VULKAN_HPP_NAMESPACE::Result                        result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetImageViewAddressNVX(
          static_cast<VkDevice>( m_device ),
          static_cast<VkImageView>( m_imageView ),
          reinterpret_cast<VkImageViewAddressPropertiesNVX *>( &properties ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::ImageView::getAddressNVX" );
      }
      return properties;
    }

    //=== VK_AMD_draw_indirect_count ===

    VULKAN_HPP_INLINE void CommandBuffer::drawIndirectCountAMD( VULKAN_HPP_NAMESPACE::Buffer     buffer,
                                                                VULKAN_HPP_NAMESPACE::DeviceSize offset,
                                                                VULKAN_HPP_NAMESPACE::Buffer     countBuffer,
                                                                VULKAN_HPP_NAMESPACE::DeviceSize countBufferOffset,
                                                                uint32_t                         maxDrawCount,
                                                                uint32_t stride ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdDrawIndirectCountAMD &&
                         "Function <vkCmdDrawIndirectCountAMD> needs extension <VK_AMD_draw_indirect_count> enabled!" );

      getDispatcher()->vkCmdDrawIndirectCountAMD( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                  static_cast<VkBuffer>( buffer ),
                                                  static_cast<VkDeviceSize>( offset ),
                                                  static_cast<VkBuffer>( countBuffer ),
                                                  static_cast<VkDeviceSize>( countBufferOffset ),
                                                  maxDrawCount,
                                                  stride );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::drawIndexedIndirectCountAMD( VULKAN_HPP_NAMESPACE::Buffer     buffer,
                                                  VULKAN_HPP_NAMESPACE::DeviceSize offset,
                                                  VULKAN_HPP_NAMESPACE::Buffer     countBuffer,
                                                  VULKAN_HPP_NAMESPACE::DeviceSize countBufferOffset,
                                                  uint32_t                         maxDrawCount,
                                                  uint32_t                         stride ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdDrawIndexedIndirectCountAMD &&
        "Function <vkCmdDrawIndexedIndirectCountAMD> needs extension <VK_AMD_draw_indirect_count> enabled!" );

      getDispatcher()->vkCmdDrawIndexedIndirectCountAMD( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                         static_cast<VkBuffer>( buffer ),
                                                         static_cast<VkDeviceSize>( offset ),
                                                         static_cast<VkBuffer>( countBuffer ),
                                                         static_cast<VkDeviceSize>( countBufferOffset ),
                                                         maxDrawCount,
                                                         stride );
    }

    //=== VK_AMD_shader_info ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<uint8_t>
                                           Pipeline::getShaderInfoAMD( VULKAN_HPP_NAMESPACE::ShaderStageFlagBits shaderStage,
                                  VULKAN_HPP_NAMESPACE::ShaderInfoTypeAMD   infoType ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkGetShaderInfoAMD &&
                         "Function <vkGetShaderInfoAMD> needs extension <VK_AMD_shader_info> enabled!" );

      std::vector<uint8_t>         info;
      size_t                       infoSize;
      VULKAN_HPP_NAMESPACE::Result result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkGetShaderInfoAMD( static_cast<VkDevice>( m_device ),
                                               static_cast<VkPipeline>( m_pipeline ),
                                               static_cast<VkShaderStageFlagBits>( shaderStage ),
                                               static_cast<VkShaderInfoTypeAMD>( infoType ),
                                               &infoSize,
                                               nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && infoSize )
        {
          info.resize( infoSize );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
            getDispatcher()->vkGetShaderInfoAMD( static_cast<VkDevice>( m_device ),
                                                 static_cast<VkPipeline>( m_pipeline ),
                                                 static_cast<VkShaderStageFlagBits>( shaderStage ),
                                                 static_cast<VkShaderInfoTypeAMD>( infoType ),
                                                 &infoSize,
                                                 reinterpret_cast<void *>( info.data() ) ) );
          VULKAN_HPP_ASSERT( infoSize <= info.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( infoSize < info.size() ) )
      {
        info.resize( infoSize );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Pipeline::getShaderInfoAMD" );
      }
      return info;
    }

    //=== VK_NV_external_memory_capabilities ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::ExternalImageFormatPropertiesNV
                                           PhysicalDevice::getExternalImageFormatPropertiesNV(
        VULKAN_HPP_NAMESPACE::Format                          format,
        VULKAN_HPP_NAMESPACE::ImageType                       type,
        VULKAN_HPP_NAMESPACE::ImageTiling                     tiling,
        VULKAN_HPP_NAMESPACE::ImageUsageFlags                 usage,
        VULKAN_HPP_NAMESPACE::ImageCreateFlags                flags,
        VULKAN_HPP_NAMESPACE::ExternalMemoryHandleTypeFlagsNV externalHandleType ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceExternalImageFormatPropertiesNV &&
        "Function <vkGetPhysicalDeviceExternalImageFormatPropertiesNV> needs extension <VK_NV_external_memory_capabilities> enabled!" );

      VULKAN_HPP_NAMESPACE::ExternalImageFormatPropertiesNV externalImageFormatProperties;
      VULKAN_HPP_NAMESPACE::Result                          result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceExternalImageFormatPropertiesNV(
          static_cast<VkPhysicalDevice>( m_physicalDevice ),
          static_cast<VkFormat>( format ),
          static_cast<VkImageType>( type ),
          static_cast<VkImageTiling>( tiling ),
          static_cast<VkImageUsageFlags>( usage ),
          static_cast<VkImageCreateFlags>( flags ),
          static_cast<VkExternalMemoryHandleTypeFlagsNV>( externalHandleType ),
          reinterpret_cast<VkExternalImageFormatPropertiesNV *>( &externalImageFormatProperties ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result,
                              VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getExternalImageFormatPropertiesNV" );
      }
      return externalImageFormatProperties;
    }

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_NV_external_memory_win32 ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE HANDLE
                                           DeviceMemory::getMemoryWin32HandleNV( VULKAN_HPP_NAMESPACE::ExternalMemoryHandleTypeFlagsNV handleType ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkGetMemoryWin32HandleNV &&
                         "Function <vkGetMemoryWin32HandleNV> needs extension <VK_NV_external_memory_win32> enabled!" );

      HANDLE                       handle;
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkGetMemoryWin32HandleNV( static_cast<VkDevice>( m_device ),
                                                   static_cast<VkDeviceMemory>( m_deviceMemory ),
                                                   static_cast<VkExternalMemoryHandleTypeFlagsNV>( handleType ),
                                                   &handle ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::DeviceMemory::getMemoryWin32HandleNV" );
      }
      return handle;
    }
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

    //=== VK_KHR_get_physical_device_properties2 ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures2
                                           PhysicalDevice::getFeatures2KHR() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceFeatures2KHR &&
        "Function <vkGetPhysicalDeviceFeatures2KHR> needs extension <VK_KHR_get_physical_device_properties2> enabled!" );

      VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures2 features;
      getDispatcher()->vkGetPhysicalDeviceFeatures2KHR( static_cast<VkPhysicalDevice>( m_physicalDevice ),
                                                        reinterpret_cast<VkPhysicalDeviceFeatures2 *>( &features ) );
      return features;
    }

    template <typename X, typename Y, typename... Z>
    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE StructureChain<X, Y, Z...>
                                           PhysicalDevice::getFeatures2KHR() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceFeatures2KHR &&
        "Function <vkGetPhysicalDeviceFeatures2KHR> needs extension <VK_KHR_get_physical_device_properties2> enabled!" );

      StructureChain<X, Y, Z...>                      structureChain;
      VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures2 & features =
        structureChain.template get<VULKAN_HPP_NAMESPACE::PhysicalDeviceFeatures2>();
      getDispatcher()->vkGetPhysicalDeviceFeatures2KHR( static_cast<VkPhysicalDevice>( m_physicalDevice ),
                                                        reinterpret_cast<VkPhysicalDeviceFeatures2 *>( &features ) );
      return structureChain;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties2
                                           PhysicalDevice::getProperties2KHR() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceProperties2KHR &&
        "Function <vkGetPhysicalDeviceProperties2KHR> needs extension <VK_KHR_get_physical_device_properties2> enabled!" );

      VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties2 properties;
      getDispatcher()->vkGetPhysicalDeviceProperties2KHR(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        reinterpret_cast<VkPhysicalDeviceProperties2 *>( &properties ) );
      return properties;
    }

    template <typename X, typename Y, typename... Z>
    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE StructureChain<X, Y, Z...>
                                           PhysicalDevice::getProperties2KHR() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceProperties2KHR &&
        "Function <vkGetPhysicalDeviceProperties2KHR> needs extension <VK_KHR_get_physical_device_properties2> enabled!" );

      StructureChain<X, Y, Z...>                        structureChain;
      VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties2 & properties =
        structureChain.template get<VULKAN_HPP_NAMESPACE::PhysicalDeviceProperties2>();
      getDispatcher()->vkGetPhysicalDeviceProperties2KHR(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        reinterpret_cast<VkPhysicalDeviceProperties2 *>( &properties ) );
      return structureChain;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::FormatProperties2
                                           PhysicalDevice::getFormatProperties2KHR( VULKAN_HPP_NAMESPACE::Format format ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceFormatProperties2KHR &&
        "Function <vkGetPhysicalDeviceFormatProperties2KHR> needs extension <VK_KHR_get_physical_device_properties2> enabled!" );

      VULKAN_HPP_NAMESPACE::FormatProperties2 formatProperties;
      getDispatcher()->vkGetPhysicalDeviceFormatProperties2KHR(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        static_cast<VkFormat>( format ),
        reinterpret_cast<VkFormatProperties2 *>( &formatProperties ) );
      return formatProperties;
    }

    template <typename X, typename Y, typename... Z>
    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE StructureChain<X, Y, Z...>
                                           PhysicalDevice::getFormatProperties2KHR( VULKAN_HPP_NAMESPACE::Format format ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceFormatProperties2KHR &&
        "Function <vkGetPhysicalDeviceFormatProperties2KHR> needs extension <VK_KHR_get_physical_device_properties2> enabled!" );

      StructureChain<X, Y, Z...>                structureChain;
      VULKAN_HPP_NAMESPACE::FormatProperties2 & formatProperties =
        structureChain.template get<VULKAN_HPP_NAMESPACE::FormatProperties2>();
      getDispatcher()->vkGetPhysicalDeviceFormatProperties2KHR(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        static_cast<VkFormat>( format ),
        reinterpret_cast<VkFormatProperties2 *>( &formatProperties ) );
      return structureChain;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::ImageFormatProperties2
                                           PhysicalDevice::getImageFormatProperties2KHR( const PhysicalDeviceImageFormatInfo2 & imageFormatInfo ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceImageFormatProperties2KHR &&
        "Function <vkGetPhysicalDeviceImageFormatProperties2KHR> needs extension <VK_KHR_get_physical_device_properties2> enabled!" );

      VULKAN_HPP_NAMESPACE::ImageFormatProperties2 imageFormatProperties;
      VULKAN_HPP_NAMESPACE::Result                 result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceImageFormatProperties2KHR(
          static_cast<VkPhysicalDevice>( m_physicalDevice ),
          reinterpret_cast<const VkPhysicalDeviceImageFormatInfo2 *>( &imageFormatInfo ),
          reinterpret_cast<VkImageFormatProperties2 *>( &imageFormatProperties ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getImageFormatProperties2KHR" );
      }
      return imageFormatProperties;
    }

    template <typename X, typename Y, typename... Z>
    VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...>
                         PhysicalDevice::getImageFormatProperties2KHR( const PhysicalDeviceImageFormatInfo2 & imageFormatInfo ) const
    {
      StructureChain<X, Y, Z...>                     structureChain;
      VULKAN_HPP_NAMESPACE::ImageFormatProperties2 & imageFormatProperties =
        structureChain.template get<VULKAN_HPP_NAMESPACE::ImageFormatProperties2>();
      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceImageFormatProperties2KHR(
          static_cast<VkPhysicalDevice>( m_physicalDevice ),
          reinterpret_cast<const VkPhysicalDeviceImageFormatInfo2 *>( &imageFormatInfo ),
          reinterpret_cast<VkImageFormatProperties2 *>( &imageFormatProperties ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getImageFormatProperties2KHR" );
      }
      return structureChain;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::QueueFamilyProperties2>
                                           PhysicalDevice::getQueueFamilyProperties2KHR() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceQueueFamilyProperties2KHR &&
        "Function <vkGetPhysicalDeviceQueueFamilyProperties2KHR> needs extension <VK_KHR_get_physical_device_properties2> enabled!" );

      uint32_t queueFamilyPropertyCount;
      getDispatcher()->vkGetPhysicalDeviceQueueFamilyProperties2KHR(
        static_cast<VkPhysicalDevice>( m_physicalDevice ), &queueFamilyPropertyCount, nullptr );
      std::vector<VULKAN_HPP_NAMESPACE::QueueFamilyProperties2> queueFamilyProperties( queueFamilyPropertyCount );
      getDispatcher()->vkGetPhysicalDeviceQueueFamilyProperties2KHR(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        &queueFamilyPropertyCount,
        reinterpret_cast<VkQueueFamilyProperties2 *>( queueFamilyProperties.data() ) );
      VULKAN_HPP_ASSERT( queueFamilyPropertyCount <= queueFamilyProperties.size() );
      return queueFamilyProperties;
    }

    template <typename StructureChain>
    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<StructureChain>
                                           PhysicalDevice::getQueueFamilyProperties2KHR() const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceQueueFamilyProperties2KHR &&
        "Function <vkGetPhysicalDeviceQueueFamilyProperties2KHR> needs extension <VK_KHR_get_physical_device_properties2> enabled!" );

      uint32_t queueFamilyPropertyCount;
      getDispatcher()->vkGetPhysicalDeviceQueueFamilyProperties2KHR(
        static_cast<VkPhysicalDevice>( m_physicalDevice ), &queueFamilyPropertyCount, nullptr );
      std::vector<StructureChain>                               returnVector( queueFamilyPropertyCount );
      std::vector<VULKAN_HPP_NAMESPACE::QueueFamilyProperties2> queueFamilyProperties( queueFamilyPropertyCount );
      for ( uint32_t i = 0; i < queueFamilyPropertyCount; i++ )
      {
        queueFamilyProperties[i].pNext =
          returnVector[i].template get<VULKAN_HPP_NAMESPACE::QueueFamilyProperties2>().pNext;
      }
      getDispatcher()->vkGetPhysicalDeviceQueueFamilyProperties2KHR(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        &queueFamilyPropertyCount,
        reinterpret_cast<VkQueueFamilyProperties2 *>( queueFamilyProperties.data() ) );
      VULKAN_HPP_ASSERT( queueFamilyPropertyCount <= queueFamilyProperties.size() );
      for ( uint32_t i = 0; i < queueFamilyPropertyCount; i++ )
      {
        returnVector[i].template get<VULKAN_HPP_NAMESPACE::QueueFamilyProperties2>() = queueFamilyProperties[i];
      }
      return returnVector;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties2
                                           PhysicalDevice::getMemoryProperties2KHR() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceMemoryProperties2KHR &&
        "Function <vkGetPhysicalDeviceMemoryProperties2KHR> needs extension <VK_KHR_get_physical_device_properties2> enabled!" );

      VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties2 memoryProperties;
      getDispatcher()->vkGetPhysicalDeviceMemoryProperties2KHR(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        reinterpret_cast<VkPhysicalDeviceMemoryProperties2 *>( &memoryProperties ) );
      return memoryProperties;
    }

    template <typename X, typename Y, typename... Z>
    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE StructureChain<X, Y, Z...>
                                           PhysicalDevice::getMemoryProperties2KHR() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceMemoryProperties2KHR &&
        "Function <vkGetPhysicalDeviceMemoryProperties2KHR> needs extension <VK_KHR_get_physical_device_properties2> enabled!" );

      StructureChain<X, Y, Z...>                              structureChain;
      VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties2 & memoryProperties =
        structureChain.template get<VULKAN_HPP_NAMESPACE::PhysicalDeviceMemoryProperties2>();
      getDispatcher()->vkGetPhysicalDeviceMemoryProperties2KHR(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        reinterpret_cast<VkPhysicalDeviceMemoryProperties2 *>( &memoryProperties ) );
      return structureChain;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::SparseImageFormatProperties2>
                                           PhysicalDevice::getSparseImageFormatProperties2KHR(
        const PhysicalDeviceSparseImageFormatInfo2 & formatInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceSparseImageFormatProperties2KHR &&
        "Function <vkGetPhysicalDeviceSparseImageFormatProperties2KHR> needs extension <VK_KHR_get_physical_device_properties2> enabled!" );

      uint32_t propertyCount;
      getDispatcher()->vkGetPhysicalDeviceSparseImageFormatProperties2KHR(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        reinterpret_cast<const VkPhysicalDeviceSparseImageFormatInfo2 *>( &formatInfo ),
        &propertyCount,
        nullptr );
      std::vector<VULKAN_HPP_NAMESPACE::SparseImageFormatProperties2> properties( propertyCount );
      getDispatcher()->vkGetPhysicalDeviceSparseImageFormatProperties2KHR(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        reinterpret_cast<const VkPhysicalDeviceSparseImageFormatInfo2 *>( &formatInfo ),
        &propertyCount,
        reinterpret_cast<VkSparseImageFormatProperties2 *>( properties.data() ) );
      VULKAN_HPP_ASSERT( propertyCount <= properties.size() );
      return properties;
    }

    //=== VK_KHR_device_group ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::PeerMemoryFeatureFlags
                                           Device::getGroupPeerMemoryFeaturesKHR( uint32_t heapIndex,
                                             uint32_t localDeviceIndex,
                                             uint32_t remoteDeviceIndex ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetDeviceGroupPeerMemoryFeaturesKHR &&
        "Function <vkGetDeviceGroupPeerMemoryFeaturesKHR> needs extension <VK_KHR_device_group> enabled!" );

      VULKAN_HPP_NAMESPACE::PeerMemoryFeatureFlags peerMemoryFeatures;
      getDispatcher()->vkGetDeviceGroupPeerMemoryFeaturesKHR(
        static_cast<VkDevice>( m_device ),
        heapIndex,
        localDeviceIndex,
        remoteDeviceIndex,
        reinterpret_cast<VkPeerMemoryFeatureFlags *>( &peerMemoryFeatures ) );
      return peerMemoryFeatures;
    }

    VULKAN_HPP_INLINE void CommandBuffer::setDeviceMaskKHR( uint32_t deviceMask ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdSetDeviceMaskKHR &&
                         "Function <vkCmdSetDeviceMaskKHR> needs extension <VK_KHR_device_group> enabled!" );

      getDispatcher()->vkCmdSetDeviceMaskKHR( static_cast<VkCommandBuffer>( m_commandBuffer ), deviceMask );
    }

    VULKAN_HPP_INLINE void CommandBuffer::dispatchBaseKHR( uint32_t baseGroupX,
                                                           uint32_t baseGroupY,
                                                           uint32_t baseGroupZ,
                                                           uint32_t groupCountX,
                                                           uint32_t groupCountY,
                                                           uint32_t groupCountZ ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdDispatchBaseKHR &&
                         "Function <vkCmdDispatchBaseKHR> needs extension <VK_KHR_device_group> enabled!" );

      getDispatcher()->vkCmdDispatchBaseKHR( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                             baseGroupX,
                                             baseGroupY,
                                             baseGroupZ,
                                             groupCountX,
                                             groupCountY,
                                             groupCountZ );
    }

    //=== VK_KHR_maintenance1 ===

    VULKAN_HPP_INLINE void
      CommandPool::trimKHR( VULKAN_HPP_NAMESPACE::CommandPoolTrimFlags flags ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkTrimCommandPoolKHR &&
                         "Function <vkTrimCommandPoolKHR> needs extension <VK_KHR_maintenance1> enabled!" );

      getDispatcher()->vkTrimCommandPoolKHR( static_cast<VkDevice>( m_device ),
                                             static_cast<VkCommandPool>( m_commandPool ),
                                             static_cast<VkCommandPoolTrimFlags>( flags ) );
    }

    //=== VK_KHR_device_group_creation ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::PhysicalDeviceGroupProperties>
                                           Instance::enumeratePhysicalDeviceGroupsKHR() const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkEnumeratePhysicalDeviceGroupsKHR &&
        "Function <vkEnumeratePhysicalDeviceGroupsKHR> needs extension <VK_KHR_device_group_creation> enabled!" );

      std::vector<VULKAN_HPP_NAMESPACE::PhysicalDeviceGroupProperties> physicalDeviceGroupProperties;
      uint32_t                                                         physicalDeviceGroupCount;
      VULKAN_HPP_NAMESPACE::Result                                     result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkEnumeratePhysicalDeviceGroupsKHR(
          static_cast<VkInstance>( m_instance ), &physicalDeviceGroupCount, nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && physicalDeviceGroupCount )
        {
          physicalDeviceGroupProperties.resize( physicalDeviceGroupCount );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkEnumeratePhysicalDeviceGroupsKHR(
            static_cast<VkInstance>( m_instance ),
            &physicalDeviceGroupCount,
            reinterpret_cast<VkPhysicalDeviceGroupProperties *>( physicalDeviceGroupProperties.data() ) ) );
          VULKAN_HPP_ASSERT( physicalDeviceGroupCount <= physicalDeviceGroupProperties.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
           ( physicalDeviceGroupCount < physicalDeviceGroupProperties.size() ) )
      {
        physicalDeviceGroupProperties.resize( physicalDeviceGroupCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Instance::enumeratePhysicalDeviceGroupsKHR" );
      }
      return physicalDeviceGroupProperties;
    }

    //=== VK_KHR_external_memory_capabilities ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::ExternalBufferProperties
                                           PhysicalDevice::getExternalBufferPropertiesKHR(
        const PhysicalDeviceExternalBufferInfo & externalBufferInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceExternalBufferPropertiesKHR &&
        "Function <vkGetPhysicalDeviceExternalBufferPropertiesKHR> needs extension <VK_KHR_external_memory_capabilities> enabled!" );

      VULKAN_HPP_NAMESPACE::ExternalBufferProperties externalBufferProperties;
      getDispatcher()->vkGetPhysicalDeviceExternalBufferPropertiesKHR(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        reinterpret_cast<const VkPhysicalDeviceExternalBufferInfo *>( &externalBufferInfo ),
        reinterpret_cast<VkExternalBufferProperties *>( &externalBufferProperties ) );
      return externalBufferProperties;
    }

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_KHR_external_memory_win32 ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE HANDLE
                                           Device::getMemoryWin32HandleKHR( const MemoryGetWin32HandleInfoKHR & getWin32HandleInfo ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetMemoryWin32HandleKHR &&
        "Function <vkGetMemoryWin32HandleKHR> needs extension <VK_KHR_external_memory_win32> enabled!" );

      HANDLE                       handle;
      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetMemoryWin32HandleKHR(
          static_cast<VkDevice>( m_device ),
          reinterpret_cast<const VkMemoryGetWin32HandleInfoKHR *>( &getWin32HandleInfo ),
          &handle ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::getMemoryWin32HandleKHR" );
      }
      return handle;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::MemoryWin32HandlePropertiesKHR
                                           Device::getMemoryWin32HandlePropertiesKHR( VULKAN_HPP_NAMESPACE::ExternalMemoryHandleTypeFlagBits handleType,
                                                 HANDLE                                                 handle ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetMemoryWin32HandlePropertiesKHR &&
        "Function <vkGetMemoryWin32HandlePropertiesKHR> needs extension <VK_KHR_external_memory_win32> enabled!" );

      VULKAN_HPP_NAMESPACE::MemoryWin32HandlePropertiesKHR memoryWin32HandleProperties;
      VULKAN_HPP_NAMESPACE::Result                         result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetMemoryWin32HandlePropertiesKHR(
          static_cast<VkDevice>( m_device ),
          static_cast<VkExternalMemoryHandleTypeFlagBits>( handleType ),
          handle,
          reinterpret_cast<VkMemoryWin32HandlePropertiesKHR *>( &memoryWin32HandleProperties ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::getMemoryWin32HandlePropertiesKHR" );
      }
      return memoryWin32HandleProperties;
    }
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

    //=== VK_KHR_external_memory_fd ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE int Device::getMemoryFdKHR( const MemoryGetFdInfoKHR & getFdInfo ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkGetMemoryFdKHR &&
                         "Function <vkGetMemoryFdKHR> needs extension <VK_KHR_external_memory_fd> enabled!" );

      int                          fd;
      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetMemoryFdKHR(
          static_cast<VkDevice>( m_device ), reinterpret_cast<const VkMemoryGetFdInfoKHR *>( &getFdInfo ), &fd ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::getMemoryFdKHR" );
      }
      return fd;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::MemoryFdPropertiesKHR
                                           Device::getMemoryFdPropertiesKHR( VULKAN_HPP_NAMESPACE::ExternalMemoryHandleTypeFlagBits handleType,
                                        int                                                    fd ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkGetMemoryFdPropertiesKHR &&
                         "Function <vkGetMemoryFdPropertiesKHR> needs extension <VK_KHR_external_memory_fd> enabled!" );

      VULKAN_HPP_NAMESPACE::MemoryFdPropertiesKHR memoryFdProperties;
      VULKAN_HPP_NAMESPACE::Result                result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetMemoryFdPropertiesKHR(
          static_cast<VkDevice>( m_device ),
          static_cast<VkExternalMemoryHandleTypeFlagBits>( handleType ),
          fd,
          reinterpret_cast<VkMemoryFdPropertiesKHR *>( &memoryFdProperties ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::getMemoryFdPropertiesKHR" );
      }
      return memoryFdProperties;
    }

    //=== VK_KHR_external_semaphore_capabilities ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::ExternalSemaphoreProperties
                                           PhysicalDevice::getExternalSemaphorePropertiesKHR(
        const PhysicalDeviceExternalSemaphoreInfo & externalSemaphoreInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceExternalSemaphorePropertiesKHR &&
        "Function <vkGetPhysicalDeviceExternalSemaphorePropertiesKHR> needs extension <VK_KHR_external_semaphore_capabilities> enabled!" );

      VULKAN_HPP_NAMESPACE::ExternalSemaphoreProperties externalSemaphoreProperties;
      getDispatcher()->vkGetPhysicalDeviceExternalSemaphorePropertiesKHR(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        reinterpret_cast<const VkPhysicalDeviceExternalSemaphoreInfo *>( &externalSemaphoreInfo ),
        reinterpret_cast<VkExternalSemaphoreProperties *>( &externalSemaphoreProperties ) );
      return externalSemaphoreProperties;
    }

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_KHR_external_semaphore_win32 ===

    VULKAN_HPP_INLINE void Device::importSemaphoreWin32HandleKHR(
      const ImportSemaphoreWin32HandleInfoKHR & importSemaphoreWin32HandleInfo ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkImportSemaphoreWin32HandleKHR &&
        "Function <vkImportSemaphoreWin32HandleKHR> needs extension <VK_KHR_external_semaphore_win32> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkImportSemaphoreWin32HandleKHR(
          static_cast<VkDevice>( m_device ),
          reinterpret_cast<const VkImportSemaphoreWin32HandleInfoKHR *>( &importSemaphoreWin32HandleInfo ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::importSemaphoreWin32HandleKHR" );
      }
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE HANDLE
                                           Device::getSemaphoreWin32HandleKHR( const SemaphoreGetWin32HandleInfoKHR & getWin32HandleInfo ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetSemaphoreWin32HandleKHR &&
        "Function <vkGetSemaphoreWin32HandleKHR> needs extension <VK_KHR_external_semaphore_win32> enabled!" );

      HANDLE                       handle;
      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetSemaphoreWin32HandleKHR(
          static_cast<VkDevice>( m_device ),
          reinterpret_cast<const VkSemaphoreGetWin32HandleInfoKHR *>( &getWin32HandleInfo ),
          &handle ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::getSemaphoreWin32HandleKHR" );
      }
      return handle;
    }
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

    //=== VK_KHR_external_semaphore_fd ===

    VULKAN_HPP_INLINE void Device::importSemaphoreFdKHR( const ImportSemaphoreFdInfoKHR & importSemaphoreFdInfo ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkImportSemaphoreFdKHR &&
                         "Function <vkImportSemaphoreFdKHR> needs extension <VK_KHR_external_semaphore_fd> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkImportSemaphoreFdKHR(
          static_cast<VkDevice>( m_device ),
          reinterpret_cast<const VkImportSemaphoreFdInfoKHR *>( &importSemaphoreFdInfo ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::importSemaphoreFdKHR" );
      }
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE int
                         Device::getSemaphoreFdKHR( const SemaphoreGetFdInfoKHR & getFdInfo ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkGetSemaphoreFdKHR &&
                         "Function <vkGetSemaphoreFdKHR> needs extension <VK_KHR_external_semaphore_fd> enabled!" );

      int                          fd;
      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetSemaphoreFdKHR(
          static_cast<VkDevice>( m_device ), reinterpret_cast<const VkSemaphoreGetFdInfoKHR *>( &getFdInfo ), &fd ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::getSemaphoreFdKHR" );
      }
      return fd;
    }

    //=== VK_KHR_push_descriptor ===

    VULKAN_HPP_INLINE void CommandBuffer::pushDescriptorSetKHR(
      VULKAN_HPP_NAMESPACE::PipelineBindPoint                            pipelineBindPoint,
      VULKAN_HPP_NAMESPACE::PipelineLayout                               layout,
      uint32_t                                                           set,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::WriteDescriptorSet> const & descriptorWrites ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdPushDescriptorSetKHR &&
                         "Function <vkCmdPushDescriptorSetKHR> needs extension <VK_KHR_push_descriptor> enabled!" );

      getDispatcher()->vkCmdPushDescriptorSetKHR(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        static_cast<VkPipelineBindPoint>( pipelineBindPoint ),
        static_cast<VkPipelineLayout>( layout ),
        set,
        descriptorWrites.size(),
        reinterpret_cast<const VkWriteDescriptorSet *>( descriptorWrites.data() ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::pushDescriptorSetWithTemplateKHR(
      VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplate descriptorUpdateTemplate,
      VULKAN_HPP_NAMESPACE::PipelineLayout           layout,
      uint32_t                                       set,
      const void *                                   pData ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdPushDescriptorSetWithTemplateKHR &&
        "Function <vkCmdPushDescriptorSetWithTemplateKHR> needs extension <VK_KHR_push_descriptor> enabled!" );

      getDispatcher()->vkCmdPushDescriptorSetWithTemplateKHR(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        static_cast<VkDescriptorUpdateTemplate>( descriptorUpdateTemplate ),
        static_cast<VkPipelineLayout>( layout ),
        set,
        pData );
    }

    //=== VK_EXT_conditional_rendering ===

    VULKAN_HPP_INLINE void CommandBuffer::beginConditionalRenderingEXT(
      const ConditionalRenderingBeginInfoEXT & conditionalRenderingBegin ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdBeginConditionalRenderingEXT &&
        "Function <vkCmdBeginConditionalRenderingEXT> needs extension <VK_EXT_conditional_rendering> enabled!" );

      getDispatcher()->vkCmdBeginConditionalRenderingEXT(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        reinterpret_cast<const VkConditionalRenderingBeginInfoEXT *>( &conditionalRenderingBegin ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::endConditionalRenderingEXT() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdEndConditionalRenderingEXT &&
        "Function <vkCmdEndConditionalRenderingEXT> needs extension <VK_EXT_conditional_rendering> enabled!" );

      getDispatcher()->vkCmdEndConditionalRenderingEXT( static_cast<VkCommandBuffer>( m_commandBuffer ) );
    }

    //=== VK_KHR_descriptor_update_template ===

    VULKAN_HPP_INLINE void Device::destroyDescriptorUpdateTemplateKHR(
      VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplate descriptorUpdateTemplate,
      Optional<const AllocationCallbacks>            allocator ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkDestroyDescriptorUpdateTemplateKHR &&
        "Function <vkDestroyDescriptorUpdateTemplateKHR> needs extension <VK_KHR_descriptor_update_template> enabled!" );

      getDispatcher()->vkDestroyDescriptorUpdateTemplateKHR(
        static_cast<VkDevice>( m_device ),
        static_cast<VkDescriptorUpdateTemplate>( descriptorUpdateTemplate ),
        reinterpret_cast<const VkAllocationCallbacks *>(
          static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) );
    }

    VULKAN_HPP_INLINE void
      DescriptorSet::updateWithTemplateKHR( VULKAN_HPP_NAMESPACE::DescriptorUpdateTemplate descriptorUpdateTemplate,
                                            const void * pData ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkUpdateDescriptorSetWithTemplateKHR &&
        "Function <vkUpdateDescriptorSetWithTemplateKHR> needs extension <VK_KHR_descriptor_update_template> enabled!" );

      getDispatcher()->vkUpdateDescriptorSetWithTemplateKHR(
        static_cast<VkDevice>( m_device ),
        static_cast<VkDescriptorSet>( m_descriptorSet ),
        static_cast<VkDescriptorUpdateTemplate>( descriptorUpdateTemplate ),
        pData );
    }

    //=== VK_NV_clip_space_w_scaling ===

    VULKAN_HPP_INLINE void CommandBuffer::setViewportWScalingNV(
      uint32_t                                                           firstViewport,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::ViewportWScalingNV> const & viewportWScalings ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdSetViewportWScalingNV &&
        "Function <vkCmdSetViewportWScalingNV> needs extension <VK_NV_clip_space_w_scaling> enabled!" );

      getDispatcher()->vkCmdSetViewportWScalingNV(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        firstViewport,
        viewportWScalings.size(),
        reinterpret_cast<const VkViewportWScalingNV *>( viewportWScalings.data() ) );
    }

#  if defined( VK_USE_PLATFORM_XLIB_XRANDR_EXT )
    //=== VK_EXT_acquire_xlib_display ===

    VULKAN_HPP_INLINE void PhysicalDevice::acquireXlibDisplayEXT( Display &                        dpy,
                                                                  VULKAN_HPP_NAMESPACE::DisplayKHR display ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkAcquireXlibDisplayEXT &&
                         "Function <vkAcquireXlibDisplayEXT> needs extension <VK_EXT_acquire_xlib_display> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkAcquireXlibDisplayEXT(
          static_cast<VkPhysicalDevice>( m_physicalDevice ), &dpy, static_cast<VkDisplayKHR>( display ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::acquireXlibDisplayEXT" );
      }
    }
#  endif /*VK_USE_PLATFORM_XLIB_XRANDR_EXT*/

    //=== VK_EXT_display_surface_counter ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::SurfaceCapabilities2EXT
                                           PhysicalDevice::getSurfaceCapabilities2EXT( VULKAN_HPP_NAMESPACE::SurfaceKHR surface ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceSurfaceCapabilities2EXT &&
        "Function <vkGetPhysicalDeviceSurfaceCapabilities2EXT> needs extension <VK_EXT_display_surface_counter> enabled!" );

      VULKAN_HPP_NAMESPACE::SurfaceCapabilities2EXT surfaceCapabilities;
      VULKAN_HPP_NAMESPACE::Result                  result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceSurfaceCapabilities2EXT(
          static_cast<VkPhysicalDevice>( m_physicalDevice ),
          static_cast<VkSurfaceKHR>( surface ),
          reinterpret_cast<VkSurfaceCapabilities2EXT *>( &surfaceCapabilities ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getSurfaceCapabilities2EXT" );
      }
      return surfaceCapabilities;
    }

    //=== VK_EXT_display_control ===

    VULKAN_HPP_INLINE void Device::displayPowerControlEXT( VULKAN_HPP_NAMESPACE::DisplayKHR display,
                                                           const DisplayPowerInfoEXT &      displayPowerInfo ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkDisplayPowerControlEXT &&
                         "Function <vkDisplayPowerControlEXT> needs extension <VK_EXT_display_control> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkDisplayPowerControlEXT(
          static_cast<VkDevice>( m_device ),
          static_cast<VkDisplayKHR>( display ),
          reinterpret_cast<const VkDisplayPowerInfoEXT *>( &displayPowerInfo ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::displayPowerControlEXT" );
      }
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE uint64_t
                                           SwapchainKHR::getCounterEXT( VULKAN_HPP_NAMESPACE::SurfaceCounterFlagBitsEXT counter ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkGetSwapchainCounterEXT &&
                         "Function <vkGetSwapchainCounterEXT> needs extension <VK_EXT_display_control> enabled!" );

      uint64_t                     counterValue;
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkGetSwapchainCounterEXT( static_cast<VkDevice>( m_device ),
                                                   static_cast<VkSwapchainKHR>( m_swapchainKHR ),
                                                   static_cast<VkSurfaceCounterFlagBitsEXT>( counter ),
                                                   &counterValue ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::SwapchainKHR::getCounterEXT" );
      }
      return counterValue;
    }

    //=== VK_GOOGLE_display_timing ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::RefreshCycleDurationGOOGLE
                                           SwapchainKHR::getRefreshCycleDurationGOOGLE() const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetRefreshCycleDurationGOOGLE &&
        "Function <vkGetRefreshCycleDurationGOOGLE> needs extension <VK_GOOGLE_display_timing> enabled!" );

      VULKAN_HPP_NAMESPACE::RefreshCycleDurationGOOGLE displayTimingProperties;
      VULKAN_HPP_NAMESPACE::Result                     result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetRefreshCycleDurationGOOGLE(
          static_cast<VkDevice>( m_device ),
          static_cast<VkSwapchainKHR>( m_swapchainKHR ),
          reinterpret_cast<VkRefreshCycleDurationGOOGLE *>( &displayTimingProperties ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::SwapchainKHR::getRefreshCycleDurationGOOGLE" );
      }
      return displayTimingProperties;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::PastPresentationTimingGOOGLE>
                                           SwapchainKHR::getPastPresentationTimingGOOGLE() const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPastPresentationTimingGOOGLE &&
        "Function <vkGetPastPresentationTimingGOOGLE> needs extension <VK_GOOGLE_display_timing> enabled!" );

      std::vector<VULKAN_HPP_NAMESPACE::PastPresentationTimingGOOGLE> presentationTimings;
      uint32_t                                                        presentationTimingCount;
      VULKAN_HPP_NAMESPACE::Result                                    result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkGetPastPresentationTimingGOOGLE( static_cast<VkDevice>( m_device ),
                                                              static_cast<VkSwapchainKHR>( m_swapchainKHR ),
                                                              &presentationTimingCount,
                                                              nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && presentationTimingCount )
        {
          presentationTimings.resize( presentationTimingCount );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPastPresentationTimingGOOGLE(
            static_cast<VkDevice>( m_device ),
            static_cast<VkSwapchainKHR>( m_swapchainKHR ),
            &presentationTimingCount,
            reinterpret_cast<VkPastPresentationTimingGOOGLE *>( presentationTimings.data() ) ) );
          VULKAN_HPP_ASSERT( presentationTimingCount <= presentationTimings.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
           ( presentationTimingCount < presentationTimings.size() ) )
      {
        presentationTimings.resize( presentationTimingCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::SwapchainKHR::getPastPresentationTimingGOOGLE" );
      }
      return presentationTimings;
    }

    //=== VK_EXT_discard_rectangles ===

    VULKAN_HPP_INLINE void CommandBuffer::setDiscardRectangleEXT(
      uint32_t                                               firstDiscardRectangle,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::Rect2D> const & discardRectangles ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdSetDiscardRectangleEXT &&
        "Function <vkCmdSetDiscardRectangleEXT> needs extension <VK_EXT_discard_rectangles> enabled!" );

      getDispatcher()->vkCmdSetDiscardRectangleEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                    firstDiscardRectangle,
                                                    discardRectangles.size(),
                                                    reinterpret_cast<const VkRect2D *>( discardRectangles.data() ) );
    }

    //=== VK_EXT_hdr_metadata ===

    VULKAN_HPP_INLINE void
      Device::setHdrMetadataEXT( ArrayProxy<const VULKAN_HPP_NAMESPACE::SwapchainKHR> const &   swapchains,
                                 ArrayProxy<const VULKAN_HPP_NAMESPACE::HdrMetadataEXT> const & metadata ) const
      VULKAN_HPP_NOEXCEPT_WHEN_NO_EXCEPTIONS
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkSetHdrMetadataEXT &&
                         "Function <vkSetHdrMetadataEXT> needs extension <VK_EXT_hdr_metadata> enabled!" );

#  ifdef VULKAN_HPP_NO_EXCEPTIONS
      VULKAN_HPP_ASSERT( swapchains.size() == metadata.size() );
#  else
      if ( swapchains.size() != metadata.size() )
      {
        throw LogicError( VULKAN_HPP_NAMESPACE_STRING
                          "::Device::setHdrMetadataEXT: swapchains.size() != metadata.size()" );
      }
#  endif /*VULKAN_HPP_NO_EXCEPTIONS*/

      getDispatcher()->vkSetHdrMetadataEXT( static_cast<VkDevice>( m_device ),
                                            swapchains.size(),
                                            reinterpret_cast<const VkSwapchainKHR *>( swapchains.data() ),
                                            reinterpret_cast<const VkHdrMetadataEXT *>( metadata.data() ) );
    }

    //=== VK_KHR_create_renderpass2 ===

    VULKAN_HPP_INLINE void
      CommandBuffer::beginRenderPass2KHR( const RenderPassBeginInfo & renderPassBegin,
                                          const SubpassBeginInfo &    subpassBeginInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdBeginRenderPass2KHR &&
                         "Function <vkCmdBeginRenderPass2KHR> needs extension <VK_KHR_create_renderpass2> enabled!" );

      getDispatcher()->vkCmdBeginRenderPass2KHR( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                 reinterpret_cast<const VkRenderPassBeginInfo *>( &renderPassBegin ),
                                                 reinterpret_cast<const VkSubpassBeginInfo *>( &subpassBeginInfo ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::nextSubpass2KHR( const SubpassBeginInfo & subpassBeginInfo,
                                      const SubpassEndInfo &   subpassEndInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdNextSubpass2KHR &&
                         "Function <vkCmdNextSubpass2KHR> needs extension <VK_KHR_create_renderpass2> enabled!" );

      getDispatcher()->vkCmdNextSubpass2KHR( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                             reinterpret_cast<const VkSubpassBeginInfo *>( &subpassBeginInfo ),
                                             reinterpret_cast<const VkSubpassEndInfo *>( &subpassEndInfo ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::endRenderPass2KHR( const SubpassEndInfo & subpassEndInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdEndRenderPass2KHR &&
                         "Function <vkCmdEndRenderPass2KHR> needs extension <VK_KHR_create_renderpass2> enabled!" );

      getDispatcher()->vkCmdEndRenderPass2KHR( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                               reinterpret_cast<const VkSubpassEndInfo *>( &subpassEndInfo ) );
    }

    //=== VK_KHR_shared_presentable_image ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::Result SwapchainKHR::getStatus() const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetSwapchainStatusKHR &&
        "Function <vkGetSwapchainStatusKHR> needs extension <VK_KHR_shared_presentable_image> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetSwapchainStatusKHR(
          static_cast<VkDevice>( m_device ), static_cast<VkSwapchainKHR>( m_swapchainKHR ) ) );
      if ( ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eSuboptimalKHR ) )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::SwapchainKHR::getStatus" );
      }
      return result;
    }

    //=== VK_KHR_external_fence_capabilities ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::ExternalFenceProperties
                                           PhysicalDevice::getExternalFencePropertiesKHR( const PhysicalDeviceExternalFenceInfo & externalFenceInfo ) const
      VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceExternalFencePropertiesKHR &&
        "Function <vkGetPhysicalDeviceExternalFencePropertiesKHR> needs extension <VK_KHR_external_fence_capabilities> enabled!" );

      VULKAN_HPP_NAMESPACE::ExternalFenceProperties externalFenceProperties;
      getDispatcher()->vkGetPhysicalDeviceExternalFencePropertiesKHR(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        reinterpret_cast<const VkPhysicalDeviceExternalFenceInfo *>( &externalFenceInfo ),
        reinterpret_cast<VkExternalFenceProperties *>( &externalFenceProperties ) );
      return externalFenceProperties;
    }

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_KHR_external_fence_win32 ===

    VULKAN_HPP_INLINE void
      Device::importFenceWin32HandleKHR( const ImportFenceWin32HandleInfoKHR & importFenceWin32HandleInfo ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkImportFenceWin32HandleKHR &&
        "Function <vkImportFenceWin32HandleKHR> needs extension <VK_KHR_external_fence_win32> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkImportFenceWin32HandleKHR(
          static_cast<VkDevice>( m_device ),
          reinterpret_cast<const VkImportFenceWin32HandleInfoKHR *>( &importFenceWin32HandleInfo ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::importFenceWin32HandleKHR" );
      }
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE HANDLE
                                           Device::getFenceWin32HandleKHR( const FenceGetWin32HandleInfoKHR & getWin32HandleInfo ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkGetFenceWin32HandleKHR &&
                         "Function <vkGetFenceWin32HandleKHR> needs extension <VK_KHR_external_fence_win32> enabled!" );

      HANDLE                       handle;
      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetFenceWin32HandleKHR(
          static_cast<VkDevice>( m_device ),
          reinterpret_cast<const VkFenceGetWin32HandleInfoKHR *>( &getWin32HandleInfo ),
          &handle ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::getFenceWin32HandleKHR" );
      }
      return handle;
    }
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

    //=== VK_KHR_external_fence_fd ===

    VULKAN_HPP_INLINE void Device::importFenceFdKHR( const ImportFenceFdInfoKHR & importFenceFdInfo ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkImportFenceFdKHR &&
                         "Function <vkImportFenceFdKHR> needs extension <VK_KHR_external_fence_fd> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkImportFenceFdKHR(
          static_cast<VkDevice>( m_device ), reinterpret_cast<const VkImportFenceFdInfoKHR *>( &importFenceFdInfo ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::importFenceFdKHR" );
      }
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE int Device::getFenceFdKHR( const FenceGetFdInfoKHR & getFdInfo ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkGetFenceFdKHR &&
                         "Function <vkGetFenceFdKHR> needs extension <VK_KHR_external_fence_fd> enabled!" );

      int                          fd;
      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetFenceFdKHR(
        static_cast<VkDevice>( m_device ), reinterpret_cast<const VkFenceGetFdInfoKHR *>( &getFdInfo ), &fd ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::getFenceFdKHR" );
      }
      return fd;
    }

    //=== VK_KHR_performance_query ===

    VULKAN_HPP_NODISCARD
      VULKAN_HPP_INLINE std::pair<std::vector<PerformanceCounterKHR>, std::vector<PerformanceCounterDescriptionKHR>>
                        PhysicalDevice::enumerateQueueFamilyPerformanceQueryCountersKHR( uint32_t queueFamilyIndex ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR &&
        "Function <vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR> needs extension <VK_KHR_performance_query> enabled!" );

      std::pair<std::vector<PerformanceCounterKHR>, std::vector<PerformanceCounterDescriptionKHR>> data;
      std::vector<PerformanceCounterKHR> &            counters            = data.first;
      std::vector<PerformanceCounterDescriptionKHR> & counterDescriptions = data.second;
      uint32_t                                        counterCount;
      VULKAN_HPP_NAMESPACE::Result                    result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR(
            static_cast<VkPhysicalDevice>( m_physicalDevice ), queueFamilyIndex, &counterCount, nullptr, nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && counterCount )
        {
          counters.resize( counterCount );
          counterDescriptions.resize( counterCount );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
            getDispatcher()->vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR(
              static_cast<VkPhysicalDevice>( m_physicalDevice ),
              queueFamilyIndex,
              &counterCount,
              reinterpret_cast<VkPerformanceCounterKHR *>( counters.data() ),
              reinterpret_cast<VkPerformanceCounterDescriptionKHR *>( counterDescriptions.data() ) ) );
          VULKAN_HPP_ASSERT( counterCount <= counters.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( counterCount < counters.size() ) )
      {
        counters.resize( counterCount );
        counterDescriptions.resize( counterCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException(
          result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::enumerateQueueFamilyPerformanceQueryCountersKHR" );
      }
      return data;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE uint32_t PhysicalDevice::getQueueFamilyPerformanceQueryPassesKHR(
      const QueryPoolPerformanceCreateInfoKHR & performanceQueryCreateInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR &&
        "Function <vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR> needs extension <VK_KHR_performance_query> enabled!" );

      uint32_t numPasses;
      getDispatcher()->vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        reinterpret_cast<const VkQueryPoolPerformanceCreateInfoKHR *>( &performanceQueryCreateInfo ),
        &numPasses );
      return numPasses;
    }

    VULKAN_HPP_INLINE void Device::acquireProfilingLockKHR( const AcquireProfilingLockInfoKHR & info ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkAcquireProfilingLockKHR &&
                         "Function <vkAcquireProfilingLockKHR> needs extension <VK_KHR_performance_query> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkAcquireProfilingLockKHR(
          static_cast<VkDevice>( m_device ), reinterpret_cast<const VkAcquireProfilingLockInfoKHR *>( &info ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::acquireProfilingLockKHR" );
      }
    }

    VULKAN_HPP_INLINE void Device::releaseProfilingLockKHR() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkReleaseProfilingLockKHR &&
                         "Function <vkReleaseProfilingLockKHR> needs extension <VK_KHR_performance_query> enabled!" );

      getDispatcher()->vkReleaseProfilingLockKHR( static_cast<VkDevice>( m_device ) );
    }

    //=== VK_KHR_get_surface_capabilities2 ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::SurfaceCapabilities2KHR
                                           PhysicalDevice::getSurfaceCapabilities2KHR( const PhysicalDeviceSurfaceInfo2KHR & surfaceInfo ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceSurfaceCapabilities2KHR &&
        "Function <vkGetPhysicalDeviceSurfaceCapabilities2KHR> needs extension <VK_KHR_get_surface_capabilities2> enabled!" );

      VULKAN_HPP_NAMESPACE::SurfaceCapabilities2KHR surfaceCapabilities;
      VULKAN_HPP_NAMESPACE::Result                  result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceSurfaceCapabilities2KHR(
          static_cast<VkPhysicalDevice>( m_physicalDevice ),
          reinterpret_cast<const VkPhysicalDeviceSurfaceInfo2KHR *>( &surfaceInfo ),
          reinterpret_cast<VkSurfaceCapabilities2KHR *>( &surfaceCapabilities ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getSurfaceCapabilities2KHR" );
      }
      return surfaceCapabilities;
    }

    template <typename X, typename Y, typename... Z>
    VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...>
                         PhysicalDevice::getSurfaceCapabilities2KHR( const PhysicalDeviceSurfaceInfo2KHR & surfaceInfo ) const
    {
      StructureChain<X, Y, Z...>                      structureChain;
      VULKAN_HPP_NAMESPACE::SurfaceCapabilities2KHR & surfaceCapabilities =
        structureChain.template get<VULKAN_HPP_NAMESPACE::SurfaceCapabilities2KHR>();
      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceSurfaceCapabilities2KHR(
          static_cast<VkPhysicalDevice>( m_physicalDevice ),
          reinterpret_cast<const VkPhysicalDeviceSurfaceInfo2KHR *>( &surfaceInfo ),
          reinterpret_cast<VkSurfaceCapabilities2KHR *>( &surfaceCapabilities ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getSurfaceCapabilities2KHR" );
      }
      return structureChain;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::SurfaceFormat2KHR>
                                           PhysicalDevice::getSurfaceFormats2KHR( const PhysicalDeviceSurfaceInfo2KHR & surfaceInfo ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceSurfaceFormats2KHR &&
        "Function <vkGetPhysicalDeviceSurfaceFormats2KHR> needs extension <VK_KHR_get_surface_capabilities2> enabled!" );

      std::vector<VULKAN_HPP_NAMESPACE::SurfaceFormat2KHR> surfaceFormats;
      uint32_t                                             surfaceFormatCount;
      VULKAN_HPP_NAMESPACE::Result                         result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceSurfaceFormats2KHR(
          static_cast<VkPhysicalDevice>( m_physicalDevice ),
          reinterpret_cast<const VkPhysicalDeviceSurfaceInfo2KHR *>( &surfaceInfo ),
          &surfaceFormatCount,
          nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && surfaceFormatCount )
        {
          surfaceFormats.resize( surfaceFormatCount );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceSurfaceFormats2KHR(
            static_cast<VkPhysicalDevice>( m_physicalDevice ),
            reinterpret_cast<const VkPhysicalDeviceSurfaceInfo2KHR *>( &surfaceInfo ),
            &surfaceFormatCount,
            reinterpret_cast<VkSurfaceFormat2KHR *>( surfaceFormats.data() ) ) );
          VULKAN_HPP_ASSERT( surfaceFormatCount <= surfaceFormats.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( surfaceFormatCount < surfaceFormats.size() ) )
      {
        surfaceFormats.resize( surfaceFormatCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getSurfaceFormats2KHR" );
      }
      return surfaceFormats;
    }

    //=== VK_KHR_get_display_properties2 ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::DisplayProperties2KHR>
                                           PhysicalDevice::getDisplayProperties2KHR() const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceDisplayProperties2KHR &&
        "Function <vkGetPhysicalDeviceDisplayProperties2KHR> needs extension <VK_KHR_get_display_properties2> enabled!" );

      std::vector<VULKAN_HPP_NAMESPACE::DisplayProperties2KHR> properties;
      uint32_t                                                 propertyCount;
      VULKAN_HPP_NAMESPACE::Result                             result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceDisplayProperties2KHR(
          static_cast<VkPhysicalDevice>( m_physicalDevice ), &propertyCount, nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && propertyCount )
        {
          properties.resize( propertyCount );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceDisplayProperties2KHR(
            static_cast<VkPhysicalDevice>( m_physicalDevice ),
            &propertyCount,
            reinterpret_cast<VkDisplayProperties2KHR *>( properties.data() ) ) );
          VULKAN_HPP_ASSERT( propertyCount <= properties.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( propertyCount < properties.size() ) )
      {
        properties.resize( propertyCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getDisplayProperties2KHR" );
      }
      return properties;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::DisplayPlaneProperties2KHR>
                                           PhysicalDevice::getDisplayPlaneProperties2KHR() const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceDisplayPlaneProperties2KHR &&
        "Function <vkGetPhysicalDeviceDisplayPlaneProperties2KHR> needs extension <VK_KHR_get_display_properties2> enabled!" );

      std::vector<VULKAN_HPP_NAMESPACE::DisplayPlaneProperties2KHR> properties;
      uint32_t                                                      propertyCount;
      VULKAN_HPP_NAMESPACE::Result                                  result;
      do
      {
        result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceDisplayPlaneProperties2KHR(
            static_cast<VkPhysicalDevice>( m_physicalDevice ), &propertyCount, nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && propertyCount )
        {
          properties.resize( propertyCount );
          result =
            static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceDisplayPlaneProperties2KHR(
              static_cast<VkPhysicalDevice>( m_physicalDevice ),
              &propertyCount,
              reinterpret_cast<VkDisplayPlaneProperties2KHR *>( properties.data() ) ) );
          VULKAN_HPP_ASSERT( propertyCount <= properties.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( propertyCount < properties.size() ) )
      {
        properties.resize( propertyCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getDisplayPlaneProperties2KHR" );
      }
      return properties;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::DisplayModeProperties2KHR>
                                           DisplayKHR::getModeProperties2() const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetDisplayModeProperties2KHR &&
        "Function <vkGetDisplayModeProperties2KHR> needs extension <VK_KHR_get_display_properties2> enabled!" );

      std::vector<VULKAN_HPP_NAMESPACE::DisplayModeProperties2KHR> properties;
      uint32_t                                                     propertyCount;
      VULKAN_HPP_NAMESPACE::Result                                 result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkGetDisplayModeProperties2KHR( static_cast<VkPhysicalDevice>( m_physicalDevice ),
                                                           static_cast<VkDisplayKHR>( m_displayKHR ),
                                                           &propertyCount,
                                                           nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && propertyCount )
        {
          properties.resize( propertyCount );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetDisplayModeProperties2KHR(
            static_cast<VkPhysicalDevice>( m_physicalDevice ),
            static_cast<VkDisplayKHR>( m_displayKHR ),
            &propertyCount,
            reinterpret_cast<VkDisplayModeProperties2KHR *>( properties.data() ) ) );
          VULKAN_HPP_ASSERT( propertyCount <= properties.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( propertyCount < properties.size() ) )
      {
        properties.resize( propertyCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::DisplayKHR::getModeProperties2" );
      }
      return properties;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::DisplayPlaneCapabilities2KHR
                                           PhysicalDevice::getDisplayPlaneCapabilities2KHR( const DisplayPlaneInfo2KHR & displayPlaneInfo ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetDisplayPlaneCapabilities2KHR &&
        "Function <vkGetDisplayPlaneCapabilities2KHR> needs extension <VK_KHR_get_display_properties2> enabled!" );

      VULKAN_HPP_NAMESPACE::DisplayPlaneCapabilities2KHR capabilities;
      VULKAN_HPP_NAMESPACE::Result                       result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetDisplayPlaneCapabilities2KHR(
          static_cast<VkPhysicalDevice>( m_physicalDevice ),
          reinterpret_cast<const VkDisplayPlaneInfo2KHR *>( &displayPlaneInfo ),
          reinterpret_cast<VkDisplayPlaneCapabilities2KHR *>( &capabilities ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getDisplayPlaneCapabilities2KHR" );
      }
      return capabilities;
    }

    //=== VK_EXT_debug_utils ===

    VULKAN_HPP_INLINE void Device::setDebugUtilsObjectNameEXT( const DebugUtilsObjectNameInfoEXT & nameInfo ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkSetDebugUtilsObjectNameEXT &&
                         "Function <vkSetDebugUtilsObjectNameEXT> needs extension <VK_EXT_debug_utils> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkSetDebugUtilsObjectNameEXT(
          static_cast<VkDevice>( m_device ), reinterpret_cast<const VkDebugUtilsObjectNameInfoEXT *>( &nameInfo ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::setDebugUtilsObjectNameEXT" );
      }
    }

    VULKAN_HPP_INLINE void Device::setDebugUtilsObjectTagEXT( const DebugUtilsObjectTagInfoEXT & tagInfo ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkSetDebugUtilsObjectTagEXT &&
                         "Function <vkSetDebugUtilsObjectTagEXT> needs extension <VK_EXT_debug_utils> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkSetDebugUtilsObjectTagEXT(
          static_cast<VkDevice>( m_device ), reinterpret_cast<const VkDebugUtilsObjectTagInfoEXT *>( &tagInfo ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::setDebugUtilsObjectTagEXT" );
      }
    }

    VULKAN_HPP_INLINE void
      Queue::beginDebugUtilsLabelEXT( const DebugUtilsLabelEXT & labelInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkQueueBeginDebugUtilsLabelEXT &&
                         "Function <vkQueueBeginDebugUtilsLabelEXT> needs extension <VK_EXT_debug_utils> enabled!" );

      getDispatcher()->vkQueueBeginDebugUtilsLabelEXT( static_cast<VkQueue>( m_queue ),
                                                       reinterpret_cast<const VkDebugUtilsLabelEXT *>( &labelInfo ) );
    }

    VULKAN_HPP_INLINE void Queue::endDebugUtilsLabelEXT() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkQueueEndDebugUtilsLabelEXT &&
                         "Function <vkQueueEndDebugUtilsLabelEXT> needs extension <VK_EXT_debug_utils> enabled!" );

      getDispatcher()->vkQueueEndDebugUtilsLabelEXT( static_cast<VkQueue>( m_queue ) );
    }

    VULKAN_HPP_INLINE void
      Queue::insertDebugUtilsLabelEXT( const DebugUtilsLabelEXT & labelInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkQueueInsertDebugUtilsLabelEXT &&
                         "Function <vkQueueInsertDebugUtilsLabelEXT> needs extension <VK_EXT_debug_utils> enabled!" );

      getDispatcher()->vkQueueInsertDebugUtilsLabelEXT( static_cast<VkQueue>( m_queue ),
                                                        reinterpret_cast<const VkDebugUtilsLabelEXT *>( &labelInfo ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::beginDebugUtilsLabelEXT( const DebugUtilsLabelEXT & labelInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdBeginDebugUtilsLabelEXT &&
                         "Function <vkCmdBeginDebugUtilsLabelEXT> needs extension <VK_EXT_debug_utils> enabled!" );

      getDispatcher()->vkCmdBeginDebugUtilsLabelEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                     reinterpret_cast<const VkDebugUtilsLabelEXT *>( &labelInfo ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::endDebugUtilsLabelEXT() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdEndDebugUtilsLabelEXT &&
                         "Function <vkCmdEndDebugUtilsLabelEXT> needs extension <VK_EXT_debug_utils> enabled!" );

      getDispatcher()->vkCmdEndDebugUtilsLabelEXT( static_cast<VkCommandBuffer>( m_commandBuffer ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::insertDebugUtilsLabelEXT( const DebugUtilsLabelEXT & labelInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdInsertDebugUtilsLabelEXT &&
                         "Function <vkCmdInsertDebugUtilsLabelEXT> needs extension <VK_EXT_debug_utils> enabled!" );

      getDispatcher()->vkCmdInsertDebugUtilsLabelEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                      reinterpret_cast<const VkDebugUtilsLabelEXT *>( &labelInfo ) );
    }

    VULKAN_HPP_INLINE void Instance::submitDebugUtilsMessageEXT(
      VULKAN_HPP_NAMESPACE::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
      VULKAN_HPP_NAMESPACE::DebugUtilsMessageTypeFlagsEXT        messageTypes,
      const DebugUtilsMessengerCallbackDataEXT &                 callbackData ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkSubmitDebugUtilsMessageEXT &&
                         "Function <vkSubmitDebugUtilsMessageEXT> needs extension <VK_EXT_debug_utils> enabled!" );

      getDispatcher()->vkSubmitDebugUtilsMessageEXT(
        static_cast<VkInstance>( m_instance ),
        static_cast<VkDebugUtilsMessageSeverityFlagBitsEXT>( messageSeverity ),
        static_cast<VkDebugUtilsMessageTypeFlagsEXT>( messageTypes ),
        reinterpret_cast<const VkDebugUtilsMessengerCallbackDataEXT *>( &callbackData ) );
    }

#  if defined( VK_USE_PLATFORM_ANDROID_KHR )
    //=== VK_ANDROID_external_memory_android_hardware_buffer ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::AndroidHardwareBufferPropertiesANDROID
                                           Device::getAndroidHardwareBufferPropertiesANDROID( const struct AHardwareBuffer & buffer ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetAndroidHardwareBufferPropertiesANDROID &&
        "Function <vkGetAndroidHardwareBufferPropertiesANDROID> needs extension <VK_ANDROID_external_memory_android_hardware_buffer> enabled!" );

      VULKAN_HPP_NAMESPACE::AndroidHardwareBufferPropertiesANDROID properties;
      VULKAN_HPP_NAMESPACE::Result                                 result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetAndroidHardwareBufferPropertiesANDROID(
          static_cast<VkDevice>( m_device ),
          &buffer,
          reinterpret_cast<VkAndroidHardwareBufferPropertiesANDROID *>( &properties ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result,
                              VULKAN_HPP_NAMESPACE_STRING "::Device::getAndroidHardwareBufferPropertiesANDROID" );
      }
      return properties;
    }

    template <typename X, typename Y, typename... Z>
    VULKAN_HPP_NODISCARD StructureChain<X, Y, Z...>
                         Device::getAndroidHardwareBufferPropertiesANDROID( const struct AHardwareBuffer & buffer ) const
    {
      StructureChain<X, Y, Z...>                                     structureChain;
      VULKAN_HPP_NAMESPACE::AndroidHardwareBufferPropertiesANDROID & properties =
        structureChain.template get<VULKAN_HPP_NAMESPACE::AndroidHardwareBufferPropertiesANDROID>();
      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetAndroidHardwareBufferPropertiesANDROID(
          static_cast<VkDevice>( m_device ),
          &buffer,
          reinterpret_cast<VkAndroidHardwareBufferPropertiesANDROID *>( &properties ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result,
                              VULKAN_HPP_NAMESPACE_STRING "::Device::getAndroidHardwareBufferPropertiesANDROID" );
      }
      return structureChain;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE struct AHardwareBuffer *
                         Device::getMemoryAndroidHardwareBufferANDROID( const MemoryGetAndroidHardwareBufferInfoANDROID & info ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetMemoryAndroidHardwareBufferANDROID &&
        "Function <vkGetMemoryAndroidHardwareBufferANDROID> needs extension <VK_ANDROID_external_memory_android_hardware_buffer> enabled!" );

      struct AHardwareBuffer *     buffer;
      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetMemoryAndroidHardwareBufferANDROID(
          static_cast<VkDevice>( m_device ),
          reinterpret_cast<const VkMemoryGetAndroidHardwareBufferInfoANDROID *>( &info ),
          &buffer ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::getMemoryAndroidHardwareBufferANDROID" );
      }
      return buffer;
    }
#  endif /*VK_USE_PLATFORM_ANDROID_KHR*/

    //=== VK_EXT_sample_locations ===

    VULKAN_HPP_INLINE void CommandBuffer::setSampleLocationsEXT(
      const SampleLocationsInfoEXT & sampleLocationsInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdSetSampleLocationsEXT &&
                         "Function <vkCmdSetSampleLocationsEXT> needs extension <VK_EXT_sample_locations> enabled!" );

      getDispatcher()->vkCmdSetSampleLocationsEXT(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        reinterpret_cast<const VkSampleLocationsInfoEXT *>( &sampleLocationsInfo ) );
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::MultisamplePropertiesEXT
                                           PhysicalDevice::getMultisamplePropertiesEXT( VULKAN_HPP_NAMESPACE::SampleCountFlagBits samples ) const
      VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceMultisamplePropertiesEXT &&
        "Function <vkGetPhysicalDeviceMultisamplePropertiesEXT> needs extension <VK_EXT_sample_locations> enabled!" );

      VULKAN_HPP_NAMESPACE::MultisamplePropertiesEXT multisampleProperties;
      getDispatcher()->vkGetPhysicalDeviceMultisamplePropertiesEXT(
        static_cast<VkPhysicalDevice>( m_physicalDevice ),
        static_cast<VkSampleCountFlagBits>( samples ),
        reinterpret_cast<VkMultisamplePropertiesEXT *>( &multisampleProperties ) );
      return multisampleProperties;
    }

    //=== VK_KHR_get_memory_requirements2 ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::MemoryRequirements2
                                           Device::getImageMemoryRequirements2KHR( const ImageMemoryRequirementsInfo2 & info ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetImageMemoryRequirements2KHR &&
        "Function <vkGetImageMemoryRequirements2KHR> needs extension <VK_KHR_get_memory_requirements2> enabled!" );

      VULKAN_HPP_NAMESPACE::MemoryRequirements2 memoryRequirements;
      getDispatcher()->vkGetImageMemoryRequirements2KHR(
        static_cast<VkDevice>( m_device ),
        reinterpret_cast<const VkImageMemoryRequirementsInfo2 *>( &info ),
        reinterpret_cast<VkMemoryRequirements2 *>( &memoryRequirements ) );
      return memoryRequirements;
    }

    template <typename X, typename Y, typename... Z>
    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE StructureChain<X, Y, Z...>
                                           Device::getImageMemoryRequirements2KHR( const ImageMemoryRequirementsInfo2 & info ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetImageMemoryRequirements2KHR &&
        "Function <vkGetImageMemoryRequirements2KHR> needs extension <VK_KHR_get_memory_requirements2> enabled!" );

      StructureChain<X, Y, Z...>                  structureChain;
      VULKAN_HPP_NAMESPACE::MemoryRequirements2 & memoryRequirements =
        structureChain.template get<VULKAN_HPP_NAMESPACE::MemoryRequirements2>();
      getDispatcher()->vkGetImageMemoryRequirements2KHR(
        static_cast<VkDevice>( m_device ),
        reinterpret_cast<const VkImageMemoryRequirementsInfo2 *>( &info ),
        reinterpret_cast<VkMemoryRequirements2 *>( &memoryRequirements ) );
      return structureChain;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::MemoryRequirements2
                                           Device::getBufferMemoryRequirements2KHR( const BufferMemoryRequirementsInfo2 & info ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetBufferMemoryRequirements2KHR &&
        "Function <vkGetBufferMemoryRequirements2KHR> needs extension <VK_KHR_get_memory_requirements2> enabled!" );

      VULKAN_HPP_NAMESPACE::MemoryRequirements2 memoryRequirements;
      getDispatcher()->vkGetBufferMemoryRequirements2KHR(
        static_cast<VkDevice>( m_device ),
        reinterpret_cast<const VkBufferMemoryRequirementsInfo2 *>( &info ),
        reinterpret_cast<VkMemoryRequirements2 *>( &memoryRequirements ) );
      return memoryRequirements;
    }

    template <typename X, typename Y, typename... Z>
    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE StructureChain<X, Y, Z...>
                                           Device::getBufferMemoryRequirements2KHR( const BufferMemoryRequirementsInfo2 & info ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetBufferMemoryRequirements2KHR &&
        "Function <vkGetBufferMemoryRequirements2KHR> needs extension <VK_KHR_get_memory_requirements2> enabled!" );

      StructureChain<X, Y, Z...>                  structureChain;
      VULKAN_HPP_NAMESPACE::MemoryRequirements2 & memoryRequirements =
        structureChain.template get<VULKAN_HPP_NAMESPACE::MemoryRequirements2>();
      getDispatcher()->vkGetBufferMemoryRequirements2KHR(
        static_cast<VkDevice>( m_device ),
        reinterpret_cast<const VkBufferMemoryRequirementsInfo2 *>( &info ),
        reinterpret_cast<VkMemoryRequirements2 *>( &memoryRequirements ) );
      return structureChain;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::SparseImageMemoryRequirements2>
                                           Device::getImageSparseMemoryRequirements2KHR( const ImageSparseMemoryRequirementsInfo2 & info ) const
      VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetImageSparseMemoryRequirements2KHR &&
        "Function <vkGetImageSparseMemoryRequirements2KHR> needs extension <VK_KHR_get_memory_requirements2> enabled!" );

      uint32_t sparseMemoryRequirementCount;
      getDispatcher()->vkGetImageSparseMemoryRequirements2KHR(
        static_cast<VkDevice>( m_device ),
        reinterpret_cast<const VkImageSparseMemoryRequirementsInfo2 *>( &info ),
        &sparseMemoryRequirementCount,
        nullptr );
      std::vector<VULKAN_HPP_NAMESPACE::SparseImageMemoryRequirements2> sparseMemoryRequirements(
        sparseMemoryRequirementCount );
      getDispatcher()->vkGetImageSparseMemoryRequirements2KHR(
        static_cast<VkDevice>( m_device ),
        reinterpret_cast<const VkImageSparseMemoryRequirementsInfo2 *>( &info ),
        &sparseMemoryRequirementCount,
        reinterpret_cast<VkSparseImageMemoryRequirements2 *>( sparseMemoryRequirements.data() ) );
      VULKAN_HPP_ASSERT( sparseMemoryRequirementCount <= sparseMemoryRequirements.size() );
      return sparseMemoryRequirements;
    }

    //=== VK_KHR_acceleration_structure ===

    VULKAN_HPP_INLINE void CommandBuffer::buildAccelerationStructuresKHR(
      ArrayProxy<const VULKAN_HPP_NAMESPACE::AccelerationStructureBuildGeometryInfoKHR> const &      infos,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::AccelerationStructureBuildRangeInfoKHR * const> const & pBuildRangeInfos )
      const VULKAN_HPP_NOEXCEPT_WHEN_NO_EXCEPTIONS
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdBuildAccelerationStructuresKHR &&
        "Function <vkCmdBuildAccelerationStructuresKHR> needs extension <VK_KHR_acceleration_structure> enabled!" );

#  ifdef VULKAN_HPP_NO_EXCEPTIONS
      VULKAN_HPP_ASSERT( infos.size() == pBuildRangeInfos.size() );
#  else
      if ( infos.size() != pBuildRangeInfos.size() )
      {
        throw LogicError( VULKAN_HPP_NAMESPACE_STRING
                          "::CommandBuffer::buildAccelerationStructuresKHR: infos.size() != pBuildRangeInfos.size()" );
      }
#  endif /*VULKAN_HPP_NO_EXCEPTIONS*/

      getDispatcher()->vkCmdBuildAccelerationStructuresKHR(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        infos.size(),
        reinterpret_cast<const VkAccelerationStructureBuildGeometryInfoKHR *>( infos.data() ),
        reinterpret_cast<const VkAccelerationStructureBuildRangeInfoKHR * const *>( pBuildRangeInfos.data() ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::buildAccelerationStructuresIndirectKHR(
      ArrayProxy<const VULKAN_HPP_NAMESPACE::AccelerationStructureBuildGeometryInfoKHR> const & infos,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::DeviceAddress> const &                             indirectDeviceAddresses,
      ArrayProxy<const uint32_t> const &                                                        indirectStrides,
      ArrayProxy<const uint32_t * const> const & pMaxPrimitiveCounts ) const VULKAN_HPP_NOEXCEPT_WHEN_NO_EXCEPTIONS
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdBuildAccelerationStructuresIndirectKHR &&
        "Function <vkCmdBuildAccelerationStructuresIndirectKHR> needs extension <VK_KHR_acceleration_structure> enabled!" );

#  ifdef VULKAN_HPP_NO_EXCEPTIONS
      VULKAN_HPP_ASSERT( infos.size() == indirectDeviceAddresses.size() );
      VULKAN_HPP_ASSERT( infos.size() == indirectStrides.size() );
      VULKAN_HPP_ASSERT( infos.size() == pMaxPrimitiveCounts.size() );
#  else
      if ( infos.size() != indirectDeviceAddresses.size() )
      {
        throw LogicError(
          VULKAN_HPP_NAMESPACE_STRING
          "::CommandBuffer::buildAccelerationStructuresIndirectKHR: infos.size() != indirectDeviceAddresses.size()" );
      }
      if ( infos.size() != indirectStrides.size() )
      {
        throw LogicError(
          VULKAN_HPP_NAMESPACE_STRING
          "::CommandBuffer::buildAccelerationStructuresIndirectKHR: infos.size() != indirectStrides.size()" );
      }
      if ( infos.size() != pMaxPrimitiveCounts.size() )
      {
        throw LogicError(
          VULKAN_HPP_NAMESPACE_STRING
          "::CommandBuffer::buildAccelerationStructuresIndirectKHR: infos.size() != pMaxPrimitiveCounts.size()" );
      }
#  endif /*VULKAN_HPP_NO_EXCEPTIONS*/

      getDispatcher()->vkCmdBuildAccelerationStructuresIndirectKHR(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        infos.size(),
        reinterpret_cast<const VkAccelerationStructureBuildGeometryInfoKHR *>( infos.data() ),
        reinterpret_cast<const VkDeviceAddress *>( indirectDeviceAddresses.data() ),
        indirectStrides.data(),
        pMaxPrimitiveCounts.data() );
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::Result Device::buildAccelerationStructuresKHR(
      VULKAN_HPP_NAMESPACE::DeferredOperationKHR                                                     deferredOperation,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::AccelerationStructureBuildGeometryInfoKHR> const &      infos,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::AccelerationStructureBuildRangeInfoKHR * const> const & pBuildRangeInfos )
      const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkBuildAccelerationStructuresKHR &&
        "Function <vkBuildAccelerationStructuresKHR> needs extension <VK_KHR_acceleration_structure> enabled!" );
      if ( infos.size() != pBuildRangeInfos.size() )
      {
        throw LogicError( VULKAN_HPP_NAMESPACE_STRING
                          "::Device::buildAccelerationStructuresKHR: infos.size() != pBuildRangeInfos.size()" );
      }

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkBuildAccelerationStructuresKHR(
          static_cast<VkDevice>( m_device ),
          static_cast<VkDeferredOperationKHR>( deferredOperation ),
          infos.size(),
          reinterpret_cast<const VkAccelerationStructureBuildGeometryInfoKHR *>( infos.data() ),
          reinterpret_cast<const VkAccelerationStructureBuildRangeInfoKHR * const *>( pBuildRangeInfos.data() ) ) );
      if ( ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eOperationDeferredKHR ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eOperationNotDeferredKHR ) )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::buildAccelerationStructuresKHR" );
      }
      return result;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::Result
                                           Device::copyAccelerationStructureKHR( VULKAN_HPP_NAMESPACE::DeferredOperationKHR deferredOperation,
                                            const CopyAccelerationStructureInfoKHR &   info ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCopyAccelerationStructureKHR &&
        "Function <vkCopyAccelerationStructureKHR> needs extension <VK_KHR_acceleration_structure> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCopyAccelerationStructureKHR(
          static_cast<VkDevice>( m_device ),
          static_cast<VkDeferredOperationKHR>( deferredOperation ),
          reinterpret_cast<const VkCopyAccelerationStructureInfoKHR *>( &info ) ) );
      if ( ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eOperationDeferredKHR ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eOperationNotDeferredKHR ) )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::copyAccelerationStructureKHR" );
      }
      return result;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::Result
                                           Device::copyAccelerationStructureToMemoryKHR( VULKAN_HPP_NAMESPACE::DeferredOperationKHR       deferredOperation,
                                                    const CopyAccelerationStructureToMemoryInfoKHR & info ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCopyAccelerationStructureToMemoryKHR &&
        "Function <vkCopyAccelerationStructureToMemoryKHR> needs extension <VK_KHR_acceleration_structure> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCopyAccelerationStructureToMemoryKHR(
          static_cast<VkDevice>( m_device ),
          static_cast<VkDeferredOperationKHR>( deferredOperation ),
          reinterpret_cast<const VkCopyAccelerationStructureToMemoryInfoKHR *>( &info ) ) );
      if ( ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eOperationDeferredKHR ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eOperationNotDeferredKHR ) )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::copyAccelerationStructureToMemoryKHR" );
      }
      return result;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::Result
                                           Device::copyMemoryToAccelerationStructureKHR( VULKAN_HPP_NAMESPACE::DeferredOperationKHR       deferredOperation,
                                                    const CopyMemoryToAccelerationStructureInfoKHR & info ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCopyMemoryToAccelerationStructureKHR &&
        "Function <vkCopyMemoryToAccelerationStructureKHR> needs extension <VK_KHR_acceleration_structure> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCopyMemoryToAccelerationStructureKHR(
          static_cast<VkDevice>( m_device ),
          static_cast<VkDeferredOperationKHR>( deferredOperation ),
          reinterpret_cast<const VkCopyMemoryToAccelerationStructureInfoKHR *>( &info ) ) );
      if ( ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eOperationDeferredKHR ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eOperationNotDeferredKHR ) )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::copyMemoryToAccelerationStructureKHR" );
      }
      return result;
    }

    template <typename T>
    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<T> Device::writeAccelerationStructuresPropertiesKHR(
      ArrayProxy<const VULKAN_HPP_NAMESPACE::AccelerationStructureKHR> const & accelerationStructures,
      VULKAN_HPP_NAMESPACE::QueryType                                          queryType,
      size_t                                                                   dataSize,
      size_t                                                                   stride ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkWriteAccelerationStructuresPropertiesKHR &&
        "Function <vkWriteAccelerationStructuresPropertiesKHR> needs extension <VK_KHR_acceleration_structure> enabled!" );

      VULKAN_HPP_ASSERT( dataSize % sizeof( T ) == 0 );
      std::vector<T> data( dataSize / sizeof( T ) );
      Result         result = static_cast<Result>( getDispatcher()->vkWriteAccelerationStructuresPropertiesKHR(
        static_cast<VkDevice>( m_device ),
        accelerationStructures.size(),
        reinterpret_cast<const VkAccelerationStructureKHR *>( accelerationStructures.data() ),
        static_cast<VkQueryType>( queryType ),
        data.size() * sizeof( T ),
        reinterpret_cast<void *>( data.data() ),
        stride ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result,
                              VULKAN_HPP_NAMESPACE_STRING "::Device::writeAccelerationStructuresPropertiesKHR" );
      }
      return data;
    }

    template <typename T>
    VULKAN_HPP_NODISCARD T Device::writeAccelerationStructuresPropertyKHR(
      ArrayProxy<const VULKAN_HPP_NAMESPACE::AccelerationStructureKHR> const & accelerationStructures,
      VULKAN_HPP_NAMESPACE::QueryType                                          queryType,
      size_t                                                                   stride ) const
    {
      T      data;
      Result result = static_cast<Result>( getDispatcher()->vkWriteAccelerationStructuresPropertiesKHR(
        static_cast<VkDevice>( m_device ),
        accelerationStructures.size(),
        reinterpret_cast<const VkAccelerationStructureKHR *>( accelerationStructures.data() ),
        static_cast<VkQueryType>( queryType ),
        sizeof( T ),
        reinterpret_cast<void *>( &data ),
        stride ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::writeAccelerationStructuresPropertyKHR" );
      }
      return data;
    }

    VULKAN_HPP_INLINE void CommandBuffer::copyAccelerationStructureKHR(
      const CopyAccelerationStructureInfoKHR & info ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdCopyAccelerationStructureKHR &&
        "Function <vkCmdCopyAccelerationStructureKHR> needs extension <VK_KHR_acceleration_structure> enabled!" );

      getDispatcher()->vkCmdCopyAccelerationStructureKHR(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        reinterpret_cast<const VkCopyAccelerationStructureInfoKHR *>( &info ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::copyAccelerationStructureToMemoryKHR(
      const CopyAccelerationStructureToMemoryInfoKHR & info ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdCopyAccelerationStructureToMemoryKHR &&
        "Function <vkCmdCopyAccelerationStructureToMemoryKHR> needs extension <VK_KHR_acceleration_structure> enabled!" );

      getDispatcher()->vkCmdCopyAccelerationStructureToMemoryKHR(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        reinterpret_cast<const VkCopyAccelerationStructureToMemoryInfoKHR *>( &info ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::copyMemoryToAccelerationStructureKHR(
      const CopyMemoryToAccelerationStructureInfoKHR & info ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdCopyMemoryToAccelerationStructureKHR &&
        "Function <vkCmdCopyMemoryToAccelerationStructureKHR> needs extension <VK_KHR_acceleration_structure> enabled!" );

      getDispatcher()->vkCmdCopyMemoryToAccelerationStructureKHR(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        reinterpret_cast<const VkCopyMemoryToAccelerationStructureInfoKHR *>( &info ) );
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::DeviceAddress
                                           Device::getAccelerationStructureAddressKHR( const AccelerationStructureDeviceAddressInfoKHR & info ) const
      VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetAccelerationStructureDeviceAddressKHR &&
        "Function <vkGetAccelerationStructureDeviceAddressKHR> needs extension <VK_KHR_acceleration_structure> enabled!" );

      return static_cast<VULKAN_HPP_NAMESPACE::DeviceAddress>(
        getDispatcher()->vkGetAccelerationStructureDeviceAddressKHR(
          static_cast<VkDevice>( m_device ),
          reinterpret_cast<const VkAccelerationStructureDeviceAddressInfoKHR *>( &info ) ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::writeAccelerationStructuresPropertiesKHR(
      ArrayProxy<const VULKAN_HPP_NAMESPACE::AccelerationStructureKHR> const & accelerationStructures,
      VULKAN_HPP_NAMESPACE::QueryType                                          queryType,
      VULKAN_HPP_NAMESPACE::QueryPool                                          queryPool,
      uint32_t                                                                 firstQuery ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdWriteAccelerationStructuresPropertiesKHR &&
        "Function <vkCmdWriteAccelerationStructuresPropertiesKHR> needs extension <VK_KHR_acceleration_structure> enabled!" );

      getDispatcher()->vkCmdWriteAccelerationStructuresPropertiesKHR(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        accelerationStructures.size(),
        reinterpret_cast<const VkAccelerationStructureKHR *>( accelerationStructures.data() ),
        static_cast<VkQueryType>( queryType ),
        static_cast<VkQueryPool>( queryPool ),
        firstQuery );
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::AccelerationStructureCompatibilityKHR
                                           Device::getAccelerationStructureCompatibilityKHR( const AccelerationStructureVersionInfoKHR & versionInfo ) const
      VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetDeviceAccelerationStructureCompatibilityKHR &&
        "Function <vkGetDeviceAccelerationStructureCompatibilityKHR> needs extension <VK_KHR_acceleration_structure> enabled!" );

      VULKAN_HPP_NAMESPACE::AccelerationStructureCompatibilityKHR compatibility;
      getDispatcher()->vkGetDeviceAccelerationStructureCompatibilityKHR(
        static_cast<VkDevice>( m_device ),
        reinterpret_cast<const VkAccelerationStructureVersionInfoKHR *>( &versionInfo ),
        reinterpret_cast<VkAccelerationStructureCompatibilityKHR *>( &compatibility ) );
      return compatibility;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::AccelerationStructureBuildSizesInfoKHR
                                           Device::getAccelerationStructureBuildSizesKHR( VULKAN_HPP_NAMESPACE::AccelerationStructureBuildTypeKHR buildType,
                                                     const AccelerationStructureBuildGeometryInfoKHR &       buildInfo,
                                                     ArrayProxy<const uint32_t> const & maxPrimitiveCounts ) const
      VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetAccelerationStructureBuildSizesKHR &&
        "Function <vkGetAccelerationStructureBuildSizesKHR> needs extension <VK_KHR_acceleration_structure> enabled!" );

      VULKAN_HPP_NAMESPACE::AccelerationStructureBuildSizesInfoKHR sizeInfo;
      getDispatcher()->vkGetAccelerationStructureBuildSizesKHR(
        static_cast<VkDevice>( m_device ),
        static_cast<VkAccelerationStructureBuildTypeKHR>( buildType ),
        reinterpret_cast<const VkAccelerationStructureBuildGeometryInfoKHR *>( &buildInfo ),
        maxPrimitiveCounts.data(),
        reinterpret_cast<VkAccelerationStructureBuildSizesInfoKHR *>( &sizeInfo ) );
      return sizeInfo;
    }

    //=== VK_KHR_sampler_ycbcr_conversion ===

    VULKAN_HPP_INLINE void Device::destroySamplerYcbcrConversionKHR(
      VULKAN_HPP_NAMESPACE::SamplerYcbcrConversion ycbcrConversion,
      Optional<const AllocationCallbacks>          allocator ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkDestroySamplerYcbcrConversionKHR &&
        "Function <vkDestroySamplerYcbcrConversionKHR> needs extension <VK_KHR_sampler_ycbcr_conversion> enabled!" );

      getDispatcher()->vkDestroySamplerYcbcrConversionKHR(
        static_cast<VkDevice>( m_device ),
        static_cast<VkSamplerYcbcrConversion>( ycbcrConversion ),
        reinterpret_cast<const VkAllocationCallbacks *>(
          static_cast<const VULKAN_HPP_NAMESPACE::AllocationCallbacks *>( allocator ) ) );
    }

    //=== VK_KHR_bind_memory2 ===

    VULKAN_HPP_INLINE void Device::bindBufferMemory2KHR(
      ArrayProxy<const VULKAN_HPP_NAMESPACE::BindBufferMemoryInfo> const & bindInfos ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkBindBufferMemory2KHR &&
                         "Function <vkBindBufferMemory2KHR> needs extension <VK_KHR_bind_memory2> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkBindBufferMemory2KHR(
          static_cast<VkDevice>( m_device ),
          bindInfos.size(),
          reinterpret_cast<const VkBindBufferMemoryInfo *>( bindInfos.data() ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::bindBufferMemory2KHR" );
      }
    }

    VULKAN_HPP_INLINE void
      Device::bindImageMemory2KHR( ArrayProxy<const VULKAN_HPP_NAMESPACE::BindImageMemoryInfo> const & bindInfos ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkBindImageMemory2KHR &&
                         "Function <vkBindImageMemory2KHR> needs extension <VK_KHR_bind_memory2> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkBindImageMemory2KHR( static_cast<VkDevice>( m_device ),
                                                bindInfos.size(),
                                                reinterpret_cast<const VkBindImageMemoryInfo *>( bindInfos.data() ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::bindImageMemory2KHR" );
      }
    }

    //=== VK_EXT_image_drm_format_modifier ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::ImageDrmFormatModifierPropertiesEXT
                                           Image::getDrmFormatModifierPropertiesEXT() const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetImageDrmFormatModifierPropertiesEXT &&
        "Function <vkGetImageDrmFormatModifierPropertiesEXT> needs extension <VK_EXT_image_drm_format_modifier> enabled!" );

      VULKAN_HPP_NAMESPACE::ImageDrmFormatModifierPropertiesEXT properties;
      VULKAN_HPP_NAMESPACE::Result                              result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetImageDrmFormatModifierPropertiesEXT(
          static_cast<VkDevice>( m_device ),
          static_cast<VkImage>( m_image ),
          reinterpret_cast<VkImageDrmFormatModifierPropertiesEXT *>( &properties ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Image::getDrmFormatModifierPropertiesEXT" );
      }
      return properties;
    }

    //=== VK_EXT_validation_cache ===

    VULKAN_HPP_INLINE void
      ValidationCacheEXT::merge( ArrayProxy<const VULKAN_HPP_NAMESPACE::ValidationCacheEXT> const & srcCaches ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkMergeValidationCachesEXT &&
                         "Function <vkMergeValidationCachesEXT> needs extension <VK_EXT_validation_cache> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkMergeValidationCachesEXT(
          static_cast<VkDevice>( m_device ),
          static_cast<VkValidationCacheEXT>( m_validationCacheEXT ),
          srcCaches.size(),
          reinterpret_cast<const VkValidationCacheEXT *>( srcCaches.data() ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::ValidationCacheEXT::merge" );
      }
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<uint8_t> ValidationCacheEXT::getData() const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkGetValidationCacheDataEXT &&
                         "Function <vkGetValidationCacheDataEXT> needs extension <VK_EXT_validation_cache> enabled!" );

      std::vector<uint8_t>         data;
      size_t                       dataSize;
      VULKAN_HPP_NAMESPACE::Result result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkGetValidationCacheDataEXT( static_cast<VkDevice>( m_device ),
                                                        static_cast<VkValidationCacheEXT>( m_validationCacheEXT ),
                                                        &dataSize,
                                                        nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && dataSize )
        {
          data.resize( dataSize );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
            getDispatcher()->vkGetValidationCacheDataEXT( static_cast<VkDevice>( m_device ),
                                                          static_cast<VkValidationCacheEXT>( m_validationCacheEXT ),
                                                          &dataSize,
                                                          reinterpret_cast<void *>( data.data() ) ) );
          VULKAN_HPP_ASSERT( dataSize <= data.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( dataSize < data.size() ) )
      {
        data.resize( dataSize );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::ValidationCacheEXT::getData" );
      }
      return data;
    }

    //=== VK_NV_shading_rate_image ===

    VULKAN_HPP_INLINE void
      CommandBuffer::bindShadingRateImageNV( VULKAN_HPP_NAMESPACE::ImageView   imageView,
                                             VULKAN_HPP_NAMESPACE::ImageLayout imageLayout ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdBindShadingRateImageNV &&
                         "Function <vkCmdBindShadingRateImageNV> needs extension <VK_NV_shading_rate_image> enabled!" );

      getDispatcher()->vkCmdBindShadingRateImageNV( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                    static_cast<VkImageView>( imageView ),
                                                    static_cast<VkImageLayout>( imageLayout ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::setViewportShadingRatePaletteNV(
      uint32_t firstViewport, ArrayProxy<const VULKAN_HPP_NAMESPACE::ShadingRatePaletteNV> const & shadingRatePalettes )
      const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdSetViewportShadingRatePaletteNV &&
        "Function <vkCmdSetViewportShadingRatePaletteNV> needs extension <VK_NV_shading_rate_image> enabled!" );

      getDispatcher()->vkCmdSetViewportShadingRatePaletteNV(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        firstViewport,
        shadingRatePalettes.size(),
        reinterpret_cast<const VkShadingRatePaletteNV *>( shadingRatePalettes.data() ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::setCoarseSampleOrderNV(
      VULKAN_HPP_NAMESPACE::CoarseSampleOrderTypeNV                             sampleOrderType,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::CoarseSampleOrderCustomNV> const & customSampleOrders ) const
      VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdSetCoarseSampleOrderNV &&
                         "Function <vkCmdSetCoarseSampleOrderNV> needs extension <VK_NV_shading_rate_image> enabled!" );

      getDispatcher()->vkCmdSetCoarseSampleOrderNV(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        static_cast<VkCoarseSampleOrderTypeNV>( sampleOrderType ),
        customSampleOrders.size(),
        reinterpret_cast<const VkCoarseSampleOrderCustomNV *>( customSampleOrders.data() ) );
    }

    //=== VK_NV_ray_tracing ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::MemoryRequirements2KHR
                                           Device::getAccelerationStructureMemoryRequirementsNV(
        const AccelerationStructureMemoryRequirementsInfoNV & info ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetAccelerationStructureMemoryRequirementsNV &&
        "Function <vkGetAccelerationStructureMemoryRequirementsNV> needs extension <VK_NV_ray_tracing> enabled!" );

      VULKAN_HPP_NAMESPACE::MemoryRequirements2KHR memoryRequirements;
      getDispatcher()->vkGetAccelerationStructureMemoryRequirementsNV(
        static_cast<VkDevice>( m_device ),
        reinterpret_cast<const VkAccelerationStructureMemoryRequirementsInfoNV *>( &info ),
        reinterpret_cast<VkMemoryRequirements2KHR *>( &memoryRequirements ) );
      return memoryRequirements;
    }

    template <typename X, typename Y, typename... Z>
    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE StructureChain<X, Y, Z...>
                                           Device::getAccelerationStructureMemoryRequirementsNV(
        const AccelerationStructureMemoryRequirementsInfoNV & info ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetAccelerationStructureMemoryRequirementsNV &&
        "Function <vkGetAccelerationStructureMemoryRequirementsNV> needs extension <VK_NV_ray_tracing> enabled!" );

      StructureChain<X, Y, Z...>                     structureChain;
      VULKAN_HPP_NAMESPACE::MemoryRequirements2KHR & memoryRequirements =
        structureChain.template get<VULKAN_HPP_NAMESPACE::MemoryRequirements2KHR>();
      getDispatcher()->vkGetAccelerationStructureMemoryRequirementsNV(
        static_cast<VkDevice>( m_device ),
        reinterpret_cast<const VkAccelerationStructureMemoryRequirementsInfoNV *>( &info ),
        reinterpret_cast<VkMemoryRequirements2KHR *>( &memoryRequirements ) );
      return structureChain;
    }

    VULKAN_HPP_INLINE void Device::bindAccelerationStructureMemoryNV(
      ArrayProxy<const VULKAN_HPP_NAMESPACE::BindAccelerationStructureMemoryInfoNV> const & bindInfos ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkBindAccelerationStructureMemoryNV &&
        "Function <vkBindAccelerationStructureMemoryNV> needs extension <VK_NV_ray_tracing> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkBindAccelerationStructureMemoryNV(
          static_cast<VkDevice>( m_device ),
          bindInfos.size(),
          reinterpret_cast<const VkBindAccelerationStructureMemoryInfoNV *>( bindInfos.data() ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::bindAccelerationStructureMemoryNV" );
      }
    }

    VULKAN_HPP_INLINE void CommandBuffer::buildAccelerationStructureNV(
      const AccelerationStructureInfoNV &           info,
      VULKAN_HPP_NAMESPACE::Buffer                  instanceData,
      VULKAN_HPP_NAMESPACE::DeviceSize              instanceOffset,
      VULKAN_HPP_NAMESPACE::Bool32                  update,
      VULKAN_HPP_NAMESPACE::AccelerationStructureNV dst,
      VULKAN_HPP_NAMESPACE::AccelerationStructureNV src,
      VULKAN_HPP_NAMESPACE::Buffer                  scratch,
      VULKAN_HPP_NAMESPACE::DeviceSize              scratchOffset ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdBuildAccelerationStructureNV &&
                         "Function <vkCmdBuildAccelerationStructureNV> needs extension <VK_NV_ray_tracing> enabled!" );

      getDispatcher()->vkCmdBuildAccelerationStructureNV(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        reinterpret_cast<const VkAccelerationStructureInfoNV *>( &info ),
        static_cast<VkBuffer>( instanceData ),
        static_cast<VkDeviceSize>( instanceOffset ),
        static_cast<VkBool32>( update ),
        static_cast<VkAccelerationStructureNV>( dst ),
        static_cast<VkAccelerationStructureNV>( src ),
        static_cast<VkBuffer>( scratch ),
        static_cast<VkDeviceSize>( scratchOffset ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::copyAccelerationStructureNV(
      VULKAN_HPP_NAMESPACE::AccelerationStructureNV          dst,
      VULKAN_HPP_NAMESPACE::AccelerationStructureNV          src,
      VULKAN_HPP_NAMESPACE::CopyAccelerationStructureModeKHR mode ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdCopyAccelerationStructureNV &&
                         "Function <vkCmdCopyAccelerationStructureNV> needs extension <VK_NV_ray_tracing> enabled!" );

      getDispatcher()->vkCmdCopyAccelerationStructureNV( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                         static_cast<VkAccelerationStructureNV>( dst ),
                                                         static_cast<VkAccelerationStructureNV>( src ),
                                                         static_cast<VkCopyAccelerationStructureModeKHR>( mode ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::traceRaysNV( VULKAN_HPP_NAMESPACE::Buffer     raygenShaderBindingTableBuffer,
                                                       VULKAN_HPP_NAMESPACE::DeviceSize raygenShaderBindingOffset,
                                                       VULKAN_HPP_NAMESPACE::Buffer     missShaderBindingTableBuffer,
                                                       VULKAN_HPP_NAMESPACE::DeviceSize missShaderBindingOffset,
                                                       VULKAN_HPP_NAMESPACE::DeviceSize missShaderBindingStride,
                                                       VULKAN_HPP_NAMESPACE::Buffer     hitShaderBindingTableBuffer,
                                                       VULKAN_HPP_NAMESPACE::DeviceSize hitShaderBindingOffset,
                                                       VULKAN_HPP_NAMESPACE::DeviceSize hitShaderBindingStride,
                                                       VULKAN_HPP_NAMESPACE::Buffer callableShaderBindingTableBuffer,
                                                       VULKAN_HPP_NAMESPACE::DeviceSize callableShaderBindingOffset,
                                                       VULKAN_HPP_NAMESPACE::DeviceSize callableShaderBindingStride,
                                                       uint32_t                         width,
                                                       uint32_t                         height,
                                                       uint32_t depth ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdTraceRaysNV &&
                         "Function <vkCmdTraceRaysNV> needs extension <VK_NV_ray_tracing> enabled!" );

      getDispatcher()->vkCmdTraceRaysNV( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                         static_cast<VkBuffer>( raygenShaderBindingTableBuffer ),
                                         static_cast<VkDeviceSize>( raygenShaderBindingOffset ),
                                         static_cast<VkBuffer>( missShaderBindingTableBuffer ),
                                         static_cast<VkDeviceSize>( missShaderBindingOffset ),
                                         static_cast<VkDeviceSize>( missShaderBindingStride ),
                                         static_cast<VkBuffer>( hitShaderBindingTableBuffer ),
                                         static_cast<VkDeviceSize>( hitShaderBindingOffset ),
                                         static_cast<VkDeviceSize>( hitShaderBindingStride ),
                                         static_cast<VkBuffer>( callableShaderBindingTableBuffer ),
                                         static_cast<VkDeviceSize>( callableShaderBindingOffset ),
                                         static_cast<VkDeviceSize>( callableShaderBindingStride ),
                                         width,
                                         height,
                                         depth );
    }

    template <typename T>
    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<T>
                                           Pipeline::getRayTracingShaderGroupHandlesNV( uint32_t firstGroup, uint32_t groupCount, size_t dataSize ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetRayTracingShaderGroupHandlesNV &&
        "Function <vkGetRayTracingShaderGroupHandlesNV> needs extension <VK_NV_ray_tracing> enabled!" );

      VULKAN_HPP_ASSERT( dataSize % sizeof( T ) == 0 );
      std::vector<T> data( dataSize / sizeof( T ) );
      Result         result = static_cast<Result>(
        getDispatcher()->vkGetRayTracingShaderGroupHandlesNV( static_cast<VkDevice>( m_device ),
                                                              static_cast<VkPipeline>( m_pipeline ),
                                                              firstGroup,
                                                              groupCount,
                                                              data.size() * sizeof( T ),
                                                              reinterpret_cast<void *>( data.data() ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Pipeline::getRayTracingShaderGroupHandlesNV" );
      }
      return data;
    }

    template <typename T>
    VULKAN_HPP_NODISCARD T Pipeline::getRayTracingShaderGroupHandleNV( uint32_t firstGroup, uint32_t groupCount ) const
    {
      T      data;
      Result result = static_cast<Result>(
        getDispatcher()->vkGetRayTracingShaderGroupHandlesNV( static_cast<VkDevice>( m_device ),
                                                              static_cast<VkPipeline>( m_pipeline ),
                                                              firstGroup,
                                                              groupCount,
                                                              sizeof( T ),
                                                              reinterpret_cast<void *>( &data ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Pipeline::getRayTracingShaderGroupHandleNV" );
      }
      return data;
    }

    template <typename T>
    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<T> AccelerationStructureNV::getHandle( size_t dataSize ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkGetAccelerationStructureHandleNV &&
                         "Function <vkGetAccelerationStructureHandleNV> needs extension <VK_NV_ray_tracing> enabled!" );

      VULKAN_HPP_ASSERT( dataSize % sizeof( T ) == 0 );
      std::vector<T> data( dataSize / sizeof( T ) );
      Result         result = static_cast<Result>( getDispatcher()->vkGetAccelerationStructureHandleNV(
        static_cast<VkDevice>( m_device ),
        static_cast<VkAccelerationStructureNV>( m_accelerationStructureNV ),
        data.size() * sizeof( T ),
        reinterpret_cast<void *>( data.data() ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::AccelerationStructureNV::getHandle" );
      }
      return data;
    }

    template <typename T>
    VULKAN_HPP_NODISCARD T AccelerationStructureNV::getHandle() const
    {
      T      data;
      Result result = static_cast<Result>( getDispatcher()->vkGetAccelerationStructureHandleNV(
        static_cast<VkDevice>( m_device ),
        static_cast<VkAccelerationStructureNV>( m_accelerationStructureNV ),
        sizeof( T ),
        reinterpret_cast<void *>( &data ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::AccelerationStructureNV::getHandle" );
      }
      return data;
    }

    VULKAN_HPP_INLINE void CommandBuffer::writeAccelerationStructuresPropertiesNV(
      ArrayProxy<const VULKAN_HPP_NAMESPACE::AccelerationStructureNV> const & accelerationStructures,
      VULKAN_HPP_NAMESPACE::QueryType                                         queryType,
      VULKAN_HPP_NAMESPACE::QueryPool                                         queryPool,
      uint32_t                                                                firstQuery ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdWriteAccelerationStructuresPropertiesNV &&
        "Function <vkCmdWriteAccelerationStructuresPropertiesNV> needs extension <VK_NV_ray_tracing> enabled!" );

      getDispatcher()->vkCmdWriteAccelerationStructuresPropertiesNV(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        accelerationStructures.size(),
        reinterpret_cast<const VkAccelerationStructureNV *>( accelerationStructures.data() ),
        static_cast<VkQueryType>( queryType ),
        static_cast<VkQueryPool>( queryPool ),
        firstQuery );
    }

    VULKAN_HPP_INLINE void Pipeline::compileDeferredNV( uint32_t shader ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCompileDeferredNV &&
                         "Function <vkCompileDeferredNV> needs extension <VK_NV_ray_tracing> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCompileDeferredNV(
          static_cast<VkDevice>( m_device ), static_cast<VkPipeline>( m_pipeline ), shader ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Pipeline::compileDeferredNV" );
      }
    }

    //=== VK_KHR_maintenance3 ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::DescriptorSetLayoutSupport
                                           Device::getDescriptorSetLayoutSupportKHR( const DescriptorSetLayoutCreateInfo & createInfo ) const
      VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetDescriptorSetLayoutSupportKHR &&
        "Function <vkGetDescriptorSetLayoutSupportKHR> needs extension <VK_KHR_maintenance3> enabled!" );

      VULKAN_HPP_NAMESPACE::DescriptorSetLayoutSupport support;
      getDispatcher()->vkGetDescriptorSetLayoutSupportKHR(
        static_cast<VkDevice>( m_device ),
        reinterpret_cast<const VkDescriptorSetLayoutCreateInfo *>( &createInfo ),
        reinterpret_cast<VkDescriptorSetLayoutSupport *>( &support ) );
      return support;
    }

    template <typename X, typename Y, typename... Z>
    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE StructureChain<X, Y, Z...> Device::getDescriptorSetLayoutSupportKHR(
      const DescriptorSetLayoutCreateInfo & createInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetDescriptorSetLayoutSupportKHR &&
        "Function <vkGetDescriptorSetLayoutSupportKHR> needs extension <VK_KHR_maintenance3> enabled!" );

      StructureChain<X, Y, Z...>                         structureChain;
      VULKAN_HPP_NAMESPACE::DescriptorSetLayoutSupport & support =
        structureChain.template get<VULKAN_HPP_NAMESPACE::DescriptorSetLayoutSupport>();
      getDispatcher()->vkGetDescriptorSetLayoutSupportKHR(
        static_cast<VkDevice>( m_device ),
        reinterpret_cast<const VkDescriptorSetLayoutCreateInfo *>( &createInfo ),
        reinterpret_cast<VkDescriptorSetLayoutSupport *>( &support ) );
      return structureChain;
    }

    //=== VK_KHR_draw_indirect_count ===

    VULKAN_HPP_INLINE void CommandBuffer::drawIndirectCountKHR( VULKAN_HPP_NAMESPACE::Buffer     buffer,
                                                                VULKAN_HPP_NAMESPACE::DeviceSize offset,
                                                                VULKAN_HPP_NAMESPACE::Buffer     countBuffer,
                                                                VULKAN_HPP_NAMESPACE::DeviceSize countBufferOffset,
                                                                uint32_t                         maxDrawCount,
                                                                uint32_t stride ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdDrawIndirectCountKHR &&
                         "Function <vkCmdDrawIndirectCountKHR> needs extension <VK_KHR_draw_indirect_count> enabled!" );

      getDispatcher()->vkCmdDrawIndirectCountKHR( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                  static_cast<VkBuffer>( buffer ),
                                                  static_cast<VkDeviceSize>( offset ),
                                                  static_cast<VkBuffer>( countBuffer ),
                                                  static_cast<VkDeviceSize>( countBufferOffset ),
                                                  maxDrawCount,
                                                  stride );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::drawIndexedIndirectCountKHR( VULKAN_HPP_NAMESPACE::Buffer     buffer,
                                                  VULKAN_HPP_NAMESPACE::DeviceSize offset,
                                                  VULKAN_HPP_NAMESPACE::Buffer     countBuffer,
                                                  VULKAN_HPP_NAMESPACE::DeviceSize countBufferOffset,
                                                  uint32_t                         maxDrawCount,
                                                  uint32_t                         stride ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdDrawIndexedIndirectCountKHR &&
        "Function <vkCmdDrawIndexedIndirectCountKHR> needs extension <VK_KHR_draw_indirect_count> enabled!" );

      getDispatcher()->vkCmdDrawIndexedIndirectCountKHR( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                         static_cast<VkBuffer>( buffer ),
                                                         static_cast<VkDeviceSize>( offset ),
                                                         static_cast<VkBuffer>( countBuffer ),
                                                         static_cast<VkDeviceSize>( countBufferOffset ),
                                                         maxDrawCount,
                                                         stride );
    }

    //=== VK_EXT_external_memory_host ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::MemoryHostPointerPropertiesEXT
                                           Device::getMemoryHostPointerPropertiesEXT( VULKAN_HPP_NAMESPACE::ExternalMemoryHandleTypeFlagBits handleType,
                                                 const void * pHostPointer ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetMemoryHostPointerPropertiesEXT &&
        "Function <vkGetMemoryHostPointerPropertiesEXT> needs extension <VK_EXT_external_memory_host> enabled!" );

      VULKAN_HPP_NAMESPACE::MemoryHostPointerPropertiesEXT memoryHostPointerProperties;
      VULKAN_HPP_NAMESPACE::Result                         result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetMemoryHostPointerPropertiesEXT(
          static_cast<VkDevice>( m_device ),
          static_cast<VkExternalMemoryHandleTypeFlagBits>( handleType ),
          pHostPointer,
          reinterpret_cast<VkMemoryHostPointerPropertiesEXT *>( &memoryHostPointerProperties ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::getMemoryHostPointerPropertiesEXT" );
      }
      return memoryHostPointerProperties;
    }

    //=== VK_AMD_buffer_marker ===

    VULKAN_HPP_INLINE void
      CommandBuffer::writeBufferMarkerAMD( VULKAN_HPP_NAMESPACE::PipelineStageFlagBits pipelineStage,
                                           VULKAN_HPP_NAMESPACE::Buffer                dstBuffer,
                                           VULKAN_HPP_NAMESPACE::DeviceSize            dstOffset,
                                           uint32_t marker ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdWriteBufferMarkerAMD &&
                         "Function <vkCmdWriteBufferMarkerAMD> needs extension <VK_AMD_buffer_marker> enabled!" );

      getDispatcher()->vkCmdWriteBufferMarkerAMD( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                  static_cast<VkPipelineStageFlagBits>( pipelineStage ),
                                                  static_cast<VkBuffer>( dstBuffer ),
                                                  static_cast<VkDeviceSize>( dstOffset ),
                                                  marker );
    }

    //=== VK_EXT_calibrated_timestamps ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::TimeDomainEXT>
                                           PhysicalDevice::getCalibrateableTimeDomainsEXT() const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceCalibrateableTimeDomainsEXT &&
        "Function <vkGetPhysicalDeviceCalibrateableTimeDomainsEXT> needs extension <VK_EXT_calibrated_timestamps> enabled!" );

      std::vector<VULKAN_HPP_NAMESPACE::TimeDomainEXT> timeDomains;
      uint32_t                                         timeDomainCount;
      VULKAN_HPP_NAMESPACE::Result                     result;
      do
      {
        result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceCalibrateableTimeDomainsEXT(
            static_cast<VkPhysicalDevice>( m_physicalDevice ), &timeDomainCount, nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && timeDomainCount )
        {
          timeDomains.resize( timeDomainCount );
          result =
            static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceCalibrateableTimeDomainsEXT(
              static_cast<VkPhysicalDevice>( m_physicalDevice ),
              &timeDomainCount,
              reinterpret_cast<VkTimeDomainEXT *>( timeDomains.data() ) ) );
          VULKAN_HPP_ASSERT( timeDomainCount <= timeDomains.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( timeDomainCount < timeDomains.size() ) )
      {
        timeDomains.resize( timeDomainCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getCalibrateableTimeDomainsEXT" );
      }
      return timeDomains;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::pair<std::vector<uint64_t>, uint64_t>
                                           Device::getCalibratedTimestampsEXT(
        ArrayProxy<const VULKAN_HPP_NAMESPACE::CalibratedTimestampInfoEXT> const & timestampInfos ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetCalibratedTimestampsEXT &&
        "Function <vkGetCalibratedTimestampsEXT> needs extension <VK_EXT_calibrated_timestamps> enabled!" );

      std::pair<std::vector<uint64_t>, uint64_t> data(
        std::piecewise_construct, std::forward_as_tuple( timestampInfos.size() ), std::forward_as_tuple( 0 ) );
      std::vector<uint64_t> &      timestamps   = data.first;
      uint64_t &                   maxDeviation = data.second;
      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetCalibratedTimestampsEXT(
          static_cast<VkDevice>( m_device ),
          timestampInfos.size(),
          reinterpret_cast<const VkCalibratedTimestampInfoEXT *>( timestampInfos.data() ),
          timestamps.data(),
          &maxDeviation ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::getCalibratedTimestampsEXT" );
      }
      return data;
    }

    //=== VK_NV_mesh_shader ===

    VULKAN_HPP_INLINE void CommandBuffer::drawMeshTasksNV( uint32_t taskCount,
                                                           uint32_t firstTask ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdDrawMeshTasksNV &&
                         "Function <vkCmdDrawMeshTasksNV> needs extension <VK_NV_mesh_shader> enabled!" );

      getDispatcher()->vkCmdDrawMeshTasksNV( static_cast<VkCommandBuffer>( m_commandBuffer ), taskCount, firstTask );
    }

    VULKAN_HPP_INLINE void CommandBuffer::drawMeshTasksIndirectNV( VULKAN_HPP_NAMESPACE::Buffer     buffer,
                                                                   VULKAN_HPP_NAMESPACE::DeviceSize offset,
                                                                   uint32_t                         drawCount,
                                                                   uint32_t stride ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdDrawMeshTasksIndirectNV &&
                         "Function <vkCmdDrawMeshTasksIndirectNV> needs extension <VK_NV_mesh_shader> enabled!" );

      getDispatcher()->vkCmdDrawMeshTasksIndirectNV( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                     static_cast<VkBuffer>( buffer ),
                                                     static_cast<VkDeviceSize>( offset ),
                                                     drawCount,
                                                     stride );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::drawMeshTasksIndirectCountNV( VULKAN_HPP_NAMESPACE::Buffer     buffer,
                                                   VULKAN_HPP_NAMESPACE::DeviceSize offset,
                                                   VULKAN_HPP_NAMESPACE::Buffer     countBuffer,
                                                   VULKAN_HPP_NAMESPACE::DeviceSize countBufferOffset,
                                                   uint32_t                         maxDrawCount,
                                                   uint32_t                         stride ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdDrawMeshTasksIndirectCountNV &&
                         "Function <vkCmdDrawMeshTasksIndirectCountNV> needs extension <VK_NV_mesh_shader> enabled!" );

      getDispatcher()->vkCmdDrawMeshTasksIndirectCountNV( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                          static_cast<VkBuffer>( buffer ),
                                                          static_cast<VkDeviceSize>( offset ),
                                                          static_cast<VkBuffer>( countBuffer ),
                                                          static_cast<VkDeviceSize>( countBufferOffset ),
                                                          maxDrawCount,
                                                          stride );
    }

    //=== VK_NV_scissor_exclusive ===

    VULKAN_HPP_INLINE void CommandBuffer::setExclusiveScissorNV(
      uint32_t                                               firstExclusiveScissor,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::Rect2D> const & exclusiveScissors ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdSetExclusiveScissorNV &&
                         "Function <vkCmdSetExclusiveScissorNV> needs extension <VK_NV_scissor_exclusive> enabled!" );

      getDispatcher()->vkCmdSetExclusiveScissorNV( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                   firstExclusiveScissor,
                                                   exclusiveScissors.size(),
                                                   reinterpret_cast<const VkRect2D *>( exclusiveScissors.data() ) );
    }

    //=== VK_NV_device_diagnostic_checkpoints ===

    VULKAN_HPP_INLINE void CommandBuffer::setCheckpointNV( const void * pCheckpointMarker ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdSetCheckpointNV &&
        "Function <vkCmdSetCheckpointNV> needs extension <VK_NV_device_diagnostic_checkpoints> enabled!" );

      getDispatcher()->vkCmdSetCheckpointNV( static_cast<VkCommandBuffer>( m_commandBuffer ), pCheckpointMarker );
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::CheckpointDataNV>
                                           Queue::getCheckpointDataNV() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetQueueCheckpointDataNV &&
        "Function <vkGetQueueCheckpointDataNV> needs extension <VK_NV_device_diagnostic_checkpoints> enabled!" );

      uint32_t checkpointDataCount;
      getDispatcher()->vkGetQueueCheckpointDataNV( static_cast<VkQueue>( m_queue ), &checkpointDataCount, nullptr );
      std::vector<VULKAN_HPP_NAMESPACE::CheckpointDataNV> checkpointData( checkpointDataCount );
      getDispatcher()->vkGetQueueCheckpointDataNV( static_cast<VkQueue>( m_queue ),
                                                   &checkpointDataCount,
                                                   reinterpret_cast<VkCheckpointDataNV *>( checkpointData.data() ) );
      VULKAN_HPP_ASSERT( checkpointDataCount <= checkpointData.size() );
      return checkpointData;
    }

    //=== VK_KHR_timeline_semaphore ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE uint64_t Semaphore::getCounterValueKHR() const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetSemaphoreCounterValueKHR &&
        "Function <vkGetSemaphoreCounterValueKHR> needs extension <VK_KHR_timeline_semaphore> enabled!" );

      uint64_t                     value;
      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetSemaphoreCounterValueKHR(
          static_cast<VkDevice>( m_device ), static_cast<VkSemaphore>( m_semaphore ), &value ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Semaphore::getCounterValueKHR" );
      }
      return value;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::Result
                                           Device::waitSemaphoresKHR( const SemaphoreWaitInfo & waitInfo, uint64_t timeout ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkWaitSemaphoresKHR &&
                         "Function <vkWaitSemaphoresKHR> needs extension <VK_KHR_timeline_semaphore> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkWaitSemaphoresKHR(
          static_cast<VkDevice>( m_device ), reinterpret_cast<const VkSemaphoreWaitInfo *>( &waitInfo ), timeout ) );
      if ( ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eTimeout ) )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::waitSemaphoresKHR" );
      }
      return result;
    }

    VULKAN_HPP_INLINE void Device::signalSemaphoreKHR( const SemaphoreSignalInfo & signalInfo ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkSignalSemaphoreKHR &&
                         "Function <vkSignalSemaphoreKHR> needs extension <VK_KHR_timeline_semaphore> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkSignalSemaphoreKHR(
          static_cast<VkDevice>( m_device ), reinterpret_cast<const VkSemaphoreSignalInfo *>( &signalInfo ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::signalSemaphoreKHR" );
      }
    }

    //=== VK_INTEL_performance_query ===

    VULKAN_HPP_INLINE void
      Device::initializePerformanceApiINTEL( const InitializePerformanceApiInfoINTEL & initializeInfo ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkInitializePerformanceApiINTEL &&
        "Function <vkInitializePerformanceApiINTEL> needs extension <VK_INTEL_performance_query> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkInitializePerformanceApiINTEL(
          static_cast<VkDevice>( m_device ),
          reinterpret_cast<const VkInitializePerformanceApiInfoINTEL *>( &initializeInfo ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::initializePerformanceApiINTEL" );
      }
    }

    VULKAN_HPP_INLINE void Device::uninitializePerformanceApiINTEL() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkUninitializePerformanceApiINTEL &&
        "Function <vkUninitializePerformanceApiINTEL> needs extension <VK_INTEL_performance_query> enabled!" );

      getDispatcher()->vkUninitializePerformanceApiINTEL( static_cast<VkDevice>( m_device ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::setPerformanceMarkerINTEL( const PerformanceMarkerInfoINTEL & markerInfo ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdSetPerformanceMarkerINTEL &&
        "Function <vkCmdSetPerformanceMarkerINTEL> needs extension <VK_INTEL_performance_query> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCmdSetPerformanceMarkerINTEL(
          static_cast<VkCommandBuffer>( m_commandBuffer ),
          reinterpret_cast<const VkPerformanceMarkerInfoINTEL *>( &markerInfo ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::CommandBuffer::setPerformanceMarkerINTEL" );
      }
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::setPerformanceStreamMarkerINTEL( const PerformanceStreamMarkerInfoINTEL & markerInfo ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdSetPerformanceStreamMarkerINTEL &&
        "Function <vkCmdSetPerformanceStreamMarkerINTEL> needs extension <VK_INTEL_performance_query> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCmdSetPerformanceStreamMarkerINTEL(
          static_cast<VkCommandBuffer>( m_commandBuffer ),
          reinterpret_cast<const VkPerformanceStreamMarkerInfoINTEL *>( &markerInfo ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::CommandBuffer::setPerformanceStreamMarkerINTEL" );
      }
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::setPerformanceOverrideINTEL( const PerformanceOverrideInfoINTEL & overrideInfo ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdSetPerformanceOverrideINTEL &&
        "Function <vkCmdSetPerformanceOverrideINTEL> needs extension <VK_INTEL_performance_query> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkCmdSetPerformanceOverrideINTEL(
          static_cast<VkCommandBuffer>( m_commandBuffer ),
          reinterpret_cast<const VkPerformanceOverrideInfoINTEL *>( &overrideInfo ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::CommandBuffer::setPerformanceOverrideINTEL" );
      }
    }

    VULKAN_HPP_INLINE void
      Queue::setPerformanceConfigurationINTEL( VULKAN_HPP_NAMESPACE::PerformanceConfigurationINTEL configuration ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkQueueSetPerformanceConfigurationINTEL &&
        "Function <vkQueueSetPerformanceConfigurationINTEL> needs extension <VK_INTEL_performance_query> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkQueueSetPerformanceConfigurationINTEL(
          static_cast<VkQueue>( m_queue ), static_cast<VkPerformanceConfigurationINTEL>( configuration ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Queue::setPerformanceConfigurationINTEL" );
      }
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::PerformanceValueINTEL
                                           Device::getPerformanceParameterINTEL( VULKAN_HPP_NAMESPACE::PerformanceParameterTypeINTEL parameter ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPerformanceParameterINTEL &&
        "Function <vkGetPerformanceParameterINTEL> needs extension <VK_INTEL_performance_query> enabled!" );

      VULKAN_HPP_NAMESPACE::PerformanceValueINTEL value;
      VULKAN_HPP_NAMESPACE::Result                result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkGetPerformanceParameterINTEL( static_cast<VkDevice>( m_device ),
                                                         static_cast<VkPerformanceParameterTypeINTEL>( parameter ),
                                                         reinterpret_cast<VkPerformanceValueINTEL *>( &value ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::getPerformanceParameterINTEL" );
      }
      return value;
    }

    //=== VK_AMD_display_native_hdr ===

    VULKAN_HPP_INLINE void
      SwapchainKHR::setLocalDimmingAMD( VULKAN_HPP_NAMESPACE::Bool32 localDimmingEnable ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkSetLocalDimmingAMD &&
                         "Function <vkSetLocalDimmingAMD> needs extension <VK_AMD_display_native_hdr> enabled!" );

      getDispatcher()->vkSetLocalDimmingAMD( static_cast<VkDevice>( m_device ),
                                             static_cast<VkSwapchainKHR>( m_swapchainKHR ),
                                             static_cast<VkBool32>( localDimmingEnable ) );
    }

    //=== VK_KHR_fragment_shading_rate ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateKHR>
                                           PhysicalDevice::getFragmentShadingRatesKHR() const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceFragmentShadingRatesKHR &&
        "Function <vkGetPhysicalDeviceFragmentShadingRatesKHR> needs extension <VK_KHR_fragment_shading_rate> enabled!" );

      std::vector<VULKAN_HPP_NAMESPACE::PhysicalDeviceFragmentShadingRateKHR> fragmentShadingRates;
      uint32_t                                                                fragmentShadingRateCount;
      VULKAN_HPP_NAMESPACE::Result                                            result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceFragmentShadingRatesKHR(
          static_cast<VkPhysicalDevice>( m_physicalDevice ), &fragmentShadingRateCount, nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && fragmentShadingRateCount )
        {
          fragmentShadingRates.resize( fragmentShadingRateCount );
          result =
            static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceFragmentShadingRatesKHR(
              static_cast<VkPhysicalDevice>( m_physicalDevice ),
              &fragmentShadingRateCount,
              reinterpret_cast<VkPhysicalDeviceFragmentShadingRateKHR *>( fragmentShadingRates.data() ) ) );
          VULKAN_HPP_ASSERT( fragmentShadingRateCount <= fragmentShadingRates.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
           ( fragmentShadingRateCount < fragmentShadingRates.size() ) )
      {
        fragmentShadingRates.resize( fragmentShadingRateCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getFragmentShadingRatesKHR" );
      }
      return fragmentShadingRates;
    }

    VULKAN_HPP_INLINE void CommandBuffer::setFragmentShadingRateKHR(
      const Extent2D &                                             fragmentSize,
      const VULKAN_HPP_NAMESPACE::FragmentShadingRateCombinerOpKHR combinerOps[2] ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdSetFragmentShadingRateKHR &&
        "Function <vkCmdSetFragmentShadingRateKHR> needs extension <VK_KHR_fragment_shading_rate> enabled!" );

      getDispatcher()->vkCmdSetFragmentShadingRateKHR(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        reinterpret_cast<const VkExtent2D *>( &fragmentSize ),
        reinterpret_cast<const VkFragmentShadingRateCombinerOpKHR *>( combinerOps ) );
    }

    //=== VK_EXT_buffer_device_address ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::DeviceAddress
                                           Device::getBufferAddressEXT( const BufferDeviceAddressInfo & info ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetBufferDeviceAddressEXT &&
        "Function <vkGetBufferDeviceAddressEXT> needs extension <VK_EXT_buffer_device_address> enabled!" );

      return static_cast<VULKAN_HPP_NAMESPACE::DeviceAddress>( getDispatcher()->vkGetBufferDeviceAddressEXT(
        static_cast<VkDevice>( m_device ), reinterpret_cast<const VkBufferDeviceAddressInfo *>( &info ) ) );
    }

    //=== VK_EXT_tooling_info ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::PhysicalDeviceToolPropertiesEXT>
                                           PhysicalDevice::getToolPropertiesEXT() const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceToolPropertiesEXT &&
        "Function <vkGetPhysicalDeviceToolPropertiesEXT> needs extension <VK_EXT_tooling_info> enabled!" );

      std::vector<VULKAN_HPP_NAMESPACE::PhysicalDeviceToolPropertiesEXT> toolProperties;
      uint32_t                                                           toolCount;
      VULKAN_HPP_NAMESPACE::Result                                       result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceToolPropertiesEXT(
          static_cast<VkPhysicalDevice>( m_physicalDevice ), &toolCount, nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && toolCount )
        {
          toolProperties.resize( toolCount );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceToolPropertiesEXT(
            static_cast<VkPhysicalDevice>( m_physicalDevice ),
            &toolCount,
            reinterpret_cast<VkPhysicalDeviceToolPropertiesEXT *>( toolProperties.data() ) ) );
          VULKAN_HPP_ASSERT( toolCount <= toolProperties.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( toolCount < toolProperties.size() ) )
      {
        toolProperties.resize( toolCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getToolPropertiesEXT" );
      }
      return toolProperties;
    }

    //=== VK_KHR_present_wait ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::Result
                                           SwapchainKHR::waitForPresent( uint64_t presentId, uint64_t timeout ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkWaitForPresentKHR &&
                         "Function <vkWaitForPresentKHR> needs extension <VK_KHR_present_wait> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkWaitForPresentKHR(
          static_cast<VkDevice>( m_device ), static_cast<VkSwapchainKHR>( m_swapchainKHR ), presentId, timeout ) );
      if ( ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eTimeout ) )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::SwapchainKHR::waitForPresent" );
      }
      return result;
    }

    //=== VK_NV_cooperative_matrix ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::CooperativeMatrixPropertiesNV>
                                           PhysicalDevice::getCooperativeMatrixPropertiesNV() const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceCooperativeMatrixPropertiesNV &&
        "Function <vkGetPhysicalDeviceCooperativeMatrixPropertiesNV> needs extension <VK_NV_cooperative_matrix> enabled!" );

      std::vector<VULKAN_HPP_NAMESPACE::CooperativeMatrixPropertiesNV> properties;
      uint32_t                                                         propertyCount;
      VULKAN_HPP_NAMESPACE::Result                                     result;
      do
      {
        result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceCooperativeMatrixPropertiesNV(
            static_cast<VkPhysicalDevice>( m_physicalDevice ), &propertyCount, nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && propertyCount )
        {
          properties.resize( propertyCount );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
            getDispatcher()->vkGetPhysicalDeviceCooperativeMatrixPropertiesNV(
              static_cast<VkPhysicalDevice>( m_physicalDevice ),
              &propertyCount,
              reinterpret_cast<VkCooperativeMatrixPropertiesNV *>( properties.data() ) ) );
          VULKAN_HPP_ASSERT( propertyCount <= properties.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( propertyCount < properties.size() ) )
      {
        properties.resize( propertyCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result,
                              VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getCooperativeMatrixPropertiesNV" );
      }
      return properties;
    }

    //=== VK_NV_coverage_reduction_mode ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::FramebufferMixedSamplesCombinationNV>
                                           PhysicalDevice::getSupportedFramebufferMixedSamplesCombinationsNV() const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV &&
        "Function <vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV> needs extension <VK_NV_coverage_reduction_mode> enabled!" );

      std::vector<VULKAN_HPP_NAMESPACE::FramebufferMixedSamplesCombinationNV> combinations;
      uint32_t                                                                combinationCount;
      VULKAN_HPP_NAMESPACE::Result                                            result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
          getDispatcher()->vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV(
            static_cast<VkPhysicalDevice>( m_physicalDevice ), &combinationCount, nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && combinationCount )
        {
          combinations.resize( combinationCount );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
            getDispatcher()->vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV(
              static_cast<VkPhysicalDevice>( m_physicalDevice ),
              &combinationCount,
              reinterpret_cast<VkFramebufferMixedSamplesCombinationNV *>( combinations.data() ) ) );
          VULKAN_HPP_ASSERT( combinationCount <= combinations.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( combinationCount < combinations.size() ) )
      {
        combinations.resize( combinationCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException(
          result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getSupportedFramebufferMixedSamplesCombinationsNV" );
      }
      return combinations;
    }

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_EXT_full_screen_exclusive ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::PresentModeKHR>
                                           PhysicalDevice::getSurfacePresentModes2EXT( const PhysicalDeviceSurfaceInfo2KHR & surfaceInfo ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceSurfacePresentModes2EXT &&
        "Function <vkGetPhysicalDeviceSurfacePresentModes2EXT> needs extension <VK_EXT_full_screen_exclusive> enabled!" );

      std::vector<VULKAN_HPP_NAMESPACE::PresentModeKHR> presentModes;
      uint32_t                                          presentModeCount;
      VULKAN_HPP_NAMESPACE::Result                      result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceSurfacePresentModes2EXT(
          static_cast<VkPhysicalDevice>( m_physicalDevice ),
          reinterpret_cast<const VkPhysicalDeviceSurfaceInfo2KHR *>( &surfaceInfo ),
          &presentModeCount,
          nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && presentModeCount )
        {
          presentModes.resize( presentModeCount );
          result =
            static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPhysicalDeviceSurfacePresentModes2EXT(
              static_cast<VkPhysicalDevice>( m_physicalDevice ),
              reinterpret_cast<const VkPhysicalDeviceSurfaceInfo2KHR *>( &surfaceInfo ),
              &presentModeCount,
              reinterpret_cast<VkPresentModeKHR *>( presentModes.data() ) ) );
          VULKAN_HPP_ASSERT( presentModeCount <= presentModes.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( presentModeCount < presentModes.size() ) )
      {
        presentModes.resize( presentModeCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::getSurfacePresentModes2EXT" );
      }
      return presentModes;
    }

    VULKAN_HPP_INLINE void SwapchainKHR::acquireFullScreenExclusiveModeEXT() const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkAcquireFullScreenExclusiveModeEXT &&
        "Function <vkAcquireFullScreenExclusiveModeEXT> needs extension <VK_EXT_full_screen_exclusive> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkAcquireFullScreenExclusiveModeEXT(
          static_cast<VkDevice>( m_device ), static_cast<VkSwapchainKHR>( m_swapchainKHR ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::SwapchainKHR::acquireFullScreenExclusiveModeEXT" );
      }
    }

    VULKAN_HPP_INLINE void SwapchainKHR::releaseFullScreenExclusiveModeEXT() const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkReleaseFullScreenExclusiveModeEXT &&
        "Function <vkReleaseFullScreenExclusiveModeEXT> needs extension <VK_EXT_full_screen_exclusive> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkReleaseFullScreenExclusiveModeEXT(
          static_cast<VkDevice>( m_device ), static_cast<VkSwapchainKHR>( m_swapchainKHR ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::SwapchainKHR::releaseFullScreenExclusiveModeEXT" );
      }
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::DeviceGroupPresentModeFlagsKHR
                                           Device::getGroupSurfacePresentModes2EXT( const PhysicalDeviceSurfaceInfo2KHR & surfaceInfo ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetDeviceGroupSurfacePresentModes2EXT &&
        "Function <vkGetDeviceGroupSurfacePresentModes2EXT> needs extension <VK_EXT_full_screen_exclusive> enabled!" );

      VULKAN_HPP_NAMESPACE::DeviceGroupPresentModeFlagsKHR modes;
      VULKAN_HPP_NAMESPACE::Result                         result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetDeviceGroupSurfacePresentModes2EXT(
          static_cast<VkDevice>( m_device ),
          reinterpret_cast<const VkPhysicalDeviceSurfaceInfo2KHR *>( &surfaceInfo ),
          reinterpret_cast<VkDeviceGroupPresentModeFlagsKHR *>( &modes ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::getGroupSurfacePresentModes2EXT" );
      }
      return modes;
    }
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

    //=== VK_KHR_buffer_device_address ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::DeviceAddress
                                           Device::getBufferAddressKHR( const BufferDeviceAddressInfo & info ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetBufferDeviceAddressKHR &&
        "Function <vkGetBufferDeviceAddressKHR> needs extension <VK_KHR_buffer_device_address> enabled!" );

      return static_cast<VULKAN_HPP_NAMESPACE::DeviceAddress>( getDispatcher()->vkGetBufferDeviceAddressKHR(
        static_cast<VkDevice>( m_device ), reinterpret_cast<const VkBufferDeviceAddressInfo *>( &info ) ) );
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE uint64_t
                                           Device::getBufferOpaqueCaptureAddressKHR( const BufferDeviceAddressInfo & info ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetBufferOpaqueCaptureAddressKHR &&
        "Function <vkGetBufferOpaqueCaptureAddressKHR> needs extension <VK_KHR_buffer_device_address> enabled!" );

      return getDispatcher()->vkGetBufferOpaqueCaptureAddressKHR(
        static_cast<VkDevice>( m_device ), reinterpret_cast<const VkBufferDeviceAddressInfo *>( &info ) );
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE uint64_t Device::getMemoryOpaqueCaptureAddressKHR(
      const DeviceMemoryOpaqueCaptureAddressInfo & info ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetDeviceMemoryOpaqueCaptureAddressKHR &&
        "Function <vkGetDeviceMemoryOpaqueCaptureAddressKHR> needs extension <VK_KHR_buffer_device_address> enabled!" );

      return getDispatcher()->vkGetDeviceMemoryOpaqueCaptureAddressKHR(
        static_cast<VkDevice>( m_device ), reinterpret_cast<const VkDeviceMemoryOpaqueCaptureAddressInfo *>( &info ) );
    }

    //=== VK_EXT_line_rasterization ===

    VULKAN_HPP_INLINE void CommandBuffer::setLineStippleEXT( uint32_t lineStippleFactor,
                                                             uint16_t lineStipplePattern ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdSetLineStippleEXT &&
                         "Function <vkCmdSetLineStippleEXT> needs extension <VK_EXT_line_rasterization> enabled!" );

      getDispatcher()->vkCmdSetLineStippleEXT(
        static_cast<VkCommandBuffer>( m_commandBuffer ), lineStippleFactor, lineStipplePattern );
    }

    //=== VK_EXT_host_query_reset ===

    VULKAN_HPP_INLINE void QueryPool::resetEXT( uint32_t firstQuery, uint32_t queryCount ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkResetQueryPoolEXT &&
                         "Function <vkResetQueryPoolEXT> needs extension <VK_EXT_host_query_reset> enabled!" );

      getDispatcher()->vkResetQueryPoolEXT(
        static_cast<VkDevice>( m_device ), static_cast<VkQueryPool>( m_queryPool ), firstQuery, queryCount );
    }

    //=== VK_EXT_extended_dynamic_state ===

    VULKAN_HPP_INLINE void
      CommandBuffer::setCullModeEXT( VULKAN_HPP_NAMESPACE::CullModeFlags cullMode ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdSetCullModeEXT &&
                         "Function <vkCmdSetCullModeEXT> needs extension <VK_EXT_extended_dynamic_state> enabled!" );

      getDispatcher()->vkCmdSetCullModeEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                            static_cast<VkCullModeFlags>( cullMode ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::setFrontFaceEXT( VULKAN_HPP_NAMESPACE::FrontFace frontFace ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdSetFrontFaceEXT &&
                         "Function <vkCmdSetFrontFaceEXT> needs extension <VK_EXT_extended_dynamic_state> enabled!" );

      getDispatcher()->vkCmdSetFrontFaceEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                             static_cast<VkFrontFace>( frontFace ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::setPrimitiveTopologyEXT(
      VULKAN_HPP_NAMESPACE::PrimitiveTopology primitiveTopology ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdSetPrimitiveTopologyEXT &&
        "Function <vkCmdSetPrimitiveTopologyEXT> needs extension <VK_EXT_extended_dynamic_state> enabled!" );

      getDispatcher()->vkCmdSetPrimitiveTopologyEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                     static_cast<VkPrimitiveTopology>( primitiveTopology ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::setViewportWithCountEXT(
      ArrayProxy<const VULKAN_HPP_NAMESPACE::Viewport> const & viewports ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdSetViewportWithCountEXT &&
        "Function <vkCmdSetViewportWithCountEXT> needs extension <VK_EXT_extended_dynamic_state> enabled!" );

      getDispatcher()->vkCmdSetViewportWithCountEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                     viewports.size(),
                                                     reinterpret_cast<const VkViewport *>( viewports.data() ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::setScissorWithCountEXT(
      ArrayProxy<const VULKAN_HPP_NAMESPACE::Rect2D> const & scissors ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdSetScissorWithCountEXT &&
        "Function <vkCmdSetScissorWithCountEXT> needs extension <VK_EXT_extended_dynamic_state> enabled!" );

      getDispatcher()->vkCmdSetScissorWithCountEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                    scissors.size(),
                                                    reinterpret_cast<const VkRect2D *>( scissors.data() ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::bindVertexBuffers2EXT(
      uint32_t                                                   firstBinding,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::Buffer> const &     buffers,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::DeviceSize> const & offsets,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::DeviceSize> const & sizes,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::DeviceSize> const & strides ) const VULKAN_HPP_NOEXCEPT_WHEN_NO_EXCEPTIONS
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdBindVertexBuffers2EXT &&
        "Function <vkCmdBindVertexBuffers2EXT> needs extension <VK_EXT_extended_dynamic_state> enabled!" );

#  ifdef VULKAN_HPP_NO_EXCEPTIONS
      VULKAN_HPP_ASSERT( buffers.size() == offsets.size() );
      VULKAN_HPP_ASSERT( sizes.empty() || buffers.size() == sizes.size() );
      VULKAN_HPP_ASSERT( strides.empty() || buffers.size() == strides.size() );
#  else
      if ( buffers.size() != offsets.size() )
      {
        throw LogicError( VULKAN_HPP_NAMESPACE_STRING
                          "::CommandBuffer::bindVertexBuffers2EXT: buffers.size() != offsets.size()" );
      }
      if ( !sizes.empty() && buffers.size() != sizes.size() )
      {
        throw LogicError( VULKAN_HPP_NAMESPACE_STRING
                          "::CommandBuffer::bindVertexBuffers2EXT: buffers.size() != sizes.size()" );
      }
      if ( !strides.empty() && buffers.size() != strides.size() )
      {
        throw LogicError( VULKAN_HPP_NAMESPACE_STRING
                          "::CommandBuffer::bindVertexBuffers2EXT: buffers.size() != strides.size()" );
      }
#  endif /*VULKAN_HPP_NO_EXCEPTIONS*/

      getDispatcher()->vkCmdBindVertexBuffers2EXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                   firstBinding,
                                                   buffers.size(),
                                                   reinterpret_cast<const VkBuffer *>( buffers.data() ),
                                                   reinterpret_cast<const VkDeviceSize *>( offsets.data() ),
                                                   reinterpret_cast<const VkDeviceSize *>( sizes.data() ),
                                                   reinterpret_cast<const VkDeviceSize *>( strides.data() ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::setDepthTestEnableEXT( VULKAN_HPP_NAMESPACE::Bool32 depthTestEnable ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdSetDepthTestEnableEXT &&
        "Function <vkCmdSetDepthTestEnableEXT> needs extension <VK_EXT_extended_dynamic_state> enabled!" );

      getDispatcher()->vkCmdSetDepthTestEnableEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                   static_cast<VkBool32>( depthTestEnable ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::setDepthWriteEnableEXT( VULKAN_HPP_NAMESPACE::Bool32 depthWriteEnable ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdSetDepthWriteEnableEXT &&
        "Function <vkCmdSetDepthWriteEnableEXT> needs extension <VK_EXT_extended_dynamic_state> enabled!" );

      getDispatcher()->vkCmdSetDepthWriteEnableEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                    static_cast<VkBool32>( depthWriteEnable ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::setDepthCompareOpEXT( VULKAN_HPP_NAMESPACE::CompareOp depthCompareOp ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdSetDepthCompareOpEXT &&
        "Function <vkCmdSetDepthCompareOpEXT> needs extension <VK_EXT_extended_dynamic_state> enabled!" );

      getDispatcher()->vkCmdSetDepthCompareOpEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                  static_cast<VkCompareOp>( depthCompareOp ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::setDepthBoundsTestEnableEXT(
      VULKAN_HPP_NAMESPACE::Bool32 depthBoundsTestEnable ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdSetDepthBoundsTestEnableEXT &&
        "Function <vkCmdSetDepthBoundsTestEnableEXT> needs extension <VK_EXT_extended_dynamic_state> enabled!" );

      getDispatcher()->vkCmdSetDepthBoundsTestEnableEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                         static_cast<VkBool32>( depthBoundsTestEnable ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::setStencilTestEnableEXT( VULKAN_HPP_NAMESPACE::Bool32 stencilTestEnable ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdSetStencilTestEnableEXT &&
        "Function <vkCmdSetStencilTestEnableEXT> needs extension <VK_EXT_extended_dynamic_state> enabled!" );

      getDispatcher()->vkCmdSetStencilTestEnableEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                     static_cast<VkBool32>( stencilTestEnable ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::setStencilOpEXT( VULKAN_HPP_NAMESPACE::StencilFaceFlags faceMask,
                                      VULKAN_HPP_NAMESPACE::StencilOp        failOp,
                                      VULKAN_HPP_NAMESPACE::StencilOp        passOp,
                                      VULKAN_HPP_NAMESPACE::StencilOp        depthFailOp,
                                      VULKAN_HPP_NAMESPACE::CompareOp        compareOp ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdSetStencilOpEXT &&
                         "Function <vkCmdSetStencilOpEXT> needs extension <VK_EXT_extended_dynamic_state> enabled!" );

      getDispatcher()->vkCmdSetStencilOpEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                             static_cast<VkStencilFaceFlags>( faceMask ),
                                             static_cast<VkStencilOp>( failOp ),
                                             static_cast<VkStencilOp>( passOp ),
                                             static_cast<VkStencilOp>( depthFailOp ),
                                             static_cast<VkCompareOp>( compareOp ) );
    }

    //=== VK_KHR_deferred_host_operations ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE uint32_t DeferredOperationKHR::getMaxConcurrency() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetDeferredOperationMaxConcurrencyKHR &&
        "Function <vkGetDeferredOperationMaxConcurrencyKHR> needs extension <VK_KHR_deferred_host_operations> enabled!" );

      return getDispatcher()->vkGetDeferredOperationMaxConcurrencyKHR(
        static_cast<VkDevice>( m_device ), static_cast<VkDeferredOperationKHR>( m_deferredOperationKHR ) );
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::Result
                                           DeferredOperationKHR::getResult() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetDeferredOperationResultKHR &&
        "Function <vkGetDeferredOperationResultKHR> needs extension <VK_KHR_deferred_host_operations> enabled!" );

      return static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetDeferredOperationResultKHR(
        static_cast<VkDevice>( m_device ), static_cast<VkDeferredOperationKHR>( m_deferredOperationKHR ) ) );
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::Result DeferredOperationKHR::join() const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkDeferredOperationJoinKHR &&
        "Function <vkDeferredOperationJoinKHR> needs extension <VK_KHR_deferred_host_operations> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkDeferredOperationJoinKHR(
          static_cast<VkDevice>( m_device ), static_cast<VkDeferredOperationKHR>( m_deferredOperationKHR ) ) );
      if ( ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eThreadDoneKHR ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eThreadIdleKHR ) )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::DeferredOperationKHR::join" );
      }
      return result;
    }

    //=== VK_KHR_pipeline_executable_properties ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::PipelineExecutablePropertiesKHR>
                                           Device::getPipelineExecutablePropertiesKHR( const PipelineInfoKHR & pipelineInfo ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPipelineExecutablePropertiesKHR &&
        "Function <vkGetPipelineExecutablePropertiesKHR> needs extension <VK_KHR_pipeline_executable_properties> enabled!" );

      std::vector<VULKAN_HPP_NAMESPACE::PipelineExecutablePropertiesKHR> properties;
      uint32_t                                                           executableCount;
      VULKAN_HPP_NAMESPACE::Result                                       result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPipelineExecutablePropertiesKHR(
          static_cast<VkDevice>( m_device ),
          reinterpret_cast<const VkPipelineInfoKHR *>( &pipelineInfo ),
          &executableCount,
          nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && executableCount )
        {
          properties.resize( executableCount );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPipelineExecutablePropertiesKHR(
            static_cast<VkDevice>( m_device ),
            reinterpret_cast<const VkPipelineInfoKHR *>( &pipelineInfo ),
            &executableCount,
            reinterpret_cast<VkPipelineExecutablePropertiesKHR *>( properties.data() ) ) );
          VULKAN_HPP_ASSERT( executableCount <= properties.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( executableCount < properties.size() ) )
      {
        properties.resize( executableCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::getPipelineExecutablePropertiesKHR" );
      }
      return properties;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::PipelineExecutableStatisticKHR>
                                           Device::getPipelineExecutableStatisticsKHR( const PipelineExecutableInfoKHR & executableInfo ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPipelineExecutableStatisticsKHR &&
        "Function <vkGetPipelineExecutableStatisticsKHR> needs extension <VK_KHR_pipeline_executable_properties> enabled!" );

      std::vector<VULKAN_HPP_NAMESPACE::PipelineExecutableStatisticKHR> statistics;
      uint32_t                                                          statisticCount;
      VULKAN_HPP_NAMESPACE::Result                                      result;
      do
      {
        result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPipelineExecutableStatisticsKHR(
          static_cast<VkDevice>( m_device ),
          reinterpret_cast<const VkPipelineExecutableInfoKHR *>( &executableInfo ),
          &statisticCount,
          nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && statisticCount )
        {
          statistics.resize( statisticCount );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPipelineExecutableStatisticsKHR(
            static_cast<VkDevice>( m_device ),
            reinterpret_cast<const VkPipelineExecutableInfoKHR *>( &executableInfo ),
            &statisticCount,
            reinterpret_cast<VkPipelineExecutableStatisticKHR *>( statistics.data() ) ) );
          VULKAN_HPP_ASSERT( statisticCount <= statistics.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && ( statisticCount < statistics.size() ) )
      {
        statistics.resize( statisticCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::getPipelineExecutableStatisticsKHR" );
      }
      return statistics;
    }

    VULKAN_HPP_NODISCARD
      VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::PipelineExecutableInternalRepresentationKHR>
                        Device::getPipelineExecutableInternalRepresentationsKHR( const PipelineExecutableInfoKHR & executableInfo ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPipelineExecutableInternalRepresentationsKHR &&
        "Function <vkGetPipelineExecutableInternalRepresentationsKHR> needs extension <VK_KHR_pipeline_executable_properties> enabled!" );

      std::vector<VULKAN_HPP_NAMESPACE::PipelineExecutableInternalRepresentationKHR> internalRepresentations;
      uint32_t                                                                       internalRepresentationCount;
      VULKAN_HPP_NAMESPACE::Result                                                   result;
      do
      {
        result =
          static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetPipelineExecutableInternalRepresentationsKHR(
            static_cast<VkDevice>( m_device ),
            reinterpret_cast<const VkPipelineExecutableInfoKHR *>( &executableInfo ),
            &internalRepresentationCount,
            nullptr ) );
        if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) && internalRepresentationCount )
        {
          internalRepresentations.resize( internalRepresentationCount );
          result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
            getDispatcher()->vkGetPipelineExecutableInternalRepresentationsKHR(
              static_cast<VkDevice>( m_device ),
              reinterpret_cast<const VkPipelineExecutableInfoKHR *>( &executableInfo ),
              &internalRepresentationCount,
              reinterpret_cast<VkPipelineExecutableInternalRepresentationKHR *>( internalRepresentations.data() ) ) );
          VULKAN_HPP_ASSERT( internalRepresentationCount <= internalRepresentations.size() );
        }
      } while ( result == VULKAN_HPP_NAMESPACE::Result::eIncomplete );
      if ( ( result == VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
           ( internalRepresentationCount < internalRepresentations.size() ) )
      {
        internalRepresentations.resize( internalRepresentationCount );
      }
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result,
                              VULKAN_HPP_NAMESPACE_STRING "::Device::getPipelineExecutableInternalRepresentationsKHR" );
      }
      return internalRepresentations;
    }

    //=== VK_NV_device_generated_commands ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::MemoryRequirements2
                                           Device::getGeneratedCommandsMemoryRequirementsNV( const GeneratedCommandsMemoryRequirementsInfoNV & info ) const
      VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetGeneratedCommandsMemoryRequirementsNV &&
        "Function <vkGetGeneratedCommandsMemoryRequirementsNV> needs extension <VK_NV_device_generated_commands> enabled!" );

      VULKAN_HPP_NAMESPACE::MemoryRequirements2 memoryRequirements;
      getDispatcher()->vkGetGeneratedCommandsMemoryRequirementsNV(
        static_cast<VkDevice>( m_device ),
        reinterpret_cast<const VkGeneratedCommandsMemoryRequirementsInfoNV *>( &info ),
        reinterpret_cast<VkMemoryRequirements2 *>( &memoryRequirements ) );
      return memoryRequirements;
    }

    template <typename X, typename Y, typename... Z>
    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE StructureChain<X, Y, Z...> Device::getGeneratedCommandsMemoryRequirementsNV(
      const GeneratedCommandsMemoryRequirementsInfoNV & info ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetGeneratedCommandsMemoryRequirementsNV &&
        "Function <vkGetGeneratedCommandsMemoryRequirementsNV> needs extension <VK_NV_device_generated_commands> enabled!" );

      StructureChain<X, Y, Z...>                  structureChain;
      VULKAN_HPP_NAMESPACE::MemoryRequirements2 & memoryRequirements =
        structureChain.template get<VULKAN_HPP_NAMESPACE::MemoryRequirements2>();
      getDispatcher()->vkGetGeneratedCommandsMemoryRequirementsNV(
        static_cast<VkDevice>( m_device ),
        reinterpret_cast<const VkGeneratedCommandsMemoryRequirementsInfoNV *>( &info ),
        reinterpret_cast<VkMemoryRequirements2 *>( &memoryRequirements ) );
      return structureChain;
    }

    VULKAN_HPP_INLINE void CommandBuffer::preprocessGeneratedCommandsNV(
      const GeneratedCommandsInfoNV & generatedCommandsInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdPreprocessGeneratedCommandsNV &&
        "Function <vkCmdPreprocessGeneratedCommandsNV> needs extension <VK_NV_device_generated_commands> enabled!" );

      getDispatcher()->vkCmdPreprocessGeneratedCommandsNV(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        reinterpret_cast<const VkGeneratedCommandsInfoNV *>( &generatedCommandsInfo ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::executeGeneratedCommandsNV(
      VULKAN_HPP_NAMESPACE::Bool32    isPreprocessed,
      const GeneratedCommandsInfoNV & generatedCommandsInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdExecuteGeneratedCommandsNV &&
        "Function <vkCmdExecuteGeneratedCommandsNV> needs extension <VK_NV_device_generated_commands> enabled!" );

      getDispatcher()->vkCmdExecuteGeneratedCommandsNV(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        static_cast<VkBool32>( isPreprocessed ),
        reinterpret_cast<const VkGeneratedCommandsInfoNV *>( &generatedCommandsInfo ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::bindPipelineShaderGroupNV( VULKAN_HPP_NAMESPACE::PipelineBindPoint pipelineBindPoint,
                                                VULKAN_HPP_NAMESPACE::Pipeline          pipeline,
                                                uint32_t groupIndex ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdBindPipelineShaderGroupNV &&
        "Function <vkCmdBindPipelineShaderGroupNV> needs extension <VK_NV_device_generated_commands> enabled!" );

      getDispatcher()->vkCmdBindPipelineShaderGroupNV( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                       static_cast<VkPipelineBindPoint>( pipelineBindPoint ),
                                                       static_cast<VkPipeline>( pipeline ),
                                                       groupIndex );
    }

    //=== VK_EXT_acquire_drm_display ===

    VULKAN_HPP_INLINE void PhysicalDevice::acquireDrmDisplayEXT( int32_t                          drmFd,
                                                                 VULKAN_HPP_NAMESPACE::DisplayKHR display ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkAcquireDrmDisplayEXT &&
                         "Function <vkAcquireDrmDisplayEXT> needs extension <VK_EXT_acquire_drm_display> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkAcquireDrmDisplayEXT(
          static_cast<VkPhysicalDevice>( m_physicalDevice ), drmFd, static_cast<VkDisplayKHR>( display ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::PhysicalDevice::acquireDrmDisplayEXT" );
      }
    }

    //=== VK_EXT_private_data ===

    VULKAN_HPP_INLINE void Device::setPrivateDataEXT( VULKAN_HPP_NAMESPACE::ObjectType         objectType_,
                                                      uint64_t                                 objectHandle,
                                                      VULKAN_HPP_NAMESPACE::PrivateDataSlotEXT privateDataSlot,
                                                      uint64_t                                 data ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkSetPrivateDataEXT &&
                         "Function <vkSetPrivateDataEXT> needs extension <VK_EXT_private_data> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkSetPrivateDataEXT( static_cast<VkDevice>( m_device ),
                                              static_cast<VkObjectType>( objectType_ ),
                                              objectHandle,
                                              static_cast<VkPrivateDataSlotEXT>( privateDataSlot ),
                                              data ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::setPrivateDataEXT" );
      }
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE uint64_t
                                           Device::getPrivateDataEXT( VULKAN_HPP_NAMESPACE::ObjectType         objectType_,
                                 uint64_t                                 objectHandle,
                                 VULKAN_HPP_NAMESPACE::PrivateDataSlotEXT privateDataSlot ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkGetPrivateDataEXT &&
                         "Function <vkGetPrivateDataEXT> needs extension <VK_EXT_private_data> enabled!" );

      uint64_t data;
      getDispatcher()->vkGetPrivateDataEXT( static_cast<VkDevice>( m_device ),
                                            static_cast<VkObjectType>( objectType_ ),
                                            objectHandle,
                                            static_cast<VkPrivateDataSlotEXT>( privateDataSlot ),
                                            &data );
      return data;
    }

#  if defined( VK_ENABLE_BETA_EXTENSIONS )
    //=== VK_KHR_video_encode_queue ===

    VULKAN_HPP_INLINE void
      CommandBuffer::encodeVideoKHR( const VideoEncodeInfoKHR & encodeInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdEncodeVideoKHR &&
                         "Function <vkCmdEncodeVideoKHR> needs extension <VK_KHR_video_encode_queue> enabled!" );

      getDispatcher()->vkCmdEncodeVideoKHR( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                            reinterpret_cast<const VkVideoEncodeInfoKHR *>( &encodeInfo ) );
    }
#  endif /*VK_ENABLE_BETA_EXTENSIONS*/

    //=== VK_KHR_synchronization2 ===

    VULKAN_HPP_INLINE void
      CommandBuffer::setEvent2KHR( VULKAN_HPP_NAMESPACE::Event event,
                                   const DependencyInfoKHR &   dependencyInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdSetEvent2KHR &&
                         "Function <vkCmdSetEvent2KHR> needs extension <VK_KHR_synchronization2> enabled!" );

      getDispatcher()->vkCmdSetEvent2KHR( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                          static_cast<VkEvent>( event ),
                                          reinterpret_cast<const VkDependencyInfoKHR *>( &dependencyInfo ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::resetEvent2KHR( VULKAN_HPP_NAMESPACE::Event                  event,
                                     VULKAN_HPP_NAMESPACE::PipelineStageFlags2KHR stageMask ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdResetEvent2KHR &&
                         "Function <vkCmdResetEvent2KHR> needs extension <VK_KHR_synchronization2> enabled!" );

      getDispatcher()->vkCmdResetEvent2KHR( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                            static_cast<VkEvent>( event ),
                                            static_cast<VkPipelineStageFlags2KHR>( stageMask ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::waitEvents2KHR( ArrayProxy<const VULKAN_HPP_NAMESPACE::Event> const &             events,
                                     ArrayProxy<const VULKAN_HPP_NAMESPACE::DependencyInfoKHR> const & dependencyInfos )
        const VULKAN_HPP_NOEXCEPT_WHEN_NO_EXCEPTIONS
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdWaitEvents2KHR &&
                         "Function <vkCmdWaitEvents2KHR> needs extension <VK_KHR_synchronization2> enabled!" );

#  ifdef VULKAN_HPP_NO_EXCEPTIONS
      VULKAN_HPP_ASSERT( events.size() == dependencyInfos.size() );
#  else
      if ( events.size() != dependencyInfos.size() )
      {
        throw LogicError( VULKAN_HPP_NAMESPACE_STRING
                          "::CommandBuffer::waitEvents2KHR: events.size() != dependencyInfos.size()" );
      }
#  endif /*VULKAN_HPP_NO_EXCEPTIONS*/

      getDispatcher()->vkCmdWaitEvents2KHR( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                            events.size(),
                                            reinterpret_cast<const VkEvent *>( events.data() ),
                                            reinterpret_cast<const VkDependencyInfoKHR *>( dependencyInfos.data() ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::pipelineBarrier2KHR( const DependencyInfoKHR & dependencyInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdPipelineBarrier2KHR &&
                         "Function <vkCmdPipelineBarrier2KHR> needs extension <VK_KHR_synchronization2> enabled!" );

      getDispatcher()->vkCmdPipelineBarrier2KHR( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                 reinterpret_cast<const VkDependencyInfoKHR *>( &dependencyInfo ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::writeTimestamp2KHR( VULKAN_HPP_NAMESPACE::PipelineStageFlags2KHR stage,
                                                              VULKAN_HPP_NAMESPACE::QueryPool              queryPool,
                                                              uint32_t query ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdWriteTimestamp2KHR &&
                         "Function <vkCmdWriteTimestamp2KHR> needs extension <VK_KHR_synchronization2> enabled!" );

      getDispatcher()->vkCmdWriteTimestamp2KHR( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                static_cast<VkPipelineStageFlags2KHR>( stage ),
                                                static_cast<VkQueryPool>( queryPool ),
                                                query );
    }

    VULKAN_HPP_INLINE void Queue::submit2KHR( ArrayProxy<const VULKAN_HPP_NAMESPACE::SubmitInfo2KHR> const & submits,
                                              VULKAN_HPP_NAMESPACE::Fence fence ) const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkQueueSubmit2KHR &&
                         "Function <vkQueueSubmit2KHR> needs extension <VK_KHR_synchronization2> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result = static_cast<VULKAN_HPP_NAMESPACE::Result>(
        getDispatcher()->vkQueueSubmit2KHR( static_cast<VkQueue>( m_queue ),
                                            submits.size(),
                                            reinterpret_cast<const VkSubmitInfo2KHR *>( submits.data() ),
                                            static_cast<VkFence>( fence ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Queue::submit2KHR" );
      }
    }

    VULKAN_HPP_INLINE void CommandBuffer::writeBufferMarker2AMD( VULKAN_HPP_NAMESPACE::PipelineStageFlags2KHR stage,
                                                                 VULKAN_HPP_NAMESPACE::Buffer                 dstBuffer,
                                                                 VULKAN_HPP_NAMESPACE::DeviceSize             dstOffset,
                                                                 uint32_t marker ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdWriteBufferMarker2AMD &&
                         "Function <vkCmdWriteBufferMarker2AMD> needs extension <VK_KHR_synchronization2> enabled!" );

      getDispatcher()->vkCmdWriteBufferMarker2AMD( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                   static_cast<VkPipelineStageFlags2KHR>( stage ),
                                                   static_cast<VkBuffer>( dstBuffer ),
                                                   static_cast<VkDeviceSize>( dstOffset ),
                                                   marker );
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<VULKAN_HPP_NAMESPACE::CheckpointData2NV>
                                           Queue::getCheckpointData2NV() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkGetQueueCheckpointData2NV &&
                         "Function <vkGetQueueCheckpointData2NV> needs extension <VK_KHR_synchronization2> enabled!" );

      uint32_t checkpointDataCount;
      getDispatcher()->vkGetQueueCheckpointData2NV( static_cast<VkQueue>( m_queue ), &checkpointDataCount, nullptr );
      std::vector<VULKAN_HPP_NAMESPACE::CheckpointData2NV> checkpointData( checkpointDataCount );
      getDispatcher()->vkGetQueueCheckpointData2NV( static_cast<VkQueue>( m_queue ),
                                                    &checkpointDataCount,
                                                    reinterpret_cast<VkCheckpointData2NV *>( checkpointData.data() ) );
      VULKAN_HPP_ASSERT( checkpointDataCount <= checkpointData.size() );
      return checkpointData;
    }

    //=== VK_NV_fragment_shading_rate_enums ===

    VULKAN_HPP_INLINE void CommandBuffer::setFragmentShadingRateEnumNV(
      VULKAN_HPP_NAMESPACE::FragmentShadingRateNV                  shadingRate,
      const VULKAN_HPP_NAMESPACE::FragmentShadingRateCombinerOpKHR combinerOps[2] ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdSetFragmentShadingRateEnumNV &&
        "Function <vkCmdSetFragmentShadingRateEnumNV> needs extension <VK_NV_fragment_shading_rate_enums> enabled!" );

      getDispatcher()->vkCmdSetFragmentShadingRateEnumNV(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        static_cast<VkFragmentShadingRateNV>( shadingRate ),
        reinterpret_cast<const VkFragmentShadingRateCombinerOpKHR *>( combinerOps ) );
    }

    //=== VK_KHR_copy_commands2 ===

    VULKAN_HPP_INLINE void
      CommandBuffer::copyBuffer2KHR( const CopyBufferInfo2KHR & copyBufferInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdCopyBuffer2KHR &&
                         "Function <vkCmdCopyBuffer2KHR> needs extension <VK_KHR_copy_commands2> enabled!" );

      getDispatcher()->vkCmdCopyBuffer2KHR( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                            reinterpret_cast<const VkCopyBufferInfo2KHR *>( &copyBufferInfo ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::copyImage2KHR( const CopyImageInfo2KHR & copyImageInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdCopyImage2KHR &&
                         "Function <vkCmdCopyImage2KHR> needs extension <VK_KHR_copy_commands2> enabled!" );

      getDispatcher()->vkCmdCopyImage2KHR( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                           reinterpret_cast<const VkCopyImageInfo2KHR *>( &copyImageInfo ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::copyBufferToImage2KHR(
      const CopyBufferToImageInfo2KHR & copyBufferToImageInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdCopyBufferToImage2KHR &&
                         "Function <vkCmdCopyBufferToImage2KHR> needs extension <VK_KHR_copy_commands2> enabled!" );

      getDispatcher()->vkCmdCopyBufferToImage2KHR(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        reinterpret_cast<const VkCopyBufferToImageInfo2KHR *>( &copyBufferToImageInfo ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::copyImageToBuffer2KHR(
      const CopyImageToBufferInfo2KHR & copyImageToBufferInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdCopyImageToBuffer2KHR &&
                         "Function <vkCmdCopyImageToBuffer2KHR> needs extension <VK_KHR_copy_commands2> enabled!" );

      getDispatcher()->vkCmdCopyImageToBuffer2KHR(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        reinterpret_cast<const VkCopyImageToBufferInfo2KHR *>( &copyImageToBufferInfo ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::blitImage2KHR( const BlitImageInfo2KHR & blitImageInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdBlitImage2KHR &&
                         "Function <vkCmdBlitImage2KHR> needs extension <VK_KHR_copy_commands2> enabled!" );

      getDispatcher()->vkCmdBlitImage2KHR( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                           reinterpret_cast<const VkBlitImageInfo2KHR *>( &blitImageInfo ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::resolveImage2KHR( const ResolveImageInfo2KHR & resolveImageInfo ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdResolveImage2KHR &&
                         "Function <vkCmdResolveImage2KHR> needs extension <VK_KHR_copy_commands2> enabled!" );

      getDispatcher()->vkCmdResolveImage2KHR( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                              reinterpret_cast<const VkResolveImageInfo2KHR *>( &resolveImageInfo ) );
    }

#  if defined( VK_USE_PLATFORM_WIN32_KHR )
    //=== VK_NV_acquire_winrt_display ===

    VULKAN_HPP_INLINE void DisplayKHR::acquireWinrtNV() const
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkAcquireWinrtDisplayNV &&
                         "Function <vkAcquireWinrtDisplayNV> needs extension <VK_NV_acquire_winrt_display> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkAcquireWinrtDisplayNV(
          static_cast<VkPhysicalDevice>( m_physicalDevice ), static_cast<VkDisplayKHR>( m_displayKHR ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::DisplayKHR::acquireWinrtNV" );
      }
    }
#  endif /*VK_USE_PLATFORM_WIN32_KHR*/

#  if defined( VK_USE_PLATFORM_DIRECTFB_EXT )
    //=== VK_EXT_directfb_surface ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::Bool32
                                           PhysicalDevice::getDirectFBPresentationSupportEXT( uint32_t    queueFamilyIndex,
                                                         IDirectFB & dfb ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceDirectFBPresentationSupportEXT &&
        "Function <vkGetPhysicalDeviceDirectFBPresentationSupportEXT> needs extension <VK_EXT_directfb_surface> enabled!" );

      return static_cast<VULKAN_HPP_NAMESPACE::Bool32>(
        getDispatcher()->vkGetPhysicalDeviceDirectFBPresentationSupportEXT(
          static_cast<VkPhysicalDevice>( m_physicalDevice ), queueFamilyIndex, &dfb ) );
    }
#  endif /*VK_USE_PLATFORM_DIRECTFB_EXT*/

    //=== VK_KHR_ray_tracing_pipeline ===

    VULKAN_HPP_INLINE void
      CommandBuffer::traceRaysKHR( const StridedDeviceAddressRegionKHR & raygenShaderBindingTable,
                                   const StridedDeviceAddressRegionKHR & missShaderBindingTable,
                                   const StridedDeviceAddressRegionKHR & hitShaderBindingTable,
                                   const StridedDeviceAddressRegionKHR & callableShaderBindingTable,
                                   uint32_t                              width,
                                   uint32_t                              height,
                                   uint32_t                              depth ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdTraceRaysKHR &&
                         "Function <vkCmdTraceRaysKHR> needs extension <VK_KHR_ray_tracing_pipeline> enabled!" );

      getDispatcher()->vkCmdTraceRaysKHR(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR *>( &raygenShaderBindingTable ),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR *>( &missShaderBindingTable ),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR *>( &hitShaderBindingTable ),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR *>( &callableShaderBindingTable ),
        width,
        height,
        depth );
    }

    template <typename T>
    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<T>
                                           Pipeline::getRayTracingShaderGroupHandlesKHR( uint32_t firstGroup, uint32_t groupCount, size_t dataSize ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetRayTracingShaderGroupHandlesKHR &&
        "Function <vkGetRayTracingShaderGroupHandlesKHR> needs extension <VK_KHR_ray_tracing_pipeline> enabled!" );

      VULKAN_HPP_ASSERT( dataSize % sizeof( T ) == 0 );
      std::vector<T> data( dataSize / sizeof( T ) );
      Result         result = static_cast<Result>(
        getDispatcher()->vkGetRayTracingShaderGroupHandlesKHR( static_cast<VkDevice>( m_device ),
                                                               static_cast<VkPipeline>( m_pipeline ),
                                                               firstGroup,
                                                               groupCount,
                                                               data.size() * sizeof( T ),
                                                               reinterpret_cast<void *>( data.data() ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Pipeline::getRayTracingShaderGroupHandlesKHR" );
      }
      return data;
    }

    template <typename T>
    VULKAN_HPP_NODISCARD T Pipeline::getRayTracingShaderGroupHandleKHR( uint32_t firstGroup, uint32_t groupCount ) const
    {
      T      data;
      Result result = static_cast<Result>(
        getDispatcher()->vkGetRayTracingShaderGroupHandlesKHR( static_cast<VkDevice>( m_device ),
                                                               static_cast<VkPipeline>( m_pipeline ),
                                                               firstGroup,
                                                               groupCount,
                                                               sizeof( T ),
                                                               reinterpret_cast<void *>( &data ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Pipeline::getRayTracingShaderGroupHandleKHR" );
      }
      return data;
    }

    template <typename T>
    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::vector<T> Pipeline::getRayTracingCaptureReplayShaderGroupHandlesKHR(
      uint32_t firstGroup, uint32_t groupCount, size_t dataSize ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetRayTracingCaptureReplayShaderGroupHandlesKHR &&
        "Function <vkGetRayTracingCaptureReplayShaderGroupHandlesKHR> needs extension <VK_KHR_ray_tracing_pipeline> enabled!" );

      VULKAN_HPP_ASSERT( dataSize % sizeof( T ) == 0 );
      std::vector<T> data( dataSize / sizeof( T ) );
      Result         result = static_cast<Result>(
        getDispatcher()->vkGetRayTracingCaptureReplayShaderGroupHandlesKHR( static_cast<VkDevice>( m_device ),
                                                                            static_cast<VkPipeline>( m_pipeline ),
                                                                            firstGroup,
                                                                            groupCount,
                                                                            data.size() * sizeof( T ),
                                                                            reinterpret_cast<void *>( data.data() ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException(
          result, VULKAN_HPP_NAMESPACE_STRING "::Pipeline::getRayTracingCaptureReplayShaderGroupHandlesKHR" );
      }
      return data;
    }

    template <typename T>
    VULKAN_HPP_NODISCARD T Pipeline::getRayTracingCaptureReplayShaderGroupHandleKHR( uint32_t firstGroup,
                                                                                     uint32_t groupCount ) const
    {
      T      data;
      Result result = static_cast<Result>(
        getDispatcher()->vkGetRayTracingCaptureReplayShaderGroupHandlesKHR( static_cast<VkDevice>( m_device ),
                                                                            static_cast<VkPipeline>( m_pipeline ),
                                                                            firstGroup,
                                                                            groupCount,
                                                                            sizeof( T ),
                                                                            reinterpret_cast<void *>( &data ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException(
          result, VULKAN_HPP_NAMESPACE_STRING "::Pipeline::getRayTracingCaptureReplayShaderGroupHandleKHR" );
      }
      return data;
    }

    VULKAN_HPP_INLINE void CommandBuffer::traceRaysIndirectKHR(
      const StridedDeviceAddressRegionKHR & raygenShaderBindingTable,
      const StridedDeviceAddressRegionKHR & missShaderBindingTable,
      const StridedDeviceAddressRegionKHR & hitShaderBindingTable,
      const StridedDeviceAddressRegionKHR & callableShaderBindingTable,
      VULKAN_HPP_NAMESPACE::DeviceAddress   indirectDeviceAddress ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdTraceRaysIndirectKHR &&
        "Function <vkCmdTraceRaysIndirectKHR> needs extension <VK_KHR_ray_tracing_pipeline> enabled!" );

      getDispatcher()->vkCmdTraceRaysIndirectKHR(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR *>( &raygenShaderBindingTable ),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR *>( &missShaderBindingTable ),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR *>( &hitShaderBindingTable ),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR *>( &callableShaderBindingTable ),
        static_cast<VkDeviceAddress>( indirectDeviceAddress ) );
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::DeviceSize
                                           Pipeline::getRayTracingShaderGroupStackSizeKHR(
        uint32_t group, VULKAN_HPP_NAMESPACE::ShaderGroupShaderKHR groupShader ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetRayTracingShaderGroupStackSizeKHR &&
        "Function <vkGetRayTracingShaderGroupStackSizeKHR> needs extension <VK_KHR_ray_tracing_pipeline> enabled!" );

      return static_cast<VULKAN_HPP_NAMESPACE::DeviceSize>(
        getDispatcher()->vkGetRayTracingShaderGroupStackSizeKHR( static_cast<VkDevice>( m_device ),
                                                                 static_cast<VkPipeline>( m_pipeline ),
                                                                 group,
                                                                 static_cast<VkShaderGroupShaderKHR>( groupShader ) ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::setRayTracingPipelineStackSizeKHR( uint32_t pipelineStackSize ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdSetRayTracingPipelineStackSizeKHR &&
        "Function <vkCmdSetRayTracingPipelineStackSizeKHR> needs extension <VK_KHR_ray_tracing_pipeline> enabled!" );

      getDispatcher()->vkCmdSetRayTracingPipelineStackSizeKHR( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                               pipelineStackSize );
    }

    //=== VK_EXT_vertex_input_dynamic_state ===

    VULKAN_HPP_INLINE void CommandBuffer::setVertexInputEXT(
      ArrayProxy<const VULKAN_HPP_NAMESPACE::VertexInputBindingDescription2EXT> const &   vertexBindingDescriptions,
      ArrayProxy<const VULKAN_HPP_NAMESPACE::VertexInputAttributeDescription2EXT> const & vertexAttributeDescriptions )
      const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdSetVertexInputEXT &&
        "Function <vkCmdSetVertexInputEXT> needs extension <VK_EXT_vertex_input_dynamic_state> enabled!" );

      getDispatcher()->vkCmdSetVertexInputEXT(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        vertexBindingDescriptions.size(),
        reinterpret_cast<const VkVertexInputBindingDescription2EXT *>( vertexBindingDescriptions.data() ),
        vertexAttributeDescriptions.size(),
        reinterpret_cast<const VkVertexInputAttributeDescription2EXT *>( vertexAttributeDescriptions.data() ) );
    }

#  if defined( VK_USE_PLATFORM_FUCHSIA )
    //=== VK_FUCHSIA_external_memory ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE zx_handle_t
                                           Device::getMemoryZirconHandleFUCHSIA( const MemoryGetZirconHandleInfoFUCHSIA & getZirconHandleInfo ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetMemoryZirconHandleFUCHSIA &&
        "Function <vkGetMemoryZirconHandleFUCHSIA> needs extension <VK_FUCHSIA_external_memory> enabled!" );

      zx_handle_t                  zirconHandle;
      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetMemoryZirconHandleFUCHSIA(
          static_cast<VkDevice>( m_device ),
          reinterpret_cast<const VkMemoryGetZirconHandleInfoFUCHSIA *>( &getZirconHandleInfo ),
          &zirconHandle ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::getMemoryZirconHandleFUCHSIA" );
      }
      return zirconHandle;
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::MemoryZirconHandlePropertiesFUCHSIA
                                           Device::getMemoryZirconHandlePropertiesFUCHSIA( VULKAN_HPP_NAMESPACE::ExternalMemoryHandleTypeFlagBits handleType,
                                                      zx_handle_t zirconHandle ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetMemoryZirconHandlePropertiesFUCHSIA &&
        "Function <vkGetMemoryZirconHandlePropertiesFUCHSIA> needs extension <VK_FUCHSIA_external_memory> enabled!" );

      VULKAN_HPP_NAMESPACE::MemoryZirconHandlePropertiesFUCHSIA memoryZirconHandleProperties;
      VULKAN_HPP_NAMESPACE::Result                              result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetMemoryZirconHandlePropertiesFUCHSIA(
          static_cast<VkDevice>( m_device ),
          static_cast<VkExternalMemoryHandleTypeFlagBits>( handleType ),
          zirconHandle,
          reinterpret_cast<VkMemoryZirconHandlePropertiesFUCHSIA *>( &memoryZirconHandleProperties ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::getMemoryZirconHandlePropertiesFUCHSIA" );
      }
      return memoryZirconHandleProperties;
    }
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

#  if defined( VK_USE_PLATFORM_FUCHSIA )
    //=== VK_FUCHSIA_external_semaphore ===

    VULKAN_HPP_INLINE void Device::importSemaphoreZirconHandleFUCHSIA(
      const ImportSemaphoreZirconHandleInfoFUCHSIA & importSemaphoreZirconHandleInfo ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkImportSemaphoreZirconHandleFUCHSIA &&
        "Function <vkImportSemaphoreZirconHandleFUCHSIA> needs extension <VK_FUCHSIA_external_semaphore> enabled!" );

      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkImportSemaphoreZirconHandleFUCHSIA(
          static_cast<VkDevice>( m_device ),
          reinterpret_cast<const VkImportSemaphoreZirconHandleInfoFUCHSIA *>( &importSemaphoreZirconHandleInfo ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::importSemaphoreZirconHandleFUCHSIA" );
      }
    }

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE zx_handle_t
                                           Device::getSemaphoreZirconHandleFUCHSIA( const SemaphoreGetZirconHandleInfoFUCHSIA & getZirconHandleInfo ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetSemaphoreZirconHandleFUCHSIA &&
        "Function <vkGetSemaphoreZirconHandleFUCHSIA> needs extension <VK_FUCHSIA_external_semaphore> enabled!" );

      zx_handle_t                  zirconHandle;
      VULKAN_HPP_NAMESPACE::Result result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetSemaphoreZirconHandleFUCHSIA(
          static_cast<VkDevice>( m_device ),
          reinterpret_cast<const VkSemaphoreGetZirconHandleInfoFUCHSIA *>( &getZirconHandleInfo ),
          &zirconHandle ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::getSemaphoreZirconHandleFUCHSIA" );
      }
      return zirconHandle;
    }
#  endif /*VK_USE_PLATFORM_FUCHSIA*/

    //=== VK_HUAWEI_subpass_shading ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE std::pair<VULKAN_HPP_NAMESPACE::Result, VULKAN_HPP_NAMESPACE::Extent2D>
                                           RenderPass::getSubpassShadingMaxWorkgroupSizeHUAWEI() const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI &&
        "Function <vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI> needs extension <VK_HUAWEI_subpass_shading> enabled!" );

      VULKAN_HPP_NAMESPACE::Extent2D maxWorkgroupSize;
      VULKAN_HPP_NAMESPACE::Result   result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI(
          static_cast<VkDevice>( m_device ),
          static_cast<VkRenderPass>( m_renderPass ),
          reinterpret_cast<VkExtent2D *>( &maxWorkgroupSize ) ) );
      if ( ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess ) &&
           ( result != VULKAN_HPP_NAMESPACE::Result::eIncomplete ) )
      {
        throwResultException( result,
                              VULKAN_HPP_NAMESPACE_STRING "::RenderPass::getSubpassShadingMaxWorkgroupSizeHUAWEI" );
      }
      return std::make_pair( result, maxWorkgroupSize );
    }

    VULKAN_HPP_INLINE void CommandBuffer::subpassShadingHUAWEI() const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdSubpassShadingHUAWEI &&
                         "Function <vkCmdSubpassShadingHUAWEI> needs extension <VK_HUAWEI_subpass_shading> enabled!" );

      getDispatcher()->vkCmdSubpassShadingHUAWEI( static_cast<VkCommandBuffer>( m_commandBuffer ) );
    }

    //=== VK_HUAWEI_invocation_mask ===

    VULKAN_HPP_INLINE void
      CommandBuffer::bindInvocationMaskHUAWEI( VULKAN_HPP_NAMESPACE::ImageView   imageView,
                                               VULKAN_HPP_NAMESPACE::ImageLayout imageLayout ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdBindInvocationMaskHUAWEI &&
        "Function <vkCmdBindInvocationMaskHUAWEI> needs extension <VK_HUAWEI_invocation_mask> enabled!" );

      getDispatcher()->vkCmdBindInvocationMaskHUAWEI( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                      static_cast<VkImageView>( imageView ),
                                                      static_cast<VkImageLayout>( imageLayout ) );
    }

    //=== VK_NV_external_memory_rdma ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::RemoteAddressNV
                                           Device::getMemoryRemoteAddressNV( const MemoryGetRemoteAddressInfoNV & memoryGetRemoteAddressInfo ) const
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetMemoryRemoteAddressNV &&
        "Function <vkGetMemoryRemoteAddressNV> needs extension <VK_NV_external_memory_rdma> enabled!" );

      VULKAN_HPP_NAMESPACE::RemoteAddressNV address;
      VULKAN_HPP_NAMESPACE::Result          result =
        static_cast<VULKAN_HPP_NAMESPACE::Result>( getDispatcher()->vkGetMemoryRemoteAddressNV(
          static_cast<VkDevice>( m_device ),
          reinterpret_cast<const VkMemoryGetRemoteAddressInfoNV *>( &memoryGetRemoteAddressInfo ),
          reinterpret_cast<VkRemoteAddressNV *>( &address ) ) );
      if ( result != VULKAN_HPP_NAMESPACE::Result::eSuccess )
      {
        throwResultException( result, VULKAN_HPP_NAMESPACE_STRING "::Device::getMemoryRemoteAddressNV" );
      }
      return address;
    }

    //=== VK_EXT_extended_dynamic_state2 ===

    VULKAN_HPP_INLINE void
      CommandBuffer::setPatchControlPointsEXT( uint32_t patchControlPoints ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdSetPatchControlPointsEXT &&
        "Function <vkCmdSetPatchControlPointsEXT> needs extension <VK_EXT_extended_dynamic_state2> enabled!" );

      getDispatcher()->vkCmdSetPatchControlPointsEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                      patchControlPoints );
    }

    VULKAN_HPP_INLINE void CommandBuffer::setRasterizerDiscardEnableEXT(
      VULKAN_HPP_NAMESPACE::Bool32 rasterizerDiscardEnable ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdSetRasterizerDiscardEnableEXT &&
        "Function <vkCmdSetRasterizerDiscardEnableEXT> needs extension <VK_EXT_extended_dynamic_state2> enabled!" );

      getDispatcher()->vkCmdSetRasterizerDiscardEnableEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                           static_cast<VkBool32>( rasterizerDiscardEnable ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::setDepthBiasEnableEXT( VULKAN_HPP_NAMESPACE::Bool32 depthBiasEnable ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdSetDepthBiasEnableEXT &&
        "Function <vkCmdSetDepthBiasEnableEXT> needs extension <VK_EXT_extended_dynamic_state2> enabled!" );

      getDispatcher()->vkCmdSetDepthBiasEnableEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                   static_cast<VkBool32>( depthBiasEnable ) );
    }

    VULKAN_HPP_INLINE void
      CommandBuffer::setLogicOpEXT( VULKAN_HPP_NAMESPACE::LogicOp logicOp ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdSetLogicOpEXT &&
                         "Function <vkCmdSetLogicOpEXT> needs extension <VK_EXT_extended_dynamic_state2> enabled!" );

      getDispatcher()->vkCmdSetLogicOpEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                           static_cast<VkLogicOp>( logicOp ) );
    }

    VULKAN_HPP_INLINE void CommandBuffer::setPrimitiveRestartEnableEXT(
      VULKAN_HPP_NAMESPACE::Bool32 primitiveRestartEnable ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdSetPrimitiveRestartEnableEXT &&
        "Function <vkCmdSetPrimitiveRestartEnableEXT> needs extension <VK_EXT_extended_dynamic_state2> enabled!" );

      getDispatcher()->vkCmdSetPrimitiveRestartEnableEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                          static_cast<VkBool32>( primitiveRestartEnable ) );
    }

#  if defined( VK_USE_PLATFORM_SCREEN_QNX )
    //=== VK_QNX_screen_surface ===

    VULKAN_HPP_NODISCARD VULKAN_HPP_INLINE VULKAN_HPP_NAMESPACE::Bool32
                                           PhysicalDevice::getScreenPresentationSupportQNX( uint32_t                queueFamilyIndex,
                                                       struct _screen_window & window ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkGetPhysicalDeviceScreenPresentationSupportQNX &&
        "Function <vkGetPhysicalDeviceScreenPresentationSupportQNX> needs extension <VK_QNX_screen_surface> enabled!" );

      return static_cast<VULKAN_HPP_NAMESPACE::Bool32>(
        getDispatcher()->vkGetPhysicalDeviceScreenPresentationSupportQNX(
          static_cast<VkPhysicalDevice>( m_physicalDevice ), queueFamilyIndex, &window ) );
    }
#  endif /*VK_USE_PLATFORM_SCREEN_QNX*/

    //=== VK_EXT_color_write_enable ===

    VULKAN_HPP_INLINE void CommandBuffer::setColorWriteEnableEXT(
      ArrayProxy<const VULKAN_HPP_NAMESPACE::Bool32> const & colorWriteEnables ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT(
        getDispatcher()->vkCmdSetColorWriteEnableEXT &&
        "Function <vkCmdSetColorWriteEnableEXT> needs extension <VK_EXT_color_write_enable> enabled!" );

      getDispatcher()->vkCmdSetColorWriteEnableEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                                    colorWriteEnables.size(),
                                                    reinterpret_cast<const VkBool32 *>( colorWriteEnables.data() ) );
    }

    //=== VK_EXT_multi_draw ===

    VULKAN_HPP_INLINE void
      CommandBuffer::drawMultiEXT( ArrayProxy<const VULKAN_HPP_NAMESPACE::MultiDrawInfoEXT> const & vertexInfo,
                                   uint32_t                                                         instanceCount,
                                   uint32_t                                                         firstInstance,
                                   uint32_t stride ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdDrawMultiEXT &&
                         "Function <vkCmdDrawMultiEXT> needs extension <VK_EXT_multi_draw> enabled!" );

      getDispatcher()->vkCmdDrawMultiEXT( static_cast<VkCommandBuffer>( m_commandBuffer ),
                                          vertexInfo.size(),
                                          reinterpret_cast<const VkMultiDrawInfoEXT *>( vertexInfo.data() ),
                                          instanceCount,
                                          firstInstance,
                                          stride );
    }

    VULKAN_HPP_INLINE void CommandBuffer::drawMultiIndexedEXT(
      ArrayProxy<const VULKAN_HPP_NAMESPACE::MultiDrawIndexedInfoEXT> const & indexInfo,
      uint32_t                                                                instanceCount,
      uint32_t                                                                firstInstance,
      uint32_t                                                                stride,
      Optional<const int32_t>                                                 vertexOffset ) const VULKAN_HPP_NOEXCEPT
    {
      VULKAN_HPP_ASSERT( getDispatcher()->vkCmdDrawMultiIndexedEXT &&
                         "Function <vkCmdDrawMultiIndexedEXT> needs extension <VK_EXT_multi_draw> enabled!" );

      getDispatcher()->vkCmdDrawMultiIndexedEXT(
        static_cast<VkCommandBuffer>( m_commandBuffer ),
        indexInfo.size(),
        reinterpret_cast<const VkMultiDrawIndexedInfoEXT *>( indexInfo.data() ),
        instanceCount,
        firstInstance,
        stride,
        static_cast<const int32_t *>( vertexOffset ) );
    }

#endif
  }  // namespace VULKAN_HPP_RAII_NAMESPACE
}  // namespace VULKAN_HPP_NAMESPACE
#endif
