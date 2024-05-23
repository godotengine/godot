/* This file is part of volk library; see volk.h for version/license details */
/* clang-format off */
#include "volk.h"

#ifdef _WIN32
	typedef const char* LPCSTR;
	typedef struct HINSTANCE__* HINSTANCE;
	typedef HINSTANCE HMODULE;
	#if defined(_MINWINDEF_)
		/* minwindef.h defines FARPROC, and attempting to redefine it may conflict with -Wstrict-prototypes */
	#elif defined(_WIN64)
		typedef __int64 (__stdcall* FARPROC)(void);
	#else
		typedef int (__stdcall* FARPROC)(void);
	#endif
#else
#	include <dlfcn.h>
#endif

#ifdef __APPLE__
#	include <stdlib.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
__declspec(dllimport) HMODULE __stdcall LoadLibraryA(LPCSTR);
__declspec(dllimport) FARPROC __stdcall GetProcAddress(HMODULE, LPCSTR);
__declspec(dllimport) int __stdcall FreeLibrary(HMODULE);
#endif

static void* loadedModule = NULL;
static VkInstance loadedInstance = VK_NULL_HANDLE;
static VkDevice loadedDevice = VK_NULL_HANDLE;

static void volkGenLoadLoader(void* context, PFN_vkVoidFunction (*load)(void*, const char*));
static void volkGenLoadInstance(void* context, PFN_vkVoidFunction (*load)(void*, const char*));
static void volkGenLoadDevice(void* context, PFN_vkVoidFunction (*load)(void*, const char*));
static void volkGenLoadDeviceTable(struct VolkDeviceTable* table, void* context, PFN_vkVoidFunction (*load)(void*, const char*));

static PFN_vkVoidFunction vkGetInstanceProcAddrStub(void* context, const char* name)
{
	return vkGetInstanceProcAddr((VkInstance)context, name);
}

static PFN_vkVoidFunction vkGetDeviceProcAddrStub(void* context, const char* name)
{
	return vkGetDeviceProcAddr((VkDevice)context, name);
}

static PFN_vkVoidFunction nullProcAddrStub(void* context, const char* name)
{
	(void)context;
	(void)name;
	return NULL;
}

VkResult volkInitialize(void)
{
#if defined(_WIN32)
	HMODULE module = LoadLibraryA("vulkan-1.dll");
	if (!module)
		return VK_ERROR_INITIALIZATION_FAILED;

	// note: function pointer is cast through void function pointer to silence cast-function-type warning on gcc8
	vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)(void(*)(void))GetProcAddress(module, "vkGetInstanceProcAddr");
#elif defined(__APPLE__)
	void* module = dlopen("libvulkan.dylib", RTLD_NOW | RTLD_LOCAL);
	if (!module)
		module = dlopen("libvulkan.1.dylib", RTLD_NOW | RTLD_LOCAL);
	if (!module)
		module = dlopen("libMoltenVK.dylib", RTLD_NOW | RTLD_LOCAL);
	// modern versions of macOS don't search /usr/local/lib automatically contrary to what man dlopen says
	// Vulkan SDK uses this as the system-wide installation location, so we're going to fallback to this if all else fails
	if (!module && getenv("DYLD_FALLBACK_LIBRARY_PATH") == NULL)
		module = dlopen("/usr/local/lib/libvulkan.dylib", RTLD_NOW | RTLD_LOCAL);
	if (!module)
		return VK_ERROR_INITIALIZATION_FAILED;

	vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(module, "vkGetInstanceProcAddr");
#else
	void* module = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
	if (!module)
		module = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
	if (!module)
		return VK_ERROR_INITIALIZATION_FAILED;

	vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(module, "vkGetInstanceProcAddr");
#endif

	loadedModule = module;
	volkGenLoadLoader(NULL, vkGetInstanceProcAddrStub);

	return VK_SUCCESS;
}

void volkInitializeCustom(PFN_vkGetInstanceProcAddr handler)
{
	vkGetInstanceProcAddr = handler;

	loadedModule = NULL;
	volkGenLoadLoader(NULL, vkGetInstanceProcAddrStub);
}

void volkFinalize(void)
{
	if (loadedModule)
	{
#if defined(_WIN32)
		FreeLibrary((HMODULE)loadedModule);
#else
		dlclose(loadedModule);
#endif
	}

	vkGetInstanceProcAddr = NULL;
	volkGenLoadLoader(NULL, nullProcAddrStub);
	volkGenLoadInstance(NULL, nullProcAddrStub);
	volkGenLoadDevice(NULL, nullProcAddrStub);

	loadedModule = NULL;
	loadedInstance = VK_NULL_HANDLE;
	loadedDevice = VK_NULL_HANDLE;
}

uint32_t volkGetInstanceVersion(void)
{
#if defined(VK_VERSION_1_1)
	uint32_t apiVersion = 0;
	if (vkEnumerateInstanceVersion && vkEnumerateInstanceVersion(&apiVersion) == VK_SUCCESS)
		return apiVersion;
#endif

	if (vkCreateInstance)
		return VK_API_VERSION_1_0;

	return 0;
}

void volkLoadInstance(VkInstance instance)
{
	loadedInstance = instance;
	volkGenLoadInstance(instance, vkGetInstanceProcAddrStub);
	volkGenLoadDevice(instance, vkGetInstanceProcAddrStub);
}

void volkLoadInstanceOnly(VkInstance instance)
{
	loadedInstance = instance;
	volkGenLoadInstance(instance, vkGetInstanceProcAddrStub);
}

VkInstance volkGetLoadedInstance(void)
{
	return loadedInstance;
}

void volkLoadDevice(VkDevice device)
{
	loadedDevice = device;
	volkGenLoadDevice(device, vkGetDeviceProcAddrStub);
}

VkDevice volkGetLoadedDevice(void)
{
	return loadedDevice;
}

void volkLoadDeviceTable(struct VolkDeviceTable* table, VkDevice device)
{
	volkGenLoadDeviceTable(table, device, vkGetDeviceProcAddrStub);
}

static void volkGenLoadLoader(void* context, PFN_vkVoidFunction (*load)(void*, const char*))
{
	/* VOLK_GENERATE_LOAD_LOADER */
#if defined(VK_VERSION_1_0)
	vkCreateInstance = (PFN_vkCreateInstance)load(context, "vkCreateInstance");
	vkEnumerateInstanceExtensionProperties = (PFN_vkEnumerateInstanceExtensionProperties)load(context, "vkEnumerateInstanceExtensionProperties");
	vkEnumerateInstanceLayerProperties = (PFN_vkEnumerateInstanceLayerProperties)load(context, "vkEnumerateInstanceLayerProperties");
#endif /* defined(VK_VERSION_1_0) */
#if defined(VK_VERSION_1_1)
	vkEnumerateInstanceVersion = (PFN_vkEnumerateInstanceVersion)load(context, "vkEnumerateInstanceVersion");
#endif /* defined(VK_VERSION_1_1) */
	/* VOLK_GENERATE_LOAD_LOADER */
}

static void volkGenLoadInstance(void* context, PFN_vkVoidFunction (*load)(void*, const char*))
{
	/* VOLK_GENERATE_LOAD_INSTANCE */
#if defined(VK_VERSION_1_0)
	vkCreateDevice = (PFN_vkCreateDevice)load(context, "vkCreateDevice");
	vkDestroyInstance = (PFN_vkDestroyInstance)load(context, "vkDestroyInstance");
	vkEnumerateDeviceExtensionProperties = (PFN_vkEnumerateDeviceExtensionProperties)load(context, "vkEnumerateDeviceExtensionProperties");
	vkEnumerateDeviceLayerProperties = (PFN_vkEnumerateDeviceLayerProperties)load(context, "vkEnumerateDeviceLayerProperties");
	vkEnumeratePhysicalDevices = (PFN_vkEnumeratePhysicalDevices)load(context, "vkEnumeratePhysicalDevices");
	vkGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)load(context, "vkGetDeviceProcAddr");
	vkGetPhysicalDeviceFeatures = (PFN_vkGetPhysicalDeviceFeatures)load(context, "vkGetPhysicalDeviceFeatures");
	vkGetPhysicalDeviceFormatProperties = (PFN_vkGetPhysicalDeviceFormatProperties)load(context, "vkGetPhysicalDeviceFormatProperties");
	vkGetPhysicalDeviceImageFormatProperties = (PFN_vkGetPhysicalDeviceImageFormatProperties)load(context, "vkGetPhysicalDeviceImageFormatProperties");
	vkGetPhysicalDeviceMemoryProperties = (PFN_vkGetPhysicalDeviceMemoryProperties)load(context, "vkGetPhysicalDeviceMemoryProperties");
	vkGetPhysicalDeviceProperties = (PFN_vkGetPhysicalDeviceProperties)load(context, "vkGetPhysicalDeviceProperties");
	vkGetPhysicalDeviceQueueFamilyProperties = (PFN_vkGetPhysicalDeviceQueueFamilyProperties)load(context, "vkGetPhysicalDeviceQueueFamilyProperties");
	vkGetPhysicalDeviceSparseImageFormatProperties = (PFN_vkGetPhysicalDeviceSparseImageFormatProperties)load(context, "vkGetPhysicalDeviceSparseImageFormatProperties");
#endif /* defined(VK_VERSION_1_0) */
#if defined(VK_VERSION_1_1)
	vkEnumeratePhysicalDeviceGroups = (PFN_vkEnumeratePhysicalDeviceGroups)load(context, "vkEnumeratePhysicalDeviceGroups");
	vkGetPhysicalDeviceExternalBufferProperties = (PFN_vkGetPhysicalDeviceExternalBufferProperties)load(context, "vkGetPhysicalDeviceExternalBufferProperties");
	vkGetPhysicalDeviceExternalFenceProperties = (PFN_vkGetPhysicalDeviceExternalFenceProperties)load(context, "vkGetPhysicalDeviceExternalFenceProperties");
	vkGetPhysicalDeviceExternalSemaphoreProperties = (PFN_vkGetPhysicalDeviceExternalSemaphoreProperties)load(context, "vkGetPhysicalDeviceExternalSemaphoreProperties");
	vkGetPhysicalDeviceFeatures2 = (PFN_vkGetPhysicalDeviceFeatures2)load(context, "vkGetPhysicalDeviceFeatures2");
	vkGetPhysicalDeviceFormatProperties2 = (PFN_vkGetPhysicalDeviceFormatProperties2)load(context, "vkGetPhysicalDeviceFormatProperties2");
	vkGetPhysicalDeviceImageFormatProperties2 = (PFN_vkGetPhysicalDeviceImageFormatProperties2)load(context, "vkGetPhysicalDeviceImageFormatProperties2");
	vkGetPhysicalDeviceMemoryProperties2 = (PFN_vkGetPhysicalDeviceMemoryProperties2)load(context, "vkGetPhysicalDeviceMemoryProperties2");
	vkGetPhysicalDeviceProperties2 = (PFN_vkGetPhysicalDeviceProperties2)load(context, "vkGetPhysicalDeviceProperties2");
	vkGetPhysicalDeviceQueueFamilyProperties2 = (PFN_vkGetPhysicalDeviceQueueFamilyProperties2)load(context, "vkGetPhysicalDeviceQueueFamilyProperties2");
	vkGetPhysicalDeviceSparseImageFormatProperties2 = (PFN_vkGetPhysicalDeviceSparseImageFormatProperties2)load(context, "vkGetPhysicalDeviceSparseImageFormatProperties2");
#endif /* defined(VK_VERSION_1_1) */
#if defined(VK_VERSION_1_3)
	vkGetPhysicalDeviceToolProperties = (PFN_vkGetPhysicalDeviceToolProperties)load(context, "vkGetPhysicalDeviceToolProperties");
#endif /* defined(VK_VERSION_1_3) */
#if defined(VK_EXT_acquire_drm_display)
	vkAcquireDrmDisplayEXT = (PFN_vkAcquireDrmDisplayEXT)load(context, "vkAcquireDrmDisplayEXT");
	vkGetDrmDisplayEXT = (PFN_vkGetDrmDisplayEXT)load(context, "vkGetDrmDisplayEXT");
#endif /* defined(VK_EXT_acquire_drm_display) */
#if defined(VK_EXT_acquire_xlib_display)
	vkAcquireXlibDisplayEXT = (PFN_vkAcquireXlibDisplayEXT)load(context, "vkAcquireXlibDisplayEXT");
	vkGetRandROutputDisplayEXT = (PFN_vkGetRandROutputDisplayEXT)load(context, "vkGetRandROutputDisplayEXT");
#endif /* defined(VK_EXT_acquire_xlib_display) */
#if defined(VK_EXT_calibrated_timestamps)
	vkGetPhysicalDeviceCalibrateableTimeDomainsEXT = (PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT)load(context, "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT");
#endif /* defined(VK_EXT_calibrated_timestamps) */
#if defined(VK_EXT_debug_report)
	vkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)load(context, "vkCreateDebugReportCallbackEXT");
	vkDebugReportMessageEXT = (PFN_vkDebugReportMessageEXT)load(context, "vkDebugReportMessageEXT");
	vkDestroyDebugReportCallbackEXT = (PFN_vkDestroyDebugReportCallbackEXT)load(context, "vkDestroyDebugReportCallbackEXT");
#endif /* defined(VK_EXT_debug_report) */
#if defined(VK_EXT_debug_utils)
	vkCmdBeginDebugUtilsLabelEXT = (PFN_vkCmdBeginDebugUtilsLabelEXT)load(context, "vkCmdBeginDebugUtilsLabelEXT");
	vkCmdEndDebugUtilsLabelEXT = (PFN_vkCmdEndDebugUtilsLabelEXT)load(context, "vkCmdEndDebugUtilsLabelEXT");
	vkCmdInsertDebugUtilsLabelEXT = (PFN_vkCmdInsertDebugUtilsLabelEXT)load(context, "vkCmdInsertDebugUtilsLabelEXT");
	vkCreateDebugUtilsMessengerEXT = (PFN_vkCreateDebugUtilsMessengerEXT)load(context, "vkCreateDebugUtilsMessengerEXT");
	vkDestroyDebugUtilsMessengerEXT = (PFN_vkDestroyDebugUtilsMessengerEXT)load(context, "vkDestroyDebugUtilsMessengerEXT");
	vkQueueBeginDebugUtilsLabelEXT = (PFN_vkQueueBeginDebugUtilsLabelEXT)load(context, "vkQueueBeginDebugUtilsLabelEXT");
	vkQueueEndDebugUtilsLabelEXT = (PFN_vkQueueEndDebugUtilsLabelEXT)load(context, "vkQueueEndDebugUtilsLabelEXT");
	vkQueueInsertDebugUtilsLabelEXT = (PFN_vkQueueInsertDebugUtilsLabelEXT)load(context, "vkQueueInsertDebugUtilsLabelEXT");
	vkSetDebugUtilsObjectNameEXT = (PFN_vkSetDebugUtilsObjectNameEXT)load(context, "vkSetDebugUtilsObjectNameEXT");
	vkSetDebugUtilsObjectTagEXT = (PFN_vkSetDebugUtilsObjectTagEXT)load(context, "vkSetDebugUtilsObjectTagEXT");
	vkSubmitDebugUtilsMessageEXT = (PFN_vkSubmitDebugUtilsMessageEXT)load(context, "vkSubmitDebugUtilsMessageEXT");
#endif /* defined(VK_EXT_debug_utils) */
#if defined(VK_EXT_direct_mode_display)
	vkReleaseDisplayEXT = (PFN_vkReleaseDisplayEXT)load(context, "vkReleaseDisplayEXT");
#endif /* defined(VK_EXT_direct_mode_display) */
#if defined(VK_EXT_directfb_surface)
	vkCreateDirectFBSurfaceEXT = (PFN_vkCreateDirectFBSurfaceEXT)load(context, "vkCreateDirectFBSurfaceEXT");
	vkGetPhysicalDeviceDirectFBPresentationSupportEXT = (PFN_vkGetPhysicalDeviceDirectFBPresentationSupportEXT)load(context, "vkGetPhysicalDeviceDirectFBPresentationSupportEXT");
#endif /* defined(VK_EXT_directfb_surface) */
#if defined(VK_EXT_display_surface_counter)
	vkGetPhysicalDeviceSurfaceCapabilities2EXT = (PFN_vkGetPhysicalDeviceSurfaceCapabilities2EXT)load(context, "vkGetPhysicalDeviceSurfaceCapabilities2EXT");
#endif /* defined(VK_EXT_display_surface_counter) */
#if defined(VK_EXT_full_screen_exclusive)
	vkGetPhysicalDeviceSurfacePresentModes2EXT = (PFN_vkGetPhysicalDeviceSurfacePresentModes2EXT)load(context, "vkGetPhysicalDeviceSurfacePresentModes2EXT");
#endif /* defined(VK_EXT_full_screen_exclusive) */
#if defined(VK_EXT_headless_surface)
	vkCreateHeadlessSurfaceEXT = (PFN_vkCreateHeadlessSurfaceEXT)load(context, "vkCreateHeadlessSurfaceEXT");
#endif /* defined(VK_EXT_headless_surface) */
#if defined(VK_EXT_metal_surface)
	vkCreateMetalSurfaceEXT = (PFN_vkCreateMetalSurfaceEXT)load(context, "vkCreateMetalSurfaceEXT");
#endif /* defined(VK_EXT_metal_surface) */
#if defined(VK_EXT_sample_locations)
	vkGetPhysicalDeviceMultisamplePropertiesEXT = (PFN_vkGetPhysicalDeviceMultisamplePropertiesEXT)load(context, "vkGetPhysicalDeviceMultisamplePropertiesEXT");
#endif /* defined(VK_EXT_sample_locations) */
#if defined(VK_EXT_tooling_info)
	vkGetPhysicalDeviceToolPropertiesEXT = (PFN_vkGetPhysicalDeviceToolPropertiesEXT)load(context, "vkGetPhysicalDeviceToolPropertiesEXT");
#endif /* defined(VK_EXT_tooling_info) */
#if defined(VK_FUCHSIA_imagepipe_surface)
	vkCreateImagePipeSurfaceFUCHSIA = (PFN_vkCreateImagePipeSurfaceFUCHSIA)load(context, "vkCreateImagePipeSurfaceFUCHSIA");
#endif /* defined(VK_FUCHSIA_imagepipe_surface) */
#if defined(VK_GGP_stream_descriptor_surface)
	vkCreateStreamDescriptorSurfaceGGP = (PFN_vkCreateStreamDescriptorSurfaceGGP)load(context, "vkCreateStreamDescriptorSurfaceGGP");
#endif /* defined(VK_GGP_stream_descriptor_surface) */
#if defined(VK_KHR_android_surface)
	vkCreateAndroidSurfaceKHR = (PFN_vkCreateAndroidSurfaceKHR)load(context, "vkCreateAndroidSurfaceKHR");
#endif /* defined(VK_KHR_android_surface) */
#if defined(VK_KHR_calibrated_timestamps)
	vkGetPhysicalDeviceCalibrateableTimeDomainsKHR = (PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR)load(context, "vkGetPhysicalDeviceCalibrateableTimeDomainsKHR");
#endif /* defined(VK_KHR_calibrated_timestamps) */
#if defined(VK_KHR_cooperative_matrix)
	vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)load(context, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
#endif /* defined(VK_KHR_cooperative_matrix) */
#if defined(VK_KHR_device_group_creation)
	vkEnumeratePhysicalDeviceGroupsKHR = (PFN_vkEnumeratePhysicalDeviceGroupsKHR)load(context, "vkEnumeratePhysicalDeviceGroupsKHR");
#endif /* defined(VK_KHR_device_group_creation) */
#if defined(VK_KHR_display)
	vkCreateDisplayModeKHR = (PFN_vkCreateDisplayModeKHR)load(context, "vkCreateDisplayModeKHR");
	vkCreateDisplayPlaneSurfaceKHR = (PFN_vkCreateDisplayPlaneSurfaceKHR)load(context, "vkCreateDisplayPlaneSurfaceKHR");
	vkGetDisplayModePropertiesKHR = (PFN_vkGetDisplayModePropertiesKHR)load(context, "vkGetDisplayModePropertiesKHR");
	vkGetDisplayPlaneCapabilitiesKHR = (PFN_vkGetDisplayPlaneCapabilitiesKHR)load(context, "vkGetDisplayPlaneCapabilitiesKHR");
	vkGetDisplayPlaneSupportedDisplaysKHR = (PFN_vkGetDisplayPlaneSupportedDisplaysKHR)load(context, "vkGetDisplayPlaneSupportedDisplaysKHR");
	vkGetPhysicalDeviceDisplayPlanePropertiesKHR = (PFN_vkGetPhysicalDeviceDisplayPlanePropertiesKHR)load(context, "vkGetPhysicalDeviceDisplayPlanePropertiesKHR");
	vkGetPhysicalDeviceDisplayPropertiesKHR = (PFN_vkGetPhysicalDeviceDisplayPropertiesKHR)load(context, "vkGetPhysicalDeviceDisplayPropertiesKHR");
#endif /* defined(VK_KHR_display) */
#if defined(VK_KHR_external_fence_capabilities)
	vkGetPhysicalDeviceExternalFencePropertiesKHR = (PFN_vkGetPhysicalDeviceExternalFencePropertiesKHR)load(context, "vkGetPhysicalDeviceExternalFencePropertiesKHR");
#endif /* defined(VK_KHR_external_fence_capabilities) */
#if defined(VK_KHR_external_memory_capabilities)
	vkGetPhysicalDeviceExternalBufferPropertiesKHR = (PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR)load(context, "vkGetPhysicalDeviceExternalBufferPropertiesKHR");
#endif /* defined(VK_KHR_external_memory_capabilities) */
#if defined(VK_KHR_external_semaphore_capabilities)
	vkGetPhysicalDeviceExternalSemaphorePropertiesKHR = (PFN_vkGetPhysicalDeviceExternalSemaphorePropertiesKHR)load(context, "vkGetPhysicalDeviceExternalSemaphorePropertiesKHR");
#endif /* defined(VK_KHR_external_semaphore_capabilities) */
#if defined(VK_KHR_fragment_shading_rate)
	vkGetPhysicalDeviceFragmentShadingRatesKHR = (PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR)load(context, "vkGetPhysicalDeviceFragmentShadingRatesKHR");
#endif /* defined(VK_KHR_fragment_shading_rate) */
#if defined(VK_KHR_get_display_properties2)
	vkGetDisplayModeProperties2KHR = (PFN_vkGetDisplayModeProperties2KHR)load(context, "vkGetDisplayModeProperties2KHR");
	vkGetDisplayPlaneCapabilities2KHR = (PFN_vkGetDisplayPlaneCapabilities2KHR)load(context, "vkGetDisplayPlaneCapabilities2KHR");
	vkGetPhysicalDeviceDisplayPlaneProperties2KHR = (PFN_vkGetPhysicalDeviceDisplayPlaneProperties2KHR)load(context, "vkGetPhysicalDeviceDisplayPlaneProperties2KHR");
	vkGetPhysicalDeviceDisplayProperties2KHR = (PFN_vkGetPhysicalDeviceDisplayProperties2KHR)load(context, "vkGetPhysicalDeviceDisplayProperties2KHR");
#endif /* defined(VK_KHR_get_display_properties2) */
#if defined(VK_KHR_get_physical_device_properties2)
	vkGetPhysicalDeviceFeatures2KHR = (PFN_vkGetPhysicalDeviceFeatures2KHR)load(context, "vkGetPhysicalDeviceFeatures2KHR");
	vkGetPhysicalDeviceFormatProperties2KHR = (PFN_vkGetPhysicalDeviceFormatProperties2KHR)load(context, "vkGetPhysicalDeviceFormatProperties2KHR");
	vkGetPhysicalDeviceImageFormatProperties2KHR = (PFN_vkGetPhysicalDeviceImageFormatProperties2KHR)load(context, "vkGetPhysicalDeviceImageFormatProperties2KHR");
	vkGetPhysicalDeviceMemoryProperties2KHR = (PFN_vkGetPhysicalDeviceMemoryProperties2KHR)load(context, "vkGetPhysicalDeviceMemoryProperties2KHR");
	vkGetPhysicalDeviceProperties2KHR = (PFN_vkGetPhysicalDeviceProperties2KHR)load(context, "vkGetPhysicalDeviceProperties2KHR");
	vkGetPhysicalDeviceQueueFamilyProperties2KHR = (PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR)load(context, "vkGetPhysicalDeviceQueueFamilyProperties2KHR");
	vkGetPhysicalDeviceSparseImageFormatProperties2KHR = (PFN_vkGetPhysicalDeviceSparseImageFormatProperties2KHR)load(context, "vkGetPhysicalDeviceSparseImageFormatProperties2KHR");
#endif /* defined(VK_KHR_get_physical_device_properties2) */
#if defined(VK_KHR_get_surface_capabilities2)
	vkGetPhysicalDeviceSurfaceCapabilities2KHR = (PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR)load(context, "vkGetPhysicalDeviceSurfaceCapabilities2KHR");
	vkGetPhysicalDeviceSurfaceFormats2KHR = (PFN_vkGetPhysicalDeviceSurfaceFormats2KHR)load(context, "vkGetPhysicalDeviceSurfaceFormats2KHR");
#endif /* defined(VK_KHR_get_surface_capabilities2) */
#if defined(VK_KHR_performance_query)
	vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR = (PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR)load(context, "vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR");
	vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR = (PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR)load(context, "vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR");
#endif /* defined(VK_KHR_performance_query) */
#if defined(VK_KHR_surface)
	vkDestroySurfaceKHR = (PFN_vkDestroySurfaceKHR)load(context, "vkDestroySurfaceKHR");
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR = (PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR)load(context, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
	vkGetPhysicalDeviceSurfaceFormatsKHR = (PFN_vkGetPhysicalDeviceSurfaceFormatsKHR)load(context, "vkGetPhysicalDeviceSurfaceFormatsKHR");
	vkGetPhysicalDeviceSurfacePresentModesKHR = (PFN_vkGetPhysicalDeviceSurfacePresentModesKHR)load(context, "vkGetPhysicalDeviceSurfacePresentModesKHR");
	vkGetPhysicalDeviceSurfaceSupportKHR = (PFN_vkGetPhysicalDeviceSurfaceSupportKHR)load(context, "vkGetPhysicalDeviceSurfaceSupportKHR");
#endif /* defined(VK_KHR_surface) */
#if defined(VK_KHR_video_encode_queue)
	vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR = (PFN_vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR)load(context, "vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR");
#endif /* defined(VK_KHR_video_encode_queue) */
#if defined(VK_KHR_video_queue)
	vkGetPhysicalDeviceVideoCapabilitiesKHR = (PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR)load(context, "vkGetPhysicalDeviceVideoCapabilitiesKHR");
	vkGetPhysicalDeviceVideoFormatPropertiesKHR = (PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR)load(context, "vkGetPhysicalDeviceVideoFormatPropertiesKHR");
#endif /* defined(VK_KHR_video_queue) */
#if defined(VK_KHR_wayland_surface)
	vkCreateWaylandSurfaceKHR = (PFN_vkCreateWaylandSurfaceKHR)load(context, "vkCreateWaylandSurfaceKHR");
	vkGetPhysicalDeviceWaylandPresentationSupportKHR = (PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR)load(context, "vkGetPhysicalDeviceWaylandPresentationSupportKHR");
#endif /* defined(VK_KHR_wayland_surface) */
#if defined(VK_KHR_win32_surface)
	vkCreateWin32SurfaceKHR = (PFN_vkCreateWin32SurfaceKHR)load(context, "vkCreateWin32SurfaceKHR");
	vkGetPhysicalDeviceWin32PresentationSupportKHR = (PFN_vkGetPhysicalDeviceWin32PresentationSupportKHR)load(context, "vkGetPhysicalDeviceWin32PresentationSupportKHR");
#endif /* defined(VK_KHR_win32_surface) */
#if defined(VK_KHR_xcb_surface)
	vkCreateXcbSurfaceKHR = (PFN_vkCreateXcbSurfaceKHR)load(context, "vkCreateXcbSurfaceKHR");
	vkGetPhysicalDeviceXcbPresentationSupportKHR = (PFN_vkGetPhysicalDeviceXcbPresentationSupportKHR)load(context, "vkGetPhysicalDeviceXcbPresentationSupportKHR");
#endif /* defined(VK_KHR_xcb_surface) */
#if defined(VK_KHR_xlib_surface)
	vkCreateXlibSurfaceKHR = (PFN_vkCreateXlibSurfaceKHR)load(context, "vkCreateXlibSurfaceKHR");
	vkGetPhysicalDeviceXlibPresentationSupportKHR = (PFN_vkGetPhysicalDeviceXlibPresentationSupportKHR)load(context, "vkGetPhysicalDeviceXlibPresentationSupportKHR");
#endif /* defined(VK_KHR_xlib_surface) */
#if defined(VK_MVK_ios_surface)
	vkCreateIOSSurfaceMVK = (PFN_vkCreateIOSSurfaceMVK)load(context, "vkCreateIOSSurfaceMVK");
#endif /* defined(VK_MVK_ios_surface) */
#if defined(VK_MVK_macos_surface)
	vkCreateMacOSSurfaceMVK = (PFN_vkCreateMacOSSurfaceMVK)load(context, "vkCreateMacOSSurfaceMVK");
#endif /* defined(VK_MVK_macos_surface) */
#if defined(VK_NN_vi_surface)
	vkCreateViSurfaceNN = (PFN_vkCreateViSurfaceNN)load(context, "vkCreateViSurfaceNN");
#endif /* defined(VK_NN_vi_surface) */
#if defined(VK_NV_acquire_winrt_display)
	vkAcquireWinrtDisplayNV = (PFN_vkAcquireWinrtDisplayNV)load(context, "vkAcquireWinrtDisplayNV");
	vkGetWinrtDisplayNV = (PFN_vkGetWinrtDisplayNV)load(context, "vkGetWinrtDisplayNV");
#endif /* defined(VK_NV_acquire_winrt_display) */
#if defined(VK_NV_cooperative_matrix)
	vkGetPhysicalDeviceCooperativeMatrixPropertiesNV = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV)load(context, "vkGetPhysicalDeviceCooperativeMatrixPropertiesNV");
#endif /* defined(VK_NV_cooperative_matrix) */
#if defined(VK_NV_coverage_reduction_mode)
	vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV = (PFN_vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV)load(context, "vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV");
#endif /* defined(VK_NV_coverage_reduction_mode) */
#if defined(VK_NV_external_memory_capabilities)
	vkGetPhysicalDeviceExternalImageFormatPropertiesNV = (PFN_vkGetPhysicalDeviceExternalImageFormatPropertiesNV)load(context, "vkGetPhysicalDeviceExternalImageFormatPropertiesNV");
#endif /* defined(VK_NV_external_memory_capabilities) */
#if defined(VK_NV_optical_flow)
	vkGetPhysicalDeviceOpticalFlowImageFormatsNV = (PFN_vkGetPhysicalDeviceOpticalFlowImageFormatsNV)load(context, "vkGetPhysicalDeviceOpticalFlowImageFormatsNV");
#endif /* defined(VK_NV_optical_flow) */
#if defined(VK_QNX_screen_surface)
	vkCreateScreenSurfaceQNX = (PFN_vkCreateScreenSurfaceQNX)load(context, "vkCreateScreenSurfaceQNX");
	vkGetPhysicalDeviceScreenPresentationSupportQNX = (PFN_vkGetPhysicalDeviceScreenPresentationSupportQNX)load(context, "vkGetPhysicalDeviceScreenPresentationSupportQNX");
#endif /* defined(VK_QNX_screen_surface) */
#if (defined(VK_KHR_device_group) && defined(VK_KHR_surface)) || (defined(VK_KHR_swapchain) && defined(VK_VERSION_1_1))
	vkGetPhysicalDevicePresentRectanglesKHR = (PFN_vkGetPhysicalDevicePresentRectanglesKHR)load(context, "vkGetPhysicalDevicePresentRectanglesKHR");
#endif /* (defined(VK_KHR_device_group) && defined(VK_KHR_surface)) || (defined(VK_KHR_swapchain) && defined(VK_VERSION_1_1)) */
	/* VOLK_GENERATE_LOAD_INSTANCE */
}

static void volkGenLoadDevice(void* context, PFN_vkVoidFunction (*load)(void*, const char*))
{
	/* VOLK_GENERATE_LOAD_DEVICE */
#if defined(VK_VERSION_1_0)
	vkAllocateCommandBuffers = (PFN_vkAllocateCommandBuffers)load(context, "vkAllocateCommandBuffers");
	vkAllocateDescriptorSets = (PFN_vkAllocateDescriptorSets)load(context, "vkAllocateDescriptorSets");
	vkAllocateMemory = (PFN_vkAllocateMemory)load(context, "vkAllocateMemory");
	vkBeginCommandBuffer = (PFN_vkBeginCommandBuffer)load(context, "vkBeginCommandBuffer");
	vkBindBufferMemory = (PFN_vkBindBufferMemory)load(context, "vkBindBufferMemory");
	vkBindImageMemory = (PFN_vkBindImageMemory)load(context, "vkBindImageMemory");
	vkCmdBeginQuery = (PFN_vkCmdBeginQuery)load(context, "vkCmdBeginQuery");
	vkCmdBeginRenderPass = (PFN_vkCmdBeginRenderPass)load(context, "vkCmdBeginRenderPass");
	vkCmdBindDescriptorSets = (PFN_vkCmdBindDescriptorSets)load(context, "vkCmdBindDescriptorSets");
	vkCmdBindIndexBuffer = (PFN_vkCmdBindIndexBuffer)load(context, "vkCmdBindIndexBuffer");
	vkCmdBindPipeline = (PFN_vkCmdBindPipeline)load(context, "vkCmdBindPipeline");
	vkCmdBindVertexBuffers = (PFN_vkCmdBindVertexBuffers)load(context, "vkCmdBindVertexBuffers");
	vkCmdBlitImage = (PFN_vkCmdBlitImage)load(context, "vkCmdBlitImage");
	vkCmdClearAttachments = (PFN_vkCmdClearAttachments)load(context, "vkCmdClearAttachments");
	vkCmdClearColorImage = (PFN_vkCmdClearColorImage)load(context, "vkCmdClearColorImage");
	vkCmdClearDepthStencilImage = (PFN_vkCmdClearDepthStencilImage)load(context, "vkCmdClearDepthStencilImage");
	vkCmdCopyBuffer = (PFN_vkCmdCopyBuffer)load(context, "vkCmdCopyBuffer");
	vkCmdCopyBufferToImage = (PFN_vkCmdCopyBufferToImage)load(context, "vkCmdCopyBufferToImage");
	vkCmdCopyImage = (PFN_vkCmdCopyImage)load(context, "vkCmdCopyImage");
	vkCmdCopyImageToBuffer = (PFN_vkCmdCopyImageToBuffer)load(context, "vkCmdCopyImageToBuffer");
	vkCmdCopyQueryPoolResults = (PFN_vkCmdCopyQueryPoolResults)load(context, "vkCmdCopyQueryPoolResults");
	vkCmdDispatch = (PFN_vkCmdDispatch)load(context, "vkCmdDispatch");
	vkCmdDispatchIndirect = (PFN_vkCmdDispatchIndirect)load(context, "vkCmdDispatchIndirect");
	vkCmdDraw = (PFN_vkCmdDraw)load(context, "vkCmdDraw");
	vkCmdDrawIndexed = (PFN_vkCmdDrawIndexed)load(context, "vkCmdDrawIndexed");
	vkCmdDrawIndexedIndirect = (PFN_vkCmdDrawIndexedIndirect)load(context, "vkCmdDrawIndexedIndirect");
	vkCmdDrawIndirect = (PFN_vkCmdDrawIndirect)load(context, "vkCmdDrawIndirect");
	vkCmdEndQuery = (PFN_vkCmdEndQuery)load(context, "vkCmdEndQuery");
	vkCmdEndRenderPass = (PFN_vkCmdEndRenderPass)load(context, "vkCmdEndRenderPass");
	vkCmdExecuteCommands = (PFN_vkCmdExecuteCommands)load(context, "vkCmdExecuteCommands");
	vkCmdFillBuffer = (PFN_vkCmdFillBuffer)load(context, "vkCmdFillBuffer");
	vkCmdNextSubpass = (PFN_vkCmdNextSubpass)load(context, "vkCmdNextSubpass");
	vkCmdPipelineBarrier = (PFN_vkCmdPipelineBarrier)load(context, "vkCmdPipelineBarrier");
	vkCmdPushConstants = (PFN_vkCmdPushConstants)load(context, "vkCmdPushConstants");
	vkCmdResetEvent = (PFN_vkCmdResetEvent)load(context, "vkCmdResetEvent");
	vkCmdResetQueryPool = (PFN_vkCmdResetQueryPool)load(context, "vkCmdResetQueryPool");
	vkCmdResolveImage = (PFN_vkCmdResolveImage)load(context, "vkCmdResolveImage");
	vkCmdSetBlendConstants = (PFN_vkCmdSetBlendConstants)load(context, "vkCmdSetBlendConstants");
	vkCmdSetDepthBias = (PFN_vkCmdSetDepthBias)load(context, "vkCmdSetDepthBias");
	vkCmdSetDepthBounds = (PFN_vkCmdSetDepthBounds)load(context, "vkCmdSetDepthBounds");
	vkCmdSetEvent = (PFN_vkCmdSetEvent)load(context, "vkCmdSetEvent");
	vkCmdSetLineWidth = (PFN_vkCmdSetLineWidth)load(context, "vkCmdSetLineWidth");
	vkCmdSetScissor = (PFN_vkCmdSetScissor)load(context, "vkCmdSetScissor");
	vkCmdSetStencilCompareMask = (PFN_vkCmdSetStencilCompareMask)load(context, "vkCmdSetStencilCompareMask");
	vkCmdSetStencilReference = (PFN_vkCmdSetStencilReference)load(context, "vkCmdSetStencilReference");
	vkCmdSetStencilWriteMask = (PFN_vkCmdSetStencilWriteMask)load(context, "vkCmdSetStencilWriteMask");
	vkCmdSetViewport = (PFN_vkCmdSetViewport)load(context, "vkCmdSetViewport");
	vkCmdUpdateBuffer = (PFN_vkCmdUpdateBuffer)load(context, "vkCmdUpdateBuffer");
	vkCmdWaitEvents = (PFN_vkCmdWaitEvents)load(context, "vkCmdWaitEvents");
	vkCmdWriteTimestamp = (PFN_vkCmdWriteTimestamp)load(context, "vkCmdWriteTimestamp");
	vkCreateBuffer = (PFN_vkCreateBuffer)load(context, "vkCreateBuffer");
	vkCreateBufferView = (PFN_vkCreateBufferView)load(context, "vkCreateBufferView");
	vkCreateCommandPool = (PFN_vkCreateCommandPool)load(context, "vkCreateCommandPool");
	vkCreateComputePipelines = (PFN_vkCreateComputePipelines)load(context, "vkCreateComputePipelines");
	vkCreateDescriptorPool = (PFN_vkCreateDescriptorPool)load(context, "vkCreateDescriptorPool");
	vkCreateDescriptorSetLayout = (PFN_vkCreateDescriptorSetLayout)load(context, "vkCreateDescriptorSetLayout");
	vkCreateEvent = (PFN_vkCreateEvent)load(context, "vkCreateEvent");
	vkCreateFence = (PFN_vkCreateFence)load(context, "vkCreateFence");
	vkCreateFramebuffer = (PFN_vkCreateFramebuffer)load(context, "vkCreateFramebuffer");
	vkCreateGraphicsPipelines = (PFN_vkCreateGraphicsPipelines)load(context, "vkCreateGraphicsPipelines");
	vkCreateImage = (PFN_vkCreateImage)load(context, "vkCreateImage");
	vkCreateImageView = (PFN_vkCreateImageView)load(context, "vkCreateImageView");
	vkCreatePipelineCache = (PFN_vkCreatePipelineCache)load(context, "vkCreatePipelineCache");
	vkCreatePipelineLayout = (PFN_vkCreatePipelineLayout)load(context, "vkCreatePipelineLayout");
	vkCreateQueryPool = (PFN_vkCreateQueryPool)load(context, "vkCreateQueryPool");
	vkCreateRenderPass = (PFN_vkCreateRenderPass)load(context, "vkCreateRenderPass");
	vkCreateSampler = (PFN_vkCreateSampler)load(context, "vkCreateSampler");
	vkCreateSemaphore = (PFN_vkCreateSemaphore)load(context, "vkCreateSemaphore");
	vkCreateShaderModule = (PFN_vkCreateShaderModule)load(context, "vkCreateShaderModule");
	vkDestroyBuffer = (PFN_vkDestroyBuffer)load(context, "vkDestroyBuffer");
	vkDestroyBufferView = (PFN_vkDestroyBufferView)load(context, "vkDestroyBufferView");
	vkDestroyCommandPool = (PFN_vkDestroyCommandPool)load(context, "vkDestroyCommandPool");
	vkDestroyDescriptorPool = (PFN_vkDestroyDescriptorPool)load(context, "vkDestroyDescriptorPool");
	vkDestroyDescriptorSetLayout = (PFN_vkDestroyDescriptorSetLayout)load(context, "vkDestroyDescriptorSetLayout");
	vkDestroyDevice = (PFN_vkDestroyDevice)load(context, "vkDestroyDevice");
	vkDestroyEvent = (PFN_vkDestroyEvent)load(context, "vkDestroyEvent");
	vkDestroyFence = (PFN_vkDestroyFence)load(context, "vkDestroyFence");
	vkDestroyFramebuffer = (PFN_vkDestroyFramebuffer)load(context, "vkDestroyFramebuffer");
	vkDestroyImage = (PFN_vkDestroyImage)load(context, "vkDestroyImage");
	vkDestroyImageView = (PFN_vkDestroyImageView)load(context, "vkDestroyImageView");
	vkDestroyPipeline = (PFN_vkDestroyPipeline)load(context, "vkDestroyPipeline");
	vkDestroyPipelineCache = (PFN_vkDestroyPipelineCache)load(context, "vkDestroyPipelineCache");
	vkDestroyPipelineLayout = (PFN_vkDestroyPipelineLayout)load(context, "vkDestroyPipelineLayout");
	vkDestroyQueryPool = (PFN_vkDestroyQueryPool)load(context, "vkDestroyQueryPool");
	vkDestroyRenderPass = (PFN_vkDestroyRenderPass)load(context, "vkDestroyRenderPass");
	vkDestroySampler = (PFN_vkDestroySampler)load(context, "vkDestroySampler");
	vkDestroySemaphore = (PFN_vkDestroySemaphore)load(context, "vkDestroySemaphore");
	vkDestroyShaderModule = (PFN_vkDestroyShaderModule)load(context, "vkDestroyShaderModule");
	vkDeviceWaitIdle = (PFN_vkDeviceWaitIdle)load(context, "vkDeviceWaitIdle");
	vkEndCommandBuffer = (PFN_vkEndCommandBuffer)load(context, "vkEndCommandBuffer");
	vkFlushMappedMemoryRanges = (PFN_vkFlushMappedMemoryRanges)load(context, "vkFlushMappedMemoryRanges");
	vkFreeCommandBuffers = (PFN_vkFreeCommandBuffers)load(context, "vkFreeCommandBuffers");
	vkFreeDescriptorSets = (PFN_vkFreeDescriptorSets)load(context, "vkFreeDescriptorSets");
	vkFreeMemory = (PFN_vkFreeMemory)load(context, "vkFreeMemory");
	vkGetBufferMemoryRequirements = (PFN_vkGetBufferMemoryRequirements)load(context, "vkGetBufferMemoryRequirements");
	vkGetDeviceMemoryCommitment = (PFN_vkGetDeviceMemoryCommitment)load(context, "vkGetDeviceMemoryCommitment");
	vkGetDeviceQueue = (PFN_vkGetDeviceQueue)load(context, "vkGetDeviceQueue");
	vkGetEventStatus = (PFN_vkGetEventStatus)load(context, "vkGetEventStatus");
	vkGetFenceStatus = (PFN_vkGetFenceStatus)load(context, "vkGetFenceStatus");
	vkGetImageMemoryRequirements = (PFN_vkGetImageMemoryRequirements)load(context, "vkGetImageMemoryRequirements");
	vkGetImageSparseMemoryRequirements = (PFN_vkGetImageSparseMemoryRequirements)load(context, "vkGetImageSparseMemoryRequirements");
	vkGetImageSubresourceLayout = (PFN_vkGetImageSubresourceLayout)load(context, "vkGetImageSubresourceLayout");
	vkGetPipelineCacheData = (PFN_vkGetPipelineCacheData)load(context, "vkGetPipelineCacheData");
	vkGetQueryPoolResults = (PFN_vkGetQueryPoolResults)load(context, "vkGetQueryPoolResults");
	vkGetRenderAreaGranularity = (PFN_vkGetRenderAreaGranularity)load(context, "vkGetRenderAreaGranularity");
	vkInvalidateMappedMemoryRanges = (PFN_vkInvalidateMappedMemoryRanges)load(context, "vkInvalidateMappedMemoryRanges");
	vkMapMemory = (PFN_vkMapMemory)load(context, "vkMapMemory");
	vkMergePipelineCaches = (PFN_vkMergePipelineCaches)load(context, "vkMergePipelineCaches");
	vkQueueBindSparse = (PFN_vkQueueBindSparse)load(context, "vkQueueBindSparse");
	vkQueueSubmit = (PFN_vkQueueSubmit)load(context, "vkQueueSubmit");
	vkQueueWaitIdle = (PFN_vkQueueWaitIdle)load(context, "vkQueueWaitIdle");
	vkResetCommandBuffer = (PFN_vkResetCommandBuffer)load(context, "vkResetCommandBuffer");
	vkResetCommandPool = (PFN_vkResetCommandPool)load(context, "vkResetCommandPool");
	vkResetDescriptorPool = (PFN_vkResetDescriptorPool)load(context, "vkResetDescriptorPool");
	vkResetEvent = (PFN_vkResetEvent)load(context, "vkResetEvent");
	vkResetFences = (PFN_vkResetFences)load(context, "vkResetFences");
	vkSetEvent = (PFN_vkSetEvent)load(context, "vkSetEvent");
	vkUnmapMemory = (PFN_vkUnmapMemory)load(context, "vkUnmapMemory");
	vkUpdateDescriptorSets = (PFN_vkUpdateDescriptorSets)load(context, "vkUpdateDescriptorSets");
	vkWaitForFences = (PFN_vkWaitForFences)load(context, "vkWaitForFences");
#endif /* defined(VK_VERSION_1_0) */
#if defined(VK_VERSION_1_1)
	vkBindBufferMemory2 = (PFN_vkBindBufferMemory2)load(context, "vkBindBufferMemory2");
	vkBindImageMemory2 = (PFN_vkBindImageMemory2)load(context, "vkBindImageMemory2");
	vkCmdDispatchBase = (PFN_vkCmdDispatchBase)load(context, "vkCmdDispatchBase");
	vkCmdSetDeviceMask = (PFN_vkCmdSetDeviceMask)load(context, "vkCmdSetDeviceMask");
	vkCreateDescriptorUpdateTemplate = (PFN_vkCreateDescriptorUpdateTemplate)load(context, "vkCreateDescriptorUpdateTemplate");
	vkCreateSamplerYcbcrConversion = (PFN_vkCreateSamplerYcbcrConversion)load(context, "vkCreateSamplerYcbcrConversion");
	vkDestroyDescriptorUpdateTemplate = (PFN_vkDestroyDescriptorUpdateTemplate)load(context, "vkDestroyDescriptorUpdateTemplate");
	vkDestroySamplerYcbcrConversion = (PFN_vkDestroySamplerYcbcrConversion)load(context, "vkDestroySamplerYcbcrConversion");
	vkGetBufferMemoryRequirements2 = (PFN_vkGetBufferMemoryRequirements2)load(context, "vkGetBufferMemoryRequirements2");
	vkGetDescriptorSetLayoutSupport = (PFN_vkGetDescriptorSetLayoutSupport)load(context, "vkGetDescriptorSetLayoutSupport");
	vkGetDeviceGroupPeerMemoryFeatures = (PFN_vkGetDeviceGroupPeerMemoryFeatures)load(context, "vkGetDeviceGroupPeerMemoryFeatures");
	vkGetDeviceQueue2 = (PFN_vkGetDeviceQueue2)load(context, "vkGetDeviceQueue2");
	vkGetImageMemoryRequirements2 = (PFN_vkGetImageMemoryRequirements2)load(context, "vkGetImageMemoryRequirements2");
	vkGetImageSparseMemoryRequirements2 = (PFN_vkGetImageSparseMemoryRequirements2)load(context, "vkGetImageSparseMemoryRequirements2");
	vkTrimCommandPool = (PFN_vkTrimCommandPool)load(context, "vkTrimCommandPool");
	vkUpdateDescriptorSetWithTemplate = (PFN_vkUpdateDescriptorSetWithTemplate)load(context, "vkUpdateDescriptorSetWithTemplate");
#endif /* defined(VK_VERSION_1_1) */
#if defined(VK_VERSION_1_2)
	vkCmdBeginRenderPass2 = (PFN_vkCmdBeginRenderPass2)load(context, "vkCmdBeginRenderPass2");
	vkCmdDrawIndexedIndirectCount = (PFN_vkCmdDrawIndexedIndirectCount)load(context, "vkCmdDrawIndexedIndirectCount");
	vkCmdDrawIndirectCount = (PFN_vkCmdDrawIndirectCount)load(context, "vkCmdDrawIndirectCount");
	vkCmdEndRenderPass2 = (PFN_vkCmdEndRenderPass2)load(context, "vkCmdEndRenderPass2");
	vkCmdNextSubpass2 = (PFN_vkCmdNextSubpass2)load(context, "vkCmdNextSubpass2");
	vkCreateRenderPass2 = (PFN_vkCreateRenderPass2)load(context, "vkCreateRenderPass2");
	vkGetBufferDeviceAddress = (PFN_vkGetBufferDeviceAddress)load(context, "vkGetBufferDeviceAddress");
	vkGetBufferOpaqueCaptureAddress = (PFN_vkGetBufferOpaqueCaptureAddress)load(context, "vkGetBufferOpaqueCaptureAddress");
	vkGetDeviceMemoryOpaqueCaptureAddress = (PFN_vkGetDeviceMemoryOpaqueCaptureAddress)load(context, "vkGetDeviceMemoryOpaqueCaptureAddress");
	vkGetSemaphoreCounterValue = (PFN_vkGetSemaphoreCounterValue)load(context, "vkGetSemaphoreCounterValue");
	vkResetQueryPool = (PFN_vkResetQueryPool)load(context, "vkResetQueryPool");
	vkSignalSemaphore = (PFN_vkSignalSemaphore)load(context, "vkSignalSemaphore");
	vkWaitSemaphores = (PFN_vkWaitSemaphores)load(context, "vkWaitSemaphores");
#endif /* defined(VK_VERSION_1_2) */
#if defined(VK_VERSION_1_3)
	vkCmdBeginRendering = (PFN_vkCmdBeginRendering)load(context, "vkCmdBeginRendering");
	vkCmdBindVertexBuffers2 = (PFN_vkCmdBindVertexBuffers2)load(context, "vkCmdBindVertexBuffers2");
	vkCmdBlitImage2 = (PFN_vkCmdBlitImage2)load(context, "vkCmdBlitImage2");
	vkCmdCopyBuffer2 = (PFN_vkCmdCopyBuffer2)load(context, "vkCmdCopyBuffer2");
	vkCmdCopyBufferToImage2 = (PFN_vkCmdCopyBufferToImage2)load(context, "vkCmdCopyBufferToImage2");
	vkCmdCopyImage2 = (PFN_vkCmdCopyImage2)load(context, "vkCmdCopyImage2");
	vkCmdCopyImageToBuffer2 = (PFN_vkCmdCopyImageToBuffer2)load(context, "vkCmdCopyImageToBuffer2");
	vkCmdEndRendering = (PFN_vkCmdEndRendering)load(context, "vkCmdEndRendering");
	vkCmdPipelineBarrier2 = (PFN_vkCmdPipelineBarrier2)load(context, "vkCmdPipelineBarrier2");
	vkCmdResetEvent2 = (PFN_vkCmdResetEvent2)load(context, "vkCmdResetEvent2");
	vkCmdResolveImage2 = (PFN_vkCmdResolveImage2)load(context, "vkCmdResolveImage2");
	vkCmdSetCullMode = (PFN_vkCmdSetCullMode)load(context, "vkCmdSetCullMode");
	vkCmdSetDepthBiasEnable = (PFN_vkCmdSetDepthBiasEnable)load(context, "vkCmdSetDepthBiasEnable");
	vkCmdSetDepthBoundsTestEnable = (PFN_vkCmdSetDepthBoundsTestEnable)load(context, "vkCmdSetDepthBoundsTestEnable");
	vkCmdSetDepthCompareOp = (PFN_vkCmdSetDepthCompareOp)load(context, "vkCmdSetDepthCompareOp");
	vkCmdSetDepthTestEnable = (PFN_vkCmdSetDepthTestEnable)load(context, "vkCmdSetDepthTestEnable");
	vkCmdSetDepthWriteEnable = (PFN_vkCmdSetDepthWriteEnable)load(context, "vkCmdSetDepthWriteEnable");
	vkCmdSetEvent2 = (PFN_vkCmdSetEvent2)load(context, "vkCmdSetEvent2");
	vkCmdSetFrontFace = (PFN_vkCmdSetFrontFace)load(context, "vkCmdSetFrontFace");
	vkCmdSetPrimitiveRestartEnable = (PFN_vkCmdSetPrimitiveRestartEnable)load(context, "vkCmdSetPrimitiveRestartEnable");
	vkCmdSetPrimitiveTopology = (PFN_vkCmdSetPrimitiveTopology)load(context, "vkCmdSetPrimitiveTopology");
	vkCmdSetRasterizerDiscardEnable = (PFN_vkCmdSetRasterizerDiscardEnable)load(context, "vkCmdSetRasterizerDiscardEnable");
	vkCmdSetScissorWithCount = (PFN_vkCmdSetScissorWithCount)load(context, "vkCmdSetScissorWithCount");
	vkCmdSetStencilOp = (PFN_vkCmdSetStencilOp)load(context, "vkCmdSetStencilOp");
	vkCmdSetStencilTestEnable = (PFN_vkCmdSetStencilTestEnable)load(context, "vkCmdSetStencilTestEnable");
	vkCmdSetViewportWithCount = (PFN_vkCmdSetViewportWithCount)load(context, "vkCmdSetViewportWithCount");
	vkCmdWaitEvents2 = (PFN_vkCmdWaitEvents2)load(context, "vkCmdWaitEvents2");
	vkCmdWriteTimestamp2 = (PFN_vkCmdWriteTimestamp2)load(context, "vkCmdWriteTimestamp2");
	vkCreatePrivateDataSlot = (PFN_vkCreatePrivateDataSlot)load(context, "vkCreatePrivateDataSlot");
	vkDestroyPrivateDataSlot = (PFN_vkDestroyPrivateDataSlot)load(context, "vkDestroyPrivateDataSlot");
	vkGetDeviceBufferMemoryRequirements = (PFN_vkGetDeviceBufferMemoryRequirements)load(context, "vkGetDeviceBufferMemoryRequirements");
	vkGetDeviceImageMemoryRequirements = (PFN_vkGetDeviceImageMemoryRequirements)load(context, "vkGetDeviceImageMemoryRequirements");
	vkGetDeviceImageSparseMemoryRequirements = (PFN_vkGetDeviceImageSparseMemoryRequirements)load(context, "vkGetDeviceImageSparseMemoryRequirements");
	vkGetPrivateData = (PFN_vkGetPrivateData)load(context, "vkGetPrivateData");
	vkQueueSubmit2 = (PFN_vkQueueSubmit2)load(context, "vkQueueSubmit2");
	vkSetPrivateData = (PFN_vkSetPrivateData)load(context, "vkSetPrivateData");
#endif /* defined(VK_VERSION_1_3) */
#if defined(VK_AMDX_shader_enqueue)
	vkCmdDispatchGraphAMDX = (PFN_vkCmdDispatchGraphAMDX)load(context, "vkCmdDispatchGraphAMDX");
	vkCmdDispatchGraphIndirectAMDX = (PFN_vkCmdDispatchGraphIndirectAMDX)load(context, "vkCmdDispatchGraphIndirectAMDX");
	vkCmdDispatchGraphIndirectCountAMDX = (PFN_vkCmdDispatchGraphIndirectCountAMDX)load(context, "vkCmdDispatchGraphIndirectCountAMDX");
	vkCmdInitializeGraphScratchMemoryAMDX = (PFN_vkCmdInitializeGraphScratchMemoryAMDX)load(context, "vkCmdInitializeGraphScratchMemoryAMDX");
	vkCreateExecutionGraphPipelinesAMDX = (PFN_vkCreateExecutionGraphPipelinesAMDX)load(context, "vkCreateExecutionGraphPipelinesAMDX");
	vkGetExecutionGraphPipelineNodeIndexAMDX = (PFN_vkGetExecutionGraphPipelineNodeIndexAMDX)load(context, "vkGetExecutionGraphPipelineNodeIndexAMDX");
	vkGetExecutionGraphPipelineScratchSizeAMDX = (PFN_vkGetExecutionGraphPipelineScratchSizeAMDX)load(context, "vkGetExecutionGraphPipelineScratchSizeAMDX");
#endif /* defined(VK_AMDX_shader_enqueue) */
#if defined(VK_AMD_buffer_marker)
	vkCmdWriteBufferMarkerAMD = (PFN_vkCmdWriteBufferMarkerAMD)load(context, "vkCmdWriteBufferMarkerAMD");
#endif /* defined(VK_AMD_buffer_marker) */
#if defined(VK_AMD_display_native_hdr)
	vkSetLocalDimmingAMD = (PFN_vkSetLocalDimmingAMD)load(context, "vkSetLocalDimmingAMD");
#endif /* defined(VK_AMD_display_native_hdr) */
#if defined(VK_AMD_draw_indirect_count)
	vkCmdDrawIndexedIndirectCountAMD = (PFN_vkCmdDrawIndexedIndirectCountAMD)load(context, "vkCmdDrawIndexedIndirectCountAMD");
	vkCmdDrawIndirectCountAMD = (PFN_vkCmdDrawIndirectCountAMD)load(context, "vkCmdDrawIndirectCountAMD");
#endif /* defined(VK_AMD_draw_indirect_count) */
#if defined(VK_AMD_shader_info)
	vkGetShaderInfoAMD = (PFN_vkGetShaderInfoAMD)load(context, "vkGetShaderInfoAMD");
#endif /* defined(VK_AMD_shader_info) */
#if defined(VK_ANDROID_external_memory_android_hardware_buffer)
	vkGetAndroidHardwareBufferPropertiesANDROID = (PFN_vkGetAndroidHardwareBufferPropertiesANDROID)load(context, "vkGetAndroidHardwareBufferPropertiesANDROID");
	vkGetMemoryAndroidHardwareBufferANDROID = (PFN_vkGetMemoryAndroidHardwareBufferANDROID)load(context, "vkGetMemoryAndroidHardwareBufferANDROID");
#endif /* defined(VK_ANDROID_external_memory_android_hardware_buffer) */
#if defined(VK_EXT_attachment_feedback_loop_dynamic_state)
	vkCmdSetAttachmentFeedbackLoopEnableEXT = (PFN_vkCmdSetAttachmentFeedbackLoopEnableEXT)load(context, "vkCmdSetAttachmentFeedbackLoopEnableEXT");
#endif /* defined(VK_EXT_attachment_feedback_loop_dynamic_state) */
#if defined(VK_EXT_buffer_device_address)
	vkGetBufferDeviceAddressEXT = (PFN_vkGetBufferDeviceAddressEXT)load(context, "vkGetBufferDeviceAddressEXT");
#endif /* defined(VK_EXT_buffer_device_address) */
#if defined(VK_EXT_calibrated_timestamps)
	vkGetCalibratedTimestampsEXT = (PFN_vkGetCalibratedTimestampsEXT)load(context, "vkGetCalibratedTimestampsEXT");
#endif /* defined(VK_EXT_calibrated_timestamps) */
#if defined(VK_EXT_color_write_enable)
	vkCmdSetColorWriteEnableEXT = (PFN_vkCmdSetColorWriteEnableEXT)load(context, "vkCmdSetColorWriteEnableEXT");
#endif /* defined(VK_EXT_color_write_enable) */
#if defined(VK_EXT_conditional_rendering)
	vkCmdBeginConditionalRenderingEXT = (PFN_vkCmdBeginConditionalRenderingEXT)load(context, "vkCmdBeginConditionalRenderingEXT");
	vkCmdEndConditionalRenderingEXT = (PFN_vkCmdEndConditionalRenderingEXT)load(context, "vkCmdEndConditionalRenderingEXT");
#endif /* defined(VK_EXT_conditional_rendering) */
#if defined(VK_EXT_debug_marker)
	vkCmdDebugMarkerBeginEXT = (PFN_vkCmdDebugMarkerBeginEXT)load(context, "vkCmdDebugMarkerBeginEXT");
	vkCmdDebugMarkerEndEXT = (PFN_vkCmdDebugMarkerEndEXT)load(context, "vkCmdDebugMarkerEndEXT");
	vkCmdDebugMarkerInsertEXT = (PFN_vkCmdDebugMarkerInsertEXT)load(context, "vkCmdDebugMarkerInsertEXT");
	vkDebugMarkerSetObjectNameEXT = (PFN_vkDebugMarkerSetObjectNameEXT)load(context, "vkDebugMarkerSetObjectNameEXT");
	vkDebugMarkerSetObjectTagEXT = (PFN_vkDebugMarkerSetObjectTagEXT)load(context, "vkDebugMarkerSetObjectTagEXT");
#endif /* defined(VK_EXT_debug_marker) */
#if defined(VK_EXT_depth_bias_control)
	vkCmdSetDepthBias2EXT = (PFN_vkCmdSetDepthBias2EXT)load(context, "vkCmdSetDepthBias2EXT");
#endif /* defined(VK_EXT_depth_bias_control) */
#if defined(VK_EXT_descriptor_buffer)
	vkCmdBindDescriptorBufferEmbeddedSamplersEXT = (PFN_vkCmdBindDescriptorBufferEmbeddedSamplersEXT)load(context, "vkCmdBindDescriptorBufferEmbeddedSamplersEXT");
	vkCmdBindDescriptorBuffersEXT = (PFN_vkCmdBindDescriptorBuffersEXT)load(context, "vkCmdBindDescriptorBuffersEXT");
	vkCmdSetDescriptorBufferOffsetsEXT = (PFN_vkCmdSetDescriptorBufferOffsetsEXT)load(context, "vkCmdSetDescriptorBufferOffsetsEXT");
	vkGetBufferOpaqueCaptureDescriptorDataEXT = (PFN_vkGetBufferOpaqueCaptureDescriptorDataEXT)load(context, "vkGetBufferOpaqueCaptureDescriptorDataEXT");
	vkGetDescriptorEXT = (PFN_vkGetDescriptorEXT)load(context, "vkGetDescriptorEXT");
	vkGetDescriptorSetLayoutBindingOffsetEXT = (PFN_vkGetDescriptorSetLayoutBindingOffsetEXT)load(context, "vkGetDescriptorSetLayoutBindingOffsetEXT");
	vkGetDescriptorSetLayoutSizeEXT = (PFN_vkGetDescriptorSetLayoutSizeEXT)load(context, "vkGetDescriptorSetLayoutSizeEXT");
	vkGetImageOpaqueCaptureDescriptorDataEXT = (PFN_vkGetImageOpaqueCaptureDescriptorDataEXT)load(context, "vkGetImageOpaqueCaptureDescriptorDataEXT");
	vkGetImageViewOpaqueCaptureDescriptorDataEXT = (PFN_vkGetImageViewOpaqueCaptureDescriptorDataEXT)load(context, "vkGetImageViewOpaqueCaptureDescriptorDataEXT");
	vkGetSamplerOpaqueCaptureDescriptorDataEXT = (PFN_vkGetSamplerOpaqueCaptureDescriptorDataEXT)load(context, "vkGetSamplerOpaqueCaptureDescriptorDataEXT");
#endif /* defined(VK_EXT_descriptor_buffer) */
#if defined(VK_EXT_descriptor_buffer) && (defined(VK_KHR_acceleration_structure) || defined(VK_NV_ray_tracing))
	vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT = (PFN_vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT)load(context, "vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT");
#endif /* defined(VK_EXT_descriptor_buffer) && (defined(VK_KHR_acceleration_structure) || defined(VK_NV_ray_tracing)) */
#if defined(VK_EXT_device_fault)
	vkGetDeviceFaultInfoEXT = (PFN_vkGetDeviceFaultInfoEXT)load(context, "vkGetDeviceFaultInfoEXT");
#endif /* defined(VK_EXT_device_fault) */
#if defined(VK_EXT_discard_rectangles)
	vkCmdSetDiscardRectangleEXT = (PFN_vkCmdSetDiscardRectangleEXT)load(context, "vkCmdSetDiscardRectangleEXT");
#endif /* defined(VK_EXT_discard_rectangles) */
#if defined(VK_EXT_discard_rectangles) && VK_EXT_DISCARD_RECTANGLES_SPEC_VERSION >= 2
	vkCmdSetDiscardRectangleEnableEXT = (PFN_vkCmdSetDiscardRectangleEnableEXT)load(context, "vkCmdSetDiscardRectangleEnableEXT");
	vkCmdSetDiscardRectangleModeEXT = (PFN_vkCmdSetDiscardRectangleModeEXT)load(context, "vkCmdSetDiscardRectangleModeEXT");
#endif /* defined(VK_EXT_discard_rectangles) && VK_EXT_DISCARD_RECTANGLES_SPEC_VERSION >= 2 */
#if defined(VK_EXT_display_control)
	vkDisplayPowerControlEXT = (PFN_vkDisplayPowerControlEXT)load(context, "vkDisplayPowerControlEXT");
	vkGetSwapchainCounterEXT = (PFN_vkGetSwapchainCounterEXT)load(context, "vkGetSwapchainCounterEXT");
	vkRegisterDeviceEventEXT = (PFN_vkRegisterDeviceEventEXT)load(context, "vkRegisterDeviceEventEXT");
	vkRegisterDisplayEventEXT = (PFN_vkRegisterDisplayEventEXT)load(context, "vkRegisterDisplayEventEXT");
#endif /* defined(VK_EXT_display_control) */
#if defined(VK_EXT_external_memory_host)
	vkGetMemoryHostPointerPropertiesEXT = (PFN_vkGetMemoryHostPointerPropertiesEXT)load(context, "vkGetMemoryHostPointerPropertiesEXT");
#endif /* defined(VK_EXT_external_memory_host) */
#if defined(VK_EXT_full_screen_exclusive)
	vkAcquireFullScreenExclusiveModeEXT = (PFN_vkAcquireFullScreenExclusiveModeEXT)load(context, "vkAcquireFullScreenExclusiveModeEXT");
	vkReleaseFullScreenExclusiveModeEXT = (PFN_vkReleaseFullScreenExclusiveModeEXT)load(context, "vkReleaseFullScreenExclusiveModeEXT");
#endif /* defined(VK_EXT_full_screen_exclusive) */
#if defined(VK_EXT_hdr_metadata)
	vkSetHdrMetadataEXT = (PFN_vkSetHdrMetadataEXT)load(context, "vkSetHdrMetadataEXT");
#endif /* defined(VK_EXT_hdr_metadata) */
#if defined(VK_EXT_host_image_copy)
	vkCopyImageToImageEXT = (PFN_vkCopyImageToImageEXT)load(context, "vkCopyImageToImageEXT");
	vkCopyImageToMemoryEXT = (PFN_vkCopyImageToMemoryEXT)load(context, "vkCopyImageToMemoryEXT");
	vkCopyMemoryToImageEXT = (PFN_vkCopyMemoryToImageEXT)load(context, "vkCopyMemoryToImageEXT");
	vkTransitionImageLayoutEXT = (PFN_vkTransitionImageLayoutEXT)load(context, "vkTransitionImageLayoutEXT");
#endif /* defined(VK_EXT_host_image_copy) */
#if defined(VK_EXT_host_query_reset)
	vkResetQueryPoolEXT = (PFN_vkResetQueryPoolEXT)load(context, "vkResetQueryPoolEXT");
#endif /* defined(VK_EXT_host_query_reset) */
#if defined(VK_EXT_image_drm_format_modifier)
	vkGetImageDrmFormatModifierPropertiesEXT = (PFN_vkGetImageDrmFormatModifierPropertiesEXT)load(context, "vkGetImageDrmFormatModifierPropertiesEXT");
#endif /* defined(VK_EXT_image_drm_format_modifier) */
#if defined(VK_EXT_line_rasterization)
	vkCmdSetLineStippleEXT = (PFN_vkCmdSetLineStippleEXT)load(context, "vkCmdSetLineStippleEXT");
#endif /* defined(VK_EXT_line_rasterization) */
#if defined(VK_EXT_mesh_shader)
	vkCmdDrawMeshTasksEXT = (PFN_vkCmdDrawMeshTasksEXT)load(context, "vkCmdDrawMeshTasksEXT");
	vkCmdDrawMeshTasksIndirectCountEXT = (PFN_vkCmdDrawMeshTasksIndirectCountEXT)load(context, "vkCmdDrawMeshTasksIndirectCountEXT");
	vkCmdDrawMeshTasksIndirectEXT = (PFN_vkCmdDrawMeshTasksIndirectEXT)load(context, "vkCmdDrawMeshTasksIndirectEXT");
#endif /* defined(VK_EXT_mesh_shader) */
#if defined(VK_EXT_metal_objects)
	vkExportMetalObjectsEXT = (PFN_vkExportMetalObjectsEXT)load(context, "vkExportMetalObjectsEXT");
#endif /* defined(VK_EXT_metal_objects) */
#if defined(VK_EXT_multi_draw)
	vkCmdDrawMultiEXT = (PFN_vkCmdDrawMultiEXT)load(context, "vkCmdDrawMultiEXT");
	vkCmdDrawMultiIndexedEXT = (PFN_vkCmdDrawMultiIndexedEXT)load(context, "vkCmdDrawMultiIndexedEXT");
#endif /* defined(VK_EXT_multi_draw) */
#if defined(VK_EXT_opacity_micromap)
	vkBuildMicromapsEXT = (PFN_vkBuildMicromapsEXT)load(context, "vkBuildMicromapsEXT");
	vkCmdBuildMicromapsEXT = (PFN_vkCmdBuildMicromapsEXT)load(context, "vkCmdBuildMicromapsEXT");
	vkCmdCopyMemoryToMicromapEXT = (PFN_vkCmdCopyMemoryToMicromapEXT)load(context, "vkCmdCopyMemoryToMicromapEXT");
	vkCmdCopyMicromapEXT = (PFN_vkCmdCopyMicromapEXT)load(context, "vkCmdCopyMicromapEXT");
	vkCmdCopyMicromapToMemoryEXT = (PFN_vkCmdCopyMicromapToMemoryEXT)load(context, "vkCmdCopyMicromapToMemoryEXT");
	vkCmdWriteMicromapsPropertiesEXT = (PFN_vkCmdWriteMicromapsPropertiesEXT)load(context, "vkCmdWriteMicromapsPropertiesEXT");
	vkCopyMemoryToMicromapEXT = (PFN_vkCopyMemoryToMicromapEXT)load(context, "vkCopyMemoryToMicromapEXT");
	vkCopyMicromapEXT = (PFN_vkCopyMicromapEXT)load(context, "vkCopyMicromapEXT");
	vkCopyMicromapToMemoryEXT = (PFN_vkCopyMicromapToMemoryEXT)load(context, "vkCopyMicromapToMemoryEXT");
	vkCreateMicromapEXT = (PFN_vkCreateMicromapEXT)load(context, "vkCreateMicromapEXT");
	vkDestroyMicromapEXT = (PFN_vkDestroyMicromapEXT)load(context, "vkDestroyMicromapEXT");
	vkGetDeviceMicromapCompatibilityEXT = (PFN_vkGetDeviceMicromapCompatibilityEXT)load(context, "vkGetDeviceMicromapCompatibilityEXT");
	vkGetMicromapBuildSizesEXT = (PFN_vkGetMicromapBuildSizesEXT)load(context, "vkGetMicromapBuildSizesEXT");
	vkWriteMicromapsPropertiesEXT = (PFN_vkWriteMicromapsPropertiesEXT)load(context, "vkWriteMicromapsPropertiesEXT");
#endif /* defined(VK_EXT_opacity_micromap) */
#if defined(VK_EXT_pageable_device_local_memory)
	vkSetDeviceMemoryPriorityEXT = (PFN_vkSetDeviceMemoryPriorityEXT)load(context, "vkSetDeviceMemoryPriorityEXT");
#endif /* defined(VK_EXT_pageable_device_local_memory) */
#if defined(VK_EXT_pipeline_properties)
	vkGetPipelinePropertiesEXT = (PFN_vkGetPipelinePropertiesEXT)load(context, "vkGetPipelinePropertiesEXT");
#endif /* defined(VK_EXT_pipeline_properties) */
#if defined(VK_EXT_private_data)
	vkCreatePrivateDataSlotEXT = (PFN_vkCreatePrivateDataSlotEXT)load(context, "vkCreatePrivateDataSlotEXT");
	vkDestroyPrivateDataSlotEXT = (PFN_vkDestroyPrivateDataSlotEXT)load(context, "vkDestroyPrivateDataSlotEXT");
	vkGetPrivateDataEXT = (PFN_vkGetPrivateDataEXT)load(context, "vkGetPrivateDataEXT");
	vkSetPrivateDataEXT = (PFN_vkSetPrivateDataEXT)load(context, "vkSetPrivateDataEXT");
#endif /* defined(VK_EXT_private_data) */
#if defined(VK_EXT_sample_locations)
	vkCmdSetSampleLocationsEXT = (PFN_vkCmdSetSampleLocationsEXT)load(context, "vkCmdSetSampleLocationsEXT");
#endif /* defined(VK_EXT_sample_locations) */
#if defined(VK_EXT_shader_module_identifier)
	vkGetShaderModuleCreateInfoIdentifierEXT = (PFN_vkGetShaderModuleCreateInfoIdentifierEXT)load(context, "vkGetShaderModuleCreateInfoIdentifierEXT");
	vkGetShaderModuleIdentifierEXT = (PFN_vkGetShaderModuleIdentifierEXT)load(context, "vkGetShaderModuleIdentifierEXT");
#endif /* defined(VK_EXT_shader_module_identifier) */
#if defined(VK_EXT_shader_object)
	vkCmdBindShadersEXT = (PFN_vkCmdBindShadersEXT)load(context, "vkCmdBindShadersEXT");
	vkCreateShadersEXT = (PFN_vkCreateShadersEXT)load(context, "vkCreateShadersEXT");
	vkDestroyShaderEXT = (PFN_vkDestroyShaderEXT)load(context, "vkDestroyShaderEXT");
	vkGetShaderBinaryDataEXT = (PFN_vkGetShaderBinaryDataEXT)load(context, "vkGetShaderBinaryDataEXT");
#endif /* defined(VK_EXT_shader_object) */
#if defined(VK_EXT_swapchain_maintenance1)
	vkReleaseSwapchainImagesEXT = (PFN_vkReleaseSwapchainImagesEXT)load(context, "vkReleaseSwapchainImagesEXT");
#endif /* defined(VK_EXT_swapchain_maintenance1) */
#if defined(VK_EXT_transform_feedback)
	vkCmdBeginQueryIndexedEXT = (PFN_vkCmdBeginQueryIndexedEXT)load(context, "vkCmdBeginQueryIndexedEXT");
	vkCmdBeginTransformFeedbackEXT = (PFN_vkCmdBeginTransformFeedbackEXT)load(context, "vkCmdBeginTransformFeedbackEXT");
	vkCmdBindTransformFeedbackBuffersEXT = (PFN_vkCmdBindTransformFeedbackBuffersEXT)load(context, "vkCmdBindTransformFeedbackBuffersEXT");
	vkCmdDrawIndirectByteCountEXT = (PFN_vkCmdDrawIndirectByteCountEXT)load(context, "vkCmdDrawIndirectByteCountEXT");
	vkCmdEndQueryIndexedEXT = (PFN_vkCmdEndQueryIndexedEXT)load(context, "vkCmdEndQueryIndexedEXT");
	vkCmdEndTransformFeedbackEXT = (PFN_vkCmdEndTransformFeedbackEXT)load(context, "vkCmdEndTransformFeedbackEXT");
#endif /* defined(VK_EXT_transform_feedback) */
#if defined(VK_EXT_validation_cache)
	vkCreateValidationCacheEXT = (PFN_vkCreateValidationCacheEXT)load(context, "vkCreateValidationCacheEXT");
	vkDestroyValidationCacheEXT = (PFN_vkDestroyValidationCacheEXT)load(context, "vkDestroyValidationCacheEXT");
	vkGetValidationCacheDataEXT = (PFN_vkGetValidationCacheDataEXT)load(context, "vkGetValidationCacheDataEXT");
	vkMergeValidationCachesEXT = (PFN_vkMergeValidationCachesEXT)load(context, "vkMergeValidationCachesEXT");
#endif /* defined(VK_EXT_validation_cache) */
#if defined(VK_FUCHSIA_buffer_collection)
	vkCreateBufferCollectionFUCHSIA = (PFN_vkCreateBufferCollectionFUCHSIA)load(context, "vkCreateBufferCollectionFUCHSIA");
	vkDestroyBufferCollectionFUCHSIA = (PFN_vkDestroyBufferCollectionFUCHSIA)load(context, "vkDestroyBufferCollectionFUCHSIA");
	vkGetBufferCollectionPropertiesFUCHSIA = (PFN_vkGetBufferCollectionPropertiesFUCHSIA)load(context, "vkGetBufferCollectionPropertiesFUCHSIA");
	vkSetBufferCollectionBufferConstraintsFUCHSIA = (PFN_vkSetBufferCollectionBufferConstraintsFUCHSIA)load(context, "vkSetBufferCollectionBufferConstraintsFUCHSIA");
	vkSetBufferCollectionImageConstraintsFUCHSIA = (PFN_vkSetBufferCollectionImageConstraintsFUCHSIA)load(context, "vkSetBufferCollectionImageConstraintsFUCHSIA");
#endif /* defined(VK_FUCHSIA_buffer_collection) */
#if defined(VK_FUCHSIA_external_memory)
	vkGetMemoryZirconHandleFUCHSIA = (PFN_vkGetMemoryZirconHandleFUCHSIA)load(context, "vkGetMemoryZirconHandleFUCHSIA");
	vkGetMemoryZirconHandlePropertiesFUCHSIA = (PFN_vkGetMemoryZirconHandlePropertiesFUCHSIA)load(context, "vkGetMemoryZirconHandlePropertiesFUCHSIA");
#endif /* defined(VK_FUCHSIA_external_memory) */
#if defined(VK_FUCHSIA_external_semaphore)
	vkGetSemaphoreZirconHandleFUCHSIA = (PFN_vkGetSemaphoreZirconHandleFUCHSIA)load(context, "vkGetSemaphoreZirconHandleFUCHSIA");
	vkImportSemaphoreZirconHandleFUCHSIA = (PFN_vkImportSemaphoreZirconHandleFUCHSIA)load(context, "vkImportSemaphoreZirconHandleFUCHSIA");
#endif /* defined(VK_FUCHSIA_external_semaphore) */
#if defined(VK_GOOGLE_display_timing)
	vkGetPastPresentationTimingGOOGLE = (PFN_vkGetPastPresentationTimingGOOGLE)load(context, "vkGetPastPresentationTimingGOOGLE");
	vkGetRefreshCycleDurationGOOGLE = (PFN_vkGetRefreshCycleDurationGOOGLE)load(context, "vkGetRefreshCycleDurationGOOGLE");
#endif /* defined(VK_GOOGLE_display_timing) */
#if defined(VK_HUAWEI_cluster_culling_shader)
	vkCmdDrawClusterHUAWEI = (PFN_vkCmdDrawClusterHUAWEI)load(context, "vkCmdDrawClusterHUAWEI");
	vkCmdDrawClusterIndirectHUAWEI = (PFN_vkCmdDrawClusterIndirectHUAWEI)load(context, "vkCmdDrawClusterIndirectHUAWEI");
#endif /* defined(VK_HUAWEI_cluster_culling_shader) */
#if defined(VK_HUAWEI_invocation_mask)
	vkCmdBindInvocationMaskHUAWEI = (PFN_vkCmdBindInvocationMaskHUAWEI)load(context, "vkCmdBindInvocationMaskHUAWEI");
#endif /* defined(VK_HUAWEI_invocation_mask) */
#if defined(VK_HUAWEI_subpass_shading)
	vkCmdSubpassShadingHUAWEI = (PFN_vkCmdSubpassShadingHUAWEI)load(context, "vkCmdSubpassShadingHUAWEI");
	vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI = (PFN_vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI)load(context, "vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI");
#endif /* defined(VK_HUAWEI_subpass_shading) */
#if defined(VK_INTEL_performance_query)
	vkAcquirePerformanceConfigurationINTEL = (PFN_vkAcquirePerformanceConfigurationINTEL)load(context, "vkAcquirePerformanceConfigurationINTEL");
	vkCmdSetPerformanceMarkerINTEL = (PFN_vkCmdSetPerformanceMarkerINTEL)load(context, "vkCmdSetPerformanceMarkerINTEL");
	vkCmdSetPerformanceOverrideINTEL = (PFN_vkCmdSetPerformanceOverrideINTEL)load(context, "vkCmdSetPerformanceOverrideINTEL");
	vkCmdSetPerformanceStreamMarkerINTEL = (PFN_vkCmdSetPerformanceStreamMarkerINTEL)load(context, "vkCmdSetPerformanceStreamMarkerINTEL");
	vkGetPerformanceParameterINTEL = (PFN_vkGetPerformanceParameterINTEL)load(context, "vkGetPerformanceParameterINTEL");
	vkInitializePerformanceApiINTEL = (PFN_vkInitializePerformanceApiINTEL)load(context, "vkInitializePerformanceApiINTEL");
	vkQueueSetPerformanceConfigurationINTEL = (PFN_vkQueueSetPerformanceConfigurationINTEL)load(context, "vkQueueSetPerformanceConfigurationINTEL");
	vkReleasePerformanceConfigurationINTEL = (PFN_vkReleasePerformanceConfigurationINTEL)load(context, "vkReleasePerformanceConfigurationINTEL");
	vkUninitializePerformanceApiINTEL = (PFN_vkUninitializePerformanceApiINTEL)load(context, "vkUninitializePerformanceApiINTEL");
#endif /* defined(VK_INTEL_performance_query) */
#if defined(VK_KHR_acceleration_structure)
	vkBuildAccelerationStructuresKHR = (PFN_vkBuildAccelerationStructuresKHR)load(context, "vkBuildAccelerationStructuresKHR");
	vkCmdBuildAccelerationStructuresIndirectKHR = (PFN_vkCmdBuildAccelerationStructuresIndirectKHR)load(context, "vkCmdBuildAccelerationStructuresIndirectKHR");
	vkCmdBuildAccelerationStructuresKHR = (PFN_vkCmdBuildAccelerationStructuresKHR)load(context, "vkCmdBuildAccelerationStructuresKHR");
	vkCmdCopyAccelerationStructureKHR = (PFN_vkCmdCopyAccelerationStructureKHR)load(context, "vkCmdCopyAccelerationStructureKHR");
	vkCmdCopyAccelerationStructureToMemoryKHR = (PFN_vkCmdCopyAccelerationStructureToMemoryKHR)load(context, "vkCmdCopyAccelerationStructureToMemoryKHR");
	vkCmdCopyMemoryToAccelerationStructureKHR = (PFN_vkCmdCopyMemoryToAccelerationStructureKHR)load(context, "vkCmdCopyMemoryToAccelerationStructureKHR");
	vkCmdWriteAccelerationStructuresPropertiesKHR = (PFN_vkCmdWriteAccelerationStructuresPropertiesKHR)load(context, "vkCmdWriteAccelerationStructuresPropertiesKHR");
	vkCopyAccelerationStructureKHR = (PFN_vkCopyAccelerationStructureKHR)load(context, "vkCopyAccelerationStructureKHR");
	vkCopyAccelerationStructureToMemoryKHR = (PFN_vkCopyAccelerationStructureToMemoryKHR)load(context, "vkCopyAccelerationStructureToMemoryKHR");
	vkCopyMemoryToAccelerationStructureKHR = (PFN_vkCopyMemoryToAccelerationStructureKHR)load(context, "vkCopyMemoryToAccelerationStructureKHR");
	vkCreateAccelerationStructureKHR = (PFN_vkCreateAccelerationStructureKHR)load(context, "vkCreateAccelerationStructureKHR");
	vkDestroyAccelerationStructureKHR = (PFN_vkDestroyAccelerationStructureKHR)load(context, "vkDestroyAccelerationStructureKHR");
	vkGetAccelerationStructureBuildSizesKHR = (PFN_vkGetAccelerationStructureBuildSizesKHR)load(context, "vkGetAccelerationStructureBuildSizesKHR");
	vkGetAccelerationStructureDeviceAddressKHR = (PFN_vkGetAccelerationStructureDeviceAddressKHR)load(context, "vkGetAccelerationStructureDeviceAddressKHR");
	vkGetDeviceAccelerationStructureCompatibilityKHR = (PFN_vkGetDeviceAccelerationStructureCompatibilityKHR)load(context, "vkGetDeviceAccelerationStructureCompatibilityKHR");
	vkWriteAccelerationStructuresPropertiesKHR = (PFN_vkWriteAccelerationStructuresPropertiesKHR)load(context, "vkWriteAccelerationStructuresPropertiesKHR");
#endif /* defined(VK_KHR_acceleration_structure) */
#if defined(VK_KHR_bind_memory2)
	vkBindBufferMemory2KHR = (PFN_vkBindBufferMemory2KHR)load(context, "vkBindBufferMemory2KHR");
	vkBindImageMemory2KHR = (PFN_vkBindImageMemory2KHR)load(context, "vkBindImageMemory2KHR");
#endif /* defined(VK_KHR_bind_memory2) */
#if defined(VK_KHR_buffer_device_address)
	vkGetBufferDeviceAddressKHR = (PFN_vkGetBufferDeviceAddressKHR)load(context, "vkGetBufferDeviceAddressKHR");
	vkGetBufferOpaqueCaptureAddressKHR = (PFN_vkGetBufferOpaqueCaptureAddressKHR)load(context, "vkGetBufferOpaqueCaptureAddressKHR");
	vkGetDeviceMemoryOpaqueCaptureAddressKHR = (PFN_vkGetDeviceMemoryOpaqueCaptureAddressKHR)load(context, "vkGetDeviceMemoryOpaqueCaptureAddressKHR");
#endif /* defined(VK_KHR_buffer_device_address) */
#if defined(VK_KHR_calibrated_timestamps)
	vkGetCalibratedTimestampsKHR = (PFN_vkGetCalibratedTimestampsKHR)load(context, "vkGetCalibratedTimestampsKHR");
#endif /* defined(VK_KHR_calibrated_timestamps) */
#if defined(VK_KHR_copy_commands2)
	vkCmdBlitImage2KHR = (PFN_vkCmdBlitImage2KHR)load(context, "vkCmdBlitImage2KHR");
	vkCmdCopyBuffer2KHR = (PFN_vkCmdCopyBuffer2KHR)load(context, "vkCmdCopyBuffer2KHR");
	vkCmdCopyBufferToImage2KHR = (PFN_vkCmdCopyBufferToImage2KHR)load(context, "vkCmdCopyBufferToImage2KHR");
	vkCmdCopyImage2KHR = (PFN_vkCmdCopyImage2KHR)load(context, "vkCmdCopyImage2KHR");
	vkCmdCopyImageToBuffer2KHR = (PFN_vkCmdCopyImageToBuffer2KHR)load(context, "vkCmdCopyImageToBuffer2KHR");
	vkCmdResolveImage2KHR = (PFN_vkCmdResolveImage2KHR)load(context, "vkCmdResolveImage2KHR");
#endif /* defined(VK_KHR_copy_commands2) */
#if defined(VK_KHR_create_renderpass2)
	vkCmdBeginRenderPass2KHR = (PFN_vkCmdBeginRenderPass2KHR)load(context, "vkCmdBeginRenderPass2KHR");
	vkCmdEndRenderPass2KHR = (PFN_vkCmdEndRenderPass2KHR)load(context, "vkCmdEndRenderPass2KHR");
	vkCmdNextSubpass2KHR = (PFN_vkCmdNextSubpass2KHR)load(context, "vkCmdNextSubpass2KHR");
	vkCreateRenderPass2KHR = (PFN_vkCreateRenderPass2KHR)load(context, "vkCreateRenderPass2KHR");
#endif /* defined(VK_KHR_create_renderpass2) */
#if defined(VK_KHR_deferred_host_operations)
	vkCreateDeferredOperationKHR = (PFN_vkCreateDeferredOperationKHR)load(context, "vkCreateDeferredOperationKHR");
	vkDeferredOperationJoinKHR = (PFN_vkDeferredOperationJoinKHR)load(context, "vkDeferredOperationJoinKHR");
	vkDestroyDeferredOperationKHR = (PFN_vkDestroyDeferredOperationKHR)load(context, "vkDestroyDeferredOperationKHR");
	vkGetDeferredOperationMaxConcurrencyKHR = (PFN_vkGetDeferredOperationMaxConcurrencyKHR)load(context, "vkGetDeferredOperationMaxConcurrencyKHR");
	vkGetDeferredOperationResultKHR = (PFN_vkGetDeferredOperationResultKHR)load(context, "vkGetDeferredOperationResultKHR");
#endif /* defined(VK_KHR_deferred_host_operations) */
#if defined(VK_KHR_descriptor_update_template)
	vkCreateDescriptorUpdateTemplateKHR = (PFN_vkCreateDescriptorUpdateTemplateKHR)load(context, "vkCreateDescriptorUpdateTemplateKHR");
	vkDestroyDescriptorUpdateTemplateKHR = (PFN_vkDestroyDescriptorUpdateTemplateKHR)load(context, "vkDestroyDescriptorUpdateTemplateKHR");
	vkUpdateDescriptorSetWithTemplateKHR = (PFN_vkUpdateDescriptorSetWithTemplateKHR)load(context, "vkUpdateDescriptorSetWithTemplateKHR");
#endif /* defined(VK_KHR_descriptor_update_template) */
#if defined(VK_KHR_device_group)
	vkCmdDispatchBaseKHR = (PFN_vkCmdDispatchBaseKHR)load(context, "vkCmdDispatchBaseKHR");
	vkCmdSetDeviceMaskKHR = (PFN_vkCmdSetDeviceMaskKHR)load(context, "vkCmdSetDeviceMaskKHR");
	vkGetDeviceGroupPeerMemoryFeaturesKHR = (PFN_vkGetDeviceGroupPeerMemoryFeaturesKHR)load(context, "vkGetDeviceGroupPeerMemoryFeaturesKHR");
#endif /* defined(VK_KHR_device_group) */
#if defined(VK_KHR_display_swapchain)
	vkCreateSharedSwapchainsKHR = (PFN_vkCreateSharedSwapchainsKHR)load(context, "vkCreateSharedSwapchainsKHR");
#endif /* defined(VK_KHR_display_swapchain) */
#if defined(VK_KHR_draw_indirect_count)
	vkCmdDrawIndexedIndirectCountKHR = (PFN_vkCmdDrawIndexedIndirectCountKHR)load(context, "vkCmdDrawIndexedIndirectCountKHR");
	vkCmdDrawIndirectCountKHR = (PFN_vkCmdDrawIndirectCountKHR)load(context, "vkCmdDrawIndirectCountKHR");
#endif /* defined(VK_KHR_draw_indirect_count) */
#if defined(VK_KHR_dynamic_rendering)
	vkCmdBeginRenderingKHR = (PFN_vkCmdBeginRenderingKHR)load(context, "vkCmdBeginRenderingKHR");
	vkCmdEndRenderingKHR = (PFN_vkCmdEndRenderingKHR)load(context, "vkCmdEndRenderingKHR");
#endif /* defined(VK_KHR_dynamic_rendering) */
#if defined(VK_KHR_external_fence_fd)
	vkGetFenceFdKHR = (PFN_vkGetFenceFdKHR)load(context, "vkGetFenceFdKHR");
	vkImportFenceFdKHR = (PFN_vkImportFenceFdKHR)load(context, "vkImportFenceFdKHR");
#endif /* defined(VK_KHR_external_fence_fd) */
#if defined(VK_KHR_external_fence_win32)
	vkGetFenceWin32HandleKHR = (PFN_vkGetFenceWin32HandleKHR)load(context, "vkGetFenceWin32HandleKHR");
	vkImportFenceWin32HandleKHR = (PFN_vkImportFenceWin32HandleKHR)load(context, "vkImportFenceWin32HandleKHR");
#endif /* defined(VK_KHR_external_fence_win32) */
#if defined(VK_KHR_external_memory_fd)
	vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)load(context, "vkGetMemoryFdKHR");
	vkGetMemoryFdPropertiesKHR = (PFN_vkGetMemoryFdPropertiesKHR)load(context, "vkGetMemoryFdPropertiesKHR");
#endif /* defined(VK_KHR_external_memory_fd) */
#if defined(VK_KHR_external_memory_win32)
	vkGetMemoryWin32HandleKHR = (PFN_vkGetMemoryWin32HandleKHR)load(context, "vkGetMemoryWin32HandleKHR");
	vkGetMemoryWin32HandlePropertiesKHR = (PFN_vkGetMemoryWin32HandlePropertiesKHR)load(context, "vkGetMemoryWin32HandlePropertiesKHR");
#endif /* defined(VK_KHR_external_memory_win32) */
#if defined(VK_KHR_external_semaphore_fd)
	vkGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)load(context, "vkGetSemaphoreFdKHR");
	vkImportSemaphoreFdKHR = (PFN_vkImportSemaphoreFdKHR)load(context, "vkImportSemaphoreFdKHR");
#endif /* defined(VK_KHR_external_semaphore_fd) */
#if defined(VK_KHR_external_semaphore_win32)
	vkGetSemaphoreWin32HandleKHR = (PFN_vkGetSemaphoreWin32HandleKHR)load(context, "vkGetSemaphoreWin32HandleKHR");
	vkImportSemaphoreWin32HandleKHR = (PFN_vkImportSemaphoreWin32HandleKHR)load(context, "vkImportSemaphoreWin32HandleKHR");
#endif /* defined(VK_KHR_external_semaphore_win32) */
#if defined(VK_KHR_fragment_shading_rate)
	vkCmdSetFragmentShadingRateKHR = (PFN_vkCmdSetFragmentShadingRateKHR)load(context, "vkCmdSetFragmentShadingRateKHR");
#endif /* defined(VK_KHR_fragment_shading_rate) */
#if defined(VK_KHR_get_memory_requirements2)
	vkGetBufferMemoryRequirements2KHR = (PFN_vkGetBufferMemoryRequirements2KHR)load(context, "vkGetBufferMemoryRequirements2KHR");
	vkGetImageMemoryRequirements2KHR = (PFN_vkGetImageMemoryRequirements2KHR)load(context, "vkGetImageMemoryRequirements2KHR");
	vkGetImageSparseMemoryRequirements2KHR = (PFN_vkGetImageSparseMemoryRequirements2KHR)load(context, "vkGetImageSparseMemoryRequirements2KHR");
#endif /* defined(VK_KHR_get_memory_requirements2) */
#if defined(VK_KHR_maintenance1)
	vkTrimCommandPoolKHR = (PFN_vkTrimCommandPoolKHR)load(context, "vkTrimCommandPoolKHR");
#endif /* defined(VK_KHR_maintenance1) */
#if defined(VK_KHR_maintenance3)
	vkGetDescriptorSetLayoutSupportKHR = (PFN_vkGetDescriptorSetLayoutSupportKHR)load(context, "vkGetDescriptorSetLayoutSupportKHR");
#endif /* defined(VK_KHR_maintenance3) */
#if defined(VK_KHR_maintenance4)
	vkGetDeviceBufferMemoryRequirementsKHR = (PFN_vkGetDeviceBufferMemoryRequirementsKHR)load(context, "vkGetDeviceBufferMemoryRequirementsKHR");
	vkGetDeviceImageMemoryRequirementsKHR = (PFN_vkGetDeviceImageMemoryRequirementsKHR)load(context, "vkGetDeviceImageMemoryRequirementsKHR");
	vkGetDeviceImageSparseMemoryRequirementsKHR = (PFN_vkGetDeviceImageSparseMemoryRequirementsKHR)load(context, "vkGetDeviceImageSparseMemoryRequirementsKHR");
#endif /* defined(VK_KHR_maintenance4) */
#if defined(VK_KHR_maintenance5)
	vkCmdBindIndexBuffer2KHR = (PFN_vkCmdBindIndexBuffer2KHR)load(context, "vkCmdBindIndexBuffer2KHR");
	vkGetDeviceImageSubresourceLayoutKHR = (PFN_vkGetDeviceImageSubresourceLayoutKHR)load(context, "vkGetDeviceImageSubresourceLayoutKHR");
	vkGetImageSubresourceLayout2KHR = (PFN_vkGetImageSubresourceLayout2KHR)load(context, "vkGetImageSubresourceLayout2KHR");
	vkGetRenderingAreaGranularityKHR = (PFN_vkGetRenderingAreaGranularityKHR)load(context, "vkGetRenderingAreaGranularityKHR");
#endif /* defined(VK_KHR_maintenance5) */
#if defined(VK_KHR_maintenance6)
	vkCmdBindDescriptorSets2KHR = (PFN_vkCmdBindDescriptorSets2KHR)load(context, "vkCmdBindDescriptorSets2KHR");
	vkCmdPushConstants2KHR = (PFN_vkCmdPushConstants2KHR)load(context, "vkCmdPushConstants2KHR");
#endif /* defined(VK_KHR_maintenance6) */
#if defined(VK_KHR_maintenance6) && defined(VK_KHR_push_descriptor)
	vkCmdPushDescriptorSet2KHR = (PFN_vkCmdPushDescriptorSet2KHR)load(context, "vkCmdPushDescriptorSet2KHR");
	vkCmdPushDescriptorSetWithTemplate2KHR = (PFN_vkCmdPushDescriptorSetWithTemplate2KHR)load(context, "vkCmdPushDescriptorSetWithTemplate2KHR");
#endif /* defined(VK_KHR_maintenance6) && defined(VK_KHR_push_descriptor) */
#if defined(VK_KHR_maintenance6) && defined(VK_EXT_descriptor_buffer)
	vkCmdBindDescriptorBufferEmbeddedSamplers2EXT = (PFN_vkCmdBindDescriptorBufferEmbeddedSamplers2EXT)load(context, "vkCmdBindDescriptorBufferEmbeddedSamplers2EXT");
	vkCmdSetDescriptorBufferOffsets2EXT = (PFN_vkCmdSetDescriptorBufferOffsets2EXT)load(context, "vkCmdSetDescriptorBufferOffsets2EXT");
#endif /* defined(VK_KHR_maintenance6) && defined(VK_EXT_descriptor_buffer) */
#if defined(VK_KHR_map_memory2)
	vkMapMemory2KHR = (PFN_vkMapMemory2KHR)load(context, "vkMapMemory2KHR");
	vkUnmapMemory2KHR = (PFN_vkUnmapMemory2KHR)load(context, "vkUnmapMemory2KHR");
#endif /* defined(VK_KHR_map_memory2) */
#if defined(VK_KHR_performance_query)
	vkAcquireProfilingLockKHR = (PFN_vkAcquireProfilingLockKHR)load(context, "vkAcquireProfilingLockKHR");
	vkReleaseProfilingLockKHR = (PFN_vkReleaseProfilingLockKHR)load(context, "vkReleaseProfilingLockKHR");
#endif /* defined(VK_KHR_performance_query) */
#if defined(VK_KHR_pipeline_executable_properties)
	vkGetPipelineExecutableInternalRepresentationsKHR = (PFN_vkGetPipelineExecutableInternalRepresentationsKHR)load(context, "vkGetPipelineExecutableInternalRepresentationsKHR");
	vkGetPipelineExecutablePropertiesKHR = (PFN_vkGetPipelineExecutablePropertiesKHR)load(context, "vkGetPipelineExecutablePropertiesKHR");
	vkGetPipelineExecutableStatisticsKHR = (PFN_vkGetPipelineExecutableStatisticsKHR)load(context, "vkGetPipelineExecutableStatisticsKHR");
#endif /* defined(VK_KHR_pipeline_executable_properties) */
#if defined(VK_KHR_present_wait)
	vkWaitForPresentKHR = (PFN_vkWaitForPresentKHR)load(context, "vkWaitForPresentKHR");
#endif /* defined(VK_KHR_present_wait) */
#if defined(VK_KHR_push_descriptor)
	vkCmdPushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)load(context, "vkCmdPushDescriptorSetKHR");
#endif /* defined(VK_KHR_push_descriptor) */
#if defined(VK_KHR_ray_tracing_maintenance1) && defined(VK_KHR_ray_tracing_pipeline)
	vkCmdTraceRaysIndirect2KHR = (PFN_vkCmdTraceRaysIndirect2KHR)load(context, "vkCmdTraceRaysIndirect2KHR");
#endif /* defined(VK_KHR_ray_tracing_maintenance1) && defined(VK_KHR_ray_tracing_pipeline) */
#if defined(VK_KHR_ray_tracing_pipeline)
	vkCmdSetRayTracingPipelineStackSizeKHR = (PFN_vkCmdSetRayTracingPipelineStackSizeKHR)load(context, "vkCmdSetRayTracingPipelineStackSizeKHR");
	vkCmdTraceRaysIndirectKHR = (PFN_vkCmdTraceRaysIndirectKHR)load(context, "vkCmdTraceRaysIndirectKHR");
	vkCmdTraceRaysKHR = (PFN_vkCmdTraceRaysKHR)load(context, "vkCmdTraceRaysKHR");
	vkCreateRayTracingPipelinesKHR = (PFN_vkCreateRayTracingPipelinesKHR)load(context, "vkCreateRayTracingPipelinesKHR");
	vkGetRayTracingCaptureReplayShaderGroupHandlesKHR = (PFN_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR)load(context, "vkGetRayTracingCaptureReplayShaderGroupHandlesKHR");
	vkGetRayTracingShaderGroupHandlesKHR = (PFN_vkGetRayTracingShaderGroupHandlesKHR)load(context, "vkGetRayTracingShaderGroupHandlesKHR");
	vkGetRayTracingShaderGroupStackSizeKHR = (PFN_vkGetRayTracingShaderGroupStackSizeKHR)load(context, "vkGetRayTracingShaderGroupStackSizeKHR");
#endif /* defined(VK_KHR_ray_tracing_pipeline) */
#if defined(VK_KHR_sampler_ycbcr_conversion)
	vkCreateSamplerYcbcrConversionKHR = (PFN_vkCreateSamplerYcbcrConversionKHR)load(context, "vkCreateSamplerYcbcrConversionKHR");
	vkDestroySamplerYcbcrConversionKHR = (PFN_vkDestroySamplerYcbcrConversionKHR)load(context, "vkDestroySamplerYcbcrConversionKHR");
#endif /* defined(VK_KHR_sampler_ycbcr_conversion) */
#if defined(VK_KHR_shared_presentable_image)
	vkGetSwapchainStatusKHR = (PFN_vkGetSwapchainStatusKHR)load(context, "vkGetSwapchainStatusKHR");
#endif /* defined(VK_KHR_shared_presentable_image) */
#if defined(VK_KHR_swapchain)
	vkAcquireNextImageKHR = (PFN_vkAcquireNextImageKHR)load(context, "vkAcquireNextImageKHR");
	vkCreateSwapchainKHR = (PFN_vkCreateSwapchainKHR)load(context, "vkCreateSwapchainKHR");
	vkDestroySwapchainKHR = (PFN_vkDestroySwapchainKHR)load(context, "vkDestroySwapchainKHR");
	vkGetSwapchainImagesKHR = (PFN_vkGetSwapchainImagesKHR)load(context, "vkGetSwapchainImagesKHR");
	vkQueuePresentKHR = (PFN_vkQueuePresentKHR)load(context, "vkQueuePresentKHR");
#endif /* defined(VK_KHR_swapchain) */
#if defined(VK_KHR_synchronization2)
	vkCmdPipelineBarrier2KHR = (PFN_vkCmdPipelineBarrier2KHR)load(context, "vkCmdPipelineBarrier2KHR");
	vkCmdResetEvent2KHR = (PFN_vkCmdResetEvent2KHR)load(context, "vkCmdResetEvent2KHR");
	vkCmdSetEvent2KHR = (PFN_vkCmdSetEvent2KHR)load(context, "vkCmdSetEvent2KHR");
	vkCmdWaitEvents2KHR = (PFN_vkCmdWaitEvents2KHR)load(context, "vkCmdWaitEvents2KHR");
	vkCmdWriteTimestamp2KHR = (PFN_vkCmdWriteTimestamp2KHR)load(context, "vkCmdWriteTimestamp2KHR");
	vkQueueSubmit2KHR = (PFN_vkQueueSubmit2KHR)load(context, "vkQueueSubmit2KHR");
#endif /* defined(VK_KHR_synchronization2) */
#if defined(VK_KHR_synchronization2) && defined(VK_AMD_buffer_marker)
	vkCmdWriteBufferMarker2AMD = (PFN_vkCmdWriteBufferMarker2AMD)load(context, "vkCmdWriteBufferMarker2AMD");
#endif /* defined(VK_KHR_synchronization2) && defined(VK_AMD_buffer_marker) */
#if defined(VK_KHR_synchronization2) && defined(VK_NV_device_diagnostic_checkpoints)
	vkGetQueueCheckpointData2NV = (PFN_vkGetQueueCheckpointData2NV)load(context, "vkGetQueueCheckpointData2NV");
#endif /* defined(VK_KHR_synchronization2) && defined(VK_NV_device_diagnostic_checkpoints) */
#if defined(VK_KHR_timeline_semaphore)
	vkGetSemaphoreCounterValueKHR = (PFN_vkGetSemaphoreCounterValueKHR)load(context, "vkGetSemaphoreCounterValueKHR");
	vkSignalSemaphoreKHR = (PFN_vkSignalSemaphoreKHR)load(context, "vkSignalSemaphoreKHR");
	vkWaitSemaphoresKHR = (PFN_vkWaitSemaphoresKHR)load(context, "vkWaitSemaphoresKHR");
#endif /* defined(VK_KHR_timeline_semaphore) */
#if defined(VK_KHR_video_decode_queue)
	vkCmdDecodeVideoKHR = (PFN_vkCmdDecodeVideoKHR)load(context, "vkCmdDecodeVideoKHR");
#endif /* defined(VK_KHR_video_decode_queue) */
#if defined(VK_KHR_video_encode_queue)
	vkCmdEncodeVideoKHR = (PFN_vkCmdEncodeVideoKHR)load(context, "vkCmdEncodeVideoKHR");
	vkGetEncodedVideoSessionParametersKHR = (PFN_vkGetEncodedVideoSessionParametersKHR)load(context, "vkGetEncodedVideoSessionParametersKHR");
#endif /* defined(VK_KHR_video_encode_queue) */
#if defined(VK_KHR_video_queue)
	vkBindVideoSessionMemoryKHR = (PFN_vkBindVideoSessionMemoryKHR)load(context, "vkBindVideoSessionMemoryKHR");
	vkCmdBeginVideoCodingKHR = (PFN_vkCmdBeginVideoCodingKHR)load(context, "vkCmdBeginVideoCodingKHR");
	vkCmdControlVideoCodingKHR = (PFN_vkCmdControlVideoCodingKHR)load(context, "vkCmdControlVideoCodingKHR");
	vkCmdEndVideoCodingKHR = (PFN_vkCmdEndVideoCodingKHR)load(context, "vkCmdEndVideoCodingKHR");
	vkCreateVideoSessionKHR = (PFN_vkCreateVideoSessionKHR)load(context, "vkCreateVideoSessionKHR");
	vkCreateVideoSessionParametersKHR = (PFN_vkCreateVideoSessionParametersKHR)load(context, "vkCreateVideoSessionParametersKHR");
	vkDestroyVideoSessionKHR = (PFN_vkDestroyVideoSessionKHR)load(context, "vkDestroyVideoSessionKHR");
	vkDestroyVideoSessionParametersKHR = (PFN_vkDestroyVideoSessionParametersKHR)load(context, "vkDestroyVideoSessionParametersKHR");
	vkGetVideoSessionMemoryRequirementsKHR = (PFN_vkGetVideoSessionMemoryRequirementsKHR)load(context, "vkGetVideoSessionMemoryRequirementsKHR");
	vkUpdateVideoSessionParametersKHR = (PFN_vkUpdateVideoSessionParametersKHR)load(context, "vkUpdateVideoSessionParametersKHR");
#endif /* defined(VK_KHR_video_queue) */
#if defined(VK_NVX_binary_import)
	vkCmdCuLaunchKernelNVX = (PFN_vkCmdCuLaunchKernelNVX)load(context, "vkCmdCuLaunchKernelNVX");
	vkCreateCuFunctionNVX = (PFN_vkCreateCuFunctionNVX)load(context, "vkCreateCuFunctionNVX");
	vkCreateCuModuleNVX = (PFN_vkCreateCuModuleNVX)load(context, "vkCreateCuModuleNVX");
	vkDestroyCuFunctionNVX = (PFN_vkDestroyCuFunctionNVX)load(context, "vkDestroyCuFunctionNVX");
	vkDestroyCuModuleNVX = (PFN_vkDestroyCuModuleNVX)load(context, "vkDestroyCuModuleNVX");
#endif /* defined(VK_NVX_binary_import) */
#if defined(VK_NVX_image_view_handle)
	vkGetImageViewAddressNVX = (PFN_vkGetImageViewAddressNVX)load(context, "vkGetImageViewAddressNVX");
	vkGetImageViewHandleNVX = (PFN_vkGetImageViewHandleNVX)load(context, "vkGetImageViewHandleNVX");
#endif /* defined(VK_NVX_image_view_handle) */
#if defined(VK_NV_clip_space_w_scaling)
	vkCmdSetViewportWScalingNV = (PFN_vkCmdSetViewportWScalingNV)load(context, "vkCmdSetViewportWScalingNV");
#endif /* defined(VK_NV_clip_space_w_scaling) */
#if defined(VK_NV_copy_memory_indirect)
	vkCmdCopyMemoryIndirectNV = (PFN_vkCmdCopyMemoryIndirectNV)load(context, "vkCmdCopyMemoryIndirectNV");
	vkCmdCopyMemoryToImageIndirectNV = (PFN_vkCmdCopyMemoryToImageIndirectNV)load(context, "vkCmdCopyMemoryToImageIndirectNV");
#endif /* defined(VK_NV_copy_memory_indirect) */
#if defined(VK_NV_cuda_kernel_launch)
	vkCmdCudaLaunchKernelNV = (PFN_vkCmdCudaLaunchKernelNV)load(context, "vkCmdCudaLaunchKernelNV");
	vkCreateCudaFunctionNV = (PFN_vkCreateCudaFunctionNV)load(context, "vkCreateCudaFunctionNV");
	vkCreateCudaModuleNV = (PFN_vkCreateCudaModuleNV)load(context, "vkCreateCudaModuleNV");
	vkDestroyCudaFunctionNV = (PFN_vkDestroyCudaFunctionNV)load(context, "vkDestroyCudaFunctionNV");
	vkDestroyCudaModuleNV = (PFN_vkDestroyCudaModuleNV)load(context, "vkDestroyCudaModuleNV");
	vkGetCudaModuleCacheNV = (PFN_vkGetCudaModuleCacheNV)load(context, "vkGetCudaModuleCacheNV");
#endif /* defined(VK_NV_cuda_kernel_launch) */
#if defined(VK_NV_device_diagnostic_checkpoints)
	vkCmdSetCheckpointNV = (PFN_vkCmdSetCheckpointNV)load(context, "vkCmdSetCheckpointNV");
	vkGetQueueCheckpointDataNV = (PFN_vkGetQueueCheckpointDataNV)load(context, "vkGetQueueCheckpointDataNV");
#endif /* defined(VK_NV_device_diagnostic_checkpoints) */
#if defined(VK_NV_device_generated_commands)
	vkCmdBindPipelineShaderGroupNV = (PFN_vkCmdBindPipelineShaderGroupNV)load(context, "vkCmdBindPipelineShaderGroupNV");
	vkCmdExecuteGeneratedCommandsNV = (PFN_vkCmdExecuteGeneratedCommandsNV)load(context, "vkCmdExecuteGeneratedCommandsNV");
	vkCmdPreprocessGeneratedCommandsNV = (PFN_vkCmdPreprocessGeneratedCommandsNV)load(context, "vkCmdPreprocessGeneratedCommandsNV");
	vkCreateIndirectCommandsLayoutNV = (PFN_vkCreateIndirectCommandsLayoutNV)load(context, "vkCreateIndirectCommandsLayoutNV");
	vkDestroyIndirectCommandsLayoutNV = (PFN_vkDestroyIndirectCommandsLayoutNV)load(context, "vkDestroyIndirectCommandsLayoutNV");
	vkGetGeneratedCommandsMemoryRequirementsNV = (PFN_vkGetGeneratedCommandsMemoryRequirementsNV)load(context, "vkGetGeneratedCommandsMemoryRequirementsNV");
#endif /* defined(VK_NV_device_generated_commands) */
#if defined(VK_NV_device_generated_commands_compute)
	vkCmdUpdatePipelineIndirectBufferNV = (PFN_vkCmdUpdatePipelineIndirectBufferNV)load(context, "vkCmdUpdatePipelineIndirectBufferNV");
	vkGetPipelineIndirectDeviceAddressNV = (PFN_vkGetPipelineIndirectDeviceAddressNV)load(context, "vkGetPipelineIndirectDeviceAddressNV");
	vkGetPipelineIndirectMemoryRequirementsNV = (PFN_vkGetPipelineIndirectMemoryRequirementsNV)load(context, "vkGetPipelineIndirectMemoryRequirementsNV");
#endif /* defined(VK_NV_device_generated_commands_compute) */
#if defined(VK_NV_external_memory_rdma)
	vkGetMemoryRemoteAddressNV = (PFN_vkGetMemoryRemoteAddressNV)load(context, "vkGetMemoryRemoteAddressNV");
#endif /* defined(VK_NV_external_memory_rdma) */
#if defined(VK_NV_external_memory_win32)
	vkGetMemoryWin32HandleNV = (PFN_vkGetMemoryWin32HandleNV)load(context, "vkGetMemoryWin32HandleNV");
#endif /* defined(VK_NV_external_memory_win32) */
#if defined(VK_NV_fragment_shading_rate_enums)
	vkCmdSetFragmentShadingRateEnumNV = (PFN_vkCmdSetFragmentShadingRateEnumNV)load(context, "vkCmdSetFragmentShadingRateEnumNV");
#endif /* defined(VK_NV_fragment_shading_rate_enums) */
#if defined(VK_NV_low_latency2)
	vkGetLatencyTimingsNV = (PFN_vkGetLatencyTimingsNV)load(context, "vkGetLatencyTimingsNV");
	vkLatencySleepNV = (PFN_vkLatencySleepNV)load(context, "vkLatencySleepNV");
	vkQueueNotifyOutOfBandNV = (PFN_vkQueueNotifyOutOfBandNV)load(context, "vkQueueNotifyOutOfBandNV");
	vkSetLatencyMarkerNV = (PFN_vkSetLatencyMarkerNV)load(context, "vkSetLatencyMarkerNV");
	vkSetLatencySleepModeNV = (PFN_vkSetLatencySleepModeNV)load(context, "vkSetLatencySleepModeNV");
#endif /* defined(VK_NV_low_latency2) */
#if defined(VK_NV_memory_decompression)
	vkCmdDecompressMemoryIndirectCountNV = (PFN_vkCmdDecompressMemoryIndirectCountNV)load(context, "vkCmdDecompressMemoryIndirectCountNV");
	vkCmdDecompressMemoryNV = (PFN_vkCmdDecompressMemoryNV)load(context, "vkCmdDecompressMemoryNV");
#endif /* defined(VK_NV_memory_decompression) */
#if defined(VK_NV_mesh_shader)
	vkCmdDrawMeshTasksIndirectCountNV = (PFN_vkCmdDrawMeshTasksIndirectCountNV)load(context, "vkCmdDrawMeshTasksIndirectCountNV");
	vkCmdDrawMeshTasksIndirectNV = (PFN_vkCmdDrawMeshTasksIndirectNV)load(context, "vkCmdDrawMeshTasksIndirectNV");
	vkCmdDrawMeshTasksNV = (PFN_vkCmdDrawMeshTasksNV)load(context, "vkCmdDrawMeshTasksNV");
#endif /* defined(VK_NV_mesh_shader) */
#if defined(VK_NV_optical_flow)
	vkBindOpticalFlowSessionImageNV = (PFN_vkBindOpticalFlowSessionImageNV)load(context, "vkBindOpticalFlowSessionImageNV");
	vkCmdOpticalFlowExecuteNV = (PFN_vkCmdOpticalFlowExecuteNV)load(context, "vkCmdOpticalFlowExecuteNV");
	vkCreateOpticalFlowSessionNV = (PFN_vkCreateOpticalFlowSessionNV)load(context, "vkCreateOpticalFlowSessionNV");
	vkDestroyOpticalFlowSessionNV = (PFN_vkDestroyOpticalFlowSessionNV)load(context, "vkDestroyOpticalFlowSessionNV");
#endif /* defined(VK_NV_optical_flow) */
#if defined(VK_NV_ray_tracing)
	vkBindAccelerationStructureMemoryNV = (PFN_vkBindAccelerationStructureMemoryNV)load(context, "vkBindAccelerationStructureMemoryNV");
	vkCmdBuildAccelerationStructureNV = (PFN_vkCmdBuildAccelerationStructureNV)load(context, "vkCmdBuildAccelerationStructureNV");
	vkCmdCopyAccelerationStructureNV = (PFN_vkCmdCopyAccelerationStructureNV)load(context, "vkCmdCopyAccelerationStructureNV");
	vkCmdTraceRaysNV = (PFN_vkCmdTraceRaysNV)load(context, "vkCmdTraceRaysNV");
	vkCmdWriteAccelerationStructuresPropertiesNV = (PFN_vkCmdWriteAccelerationStructuresPropertiesNV)load(context, "vkCmdWriteAccelerationStructuresPropertiesNV");
	vkCompileDeferredNV = (PFN_vkCompileDeferredNV)load(context, "vkCompileDeferredNV");
	vkCreateAccelerationStructureNV = (PFN_vkCreateAccelerationStructureNV)load(context, "vkCreateAccelerationStructureNV");
	vkCreateRayTracingPipelinesNV = (PFN_vkCreateRayTracingPipelinesNV)load(context, "vkCreateRayTracingPipelinesNV");
	vkDestroyAccelerationStructureNV = (PFN_vkDestroyAccelerationStructureNV)load(context, "vkDestroyAccelerationStructureNV");
	vkGetAccelerationStructureHandleNV = (PFN_vkGetAccelerationStructureHandleNV)load(context, "vkGetAccelerationStructureHandleNV");
	vkGetAccelerationStructureMemoryRequirementsNV = (PFN_vkGetAccelerationStructureMemoryRequirementsNV)load(context, "vkGetAccelerationStructureMemoryRequirementsNV");
	vkGetRayTracingShaderGroupHandlesNV = (PFN_vkGetRayTracingShaderGroupHandlesNV)load(context, "vkGetRayTracingShaderGroupHandlesNV");
#endif /* defined(VK_NV_ray_tracing) */
#if defined(VK_NV_scissor_exclusive) && VK_NV_SCISSOR_EXCLUSIVE_SPEC_VERSION >= 2
	vkCmdSetExclusiveScissorEnableNV = (PFN_vkCmdSetExclusiveScissorEnableNV)load(context, "vkCmdSetExclusiveScissorEnableNV");
#endif /* defined(VK_NV_scissor_exclusive) && VK_NV_SCISSOR_EXCLUSIVE_SPEC_VERSION >= 2 */
#if defined(VK_NV_scissor_exclusive)
	vkCmdSetExclusiveScissorNV = (PFN_vkCmdSetExclusiveScissorNV)load(context, "vkCmdSetExclusiveScissorNV");
#endif /* defined(VK_NV_scissor_exclusive) */
#if defined(VK_NV_shading_rate_image)
	vkCmdBindShadingRateImageNV = (PFN_vkCmdBindShadingRateImageNV)load(context, "vkCmdBindShadingRateImageNV");
	vkCmdSetCoarseSampleOrderNV = (PFN_vkCmdSetCoarseSampleOrderNV)load(context, "vkCmdSetCoarseSampleOrderNV");
	vkCmdSetViewportShadingRatePaletteNV = (PFN_vkCmdSetViewportShadingRatePaletteNV)load(context, "vkCmdSetViewportShadingRatePaletteNV");
#endif /* defined(VK_NV_shading_rate_image) */
#if defined(VK_QCOM_tile_properties)
	vkGetDynamicRenderingTilePropertiesQCOM = (PFN_vkGetDynamicRenderingTilePropertiesQCOM)load(context, "vkGetDynamicRenderingTilePropertiesQCOM");
	vkGetFramebufferTilePropertiesQCOM = (PFN_vkGetFramebufferTilePropertiesQCOM)load(context, "vkGetFramebufferTilePropertiesQCOM");
#endif /* defined(VK_QCOM_tile_properties) */
#if defined(VK_QNX_external_memory_screen_buffer)
	vkGetScreenBufferPropertiesQNX = (PFN_vkGetScreenBufferPropertiesQNX)load(context, "vkGetScreenBufferPropertiesQNX");
#endif /* defined(VK_QNX_external_memory_screen_buffer) */
#if defined(VK_VALVE_descriptor_set_host_mapping)
	vkGetDescriptorSetHostMappingVALVE = (PFN_vkGetDescriptorSetHostMappingVALVE)load(context, "vkGetDescriptorSetHostMappingVALVE");
	vkGetDescriptorSetLayoutHostMappingInfoVALVE = (PFN_vkGetDescriptorSetLayoutHostMappingInfoVALVE)load(context, "vkGetDescriptorSetLayoutHostMappingInfoVALVE");
#endif /* defined(VK_VALVE_descriptor_set_host_mapping) */
#if (defined(VK_EXT_extended_dynamic_state)) || (defined(VK_EXT_shader_object))
	vkCmdBindVertexBuffers2EXT = (PFN_vkCmdBindVertexBuffers2EXT)load(context, "vkCmdBindVertexBuffers2EXT");
	vkCmdSetCullModeEXT = (PFN_vkCmdSetCullModeEXT)load(context, "vkCmdSetCullModeEXT");
	vkCmdSetDepthBoundsTestEnableEXT = (PFN_vkCmdSetDepthBoundsTestEnableEXT)load(context, "vkCmdSetDepthBoundsTestEnableEXT");
	vkCmdSetDepthCompareOpEXT = (PFN_vkCmdSetDepthCompareOpEXT)load(context, "vkCmdSetDepthCompareOpEXT");
	vkCmdSetDepthTestEnableEXT = (PFN_vkCmdSetDepthTestEnableEXT)load(context, "vkCmdSetDepthTestEnableEXT");
	vkCmdSetDepthWriteEnableEXT = (PFN_vkCmdSetDepthWriteEnableEXT)load(context, "vkCmdSetDepthWriteEnableEXT");
	vkCmdSetFrontFaceEXT = (PFN_vkCmdSetFrontFaceEXT)load(context, "vkCmdSetFrontFaceEXT");
	vkCmdSetPrimitiveTopologyEXT = (PFN_vkCmdSetPrimitiveTopologyEXT)load(context, "vkCmdSetPrimitiveTopologyEXT");
	vkCmdSetScissorWithCountEXT = (PFN_vkCmdSetScissorWithCountEXT)load(context, "vkCmdSetScissorWithCountEXT");
	vkCmdSetStencilOpEXT = (PFN_vkCmdSetStencilOpEXT)load(context, "vkCmdSetStencilOpEXT");
	vkCmdSetStencilTestEnableEXT = (PFN_vkCmdSetStencilTestEnableEXT)load(context, "vkCmdSetStencilTestEnableEXT");
	vkCmdSetViewportWithCountEXT = (PFN_vkCmdSetViewportWithCountEXT)load(context, "vkCmdSetViewportWithCountEXT");
#endif /* (defined(VK_EXT_extended_dynamic_state)) || (defined(VK_EXT_shader_object)) */
#if (defined(VK_EXT_extended_dynamic_state2)) || (defined(VK_EXT_shader_object))
	vkCmdSetDepthBiasEnableEXT = (PFN_vkCmdSetDepthBiasEnableEXT)load(context, "vkCmdSetDepthBiasEnableEXT");
	vkCmdSetLogicOpEXT = (PFN_vkCmdSetLogicOpEXT)load(context, "vkCmdSetLogicOpEXT");
	vkCmdSetPatchControlPointsEXT = (PFN_vkCmdSetPatchControlPointsEXT)load(context, "vkCmdSetPatchControlPointsEXT");
	vkCmdSetPrimitiveRestartEnableEXT = (PFN_vkCmdSetPrimitiveRestartEnableEXT)load(context, "vkCmdSetPrimitiveRestartEnableEXT");
	vkCmdSetRasterizerDiscardEnableEXT = (PFN_vkCmdSetRasterizerDiscardEnableEXT)load(context, "vkCmdSetRasterizerDiscardEnableEXT");
#endif /* (defined(VK_EXT_extended_dynamic_state2)) || (defined(VK_EXT_shader_object)) */
#if (defined(VK_EXT_extended_dynamic_state3)) || (defined(VK_EXT_shader_object))
	vkCmdSetAlphaToCoverageEnableEXT = (PFN_vkCmdSetAlphaToCoverageEnableEXT)load(context, "vkCmdSetAlphaToCoverageEnableEXT");
	vkCmdSetAlphaToOneEnableEXT = (PFN_vkCmdSetAlphaToOneEnableEXT)load(context, "vkCmdSetAlphaToOneEnableEXT");
	vkCmdSetColorBlendAdvancedEXT = (PFN_vkCmdSetColorBlendAdvancedEXT)load(context, "vkCmdSetColorBlendAdvancedEXT");
	vkCmdSetColorBlendEnableEXT = (PFN_vkCmdSetColorBlendEnableEXT)load(context, "vkCmdSetColorBlendEnableEXT");
	vkCmdSetColorBlendEquationEXT = (PFN_vkCmdSetColorBlendEquationEXT)load(context, "vkCmdSetColorBlendEquationEXT");
	vkCmdSetColorWriteMaskEXT = (PFN_vkCmdSetColorWriteMaskEXT)load(context, "vkCmdSetColorWriteMaskEXT");
	vkCmdSetConservativeRasterizationModeEXT = (PFN_vkCmdSetConservativeRasterizationModeEXT)load(context, "vkCmdSetConservativeRasterizationModeEXT");
	vkCmdSetDepthClampEnableEXT = (PFN_vkCmdSetDepthClampEnableEXT)load(context, "vkCmdSetDepthClampEnableEXT");
	vkCmdSetDepthClipEnableEXT = (PFN_vkCmdSetDepthClipEnableEXT)load(context, "vkCmdSetDepthClipEnableEXT");
	vkCmdSetDepthClipNegativeOneToOneEXT = (PFN_vkCmdSetDepthClipNegativeOneToOneEXT)load(context, "vkCmdSetDepthClipNegativeOneToOneEXT");
	vkCmdSetExtraPrimitiveOverestimationSizeEXT = (PFN_vkCmdSetExtraPrimitiveOverestimationSizeEXT)load(context, "vkCmdSetExtraPrimitiveOverestimationSizeEXT");
	vkCmdSetLineRasterizationModeEXT = (PFN_vkCmdSetLineRasterizationModeEXT)load(context, "vkCmdSetLineRasterizationModeEXT");
	vkCmdSetLineStippleEnableEXT = (PFN_vkCmdSetLineStippleEnableEXT)load(context, "vkCmdSetLineStippleEnableEXT");
	vkCmdSetLogicOpEnableEXT = (PFN_vkCmdSetLogicOpEnableEXT)load(context, "vkCmdSetLogicOpEnableEXT");
	vkCmdSetPolygonModeEXT = (PFN_vkCmdSetPolygonModeEXT)load(context, "vkCmdSetPolygonModeEXT");
	vkCmdSetProvokingVertexModeEXT = (PFN_vkCmdSetProvokingVertexModeEXT)load(context, "vkCmdSetProvokingVertexModeEXT");
	vkCmdSetRasterizationSamplesEXT = (PFN_vkCmdSetRasterizationSamplesEXT)load(context, "vkCmdSetRasterizationSamplesEXT");
	vkCmdSetRasterizationStreamEXT = (PFN_vkCmdSetRasterizationStreamEXT)load(context, "vkCmdSetRasterizationStreamEXT");
	vkCmdSetSampleLocationsEnableEXT = (PFN_vkCmdSetSampleLocationsEnableEXT)load(context, "vkCmdSetSampleLocationsEnableEXT");
	vkCmdSetSampleMaskEXT = (PFN_vkCmdSetSampleMaskEXT)load(context, "vkCmdSetSampleMaskEXT");
	vkCmdSetTessellationDomainOriginEXT = (PFN_vkCmdSetTessellationDomainOriginEXT)load(context, "vkCmdSetTessellationDomainOriginEXT");
#endif /* (defined(VK_EXT_extended_dynamic_state3)) || (defined(VK_EXT_shader_object)) */
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_clip_space_w_scaling)) || (defined(VK_EXT_shader_object) && defined(VK_NV_clip_space_w_scaling))
	vkCmdSetViewportWScalingEnableNV = (PFN_vkCmdSetViewportWScalingEnableNV)load(context, "vkCmdSetViewportWScalingEnableNV");
#endif /* (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_clip_space_w_scaling)) || (defined(VK_EXT_shader_object) && defined(VK_NV_clip_space_w_scaling)) */
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_viewport_swizzle)) || (defined(VK_EXT_shader_object) && defined(VK_NV_viewport_swizzle))
	vkCmdSetViewportSwizzleNV = (PFN_vkCmdSetViewportSwizzleNV)load(context, "vkCmdSetViewportSwizzleNV");
#endif /* (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_viewport_swizzle)) || (defined(VK_EXT_shader_object) && defined(VK_NV_viewport_swizzle)) */
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_fragment_coverage_to_color)) || (defined(VK_EXT_shader_object) && defined(VK_NV_fragment_coverage_to_color))
	vkCmdSetCoverageToColorEnableNV = (PFN_vkCmdSetCoverageToColorEnableNV)load(context, "vkCmdSetCoverageToColorEnableNV");
	vkCmdSetCoverageToColorLocationNV = (PFN_vkCmdSetCoverageToColorLocationNV)load(context, "vkCmdSetCoverageToColorLocationNV");
#endif /* (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_fragment_coverage_to_color)) || (defined(VK_EXT_shader_object) && defined(VK_NV_fragment_coverage_to_color)) */
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_framebuffer_mixed_samples)) || (defined(VK_EXT_shader_object) && defined(VK_NV_framebuffer_mixed_samples))
	vkCmdSetCoverageModulationModeNV = (PFN_vkCmdSetCoverageModulationModeNV)load(context, "vkCmdSetCoverageModulationModeNV");
	vkCmdSetCoverageModulationTableEnableNV = (PFN_vkCmdSetCoverageModulationTableEnableNV)load(context, "vkCmdSetCoverageModulationTableEnableNV");
	vkCmdSetCoverageModulationTableNV = (PFN_vkCmdSetCoverageModulationTableNV)load(context, "vkCmdSetCoverageModulationTableNV");
#endif /* (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_framebuffer_mixed_samples)) || (defined(VK_EXT_shader_object) && defined(VK_NV_framebuffer_mixed_samples)) */
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_shading_rate_image)) || (defined(VK_EXT_shader_object) && defined(VK_NV_shading_rate_image))
	vkCmdSetShadingRateImageEnableNV = (PFN_vkCmdSetShadingRateImageEnableNV)load(context, "vkCmdSetShadingRateImageEnableNV");
#endif /* (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_shading_rate_image)) || (defined(VK_EXT_shader_object) && defined(VK_NV_shading_rate_image)) */
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_representative_fragment_test)) || (defined(VK_EXT_shader_object) && defined(VK_NV_representative_fragment_test))
	vkCmdSetRepresentativeFragmentTestEnableNV = (PFN_vkCmdSetRepresentativeFragmentTestEnableNV)load(context, "vkCmdSetRepresentativeFragmentTestEnableNV");
#endif /* (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_representative_fragment_test)) || (defined(VK_EXT_shader_object) && defined(VK_NV_representative_fragment_test)) */
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_coverage_reduction_mode)) || (defined(VK_EXT_shader_object) && defined(VK_NV_coverage_reduction_mode))
	vkCmdSetCoverageReductionModeNV = (PFN_vkCmdSetCoverageReductionModeNV)load(context, "vkCmdSetCoverageReductionModeNV");
#endif /* (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_coverage_reduction_mode)) || (defined(VK_EXT_shader_object) && defined(VK_NV_coverage_reduction_mode)) */
#if (defined(VK_EXT_full_screen_exclusive) && defined(VK_KHR_device_group)) || (defined(VK_EXT_full_screen_exclusive) && defined(VK_VERSION_1_1))
	vkGetDeviceGroupSurfacePresentModes2EXT = (PFN_vkGetDeviceGroupSurfacePresentModes2EXT)load(context, "vkGetDeviceGroupSurfacePresentModes2EXT");
#endif /* (defined(VK_EXT_full_screen_exclusive) && defined(VK_KHR_device_group)) || (defined(VK_EXT_full_screen_exclusive) && defined(VK_VERSION_1_1)) */
#if (defined(VK_EXT_host_image_copy)) || (defined(VK_EXT_image_compression_control))
	vkGetImageSubresourceLayout2EXT = (PFN_vkGetImageSubresourceLayout2EXT)load(context, "vkGetImageSubresourceLayout2EXT");
#endif /* (defined(VK_EXT_host_image_copy)) || (defined(VK_EXT_image_compression_control)) */
#if (defined(VK_EXT_shader_object)) || (defined(VK_EXT_vertex_input_dynamic_state))
	vkCmdSetVertexInputEXT = (PFN_vkCmdSetVertexInputEXT)load(context, "vkCmdSetVertexInputEXT");
#endif /* (defined(VK_EXT_shader_object)) || (defined(VK_EXT_vertex_input_dynamic_state)) */
#if (defined(VK_KHR_descriptor_update_template) && defined(VK_KHR_push_descriptor)) || (defined(VK_KHR_push_descriptor) && defined(VK_VERSION_1_1)) || (defined(VK_KHR_push_descriptor) && defined(VK_KHR_descriptor_update_template))
	vkCmdPushDescriptorSetWithTemplateKHR = (PFN_vkCmdPushDescriptorSetWithTemplateKHR)load(context, "vkCmdPushDescriptorSetWithTemplateKHR");
#endif /* (defined(VK_KHR_descriptor_update_template) && defined(VK_KHR_push_descriptor)) || (defined(VK_KHR_push_descriptor) && defined(VK_VERSION_1_1)) || (defined(VK_KHR_push_descriptor) && defined(VK_KHR_descriptor_update_template)) */
#if (defined(VK_KHR_device_group) && defined(VK_KHR_surface)) || (defined(VK_KHR_swapchain) && defined(VK_VERSION_1_1))
	vkGetDeviceGroupPresentCapabilitiesKHR = (PFN_vkGetDeviceGroupPresentCapabilitiesKHR)load(context, "vkGetDeviceGroupPresentCapabilitiesKHR");
	vkGetDeviceGroupSurfacePresentModesKHR = (PFN_vkGetDeviceGroupSurfacePresentModesKHR)load(context, "vkGetDeviceGroupSurfacePresentModesKHR");
#endif /* (defined(VK_KHR_device_group) && defined(VK_KHR_surface)) || (defined(VK_KHR_swapchain) && defined(VK_VERSION_1_1)) */
#if (defined(VK_KHR_device_group) && defined(VK_KHR_swapchain)) || (defined(VK_KHR_swapchain) && defined(VK_VERSION_1_1))
	vkAcquireNextImage2KHR = (PFN_vkAcquireNextImage2KHR)load(context, "vkAcquireNextImage2KHR");
#endif /* (defined(VK_KHR_device_group) && defined(VK_KHR_swapchain)) || (defined(VK_KHR_swapchain) && defined(VK_VERSION_1_1)) */
	/* VOLK_GENERATE_LOAD_DEVICE */
}

static void volkGenLoadDeviceTable(struct VolkDeviceTable* table, void* context, PFN_vkVoidFunction (*load)(void*, const char*))
{
	/* VOLK_GENERATE_LOAD_DEVICE_TABLE */
#if defined(VK_VERSION_1_0)
	table->vkAllocateCommandBuffers = (PFN_vkAllocateCommandBuffers)load(context, "vkAllocateCommandBuffers");
	table->vkAllocateDescriptorSets = (PFN_vkAllocateDescriptorSets)load(context, "vkAllocateDescriptorSets");
	table->vkAllocateMemory = (PFN_vkAllocateMemory)load(context, "vkAllocateMemory");
	table->vkBeginCommandBuffer = (PFN_vkBeginCommandBuffer)load(context, "vkBeginCommandBuffer");
	table->vkBindBufferMemory = (PFN_vkBindBufferMemory)load(context, "vkBindBufferMemory");
	table->vkBindImageMemory = (PFN_vkBindImageMemory)load(context, "vkBindImageMemory");
	table->vkCmdBeginQuery = (PFN_vkCmdBeginQuery)load(context, "vkCmdBeginQuery");
	table->vkCmdBeginRenderPass = (PFN_vkCmdBeginRenderPass)load(context, "vkCmdBeginRenderPass");
	table->vkCmdBindDescriptorSets = (PFN_vkCmdBindDescriptorSets)load(context, "vkCmdBindDescriptorSets");
	table->vkCmdBindIndexBuffer = (PFN_vkCmdBindIndexBuffer)load(context, "vkCmdBindIndexBuffer");
	table->vkCmdBindPipeline = (PFN_vkCmdBindPipeline)load(context, "vkCmdBindPipeline");
	table->vkCmdBindVertexBuffers = (PFN_vkCmdBindVertexBuffers)load(context, "vkCmdBindVertexBuffers");
	table->vkCmdBlitImage = (PFN_vkCmdBlitImage)load(context, "vkCmdBlitImage");
	table->vkCmdClearAttachments = (PFN_vkCmdClearAttachments)load(context, "vkCmdClearAttachments");
	table->vkCmdClearColorImage = (PFN_vkCmdClearColorImage)load(context, "vkCmdClearColorImage");
	table->vkCmdClearDepthStencilImage = (PFN_vkCmdClearDepthStencilImage)load(context, "vkCmdClearDepthStencilImage");
	table->vkCmdCopyBuffer = (PFN_vkCmdCopyBuffer)load(context, "vkCmdCopyBuffer");
	table->vkCmdCopyBufferToImage = (PFN_vkCmdCopyBufferToImage)load(context, "vkCmdCopyBufferToImage");
	table->vkCmdCopyImage = (PFN_vkCmdCopyImage)load(context, "vkCmdCopyImage");
	table->vkCmdCopyImageToBuffer = (PFN_vkCmdCopyImageToBuffer)load(context, "vkCmdCopyImageToBuffer");
	table->vkCmdCopyQueryPoolResults = (PFN_vkCmdCopyQueryPoolResults)load(context, "vkCmdCopyQueryPoolResults");
	table->vkCmdDispatch = (PFN_vkCmdDispatch)load(context, "vkCmdDispatch");
	table->vkCmdDispatchIndirect = (PFN_vkCmdDispatchIndirect)load(context, "vkCmdDispatchIndirect");
	table->vkCmdDraw = (PFN_vkCmdDraw)load(context, "vkCmdDraw");
	table->vkCmdDrawIndexed = (PFN_vkCmdDrawIndexed)load(context, "vkCmdDrawIndexed");
	table->vkCmdDrawIndexedIndirect = (PFN_vkCmdDrawIndexedIndirect)load(context, "vkCmdDrawIndexedIndirect");
	table->vkCmdDrawIndirect = (PFN_vkCmdDrawIndirect)load(context, "vkCmdDrawIndirect");
	table->vkCmdEndQuery = (PFN_vkCmdEndQuery)load(context, "vkCmdEndQuery");
	table->vkCmdEndRenderPass = (PFN_vkCmdEndRenderPass)load(context, "vkCmdEndRenderPass");
	table->vkCmdExecuteCommands = (PFN_vkCmdExecuteCommands)load(context, "vkCmdExecuteCommands");
	table->vkCmdFillBuffer = (PFN_vkCmdFillBuffer)load(context, "vkCmdFillBuffer");
	table->vkCmdNextSubpass = (PFN_vkCmdNextSubpass)load(context, "vkCmdNextSubpass");
	table->vkCmdPipelineBarrier = (PFN_vkCmdPipelineBarrier)load(context, "vkCmdPipelineBarrier");
	table->vkCmdPushConstants = (PFN_vkCmdPushConstants)load(context, "vkCmdPushConstants");
	table->vkCmdResetEvent = (PFN_vkCmdResetEvent)load(context, "vkCmdResetEvent");
	table->vkCmdResetQueryPool = (PFN_vkCmdResetQueryPool)load(context, "vkCmdResetQueryPool");
	table->vkCmdResolveImage = (PFN_vkCmdResolveImage)load(context, "vkCmdResolveImage");
	table->vkCmdSetBlendConstants = (PFN_vkCmdSetBlendConstants)load(context, "vkCmdSetBlendConstants");
	table->vkCmdSetDepthBias = (PFN_vkCmdSetDepthBias)load(context, "vkCmdSetDepthBias");
	table->vkCmdSetDepthBounds = (PFN_vkCmdSetDepthBounds)load(context, "vkCmdSetDepthBounds");
	table->vkCmdSetEvent = (PFN_vkCmdSetEvent)load(context, "vkCmdSetEvent");
	table->vkCmdSetLineWidth = (PFN_vkCmdSetLineWidth)load(context, "vkCmdSetLineWidth");
	table->vkCmdSetScissor = (PFN_vkCmdSetScissor)load(context, "vkCmdSetScissor");
	table->vkCmdSetStencilCompareMask = (PFN_vkCmdSetStencilCompareMask)load(context, "vkCmdSetStencilCompareMask");
	table->vkCmdSetStencilReference = (PFN_vkCmdSetStencilReference)load(context, "vkCmdSetStencilReference");
	table->vkCmdSetStencilWriteMask = (PFN_vkCmdSetStencilWriteMask)load(context, "vkCmdSetStencilWriteMask");
	table->vkCmdSetViewport = (PFN_vkCmdSetViewport)load(context, "vkCmdSetViewport");
	table->vkCmdUpdateBuffer = (PFN_vkCmdUpdateBuffer)load(context, "vkCmdUpdateBuffer");
	table->vkCmdWaitEvents = (PFN_vkCmdWaitEvents)load(context, "vkCmdWaitEvents");
	table->vkCmdWriteTimestamp = (PFN_vkCmdWriteTimestamp)load(context, "vkCmdWriteTimestamp");
	table->vkCreateBuffer = (PFN_vkCreateBuffer)load(context, "vkCreateBuffer");
	table->vkCreateBufferView = (PFN_vkCreateBufferView)load(context, "vkCreateBufferView");
	table->vkCreateCommandPool = (PFN_vkCreateCommandPool)load(context, "vkCreateCommandPool");
	table->vkCreateComputePipelines = (PFN_vkCreateComputePipelines)load(context, "vkCreateComputePipelines");
	table->vkCreateDescriptorPool = (PFN_vkCreateDescriptorPool)load(context, "vkCreateDescriptorPool");
	table->vkCreateDescriptorSetLayout = (PFN_vkCreateDescriptorSetLayout)load(context, "vkCreateDescriptorSetLayout");
	table->vkCreateEvent = (PFN_vkCreateEvent)load(context, "vkCreateEvent");
	table->vkCreateFence = (PFN_vkCreateFence)load(context, "vkCreateFence");
	table->vkCreateFramebuffer = (PFN_vkCreateFramebuffer)load(context, "vkCreateFramebuffer");
	table->vkCreateGraphicsPipelines = (PFN_vkCreateGraphicsPipelines)load(context, "vkCreateGraphicsPipelines");
	table->vkCreateImage = (PFN_vkCreateImage)load(context, "vkCreateImage");
	table->vkCreateImageView = (PFN_vkCreateImageView)load(context, "vkCreateImageView");
	table->vkCreatePipelineCache = (PFN_vkCreatePipelineCache)load(context, "vkCreatePipelineCache");
	table->vkCreatePipelineLayout = (PFN_vkCreatePipelineLayout)load(context, "vkCreatePipelineLayout");
	table->vkCreateQueryPool = (PFN_vkCreateQueryPool)load(context, "vkCreateQueryPool");
	table->vkCreateRenderPass = (PFN_vkCreateRenderPass)load(context, "vkCreateRenderPass");
	table->vkCreateSampler = (PFN_vkCreateSampler)load(context, "vkCreateSampler");
	table->vkCreateSemaphore = (PFN_vkCreateSemaphore)load(context, "vkCreateSemaphore");
	table->vkCreateShaderModule = (PFN_vkCreateShaderModule)load(context, "vkCreateShaderModule");
	table->vkDestroyBuffer = (PFN_vkDestroyBuffer)load(context, "vkDestroyBuffer");
	table->vkDestroyBufferView = (PFN_vkDestroyBufferView)load(context, "vkDestroyBufferView");
	table->vkDestroyCommandPool = (PFN_vkDestroyCommandPool)load(context, "vkDestroyCommandPool");
	table->vkDestroyDescriptorPool = (PFN_vkDestroyDescriptorPool)load(context, "vkDestroyDescriptorPool");
	table->vkDestroyDescriptorSetLayout = (PFN_vkDestroyDescriptorSetLayout)load(context, "vkDestroyDescriptorSetLayout");
	table->vkDestroyDevice = (PFN_vkDestroyDevice)load(context, "vkDestroyDevice");
	table->vkDestroyEvent = (PFN_vkDestroyEvent)load(context, "vkDestroyEvent");
	table->vkDestroyFence = (PFN_vkDestroyFence)load(context, "vkDestroyFence");
	table->vkDestroyFramebuffer = (PFN_vkDestroyFramebuffer)load(context, "vkDestroyFramebuffer");
	table->vkDestroyImage = (PFN_vkDestroyImage)load(context, "vkDestroyImage");
	table->vkDestroyImageView = (PFN_vkDestroyImageView)load(context, "vkDestroyImageView");
	table->vkDestroyPipeline = (PFN_vkDestroyPipeline)load(context, "vkDestroyPipeline");
	table->vkDestroyPipelineCache = (PFN_vkDestroyPipelineCache)load(context, "vkDestroyPipelineCache");
	table->vkDestroyPipelineLayout = (PFN_vkDestroyPipelineLayout)load(context, "vkDestroyPipelineLayout");
	table->vkDestroyQueryPool = (PFN_vkDestroyQueryPool)load(context, "vkDestroyQueryPool");
	table->vkDestroyRenderPass = (PFN_vkDestroyRenderPass)load(context, "vkDestroyRenderPass");
	table->vkDestroySampler = (PFN_vkDestroySampler)load(context, "vkDestroySampler");
	table->vkDestroySemaphore = (PFN_vkDestroySemaphore)load(context, "vkDestroySemaphore");
	table->vkDestroyShaderModule = (PFN_vkDestroyShaderModule)load(context, "vkDestroyShaderModule");
	table->vkDeviceWaitIdle = (PFN_vkDeviceWaitIdle)load(context, "vkDeviceWaitIdle");
	table->vkEndCommandBuffer = (PFN_vkEndCommandBuffer)load(context, "vkEndCommandBuffer");
	table->vkFlushMappedMemoryRanges = (PFN_vkFlushMappedMemoryRanges)load(context, "vkFlushMappedMemoryRanges");
	table->vkFreeCommandBuffers = (PFN_vkFreeCommandBuffers)load(context, "vkFreeCommandBuffers");
	table->vkFreeDescriptorSets = (PFN_vkFreeDescriptorSets)load(context, "vkFreeDescriptorSets");
	table->vkFreeMemory = (PFN_vkFreeMemory)load(context, "vkFreeMemory");
	table->vkGetBufferMemoryRequirements = (PFN_vkGetBufferMemoryRequirements)load(context, "vkGetBufferMemoryRequirements");
	table->vkGetDeviceMemoryCommitment = (PFN_vkGetDeviceMemoryCommitment)load(context, "vkGetDeviceMemoryCommitment");
	table->vkGetDeviceQueue = (PFN_vkGetDeviceQueue)load(context, "vkGetDeviceQueue");
	table->vkGetEventStatus = (PFN_vkGetEventStatus)load(context, "vkGetEventStatus");
	table->vkGetFenceStatus = (PFN_vkGetFenceStatus)load(context, "vkGetFenceStatus");
	table->vkGetImageMemoryRequirements = (PFN_vkGetImageMemoryRequirements)load(context, "vkGetImageMemoryRequirements");
	table->vkGetImageSparseMemoryRequirements = (PFN_vkGetImageSparseMemoryRequirements)load(context, "vkGetImageSparseMemoryRequirements");
	table->vkGetImageSubresourceLayout = (PFN_vkGetImageSubresourceLayout)load(context, "vkGetImageSubresourceLayout");
	table->vkGetPipelineCacheData = (PFN_vkGetPipelineCacheData)load(context, "vkGetPipelineCacheData");
	table->vkGetQueryPoolResults = (PFN_vkGetQueryPoolResults)load(context, "vkGetQueryPoolResults");
	table->vkGetRenderAreaGranularity = (PFN_vkGetRenderAreaGranularity)load(context, "vkGetRenderAreaGranularity");
	table->vkInvalidateMappedMemoryRanges = (PFN_vkInvalidateMappedMemoryRanges)load(context, "vkInvalidateMappedMemoryRanges");
	table->vkMapMemory = (PFN_vkMapMemory)load(context, "vkMapMemory");
	table->vkMergePipelineCaches = (PFN_vkMergePipelineCaches)load(context, "vkMergePipelineCaches");
	table->vkQueueBindSparse = (PFN_vkQueueBindSparse)load(context, "vkQueueBindSparse");
	table->vkQueueSubmit = (PFN_vkQueueSubmit)load(context, "vkQueueSubmit");
	table->vkQueueWaitIdle = (PFN_vkQueueWaitIdle)load(context, "vkQueueWaitIdle");
	table->vkResetCommandBuffer = (PFN_vkResetCommandBuffer)load(context, "vkResetCommandBuffer");
	table->vkResetCommandPool = (PFN_vkResetCommandPool)load(context, "vkResetCommandPool");
	table->vkResetDescriptorPool = (PFN_vkResetDescriptorPool)load(context, "vkResetDescriptorPool");
	table->vkResetEvent = (PFN_vkResetEvent)load(context, "vkResetEvent");
	table->vkResetFences = (PFN_vkResetFences)load(context, "vkResetFences");
	table->vkSetEvent = (PFN_vkSetEvent)load(context, "vkSetEvent");
	table->vkUnmapMemory = (PFN_vkUnmapMemory)load(context, "vkUnmapMemory");
	table->vkUpdateDescriptorSets = (PFN_vkUpdateDescriptorSets)load(context, "vkUpdateDescriptorSets");
	table->vkWaitForFences = (PFN_vkWaitForFences)load(context, "vkWaitForFences");
#endif /* defined(VK_VERSION_1_0) */
#if defined(VK_VERSION_1_1)
	table->vkBindBufferMemory2 = (PFN_vkBindBufferMemory2)load(context, "vkBindBufferMemory2");
	table->vkBindImageMemory2 = (PFN_vkBindImageMemory2)load(context, "vkBindImageMemory2");
	table->vkCmdDispatchBase = (PFN_vkCmdDispatchBase)load(context, "vkCmdDispatchBase");
	table->vkCmdSetDeviceMask = (PFN_vkCmdSetDeviceMask)load(context, "vkCmdSetDeviceMask");
	table->vkCreateDescriptorUpdateTemplate = (PFN_vkCreateDescriptorUpdateTemplate)load(context, "vkCreateDescriptorUpdateTemplate");
	table->vkCreateSamplerYcbcrConversion = (PFN_vkCreateSamplerYcbcrConversion)load(context, "vkCreateSamplerYcbcrConversion");
	table->vkDestroyDescriptorUpdateTemplate = (PFN_vkDestroyDescriptorUpdateTemplate)load(context, "vkDestroyDescriptorUpdateTemplate");
	table->vkDestroySamplerYcbcrConversion = (PFN_vkDestroySamplerYcbcrConversion)load(context, "vkDestroySamplerYcbcrConversion");
	table->vkGetBufferMemoryRequirements2 = (PFN_vkGetBufferMemoryRequirements2)load(context, "vkGetBufferMemoryRequirements2");
	table->vkGetDescriptorSetLayoutSupport = (PFN_vkGetDescriptorSetLayoutSupport)load(context, "vkGetDescriptorSetLayoutSupport");
	table->vkGetDeviceGroupPeerMemoryFeatures = (PFN_vkGetDeviceGroupPeerMemoryFeatures)load(context, "vkGetDeviceGroupPeerMemoryFeatures");
	table->vkGetDeviceQueue2 = (PFN_vkGetDeviceQueue2)load(context, "vkGetDeviceQueue2");
	table->vkGetImageMemoryRequirements2 = (PFN_vkGetImageMemoryRequirements2)load(context, "vkGetImageMemoryRequirements2");
	table->vkGetImageSparseMemoryRequirements2 = (PFN_vkGetImageSparseMemoryRequirements2)load(context, "vkGetImageSparseMemoryRequirements2");
	table->vkTrimCommandPool = (PFN_vkTrimCommandPool)load(context, "vkTrimCommandPool");
	table->vkUpdateDescriptorSetWithTemplate = (PFN_vkUpdateDescriptorSetWithTemplate)load(context, "vkUpdateDescriptorSetWithTemplate");
#endif /* defined(VK_VERSION_1_1) */
#if defined(VK_VERSION_1_2)
	table->vkCmdBeginRenderPass2 = (PFN_vkCmdBeginRenderPass2)load(context, "vkCmdBeginRenderPass2");
	table->vkCmdDrawIndexedIndirectCount = (PFN_vkCmdDrawIndexedIndirectCount)load(context, "vkCmdDrawIndexedIndirectCount");
	table->vkCmdDrawIndirectCount = (PFN_vkCmdDrawIndirectCount)load(context, "vkCmdDrawIndirectCount");
	table->vkCmdEndRenderPass2 = (PFN_vkCmdEndRenderPass2)load(context, "vkCmdEndRenderPass2");
	table->vkCmdNextSubpass2 = (PFN_vkCmdNextSubpass2)load(context, "vkCmdNextSubpass2");
	table->vkCreateRenderPass2 = (PFN_vkCreateRenderPass2)load(context, "vkCreateRenderPass2");
	table->vkGetBufferDeviceAddress = (PFN_vkGetBufferDeviceAddress)load(context, "vkGetBufferDeviceAddress");
	table->vkGetBufferOpaqueCaptureAddress = (PFN_vkGetBufferOpaqueCaptureAddress)load(context, "vkGetBufferOpaqueCaptureAddress");
	table->vkGetDeviceMemoryOpaqueCaptureAddress = (PFN_vkGetDeviceMemoryOpaqueCaptureAddress)load(context, "vkGetDeviceMemoryOpaqueCaptureAddress");
	table->vkGetSemaphoreCounterValue = (PFN_vkGetSemaphoreCounterValue)load(context, "vkGetSemaphoreCounterValue");
	table->vkResetQueryPool = (PFN_vkResetQueryPool)load(context, "vkResetQueryPool");
	table->vkSignalSemaphore = (PFN_vkSignalSemaphore)load(context, "vkSignalSemaphore");
	table->vkWaitSemaphores = (PFN_vkWaitSemaphores)load(context, "vkWaitSemaphores");
#endif /* defined(VK_VERSION_1_2) */
#if defined(VK_VERSION_1_3)
	table->vkCmdBeginRendering = (PFN_vkCmdBeginRendering)load(context, "vkCmdBeginRendering");
	table->vkCmdBindVertexBuffers2 = (PFN_vkCmdBindVertexBuffers2)load(context, "vkCmdBindVertexBuffers2");
	table->vkCmdBlitImage2 = (PFN_vkCmdBlitImage2)load(context, "vkCmdBlitImage2");
	table->vkCmdCopyBuffer2 = (PFN_vkCmdCopyBuffer2)load(context, "vkCmdCopyBuffer2");
	table->vkCmdCopyBufferToImage2 = (PFN_vkCmdCopyBufferToImage2)load(context, "vkCmdCopyBufferToImage2");
	table->vkCmdCopyImage2 = (PFN_vkCmdCopyImage2)load(context, "vkCmdCopyImage2");
	table->vkCmdCopyImageToBuffer2 = (PFN_vkCmdCopyImageToBuffer2)load(context, "vkCmdCopyImageToBuffer2");
	table->vkCmdEndRendering = (PFN_vkCmdEndRendering)load(context, "vkCmdEndRendering");
	table->vkCmdPipelineBarrier2 = (PFN_vkCmdPipelineBarrier2)load(context, "vkCmdPipelineBarrier2");
	table->vkCmdResetEvent2 = (PFN_vkCmdResetEvent2)load(context, "vkCmdResetEvent2");
	table->vkCmdResolveImage2 = (PFN_vkCmdResolveImage2)load(context, "vkCmdResolveImage2");
	table->vkCmdSetCullMode = (PFN_vkCmdSetCullMode)load(context, "vkCmdSetCullMode");
	table->vkCmdSetDepthBiasEnable = (PFN_vkCmdSetDepthBiasEnable)load(context, "vkCmdSetDepthBiasEnable");
	table->vkCmdSetDepthBoundsTestEnable = (PFN_vkCmdSetDepthBoundsTestEnable)load(context, "vkCmdSetDepthBoundsTestEnable");
	table->vkCmdSetDepthCompareOp = (PFN_vkCmdSetDepthCompareOp)load(context, "vkCmdSetDepthCompareOp");
	table->vkCmdSetDepthTestEnable = (PFN_vkCmdSetDepthTestEnable)load(context, "vkCmdSetDepthTestEnable");
	table->vkCmdSetDepthWriteEnable = (PFN_vkCmdSetDepthWriteEnable)load(context, "vkCmdSetDepthWriteEnable");
	table->vkCmdSetEvent2 = (PFN_vkCmdSetEvent2)load(context, "vkCmdSetEvent2");
	table->vkCmdSetFrontFace = (PFN_vkCmdSetFrontFace)load(context, "vkCmdSetFrontFace");
	table->vkCmdSetPrimitiveRestartEnable = (PFN_vkCmdSetPrimitiveRestartEnable)load(context, "vkCmdSetPrimitiveRestartEnable");
	table->vkCmdSetPrimitiveTopology = (PFN_vkCmdSetPrimitiveTopology)load(context, "vkCmdSetPrimitiveTopology");
	table->vkCmdSetRasterizerDiscardEnable = (PFN_vkCmdSetRasterizerDiscardEnable)load(context, "vkCmdSetRasterizerDiscardEnable");
	table->vkCmdSetScissorWithCount = (PFN_vkCmdSetScissorWithCount)load(context, "vkCmdSetScissorWithCount");
	table->vkCmdSetStencilOp = (PFN_vkCmdSetStencilOp)load(context, "vkCmdSetStencilOp");
	table->vkCmdSetStencilTestEnable = (PFN_vkCmdSetStencilTestEnable)load(context, "vkCmdSetStencilTestEnable");
	table->vkCmdSetViewportWithCount = (PFN_vkCmdSetViewportWithCount)load(context, "vkCmdSetViewportWithCount");
	table->vkCmdWaitEvents2 = (PFN_vkCmdWaitEvents2)load(context, "vkCmdWaitEvents2");
	table->vkCmdWriteTimestamp2 = (PFN_vkCmdWriteTimestamp2)load(context, "vkCmdWriteTimestamp2");
	table->vkCreatePrivateDataSlot = (PFN_vkCreatePrivateDataSlot)load(context, "vkCreatePrivateDataSlot");
	table->vkDestroyPrivateDataSlot = (PFN_vkDestroyPrivateDataSlot)load(context, "vkDestroyPrivateDataSlot");
	table->vkGetDeviceBufferMemoryRequirements = (PFN_vkGetDeviceBufferMemoryRequirements)load(context, "vkGetDeviceBufferMemoryRequirements");
	table->vkGetDeviceImageMemoryRequirements = (PFN_vkGetDeviceImageMemoryRequirements)load(context, "vkGetDeviceImageMemoryRequirements");
	table->vkGetDeviceImageSparseMemoryRequirements = (PFN_vkGetDeviceImageSparseMemoryRequirements)load(context, "vkGetDeviceImageSparseMemoryRequirements");
	table->vkGetPrivateData = (PFN_vkGetPrivateData)load(context, "vkGetPrivateData");
	table->vkQueueSubmit2 = (PFN_vkQueueSubmit2)load(context, "vkQueueSubmit2");
	table->vkSetPrivateData = (PFN_vkSetPrivateData)load(context, "vkSetPrivateData");
#endif /* defined(VK_VERSION_1_3) */
#if defined(VK_AMDX_shader_enqueue)
	table->vkCmdDispatchGraphAMDX = (PFN_vkCmdDispatchGraphAMDX)load(context, "vkCmdDispatchGraphAMDX");
	table->vkCmdDispatchGraphIndirectAMDX = (PFN_vkCmdDispatchGraphIndirectAMDX)load(context, "vkCmdDispatchGraphIndirectAMDX");
	table->vkCmdDispatchGraphIndirectCountAMDX = (PFN_vkCmdDispatchGraphIndirectCountAMDX)load(context, "vkCmdDispatchGraphIndirectCountAMDX");
	table->vkCmdInitializeGraphScratchMemoryAMDX = (PFN_vkCmdInitializeGraphScratchMemoryAMDX)load(context, "vkCmdInitializeGraphScratchMemoryAMDX");
	table->vkCreateExecutionGraphPipelinesAMDX = (PFN_vkCreateExecutionGraphPipelinesAMDX)load(context, "vkCreateExecutionGraphPipelinesAMDX");
	table->vkGetExecutionGraphPipelineNodeIndexAMDX = (PFN_vkGetExecutionGraphPipelineNodeIndexAMDX)load(context, "vkGetExecutionGraphPipelineNodeIndexAMDX");
	table->vkGetExecutionGraphPipelineScratchSizeAMDX = (PFN_vkGetExecutionGraphPipelineScratchSizeAMDX)load(context, "vkGetExecutionGraphPipelineScratchSizeAMDX");
#endif /* defined(VK_AMDX_shader_enqueue) */
#if defined(VK_AMD_buffer_marker)
	table->vkCmdWriteBufferMarkerAMD = (PFN_vkCmdWriteBufferMarkerAMD)load(context, "vkCmdWriteBufferMarkerAMD");
#endif /* defined(VK_AMD_buffer_marker) */
#if defined(VK_AMD_display_native_hdr)
	table->vkSetLocalDimmingAMD = (PFN_vkSetLocalDimmingAMD)load(context, "vkSetLocalDimmingAMD");
#endif /* defined(VK_AMD_display_native_hdr) */
#if defined(VK_AMD_draw_indirect_count)
	table->vkCmdDrawIndexedIndirectCountAMD = (PFN_vkCmdDrawIndexedIndirectCountAMD)load(context, "vkCmdDrawIndexedIndirectCountAMD");
	table->vkCmdDrawIndirectCountAMD = (PFN_vkCmdDrawIndirectCountAMD)load(context, "vkCmdDrawIndirectCountAMD");
#endif /* defined(VK_AMD_draw_indirect_count) */
#if defined(VK_AMD_shader_info)
	table->vkGetShaderInfoAMD = (PFN_vkGetShaderInfoAMD)load(context, "vkGetShaderInfoAMD");
#endif /* defined(VK_AMD_shader_info) */
#if defined(VK_ANDROID_external_memory_android_hardware_buffer)
	table->vkGetAndroidHardwareBufferPropertiesANDROID = (PFN_vkGetAndroidHardwareBufferPropertiesANDROID)load(context, "vkGetAndroidHardwareBufferPropertiesANDROID");
	table->vkGetMemoryAndroidHardwareBufferANDROID = (PFN_vkGetMemoryAndroidHardwareBufferANDROID)load(context, "vkGetMemoryAndroidHardwareBufferANDROID");
#endif /* defined(VK_ANDROID_external_memory_android_hardware_buffer) */
#if defined(VK_EXT_attachment_feedback_loop_dynamic_state)
	table->vkCmdSetAttachmentFeedbackLoopEnableEXT = (PFN_vkCmdSetAttachmentFeedbackLoopEnableEXT)load(context, "vkCmdSetAttachmentFeedbackLoopEnableEXT");
#endif /* defined(VK_EXT_attachment_feedback_loop_dynamic_state) */
#if defined(VK_EXT_buffer_device_address)
	table->vkGetBufferDeviceAddressEXT = (PFN_vkGetBufferDeviceAddressEXT)load(context, "vkGetBufferDeviceAddressEXT");
#endif /* defined(VK_EXT_buffer_device_address) */
#if defined(VK_EXT_calibrated_timestamps)
	table->vkGetCalibratedTimestampsEXT = (PFN_vkGetCalibratedTimestampsEXT)load(context, "vkGetCalibratedTimestampsEXT");
#endif /* defined(VK_EXT_calibrated_timestamps) */
#if defined(VK_EXT_color_write_enable)
	table->vkCmdSetColorWriteEnableEXT = (PFN_vkCmdSetColorWriteEnableEXT)load(context, "vkCmdSetColorWriteEnableEXT");
#endif /* defined(VK_EXT_color_write_enable) */
#if defined(VK_EXT_conditional_rendering)
	table->vkCmdBeginConditionalRenderingEXT = (PFN_vkCmdBeginConditionalRenderingEXT)load(context, "vkCmdBeginConditionalRenderingEXT");
	table->vkCmdEndConditionalRenderingEXT = (PFN_vkCmdEndConditionalRenderingEXT)load(context, "vkCmdEndConditionalRenderingEXT");
#endif /* defined(VK_EXT_conditional_rendering) */
#if defined(VK_EXT_debug_marker)
	table->vkCmdDebugMarkerBeginEXT = (PFN_vkCmdDebugMarkerBeginEXT)load(context, "vkCmdDebugMarkerBeginEXT");
	table->vkCmdDebugMarkerEndEXT = (PFN_vkCmdDebugMarkerEndEXT)load(context, "vkCmdDebugMarkerEndEXT");
	table->vkCmdDebugMarkerInsertEXT = (PFN_vkCmdDebugMarkerInsertEXT)load(context, "vkCmdDebugMarkerInsertEXT");
	table->vkDebugMarkerSetObjectNameEXT = (PFN_vkDebugMarkerSetObjectNameEXT)load(context, "vkDebugMarkerSetObjectNameEXT");
	table->vkDebugMarkerSetObjectTagEXT = (PFN_vkDebugMarkerSetObjectTagEXT)load(context, "vkDebugMarkerSetObjectTagEXT");
#endif /* defined(VK_EXT_debug_marker) */
#if defined(VK_EXT_depth_bias_control)
	table->vkCmdSetDepthBias2EXT = (PFN_vkCmdSetDepthBias2EXT)load(context, "vkCmdSetDepthBias2EXT");
#endif /* defined(VK_EXT_depth_bias_control) */
#if defined(VK_EXT_descriptor_buffer)
	table->vkCmdBindDescriptorBufferEmbeddedSamplersEXT = (PFN_vkCmdBindDescriptorBufferEmbeddedSamplersEXT)load(context, "vkCmdBindDescriptorBufferEmbeddedSamplersEXT");
	table->vkCmdBindDescriptorBuffersEXT = (PFN_vkCmdBindDescriptorBuffersEXT)load(context, "vkCmdBindDescriptorBuffersEXT");
	table->vkCmdSetDescriptorBufferOffsetsEXT = (PFN_vkCmdSetDescriptorBufferOffsetsEXT)load(context, "vkCmdSetDescriptorBufferOffsetsEXT");
	table->vkGetBufferOpaqueCaptureDescriptorDataEXT = (PFN_vkGetBufferOpaqueCaptureDescriptorDataEXT)load(context, "vkGetBufferOpaqueCaptureDescriptorDataEXT");
	table->vkGetDescriptorEXT = (PFN_vkGetDescriptorEXT)load(context, "vkGetDescriptorEXT");
	table->vkGetDescriptorSetLayoutBindingOffsetEXT = (PFN_vkGetDescriptorSetLayoutBindingOffsetEXT)load(context, "vkGetDescriptorSetLayoutBindingOffsetEXT");
	table->vkGetDescriptorSetLayoutSizeEXT = (PFN_vkGetDescriptorSetLayoutSizeEXT)load(context, "vkGetDescriptorSetLayoutSizeEXT");
	table->vkGetImageOpaqueCaptureDescriptorDataEXT = (PFN_vkGetImageOpaqueCaptureDescriptorDataEXT)load(context, "vkGetImageOpaqueCaptureDescriptorDataEXT");
	table->vkGetImageViewOpaqueCaptureDescriptorDataEXT = (PFN_vkGetImageViewOpaqueCaptureDescriptorDataEXT)load(context, "vkGetImageViewOpaqueCaptureDescriptorDataEXT");
	table->vkGetSamplerOpaqueCaptureDescriptorDataEXT = (PFN_vkGetSamplerOpaqueCaptureDescriptorDataEXT)load(context, "vkGetSamplerOpaqueCaptureDescriptorDataEXT");
#endif /* defined(VK_EXT_descriptor_buffer) */
#if defined(VK_EXT_descriptor_buffer) && (defined(VK_KHR_acceleration_structure) || defined(VK_NV_ray_tracing))
	table->vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT = (PFN_vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT)load(context, "vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT");
#endif /* defined(VK_EXT_descriptor_buffer) && (defined(VK_KHR_acceleration_structure) || defined(VK_NV_ray_tracing)) */
#if defined(VK_EXT_device_fault)
	table->vkGetDeviceFaultInfoEXT = (PFN_vkGetDeviceFaultInfoEXT)load(context, "vkGetDeviceFaultInfoEXT");
#endif /* defined(VK_EXT_device_fault) */
#if defined(VK_EXT_discard_rectangles)
	table->vkCmdSetDiscardRectangleEXT = (PFN_vkCmdSetDiscardRectangleEXT)load(context, "vkCmdSetDiscardRectangleEXT");
#endif /* defined(VK_EXT_discard_rectangles) */
#if defined(VK_EXT_discard_rectangles) && VK_EXT_DISCARD_RECTANGLES_SPEC_VERSION >= 2
	table->vkCmdSetDiscardRectangleEnableEXT = (PFN_vkCmdSetDiscardRectangleEnableEXT)load(context, "vkCmdSetDiscardRectangleEnableEXT");
	table->vkCmdSetDiscardRectangleModeEXT = (PFN_vkCmdSetDiscardRectangleModeEXT)load(context, "vkCmdSetDiscardRectangleModeEXT");
#endif /* defined(VK_EXT_discard_rectangles) && VK_EXT_DISCARD_RECTANGLES_SPEC_VERSION >= 2 */
#if defined(VK_EXT_display_control)
	table->vkDisplayPowerControlEXT = (PFN_vkDisplayPowerControlEXT)load(context, "vkDisplayPowerControlEXT");
	table->vkGetSwapchainCounterEXT = (PFN_vkGetSwapchainCounterEXT)load(context, "vkGetSwapchainCounterEXT");
	table->vkRegisterDeviceEventEXT = (PFN_vkRegisterDeviceEventEXT)load(context, "vkRegisterDeviceEventEXT");
	table->vkRegisterDisplayEventEXT = (PFN_vkRegisterDisplayEventEXT)load(context, "vkRegisterDisplayEventEXT");
#endif /* defined(VK_EXT_display_control) */
#if defined(VK_EXT_external_memory_host)
	table->vkGetMemoryHostPointerPropertiesEXT = (PFN_vkGetMemoryHostPointerPropertiesEXT)load(context, "vkGetMemoryHostPointerPropertiesEXT");
#endif /* defined(VK_EXT_external_memory_host) */
#if defined(VK_EXT_full_screen_exclusive)
	table->vkAcquireFullScreenExclusiveModeEXT = (PFN_vkAcquireFullScreenExclusiveModeEXT)load(context, "vkAcquireFullScreenExclusiveModeEXT");
	table->vkReleaseFullScreenExclusiveModeEXT = (PFN_vkReleaseFullScreenExclusiveModeEXT)load(context, "vkReleaseFullScreenExclusiveModeEXT");
#endif /* defined(VK_EXT_full_screen_exclusive) */
#if defined(VK_EXT_hdr_metadata)
	table->vkSetHdrMetadataEXT = (PFN_vkSetHdrMetadataEXT)load(context, "vkSetHdrMetadataEXT");
#endif /* defined(VK_EXT_hdr_metadata) */
#if defined(VK_EXT_host_image_copy)
	table->vkCopyImageToImageEXT = (PFN_vkCopyImageToImageEXT)load(context, "vkCopyImageToImageEXT");
	table->vkCopyImageToMemoryEXT = (PFN_vkCopyImageToMemoryEXT)load(context, "vkCopyImageToMemoryEXT");
	table->vkCopyMemoryToImageEXT = (PFN_vkCopyMemoryToImageEXT)load(context, "vkCopyMemoryToImageEXT");
	table->vkTransitionImageLayoutEXT = (PFN_vkTransitionImageLayoutEXT)load(context, "vkTransitionImageLayoutEXT");
#endif /* defined(VK_EXT_host_image_copy) */
#if defined(VK_EXT_host_query_reset)
	table->vkResetQueryPoolEXT = (PFN_vkResetQueryPoolEXT)load(context, "vkResetQueryPoolEXT");
#endif /* defined(VK_EXT_host_query_reset) */
#if defined(VK_EXT_image_drm_format_modifier)
	table->vkGetImageDrmFormatModifierPropertiesEXT = (PFN_vkGetImageDrmFormatModifierPropertiesEXT)load(context, "vkGetImageDrmFormatModifierPropertiesEXT");
#endif /* defined(VK_EXT_image_drm_format_modifier) */
#if defined(VK_EXT_line_rasterization)
	table->vkCmdSetLineStippleEXT = (PFN_vkCmdSetLineStippleEXT)load(context, "vkCmdSetLineStippleEXT");
#endif /* defined(VK_EXT_line_rasterization) */
#if defined(VK_EXT_mesh_shader)
	table->vkCmdDrawMeshTasksEXT = (PFN_vkCmdDrawMeshTasksEXT)load(context, "vkCmdDrawMeshTasksEXT");
	table->vkCmdDrawMeshTasksIndirectCountEXT = (PFN_vkCmdDrawMeshTasksIndirectCountEXT)load(context, "vkCmdDrawMeshTasksIndirectCountEXT");
	table->vkCmdDrawMeshTasksIndirectEXT = (PFN_vkCmdDrawMeshTasksIndirectEXT)load(context, "vkCmdDrawMeshTasksIndirectEXT");
#endif /* defined(VK_EXT_mesh_shader) */
#if defined(VK_EXT_metal_objects)
	table->vkExportMetalObjectsEXT = (PFN_vkExportMetalObjectsEXT)load(context, "vkExportMetalObjectsEXT");
#endif /* defined(VK_EXT_metal_objects) */
#if defined(VK_EXT_multi_draw)
	table->vkCmdDrawMultiEXT = (PFN_vkCmdDrawMultiEXT)load(context, "vkCmdDrawMultiEXT");
	table->vkCmdDrawMultiIndexedEXT = (PFN_vkCmdDrawMultiIndexedEXT)load(context, "vkCmdDrawMultiIndexedEXT");
#endif /* defined(VK_EXT_multi_draw) */
#if defined(VK_EXT_opacity_micromap)
	table->vkBuildMicromapsEXT = (PFN_vkBuildMicromapsEXT)load(context, "vkBuildMicromapsEXT");
	table->vkCmdBuildMicromapsEXT = (PFN_vkCmdBuildMicromapsEXT)load(context, "vkCmdBuildMicromapsEXT");
	table->vkCmdCopyMemoryToMicromapEXT = (PFN_vkCmdCopyMemoryToMicromapEXT)load(context, "vkCmdCopyMemoryToMicromapEXT");
	table->vkCmdCopyMicromapEXT = (PFN_vkCmdCopyMicromapEXT)load(context, "vkCmdCopyMicromapEXT");
	table->vkCmdCopyMicromapToMemoryEXT = (PFN_vkCmdCopyMicromapToMemoryEXT)load(context, "vkCmdCopyMicromapToMemoryEXT");
	table->vkCmdWriteMicromapsPropertiesEXT = (PFN_vkCmdWriteMicromapsPropertiesEXT)load(context, "vkCmdWriteMicromapsPropertiesEXT");
	table->vkCopyMemoryToMicromapEXT = (PFN_vkCopyMemoryToMicromapEXT)load(context, "vkCopyMemoryToMicromapEXT");
	table->vkCopyMicromapEXT = (PFN_vkCopyMicromapEXT)load(context, "vkCopyMicromapEXT");
	table->vkCopyMicromapToMemoryEXT = (PFN_vkCopyMicromapToMemoryEXT)load(context, "vkCopyMicromapToMemoryEXT");
	table->vkCreateMicromapEXT = (PFN_vkCreateMicromapEXT)load(context, "vkCreateMicromapEXT");
	table->vkDestroyMicromapEXT = (PFN_vkDestroyMicromapEXT)load(context, "vkDestroyMicromapEXT");
	table->vkGetDeviceMicromapCompatibilityEXT = (PFN_vkGetDeviceMicromapCompatibilityEXT)load(context, "vkGetDeviceMicromapCompatibilityEXT");
	table->vkGetMicromapBuildSizesEXT = (PFN_vkGetMicromapBuildSizesEXT)load(context, "vkGetMicromapBuildSizesEXT");
	table->vkWriteMicromapsPropertiesEXT = (PFN_vkWriteMicromapsPropertiesEXT)load(context, "vkWriteMicromapsPropertiesEXT");
#endif /* defined(VK_EXT_opacity_micromap) */
#if defined(VK_EXT_pageable_device_local_memory)
	table->vkSetDeviceMemoryPriorityEXT = (PFN_vkSetDeviceMemoryPriorityEXT)load(context, "vkSetDeviceMemoryPriorityEXT");
#endif /* defined(VK_EXT_pageable_device_local_memory) */
#if defined(VK_EXT_pipeline_properties)
	table->vkGetPipelinePropertiesEXT = (PFN_vkGetPipelinePropertiesEXT)load(context, "vkGetPipelinePropertiesEXT");
#endif /* defined(VK_EXT_pipeline_properties) */
#if defined(VK_EXT_private_data)
	table->vkCreatePrivateDataSlotEXT = (PFN_vkCreatePrivateDataSlotEXT)load(context, "vkCreatePrivateDataSlotEXT");
	table->vkDestroyPrivateDataSlotEXT = (PFN_vkDestroyPrivateDataSlotEXT)load(context, "vkDestroyPrivateDataSlotEXT");
	table->vkGetPrivateDataEXT = (PFN_vkGetPrivateDataEXT)load(context, "vkGetPrivateDataEXT");
	table->vkSetPrivateDataEXT = (PFN_vkSetPrivateDataEXT)load(context, "vkSetPrivateDataEXT");
#endif /* defined(VK_EXT_private_data) */
#if defined(VK_EXT_sample_locations)
	table->vkCmdSetSampleLocationsEXT = (PFN_vkCmdSetSampleLocationsEXT)load(context, "vkCmdSetSampleLocationsEXT");
#endif /* defined(VK_EXT_sample_locations) */
#if defined(VK_EXT_shader_module_identifier)
	table->vkGetShaderModuleCreateInfoIdentifierEXT = (PFN_vkGetShaderModuleCreateInfoIdentifierEXT)load(context, "vkGetShaderModuleCreateInfoIdentifierEXT");
	table->vkGetShaderModuleIdentifierEXT = (PFN_vkGetShaderModuleIdentifierEXT)load(context, "vkGetShaderModuleIdentifierEXT");
#endif /* defined(VK_EXT_shader_module_identifier) */
#if defined(VK_EXT_shader_object)
	table->vkCmdBindShadersEXT = (PFN_vkCmdBindShadersEXT)load(context, "vkCmdBindShadersEXT");
	table->vkCreateShadersEXT = (PFN_vkCreateShadersEXT)load(context, "vkCreateShadersEXT");
	table->vkDestroyShaderEXT = (PFN_vkDestroyShaderEXT)load(context, "vkDestroyShaderEXT");
	table->vkGetShaderBinaryDataEXT = (PFN_vkGetShaderBinaryDataEXT)load(context, "vkGetShaderBinaryDataEXT");
#endif /* defined(VK_EXT_shader_object) */
#if defined(VK_EXT_swapchain_maintenance1)
	table->vkReleaseSwapchainImagesEXT = (PFN_vkReleaseSwapchainImagesEXT)load(context, "vkReleaseSwapchainImagesEXT");
#endif /* defined(VK_EXT_swapchain_maintenance1) */
#if defined(VK_EXT_transform_feedback)
	table->vkCmdBeginQueryIndexedEXT = (PFN_vkCmdBeginQueryIndexedEXT)load(context, "vkCmdBeginQueryIndexedEXT");
	table->vkCmdBeginTransformFeedbackEXT = (PFN_vkCmdBeginTransformFeedbackEXT)load(context, "vkCmdBeginTransformFeedbackEXT");
	table->vkCmdBindTransformFeedbackBuffersEXT = (PFN_vkCmdBindTransformFeedbackBuffersEXT)load(context, "vkCmdBindTransformFeedbackBuffersEXT");
	table->vkCmdDrawIndirectByteCountEXT = (PFN_vkCmdDrawIndirectByteCountEXT)load(context, "vkCmdDrawIndirectByteCountEXT");
	table->vkCmdEndQueryIndexedEXT = (PFN_vkCmdEndQueryIndexedEXT)load(context, "vkCmdEndQueryIndexedEXT");
	table->vkCmdEndTransformFeedbackEXT = (PFN_vkCmdEndTransformFeedbackEXT)load(context, "vkCmdEndTransformFeedbackEXT");
#endif /* defined(VK_EXT_transform_feedback) */
#if defined(VK_EXT_validation_cache)
	table->vkCreateValidationCacheEXT = (PFN_vkCreateValidationCacheEXT)load(context, "vkCreateValidationCacheEXT");
	table->vkDestroyValidationCacheEXT = (PFN_vkDestroyValidationCacheEXT)load(context, "vkDestroyValidationCacheEXT");
	table->vkGetValidationCacheDataEXT = (PFN_vkGetValidationCacheDataEXT)load(context, "vkGetValidationCacheDataEXT");
	table->vkMergeValidationCachesEXT = (PFN_vkMergeValidationCachesEXT)load(context, "vkMergeValidationCachesEXT");
#endif /* defined(VK_EXT_validation_cache) */
#if defined(VK_FUCHSIA_buffer_collection)
	table->vkCreateBufferCollectionFUCHSIA = (PFN_vkCreateBufferCollectionFUCHSIA)load(context, "vkCreateBufferCollectionFUCHSIA");
	table->vkDestroyBufferCollectionFUCHSIA = (PFN_vkDestroyBufferCollectionFUCHSIA)load(context, "vkDestroyBufferCollectionFUCHSIA");
	table->vkGetBufferCollectionPropertiesFUCHSIA = (PFN_vkGetBufferCollectionPropertiesFUCHSIA)load(context, "vkGetBufferCollectionPropertiesFUCHSIA");
	table->vkSetBufferCollectionBufferConstraintsFUCHSIA = (PFN_vkSetBufferCollectionBufferConstraintsFUCHSIA)load(context, "vkSetBufferCollectionBufferConstraintsFUCHSIA");
	table->vkSetBufferCollectionImageConstraintsFUCHSIA = (PFN_vkSetBufferCollectionImageConstraintsFUCHSIA)load(context, "vkSetBufferCollectionImageConstraintsFUCHSIA");
#endif /* defined(VK_FUCHSIA_buffer_collection) */
#if defined(VK_FUCHSIA_external_memory)
	table->vkGetMemoryZirconHandleFUCHSIA = (PFN_vkGetMemoryZirconHandleFUCHSIA)load(context, "vkGetMemoryZirconHandleFUCHSIA");
	table->vkGetMemoryZirconHandlePropertiesFUCHSIA = (PFN_vkGetMemoryZirconHandlePropertiesFUCHSIA)load(context, "vkGetMemoryZirconHandlePropertiesFUCHSIA");
#endif /* defined(VK_FUCHSIA_external_memory) */
#if defined(VK_FUCHSIA_external_semaphore)
	table->vkGetSemaphoreZirconHandleFUCHSIA = (PFN_vkGetSemaphoreZirconHandleFUCHSIA)load(context, "vkGetSemaphoreZirconHandleFUCHSIA");
	table->vkImportSemaphoreZirconHandleFUCHSIA = (PFN_vkImportSemaphoreZirconHandleFUCHSIA)load(context, "vkImportSemaphoreZirconHandleFUCHSIA");
#endif /* defined(VK_FUCHSIA_external_semaphore) */
#if defined(VK_GOOGLE_display_timing)
	table->vkGetPastPresentationTimingGOOGLE = (PFN_vkGetPastPresentationTimingGOOGLE)load(context, "vkGetPastPresentationTimingGOOGLE");
	table->vkGetRefreshCycleDurationGOOGLE = (PFN_vkGetRefreshCycleDurationGOOGLE)load(context, "vkGetRefreshCycleDurationGOOGLE");
#endif /* defined(VK_GOOGLE_display_timing) */
#if defined(VK_HUAWEI_cluster_culling_shader)
	table->vkCmdDrawClusterHUAWEI = (PFN_vkCmdDrawClusterHUAWEI)load(context, "vkCmdDrawClusterHUAWEI");
	table->vkCmdDrawClusterIndirectHUAWEI = (PFN_vkCmdDrawClusterIndirectHUAWEI)load(context, "vkCmdDrawClusterIndirectHUAWEI");
#endif /* defined(VK_HUAWEI_cluster_culling_shader) */
#if defined(VK_HUAWEI_invocation_mask)
	table->vkCmdBindInvocationMaskHUAWEI = (PFN_vkCmdBindInvocationMaskHUAWEI)load(context, "vkCmdBindInvocationMaskHUAWEI");
#endif /* defined(VK_HUAWEI_invocation_mask) */
#if defined(VK_HUAWEI_subpass_shading)
	table->vkCmdSubpassShadingHUAWEI = (PFN_vkCmdSubpassShadingHUAWEI)load(context, "vkCmdSubpassShadingHUAWEI");
	table->vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI = (PFN_vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI)load(context, "vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI");
#endif /* defined(VK_HUAWEI_subpass_shading) */
#if defined(VK_INTEL_performance_query)
	table->vkAcquirePerformanceConfigurationINTEL = (PFN_vkAcquirePerformanceConfigurationINTEL)load(context, "vkAcquirePerformanceConfigurationINTEL");
	table->vkCmdSetPerformanceMarkerINTEL = (PFN_vkCmdSetPerformanceMarkerINTEL)load(context, "vkCmdSetPerformanceMarkerINTEL");
	table->vkCmdSetPerformanceOverrideINTEL = (PFN_vkCmdSetPerformanceOverrideINTEL)load(context, "vkCmdSetPerformanceOverrideINTEL");
	table->vkCmdSetPerformanceStreamMarkerINTEL = (PFN_vkCmdSetPerformanceStreamMarkerINTEL)load(context, "vkCmdSetPerformanceStreamMarkerINTEL");
	table->vkGetPerformanceParameterINTEL = (PFN_vkGetPerformanceParameterINTEL)load(context, "vkGetPerformanceParameterINTEL");
	table->vkInitializePerformanceApiINTEL = (PFN_vkInitializePerformanceApiINTEL)load(context, "vkInitializePerformanceApiINTEL");
	table->vkQueueSetPerformanceConfigurationINTEL = (PFN_vkQueueSetPerformanceConfigurationINTEL)load(context, "vkQueueSetPerformanceConfigurationINTEL");
	table->vkReleasePerformanceConfigurationINTEL = (PFN_vkReleasePerformanceConfigurationINTEL)load(context, "vkReleasePerformanceConfigurationINTEL");
	table->vkUninitializePerformanceApiINTEL = (PFN_vkUninitializePerformanceApiINTEL)load(context, "vkUninitializePerformanceApiINTEL");
#endif /* defined(VK_INTEL_performance_query) */
#if defined(VK_KHR_acceleration_structure)
	table->vkBuildAccelerationStructuresKHR = (PFN_vkBuildAccelerationStructuresKHR)load(context, "vkBuildAccelerationStructuresKHR");
	table->vkCmdBuildAccelerationStructuresIndirectKHR = (PFN_vkCmdBuildAccelerationStructuresIndirectKHR)load(context, "vkCmdBuildAccelerationStructuresIndirectKHR");
	table->vkCmdBuildAccelerationStructuresKHR = (PFN_vkCmdBuildAccelerationStructuresKHR)load(context, "vkCmdBuildAccelerationStructuresKHR");
	table->vkCmdCopyAccelerationStructureKHR = (PFN_vkCmdCopyAccelerationStructureKHR)load(context, "vkCmdCopyAccelerationStructureKHR");
	table->vkCmdCopyAccelerationStructureToMemoryKHR = (PFN_vkCmdCopyAccelerationStructureToMemoryKHR)load(context, "vkCmdCopyAccelerationStructureToMemoryKHR");
	table->vkCmdCopyMemoryToAccelerationStructureKHR = (PFN_vkCmdCopyMemoryToAccelerationStructureKHR)load(context, "vkCmdCopyMemoryToAccelerationStructureKHR");
	table->vkCmdWriteAccelerationStructuresPropertiesKHR = (PFN_vkCmdWriteAccelerationStructuresPropertiesKHR)load(context, "vkCmdWriteAccelerationStructuresPropertiesKHR");
	table->vkCopyAccelerationStructureKHR = (PFN_vkCopyAccelerationStructureKHR)load(context, "vkCopyAccelerationStructureKHR");
	table->vkCopyAccelerationStructureToMemoryKHR = (PFN_vkCopyAccelerationStructureToMemoryKHR)load(context, "vkCopyAccelerationStructureToMemoryKHR");
	table->vkCopyMemoryToAccelerationStructureKHR = (PFN_vkCopyMemoryToAccelerationStructureKHR)load(context, "vkCopyMemoryToAccelerationStructureKHR");
	table->vkCreateAccelerationStructureKHR = (PFN_vkCreateAccelerationStructureKHR)load(context, "vkCreateAccelerationStructureKHR");
	table->vkDestroyAccelerationStructureKHR = (PFN_vkDestroyAccelerationStructureKHR)load(context, "vkDestroyAccelerationStructureKHR");
	table->vkGetAccelerationStructureBuildSizesKHR = (PFN_vkGetAccelerationStructureBuildSizesKHR)load(context, "vkGetAccelerationStructureBuildSizesKHR");
	table->vkGetAccelerationStructureDeviceAddressKHR = (PFN_vkGetAccelerationStructureDeviceAddressKHR)load(context, "vkGetAccelerationStructureDeviceAddressKHR");
	table->vkGetDeviceAccelerationStructureCompatibilityKHR = (PFN_vkGetDeviceAccelerationStructureCompatibilityKHR)load(context, "vkGetDeviceAccelerationStructureCompatibilityKHR");
	table->vkWriteAccelerationStructuresPropertiesKHR = (PFN_vkWriteAccelerationStructuresPropertiesKHR)load(context, "vkWriteAccelerationStructuresPropertiesKHR");
#endif /* defined(VK_KHR_acceleration_structure) */
#if defined(VK_KHR_bind_memory2)
	table->vkBindBufferMemory2KHR = (PFN_vkBindBufferMemory2KHR)load(context, "vkBindBufferMemory2KHR");
	table->vkBindImageMemory2KHR = (PFN_vkBindImageMemory2KHR)load(context, "vkBindImageMemory2KHR");
#endif /* defined(VK_KHR_bind_memory2) */
#if defined(VK_KHR_buffer_device_address)
	table->vkGetBufferDeviceAddressKHR = (PFN_vkGetBufferDeviceAddressKHR)load(context, "vkGetBufferDeviceAddressKHR");
	table->vkGetBufferOpaqueCaptureAddressKHR = (PFN_vkGetBufferOpaqueCaptureAddressKHR)load(context, "vkGetBufferOpaqueCaptureAddressKHR");
	table->vkGetDeviceMemoryOpaqueCaptureAddressKHR = (PFN_vkGetDeviceMemoryOpaqueCaptureAddressKHR)load(context, "vkGetDeviceMemoryOpaqueCaptureAddressKHR");
#endif /* defined(VK_KHR_buffer_device_address) */
#if defined(VK_KHR_calibrated_timestamps)
	table->vkGetCalibratedTimestampsKHR = (PFN_vkGetCalibratedTimestampsKHR)load(context, "vkGetCalibratedTimestampsKHR");
#endif /* defined(VK_KHR_calibrated_timestamps) */
#if defined(VK_KHR_copy_commands2)
	table->vkCmdBlitImage2KHR = (PFN_vkCmdBlitImage2KHR)load(context, "vkCmdBlitImage2KHR");
	table->vkCmdCopyBuffer2KHR = (PFN_vkCmdCopyBuffer2KHR)load(context, "vkCmdCopyBuffer2KHR");
	table->vkCmdCopyBufferToImage2KHR = (PFN_vkCmdCopyBufferToImage2KHR)load(context, "vkCmdCopyBufferToImage2KHR");
	table->vkCmdCopyImage2KHR = (PFN_vkCmdCopyImage2KHR)load(context, "vkCmdCopyImage2KHR");
	table->vkCmdCopyImageToBuffer2KHR = (PFN_vkCmdCopyImageToBuffer2KHR)load(context, "vkCmdCopyImageToBuffer2KHR");
	table->vkCmdResolveImage2KHR = (PFN_vkCmdResolveImage2KHR)load(context, "vkCmdResolveImage2KHR");
#endif /* defined(VK_KHR_copy_commands2) */
#if defined(VK_KHR_create_renderpass2)
	table->vkCmdBeginRenderPass2KHR = (PFN_vkCmdBeginRenderPass2KHR)load(context, "vkCmdBeginRenderPass2KHR");
	table->vkCmdEndRenderPass2KHR = (PFN_vkCmdEndRenderPass2KHR)load(context, "vkCmdEndRenderPass2KHR");
	table->vkCmdNextSubpass2KHR = (PFN_vkCmdNextSubpass2KHR)load(context, "vkCmdNextSubpass2KHR");
	table->vkCreateRenderPass2KHR = (PFN_vkCreateRenderPass2KHR)load(context, "vkCreateRenderPass2KHR");
#endif /* defined(VK_KHR_create_renderpass2) */
#if defined(VK_KHR_deferred_host_operations)
	table->vkCreateDeferredOperationKHR = (PFN_vkCreateDeferredOperationKHR)load(context, "vkCreateDeferredOperationKHR");
	table->vkDeferredOperationJoinKHR = (PFN_vkDeferredOperationJoinKHR)load(context, "vkDeferredOperationJoinKHR");
	table->vkDestroyDeferredOperationKHR = (PFN_vkDestroyDeferredOperationKHR)load(context, "vkDestroyDeferredOperationKHR");
	table->vkGetDeferredOperationMaxConcurrencyKHR = (PFN_vkGetDeferredOperationMaxConcurrencyKHR)load(context, "vkGetDeferredOperationMaxConcurrencyKHR");
	table->vkGetDeferredOperationResultKHR = (PFN_vkGetDeferredOperationResultKHR)load(context, "vkGetDeferredOperationResultKHR");
#endif /* defined(VK_KHR_deferred_host_operations) */
#if defined(VK_KHR_descriptor_update_template)
	table->vkCreateDescriptorUpdateTemplateKHR = (PFN_vkCreateDescriptorUpdateTemplateKHR)load(context, "vkCreateDescriptorUpdateTemplateKHR");
	table->vkDestroyDescriptorUpdateTemplateKHR = (PFN_vkDestroyDescriptorUpdateTemplateKHR)load(context, "vkDestroyDescriptorUpdateTemplateKHR");
	table->vkUpdateDescriptorSetWithTemplateKHR = (PFN_vkUpdateDescriptorSetWithTemplateKHR)load(context, "vkUpdateDescriptorSetWithTemplateKHR");
#endif /* defined(VK_KHR_descriptor_update_template) */
#if defined(VK_KHR_device_group)
	table->vkCmdDispatchBaseKHR = (PFN_vkCmdDispatchBaseKHR)load(context, "vkCmdDispatchBaseKHR");
	table->vkCmdSetDeviceMaskKHR = (PFN_vkCmdSetDeviceMaskKHR)load(context, "vkCmdSetDeviceMaskKHR");
	table->vkGetDeviceGroupPeerMemoryFeaturesKHR = (PFN_vkGetDeviceGroupPeerMemoryFeaturesKHR)load(context, "vkGetDeviceGroupPeerMemoryFeaturesKHR");
#endif /* defined(VK_KHR_device_group) */
#if defined(VK_KHR_display_swapchain)
	table->vkCreateSharedSwapchainsKHR = (PFN_vkCreateSharedSwapchainsKHR)load(context, "vkCreateSharedSwapchainsKHR");
#endif /* defined(VK_KHR_display_swapchain) */
#if defined(VK_KHR_draw_indirect_count)
	table->vkCmdDrawIndexedIndirectCountKHR = (PFN_vkCmdDrawIndexedIndirectCountKHR)load(context, "vkCmdDrawIndexedIndirectCountKHR");
	table->vkCmdDrawIndirectCountKHR = (PFN_vkCmdDrawIndirectCountKHR)load(context, "vkCmdDrawIndirectCountKHR");
#endif /* defined(VK_KHR_draw_indirect_count) */
#if defined(VK_KHR_dynamic_rendering)
	table->vkCmdBeginRenderingKHR = (PFN_vkCmdBeginRenderingKHR)load(context, "vkCmdBeginRenderingKHR");
	table->vkCmdEndRenderingKHR = (PFN_vkCmdEndRenderingKHR)load(context, "vkCmdEndRenderingKHR");
#endif /* defined(VK_KHR_dynamic_rendering) */
#if defined(VK_KHR_external_fence_fd)
	table->vkGetFenceFdKHR = (PFN_vkGetFenceFdKHR)load(context, "vkGetFenceFdKHR");
	table->vkImportFenceFdKHR = (PFN_vkImportFenceFdKHR)load(context, "vkImportFenceFdKHR");
#endif /* defined(VK_KHR_external_fence_fd) */
#if defined(VK_KHR_external_fence_win32)
	table->vkGetFenceWin32HandleKHR = (PFN_vkGetFenceWin32HandleKHR)load(context, "vkGetFenceWin32HandleKHR");
	table->vkImportFenceWin32HandleKHR = (PFN_vkImportFenceWin32HandleKHR)load(context, "vkImportFenceWin32HandleKHR");
#endif /* defined(VK_KHR_external_fence_win32) */
#if defined(VK_KHR_external_memory_fd)
	table->vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)load(context, "vkGetMemoryFdKHR");
	table->vkGetMemoryFdPropertiesKHR = (PFN_vkGetMemoryFdPropertiesKHR)load(context, "vkGetMemoryFdPropertiesKHR");
#endif /* defined(VK_KHR_external_memory_fd) */
#if defined(VK_KHR_external_memory_win32)
	table->vkGetMemoryWin32HandleKHR = (PFN_vkGetMemoryWin32HandleKHR)load(context, "vkGetMemoryWin32HandleKHR");
	table->vkGetMemoryWin32HandlePropertiesKHR = (PFN_vkGetMemoryWin32HandlePropertiesKHR)load(context, "vkGetMemoryWin32HandlePropertiesKHR");
#endif /* defined(VK_KHR_external_memory_win32) */
#if defined(VK_KHR_external_semaphore_fd)
	table->vkGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)load(context, "vkGetSemaphoreFdKHR");
	table->vkImportSemaphoreFdKHR = (PFN_vkImportSemaphoreFdKHR)load(context, "vkImportSemaphoreFdKHR");
#endif /* defined(VK_KHR_external_semaphore_fd) */
#if defined(VK_KHR_external_semaphore_win32)
	table->vkGetSemaphoreWin32HandleKHR = (PFN_vkGetSemaphoreWin32HandleKHR)load(context, "vkGetSemaphoreWin32HandleKHR");
	table->vkImportSemaphoreWin32HandleKHR = (PFN_vkImportSemaphoreWin32HandleKHR)load(context, "vkImportSemaphoreWin32HandleKHR");
#endif /* defined(VK_KHR_external_semaphore_win32) */
#if defined(VK_KHR_fragment_shading_rate)
	table->vkCmdSetFragmentShadingRateKHR = (PFN_vkCmdSetFragmentShadingRateKHR)load(context, "vkCmdSetFragmentShadingRateKHR");
#endif /* defined(VK_KHR_fragment_shading_rate) */
#if defined(VK_KHR_get_memory_requirements2)
	table->vkGetBufferMemoryRequirements2KHR = (PFN_vkGetBufferMemoryRequirements2KHR)load(context, "vkGetBufferMemoryRequirements2KHR");
	table->vkGetImageMemoryRequirements2KHR = (PFN_vkGetImageMemoryRequirements2KHR)load(context, "vkGetImageMemoryRequirements2KHR");
	table->vkGetImageSparseMemoryRequirements2KHR = (PFN_vkGetImageSparseMemoryRequirements2KHR)load(context, "vkGetImageSparseMemoryRequirements2KHR");
#endif /* defined(VK_KHR_get_memory_requirements2) */
#if defined(VK_KHR_maintenance1)
	table->vkTrimCommandPoolKHR = (PFN_vkTrimCommandPoolKHR)load(context, "vkTrimCommandPoolKHR");
#endif /* defined(VK_KHR_maintenance1) */
#if defined(VK_KHR_maintenance3)
	table->vkGetDescriptorSetLayoutSupportKHR = (PFN_vkGetDescriptorSetLayoutSupportKHR)load(context, "vkGetDescriptorSetLayoutSupportKHR");
#endif /* defined(VK_KHR_maintenance3) */
#if defined(VK_KHR_maintenance4)
	table->vkGetDeviceBufferMemoryRequirementsKHR = (PFN_vkGetDeviceBufferMemoryRequirementsKHR)load(context, "vkGetDeviceBufferMemoryRequirementsKHR");
	table->vkGetDeviceImageMemoryRequirementsKHR = (PFN_vkGetDeviceImageMemoryRequirementsKHR)load(context, "vkGetDeviceImageMemoryRequirementsKHR");
	table->vkGetDeviceImageSparseMemoryRequirementsKHR = (PFN_vkGetDeviceImageSparseMemoryRequirementsKHR)load(context, "vkGetDeviceImageSparseMemoryRequirementsKHR");
#endif /* defined(VK_KHR_maintenance4) */
#if defined(VK_KHR_maintenance5)
	table->vkCmdBindIndexBuffer2KHR = (PFN_vkCmdBindIndexBuffer2KHR)load(context, "vkCmdBindIndexBuffer2KHR");
	table->vkGetDeviceImageSubresourceLayoutKHR = (PFN_vkGetDeviceImageSubresourceLayoutKHR)load(context, "vkGetDeviceImageSubresourceLayoutKHR");
	table->vkGetImageSubresourceLayout2KHR = (PFN_vkGetImageSubresourceLayout2KHR)load(context, "vkGetImageSubresourceLayout2KHR");
	table->vkGetRenderingAreaGranularityKHR = (PFN_vkGetRenderingAreaGranularityKHR)load(context, "vkGetRenderingAreaGranularityKHR");
#endif /* defined(VK_KHR_maintenance5) */
#if defined(VK_KHR_maintenance6)
	table->vkCmdBindDescriptorSets2KHR = (PFN_vkCmdBindDescriptorSets2KHR)load(context, "vkCmdBindDescriptorSets2KHR");
	table->vkCmdPushConstants2KHR = (PFN_vkCmdPushConstants2KHR)load(context, "vkCmdPushConstants2KHR");
#endif /* defined(VK_KHR_maintenance6) */
#if defined(VK_KHR_maintenance6) && defined(VK_KHR_push_descriptor)
	table->vkCmdPushDescriptorSet2KHR = (PFN_vkCmdPushDescriptorSet2KHR)load(context, "vkCmdPushDescriptorSet2KHR");
	table->vkCmdPushDescriptorSetWithTemplate2KHR = (PFN_vkCmdPushDescriptorSetWithTemplate2KHR)load(context, "vkCmdPushDescriptorSetWithTemplate2KHR");
#endif /* defined(VK_KHR_maintenance6) && defined(VK_KHR_push_descriptor) */
#if defined(VK_KHR_maintenance6) && defined(VK_EXT_descriptor_buffer)
	table->vkCmdBindDescriptorBufferEmbeddedSamplers2EXT = (PFN_vkCmdBindDescriptorBufferEmbeddedSamplers2EXT)load(context, "vkCmdBindDescriptorBufferEmbeddedSamplers2EXT");
	table->vkCmdSetDescriptorBufferOffsets2EXT = (PFN_vkCmdSetDescriptorBufferOffsets2EXT)load(context, "vkCmdSetDescriptorBufferOffsets2EXT");
#endif /* defined(VK_KHR_maintenance6) && defined(VK_EXT_descriptor_buffer) */
#if defined(VK_KHR_map_memory2)
	table->vkMapMemory2KHR = (PFN_vkMapMemory2KHR)load(context, "vkMapMemory2KHR");
	table->vkUnmapMemory2KHR = (PFN_vkUnmapMemory2KHR)load(context, "vkUnmapMemory2KHR");
#endif /* defined(VK_KHR_map_memory2) */
#if defined(VK_KHR_performance_query)
	table->vkAcquireProfilingLockKHR = (PFN_vkAcquireProfilingLockKHR)load(context, "vkAcquireProfilingLockKHR");
	table->vkReleaseProfilingLockKHR = (PFN_vkReleaseProfilingLockKHR)load(context, "vkReleaseProfilingLockKHR");
#endif /* defined(VK_KHR_performance_query) */
#if defined(VK_KHR_pipeline_executable_properties)
	table->vkGetPipelineExecutableInternalRepresentationsKHR = (PFN_vkGetPipelineExecutableInternalRepresentationsKHR)load(context, "vkGetPipelineExecutableInternalRepresentationsKHR");
	table->vkGetPipelineExecutablePropertiesKHR = (PFN_vkGetPipelineExecutablePropertiesKHR)load(context, "vkGetPipelineExecutablePropertiesKHR");
	table->vkGetPipelineExecutableStatisticsKHR = (PFN_vkGetPipelineExecutableStatisticsKHR)load(context, "vkGetPipelineExecutableStatisticsKHR");
#endif /* defined(VK_KHR_pipeline_executable_properties) */
#if defined(VK_KHR_present_wait)
	table->vkWaitForPresentKHR = (PFN_vkWaitForPresentKHR)load(context, "vkWaitForPresentKHR");
#endif /* defined(VK_KHR_present_wait) */
#if defined(VK_KHR_push_descriptor)
	table->vkCmdPushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)load(context, "vkCmdPushDescriptorSetKHR");
#endif /* defined(VK_KHR_push_descriptor) */
#if defined(VK_KHR_ray_tracing_maintenance1) && defined(VK_KHR_ray_tracing_pipeline)
	table->vkCmdTraceRaysIndirect2KHR = (PFN_vkCmdTraceRaysIndirect2KHR)load(context, "vkCmdTraceRaysIndirect2KHR");
#endif /* defined(VK_KHR_ray_tracing_maintenance1) && defined(VK_KHR_ray_tracing_pipeline) */
#if defined(VK_KHR_ray_tracing_pipeline)
	table->vkCmdSetRayTracingPipelineStackSizeKHR = (PFN_vkCmdSetRayTracingPipelineStackSizeKHR)load(context, "vkCmdSetRayTracingPipelineStackSizeKHR");
	table->vkCmdTraceRaysIndirectKHR = (PFN_vkCmdTraceRaysIndirectKHR)load(context, "vkCmdTraceRaysIndirectKHR");
	table->vkCmdTraceRaysKHR = (PFN_vkCmdTraceRaysKHR)load(context, "vkCmdTraceRaysKHR");
	table->vkCreateRayTracingPipelinesKHR = (PFN_vkCreateRayTracingPipelinesKHR)load(context, "vkCreateRayTracingPipelinesKHR");
	table->vkGetRayTracingCaptureReplayShaderGroupHandlesKHR = (PFN_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR)load(context, "vkGetRayTracingCaptureReplayShaderGroupHandlesKHR");
	table->vkGetRayTracingShaderGroupHandlesKHR = (PFN_vkGetRayTracingShaderGroupHandlesKHR)load(context, "vkGetRayTracingShaderGroupHandlesKHR");
	table->vkGetRayTracingShaderGroupStackSizeKHR = (PFN_vkGetRayTracingShaderGroupStackSizeKHR)load(context, "vkGetRayTracingShaderGroupStackSizeKHR");
#endif /* defined(VK_KHR_ray_tracing_pipeline) */
#if defined(VK_KHR_sampler_ycbcr_conversion)
	table->vkCreateSamplerYcbcrConversionKHR = (PFN_vkCreateSamplerYcbcrConversionKHR)load(context, "vkCreateSamplerYcbcrConversionKHR");
	table->vkDestroySamplerYcbcrConversionKHR = (PFN_vkDestroySamplerYcbcrConversionKHR)load(context, "vkDestroySamplerYcbcrConversionKHR");
#endif /* defined(VK_KHR_sampler_ycbcr_conversion) */
#if defined(VK_KHR_shared_presentable_image)
	table->vkGetSwapchainStatusKHR = (PFN_vkGetSwapchainStatusKHR)load(context, "vkGetSwapchainStatusKHR");
#endif /* defined(VK_KHR_shared_presentable_image) */
#if defined(VK_KHR_swapchain)
	table->vkAcquireNextImageKHR = (PFN_vkAcquireNextImageKHR)load(context, "vkAcquireNextImageKHR");
	table->vkCreateSwapchainKHR = (PFN_vkCreateSwapchainKHR)load(context, "vkCreateSwapchainKHR");
	table->vkDestroySwapchainKHR = (PFN_vkDestroySwapchainKHR)load(context, "vkDestroySwapchainKHR");
	table->vkGetSwapchainImagesKHR = (PFN_vkGetSwapchainImagesKHR)load(context, "vkGetSwapchainImagesKHR");
	table->vkQueuePresentKHR = (PFN_vkQueuePresentKHR)load(context, "vkQueuePresentKHR");
#endif /* defined(VK_KHR_swapchain) */
#if defined(VK_KHR_synchronization2)
	table->vkCmdPipelineBarrier2KHR = (PFN_vkCmdPipelineBarrier2KHR)load(context, "vkCmdPipelineBarrier2KHR");
	table->vkCmdResetEvent2KHR = (PFN_vkCmdResetEvent2KHR)load(context, "vkCmdResetEvent2KHR");
	table->vkCmdSetEvent2KHR = (PFN_vkCmdSetEvent2KHR)load(context, "vkCmdSetEvent2KHR");
	table->vkCmdWaitEvents2KHR = (PFN_vkCmdWaitEvents2KHR)load(context, "vkCmdWaitEvents2KHR");
	table->vkCmdWriteTimestamp2KHR = (PFN_vkCmdWriteTimestamp2KHR)load(context, "vkCmdWriteTimestamp2KHR");
	table->vkQueueSubmit2KHR = (PFN_vkQueueSubmit2KHR)load(context, "vkQueueSubmit2KHR");
#endif /* defined(VK_KHR_synchronization2) */
#if defined(VK_KHR_synchronization2) && defined(VK_AMD_buffer_marker)
	table->vkCmdWriteBufferMarker2AMD = (PFN_vkCmdWriteBufferMarker2AMD)load(context, "vkCmdWriteBufferMarker2AMD");
#endif /* defined(VK_KHR_synchronization2) && defined(VK_AMD_buffer_marker) */
#if defined(VK_KHR_synchronization2) && defined(VK_NV_device_diagnostic_checkpoints)
	table->vkGetQueueCheckpointData2NV = (PFN_vkGetQueueCheckpointData2NV)load(context, "vkGetQueueCheckpointData2NV");
#endif /* defined(VK_KHR_synchronization2) && defined(VK_NV_device_diagnostic_checkpoints) */
#if defined(VK_KHR_timeline_semaphore)
	table->vkGetSemaphoreCounterValueKHR = (PFN_vkGetSemaphoreCounterValueKHR)load(context, "vkGetSemaphoreCounterValueKHR");
	table->vkSignalSemaphoreKHR = (PFN_vkSignalSemaphoreKHR)load(context, "vkSignalSemaphoreKHR");
	table->vkWaitSemaphoresKHR = (PFN_vkWaitSemaphoresKHR)load(context, "vkWaitSemaphoresKHR");
#endif /* defined(VK_KHR_timeline_semaphore) */
#if defined(VK_KHR_video_decode_queue)
	table->vkCmdDecodeVideoKHR = (PFN_vkCmdDecodeVideoKHR)load(context, "vkCmdDecodeVideoKHR");
#endif /* defined(VK_KHR_video_decode_queue) */
#if defined(VK_KHR_video_encode_queue)
	table->vkCmdEncodeVideoKHR = (PFN_vkCmdEncodeVideoKHR)load(context, "vkCmdEncodeVideoKHR");
	table->vkGetEncodedVideoSessionParametersKHR = (PFN_vkGetEncodedVideoSessionParametersKHR)load(context, "vkGetEncodedVideoSessionParametersKHR");
#endif /* defined(VK_KHR_video_encode_queue) */
#if defined(VK_KHR_video_queue)
	table->vkBindVideoSessionMemoryKHR = (PFN_vkBindVideoSessionMemoryKHR)load(context, "vkBindVideoSessionMemoryKHR");
	table->vkCmdBeginVideoCodingKHR = (PFN_vkCmdBeginVideoCodingKHR)load(context, "vkCmdBeginVideoCodingKHR");
	table->vkCmdControlVideoCodingKHR = (PFN_vkCmdControlVideoCodingKHR)load(context, "vkCmdControlVideoCodingKHR");
	table->vkCmdEndVideoCodingKHR = (PFN_vkCmdEndVideoCodingKHR)load(context, "vkCmdEndVideoCodingKHR");
	table->vkCreateVideoSessionKHR = (PFN_vkCreateVideoSessionKHR)load(context, "vkCreateVideoSessionKHR");
	table->vkCreateVideoSessionParametersKHR = (PFN_vkCreateVideoSessionParametersKHR)load(context, "vkCreateVideoSessionParametersKHR");
	table->vkDestroyVideoSessionKHR = (PFN_vkDestroyVideoSessionKHR)load(context, "vkDestroyVideoSessionKHR");
	table->vkDestroyVideoSessionParametersKHR = (PFN_vkDestroyVideoSessionParametersKHR)load(context, "vkDestroyVideoSessionParametersKHR");
	table->vkGetVideoSessionMemoryRequirementsKHR = (PFN_vkGetVideoSessionMemoryRequirementsKHR)load(context, "vkGetVideoSessionMemoryRequirementsKHR");
	table->vkUpdateVideoSessionParametersKHR = (PFN_vkUpdateVideoSessionParametersKHR)load(context, "vkUpdateVideoSessionParametersKHR");
#endif /* defined(VK_KHR_video_queue) */
#if defined(VK_NVX_binary_import)
	table->vkCmdCuLaunchKernelNVX = (PFN_vkCmdCuLaunchKernelNVX)load(context, "vkCmdCuLaunchKernelNVX");
	table->vkCreateCuFunctionNVX = (PFN_vkCreateCuFunctionNVX)load(context, "vkCreateCuFunctionNVX");
	table->vkCreateCuModuleNVX = (PFN_vkCreateCuModuleNVX)load(context, "vkCreateCuModuleNVX");
	table->vkDestroyCuFunctionNVX = (PFN_vkDestroyCuFunctionNVX)load(context, "vkDestroyCuFunctionNVX");
	table->vkDestroyCuModuleNVX = (PFN_vkDestroyCuModuleNVX)load(context, "vkDestroyCuModuleNVX");
#endif /* defined(VK_NVX_binary_import) */
#if defined(VK_NVX_image_view_handle)
	table->vkGetImageViewAddressNVX = (PFN_vkGetImageViewAddressNVX)load(context, "vkGetImageViewAddressNVX");
	table->vkGetImageViewHandleNVX = (PFN_vkGetImageViewHandleNVX)load(context, "vkGetImageViewHandleNVX");
#endif /* defined(VK_NVX_image_view_handle) */
#if defined(VK_NV_clip_space_w_scaling)
	table->vkCmdSetViewportWScalingNV = (PFN_vkCmdSetViewportWScalingNV)load(context, "vkCmdSetViewportWScalingNV");
#endif /* defined(VK_NV_clip_space_w_scaling) */
#if defined(VK_NV_copy_memory_indirect)
	table->vkCmdCopyMemoryIndirectNV = (PFN_vkCmdCopyMemoryIndirectNV)load(context, "vkCmdCopyMemoryIndirectNV");
	table->vkCmdCopyMemoryToImageIndirectNV = (PFN_vkCmdCopyMemoryToImageIndirectNV)load(context, "vkCmdCopyMemoryToImageIndirectNV");
#endif /* defined(VK_NV_copy_memory_indirect) */
#if defined(VK_NV_cuda_kernel_launch)
	table->vkCmdCudaLaunchKernelNV = (PFN_vkCmdCudaLaunchKernelNV)load(context, "vkCmdCudaLaunchKernelNV");
	table->vkCreateCudaFunctionNV = (PFN_vkCreateCudaFunctionNV)load(context, "vkCreateCudaFunctionNV");
	table->vkCreateCudaModuleNV = (PFN_vkCreateCudaModuleNV)load(context, "vkCreateCudaModuleNV");
	table->vkDestroyCudaFunctionNV = (PFN_vkDestroyCudaFunctionNV)load(context, "vkDestroyCudaFunctionNV");
	table->vkDestroyCudaModuleNV = (PFN_vkDestroyCudaModuleNV)load(context, "vkDestroyCudaModuleNV");
	table->vkGetCudaModuleCacheNV = (PFN_vkGetCudaModuleCacheNV)load(context, "vkGetCudaModuleCacheNV");
#endif /* defined(VK_NV_cuda_kernel_launch) */
#if defined(VK_NV_device_diagnostic_checkpoints)
	table->vkCmdSetCheckpointNV = (PFN_vkCmdSetCheckpointNV)load(context, "vkCmdSetCheckpointNV");
	table->vkGetQueueCheckpointDataNV = (PFN_vkGetQueueCheckpointDataNV)load(context, "vkGetQueueCheckpointDataNV");
#endif /* defined(VK_NV_device_diagnostic_checkpoints) */
#if defined(VK_NV_device_generated_commands)
	table->vkCmdBindPipelineShaderGroupNV = (PFN_vkCmdBindPipelineShaderGroupNV)load(context, "vkCmdBindPipelineShaderGroupNV");
	table->vkCmdExecuteGeneratedCommandsNV = (PFN_vkCmdExecuteGeneratedCommandsNV)load(context, "vkCmdExecuteGeneratedCommandsNV");
	table->vkCmdPreprocessGeneratedCommandsNV = (PFN_vkCmdPreprocessGeneratedCommandsNV)load(context, "vkCmdPreprocessGeneratedCommandsNV");
	table->vkCreateIndirectCommandsLayoutNV = (PFN_vkCreateIndirectCommandsLayoutNV)load(context, "vkCreateIndirectCommandsLayoutNV");
	table->vkDestroyIndirectCommandsLayoutNV = (PFN_vkDestroyIndirectCommandsLayoutNV)load(context, "vkDestroyIndirectCommandsLayoutNV");
	table->vkGetGeneratedCommandsMemoryRequirementsNV = (PFN_vkGetGeneratedCommandsMemoryRequirementsNV)load(context, "vkGetGeneratedCommandsMemoryRequirementsNV");
#endif /* defined(VK_NV_device_generated_commands) */
#if defined(VK_NV_device_generated_commands_compute)
	table->vkCmdUpdatePipelineIndirectBufferNV = (PFN_vkCmdUpdatePipelineIndirectBufferNV)load(context, "vkCmdUpdatePipelineIndirectBufferNV");
	table->vkGetPipelineIndirectDeviceAddressNV = (PFN_vkGetPipelineIndirectDeviceAddressNV)load(context, "vkGetPipelineIndirectDeviceAddressNV");
	table->vkGetPipelineIndirectMemoryRequirementsNV = (PFN_vkGetPipelineIndirectMemoryRequirementsNV)load(context, "vkGetPipelineIndirectMemoryRequirementsNV");
#endif /* defined(VK_NV_device_generated_commands_compute) */
#if defined(VK_NV_external_memory_rdma)
	table->vkGetMemoryRemoteAddressNV = (PFN_vkGetMemoryRemoteAddressNV)load(context, "vkGetMemoryRemoteAddressNV");
#endif /* defined(VK_NV_external_memory_rdma) */
#if defined(VK_NV_external_memory_win32)
	table->vkGetMemoryWin32HandleNV = (PFN_vkGetMemoryWin32HandleNV)load(context, "vkGetMemoryWin32HandleNV");
#endif /* defined(VK_NV_external_memory_win32) */
#if defined(VK_NV_fragment_shading_rate_enums)
	table->vkCmdSetFragmentShadingRateEnumNV = (PFN_vkCmdSetFragmentShadingRateEnumNV)load(context, "vkCmdSetFragmentShadingRateEnumNV");
#endif /* defined(VK_NV_fragment_shading_rate_enums) */
#if defined(VK_NV_low_latency2)
	table->vkGetLatencyTimingsNV = (PFN_vkGetLatencyTimingsNV)load(context, "vkGetLatencyTimingsNV");
	table->vkLatencySleepNV = (PFN_vkLatencySleepNV)load(context, "vkLatencySleepNV");
	table->vkQueueNotifyOutOfBandNV = (PFN_vkQueueNotifyOutOfBandNV)load(context, "vkQueueNotifyOutOfBandNV");
	table->vkSetLatencyMarkerNV = (PFN_vkSetLatencyMarkerNV)load(context, "vkSetLatencyMarkerNV");
	table->vkSetLatencySleepModeNV = (PFN_vkSetLatencySleepModeNV)load(context, "vkSetLatencySleepModeNV");
#endif /* defined(VK_NV_low_latency2) */
#if defined(VK_NV_memory_decompression)
	table->vkCmdDecompressMemoryIndirectCountNV = (PFN_vkCmdDecompressMemoryIndirectCountNV)load(context, "vkCmdDecompressMemoryIndirectCountNV");
	table->vkCmdDecompressMemoryNV = (PFN_vkCmdDecompressMemoryNV)load(context, "vkCmdDecompressMemoryNV");
#endif /* defined(VK_NV_memory_decompression) */
#if defined(VK_NV_mesh_shader)
	table->vkCmdDrawMeshTasksIndirectCountNV = (PFN_vkCmdDrawMeshTasksIndirectCountNV)load(context, "vkCmdDrawMeshTasksIndirectCountNV");
	table->vkCmdDrawMeshTasksIndirectNV = (PFN_vkCmdDrawMeshTasksIndirectNV)load(context, "vkCmdDrawMeshTasksIndirectNV");
	table->vkCmdDrawMeshTasksNV = (PFN_vkCmdDrawMeshTasksNV)load(context, "vkCmdDrawMeshTasksNV");
#endif /* defined(VK_NV_mesh_shader) */
#if defined(VK_NV_optical_flow)
	table->vkBindOpticalFlowSessionImageNV = (PFN_vkBindOpticalFlowSessionImageNV)load(context, "vkBindOpticalFlowSessionImageNV");
	table->vkCmdOpticalFlowExecuteNV = (PFN_vkCmdOpticalFlowExecuteNV)load(context, "vkCmdOpticalFlowExecuteNV");
	table->vkCreateOpticalFlowSessionNV = (PFN_vkCreateOpticalFlowSessionNV)load(context, "vkCreateOpticalFlowSessionNV");
	table->vkDestroyOpticalFlowSessionNV = (PFN_vkDestroyOpticalFlowSessionNV)load(context, "vkDestroyOpticalFlowSessionNV");
#endif /* defined(VK_NV_optical_flow) */
#if defined(VK_NV_ray_tracing)
	table->vkBindAccelerationStructureMemoryNV = (PFN_vkBindAccelerationStructureMemoryNV)load(context, "vkBindAccelerationStructureMemoryNV");
	table->vkCmdBuildAccelerationStructureNV = (PFN_vkCmdBuildAccelerationStructureNV)load(context, "vkCmdBuildAccelerationStructureNV");
	table->vkCmdCopyAccelerationStructureNV = (PFN_vkCmdCopyAccelerationStructureNV)load(context, "vkCmdCopyAccelerationStructureNV");
	table->vkCmdTraceRaysNV = (PFN_vkCmdTraceRaysNV)load(context, "vkCmdTraceRaysNV");
	table->vkCmdWriteAccelerationStructuresPropertiesNV = (PFN_vkCmdWriteAccelerationStructuresPropertiesNV)load(context, "vkCmdWriteAccelerationStructuresPropertiesNV");
	table->vkCompileDeferredNV = (PFN_vkCompileDeferredNV)load(context, "vkCompileDeferredNV");
	table->vkCreateAccelerationStructureNV = (PFN_vkCreateAccelerationStructureNV)load(context, "vkCreateAccelerationStructureNV");
	table->vkCreateRayTracingPipelinesNV = (PFN_vkCreateRayTracingPipelinesNV)load(context, "vkCreateRayTracingPipelinesNV");
	table->vkDestroyAccelerationStructureNV = (PFN_vkDestroyAccelerationStructureNV)load(context, "vkDestroyAccelerationStructureNV");
	table->vkGetAccelerationStructureHandleNV = (PFN_vkGetAccelerationStructureHandleNV)load(context, "vkGetAccelerationStructureHandleNV");
	table->vkGetAccelerationStructureMemoryRequirementsNV = (PFN_vkGetAccelerationStructureMemoryRequirementsNV)load(context, "vkGetAccelerationStructureMemoryRequirementsNV");
	table->vkGetRayTracingShaderGroupHandlesNV = (PFN_vkGetRayTracingShaderGroupHandlesNV)load(context, "vkGetRayTracingShaderGroupHandlesNV");
#endif /* defined(VK_NV_ray_tracing) */
#if defined(VK_NV_scissor_exclusive) && VK_NV_SCISSOR_EXCLUSIVE_SPEC_VERSION >= 2
	table->vkCmdSetExclusiveScissorEnableNV = (PFN_vkCmdSetExclusiveScissorEnableNV)load(context, "vkCmdSetExclusiveScissorEnableNV");
#endif /* defined(VK_NV_scissor_exclusive) && VK_NV_SCISSOR_EXCLUSIVE_SPEC_VERSION >= 2 */
#if defined(VK_NV_scissor_exclusive)
	table->vkCmdSetExclusiveScissorNV = (PFN_vkCmdSetExclusiveScissorNV)load(context, "vkCmdSetExclusiveScissorNV");
#endif /* defined(VK_NV_scissor_exclusive) */
#if defined(VK_NV_shading_rate_image)
	table->vkCmdBindShadingRateImageNV = (PFN_vkCmdBindShadingRateImageNV)load(context, "vkCmdBindShadingRateImageNV");
	table->vkCmdSetCoarseSampleOrderNV = (PFN_vkCmdSetCoarseSampleOrderNV)load(context, "vkCmdSetCoarseSampleOrderNV");
	table->vkCmdSetViewportShadingRatePaletteNV = (PFN_vkCmdSetViewportShadingRatePaletteNV)load(context, "vkCmdSetViewportShadingRatePaletteNV");
#endif /* defined(VK_NV_shading_rate_image) */
#if defined(VK_QCOM_tile_properties)
	table->vkGetDynamicRenderingTilePropertiesQCOM = (PFN_vkGetDynamicRenderingTilePropertiesQCOM)load(context, "vkGetDynamicRenderingTilePropertiesQCOM");
	table->vkGetFramebufferTilePropertiesQCOM = (PFN_vkGetFramebufferTilePropertiesQCOM)load(context, "vkGetFramebufferTilePropertiesQCOM");
#endif /* defined(VK_QCOM_tile_properties) */
#if defined(VK_QNX_external_memory_screen_buffer)
	table->vkGetScreenBufferPropertiesQNX = (PFN_vkGetScreenBufferPropertiesQNX)load(context, "vkGetScreenBufferPropertiesQNX");
#endif /* defined(VK_QNX_external_memory_screen_buffer) */
#if defined(VK_VALVE_descriptor_set_host_mapping)
	table->vkGetDescriptorSetHostMappingVALVE = (PFN_vkGetDescriptorSetHostMappingVALVE)load(context, "vkGetDescriptorSetHostMappingVALVE");
	table->vkGetDescriptorSetLayoutHostMappingInfoVALVE = (PFN_vkGetDescriptorSetLayoutHostMappingInfoVALVE)load(context, "vkGetDescriptorSetLayoutHostMappingInfoVALVE");
#endif /* defined(VK_VALVE_descriptor_set_host_mapping) */
#if (defined(VK_EXT_extended_dynamic_state)) || (defined(VK_EXT_shader_object))
	table->vkCmdBindVertexBuffers2EXT = (PFN_vkCmdBindVertexBuffers2EXT)load(context, "vkCmdBindVertexBuffers2EXT");
	table->vkCmdSetCullModeEXT = (PFN_vkCmdSetCullModeEXT)load(context, "vkCmdSetCullModeEXT");
	table->vkCmdSetDepthBoundsTestEnableEXT = (PFN_vkCmdSetDepthBoundsTestEnableEXT)load(context, "vkCmdSetDepthBoundsTestEnableEXT");
	table->vkCmdSetDepthCompareOpEXT = (PFN_vkCmdSetDepthCompareOpEXT)load(context, "vkCmdSetDepthCompareOpEXT");
	table->vkCmdSetDepthTestEnableEXT = (PFN_vkCmdSetDepthTestEnableEXT)load(context, "vkCmdSetDepthTestEnableEXT");
	table->vkCmdSetDepthWriteEnableEXT = (PFN_vkCmdSetDepthWriteEnableEXT)load(context, "vkCmdSetDepthWriteEnableEXT");
	table->vkCmdSetFrontFaceEXT = (PFN_vkCmdSetFrontFaceEXT)load(context, "vkCmdSetFrontFaceEXT");
	table->vkCmdSetPrimitiveTopologyEXT = (PFN_vkCmdSetPrimitiveTopologyEXT)load(context, "vkCmdSetPrimitiveTopologyEXT");
	table->vkCmdSetScissorWithCountEXT = (PFN_vkCmdSetScissorWithCountEXT)load(context, "vkCmdSetScissorWithCountEXT");
	table->vkCmdSetStencilOpEXT = (PFN_vkCmdSetStencilOpEXT)load(context, "vkCmdSetStencilOpEXT");
	table->vkCmdSetStencilTestEnableEXT = (PFN_vkCmdSetStencilTestEnableEXT)load(context, "vkCmdSetStencilTestEnableEXT");
	table->vkCmdSetViewportWithCountEXT = (PFN_vkCmdSetViewportWithCountEXT)load(context, "vkCmdSetViewportWithCountEXT");
#endif /* (defined(VK_EXT_extended_dynamic_state)) || (defined(VK_EXT_shader_object)) */
#if (defined(VK_EXT_extended_dynamic_state2)) || (defined(VK_EXT_shader_object))
	table->vkCmdSetDepthBiasEnableEXT = (PFN_vkCmdSetDepthBiasEnableEXT)load(context, "vkCmdSetDepthBiasEnableEXT");
	table->vkCmdSetLogicOpEXT = (PFN_vkCmdSetLogicOpEXT)load(context, "vkCmdSetLogicOpEXT");
	table->vkCmdSetPatchControlPointsEXT = (PFN_vkCmdSetPatchControlPointsEXT)load(context, "vkCmdSetPatchControlPointsEXT");
	table->vkCmdSetPrimitiveRestartEnableEXT = (PFN_vkCmdSetPrimitiveRestartEnableEXT)load(context, "vkCmdSetPrimitiveRestartEnableEXT");
	table->vkCmdSetRasterizerDiscardEnableEXT = (PFN_vkCmdSetRasterizerDiscardEnableEXT)load(context, "vkCmdSetRasterizerDiscardEnableEXT");
#endif /* (defined(VK_EXT_extended_dynamic_state2)) || (defined(VK_EXT_shader_object)) */
#if (defined(VK_EXT_extended_dynamic_state3)) || (defined(VK_EXT_shader_object))
	table->vkCmdSetAlphaToCoverageEnableEXT = (PFN_vkCmdSetAlphaToCoverageEnableEXT)load(context, "vkCmdSetAlphaToCoverageEnableEXT");
	table->vkCmdSetAlphaToOneEnableEXT = (PFN_vkCmdSetAlphaToOneEnableEXT)load(context, "vkCmdSetAlphaToOneEnableEXT");
	table->vkCmdSetColorBlendAdvancedEXT = (PFN_vkCmdSetColorBlendAdvancedEXT)load(context, "vkCmdSetColorBlendAdvancedEXT");
	table->vkCmdSetColorBlendEnableEXT = (PFN_vkCmdSetColorBlendEnableEXT)load(context, "vkCmdSetColorBlendEnableEXT");
	table->vkCmdSetColorBlendEquationEXT = (PFN_vkCmdSetColorBlendEquationEXT)load(context, "vkCmdSetColorBlendEquationEXT");
	table->vkCmdSetColorWriteMaskEXT = (PFN_vkCmdSetColorWriteMaskEXT)load(context, "vkCmdSetColorWriteMaskEXT");
	table->vkCmdSetConservativeRasterizationModeEXT = (PFN_vkCmdSetConservativeRasterizationModeEXT)load(context, "vkCmdSetConservativeRasterizationModeEXT");
	table->vkCmdSetDepthClampEnableEXT = (PFN_vkCmdSetDepthClampEnableEXT)load(context, "vkCmdSetDepthClampEnableEXT");
	table->vkCmdSetDepthClipEnableEXT = (PFN_vkCmdSetDepthClipEnableEXT)load(context, "vkCmdSetDepthClipEnableEXT");
	table->vkCmdSetDepthClipNegativeOneToOneEXT = (PFN_vkCmdSetDepthClipNegativeOneToOneEXT)load(context, "vkCmdSetDepthClipNegativeOneToOneEXT");
	table->vkCmdSetExtraPrimitiveOverestimationSizeEXT = (PFN_vkCmdSetExtraPrimitiveOverestimationSizeEXT)load(context, "vkCmdSetExtraPrimitiveOverestimationSizeEXT");
	table->vkCmdSetLineRasterizationModeEXT = (PFN_vkCmdSetLineRasterizationModeEXT)load(context, "vkCmdSetLineRasterizationModeEXT");
	table->vkCmdSetLineStippleEnableEXT = (PFN_vkCmdSetLineStippleEnableEXT)load(context, "vkCmdSetLineStippleEnableEXT");
	table->vkCmdSetLogicOpEnableEXT = (PFN_vkCmdSetLogicOpEnableEXT)load(context, "vkCmdSetLogicOpEnableEXT");
	table->vkCmdSetPolygonModeEXT = (PFN_vkCmdSetPolygonModeEXT)load(context, "vkCmdSetPolygonModeEXT");
	table->vkCmdSetProvokingVertexModeEXT = (PFN_vkCmdSetProvokingVertexModeEXT)load(context, "vkCmdSetProvokingVertexModeEXT");
	table->vkCmdSetRasterizationSamplesEXT = (PFN_vkCmdSetRasterizationSamplesEXT)load(context, "vkCmdSetRasterizationSamplesEXT");
	table->vkCmdSetRasterizationStreamEXT = (PFN_vkCmdSetRasterizationStreamEXT)load(context, "vkCmdSetRasterizationStreamEXT");
	table->vkCmdSetSampleLocationsEnableEXT = (PFN_vkCmdSetSampleLocationsEnableEXT)load(context, "vkCmdSetSampleLocationsEnableEXT");
	table->vkCmdSetSampleMaskEXT = (PFN_vkCmdSetSampleMaskEXT)load(context, "vkCmdSetSampleMaskEXT");
	table->vkCmdSetTessellationDomainOriginEXT = (PFN_vkCmdSetTessellationDomainOriginEXT)load(context, "vkCmdSetTessellationDomainOriginEXT");
#endif /* (defined(VK_EXT_extended_dynamic_state3)) || (defined(VK_EXT_shader_object)) */
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_clip_space_w_scaling)) || (defined(VK_EXT_shader_object) && defined(VK_NV_clip_space_w_scaling))
	table->vkCmdSetViewportWScalingEnableNV = (PFN_vkCmdSetViewportWScalingEnableNV)load(context, "vkCmdSetViewportWScalingEnableNV");
#endif /* (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_clip_space_w_scaling)) || (defined(VK_EXT_shader_object) && defined(VK_NV_clip_space_w_scaling)) */
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_viewport_swizzle)) || (defined(VK_EXT_shader_object) && defined(VK_NV_viewport_swizzle))
	table->vkCmdSetViewportSwizzleNV = (PFN_vkCmdSetViewportSwizzleNV)load(context, "vkCmdSetViewportSwizzleNV");
#endif /* (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_viewport_swizzle)) || (defined(VK_EXT_shader_object) && defined(VK_NV_viewport_swizzle)) */
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_fragment_coverage_to_color)) || (defined(VK_EXT_shader_object) && defined(VK_NV_fragment_coverage_to_color))
	table->vkCmdSetCoverageToColorEnableNV = (PFN_vkCmdSetCoverageToColorEnableNV)load(context, "vkCmdSetCoverageToColorEnableNV");
	table->vkCmdSetCoverageToColorLocationNV = (PFN_vkCmdSetCoverageToColorLocationNV)load(context, "vkCmdSetCoverageToColorLocationNV");
#endif /* (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_fragment_coverage_to_color)) || (defined(VK_EXT_shader_object) && defined(VK_NV_fragment_coverage_to_color)) */
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_framebuffer_mixed_samples)) || (defined(VK_EXT_shader_object) && defined(VK_NV_framebuffer_mixed_samples))
	table->vkCmdSetCoverageModulationModeNV = (PFN_vkCmdSetCoverageModulationModeNV)load(context, "vkCmdSetCoverageModulationModeNV");
	table->vkCmdSetCoverageModulationTableEnableNV = (PFN_vkCmdSetCoverageModulationTableEnableNV)load(context, "vkCmdSetCoverageModulationTableEnableNV");
	table->vkCmdSetCoverageModulationTableNV = (PFN_vkCmdSetCoverageModulationTableNV)load(context, "vkCmdSetCoverageModulationTableNV");
#endif /* (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_framebuffer_mixed_samples)) || (defined(VK_EXT_shader_object) && defined(VK_NV_framebuffer_mixed_samples)) */
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_shading_rate_image)) || (defined(VK_EXT_shader_object) && defined(VK_NV_shading_rate_image))
	table->vkCmdSetShadingRateImageEnableNV = (PFN_vkCmdSetShadingRateImageEnableNV)load(context, "vkCmdSetShadingRateImageEnableNV");
#endif /* (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_shading_rate_image)) || (defined(VK_EXT_shader_object) && defined(VK_NV_shading_rate_image)) */
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_representative_fragment_test)) || (defined(VK_EXT_shader_object) && defined(VK_NV_representative_fragment_test))
	table->vkCmdSetRepresentativeFragmentTestEnableNV = (PFN_vkCmdSetRepresentativeFragmentTestEnableNV)load(context, "vkCmdSetRepresentativeFragmentTestEnableNV");
#endif /* (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_representative_fragment_test)) || (defined(VK_EXT_shader_object) && defined(VK_NV_representative_fragment_test)) */
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_coverage_reduction_mode)) || (defined(VK_EXT_shader_object) && defined(VK_NV_coverage_reduction_mode))
	table->vkCmdSetCoverageReductionModeNV = (PFN_vkCmdSetCoverageReductionModeNV)load(context, "vkCmdSetCoverageReductionModeNV");
#endif /* (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_coverage_reduction_mode)) || (defined(VK_EXT_shader_object) && defined(VK_NV_coverage_reduction_mode)) */
#if (defined(VK_EXT_full_screen_exclusive) && defined(VK_KHR_device_group)) || (defined(VK_EXT_full_screen_exclusive) && defined(VK_VERSION_1_1))
	table->vkGetDeviceGroupSurfacePresentModes2EXT = (PFN_vkGetDeviceGroupSurfacePresentModes2EXT)load(context, "vkGetDeviceGroupSurfacePresentModes2EXT");
#endif /* (defined(VK_EXT_full_screen_exclusive) && defined(VK_KHR_device_group)) || (defined(VK_EXT_full_screen_exclusive) && defined(VK_VERSION_1_1)) */
#if (defined(VK_EXT_host_image_copy)) || (defined(VK_EXT_image_compression_control))
	table->vkGetImageSubresourceLayout2EXT = (PFN_vkGetImageSubresourceLayout2EXT)load(context, "vkGetImageSubresourceLayout2EXT");
#endif /* (defined(VK_EXT_host_image_copy)) || (defined(VK_EXT_image_compression_control)) */
#if (defined(VK_EXT_shader_object)) || (defined(VK_EXT_vertex_input_dynamic_state))
	table->vkCmdSetVertexInputEXT = (PFN_vkCmdSetVertexInputEXT)load(context, "vkCmdSetVertexInputEXT");
#endif /* (defined(VK_EXT_shader_object)) || (defined(VK_EXT_vertex_input_dynamic_state)) */
#if (defined(VK_KHR_descriptor_update_template) && defined(VK_KHR_push_descriptor)) || (defined(VK_KHR_push_descriptor) && defined(VK_VERSION_1_1)) || (defined(VK_KHR_push_descriptor) && defined(VK_KHR_descriptor_update_template))
	table->vkCmdPushDescriptorSetWithTemplateKHR = (PFN_vkCmdPushDescriptorSetWithTemplateKHR)load(context, "vkCmdPushDescriptorSetWithTemplateKHR");
#endif /* (defined(VK_KHR_descriptor_update_template) && defined(VK_KHR_push_descriptor)) || (defined(VK_KHR_push_descriptor) && defined(VK_VERSION_1_1)) || (defined(VK_KHR_push_descriptor) && defined(VK_KHR_descriptor_update_template)) */
#if (defined(VK_KHR_device_group) && defined(VK_KHR_surface)) || (defined(VK_KHR_swapchain) && defined(VK_VERSION_1_1))
	table->vkGetDeviceGroupPresentCapabilitiesKHR = (PFN_vkGetDeviceGroupPresentCapabilitiesKHR)load(context, "vkGetDeviceGroupPresentCapabilitiesKHR");
	table->vkGetDeviceGroupSurfacePresentModesKHR = (PFN_vkGetDeviceGroupSurfacePresentModesKHR)load(context, "vkGetDeviceGroupSurfacePresentModesKHR");
#endif /* (defined(VK_KHR_device_group) && defined(VK_KHR_surface)) || (defined(VK_KHR_swapchain) && defined(VK_VERSION_1_1)) */
#if (defined(VK_KHR_device_group) && defined(VK_KHR_swapchain)) || (defined(VK_KHR_swapchain) && defined(VK_VERSION_1_1))
	table->vkAcquireNextImage2KHR = (PFN_vkAcquireNextImage2KHR)load(context, "vkAcquireNextImage2KHR");
#endif /* (defined(VK_KHR_device_group) && defined(VK_KHR_swapchain)) || (defined(VK_KHR_swapchain) && defined(VK_VERSION_1_1)) */
	/* VOLK_GENERATE_LOAD_DEVICE_TABLE */
}

#ifdef __GNUC__
#ifdef VOLK_DEFAULT_VISIBILITY
#	pragma GCC visibility push(default)
#else
#	pragma GCC visibility push(hidden)
#endif
#endif

/* VOLK_GENERATE_PROTOTYPES_C */
#if defined(VK_VERSION_1_0)
PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers;
PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets;
PFN_vkAllocateMemory vkAllocateMemory;
PFN_vkBeginCommandBuffer vkBeginCommandBuffer;
PFN_vkBindBufferMemory vkBindBufferMemory;
PFN_vkBindImageMemory vkBindImageMemory;
PFN_vkCmdBeginQuery vkCmdBeginQuery;
PFN_vkCmdBeginRenderPass vkCmdBeginRenderPass;
PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets;
PFN_vkCmdBindIndexBuffer vkCmdBindIndexBuffer;
PFN_vkCmdBindPipeline vkCmdBindPipeline;
PFN_vkCmdBindVertexBuffers vkCmdBindVertexBuffers;
PFN_vkCmdBlitImage vkCmdBlitImage;
PFN_vkCmdClearAttachments vkCmdClearAttachments;
PFN_vkCmdClearColorImage vkCmdClearColorImage;
PFN_vkCmdClearDepthStencilImage vkCmdClearDepthStencilImage;
PFN_vkCmdCopyBuffer vkCmdCopyBuffer;
PFN_vkCmdCopyBufferToImage vkCmdCopyBufferToImage;
PFN_vkCmdCopyImage vkCmdCopyImage;
PFN_vkCmdCopyImageToBuffer vkCmdCopyImageToBuffer;
PFN_vkCmdCopyQueryPoolResults vkCmdCopyQueryPoolResults;
PFN_vkCmdDispatch vkCmdDispatch;
PFN_vkCmdDispatchIndirect vkCmdDispatchIndirect;
PFN_vkCmdDraw vkCmdDraw;
PFN_vkCmdDrawIndexed vkCmdDrawIndexed;
PFN_vkCmdDrawIndexedIndirect vkCmdDrawIndexedIndirect;
PFN_vkCmdDrawIndirect vkCmdDrawIndirect;
PFN_vkCmdEndQuery vkCmdEndQuery;
PFN_vkCmdEndRenderPass vkCmdEndRenderPass;
PFN_vkCmdExecuteCommands vkCmdExecuteCommands;
PFN_vkCmdFillBuffer vkCmdFillBuffer;
PFN_vkCmdNextSubpass vkCmdNextSubpass;
PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier;
PFN_vkCmdPushConstants vkCmdPushConstants;
PFN_vkCmdResetEvent vkCmdResetEvent;
PFN_vkCmdResetQueryPool vkCmdResetQueryPool;
PFN_vkCmdResolveImage vkCmdResolveImage;
PFN_vkCmdSetBlendConstants vkCmdSetBlendConstants;
PFN_vkCmdSetDepthBias vkCmdSetDepthBias;
PFN_vkCmdSetDepthBounds vkCmdSetDepthBounds;
PFN_vkCmdSetEvent vkCmdSetEvent;
PFN_vkCmdSetLineWidth vkCmdSetLineWidth;
PFN_vkCmdSetScissor vkCmdSetScissor;
PFN_vkCmdSetStencilCompareMask vkCmdSetStencilCompareMask;
PFN_vkCmdSetStencilReference vkCmdSetStencilReference;
PFN_vkCmdSetStencilWriteMask vkCmdSetStencilWriteMask;
PFN_vkCmdSetViewport vkCmdSetViewport;
PFN_vkCmdUpdateBuffer vkCmdUpdateBuffer;
PFN_vkCmdWaitEvents vkCmdWaitEvents;
PFN_vkCmdWriteTimestamp vkCmdWriteTimestamp;
PFN_vkCreateBuffer vkCreateBuffer;
PFN_vkCreateBufferView vkCreateBufferView;
PFN_vkCreateCommandPool vkCreateCommandPool;
PFN_vkCreateComputePipelines vkCreateComputePipelines;
PFN_vkCreateDescriptorPool vkCreateDescriptorPool;
PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout;
PFN_vkCreateDevice vkCreateDevice;
PFN_vkCreateEvent vkCreateEvent;
PFN_vkCreateFence vkCreateFence;
PFN_vkCreateFramebuffer vkCreateFramebuffer;
PFN_vkCreateGraphicsPipelines vkCreateGraphicsPipelines;
PFN_vkCreateImage vkCreateImage;
PFN_vkCreateImageView vkCreateImageView;
PFN_vkCreateInstance vkCreateInstance;
PFN_vkCreatePipelineCache vkCreatePipelineCache;
PFN_vkCreatePipelineLayout vkCreatePipelineLayout;
PFN_vkCreateQueryPool vkCreateQueryPool;
PFN_vkCreateRenderPass vkCreateRenderPass;
PFN_vkCreateSampler vkCreateSampler;
PFN_vkCreateSemaphore vkCreateSemaphore;
PFN_vkCreateShaderModule vkCreateShaderModule;
PFN_vkDestroyBuffer vkDestroyBuffer;
PFN_vkDestroyBufferView vkDestroyBufferView;
PFN_vkDestroyCommandPool vkDestroyCommandPool;
PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool;
PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout;
PFN_vkDestroyDevice vkDestroyDevice;
PFN_vkDestroyEvent vkDestroyEvent;
PFN_vkDestroyFence vkDestroyFence;
PFN_vkDestroyFramebuffer vkDestroyFramebuffer;
PFN_vkDestroyImage vkDestroyImage;
PFN_vkDestroyImageView vkDestroyImageView;
PFN_vkDestroyInstance vkDestroyInstance;
PFN_vkDestroyPipeline vkDestroyPipeline;
PFN_vkDestroyPipelineCache vkDestroyPipelineCache;
PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout;
PFN_vkDestroyQueryPool vkDestroyQueryPool;
PFN_vkDestroyRenderPass vkDestroyRenderPass;
PFN_vkDestroySampler vkDestroySampler;
PFN_vkDestroySemaphore vkDestroySemaphore;
PFN_vkDestroyShaderModule vkDestroyShaderModule;
PFN_vkDeviceWaitIdle vkDeviceWaitIdle;
PFN_vkEndCommandBuffer vkEndCommandBuffer;
PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties;
PFN_vkEnumerateDeviceLayerProperties vkEnumerateDeviceLayerProperties;
PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties;
PFN_vkEnumerateInstanceLayerProperties vkEnumerateInstanceLayerProperties;
PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices;
PFN_vkFlushMappedMemoryRanges vkFlushMappedMemoryRanges;
PFN_vkFreeCommandBuffers vkFreeCommandBuffers;
PFN_vkFreeDescriptorSets vkFreeDescriptorSets;
PFN_vkFreeMemory vkFreeMemory;
PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements;
PFN_vkGetDeviceMemoryCommitment vkGetDeviceMemoryCommitment;
PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr;
PFN_vkGetDeviceQueue vkGetDeviceQueue;
PFN_vkGetEventStatus vkGetEventStatus;
PFN_vkGetFenceStatus vkGetFenceStatus;
PFN_vkGetImageMemoryRequirements vkGetImageMemoryRequirements;
PFN_vkGetImageSparseMemoryRequirements vkGetImageSparseMemoryRequirements;
PFN_vkGetImageSubresourceLayout vkGetImageSubresourceLayout;
PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
PFN_vkGetPhysicalDeviceFeatures vkGetPhysicalDeviceFeatures;
PFN_vkGetPhysicalDeviceFormatProperties vkGetPhysicalDeviceFormatProperties;
PFN_vkGetPhysicalDeviceImageFormatProperties vkGetPhysicalDeviceImageFormatProperties;
PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties;
PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties;
PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties;
PFN_vkGetPhysicalDeviceSparseImageFormatProperties vkGetPhysicalDeviceSparseImageFormatProperties;
PFN_vkGetPipelineCacheData vkGetPipelineCacheData;
PFN_vkGetQueryPoolResults vkGetQueryPoolResults;
PFN_vkGetRenderAreaGranularity vkGetRenderAreaGranularity;
PFN_vkInvalidateMappedMemoryRanges vkInvalidateMappedMemoryRanges;
PFN_vkMapMemory vkMapMemory;
PFN_vkMergePipelineCaches vkMergePipelineCaches;
PFN_vkQueueBindSparse vkQueueBindSparse;
PFN_vkQueueSubmit vkQueueSubmit;
PFN_vkQueueWaitIdle vkQueueWaitIdle;
PFN_vkResetCommandBuffer vkResetCommandBuffer;
PFN_vkResetCommandPool vkResetCommandPool;
PFN_vkResetDescriptorPool vkResetDescriptorPool;
PFN_vkResetEvent vkResetEvent;
PFN_vkResetFences vkResetFences;
PFN_vkSetEvent vkSetEvent;
PFN_vkUnmapMemory vkUnmapMemory;
PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets;
PFN_vkWaitForFences vkWaitForFences;
#endif /* defined(VK_VERSION_1_0) */
#if defined(VK_VERSION_1_1)
PFN_vkBindBufferMemory2 vkBindBufferMemory2;
PFN_vkBindImageMemory2 vkBindImageMemory2;
PFN_vkCmdDispatchBase vkCmdDispatchBase;
PFN_vkCmdSetDeviceMask vkCmdSetDeviceMask;
PFN_vkCreateDescriptorUpdateTemplate vkCreateDescriptorUpdateTemplate;
PFN_vkCreateSamplerYcbcrConversion vkCreateSamplerYcbcrConversion;
PFN_vkDestroyDescriptorUpdateTemplate vkDestroyDescriptorUpdateTemplate;
PFN_vkDestroySamplerYcbcrConversion vkDestroySamplerYcbcrConversion;
PFN_vkEnumerateInstanceVersion vkEnumerateInstanceVersion;
PFN_vkEnumeratePhysicalDeviceGroups vkEnumeratePhysicalDeviceGroups;
PFN_vkGetBufferMemoryRequirements2 vkGetBufferMemoryRequirements2;
PFN_vkGetDescriptorSetLayoutSupport vkGetDescriptorSetLayoutSupport;
PFN_vkGetDeviceGroupPeerMemoryFeatures vkGetDeviceGroupPeerMemoryFeatures;
PFN_vkGetDeviceQueue2 vkGetDeviceQueue2;
PFN_vkGetImageMemoryRequirements2 vkGetImageMemoryRequirements2;
PFN_vkGetImageSparseMemoryRequirements2 vkGetImageSparseMemoryRequirements2;
PFN_vkGetPhysicalDeviceExternalBufferProperties vkGetPhysicalDeviceExternalBufferProperties;
PFN_vkGetPhysicalDeviceExternalFenceProperties vkGetPhysicalDeviceExternalFenceProperties;
PFN_vkGetPhysicalDeviceExternalSemaphoreProperties vkGetPhysicalDeviceExternalSemaphoreProperties;
PFN_vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFeatures2;
PFN_vkGetPhysicalDeviceFormatProperties2 vkGetPhysicalDeviceFormatProperties2;
PFN_vkGetPhysicalDeviceImageFormatProperties2 vkGetPhysicalDeviceImageFormatProperties2;
PFN_vkGetPhysicalDeviceMemoryProperties2 vkGetPhysicalDeviceMemoryProperties2;
PFN_vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceProperties2;
PFN_vkGetPhysicalDeviceQueueFamilyProperties2 vkGetPhysicalDeviceQueueFamilyProperties2;
PFN_vkGetPhysicalDeviceSparseImageFormatProperties2 vkGetPhysicalDeviceSparseImageFormatProperties2;
PFN_vkTrimCommandPool vkTrimCommandPool;
PFN_vkUpdateDescriptorSetWithTemplate vkUpdateDescriptorSetWithTemplate;
#endif /* defined(VK_VERSION_1_1) */
#if defined(VK_VERSION_1_2)
PFN_vkCmdBeginRenderPass2 vkCmdBeginRenderPass2;
PFN_vkCmdDrawIndexedIndirectCount vkCmdDrawIndexedIndirectCount;
PFN_vkCmdDrawIndirectCount vkCmdDrawIndirectCount;
PFN_vkCmdEndRenderPass2 vkCmdEndRenderPass2;
PFN_vkCmdNextSubpass2 vkCmdNextSubpass2;
PFN_vkCreateRenderPass2 vkCreateRenderPass2;
PFN_vkGetBufferDeviceAddress vkGetBufferDeviceAddress;
PFN_vkGetBufferOpaqueCaptureAddress vkGetBufferOpaqueCaptureAddress;
PFN_vkGetDeviceMemoryOpaqueCaptureAddress vkGetDeviceMemoryOpaqueCaptureAddress;
PFN_vkGetSemaphoreCounterValue vkGetSemaphoreCounterValue;
PFN_vkResetQueryPool vkResetQueryPool;
PFN_vkSignalSemaphore vkSignalSemaphore;
PFN_vkWaitSemaphores vkWaitSemaphores;
#endif /* defined(VK_VERSION_1_2) */
#if defined(VK_VERSION_1_3)
PFN_vkCmdBeginRendering vkCmdBeginRendering;
PFN_vkCmdBindVertexBuffers2 vkCmdBindVertexBuffers2;
PFN_vkCmdBlitImage2 vkCmdBlitImage2;
PFN_vkCmdCopyBuffer2 vkCmdCopyBuffer2;
PFN_vkCmdCopyBufferToImage2 vkCmdCopyBufferToImage2;
PFN_vkCmdCopyImage2 vkCmdCopyImage2;
PFN_vkCmdCopyImageToBuffer2 vkCmdCopyImageToBuffer2;
PFN_vkCmdEndRendering vkCmdEndRendering;
PFN_vkCmdPipelineBarrier2 vkCmdPipelineBarrier2;
PFN_vkCmdResetEvent2 vkCmdResetEvent2;
PFN_vkCmdResolveImage2 vkCmdResolveImage2;
PFN_vkCmdSetCullMode vkCmdSetCullMode;
PFN_vkCmdSetDepthBiasEnable vkCmdSetDepthBiasEnable;
PFN_vkCmdSetDepthBoundsTestEnable vkCmdSetDepthBoundsTestEnable;
PFN_vkCmdSetDepthCompareOp vkCmdSetDepthCompareOp;
PFN_vkCmdSetDepthTestEnable vkCmdSetDepthTestEnable;
PFN_vkCmdSetDepthWriteEnable vkCmdSetDepthWriteEnable;
PFN_vkCmdSetEvent2 vkCmdSetEvent2;
PFN_vkCmdSetFrontFace vkCmdSetFrontFace;
PFN_vkCmdSetPrimitiveRestartEnable vkCmdSetPrimitiveRestartEnable;
PFN_vkCmdSetPrimitiveTopology vkCmdSetPrimitiveTopology;
PFN_vkCmdSetRasterizerDiscardEnable vkCmdSetRasterizerDiscardEnable;
PFN_vkCmdSetScissorWithCount vkCmdSetScissorWithCount;
PFN_vkCmdSetStencilOp vkCmdSetStencilOp;
PFN_vkCmdSetStencilTestEnable vkCmdSetStencilTestEnable;
PFN_vkCmdSetViewportWithCount vkCmdSetViewportWithCount;
PFN_vkCmdWaitEvents2 vkCmdWaitEvents2;
PFN_vkCmdWriteTimestamp2 vkCmdWriteTimestamp2;
PFN_vkCreatePrivateDataSlot vkCreatePrivateDataSlot;
PFN_vkDestroyPrivateDataSlot vkDestroyPrivateDataSlot;
PFN_vkGetDeviceBufferMemoryRequirements vkGetDeviceBufferMemoryRequirements;
PFN_vkGetDeviceImageMemoryRequirements vkGetDeviceImageMemoryRequirements;
PFN_vkGetDeviceImageSparseMemoryRequirements vkGetDeviceImageSparseMemoryRequirements;
PFN_vkGetPhysicalDeviceToolProperties vkGetPhysicalDeviceToolProperties;
PFN_vkGetPrivateData vkGetPrivateData;
PFN_vkQueueSubmit2 vkQueueSubmit2;
PFN_vkSetPrivateData vkSetPrivateData;
#endif /* defined(VK_VERSION_1_3) */
#if defined(VK_AMDX_shader_enqueue)
PFN_vkCmdDispatchGraphAMDX vkCmdDispatchGraphAMDX;
PFN_vkCmdDispatchGraphIndirectAMDX vkCmdDispatchGraphIndirectAMDX;
PFN_vkCmdDispatchGraphIndirectCountAMDX vkCmdDispatchGraphIndirectCountAMDX;
PFN_vkCmdInitializeGraphScratchMemoryAMDX vkCmdInitializeGraphScratchMemoryAMDX;
PFN_vkCreateExecutionGraphPipelinesAMDX vkCreateExecutionGraphPipelinesAMDX;
PFN_vkGetExecutionGraphPipelineNodeIndexAMDX vkGetExecutionGraphPipelineNodeIndexAMDX;
PFN_vkGetExecutionGraphPipelineScratchSizeAMDX vkGetExecutionGraphPipelineScratchSizeAMDX;
#endif /* defined(VK_AMDX_shader_enqueue) */
#if defined(VK_AMD_buffer_marker)
PFN_vkCmdWriteBufferMarkerAMD vkCmdWriteBufferMarkerAMD;
#endif /* defined(VK_AMD_buffer_marker) */
#if defined(VK_AMD_display_native_hdr)
PFN_vkSetLocalDimmingAMD vkSetLocalDimmingAMD;
#endif /* defined(VK_AMD_display_native_hdr) */
#if defined(VK_AMD_draw_indirect_count)
PFN_vkCmdDrawIndexedIndirectCountAMD vkCmdDrawIndexedIndirectCountAMD;
PFN_vkCmdDrawIndirectCountAMD vkCmdDrawIndirectCountAMD;
#endif /* defined(VK_AMD_draw_indirect_count) */
#if defined(VK_AMD_shader_info)
PFN_vkGetShaderInfoAMD vkGetShaderInfoAMD;
#endif /* defined(VK_AMD_shader_info) */
#if defined(VK_ANDROID_external_memory_android_hardware_buffer)
PFN_vkGetAndroidHardwareBufferPropertiesANDROID vkGetAndroidHardwareBufferPropertiesANDROID;
PFN_vkGetMemoryAndroidHardwareBufferANDROID vkGetMemoryAndroidHardwareBufferANDROID;
#endif /* defined(VK_ANDROID_external_memory_android_hardware_buffer) */
#if defined(VK_EXT_acquire_drm_display)
PFN_vkAcquireDrmDisplayEXT vkAcquireDrmDisplayEXT;
PFN_vkGetDrmDisplayEXT vkGetDrmDisplayEXT;
#endif /* defined(VK_EXT_acquire_drm_display) */
#if defined(VK_EXT_acquire_xlib_display)
PFN_vkAcquireXlibDisplayEXT vkAcquireXlibDisplayEXT;
PFN_vkGetRandROutputDisplayEXT vkGetRandROutputDisplayEXT;
#endif /* defined(VK_EXT_acquire_xlib_display) */
#if defined(VK_EXT_attachment_feedback_loop_dynamic_state)
PFN_vkCmdSetAttachmentFeedbackLoopEnableEXT vkCmdSetAttachmentFeedbackLoopEnableEXT;
#endif /* defined(VK_EXT_attachment_feedback_loop_dynamic_state) */
#if defined(VK_EXT_buffer_device_address)
PFN_vkGetBufferDeviceAddressEXT vkGetBufferDeviceAddressEXT;
#endif /* defined(VK_EXT_buffer_device_address) */
#if defined(VK_EXT_calibrated_timestamps)
PFN_vkGetCalibratedTimestampsEXT vkGetCalibratedTimestampsEXT;
PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT vkGetPhysicalDeviceCalibrateableTimeDomainsEXT;
#endif /* defined(VK_EXT_calibrated_timestamps) */
#if defined(VK_EXT_color_write_enable)
PFN_vkCmdSetColorWriteEnableEXT vkCmdSetColorWriteEnableEXT;
#endif /* defined(VK_EXT_color_write_enable) */
#if defined(VK_EXT_conditional_rendering)
PFN_vkCmdBeginConditionalRenderingEXT vkCmdBeginConditionalRenderingEXT;
PFN_vkCmdEndConditionalRenderingEXT vkCmdEndConditionalRenderingEXT;
#endif /* defined(VK_EXT_conditional_rendering) */
#if defined(VK_EXT_debug_marker)
PFN_vkCmdDebugMarkerBeginEXT vkCmdDebugMarkerBeginEXT;
PFN_vkCmdDebugMarkerEndEXT vkCmdDebugMarkerEndEXT;
PFN_vkCmdDebugMarkerInsertEXT vkCmdDebugMarkerInsertEXT;
PFN_vkDebugMarkerSetObjectNameEXT vkDebugMarkerSetObjectNameEXT;
PFN_vkDebugMarkerSetObjectTagEXT vkDebugMarkerSetObjectTagEXT;
#endif /* defined(VK_EXT_debug_marker) */
#if defined(VK_EXT_debug_report)
PFN_vkCreateDebugReportCallbackEXT vkCreateDebugReportCallbackEXT;
PFN_vkDebugReportMessageEXT vkDebugReportMessageEXT;
PFN_vkDestroyDebugReportCallbackEXT vkDestroyDebugReportCallbackEXT;
#endif /* defined(VK_EXT_debug_report) */
#if defined(VK_EXT_debug_utils)
PFN_vkCmdBeginDebugUtilsLabelEXT vkCmdBeginDebugUtilsLabelEXT;
PFN_vkCmdEndDebugUtilsLabelEXT vkCmdEndDebugUtilsLabelEXT;
PFN_vkCmdInsertDebugUtilsLabelEXT vkCmdInsertDebugUtilsLabelEXT;
PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT;
PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT;
PFN_vkQueueBeginDebugUtilsLabelEXT vkQueueBeginDebugUtilsLabelEXT;
PFN_vkQueueEndDebugUtilsLabelEXT vkQueueEndDebugUtilsLabelEXT;
PFN_vkQueueInsertDebugUtilsLabelEXT vkQueueInsertDebugUtilsLabelEXT;
PFN_vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectNameEXT;
PFN_vkSetDebugUtilsObjectTagEXT vkSetDebugUtilsObjectTagEXT;
PFN_vkSubmitDebugUtilsMessageEXT vkSubmitDebugUtilsMessageEXT;
#endif /* defined(VK_EXT_debug_utils) */
#if defined(VK_EXT_depth_bias_control)
PFN_vkCmdSetDepthBias2EXT vkCmdSetDepthBias2EXT;
#endif /* defined(VK_EXT_depth_bias_control) */
#if defined(VK_EXT_descriptor_buffer)
PFN_vkCmdBindDescriptorBufferEmbeddedSamplersEXT vkCmdBindDescriptorBufferEmbeddedSamplersEXT;
PFN_vkCmdBindDescriptorBuffersEXT vkCmdBindDescriptorBuffersEXT;
PFN_vkCmdSetDescriptorBufferOffsetsEXT vkCmdSetDescriptorBufferOffsetsEXT;
PFN_vkGetBufferOpaqueCaptureDescriptorDataEXT vkGetBufferOpaqueCaptureDescriptorDataEXT;
PFN_vkGetDescriptorEXT vkGetDescriptorEXT;
PFN_vkGetDescriptorSetLayoutBindingOffsetEXT vkGetDescriptorSetLayoutBindingOffsetEXT;
PFN_vkGetDescriptorSetLayoutSizeEXT vkGetDescriptorSetLayoutSizeEXT;
PFN_vkGetImageOpaqueCaptureDescriptorDataEXT vkGetImageOpaqueCaptureDescriptorDataEXT;
PFN_vkGetImageViewOpaqueCaptureDescriptorDataEXT vkGetImageViewOpaqueCaptureDescriptorDataEXT;
PFN_vkGetSamplerOpaqueCaptureDescriptorDataEXT vkGetSamplerOpaqueCaptureDescriptorDataEXT;
#endif /* defined(VK_EXT_descriptor_buffer) */
#if defined(VK_EXT_descriptor_buffer) && (defined(VK_KHR_acceleration_structure) || defined(VK_NV_ray_tracing))
PFN_vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT vkGetAccelerationStructureOpaqueCaptureDescriptorDataEXT;
#endif /* defined(VK_EXT_descriptor_buffer) && (defined(VK_KHR_acceleration_structure) || defined(VK_NV_ray_tracing)) */
#if defined(VK_EXT_device_fault)
PFN_vkGetDeviceFaultInfoEXT vkGetDeviceFaultInfoEXT;
#endif /* defined(VK_EXT_device_fault) */
#if defined(VK_EXT_direct_mode_display)
PFN_vkReleaseDisplayEXT vkReleaseDisplayEXT;
#endif /* defined(VK_EXT_direct_mode_display) */
#if defined(VK_EXT_directfb_surface)
PFN_vkCreateDirectFBSurfaceEXT vkCreateDirectFBSurfaceEXT;
PFN_vkGetPhysicalDeviceDirectFBPresentationSupportEXT vkGetPhysicalDeviceDirectFBPresentationSupportEXT;
#endif /* defined(VK_EXT_directfb_surface) */
#if defined(VK_EXT_discard_rectangles)
PFN_vkCmdSetDiscardRectangleEXT vkCmdSetDiscardRectangleEXT;
#endif /* defined(VK_EXT_discard_rectangles) */
#if defined(VK_EXT_discard_rectangles) && VK_EXT_DISCARD_RECTANGLES_SPEC_VERSION >= 2
PFN_vkCmdSetDiscardRectangleEnableEXT vkCmdSetDiscardRectangleEnableEXT;
PFN_vkCmdSetDiscardRectangleModeEXT vkCmdSetDiscardRectangleModeEXT;
#endif /* defined(VK_EXT_discard_rectangles) && VK_EXT_DISCARD_RECTANGLES_SPEC_VERSION >= 2 */
#if defined(VK_EXT_display_control)
PFN_vkDisplayPowerControlEXT vkDisplayPowerControlEXT;
PFN_vkGetSwapchainCounterEXT vkGetSwapchainCounterEXT;
PFN_vkRegisterDeviceEventEXT vkRegisterDeviceEventEXT;
PFN_vkRegisterDisplayEventEXT vkRegisterDisplayEventEXT;
#endif /* defined(VK_EXT_display_control) */
#if defined(VK_EXT_display_surface_counter)
PFN_vkGetPhysicalDeviceSurfaceCapabilities2EXT vkGetPhysicalDeviceSurfaceCapabilities2EXT;
#endif /* defined(VK_EXT_display_surface_counter) */
#if defined(VK_EXT_external_memory_host)
PFN_vkGetMemoryHostPointerPropertiesEXT vkGetMemoryHostPointerPropertiesEXT;
#endif /* defined(VK_EXT_external_memory_host) */
#if defined(VK_EXT_full_screen_exclusive)
PFN_vkAcquireFullScreenExclusiveModeEXT vkAcquireFullScreenExclusiveModeEXT;
PFN_vkGetPhysicalDeviceSurfacePresentModes2EXT vkGetPhysicalDeviceSurfacePresentModes2EXT;
PFN_vkReleaseFullScreenExclusiveModeEXT vkReleaseFullScreenExclusiveModeEXT;
#endif /* defined(VK_EXT_full_screen_exclusive) */
#if defined(VK_EXT_hdr_metadata)
PFN_vkSetHdrMetadataEXT vkSetHdrMetadataEXT;
#endif /* defined(VK_EXT_hdr_metadata) */
#if defined(VK_EXT_headless_surface)
PFN_vkCreateHeadlessSurfaceEXT vkCreateHeadlessSurfaceEXT;
#endif /* defined(VK_EXT_headless_surface) */
#if defined(VK_EXT_host_image_copy)
PFN_vkCopyImageToImageEXT vkCopyImageToImageEXT;
PFN_vkCopyImageToMemoryEXT vkCopyImageToMemoryEXT;
PFN_vkCopyMemoryToImageEXT vkCopyMemoryToImageEXT;
PFN_vkTransitionImageLayoutEXT vkTransitionImageLayoutEXT;
#endif /* defined(VK_EXT_host_image_copy) */
#if defined(VK_EXT_host_query_reset)
PFN_vkResetQueryPoolEXT vkResetQueryPoolEXT;
#endif /* defined(VK_EXT_host_query_reset) */
#if defined(VK_EXT_image_drm_format_modifier)
PFN_vkGetImageDrmFormatModifierPropertiesEXT vkGetImageDrmFormatModifierPropertiesEXT;
#endif /* defined(VK_EXT_image_drm_format_modifier) */
#if defined(VK_EXT_line_rasterization)
PFN_vkCmdSetLineStippleEXT vkCmdSetLineStippleEXT;
#endif /* defined(VK_EXT_line_rasterization) */
#if defined(VK_EXT_mesh_shader)
PFN_vkCmdDrawMeshTasksEXT vkCmdDrawMeshTasksEXT;
PFN_vkCmdDrawMeshTasksIndirectCountEXT vkCmdDrawMeshTasksIndirectCountEXT;
PFN_vkCmdDrawMeshTasksIndirectEXT vkCmdDrawMeshTasksIndirectEXT;
#endif /* defined(VK_EXT_mesh_shader) */
#if defined(VK_EXT_metal_objects)
PFN_vkExportMetalObjectsEXT vkExportMetalObjectsEXT;
#endif /* defined(VK_EXT_metal_objects) */
#if defined(VK_EXT_metal_surface)
PFN_vkCreateMetalSurfaceEXT vkCreateMetalSurfaceEXT;
#endif /* defined(VK_EXT_metal_surface) */
#if defined(VK_EXT_multi_draw)
PFN_vkCmdDrawMultiEXT vkCmdDrawMultiEXT;
PFN_vkCmdDrawMultiIndexedEXT vkCmdDrawMultiIndexedEXT;
#endif /* defined(VK_EXT_multi_draw) */
#if defined(VK_EXT_opacity_micromap)
PFN_vkBuildMicromapsEXT vkBuildMicromapsEXT;
PFN_vkCmdBuildMicromapsEXT vkCmdBuildMicromapsEXT;
PFN_vkCmdCopyMemoryToMicromapEXT vkCmdCopyMemoryToMicromapEXT;
PFN_vkCmdCopyMicromapEXT vkCmdCopyMicromapEXT;
PFN_vkCmdCopyMicromapToMemoryEXT vkCmdCopyMicromapToMemoryEXT;
PFN_vkCmdWriteMicromapsPropertiesEXT vkCmdWriteMicromapsPropertiesEXT;
PFN_vkCopyMemoryToMicromapEXT vkCopyMemoryToMicromapEXT;
PFN_vkCopyMicromapEXT vkCopyMicromapEXT;
PFN_vkCopyMicromapToMemoryEXT vkCopyMicromapToMemoryEXT;
PFN_vkCreateMicromapEXT vkCreateMicromapEXT;
PFN_vkDestroyMicromapEXT vkDestroyMicromapEXT;
PFN_vkGetDeviceMicromapCompatibilityEXT vkGetDeviceMicromapCompatibilityEXT;
PFN_vkGetMicromapBuildSizesEXT vkGetMicromapBuildSizesEXT;
PFN_vkWriteMicromapsPropertiesEXT vkWriteMicromapsPropertiesEXT;
#endif /* defined(VK_EXT_opacity_micromap) */
#if defined(VK_EXT_pageable_device_local_memory)
PFN_vkSetDeviceMemoryPriorityEXT vkSetDeviceMemoryPriorityEXT;
#endif /* defined(VK_EXT_pageable_device_local_memory) */
#if defined(VK_EXT_pipeline_properties)
PFN_vkGetPipelinePropertiesEXT vkGetPipelinePropertiesEXT;
#endif /* defined(VK_EXT_pipeline_properties) */
#if defined(VK_EXT_private_data)
PFN_vkCreatePrivateDataSlotEXT vkCreatePrivateDataSlotEXT;
PFN_vkDestroyPrivateDataSlotEXT vkDestroyPrivateDataSlotEXT;
PFN_vkGetPrivateDataEXT vkGetPrivateDataEXT;
PFN_vkSetPrivateDataEXT vkSetPrivateDataEXT;
#endif /* defined(VK_EXT_private_data) */
#if defined(VK_EXT_sample_locations)
PFN_vkCmdSetSampleLocationsEXT vkCmdSetSampleLocationsEXT;
PFN_vkGetPhysicalDeviceMultisamplePropertiesEXT vkGetPhysicalDeviceMultisamplePropertiesEXT;
#endif /* defined(VK_EXT_sample_locations) */
#if defined(VK_EXT_shader_module_identifier)
PFN_vkGetShaderModuleCreateInfoIdentifierEXT vkGetShaderModuleCreateInfoIdentifierEXT;
PFN_vkGetShaderModuleIdentifierEXT vkGetShaderModuleIdentifierEXT;
#endif /* defined(VK_EXT_shader_module_identifier) */
#if defined(VK_EXT_shader_object)
PFN_vkCmdBindShadersEXT vkCmdBindShadersEXT;
PFN_vkCreateShadersEXT vkCreateShadersEXT;
PFN_vkDestroyShaderEXT vkDestroyShaderEXT;
PFN_vkGetShaderBinaryDataEXT vkGetShaderBinaryDataEXT;
#endif /* defined(VK_EXT_shader_object) */
#if defined(VK_EXT_swapchain_maintenance1)
PFN_vkReleaseSwapchainImagesEXT vkReleaseSwapchainImagesEXT;
#endif /* defined(VK_EXT_swapchain_maintenance1) */
#if defined(VK_EXT_tooling_info)
PFN_vkGetPhysicalDeviceToolPropertiesEXT vkGetPhysicalDeviceToolPropertiesEXT;
#endif /* defined(VK_EXT_tooling_info) */
#if defined(VK_EXT_transform_feedback)
PFN_vkCmdBeginQueryIndexedEXT vkCmdBeginQueryIndexedEXT;
PFN_vkCmdBeginTransformFeedbackEXT vkCmdBeginTransformFeedbackEXT;
PFN_vkCmdBindTransformFeedbackBuffersEXT vkCmdBindTransformFeedbackBuffersEXT;
PFN_vkCmdDrawIndirectByteCountEXT vkCmdDrawIndirectByteCountEXT;
PFN_vkCmdEndQueryIndexedEXT vkCmdEndQueryIndexedEXT;
PFN_vkCmdEndTransformFeedbackEXT vkCmdEndTransformFeedbackEXT;
#endif /* defined(VK_EXT_transform_feedback) */
#if defined(VK_EXT_validation_cache)
PFN_vkCreateValidationCacheEXT vkCreateValidationCacheEXT;
PFN_vkDestroyValidationCacheEXT vkDestroyValidationCacheEXT;
PFN_vkGetValidationCacheDataEXT vkGetValidationCacheDataEXT;
PFN_vkMergeValidationCachesEXT vkMergeValidationCachesEXT;
#endif /* defined(VK_EXT_validation_cache) */
#if defined(VK_FUCHSIA_buffer_collection)
PFN_vkCreateBufferCollectionFUCHSIA vkCreateBufferCollectionFUCHSIA;
PFN_vkDestroyBufferCollectionFUCHSIA vkDestroyBufferCollectionFUCHSIA;
PFN_vkGetBufferCollectionPropertiesFUCHSIA vkGetBufferCollectionPropertiesFUCHSIA;
PFN_vkSetBufferCollectionBufferConstraintsFUCHSIA vkSetBufferCollectionBufferConstraintsFUCHSIA;
PFN_vkSetBufferCollectionImageConstraintsFUCHSIA vkSetBufferCollectionImageConstraintsFUCHSIA;
#endif /* defined(VK_FUCHSIA_buffer_collection) */
#if defined(VK_FUCHSIA_external_memory)
PFN_vkGetMemoryZirconHandleFUCHSIA vkGetMemoryZirconHandleFUCHSIA;
PFN_vkGetMemoryZirconHandlePropertiesFUCHSIA vkGetMemoryZirconHandlePropertiesFUCHSIA;
#endif /* defined(VK_FUCHSIA_external_memory) */
#if defined(VK_FUCHSIA_external_semaphore)
PFN_vkGetSemaphoreZirconHandleFUCHSIA vkGetSemaphoreZirconHandleFUCHSIA;
PFN_vkImportSemaphoreZirconHandleFUCHSIA vkImportSemaphoreZirconHandleFUCHSIA;
#endif /* defined(VK_FUCHSIA_external_semaphore) */
#if defined(VK_FUCHSIA_imagepipe_surface)
PFN_vkCreateImagePipeSurfaceFUCHSIA vkCreateImagePipeSurfaceFUCHSIA;
#endif /* defined(VK_FUCHSIA_imagepipe_surface) */
#if defined(VK_GGP_stream_descriptor_surface)
PFN_vkCreateStreamDescriptorSurfaceGGP vkCreateStreamDescriptorSurfaceGGP;
#endif /* defined(VK_GGP_stream_descriptor_surface) */
#if defined(VK_GOOGLE_display_timing)
PFN_vkGetPastPresentationTimingGOOGLE vkGetPastPresentationTimingGOOGLE;
PFN_vkGetRefreshCycleDurationGOOGLE vkGetRefreshCycleDurationGOOGLE;
#endif /* defined(VK_GOOGLE_display_timing) */
#if defined(VK_HUAWEI_cluster_culling_shader)
PFN_vkCmdDrawClusterHUAWEI vkCmdDrawClusterHUAWEI;
PFN_vkCmdDrawClusterIndirectHUAWEI vkCmdDrawClusterIndirectHUAWEI;
#endif /* defined(VK_HUAWEI_cluster_culling_shader) */
#if defined(VK_HUAWEI_invocation_mask)
PFN_vkCmdBindInvocationMaskHUAWEI vkCmdBindInvocationMaskHUAWEI;
#endif /* defined(VK_HUAWEI_invocation_mask) */
#if defined(VK_HUAWEI_subpass_shading)
PFN_vkCmdSubpassShadingHUAWEI vkCmdSubpassShadingHUAWEI;
PFN_vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI vkGetDeviceSubpassShadingMaxWorkgroupSizeHUAWEI;
#endif /* defined(VK_HUAWEI_subpass_shading) */
#if defined(VK_INTEL_performance_query)
PFN_vkAcquirePerformanceConfigurationINTEL vkAcquirePerformanceConfigurationINTEL;
PFN_vkCmdSetPerformanceMarkerINTEL vkCmdSetPerformanceMarkerINTEL;
PFN_vkCmdSetPerformanceOverrideINTEL vkCmdSetPerformanceOverrideINTEL;
PFN_vkCmdSetPerformanceStreamMarkerINTEL vkCmdSetPerformanceStreamMarkerINTEL;
PFN_vkGetPerformanceParameterINTEL vkGetPerformanceParameterINTEL;
PFN_vkInitializePerformanceApiINTEL vkInitializePerformanceApiINTEL;
PFN_vkQueueSetPerformanceConfigurationINTEL vkQueueSetPerformanceConfigurationINTEL;
PFN_vkReleasePerformanceConfigurationINTEL vkReleasePerformanceConfigurationINTEL;
PFN_vkUninitializePerformanceApiINTEL vkUninitializePerformanceApiINTEL;
#endif /* defined(VK_INTEL_performance_query) */
#if defined(VK_KHR_acceleration_structure)
PFN_vkBuildAccelerationStructuresKHR vkBuildAccelerationStructuresKHR;
PFN_vkCmdBuildAccelerationStructuresIndirectKHR vkCmdBuildAccelerationStructuresIndirectKHR;
PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR;
PFN_vkCmdCopyAccelerationStructureKHR vkCmdCopyAccelerationStructureKHR;
PFN_vkCmdCopyAccelerationStructureToMemoryKHR vkCmdCopyAccelerationStructureToMemoryKHR;
PFN_vkCmdCopyMemoryToAccelerationStructureKHR vkCmdCopyMemoryToAccelerationStructureKHR;
PFN_vkCmdWriteAccelerationStructuresPropertiesKHR vkCmdWriteAccelerationStructuresPropertiesKHR;
PFN_vkCopyAccelerationStructureKHR vkCopyAccelerationStructureKHR;
PFN_vkCopyAccelerationStructureToMemoryKHR vkCopyAccelerationStructureToMemoryKHR;
PFN_vkCopyMemoryToAccelerationStructureKHR vkCopyMemoryToAccelerationStructureKHR;
PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR;
PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR;
PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR;
PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR;
PFN_vkGetDeviceAccelerationStructureCompatibilityKHR vkGetDeviceAccelerationStructureCompatibilityKHR;
PFN_vkWriteAccelerationStructuresPropertiesKHR vkWriteAccelerationStructuresPropertiesKHR;
#endif /* defined(VK_KHR_acceleration_structure) */
#if defined(VK_KHR_android_surface)
PFN_vkCreateAndroidSurfaceKHR vkCreateAndroidSurfaceKHR;
#endif /* defined(VK_KHR_android_surface) */
#if defined(VK_KHR_bind_memory2)
PFN_vkBindBufferMemory2KHR vkBindBufferMemory2KHR;
PFN_vkBindImageMemory2KHR vkBindImageMemory2KHR;
#endif /* defined(VK_KHR_bind_memory2) */
#if defined(VK_KHR_buffer_device_address)
PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR;
PFN_vkGetBufferOpaqueCaptureAddressKHR vkGetBufferOpaqueCaptureAddressKHR;
PFN_vkGetDeviceMemoryOpaqueCaptureAddressKHR vkGetDeviceMemoryOpaqueCaptureAddressKHR;
#endif /* defined(VK_KHR_buffer_device_address) */
#if defined(VK_KHR_calibrated_timestamps)
PFN_vkGetCalibratedTimestampsKHR vkGetCalibratedTimestampsKHR;
PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsKHR vkGetPhysicalDeviceCalibrateableTimeDomainsKHR;
#endif /* defined(VK_KHR_calibrated_timestamps) */
#if defined(VK_KHR_cooperative_matrix)
PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR;
#endif /* defined(VK_KHR_cooperative_matrix) */
#if defined(VK_KHR_copy_commands2)
PFN_vkCmdBlitImage2KHR vkCmdBlitImage2KHR;
PFN_vkCmdCopyBuffer2KHR vkCmdCopyBuffer2KHR;
PFN_vkCmdCopyBufferToImage2KHR vkCmdCopyBufferToImage2KHR;
PFN_vkCmdCopyImage2KHR vkCmdCopyImage2KHR;
PFN_vkCmdCopyImageToBuffer2KHR vkCmdCopyImageToBuffer2KHR;
PFN_vkCmdResolveImage2KHR vkCmdResolveImage2KHR;
#endif /* defined(VK_KHR_copy_commands2) */
#if defined(VK_KHR_create_renderpass2)
PFN_vkCmdBeginRenderPass2KHR vkCmdBeginRenderPass2KHR;
PFN_vkCmdEndRenderPass2KHR vkCmdEndRenderPass2KHR;
PFN_vkCmdNextSubpass2KHR vkCmdNextSubpass2KHR;
PFN_vkCreateRenderPass2KHR vkCreateRenderPass2KHR;
#endif /* defined(VK_KHR_create_renderpass2) */
#if defined(VK_KHR_deferred_host_operations)
PFN_vkCreateDeferredOperationKHR vkCreateDeferredOperationKHR;
PFN_vkDeferredOperationJoinKHR vkDeferredOperationJoinKHR;
PFN_vkDestroyDeferredOperationKHR vkDestroyDeferredOperationKHR;
PFN_vkGetDeferredOperationMaxConcurrencyKHR vkGetDeferredOperationMaxConcurrencyKHR;
PFN_vkGetDeferredOperationResultKHR vkGetDeferredOperationResultKHR;
#endif /* defined(VK_KHR_deferred_host_operations) */
#if defined(VK_KHR_descriptor_update_template)
PFN_vkCreateDescriptorUpdateTemplateKHR vkCreateDescriptorUpdateTemplateKHR;
PFN_vkDestroyDescriptorUpdateTemplateKHR vkDestroyDescriptorUpdateTemplateKHR;
PFN_vkUpdateDescriptorSetWithTemplateKHR vkUpdateDescriptorSetWithTemplateKHR;
#endif /* defined(VK_KHR_descriptor_update_template) */
#if defined(VK_KHR_device_group)
PFN_vkCmdDispatchBaseKHR vkCmdDispatchBaseKHR;
PFN_vkCmdSetDeviceMaskKHR vkCmdSetDeviceMaskKHR;
PFN_vkGetDeviceGroupPeerMemoryFeaturesKHR vkGetDeviceGroupPeerMemoryFeaturesKHR;
#endif /* defined(VK_KHR_device_group) */
#if defined(VK_KHR_device_group_creation)
PFN_vkEnumeratePhysicalDeviceGroupsKHR vkEnumeratePhysicalDeviceGroupsKHR;
#endif /* defined(VK_KHR_device_group_creation) */
#if defined(VK_KHR_display)
PFN_vkCreateDisplayModeKHR vkCreateDisplayModeKHR;
PFN_vkCreateDisplayPlaneSurfaceKHR vkCreateDisplayPlaneSurfaceKHR;
PFN_vkGetDisplayModePropertiesKHR vkGetDisplayModePropertiesKHR;
PFN_vkGetDisplayPlaneCapabilitiesKHR vkGetDisplayPlaneCapabilitiesKHR;
PFN_vkGetDisplayPlaneSupportedDisplaysKHR vkGetDisplayPlaneSupportedDisplaysKHR;
PFN_vkGetPhysicalDeviceDisplayPlanePropertiesKHR vkGetPhysicalDeviceDisplayPlanePropertiesKHR;
PFN_vkGetPhysicalDeviceDisplayPropertiesKHR vkGetPhysicalDeviceDisplayPropertiesKHR;
#endif /* defined(VK_KHR_display) */
#if defined(VK_KHR_display_swapchain)
PFN_vkCreateSharedSwapchainsKHR vkCreateSharedSwapchainsKHR;
#endif /* defined(VK_KHR_display_swapchain) */
#if defined(VK_KHR_draw_indirect_count)
PFN_vkCmdDrawIndexedIndirectCountKHR vkCmdDrawIndexedIndirectCountKHR;
PFN_vkCmdDrawIndirectCountKHR vkCmdDrawIndirectCountKHR;
#endif /* defined(VK_KHR_draw_indirect_count) */
#if defined(VK_KHR_dynamic_rendering)
PFN_vkCmdBeginRenderingKHR vkCmdBeginRenderingKHR;
PFN_vkCmdEndRenderingKHR vkCmdEndRenderingKHR;
#endif /* defined(VK_KHR_dynamic_rendering) */
#if defined(VK_KHR_external_fence_capabilities)
PFN_vkGetPhysicalDeviceExternalFencePropertiesKHR vkGetPhysicalDeviceExternalFencePropertiesKHR;
#endif /* defined(VK_KHR_external_fence_capabilities) */
#if defined(VK_KHR_external_fence_fd)
PFN_vkGetFenceFdKHR vkGetFenceFdKHR;
PFN_vkImportFenceFdKHR vkImportFenceFdKHR;
#endif /* defined(VK_KHR_external_fence_fd) */
#if defined(VK_KHR_external_fence_win32)
PFN_vkGetFenceWin32HandleKHR vkGetFenceWin32HandleKHR;
PFN_vkImportFenceWin32HandleKHR vkImportFenceWin32HandleKHR;
#endif /* defined(VK_KHR_external_fence_win32) */
#if defined(VK_KHR_external_memory_capabilities)
PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR vkGetPhysicalDeviceExternalBufferPropertiesKHR;
#endif /* defined(VK_KHR_external_memory_capabilities) */
#if defined(VK_KHR_external_memory_fd)
PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR;
PFN_vkGetMemoryFdPropertiesKHR vkGetMemoryFdPropertiesKHR;
#endif /* defined(VK_KHR_external_memory_fd) */
#if defined(VK_KHR_external_memory_win32)
PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR;
PFN_vkGetMemoryWin32HandlePropertiesKHR vkGetMemoryWin32HandlePropertiesKHR;
#endif /* defined(VK_KHR_external_memory_win32) */
#if defined(VK_KHR_external_semaphore_capabilities)
PFN_vkGetPhysicalDeviceExternalSemaphorePropertiesKHR vkGetPhysicalDeviceExternalSemaphorePropertiesKHR;
#endif /* defined(VK_KHR_external_semaphore_capabilities) */
#if defined(VK_KHR_external_semaphore_fd)
PFN_vkGetSemaphoreFdKHR vkGetSemaphoreFdKHR;
PFN_vkImportSemaphoreFdKHR vkImportSemaphoreFdKHR;
#endif /* defined(VK_KHR_external_semaphore_fd) */
#if defined(VK_KHR_external_semaphore_win32)
PFN_vkGetSemaphoreWin32HandleKHR vkGetSemaphoreWin32HandleKHR;
PFN_vkImportSemaphoreWin32HandleKHR vkImportSemaphoreWin32HandleKHR;
#endif /* defined(VK_KHR_external_semaphore_win32) */
#if defined(VK_KHR_fragment_shading_rate)
PFN_vkCmdSetFragmentShadingRateKHR vkCmdSetFragmentShadingRateKHR;
PFN_vkGetPhysicalDeviceFragmentShadingRatesKHR vkGetPhysicalDeviceFragmentShadingRatesKHR;
#endif /* defined(VK_KHR_fragment_shading_rate) */
#if defined(VK_KHR_get_display_properties2)
PFN_vkGetDisplayModeProperties2KHR vkGetDisplayModeProperties2KHR;
PFN_vkGetDisplayPlaneCapabilities2KHR vkGetDisplayPlaneCapabilities2KHR;
PFN_vkGetPhysicalDeviceDisplayPlaneProperties2KHR vkGetPhysicalDeviceDisplayPlaneProperties2KHR;
PFN_vkGetPhysicalDeviceDisplayProperties2KHR vkGetPhysicalDeviceDisplayProperties2KHR;
#endif /* defined(VK_KHR_get_display_properties2) */
#if defined(VK_KHR_get_memory_requirements2)
PFN_vkGetBufferMemoryRequirements2KHR vkGetBufferMemoryRequirements2KHR;
PFN_vkGetImageMemoryRequirements2KHR vkGetImageMemoryRequirements2KHR;
PFN_vkGetImageSparseMemoryRequirements2KHR vkGetImageSparseMemoryRequirements2KHR;
#endif /* defined(VK_KHR_get_memory_requirements2) */
#if defined(VK_KHR_get_physical_device_properties2)
PFN_vkGetPhysicalDeviceFeatures2KHR vkGetPhysicalDeviceFeatures2KHR;
PFN_vkGetPhysicalDeviceFormatProperties2KHR vkGetPhysicalDeviceFormatProperties2KHR;
PFN_vkGetPhysicalDeviceImageFormatProperties2KHR vkGetPhysicalDeviceImageFormatProperties2KHR;
PFN_vkGetPhysicalDeviceMemoryProperties2KHR vkGetPhysicalDeviceMemoryProperties2KHR;
PFN_vkGetPhysicalDeviceProperties2KHR vkGetPhysicalDeviceProperties2KHR;
PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR vkGetPhysicalDeviceQueueFamilyProperties2KHR;
PFN_vkGetPhysicalDeviceSparseImageFormatProperties2KHR vkGetPhysicalDeviceSparseImageFormatProperties2KHR;
#endif /* defined(VK_KHR_get_physical_device_properties2) */
#if defined(VK_KHR_get_surface_capabilities2)
PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR vkGetPhysicalDeviceSurfaceCapabilities2KHR;
PFN_vkGetPhysicalDeviceSurfaceFormats2KHR vkGetPhysicalDeviceSurfaceFormats2KHR;
#endif /* defined(VK_KHR_get_surface_capabilities2) */
#if defined(VK_KHR_maintenance1)
PFN_vkTrimCommandPoolKHR vkTrimCommandPoolKHR;
#endif /* defined(VK_KHR_maintenance1) */
#if defined(VK_KHR_maintenance3)
PFN_vkGetDescriptorSetLayoutSupportKHR vkGetDescriptorSetLayoutSupportKHR;
#endif /* defined(VK_KHR_maintenance3) */
#if defined(VK_KHR_maintenance4)
PFN_vkGetDeviceBufferMemoryRequirementsKHR vkGetDeviceBufferMemoryRequirementsKHR;
PFN_vkGetDeviceImageMemoryRequirementsKHR vkGetDeviceImageMemoryRequirementsKHR;
PFN_vkGetDeviceImageSparseMemoryRequirementsKHR vkGetDeviceImageSparseMemoryRequirementsKHR;
#endif /* defined(VK_KHR_maintenance4) */
#if defined(VK_KHR_maintenance5)
PFN_vkCmdBindIndexBuffer2KHR vkCmdBindIndexBuffer2KHR;
PFN_vkGetDeviceImageSubresourceLayoutKHR vkGetDeviceImageSubresourceLayoutKHR;
PFN_vkGetImageSubresourceLayout2KHR vkGetImageSubresourceLayout2KHR;
PFN_vkGetRenderingAreaGranularityKHR vkGetRenderingAreaGranularityKHR;
#endif /* defined(VK_KHR_maintenance5) */
#if defined(VK_KHR_maintenance6)
PFN_vkCmdBindDescriptorSets2KHR vkCmdBindDescriptorSets2KHR;
PFN_vkCmdPushConstants2KHR vkCmdPushConstants2KHR;
#endif /* defined(VK_KHR_maintenance6) */
#if defined(VK_KHR_maintenance6) && defined(VK_KHR_push_descriptor)
PFN_vkCmdPushDescriptorSet2KHR vkCmdPushDescriptorSet2KHR;
PFN_vkCmdPushDescriptorSetWithTemplate2KHR vkCmdPushDescriptorSetWithTemplate2KHR;
#endif /* defined(VK_KHR_maintenance6) && defined(VK_KHR_push_descriptor) */
#if defined(VK_KHR_maintenance6) && defined(VK_EXT_descriptor_buffer)
PFN_vkCmdBindDescriptorBufferEmbeddedSamplers2EXT vkCmdBindDescriptorBufferEmbeddedSamplers2EXT;
PFN_vkCmdSetDescriptorBufferOffsets2EXT vkCmdSetDescriptorBufferOffsets2EXT;
#endif /* defined(VK_KHR_maintenance6) && defined(VK_EXT_descriptor_buffer) */
#if defined(VK_KHR_map_memory2)
PFN_vkMapMemory2KHR vkMapMemory2KHR;
PFN_vkUnmapMemory2KHR vkUnmapMemory2KHR;
#endif /* defined(VK_KHR_map_memory2) */
#if defined(VK_KHR_performance_query)
PFN_vkAcquireProfilingLockKHR vkAcquireProfilingLockKHR;
PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR;
PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR;
PFN_vkReleaseProfilingLockKHR vkReleaseProfilingLockKHR;
#endif /* defined(VK_KHR_performance_query) */
#if defined(VK_KHR_pipeline_executable_properties)
PFN_vkGetPipelineExecutableInternalRepresentationsKHR vkGetPipelineExecutableInternalRepresentationsKHR;
PFN_vkGetPipelineExecutablePropertiesKHR vkGetPipelineExecutablePropertiesKHR;
PFN_vkGetPipelineExecutableStatisticsKHR vkGetPipelineExecutableStatisticsKHR;
#endif /* defined(VK_KHR_pipeline_executable_properties) */
#if defined(VK_KHR_present_wait)
PFN_vkWaitForPresentKHR vkWaitForPresentKHR;
#endif /* defined(VK_KHR_present_wait) */
#if defined(VK_KHR_push_descriptor)
PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR;
#endif /* defined(VK_KHR_push_descriptor) */
#if defined(VK_KHR_ray_tracing_maintenance1) && defined(VK_KHR_ray_tracing_pipeline)
PFN_vkCmdTraceRaysIndirect2KHR vkCmdTraceRaysIndirect2KHR;
#endif /* defined(VK_KHR_ray_tracing_maintenance1) && defined(VK_KHR_ray_tracing_pipeline) */
#if defined(VK_KHR_ray_tracing_pipeline)
PFN_vkCmdSetRayTracingPipelineStackSizeKHR vkCmdSetRayTracingPipelineStackSizeKHR;
PFN_vkCmdTraceRaysIndirectKHR vkCmdTraceRaysIndirectKHR;
PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR;
PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR;
PFN_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR vkGetRayTracingCaptureReplayShaderGroupHandlesKHR;
PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR;
PFN_vkGetRayTracingShaderGroupStackSizeKHR vkGetRayTracingShaderGroupStackSizeKHR;
#endif /* defined(VK_KHR_ray_tracing_pipeline) */
#if defined(VK_KHR_sampler_ycbcr_conversion)
PFN_vkCreateSamplerYcbcrConversionKHR vkCreateSamplerYcbcrConversionKHR;
PFN_vkDestroySamplerYcbcrConversionKHR vkDestroySamplerYcbcrConversionKHR;
#endif /* defined(VK_KHR_sampler_ycbcr_conversion) */
#if defined(VK_KHR_shared_presentable_image)
PFN_vkGetSwapchainStatusKHR vkGetSwapchainStatusKHR;
#endif /* defined(VK_KHR_shared_presentable_image) */
#if defined(VK_KHR_surface)
PFN_vkDestroySurfaceKHR vkDestroySurfaceKHR;
PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR vkGetPhysicalDeviceSurfaceCapabilitiesKHR;
PFN_vkGetPhysicalDeviceSurfaceFormatsKHR vkGetPhysicalDeviceSurfaceFormatsKHR;
PFN_vkGetPhysicalDeviceSurfacePresentModesKHR vkGetPhysicalDeviceSurfacePresentModesKHR;
PFN_vkGetPhysicalDeviceSurfaceSupportKHR vkGetPhysicalDeviceSurfaceSupportKHR;
#endif /* defined(VK_KHR_surface) */
#if defined(VK_KHR_swapchain)
PFN_vkAcquireNextImageKHR vkAcquireNextImageKHR;
PFN_vkCreateSwapchainKHR vkCreateSwapchainKHR;
PFN_vkDestroySwapchainKHR vkDestroySwapchainKHR;
PFN_vkGetSwapchainImagesKHR vkGetSwapchainImagesKHR;
PFN_vkQueuePresentKHR vkQueuePresentKHR;
#endif /* defined(VK_KHR_swapchain) */
#if defined(VK_KHR_synchronization2)
PFN_vkCmdPipelineBarrier2KHR vkCmdPipelineBarrier2KHR;
PFN_vkCmdResetEvent2KHR vkCmdResetEvent2KHR;
PFN_vkCmdSetEvent2KHR vkCmdSetEvent2KHR;
PFN_vkCmdWaitEvents2KHR vkCmdWaitEvents2KHR;
PFN_vkCmdWriteTimestamp2KHR vkCmdWriteTimestamp2KHR;
PFN_vkQueueSubmit2KHR vkQueueSubmit2KHR;
#endif /* defined(VK_KHR_synchronization2) */
#if defined(VK_KHR_synchronization2) && defined(VK_AMD_buffer_marker)
PFN_vkCmdWriteBufferMarker2AMD vkCmdWriteBufferMarker2AMD;
#endif /* defined(VK_KHR_synchronization2) && defined(VK_AMD_buffer_marker) */
#if defined(VK_KHR_synchronization2) && defined(VK_NV_device_diagnostic_checkpoints)
PFN_vkGetQueueCheckpointData2NV vkGetQueueCheckpointData2NV;
#endif /* defined(VK_KHR_synchronization2) && defined(VK_NV_device_diagnostic_checkpoints) */
#if defined(VK_KHR_timeline_semaphore)
PFN_vkGetSemaphoreCounterValueKHR vkGetSemaphoreCounterValueKHR;
PFN_vkSignalSemaphoreKHR vkSignalSemaphoreKHR;
PFN_vkWaitSemaphoresKHR vkWaitSemaphoresKHR;
#endif /* defined(VK_KHR_timeline_semaphore) */
#if defined(VK_KHR_video_decode_queue)
PFN_vkCmdDecodeVideoKHR vkCmdDecodeVideoKHR;
#endif /* defined(VK_KHR_video_decode_queue) */
#if defined(VK_KHR_video_encode_queue)
PFN_vkCmdEncodeVideoKHR vkCmdEncodeVideoKHR;
PFN_vkGetEncodedVideoSessionParametersKHR vkGetEncodedVideoSessionParametersKHR;
PFN_vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR;
#endif /* defined(VK_KHR_video_encode_queue) */
#if defined(VK_KHR_video_queue)
PFN_vkBindVideoSessionMemoryKHR vkBindVideoSessionMemoryKHR;
PFN_vkCmdBeginVideoCodingKHR vkCmdBeginVideoCodingKHR;
PFN_vkCmdControlVideoCodingKHR vkCmdControlVideoCodingKHR;
PFN_vkCmdEndVideoCodingKHR vkCmdEndVideoCodingKHR;
PFN_vkCreateVideoSessionKHR vkCreateVideoSessionKHR;
PFN_vkCreateVideoSessionParametersKHR vkCreateVideoSessionParametersKHR;
PFN_vkDestroyVideoSessionKHR vkDestroyVideoSessionKHR;
PFN_vkDestroyVideoSessionParametersKHR vkDestroyVideoSessionParametersKHR;
PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR vkGetPhysicalDeviceVideoCapabilitiesKHR;
PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR vkGetPhysicalDeviceVideoFormatPropertiesKHR;
PFN_vkGetVideoSessionMemoryRequirementsKHR vkGetVideoSessionMemoryRequirementsKHR;
PFN_vkUpdateVideoSessionParametersKHR vkUpdateVideoSessionParametersKHR;
#endif /* defined(VK_KHR_video_queue) */
#if defined(VK_KHR_wayland_surface)
PFN_vkCreateWaylandSurfaceKHR vkCreateWaylandSurfaceKHR;
PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR vkGetPhysicalDeviceWaylandPresentationSupportKHR;
#endif /* defined(VK_KHR_wayland_surface) */
#if defined(VK_KHR_win32_surface)
PFN_vkCreateWin32SurfaceKHR vkCreateWin32SurfaceKHR;
PFN_vkGetPhysicalDeviceWin32PresentationSupportKHR vkGetPhysicalDeviceWin32PresentationSupportKHR;
#endif /* defined(VK_KHR_win32_surface) */
#if defined(VK_KHR_xcb_surface)
PFN_vkCreateXcbSurfaceKHR vkCreateXcbSurfaceKHR;
PFN_vkGetPhysicalDeviceXcbPresentationSupportKHR vkGetPhysicalDeviceXcbPresentationSupportKHR;
#endif /* defined(VK_KHR_xcb_surface) */
#if defined(VK_KHR_xlib_surface)
PFN_vkCreateXlibSurfaceKHR vkCreateXlibSurfaceKHR;
PFN_vkGetPhysicalDeviceXlibPresentationSupportKHR vkGetPhysicalDeviceXlibPresentationSupportKHR;
#endif /* defined(VK_KHR_xlib_surface) */
#if defined(VK_MVK_ios_surface)
PFN_vkCreateIOSSurfaceMVK vkCreateIOSSurfaceMVK;
#endif /* defined(VK_MVK_ios_surface) */
#if defined(VK_MVK_macos_surface)
PFN_vkCreateMacOSSurfaceMVK vkCreateMacOSSurfaceMVK;
#endif /* defined(VK_MVK_macos_surface) */
#if defined(VK_NN_vi_surface)
PFN_vkCreateViSurfaceNN vkCreateViSurfaceNN;
#endif /* defined(VK_NN_vi_surface) */
#if defined(VK_NVX_binary_import)
PFN_vkCmdCuLaunchKernelNVX vkCmdCuLaunchKernelNVX;
PFN_vkCreateCuFunctionNVX vkCreateCuFunctionNVX;
PFN_vkCreateCuModuleNVX vkCreateCuModuleNVX;
PFN_vkDestroyCuFunctionNVX vkDestroyCuFunctionNVX;
PFN_vkDestroyCuModuleNVX vkDestroyCuModuleNVX;
#endif /* defined(VK_NVX_binary_import) */
#if defined(VK_NVX_image_view_handle)
PFN_vkGetImageViewAddressNVX vkGetImageViewAddressNVX;
PFN_vkGetImageViewHandleNVX vkGetImageViewHandleNVX;
#endif /* defined(VK_NVX_image_view_handle) */
#if defined(VK_NV_acquire_winrt_display)
PFN_vkAcquireWinrtDisplayNV vkAcquireWinrtDisplayNV;
PFN_vkGetWinrtDisplayNV vkGetWinrtDisplayNV;
#endif /* defined(VK_NV_acquire_winrt_display) */
#if defined(VK_NV_clip_space_w_scaling)
PFN_vkCmdSetViewportWScalingNV vkCmdSetViewportWScalingNV;
#endif /* defined(VK_NV_clip_space_w_scaling) */
#if defined(VK_NV_cooperative_matrix)
PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV vkGetPhysicalDeviceCooperativeMatrixPropertiesNV;
#endif /* defined(VK_NV_cooperative_matrix) */
#if defined(VK_NV_copy_memory_indirect)
PFN_vkCmdCopyMemoryIndirectNV vkCmdCopyMemoryIndirectNV;
PFN_vkCmdCopyMemoryToImageIndirectNV vkCmdCopyMemoryToImageIndirectNV;
#endif /* defined(VK_NV_copy_memory_indirect) */
#if defined(VK_NV_coverage_reduction_mode)
PFN_vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV;
#endif /* defined(VK_NV_coverage_reduction_mode) */
#if defined(VK_NV_cuda_kernel_launch)
PFN_vkCmdCudaLaunchKernelNV vkCmdCudaLaunchKernelNV;
PFN_vkCreateCudaFunctionNV vkCreateCudaFunctionNV;
PFN_vkCreateCudaModuleNV vkCreateCudaModuleNV;
PFN_vkDestroyCudaFunctionNV vkDestroyCudaFunctionNV;
PFN_vkDestroyCudaModuleNV vkDestroyCudaModuleNV;
PFN_vkGetCudaModuleCacheNV vkGetCudaModuleCacheNV;
#endif /* defined(VK_NV_cuda_kernel_launch) */
#if defined(VK_NV_device_diagnostic_checkpoints)
PFN_vkCmdSetCheckpointNV vkCmdSetCheckpointNV;
PFN_vkGetQueueCheckpointDataNV vkGetQueueCheckpointDataNV;
#endif /* defined(VK_NV_device_diagnostic_checkpoints) */
#if defined(VK_NV_device_generated_commands)
PFN_vkCmdBindPipelineShaderGroupNV vkCmdBindPipelineShaderGroupNV;
PFN_vkCmdExecuteGeneratedCommandsNV vkCmdExecuteGeneratedCommandsNV;
PFN_vkCmdPreprocessGeneratedCommandsNV vkCmdPreprocessGeneratedCommandsNV;
PFN_vkCreateIndirectCommandsLayoutNV vkCreateIndirectCommandsLayoutNV;
PFN_vkDestroyIndirectCommandsLayoutNV vkDestroyIndirectCommandsLayoutNV;
PFN_vkGetGeneratedCommandsMemoryRequirementsNV vkGetGeneratedCommandsMemoryRequirementsNV;
#endif /* defined(VK_NV_device_generated_commands) */
#if defined(VK_NV_device_generated_commands_compute)
PFN_vkCmdUpdatePipelineIndirectBufferNV vkCmdUpdatePipelineIndirectBufferNV;
PFN_vkGetPipelineIndirectDeviceAddressNV vkGetPipelineIndirectDeviceAddressNV;
PFN_vkGetPipelineIndirectMemoryRequirementsNV vkGetPipelineIndirectMemoryRequirementsNV;
#endif /* defined(VK_NV_device_generated_commands_compute) */
#if defined(VK_NV_external_memory_capabilities)
PFN_vkGetPhysicalDeviceExternalImageFormatPropertiesNV vkGetPhysicalDeviceExternalImageFormatPropertiesNV;
#endif /* defined(VK_NV_external_memory_capabilities) */
#if defined(VK_NV_external_memory_rdma)
PFN_vkGetMemoryRemoteAddressNV vkGetMemoryRemoteAddressNV;
#endif /* defined(VK_NV_external_memory_rdma) */
#if defined(VK_NV_external_memory_win32)
PFN_vkGetMemoryWin32HandleNV vkGetMemoryWin32HandleNV;
#endif /* defined(VK_NV_external_memory_win32) */
#if defined(VK_NV_fragment_shading_rate_enums)
PFN_vkCmdSetFragmentShadingRateEnumNV vkCmdSetFragmentShadingRateEnumNV;
#endif /* defined(VK_NV_fragment_shading_rate_enums) */
#if defined(VK_NV_low_latency2)
PFN_vkGetLatencyTimingsNV vkGetLatencyTimingsNV;
PFN_vkLatencySleepNV vkLatencySleepNV;
PFN_vkQueueNotifyOutOfBandNV vkQueueNotifyOutOfBandNV;
PFN_vkSetLatencyMarkerNV vkSetLatencyMarkerNV;
PFN_vkSetLatencySleepModeNV vkSetLatencySleepModeNV;
#endif /* defined(VK_NV_low_latency2) */
#if defined(VK_NV_memory_decompression)
PFN_vkCmdDecompressMemoryIndirectCountNV vkCmdDecompressMemoryIndirectCountNV;
PFN_vkCmdDecompressMemoryNV vkCmdDecompressMemoryNV;
#endif /* defined(VK_NV_memory_decompression) */
#if defined(VK_NV_mesh_shader)
PFN_vkCmdDrawMeshTasksIndirectCountNV vkCmdDrawMeshTasksIndirectCountNV;
PFN_vkCmdDrawMeshTasksIndirectNV vkCmdDrawMeshTasksIndirectNV;
PFN_vkCmdDrawMeshTasksNV vkCmdDrawMeshTasksNV;
#endif /* defined(VK_NV_mesh_shader) */
#if defined(VK_NV_optical_flow)
PFN_vkBindOpticalFlowSessionImageNV vkBindOpticalFlowSessionImageNV;
PFN_vkCmdOpticalFlowExecuteNV vkCmdOpticalFlowExecuteNV;
PFN_vkCreateOpticalFlowSessionNV vkCreateOpticalFlowSessionNV;
PFN_vkDestroyOpticalFlowSessionNV vkDestroyOpticalFlowSessionNV;
PFN_vkGetPhysicalDeviceOpticalFlowImageFormatsNV vkGetPhysicalDeviceOpticalFlowImageFormatsNV;
#endif /* defined(VK_NV_optical_flow) */
#if defined(VK_NV_ray_tracing)
PFN_vkBindAccelerationStructureMemoryNV vkBindAccelerationStructureMemoryNV;
PFN_vkCmdBuildAccelerationStructureNV vkCmdBuildAccelerationStructureNV;
PFN_vkCmdCopyAccelerationStructureNV vkCmdCopyAccelerationStructureNV;
PFN_vkCmdTraceRaysNV vkCmdTraceRaysNV;
PFN_vkCmdWriteAccelerationStructuresPropertiesNV vkCmdWriteAccelerationStructuresPropertiesNV;
PFN_vkCompileDeferredNV vkCompileDeferredNV;
PFN_vkCreateAccelerationStructureNV vkCreateAccelerationStructureNV;
PFN_vkCreateRayTracingPipelinesNV vkCreateRayTracingPipelinesNV;
PFN_vkDestroyAccelerationStructureNV vkDestroyAccelerationStructureNV;
PFN_vkGetAccelerationStructureHandleNV vkGetAccelerationStructureHandleNV;
PFN_vkGetAccelerationStructureMemoryRequirementsNV vkGetAccelerationStructureMemoryRequirementsNV;
PFN_vkGetRayTracingShaderGroupHandlesNV vkGetRayTracingShaderGroupHandlesNV;
#endif /* defined(VK_NV_ray_tracing) */
#if defined(VK_NV_scissor_exclusive) && VK_NV_SCISSOR_EXCLUSIVE_SPEC_VERSION >= 2
PFN_vkCmdSetExclusiveScissorEnableNV vkCmdSetExclusiveScissorEnableNV;
#endif /* defined(VK_NV_scissor_exclusive) && VK_NV_SCISSOR_EXCLUSIVE_SPEC_VERSION >= 2 */
#if defined(VK_NV_scissor_exclusive)
PFN_vkCmdSetExclusiveScissorNV vkCmdSetExclusiveScissorNV;
#endif /* defined(VK_NV_scissor_exclusive) */
#if defined(VK_NV_shading_rate_image)
PFN_vkCmdBindShadingRateImageNV vkCmdBindShadingRateImageNV;
PFN_vkCmdSetCoarseSampleOrderNV vkCmdSetCoarseSampleOrderNV;
PFN_vkCmdSetViewportShadingRatePaletteNV vkCmdSetViewportShadingRatePaletteNV;
#endif /* defined(VK_NV_shading_rate_image) */
#if defined(VK_QCOM_tile_properties)
PFN_vkGetDynamicRenderingTilePropertiesQCOM vkGetDynamicRenderingTilePropertiesQCOM;
PFN_vkGetFramebufferTilePropertiesQCOM vkGetFramebufferTilePropertiesQCOM;
#endif /* defined(VK_QCOM_tile_properties) */
#if defined(VK_QNX_external_memory_screen_buffer)
PFN_vkGetScreenBufferPropertiesQNX vkGetScreenBufferPropertiesQNX;
#endif /* defined(VK_QNX_external_memory_screen_buffer) */
#if defined(VK_QNX_screen_surface)
PFN_vkCreateScreenSurfaceQNX vkCreateScreenSurfaceQNX;
PFN_vkGetPhysicalDeviceScreenPresentationSupportQNX vkGetPhysicalDeviceScreenPresentationSupportQNX;
#endif /* defined(VK_QNX_screen_surface) */
#if defined(VK_VALVE_descriptor_set_host_mapping)
PFN_vkGetDescriptorSetHostMappingVALVE vkGetDescriptorSetHostMappingVALVE;
PFN_vkGetDescriptorSetLayoutHostMappingInfoVALVE vkGetDescriptorSetLayoutHostMappingInfoVALVE;
#endif /* defined(VK_VALVE_descriptor_set_host_mapping) */
#if (defined(VK_EXT_extended_dynamic_state)) || (defined(VK_EXT_shader_object))
PFN_vkCmdBindVertexBuffers2EXT vkCmdBindVertexBuffers2EXT;
PFN_vkCmdSetCullModeEXT vkCmdSetCullModeEXT;
PFN_vkCmdSetDepthBoundsTestEnableEXT vkCmdSetDepthBoundsTestEnableEXT;
PFN_vkCmdSetDepthCompareOpEXT vkCmdSetDepthCompareOpEXT;
PFN_vkCmdSetDepthTestEnableEXT vkCmdSetDepthTestEnableEXT;
PFN_vkCmdSetDepthWriteEnableEXT vkCmdSetDepthWriteEnableEXT;
PFN_vkCmdSetFrontFaceEXT vkCmdSetFrontFaceEXT;
PFN_vkCmdSetPrimitiveTopologyEXT vkCmdSetPrimitiveTopologyEXT;
PFN_vkCmdSetScissorWithCountEXT vkCmdSetScissorWithCountEXT;
PFN_vkCmdSetStencilOpEXT vkCmdSetStencilOpEXT;
PFN_vkCmdSetStencilTestEnableEXT vkCmdSetStencilTestEnableEXT;
PFN_vkCmdSetViewportWithCountEXT vkCmdSetViewportWithCountEXT;
#endif /* (defined(VK_EXT_extended_dynamic_state)) || (defined(VK_EXT_shader_object)) */
#if (defined(VK_EXT_extended_dynamic_state2)) || (defined(VK_EXT_shader_object))
PFN_vkCmdSetDepthBiasEnableEXT vkCmdSetDepthBiasEnableEXT;
PFN_vkCmdSetLogicOpEXT vkCmdSetLogicOpEXT;
PFN_vkCmdSetPatchControlPointsEXT vkCmdSetPatchControlPointsEXT;
PFN_vkCmdSetPrimitiveRestartEnableEXT vkCmdSetPrimitiveRestartEnableEXT;
PFN_vkCmdSetRasterizerDiscardEnableEXT vkCmdSetRasterizerDiscardEnableEXT;
#endif /* (defined(VK_EXT_extended_dynamic_state2)) || (defined(VK_EXT_shader_object)) */
#if (defined(VK_EXT_extended_dynamic_state3)) || (defined(VK_EXT_shader_object))
PFN_vkCmdSetAlphaToCoverageEnableEXT vkCmdSetAlphaToCoverageEnableEXT;
PFN_vkCmdSetAlphaToOneEnableEXT vkCmdSetAlphaToOneEnableEXT;
PFN_vkCmdSetColorBlendAdvancedEXT vkCmdSetColorBlendAdvancedEXT;
PFN_vkCmdSetColorBlendEnableEXT vkCmdSetColorBlendEnableEXT;
PFN_vkCmdSetColorBlendEquationEXT vkCmdSetColorBlendEquationEXT;
PFN_vkCmdSetColorWriteMaskEXT vkCmdSetColorWriteMaskEXT;
PFN_vkCmdSetConservativeRasterizationModeEXT vkCmdSetConservativeRasterizationModeEXT;
PFN_vkCmdSetDepthClampEnableEXT vkCmdSetDepthClampEnableEXT;
PFN_vkCmdSetDepthClipEnableEXT vkCmdSetDepthClipEnableEXT;
PFN_vkCmdSetDepthClipNegativeOneToOneEXT vkCmdSetDepthClipNegativeOneToOneEXT;
PFN_vkCmdSetExtraPrimitiveOverestimationSizeEXT vkCmdSetExtraPrimitiveOverestimationSizeEXT;
PFN_vkCmdSetLineRasterizationModeEXT vkCmdSetLineRasterizationModeEXT;
PFN_vkCmdSetLineStippleEnableEXT vkCmdSetLineStippleEnableEXT;
PFN_vkCmdSetLogicOpEnableEXT vkCmdSetLogicOpEnableEXT;
PFN_vkCmdSetPolygonModeEXT vkCmdSetPolygonModeEXT;
PFN_vkCmdSetProvokingVertexModeEXT vkCmdSetProvokingVertexModeEXT;
PFN_vkCmdSetRasterizationSamplesEXT vkCmdSetRasterizationSamplesEXT;
PFN_vkCmdSetRasterizationStreamEXT vkCmdSetRasterizationStreamEXT;
PFN_vkCmdSetSampleLocationsEnableEXT vkCmdSetSampleLocationsEnableEXT;
PFN_vkCmdSetSampleMaskEXT vkCmdSetSampleMaskEXT;
PFN_vkCmdSetTessellationDomainOriginEXT vkCmdSetTessellationDomainOriginEXT;
#endif /* (defined(VK_EXT_extended_dynamic_state3)) || (defined(VK_EXT_shader_object)) */
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_clip_space_w_scaling)) || (defined(VK_EXT_shader_object) && defined(VK_NV_clip_space_w_scaling))
PFN_vkCmdSetViewportWScalingEnableNV vkCmdSetViewportWScalingEnableNV;
#endif /* (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_clip_space_w_scaling)) || (defined(VK_EXT_shader_object) && defined(VK_NV_clip_space_w_scaling)) */
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_viewport_swizzle)) || (defined(VK_EXT_shader_object) && defined(VK_NV_viewport_swizzle))
PFN_vkCmdSetViewportSwizzleNV vkCmdSetViewportSwizzleNV;
#endif /* (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_viewport_swizzle)) || (defined(VK_EXT_shader_object) && defined(VK_NV_viewport_swizzle)) */
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_fragment_coverage_to_color)) || (defined(VK_EXT_shader_object) && defined(VK_NV_fragment_coverage_to_color))
PFN_vkCmdSetCoverageToColorEnableNV vkCmdSetCoverageToColorEnableNV;
PFN_vkCmdSetCoverageToColorLocationNV vkCmdSetCoverageToColorLocationNV;
#endif /* (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_fragment_coverage_to_color)) || (defined(VK_EXT_shader_object) && defined(VK_NV_fragment_coverage_to_color)) */
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_framebuffer_mixed_samples)) || (defined(VK_EXT_shader_object) && defined(VK_NV_framebuffer_mixed_samples))
PFN_vkCmdSetCoverageModulationModeNV vkCmdSetCoverageModulationModeNV;
PFN_vkCmdSetCoverageModulationTableEnableNV vkCmdSetCoverageModulationTableEnableNV;
PFN_vkCmdSetCoverageModulationTableNV vkCmdSetCoverageModulationTableNV;
#endif /* (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_framebuffer_mixed_samples)) || (defined(VK_EXT_shader_object) && defined(VK_NV_framebuffer_mixed_samples)) */
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_shading_rate_image)) || (defined(VK_EXT_shader_object) && defined(VK_NV_shading_rate_image))
PFN_vkCmdSetShadingRateImageEnableNV vkCmdSetShadingRateImageEnableNV;
#endif /* (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_shading_rate_image)) || (defined(VK_EXT_shader_object) && defined(VK_NV_shading_rate_image)) */
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_representative_fragment_test)) || (defined(VK_EXT_shader_object) && defined(VK_NV_representative_fragment_test))
PFN_vkCmdSetRepresentativeFragmentTestEnableNV vkCmdSetRepresentativeFragmentTestEnableNV;
#endif /* (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_representative_fragment_test)) || (defined(VK_EXT_shader_object) && defined(VK_NV_representative_fragment_test)) */
#if (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_coverage_reduction_mode)) || (defined(VK_EXT_shader_object) && defined(VK_NV_coverage_reduction_mode))
PFN_vkCmdSetCoverageReductionModeNV vkCmdSetCoverageReductionModeNV;
#endif /* (defined(VK_EXT_extended_dynamic_state3) && defined(VK_NV_coverage_reduction_mode)) || (defined(VK_EXT_shader_object) && defined(VK_NV_coverage_reduction_mode)) */
#if (defined(VK_EXT_full_screen_exclusive) && defined(VK_KHR_device_group)) || (defined(VK_EXT_full_screen_exclusive) && defined(VK_VERSION_1_1))
PFN_vkGetDeviceGroupSurfacePresentModes2EXT vkGetDeviceGroupSurfacePresentModes2EXT;
#endif /* (defined(VK_EXT_full_screen_exclusive) && defined(VK_KHR_device_group)) || (defined(VK_EXT_full_screen_exclusive) && defined(VK_VERSION_1_1)) */
#if (defined(VK_EXT_host_image_copy)) || (defined(VK_EXT_image_compression_control))
PFN_vkGetImageSubresourceLayout2EXT vkGetImageSubresourceLayout2EXT;
#endif /* (defined(VK_EXT_host_image_copy)) || (defined(VK_EXT_image_compression_control)) */
#if (defined(VK_EXT_shader_object)) || (defined(VK_EXT_vertex_input_dynamic_state))
PFN_vkCmdSetVertexInputEXT vkCmdSetVertexInputEXT;
#endif /* (defined(VK_EXT_shader_object)) || (defined(VK_EXT_vertex_input_dynamic_state)) */
#if (defined(VK_KHR_descriptor_update_template) && defined(VK_KHR_push_descriptor)) || (defined(VK_KHR_push_descriptor) && defined(VK_VERSION_1_1)) || (defined(VK_KHR_push_descriptor) && defined(VK_KHR_descriptor_update_template))
PFN_vkCmdPushDescriptorSetWithTemplateKHR vkCmdPushDescriptorSetWithTemplateKHR;
#endif /* (defined(VK_KHR_descriptor_update_template) && defined(VK_KHR_push_descriptor)) || (defined(VK_KHR_push_descriptor) && defined(VK_VERSION_1_1)) || (defined(VK_KHR_push_descriptor) && defined(VK_KHR_descriptor_update_template)) */
#if (defined(VK_KHR_device_group) && defined(VK_KHR_surface)) || (defined(VK_KHR_swapchain) && defined(VK_VERSION_1_1))
PFN_vkGetDeviceGroupPresentCapabilitiesKHR vkGetDeviceGroupPresentCapabilitiesKHR;
PFN_vkGetDeviceGroupSurfacePresentModesKHR vkGetDeviceGroupSurfacePresentModesKHR;
PFN_vkGetPhysicalDevicePresentRectanglesKHR vkGetPhysicalDevicePresentRectanglesKHR;
#endif /* (defined(VK_KHR_device_group) && defined(VK_KHR_surface)) || (defined(VK_KHR_swapchain) && defined(VK_VERSION_1_1)) */
#if (defined(VK_KHR_device_group) && defined(VK_KHR_swapchain)) || (defined(VK_KHR_swapchain) && defined(VK_VERSION_1_1))
PFN_vkAcquireNextImage2KHR vkAcquireNextImage2KHR;
#endif /* (defined(VK_KHR_device_group) && defined(VK_KHR_swapchain)) || (defined(VK_KHR_swapchain) && defined(VK_VERSION_1_1)) */
/* VOLK_GENERATE_PROTOTYPES_C */

#ifdef __GNUC__
#	pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif
/* clang-format on */
