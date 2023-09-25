/**************************************************************************/
/*  vulkan_context.h                                                      */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef VULKAN_CONTEXT_H
#define VULKAN_CONTEXT_H

#include "core/error/error_list.h"
#include "core/os/mutex.h"
#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "core/templates/rb_map.h"
#include "core/templates/rid_owner.h"
#include "servers/display_server.h"
#include "servers/rendering/rendering_device.h"

#ifdef USE_VOLK
#include <volk.h>
#else
#include <vulkan/vulkan.h>
#endif

#include "vulkan_hooks.h"

class VulkanContext {
public:
	struct SubgroupCapabilities {
		uint32_t size;
		uint32_t min_size;
		uint32_t max_size;
		VkShaderStageFlags supportedStages;
		VkSubgroupFeatureFlags supportedOperations;
		VkBool32 quadOperationsInAllStages;
		bool size_control_is_supported;

		uint32_t supported_stages_flags_rd() const;
		String supported_stages_desc() const;
		uint32_t supported_operations_flags_rd() const;
		String supported_operations_desc() const;
	};

	struct MultiviewCapabilities {
		bool is_supported;
		bool geometry_shader_is_supported;
		bool tessellation_shader_is_supported;
		uint32_t max_view_count;
		uint32_t max_instance_count;
	};

	struct VRSCapabilities {
		bool pipeline_vrs_supported; // We can specify our fragment rate on a pipeline level.
		bool primitive_vrs_supported; // We can specify our fragment rate on each drawcall.
		bool attachment_vrs_supported; // We can provide a density map attachment on our framebuffer.

		Size2i min_texel_size;
		Size2i max_texel_size;

		Size2i texel_size; // The texel size we'll use
	};

	struct ShaderCapabilities {
		bool shader_float16_is_supported;
		bool shader_int8_is_supported;
	};

	struct StorageBufferCapabilities {
		bool storage_buffer_16_bit_access_is_supported;
		bool uniform_and_storage_buffer_16_bit_access_is_supported;
		bool storage_push_constant_16_is_supported;
		bool storage_input_output_16;
	};

private:
	enum {
		MAX_EXTENSIONS = 128,
		MAX_LAYERS = 64,
		FRAME_LAG = 2
	};

	static VulkanHooks *vulkan_hooks;
	VkInstance inst = VK_NULL_HANDLE;
	VkPhysicalDevice gpu = VK_NULL_HANDLE;
	VkPhysicalDeviceProperties gpu_props;
	uint32_t queue_family_count = 0;
	VkQueueFamilyProperties *queue_props = nullptr;
	VkDevice device = VK_NULL_HANDLE;
	bool device_initialized = false;
	bool inst_initialized = false;

	uint32_t instance_api_version = VK_API_VERSION_1_0;
	SubgroupCapabilities subgroup_capabilities;
	MultiviewCapabilities multiview_capabilities;
	VRSCapabilities vrs_capabilities;
	ShaderCapabilities shader_capabilities;
	StorageBufferCapabilities storage_buffer_capabilities;
	bool pipeline_cache_control_support = false;

	String device_vendor;
	String device_name;
	VkPhysicalDeviceType device_type;
	String pipeline_cache_id;
	uint32_t device_api_version = 0;

	bool buffers_prepared = false;

	// Present queue.
	bool queues_initialized = false;
	uint32_t graphics_queue_family_index = UINT32_MAX;
	uint32_t present_queue_family_index = UINT32_MAX;
	bool separate_present_queue = false;
	VkQueue graphics_queue = VK_NULL_HANDLE;
	VkQueue present_queue = VK_NULL_HANDLE;
	VkColorSpaceKHR color_space;
	VkFormat format;
	VkSemaphore draw_complete_semaphores[FRAME_LAG];
	VkSemaphore image_ownership_semaphores[FRAME_LAG];
	int frame_index = 0;
	VkFence fences[FRAME_LAG];
	VkPhysicalDeviceMemoryProperties memory_properties;
	VkPhysicalDeviceFeatures physical_device_features;

	typedef struct {
		VkImage image;
		VkCommandBuffer graphics_to_present_cmd;
		VkImageView view;
		VkFramebuffer framebuffer;
	} SwapchainImageResources;

	struct Window {
		VkSurfaceKHR surface = VK_NULL_HANDLE;
		VkSwapchainKHR swapchain = VK_NULL_HANDLE;
		SwapchainImageResources *swapchain_image_resources = VK_NULL_HANDLE;
		VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
		VkSemaphore image_acquired_semaphores[FRAME_LAG];
		bool semaphore_acquired = false;
		uint32_t current_buffer = 0;
		int width = 0;
		int height = 0;
		DisplayServer::VSyncMode vsync_mode = DisplayServer::VSYNC_ENABLED;
		VkCommandPool present_cmd_pool = VK_NULL_HANDLE; // For separate present queue.
		VkRenderPass render_pass = VK_NULL_HANDLE;
	};

	struct LocalDevice {
		bool waiting = false;
		VkDevice device = VK_NULL_HANDLE;
		VkQueue queue = VK_NULL_HANDLE;
	};

	RID_Owner<LocalDevice, true> local_device_owner;

	HashMap<DisplayServer::WindowID, Window> windows;
	uint32_t swapchainImageCount = 0;

	// Commands.

	bool prepared = false;

	Vector<VkCommandBuffer> command_buffer_queue;
	int command_buffer_count = 1;

	// Extensions.
	static bool instance_extensions_initialized;
	static HashMap<CharString, bool> requested_instance_extensions;
	HashSet<CharString> enabled_instance_extension_names;

	static bool device_extensions_initialized;
	static HashMap<CharString, bool> requested_device_extensions;
	HashSet<CharString> enabled_device_extension_names;
	bool VK_KHR_incremental_present_enabled = true;
	bool VK_GOOGLE_display_timing_enabled = true;

	PFN_vkCreateDebugUtilsMessengerEXT CreateDebugUtilsMessengerEXT = nullptr;
	PFN_vkDestroyDebugUtilsMessengerEXT DestroyDebugUtilsMessengerEXT = nullptr;
	PFN_vkSubmitDebugUtilsMessageEXT SubmitDebugUtilsMessageEXT = nullptr;
	PFN_vkCmdBeginDebugUtilsLabelEXT CmdBeginDebugUtilsLabelEXT = nullptr;
	PFN_vkCmdEndDebugUtilsLabelEXT CmdEndDebugUtilsLabelEXT = nullptr;
	PFN_vkCmdInsertDebugUtilsLabelEXT CmdInsertDebugUtilsLabelEXT = nullptr;
	PFN_vkSetDebugUtilsObjectNameEXT SetDebugUtilsObjectNameEXT = nullptr;
	PFN_vkCreateDebugReportCallbackEXT CreateDebugReportCallbackEXT = nullptr;
	PFN_vkDebugReportMessageEXT DebugReportMessageEXT = nullptr;
	PFN_vkDestroyDebugReportCallbackEXT DestroyDebugReportCallbackEXT = nullptr;
	PFN_vkGetPhysicalDeviceSurfaceSupportKHR fpGetPhysicalDeviceSurfaceSupportKHR = nullptr;
	PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR fpGetPhysicalDeviceSurfaceCapabilitiesKHR = nullptr;
	PFN_vkGetPhysicalDeviceSurfaceFormatsKHR fpGetPhysicalDeviceSurfaceFormatsKHR = nullptr;
	PFN_vkGetPhysicalDeviceSurfacePresentModesKHR fpGetPhysicalDeviceSurfacePresentModesKHR = nullptr;
	PFN_vkCreateSwapchainKHR fpCreateSwapchainKHR = nullptr;
	PFN_vkDestroySwapchainKHR fpDestroySwapchainKHR = nullptr;
	PFN_vkGetSwapchainImagesKHR fpGetSwapchainImagesKHR = nullptr;
	PFN_vkAcquireNextImageKHR fpAcquireNextImageKHR = nullptr;
	PFN_vkQueuePresentKHR fpQueuePresentKHR = nullptr;
	PFN_vkGetRefreshCycleDurationGOOGLE fpGetRefreshCycleDurationGOOGLE = nullptr;
	PFN_vkGetPastPresentationTimingGOOGLE fpGetPastPresentationTimingGOOGLE = nullptr;
	PFN_vkCreateRenderPass2KHR fpCreateRenderPass2KHR = nullptr;

	VkDebugUtilsMessengerEXT dbg_messenger = VK_NULL_HANDLE;
	VkDebugReportCallbackEXT dbg_debug_report = VK_NULL_HANDLE;

	Error _obtain_vulkan_version();
	Error _initialize_instance_extensions();
	Error _initialize_device_extensions();
	Error _check_capabilities();

	VkBool32 _check_layers(uint32_t check_count, const char *const *check_names, uint32_t layer_count, VkLayerProperties *layers);
	static VKAPI_ATTR VkBool32 VKAPI_CALL _debug_messenger_callback(
			VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
			VkDebugUtilsMessageTypeFlagsEXT messageType,
			const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
			void *pUserData);

	static VKAPI_ATTR VkBool32 VKAPI_CALL _debug_report_callback(
			VkDebugReportFlagsEXT flags,
			VkDebugReportObjectTypeEXT objectType,
			uint64_t object,
			size_t location,
			int32_t messageCode,
			const char *pLayerPrefix,
			const char *pMessage,
			void *pUserData);

	Error _create_instance();

	Error _create_physical_device(VkSurfaceKHR p_surface);

	Error _initialize_queues(VkSurfaceKHR p_surface);

	Error _create_device();

	Error _clean_up_swap_chain(Window *window);

	Error _update_swap_chain(Window *window);

	Error _create_swap_chain();
	Error _create_semaphores();

	Vector<VkAttachmentReference> _convert_VkAttachmentReference2(uint32_t p_count, const VkAttachmentReference2 *p_refs);

protected:
	virtual const char *_get_platform_surface_extension() const = 0;

	virtual Error _window_create(DisplayServer::WindowID p_window_id, DisplayServer::VSyncMode p_vsync_mode, VkSurfaceKHR p_surface, int p_width, int p_height);

	virtual bool _use_validation_layers();

	Error _get_preferred_validation_layers(uint32_t *count, const char *const **names);

	virtual VkExtent2D _compute_swapchain_extent(const VkSurfaceCapabilitiesKHR &p_surf_capabilities, int *p_window_width, int *p_window_height) const;

public:
	// Extension calls.
	bool supports_renderpass2() const { return is_device_extension_enabled(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME); }
	VkResult vkCreateRenderPass2KHR(VkDevice p_device, const VkRenderPassCreateInfo2 *p_create_info, const VkAllocationCallbacks *p_allocator, VkRenderPass *p_render_pass);

	uint32_t get_vulkan_major() const { return VK_API_VERSION_MAJOR(device_api_version); };
	uint32_t get_vulkan_minor() const { return VK_API_VERSION_MINOR(device_api_version); };
	const SubgroupCapabilities &get_subgroup_capabilities() const { return subgroup_capabilities; };
	const MultiviewCapabilities &get_multiview_capabilities() const { return multiview_capabilities; };
	const VRSCapabilities &get_vrs_capabilities() const { return vrs_capabilities; };
	const ShaderCapabilities &get_shader_capabilities() const { return shader_capabilities; };
	const StorageBufferCapabilities &get_storage_buffer_capabilities() const { return storage_buffer_capabilities; };
	const VkPhysicalDeviceFeatures &get_physical_device_features() const { return physical_device_features; };
	bool get_pipeline_cache_control_support() const { return pipeline_cache_control_support; };

	VkDevice get_device();
	VkPhysicalDevice get_physical_device();
	VkInstance get_instance() { return inst; }
	int get_swapchain_image_count() const;
	VkQueue get_graphics_queue() const;
	uint32_t get_graphics_queue_family_index() const;

	static void set_vulkan_hooks(VulkanHooks *p_vulkan_hooks) { vulkan_hooks = p_vulkan_hooks; };

	static void register_requested_instance_extension(const CharString &extension_name, bool p_required);
	bool is_instance_extension_enabled(const CharString &extension_name) const {
		return enabled_instance_extension_names.has(extension_name);
	}

	static void register_requested_device_extension(const CharString &extension_name, bool p_required);
	bool is_device_extension_enabled(const CharString &extension_name) const {
		return enabled_device_extension_names.has(extension_name);
	}

	void window_resize(DisplayServer::WindowID p_window_id, int p_width, int p_height);
	int window_get_width(DisplayServer::WindowID p_window = 0);
	int window_get_height(DisplayServer::WindowID p_window = 0);
	bool window_is_valid_swapchain(DisplayServer::WindowID p_window = 0);
	void window_destroy(DisplayServer::WindowID p_window_id);
	VkFramebuffer window_get_framebuffer(DisplayServer::WindowID p_window = 0);
	VkRenderPass window_get_render_pass(DisplayServer::WindowID p_window = 0);

	RID local_device_create();
	VkDevice local_device_get_vk_device(RID p_local_device);
	void local_device_push_command_buffers(RID p_local_device, const VkCommandBuffer *p_buffers, int p_count);
	void local_device_sync(RID p_local_device);
	void local_device_free(RID p_local_device);

	VkFormat get_screen_format() const;
	VkPhysicalDeviceLimits get_device_limits() const;

	void set_setup_buffer(VkCommandBuffer p_command_buffer);
	void append_command_buffer(VkCommandBuffer p_command_buffer);
	void resize_notify();
	void flush(bool p_flush_setup = false, bool p_flush_pending = false);
	Error prepare_buffers();
	Error swap_buffers();
	Error initialize();

	void command_begin_label(VkCommandBuffer p_command_buffer, String p_label_name, const Color p_color);
	void command_insert_label(VkCommandBuffer p_command_buffer, String p_label_name, const Color p_color);
	void command_end_label(VkCommandBuffer p_command_buffer);
	void set_object_name(VkObjectType p_object_type, uint64_t p_object_handle, String p_object_name);

	String get_device_vendor_name() const;
	String get_device_name() const;
	RenderingDevice::DeviceType get_device_type() const;
	String get_device_api_version() const;
	String get_device_pipeline_cache_uuid() const;

	void set_vsync_mode(DisplayServer::WindowID p_window, DisplayServer::VSyncMode p_mode);
	DisplayServer::VSyncMode get_vsync_mode(DisplayServer::WindowID p_window = 0) const;

	VulkanContext();
	virtual ~VulkanContext();
};

#endif // VULKAN_CONTEXT_H
