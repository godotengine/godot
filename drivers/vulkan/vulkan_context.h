#ifndef VULKAN_CONTEXT_H
#define VULKAN_CONTEXT_H

#include "core/error_list.h"
#include "core/ustring.h"
#include <vulkan/vulkan.h>

class VulkanContext {

	enum {
		MAX_EXTENSIONS = 128,
		MAX_LAYERS = 64,
		FRAME_LAG = 2
	};

	bool use_validation_layers;

	VkInstance inst;
	VkSurfaceKHR surface;
	VkPhysicalDevice gpu;
	VkPhysicalDeviceProperties gpu_props;
	uint32_t queue_family_count;
	VkQueueFamilyProperties *queue_props;
	VkDevice device;

	//present
	uint32_t graphics_queue_family_index;
	uint32_t present_queue_family_index;
	bool separate_present_queue;
	VkQueue graphics_queue;
	VkQueue present_queue;
	VkColorSpaceKHR color_space;
	VkFormat format;
	VkSemaphore image_acquired_semaphores[FRAME_LAG];
	VkSemaphore draw_complete_semaphores[FRAME_LAG];
	VkSemaphore image_ownership_semaphores[FRAME_LAG];
	int frame_index;
	VkFence fences[FRAME_LAG];
	VkPhysicalDeviceMemoryProperties memory_properties;

	typedef struct {
		VkImage image;
		VkCommandBuffer cmd;
		VkCommandBuffer graphics_to_present_cmd;
		VkImageView view;
		VkBuffer uniform_buffer;
		VkDeviceMemory uniform_memory;
		VkFramebuffer framebuffer;
		VkDescriptorSet descriptor_set;
	} SwapchainImageResources;

	VkSwapchainKHR swapchain;
	SwapchainImageResources *swapchain_image_resources;
	VkPresentModeKHR presentMode;
	uint32_t swapchainImageCount;
	uint64_t refresh_duration;
	bool syncd_with_actual_presents;
	uint64_t refresh_duration_multiplier;
	uint64_t target_IPD; // image present duration (inverse of frame rate)
	uint64_t prev_desired_present_time;
	uint32_t next_present_id;
	uint32_t last_early_id; // 0 if no early images
	uint32_t last_late_id; // 0 if no late images
	bool is_minimized;
	uint32_t current_buffer;

	//commands
	VkRenderPass render_pass;
	VkCommandPool present_cmd_pool; //for separate present queue

	bool prepared;
	int width, height;

	//extensions
	bool VK_KHR_incremental_present_enabled;
	bool VK_GOOGLE_display_timing_enabled;
	const char **instance_validation_layers;
	uint32_t enabled_extension_count;
	uint32_t enabled_layer_count;
	const char *extension_names[MAX_EXTENSIONS];
	const char *enabled_layers[MAX_LAYERS];

	PFN_vkCreateDebugUtilsMessengerEXT CreateDebugUtilsMessengerEXT;
	PFN_vkDestroyDebugUtilsMessengerEXT DestroyDebugUtilsMessengerEXT;
	PFN_vkSubmitDebugUtilsMessageEXT SubmitDebugUtilsMessageEXT;
	PFN_vkCmdBeginDebugUtilsLabelEXT CmdBeginDebugUtilsLabelEXT;
	PFN_vkCmdEndDebugUtilsLabelEXT CmdEndDebugUtilsLabelEXT;
	PFN_vkCmdInsertDebugUtilsLabelEXT CmdInsertDebugUtilsLabelEXT;
	PFN_vkSetDebugUtilsObjectNameEXT SetDebugUtilsObjectNameEXT;
	PFN_vkGetPhysicalDeviceSurfaceSupportKHR fpGetPhysicalDeviceSurfaceSupportKHR;
	PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR fpGetPhysicalDeviceSurfaceCapabilitiesKHR;
	PFN_vkGetPhysicalDeviceSurfaceFormatsKHR fpGetPhysicalDeviceSurfaceFormatsKHR;
	PFN_vkGetPhysicalDeviceSurfacePresentModesKHR fpGetPhysicalDeviceSurfacePresentModesKHR;
	PFN_vkCreateSwapchainKHR fpCreateSwapchainKHR;
	PFN_vkDestroySwapchainKHR fpDestroySwapchainKHR;
	PFN_vkGetSwapchainImagesKHR fpGetSwapchainImagesKHR;
	PFN_vkAcquireNextImageKHR fpAcquireNextImageKHR;
	PFN_vkQueuePresentKHR fpQueuePresentKHR;
	PFN_vkGetRefreshCycleDurationGOOGLE fpGetRefreshCycleDurationGOOGLE;
	PFN_vkGetPastPresentationTimingGOOGLE fpGetPastPresentationTimingGOOGLE;

	VkDebugUtilsMessengerEXT dbg_messenger;

	Error _create_validation_layers();
	Error _initialize_extensions();

	VkBool32 _check_layers(uint32_t check_count, const char **check_names, uint32_t layer_count, VkLayerProperties *layers);
	static VKAPI_ATTR VkBool32 VKAPI_CALL _debug_messenger_callback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
			VkDebugUtilsMessageTypeFlagsEXT messageType,
			const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
			void *pUserData);

	Error _create_physical_device();
	Error _create_device();
	Error _create_swap_chain();
	Error _create_semaphores();

	Error _prepare_buffers();
	Error _prepare_framebuffers();
	Error _create_buffers();

	int screen_width;
	int screen_height;
	bool minimized;

	Vector<VkCommandBuffer> command_buffer_queue;
	int command_buffer_count;

protected:
	virtual const char *_get_platform_surface_extension() const = 0;
	virtual VkResult _create_surface(VkSurfaceKHR *surface, VkInstance p_instance) = 0;

	VkSurfaceKHR &get_surface() { return surface; }

public:
	VkDevice get_device();
	VkPhysicalDevice get_physical_device();
	int get_frame_count() const;
	uint32_t get_graphics_queue() const;

	int get_screen_width(int p_screen = 0);
	int get_screen_height(int p_screen = 0);

	VkFramebuffer get_frame_framebuffer(int p_frame);
	VkRenderPass get_render_pass();
	VkFormat get_screen_format() const;
	VkPhysicalDeviceLimits get_device_limits() const;

	void set_setup_buffer(const VkCommandBuffer &pCommandBuffer);
	void append_command_buffer(const VkCommandBuffer &pCommandBuffer);
	void resize_notify();
	void flush(bool p_flush_setup = false, bool p_flush_pending = false);
	Error swap_buffers();
	Error initialize(int p_width, int p_height, bool p_minimized);
	VulkanContext();
};

#endif // VULKAN_DEVICE_H
