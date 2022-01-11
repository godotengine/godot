#include "vulkan_context_wayland.h"

#ifdef USE_VOLK
#include <volk.h>
#else
#include <vulkan/vulkan.h>
#endif

const char *VulkanContextWayland::_get_platform_surface_extension() const {
	return VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME;
}

Error VulkanContextWayland::window_create(DisplayServer::WindowID p_window_id, DisplayServer::VSyncMode p_vsync_mode, struct wl_display* p_display, struct wl_surface* p_surface, int p_width, int p_height) {
	VkWaylandSurfaceCreateInfoKHR createInfo;
	createInfo.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
	createInfo.pNext = nullptr;
	createInfo.flags = 0;
	createInfo.display = p_display;
	createInfo.surface = p_surface;

	VkSurfaceKHR surface;
	VkResult err = vkCreateWaylandSurfaceKHR(get_instance(), &createInfo, nullptr, &surface);
	ERR_FAIL_COND_V(err, ERR_CANT_CREATE);
	return _window_create(p_window_id, p_vsync_mode, surface, p_width, p_height);
}

VulkanContextWayland::VulkanContextWayland() {
}

VulkanContextWayland::~VulkanContextWayland() {
}
