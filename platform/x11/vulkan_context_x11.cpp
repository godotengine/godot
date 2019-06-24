#include "vulkan_context_x11.h"
#include <vulkan/vulkan_xlib.h>
const char *VulkanContextX11::_get_platform_surface_extension() const {
	return VK_KHR_XLIB_SURFACE_EXTENSION_NAME;
}

int VulkanContextX11::window_create(::Window p_window, Display *p_display, int p_width, int p_height) {

	VkXlibSurfaceCreateInfoKHR createInfo;
	createInfo.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
	createInfo.pNext = NULL;
	createInfo.flags = 0;
	createInfo.dpy = p_display;
	createInfo.window = p_window;

	VkSurfaceKHR surface;
	VkResult err = vkCreateXlibSurfaceKHR(_get_instance(), &createInfo, NULL, &surface);
	ERR_FAIL_COND_V(err, -1);
	return _window_create(surface, p_width, p_height);
}

VulkanContextX11::VulkanContextX11() {
}
