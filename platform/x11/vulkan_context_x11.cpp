#include "vulkan_context_x11.h"
#include <vulkan/vulkan_xlib.h>
const char *VulkanContextX11::_get_platform_surface_extension() const {
	return VK_KHR_XLIB_SURFACE_EXTENSION_NAME;
}

VkResult VulkanContextX11::_create_surface(VkSurfaceKHR *surface, VkInstance p_instance) {

	VkXlibSurfaceCreateInfoKHR createInfo;
	createInfo.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
	createInfo.pNext = NULL;
	createInfo.flags = 0;
	createInfo.dpy = display;
	createInfo.window = window;

	return vkCreateXlibSurfaceKHR(p_instance, &createInfo, NULL, surface);
}

VulkanContextX11::VulkanContextX11(Window p_window, Display *p_display) {
	window = p_window;
	display = p_display;
}
