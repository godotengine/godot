#ifndef VULKAN_DEVICE_X11_H
#define VULKAN_DEVICE_X11_H

#include "drivers/vulkan/vulkan_context.h"
#include <X11/Xlib.h>

class VulkanContextX11 : public VulkanContext {
	Window window;
	Display *display;

	virtual const char *_get_platform_surface_extension() const;
	virtual VkResult _create_surface(VkSurfaceKHR *surface, VkInstance p_instance);

public:
	VulkanContextX11(Window p_window, Display *display);
};

#endif // VULKAN_DEVICE_X11_H
