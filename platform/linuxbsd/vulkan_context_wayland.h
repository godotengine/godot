#ifndef VULKAN_DEVICE_WAYLAND_H
#define VULKAN_DEVICE_WAYLAND_H

#include "drivers/vulkan/vulkan_context.h"

class VulkanContextWayland : public VulkanContext {

	const char *_get_platform_surface_extension() const;

public:
	Error window_create(DisplayServer::WindowID p_window_id, DisplayServer::VSyncMode p_vsync_mode, struct wl_display* p_display, struct wl_surface* p_surface, int p_width, int p_height);

	VulkanContextWayland();
	~VulkanContextWayland();
};

#endif // VULKAN_DEVICE_WAYLAND_H
