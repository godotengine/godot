/**************************************************************************/
/*  rendering_context_driver_vulkan_x11.cpp                               */
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

#ifdef VULKAN_ENABLED

#include "rendering_context_driver_vulkan_x11.h"
#include "drivers/vulkan/rendering_native_surface_vulkan.h"
#include "rendering_native_surface_x11.h"

#include "drivers/vulkan/godot_vulkan.h"

const char *RenderingContextDriverVulkanX11::_get_platform_surface_extension() const {
	return VK_KHR_XLIB_SURFACE_EXTENSION_NAME;
}

RenderingContextDriver::SurfaceID RenderingContextDriverVulkanX11::surface_create(Ref<RenderingNativeSurface> p_native_surface) {
	Ref<RenderingNativeSurfaceX11> x11_native_surface = Object::cast_to<RenderingNativeSurfaceX11>(*p_native_surface);
	ERR_FAIL_COND_V(x11_native_surface.is_null(), SurfaceID());

	VkXlibSurfaceCreateInfoKHR create_info = {};
	create_info.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
	create_info.dpy = x11_native_surface->get_display();
	create_info.window = x11_native_surface->get_window();

	VkSurfaceKHR vk_surface = VK_NULL_HANDLE;
	VkResult err = vkCreateXlibSurfaceKHR(instance_get(), &create_info, get_allocation_callbacks(VK_OBJECT_TYPE_SURFACE_KHR), &vk_surface);
	ERR_FAIL_COND_V(err != VK_SUCCESS, SurfaceID());

	Ref<RenderingNativeSurfaceVulkan> vulkan_surface = RenderingNativeSurfaceVulkan::create(vk_surface);
	RenderingContextDriver::SurfaceID result = RenderingContextDriverVulkan::surface_create(vulkan_surface);
	return result;
}

RenderingContextDriverVulkanX11::RenderingContextDriverVulkanX11() {
	// Does nothing.
}

RenderingContextDriverVulkanX11::~RenderingContextDriverVulkanX11() {
	// Does nothing.
}

#endif // VULKAN_ENABLED
