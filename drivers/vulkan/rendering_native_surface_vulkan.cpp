/**************************************************************************/
/*  rendering_native_surface_vulkan.cpp                                   */
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

#include "rendering_native_surface_vulkan.h"

#include "drivers/vulkan/rendering_context_driver_vulkan.h"

void RenderingNativeSurfaceVulkan::_bind_methods() {
	ClassDB::bind_static_method("RenderingNativeSurfaceVulkan", D_METHOD("create", "vulkan_surface"), &RenderingNativeSurfaceVulkan::create_api);
}

Ref<RenderingNativeSurfaceVulkan> RenderingNativeSurfaceVulkan::create_api(GDExtensionConstPtr<const void> p_vulkan_surface) {
	Ref<RenderingNativeSurfaceVulkan> result = nullptr;
#ifdef VULKAN_ENABLED
	result = RenderingNativeSurfaceVulkan::create((VkSurfaceKHR)p_vulkan_surface.operator const void *());
#endif
	return result;
}

#ifdef VULKAN_ENABLED

Ref<RenderingNativeSurfaceVulkan> RenderingNativeSurfaceVulkan::create(VkSurfaceKHR p_vulkan_surface) {
	Ref<RenderingNativeSurfaceVulkan> result = memnew(RenderingNativeSurfaceVulkan);
	result->vulkan_surface = p_vulkan_surface;
	return result;
}

#endif

RenderingContextDriver *RenderingNativeSurfaceVulkan::create_rendering_context() {
#if defined(VULKAN_ENABLED)
	return memnew(RenderingContextDriverVulkan);
#else
	return nullptr;
#endif
}

RenderingNativeSurfaceVulkan::RenderingNativeSurfaceVulkan() {
	// Does nothing.
}

RenderingNativeSurfaceVulkan::~RenderingNativeSurfaceVulkan() {
	// Does nothing.
}
