/**************************************************************************/
/*  rendering_native_surface_vulkan.h                                     */
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

#ifndef RENDERING_NATIVE_SURFACE_VULKAN_H
#define RENDERING_NATIVE_SURFACE_VULKAN_H

#include "core/variant/native_ptr.h"
#include "servers/rendering/rendering_native_surface.h"

#ifdef VULKAN_ENABLED
#ifdef USE_VOLK
#include <volk.h>
#else
#include <vulkan/vulkan.h>
#endif
#endif

class RenderingNativeSurfaceVulkan : public RenderingNativeSurface {
	GDCLASS(RenderingNativeSurfaceVulkan, RenderingNativeSurface);

	static void _bind_methods();

#ifdef VULKAN_ENABLED
	VkSurfaceKHR vulkan_surface = VK_NULL_HANDLE;
#endif

public:
	static Ref<RenderingNativeSurfaceVulkan> create_api(GDExtensionConstPtr<const void> vulkan_surface);

#ifdef VULKAN_ENABLED
	static Ref<RenderingNativeSurfaceVulkan> create(VkSurfaceKHR vulkan_surface);

	VkSurfaceKHR get_vulkan_surface() const {
		return vulkan_surface;
	};
#endif

	RenderingContextDriver *create_rendering_context() override;

	RenderingNativeSurfaceVulkan();
	~RenderingNativeSurfaceVulkan();
};

#endif // RENDERING_NATIVE_SURFACE_VULKAN_H
