/**************************************************************************/
/*  rendering_context_driver_vulkan_sdl3.cpp                              */
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

#include "rendering_context_driver_vulkan_sdl3.h"

#ifdef VULKAN_ENABLED

#include "core/error/error_macros.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"

#include <SDL3/SDL_vulkan.h>

#include <drivers/vulkan/godot_vulkan.h>

RenderingContextDriverVulkanSDL3::RenderingContextDriverVulkanSDL3() {
	Uint32 extension_count = 0;
	const char *const *extensions = SDL_Vulkan_GetInstanceExtensions(&extension_count);
	for (Uint32 i = 0; extensions != nullptr && i < extension_count; i++) {
		if (SDL_strcmp(extensions[i], VK_KHR_SURFACE_EXTENSION_NAME) != 0) {
			SDL_strlcpy(platform_surface_extension, extensions[i], sizeof(platform_surface_extension));
			break;
		}
	}
	if (platform_surface_extension[0] == '\0') {
		ERR_PRINT("SDL did not report a Vulkan platform surface extension.");
	}
}

RenderingContextDriverVulkanSDL3::~RenderingContextDriverVulkanSDL3() {
}

Error RenderingContextDriverVulkanSDL3::_volk_initialize() {
	// SDL owns the Vulkan loader here, so volk must use SDL's entry point rather
	// than opening a loader itself: on consoles there is no loader to open.
	if (!SDL_Vulkan_LoadLibrary(nullptr)) {
		ERR_PRINT(vformat("SDL_Vulkan_LoadLibrary failed: %s", SDL_GetError()));
		return FAILED;
	}
	PFN_vkGetInstanceProcAddr get_instance_proc_addr =
			(PFN_vkGetInstanceProcAddr)SDL_Vulkan_GetVkGetInstanceProcAddr();
	if (get_instance_proc_addr == nullptr) {
		ERR_PRINT("SDL_Vulkan_GetVkGetInstanceProcAddr returned null.");
		return FAILED;
	}
	volkInitializeCustom(get_instance_proc_addr);
	return OK;
}

const char *RenderingContextDriverVulkanSDL3::_get_platform_surface_extension() const {
	return platform_surface_extension[0] == '\0' ? nullptr : platform_surface_extension;
}

RenderingContextDriver::SurfaceID RenderingContextDriverVulkanSDL3::surface_create(const void *p_platform_data) {
	const WindowPlatformData *wpd = (const WindowPlatformData *)(p_platform_data);

	VkSurfaceKHR vk_surface = VK_NULL_HANDLE;
	if (!SDL_Vulkan_CreateSurface(wpd->window, instance_get(), get_allocation_callbacks(VK_OBJECT_TYPE_SURFACE_KHR), &vk_surface)) {
		ERR_PRINT(vformat("SDL_Vulkan_CreateSurface failed: %s", SDL_GetError()));
		return SurfaceID();
	}

	Surface *surface = memnew(Surface);
	surface->vk_surface = vk_surface;
	return SurfaceID(surface);
}

#endif // VULKAN_ENABLED
