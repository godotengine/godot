/**************************************************************************/
/*  rendering_native_surface_windows.cpp                                  */
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

#include "rendering_native_surface_windows.h"
#include "rendering_context_driver_vulkan_windows.h"

#if defined(D3D12_ENABLED)
#include "drivers/d3d12/rendering_context_driver_d3d12.h"
#endif

#if defined(VULKAN_ENABLED)
#include "platform/windows/rendering_context_driver_vulkan_windows.h"
#endif

void RenderingNativeSurfaceWindows::_bind_methods() {
	ClassDB::bind_static_method("RenderingNativeSurfaceWindows", D_METHOD("create_api", "hwnd", "instance"), &RenderingNativeSurfaceWindows::create_api);
}

Ref<RenderingNativeSurfaceWindows> RenderingNativeSurfaceWindows::create_api(GDExtensionConstPtr<const void> p_window, GDExtensionConstPtr<const void> p_instance) {
	return RenderingNativeSurfaceWindows::create((HWND)p_window.operator const void *(), (HINSTANCE)p_instance.operator const void *());
}

Ref<RenderingNativeSurfaceWindows> RenderingNativeSurfaceWindows::create(HWND p_window, HINSTANCE p_instance) {
	Ref<RenderingNativeSurfaceWindows> result = memnew(RenderingNativeSurfaceWindows);
	result->window = p_window;
	result->instance = p_instance;
	return result;
}

RenderingContextDriver *RenderingNativeSurfaceWindows::create_rendering_context(const String &p_driver_name) {
#if defined(VULKAN_ENABLED)
	if (p_driver_name == "vulkan") {
		return memnew(RenderingContextDriverVulkanWindows);
	}
#endif
#if defined(D3D12_ENABLED)
	if (p_driver_name == "d3d12") {
		return memnew(RenderingContextDriverD3D12);
	}
#endif
	return nullptr;
}

RenderingNativeSurfaceWindows::RenderingNativeSurfaceWindows() {
	// Does nothing.
}

RenderingNativeSurfaceWindows::~RenderingNativeSurfaceWindows() {
	// Does nothing.
}
