/**************************************************************************/
/*  rendering_native_surface_wayland.cpp                                  */
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

#include "rendering_native_surface_wayland.h"
#include "core/object/class_db.h"

#ifdef VULKAN_ENABLED
#include "wayland/rendering_context_driver_vulkan_wayland.h"
#endif

void RenderingNativeSurfaceWayland::_bind_methods() {
	ClassDB::bind_static_method("RenderingNativeSurfaceWayland", D_METHOD("create", "window", "display"), &RenderingNativeSurfaceWayland::create_api);
}

Ref<RenderingNativeSurfaceWayland> RenderingNativeSurfaceWayland::create_api(GDExtensionConstPtr<const void> p_display, GDExtensionConstPtr<const void> p_surface) {
	return RenderingNativeSurfaceWayland::create((struct wl_display *)p_display.operator const void *(), (struct wl_surface *)p_surface.operator const void *());
}

Ref<RenderingNativeSurfaceWayland> RenderingNativeSurfaceWayland::create(struct wl_display *p_display, struct wl_surface *p_surface) {
	Ref<RenderingNativeSurfaceWayland> result = memnew(RenderingNativeSurfaceWayland);
	result->surface = p_surface;
	result->display = p_display;
	return result;
}

RenderingContextDriver *RenderingNativeSurfaceWayland::create_rendering_context() {
#if defined(VULKAN_ENABLED)
	return memnew(RenderingContextDriverVulkanWayland);
#else
	return nullptr;
#endif
}

RenderingNativeSurfaceWayland::RenderingNativeSurfaceWayland() {
	// Does nothing.
}

RenderingNativeSurfaceWayland::~RenderingNativeSurfaceWayland() {
	// Does nothing.
}
