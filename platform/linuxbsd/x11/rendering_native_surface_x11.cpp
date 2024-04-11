/**************************************************************************/
/*  rendering_native_surface_x11.cpp                                      */
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

#include "rendering_native_surface_x11.h"
#include "core/object/class_db.h"

#if defined(VULKAN_ENABLED)
#include "x11/rendering_context_driver_vulkan_x11.h"
#endif

void RenderingNativeSurfaceX11::_bind_methods() {
	ClassDB::bind_static_method("RenderingNativeSurfaceX11", D_METHOD("create", "window", "display"), &RenderingNativeSurfaceX11::create_api);
}

Ref<RenderingNativeSurfaceX11> RenderingNativeSurfaceX11::create_api(GDExtensionConstPtr<const void> p_window, GDExtensionConstPtr<const void> p_display) {
	return RenderingNativeSurfaceX11::create((::Window)p_window.operator const void *(), (Display *)p_display.operator const void *());
}

Ref<RenderingNativeSurfaceX11> RenderingNativeSurfaceX11::create(::Window p_window, Display *p_display) {
	Ref<RenderingNativeSurfaceX11> result = memnew(RenderingNativeSurfaceX11);
	result->window = p_window;
	result->display = p_display;
	return result;
}

RenderingContextDriver *RenderingNativeSurfaceX11::create_rendering_context() {
#if defined(VULKAN_ENABLED)
	return memnew(RenderingContextDriverVulkanX11);
#else
	return nullptr;
#endif
}

RenderingNativeSurfaceX11::RenderingNativeSurfaceX11() {
	// Does nothing.
}

RenderingNativeSurfaceX11::~RenderingNativeSurfaceX11() {
	// Does nothing.
}
