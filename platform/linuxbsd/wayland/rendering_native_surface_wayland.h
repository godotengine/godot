/**************************************************************************/
/*  rendering_native_surface_wayland.h                                    */
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

#ifndef RENDERING_NATIVE_SURFACE_WAYLAND_H
#define RENDERING_NATIVE_SURFACE_WAYLAND_H

#include "core/variant/native_ptr.h"
#include "servers/rendering/rendering_native_surface.h"

class RenderingNativeSurfaceWayland : public RenderingNativeSurface {
	GDCLASS(RenderingNativeSurfaceWayland, RenderingNativeSurface);

	static void _bind_methods();

	struct wl_display *display;
	struct wl_surface *surface;

public:
	static Ref<RenderingNativeSurfaceWayland> create_api(GDExtensionConstPtr<const void> p_display, GDExtensionConstPtr<const void> p_surface);

	static Ref<RenderingNativeSurfaceWayland> create(struct wl_display *p_display, wl_surface *p_surface);

	struct wl_display *get_display() const {
		return display;
	};

	struct wl_surface *get_surface() const {
		return surface;
	};

	RenderingContextDriver *create_rendering_context() override;

	RenderingNativeSurfaceWayland();
	~RenderingNativeSurfaceWayland();
};

#endif // RENDERING_NATIVE_SURFACE_WAYLAND_H
