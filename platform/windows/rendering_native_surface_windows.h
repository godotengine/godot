/**************************************************************************/
/*  rendering_native_surface_windows.h                                    */
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

#pragma once

#include "core/variant/native_ptr.h"
#include "servers/rendering/rendering_native_surface.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

class RenderingNativeSurfaceWindows : public RenderingNativeSurface {
public:
private:
	GDCLASS(RenderingNativeSurfaceWindows, RenderingNativeSurface);

	static void _bind_methods();

	HWND window;
	HINSTANCE instance;

public:
	static Ref<RenderingNativeSurfaceWindows> create_api(GDExtensionConstPtr<const void> p_window, GDExtensionConstPtr<const void> p_instance);

	static Ref<RenderingNativeSurfaceWindows> create(HWND p_window, HINSTANCE p_instance);

	HWND get_window_handle() const {
		return window;
	}

	HINSTANCE get_instance() const {
		return instance;
	}

	RenderingContextDriver *create_rendering_context(const String &p_driver_name) override;

	RenderingNativeSurfaceWindows();
	~RenderingNativeSurfaceWindows();
};
