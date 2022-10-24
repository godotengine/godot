/*************************************************************************/
/*  gl_manager_wayland.h                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef GL_MANAGER_WAYLAND_H
#define GL_MANAGER_WAYLAND_H

#ifdef WAYLAND_ENABLED
#ifdef GLES3_ENABLED

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include "dynwrappers/wayland-egl-core.h"

#include "core/templates/local_vector.h"
#include "servers/display_server.h"

class GLManagerWayland {
private:
	// An EGL side rappresentation of a wl_display with its own rendering
	// context.
	struct GLDisplay {
		struct wl_display *wl_display = nullptr;

		EGLDisplay egl_display = EGL_NO_DISPLAY;
		EGLContext egl_context = EGL_NO_CONTEXT;
		EGLConfig egl_config;
	};

	// EGL specific window data.
	struct GLWindow {
		bool initialized = false;

		int width = 0;
		int height = 0;

		// An handle to the GLDisplay associated with this window.
		int gldisplay_id = -1;

		struct wl_egl_window *wl_egl_window = nullptr;
		EGLSurface egl_surface = EGL_NO_SURFACE;
	};

	LocalVector<GLDisplay> displays;
	LocalVector<GLWindow> windows;

	GLWindow *current_window = nullptr;

	int _get_gldisplay_id(struct wl_display *p_display);
	Error _gldisplay_create_context(GLDisplay &p_gldisplay);

public:
	Error window_create(DisplayServer::WindowID p_window_id, struct wl_display *p_display, struct wl_surface *p_surface, int p_width, int p_height);

	void window_destroy(DisplayServer::WindowID p_window_id);
	void window_resize(DisplayServer::WindowID p_window_id, int p_width, int p_height);

	void release_current();
	void make_current();
	void swap_buffers();

	void window_make_current(DisplayServer::WindowID p_window_id);

	Error initialize();

	GLManagerWayland();
	~GLManagerWayland();
};

#endif // GLES3_ENABLED
#endif // WAYLAND_ENABLED

#endif // GL_MANAGER_WAYLAND_H
