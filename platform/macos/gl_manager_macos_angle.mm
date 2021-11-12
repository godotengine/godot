/*************************************************************************/
/*  gl_manager_macos_angle.mm                                            */
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

#include "gl_manager_macos_angle.h"

#ifdef MACOS_ENABLED
#ifdef USE_OPENGL_ANGLE

#include <stdio.h>
#include <stdlib.h>

Error GLManager_MacOS::create_context(GLWindow &win) {
	if (display == EGL_NO_DISPLAY) {
		EGLint angle_platform_type = EGL_PLATFORM_ANGLE_TYPE_METAL_ANGLE;

		List<String> args = OS::get_singleton()->get_cmdline_args();
		for (const List<String>::Element *E = args.front(); E; E = E->next()) {
			if (E->get() == "--angle-platform-type" && E->next()) {
				String cmd = E->next()->get().to_lower();
				if (cmd == "metal") {
					angle_platform_type = EGL_PLATFORM_ANGLE_TYPE_METAL_ANGLE;
				} else if (cmd == "opengl") {
					angle_platform_type = EGL_PLATFORM_ANGLE_TYPE_OPENGL_ANGLE;
				} else {
					WARN_PRINT("Invalid ANGLE platform type, it should be \"metal\" or \"opengl\".");
				}
			}
		}

		EGLAttrib display_attributes[] = {
			EGL_PLATFORM_ANGLE_TYPE_ANGLE,
			angle_platform_type,
			EGL_NONE,
		};

		display = eglGetPlatformDisplay(EGL_PLATFORM_ANGLE_ANGLE, nullptr, display_attributes);
		if (display == EGL_NO_DISPLAY) {
			WARN_PRINT("Can't get ANGLE display with the selected platform type, falling back to another one.");
			if (display_attributes[1] == EGL_PLATFORM_ANGLE_TYPE_METAL_ANGLE) {
				display_attributes[1] = EGL_PLATFORM_ANGLE_TYPE_OPENGL_ANGLE;
			} else {
				display_attributes[1] = EGL_PLATFORM_ANGLE_TYPE_METAL_ANGLE;
			}
			display = eglGetPlatformDisplay(EGL_PLATFORM_ANGLE_ANGLE, nullptr, display_attributes);
		}
		ERR_FAIL_COND_V(eglInitialize(display, nullptr, nullptr) == EGL_FALSE, ERR_CANT_CREATE);
	}

	EGLint config_attribs[] = {
		EGL_BUFFER_SIZE,
		32,
		EGL_DEPTH_SIZE,
		24,
		EGL_STENCIL_SIZE,
		8,
		EGL_SAMPLE_BUFFERS,
		0,
		EGL_NONE,
	};

	EGLint surface_attribs[] = {
		EGL_NONE,
	};

	EGLint num_configs = 0;
	EGLConfig config = nullptr;
	EGLint context_attribs[]{
		EGL_CONTEXT_CLIENT_VERSION,
		3,
		EGL_NONE,
	};

	ERR_FAIL_COND_V(eglGetConfigs(display, NULL, 0, &num_configs) == EGL_FALSE, ERR_CANT_CREATE);
	ERR_FAIL_COND_V(eglChooseConfig(display, config_attribs, &config, 1, &num_configs) == EGL_FALSE, ERR_CANT_CREATE);

	CALayer *layer = [win.window_view layer];
	win.surface = eglCreateWindowSurface(display, config, (__bridge void *)layer, surface_attribs);
	ERR_FAIL_COND_V(win.surface == EGL_NO_SURFACE, ERR_CANT_CREATE);

	if (shared_context == EGL_NO_CONTEXT) {
		shared_context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attribs);
		ERR_FAIL_COND_V(shared_context == EGL_NO_CONTEXT, ERR_CANT_CREATE);
	}
	win.context = shared_context;

	eglMakeCurrent(display, win.surface, win.surface, win.context);

	return OK;
}

Error GLManager_MacOS::window_create(DisplayServer::WindowID p_window_id, id p_view, int p_width, int p_height) {
	GLWindow win;
	win.width = p_width;
	win.height = p_height;
	win.window_view = p_view;

	if (create_context(win) != OK) {
		return FAILED;
	}

	windows[p_window_id] = win;
	window_make_current(p_window_id);

	return OK;
}

void GLManager_MacOS::window_resize(DisplayServer::WindowID p_window_id, int p_width, int p_height) {
	if (!windows.has(p_window_id)) {
		return;
	}

	GLWindow &win = windows[p_window_id];
	win.width = p_width;
	win.height = p_height;
}

int GLManager_MacOS::window_get_width(DisplayServer::WindowID p_window_id) {
	if (!windows.has(p_window_id)) {
		return 0;
	}

	GLWindow &win = windows[p_window_id];
	return win.width;
}

int GLManager_MacOS::window_get_height(DisplayServer::WindowID p_window_id) {
	if (!windows.has(p_window_id)) {
		return 0;
	}

	GLWindow &win = windows[p_window_id];
	return win.height;
}

void GLManager_MacOS::window_destroy(DisplayServer::WindowID p_window_id) {
	if (!windows.has(p_window_id)) {
		return;
	}

	GLWindow &win = windows[p_window_id];
	if (win.surface != EGL_NO_SURFACE) {
		eglDestroySurface(display, win.surface);
	}

	if (current_window == p_window_id) {
		current_window = DisplayServer::INVALID_WINDOW_ID;
	}

	windows.erase(p_window_id);
}

void GLManager_MacOS::release_current() {
	if (current_window == DisplayServer::INVALID_WINDOW_ID) {
		return;
	}

	eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
}

void GLManager_MacOS::window_make_current(DisplayServer::WindowID p_window_id) {
	if (current_window == p_window_id) {
		return;
	}
	if (!windows.has(p_window_id)) {
		return;
	}

	GLWindow &win = windows[p_window_id];
	eglMakeCurrent(display, win.surface, win.surface, win.context);

	current_window = p_window_id;
}

void GLManager_MacOS::make_current() {
	if (current_window == DisplayServer::INVALID_WINDOW_ID) {
		return;
	}
	if (!windows.has(current_window)) {
		return;
	}

	GLWindow &win = windows[current_window];
	eglMakeCurrent(display, win.surface, win.surface, win.context);
}

void GLManager_MacOS::swap_buffers() {
	for (const KeyValue<DisplayServer::WindowID, GLWindow> &E : windows) {
		eglSwapBuffers(display, E.value.surface);
	}
}

void GLManager_MacOS::window_update(DisplayServer::WindowID p_window_id) {
	// Not used.
}

Error GLManager_MacOS::initialize() {
	return OK;
}

void GLManager_MacOS::set_use_vsync(bool p_use) {
	use_vsync = p_use;

	if (!p_use) {
		eglSwapInterval(display, 0);
	} else {
		eglSwapInterval(display, 1);
	}
}

bool GLManager_MacOS::is_using_vsync() const {
	return use_vsync;
}

GLManager_MacOS::GLManager_MacOS(ContextType p_context_type) {
	context_type = p_context_type;
}

GLManager_MacOS::~GLManager_MacOS() {
	release_current();

	for (const KeyValue<DisplayServer::WindowID, GLWindow> &E : windows) {
		if (E.value.surface != EGL_NO_SURFACE) {
			eglDestroySurface(display, E.value.surface);
		}
	}

	if (shared_context != EGL_NO_CONTEXT) {
		eglDestroyContext(display, shared_context);
	}
	if (display != EGL_NO_DISPLAY) {
		eglTerminate(display);
	}
}

#endif // USE_OPENGL_ANGLE
#endif // MACOS_ENABLED
