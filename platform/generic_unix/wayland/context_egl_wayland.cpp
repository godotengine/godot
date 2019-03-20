/*************************************************************************/
/*  context_egl_wayland.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <wayland-client-protocol.h>
#include <wayland-client.h>
#include <wayland-egl.h>
#include <wayland-server.h>

#include "context_egl_wayland.h"

void ContextGL_EGL::release_current() {
	eglMakeCurrent(egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, egl_context);
}

void ContextGL_EGL::make_current() {
	eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context);
}

void ContextGL_EGL::swap_buffers() {
	if (eglSwapBuffers(egl_display, egl_surface) != EGL_TRUE) {
		cleanup();
		initialize();
		// tell rasterizer to reload textures and stuff?
	}
}

void ContextGL_EGL::set_window_size(int width, int height) {
	eglSwapBuffers(egl_display, NULL);
	this->width = (EGLint)width;
	this->height = (EGLint)height;
}

Error ContextGL_EGL::initialize() {
	eglBindAPI(EGL_OPENGL_API);
	EGLint configAttribList[] = {
		EGL_RED_SIZE, 8,
		EGL_GREEN_SIZE, 8,
		EGL_BLUE_SIZE, 8,
		// EGL_DEPTH_SIZE, 8,
		// EGL_STENCIL_SIZE, 8,
		// EGL_SAMPLE_BUFFERS, 0,
		EGL_NONE
	};
	EGLint surfaceAttribList[] = {
		EGL_NONE, EGL_NONE
	};

	EGLint numConfigs = 8;
	EGLint majorVersion = 1;
	EGLint minorVersion;
	if (context_type == GLES_2_0) {
		minorVersion = 0;
	} else {
		minorVersion = 5;
	}
	egl_display = EGL_NO_DISPLAY;
	egl_context = EGL_NO_CONTEXT;
	egl_surface = EGL_NO_SURFACE;
	EGLConfig config = nullptr;

	EGLint contextAttribs[3];
	// if (context_type == GLES_2_0) {
	// 	contextAttribs[0] = EGL_CONTEXT_MAJOR_VERSION;
	// 	contextAttribs[1] = 3;
	// 	contextAttribs[0] = EGL_CONTEXT_MINOR_VERSION;
	// 	contextAttribs[1] = 3;
	// 	contextAttribs[4] = EGL_NONE;
	// } else {
	// 	contextAttribs[0] = EGL_CONTEXT_MAJOR_VERSION;
	// 	contextAttribs[1] = 3;
	// 	contextAttribs[0] = EGL_CONTEXT_MINOR_VERSION;
	// 	contextAttribs[1] = 3;
	// 	contextAttribs[4] = EGL_NONE;
	// }
	contextAttribs[0] = EGL_CONTEXT_CLIENT_VERSION;
	contextAttribs[1] = 4;
	contextAttribs[2] = EGL_NONE;

	EGLDisplay display = eglGetDisplay(native_display);
	EGLSurface surface;
	EGLContext context;

	if (display == EGL_NO_DISPLAY) {
		print_verbose("Error: unable to get EGL display");
		return FAILED;
	}

	if (!eglInitialize(display, &majorVersion, &minorVersion)) {
		print_verbose("Error: unable to initialize EGL");
		return FAILED;
	}

	if ((eglChooseConfig(display, configAttribList, &config, 1, &numConfigs) != EGL_TRUE) || (numConfigs != 1)) {
		print_verbose("Error: unable to configure EGL");
		return FAILED;
	}

	surface = eglCreateWindowSurface(display, config, native_window, surfaceAttribList);
	if (surface == EGL_NO_SURFACE) {
		print_verbose("Error: unable to create EGL window");
		return FAILED;
	}

	context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttribs);
	if (context == EGL_NO_CONTEXT) {
		print_verbose("Error: unable to create EGL context");
		return FAILED;
	}

	if (!eglMakeCurrent(display, surface, surface, context)) {
		print_verbose("Error: unable to make EGL context current");
		return FAILED;
	}
	egl_display = display;
	egl_surface = surface;
	egl_context = context;

	eglQuerySurface(display, surface, EGL_WIDTH, &width);
	eglQuerySurface(display, surface, EGL_HEIGHT, &height);
	return OK;
}

int ContextGL_EGL::get_window_width() {
	return width;
}

int ContextGL_EGL::get_window_height() {
	return height;
}

void ContextGL_EGL::set_use_vsync(bool p_use) {
	use_vsync = p_use;
}

bool ContextGL_EGL::is_using_vsync() const {
	return use_vsync;
}

void ContextGL_EGL::cleanup() {
	if (egl_display != EGL_NO_DISPLAY && egl_surface != EGL_NO_SURFACE) {
		eglDestroySurface(egl_display, egl_surface);
		egl_surface = EGL_NO_SURFACE;
	}

	if (egl_display != EGL_NO_DISPLAY && egl_context != EGL_NO_CONTEXT) {
		eglDestroyContext(egl_display, egl_context);
		egl_context = EGL_NO_CONTEXT;
	}

	if (egl_display != EGL_NO_DISPLAY) {
		eglTerminate(egl_display);
		egl_display = EGL_NO_DISPLAY;
	}
}

ContextGL_EGL::ContextGL_EGL(EGLNativeDisplayType p_egl_display,
		EGLNativeWindowType &p_egl_window,
		const OS::VideoMode &p_default_video_mode,
		Driver p_context_type) {
	default_video_mode = p_default_video_mode;

	context_type = p_context_type;

	double_buffer = false;
	direct_render = false;
	native_display = p_egl_display;
	native_window = p_egl_window;
	use_vsync = false;
}

ContextGL_EGL::~ContextGL_EGL() {
	cleanup();
}
