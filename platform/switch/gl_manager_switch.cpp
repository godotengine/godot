/*************************************************************************/
/*  gl_manager_swotch.cpp                                                */
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

#include "gl_manager_switch.h"

GLManagerSwitch::GLManagerSwitch() {
	setenv("EGL_LOG_LEVEL", "debug", 1);
	setenv("MESA_VERBOSE", "all", 1);
	setenv("NOUVEAU_MESA_DEBUG", "1", 1);
}

GLManagerSwitch::~GLManagerSwitch() {
	cleanup();
}

Error GLManagerSwitch::initialize(NWindow *window) {
	OS::get_singleton()->print("GLManagerSwitch::initialize\n");

	// Connect to the EGL default display
	_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
	if (!_display) {
		OS::get_singleton()->print("Could not connect to display! error: %d\n", eglGetError());
		return ERR_UNCONFIGURED;
	}

	// Initialize the EGL display connection
	eglInitialize(_display, NULL, NULL);

	// Select OpenGL (Core) as the desired graphics API
	if (eglBindAPI(EGL_OPENGL_API) == EGL_FALSE) {
		printf("Could not set API! error: %d", eglGetError());
		eglTerminate(_display);
		_display = NULL;
		return ERR_UNCONFIGURED;
	}

	// Get an appropriate EGL framebuffer configuration
	EGLConfig config;
	EGLint numConfigs;
	static const EGLint framebufferAttributeList[] = {
		EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
		EGL_RED_SIZE, 8,
		EGL_GREEN_SIZE, 8,
		EGL_BLUE_SIZE, 8,
		EGL_ALPHA_SIZE, 8,
		EGL_DEPTH_SIZE, 24,
		EGL_STENCIL_SIZE, 8,
		EGL_NONE
	};

	eglChooseConfig(_display, framebufferAttributeList, &config, 1, &numConfigs);

	if (numConfigs == 0) {
		OS::get_singleton()->print("No config found! error: %d\n", eglGetError());
		eglTerminate(_display);
		_display = NULL;
		return ERR_UNCONFIGURED;
	}

	// Create an EGL window surface
	_surface = eglCreateWindowSurface(_display, config, window, NULL);
	if (!_surface) {
		OS::get_singleton()->print("Surface creation failed! error: %d\n", eglGetError());
		eglTerminate(_display);
		_display = NULL;
		return ERR_UNCONFIGURED;
	}

	static const EGLint contextAttributeList[] = {
		EGL_CONTEXT_OPENGL_PROFILE_MASK_KHR, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_KHR,
		EGL_CONTEXT_MAJOR_VERSION_KHR, 4,
		EGL_CONTEXT_MINOR_VERSION_KHR, 3,
		EGL_NONE
	};

	// Create an EGL rendering context
	_context = eglCreateContext(_display, config, EGL_NO_CONTEXT, contextAttributeList);

	if (!_context) {
		OS::get_singleton()->print("Context creation failed! error: %d\n", eglGetError());
		eglDestroySurface(_display, _surface);
		eglTerminate(_display);
		_surface = nullptr;
		_display = nullptr;
		return ERR_UNCONFIGURED;
	}

	// Connect the context to the surface
	if (!eglMakeCurrent(_display, _surface, _surface, _context)) {
		OS::get_singleton()->print("Make current failed! error: %d\n", eglGetError());
        eglDestroyContext(_display, _context);
		eglDestroySurface(_display, _surface);
		eglTerminate(_display);
        _context = nullptr;
		_surface = nullptr;
		_display = nullptr;
		return ERR_UNCONFIGURED;
	}

	OS::get_singleton()->print("EGL context created\n");
	return OK;
}

void GLManagerSwitch::cleanup() {
	if (_display) {
		eglMakeCurrent(_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
		if (_context) {
			eglDestroyContext(_display, _context);
			_context = NULL;
		}
		if (_surface) {
			eglDestroySurface(_display, _surface);
			_surface = NULL;
		}
		eglTerminate(_display);
		_display = NULL;
	}
}

void GLManagerSwitch::release_current() {
	eglMakeCurrent(_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
}

void GLManagerSwitch::make_current() {
	eglMakeCurrent(_display, _surface, _surface, _context);
}

void GLManagerSwitch::swap_buffers() {
	eglSwapBuffers(_display, _surface);
}
