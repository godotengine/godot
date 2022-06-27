/*************************************************************************/
/*  gl_manager_osx_legacy.mm                                             */
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

#include "gl_manager_osx_legacy.h"

#ifdef OSX_ENABLED
#ifdef GLES3_ENABLED

#include <stdio.h>
#include <stdlib.h>

Error GLManager_OSX::create_context(GLWindow &win) {
	NSOpenGLPixelFormatAttribute attributes[] = {
		NSOpenGLPFADoubleBuffer,
		NSOpenGLPFAClosestPolicy,
		NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersion3_2Core,
		NSOpenGLPFAColorSize, 32,
		NSOpenGLPFADepthSize, 24,
		NSOpenGLPFAStencilSize, 8,
		0
	};

	NSOpenGLPixelFormat *pixel_format = [[NSOpenGLPixelFormat alloc] initWithAttributes:attributes];
	ERR_FAIL_COND_V(pixel_format == nil, ERR_CANT_CREATE);

	win.context = [[NSOpenGLContext alloc] initWithFormat:pixel_format shareContext:shared_context];
	ERR_FAIL_COND_V(win.context == nil, ERR_CANT_CREATE);
	if (shared_context == nullptr) {
		shared_context = win.context;
	}

	[win.context setView:win.window_view];
	[win.context makeCurrentContext];

	return OK;
}

Error GLManager_OSX::window_create(DisplayServer::WindowID p_window_id, id p_view, int p_width, int p_height) {
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

void GLManager_OSX::window_resize(DisplayServer::WindowID p_window_id, int p_width, int p_height) {
	if (!windows.has(p_window_id)) {
		return;
	}

	GLWindow &win = windows[p_window_id];

	win.width = p_width;
	win.height = p_height;

	GLint dim[2];
	dim[0] = p_width;
	dim[1] = p_height;
	CGLSetParameter((CGLContextObj)[win.context CGLContextObj], kCGLCPSurfaceBackingSize, &dim[0]);
	CGLEnable((CGLContextObj)[win.context CGLContextObj], kCGLCESurfaceBackingSize);
	if (OS::get_singleton()->is_hidpi_allowed()) {
		[win.window_view setWantsBestResolutionOpenGLSurface:YES];
	} else {
		[win.window_view setWantsBestResolutionOpenGLSurface:NO];
	}

	[win.context update];
}

int GLManager_OSX::window_get_width(DisplayServer::WindowID p_window_id) {
	if (!windows.has(p_window_id)) {
		return 0;
	}

	GLWindow &win = windows[p_window_id];
	return win.width;
}

int GLManager_OSX::window_get_height(DisplayServer::WindowID p_window_id) {
	if (!windows.has(p_window_id)) {
		return 0;
	}

	GLWindow &win = windows[p_window_id];
	return win.height;
}

void GLManager_OSX::window_destroy(DisplayServer::WindowID p_window_id) {
	if (!windows.has(p_window_id)) {
		return;
	}

	if (current_window == p_window_id) {
		current_window = DisplayServer::INVALID_WINDOW_ID;
	}

	windows.erase(p_window_id);
}

void GLManager_OSX::release_current() {
	if (current_window == DisplayServer::INVALID_WINDOW_ID) {
		return;
	}

	[NSOpenGLContext clearCurrentContext];
}

void GLManager_OSX::window_make_current(DisplayServer::WindowID p_window_id) {
	if (current_window == p_window_id) {
		return;
	}
	if (!windows.has(p_window_id)) {
		return;
	}

	GLWindow &win = windows[p_window_id];
	[win.context makeCurrentContext];

	current_window = p_window_id;
}

void GLManager_OSX::make_current() {
	if (current_window == DisplayServer::INVALID_WINDOW_ID) {
		return;
	}
	if (!windows.has(current_window)) {
		return;
	}

	GLWindow &win = windows[current_window];
	[win.context makeCurrentContext];
}

void GLManager_OSX::swap_buffers() {
	for (const KeyValue<DisplayServer::WindowID, GLWindow> &E : windows) {
		[E.value.context flushBuffer];
	}
}

void GLManager_OSX::window_update(DisplayServer::WindowID p_window_id) {
	if (!windows.has(p_window_id)) {
		return;
	}

	GLWindow &win = windows[p_window_id];
	[win.context update];
}

void GLManager_OSX::window_set_per_pixel_transparency_enabled(DisplayServer::WindowID p_window_id, bool p_enabled) {
	if (!windows.has(p_window_id)) {
		return;
	}

	GLWindow &win = windows[p_window_id];
	if (p_enabled) {
		GLint opacity = 0;
		[win.context setValues:&opacity forParameter:NSOpenGLContextParameterSurfaceOpacity];
	} else {
		GLint opacity = 1;
		[win.context setValues:&opacity forParameter:NSOpenGLContextParameterSurfaceOpacity];
	}
	[win.context update];
}

Error GLManager_OSX::initialize() {
	return OK;
}

void GLManager_OSX::set_use_vsync(bool p_use) {
	use_vsync = p_use;

	CGLContextObj ctx = CGLGetCurrentContext();
	if (ctx) {
		GLint swapInterval = p_use ? 1 : 0;
		CGLSetParameter(ctx, kCGLCPSwapInterval, &swapInterval);
		use_vsync = p_use;
	}
}

bool GLManager_OSX::is_using_vsync() const {
	return use_vsync;
}

GLManager_OSX::GLManager_OSX(ContextType p_context_type) {
	context_type = p_context_type;
}

GLManager_OSX::~GLManager_OSX() {
	release_current();
}

#endif // GLES3_ENABLED
#endif // OSX
