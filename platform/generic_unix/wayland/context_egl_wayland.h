/*************************************************************************/
/*  context_egl_wayland.h                                                */
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

/**
	@author Timo Kandra <toger5@hotmail.de>
*/
#ifndef CONTEXT_EGL_H
#define CONTEXT_EGL_H

#include "core/os/os.h"
#include "drivers/gl_context/context_gl.h"
#include <EGL/egl.h>
#include <EGL/eglplatform.h>

class ContextGL_EGL : public ContextGL {

public:
	enum Driver {
		GLES_2_0,
		GLES_3_0
	};

private:
	OS::VideoMode default_video_mode;
	Driver context_type;

	EGLint width;
	EGLint height;

	EGLNativeDisplayType native_display;
	EGLNativeWindowType native_window;

	EGLSurface egl_surface;
	EGLDisplay egl_display;
	EGLContext egl_context;

	bool double_buffer;
	bool direct_render;
	bool use_vsync;

public:
	virtual void release_current();
	virtual void make_current();
	virtual void swap_buffers();
	virtual int get_window_width();
	virtual int get_window_height();

	void set_window_size(int width, int height);

	virtual Error initialize();

	virtual void set_use_vsync(bool p_use);
	virtual bool is_using_vsync() const;
	void cleanup();
	ContextGL_EGL(EGLNativeDisplayType p_egl_display, EGLNativeWindowType &p_egl_window, const OS::VideoMode &p_default_video_mode, Driver p_context_type);
	virtual ~ContextGL_EGL();
};

#endif
