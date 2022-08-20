/*************************************************************************/
/*  context_egl_uwp.h                                                    */
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

#ifndef CONTEXT_EGL_UWP_H
#define CONTEXT_EGL_UWP_H

#include <wrl.h>

#include <EGL/egl.h>

#include "core/error/error_list.h"
#include "core/os/os.h"

using namespace Windows::UI::Core;

class ContextEGL_UWP {
public:
	enum Driver {
		GLES_2_0,
		VULKAN, // FIXME: Add Vulkan support.
	};

private:
	CoreWindow ^ window;

	EGLDisplay mEglDisplay;
	EGLContext mEglContext;
	EGLSurface mEglSurface;

	EGLint width;
	EGLint height;

	bool vsync;

	Driver driver;

public:
	void release_current();

	void make_current();

	int get_window_width();
	int get_window_height();
	void swap_buffers();

	void set_use_vsync(bool use) { vsync = use; }
	bool is_using_vsync() const { return vsync; }

	Error initialize();
	void reset();

	void cleanup();

	ContextEGL_UWP(CoreWindow ^ p_window, Driver p_driver);
	~ContextEGL_UWP();
};

#endif // CONTEXT_EGL_UWP_H
