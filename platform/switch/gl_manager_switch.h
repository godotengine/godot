/*************************************************************************/
/*  gl_manager_switch.h                                                  */
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

#pragma once
#include "core/os/os.h"

#include <EGL/egl.h> // EGL library
#include <EGL/eglext.h> // EGL extensions
#include <GL/gl.h>
#include <GLES3/gl3.h>

#include "switch_wrapper.h"

class GLManagerSwitch {
	bool _vsync = true;
	EGLDisplay _display = nullptr;
	EGLContext _context = nullptr;
	EGLSurface _surface = nullptr;

public:
	void release_current();
	void make_current();

	void swap_buffers();

	//TODO:vrince this is not doing anything
	void set_use_vsync(bool use) { _vsync = use; }
	bool is_using_vsync() const { return _vsync; }

	Error initialize(NWindow *window);
	void cleanup();

	GLManagerSwitch();
	virtual ~GLManagerSwitch();
};
