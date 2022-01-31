/*************************************************************************/
/*  gl_manager_x11.h                                                     */
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

#ifndef GL_MANAGER_X11_H
#define GL_MANAGER_X11_H

#ifdef X11_ENABLED

#ifdef GLES3_ENABLED

#include "core/os/os.h"
#include "core/templates/local_vector.h"
#include "servers/display_server.h"
#include <X11/Xlib.h>
#include <X11/extensions/Xrender.h>

struct GLManager_X11_Private;

class GLManager_X11 {
public:
	enum ContextType {
		GLES_3_0_COMPATIBLE,
	};

private:
	// any data specific to the window
	struct GLWindow {
		GLWindow() { in_use = false; }
		bool in_use;

		// the external ID .. should match the GL window number .. unused I think
		DisplayServer::WindowID window_id;
		int width;
		int height;
		::Window x11_window;
		int gldisplay_id;
	};

	struct GLDisplay {
		GLDisplay() { context = nullptr; }
		~GLDisplay();
		GLManager_X11_Private *context;
		::Display *x11_display;
		XVisualInfo x_vi;
		XSetWindowAttributes x_swa;
		unsigned long x_valuemask;
	};

	// just for convenience, window and display struct
	struct XWinDisp {
		::Window x11_window;
		::Display *x11_display;
	} _x_windisp;

	LocalVector<GLWindow> _windows;
	LocalVector<GLDisplay> _displays;

	GLWindow *_current_window;

	void _internal_set_current_window(GLWindow *p_win);

	GLWindow &get_window(unsigned int id) { return _windows[id]; }
	const GLWindow &get_window(unsigned int id) const { return _windows[id]; }

	const GLDisplay &get_current_display() const { return _displays[_current_window->gldisplay_id]; }
	const GLDisplay &get_display(unsigned int id) { return _displays[id]; }

	bool double_buffer;
	bool direct_render;
	int glx_minor, glx_major;
	bool use_vsync;
	ContextType context_type;

private:
	int _find_or_create_display(Display *p_x11_display);
	Error _create_context(GLDisplay &gl_display);

public:
	Error window_create(DisplayServer::WindowID p_window_id, ::Window p_window, Display *p_display, int p_width, int p_height);
	void window_destroy(DisplayServer::WindowID p_window_id);
	void window_resize(DisplayServer::WindowID p_window_id, int p_width, int p_height);

	void release_current();
	void make_current();
	void swap_buffers();

	void window_make_current(DisplayServer::WindowID p_window_id);

	Error initialize();

	void set_use_vsync(bool p_use);
	bool is_using_vsync() const;

	GLManager_X11(const Vector2i &p_size, ContextType p_context_type);
	~GLManager_X11();
};

#endif // GLES3_ENABLED
#endif // X11_ENABLED

#endif // GL_MANAGER_X11_H
