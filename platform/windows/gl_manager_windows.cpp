/*************************************************************************/
/*  gl_manager_windows.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "gl_manager_windows.h"

#ifdef WINDOWS_ENABLED
#ifdef OPENGL_ENABLED

#include <stdio.h>
#include <stdlib.h>

#include <dwmapi.h>

#define WGL_CONTEXT_MAJOR_VERSION_ARB 0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB 0x2092
#define WGL_CONTEXT_FLAGS_ARB 0x2094
#define WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB 0x00000002
#define WGL_CONTEXT_PROFILE_MASK_ARB 0x9126
#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB 0x00000001

#define _WGL_CONTEXT_DEBUG_BIT_ARB 0x0001

#if defined(__GNUC__)
// Workaround GCC warning from -Wcast-function-type.
#define wglGetProcAddress (void *)wglGetProcAddress
#endif

typedef HGLRC(APIENTRY *PFNWGLCREATECONTEXTATTRIBSARBPROC)(HDC, HGLRC, const int *);

int GLManager_Windows::_find_or_create_display(GLWindow &win) {
	// find display NYI, only 1 supported so far
	if (_displays.size())
		return 0;

	//	for (unsigned int n = 0; n < _displays.size(); n++) {
	//		const GLDisplay &d = _displays[n];
	//		if (d.x11_display == p_x11_display)
	//			return n;
	//	}

	// create
	GLDisplay d_temp = {};
	_displays.push_back(d_temp);
	int new_display_id = _displays.size() - 1;

	// create context
	GLDisplay &d = _displays[new_display_id];
	Error err = _create_context(win, d);

	if (err != OK) {
		// not good
		// delete the _display?
		_displays.remove(new_display_id);
		return -1;
	}

	return new_display_id;
}

Error GLManager_Windows::_create_context(GLWindow &win, GLDisplay &gl_display) {
	static PIXELFORMATDESCRIPTOR pfd = {
		sizeof(PIXELFORMATDESCRIPTOR), // Size Of This Pixel Format Descriptor
		1,
		PFD_DRAW_TO_WINDOW | // Format Must Support Window
				PFD_SUPPORT_OPENGL | // Format Must Support OpenGL
				PFD_DOUBLEBUFFER,
		(BYTE)PFD_TYPE_RGBA,
		(BYTE)(OS::get_singleton()->is_layered_allowed() ? 32 : 24),
		(BYTE)0, (BYTE)0, (BYTE)0, (BYTE)0, (BYTE)0, (BYTE)0, // Color Bits Ignored
		(BYTE)(OS::get_singleton()->is_layered_allowed() ? 8 : 0), // Alpha Buffer
		(BYTE)0, // Shift Bit Ignored
		(BYTE)0, // No Accumulation Buffer
		(BYTE)0, (BYTE)0, (BYTE)0, (BYTE)0, // Accumulation Bits Ignored
		(BYTE)24, // 24Bit Z-Buffer (Depth Buffer)
		(BYTE)0, // No Stencil Buffer
		(BYTE)0, // No Auxiliary Buffer
		(BYTE)PFD_MAIN_PLANE, // Main Drawing Layer
		(BYTE)0, // Reserved
		0, 0, 0 // Layer Masks Ignored
	};

	// alias
	HDC hDC = win.hDC;

	int pixel_format = ChoosePixelFormat(hDC, &pfd);
	if (!pixel_format) // Did Windows Find A Matching Pixel Format?
	{
		return ERR_CANT_CREATE; // Return FALSE
	}

	BOOL ret = SetPixelFormat(hDC, pixel_format, &pfd);
	if (!ret) // Are We Able To Set The Pixel Format?
	{
		return ERR_CANT_CREATE; // Return FALSE
	}

	gl_display.hRC = wglCreateContext(hDC);
	if (!gl_display.hRC) // Are We Able To Get A Rendering Context?
	{
		return ERR_CANT_CREATE; // Return FALSE
	}

	wglMakeCurrent(hDC, gl_display.hRC);

	if (opengl_3_context) {
		int attribs[] = {
			WGL_CONTEXT_MAJOR_VERSION_ARB, 3, //we want a 3.3 context
			WGL_CONTEXT_MINOR_VERSION_ARB, 3,
			//and it shall be forward compatible so that we can only use up to date functionality
			WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
			WGL_CONTEXT_FLAGS_ARB, WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB /*| _WGL_CONTEXT_DEBUG_BIT_ARB*/,
			0
		}; //zero indicates the end of the array

		PFNWGLCREATECONTEXTATTRIBSARBPROC wglCreateContextAttribsARB = nullptr; //pointer to the method
		wglCreateContextAttribsARB = (PFNWGLCREATECONTEXTATTRIBSARBPROC)wglGetProcAddress("wglCreateContextAttribsARB");

		if (wglCreateContextAttribsARB == nullptr) //OpenGL 3.0 is not supported
		{
			wglDeleteContext(gl_display.hRC);
			gl_display.hRC = 0;
			return ERR_CANT_CREATE;
		}

		HGLRC new_hRC = wglCreateContextAttribsARB(hDC, 0, attribs);
		if (!new_hRC) {
			wglDeleteContext(gl_display.hRC);
			gl_display.hRC = 0;
			return ERR_CANT_CREATE; // Return false
		}
		wglMakeCurrent(hDC, nullptr);
		wglDeleteContext(gl_display.hRC);
		gl_display.hRC = new_hRC;

		if (!wglMakeCurrent(hDC, gl_display.hRC)) // Try To Activate The Rendering Context
		{
			wglDeleteContext(gl_display.hRC);
			gl_display.hRC = 0;
			return ERR_CANT_CREATE; // Return FALSE
		}
	}

	return OK;
}

Error GLManager_Windows::window_create(DisplayServer::WindowID p_window_id, HWND p_hwnd, HINSTANCE p_hinstance, int p_width, int p_height) {
	print_line("window_create window id " + itos(p_window_id));

	HDC hdc = GetDC(p_hwnd);
	if (!hdc) {
		return ERR_CANT_CREATE; // Return FALSE
	}

	// make sure vector is big enough...
	// we can mirror the external vector, it is simpler
	// to keep the IDs identical for fast lookup
	if (p_window_id >= (int)_windows.size()) {
		_windows.resize(p_window_id + 1);
	}

	GLWindow &win = _windows[p_window_id];
	win.in_use = true;
	win.window_id = p_window_id;
	win.width = p_width;
	win.height = p_height;
	win.hwnd = p_hwnd;
	win.hDC = hdc;

	win.gldisplay_id = _find_or_create_display(win);

	if (win.gldisplay_id == -1) {
		// release DC?
		_windows.remove(_windows.size() - 1);
		return FAILED;
	}

	// the display could be invalid .. check NYI
	GLDisplay &gl_display = _displays[win.gldisplay_id];

	// make current
	window_make_current(_windows.size() - 1);

	return OK;
}

void GLManager_Windows::_internal_set_current_window(GLWindow *p_win) {
	_current_window = p_win;
}

void GLManager_Windows::window_resize(DisplayServer::WindowID p_window_id, int p_width, int p_height) {
	get_window(p_window_id).width = p_width;
	get_window(p_window_id).height = p_height;
}

int GLManager_Windows::window_get_width(DisplayServer::WindowID p_window_id) {
	return get_window(p_window_id).width;
}

int GLManager_Windows::window_get_height(DisplayServer::WindowID p_window_id) {
	return get_window(p_window_id).height;
}

void GLManager_Windows::window_destroy(DisplayServer::WindowID p_window_id) {
	GLWindow &win = get_window(p_window_id);
	win.in_use = false;

	if (_current_window == &win) {
		_current_window = nullptr;
	}
}

void GLManager_Windows::release_current() {
	if (!_current_window)
		return;

	wglMakeCurrent(_current_window->hDC, nullptr);
}

void GLManager_Windows::window_make_current(DisplayServer::WindowID p_window_id) {
	if (p_window_id == -1)
		return;

	GLWindow &win = _windows[p_window_id];
	if (!win.in_use)
		return;

	// noop
	if (&win == _current_window)
		return;

	const GLDisplay &disp = get_display(win.gldisplay_id);
	wglMakeCurrent(win.hDC, disp.hRC);

	_internal_set_current_window(&win);
}

void GLManager_Windows::make_current() {
	if (!_current_window)
		return;
	if (!_current_window->in_use) {
		WARN_PRINT("current window not in use!");
		return;
	}
	const GLDisplay &disp = get_current_display();
	wglMakeCurrent(_current_window->hDC, disp.hRC);
}

void GLManager_Windows::swap_buffers() {
	// NO NEED TO CALL SWAP BUFFERS for each window...
	// see https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glXSwapBuffers.xml

	if (!_current_window)
		return;
	if (!_current_window->in_use) {
		WARN_PRINT("current window not in use!");
		return;
	}

	//	print_line("\tswap_buffers");

	// only for debugging without drawing anything
	//	glClearColor(Math::randf(), 0, 1, 1);
	//glClear(GL_COLOR_BUFFER_BIT);

	//	const GLDisplay &disp = get_current_display();
	SwapBuffers(_current_window->hDC);
}

Error GLManager_Windows::initialize() {
	wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");
	wglGetSwapIntervalEXT = (PFNWGLGETSWAPINTERVALEXTPROC)wglGetProcAddress("wglGetSwapIntervalEXT");
	//glWrapperInit(wrapper_get_proc_address);

	return OK;
}

void GLManager_Windows::set_use_vsync(bool p_use) {
	/*
	static bool setup = false;
	static PFNGLXSWAPINTERVALEXTPROC glXSwapIntervalEXT = nullptr;
	static PFNGLXSWAPINTERVALSGIPROC glXSwapIntervalMESA = nullptr;
	static PFNGLXSWAPINTERVALSGIPROC glXSwapIntervalSGI = nullptr;

	if (!setup) {
		setup = true;
		String extensions = glXQueryExtensionsString(x11_display, DefaultScreen(x11_display));
		if (extensions.find("GLX_EXT_swap_control") != -1)
			glXSwapIntervalEXT = (PFNGLXSWAPINTERVALEXTPROC)glXGetProcAddressARB((const GLubyte *)"glXSwapIntervalEXT");
		if (extensions.find("GLX_MESA_swap_control") != -1)
			glXSwapIntervalMESA = (PFNGLXSWAPINTERVALSGIPROC)glXGetProcAddressARB((const GLubyte *)"glXSwapIntervalMESA");
		if (extensions.find("GLX_SGI_swap_control") != -1)
			glXSwapIntervalSGI = (PFNGLXSWAPINTERVALSGIPROC)glXGetProcAddressARB((const GLubyte *)"glXSwapIntervalSGI");
	}
	int val = p_use ? 1 : 0;
	if (glXSwapIntervalMESA) {
		glXSwapIntervalMESA(val);
	} else if (glXSwapIntervalSGI) {
		glXSwapIntervalSGI(val);
	} else if (glXSwapIntervalEXT) {
		GLXDrawable drawable = glXGetCurrentDrawable();
		glXSwapIntervalEXT(x11_display, drawable, val);
	} else
		return;
	use_vsync = p_use;
	*/
}

bool GLManager_Windows::is_using_vsync() const {
	return use_vsync;
}

GLManager_Windows::GLManager_Windows(ContextType p_context_type) {
	context_type = p_context_type;

	direct_render = false;
	glx_minor = glx_major = 0;
	use_vsync = false;
	_current_window = nullptr;
}

GLManager_Windows::~GLManager_Windows() {
	release_current();
}

#endif // OPENGL_ENABLED
#endif // WINDOWS
