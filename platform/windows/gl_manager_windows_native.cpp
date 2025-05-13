/**************************************************************************/
/*  gl_manager_windows_native.cpp                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "gl_manager_windows_native.h"

#if defined(WINDOWS_ENABLED) && defined(GLES3_ENABLED)

#include "core/config/project_settings.h"
#include "core/version.h"

#include "nvapi_profile.h"

#include <dwmapi.h>
#include <cstdio>
#include <cstdlib>

#define WGL_CONTEXT_MAJOR_VERSION_ARB 0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB 0x2092
#define WGL_CONTEXT_FLAGS_ARB 0x2094
#define WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB 0x00000002
#define WGL_CONTEXT_PROFILE_MASK_ARB 0x9126
#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB 0x00000001

#define _WGL_CONTEXT_DEBUG_BIT_ARB 0x0001

typedef HGLRC(APIENTRY *PFNWGLCREATECONTEXT)(HDC);
typedef BOOL(APIENTRY *PFNWGLDELETECONTEXT)(HGLRC);
typedef BOOL(APIENTRY *PFNWGLMAKECURRENT)(HDC, HGLRC);
typedef HGLRC(APIENTRY *PFNWGLCREATECONTEXTATTRIBSARBPROC)(HDC, HGLRC, const int *);
typedef void *(APIENTRY *PFNWGLGETPROCADDRESS)(LPCSTR);

static String format_error_message(DWORD id) {
	LPWSTR messageBuffer = nullptr;
	size_t size = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
			nullptr, id, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPWSTR)&messageBuffer, 0, nullptr);

	String msg = "Error " + itos(id) + ": " + String::utf16((const char16_t *)messageBuffer, size);

	LocalFree(messageBuffer);

	return msg;
}

int GLManagerNative_Windows::_find_or_create_display(GLWindow &win) {
	// find display NYI, only 1 supported so far
	if (_displays.size()) {
		return 0;
	}

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
		_displays.remove_at(new_display_id);
		return -1;
	}

	return new_display_id;
}

static Error _configure_pixel_format(HDC hDC) {
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

	return OK;
}

PFNWGLCREATECONTEXT gd_wglCreateContext;
PFNWGLMAKECURRENT gd_wglMakeCurrent;
PFNWGLDELETECONTEXT gd_wglDeleteContext;
PFNWGLGETPROCADDRESS gd_wglGetProcAddress;

Error GLManagerNative_Windows::_create_context(GLWindow &win, GLDisplay &gl_display) {
	Error err = _configure_pixel_format(win.hDC);
	if (err != OK) {
		return err;
	}

	HMODULE module = LoadLibraryW(L"opengl32.dll");
	if (!module) {
		return ERR_CANT_CREATE;
	}
	gd_wglCreateContext = (PFNWGLCREATECONTEXT)(void *)GetProcAddress(module, "wglCreateContext");
	gd_wglMakeCurrent = (PFNWGLMAKECURRENT)(void *)GetProcAddress(module, "wglMakeCurrent");
	gd_wglDeleteContext = (PFNWGLDELETECONTEXT)(void *)GetProcAddress(module, "wglDeleteContext");
	gd_wglGetProcAddress = (PFNWGLGETPROCADDRESS)(void *)GetProcAddress(module, "wglGetProcAddress");
	if (!gd_wglCreateContext || !gd_wglMakeCurrent || !gd_wglDeleteContext || !gd_wglGetProcAddress) {
		return ERR_CANT_CREATE;
	}

	gl_display.hRC = gd_wglCreateContext(win.hDC);
	if (!gl_display.hRC) // Are We Able To Get A Rendering Context?
	{
		return ERR_CANT_CREATE; // Return FALSE
	}

	if (!gd_wglMakeCurrent(win.hDC, gl_display.hRC)) {
		ERR_PRINT("Could not attach OpenGL context to newly created window: " + format_error_message(GetLastError()));
	}

	int attribs[] = {
		WGL_CONTEXT_MAJOR_VERSION_ARB, 3, //we want a 3.3 context
		WGL_CONTEXT_MINOR_VERSION_ARB, 3,
		//and it shall be forward compatible so that we can only use up to date functionality
		WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
		WGL_CONTEXT_FLAGS_ARB, WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB /*| _WGL_CONTEXT_DEBUG_BIT_ARB*/,
		0
	}; //zero indicates the end of the array

	PFNWGLCREATECONTEXTATTRIBSARBPROC wglCreateContextAttribsARB = nullptr; //pointer to the method
	wglCreateContextAttribsARB = (PFNWGLCREATECONTEXTATTRIBSARBPROC)gd_wglGetProcAddress("wglCreateContextAttribsARB");

	if (wglCreateContextAttribsARB == nullptr) //OpenGL 3.0 is not supported
	{
		gd_wglDeleteContext(gl_display.hRC);
		gl_display.hRC = nullptr;
		return ERR_CANT_CREATE;
	}

	HGLRC new_hRC = wglCreateContextAttribsARB(win.hDC, nullptr, attribs);
	if (!new_hRC) {
		gd_wglDeleteContext(gl_display.hRC);
		gl_display.hRC = nullptr;
		return ERR_CANT_CREATE;
	}

	if (!gd_wglMakeCurrent(win.hDC, nullptr)) {
		ERR_PRINT("Could not detach OpenGL context from newly created window: " + format_error_message(GetLastError()));
	}

	gd_wglDeleteContext(gl_display.hRC);
	gl_display.hRC = new_hRC;

	if (!gd_wglMakeCurrent(win.hDC, gl_display.hRC)) // Try to activate the rendering context.
	{
		ERR_PRINT("Could not attach OpenGL context to newly created window with replaced OpenGL context: " + format_error_message(GetLastError()));
		gd_wglDeleteContext(gl_display.hRC);
		gl_display.hRC = nullptr;
		return ERR_CANT_CREATE;
	}

	if (!wglSwapIntervalEXT) {
		wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC)gd_wglGetProcAddress("wglSwapIntervalEXT");
	}

	return OK;
}

Error GLManagerNative_Windows::window_create(DisplayServer::WindowID p_window_id, HWND p_hwnd, HINSTANCE p_hinstance, int p_width, int p_height) {
	HDC hDC = GetDC(p_hwnd);
	if (!hDC) {
		return ERR_CANT_CREATE;
	}

	// configure the HDC to use a compatible pixel format
	Error result = _configure_pixel_format(hDC);
	if (result != OK) {
		return result;
	}

	GLWindow win;
	win.hwnd = p_hwnd;
	win.hDC = hDC;

	win.gldisplay_id = _find_or_create_display(win);

	if (win.gldisplay_id == -1) {
		return FAILED;
	}

	// WARNING: `p_window_id` is an eternally growing integer since popup windows keep coming and going
	// and each of them has a higher id than the previous, so it must be used in a map not a vector.
	_windows[p_window_id] = win;

	// make current
	window_make_current(p_window_id);

	return OK;
}

void GLManagerNative_Windows::window_destroy(DisplayServer::WindowID p_window_id) {
	GLWindow &win = get_window(p_window_id);
	if (_current_window == &win) {
		_current_window = nullptr;
	}
	_windows.erase(p_window_id);
}

void GLManagerNative_Windows::release_current() {
	if (!_current_window) {
		return;
	}

	if (!gd_wglMakeCurrent(_current_window->hDC, nullptr)) {
		ERR_PRINT("Could not detach OpenGL context from window marked current: " + format_error_message(GetLastError()));
	}

	_current_window = nullptr;
}

void GLManagerNative_Windows::window_make_current(DisplayServer::WindowID p_window_id) {
	if (p_window_id == -1) {
		return;
	}

	// crash if our data structures are out of sync, i.e. not found
	GLWindow &win = _windows[p_window_id];

	// noop
	if (&win == _current_window) {
		return;
	}

	const GLDisplay &disp = get_display(win.gldisplay_id);
	if (!gd_wglMakeCurrent(win.hDC, disp.hRC)) {
		ERR_PRINT("Could not switch OpenGL context to other window: " + format_error_message(GetLastError()));
	}

	_current_window = &win;
}

void GLManagerNative_Windows::swap_buffers() {
	SwapBuffers(_current_window->hDC);
}

Error GLManagerNative_Windows::initialize() {
	Dictionary props;
	int thread_control_val = OGL_THREAD_CONTROL_DISABLE;
	if (!GLOBAL_GET("rendering/gl_compatibility/nvidia_disable_threaded_optimization")) {
		thread_control_val = OGL_THREAD_CONTROL_ENABLE;
	}
	props[OGL_THREAD_CONTROL_ID] = thread_control_val;
	props[VRR_MODE_ID] = VRR_MODE_FULLSCREEN_ONLY;
	if (nvapi_setup_profile(props)) {
		if (thread_control_val == OGL_THREAD_CONTROL_DISABLE) {
			print_verbose("NVAPI: Disabled OpenGL threaded optimization successfully");
		} else {
			print_verbose("NVAPI: Enabled OpenGL threaded optimization successfully");
		}
		print_verbose("NVAPI: Disabled G-SYNC for windowed mode successfully");
	}
	return OK;
}

void GLManagerNative_Windows::set_use_vsync(DisplayServer::WindowID p_window_id, bool p_use) {
	GLWindow &win = get_window(p_window_id);

	if (&win != _current_window) {
		window_make_current(p_window_id);
	}

	if (wglSwapIntervalEXT) {
		win.use_vsync = p_use;

		if (!wglSwapIntervalEXT(p_use ? 1 : 0)) {
			WARN_PRINT_ONCE("Could not set V-Sync mode, as changing V-Sync mode is not supported by the graphics driver.");
		}
	} else {
		WARN_PRINT_ONCE("Could not set V-Sync mode, as changing V-Sync mode is not supported by the graphics driver.");
	}
}

bool GLManagerNative_Windows::is_using_vsync(DisplayServer::WindowID p_window_id) const {
	return get_window(p_window_id).use_vsync;
}

HDC GLManagerNative_Windows::get_hdc(DisplayServer::WindowID p_window_id) {
	return get_window(p_window_id).hDC;
}

HGLRC GLManagerNative_Windows::get_hglrc(DisplayServer::WindowID p_window_id) {
	const GLWindow &win = get_window(p_window_id);
	const GLDisplay &disp = get_display(win.gldisplay_id);
	return disp.hRC;
}

GLManagerNative_Windows::GLManagerNative_Windows() {
	direct_render = false;
	glx_minor = glx_major = 0;
	_current_window = nullptr;
}

GLManagerNative_Windows::~GLManagerNative_Windows() {
	release_current();
}

#endif // WINDOWS_ENABLED && GLES3_ENABLED
