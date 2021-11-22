/*************************************************************************/
/*  gl_manager_x11.cpp                                                   */
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

#include "gl_manager_x11.h"

#ifdef X11_ENABLED
#if defined(GLES3_ENABLED)

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define GLX_GLXEXT_PROTOTYPES
#include <GL/glx.h>
#include <GL/glxext.h>

#define GLX_CONTEXT_MAJOR_VERSION_ARB 0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB 0x2092

typedef GLXContext (*GLXCREATECONTEXTATTRIBSARBPROC)(Display *, GLXFBConfig, GLXContext, Bool, const int *);

struct GLManager_X11_Private {
	::GLXContext glx_context;
};

GLManager_X11::GLDisplay::~GLDisplay() {
	if (context) {
		//release_current();
		glXDestroyContext(x11_display, context->glx_context);
		memdelete(context);
		context = nullptr;
	}
}

static bool ctxErrorOccurred = false;
static int ctxErrorHandler(Display *dpy, XErrorEvent *ev) {
	ctxErrorOccurred = true;
	return 0;
}

int GLManager_X11::_find_or_create_display(Display *p_x11_display) {
	for (unsigned int n = 0; n < _displays.size(); n++) {
		const GLDisplay &d = _displays[n];
		if (d.x11_display == p_x11_display)
			return n;
	}

	// create
	GLDisplay d_temp;
	d_temp.x11_display = p_x11_display;
	_displays.push_back(d_temp);
	int new_display_id = _displays.size() - 1;

	// create context
	GLDisplay &d = _displays[new_display_id];

	d.context = memnew(GLManager_X11_Private);
	;
	d.context->glx_context = 0;

	//Error err = _create_context(d);
	_create_context(d);
	return new_display_id;
}

Error GLManager_X11::_create_context(GLDisplay &gl_display) {
	// some aliases
	::Display *x11_display = gl_display.x11_display;

	//const char *extensions = glXQueryExtensionsString(x11_display, DefaultScreen(x11_display));

	GLXCREATECONTEXTATTRIBSARBPROC glXCreateContextAttribsARB = (GLXCREATECONTEXTATTRIBSARBPROC)glXGetProcAddress((const GLubyte *)"glXCreateContextAttribsARB");

	ERR_FAIL_COND_V(!glXCreateContextAttribsARB, ERR_UNCONFIGURED);

	static int visual_attribs[] = {
		GLX_RENDER_TYPE, GLX_RGBA_BIT,
		GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
		GLX_DOUBLEBUFFER, true,
		GLX_RED_SIZE, 1,
		GLX_GREEN_SIZE, 1,
		GLX_BLUE_SIZE, 1,
		GLX_DEPTH_SIZE, 24,
		None
	};

	static int visual_attribs_layered[] = {
		GLX_RENDER_TYPE, GLX_RGBA_BIT,
		GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
		GLX_DOUBLEBUFFER, true,
		GLX_RED_SIZE, 8,
		GLX_GREEN_SIZE, 8,
		GLX_BLUE_SIZE, 8,
		GLX_ALPHA_SIZE, 8,
		GLX_DEPTH_SIZE, 24,
		None
	};

	int fbcount;
	GLXFBConfig fbconfig = 0;
	XVisualInfo *vi = nullptr;

	gl_display.x_swa.event_mask = StructureNotifyMask;
	gl_display.x_swa.border_pixel = 0;
	gl_display.x_valuemask = CWBorderPixel | CWColormap | CWEventMask;

	if (OS::get_singleton()->is_layered_allowed()) {
		GLXFBConfig *fbc = glXChooseFBConfig(x11_display, DefaultScreen(x11_display), visual_attribs_layered, &fbcount);
		ERR_FAIL_COND_V(!fbc, ERR_UNCONFIGURED);

		for (int i = 0; i < fbcount; i++) {
			vi = (XVisualInfo *)glXGetVisualFromFBConfig(x11_display, fbc[i]);
			if (!vi)
				continue;

			XRenderPictFormat *pict_format = XRenderFindVisualFormat(x11_display, vi->visual);
			if (!pict_format) {
				XFree(vi);
				vi = nullptr;
				continue;
			}

			fbconfig = fbc[i];
			if (pict_format->direct.alphaMask > 0) {
				break;
			}
		}
		XFree(fbc);

		ERR_FAIL_COND_V(!fbconfig, ERR_UNCONFIGURED);

		gl_display.x_swa.background_pixmap = None;
		gl_display.x_swa.background_pixel = 0;
		gl_display.x_swa.border_pixmap = None;
		gl_display.x_valuemask |= CWBackPixel;

	} else {
		GLXFBConfig *fbc = glXChooseFBConfig(x11_display, DefaultScreen(x11_display), visual_attribs, &fbcount);
		ERR_FAIL_COND_V(!fbc, ERR_UNCONFIGURED);

		vi = glXGetVisualFromFBConfig(x11_display, fbc[0]);

		fbconfig = fbc[0];
		XFree(fbc);
	}

	int (*oldHandler)(Display *, XErrorEvent *) = XSetErrorHandler(&ctxErrorHandler);

	switch (context_type) {
		case GLES_3_0_COMPATIBLE: {
			static int context_attribs[] = {
				GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
				GLX_CONTEXT_MINOR_VERSION_ARB, 3,
				GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
				GLX_CONTEXT_FLAGS_ARB, GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB /*|GLX_CONTEXT_DEBUG_BIT_ARB*/,
				None
			};

			gl_display.context->glx_context = glXCreateContextAttribsARB(x11_display, fbconfig, nullptr, true, context_attribs);
			ERR_FAIL_COND_V(ctxErrorOccurred || !gl_display.context->glx_context, ERR_UNCONFIGURED);
		} break;
	}

	gl_display.x_swa.colormap = XCreateColormap(x11_display, RootWindow(x11_display, vi->screen), vi->visual, AllocNone);

	XSync(x11_display, False);
	XSetErrorHandler(oldHandler);

	// make our own copy of the vi data
	// for later creating windows using this display
	if (vi) {
		gl_display.x_vi = *vi;
	}

	XFree(vi);

	return OK;
}

Error GLManager_X11::window_create(DisplayServer::WindowID p_window_id, ::Window p_window, Display *p_display, int p_width, int p_height) {
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
	win.x11_window = p_window;
	win.gldisplay_id = _find_or_create_display(p_display);

	// the display could be invalid .. check NYI
	GLDisplay &gl_display = _displays[win.gldisplay_id];
	//const XVisualInfo &vi = gl_display.x_vi;
	//XSetWindowAttributes &swa = gl_display.x_swa;
	::Display *x11_display = gl_display.x11_display;
	::Window &x11_window = win.x11_window;

	if (!glXMakeCurrent(x11_display, x11_window, gl_display.context->glx_context)) {
		ERR_PRINT("glXMakeCurrent failed");
	}

	_internal_set_current_window(&win);

	return OK;
}

void GLManager_X11::_internal_set_current_window(GLWindow *p_win) {
	_current_window = p_win;

	// quick access to x info
	_x_windisp.x11_window = _current_window->x11_window;
	const GLDisplay &disp = get_current_display();
	_x_windisp.x11_display = disp.x11_display;
}

void GLManager_X11::window_resize(DisplayServer::WindowID p_window_id, int p_width, int p_height) {
	get_window(p_window_id).width = p_width;
	get_window(p_window_id).height = p_height;
}

void GLManager_X11::window_destroy(DisplayServer::WindowID p_window_id) {
	GLWindow &win = get_window(p_window_id);
	win.in_use = false;

	if (_current_window == &win) {
		_current_window = nullptr;
		_x_windisp.x11_display = nullptr;
		_x_windisp.x11_window = -1;
	}
}

void GLManager_X11::release_current() {
	if (!_current_window)
		return;
	glXMakeCurrent(_x_windisp.x11_display, None, nullptr);
}

void GLManager_X11::window_make_current(DisplayServer::WindowID p_window_id) {
	if (p_window_id == -1)
		return;

	GLWindow &win = _windows[p_window_id];
	if (!win.in_use)
		return;

	// noop
	if (&win == _current_window)
		return;

	const GLDisplay &disp = get_display(win.gldisplay_id);

	glXMakeCurrent(disp.x11_display, win.x11_window, disp.context->glx_context);

	_internal_set_current_window(&win);
}

void GLManager_X11::make_current() {
	if (!_current_window)
		return;
	if (!_current_window->in_use) {
		WARN_PRINT("current window not in use!");
		return;
	}
	const GLDisplay &disp = get_current_display();
	glXMakeCurrent(_x_windisp.x11_display, _x_windisp.x11_window, disp.context->glx_context);
}

void GLManager_X11::swap_buffers() {
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

	//const GLDisplay &disp = get_current_display();
	glXSwapBuffers(_x_windisp.x11_display, _x_windisp.x11_window);
}

Error GLManager_X11::initialize() {
	return OK;
}

void GLManager_X11::set_use_vsync(bool p_use) {
	static bool setup = false;
	static PFNGLXSWAPINTERVALEXTPROC glXSwapIntervalEXT = nullptr;
	static PFNGLXSWAPINTERVALSGIPROC glXSwapIntervalMESA = nullptr;
	static PFNGLXSWAPINTERVALSGIPROC glXSwapIntervalSGI = nullptr;

	// force vsync in the editor for now, as a safety measure
	bool is_editor = Engine::get_singleton()->is_editor_hint();
	if (is_editor) {
		p_use = true;
	}

	// we need an active window to get a display to set the vsync
	if (!_current_window)
		return;
	const GLDisplay &disp = get_current_display();

	if (!setup) {
		setup = true;
		String extensions = glXQueryExtensionsString(disp.x11_display, DefaultScreen(disp.x11_display));
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
		glXSwapIntervalEXT(disp.x11_display, drawable, val);
	} else
		return;
	use_vsync = p_use;
}

bool GLManager_X11::is_using_vsync() const {
	return use_vsync;
}

GLManager_X11::GLManager_X11(const Vector2i &p_size, ContextType p_context_type) {
	context_type = p_context_type;

	double_buffer = false;
	direct_render = false;
	glx_minor = glx_major = 0;
	use_vsync = false;
	_current_window = nullptr;
}

GLManager_X11::~GLManager_X11() {
	release_current();
}

#endif
#endif
