/*************************************************************************/
/*  context_gl_windows_angle.cpp                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#if defined(OPENGL_ENABLED) || defined(GLES_ENABLED)

// Author: Juan Linietsky <reduzio@gmail.com>, (C) 2008

#include "context_gl_windows_angle.h"
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#define WGL_EXT_swap_control 1
#include <WGL/wgl.h>

#include <dwmapi.h>

#define WGL_CONTEXT_MAJOR_VERSION_ARB 0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB 0x2092
#define WGL_CONTEXT_FLAGS_ARB 0x2094
#define WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB 0x00000002
#define WGL_CONTEXT_PROFILE_MASK_ARB 0x9126
#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB 0x00000001

#if defined(__GNUC__)
// Workaround GCC warning from -Wcast-function-type.
#define wglGetProcAddress (void *)wglGetProcAddress
#endif

typedef HGLRC(APIENTRY *PFNWGLCREATECONTEXTATTRIBSARBPROC)(HDC, HGLRC, const int *);

void ContextGL_Windows::release_current() {
	eglMakeCurrent(mEglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, mEglContext);
}

void ContextGL_Windows::make_current() {
	eglMakeCurrent(mEglDisplay, mEglSurface, mEglSurface, mEglContext);
}

HDC ContextGL_Windows::get_hdc() {
	return hDC;
}

HGLRC ContextGL_Windows::get_hglrc() {
	return hRC;
}

int ContextGL_Windows::get_window_width() {

	return OS::get_singleton()->get_video_mode().width;
}

int ContextGL_Windows::get_window_height() {

	return OS::get_singleton()->get_video_mode().height;
}

bool ContextGL_Windows::should_vsync_via_compositor() {

	if (OS::get_singleton()->is_window_fullscreen() || !OS::get_singleton()->is_vsync_via_compositor_enabled()) {
		return false;
	}

	// Note: All Windows versions supported by Godot have a compositor.
	// It can be disabled on earlier Windows versions.
	BOOL dwm_enabled;

	if (SUCCEEDED(DwmIsCompositionEnabled(&dwm_enabled))) {
		return dwm_enabled;
	}

	return false;
}

void ContextGL_Windows::swap_buffers() {

	if (eglSwapBuffers(mEglDisplay, mEglSurface) != EGL_TRUE) {
		cleanup();

		initialize();

		// tell rasterizer to reload textures and stuff?
	}

	if (use_vsync) {
		bool vsync_via_compositor_now = should_vsync_via_compositor();

		if (vsync_via_compositor_now && wglGetSwapIntervalEXT() == 0) {
			DwmFlush();
		}

		if (vsync_via_compositor_now != vsync_via_compositor) {
			// The previous frame had a different operating mode than this
			// frame.  Set the 'vsync_via_compositor' member variable and the
			// OpenGL swap interval to their proper values.
			set_use_vsync(true);
		}
	}
}

void ContextGL_Windows::set_use_vsync(bool p_use) {
	use_vsync = p_use;
	if (!p_use) {
		eglSwapInterval(mEglDisplay, 0);
	} else {
		eglSwapInterval(mEglDisplay, 1);
	}
}

bool ContextGL_Windows::is_using_vsync() const {

	return use_vsync;
}

void ContextGL_Windows::cleanup() {
	if (mEglDisplay != EGL_NO_DISPLAY && mEglSurface != EGL_NO_SURFACE) {
		eglDestroySurface(mEglDisplay, mEglSurface);
		mEglSurface = EGL_NO_SURFACE;
	}

	if (mEglDisplay != EGL_NO_DISPLAY && mEglContext != EGL_NO_CONTEXT) {
		eglDestroyContext(mEglDisplay, mEglContext);
		mEglContext = EGL_NO_CONTEXT;
	}

	if (mEglDisplay != EGL_NO_DISPLAY) {
		eglTerminate(mEglDisplay);
		mEglDisplay = EGL_NO_DISPLAY;
	}
};

#define _WGL_CONTEXT_DEBUG_BIT_ARB 0x0001

EGLDisplay get_egl_display(EGLint platform_type) {
	EGLDisplay display = EGL_NO_DISPLAY;
	if (platform_type == EGL_PLATFORM_ANGLE_TYPE_D3D11_ANGLE) {
		EGLint display_attributes[] = {
			/*EGL_PLATFORM_ANGLE_TYPE_ANGLE, EGL_PLATFORM_ANGLE_TYPE_D3D11_ANGLE,
			EGL_PLATFORM_ANGLE_MAX_VERSION_MAJOR_ANGLE, 9,
			EGL_PLATFORM_ANGLE_MAX_VERSION_MINOR_ANGLE, 3,
			EGL_NONE,*/
			// These are the default display attributes, used to request ANGLE's D3D11 renderer.
			// eglInitialize will only succeed with these attributes if the hardware supports D3D11 Feature Level 10_0+.
			EGL_PLATFORM_ANGLE_TYPE_ANGLE,
			platform_type,

			EGL_EXPERIMENTAL_PRESENT_PATH_ANGLE,
			EGL_EXPERIMENTAL_PRESENT_PATH_FAST_ANGLE,

			EGL_NONE,
		};
		display = eglGetPlatformDisplayEXT(EGL_PLATFORM_ANGLE_ANGLE, EGL_DEFAULT_DISPLAY, display_attributes);
	} else {
		EGLint display_attributes[] = {
			/*EGL_PLATFORM_ANGLE_TYPE_ANGLE, EGL_PLATFORM_ANGLE_TYPE_D3D11_ANGLE,
			EGL_PLATFORM_ANGLE_MAX_VERSION_MAJOR_ANGLE, 9,
			EGL_PLATFORM_ANGLE_MAX_VERSION_MINOR_ANGLE, 3,
			EGL_NONE,*/
			// These are the default display attributes, used to request ANGLE's D3D11 renderer.
			// eglInitialize will only succeed with these attributes if the hardware supports D3D11 Feature Level 10_0+.
			EGL_PLATFORM_ANGLE_TYPE_ANGLE,
			platform_type,

			EGL_NONE,
		};
		display = eglGetPlatformDisplayEXT(EGL_PLATFORM_ANGLE_ANGLE, EGL_DEFAULT_DISPLAY, display_attributes);
	};
	return display;
}

Error ContextGL_Windows::initialize() {

	EGLint configAttribList[] = {
		EGL_RED_SIZE, 8,
		EGL_GREEN_SIZE, 8,
		EGL_BLUE_SIZE, 8,
		EGL_ALPHA_SIZE, 8,
		EGL_DEPTH_SIZE, 8,
		EGL_STENCIL_SIZE, 8,
		EGL_SAMPLE_BUFFERS, 0,
		EGL_NONE
	};

	EGLint surfaceAttribList[] = {
		EGL_NONE, EGL_NONE
	};

	EGLint numConfigs = 0;
	EGLint majorVersion = 1;
	EGLint minorVersion;
	minorVersion = 5;
	EGLDisplay display = EGL_NO_DISPLAY;
	EGLContext context = EGL_NO_CONTEXT;
	EGLSurface surface = EGL_NO_SURFACE;
	EGLConfig config = nullptr;
	EGLint contextAttribs[3];

	contextAttribs[0] = EGL_CONTEXT_CLIENT_VERSION;
	contextAttribs[1] = 3;
	contextAttribs[2] = EGL_NONE;

	try {

		int platform_type = EGL_PLATFORM_ANGLE_TYPE_VULKAN_ANGLE;

		List<String> args = OS::get_singleton()->get_cmdline_args();
		List<String>::Element *I = args.front();
		bool backend_forced_by_user = false;
		while (I) {
			if (I->get() == "--angle-backend") {
				String backend = I->next()->get().to_lower();
				if (backend == "vulkan") {
					platform_type = EGL_PLATFORM_ANGLE_TYPE_VULKAN_ANGLE;
					backend_forced_by_user = true;
				} else if (backend == "d3d11") {
					platform_type = EGL_PLATFORM_ANGLE_TYPE_D3D11_ANGLE;
					backend_forced_by_user = true;
				} else if (backend == "opengl") {
					platform_type = EGL_PLATFORM_ANGLE_TYPE_OPENGL_ANGLE;
					backend_forced_by_user = true;
				}
			}
			I = I->next();
		}

		display = EGL_NO_DISPLAY;

		PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(eglGetProcAddress("eglGetPlatformDisplayEXT"));

		if (!eglGetPlatformDisplayEXT) {
			throw "Failed to get function eglGetPlatformDisplayEXT";
		}

		// When the user doesn't force a specific backend through the command line argument
		// --angle-backend we try to use vulkan, d3d11 and opengl in that order.
		if (!backend_forced_by_user) {
			const EGLint platform_fallbacks[] = {
				EGL_PLATFORM_ANGLE_TYPE_VULKAN_ANGLE,
				EGL_PLATFORM_ANGLE_TYPE_D3D11_ANGLE,
				EGL_PLATFORM_ANGLE_TYPE_OPENGL_ANGLE,
			};

			for (int i = 0; i < sizeof(platform_fallbacks); i++) {
				platform_type = platform_fallbacks[i];
				display = get_egl_display(platform_type);
				if (display != EGL_NO_DISPLAY) {
					break;
				}
			}
		} else {
			display = get_egl_display(platform_type);
		}

		if (display == EGL_NO_DISPLAY) {
			throw "Failed to get EGL display";
		}

		if (eglInitialize(display, &majorVersion, &minorVersion) == EGL_FALSE) {
			throw "Failed to initialize EGL";
		}

		if (eglGetConfigs(display, NULL, 0, &numConfigs) == EGL_FALSE) {
			throw "Failed to get EGLConfig count";
		}

		if (eglChooseConfig(display, configAttribList, &config, 1, &numConfigs) == EGL_FALSE) {
			throw "Failed to choose first EGLConfig count";
		}

		surface = eglCreateWindowSurface(display, config, hWnd, surfaceAttribList);
		if (surface == EGL_NO_SURFACE) {
			throw "Failed to create EGL fullscreen surface";
		}

		context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttribs);
		if (context == EGL_NO_CONTEXT) {
			throw "Failed to create EGL context";
		}

		if (eglMakeCurrent(display, surface, surface, context) == EGL_FALSE) {
			throw "Failed to make fullscreen EGLSurface current";
		}
	} catch (const char *err) {
		print_error(String(err));
		return FAILED;
	};

	mEglDisplay = display;
	mEglSurface = surface;
	mEglContext = context;

	eglQuerySurface(display, surface, EGL_WIDTH, &width);
	eglQuerySurface(display, surface, EGL_HEIGHT, &height);

	return OK;
}

ContextGL_Windows::ContextGL_Windows(HWND hwnd, bool p_opengl_3_context) {

	opengl_3_context = p_opengl_3_context;
	hWnd = hwnd;
	use_vsync = false;
	vsync_via_compositor = false;
}

ContextGL_Windows::~ContextGL_Windows() {
}

#endif
