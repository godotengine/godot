/**************************************************************************/
/*  context_egl_uwp.cpp                                                   */
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

#include "context_egl_uwp.h"

#include "EGL/eglext.h"

using Platform::Exception;

void ContextEGL_UWP::release_current() {
	eglMakeCurrent(mEglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, mEglContext);
};

void ContextEGL_UWP::make_current() {
	eglMakeCurrent(mEglDisplay, mEglSurface, mEglSurface, mEglContext);
};

int ContextEGL_UWP::get_window_width() {
	return width;
};

int ContextEGL_UWP::get_window_height() {
	return height;
};

void ContextEGL_UWP::reset() {
	cleanup();

	window = CoreWindow::GetForCurrentThread();
	initialize();
};

void ContextEGL_UWP::swap_buffers() {
	if (eglSwapBuffers(mEglDisplay, mEglSurface) != EGL_TRUE) {
		cleanup();

		window = CoreWindow::GetForCurrentThread();
		initialize();

		// tell rasterizer to reload textures and stuff?
	}
};

Error ContextEGL_UWP::initialize() {
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
	if (driver == GLES_2_0) {
		minorVersion = 0;
	} else {
		minorVersion = 5;
	}
	EGLDisplay display = EGL_NO_DISPLAY;
	EGLContext context = EGL_NO_CONTEXT;
	EGLSurface surface = EGL_NO_SURFACE;
	EGLConfig config = nullptr;
	EGLint contextAttribs[3];
	if (driver == GLES_2_0) {
		contextAttribs[0] = EGL_CONTEXT_CLIENT_VERSION;
		contextAttribs[1] = 2;
		contextAttribs[2] = EGL_NONE;
	} else {
		contextAttribs[0] = EGL_CONTEXT_CLIENT_VERSION;
		contextAttribs[1] = 3;
		contextAttribs[2] = EGL_NONE;
	}

	try {
		const EGLint displayAttributes[] = {
			/*EGL_PLATFORM_ANGLE_TYPE_ANGLE, EGL_PLATFORM_ANGLE_TYPE_D3D11_ANGLE,
			EGL_PLATFORM_ANGLE_MAX_VERSION_MAJOR_ANGLE, 9,
			EGL_PLATFORM_ANGLE_MAX_VERSION_MINOR_ANGLE, 3,
			EGL_NONE,*/
			// These are the default display attributes, used to request ANGLE's D3D11 renderer.
			// eglInitialize will only succeed with these attributes if the hardware supports D3D11 Feature Level 10_0+.
			EGL_PLATFORM_ANGLE_TYPE_ANGLE,
			EGL_PLATFORM_ANGLE_TYPE_D3D11_ANGLE,

#ifdef EGL_ANGLE_DISPLAY_ALLOW_RENDER_TO_BACK_BUFFER
			// EGL_ANGLE_DISPLAY_ALLOW_RENDER_TO_BACK_BUFFER is an optimization that can have large performance benefits on mobile devices.
			// Its syntax is subject to change, though. Please update your Visual Studio templates if you experience compilation issues with it.
			EGL_ANGLE_DISPLAY_ALLOW_RENDER_TO_BACK_BUFFER,
			EGL_TRUE,
#endif

			// EGL_PLATFORM_ANGLE_ENABLE_AUTOMATIC_TRIM_ANGLE is an option that enables ANGLE to automatically call
			// the IDXGIDevice3::Trim method on behalf of the application when it gets suspended.
			// Calling IDXGIDevice3::Trim when an application is suspended is a Windows Store application certification requirement.
			EGL_PLATFORM_ANGLE_ENABLE_AUTOMATIC_TRIM_ANGLE,
			EGL_TRUE,
			EGL_NONE,
		};

		PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(eglGetProcAddress("eglGetPlatformDisplayEXT"));

		if (!eglGetPlatformDisplayEXT) {
			throw Exception::CreateException(E_FAIL, L"Failed to get function eglGetPlatformDisplayEXT");
		}

		display = eglGetPlatformDisplayEXT(EGL_PLATFORM_ANGLE_ANGLE, EGL_DEFAULT_DISPLAY, displayAttributes);

		if (display == EGL_NO_DISPLAY) {
			throw Exception::CreateException(E_FAIL, L"Failed to get default EGL display");
		}

		if (eglInitialize(display, &majorVersion, &minorVersion) == EGL_FALSE) {
			throw Exception::CreateException(E_FAIL, L"Failed to initialize EGL");
		}

		if (eglGetConfigs(display, NULL, 0, &numConfigs) == EGL_FALSE) {
			throw Exception::CreateException(E_FAIL, L"Failed to get EGLConfig count");
		}

		if (eglChooseConfig(display, configAttribList, &config, 1, &numConfigs) == EGL_FALSE) {
			throw Exception::CreateException(E_FAIL, L"Failed to choose first EGLConfig count");
		}

		surface = eglCreateWindowSurface(display, config, reinterpret_cast<IInspectable *>(window), surfaceAttribList);
		if (surface == EGL_NO_SURFACE) {
			throw Exception::CreateException(E_FAIL, L"Failed to create EGL fullscreen surface");
		}

		context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttribs);
		if (context == EGL_NO_CONTEXT) {
			throw Exception::CreateException(E_FAIL, L"Failed to create EGL context");
		}

		if (eglMakeCurrent(display, surface, surface, context) == EGL_FALSE) {
			throw Exception::CreateException(E_FAIL, L"Failed to make fullscreen EGLSurface current");
		}
	} catch (...) {
		return FAILED;
	};

	mEglDisplay = display;
	mEglSurface = surface;
	mEglContext = context;

	eglQuerySurface(display, surface, EGL_WIDTH, &width);
	eglQuerySurface(display, surface, EGL_HEIGHT, &height);

	return OK;
};

void ContextEGL_UWP::cleanup() {
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

ContextEGL_UWP::ContextEGL_UWP(CoreWindow ^ p_window, Driver p_driver) :
		mEglDisplay(EGL_NO_DISPLAY),
		mEglContext(EGL_NO_CONTEXT),
		mEglSurface(EGL_NO_SURFACE),
		driver(p_driver),
		window(p_window) {}

ContextEGL_UWP::~ContextEGL_UWP() {
	cleanup();
};
