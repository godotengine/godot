/**************************************************************************/
/*  RegularContextFactory.java                                            */
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

package org.godotengine.godot.xr.regular;

import org.godotengine.godot.gl.GLSurfaceView;
import org.godotengine.godot.utils.GLUtils;

import android.util.Log;

import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.EGLContext;
import javax.microedition.khronos.egl.EGLDisplay;

/**
 * Factory used to setup the opengl context for pancake games.
 */
public class RegularContextFactory implements GLSurfaceView.EGLContextFactory {
	private static final String TAG = RegularContextFactory.class.getSimpleName();

	private static final int _EGL_CONTEXT_FLAGS_KHR = 0x30FC;
	private static final int _EGL_CONTEXT_OPENGL_DEBUG_BIT_KHR = 0x00000001;

	private static int EGL_CONTEXT_CLIENT_VERSION = 0x3098;

	private final boolean mUseDebugOpengl;

	public RegularContextFactory() {
		this(false);
	}

	public RegularContextFactory(boolean useDebugOpengl) {
		this.mUseDebugOpengl = useDebugOpengl;
	}

	public EGLContext createContext(EGL10 egl, EGLDisplay display, EGLConfig eglConfig) {
		Log.w(TAG, "creating OpenGL ES 3.0 context :");

		GLUtils.checkEglError(TAG, "Before eglCreateContext", egl);
		EGLContext context;
		if (mUseDebugOpengl) {
			int[] attrib_list = { EGL_CONTEXT_CLIENT_VERSION, 3, _EGL_CONTEXT_FLAGS_KHR, _EGL_CONTEXT_OPENGL_DEBUG_BIT_KHR, EGL10.EGL_NONE };
			context = egl.eglCreateContext(display, eglConfig, EGL10.EGL_NO_CONTEXT, attrib_list);
		} else {
			int[] attrib_list = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL10.EGL_NONE };
			context = egl.eglCreateContext(display, eglConfig, EGL10.EGL_NO_CONTEXT, attrib_list);
		}
		GLUtils.checkEglError(TAG, "After eglCreateContext", egl);
		return context;
	}

	public void destroyContext(EGL10 egl, EGLDisplay display, EGLContext context) {
		egl.eglDestroyContext(display, context);
	}
}
