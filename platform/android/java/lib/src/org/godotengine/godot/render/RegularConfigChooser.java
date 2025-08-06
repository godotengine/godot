/**************************************************************************/
/*  RegularConfigChooser.java                                             */
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

package org.godotengine.godot.render;

import org.godotengine.godot.utils.GLUtils;

import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLDisplay;

/**
 * Used to select the egl config for pancake games.
 */
class RegularConfigChooser implements GLSurfaceView.EGLConfigChooser {
	private static final String TAG = RegularConfigChooser.class.getSimpleName();

	private final int[] mValue = new int[1];

	/* This EGL config specification is used to specify 3.0 rendering.
	 * We use a minimum size of 4 bits for red/green/blue, but will
	 * perform actual matching in chooseConfig() below.
	 */
	private static final int[] s_configAttribs = {
		EGL14.EGL_RED_SIZE, 4,
		EGL14.EGL_GREEN_SIZE, 4,
		EGL14.EGL_BLUE_SIZE, 4,
		// EGL14.EGL_DEPTH_SIZE,     16,
		// EGL14.EGL_STENCIL_SIZE,   EGL14.EGL_DONT_CARE,
		EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT, //apparently there is no EGL_OPENGL_ES3_BIT
		EGL14.EGL_NONE
	};

	public RegularConfigChooser(int r, int g, int b, int a, int depth, int stencil) {
		mRedSize = r;
		mGreenSize = g;
		mBlueSize = b;
		mAlphaSize = a;
		mDepthSize = depth;
		mStencilSize = stencil;
	}

	public EGLConfig chooseConfig(EGLDisplay display) {
		/* Get the number of minimally matching EGL configurations
		 */
		int[] num_config = new int[1];
		EGL14.eglChooseConfig(display, s_configAttribs, 0, null, 0, 0, num_config, 0);

		int numConfigs = num_config[0];

		if (numConfigs <= 0) {
			throw new IllegalArgumentException("No configs match configSpec");
		}

		/* Allocate then read the array of minimally matching EGL configs
		 */
		EGLConfig[] configs = new EGLConfig[numConfigs];
		EGL14.eglChooseConfig(display, s_configAttribs, 0, configs, 0, numConfigs, num_config, 0);

		if (GLUtils.DEBUG) {
			GLUtils.printConfigs(display, configs);
		}
		/* Now return the "best" one
		 */
		return chooseConfig(display, configs);
	}

	public EGLConfig chooseConfig(EGLDisplay display, EGLConfig[] configs) {
		for (EGLConfig config : configs) {
			int d = findConfigAttrib(display, config,
					EGL14.EGL_DEPTH_SIZE, 0);
			int s = findConfigAttrib(display, config,
					EGL14.EGL_STENCIL_SIZE, 0);

			// We need at least mDepthSize and mStencilSize bits
			if (d < mDepthSize || s < mStencilSize)
				continue;

			// We want an *exact* match for red/green/blue/alpha
			int r = findConfigAttrib(display, config,
					EGL14.EGL_RED_SIZE, 0);
			int g = findConfigAttrib(display, config,
					EGL14.EGL_GREEN_SIZE, 0);
			int b = findConfigAttrib(display, config,
					EGL14.EGL_BLUE_SIZE, 0);
			int a = findConfigAttrib(display, config,
					EGL14.EGL_ALPHA_SIZE, 0);

			if (r == mRedSize && g == mGreenSize && b == mBlueSize && a == mAlphaSize)
				return config;
		}
		return null;
	}

	private int findConfigAttrib(EGLDisplay display,
			EGLConfig config, int attribute, int defaultValue) {
		if (EGL14.eglGetConfigAttrib(display, config, attribute, mValue, 0)) {
			return mValue[0];
		}
		return defaultValue;
	}

	// Subclasses can adjust these values:
	protected int mRedSize;
	protected int mGreenSize;
	protected int mBlueSize;
	protected int mAlphaSize;
	protected int mDepthSize;
	protected int mStencilSize;
}
