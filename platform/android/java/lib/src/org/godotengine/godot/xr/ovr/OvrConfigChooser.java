/*************************************************************************/
/*  OvrConfigChooser.java                                                */
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

package org.godotengine.godot.xr.ovr;

import android.opengl.EGLExt;
import android.opengl.GLSurfaceView;

import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.EGLDisplay;

/**
 * EGL config chooser for the Oculus Mobile VR SDK.
 */
public class OvrConfigChooser implements GLSurfaceView.EGLConfigChooser {
	private static final int[] CONFIG_ATTRIBS = {
		EGL10.EGL_RED_SIZE, 8,
		EGL10.EGL_GREEN_SIZE, 8,
		EGL10.EGL_BLUE_SIZE, 8,
		EGL10.EGL_ALPHA_SIZE, 8, // Need alpha for the multi-pass timewarp compositor
		EGL10.EGL_DEPTH_SIZE, 0,
		EGL10.EGL_STENCIL_SIZE, 0,
		EGL10.EGL_SAMPLES, 0,
		EGL10.EGL_NONE
	};

	@Override
	public EGLConfig chooseConfig(EGL10 egl, EGLDisplay display) {
		// Do NOT use eglChooseConfig, because the Android EGL code pushes in
		// multisample flags in eglChooseConfig if the user has selected the "force 4x
		// MSAA" option in settings, and that is completely wasted for our warp
		// target.
		int[] numConfig = new int[1];
		if (!egl.eglGetConfigs(display, null, 0, numConfig)) {
			throw new IllegalArgumentException("eglGetConfigs failed.");
		}

		int configsCount = numConfig[0];
		if (configsCount <= 0) {
			throw new IllegalArgumentException("No configs match configSpec");
		}

		EGLConfig[] configs = new EGLConfig[configsCount];
		if (!egl.eglGetConfigs(display, configs, configsCount, numConfig)) {
			throw new IllegalArgumentException("eglGetConfigs #2 failed.");
		}

		int[] value = new int[1];
		for (EGLConfig config : configs) {
			egl.eglGetConfigAttrib(display, config, EGL10.EGL_RENDERABLE_TYPE, value);
			if ((value[0] & EGLExt.EGL_OPENGL_ES3_BIT_KHR) != EGLExt.EGL_OPENGL_ES3_BIT_KHR) {
				continue;
			}

			// The pbuffer config also needs to be compatible with normal window rendering
			// so it can share textures with the window context.
			egl.eglGetConfigAttrib(display, config, EGL10.EGL_SURFACE_TYPE, value);
			if ((value[0] & (EGL10.EGL_WINDOW_BIT | EGL10.EGL_PBUFFER_BIT)) != (EGL10.EGL_WINDOW_BIT | EGL10.EGL_PBUFFER_BIT)) {
				continue;
			}

			// Check each attribute in CONFIG_ATTRIBS (which are the attributes we care about)
			// and ensure the value in config matches.
			int attribIndex = 0;
			while (CONFIG_ATTRIBS[attribIndex] != EGL10.EGL_NONE) {
				egl.eglGetConfigAttrib(display, config, CONFIG_ATTRIBS[attribIndex], value);
				if (value[0] != CONFIG_ATTRIBS[attribIndex + 1]) {
					// Attribute key's value does not match the configs value.
					// Start checking next config.
					break;
				}

				// Step by two because CONFIG_ATTRIBS is in key/value pairs.
				attribIndex += 2;
			}

			if (CONFIG_ATTRIBS[attribIndex] == EGL10.EGL_NONE) {
				// All relevant attributes match, set the config and stop checking the rest.
				return config;
			}
		}
		return null;
	}
}
