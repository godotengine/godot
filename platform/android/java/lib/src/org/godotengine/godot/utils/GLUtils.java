/*************************************************************************/
/*  GLUtils.java                                                         */
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

package org.godotengine.godot.utils;

import android.util.Log;

import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.EGLDisplay;

/**
 * Contains GL utilities methods.
 */
public class GLUtils {
	private static final String TAG = GLUtils.class.getSimpleName();

	public static final boolean DEBUG = false;

	public static boolean use_gl3 = false;
	public static boolean use_32 = false;
	public static boolean use_debug_opengl = false;

	private static final String[] ATTRIBUTES_NAMES = new String[] {
		"EGL_BUFFER_SIZE",
		"EGL_ALPHA_SIZE",
		"EGL_BLUE_SIZE",
		"EGL_GREEN_SIZE",
		"EGL_RED_SIZE",
		"EGL_DEPTH_SIZE",
		"EGL_STENCIL_SIZE",
		"EGL_CONFIG_CAVEAT",
		"EGL_CONFIG_ID",
		"EGL_LEVEL",
		"EGL_MAX_PBUFFER_HEIGHT",
		"EGL_MAX_PBUFFER_PIXELS",
		"EGL_MAX_PBUFFER_WIDTH",
		"EGL_NATIVE_RENDERABLE",
		"EGL_NATIVE_VISUAL_ID",
		"EGL_NATIVE_VISUAL_TYPE",
		"EGL_PRESERVED_RESOURCES",
		"EGL_SAMPLES",
		"EGL_SAMPLE_BUFFERS",
		"EGL_SURFACE_TYPE",
		"EGL_TRANSPARENT_TYPE",
		"EGL_TRANSPARENT_RED_VALUE",
		"EGL_TRANSPARENT_GREEN_VALUE",
		"EGL_TRANSPARENT_BLUE_VALUE",
		"EGL_BIND_TO_TEXTURE_RGB",
		"EGL_BIND_TO_TEXTURE_RGBA",
		"EGL_MIN_SWAP_INTERVAL",
		"EGL_MAX_SWAP_INTERVAL",
		"EGL_LUMINANCE_SIZE",
		"EGL_ALPHA_MASK_SIZE",
		"EGL_COLOR_BUFFER_TYPE",
		"EGL_RENDERABLE_TYPE",
		"EGL_CONFORMANT"
	};

	private static final int[] ATTRIBUTES = new int[] {
		EGL10.EGL_BUFFER_SIZE,
		EGL10.EGL_ALPHA_SIZE,
		EGL10.EGL_BLUE_SIZE,
		EGL10.EGL_GREEN_SIZE,
		EGL10.EGL_RED_SIZE,
		EGL10.EGL_DEPTH_SIZE,
		EGL10.EGL_STENCIL_SIZE,
		EGL10.EGL_CONFIG_CAVEAT,
		EGL10.EGL_CONFIG_ID,
		EGL10.EGL_LEVEL,
		EGL10.EGL_MAX_PBUFFER_HEIGHT,
		EGL10.EGL_MAX_PBUFFER_PIXELS,
		EGL10.EGL_MAX_PBUFFER_WIDTH,
		EGL10.EGL_NATIVE_RENDERABLE,
		EGL10.EGL_NATIVE_VISUAL_ID,
		EGL10.EGL_NATIVE_VISUAL_TYPE,
		0x3030, // EGL10.EGL_PRESERVED_RESOURCES,
		EGL10.EGL_SAMPLES,
		EGL10.EGL_SAMPLE_BUFFERS,
		EGL10.EGL_SURFACE_TYPE,
		EGL10.EGL_TRANSPARENT_TYPE,
		EGL10.EGL_TRANSPARENT_RED_VALUE,
		EGL10.EGL_TRANSPARENT_GREEN_VALUE,
		EGL10.EGL_TRANSPARENT_BLUE_VALUE,
		0x3039, // EGL10.EGL_BIND_TO_TEXTURE_RGB,
		0x303A, // EGL10.EGL_BIND_TO_TEXTURE_RGBA,
		0x303B, // EGL10.EGL_MIN_SWAP_INTERVAL,
		0x303C, // EGL10.EGL_MAX_SWAP_INTERVAL,
		EGL10.EGL_LUMINANCE_SIZE,
		EGL10.EGL_ALPHA_MASK_SIZE,
		EGL10.EGL_COLOR_BUFFER_TYPE,
		EGL10.EGL_RENDERABLE_TYPE,
		0x3042 // EGL10.EGL_CONFORMANT
	};

	private GLUtils() {}

	public static void checkEglError(String tag, String prompt, EGL10 egl) {
		int error;
		while ((error = egl.eglGetError()) != EGL10.EGL_SUCCESS) {
			Log.e(tag, String.format("%s: EGL error: 0x%x", prompt, error));
		}
	}

	public static void printConfigs(EGL10 egl, EGLDisplay display,
			EGLConfig[] configs) {
		int numConfigs = configs.length;
		Log.v(TAG, String.format("%d configurations", numConfigs));
		for (int i = 0; i < numConfigs; i++) {
			Log.v(TAG, String.format("Configuration %d:\n", i));
			printConfig(egl, display, configs[i]);
		}
	}

	private static void printConfig(EGL10 egl, EGLDisplay display,
			EGLConfig config) {
		int[] value = new int[1];
		for (int i = 0; i < ATTRIBUTES.length; i++) {
			int attribute = ATTRIBUTES[i];
			String name = ATTRIBUTES_NAMES[i];
			if (egl.eglGetConfigAttrib(display, config, attribute, value)) {
				Log.i(TAG, String.format("  %s: %d\n", name, value[0]));
			} else {
				// Log.w(TAG, String.format("  %s: failed\n", name));
				while (egl.eglGetError() != EGL10.EGL_SUCCESS)
					;
			}
		}
	}
}
