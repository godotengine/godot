/**************************************************************************/
/*  GLUtils.java                                                          */
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

package org.godotengine.godot.utils;

import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLDisplay;
import android.util.Log;

/**
 * Contains GL utilities methods.
 */
public class GLUtils {
	private static final String TAG = GLUtils.class.getSimpleName();

	public static final boolean DEBUG = false;

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
		EGL14.EGL_BUFFER_SIZE,
		EGL14.EGL_ALPHA_SIZE,
		EGL14.EGL_BLUE_SIZE,
		EGL14.EGL_GREEN_SIZE,
		EGL14.EGL_RED_SIZE,
		EGL14.EGL_DEPTH_SIZE,
		EGL14.EGL_STENCIL_SIZE,
		EGL14.EGL_CONFIG_CAVEAT,
		EGL14.EGL_CONFIG_ID,
		EGL14.EGL_LEVEL,
		EGL14.EGL_MAX_PBUFFER_HEIGHT,
		EGL14.EGL_MAX_PBUFFER_PIXELS,
		EGL14.EGL_MAX_PBUFFER_WIDTH,
		EGL14.EGL_NATIVE_RENDERABLE,
		EGL14.EGL_NATIVE_VISUAL_ID,
		EGL14.EGL_NATIVE_VISUAL_TYPE,
		0x3030, // EGL14.EGL_PRESERVED_RESOURCES,
		EGL14.EGL_SAMPLES,
		EGL14.EGL_SAMPLE_BUFFERS,
		EGL14.EGL_SURFACE_TYPE,
		EGL14.EGL_TRANSPARENT_TYPE,
		EGL14.EGL_TRANSPARENT_RED_VALUE,
		EGL14.EGL_TRANSPARENT_GREEN_VALUE,
		EGL14.EGL_TRANSPARENT_BLUE_VALUE,
		EGL14.EGL_BIND_TO_TEXTURE_RGB,
		EGL14.EGL_BIND_TO_TEXTURE_RGBA,
		EGL14.EGL_MIN_SWAP_INTERVAL,
		EGL14.EGL_MAX_SWAP_INTERVAL,
		EGL14.EGL_LUMINANCE_SIZE,
		EGL14.EGL_ALPHA_MASK_SIZE,
		EGL14.EGL_COLOR_BUFFER_TYPE,
		EGL14.EGL_RENDERABLE_TYPE,
		EGL14.EGL_CONFORMANT
	};

	private GLUtils() {}

	public static void checkEglError(String tag, String prompt) {
		int error;
		while ((error = EGL14.eglGetError()) != EGL14.EGL_SUCCESS) {
			Log.e(tag, String.format("%s: EGL error: 0x%x", prompt, error));
		}
	}

	public static void printConfigs(EGLDisplay display,
			EGLConfig[] configs) {
		int numConfigs = configs.length;
		Log.v(TAG, String.format("%d configurations", numConfigs));
		for (int i = 0; i < numConfigs; i++) {
			Log.v(TAG, String.format("Configuration %d:\n", i));
			printConfig(display, configs[i]);
		}
	}

	private static void printConfig(EGLDisplay display,
			EGLConfig config) {
		int[] value = new int[1];
		for (int i = 0; i < ATTRIBUTES.length; i++) {
			int attribute = ATTRIBUTES[i];
			String name = ATTRIBUTES_NAMES[i];
			if (EGL14.eglGetConfigAttrib(display, config, attribute, value, 0)) {
				Log.i(TAG, String.format("  %s: %d\n", name, value[0]));
			} else {
				// Log.w(TAG, String.format("  %s: failed\n", name));
				while (EGL14.eglGetError() != EGL14.EGL_SUCCESS) {
					// Continue.
				}
			}
		}
	}

	public static String getErrorString(int error) {
		switch (error) {
			case EGL14.EGL_SUCCESS:
				return "EGL_SUCCESS";
			case EGL14.EGL_NOT_INITIALIZED:
				return "EGL_NOT_INITIALIZED";
			case EGL14.EGL_BAD_ACCESS:
				return "EGL_BAD_ACCESS";
			case EGL14.EGL_BAD_ALLOC:
				return "EGL_BAD_ALLOC";
			case EGL14.EGL_BAD_ATTRIBUTE:
				return "EGL_BAD_ATTRIBUTE";
			case EGL14.EGL_BAD_CONFIG:
				return "EGL_BAD_CONFIG";
			case EGL14.EGL_BAD_CONTEXT:
				return "EGL_BAD_CONTEXT";
			case EGL14.EGL_BAD_CURRENT_SURFACE:
				return "EGL_BAD_CURRENT_SURFACE";
			case EGL14.EGL_BAD_DISPLAY:
				return "EGL_BAD_DISPLAY";
			case EGL14.EGL_BAD_MATCH:
				return "EGL_BAD_MATCH";
			case EGL14.EGL_BAD_NATIVE_PIXMAP:
				return "EGL_BAD_NATIVE_PIXMAP";
			case EGL14.EGL_BAD_NATIVE_WINDOW:
				return "EGL_BAD_NATIVE_WINDOW";
			case EGL14.EGL_BAD_PARAMETER:
				return "EGL_BAD_PARAMETER";
			case EGL14.EGL_BAD_SURFACE:
				return "EGL_BAD_SURFACE";
			case EGL14.EGL_CONTEXT_LOST:
				return "EGL_CONTEXT_LOST";
			default:
				return getHex(error);
		}
	}

	private static String getHex(int value) {
		return "0x" + Integer.toHexString(value);
	}
}
