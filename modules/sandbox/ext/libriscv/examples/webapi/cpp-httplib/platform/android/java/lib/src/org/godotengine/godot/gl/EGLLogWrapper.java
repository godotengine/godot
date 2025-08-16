// clang-format off

/*
 * Copyright (C) 2008 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.godotengine.godot.gl;

import android.opengl.GLDebugHelper;
import android.opengl.GLException;

import java.io.IOException;
import java.io.Writer;

import javax.microedition.khronos.egl.EGL;
import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGL11;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.EGLContext;
import javax.microedition.khronos.egl.EGLDisplay;
import javax.microedition.khronos.egl.EGLSurface;

class EGLLogWrapper implements EGL11 {
	private EGL10 mEgl10;
	Writer mLog;
	boolean mLogArgumentNames;
	boolean mCheckError;
	private int mArgCount;


	public EGLLogWrapper(EGL egl, int configFlags, Writer log) {
		mEgl10 = (EGL10) egl;
		mLog = log;
		mLogArgumentNames =
				(GLDebugHelper.CONFIG_LOG_ARGUMENT_NAMES & configFlags) != 0;
		mCheckError =
				(GLDebugHelper.CONFIG_CHECK_GL_ERROR & configFlags) != 0;
	}

	public boolean eglChooseConfig(EGLDisplay display, int[] attrib_list,
								   EGLConfig[] configs, int config_size, int[] num_config) {
		begin("eglChooseConfig");
		arg("display", display);
		arg("attrib_list", attrib_list);
		arg("config_size", config_size);
		end();

		boolean result = mEgl10.eglChooseConfig(display, attrib_list, configs,
				config_size, num_config);
		arg("configs", configs);
		arg("num_config", num_config);
		returns(result);
		checkError();
		return result;
	}

	public boolean eglCopyBuffers(EGLDisplay display, EGLSurface surface,
								  Object native_pixmap) {
		begin("eglCopyBuffers");
		arg("display", display);
		arg("surface", surface);
		arg("native_pixmap", native_pixmap);
		end();

		boolean result = mEgl10.eglCopyBuffers(display, surface, native_pixmap);
		returns(result);
		checkError();
		return result;
	}

	public EGLContext eglCreateContext(EGLDisplay display, EGLConfig config,
									   EGLContext share_context, int[] attrib_list) {
		begin("eglCreateContext");
		arg("display", display);
		arg("config", config);
		arg("share_context", share_context);
		arg("attrib_list", attrib_list);
		end();

		EGLContext result = mEgl10.eglCreateContext(display, config,
				share_context, attrib_list);
		returns(result);
		checkError();
		return result;
	}

	public EGLSurface eglCreatePbufferSurface(EGLDisplay display,
											  EGLConfig config, int[] attrib_list) {
		begin("eglCreatePbufferSurface");
		arg("display", display);
		arg("config", config);
		arg("attrib_list", attrib_list);
		end();

		EGLSurface result = mEgl10.eglCreatePbufferSurface(display, config,
				attrib_list);
		returns(result);
		checkError();
		return result;
	}

	public EGLSurface eglCreatePixmapSurface(EGLDisplay display,
											 EGLConfig config, Object native_pixmap, int[] attrib_list) {
		begin("eglCreatePixmapSurface");
		arg("display", display);
		arg("config", config);
		arg("native_pixmap", native_pixmap);
		arg("attrib_list", attrib_list);
		end();

		EGLSurface result = mEgl10.eglCreatePixmapSurface(display, config,
				native_pixmap, attrib_list);
		returns(result);
		checkError();
		return result;
	}

	public EGLSurface eglCreateWindowSurface(EGLDisplay display,
											 EGLConfig config, Object native_window, int[] attrib_list) {
		begin("eglCreateWindowSurface");
		arg("display", display);
		arg("config", config);
		arg("native_window", native_window);
		arg("attrib_list", attrib_list);
		end();

		EGLSurface result = mEgl10.eglCreateWindowSurface(display, config,
				native_window, attrib_list);
		returns(result);
		checkError();
		return result;
	}

	public boolean eglDestroyContext(EGLDisplay display, EGLContext context) {
		begin("eglDestroyContext");
		arg("display", display);
		arg("context", context);
		end();

		boolean result = mEgl10.eglDestroyContext(display, context);
		returns(result);
		checkError();
		return result;
	}

	public boolean eglDestroySurface(EGLDisplay display, EGLSurface surface) {
		begin("eglDestroySurface");
		arg("display", display);
		arg("surface", surface);
		end();

		boolean result = mEgl10.eglDestroySurface(display, surface);
		returns(result);
		checkError();
		return result;
	}

	public boolean eglGetConfigAttrib(EGLDisplay display, EGLConfig config,
									  int attribute, int[] value) {
		begin("eglGetConfigAttrib");
		arg("display", display);
		arg("config", config);
		arg("attribute", attribute);
		end();
		boolean result = mEgl10.eglGetConfigAttrib(display, config, attribute,
				value);
		arg("value", value);
		returns(result);
		checkError();
		return false;
	}

	public boolean eglGetConfigs(EGLDisplay display, EGLConfig[] configs,
								 int config_size, int[] num_config) {
		begin("eglGetConfigs");
		arg("display", display);
		arg("config_size", config_size);
		end();

		boolean result = mEgl10.eglGetConfigs(display, configs, config_size,
				num_config);
		arg("configs", configs);
		arg("num_config", num_config);
		returns(result);
		checkError();
		return result;
	}

	public EGLContext eglGetCurrentContext() {
		begin("eglGetCurrentContext");
		end();

		EGLContext result = mEgl10.eglGetCurrentContext();
		returns(result);

		checkError();
		return result;
	}

	public EGLDisplay eglGetCurrentDisplay() {
		begin("eglGetCurrentDisplay");
		end();

		EGLDisplay result = mEgl10.eglGetCurrentDisplay();
		returns(result);

		checkError();
		return result;
	}

	public EGLSurface eglGetCurrentSurface(int readdraw) {
		begin("eglGetCurrentSurface");
		arg("readdraw", readdraw);
		end();

		EGLSurface result = mEgl10.eglGetCurrentSurface(readdraw);
		returns(result);

		checkError();
		return result;
	}

	public EGLDisplay eglGetDisplay(Object native_display) {
		begin("eglGetDisplay");
		arg("native_display", native_display);
		end();

		EGLDisplay result = mEgl10.eglGetDisplay(native_display);
		returns(result);

		checkError();
		return result;
	}

	public int eglGetError() {
		begin("eglGetError");
		end();

		int result = mEgl10.eglGetError();
		returns(getErrorString(result));

		return result;
	}

	public boolean eglInitialize(EGLDisplay display, int[] major_minor) {
		begin("eglInitialize");
		arg("display", display);
		end();
		boolean result = mEgl10.eglInitialize(display, major_minor);
		returns(result);
		arg("major_minor", major_minor);
		checkError();
		return result;
	}

	public boolean eglMakeCurrent(EGLDisplay display, EGLSurface draw,
								  EGLSurface read, EGLContext context) {
		begin("eglMakeCurrent");
		arg("display", display);
		arg("draw", draw);
		arg("read", read);
		arg("context", context);
		end();
		boolean result = mEgl10.eglMakeCurrent(display, draw, read, context);
		returns(result);
		checkError();
		return result;
	}

	public boolean eglQueryContext(EGLDisplay display, EGLContext context,
								   int attribute, int[] value) {
		begin("eglQueryContext");
		arg("display", display);
		arg("context", context);
		arg("attribute", attribute);
		end();
		boolean result = mEgl10.eglQueryContext(display, context, attribute,
				value);
		returns(value[0]);
		returns(result);
		checkError();
		return result;
	}

	public String eglQueryString(EGLDisplay display, int name) {
		begin("eglQueryString");
		arg("display", display);
		arg("name", name);
		end();
		String result = mEgl10.eglQueryString(display, name);
		returns(result);
		checkError();
		return result;
	}

	public boolean eglQuerySurface(EGLDisplay display, EGLSurface surface,
								   int attribute, int[] value) {
		begin("eglQuerySurface");
		arg("display", display);
		arg("surface", surface);
		arg("attribute", attribute);
		end();
		boolean result = mEgl10.eglQuerySurface(display, surface, attribute,
				value);
		returns(value[0]);
		returns(result);
		checkError();
		return result;
	}

	public boolean eglSwapBuffers(EGLDisplay display, EGLSurface surface) {
		begin("eglSwapBuffers");
		arg("display", display);
		arg("surface", surface);
		end();
		boolean result = mEgl10.eglSwapBuffers(display, surface);
		returns(result);
		checkError();
		return result;
	}

	public boolean eglTerminate(EGLDisplay display) {
		begin("eglTerminate");
		arg("display", display);
		end();
		boolean result = mEgl10.eglTerminate(display);
		returns(result);
		checkError();
		return result;
	}

	public boolean eglWaitGL() {
		begin("eglWaitGL");
		end();
		boolean result = mEgl10.eglWaitGL();
		returns(result);
		checkError();
		return result;
	}

	public boolean eglWaitNative(int engine, Object bindTarget) {
		begin("eglWaitNative");
		arg("engine", engine);
		arg("bindTarget", bindTarget);
		end();
		boolean result = mEgl10.eglWaitNative(engine, bindTarget);
		returns(result);
		checkError();
		return result;
	}

	private void checkError() {
		int eglError;
		if ((eglError = mEgl10.eglGetError()) != EGL_SUCCESS) {
			String errorMessage = "eglError: " + getErrorString(eglError);
			logLine(errorMessage);
			if (mCheckError) {
				throw new GLException(eglError, errorMessage);
			}
		}
	}

	private void logLine(String message) {
		log(message + '\n');
	}

	private void log(String message) {
		try {
			mLog.write(message);
		} catch (IOException e) {
			// Ignore exception, keep on trying
		}
	}

	private void begin(String name) {
		log(name + '(');
		mArgCount = 0;
	}

	private void arg(String name, String value) {
		if (mArgCount++ > 0) {
			log(", ");
		}
		if (mLogArgumentNames) {
			log(name + "=");
		}
		log(value);
	}

	private void end() {
		log(");\n");
		flush();
	}

	private void flush() {
		try {
			mLog.flush();
		} catch (IOException e) {
			mLog = null;
		}
	}

	private void arg(String name, int value) {
		arg(name, Integer.toString(value));
	}

	private void arg(String name, Object object) {
		arg(name, toString(object));
	}

	private void arg(String name, EGLDisplay object) {
		if (object == EGL10.EGL_DEFAULT_DISPLAY) {
			arg(name, "EGL10.EGL_DEFAULT_DISPLAY");
		} else if (object == EGL_NO_DISPLAY) {
			arg(name, "EGL10.EGL_NO_DISPLAY");
		} else {
			arg(name, toString(object));
		}
	}

	private void arg(String name, EGLContext object) {
		if (object == EGL10.EGL_NO_CONTEXT) {
			arg(name, "EGL10.EGL_NO_CONTEXT");
		} else {
			arg(name, toString(object));
		}
	}

	private void arg(String name, EGLSurface object) {
		if (object == EGL10.EGL_NO_SURFACE) {
			arg(name, "EGL10.EGL_NO_SURFACE");
		} else {
			arg(name, toString(object));
		}
	}

	private void returns(String result) {
		log(" returns " + result + ";\n");
		flush();
	}

	private void returns(int result) {
		returns(Integer.toString(result));
	}

	private void returns(boolean result) {
		returns(Boolean.toString(result));
	}

	private void returns(Object result) {
		returns(toString(result));
	}

	private String toString(Object obj) {
		if (obj == null) {
			return "null";
		} else {
			return obj.toString();
		}
	}

	private void arg(String name, int[] arr) {
		if (arr == null) {
			arg(name, "null");
		} else {
			arg(name, toString(arr.length, arr, 0));
		}
	}

	private void arg(String name, Object[] arr) {
		if (arr == null) {
			arg(name, "null");
		} else {
			arg(name, toString(arr.length, arr, 0));
		}
	}

	private String toString(int n, int[] arr, int offset) {
		StringBuilder buf = new StringBuilder();
		buf.append("{\n");
		int arrLen = arr.length;
		for (int i = 0; i < n; i++) {
			int index = offset + i;
			buf.append(" [" + index + "] = ");
			if (index < 0 || index >= arrLen) {
				buf.append("out of bounds");
			} else {
				buf.append(arr[index]);
			}
			buf.append('\n');
		}
		buf.append("}");
		return buf.toString();
	}

	private String toString(int n, Object[] arr, int offset) {
		StringBuilder buf = new StringBuilder();
		buf.append("{\n");
		int arrLen = arr.length;
		for (int i = 0; i < n; i++) {
			int index = offset + i;
			buf.append(" [" + index + "] = ");
			if (index < 0 || index >= arrLen) {
				buf.append("out of bounds");
			} else {
				buf.append(arr[index]);
			}
			buf.append('\n');
		}
		buf.append("}");
		return buf.toString();
	}

	private static String getHex(int value) {
		return "0x" + Integer.toHexString(value);
	}

	public static String getErrorString(int error) {
		switch (error) {
			case EGL_SUCCESS:
				return "EGL_SUCCESS";
			case EGL_NOT_INITIALIZED:
				return "EGL_NOT_INITIALIZED";
			case EGL_BAD_ACCESS:
				return "EGL_BAD_ACCESS";
			case EGL_BAD_ALLOC:
				return "EGL_BAD_ALLOC";
			case EGL_BAD_ATTRIBUTE:
				return "EGL_BAD_ATTRIBUTE";
			case EGL_BAD_CONFIG:
				return "EGL_BAD_CONFIG";
			case EGL_BAD_CONTEXT:
				return "EGL_BAD_CONTEXT";
			case EGL_BAD_CURRENT_SURFACE:
				return "EGL_BAD_CURRENT_SURFACE";
			case EGL_BAD_DISPLAY:
				return "EGL_BAD_DISPLAY";
			case EGL_BAD_MATCH:
				return "EGL_BAD_MATCH";
			case EGL_BAD_NATIVE_PIXMAP:
				return "EGL_BAD_NATIVE_PIXMAP";
			case EGL_BAD_NATIVE_WINDOW:
				return "EGL_BAD_NATIVE_WINDOW";
			case EGL_BAD_PARAMETER:
				return "EGL_BAD_PARAMETER";
			case EGL_BAD_SURFACE:
				return "EGL_BAD_SURFACE";
			case EGL11.EGL_CONTEXT_LOST:
				return "EGL_CONTEXT_LOST";
			default:
				return getHex(error);
		}
	}
}
