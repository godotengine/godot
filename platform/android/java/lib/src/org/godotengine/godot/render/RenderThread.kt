/**************************************************************************/
/*  RenderThread.kt                                                       */
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

@file:JvmName("RenderThread")
package org.godotengine.godot.render

import android.view.SurfaceHolder
import java.lang.ref.WeakReference

/**
 * Base class for the OpenGL / Vulkan render thread implementations.
 */
internal abstract class RenderThread(tag: String) : Thread(tag) {

	/**
	 * Queues an event on the render thread
	 */
	abstract fun queueEvent(event: Runnable)

	/**
	 * Request the thread to exit and block until it's done.
	 */
	abstract fun requestExitAndWait()

	/**
	 * Invoked when the app resumes.
	 */
	abstract fun onResume()

	/**
	 * Invoked when the app pauses.
	 */
	abstract fun onPause()

	abstract fun setRenderMode(renderMode: Renderer.RenderMode)

	abstract fun getRenderMode(): Renderer.RenderMode

	abstract fun requestRender()

	/**
	 * Invoked when the [android.view.Surface] has been created.
	 */
	open fun surfaceCreated(holder: SurfaceHolder, surfaceViewWeakRef: WeakReference<GLSurfaceView>? = null) { }

	/**
	 * Invoked following structural updates to [android.view.Surface].
	 */
	open fun surfaceChanged(holder: SurfaceHolder, width: Int, height: Int) { }

	/**
	 * Invoked when the [android.view.Surface] is no longer available.
	 */
	open fun surfaceDestroyed(holder: SurfaceHolder) { }

	open fun setSeparateRenderThreadEnabled(enabled: Boolean) {}

	open fun makeEglCurrent(windowId: Int): Boolean = false

	open fun eglSwapBuffers(windowId: Int) {}

	open fun releaseCurrentGLWindow(windowId: Int) {}
}
