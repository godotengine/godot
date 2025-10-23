/**************************************************************************/
/*  GodotRenderer.kt                                                      */
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

@file:JvmName("GodotRenderer")

package org.godotengine.godot.render

import android.util.Log
import android.view.Surface
import org.godotengine.godot.GodotLib
import org.godotengine.godot.plugin.GodotPluginRegistry
import org.godotengine.godot.render.GLSurfaceView.GLThread

/**
 * Responsible for setting up and driving Godot rendering logic.
 *
 * <h3>Threading</h3>
 * The renderer will create a separate render thread, so that rendering
 * performance is decoupled from the UI thread. Clients typically need to
 * communicate with the renderer from the UI thread, because that's where
 * input events are received. Clients can communicate using any of the
 * standard Java techniques for cross-thread communication, or they can
 * use the [GodotRenderer.queueOnRenderThread] convenience method.
 */
internal class GodotRenderer(val useVulkan: Boolean) : Renderer {
	companion object {
		private val TAG = GodotRenderer::class.java.simpleName
	}

	private val pluginRegistry: GodotPluginRegistry by lazy {
		GodotPluginRegistry.getPluginRegistry()
	}

	private var isStarted = false
	private var glRendererJustResumed = false

	private var rendererResumed = false

	/**
	 * Thread used to drive the render logic.
	 */
	val renderThread: RenderThread by lazy {
		if (useVulkan) {
			VkThread(this)
		} else {
			GLThread(this)
		}
	}

	/**
	 * Start the renderer which starts the render thread which in turn causes the
	 * rendering to start.
	 *
	 * <p>
	 * The following methods can only be called <em>after</em>
	 * startRenderer is called:
	 * <ul>
	 * <li>{@link #getRenderMode()}
	 * <li>{@link #requestRender()}
	 * <li>{@link #setRenderMode(int)}
	 * </ul>
	 */
	fun startRenderer() {
		if (isStarted) {
			return
		}
		isStarted = true
		renderThread.start()
	}

	fun queueOnRenderThread(runnable: Runnable) {
		renderThread.queueEvent(runnable)
	}

	override fun setRenderMode(renderMode: Renderer.RenderMode) {
		renderThread.setRenderMode(renderMode)
	}

	override fun getRenderMode(): Renderer.RenderMode {
		return renderThread.getRenderMode()
	}

	override fun requestRender() {
		renderThread.requestRender()
	}

	/**
	 * Called to resume the renderer.
	 * <p>
	 * The renderer can be resumed either because the activity is resumed or because [Surface] to
	 * render onto just became available.
	 * <p>
	 */
	private fun resumeRenderer() {
		Log.d(TAG, "Resuming renderer")

		rendererResumed = true
		if (useVulkan) {
			GodotLib.onRendererResumed()
		} else {
			// For OpenGL, we defer invoking GodotLib.onRendererResumed() until the first draw frame call.
			// This ensures we have a valid GL context and surface when we do so.
			glRendererJustResumed = true
		}
	}

	/**
	 * Called to pause the renderer.
	 *
	 * <p>
	 * The renderer can be paused either because the activity is paused or because there are
	 * no [Surface] to render on.
	 * <p>
	 */
	private fun pauseRenderer() {
		Log.d(TAG, "Pausing renderer")

		GodotLib.onRendererPaused()
		rendererResumed = false
	}

	/**
	 * Called when the rendering thread is starting and used as signal to complete engine setup.
	 */
	override fun onRenderThreadStarting() {
		Log.d(TAG, "Starting Godot Engine render thread")
	}

	/**
	 * Called when the rendering thread is exiting and used as signal to tear down the native logic.
	 */
	override fun onRenderThreadExiting() {
		Log.d(TAG, "Destroying Godot Engine")
		GodotLib.ondestroy()
	}

	/**
	 * The Activity was resumed, let's resume the render thread.
	 */
	fun onActivityStarted() {
		renderThread.onResume()
	}

	/**
	 * The Activity was resumed, let's resume the renderer.
	 */
	fun onActivityResumed() {
		queueOnRenderThread {
			resumeRenderer()
			GodotLib.focusin()
		}
	}

	/**
	 * Pauses the renderer.
	 */
	fun onActivityPaused() {
		queueOnRenderThread {
			GodotLib.focusout()
			pauseRenderer()
		}
	}

	/**
	 * Pauses the render thread.
	 */
	fun onActivityStopped() {
		renderThread.onPause()
	}

	/**
	 * Destroy the render thread.
	 */
	fun onActivityDestroyed() {
		renderThread.requestExitAndWait()
	}

	/**
	 * Called to draw the current frame.
	 * <p>
	 * This method is responsible for drawing the current frame.
	 * <p>
	 * @param gl the GL interface. Use <code>instanceof</code> to
	 * test if the interface supports GL11 or higher interfaces.
	 *
	 * @return true if the buffers should be swapped, false otherwise.
	 */
	override fun onRenderDrawFrame() {
		if (!useVulkan) {
			// For OpenGL, we defer invoking GodotLib.onRendererResumed() until the first draw frame call.
			// This ensures we have a valid GL context and surface when we do so.
			if (glRendererJustResumed) {
				GodotLib.onRendererResumed()
				glRendererJustResumed = false
			}
		}

		GodotLib.step()
		for (plugin in pluginRegistry.allPlugins) {
			if (useVulkan) {
				plugin.onVkDrawFrame()
			} else {
				plugin.onGLDrawFrame(null)
			}
			plugin.onRenderDrawFrame()
		}
	}

	override fun onRenderSurfaceChanged(surface: Surface?, width: Int, height: Int) {
		Log.d(TAG, "Rendering surface changed: $surface, w-$width, h-$height")
		GodotLib.resize(surface, width, height)

		for (plugin in pluginRegistry.allPlugins) {
				if (useVulkan) {
					plugin.onVkSurfaceChanged(surface, width, height)
				} else {
					plugin.onGLSurfaceChanged(null, width, height)
				}
			plugin.onRenderSurfaceChanged(surface, width, height)
		}
	}

	override fun onRenderSurfaceCreated(surface: Surface?) {
		Log.d(TAG, "Rendering surface created: $surface")
		GodotLib.newcontext(surface)

		for (plugin in pluginRegistry.allPlugins) {
				if (useVulkan) {
					plugin.onVkSurfaceCreated(surface)
				} else {
					plugin.onGLSurfaceCreated(null, null)
				}
			plugin.onRenderSurfaceCreated(surface)
		}
	}
}
