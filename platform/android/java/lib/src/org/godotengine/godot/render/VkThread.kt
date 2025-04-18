/**************************************************************************/
/*  VkThread.kt                                                           */
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

@file:JvmName("VkThread")
package org.godotengine.godot.render

import android.util.Log
import android.view.SurfaceHolder
import java.lang.ref.WeakReference
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Implementation of the thread used by the [GodotRenderer] to drive render logic.
 *
 * The implementation is modeled after [android.opengl.GLSurfaceView]'s GLThread
 */
internal class VkThread(private val renderer: GodotRenderer) : RenderThread(TAG) {
	companion object {
		private val TAG = VkThread::class.java.simpleName
	}

	/**
	 * Store [Surface] related data.
	 */
	data class SurfaceInfo(var holder: SurfaceHolder, var width: Int, var height: Int, var surfaceChanged: Boolean = true)

	/**
	 * Used to run events scheduled on the thread.
	 */
	private val eventQueue = ArrayList<Runnable>()

	private var renderVkSurfaceInfo: SurfaceInfo? = null
	private fun hasSurface() = renderVkSurfaceInfo != null

	/**
	 * Used to synchronize interaction with other threads (e.g: main thread).
	 */
	private val lock = ReentrantLock()
	private val lockCondition = lock.newCondition()

	private var shouldExit = false
	private var exited = false
	private var rendererInitialized = false
	private var threadResumed = false

	private var renderMode = Renderer.RenderMode.CONTINUOUSLY
	private var requestRender = true

	/**
	 * Determine when drawing can occur on the thread. This usually occurs after the
	 * [android.view.Surface] is available, the app is in a resumed state.
	 */
	private fun readyToDraw(): Boolean  {
		return threadResumed && (requestRender || (renderMode == Renderer.RenderMode.CONTINUOUSLY)) && hasSurface()
	}

	private fun threadStarting() {
		lock.withLock {
			Log.d(TAG, "Starting render thread")
			renderer.onRenderThreadStarting()
			lockCondition.signalAll()
		}
	}

	private fun threadExiting() {
		lock.withLock {
			Log.d(TAG, "Exiting render thread")
			renderer.onRenderThreadExiting()

			exited = true
			lockCondition.signalAll()
		}
	}

	override fun setRenderMode(renderMode: Renderer.RenderMode) {
		lock.withLock {
			this.renderMode = renderMode
			lockCondition.signalAll()
		}
	}

	override fun getRenderMode(): Renderer.RenderMode {
		return lock.withLock {
			renderMode
		}
	}

	override fun requestRender() {
		lock.withLock {
			requestRender = true
			lockCondition.signalAll()
		}
	}

	override fun queueEvent(event: Runnable) {
		lock.withLock {
			eventQueue.add(event)
			lockCondition.signalAll()
		}
	}

	override fun requestExitAndWait() {
		lock.withLock {
			shouldExit = true
			lockCondition.signalAll()
			while (!exited) {
				try {
					Log.i(TAG, "Waiting on exit for $name")
					lockCondition.await()
				} catch (ex: InterruptedException) {
					currentThread().interrupt()
				}
			}
		}
	}

	override fun onResume() {
		lock.withLock {
			Log.d(TAG, "Resuming render thread")

			threadResumed = true
			requestRender = true
			lockCondition.signalAll()
		}
	}

	override fun onPause() {
		lock.withLock {
			Log.d(TAG, "Pausing render thread")

			threadResumed = false
			lockCondition.signalAll()
		}
	}

	override fun surfaceCreated(holder: SurfaceHolder, surfaceViewWeakRef: WeakReference<GLSurfaceView>?) {
		// This is a no op because surface creation will always be followed by surfaceChanged()
		// which provide all the needed information.
	}

	override fun surfaceChanged(holder: SurfaceHolder, width: Int, height: Int) {
		lock.withLock {
			val surfaceInfo = renderVkSurfaceInfo ?: SurfaceInfo(holder, width, height)
			surfaceInfo.apply {
				surfaceChanged = true
				this.width = width
				this.height = height
			}

			requestRender = true
			renderVkSurfaceInfo = surfaceInfo

			lockCondition.signalAll()
		}
	}

	override fun surfaceDestroyed(holder: SurfaceHolder) {
		lock.withLock {
			renderVkSurfaceInfo = null
			lockCondition.signalAll()
		}
	}

	/**
	 * Thread loop modeled after [android.opengl.GLSurfaceView]'s GLThread.
	 */
	override fun run() {
		try {
			threadStarting()

			while (true) {
				var event: Runnable? = null
				lock.withLock {
					while (true) {
						// Code path for exiting the thread loop.
						if (shouldExit) {
							return
						}

						// Check for events and execute them outside of the loop if found to avoid
						// blocking the thread lifecycle by holding onto the lock.
						if (eventQueue.isNotEmpty()) {
							event = eventQueue.removeAt(0)
							break
						}

						if (readyToDraw()) {
							if (!rendererInitialized) {
								rendererInitialized = true
								renderVkSurfaceInfo?.apply {
									renderer.onRenderSurfaceCreated(holder.surface)
								}
							}

							renderVkSurfaceInfo?.let {
								if (it.surfaceChanged) {
									renderer.onRenderSurfaceChanged(it.holder.surface, it.width, it.height)
									it.surfaceChanged = false
								}
							}

							// Break out of the loop so drawing can occur without holding onto the lock.
							requestRender = false
							break
						}
						// We only reach this state if we are not ready to draw and have no queued events, so
						// we wait.
						// On state change, the thread will be awoken using the [lock] and [lockCondition], and
						// we will resume execution.
						lockCondition.await()
					}
				}

				// Run queued event.
				if (event != null) {
					event?.run()
					event = null
					continue
				}

				// Draw only when there no more queued events.
				renderer.onRenderDrawFrame()
			}
		} catch (ex: InterruptedException) {
			Log.i(TAG, "InterruptedException", ex)
		} catch (ex: IllegalStateException) {
			Log.i(TAG, "IllegalStateException", ex)
		} finally {
			threadExiting()
		}
	}
}
