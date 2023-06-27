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
package org.godotengine.godot.vulkan

import android.util.Log
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Thread implementation for the [VkSurfaceView] onto which the vulkan logic is ran.
 *
 * The implementation is modeled after [android.opengl.GLSurfaceView]'s GLThread.
 */
internal class VkThread(private val vkSurfaceView: VkSurfaceView, private val vkRenderer: VkRenderer) : Thread(TAG) {
	companion object {
		private val TAG = VkThread::class.java.simpleName
	}

	/**
	 * Used to run events scheduled on the thread.
	 */
	private val eventQueue = ArrayList<Runnable>()

	/**
	 * Used to synchronize interaction with other threads (e.g: main thread).
	 */
	private val lock = ReentrantLock()
	private val lockCondition = lock.newCondition()

	private var shouldExit = false
	private var exited = false
	private var rendererInitialized = false
	private var rendererResumed = false
	private var resumed = false
	private var surfaceChanged = false
	private var hasSurface = false
	private var width = 0
	private var height = 0

	/**
	 * Determine when drawing can occur on the thread. This usually occurs after the
	 * [android.view.Surface] is available, the app is in a resumed state.
	 */
	private val readyToDraw
		get() = hasSurface && resumed

	private fun threadExiting() {
		lock.withLock {
			exited = true
			lockCondition.signalAll()
		}
	}

	/**
	 * Queue an event on the [VkThread].
	 */
	fun queueEvent(event: Runnable) {
		lock.withLock {
			eventQueue.add(event)
			lockCondition.signalAll()
		}
	}

	/**
	 * Request the thread to exit and block until it's done.
	 */
	fun blockingExit() {
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

	/**
	 * Invoked when the app resumes.
	 */
	fun onResume() {
		lock.withLock {
			resumed = true
			lockCondition.signalAll()
		}
	}

	/**
	 * Invoked when the app pauses.
	 */
	fun onPause() {
		lock.withLock {
			resumed = false
			lockCondition.signalAll()
		}
	}

	/**
	 * Invoked when the [android.view.Surface] has been created.
	 */
	fun onSurfaceCreated() {
		// This is a no op because surface creation will always be followed by surfaceChanged()
		// which provide all the needed information.
	}

	/**
	 * Invoked following structural updates to [android.view.Surface].
	 */
	fun onSurfaceChanged(width: Int, height: Int) {
		lock.withLock {
			hasSurface = true
			surfaceChanged = true;
			this.width = width
			this.height = height

			lockCondition.signalAll()
		}
	}

	/**
	 * Invoked when the [android.view.Surface] is no longer available.
	 */
	fun onSurfaceDestroyed() {
		lock.withLock {
			hasSurface = false
			lockCondition.signalAll()
		}
	}

	/**
	 * Thread loop modeled after [android.opengl.GLSurfaceView]'s GLThread.
	 */
	override fun run() {
		try {
			while (true) {
				var event: Runnable? = null
				lock.withLock {
					while (true) {
						// Code path for exiting the thread loop.
						if (shouldExit) {
							vkRenderer.onVkDestroy()
							return
						}

						// Check for events and execute them outside of the loop if found to avoid
						// blocking the thread lifecycle by holding onto the lock.
						if (eventQueue.isNotEmpty()) {
							event = eventQueue.removeAt(0)
							break;
						}

						if (readyToDraw) {
							if (!rendererResumed) {
								rendererResumed = true
								vkRenderer.onVkResume()

								if (!rendererInitialized) {
									rendererInitialized = true
									vkRenderer.onVkSurfaceCreated(vkSurfaceView.holder.surface)
								}
							}

							if (surfaceChanged) {
								vkRenderer.onVkSurfaceChanged(vkSurfaceView.holder.surface, width, height)
								surfaceChanged = false
							}

							// Break out of the loop so drawing can occur without holding onto the lock.
							break;
						} else if (rendererResumed) {
							// If we aren't ready to draw but are resumed, that means we either lost a surface
							// or the app was paused.
							rendererResumed = false
							vkRenderer.onVkPause()
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
					continue
				}

				// Draw only when there no more queued events.
				vkRenderer.onVkDrawFrame()
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
