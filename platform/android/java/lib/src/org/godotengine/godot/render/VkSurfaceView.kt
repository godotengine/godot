/**************************************************************************/
/*  VkSurfaceView.kt                                                      */
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

@file:JvmName("VkSurfaceView")
package org.godotengine.godot.render

import android.content.Context
import android.view.SurfaceHolder
import android.view.SurfaceView

/**
 * An implementation of SurfaceView that uses the dedicated surface for
 * displaying Vulkan rendering.
 * <p>
 * A [VkSurfaceView] provides the following features:
 * <p>
 * <ul>
 * <li>Manages a surface, which is a special piece of memory that can be
 * composited into the Android view system.
 * <li>Accepts a [GodotRenderer] object that does the actual rendering.
 * <li>Renders on the [RenderThread] thread provided by the [GodotRenderer] to decouple rendering
 *  * performance from the UI thread.
 * </ul>
 */
internal open class VkSurfaceView(context: Context) : SurfaceView(context), SurfaceHolder.Callback {
	companion object {
		fun checkState(expression: Boolean, errorMessage: Any) {
			check(expression) { errorMessage.toString() }
		}
	}

	/**
	 * Performs the actual rendering.
	 */
	private lateinit var renderer: GodotRenderer

	init {
		isClickable = true
		holder.addCallback(this)
	}

	/**
	 * Set the [GodotRenderer] associated with the view, and starts the thread that will drive the vulkan
	 * rendering.
	 *
	 * This method should be called once and only once in the life-cycle of [VkSurfaceView].
	 */
	fun startRenderer(renderer: GodotRenderer) {
		checkState(!this::renderer.isInitialized, "startRenderer must only be invoked once")
		this.renderer = renderer
		this.renderer.startRenderer()
	}

	override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
		renderer.renderThread.surfaceChanged(holder, width, height)
	}

	override fun surfaceDestroyed(holder: SurfaceHolder) {
		renderer.renderThread.surfaceDestroyed(holder)
	}

	override fun surfaceCreated(holder: SurfaceHolder) {
		renderer.renderThread.surfaceCreated(holder)
	}
}
