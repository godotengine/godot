/**************************************************************************/
/*  GodotVulkanRenderView.kt                                              */
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

package org.godotengine.godot.render

import android.annotation.SuppressLint
import android.graphics.PixelFormat
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.PointerIcon
import org.godotengine.godot.Godot
import org.godotengine.godot.GodotRenderView
import org.godotengine.godot.input.GodotInputHandler

internal class GodotVulkanRenderView(
	private val godot: Godot,
	renderer: GodotRenderer,
	private val mInputHandler: GodotInputHandler,
	shouldBeTranslucent: Boolean
) : VkSurfaceView(
	godot.context,
	renderer
), GodotRenderView {

	init {
		pointerIcon = PointerIcon.getSystemIcon(context, PointerIcon.TYPE_DEFAULT)
		isFocusableInTouchMode = true
		isClickable = false

		if (shouldBeTranslucent) {
			this.holder.setFormat(PixelFormat.TRANSLUCENT)
		}
	}

	override fun getView() = this

	override fun getInputHandler() = mInputHandler

	@SuppressLint("ClickableViewAccessibility")
	override fun onTouchEvent(event: MotionEvent): Boolean {
		super.onTouchEvent(event)
		return mInputHandler.onTouchEvent(event)
	}

	override fun onKeyUp(keyCode: Int, event: KeyEvent): Boolean {
		return mInputHandler.onKeyUp(keyCode, event) || super.onKeyUp(keyCode, event)
	}

	override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean {
		return mInputHandler.onKeyDown(keyCode, event) || super.onKeyDown(keyCode, event)
	}

	override fun onGenericMotionEvent(event: MotionEvent): Boolean {
		return mInputHandler.onGenericMotionEvent(event) || super.onGenericMotionEvent(event)
	}

	override fun onCapturedPointerEvent(event: MotionEvent): Boolean {
		return mInputHandler.onGenericMotionEvent(event)
	}

	override fun requestPointerCapture() {
		if (godot.canCapturePointer()) {
			super.requestPointerCapture()
			mInputHandler.onPointerCaptureChange(true)
		}
	}

	override fun releasePointerCapture() {
		super.releasePointerCapture()
		mInputHandler.onPointerCaptureChange(false)
	}

	override fun onPointerCaptureChange(hasCapture: Boolean) {
		super.onPointerCaptureChange(hasCapture)
		mInputHandler.onPointerCaptureChange(hasCapture)
	}

	override fun onResolvePointerIcon(me: MotionEvent, pointerIndex: Int): PointerIcon {
		return pointerIcon
	}
}
