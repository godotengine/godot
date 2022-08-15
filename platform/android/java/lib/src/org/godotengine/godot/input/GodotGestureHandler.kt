/*************************************************************************/
/*  GodotGestureHandler.kt                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

package org.godotengine.godot.input

import android.view.GestureDetector.SimpleOnGestureListener
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.ScaleGestureDetector.OnScaleGestureListener
import org.godotengine.godot.GodotLib

/**
 * Handles regular and scale gesture input related events for the [GodotView] view.
 *
 * @See https://developer.android.com/reference/android/view/GestureDetector.SimpleOnGestureListener
 * @See https://developer.android.com/reference/android/view/ScaleGestureDetector.OnScaleGestureListener
 */
internal class GodotGestureHandler : SimpleOnGestureListener(), OnScaleGestureListener {

	companion object {
		private val TAG = GodotGestureHandler::class.java.simpleName
	}

	/**
	 * Enable pan and scale gestures
	 */
	var panningAndScalingEnabled = false

	private var doubleTapInProgress = false
	private var dragInProgress = false
	private var scaleInProgress = false
	private var contextClickInProgress = false

	override fun onDown(event: MotionEvent): Boolean {
		// Don't send / register a down event while we're in the middle of a double-tap
		if (!doubleTapInProgress) {
			// Send the down event
			GodotInputHandler.handleMotionEvent(event)
		}
		return true
	}

	override fun onSingleTapUp(event: MotionEvent): Boolean {
		GodotInputHandler.handleMotionEvent(event)
		return true
	}

	override fun onLongPress(event: MotionEvent) {
		contextClickRouter(event)
	}

	private fun contextClickRouter(event: MotionEvent) {
		if (scaleInProgress) {
			return
		}

		// Cancel the previous down event
		GodotInputHandler.handleMotionEvent(
			event.source,
			MotionEvent.ACTION_CANCEL,
			event.buttonState,
			event.x,
			event.y
		)

		// Turn a context click into a single tap right mouse button click.
		GodotInputHandler.handleMouseEvent(
			MotionEvent.ACTION_DOWN,
			MotionEvent.BUTTON_SECONDARY,
			event.x,
			event.y
		)
		contextClickInProgress = true
	}

	fun onMotionEvent(event: MotionEvent): Boolean {
		return when (event.actionMasked) {
			MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL, MotionEvent.ACTION_BUTTON_RELEASE -> {
				onActionUp(event)
			}
			MotionEvent.ACTION_MOVE -> {
				onActionMove(event)
			}
			else -> false
		}
	}

	private fun onActionUp(event: MotionEvent): Boolean {
		if (dragInProgress) {
			GodotInputHandler.handleMotionEvent(event)
			dragInProgress = false
			return true
		} else if (contextClickInProgress) {
			GodotInputHandler.handleMouseEvent(
				event.actionMasked,
				0,
				event.x,
				event.y
			)
			contextClickInProgress = false
			return true
		}
		return false
	}

	private fun onActionMove(event: MotionEvent): Boolean {
		if (contextClickInProgress) {
			GodotInputHandler.handleMouseEvent(
				event.actionMasked,
				MotionEvent.BUTTON_SECONDARY,
				event.x,
				event.y
			)
			return true
		}
		return false
	}

	override fun onDoubleTapEvent(event: MotionEvent): Boolean {
		if (event.actionMasked == MotionEvent.ACTION_UP) {
			doubleTapInProgress = false
		}
		return true
	}

	override fun onDoubleTap(event: MotionEvent): Boolean {
		doubleTapInProgress = true
		val x = event.x
		val y = event.y
		val buttonMask =
			if (GodotInputHandler.isMouseEvent(event)) {
				event.buttonState
			} else {
				MotionEvent.BUTTON_PRIMARY
			}
		GodotInputHandler.handleMouseEvent(MotionEvent.ACTION_DOWN, buttonMask, x, y, true)
		GodotInputHandler.handleMouseEvent(MotionEvent.ACTION_UP, 0, x, y, false)

		return true
	}

	override fun onScroll(
		originEvent: MotionEvent,
		terminusEvent: MotionEvent,
		distanceX: Float,
		distanceY: Float
	): Boolean {
		if (scaleInProgress) {
			if (dragInProgress) {
				// Cancel the drag
				GodotInputHandler.handleMotionEvent(
					originEvent.source,
					MotionEvent.ACTION_CANCEL,
					originEvent.buttonState,
					originEvent.x,
					originEvent.y
				)
				dragInProgress = false
			}
			return true
		}

		dragInProgress = true

		val x = terminusEvent.x
		val y = terminusEvent.y
		if (terminusEvent.pointerCount >= 2 && panningAndScalingEnabled) {
			GodotLib.pan(x, y, distanceX / 5f, distanceY / 5f)
		} else {
			GodotInputHandler.handleMotionEvent(terminusEvent)
		}
		return true
	}

	override fun onScale(detector: ScaleGestureDetector?): Boolean {
		if (detector == null || !panningAndScalingEnabled) {
			return false
		}
		GodotLib.magnify(
			detector.focusX,
			detector.focusY,
			detector.scaleFactor
		)
		return true
	}

	override fun onScaleBegin(detector: ScaleGestureDetector?): Boolean {
		if (detector == null || !panningAndScalingEnabled) {
			return false
		}
		scaleInProgress = true
		return true
	}

	override fun onScaleEnd(detector: ScaleGestureDetector?) {
		scaleInProgress = false
	}
}
