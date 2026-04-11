/**************************************************************************/
/*  EmbeddedGodotGame.kt                                                  */
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

package org.godotengine.editor.embed

import android.annotation.SuppressLint
import android.content.pm.ActivityInfo
import android.graphics.Color
import android.graphics.Point
import android.os.Bundle
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.view.WindowManager.LayoutParams.FLAG_DIM_BEHIND
import android.view.WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL
import android.view.WindowManager.LayoutParams.FLAG_WATCH_OUTSIDE_TOUCH
import android.widget.FrameLayout
import android.widget.TextView
import org.godotengine.editor.GodotGame
import org.godotengine.editor.R
import org.godotengine.godot.editor.utils.GameMenuUtils
import androidx.core.graphics.toColorInt

/**
 * Host the Godot game from the editor when the embedded mode is enabled.
 */
class EmbeddedGodotGame : GodotGame() {

	companion object {
		private val TAG = EmbeddedGodotGame::class.java.simpleName

		private const val PREFS_NAME = "embedded_game_prefs"
		private const val KEY_X = "embedded_window_x"
		private const val KEY_Y = "embedded_window_y"
		private const val KEY_WIDTH = "embedded_window_width"
		private const val KEY_HEIGHT = "embedded_window_height"

		private const val RESIZE_THRESHOLD = 100f
		private const val MIN_WINDOW_SIZE = 300
		private const val MAX_SCREEN_PERCENT = 0.85f
	}

	private val defaultWidthInPx: Int by lazy { resources.getDimensionPixelSize(R.dimen.embed_game_window_default_width) }
	private val defaultHeightInPx: Int by lazy { resources.getDimensionPixelSize(R.dimen.embed_game_window_default_height) }

	private var layoutWidthInPx = 0
	private var layoutHeightInPx = 0
	private var isFullscreen = false
	private var gameRequestedOrientation = ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED

	private var isEditMode = false
	private var isResizing = false
	private var isMoving = false
	private var activeCorner = 0
	private var isFreeResize = false

	private var initialWinX = 0
	private var initialWinY = 0
	private var initialWidth = 0
	private var initialHeight = 0
	private var initialTouchX = 0f
	private var initialTouchY = 0f

	private lateinit var dimensionLabel: TextView
	private val cornerHandles = mutableListOf<View>()

	private val screenBounds: android.graphics.Rect by lazy {
		if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
			windowManager.currentWindowMetrics.bounds
		} else {
			val size = Point()
			windowManager.defaultDisplay.getRealSize(size)
			android.graphics.Rect(0, 0, size.x, size.y)
		}
	}

	private val maxAllowedWidth: Int get() = (screenBounds.width() * MAX_SCREEN_PERCENT).toInt()
	private val maxAllowedHeight: Int get() = (screenBounds.height() * MAX_SCREEN_PERCENT).toInt()

	private val defaultAspectRatio: Float by lazy {
		defaultWidthInPx.toFloat() / defaultHeightInPx.toFloat()
	}

	override fun onCreate(savedInstanceState: Bundle?) {
		super.onCreate(savedInstanceState)

		setFinishOnTouchOutside(false)

		val layoutParams = window.attributes
		layoutParams.flags = layoutParams.flags or FLAG_NOT_TOUCH_MODAL or FLAG_WATCH_OUTSIDE_TOUCH
		layoutParams.flags = layoutParams.flags and FLAG_DIM_BEHIND.inv()
		layoutParams.gravity = Gravity.TOP or Gravity.START

		loadWindowBounds(layoutParams)
		window.attributes = layoutParams

		setupOverlayUI()
	}

	private fun setupOverlayUI() {
		dimensionLabel = TextView(this).apply {
			setTextColor(Color.WHITE)
			setBackgroundColor("#CC000000".toColorInt())
			setPadding(24, 12, 24, 12)
			visibility = View.GONE
			textSize = 14f
			isClickable = true
			isFocusable = true

			setOnClickListener {
				isFreeResize = !isFreeResize
				updateLabelText(window.attributes.width, window.attributes.height)
			}
		}

		val labelParams = FrameLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
			gravity = Gravity.CENTER_HORIZONTAL or Gravity.TOP
			topMargin = 60
		}
		addContentView(dimensionLabel, labelParams)

		val corners = listOf(Gravity.TOP or Gravity.START, Gravity.TOP or Gravity.END, Gravity.BOTTOM or Gravity.START, Gravity.BOTTOM or Gravity.END)
		corners.forEach { gravity ->
			val handle = View(this).apply {
				setBackgroundColor("#6600FF".toColorInt())
				visibility = View.GONE
			}
			addContentView(handle, FrameLayout.LayoutParams(80, 80, gravity))
			cornerHandles.add(handle)
		}
	}

	@SuppressLint("SetTextI18n")
	private fun updateLabelText(w: Int, h: Int) {
		val lockStatus = if (isFreeResize) "Free" else "Locked"
		dimensionLabel.text = "$w x $h\n$lockStatus"
	}

	override fun setRequestedOrientation(requestedOrientation: Int) {
		// Allow orientation change only if fullscreen mode is active
		// or if the requestedOrientation is unspecified (i.e switching to default).
		if (isFullscreen || requestedOrientation == ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED) {
			super.setRequestedOrientation(requestedOrientation)
		} else {
			// Cache the requestedOrientation to apply when switching to fullscreen.
			gameRequestedOrientation = requestedOrientation
		}
	}

	override fun dispatchTouchEvent(event: MotionEvent): Boolean {
		if (isFullscreen) return super.dispatchTouchEvent(event)
		val layoutParams = window.attributes

		when (event.actionMasked) {
			MotionEvent.ACTION_OUTSIDE -> {
				toggleEditMode(!isEditMode)
				return true
			}
			MotionEvent.ACTION_DOWN -> {
				if (isEditMode) {
					// Check if the click is inside the label's bounds
					val location = IntArray(2)
					dimensionLabel.getLocationOnScreen(location)
					val labelRect = android.graphics.Rect(
						location[0], location[1],
						location[0] + dimensionLabel.width,
						location[1] + dimensionLabel.height
					)

					if (labelRect.contains(event.rawX.toInt(), event.rawY.toInt())) {
						// Pass it so dimensionLabel can handle it's click
						return super.dispatchTouchEvent(event)
					}

					initialTouchX = event.rawX
					initialTouchY = event.rawY
					initialWinX = layoutParams.x
					initialWinY = layoutParams.y
					initialWidth = layoutParams.width
					initialHeight = layoutParams.height
					activeCorner = getTouchedCorner(event.x, event.y, layoutParams.width, layoutParams.height)

					isResizing = activeCorner != 0
					isMoving = !isResizing
					return true
				}
			}
			MotionEvent.ACTION_MOVE -> {
				if (isEditMode && (isResizing || isMoving)) {
					val dx = (event.rawX - initialTouchX).toInt()
					val dy = (event.rawY - initialTouchY).toInt()

					if (isResizing) {
						applyResizeLogic(layoutParams, dx, dy)
						updateLabelText(layoutParams.width, layoutParams.height)
					} else if (isMoving) {
						layoutParams.x = initialWinX + dx
						layoutParams.y = initialWinY + dy
					}
					window.attributes = layoutParams
					return true
				}
			}
			MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
				if (isResizing || isMoving) {
					isResizing = false
					isMoving = false
					saveWindowBounds()
					return true
				}
			}
		}
		return super.dispatchTouchEvent(event)
	}

	private fun applyResizeLogic(layoutParams: WindowManager.LayoutParams, dx: Int, dy: Int) {
		var newW = initialWidth
		var newH = initialHeight

		when (activeCorner) {
			Gravity.TOP or Gravity.START -> {
				newW = (initialWidth - dx).coerceIn(MIN_WINDOW_SIZE, maxAllowedWidth)
				newH = if (isFreeResize) {
					(initialHeight - dy).coerceIn(MIN_WINDOW_SIZE, maxAllowedHeight)
				} else {
					(newW / defaultAspectRatio).toInt()
				}
				layoutParams.x = initialWinX + (initialWidth - newW)
				layoutParams.y = initialWinY + (initialHeight - newH)
			}

			Gravity.TOP or Gravity.END -> {
				newW = (initialWidth + dx).coerceIn(MIN_WINDOW_SIZE, maxAllowedWidth)
				newH = if (isFreeResize) {
					(initialHeight - dy).coerceIn(MIN_WINDOW_SIZE, maxAllowedHeight)
				} else {
					(newW / defaultAspectRatio).toInt()
				}
				layoutParams.y = initialWinY + (initialHeight - newH)
			}

			Gravity.BOTTOM or Gravity.START -> {
				newW = (initialWidth - dx).coerceIn(MIN_WINDOW_SIZE, maxAllowedWidth)
				newH = if (isFreeResize) {
					(initialHeight + dy).coerceIn(MIN_WINDOW_SIZE, maxAllowedHeight)
				} else {
					(newW / defaultAspectRatio).toInt()
				}
				layoutParams.x = initialWinX + (initialWidth - newW)
			}

			Gravity.BOTTOM or Gravity.END -> {
				newW = (initialWidth + dx).coerceIn(MIN_WINDOW_SIZE, maxAllowedWidth)
				newH = if (isFreeResize) {
					(initialHeight + dy).coerceIn(MIN_WINDOW_SIZE, maxAllowedHeight)
				} else {
					(newW / defaultAspectRatio).toInt()
				}
			}
		}

		// Final aspect-lock check
		if (!isFreeResize && newH > maxAllowedHeight) {
			newH = maxAllowedHeight
			newW = (newH * defaultAspectRatio).toInt()

			// Re-adjust pivots if it hits the vertical cap
			if (activeCorner == (Gravity.TOP or Gravity.START) || activeCorner == (Gravity.TOP or Gravity.END)) {
				layoutParams.y = initialWinY + (initialHeight - newH)
			}
			if (activeCorner == (Gravity.TOP or Gravity.START) || activeCorner == (Gravity.BOTTOM or Gravity.START)) {
				layoutParams.x = initialWinX + (initialWidth - newW)
			}
		}

		layoutParams.width = newW
		layoutParams.height = newH

		val hittingLimit = newW >= maxAllowedWidth || newH >= maxAllowedHeight
		dimensionLabel.setTextColor(if (hittingLimit) Color.RED else Color.WHITE)
	}

	private fun getTouchedCorner(x: Float, y: Float, w: Int, h: Int): Int {
		return when {
			x < RESIZE_THRESHOLD && y < RESIZE_THRESHOLD -> Gravity.TOP or Gravity.START
			x > w - RESIZE_THRESHOLD && y < RESIZE_THRESHOLD -> Gravity.TOP or Gravity.END
			x < RESIZE_THRESHOLD && y > h - RESIZE_THRESHOLD -> Gravity.BOTTOM or Gravity.START
			x > w - RESIZE_THRESHOLD && y > h - RESIZE_THRESHOLD -> Gravity.BOTTOM or Gravity.END
			else -> 0
		}
	}

	private fun toggleEditMode(enabled: Boolean) {
		isEditMode = enabled
		val visibility = if (enabled) View.VISIBLE else View.GONE

		cornerHandles.forEach { it.visibility = visibility }
		dimensionLabel.visibility = visibility

		if (enabled) {
			updateLabelText(window.attributes.width, window.attributes.height)
			window.addFlags(FLAG_DIM_BEHIND)
			window.setDimAmount(0.15f)
		} else {
			window.clearFlags(FLAG_DIM_BEHIND)
			window.setDimAmount(0.0f)
		}
	}

	private fun saveWindowBounds() {
		if (isFullscreen) return

		val layoutParams = window.attributes
		getSharedPreferences(PREFS_NAME, MODE_PRIVATE).edit().apply {
			putInt(KEY_X, layoutParams.x)
			putInt(KEY_Y, layoutParams.y)
			putInt(KEY_WIDTH, layoutParams.width)
			putInt(KEY_HEIGHT, layoutParams.height)
			apply()
		}
	}

	private fun loadWindowBounds(layoutParams: WindowManager.LayoutParams) {
		val prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
		layoutParams.x = prefs.getInt(KEY_X, 100)
		layoutParams.y = prefs.getInt(KEY_Y, 100)
		layoutWidthInPx = prefs.getInt(KEY_WIDTH, defaultWidthInPx)
		layoutHeightInPx = prefs.getInt(KEY_HEIGHT, defaultHeightInPx)
		layoutParams.width = layoutWidthInPx
		layoutParams.height = layoutHeightInPx
	}

	override fun getEditorWindowInfo() = EMBEDDED_RUN_GAME_INFO

	override fun getEditorGameEmbedMode() = GameMenuUtils.GameEmbedMode.ENABLED

	override fun isGameEmbedded() = true

	override fun isMinimizedButtonEnabled() = true

	override fun isCloseButtonEnabled() = true

	override fun isFullScreenButtonEnabled() = true

	override fun isPiPButtonEnabled() = false

	override fun isMenuBarCollapsable() = false

	override fun isAlwaysOnTopSupported() = isPiPModeSupported()

	override fun minimizeGameWindow() {
		if (!isFullscreen && gameMenuFragment?.isAlwaysOnTop() == true) {
			enterPiPMode()
		} else {
			moveTaskToBack(false)
		}
	}

	override fun onFullScreenUpdated(enabled: Boolean) {
		godot?.enableImmersiveMode(enabled)
		isFullscreen = enabled

		val layoutParams = window.attributes
		if (enabled) {
			toggleEditMode(false)
			layoutWidthInPx = WindowManager.LayoutParams.MATCH_PARENT
			layoutHeightInPx = WindowManager.LayoutParams.MATCH_PARENT
			requestedOrientation = gameRequestedOrientation
			layoutParams.x = 0
			layoutParams.y = 0
		} else {
			loadWindowBounds(layoutParams)

			// Cache the last used orientation in fullscreen to reapply when re-entering fullscreen.
			gameRequestedOrientation = requestedOrientation
			requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED
		}
		updateWindowDimensions(layoutWidthInPx, layoutHeightInPx)
	}

	private fun updateWindowDimensions(widthInPx: Int, heightInPx: Int) {
		val layoutParams = window.attributes
		layoutParams.width = widthInPx
		layoutParams.height = heightInPx
		window.attributes = layoutParams
	}

	override fun onPictureInPictureModeChanged(isInPictureInPictureMode: Boolean) {
		super.onPictureInPictureModeChanged(isInPictureInPictureMode)
		// Maximize the dimensions when entering PiP so the window fills the full PiP bounds.
		onFullScreenUpdated(isInPictureInPictureMode)
	}

	override fun shouldShowGameMenuBar() = gameMenuContainer != null
}
