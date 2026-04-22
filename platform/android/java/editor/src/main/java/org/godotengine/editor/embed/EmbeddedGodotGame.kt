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

import android.content.pm.ActivityInfo
import android.graphics.Color
import android.graphics.Point
import android.os.Bundle
import android.util.Rational
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.view.WindowManager.LayoutParams.FLAG_DIM_BEHIND
import android.view.WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL
import android.view.WindowManager.LayoutParams.FLAG_WATCH_OUTSIDE_TOUCH
import android.widget.CheckBox
import org.godotengine.editor.GodotGame
import org.godotengine.editor.R
import org.godotengine.godot.editor.utils.GameMenuUtils

/**
 * Host the Godot game from the editor when the embedded mode is enabled.
 */
class EmbeddedGodotGame : GodotGame() {

	companion object {
		private val TAG = EmbeddedGodotGame::class.java.simpleName

		private const val PREFS_NAME = "embedded_game_window_prefs"
		private const val KEY_X = "embedded_window_x"
		private const val KEY_Y = "embedded_window_y"
		private const val KEY_WIDTH = "embedded_window_width"
		private const val KEY_HEIGHT = "embedded_window_height"
		private const val KEY_FREE_RESIZE = "is_free_resize"

		private const val RESIZE_THRESHOLD = 80f
		private const val MIN_WINDOW_SIZE = 400
		private const val MAX_SCREEN_PERCENT = 0.9f

		private const val RESIZE_UI_HIDE_DELAY_MS = 2000L
	}

	private val defaultWidthInPx: Int by lazy { resources.getDimensionPixelSize(R.dimen.embed_game_window_default_width) }
	private val defaultHeightInPx: Int by lazy { resources.getDimensionPixelSize(R.dimen.embed_game_window_default_height) }

	private var layoutWidthInPx = 0
	private var layoutHeightInPx = 0
	private var isFullscreen = false
	private var gameRequestedOrientation = ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED

	private var resizingEnabled = false
	private var isResizing = false
	private var activeCorner = 0
	private var isFreeResize = false

	private var initialWinX = 0
	private var initialWinY = 0
	private var initialWidth = 0
	private var initialHeight = 0
	private var initialTouchX = 0f
	private var initialTouchY = 0f

	private val lockAspectRatioCheckBox: CheckBox by lazy { findViewById(R.id.lockAspectRatioCheckBox) }
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

	private var lockedAspectRatio: Float = 1.6f

	private val disableResizeHandler = android.os.Handler(android.os.Looper.getMainLooper())

	private val disableResizeRunnable = Runnable {
		if (isResizing) return@Runnable // Keep it active user is resizing again
		resizingEnabled = false
		gameMenuFragment?.toggleDragButton(false)
		lockAspectRatioCheckBox.visibility = View.GONE
		cornerHandles.forEach { it.visibility = View.GONE }
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

	override fun getGodotAppLayout() = R.layout.godot_embedded_game_layout

	private fun setupOverlayUI() {
		lockAspectRatioCheckBox.isChecked = !isFreeResize

		lockAspectRatioCheckBox.setOnCheckedChangeListener { _, isChecked ->
			isFreeResize = !isChecked
			hideResizeUI()
			if (isChecked) {
				val lp = window.attributes
				lockedAspectRatio = lp.width.toFloat() / lp.height.toFloat().coerceAtLeast(1f)
			}
			saveWindowBounds(updatePiPParams = false)
		}

		cornerHandles.apply {
			clear()
			add(findViewById(R.id.handleTopLeft))
			add(findViewById(R.id.handleTopRight))
			add(findViewById(R.id.handleBottomLeft))
			add(findViewById(R.id.handleBottomRight))
		}
	}

	private fun updateLabelText(w: Int, h: Int) {
		lockAspectRatioCheckBox.text = getString(R.string.lock_aspect_ratio_btn_text, w, h)
	}

	private fun showResizeUI() {
		disableResizeHandler.removeCallbacks(disableResizeRunnable)
		if (!resizingEnabled) {
			resizingEnabled = true
			lockAspectRatioCheckBox.visibility = View.VISIBLE
			cornerHandles.forEach { it.visibility = View.VISIBLE }
		}
	}

	private fun hideResizeUI() {
		disableResizeHandler.removeCallbacks(disableResizeRunnable)
		disableResizeHandler.postDelayed(disableResizeRunnable, RESIZE_UI_HIDE_DELAY_MS)
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
				if (gameMenuFragment?.isAlwaysOnTop() == true) {
					updatePiPParams(aspectRatio = Rational(layoutWidthInPx, layoutHeightInPx))
					enterPiPMode()
				} else {
					minimizeGameWindow()
				}
			}
			MotionEvent.ACTION_DOWN -> {
				if (resizingEnabled) {
					// Check if the click is inside the label's bounds
					val location = IntArray(2)
					lockAspectRatioCheckBox.getLocationOnScreen(location)
					val checkBoxRect = android.graphics.Rect(
						location[0], location[1],
						location[0] + lockAspectRatioCheckBox.width,
						location[1] + lockAspectRatioCheckBox.height
					)

					if (checkBoxRect.contains(event.rawX.toInt(), event.rawY.toInt())) {
						// Let the CheckBox handle the click itself
						return super.dispatchTouchEvent(event)
					}
					activeCorner = getTouchedCorner(event.x, event.y, layoutParams.width, layoutParams.height)
					if (activeCorner != 0) {
						isResizing = true

						initialTouchX = event.rawX
						initialTouchY = event.rawY
						initialWinX = layoutParams.x
						initialWinY = layoutParams.y
						initialWidth = layoutParams.width
						initialHeight = layoutParams.height
						return true
					}
				}
			}
			MotionEvent.ACTION_MOVE -> {
				if (resizingEnabled && isResizing) {
					val dx = (event.rawX - initialTouchX).toInt()
					val dy = (event.rawY - initialTouchY).toInt()
					applyResizeLogic(layoutParams, dx, dy)
					updateLabelText(layoutParams.width, layoutParams.height)
					window.attributes = layoutParams
					return true
				}
			}
			MotionEvent.ACTION_UP -> {
				if (isResizing) {
					isResizing = false
					saveWindowBounds()
					hideResizeUI()
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
					(newW / lockedAspectRatio).toInt()
				}
				layoutParams.x = initialWinX + (initialWidth - newW)
				layoutParams.y = initialWinY + (initialHeight - newH)
			}

			Gravity.TOP or Gravity.END -> {
				newW = (initialWidth + dx).coerceIn(MIN_WINDOW_SIZE, maxAllowedWidth)
				newH = if (isFreeResize) {
					(initialHeight - dy).coerceIn(MIN_WINDOW_SIZE, maxAllowedHeight)
				} else {
					(newW / lockedAspectRatio).toInt()
				}
				layoutParams.y = initialWinY + (initialHeight - newH)
			}

			Gravity.BOTTOM or Gravity.START -> {
				newW = (initialWidth - dx).coerceIn(MIN_WINDOW_SIZE, maxAllowedWidth)
				newH = if (isFreeResize) {
					(initialHeight + dy).coerceIn(MIN_WINDOW_SIZE, maxAllowedHeight)
				} else {
					(newW / lockedAspectRatio).toInt()
				}
				layoutParams.x = initialWinX + (initialWidth - newW)
			}

			Gravity.BOTTOM or Gravity.END -> {
				newW = (initialWidth + dx).coerceIn(MIN_WINDOW_SIZE, maxAllowedWidth)
				newH = if (isFreeResize) {
					(initialHeight + dy).coerceIn(MIN_WINDOW_SIZE, maxAllowedHeight)
				} else {
					(newW / lockedAspectRatio).toInt()
				}
			}
		}

		// Final aspect-lock check
		if (!isFreeResize) {
			// Cap height based on both max height and width-constrained limit.
			// effectiveMaxHeight helps ensure that the width (newW) calculated below is guaranteed to be <= maxAllowedWidth.
			val maxHeightFromWidth = (maxAllowedWidth / lockedAspectRatio).toInt()
			val effectiveMaxHeight = minOf(maxAllowedHeight, maxHeightFromWidth)

			val clampedH = newH.coerceIn(MIN_WINDOW_SIZE, effectiveMaxHeight)

			if (clampedH != newH) {
				newH = clampedH
				newW = (newH * lockedAspectRatio).toInt()

				// Re-adjust pivots for the new clamped dimensions
				if (activeCorner == (Gravity.TOP or Gravity.START) || activeCorner == (Gravity.TOP or Gravity.END)) {
					layoutParams.y = initialWinY + (initialHeight - newH)
				}
				if (activeCorner == (Gravity.TOP or Gravity.START) || activeCorner == (Gravity.BOTTOM or Gravity.START)) {
					layoutParams.x = initialWinX + (initialWidth - newW)
				}
			}
		}

		layoutParams.width = newW
		layoutParams.height = newH

		val hittingLimit = newW >= maxAllowedWidth || newH >= maxAllowedHeight
		lockAspectRatioCheckBox.setTextColor(if (hittingLimit) Color.RED else Color.WHITE)
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

	private fun saveWindowBounds(updatePiPParams: Boolean = true) {
		if (isFullscreen) return

		val layoutParams = window.attributes
		layoutWidthInPx = layoutParams.width
		layoutHeightInPx = layoutParams.height
		getSharedPreferences(PREFS_NAME, MODE_PRIVATE).edit().apply {
			putInt(KEY_X, layoutParams.x)
			putInt(KEY_Y, layoutParams.y)
			putInt(KEY_WIDTH, layoutParams.width)
			putInt(KEY_HEIGHT, layoutParams.height)
			putBoolean(KEY_FREE_RESIZE, isFreeResize)
			apply()
		}
		if (updatePiPParams) {
			updatePiPParams(aspectRatio = Rational(layoutParams.width, layoutParams.height))
		}
	}

	private fun loadWindowBounds(layoutParams: WindowManager.LayoutParams) {
		val prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
		isFreeResize = prefs.getBoolean(KEY_FREE_RESIZE, false)
		layoutWidthInPx = prefs.getInt(KEY_WIDTH, defaultWidthInPx)
		layoutHeightInPx = prefs.getInt(KEY_HEIGHT, defaultHeightInPx)
		layoutParams.x = prefs.getInt(KEY_X, screenBounds.width() - layoutWidthInPx)
		layoutParams.y = prefs.getInt(KEY_Y, screenBounds.height() - layoutHeightInPx)
		layoutParams.width = layoutWidthInPx
		layoutParams.height = layoutHeightInPx
		lockedAspectRatio = layoutParams.width.toFloat() / layoutParams.height.toFloat().coerceAtLeast(1f)
		updateLabelText(layoutWidthInPx, layoutHeightInPx)
	}

	override fun dragGameWindow(view: View, event: MotionEvent): Boolean {
		if (isFullscreen) return false
		val lp = window.attributes
		when (event.actionMasked) {
			MotionEvent.ACTION_DOWN -> {
				showResizeUI()
				gameMenuFragment?.toggleDragButton(true)
				initialTouchX = event.rawX
				initialTouchY = event.rawY
				initialWinX = lp.x
				initialWinY = lp.y
				return true
			}
			MotionEvent.ACTION_MOVE -> {
				lp.x = (initialWinX + (event.rawX - initialTouchX)).toInt()
				lp.y = (initialWinY + (event.rawY - initialTouchY)).toInt()
				window.attributes = lp
				return true
			}
			MotionEvent.ACTION_UP -> {
				saveWindowBounds(updatePiPParams = false) // Only window position is changed, no need to update aspect ratio.
				hideResizeUI()
				return true
			}
		}
		return false
	}

	override fun getEditorWindowInfo() = EMBEDDED_RUN_GAME_INFO

	override fun getEditorGameEmbedMode() = GameMenuUtils.GameEmbedMode.ENABLED

	override fun isGameEmbedded() = true

	override fun isMinimizedButtonEnabled() = isFullscreen

	override fun isCloseButtonEnabled() = true

	override fun isFullScreenButtonEnabled() = true

	override fun isPiPButtonEnabled() = false

	override fun isMenuBarCollapsable() = false

	override fun isDragButtonEnabled() = !isFullscreen

	override fun isAlwaysOnTopSupported() = isPiPModeSupported()

	override fun onFullScreenUpdated(enabled: Boolean) {
		godot?.enableImmersiveMode(enabled)
		isFullscreen = enabled

		val layoutParams = window.attributes
		if (enabled) {
			layoutWidthInPx = WindowManager.LayoutParams.MATCH_PARENT
			layoutHeightInPx = WindowManager.LayoutParams.MATCH_PARENT
			requestedOrientation = gameRequestedOrientation
			layoutParams.x = 0
			layoutParams.y = 0
			if (resizingEnabled) {
				disableResizeRunnable.run()
			}
		} else {
			loadWindowBounds(layoutParams)

			// Cache the last used orientation in fullscreen to reapply when re-entering fullscreen.
			gameRequestedOrientation = requestedOrientation
			requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED
		}
		updateWindowDimensions(layoutWidthInPx, layoutHeightInPx)
		gameMenuFragment?.refreshButtonsVisibility()
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
