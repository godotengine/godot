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
import android.os.Bundle
import android.view.Gravity
import android.view.MotionEvent
import android.view.WindowManager
import android.view.WindowManager.LayoutParams.FLAG_DIM_BEHIND
import android.view.WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL
import android.view.WindowManager.LayoutParams.FLAG_WATCH_OUTSIDE_TOUCH
import org.godotengine.editor.GodotGame
import org.godotengine.editor.R
import org.godotengine.godot.editor.utils.GameMenuUtils

/**
 * Host the Godot game from the editor when the embedded mode is enabled.
 */
class EmbeddedGodotGame : GodotGame() {

	companion object {
		private val TAG = EmbeddedGodotGame::class.java.simpleName

		private const val FULL_SCREEN_WIDTH = WindowManager.LayoutParams.MATCH_PARENT
		private const val FULL_SCREEN_HEIGHT = WindowManager.LayoutParams.MATCH_PARENT
	}

	private val defaultWidthInPx : Int by lazy {
		resources.getDimensionPixelSize(R.dimen.embed_game_window_default_width)
	}
	private val defaultHeightInPx : Int by lazy {
		resources.getDimensionPixelSize(R.dimen.embed_game_window_default_height)
	}

	private var layoutWidthInPx = 0
	private var layoutHeightInPx = 0

	private var gameRequestedOrientation = ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED
	private var isFullscreen = false

	override fun onCreate(savedInstanceState: Bundle?) {
		super.onCreate(savedInstanceState)

		setFinishOnTouchOutside(false)

		val layoutParams = window.attributes
		layoutParams.flags = layoutParams.flags or FLAG_NOT_TOUCH_MODAL or FLAG_WATCH_OUTSIDE_TOUCH
		layoutParams.flags = layoutParams.flags and FLAG_DIM_BEHIND.inv()
		layoutParams.gravity = Gravity.END or Gravity.BOTTOM

		layoutWidthInPx = defaultWidthInPx
		layoutHeightInPx = defaultHeightInPx

		layoutParams.width = layoutWidthInPx
		layoutParams.height = layoutHeightInPx
		window.attributes = layoutParams
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
		when (event.actionMasked) {
			MotionEvent.ACTION_OUTSIDE -> {
				if (!isFullscreen) {
					if (gameMenuFragment?.isAlwaysOnTop() == true) {
						enterPiPMode()
					} else {
						minimizeGameWindow()
					}
				}
			}

			MotionEvent.ACTION_MOVE -> {
//				val layoutParams = window.attributes
				// TODO: Add logic to move the embedded window.
//				window.attributes = layoutParams
			}
		}
		return super.dispatchTouchEvent(event)
	}

	override fun getEditorWindowInfo() = EMBEDDED_RUN_GAME_INFO

	override fun getEditorGameEmbedMode() = GameMenuUtils.GameEmbedMode.ENABLED

	override fun isGameEmbedded() = true

	private fun updateWindowDimensions(widthInPx: Int, heightInPx: Int) {
		val layoutParams = window.attributes
		layoutParams.width = widthInPx
		layoutParams.height = heightInPx
		window.attributes = layoutParams
	}

	override fun isMinimizedButtonEnabled() = true

	override fun isCloseButtonEnabled() = true

	override fun isFullScreenButtonEnabled() = true

	override fun isPiPButtonEnabled() = false

	override fun isMenuBarCollapsable() = false

	override fun isAlwaysOnTopSupported() = hasPiPSystemFeature()

	override fun onFullScreenUpdated(enabled: Boolean) {
		godot?.enableImmersiveMode(enabled)
		isFullscreen = enabled
		if (enabled) {
			layoutWidthInPx = FULL_SCREEN_WIDTH
			layoutHeightInPx = FULL_SCREEN_HEIGHT
			requestedOrientation = gameRequestedOrientation
		} else {
			layoutWidthInPx = defaultWidthInPx
			layoutHeightInPx = defaultHeightInPx

			// Cache the last used orientation in fullscreen to reapply when re-entering fullscreen.
			gameRequestedOrientation = requestedOrientation
			requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED
		}
		updateWindowDimensions(layoutWidthInPx, layoutHeightInPx)
	}

	override fun onPictureInPictureModeChanged(isInPictureInPictureMode: Boolean) {
		super.onPictureInPictureModeChanged(isInPictureInPictureMode)
		// Maximize the dimensions when entering PiP so the window fills the full PiP bounds.
		onFullScreenUpdated(isInPictureInPictureMode)
	}

	override fun shouldShowGameMenuBar() = gameMenuContainer != null
}
