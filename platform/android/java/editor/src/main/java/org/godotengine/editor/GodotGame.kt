/**************************************************************************/
/*  GodotGame.kt                                                          */
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

package org.godotengine.editor

import android.Manifest
import android.annotation.SuppressLint
import android.app.PictureInPictureParams
import android.content.Intent
import android.graphics.Rect
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.View
import androidx.annotation.CallSuper
import org.godotengine.godot.GodotLib
import org.godotengine.godot.utils.PermissionsUtil
import org.godotengine.godot.utils.ProcessPhoenix

/**
 * Drives the 'run project' window of the Godot Editor.
 */
open class GodotGame : GodotEditor() {

	companion object {
		private val TAG = GodotGame::class.java.simpleName
	}

	private val gameViewSourceRectHint = Rect()
	private val pipButton: View? by lazy {
		findViewById(R.id.godot_pip_button)
	}

	private var pipAvailable = false

	@SuppressLint("ClickableViewAccessibility")
	override fun onCreate(savedInstanceState: Bundle?) {
		super.onCreate(savedInstanceState)

		if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
			val gameView = findViewById<View>(R.id.godot_fragment_container)
			gameView?.addOnLayoutChangeListener { v, left, top, right, bottom, oldLeft, oldTop, oldRight, oldBottom ->
				gameView.getGlobalVisibleRect(gameViewSourceRectHint)
			}
		}

		pipButton?.setOnClickListener { enterPiPMode() }

		handleStartIntent(intent)
	}

	override fun onNewIntent(newIntent: Intent) {
		super.onNewIntent(newIntent)
		handleStartIntent(newIntent)
	}

	private fun handleStartIntent(intent: Intent) {
		pipAvailable = intent.getBooleanExtra(EXTRA_PIP_AVAILABLE, pipAvailable)
		updatePiPButtonVisibility()

		val pipLaunchRequested = intent.getBooleanExtra(EXTRA_LAUNCH_IN_PIP, false)
		if (pipLaunchRequested) {
			enterPiPMode()
		}
	}

	private fun updatePiPButtonVisibility() {
		pipButton?.visibility = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N && pipAvailable && !isInPictureInPictureMode) {
			View.VISIBLE
		} else {
			View.GONE
		}
	}

	private fun enterPiPMode() {
		if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N && pipAvailable) {
			if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
				val builder = PictureInPictureParams.Builder().setSourceRectHint(gameViewSourceRectHint)
				if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
					builder.setSeamlessResizeEnabled(false)
				}
				setPictureInPictureParams(builder.build())
			}

			Log.v(TAG, "Entering PiP mode")
			enterPictureInPictureMode()
		}
	}

	override fun onPictureInPictureModeChanged(isInPictureInPictureMode: Boolean) {
		super.onPictureInPictureModeChanged(isInPictureInPictureMode)
		Log.v(TAG, "onPictureInPictureModeChanged: $isInPictureInPictureMode")
		updatePiPButtonVisibility()
	}

	override fun onStop() {
		super.onStop()

		val isInPiPMode = Build.VERSION.SDK_INT >= Build.VERSION_CODES.N && isInPictureInPictureMode
		if (isInPiPMode && !isFinishing) {
			// We get in this state when PiP is closed, so we terminate the activity.
			finish()
		}
	}

	override fun getGodotAppLayout() = R.layout.godot_game_layout

	override fun getEditorWindowInfo() = RUN_GAME_INFO

	override fun overrideOrientationRequest() = false

	override fun enableLongPressGestures() = java.lang.Boolean.parseBoolean(GodotLib.getGlobal("input_devices/pointing/android/enable_long_press_as_right_click"))

	override fun enablePanAndScaleGestures() = java.lang.Boolean.parseBoolean(GodotLib.getGlobal("input_devices/pointing/android/enable_pan_and_scale_gestures"))

	override fun onGodotSetupCompleted() {
		super.onGodotSetupCompleted()
		Log.v(TAG, "OnGodotSetupCompleted")

		// Check if we should be running in XR instead (if available) as it's possible we were
		// launched from the project manager which doesn't have that information.
		val launchingArgs = intent.getStringArrayExtra(EXTRA_COMMAND_LINE_PARAMS)
		if (launchingArgs != null) {
			val editorWindowInfo = retrieveEditorWindowInfo(launchingArgs)
			if (editorWindowInfo != getEditorWindowInfo()) {
				val relaunchIntent = getNewGodotInstanceIntent(editorWindowInfo, launchingArgs)
				relaunchIntent.putExtra(EXTRA_NEW_LAUNCH, true)
					.putExtra(EditorMessageDispatcher.EXTRA_MSG_DISPATCHER_PAYLOAD, intent.getBundleExtra(EditorMessageDispatcher.EXTRA_MSG_DISPATCHER_PAYLOAD))

				Log.d(TAG, "Relaunching XR project using ${editorWindowInfo.windowClassName} with parameters ${launchingArgs.contentToString()}")
				val godot = godot
				if (godot != null) {
					godot.destroyAndKillProcess {
						ProcessPhoenix.triggerRebirth(this, relaunchIntent)
					}
				} else {
					ProcessPhoenix.triggerRebirth(this, relaunchIntent)
				}
				return
			}
		}

		// Request project runtime permissions if necessary
		val permissionsToEnable = getProjectPermissionsToEnable()
		if (permissionsToEnable.isNotEmpty()) {
			PermissionsUtil.requestPermissions(this, permissionsToEnable)
		}
	}

	/**
	 * Check for project permissions to enable
	 */
	@CallSuper
	protected open fun getProjectPermissionsToEnable(): MutableList<String> {
		val permissionsToEnable = mutableListOf<String>()

		// Check for RECORD_AUDIO permission
		val audioInputEnabled = java.lang.Boolean.parseBoolean(GodotLib.getGlobal("audio/driver/enable_input"))
		if (audioInputEnabled) {
			permissionsToEnable.add(Manifest.permission.RECORD_AUDIO)
		}

		return permissionsToEnable
	}
}
