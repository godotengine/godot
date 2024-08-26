/**************************************************************************/
/*  GodotEditor.kt                                                        */
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
import android.app.ActivityManager
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.*
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.annotation.CallSuper
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.window.layout.WindowMetricsCalculator
import org.godotengine.editor.utils.signApk
import org.godotengine.editor.utils.verifyApk
import org.godotengine.godot.GodotActivity
import org.godotengine.godot.GodotLib
import org.godotengine.godot.error.Error
import org.godotengine.godot.utils.PermissionsUtil
import org.godotengine.godot.utils.ProcessPhoenix
import java.util.*
import kotlin.math.min

/**
 * Base class for the Godot Android Editor activities.
 *
 * This provides the basic templates for the activities making up this application.
 * Each derived activity runs in its own process, which enable up to have several instances of
 * the Godot engine up and running at the same time.
 *
 * It also plays the role of the primary editor window.
 */
open class GodotEditor : GodotActivity() {

	companion object {
		private val TAG = GodotEditor::class.java.simpleName

		private const val WAIT_FOR_DEBUGGER = false

		private const val EXTRA_COMMAND_LINE_PARAMS = "command_line_params"

		// Command line arguments
		private const val EDITOR_ARG = "--editor"
		private const val EDITOR_ARG_SHORT = "-e"
		private const val EDITOR_PROJECT_MANAGER_ARG = "--project-manager"
		private const val EDITOR_PROJECT_MANAGER_ARG_SHORT = "-p"

		// Info for the various classes used by the editor
		internal val EDITOR_MAIN_INFO = EditorWindowInfo(GodotEditor::class.java, 777, "")
		internal val RUN_GAME_INFO = EditorWindowInfo(GodotGame::class.java, 667, ":GodotGame", LaunchAdjacentPolicy.AUTO)

		/**
		 * Sets of constants to specify the window to use to run the project.
		 *
		 * Should match the values in 'editor/editor_settings.cpp' for the
		 * 'run/window_placement/android_window' setting.
		 */
		private const val ANDROID_WINDOW_AUTO = 0
		private const val ANDROID_WINDOW_SAME_AS_EDITOR = 1
		private const val ANDROID_WINDOW_SIDE_BY_SIDE_WITH_EDITOR = 2
	}

	private val commandLineParams = ArrayList<String>()
	private val editorLoadingIndicator: View? by lazy { findViewById(R.id.editor_loading_indicator) }

	override fun getGodotAppLayout() = R.layout.godot_editor_layout

	override fun onCreate(savedInstanceState: Bundle?) {
		installSplashScreen()

		// We exclude certain permissions from the set we request at startup, as they'll be
		// requested on demand based on use-cases.
		PermissionsUtil.requestManifestPermissions(this, setOf(Manifest.permission.RECORD_AUDIO))

		val params = intent.getStringArrayExtra(EXTRA_COMMAND_LINE_PARAMS)
		Log.d(TAG, "Starting intent $intent with parameters ${params.contentToString()}")
		updateCommandLineParams(params?.asList() ?: emptyList())

		if (BuildConfig.BUILD_TYPE == "dev" && WAIT_FOR_DEBUGGER) {
			Debug.waitForDebugger()
		}

		super.onCreate(savedInstanceState)
	}

	override fun onGodotSetupCompleted() {
		super.onGodotSetupCompleted()
		val longPressEnabled = enableLongPressGestures()
		val panScaleEnabled = enablePanAndScaleGestures()

		checkForProjectPermissionsToEnable()

		runOnUiThread {
			// Enable long press, panning and scaling gestures
			godotFragment?.godot?.renderView?.inputHandler?.apply {
				enableLongPress(longPressEnabled)
				enablePanningAndScalingGestures(panScaleEnabled)
			}
		}
	}

	override fun onGodotMainLoopStarted() {
		super.onGodotMainLoopStarted()
		runOnUiThread {
			// Hide the loading indicator
			editorLoadingIndicator?.visibility = View.GONE
		}
	}

	/**
	 * Check for project permissions to enable
	 */
	protected open fun checkForProjectPermissionsToEnable() {
		// Check for RECORD_AUDIO permission
		val audioInputEnabled = java.lang.Boolean.parseBoolean(GodotLib.getGlobal("audio/driver/enable_input"))
		if (audioInputEnabled) {
			PermissionsUtil.requestPermission(Manifest.permission.RECORD_AUDIO, this)
		}
	}

	@CallSuper
	protected open fun updateCommandLineParams(args: List<String>) {
		// Update the list of command line params with the new args
		commandLineParams.clear()
		if (args.isNotEmpty()) {
			commandLineParams.addAll(args)
		}
		if (BuildConfig.BUILD_TYPE == "dev") {
			commandLineParams.add("--benchmark")
		}
	}

	final override fun getCommandLine() = commandLineParams

	protected open fun getEditorWindowInfo(args: Array<String>): EditorWindowInfo {
		var hasEditor = false

		var i = 0
		while (i < args.size) {
			when (args[i++]) {
				EDITOR_ARG, EDITOR_ARG_SHORT, EDITOR_PROJECT_MANAGER_ARG, EDITOR_PROJECT_MANAGER_ARG_SHORT -> hasEditor = true
			}
		}

		return if (hasEditor) {
			EDITOR_MAIN_INFO
		} else {
			RUN_GAME_INFO
		}
	}

	protected open fun getEditorWindowInfoForInstanceId(instanceId: Int): EditorWindowInfo? {
		return when (instanceId) {
			RUN_GAME_INFO.windowId -> RUN_GAME_INFO
			EDITOR_MAIN_INFO.windowId -> EDITOR_MAIN_INFO
			else -> null
		}
	}

	override fun onNewGodotInstanceRequested(args: Array<String>): Int {
		val editorWindowInfo = getEditorWindowInfo(args)

		// Launch a new activity
		val newInstance = Intent()
			.setComponent(ComponentName(this, editorWindowInfo.windowClassName))
			.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
			.putExtra(EXTRA_COMMAND_LINE_PARAMS, args)
		if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
			if (editorWindowInfo.launchAdjacentPolicy == LaunchAdjacentPolicy.ENABLED ||
				(editorWindowInfo.launchAdjacentPolicy == LaunchAdjacentPolicy.AUTO && shouldGameLaunchAdjacent())) {
				Log.v(TAG, "Adding flag for adjacent launch")
				newInstance.addFlags(Intent.FLAG_ACTIVITY_LAUNCH_ADJACENT)
			}
		}
		if (editorWindowInfo.windowClassName == javaClass.name) {
			Log.d(TAG, "Restarting ${editorWindowInfo.windowClassName} with parameters ${args.contentToString()}")
			val godot = godot
			if (godot != null) {
				godot.destroyAndKillProcess {
					ProcessPhoenix.triggerRebirth(this, newInstance)
				}
			} else {
				ProcessPhoenix.triggerRebirth(this, newInstance)
			}
		} else {
			Log.d(TAG, "Starting ${editorWindowInfo.windowClassName} with parameters ${args.contentToString()}")
			newInstance.putExtra(EXTRA_NEW_LAUNCH, true)
			startActivity(newInstance)
		}
		return editorWindowInfo.windowId
	}

	final override fun onGodotForceQuit(godotInstanceId: Int): Boolean {
		val editorWindowInfo = getEditorWindowInfoForInstanceId(godotInstanceId) ?: return super.onGodotForceQuit(godotInstanceId)

		if (editorWindowInfo.windowClassName == javaClass.name) {
			Log.d(TAG, "Force quitting ${editorWindowInfo.windowClassName}")
			ProcessPhoenix.forceQuit(this)
			return true
		}

		val processName = packageName + editorWindowInfo.processNameSuffix
		val activityManager = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
		val runningProcesses = activityManager.runningAppProcesses
		for (runningProcess in runningProcesses) {
			if (runningProcess.processName == processName) {
				// Killing process directly
				Log.v(TAG, "Killing Godot process ${runningProcess.processName}")
				Process.killProcess(runningProcess.pid)
				return true
			}
		}

		return super.onGodotForceQuit(godotInstanceId)
	}

	// Get the screen's density scale
	private val isLargeScreen: Boolean
		// Get the minimum window size // Correspond to the EXPANDED window size class.
		get() {
			val metrics = WindowMetricsCalculator.getOrCreate().computeMaximumWindowMetrics(this)

			// Get the screen's density scale
			val scale = resources.displayMetrics.density

			// Get the minimum window size
			val minSize = min(metrics.bounds.width(), metrics.bounds.height()).toFloat()
			val minSizeDp = minSize / scale
			return minSizeDp >= 840f // Correspond to the EXPANDED window size class.
		}

	override fun setRequestedOrientation(requestedOrientation: Int) {
		if (!overrideOrientationRequest()) {
			super.setRequestedOrientation(requestedOrientation)
		}
	}

	/**
	 * The Godot Android Editor sets its own orientation via its AndroidManifest
	 */
	protected open fun overrideOrientationRequest() = true

	/**
	 * Enable long press gestures for the Godot Android editor.
	 */
	protected open fun enableLongPressGestures() =
		java.lang.Boolean.parseBoolean(GodotLib.getEditorSetting("interface/touchscreen/enable_long_press_as_right_click"))

	/**
	 * Enable pan and scale gestures for the Godot Android editor.
	 */
	protected open fun enablePanAndScaleGestures() =
		java.lang.Boolean.parseBoolean(GodotLib.getEditorSetting("interface/touchscreen/enable_pan_and_scale_gestures"))

	/**
	 * Whether we should launch the new godot instance in an adjacent window
	 * @see https://developer.android.com/reference/android/content/Intent#FLAG_ACTIVITY_LAUNCH_ADJACENT
	 */
	private fun shouldGameLaunchAdjacent(): Boolean {
		return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
			try {
				when (Integer.parseInt(GodotLib.getEditorSetting("run/window_placement/android_window"))) {
					ANDROID_WINDOW_SAME_AS_EDITOR -> false
					ANDROID_WINDOW_SIDE_BY_SIDE_WITH_EDITOR -> true
					else -> {
						// ANDROID_WINDOW_AUTO
						isInMultiWindowMode || isLargeScreen
					}
				}
			} catch (e: NumberFormatException) {
				// Fall-back to the 'Auto' behavior
				isInMultiWindowMode || isLargeScreen
			}
		} else {
			false
		}
	}

	override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
		super.onActivityResult(requestCode, resultCode, data)
		// Check if we got the MANAGE_EXTERNAL_STORAGE permission
		if (requestCode == PermissionsUtil.REQUEST_MANAGE_EXTERNAL_STORAGE_REQ_CODE) {
			if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
				if (!Environment.isExternalStorageManager()) {
					Toast.makeText(
						this,
						R.string.denied_storage_permission_error_msg,
						Toast.LENGTH_LONG
					).show()
				}
			}
		}
	}

	override fun onRequestPermissionsResult(
		requestCode: Int,
		permissions: Array<String>,
		grantResults: IntArray
	) {
		super.onRequestPermissionsResult(requestCode, permissions, grantResults)
		// Check if we got access to the necessary storage permissions
		if (requestCode == PermissionsUtil.REQUEST_ALL_PERMISSION_REQ_CODE) {
			if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) {
				var hasReadAccess = false
				var hasWriteAccess = false
				for (i in permissions.indices) {
					if (Manifest.permission.READ_EXTERNAL_STORAGE == permissions[i] && grantResults[i] == PackageManager.PERMISSION_GRANTED) {
						hasReadAccess = true
					}
					if (Manifest.permission.WRITE_EXTERNAL_STORAGE == permissions[i] && grantResults[i] == PackageManager.PERMISSION_GRANTED) {
						hasWriteAccess = true
					}
				}
				if (!hasReadAccess || !hasWriteAccess) {
					Toast.makeText(
						this,
						R.string.denied_storage_permission_error_msg,
						Toast.LENGTH_LONG
					).show()
				}
			}
		}
	}

	override fun signApk(
		inputPath: String,
		outputPath: String,
		keystorePath: String,
		keystoreUser: String,
		keystorePassword: String
	): Error {
		val godot = godot ?: return Error.ERR_UNCONFIGURED
		return signApk(godot.fileAccessHandler, inputPath, outputPath, keystorePath, keystoreUser, keystorePassword)
	}

	override fun verifyApk(apkPath: String): Error {
		val godot = godot ?: return Error.ERR_UNCONFIGURED
		return verifyApk(godot.fileAccessHandler, apkPath)
	}
}
