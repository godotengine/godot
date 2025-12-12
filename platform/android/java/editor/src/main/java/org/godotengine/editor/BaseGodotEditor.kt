/**************************************************************************/
/*  BaseGodotEditor.kt                                                    */
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
import android.app.ActivityOptions
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Debug
import android.os.Environment
import android.os.Process
import android.preference.PreferenceManager
import android.util.Log
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.annotation.CallSuper
import androidx.core.content.edit
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.core.view.isVisible
import androidx.window.layout.WindowMetricsCalculator
import org.godotengine.editor.buildprovider.GradleBuildProvider
import org.godotengine.editor.embed.EmbeddedGodotGame
import org.godotengine.editor.embed.GameMenuFragment
import org.godotengine.editor.utils.signApk
import org.godotengine.editor.utils.verifyApk
import org.godotengine.godot.BuildProvider
import org.godotengine.godot.Godot
import org.godotengine.godot.GodotActivity
import org.godotengine.godot.GodotLib
import org.godotengine.godot.editor.utils.EditorUtils
import org.godotengine.godot.editor.utils.GameMenuUtils
import org.godotengine.godot.editor.utils.GameMenuUtils.GameEmbedMode
import org.godotengine.godot.editor.utils.GameMenuUtils.fetchGameEmbedMode
import org.godotengine.godot.error.Error
import org.godotengine.godot.utils.DialogUtils
import org.godotengine.godot.utils.PermissionsUtil
import org.godotengine.godot.utils.ProcessPhoenix
import org.godotengine.openxr.vendors.utils.*
import kotlin.math.min

/**
 * Base class for the Godot Android Editor activities.
 *
 * This provides the basic templates for the activities making up this application.
 * Each derived activity runs in its own process, which enable up to have several instances of
 * the Godot engine up and running at the same time.
 */
abstract class BaseGodotEditor : GodotActivity(), GameMenuFragment.GameMenuListener {

	companion object {
		private val TAG = BaseGodotEditor::class.java.simpleName

		private const val WAIT_FOR_DEBUGGER = false

		internal const val EXTRA_EDITOR_HINT = "editor_hint"
		internal const val EXTRA_PROJECT_MANAGER_HINT = "project_manager_hint"
		internal const val EXTRA_GAME_MENU_STATE = "game_menu_state"
		internal const val EXTRA_IS_GAME_EMBEDDED = "is_game_embedded"
		internal const val EXTRA_IS_GAME_RUNNING = "is_game_running"

		// Command line arguments.
		private const val FULLSCREEN_ARG = "--fullscreen"
		private const val FULLSCREEN_ARG_SHORT = "-f"
		private const val EDITOR_ARG = "--editor"
		private const val EDITOR_ARG_SHORT = "-e"
		private const val EDITOR_PROJECT_MANAGER_ARG = "--project-manager"
		private const val EDITOR_PROJECT_MANAGER_ARG_SHORT = "-p"
		internal const val XR_MODE_ARG = "--xr-mode"
		private const val SCENE_ARG = "--scene"
		private const val PATH_ARG = "--path"

		// Info for the various classes used by the editor.
		internal val EDITOR_MAIN_INFO = EditorWindowInfo(GodotEditor::class.java, 777, "")
		internal val RUN_GAME_INFO = EditorWindowInfo(GodotGame::class.java, 667, ":GodotGame", LaunchPolicy.AUTO)
		internal val EMBEDDED_RUN_GAME_INFO = EditorWindowInfo(EmbeddedGodotGame::class.java, 2667, ":EmbeddedGodotGame")
		internal val XR_RUN_GAME_INFO = EditorWindowInfo(GodotXRGame::class.java, 1667, ":GodotXRGame")

		/** Default behavior, means we check project settings **/
		private const val XR_MODE_DEFAULT = "default"

		/**
		 * Ignore project settings, OpenXR is disabled
		 */
		private const val XR_MODE_OFF = "off"

		/**
		 * Ignore project settings, OpenXR is enabled
		 */
		private const val XR_MODE_ON = "on"

		/**
		 * Sets of constants to specify the window to use to run the project.
		 *
		 * Should match the values in 'editor/editor_settings.cpp' for the
		 * 'run/window_placement/android_window' setting.
		 */
		private const val ANDROID_WINDOW_AUTO = 0
		private const val ANDROID_WINDOW_SAME_AS_EDITOR = 1
		private const val ANDROID_WINDOW_SIDE_BY_SIDE_WITH_EDITOR = 2

		// Game menu constants.
		internal const val KEY_GAME_MENU_ACTION = "key_game_menu_action"
		internal const val KEY_GAME_MENU_ACTION_PARAM1 = "key_game_menu_action_param1"

		internal const val GAME_MENU_ACTION_SET_SUSPEND = "setSuspend"
		internal const val GAME_MENU_ACTION_NEXT_FRAME = "nextFrame"
		internal const val GAME_MENU_ACTION_SET_NODE_TYPE = "setNodeType"
		internal const val GAME_MENU_ACTION_SET_SELECT_MODE = "setSelectMode"
		internal const val GAME_MENU_ACTION_SET_SELECTION_VISIBLE = "setSelectionVisible"
		internal const val GAME_MENU_ACTION_SET_CAMERA_OVERRIDE = "setCameraOverride"
		internal const val GAME_MENU_ACTION_SET_CAMERA_MANIPULATE_MODE = "setCameraManipulateMode"
		internal const val GAME_MENU_ACTION_RESET_CAMERA_2D_POSITION = "resetCamera2DPosition"
		internal const val GAME_MENU_ACTION_RESET_CAMERA_3D_POSITION = "resetCamera3DPosition"
		internal const val GAME_MENU_ACTION_EMBED_GAME_ON_PLAY = "embedGameOnPlay"
		internal const val GAME_MENU_ACTION_SET_DEBUG_MUTE_AUDIO = "setDebugMuteAudio"
		internal const val GAME_MENU_ACTION_RESET_TIME_SCALE = "resetTimeScale"
		internal const val GAME_MENU_ACTION_SET_TIME_SCALE = "setTimeScale"

		private const val GAME_WORKSPACE = "Game"

		internal const val SNACKBAR_SHOW_DURATION_MS = 5000L

		private const val PREF_KEY_DONT_SHOW_GAME_RESUME_HINT = "pref_key_dont_show_game_resume_hint"

		@JvmStatic
		fun isRunningInInstrumentation(): Boolean {
			if (BuildConfig.BUILD_TYPE == "release") {
				return false
			}

			return try {
				Class.forName("org.godotengine.editor.GodotEditorTest")
				true
			} catch (_: ClassNotFoundException) {
				false
			}
		}
	}

	internal val gradleBuildProvider: GradleBuildProvider = GradleBuildProvider(this, this)
	internal val editorMessageDispatcher = EditorMessageDispatcher(this)
	private val editorLoadingIndicator: View? by lazy { findViewById(R.id.editor_loading_indicator) }

	private val embeddedGameViewContainerWindow: View? by lazy { findViewById<View?>(R.id.embedded_game_view_container_window)?.apply {
		setOnClickListener {
			// Hide the game menu screen overlay.
			it.isVisible = false
		}

		// Prevent the game menu screen overlay from hiding when clicking inside of the panel bounds.
		findViewById<View?>(R.id.embedded_game_view_container)?.isClickable = true
	} }
	private val embeddedGameStateLabel: TextView? by lazy { findViewById<TextView?>(R.id.embedded_game_state_label)?.apply {
		setOnClickListener {
			godot?.runOnRenderThread {
				GameMenuUtils.playMainScene()
			}
		}
	} }
	protected val gameMenuContainer: View? by lazy {
		findViewById(R.id.game_menu_fragment_container)
	}
	protected var gameMenuFragment: GameMenuFragment? = null
	protected val gameMenuState = Bundle()

	override fun getGodotAppLayout() = R.layout.godot_editor_layout

	internal open fun getEditorWindowInfo() = EDITOR_MAIN_INFO

	/**
	 * Set of permissions to be excluded when requesting all permissions at startup.
	 *
	 * The permissions in this set will be requested on demand based on use cases.
	 */
	@CallSuper
	protected open fun getExcludedPermissions(): MutableSet<String> {
		val excludedPermissions = mutableSetOf(
			// The RECORD_AUDIO permission is requested when the "audio/driver/enable_input" project
			// setting is enabled.
			Manifest.permission.RECORD_AUDIO,
			// The CAMERA permission is requested when `CameraFeed.feed_is_active` is enabled.
			Manifest.permission.CAMERA,
			// The REQUEST_INSTALL_PACKAGES permission is requested the first time we attempt to
			// open an apk file.
			Manifest.permission.REQUEST_INSTALL_PACKAGES,
		)

		// XR runtime permissions should only be requested when the "xr/openxr/enabled" project setting
		// is enabled.
		excludedPermissions.addAll(getXRRuntimePermissions())
		return excludedPermissions
	}

	/**
	 * Set of permissions to request when the "xr/openxr/enabled" project setting is enabled.
	 */
	@CallSuper
	protected open fun getXRRuntimePermissions(): MutableSet<String> {
		return mutableSetOf()
	}

	override fun onCreate(savedInstanceState: Bundle?) {
		installSplashScreen()

		val editorWindowInfo = getEditorWindowInfo()
		if (editorWindowInfo == EDITOR_MAIN_INFO || editorWindowInfo == RUN_GAME_INFO) {
			enableEdgeToEdge()
		}

		// Skip permissions request if running in a device farm (e.g. firebase test lab) or if requested via the launch
		// intent (e.g. instrumentation tests).
		val skipPermissionsRequest = isRunningInInstrumentation() ||
			Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q && ActivityManager.isRunningInUserTestHarness()
		if (!skipPermissionsRequest) {
			// We exclude certain permissions from the set we request at startup, as they'll be
			// requested on demand based on use cases.
			PermissionsUtil.requestManifestPermissions(this, getExcludedPermissions())
		}

		editorMessageDispatcher.parseStartIntent(packageManager, intent)

		if (BuildConfig.BUILD_TYPE == "dev" && WAIT_FOR_DEBUGGER) {
			Debug.waitForDebugger()
		}

		super.onCreate(savedInstanceState)

		// Add the game menu bar.
		setupGameMenuBar()
	}

	override fun onDestroy() {
		gradleBuildProvider.buildEnvDisconnect()
		super.onDestroy()
	}

	override fun onNewIntent(newIntent: Intent) {
		if (newIntent.hasCategory(HYBRID_APP_PANEL_CATEGORY) || newIntent.hasCategory(HYBRID_APP_IMMERSIVE_CATEGORY)) {
			val params = retrieveCommandLineParamsFromLaunchIntent(newIntent)
			Log.d(TAG, "Received hybrid transition intent $newIntent with parameters ${params.contentToString()}")
			// Override EXTRA_NEW_LAUNCH so the editor is not restarted
			newIntent.putExtra(EXTRA_NEW_LAUNCH, false)

			godot?.runOnRenderThread {
				// Look for the scene, XR-mode, and hybrid data arguments.
				var scene = ""
				var xrMode = XR_MODE_DEFAULT
				var path = ""
				var base64HybridData = ""
				if (params.isNotEmpty()) {
					val sceneIndex = params.indexOf(SCENE_ARG)
					if (sceneIndex != -1 && sceneIndex + 1 < params.size) {
						scene = params[sceneIndex +1]
					}

					val xrModeIndex = params.indexOf(XR_MODE_ARG)
					if (xrModeIndex != -1 && xrModeIndex + 1 < params.size) {
						xrMode = params[xrModeIndex + 1]
					}

					val pathIndex = params.indexOf(PATH_ARG)
					if (pathIndex != -1 && pathIndex + 1 < params.size) {
						path = params[pathIndex + 1]
					}

					val hybridDataIndex = params.indexOf(HYBRID_DATA_ARG)
					if (hybridDataIndex != -1 && hybridDataIndex + 1 < params.size) {
						base64HybridData = params[hybridDataIndex + 1]
					}
				}

				val sceneArgs = mutableSetOf(XR_MODE_ARG, xrMode, HYBRID_DATA_ARG, base64HybridData).apply {
					if (path.isNotEmpty() && scene.isEmpty()) {
						add(PATH_ARG)
						add(path)
					}
				}

				Log.d(TAG, "Running scene $scene with arguments: $sceneArgs")
				EditorUtils.runScene(scene, sceneArgs.toTypedArray())
			}
		}

		super.onNewIntent(newIntent)
	}

	protected open fun shouldShowGameMenuBar() = gameMenuContainer != null

	private fun setupGameMenuBar() {
		if (shouldShowGameMenuBar()) {
			var currentFragment = supportFragmentManager.findFragmentById(R.id.game_menu_fragment_container)
			if (currentFragment !is GameMenuFragment) {
				Log.v(TAG, "Creating game menu fragment instance")
				currentFragment = GameMenuFragment().apply {
					arguments = Bundle().apply {
						putBundle(EXTRA_GAME_MENU_STATE, gameMenuState)
					}
				}
				supportFragmentManager.beginTransaction()
					.replace(R.id.game_menu_fragment_container, currentFragment, GameMenuFragment.TAG)
					.commitNowAllowingStateLoss()
			}

			gameMenuFragment = currentFragment
		}
	}

	override fun onGodotSetupCompleted() {
		super.onGodotSetupCompleted()
		val longPressEnabled = enableLongPressGestures()
		val panScaleEnabled = enablePanAndScaleGestures()
		val overrideVolumeButtonsEnabled = overrideVolumeButtons()

		runOnUiThread {
			// Enable long press, panning and scaling gestures
			godotFragment?.godot?.renderView?.inputHandler?.apply {
				enableLongPress(longPressEnabled)
				enablePanningAndScalingGestures(panScaleEnabled)
				setOverrideVolumeButtons(overrideVolumeButtonsEnabled)
			}
		}
	}

	private fun updateWindowAppearance() {
		val editorWindowInfo = getEditorWindowInfo()
		if (editorWindowInfo == EDITOR_MAIN_INFO || editorWindowInfo == RUN_GAME_INFO) {
			godot?.apply {
				enableImmersiveMode(isInImmersiveMode(), true)
				enableEdgeToEdge(isInEdgeToEdgeMode(), true)
				setSystemBarsAppearance()
			}
		}
	}

	override fun onGodotMainLoopStarted() {
		super.onGodotMainLoopStarted()
		runOnUiThread {
			// Hide the loading indicator
			editorLoadingIndicator?.visibility = View.GONE
			updateWindowAppearance()
		}
	}

	override fun onResume() {
		super.onResume()
		updateWindowAppearance()

		if (getEditorWindowInfo() == EDITOR_MAIN_INFO &&
			godot?.isEditorHint() == true &&
			(editorMessageDispatcher.hasEditorConnection(EMBEDDED_RUN_GAME_INFO) ||
				editorMessageDispatcher.hasEditorConnection(RUN_GAME_INFO))) {
			// If this is the editor window, and this is not the project manager, and we have a running game, then show
			// a hint for how to resume the playing game.
			val sharedPrefs = PreferenceManager.getDefaultSharedPreferences(applicationContext)
			if (!sharedPrefs.getBoolean(PREF_KEY_DONT_SHOW_GAME_RESUME_HINT, false)) {
				DialogUtils.showSnackbar(
					this,
					getString(R.string.show_game_resume_hint),
					SNACKBAR_SHOW_DURATION_MS,
					getString(R.string.dont_show_again_message)
				) {
					sharedPrefs.edit {
						putBoolean(PREF_KEY_DONT_SHOW_GAME_RESUME_HINT, true)
					}
				}
			}
		}
	}

	override fun getCommandLine(): MutableList<String> {
		val params = super.getCommandLine()
		if (BuildConfig.BUILD_TYPE == "dev" && !params.contains("--benchmark")) {
			params.add("--benchmark")
		}
		return params
	}

	protected fun retrieveEditorWindowInfo(args: Array<String>, gameEmbedMode: GameEmbedMode): EditorWindowInfo {
		var hasEditor = false
		var xrMode = XR_MODE_DEFAULT

		var i = 0
		while (i < args.size) {
			when (args[i++]) {
				EDITOR_ARG, EDITOR_ARG_SHORT, EDITOR_PROJECT_MANAGER_ARG, EDITOR_PROJECT_MANAGER_ARG_SHORT -> hasEditor = true
				XR_MODE_ARG -> {
					xrMode = args[i++]
				}
			}
		}

		if (hasEditor) {
			return EDITOR_MAIN_INFO
		}

		// Launching a game.
		if (isNativeXRDevice(applicationContext)) {
			if (xrMode == XR_MODE_ON) {
				return XR_RUN_GAME_INFO
			}

			if ((xrMode == XR_MODE_DEFAULT && GodotLib.getGlobal("xr/openxr/enabled").toBoolean())) {
				val hybridLaunchMode = getHybridAppLaunchMode()

				return if (hybridLaunchMode == HybridMode.PANEL) {
					RUN_GAME_INFO
				} else {
					XR_RUN_GAME_INFO
				}
			}

			// Native XR devices don't support embed mode yet.
			return RUN_GAME_INFO
		}

		// Project manager doesn't support embed mode.
		if (godot?.isProjectManagerHint() == true) {
			return RUN_GAME_INFO
		}

		// Check for embed mode launch.
		val resolvedEmbedMode = resolveGameEmbedModeIfNeeded(gameEmbedMode)
		return if (resolvedEmbedMode == GameEmbedMode.DISABLED) {
			RUN_GAME_INFO
		} else {
			EMBEDDED_RUN_GAME_INFO
		}
	}

	private fun getEditorWindowInfoForInstanceId(instanceId: Int): EditorWindowInfo? {
		return when (instanceId) {
			RUN_GAME_INFO.windowId -> RUN_GAME_INFO
			EDITOR_MAIN_INFO.windowId -> EDITOR_MAIN_INFO
			XR_RUN_GAME_INFO.windowId -> XR_RUN_GAME_INFO
			EMBEDDED_RUN_GAME_INFO.windowId -> EMBEDDED_RUN_GAME_INFO
			else -> null
		}
	}

	protected fun getNewGodotInstanceIntent(editorWindowInfo: EditorWindowInfo, args: Array<String>): Intent {
		// If we're launching an editor window (project manager or editor) and we're in
		// fullscreen mode, we want to remain in fullscreen mode.
		// This doesn't apply to the play / game window since for that window fullscreen is
		// controlled by the game logic.
		val updatedArgs = if ((editorWindowInfo == EDITOR_MAIN_INFO || editorWindowInfo == RUN_GAME_INFO) &&
			godot?.isInImmersiveMode() == true &&
			!args.contains(FULLSCREEN_ARG) &&
			!args.contains(FULLSCREEN_ARG_SHORT)
		) {
			args + FULLSCREEN_ARG
		} else {
			args
		}

		val newInstance = Intent()
			.setComponent(ComponentName(this, editorWindowInfo.windowClassName))
			.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
			.putExtra(EXTRA_COMMAND_LINE_PARAMS, updatedArgs)

		val launchPolicy = resolveLaunchPolicyIfNeeded(editorWindowInfo.launchPolicy)
		if (launchPolicy == LaunchPolicy.ADJACENT) {
			Log.v(TAG, "Adding flag for adjacent launch")
			newInstance.addFlags(Intent.FLAG_ACTIVITY_LAUNCH_ADJACENT)
		}
		return newInstance
	}

	final override fun onNewGodotInstanceRequested(args: Array<String>): Int {
		val editorWindowInfo = retrieveEditorWindowInfo(args, fetchGameEmbedMode())

		// Check if this editor window is being terminated. If it's, delay the creation of a new instance until the
		// termination is complete.
		if (editorMessageDispatcher.isPendingForceQuit(editorWindowInfo)) {
			Log.v(TAG, "Scheduling new launch after termination of ${editorWindowInfo.windowId}")
			editorMessageDispatcher.runTaskAfterForceQuit(editorWindowInfo) {
				onNewGodotInstanceRequested(args)
			}
			return editorWindowInfo.windowId
		}

		val sourceView = godotFragment?.view
		val activityOptions = if (sourceView == null) {
			null
		} else {
			val startX = sourceView.width / 2
			val startY = sourceView.height / 2
			ActivityOptions.makeScaleUpAnimation(sourceView, startX, startY, 0, 0)
		}

		val newInstance = getNewGodotInstanceIntent(editorWindowInfo, args)
		newInstance.apply {
			putExtra(EXTRA_EDITOR_HINT, godot?.isEditorHint() == true)
			putExtra(EXTRA_PROJECT_MANAGER_HINT, godot?.isProjectManagerHint() == true)
			putExtra(EXTRA_GAME_MENU_STATE, gameMenuState)
		}

		if (editorWindowInfo.windowClassName == javaClass.name) {
			Log.d(TAG, "Restarting ${editorWindowInfo.windowClassName} with parameters ${args.contentToString()}")
			triggerRebirth(activityOptions?.toBundle(), newInstance)
		} else {
			Log.d(TAG, "Starting ${editorWindowInfo.windowClassName} with parameters ${args.contentToString()}")
			newInstance.putExtra(EXTRA_NEW_LAUNCH, true)
				.putExtra(EditorMessageDispatcher.EXTRA_MSG_DISPATCHER_PAYLOAD, editorMessageDispatcher.getMessageDispatcherPayload())
			startActivity(newInstance, activityOptions?.toBundle())
		}
		return editorWindowInfo.windowId
	}

	override fun onGodotForceQuit(instance: Godot) {
		if (!isRunningInInstrumentation()) {
			// For instrumented tests, we disable force-quitting to allow the tests to complete successfully, otherwise
			// they fail when the process crashes.
			super.onGodotForceQuit(instance)
		}
	}

	final override fun onGodotForceQuit(godotInstanceId: Int): Boolean {
		val editorWindowInfo = getEditorWindowInfoForInstanceId(godotInstanceId) ?: return super.onGodotForceQuit(godotInstanceId)

		if (editorWindowInfo.windowClassName == javaClass.name) {
			Log.d(TAG, "Force quitting ${editorWindowInfo.windowClassName}")
			ProcessPhoenix.forceQuit(this)
			return true
		}

		// Send an inter-process message to request the target editor window to force quit.
		if (editorMessageDispatcher.requestForceQuit(editorWindowInfo)) {
			return true
		}

		// Fallback to killing the target process.
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

	protected open fun overrideVolumeButtons() = false

	/**
	 * Enable long press gestures for the Godot Android editor.
	 */
	protected open fun enableLongPressGestures() =
		java.lang.Boolean.parseBoolean(GodotLib.getEditorSetting("interface/touchscreen/enable_long_press_as_right_click"))

	/**
	 * Disable scroll deadzone for the Godot Android editor.
	 */
	protected open fun disableScrollDeadzone() = true

	/**
	 * Enable pan and scale gestures for the Godot Android editor.
	 */
	protected open fun enablePanAndScaleGestures() =
		java.lang.Boolean.parseBoolean(GodotLib.getEditorSetting("interface/touchscreen/enable_pan_and_scale_gestures"))

	private fun resolveGameEmbedModeIfNeeded(embedMode: GameEmbedMode): GameEmbedMode {
		return when (embedMode) {
			GameEmbedMode.AUTO -> {
				if (isInMultiWindowMode || isLargeScreen || isNativeXRDevice(applicationContext)) {
					GameEmbedMode.DISABLED
				} else {
					GameEmbedMode.ENABLED
				}
			}

			else -> embedMode
		}
	}

	/**
	 * If the launch policy is [LaunchPolicy.AUTO], resolve it into a specific policy based on the
	 * editor setting or device and screen metrics.
	 */
	private fun resolveLaunchPolicyIfNeeded(policy: LaunchPolicy): LaunchPolicy {
		return when (policy) {
			LaunchPolicy.AUTO -> {
				val defaultLaunchPolicy = if (isInMultiWindowMode || isLargeScreen || isNativeXRDevice(applicationContext)) {
					LaunchPolicy.ADJACENT
				} else {
					LaunchPolicy.SAME
				}

				try {
					when (Integer.parseInt(GodotLib.getEditorSetting("run/window_placement/android_window"))) {
						ANDROID_WINDOW_SAME_AS_EDITOR -> LaunchPolicy.SAME
						ANDROID_WINDOW_SIDE_BY_SIDE_WITH_EDITOR -> LaunchPolicy.ADJACENT

						else -> {
							// ANDROID_WINDOW_AUTO
							defaultLaunchPolicy
						}
					}
				} catch (e: NumberFormatException) {
					Log.w(TAG, "Error parsing the Android window placement editor setting", e)
					// Fall-back to the default launch policy.
					defaultLaunchPolicy
				}
			}

			else -> {
				policy
			}
		}
	}

	override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
		super.onActivityResult(requestCode, resultCode, data)
		// Check if we got the MANAGE_EXTERNAL_STORAGE permission
		when (requestCode) {
			PermissionsUtil.REQUEST_MANAGE_EXTERNAL_STORAGE_REQ_CODE -> {
				if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && !Environment.isExternalStorageManager()) {
					Toast.makeText(
						this,
						R.string.denied_storage_permission_error_msg,
						Toast.LENGTH_LONG
					).show()
				}
			}

			PermissionsUtil.REQUEST_INSTALL_PACKAGES_REQ_CODE -> {
				if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O && !packageManager.canRequestPackageInstalls()) {
					Toast.makeText(
						this,
						R.string.denied_install_packages_permission_error_msg,
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

	@CallSuper
	override fun supportsFeature(featureTag: String): Boolean {
		if (featureTag == "xr_editor") {
			return isNativeXRDevice(applicationContext)
		}

		if (featureTag == BuildConfig.FLAVOR) {
			return true
		}

        return super.supportsFeature(featureTag)
    }

	internal fun onEditorConnected(editorId: Int) {
		Log.d(TAG, "Editor $editorId connected!")
		when (editorId) {
			EMBEDDED_RUN_GAME_INFO.windowId, RUN_GAME_INFO.windowId -> {
				runOnUiThread {
					embeddedGameViewContainerWindow?.isVisible = false
				}
			}

			XR_RUN_GAME_INFO.windowId -> {
				runOnUiThread {
					updateEmbeddedGameView(gameRunning = true, gameEmbedded = false)
				}
			}
		}
	}

	internal fun onEditorDisconnected(editorId: Int) {
		Log.d(TAG, "Editor $editorId disconnected!")
	}

	private fun updateEmbeddedGameView(gameRunning: Boolean, gameEmbedded: Boolean) {
		if (gameRunning) {
			embeddedGameStateLabel?.apply {
				setText(R.string.running_game_not_embedded_message)
				setCompoundDrawablesRelativeWithIntrinsicBounds(0, 0, 0, 0)
				isClickable = false
			}
		} else {
			embeddedGameStateLabel?.apply{
				setText(R.string.embedded_game_not_running_message)
				setCompoundDrawablesRelativeWithIntrinsicBounds(0, 0, 0, R.drawable.play_48dp)
				isClickable = true
			}
		}

		gameMenuState.putBoolean(EXTRA_IS_GAME_EMBEDDED, gameEmbedded)
		gameMenuState.putBoolean(EXTRA_IS_GAME_RUNNING, gameRunning)
		gameMenuFragment?.refreshGameMenu(gameMenuState)
	}

	override fun onEditorWorkspaceSelected(workspace: String) {
		if (workspace == GAME_WORKSPACE && shouldShowGameMenuBar()) {
			if (editorMessageDispatcher.bringEditorWindowToFront(EMBEDDED_RUN_GAME_INFO) || editorMessageDispatcher.bringEditorWindowToFront(RUN_GAME_INFO)) {
				return
			}

			val xrGameRunning = editorMessageDispatcher.hasEditorConnection(XR_RUN_GAME_INFO)
			val gameEmbedMode = resolveGameEmbedModeIfNeeded(fetchGameEmbedMode())
			runOnUiThread {
				updateEmbeddedGameView(xrGameRunning, gameEmbedMode != GameEmbedMode.DISABLED)
				embeddedGameViewContainerWindow?.isVisible = true
			}
		}
	}

	internal open fun bringSelfToFront() {
		runOnUiThread {
			Log.v(TAG, "Bringing self to front")
			val relaunchIntent = Intent(intent)
			// Don't restart.
			relaunchIntent.putExtra(EXTRA_NEW_LAUNCH, false)
			startActivity(relaunchIntent)
		}
	}

	internal fun parseGameMenuAction(actionData: Bundle) {
		val action = actionData.getString(KEY_GAME_MENU_ACTION) ?: return
		when (action) {
			GAME_MENU_ACTION_SET_SUSPEND -> {
				val suspended = actionData.getBoolean(KEY_GAME_MENU_ACTION_PARAM1)
				suspendGame(suspended)
			}
			GAME_MENU_ACTION_NEXT_FRAME -> {
				dispatchNextFrame()
			}
			GAME_MENU_ACTION_SET_NODE_TYPE -> {
				val nodeType = actionData.getSerializable(KEY_GAME_MENU_ACTION_PARAM1) as GameMenuFragment.GameMenuListener.NodeType?
				if (nodeType != null) {
					selectRuntimeNode(nodeType)
				}
			}
			GAME_MENU_ACTION_SET_SELECTION_VISIBLE -> {
				val enabled = actionData.getBoolean(KEY_GAME_MENU_ACTION_PARAM1)
				toggleSelectionVisibility(enabled)
			}
			GAME_MENU_ACTION_SET_CAMERA_OVERRIDE -> {
				val enabled = actionData.getBoolean(KEY_GAME_MENU_ACTION_PARAM1)
				overrideCamera(enabled)
			}
			GAME_MENU_ACTION_SET_SELECT_MODE -> {
				val selectMode = actionData.getSerializable(KEY_GAME_MENU_ACTION_PARAM1) as GameMenuFragment.GameMenuListener.SelectMode?
				if (selectMode != null) {
					selectRuntimeNodeSelectMode(selectMode)
				}
			}
			GAME_MENU_ACTION_RESET_CAMERA_2D_POSITION -> {
				reset2DCamera()
			}
			GAME_MENU_ACTION_RESET_CAMERA_3D_POSITION -> {
				reset3DCamera()
			}
			GAME_MENU_ACTION_SET_CAMERA_MANIPULATE_MODE -> {
				val mode = actionData.getSerializable(KEY_GAME_MENU_ACTION_PARAM1) as? GameMenuFragment.GameMenuListener.CameraMode?
				if (mode != null) {
					manipulateCamera(mode)
				}
			}
			GAME_MENU_ACTION_EMBED_GAME_ON_PLAY -> {
				val embedded = actionData.getBoolean(KEY_GAME_MENU_ACTION_PARAM1)
				embedGameOnPlay(embedded)
			}
			GAME_MENU_ACTION_SET_DEBUG_MUTE_AUDIO -> {
				val enabled = actionData.getBoolean(KEY_GAME_MENU_ACTION_PARAM1)
				muteAudio(enabled)
			}
			GAME_MENU_ACTION_RESET_TIME_SCALE -> {
				resetTimeScale()
			}
			GAME_MENU_ACTION_SET_TIME_SCALE -> {
				val scale = actionData.getDouble(KEY_GAME_MENU_ACTION_PARAM1)
				setTimeScale(scale)
			}
		}
	}

	override fun suspendGame(suspended: Boolean) {
		gameMenuState.putBoolean(GAME_MENU_ACTION_SET_SUSPEND, suspended)
		godot?.runOnRenderThread {
			GameMenuUtils.setSuspend(suspended)
		}
	}

	override fun dispatchNextFrame() {
		godot?.runOnRenderThread {
			GameMenuUtils.nextFrame()
		}
	}

	override fun toggleSelectionVisibility(enabled: Boolean) {
		gameMenuState.putBoolean(GAME_MENU_ACTION_SET_SELECTION_VISIBLE, enabled)
		godot?.runOnRenderThread {
			GameMenuUtils.setSelectionVisible(enabled)
		}
	}

	override fun overrideCamera(enabled: Boolean) {
		gameMenuState.putBoolean(GAME_MENU_ACTION_SET_CAMERA_OVERRIDE, enabled)
		godot?.runOnRenderThread {
			GameMenuUtils.setCameraOverride(enabled)
		}
	}

	override fun selectRuntimeNode(nodeType: GameMenuFragment.GameMenuListener.NodeType) {
		gameMenuState.putSerializable(GAME_MENU_ACTION_SET_NODE_TYPE, nodeType)
		godot?.runOnRenderThread {
			GameMenuUtils.setNodeType(nodeType.ordinal)
		}
	}

	override fun selectRuntimeNodeSelectMode(selectMode: GameMenuFragment.GameMenuListener.SelectMode) {
		gameMenuState.putSerializable(GAME_MENU_ACTION_SET_SELECT_MODE, selectMode)
		godot?.runOnRenderThread {
			GameMenuUtils.setSelectMode(selectMode.ordinal)
		}
	}

	override fun reset2DCamera() {
		godot?.runOnRenderThread {
			GameMenuUtils.resetCamera2DPosition()
		}
	}

	override fun reset3DCamera() {
		godot?.runOnRenderThread {
			GameMenuUtils.resetCamera3DPosition()
		}
	}

	override fun manipulateCamera(mode: GameMenuFragment.GameMenuListener.CameraMode) {
		gameMenuState.putSerializable(GAME_MENU_ACTION_SET_CAMERA_MANIPULATE_MODE, mode)
		godot?.runOnRenderThread {
			GameMenuUtils.setCameraManipulateMode(mode.ordinal)
		}
	}

	override fun muteAudio(enabled: Boolean) {
		gameMenuState.putBoolean(GAME_MENU_ACTION_SET_DEBUG_MUTE_AUDIO, enabled)
		godot?.runOnRenderThread {
			GameMenuUtils.setDebugMuteAudio(enabled)
		}
	}

	override fun resetTimeScale() {
		gameMenuState.putDouble(GAME_MENU_ACTION_SET_TIME_SCALE, 1.0)
		godot?.runOnRenderThread {
			GameMenuUtils.resetTimeScale()
		}
	}

	override fun setTimeScale(scale: Double) {
		gameMenuState.putDouble(GAME_MENU_ACTION_SET_TIME_SCALE, scale)
		godot?.runOnRenderThread {
			GameMenuUtils.setTimeScale(scale)
		}
	}

	override fun embedGameOnPlay(embedded: Boolean) {
		gameMenuState.putBoolean(GAME_MENU_ACTION_EMBED_GAME_ON_PLAY, embedded)
		godot?.runOnRenderThread {
			val gameEmbedMode = if (embedded) GameEmbedMode.ENABLED else GameEmbedMode.DISABLED
			GameMenuUtils.saveGameEmbedMode(gameEmbedMode)
		}
	}

	override fun isGameEmbeddingSupported() = !isNativeXRDevice(applicationContext)

	override fun getBuildProvider(): BuildProvider? {
		return gradleBuildProvider
	}
}
