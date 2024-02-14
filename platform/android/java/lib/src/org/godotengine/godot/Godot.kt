/**************************************************************************/
/*  Godot.kt                                                              */
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

package org.godotengine.godot

import android.annotation.SuppressLint
import android.app.Activity
import android.app.AlertDialog
import android.content.*
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.content.res.Resources
import android.graphics.Color
import android.graphics.Rect
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.*
import android.util.Log
import android.view.*
import android.view.ViewTreeObserver.OnGlobalLayoutListener
import android.widget.FrameLayout
import androidx.annotation.Keep
import androidx.annotation.StringRes
import com.google.android.vending.expansion.downloader.*
import org.godotengine.godot.input.GodotEditText
import org.godotengine.godot.io.directory.DirectoryAccessHandler
import org.godotengine.godot.io.file.FileAccessHandler
import org.godotengine.godot.plugin.GodotPluginRegistry
import org.godotengine.godot.tts.GodotTTS
import org.godotengine.godot.utils.GodotNetUtils
import org.godotengine.godot.utils.PermissionsUtil
import org.godotengine.godot.utils.PermissionsUtil.requestPermission
import org.godotengine.godot.utils.beginBenchmarkMeasure
import org.godotengine.godot.utils.benchmarkFile
import org.godotengine.godot.utils.dumpBenchmark
import org.godotengine.godot.utils.endBenchmarkMeasure
import org.godotengine.godot.utils.useBenchmark
import org.godotengine.godot.xr.XRMode
import java.io.File
import java.io.FileInputStream
import java.io.InputStream
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.*

/**
 * Core component used to interface with the native layer of the engine.
 *
 * Can be hosted by [Activity], [Fragment] or [Service] android components, so long as its
 * lifecycle methods are properly invoked.
 */
class Godot(private val context: Context) : SensorEventListener {

	private companion object {
		private val TAG = Godot::class.java.simpleName
	}

	private val pluginRegistry: GodotPluginRegistry by lazy {
		GodotPluginRegistry.getPluginRegistry()
	}
	private val mSensorManager: SensorManager by lazy {
		requireActivity().getSystemService(Context.SENSOR_SERVICE) as SensorManager
	}
	private val mAccelerometer: Sensor? by lazy {
		mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
	}
	private val mGravity: Sensor? by lazy {
		mSensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)
	}
	private val mMagnetometer: Sensor? by lazy {
		mSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
	}
	private val mGyroscope: Sensor? by lazy {
		mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
	}
	private val mClipboard: ClipboardManager by lazy {
		requireActivity().getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
	}

	private val uiChangeListener = View.OnSystemUiVisibilityChangeListener { visibility: Int ->
		if (visibility and View.SYSTEM_UI_FLAG_FULLSCREEN == 0) {
			val decorView = requireActivity().window.decorView
			decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_LAYOUT_STABLE or
					View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION or
					View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN or
					View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
					View.SYSTEM_UI_FLAG_FULLSCREEN or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
		}}

	val tts = GodotTTS(context)
	val directoryAccessHandler = DirectoryAccessHandler(context)
	val fileAccessHandler = FileAccessHandler(context)
	val netUtils = GodotNetUtils(context)

	/**
	 * Tracks whether [onCreate] was completed successfully.
	 */
	private var initializationStarted = false

	/**
	 * Tracks whether [GodotLib.initialize] was completed successfully.
	 */
	private var nativeLayerInitializeCompleted = false

	/**
	 * Tracks whether [GodotLib.setup] was completed successfully.
	 */
	private var nativeLayerSetupCompleted = false

	/**
	 * Tracks whether [onInitRenderView] was completed successfully.
	 */
	private var renderViewInitialized = false
	private var primaryHost: GodotHost? = null

	var io: GodotIO? = null

	private var commandLine : MutableList<String> = ArrayList<String>()
	private var xrMode = XRMode.REGULAR
	private var expansionPackPath: String = ""
	private var useApkExpansion = false
	private var useImmersive = false
	private var useDebugOpengl = false

	private var containerLayout: FrameLayout? = null
	var renderView: GodotRenderView? = null

	/**
	 * Returns true if the native engine has been initialized through [onInitNativeLayer], false otherwise.
	 */
	private fun isNativeInitialized() = nativeLayerInitializeCompleted && nativeLayerSetupCompleted

	/**
	 * Returns true if the engine has been initialized, false otherwise.
	 */
	fun isInitialized() = initializationStarted && isNativeInitialized() && renderViewInitialized

	/**
	 * Provides access to the primary host [Activity]
	 */
	fun getActivity() = primaryHost?.activity
	private fun requireActivity() = getActivity() ?: throw IllegalStateException("Host activity must be non-null")

	/**
	 * Start initialization of the Godot engine.
	 *
	 * This must be followed by [onInitNativeLayer] and [onInitRenderView] in that order to complete
	 * initialization of the engine.
	 *
	 * @throws IllegalArgumentException exception if the specified expansion pack (if any)
	 * is invalid.
	 */
	fun onCreate(primaryHost: GodotHost) {
		if (this.primaryHost != null || initializationStarted) {
			Log.d(TAG, "OnCreate already invoked")
			return
		}

		beginBenchmarkMeasure("Godot::onCreate")
		try {
			this.primaryHost = primaryHost
			val activity = requireActivity()
			val window = activity.window
			window.addFlags(WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON)
			GodotPluginRegistry.initializePluginRegistry(this, primaryHost.getHostPlugins(this))
			if (io == null) {
				io = GodotIO(activity)
			}

			// check for apk expansion API
			commandLine = getCommandLine()
			var mainPackMd5: String? = null
			var mainPackKey: String? = null
			val newArgs: MutableList<String> = ArrayList()
			var i = 0
			while (i < commandLine.size) {
				val hasExtra: Boolean = i < commandLine.size - 1
				if (commandLine[i] == XRMode.REGULAR.cmdLineArg) {
					xrMode = XRMode.REGULAR
				} else if (commandLine[i] == XRMode.OPENXR.cmdLineArg) {
					xrMode = XRMode.OPENXR
				} else if (commandLine[i] == "--debug_opengl") {
					useDebugOpengl = true
				} else if (commandLine[i] == "--use_immersive") {
					useImmersive = true
					window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_LAYOUT_STABLE or
							View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION or
							View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN or
							View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or  // hide nav bar
							View.SYSTEM_UI_FLAG_FULLSCREEN or  // hide status bar
							View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
					registerUiChangeListener()
				} else if (commandLine[i] == "--use_apk_expansion") {
					useApkExpansion = true
				} else if (hasExtra && commandLine[i] == "--apk_expansion_md5") {
					mainPackMd5 = commandLine[i + 1]
					i++
				} else if (hasExtra && commandLine[i] == "--apk_expansion_key") {
					mainPackKey = commandLine[i + 1]
					val prefs = activity.getSharedPreferences(
							"app_data_keys",
							Context.MODE_PRIVATE
					)
					val editor = prefs.edit()
					editor.putString("store_public_key", mainPackKey)
					editor.apply()
					i++
				} else if (commandLine[i] == "--benchmark") {
					useBenchmark = true
					newArgs.add(commandLine[i])
				} else if (hasExtra && commandLine[i] == "--benchmark-file") {
					useBenchmark = true
					newArgs.add(commandLine[i])

					// Retrieve the filepath
					benchmarkFile = commandLine[i + 1]
					newArgs.add(commandLine[i + 1])

					i++
				} else if (commandLine[i].trim().isNotEmpty()) {
					newArgs.add(commandLine[i])
				}
				i++
			}
			commandLine = if (newArgs.isEmpty()) { mutableListOf() } else { newArgs }
			if (useApkExpansion && mainPackMd5 != null && mainPackKey != null) {
				// Build the full path to the app's expansion files
				try {
					expansionPackPath = Helpers.getSaveFilePath(context)
					expansionPackPath += "/main." + activity.packageManager.getPackageInfo(
							activity.packageName,
							0
					).versionCode + "." + activity.packageName + ".obb"
				} catch (e: java.lang.Exception) {
					Log.e(TAG, "Unable to build full path to the app's expansion files", e)
				}
				val f = File(expansionPackPath)
				var packValid = true
				if (!f.exists()) {
					packValid = false
				} else if (obbIsCorrupted(expansionPackPath, mainPackMd5)) {
					packValid = false
					try {
						f.delete()
					} catch (_: java.lang.Exception) {
					}
				}
				if (!packValid) {
					// Aborting engine initialization
					throw IllegalArgumentException("Invalid expansion pack")
				}
			}

			initializationStarted = true
		} catch (e: java.lang.Exception) {
			// Clear the primary host and rethrow
			this.primaryHost = null
			initializationStarted = false
			throw e
		} finally {
			endBenchmarkMeasure("Godot::onCreate");
		}
	}

	/**
	 * Initializes the native layer of the Godot engine.
	 *
	 * This must be preceded by [onCreate] and followed by [onInitRenderView] to complete
	 * initialization of the engine.
	 *
	 * @return false if initialization of the native layer fails, true otherwise.
	 *
	 * @throws IllegalStateException if [onCreate] has not been called.
	 */
	fun onInitNativeLayer(host: GodotHost): Boolean {
		if (!initializationStarted) {
			throw IllegalStateException("OnCreate must be invoked successfully prior to initializing the native layer")
		}
		if (isNativeInitialized()) {
			Log.d(TAG, "OnInitNativeLayer already invoked")
			return true
		}
		if (host != primaryHost) {
			Log.e(TAG, "Native initialization is only supported for the primary host")
			return false
		}

		if (expansionPackPath.isNotEmpty()) {
			commandLine.add("--main-pack")
			commandLine.add(expansionPackPath)
		}
		val activity = requireActivity()
		if (!nativeLayerInitializeCompleted) {
			nativeLayerInitializeCompleted = GodotLib.initialize(
					activity,
					this,
					activity.assets,
					io,
					netUtils,
					directoryAccessHandler,
					fileAccessHandler,
					useApkExpansion,
			)
		}

		if (nativeLayerInitializeCompleted && !nativeLayerSetupCompleted) {
			nativeLayerSetupCompleted = GodotLib.setup(commandLine.toTypedArray(), tts)
			if (!nativeLayerSetupCompleted) {
				Log.e(TAG, "Unable to setup the Godot engine! Aborting...")
				alert(R.string.error_engine_setup_message, R.string.text_error_title, this::forceQuit)
			}
		}
		return isNativeInitialized()
	}

	/**
	 * Used to complete initialization of the view used by the engine for rendering.
	 *
	 * This must be preceded by [onCreate] and [onInitNativeLayer] in that order to properly
	 * initialize the engine.
	 *
	 * @param host The [GodotHost] that's initializing the render views
	 * @param providedContainerLayout Optional argument; if provided, this is reused to host the Godot's render views
	 *
	 * @return A [FrameLayout] instance containing Godot's render views if initialization is successful, null otherwise.
	 *
	 * @throws IllegalStateException if [onInitNativeLayer] has not been called
	 */
	@JvmOverloads
	fun onInitRenderView(host: GodotHost, providedContainerLayout: FrameLayout = FrameLayout(host.activity)): FrameLayout? {
		if (!isNativeInitialized()) {
			throw IllegalStateException("onInitNativeLayer() must be invoked successfully prior to initializing the render view")
		}

		try {
			val activity: Activity = host.activity
			containerLayout = providedContainerLayout
			containerLayout?.removeAllViews()
			containerLayout?.layoutParams = ViewGroup.LayoutParams(
					ViewGroup.LayoutParams.MATCH_PARENT,
					ViewGroup.LayoutParams.MATCH_PARENT
			)

			// GodotEditText layout
			val editText = GodotEditText(activity)
			editText.layoutParams =
					ViewGroup.LayoutParams(
							ViewGroup.LayoutParams.MATCH_PARENT,
							activity.resources.getDimension(R.dimen.text_edit_height).toInt()
					)
			// Prevent GodotEditText from showing on splash screen on devices with Android 14 or newer.
			editText.setBackgroundColor(Color.TRANSPARENT)
			// ...add to FrameLayout
			containerLayout?.addView(editText)
			renderView = if (usesVulkan()) {
				if (!meetsVulkanRequirements(activity.packageManager)) {
					alert(R.string.error_missing_vulkan_requirements_message, R.string.text_error_title, this::forceQuit)
					return null
				}
				GodotVulkanRenderView(host, this)
			} else {
				// Fallback to openGl
				GodotGLRenderView(host, this, xrMode, useDebugOpengl)
			}

			if (host == primaryHost) {
				renderView!!.startRenderer()
			}
			val view: View = renderView!!.view
			containerLayout?.addView(
					view,
					ViewGroup.LayoutParams(
							ViewGroup.LayoutParams.MATCH_PARENT,
							ViewGroup.LayoutParams.MATCH_PARENT
					)
			)
			editText.setView(renderView)
			io?.setEdit(editText)

			// Listeners for keyboard height.
			if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
				// Report the height of virtual keyboard as it changes during the animation.
				val decorView = activity.window.decorView
				decorView.setWindowInsetsAnimationCallback(object : WindowInsetsAnimation.Callback(DISPATCH_MODE_STOP) {
					var startBottom = 0
					var endBottom = 0
					override fun onPrepare(animation: WindowInsetsAnimation) {
						startBottom = decorView.rootWindowInsets.getInsets(WindowInsets.Type.ime()).bottom
					}

					override fun onStart(animation: WindowInsetsAnimation, bounds: WindowInsetsAnimation.Bounds): WindowInsetsAnimation.Bounds {
						endBottom = decorView.rootWindowInsets.getInsets(WindowInsets.Type.ime()).bottom
						return bounds
					}

					override fun onProgress(windowInsets: WindowInsets, list: List<WindowInsetsAnimation>): WindowInsets {
						// Find the IME animation.
						var imeAnimation: WindowInsetsAnimation? = null
						for (animation in list) {
							if (animation.typeMask and WindowInsets.Type.ime() != 0) {
								imeAnimation = animation
								break
							}
						}
						// Update keyboard height based on IME animation.
						if (imeAnimation != null) {
							val interpolatedFraction = imeAnimation.interpolatedFraction
							// Linear interpolation between start and end values.
							val keyboardHeight = startBottom * (1.0f - interpolatedFraction) + endBottom * interpolatedFraction
							GodotLib.setVirtualKeyboardHeight(keyboardHeight.toInt())
						}
						return windowInsets
					}

					override fun onEnd(animation: WindowInsetsAnimation) {}
				})
			} else {
				// Infer the virtual keyboard height using visible area.
				view.viewTreeObserver.addOnGlobalLayoutListener(object : OnGlobalLayoutListener {
					// Don't allocate a new Rect every time the callback is called.
					val visibleSize = Rect()
					override fun onGlobalLayout() {
						val surfaceView = renderView!!.view
						surfaceView.getWindowVisibleDisplayFrame(visibleSize)
						val keyboardHeight = surfaceView.height - visibleSize.bottom
						GodotLib.setVirtualKeyboardHeight(keyboardHeight)
					}
				})
			}

			if (host == primaryHost) {
				renderView!!.queueOnRenderThread {
					for (plugin in pluginRegistry.allPlugins) {
						plugin.onRegisterPluginWithGodotNative()
					}
					setKeepScreenOn(java.lang.Boolean.parseBoolean(GodotLib.getGlobal("display/window/energy_saving/keep_screen_on")))
				}

				// Include the returned non-null views in the Godot view hierarchy.
				for (plugin in pluginRegistry.allPlugins) {
					val pluginView = plugin.onMainCreate(activity)
					if (pluginView != null) {
						if (plugin.shouldBeOnTop()) {
							containerLayout?.addView(pluginView)
						} else {
							containerLayout?.addView(pluginView, 0)
						}
					}
				}
			}
			renderViewInitialized = true
		} finally {
			if (!renderViewInitialized) {
				containerLayout?.removeAllViews()
				containerLayout = null
			}
		}
		return containerLayout
	}

	fun onResume(host: GodotHost) {
		if (host != primaryHost) {
			return
		}

		renderView!!.onActivityResumed()
		if (mAccelerometer != null) {
			mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_GAME)
		}
		if (mGravity != null) {
			mSensorManager.registerListener(this, mGravity, SensorManager.SENSOR_DELAY_GAME)
		}
		if (mMagnetometer != null) {
			mSensorManager.registerListener(this, mMagnetometer, SensorManager.SENSOR_DELAY_GAME)
		}
		if (mGyroscope != null) {
			mSensorManager.registerListener(this, mGyroscope, SensorManager.SENSOR_DELAY_GAME)
		}
		if (useImmersive) {
			val window = requireActivity().window
			window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_LAYOUT_STABLE or
					View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION or
					View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN or
					View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or  // hide nav bar
					View.SYSTEM_UI_FLAG_FULLSCREEN or  // hide status bar
					View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
		}
		for (plugin in pluginRegistry.allPlugins) {
			plugin.onMainResume()
		}
	}

	fun onPause(host: GodotHost) {
		if (host != primaryHost) {
			return
		}

		renderView!!.onActivityPaused()
		mSensorManager.unregisterListener(this)
		for (plugin in pluginRegistry.allPlugins) {
			plugin.onMainPause()
		}
	}

	fun onDestroy(primaryHost: GodotHost) {
		if (this.primaryHost != primaryHost) {
			return
		}

		for (plugin in pluginRegistry.allPlugins) {
			plugin.onMainDestroy()
		}

		runOnRenderThread {
			GodotLib.ondestroy()
			forceQuit()
		}
	}

	/**
	 * Activity result callback
	 */
	fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
		for (plugin in pluginRegistry.allPlugins) {
			plugin.onMainActivityResult(requestCode, resultCode, data)
		}
	}

	/**
	 * Permissions request callback
	 */
	fun onRequestPermissionsResult(
		requestCode: Int,
		permissions: Array<String?>,
		grantResults: IntArray
	) {
		for (plugin in pluginRegistry.allPlugins) {
			plugin.onMainRequestPermissionsResult(requestCode, permissions, grantResults)
		}
		for (i in permissions.indices) {
			GodotLib.requestPermissionResult(
				permissions[i],
				grantResults[i] == PackageManager.PERMISSION_GRANTED
			)
		}
	}

	/**
	 * Invoked on the render thread when the Godot setup is complete.
	 */
	private fun onGodotSetupCompleted() {
		Log.d(TAG, "OnGodotSetupCompleted")

		// These properties are defined after Godot setup completion, so we retrieve them here.
		val longPressEnabled = java.lang.Boolean.parseBoolean(GodotLib.getGlobal("input_devices/pointing/android/enable_long_press_as_right_click"))
		val panScaleEnabled = java.lang.Boolean.parseBoolean(GodotLib.getGlobal("input_devices/pointing/android/enable_pan_and_scale_gestures"))

		runOnUiThread {
			renderView?.inputHandler?.apply {
				enableLongPress(longPressEnabled)
				enablePanningAndScalingGestures(panScaleEnabled)
			}
		}

		for (plugin in pluginRegistry.allPlugins) {
			plugin.onGodotSetupCompleted()
		}
		primaryHost?.onGodotSetupCompleted()
	}

	/**
	 * Invoked on the render thread when the Godot main loop has started.
	 */
	private fun onGodotMainLoopStarted() {
		Log.d(TAG, "OnGodotMainLoopStarted")

		for (plugin in pluginRegistry.allPlugins) {
			plugin.onGodotMainLoopStarted()
		}
		primaryHost?.onGodotMainLoopStarted()
	}

	private fun restart() {
		primaryHost?.onGodotRestartRequested(this)
	}

	private fun registerUiChangeListener() {
		val decorView = requireActivity().window.decorView
		decorView.setOnSystemUiVisibilityChangeListener(uiChangeListener)
	}

	@Keep
	private fun alert(message: String, title: String) {
		alert(message, title, null)
	}

	private fun alert(
		@StringRes messageResId: Int,
		@StringRes titleResId: Int,
		okCallback: Runnable?
	) {
		val res: Resources = getActivity()?.resources ?: return
		alert(res.getString(messageResId), res.getString(titleResId), okCallback)
	}

	private fun alert(message: String, title: String, okCallback: Runnable?) {
		val activity: Activity = getActivity() ?: return
		runOnUiThread {
			val builder = AlertDialog.Builder(activity)
			builder.setMessage(message).setTitle(title)
			builder.setPositiveButton(
				"OK"
			) { dialog: DialogInterface, id: Int ->
				okCallback?.run()
				dialog.cancel()
			}
			val dialog = builder.create()
			dialog.show()
		}
	}

	/**
	 * Queue a runnable to be run on the render thread.
	 *
	 * This must be called after the render thread has started.
	 */
	fun runOnRenderThread(action: Runnable) {
		if (renderView != null) {
			renderView!!.queueOnRenderThread(action)
		}
	}

	/**
	 * Runs the specified action on the UI thread.
	 * If the current thread is the UI thread, then the action is executed immediately.
	 * If the current thread is not the UI thread, the action is posted to the event queue
	 * of the UI thread.
	 */
	fun runOnUiThread(action: Runnable) {
		val activity: Activity = getActivity() ?: return
		activity.runOnUiThread(action)
	}

	/**
	 * Returns true if the call is being made on the Ui thread.
	 */
	private fun isOnUiThread() = Looper.myLooper() == Looper.getMainLooper()

	/**
	 * Returns true if `Vulkan` is used for rendering.
	 */
	private fun usesVulkan(): Boolean {
		val renderer = GodotLib.getGlobal("rendering/renderer/rendering_method")
		val renderingDevice = GodotLib.getGlobal("rendering/rendering_device/driver")
		return ("forward_plus" == renderer || "mobile" == renderer) && "vulkan" == renderingDevice
	}

	/**
	 * Returns true if the device meets the base requirements for Vulkan support, false otherwise.
	 */
	private fun meetsVulkanRequirements(packageManager: PackageManager?): Boolean {
		if (packageManager == null) {
			return false
		}
		if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
			if (!packageManager.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_LEVEL, 1)) {
				// Optional requirements.. log as warning if missing
				Log.w(TAG, "The vulkan hardware level does not meet the minimum requirement: 1")
			}

			// Check for api version 1.0
			return packageManager.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_VERSION, 0x400003)
		}
		return false
	}

	private fun setKeepScreenOn(enabled: Boolean) {
		runOnUiThread {
			if (enabled) {
				getActivity()?.window?.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
			} else {
				getActivity()?.window?.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
			}
		}
	}

	/**
	 * Returns true if dark mode is supported, false otherwise.
	 */
	@Keep
	private fun isDarkModeSupported(): Boolean {
		return Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q
	}

	/**
	 * Returns true if dark mode is supported and enabled, false otherwise.
	 */
	@Keep
	private fun isDarkMode(): Boolean {
		if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
			return context.resources?.configuration?.uiMode?.and(Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES
		}
		return false
	}

	fun hasClipboard(): Boolean {
		return mClipboard.hasPrimaryClip()
	}

	fun getClipboard(): String? {
		val clipData = mClipboard.primaryClip ?: return ""
		val text = clipData.getItemAt(0).text ?: return ""
		return text.toString()
	}

	fun setClipboard(text: String?) {
		val clip = ClipData.newPlainText("myLabel", text)
		mClipboard.setPrimaryClip(clip)
	}

	private fun forceQuit() {
		forceQuit(0)
	}

	@Keep
	private fun forceQuit(instanceId: Int): Boolean {
		if (primaryHost == null) {
			return false
		}
		return if (instanceId == 0) {
			primaryHost!!.onGodotForceQuit(this)
			true
		} else {
			primaryHost!!.onGodotForceQuit(instanceId)
		}
	}

	fun onBackPressed(host: GodotHost) {
		if (host != primaryHost) {
			return
		}

		var shouldQuit = true
		for (plugin in pluginRegistry.allPlugins) {
			if (plugin.onMainBackPressed()) {
				shouldQuit = false
			}
		}
		if (shouldQuit && renderView != null) {
			renderView!!.queueOnRenderThread { GodotLib.back() }
		}
	}

	private fun getRotatedValues(values: FloatArray?): FloatArray? {
		if (values == null || values.size != 3) {
			return values
		}
		val display =
			(requireActivity().getSystemService(Context.WINDOW_SERVICE) as WindowManager).defaultDisplay
		val displayRotation = display.rotation
		val rotatedValues = FloatArray(3)
		when (displayRotation) {
			Surface.ROTATION_0 -> {
				rotatedValues[0] = values[0]
				rotatedValues[1] = values[1]
				rotatedValues[2] = values[2]
			}
			Surface.ROTATION_90 -> {
				rotatedValues[0] = -values[1]
				rotatedValues[1] = values[0]
				rotatedValues[2] = values[2]
			}
			Surface.ROTATION_180 -> {
				rotatedValues[0] = -values[0]
				rotatedValues[1] = -values[1]
				rotatedValues[2] = values[2]
			}
			Surface.ROTATION_270 -> {
				rotatedValues[0] = values[1]
				rotatedValues[1] = -values[0]
				rotatedValues[2] = values[2]
			}
		}
		return rotatedValues
	}

	override fun onSensorChanged(event: SensorEvent) {
		if (renderView == null) {
			return
		}
		when (event.sensor.type) {
			Sensor.TYPE_ACCELEROMETER -> {
				val rotatedValues = getRotatedValues(event.values)
				renderView!!.queueOnRenderThread {
					GodotLib.accelerometer(
						-rotatedValues!![0], -rotatedValues[1], -rotatedValues[2]
					)
				}
			}
			Sensor.TYPE_GRAVITY -> {
				val rotatedValues = getRotatedValues(event.values)
				renderView!!.queueOnRenderThread {
					GodotLib.gravity(
						-rotatedValues!![0], -rotatedValues[1], -rotatedValues[2]
					)
				}
			}
			Sensor.TYPE_MAGNETIC_FIELD -> {
				val rotatedValues = getRotatedValues(event.values)
				renderView!!.queueOnRenderThread {
					GodotLib.magnetometer(
						-rotatedValues!![0], -rotatedValues[1], -rotatedValues[2]
					)
				}
			}
			Sensor.TYPE_GYROSCOPE -> {
				val rotatedValues = getRotatedValues(event.values)
				renderView!!.queueOnRenderThread {
					GodotLib.gyroscope(
						rotatedValues!![0], rotatedValues[1], rotatedValues[2]
					)
				}
			}
		}
	}

	override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

	/**
	 * Used by the native code (java_godot_wrapper.h) to vibrate the device.
	 * @param durationMs
	 */
	@SuppressLint("MissingPermission")
	@Keep
	private fun vibrate(durationMs: Int) {
		if (durationMs > 0 && requestPermission("VIBRATE")) {
			val vibratorService = getActivity()?.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator? ?: return
			if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
				vibratorService.vibrate(
					VibrationEffect.createOneShot(
						durationMs.toLong(),
						VibrationEffect.DEFAULT_AMPLITUDE
					)
				)
			} else {
				// deprecated in API 26
				vibratorService.vibrate(durationMs.toLong())
			}
		}
	}

	private fun getCommandLine(): MutableList<String> {
		val original: MutableList<String> = parseCommandLine()
		val hostCommandLine = primaryHost?.commandLine
		if (!hostCommandLine.isNullOrEmpty()) {
			original.addAll(hostCommandLine)
		}
		return original
	}

	private fun parseCommandLine(): MutableList<String> {
		val inputStream: InputStream
		return try {
			inputStream = requireActivity().assets.open("_cl_")
			val len = ByteArray(4)
			var r = inputStream.read(len)
			if (r < 4) {
				return mutableListOf()
			}
			val argc =
				(len[3].toInt() and 0xFF) shl 24 or ((len[2].toInt() and 0xFF) shl 16) or ((len[1].toInt() and 0xFF) shl 8) or (len[0].toInt() and 0xFF)
			val cmdline = ArrayList<String>(argc)
			for (i in 0 until argc) {
				r = inputStream.read(len)
				if (r < 4) {
					return mutableListOf()
				}
				val strlen =
					(len[3].toInt() and 0xFF) shl 24 or ((len[2].toInt() and 0xFF) shl 16) or ((len[1].toInt() and 0xFF) shl 8) or (len[0].toInt() and 0xFF)
				if (strlen > 65535) {
					return mutableListOf()
				}
				val arg = ByteArray(strlen)
				r = inputStream.read(arg)
				if (r == strlen) {
					cmdline.add(String(arg, StandardCharsets.UTF_8))
				}
			}
			cmdline
		} catch (e: Exception) {
			// The _cl_ file can be missing with no adverse effect
			mutableListOf()
		}
	}

	/**
	 * Used by the native code (java_godot_wrapper.h) to access the input fallback mapping.
	 * @return The input fallback mapping for the current XR mode.
	 */
	@Keep
	private fun getInputFallbackMapping(): String? {
		return xrMode.inputFallbackMapping
	}

	fun requestPermission(name: String?): Boolean {
		return requestPermission(name, getActivity())
	}

	fun requestPermissions(): Boolean {
		return PermissionsUtil.requestManifestPermissions(getActivity())
	}

	fun getGrantedPermissions(): Array<String?>? {
		return PermissionsUtil.getGrantedPermissions(getActivity())
	}

	/**
	 * Return true if the given feature is supported.
	 */
	@Keep
	private fun hasFeature(feature: String): Boolean {
		for (plugin in pluginRegistry.allPlugins) {
			if (plugin.supportsFeature(feature)) {
				return true
			}
		}
		return false
	}

	/**
	 * Get the list of gdextension modules to register.
	 */
	@Keep
	private fun getGDExtensionConfigFiles(): Array<String> {
		val configFiles = mutableSetOf<String>()
		for (plugin in pluginRegistry.allPlugins) {
			configFiles.addAll(plugin.pluginGDExtensionLibrariesPaths)
		}

		return configFiles.toTypedArray()
	}

	@Keep
	private fun getCACertificates(): String {
		return GodotNetUtils.getCACertificates()
	}

	private fun obbIsCorrupted(f: String, mainPackMd5: String): Boolean {
		return try {
			val fis: InputStream = FileInputStream(f)

			// Create MD5 Hash
			val buffer = ByteArray(16384)
			val complete = MessageDigest.getInstance("MD5")
			var numRead: Int
			do {
				numRead = fis.read(buffer)
				if (numRead > 0) {
					complete.update(buffer, 0, numRead)
				}
			} while (numRead != -1)
			fis.close()
			val messageDigest = complete.digest()

			// Create Hex String
			val hexString = StringBuilder()
			for (b in messageDigest) {
				var s = Integer.toHexString(0xFF and b.toInt())
				if (s.length == 1) {
					s = "0$s"
				}
				hexString.append(s)
			}
			val md5str = hexString.toString()
			md5str != mainPackMd5
		} catch (e: java.lang.Exception) {
			e.printStackTrace()
			true
		}
	}

	@Keep
	private fun initInputDevices() {
		renderView!!.initInputDevices()
	}

	@Keep
	private fun createNewGodotInstance(args: Array<String>): Int {
		return primaryHost?.onNewGodotInstanceRequested(args) ?: 0
	}

	@Keep
	private fun nativeBeginBenchmarkMeasure(label: String) {
		beginBenchmarkMeasure(label)
	}

	@Keep
	private fun nativeEndBenchmarkMeasure(label: String) {
		endBenchmarkMeasure(label)
	}

	@Keep
	private fun nativeDumpBenchmark(benchmarkFile: String) {
		dumpBenchmark(fileAccessHandler, benchmarkFile)
	}
}
