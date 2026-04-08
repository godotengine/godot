/**************************************************************************/
/*  GodotNativeBridge.kt                                                  */
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

package org.godotengine.godot.nativeapi

import android.annotation.SuppressLint
import android.content.Context
import android.content.res.Configuration
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.util.Log
import android.util.Rational
import android.util.TypedValue
import androidx.annotation.Keep
import org.godotengine.godot.Godot
import org.godotengine.godot.GodotActivity
import org.godotengine.godot.GodotLib
import org.godotengine.godot.error.Error
import org.godotengine.godot.feature.PictureInPictureProvider
import org.godotengine.godot.io.FilePicker
import org.godotengine.godot.utils.DialogUtils
import org.godotengine.godot.utils.GodotNetUtils
import org.godotengine.godot.utils.beginBenchmarkMeasure
import org.godotengine.godot.utils.dumpBenchmark
import org.godotengine.godot.utils.endBenchmarkMeasure
import org.godotengine.godot.variant.Callable as GodotCallable

/**
 * Holds and expose Godot apis to the native layer.
 *
 * All the methods in this class are accessed by the native code (java_godot_wrapper.h) and as such are kept private to
 * not be accessible by the rest of the java/kotlin code.
 */
@Keep
internal class GodotNativeBridge(private val godot: Godot) {

	companion object {
		private val TAG = GodotNativeBridge::class.java.simpleName

	}

	private val vibratorService: Vibrator? by lazy { godot.context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator }

	/**
	 * Invoked on the render thread when the Godot setup is complete.
	 */
	private fun onGodotSetupCompleted() {
		godot.onGodotSetupCompleted()
	}

	/**
	 * Invoked on the render thread when the Godot main loop has started.
	 */
	private fun onGodotMainLoopStarted() {
		godot.onGodotMainLoopStarted()
	}

	/**
	 * Invoked on the render thread when the engine is about to terminate.
	 */
	private fun onGodotTerminating() = godot.onGodotTerminating()

	/**
	 * Invoked from the render thread to toggle the immersive mode.
	 */
	private fun nativeEnableImmersiveMode(enabled: Boolean) {
		godot.runOnHostThread {
			godot.enableImmersiveMode(enabled)
		}
	}

	private fun isInImmersiveMode() = godot.isInImmersiveMode()

	private fun isInEdgeToEdgeMode() = godot.isInEdgeToEdgeMode()

	private fun setKeepScreenOn(enabled: Boolean) = godot.setKeepScreenOn(enabled)

	private fun restart() { godot.primaryHost?.onGodotRestartRequested(godot) }

	private fun alert(message: String, title: String) {
		godot.alert(message, title)
	}

	private fun forceQuit(instanceId: Int) = godot.forceQuit(instanceId)

	/**
	 * Returns true if dark mode is supported, false otherwise.
	 */
	private fun isDarkModeSupported(): Boolean {
		return godot.context.resources?.configuration?.uiMode?.and(Configuration.UI_MODE_NIGHT_MASK) != Configuration.UI_MODE_NIGHT_UNDEFINED
	}

	/**
	 * Returns true if dark mode is supported and enabled, false otherwise.
	 */
	private fun isDarkMode() = godot.darkMode

	private fun showFilePicker(currentDirectory: String, filename: String, fileMode: Int, filters: Array<String>) {
		FilePicker.showFilePicker(godot.context, godot.getActivity(), currentDirectory, filename, fileMode, filters)
	}

	/**
	 * This method shows a dialog with multiple buttons.
	 *
	 * @param title The title of the dialog.
	 * @param message The message displayed in the dialog.
	 * @param buttons An array of button labels to display.
	 */
	private fun showDialog(title: String, message: String, buttons: Array<String>) {
		godot.getActivity()?.let { DialogUtils.showDialog(it, title, message, buttons) }
	}

	/**
	 * This method shows a dialog with a text input field, allowing the user to input text.
	 *
	 * @param title The title of the input dialog.
	 * @param message The message displayed in the input dialog.
	 * @param existingText The existing text that will be pre-filled in the input field.
	 */
	private fun showInputDialog(title: String, message: String, existingText: String) {
		godot.getActivity()?.let { DialogUtils.showInputDialog(it, title, message, existingText) }
	}

	private fun getAccentColor(): Int {
		val value = TypedValue()
		godot.context.theme.resolveAttribute(android.R.attr.colorAccent, value, true)
		return value.data
	}

	private fun getBaseColor(): Int {
		val value = TypedValue()
		godot.context.theme.resolveAttribute(android.R.attr.colorBackground, value, true)
		return value.data
	}

	private fun requestPermission(name: String?) = godot.requestPermission(name)

	private fun requestPermissions() = godot.requestPermissions()

	private fun getGrantedPermissions() = godot.getGrantedPermissions()

	/**
	 * Used by the native code (java_godot_wrapper.h) to vibrate the device.
	 * @param durationMs
	 */
	@SuppressLint("MissingPermission")
	private fun vibrate(durationMs: Int, amplitude: Int) {
		if (durationMs > 0 && godot.requestPermission("VIBRATE")) {
			try {
				if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
					if (amplitude <= -1) {
						vibratorService?.vibrate(
							VibrationEffect.createOneShot(
								durationMs.toLong(),
								VibrationEffect.DEFAULT_AMPLITUDE
							)
						)
					} else {
						vibratorService?.vibrate(
							VibrationEffect.createOneShot(
								durationMs.toLong(),
								amplitude
							)
						)
					}
				} else {
					// deprecated in API 26
					vibratorService?.vibrate(durationMs.toLong())
				}
			} catch (e: SecurityException) {
				Log.w(
					TAG,
					"SecurityException: VIBRATE permission not found. Make sure it is declared in the manifest or enabled in the export preset."
				)
			}
		}
	}

	/**
	 * Internal method used to query whether the host or the registered plugins supports a given feature.
	 *
	 * This is invoked by the native code, and should not be confused with [hasFeature] which is the Android version of
	 * https://docs.godotengine.org/en/stable/classes/class_os.html#class-os-method-has-feature
	 */
	private fun checkInternalFeatureSupport(feature: String): Boolean {
		if (godot.primaryHost?.supportsFeature(feature) == true) {
			return true
		}

		for (plugin in godot.pluginRegistry.allPlugins) {
			if (plugin.supportsFeature(feature)) {
				return true
			}
		}
		return false
	}

	/**
	 * Get the list of gdextension modules to register.
	 */
	private fun getGDExtensionConfigFiles(): Array<String> {
		val configFiles = mutableSetOf<String>()
		for (plugin in godot.pluginRegistry.allPlugins) {
			configFiles.addAll(plugin.pluginGDExtensionLibrariesPaths)
		}

		return configFiles.toTypedArray()
	}

	private fun getCACertificates(): String {
		return GodotNetUtils.getCACertificates()
	}

	private fun getActivity() = godot.getActivity()

	private fun getRenderView() = godot.renderView

	private fun getClipboard() = godot.getClipboard()

	private fun setClipboard(text: String) = godot.setClipboard(text)

	private fun hasClipboard() = godot.hasClipboard()

	private fun setWindowColor(color: String) = godot.setWindowColor(color)

	/**
	 * Used by the native code (java_godot_wrapper.h) to access the input fallback mapping.
	 * @return The input fallback mapping for the current XR mode.
	 */
	private fun getInputFallbackMapping(): String? {
		return godot.xrMode.inputFallbackMapping
	}

	private fun initInputDevices() {
		godot.godotInputHandler.initInputDevices()
	}

	private fun createNewGodotInstance(args: Array<String>): Int {
		return godot.primaryHost?.onNewGodotInstanceRequested(args) ?: -1
	}

	private fun nativeBeginBenchmarkMeasure(scope: String, label: String) {
		beginBenchmarkMeasure(scope, label)
	}

	private fun nativeEndBenchmarkMeasure(scope: String, label: String) {
		endBenchmarkMeasure(scope, label)
	}

	private fun nativeDumpBenchmark(benchmarkFile: String) {
		dumpBenchmark(godot.fileAccessHandler, benchmarkFile)
	}

	private fun nativeSignApk(
		inputPath: String,
		outputPath: String,
		keystorePath: String,
		keystoreUser: String,
		keystorePassword: String
	): Int {
		val signResult = godot.primaryHost?.signApk(inputPath, outputPath, keystorePath, keystoreUser, keystorePassword)
			?: org.godotengine.godot.error.Error.ERR_UNAVAILABLE
		return signResult.toNativeValue()
	}

	private fun nativeVerifyApk(apkPath: String): Int {
		val verifyResult = godot.primaryHost?.verifyApk(apkPath) ?: Error.ERR_UNAVAILABLE
		return verifyResult.toNativeValue()
	}

	private fun nativeOnEditorWorkspaceSelected(workspace: String) {
		godot.primaryHost?.onEditorWorkspaceSelected(workspace)
	}

	private fun nativeOnDistractionFreeModeChanged(enabled: Boolean) {
		godot.primaryHost?.onDistractionFreeModeChanged(enabled)
	}

	private fun nativeBuildEnvConnect(callback: GodotCallable): Boolean {
		try {
			val buildProvider = godot.primaryHost?.getBuildProvider()
			return buildProvider?.buildEnvConnect(callback) ?: false
		} catch (e: Exception) {
			Log.e(TAG, "Unable to connect to build environment", e)
			return false
		}
	}

	private fun nativeBuildEnvDisconnect() {
		try {
			val buildProvider = godot.primaryHost?.getBuildProvider()
			buildProvider?.buildEnvDisconnect()
		} catch (e: Exception) {
			Log.e(TAG, "Unable to disconnect from build environment", e)
		}
	}

	private fun nativeBuildEnvExecute(
		buildTool: String,
		arguments: Array<String>,
		projectPath: String,
		buildDir: String,
		outputCallback: GodotCallable,
		resultCallback: GodotCallable
	): Int {
		try {
			val buildProvider = godot.primaryHost?.getBuildProvider()
			return buildProvider?.buildEnvExecute(
				buildTool,
				arguments,
				projectPath,
				buildDir,
				outputCallback,
				resultCallback
			) ?: -1
		} catch (e: Exception) {
			Log.e(TAG, "Unable to execute Gradle command in build environment", e);
			return -1
		}
	}

	private fun nativeBuildEnvCancel(jobId: Int) {
		try {
			val buildProvider = godot.primaryHost?.getBuildProvider()
			buildProvider?.buildEnvCancel(jobId)
		} catch (e: Exception) {
			Log.e(TAG, "Unable to cancel command in build environment", e)
		}
	}

	private fun nativeBuildEnvCleanProject(projectPath: String, buildDir: String, callback: GodotCallable) {
		try {
			val buildProvider = godot.primaryHost?.getBuildProvider()
			buildProvider?.buildEnvCleanProject(projectPath, buildDir, callback)
		} catch (e: Exception) {
			Log.e(TAG, "Unable to clean project in build environment", e)
		}
	}

	private fun nativeIsPiPModeSupported(): Boolean {
		val hostActivity = godot.getActivity()
		if (hostActivity is PictureInPictureProvider) {
			return hostActivity.isPiPModeSupported()
		}
		return false
	}

	private fun nativeIsInPiPMode(): Boolean {
		val hostActivity = godot.getActivity()
		if (hostActivity is GodotActivity) {
			return hostActivity.isInPictureInPictureMode
		}
		return false
	}

	private fun nativeEnterPiPMode() {
		val hostActivity = godot.getActivity()
		if (hostActivity is PictureInPictureProvider) {
			godot.runOnHostThread {
				hostActivity.enterPiPMode()
			}
		}
	}

	private fun nativeSetPiPModeAspectRatio(numerator: Int, denominator: Int) {
		val hostActivity = godot.getActivity()
		if (hostActivity is GodotActivity) {
			godot.runOnHostThread {
				hostActivity.updatePiPParams(aspectRatio = Rational(numerator, denominator))
			}
		}
	}

	private fun nativeSetAutoEnterPiPModeOnBackground(autoEnterPiPOnBackground: Boolean) {
		val hostActivity = godot.getActivity()
		if (hostActivity is GodotActivity) {
			godot.runOnHostThread {
				hostActivity.updatePiPParams(enableAutoEnter = autoEnterPiPOnBackground)
			}
		}
	}
}
