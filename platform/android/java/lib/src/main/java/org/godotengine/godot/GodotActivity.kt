/**************************************************************************/
/*  GodotActivity.kt                                                      */
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

import android.app.Activity
import android.app.PictureInPictureParams
import android.content.ComponentName
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Rect
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.util.Rational
import android.view.View
import androidx.annotation.CallSuper
import androidx.annotation.LayoutRes
import androidx.fragment.app.FragmentActivity
import org.godotengine.godot.feature.PictureInPictureProvider
import org.godotengine.godot.utils.CommandLineFileParser
import org.godotengine.godot.utils.PermissionsUtil
import org.godotengine.godot.utils.ProcessPhoenix
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/**
 * Base abstract activity for Android apps intending to use Godot as the primary screen.
 *
 * Also a reference implementation for how to setup and use the [GodotFragment] fragment
 * within an Android app.
 */
abstract class GodotActivity : FragmentActivity(), GodotHost, PictureInPictureProvider {

	companion object {
		private val TAG = GodotActivity::class.java.simpleName

		@JvmStatic
		val EXTRA_COMMAND_LINE_PARAMS = "command_line_params"

		@JvmStatic
		protected val EXTRA_NEW_LAUNCH = "new_launch_requested"

		// This window must not match those in BaseGodotEditor.RUN_GAME_INFO etc
		@JvmStatic
		private final val DEFAULT_WINDOW_ID = 664;
	}

	/**
	 * Set to true if the activity should automatically enter picture-in-picture when put in the background.
	 */
	private val pipAspectRatio = AtomicReference<Rational>()
	private val autoEnterPiP = AtomicBoolean(false)
	private val gameViewSourceRectHint = Rect()
	private val commandLineParams = ArrayList<String>()
	/**
	 * Interaction with the [Godot] object is delegated to the [GodotFragment] class.
	 */
	protected var godotFragment: GodotFragment? = null
		private set

	/**
	 * Strip out the command line parameters from intent targeting exported activities.
	 */
	protected fun sanitizeLaunchIntent(launchIntent: Intent = intent): Intent {
		val targetComponent = launchIntent.component ?: componentName
		val activityInfo = packageManager.getActivityInfo(targetComponent, 0)
		if (activityInfo.exported) {
			launchIntent.removeExtra(EXTRA_COMMAND_LINE_PARAMS)
		}

		return launchIntent
	}

	/**
	 * Only retrieve the command line parameters from the intent from non-exported activities.
	 * This ensures only internal components can configure how the engine is run.
	 */
	protected fun retrieveCommandLineParamsFromLaunchIntent(launchIntent: Intent = intent): Array<String> {
		val targetComponent = launchIntent.component ?: componentName
		val activityInfo = packageManager.getActivityInfo(targetComponent, 0)
		if (!activityInfo.exported) {
			val params = launchIntent.getStringArrayExtra(EXTRA_COMMAND_LINE_PARAMS)
			return params ?: emptyArray()
		}
		return emptyArray()
	}

	@CallSuper
	override fun onCreate(savedInstanceState: Bundle?) {
		intent = sanitizeLaunchIntent(intent)

		val assetsCommandLine = try {
			CommandLineFileParser.parseCommandLine(assets.open("_cl_"))
		} catch (_: Exception) {
			mutableListOf()
		}
		Log.d(TAG, "Project command line parameters: $assetsCommandLine")
		commandLineParams.addAll(assetsCommandLine)

		val intentCommandLine = retrieveCommandLineParamsFromLaunchIntent()
		Log.d(TAG, "Launch intent $intent with parameters ${intentCommandLine.contentToString()}")
		commandLineParams.addAll(intentCommandLine)

		super.onCreate(savedInstanceState)

		setContentView(getGodotAppLayout())

		handleStartIntent(intent, true)

		val currentFragment = supportFragmentManager.findFragmentById(R.id.godot_fragment_container)
		if (currentFragment is GodotFragment) {
			Log.v(TAG, "Reusing existing Godot fragment instance.")
			godotFragment = currentFragment
		} else {
			Log.v(TAG, "Creating new Godot fragment instance.")
			godotFragment = initGodotInstance()

			val transaction = supportFragmentManager.beginTransaction()
			if (currentFragment != null) {
				Log.v(TAG, "Removing existing fragment before replacement.")
				transaction.remove(currentFragment)
			}

			transaction.replace(R.id.godot_fragment_container, godotFragment!!)
				.setPrimaryNavigationFragment(godotFragment)
				.commitNowAllowingStateLoss()
		}

		if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
			val gameView = findViewById<View>(R.id.godot_fragment_container)
			gameView?.addOnLayoutChangeListener { v, left, top, right, bottom, oldLeft, oldTop, oldRight, oldBottom ->
				gameView.getGlobalVisibleRect(gameViewSourceRectHint)
			}
		}
	}

	override fun onNewGodotInstanceRequested(args: Array<String>): Int {
		Log.d(TAG, "Restarting with parameters ${args.contentToString()}")
		val intent = Intent()
			.setComponent(ComponentName(this, javaClass.name))
			.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
			.putExtra(EXTRA_COMMAND_LINE_PARAMS, args)
		triggerRebirth(null, intent)
		// fake 'process' id returned by create_instance() etc
		return DEFAULT_WINDOW_ID
	}

	protected fun triggerRebirth(bundle: Bundle?, intent: Intent) {
		// Launch a new activity
		Godot.getInstance(applicationContext).destroyAndKillProcess {
			ProcessPhoenix.triggerRebirth(this, bundle, intent)
		}
	}

	@LayoutRes
	protected open fun getGodotAppLayout() = R.layout.godot_app_layout

	override fun onDestroy() {
		Log.v(TAG, "Destroying GodotActivity $this...")
		super.onDestroy()
	}

	override fun onStop() {
		super.onStop()

		if (isInPictureInPictureMode && !isFinishing) {
			// We get in this state when PiP is closed, so we terminate the activity.
			finish()
		}
	}

	override fun onGodotForceQuit(instance: Godot) {
		runOnUiThread { terminateGodotInstance(instance) }
	}

	private fun terminateGodotInstance(instance: Godot) {
		godotFragment?.let {
			if (instance === it.godot) {
				Log.v(TAG, "Force quitting Godot instance")
				ProcessPhoenix.forceQuit(this)
			}
		}
	}

	override fun onGodotRestartRequested(instance: Godot) {
		runOnUiThread {
			godotFragment?.let {
				if (instance === it.godot) {
					// It's very hard to properly de-initialize Godot on Android to restart the game
					// from scratch. Therefore, we need to kill the whole app process and relaunch it.
					//
					// Restarting only the activity, wouldn't be enough unless it did proper cleanup (including
					// releasing and reloading native libs or resetting their state somehow and clearing static data).
					Log.v(TAG, "Restarting Godot instance...")
					ProcessPhoenix.triggerRebirth(this)
				}
			}
		}
	}

	override fun onGodotSetupCompleted() {
		super.onGodotSetupCompleted()

		if (isPiPEnabled()) {
			try {
				// Update the aspect ratio for picture-in-picture mode.
				val viewportWidth = Integer.parseInt(GodotLib.getGlobal("display/window/size/viewport_width"))
				val viewportHeight = Integer.parseInt(GodotLib.getGlobal("display/window/size/viewport_height"))
				pipAspectRatio.set(Rational(viewportWidth, viewportHeight))
			} catch (e: NumberFormatException) {
				Log.w(TAG, "Unable to parse viewport dimensions.", e)
			}

			runOnHostThread { updatePiPParams() }
		}
	}

	override fun onNewIntent(newIntent: Intent) {
		intent = sanitizeLaunchIntent(newIntent)
		super.onNewIntent(intent)
		handleStartIntent(intent, false)
	}

	private fun handleStartIntent(intent: Intent, newLaunch: Boolean) {
		if (!newLaunch) {
			val newLaunchRequested = intent.getBooleanExtra(EXTRA_NEW_LAUNCH, false)
			if (newLaunchRequested) {
				Log.d(TAG, "New launch requested, restarting..")
				val restartIntent = Intent(intent).putExtra(EXTRA_NEW_LAUNCH, false)
				ProcessPhoenix.triggerRebirth(this, restartIntent)
				return
			}
		}
	}

	@CallSuper
	override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
		super.onActivityResult(requestCode, resultCode, data)
		godotFragment?.onActivityResult(requestCode, resultCode, data)
	}

	@CallSuper
	override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
		super.onRequestPermissionsResult(requestCode, permissions, grantResults)
		godotFragment?.onRequestPermissionsResult(requestCode, permissions, grantResults)

		// Logging the result of permission requests
		if (requestCode == PermissionsUtil.REQUEST_ALL_PERMISSION_REQ_CODE || requestCode == PermissionsUtil.REQUEST_SINGLE_PERMISSION_REQ_CODE) {
			Log.d(TAG, "Received permissions request result..")
			for (i in permissions.indices) {
				val permissionGranted = grantResults[i] == PackageManager.PERMISSION_GRANTED
				Log.d(TAG, "Permission ${permissions[i]} ${if (permissionGranted) { "granted"} else { "denied" }}")
			}
		}
	}

	override fun onBackPressed() {
		godotFragment?.onBackPressed() ?: super.onBackPressed()
	}

	override fun getActivity(): Activity? {
		return this
	}

	override fun getGodot(): Godot? {
		return godotFragment?.godot
	}

	/**
	 * Used to initialize the Godot fragment instance in [onCreate].
	 */
	protected open fun initGodotInstance(): GodotFragment {
		return GodotFragment()
	}

	@CallSuper
	override fun getCommandLine(): MutableList<String> = commandLineParams

	override fun onPictureInPictureModeChanged(isInPictureInPictureMode: Boolean) {
		super.onPictureInPictureModeChanged(isInPictureInPictureMode)
		godot?.onPictureInPictureModeChanged(isInPictureInPictureMode)
	}

	/**
	 * Returns true if picture-in-picture (PiP) mode is supported.
	 */
	override fun isPiPModeSupported() = isPiPEnabled() && packageManager.hasSystemFeature(PackageManager.FEATURE_PICTURE_IN_PICTURE)

	/**
	 * Returns true if the current activity has enabled picture-in-picture in its manifest declaration using
	 * 'android:supportsPictureInPicture="true"'
	 */
	protected open fun isPiPEnabled() = false

	internal fun updatePiPParams(enableAutoEnter: Boolean = autoEnterPiP.get(), aspectRatio: Rational? = pipAspectRatio.get()) {
		if (isPiPModeSupported()) {
			autoEnterPiP.set(enableAutoEnter)
			pipAspectRatio.set(aspectRatio)

			if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
				val builder = PictureInPictureParams.Builder()
					.setSourceRectHint(gameViewSourceRectHint)
					.setAspectRatio(aspectRatio)
				if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
					builder.setSeamlessResizeEnabled(false)
						.setAutoEnterEnabled(enableAutoEnter)
				}
				if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
					builder.setExpandedAspectRatio(aspectRatio)
				}
				setPictureInPictureParams(builder.build())
			}
		}
	}

	override fun enterPiPMode() {
		if (isPiPModeSupported()) {
			updatePiPParams()

			Log.v(TAG, "Entering PiP mode")
			enterPictureInPictureMode()
		}
	}

	override fun onUserLeaveHint() {
		if (autoEnterPiP.get()) {
			enterPiPMode()
		}
	}
}
