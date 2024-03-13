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
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.annotation.CallSuper
import androidx.fragment.app.FragmentActivity
import org.godotengine.godot.utils.PermissionsUtil
import org.godotengine.godot.utils.ProcessPhoenix

// <TF>
// @ShadyTF : Thermal status support
import android.os.Build
import android.os.PowerManager
import androidx.annotation.RequiresApi



class GodotThermalWrapper {
	companion object {
		public var power_manager: PowerManager? = null
		@RequiresApi(Build.VERSION_CODES.R)
		public fun getThermalHeadRoom(forecast: Int): Float {
			if (Build.VERSION.SDK_INT >= 30) {
				return power_manager!!.getThermalHeadroom(forecast);
			} else {
				return -1.0f;
			}
		}
	}
}
// </TF>
/**
 * Base abstract activity for Android apps intending to use Godot as the primary screen.
 *
 * Also a reference implementation for how to setup and use the [GodotFragment] fragment
 * within an Android app.
 */
abstract class GodotActivity : FragmentActivity(), GodotHost {

	companion object {
		private val TAG = GodotActivity::class.java.simpleName

		@JvmStatic
		protected val EXTRA_FORCE_QUIT = "force_quit_requested"
		@JvmStatic
		protected val EXTRA_NEW_LAUNCH = "new_launch_requested"
	}

	// <TF>
	// @ShadyTF : Thermal status support
	private var thermalStatusMap: HashMap<Int, Int>? = null
	private val MOBILE_THERMAL_NOT_SUPPORTED = -2
	private val MOBILE_THERMAL_ERROR = -1
	private external fun nativeThermalEvent(nativeStatus: Int)
	// </TF>
	/**
	 * Interaction with the [Godot] object is delegated to the [GodotFragment] class.
	 */
	protected var godotFragment: GodotFragment? = null
		private set

	override fun onCreate(savedInstanceState: Bundle?) {
		super.onCreate(savedInstanceState)
		setContentView(R.layout.godot_app_layout)

		handleStartIntent(intent, true)
		// <TF>
		// @ShadyTF : Thermal status support
		thermalStatusMap = java.util.HashMap()

		try {
			if (Build.VERSION.SDK_INT >= 29) {
				thermalStatusMap!![PowerManager.THERMAL_STATUS_NONE]      = 0 // Main::ThermalState::THERMAL_STATE_NONE
				thermalStatusMap!![PowerManager.THERMAL_STATUS_LIGHT]     = 1 // Main::ThermalState::THERMAL_STATE_LIGHT
				thermalStatusMap!![PowerManager.THERMAL_STATUS_MODERATE]  = 2 // Main::ThermalState::THERMAL_STATE_MODERATE
				thermalStatusMap!![PowerManager.THERMAL_STATUS_SEVERE]    = 3 // Main::ThermalState::THERMAL_STATE_SEVERE
				thermalStatusMap!![PowerManager.THERMAL_STATUS_CRITICAL]  = 4 // Main::ThermalState::THERMAL_STATE_CRITICAL
				thermalStatusMap!![PowerManager.THERMAL_STATUS_EMERGENCY] = 5 // Main::ThermalState::THERMAL_STATE_EMERGENCY
				thermalStatusMap!![PowerManager.THERMAL_STATUS_SHUTDOWN]  = 6 // Main::ThermalState::THERMAL_STATE_SHUTDOWN
			}
		} catch (e: Exception) {
			Log.i(TAG, "Failed to populate mMobileThermalMap: " + e.message)
		}

		try {
			if (Build.VERSION.SDK_INT >= 29) {
				val pManager = applicationContext.getSystemService(POWER_SERVICE) as PowerManager
				GodotThermalWrapper.Companion.power_manager = pManager;
				pManager.addThermalStatusListener { status: Int ->
					val nativeStatus = thermalStatusMap!![status]
					nativeThermalEvent(nativeStatus ?: MOBILE_THERMAL_ERROR)
				}
			} else {
				nativeThermalEvent(MOBILE_THERMAL_NOT_SUPPORTED)
			}
		} catch (e: java.lang.Exception) {
			Log.i(TAG, "Failed to setup thermal status listener: " + e.message)
		}
		// </TF>
		val currentFragment = supportFragmentManager.findFragmentById(R.id.godot_fragment_container)
		if (currentFragment is GodotFragment) {
			Log.v(TAG, "Reusing existing Godot fragment instance.")
			godotFragment = currentFragment
		} else {
			Log.v(TAG, "Creating new Godot fragment instance.")
			godotFragment = initGodotInstance()
			supportFragmentManager.beginTransaction().replace(R.id.godot_fragment_container, godotFragment!!).setPrimaryNavigationFragment(godotFragment).commitNowAllowingStateLoss()
		}
	}

	override fun onDestroy() {
		Log.v(TAG, "Destroying Godot app...")
		super.onDestroy()
		if (godotFragment != null) {
			terminateGodotInstance(godotFragment!!.godot)
		}
	}

	override fun onGodotForceQuit(instance: Godot) {
		runOnUiThread { terminateGodotInstance(instance) }
	}

	private fun terminateGodotInstance(instance: Godot) {
		if (godotFragment != null && instance === godotFragment!!.godot) {
			Log.v(TAG, "Force quitting Godot instance")
			ProcessPhoenix.forceQuit(this)
		}
	}

	override fun onGodotRestartRequested(instance: Godot) {
		runOnUiThread {
			if (godotFragment != null && instance === godotFragment!!.godot) {
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

	override fun onNewIntent(newIntent: Intent) {
		super.onNewIntent(newIntent)
		intent = newIntent

		handleStartIntent(newIntent, false)

		godotFragment?.onNewIntent(newIntent)
	}

	private fun handleStartIntent(intent: Intent, newLaunch: Boolean) {
		val forceQuitRequested = intent.getBooleanExtra(EXTRA_FORCE_QUIT, false)
		if (forceQuitRequested) {
			Log.d(TAG, "Force quit requested, terminating..")
			ProcessPhoenix.forceQuit(this)
			return
		}
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
}
