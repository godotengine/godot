/**************************************************************************/
/*  GodotService.kt                                                       */
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

package org.godotengine.godot.remote

import android.app.Service
import android.content.Intent
import android.hardware.display.DisplayManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Message
import android.os.Messenger
import android.os.Process
import android.os.RemoteException
import android.text.TextUtils
import android.util.Log
import android.view.SurfaceControlViewHost
import androidx.annotation.CallSuper
import androidx.annotation.RequiresApi
import org.godotengine.godot.Godot
import org.godotengine.godot.GodotHost
import org.godotengine.godot.R
import java.lang.ref.WeakReference

/**
 * Specialized [Service] implementation able to host a Godot engine instance.
 *
 * A core characteristic of this component is that it lacks access to an [android.app.Activity]
 * instance, and as such it does not have full access to the set of Godot UI capabilities.
 */
@RequiresApi(Build.VERSION_CODES.R)
open class GodotService : Service(), GodotHost {

	companion object {
		private val TAG = GodotService::class.java.simpleName

		// Keys to store / retrieve msg payloads
		const val KEY_COMMAND_LINE_PARAMETERS = "commandLineParameters"
		const val KEY_HOST_TOKEN = "hostToken"
		const val KEY_DISPLAY_ID = "displayId"
		const val KEY_WIDTH = "width"
		const val KEY_HEIGHT = "height"
		const val KEY_SURFACE_PACKAGE = "surfacePackage"
		const val KEY_ENGINE_STATUS = "engineStatus"
		const val KEY_ENGINE_ERROR = "engineError"

		// Set of commands from the client to the service
		const val MSG_INIT_ENGINE = 0
		const val MSG_START_ENGINE = MSG_INIT_ENGINE + 1
		const val MSG_STOP_ENGINE = MSG_START_ENGINE + 1
		const val MSG_DESTROY_ENGINE = MSG_STOP_ENGINE + 1

		// Set of commands from the service to the client
		const val MSG_ENGINE_ERROR = 100
		const val MSG_ENGINE_STATUS_UPDATE = 101
		const val MSG_ENGINE_RESTART_REQUESTED = 102
	}

	enum class EngineStatus {
		INITIALIZED,
		STARTED,
		STOPPED,
		DESTROYED,
	}

	enum class EngineError {
		ALREADY_BOUND,
		INIT_FAILED,
	}

	/**
	 * Handler of incoming messages from clients.
	 */
	private class IncomingHandler(private val serviceRef: WeakReference<GodotService>) : Handler() {
		private var viewHost: SurfaceControlViewHost? = null
		private var boundClient: Messenger? = null

		override fun handleMessage(msg: Message) {
			val service = serviceRef.get() ?: return

			Log.d(TAG, "HandleMessage: $msg")

			if (msg.replyTo == null) {
				// Messages for this handler must have a valid 'replyTo' field
				super.handleMessage(msg)
				return
			}

			try {
				if (boundClient != null && boundClient != msg.replyTo) {
					Log.e(TAG, "Engine is already bound to another client")
					msg.replyTo.send(Message.obtain().apply {
						what = MSG_ENGINE_ERROR
						data.putString(KEY_ENGINE_ERROR, EngineError.ALREADY_BOUND.name)
					})
					return
				}

				when (msg.what) {
					MSG_INIT_ENGINE -> {
						val msgData = msg.data
						if (msgData.isEmpty) {
							Log.e(TAG, "Invalid message data from binding client.. Aborting")
							return
						}

						if (!service.godot.isInitialized()) {
							val initArgs = msgData.getStringArray(KEY_COMMAND_LINE_PARAMETERS)
							if (!initArgs.isNullOrEmpty()) {
								service.updateCommandLineParams(initArgs.asList())
							}

							if (!service.performEngineInitialization()) {
								Log.e(TAG, "Unable to initialize Godot engine")
								msg.replyTo.send(Message.obtain().apply {
									what = MSG_ENGINE_ERROR
									data.putString(KEY_ENGINE_ERROR, EngineError.INIT_FAILED.name)
								})
								return
							}
						}

						if (viewHost != null) {
							return
						}

						val hostToken = msgData.getBinder(KEY_HOST_TOKEN)
						val width = msgData.getInt(KEY_WIDTH)
						val height = msgData.getInt(KEY_HEIGHT)
						val displayId = msgData.getInt(KEY_DISPLAY_ID)
						val display = service.getSystemService(DisplayManager::class.java)
							.getDisplay(displayId)

						Log.d(TAG, "Setting up SurfaceControlViewHost")
						val godotContainerLayout = service.godot.containerLayout
						if (godotContainerLayout == null) {
							Log.e(TAG, "Unable to retrieve the Godot container layout")
							return
						}

						viewHost = SurfaceControlViewHost(service, display, hostToken).apply {
							setView(godotContainerLayout, width, height)

							Log.i(TAG, "Initialized Godot engine")
							msg.replyTo.send(Message.obtain().apply {
								what = MSG_ENGINE_STATUS_UPDATE
								data.apply {
									putString(KEY_ENGINE_STATUS, EngineStatus.INITIALIZED.name)
									putParcelable(KEY_SURFACE_PACKAGE, surfacePackage)
								}
							})
						}
						boundClient = msg.replyTo
					}

					MSG_START_ENGINE -> {
						if (boundClient == null || !service.godot.isInitialized()) {
							Log.e(TAG, "Attempting to start uninitialized Godot engine instance")
							return
						}

						Log.d(TAG, "Starting Godot engine")
						service.godot.onStart(service)
						service.godot.onResume(service)

						boundClient?.send(Message.obtain().apply {
							what = MSG_ENGINE_STATUS_UPDATE
							data.putString(KEY_ENGINE_STATUS, EngineStatus.STARTED.name)
						})
					}

					MSG_STOP_ENGINE -> {
						if (boundClient == null || !service.godot.isInitialized()) {
							Log.e(TAG, "Attempting to stop uninitialized Godot engine instance")
							return
						}

						Log.d(TAG, "Stopping Godot engine")
						service.godot.onPause(service)
						service.godot.onStop(service)

						boundClient?.send(Message.obtain().apply {
							what = MSG_ENGINE_STATUS_UPDATE
							data.putString(KEY_ENGINE_STATUS, EngineStatus.STOPPED.name)
						})
					}

					MSG_DESTROY_ENGINE -> {
						destroyEngine()
					}

					else -> super.handleMessage(msg)
				}
			} catch (e: RemoteException) {
				Log.e(TAG, "Unable to handle message", e)
			}
		}

		fun destroyEngine() {
			val service = serviceRef.get() ?: return

			if (viewHost != null) {
				Log.d(TAG, "Releasing SurfaceControlViewHost")
				viewHost?.release()
				viewHost = null
			}

			if (service.godot.isInitialized()) {
				service.godot.onDestroy(service)

				boundClient?.send(Message.obtain().apply {
					what = MSG_ENGINE_STATUS_UPDATE
					data.putString(KEY_ENGINE_STATUS, EngineStatus.DESTROYED.name)
				})
			}
			boundClient = null
		}

		fun requestRestart() {
			try {
				boundClient?.send(Message.obtain(null, MSG_ENGINE_RESTART_REQUESTED))
			} catch (e: RemoteException) {
				Log.w(TAG, "Unable to send restart request", e)
			}
		}
	}

	private val commandLineParams = ArrayList<String>()

	private lateinit var godot: Godot

	private val handler = IncomingHandler(WeakReference(this))
	private val messenger = Messenger(handler)

	override fun getActivity() = null
	override fun getGodot() = godot
	final override fun getCommandLine() = commandLineParams

	override fun onCreate() {
		Log.d(TAG, "OnCreate")
		super.onCreate()
		godot = Godot(this)
	}

	@CallSuper
	protected open fun updateCommandLineParams(args: List<String>) {
		// Update the list of command line params with the new args
		commandLineParams.clear()
		if (args.isNotEmpty()) {
			commandLineParams.addAll(args)
		}
	}

	private fun performEngineInitialization(): Boolean {
		Log.d(TAG, "Performing engine initialization")
		try {
			// Initialize the Godot instance
			godot.onCreate(this)

			if (!godot.onInitNativeLayer(this)) {
				throw IllegalStateException("Unable to initialize engine native layer")
			}

			if (godot.onInitRenderView(this) == null) {
				throw IllegalStateException("Unable to initialize engine render view")
			}
			return true
		} catch (e: IllegalStateException) {
			Log.e(TAG, "Engine initialization failed", e)
			val errorMessage = if (TextUtils.isEmpty(e.message)
			) {
				getString(R.string.error_engine_setup_message)
			} else {
				e.message!!
			}
			godot.alert(errorMessage, getString(R.string.text_error_title)) { godot.destroyAndKillProcess() }
			return false
		}
	}

	override fun onDestroy() {
		Log.d(TAG, "OnDestroy")
		super.onDestroy()
		handler.destroyEngine()
	}

	private fun forceQuitService() {
		Process.killProcess(Process.myPid())
		Runtime.getRuntime().exit(0)
	}

	override fun onBind(intent: Intent?) : IBinder? = messenger.binder

	override fun runOnHostThread(action: Runnable) {
		if (Thread.currentThread() != handler.looper.thread) {
			handler.post(action)
		} else {
			action.run()
		}
	}

	override fun onGodotForceQuit(instance: Godot) {
		if (instance === godot) {
			Log.d(TAG, "Force quitting Godot service")
			forceQuitService()
		}
	}

	override fun onGodotRestartRequested(instance: Godot) {
		if (instance === godot) {
			Log.d(TAG, "Restarting Godot service")
			handler.requestRestart()
		}
	}

}
