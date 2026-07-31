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

package org.godotengine.godot.service

import android.app.Service
import android.content.Intent
import android.hardware.display.DisplayManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Message
import android.os.Messenger
import android.os.Process
import android.os.RemoteException
import android.text.TextUtils
import android.util.Log
import android.view.SurfaceControlViewHost
import android.widget.FrameLayout
import androidx.annotation.CallSuper
import androidx.annotation.RequiresApi
import androidx.core.os.bundleOf
import org.godotengine.godot.Godot
import org.godotengine.godot.GodotHost
import org.godotengine.godot.R
import java.lang.ref.WeakReference

/**
 * Specialized [Service] implementation able to host a Godot engine instance.
 *
 * When used remotely (from another process), this component lacks access to an [android.app.Activity]
 * instance, and as such it does not have full access to the set of Godot UI capabilities.
 *
 * Limitations: As of version 4.5, use of vulkan + swappy causes [GodotService] to crash as swappy requires an Activity
 * context. So [GodotService] should be used with OpenGL or with Vulkan with swappy disabled.
 */
open class GodotService : Service() {

	companion object {
		private val TAG = GodotService::class.java.simpleName

		const val EXTRA_MSG_PAYLOAD = "extraMsgPayload"

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

		@RequiresApi(Build.VERSION_CODES.R)
		const val MSG_WRAP_ENGINE_WITH_SCVH = MSG_DESTROY_ENGINE + 1

		// Set of commands from the service to the client
		const val MSG_ENGINE_ERROR = 100
		const val MSG_ENGINE_STATUS_UPDATE = 101
		const val MSG_ENGINE_RESTART_REQUESTED = 102
	}

	enum class EngineStatus {
		INITIALIZED,
		SCVH_CREATED,
		STARTED,
		STOPPED,
		DESTROYED,
	}

	enum class EngineError {
		ALREADY_BOUND,
		INIT_FAILED,
		SCVH_CREATION_FAILED,
	}

	/**
	 * Used to subscribe to engine's updates.
	 */
	private class RemoteListener(val handlerRef: WeakReference<IncomingHandler>, val replyTo: Messenger) {
		fun onEngineError(error: EngineError, extras: Bundle? = null) {
			try {
				replyTo.send(Message.obtain().apply {
					what = MSG_ENGINE_ERROR
					data.putString(KEY_ENGINE_ERROR, error.name)
					if (extras != null && !extras.isEmpty) {
						data.putAll(extras)
					}
				})
			} catch (e: RemoteException) {
				Log.e(TAG, "Unable to send engine error", e)
			}
		}

		fun onEngineStatusUpdate(status: EngineStatus, extras: Bundle? = null) {
			try {
				replyTo.send(Message.obtain().apply {
					what = MSG_ENGINE_STATUS_UPDATE
					data.putString(KEY_ENGINE_STATUS, status.name)
					if (extras != null && !extras.isEmpty) {
						data.putAll(extras)
					}
				})
			} catch (e: RemoteException) {
				Log.e(TAG, "Unable to send engine status update", e)
			}

			if (status == EngineStatus.DESTROYED) {
				val handler = handlerRef.get() ?: return
				if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && handler.viewHost != null) {
					Log.d(TAG, "Releasing SurfaceControlViewHost")
					handler.viewHost?.release()
					handler.viewHost = null
				}
			}
		}

		fun onEngineRestartRequested() {
			try {
				replyTo.send(Message.obtain(null, MSG_ENGINE_RESTART_REQUESTED))
			} catch (e: RemoteException) {
				Log.w(TAG, "Unable to send restart request", e)
			}
		}
	}

	/**
	 * Handler of incoming messages from remote clients.
	 */
	private class IncomingHandler(private val serviceRef: WeakReference<GodotService>) : Handler() {

		var viewHost: SurfaceControlViewHost? = null

		override fun handleMessage(msg: Message) {
			val service = serviceRef.get() ?: return

			Log.d(TAG, "HandleMessage: $msg")

			if (msg.replyTo == null) {
				// Messages for this handler must have a valid 'replyTo' field
				super.handleMessage(msg)
				return
			}

			try {
				val serviceListener = service.listener
				if (serviceListener == null) {
					service.listener = RemoteListener(WeakReference(this), msg.replyTo)
				} else if (serviceListener.replyTo != msg.replyTo) {
					Log.e(TAG, "Engine is already bound to another client")
					msg.replyTo.send(Message.obtain().apply {
						what = MSG_ENGINE_ERROR
						data.putString(KEY_ENGINE_ERROR, EngineError.ALREADY_BOUND.name)
					})
					return
				}

				when (msg.what) {
					MSG_INIT_ENGINE -> service.initEngine(msg.data.getStringArray(KEY_COMMAND_LINE_PARAMETERS))

					MSG_START_ENGINE -> service.startEngine()

					MSG_STOP_ENGINE -> service.stopEngine()

					MSG_DESTROY_ENGINE -> service.destroyEngine()

					MSG_WRAP_ENGINE_WITH_SCVH -> {
						if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) {
							Log.e(TAG, "SDK version is less than the minimum required (${Build.VERSION_CODES.R})")
							service.listener?.onEngineError(EngineError.SCVH_CREATION_FAILED)
							return
						}

						var currentViewHost = viewHost
						if (currentViewHost != null) {
							Log.i(TAG, "Attached Godot engine to SurfaceControlViewHost")
							service.listener?.onEngineStatusUpdate(
								EngineStatus.SCVH_CREATED,
								bundleOf(KEY_SURFACE_PACKAGE to currentViewHost.surfacePackage)
							)
							return
						}

						val msgData = msg.data
						if (msgData.isEmpty) {
							Log.e(TAG, "Invalid message data from binding client.. Aborting")
							service.listener?.onEngineError(EngineError.SCVH_CREATION_FAILED)
							return
						}

						val godotContainerLayout = service.godot.containerLayout
						if (godotContainerLayout == null) {
							Log.e(TAG, "Invalid godot layout.. Aborting")
							service.listener?.onEngineError(EngineError.SCVH_CREATION_FAILED)
							return
						}

						val hostToken = msgData.getBinder(KEY_HOST_TOKEN)
						val width = msgData.getInt(KEY_WIDTH)
						val height = msgData.getInt(KEY_HEIGHT)
						val displayId = msgData.getInt(KEY_DISPLAY_ID)
						val display = service.getSystemService(DisplayManager::class.java)
							.getDisplay(displayId)

						Log.d(TAG, "Setting up SurfaceControlViewHost")
						currentViewHost = SurfaceControlViewHost(service, display, hostToken).apply {
							setView(godotContainerLayout, width, height)

							Log.i(TAG, "Attached Godot engine to SurfaceControlViewHost")
							service.listener?.onEngineStatusUpdate(
								EngineStatus.SCVH_CREATED,
								bundleOf(KEY_SURFACE_PACKAGE to surfacePackage)
							)
						}
						viewHost = currentViewHost
					}

					else -> super.handleMessage(msg)
				}
			} catch (e: RemoteException) {
				Log.e(TAG, "Unable to handle message", e)
			}
		}
	}

	private inner class GodotServiceHost : GodotHost {
		override fun getActivity() = null
		override fun getGodot() = this@GodotService.godot
		override fun getCommandLine() = commandLineParams

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
				listener?.onEngineRestartRequested()
			}
		}
	}

	private val commandLineParams = ArrayList<String>()
	private val handler = IncomingHandler(WeakReference(this))
	private val messenger = Messenger(handler)
	private val godotHost = GodotServiceHost()

	private val godot: Godot by lazy { Godot.getInstance(applicationContext) }
	private var listener: RemoteListener? = null

	override fun onCreate() {
		Log.d(TAG, "OnCreate")
		super.onCreate()
	}

	override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
		// Dispatch the start payload to the incoming handler
		Log.d(TAG, "Processing start command $intent")
		val msg = intent?.getParcelableExtra<Message>(EXTRA_MSG_PAYLOAD)
		if (msg != null) {
			handler.sendMessage(msg)
		}
		return START_NOT_STICKY
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
			if (!godot.initEngine(godotHost, godotHost.commandLine, godotHost.getHostPlugins(godot))) {
				throw IllegalStateException("Unable to initialize Godot engine layer")
			}

			if (godot.onInitRenderView(godotHost) == null) {
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
		destroyEngine()
	}

	private fun forceQuitService() {
		Log.d(TAG, "Force quitting service")
		stopSelf()
		Process.killProcess(Process.myPid())
		Runtime.getRuntime().exit(0)
	}

	override fun onBind(intent: Intent?): IBinder? = messenger.binder

	override fun onUnbind(intent: Intent?): Boolean {
		stopEngine()
		return false
	}

	private fun initEngine(args: Array<String>?): FrameLayout? {
		if (!godot.isInitialized()) {
			if (!args.isNullOrEmpty()) {
				updateCommandLineParams(args.asList())
			}

			if (!performEngineInitialization()) {
				Log.e(TAG, "Unable to initialize Godot engine")
				return null
			} else {
				Log.i(TAG, "Engine initialization complete!")
			}
		}
		val godotContainerLayout = godot.containerLayout
		if (godotContainerLayout == null) {
			listener?.onEngineError(EngineError.INIT_FAILED)
		} else {
			Log.i(TAG, "Initialized Godot engine")
			listener?.onEngineStatusUpdate(EngineStatus.INITIALIZED)
		}

		return godotContainerLayout
	}

	private fun startEngine() {
		if (!godot.isInitialized()) {
			Log.e(TAG, "Attempting to start uninitialized Godot engine instance")
			return
		}

		Log.d(TAG, "Starting Godot engine")
		godot.onStart(godotHost)
		godot.onResume(godotHost)

		listener?.onEngineStatusUpdate(EngineStatus.STARTED)
	}

	private fun stopEngine() {
		if (!godot.isInitialized()) {
			Log.e(TAG, "Attempting to stop uninitialized Godot engine instance")
			return
		}

		Log.d(TAG, "Stopping Godot engine")
		godot.onPause(godotHost)
		godot.onStop(godotHost)

		listener?.onEngineStatusUpdate(EngineStatus.STOPPED)
	}

	private fun destroyEngine() {
		if (!godot.isInitialized()) {
			return
		}

		godot.onDestroy(godotHost)

		listener?.onEngineStatusUpdate(EngineStatus.DESTROYED)
		listener = null
		forceQuitService()
	}

}
