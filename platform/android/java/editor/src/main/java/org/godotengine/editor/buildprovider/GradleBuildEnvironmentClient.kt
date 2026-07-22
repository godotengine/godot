/**************************************************************************/
/*  GradleBuildEnvironmentClient.kt                                       */
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

package org.godotengine.editor.buildprovider

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Message
import android.os.Messenger
import android.os.RemoteException
import android.util.Log
import kotlin.collections.set

private const val MSG_EXECUTE_GRADLE = 1
private const val MSG_COMMAND_RESULT = 2
private const val MSG_COMMAND_OUTPUT = 3
private const val MSG_CANCEL_COMMAND = 4
private const val MSG_CLEAN_PROJECT = 5

internal class GradleBuildEnvironmentClient(private val context: Context) {

	companion object {
		private val TAG = GradleBuildEnvironmentClient::class.java.simpleName
	}

	private var bound: Boolean = false
	private var outgoingMessenger: Messenger? = null
	private val connection = object : ServiceConnection {
		override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
			outgoingMessenger = Messenger(service)
			bound = true

			Log.i(TAG, "Service connected")
			for (callable in connectionCallbacks) {
				callable()
			}
			connectionCallbacks.clear()
			connecting = false
		}

		override fun onServiceDisconnected(name: ComponentName?) {
			outgoingMessenger = null
			bound = false
			Log.i(TAG, "Service disconnected")
		}
	}

	private inner class IncomingHandler: Handler() {
		override fun handleMessage(msg: Message) {
			when (msg.what) {
				MSG_COMMAND_RESULT -> {
					this@GradleBuildEnvironmentClient.receiveCommandResult(msg)
				}
				MSG_COMMAND_OUTPUT -> {
					this@GradleBuildEnvironmentClient.receiveCommandOutput(msg)
				}
				else -> super.handleMessage(msg)
			}
		}
	}
	private val incomingMessenger = Messenger(IncomingHandler())

	private val connectionCallbacks = mutableListOf<() -> Unit>()
	private var connecting = false
	private var executionId = 1000

	private class ExecutionInfo(val outputCallback: (Int, String) -> Unit, val resultCallback: (Int) -> Unit)
	private val executionMap = HashMap<Int, ExecutionInfo>()

	fun connect(callback: () -> Unit): Boolean {
		if (bound) {
			callback()
			return true;
		}
		connectionCallbacks.add(callback)
		if (connecting) {
			return true;
		}
		connecting = true;

		val intent = Intent("org.godotengine.action.BUILD_PROVIDER").apply {
			setPackage("org.godotengine.godot_gradle_build_environment")
		}
		val info = context.packageManager.resolveService(intent, 0)
		if (info == null) {
			connecting = false;
			Log.e(TAG, "Unable to resolve service")
			return false
		}

		val result = context.bindService(intent, connection, Context.BIND_AUTO_CREATE)
		if (!result) {
			Log.e(TAG, "Unable to bind to service")
			connecting = false;
		}
		return result;
	}

	fun disconnect() {
		if (bound) {
			context.unbindService(connection)
			bound = false
		}
	}

	private fun getNextExecutionId(outputCallback: (Int, String) -> Unit, resultCallback: (Int) -> Unit): Int {
		val id = executionId++
		executionMap[id] = ExecutionInfo(outputCallback, resultCallback)
		return id
	}

	fun execute(arguments: Array<String>, projectPath: String, gradleBuildDir: String, outputCallback: (Int, String) -> Unit, resultCallback: (Int) -> Unit): Int {
		if (outgoingMessenger == null) {
			return -1
		}

		val msg: Message = Message.obtain(null, MSG_EXECUTE_GRADLE, getNextExecutionId(outputCallback, resultCallback),0)
		msg.replyTo = incomingMessenger

		val data = Bundle()
		data.putStringArrayList("arguments", ArrayList(arguments.toList()))
		data.putString("project_path", projectPath)
		data.putString("gradle_build_directory", gradleBuildDir)
		msg.data = data

		try {
			outgoingMessenger?.send(msg)
		} catch (e: RemoteException) {
			Log.e(TAG, "Unable to execute Gradle command: gradlew ${arguments.joinToString(" ")}", e)
			e.printStackTrace()
			executionMap.remove(msg.arg1)
			resultCallback(255)
			return -1
		}

		return msg.arg1
	}

	private fun receiveCommandResult(msg: Message) {
		val executionInfo = executionMap.remove(msg.arg1)
		executionInfo?.resultCallback?.invoke(msg.arg2)
	}

	private fun receiveCommandOutput(msg: Message) {
		val data = msg.data
		val line = data.getString("line")

		if (line != null) {
			val executionInfo = executionMap.get(msg.arg1)
			executionInfo?.outputCallback?.invoke(msg.arg2, line)
		}
	}

	fun cancel(jobId: Int) {
		if (outgoingMessenger == null) {
			return
		}

		val msg: Message = Message.obtain(null, MSG_CANCEL_COMMAND, jobId, 0)
		try {
			outgoingMessenger?.send(msg)
		} catch (e: RemoteException) {
			Log.e(TAG, "Unable to cancel Gradle command: ${jobId}", e)
			e.printStackTrace()
		}
	}

	fun cleanProject(projectPath: String, gradleBuildDir: String, resultCallback: (Int) -> Unit) {
		if (outgoingMessenger == null) {
			return
		}

		val emptyOutputCallback: (Int, String) -> Unit = { outputType, line -> }

		val msg: Message = Message.obtain(null, MSG_CLEAN_PROJECT, getNextExecutionId(emptyOutputCallback, resultCallback), 0)
		msg.replyTo = incomingMessenger

		val data = Bundle()
		data.putString("project_path", projectPath)
		data.putString("gradle_build_directory", gradleBuildDir)
		msg.data = data

		try {
			outgoingMessenger?.send(msg)
		} catch (e: RemoteException) {
			Log.e(TAG, "Unable to clean Gradle project", e)
			executionMap.remove(msg.arg1)
			resultCallback(0)
			e.printStackTrace()
		}
	}

}
