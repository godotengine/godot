/**************************************************************************/
/*  EditorMessageDispatcher.kt                                            */
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

import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Message
import android.os.Messenger
import android.os.RemoteException
import android.util.Log
import java.util.concurrent.ConcurrentHashMap

/**
 * Used by the [GodotEditor] classes to dispatch messages across processes.
 */
internal class EditorMessageDispatcher(private val editor: GodotEditor) {

	companion object {
		private val TAG = EditorMessageDispatcher::class.java.simpleName

		/**
		 * Extra used to pass the message dispatcher payload through an [Intent]
		 */
		const val EXTRA_MSG_DISPATCHER_PAYLOAD = "message_dispatcher_payload"

		/**
		 * Key used to pass the editor id through a [Bundle]
		 */
		private const val KEY_EDITOR_ID = "editor_id"

		/**
		 * Key used to pass the editor messenger through a [Bundle]
		 */
		private const val KEY_EDITOR_MESSENGER = "editor_messenger"

		/**
		 * Requests the recipient to quit right away.
		 */
		private const val MSG_FORCE_QUIT = 0

		/**
		 * Requests the recipient to store the passed [android.os.Messenger] instance.
		 */
		private const val MSG_REGISTER_MESSENGER = 1
	}

	private val recipientsMessengers = ConcurrentHashMap<Int, Messenger>()

	@SuppressLint("HandlerLeak")
	private val dispatcherHandler = object : Handler() {
		override fun handleMessage(msg: Message) {
			when (msg.what) {
				MSG_FORCE_QUIT -> editor.finish()

				MSG_REGISTER_MESSENGER -> {
					val editorId = msg.arg1
					val messenger = msg.replyTo
					registerMessenger(editorId, messenger)
				}

				else -> super.handleMessage(msg)
			}
		}
	}

	/**
	 * Request the window with the given [editorId] to force quit.
	 */
	fun requestForceQuit(editorId: Int): Boolean {
		val messenger = recipientsMessengers[editorId] ?: return false
		return try {
			Log.v(TAG, "Requesting 'forceQuit' for $editorId")
			val msg = Message.obtain(null, MSG_FORCE_QUIT)
			messenger.send(msg)
			true
		} catch (e: RemoteException) {
			Log.e(TAG, "Error requesting 'forceQuit' to $editorId", e)
			recipientsMessengers.remove(editorId)
			false
		}
	}

	/**
	 * Utility method to register a receiver messenger.
	 */
	private fun registerMessenger(editorId: Int, messenger: Messenger?, messengerDeathCallback: Runnable? = null) {
		try {
			if (messenger == null) {
				Log.w(TAG, "Invalid 'replyTo' payload")
			} else if (messenger.binder.isBinderAlive) {
				messenger.binder.linkToDeath({
					Log.v(TAG, "Removing messenger for $editorId")
					recipientsMessengers.remove(editorId)
					messengerDeathCallback?.run()
				}, 0)
				recipientsMessengers[editorId] = messenger
			}
		} catch (e: RemoteException) {
			Log.e(TAG, "Unable to register messenger from $editorId", e)
			recipientsMessengers.remove(editorId)
		}
	}

	/**
	 * Utility method to register a [Messenger] attached to this handler with a host.
	 *
	 * This is done so that the host can send request to the editor instance attached to this handle.
	 *
	 * Note that this is only done when the editor instance is internal (not exported) to prevent
	 * arbitrary apps from having the ability to send requests.
	 */
	private fun registerSelfTo(pm: PackageManager, host: Messenger?, selfId: Int) {
		try {
			if (host == null || !host.binder.isBinderAlive) {
				Log.v(TAG, "Host is unavailable")
				return
			}

			val activityInfo = pm.getActivityInfo(editor.componentName, 0)
			if (activityInfo.exported) {
				Log.v(TAG, "Not registering self to host as we're exported")
				return
			}

			Log.v(TAG, "Registering self $selfId to host")
			val msg = Message.obtain(null, MSG_REGISTER_MESSENGER)
			msg.arg1 = selfId
			msg.replyTo = Messenger(dispatcherHandler)
			host.send(msg)
		} catch (e: RemoteException) {
			Log.e(TAG, "Unable to register self with host", e)
		}
	}

	/**
	 * Parses the starting intent and retrieve an editor messenger if available
	 */
	fun parseStartIntent(pm: PackageManager, intent: Intent) {
		val messengerBundle = intent.getBundleExtra(EXTRA_MSG_DISPATCHER_PAYLOAD) ?: return

		// Retrieve the sender messenger payload and store it. This can be used to communicate back
		// to the sender.
		val senderId = messengerBundle.getInt(KEY_EDITOR_ID)
		val senderMessenger: Messenger? = messengerBundle.getParcelable(KEY_EDITOR_MESSENGER)
		registerMessenger(senderId, senderMessenger)

		// Register ourselves to the sender so that it can communicate with us.
		registerSelfTo(pm, senderMessenger, editor.getEditorId())
	}

	/**
	 * Returns the payload used by the [EditorMessageDispatcher] class to establish an IPC bridge
	 * across editor instances.
	 */
	fun getMessageDispatcherPayload(): Bundle {
		return Bundle().apply {
			putInt(KEY_EDITOR_ID, editor.getEditorId())
			putParcelable(KEY_EDITOR_MESSENGER, Messenger(dispatcherHandler))
		}
	}
}
