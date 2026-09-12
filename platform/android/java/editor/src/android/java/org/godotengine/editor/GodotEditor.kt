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

import android.app.ActivityOptions
import android.content.Intent
import android.os.Build
import android.os.Handler
import android.util.Log
import androidx.xr.runtime.Session
import androidx.xr.runtime.SessionCreateSuccess
import androidx.xr.runtime.math.IntSize2d
import androidx.xr.runtime.math.Pose
import androidx.xr.runtime.math.Quaternion
import androidx.xr.runtime.math.Vector3
import androidx.xr.scenecore.ActivityPanelEntity
import androidx.xr.scenecore.MovableComponent
import androidx.xr.scenecore.SpatialCapability
import androidx.xr.scenecore.scene
import java.util.LinkedList
import java.util.function.Consumer

/**
 * Primary window of the Godot Editor.
 *
 * This is the implementation of the editor used when running on Android devices.
 */
open class GodotEditor : BaseGodotEditor() {

	companion object {
		private val TAG = GodotEditor::class.java.simpleName
	}

	private val handler = Handler()

	private var pendingFullSpaceRequest = false
	private val pendingSpatialContainerLaunches = LinkedList<Intent>()

	private val spatialCapabilitiesChangedListener = Consumer<Set<SpatialCapability>> {
		if (canEmbedActivity()) {
			pendingFullSpaceRequest = false

			// Launch any queued spatial containers.
			Log.v(TAG, "Launching pending spatial container launches..")
			for (pendingLaunch in pendingSpatialContainerLaunches) {
				startSpatialContainerActivity(pendingLaunch)
			}
			pendingSpatialContainerLaunches.clear()
		} else {
			// Already in HSM, so remove any request to transition.
			handler.removeCallbacks(requestHomeSpaceRunnable)
		}
	}

	private val session: Session? by lazy {
		val result = Session.create(this)
		if (result is SessionCreateSuccess) {
			result.session.scene.apply{
				addSpatialCapabilitiesChangedListener(spatialCapabilitiesChangedListener)
				mainPanelEntity.addComponent(MovableComponent.createSystemMovable(result.session))
			}
			return@lazy result.session
		} else {
			return@lazy null
		}
	}

	private val requestHomeSpaceRunnable: Runnable = Runnable {
		val scene = session?.scene ?: return@Runnable

		handler.removeCallbacks(requestHomeSpaceRunnable)
		if (canEmbedActivity()) {
			scene.requestHomeSpace()
		}
	}

	private fun canEmbedActivity(): Boolean {
		val scene = session?.scene ?: return false
		return scene.spatialCapabilities.contains(SpatialCapability.EMBED_ACTIVITY)
	}

	private fun launchSpatialContainerUsingSceneCore(newInstance: Intent, activityOptions: ActivityOptions?): Boolean {
		if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
			try {
				if (session != null) {
					val scene = session!!.scene

					handler.removeCallbacks(requestHomeSpaceRunnable)
					if (canEmbedActivity()) {
						Log.v(TAG, "Launching spatial container in FSM")
						startSpatialContainerActivity(newInstance)
					} else {
						pendingSpatialContainerLaunches.push(newInstance)
						if (!pendingFullSpaceRequest) {
							Log.v(TAG, "Requesting full space")
							scene.requestFullSpace()
							pendingFullSpaceRequest = true
						}
					}
					return true
				}
			} catch (e: Exception) {
				Log.e(TAG, "Unable to launch spatial container", e)
			}
		}
		return false
	}

	private fun startSpatialContainerActivity(newInstance: Intent) {
		if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
			val scene = session!!.scene

			val activityPanel = ActivityPanelEntity.create(
				session!!,
				IntSize2d(500, 500),
				"Godot Spatial Container Panel for $newInstance",
				Pose(Vector3(0.0f, -scene.mainPanelEntity.size.height, 0.05f), Quaternion.Identity),
				scene.activitySpace
			)
			activityPanel.addComponent(MovableComponent.createSystemMovable(session!!))

			// We remove the 'NEW_TASK' flag as launching in a new task prevents embedding.
			newInstance.removeFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
			activityPanel.startActivity(newInstance)
		}
	}

	private fun exitFullSpaceUsingSceneCore() {
		if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
			handler.postDelayed(requestHomeSpaceRunnable, 250L)

			session?.scene?.activitySpace?.children?.forEach { child ->
				if (child is ActivityPanelEntity) {
					Log.d(TAG, "Removing entity $child from the scene graph.")
					child.parent = null
				}
			}
		}
	}

	override fun getXRRuntimePermissions(): MutableSet<String> {
		val xrRuntimePermissions = super.getXRRuntimePermissions()

		xrRuntimePermissions.add("android.permission.EYE_TRACKING_FINE")
		xrRuntimePermissions.add("android.permission.HAND_TRACKING")

		return xrRuntimePermissions
	}

	override fun dispatchNewInstance(
		editorWindowInfo: EditorWindowInfo,
		newInstance: Intent,
		activityOptions: ActivityOptions?
	) {
		if (isSpatialContainerRunGameInfo(editorWindowInfo) && launchSpatialContainerUsingSceneCore(newInstance, activityOptions)) {
			return
		}
		super.dispatchNewInstance(editorWindowInfo, newInstance, activityOptions)
	}

	override fun onEditorDisconnected(editorId: Int) {
		if (isSpatialContainerRunGameInfoWindowId(editorId)) {
			val spatialContainersRunning =
				(SPATIAL_CONTAINER_RUN_GAME_INFO_0.windowId != editorId && editorMessageDispatcher.hasEditorConnection(SPATIAL_CONTAINER_RUN_GAME_INFO_0)) ||
					(SPATIAL_CONTAINER_RUN_GAME_INFO_1.windowId != editorId && editorMessageDispatcher.hasEditorConnection(SPATIAL_CONTAINER_RUN_GAME_INFO_1))
			if (!spatialContainersRunning) {
				// No spatial container is running, time to exit FSM.
				Log.v(TAG, "Exiting FSM.")
				exitFullSpaceUsingSceneCore()
			}
		}
		super.onEditorDisconnected(editorId)
	}
}
