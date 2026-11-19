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
import androidx.xr.runtime.math.FloatSize3d
import androidx.xr.runtime.math.IntSize2d
import androidx.xr.runtime.math.Pose
import androidx.xr.runtime.math.Quaternion
import androidx.xr.runtime.math.Vector3
import androidx.xr.scenecore.ActivityPanelEntity
import androidx.xr.scenecore.MovableComponent
import androidx.xr.scenecore.PanelEntity
import androidx.xr.scenecore.ResizableComponent
import androidx.xr.scenecore.ResizeEvent
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

		private val DEFAULT_ACTIVITY_PANEL_PIXEL_DIMENSIONS = IntSize2d(250, 250)
		private val MAIN_WINDOW_PANEL_MIN_SIZE = FloatSize3d(0.45f, 0.25f, 0.0f)
	}

	private val handler = Handler()
	private val windowIdToPanelEntity = HashMap<Int, ActivityPanelEntity>()

	private val pendingNewInstanceLaunches = LinkedList<Triple<EditorWindowInfo, Intent, ActivityOptions?>>()
	private val spatialCapabilitiesChangedListener = Consumer<Set<SpatialCapability>> {
		if (!isInFullSpace()) {
			// TODO: There's an OS bug that prevents properly resuming home space when launching new Activity
			//  instances while in full space. As a workaround, we launch those instances in home space.
			//  This logic should be removed when the OS bug is fixed.
			// Launch any queued spatial containers.
			Log.v(TAG, "Launching pending new instances..")
			for ((first, second, third) in pendingNewInstanceLaunches) {
				dispatchNewInstance(first, second, third)
			}
			pendingNewInstanceLaunches.clear()
		}
	}

	private val session: Session? by lazy {
		val result = Session.create(this)
		if (result is SessionCreateSuccess) {
			result.session.scene.apply{
				addSpatialCapabilitiesChangedListener(spatialCapabilitiesChangedListener)
				mainPanelEntity.addComponent(MovableComponent.createSystemMovable(result.session))
				mainPanelEntity.addComponent(ResizableComponent.create(result.session, MAIN_WINDOW_PANEL_MIN_SIZE) { resizePanelEntity(it) })
			}
			return@lazy result.session
		} else {
			return@lazy null
		}
	}

	private val requestHomeSpaceRunnable: Runnable = Runnable {
		handler.removeCallbacks(requestHomeSpaceRunnable)
		if (isInFullSpace()) {
			Log.v(TAG, "Requesting home space")
			session?.scene?.requestHomeSpace()
		}
	}

	private val requestFullSpaceRunnable: Runnable = Runnable {
		handler.removeCallbacks(requestFullSpaceRunnable)
		if (!isInFullSpace()) {
			Log.v(TAG, "Requesting full space")
			session?.scene?.requestFullSpace()
		}
	}

	private fun isInFullSpace(): Boolean {
		val scene = session?.scene ?: return false
		return scene.spatialCapabilities.contains(SpatialCapability.EMBED_ACTIVITY)
	}

	private fun resizePanelEntity(resizeEvent: ResizeEvent) {
		if (resizeEvent.entity is PanelEntity && resizeEvent.resizeState == ResizeEvent.ResizeState.END) {
			Log.v(TAG, "Resizing panel entity ${resizeEvent.entity} to ${resizeEvent.newSize}")
			(resizeEvent.entity as PanelEntity).size = resizeEvent.newSize.to2d()
		}
	}

	private fun startSpatialContainerActivity(editorWindowInfo: EditorWindowInfo, newInstance: Intent): Boolean {
		if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
			val scene = session?.scene ?: return false

			val mainPanelWidth = scene.mainPanelEntity.size.width
			val mainPanelHeight = scene.mainPanelEntity.size.height
			val spatialContainerBounds = BaseGodotSpatialContainerGame.getSpatialContainerBounds()

			val translationX = if (editorWindowInfo == SPATIAL_CONTAINER_RUN_GAME_INFO_0) {
				// The first spatial container is positioned to the right of the main panel.
				mainPanelWidth + spatialContainerBounds.first / 2
			} else {
				// The second spatial container is positioned to the left of the main panel.
				-(mainPanelWidth + spatialContainerBounds.first / 2)
			}
			val translationY = -(mainPanelHeight / 2)

			val activityPanel = ActivityPanelEntity.create(
				session!!,
				DEFAULT_ACTIVITY_PANEL_PIXEL_DIMENSIONS,
				"Godot Spatial Container Panel for $newInstance",
				Pose(Vector3(translationX, translationY, 0.25f), Quaternion.Identity),
				scene.activitySpace
			)

			windowIdToPanelEntity[editorWindowInfo.windowId] = activityPanel

			// We remove the 'NEW_TASK' flag as launching in a new task prevents embedding.
			newInstance.removeFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
			Log.v(TAG, "Launching spatial container: $newInstance")
			activityPanel.startActivity(newInstance)
			return true
		}
		return false
	}

	private fun exitFullSpace() {
		handler.postDelayed(requestHomeSpaceRunnable, 50L)
	}

	private fun enterFullSpace() {
		handler.postDelayed(requestFullSpaceRunnable, 50L)
	}

	override fun getExcludedPermissions(): MutableSet<String> {
		val excludedPermissions = super.getExcludedPermissions()
		if (!packageManager.hasSystemFeature("android.hardware.xr.input.eye_tracking")) {
			// No need to ask for this permission if the device doesn't support eye tracking.
			excludedPermissions.add("android.permission.EYE_TRACKING_FINE")
		}
		return excludedPermissions
	}

	override fun getXRRuntimePermissions(): MutableSet<String> {
		val xrRuntimePermissions = super.getXRRuntimePermissions()

		if (packageManager.hasSystemFeature("android.hardware.xr.input.eye_tracking")) {
			// Only ask for this permission at runtime if the device supports eye tracking.
			xrRuntimePermissions.add("android.permission.EYE_TRACKING_FINE")
		}
		xrRuntimePermissions.add("android.permission.HAND_TRACKING")

		return xrRuntimePermissions
	}

	override fun dispatchNewInstance(
		editorWindowInfo: EditorWindowInfo,
		newInstance: Intent,
		activityOptions: ActivityOptions?
	) {
		// TODO: There's an OS bug that prevents properly resuming home space when launching new Activity
		//  instances while in full space. As a workaround, we launch those instances in home space.
		//  This logic should be removed when the OS bug is fixed.
		if (!isInFullSpace()) {
			if (isSpatialContainerRunGameInfo(editorWindowInfo) && startSpatialContainerActivity(editorWindowInfo, newInstance)) {
				return
			}
			super.dispatchNewInstance(editorWindowInfo, newInstance, activityOptions)
		} else {
			pendingNewInstanceLaunches.push(Triple(editorWindowInfo, newInstance, activityOptions))
			exitFullSpace()
		}
	}

	override fun onEditorConnected(editorId: Int) {
		super.onEditorConnected(editorId)
		if (isSpatialContainerRunGameInfoWindowId(editorId)) {
			enterFullSpace()
		}
	}

	override fun onEditorDisconnected(editorId: Int) {
		if (isSpatialContainerRunGameInfoWindowId(editorId)) {
			// Unparent the ActivityPanelEntity.
			windowIdToPanelEntity.remove(editorId)?.apply {
				Log.d(TAG, "Removing entity $this from the scene graph.")
				parent = null
			}

			val spatialContainersRunning = windowIdToPanelEntity.isNotEmpty()
			if (!spatialContainersRunning) {
				// No spatial container is running, time to exit FSM.
				exitFullSpace()
			}
		}
		super.onEditorDisconnected(editorId)
	}

	override fun getCommandLine(): MutableList<String> {
		val updatedArgs = super.getCommandLine()
		if (getEditorWindowInfo() == EDITOR_MAIN_INFO && packageManager.hasSystemFeature("android.hardware.xr.input.eye_tracking")) {
			if (!updatedArgs.contains("--accessibility ")) {
				Log.v(TAG, "Enabling accessibility for Android XR to add support for eye aiming.")
				updatedArgs.add("--accessibility ")
				updatedArgs.add("always")
			}
		}
		return updatedArgs
	}
}
