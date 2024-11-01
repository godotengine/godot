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

import org.godotengine.godot.GodotLib
import org.godotengine.godot.utils.isNativeXRDevice

/**
 * Primary window of the Godot Editor.
 *
 * This is the implementation of the editor used when running on HorizonOS devices.
 */
open class GodotEditor : BaseGodotEditor() {

	companion object {
		private val TAG = GodotEditor::class.java.simpleName

		internal val XR_RUN_GAME_INFO = EditorWindowInfo(GodotXRGame::class.java, 1667, ":GodotXRGame")

		internal val USE_SCENE_PERMISSIONS = listOf("com.oculus.permission.USE_SCENE", "horizonos.permission.USE_SCENE")
	}

	override fun getExcludedPermissions(): MutableSet<String> {
		val excludedPermissions = super.getExcludedPermissions()
		// The USE_SCENE permission is requested when the "xr/openxr/enabled" project setting
		// is enabled.
		excludedPermissions.addAll(USE_SCENE_PERMISSIONS)
		return excludedPermissions
	}

	override fun retrieveEditorWindowInfo(args: Array<String>): EditorWindowInfo {
		var hasEditor = false
		var xrModeOn = false

		var i = 0
		while (i < args.size) {
			when (args[i++]) {
				EDITOR_ARG, EDITOR_ARG_SHORT, EDITOR_PROJECT_MANAGER_ARG, EDITOR_PROJECT_MANAGER_ARG_SHORT -> hasEditor = true
				XR_MODE_ARG -> {
					val argValue = args[i++]
					xrModeOn = xrModeOn || ("on" == argValue)
				}
			}
		}

		return if (hasEditor) {
			EDITOR_MAIN_INFO
		} else {
			val openxrEnabled = GodotLib.getGlobal("xr/openxr/enabled").toBoolean()
			if (openxrEnabled && isNativeXRDevice()) {
				XR_RUN_GAME_INFO
			} else {
				RUN_GAME_INFO
			}
		}
	}

	override fun getEditorWindowInfoForInstanceId(instanceId: Int): EditorWindowInfo? {
		return when (instanceId) {
			XR_RUN_GAME_INFO.windowId -> XR_RUN_GAME_INFO
			else -> super.getEditorWindowInfoForInstanceId(instanceId)
		}
	}
}
