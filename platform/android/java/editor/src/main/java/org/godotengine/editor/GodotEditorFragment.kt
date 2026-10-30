/**************************************************************************/
/*  GodotEditorFragment.kt                                                */
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

import org.godotengine.godot.BuildProvider
import org.godotengine.godot.GodotFragment
import org.godotengine.godot.editor.GodotEditorHost
import org.godotengine.godot.error.Error

/**
 * Fragment used to host the editor build of the [org.godotengine.godot.Godot] engine.
 */
internal class GodotEditorFragment : GodotFragment(), GodotEditorHost {

	override fun signApk(
		inputPath: String,
		outputPath: String,
		keystorePath: String,
		keystoreUser: String,
		keystorePassword: String
	): Error? {
		val host = parentHost
		if (host is GodotEditorHost) {
			return host.signApk(inputPath, outputPath, keystorePath, keystoreUser, keystorePassword)
		}
		return Error.ERR_UNAVAILABLE
	}

	override fun verifyApk(apkPath: String): Error? {
		val host = parentHost
		if (host is GodotEditorHost) {
			return host.verifyApk(apkPath)
		}
		return Error.ERR_UNAVAILABLE
	}

	override fun onEditorWorkspaceSelected(workspace: String) {
		val host = parentHost
		if (host is GodotEditorHost) {
			return host.onEditorWorkspaceSelected(workspace)
		}
	}

	override fun showEditorGameDock() {
		val host = parentHost
		if (host is GodotEditorHost) {
			return host.showEditorGameDock()
		}
	}

	override fun onDistractionFreeModeChanged(enabled: Boolean) {
		val host = parentHost
		if (host is GodotEditorHost) {
			return host.onDistractionFreeModeChanged(enabled)
		}
	}

	override fun getBuildProvider(): BuildProvider? {
		val host = parentHost
		if (host is GodotEditorHost) {
			return host.getBuildProvider()
		}
		return null
	}
}
