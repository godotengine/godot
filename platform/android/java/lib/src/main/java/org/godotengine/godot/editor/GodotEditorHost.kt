/**************************************************************************/
/*  GodotEditorHost.kt                                                    */
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

package org.godotengine.godot.editor

import org.godotengine.godot.BuildProvider
import org.godotengine.godot.Godot
import org.godotengine.godot.GodotHost
import org.godotengine.godot.error.Error

/**
 * Implemented by components (e.g. Activity, Fragment) that host the editor build of the [Godot] engine.
 */
interface GodotEditorHost : GodotHost {

	/**
	 * Signs the given Android apk
	 *
	 * @param inputPath Path to the apk that should be signed
	 * @param outputPath Path for the signed output apk; can be the same as inputPath
	 * @param keystorePath Path to the keystore to use for signing the apk
	 * @param keystoreUser Keystore user credential
	 * @param keystorePassword Keystore password credential
	 *
	 * @return [Error.OK] if signing is successful
	 */
	fun signApk(
		inputPath: String,
		outputPath: String,
		keystorePath: String,
		keystoreUser: String,
		keystorePassword: String
	): Error? {
		return Error.ERR_UNAVAILABLE
	}

	/**
	 * Verifies the given Android apk is signed
	 *
	 * @param apkPath Path to the apk that should be verified
	 * @return [Error.OK] if verification was successful
	 */
	fun verifyApk(apkPath: String): Error? {
		return Error.ERR_UNAVAILABLE
	}

	/**
	 * Invoked on the render thread when an editor workspace has been selected.
	 */
	fun onEditorWorkspaceSelected(workspace: String) {}

	/**
	 * Invoked on the render thread to request showing the editor game dock.
	 */
	fun showEditorGameDock() {}

	/**
	 * Triggered when the editor's distraction-free mode changes.
	 */
	fun onDistractionFreeModeChanged(enabled: Boolean) {}

	/**
	 * Gets the build provider, if available.
	 *
	 * @return the build provider, if available; otherwise, null.
	 */
	fun getBuildProvider(): BuildProvider? {
		return null
	}
}
