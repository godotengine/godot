/**************************************************************************/
/*  AndroidRuntimePlugin.kt                                               */
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

package org.godotengine.godot.plugin

import android.content.Intent
import android.util.Log
import androidx.core.net.toUri

import org.godotengine.godot.Godot
import org.godotengine.godot.variant.Callable

/**
 * Built-in Godot Android plugin used to provide access to the Android runtime capabilities.
 *
 * @see <a href="https://docs.godotengine.org/en/latest/tutorials/platform/android/javaclasswrapper_and_androidruntimeplugin.html">Integrating with Android APIs</a>
 */
class AndroidRuntimePlugin(godot: Godot) : GodotPlugin(godot) {
	private val TAG = AndroidRuntimePlugin::class.java.simpleName

	override fun getPluginName() = "AndroidRuntime"

	/**
	 * Provides access to the application [android.content.Context] to GDScript
	 */
	@UsedByGodot
	fun getApplicationContext() = activity?.applicationContext

	/**
	 * Provides access to the host [android.app.Activity] to GDScript
	 */
	@UsedByGodot
	public override fun getActivity() = super.getActivity()

	/**
	 * Utility method used to create [Runnable] from Godot [Callable].
	 */
	@UsedByGodot
	fun createRunnableFromGodotCallable(godotCallable: Callable): Runnable {
		return Runnable { godotCallable.call() }
	}

	/**
	 * Utility method used to create [java.util.concurrent.Callable] from Godot [Callable].
	 */
	@UsedByGodot
	fun createCallableFromGodotCallable(godotCallable: Callable): java.util.concurrent.Callable<Any> {
		return java.util.concurrent.Callable { godotCallable.call() }
	}

	/**
	 * Helper method to take/release persistable URI permission.
	 */
	@UsedByGodot
	fun updatePersistableUriPermission(uriString: String, persist: Boolean): Boolean {
		try {
			val uri = uriString.toUri()
			val contentResolver = context.contentResolver
			if (persist) {
				contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
			} else {
				contentResolver.releasePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
			}
		} catch (e: RuntimeException) {
			Log.d(TAG, "Error updating persistable permission: ", e)
			return false
		}
		return true
	}
}
