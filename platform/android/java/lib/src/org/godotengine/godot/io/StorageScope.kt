/**************************************************************************/
/*  StorageScope.kt                                                       */
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

package org.godotengine.godot.io

import android.content.Context
import android.os.Build
import android.os.Environment
import java.io.File
import org.godotengine.godot.GodotLib

/**
 * Represents the different storage scopes.
 */
internal enum class StorageScope {
	/**
	 * Covers the 'assets' directory
	 */
	ASSETS,

	/**
	 * Covers internal and external directories accessible to the app without restrictions.
	 */
	APP,

	/**
	 * Covers shared directories (from Android 10 and higher).
	 */
	SHARED,

	/**
	 * Everything else..
	 */
	UNKNOWN;

	class Identifier(context: Context) {

		companion object {
			internal const val ASSETS_PREFIX = "assets://"
		}

		private val internalAppDir: String? = context.filesDir.canonicalPath
		private val internalCacheDir: String? = context.cacheDir.canonicalPath
		private val externalAppDir: String? = context.getExternalFilesDir(null)?.canonicalPath
		private val sharedDir : String? = Environment.getExternalStorageDirectory().canonicalPath
		private val downloadsSharedDir: String? = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).canonicalPath
		private val documentsSharedDir: String? = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS).canonicalPath

		/**
		 * Determines which [StorageScope] the given path falls under.
		 */
		fun identifyStorageScope(path: String?): StorageScope {
			if (path == null) {
				return UNKNOWN
			}

			if (path.startsWith(ASSETS_PREFIX)) {
				return ASSETS
			}

			var pathFile = File(path)
			if (!pathFile.isAbsolute) {
				pathFile = File(GodotLib.getProjectResourceDir(), path)
				if (!pathFile.isAbsolute) {
					return UNKNOWN
				}
			}

			// If we have 'All Files Access' permission, we can access all directories without
			// restriction.
			if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R
				&& Environment.isExternalStorageManager()) {
				return APP
			}

			val canonicalPathFile = pathFile.canonicalPath

			if (internalAppDir != null && canonicalPathFile.startsWith(internalAppDir)) {
				return APP
			}

			if (internalCacheDir != null && canonicalPathFile.startsWith(internalCacheDir)) {
				return APP
			}

			if (externalAppDir != null && canonicalPathFile.startsWith(externalAppDir)) {
				return APP
			}

			val rootDir: String? = System.getenv("ANDROID_ROOT")
			if (rootDir != null && canonicalPathFile.startsWith(rootDir)) {
				return APP
			}

			if (sharedDir != null && canonicalPathFile.startsWith(sharedDir)) {
				if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) {
					// Before R, apps had access to shared storage so long as they have the right
					// permissions (and flag on Q).
					return APP
				}

				// Post R, access is limited based on the target destination
				// 'Downloads' and 'Documents' are still accessible
				if ((downloadsSharedDir != null && canonicalPathFile.startsWith(downloadsSharedDir))
					|| (documentsSharedDir != null && canonicalPathFile.startsWith(documentsSharedDir))) {
					return APP
				}

				return SHARED
			}

			return UNKNOWN
		}
	}
}
