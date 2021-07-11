/*************************************************************************/
/*  FileSystemDirectoryAccess.kt                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

package org.godotengine.godot.io.directory

import android.annotation.SuppressLint
import android.content.Context
import android.os.Build
import android.os.storage.StorageManager
import android.util.Log
import android.util.SparseArray
import org.godotengine.godot.io.StorageScope
import org.godotengine.godot.io.directory.DirectoryAccessHandler.Companion.INVALID_DIR_ID
import org.godotengine.godot.io.directory.DirectoryAccessHandler.Companion.STARTING_DIR_ID
import org.godotengine.godot.io.file.FileAccessHandler
import java.io.File

/**
 * Handles directories access with the internal and external filesystem.
 */
internal class FilesystemDirectoryAccess(private val context: Context):
	DirectoryAccessHandler.DirectoryAccess {

	companion object {
		private val TAG = FilesystemDirectoryAccess::class.java.simpleName
	}

	private data class DirData(val dirFile: File, val files: Array<File>, var current: Int = 0)

	private val storageManager = context.getSystemService(Context.STORAGE_SERVICE) as StorageManager
	private var lastDirId = STARTING_DIR_ID
	private val dirs = SparseArray<DirData>()

	private fun inScope(path: String): Boolean {
		// Directory access is available for shared storage on Android 11+
		// On Android 10, access is also available as long as the `requestLegacyExternalStorage`
		// tag is available.
		return StorageScope.getStorageScope(context, path) != StorageScope.UNKNOWN
	}

	override fun hasDirId(dirId: Int) = dirs.indexOfKey(dirId) >= 0

	override fun dirOpen(path: String): Int {
		if (!inScope(path)) {
			Log.w(TAG, "Path $path is not accessible.")
			return INVALID_DIR_ID
		}

		// Check this is a directory.
		val dirFile = File(path)
		if (!dirFile.isDirectory) {
			return INVALID_DIR_ID
		}

		// Get the files in the directory
		val files = dirFile.listFiles()?: return INVALID_DIR_ID

		// Create the data representing this directory
		val dirData = DirData(dirFile, files)

		dirs.put(++lastDirId, dirData)
		return lastDirId
	}

	override fun dirExists(path: String): Boolean {
		if (!inScope(path)) {
			Log.w(TAG, "Path $path is not accessible.")
			return false
		}

		try {
			return File(path).isDirectory
		} catch (e: SecurityException) {
			return false
		}
	}

	override fun fileExists(path: String) = FileAccessHandler.fileExists(context, path)

	override fun dirNext(dirId: Int): String {
		val dirData = dirs[dirId]
		if (dirData.current >= dirData.files.size) {
			dirData.current++
			return ""
		}

		return dirData.files[dirData.current++].name
	}

	override fun dirClose(dirId: Int) {
		dirs.remove(dirId)
	}

	override fun dirIsDir(dirId: Int): Boolean {
		val dirData = dirs[dirId]

		var index = dirData.current
		if (index > 0) {
			index--
		}

		if (index >= dirData.files.size) {
			return false
		}

		return dirData.files[index].isDirectory
	}

	override fun isCurrentHidden(dirId: Int): Boolean {
		val dirData = dirs[dirId]

		var index = dirData.current
		if (index > 0) {
			index--
		}

		if (index >= dirData.files.size) {
			return false
		}

		return dirData.files[index].isHidden
	}

	override fun getDriveCount(): Int {
		return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
			storageManager.storageVolumes.size
		} else {
			0
		}
	}

	override fun getDrive(drive: Int): String {
		if (Build.VERSION.SDK_INT < Build.VERSION_CODES.N) {
			return ""
		}

		if (drive < 0 || drive >= storageManager.storageVolumes.size) {
			return ""
		}

		val storageVolume = storageManager.storageVolumes[drive]
		return storageVolume.getDescription(context)
	}

	override fun makeDir(dir: String): Boolean {
		if (!inScope(dir)) {
			Log.w(TAG, "Directory $dir is not accessible.")
			return false
		}

		try {
			val dirFile = File(dir)
			return dirFile.isDirectory || dirFile.mkdirs()
		} catch (e: SecurityException) {
			return false
		}
	}

	@SuppressLint("UsableSpace")
	override fun getSpaceLeft() = context.getExternalFilesDir(null)?.usableSpace ?: 0L

	override fun rename(from: String, to: String): Boolean {
		if (!inScope(from) || !inScope(to)) {
			Log.w(TAG, "Argument filenames are not accessible:\n" +
					"from: $from\n" +
					"to: $to")
			return false
		}

		return try {
			val fromFile = File(from)
			if (fromFile.isDirectory) {
				fromFile.renameTo(File(to))
			} else {
				FileAccessHandler.renameFile(context, from, to)
			}
		} catch (e: SecurityException) {
			false
		}
	}

	override fun remove(filename: String): Boolean {
		if (!inScope(filename)) {
			Log.w(TAG, "Filename $filename is not accessible.")
			return false
		}

		return try {
			val deleteFile = File(filename)
			if (deleteFile.exists()) {
				if (deleteFile.isDirectory) {
					deleteFile.delete()
				} else {
					FileAccessHandler.removeFile(context, filename)
				}
			} else {
				true
			}
		} catch (e: SecurityException) {
			false
		}
	}
}
