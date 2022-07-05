/*************************************************************************/
/*  FileAccessHandler.kt                                                 */
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

package org.godotengine.godot.io.file

import android.content.Context
import android.util.Log
import android.util.SparseArray
import org.godotengine.godot.io.StorageScope
import java.io.FileNotFoundException
import java.nio.ByteBuffer

/**
 * Handles regular and media store file access and interactions.
 */
class FileAccessHandler(val context: Context) {

	companion object {
		private val TAG = FileAccessHandler::class.java.simpleName

		private const val FILE_NOT_FOUND_ERROR_ID = -1
		private const val INVALID_FILE_ID = 0
		private const val STARTING_FILE_ID = 1

		fun fileExists(context: Context, path: String?): Boolean {
			val storageScope = StorageScope.getStorageScope(context, path)
			if (storageScope == StorageScope.UNKNOWN) {
				return false
			}

			return try {
				DataAccess.fileExists(storageScope, context, path!!)
			} catch (e: SecurityException) {
				false
			}
		}

		fun removeFile(context: Context, path: String?): Boolean {
			val storageScope = StorageScope.getStorageScope(context, path)
			if (storageScope == StorageScope.UNKNOWN) {
				return false
			}

			return try {
				DataAccess.removeFile(storageScope, context, path!!)
			} catch (e: Exception) {
				false
			}
		}

		fun renameFile(context: Context, from: String?, to: String?): Boolean {
			val storageScope = StorageScope.getStorageScope(context, from)
			if (storageScope == StorageScope.UNKNOWN) {
				return false
			}

			return try {
				DataAccess.renameFile(storageScope, context, from!!, to!!)
			} catch (e: Exception) {
				false
			}
		}
	}

	private val files = SparseArray<DataAccess>()
	private var lastFileId = STARTING_FILE_ID

	private fun hasFileId(fileId: Int) = files.indexOfKey(fileId) >= 0

	fun fileOpen(path: String?, modeFlags: Int): Int {
		val storageScope = StorageScope.getStorageScope(context, path)
		if (storageScope == StorageScope.UNKNOWN) {
			return INVALID_FILE_ID
		}

		try {
			val accessFlag = FileAccessFlags.fromNativeModeFlags(modeFlags) ?: return INVALID_FILE_ID
			val dataAccess = DataAccess.generateDataAccess(storageScope, context, path!!, accessFlag) ?: return INVALID_FILE_ID

			files.put(++lastFileId, dataAccess)
			return lastFileId
		} catch (e: FileNotFoundException) {
			return FILE_NOT_FOUND_ERROR_ID
		} catch (e: Exception) {
			Log.w(TAG, "Error while opening $path", e)
			return INVALID_FILE_ID
		}
	}

	fun fileGetSize(fileId: Int): Long {
		if (!hasFileId(fileId)) {
			return 0L
		}

		return files[fileId].size()
	}

	fun fileSeek(fileId: Int, position: Long) {
		if (!hasFileId(fileId)) {
			return
		}

		files[fileId].seek(position)
	}

	fun fileSeekFromEnd(fileId: Int, position: Long) {
		if (!hasFileId(fileId)) {
			return
		}

		files[fileId].seekFromEnd(position)
	}

	fun fileRead(fileId: Int, byteBuffer: ByteBuffer?): Int {
		if (!hasFileId(fileId) || byteBuffer == null) {
			return 0
		}

		return files[fileId].read(byteBuffer)
	}

	fun fileWrite(fileId: Int, byteBuffer: ByteBuffer?) {
		if (!hasFileId(fileId) || byteBuffer == null) {
			return
		}

		files[fileId].write(byteBuffer)
	}

	fun fileFlush(fileId: Int) {
		if (!hasFileId(fileId)) {
			return
		}

		files[fileId].flush()
	}

	fun fileExists(path: String?) = Companion.fileExists(context, path)

	fun fileLastModified(filepath: String?): Long {
		val storageScope = StorageScope.getStorageScope(context, filepath)
		if (storageScope == StorageScope.UNKNOWN) {
			return 0L
		}

		return try {
			DataAccess.fileLastModified(storageScope, context, filepath!!)
		} catch (e: SecurityException) {
			0L
		}
	}

	fun fileGetPosition(fileId: Int): Long {
		if (!hasFileId(fileId)) {
			return 0L
		}

		return files[fileId].position()
	}

	fun isFileEof(fileId: Int): Boolean {
		if (!hasFileId(fileId)) {
			return false
		}

		return files[fileId].endOfFile
	}

	fun fileClose(fileId: Int) {
		if (hasFileId(fileId)) {
			files[fileId].close()
			files.remove(fileId)
		}
	}
}
