/**************************************************************************/
/*  FileAccessHandler.kt                                                  */
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
		internal const val INVALID_FILE_ID = 0
		private const val STARTING_FILE_ID = 1

		internal fun fileExists(context: Context, storageScopeIdentifier: StorageScope.Identifier, path: String?): Boolean {
			val storageScope = storageScopeIdentifier.identifyStorageScope(path)
			if (storageScope == StorageScope.UNKNOWN) {
				return false
			}

			return try {
				DataAccess.fileExists(storageScope, context, path!!)
			} catch (e: SecurityException) {
				false
			}
		}

		internal fun removeFile(context: Context, storageScopeIdentifier: StorageScope.Identifier, path: String?): Boolean {
			val storageScope = storageScopeIdentifier.identifyStorageScope(path)
			if (storageScope == StorageScope.UNKNOWN) {
				return false
			}

			return try {
				DataAccess.removeFile(storageScope, context, path!!)
			} catch (e: Exception) {
				false
			}
		}

		internal fun renameFile(context: Context, storageScopeIdentifier: StorageScope.Identifier, from: String?, to: String?): Boolean {
			val storageScope = storageScopeIdentifier.identifyStorageScope(from)
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

	private val storageScopeIdentifier = StorageScope.Identifier(context)
	private val files = SparseArray<DataAccess>()
	private var lastFileId = STARTING_FILE_ID

	private fun hasFileId(fileId: Int) = files.indexOfKey(fileId) >= 0

	fun fileOpen(path: String?, modeFlags: Int): Int {
		val accessFlag = FileAccessFlags.fromNativeModeFlags(modeFlags) ?: return INVALID_FILE_ID
		return fileOpen(path, accessFlag)
	}

	internal fun fileOpen(path: String?, accessFlag: FileAccessFlags): Int {
		val storageScope = storageScopeIdentifier.identifyStorageScope(path)
		if (storageScope == StorageScope.UNKNOWN) {
			return INVALID_FILE_ID
		}

		try {
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

	fun fileExists(path: String?) = Companion.fileExists(context, storageScopeIdentifier, path)

	fun fileLastModified(filepath: String?): Long {
		val storageScope = storageScopeIdentifier.identifyStorageScope(filepath)
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

	fun setFileEof(fileId: Int, eof: Boolean) {
		val file = files[fileId] ?: return
		file.endOfFile = eof
	}

	fun fileClose(fileId: Int) {
		if (hasFileId(fileId)) {
			files[fileId].close()
			files.remove(fileId)
		}
	}
}
