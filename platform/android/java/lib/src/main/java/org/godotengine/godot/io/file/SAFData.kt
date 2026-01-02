/**************************************************************************/
/*  SAFData.kt                                                            */
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
import android.net.Uri
import android.provider.DocumentsContract
import android.util.Log
import androidx.core.net.toUri
import androidx.documentfile.provider.DocumentFile
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.channels.FileChannel

/**
 * Implementation of [DataAccess] which handles file access via a content URI obtained using the Android
 * Storage Access Framework (SAF).
 */
internal class SAFData(context: Context, path: String, accessFlag: FileAccessFlags) :
	DataAccess.FileChannelDataAccess(path) {

	companion object {
		private val TAG = SAFData::class.java.simpleName

		fun fileExists(context: Context, path: String): Boolean {
			return try {
				val uri = resolvePath(context, path, FileAccessFlags.READ)
				context.contentResolver.query(uri, arrayOf(DocumentsContract.Document.COLUMN_DISPLAY_NAME), null, null, null)
					?.use { cursor -> cursor.moveToFirst() } == true
			} catch (e: Exception) {
				Log.d(TAG, "Error checking file existence", e)
				false
			}
		}

		fun fileLastModified(context: Context, path: String): Long {
			return try {
				val uri = resolvePath(context, path, FileAccessFlags.READ)
				val projection = arrayOf(DocumentsContract.Document.COLUMN_LAST_MODIFIED)
				context.contentResolver.query(uri, projection, null, null, null)?.use { cursor ->
					if (cursor.moveToFirst()) {
						val index = cursor.getColumnIndex(DocumentsContract.Document.COLUMN_LAST_MODIFIED)
						if (index != -1) {
							return cursor.getLong(index) / 1000L
						}
					}
				}
				0L
			} catch (e: Exception) {
				Log.d(TAG, "Error reading last modified for", e)
				0L
			}
		}

		fun fileSize(context: Context, path: String): Long {
			return try {
				val uri = resolvePath(context, path, FileAccessFlags.READ)
				val projection = arrayOf(DocumentsContract.Document.COLUMN_SIZE)
				context.contentResolver.query(uri, projection, null, null, null)?.use { cursor ->
					if (cursor.moveToFirst()) {
						val index = cursor.getColumnIndex(DocumentsContract.Document.COLUMN_SIZE)
						if (index != -1) {
							return cursor.getLong(index)
						}
					}
				}
				-1L
			} catch (e: Exception) {
				Log.d(TAG, "Error reading file size", e)
				-1L
			}
		}

		fun delete(context: Context, path: String): Boolean {
			return try {
				val uri = resolvePath(context, path, FileAccessFlags.READ)
				DocumentsContract.deleteDocument(context.contentResolver, uri)
			} catch (e: Exception) {
				Log.d(TAG, "Error deleting file", e)
				false
			}
		}

		fun rename(context: Context, from: String, to: String): Boolean {
			// See https://github.com/godotengine/godot/pull/112215#discussion_r2479311235
			return false
		}

		private fun resolvePath(context: Context, path: String, accessFlag: FileAccessFlags): Uri {
			val uri = path.toUri()
			val fragment = uri.fragment

			if (fragment == null) {
				return uri
			}

			// For directory format: content://treeUri#relative/path/to/file
			val treeUri = uri.buildUpon().fragment(null).build()
			val relativePath = fragment

			val rootDir = DocumentFile.fromTreeUri(context, treeUri)
				?: throw IllegalStateException("Unable to resolve tree URI: $treeUri")

			val parts = relativePath.split('/')
			val filename = parts.last()
			val folderParts = parts.dropLast(1)

			var current: DocumentFile? = rootDir

			val isWriteMode = when (accessFlag) {
				FileAccessFlags.WRITE,
				FileAccessFlags.READ_WRITE,
				FileAccessFlags.WRITE_READ -> true
				else -> false
			}

			for (folder in folderParts) {
				var next = current?.findFile(folder)

				if (next == null) {
					if (isWriteMode) {
						next = current?.createDirectory(folder)
							?: throw IllegalStateException("Failed to create directory: $folder")
					} else {
						throw IllegalStateException("Directory not found: $folder")
					}
				}

				current = next
			}

			var file = current?.findFile(filename)

			if (file == null) {
				if (isWriteMode) {
					file = current?.createFile("*/*", filename)
						?: throw IllegalStateException("Failed to create file: $filename")
				} else {
					throw IllegalStateException("File does not exist: $relativePath")
				}
			}

			return file.uri
		}
	}

	override val fileChannel: FileChannel
	init {
		val uri = resolvePath(context, path, accessFlag)
		val parcelFileDescriptor = context.contentResolver.openFileDescriptor(uri, accessFlag.getMode())
			?: throw IllegalStateException("Unable to access file descriptor")
		fileChannel = if (accessFlag == FileAccessFlags.READ) {
			FileInputStream(parcelFileDescriptor.fileDescriptor).channel
		} else {
			FileOutputStream(parcelFileDescriptor.fileDescriptor).channel
		}

		if (accessFlag.shouldTruncate()) {
			fileChannel.truncate(0)
		}
	}
}
