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

internal class SAFData(context: Context, path: String, accessFlag: FileAccessFlags) :
	DataAccess.FileChannelDataAccess(path) {

	companion object {
		private val TAG = FileData::class.java.simpleName

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

			if (uri.fragment == null) {
				return uri
			}

			// For directory format: content://treeUri#relative/path/to/file
			val treeUri = uri.buildUpon().fragment(null).build()
			val relativePath = uri.fragment!!

			val rootDir = DocumentFile.fromTreeUri(context, treeUri)
				?: throw IllegalStateException("Unable to resolve tree URI: $treeUri")

			val parts = relativePath.split('/')
			val filename = parts.last()
			val folderParts = parts.dropLast(1)
			rootDir.exists()

			var current: DocumentFile? = rootDir

			for (folder in folderParts) {
				var next = current?.findFile(folder)

				if (next == null) {
					if (isWriteMode(accessFlag)) {
						next = current!!.createDirectory(folder)
							?: throw IllegalStateException("Failed to create directory: $folder")
					} else {
						throw IllegalStateException("Directory not found: $folder")
					}
				}

				current = next
			}

			var file = current?.findFile(filename)

			if (file == null) {
				if (isWriteMode(accessFlag)) {
					file = current!!.createFile("*/*", filename)
						?: throw IllegalStateException("Failed to create file: $filename")
				} else {
					throw IllegalStateException("File does not exist: $relativePath")
				}
			}

			return file.uri
		}

		private fun isWriteMode(accessFlag: FileAccessFlags): Boolean {
			return accessFlag == FileAccessFlags.WRITE ||
				accessFlag == FileAccessFlags.READ_WRITE ||
				accessFlag == FileAccessFlags.WRITE_READ
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
