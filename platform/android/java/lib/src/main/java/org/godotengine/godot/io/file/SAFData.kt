package org.godotengine.godot.io.file

import android.content.Context
import android.net.Uri
import android.provider.DocumentsContract
import android.util.Log
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.channels.FileChannel

internal class SAFData(context: Context, fileUri: Uri, accessFlag: FileAccessFlags) :
	DataAccess.FileChannelDataAccess(fileUri.toString()) {

	companion object {
		private val TAG = FileData::class.java.simpleName

		fun fileExists(context: Context, uri: Uri): Boolean {
			return try {
				context.contentResolver.query(uri, arrayOf(DocumentsContract.Document.COLUMN_DISPLAY_NAME), null, null, null)
					?.use { cursor -> cursor.moveToFirst() } == true
			} catch (e: Exception) {
				Log.d(TAG, "Error checking file existence", e)
				false
			}
		}

		fun fileLastModified(context: Context, uri: Uri): Long {
			return try {
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
				Log.d(TAG, "Error reading lastModified for $uri", e)
				0L
			}
		}

		fun fileSize(context: Context, uri: Uri): Long {
			return try {
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
				Log.d(TAG, "Error reading lastModified for $uri", e)
				-1L
			}
		}

		fun delete(context: Context, uri: Uri): Boolean {
			return try {
				DocumentsContract.deleteDocument(context.contentResolver, uri)
			} catch (e: Exception) {
				Log.d(TAG, "Error deleting file: $uri", e)
				false
			}
		}

		fun rename(context: Context, uri: Uri, newName: String): Boolean {
		    return try {
				val newUri = DocumentsContract.renameDocument(context.contentResolver, uri, newName)
				newUri != null
			} catch (e: Exception) {
				Log.d(TAG, "Error renaming file: $uri", e)
				false
			}
		}
	}

	override val fileChannel: FileChannel
	init {
		val contentResolver = context.contentResolver
		val parcelFileDescriptor = contentResolver.openFileDescriptor(fileUri, accessFlag.getMode())
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
