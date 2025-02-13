/**************************************************************************/
/*  DirectoryAccessHandler.kt                                             */
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

package org.godotengine.godot.io.directory

import android.content.Context
import android.util.Log
import org.godotengine.godot.Godot
import org.godotengine.godot.io.StorageScope
import org.godotengine.godot.io.directory.DirectoryAccessHandler.AccessType.ACCESS_RESOURCES

/**
 * Handles files and directories access and manipulation for the Android platform
 */
class DirectoryAccessHandler(context: Context) {

	companion object {
		private val TAG = DirectoryAccessHandler::class.java.simpleName

		internal const val INVALID_DIR_ID = -1
		internal const val STARTING_DIR_ID = 1
	}

	private enum class AccessType(val nativeValue: Int) {
		ACCESS_RESOURCES(0),

		/**
		 * Maps to [ACCESS_FILESYSTEM]
		 */
		ACCESS_USERDATA(1),
		ACCESS_FILESYSTEM(2);

		fun generateDirAccessId(dirId: Int) = (dirId * DIR_ACCESS_ID_MULTIPLIER) + nativeValue

		companion object {
			const val DIR_ACCESS_ID_MULTIPLIER = 10

			fun fromDirAccessId(dirAccessId: Int): Pair<AccessType?, Int> {
				val nativeValue = dirAccessId % DIR_ACCESS_ID_MULTIPLIER
				val dirId = dirAccessId / DIR_ACCESS_ID_MULTIPLIER
				return Pair(fromNative(nativeValue), dirId)
			}

			private fun fromNative(nativeAccessType: Int): AccessType? {
				for (accessType in entries) {
					if (accessType.nativeValue == nativeAccessType) {
						return accessType
					}
				}
				return null
			}

			fun fromNative(nativeAccessType: Int, storageScope: StorageScope? = null): AccessType? {
				val accessType = fromNative(nativeAccessType)
				if (accessType == null) {
					Log.w(TAG, "Unsupported access type $nativeAccessType")
					return null
				}

				// 'Resources' access type takes precedence as it is simple to handle:
				// if we receive a 'Resources' access type and this is a template build,
				// we provide a 'Resources' directory handler.
				// If this is an editor build, 'Resources' refers to the opened project resources
				// and so we provide a 'Filesystem' directory handler.
				if (accessType == ACCESS_RESOURCES) {
					return if (Godot.isEditorBuild()) {
						ACCESS_FILESYSTEM
					} else {
						ACCESS_RESOURCES
					}
				} else {
					// We've received a 'Filesystem' or 'Userdata' access type. On Android, this
					// may refer to:
					// - assets directory (path has 'assets:/' prefix)
					// - app directories
					// - device shared directories
					// As such we check the storage scope (if available) to figure what type of
					// directory handler to provide
					if (storageScope != null) {
						val accessTypeFromStorageScope = when (storageScope) {
							StorageScope.ASSETS -> ACCESS_RESOURCES
							StorageScope.APP, StorageScope.SHARED -> ACCESS_FILESYSTEM
							StorageScope.UNKNOWN -> null
						}

						if (accessTypeFromStorageScope != null) {
							return accessTypeFromStorageScope
						}
					}
					// If we're not able to infer the type of directory handler from the storage
					// scope, we fall-back to the 'Filesystem' directory handler as it's the default
					// for the 'Filesystem' access type.
					// Note that ACCESS_USERDATA also maps to ACCESS_FILESYSTEM
					return ACCESS_FILESYSTEM
				}
			}
		}
	}

	internal interface DirectoryAccess {
		fun dirOpen(path: String): Int
		fun dirNext(dirId: Int): String
		fun dirClose(dirId: Int)
		fun dirIsDir(dirId: Int): Boolean
		fun dirExists(path: String): Boolean
		fun fileExists(path: String): Boolean
		fun hasDirId(dirId: Int): Boolean
		fun isCurrentHidden(dirId: Int): Boolean
		fun getDriveCount() : Int
		fun getDrive(drive: Int): String
		fun makeDir(dir: String): Boolean
		fun getSpaceLeft(): Long
		fun rename(from: String, to: String): Boolean
		fun remove(filename: String): Boolean
	}

	private val storageScopeIdentifier = StorageScope.Identifier(context)

	private val assetsDirAccess = AssetsDirectoryAccess(context)
	private val fileSystemDirAccess = FilesystemDirectoryAccess(context, storageScopeIdentifier)

	fun assetsFileExists(assetsPath: String) = assetsDirAccess.fileExists(assetsPath)
	fun filesystemFileExists(path: String) = fileSystemDirAccess.fileExists(path)

	private fun hasDirId(accessType: AccessType, dirId: Int): Boolean {
		return when (accessType) {
			ACCESS_RESOURCES -> assetsDirAccess.hasDirId(dirId)
			else -> fileSystemDirAccess.hasDirId(dirId)
		}
	}

	fun dirOpen(nativeAccessType: Int, path: String?): Int {
		if (path == null) {
			return INVALID_DIR_ID
		}

		val storageScope = storageScopeIdentifier.identifyStorageScope(path)
		val accessType = AccessType.fromNative(nativeAccessType, storageScope) ?: return INVALID_DIR_ID

		val dirId = when (accessType) {
			ACCESS_RESOURCES -> assetsDirAccess.dirOpen(path)
			else -> fileSystemDirAccess.dirOpen(path)
		}
		if (dirId == INVALID_DIR_ID) {
			return INVALID_DIR_ID
		}

		val dirAccessId = accessType.generateDirAccessId(dirId)
		return dirAccessId
	}

	fun dirNext(dirAccessId: Int): String {
		val (accessType, dirId) = AccessType.fromDirAccessId(dirAccessId)
		if (accessType == null || !hasDirId(accessType, dirId)) {
			Log.w(TAG, "dirNext: Invalid dir id: $dirId")
			return ""
		}

		return when (accessType) {
			ACCESS_RESOURCES -> assetsDirAccess.dirNext(dirId)
			else -> fileSystemDirAccess.dirNext(dirId)
		}
	}

	fun dirClose(dirAccessId: Int) {
		val (accessType, dirId) = AccessType.fromDirAccessId(dirAccessId)
		if (accessType == null || !hasDirId(accessType, dirId)) {
			Log.w(TAG, "dirClose: Invalid dir id: $dirId")
			return
		}

		when (accessType) {
			ACCESS_RESOURCES -> assetsDirAccess.dirClose(dirId)
			else -> fileSystemDirAccess.dirClose(dirId)
		}
	}

	fun dirIsDir(dirAccessId: Int): Boolean {
		val (accessType, dirId) = AccessType.fromDirAccessId(dirAccessId)
		if (accessType == null || !hasDirId(accessType, dirId)) {
			Log.w(TAG, "dirIsDir: Invalid dir id: $dirId")
			return false
		}

		return when (accessType) {
			ACCESS_RESOURCES -> assetsDirAccess.dirIsDir(dirId)
			else -> fileSystemDirAccess.dirIsDir(dirId)
		}
	}

	fun isCurrentHidden(dirAccessId: Int): Boolean {
		val (accessType, dirId) = AccessType.fromDirAccessId(dirAccessId)
		if (accessType == null || !hasDirId(accessType, dirId)) {
			return false
		}

		return when (accessType) {
			ACCESS_RESOURCES -> assetsDirAccess.isCurrentHidden(dirId)
			else -> fileSystemDirAccess.isCurrentHidden(dirId)
		}
	}

	fun dirExists(nativeAccessType: Int, path: String?): Boolean {
		if (path == null) {
			return false
		}

		val storageScope = storageScopeIdentifier.identifyStorageScope(path)
		val accessType = AccessType.fromNative(nativeAccessType, storageScope) ?: return false

		return when (accessType) {
			ACCESS_RESOURCES -> assetsDirAccess.dirExists(path)
			else -> fileSystemDirAccess.dirExists(path)
		}
	}

	fun fileExists(nativeAccessType: Int, path: String?): Boolean {
		if (path == null) {
			return false
		}

		val storageScope = storageScopeIdentifier.identifyStorageScope(path)
		val accessType = AccessType.fromNative(nativeAccessType, storageScope) ?: return false

		return when (accessType) {
			ACCESS_RESOURCES -> assetsDirAccess.fileExists(path)
			else -> fileSystemDirAccess.fileExists(path)
		}
	}

	fun getDriveCount(nativeAccessType: Int): Int {
		val accessType = AccessType.fromNative(nativeAccessType) ?: return 0
		return when(accessType) {
			ACCESS_RESOURCES -> assetsDirAccess.getDriveCount()
			else -> fileSystemDirAccess.getDriveCount()
		}
	}

	fun getDrive(nativeAccessType: Int, drive: Int): String {
		val accessType = AccessType.fromNative(nativeAccessType) ?: return ""
		return when (accessType) {
			ACCESS_RESOURCES -> assetsDirAccess.getDrive(drive)
			else -> fileSystemDirAccess.getDrive(drive)
		}
	}

	fun makeDir(nativeAccessType: Int, dir: String?): Boolean {
		if (dir == null) {
			return false
		}

		val storageScope = storageScopeIdentifier.identifyStorageScope(dir)
		val accessType = AccessType.fromNative(nativeAccessType, storageScope) ?: return false

		return when (accessType) {
			ACCESS_RESOURCES -> assetsDirAccess.makeDir(dir)
			else -> fileSystemDirAccess.makeDir(dir)
		}
	}

	fun getSpaceLeft(nativeAccessType: Int): Long {
		val accessType = AccessType.fromNative(nativeAccessType) ?: return 0L
		return when (accessType) {
			ACCESS_RESOURCES -> assetsDirAccess.getSpaceLeft()
			else -> fileSystemDirAccess.getSpaceLeft()
		}
	}

	fun rename(nativeAccessType: Int, from: String, to: String): Boolean {
		val accessType = AccessType.fromNative(nativeAccessType) ?: return false
		return when (accessType) {
			ACCESS_RESOURCES -> assetsDirAccess.rename(from, to)
			else -> fileSystemDirAccess.rename(from, to)
		}
	}

	fun remove(nativeAccessType: Int, filename: String?): Boolean {
		if (filename == null) {
			return false
		}

		val storageScope = storageScopeIdentifier.identifyStorageScope(filename)
		val accessType = AccessType.fromNative(nativeAccessType, storageScope) ?: return false
		return when (accessType) {
			ACCESS_RESOURCES -> assetsDirAccess.remove(filename)
			else -> fileSystemDirAccess.remove(filename)
		}
	}

}
