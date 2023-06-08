/**************************************************************************/
/*  FileAccessFlags.kt                                                    */
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

/**
 * Android representation of Godot native access flag bit fields.
 * Reference: core/io/file_access.h
 */
internal enum class ModeBitFields(val nativeValue: Int) {
    READ_FIELD(1), // r
    WRITE_FIELD(2), // w
    APPEND_FIELD(4), // a
    TEMPORARY_FIELD(8); // D
    //TEXT_FIELD(16); // t/b
}
/**
 * Android representation of Godot native access flags.
 */
internal enum class FileAccessFlags(val nativeValue: Int) {
    /**
     * Opens the file for read operations.
     * The cursor is positioned at the beginning of the file.
     */
    READ(ModeBitFields.READ_FIELD.nativeValue),

    /**
     * Opens the file for write operations.
     * The file is created if it does not exist, and truncated if it does.
     */
    WRITE(ModeBitFields.WRITE_FIELD.nativeValue),

    /**
     * Opens the file for write operations.
     * The file is created if it does not exist, but does not truncate if it does.
     * The cursor is positioned at the end of the file.
     */
    APPEND(ModeBitFields.APPEND_FIELD.nativeValue),

    /**
     * Opens the file for read and write operations.
     * Does not truncate the file. The cursor is positioned at the beginning of the file.
     */
    READ_WRITE(ModeBitFields.READ_FIELD.nativeValue or ModeBitFields.APPEND_FIELD.nativeValue),

    /**
     * Opens the file for read and write operations.
     * The file is created if it does not exist, and truncated if it does.
     * The cursor is positioned at the beginning of the file.
     */
    WRITE_READ(ModeBitFields.WRITE_FIELD.nativeValue or ModeBitFields.APPEND_FIELD.nativeValue),

    /**
     * Opens the file for read operations. File is flagged for removal.
     * Does not truncate the file. The cursor is positioned at the beginning of the file.
     */
    TEMPORARY_READ(ModeBitFields.TEMPORARY_FIELD.nativeValue or ModeBitFields.READ_FIELD.nativeValue),

    /**
     * Opens the file for write operations. File is flagged for removal.
     * The file is created if it does not exist, and truncated if it does.
	 * The cursor is positioned at the beginning of the file.
     */
    TEMPORARY_WRITE(ModeBitFields.TEMPORARY_FIELD.nativeValue or ModeBitFields.WRITE_FIELD.nativeValue),

    /**
     * Opens the file for write operations. File is flagged for removal.
     * The file is created if it does not exist, but does not truncate if it does.
     * The cursor is positioned at the end of the file.
     */
    TEMPORARY_APPEND(ModeBitFields.TEMPORARY_FIELD.nativeValue or ModeBitFields.APPEND_FIELD.nativeValue),

    /**
     * Opens the file for read and write operations. File is flagged for removal.
     * Does not truncate the file. The cursor is positioned at the beginning of the file.
     */
    TEMPORARY_READ_WRITE(ModeBitFields.TEMPORARY_FIELD.nativeValue or ModeBitFields.READ_FIELD.nativeValue or ModeBitFields.APPEND_FIELD.nativeValue),

    /**
     * Opens the file for read and write operations. File is flagged for removal.
     * The file is created if it does not exist, and truncated if it does.
     * The cursor is positioned at the beginning of the file.
     */
    TEMPORARY_WRITE_READ(ModeBitFields.TEMPORARY_FIELD.nativeValue or ModeBitFields.WRITE_FIELD.nativeValue or ModeBitFields.APPEND_FIELD.nativeValue);

    fun getMode(): String {
        return when (this) {
            READ, TEMPORARY_READ -> "r"
            WRITE, TEMPORARY_WRITE -> "w"
            APPEND, TEMPORARY_APPEND -> "a"
            READ_WRITE, WRITE_READ, TEMPORARY_READ_WRITE, TEMPORARY_WRITE_READ -> "rw"
        }
    }

    fun shouldTruncate(): Boolean {
        return when (this) {
            READ, READ_WRITE, APPEND, TEMPORARY_READ, TEMPORARY_READ_WRITE, TEMPORARY_APPEND -> false
            WRITE, WRITE_READ, TEMPORARY_WRITE, TEMPORARY_WRITE_READ -> true
        }
    }

    fun shouldCreate(): Boolean {
        return when (this) {
            READ, TEMPORARY_READ  -> false
            READ_WRITE, APPEND, TEMPORARY_READ_WRITE, TEMPORARY_APPEND, WRITE, WRITE_READ, TEMPORARY_WRITE, TEMPORARY_WRITE_READ -> true
        }
    }

    fun shouldRemove(): Boolean {
        return when (this) {
            READ, READ_WRITE, APPEND, WRITE, WRITE_READ -> false
            TEMPORARY_READ, TEMPORARY_READ_WRITE, TEMPORARY_APPEND, TEMPORARY_WRITE, TEMPORARY_WRITE_READ -> true
        }
    }

    companion object {
        fun fromNativeModeFlags(modeFlag: Int): FileAccessFlags? {
            for (flag in values()) {
                if (flag.nativeValue == modeFlag) {
                    return flag
                }
            }
            return null
        }
    }
}
