/*************************************************************************/
/*  FileAccessFlags.kt                                                   */
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

/**
 * Android representation of Godot native access flags.
 */
internal enum class FileAccessFlags(val nativeValue: Int) {
    /**
     * Opens the file for read operations.
     * The cursor is positioned at the beginning of the file.
     */
    READ(1),

    /**
     * Opens the file for write operations.
     * The file is created if it does not exist, and truncated if it does.
     */
    WRITE(2),

    /**
     * Opens the file for read and write operations.
     * Does not truncate the file. The cursor is positioned at the beginning of the file.
     */
    READ_WRITE(3),

    /**
     * Opens the file for read and write operations.
     * The file is created if it does not exist, and truncated if it does.
     * The cursor is positioned at the beginning of the file.
     */
    WRITE_READ(7);

    fun getMode(): String {
        return when (this) {
            READ -> "r"
            WRITE -> "w"
            READ_WRITE, WRITE_READ -> "rw"
        }
    }

    fun shouldTruncate(): Boolean {
        return when (this) {
            READ, READ_WRITE -> false
            WRITE, WRITE_READ -> true
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
