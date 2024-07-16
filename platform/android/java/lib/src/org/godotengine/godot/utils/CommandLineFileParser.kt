/**************************************************************************/
/*  CommandLineFileParser.kt                                              */
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

package org.godotengine.godot.utils

import java.io.InputStream
import java.nio.charset.StandardCharsets
import java.util.ArrayList

/**
 * A class that parses the content of file storing command line params. Usually, this file is saved
 * in `assets/_cl_` on exporting an apk
 *
 * Returns a mutable list of command lines
 */
internal class CommandLineFileParser {
	fun parseCommandLine(inputStream: InputStream): MutableList<String> {
		return try {
			val headerBytes = ByteArray(4)
			var argBytes = inputStream.read(headerBytes)
			if (argBytes < 4) {
				return mutableListOf()
			}
			val argc = decodeHeaderIntValue(headerBytes)

			val cmdline = ArrayList<String>(argc)
			for (i in 0 until argc) {
				argBytes = inputStream.read(headerBytes)
				if (argBytes < 4) {
					return mutableListOf()
				}
				val strlen = decodeHeaderIntValue(headerBytes)

				if (strlen > 65535) {
					return mutableListOf()
				}

				val arg = ByteArray(strlen)
				argBytes = inputStream.read(arg)
				if (argBytes == strlen) {
					cmdline.add(String(arg, StandardCharsets.UTF_8))
				}
			}
			cmdline
		} catch (e: Exception) {
			// The _cl_ file can be missing with no adverse effect
			mutableListOf()
		}
	}

	private fun decodeHeaderIntValue(headerBytes: ByteArray): Int =
		(headerBytes[3].toInt() and 0xFF) shl 24 or
		((headerBytes[2].toInt() and 0xFF) shl 16) or
		((headerBytes[1].toInt() and 0xFF) shl 8) or
		(headerBytes[0].toInt() and 0xFF)
}
