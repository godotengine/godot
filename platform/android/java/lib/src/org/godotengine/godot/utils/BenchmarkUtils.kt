/**************************************************************************/
/*  BenchmarkUtils.kt                                                     */
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

@file:JvmName("BenchmarkUtils")

package org.godotengine.godot.utils

import android.os.Build
import android.os.SystemClock
import android.os.Trace
import android.util.Log
import org.godotengine.godot.BuildConfig
import org.godotengine.godot.io.file.FileAccessFlags
import org.godotengine.godot.io.file.FileAccessHandler
import org.json.JSONObject
import java.nio.ByteBuffer
import java.util.concurrent.ConcurrentSkipListMap

/**
 * Contains benchmark related utilities methods
 */
private const val TAG = "GodotBenchmark"

var useBenchmark = false
var benchmarkFile = ""

private val startBenchmarkFrom = ConcurrentSkipListMap<String, Long>()
private val benchmarkTracker = ConcurrentSkipListMap<String, Double>()

/**
 * Start measuring and tracing the execution of a given section of code using the given label.
 *
 * Must be followed by a call to [endBenchmarkMeasure].
 *
 * Note: Only enabled on 'editorDev' build variant.
 */
fun beginBenchmarkMeasure(label: String) {
	if (BuildConfig.FLAVOR != "editor" || BuildConfig.BUILD_TYPE != "dev") {
		return
	}
	startBenchmarkFrom[label] = SystemClock.elapsedRealtime()

	if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
		Trace.beginAsyncSection(label, 0)
	}
}

/**
 * End measuring and tracing of the section of code with the given label.
 *
 * Must be preceded by a call [beginBenchmarkMeasure]
 *
 * Note: Only enabled on 'editorDev' build variant.
 */
fun endBenchmarkMeasure(label: String) {
	if (BuildConfig.FLAVOR != "editor" || BuildConfig.BUILD_TYPE != "dev") {
		return
	}
	val startTime = startBenchmarkFrom[label] ?: return
	val total = SystemClock.elapsedRealtime() - startTime
	benchmarkTracker[label] = total / 1000.0

	if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
		Trace.endAsyncSection(label, 0)
	}
}

/**
 * Dump the benchmark measurements.
 * If [filepath] is valid, the data is also written in json format to the specified file.
 *
 * Note: Only enabled on 'editorDev' build variant.
 */
@JvmOverloads
fun dumpBenchmark(fileAccessHandler: FileAccessHandler?, filepath: String? = benchmarkFile) {
	if (BuildConfig.FLAVOR != "editor" || BuildConfig.BUILD_TYPE != "dev") {
		return
	}
	if (!useBenchmark) {
		return
	}

	val printOut =
		benchmarkTracker.map { "\t- ${it.key} : ${it.value} sec." }.joinToString("\n")
	Log.i(TAG, "BENCHMARK:\n$printOut")

	if (fileAccessHandler != null && !filepath.isNullOrBlank()) {
		val fileId = fileAccessHandler.fileOpen(filepath, FileAccessFlags.WRITE)
		if (fileId != FileAccessHandler.INVALID_FILE_ID) {
			val jsonOutput = JSONObject(benchmarkTracker.toMap()).toString(4)
			fileAccessHandler.fileWrite(fileId, ByteBuffer.wrap(jsonOutput.toByteArray()))
			fileAccessHandler.fileClose(fileId)
		}
	}
}
