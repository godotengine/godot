/**************************************************************************/
/*  DeviceUtils.kt                                                        */
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

/**
 * Contains utility methods for detecting specific devices.
 */
@file:JvmName("DeviceUtils")

package org.godotengine.godot.utils

import android.content.Context
import android.os.Build
import java.io.BufferedReader
import java.io.FileReader

/**
 * Returns true if running on Meta Horizon OS.
 */
fun isHorizonOSDevice(context: Context): Boolean {
	return context.packageManager.hasSystemFeature("oculus.hardware.standalone_vr")
}

/**
 * Returns true if running on PICO OS.
 */
fun isPicoOSDevice(): Boolean {
	return ("Pico".equals(Build.BRAND, true))
}

/**
 * Returns true if running on a native Android XR device.
 */
fun isNativeXRDevice(context: Context): Boolean {
	return isHorizonOSDevice(context) || isPicoOSDevice()
}

/**
 * Checks if the device has a problematic Adreno GPU configuration.
 * This method detects known problematic SoCs with Adreno 5XX GPUs that may have
 * issues with Vulkan or OpenGL ES rendering.
 *
 * @return true if this is a problematic Adreno GPU configuration
 */
fun isProblematicAdrenoGpu(): Boolean {
	try {
		val hardware = Build.HARDWARE.lowercase()
		val board = Build.BOARD.lowercase()

		// Known problematic SoCs with Adreno 5XX GPUs:
		// - msm8953: Snapdragon 625 (Adreno 506)
		// - msm8937/msm8940: Snapdragon 430/435 (Adreno 505)
		// - msm8917: Snapdragon 425 (Adreno 505)
		// - msm8976: Snapdragon 652/653 (Adreno 510)
		// - msm8956: Snapdragon 650 (Adreno 510)
		// - sdm660: Snapdragon 660 (Adreno 509/512)
		// - sdm636: Snapdragon 636 (Adreno 509)
		val isProblematicSoc = hardware.contains("qcom") && (
			hardware.contains("msm8953") ||  // Snapdragon 625 (Adreno 506)
			hardware.contains("msm8937") ||  // Snapdragon 430 (Adreno 505)
			hardware.contains("msm8940") ||  // Snapdragon 435 (Adreno 505)
			hardware.contains("msm8917") ||  // Snapdragon 425 (Adreno 505)
			hardware.contains("msm8976") ||  // Snapdragon 652/653 (Adreno 510)
			hardware.contains("msm8956") ||  // Snapdragon 650 (Adreno 510)
			hardware.contains("sdm660") ||   // Snapdragon 660 (Adreno 509/512)
			hardware.contains("sdm636") ||   // Snapdragon 636 (Adreno 509)
			board.contains("msm8953") ||
			board.contains("msm8937") ||
			board.contains("msm8940") ||
			board.contains("msm8917") ||
			board.contains("msm8976") ||
			board.contains("msm8956") ||
			board.contains("sdm660") ||
			board.contains("sdm636")
		)

		// Also check for Snapdragon 6XX series on Android 9 and below
		// as these often have Adreno 5XX GPUs with similar issues
		if (!isProblematicSoc && Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
			try {
				BufferedReader(FileReader("/proc/cpuinfo")).use { reader ->
					var line: String?
					while (reader.readLine().also { line = it } != null) {
						val lowerLine = line!!.lowercase()
						if (lowerLine.contains("hardware") && lowerLine.contains("qualcomm")) {
							if (lowerLine.contains("msm8953") || lowerLine.contains("msm8937") ||
								lowerLine.contains("msm8940") || lowerLine.contains("msm8917") ||
								lowerLine.contains("msm8976") || lowerLine.contains("msm8956") ||
								lowerLine.contains("sdm660") || lowerLine.contains("sdm636")) {
								return true
							}
						}
					}
				}
			} catch (e: Exception) {
				// Ignore errors when reading /proc/cpuinfo
			}
		}

		return isProblematicSoc
	} catch (e: Exception) {
		// Be conservative on Android 9 and below - assume problematic if detection fails
		return Build.VERSION.SDK_INT <= Build.VERSION_CODES.P
	}
}
