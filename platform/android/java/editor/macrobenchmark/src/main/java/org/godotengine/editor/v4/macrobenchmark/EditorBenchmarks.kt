/**************************************************************************/
/*  EditorBenchmarks.kt                                                   */
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


package org.godotengine.editor.v4.macrobenchmark

import android.Manifest
import androidx.benchmark.macro.ExperimentalMetricApi
import androidx.benchmark.macro.FrameTimingGfxInfoMetric
import androidx.benchmark.macro.MemoryUsageMetric
import androidx.benchmark.macro.StartupMode
import androidx.benchmark.macro.StartupTimingMetric
import androidx.benchmark.macro.TraceSectionMetric
import androidx.benchmark.macro.junit4.MacrobenchmarkRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.rule.GrantPermissionRule
import androidx.test.uiautomator.By
import androidx.test.uiautomator.StaleObjectException
import androidx.test.uiautomator.Until
import androidx.test.uiautomator.uiAutomator
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Set of editor macro benchmarks.
 *
 * Before running, switch the editor's active build variant to 'benchmark'
 */
@RunWith(AndroidJUnit4::class)
class EditorBenchmarks {

	companion object {
		private val TAG = EditorBenchmarks::class.java.simpleName

		private const val PACKAGE_NAME = "org.godotengine.editor.v4.benchmark"
		private const val ITERATION_COUNT = 5
		private const val EXTRA_LOAD_EMPTY_BENCHMARK_PROJECT = "load_empty_benchmark_project"
	}

	@get:Rule
    val benchmarkRule = MacrobenchmarkRule()
	@get:Rule
    val grantPermissionRule = GrantPermissionRule.grant(
		Manifest.permission.READ_EXTERNAL_STORAGE,
		Manifest.permission.WRITE_EXTERNAL_STORAGE)

	/**
	 * Navigates to the device's home screen, and launches the Project Manager.
	 */
	@OptIn(ExperimentalMetricApi::class)
	@Test
	fun startupProjectManager() = benchmarkRule.measureRepeated(
			packageName = PACKAGE_NAME,
			metrics = listOf(
				StartupTimingMetric(),
				MemoryUsageMetric(MemoryUsageMetric.Mode.Max),
				FrameTimingGfxInfoMetric(),
			),
			iterations = ITERATION_COUNT,
			startupMode = StartupMode.COLD
	) {
		pressHome()
		startActivityAndWait()
	}

	@OptIn(ExperimentalMetricApi::class)
	@Test
	fun startupEditorWindow() = benchmarkRule.measureRepeated(
		packageName = PACKAGE_NAME,
		metrics = listOf(
			StartupTimingMetric(),
			MemoryUsageMetric(MemoryUsageMetric.Mode.Max),
			FrameTimingGfxInfoMetric(),
		),
		iterations = ITERATION_COUNT,
		startupMode = StartupMode.COLD,
	) {
		pressHome()
		startActivityAndWait { intent ->
			intent.putExtra(EXTRA_LOAD_EMPTY_BENCHMARK_PROJECT, true)
		}

		uiAutomator {
			try {
				val selector = By.res("$PACKAGE_NAME:id/editor_loading_indicator")
				val editorLoadingIndicator = device.findObject(selector)
				editorLoadingIndicator?.wait(Until.gone(selector), 60_000L)
			} catch (_: StaleObjectException) {
			}
		}
	}
}
