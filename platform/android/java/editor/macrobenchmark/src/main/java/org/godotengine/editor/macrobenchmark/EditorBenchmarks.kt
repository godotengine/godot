package org.godotengine.editor.macrobenchmark

import android.Manifest
import androidx.benchmark.macro.ExperimentalMetricApi
import androidx.benchmark.macro.MemoryUsageMetric
import androidx.benchmark.macro.StartupMode
import androidx.benchmark.macro.StartupTimingMetric
import androidx.benchmark.macro.junit4.MacrobenchmarkRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.rule.GrantPermissionRule
import androidx.test.uiautomator.By
import androidx.test.uiautomator.StaleObjectException
import androidx.test.uiautomator.Until
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
		const val PACKAGE_NAME = "org.godotengine.editor.v4.benchmark"
	}

	@get:Rule val benchmarkRule = MacrobenchmarkRule()
	@get:Rule val grantPermissionRule = GrantPermissionRule.grant(
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
			),
			iterations = 5,
			startupMode = StartupMode.COLD
	) {
		pressHome()
		startActivityAndWait()

		try {
			val editorLoadingIndicator = device.findObject(By.res(PACKAGE_NAME, "editor_loading_indicator"))
			editorLoadingIndicator.wait(Until.gone(By.res(PACKAGE_NAME, "editor_loading_indicator")), 5_000)
		} catch (ignored: StaleObjectException) {}
	}
}
