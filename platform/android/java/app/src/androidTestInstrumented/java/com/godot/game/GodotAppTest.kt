/**************************************************************************/
/*  GodotAppTest.kt                                                       */
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

package com.godot.game

import android.util.Log
import androidx.test.ext.junit.rules.ActivityScenarioRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.godot.game.test.GodotAppInstrumentedTestPlugin
import org.godotengine.godot.plugin.GodotPluginRegistry
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * This instrumented test will launch the `instrumented` version of GodotApp and run a set of tests against it.
 */
@RunWith(AndroidJUnit4::class)
class GodotAppTest {

	companion object {
		private val TAG = GodotAppTest::class.java.simpleName
	}

	@get:Rule
	val godotAppRule = ActivityScenarioRule(GodotApp::class.java)

	/**
	 * Runs the JavaClassWrapper tests via the GodotAppInstrumentedTestPlugin.
	 */
	@Test
	fun runJavaClassWrapperTests() {
		val testPlugin = GodotPluginRegistry.getPluginRegistry()
			.getPlugin("GodotAppInstrumentedTestPlugin") as GodotAppInstrumentedTestPlugin?
		assertNotNull(testPlugin)

		Log.d(TAG, "Waiting for the Godot main loop to start...")
		testPlugin.waitForGodotMainLoopStarted()

		Log.d(TAG, "Running JavaClassWrapper tests...")
		val result = testPlugin.runJavaClassWrapperTests()
		assertNotNull(result)
		result.exceptionOrNull()?.let { throw it }
		assertTrue(result.isSuccess)
		Log.d(TAG, "Passed ${result.getOrNull()} tests")
	}
}
