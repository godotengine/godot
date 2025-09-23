/**************************************************************************/
/*  GodotAppInstrumentedTestPlugin.kt                                     */
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

package com.godot.game.test

import android.util.Log
import android.widget.Toast
import org.godotengine.godot.Godot
import org.godotengine.godot.plugin.GodotPlugin
import org.godotengine.godot.plugin.UsedByGodot
import org.godotengine.godot.Dictionary;
import org.godotengine.godot.plugin.SignalInfo
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicBoolean

/**
 * [GodotPlugin] used to drive instrumented tests.
 */
class GodotAppInstrumentedTestPlugin(godot: Godot): GodotPlugin(godot) {

	companion object {
		private val TAG = GodotAppInstrumentedTestPlugin::class.java.simpleName
		private const val MAIN_LOOP_STARTED_LATCH_KEY = "main_loop_started_latch"

		private val JAVACLASSWRAPPER_TESTS_SIGNAL = SignalInfo("launch_javaclasswrapper_tests")

		private val SIGNALS = setOf(
			JAVACLASSWRAPPER_TESTS_SIGNAL
		)
	}

	private val testResults = ConcurrentHashMap<String, Result<Any>>()
	private val latches = ConcurrentHashMap<String, CountDownLatch>()
	private val godotMainLoopStarted = AtomicBoolean(false)

    override fun getPluginName() = "GodotAppInstrumentedTestPlugin"

	override fun getPluginSignals() = SIGNALS

	override fun onGodotMainLoopStarted() {
		super.onGodotMainLoopStarted()
		godotMainLoopStarted.set(true)
		latches.remove(MAIN_LOOP_STARTED_LATCH_KEY)?.countDown()
	}

	/**
	 * Used by the instrumented test to wait until the Godot main loop is up and running.
	 */
	internal fun waitForGodotMainLoopStarted() {
		if (!godotMainLoopStarted.get()) {
			// Register a countdown latch and wait
			val mainLoopStartedLatch = latches.getOrPut(MAIN_LOOP_STARTED_LATCH_KEY) {
				CountDownLatch(1)
			}
			try {
				mainLoopStartedLatch.await()
			} catch (e: InterruptedException) {
				Log.e(TAG, "Unable to wait for Godot main loop started event.", e)
			}
		}
	}

	/**
	 * This launches the JavaClassWrapper tests, and wait until the tests are complete before returning.
	 */
	internal fun runJavaClassWrapperTests(): Result<Any>? {
		val latch = latches.getOrPut(JAVACLASSWRAPPER_TESTS_SIGNAL.name) { CountDownLatch(1) }
		emitSignal(JAVACLASSWRAPPER_TESTS_SIGNAL.name)
		return try {
			latch.await()
			val result = testResults.remove(JAVACLASSWRAPPER_TESTS_SIGNAL.name)
			result
		} catch (e: InterruptedException) {
			Log.e(TAG, "Unable to wait for JavaClassWrapper tests completion", e)
			null
		}
	}

	/**
	 * Callback invoked from gdscript when the JavaClassWrapper tests complete.
	 */
	@UsedByGodot
	fun onJavaClassWrapperTestsCompleted(passes: Int, failures: Int) {
		Log.d(TAG, "JavaClassWrapper tests completed")
		val result = if (failures == 0) {
			Result.success(passes)
		} else {
			Result.failure(AssertionError("$failures tests failed!"))
		}

		completeTest(JAVACLASSWRAPPER_TESTS_SIGNAL.name, result)
	}

	private fun completeTest(testKey: String, result: Result<Any>) {
		testResults[testKey] = result
		latches.remove(testKey)?.countDown()
	}

    /**
     * Example showing how to declare a method that's used by Godot.
     *
     * Shows a 'Hello World' toast.
     */
    @UsedByGodot
    fun helloWorld() {
        runOnUiThread {
            Toast.makeText(activity, "Toast from Android plugin", Toast.LENGTH_LONG).show()
            Log.v(pluginName, "Hello World")
        }
    }

    fun stringify(value: Any?): String {
        return when (value) {
            null -> "null"
            is Map<*, *> -> {
                val entries = value.entries.joinToString(", ") { (k, v) -> "${stringify(k)}: ${stringify(v)}" }
                "{$entries}"
            }
            is List<*> -> value.joinToString(prefix = "[", postfix = "]") { stringify(it) }
            is Array<*> -> value.joinToString(prefix = "[", postfix = "]") { stringify(it) }
            is IntArray -> value.joinToString(prefix = "[", postfix = "]")
            is LongArray -> value.joinToString(prefix = "[", postfix = "]")
            is FloatArray -> value.joinToString(prefix = "[", postfix = "]")
            is DoubleArray -> value.joinToString(prefix = "[", postfix = "]")
            is BooleanArray -> value.joinToString(prefix = "[", postfix = "]")
            is CharArray -> value.joinToString(prefix = "[", postfix = "]")
            else -> value.toString()
        }
    }

    @UsedByGodot
    fun testDictionary(d: Dictionary): String {
        return d.toString()
    }

    @UsedByGodot
    fun testDictionaryNested(d: Dictionary): String {
        return stringify(d)
    }

    @UsedByGodot
    fun testRetDictionary(): Dictionary {
        var d = Dictionary()
        d.putAll(mapOf("a" to 1, "b" to 2))
        return d
    }

    @UsedByGodot
    fun testRetDictionaryArray(): Array<Dictionary> {
        var d = Dictionary()
        d.putAll(mapOf("a" to 1, "b" to 2))
        return arrayOf(d)
    }


}
