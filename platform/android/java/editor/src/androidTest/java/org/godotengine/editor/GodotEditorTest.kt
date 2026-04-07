/**************************************************************************/
/*  GodotEditorTest.kt                                                    */
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

package org.godotengine.editor

import android.content.ComponentName
import android.content.Intent
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.godotengine.godot.GodotActivity.Companion.EXTRA_COMMAND_LINE_PARAMS
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Instrumented test for the Godot editor.
 */
@RunWith(AndroidJUnit4::class)
class GodotEditorTest {
	companion object {
		private val TAG = GodotEditorTest::class.simpleName

		private val TEST_COMMAND_LINE_PARAMS = arrayOf("This is a test")
		private const val PROJECT_MANAGER_CLASS_NAME = "org.godotengine.editor.ProjectManager"
		private const val GODOT_EDITOR_CLASS_NAME = "org.godotengine.editor.GodotEditor"
	}

	/**
	 * Implicitly launch the project manager.
	 */
	@Test
	fun testImplicitProjectManagerLaunch() {
		val implicitLaunchIntent = Intent().apply {
			setPackage(BuildConfig.APPLICATION_ID)
			action = Intent.ACTION_MAIN
			addCategory(Intent.CATEGORY_LAUNCHER)
			putExtra(EXTRA_COMMAND_LINE_PARAMS, TEST_COMMAND_LINE_PARAMS)
		}
		ActivityScenario.launch<GodotEditor>(implicitLaunchIntent).use { scenario ->
			scenario.onActivity { activity ->
				assertEquals(activity.intent.component?.className, PROJECT_MANAGER_CLASS_NAME)

				val commandLineParams = activity.intent.getStringArrayExtra(EXTRA_COMMAND_LINE_PARAMS)
				assertNull(commandLineParams)
			}
		}
	}

	/**
	 * Explicitly launch the project manager.
	 */
	@Test
	fun testExplicitProjectManagerLaunch() {
		val explicitProjectManagerIntent = Intent().apply {
			component = ComponentName(BuildConfig.APPLICATION_ID, PROJECT_MANAGER_CLASS_NAME)
			putExtra(EXTRA_COMMAND_LINE_PARAMS, TEST_COMMAND_LINE_PARAMS)
		}
		ActivityScenario.launch<GodotEditor>(explicitProjectManagerIntent).use { scenario ->
			scenario.onActivity { activity ->
				assertEquals(activity.intent.component?.className, PROJECT_MANAGER_CLASS_NAME)

				val commandLineParams = activity.intent.getStringArrayExtra(EXTRA_COMMAND_LINE_PARAMS)
				assertNull(commandLineParams)
			}
		}
	}

	/**
	 * Explicitly launch the `GodotEditor` activity.
	 */
	@Test
	fun testExplicitGodotEditorLaunch() {
		val godotEditorIntent = Intent().apply {
			component = ComponentName(BuildConfig.APPLICATION_ID, GODOT_EDITOR_CLASS_NAME)
			putExtra(EXTRA_COMMAND_LINE_PARAMS, TEST_COMMAND_LINE_PARAMS)
		}
		ActivityScenario.launch<GodotEditor>(godotEditorIntent).use { scenario ->
			scenario.onActivity { activity ->
				assertEquals(activity.intent.component?.className, GODOT_EDITOR_CLASS_NAME)

				val commandLineParams = activity.intent.getStringArrayExtra(EXTRA_COMMAND_LINE_PARAMS)
				assertNotNull(commandLineParams)
				assertTrue(commandLineParams.contentEquals(TEST_COMMAND_LINE_PARAMS))
			}
		}
	}
}
