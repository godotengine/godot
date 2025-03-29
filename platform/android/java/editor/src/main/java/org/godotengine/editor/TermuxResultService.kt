/**************************************************************************/
/*  TermuxResultService.kt                                                */
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

import android.app.IntentService
import android.content.Intent
import android.os.Bundle
import android.util.Log
import java.util.concurrent.ConcurrentHashMap
import org.godotengine.godot.variant.Callable

class TermuxResultService : IntentService(PLUGIN_SERVICE_LABEL) {

	companion object {
		const val EXTRA_EXECUTION_ID = "execution_id"
		const val PLUGIN_SERVICE_LABEL = "TermuxResultService"
		private const val LOG_TAG = "TermuxResultService"
		private var EXECUTION_ID = 1000
		private val executionMap = ConcurrentHashMap<Int, Callable>()

		@Synchronized
		fun getNextExecutionId(resultCallback: Callable): Int {
			val id = EXECUTION_ID++
			executionMap[id] = resultCallback
			return id
		}
	}

	override fun onHandleIntent(intent: Intent?) {
		if (intent == null) return

		Log.d(LOG_TAG, "$PLUGIN_SERVICE_LABEL received execution result")

		val resultBundle = intent.getBundleExtra("result")
		if (resultBundle == null) {
			Log.e(LOG_TAG, "The intent does not contain the result bundle at the \"result\" key.")
			return
		}

		val executionId = intent.getIntExtra(EXTRA_EXECUTION_ID, 0)
		val resultCallback = executionMap.remove(executionId)

		// @todo This should use constants
		val exitCode = resultBundle.getInt("exitCode") ?: 127;
		var stdout = resultBundle.getString("stdout") ?: "";
		val stderr = resultBundle.getString("stderr") ?: "";

		Log.d(LOG_TAG, "Execution id $executionId result:\n" +
				"errCode: " + exitCode.toString() + "\n" +
				"stdout:\n" + stdout + "\n" +
				"stderr:\n" + stderr + "\n")

		resultCallback?.call(exitCode, stdout, stderr)
	}
}
