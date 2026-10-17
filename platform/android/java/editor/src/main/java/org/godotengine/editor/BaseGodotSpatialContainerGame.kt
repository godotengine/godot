/*************************************************************************/
/*  BaseGodotSpatialContainerGame.kt                                     */
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

package org.godotengine.editor

import android.util.Log
import org.godotengine.godot.GodotLib

/**
 * Provide support for running spatial container apps / games from the editor window.
 */
abstract class BaseGodotSpatialContainerGame : GodotXRGame() {
	companion object {
		private val TAG = BaseGodotSpatialContainerGame::class.java.simpleName

		private val DEFAULT_SPATIAL_CONTAINER_BOUNDS = Triple(0.5f, 0.5f, 0.5f)

		@JvmStatic
		protected fun isSpatialContainerEnabled() = GodotLib.getGlobal("xr/openxr/extensions/spatial_container/enabled").toBoolean()

		internal fun getSpatialContainerBounds(): Triple<Float, Float, Float> {
			val spatialContainerBounds = GodotLib.getGlobal("xr/openxr/extensions/spatial_container/bounds")
			if (!spatialContainerBounds.isNullOrBlank()) {
				// The returned string has the following format: (x, y, z)
				val splitComponents = spatialContainerBounds
					.trim() // remove any leading / trailing whitespaces
					.trim('(', ')') // remove the leading and trailing parentheses
					.split(',')
				if (splitComponents.size == 3) {
					try {
						val bounds = Triple(
							splitComponents[0].toFloat(),
							splitComponents[1].toFloat(),
							splitComponents[2].toFloat()
						)
						return bounds
					} catch (_: NumberFormatException) {}
				}
				Log.w(TAG, "Error parsing spatial container bounds $spatialContainerBounds")
			}
			return DEFAULT_SPATIAL_CONTAINER_BOUNDS
		}
	}
}
