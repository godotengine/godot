/**************************************************************************/
/*  set_offscreen_canvas_size.js                                          */
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

// This is only needed if the offscreen canvas is being used on the main thread.
// We are only calling it from JS, so it's fine that wasm is already imported the old version.
// https://github.com/emscripten-core/emscripten/issues/26394
const original_emscripten_set_canvas_element_size = _emscripten_set_canvas_element_size;
_emscripten_set_canvas_element_size = (target, width, height) => {
	const result = original_emscripten_set_canvas_element_size(target, width, height);
	const canvasRecord = GL.offscreenCanvases[GodotRuntime.parseString(target).slice(1)];
	if (canvasRecord && canvasRecord.canvas) {
		const canvas = canvasRecord.canvas;
		canvas.width = width;
		canvas.height = height;
		if (canvas.GLctxObject) {
			GL.resizeOffscreenFramebuffer(canvas.GLctxObject);
		}
	}
	return result;
};
