/*************************************************************************/
/*  library_godot_eval.js                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

const GodotEval = {
	godot_js_eval__deps: ['$GodotRuntime'],
	godot_js_eval: function(p_js, p_use_global_ctx, p_union_ptr, p_byte_arr, p_byte_arr_write, p_callback) {
		const js_code = GodotRuntime.parseString(p_js);
		let eval_ret = null;
		try {
			if (p_use_global_ctx) {
				// indirect eval call grants global execution context
				const global_eval = eval; // eslint-disable-line no-eval
				eval_ret = global_eval(js_code);
			} else {
				eval_ret = eval(js_code); // eslint-disable-line no-eval
			}
		} catch (e) {
			GodotRuntime.error(e);
		}

		switch (typeof eval_ret) {

			case 'boolean':
				GodotRuntime.setHeapValue(p_union_ptr, eval_ret, 'i32');
				return 1; // BOOL

			case 'number':
				GodotRuntime.setHeapValue(p_union_ptr, eval_ret, 'double');
				return 3; // REAL

			case 'string':
				GodotRuntime.setHeapValue(p_union_ptr, GodotRuntime.allocString(eval_ret), '*');
				return 4; // STRING

			case 'object':
				if (eval_ret === null) {
					break;
				}

				if (ArrayBuffer.isView(eval_ret) && !(eval_ret instanceof Uint8Array)) {
					eval_ret = new Uint8Array(eval_ret.buffer);
				}
				else if (eval_ret instanceof ArrayBuffer) {
					eval_ret = new Uint8Array(eval_ret);
				}
				if (eval_ret instanceof Uint8Array) {
					const func = GodotRuntime.get_func(p_callback);
					const bytes_ptr = func(p_byte_arr, p_byte_arr_write,  eval_ret.length);
					HEAPU8.set(eval_ret, bytes_ptr);
					return 20; // POOL_BYTE_ARRAY
				}
				break;

			// no default
		}
		return 0; // NIL
	},
}

mergeInto(LibraryManager.library, GodotEval);
