/*************************************************************************/
/*  javascript_eval.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifdef JAVASCRIPT_EVAL_ENABLED

#include "javascript_eval.h"
#include "emscripten.h"

JavaScript *JavaScript::singleton = NULL;

JavaScript *JavaScript::get_singleton() {

	return singleton;
}

Variant JavaScript::eval(const String &p_code, bool p_use_global_exec_context) {

	union {
		int i;
		double d;
		char *s;
	} js_data[4];
	/* clang-format off */
	Variant::Type return_type = static_cast<Variant::Type>(EM_ASM_INT({

		var eval_ret;
		try {
			if ($3) { // p_use_global_exec_context
				// indirect eval call grants global execution context
				var global_eval = eval;
				eval_ret = global_eval(UTF8ToString($2));
			} else {
				eval_ret = eval(UTF8ToString($2));
			}
		} catch (e) {
			Module.printErr(e);
			eval_ret = null;
		}

		switch (typeof eval_ret) {

			case 'boolean':
				// bitwise op yields 32-bit int
				setValue($0, eval_ret|0, 'i32');
				return 1; // BOOL

			case 'number':
				if ((eval_ret|0)===eval_ret) {
					setValue($0, eval_ret|0, 'i32');
					return 2; // INT
				}
				setValue($0, eval_ret, 'double');
				return 3; // REAL

			case 'string':
				var array_len = lengthBytesUTF8(eval_ret)+1;
				var array_ptr = _malloc(array_len);
				try {
					if (array_ptr===0) {
						throw new Error('String allocation failed (probably out of memory)');
					}
					setValue($0, array_ptr|0 , '*');
					stringToUTF8(eval_ret, array_ptr, array_len);
					return 4; // STRING
				} catch (e) {
					if (array_ptr!==0) {
						_free(array_ptr)
					}
					Module.printErr(e);
					// fall through
				}
				break;

			case 'object':
				if (eval_ret === null) {
					break;
				}

				else if (typeof eval_ret.x==='number' && typeof eval_ret.y==='number') {
					setValue($0, eval_ret.x, 'double');
					setValue($0+$1, eval_ret.y, 'double');
					if (typeof eval_ret.z==='number') {
						setValue($0+$1*2, eval_ret.z, 'double');
						return 7; // VECTOR3
					}
					else if (typeof eval_ret.width==='number' && typeof eval_ret.height==='number') {
						setValue($0+$1*2, eval_ret.width, 'double');
						setValue($0+$1*3, eval_ret.height, 'double');
						return 6; // RECT2
					}
					return 5; // VECTOR2
				}

				else if (typeof eval_ret.r==='number' && typeof eval_ret.g==='number' && typeof eval_ret.b==='number') {
					// assume 8-bit rgb components since we're on the web
					setValue($0, eval_ret.r, 'double');
					setValue($0+$1, eval_ret.g, 'double');
					setValue($0+$1*2, eval_ret.b, 'double');
					setValue($0+$1*3, typeof eval_ret.a==='number' ? eval_ret.a : 1, 'double');
					return 14; // COLOR
				}
				break;
		}
		return 0; // NIL

	}, js_data, sizeof *js_data, p_code.utf8().get_data(), p_use_global_exec_context));
	/* clang-format on */

	switch (return_type) {
		case Variant::BOOL:
			return !!js_data->i;
		case Variant::INT:
			return js_data->i;
		case Variant::REAL:
			return js_data->d;
		case Variant::STRING: {
			String str = String::utf8(js_data->s);
			/* clang-format off */
				EM_ASM_({ _free($0); }, js_data->s);
			/* clang-format on */
			return str;
		}
		case Variant::VECTOR2:
			return Vector2(js_data[0].d, js_data[1].d);
		case Variant::VECTOR3:
			return Vector3(js_data[0].d, js_data[1].d, js_data[2].d);
		case Variant::RECT2:
			return Rect2(js_data[0].d, js_data[1].d, js_data[2].d, js_data[3].d);
		case Variant::COLOR:
			return Color(js_data[0].d / 255., js_data[1].d / 255., js_data[2].d / 255., js_data[3].d);
	}
	return Variant();
}

void JavaScript::_bind_methods() {

	ClassDB::bind_method(D_METHOD("eval", "code", "use_global_execution_context"), &JavaScript::eval, false);
}

JavaScript::JavaScript() {

	ERR_FAIL_COND(singleton != NULL);
	singleton = this;
}

JavaScript::~JavaScript() {
}

#endif // JAVASCRIPT_EVAL_ENABLED
