/*************************************************************************/
/*  javascript_eval.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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

extern "C" EMSCRIPTEN_KEEPALIVE uint8_t *resize_poolbytearray_and_open_write(PoolByteArray *p_arr, PoolByteArray::Write *r_write, int p_len) {

	p_arr->resize(p_len);
	*r_write = p_arr->write();
	return r_write->ptr();
}

Variant JavaScript::eval(const String &p_code, bool p_use_global_exec_context) {

	union {
		bool b;
		double d;
		char *s;
	} js_data[4];

	PoolByteArray arr;
	PoolByteArray::Write arr_write;

	/* clang-format off */
	Variant::Type return_type = static_cast<Variant::Type>(EM_ASM_INT({

		const CODE = $0;
		const USE_GLOBAL_EXEC_CONTEXT = $1;
		const PTR = $2;
		const ELEM_LEN = $3;
		const BYTEARRAY_PTR = $4;
		const BYTEARRAY_WRITE_PTR = $5;
		var eval_ret;
		try {
			if (USE_GLOBAL_EXEC_CONTEXT) {
				// indirect eval call grants global execution context
				var global_eval = eval;
				eval_ret = global_eval(UTF8ToString(CODE));
			} else {
				eval_ret = eval(UTF8ToString(CODE));
			}
		} catch (e) {
			Module.printErr(e);
			eval_ret = null;
		}

		switch (typeof eval_ret) {

			case 'boolean':
				setValue(PTR, eval_ret, 'i32');
				return 1; // BOOL

			case 'number':
				setValue(PTR, eval_ret, 'double');
				return 3; // REAL

			case 'string':
				var array_len = lengthBytesUTF8(eval_ret)+1;
				var array_ptr = _malloc(array_len);
				try {
					if (array_ptr===0) {
						throw new Error('String allocation failed (probably out of memory)');
					}
					setValue(PTR, array_ptr , '*');
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

				if (ArrayBuffer.isView(eval_ret) && !(eval_ret instanceof Uint8Array)) {
					eval_ret = new Uint8Array(eval_ret.buffer);
				}
				else if (eval_ret instanceof ArrayBuffer) {
					eval_ret = new Uint8Array(eval_ret);
				}
				if (eval_ret instanceof Uint8Array) {
					var bytes_ptr = ccall('resize_poolbytearray_and_open_write', 'number', ['number', 'number' ,'number'], [BYTEARRAY_PTR, BYTEARRAY_WRITE_PTR, eval_ret.length]);
					HEAPU8.set(eval_ret, bytes_ptr);
					return 20; // POOL_BYTE_ARRAY
				}

				if (typeof eval_ret.x==='number' && typeof eval_ret.y==='number') {
					setValue(PTR, eval_ret.x, 'double');
					setValue(PTR + ELEM_LEN, eval_ret.y, 'double');
					if (typeof eval_ret.z==='number') {
						setValue(PTR + ELEM_LEN*2, eval_ret.z, 'double');
						return 7; // VECTOR3
					}
					else if (typeof eval_ret.width==='number' && typeof eval_ret.height==='number') {
						setValue(PTR + ELEM_LEN*2, eval_ret.width, 'double');
						setValue(PTR + ELEM_LEN*3, eval_ret.height, 'double');
						return 6; // RECT2
					}
					return 5; // VECTOR2
				}

				if (typeof eval_ret.r === 'number' && typeof eval_ret.g === 'number' && typeof eval_ret.b === 'number') {
					setValue(PTR, eval_ret.r, 'double');
					setValue(PTR + ELEM_LEN, eval_ret.g, 'double');
					setValue(PTR + ELEM_LEN*2, eval_ret.b, 'double');
					setValue(PTR + ELEM_LEN*3, typeof eval_ret.a === 'number' ? eval_ret.a : 1, 'double');
					return 14; // COLOR
				}
				break;
		}
		return 0; // NIL

	}, p_code.utf8().get_data(), p_use_global_exec_context, js_data, sizeof *js_data, &arr, &arr_write));
	/* clang-format on */

	switch (return_type) {
		case Variant::BOOL:
			return js_data->b;
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
			return Color(js_data[0].d, js_data[1].d, js_data[2].d, js_data[3].d);
		case Variant::POOL_BYTE_ARRAY:
			arr_write = PoolByteArray::Write();
			return arr;
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
