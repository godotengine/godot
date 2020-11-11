/*************************************************************************/
/*  javascript_eval.cpp                                                  */
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

#ifdef JAVASCRIPT_EVAL_ENABLED

#include "api/javascript_eval.h"
#include "emscripten.h"

extern "C" {
union js_eval_ret {
	uint32_t b;
	double d;
	char *s;
};

extern int godot_js_eval(const char *p_js, int p_use_global_ctx, union js_eval_ret *p_union_ptr, void *p_byte_arr, void *p_byte_arr_write, void *(*p_callback)(void *p_ptr, void *p_ptr2, int p_len));
}

void *resize_poolbytearray_and_open_write(void *p_arr, void *r_write, int p_len) {

	PoolByteArray *arr = (PoolByteArray *)p_arr;
	PoolByteArray::Write *write = (PoolByteArray::Write *)r_write;
	arr->resize(p_len);
	*write = arr->write();
	return write->ptr();
}

Variant JavaScript::eval(const String &p_code, bool p_use_global_exec_context) {

	PoolByteArray arr;
	PoolByteArray::Write arr_write;
	union js_eval_ret js_data;
	memset(&js_data, 0, sizeof(js_data));
	Variant::Type return_type = static_cast<Variant::Type>(godot_js_eval(p_code.utf8().get_data(), p_use_global_exec_context, &js_data, &arr, &arr_write, resize_poolbytearray_and_open_write));

	switch (return_type) {
		case Variant::BOOL:
			return js_data.b == 1;
		case Variant::REAL:
			return js_data.d;
		case Variant::STRING: {
			String str = String::utf8(js_data.s);
			free(js_data.s); // Must free the string allocated in JS.
			return str;
		}
		case Variant::POOL_BYTE_ARRAY:
			arr_write = PoolByteArray::Write();
			return arr;
		default:
			return Variant();
	}
}

#endif // JAVASCRIPT_EVAL_ENABLED
