/**************************************************************************/
/*  library_godot_runtime.js                                              */
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

const GodotRuntime = {
	$GodotRuntime: {
		/*
		 * Functions
		 */
		get_func: (ptr) => {
			return wasmTable.get(ptr);
		},

		/*
		 * Prints
		 */
		error: (...args) => {
			err.apply(null, Array.from(args));
		},

		print: (...args) => {
			out.apply(null, Array.from(args));
		},

		/*
		 * Memory
		 */
		malloc: (p_size) => {
			return _malloc(p_size);
		},

		free: (p_ptr) => {
			_free(p_ptr);
		},

		getHeapValue: (p_ptr, p_type) => {
			return getValue(p_ptr, p_type);
		},

		setHeapValue: (p_ptr, p_value, p_type) => {
			setValue(p_ptr, p_value, p_type);
		},

		heapSub: (p_heap, p_ptr, p_len) => {
			const bytes = p_heap.BYTES_PER_ELEMENT;
			return p_heap.subarray(p_ptr / bytes, p_ptr / bytes + p_len);
		},

		heapSlice: (p_heap, p_ptr, p_len) => {
			const bytes = p_heap.BYTES_PER_ELEMENT;
			return p_heap.slice(p_ptr / bytes, p_ptr / bytes + p_len);
		},

		heapCopy: (p_dst, p_src, p_ptr) => {
			const bytes = p_src.BYTES_PER_ELEMENT;
			return p_dst.set(p_src, p_ptr / bytes);
		},

		/*
		 * Strings
		 */
		parseString: (p_ptr) => {
			return UTF8ToString(p_ptr);
		},

		parseStringArray: (p_ptr, p_size) => {
			const strings = [];
			const ptrs = GodotRuntime.heapSub(HEAP32, p_ptr, p_size); // TODO wasm64
			for (const ptr of ptrs) {
				strings.push(GodotRuntime.parseString(ptr));
			}
			return strings;
		},

		strlen: (p_str) => {
			return lengthBytesUTF8(p_str);
		},

		allocString: (p_str) => {
			const length = GodotRuntime.strlen(p_str) + 1;
			const c_str = GodotRuntime.malloc(length);
			stringToUTF8(p_str, c_str, length);
			return c_str;
		},

		allocStringArray: (p_strings) => {
			const size = p_strings.length;
			const c_ptr = GodotRuntime.malloc(size * 4);
			for (let i = 0; i < size; i++) {
				HEAP32[(c_ptr >> 2) + i] = GodotRuntime.allocString(p_strings[i]);
			}
			return c_ptr;
		},

		freeStringArray: (p_ptr, p_len) => {
			for (let i = 0; i < p_len; i++) {
				GodotRuntime.free(HEAP32[(p_ptr >> 2) + i]);
			}
			GodotRuntime.free(p_ptr);
		},

		stringToHeap: (p_str, p_ptr, p_len) => {
			return stringToUTF8Array(p_str, HEAP8, p_ptr, p_len);
		},
	},
};
autoAddDeps(GodotRuntime, "$GodotRuntime");
mergeInto(LibraryManager.library, GodotRuntime);
