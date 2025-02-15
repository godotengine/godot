/**************************************************************************/
/*  erase_array.h                                                         */
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

#ifndef ERASE_ARRAY_H
#define ERASE_ARRAY_H

#include "core/typedefs.h"

template <typename T, typename Predicate>
struct EraseArray {
	Predicate predicate;

	inline int64_t find_first(T *p_array, int64_t p_len) {
		for (int64_t i = 0; i < p_len; i++) {
			if (predicate(p_array[i])) {
				return i;
			}
		}
		return p_len;
	}

	inline int64_t erase(T *p_array, int64_t p_len) {
		int64_t first = find_first(p_array, p_len);
		if (first == p_len) {
			return p_len;
		}

		for (int64_t i = first + 1; i < p_len; i++) {
			if (!predicate(p_array[i])) {
				p_array[first++] = std::move(p_array[i]);
			}
		}
		return first; // new length
	}
};

#endif // ERASE_ARRAY_H
