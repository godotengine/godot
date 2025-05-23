/**************************************************************************/
/*  char_buffer.h                                                         */
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

#pragma once

#include "core/templates/local_vector.h"

template <typename T, size_t FIXED_BUFFER_SIZE = 64>
class CharBuffer {
	T _fixed_buffer[FIXED_BUFFER_SIZE];
	LocalVector<T> _dynamic_buffer;
	size_t _length = 0;
	size_t _capacity = FIXED_BUFFER_SIZE;

	// Size should include room for null terminator
	void _reserve(size_t p_size) {
		if (p_size <= _capacity) {
			return;
		}

		bool copy = _dynamic_buffer.is_empty() && _length > 0;
		_capacity = next_power_of_2(p_size);
		_dynamic_buffer.resize(_capacity);
		if (copy) {
			memcpy(_dynamic_buffer.ptr(), _fixed_buffer, _length * sizeof(T));
		}
	}

	T *_get_current_buffer() {
		return _dynamic_buffer.is_empty() ? _fixed_buffer : _dynamic_buffer.ptr();
	}

public:
	_FORCE_INLINE_ void operator+=(T p_char) { append(p_char); }
	CharBuffer &append(T p_char) {
		_reserve(_length + 2);
		_get_current_buffer()[_length++] = p_char;
		return *this;
	}

	const T *get_terminated_buffer() {
		T *current_buffer = _get_current_buffer();
		current_buffer[_length] = '\0';
		return current_buffer;
	}
};
