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

#include "core/string/ustring.h"

template <typename T, uint32_t FIXED_BUFFER_SIZE = 64>
class CharBuffer {
	T _fixed_buffer[FIXED_BUFFER_SIZE];
	T *_active_buffer = _fixed_buffer;

	using StringType = std::conditional_t<std::is_same_v<T, char32_t>, String, CharStringT<T>>;
	StringType _string;

	uint32_t _length = 0;
	uint32_t _capacity = FIXED_BUFFER_SIZE;

	// Size should include room for null terminator.
	void _reserve(uint32_t p_size) {
		if (p_size <= _capacity) {
			return;
		}

		bool copy = _active_buffer == _fixed_buffer && _length > 0;
		_capacity = next_power_of_2(p_size);
		_string.reserve_exact(_capacity);
		_active_buffer = const_cast<T *>(_string.ptr());
		if (copy) {
			memcpy(_active_buffer, _fixed_buffer, _length * sizeof(T));
		}
	}

public:
	_FORCE_INLINE_ void operator+=(T p_char) { append(p_char); }
	CharBuffer &append(T p_char) {
		_reserve(_length + 2);
		_active_buffer[_length++] = p_char;
		return *this;
	}

	// Returns pointer to null-terminated string without resetting state.
	const T *get_terminated_buffer() {
		_active_buffer[_length] = '\0';
		return _active_buffer;
	}

	// Returns string and resets state.
	StringType finalize() {
		StringType result;

		if (_active_buffer == _fixed_buffer) {
			result = get_terminated_buffer();
		} else {
			_active_buffer[_length] = '\0';
			result = std::move(_string);
			result.resize_uninitialized(_length + 1);

			_string = StringType();
			_active_buffer = _fixed_buffer;
			_capacity = FIXED_BUFFER_SIZE;
		}

		_length = 0;
		return result;
	}
};
