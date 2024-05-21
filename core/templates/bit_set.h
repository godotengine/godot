/**************************************************************************/
/*  bit_set.h                                                             */
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

#ifndef BIT_SET_H
#define BIT_SET_H

#include "core/templates/local_vector.h"

template <typename U = uint64_t, bool tight = false>
class BitSet {
private:
	LocalVector<U, uint32_t, false, tight> values;
	uint32_t count = 0;

	_FORCE_INLINE_ U _get_value_index(uint32_t p_index) const {
		return p_index / (sizeof(U) * 8);
	}

	_FORCE_INLINE_ U _get_value_mask(uint32_t p_index) const {
		return U(1) << (p_index % (sizeof(U) * 8));
	}

public:
	_FORCE_INLINE_ void clear() {
		values.clear();
		count = 0;
	}

	_FORCE_INLINE_ void resize(uint32_t p_size) {
		uint32_t previous_size = values.size();
		values.resize((p_size + (sizeof(U) * 8 - 1)) / (sizeof(U) * 8));

		uint32_t count_remainder = count % (sizeof(U) * 8);
		if (count < p_size && count_remainder != 0) {
			// The remaining bits of the value should be cleared.
			values[_get_value_index(count)] &= (U(1) << count_remainder) - U(1);
		}

		for (uint32_t i = previous_size; i < values.size(); i++) {
			// All the remaining values should be initialized to zero.
			values[i] = U(0);
		}

		count = p_size;
	}

	_FORCE_INLINE_ void set(uint32_t p_index, bool p_value) {
		CRASH_BAD_UNSIGNED_INDEX(p_index, count);

		if (p_value) {
			values[_get_value_index(p_index)] |= _get_value_mask(p_index);
		} else {
			values[_get_value_index(p_index)] &= ~_get_value_mask(p_index);
		}
	}

	_FORCE_INLINE_ bool get(uint32_t p_index) const {
		CRASH_BAD_UNSIGNED_INDEX(p_index, count);

		return (values[_get_value_index(p_index)] & _get_value_mask(p_index)) != 0;
	}

	_FORCE_INLINE_ uint32_t size() const {
		return count;
	}

	_FORCE_INLINE_ BitSet() {}
};

#endif // BIT_SET_H
