/**************************************************************************/
/*  callable_method_pointer.cpp                                           */
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

#include "callable_method_pointer.h"

bool CallableCustomMethodPointerBase::compare_equal(const CallableCustom *p_a, const CallableCustom *p_b) {
	const CallableCustomMethodPointerBase *a = static_cast<const CallableCustomMethodPointerBase *>(p_a);
	const CallableCustomMethodPointerBase *b = static_cast<const CallableCustomMethodPointerBase *>(p_b);

	if (a->comp_size != b->comp_size) {
		return false;
	}

	// Avoid sorting by memory address proximity, which leads to unpredictable performance over time
	// due to the reuse of old addresses for newer objects. Use byte-wise comparison to leverage the
	// backwards encoding of little-endian systems as a way to decouple spatiality and time.
	return memcmp(a->comp_ptr, b->comp_ptr, a->comp_size * 4) == 0;
}

bool CallableCustomMethodPointerBase::compare_less(const CallableCustom *p_a, const CallableCustom *p_b) {
	const CallableCustomMethodPointerBase *a = static_cast<const CallableCustomMethodPointerBase *>(p_a);
	const CallableCustomMethodPointerBase *b = static_cast<const CallableCustomMethodPointerBase *>(p_b);

	if (a->comp_size != b->comp_size) {
		return a->comp_size < b->comp_size;
	}

	// See note in compare_equal().
	return memcmp(a->comp_ptr, b->comp_ptr, a->comp_size * 4) < 0;
}

CallableCustom::CompareEqualFunc CallableCustomMethodPointerBase::get_compare_equal_func() const {
	return compare_equal;
}

CallableCustom::CompareLessFunc CallableCustomMethodPointerBase::get_compare_less_func() const {
	return compare_less;
}

uint32_t CallableCustomMethodPointerBase::hash() const {
	return h;
}

void CallableCustomMethodPointerBase::_setup(uint32_t *p_base_ptr, uint32_t p_ptr_size) {
	comp_ptr = p_base_ptr;
	comp_size = p_ptr_size / 4;

	// Precompute hash.
	for (uint32_t i = 0; i < comp_size; i++) {
		if (i == 0) {
			h = hash_murmur3_one_32(comp_ptr[i]);
		} else {
			h = hash_murmur3_one_32(comp_ptr[i], h);
		}
	}
}
