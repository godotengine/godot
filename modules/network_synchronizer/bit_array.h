/*************************************************************************/
/*  bit_array.h                                                          */
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

/**
	@author AndreaCatania
*/

#include "core/templates/vector.h"

#ifndef BITARRAY_H
#define BITARRAY_H

class BitArray {
	Vector<uint8_t> bytes;

public:
	BitArray() {}
	BitArray(uint32_t p_initial_size_in_bit);
	BitArray(Vector<uint8_t> p_bytes);

	const Vector<uint8_t> &get_bytes() const {
		return bytes;
	}

	Vector<uint8_t> &get_bytes_mut() {
		return bytes;
	}

	void resize_in_bytes(int p_bits);
	int size_in_bytes() const;

	void resize_in_bits(int p_bits);
	int size_in_bits() const;

	void store_bits(int p_bit_offset, uint64_t p_value, int p_bits);
	uint64_t read_bits(int p_bit_offset, int p_bits) const;

	// Puts all the bytes to 0.
	void zero();
};

#endif
