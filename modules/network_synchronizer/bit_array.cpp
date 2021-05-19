/*************************************************************************/
/*  bit_array.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "bit_array.h"

#include "core/math/math_funcs.h"
#include "core/string/ustring.h"

BitArray::BitArray(uint32_t p_initial_size_in_bit) {
	resize_in_bits(p_initial_size_in_bit);
}

BitArray::BitArray(const Vector<uint8_t> &p_bytes) :
		bytes(p_bytes) {
}

void BitArray::resize_in_bytes(int p_bytes) {
	bytes.resize(p_bytes);
}

int BitArray::size_in_bytes() const {
	return bytes.size();
}

void BitArray::resize_in_bits(int p_bits) {
	const int min_size = Math::ceil((static_cast<float>(p_bits)) / 8);
	bytes.resize(min_size);
}

int BitArray::size_in_bits() const {
	return bytes.size() * 8;
}

void BitArray::store_bits(int p_bit_offset, uint64_t p_value, int p_bits) {
	ERR_FAIL_COND_MSG(p_bit_offset + p_bits > size_in_bits(), "The bit array is smaller than the bits that you are trying to write.");

	int bits = p_bits;
	int bit_offset = p_bit_offset;
	uint64_t val = p_value;

	while (bits > 0) {
		const int bits_to_write = MIN(bits, 8 - bit_offset % 8);
		const int bits_to_jump = bit_offset % 8;
		const int bits_to_skip = 8 - (bits_to_write + bits_to_jump);
		const int byte_offset = bit_offset / 8;

		// Clear the bits that we have to write
		//const uint8_t byte_clear = ~(((0xFF >> bits_to_jump) << (bits_to_jump + bits_to_skip)) >> bits_to_skip);
		uint8_t byte_clear = 0xFF >> bits_to_jump;
		byte_clear = byte_clear << (bits_to_jump + bits_to_skip);
		byte_clear = ~(byte_clear >> bits_to_skip);
		bytes.write[byte_offset] &= byte_clear;

		// Now we can continue to write bits
		bytes.write[byte_offset] |= (val & 0xFF) << bits_to_jump;

		bits -= bits_to_write;
		bit_offset += bits_to_write;

		val >>= bits_to_write;
	}
}

uint64_t BitArray::read_bits(int p_bit_offset, int p_bits) const {
	ERR_FAIL_COND_V_MSG(p_bit_offset + p_bits > size_in_bits(), 0, "The bit array size is `" + itos(size_in_bits()) + "` while you are trying to read `" + itos(p_bits) + "` starting from `" + itos(p_bit_offset) + "`.");

	int bits = p_bits;
	int bit_offset = p_bit_offset;
	uint64_t val = 0;

	const uint8_t *bytes_ptr = bytes.ptr();

	int val_bits_to_jump = 0;
	while (bits > 0) {
		const int bits_to_read = MIN(bits, 8 - bit_offset % 8);
		const int bits_to_jump = bit_offset % 8;
		const int bits_to_skip = 8 - (bits_to_read + bits_to_jump);
		const int byte_offset = bit_offset / 8;

		uint8_t byte_mask = 0xFF >> bits_to_jump;
		byte_mask = byte_mask << (bits_to_skip + bits_to_jump);
		byte_mask = byte_mask >> bits_to_skip;
		const uint64_t byte_val = static_cast<uint64_t>((bytes_ptr[byte_offset] & byte_mask) >> bits_to_jump);
		val |= byte_val << val_bits_to_jump;

		bits -= bits_to_read;
		bit_offset += bits_to_read;
		val_bits_to_jump += bits_to_read;
	}

	return val;
}

void BitArray::zero() {
	if (bytes.size() > 0) {
		memset(bytes.ptrw(), 0, sizeof(uint8_t) * bytes.size());
	}
}
