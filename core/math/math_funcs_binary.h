/**************************************************************************/
/*  math_funcs_binary.h                                                   */
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

#include "core/typedefs.h"

namespace Math {

/* Functions to handle powers of 2 and shifting. */

// Returns `true` if a positive integer is a power of 2, `false` otherwise.
template <typename T>
inline bool is_power_of_2(const T x) {
	return x && ((x & (x - 1)) == 0);
}

// Function to find the next power of 2 to an integer.
constexpr uint64_t next_power_of_2(uint64_t p_number) {
	if (p_number == 0) {
		return 0;
	}

	--p_number;
	p_number |= p_number >> 1;
	p_number |= p_number >> 2;
	p_number |= p_number >> 4;
	p_number |= p_number >> 8;
	p_number |= p_number >> 16;
	p_number |= p_number >> 32;

	return ++p_number;
}

constexpr uint32_t next_power_of_2(uint32_t p_number) {
	if (p_number == 0) {
		return 0;
	}

	--p_number;
	p_number |= p_number >> 1;
	p_number |= p_number >> 2;
	p_number |= p_number >> 4;
	p_number |= p_number >> 8;
	p_number |= p_number >> 16;

	return ++p_number;
}

// Function to find the previous power of 2 to an integer.
constexpr uint64_t previous_power_of_2(uint64_t p_number) {
	p_number |= p_number >> 1;
	p_number |= p_number >> 2;
	p_number |= p_number >> 4;
	p_number |= p_number >> 8;
	p_number |= p_number >> 16;
	p_number |= p_number >> 32;
	return p_number - (p_number >> 1);
}

constexpr uint32_t previous_power_of_2(uint32_t p_number) {
	p_number |= p_number >> 1;
	p_number |= p_number >> 2;
	p_number |= p_number >> 4;
	p_number |= p_number >> 8;
	p_number |= p_number >> 16;
	return p_number - (p_number >> 1);
}

// Function to find the closest power of 2 to an integer.
constexpr uint64_t closest_power_of_2(uint64_t p_number) {
	uint64_t nx = next_power_of_2(p_number);
	uint64_t px = previous_power_of_2(p_number);
	return (nx - p_number) > (p_number - px) ? px : nx;
}

constexpr uint32_t closest_power_of_2(uint32_t p_number) {
	uint32_t nx = next_power_of_2(p_number);
	uint32_t px = previous_power_of_2(p_number);
	return (nx - p_number) > (p_number - px) ? px : nx;
}

// Get a shift value from a power of 2.
constexpr int32_t get_shift_from_power_of_2(uint64_t p_bits) {
	for (uint64_t i = 0; i < (uint64_t)64; i++) {
		if (p_bits == (uint64_t)((uint64_t)1 << i)) {
			return i;
		}
	}

	return -1;
}

constexpr int32_t get_shift_from_power_of_2(uint32_t p_bits) {
	for (uint32_t i = 0; i < (uint32_t)32; i++) {
		if (p_bits == (uint32_t)((uint32_t)1 << i)) {
			return i;
		}
	}

	return -1;
}

template <typename T>
_FORCE_INLINE_ T nearest_power_of_2_templated(T p_number) {
	--p_number;

	// The number of operations on x is the base two logarithm
	// of the number of bits in the type. Add three to account
	// for sizeof(T) being in bytes.
	constexpr size_t shift_steps = get_shift_from_power_of_2((uint64_t)sizeof(T)) + 3;

	// If the compiler is smart, it unrolls this loop.
	// If it's dumb, this is a bit slow.
	for (size_t i = 0; i < shift_steps; i++) {
		p_number |= p_number >> (1 << i);
	}

	return ++p_number;
}

// Function to find the nearest (bigger) power of 2 to an integer.
constexpr uint64_t nearest_shift(uint64_t p_number) {
	uint64_t i = 63;
	do {
		i--;
		if (p_number & ((uint64_t)1 << i)) {
			return i + (uint64_t)1;
		}
	} while (i != 0);

	return 0;
}

constexpr uint32_t nearest_shift(uint32_t p_number) {
	uint32_t i = 31;
	do {
		i--;
		if (p_number & ((uint32_t)1 << i)) {
			return i + (uint32_t)1;
		}
	} while (i != 0);

	return 0;
}

// constexpr function to find the floored log2 of a number
template <typename T>
constexpr T floor_log2(T x) {
	return x < 2 ? x : 1 + floor_log2(x >> 1);
}

// Get the number of bits needed to represent the number.
// IE, if you pass in 8, you will get 4.
// If you want to know how many bits are needed to store 8 values however, pass in (8 - 1).
template <typename T>
constexpr T get_num_bits(T x) {
	return floor_log2(x);
}

} //namespace Math
