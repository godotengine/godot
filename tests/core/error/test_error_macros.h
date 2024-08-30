/**************************************************************************/
/*  test_error_macros.h                                                   */
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

#ifndef TEST_ERROR_MACROS_H
#define TEST_ERROR_MACROS_H

#include "core/error/error_macros.h"

#include "tests/test_macros.h"

#include <cstdint>

TEST_CASE("[Error macros] ___gd_is_index_out_of_bounds macro returns expected values") {
	// ===============
	// Inbound checks.
	// ===============
	int in_a_index = 0;
	int in_a_size = 5;
	CHECK(___gd_is_index_out_of_bounds(in_a_index, in_a_size) == false);
	int in_b_index = 343214;
	int in_b_size = 943928;
	CHECK(___gd_is_index_out_of_bounds(in_b_index, in_b_size) == false);
	uint8_t in_c_index = 0;
	uint16_t in_c_size = 1;
	CHECK(___gd_is_index_out_of_bounds(in_c_index, in_c_size) == false);
	uint32_t in_d_index = 343214;
	uint64_t in_d_size = 943928;
	CHECK(___gd_is_index_out_of_bounds(in_d_index, in_d_size) == false);
	int in_e_index = 0;
	unsigned int in_e_size = 1;
	CHECK(___gd_is_index_out_of_bounds(in_e_index, in_e_size) == false);
	unsigned int in_f_index = 0;
	int in_f_size = 1;
	CHECK(___gd_is_index_out_of_bounds(in_f_index, in_f_size) == false);

	// =====================
	// Out of bounds checks.
	// =====================
	// Size is 0.
	int out_a_index = 0;
	int out_a_size = 0;
	CHECK(___gd_is_index_out_of_bounds(out_a_index, out_a_size) == true);
	// Index is negative.
	int out_b_index = -1;
	int out_b_size = 1;
	CHECK(___gd_is_index_out_of_bounds(out_b_index, out_b_size) == true);
	// Size is negative.
	int out_c_index = 1;
	int out_c_size = -1;
	CHECK(___gd_is_index_out_of_bounds(out_c_index, out_c_size) == true);
	// Both are negative.
	int out_d_index = -1;
	int out_d_size = -1;
	CHECK(___gd_is_index_out_of_bounds(out_d_index, out_d_size) == true);
	// Index is bigger than size.
	int out_e_index = 127;
	int out_e_size = 126;
	CHECK(___gd_is_index_out_of_bounds(out_e_index, out_e_size) == true);
	// Index would be bigger than size if index was read as `uint8_t`.
	int8_t out_f_index = -1;
	uint8_t out_f_size = 1;
	CHECK(___gd_is_index_out_of_bounds(out_f_index, out_f_size) == true);
	// This would be alright if size was read as `uint8_t`.
	uint8_t out_g_index = 1;
	int8_t out_g_size = -1;
	CHECK(___gd_is_index_out_of_bounds(out_g_index, out_g_size) == true);
}

#endif // TEST_ERROR_MACROS_H
