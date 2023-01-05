/*************************************************************************/
/*  test_packed_array.h                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef TEST_PACKED_ARRAY_H
#define TEST_PACKED_ARRAY_H

#include "core/variant/variant.h"
#include "tests/test_macros.h"

namespace TestPacked_array {

TEST_CASE("[PackedArray] simple fill and refill") {
	PackedInt32Array array;

	for (int32_t i = 0; i < 12345; i++) {
		array.push_back(i);
	}
	CHECK_MESSAGE(
			array.size() == 12345,
			"PackedInt32Array should have 12345 elements.");

	bool all_match = true;
	for (int32_t i = 0; i < 12345; i++) {
		if (array[i] != i) {
			all_match = false;
			break;
		}
	}

	CHECK_MESSAGE(
			all_match,
			"PackedInt32 elements should match from 0 to 12344.");

	array.clear();

	CHECK_MESSAGE(
			array.size() == 0,
			"PackedInt32 elements should be 0 after clear.");

	for (int32_t i = 0; i < 999; i++) {
		array.push_back(i);
	}
	CHECK_MESSAGE(
			array.size() == 999,
			"PackedInt32 should have 999 elements.");

	all_match = true;
	for (int32_t i = 0; i < 999; i++) {
		if (array[i] != i) {
			all_match = false;
		}
	}

	CHECK_MESSAGE(
			all_match,
			"PackedInt32 elements should match from 0 to 998.");
}

} // namespace TestPacked_array

#endif // TEST_PACKED_ARRAY_H
