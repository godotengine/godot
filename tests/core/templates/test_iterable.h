/**************************************************************************/
/*  test_iterable.h                                                       */
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

#include "core/templates/iterable.h"

#include "tests/test_macros.h"

namespace TestZip {
TEST_CASE("[zip] Basic tests") {
	constexpr uint16_t a[5] = { 1, 2, 3, 4, 5 };
	constexpr uint32_t b[5] = { 2, 3, 4, 5, 6 };

	size_t i = 0;
	for (auto [ai, bi] : zip_shortest<uint16_t, uint32_t>(a, b)) {
		CHECK_EQ(ai, i + 1);
		CHECK_EQ(bi, i + 2);
		i++;
	}
	CHECK_EQ(i, 5);
}
} //namespace TestZip

namespace TestEnumerate {

TEST_CASE("[zip] Enumerate tests") {
	constexpr uint32_t a[5] = { 1, 2, 3, 4, 5 };

	size_t i = 0;
	for (auto [idx, ai] : enumerate<size_t, uint32_t>(a)) {
		CHECK_EQ(idx, i);
		CHECK_EQ(ai, i + 1);
		i++;
	}
	CHECK_EQ(i, 5);
}

} //namespace TestEnumerate
