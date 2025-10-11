/**************************************************************************/
/*  test_triangle2.h                                                      */
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

#include "modules/navigation_2d/triangle2.h"

#include "tests/test_macros.h"

namespace TestTriangle2 {
TEST_SUITE("[Triangle2]") {
	TEST_CASE("[Triangle2] Test get_area") {
		const Vector2 p0(5.0, 5.0);
		const Vector2 p1(6.0, 7.0);
		const Vector2 p2(7.0, 6.0);

		CHECK_EQ(Triangle2(p0, p1, p2).get_area(), doctest::Approx(1.5));
		CHECK_EQ(Triangle2(p0, p2, p1).get_area(), doctest::Approx(1.5));

		CHECK_EQ(Triangle2(p0, p2, p2).get_area(), doctest::Approx(0.0));
		CHECK_EQ(Triangle2(p0, p1, p1).get_area(), doctest::Approx(0.0));
	}

	TEST_CASE("[Triangle2] Test get_closest_point_to") {
		const Vector2 p0(5.0, 5.0);
		const Vector2 p1(6.0, 7.0);
		const Vector2 p2(7.0, 6.0);

		const Vector2 p3(0.0, 0.0);
		const Vector2 p4(6.0, 6.5);

		const Triangle2 t(p0, p1, p2);

		CHECK(t.get_closest_point_to(p0).is_equal_approx(p0));
		CHECK(t.get_closest_point_to(p1).is_equal_approx(p1));
		CHECK(t.get_closest_point_to(p2).is_equal_approx(p2));

		CHECK(t.get_closest_point_to(p3).is_equal_approx(p0));

		CHECK(t.get_closest_point_to(p4).is_equal_approx(p4));
	}
}
} // namespace TestTriangle2
