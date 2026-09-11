/**************************************************************************/
/*  test_triangulate.cpp                                                  */
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

#include "core/math/triangulate.h"
#include "tests/test_macros.h"

TEST_FORCE_LINK(test_triangulate)

namespace TestTriangulate {

TEST_CASE("[Triangulate] Get area of a triangle") {
	Vector<Vector2> contour;
	contour.push_back(Vector2(0, 0));
	contour.push_back(Vector2(4, 0));
	contour.push_back(Vector2(0, 3));

	real_t area = Triangulate::get_area(contour);

	CHECK(area == doctest::Approx(6.0));
}

TEST_CASE("[Triangulate] Point inside triangle") {
	CHECK(Triangulate::is_inside_triangle(
			0.0, 0.0,
			4.0, 0.0,
			0.0, 4.0,
			1.0, 1.0,
			true));
}

} // namespace TestTriangulate
