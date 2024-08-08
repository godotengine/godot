/**************************************************************************/
/*  test_projection.h                                                     */
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

#ifndef TEST_PROJECTION_H
#define TEST_PROJECTION_H

#include "core/math/projection.h"

#include "core/string/print_string.h"
#include "tests/test_macros.h"

namespace TestProjection {

TEST_CASE("[Projection] Default construct") {
	Projection p;
	CHECK(p.columns[0][0] == 1.0);
	CHECK(p.columns[0][1] == 0.0);
	CHECK(p.columns[0][2] == 0.0);
	CHECK(p.columns[0][3] == 0.0);

	CHECK(p.columns[1][0] == 0.0);
	CHECK(p.columns[1][1] == 1.0);
	CHECK(p.columns[1][2] == 0.0);
	CHECK(p.columns[1][3] == 0.0);

	CHECK(p.columns[2][0] == 0.0);
	CHECK(p.columns[2][1] == 0.0);
	CHECK(p.columns[2][2] == 1.0);
	CHECK(p.columns[2][3] == 0.0);

	CHECK(p.columns[3][0] == 0.0);
	CHECK(p.columns[3][1] == 0.0);
	CHECK(p.columns[3][2] == 0.0);
	CHECK(p.columns[3][3] == 1.0);
}

bool projection_is_equal_approx(const Projection &p_a, const Projection &p_b) {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			if (!Math::is_equal_approx(p_a.columns[i][j], p_b.columns[i][j])) {
				return false;
			}
		}
	}
	return true;
}

TEST_CASE("[Projection] Orthogonal projection matrix inversion") {
	Projection p = Projection::create_orthogonal(-125.0f, 125.0f, -125.0f, 125.0f, 0.01f, 25.0f);
	CHECK(projection_is_equal_approx(p.inverse() * p, Projection()));
}

TEST_CASE("[Projection] Perspective projection matrix inversion") {
	Projection p = Projection::create_perspective(90.0f, 1.77777f, 0.05f, 4000.0f);
	CHECK(projection_is_equal_approx(p.inverse() * p, Projection()));
}

} //namespace TestProjection

#endif // TEST_PROJECTION_H
