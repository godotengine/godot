/*************************************************************************/
/*  test_vector3i.h                                                      */
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

#ifndef TEST_VECTOR3I_H
#define TEST_VECTOR3I_H

#include "core/math/basis.h"
#include "core/math/math_funcs.h"
#include "core/math/vector3i.h"

namespace TestVector3i {
// Only Vector3i is tested here as its defined on it's own file.

TEST_CASE("[Vector3i] Constructor methods") {
	const Vector3i vector_empty = Vector3i();
	const Vector3i vector_xyz = Vector3i(0, 0, 0);

	CHECK_MESSAGE(
			vector_empty == vector_xyz,
			"Different methods should yield the same Vector3i.");
}

TEST_CASE("[Vector3i] String representation") {
	const Vector3i vector_a = Vector3i(1, 2, 3);
	const Vector3i vector_c = Vector3i();

	CHECK_MESSAGE(
			vector_a == "(1, 2, 3)",
			"String representation must match expected value.");

	CHECK_MESSAGE(
			vector_c == "(0, 0, 0)",
			"String representation must match expected value.");
}

TEST_CASE("[Vector3i] Set Axis, Get Axis, Min Axis, Max Axis") {
	Vector3i vector_set = Vector3i();
	const Vector3i vector_a = Vector3i(1, 0, 0);
	const Vector3i vector_b = Vector3i(1, 2, 0);
	const Vector3i vector_c = Vector3i(1, 2, 3);

	vector_set.set_axis(0, 1);
	// vector_set is now (1, 0, 0)

	CHECK_MESSAGE(
			vector_set == (vector_a),
			"Vectors with the same axis should be equal.");

	CHECK_MESSAGE(
			Math::is_equal_approx(vector_set.get_axis(0), 1),
			"Axis must contain expected value.");

	vector_set.set_axis(1, 2);
	// vector_set is now (1, 2, 0)

	CHECK_MESSAGE(
			vector_set == (vector_b),
			"Vectors with the same axis should be equal.");

	CHECK_MESSAGE(
			Math::is_equal_approx(vector_set.get_axis(1), 2),
			"Axis must contain expected value.");

	vector_set.set_axis(2, 3);
	// vector_set is now (1, 1, 1)

	CHECK_MESSAGE(
			vector_set == (vector_c),
			"Vectors with the same axis should be equal.");

	CHECK_MESSAGE(
			Math::is_equal_approx(vector_set.get_axis(2), 3),
			"Axis must contain expected value.");

	// Min Max Axis

	CHECK_MESSAGE(
			Math::is_equal_approx(vector_set.min_axis(), 0),
			"min_axis() must return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(vector_set.max_axis(), 2),
			"max_axis() must contain expected value.");
}

TEST_CASE("[Vector3i] Zero") {
	Vector3i test_subject = Vector3i(1, 2, 3);
	test_subject.zero();

	CHECK_MESSAGE(
			Vector3i() == (test_subject),
			"zero() must return expected value.");
}

TEST_CASE("[Vector3i] Abs") {
	const Vector3i test_subject_a = Vector3i(1, 2, 3);
	const Vector3i test_subject_b = Vector3i(-1, 3, -2);
	const Vector3i test_subject_c = Vector3i(3, -2, 1);

	CHECK_MESSAGE(
			Vector3i(1, 2, 3) == (test_subject_a.abs()),
			"abs() must return expected value.");

	CHECK_MESSAGE(
			Vector3i(1, 3, 2) == (test_subject_b.abs()),
			"abs() must return expected value.");

	CHECK_MESSAGE(
			Vector3i(3, 2, 1) == (test_subject_c.abs()),
			"abs() must return expected value.");
}

TEST_CASE("[Vector3i] Sign") {
	const Vector3i test_subject_a = Vector3i(1, 2, 3);
	const Vector3i test_subject_b = Vector3i(-1, 3, -2);
	const Vector3i test_subject_c = Vector3i(3, -2, 1);

	CHECK_MESSAGE(
			Vector3i(1, 1, 1) == (test_subject_a.sign()),
			"sign() must return expected value.");

	CHECK_MESSAGE(
			Vector3i(-1, 1, -1) == (test_subject_b.sign()),
			"sign() must return expected value.");

	CHECK_MESSAGE(
			Vector3i(1, -1, 1) == (test_subject_c.sign()),
			"sign() must return expected value.");
}

TEST_CASE("[Vector3i] Operators") {
	const Vector3i test_subject_a = Vector3i(1, 2, 3);
	Vector3i test_subject_b = Vector3i(0, 1, 4);

	CHECK_MESSAGE(
			test_subject_a[0] == 1,
			"operator[] should return expected value.");

	CHECK_MESSAGE(
			test_subject_a[1] == 2,
			"operator[] should return expected value.");

	CHECK_MESSAGE(
			Vector3i(2, 4, 6) == (test_subject_a * 2),
			"operator* should return expected value.");

	CHECK_MESSAGE(
			Vector3i(0, 2, 12) == (test_subject_a * test_subject_b),
			"operator* should return expected value.");

	CHECK_MESSAGE(
			Vector3i(1, 3, 7) == (test_subject_a + test_subject_b),
			"operator+ should return expected value.");

	test_subject_b += test_subject_a;
	// test_subject_b is now (1, 3, 7)
	CHECK_MESSAGE(
			Vector3i(1, 3, 7) == (test_subject_b),
			"operator+= should return expected value.");

	CHECK_MESSAGE(
			Vector3i(0, -1, -4) == (test_subject_a - test_subject_b),
			"operator- should return expected value.");

	test_subject_b -= test_subject_a;
	// test_subject_b is now (0, 1, 4)
	CHECK_MESSAGE(
			Vector3i(0, 1, 4) == (test_subject_b),
			"operator-= should return expected value.");

	test_subject_b *= 2;
	// test_subject_b is now (0, 2, 8)
	CHECK_MESSAGE(
			Vector3i(0, 2, 8) == (test_subject_b),
			"operator*= should return expected value.");

	CHECK_MESSAGE(
			Vector3i(0, 1, 2) == (test_subject_b / test_subject_a),
			"operator/ should return expected value.");

	test_subject_b /= 2;
	CHECK_MESSAGE(
			Vector3i(0, 1, 4) == (test_subject_b),
			"operator/= should return expected value.");
	// test_subject_b is now (0, 1, 4)

	CHECK_MESSAGE(
			Vector3i(-1, -2, -3) == (-test_subject_a),
			"operator- should return expected value.");

	CHECK_MESSAGE(
			Vector3i(0, 1, 1) == (test_subject_b % test_subject_a),
			"operator% should return expected value.");

	CHECK_MESSAGE(
			Vector3i(0, 1, 1) == (test_subject_b %= test_subject_a),
			"operator% should return expected value.");

	CHECK_MESSAGE(
			test_subject_a == Vector3i(1, 2, 3),
			"operator== should return expected value.");

	CHECK_FALSE_MESSAGE(
			test_subject_a == test_subject_b,
			"operator== should return expected value.");

	CHECK_MESSAGE(
			test_subject_a != test_subject_b,
			"operator!= should return expected value.");

	CHECK_FALSE_MESSAGE(
			test_subject_a != Vector3i(1, 2, 3),
			"operator!= should return expected value.");

	CHECK_MESSAGE(
			test_subject_a > test_subject_b,
			"operator> should return expected value.");

	CHECK_MESSAGE(
			test_subject_a >= test_subject_b,
			"operator> should return expected value.");

	CHECK_FALSE_MESSAGE(
			test_subject_a < test_subject_b,
			"operator< should return expected value.");

	CHECK_FALSE_MESSAGE(
			test_subject_a <= test_subject_b,
			"operator< should return expected value.");
}

} // namespace TestVector3i

#endif // TEST_VECTOR3I_H
