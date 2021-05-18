/*************************************************************************/
/*  test_vector2.h                                                       */
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

#ifndef TEST_VECTOR2_H
#define TEST_VECTOR2_H

#include "core/math/vector2.h"

namespace TestVector2 {
// Vector2 and Vector2i are tested here as they are both
// defined in the same file.

TEST_CASE("[Vector2] Constructor methods") {
	const Vector2 vector_empty = Vector2();
	const Vector2 vector_xy = Vector2(0.0, 0.0);

	CHECK_MESSAGE(
			vector_empty == vector_xy,
			"Different methods should yield the same Vector2.");
}

TEST_CASE("[Vector2] String representation") {
	const Vector2 vector_a = Vector2(18543, 85222);
	const Vector2 vector_c = Vector2();

	CHECK_MESSAGE(
			(vector_a == "(18543, 85222)" || vector_a == "18543, 85222"),
			"String representation must match expected value.");

	CHECK_MESSAGE(
			(vector_c == "(0, 0)" || vector_c == "0, 0"),
			"String representation must match expected value.");
}

TEST_CASE("[Vector2] Angle") {
	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(0, 0).angle(), 0),
			"angle() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(1.5, 0).angle(), 0),
			"angle() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(-1.5, 0).angle(), 3.14159),
			"angle() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(0, 1.5).angle(), 1.5708),
			"angle() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(0, -1.5).angle(), -1.5708),
			"angle() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(1.5, 1.5).angle(), 0.785398),
			"angle() should return expected value.");
}

TEST_CASE("[Vector2] Length") {
	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(0, 0).length(), 0),
			"length() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(1.5, 0).length(), 1.5),
			"length() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(0, -1.5).length(), 1.5),
			"length() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(1.5, 1.5).length(), 2.12132),
			"length() should return expected value.");
}

TEST_CASE("[Vector2] Aspect") {
	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(0, -1.5).aspect(), 0),
			"aspect() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(1.5, 1.5).aspect(), 1),
			"aspect() should return expected value.");
}

TEST_CASE("[Vector2] Length Squared") {
	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(0, 0).length_squared(), 0),
			"length_squared() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(1.5, 0).length_squared(), 2.25),
			"length_squared() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(0, -1.5).length_squared(), 2.25),
			"length_squared() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(1.5, 1.5).length_squared(), 4.5),
			"length_squared() should return expected value.");
}

TEST_CASE("[Vector2] Normalize, and Normalized") {
	// Normalized wraps around normalize and returns
	// the normalized vector

	Vector2 test_subject_a = Vector2(3.2, -5.4);
	Vector2 test_subject_b = Vector2(3.2, -5.4);

	test_subject_a.normalize();
	CHECK_MESSAGE(
			test_subject_a == test_subject_b.normalized(),
			"the same Vector normalized in different methods should return the same value.");

	CHECK_MESSAGE(
			test_subject_a.is_equal_approx(Vector2(0.509802, -0.860291)),
			"normalize() should return the expected value.");

	CHECK_MESSAGE(
			test_subject_b.normalized().is_equal_approx(Vector2(0.509802, -0.860291)),
			"normalized() should return the expected value.");
}

TEST_CASE("[Vector2] Is Normalized") {
	CHECK_MESSAGE(
			Vector2(0.707107, 0.707107).is_normalized(),
			"is_normalized() should return `true` for a unit vector.");

	CHECK_MESSAGE(
			Vector2(1, 0).is_normalized(),
			"is_normalized() should return `true` for a unit vector.");

	CHECK_MESSAGE(
			Vector2(0, -1).is_normalized(),
			"is_normalized() should return `true` for a unit vector.");

	CHECK_MESSAGE(
			!Vector2(1, 1).is_normalized(),
			"is_normalized() should return `false` for a non-unit vector.");

	CHECK_MESSAGE(
			!Vector2(-1.5, -1.5).is_normalized(),
			"is_normalized() should return `false` for a non-unit vector.");

	CHECK_MESSAGE(
			!Vector2(35.2, -2.9).is_normalized(),
			"is_normalized() should return `false` for a non-unit vector.");

	CHECK_MESSAGE(
			!Vector2(-5.2, 2.9).is_normalized(),
			"is_normalized() should return `false` for a non-unit vector.");
}

TEST_CASE("[Vector2] Distance To") {
	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(0.4356, 1.3455).distance_to(Vector2(2.4555, 2.5966)), 2.37597),
			"distance_to() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(3.556, 2.5913).distance_to(Vector2(10.5990, 0)), 7.50458),
			"distance_to() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(0, 0).distance_to(Vector2(6.9420, 213.4900)), 213.60283),
			"distance_to() should return expected value.");
}

TEST_CASE("[Vector2] Distance Squared To") {
	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(0.4356, 1.3455).distance_squared_to(Vector2(2.4555, 2.5966)), 5.64523),
			"distance_squared_to() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(3.556, 2.5913).distance_squared_to(Vector2(10.5990, 0)), 56.31872),
			"distance_squared_to() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(0, 0).distance_squared_to(Vector2(6.9420, 213.4900)), 45626.1689),
			"distance_squared_to() should return expected value.");
}

TEST_CASE("[Vector2] Angle To") {
	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(1.5493, -0.2344).angle_to(Vector2(2.4555, 2.5966)), 0.963475),
			"angle_to() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(0, 0).angle_to(Vector2(1.5, 0)), 0),
			"angle_to() should return expected value.");
}

TEST_CASE("[Vector2] Angle To Point") {
	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(1.250, 2.3499).angle_to_point(Vector2(1.2340, 2.3045)), 1.23196),
			"angle_to_point() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(0, 1.5).angle_to_point(Vector2(3.2995, 2.3045)), -2.90243),
			"angle_to_point() should return expected value.");
}

TEST_CASE("[Vector2] Dot") {
	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(3.5418, 8.5152).dot(Vector2(5.1864, 4.6373)), 57.85672),
			"dot() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(-3.5418, 8.5152).dot(Vector2(5.1864, 4.6373)), 21.11834),
			"dot() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(-3.5418, -8.5152).dot(Vector2(5.1864, 4.6373)), -57.85672),
			"dot() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(-3.5418, -8.5152).dot(Vector2(-5.1864, 4.6373)), -21.11834),
			"dot() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(-3.5418, -8.5152).dot(Vector2(-5.1864, -4.6373)), 57.85672),
			"dot() should return expected value.");
}

TEST_CASE("[Vector2] Cross") {
	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(3.5418, 8.5152).cross(Vector2(5.1864, 4.6373)), -27.73884),
			"cross() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(-3.5418, 8.5152).cross(Vector2(5.1864, 4.6373)), -60.58764),
			"cross() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(-3.5418, -8.5152).cross(Vector2(5.1864, 4.6373)), 27.73884),
			"cross() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(-3.5418, -8.5152).cross(Vector2(-5.1864, 4.6373)), -60.58762),
			"cross() should return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(Vector2(-3.5418, -8.5152).cross(Vector2(-5.1864, -4.6373)), -27.73884),
			"cross() should return expected value.");
}

TEST_CASE("[Vector2] Sign") {
	CHECK_MESSAGE(
			Vector2(1, 1).is_equal_approx(Vector2(3.5418, 8.5152).sign()),
			"sign() should return expected value.");

	CHECK_MESSAGE(
			Vector2(-1, 1).is_equal_approx(Vector2(-3.5418, 8.5152).sign()),
			"sign() should return expected value.");

	CHECK_MESSAGE(
			Vector2(1, -1).is_equal_approx(Vector2(3.5418, -8.5152).sign()),
			"sign() should return expected value.");

	CHECK_MESSAGE(
			Vector2(-1, -1).is_equal_approx(Vector2(-3.5418, -8.5152).sign()),
			"sign() should return expected value.");
}

TEST_CASE("[Vector2] Floor") {
	CHECK_MESSAGE(
			Vector2(3, 8).is_equal_approx(Vector2(3.5418, 8.5152).floor()),
			"floor() should return expected value.");

	CHECK_MESSAGE(
			Vector2(-4, 8).is_equal_approx(Vector2(-3.5418, 8.5152).floor()),
			"floor() should return expected value.");

	CHECK_MESSAGE(
			Vector2(3, -9).is_equal_approx(Vector2(3.5418, -8.5152).floor()),
			"floor() should return expected value.");

	CHECK_MESSAGE(
			Vector2(-4, -9).is_equal_approx(Vector2(-3.5418, -8.5152).floor()),
			"floor() should return expected value.");

	CHECK_MESSAGE(
			Vector2(5, 4).is_equal_approx(Vector2(5.1864, 4.6373).floor()),
			"floor() should return expected value.");
}

TEST_CASE("[Vector2] Ceil") {
	CHECK_MESSAGE(
			Vector2(4, 9).is_equal_approx(Vector2(3.5418, 8.5152).ceil()),
			"ceil() should return expected value.");

	CHECK_MESSAGE(
			Vector2(-3, 9).is_equal_approx(Vector2(-3.5418, 8.5152).ceil()),
			"ceil() should return expected value.");

	CHECK_MESSAGE(
			Vector2(4, -8).is_equal_approx(Vector2(3.5418, -8.5152).ceil()),
			"ceil() should return expected value.");

	CHECK_MESSAGE(
			Vector2(-3, -8).is_equal_approx(Vector2(-3.5418, -8.5152).ceil()),
			"ceil() should return expected value.");

	CHECK_MESSAGE(
			Vector2(6, 5).is_equal_approx(Vector2(5.1864, 4.6373).ceil()),
			"ceil() should return expected value.");
}

TEST_CASE("[Vector2] Round") {
	CHECK_MESSAGE(
			Vector2(4, 9).is_equal_approx(Vector2(3.5418, 8.5152).round()),
			"round() should return expected value.");

	CHECK_MESSAGE(
			Vector2(-4, 9).is_equal_approx(Vector2(-3.5418, 8.5152).round()),
			"round() should return expected value.");

	CHECK_MESSAGE(
			Vector2(4, -9).is_equal_approx(Vector2(3.5418, -8.5152).round()),
			"round() should return expected value.");

	CHECK_MESSAGE(
			Vector2(-4, -9).is_equal_approx(Vector2(-3.5418, -8.5152).round()),
			"round() should return expected value.");

	CHECK_MESSAGE(
			Vector2(5, 5).is_equal_approx(Vector2(5.1864, 4.6373).round()),
			"round() should return expected value.");
}

TEST_CASE("[Vector2] Rotated") {
	CHECK_MESSAGE(
			Vector2(-8.51521, 3.54176).is_equal_approx(Vector2(3.5418, 8.5152).rotated(1.5708)),
			"rotated() should return expected value.");

	CHECK_MESSAGE(
			Vector2(-3.54182, -8.51519).is_equal_approx(Vector2(3.5418, 8.5152).rotated(3.14159)),
			"rotated() should return expected value.");

	CHECK_MESSAGE(
			Vector2(8.51520, -3.54179).is_equal_approx(Vector2(3.5418, 8.5152).rotated(4.71239)),
			"rotated() should return expected value.");

	CHECK_MESSAGE(
			Vector2(3.54176, 8.51521).is_equal_approx(Vector2(3.5418, 8.5152).rotated(6.28319)),
			"rotated() should return expected value.");
}

TEST_CASE("[Vector2] Posmod") {
	CHECK_MESSAGE(
			Vector2(0.4002, 0.6612).is_equal_approx(Vector2(3.5418, 8.5152).posmod(1.5708)),
			"posmod() should return expected value.");

	CHECK_MESSAGE(
			Vector2(0.40020, 2.23202).is_equal_approx(Vector2(3.5418, 8.5152).posmod(3.14159)),
			"posmod() should return expected value.");
}

TEST_CASE("[Vector2] Posmodv") {
	CHECK_MESSAGE(
			Vector2(3.5418, 1.10420).is_equal_approx(Vector2(3.5418, 8.5152).posmodv(Vector2(5.4820, 1.4822))),
			"posmodv() should return expected value.");

	CHECK_MESSAGE(
			Vector2(1.9402, 1.10420).is_equal_approx(Vector2(-3.5418, 8.5152).posmodv(Vector2(5.4820, 1.4822))),
			"posmodv() should return expected value.");

	CHECK_MESSAGE(
			Vector2(3.5418, 0.378).is_equal_approx(Vector2(3.5418, -8.5152).posmodv(Vector2(5.4820, 1.4822))),
			"posmodv() should return expected value.");

	CHECK_MESSAGE(
			Vector2(1.9402, 0.378).is_equal_approx(Vector2(-3.5418, -8.5152).posmodv(Vector2(5.4820, 1.4822))),
			"posmodv() should return expected value.");
}

TEST_CASE("[Vector2] Project") {
	CHECK_MESSAGE(
			Vector2(5.44599, 1.47246).is_equal_approx(Vector2(3.5418, 8.5152).project(Vector2(5.4820, 1.4822))),
			"project() should return expected value.");

	CHECK_MESSAGE(
			Vector2(-1.15506, -0.3123).is_equal_approx(Vector2(-3.5418, 8.5152).project(Vector2(5.4820, 1.4822))),
			"project() should return expected value.");

	CHECK_MESSAGE(
			Vector2(1.15506, 0.3123).is_equal_approx(Vector2(3.5418, -8.5152).project(Vector2(5.4820, 1.4822))),
			"project() should return expected value.");

	CHECK_MESSAGE(
			Vector2(-5.44599, -1.47246).is_equal_approx(Vector2(-3.5418, -8.5152).project(Vector2(5.4820, 1.4822))),
			"project() should return expected value.");
}

TEST_CASE("[Vector2] Snapped") {
	CHECK_MESSAGE(
			Vector2(5.482, 8.8932).is_equal_approx(Vector2(3.5418, 8.5152).snapped(Vector2(5.4820, 1.4822))),
			"snapped() should return expected value.");

	CHECK_MESSAGE(
			Vector2(-5.482, 8.8932).is_equal_approx(Vector2(-3.5418, 8.5152).snapped(Vector2(5.4820, 1.4822))),
			"snapped() should return expected value.");

	CHECK_MESSAGE(
			Vector2(5.482, -8.8932).is_equal_approx(Vector2(3.5418, -8.5152).snapped(Vector2(5.4820, 1.4822))),
			"snapped() should return expected value.");

	CHECK_MESSAGE(
			Vector2(-5.482, -8.8932).is_equal_approx(Vector2(-3.5418, -8.5152).snapped(Vector2(5.4820, 1.4822))),
			"snapped() should return expected value.");
}

TEST_CASE("[Vector2] Clamped") {
	CHECK_MESSAGE(
			Vector2(1.41421, 1.41421).is_equal_approx(Vector2(1.5, 1.5).clamped(2)),
			"clamped() should return expected value.");

	CHECK_MESSAGE(
			Vector2(-0.353553, -0.353553).is_equal_approx(Vector2(-1.5, -1.5).clamped(0.5)),
			"clamped() should return expected value.");
}

TEST_CASE("[Vector2] Cubic Interpolate") {
	CHECK_MESSAGE(
			Vector2(1.78125, 1.78125).is_equal_approx(Vector2(1.5, 1.5).cubic_interpolate(Vector2(2, 2), Vector2(0.5, 0.5), Vector2(2.5, 2.5), 0.5)),
			"cubic_interpolate() should return expected value.");

	CHECK_MESSAGE(
			Vector2(2, 2).is_equal_approx(Vector2(1.5, 1.5).cubic_interpolate(Vector2(2, 2), Vector2(0.5, 0.5), Vector2(2.5, 2.5), 1)),
			"cubic_interpolate() should return expected value.");
}

TEST_CASE("[Vector2] Move Toward") {
	CHECK_MESSAGE(
			Vector2(1.85355, 1.85355).is_equal_approx(Vector2(1.5, 1.5).move_toward(Vector2(3, 3), 0.5)),
			"move_toward() should return expected value.");

	CHECK_MESSAGE(
			Vector2(2.2064, 2.2064).is_equal_approx(Vector2(1.5, 1.5).move_toward(Vector2(3, 3), 0.999)),
			"move_toward() should return expected value.");
}

TEST_CASE("[Vector2] Slide") {
	CHECK_MESSAGE(
			Vector2(0, 1.5).is_equal_approx(Vector2(1.5, 1.5).slide(Vector2(1, 0))),
			"slide() should return expected value.");

	ERR_PRINT_OFF;
	// There's probably a better way to test these ones
	CHECK_MESSAGE(
			Vector2().is_equal_approx(Vector2(1.5, 1.5).slide(Vector2(3, 3))),
			"slide() shouldn't burst into flames if a none normalized parameter is given.");
	ERR_PRINT_ON;
}

TEST_CASE("[Vector2] Reflect") {
	CHECK_MESSAGE(
			Vector2(1.5, -1.5).is_equal_approx(Vector2(1.5, 1.5).reflect(Vector2(1, 0))),
			"reflect() should return expected value.");

	ERR_PRINT_OFF;
	CHECK_MESSAGE(
			Vector2().is_equal_approx(Vector2(1.5, 1.5).reflect(Vector2(3, 3))),
			"reflect() shouldn't burst into flames if a none normalized parameter is given.");
	ERR_PRINT_ON;
}

TEST_CASE("[Vector2] Bounce") {
	CHECK_MESSAGE(
			Vector2(-1.5, 1.5).is_equal_approx(Vector2(1.5, 1.5).bounce(Vector2(1, 0))),
			"bounce() should return expected value.");

	ERR_PRINT_OFF;
	CHECK_MESSAGE(
			Vector2().is_equal_approx(Vector2(1.5, 1.5).bounce(Vector2(3, 3))),
			"bounce() shouldn't burst into flames if a none normalized parameter is given.");
	ERR_PRINT_ON;
}

TEST_CASE("[Vector2] Is Equal Approx") {
	CHECK_MESSAGE(
			Vector2(1.5, 1.5).is_equal_approx(Vector2(1.5, 1.5)),
			"is_equal_approx() should return expected value.");

	CHECK_FALSE_MESSAGE(
			Vector2(-1.45, 1.45).is_equal_approx(Vector2(1.5, 1.5)),
			"is_equal_approx() should return expected value.");
}

TEST_CASE("[Vector2] Operators") {
	const Vector2 test_subject_a = Vector2(1.5, 2.5);
	Vector2 test_subject_b = Vector2(0, 1.0);

	CHECK_MESSAGE(
			test_subject_a[0] == 1.5,
			"operator[] should return expected value.");

	CHECK_MESSAGE(
			test_subject_a[1] == 2.5,
			"operator[] should return expected value.");

	CHECK_MESSAGE(
			Vector2(3, 5).is_equal_approx(test_subject_a * 2),
			"operator* should return expected value.");

	CHECK_MESSAGE(
			Vector2(3.75, 6.25).is_equal_approx(test_subject_a * 2.5),
			"operator* should return expected value.");

	CHECK_MESSAGE(
			Vector2(0, 2.5).is_equal_approx(test_subject_a * test_subject_b),
			"operator* should return expected value.");

	CHECK_MESSAGE(
			Vector2(1.5, 3.5).is_equal_approx(test_subject_a + test_subject_b),
			"operator+ should return expected value.");

	test_subject_b += test_subject_a;
	CHECK_MESSAGE(
			Vector2(1.5, 3.5).is_equal_approx(test_subject_b),
			"operator+= should return expected value.");
	// test_subject_b is now (1.5, 3.5)

	CHECK_MESSAGE(
			Vector2(0, -1).is_equal_approx(test_subject_a - test_subject_b),
			"operator- should return expected value.");

	test_subject_b -= test_subject_a;
	CHECK_MESSAGE(
			Vector2(0, 1).is_equal_approx(test_subject_b),
			"operator-= should return expected value.");
	// test_subject_b is now (0, 1)

	test_subject_b *= test_subject_a;
	CHECK_MESSAGE(
			Vector2(0, 2.5).is_equal_approx(test_subject_b),
			"operator*= should return expected value.");
	// test_subject_b is now (0, 2.5)

	CHECK_MESSAGE(
			Vector2(0, 1).is_equal_approx(test_subject_b / test_subject_a),
			"operator/ should return expected value.");

	CHECK_MESSAGE(
			Vector2(0, 5).is_equal_approx(test_subject_b / 0.5),
			"operator/ should return expected value.");

	test_subject_b /= test_subject_a;
	CHECK_MESSAGE(
			Vector2(0, 1).is_equal_approx(test_subject_b),
			"operator/= should return expected value.");
	// test_subject_b is now (0, 1)

	CHECK_MESSAGE(
			Vector2(-1.5, -2.5).is_equal_approx(-test_subject_a),
			"operator- should return expected value.");

	CHECK_MESSAGE(
			test_subject_a == Vector2(1.5, 2.5),
			"operator== should return expected value.");

	CHECK_FALSE_MESSAGE(
			test_subject_a == test_subject_b,
			"operator== should return expected value.");

	CHECK_MESSAGE(
			test_subject_a != test_subject_b,
			"operator!= should return expected value.");

	CHECK_FALSE_MESSAGE(
			test_subject_a != Vector2(1.5, 2.5),
			"operator!= should return expected value.");
}

TEST_CASE("[Vector2] Min Max") {
	const Vector2 test_subject_a = Vector2(3.0, 4.0);
	const Vector2 test_subject_b = Vector2(5.3, 2.5);
	CHECK_MESSAGE(
			Vector2(3.0, 2.5).is_equal_approx(test_subject_a.min(test_subject_b)),
			"min() should return expected value.");

	CHECK_MESSAGE(
			Vector2(5.3, 4.0).is_equal_approx(test_subject_a.max(test_subject_b)),
			"max() should return expected value.");
}

TEST_CASE("[Vector2] Direction To") {
	CHECK_MESSAGE(
			Vector2(0, 0).is_equal_approx(Vector2(1, 1).direction_to(Vector2(1, 1))),
			"direction_to() should return expected value.");

	CHECK_MESSAGE(
			Vector2(-0.7071, -0.7071).is_equal_approx(Vector2(1, 1).direction_to(Vector2(-1, -1))),
			"direction_to() should return expected value.");
}

TEST_CASE("[Vector2] Lerp") {
	CHECK_MESSAGE(
			Vector2(0, 0.5).is_equal_approx(Vector2(1, 0).lerp(Vector2(-1, 1), 0.5)),
			"lerp() should return expected value.");

	CHECK_MESSAGE(
			Vector2(-0.8, 0.9).is_equal_approx(Vector2(1, 0).lerp(Vector2(-1, 1), 0.9)),
			"lerp() should return expected value.");
}

TEST_CASE("[Vector2] Slerp") {
	CHECK_MESSAGE(
			Vector2(0.382683, 0.92388).is_equal_approx(Vector2(1, 0).slerp(Vector2(-1, 1), 0.5)),
			"slerp() should return expected value.");

	CHECK_MESSAGE(
			Vector2(-0.522498, 0.85264).is_equal_approx(Vector2(1, 0).slerp(Vector2(-1, 1), 0.9)),
			"slerp() should return expected value.");

	// Probably a better way to test this
	ERR_PRINT_OFF;
	CHECK_MESSAGE(
			Vector2().is_equal_approx(Vector2(1.5, 1.5).slerp(Vector2(3, 3), 1)),
			"slerp() shouldn't burst into flames if a none normalized parameter is given.");
	ERR_PRINT_ON;
}

TEST_CASE("[Vector2] Abs") {
	CHECK_MESSAGE(
			Vector2(1, 1).is_equal_approx(Vector2(-1, -1).abs()),
			"abs() should return expected value.");
}

/* Vector2i */

TEST_CASE("[Vector2][Vector2i] Constructor methods") {
	CHECK_MESSAGE(
			Vector2i() == Vector2i(0, 0),
			"Different methods should yield the same Vector2i.");

	CHECK_MESSAGE(
			Vector2i(Vector2(2.5, 2.5)) == Vector2i(2, 2),
			"Different methods should yield the same Vector2i.");
}

TEST_CASE("[Vector2][Vector2i] Sign") {
	CHECK_MESSAGE(
			Vector2(1, 1).is_equal_approx(Vector2i(3, 5).sign()),
			"sign() should return expected value.");

	CHECK_MESSAGE(
			Vector2(-1, 1).is_equal_approx(Vector2i(-3, 5).sign()),
			"sign() should return expected value.");

	CHECK_MESSAGE(
			Vector2(-1, -1).is_equal_approx(Vector2i(-3, -5).sign()),
			"sign() should return expected value.");

	CHECK_MESSAGE(
			Vector2(0, 0).is_equal_approx(Vector2i(0, 0).sign()),
			"sign() should return expected value.");
}

TEST_CASE("[Vector2][Vector2i] Abs") {
	CHECK_MESSAGE(
			Vector2(3, 5).is_equal_approx(Vector2i(-3, 5).abs()),
			"abs() should return expected value.");

	CHECK_MESSAGE(
			Vector2(3, 5).is_equal_approx(Vector2i(-3, -5).abs()),
			"abs() should return expected value.");

	CHECK_MESSAGE(
			Vector2(0, 0).is_equal_approx(Vector2i(0, 0).abs()),
			"abs() should return expected value.");
}

TEST_CASE("[Vector2][Vector2i] Operators") {
	const Vector2i test_subject_a = Vector2i(1, 2);
	Vector2i test_subject_b = Vector2i(0, 1);

	CHECK_MESSAGE(
			test_subject_a[0] == 1,
			"operator[] should return expected value.");

	CHECK_MESSAGE(
			test_subject_a[1] == 2,
			"operator[] should return expected value.");

	CHECK_MESSAGE(
			Vector2(2, 4).is_equal_approx(test_subject_a * 2),
			"operator* should return expected value.");

	CHECK_MESSAGE(
			Vector2(0, 2).is_equal_approx(test_subject_a * test_subject_b),
			"operator* should return expected value.");

	CHECK_MESSAGE(
			Vector2(1, 3).is_equal_approx(test_subject_a + test_subject_b),
			"operator+ should return expected value.");

	test_subject_b += test_subject_a;
	CHECK_MESSAGE(
			Vector2(1, 3).is_equal_approx(test_subject_b),
			"operator+= should return expected value.");
	// test_subject_b is now (1, 3)

	CHECK_MESSAGE(
			Vector2(0, -1).is_equal_approx(test_subject_a - test_subject_b),
			"operator- should return expected value.");

	test_subject_b -= test_subject_a;
	CHECK_MESSAGE(
			Vector2(0, 1).is_equal_approx(test_subject_b),
			"operator-= should return expected value.");
	// test_subject_b is now (0, 1)

	test_subject_b *= 2;
	CHECK_MESSAGE(
			Vector2(0, 2).is_equal_approx(test_subject_b),
			"operator*= should return expected value.");
	// test_subject_b is now (0, 2)

	CHECK_MESSAGE(
			Vector2(0, 1).is_equal_approx(test_subject_b / test_subject_a),
			"operator/ should return expected value.");

	test_subject_b /= 2;
	CHECK_MESSAGE(
			Vector2(0, 1).is_equal_approx(test_subject_b),
			"operator/= should return expected value.");
	// test_subject_b is now (0, 1)

	CHECK_MESSAGE(
			Vector2(-1, -2).is_equal_approx(-test_subject_a),
			"operator- should return expected value.");

	CHECK_MESSAGE(
			test_subject_a == Vector2(1, 2),
			"operator== should return expected value.");

	CHECK_FALSE_MESSAGE(
			test_subject_a == test_subject_b,
			"operator== should return expected value.");

	CHECK_MESSAGE(
			test_subject_a != test_subject_b,
			"operator!= should return expected value.");

	CHECK_FALSE_MESSAGE(
			test_subject_a != Vector2(1, 2),
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

} // namespace TestVector2

#endif // TEST_VECTOR2_H
