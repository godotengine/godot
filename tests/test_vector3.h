/*************************************************************************/
/*  test_vector3.h                                                       */
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

#ifndef TEST_VECTOR3_H
#define TEST_VECTOR3_H

#include "core/math/basis.h"
#include "core/math/math_funcs.h"
#include "core/math/vector3.h"
#include "core/math/vector3i.h"

namespace TestVector3 {
// Only Vector3 is tested here as Vector3i is defined in another file.

TEST_CASE("[Vector3] Constructor methods") {
	const Vector3 vector_empty = Vector3();
	const Vector3 vector_xy = Vector3(0.0, 0.0, 0.0);
	const Vector3 vector_i = Vector3(Vector3i(0, 0, 0));
	const Vector3i vector_3i = Vector3i(0, 0, 0);

	CHECK_MESSAGE(
			vector_empty == vector_xy,
			"Different methods should yield the same Vector3.");

	CHECK_MESSAGE(
			vector_i == vector_xy,
			"Different methods should yield the same Vector3.");

	CHECK_MESSAGE(
			vector_i == vector_3i,
			"Different methods should yield the same Vector3i.");
}

TEST_CASE("[Vector3] String representation") {
	const Vector3 vector_a = Vector3(1.1, 1.1, 1.1);
	const Vector3 vector_c = Vector3();

	CHECK_MESSAGE(
			(vector_a == "(1.1, 1.1, 1.1)" || vector_a == "1.1, 1.1, 1.1"),
			"String representation must match expected value.");

	CHECK_MESSAGE(
			(vector_c == "(0, 0, 0)" || vector_c == "0, 0, 0"),
			"String representation must match expected value.");
}

TEST_CASE("[Vector3] Set Axis, Get Axis, Min Axis, Max Axis") {
	Vector3 vector_set = Vector3();
	const Vector3 vector_a = Vector3(1.1, 0, 0);
	const Vector3 vector_b = Vector3(1.1, 2.1, 0);
	const Vector3 vector_c = Vector3(1.1, 2.1, 3.1);

	vector_set.set_axis(0, 1.1);
	// vector_set is now (1.1, 0, 0)

	CHECK_MESSAGE(
			vector_set.is_equal_approx(vector_a),
			"Vectors with the same axis should be equal.");

	CHECK_MESSAGE(
			Math::is_equal_approx(vector_set.get_axis(0), 1.1),
			"Axis must contain expected value.");

	vector_set.set_axis(1, 2.1);
	// vector_set is now (1.1, 2.1, 0)

	CHECK_MESSAGE(
			vector_set.is_equal_approx(vector_b),
			"Vectors with the same axis should be equal .");

	CHECK_MESSAGE(
			Math::is_equal_approx(vector_set.get_axis(1), 2.1),
			"Axis must contain expected value.");

	vector_set.set_axis(2, 3.1);
	// vector_set is now (1.1, 1.1, 1.1)

	CHECK_MESSAGE(
			vector_set.is_equal_approx(vector_c),
			"Vectors with the same axis should be equal .");

	CHECK_MESSAGE(
			Math::is_equal_approx(vector_set.get_axis(2), 3.1),
			"Axis must contain expected value.");

	// Min Max Axis

	CHECK_MESSAGE(
			Math::is_equal_approx(vector_set.min_axis(), 0),
			"min_axis() must return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(vector_set.max_axis(), 2),
			"max_axis() must contain expected value.");
}

TEST_CASE("[Vector3] Length, Length Squared") {
	const Vector3 test_subject = Vector3(1.0, 2.0, 3.0);

	CHECK_MESSAGE(
			Math::is_equal_approx(test_subject.length(), 3.74166),
			"length() must return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(test_subject.length_squared(), 14),
			"length_squared() must return expected value.");
}

TEST_CASE("[Vector3] Normalize, Normalized, Is Normalized") {
	Vector3 test_subject = Vector3(1.0, 2.0, 3.0);
	Vector3 test_subject_normalized = test_subject.normalized();

	CHECK_MESSAGE(
			!test_subject.is_normalized(),
			"is_normalized() must return `false` for non-normal vectors.");

	CHECK_MESSAGE(
			test_subject_normalized.is_normalized(),
			"is_normalized() must return `true` for non-normal vectors.");

	test_subject.normalize();

	CHECK_MESSAGE(
			test_subject.is_equal_approx(Vector3(0.267261, 0.534522, 0.801784)),
			"normalize() must return expected value.");

	CHECK_MESSAGE(
			test_subject.is_equal_approx(test_subject_normalized),
			"Different methods must return same value.");
}

TEST_CASE("[Vector3] Inverse") {
	const Vector3 test_subject_a = Vector3(1.0, 2.0, 3.0).inverse();
	const Vector3 test_subject_b = Vector3(-1.0, -2.0, -3.0).inverse();
	const Vector3 test_subject_c = Vector3(100, 240, -300).inverse();

	CHECK_MESSAGE(
			test_subject_a.is_equal_approx(Vector3(1, 0.5, 0.333333)),
			"inverse() must return expected value.");

	CHECK_MESSAGE(
			test_subject_b.is_equal_approx(Vector3(-1, -0.5, -0.333333)),
			"inverse() must return expected value.");

	CHECK_MESSAGE(
			test_subject_c.is_equal_approx(Vector3(0.01, 0.00416, -0.00333)),
			"inverse() must return expected value.");
}

TEST_CASE("[Vector3] Zero") {
	Vector3 test_subject = Vector3(1.0, 2.0, 3.0);
	test_subject.zero();

	CHECK_MESSAGE(
			Vector3().is_equal_approx(test_subject),
			"zero() must return expected value.");
}

TEST_CASE("[Vector3] Posmod") {
	const Vector3 test_subject = Vector3(1.0, 2.0, 3.0);

	CHECK_MESSAGE(
			Vector3(1, 0, 1).is_equal_approx(test_subject.posmod(2)),
			"posmod() must return expected value.");

	CHECK_MESSAGE(
			Vector3(1, 2, 0).is_equal_approx(test_subject.posmod(3)),
			"posmod() must return expected value.");
}

TEST_CASE("[Vector3] Snap, Snapped") {
	Vector3 test_subject = Vector3(1.0, 2.0, 3.0);
	const Vector3 test_subject_snapped = Vector3(1.0, 2.0, 3.0).snapped(Vector3(2.5, 3.5, 4.5));

	CHECK_MESSAGE(
			test_subject_snapped.is_equal_approx(Vector3(0, 3.5, 4.5)),
			"snapped() must return expected value.");

	test_subject.snap(Vector3(2.5, 3.5, 4.5));
	CHECK_MESSAGE(
			test_subject.is_equal_approx(test_subject_snapped),
			"Same Vector snapped in different methods must return expected value.");
}

TEST_CASE("[Vector3] Rotate, Rotated") {
	Vector3 test_subject = Vector3(1.0, 2.0, 3.0);
	const Vector3 test_subject_rotated = Vector3(1.0, 2.0, 3.0).rotated(Vector3(0, 1, 0), 2.70);

	CHECK_MESSAGE(
			test_subject_rotated.is_equal_approx(Vector3(0.378067, 2, -3.139596)),
			"rotated() must return expected value.");

	test_subject.rotate(Vector3(0, 1, 0), 2.70);
	CHECK_MESSAGE(
			test_subject.is_equal_approx(test_subject_rotated),
			"Same Vector rotated in different methods must return expected value.");
}

TEST_CASE("[Vector3] Outer") {
	const Vector3 test_subject_a = Vector3(1.0, 2.0, 3.0);
	const Vector3 test_subject_b = Vector3(-1.0, 3.0, -2.0);
	const Vector3 test_subject_c = Vector3(3.0, -2.0, 1.0);

	CHECK_MESSAGE(
			Basis(-1, 3, -2, -2, 6, -4, -3, 9, -6).is_equal_approx(test_subject_a.outer(test_subject_b)),
			"outer() must return expected value.");

	CHECK_MESSAGE(
			Basis(3, -2, 1, 6, -4, 2, 9, -6, 3).is_equal_approx(test_subject_a.outer(test_subject_c)),
			"outer() must return expected value.");

	CHECK_MESSAGE(
			Basis(-3, 2, -1, 9, -6, 3, -6, 4, -2).is_equal_approx(test_subject_b.outer(test_subject_c)),
			"outer() must return expected value.");
}

TEST_CASE("[Vector3] To Diagonal Matrix") {
	const Vector3 test_subject_a = Vector3(1.0, 2.0, 3.0);
	const Vector3 test_subject_b = Vector3(-1.0, 3.0, -2.0);
	const Vector3 test_subject_c = Vector3(3.0, -2.0, 1.0);

	CHECK_MESSAGE(
			Basis(1, 0, 0, 0, 2, 0, 0, 0, 3).is_equal_approx(test_subject_a.to_diagonal_matrix()),
			"to_diagonal_matrix() must return expected value.");

	CHECK_MESSAGE(
			Basis(-1, 0, 0, 0, 3, 0, 0, 0, -2).is_equal_approx(test_subject_b.to_diagonal_matrix()),
			"to_diagonal_matrix() must return expected value.");

	CHECK_MESSAGE(
			Basis(3, 0, 0, 0, -2, 0, 0, 0, 1).is_equal_approx(test_subject_c.to_diagonal_matrix()),
			"to_diagonal_matrix() must return expected value.");
}

TEST_CASE("[Vector3] Abs") {
	const Vector3 test_subject_a = Vector3(1.0, 2.0, 3.0);
	const Vector3 test_subject_b = Vector3(-1.0, 3.0, -2.0);
	const Vector3 test_subject_c = Vector3(3.0, -2.0, 1.0);

	CHECK_MESSAGE(
			Vector3(1, 2, 3).is_equal_approx(test_subject_a.abs()),
			"abs() must return expected value.");

	CHECK_MESSAGE(
			Vector3(1, 3, 2).is_equal_approx(test_subject_b.abs()),
			"abs() must return expected value.");

	CHECK_MESSAGE(
			Vector3(3, 2, 1).is_equal_approx(test_subject_c.abs()),
			"abs() must return expected value.");
}

TEST_CASE("[Vector3] Floor") {
	const Vector3 test_subject_a = Vector3(1.3, 2.7, 3.4);
	const Vector3 test_subject_b = Vector3(-1.2, 3.9, -2.9);
	const Vector3 test_subject_c = Vector3(3.7, -2.3, 1.5);

	CHECK_MESSAGE(
			Vector3(1, 2, 3).is_equal_approx(test_subject_a.floor()),
			"floor() must return expected value.");

	CHECK_MESSAGE(
			Vector3(-2, 3, -3).is_equal_approx(test_subject_b.floor()),
			"floor() must return expected value.");

	CHECK_MESSAGE(
			Vector3(3, -3, 1).is_equal_approx(test_subject_c.floor()),
			"floor() must return expected value.");
}

TEST_CASE("[Vector3] Sign") {
	const Vector3 test_subject_a = Vector3(1.3, 2.7, 3.4);
	const Vector3 test_subject_b = Vector3(-1.2, 3.9, -2.9);
	const Vector3 test_subject_c = Vector3(3.7, -2.3, 1.5);

	CHECK_MESSAGE(
			Vector3(1, 1, 1).is_equal_approx(test_subject_a.sign()),
			"sign() must return expected value.");

	CHECK_MESSAGE(
			Vector3(-1, 1, -1).is_equal_approx(test_subject_b.sign()),
			"sign() must return expected value.");

	CHECK_MESSAGE(
			Vector3(1, -1, 1).is_equal_approx(test_subject_c.sign()),
			"sign() must return expected value.");
}

TEST_CASE("[Vector3] Ceil") {
	const Vector3 test_subject_a = Vector3(1.3, 2.7, 3.4);
	const Vector3 test_subject_b = Vector3(-1.2, 3.9, -2.9);
	const Vector3 test_subject_c = Vector3(3.7, -2.3, 1.5);

	CHECK_MESSAGE(
			Vector3(2, 3, 4).is_equal_approx(test_subject_a.ceil()),
			"ceil() must return expected value.");

	CHECK_MESSAGE(
			Vector3(-1, 4, -2).is_equal_approx(test_subject_b.ceil()),
			"ceil() must return expected value.");

	CHECK_MESSAGE(
			Vector3(4, -2, 2).is_equal_approx(test_subject_c.ceil()),
			"ceil() must return expected value.");
}

TEST_CASE("[Vector3] Round") {
	const Vector3 test_subject_a = Vector3(1.3, 2.7, 3.4);
	const Vector3 test_subject_b = Vector3(-1.2, 3.9, -2.9);
	const Vector3 test_subject_c = Vector3(3.7, -2.3, 1.5);

	CHECK_MESSAGE(
			Vector3(1, 3, 3).is_equal_approx(test_subject_a.round()),
			"round() must return expected value.");

	CHECK_MESSAGE(
			Vector3(-1, 4, -3).is_equal_approx(test_subject_b.round()),
			"round() must return expected value.");

	CHECK_MESSAGE(
			Vector3(4, -2, 2).is_equal_approx(test_subject_c.round()),
			"round() must return expected value.");
}

TEST_CASE("[Vector3] Lerp") {
	Vector3 test_subject = Vector3(1.0, 2.0, 3.0);
	const Vector3 test_subject_end = Vector3(3.0, 3.0, 3.0);

	CHECK_MESSAGE(
			Vector3(1.2, 2.1, 3.0).is_equal_approx(test_subject.lerp(test_subject_end, 0.1)),
			"lerp() must return expected value.");

	CHECK_MESSAGE(
			Vector3(2, 2.5, 3.0).is_equal_approx(test_subject.lerp(test_subject_end, 0.5)),
			"lerp() must return expected value.");

	CHECK_MESSAGE(
			Vector3(3.0, 3.0, 3.0).is_equal_approx(test_subject.lerp(test_subject_end, 1.0)),
			"lerp() must return expected value.");
}

TEST_CASE("[Vector3] Slerp") {
	const Vector3 test_subject = Vector3(1.0, 2.0, 3.0);
	const Vector3 test_subject_end = Vector3(3.0, 3.0, 3.0);

	CHECK_MESSAGE(
			Vector3(1.12581, 2.03014, 2.93447).is_equal_approx(test_subject.slerp(test_subject_end, 0.1)),
			"slerp() must return expected value.");

	CHECK_MESSAGE(
			Vector3(1.61027, 2.11981, 2.62935).is_equal_approx(test_subject.slerp(test_subject_end, 0.5)),
			"slerp() must return expected value.");

	CHECK_MESSAGE(
			Vector3(2.16025, 2.16025, 2.16025).is_equal_approx(test_subject.slerp(test_subject_end, 1.0)),
			"slerp() must return expected value.");
}

TEST_CASE("[Vector3] Cubic Interpolate") {
	const Vector3 point_a = Vector3(1.0, 2.0, 3.0);
	const Vector3 point_b = Vector3(1.8, 1.0, 1.0);
	const Vector3 point_c = Vector3(2.4, 2.5, 1.7);
	const Vector3 point_d = Vector3(3.0, 3.0, 3.0);

	CHECK_MESSAGE(
			Vector3(1.8681, 1.05325, 0.95795).is_equal_approx(point_b.cubic_interpolate(point_c, point_a, point_d, 0.1)),
			"cubic_interpolate() must return expected value.");

	CHECK_MESSAGE(
			Vector3(2.1125, 1.65625, 1.14375).is_equal_approx(point_b.cubic_interpolate(point_c, point_a, point_d, 0.5)),
			"cubic_interpolate() must return expected value.");

	CHECK_MESSAGE(
			Vector3(2.4, 2.5, 1.7).is_equal_approx(point_b.cubic_interpolate(point_c, point_a, point_d, 1.0)),
			"cubic_interpolate() must return expected value.");
}

TEST_CASE("[Vector3] Move Toward") {
	const Vector3 test_subject = Vector3(1.0, 2.0, 3.0);
	const Vector3 test_subject_end = Vector3(3.0, 3.0, 3.0);

	CHECK_MESSAGE(
			Vector3(1.08944, 2.04472, 3).is_equal_approx(test_subject.move_toward(test_subject_end, 0.1)),
			"move_toward() must return expected value.");

	CHECK_MESSAGE(
			Vector3(1.44721, 2.22361, 3).is_equal_approx(test_subject.move_toward(test_subject_end, 0.5)),
			"move_toward() must return expected value.");

	CHECK_MESSAGE(
			Vector3(1.89443, 2.44721, 3).is_equal_approx(test_subject.move_toward(test_subject_end, 1.0)),
			"move_toward() must return expected value.");
}

TEST_CASE("[Vector3] Cross") {
	const Vector3 test_subject_a = Vector3(1.0, 2.0, 3.0);
	const Vector3 test_subject_b = Vector3(-1.0, 3.0, -2.0);
	const Vector3 test_subject_c = Vector3(3.0, -2.0, 1.0);

	CHECK_MESSAGE(
			Vector3(-13, -1, 5).is_equal_approx(test_subject_a.cross(test_subject_b)),
			"cross() must return expected value.");

	CHECK_MESSAGE(
			Vector3(8, 8, -8).is_equal_approx(test_subject_a.cross(test_subject_c)),
			"cross() must return expected value.");

	CHECK_MESSAGE(
			Vector3(-1, -5, -7).is_equal_approx(test_subject_b.cross(test_subject_c)),
			"cross() must return expected value.");
}

TEST_CASE("[Vector3] Dot") {
	const Vector3 test_subject_a = Vector3(1.0, 2.0, 3.0);
	const Vector3 test_subject_b = Vector3(-1.0, 3.0, -2.0);
	const Vector3 test_subject_c = Vector3(3.0, -2.0, 1.0);

	CHECK_MESSAGE(
			Math::is_equal_approx(test_subject_a.dot(test_subject_b), -1),
			"dot() must return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(test_subject_a.dot(test_subject_c), 2),
			"dot() must return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(test_subject_b.dot(test_subject_c), -11),
			"dot() must return expected value.");
}

TEST_CASE("[Vector3] Distance To") {
	const Vector3 test_subject_a = Vector3(1.0, 2.0, 3.0);
	const Vector3 test_subject_b = Vector3(-1.0, 3.0, -2.0);
	const Vector3 test_subject_c = Vector3(3.0, -2.0, 1.0);

	CHECK_MESSAGE(
			Math::is_equal_approx(test_subject_a.distance_to(test_subject_b), 5.47723),
			"distance_to() must return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(test_subject_a.distance_to(test_subject_c), 4.89898),
			"distance_to() must return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(test_subject_b.distance_to(test_subject_c), 7.07107),
			"distance_to() must return expected value.");
}

TEST_CASE("[Vector3] Distance Squared To") {
	const Vector3 test_subject_a = Vector3(1.0, 2.0, 3.0);
	const Vector3 test_subject_b = Vector3(-1.0, 3.0, -2.0);
	const Vector3 test_subject_c = Vector3(3.0, -2.0, 1.0);

	CHECK_MESSAGE(
			Math::is_equal_approx(test_subject_a.distance_squared_to(test_subject_b), 30.00005),
			"distance_squared_to() must return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(test_subject_a.distance_squared_to(test_subject_c), 24.00001),
			"distance_squared_to() must return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(test_subject_b.distance_squared_to(test_subject_c), 50.00003),
			"distance_squared_to() must return expected value.");
}

TEST_CASE("[Vector3] Posmodv") {
	const Vector3 test_subject_a = Vector3(1.0, 2.0, 3.0);
	const Vector3 test_subject_b = Vector3(-1.0, 3.0, -2.0);
	const Vector3 test_subject_c = Vector3(3.0, -2.0, 1.0);

	CHECK_MESSAGE(
			Vector3(0, 2, -1).is_equal_approx(test_subject_a.posmodv(test_subject_b)),
			"posmodv() must return expected value.");

	CHECK_MESSAGE(
			Vector3(1, 0, 0).is_equal_approx(test_subject_a.posmodv(test_subject_c)),
			"posmodv() must return expected value.");

	CHECK_MESSAGE(
			Vector3(2, -1, 0).is_equal_approx(test_subject_b.posmodv(test_subject_c)),
			"posmodv() must return expected value.");
}

TEST_CASE("[Vector3] Angle To") {
	const Vector3 test_subject_a = Vector3(1.0, 2.0, 3.0);
	const Vector3 test_subject_b = Vector3(-1.0, 3.0, -2.0);
	const Vector3 test_subject_c = Vector3(3.0, -2.0, 1.0);

	CHECK_MESSAGE(
			Math::is_equal_approx(test_subject_a.angle_to(test_subject_b), 1.64229),
			"angle_to() must return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(test_subject_a.angle_to(test_subject_c), 1.42745),
			"angle_to() must return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(test_subject_b.angle_to(test_subject_c), 2.47465),
			"angle_to() must return expected value.");
}

TEST_CASE("[Vector3] Signed Angle To") {
	const Vector3 test_subject_a = Vector3(1.0, 2.0, 3.0);
	const Vector3 test_subject_b = Vector3(-1.0, 3.0, -2.0);
	const Vector3 test_subject_c = Vector3(3.0, -2.0, 1.0);

	CHECK_MESSAGE(
			Math::is_equal_approx(test_subject_a.signed_angle_to(test_subject_b, test_subject_c), -1.64229),
			"signed_angle_to() must return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(test_subject_a.signed_angle_to(test_subject_c, test_subject_b), 1.42745),
			"signed_angle_to() must return expected value.");

	CHECK_MESSAGE(
			Math::is_equal_approx(test_subject_b.signed_angle_to(test_subject_c, test_subject_b), 2.47465),
			"signed_angle_to() must return expected value.");
}

TEST_CASE("[Vector3] Direction To") {
	const Vector3 test_subject_a = Vector3(1.0, 2.0, 3.0);
	const Vector3 test_subject_b = Vector3(-1.0, 3.0, -2.0);
	const Vector3 test_subject_c = Vector3(3.0, -2.0, 1.0);

	CHECK_MESSAGE(
			Vector3(-0.365148, 0.182574, -0.912871).is_equal_approx(test_subject_a.direction_to(test_subject_b)),
			"direction_to() must return expected value.");

	CHECK_MESSAGE(
			Vector3(0.408248, -0.816497, -0.408248).is_equal_approx(test_subject_a.direction_to(test_subject_c)),
			"direction_to() must return expected value.");

	CHECK_MESSAGE(
			Vector3(0.565685, -0.707107, 0.424264).is_equal_approx(test_subject_b.direction_to(test_subject_c)),
			"direction_to() must return expected value.");
}

TEST_CASE("[Vector3] Project") {
	const Vector3 test_subject_a = Vector3(1.0, 2.0, 3.0);
	const Vector3 test_subject_b = Vector3(-1.0, 3.0, -2.0);
	const Vector3 test_subject_c = Vector3(3.0, -2.0, 1.0);

	CHECK_MESSAGE(
			Vector3(0.0714286, -0.214286, 0.142857).is_equal_approx(test_subject_a.project(test_subject_b)),
			"project() must return expected value.");

	CHECK_MESSAGE(
			Vector3(0.428571, -0.285714, 0.142857).is_equal_approx(test_subject_a.project(test_subject_c)),
			"project() must return expected value.");

	CHECK_MESSAGE(
			Vector3(-2.35714, 1.57143, -0.785714).is_equal_approx(test_subject_b.project(test_subject_c)),
			"project() must return expected value.");
}

TEST_CASE("[Vector3] Slide, Reflect, Bounce") {
	const Vector3 test_subject = Vector3(1.0, 2.0, 3.0);
	const Vector3 normal = Vector3(1, -1, 0).normalized();

	CHECK_MESSAGE(
			Vector3(1.5, 1.5, 3).is_equal_approx(test_subject.slide(normal)),
			"slide() must return expected value.");

	CHECK_MESSAGE(
			Vector3(-2, -1, -3).is_equal_approx(test_subject.reflect(normal)),
			"reflect() must return expected value.");

	CHECK_MESSAGE(
			Vector3(2, 1, 3).is_equal_approx(test_subject.bounce(normal)),
			"bounce() must return expected value.");
}

TEST_CASE("[Vector3] Is Equal Approx") {
	const Vector3 test_subject_a = Vector3(1.0, 2.0, 3.0);
	const Vector3 test_subject_b = Vector3(-1.0, 3.0, -2.0);

	CHECK_MESSAGE(
			Vector3(1, 2, 3).is_equal_approx(test_subject_a),
			"is_equal_approx() must return `true` for equal Vector3.");

	CHECK_FALSE_MESSAGE(
			test_subject_b.is_equal_approx(test_subject_a),
			"is_equal_approx() must return `false` for unequal Vector3.");
}

TEST_CASE("[Vector3] Operators") {
	const Vector3 test_subject_a = Vector3(1, 2, 3);
	Vector3 test_subject_b = Vector3(0, 1, 4);

	CHECK_MESSAGE(
			test_subject_a[0] == 1,
			"operator[] should return expected value.");

	CHECK_MESSAGE(
			test_subject_a[1] == 2,
			"operator[] should return expected value.");

	CHECK_MESSAGE(
			Vector3(2, 4, 6).is_equal_approx(test_subject_a * 2),
			"operator* should return expected value.");

	CHECK_MESSAGE(
			Vector3(0, 2, 12).is_equal_approx(test_subject_a * test_subject_b),
			"operator* should return expected value.");

	CHECK_MESSAGE(
			Vector3(1, 3, 7).is_equal_approx(test_subject_a + test_subject_b),
			"operator+ should return expected value.");

	test_subject_b += test_subject_a;
	// test_subject_b is now (1, 3, 7)
	CHECK_MESSAGE(
			Vector3(1, 3, 7).is_equal_approx(test_subject_b),
			"operator+= should return expected value.");

	CHECK_MESSAGE(
			Vector3(0, -1, -4).is_equal_approx(test_subject_a - test_subject_b),
			"operator- should return expected value.");

	test_subject_b -= test_subject_a;
	// test_subject_b is now (0, 1, 4)
	CHECK_MESSAGE(
			Vector3(0, 1, 4).is_equal_approx(test_subject_b),
			"operator-= should return expected value.");

	test_subject_b *= 2;
	// test_subject_b is now (0, 2, 8)
	CHECK_MESSAGE(
			Vector3(0, 2, 8).is_equal_approx(test_subject_b),
			"operator*= should return expected value.");

	CHECK_MESSAGE(
			Vector3(0, 1, 2.66667).is_equal_approx(test_subject_b / test_subject_a),
			"operator/ should return expected value.");

	test_subject_b /= 2;
	CHECK_MESSAGE(
			Vector3(0, 1, 4).is_equal_approx(test_subject_b),
			"operator/= should return expected value.");
	// test_subject_b is now (0, 1, 4)

	CHECK_MESSAGE(
			Vector3(-1, -2, -3).is_equal_approx(-test_subject_a),
			"operator- should return expected value.");

	CHECK_MESSAGE(
			test_subject_a == Vector3(1, 2, 3),
			"operator== should return expected value.");

	CHECK_FALSE_MESSAGE(
			test_subject_a == test_subject_b,
			"operator== should return expected value.");

	CHECK_MESSAGE(
			test_subject_a != test_subject_b,
			"operator!= should return expected value.");

	CHECK_FALSE_MESSAGE(
			test_subject_a != Vector3(1, 2, 3),
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

} // namespace TestVector3

#endif // TEST_VECTOR3_H
