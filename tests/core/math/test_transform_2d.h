/**************************************************************************/
/*  test_transform_2d.h                                                   */
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

#ifndef TEST_TRANSFORM_2D_H
#define TEST_TRANSFORM_2D_H

#include "core/math/transform_2d.h"

#include "tests/test_macros.h"

namespace TestTransform2D {

Transform2D create_dummy_transform() {
	return Transform2D(Vector2(1, 2), Vector2(3, 4), Vector2(5, 6));
}

Transform2D identity() {
	return Transform2D();
}

TEST_CASE("[Transform2D] translation") {
	Vector2 offset = Vector2(1, 2);

	// Both versions should give the same result applied to identity.
	CHECK(identity().translated(offset) == identity().translated_local(offset));

	// Check both versions against left and right multiplications.
	Transform2D orig = create_dummy_transform();
	Transform2D T = identity().translated(offset);
	CHECK(orig.translated(offset) == T * orig);
	CHECK(orig.translated_local(offset) == orig * T);
}

TEST_CASE("[Transform2D] scaling") {
	Vector2 scaling = Vector2(1, 2);

	// Both versions should give the same result applied to identity.
	CHECK(identity().scaled(scaling) == identity().scaled_local(scaling));

	// Check both versions against left and right multiplications.
	Transform2D orig = create_dummy_transform();
	Transform2D S = identity().scaled(scaling);
	CHECK(orig.scaled(scaling) == S * orig);
	CHECK(orig.scaled_local(scaling) == orig * S);
}

TEST_CASE("[Transform2D] rotation") {
	real_t phi = 1.0;

	// Both versions should give the same result applied to identity.
	CHECK(identity().rotated(phi) == identity().rotated_local(phi));

	// Check both versions against left and right multiplications.
	Transform2D orig = create_dummy_transform();
	Transform2D R = identity().rotated(phi);
	CHECK(orig.rotated(phi) == R * orig);
	CHECK(orig.rotated_local(phi) == orig * R);
}

TEST_CASE("[Transform2D] Interpolation") {
	Transform2D rotate_scale_skew_pos = Transform2D(Math::deg_to_rad(170.0), Vector2(3.6, 8.0), Math::deg_to_rad(20.0), Vector2(2.4, 6.8));
	Transform2D rotate_scale_skew_pos_halfway = Transform2D(Math::deg_to_rad(85.0), Vector2(2.3, 4.5), Math::deg_to_rad(10.0), Vector2(1.2, 3.4));
	Transform2D interpolated = Transform2D().interpolate_with(rotate_scale_skew_pos, 0.5);
	CHECK(interpolated.get_origin().is_equal_approx(rotate_scale_skew_pos_halfway.get_origin()));
	CHECK(interpolated.get_rotation() == doctest::Approx(rotate_scale_skew_pos_halfway.get_rotation()));
	CHECK(interpolated.get_scale().is_equal_approx(rotate_scale_skew_pos_halfway.get_scale()));
	CHECK(interpolated.get_skew() == doctest::Approx(rotate_scale_skew_pos_halfway.get_skew()));
	CHECK(interpolated.is_equal_approx(rotate_scale_skew_pos_halfway));
	interpolated = rotate_scale_skew_pos.interpolate_with(Transform2D(), 0.5);
	CHECK(interpolated.is_equal_approx(rotate_scale_skew_pos_halfway));
}

TEST_CASE("[Transform2D] Finite number checks") {
	const Vector2 x(0, 1);
	const Vector2 infinite(NAN, NAN);

	CHECK_MESSAGE(
			Transform2D(x, x, x).is_finite(),
			"Transform2D with all components finite should be finite");

	CHECK_FALSE_MESSAGE(
			Transform2D(infinite, x, x).is_finite(),
			"Transform2D with one component infinite should not be finite.");
	CHECK_FALSE_MESSAGE(
			Transform2D(x, infinite, x).is_finite(),
			"Transform2D with one component infinite should not be finite.");
	CHECK_FALSE_MESSAGE(
			Transform2D(x, x, infinite).is_finite(),
			"Transform2D with one component infinite should not be finite.");

	CHECK_FALSE_MESSAGE(
			Transform2D(infinite, infinite, x).is_finite(),
			"Transform2D with two components infinite should not be finite.");
	CHECK_FALSE_MESSAGE(
			Transform2D(infinite, x, infinite).is_finite(),
			"Transform2D with two components infinite should not be finite.");
	CHECK_FALSE_MESSAGE(
			Transform2D(x, infinite, infinite).is_finite(),
			"Transform2D with two components infinite should not be finite.");

	CHECK_FALSE_MESSAGE(
			Transform2D(infinite, infinite, infinite).is_finite(),
			"Transform2D with three components infinite should not be finite.");
}

TEST_CASE("[Transform2D] Is conformal checks") {
	CHECK_MESSAGE(
			Transform2D().is_conformal(),
			"Identity Transform2D should be conformal.");

	CHECK_MESSAGE(
			Transform2D(1.2, Vector2()).is_conformal(),
			"Transform2D with only rotation should be conformal.");

	CHECK_MESSAGE(
			Transform2D(Vector2(1, 0), Vector2(0, -1), Vector2()).is_conformal(),
			"Transform2D with only a flip should be conformal.");

	CHECK_MESSAGE(
			Transform2D(Vector2(1.2, 0), Vector2(0, 1.2), Vector2()).is_conformal(),
			"Transform2D with only uniform scale should be conformal.");

	CHECK_MESSAGE(
			Transform2D(Vector2(1.2, 3.4), Vector2(3.4, -1.2), Vector2()).is_conformal(),
			"Transform2D with a flip, rotation, and uniform scale should be conformal.");

	CHECK_FALSE_MESSAGE(
			Transform2D(Vector2(1.2, 0), Vector2(0, 3.4), Vector2()).is_conformal(),
			"Transform2D with non-uniform scale should not be conformal.");

	CHECK_FALSE_MESSAGE(
			Transform2D(Vector2(Math_SQRT12, Math_SQRT12), Vector2(0, 1), Vector2()).is_conformal(),
			"Transform2D with the X axis skewed 45 degrees should not be conformal.");
}

} // namespace TestTransform2D

#endif // TEST_TRANSFORM_2D_H
