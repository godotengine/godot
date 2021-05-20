/*************************************************************************/
/*  test_curve.h                                                         */
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

#ifndef TEST_CURVE_H
#define TEST_CURVE_H

#include "scene/resources/curve.h"

#include "thirdparty/doctest/doctest.h"

namespace TestCurve {

TEST_CASE("[Curve] Default curve") {
	const Ref<Curve> curve = memnew(Curve);

	CHECK_MESSAGE(
			curve->get_point_count() == 0,
			"Default curve should contain the expected number of points.");
	CHECK_MESSAGE(
			Math::is_zero_approx(curve->interpolate(0)),
			"Default curve should return the expected value at offset 0.0.");
	CHECK_MESSAGE(
			Math::is_zero_approx(curve->interpolate(0.5)),
			"Default curve should return the expected value at offset 0.5.");
	CHECK_MESSAGE(
			Math::is_zero_approx(curve->interpolate(1)),
			"Default curve should return the expected value at offset 1.0.");
}

TEST_CASE("[Curve] Custom curve with free tangents") {
	Ref<Curve> curve = memnew(Curve);
	// "Sawtooth" curve with an open ending towards the 1.0 offset.
	curve->add_point(Vector2(0, 0));
	curve->add_point(Vector2(0.25, 1));
	curve->add_point(Vector2(0.5, 0));
	curve->add_point(Vector2(0.75, 1));

	CHECK_MESSAGE(
			Math::is_zero_approx(curve->get_point_left_tangent(0)),
			"get_point_left_tangent() should return the expected value for point index 0.");
	CHECK_MESSAGE(
			Math::is_zero_approx(curve->get_point_right_tangent(0)),
			"get_point_right_tangent() should return the expected value for point index 0.");
	CHECK_MESSAGE(
			curve->get_point_left_mode(0) == Curve::TangentMode::TANGENT_FREE,
			"get_point_left_mode() should return the expected value for point index 0.");
	CHECK_MESSAGE(
			curve->get_point_right_mode(0) == Curve::TangentMode::TANGENT_FREE,
			"get_point_right_mode() should return the expected value for point index 0.");

	CHECK_MESSAGE(
			curve->get_point_count() == 4,
			"Custom free curve should contain the expected number of points.");

	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate(-0.1), 0),
			"Custom free curve should return the expected value at offset 0.1.");
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate(0.1), (real_t)0.352),
			"Custom free curve should return the expected value at offset 0.1.");
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate(0.4), (real_t)0.352),
			"Custom free curve should return the expected value at offset 0.1.");
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate(0.7), (real_t)0.896),
			"Custom free curve should return the expected value at offset 0.1.");
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate(1), 1),
			"Custom free curve should return the expected value at offset 0.1.");
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate(2), 1),
			"Custom free curve should return the expected value at offset 0.1.");

	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate_baked(-0.1), 0),
			"Custom free curve should return the expected baked value at offset 0.1.");
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate_baked(0.1), (real_t)0.352),
			"Custom free curve should return the expected baked value at offset 0.1.");
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate_baked(0.4), (real_t)0.352),
			"Custom free curve should return the expected baked value at offset 0.1.");
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate_baked(0.7), (real_t)0.896),
			"Custom free curve should return the expected baked value at offset 0.1.");
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate_baked(1), 1),
			"Custom free curve should return the expected baked value at offset 0.1.");
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate_baked(2), 1),
			"Custom free curve should return the expected baked value at offset 0.1.");

	curve->remove_point(1);
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate(0.1), 0),
			"Custom free curve should return the expected value at offset 0.1 after removing point at index 1.");
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate_baked(0.1), 0),
			"Custom free curve should return the expected baked value at offset 0.1 after removing point at index 1.");

	curve->clear_points();
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate(0.6), 0),
			"Custom free curve should return the expected value at offset 0.6 after clearing all points.");
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate_baked(0.6), 0),
			"Custom free curve should return the expected baked value at offset 0.6 after clearing all points.");
}

TEST_CASE("[Curve] Custom curve with linear tangents") {
	Ref<Curve> curve = memnew(Curve);
	// "Sawtooth" curve with an open ending towards the 1.0 offset.
	curve->add_point(Vector2(0, 0), 0, 0, Curve::TangentMode::TANGENT_LINEAR, Curve::TangentMode::TANGENT_LINEAR);
	curve->add_point(Vector2(0.25, 1), 0, 0, Curve::TangentMode::TANGENT_LINEAR, Curve::TangentMode::TANGENT_LINEAR);
	curve->add_point(Vector2(0.5, 0), 0, 0, Curve::TangentMode::TANGENT_LINEAR, Curve::TangentMode::TANGENT_LINEAR);
	curve->add_point(Vector2(0.75, 1), 0, 0, Curve::TangentMode::TANGENT_LINEAR, Curve::TangentMode::TANGENT_LINEAR);

	CHECK_MESSAGE(
			Math::is_equal_approx(curve->get_point_left_tangent(3), 4),
			"get_point_left_tangent() should return the expected value for point index 3.");
	CHECK_MESSAGE(
			Math::is_zero_approx(curve->get_point_right_tangent(3)),
			"get_point_right_tangent() should return the expected value for point index 3.");
	CHECK_MESSAGE(
			curve->get_point_left_mode(3) == Curve::TangentMode::TANGENT_LINEAR,
			"get_point_left_mode() should return the expected value for point index 3.");
	CHECK_MESSAGE(
			curve->get_point_right_mode(3) == Curve::TangentMode::TANGENT_LINEAR,
			"get_point_right_mode() should return the expected value for point index 3.");

	ERR_PRINT_OFF;
	CHECK_MESSAGE(
			Math::is_zero_approx(curve->get_point_right_tangent(300)),
			"get_point_right_tangent() should return the expected value for invalid point index 300.");
	CHECK_MESSAGE(
			curve->get_point_left_mode(-12345) == Curve::TangentMode::TANGENT_FREE,
			"get_point_left_mode() should return the expected value for invalid point index -12345.");
	ERR_PRINT_ON;

	CHECK_MESSAGE(
			curve->get_point_count() == 4,
			"Custom linear curve should contain the expected number of points.");

	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate(-0.1), 0),
			"Custom linear curve should return the expected value at offset -0.1.");
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate(0.1), (real_t)0.4),
			"Custom linear curve should return the expected value at offset 0.1.");
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate(0.4), (real_t)0.4),
			"Custom linear curve should return the expected value at offset 0.4.");
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate(0.7), (real_t)0.8),
			"Custom linear curve should return the expected value at offset 0.7.");
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate(1), 1),
			"Custom linear curve should return the expected value at offset 1.0.");
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate(2), 1),
			"Custom linear curve should return the expected value at offset 2.0.");

	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate_baked(-0.1), 0),
			"Custom linear curve should return the expected baked value at offset -0.1.");
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate_baked(0.1), (real_t)0.4),
			"Custom linear curve should return the expected baked value at offset 0.1.");
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate_baked(0.4), (real_t)0.4),
			"Custom linear curve should return the expected baked value at offset 0.4.");
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate_baked(0.7), (real_t)0.8),
			"Custom linear curve should return the expected baked value at offset 0.7.");
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate_baked(1), 1),
			"Custom linear curve should return the expected baked value at offset 1.0.");
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate_baked(2), 1),
			"Custom linear curve should return the expected baked value at offset 2.0.");

	ERR_PRINT_OFF;
	curve->remove_point(10);
	ERR_PRINT_ON;
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate(0.7), (real_t)0.8),
			"Custom free curve should return the expected value at offset 0.7 after removing point at invalid index 10.");
	CHECK_MESSAGE(
			Math::is_equal_approx(curve->interpolate_baked(0.7), (real_t)0.8),
			"Custom free curve should return the expected baked value at offset 0.7 after removing point at invalid index 10.");
}
} // namespace TestCurve

#endif // TEST_CURVE_H
