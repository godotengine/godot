/*************************************************************************/
/*  test_math.h                                                          */
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

#ifndef TEST_MATH_H
#define TEST_MATH_H

#include "tests/test_macros.h"

namespace TestMath {

TEST_CASE("[Math] C++ macros") {
	CHECK(MIN(-2, 2) == -2);
	CHECK(MIN(600, 2) == 2);

	CHECK(MAX(-2, 2) == 2);
	CHECK(MAX(600, 2) == 600);

	CHECK(CLAMP(600, -2, 2) == 2);
	CHECK(CLAMP(620, 600, 650) == 620);
	// `max` is lower than `min`.
	CHECK(CLAMP(620, 600, 50) == 50);

	CHECK(ABS(-5) == 5);
	CHECK(ABS(0) == 0);
	CHECK(ABS(5) == 5);

	CHECK(Math::is_equal_approx(SGN(-5), -1.0));
	CHECK(Math::is_equal_approx(SGN(0), 0.0));
	CHECK(Math::is_equal_approx(SGN(5), 1.0));
}

TEST_CASE("[Math] Power of two functions") {
	CHECK(next_power_of_2(0) == 0);
	CHECK(next_power_of_2(1) == 1);
	CHECK(next_power_of_2(16) == 16);
	CHECK(next_power_of_2(17) == 32);
	CHECK(next_power_of_2(65535) == 65536);

	CHECK(previous_power_of_2(0) == 0);
	CHECK(previous_power_of_2(1) == 1);
	CHECK(previous_power_of_2(16) == 16);
	CHECK(previous_power_of_2(17) == 16);
	CHECK(previous_power_of_2(65535) == 32768);

	CHECK(closest_power_of_2(0) == 0);
	CHECK(closest_power_of_2(1) == 1);
	CHECK(closest_power_of_2(16) == 16);
	CHECK(closest_power_of_2(17) == 16);
	CHECK(closest_power_of_2(65535) == 65536);

	CHECK(get_shift_from_power_of_2(0) == -1);
	CHECK(get_shift_from_power_of_2(1) == 0);
	CHECK(get_shift_from_power_of_2(16) == 4);
	CHECK(get_shift_from_power_of_2(17) == -1);
	CHECK(get_shift_from_power_of_2(65535) == -1);

	CHECK(nearest_shift(0) == 0);
	CHECK(nearest_shift(1) == 1);
	CHECK(nearest_shift(16) == 5);
	CHECK(nearest_shift(17) == 5);
	CHECK(nearest_shift(65535) == 16);
}

TEST_CASE("[Math] abs") {
	// int
	CHECK(Math::abs(-1) == 1);
	CHECK(Math::abs(0) == 0);
	CHECK(Math::abs(1) == 1);

	// double
	CHECK(Math::is_equal_approx(Math::abs(-0.1), 0.1));
	CHECK(Math::is_equal_approx(Math::abs(0.0), 0.0));
	CHECK(Math::is_equal_approx(Math::abs(0.1), 0.1));

	// float
	CHECK(Math::is_equal_approx(Math::abs(-0.1f), 0.1f));
	CHECK(Math::is_equal_approx(Math::abs(0.0f), 0.0f));
	CHECK(Math::is_equal_approx(Math::abs(0.1f), 0.1f));
}

TEST_CASE("[Math] round/floor/ceil") {
	CHECK(Math::is_equal_approx(Math::round(1.5), 2.0));
	CHECK(Math::is_equal_approx(Math::floor(1.5), 1.0));
	CHECK(Math::is_equal_approx(Math::ceil(1.5), 2.0));
}

TEST_CASE("[Math] sin/cos/tan") {
	CHECK(Math::is_equal_approx(Math::sin(-0.1), -0.0998334166));
	CHECK(Math::is_equal_approx(Math::sin(0.1), 0.0998334166));
	CHECK(Math::is_equal_approx(Math::sin(0.5), 0.4794255386));
	CHECK(Math::is_equal_approx(Math::sin(1.0), 0.8414709848));
	CHECK(Math::is_equal_approx(Math::sin(1.5), 0.9974949866));
	CHECK(Math::is_equal_approx(Math::sin(450.0), -0.683283725));

	CHECK(Math::is_equal_approx(Math::cos(-0.1), 0.99500416530));
	CHECK(Math::is_equal_approx(Math::cos(0.1), 0.9950041653));
	CHECK(Math::is_equal_approx(Math::cos(0.5), 0.8775825619));
	CHECK(Math::is_equal_approx(Math::cos(1.0), 0.5403023059));
	CHECK(Math::is_equal_approx(Math::cos(1.5), 0.0707372017));
	CHECK(Math::is_equal_approx(Math::cos(450.0), -0.7301529642));

	CHECK(Math::is_equal_approx(Math::tan(-0.1), -0.1003346721));
	CHECK(Math::is_equal_approx(Math::tan(0.1), 0.1003346721));
	CHECK(Math::is_equal_approx(Math::tan(0.5), 0.5463024898));
	CHECK(Math::is_equal_approx(Math::tan(1.0), 1.5574077247));
	CHECK(Math::is_equal_approx(Math::tan(1.5), 14.1014199472));
	CHECK(Math::is_equal_approx(Math::tan(450.0), 0.9358090134));
}

TEST_CASE("[Math] sinh/cosh/tanh") {
	CHECK(Math::is_equal_approx(Math::sinh(-0.1), -0.10016675));
	CHECK(Math::is_equal_approx(Math::sinh(0.1), 0.10016675));
	CHECK(Math::is_equal_approx(Math::sinh(0.5), 0.5210953055));
	CHECK(Math::is_equal_approx(Math::sinh(1.0), 1.1752011936));
	CHECK(Math::is_equal_approx(Math::sinh(1.5), 2.1292794551));

	CHECK(Math::is_equal_approx(Math::cosh(-0.1), 1.0050041681));
	CHECK(Math::is_equal_approx(Math::cosh(0.1), 1.0050041681));
	CHECK(Math::is_equal_approx(Math::cosh(0.5), 1.1276259652));
	CHECK(Math::is_equal_approx(Math::cosh(1.0), 1.5430806348));
	CHECK(Math::is_equal_approx(Math::cosh(1.5), 2.3524096152));

	CHECK(Math::is_equal_approx(Math::tanh(-0.1), -0.0996679946));
	CHECK(Math::is_equal_approx(Math::tanh(0.1), 0.0996679946));
	CHECK(Math::is_equal_approx(Math::tanh(0.5), 0.4621171573));
	CHECK(Math::is_equal_approx(Math::tanh(1.0), 0.761594156));
	CHECK(Math::is_equal_approx(Math::tanh(1.5), 0.9051482536));
	CHECK(Math::is_equal_approx(Math::tanh(450.0), 1.0));
}

TEST_CASE("[Math] asin/acos/atan") {
	CHECK(Math::is_equal_approx(Math::asin(-0.1), -0.1001674212));
	CHECK(Math::is_equal_approx(Math::asin(0.1), 0.1001674212));
	CHECK(Math::is_equal_approx(Math::asin(0.5), 0.5235987756));
	CHECK(Math::is_equal_approx(Math::asin(1.0), 1.5707963268));
	CHECK(Math::is_nan(Math::asin(1.5)));
	CHECK(Math::is_nan(Math::asin(450.0)));

	CHECK(Math::is_equal_approx(Math::acos(-0.1), 1.670963748));
	CHECK(Math::is_equal_approx(Math::acos(0.1), 1.4706289056));
	CHECK(Math::is_equal_approx(Math::acos(0.5), 1.0471975512));
	CHECK(Math::is_equal_approx(Math::acos(1.0), 0.0));
	CHECK(Math::is_nan(Math::acos(1.5)));
	CHECK(Math::is_nan(Math::acos(450.0)));

	CHECK(Math::is_equal_approx(Math::atan(-0.1), -0.0996686525));
	CHECK(Math::is_equal_approx(Math::atan(0.1), 0.0996686525));
	CHECK(Math::is_equal_approx(Math::atan(0.5), 0.463647609));
	CHECK(Math::is_equal_approx(Math::atan(1.0), 0.7853981634));
	CHECK(Math::is_equal_approx(Math::atan(1.5), 0.9827937232));
	CHECK(Math::is_equal_approx(Math::atan(450.0), 1.5685741082));
}

TEST_CASE("[Math] sinc/sincn/atan2") {
	CHECK(Math::is_equal_approx(Math::sinc(-0.1), 0.9983341665));
	CHECK(Math::is_equal_approx(Math::sinc(0.1), 0.9983341665));
	CHECK(Math::is_equal_approx(Math::sinc(0.5), 0.9588510772));
	CHECK(Math::is_equal_approx(Math::sinc(1.0), 0.8414709848));
	CHECK(Math::is_equal_approx(Math::sinc(1.5), 0.6649966577));
	CHECK(Math::is_equal_approx(Math::sinc(450.0), -0.0015184083));

	CHECK(Math::is_equal_approx(Math::sincn(-0.1), 0.9836316431));
	CHECK(Math::is_equal_approx(Math::sincn(0.1), 0.9836316431));
	CHECK(Math::is_equal_approx(Math::sincn(0.5), 0.6366197724));
	CHECK(Math::is_equal_approx(Math::sincn(1.0), 0.0));
	CHECK(Math::is_equal_approx(Math::sincn(1.5), -0.2122065908));
	CHECK(Math::is_equal_approx(Math::sincn(450.0), 0.0));

	CHECK(Math::is_equal_approx(Math::atan2(-0.1, 0.5), -0.1973955598));
	CHECK(Math::is_equal_approx(Math::atan2(0.1, -0.5), 2.9441970937));
	CHECK(Math::is_equal_approx(Math::atan2(0.5, 1.5), 0.3217505544));
	CHECK(Math::is_equal_approx(Math::atan2(1.0, 2.5), 0.3805063771));
	CHECK(Math::is_equal_approx(Math::atan2(1.5, 1.0), 0.9827937232));
	CHECK(Math::is_equal_approx(Math::atan2(450.0, 1.0), 1.5685741082));
}

TEST_CASE("[Math] pow/log/log2/exp/sqrt") {
	CHECK(Math::is_equal_approx(Math::pow(-0.1, 2.0), 0.01));
	CHECK(Math::is_equal_approx(Math::pow(0.1, 2.5), 0.0031622777));
	CHECK(Math::is_equal_approx(Math::pow(0.5, 0.5), 0.7071067812));
	CHECK(Math::is_equal_approx(Math::pow(1.0, 1.0), 1.0));
	CHECK(Math::is_equal_approx(Math::pow(1.5, -1.0), 0.6666666667));
	CHECK(Math::is_equal_approx(Math::pow(450.0, -2.0), 0.0000049383));
	CHECK(Math::is_equal_approx(Math::pow(450.0, 0.0), 1.0));

	CHECK(Math::is_nan(Math::log(-0.1)));
	CHECK(Math::is_equal_approx(Math::log(0.1), -2.302585093));
	CHECK(Math::is_equal_approx(Math::log(0.5), -0.6931471806));
	CHECK(Math::is_equal_approx(Math::log(1.0), 0.0));
	CHECK(Math::is_equal_approx(Math::log(1.5), 0.4054651081));
	CHECK(Math::is_equal_approx(Math::log(450.0), 6.1092475828));

	CHECK(Math::is_nan(Math::log2(-0.1)));
	CHECK(Math::is_equal_approx(Math::log2(0.1), -3.3219280949));
	CHECK(Math::is_equal_approx(Math::log2(0.5), -1.0));
	CHECK(Math::is_equal_approx(Math::log2(1.0), 0.0));
	CHECK(Math::is_equal_approx(Math::log2(1.5), 0.5849625007));
	CHECK(Math::is_equal_approx(Math::log2(450.0), 8.8137811912));

	CHECK(Math::is_equal_approx(Math::exp(-0.1), 0.904837418));
	CHECK(Math::is_equal_approx(Math::exp(0.1), 1.1051709181));
	CHECK(Math::is_equal_approx(Math::exp(0.5), 1.6487212707));
	CHECK(Math::is_equal_approx(Math::exp(1.0), 2.7182818285));
	CHECK(Math::is_equal_approx(Math::exp(1.5), 4.4816890703));

	CHECK(Math::is_nan(Math::sqrt(-0.1)));
	CHECK(Math::is_equal_approx(Math::sqrt(0.1), 0.316228));
	CHECK(Math::is_equal_approx(Math::sqrt(0.5), 0.707107));
	CHECK(Math::is_equal_approx(Math::sqrt(1.0), 1.0));
	CHECK(Math::is_equal_approx(Math::sqrt(1.5), 1.224745));
}

TEST_CASE("[Math] is_nan/is_inf") {
// Disable division by 0 warning.
#ifdef _MSC_VER
#pragma warning(disable : 2124)
#endif
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdiv-by-zero"
#endif
	CHECK(!Math::is_nan(0.0));
	CHECK(Math::is_nan(0.0 / 0.0));
	CHECK(Math::is_nan(0.0 / 0));
	CHECK(Math::is_nan(0 / 0.0));

	CHECK(!Math::is_inf(0.0));
	CHECK(Math::is_inf(1.0 / 0.0));
	CHECK(!Math::is_inf(0.0 / 0.0));
	CHECK(!Math::is_inf(0.0 / 0));
	CHECK(!Math::is_inf(0 / 0.0));
// Re-enable division by 0 warning.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
#ifdef _MSC_VER
#pragma warning(default : 2124)
#endif
}

TEST_CASE("[Math] linear2db") {
	CHECK(Math::is_equal_approx(Math::linear2db(1.0), 0.0));
	CHECK(Math::is_equal_approx(Math::linear2db(20.0), 26.0206));
	CHECK(Math::is_inf(Math::linear2db(0.0)));
	CHECK(Math::is_nan(Math::linear2db(-20.0)));
}

TEST_CASE("[Math] db2linear") {
	CHECK(Math::is_equal_approx(Math::db2linear(0.0), 1.0));
	CHECK(Math::is_equal_approx(Math::db2linear(1.0), 1.122018));
	CHECK(Math::is_equal_approx(Math::db2linear(20.0), 10.0));
	CHECK(Math::is_equal_approx(Math::db2linear(-20.0), 0.1));
}

TEST_CASE("[Math] step_decimals") {
	CHECK(Math::step_decimals(-0.5) == 1);
	CHECK(Math::step_decimals(0) == 0);
	CHECK(Math::step_decimals(1) == 0);
	CHECK(Math::step_decimals(0.1) == 1);
	CHECK(Math::step_decimals(0.01) == 2);
	CHECK(Math::step_decimals(0.001) == 3);
	CHECK(Math::step_decimals(0.0001) == 4);
	CHECK(Math::step_decimals(0.00001) == 5);
	CHECK(Math::step_decimals(0.000001) == 6);
	CHECK(Math::step_decimals(0.0000001) == 7);
	CHECK(Math::step_decimals(0.00000001) == 8);
	CHECK(Math::step_decimals(0.000000001) == 9);
	// Too many decimals to handle.
	CHECK(Math::step_decimals(0.0000000001) == 0);
}

TEST_CASE("[Math] range_step_decimals") {
	CHECK(Math::range_step_decimals(0.000000001) == 9);
	// Too many decimals to handle.
	CHECK(Math::range_step_decimals(0.0000000001) == 0);
	// Should be treated as a step of 0 for use by the editor.
	CHECK(Math::range_step_decimals(0.0) == 16);
	CHECK(Math::range_step_decimals(-0.5) == 16);
}

TEST_CASE("[Math] lerp") {
	CHECK(Math::is_equal_approx(Math::lerp(2.0, 5.0, -0.1), 1.7));
	CHECK(Math::is_equal_approx(Math::lerp(2.0, 5.0, 0.0), 2.0));
	CHECK(Math::is_equal_approx(Math::lerp(2.0, 5.0, 0.1), 2.3));
	CHECK(Math::is_equal_approx(Math::lerp(2.0, 5.0, 1.0), 5.0));
	CHECK(Math::is_equal_approx(Math::lerp(2.0, 5.0, 2.0), 8.0));

	CHECK(Math::is_equal_approx(Math::lerp(-2.0, -5.0, -0.1), -1.7));
	CHECK(Math::is_equal_approx(Math::lerp(-2.0, -5.0, 0.0), -2.0));
	CHECK(Math::is_equal_approx(Math::lerp(-2.0, -5.0, 0.1), -2.3));
	CHECK(Math::is_equal_approx(Math::lerp(-2.0, -5.0, 1.0), -5.0));
	CHECK(Math::is_equal_approx(Math::lerp(-2.0, -5.0, 2.0), -8.0));
}

TEST_CASE("[Math] inverse_lerp") {
	CHECK(Math::is_equal_approx(Math::inverse_lerp(2.0, 5.0, 1.7), -0.1));
	CHECK(Math::is_equal_approx(Math::inverse_lerp(2.0, 5.0, 2.0), 0.0));
	CHECK(Math::is_equal_approx(Math::inverse_lerp(2.0, 5.0, 2.3), 0.1));
	CHECK(Math::is_equal_approx(Math::inverse_lerp(2.0, 5.0, 5.0), 1.0));
	CHECK(Math::is_equal_approx(Math::inverse_lerp(2.0, 5.0, 8.0), 2.0));

	CHECK(Math::is_equal_approx(Math::inverse_lerp(-2.0, -5.0, -1.7), -0.1));
	CHECK(Math::is_equal_approx(Math::inverse_lerp(-2.0, -5.0, -2.0), 0.0));
	CHECK(Math::is_equal_approx(Math::inverse_lerp(-2.0, -5.0, -2.3), 0.1));
	CHECK(Math::is_equal_approx(Math::inverse_lerp(-2.0, -5.0, -5.0), 1.0));
	CHECK(Math::is_equal_approx(Math::inverse_lerp(-2.0, -5.0, -8.0), 2.0));
}

TEST_CASE("[Math] range_lerp") {
	CHECK(Math::is_equal_approx(Math::range_lerp(50.0, 100.0, 200.0, 0.0, 1000.0), -500.0));
	CHECK(Math::is_equal_approx(Math::range_lerp(100.0, 100.0, 200.0, 0.0, 1000.0), 0.0));
	CHECK(Math::is_equal_approx(Math::range_lerp(200.0, 100.0, 200.0, 0.0, 1000.0), 1000.0));
	CHECK(Math::is_equal_approx(Math::range_lerp(250.0, 100.0, 200.0, 0.0, 1000.0), 1500.0));

	CHECK(Math::is_equal_approx(Math::range_lerp(-50.0, -100.0, -200.0, 0.0, 1000.0), -500.0));
	CHECK(Math::is_equal_approx(Math::range_lerp(-100.0, -100.0, -200.0, 0.0, 1000.0), 0.0));
	CHECK(Math::is_equal_approx(Math::range_lerp(-200.0, -100.0, -200.0, 0.0, 1000.0), 1000.0));
	CHECK(Math::is_equal_approx(Math::range_lerp(-250.0, -100.0, -200.0, 0.0, 1000.0), 1500.0));

	CHECK(Math::is_equal_approx(Math::range_lerp(-50.0, -100.0, -200.0, 0.0, -1000.0), 500.0));
	CHECK(Math::is_equal_approx(Math::range_lerp(-100.0, -100.0, -200.0, 0.0, -1000.0), 0.0));
	CHECK(Math::is_equal_approx(Math::range_lerp(-200.0, -100.0, -200.0, 0.0, -1000.0), -1000.0));
	CHECK(Math::is_equal_approx(Math::range_lerp(-250.0, -100.0, -200.0, 0.0, -1000.0), -1500.0));
}

TEST_CASE("[Math] lerp_angle") {
	// Counter-clockwise rotation.
	CHECK(Math::is_equal_approx(Math::lerp_angle(0.24 * Math_TAU, 0.75 * Math_TAU, 0.5), -0.005 * Math_TAU));
	// Counter-clockwise rotation.
	CHECK(Math::is_equal_approx(Math::lerp_angle(0.25 * Math_TAU, 0.75 * Math_TAU, 0.5), 0.0));
	// Clockwise rotation.
	CHECK(Math::is_equal_approx(Math::lerp_angle(0.26 * Math_TAU, 0.75 * Math_TAU, 0.5), 0.505 * Math_TAU));

	CHECK(Math::is_equal_approx(Math::lerp_angle(-0.25 * Math_TAU, 1.25 * Math_TAU, 0.5), -0.5 * Math_TAU));
	CHECK(Math::is_equal_approx(Math::lerp_angle(0.72 * Math_TAU, 1.44 * Math_TAU, 0.96), 0.4512 * Math_TAU));
	CHECK(Math::is_equal_approx(Math::lerp_angle(0.72 * Math_TAU, 1.44 * Math_TAU, 1.04), 0.4288 * Math_TAU));

	// Initial and final angles are effectively identical, so the value returned
	// should always be the same regardless of the `weight` parameter.
	CHECK(Math::is_equal_approx(Math::lerp_angle(-4 * Math_TAU, 4 * Math_TAU, -1.0), -4.0 * Math_TAU));
	CHECK(Math::is_equal_approx(Math::lerp_angle(-4 * Math_TAU, 4 * Math_TAU, 0.0), -4.0 * Math_TAU));
	CHECK(Math::is_equal_approx(Math::lerp_angle(-4 * Math_TAU, 4 * Math_TAU, 0.5), -4.0 * Math_TAU));
	CHECK(Math::is_equal_approx(Math::lerp_angle(-4 * Math_TAU, 4 * Math_TAU, 1.0), -4.0 * Math_TAU));
	CHECK(Math::is_equal_approx(Math::lerp_angle(-4 * Math_TAU, 4 * Math_TAU, 500.0), -4.0 * Math_TAU));
}

TEST_CASE("[Math] move_toward") {
	CHECK(Math::is_equal_approx(Math::move_toward(2.0, 5.0, -1.0), 1.0));
	CHECK(Math::is_equal_approx(Math::move_toward(2.0, 5.0, 2.5), 4.5));
	CHECK(Math::is_equal_approx(Math::move_toward(2.0, 5.0, 4.0), 5.0));
	CHECK(Math::is_equal_approx(Math::move_toward(-2.0, -5.0, -1.0), -1.0));
	CHECK(Math::is_equal_approx(Math::move_toward(-2.0, -5.0, 2.5), -4.5));
	CHECK(Math::is_equal_approx(Math::move_toward(-2.0, -5.0, 4.0), -5.0));
}

TEST_CASE("[Math] smoothstep") {
	CHECK(Math::is_equal_approx(Math::smoothstep(0.0, 2.0, -5.0), 0.0));
	CHECK(Math::is_equal_approx(Math::smoothstep(0.0, 2.0, 0.5), 0.15625));
	CHECK(Math::is_equal_approx(Math::smoothstep(0.0, 2.0, 1.0), 0.5));
	CHECK(Math::is_equal_approx(Math::smoothstep(0.0, 2.0, 2.0), 1.0));
}

TEST_CASE("[Math] ease") {
	CHECK(Math::is_equal_approx(Math::ease(0.1, 1.0), 0.1));
	CHECK(Math::is_equal_approx(Math::ease(0.1, 2.0), 0.01));
	CHECK(Math::is_equal_approx(Math::ease(0.1, 0.5), 0.19));
	CHECK(Math::is_equal_approx(Math::ease(0.1, 0.0), 0));
	CHECK(Math::is_equal_approx(Math::ease(0.1, -0.5), 0.2236067977));
	CHECK(Math::is_equal_approx(Math::ease(0.1, -1.0), 0.1));
	CHECK(Math::is_equal_approx(Math::ease(0.1, -2.0), 0.02));

	CHECK(Math::is_equal_approx(Math::ease(-1.0, 1.0), 0));
	CHECK(Math::is_equal_approx(Math::ease(-1.0, 2.0), 0));
	CHECK(Math::is_equal_approx(Math::ease(-1.0, 0.5), 0));
	CHECK(Math::is_equal_approx(Math::ease(-1.0, 0.0), 0));
	CHECK(Math::is_equal_approx(Math::ease(-1.0, -0.5), 0));
	CHECK(Math::is_equal_approx(Math::ease(-1.0, -1.0), 0));
	CHECK(Math::is_equal_approx(Math::ease(-1.0, -2.0), 0));
}

TEST_CASE("[Math] snapped") {
	CHECK(Math::is_equal_approx(Math::snapped(0.5, 0.04), 0.52));
	CHECK(Math::is_equal_approx(Math::snapped(-0.5, 0.04), -0.48));
	CHECK(Math::is_equal_approx(Math::snapped(0.0, 0.04), 0));
	CHECK(Math::is_equal_approx(Math::snapped(128'000.025, 0.04), 128'000.04));

	CHECK(Math::is_equal_approx(Math::snapped(0.5, 400), 0));
	CHECK(Math::is_equal_approx(Math::snapped(-0.5, 400), 0));
	CHECK(Math::is_equal_approx(Math::snapped(0.0, 400), 0));
	CHECK(Math::is_equal_approx(Math::snapped(128'000.025, 400), 128'000.0));

	CHECK(Math::is_equal_approx(Math::snapped(0.5, 0.0), 0.5));
	CHECK(Math::is_equal_approx(Math::snapped(-0.5, 0.0), -0.5));
	CHECK(Math::is_equal_approx(Math::snapped(0.0, 0.0), 0.0));
	CHECK(Math::is_equal_approx(Math::snapped(128'000.025, 0.0), 128'000.0));

	CHECK(Math::is_equal_approx(Math::snapped(0.5, -1.0), 0));
	CHECK(Math::is_equal_approx(Math::snapped(-0.5, -1.0), -1.0));
	CHECK(Math::is_equal_approx(Math::snapped(0.0, -1.0), 0));
	CHECK(Math::is_equal_approx(Math::snapped(128'000.025, -1.0), 128'000.0));
}

TEST_CASE("[Math] dectime") {
	CHECK(Math::is_equal_approx(Math::dectime(60.0, 10.0, 0.1), 59.0));
	CHECK(Math::is_equal_approx(Math::dectime(60.0, -10.0, 0.1), 61.0));
	CHECK(Math::is_equal_approx(Math::dectime(60.0, 100.0, 1.0), 0.0));
}

TEST_CASE("[Math] larger_prime") {
	CHECK(Math::larger_prime(0) == 5);
	CHECK(Math::larger_prime(1) == 5);
	CHECK(Math::larger_prime(2) == 5);
	CHECK(Math::larger_prime(5) == 13);
	CHECK(Math::larger_prime(500) == 769);
	CHECK(Math::larger_prime(1'000'000) == 1'572'869);
	CHECK(Math::larger_prime(1'000'000'000) == 1'610'612'741);

	// The next prime is larger than `INT32_MAX` and is not present in the built-in prime table.
	ERR_PRINT_OFF;
	CHECK(Math::larger_prime(2'000'000'000) == 0);
	ERR_PRINT_ON;
}

TEST_CASE("[Math] fmod") {
	CHECK(Math::is_equal_approx(Math::fmod(-2.0, 0.3), -0.2));
	CHECK(Math::is_equal_approx(Math::fmod(0.0, 0.3), 0.0));
	CHECK(Math::is_equal_approx(Math::fmod(2.0, 0.3), 0.2));

	CHECK(Math::is_equal_approx(Math::fmod(-2.0, -0.3), -0.2));
	CHECK(Math::is_equal_approx(Math::fmod(0.0, -0.3), 0.0));
	CHECK(Math::is_equal_approx(Math::fmod(2.0, -0.3), 0.2));
}

TEST_CASE("[Math] fposmod") {
	CHECK(Math::is_equal_approx(Math::fposmod(-2.0, 0.3), 0.1));
	CHECK(Math::is_equal_approx(Math::fposmod(0.0, 0.3), 0.0));
	CHECK(Math::is_equal_approx(Math::fposmod(2.0, 0.3), 0.2));

	CHECK(Math::is_equal_approx(Math::fposmod(-2.0, -0.3), -0.2));
	CHECK(Math::is_equal_approx(Math::fposmod(0.0, -0.3), 0.0));
	CHECK(Math::is_equal_approx(Math::fposmod(2.0, -0.3), -0.1));
}

TEST_CASE("[Math] fposmodp") {
	CHECK(Math::is_equal_approx(Math::fposmodp(-2.0, 0.3), 0.1));
	CHECK(Math::is_equal_approx(Math::fposmodp(0.0, 0.3), 0.0));
	CHECK(Math::is_equal_approx(Math::fposmodp(2.0, 0.3), 0.2));

	CHECK(Math::is_equal_approx(Math::fposmodp(-2.0, -0.3), -0.5));
	CHECK(Math::is_equal_approx(Math::fposmodp(0.0, -0.3), 0.0));
	CHECK(Math::is_equal_approx(Math::fposmodp(2.0, -0.3), 0.2));
}

TEST_CASE("[Math] posmod") {
	CHECK(Math::posmod(-20, 3) == 1);
	CHECK(Math::posmod(0, 3) == 0);
	CHECK(Math::posmod(20, 3) == 2);
	CHECK(Math::posmod(-20, -3) == -2);
	CHECK(Math::posmod(0, -3) == 0);
	CHECK(Math::posmod(20, -3) == -1);
}

TEST_CASE("[Math] wrapi") {
	CHECK(Math::wrapi(-30, -20, 160) == 150);
	CHECK(Math::wrapi(30, -20, 160) == 30);
	CHECK(Math::wrapi(300, -20, 160) == 120);
	CHECK(Math::wrapi(300'000'000'000, -20, 160) == 120);
}

TEST_CASE("[Math] wrapf") {
	CHECK(Math::is_equal_approx(Math::wrapf(-30.0, -20.0, 160.0), 150.0));
	CHECK(Math::is_equal_approx(Math::wrapf(30.0, -2.0, 160.0), 30.0));
	CHECK(Math::is_equal_approx(Math::wrapf(300.0, -20.0, 160.0), 120.0));
	CHECK(Math::is_equal_approx(Math::wrapf(300'000'000'000.0, -20.0, 160.0), 120.0));
}

} // namespace TestMath

#endif // TEST_MATH_H
