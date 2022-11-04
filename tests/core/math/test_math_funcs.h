/*************************************************************************/
/*  test_math_funcs.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef TEST_MATH_FUNCS_H
#define TEST_MATH_FUNCS_H

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

	CHECK(SIGN(-5) == -1.0);
	CHECK(SIGN(0) == 0.0);
	CHECK(SIGN(5) == 1.0);
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
	CHECK(Math::abs(-0.1) == 0.1);
	CHECK(Math::abs(0.0) == 0.0);
	CHECK(Math::abs(0.1) == 0.1);

	// float
	CHECK(Math::abs(-0.1f) == 0.1f);
	CHECK(Math::abs(0.0f) == 0.0f);
	CHECK(Math::abs(0.1f) == 0.1f);
}

TEST_CASE("[Math] round/floor/ceil") {
	CHECK(Math::round(1.5) == 2.0);
	CHECK(Math::round(1.6) == 2.0);
	CHECK(Math::round(-1.5) == -2.0);
	CHECK(Math::round(-1.1) == -1.0);

	CHECK(Math::floor(1.5) == 1.0);
	CHECK(Math::floor(-1.5) == -2.0);

	CHECK(Math::ceil(1.5) == 2.0);
	CHECK(Math::ceil(-1.9) == -1.0);
}

TEST_CASE("[Math] sin/cos/tan") {
	CHECK(Math::sin(-0.1) == doctest::Approx(-0.0998334166));
	CHECK(Math::sin(0.1) == doctest::Approx(0.0998334166));
	CHECK(Math::sin(0.5) == doctest::Approx(0.4794255386));
	CHECK(Math::sin(1.0) == doctest::Approx(0.8414709848));
	CHECK(Math::sin(1.5) == doctest::Approx(0.9974949866));
	CHECK(Math::sin(450.0) == doctest::Approx(-0.683283725));

	CHECK(Math::cos(-0.1) == doctest::Approx(0.99500416530));
	CHECK(Math::cos(0.1) == doctest::Approx(0.9950041653));
	CHECK(Math::cos(0.5) == doctest::Approx(0.8775825619));
	CHECK(Math::cos(1.0) == doctest::Approx(0.5403023059));
	CHECK(Math::cos(1.5) == doctest::Approx(0.0707372017));
	CHECK(Math::cos(450.0) == doctest::Approx(-0.7301529642));

	CHECK(Math::tan(-0.1) == doctest::Approx(-0.1003346721));
	CHECK(Math::tan(0.1) == doctest::Approx(0.1003346721));
	CHECK(Math::tan(0.5) == doctest::Approx(0.5463024898));
	CHECK(Math::tan(1.0) == doctest::Approx(1.5574077247));
	CHECK(Math::tan(1.5) == doctest::Approx(14.1014199472));
	CHECK(Math::tan(450.0) == doctest::Approx(0.9358090134));
}

TEST_CASE("[Math] sinh/cosh/tanh") {
	CHECK(Math::sinh(-0.1) == doctest::Approx(-0.10016675));
	CHECK(Math::sinh(0.1) == doctest::Approx(0.10016675));
	CHECK(Math::sinh(0.5) == doctest::Approx(0.5210953055));
	CHECK(Math::sinh(1.0) == doctest::Approx(1.1752011936));
	CHECK(Math::sinh(1.5) == doctest::Approx(2.1292794551));

	CHECK(Math::cosh(-0.1) == doctest::Approx(1.0050041681));
	CHECK(Math::cosh(0.1) == doctest::Approx(1.0050041681));
	CHECK(Math::cosh(0.5) == doctest::Approx(1.1276259652));
	CHECK(Math::cosh(1.0) == doctest::Approx(1.5430806348));
	CHECK(Math::cosh(1.5) == doctest::Approx(2.3524096152));

	CHECK(Math::tanh(-0.1) == doctest::Approx(-0.0996679946));
	CHECK(Math::tanh(0.1) == doctest::Approx(0.0996679946));
	CHECK(Math::tanh(0.5) == doctest::Approx(0.4621171573));
	CHECK(Math::tanh(1.0) == doctest::Approx(0.761594156));
	CHECK(Math::tanh(1.5) == doctest::Approx(0.9051482536));
	CHECK(Math::tanh(450.0) == doctest::Approx(1.0));
}

TEST_CASE("[Math] asin/acos/atan") {
	CHECK(Math::asin(-0.1) == doctest::Approx(-0.1001674212));
	CHECK(Math::asin(0.1) == doctest::Approx(0.1001674212));
	CHECK(Math::asin(0.5) == doctest::Approx(0.5235987756));
	CHECK(Math::asin(1.0) == doctest::Approx(1.5707963268));
	CHECK(Math::is_nan(Math::asin(1.5)));
	CHECK(Math::is_nan(Math::asin(450.0)));

	CHECK(Math::acos(-0.1) == doctest::Approx(1.670963748));
	CHECK(Math::acos(0.1) == doctest::Approx(1.4706289056));
	CHECK(Math::acos(0.5) == doctest::Approx(1.0471975512));
	CHECK(Math::acos(1.0) == doctest::Approx(0.0));
	CHECK(Math::is_nan(Math::acos(1.5)));
	CHECK(Math::is_nan(Math::acos(450.0)));

	CHECK(Math::atan(-0.1) == doctest::Approx(-0.0996686525));
	CHECK(Math::atan(0.1) == doctest::Approx(0.0996686525));
	CHECK(Math::atan(0.5) == doctest::Approx(0.463647609));
	CHECK(Math::atan(1.0) == doctest::Approx(0.7853981634));
	CHECK(Math::atan(1.5) == doctest::Approx(0.9827937232));
	CHECK(Math::atan(450.0) == doctest::Approx(1.5685741082));
}

TEST_CASE("[Math] sinc/sincn/atan2") {
	CHECK(Math::sinc(-0.1) == doctest::Approx(0.9983341665));
	CHECK(Math::sinc(0.1) == doctest::Approx(0.9983341665));
	CHECK(Math::sinc(0.5) == doctest::Approx(0.9588510772));
	CHECK(Math::sinc(1.0) == doctest::Approx(0.8414709848));
	CHECK(Math::sinc(1.5) == doctest::Approx(0.6649966577));
	CHECK(Math::sinc(450.0) == doctest::Approx(-0.0015184083));

	CHECK(Math::sincn(-0.1) == doctest::Approx(0.9836316431));
	CHECK(Math::sincn(0.1) == doctest::Approx(0.9836316431));
	CHECK(Math::sincn(0.5) == doctest::Approx(0.6366197724));
	CHECK(Math::sincn(1.0) == doctest::Approx(0.0));
	CHECK(Math::sincn(1.5) == doctest::Approx(-0.2122065908));
	CHECK(Math::sincn(450.0) == doctest::Approx(0.0));

	CHECK(Math::atan2(-0.1, 0.5) == doctest::Approx(-0.1973955598));
	CHECK(Math::atan2(0.1, -0.5) == doctest::Approx(2.9441970937));
	CHECK(Math::atan2(0.5, 1.5) == doctest::Approx(0.3217505544));
	CHECK(Math::atan2(1.0, 2.5) == doctest::Approx(0.3805063771));
	CHECK(Math::atan2(1.5, 1.0) == doctest::Approx(0.9827937232));
	CHECK(Math::atan2(450.0, 1.0) == doctest::Approx(1.5685741082));
}

TEST_CASE("[Math] pow/log/log2/exp/sqrt") {
	CHECK(Math::pow(-0.1, 2.0) == doctest::Approx(0.01));
	CHECK(Math::pow(0.1, 2.5) == doctest::Approx(0.0031622777));
	CHECK(Math::pow(0.5, 0.5) == doctest::Approx(0.7071067812));
	CHECK(Math::pow(1.0, 1.0) == doctest::Approx(1.0));
	CHECK(Math::pow(1.5, -1.0) == doctest::Approx(0.6666666667));
	CHECK(Math::pow(450.0, -2.0) == doctest::Approx(0.0000049383));
	CHECK(Math::pow(450.0, 0.0) == doctest::Approx(1.0));

	CHECK(Math::is_nan(Math::log(-0.1)));
	CHECK(Math::log(0.1) == doctest::Approx(-2.302585093));
	CHECK(Math::log(0.5) == doctest::Approx(-0.6931471806));
	CHECK(Math::log(1.0) == doctest::Approx(0.0));
	CHECK(Math::log(1.5) == doctest::Approx(0.4054651081));
	CHECK(Math::log(450.0) == doctest::Approx(6.1092475828));

	CHECK(Math::is_nan(Math::log2(-0.1)));
	CHECK(Math::log2(0.1) == doctest::Approx(-3.3219280949));
	CHECK(Math::log2(0.5) == doctest::Approx(-1.0));
	CHECK(Math::log2(1.0) == doctest::Approx(0.0));
	CHECK(Math::log2(1.5) == doctest::Approx(0.5849625007));
	CHECK(Math::log2(450.0) == doctest::Approx(8.8137811912));

	CHECK(Math::exp(-0.1) == doctest::Approx(0.904837418));
	CHECK(Math::exp(0.1) == doctest::Approx(1.1051709181));
	CHECK(Math::exp(0.5) == doctest::Approx(1.6487212707));
	CHECK(Math::exp(1.0) == doctest::Approx(2.7182818285));
	CHECK(Math::exp(1.5) == doctest::Approx(4.4816890703));

	CHECK(Math::is_nan(Math::sqrt(-0.1)));
	CHECK(Math::sqrt(0.1) == doctest::Approx(0.316228));
	CHECK(Math::sqrt(0.5) == doctest::Approx(0.707107));
	CHECK(Math::sqrt(1.0) == doctest::Approx(1.0));
	CHECK(Math::sqrt(1.5) == doctest::Approx(1.224745));
}

TEST_CASE("[Math] is_nan/is_inf") {
	CHECK(!Math::is_nan(0.0));
	CHECK(Math::is_nan(NAN));

	CHECK(!Math::is_inf(0.0));
	CHECK(Math::is_inf(INFINITY));
}

TEST_CASE("[Math] linear_to_db") {
	CHECK(Math::linear_to_db(1.0) == doctest::Approx(0.0));
	CHECK(Math::linear_to_db(20.0) == doctest::Approx(26.0206));
	CHECK(Math::is_inf(Math::linear_to_db(0.0)));
	CHECK(Math::is_nan(Math::linear_to_db(-20.0)));
}

TEST_CASE("[Math] db_to_linear") {
	CHECK(Math::db_to_linear(0.0) == doctest::Approx(1.0));
	CHECK(Math::db_to_linear(1.0) == doctest::Approx(1.122018));
	CHECK(Math::db_to_linear(20.0) == doctest::Approx(10.0));
	CHECK(Math::db_to_linear(-20.0) == doctest::Approx(0.1));
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
	CHECK(Math::lerp(2.0, 5.0, -0.1) == doctest::Approx(1.7));
	CHECK(Math::lerp(2.0, 5.0, 0.0) == doctest::Approx(2.0));
	CHECK(Math::lerp(2.0, 5.0, 0.1) == doctest::Approx(2.3));
	CHECK(Math::lerp(2.0, 5.0, 1.0) == doctest::Approx(5.0));
	CHECK(Math::lerp(2.0, 5.0, 2.0) == doctest::Approx(8.0));

	CHECK(Math::lerp(-2.0, -5.0, -0.1) == doctest::Approx(-1.7));
	CHECK(Math::lerp(-2.0, -5.0, 0.0) == doctest::Approx(-2.0));
	CHECK(Math::lerp(-2.0, -5.0, 0.1) == doctest::Approx(-2.3));
	CHECK(Math::lerp(-2.0, -5.0, 1.0) == doctest::Approx(-5.0));
	CHECK(Math::lerp(-2.0, -5.0, 2.0) == doctest::Approx(-8.0));
}

TEST_CASE("[Math] inverse_lerp") {
	CHECK(Math::inverse_lerp(2.0, 5.0, 1.7) == doctest::Approx(-0.1));
	CHECK(Math::inverse_lerp(2.0, 5.0, 2.0) == doctest::Approx(0.0));
	CHECK(Math::inverse_lerp(2.0, 5.0, 2.3) == doctest::Approx(0.1));
	CHECK(Math::inverse_lerp(2.0, 5.0, 5.0) == doctest::Approx(1.0));
	CHECK(Math::inverse_lerp(2.0, 5.0, 8.0) == doctest::Approx(2.0));

	CHECK(Math::inverse_lerp(-2.0, -5.0, -1.7) == doctest::Approx(-0.1));
	CHECK(Math::inverse_lerp(-2.0, -5.0, -2.0) == doctest::Approx(0.0));
	CHECK(Math::inverse_lerp(-2.0, -5.0, -2.3) == doctest::Approx(0.1));
	CHECK(Math::inverse_lerp(-2.0, -5.0, -5.0) == doctest::Approx(1.0));
	CHECK(Math::inverse_lerp(-2.0, -5.0, -8.0) == doctest::Approx(2.0));
}

TEST_CASE("[Math] remap") {
	CHECK(Math::remap(50.0, 100.0, 200.0, 0.0, 1000.0) == doctest::Approx(-500.0));
	CHECK(Math::remap(100.0, 100.0, 200.0, 0.0, 1000.0) == doctest::Approx(0.0));
	CHECK(Math::remap(200.0, 100.0, 200.0, 0.0, 1000.0) == doctest::Approx(1000.0));
	CHECK(Math::remap(250.0, 100.0, 200.0, 0.0, 1000.0) == doctest::Approx(1500.0));

	CHECK(Math::remap(-50.0, -100.0, -200.0, 0.0, 1000.0) == doctest::Approx(-500.0));
	CHECK(Math::remap(-100.0, -100.0, -200.0, 0.0, 1000.0) == doctest::Approx(0.0));
	CHECK(Math::remap(-200.0, -100.0, -200.0, 0.0, 1000.0) == doctest::Approx(1000.0));
	CHECK(Math::remap(-250.0, -100.0, -200.0, 0.0, 1000.0) == doctest::Approx(1500.0));

	CHECK(Math::remap(-50.0, -100.0, -200.0, 0.0, -1000.0) == doctest::Approx(500.0));
	CHECK(Math::remap(-100.0, -100.0, -200.0, 0.0, -1000.0) == doctest::Approx(0.0));
	CHECK(Math::remap(-200.0, -100.0, -200.0, 0.0, -1000.0) == doctest::Approx(-1000.0));
	CHECK(Math::remap(-250.0, -100.0, -200.0, 0.0, -1000.0) == doctest::Approx(-1500.0));
}

TEST_CASE("[Math] lerp_angle") {
	// Counter-clockwise rotation.
	CHECK(Math::lerp_angle(0.24 * Math_TAU, 0.75 * Math_TAU, 0.5) == doctest::Approx(-0.005 * Math_TAU));
	// Counter-clockwise rotation.
	CHECK(Math::lerp_angle(0.25 * Math_TAU, 0.75 * Math_TAU, 0.5) == doctest::Approx(0.0));
	// Clockwise rotation.
	CHECK(Math::lerp_angle(0.26 * Math_TAU, 0.75 * Math_TAU, 0.5) == doctest::Approx(0.505 * Math_TAU));

	CHECK(Math::lerp_angle(-0.25 * Math_TAU, 1.25 * Math_TAU, 0.5) == doctest::Approx(-0.5 * Math_TAU));
	CHECK(Math::lerp_angle(0.72 * Math_TAU, 1.44 * Math_TAU, 0.96) == doctest::Approx(0.4512 * Math_TAU));
	CHECK(Math::lerp_angle(0.72 * Math_TAU, 1.44 * Math_TAU, 1.04) == doctest::Approx(0.4288 * Math_TAU));

	// Initial and final angles are effectively identical, so the value returned
	// should always be the same regardless of the `weight` parameter.
	CHECK(Math::lerp_angle(-4 * Math_TAU, 4 * Math_TAU, -1.0) == doctest::Approx(-4.0 * Math_TAU));
	CHECK(Math::lerp_angle(-4 * Math_TAU, 4 * Math_TAU, 0.0) == doctest::Approx(-4.0 * Math_TAU));
	CHECK(Math::lerp_angle(-4 * Math_TAU, 4 * Math_TAU, 0.5) == doctest::Approx(-4.0 * Math_TAU));
	CHECK(Math::lerp_angle(-4 * Math_TAU, 4 * Math_TAU, 1.0) == doctest::Approx(-4.0 * Math_TAU));
	CHECK(Math::lerp_angle(-4 * Math_TAU, 4 * Math_TAU, 500.0) == doctest::Approx(-4.0 * Math_TAU));
}

TEST_CASE("[Math] move_toward") {
	CHECK(Math::move_toward(2.0, 5.0, -1.0) == doctest::Approx(1.0));
	CHECK(Math::move_toward(2.0, 5.0, 2.5) == doctest::Approx(4.5));
	CHECK(Math::move_toward(2.0, 5.0, 4.0) == doctest::Approx(5.0));
	CHECK(Math::move_toward(-2.0, -5.0, -1.0) == doctest::Approx(-1.0));
	CHECK(Math::move_toward(-2.0, -5.0, 2.5) == doctest::Approx(-4.5));
	CHECK(Math::move_toward(-2.0, -5.0, 4.0) == doctest::Approx(-5.0));
}

TEST_CASE("[Math] smoothstep") {
	CHECK(Math::smoothstep(0.0, 2.0, -5.0) == doctest::Approx(0.0));
	CHECK(Math::smoothstep(0.0, 2.0, 0.5) == doctest::Approx(0.15625));
	CHECK(Math::smoothstep(0.0, 2.0, 1.0) == doctest::Approx(0.5));
	CHECK(Math::smoothstep(0.0, 2.0, 2.0) == doctest::Approx(1.0));
}

TEST_CASE("[Math] ease") {
	CHECK(Math::ease(0.1, 1.0) == doctest::Approx(0.1));
	CHECK(Math::ease(0.1, 2.0) == doctest::Approx(0.01));
	CHECK(Math::ease(0.1, 0.5) == doctest::Approx(0.19));
	CHECK(Math::ease(0.1, 0.0) == doctest::Approx(0));
	CHECK(Math::ease(0.1, -0.5) == doctest::Approx(0.2236067977));
	CHECK(Math::ease(0.1, -1.0) == doctest::Approx(0.1));
	CHECK(Math::ease(0.1, -2.0) == doctest::Approx(0.02));

	CHECK(Math::ease(-1.0, 1.0) == doctest::Approx(0));
	CHECK(Math::ease(-1.0, 2.0) == doctest::Approx(0));
	CHECK(Math::ease(-1.0, 0.5) == doctest::Approx(0));
	CHECK(Math::ease(-1.0, 0.0) == doctest::Approx(0));
	CHECK(Math::ease(-1.0, -0.5) == doctest::Approx(0));
	CHECK(Math::ease(-1.0, -1.0) == doctest::Approx(0));
	CHECK(Math::ease(-1.0, -2.0) == doctest::Approx(0));
}

TEST_CASE("[Math] snapped") {
	CHECK(Math::snapped(0.5, 0.04) == doctest::Approx(0.52));
	CHECK(Math::snapped(-0.5, 0.04) == doctest::Approx(-0.48));
	CHECK(Math::snapped(0.0, 0.04) == doctest::Approx(0));
	CHECK(Math::snapped(128'000.025, 0.04) == doctest::Approx(128'000.04));

	CHECK(Math::snapped(0.5, 400) == doctest::Approx(0));
	CHECK(Math::snapped(-0.5, 400) == doctest::Approx(0));
	CHECK(Math::snapped(0.0, 400) == doctest::Approx(0));
	CHECK(Math::snapped(128'000.025, 400) == doctest::Approx(128'000.0));

	CHECK(Math::snapped(0.5, 0.0) == doctest::Approx(0.5));
	CHECK(Math::snapped(-0.5, 0.0) == doctest::Approx(-0.5));
	CHECK(Math::snapped(0.0, 0.0) == doctest::Approx(0.0));
	CHECK(Math::snapped(128'000.025, 0.0) == doctest::Approx(128'000.0));

	CHECK(Math::snapped(0.5, -1.0) == doctest::Approx(0));
	CHECK(Math::snapped(-0.5, -1.0) == doctest::Approx(-1.0));
	CHECK(Math::snapped(0.0, -1.0) == doctest::Approx(0));
	CHECK(Math::snapped(128'000.025, -1.0) == doctest::Approx(128'000.0));
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
	CHECK(Math::fmod(-2.0, 0.3) == doctest::Approx(-0.2));
	CHECK(Math::fmod(0.0, 0.3) == doctest::Approx(0.0));
	CHECK(Math::fmod(2.0, 0.3) == doctest::Approx(0.2));

	CHECK(Math::fmod(-2.0, -0.3) == doctest::Approx(-0.2));
	CHECK(Math::fmod(0.0, -0.3) == doctest::Approx(0.0));
	CHECK(Math::fmod(2.0, -0.3) == doctest::Approx(0.2));
}

TEST_CASE("[Math] fposmod") {
	CHECK(Math::fposmod(-2.0, 0.3) == doctest::Approx(0.1));
	CHECK(Math::fposmod(0.0, 0.3) == doctest::Approx(0.0));
	CHECK(Math::fposmod(2.0, 0.3) == doctest::Approx(0.2));

	CHECK(Math::fposmod(-2.0, -0.3) == doctest::Approx(-0.2));
	CHECK(Math::fposmod(0.0, -0.3) == doctest::Approx(0.0));
	CHECK(Math::fposmod(2.0, -0.3) == doctest::Approx(-0.1));
}

TEST_CASE("[Math] fposmodp") {
	CHECK(Math::fposmodp(-2.0, 0.3) == doctest::Approx(0.1));
	CHECK(Math::fposmodp(0.0, 0.3) == doctest::Approx(0.0));
	CHECK(Math::fposmodp(2.0, 0.3) == doctest::Approx(0.2));

	CHECK(Math::fposmodp(-2.0, -0.3) == doctest::Approx(-0.5));
	CHECK(Math::fposmodp(0.0, -0.3) == doctest::Approx(0.0));
	CHECK(Math::fposmodp(2.0, -0.3) == doctest::Approx(0.2));
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
	CHECK(Math::wrapf(-30.0, -20.0, 160.0) == doctest::Approx(150.0));
	CHECK(Math::wrapf(30.0, -2.0, 160.0) == doctest::Approx(30.0));
	CHECK(Math::wrapf(300.0, -20.0, 160.0) == doctest::Approx(120.0));
	CHECK(Math::wrapf(300'000'000'000.0, -20.0, 160.0) == doctest::Approx(120.0));
}

TEST_CASE("[Math] fract") {
	CHECK(Math::fract(1.0) == doctest::Approx(0.0));
	CHECK(Math::fract(77.8) == doctest::Approx(0.8));
	CHECK(Math::fract(-10.1) == doctest::Approx(0.9));
}

TEST_CASE("[Math] pingpong") {
	CHECK(Math::pingpong(0.0, 0.0) == doctest::Approx(0.0));
	CHECK(Math::pingpong(1.0, 1.0) == doctest::Approx(1.0));
	CHECK(Math::pingpong(0.5, 2.0) == doctest::Approx(0.5));
	CHECK(Math::pingpong(3.5, 2.0) == doctest::Approx(0.5));
	CHECK(Math::pingpong(11.5, 2.0) == doctest::Approx(0.5));
	CHECK(Math::pingpong(-2.5, 2.0) == doctest::Approx(1.5));
}

TEST_CASE("[Math] deg_to_rad/rad_to_deg") {
	CHECK(Math::deg_to_rad(180.0) == doctest::Approx(Math_PI));
	CHECK(Math::deg_to_rad(-27.0) == doctest::Approx(-0.471239));

	CHECK(Math::rad_to_deg(Math_PI) == doctest::Approx(180.0));
	CHECK(Math::rad_to_deg(-1.5) == doctest::Approx(-85.94366927));
}

TEST_CASE("[Math] cubic_interpolate") {
	CHECK(Math::cubic_interpolate(0.2, 0.8, 0.0, 1.0, 0.0) == doctest::Approx(0.2));
	CHECK(Math::cubic_interpolate(0.2, 0.8, 0.0, 1.0, 0.25) == doctest::Approx(0.33125));
	CHECK(Math::cubic_interpolate(0.2, 0.8, 0.0, 1.0, 0.5) == doctest::Approx(0.5));
	CHECK(Math::cubic_interpolate(0.2, 0.8, 0.0, 1.0, 0.75) == doctest::Approx(0.66875));
	CHECK(Math::cubic_interpolate(0.2, 0.8, 0.0, 1.0, 1.0) == doctest::Approx(0.8));

	CHECK(Math::cubic_interpolate(20.2, 30.1, -100.0, 32.0, -50.0) == doctest::Approx(-6662732.3));
	CHECK(Math::cubic_interpolate(20.2, 30.1, -100.0, 32.0, -5.0) == doctest::Approx(-9356.3));
	CHECK(Math::cubic_interpolate(20.2, 30.1, -100.0, 32.0, 0.0) == doctest::Approx(20.2));
	CHECK(Math::cubic_interpolate(20.2, 30.1, -100.0, 32.0, 1.0) == doctest::Approx(30.1));
	CHECK(Math::cubic_interpolate(20.2, 30.1, -100.0, 32.0, 4.0) == doctest::Approx(1853.2));
}

TEST_CASE("[Math] cubic_interpolate_angle") {
	CHECK(Math::cubic_interpolate_angle(Math_PI * (1.0 / 6.0), Math_PI * (5.0 / 6.0), 0.0, Math_PI, 0.0) == doctest::Approx(Math_PI * (1.0 / 6.0)));
	CHECK(Math::cubic_interpolate_angle(Math_PI * (1.0 / 6.0), Math_PI * (5.0 / 6.0), 0.0, Math_PI, 0.25) == doctest::Approx(0.973566));
	CHECK(Math::cubic_interpolate_angle(Math_PI * (1.0 / 6.0), Math_PI * (5.0 / 6.0), 0.0, Math_PI, 0.5) == doctest::Approx(Math_PI / 2.0));
	CHECK(Math::cubic_interpolate_angle(Math_PI * (1.0 / 6.0), Math_PI * (5.0 / 6.0), 0.0, Math_PI, 0.75) == doctest::Approx(2.16803));
	CHECK(Math::cubic_interpolate_angle(Math_PI * (1.0 / 6.0), Math_PI * (5.0 / 6.0), 0.0, Math_PI, 1.0) == doctest::Approx(Math_PI * (5.0 / 6.0)));
}

TEST_CASE("[Math] cubic_interpolate_in_time") {
	CHECK(Math::cubic_interpolate_in_time(0.2, 0.8, 0.0, 1.0, 0.0, 0.5, 0.0, 1.0) == doctest::Approx(0.0));
	CHECK(Math::cubic_interpolate_in_time(0.2, 0.8, 0.0, 1.0, 0.25, 0.5, 0.0, 1.0) == doctest::Approx(0.1625));
	CHECK(Math::cubic_interpolate_in_time(0.2, 0.8, 0.0, 1.0, 0.5, 0.5, 0.0, 1.0) == doctest::Approx(0.4));
	CHECK(Math::cubic_interpolate_in_time(0.2, 0.8, 0.0, 1.0, 0.75, 0.5, 0.0, 1.0) == doctest::Approx(0.6375));
	CHECK(Math::cubic_interpolate_in_time(0.2, 0.8, 0.0, 1.0, 1.0, 0.5, 0.0, 1.0) == doctest::Approx(0.8));
}

TEST_CASE("[Math] cubic_interpolate_angle_in_time") {
	CHECK(Math::cubic_interpolate_angle_in_time(Math_PI * (1.0 / 6.0), Math_PI * (5.0 / 6.0), 0.0, Math_PI, 0.0, 0.5, 0.0, 1.0) == doctest::Approx(0.0));
	CHECK(Math::cubic_interpolate_angle_in_time(Math_PI * (1.0 / 6.0), Math_PI * (5.0 / 6.0), 0.0, Math_PI, 0.25, 0.5, 0.0, 1.0) == doctest::Approx(0.494964));
	CHECK(Math::cubic_interpolate_angle_in_time(Math_PI * (1.0 / 6.0), Math_PI * (5.0 / 6.0), 0.0, Math_PI, 0.5, 0.5, 0.0, 1.0) == doctest::Approx(1.27627));
	CHECK(Math::cubic_interpolate_angle_in_time(Math_PI * (1.0 / 6.0), Math_PI * (5.0 / 6.0), 0.0, Math_PI, 0.75, 0.5, 0.0, 1.0) == doctest::Approx(2.07394));
	CHECK(Math::cubic_interpolate_angle_in_time(Math_PI * (1.0 / 6.0), Math_PI * (5.0 / 6.0), 0.0, Math_PI, 1.0, 0.5, 0.0, 1.0) == doctest::Approx(Math_PI * (5.0 / 6.0)));
}

TEST_CASE("[Math] bezier_interpolate") {
	CHECK(Math::bezier_interpolate(0.0, 0.2, 0.8, 1.0, 0.0) == doctest::Approx(0.0));
	CHECK(Math::bezier_interpolate(0.0, 0.2, 0.8, 1.0, 0.25) == doctest::Approx(0.2125));
	CHECK(Math::bezier_interpolate(0.0, 0.2, 0.8, 1.0, 0.5) == doctest::Approx(0.5));
	CHECK(Math::bezier_interpolate(0.0, 0.2, 0.8, 1.0, 0.75) == doctest::Approx(0.7875));
	CHECK(Math::bezier_interpolate(0.0, 0.2, 0.8, 1.0, 1.0) == doctest::Approx(1.0));
}

} // namespace TestMath

#endif // TEST_MATH_FUNCS_H
