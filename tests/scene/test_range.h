/**************************************************************************/
/*  test_range.h                                                          */
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

#ifndef TEST_RANGE_H
#define TEST_RANGE_H

#include "scene/gui/range.h"

#include "tests/test_macros.h"

namespace TestRange {
static inline Array build_array() {
	return Array();
}
template <typename... Targs>
static inline Array build_array(Variant item, Targs... Fargs) {
	Array a = build_array(Fargs...);
	a.push_front(item);
	return a;
}

TEST_CASE("[SceneTree][Range] range control") {
	Range *range = memnew(Range);
	SceneTree::get_singleton()->get_root()->add_child(range);

	SIGNAL_WATCH(range, "value_changed");
	SIGNAL_WATCH(range, "changed");

	REQUIRE(range->get_value() == doctest::Approx(0.0));
	REQUIRE(range->get_min() == doctest::Approx(0.0));
	REQUIRE(range->get_max() == doctest::Approx(100.0));
	REQUIRE(range->get_step() == doctest::Approx(1.0));
	REQUIRE(range->get_page() == doctest::Approx(0.0));
	REQUIRE(range->is_lesser_allowed() == false);
	REQUIRE(range->is_greater_allowed() == false);
	REQUIRE(range->is_ratio_exp() == false);

	SUBCASE("[SceneTree][Range] set_value default step") {
		CHECK(range->get_step() == doctest::Approx(1.0));
		range->set_value(43.6);
		CHECK(range->get_value() == doctest::Approx(44.0));
		SIGNAL_CHECK("value_changed", build_array(build_array(44.0)));
		SIGNAL_CHECK_FALSE("changed");

		range->set_value(44.0);
		SIGNAL_CHECK_FALSE("value_changed");
	}

	SUBCASE("[SceneTree][Range] set_value zero step") {
		range->set_step(0);
		CHECK(range->get_step() == doctest::Approx(0.0));
		range->set_value(43.6);
		CHECK(range->get_step() == doctest::Approx(0.0));
		CHECK(range->get_value() == doctest::Approx(43.6));
		SIGNAL_CHECK("value_changed", build_array(build_array(43.6)));
		SIGNAL_CHECK("changed", build_array(build_array()));

		range->set_value(43.6);
		SIGNAL_CHECK_FALSE("value_changed");
	}

	SUBCASE("[SceneTree][Range] set_min zero step") {
		range->set_step(0);
		SIGNAL_DISCARD("changed");

		range->set_min(5.1);
		CHECK(range->get_min() == doctest::Approx(5.1));
		CHECK(range->get_value() == doctest::Approx(5.1));
		SIGNAL_CHECK("value_changed", build_array(build_array(5.1)));
		SIGNAL_CHECK("changed", build_array(build_array()));

		range->set_value(13.2);
		CHECK(range->get_value() == doctest::Approx(13.2));
		SIGNAL_CHECK("value_changed", build_array(build_array(13.2)));

		range->set_value(3.8);
		CHECK(range->get_value() == doctest::Approx(5.1));
		SIGNAL_CHECK("value_changed", build_array(build_array(5.1)));
		SIGNAL_CHECK_FALSE("changed");

		range->set_min(-43.9);
		range->set_value(13.2);
		CHECK(range->get_min() == doctest::Approx(-43.9));
		CHECK(range->get_value() == doctest::Approx(13.2));
		SIGNAL_CHECK("value_changed", build_array(build_array(13.2)));

		range->set_value(-34.5);
		CHECK(range->get_value() == doctest::Approx(-34.5));
		SIGNAL_CHECK("value_changed", build_array(build_array(-34.5)));

		range->set_value(-54.8);
		CHECK(range->get_value() == doctest::Approx(-43.9));
		SIGNAL_CHECK("value_changed", build_array(build_array(-43.9)));

		range->set_min(-16.6);
		CHECK(range->get_min() == doctest::Approx(-16.6));
		CHECK(range->get_value() == doctest::Approx(-16.6));
		SIGNAL_CHECK("value_changed", build_array(build_array(-16.6)));
	}

	SUBCASE("[SceneTree][Range] set_min with step") {
		range->set_step(3.4);
		CHECK(range->get_step() == doctest::Approx(3.4));
		SIGNAL_DISCARD("changed");

		range->set_min(5.1);
		CHECK(range->get_min() == doctest::Approx(5.1));
		CHECK(range->get_value() == doctest::Approx(5.1));
		SIGNAL_CHECK("value_changed", build_array(build_array(5.1)));
		SIGNAL_CHECK("changed", build_array(build_array()));

		range->set_value(13.2);
		CHECK(range->get_value() == doctest::Approx(11.9));

		range->set_value(3.8);
		CHECK(range->get_value() == doctest::Approx(5.1));

		range->set_min(-43.9);
		range->set_value(13.2);
		CHECK(range->get_value() == doctest::Approx(13.9));

		range->set_value(-34.5);
		CHECK(range->get_value() == doctest::Approx(-33.7));

		range->set_value(-54.8);
		CHECK(range->get_value() == doctest::Approx(-43.9));

		range->set_min(-16.6);
		CHECK(range->get_min() == doctest::Approx(-16.6));
		CHECK(range->get_value() == doctest::Approx(-16.6));
	}

	SUBCASE("[SceneTree][Range] set_min allow_lesser") {
		range->set_step(0);
		SIGNAL_DISCARD("changed");

		range->set_allow_lesser(true);
		CHECK(range->is_lesser_allowed() == true);

		range->set_min(5.1);
		CHECK(range->get_min() == doctest::Approx(5.1));
		CHECK(range->get_value() == doctest::Approx(0.0));
		SIGNAL_CHECK_FALSE("value_changed");
		SIGNAL_CHECK("changed", build_array(build_array()));

		range->set_value(3.8);
		CHECK(range->get_value() == doctest::Approx(3.8));
		SIGNAL_CHECK("value_changed", build_array(build_array(3.8)));
		SIGNAL_CHECK_FALSE("changed");

		range->set_min(-43.9);
		range->set_value(-54.8);
		CHECK(range->get_value() == doctest::Approx(-54.8));
		SIGNAL_CHECK("value_changed", build_array(build_array(-54.8)));

		range->set_allow_lesser(false);
		CHECK(range->is_lesser_allowed() == false);

		range->set_min(-16.6);
		CHECK(range->get_min() == doctest::Approx(-16.6));
		CHECK(range->get_value() == doctest::Approx(-16.6));
		SIGNAL_CHECK("value_changed", build_array(build_array(-16.6)));
	}

	SUBCASE("[SceneTree][Range] set_max zero step") {
		range->set_step(0);
		range->set_value(90);
		SIGNAL_DISCARD("changed");
		SIGNAL_DISCARD("value_changed");

		range->set_max(63.7);
		CHECK(range->get_max() == doctest::Approx(63.7));
		CHECK(range->get_value() == doctest::Approx(63.7));
		SIGNAL_CHECK("value_changed", build_array(build_array(63.7)));
		SIGNAL_CHECK("changed", build_array(build_array()));

		range->set_value(25.2);
		CHECK(range->get_value() == doctest::Approx(25.2));
		SIGNAL_CHECK("value_changed", build_array(build_array(25.2)));

		range->set_value(85.6);
		CHECK(range->get_value() == doctest::Approx(63.7));
		SIGNAL_CHECK("value_changed", build_array(build_array(63.7)));
		SIGNAL_CHECK_FALSE("changed");

		range->set_min(-100.0);
		range->set_max(-10.8);
		CHECK(range->get_max() == doctest::Approx(-10.8));
		CHECK(range->get_value() == doctest::Approx(-10.8));
		SIGNAL_CHECK("value_changed", build_array(build_array(-10.8)));

		range->set_value(-38.4);
		CHECK(range->get_value() == doctest::Approx(-38.4));
		SIGNAL_CHECK("value_changed", build_array(build_array(-38.4)));

		range->set_value(13.2);
		CHECK(range->get_value() == doctest::Approx(-10.8));
		SIGNAL_CHECK("value_changed", build_array(build_array(-10.8)));

		range->set_value(-5.1);
		CHECK(range->get_value() == doctest::Approx(-10.8));
		SIGNAL_CHECK_FALSE("value_changed");

		range->set_max(-16.6);
		CHECK(range->get_max() == doctest::Approx(-16.6));
		CHECK(range->get_value() == doctest::Approx(-16.6));
		SIGNAL_CHECK("value_changed", build_array(build_array(-16.6)));
	}

	SUBCASE("[SceneTree][Range] set_max with step") {
		range->set_step(3.4);
		range->set_value(90);
		CHECK(range->get_step() == doctest::Approx(3.4));
		SIGNAL_DISCARD("changed");
		SIGNAL_DISCARD("value_changed");

		range->set_max(63.7);
		CHECK(range->get_max() == doctest::Approx(63.7));
		CHECK(range->get_value() == doctest::Approx(63.7));
		SIGNAL_CHECK("value_changed", build_array(build_array(63.7)));
		SIGNAL_CHECK("changed", build_array(build_array()));

		range->set_value(25.2);
		CHECK(range->get_value() == doctest::Approx(23.8));

		range->set_value(85.6);
		CHECK(range->get_value() == doctest::Approx(63.7));

		range->set_min(-100.0);
		range->set_max(-10.8);
		CHECK(range->get_max() == doctest::Approx(-10.8));
		CHECK(range->get_value() == doctest::Approx(-10.8));

		range->set_value(-38.4);
		CHECK(range->get_value() == doctest::Approx(-38.8));

		range->set_value(13.2);
		CHECK(range->get_value() == doctest::Approx(-10.8));

		range->set_value(-5.1);
		CHECK(range->get_value() == doctest::Approx(-10.8));

		range->set_max(-16.6);
		CHECK(range->get_max() == doctest::Approx(-16.6));
		CHECK(range->get_value() == doctest::Approx(-16.6));
	}

	SUBCASE("[SceneTree][Range] set_max allow_greater") {
		range->set_step(0);
		range->set_value(90);
		SIGNAL_DISCARD("changed");
		SIGNAL_DISCARD("value_changed");

		range->set_allow_greater(true);
		CHECK(range->is_greater_allowed() == true);

		range->set_max(63.7);
		CHECK(range->get_max() == doctest::Approx(63.7));
		CHECK(range->get_value() == doctest::Approx(90.0));
		SIGNAL_CHECK("changed", build_array(build_array()));

		range->set_value(85.6);
		CHECK(range->get_value() == doctest::Approx(85.6));
		SIGNAL_CHECK("value_changed", build_array(build_array(85.6)));
		SIGNAL_CHECK_FALSE("changed");

		range->set_min(-100.0);
		range->set_max(-10.8);
		range->set_value(13.2);
		CHECK(range->get_value() == doctest::Approx(13.2));
		SIGNAL_CHECK("value_changed", build_array(build_array(13.2)));

		range->set_value(-5.1);
		CHECK(range->get_value() == doctest::Approx(-5.1));
		SIGNAL_CHECK("value_changed", build_array(build_array(-5.1)));

		range->set_allow_greater(false);
		CHECK(range->is_greater_allowed() == false);

		range->set_max(-16.6);
		CHECK(range->get_max() == doctest::Approx(-16.6));
		CHECK(range->get_value() == doctest::Approx(-16.6));
		SIGNAL_CHECK("value_changed", build_array(build_array(-16.6)));
	}

	SUBCASE("[SceneTree][Range] set_max with page") {
		range->set_page(7.2);
		CHECK(range->get_page() == doctest::Approx(7.2));
		SIGNAL_CHECK("changed", build_array(build_array()));

		range->set_value(90);
		SIGNAL_DISCARD("value_changed");

		range->set_max(63.7);
		CHECK(range->get_max() == doctest::Approx(63.7));
		CHECK(range->get_value() == doctest::Approx(56.5));
		SIGNAL_CHECK("value_changed", build_array(build_array(56.5)));
		SIGNAL_CHECK("changed", build_array(build_array()));

		range->set_value(85.6);
		CHECK(range->get_value() == doctest::Approx(56.5));

		range->set_min(-100.0);
		range->set_max(-10.8);
		CHECK(range->get_max() == doctest::Approx(-10.8));
		CHECK(range->get_value() == doctest::Approx(-18.0));

		range->set_value(-5.1);
		CHECK(range->get_value() == doctest::Approx(-18.0));

		range->set_max(-16.6);
		CHECK(range->get_max() == doctest::Approx(-16.6));
		CHECK(range->get_value() == doctest::Approx(-23.8));
	}

	SUBCASE("[SceneTree][Range] set_max with step and page") {
		range->set_step(8.1);
		range->set_page(2.3);
		CHECK(range->get_step() == doctest::Approx(8.1));
		CHECK(range->get_page() == doctest::Approx(2.3));
		SIGNAL_CHECK("changed", build_array(build_array(), build_array()));

		range->set_value(90);
		SIGNAL_DISCARD("value_changed");

		range->set_max(63.7);
		CHECK(range->get_max() == doctest::Approx(63.7));
		CHECK(range->get_value() == doctest::Approx(61.4));
		SIGNAL_CHECK("changed", build_array(build_array()));

		range->set_value(85.6);
		CHECK(range->get_value() == doctest::Approx(61.4));

		range->set_min(-100.0);
		range->set_max(-10.8);
		CHECK(range->get_max() == doctest::Approx(-10.8));
		CHECK(range->get_value() == doctest::Approx(-13.1));

		range->set_value(-5.1);
		CHECK(range->get_value() == doctest::Approx(-13.1));

		range->set_max(-16.6);
		CHECK(range->get_max() == doctest::Approx(-16.6));
		CHECK(range->get_value() == doctest::Approx(-18.9));

		// This results in a different value than set_max() above because
		// step is calulcated before page.
		range->set_value(-16.6);
		CHECK(range->get_value() == doctest::Approx(-19.0));
	}

	SUBCASE("[SceneTree][Range] set_value large step") {
		range->set_max(10000000000000.0);
		range->set_step(72389512.0);
		CHECK(range->get_step() == doctest::Approx(72389512.0));

		range->set_value(43.6);
		CHECK(range->get_value() == doctest::Approx(0));

		range->set_value(58395727.0);
		CHECK(range->get_value() == doctest::Approx(72389512.0));

		range->set_value(92957238.0);
		CHECK(range->get_value() == doctest::Approx(72389512.0));

		range->set_value(1094218303.0);
		CHECK(range->get_value() == doctest::Approx(1085842680.0));
	}

	SUBCASE("[SceneTree][Range] ratio zero step") {
		range->set_step(0);

		CHECK(range->get_as_ratio() == doctest::Approx(0.0));

		range->set_value(50.0);
		CHECK(range->get_as_ratio() == doctest::Approx(0.5));

		range->set_value(100.0);
		CHECK(range->get_as_ratio() == doctest::Approx(1.0));

		range->set_as_ratio(0.0);
		CHECK(range->get_value() == doctest::Approx(0.0));

		range->set_as_ratio(0.5);
		CHECK(range->get_value() == doctest::Approx(50.0));

		range->set_as_ratio(1.0);
		CHECK(range->get_value() == doctest::Approx(100.0));

		range->set_min(30.0);
		range->set_max(145.0);
		range->set_as_ratio(0.4);
		CHECK(range->get_value() == doctest::Approx(76.0));

		range->set_value(92.0);
		CHECK(range->get_as_ratio() == doctest::Approx(0.53913043478));

		range->set_min(-234.0);
		range->set_max(-47.0);
		range->set_as_ratio(0.8);
		CHECK(range->get_value() == doctest::Approx(-84.4));

		range->set_value(-199.0);
		CHECK(range->get_as_ratio() == doctest::Approx(0.1871657754));

		range->set_min(-193.67);
		range->set_max(68.29);
		range->set_as_ratio(0.14);
		CHECK(range->get_value() == doctest::Approx(-156.9956));

		range->set_value(14.5);
		CHECK(range->get_as_ratio() == doctest::Approx(0.79466330737));

		range->set_as_ratio(0.0);
		CHECK(range->get_value() == doctest::Approx(-193.67));

		range->set_as_ratio(1.0);
		CHECK(range->get_value() == doctest::Approx(68.29));

		range->set_as_ratio(7.0);
		CHECK(range->get_value() == doctest::Approx(68.29));

		range->set_as_ratio(-5.0);
		CHECK(range->get_value() == doctest::Approx(-193.67));

		range->set_allow_lesser(true);
		range->set_allow_greater(true);

		range->set_value(-5435.0);
		CHECK(range->get_as_ratio() == doctest::Approx(0.0));

		range->set_value(1926.0);
		CHECK(range->get_as_ratio() == doctest::Approx(1.0));
	}

	SUBCASE("[SceneTree][Range] ratio with step") {
		range->set_step(3);

		CHECK(range->get_as_ratio() == doctest::Approx(0.0));

		range->set_value(50.0);
		CHECK(range->get_value() == doctest::Approx(51.0));
		CHECK(range->get_as_ratio() == doctest::Approx(0.51));

		range->set_value(100.0);
		CHECK(range->get_value() == doctest::Approx(99.0));
		CHECK(range->get_as_ratio() == doctest::Approx(0.99));

		range->set_as_ratio(0.0);
		CHECK(range->get_value() == doctest::Approx(0.0));

		range->set_as_ratio(0.5);
		CHECK(range->get_value() == doctest::Approx(51.0));

		range->set_as_ratio(1.0);
		CHECK(range->get_value() == doctest::Approx(99.0));

		range->set_min(30.0);
		range->set_max(145.0);
		range->set_as_ratio(0.4);
		CHECK(range->get_value() == doctest::Approx(75.0));

		range->set_value(92.0);
		CHECK(range->get_value() == doctest::Approx(93.0));
		CHECK(range->get_as_ratio() == doctest::Approx(0.54782608696));

		range->set_min(-234.0);
		range->set_max(-47.0);
		range->set_as_ratio(0.8);
		CHECK(range->get_value() == doctest::Approx(-84));

		range->set_value(-199.0);
		CHECK(range->get_value() == doctest::Approx(-198.0));
		CHECK(range->get_as_ratio() == doctest::Approx(0.19251336898));

		range->set_min(-193.67);
		range->set_max(68.29);
		range->set_as_ratio(0.14);
		CHECK(range->get_value() == doctest::Approx(-157.67));

		range->set_value(14.5);
		CHECK(range->get_value() == doctest::Approx(13.33));
		CHECK(range->get_as_ratio() == doctest::Approx(0.79019697664));

		range->set_as_ratio(0.0);
		CHECK(range->get_value() == doctest::Approx(-193.67));

		range->set_as_ratio(1.0);
		CHECK(range->get_value() == doctest::Approx(67.33));

		range->set_as_ratio(7.0);
		CHECK(range->get_value() == doctest::Approx(67.33));

		range->set_as_ratio(-5.0);
		CHECK(range->get_value() == doctest::Approx(-193.67));
	}

	SUBCASE("[SceneTree][Range] ratio as exp") {
		range->set_step(0);
		range->set_exp_ratio(true);

		CHECK(range->get_as_ratio() == doctest::Approx(0.0));

		range->set_value(50.0);
		CHECK(range->get_as_ratio() == doctest::Approx(0.84948500217));

		range->set_value(100.0);
		CHECK(range->get_as_ratio() == doctest::Approx(1.0));

		range->set_as_ratio(0.0);
		CHECK(range->get_value() == doctest::Approx(1.0));

		range->set_as_ratio(0.5);
		CHECK(range->get_value() == doctest::Approx(10.0));

		range->set_as_ratio(1.0);
		CHECK(range->get_value() == doctest::Approx(100.0));

		range->set_min(30.0);
		range->set_max(145.0);
		range->set_as_ratio(0.4);
		CHECK(range->get_value() == doctest::Approx(56.340403594));

		range->set_value(92.0);
		CHECK(range->get_as_ratio() == doctest::Approx(0.71124426151));

		// TODO: Test negative values once behavior is well defined.
	}

	SUBCASE("[SceneTree][Range] share") {
		range->set_value(38.0);
		range->set_min(10.0);
		range->set_max(75.0);
		range->set_step(4.0);
		range->set_page(3.0);
		range->set_exp_ratio(true);
		range->set_allow_greater(true);
		range->set_allow_lesser(true);

		Range *alt1 = memnew(Range);
		SceneTree::get_singleton()->get_root()->add_child(alt1);

		range->share(alt1);
		CHECK(alt1->get_value() == doctest::Approx(38.0));
		CHECK(alt1->get_min() == doctest::Approx(10.0));
		CHECK(alt1->get_max() == doctest::Approx(75.0));
		CHECK(alt1->get_step() == doctest::Approx(4.0));
		CHECK(alt1->get_page() == doctest::Approx(3.0));
		CHECK(alt1->is_ratio_exp() == true);
		CHECK(alt1->is_greater_allowed() == true);
		CHECK(alt1->is_lesser_allowed() == true);

		range->set_step(5.4);
		CHECK(alt1->get_step() == doctest::Approx(5.4));

		SIGNAL_DISCARD("changed");
		alt1->set_max(87.0);
		CHECK(range->get_max() == doctest::Approx(87.0));
		SIGNAL_CHECK("changed", build_array(build_array()));

		SIGNAL_DISCARD("value_changed");
		alt1->set_value(64.0);
		SIGNAL_CHECK("value_changed", build_array(build_array(64.0)));

		Range *alt2 = memnew(Range);
		SceneTree::get_singleton()->get_root()->add_child(alt2);

		alt2->set_page(7.5);
		alt1->share(alt2);
		CHECK(alt2->get_page() == doctest::Approx(3.0));

		alt2->set_min(9.0);
		CHECK(range->get_min() == doctest::Approx(9.0));
		CHECK(alt1->get_min() == doctest::Approx(9.0));

		alt2->set_step(0.0);
		SIGNAL_DISCARD("value_changed");
		alt2->set_value(47.5);
		CHECK(range->get_value() == doctest::Approx(47.5));
		SIGNAL_CHECK("value_changed", build_array(build_array(47.5)));

		alt1->unshare();
		alt1->set_allow_lesser(false);
		CHECK(range->is_lesser_allowed() == true);
		CHECK(alt1->is_lesser_allowed() == false);
		CHECK(alt2->is_lesser_allowed() == true);

		SIGNAL_DISCARD("value_changed");
		alt1->set_value(17.9);
		CHECK(range->get_value() == doctest::Approx(47.5));
		SIGNAL_CHECK_FALSE("value_changed");

		SceneTree::get_singleton()->get_root()->remove_child(alt1);
		SceneTree::get_singleton()->get_root()->remove_child(alt2);
		memdelete(alt1);
		memdelete(alt2);
	}

	SUBCASE("[SceneTree][Range] use rounded") {
		range->set_step(3.4);
		range->set_use_rounded_values(true);

		range->set_value(47.6);
		CHECK(range->get_value() == doctest::Approx(48.0));
		SIGNAL_CHECK("value_changed", build_array(build_array(48.0)));

		range->set_value(37.4);
		CHECK(range->get_value() == doctest::Approx(37.0));
		SIGNAL_CHECK("value_changed", build_array(build_array(37.0)));

		// We round to closest step (20.4) before closest integer.
		range->set_value(20.7);
		CHECK(range->get_value() == doctest::Approx(20.0));
		SIGNAL_CHECK("value_changed", build_array(build_array(20.0)));

		range->set_use_rounded_values(false);

		range->set_value(47.6);
		CHECK(range->get_value() == doctest::Approx(47.6));
		SIGNAL_CHECK("value_changed", build_array(build_array(47.6)));

		range->set_value(37.4);
		CHECK(range->get_value() == doctest::Approx(37.4));
		SIGNAL_CHECK("value_changed", build_array(build_array(37.4)));

		range->set_value(20.7);
		CHECK(range->get_value() == doctest::Approx(20.4));
		SIGNAL_CHECK("value_changed", build_array(build_array(20.4)));
	}

	SIGNAL_UNWATCH(range, "value_changed");
	SIGNAL_UNWATCH(range, "changed");

	SceneTree::get_singleton()->get_root()->remove_child(range);
	memdelete(range);
}

TEST_CASE("[SceneTree][RangeInt] range control, integer") {
	RangeInt *range = memnew(RangeInt);
	SceneTree::get_singleton()->get_root()->add_child(range);

	SIGNAL_WATCH(range, "value_changed");
	SIGNAL_WATCH(range, "changed");

	REQUIRE(range->get_value() == doctest::Approx(0.0));
	REQUIRE(range->get_min() == doctest::Approx(0.0));
	REQUIRE(range->get_max() == doctest::Approx(100.0));
	REQUIRE(range->get_step() == doctest::Approx(1.0));
	REQUIRE(range->get_page() == doctest::Approx(0.0));
	REQUIRE(range->is_lesser_allowed() == false);
	REQUIRE(range->is_greater_allowed() == false);
	REQUIRE(range->is_ratio_exp() == false);

	SUBCASE("[SceneTree][RangeInt] set_value default step") {
		CHECK(range->get_step() == 1);
		range->set_value(43.6);
		CHECK(range->get_value() == 43);
		SIGNAL_CHECK("value_changed", build_array(build_array(43)));
		SIGNAL_CHECK_FALSE("changed");

		range->set_value(43);
		SIGNAL_CHECK_FALSE("value_changed");
	}

	SUBCASE("[SceneTree][RangeInt] set_value zero step") {
		range->set_step(0);
		CHECK(range->get_step() == 0);
		range->set_value(43.6);
		CHECK(range->get_value() == 43);
		SIGNAL_CHECK("value_changed", build_array(build_array(43)));
		SIGNAL_CHECK("changed", build_array(build_array()));

		range->set_value(43.6);
		SIGNAL_CHECK_FALSE("value_changed");
	}

	// TODO: Better testing of large step values

	SUBCASE("[SceneTree][RangeInt] set_min zero step") {
		range->set_step(0);
		SIGNAL_DISCARD("changed")

		range->set_min(5.1);
		CHECK(range->get_min() == 5);
		SIGNAL_CHECK("value_changed", build_array(build_array(5)));
		SIGNAL_CHECK("changed", build_array(build_array()));

		range->set_value(13.2);
		CHECK(range->get_value() == 13);
		SIGNAL_CHECK("value_changed", build_array(build_array(13)));

		range->set_value(3.8);
		CHECK(range->get_value() == 5);
		SIGNAL_CHECK("value_changed", build_array(build_array(5)));

		range->set_min(-43.9);
		range->set_value(13.2);
		CHECK(range->get_min() == -43);
		CHECK(range->get_value() == 13);
		SIGNAL_CHECK("value_changed", build_array(build_array(13)));

		range->set_value(-34.5);
		CHECK(range->get_value() == -34);
		SIGNAL_CHECK("value_changed", build_array(build_array(-34)));

		range->set_value(-54.8);
		CHECK(range->get_value() == -43);
		SIGNAL_CHECK("value_changed", build_array(build_array(-43)));

		range->set_min(-16.6);
		CHECK(range->get_min() == -16);
		CHECK(range->get_value() == -16);
		SIGNAL_CHECK("value_changed", build_array(build_array(-16)));
	}

	SUBCASE("[SceneTree][RangeInt] set_min with step") {
		range->set_step(3.4);
		CHECK(range->get_step() == 3);
		SIGNAL_DISCARD("changed")

		range->set_min(5.1);
		CHECK(range->get_min() == 5);
		CHECK(range->get_value() == 5);
		SIGNAL_CHECK("value_changed", build_array(build_array(5)));
		SIGNAL_CHECK("changed", build_array(build_array()));

		range->set_value(13.2);
		CHECK(range->get_value() == 14);

		range->set_value(3.8);
		CHECK(range->get_value() == 5);

		range->set_min(-43.9);
		range->set_value(13.2);
		CHECK(range->get_value() == 14);

		range->set_value(-34.5);
		CHECK(range->get_value() == -34);

		range->set_value(-54.8);
		CHECK(range->get_value() == -43);

		range->set_min(-16.6);
		CHECK(range->get_min() == -16);
		CHECK(range->get_value() == -16);
	}

	SUBCASE("[SceneTree][RangeInt] set_min allow_lesser") {
		range->set_step(0);
		SIGNAL_DISCARD("changed");

		range->set_allow_lesser(true);
		CHECK(range->is_lesser_allowed() == true);

		range->set_min(5.1);
		CHECK(range->get_min() == 5);
		CHECK(range->get_value() == 0);
		SIGNAL_CHECK_FALSE("value_changed");
		SIGNAL_CHECK("changed", build_array(build_array()));

		range->set_value(3.8);
		CHECK(range->get_value() == 3);
		SIGNAL_CHECK("value_changed", build_array(build_array(3)));
		SIGNAL_CHECK_FALSE("changed");

		range->set_min(-43.9);
		range->set_value(-54.8);
		CHECK(range->get_value() == -54);
		SIGNAL_CHECK("value_changed", build_array(build_array(-54)));

		range->set_allow_lesser(false);
		CHECK(range->is_lesser_allowed() == false);

		range->set_min(-16.6);
		CHECK(range->get_min() == -16);
		CHECK(range->get_value() == -16);
		SIGNAL_CHECK("value_changed", build_array(build_array(-16)));
	}

	SUBCASE("[SceneTree][RangeInt] set_max zero step") {
		range->set_step(0);
		range->set_value(90);
		SIGNAL_DISCARD("changed");
		SIGNAL_DISCARD("value_changed");

		range->set_max(63.7);
		CHECK(range->get_max() == 63);
		CHECK(range->get_value() == 63);
		SIGNAL_CHECK("value_changed", build_array(build_array(63)));
		SIGNAL_CHECK("changed", build_array(build_array()));

		range->set_value(25);
		CHECK(range->get_value() == 25);
		SIGNAL_CHECK("value_changed", build_array(build_array(25)));

		range->set_value(85.6);
		CHECK(range->get_value() == 63);
		SIGNAL_CHECK("value_changed", build_array(build_array(63)));
		SIGNAL_CHECK_FALSE("changed");

		range->set_min(-100.0);
		range->set_max(-10.8);
		CHECK(range->get_max() == -10);
		CHECK(range->get_value() == -10);
		SIGNAL_CHECK("value_changed", build_array(build_array(-10)));

		range->set_value(-38.4);
		CHECK(range->get_value() == -38);
		SIGNAL_CHECK("value_changed", build_array(build_array(-38)));

		range->set_value(13.2);
		CHECK(range->get_value() == -10);
		SIGNAL_CHECK("value_changed", build_array(build_array(-10)));

		range->set_value(-5.1);
		CHECK(range->get_value() == -10);
		SIGNAL_CHECK_FALSE("value_changed");

		range->set_max(-16.6);
		CHECK(range->get_max() == -16);
		CHECK(range->get_value() == -16);
		SIGNAL_CHECK("value_changed", build_array(build_array(-16)));
	}

	SUBCASE("[SceneTree][RangeInt] set_max with step") {
		range->set_step(3.4);
		range->set_value(90);
		CHECK(range->get_step() == 3);
		SIGNAL_DISCARD("changed");
		SIGNAL_DISCARD("value_changed");

		range->set_max(63.7);
		CHECK(range->get_max() == 63);
		CHECK(range->get_value() == 63);
		SIGNAL_CHECK("value_changed", build_array(build_array(63)));
		SIGNAL_CHECK("changed", build_array(build_array()));

		range->set_value(25.2);
		CHECK(range->get_value() == 24);

		range->set_value(85.6);
		CHECK(range->get_value() == 63);

		range->set_min(-100.0);
		range->set_max(-10.8);
		CHECK(range->get_max() == -10);
		CHECK(range->get_value() == -10);

		range->set_value(-38.4);
		CHECK(range->get_value() == -37);

		range->set_value(13.2);
		CHECK(range->get_value() == -10);

		range->set_value(-5.1);
		CHECK(range->get_value() == -10);

		range->set_max(-16.6);
		CHECK(range->get_max() == -16);
		CHECK(range->get_value() == -16);
	}

	SUBCASE("[SceneTree][RangeInt] set_max allow_greater") {
		range->set_step(0);
		range->set_value(90);
		SIGNAL_DISCARD("changed");
		SIGNAL_DISCARD("value_changed");

		range->set_allow_greater(true);
		CHECK(range->is_greater_allowed() == true);

		range->set_max(63.7);
		CHECK(range->get_max() == 63);
		CHECK(range->get_value() == 90);
		SIGNAL_CHECK("changed", build_array(build_array()));

		range->set_value(85.6);
		CHECK(range->get_value() == 85);
		SIGNAL_CHECK("value_changed", build_array(build_array(85)));
		SIGNAL_CHECK_FALSE("changed");

		range->set_min(-100.0);
		range->set_max(-10.8);
		range->set_value(13.2);
		CHECK(range->get_value() == 13);
		SIGNAL_CHECK("value_changed", build_array(build_array(13)));

		range->set_value(-5.1);
		CHECK(range->get_value() == -5);
		SIGNAL_CHECK("value_changed", build_array(build_array(-5)));

		range->set_allow_greater(false);
		CHECK(range->is_greater_allowed() == false);

		range->set_max(-16.6);
		CHECK(range->get_max() == -16);
		CHECK(range->get_value() == -16);
		SIGNAL_CHECK("value_changed", build_array(build_array(-16)));
	}

	SUBCASE("[SceneTree][RangeInt] set_max with page") {
		range->set_page(7.2);
		CHECK(range->get_page() == 7);
		SIGNAL_CHECK("changed", build_array(build_array()));

		range->set_value(90);
		SIGNAL_DISCARD("value_changed");

		range->set_max(63.7);
		CHECK(range->get_max() == 63);
		CHECK(range->get_value() == 56);
		SIGNAL_CHECK("value_changed", build_array(build_array(56)));
		SIGNAL_CHECK("changed", build_array(build_array()));

		range->set_value(85.6);
		CHECK(range->get_value() == 56);

		range->set_min(-100.0);
		range->set_max(-10.8);
		CHECK(range->get_max() == -10);
		CHECK(range->get_value() == -17);

		range->set_value(-5.1);
		CHECK(range->get_value() == -17);

		range->set_max(-16.6);
		CHECK(range->get_max() == -16);
		CHECK(range->get_value() == -23);
	}

	SUBCASE("[SceneTree][RangeInt] set_max with step and page") {
		range->set_step(8.1);
		range->set_page(2.3);
		CHECK(range->get_step() == 8);
		CHECK(range->get_page() == 2);
		SIGNAL_CHECK("changed", build_array(build_array(), build_array()));

		range->set_value(90);
		SIGNAL_DISCARD("value_changed");

		range->set_max(63.7);
		CHECK(range->get_max() == 63);
		CHECK(range->get_value() == 61);
		SIGNAL_CHECK("changed", build_array(build_array()));

		range->set_value(85.6);
		CHECK(range->get_value() == 61);

		range->set_min(-100.0);
		range->set_max(-10.8);
		CHECK(range->get_max() == -10);
		CHECK(range->get_value() == -12);

		range->set_value(-5.1);
		CHECK(range->get_value() == -12);

		range->set_max(-16.6);
		CHECK(range->get_max() == -16);
		CHECK(range->get_value() == -18);

		// This results in a different value than set_max() above because
		// step is calulcated before page.
		range->set_value(-16.6);
		CHECK(range->get_value() == -20);
	}

	SUBCASE("[SceneTree][RangeInt] set_value large step") {
		range->set_max(10000000000000);
		range->set_step(72389512);
		CHECK(range->get_step() == 72389512);

		range->set_value(43);
		CHECK(range->get_value() == 0);

		range->set_value(58395727);
		CHECK(range->get_value() == 72389512);

		range->set_value(92957238);
		CHECK(range->get_value() == 72389512);

		range->set_value(1094218303);
		CHECK(range->get_value() == 1085842680);
	}

	SUBCASE("[SceneTree][RangeInt] ratio zero step") {
		range->set_step(0);

		CHECK(range->get_as_ratio() == doctest::Approx(0.0));

		range->set_value(50);
		CHECK(range->get_as_ratio() == doctest::Approx(0.5));

		range->set_value(100);
		CHECK(range->get_as_ratio() == doctest::Approx(1.0));

		range->set_as_ratio(0.0);
		CHECK(range->get_value() == 0);

		range->set_as_ratio(0.5);
		CHECK(range->get_value() == 50);

		range->set_as_ratio(1.0);
		CHECK(range->get_value() == 100);

		range->set_min(30);
		range->set_max(145);
		range->set_as_ratio(0.4);
		CHECK(range->get_value() == 76);

		range->set_value(92);
		CHECK(range->get_as_ratio() == doctest::Approx(0.53913043478));

		range->set_min(-234);
		range->set_max(-47);
		range->set_as_ratio(0.8);
		CHECK(range->get_value() == -84);

		range->set_value(-199);
		CHECK(range->get_as_ratio() == doctest::Approx(0.1871657754));

		range->set_min(-193);
		range->set_max(68);
		range->set_as_ratio(0.14);
		CHECK(range->get_value() == -156);

		range->set_value(14);
		CHECK(range->get_as_ratio() == doctest::Approx(0.79310344827));

		range->set_as_ratio(0.0);
		CHECK(range->get_value() == -193);

		range->set_as_ratio(1.0);
		CHECK(range->get_value() == 68);

		range->set_as_ratio(7.0);
		CHECK(range->get_value() == 68);

		range->set_as_ratio(-5.0);
		CHECK(range->get_value() == -193);

		range->set_allow_lesser(true);
		range->set_allow_greater(true);

		range->set_value(-5435);
		CHECK(range->get_as_ratio() == doctest::Approx(0.0));

		range->set_value(1926);
		CHECK(range->get_as_ratio() == doctest::Approx(1.0));
	}

	SUBCASE("[SceneTree][RangeInt] ratio with step") {
		range->set_step(3);

		CHECK(range->get_as_ratio() == doctest::Approx(0.0));

		range->set_value(50);
		CHECK(range->get_value() == 51.0);
		CHECK(range->get_as_ratio() == doctest::Approx(0.51));

		range->set_value(100);
		CHECK(range->get_value() == 99);
		CHECK(range->get_as_ratio() == doctest::Approx(0.99));

		range->set_as_ratio(0.0);
		CHECK(range->get_value() == 0);

		range->set_as_ratio(0.5);
		CHECK(range->get_value() == 51);

		range->set_as_ratio(1.0);
		CHECK(range->get_value() == 99);

		range->set_min(30);
		range->set_max(145);
		range->set_as_ratio(0.4);
		CHECK(range->get_value() == 75.0);

		range->set_value(92);
		CHECK(range->get_value() == 93);
		CHECK(range->get_as_ratio() == doctest::Approx(0.54782608696));

		range->set_min(-234);
		range->set_max(-47);
		range->set_as_ratio(0.8);
		CHECK(range->get_value() == -84);

		range->set_value(-199.0);
		CHECK(range->get_value() == -198.0);
		CHECK(range->get_as_ratio() == doctest::Approx(0.19251336898));

		range->set_min(-193);
		range->set_max(68);
		range->set_as_ratio(0.14);
		CHECK(range->get_value() == -157);

		range->set_value(14);
		CHECK(range->get_value() == 14);
		CHECK(range->get_as_ratio() == doctest::Approx(0.79310344828));

		range->set_as_ratio(0.0);
		CHECK(range->get_value() == doctest::Approx(-193));

		range->set_as_ratio(1.0);
		CHECK(range->get_value() == doctest::Approx(68));

		range->set_as_ratio(7.0);
		CHECK(range->get_value() == doctest::Approx(68));

		range->set_as_ratio(-5.0);
		CHECK(range->get_value() == doctest::Approx(-193));
	}

	SUBCASE("[SceneTree][RangeInt] ratio as exp") {
		range->set_step(0);
		range->set_exp_ratio(true);

		CHECK(range->get_as_ratio() == doctest::Approx(0.0));

		range->set_value(50);
		CHECK(range->get_as_ratio() == doctest::Approx(0.84948500217));

		range->set_value(100);
		CHECK(range->get_as_ratio() == doctest::Approx(1.0));

		range->set_as_ratio(0.0);
		CHECK(range->get_value() == 1);

		range->set_as_ratio(0.5);
		CHECK(range->get_value() == 10);

		range->set_as_ratio(1.0);
		CHECK(range->get_value() == 100);

		range->set_min(30);
		range->set_max(145);
		range->set_as_ratio(0.4);
		CHECK(range->get_value() == 56);

		range->set_value(92);
		CHECK(range->get_as_ratio() == doctest::Approx(0.71124426151));

		// TODO: Test negative values once behavior is well defined.
	}

	SUBCASE("[SceneTree][RangeInt] share") {
		range->set_value(38);
		range->set_min(10);
		range->set_max(75);
		range->set_step(4);
		range->set_page(3);
		range->set_exp_ratio(true);
		range->set_allow_greater(true);
		range->set_allow_lesser(true);

		RangeInt *alt1 = memnew(RangeInt);
		SceneTree::get_singleton()->get_root()->add_child(alt1);

		range->share(alt1);
		CHECK(alt1->get_value() == 38);
		CHECK(alt1->get_min() == 10);
		CHECK(alt1->get_max() == 75);
		CHECK(alt1->get_step() == 4);
		CHECK(alt1->get_page() == 3);
		CHECK(alt1->is_ratio_exp() == true);
		CHECK(alt1->is_greater_allowed() == true);
		CHECK(alt1->is_lesser_allowed() == true);

		range->set_step(5);
		CHECK(alt1->get_step() == 5);

		SIGNAL_DISCARD("changed");
		alt1->set_max(87.0);
		CHECK(range->get_max() == 87);
		SIGNAL_CHECK("changed", build_array(build_array()));

		SIGNAL_DISCARD("value_changed");
		alt1->set_value(65);
		SIGNAL_CHECK("value_changed", build_array(build_array(65)));

		RangeInt *alt2 = memnew(RangeInt);
		SceneTree::get_singleton()->get_root()->add_child(alt2);

		alt2->set_page(7);
		alt1->share(alt2);
		CHECK(alt2->get_page() == 3);

		alt2->set_min(9);
		CHECK(range->get_min() == 9);
		CHECK(alt1->get_min() == 9);

		alt2->set_step(0);
		SIGNAL_DISCARD("value_changed");
		alt2->set_value(47);
		CHECK(range->get_value() == 47);
		SIGNAL_CHECK("value_changed", build_array(build_array(47)));

		alt1->unshare();
		alt1->set_allow_lesser(false);
		CHECK(range->is_lesser_allowed() == true);
		CHECK(alt1->is_lesser_allowed() == false);
		CHECK(alt2->is_lesser_allowed() == true);

		SIGNAL_DISCARD("value_changed");
		alt1->set_value(17);
		CHECK(range->get_value() == 47);
		SIGNAL_CHECK_FALSE("value_changed");

		SceneTree::get_singleton()->get_root()->remove_child(alt1);
		SceneTree::get_singleton()->get_root()->remove_child(alt2);
		memdelete(alt1);
		memdelete(alt2);
	}

	SUBCASE("[SceneTree][RangeInt] large values") {
		range->set_min(INT64_MIN);
		CHECK(range->get_min() == INT64_MIN);
		CHECK(range->get_value() == 0);

		range->set_max(INT64_MAX);
		CHECK(range->get_max() == INT64_MAX);
		CHECK(range->get_value() == 0);

		SIGNAL_DISCARD("value_changed")
		for (int64_t i = 9223372036854775790; i < INT64_MAX; i++) {
			range->set_value(i + 1);
			CHECK(range->get_value() == i + 1);
			SIGNAL_CHECK("value_changed", build_array(build_array(i + 1)));
		}

		for (int64_t i = -9223372036854775790; i > INT64_MIN; i--) {
			range->set_value(i - 1);
			CHECK(range->get_value() == i - 1);
			SIGNAL_CHECK("value_changed", build_array(build_array(i - 1)));
		}

		range->set_page(INT64_MAX);
		range->set_value(INT64_MAX);
		CHECK(range->get_value() == 0);
	}

	SUBCASE("[SceneTree][RangeInt] large values ratio") {
		range->set_min(INT64_MIN);
		range->set_max(INT64_MAX);
		SIGNAL_DISCARD("value_changed")

		range->set_as_ratio(1.0);
		CHECK(range->get_value() == INT64_MAX);
		SIGNAL_CHECK("value_changed", build_array(build_array(INT64_MAX)));

		range->set_as_ratio(0.0);
		CHECK(range->get_value() == INT64_MIN);
		SIGNAL_CHECK("value_changed", build_array(build_array(INT64_MIN)));

		range->set_min(0);
		range->set_exp_ratio(true);
		SIGNAL_DISCARD("value_changed");

		range->set_as_ratio(1.0);
		CHECK(range->get_value() == INT64_MAX);
		SIGNAL_CHECK("value_changed", build_array(build_array(INT64_MAX)));

		range->set_as_ratio(0.0);
		CHECK(range->get_value() == 1);
		SIGNAL_CHECK("value_changed", build_array(build_array(1)));
	}

	SIGNAL_UNWATCH(range, "value_changed");
	SIGNAL_UNWATCH(range, "changed");

	SceneTree::get_singleton()->get_root()->remove_child(range);
	memdelete(range);
}

} // namespace TestRange

#endif // TEST_RANGE_H
