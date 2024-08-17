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

TEST_CASE("[SceneTree][Range] Range value constraints") {
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

	SUBCASE("Change value with default step size") {
		CHECK(range->get_step() == doctest::Approx(1.0));
		range->set_value(43.6);
		CHECK(range->get_value() == doctest::Approx(44.0));
		SIGNAL_CHECK("value_changed", build_array(build_array(44.0)));
		SIGNAL_CHECK_FALSE("changed");

		range->set_value(44.0);
		SIGNAL_CHECK_FALSE("value_changed");
	}

	SUBCASE("Change value with zero step size") {
		range->set_step(0);
		CHECK(range->get_step() == doctest::Approx(0.0));
		SIGNAL_CHECK("changed", build_array(build_array()));

		range->set_value(43.6);
		CHECK(range->get_step() == doctest::Approx(0.0));
		CHECK(range->get_value() == doctest::Approx(43.6));
		SIGNAL_CHECK("value_changed", build_array(build_array(43.6)));

		range->set_value(43.6);
		SIGNAL_CHECK_FALSE("value_changed");
	}

	SUBCASE("Set non-default step size") {
		range->set_step(7.8);
		CHECK(range->get_step() == doctest::Approx(7.8));
		SIGNAL_CHECK("changed", build_array(build_array()));
	}

	SUBCASE("Set minimum to same value") {
		range->set_min(6.9);
		SIGNAL_DISCARD("changed");

		range->set_min(6.9);
		SIGNAL_CHECK_FALSE("changed");
	}

	SUBCASE("Set minimum with zero step size") {
		range->set_step(0);

		SUBCASE("Increase minimum above current value") {
			range->set_min(5.1);
			CHECK(range->get_min() == doctest::Approx(5.1));
			CHECK(range->get_value() == doctest::Approx(5.1));
			SIGNAL_CHECK("value_changed", build_array(build_array(5.1)));
			SIGNAL_CHECK("changed", build_array(build_array()));
		}

		SUBCASE("Positive minimum") {
			range->set_min(5.1);

			SUBCASE("Set value above current minimum") {
				range->set_value(13.2);
				CHECK(range->get_value() == doctest::Approx(13.2));
				SIGNAL_CHECK("value_changed", build_array(build_array(13.2)));
			}

			SUBCASE("Set value below current minimum") {
				range->set_value(3.8);
				CHECK(range->get_value() == doctest::Approx(5.1));
				SIGNAL_CHECK_FALSE("value_changed");
			}
		}

		SUBCASE("Decrease minimum below current value") {
			range->set_min(-43.9);
			CHECK(range->get_min() == doctest::Approx(-43.9));
			CHECK(range->get_value() == doctest::Approx(0.0));
			SIGNAL_CHECK_FALSE("value_changed");
			SIGNAL_CHECK("changed", build_array(build_array()));
		}

		SUBCASE("Negative minimum") {
			range->set_min(-43.9);

			SUBCASE("Set positive value above negative minimum") {
				range->set_value(13.2);
				CHECK(range->get_value() == doctest::Approx(13.2));
				SIGNAL_CHECK("value_changed", build_array(build_array(13.2)));
			}

			SUBCASE("Set negative value above negative minimum") {
				range->set_value(-34.5);
				CHECK(range->get_value() == doctest::Approx(-34.5));
				SIGNAL_CHECK("value_changed", build_array(build_array(-34.5)));
			}

			SUBCASE("Set negative value below negative minimum") {
				range->set_value(-54.8);
				CHECK(range->get_value() == doctest::Approx(-43.9));
				SIGNAL_CHECK("value_changed", build_array(build_array(-43.9)));
			}

			SUBCASE("Set negative minimum above current value") {
				range->set_value(-42.9);
				SIGNAL_DISCARD("value_changed");

				range->set_min(-16.6);
				CHECK(range->get_min() == doctest::Approx(-16.6));
				CHECK(range->get_value() == doctest::Approx(-16.6));
				SIGNAL_CHECK("value_changed", build_array(build_array(-16.6)));
			}
		}
	}

	SUBCASE("Set minimum with non-default step size") {
		range->set_step(3.4);

		SUBCASE("Set positive minimum changes value") {
			range->set_min(5.1);
			CHECK(range->get_min() == doctest::Approx(5.1));
			CHECK(range->get_value() == doctest::Approx(5.1));
			SIGNAL_CHECK("value_changed", build_array(build_array(5.1)));
			SIGNAL_CHECK("changed", build_array(build_array()));
		}

		SUBCASE("Positive minimum") {
			range->set_min(5.1);

			SUBCASE("Set value above positive minimum") {
				range->set_value(13.2);
				CHECK(range->get_value() == doctest::Approx(11.9));
			}

			SUBCASE("Set value below positive minimum") {
				range->set_value(3.8);
				CHECK(range->get_value() == doctest::Approx(5.1));
			}
		}

		SUBCASE("Negative minimum") {
			range->set_min(-43.9);

			SUBCASE("Set positive value above negative minimum") {
				range->set_value(13.2);
				CHECK(range->get_value() == doctest::Approx(13.9));
			}

			SUBCASE("Set negative value above negative minimum") {
				range->set_value(-34.5);
				CHECK(range->get_value() == doctest::Approx(-33.7));
			}

			SUBCASE("Set negative value below negative minimum") {
				range->set_value(-54.8);
				CHECK(range->get_value() == doctest::Approx(-43.9));
			}

			SUBCASE("Set minimum higher than current value") {
				range->set_value(-43.9);
				range->set_min(-16.6);
				CHECK(range->get_min() == doctest::Approx(-16.6));
				CHECK(range->get_value() == doctest::Approx(-16.6));
			}
		}
	}

	SUBCASE("Set minimum while allowing lesser") {
		range->set_step(0);
		range->set_allow_lesser(true);
		CHECK(range->is_lesser_allowed() == true);

		SUBCASE("Set minimum above current value") {
			range->set_min(5.1);
			CHECK(range->get_min() == doctest::Approx(5.1));
			CHECK(range->get_value() == doctest::Approx(0.0));
			SIGNAL_CHECK_FALSE("value_changed");
			SIGNAL_CHECK("changed", build_array(build_array()));
		}

		SUBCASE("Set positive value below positive minimum") {
			range->set_min(5.1);
			range->set_value(3.8);
			CHECK(range->get_value() == doctest::Approx(3.8));
			SIGNAL_CHECK("value_changed", build_array(build_array(3.8)));
		}

		SUBCASE("Set negative value below negative minimum") {
			range->set_min(-43.9);
			range->set_value(-54.8);
			CHECK(range->get_value() == doctest::Approx(-54.8));
			SIGNAL_CHECK("value_changed", build_array(build_array(-54.8)));
		}

		SUBCASE("Forbid lesser while value below minimum") {
			range->set_min(-13);
			range->set_value(-76.5);
			SIGNAL_DISCARD("value_changed")

			range->set_allow_lesser(false);
			CHECK(range->is_lesser_allowed() == false);

			range->set_min(-16.6);
			CHECK(range->get_min() == doctest::Approx(-16.6));
			CHECK(range->get_value() == doctest::Approx(-16.6));
			SIGNAL_CHECK("value_changed", build_array(build_array(-16.6)));
		}
	}

	SUBCASE("Set maximum with zero step size") {
		range->set_step(0);
		range->set_value(90);

		SUBCASE("Increase maximum above current value") {
			range->set_max(435.6);
			CHECK(range->get_max() == doctest::Approx(435.6));
			CHECK(range->get_value() == doctest::Approx(90.0));
			SIGNAL_CHECK("changed", build_array(build_array()));
		}

		SUBCASE("Decrease maximum below current value") {
			range->set_max(63.7);
			CHECK(range->get_max() == doctest::Approx(63.7));
			CHECK(range->get_value() == doctest::Approx(63.7));
			SIGNAL_CHECK("value_changed", build_array(build_array(63.7)));
			SIGNAL_CHECK("changed", build_array(build_array()));
		}

		SUBCASE("Positive maximum") {
			range->set_max(63.7);

			SUBCASE("Set value below current maximum") {
				range->set_value(25.2);
				CHECK(range->get_value() == doctest::Approx(25.2));
				SIGNAL_CHECK("value_changed", build_array(build_array(25.2)));
			}

			range->set_value(0.0);

			SUBCASE("Set value above current maximum") {
				range->set_value(85.6);
				CHECK(range->get_value() == doctest::Approx(63.7));
				SIGNAL_CHECK("value_changed", build_array(build_array(63.7)));
			}
		}

		SUBCASE("Decrease minimum below current value") {
			range->set_min(-100.0);
			range->set_max(-10.8);
			CHECK(range->get_max() == doctest::Approx(-10.8));
			CHECK(range->get_value() == doctest::Approx(-10.8));
			SIGNAL_CHECK("value_changed", build_array(build_array(-10.8)));
		}

		SUBCASE("Negative maximum") {
			range->set_min(-100.0);
			range->set_max(-10.8);

			SUBCASE("Set value below negative maximum") {
				range->set_value(-38.4);
				CHECK(range->get_value() == doctest::Approx(-38.4));
				SIGNAL_CHECK("value_changed", build_array(build_array(-38.4)));
			}

			SUBCASE("Set positive value above negative maximum") {
				range->set_value(-38.4);
				SIGNAL_DISCARD("value_changed")

				range->set_value(13.2);
				CHECK(range->get_value() == doctest::Approx(-10.8));
				SIGNAL_CHECK("value_changed", build_array(build_array(-10.8)));
			}

			SUBCASE("Set negative value below negative maximum") {
				range->set_value(-5.1);
				CHECK(range->get_value() == doctest::Approx(-10.8));
				SIGNAL_CHECK_FALSE("value_changed");
			}

			SUBCASE("Set negative maximum below current value") {
				range->set_value(-18.2);
				SIGNAL_DISCARD("value_changed");

				range->set_max(-36.6);
				CHECK(range->get_max() == doctest::Approx(-36.6));
				CHECK(range->get_value() == doctest::Approx(-36.6));
				SIGNAL_CHECK("value_changed", build_array(build_array(-36.6)));
			}
		}
	}

	SUBCASE("Set maximum with non-default step size") {
		range->set_step(3.4);
		range->set_value(90);

		SUBCASE("Set positive maximum changes value") {
			range->set_max(63.7);
			CHECK(range->get_max() == doctest::Approx(63.7));
			CHECK(range->get_value() == doctest::Approx(63.7));
			SIGNAL_CHECK("value_changed", build_array(build_array(63.7)));
			SIGNAL_CHECK("changed", build_array(build_array()));
		}

		SUBCASE("Positive maximum") {
			range->set_max(63.7);

			SUBCASE("Set value below positive maximum") {
				range->set_value(25.2);
				CHECK(range->get_value() == doctest::Approx(23.8));
			}

			SUBCASE("Set value above positive maximum") {
				range->set_value(85.6);
				CHECK(range->get_value() == doctest::Approx(63.7));
			}
		}

		SUBCASE("Set negative maximum changes value") {
			range->set_min(-100.0);
			range->set_max(-10.8);
			CHECK(range->get_max() == doctest::Approx(-10.8));
			CHECK(range->get_value() == doctest::Approx(-10.8));
		}

		SUBCASE("Negative maximum") {
			range->set_min(-100.0);
			range->set_max(-10.8);

			SUBCASE("Set negative value below negative maximum") {
				range->set_value(-38.4);
				CHECK(range->get_value() == doctest::Approx(-38.8));
			}

			SUBCASE("Set positive value above negative maximum") {
				range->set_value(13.2);
				CHECK(range->get_value() == doctest::Approx(-10.8));
			}

			SUBCASE("Set negative value above negative maximum") {
				range->set_value(-5.1);
				CHECK(range->get_value() == doctest::Approx(-10.8));
			}

			SUBCASE("Set maximum lower than current value") {
				range->set_max(-16.6);
				CHECK(range->get_max() == doctest::Approx(-16.6));
				CHECK(range->get_value() == doctest::Approx(-16.6));
			}
		}
	}

	SUBCASE("Set maximum while allowing greater") {
		range->set_step(0);
		range->set_value(90);
		range->set_allow_greater(true);
		CHECK(range->is_greater_allowed() == true);

		SUBCASE("Set maximum below current value") {
			range->set_max(63.7);
			CHECK(range->get_max() == doctest::Approx(63.7));
			CHECK(range->get_value() == doctest::Approx(90.0));
			SIGNAL_CHECK_FALSE("value_changed");
			SIGNAL_CHECK("changed", build_array(build_array()));
		}

		SUBCASE("Set positive value above positive maximum") {
			range->set_value(175.9);
			CHECK(range->get_value() == doctest::Approx(175.9));
			SIGNAL_CHECK("value_changed", build_array(build_array(175.9)));
			SIGNAL_CHECK_FALSE("changed");
		}

		range->set_min(-100.0);
		range->set_max(-10.8);

		SUBCASE("Set positive value above negative maximum") {
			range->set_value(13.2);
			CHECK(range->get_value() == doctest::Approx(13.2));
			SIGNAL_CHECK("value_changed", build_array(build_array(13.2)));
		}

		SUBCASE("Set negative value above negative maximum") {
			range->set_value(-5.1);
			CHECK(range->get_value() == doctest::Approx(-5.1));
			SIGNAL_CHECK("value_changed", build_array(build_array(-5.1)));
		}

		SUBCASE("Forbid greater while value above maximum") {
			range->set_value(45.3);
			SIGNAL_DISCARD("value_changed")

			range->set_allow_greater(false);
			CHECK(range->is_greater_allowed() == false);

			range->set_max(-16.6);
			CHECK(range->get_max() == doctest::Approx(-16.6));
			CHECK(range->get_value() == doctest::Approx(-16.6));
			SIGNAL_CHECK("value_changed", build_array(build_array(-16.6)));
		}
	}

	SUBCASE("Set maximum with default step and non-zero page") {
		range->set_page(7.2);
		CHECK(range->get_page() == doctest::Approx(7.2));
		SIGNAL_CHECK("changed", build_array(build_array()));
		range->set_value(90);
		range->set_max(63.7);

		SUBCASE("Set positive maximum below current value") {
			CHECK(range->get_max() == doctest::Approx(63.7));
			CHECK(range->get_value() == doctest::Approx(56.5));
		}

		SUBCASE("Set positive value above maximum") {
			range->set_value(85.6);
			CHECK(range->get_value() == doctest::Approx(56.5));
		}

		range->set_min(-100.0);
		range->set_max(-10.8);

		SUBCASE("Set negative maximum below current value") {
			CHECK(range->get_max() == doctest::Approx(-10.8));
			CHECK(range->get_value() == doctest::Approx(-18.0));
		}

		SUBCASE("Set negative value above maximum") {
			range->set_value(-5.1);
			CHECK(range->get_value() == doctest::Approx(-18.0));
		}

		SUBCASE("Set negative maximum within page distance of current value") {
			range->set_value(-18.0);
			range->set_max(-16.6);
			CHECK(range->get_max() == doctest::Approx(-16.6));
			CHECK(range->get_value() == doctest::Approx(-23.8));
		}
	}

	SUBCASE("Set maximum with non-default step and non-zero page") {
		range->set_step(8.1);
		range->set_page(2.3);
		CHECK(range->get_step() == doctest::Approx(8.1));
		CHECK(range->get_page() == doctest::Approx(2.3));
		SIGNAL_CHECK("changed", build_array(build_array(), build_array()));
		range->set_value(90);
		range->set_max(63.7);

		SUBCASE("Set positive maximum below current value") {
			CHECK(range->get_max() == doctest::Approx(63.7));
			CHECK(range->get_value() == doctest::Approx(61.4));
		}

		SUBCASE("Set positive value above maximum") {
			range->set_value(85.6);
			CHECK(range->get_value() == doctest::Approx(61.4));
		}

		range->set_min(-100.0);
		range->set_max(-10.8);

		SUBCASE("Set negative maximum below current value") {
			CHECK(range->get_max() == doctest::Approx(-10.8));
			CHECK(range->get_value() == doctest::Approx(-13.1));
		}

		SUBCASE("Set negative value above maximum") {
			range->set_value(-5.1);
			CHECK(range->get_value() == doctest::Approx(-13.1));
		}

		SUBCASE("Set negative maximum within page distance of current value") {
			range->set_value(-13.1);
			range->set_max(-16.6);
			CHECK(range->get_max() == doctest::Approx(-16.6));
			CHECK(range->get_value() == doctest::Approx(-18.9));
		}

		SUBCASE("Reset value to trigger page calculation") {
			// This results in a different value than set_max() above because
			// step is calulcated before page.
			range->set_value(-16.6);
			CHECK(range->get_value() == doctest::Approx(-19.0));
		}
	}

	SUBCASE("Set value with large step") {
		range->set_max(10000000000000.0);
		range->set_step(72389512.0);
		CHECK(range->get_step() == doctest::Approx(72389512.0));

		SUBCASE("Round down to first step") {
			range->set_value(43.6);
			CHECK(range->get_value() == doctest::Approx(0));
		}

		SUBCASE("Round up to second step") {
			range->set_value(58395727.0);
			CHECK(range->get_value() == doctest::Approx(72389512.0));
		}

		SUBCASE("Round down to second step") {
			range->set_value(92957238.0);
			CHECK(range->get_value() == doctest::Approx(72389512.0));
		}

		SUBCASE("Round down to much higher step") {
			range->set_value(1094218303.0);
			CHECK(range->get_value() == doctest::Approx(1085842680.0));
		}
	}

	SUBCASE("Ratio with zero step") {
		range->set_step(0);

		SUBCASE("Set value default range") {
			CHECK(range->get_as_ratio() == doctest::Approx(0.0));

			range->set_value(50.0);
			CHECK(range->get_as_ratio() == doctest::Approx(0.5));

			range->set_value(100.0);
			CHECK(range->get_as_ratio() == doctest::Approx(1.0));
		}

		SUBCASE("Set ratio default range") {
			range->set_as_ratio(0.0);
			CHECK(range->get_value() == doctest::Approx(0.0));

			range->set_as_ratio(0.5);
			CHECK(range->get_value() == doctest::Approx(50.0));

			range->set_as_ratio(1.0);
			CHECK(range->get_value() == doctest::Approx(100.0));
		}

		SUBCASE("Positive range") {
			range->set_min(30.0);
			range->set_max(145.0);

			range->set_as_ratio(0.4);
			CHECK(range->get_value() == doctest::Approx(76.0));

			range->set_value(92.0);
			CHECK(range->get_as_ratio() == doctest::Approx(0.53913043478));
		}

		SUBCASE("Negative range") {
			range->set_min(-234.0);
			range->set_max(-47.0);

			range->set_as_ratio(0.8);
			CHECK(range->get_value() == doctest::Approx(-84.4));

			range->set_value(-199.0);
			CHECK(range->get_as_ratio() == doctest::Approx(0.1871657754));
		}

		range->set_min(-193.67);
		range->set_max(68.29);

		SUBCASE("Range spans negative to positive") {
			range->set_as_ratio(0.14);
			CHECK(range->get_value() == doctest::Approx(-156.9956));

			range->set_value(14.5);
			CHECK(range->get_as_ratio() == doctest::Approx(0.79466330737));

			range->set_as_ratio(0.0);
			CHECK(range->get_value() == doctest::Approx(-193.67));

			range->set_as_ratio(1.0);
			CHECK(range->get_value() == doctest::Approx(68.29));
		}

		SUBCASE("Set ratio out of bounds") {
			range->set_as_ratio(7.0);
			CHECK(range->get_value() == doctest::Approx(68.29));

			range->set_as_ratio(-5.0);
			CHECK(range->get_value() == doctest::Approx(-193.67));
		}

		SUBCASE("Set value beyond minimum and maximum") {
			range->set_allow_lesser(true);
			range->set_allow_greater(true);

			range->set_value(-5435.0);
			CHECK(range->get_as_ratio() == doctest::Approx(0.0));

			range->set_value(1926.0);
			CHECK(range->get_as_ratio() == doctest::Approx(1.0));
		}
	}

	SUBCASE("ratio with step") {
		range->set_step(3);

		SUBCASE("Set value default range") {
			CHECK(range->get_as_ratio() == doctest::Approx(0.0));

			range->set_value(50.0);
			CHECK(range->get_value() == doctest::Approx(51.0));
			CHECK(range->get_as_ratio() == doctest::Approx(0.51));

			range->set_value(100.0);
			CHECK(range->get_value() == doctest::Approx(99.0));
			CHECK(range->get_as_ratio() == doctest::Approx(0.99));
		}

		SUBCASE("Set ratio default range") {
			range->set_as_ratio(0.0);
			CHECK(range->get_value() == doctest::Approx(0.0));

			range->set_as_ratio(0.5);
			CHECK(range->get_value() == doctest::Approx(51.0));

			range->set_as_ratio(1.0);
			CHECK(range->get_value() == doctest::Approx(99.0));
		}

		SUBCASE("Positive range") {
			range->set_min(30.0);
			range->set_max(145.0);
			range->set_as_ratio(0.4);
			CHECK(range->get_value() == doctest::Approx(75.0));

			range->set_value(92.0);
			CHECK(range->get_value() == doctest::Approx(93.0));
			CHECK(range->get_as_ratio() == doctest::Approx(0.54782608696));
		}

		SUBCASE("Negative range") {
			range->set_min(-234.0);
			range->set_max(-47.0);
			range->set_as_ratio(0.8);
			CHECK(range->get_value() == doctest::Approx(-84));

			range->set_value(-199.0);
			CHECK(range->get_value() == doctest::Approx(-198.0));
			CHECK(range->get_as_ratio() == doctest::Approx(0.19251336898));
		}

		range->set_min(-193.67);
		range->set_max(68.29);

		SUBCASE("Range spans negative to positive") {
			range->set_as_ratio(0.14);
			CHECK(range->get_value() == doctest::Approx(-157.67));

			range->set_value(14.5);
			CHECK(range->get_value() == doctest::Approx(13.33));
			CHECK(range->get_as_ratio() == doctest::Approx(0.79019697664));

			range->set_as_ratio(0.0);
			CHECK(range->get_value() == doctest::Approx(-193.67));

			range->set_as_ratio(1.0);
			CHECK(range->get_value() == doctest::Approx(67.33));
		}

		SUBCASE("Set ratio out of bounds") {
			range->set_as_ratio(7.0);
			CHECK(range->get_value() == doctest::Approx(67.33));

			range->set_as_ratio(-5.0);
			CHECK(range->get_value() == doctest::Approx(-193.67));
		}
	}

	SUBCASE("Exponential ratio") {
		range->set_step(0);
		range->set_exp_ratio(true);

		SUBCASE("Set value default range") {
			CHECK(range->get_as_ratio() == doctest::Approx(0.0));

			range->set_value(50.0);
			CHECK(range->get_as_ratio() == doctest::Approx(0.84948500217));

			range->set_value(100.0);
			CHECK(range->get_as_ratio() == doctest::Approx(1.0));
		}

		SUBCASE("Set ratio default range") {
			range->set_as_ratio(0.0);
			CHECK(range->get_value() == doctest::Approx(1.0));

			range->set_as_ratio(0.5);
			CHECK(range->get_value() == doctest::Approx(10.0));

			range->set_as_ratio(1.0);
			CHECK(range->get_value() == doctest::Approx(100.0));
		}

		SUBCASE("Non-default range") {
			range->set_min(30.0);
			range->set_max(145.0);
			range->set_as_ratio(0.4);
			CHECK(range->get_value() == doctest::Approx(56.340403594));

			range->set_value(92.0);
			CHECK(range->get_as_ratio() == doctest::Approx(0.71124426151));
		}

		// TODO: Test negative values once behavior is well defined.
	}

	SUBCASE("Rounded values") {
		range->set_step(3.4);
		range->set_use_rounded_values(true);

		SUBCASE("Round up") {
			range->set_value(47.6);
			CHECK(range->get_value() == doctest::Approx(48.0));
			SIGNAL_CHECK("value_changed", build_array(build_array(48.0)));
		}

		SUBCASE("Round down") {
			range->set_value(37.4);
			CHECK(range->get_value() == doctest::Approx(37.0));
			SIGNAL_CHECK("value_changed", build_array(build_array(37.0)));
		}

		SUBCASE("Round after aligning to step") {
			// We round to closest step (20.4) before closest integer.
			range->set_value(20.7);
			CHECK(range->get_value() == doctest::Approx(20.0));
			SIGNAL_CHECK("value_changed", build_array(build_array(20.0)));
		}

		SUBCASE("Disable rounding") {
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
	}

	SIGNAL_UNWATCH(range, "value_changed");
	SIGNAL_UNWATCH(range, "changed");

	SceneTree::get_singleton()->get_root()->remove_child(range);
	memdelete(range);
}

TEST_CASE("[SceneTree][Range] Sharing data") {
	Range *range1 = memnew(Range);
	Range *range2 = memnew(Range);
	Range *range3 = memnew(Range);
	SceneTree::get_singleton()->get_root()->add_child(range1);
	SceneTree::get_singleton()->get_root()->add_child(range2);
	SceneTree::get_singleton()->get_root()->add_child(range3);
	SIGNAL_WATCH(range1, "value_changed");
	SIGNAL_WATCH(range1, "changed");

	range1->set_value(38.0);
	range1->set_min(10.0);
	range1->set_max(75.0);
	range1->set_step(4.0);
	range1->set_page(3.0);
	range1->set_exp_ratio(true);
	range1->set_allow_greater(true);
	range1->set_allow_lesser(true);

	range1->share(range2);

	SUBCASE("Share existing data") {
		CHECK(range2->get_value() == doctest::Approx(38.0));
		CHECK(range2->get_min() == doctest::Approx(10.0));
		CHECK(range2->get_max() == doctest::Approx(75.0));
		CHECK(range2->get_step() == doctest::Approx(4.0));
		CHECK(range2->get_page() == doctest::Approx(3.0));
		CHECK(range2->is_ratio_exp() == true);
		CHECK(range2->is_greater_allowed() == true);
		CHECK(range2->is_lesser_allowed() == true);
	}

	SUBCASE("Change step in original after sharing") {
		range1->set_step(5.4);
		CHECK(range2->get_step() == doctest::Approx(5.4));
	}

	SUBCASE("Change max in second after sharing") {
		range2->set_max(87.0);
		CHECK(range1->get_max() == doctest::Approx(87.0));
		SIGNAL_CHECK("changed", build_array(build_array()));
	}

	SUBCASE("Set value in second triggers signal in original") {
		range2->set_value(66.0);
		SIGNAL_CHECK("value_changed", build_array(build_array(66.0)));
	}

	range3->set_page(7.5);
	range2->share(range3);

	SUBCASE("Sharing overwrites existing data") {
		CHECK(range3->get_page() == doctest::Approx(3.0));
	}

	SUBCASE("Changing third affects original and second") {
		range3->set_min(9.0);
		CHECK(range1->get_min() == doctest::Approx(9.0));
		CHECK(range2->get_min() == doctest::Approx(9.0));
	}

	SUBCASE("Changing step in third is respected in original") {
		range3->set_step(0.0);
		range3->set_value(47.5);
		CHECK(range1->get_value() == doctest::Approx(47.5));
		SIGNAL_CHECK("value_changed", build_array(build_array(47.5)));
		SIGNAL_CHECK("changed", build_array(build_array()));
	}

	range2->unshare();

	SUBCASE("Change allow lesser in second after unsharing") {
		range2->set_allow_lesser(false);
		CHECK(range1->is_lesser_allowed() == true);
		CHECK(range2->is_lesser_allowed() == false);
		CHECK(range3->is_lesser_allowed() == true);
	}

	SUBCASE("Change value in second after unsharing") {
		SIGNAL_DISCARD("value_changed");
		range2->set_value(22.0);
		CHECK(range1->get_value() == doctest::Approx(38.0));
		SIGNAL_CHECK_FALSE("value_changed");
	}

	SceneTree::get_singleton()->get_root()->remove_child(range1);
	SceneTree::get_singleton()->get_root()->remove_child(range2);
	SceneTree::get_singleton()->get_root()->remove_child(range3);
	memdelete(range1);
	memdelete(range2);
	memdelete(range3);
}

} // namespace TestRange

#endif // TEST_RANGE_H
