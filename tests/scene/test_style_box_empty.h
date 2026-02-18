/**************************************************************************/
/*  test_style_box_empty.h                                                */
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

#pragma once

#include "scene/resources/style_box.h"

#include "tests/test_macros.h"

namespace TestStyleBoxEmpty {

TEST_CASE("[StyleBoxEmpty] Constructor") {
	Ref<StyleBoxEmpty> style_box_empty = memnew(StyleBoxEmpty);

	CHECK(style_box_empty->get_offset() == Point2(0, 0));
	CHECK(style_box_empty->get_current_item_drawn() == nullptr);
}

TEST_CASE("[StyleBoxEmpty] set_content_margin, set_content_margin_all, set_content_margin_individual, get_content_margin, get_margin") {
	Ref<StyleBoxEmpty> style_box_empty = memnew(StyleBoxEmpty);

	SUBCASE("set_content_margin, get_content_margin") {
		style_box_empty->set_content_margin(SIDE_LEFT, 2);
		style_box_empty->set_content_margin(SIDE_TOP, 3);
		style_box_empty->set_content_margin(SIDE_RIGHT, 4);
		style_box_empty->set_content_margin(SIDE_BOTTOM, 5);

		CHECK(style_box_empty->get_content_margin(SIDE_LEFT) == 2);
		CHECK(style_box_empty->get_content_margin(SIDE_TOP) == 3);
		CHECK(style_box_empty->get_content_margin(SIDE_RIGHT) == 4);
		CHECK(style_box_empty->get_content_margin(SIDE_BOTTOM) == 5);
	}

	SUBCASE("set_content_margin_all, get_content_margin") {
		style_box_empty->set_content_margin_all(10);

		CHECK(style_box_empty->get_content_margin(SIDE_LEFT) == 10);
		CHECK(style_box_empty->get_content_margin(SIDE_TOP) == 10);
		CHECK(style_box_empty->get_content_margin(SIDE_RIGHT) == 10);
		CHECK(style_box_empty->get_content_margin(SIDE_BOTTOM) == 10);
	}

	SUBCASE("set_content_margin, get_margin") {
		style_box_empty->set_content_margin(SIDE_LEFT, -1);
		style_box_empty->set_content_margin(SIDE_RIGHT, -2);
		style_box_empty->set_content_margin(SIDE_TOP, 3);
		style_box_empty->set_content_margin(SIDE_BOTTOM, 4);

		CHECK_MESSAGE(style_box_empty->get_margin(SIDE_LEFT) == 0,
				"Value is lesser than zero, so it returns 0");
		CHECK_MESSAGE(style_box_empty->get_margin(SIDE_RIGHT) == 0,
				"Value is lesser than zero, so it returns 0");
		CHECK_MESSAGE(style_box_empty->get_margin(SIDE_TOP) == 3,
				"Value is higher than zero, so it returns value");
		CHECK_MESSAGE(style_box_empty->get_margin(SIDE_BOTTOM) == 4,
				"Value is higher than zero, so it returns value");
	}

	SUBCASE("set_content_margin_individual, get_minimum_size") {
		style_box_empty->set_content_margin_individual(-1, 2, 5, 15);

		CHECK(style_box_empty->get_minimum_size() == Size2(5, 17));
	}

	SUBCASE("set_content_margin_individual, get_offset") {
		style_box_empty->set_content_margin_individual(-3, 5, 1, 2);

		CHECK_MESSAGE(style_box_empty->get_offset() == Point2(0, 5),
				"Returns Point2 with get_margin of SIDE_LEFT and SIDE_TOP");
	}
}

} // namespace TestStyleBoxEmpty
