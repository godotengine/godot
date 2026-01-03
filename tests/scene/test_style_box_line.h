/**************************************************************************/
/*  test_style_box_line.h                                                 */
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

#include "scene/resources/style_box_line.h"

#include "tests/test_macros.h"

namespace TestStyleBoxLine {

TEST_CASE("[StyleBoxLine] Constructor") {
	Ref<StyleBoxLine> style_box_line = memnew(StyleBoxLine);

	CHECK(style_box_line->get_color() == Color(0, 0, 0, 1));
	CHECK(style_box_line->get_thickness() == 1);
	CHECK(style_box_line->is_vertical() == false);
	CHECK(style_box_line->get_grow_begin() == 1.0);
	CHECK(style_box_line->get_grow_end() == 1.0);
}

TEST_CASE("[StyleBoxLine] set_color, get_color") {
	Ref<StyleBoxLine> style_box_line = memnew(StyleBoxLine);
	Color color = Color(0.1, 0.2, 0.3, 1.0);

	style_box_line->set_color(color);
	CHECK(style_box_line->get_color() == color);
}

TEST_CASE("[StyleBoxLine] set_thickness, get_thickness") {
	Ref<StyleBoxLine> style_box_line = memnew(StyleBoxLine);

	style_box_line->set_thickness(5);
	CHECK(style_box_line->get_thickness() == 5);
}

TEST_CASE("[StyleBoxLine] set_vertical, is_vertical") {
	Ref<StyleBoxLine> style_box_line = memnew(StyleBoxLine);

	style_box_line->set_vertical(true);
	CHECK(style_box_line->is_vertical() == true);
}

TEST_CASE("[StyleBoxLine] set_vertical, is_vertical") {
	Ref<StyleBoxLine> style_box_line = memnew(StyleBoxLine);

	style_box_line->set_vertical(true);
	CHECK(style_box_line->is_vertical() == true);
}

TEST_CASE("[StyleBoxLine] set_grow_begin, get_grow_begin, set_grow_end, get_grow_end") {
	Ref<StyleBoxLine> style_box_line = memnew(StyleBoxLine);
	float grow_value = 3.5;

	SUBCASE("set_grow_begin, get_grow_begin") {
		style_box_line->set_grow_begin(grow_value);
		CHECK(style_box_line->get_grow_begin() == grow_value);
	}

	SUBCASE("set_grow_end, get_grow_end") {
		style_box_line->set_grow_end(grow_value);
		CHECK(style_box_line->get_grow_end() == grow_value);
	}
}

} // namespace TestStyleBoxLine
