/**************************************************************************/
/*  test_style_box_flat.cpp                                              */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_style_box_flat)

#include "scene/resources/style_box_flat.h"

namespace TestStyleBoxFlat {

TEST_CASE("[StyleBoxFlat] Constructor") {
	Ref<StyleBoxFlat> style_box_flat = memnew(StyleBoxFlat);

	CHECK(style_box_flat->get_bg_color() == Color(0.6, 0.6, 0.6));
	CHECK(style_box_flat->get_border_color() == Color(0.8, 0.8, 0.8));
	CHECK(style_box_flat->get_shadow_color() == Color(0, 0, 0, 0.6));
	CHECK(style_box_flat->get_shadow_size() == 0);
	CHECK(style_box_flat->get_shadow_offset() == Point2(0, 0));
	CHECK(style_box_flat->get_aa_size() == 1);
	CHECK(style_box_flat->get_corner_detail() == 8);
	CHECK(style_box_flat->get_corner_radius(CORNER_TOP_LEFT) == 0);
	CHECK(style_box_flat->get_corner_radius(CORNER_TOP_RIGHT) == 0);
	CHECK(style_box_flat->get_corner_radius(CORNER_BOTTOM_RIGHT) == 0);
	CHECK(style_box_flat->get_corner_radius(CORNER_BOTTOM_LEFT) == 0);
	CHECK(style_box_flat->get_expand_margin(SIDE_LEFT) == 0);
	CHECK(style_box_flat->get_expand_margin(SIDE_TOP) == 0);
	CHECK(style_box_flat->get_expand_margin(SIDE_RIGHT) == 0);
	CHECK(style_box_flat->get_expand_margin(SIDE_BOTTOM) == 0);
}

TEST_CASE("[StyleBoxFlat] set_bg_color, get_bg_color") {
	Ref<StyleBoxFlat> style_box_flat = memnew(StyleBoxFlat);

	style_box_flat->set_bg_color(Color(1, 1, 1));
	CHECK(style_box_flat->get_bg_color() == Color(1, 1, 1));
}

TEST_CASE("[StyleBoxFlat] set_border_color, get_border_color") {
	Ref<StyleBoxFlat> style_box_flat = memnew(StyleBoxFlat);

	style_box_flat->set_border_color(Color(1, 1, 1));
	CHECK(style_box_flat->get_border_color() == Color(1, 1, 1));
}

TEST_CASE("[StyleBoxFlat] set_border_width_all, get_border_width_min") {
	Ref<StyleBoxFlat> style_box_flat = memnew(StyleBoxFlat);

	style_box_flat->set_border_width_all(1);
	CHECK(style_box_flat->get_border_width_min() == 1);
}

TEST_CASE("[StyleBoxFlat] set_border_width, get_border_width") {
	Ref<StyleBoxFlat> style_box_flat = memnew(StyleBoxFlat);

	style_box_flat->set_border_width(SIDE_LEFT, 1);
	CHECK(style_box_flat->get_border_width(SIDE_LEFT) == 1);
	style_box_flat->set_border_width(SIDE_TOP, 1);
	style_box_flat->set_border_width(SIDE_RIGHT, 1);
	style_box_flat->set_border_width(SIDE_BOTTOM, 1);
	CHECK(style_box_flat->get_border_width(SIDE_LEFT) == 1);
	CHECK(style_box_flat->get_border_width(SIDE_TOP) == 1);
	CHECK(style_box_flat->get_border_width(SIDE_RIGHT) == 1);
	CHECK(style_box_flat->get_border_width(SIDE_BOTTOM) == 1);
}

TEST_CASE("[StyleBoxFlat] set_border_blend, get_border_blend") {
	Ref<StyleBoxFlat> style_box_flat = memnew(StyleBoxFlat);

	style_box_flat->set_border_blend(true);
	CHECK(style_box_flat->get_border_blend() == true);
}

TEST_CASE("[StyleBoxFlat] set_corner_radius_all, get_corner_radius") {
	Ref<StyleBoxFlat> style_box_flat = memnew(StyleBoxFlat);

	style_box_flat->set_corner_radius_all(1);
	CHECK(style_box_flat->get_corner_radius(CORNER_TOP_LEFT) == 1);
	CHECK(style_box_flat->get_corner_radius(CORNER_TOP_RIGHT) == 1);
	CHECK(style_box_flat->get_corner_radius(CORNER_BOTTOM_RIGHT) == 1);
	CHECK(style_box_flat->get_corner_radius(CORNER_BOTTOM_LEFT) == 1);
}

TEST_CASE("[StyleBoxFlat] set_corner_radius, get_corner_radius") {
	Ref<StyleBoxFlat> style_box_flat = memnew(StyleBoxFlat);

	style_box_flat->set_corner_radius(CORNER_TOP_LEFT, 1);
	style_box_flat->set_corner_radius(CORNER_TOP_RIGHT, 2);
	style_box_flat->set_corner_radius(CORNER_BOTTOM_RIGHT, 3);
	style_box_flat->set_corner_radius(CORNER_BOTTOM_LEFT, 4);
	CHECK(style_box_flat->get_corner_radius(CORNER_TOP_LEFT) == 1);
	CHECK(style_box_flat->get_corner_radius(CORNER_TOP_RIGHT) == 2);
	CHECK(style_box_flat->get_corner_radius(CORNER_BOTTOM_RIGHT) == 3);
	CHECK(style_box_flat->get_corner_radius(CORNER_BOTTOM_LEFT) == 4);
}

TEST_CASE("[StyleBoxFlat] set_corner_radius_individual, get_corner_radius") {
	Ref<StyleBoxFlat> style_box_flat = memnew(StyleBoxFlat);

	style_box_flat->set_corner_radius_individual(1, 2, 3, 4);
	CHECK(style_box_flat->get_corner_radius(CORNER_TOP_LEFT) == 1);
	CHECK(style_box_flat->get_corner_radius(CORNER_TOP_RIGHT) == 2);
	CHECK(style_box_flat->get_corner_radius(CORNER_BOTTOM_RIGHT) == 3);
	CHECK(style_box_flat->get_corner_radius(CORNER_BOTTOM_LEFT) == 4);
}

TEST_CASE("[StyleBoxFlat] set_corner_detail, get_corner_detail") {
	Ref<StyleBoxFlat> style_box_flat = memnew(StyleBoxFlat);

	style_box_flat->set_corner_detail(12);
	CHECK(style_box_flat->get_corner_detail() == 12);
}

TEST_CASE("[StyleBoxFlat] set_expand_margin, get_expand_margin") {
	Ref<StyleBoxFlat> style_box_flat = memnew(StyleBoxFlat);

	style_box_flat->set_expand_margin(SIDE_LEFT, 1);
	style_box_flat->set_expand_margin(SIDE_TOP, 2);
	style_box_flat->set_expand_margin(SIDE_RIGHT, 3);
	style_box_flat->set_expand_margin(SIDE_BOTTOM, 4);
	CHECK(style_box_flat->get_expand_margin(SIDE_LEFT) == 1);
	CHECK(style_box_flat->get_expand_margin(SIDE_TOP) == 2);
	CHECK(style_box_flat->get_expand_margin(SIDE_RIGHT) == 3);
	CHECK(style_box_flat->get_expand_margin(SIDE_BOTTOM) == 4);
}

TEST_CASE("[StyleBoxFlat] set_expand_margin_all, get_expand_margin") {
	Ref<StyleBoxFlat> style_box_flat = memnew(StyleBoxFlat);

	style_box_flat->set_expand_margin_all(5);
	CHECK(style_box_flat->get_expand_margin(SIDE_LEFT) == 5);
	CHECK(style_box_flat->get_expand_margin(SIDE_TOP) == 5);
	CHECK(style_box_flat->get_expand_margin(SIDE_RIGHT) == 5);
	CHECK(style_box_flat->get_expand_margin(SIDE_BOTTOM) == 5);
}

TEST_CASE("[StyleBoxFlat] set_expand_margin_individual, get_expand_margin") {
	Ref<StyleBoxFlat> style_box_flat = memnew(StyleBoxFlat);

	style_box_flat->set_expand_margin_individual(1, 2, 3, 4);
	CHECK(style_box_flat->get_expand_margin(SIDE_LEFT) == 1);
	CHECK(style_box_flat->get_expand_margin(SIDE_TOP) == 2);
	CHECK(style_box_flat->get_expand_margin(SIDE_RIGHT) == 3);
	CHECK(style_box_flat->get_expand_margin(SIDE_BOTTOM) == 4);
}

TEST_CASE("[StyleBoxFlat] set_shadow_color, get_shadow_color") {
	Ref<StyleBoxFlat> style_box_flat = memnew(StyleBoxFlat);

	style_box_flat->set_shadow_color(Color(1, 1, 1));
	CHECK(style_box_flat->get_shadow_color() == Color(1, 1, 1));
}

TEST_CASE("[StyleBoxFlat] set_shadow_size, get_shadow_size") {
	Ref<StyleBoxFlat> style_box_flat = memnew(StyleBoxFlat);

	style_box_flat->set_shadow_size(5);
	CHECK(style_box_flat->get_shadow_size() == 5);
}

TEST_CASE("[StyleBoxFlat] set_shadow_offset, get_shadow_offset") {
	Ref<StyleBoxFlat> style_box_flat = memnew(StyleBoxFlat);

	style_box_flat->set_shadow_offset(Point2(1, 2));
	CHECK(style_box_flat->get_shadow_offset() == Point2(1, 2));
}

TEST_CASE("[StyleBoxFlat] set_anti_aliased, is_anti_aliased") {
	Ref<StyleBoxFlat> style_box_flat = memnew(StyleBoxFlat);

	style_box_flat->set_anti_aliased(false);
	CHECK(style_box_flat->is_anti_aliased() == false);
}

TEST_CASE("[StyleBoxFlat] set_aa_size, get_aa_size") {
	Ref<StyleBoxFlat> style_box_flat = memnew(StyleBoxFlat);

	style_box_flat->set_aa_size(2);
	CHECK(style_box_flat->get_aa_size() == 2);
}

TEST_CASE("[StyleBoxFlat] set_skew, get_skew") {
	Ref<StyleBoxFlat> style_box_flat = memnew(StyleBoxFlat);

	style_box_flat->set_skew(Vector2(1, 2));
	CHECK(style_box_flat->get_skew() == Vector2(1, 2));
}

TEST_CASE("[StyleBoxFlat] set_draw_center, is_draw_center_enabled") {
	Ref<StyleBoxFlat> style_box_flat = memnew(StyleBoxFlat);

	style_box_flat->set_draw_center(false);
	CHECK(style_box_flat->is_draw_center_enabled() == false);
}
} // namespace TestStyleBoxFlat
