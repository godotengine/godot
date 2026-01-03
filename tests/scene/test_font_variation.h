/**************************************************************************/
/*  test_font_variation.h                                                 */
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

#include "modules/modules_enabled.gen.h"

#include "scene/resources/font.h"
#include "tests/test_macros.h"

namespace TestFontVariation {

Ref<FontVariation> create_font_variation() {
	String test_variable_font = "tests/data/font/Inter-ASCII.ttf";
	Ref<FontFile> ff;
	ff.instantiate();
	REQUIRE_EQ(ff->load_dynamic_font(test_variable_font), OK);

	Ref<FontVariation> fvar;
	fvar.instantiate();
	fvar->set_base_font(ff);
	REQUIRE_EQ(fvar->get_base_font(), ff);

	return fvar;
}

Transform2D create_test_transform() {
	Transform2D transform;
	transform.columns[0].x = 1.0;
	transform.columns[0].y = 5.2;
	transform.columns[1].x = -1.0;
	transform.columns[1].y = 3.5;
	transform.columns[2].x = 1.3;
	transform.columns[2].y = -4.5;

	return transform;
}

TEST_CASE("[FontVariation] simple set then get") {
#ifdef MODULE_FREETYPE_ENABLED
	Ref<FontVariation> fvar = create_font_variation();

	SUBCASE("OpenType") {
		Dictionary coords;
		coords["axis_tag"] = "wght";
		coords["min"] = 0.0;
		coords["default"] = 1.0;
		coords["max"] = 2.0;

		fvar->set_variation_opentype(coords);
		REQUIRE_EQ(fvar->get_variation_opentype(), coords);
	}

	SUBCASE("Embolden") {
		float strength = 42.1;
		fvar->set_variation_embolden(strength);
		REQUIRE_EQ(fvar->get_variation_embolden(), strength);
	}

	SUBCASE("Transform") {
		Transform2D transform = create_test_transform();
		fvar->set_variation_transform(transform);
		REQUIRE_EQ(fvar->get_variation_transform(), transform);
	}

	SUBCASE("Face index") {
		int face_index = 6;
		fvar->set_variation_face_index(face_index);
		REQUIRE_EQ(fvar->get_variation_face_index(), face_index);
	}

	SUBCASE("OpenType features") {
		Dictionary features;
		features["feature_tag"] = "liga";
		features["enabled"] = true;

		fvar->set_opentype_features(features);
		REQUIRE_EQ(fvar->get_opentype_features(), features);
	}

	SUBCASE("Spacing") {
		TextServer::SpacingType spacing = TextServer::SpacingType::SPACING_BOTTOM;
		int value = 2;

		fvar->set_spacing(spacing, value);
		REQUIRE_EQ(fvar->get_spacing(spacing), value);
	}

	SUBCASE("Baseline offset") {
		float baseline_offset = 5.3;
		fvar->set_baseline_offset(baseline_offset);
		REQUIRE_EQ(fvar->get_baseline_offset(), baseline_offset);
	}
#endif
}

TEST_CASE("[FontVariation] complex operations against known good values") {
#ifdef MODULE_FREETYPE_ENABLED
	Ref<FontVariation> fvar = create_font_variation();

	fvar->set_variation_embolden(50.0);
	fvar->set_spacing(TextServer::SpacingType::SPACING_TOP, 3);
	fvar->set_variation_transform(create_test_transform());

	SUBCASE("The size of a single character") {
		Size2 char_size = fvar->get_char_size('Y', 30);

		CHECK_EQ(char_size.x, doctest::Approx((real_t)44));
		CHECK_EQ(char_size.y, doctest::Approx((real_t)41));
	}

	SUBCASE("The size of a string") {
		Size2 string_size = fvar->get_string_size("hello, GODOT!", HORIZONTAL_ALIGNMENT_RIGHT, -1, 24);

		CHECK_EQ(string_size.x, doctest::Approx((real_t)406));
		CHECK_EQ(string_size.y, doctest::Approx((real_t)33));
	}

	SUBCASE("The size of a multiline string") {
		Size2 multiline_size = fvar->get_multiline_string_size("spreading over\n\nmany lines\n\nETERNALLY", HORIZONTAL_ALIGNMENT_FILL, -1, 18);

		CHECK_EQ(multiline_size.x, doctest::Approx((real_t)307));
		CHECK_EQ(multiline_size.y, doctest::Approx((real_t)130));
	}
#endif
}

} // namespace TestFontVariation
