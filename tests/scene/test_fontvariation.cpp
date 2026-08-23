/**************************************************************************/
/*  test_fontvariation.cpp                                                */
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

TEST_FORCE_LINK(test_fontvariation)

#include "scene/resources/font.h"

namespace TestFontVariation {

TEST_CASE("[FontVariation] Default values") {
	Ref<FontVariation> font_variation;
	font_variation.instantiate();

	CHECK_MESSAGE(font_variation->get_base_font().is_null(), "Base font should be null by default.");
	CHECK_MESSAGE(font_variation->get_variation_opentype() == Dictionary(), "OpenType variation coordinates should be empty by default.");
	CHECK_MESSAGE(font_variation->get_variation_embolden() == 0.0f, "Embolden strength should be zero by default.");
	CHECK_MESSAGE(font_variation->get_variation_transform() == Transform2D(), "Variation transform should be identity by default.");
	CHECK_MESSAGE(font_variation->get_variation_face_index() == 0, "Face index should be zero by default.");
	CHECK_MESSAGE(font_variation->get_opentype_features() == Dictionary(), "OpenType features should be empty by default.");
	CHECK_MESSAGE(font_variation->get_baseline_offset() == 0.0f, "Baseline offset should be zero by default.");
	CHECK_MESSAGE(font_variation->get_palette_index() == 0, "Palette index should be zero by default.");
	CHECK_MESSAGE(font_variation->get_palette_custom_colors().is_empty(), "Custom palette colors should be empty by default.");

	for (int i = 0; i < TextServer::SPACING_MAX; i++) {
		CHECK_MESSAGE(font_variation->get_spacing((TextServer::SpacingType)i) == 0, "Extra spacing should be zero by default.");
	}
}

TEST_CASE("[FontVariation] Setters and getters") {
	Ref<FontVariation> font_variation;
	font_variation.instantiate();

	font_variation->set_variation_embolden(0.75f);
	CHECK_MESSAGE(font_variation->get_variation_embolden() == doctest::Approx((real_t)0.75), "Embolden strength should match the value that was set.");
	font_variation->set_variation_embolden(-1.25f);
	CHECK_MESSAGE(font_variation->get_variation_embolden() == doctest::Approx((real_t)-1.25), "Negative embolden strength should match the value that was set.");

	Transform2D transform = Transform2D(1.0, 0.5, -0.25, 2.0, 10.0, -20.0);
	font_variation->set_variation_transform(transform);
	CHECK_MESSAGE(font_variation->get_variation_transform() == transform, "Variation transform should match the value that was set.");

	font_variation->set_variation_face_index(2);
	CHECK_MESSAGE(font_variation->get_variation_face_index() == 2, "Face index should match the value that was set.");
	font_variation->set_variation_face_index(-1);
	CHECK_MESSAGE(font_variation->get_variation_face_index() == -1, "Negative face index should match the value that was set.");

	font_variation->set_baseline_offset(12.5f);
	CHECK_MESSAGE(font_variation->get_baseline_offset() == doctest::Approx((real_t)12.5), "Baseline offset should match the value that was set.");
	font_variation->set_baseline_offset(-7.25f);
	CHECK_MESSAGE(font_variation->get_baseline_offset() == doctest::Approx((real_t)-7.25), "Negative baseline offset should match the value that was set.");

	font_variation->set_palette_index(3);
	CHECK_MESSAGE(font_variation->get_palette_index() == 3, "Palette index should match the value that was set.");

	Vector<Color> custom_colors;
	custom_colors.push_back(Color(0.1f, 0.2f, 0.3f, 0.4f));
	custom_colors.push_back(Color(0.5f, 0.6f, 0.7f, 0.8f));
	font_variation->set_palette_custom_colors(custom_colors);
	CHECK_MESSAGE(font_variation->get_palette_custom_colors() == custom_colors, "Custom palette colors should match the value that was set.");

	Dictionary opentype_features;
	opentype_features["ss01"] = 1;
	opentype_features["liga"] = 0;
	font_variation->set_opentype_features(opentype_features);
	CHECK_MESSAGE(font_variation->get_opentype_features() == opentype_features, "OpenType features should match the value that was set.");

	Dictionary variation_coordinates;
	variation_coordinates["wght"] = 700;
	variation_coordinates["wdth"] = 80;
	font_variation->set_variation_opentype(variation_coordinates);
	CHECK_MESSAGE(font_variation->get_variation_opentype() == variation_coordinates, "OpenType variation coordinates should match the value that was set.");
}

TEST_CASE("[FontVariation] Setters store copies of dictionaries") {
	Ref<FontVariation> font_variation;
	font_variation.instantiate();

	Dictionary variation_coordinates;
	variation_coordinates["wght"] = 400;
	font_variation->set_variation_opentype(variation_coordinates);

	// Mutating the original dictionary must not affect the stored copy.
	variation_coordinates["wght"] = 900;
	variation_coordinates["slnt"] = -10;

	Dictionary stored_coordinates = font_variation->get_variation_opentype();
	CHECK_MESSAGE(stored_coordinates["wght"] == Variant(400), "Stored OpenType variation coordinates should not be affected by later changes to the original dictionary.");
	CHECK_MESSAGE(!stored_coordinates.has("slnt"), "Keys added to the original dictionary after assignment should not appear in the stored copy.");

	Dictionary opentype_features;
	opentype_features["ss01"] = 1;
	font_variation->set_opentype_features(opentype_features);

	opentype_features["ss02"] = 2;
	CHECK_MESSAGE(!font_variation->get_opentype_features().has("ss02"), "Keys added to the original dictionary after assignment should not appear in the stored OpenType features.");
}

TEST_CASE("[FontVariation] Extra spacing") {
	Ref<FontVariation> font_variation;
	font_variation.instantiate();

	font_variation->set_spacing(TextServer::SPACING_GLYPH, 5);
	font_variation->set_spacing(TextServer::SPACING_SPACE, -3);
	font_variation->set_spacing(TextServer::SPACING_TOP, 10);
	font_variation->set_spacing(TextServer::SPACING_BOTTOM, 100);

	CHECK_MESSAGE(font_variation->get_spacing(TextServer::SPACING_GLYPH) == 5, "Glyph spacing should match the value that was set.");
	CHECK_MESSAGE(font_variation->get_spacing(TextServer::SPACING_SPACE) == -3, "Space spacing should match the value that was set.");
	CHECK_MESSAGE(font_variation->get_spacing(TextServer::SPACING_TOP) == 10, "Top spacing should match the value that was set.");
	CHECK_MESSAGE(font_variation->get_spacing(TextServer::SPACING_BOTTOM) == 100, "Bottom spacing should match the value that was set.");

	ERR_PRINT_OFF
	// Out-of-bounds spacing types are rejected and read as zero.
	font_variation->set_spacing((TextServer::SpacingType)TextServer::SPACING_MAX, 42);
	CHECK_MESSAGE(font_variation->get_spacing((TextServer::SpacingType)TextServer::SPACING_MAX) == 0, "Invalid spacing type should read as zero.");
	font_variation->set_spacing((TextServer::SpacingType)-1, 42);
	CHECK_MESSAGE(font_variation->get_spacing((TextServer::SpacingType)-1) == 0, "Invalid spacing type should read as zero.");
	ERR_PRINT_ON
}

TEST_CASE("[FontVariation] Base font") {
	Ref<FontVariation> font_variation;
	font_variation.instantiate();
	Ref<FontVariation> base_font;
	base_font.instantiate();

	font_variation->set_base_font(base_font);
	CHECK_MESSAGE(font_variation->get_base_font() == base_font, "Base font should match the font that was set.");

	// Re-setting the same font should be a no-op and not crash.
	font_variation->set_base_font(base_font);
	CHECK_MESSAGE(font_variation->get_base_font() == base_font, "Base font should remain unchanged after re-setting the same font.");

	// Chained variations are allowed since both derive from Font.
	Ref<FontVariation> nested_font;
	nested_font.instantiate();
	nested_font->set_base_font(font_variation);
	CHECK_MESSAGE(nested_font->get_base_font() == font_variation, "Nested base font should match the font that was set.");

	// Resetting the base font to null.
	font_variation->set_base_font(nullptr);
	CHECK_MESSAGE(font_variation->get_base_font().is_null(), "Base font should be null after resetting it.");
}

} // namespace TestFontVariation
