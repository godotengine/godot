/**************************************************************************/
/*  test_font_variation.cpp                                               */
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

TEST_FORCE_LINK(test_font_variation)

#include "scene/resources/font.h"

namespace TestFontVariation {

TEST_CASE("[FontVariation] Default property values") {
	Ref<FontVariation> fv;
	fv.instantiate();
	REQUIRE(fv.is_valid());

	CHECK(fv->get_base_font().is_null());
	CHECK(fv->get_variation_opentype() == Dictionary());
	CHECK_LE(Math::abs(fv->get_variation_embolden()), 0.0001f);
	CHECK(fv->get_variation_face_index() == 0);
	CHECK(fv->get_variation_transform() == Transform2D());
	CHECK(fv->get_opentype_features() == Dictionary());

	CHECK(fv->get_spacing(TextServer::SPACING_GLYPH) == 0);
	CHECK(fv->get_spacing(TextServer::SPACING_SPACE) == 0);
	CHECK(fv->get_spacing(TextServer::SPACING_TOP) == 0);
	CHECK(fv->get_spacing(TextServer::SPACING_BOTTOM) == 0);

	CHECK_LE(Math::abs(fv->get_baseline_offset()), 0.0001f);
	CHECK(fv->get_palette_index() == 0);
	CHECK(fv->get_palette_custom_colors().is_empty());
}

TEST_CASE("[FontVariation] Base font management") {
	Ref<FontVariation> fv;
	fv.instantiate();

	Ref<FontFile> base_font;
	base_font.instantiate();

	fv->set_base_font(base_font);
	CHECK(fv->get_base_font() == base_font);

	fv->set_base_font(Ref<Font>());
	CHECK(fv->get_base_font().is_null());
}

TEST_CASE("[FontVariation] OpenType variation coordinates") {
	Ref<FontVariation> fv;
	fv.instantiate();

	Dictionary coords;
	coords["wght"] = 700;
	coords["ital"] = 1;

	fv->set_variation_opentype(coords);
	CHECK(fv->get_variation_opentype() == coords);

	Dictionary updated_coords = fv->get_variation_opentype();
	CHECK(int(updated_coords["wght"]) == 700);
	CHECK(int(updated_coords["ital"]) == 1);
}

TEST_CASE("[FontVariation] Embolden, Face Index, and Transform") {
	Ref<FontVariation> fv;
	fv.instantiate();

	fv->set_variation_embolden(1.25f);
	CHECK_LE(Math::abs(fv->get_variation_embolden() - 1.25f), 0.0001f);

	fv->set_variation_face_index(3);
	CHECK(fv->get_variation_face_index() == 3);

	Transform2D custom_transform(1.0, 0.2, 0.0, 1.0, 0.0, 0.0);
	fv->set_variation_transform(custom_transform);
	CHECK(fv->get_variation_transform() == custom_transform);
}

TEST_CASE("[FontVariation] OpenType features") {
	Ref<FontVariation> fv;
	fv.instantiate();

	Dictionary features;
	features["liga"] = 1;
	features["kern"] = 0;

	fv->set_opentype_features(features);
	CHECK(fv->get_opentype_features() == features);
}

TEST_CASE("[FontVariation] Spacing adjustments") {
	Ref<FontVariation> fv;
	fv.instantiate();

	fv->set_spacing(TextServer::SPACING_GLYPH, 4);
	CHECK(fv->get_spacing(TextServer::SPACING_GLYPH) == 4);

	fv->set_spacing(TextServer::SPACING_SPACE, 10);
	CHECK(fv->get_spacing(TextServer::SPACING_SPACE) == 10);

	fv->set_spacing(TextServer::SPACING_TOP, 2);
	CHECK(fv->get_spacing(TextServer::SPACING_TOP) == 2);

	fv->set_spacing(TextServer::SPACING_BOTTOM, 5);
	CHECK(fv->get_spacing(TextServer::SPACING_BOTTOM) == 5);
}

TEST_CASE("[FontVariation] Baseline offset and Palette colors") {
	Ref<FontVariation> fv;
	fv.instantiate();

	fv->set_baseline_offset(3.5f);
	CHECK_LE(Math::abs(fv->get_baseline_offset() - 3.5f), 0.0001f);

	fv->set_palette_index(2);
	CHECK(fv->get_palette_index() == 2);

	Vector<Color> custom_colors;
	custom_colors.push_back(Color(1.0f, 0.0f, 0.0f, 1.0f));
	custom_colors.push_back(Color(0.0f, 1.0f, 0.0f, 1.0f));
	fv->set_palette_custom_colors(custom_colors);

	Vector<Color> retrieved_colors = fv->get_palette_custom_colors();
	REQUIRE(retrieved_colors.size() == 2);
	CHECK(retrieved_colors[0] == Color(1.0f, 0.0f, 0.0f, 1.0f));
	CHECK(retrieved_colors[1] == Color(0.0f, 1.0f, 0.0f, 1.0f));
}

} // namespace TestFontVariation
