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
/* without limitation the rights to use, copy, modify, merge, publish,   */
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

	CHECK(fv->get_base_font().is_null());
	CHECK(fv->get_variation_opentype().is_empty());
	CHECK(fv->get_variation_embolden() == doctest::Approx(0.0));
	CHECK(fv->get_variation_face_index() == 0);
	CHECK(fv->get_variation_transform() == Transform2D());
	CHECK(fv->get_opentype_features().is_empty());
	CHECK(fv->get_spacing(TextServer::SPACING_TOP) == 0);
	CHECK(fv->get_spacing(TextServer::SPACING_BOTTOM) == 0);
	CHECK(fv->get_spacing(TextServer::SPACING_SPACE) == 0);
	CHECK(fv->get_spacing(TextServer::SPACING_GLYPH) == 0);
	CHECK(fv->get_baseline_offset() == doctest::Approx(0.0));
	CHECK(fv->get_palette_index() == 0);
	CHECK(fv->get_palette_custom_colors().is_empty());
}

TEST_CASE("[FontVariation] Base font setter and getter") {
	Ref<FontVariation> fv;
	fv.instantiate();

	Ref<FontFile> base;
	base.instantiate();
	fv->set_base_font(base);
	CHECK(fv->get_base_font() == base);

	// Replacing the base font works.
	Ref<FontFile> base2;
	base2.instantiate();
	fv->set_base_font(base2);
	CHECK(fv->get_base_font() == base2);

	// Clearing the base font works.
	fv->set_base_font(Ref<Font>());
	CHECK(fv->get_base_font().is_null());
}

TEST_CASE("[FontVariation] Variation coordinates") {
	Ref<FontVariation> fv;
	fv.instantiate();

	Dictionary coords;
	coords[0x77676874] = 700; // 'wght' tag.
	coords[0x77647468] = 100; // 'wdth' tag.
	fv->set_variation_opentype(coords);
	CHECK(fv->get_variation_opentype() == coords);

	// The setter duplicates the dictionary: later mutations of the
	// original must not affect the stored coordinates.
	coords[0x77676874] = 400;
	CHECK(int(fv->get_variation_opentype()[0x77676874]) == 700);

	// The getter returns a duplicate: mutating it must not affect
	// the stored coordinates.
	Dictionary retrieved = fv->get_variation_opentype();
	retrieved[0x77676874] = 100;
	CHECK(int(fv->get_variation_opentype()[0x77676874]) == 700);

	// Clearing works.
	fv->set_variation_opentype(Dictionary());
	CHECK(fv->get_variation_opentype().is_empty());
}

TEST_CASE("[FontVariation] Variation embolden, transform and face index") {
	Ref<FontVariation> fv;
	fv.instantiate();

	fv->set_variation_embolden(1.2f);
	CHECK(fv->get_variation_embolden() == doctest::Approx(1.2));

	fv->set_variation_embolden(-0.5f);
	CHECK(fv->get_variation_embolden() == doctest::Approx(-0.5));

	Transform2D transform(0.0f, -1.0f, 2.0f, 0.0f, 10.0f, 20.0f);
	fv->set_variation_transform(transform);
	CHECK(fv->get_variation_transform() == transform);

	fv->set_variation_face_index(3);
	CHECK(fv->get_variation_face_index() == 3);
}

TEST_CASE("[FontVariation] OpenType features") {
	Ref<FontVariation> fv;
	fv.instantiate();

	Dictionary features;
	features["liga"] = 1;
	features["smcp"] = 0;
	fv->set_opentype_features(features);
	CHECK(fv->get_opentype_features() == features);

	// The setter duplicates the dictionary.
	features["liga"] = 0;
	CHECK(int(fv->get_opentype_features()["liga"]) == 1);

	// Clearing works.
	fv->set_opentype_features(Dictionary());
	CHECK(fv->get_opentype_features().is_empty());
}

TEST_CASE("[FontVariation] Extra spacing") {
	Ref<FontVariation> fv;
	fv.instantiate();

	fv->set_spacing(TextServer::SPACING_TOP, 4);
	fv->set_spacing(TextServer::SPACING_BOTTOM, -2);
	fv->set_spacing(TextServer::SPACING_SPACE, 8);
	fv->set_spacing(TextServer::SPACING_GLYPH, 1);

	CHECK(fv->get_spacing(TextServer::SPACING_TOP) == 4);
	CHECK(fv->get_spacing(TextServer::SPACING_BOTTOM) == -2);
	CHECK(fv->get_spacing(TextServer::SPACING_SPACE) == 8);
	CHECK(fv->get_spacing(TextServer::SPACING_GLYPH) == 1);

	// Spacing types are independent of each other.
	fv->set_spacing(TextServer::SPACING_TOP, 0);
	CHECK(fv->get_spacing(TextServer::SPACING_TOP) == 0);
	CHECK(fv->get_spacing(TextServer::SPACING_BOTTOM) == -2);
}

TEST_CASE("[FontVariation] Baseline offset and palette") {
	Ref<FontVariation> fv;
	fv.instantiate();

	fv->set_baseline_offset(2.5f);
	CHECK(fv->get_baseline_offset() == doctest::Approx(2.5));

	fv->set_palette_index(2);
	CHECK(fv->get_palette_index() == 2);

	Vector<Color> colors;
	colors.push_back(Color(1.0f, 0.0f, 0.0f));
	colors.push_back(Color(0.0f, 1.0f, 0.0f, 0.5f));
	fv->set_palette_custom_colors(colors);
	CHECK(fv->get_palette_custom_colors() == colors);

	fv->set_palette_custom_colors(Vector<Color>());
	CHECK(fv->get_palette_custom_colors().is_empty());
}

} // namespace TestFontVariation
