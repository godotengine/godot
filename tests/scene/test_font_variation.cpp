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
#include "tests/signal_watcher.h"
#include "tests/test_utils.h"

#include "modules/modules_enabled.gen.h" // For freetype.

namespace TestFontVariation {

TEST_CASE("[FontVariation] Default values") {
	Ref<FontVariation> fv;
	fv.instantiate();

	CHECK(fv->get_base_font().is_null());
	CHECK(fv->get_variation_embolden() == doctest::Approx(0.0));
	CHECK(fv->get_variation_face_index() == 0);
	CHECK(fv->get_variation_transform() == Transform2D());
	CHECK(fv->get_variation_opentype().is_empty());
	CHECK(fv->get_opentype_features().is_empty());
	CHECK(fv->get_baseline_offset() == doctest::Approx(0.0));
	CHECK(fv->get_palette_index() == 0);
	CHECK(fv->get_palette_custom_colors().is_empty());

	for (int i = 0; i < TextServer::SPACING_MAX; i++) {
		CHECK(fv->get_spacing(TextServer::SpacingType(i)) == 0);
	}
}

TEST_CASE("[FontVariation] Base font") {
	Ref<FontVariation> fv;
	fv.instantiate();

	Ref<FontFile> ff;
	ff.instantiate();

	CHECK(fv->get_base_font().is_null());
	fv->set_base_font(ff);
	CHECK(fv->get_base_font() == ff);

	fv->set_base_font(Ref<Font>());
	CHECK(fv->get_base_font().is_null());
}

TEST_CASE("[FontVariation] Setters and getters") {
	Ref<FontVariation> fv;
	fv.instantiate();

	fv->set_variation_embolden(1.5);
	CHECK(fv->get_variation_embolden() == doctest::Approx(1.5));

	fv->set_variation_face_index(2);
	CHECK(fv->get_variation_face_index() == 2);

	Transform2D t(1, 2, 3, 4, 5, 6);
	fv->set_variation_transform(t);
	CHECK(fv->get_variation_transform() == t);

	Dictionary opentype;
	opentype["wght"] = 500;
	fv->set_variation_opentype(opentype);
	CHECK(fv->get_variation_opentype() == opentype);

	Dictionary opentype_features;
	opentype_features["liga"] = 1;
	fv->set_opentype_features(opentype_features);
	CHECK(fv->get_opentype_features() == opentype_features);

	fv->set_spacing(TextServer::SPACING_GLYPH, 5);
	CHECK(fv->get_spacing(TextServer::SPACING_GLYPH) == 5);

	fv->set_spacing(TextServer::SPACING_SPACE, 10);
	CHECK(fv->get_spacing(TextServer::SPACING_SPACE) == 10);

	fv->set_baseline_offset(2.5);
	CHECK(fv->get_baseline_offset() == doctest::Approx(2.5));

	fv->set_palette_index(1);
	CHECK(fv->get_palette_index() == 1);

	Vector<Color> custom_colors;
	custom_colors.push_back(Color(1, 0, 0));
	custom_colors.push_back(Color(0, 1, 0));
	fv->set_palette_custom_colors(custom_colors);
	CHECK(fv->get_palette_custom_colors() == custom_colors);
}

TEST_CASE("[FontVariation] Dictionaries are stored as copies") {
	Ref<FontVariation> fv;
	fv.instantiate();

	Dictionary opentype;
	opentype["wght"] = 500;
	fv->set_variation_opentype(opentype);

	Dictionary features;
	features["liga"] = 1;
	fv->set_opentype_features(features);

	// Mutating the caller's dictionaries must not reach into the font.
	opentype["wght"] = 700;
	features["liga"] = 0;

	CHECK(int(fv->get_variation_opentype()["wght"]) == 500);
	CHECK(int(fv->get_opentype_features()["liga"]) == 1);
}

TEST_CASE("[FontVariation] Reset state") {
	Ref<FontVariation> fv;
	fv.instantiate();

	Ref<FontFile> ff;
	ff.instantiate();

	Dictionary opentype;
	opentype["wght"] = 500;

	Vector<Color> custom_colors;
	custom_colors.push_back(Color(1, 0, 0));

	fv->set_base_font(ff);
	fv->set_variation_embolden(1.5);
	fv->set_variation_face_index(2);
	fv->set_variation_transform(Transform2D(1, 2, 3, 4, 5, 6));
	fv->set_variation_opentype(opentype);
	fv->set_opentype_features(opentype);
	fv->set_spacing(TextServer::SPACING_GLYPH, 5);
	fv->set_baseline_offset(2.5);
	fv->set_palette_index(1);
	fv->set_palette_custom_colors(custom_colors);

	// Reloading a resource from disk resets it through its base `Resource` handle,
	// which is the only way `reset_state()` is reached outside the class.
	Ref<Resource> res = fv;
	res->reset_state();

	CHECK(fv->get_base_font().is_null());
	CHECK(fv->get_variation_embolden() == doctest::Approx(0.0));
	CHECK(fv->get_variation_face_index() == 0);
	CHECK(fv->get_variation_transform() == Transform2D());
	CHECK(fv->get_variation_opentype().is_empty());
	CHECK(fv->get_opentype_features().is_empty());
	CHECK(fv->get_baseline_offset() == doctest::Approx(0.0));
	CHECK(fv->get_palette_index() == 0);
	CHECK(fv->get_palette_custom_colors().is_empty());

	for (int i = 0; i < TextServer::SPACING_MAX; i++) {
		CHECK(fv->get_spacing(TextServer::SpacingType(i)) == 0);
	}
}

TEST_CASE("[FontVariation] Changed signal is only emitted when a value differs") {
	Ref<FontVariation> fv;
	fv.instantiate();

	SIGNAL_WATCH(*fv, CoreStringName(changed));
	Array empty_args = { {} };

	SUBCASE("Embolden") {
		fv->set_variation_embolden(1.5);
		SIGNAL_CHECK("changed", empty_args);

		fv->set_variation_embolden(1.5);
		SIGNAL_CHECK_FALSE("changed");
	}

	SUBCASE("Face index") {
		fv->set_variation_face_index(2);
		SIGNAL_CHECK("changed", empty_args);

		fv->set_variation_face_index(2);
		SIGNAL_CHECK_FALSE("changed");
	}

	SUBCASE("Spacing") {
		fv->set_spacing(TextServer::SPACING_GLYPH, 5);
		SIGNAL_CHECK("changed", empty_args);

		fv->set_spacing(TextServer::SPACING_GLYPH, 5);
		SIGNAL_CHECK_FALSE("changed");
	}

	SUBCASE("Palette index") {
		fv->set_palette_index(1);
		SIGNAL_CHECK("changed", empty_args);

		fv->set_palette_index(1);
		SIGNAL_CHECK_FALSE("changed");
	}

	SUBCASE("Base font") {
		Ref<FontFile> ff;
		ff.instantiate();

		fv->set_base_font(ff);
		SIGNAL_CHECK("changed", empty_args);

		fv->set_base_font(ff);
		SIGNAL_CHECK_FALSE("changed");
	}

	SIGNAL_UNWATCH(*fv, CoreStringName(changed));
}

TEST_CASE("[FontVariation] Changes to the base font are propagated") {
	Ref<FontVariation> fv;
	fv.instantiate();

	Ref<FontFile> ff;
	ff.instantiate();
	fv->set_base_font(ff);

	SIGNAL_WATCH(*fv, CoreStringName(changed));
	Array empty_args = { {} };

	ff->set_fixed_size(16);
	SIGNAL_CHECK("changed", empty_args);

	// After being detached, the old base font no longer notifies the variation.
	fv->set_base_font(Ref<Font>());
	SIGNAL_DISCARD("changed");

	ff->set_fixed_size(32);
	SIGNAL_CHECK_FALSE("changed");

	SIGNAL_UNWATCH(*fv, CoreStringName(changed));
}

#ifdef MODULE_FREETYPE_ENABLED

// Loads the ASCII subset of the variable Inter font shipped in `tests/data/fonts`.
// It exposes a `wght` axis (100..900), which is what makes actual variation testable.
static Ref<FontFile> load_variable_font() {
	Ref<FontFile> ff;
	ff.instantiate();
	REQUIRE(ff->load_dynamic_font(TestUtils::get_data_path("fonts/InterVariable-ASCII.woff2")) == OK);
	return ff;
}

TEST_CASE("[FontVariation] Without modifications it measures like its base font") {
	Ref<FontFile> ff = load_variable_font();

	Ref<FontVariation> fv;
	fv.instantiate();
	fv->set_base_font(ff);

	const int font_size = 16;
	CHECK(fv->get_height(font_size) == doctest::Approx(ff->get_height(font_size)));
	CHECK(fv->get_ascent(font_size) == doctest::Approx(ff->get_ascent(font_size)));
	CHECK(fv->get_descent(font_size) == doctest::Approx(ff->get_descent(font_size)));
	CHECK(fv->get_string_size("Test", HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).x ==
			doctest::Approx(ff->get_string_size("Test", HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).x));
}

TEST_CASE("[FontVariation] OpenType variation axes change glyph metrics") {
	Ref<FontFile> ff = load_variable_font();

	Ref<FontVariation> light;
	light.instantiate();
	light->set_base_font(ff);

	Ref<FontVariation> heavy;
	heavy.instantiate();
	heavy->set_base_font(ff);

	// Axes are keyed by their OpenType tag. `Font::get_supported_variation_list()`
	// reports this font's `wght` axis as spanning 100..900 with a default of 400.
	const int64_t wght = TS->name_to_tag("wght");

	Dictionary light_coords;
	light_coords[wght] = 100;
	light->set_variation_opentype(light_coords);

	Dictionary heavy_coords;
	heavy_coords[wght] = 900;
	heavy->set_variation_opentype(heavy_coords);

	const String text = "Test text";
	const int font_size = 32;

	// A heavier weight makes the same string wider.
	CHECK(heavy->get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).x >
			light->get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).x);

	// `find_variation()` resolves the coordinates it is given against the base font,
	// independently of the variation's own settings, so distinct coordinates must
	// map to distinct font RIDs.
	CHECK(light->find_variation(light_coords) != light->find_variation(heavy_coords));
}

TEST_CASE("[FontVariation] Embolden changes glyph metrics") {
	Ref<FontFile> ff = load_variable_font();

	Ref<FontVariation> fv;
	fv.instantiate();
	fv->set_base_font(ff);

	const String text = "Test text";
	const int font_size = 32;
	const real_t plain_width = fv->get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).x;

	// Unlike `wght`, embolden is a synthetic effect and does not need a variable font.
	// 0.6 is the strength the editor theme itself uses to synthesize its bold font.
	fv->set_variation_embolden(0.6);
	CHECK(fv->get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).x > plain_width);
}

TEST_CASE("[FontVariation] Spacing changes measured size") {
	Ref<FontFile> ff = load_variable_font();

	Ref<FontVariation> fv;
	fv.instantiate();
	fv->set_base_font(ff);

	// "Test text" is 8 glyphs plus one space. Glyph spacing and space spacing are
	// applied separately, so the space is not counted as a glyph.
	const String text = "Test text";
	const int glyph_count = 8;
	const int space_count = 1;
	const int font_size = 16;
	const real_t plain_width = fv->get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).x;

	fv->set_spacing(TextServer::SPACING_GLYPH, 4);
	CHECK(fv->get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).x ==
			doctest::Approx(plain_width + 4 * glyph_count));

	fv->set_spacing(TextServer::SPACING_GLYPH, 0);
	fv->set_spacing(TextServer::SPACING_SPACE, 8);
	CHECK(fv->get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).x ==
			doctest::Approx(plain_width + 8 * space_count));

	// Top and bottom spacing extend the line height instead of its width.
	fv->set_spacing(TextServer::SPACING_SPACE, 0);
	fv->set_spacing(TextServer::SPACING_TOP, 5);
	fv->set_spacing(TextServer::SPACING_BOTTOM, 5);
	CHECK(fv->get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).x == doctest::Approx(plain_width));
	CHECK(fv->get_height(font_size) == doctest::Approx(ff->get_height(font_size) + 10));
}

#endif // MODULE_FREETYPE_ENABLED

} // namespace TestFontVariation
