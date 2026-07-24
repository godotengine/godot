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

#include "modules/modules_enabled.gen.h"

namespace TestFontVariation {

TEST_CASE("[FontVariation] Set and get properties") {
	Ref<FontVariation> fv;
	fv.instantiate();

	fv->set_variation_embolden(0.5f);
	CHECK(fv->get_variation_embolden() == doctest::Approx(0.5f));

	fv->set_variation_face_index(3);
	CHECK(fv->get_variation_face_index() == 3);

	fv->set_baseline_offset(0.25f);
	CHECK(fv->get_baseline_offset() == doctest::Approx(0.25f));

	fv->set_spacing(TextServer::SPACING_TOP, 5);
	CHECK(fv->get_spacing(TextServer::SPACING_TOP) == 5);
}

TEST_CASE("[FontVariation] Extra spacing is applied to font metrics") {
#ifdef MODULE_FREETYPE_ENABLED
	Ref<FontFile> base;
	base.instantiate();

	String test_variable_font = "tests/data/font/InterVariable.ttf";
	REQUIRE_EQ(base->load_dynamic_font(test_variable_font), OK);

	Ref<FontVariation> fv;
	fv.instantiate();
	fv->set_base_font(base);

	const int size = 32;
	real_t base_h = base->get_height(size);
	real_t base_asc = base->get_ascent(size);
	real_t base_desc = base->get_descent(size);

	fv->set_spacing(TextServer::SPACING_TOP, 5);
	fv->set_spacing(TextServer::SPACING_BOTTOM, 3);

	CHECK(fv->get_ascent(size) == doctest::Approx(base_asc + 5));
	CHECK(fv->get_descent(size) == doctest::Approx(base_desc + 3));

	// SPACING_GLYPH adds space after each glyph EXCEPT the last glyph
	real_t base_w = base->get_string_size("AAA", HORIZONTAL_ALIGNMENT_LEFT, -1, size).width;
	fv->set_spacing(TextServer::SPACING_GLYPH, 4);
	CHECK(fv->get_string_size("AAA", HORIZONTAL_ALIGNMENT_LEFT, -1, size).width == doctest::Approx(base_w + 8));
#endif
}

TEST_CASE("[FontVariation] Spacing accepts valid types and rejects out-of-range ones") {
	Ref<FontVariation> fv;
	fv.instantiate();

	fv->set_spacing(TextServer::SPACING_TOP, 7);
	fv->set_spacing(TextServer::SPACING_BOTTOM, 9);
	fv->set_spacing((TextServer::SpacingType)2, 100);

	// Indices outside SpacingType = [0, SPACING_MAX) should be be rejected
	ERR_PRINT_OFF
	fv->set_spacing((TextServer::SpacingType)TextServer::SPACING_MAX, 5);
	fv->set_spacing((TextServer::SpacingType)-1, 100);
	fv->set_spacing((TextServer::SpacingType)9999, 10);

	// Returns default 0 for out-of-range read
	CHECK(fv->get_spacing((TextServer::SpacingType)9999) == 0);
	ERR_PRINT_ON

	// The rejected calls above must not have altered any valid state.
	CHECK(fv->get_spacing(TextServer::SPACING_TOP) == 100);
	CHECK(fv->get_spacing(TextServer::SPACING_BOTTOM) == 9);
}

} // namespace TestFontVariation
