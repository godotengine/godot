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

#include "modules/modules_enabled.gen.h" // For freetype.

namespace TestFontVariation {

TEST_CASE("[FontVariation] Variation settings roundtrip") {
	Ref<FontVariation> fv;
	fv.instantiate();

	// No base font is set by default; queries then fall back to the default theme font.
	CHECK(fv->get_base_font().is_null());

	// OpenType variation coordinates are stored and returned as-is.
	Dictionary coords;
	coords["wght"] = 700;
	fv->set_variation_opentype(coords);
	CHECK(fv->get_variation_opentype() == coords);

	// Synthetic embolden strength.
	fv->set_variation_embolden(1.5);
	CHECK(fv->get_variation_embolden() == doctest::Approx(1.5));

	// Synthetic slant transform.
	Transform2D xform(1.0, 0.2, 0.0, 1.0, 0.0, 0.0);
	fv->set_variation_transform(xform);
	CHECK(fv->get_variation_transform() == xform);

	// Face index for font collections.
	fv->set_variation_face_index(1);
	CHECK(fv->get_variation_face_index() == 1);

	// Reset to defaults.
	fv->set_variation_opentype(Dictionary());
	CHECK(fv->get_variation_opentype() == Dictionary());
	fv->set_variation_embolden(0.0);
	CHECK(fv->get_variation_embolden() == doctest::Approx(0.0));
	fv->set_variation_face_index(0);
	CHECK(fv->get_variation_face_index() == 0);
}

TEST_CASE("[FontVariation] Base font delegation") {
#ifdef MODULE_FREETYPE_ENABLED
	Ref<FontFile> ff;
	ff.instantiate();
	CHECK(ff->load_dynamic_font("thirdparty/fonts/Inter_Regular.woff2") == OK);

	Ref<FontVariation> fv;
	fv.instantiate();
	fv->set_base_font(ff);
	CHECK(fv->get_base_font() == ff);

	// Without any variation applied, the variation reports the base font's
	// metadata and metrics.
	CHECK(fv->get_font_name() == ff->get_font_name());
	CHECK(fv->get_font_style_name() == ff->get_font_style_name());
	CHECK(fv->get_font_weight() == ff->get_font_weight());
	CHECK(fv->get_height(16) == doctest::Approx(ff->get_height(16)));
	CHECK(fv->get_ascent(16) == doctest::Approx(ff->get_ascent(16)));
	CHECK(fv->get_descent(16) == doctest::Approx(ff->get_descent(16)));
	CHECK(fv->get_string_size("Test") == ff->get_string_size("Test"));
#endif
}

} // namespace TestFontVariation
