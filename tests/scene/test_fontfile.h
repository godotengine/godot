/**************************************************************************/
/*  test_fontfile.h                                                       */
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

namespace TestFontfile {

#ifdef MODULE_FREETYPE_ENABLED
// Macro to generate tests for getters.
#define GETTER_TEST(m_name, m_getter, m_default_value)                                                                                          \
	TEST_CASE("[FontFile] Load Dynamic Font - get-set " m_name) {                                                                               \
		String test_dynamic_font = "thirdparty/fonts/NotoSansHebrew_Regular.woff2";                                                             \
		Ref<FontFile> ff;                                                                                                                       \
		ff.instantiate();                                                                                                                       \
		CHECK(ff->load_dynamic_font(test_dynamic_font) == OK);                                                                                  \
		CHECK_MESSAGE(ff->m_getter == m_default_value, "Unexpected original value for ", m_name, " : ", ff->m_getter, " != ", m_default_value); \
	}

#define GETTER_TEST_REAL(m_name, m_getter, m_default_value)                                                                                                      \
	TEST_CASE("[FontFile] Load Dynamic Font - get-set " m_name) {                                                                                                \
		String test_dynamic_font = "thirdparty/fonts/NotoSansHebrew_Regular.woff2";                                                                              \
		Ref<FontFile> ff;                                                                                                                                        \
		ff.instantiate();                                                                                                                                        \
		CHECK(ff->load_dynamic_font(test_dynamic_font) == OK);                                                                                                   \
		CHECK_MESSAGE(ff->m_getter == doctest::Approx(m_default_value), "Unexpected original value for ", m_name, " : ", ff->m_getter, " != ", m_default_value); \
	}

Dictionary expected_ot_name_strings() {
	Dictionary d = Dictionary();
	d["en"] = Dictionary();
	((Dictionary)d["en"])["copyright"] = "Copyright 2022 The Noto Project Authors (https://github.com/notofonts/hebrew)";
	((Dictionary)d["en"])["family_name"] = "Noto Sans Hebrew";
	((Dictionary)d["en"])["subfamily_name"] = "Regular";
	((Dictionary)d["en"])["full_name"] = "Noto Sans Hebrew Regular";
	((Dictionary)d["en"])["unique_identifier"] = "2.003;GOOG;NotoSansHebrew-Regular";
	((Dictionary)d["en"])["version"] = "Version 2.003";
	((Dictionary)d["en"])["postscript_name"] = "NotoSansHebrew-Regular";
	((Dictionary)d["en"])["trademark"] = "Noto is a trademark of Google Inc.";
	((Dictionary)d["en"])["license"] = "This Font Software is licensed under the SIL Open Font License, Version 1.1. This license is available with a FAQ at: https://scripts.sil.org/OFL";
	((Dictionary)d["en"])["license_url"] = "https://scripts.sil.org/OFL";
	((Dictionary)d["en"])["designer"] = "Monotype Design Team";
	((Dictionary)d["en"])["designer_url"] = "http://www.monotype.com/studio";
	((Dictionary)d["en"])["description"] = "Designed by Monotype design team.";
	((Dictionary)d["en"])["manufacturer"] = "Monotype Imaging Inc.";
	((Dictionary)d["en"])["vendor_url"] = "http://www.google.com/get/noto/";

	return d;
}

// These properties come from the font file itself.
GETTER_TEST("font_name", get_font_name(), "Noto Sans Hebrew")
GETTER_TEST("font_style_name", get_font_style_name(), "Regular")
GETTER_TEST("font_weight", get_font_weight(), 400)
GETTER_TEST("font_stretch", get_font_stretch(), 100)
GETTER_TEST("opentype_features", get_opentype_features(), Dictionary())
GETTER_TEST("ot_name_strings", get_ot_name_strings(), expected_ot_name_strings())

// These are dependent on size and potentially other state. Act as regression tests based of arbitrary small size 10 and large size 100.
GETTER_TEST_REAL("height-small", get_height(10), (real_t)14)
GETTER_TEST_REAL("ascent-small", get_ascent(10), (real_t)11)
GETTER_TEST_REAL("descent-small", get_descent(10), (real_t)3)
GETTER_TEST_REAL("underline_position-small", get_underline_position(10), (real_t)1.25)
GETTER_TEST_REAL("underline_thickness-small", get_underline_thickness(10), (real_t)0.5)

GETTER_TEST_REAL("height-large", get_height(100), (real_t)137)
GETTER_TEST_REAL("ascent-large", get_ascent(100), (real_t)107)
GETTER_TEST_REAL("descent-large", get_descent(100), (real_t)30)
GETTER_TEST_REAL("underline_position-large", get_underline_position(100), (real_t)12.5)
GETTER_TEST_REAL("underline_thickness-large", get_underline_thickness(100), (real_t)5)

#endif

TEST_CASE("[FontFile] Create font file and check data") {
	// Create test instance.
	Ref<FontFile> font_file;
	font_file.instantiate();

#ifdef MODULE_FREETYPE_ENABLED
	// Try to load non-existent files.
	ERR_PRINT_OFF
	CHECK(font_file->load_dynamic_font("") == OK);
	CHECK_MESSAGE(font_file->get_data().is_empty() == true, "Invalid fontfile should not be loaded.");

	CHECK(font_file->load_dynamic_font("thirdparty/fonts/nofonthasthisname.woff2") == OK);
	CHECK_MESSAGE(font_file->get_data().is_empty() == true, "Invalid fontfile should not be loaded.");
	ERR_PRINT_ON

	// Load a valid file.
	CHECK(font_file->load_dynamic_font("thirdparty/fonts/NotoSans_Regular.woff2") == OK);

	// Check fontfile data.
	CHECK_MESSAGE(font_file->get_data().is_empty() == false, "Fontfile should have been loaded.");
	CHECK_MESSAGE(font_file->get_font_name() == "Noto Sans", "Loaded correct font name.");
	CHECK_MESSAGE(font_file->get_font_style_name() == "Regular", "Loaded correct font style.");
	CHECK_MESSAGE(font_file->get_data().size() == 148480llu, "Whole fontfile was loaded.");

	// Valid glyphs.
	CHECK_MESSAGE(font_file->get_glyph_index(2, 'a', 0) != 0, "Glyph index for 'a' is valid.");
	CHECK_MESSAGE(font_file->get_glyph_index(2, 'b', 0) != 0, "Glyph index for 'b' is valid.");
	CHECK_MESSAGE(font_file->get_glyph_index(2, 0x0103, 0) != 0, "Glyph index for 'latin small letter a with breve' is valid.");
	CHECK_MESSAGE(font_file->get_glyph_index(2, 0x03a8, 0) != 0, "Glyph index for 'Greek psi' is valid.");
	CHECK_MESSAGE(font_file->get_glyph_index(2, 0x0416, 0) != 0, "Glyph index for 'Cyrillic zhe' is valid.");
	CHECK_MESSAGE(font_file->get_glyph_index(2, '&', 0) != 0, "Glyph index for '&' is valid.");

	// Invalid glyphs.
	CHECK_MESSAGE(font_file->get_glyph_index(2, 0x4416, 0) == 0, "Glyph index is invalid.");
	CHECK_MESSAGE(font_file->get_glyph_index(2, 0x5555, 0) == 0, "Glyph index is invalid.");
	CHECK_MESSAGE(font_file->get_glyph_index(2, 0x2901, 0) == 0, "Glyph index is invalid.");
#endif
}

} // namespace TestFontfile
