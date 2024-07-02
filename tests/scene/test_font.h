/**************************************************************************/
/*  test_font.h                                                           */
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

#ifndef TEST_FONT_H
#define TEST_FONT_H

#include "scene/resources/font.h"
#include "tests/test_macros.h"
#include "tests/test_utils.h"

namespace TestFont {
TEST_CASE("[String] Instantiation") {
	Ref<FontFile> ff;
	ff.instantiate();

	REQUIRE_MESSAGE(ff != nullptr, "Reference should not be null.");
}

// Macro to generate tests for getters.
#define GETTER_TEST(m_name, m_getter, m_default_value)                                                                                          \
	TEST_CASE("[String] Load Dynamic Font - get-set " m_name) {                                                                                 \
		String test_dynamic_font = String("thirdparty").path_join("fonts").path_join("NotoSansHebrew_Regular.woff2");                           \
		Ref<FontFile> ff;                                                                                                                       \
		ff.instantiate();                                                                                                                       \
		ff->load_dynamic_font(test_dynamic_font);                                                                                               \
		CHECK_MESSAGE(ff->m_getter == m_default_value, "Unexpected original value for ", m_name, " : ", ff->m_getter, " != ", m_default_value); \
	}

#define GETTER_TEST_REAL(m_name, m_getter, m_default_value)                                                                                                      \
	TEST_CASE("[String] Load Dynamic Font - get-set " m_name) {                                                                                                  \
		String test_dynamic_font = String("thirdparty").path_join("fonts").path_join("NotoSansHebrew_Regular.woff2");                                            \
		Ref<FontFile> ff;                                                                                                                                        \
		ff.instantiate();                                                                                                                                        \
		ff->load_dynamic_font(test_dynamic_font);                                                                                                                \
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

} //namespace TestFont

#endif // TEST_FONT_H
