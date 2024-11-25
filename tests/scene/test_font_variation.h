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

#ifndef TEST_FONT_VARIATION_H
#define TEST_FONT_VARIATION_H

#include "modules/modules_enabled.gen.h"

#include "scene/resources/font.h"
#include "tests/test_macros.h"

namespace TestFontVariation {

TEST_CASE("[FontVariation] Create font variation") {
	// Create FontFile, which will be used in the font_variation
	Ref<FontFile> font_file;
	font_file.instantiate();

	// Create test instance.
	Ref<FontVariation> font_variation;
	font_variation.instantiate();

	CHECK_MESSAGE(font_variation->get_base_font().is_valid() == false, "FontVariation base font should not be valid.");

#ifdef MODULE_FREETYPE_ENABLED
	// Load a valid file.
	CHECK(font_file->load_dynamic_font("thirdparty/fonts/NotoSans_Regular.woff2") == OK);

	font_variation->set_base_font(font_file);
	font_variation->get_base_font()->set_name(font_file->get_font_name());

	// Check font_variation data.
	CHECK_MESSAGE(font_variation->get_base_font().is_valid() == true, "FontVariation base font should be valid.");
	CHECK_MESSAGE(font_variation->get_base_font()->get_name() == "Noto Sans", "Loaded correct font name.");

	font_variation->set_variation_embolden(1.2);

	CHECK_MESSAGE(font_file->get_embolden(0) == 0.0f, "FontFile embolden should be default 0.0.");
	CHECK_MESSAGE(font_file->get_embolden(0) != font_variation->get_variation_embolden(), "FontFile embolden should be different than FontVariation embolden.");
	CHECK_MESSAGE(font_variation->get_variation_embolden() == 1.2f, "FontVariation embolden should be 1.2.");

	Dictionary p_coords;
	p_coords.get_or_add("wght", 900);
	p_coords["custom_hght"] = 1000;
	font_variation->set_variation_opentype(p_coords);

	CHECK_MESSAGE(font_variation->get_variation_opentype().size() == 2, "FontVariation opentype size should be 2.");

	CHECK_MESSAGE(int(font_variation->get_variation_opentype().get_valid("wght")) == 900, "FontVariation wght should be 900.");
	CHECK_MESSAGE(int(font_variation->get_variation_opentype()["custom_hght"]) == 1000, "FontVariation custom_hght should be 1000.");
#endif
}

} // namespace TestFontVariation

#endif // TEST_FONT_VARIATION_H
