/**************************************************************************/
/*  test_font_file.h                                                      */
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

#ifndef TEST_FONT_FILE_H
#define TEST_FONT_FILE_H

#include "scene/resources/font.h"

#include "tests/test_macros.h"
#include "tests/test_utils.h"

namespace TestFontFile {

TEST_CASE("[SceneTree][FontFile] Constructor") {
	Ref<FontFile> font_file = memnew(FontFile);
	CHECK(font_file->get_font_name().is_empty());
	CHECK(font_file->get_font_stretch() == 100);
	CHECK(font_file->get_font_style().is_empty());
	CHECK(font_file->get_font_style_name().is_empty());
	CHECK(font_file->get_font_weight() == 400);
}

TEST_CASE("[SceneTree][FontFile] allow_system_fallback") {
	Ref<FontFile> font_file = memnew(FontFile);
	CHECK(font_file->is_allow_system_fallback()); // Allows system fallback
	font_file->set_allow_system_fallback(false);
	CHECK(!font_file->is_allow_system_fallback()); // Disallows system fallback
}

TEST_CASE("[SceneTree][FontFile] DEFAULT_FONT_SIZE") {
	Ref<FontFile> font_file = memnew(FontFile);
	CHECK(font_file->DEFAULT_FONT_SIZE == 16);
}

TEST_CASE("[SceneTree][FontFile] set_path") {
	Ref<FontFile> font_file = memnew(FontFile);
	String path = TestUtils::get_data_path("fonts/NotoSans_Regular.woff2");
	font_file->set_path(path, true);
	CHECK(font_file->get_path() == path);
}

} // namespace TestFontFile

#endif // TEST_FONT_FILE_H
