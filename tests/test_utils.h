/**************************************************************************/
/*  test_utils.h                                                          */
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

#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "core/string/char_utils.h"
#include "tests/test_macros.h"

class String;

namespace TestUtils {

String get_data_path(const String &p_file);
String get_executable_dir();

TEST_CASE("[UTILS] Test cjk chr") {
	CHECK(is_cjk_character(0x53D8) == true); // '变'
	CHECK(is_cjk_character(0x5909) == true); // '変'
	CHECK(is_cjk_character(0xB3D9) == true); // '변'
}

TEST_CASE("[UTILS] Test cjk chr") {
	CHECK(is_cjk_punctuation(0x3002) == true); // Ideographic full stop
	CHECK(is_cjk_punctuation(0x3001) == true); // Ideographic comma
	CHECK(is_cjk_punctuation(0x3005) == true); // Ideographic iteration mark
}

} // namespace TestUtils

#endif // TEST_UTILS_H
