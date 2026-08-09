/**************************************************************************/
/*  test_string_builder.cpp                                               */
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

#include "core/string/string_builder.h"
#include "tests/test_macros.h"

TEST_FORCE_LINK(test_string_builder)

namespace TestString_builder {

TEST_CASE("[String_builder] Test append()") {
	
	String test_string("Hello, append() test!");
	CHECK(test_string == "Hello, append() test!");

	const char *test_cstring = "Hello, append() test!";
	StringBuilder string_builder_cstring = StringBuilder().append(test_cstring);
	StringBuilder appended_test_cstring = string_builder_cstring.append(" appended!");
	CHECK(appended_test_cstring.as_string() == "Hello, append() test! appended!");

	String empty_string("");
	String &empty_string_ref = empty_string;
	StringBuilder string_builder_empty = StringBuilder().append(empty_string_ref);
	CHECK(string_builder_empty.as_string() == empty_string);

	String &test_string_ref = test_string;
	StringBuilder string_builder = StringBuilder().append(test_string_ref);
	StringBuilder appended_test_string = string_builder.append(" appended!");
	CHECK(appended_test_string.as_string() == "Hello, append() test! appended!");
}

} // namespace TestString_builder
