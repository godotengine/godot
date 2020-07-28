/*************************************************************************/
/*  test_json.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef TEST_JSON_H
#define TEST_JSON_H

#include "core/io/json.h"

#include "tests/test_macros.h"

namespace TestJSON {

TEST_SUITE("[JSON] Issues") {
	TEST_CASE_PENDING("https://github.com/godotengine/godot/issues/40794") {
		// Correct:  R"({ "a": 12345, "b": 12345 })"
		String json = R"( "a": 12345, "b": 12345 })"; // Missing starting bracket.
		Variant parsed;
		String err_str;
		int line;

		Error err = JSON::parse(json, parsed, err_str, line);

		CHECK_MESSAGE(err == ERR_PARSE_ERROR,
				"Missing starting curly bracket, this should be treated as a parse error.");
		CHECK_MESSAGE(parsed.get_type() != Variant::STRING,
				"Should not prematurely parse as a string.");
	}
}

} // namespace TestJSON

#endif // TEST_JSON_H
