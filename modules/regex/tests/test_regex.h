/*************************************************************************/
/*  test_regex.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef TEST_REGEX_H
#define TEST_REGEX_H

#include "core/string/ustring.h"
#include "modules/regex/regex.h"

#include "tests/test_macros.h"

namespace TestRegEx {

TEST_CASE("[RegEx] Initialization") {
	const String pattern = "(?<vowel>[aeiou])";

	RegEx re1(pattern);
	CHECK(re1.is_valid());
	CHECK(re1.get_pattern() == pattern);
	CHECK(re1.get_group_count() == 1);

	Array names = re1.get_names();
	CHECK(names.size() == 1);
	CHECK(names[0] == "vowel");

	RegEx re2;
	CHECK(re2.is_valid() == false);
	CHECK(re2.compile(pattern) == OK);
	CHECK(re2.is_valid());

	CHECK(re1.get_pattern() == re2.get_pattern());
	CHECK(re1.get_group_count() == re2.get_group_count());

	names = re2.get_names();
	CHECK(names.size() == 1);
	CHECK(names[0] == "vowel");
}

TEST_CASE("[RegEx] Clearing") {
	RegEx re("Godot");
	REQUIRE(re.is_valid());
	re.clear();
	CHECK(re.is_valid() == false);
}

TEST_CASE("[RegEx] Searching") {
	const String s = "Searching";
	const String vowels = "[aeiou]{1,2}";
	const String numerics = "\\d";

	RegEx re(vowels);
	REQUIRE(re.is_valid());

	Ref<RegExMatch> match = re.search(s);
	REQUIRE(match != nullptr);
	CHECK(match->get_string(0) == "ea");

	match = re.search(s, 2, 4);
	REQUIRE(match != nullptr);
	CHECK(match->get_string(0) == "a");

	const Array all_results = re.search_all(s);
	CHECK(all_results.size() == 2);
	match = all_results[0];
	REQUIRE(match != nullptr);
	CHECK(match->get_string(0) == "ea");
	match = all_results[1];
	REQUIRE(match != nullptr);
	CHECK(match->get_string(0) == "i");

	CHECK(re.compile(numerics) == OK);
	CHECK(re.is_valid());
	CHECK(re.search(s) == nullptr);
	CHECK(re.search_all(s).size() == 0);
}

TEST_CASE("[RegEx] Substitution") {
	String s = "Double all the vowels.";

	RegEx re("(?<vowel>[aeiou])");
	REQUIRE(re.is_valid());
	CHECK(re.sub(s, "$0$vowel", true) == "Doouublee aall thee vooweels.");
}

TEST_CASE("[RegEx] Uninitialized use") {
	const String s = "Godot";

	RegEx re;
	ERR_PRINT_OFF;
	CHECK(re.search(s) == nullptr);
	CHECK(re.search_all(s).size() == 0);
	CHECK(re.sub(s, "") == "");
	CHECK(re.get_group_count() == 0);
	CHECK(re.get_names().size() == 0);
	ERR_PRINT_ON
}

TEST_CASE("[RegEx] Empty Pattern") {
	const String s = "Godot";

	RegEx re;
	CHECK(re.compile("") == OK);
	CHECK(re.is_valid());
}

TEST_CASE("[RegEx] Invalid offset") {
	const String s = "Godot";

	RegEx re("o");
	REQUIRE(re.is_valid());
	CHECK(re.search(s, -1) == nullptr);
	CHECK(re.search_all(s, -1).size() == 0);
	CHECK(re.sub(s, "", true, -1) == "");
}

TEST_CASE("[RegEx] Invalid end position") {
	const String s = "Godot";

	RegEx re("o");
	REQUIRE(re.is_valid());
	Ref<RegExMatch> match = re.search(s, 0, 10);
	CHECK(match->get_string(0) == "o");

	const Array all_results = re.search_all(s, 0, 10);
	CHECK(all_results.size() == 2);
	match = all_results[0];
	REQUIRE(match != nullptr);
	CHECK(match->get_string(0) == String("o"));
	match = all_results[1];
	REQUIRE(match != nullptr);
	CHECK(match->get_string(0) == String("o"));

	CHECK(re.sub(s, "", true, 0, 10) == "Gdt");
}
} // namespace TestRegEx

#endif // TEST_REGEX_H
