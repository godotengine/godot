/**************************************************************************/
/*  test_regex.h                                                          */
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

#ifndef TEST_REGEX_H
#define TEST_REGEX_H

#include "../regex.h"

#include "core/string/ustring.h"

#include "tests/test_macros.h"

namespace TestRegEx {

TEST_CASE("[RegEx] Initialization") {
	const String pattern = "(?<vowel>[aeiou])";

	RegEx re1(pattern);
	CHECK(re1.is_valid());
	CHECK(re1.get_pattern() == pattern);
	CHECK(re1.get_group_count() == 1);

	PackedStringArray names = re1.get_names();
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

	match = re.search(s, 1, 2);
	REQUIRE(match != nullptr);
	CHECK(match->get_string(0) == "e");
	match = re.search(s, 2, 4);
	REQUIRE(match != nullptr);
	CHECK(match->get_string(0) == "a");
	match = re.search(s, 3, 5);
	CHECK(match == nullptr);
	match = re.search(s, 6, 2);
	CHECK(match == nullptr);

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
	const String s1 = "Double all the vowels.";

	RegEx re1("(?<vowel>[aeiou])");
	REQUIRE(re1.is_valid());
	CHECK(re1.sub(s1, "$0$vowel", true) == "Doouublee aall thee vooweels.");

	const String s2 = "Substitution with group.";

	RegEx re2("Substitution (.+)");
	REQUIRE(re2.is_valid());
	CHECK(re2.sub(s2, "Test ${1}") == "Test with group.");

	const String s3 = "Useless substitution";

	RegEx re3("Anything");
	REQUIRE(re3.is_valid());
	CHECK(re3.sub(s3, "Something") == "Useless substitution");

	const String s4 = "acacac";

	RegEx re4("(a)(b){0}(c)");
	REQUIRE(re4.is_valid());
	CHECK(re4.sub(s4, "${1}.${3}.", true) == "a.c.a.c.a.c.");
}

TEST_CASE("[RegEx] Substitution with empty input and/or replacement") {
	const String s1 = "";
	const String s2 = "gogogo";

	RegEx re1("");
	REQUIRE(re1.is_valid());
	CHECK(re1.sub(s1, "") == "");
	CHECK(re1.sub(s1, "a") == "a");
	CHECK(re1.sub(s2, "") == "gogogo");

	RegEx re2("go");
	REQUIRE(re2.is_valid());
	CHECK(re2.sub(s2, "") == "gogo");
	CHECK(re2.sub(s2, "", true) == "");
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

TEST_CASE("[RegEx] Get match string list") {
	const String s = "Godot Engine";

	RegEx re("(Go)(dot)");
	Ref<RegExMatch> match = re.search(s);
	REQUIRE(match != nullptr);
	PackedStringArray result;
	result.append("Godot");
	result.append("Go");
	result.append("dot");
	CHECK(match->get_strings() == result);
}

TEST_CASE("[RegEx] Match start and end positions") {
	const String s = "Whole pattern";

	RegEx re1("pattern");
	REQUIRE(re1.is_valid());
	Ref<RegExMatch> match = re1.search(s);
	REQUIRE(match != nullptr);
	CHECK(match->get_start(0) == 6);
	CHECK(match->get_end(0) == 13);

	RegEx re2("(?<vowel>[aeiou])");
	REQUIRE(re2.is_valid());
	match = re2.search(s);
	REQUIRE(match != nullptr);
	CHECK(match->get_start("vowel") == 2);
	CHECK(match->get_end("vowel") == 3);
}
} // namespace TestRegEx

#endif // TEST_REGEX_H
