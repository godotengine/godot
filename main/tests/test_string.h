/*************************************************************************/
/*  test_string.h                                                        */
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

#ifndef TEST_STRING_H
#define TEST_STRING_H

#include <thirdparty/doctest/doctest.h>

#include "core/io/ip_address.h"
#include "core/os/main_loop.h"
#include "core/os/os.h"
#include "core/ustring.h"

#include "modules/regex/regex.h"

#include <wchar.h>
//#include "core/math/math_funcs.h"
#include <stdio.h>

namespace TestString {

TEST_CASE("[String] Assign from cstr") {

	OS::get_singleton()->print("\n\nTest 1: Assign from cstr\n");

	String s = "Hello";

	OS::get_singleton()->print("\tExpected: Hello\n");
	OS::get_singleton()->print("\tResulted: %ls\n", s.c_str());

	CHECK(wcscmp(s.c_str(), L"Hello") == 0);
}

TEST_CASE("[String] Assign from string (operator=)") {
	OS::get_singleton()->print("\n\nTest 2: Assign from string (operator=)\n");

	String s = "Dolly";
	const String &t = s;

	OS::get_singleton()->print("\tExpected: Dolly\n");
	OS::get_singleton()->print("\tResulted: %ls\n", t.c_str());

	CHECK(wcscmp(t.c_str(), L"Dolly") == 0);
}

TEST_CASE("[String] Assign from c-string (copycon)") {
	OS::get_singleton()->print("\n\nTest 3: Assign from c-string (copycon)\n");

	String s("Sheep");
	const String &t(s);

	OS::get_singleton()->print("\tExpected: Sheep\n");
	OS::get_singleton()->print("\tResulted: %ls\n", t.c_str());

	CHECK(wcscmp(t.c_str(), L"Sheep") == 0);
}

TEST_CASE("[String] Assign from c-widechar (operator=)") {
	OS::get_singleton()->print("\n\nTest 4: Assign from c-widechar (operator=)\n");

	String s(L"Give me");

	OS::get_singleton()->print("\tExpected: Give me\n");
	OS::get_singleton()->print("\tResulted: %ls\n", s.c_str());

	CHECK(wcscmp(s.c_str(), L"Give me") == 0);
}

TEST_CASE("[String] Assign from c-widechar (copycon)") {
	OS::get_singleton()->print("\n\nTest 5: Assign from c-widechar (copycon)\n");

	String s(L"Wool");

	OS::get_singleton()->print("\tExpected: Wool\n");
	OS::get_singleton()->print("\tResulted: %ls\n", s.c_str());

	CHECK(wcscmp(s.c_str(), L"Wool") == 0);
}

TEST_CASE("[String] Comparisons (equal)") {
	OS::get_singleton()->print("\n\nTest 6: comparisons (equal)\n");

	String s = "Test Compare";

	OS::get_singleton()->print("\tComparing to \"Test Compare\"\n");

	CHECK(s == "Test Compare");
	CHECK(s == L"Test Compare");
	CHECK(s == String("Test Compare"));
}

TEST_CASE("[String] Comparisons (not equal)") {
	OS::get_singleton()->print("\n\nTest 7: comparisons (unequal)\n");

	String s = "Test Compare";

	OS::get_singleton()->print("\tComparing to \"Test Compare\"\n");

	CHECK(s != "Peanut");
	CHECK(s != L"Coconut");
	CHECK(s != String("Butter"));
}

TEST_CASE("[String] Comparisons (operator <)") {
	OS::get_singleton()->print("\n\nTest 8: comparisons (operator<)\n");

	String s = "Bees";

	OS::get_singleton()->print("\tComparing to \"Bees\"\n");

	CHECK(s < "Elephant");
	CHECK(!(s < L"Amber"));
	CHECK(!(s < String("Beatrix")));
}

TEST_CASE("[String] Concatenation") {
	OS::get_singleton()->print("\n\nTest 9: Concatenation\n");

	String s;

	s += "Have";
	s += ' ';
	s += 'a';
	s += String(" ");
	s = s + L"Nice";
	s = s + " ";
	s = s + String("Day");

	OS::get_singleton()->print("\tComparing to \"Have a Nice Day\"\n");

	CHECK(s == "Have a Nice Day");
}

TEST_CASE("[String] Testing size and length of string") {
	// todo: expand this test to do more tests on size() as it is complicated under the hood.
	CHECK(String("Mellon").size() == 7);
	CHECK(String("Mellon1").size() == 8);

	// length works fine and is easier to test
	CHECK(String("Mellon").length() == 6);
	CHECK(String("Mellon1").length() == 7);
	CHECK(String("Mellon2").length() == 7);
	CHECK(String("Mellon3").length() == 7);
}

TEST_CASE("[String] Testing for empty string") {
	CHECK(!String("Mellon").empty());
	// do this more than once, to check for string corruption
	CHECK(String("").empty());
	CHECK(String("").empty());
	CHECK(String("").empty());
}

TEST_CASE("[String] Operator []") {
	OS::get_singleton()->print("\n\nTest 11: Operator[]\n");
	String a = "Kugar Sane";
	a[0] = 'S';
	a[6] = 'C';
	CHECK(a == "Sugar Cane");
	CHECK(a[1] == 'u');
}

TEST_CASE("[String] Case function test") {
	OS::get_singleton()->print("\n\nTest 12: case functions\n");

	String a = "MoMoNgA";

	CHECK(a.to_upper() == "MOMONGA");
	CHECK(a.nocasecmp_to("momonga") == 0);
}

TEST_CASE("[String] UTF8") {
	OS::get_singleton()->print("\n\nTest 13: UTF8\n");

	/* how can i embed UTF in here? */

	static const CharType ustr[] = { 0x304A, 0x360F, 0x3088, 0x3046, 0 };
	//static const wchar_t ustr[] = { 'P', 0xCE, 'p',0xD3, 0 };
	String s = ustr;

	OS::get_singleton()->print("\tUnicode: %ls\n", ustr);

	s.parse_utf8(s.utf8().get_data());

	OS::get_singleton()->print("\tConvert/Parse UTF8: %ls\n", s.c_str());

	CHECK(s == ustr);
}

TEST_CASE("[String] ASCII") {
	OS::get_singleton()->print("\n\nTest 14: ASCII\n");

	String s = L"Primero Leche";
	OS::get_singleton()->print("\tAscii: %s\n", s.ascii().get_data());

	String t = s.ascii().get_data();
	CHECK(s == t);
}

TEST_CASE("[String] Substr") {
	OS::get_singleton()->print("\n\nTest 15: substr\n");

	String s = "Killer Baby";
	OS::get_singleton()->print("\tsubstr(3,4) of \"%ls\" is \"%ls\"\n", s.c_str(), s.substr(3, 4).c_str());

	CHECK(s.substr(3, 4) == "ler ");
}

TEST_CASE("[string] Find") {
	OS::get_singleton()->print("\n\nTest 16: find\n");

	String s = "Pretty Woman";
	OS::get_singleton()->print("\tString: %ls\n", s.c_str());
	OS::get_singleton()->print("\t\"tty\" is at %i pos.\n", s.find("tty"));
	OS::get_singleton()->print("\t\"Revenge of the Monster Truck\" is at %i pos.\n", s.find("Revenge of the Monster Truck"));

	CHECK(s.find("tty") == 3);
	CHECK(s.find("Revenge of the Monster Truck") == -1);
}

TEST_CASE("[String] find no case") {
	OS::get_singleton()->print("\n\nTest 17: find no case\n");

	String s = "Pretty Whale";
	OS::get_singleton()->print("\tString: %ls\n", s.c_str());
	OS::get_singleton()->print("\t\"WHA\" is at %i pos.\n", s.findn("WHA"));
	OS::get_singleton()->print("\t\"Revenge of the Monster SawFish\" is at %i pos.\n", s.findn("Revenge of the Monster Truck"));

	CHECK(s.findn("WHA") == 7);
	CHECK(s.findn("Revenge of the Monster SawFish") == -1);
}

TEST_CASE("[String] Find and replace") {
	OS::get_singleton()->print("\n\nTest 19: Search & replace\n");

	String s = "Happy Birthday, Anna!";
	OS::get_singleton()->print("\tString: %ls\n", s.c_str());

	s = s.replace("Birthday", "Halloween");
	OS::get_singleton()->print("\tReplaced Birthday/Halloween: %ls.\n", s.c_str());

	CHECK(s == "Happy Halloween, Anna!");
}

TEST_CASE("[String] Insertion") {
	OS::get_singleton()->print("\n\nTest 20: Insertion\n");

	String s = "Who is Frederic?";

	OS::get_singleton()->print("\tString: %ls\n", s.c_str());
	s = s.insert(s.find("?"), " Chopin");
	OS::get_singleton()->print("\tInserted Chopin: %ls.\n", s.c_str());

	CHECK(s == "Who is Frederic Chopin?");
}

TEST_CASE("[String] Number to string") {
	OS::get_singleton()->print("\n\nTest 21: Number -> String\n");

	OS::get_singleton()->print("\tPi is %f\n", 33.141593);
	OS::get_singleton()->print("\tPi String is %ls\n", String::num(3.141593).c_str());

	CHECK(String::num(3.141593) == "3.141593");
}

TEST_CASE("[String] String to integer") {
	OS::get_singleton()->print("\n\nTest 22: String -> Int\n");

	static const char *nums[4] = { "1237461283", "- 22", "0", " - 1123412" };
	static const int num[4] = { 1237461283, -22, 0, -1123412 };

	for (int i = 0; i < 4; i++) {
		OS::get_singleton()->print("\tString: \"%s\" as Int is %i\n", nums[i], String(nums[i]).to_int());

		CHECK(String(nums[i]).to_int() == num[i]);
	}
}

TEST_CASE("[String] String to float") {
	OS::get_singleton()->print("\n\nTest 23: String -> Float\n");

	static const char *nums[4] = { "-12348298412.2", "0.05", "2.0002", " -0.0001" };
	static const double num[4] = { -12348298412.2, 0.05, 2.0002, -0.0001 };

	for (int i = 0; i < 4; i++) {
		OS::get_singleton()->print("\tString: \"%s\" as Float is %f\n", nums[i], String(nums[i]).to_double());

		CHECK(!(ABS(String(nums[i]).to_double() - num[i]) > 0.00001));
	}
}

TEST_CASE("[String] Slicing") {
	OS::get_singleton()->print("\n\nTest 24: Slicing\n");

	String s = "Mars,Jupiter,Saturn,Uranus";

	const char *slices[4] = { "Mars", "Jupiter", "Saturn", "Uranus" };

	OS::get_singleton()->print("\tSlicing \"%ls\" by \"%s\"..\n", s.c_str(), ",");

	for (int i = 0; i < s.get_slice_count(","); i++) {

		OS::get_singleton()->print("\t\t%i- %ls\n", i + 1, s.get_slice(",", i).c_str());

		CHECK(s.get_slice(",", i) == slices[i]);
	}
}

TEST_CASE("[String] Erasing") {
	OS::get_singleton()->print("\n\nTest 25: Erasing\n");

	String s = "Josephine is such a cute girl!";

	OS::get_singleton()->print("\tString: %ls\n", s.c_str());
	OS::get_singleton()->print("\tRemoving \"cute\"\n");

	s.erase(s.find("cute "), String("cute ").length());
	OS::get_singleton()->print("\tResult: %ls\n", s.c_str());

	CHECK(s == "Josephine is such a girl!");
}

TEST_CASE("[String] Regex substitution") {
	OS::get_singleton()->print("\n\nTest 26: RegEx substitution\n");

	String s = "Double all the vowels.";

	OS::get_singleton()->print("\tString: %ls\n", s.c_str());
	OS::get_singleton()->print("\tRepeating instances of 'aeiou' once\n");

	RegEx re("(?<vowel>[aeiou])");
	s = re.sub(s, "$0$vowel", true);

	OS::get_singleton()->print("\tResult: %ls\n", s.c_str());

	CHECK(s == "Doouublee aall thee vooweels.");
}

struct test_27_data {
	char const *data;
	char const *begin;
	bool expected;
};

TEST_CASE("[String] Begins with") {
	OS::get_singleton()->print("\n\nTest 27: begins_with\n");
	test_27_data tc[] = {
		{ "res://foobar", "res://", true },
		{ "res", "res://", false },
		{ "abc", "abc", true }
	};
	size_t count = sizeof(tc) / sizeof(tc[0]);
	bool state = true;
	for (size_t i = 0; state && i < count; ++i) {
		String s = tc[i].data;
		state = s.begins_with(tc[i].begin) == tc[i].expected;
		if (state) {
			String sb = tc[i].begin;
			state = s.begins_with(sb) == tc[i].expected;
		}
		CHECK(state);
		if (!state) {
			OS::get_singleton()->print("\n\t Failure on:\n\t\tstring: %s\n\t\tbegin: %s\n\t\texpected: %s\n", tc[i].data, tc[i].begin, tc[i].expected ? "true" : "false");
			break;
		}
	};
	CHECK(state);
}

TEST_CASE("[String] sprintf") {

	OS::get_singleton()->print("\n\nTest 28: sprintf\n");

	bool success, state = true;
	char output_format[] = "\tTest:\t%ls => %ls (%s)\n";
	String format, output;
	Array args;
	bool error;

	// %%
	format = "fish %% frog";
	args.clear();
	output = format.sprintf(args, &error);
	success = (output == String("fish % frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	//////// INTS

	// Int
	format = "fish %d frog";
	args.clear();
	args.push_back(5);
	output = format.sprintf(args, &error);
	success = (output == String("fish 5 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Int left padded with zeroes.
	format = "fish %05d frog";
	args.clear();
	args.push_back(5);
	output = format.sprintf(args, &error);
	success = (output == String("fish 00005 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Int left padded with spaces.
	format = "fish %5d frog";
	args.clear();
	args.push_back(5);
	output = format.sprintf(args, &error);
	success = (output == String("fish     5 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Int right padded with spaces.
	format = "fish %-5d frog";
	args.clear();
	args.push_back(5);
	output = format.sprintf(args, &error);
	success = (output == String("fish 5     frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Int with sign (positive).
	format = "fish %+d frog";
	args.clear();
	args.push_back(5);
	output = format.sprintf(args, &error);
	success = (output == String("fish +5 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Negative int.
	format = "fish %d frog";
	args.clear();
	args.push_back(-5);
	output = format.sprintf(args, &error);
	success = (output == String("fish -5 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Hex (lower)
	format = "fish %x frog";
	args.clear();
	args.push_back(45);
	output = format.sprintf(args, &error);
	success = (output == String("fish 2d frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Hex (upper)
	format = "fish %X frog";
	args.clear();
	args.push_back(45);
	output = format.sprintf(args, &error);
	success = (output == String("fish 2D frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Octal
	format = "fish %o frog";
	args.clear();
	args.push_back(99);
	output = format.sprintf(args, &error);
	success = (output == String("fish 143 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	////// REALS

	// Real
	format = "fish %f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	success = (output == String("fish 99.990000 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Real left-padded
	format = "fish %11f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	success = (output == String("fish   99.990000 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Real right-padded
	format = "fish %-11f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	success = (output == String("fish 99.990000   frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Real given int.
	format = "fish %f frog";
	args.clear();
	args.push_back(99);
	output = format.sprintf(args, &error);
	success = (output == String("fish 99.000000 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Real with sign (positive).
	format = "fish %+f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	success = (output == String("fish +99.990000 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Real with 1 decimals.
	format = "fish %.1f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	success = (output == String("fish 100.0 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Real with 12 decimals.
	format = "fish %.12f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	success = (output == String("fish 99.990000000000 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Real with no decimals.
	format = "fish %.f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	success = (output == String("fish 100 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	/////// Strings.

	// String
	format = "fish %s frog";
	args.clear();
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	success = (output == String("fish cheese frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// String left-padded
	format = "fish %10s frog";
	args.clear();
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	success = (output == String("fish     cheese frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// String right-padded
	format = "fish %-10s frog";
	args.clear();
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	success = (output == String("fish cheese     frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	///// Characters

	// Character as string.
	format = "fish %c frog";
	args.clear();
	args.push_back("A");
	output = format.sprintf(args, &error);
	success = (output == String("fish A frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Character as int.
	format = "fish %c frog";
	args.clear();
	args.push_back(65);
	output = format.sprintf(args, &error);
	success = (output == String("fish A frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	///// Dynamic width

	// String dynamic width
	format = "fish %*s frog";
	args.clear();
	args.push_back(10);
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	success = (output == String("fish     cheese frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Int dynamic width
	format = "fish %*d frog";
	args.clear();
	args.push_back(10);
	args.push_back(99);
	output = format.sprintf(args, &error);
	success = (output == String("fish         99 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Float dynamic width
	format = "fish %*.*f frog";
	args.clear();
	args.push_back(10);
	args.push_back(3);
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	success = (output == String("fish     99.990 frog") && !error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	///// Errors

	// More formats than arguments.
	format = "fish %s %s frog";
	args.clear();
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	success = (output == "not enough arguments for format string" && error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// More arguments than formats.
	format = "fish %s frog";
	args.clear();
	args.push_back("hello");
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	success = (output == "not all arguments converted during string formatting" && error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Incomplete format.
	format = "fish %10";
	args.clear();
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	success = (output == "incomplete format" && error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Bad character in format string
	format = "fish %&f frog";
	args.clear();
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	success = (output == "unsupported format character" && error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Too many decimals.
	format = "fish %2.2.2f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	success = (output == "too many decimal points in format" && error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// * not a number
	format = "fish %*f frog";
	args.clear();
	args.push_back("cheese");
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	success = (output == "* wants number" && error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Character too long.
	format = "fish %c frog";
	args.clear();
	args.push_back("sc");
	output = format.sprintf(args, &error);
	success = (output == "%c requires number or single-character string" && error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	// Character bad type.
	format = "fish %c frog";
	args.clear();
	args.push_back(Array());
	output = format.sprintf(args, &error);
	success = (output == "%c requires number or single-character string" && error);
	OS::get_singleton()->print(output_format, format.c_str(), output.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	CHECK(state);
}

TEST_CASE("[String] IPVX address to string") {

	bool state = true;

	IP_Address ip0("2001:0db8:85a3:0000:0000:8a2e:0370:7334");
	OS::get_singleton()->print("ip0 is %ls\n", String(ip0).c_str());

	IP_Address ip(0x0123, 0x4567, 0x89ab, 0xcdef, true);
	OS::get_singleton()->print("ip6 is %ls\n", String(ip).c_str());

	IP_Address ip2("fe80::52e5:49ff:fe93:1baf");
	OS::get_singleton()->print("ip6 is %ls\n", String(ip2).c_str());

	IP_Address ip3("::ffff:192.168.0.1");
	OS::get_singleton()->print("ip6 is %ls\n", String(ip3).c_str());

	String ip4 = "192.168.0.1";
	bool success = ip4.is_valid_ip_address();
	OS::get_singleton()->print("Is valid ipv4: %ls, %s\n", ip4.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	ip4 = "192.368.0.1";
	success = (!ip4.is_valid_ip_address());
	OS::get_singleton()->print("Is invalid ipv4: %ls, %s\n", ip4.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	String ip6 = "2001:0db8:85a3:0000:0000:8a2e:0370:7334";
	success = ip6.is_valid_ip_address();
	OS::get_singleton()->print("Is valid ipv6: %ls, %s\n", ip6.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	ip6 = "2001:0db8:85j3:0000:0000:8a2e:0370:7334";
	success = (!ip6.is_valid_ip_address());
	OS::get_singleton()->print("Is invalid ipv6: %ls, %s\n", ip6.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	ip6 = "2001:0db8:85f345:0000:0000:8a2e:0370:7334";
	success = (!ip6.is_valid_ip_address());
	OS::get_singleton()->print("Is invalid ipv6: %ls, %s\n", ip6.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	ip6 = "2001:0db8::0:8a2e:370:7334";
	success = (ip6.is_valid_ip_address());
	OS::get_singleton()->print("Is valid ipv6: %ls, %s\n", ip6.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	ip6 = "::ffff:192.168.0.1";
	success = (ip6.is_valid_ip_address());
	OS::get_singleton()->print("Is valid ipv6: %ls, %s\n", ip6.c_str(), success ? "OK" : "FAIL");
	state = state && success;

	CHECK(state);
}

TEST_CASE("[String] Capitalize against many strings") {
	bool state = true;
	bool success = true;
	String input = "bytes2var";
	String output = "Bytes 2 Var";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls: %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "linear2db";
	output = "Linear 2 Db";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls: %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "vector3";
	output = "Vector 3";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls: %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "sha256";
	output = "Sha 256";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls: %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "2db";
	output = "2 Db";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls: %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "PascalCase";
	output = "Pascal Case";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls: %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "PascalPascalCase";
	output = "Pascal Pascal Case";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls: %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "snake_case";
	output = "Snake Case";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls: %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "snake_snake_case";
	output = "Snake Snake Case";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls: %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "sha256sum";
	output = "Sha 256 Sum";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls: %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "cat2dog";
	output = "Cat 2 Dog";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls: %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "function(name)";
	output = "Function(name)";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls (existing incorrect behavior): %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "snake_case_function(snake_case_arg)";
	output = "Snake Case Function(snake Case Arg)";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls (existing incorrect behavior): %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	input = "snake_case_function( snake_case_arg )";
	output = "Snake Case Function( Snake Case Arg )";
	success = (input.capitalize() == output);
	state = state && success;
	OS::get_singleton()->print("Capitalize %ls: %ls, %s\n", input.c_str(), output.c_str(), success ? "OK" : "FAIL");

	CHECK(state);
}

TEST_CASE("[String] Checking string is empty when it should be") {
	bool state = true;
	bool success;

	String a = "";
	success = a[0] == 0;
	OS::get_singleton()->print("Is 0 String[0]:, %s\n", success ? "OK" : "FAIL");
	if (!success) state = false;

	String b = "Godot";
	success = b[b.size()] == 0;
	OS::get_singleton()->print("Is 0 String[size()]:, %s\n", success ? "OK" : "FAIL");
	if (!success) state = false;

	const String c = "";
	success = c[0] == 0;
	OS::get_singleton()->print("Is 0 const String[0]:, %s\n", success ? "OK" : "FAIL");
	if (!success) state = false;

	const String d = "Godot";
	success = d[d.size()] == 0;
	OS::get_singleton()->print("Is 0 const String[size()]:, %s\n", success ? "OK" : "FAIL");
	if (!success) state = false;

	CHECK(state);
}

TEST_CASE("[String] lstrip and rstrip") {
#define STRIP_TEST(x)                                            \
	{                                                            \
		bool success = x;                                        \
		state = state && success;                                \
		if (!success) {                                          \
			OS::get_singleton()->print("\tfailed at: %s\n", #x); \
		}                                                        \
	}

	OS::get_singleton()->print("\n\nTest 32: lstrip and rstrip\n");
	bool state = true;

	// strip none
	STRIP_TEST(String("abc").lstrip("") == "abc");
	STRIP_TEST(String("abc").rstrip("") == "abc");
	// strip one
	STRIP_TEST(String("abc").lstrip("a") == "bc");
	STRIP_TEST(String("abc").rstrip("c") == "ab");
	// strip lots
	STRIP_TEST(String("bababbababccc").lstrip("ab") == "ccc");
	STRIP_TEST(String("aaabcbcbcbbcbbc").rstrip("cb") == "aaa");
	// strip empty string
	STRIP_TEST(String("").lstrip("") == "");
	STRIP_TEST(String("").rstrip("") == "");
	// strip to empty string
	STRIP_TEST(String("abcabcabc").lstrip("bca") == "");
	STRIP_TEST(String("abcabcabc").rstrip("bca") == "");
	// don't strip wrong end
	STRIP_TEST(String("abc").lstrip("c") == "abc");
	STRIP_TEST(String("abca").lstrip("a") == "bca");
	STRIP_TEST(String("abc").rstrip("a") == "abc");
	STRIP_TEST(String("abca").rstrip("a") == "abc");
	// in utf-8 "¿" (\u00bf) has the same first byte as "µ" (\u00b5)
	// and the same second as "ÿ" (\u00ff)
	STRIP_TEST(String::utf8("¿").lstrip(String::utf8("µÿ")) == String::utf8("¿"));
	STRIP_TEST(String::utf8("¿").rstrip(String::utf8("µÿ")) == String::utf8("¿"));
	STRIP_TEST(String::utf8("µ¿ÿ").lstrip(String::utf8("µÿ")) == String::utf8("¿ÿ"));
	STRIP_TEST(String::utf8("µ¿ÿ").rstrip(String::utf8("µÿ")) == String::utf8("µ¿"));

	// the above tests repeated with additional superfluous strip chars

	// strip none
	STRIP_TEST(String("abc").lstrip("qwjkl") == "abc");
	STRIP_TEST(String("abc").rstrip("qwjkl") == "abc");
	// strip one
	STRIP_TEST(String("abc").lstrip("qwajkl") == "bc");
	STRIP_TEST(String("abc").rstrip("qwcjkl") == "ab");
	// strip lots
	STRIP_TEST(String("bababbababccc").lstrip("qwabjkl") == "ccc");
	STRIP_TEST(String("aaabcbcbcbbcbbc").rstrip("qwcbjkl") == "aaa");
	// strip empty string
	STRIP_TEST(String("").lstrip("qwjkl") == "");
	STRIP_TEST(String("").rstrip("qwjkl") == "");
	// strip to empty string
	STRIP_TEST(String("abcabcabc").lstrip("qwbcajkl") == "");
	STRIP_TEST(String("abcabcabc").rstrip("qwbcajkl") == "");
	// don't strip wrong end
	STRIP_TEST(String("abc").lstrip("qwcjkl") == "abc");
	STRIP_TEST(String("abca").lstrip("qwajkl") == "bca");
	STRIP_TEST(String("abc").rstrip("qwajkl") == "abc");
	STRIP_TEST(String("abca").rstrip("qwajkl") == "abc");
	// in utf-8 "¿" (\u00bf) has the same first byte as "µ" (\u00b5)
	// and the same second as "ÿ" (\u00ff)
	STRIP_TEST(String::utf8("¿").lstrip(String::utf8("qwaµÿjkl")) == String::utf8("¿"));
	STRIP_TEST(String::utf8("¿").rstrip(String::utf8("qwaµÿjkl")) == String::utf8("¿"));
	STRIP_TEST(String::utf8("µ¿ÿ").lstrip(String::utf8("qwaµÿjkl")) == String::utf8("¿ÿ"));
	STRIP_TEST(String::utf8("µ¿ÿ").rstrip(String::utf8("qwaµÿjkl")) == String::utf8("µ¿"));

	CHECK(state);

#undef STRIP_TEST
}

TEST_CASE("[String] ensuring empty string into parse_utf8 passes empty string") {
	OS::get_singleton()->print("\n\nTest 33: parse_utf8(null, -1)\n");

	String empty;
	CHECK(empty.parse_utf8(NULL, -1));
}

TEST_CASE("[String] Cyrillic to_lower()") {
	OS::get_singleton()->print("\n\nTest 34: Cyrillic to_lower()\n");

	String upper = String::utf8("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ");
	String lower = String::utf8("абвгдеёжзийклмнопрстуфхцчшщъыьэюя");

	String test = upper.to_lower();

	bool state = test == lower;

	CHECK(state);
}

TEST_CASE("[String] Count and countn functionality") {
#define COUNT_TEST(x)                                            \
	{                                                            \
		bool success = x;                                        \
		state = state && success;                                \
		if (!success) {                                          \
			OS::get_singleton()->print("\tfailed at: %s\n", #x); \
		}                                                        \
	}

	OS::get_singleton()->print("\n\nTest 35: count and countn function\n");
	bool state = true;

	COUNT_TEST(String("").count("Test") == 0);
	COUNT_TEST(String("Test").count("") == 0);
	COUNT_TEST(String("Test").count("test") == 0);
	COUNT_TEST(String("Test").count("TEST") == 0);
	COUNT_TEST(String("TEST").count("TEST") == 1);
	COUNT_TEST(String("Test").count("Test") == 1);
	COUNT_TEST(String("aTest").count("Test") == 1);
	COUNT_TEST(String("Testa").count("Test") == 1);
	COUNT_TEST(String("TestTestTest").count("Test") == 3);
	COUNT_TEST(String("TestTestTest").count("TestTest") == 1);
	COUNT_TEST(String("TestGodotTestGodotTestGodot").count("Test") == 3);

	COUNT_TEST(String("TestTestTestTest").count("Test", 4, 8) == 1);
	COUNT_TEST(String("TestTestTestTest").count("Test", 4, 12) == 2);
	COUNT_TEST(String("TestTestTestTest").count("Test", 4, 16) == 3);
	COUNT_TEST(String("TestTestTestTest").count("Test", 4) == 3);

	COUNT_TEST(String("Test").countn("test") == 1);
	COUNT_TEST(String("Test").countn("TEST") == 1);
	COUNT_TEST(String("testTest-Testatest").countn("tEst") == 4);
	COUNT_TEST(String("testTest-TeStatest").countn("tEsT", 4, 16) == 2);

	CHECK(state);
}
} // namespace TestString

#endif // TEST_STRING_H
