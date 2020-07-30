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

#include <inttypes.h>
#include <stdio.h>
#include <wchar.h>

#include "core/io/ip_address.h"
#include "core/os/main_loop.h"
#include "core/os/os.h"
#include "core/ustring.h"

#ifdef MODULE_REGEX_ENABLED
#include "modules/regex/regex.h"
#endif

#include "tests/test_macros.h"

namespace TestString {

TEST_CASE("[String] Assign from cstr") {
	String s = "Hello";
	CHECK(wcscmp(s.c_str(), L"Hello") == 0);
}

TEST_CASE("[String] Assign from string (operator=)") {
	String s = "Dolly";
	const String &t = s;
	CHECK(wcscmp(t.c_str(), L"Dolly") == 0);
}

TEST_CASE("[String] Assign from c-string (copycon)") {
	String s("Sheep");
	const String &t(s);
	CHECK(wcscmp(t.c_str(), L"Sheep") == 0);
}

TEST_CASE("[String] Assign from c-widechar (operator=)") {
	String s(L"Give me");
	CHECK(wcscmp(s.c_str(), L"Give me") == 0);
}

TEST_CASE("[String] Assign from c-widechar (copycon)") {
	String s(L"Wool");
	CHECK(wcscmp(s.c_str(), L"Wool") == 0);
}

TEST_CASE("[String] Comparisons (equal)") {
	String s = "Test Compare";
	CHECK(s == "Test Compare");
	CHECK(s == L"Test Compare");
	CHECK(s == String("Test Compare"));
}

TEST_CASE("[String] Comparisons (not equal)") {
	String s = "Test Compare";
	CHECK(s != "Peanut");
	CHECK(s != L"Coconut");
	CHECK(s != String("Butter"));
}

TEST_CASE("[String] Comparisons (operator <)") {
	String s = "Bees";
	CHECK(s < "Elephant");
	CHECK(!(s < L"Amber"));
	CHECK(!(s < String("Beatrix")));
}

TEST_CASE("[String] Concatenation") {
	String s;

	s += "Have";
	s += ' ';
	s += 'a';
	s += String(" ");
	s = s + L"Nice";
	s = s + " ";
	s = s + String("Day");

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
	String a = "Kugar Sane";
	a[0] = 'S';
	a[6] = 'C';
	CHECK(a == "Sugar Cane");
	CHECK(a[1] == 'u');
}

TEST_CASE("[String] Case function test") {
	String a = "MoMoNgA";

	CHECK(a.to_upper() == "MOMONGA");
	CHECK(a.nocasecmp_to("momonga") == 0);
}

TEST_CASE("[String] UTF8") {
	/* how can i embed UTF in here? */
	static const CharType ustr[] = { 0x304A, 0x360F, 0x3088, 0x3046, 0 };
	//static const wchar_t ustr[] = { 'P', 0xCE, 'p',0xD3, 0 };
	String s = ustr;
	s.parse_utf8(s.utf8().get_data());
	CHECK(s == ustr);
}

TEST_CASE("[String] ASCII") {
	String s = L"Primero Leche";
	String t = s.ascii().get_data();
	CHECK(s == t);
}

TEST_CASE("[String] Substr") {
	String s = "Killer Baby";
	CHECK(s.substr(3, 4) == "ler ");
}

TEST_CASE("[string] Find") {
	String s = "Pretty Woman";
	s.find("Revenge of the Monster Truck");

	CHECK(s.find("tty") == 3);
	CHECK(s.find("Revenge of the Monster Truck") == -1);
}

TEST_CASE("[String] find no case") {
	String s = "Pretty Whale";
	CHECK(s.findn("WHA") == 7);
	CHECK(s.findn("Revenge of the Monster SawFish") == -1);
}

TEST_CASE("[String] Find and replace") {
	String s = "Happy Birthday, Anna!";
	s = s.replace("Birthday", "Halloween");
	CHECK(s == "Happy Halloween, Anna!");
}

TEST_CASE("[String] Insertion") {
	String s = "Who is Frederic?";
	s = s.insert(s.find("?"), " Chopin");
	CHECK(s == "Who is Frederic Chopin?");
}

TEST_CASE("[String] Number to string") {
	CHECK(String::num(3.141593) == "3.141593");
}

TEST_CASE("[String] String to integer") {
	static const char *nums[4] = { "1237461283", "- 22", "0", " - 1123412" };
	static const int num[4] = { 1237461283, -22, 0, -1123412 };

	for (int i = 0; i < 4; i++) {
		CHECK(String(nums[i]).to_int() == num[i]);
	}
}

TEST_CASE("[String] String to float") {
	static const char *nums[4] = { "-12348298412.2", "0.05", "2.0002", " -0.0001" };
	static const double num[4] = { -12348298412.2, 0.05, 2.0002, -0.0001 };

	for (int i = 0; i < 4; i++) {
		CHECK(!(ABS(String(nums[i]).to_float() - num[i]) > 0.00001));
	}
}

TEST_CASE("[String] Slicing") {
	String s = "Mars,Jupiter,Saturn,Uranus";

	const char *slices[4] = { "Mars", "Jupiter", "Saturn", "Uranus" };
	for (int i = 0; i < s.get_slice_count(","); i++) {
		CHECK(s.get_slice(",", i) == slices[i]);
	}
}

TEST_CASE("[String] Erasing") {
	String s = "Josephine is such a cute girl!";
	s.erase(s.find("cute "), String("cute ").length());
	CHECK(s == "Josephine is such a girl!");
}

#ifdef MODULE_REGEX_ENABLED
TEST_CASE("[String] Regex substitution") {
	String s = "Double all the vowels.";
	RegEx re("(?<vowel>[aeiou])");
	s = re.sub(s, "$0$vowel", true);
	CHECK(s == "Doouublee aall thee vooweels.");
}
#endif

struct test_27_data {
	char const *data;
	char const *begin;
	bool expected;
};

TEST_CASE("[String] Begins with") {
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
			break;
		}
	};
	CHECK(state);
}

TEST_CASE("[String] sprintf") {
	String format, output;
	Array args;
	bool error;

	// %%
	format = "fish %% frog";
	args.clear();
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish % frog"));
	//////// INTS

	// Int
	format = "fish %d frog";
	args.clear();
	args.push_back(5);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish 5 frog"));

	// Int left padded with zeroes.
	format = "fish %05d frog";
	args.clear();
	args.push_back(5);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish 00005 frog"));

	// Int left padded with spaces.
	format = "fish %5d frog";
	args.clear();
	args.push_back(5);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish     5 frog"));

	// Int right padded with spaces.
	format = "fish %-5d frog";
	args.clear();
	args.push_back(5);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish 5     frog"));

	// Int with sign (positive).
	format = "fish %+d frog";
	args.clear();
	args.push_back(5);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish +5 frog"));

	// Negative int.
	format = "fish %d frog";
	args.clear();
	args.push_back(-5);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish -5 frog"));

	// Hex (lower)
	format = "fish %x frog";
	args.clear();
	args.push_back(45);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish 2d frog"));

	// Hex (upper)
	format = "fish %X frog";
	args.clear();
	args.push_back(45);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish 2D frog"));

	// Octal
	format = "fish %o frog";
	args.clear();
	args.push_back(99);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish 143 frog"));

	////// REALS

	// Real
	format = "fish %f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish 99.990000 frog"));

	// Real left-padded
	format = "fish %11f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish   99.990000 frog"));

	// Real right-padded
	format = "fish %-11f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish 99.990000   frog"));

	// Real given int.
	format = "fish %f frog";
	args.clear();
	args.push_back(99);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish 99.000000 frog"));

	// Real with sign (positive).
	format = "fish %+f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish +99.990000 frog"));

	// Real with 1 decimals.
	format = "fish %.1f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish 100.0 frog"));

	// Real with 12 decimals.
	format = "fish %.12f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish 99.990000000000 frog"));

	// Real with no decimals.
	format = "fish %.f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish 100 frog"));

	/////// Strings.

	// String
	format = "fish %s frog";
	args.clear();
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish cheese frog"));

	// String left-padded
	format = "fish %10s frog";
	args.clear();
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish     cheese frog"));

	// String right-padded
	format = "fish %-10s frog";
	args.clear();
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish cheese     frog"));

	///// Characters

	// Character as string.
	format = "fish %c frog";
	args.clear();
	args.push_back("A");
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish A frog"));

	// Character as int.
	format = "fish %c frog";
	args.clear();
	args.push_back(65);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish A frog"));

	///// Dynamic width

	// String dynamic width
	format = "fish %*s frog";
	args.clear();
	args.push_back(10);
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	REQUIRE(output == String("fish     cheese frog"));

	// Int dynamic width
	format = "fish %*d frog";
	args.clear();
	args.push_back(10);
	args.push_back(99);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	REQUIRE(output == String("fish         99 frog"));

	// Float dynamic width
	format = "fish %*.*f frog";
	args.clear();
	args.push_back(10);
	args.push_back(3);
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish     99.990 frog"));

	///// Errors

	// More formats than arguments.
	format = "fish %s %s frog";
	args.clear();
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	REQUIRE(error);
	CHECK(output == "not enough arguments for format string");

	// More arguments than formats.
	format = "fish %s frog";
	args.clear();
	args.push_back("hello");
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	REQUIRE(error);
	CHECK(output == "not all arguments converted during string formatting");

	// Incomplete format.
	format = "fish %10";
	args.clear();
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	REQUIRE(error);
	CHECK(output == "incomplete format");

	// Bad character in format string
	format = "fish %&f frog";
	args.clear();
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	REQUIRE(error);
	CHECK(output == "unsupported format character");

	// Too many decimals.
	format = "fish %2.2.2f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	REQUIRE(error);
	CHECK(output == "too many decimal points in format");

	// * not a number
	format = "fish %*f frog";
	args.clear();
	args.push_back("cheese");
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	REQUIRE(error);
	CHECK(output == "* wants number");

	// Character too long.
	format = "fish %c frog";
	args.clear();
	args.push_back("sc");
	output = format.sprintf(args, &error);
	REQUIRE(error);
	CHECK(output == "%c requires number or single-character string");

	// Character bad type.
	format = "fish %c frog";
	args.clear();
	args.push_back(Array());
	output = format.sprintf(args, &error);
	REQUIRE(error);
	CHECK(output == "%c requires number or single-character string");
}

TEST_CASE("[String] IPVX address to string") {
	IP_Address ip0("2001:0db8:85a3:0000:0000:8a2e:0370:7334");
	IP_Address ip(0x0123, 0x4567, 0x89ab, 0xcdef, true);
	IP_Address ip2("fe80::52e5:49ff:fe93:1baf");
	IP_Address ip3("::ffff:192.168.0.1");
	String ip4 = "192.168.0.1";
	CHECK(ip4.is_valid_ip_address());

	ip4 = "192.368.0.1";
	CHECK(!ip4.is_valid_ip_address());

	String ip6 = "2001:0db8:85a3:0000:0000:8a2e:0370:7334";
	CHECK(ip6.is_valid_ip_address());

	ip6 = "2001:0db8:85j3:0000:0000:8a2e:0370:7334";
	CHECK(!ip6.is_valid_ip_address());

	ip6 = "2001:0db8:85f345:0000:0000:8a2e:0370:7334";
	CHECK(!ip6.is_valid_ip_address());

	ip6 = "2001:0db8::0:8a2e:370:7334";
	CHECK(ip6.is_valid_ip_address());

	ip6 = "::ffff:192.168.0.1";
	CHECK(ip6.is_valid_ip_address());
}

TEST_CASE("[String] Capitalize against many strings") {
	String input = "bytes2var";
	String output = "Bytes 2 Var";
	CHECK(input.capitalize() == output);

	input = "linear2db";
	output = "Linear 2 Db";
	CHECK(input.capitalize() == output);

	input = "vector3";
	output = "Vector 3";
	CHECK(input.capitalize() == output);

	input = "sha256";
	output = "Sha 256";
	CHECK(input.capitalize() == output);

	input = "2db";
	output = "2 Db";
	CHECK(input.capitalize() == output);

	input = "PascalCase";
	output = "Pascal Case";
	CHECK(input.capitalize() == output);

	input = "PascalPascalCase";
	output = "Pascal Pascal Case";
	CHECK(input.capitalize() == output);

	input = "snake_case";
	output = "Snake Case";
	CHECK(input.capitalize() == output);

	input = "snake_snake_case";
	output = "Snake Snake Case";
	CHECK(input.capitalize() == output);

	input = "sha256sum";
	output = "Sha 256 Sum";
	CHECK(input.capitalize() == output);

	input = "cat2dog";
	output = "Cat 2 Dog";
	CHECK(input.capitalize() == output);

	input = "function(name)";
	output = "Function(name)";
	CHECK(input.capitalize() == output);

	input = "snake_case_function(snake_case_arg)";
	output = "Snake Case Function(snake Case Arg)";
	CHECK(input.capitalize() == output);

	input = "snake_case_function( snake_case_arg )";
	output = "Snake Case Function( Snake Case Arg )";
	CHECK(input.capitalize() == output);
}

TEST_CASE("[String] Checking string is empty when it should be") {
	bool state = true;
	bool success;

	String a = "";
	success = a[0] == 0;
	if (!success) {
		state = false;
	}
	String b = "Godot";
	success = b[b.size()] == 0;
	if (!success) {
		state = false;
	}
	const String c = "";
	success = c[0] == 0;
	if (!success) {
		state = false;
	}

	const String d = "Godot";
	success = d[d.size()] == 0;
	if (!success) {
		state = false;
	}

	CHECK(state);
}

TEST_CASE("[String] lstrip and rstrip") {
#define STRIP_TEST(x)             \
	{                             \
		bool success = x;         \
		state = state && success; \
	}

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
	String empty;
	CHECK(empty.parse_utf8(NULL, -1));
}

TEST_CASE("[String] Cyrillic to_lower()") {
	String upper = String::utf8("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ");
	String lower = String::utf8("абвгдеёжзийклмнопрстуфхцчшщъыьэюя");

	String test = upper.to_lower();

	bool state = test == lower;

	CHECK(state);
}

TEST_CASE("[String] Count and countn functionality") {
#define COUNT_TEST(x)             \
	{                             \
		bool success = x;         \
		state = state && success; \
	}

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
