/*************************************************************************/
/*  test_string.h                                                        */
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

#ifndef TEST_STRING_H
#define TEST_STRING_H

#include "core/string/ustring.h"

#include "tests/test_macros.h"

namespace TestString {

int u32scmp(const char32_t *l, const char32_t *r) {
	for (; *l == *r && *l && *r; l++, r++)
		;
	return *l - *r;
}

TEST_CASE("[String] Assign Latin-1 char string") {
	String s = "Hello";
	CHECK(u32scmp(s.get_data(), U"Hello") == 0);
}

TEST_CASE("[String] Assign from Latin-1 char string (operator=)") {
	String s = "Dolly";
	const String &t = s;
	CHECK(u32scmp(t.get_data(), U"Dolly") == 0);
}

TEST_CASE("[String] Assign from Latin-1 char string (copycon)") {
	String s("Sheep");
	const String &t1(s);
	CHECK(u32scmp(t1.get_data(), U"Sheep") == 0);

	String t2 = String("Sheep", 3);
	CHECK(u32scmp(t2.get_data(), U"She") == 0);
}

TEST_CASE("[String] Assign from wchar_t string (operator=)") {
	String s = L"Give me";
	CHECK(u32scmp(s.get_data(), U"Give me") == 0);
}

TEST_CASE("[String] Assign from wchar_t string (copycon)") {
	String s(L"Wool");
	CHECK(u32scmp(s.get_data(), U"Wool") == 0);
}

TEST_CASE("[String] Assign from char32_t string (operator=)") {
	String s = U"Give me";
	CHECK(u32scmp(s.get_data(), U"Give me") == 0);
}

TEST_CASE("[String] Assign from char32_t string (copycon)") {
	String s(U"Wool");
	CHECK(u32scmp(s.get_data(), U"Wool") == 0);
}

TEST_CASE("[String] UTF8") {
	/* how can i embed UTF in here? */
	static const char32_t u32str[] = { 0x0045, 0x0020, 0x304A, 0x360F, 0x3088, 0x3046, 0x1F3A4, 0 };
	static const uint8_t u8str[] = { 0x45, 0x20, 0xE3, 0x81, 0x8A, 0xE3, 0x98, 0x8F, 0xE3, 0x82, 0x88, 0xE3, 0x81, 0x86, 0xF0, 0x9F, 0x8E, 0xA4, 0 };
	String s = u32str;
	bool err = s.parse_utf8(s.utf8().get_data());
	CHECK(!err);
	CHECK(s == u32str);

	err = s.parse_utf8((const char *)u8str);
	CHECK(!err);
	CHECK(s == u32str);

	CharString cs = (const char *)u8str;
	CHECK(String::utf8(cs) == s);
}

TEST_CASE("[String] UTF16") {
	/* how can i embed UTF in here? */
	static const char32_t u32str[] = { 0x0045, 0x0020, 0x304A, 0x360F, 0x3088, 0x3046, 0x1F3A4, 0 };
	static const char16_t u16str[] = { 0x0045, 0x0020, 0x304A, 0x360F, 0x3088, 0x3046, 0xD83C, 0xDFA4, 0 };
	String s = u32str;
	bool err = s.parse_utf16(s.utf16().get_data());
	CHECK(!err);
	CHECK(s == u32str);

	err = s.parse_utf16(u16str);
	CHECK(!err);
	CHECK(s == u32str);

	Char16String cs = u16str;
	CHECK(String::utf16(cs) == s);
}

TEST_CASE("[String] UTF8 with BOM") {
	/* how can i embed UTF in here? */
	static const char32_t u32str[] = { 0x0045, 0x0020, 0x304A, 0x360F, 0x3088, 0x3046, 0x1F3A4, 0 };
	static const uint8_t u8str[] = { 0xEF, 0xBB, 0xBF, 0x45, 0x20, 0xE3, 0x81, 0x8A, 0xE3, 0x98, 0x8F, 0xE3, 0x82, 0x88, 0xE3, 0x81, 0x86, 0xF0, 0x9F, 0x8E, 0xA4, 0 };
	String s;
	bool err = s.parse_utf8((const char *)u8str);
	CHECK(!err);
	CHECK(s == u32str);

	CharString cs = (const char *)u8str;
	CHECK(String::utf8(cs) == s);
}

TEST_CASE("[String] UTF16 with BOM") {
	/* how can i embed UTF in here? */
	static const char32_t u32str[] = { 0x0020, 0x0045, 0x304A, 0x360F, 0x3088, 0x3046, 0x1F3A4, 0 };
	static const char16_t u16str[] = { 0xFEFF, 0x0020, 0x0045, 0x304A, 0x360F, 0x3088, 0x3046, 0xD83C, 0xDFA4, 0 };
	static const char16_t u16str_swap[] = { 0xFFFE, 0x2000, 0x4500, 0x4A30, 0x0F36, 0x8830, 0x4630, 0x3CD8, 0xA4DF, 0 };
	String s;
	bool err = s.parse_utf16(u16str);
	CHECK(!err);
	CHECK(s == u32str);

	err = s.parse_utf16(u16str_swap);
	CHECK(!err);
	CHECK(s == u32str);

	Char16String cs = u16str;
	CHECK(String::utf16(cs) == s);

	cs = u16str_swap;
	CHECK(String::utf16(cs) == s);
}

TEST_CASE("[String] Invalid UTF8") {
	ERR_PRINT_OFF
	static const uint8_t u8str[] = { 0x45, 0xE3, 0x81, 0x8A, 0x8F, 0xE3, 0xE3, 0x98, 0x8F, 0xE3, 0x82, 0x88, 0xE3, 0x81, 0x86, 0xF0, 0x9F, 0x8E, 0xA4, 0 };
	String s;
	bool err = s.parse_utf8((const char *)u8str);
	CHECK(err);
	CHECK(s == String());

	CharString cs = (const char *)u8str;
	CHECK(String::utf8(cs) == String());
	ERR_PRINT_ON
}

TEST_CASE("[String] Invalid UTF16") {
	ERR_PRINT_OFF
	static const char16_t u16str[] = { 0x0045, 0x304A, 0x3088, 0x3046, 0xDFA4, 0 };
	String s;
	bool err = s.parse_utf16(u16str);
	CHECK(err);
	CHECK(s == String());

	Char16String cs = u16str;
	CHECK(String::utf16(cs) == String());
	ERR_PRINT_ON
}

TEST_CASE("[String] ASCII") {
	String s = U"Primero Leche";
	String t = s.ascii(false).get_data();
	CHECK(s == t);

	t = s.ascii(true).get_data();
	CHECK(s == t);
}

TEST_CASE("[String] Comparisons (equal)") {
	String s = "Test Compare";
	CHECK(s == "Test Compare");
	CHECK(s == U"Test Compare");
	CHECK(s == L"Test Compare");
	CHECK(s == String("Test Compare"));
}

TEST_CASE("[String] Comparisons (not equal)") {
	String s = "Test Compare";
	CHECK(s != "Peanut");
	CHECK(s != U"Coconut");
	CHECK(s != L"Coconut");
	CHECK(s != String("Butter"));
}

TEST_CASE("[String] Comparisons (operator <)") {
	String s = "Bees";
	CHECK(s < "Elephant");
	CHECK(!(s < U"Amber"));
	CHECK(!(s < L"Amber"));
	CHECK(!(s < String("Beatrix")));
}

TEST_CASE("[String] Concatenation") {
	String s;

	s += "Have";
	s += ' ';
	s += 'a';
	s += String(" ");
	s = s + U"Nice";
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
	CHECK(!String("Mellon").is_empty());
	// do this more than once, to check for string corruption
	CHECK(String("").is_empty());
	CHECK(String("").is_empty());
	CHECK(String("").is_empty());
}

TEST_CASE("[String] Test chr") {
	CHECK(String::chr('H') == "H");
	CHECK(String::chr(0x3012)[0] == 0x3012);
	ERR_PRINT_OFF
	CHECK(String::chr(0xd812)[0] == 0xfffd); // Unpaired UTF-16 surrogate
	CHECK(String::chr(0x20d812)[0] == 0xfffd); // Outside UTF-32 range
	ERR_PRINT_ON
}

TEST_CASE("[String] Operator []") {
	String a = "Kugar Sane";
	a[0] = 'S';
	a[6] = 'C';
	CHECK(a == "Sugar Cane");
	CHECK(a[1] == 'u');
	CHECK(a.unicode_at(1) == 'u');
}

TEST_CASE("[String] Case function test") {
	String a = "MoMoNgA";

	CHECK(a.to_upper() == "MOMONGA");
	CHECK(a.to_lower() == "momonga");
}

TEST_CASE("[String] Case compare function test") {
	String a = "MoMoNgA";

	CHECK(a.casecmp_to("momonga") != 0);
	CHECK(a.nocasecmp_to("momonga") == 0);
}

TEST_CASE("[String] Natural compare function test") {
	String a = "img2.png";

	CHECK(a.nocasecmp_to("img10.png") > 0);
	CHECK(a.naturalnocasecmp_to("img10.png") < 0);
}

TEST_CASE("[String] hex_encode_buffer") {
	static const uint8_t u8str[] = { 0x45, 0xE3, 0x81, 0x8A, 0x8F, 0xE3 };
	String s = String::hex_encode_buffer(u8str, 6);
	CHECK(s == U"45e3818a8fe3");
}

TEST_CASE("[String] Substr") {
	String s = "Killer Baby";
	CHECK(s.substr(3, 4) == "ler ");
	CHECK(s.substr(3) == "ler Baby");
}

TEST_CASE("[String] Find") {
	String s = "Pretty Woman Woman";
	CHECK(s.find("tty") == 3);
	CHECK(s.find("Wo", 9) == 13);
	CHECK(s.find("Revenge of the Monster Truck") == -1);
	CHECK(s.rfind("man") == 15);
}

TEST_CASE("[String] Find no case") {
	String s = "Pretty Whale Whale";
	CHECK(s.findn("WHA") == 7);
	CHECK(s.findn("WHA", 9) == 13);
	CHECK(s.findn("Revenge of the Monster SawFish") == -1);
	CHECK(s.rfindn("WHA") == 13);
}

TEST_CASE("[String] Find MK") {
	Vector<String> keys;
	keys.push_back("sty");
	keys.push_back("tty");
	keys.push_back("man");

	String s = "Pretty Woman";
	int key = 0;

	CHECK(s.findmk(keys, 0, &key) == 3);
	CHECK(key == 1);

	CHECK(s.findmk(keys, 5, &key) == 9);
	CHECK(key == 2);
}

TEST_CASE("[String] Find and replace") {
	String s = "Happy Birthday, Anna!";
	s = s.replace("Birthday", "Halloween");
	CHECK(s == "Happy Halloween, Anna!");

	s = s.replace_first("H", "W");
	CHECK(s == "Wappy Halloween, Anna!");
}

TEST_CASE("[String] Insertion") {
	String s = "Who is Frederic?";
	s = s.insert(s.find("?"), " Chopin");
	CHECK(s == "Who is Frederic Chopin?");
}

TEST_CASE("[String] Number to string") {
	CHECK(String::num(0) == "0");
	CHECK(String::num(0.0) == "0"); // No trailing zeros.
	CHECK(String::num(-0.0) == "-0"); // Includes sign even for zero.
	CHECK(String::num(3.141593) == "3.141593");
	CHECK(String::num(3.141593, 3) == "3.142");
	CHECK(String::num_scientific(30000000) == "3e+07");
	CHECK(String::num_int64(3141593) == "3141593");
	CHECK(String::num_int64(0xA141593, 16) == "a141593");
	CHECK(String::num_int64(0xA141593, 16, true) == "A141593");
	CHECK(String::num(42.100023, 4) == "42.1"); // No trailing zeros.

	// String::num_real tests.
	CHECK(String::num_real(3.141593) == "3.141593");
	CHECK(String::num_real(3.141) == "3.141"); // No trailing zeros.
#ifdef REAL_T_IS_DOUBLE
	CHECK_MESSAGE(String::num_real(Math_PI) == "3.14159265358979", "Prints the appropriate amount of digits for real_t = double.");
	CHECK_MESSAGE(String::num_real(3.1415f) == "3.14149999618530", "Prints more digits of 32-bit float when real_t = double (ones that would be reliable for double).");
#else
	CHECK_MESSAGE(String::num_real(Math_PI) == "3.141593", "Prints the appropriate amount of digits for real_t = float.");
	CHECK_MESSAGE(String::num_real(3.1415f) == "3.1415", "Prints only reliable digits of 32-bit float when real_t = float.");
#endif // REAL_T_IS_DOUBLE

	// Checks doubles with many decimal places.
	CHECK(String::num(0.0000012345432123454321, -1) == "0.00000123454321"); // -1 uses 14 as sane default.
	CHECK(String::num(0.0000012345432123454321) == "0.00000123454321"); // -1 is the default value.
	CHECK(String::num(-0.0000012345432123454321) == "-0.00000123454321");
	CHECK(String::num(-10000.0000012345432123454321) == "-10000.0000012345");
	CHECK(String::num(0.0000000000012345432123454321) == "0.00000000000123");
	CHECK(String::num(0.0000000000012345432123454321, 3) == "0");

	// Note: When relevant (remainder > 0.5), the last digit gets rounded up,
	// which can also lead to not include a trailing zero, e.g. "...89" -> "...9".
	CHECK(String::num(0.0000056789876567898765) == "0.00000567898766"); // Should round last digit.
	CHECK(String::num(10000.000005678999999999) == "10000.000005679"); // We cut at ...789|99 which is rounded to ...79, so only 13 decimals.
	CHECK(String::num(42.12999999, 6) == "42.13"); // Also happens with lower decimals count.

	// 32 is MAX_DECIMALS. We can't reliably store that many so we can't compare against a string,
	// but we can check that the string length is 34 (32 + 2 for "0.").
	CHECK(String::num(0.00000123456789987654321123456789987654321, 32).length() == 34);
	CHECK(String::num(0.00000123456789987654321123456789987654321, 42).length() == 34); // Should enforce MAX_DECIMALS.
	CHECK(String::num(10000.00000123456789987654321123456789987654321, 42).length() == 38); // 32 decimals + "10000.".
}

TEST_CASE("[String] String to integer") {
	static const char *nums[4] = { "1237461283", "- 22", "0", " - 1123412" };
	static const int num[4] = { 1237461283, -22, 0, -1123412 };

	for (int i = 0; i < 4; i++) {
		CHECK(String(nums[i]).to_int() == num[i]);
	}
}

TEST_CASE("[String] Hex to integer") {
	static const char *nums[4] = { "0xFFAE", "22", "0", "AADDAD" };
	static const int64_t num[4] = { 0xFFAE, 0x22, 0, 0xAADDAD };

	for (int i = 0; i < 4; i++) {
		CHECK(String(nums[i]).hex_to_int() == num[i]);
	}
}

TEST_CASE("[String] String to float") {
	static const char *nums[4] = { "-12348298412.2", "0.05", "2.0002", " -0.0001" };
	static const double num[4] = { -12348298412.2, 0.05, 2.0002, -0.0001 };

	for (int i = 0; i < 4; i++) {
		CHECK(!(ABS(String(nums[i]).to_float() - num[i]) > 0.00001));
	}
}

TEST_CASE("[String] CamelCase to underscore") {
	CHECK(String("TestTestStringGD").camelcase_to_underscore(false) == String("Test_Test_String_GD"));
	CHECK(String("TestTestStringGD").camelcase_to_underscore(true) == String("test_test_string_gd"));
}

TEST_CASE("[String] Slicing") {
	String s = "Mars,Jupiter,Saturn,Uranus";

	const char *slices[4] = { "Mars", "Jupiter", "Saturn", "Uranus" };
	for (int i = 0; i < s.get_slice_count(","); i++) {
		CHECK(s.get_slice(",", i) == slices[i]);
	}
}

TEST_CASE("[String] Splitting") {
	String s = "Mars,Jupiter,Saturn,Uranus";
	Vector<String> l;

	const char *slices_l[3] = { "Mars", "Jupiter", "Saturn,Uranus" };
	const char *slices_r[3] = { "Mars,Jupiter", "Saturn", "Uranus" };

	l = s.split(",", true, 2);
	CHECK(l.size() == 3);
	for (int i = 0; i < l.size(); i++) {
		CHECK(l[i] == slices_l[i]);
	}

	l = s.rsplit(",", true, 2);
	CHECK(l.size() == 3);
	for (int i = 0; i < l.size(); i++) {
		CHECK(l[i] == slices_r[i]);
	}

	s = "Mars Jupiter Saturn Uranus";
	const char *slices_s[4] = { "Mars", "Jupiter", "Saturn", "Uranus" };
	l = s.split_spaces();
	for (int i = 0; i < l.size(); i++) {
		CHECK(l[i] == slices_s[i]);
	}

	s = "1.2;2.3 4.5";
	const double slices_d[3] = { 1.2, 2.3, 4.5 };

	Vector<float> f;
	f = s.split_floats(";");
	CHECK(f.size() == 2);
	for (int i = 0; i < f.size(); i++) {
		CHECK(ABS(f[i] - slices_d[i]) <= 0.00001);
	}

	Vector<String> keys;
	keys.push_back(";");
	keys.push_back(" ");

	f = s.split_floats_mk(keys);
	CHECK(f.size() == 3);
	for (int i = 0; i < f.size(); i++) {
		CHECK(ABS(f[i] - slices_d[i]) <= 0.00001);
	}

	s = "1;2 4";
	const int slices_i[3] = { 1, 2, 4 };

	Vector<int> ii;
	ii = s.split_ints(";");
	CHECK(ii.size() == 2);
	for (int i = 0; i < ii.size(); i++) {
		CHECK(ii[i] == slices_i[i]);
	}

	ii = s.split_ints_mk(keys);
	CHECK(ii.size() == 3);
	for (int i = 0; i < ii.size(); i++) {
		CHECK(ii[i] == slices_i[i]);
	}
}

struct test_27_data {
	char const *data;
	char const *part;
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
		state = s.begins_with(tc[i].part) == tc[i].expected;
		if (state) {
			String sb = tc[i].part;
			state = s.begins_with(sb) == tc[i].expected;
		}
		CHECK(state);
		if (!state) {
			break;
		}
	};
	CHECK(state);
}

TEST_CASE("[String] Ends with") {
	test_27_data tc[] = {
		{ "res://foobar", "foobar", true },
		{ "res", "res://", false },
		{ "abc", "abc", true }
	};
	size_t count = sizeof(tc) / sizeof(tc[0]);
	bool state = true;
	for (size_t i = 0; state && i < count; ++i) {
		String s = tc[i].data;
		state = s.ends_with(tc[i].part) == tc[i].expected;
		if (state) {
			String sb = tc[i].part;
			state = s.ends_with(sb) == tc[i].expected;
		}
		CHECK(state);
		if (!state) {
			break;
		}
	};
	CHECK(state);
}

TEST_CASE("[String] format") {
	const String value_format = "red=\"$red\" green=\"$green\" blue=\"$blue\" alpha=\"$alpha\"";

	Dictionary value_dictionary;
	value_dictionary["red"] = 10;
	value_dictionary["green"] = 20;
	value_dictionary["blue"] = "bla";
	value_dictionary["alpha"] = 0.4;
	String value = value_format.format(value_dictionary, "$_");

	CHECK(value == "red=\"10\" green=\"20\" blue=\"bla\" alpha=\"0.4\"");
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

TEST_CASE("[String] is_numeric") {
	CHECK(String("12").is_numeric());
	CHECK(String("1.2").is_numeric());
	CHECK(!String("AF").is_numeric());
	CHECK(String("-12").is_numeric());
	CHECK(String("-1.2").is_numeric());
}

TEST_CASE("[String] pad") {
	String s = String("test");
	CHECK(s.lpad(10, "x") == U"xxxxxxtest");
	CHECK(s.rpad(10, "x") == U"testxxxxxx");

	s = String("10.10");
	CHECK(s.pad_decimals(4) == U"10.1000");
	CHECK(s.pad_zeros(4) == U"0010.10");
}

TEST_CASE("[String] is_subsequence_of") {
	String a = "is subsequence of";
	CHECK(String("sub").is_subsequence_of(a));
	CHECK(!String("Sub").is_subsequence_of(a));
	CHECK(String("Sub").is_subsequence_ofi(a));
}

TEST_CASE("[String] match") {
	CHECK(String("img1.png").match("*.png"));
	CHECK(!String("img1.jpeg").match("*.png"));
	CHECK(!String("img1.Png").match("*.png"));
	CHECK(String("img1.Png").matchn("*.png"));
}

TEST_CASE("[String] IPVX address to string") {
	IPAddress ip0("2001:0db8:85a3:0000:0000:8a2e:0370:7334");
	IPAddress ip(0x0123, 0x4567, 0x89ab, 0xcdef, true);
	IPAddress ip2("fe80::52e5:49ff:fe93:1baf");
	IPAddress ip3("::ffff:192.168.0.1");
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
	CHECK(empty.parse_utf8(nullptr, -1));
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

#undef COUNT_TEST
}

TEST_CASE("[String] Bigrams") {
	String s = "abcd";
	Vector<String> bigr = s.bigrams();

	CHECK(bigr.size() == 3);
	CHECK(bigr[0] == "ab");
	CHECK(bigr[1] == "bc");
	CHECK(bigr[2] == "cd");
}

TEST_CASE("[String] c-escape/unescape") {
	String s = "\\1\a2\b\f3\n45\r6\t7\v8\'9\?0\"";
	CHECK(s.c_escape().c_unescape() == s);
}

TEST_CASE("[String] dedent") {
	String s = "      aaa\n    bbb";
	String t = "aaa\nbbb";
	CHECK(s.dedent() == t);
}

TEST_CASE("[String] Path functions") {
	static const char *path[7] = { "C:\\Godot\\project\\test.tscn", "/Godot/project/test.xscn", "../Godot/project/test.scn", "Godot\\test.doc", "C:\\test.", "res://test", "/.test" };
	static const char *base_dir[7] = { "C:\\Godot\\project", "/Godot/project", "../Godot/project", "Godot", "C:\\", "res://", "/" };
	static const char *base_name[7] = { "C:\\Godot\\project\\test", "/Godot/project/test", "../Godot/project/test", "Godot\\test", "C:\\test", "res://test", "/" };
	static const char *ext[7] = { "tscn", "xscn", "scn", "doc", "", "", "test" };
	static const char *file[7] = { "test.tscn", "test.xscn", "test.scn", "test.doc", "test.", "test", ".test" };
	static const bool abs[7] = { true, true, false, false, true, true, true };

	for (int i = 0; i < 7; i++) {
		CHECK(String(path[i]).get_base_dir() == base_dir[i]);
		CHECK(String(path[i]).get_basename() == base_name[i]);
		CHECK(String(path[i]).get_extension() == ext[i]);
		CHECK(String(path[i]).get_file() == file[i]);
		CHECK(String(path[i]).is_absolute_path() == abs[i]);
		CHECK(String(path[i]).is_relative_path() != abs[i]);
		CHECK(String(path[i]).simplify_path().get_base_dir().plus_file(file[i]) == String(path[i]).simplify_path());
	}

	static const char *file_name[3] = { "test.tscn", "test://.xscn", "?tes*t.scn" };
	static const bool valid[3] = { true, false, false };
	for (int i = 0; i < 3; i++) {
		CHECK(String(file_name[i]).is_valid_filename() == valid[i]);
	}
}

TEST_CASE("[String] hash") {
	String a = "Test";
	String b = "Test";
	String c = "West";
	CHECK(a.hash() == b.hash());
	CHECK(a.hash() != c.hash());

	CHECK(a.hash64() == b.hash64());
	CHECK(a.hash64() != c.hash64());
}

TEST_CASE("[String] uri_encode/unescape") {
	String s = "Godot Engine:'docs'";
	String t = "Godot%20Engine%3A%27docs%27";

	String x1 = "T%C4%93%C5%A1t";
	static const uint8_t u8str[] = { 0x54, 0xC4, 0x93, 0xC5, 0xA1, 0x74, 0x00 };
	String x2 = String::utf8((const char *)u8str);
	String x3 = U"Tēšt";

	CHECK(x1.uri_decode() == x2);
	CHECK(x1.uri_decode() == x3);
	CHECK((x1 + x3).uri_decode() == (x2 + x3)); // Mixed unicode and URL encoded string, e.g. GTK+ bookmark.
	CHECK(x2.uri_encode() == x1);
	CHECK(x3.uri_encode() == x1);

	CHECK(s.uri_encode() == t);
	CHECK(t.uri_decode() == s);
}

TEST_CASE("[String] xml_escape/unescape") {
	String s = "\"Test\" <test@test&'test'>";
	CHECK(s.xml_escape(true).xml_unescape() == s);
	CHECK(s.xml_escape(false).xml_unescape() == s);
}

TEST_CASE("[String] xml_unescape") {
	// Named entities
	String input = "&quot;&amp;&apos;&lt;&gt;";
	CHECK(input.xml_unescape() == "\"&\'<>");

	// Numeric entities
	input = "&#x41;&#66;";
	CHECK(input.xml_unescape() == "AB");

	input = "&#0;&x#0;More text";
	String result = input.xml_unescape();
	// Didn't put in a leading NUL and terminate the string
	CHECK(input.length() > 0);
	CHECK(input[0] != '\0');
	// Entity should be left as-is if invalid
	CHECK(input.xml_unescape() == input);

	// Check near char32_t range
	input = "&#xFFFFFFFF;";
	result = input.xml_unescape();
	CHECK(result.length() == 1);
	CHECK(result[0] == 0xFFFFFFFF);
	input = "&#4294967295;";
	result = input.xml_unescape();
	CHECK(result.length() == 1);
	CHECK(result[0] == 0xFFFFFFFF);

	// Check out of range of char32_t
	input = "&#xFFFFFFFFF;";
	CHECK(input.xml_unescape() == input);
	input = "&#4294967296;";
	CHECK(input.xml_unescape() == input);

	// Shouldn't consume without ending in a ';'
	input = "&#66";
	CHECK(input.xml_unescape() == input);
	input = "&#x41";
	CHECK(input.xml_unescape() == input);

	// Invalid characters should make the entity ignored
	input = "&#x41SomeIrrelevantText;";
	CHECK(input.xml_unescape() == input);
	input = "&#66SomeIrrelevantText;";
	CHECK(input.xml_unescape() == input);
}

TEST_CASE("[String] Strip escapes") {
	String s = "\t\tTest Test\r\n Test";
	CHECK(s.strip_escapes() == "Test Test Test");
}

TEST_CASE("[String] Similarity") {
	String a = "Test";
	String b = "West";
	String c = "Toad";
	CHECK(a.similarity(b) > a.similarity(c));
}

TEST_CASE("[String] Strip edges") {
	String s = "\t Test Test   ";
	CHECK(s.strip_edges(true, false) == "Test Test   ");
	CHECK(s.strip_edges(false, true) == "\t Test Test");
	CHECK(s.strip_edges(true, true) == "Test Test");
}

TEST_CASE("[String] Trim") {
	String s = "aaaTestbbb";
	CHECK(s.trim_prefix("aaa") == "Testbbb");
	CHECK(s.trim_suffix("bbb") == "aaaTest");
	CHECK(s.trim_suffix("Test") == s);
}

TEST_CASE("[String] Right/Left") {
	String s = "aaaTestbbb";
	//                ^
	CHECK(s.right(6) == "estbbb");
	CHECK(s.right(-6) == "tbbb");
	CHECK(s.left(6) == "aaaTes");
	CHECK(s.left(-6) == "aaaT");
}

TEST_CASE("[String] Repeat") {
	String s = "abababab";
	String x = "ab";
	String t = x.repeat(4);
	CHECK(t == s);
}

TEST_CASE("[String] SHA1/SHA256/MD5") {
	String s = "Godot";
	String sha1 = "a1e91f39b9fce6a9998b14bdbe2aa2b39dc2d201";
	static uint8_t sha1_buf[20] = {
		0xA1, 0xE9, 0x1F, 0x39, 0xB9, 0xFC, 0xE6, 0xA9, 0x99, 0x8B, 0x14, 0xBD, 0xBE, 0x2A, 0xA2, 0xB3,
		0x9D, 0xC2, 0xD2, 0x01
	};
	String sha256 = "2a02b2443f7985d89d09001086ae3dcfa6eb0f55c6ef170715d42328e16e6cb8";
	static uint8_t sha256_buf[32] = {
		0x2A, 0x02, 0xB2, 0x44, 0x3F, 0x79, 0x85, 0xD8, 0x9D, 0x09, 0x00, 0x10, 0x86, 0xAE, 0x3D, 0xCF,
		0xA6, 0xEB, 0x0F, 0x55, 0xC6, 0xEF, 0x17, 0x07, 0x15, 0xD4, 0x23, 0x28, 0xE1, 0x6E, 0x6C, 0xB8
	};
	String md5 = "4a336d087aeb0390da10ee2ea7cb87f8";
	static uint8_t md5_buf[16] = {
		0x4A, 0x33, 0x6D, 0x08, 0x7A, 0xEB, 0x03, 0x90, 0xDA, 0x10, 0xEE, 0x2E, 0xA7, 0xCB, 0x87, 0xF8
	};

	PackedByteArray buf = s.sha1_buffer();
	CHECK(memcmp(sha1_buf, buf.ptr(), 20) == 0);
	CHECK(s.sha1_text() == sha1);

	buf = s.sha256_buffer();
	CHECK(memcmp(sha256_buf, buf.ptr(), 32) == 0);
	CHECK(s.sha256_text() == sha256);

	buf = s.md5_buffer();
	CHECK(memcmp(md5_buf, buf.ptr(), 16) == 0);
	CHECK(s.md5_text() == md5);
}

TEST_CASE("[String] Join") {
	String s = ", ";
	Vector<String> parts;
	parts.push_back("One");
	parts.push_back("B");
	parts.push_back("C");
	String t = s.join(parts);
	CHECK(t == "One, B, C");
}

TEST_CASE("[String] Is_*") {
	static const char *data[12] = { "-30", "100", "10.1", "10,1", "1e2", "1e-2", "1e2e3", "0xAB", "AB", "Test1", "1Test", "Test*1" };
	static bool isnum[12] = { true, true, true, false, false, false, false, false, false, false, false, false };
	static bool isint[12] = { true, true, false, false, false, false, false, false, false, false, false, false };
	static bool ishex[12] = { true, true, false, false, true, false, true, false, true, false, false, false };
	static bool ishex_p[12] = { false, false, false, false, false, false, false, true, false, false, false, false };
	static bool isflt[12] = { true, true, true, false, true, true, false, false, false, false, false, false };
	static bool isid[12] = { false, false, false, false, false, false, false, false, true, true, false, false };
	for (int i = 0; i < 12; i++) {
		String s = String(data[i]);
		CHECK(s.is_numeric() == isnum[i]);
		CHECK(s.is_valid_int() == isint[i]);
		CHECK(s.is_valid_hex_number(false) == ishex[i]);
		CHECK(s.is_valid_hex_number(true) == ishex_p[i]);
		CHECK(s.is_valid_float() == isflt[i]);
		CHECK(s.is_valid_identifier() == isid[i]);
	}
}

TEST_CASE("[String] humanize_size") {
	CHECK(String::humanize_size(1000) == "1000 B");
	CHECK(String::humanize_size(1025) == "1.00 KiB");
	CHECK(String::humanize_size(1025300) == "1001.2 KiB");
	CHECK(String::humanize_size(100523550) == "95.86 MiB");
	CHECK(String::humanize_size(5345555000) == "4.97 GiB");
}

TEST_CASE("[String] validate_node_name") {
	String numeric_only = "12345";
	CHECK(numeric_only.validate_node_name() == "12345");

	String name_with_spaces = "Name with spaces";
	CHECK(name_with_spaces.validate_node_name() == "Name with spaces");

	String name_with_kana = "Name with kana ゴドツ";
	CHECK(name_with_kana.validate_node_name() == "Name with kana ゴドツ");

	String name_with_invalid_chars = "Name with invalid characters :.@removed!";
	CHECK(name_with_invalid_chars.validate_node_name() == "Name with invalid characters removed!");
}

TEST_CASE("[String] Variant indexed get") {
	Variant s = String("abcd");
	bool valid = false;
	bool oob = true;

	String r = s.get_indexed(1, valid, oob);

	CHECK(valid);
	CHECK_FALSE(oob);
	CHECK_EQ(r, String("b"));
}

TEST_CASE("[String] Variant validated indexed get") {
	Variant s = String("abcd");

	Variant::ValidatedIndexedGetter getter = Variant::get_member_validated_indexed_getter(Variant::STRING);

	Variant r;
	bool oob = true;
	getter(&s, 1, &r, &oob);

	CHECK_FALSE(oob);
	CHECK_EQ(r, String("b"));
}

TEST_CASE("[String] Variant ptr indexed get") {
	String s("abcd");

	Variant::PTRIndexedGetter getter = Variant::get_member_ptr_indexed_getter(Variant::STRING);

	String r;
	getter(&s, 1, &r);

	CHECK_EQ(r, String("b"));
}

TEST_CASE("[String] Variant indexed set") {
	Variant s = String("abcd");
	bool valid = false;
	bool oob = true;

	s.set_indexed(1, String("z"), valid, oob);

	CHECK(valid);
	CHECK_FALSE(oob);
	CHECK_EQ(s, String("azcd"));
}

TEST_CASE("[String] Variant validated indexed set") {
	Variant s = String("abcd");

	Variant::ValidatedIndexedSetter setter = Variant::get_member_validated_indexed_setter(Variant::STRING);

	Variant v = String("z");
	bool oob = true;
	setter(&s, 1, &v, &oob);

	CHECK_FALSE(oob);
	CHECK_EQ(s, String("azcd"));
}

TEST_CASE("[String] Variant ptr indexed set") {
	String s("abcd");

	Variant::PTRIndexedSetter setter = Variant::get_member_ptr_indexed_setter(Variant::STRING);

	String v("z");
	setter(&s, 1, &v);

	CHECK_EQ(s, String("azcd"));
}
} // namespace TestString

#endif // TEST_STRING_H
