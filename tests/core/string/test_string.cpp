/**************************************************************************/
/*  test_string.cpp                                                       */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_string)

#include "core/string/ustring.h"

namespace TestString {

int u32scmp(const char32_t *l, const char32_t *r) {
	for (; *l == *r && *l && *r; l++, r++) {
		// Continue.
	}
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

	String t2 = String::latin1(Span("Sheep", 3));
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
	String expected = u32str;
	String parsed;
	Error err = parsed.append_utf8(expected.utf8().get_data());
	CHECK(err == OK);
	CHECK(parsed == u32str);

	parsed.clear();
	err = parsed.append_utf8((const char *)u8str);
	CHECK(err == OK);
	CHECK(parsed == u32str);

	CharString cs = (const char *)u8str;
	CHECK(String::utf8(cs) == parsed);
}

TEST_CASE("[String] UTF16") {
	/* how can i embed UTF in here? */
	static const char32_t u32str[] = { 0x0045, 0x0020, 0x304A, 0x360F, 0x3088, 0x3046, 0x1F3A4, 0 };
	static const char16_t u16str[] = { 0x0045, 0x0020, 0x304A, 0x360F, 0x3088, 0x3046, 0xD83C, 0xDFA4, 0 };
	String expected = u32str;
	String parsed;
	Error err = parsed.append_utf16(expected.utf16().get_data());
	CHECK(err == OK);
	CHECK(parsed == u32str);

	parsed.clear();
	err = parsed.append_utf16(u16str);
	CHECK(err == OK);
	CHECK(parsed == u32str);

	Char16String cs = u16str;
	CHECK(String::utf16(cs) == parsed);
}

TEST_CASE("[String] UTF8 with BOM") {
	/* how can i embed UTF in here? */
	static const char32_t u32str[] = { 0x0045, 0x0020, 0x304A, 0x360F, 0x3088, 0x3046, 0x1F3A4, 0 };
	static const uint8_t u8str[] = { 0xEF, 0xBB, 0xBF, 0x45, 0x20, 0xE3, 0x81, 0x8A, 0xE3, 0x98, 0x8F, 0xE3, 0x82, 0x88, 0xE3, 0x81, 0x86, 0xF0, 0x9F, 0x8E, 0xA4, 0 };
	String s;
	Error err = s.append_utf8((const char *)u8str);
	CHECK(err == OK);
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
	Error err = s.append_utf16(u16str);
	CHECK(err == OK);
	CHECK(s == u32str);

	s.clear();
	err = s.append_utf16(u16str_swap);
	CHECK(err == OK);
	CHECK(s == u32str);

	Char16String cs = u16str;
	CHECK(String::utf16(cs) == s);

	cs = u16str_swap;
	CHECK(String::utf16(cs) == s);
}

TEST_CASE("[String] Invalid UTF8 (non shortest form sequence)") {
	ERR_PRINT_OFF
	// Examples from the unicode standard : 3.9 Unicode Encoding Forms - Table 3.8.
	static const uint8_t u8str[] = { 0xC0, 0xAF, 0xE0, 0x80, 0xBF, 0xF0, 0x81, 0x82, 0x41, 0 };
	static const char32_t u32str[] = { 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0x41, 0 };
	String s;
	Error err = s.append_utf8((const char *)u8str);
	CHECK(err == ERR_INVALID_DATA);
	CHECK(s == u32str);

	CharString cs = (const char *)u8str;
	CHECK(String::utf8(cs) == s);
	ERR_PRINT_ON
}

TEST_CASE("[String] Invalid UTF8 (ill formed sequences for surrogates)") {
	ERR_PRINT_OFF
	// Examples from the unicode standard : 3.9 Unicode Encoding Forms - Table 3.9.
	static const uint8_t u8str[] = { 0xED, 0xA0, 0x80, 0xED, 0xBF, 0xBF, 0xED, 0xAF, 0x41, 0 };
	static const char32_t u32str[] = { 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0x41, 0 };
	String s;
	Error err = s.append_utf8((const char *)u8str);
	CHECK(err == ERR_INVALID_DATA);
	CHECK(s == u32str);

	CharString cs = (const char *)u8str;
	CHECK(String::utf8(cs) == s);
	ERR_PRINT_ON
}

TEST_CASE("[String] Invalid UTF8 (other ill formed sequences)") {
	ERR_PRINT_OFF
	// Examples from the unicode standard : 3.9 Unicode Encoding Forms - Table 3.10.
	static const uint8_t u8str[] = { 0xF4, 0x91, 0x92, 0x93, 0xFF, 0x41, 0x80, 0xBF, 0x42, 0 };
	static const char32_t u32str[] = { 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0x41, 0xFFFD, 0xFFFD, 0x42, 0 };
	String s;
	Error err = s.append_utf8((const char *)u8str);
	CHECK(err == ERR_INVALID_DATA);
	CHECK(s == u32str);

	CharString cs = (const char *)u8str;
	CHECK(String::utf8(cs) == s);
	ERR_PRINT_ON
}

TEST_CASE("[String] Invalid UTF8 (truncated sequences)") {
	ERR_PRINT_OFF
	// Examples from the unicode standard : 3.9 Unicode Encoding Forms - Table 3.11.
	static const uint8_t u8str[] = { 0xE1, 0x80, 0xE2, 0xF0, 0x91, 0x92, 0xF1, 0xBF, 0x41, 0 };
	static const char32_t u32str[] = { 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0x41, 0 };
	String s;
	Error err = s.append_utf8((const char *)u8str);
	CHECK(err == ERR_INVALID_DATA);
	CHECK(s == u32str);

	CharString cs = (const char *)u8str;
	CHECK(String::utf8(cs) == s);
	ERR_PRINT_ON
}

TEST_CASE("[String] Invalid UTF16 (non-standard)") {
	ERR_PRINT_OFF
	static const char16_t u16str[] = { 0x0045, 0x304A, 0x3088, 0x3046, 0xDFA4, 0 };
	//                                 +       +       +       +       unpaired
	static const char32_t u32str[] = { 0x0045, 0x304A, 0x3088, 0x3046, 0xDFA4, 0 };
	String s;
	Error err = s.append_utf16(u16str);
	CHECK(err == ERR_PARSE_ERROR);
	CHECK(s == u32str);

	Char16String cs = u16str;
	CHECK(String::utf16(cs) == s);
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

	CharString empty = "";
	CharString cs = "Test Compare";
	CHECK(!(empty == cs));
	CHECK(!(cs == empty));
	CHECK(cs == CharString("Test Compare"));
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

TEST_CASE("[String] Contains") {
	String s = "C:\\Godot\\project\\string_test.tscn";
	CHECK(s.contains(":\\"));
	CHECK(s.contains("Godot"));
	CHECK(s.contains(String("project\\string_test")));
	CHECK(s.contains(String("\\string_test.tscn")));

	CHECK(!s.contains("://"));
	CHECK(!s.contains("Godoh"));
	CHECK(!s.contains(String("project\\string test")));
	CHECK(!s.contains(String("\\char_test.tscn")));
}

TEST_CASE("[String] Contains case insensitive") {
	String s = "C:\\Godot\\project\\string_test.tscn";
	CHECK(s.containsn("Godot"));
	CHECK(s.containsn("godot"));
	CHECK(s.containsn(String("Project\\string_test")));
	CHECK(s.containsn(String("\\string_Test.tscn")));

	CHECK(!s.containsn("Godoh"));
	CHECK(!s.containsn("godoh"));
	CHECK(!s.containsn(String("project\\string test")));
	CHECK(!s.containsn(String("\\char_test.tscn")));
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

TEST_CASE("[String] File compare function test") {
	String a = "_img2.png";
	String b = "img2.png";

	CHECK(a.nocasecmp_to("img10.png") > 0);
	CHECK_MESSAGE(a.filenocasecmp_to("img10.png") < 0, "Should sort before letters.");
	CHECK_MESSAGE(a.filenocasecmp_to(".img10.png") > 0, "Should sort after period.");
	CHECK(b.filenocasecmp_to("img3.png") < 0);
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
	MULTICHECK_STRING_EQ(s, find, "tty", 3);
	MULTICHECK_STRING_EQ(s, find, "Revenge of the Monster Truck", -1);
	MULTICHECK_STRING_INT_EQ(s, find, "Wo", 9, 13);
	MULTICHECK_STRING_INT_EQ(s, find, "Wo", 1000, -1);
	MULTICHECK_STRING_INT_EQ(s, find, "Wo", -1, -1);
	MULTICHECK_STRING_EQ(s, find, "", -1);
	MULTICHECK_STRING_EQ(s, find, "Pretty Woman Woman", 0);
	MULTICHECK_STRING_EQ(s, find, "WOMAN", -1);
	MULTICHECK_STRING_INT_EQ(s, find, "", 9, -1);

	MULTICHECK_STRING_EQ(s, rfind, "", -1);
	MULTICHECK_STRING_EQ(s, rfind, "foo", -1);
	MULTICHECK_STRING_EQ(s, rfind, "Pretty Woman Woman", 0);
	MULTICHECK_STRING_EQ(s, rfind, "man", 15);
	MULTICHECK_STRING_EQ(s, rfind, "WOMAN", -1);
	MULTICHECK_STRING_INT_EQ(s, rfind, "", 15, -1);
	MULTICHECK_STRING_INT_EQ(s, rfind, "Wo", 1000, -1);
	MULTICHECK_STRING_INT_EQ(s, rfind, "Wo", -1, 13);
}

TEST_CASE("[String] Find character") {
	String s = "racecar";
	CHECK_EQ(s.find_char('r'), 0);
	CHECK_EQ(s.find_char('r', 1), 6);
	CHECK_EQ(s.find_char('e'), 3);
	CHECK_EQ(s.find_char('e', 4), -1);

	CHECK_EQ(s.rfind_char('r'), 6);
	CHECK_EQ(s.rfind_char('r', 5), 0);
	CHECK_EQ(s.rfind_char('e'), 3);
	CHECK_EQ(s.rfind_char('e', 2), -1);
}

TEST_CASE("[String] Find case insensitive") {
	String s = "Pretty Whale Whale";
	MULTICHECK_STRING_EQ(s, findn, "WHA", 7);
	MULTICHECK_STRING_INT_EQ(s, findn, "WHA", 9, 13);
	MULTICHECK_STRING_INT_EQ(s, findn, "WHA", 1000, -1);
	MULTICHECK_STRING_INT_EQ(s, findn, "WHA", -1, -1);
	MULTICHECK_STRING_EQ(s, findn, "Revenge of the Monster SawFish", -1);
	MULTICHECK_STRING_EQ(s, findn, "", -1);
	MULTICHECK_STRING_EQ(s, findn, "wha", 7);
	MULTICHECK_STRING_EQ(s, findn, "Wha", 7);
	MULTICHECK_STRING_INT_EQ(s, findn, "", 3, -1);

	MULTICHECK_STRING_EQ(s, rfindn, "WHA", 13);
	MULTICHECK_STRING_EQ(s, rfindn, "", -1);
	MULTICHECK_STRING_EQ(s, rfindn, "wha", 13);
	MULTICHECK_STRING_EQ(s, rfindn, "Wha", 13);
	MULTICHECK_STRING_INT_EQ(s, rfindn, "", 13, -1);
	MULTICHECK_STRING_INT_EQ(s, rfindn, "WHA", 1000, -1);
	MULTICHECK_STRING_INT_EQ(s, rfindn, "WHA", -1, 13);
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

	CHECK(s.findmk(keys, -1, &key) == -1);
	CHECK(s.findmk(keys, 1000, &key) == -1);
}

TEST_CASE("[String] Find and replace") {
	String s = "Happy Birthday, Anna!";
	MULTICHECK_STRING_STRING_EQ(s, replace, "Birthday", "Halloween", "Happy Halloween, Anna!");
	MULTICHECK_STRING_STRING_EQ(s, replace_first, "y", "Y", "HappY Birthday, Anna!");
	MULTICHECK_STRING_STRING_EQ(s, replacen, "Y", "Y", "HappY BirthdaY, Anna!");
}

TEST_CASE("[String] replace_char") {
	String s = "Banana";
	CHECK(s.replace_char('n', 'x') == "Baxaxa");
	CHECK(s.replace_char('\0', 'x') == "Banana");
	ERR_PRINT_OFF
	CHECK(s.replace_char('n', '\0') == "Banana");
	ERR_PRINT_ON
}

TEST_CASE("[String] replace_chars") {
	String s = "Banana";
	CHECK(s.replace_chars(String("Bn"), 'x') == "xaxaxa");
	CHECK(s.replace_chars("Bn", 'x') == "xaxaxa");
	CHECK(s.replace_chars(String(), 'x') == "Banana");
	CHECK(s.replace_chars("", 'x') == "Banana");
	ERR_PRINT_OFF
	CHECK(s.replace_chars(String("Bn"), '\0') == "Banana");
	CHECK(s.replace_chars("Bn", '\0') == "Banana");
	ERR_PRINT_ON
}

TEST_CASE("[String] Insertion") {
	String s = "Who is Frederic?";
	s = s.insert(s.find("?"), " Chopin");
	CHECK(s == "Who is Frederic Chopin?");

	s = "foobar";
	CHECK(s.insert(0, "X") == "Xfoobar");
	CHECK(s.insert(-100, "X") == "foobar");
	CHECK(s.insert(6, "X") == "foobarX");
	CHECK(s.insert(100, "X") == "foobarX");
	CHECK(s.insert(2, "") == "foobar");

	s = "";
	CHECK(s.insert(0, "abc") == "abc");
	CHECK(s.insert(100, "abc") == "abc");
	CHECK(s.insert(-100, "abc") == "");
	CHECK(s.insert(0, "") == "");
}

TEST_CASE("[String] Erasing") {
	String s = "Josephine is such a cute girl!";
	s = s.erase(s.find("cute "), String("cute ").length());
	CHECK(s == "Josephine is such a girl!");
}

TEST_CASE("[String] remove_char") {
	String s = "Banana";
	CHECK(s.remove_char('a') == "Bnn");
	CHECK(s.remove_char('\0') == "Banana");
	CHECK(s.remove_char('x') == "Banana");
}

TEST_CASE("[String] remove_chars") {
	String s = "Banana";
	CHECK(s.remove_chars("Ba") == "nn");
	CHECK(s.remove_chars(String("Ba")) == "nn");
	CHECK(s.remove_chars("") == "Banana");
	CHECK(s.remove_chars(String()) == "Banana");
	CHECK(s.remove_chars("xy") == "Banana");
	CHECK(s.remove_chars(String("xy")) == "Banana");
}

TEST_CASE("[String] Number to string") {
	CHECK(String::num(0) == "0.0"); // The method takes double, so always add zeros.
	CHECK(String::num(0.0) == "0.0");
	CHECK(String::num(-0.0) == "-0.0"); // Includes sign even for zero.
	CHECK(String::num(3.141593) == "3.141593");
	CHECK(String::num(3.141593, 3) == "3.142");
	CHECK(String::num(42.100023, 4) == "42.1"); // No trailing zeros.

	// String::num_int64 tests.
	CHECK(String::num_int64(3141593) == "3141593");
	CHECK(String::num_int64(-3141593) == "-3141593");
	CHECK(String::num_int64(0xA141593, 16) == "a141593");
	CHECK(String::num_int64(0xA141593, 16, true) == "A141593");
	ERR_PRINT_OFF;
	CHECK(String::num_int64(3141593, 1) == ""); // Invalid base < 2.
	CHECK(String::num_int64(3141593, 37) == ""); // Invalid base > 36.
	ERR_PRINT_ON;

	// String::num_uint64 tests.
	CHECK(String::num_uint64(4294967295) == "4294967295");
	CHECK(String::num_uint64(0xF141593, 16) == "f141593");
	CHECK(String::num_uint64(0xF141593, 16, true) == "F141593");
	ERR_PRINT_OFF;
	CHECK(String::num_uint64(4294967295, 1) == ""); // Invalid base < 2.
	CHECK(String::num_uint64(4294967295, 37) == ""); // Invalid base > 36.
	ERR_PRINT_ON;

	// String::num_scientific tests.
	CHECK(String::num_scientific(30000000.0) == "30000000");
	CHECK(String::num_scientific(1234567890.0) == "1234567890");
	CHECK(String::num_scientific(3e100) == "3e+100");
	CHECK(String::num_scientific(7e-100) == "7e-100");
	CHECK(String::num_scientific(Math::TAU) == "6.283185307179586");
	CHECK(String::num_scientific(Math::INF) == "inf");
	CHECK(String::num_scientific(-Math::INF) == "-inf");
	CHECK(String::num_scientific(Math::NaN) == "nan");
	CHECK(String::num_scientific(2.0) == "2");
	CHECK(String::num_scientific(1.0) == "1");
	CHECK(String::num_scientific(0.0) == "0");
	CHECK(String::num_scientific(-0.0) == "-0");

	// String::num_real tests.
	CHECK(String::num_real(1.0) == "1.0");
	CHECK(String::num_real(1.0, false) == "1");
	CHECK(String::num_real(9.9) == "9.9");
	CHECK(String::num_real(9.99) == "9.99");
	CHECK(String::num_real(9.999) == "9.999");
	CHECK(String::num_real(9.9999) == "9.9999");
	CHECK(String::num_real(3.141593) == "3.141593");
	CHECK(String::num_real(3.141) == "3.141"); // No trailing zeros.
#ifdef REAL_T_IS_DOUBLE
	CHECK_MESSAGE(String::num_real(real_t(123.456789)) == "123.456789", "Prints the appropriate amount of digits for real_t = double.");
	CHECK_MESSAGE(String::num_real(real_t(-123.456789)) == "-123.456789", "Prints the appropriate amount of digits for real_t = double.");
	CHECK_MESSAGE(String::num_real(real_t(Math::PI)) == "3.14159265358979", "Prints the appropriate amount of digits for real_t = double.");
	CHECK_MESSAGE(String::num_real(real_t(3.1415f)) == "3.1414999961853", "Prints more digits of 32-bit float when real_t = double (ones that would be reliable for double) and no trailing zero.");
#else
	CHECK_MESSAGE(String::num_real(real_t(123.456789)) == "123.4568", "Prints the appropriate amount of digits for real_t = float.");
	CHECK_MESSAGE(String::num_real(real_t(-123.456789)) == "-123.4568", "Prints the appropriate amount of digits for real_t = float.");
	CHECK_MESSAGE(String::num_real(real_t(Math::PI)) == "3.141593", "Prints the appropriate amount of digits for real_t = float.");
	CHECK_MESSAGE(String::num_real(real_t(3.1415f)) == "3.1415", "Prints only reliable digits of 32-bit float when real_t = float.");
#endif // REAL_T_IS_DOUBLE

	// Checks doubles with many decimal places.
	CHECK(String::num(0.0000012345432123454321, -1) == "0.00000123454321"); // -1 uses 14 as sane default.
	CHECK(String::num(0.0000012345432123454321) == "0.00000123454321"); // -1 is the default value.
	CHECK(String::num(-0.0000012345432123454321) == "-0.00000123454321");
	CHECK(String::num(-10000.0000012345432123454321) == "-10000.0000012345");
	CHECK(String::num(0.0000000000012345432123454321) == "0.00000000000123");
	CHECK(String::num(0.0000000000012345432123454321, 3) == "0.0");

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
	static const char *nums[14] = { "1237461283", "- 22", "0", " - 1123412", "", "10_000_000", "-1_2_3_4", "10__000", "  1  2  34 ", "-0", "007", "--45", "---46", "-7-2" };
	static const int num[14] = { 1237461283, -22, 0, -1123412, 0, 10000000, -1234, 10000, 1234, 0, 7, 45, -46, -72 };

	for (int i = 0; i < 14; i++) {
		CHECK(String(nums[i]).to_int() == num[i]);
	}
	CHECK(String("0b1011").to_int() == 1011); // Looks like a binary number, but to_int() handles this as a base-10 number, "b" is just ignored.
	CHECK(String("0B1011").to_int() == 1011);

	CHECK(String("0x1012").to_int() == 1012); // Looks like a hexadecimal number, but to_int() handles this as a base-10 number, "x" is just ignored.
	CHECK(String("0X1012").to_int() == 1012);

	ERR_PRINT_OFF
	CHECK(String("999999999999999999999999999999999999999999999999999999999").to_int() == INT64_MAX); // Too large, largest possible is returned.
	CHECK(String("-999999999999999999999999999999999999999999999999999999999").to_int() == INT64_MIN); // Too small, smallest possible is returned.
	ERR_PRINT_ON
}

TEST_CASE("[String] Hex to integer") {
	static const char *nums[13] = { "0xFFAE", "22", "0", "AADDAD", "0x7FFFFFFFFFFFFFFF", "-0xf", "", "000", "000f", "0xaA", "-ff", "-", "0XFFAE" };
	static const int64_t num[13] = { 0xFFAE, 0x22, 0, 0xAADDAD, 0x7FFFFFFFFFFFFFFF, -0xf, 0, 0, 0xf, 0xaa, -0xff, 0x0, 0xFFAE };

	for (int i = 0; i < 13; i++) {
		CHECK(String(nums[i]).hex_to_int() == num[i]);
	}

	// Invalid hex strings should return 0.
	static const char *invalid_nums[15] = { "qwerty", "QWERTY", "0xqwerty", "0x00qwerty", "qwerty00", "0x", "0x__", "__", "x12", "+", " ff", "ff ", "f f", "+ff", "--0x78" };

	ERR_PRINT_OFF
	for (int i = 0; i < 15; i++) {
		CHECK(String(invalid_nums[i]).hex_to_int() == 0);
	}

	CHECK(String("0xFFFFFFFFFFFFFFFFFFFFFFF").hex_to_int() == INT64_MAX); // Too large, largest possible is returned.
	CHECK(String("-0xFFFFFFFFFFFFFFFFFFFFFFF").hex_to_int() == INT64_MIN); // Too small, smallest possible is returned.
	ERR_PRINT_ON
}

TEST_CASE("[String] Bin to integer") {
	static const char *nums[11] = { "", "0", "0b0", "0b1", "0b", "1", "0b1010", "-0b11", "-1010", "0b0111111111111111111111111111111111111111111111111111111111111111", "0B1010" };
	static const int64_t num[11] = { 0, 0, 0, 1, 0, 1, 10, -3, -10, 0x7FFFFFFFFFFFFFFF, 10 };

	for (int i = 0; i < 11; i++) {
		CHECK(String(nums[i]).bin_to_int() == num[i]);
	}

	// Invalid bin strings should return 0. The long "0x11...11" is just too long for a 64 bit int.
	static const char *invalid_nums[16] = { "qwerty", "QWERTY", "0bqwerty", "0b00qwerty", "qwerty00", "0x__", "0b__", "__", "b12", "+", "-", "0x12ab", " 11", "11 ", "1 1", "--0b11" };

	for (int i = 0; i < 16; i++) {
		CHECK(String(invalid_nums[i]).bin_to_int() == 0);
	}

	ERR_PRINT_OFF
	CHECK(String("0b111111111111111111111111111111111111111111111111111111111111111111111111111111111").bin_to_int() == INT64_MAX); // Too large, largest possible is returned.
	CHECK(String("-0b111111111111111111111111111111111111111111111111111111111111111111111111111111111").bin_to_int() == INT64_MIN); // Too small, smallest possible is returned.
	ERR_PRINT_ON
}

TEST_CASE("[String] String to float") {
	static const char *nums[12] = { "-12348298412.2", "0.05", "2.0002", " -0.0001", "0", "000", "123", "0.0", "000.000", "000.007", "234__", "3..14" };
	static const double num[12] = { -12348298412.2, 0.05, 2.0002, -0.0001, 0.0, 0.0, 123.0, 0.0, 0.0, 0.007, 234.0, 3.0 };

	for (int i = 0; i < 12; i++) {
		CHECK(!(Math::abs(String(nums[i]).to_float() - num[i]) > 0.00001));
	}

	// Invalid float strings should return 0.
	static const char *invalid_nums[6] = { "qwerty", "qwerty123", "0xffff", "0b1010", "--3.13", "__345" };

	for (int i = 0; i < 6; i++) {
		CHECK(String(invalid_nums[i]).to_float() == 0);
	}

	// Very large exponents.
	CHECK(String("1e308").to_float() == 1e308);
	CHECK(String("-1e308").to_float() == -1e308);

	// Exponent is so high that value is INFINITY/-INFINITY.
	CHECK(String("1e309").to_float() == Math::INF);
	CHECK(String("1e511").to_float() == Math::INF);
	CHECK(String("-1e309").to_float() == -Math::INF);
	CHECK(String("-1e511").to_float() == -Math::INF);

	// Exponent is so high that a warning message is printed. Value is INFINITY/-INFINITY.
	ERR_PRINT_OFF
	CHECK(String("1e512").to_float() == Math::INF);
	CHECK(String("-1e512").to_float() == -Math::INF);
	ERR_PRINT_ON
}

TEST_CASE("[String] Slicing") {
	String s = "Mars,Jupiter,Saturn,Uranus";
	const char *slices[4] = { "Mars", "Jupiter", "Saturn", "Uranus" };
	MULTICHECK_GET_SLICE(s, ",", slices);
}

TEST_CASE("[String] Begins with") {
	// Test cases for true:
	MULTICHECK_STRING_EQ(String("res://foobar"), begins_with, "res://", true);
	MULTICHECK_STRING_EQ(String("abc"), begins_with, "abc", true);
	MULTICHECK_STRING_EQ(String("abc"), begins_with, "", true);
	MULTICHECK_STRING_EQ(String(""), begins_with, "", true);

	// Test cases for false:
	MULTICHECK_STRING_EQ(String("res"), begins_with, "res://", false);
	MULTICHECK_STRING_EQ(String("abcdef"), begins_with, "foo", false);
	MULTICHECK_STRING_EQ(String("abc"), begins_with, "ax", false);
	MULTICHECK_STRING_EQ(String(""), begins_with, "abc", false);

	// Test "const char *" version also with nullptr.
	String s("foo");
	bool state = s.begins_with(nullptr) == false;
	CHECK_MESSAGE(state, "nullptr check failed");

	String empty("");
	state = empty.begins_with(nullptr) == false;
	CHECK_MESSAGE(state, "nullptr check with empty string failed");
}

TEST_CASE("[String] Ends with") {
	// Test cases for true:
	MULTICHECK_STRING_EQ(String("res://foobar"), ends_with, "foobar", true);
	MULTICHECK_STRING_EQ(String("abc"), ends_with, "abc", true);
	MULTICHECK_STRING_EQ(String("abc"), ends_with, "", true);
	MULTICHECK_STRING_EQ(String(""), ends_with, "", true);

	// Test cases for false:
	MULTICHECK_STRING_EQ(String("res"), ends_with, "res://", false);
	MULTICHECK_STRING_EQ(String("abcdef"), ends_with, "foo", false);
	MULTICHECK_STRING_EQ(String("abc"), ends_with, "ax", false);
	MULTICHECK_STRING_EQ(String(""), ends_with, "abc", false);

	// Test "const char *" version also with nullptr.
	String s("foo");
	bool state = s.ends_with(nullptr) == false;
	CHECK_MESSAGE(state, "nullptr check failed");

	String empty("");
	state = empty.ends_with(nullptr) == false;
	CHECK_MESSAGE(state, "nullptr check with empty string failed");
}

TEST_CASE("[String] Splitting") {
	{
		const String s = "Mars,Jupiter,Saturn,Uranus";

		const char *slices_l[3] = { "Mars", "Jupiter", "Saturn,Uranus" };
		MULTICHECK_SPLIT(s, split, ",", true, 2, slices_l, 3);

		const char *slices_r[3] = { "Mars,Jupiter", "Saturn", "Uranus" };
		MULTICHECK_SPLIT(s, rsplit, ",", true, 2, slices_r, 3);
	}

	{
		const String s = "test";
		const char *slices[4] = { "t", "e", "s", "t" };
		MULTICHECK_SPLIT(s, split, "", true, 0, slices, 4);
	}

	{
		const String s = "";
		const char *slices[1] = { "" };
		MULTICHECK_SPLIT(s, split, "", true, 0, slices, 1);
		MULTICHECK_SPLIT(s, split, "", false, 0, slices, 0);
	}

	{
		const String s = "Mars Jupiter Saturn Uranus";
		const char *slices[4] = { "Mars", "Jupiter", "Saturn", "Uranus" };
		Vector<String> l = s.split_spaces();
		for (int i = 0; i < l.size(); i++) {
			CHECK(l[i] == slices[i]);
		}
	}
	{
		const String s = "Mars Jupiter Saturn Uranus";
		const char *slices[2] = { "Mars", "Jupiter Saturn Uranus" };
		Vector<String> l = s.split_spaces(1);
		for (int i = 0; i < l.size(); i++) {
			CHECK(l[i] == slices[i]);
		}
	}

	{
		const String s = "1.2;2.3 4.5";
		const double slices[3] = { 1.2, 2.3, 4.5 };

		const Vector<double> d_arr = s.split_floats(";");
		CHECK(d_arr.size() == 2);
		for (int i = 0; i < d_arr.size(); i++) {
			CHECK(Math::abs(d_arr[i] - slices[i]) <= 0.00001);
		}

		const Vector<String> keys = { ";", " " };
		const Vector<float> f_arr = s.split_floats_mk(keys);
		CHECK(f_arr.size() == 3);
		for (int i = 0; i < f_arr.size(); i++) {
			CHECK(Math::abs(f_arr[i] - slices[i]) <= 0.00001);
		}
	}

	{
		const String s = " -2.0        5";
		const double slices[10] = { 0, -2, 0, 0, 0, 0, 0, 0, 0, 5 };

		const Vector<double> arr = s.split_floats(" ");
		CHECK(arr.size() == 10);
		for (int i = 0; i < arr.size(); i++) {
			CHECK(Math::abs(arr[i] - slices[i]) <= 0.00001);
		}

		const Vector<String> keys = { ";", " " };
		const Vector<float> mk = s.split_floats_mk(keys);
		CHECK(mk.size() == 10);
		for (int i = 0; i < mk.size(); i++) {
			CHECK(mk[i] == slices[i]);
		}
	}

	{
		const String s = "1;2 4";
		const int slices[3] = { 1, 2, 4 };

		const Vector<int> arr = s.split_ints(";");
		CHECK(arr.size() == 2);
		for (int i = 0; i < arr.size(); i++) {
			CHECK(arr[i] == slices[i]);
		}

		const Vector<String> keys = { ";", " " };
		const Vector<int> mk = s.split_ints_mk(keys);
		CHECK(mk.size() == 3);
		for (int i = 0; i < mk.size(); i++) {
			CHECK(mk[i] == slices[i]);
		}
	}
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
	///// Ints

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

	// Negative int left padded with spaces.
	format = "fish %5d frog";
	args.clear();
	args.push_back(-5);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish    -5 frog"));

	// Negative int left padded with zeros.
	format = "fish %05d frog";
	args.clear();
	args.push_back(-5);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish -0005 frog"));

	// Negative int right padded with spaces.
	format = "fish %-5d frog";
	args.clear();
	args.push_back(-5);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish -5    frog"));

	// Negative int right padded with zeros. (0 ignored)
	format = "fish %-05d frog";
	args.clear();
	args.push_back(-5);
	ERR_PRINT_OFF; // Silence warning about 0 ignored.
	output = format.sprintf(args, &error);
	ERR_PRINT_ON;
	REQUIRE(error == false);
	CHECK(output == String("fish -5    frog"));

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

	// INT64_MIN
	format = "fish %d frog";
	args.clear();
	args.push_back(INT64_MIN);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish -9223372036854775808 frog"));

	// INT64_MIN hex (lower)
	format = "fish %x frog";
	args.clear();
	args.push_back(INT64_MIN);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish -8000000000000000 frog"));

	// INT64_MIN hex (upper)
	format = "fish %X frog";
	args.clear();
	args.push_back(INT64_MIN);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish -8000000000000000 frog"));

	// INT64_MIN octal
	format = "fish %o frog";
	args.clear();
	args.push_back(INT64_MIN);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish -1000000000000000000000 frog"));

	///// Reals

	// Real
	format = "fish %f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish 99.990000 frog"));

	// Real left-padded.
	format = "fish %11f frog";
	args.clear();
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish   99.990000 frog"));

	// Real (infinity) left-padded
	format = "fish %11f frog";
	args.clear();
	args.push_back(Math::INF);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish         inf frog"));

	// Real right-padded.
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

	// Real with sign (negative zero).
	format = "fish %+f frog";
	args.clear();
	args.push_back(-0.0);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish -0.000000 frog"));

	// Real with sign (positive zero).
	format = "fish %+f frog";
	args.clear();
	args.push_back(0.0);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish +0.000000 frog"));

	// Real with 1 decimal.
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

	// Negative real right padded with zeros. (0 ignored)
	format = "fish %-011f frog";
	args.clear();
	args.push_back(-99.99);
	ERR_PRINT_OFF; // Silence warning about 0 ignored.
	output = format.sprintf(args, &error);
	ERR_PRINT_ON;
	REQUIRE(error == false);
	CHECK(output == String("fish -99.990000  frog"));

	///// Vectors

	// Vector2
	format = "fish %v frog";
	args.clear();
	args.push_back(Variant(Vector2(19.99, 1.00)));
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish (19.990000, 1.000000) frog"));

	// Vector3
	format = "fish %v frog";
	args.clear();
	args.push_back(Variant(Vector3(19.99, 1.00, -2.05)));
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish (19.990000, 1.000000, -2.050000) frog"));

	// Vector4
	format = "fish %v frog";
	args.clear();
	args.push_back(Variant(Vector4(19.99, 1.00, -2.05, 5.5)));
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish (19.990000, 1.000000, -2.050000, 5.500000) frog"));

	// Vector with negative values.
	format = "fish %v frog";
	args.clear();
	args.push_back(Variant(Vector2(-19.99, -1.00)));
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish (-19.990000, -1.000000) frog"));

	// Vector left-padded.
	format = "fish %11v frog";
	args.clear();
	args.push_back(Variant(Vector3(19.99, 1.00, -2.05)));
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish (  19.990000,    1.000000,   -2.050000) frog"));

	// Vector left-padded with inf/nan
	format = "fish %11v frog";
	args.clear();
	args.push_back(Variant(Vector2(Math::INF, Math::NaN)));
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish (        inf,         nan) frog"));

	// Vector right-padded.
	format = "fish %-11v frog";
	args.clear();
	args.push_back(Variant(Vector3(19.99, 1.00, -2.05)));
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish (19.990000  , 1.000000   , -2.050000  ) frog"));

	// Vector left-padded with zeros.
	format = "fish %011v frog";
	args.clear();
	args.push_back(Variant(Vector3(19.99, 1.00, -2.05)));
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish (0019.990000, 0001.000000, -002.050000) frog"));

	// Vector given Vector3i.
	format = "fish %v frog";
	args.clear();
	args.push_back(Variant(Vector3i(19, 1, -2)));
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish (19.000000, 1.000000, -2.000000) frog"));

	// Vector with 1 decimal.
	format = "fish %.1v frog";
	args.clear();
	args.push_back(Variant(Vector3(19.99, 1.00, -2.05)));
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish (20.0, 1.0, -2.0) frog"));

	// Vector with 12 decimals.
	format = "fish %.12v frog";
	args.clear();
	args.push_back(Variant(Vector3(19.00, 1.00, -2.00)));
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish (19.000000000000, 1.000000000000, -2.000000000000) frog"));

	// Vector with no decimals.
	format = "fish %.v frog";
	args.clear();
	args.push_back(Variant(Vector3(19.99, 1.00, -2.05)));
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish (20, 1, -2) frog"));

	///// Strings

	// String
	format = "fish %s frog";
	args.clear();
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish cheese frog"));

	// String left-padded.
	format = "fish %10s frog";
	args.clear();
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish     cheese frog"));

	// String right-padded.
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

	// String dynamic width.
	format = "fish %*s frog";
	args.clear();
	args.push_back(10);
	args.push_back("cheese");
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	REQUIRE(output == String("fish     cheese frog"));

	// Int dynamic width.
	format = "fish %*d frog";
	args.clear();
	args.push_back(10);
	args.push_back(99);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	REQUIRE(output == String("fish         99 frog"));

	// Float dynamic width.
	format = "fish %*.*f frog";
	args.clear();
	args.push_back(10);
	args.push_back(3);
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish     99.990 frog"));

	///// Argument indices.
	format = "fish %2$d frog %1$s xx";
	args.clear();
	args.push_back("test");
	args.push_back(5);
	output = format.sprintf(args, &error);
	REQUIRE(error == false);
	CHECK(output == String("fish 5 frog test xx"));

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

	// Bad character in format string.
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

	// * not a number or vector.
	format = "fish %*f frog";
	args.clear();
	args.push_back("cheese");
	args.push_back(99.99);
	output = format.sprintf(args, &error);
	REQUIRE(error);
	CHECK(output == "* wants number or vector");

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
	CHECK(s.pad_decimals(1) == U"10.1");
	CHECK(s.pad_zeros(4) == U"0010.10");
	CHECK(s.pad_zeros(1) == U"10.10");
}

TEST_CASE("[String] is_subsequence_of") {
	String a = "is subsequence of";
	CHECK(String("sub").is_subsequence_of(a));
	CHECK(!String("Sub").is_subsequence_of(a));
	CHECK(String("Sub").is_subsequence_ofn(a));
}

TEST_CASE("[String] is_lowercase") {
	CHECK(String("abcd1234 !@#$%^&*()_-=+,.<>/\\|[]{};':\"`~").is_lowercase());
	CHECK(String("").is_lowercase());
	CHECK(!String("abc_ABC").is_lowercase());
}

TEST_CASE("[String] match") {
	CHECK(String("img1.png").match("*.png"));
	CHECK(!String("img1.jpeg").match("*.png"));
	CHECK(!String("img1.Png").match("*.png"));
	CHECK(String("img1.Png").matchn("*.png"));
}

TEST_CASE("[String] IPVX address to string") {
	IPAddress ip0("2001:0db8:85a3:0000:0000:8a2e:0370:7334");
	CHECK(ip0.is_valid());

	IPAddress ip(0x0123, 0x4567, 0x89ab, 0xcdef, true);
	CHECK(ip.is_valid());

	IPAddress ip2("fe80::52e5:49ff:fe93:1baf");
	CHECK(ip2.is_valid());

	IPAddress ip3("::ffff:192.168.0.1");
	CHECK(ip3.is_valid());

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
	String input = "2D";
	String output = "2d";
	CHECK(input.capitalize() == output);

	input = "2d";
	output = "2d";
	CHECK(input.capitalize() == output);

	input = "2db";
	output = "2 Db";
	CHECK(input.capitalize() == output);

	input = "HTML5 Html5 html5 html_5";
	output = "Html 5 Html 5 Html 5 Html 5";
	CHECK(input.capitalize() == output);

	input = "Node2D Node2d NODE2D NODE_2D node_2d";
	output = "Node 2d Node 2d Node 2d Node 2d Node 2d";
	CHECK(input.capitalize() == output);

	input = "Node2DPosition";
	output = "Node 2d Position";
	CHECK(input.capitalize() == output);

	input = "Number2Digits";
	output = "Number 2 Digits";
	CHECK(input.capitalize() == output);

	input = "bytes2var";
	output = "Bytes 2 Var";
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

	input = "kebab-case";
	output = "Kebab Case";
	CHECK(input.capitalize() == output);

	input = "kebab-kebab-case";
	output = "Kebab Kebab Case";
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

	input = "kebab-case-function( kebab-case-arg )";
	output = "Kebab Case Function( Kebab Case Arg )";
	CHECK(input.capitalize() == output);

	input = "kebab_case_function( kebab_case_arg )";
	output = "Kebab Case Function( Kebab Case Arg )";
	CHECK(input.capitalize() == output);

	input = U"словоСлово_слово слово";
	output = U"Слово Слово Слово Слово";
	CHECK(input.capitalize() == output);

	input = U"λέξηΛέξη_λέξη λέξη";
	output = U"Λέξη Λέξη Λέξη Λέξη";
	CHECK(input.capitalize() == output);

	input = U"բառԲառ_բառ բառ";
	output = U"Բառ Բառ Բառ Բառ";
	CHECK(input.capitalize() == output);
}

struct StringCasesTestCase {
	const char32_t *input;
	const char32_t *camel_case;
	const char32_t *pascal_case;
	const char32_t *snake_case;
	const char32_t *kebab_case;
};

TEST_CASE("[String] Checking case conversion methods") {
	StringCasesTestCase test_cases[] = {
		/* clang-format off */
		{ U"2D",                     U"2d",                   U"2d",                   U"2d",                      U"2d"                      },
		{ U"2d",                     U"2d",                   U"2d",                   U"2d",                      U"2d"                      },
		{ U"2db",                    U"2Db",                  U"2Db",                  U"2_db",                    U"2-db"                    },
		{ U"Vector3",                U"vector3",              U"Vector3",              U"vector_3",                U"vector-3"                },
		{ U"sha256",                 U"sha256",               U"Sha256",               U"sha_256",                 U"sha-256"                 },
		{ U"Node2D",                 U"node2d",               U"Node2d",               U"node_2d",                 U"node-2d"                 },
		{ U"RichTextLabel",          U"richTextLabel",        U"RichTextLabel",        U"rich_text_label",         U"rich-text-label"         },
		{ U"HTML5",                  U"html5",                U"Html5",                U"html_5",                  U"html-5"                  },
		{ U"Node2DPosition",         U"node2dPosition",       U"Node2dPosition",       U"node_2d_position",        U"node-2d-position"        },
		{ U"Number2Digits",          U"number2Digits",        U"Number2Digits",        U"number_2_digits",         U"number-2-digits"         },
		{ U"get_property_list",      U"getPropertyList",      U"GetPropertyList",      U"get_property_list",       U"get-property-list"       },
		{ U"get_camera_2d",          U"getCamera2d",          U"GetCamera2d",          U"get_camera_2d",           U"get-camera-2d"           },
		{ U"_physics_process",       U"physicsProcess",       U"PhysicsProcess",       U"_physics_process",        U"-physics-process"        },
		{ U"bytes2var",              U"bytes2Var",            U"Bytes2Var",            U"bytes_2_var",             U"bytes-2-var"             },
		{ U"linear2db",              U"linear2Db",            U"Linear2Db",            U"linear_2_db",             U"linear-2-db"             },
		{ U"sha256sum",              U"sha256Sum",            U"Sha256Sum",            U"sha_256_sum",             U"sha-256-sum"             },
		{ U"camelCase",              U"camelCase",            U"CamelCase",            U"camel_case",              U"camel-case"              },
		{ U"PascalCase",             U"pascalCase",           U"PascalCase",           U"pascal_case",             U"pascal-case"             },
		{ U"snake_case",             U"snakeCase",            U"SnakeCase",            U"snake_case",              U"snake-case"              },
		{ U"kebab-case",             U"kebabCase",            U"KebabCase",            U"kebab_case",              U"kebab-case"              },
		{ U"Test TEST test",         U"testTestTest",         U"TestTestTest",         U"test_test_test",          U"test-test-test"          },
		{ U"словоСлово_слово слово", U"словоСловоСловоСлово", U"СловоСловоСловоСлово", U"слово_слово_слово_слово", U"слово-слово-слово-слово" },
		{ U"λέξηΛέξη_λέξη λέξη",     U"λέξηΛέξηΛέξηΛέξη",     U"ΛέξηΛέξηΛέξηΛέξη",     U"λέξη_λέξη_λέξη_λέξη",     U"λέξη-λέξη-λέξη-λέξη"     },
		{ U"բառԲառ_բառ բառ",         U"բառԲառԲառԲառ",         U"ԲառԲառԲառԲառ",         U"բառ_բառ_բառ_բառ",         U"բառ-բառ-բառ-բառ"         },
		{ nullptr,                   nullptr,                 nullptr,                 nullptr,                    nullptr                    },
		/* clang-format on */
	};

	int idx = 0;
	while (test_cases[idx].input != nullptr) {
		String input = test_cases[idx].input;
		CHECK(input.to_camel_case() == test_cases[idx].camel_case);
		CHECK(input.to_pascal_case() == test_cases[idx].pascal_case);
		CHECK(input.to_snake_case() == test_cases[idx].snake_case);
		CHECK(input.to_kebab_case() == test_cases[idx].kebab_case);
		idx++;
	}
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
#define STRIP_TEST(x) \
	{ \
		bool success = x; \
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

TEST_CASE("[String] Ensuring empty string into extend_utf8 passes empty string") {
	String empty;
	CHECK(empty.append_utf8(nullptr, -1) == ERR_INVALID_DATA);
}

TEST_CASE("[String] Cyrillic to_lower()") {
	String upper = U"АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ";
	String lower = U"абвгдеёжзийклмнопрстуфхцчшщъыьэюя";

	String test = upper.to_lower();

	bool state = test == lower;

	CHECK(state);
}

TEST_CASE("[String] Count and countn functionality") {
	String s = String("");
	MULTICHECK_STRING_EQ(s, count, "Test", 0);

	s = "Test";
	MULTICHECK_STRING_EQ(s, count, "", 0);

	s = "Test";
	MULTICHECK_STRING_EQ(s, count, "test", 0);

	s = "Test";
	MULTICHECK_STRING_EQ(s, count, "TEST", 0);

	s = "TEST";
	MULTICHECK_STRING_EQ(s, count, "TEST", 1);

	s = "Test";
	MULTICHECK_STRING_EQ(s, count, "Test", 1);

	s = "aTest";
	MULTICHECK_STRING_EQ(s, count, "Test", 1);

	s = "Testa";
	MULTICHECK_STRING_EQ(s, count, "Test", 1);

	s = "TestTestTest";
	MULTICHECK_STRING_EQ(s, count, "Test", 3);

	s = "TestTestTest";
	MULTICHECK_STRING_EQ(s, count, "TestTest", 1);

	s = "TestGodotTestGodotTestGodot";
	MULTICHECK_STRING_EQ(s, count, "Test", 3);

	s = "TestTestTestTest";
	MULTICHECK_STRING_INT_INT_EQ(s, count, "Test", 4, 8, 1);

	s = "TestTestTestTest";
	MULTICHECK_STRING_INT_INT_EQ(s, count, "Test", 4, 12, 2);

	s = "TestTestTestTest";
	MULTICHECK_STRING_INT_INT_EQ(s, count, "Test", 4, 16, 3);

	s = "TestTestTestTest";
	MULTICHECK_STRING_INT_EQ(s, count, "Test", 4, 3);

	s = "Test";
	MULTICHECK_STRING_EQ(s, countn, "test", 1);

	s = "Test";
	MULTICHECK_STRING_EQ(s, countn, "TEST", 1);

	s = "testTest-Testatest";
	MULTICHECK_STRING_EQ(s, countn, "tEst", 4);

	s = "testTest-TeStatest";
	MULTICHECK_STRING_INT_INT_EQ(s, countn, "tEsT", 4, 16, 2);
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

TEST_CASE("[String] indent") {
	static const char *input[] = {
		"",
		"aaa\nbbb",
		"\tcontains\n\tindent",
		"empty\n\nline",
	};
	static const char *expected[] = {
		"",
		"\taaa\n\tbbb",
		"\t\tcontains\n\t\tindent",
		"\tempty\n\n\tline",
	};

	for (int i = 0; i < 3; i++) {
		CHECK(String(input[i]).indent("\t") == expected[i]);
	}
}

TEST_CASE("[String] dedent") {
	String s = "      aaa\n    bbb";
	String t = "aaa\nbbb";
	CHECK(s.dedent() == t);
}

TEST_CASE("[String] Path functions") {
	static const char *path[8] = { "C:\\Godot\\project\\test.tscn", "/Godot/project/test.xscn", "../Godot/project/test.scn", "Godot\\test.doc", "C:\\test.", "res://test", "user://test", "/.test" };
	static const char *base_dir[8] = { "C:\\Godot\\project", "/Godot/project", "../Godot/project", "Godot", "C:\\", "res://", "user://", "/" };
	static const char *base_name[8] = { "C:\\Godot\\project\\test", "/Godot/project/test", "../Godot/project/test", "Godot\\test", "C:\\test", "res://test", "user://test", "/" };
	static const char *ext[8] = { "tscn", "xscn", "scn", "doc", "", "", "", "test" };
	static const char *file[8] = { "test.tscn", "test.xscn", "test.scn", "test.doc", "test.", "test", "test", ".test" };
	static const char *simplified[8] = { "C:/Godot/project/test.tscn", "/Godot/project/test.xscn", "../Godot/project/test.scn", "Godot/test.doc", "C:/test.", "res://test", "user://test", "/.test" };
	static const bool abs[8] = { true, true, false, false, true, true, true, true };

	for (int i = 0; i < 8; i++) {
		CHECK(String(path[i]).get_base_dir() == base_dir[i]);
		CHECK(String(path[i]).get_basename() == base_name[i]);
		CHECK(String(path[i]).get_extension() == ext[i]);
		CHECK(String(path[i]).get_file() == file[i]);
		CHECK(String(path[i]).is_absolute_path() == abs[i]);
		CHECK(String(path[i]).is_relative_path() != abs[i]);
		CHECK(String(path[i]).simplify_path() == String(simplified[i]));
		CHECK(String(path[i]).simplify_path().get_base_dir().path_join(file[i]) == String(path[i]).simplify_path());
	}

	CHECK(String("res://test.png").has_extension("png"));
	CHECK(String("res://test.PNG").has_extension("png"));
	CHECK_FALSE(String("res://test.png").has_extension("jpg"));
	CHECK_FALSE(String("res://test.png/README").has_extension("png"));
	CHECK_FALSE(String("res://test.").has_extension("png"));
	CHECK_FALSE(String("res://test").has_extension("png"));

	static const char *file_name[3] = { "test.tscn", "test://.xscn", "?tes*t.scn" };
	static const bool valid[3] = { true, false, false };
	for (int i = 0; i < 3; i++) {
		CHECK(String(file_name[i]).is_valid_filename() == valid[i]);
	}

	CHECK(String("res://texture.png") == String("res://folder/../folder/../texture.png").simplify_path());
	CHECK(String("res://texture.png") == String("res://folder/sub/../../texture.png").simplify_path());
	CHECK(String("res://../../texture.png") == String("res://../../texture.png").simplify_path());
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
	String x4 = U"file+name";

	CHECK(x1.uri_decode() == x2);
	CHECK(x1.uri_decode() == x3);
	CHECK((x1 + x3).uri_decode() == (x2 + x3)); // Mixed unicode and URL encoded string, e.g. GTK+ bookmark.
	CHECK(x2.uri_encode() == x1);
	CHECK(x3.uri_encode() == x1);

	CHECK(s.uri_encode() == t);
	CHECK(t.uri_decode() == s);
	CHECK(x4.uri_file_decode() == x4);
	CHECK(x4.uri_decode() == U"file name");
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
	MULTICHECK_STRING_EQ(s, trim_prefix, "aaa", "Testbbb");
	MULTICHECK_STRING_EQ(s, trim_prefix, "Test", s);
	MULTICHECK_STRING_EQ(s, trim_prefix, "", s);
	MULTICHECK_STRING_EQ(s, trim_prefix, "aaaTestbbb", "");
	MULTICHECK_STRING_EQ(s, trim_prefix, "bbb", s);
	MULTICHECK_STRING_EQ(s, trim_prefix, "AAA", s);

	MULTICHECK_STRING_EQ(s, trim_suffix, "bbb", "aaaTest");
	MULTICHECK_STRING_EQ(s, trim_suffix, "Test", s);
	MULTICHECK_STRING_EQ(s, trim_suffix, "", s);
	MULTICHECK_STRING_EQ(s, trim_suffix, "aaaTestbbb", "");
	MULTICHECK_STRING_EQ(s, trim_suffix, "aaa", s);
	MULTICHECK_STRING_EQ(s, trim_suffix, "BBB", s);
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

TEST_CASE("[String] Reverse") {
	String s = "Abcd";
	CHECK(s.reverse() == "dcbA");
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
	String comma = ", ";
	String empty = "";
	Vector<String> parts;

	CHECK(comma.join(parts) == "");
	CHECK(empty.join(parts) == "");

	parts.push_back("One");
	CHECK(comma.join(parts) == "One");
	CHECK(empty.join(parts) == "One");

	parts.push_back("B");
	parts.push_back("C");
	CHECK(comma.join(parts) == "One, B, C");
	CHECK(empty.join(parts) == "OneBC");

	parts.push_back("");
	CHECK(comma.join(parts) == "One, B, C, ");
	CHECK(empty.join(parts) == "OneBC");
}

TEST_CASE("[String] Is_*") {
	static const char *data[] = { "-30", "100", "10.1", "10,1", "1e2", "1e-2", "1e2e3", "0xAB", "AB", "Test1", "1Test", "Test*1", "文字", "1E2", "1E-2" };
	static bool isnum[] = { true, true, true, false, false, false, false, false, false, false, false, false, false, false, false };
	static bool isint[] = { true, true, false, false, false, false, false, false, false, false, false, false, false, false, false };
	static bool ishex[] = { true, true, false, false, true, false, true, false, true, false, false, false, false, true, false };
	static bool ishex_p[] = { false, false, false, false, false, false, false, true, false, false, false, false, false, false, false };
	static bool isflt[] = { true, true, true, false, true, true, false, false, false, false, false, false, false, true, true };
	static bool isaid[] = { false, false, false, false, false, false, false, false, true, true, false, false, false, false, false };
	static bool isuid[] = { false, false, false, false, false, false, false, false, true, true, false, false, true, false, false };
	for (unsigned int i = 0; i < std_size(data); i++) {
		String s = String::utf8(data[i]);
		CHECK(s.is_numeric() == isnum[i]);
		CHECK(s.is_valid_int() == isint[i]);
		CHECK(s.is_valid_hex_number(false) == ishex[i]);
		CHECK(s.is_valid_hex_number(true) == ishex_p[i]);
		CHECK(s.is_valid_float() == isflt[i]);
		CHECK(s.is_valid_ascii_identifier() == isaid[i]);
		CHECK(s.is_valid_unicode_identifier() == isuid[i]);
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

	String name_with_kana = U"Name with kana ゴドツ";
	CHECK(name_with_kana.validate_node_name() == U"Name with kana ゴドツ");

	String name_with_invalid_chars = "Name with invalid characters :.@%removed!";
	CHECK(name_with_invalid_chars.validate_node_name() == "Name with invalid characters ____removed!");
}

TEST_CASE("[String] validate_ascii_identifier") {
	String empty_string;
	CHECK(empty_string.validate_ascii_identifier() == "_");

	String numeric_only = "12345";
	CHECK(numeric_only.validate_ascii_identifier() == "_12345");

	String name_with_spaces = "Name with spaces";
	CHECK(name_with_spaces.validate_ascii_identifier() == "Name_with_spaces");

	String name_with_invalid_chars = U"Invalid characters:@*#&世界";
	CHECK(name_with_invalid_chars.validate_ascii_identifier() == "Invalid_characters_______");
}

TEST_CASE("[String] validate_unicode_identifier") {
	String empty_string;
	CHECK(empty_string.validate_unicode_identifier() == "_");

	String numeric_only = "12345";
	CHECK(numeric_only.validate_unicode_identifier() == "_12345");

	String name_with_spaces = "Name with spaces";
	CHECK(name_with_spaces.validate_unicode_identifier() == "Name_with_spaces");

	String name_with_invalid_chars = U"Invalid characters:@*#&世界";
	CHECK(name_with_invalid_chars.validate_unicode_identifier() == U"Invalid_characters_____世界");
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

TEST_CASE("[String][URL] Parse URL") {
#define CHECK_URL(m_url_to_parse, m_expected_schema, m_expected_host, m_expected_port, m_expected_path, m_expected_fragment, m_expected_error) \
	if (true) { \
		int port; \
		String url(m_url_to_parse), schema, host, path, fragment; \
\
		CHECK_EQ(url.parse_url(schema, host, port, path, fragment), m_expected_error); \
		CHECK_EQ(schema, m_expected_schema); \
		CHECK_EQ(host, m_expected_host); \
		CHECK_EQ(path, m_expected_path); \
		CHECK_EQ(fragment, m_expected_fragment); \
		CHECK_EQ(port, m_expected_port); \
	} else \
		((void)0)

	// All elements.
	CHECK_URL("https://www.example.com:8080/path/to/file.html#fragment", "https://", "www.example.com", 8080, "/path/to/file.html", "fragment", Error::OK);

	// Valid URLs.
	CHECK_URL("https://godotengine.org", "https://", "godotengine.org", 0, "", "", Error::OK);
	CHECK_URL("https://godotengine.org/", "https://", "godotengine.org", 0, "/", "", Error::OK);
	CHECK_URL("godotengine.org/", "", "godotengine.org", 0, "/", "", Error::OK);
	CHECK_URL("HTTPS://godotengine.org/", "https://", "godotengine.org", 0, "/", "", Error::OK);
	CHECK_URL("https://GODOTENGINE.ORG/", "https://", "godotengine.org", 0, "/", "", Error::OK);
	CHECK_URL("http://godotengine.org", "http://", "godotengine.org", 0, "", "", Error::OK);
	CHECK_URL("https://godotengine.org:8080", "https://", "godotengine.org", 8080, "", "", Error::OK);
	CHECK_URL("https://godotengine.org/blog", "https://", "godotengine.org", 0, "/blog", "", Error::OK);
	CHECK_URL("https://godotengine.org/blog/", "https://", "godotengine.org", 0, "/blog/", "", Error::OK);
	CHECK_URL("https://docs.godotengine.org/en/stable", "https://", "docs.godotengine.org", 0, "/en/stable", "", Error::OK);
	CHECK_URL("https://docs.godotengine.org/en/stable/", "https://", "docs.godotengine.org", 0, "/en/stable/", "", Error::OK);
	CHECK_URL("https://me:secret@godotengine.org", "https://", "godotengine.org", 0, "", "", Error::OK);
	CHECK_URL("https://[FEDC:BA98:7654:3210:FEDC:BA98:7654:3210]/ipv6", "https://", "fedc:ba98:7654:3210:fedc:ba98:7654:3210", 0, "/ipv6", "", Error::OK);

	// Scheme vs Fragment.
	CHECK_URL("google.com/#goto=http://redirect_url/", "", "google.com", 0, "/", "goto=http://redirect_url/", Error::OK);

	// Invalid URLs.

	// Invalid Scheme.
	CHECK_URL("https_://godotengine.org", "", "https_", 0, "//godotengine.org", "", Error::ERR_INVALID_PARAMETER);

	// Multiple ports.
	CHECK_URL("https://godotengine.org:8080:433", "https://", "", 0, "", "", Error::ERR_INVALID_PARAMETER);
	// Missing ] on literal IPv6.
	CHECK_URL("https://[FEDC:BA98:7654:3210:FEDC:BA98:7654:3210/ipv6", "https://", "", 0, "/ipv6", "", Error::ERR_INVALID_PARAMETER);
	// Missing host.
	CHECK_URL("https:///blog", "https://", "", 0, "/blog", "", Error::ERR_INVALID_PARAMETER);
	// Invalid ports.
	CHECK_URL("https://godotengine.org:notaport", "https://", "godotengine.org", 0, "", "", Error::ERR_INVALID_PARAMETER);
	CHECK_URL("https://godotengine.org:-8080", "https://", "godotengine.org", -8080, "", "", Error::ERR_INVALID_PARAMETER);
	CHECK_URL("https://godotengine.org:88888", "https://", "godotengine.org", 88888, "", "", Error::ERR_INVALID_PARAMETER);

#undef CHECK_URL
}

TEST_CASE("[String] IDNA encode: ASCII-only domain is a no-op") {
	CHECK(String("godotengine.org").idna_encode() == "godotengine.org");
	CHECK(String("sub.example.com").idna_encode() == "sub.example.com");
	CHECK(String("localhost").idna_encode() == "localhost");
	CHECK(String("192.168.1.1").idna_encode() == "192.168.1.1");
}

TEST_CASE("[String] IDNA encode: single non-ASCII label") {
	// räksmörgås  ->  xn--rksmrgs-5wao1o
	CHECK(String(U"r\u00e4ksm\u00f6rg\u00e5s").idna_encode() == "xn--rksmrgs-5wao1o");
	// lüfter  ->  xn--lfter-kva
	CHECK(String(U"l\u00fcfter").idna_encode() == "xn--lfter-kva");
}

TEST_CASE("[String] IDNA encode: mixed domain with one non-ASCII label") {
	// räksmörgås.josefsson.org -> xn--rksmrgs-5wao1o.josefsson.org
	CHECK(String(U"r\u00e4ksm\u00f6rg\u00e5s.josefsson.org").idna_encode() == "xn--rksmrgs-5wao1o.josefsson.org");
	// something.lüfter  ->  something.xn--lfter-kva
	CHECK(String(U"something.l\u00fcfter").idna_encode() == "something.xn--lfter-kva");
}

TEST_CASE("[String] IDNA encode: multiple non-ASCII labels") {
	// bücher.münchen.de  ->  xn--bcher-kva.xn--mnchen-3ya.de
	CHECK(String(U"b\u00fccher.m\u00fcnchen.de").idna_encode() == "xn--bcher-kva.xn--mnchen-3ya.de");
}

TEST_CASE("[String] IDNA encode: label with mixed ASCII and non-ASCII") {
	// résumé  ->  xn--rsum-bpad
	CHECK(String(U"r\u00e9sum\u00e9").idna_encode() == "xn--rsum-bpad");
}

TEST_CASE("[String] IDNA encode: RFC 3492 test vectors") {
	// All examples taken from RFC 3492 Appendix B.
	// The ACE prefix "xn--" is included in the expected output.

	// (A) Arabic (Egyptian)
	CHECK(String(U"\u0644\u064A\u0647\u0645\u0627\u0628\u062A\u0643\u0644\u0645\u0648\u0634\u0639\u0631\u0628\u064A\u061F").idna_encode() == "xn--egbpdaj6bu4bxfgehfvwxn");

	// (B) Chinese (simplified)
	CHECK(String(U"\u4ED6\u4EEC\u4E3A\u4EC0\u4E48\u4E0D\u8BF4\u4E2D\u6587").idna_encode() == "xn--ihqwcrb4cv8a8dqg056pqjye");

	// (C) Chinese (traditional)
	CHECK(String(U"\u4ED6\u5011\u7232\u4EC0\u9EBD\u4E0D\u8AAA\u4E2D\u6587").idna_encode() == "xn--ihqwctvzc91f659drss3x8bo0yb");

	// (D) Czech: Pro<ccaron>prost<ecaron>nemluv<iacute><ccaron>esky
	CHECK(String(U"Pro\u010Dprost\u011Bnemluv\u00ED\u010Desky").idna_encode() == "xn--Proprostnemluvesky-uyb24dma41a");

	// (E) Hebrew
	CHECK(String(U"\u05DC\u05DE\u05D4\u05D4\u05DD\u05E4\u05E9\u05D5\u05D8\u05DC\u05D0\u05DE\u05D3\u05D1\u05E8\u05D9\u05DD\u05E2\u05D1\u05E8\u05D9\u05EA").idna_encode() == "xn--4dbcagdahymbxekheh6e0a7fei0b");

	// (F) Hindi (Devanagari)
	CHECK(String(U"\u092F\u0939\u0932\u094B\u0917\u0939\u093F\u0928\u094D\u0926\u0940\u0915\u094D\u092F\u094B\u0902\u0928\u0939\u0940\u0902\u092C\u094B\u0932\u0938\u0915\u0924\u0947\u0939\u0948\u0902").idna_encode() == "xn--i1baa7eci9glrd9b2ae1bj0hfcgg6iyaf8o0a1dig0cd");

	// (G) Japanese (kanji and hiragana)
	CHECK(String(U"\u306A\u305C\u307F\u3093\u306A\u65E5\u672C\u8A9E\u3092\u8A71\u3057\u3066\u304F\u308C\u306A\u3044\u306E\u304B").idna_encode() == "xn--n8jok5ay5dzabd5bym9f0cm5685rrjetr6pdxa");

	// (H) Korean (Hangul syllables)
	CHECK(String(U"\uC138\uACC4\uC758\uBAA8\uB4E0\uC0AC\uB78C\uB4E4\uC774\uD55C\uAD6D\uC5B4\uB97C\uC774\uD574\uD55C\uB2E4\uBA74\uC5BC\uB9C8\uB098\uC88B\uC744\uAE4C").idna_encode() == "xn--989aomsvi5e83db1d2a355cv1e0vak1dwrv93d5xbh15a0dt30a5jpsd879ccm6fea98c");

	// (I) Russian (Cyrillic)
	// Technically not the same comparison string as in the RFC. It uses 'b1abfaaepdrnnbgefbaDotcwatmq2g4l' instead of 'b1abfaaepdrnnbgefbadotcwatmq2g4l' (upper case D in the middle).
	// This shouldn't be an issue tho, since the *decoder* is case insensitive by spec.
	CHECK(String(U"\u043F\u043E\u0447\u0435\u043C\u0443\u0436\u0435\u043E\u043D\u0438\u043D\u0435\u0433\u043E\u0432\u043E\u0440\u044F\u0442\u043F\u043E\u0440\u0443\u0441\u0441\u043A\u0438").idna_encode() == "xn--b1abfaaepdrnnbgefbadotcwatmq2g4l");

	// (J) Spanish: Porqu<eacute>nopuedensimplementehablarenEspa<ntilde>ol
	CHECK(String(U"Porqu\u00E9nopuedensimplementehablarenEspa\u00F1ol").idna_encode() == "xn--PorqunopuedensimplementehablarenEspaol-fmd56a");

	// (K) Vietnamese
	CHECK(String(U"T\u1EA1isaoh\u1ECDkh\u00F4ngth\u1EC3ch\u1EC9n\u00F3iti\u1EBFngVi\u1EC7t").idna_encode() == "xn--TisaohkhngthchnitingVit-kjcr8268qyxafd2f1b9g");

	// (L) 3<nen>B<gumi><kinpachi><sensei>
	CHECK(String(U"3\u5E74B\u7D44\u91D1\u516B\u5148\u751F").idna_encode() == "xn--3B-ww4c5e180e575a65lsy2b");

	// (M) <amuro><namie>-with-SUPER-MONKEYS
	CHECK(String(U"\u5B89\u5BA4\u5948\u7F8E\u6075-with-SUPER-MONKEYS").idna_encode() == "xn---with-SUPER-MONKEYS-pc58ag80a8qai00g7n9n");

	// (N) Hello-Another-Way-<sorezore><no><basho>
	CHECK(String(U"Hello-Another-Way-\u305D\u308C\u305E\u308C\u306E\u5834\u6240").idna_encode() == "xn--Hello-Another-Way--fc4qua05auwb3674vfr0b");

	// (O) <hitotsu><yane><no><shita>2
	CHECK(String(U"\u3072\u3068\u3064\u5C4B\u6839\u306E\u4E0B\u0032").idna_encode() == "xn--2-u9tlzr9756bt3uc0v");

	// (P) Maji<de>Koi<suru>5<byou><mae>
	CHECK(String(U"Maji\u3067Koi\u3059\u308B5\u79D2\u524D").idna_encode() == "xn--MajiKoi5-783gue6qz075azm5e");

	// (Q) <pafii>de<runba>
	CHECK(String(U"\u30D1\u30D5\u30A3\u30FCde\u30EB\u30F3\u30D0").idna_encode() == "xn--de-jg4avhby1noc0d");

	// (R) <sono><supiido><de>
	CHECK(String(U"\u305D\u306E\u30B9\u30D4\u30FC\u30C9\u3067").idna_encode() == "xn--d9juau41awczczp");
}

TEST_CASE("[String] IDNA decode: ASCII-only domain is a no-op") {
	CHECK(String("godotengine.org").idna_decode() == "godotengine.org");
	CHECK(String("sub.example.com").idna_decode() == "sub.example.com");
	CHECK(String("localhost").idna_decode() == "localhost");
	CHECK(String("192.168.1.1").idna_decode() == "192.168.1.1");
}

TEST_CASE("[String] IDNA decode: single non-ASCII label") {
	// xn--rksmrgs-5wao1o  ->  räksmörgås
	CHECK(String("xn--rksmrgs-5wao1o").idna_decode() == U"r\u00e4ksm\u00f6rg\u00e5s");
	// xn--lfter-kva  ->  lüfter
	CHECK(String("xn--lfter-kva").idna_decode() == U"l\u00fcfter");
}

TEST_CASE("[String] IDNA decode: mixed domain with one non-ASCII label") {
	// xn--rksmrgs-5wao1o.josefsson.org  ->  räksmörgås.josefsson.org
	CHECK(String("xn--rksmrgs-5wao1o.josefsson.org").idna_decode() == U"r\u00e4ksm\u00f6rg\u00e5s.josefsson.org");
	// something.xn--lfter-kva  ->  something.lüfter
	CHECK(String("something.xn--lfter-kva").idna_decode() == U"something.l\u00fcfter");
}

TEST_CASE("[String] IDNA decode: multiple non-ASCII labels") {
	// xn--bcher-kva.xn--mnchen-3ya.de  ->  bücher.münchen.de
	CHECK(String("xn--bcher-kva.xn--mnchen-3ya.de").idna_decode() == U"b\u00fccher.m\u00fcnchen.de");
}

TEST_CASE("[String] IDNA decode: label with mixed ASCII and non-ASCII") {
	// xn--rsum-bpad  ->  résumé
	CHECK(String("xn--rsum-bpad").idna_decode() == U"r\u00e9sum\u00e9");
}

TEST_CASE("[String] IDNA decode: RFC 3492 test vectors") {
	// All examples taken from RFC 3492 Appendix B.
	// The ACE prefix "xn--" is included in the input.

	// (A) Arabic (Egyptian)
	CHECK(String("xn--egbpdaj6bu4bxfgehfvwxn").idna_decode() == U"\u0644\u064A\u0647\u0645\u0627\u0628\u062A\u0643\u0644\u0645\u0648\u0634\u0639\u0631\u0628\u064A\u061F");

	// (B) Chinese (simplified)
	CHECK(String("xn--ihqwcrb4cv8a8dqg056pqjye").idna_decode() == U"\u4ED6\u4EEC\u4E3A\u4EC0\u4E48\u4E0D\u8BF4\u4E2D\u6587");

	// (C) Chinese (traditional)
	CHECK(String("xn--ihqwctvzc91f659drss3x8bo0yb").idna_decode() == U"\u4ED6\u5011\u7232\u4EC0\u9EBD\u4E0D\u8AAA\u4E2D\u6587");

	// (D) Czech: Pro<ccaron>prost<ecaron>nemluv<iacute><ccaron>esky
	CHECK(String("xn--Proprostnemluvesky-uyb24dma41a").idna_decode() == U"Pro\u010Dprost\u011Bnemluv\u00ED\u010Desky");

	// (E) Hebrew
	CHECK(String("xn--4dbcagdahymbxekheh6e0a7fei0b").idna_decode() == U"\u05DC\u05DE\u05D4\u05D4\u05DD\u05E4\u05E9\u05D5\u05D8\u05DC\u05D0\u05DE\u05D3\u05D1\u05E8\u05D9\u05DD\u05E2\u05D1\u05E8\u05D9\u05EA");

	// (F) Hindi (Devanagari)
	CHECK(String("xn--i1baa7eci9glrd9b2ae1bj0hfcgg6iyaf8o0a1dig0cd").idna_decode() == U"\u092F\u0939\u0932\u094B\u0917\u0939\u093F\u0928\u094D\u0926\u0940\u0915\u094D\u092F\u094B\u0902\u0928\u0939\u0940\u0902\u092C\u094B\u0932\u0938\u0915\u0924\u0947\u0939\u0948\u0902");

	// (G) Japanese (kanji and hiragana)
	CHECK(String("xn--n8jok5ay5dzabd5bym9f0cm5685rrjetr6pdxa").idna_decode() == U"\u306A\u305C\u307F\u3093\u306A\u65E5\u672C\u8A9E\u3092\u8A71\u3057\u3066\u304F\u308C\u306A\u3044\u306E\u304B");

	// (H) Korean (Hangul syllables)
	CHECK(String("xn--989aomsvi5e83db1d2a355cv1e0vak1dwrv93d5xbh15a0dt30a5jpsd879ccm6fea98c").idna_decode() == U"\uC138\uACC4\uC758\uBAA8\uB4E0\uC0AC\uB78C\uB4E4\uC774\uD55C\uAD6D\uC5B4\uB97C\uC774\uD574\uD55C\uB2E4\uBA74\uC5BC\uB9C8\uB098\uC88B\uC744\uAE4C");

	// (I) Russian (Cyrillic)
	CHECK(String("xn--b1abfaaepdrnnbgefbaDotcwatmq2g4l").idna_decode() == U"\u043F\u043E\u0447\u0435\u043C\u0443\u0436\u0435\u043E\u043D\u0438\u043D\u0435\u0433\u043E\u0432\u043E\u0440\u044F\u0442\u043F\u043E\u0440\u0443\u0441\u0441\u043A\u0438");

	// (J) Spanish: Porqu<eacute>nopuedensimplementehablarenEspa<ntilde>ol
	CHECK(String("xn--PorqunopuedensimplementehablarenEspaol-fmd56a").idna_decode() == U"Porqu\u00E9nopuedensimplementehablarenEspa\u00F1ol");

	// (K) Vietnamese
	CHECK(String("xn--TisaohkhngthchnitingVit-kjcr8268qyxafd2f1b9g").idna_decode() == U"T\u1EA1isaoh\u1ECDkh\u00F4ngth\u1EC3ch\u1EC9n\u00F3iti\u1EBFngVi\u1EC7t");

	// (L) 3<nen>B<gumi><kinpachi><sensei>
	CHECK(String("xn--3B-ww4c5e180e575a65lsy2b").idna_decode() == U"3\u5E74B\u7D44\u91D1\u516B\u5148\u751F");

	// (M) <amuro><namie>-with-SUPER-MONKEYS
	CHECK(String("xn---with-SUPER-MONKEYS-pc58ag80a8qai00g7n9n").idna_decode() == U"\u5B89\u5BA4\u5948\u7F8E\u6075-with-SUPER-MONKEYS");

	// (N) Hello-Another-Way-<sorezore><no><basho>
	CHECK(String("xn--Hello-Another-Way--fc4qua05auwb3674vfr0b").idna_decode() == U"Hello-Another-Way-\u305D\u308C\u305E\u308C\u306E\u5834\u6240");

	// (O) <hitotsu><yane><no><shita>2
	CHECK(String("xn--2-u9tlzr9756bt3uc0v").idna_decode() == U"\u3072\u3068\u3064\u5C4B\u6839\u306E\u4E0B\u0032");

	// (P) Maji<de>Koi<suru>5<byou><mae>
	CHECK(String("xn--MajiKoi5-783gue6qz075azm5e").idna_decode() == U"Maji\u3067Koi\u3059\u308B5\u79D2\u524D");

	// (Q) <pafii>de<runba>
	CHECK(String("xn--de-jg4avhby1noc0d").idna_decode() == U"\u30D1\u30D5\u30A3\u30FCde\u30EB\u30F3\u30D0");

	// (R) <sono><supiido><de>
	CHECK(String("xn--d9juau41awczczp").idna_decode() == U"\u305D\u306E\u30B9\u30D4\u30FC\u30C9\u3067");
}

} // namespace TestString
