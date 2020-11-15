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

#ifndef TEST_GDNATIVE_STRING_H
#define TEST_GDNATIVE_STRING_H

namespace TestGDNativeString {

#include "gdnative/string.h"

#include "tests/test_macros.h"

int u32scmp(const char32_t *l, const char32_t *r) {
	for (; *l == *r && *l && *r; l++, r++)
		;
	return *l - *r;
}

TEST_CASE("[GDNative String] Construct from Latin-1 char string") {
	godot_string s;

	godot_string_new_with_latin1_chars(&s, "Hello");
	CHECK(u32scmp(godot_string_get_data(&s), U"Hello") == 0);
	godot_string_destroy(&s);

	godot_string_new_with_latin1_chars_and_len(&s, "Hello", 3);
	CHECK(u32scmp(godot_string_get_data(&s), U"Hel") == 0);
	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] Construct from wchar_t string") {
	godot_string s;

	godot_string_new_with_wide_chars(&s, L"Give me");
	CHECK(u32scmp(godot_string_get_data(&s), U"Give me") == 0);
	godot_string_destroy(&s);

	godot_string_new_with_wide_chars_and_len(&s, L"Give me", 3);
	CHECK(u32scmp(godot_string_get_data(&s), U"Giv") == 0);
	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] Construct from UTF-8 char string") {
	static const char32_t u32str[] = { 0x0045, 0x0020, 0x304A, 0x360F, 0x3088, 0x3046, 0x1F3A4, 0 };
	static const char32_t u32str_short[] = { 0x0045, 0x0020, 0x304A, 0 };
	static const uint8_t u8str[] = { 0x45, 0x20, 0xE3, 0x81, 0x8A, 0xE3, 0x98, 0x8F, 0xE3, 0x82, 0x88, 0xE3, 0x81, 0x86, 0xF0, 0x9F, 0x8E, 0xA4, 0 };

	godot_string s;

	godot_string_new_with_utf8_chars(&s, (const char *)u8str);
	CHECK(u32scmp(godot_string_get_data(&s), u32str) == 0);
	godot_string_destroy(&s);

	godot_string_new_with_utf8_chars_and_len(&s, (const char *)u8str, 5);
	CHECK(u32scmp(godot_string_get_data(&s), u32str_short) == 0);
	godot_string_destroy(&s);

	godot_string_new_with_utf32_chars(&s, u32str);
	godot_char_string cs = godot_string_utf8(&s);
	godot_string_parse_utf8(&s, godot_char_string_get_data(&cs));
	CHECK(u32scmp(godot_string_get_data(&s), u32str) == 0);
	godot_string_destroy(&s);
	godot_char_string_destroy(&cs);

	godot_string_new_with_utf32_chars(&s, u32str);
	cs = godot_string_utf8(&s);
	godot_string_parse_utf8_with_len(&s, godot_char_string_get_data(&cs), godot_char_string_length(&cs));
	CHECK(u32scmp(godot_string_get_data(&s), u32str) == 0);
	godot_string_destroy(&s);
	godot_char_string_destroy(&cs);
}

TEST_CASE("[GDNative String] Construct from UTF-8 string with BOM") {
	static const char32_t u32str[] = { 0x0045, 0x0020, 0x304A, 0x360F, 0x3088, 0x3046, 0x1F3A4, 0 };
	static const char32_t u32str_short[] = { 0x0045, 0x0020, 0x304A, 0 };
	static const uint8_t u8str[] = { 0xEF, 0xBB, 0xBF, 0x45, 0x20, 0xE3, 0x81, 0x8A, 0xE3, 0x98, 0x8F, 0xE3, 0x82, 0x88, 0xE3, 0x81, 0x86, 0xF0, 0x9F, 0x8E, 0xA4, 0 };

	godot_string s;

	godot_string_new_with_utf8_chars(&s, (const char *)u8str);
	CHECK(u32scmp(godot_string_get_data(&s), u32str) == 0);
	godot_string_destroy(&s);

	godot_string_new_with_utf8_chars_and_len(&s, (const char *)u8str, 8);
	CHECK(u32scmp(godot_string_get_data(&s), u32str_short) == 0);
	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] Construct from UTF-16 string") {
	static const char32_t u32str[] = { 0x0045, 0x0020, 0x1F3A4, 0x360F, 0x3088, 0x3046, 0x1F3A4, 0 };
	static const char32_t u32str_short[] = { 0x0045, 0x0020, 0x1F3A4, 0 };
	static const char16_t u16str[] = { 0x0045, 0x0020, 0xD83C, 0xDFA4, 0x360F, 0x3088, 0x3046, 0xD83C, 0xDFA4, 0 };

	godot_string s;

	godot_string_new_with_utf16_chars(&s, u16str);
	CHECK(u32scmp(godot_string_get_data(&s), u32str) == 0);
	godot_string_destroy(&s);

	godot_string_new_with_utf16_chars_and_len(&s, u16str, 4);
	CHECK(u32scmp(godot_string_get_data(&s), u32str_short) == 0);
	godot_string_destroy(&s);

	godot_string_new_with_utf32_chars(&s, u32str);
	godot_char16_string cs = godot_string_utf16(&s);
	godot_string_parse_utf16(&s, godot_char16_string_get_data(&cs));
	CHECK(u32scmp(godot_string_get_data(&s), u32str) == 0);
	godot_string_destroy(&s);
	godot_char16_string_destroy(&cs);

	godot_string_new_with_utf32_chars(&s, u32str);
	cs = godot_string_utf16(&s);
	godot_string_parse_utf16_with_len(&s, godot_char16_string_get_data(&cs), godot_char16_string_length(&cs));
	CHECK(u32scmp(godot_string_get_data(&s), u32str) == 0);
	godot_string_destroy(&s);
	godot_char16_string_destroy(&cs);
}

TEST_CASE("[GDNative String] Construct from UTF-16 string with BOM ") {
	static const char32_t u32str[] = { 0x0045, 0x0020, 0x1F3A4, 0x360F, 0x3088, 0x3046, 0x1F3A4, 0 };
	static const char32_t u32str_short[] = { 0x0045, 0x0020, 0x1F3A4, 0 };
	static const char16_t u16str[] = { 0xFEFF, 0x0045, 0x0020, 0xD83C, 0xDFA4, 0x360F, 0x3088, 0x3046, 0xD83C, 0xDFA4, 0 };
	static const char16_t u16str_swap[] = { 0xFFFE, 0x4500, 0x2000, 0x3CD8, 0xA4DF, 0x0F36, 0x8830, 0x4630, 0x3CD8, 0xA4DF, 0 };

	godot_string s;

	godot_string_new_with_utf16_chars(&s, u16str);
	CHECK(u32scmp(godot_string_get_data(&s), u32str) == 0);
	godot_string_destroy(&s);

	godot_string_new_with_utf16_chars(&s, u16str_swap);
	CHECK(u32scmp(godot_string_get_data(&s), u32str) == 0);
	godot_string_destroy(&s);

	godot_string_new_with_utf16_chars_and_len(&s, u16str, 5);
	CHECK(u32scmp(godot_string_get_data(&s), u32str_short) == 0);
	godot_string_destroy(&s);

	godot_string_new_with_utf16_chars_and_len(&s, u16str_swap, 5);
	CHECK(u32scmp(godot_string_get_data(&s), u32str_short) == 0);
	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] Construct string copy") {
	godot_string s, t;

	godot_string_new_with_latin1_chars(&s, "Hello");
	godot_string_new_copy(&t, &s);
	CHECK(u32scmp(godot_string_get_data(&t), U"Hello") == 0);
	godot_string_destroy(&t);
	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] Construct empty string") {
	godot_string s;

	godot_string_new(&s);
	CHECK(u32scmp(godot_string_get_data(&s), U"") == 0);
	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] ASCII/Latin-1") {
	godot_string s;
	godot_string_new_with_utf32_chars(&s, U"Primero Leche");

	godot_char_string cs = godot_string_ascii(&s);
	CHECK(strcmp(godot_char_string_get_data(&cs), "Primero Leche") == 0);
	godot_char_string_destroy(&cs);

	cs = godot_string_latin1(&s);
	CHECK(strcmp(godot_char_string_get_data(&cs), "Primero Leche") == 0);
	godot_char_string_destroy(&cs);

	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] Comparisons (equal)") {
	godot_string s, t;

	godot_string_new_with_latin1_chars(&s, "Test Compare");
	godot_string_new_with_latin1_chars(&t, "Test Compare");
	CHECK(godot_string_operator_equal(&s, &t));
	godot_string_destroy(&s);
	godot_string_destroy(&t);
}

TEST_CASE("[GDNative String] Comparisons (operator <)") {
	godot_string s, t;

	godot_string_new_with_latin1_chars(&s, "Bees");

	godot_string_new_with_latin1_chars(&t, "Elephant");
	CHECK(godot_string_operator_less(&s, &t));
	godot_string_destroy(&t);

	godot_string_new_with_latin1_chars(&t, "Amber");
	CHECK(!godot_string_operator_less(&s, &t));
	godot_string_destroy(&t);

	godot_string_new_with_latin1_chars(&t, "Beatrix");
	CHECK(!godot_string_operator_less(&s, &t));
	godot_string_destroy(&t);

	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] Concatenation (operator +)") {
	godot_string s, t, x;

	godot_string_new_with_latin1_chars(&s, "Hel");
	godot_string_new_with_latin1_chars(&t, "lo");
	x = godot_string_operator_plus(&s, &t);
	CHECK(u32scmp(godot_string_get_data(&x), U"Hello") == 0);
	godot_string_destroy(&x);
	godot_string_destroy(&s);
	godot_string_destroy(&t);
}

TEST_CASE("[GDNative String] Testing size and length of string") {
	godot_string s;

	godot_string_new_with_latin1_chars(&s, "Mellon");
	CHECK(godot_string_length(&s) == 6);
	godot_string_destroy(&s);

	godot_string_new_with_latin1_chars(&s, "Mellon1");
	CHECK(godot_string_length(&s) == 7);
	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] Testing for empty string") {
	godot_string s;

	godot_string_new_with_latin1_chars(&s, "Mellon");
	CHECK(!godot_string_empty(&s));
	godot_string_destroy(&s);

	godot_string_new_with_latin1_chars(&s, "");
	CHECK(godot_string_empty(&s));
	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] Test chr") {
	godot_string s;

	s = godot_string_chr('H');
	CHECK(u32scmp(godot_string_get_data(&s), U"H") == 0);
	godot_string_destroy(&s);

	s = godot_string_chr(0x3012);
	CHECK(godot_string_operator_index_const(&s, 0) == 0x3012);
	godot_string_destroy(&s);

	ERR_PRINT_OFF
	s = godot_string_chr(0xd812);
	CHECK(godot_string_operator_index_const(&s, 0) == 0xfffd); // Unpaired UTF-16 surrogate
	godot_string_destroy(&s);

	s = godot_string_chr(0x20d812);
	CHECK(godot_string_operator_index_const(&s, 0) == 0xfffd); // Outside UTF-32 range
	godot_string_destroy(&s);
	ERR_PRINT_ON
}

TEST_CASE("[GDNative String] Operator []") {
	godot_string s;

	godot_string_new_with_latin1_chars(&s, "Hello");
	CHECK(*godot_string_operator_index(&s, 1) == 'e');
	CHECK(godot_string_operator_index_const(&s, 0) == 'H');
	CHECK(godot_string_ord_at(&s, 0) == 'H');
	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] Case function test") {
	godot_string s, t;

	godot_string_new_with_latin1_chars(&s, "MoMoNgA");

	t = godot_string_to_upper(&s);
	CHECK(u32scmp(godot_string_get_data(&t), U"MOMONGA") == 0);
	godot_string_destroy(&t);

	t = godot_string_to_lower(&s);
	CHECK(u32scmp(godot_string_get_data(&t), U"momonga") == 0);
	godot_string_destroy(&t);

	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] Case compare function test") {
	godot_string s, t;

	godot_string_new_with_latin1_chars(&s, "MoMoNgA");
	godot_string_new_with_latin1_chars(&t, "momonga");

	CHECK(godot_string_casecmp_to(&s, &t) != 0);
	CHECK(godot_string_nocasecmp_to(&s, &t) == 0);

	godot_string_destroy(&s);
	godot_string_destroy(&t);
}

TEST_CASE("[GDNative String] Natural compare function test") {
	godot_string s, t;

	godot_string_new_with_latin1_chars(&s, "img2.png");
	godot_string_new_with_latin1_chars(&t, "img10.png");

	CHECK(godot_string_nocasecmp_to(&s, &t) > 0);
	CHECK(godot_string_naturalnocasecmp_to(&s, &t) < 0);

	godot_string_destroy(&s);
	godot_string_destroy(&t);
}

TEST_CASE("[GDNative String] hex_encode_buffer") {
	static const uint8_t u8str[] = { 0x45, 0xE3, 0x81, 0x8A, 0x8F, 0xE3 };
	godot_string s = godot_string_hex_encode_buffer(u8str, 6);
	CHECK(u32scmp(godot_string_get_data(&s), U"45e3818a8fe3") == 0);
	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] Substr") {
	godot_string s, t;
	godot_string_new_with_latin1_chars(&s, "Killer Baby");
	t = godot_string_substr(&s, 3, 4);
	CHECK(u32scmp(godot_string_get_data(&t), U"ler ") == 0);
	godot_string_destroy(&t);
	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] Find") {
	godot_string s, t;
	godot_string_new_with_latin1_chars(&s, "Pretty Woman Woman");

	godot_string_new_with_latin1_chars(&t, "Revenge of the Monster Truck");
	CHECK(godot_string_find(&s, &t) == -1);
	godot_string_destroy(&t);

	godot_string_new_with_latin1_chars(&t, "tty");
	CHECK(godot_string_find(&s, &t) == 3);
	godot_string_destroy(&t);

	godot_string_new_with_latin1_chars(&t, "Wo");
	CHECK(godot_string_find_from(&s, &t, 9) == 13);
	godot_string_destroy(&t);

	godot_string_new_with_latin1_chars(&t, "man");
	CHECK(godot_string_rfind(&s, &t) == 15);
	godot_string_destroy(&t);

	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] Find no case") {
	godot_string s, t;
	godot_string_new_with_latin1_chars(&s, "Pretty Whale Whale");

	godot_string_new_with_latin1_chars(&t, "WHA");
	CHECK(godot_string_findn(&s, &t) == 7);
	godot_string_destroy(&t);

	godot_string_new_with_latin1_chars(&t, "WHA");
	CHECK(godot_string_findn_from(&s, &t, 9) == 13);
	godot_string_destroy(&t);

	godot_string_new_with_latin1_chars(&t, "WHA");
	CHECK(godot_string_rfindn(&s, &t) == 13);
	godot_string_destroy(&t);

	godot_string_new_with_latin1_chars(&t, "Revenge of the Monster SawFish");
	CHECK(godot_string_findn(&s, &t) == -1);
	godot_string_destroy(&t);

	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] Find MK") {
	godot_packed_string_array keys;
	godot_packed_string_array_new(&keys);

#define PUSH_KEY(x)                                     \
	{                                                   \
		godot_string t;                                 \
		godot_string_new_with_latin1_chars(&t, x);      \
		godot_packed_string_array_push_back(&keys, &t); \
		godot_string_destroy(&t);                       \
	}

	PUSH_KEY("sty")
	PUSH_KEY("tty")
	PUSH_KEY("man")

	godot_string s;
	godot_string_new_with_latin1_chars(&s, "Pretty Woman");
	godot_int key = 0;

	CHECK(godot_string_findmk(&s, &keys) == 3);
	CHECK(godot_string_findmk_from_in_place(&s, &keys, 0, &key) == 3);
	CHECK(key == 1);

	CHECK(godot_string_findmk_from(&s, &keys, 5) == 9);
	CHECK(godot_string_findmk_from_in_place(&s, &keys, 5, &key) == 9);
	CHECK(key == 2);

	godot_string_destroy(&s);
	godot_packed_string_array_destroy(&keys);

#undef PUSH_KEY
}

TEST_CASE("[GDNative String] Find and replace") {
	godot_string s, c, w;
	godot_string_new_with_latin1_chars(&s, "Happy Birthday, Anna!");
	godot_string_new_with_latin1_chars(&c, "Birthday");
	godot_string_new_with_latin1_chars(&w, "Halloween");
	godot_string t = godot_string_replace(&s, &c, &w);
	CHECK(u32scmp(godot_string_get_data(&t), U"Happy Halloween, Anna!") == 0);
	godot_string_destroy(&s);
	godot_string_destroy(&c);
	godot_string_destroy(&w);

	godot_string_new_with_latin1_chars(&c, "H");
	godot_string_new_with_latin1_chars(&w, "W");
	s = godot_string_replace_first(&t, &c, &w);
	godot_string_destroy(&t);
	godot_string_destroy(&c);
	godot_string_destroy(&w);

	CHECK(u32scmp(godot_string_get_data(&s), U"Wappy Halloween, Anna!") == 0);
	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] Insertion") {
	godot_string s, t, r, u;
	godot_string_new_with_latin1_chars(&s, "Who is Frederic?");
	godot_string_new_with_latin1_chars(&t, "?");
	godot_string_new_with_latin1_chars(&r, " Chopin");

	u = godot_string_insert(&s, godot_string_find(&s, &t), &r);
	CHECK(u32scmp(godot_string_get_data(&u), U"Who is Frederic Chopin?") == 0);

	godot_string_destroy(&s);
	godot_string_destroy(&t);
	godot_string_destroy(&r);
	godot_string_destroy(&u);
}

TEST_CASE("[GDNative String] Number to string") {
	godot_string s;
	s = godot_string_num(3.141593);
	CHECK(u32scmp(godot_string_get_data(&s), U"3.141593") == 0);
	godot_string_destroy(&s);

	s = godot_string_num_with_decimals(3.141593, 3);
	CHECK(u32scmp(godot_string_get_data(&s), U"3.142") == 0);
	godot_string_destroy(&s);

	s = godot_string_num_real(3.141593);
	CHECK(u32scmp(godot_string_get_data(&s), U"3.141593") == 0);
	godot_string_destroy(&s);

	s = godot_string_num_scientific(30000000);
	CHECK(u32scmp(godot_string_get_data(&s), U"3e+07") == 0);
	godot_string_destroy(&s);

	s = godot_string_num_int64(3141593, 10);
	CHECK(u32scmp(godot_string_get_data(&s), U"3141593") == 0);
	godot_string_destroy(&s);

	s = godot_string_num_int64(0xA141593, 16);
	CHECK(u32scmp(godot_string_get_data(&s), U"a141593") == 0);
	godot_string_destroy(&s);

	s = godot_string_num_int64_capitalized(0xA141593, 16, true);
	CHECK(u32scmp(godot_string_get_data(&s), U"A141593") == 0);
	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] String to integer") {
	static const wchar_t *wnums[4] = { L"1237461283", L"- 22", L"0", L" - 1123412" };
	static const char *nums[4] = { "1237461283", "- 22", "0", " - 1123412" };
	static const int num[4] = { 1237461283, -22, 0, -1123412 };

	for (int i = 0; i < 4; i++) {
		godot_string s;
		godot_string_new_with_latin1_chars(&s, nums[i]);
		CHECK(godot_string_to_int(&s) == num[i]);
		godot_string_destroy(&s);

		CHECK(godot_string_char_to_int(nums[i]) == num[i]);
		CHECK(godot_string_wchar_to_int(wnums[i]) == num[i]);
	}
}

TEST_CASE("[GDNative String] Hex to integer") {
	static const char *nums[4] = { "0xFFAE", "22", "0", "AADDAD" };
	static const int64_t num[4] = { 0xFFAE, 0x22, 0, 0xAADDAD };
	static const bool wo_prefix[4] = { false, true, true, true };
	static const bool w_prefix[4] = { true, false, true, false };

	for (int i = 0; i < 4; i++) {
		godot_string s;
		godot_string_new_with_latin1_chars(&s, nums[i]);
		CHECK((godot_string_hex_to_int_with_prefix(&s) == num[i]) == w_prefix[i]);
		CHECK((godot_string_hex_to_int(&s) == num[i]) == wo_prefix[i]);
		godot_string_destroy(&s);
	}
}

TEST_CASE("[GDNative String] String to float") {
	static const wchar_t *wnums[4] = { L"-12348298412.2", L"0.05", L"2.0002", L" -0.0001" };
	static const char *nums[4] = { "-12348298412.2", "0.05", "2.0002", " -0.0001" };
	static const double num[4] = { -12348298412.2, 0.05, 2.0002, -0.0001 };

	for (int i = 0; i < 4; i++) {
		godot_string s;
		godot_string_new_with_latin1_chars(&s, nums[i]);
		CHECK(!(ABS(godot_string_to_float(&s) - num[i]) > 0.00001));
		godot_string_destroy(&s);

		CHECK(!(ABS(godot_string_char_to_float(nums[i]) - num[i]) > 0.00001));
		CHECK(!(ABS(godot_string_wchar_to_float(wnums[i], nullptr) - num[i]) > 0.00001));
	}
}

TEST_CASE("[GDNative String] CamelCase to underscore") {
	godot_string s, t;
	godot_string_new_with_latin1_chars(&s, "TestTestStringGD");

	t = godot_string_camelcase_to_underscore(&s);
	CHECK(u32scmp(godot_string_get_data(&t), U"Test_Test_String_GD") == 0);
	godot_string_destroy(&t);

	t = godot_string_camelcase_to_underscore_lowercased(&s);
	CHECK(u32scmp(godot_string_get_data(&t), U"test_test_string_gd") == 0);
	godot_string_destroy(&t);

	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] Slicing") {
	godot_string s, c;
	godot_string_new_with_latin1_chars(&s, "Mars,Jupiter,Saturn,Uranus");
	godot_string_new_with_latin1_chars(&c, ",");

	const char32_t *slices[4] = { U"Mars", U"Jupiter", U"Saturn", U"Uranus" };
	for (int i = 0; i < godot_string_get_slice_count(&s, &c); i++) {
		godot_string t;
		t = godot_string_get_slice(&s, &c, i);
		CHECK(u32scmp(godot_string_get_data(&t), slices[i]) == 0);
		godot_string_destroy(&t);

		t = godot_string_get_slicec(&s, U',', i);
		CHECK(u32scmp(godot_string_get_data(&t), slices[i]) == 0);
		godot_string_destroy(&t);
	}

	godot_string_destroy(&c);
	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] Splitting") {
	godot_string s, c;
	godot_string_new_with_latin1_chars(&s, "Mars,Jupiter,Saturn,Uranus");
	godot_string_new_with_latin1_chars(&c, ",");

	godot_packed_string_array l;

	const char32_t *slices_l[3] = { U"Mars", U"Jupiter", U"Saturn,Uranus" };
	const char32_t *slices_r[3] = { U"Mars,Jupiter", U"Saturn", U"Uranus" };

	l = godot_string_split_with_maxsplit(&s, &c, true, 2);
	CHECK(godot_packed_string_array_size(&l) == 3);
	for (int i = 0; i < godot_packed_string_array_size(&l); i++) {
		godot_string t = godot_packed_string_array_get(&l, i);
		CHECK(u32scmp(godot_string_get_data(&t), slices_l[i]) == 0);
		godot_string_destroy(&t);
	}
	godot_packed_string_array_destroy(&l);

	l = godot_string_rsplit_with_maxsplit(&s, &c, true, 2);
	CHECK(godot_packed_string_array_size(&l) == 3);
	for (int i = 0; i < godot_packed_string_array_size(&l); i++) {
		godot_string t = godot_packed_string_array_get(&l, i);
		CHECK(u32scmp(godot_string_get_data(&t), slices_r[i]) == 0);
		godot_string_destroy(&t);
	}
	godot_packed_string_array_destroy(&l);
	godot_string_destroy(&s);

	godot_string_new_with_latin1_chars(&s, "Mars Jupiter Saturn Uranus");
	const char32_t *slices_s[4] = { U"Mars", U"Jupiter", U"Saturn", U"Uranus" };
	l = godot_string_split_spaces(&s);
	for (int i = 0; i < godot_packed_string_array_size(&l); i++) {
		godot_string t = godot_packed_string_array_get(&l, i);
		CHECK(u32scmp(godot_string_get_data(&t), slices_s[i]) == 0);
		godot_string_destroy(&t);
	}
	godot_packed_string_array_destroy(&l);
	godot_string_destroy(&s);

	godot_string c1, c2;
	godot_string_new_with_latin1_chars(&c1, ";");
	godot_string_new_with_latin1_chars(&c2, " ");

	godot_string_new_with_latin1_chars(&s, "1.2;2.3 4.5");
	const double slices_d[3] = { 1.2, 2.3, 4.5 };

	godot_packed_float32_array lf = godot_string_split_floats_allow_empty(&s, &c1);
	CHECK(godot_packed_float32_array_size(&lf) == 2);
	for (int i = 0; i < godot_packed_float32_array_size(&lf); i++) {
		CHECK(ABS(godot_packed_float32_array_get(&lf, i) - slices_d[i]) <= 0.00001);
	}
	godot_packed_float32_array_destroy(&lf);

	godot_packed_string_array keys;
	godot_packed_string_array_new(&keys);
	godot_packed_string_array_push_back(&keys, &c1);
	godot_packed_string_array_push_back(&keys, &c2);

	lf = godot_string_split_floats_mk_allow_empty(&s, &keys);
	CHECK(godot_packed_float32_array_size(&lf) == 3);
	for (int i = 0; i < godot_packed_float32_array_size(&lf); i++) {
		CHECK(ABS(godot_packed_float32_array_get(&lf, i) - slices_d[i]) <= 0.00001);
	}
	godot_packed_float32_array_destroy(&lf);

	godot_string_destroy(&s);
	godot_string_new_with_latin1_chars(&s, "1;2 4");
	const int slices_i[3] = { 1, 2, 4 };

	godot_packed_int32_array li = godot_string_split_ints_allow_empty(&s, &c1);
	CHECK(godot_packed_int32_array_size(&li) == 2);
	for (int i = 0; i < godot_packed_int32_array_size(&li); i++) {
		CHECK(godot_packed_int32_array_get(&li, i) == slices_i[i]);
	}
	godot_packed_int32_array_destroy(&li);

	li = godot_string_split_ints_mk_allow_empty(&s, &keys);
	CHECK(godot_packed_int32_array_size(&li) == 3);
	for (int i = 0; i < godot_packed_int32_array_size(&li); i++) {
		CHECK(godot_packed_int32_array_get(&li, i) == slices_i[i]);
	}
	godot_packed_int32_array_destroy(&li);

	godot_string_destroy(&s);
	godot_string_destroy(&c);
	godot_string_destroy(&c1);
	godot_string_destroy(&c2);
	godot_packed_string_array_destroy(&keys);
}

TEST_CASE("[GDNative String] Erasing") {
	godot_string s, t;
	godot_string_new_with_latin1_chars(&s, "Josephine is such a cute girl!");
	godot_string_new_with_latin1_chars(&t, "cute ");

	godot_string_erase(&s, godot_string_find(&s, &t), godot_string_length(&t));

	CHECK(u32scmp(godot_string_get_data(&s), U"Josephine is such a girl!") == 0);

	godot_string_destroy(&s);
	godot_string_destroy(&t);
}

struct test_27_data {
	char const *data;
	char const *part;
	bool expected;
};

TEST_CASE("[GDNative String] Begins with") {
	test_27_data tc[] = {
		{ "res://foobar", "res://", true },
		{ "res", "res://", false },
		{ "abc", "abc", true }
	};
	size_t count = sizeof(tc) / sizeof(tc[0]);
	bool state = true;
	for (size_t i = 0; state && i < count; ++i) {
		godot_string s;
		godot_string_new_with_latin1_chars(&s, tc[i].data);

		state = godot_string_begins_with_char_array(&s, tc[i].part) == tc[i].expected;
		if (state) {
			godot_string t;
			godot_string_new_with_latin1_chars(&t, tc[i].part);
			state = godot_string_begins_with(&s, &t) == tc[i].expected;
			godot_string_destroy(&t);
		}
		godot_string_destroy(&s);

		CHECK(state);
		if (!state) {
			break;
		}
	};
	CHECK(state);
}

TEST_CASE("[GDNative String] Ends with") {
	test_27_data tc[] = {
		{ "res://foobar", "foobar", true },
		{ "res", "res://", false },
		{ "abc", "abc", true }
	};
	size_t count = sizeof(tc) / sizeof(tc[0]);
	bool state = true;
	for (size_t i = 0; state && i < count; ++i) {
		godot_string s;
		godot_string_new_with_latin1_chars(&s, tc[i].data);

		state = godot_string_ends_with_char_array(&s, tc[i].part) == tc[i].expected;
		if (state) {
			godot_string t;
			godot_string_new_with_latin1_chars(&t, tc[i].part);
			state = godot_string_ends_with(&s, &t) == tc[i].expected;
			godot_string_destroy(&t);
		}
		godot_string_destroy(&s);

		CHECK(state);
		if (!state) {
			break;
		}
	};
	CHECK(state);
}

TEST_CASE("[GDNative String] format") {
	godot_string value_format, t;
	godot_string_new_with_latin1_chars(&value_format, "red=\"$red\" green=\"$green\" blue=\"$blue\" alpha=\"$alpha\"");

	godot_variant key_v, val_v;
	godot_dictionary value_dictionary;
	godot_dictionary_new(&value_dictionary);

	godot_string_new_with_latin1_chars(&t, "red");
	godot_variant_new_string(&key_v, &t);
	godot_string_destroy(&t);
	godot_variant_new_int(&val_v, 10);
	godot_dictionary_set(&value_dictionary, &key_v, &val_v);
	godot_variant_destroy(&key_v);
	godot_variant_destroy(&val_v);

	godot_string_new_with_latin1_chars(&t, "green");
	godot_variant_new_string(&key_v, &t);
	godot_string_destroy(&t);
	godot_variant_new_int(&val_v, 20);
	godot_dictionary_set(&value_dictionary, &key_v, &val_v);
	godot_variant_destroy(&key_v);
	godot_variant_destroy(&val_v);

	godot_string_new_with_latin1_chars(&t, "blue");
	godot_variant_new_string(&key_v, &t);
	godot_string_destroy(&t);
	godot_string_new_with_latin1_chars(&t, "bla");
	godot_variant_new_string(&val_v, &t);
	godot_string_destroy(&t);
	godot_dictionary_set(&value_dictionary, &key_v, &val_v);
	godot_variant_destroy(&key_v);
	godot_variant_destroy(&val_v);

	godot_string_new_with_latin1_chars(&t, "alpha");
	godot_variant_new_string(&key_v, &t);
	godot_string_destroy(&t);
	godot_variant_new_real(&val_v, 0.4);
	godot_dictionary_set(&value_dictionary, &key_v, &val_v);
	godot_variant_destroy(&key_v);
	godot_variant_destroy(&val_v);

	godot_variant dict_v;
	godot_variant_new_dictionary(&dict_v, &value_dictionary);
	godot_string s = godot_string_format_with_custom_placeholder(&value_format, &dict_v, "$_");

	CHECK(u32scmp(godot_string_get_data(&s), U"red=\"10\" green=\"20\" blue=\"bla\" alpha=\"0.4\"") == 0);

	godot_dictionary_destroy(&value_dictionary);
	godot_string_destroy(&s);
	godot_variant_destroy(&dict_v);
	godot_string_destroy(&value_format);
}

TEST_CASE("[GDNative String] sprintf") {
	//godot_string GDAPI (const godot_string *p_self, const godot_array *p_values, godot_bool *p_error);
	godot_string format, output;
	godot_array args;
	bool error;

#define ARRAY_PUSH_STRING(x)                       \
	{                                              \
		godot_variant v;                           \
		godot_string t;                            \
		godot_string_new_with_latin1_chars(&t, x); \
		godot_variant_new_string(&v, &t);          \
		godot_string_destroy(&t);                  \
		godot_array_push_back(&args, &v);          \
		godot_variant_destroy(&v);                 \
	}

#define ARRAY_PUSH_INT(x)                 \
	{                                     \
		godot_variant v;                  \
		godot_variant_new_int(&v, x);     \
		godot_array_push_back(&args, &v); \
		godot_variant_destroy(&v);        \
	}

#define ARRAY_PUSH_REAL(x)                \
	{                                     \
		godot_variant v;                  \
		godot_variant_new_real(&v, x);    \
		godot_array_push_back(&args, &v); \
		godot_variant_destroy(&v);        \
	}

	godot_array_new(&args);

	// %%
	godot_string_new_with_latin1_chars(&format, "fish %% frog");
	godot_array_clear(&args);
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	CHECK(u32scmp(godot_string_get_data(&output), U"fish % frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);
	//////// INTS

	// Int
	godot_string_new_with_latin1_chars(&format, "fish %d frog");
	godot_array_clear(&args);
	ARRAY_PUSH_INT(5);
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	CHECK(u32scmp(godot_string_get_data(&output), U"fish 5 frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// Int left padded with zeroes.
	godot_string_new_with_latin1_chars(&format, "fish %05d frog");
	godot_array_clear(&args);
	ARRAY_PUSH_INT(5);
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	CHECK(u32scmp(godot_string_get_data(&output), U"fish 00005 frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// Int left padded with spaces.
	godot_string_new_with_latin1_chars(&format, "fish %5d frog");
	godot_array_clear(&args);
	ARRAY_PUSH_INT(5);
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	CHECK(u32scmp(godot_string_get_data(&output), U"fish     5 frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// Int right padded with spaces.
	godot_string_new_with_latin1_chars(&format, "fish %-5d frog");
	godot_array_clear(&args);
	ARRAY_PUSH_INT(5);
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	CHECK(u32scmp(godot_string_get_data(&output), U"fish 5     frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// Int with sign (positive).
	godot_string_new_with_latin1_chars(&format, "fish %+d frog");
	godot_array_clear(&args);
	ARRAY_PUSH_INT(5);
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	CHECK(u32scmp(godot_string_get_data(&output), U"fish +5 frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// Negative int.
	godot_string_new_with_latin1_chars(&format, "fish %d frog");
	godot_array_clear(&args);
	ARRAY_PUSH_INT(-5);
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	CHECK(u32scmp(godot_string_get_data(&output), U"fish -5 frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// Hex (lower)
	godot_string_new_with_latin1_chars(&format, "fish %x frog");
	godot_array_clear(&args);
	ARRAY_PUSH_INT(45);
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	CHECK(u32scmp(godot_string_get_data(&output), U"fish 2d frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// Hex (upper)
	godot_string_new_with_latin1_chars(&format, "fish %X frog");
	godot_array_clear(&args);
	ARRAY_PUSH_INT(45);
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	CHECK(u32scmp(godot_string_get_data(&output), U"fish 2D frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// Octal
	godot_string_new_with_latin1_chars(&format, "fish %o frog");
	godot_array_clear(&args);
	ARRAY_PUSH_INT(99);
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	CHECK(u32scmp(godot_string_get_data(&output), U"fish 143 frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);
	////// REALS

	// Real
	godot_string_new_with_latin1_chars(&format, "fish %f frog");
	godot_array_clear(&args);
	ARRAY_PUSH_REAL(99.99);
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	CHECK(u32scmp(godot_string_get_data(&output), U"fish 99.990000 frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// Real left-padded
	godot_string_new_with_latin1_chars(&format, "fish %11f frog");
	godot_array_clear(&args);
	ARRAY_PUSH_REAL(99.99);
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	CHECK(u32scmp(godot_string_get_data(&output), U"fish   99.990000 frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// Real right-padded
	godot_string_new_with_latin1_chars(&format, "fish %-11f frog");
	godot_array_clear(&args);
	ARRAY_PUSH_REAL(99.99);
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	CHECK(u32scmp(godot_string_get_data(&output), U"fish 99.990000   frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// Real given int.
	godot_string_new_with_latin1_chars(&format, "fish %f frog");
	godot_array_clear(&args);
	ARRAY_PUSH_REAL(99);
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	CHECK(u32scmp(godot_string_get_data(&output), U"fish 99.000000 frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// Real with sign (positive).
	godot_string_new_with_latin1_chars(&format, "fish %+f frog");
	godot_array_clear(&args);
	ARRAY_PUSH_REAL(99.99);
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	CHECK(u32scmp(godot_string_get_data(&output), U"fish +99.990000 frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// Real with 1 decimals.
	godot_string_new_with_latin1_chars(&format, "fish %.1f frog");
	godot_array_clear(&args);
	ARRAY_PUSH_REAL(99.99);
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	CHECK(u32scmp(godot_string_get_data(&output), U"fish 100.0 frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// Real with 12 decimals.
	godot_string_new_with_latin1_chars(&format, "fish %.12f frog");
	godot_array_clear(&args);
	ARRAY_PUSH_REAL(99.99);
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	CHECK(u32scmp(godot_string_get_data(&output), U"fish 99.990000000000 frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// Real with no decimals.
	godot_string_new_with_latin1_chars(&format, "fish %.f frog");
	godot_array_clear(&args);
	ARRAY_PUSH_REAL(99.99);
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	CHECK(u32scmp(godot_string_get_data(&output), U"fish 100 frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	/////// Strings.

	// String
	godot_string_new_with_latin1_chars(&format, "fish %s frog");
	godot_array_clear(&args);
	ARRAY_PUSH_STRING("cheese");
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	CHECK(u32scmp(godot_string_get_data(&output), U"fish cheese frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// String left-padded
	godot_string_new_with_latin1_chars(&format, "fish %10s frog");
	godot_array_clear(&args);
	ARRAY_PUSH_STRING("cheese");
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	CHECK(u32scmp(godot_string_get_data(&output), U"fish     cheese frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// String right-padded
	godot_string_new_with_latin1_chars(&format, "fish %-10s frog");
	godot_array_clear(&args);
	ARRAY_PUSH_STRING("cheese");
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	CHECK(u32scmp(godot_string_get_data(&output), U"fish cheese     frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	///// Characters

	// Character as string.
	godot_string_new_with_latin1_chars(&format, "fish %c frog");
	godot_array_clear(&args);
	ARRAY_PUSH_STRING("A");
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	CHECK(u32scmp(godot_string_get_data(&output), U"fish A frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// Character as int.
	godot_string_new_with_latin1_chars(&format, "fish %c frog");
	godot_array_clear(&args);
	ARRAY_PUSH_INT(65);
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	CHECK(u32scmp(godot_string_get_data(&output), U"fish A frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	///// Dynamic width

	// String dynamic width
	godot_string_new_with_latin1_chars(&format, "fish %*s frog");
	godot_array_clear(&args);
	ARRAY_PUSH_INT(10);
	ARRAY_PUSH_STRING("cheese");
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	REQUIRE(u32scmp(godot_string_get_data(&output), U"fish     cheese frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// Int dynamic width
	godot_string_new_with_latin1_chars(&format, "fish %*d frog");
	godot_array_clear(&args);
	ARRAY_PUSH_INT(10);
	ARRAY_PUSH_INT(99);
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	REQUIRE(u32scmp(godot_string_get_data(&output), U"fish         99 frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// Float dynamic width
	godot_string_new_with_latin1_chars(&format, "fish %*.*f frog");
	godot_array_clear(&args);
	ARRAY_PUSH_INT(10);
	ARRAY_PUSH_INT(3);
	ARRAY_PUSH_REAL(99.99);
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error == false);
	CHECK(u32scmp(godot_string_get_data(&output), U"fish     99.990 frog") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	///// Errors

	// More formats than arguments.
	godot_string_new_with_latin1_chars(&format, "fish %s %s frog");
	godot_array_clear(&args);
	ARRAY_PUSH_STRING("cheese");
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error);
	CHECK(u32scmp(godot_string_get_data(&output), U"not enough arguments for format string") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// More arguments than formats.
	godot_string_new_with_latin1_chars(&format, "fish %s frog");
	godot_array_clear(&args);
	ARRAY_PUSH_STRING("hello");
	ARRAY_PUSH_STRING("cheese");
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error);
	CHECK(u32scmp(godot_string_get_data(&output), U"not all arguments converted during string formatting") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// Incomplete format.
	godot_string_new_with_latin1_chars(&format, "fish %10");
	godot_array_clear(&args);
	ARRAY_PUSH_STRING("cheese");
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error);
	CHECK(u32scmp(godot_string_get_data(&output), U"incomplete format") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// Bad character in format string
	godot_string_new_with_latin1_chars(&format, "fish %&f frog");
	godot_array_clear(&args);
	ARRAY_PUSH_STRING("cheese");
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error);
	CHECK(u32scmp(godot_string_get_data(&output), U"unsupported format character") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// Too many decimals.
	godot_string_new_with_latin1_chars(&format, "fish %2.2.2f frog");
	godot_array_clear(&args);
	ARRAY_PUSH_REAL(99.99);
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error);
	CHECK(u32scmp(godot_string_get_data(&output), U"too many decimal points in format") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// * not a number
	godot_string_new_with_latin1_chars(&format, "fish %*f frog");
	godot_array_clear(&args);
	ARRAY_PUSH_STRING("cheese");
	ARRAY_PUSH_REAL(99.99);
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error);
	CHECK(u32scmp(godot_string_get_data(&output), U"* wants number") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// Character too long.
	godot_string_new_with_latin1_chars(&format, "fish %c frog");
	godot_array_clear(&args);
	ARRAY_PUSH_STRING("sc");
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error);
	CHECK(u32scmp(godot_string_get_data(&output), U"%c requires number or single-character string") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	// Character bad type.
	godot_string_new_with_latin1_chars(&format, "fish %c frog");
	godot_array_clear(&args);
	godot_array t;
	godot_array_new(&t);
	godot_variant v;
	godot_variant_new_array(&v, &t);
	godot_array_destroy(&t);
	godot_array_push_back(&args, &v);
	godot_variant_destroy(&v);
	output = godot_string_sprintf(&format, &args, &error);
	REQUIRE(error);
	CHECK(u32scmp(godot_string_get_data(&output), U"%c requires number or single-character string") == 0);
	godot_string_destroy(&format);
	godot_string_destroy(&output);

	godot_array_destroy(&args);
#undef ARRAY_PUSH_INT
#undef ARRAY_PUSH_REAL
#undef ARRAY_PUSH_STRING
}

TEST_CASE("[GDNative String] is_numeric") {
#define IS_NUM_TEST(x, r)                          \
	{                                              \
		godot_string t;                            \
		godot_string_new_with_latin1_chars(&t, x); \
		CHECK(godot_string_is_numeric(&t) == r);   \
		godot_string_destroy(&t);                  \
	}

	IS_NUM_TEST("12", true);
	IS_NUM_TEST("1.2", true);
	IS_NUM_TEST("AF", false);
	IS_NUM_TEST("-12", true);
	IS_NUM_TEST("-1.2", true);

#undef IS_NUM_TEST
}

TEST_CASE("[GDNative String] pad") {
	godot_string s, c;
	godot_string_new_with_latin1_chars(&s, "test");
	godot_string_new_with_latin1_chars(&c, "x");

	godot_string l = godot_string_lpad_with_custom_character(&s, 10, &c);
	CHECK(u32scmp(godot_string_get_data(&l), U"xxxxxxtest") == 0);
	godot_string_destroy(&l);

	godot_string r = godot_string_rpad_with_custom_character(&s, 10, &c);
	CHECK(u32scmp(godot_string_get_data(&r), U"testxxxxxx") == 0);
	godot_string_destroy(&r);

	godot_string_destroy(&s);
	godot_string_destroy(&c);

	godot_string_new_with_latin1_chars(&s, "10.10");
	c = godot_string_pad_decimals(&s, 4);
	CHECK(u32scmp(godot_string_get_data(&c), U"10.1000") == 0);
	godot_string_destroy(&c);
	c = godot_string_pad_zeros(&s, 4);
	CHECK(u32scmp(godot_string_get_data(&c), U"0010.10") == 0);
	godot_string_destroy(&c);

	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] is_subsequence_of") {
	godot_string a, t;
	godot_string_new_with_latin1_chars(&a, "is subsequence of");

	godot_string_new_with_latin1_chars(&t, "sub");
	CHECK(godot_string_is_subsequence_of(&t, &a));
	godot_string_destroy(&t);

	godot_string_new_with_latin1_chars(&t, "Sub");
	CHECK(!godot_string_is_subsequence_of(&t, &a));
	godot_string_destroy(&t);

	godot_string_new_with_latin1_chars(&t, "Sub");
	CHECK(godot_string_is_subsequence_ofi(&t, &a));
	godot_string_destroy(&t);

	godot_string_destroy(&a);
}

TEST_CASE("[GDNative String] match") {
	godot_string s, t;
	godot_string_new_with_latin1_chars(&s, "*.png");

	godot_string_new_with_latin1_chars(&t, "img1.png");
	CHECK(godot_string_match(&t, &s));
	godot_string_destroy(&t);

	godot_string_new_with_latin1_chars(&t, "img1.jpeg");
	CHECK(!godot_string_match(&t, &s));
	godot_string_destroy(&t);

	godot_string_new_with_latin1_chars(&t, "img1.Png");
	CHECK(!godot_string_match(&t, &s));
	CHECK(godot_string_matchn(&t, &s));
	godot_string_destroy(&t);

	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] IPVX address to string") {
	godot_string ip;

	godot_string_new_with_latin1_chars(&ip, "192.168.0.1");
	CHECK(godot_string_is_valid_ip_address(&ip));
	godot_string_destroy(&ip);

	godot_string_new_with_latin1_chars(&ip, "192.368.0.1");
	CHECK(!godot_string_is_valid_ip_address(&ip));
	godot_string_destroy(&ip);

	godot_string_new_with_latin1_chars(&ip, "2001:0db8:85a3:0000:0000:8a2e:0370:7334");
	CHECK(godot_string_is_valid_ip_address(&ip));
	godot_string_destroy(&ip);

	godot_string_new_with_latin1_chars(&ip, "2001:0db8:85j3:0000:0000:8a2e:0370:7334");
	CHECK(!godot_string_is_valid_ip_address(&ip));
	godot_string_destroy(&ip);

	godot_string_new_with_latin1_chars(&ip, "2001:0db8:85f345:0000:0000:8a2e:0370:7334");
	CHECK(!godot_string_is_valid_ip_address(&ip));
	godot_string_destroy(&ip);

	godot_string_new_with_latin1_chars(&ip, "2001:0db8::0:8a2e:370:7334");
	CHECK(godot_string_is_valid_ip_address(&ip));
	godot_string_destroy(&ip);

	godot_string_new_with_latin1_chars(&ip, "::ffff:192.168.0.1");
	CHECK(godot_string_is_valid_ip_address(&ip));
	godot_string_destroy(&ip);
}

TEST_CASE("[GDNative String] Capitalize against many strings") {
#define CAP_TEST(i, o)                                                                 \
	godot_string_new_with_latin1_chars(&input, i);                                     \
	godot_string_new_with_latin1_chars(&output, o);                                    \
	test = godot_string_capitalize(&input);                                            \
	CHECK(u32scmp(godot_string_get_data(&output), godot_string_get_data(&test)) == 0); \
	godot_string_destroy(&input);                                                      \
	godot_string_destroy(&output);                                                     \
	godot_string_destroy(&test);

	godot_string input, output, test;

	CAP_TEST("bytes2var", "Bytes 2 Var");
	CAP_TEST("linear2db", "Linear 2 Db");
	CAP_TEST("vector3", "Vector 3");
	CAP_TEST("sha256", "Sha 256");
	CAP_TEST("2db", "2 Db");
	CAP_TEST("PascalCase", "Pascal Case");
	CAP_TEST("PascalPascalCase", "Pascal Pascal Case");
	CAP_TEST("snake_case", "Snake Case");
	CAP_TEST("snake_snake_case", "Snake Snake Case");
	CAP_TEST("sha256sum", "Sha 256 Sum");
	CAP_TEST("cat2dog", "Cat 2 Dog");
	CAP_TEST("function(name)", "Function(name)");
	CAP_TEST("snake_case_function(snake_case_arg)", "Snake Case Function(snake Case Arg)");
	CAP_TEST("snake_case_function( snake_case_arg )", "Snake Case Function( Snake Case Arg )");

#undef CAP_TEST
}

TEST_CASE("[GDNative String] lstrip and rstrip") {
#define LSTRIP_TEST(x, y, z)                                                                     \
	{                                                                                            \
		godot_string xx, yy, zz, rr;                                                             \
		godot_string_new_with_latin1_chars(&xx, x);                                              \
		godot_string_new_with_latin1_chars(&yy, y);                                              \
		godot_string_new_with_latin1_chars(&zz, z);                                              \
		rr = godot_string_lstrip(&xx, &yy);                                                      \
		state = state && (u32scmp(godot_string_get_data(&rr), godot_string_get_data(&zz)) == 0); \
		godot_string_destroy(&xx);                                                               \
		godot_string_destroy(&yy);                                                               \
		godot_string_destroy(&zz);                                                               \
		godot_string_destroy(&rr);                                                               \
	}

#define RSTRIP_TEST(x, y, z)                                                                     \
	{                                                                                            \
		godot_string xx, yy, zz, rr;                                                             \
		godot_string_new_with_latin1_chars(&xx, x);                                              \
		godot_string_new_with_latin1_chars(&yy, y);                                              \
		godot_string_new_with_latin1_chars(&zz, z);                                              \
		rr = godot_string_rstrip(&xx, &yy);                                                      \
		state = state && (u32scmp(godot_string_get_data(&rr), godot_string_get_data(&zz)) == 0); \
		godot_string_destroy(&xx);                                                               \
		godot_string_destroy(&yy);                                                               \
		godot_string_destroy(&zz);                                                               \
		godot_string_destroy(&rr);                                                               \
	}

#define LSTRIP_UTF8_TEST(x, y, z)                                                                \
	{                                                                                            \
		godot_string xx, yy, zz, rr;                                                             \
		godot_string_new_with_utf8_chars(&xx, x);                                                \
		godot_string_new_with_utf8_chars(&yy, y);                                                \
		godot_string_new_with_utf8_chars(&zz, z);                                                \
		rr = godot_string_lstrip(&xx, &yy);                                                      \
		state = state && (u32scmp(godot_string_get_data(&rr), godot_string_get_data(&zz)) == 0); \
		godot_string_destroy(&xx);                                                               \
		godot_string_destroy(&yy);                                                               \
		godot_string_destroy(&zz);                                                               \
		godot_string_destroy(&rr);                                                               \
	}

#define RSTRIP_UTF8_TEST(x, y, z)                                                                \
	{                                                                                            \
		godot_string xx, yy, zz, rr;                                                             \
		godot_string_new_with_utf8_chars(&xx, x);                                                \
		godot_string_new_with_utf8_chars(&yy, y);                                                \
		godot_string_new_with_utf8_chars(&zz, z);                                                \
		rr = godot_string_rstrip(&xx, &yy);                                                      \
		state = state && (u32scmp(godot_string_get_data(&rr), godot_string_get_data(&zz)) == 0); \
		godot_string_destroy(&xx);                                                               \
		godot_string_destroy(&yy);                                                               \
		godot_string_destroy(&zz);                                                               \
		godot_string_destroy(&rr);                                                               \
	}

	bool state = true;

	// strip none
	LSTRIP_TEST("abc", "", "abc");
	RSTRIP_TEST("abc", "", "abc");
	// strip one
	LSTRIP_TEST("abc", "a", "bc");
	RSTRIP_TEST("abc", "c", "ab");
	// strip lots
	LSTRIP_TEST("bababbababccc", "ab", "ccc");
	RSTRIP_TEST("aaabcbcbcbbcbbc", "cb", "aaa");
	// strip empty string
	LSTRIP_TEST("", "", "");
	RSTRIP_TEST("", "", "");
	// strip to empty string
	LSTRIP_TEST("abcabcabc", "bca", "");
	RSTRIP_TEST("abcabcabc", "bca", "");
	// don't strip wrong end
	LSTRIP_TEST("abc", "c", "abc");
	LSTRIP_TEST("abca", "a", "bca");
	RSTRIP_TEST("abc", "a", "abc");
	RSTRIP_TEST("abca", "a", "abc");
	// in utf-8 "¿" (\u00bf) has the same first byte as "µ" (\u00b5)
	// and the same second as "ÿ" (\u00ff)
	LSTRIP_UTF8_TEST("¿", "µÿ", "¿");
	RSTRIP_UTF8_TEST("¿", "µÿ", "¿");
	LSTRIP_UTF8_TEST("µ¿ÿ", "µÿ", "¿ÿ");
	RSTRIP_UTF8_TEST("µ¿ÿ", "µÿ", "µ¿");

	// the above tests repeated with additional superfluous strip chars

	// strip none
	LSTRIP_TEST("abc", "qwjkl", "abc");
	RSTRIP_TEST("abc", "qwjkl", "abc");
	// strip one
	LSTRIP_TEST("abc", "qwajkl", "bc");
	RSTRIP_TEST("abc", "qwcjkl", "ab");
	// strip lots
	LSTRIP_TEST("bababbababccc", "qwabjkl", "ccc");
	RSTRIP_TEST("aaabcbcbcbbcbbc", "qwcbjkl", "aaa");
	// strip empty string
	LSTRIP_TEST("", "qwjkl", "");
	RSTRIP_TEST("", "qwjkl", "");
	// strip to empty string
	LSTRIP_TEST("abcabcabc", "qwbcajkl", "");
	RSTRIP_TEST("abcabcabc", "qwbcajkl", "");
	// don't strip wrong end
	LSTRIP_TEST("abc", "qwcjkl", "abc");
	LSTRIP_TEST("abca", "qwajkl", "bca");
	RSTRIP_TEST("abc", "qwajkl", "abc");
	RSTRIP_TEST("abca", "qwajkl", "abc");
	// in utf-8 "¿" (\u00bf) has the same first byte as "µ" (\u00b5)
	// and the same second as "ÿ" (\u00ff)
	LSTRIP_UTF8_TEST("¿", "qwaµÿjkl", "¿");
	RSTRIP_UTF8_TEST("¿", "qwaµÿjkl", "¿");
	LSTRIP_UTF8_TEST("µ¿ÿ", "qwaµÿjkl", "¿ÿ");
	RSTRIP_UTF8_TEST("µ¿ÿ", "qwaµÿjkl", "µ¿");

	CHECK(state);

#undef LSTRIP_TEST
#undef RSTRIP_TEST
#undef LSTRIP_UTF8_TEST
#undef RSTRIP_UTF8_TEST
}

TEST_CASE("[GDNative String] Cyrillic to_lower()") {
	godot_string upper, lower, test;
	godot_string_new_with_utf8_chars(&upper, "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ");
	godot_string_new_with_utf8_chars(&lower, "абвгдеёжзийклмнопрстуфхцчшщъыьэюя");

	test = godot_string_to_lower(&upper);

	CHECK((u32scmp(godot_string_get_data(&test), godot_string_get_data(&lower)) == 0));

	godot_string_destroy(&upper);
	godot_string_destroy(&lower);
	godot_string_destroy(&test);
}

TEST_CASE("[GDNative String] Count and countn functionality") {
#define COUNT_TEST(x, y, r)                                       \
	{                                                             \
		godot_string s, t;                                        \
		godot_string_new_with_latin1_chars(&s, x);                \
		godot_string_new_with_latin1_chars(&t, y);                \
		state = state && (godot_string_count(&s, &t, 0, 0) == r); \
		godot_string_destroy(&s);                                 \
		godot_string_destroy(&t);                                 \
	}

#define COUNTR_TEST(x, y, a, b, r)                                \
	{                                                             \
		godot_string s, t;                                        \
		godot_string_new_with_latin1_chars(&s, x);                \
		godot_string_new_with_latin1_chars(&t, y);                \
		state = state && (godot_string_count(&s, &t, a, b) == r); \
		godot_string_destroy(&s);                                 \
		godot_string_destroy(&t);                                 \
	}

#define COUNTN_TEST(x, y, r)                                       \
	{                                                              \
		godot_string s, t;                                         \
		godot_string_new_with_latin1_chars(&s, x);                 \
		godot_string_new_with_latin1_chars(&t, y);                 \
		state = state && (godot_string_countn(&s, &t, 0, 0) == r); \
		godot_string_destroy(&s);                                  \
		godot_string_destroy(&t);                                  \
	}

#define COUNTNR_TEST(x, y, a, b, r)                                \
	{                                                              \
		godot_string s, t;                                         \
		godot_string_new_with_latin1_chars(&s, x);                 \
		godot_string_new_with_latin1_chars(&t, y);                 \
		state = state && (godot_string_countn(&s, &t, a, b) == r); \
		godot_string_destroy(&s);                                  \
		godot_string_destroy(&t);                                  \
	}
	bool state = true;

	COUNT_TEST("", "Test", 0);
	COUNT_TEST("Test", "", 0);
	COUNT_TEST("Test", "test", 0);
	COUNT_TEST("Test", "TEST", 0);
	COUNT_TEST("TEST", "TEST", 1);
	COUNT_TEST("Test", "Test", 1);
	COUNT_TEST("aTest", "Test", 1);
	COUNT_TEST("Testa", "Test", 1);
	COUNT_TEST("TestTestTest", "Test", 3);
	COUNT_TEST("TestTestTest", "TestTest", 1);
	COUNT_TEST("TestGodotTestGodotTestGodot", "Test", 3);

	COUNTR_TEST("TestTestTestTest", "Test", 4, 8, 1);
	COUNTR_TEST("TestTestTestTest", "Test", 4, 12, 2);
	COUNTR_TEST("TestTestTestTest", "Test", 4, 16, 3);
	COUNTR_TEST("TestTestTestTest", "Test", 4, 0, 3);

	COUNTN_TEST("Test", "test", 1);
	COUNTN_TEST("Test", "TEST", 1);
	COUNTN_TEST("testTest-Testatest", "tEst", 4);
	COUNTNR_TEST("testTest-TeStatest", "tEsT", 4, 16, 2);

	CHECK(state);

#undef COUNT_TEST
#undef COUNTR_TEST
#undef COUNTN_TEST
#undef COUNTNR_TEST
}

TEST_CASE("[GDNative String] Bigrams") {
	godot_string s, t;
	godot_string_new_with_latin1_chars(&s, "abcd");
	godot_packed_string_array bigr = godot_string_bigrams(&s);
	godot_string_destroy(&s);

	CHECK(godot_packed_string_array_size(&bigr) == 3);

	t = godot_packed_string_array_get(&bigr, 0);
	CHECK(u32scmp(godot_string_get_data(&t), U"ab") == 0);
	godot_string_destroy(&t);

	t = godot_packed_string_array_get(&bigr, 1);
	CHECK(u32scmp(godot_string_get_data(&t), U"bc") == 0);
	godot_string_destroy(&t);

	t = godot_packed_string_array_get(&bigr, 2);
	CHECK(u32scmp(godot_string_get_data(&t), U"cd") == 0);
	godot_string_destroy(&t);

	godot_packed_string_array_destroy(&bigr);
}

TEST_CASE("[GDNative String] c-escape/unescape") {
	godot_string s;
	godot_string_new_with_latin1_chars(&s, "\\1\a2\b\f3\n45\r6\t7\v8\'9\?0\"");
	godot_string t = godot_string_c_escape(&s);
	godot_string u = godot_string_c_unescape(&t);
	CHECK(u32scmp(godot_string_get_data(&u), godot_string_get_data(&s)) == 0);
	godot_string_destroy(&u);
	godot_string_destroy(&t);
	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] dedent") {
	godot_string s, t;
	godot_string_new_with_latin1_chars(&s, "      aaa\n    bbb");
	godot_string_new_with_latin1_chars(&t, "aaa\nbbb");
	godot_string u = godot_string_dedent(&s);
	CHECK(u32scmp(godot_string_get_data(&u), godot_string_get_data(&t)) == 0);
	godot_string_destroy(&u);
	godot_string_destroy(&t);
	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] Path functions") {
	static const char *path[4] = { "C:\\Godot\\project\\test.tscn", "/Godot/project/test.xscn", "../Godot/project/test.scn", "Godot\\test.doc" };
	static const char *base_dir[4] = { "C:\\Godot\\project", "/Godot/project", "../Godot/project", "Godot" };
	static const char *base_name[4] = { "C:\\Godot\\project\\test", "/Godot/project/test", "../Godot/project/test", "Godot\\test" };
	static const char *ext[4] = { "tscn", "xscn", "scn", "doc" };
	static const char *file[4] = { "test.tscn", "test.xscn", "test.scn", "test.doc" };
	static const bool abs[4] = { true, true, false, false };

	for (int i = 0; i < 4; i++) {
		godot_string s, t, u, f;
		godot_string_new_with_latin1_chars(&s, path[i]);

		t = godot_string_get_base_dir(&s);
		godot_string_new_with_latin1_chars(&u, base_dir[i]);
		CHECK(u32scmp(godot_string_get_data(&u), godot_string_get_data(&t)) == 0);
		godot_string_destroy(&u);
		godot_string_destroy(&t);

		t = godot_string_get_basename(&s);
		godot_string_new_with_latin1_chars(&u, base_name[i]);
		CHECK(u32scmp(godot_string_get_data(&u), godot_string_get_data(&t)) == 0);
		godot_string_destroy(&u);
		godot_string_destroy(&t);

		t = godot_string_get_extension(&s);
		godot_string_new_with_latin1_chars(&u, ext[i]);
		CHECK(u32scmp(godot_string_get_data(&u), godot_string_get_data(&t)) == 0);
		godot_string_destroy(&u);
		godot_string_destroy(&t);

		t = godot_string_get_file(&s);
		godot_string_new_with_latin1_chars(&u, file[i]);
		CHECK(u32scmp(godot_string_get_data(&u), godot_string_get_data(&t)) == 0);
		godot_string_destroy(&u);
		godot_string_destroy(&t);

		godot_string s_simp;
		s_simp = godot_string_simplify_path(&s);
		t = godot_string_get_base_dir(&s_simp);
		godot_string_new_with_latin1_chars(&u, file[i]);
		f = godot_string_plus_file(&t, &u);
		CHECK(u32scmp(godot_string_get_data(&f), godot_string_get_data(&s_simp)) == 0);
		godot_string_destroy(&f);
		godot_string_destroy(&u);
		godot_string_destroy(&t);
		godot_string_destroy(&s_simp);

		CHECK(godot_string_is_abs_path(&s) == abs[i]);
		CHECK(godot_string_is_rel_path(&s) != abs[i]);

		godot_string_destroy(&s);
	}

	static const char *file_name[3] = { "test.tscn", "test://.xscn", "?tes*t.scn" };
	static const bool valid[3] = { true, false, false };
	for (int i = 0; i < 3; i++) {
		godot_string s;
		godot_string_new_with_latin1_chars(&s, file_name[i]);
		CHECK(godot_string_is_valid_filename(&s) == valid[i]);
		godot_string_destroy(&s);
	}
}

TEST_CASE("[GDNative String] hash") {
	godot_string a, b, c;
	godot_string_new_with_latin1_chars(&a, "Test");
	godot_string_new_with_latin1_chars(&b, "Test");
	godot_string_new_with_latin1_chars(&c, "West");
	CHECK(godot_string_hash(&a) == godot_string_hash(&b));
	CHECK(godot_string_hash(&a) != godot_string_hash(&c));

	CHECK(godot_string_hash64(&a) == godot_string_hash64(&b));
	CHECK(godot_string_hash64(&a) != godot_string_hash64(&c));

	godot_string_destroy(&a);
	godot_string_destroy(&b);
	godot_string_destroy(&c);
}

TEST_CASE("[GDNative String] http_escape/unescape") {
	godot_string s, t, u;
	godot_string_new_with_latin1_chars(&s, "Godot Engine:'docs'");
	godot_string_new_with_latin1_chars(&t, "Godot%20Engine%3A%27docs%27");

	u = godot_string_http_escape(&s);
	CHECK(u32scmp(godot_string_get_data(&u), godot_string_get_data(&t)) == 0);
	godot_string_destroy(&u);

	u = godot_string_http_unescape(&t);
	CHECK(u32scmp(godot_string_get_data(&u), godot_string_get_data(&s)) == 0);
	godot_string_destroy(&u);

	godot_string_destroy(&s);
	godot_string_destroy(&t);
}

TEST_CASE("[GDNative String] percent_encode/decode") {
	godot_string s, t, u;
	godot_string_new_with_latin1_chars(&s, "Godot Engine:'docs'");
	godot_string_new_with_latin1_chars(&t, "Godot%20Engine%3a%27docs%27");

	u = godot_string_percent_encode(&s);
	CHECK(u32scmp(godot_string_get_data(&u), godot_string_get_data(&t)) == 0);
	godot_string_destroy(&u);

	u = godot_string_percent_decode(&t);
	CHECK(u32scmp(godot_string_get_data(&u), godot_string_get_data(&s)) == 0);
	godot_string_destroy(&u);

	godot_string_destroy(&s);
	godot_string_destroy(&t);
}

TEST_CASE("[GDNative String] xml_escape/unescape") {
	godot_string s, t, u;
	godot_string_new_with_latin1_chars(&s, "\"Test\" <test@test&'test'>");

	t = godot_string_xml_escape_with_quotes(&s);
	u = godot_string_xml_unescape(&t);
	CHECK(u32scmp(godot_string_get_data(&u), godot_string_get_data(&s)) == 0);
	godot_string_destroy(&u);
	godot_string_destroy(&t);

	t = godot_string_xml_escape(&s);
	u = godot_string_xml_unescape(&t);
	CHECK(u32scmp(godot_string_get_data(&u), godot_string_get_data(&s)) == 0);
	godot_string_destroy(&u);
	godot_string_destroy(&t);

	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] Strip escapes") {
	godot_string s, t, u;
	godot_string_new_with_latin1_chars(&s, "\t\tTest Test\r\n Test");
	godot_string_new_with_latin1_chars(&t, "Test Test Test");

	u = godot_string_strip_escapes(&s);
	CHECK(u32scmp(godot_string_get_data(&u), godot_string_get_data(&t)) == 0);
	godot_string_destroy(&u);

	godot_string_destroy(&t);
	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] Strip edges") {
	godot_string s, t, u;
	godot_string_new_with_latin1_chars(&s, "\t Test Test   ");

	godot_string_new_with_latin1_chars(&t, "Test Test   ");
	u = godot_string_strip_edges(&s, true, false);
	CHECK(u32scmp(godot_string_get_data(&u), godot_string_get_data(&t)) == 0);
	godot_string_destroy(&u);
	godot_string_destroy(&t);

	godot_string_new_with_latin1_chars(&t, "\t Test Test");
	u = godot_string_strip_edges(&s, false, true);
	CHECK(u32scmp(godot_string_get_data(&u), godot_string_get_data(&t)) == 0);
	godot_string_destroy(&u);
	godot_string_destroy(&t);

	godot_string_new_with_latin1_chars(&t, "Test Test");
	u = godot_string_strip_edges(&s, true, true);
	CHECK(u32scmp(godot_string_get_data(&u), godot_string_get_data(&t)) == 0);
	godot_string_destroy(&u);
	godot_string_destroy(&t);

	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] Similarity") {
	godot_string a, b, c;
	godot_string_new_with_latin1_chars(&a, "Test");
	godot_string_new_with_latin1_chars(&b, "West");
	godot_string_new_with_latin1_chars(&c, "Toad");

	CHECK(godot_string_similarity(&a, &b) > godot_string_similarity(&a, &c));

	godot_string_destroy(&a);
	godot_string_destroy(&b);
	godot_string_destroy(&c);
}

TEST_CASE("[GDNative String] Trim") {
	godot_string s, t, u, p;
	godot_string_new_with_latin1_chars(&s, "aaaTestbbb");

	godot_string_new_with_latin1_chars(&p, "aaa");
	godot_string_new_with_latin1_chars(&t, "Testbbb");
	u = godot_string_trim_prefix(&s, &p);
	CHECK(u32scmp(godot_string_get_data(&u), godot_string_get_data(&t)) == 0);
	godot_string_destroy(&u);
	godot_string_destroy(&t);
	godot_string_destroy(&p);

	godot_string_new_with_latin1_chars(&p, "bbb");
	godot_string_new_with_latin1_chars(&t, "aaaTest");
	u = godot_string_trim_suffix(&s, &p);
	CHECK(u32scmp(godot_string_get_data(&u), godot_string_get_data(&t)) == 0);
	godot_string_destroy(&u);
	godot_string_destroy(&t);
	godot_string_destroy(&p);

	godot_string_new_with_latin1_chars(&p, "Test");
	u = godot_string_trim_suffix(&s, &p);
	CHECK(u32scmp(godot_string_get_data(&u), godot_string_get_data(&s)) == 0);
	godot_string_destroy(&u);
	godot_string_destroy(&p);

	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] Right/Left") {
	godot_string s, t, u;
	godot_string_new_with_latin1_chars(&s, "aaaTestbbb");
	//                                            ^

	godot_string_new_with_latin1_chars(&t, "tbbb");
	u = godot_string_right(&s, 6);
	CHECK(u32scmp(godot_string_get_data(&u), godot_string_get_data(&t)) == 0);
	godot_string_destroy(&u);
	godot_string_destroy(&t);

	godot_string_new_with_latin1_chars(&t, "aaaTes");
	u = godot_string_left(&s, 6);
	CHECK(u32scmp(godot_string_get_data(&u), godot_string_get_data(&t)) == 0);
	godot_string_destroy(&u);
	godot_string_destroy(&t);

	godot_string_destroy(&s);
}

TEST_CASE("[GDNative String] Repeat") {
	godot_string t, u;
	godot_string_new_with_latin1_chars(&t, "ab");

	u = godot_string_repeat(&t, 4);
	CHECK(u32scmp(godot_string_get_data(&u), U"abababab") == 0);
	godot_string_destroy(&u);

	godot_string_destroy(&t);
}

TEST_CASE("[GDNative String] SHA1/SHA256/MD5") {
	godot_string s, t, sha1, sha256, md5;
	godot_string_new_with_latin1_chars(&s, "Godot");
	godot_string_new_with_latin1_chars(&sha1, "a1e91f39b9fce6a9998b14bdbe2aa2b39dc2d201");
	static uint8_t sha1_buf[20] = {
		0xA1, 0xE9, 0x1F, 0x39, 0xB9, 0xFC, 0xE6, 0xA9, 0x99, 0x8B, 0x14, 0xBD, 0xBE, 0x2A, 0xA2, 0xB3,
		0x9D, 0xC2, 0xD2, 0x01
	};
	godot_string_new_with_latin1_chars(&sha256, "2a02b2443f7985d89d09001086ae3dcfa6eb0f55c6ef170715d42328e16e6cb8");
	static uint8_t sha256_buf[32] = {
		0x2A, 0x02, 0xB2, 0x44, 0x3F, 0x79, 0x85, 0xD8, 0x9D, 0x09, 0x00, 0x10, 0x86, 0xAE, 0x3D, 0xCF,
		0xA6, 0xEB, 0x0F, 0x55, 0xC6, 0xEF, 0x17, 0x07, 0x15, 0xD4, 0x23, 0x28, 0xE1, 0x6E, 0x6C, 0xB8
	};
	godot_string_new_with_latin1_chars(&md5, "4a336d087aeb0390da10ee2ea7cb87f8");
	static uint8_t md5_buf[16] = {
		0x4A, 0x33, 0x6D, 0x08, 0x7A, 0xEB, 0x03, 0x90, 0xDA, 0x10, 0xEE, 0x2E, 0xA7, 0xCB, 0x87, 0xF8
	};

	godot_packed_byte_array buf = godot_string_sha1_buffer(&s);
	CHECK(memcmp(sha1_buf, godot_packed_byte_array_ptr(&buf), 20) == 0);
	godot_packed_byte_array_destroy(&buf);

	t = godot_string_sha1_text(&s);
	CHECK(u32scmp(godot_string_get_data(&t), godot_string_get_data(&sha1)) == 0);
	godot_string_destroy(&t);

	buf = godot_string_sha256_buffer(&s);
	CHECK(memcmp(sha256_buf, godot_packed_byte_array_ptr(&buf), 32) == 0);
	godot_packed_byte_array_destroy(&buf);

	t = godot_string_sha256_text(&s);
	CHECK(u32scmp(godot_string_get_data(&t), godot_string_get_data(&sha256)) == 0);
	godot_string_destroy(&t);

	buf = godot_string_md5_buffer(&s);
	CHECK(memcmp(md5_buf, godot_packed_byte_array_ptr(&buf), 16) == 0);
	godot_packed_byte_array_destroy(&buf);

	t = godot_string_md5_text(&s);
	CHECK(u32scmp(godot_string_get_data(&t), godot_string_get_data(&md5)) == 0);
	godot_string_destroy(&t);

	godot_string_destroy(&s);
	godot_string_destroy(&sha1);
	godot_string_destroy(&sha256);
	godot_string_destroy(&md5);
}

TEST_CASE("[GDNative String] Join") {
	godot_string s, t, u;
	godot_string_new_with_latin1_chars(&s, ", ");

	godot_packed_string_array parts;
	godot_packed_string_array_new(&parts);
	godot_string_new_with_latin1_chars(&t, "One");
	godot_packed_string_array_push_back(&parts, &t);
	godot_string_destroy(&t);
	godot_string_new_with_latin1_chars(&t, "B");
	godot_packed_string_array_push_back(&parts, &t);
	godot_string_destroy(&t);
	godot_string_new_with_latin1_chars(&t, "C");
	godot_packed_string_array_push_back(&parts, &t);
	godot_string_destroy(&t);

	godot_string_new_with_latin1_chars(&u, "One, B, C");
	t = godot_string_join(&s, &parts);
	CHECK(u32scmp(godot_string_get_data(&u), godot_string_get_data(&t)) == 0);
	godot_string_destroy(&u);
	godot_string_destroy(&t);

	godot_string_destroy(&s);
	godot_packed_string_array_destroy(&parts);
}

TEST_CASE("[GDNative String] Is_*") {
	static const char *data[12] = { "-30", "100", "10.1", "10,1", "1e2", "1e-2", "1e2e3", "0xAB", "AB", "Test1", "1Test", "Test*1" };
	static bool isnum[12] = { true, true, true, false, false, false, false, false, false, false, false, false };
	static bool isint[12] = { true, true, false, false, false, false, false, false, false, false, false, false };
	static bool ishex[12] = { true, true, false, false, true, false, true, false, true, false, false, false };
	static bool ishex_p[12] = { false, false, false, false, false, false, false, true, false, false, false, false };
	static bool isflt[12] = { true, true, true, false, true, true, false, false, false, false, false, false };
	static bool isid[12] = { false, false, false, false, false, false, false, false, true, true, false, false };

	for (int i = 0; i < 12; i++) {
		godot_string s;
		godot_string_new_with_latin1_chars(&s, data[i]);
		CHECK(godot_string_is_numeric(&s) == isnum[i]);
		CHECK(godot_string_is_valid_integer(&s) == isint[i]);
		CHECK(godot_string_is_valid_hex_number(&s, false) == ishex[i]);
		CHECK(godot_string_is_valid_hex_number(&s, true) == ishex_p[i]);
		CHECK(godot_string_is_valid_float(&s) == isflt[i]);
		CHECK(godot_string_is_valid_identifier(&s) == isid[i]);
		godot_string_destroy(&s);
	}
}

TEST_CASE("[GDNative String] humanize_size") {
	godot_string s;

	s = godot_string_humanize_size(1000);
	CHECK(u32scmp(godot_string_get_data(&s), U"1000 B") == 0);
	godot_string_destroy(&s);

	s = godot_string_humanize_size(1025);
	CHECK(u32scmp(godot_string_get_data(&s), U"1.00 KiB") == 0);
	godot_string_destroy(&s);

	s = godot_string_humanize_size(1025300);
	CHECK(u32scmp(godot_string_get_data(&s), U"1001.2 KiB") == 0);
	godot_string_destroy(&s);

	s = godot_string_humanize_size(100523550);
	CHECK(u32scmp(godot_string_get_data(&s), U"95.86 MiB") == 0);
	godot_string_destroy(&s);

	s = godot_string_humanize_size(5345555000);
	CHECK(u32scmp(godot_string_get_data(&s), U"4.97 GiB") == 0);
	godot_string_destroy(&s);
}

} // namespace TestGDNativeString

#endif // TEST_GDNATIVE_STRING_H
