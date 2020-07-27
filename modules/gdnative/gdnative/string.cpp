/*************************************************************************/
/*  string.cpp                                                           */
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

#include "gdnative/string.h"

#include "core/string_name.h"
#include "core/ustring.h"
#include "core/variant.h"

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static_assert(sizeof(godot_char16_string) == sizeof(Char16String), "Char16String size mismatch");
static_assert(sizeof(godot_char_string) == sizeof(CharString), "CharString size mismatch");
static_assert(sizeof(godot_string) == sizeof(String), "String size mismatch");
static_assert(sizeof(godot_char_type) == sizeof(char32_t), "char32_t size mismatch");

godot_int GDAPI godot_char_string_length(const godot_char_string *p_cs) {
	const CharString *cs = (const CharString *)p_cs;

	return cs->length();
}

const char GDAPI *godot_char_string_get_data(const godot_char_string *p_cs) {
	const CharString *cs = (const CharString *)p_cs;

	return cs->get_data();
}

void GDAPI godot_char_string_destroy(godot_char_string *p_cs) {
	CharString *cs = (CharString *)p_cs;

	cs->~CharString();
}

godot_int GDAPI godot_char16_string_length(const godot_char16_string *p_cs) {
	const Char16String *cs = (const Char16String *)p_cs;

	return cs->length();
}

const char16_t GDAPI *godot_char16_string_get_data(const godot_char16_string *p_cs) {
	const Char16String *cs = (const Char16String *)p_cs;

	return cs->get_data();
}

void GDAPI godot_char16_string_destroy(godot_char16_string *p_cs) {
	Char16String *cs = (Char16String *)p_cs;

	cs->~Char16String();
}

void GDAPI godot_string_new(godot_string *r_dest) {
	String *dest = (String *)r_dest;
	memnew_placement(dest, String);
}

void GDAPI godot_string_new_copy(godot_string *r_dest, const godot_string *p_src) {
	String *dest = (String *)r_dest;
	const String *src = (const String *)p_src;
	memnew_placement(dest, String);
	*dest = String(*src);
}

void GDAPI godot_string_new_with_latin1_chars(godot_string *r_dest, const char *p_contents) {
	String *dest = (String *)r_dest;
	memnew_placement(dest, String);
	*dest = String(p_contents);
}

void GDAPI godot_string_new_with_utf8_chars(godot_string *r_dest, const char *p_contents) {
	String *dest = (String *)r_dest;
	memnew_placement(dest, String);
	dest->parse_utf8(p_contents);
}

void GDAPI godot_string_new_with_utf16_chars(godot_string *r_dest, const char16_t *p_contents) {
	String *dest = (String *)r_dest;
	memnew_placement(dest, String);
	dest->parse_utf16(p_contents);
}

void GDAPI godot_string_new_with_utf32_chars(godot_string *r_dest, const char32_t *p_contents) {
	String *dest = (String *)r_dest;
	memnew_placement(dest, String);
	*dest = String((const char32_t *)p_contents);
}

void GDAPI godot_string_new_with_wide_chars(godot_string *r_dest, const wchar_t *p_contents) {
	String *dest = (String *)r_dest;
	if (sizeof(wchar_t) == 2) {
		// wchar_t is 16 bit, parse.
		memnew_placement(dest, String);
		dest->parse_utf16((const char16_t *)p_contents);
	} else {
		// wchar_t is 32 bit, copy.
		memnew_placement(dest, String);
		*dest = String((const char32_t *)p_contents);
	}
}

void GDAPI godot_string_new_with_latin1_chars_and_len(godot_string *r_dest, const char *p_contents, const int p_size) {
	String *dest = (String *)r_dest;
	memnew_placement(dest, String);
	*dest = String(p_contents, p_size);
}

void GDAPI godot_string_new_with_utf8_chars_and_len(godot_string *r_dest, const char *p_contents, const int p_size) {
	String *dest = (String *)r_dest;
	memnew_placement(dest, String);
	dest->parse_utf8(p_contents, p_size);
}

void GDAPI godot_string_new_with_utf16_chars_and_len(godot_string *r_dest, const char16_t *p_contents, const int p_size) {
	String *dest = (String *)r_dest;
	memnew_placement(dest, String);
	dest->parse_utf16(p_contents, p_size);
}

void GDAPI godot_string_new_with_utf32_chars_and_len(godot_string *r_dest, const char32_t *p_contents, const int p_size) {
	String *dest = (String *)r_dest;
	memnew_placement(dest, String);
	*dest = String((const char32_t *)p_contents, p_size);
}

void GDAPI godot_string_new_with_wide_chars_and_len(godot_string *r_dest, const wchar_t *p_contents, const int p_size) {
	String *dest = (String *)r_dest;
	if (sizeof(wchar_t) == 2) {
		// wchar_t is 16 bit, parse.
		memnew_placement(dest, String);
		dest->parse_utf16((const char16_t *)p_contents, p_size);
	} else {
		// wchar_t is 32 bit, copy.
		memnew_placement(dest, String);
		*dest = String((const char32_t *)p_contents, p_size);
	}
}

const godot_char_type GDAPI *godot_string_operator_index(godot_string *p_self, const godot_int p_idx) {
	String *self = (String *)p_self;
	return &(self->operator[](p_idx));
}

godot_char_type GDAPI godot_string_operator_index_const(const godot_string *p_self, const godot_int p_idx) {
	const String *self = (const String *)p_self;
	return self->operator[](p_idx);
}

const godot_char_type GDAPI *godot_string_get_data(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	return self->get_data();
}

godot_bool GDAPI godot_string_operator_equal(const godot_string *p_self, const godot_string *p_b) {
	const String *self = (const String *)p_self;
	const String *b = (const String *)p_b;
	return *self == *b;
}

godot_bool GDAPI godot_string_operator_less(const godot_string *p_self, const godot_string *p_b) {
	const String *self = (const String *)p_self;
	const String *b = (const String *)p_b;
	return *self < *b;
}

godot_string GDAPI godot_string_operator_plus(const godot_string *p_self, const godot_string *p_b) {
	godot_string ret;
	const String *self = (const String *)p_self;
	const String *b = (const String *)p_b;
	memnew_placement(&ret, String(*self + *b));
	return ret;
}

void GDAPI godot_string_destroy(godot_string *p_self) {
	String *self = (String *)p_self;
	self->~String();
}

/* Standard size stuff */

godot_int GDAPI godot_string_length(const godot_string *p_self) {
	const String *self = (const String *)p_self;

	return self->length();
}

/* Helpers */

signed char GDAPI godot_string_casecmp_to(const godot_string *p_self, const godot_string *p_str) {
	const String *self = (const String *)p_self;
	const String *str = (const String *)p_str;

	return self->casecmp_to(*str);
}

signed char GDAPI godot_string_nocasecmp_to(const godot_string *p_self, const godot_string *p_str) {
	const String *self = (const String *)p_self;
	const String *str = (const String *)p_str;

	return self->nocasecmp_to(*str);
}

signed char GDAPI godot_string_naturalnocasecmp_to(const godot_string *p_self, const godot_string *p_str) {
	const String *self = (const String *)p_self;
	const String *str = (const String *)p_str;

	return self->naturalnocasecmp_to(*str);
}

godot_bool GDAPI godot_string_begins_with(const godot_string *p_self, const godot_string *p_string) {
	const String *self = (const String *)p_self;
	const String *string = (const String *)p_string;

	return self->begins_with(*string);
}

godot_bool GDAPI godot_string_begins_with_char_array(const godot_string *p_self, const char *p_char_array) {
	const String *self = (const String *)p_self;

	return self->begins_with(p_char_array);
}

godot_packed_string_array GDAPI godot_string_bigrams(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_packed_string_array ret;
	memnew_placement(&ret, Vector<String>(self->bigrams()));
	return ret;
};

godot_string GDAPI godot_string_chr(godot_char_type p_character) {
	godot_string result;
	memnew_placement(&result, String(String::chr(p_character)));

	return result;
}

godot_bool GDAPI godot_string_ends_with(const godot_string *p_self, const godot_string *p_string) {
	const String *self = (const String *)p_self;
	const String *string = (const String *)p_string;

	return self->ends_with(*string);
}

godot_bool GDAPI godot_string_ends_with_char_array(const godot_string *p_self, const char *p_char_array) {
	const String *self = (const String *)p_self;

	return self->ends_with(p_char_array);
}

godot_int GDAPI godot_string_count(const godot_string *p_self, const godot_string *p_what, godot_int p_from, godot_int p_to) {
	const String *self = (const String *)p_self;
	const String *what = (const String *)p_what;

	return self->count(*what, p_from, p_to);
}

godot_int GDAPI godot_string_countn(const godot_string *p_self, const godot_string *p_what, godot_int p_from, godot_int p_to) {
	const String *self = (const String *)p_self;
	const String *what = (const String *)p_what;

	return self->countn(*what, p_from, p_to);
}

godot_int GDAPI godot_string_find(const godot_string *p_self, const godot_string *p_what) {
	const String *self = (const String *)p_self;
	const String *what = (const String *)p_what;

	return self->find(*what);
}

godot_int GDAPI godot_string_find_from(const godot_string *p_self, const godot_string *p_what, godot_int p_from) {
	const String *self = (const String *)p_self;
	const String *what = (const String *)p_what;

	return self->find(*what, p_from);
}

godot_int GDAPI godot_string_findmk(const godot_string *p_self, const godot_packed_string_array *p_keys) {
	const String *self = (const String *)p_self;
	const Vector<String> *keys = (const Vector<String> *)p_keys;
	return self->findmk(*keys);
}

godot_int GDAPI godot_string_findmk_from(const godot_string *p_self, const godot_packed_string_array *p_keys, godot_int p_from) {
	const String *self = (const String *)p_self;
	const Vector<String> *keys = (const Vector<String> *)p_keys;
	return self->findmk(*keys, p_from);
}

godot_int GDAPI godot_string_findmk_from_in_place(const godot_string *p_self, const godot_packed_string_array *p_keys, godot_int p_from, godot_int *r_key) {
	const String *self = (const String *)p_self;
	const Vector<String> *keys = (const Vector<String> *)p_keys;
	int key;
	int ret = self->findmk(*keys, p_from, &key);
	if (r_key) {
		*r_key = key;
	}
	return ret;
}

godot_int GDAPI godot_string_findn(const godot_string *p_self, const godot_string *p_what) {
	const String *self = (const String *)p_self;
	const String *what = (const String *)p_what;

	return self->findn(*what);
}

godot_int GDAPI godot_string_findn_from(const godot_string *p_self, const godot_string *p_what, godot_int p_from) {
	const String *self = (const String *)p_self;
	const String *what = (const String *)p_what;

	return self->findn(*what, p_from);
}

godot_string GDAPI godot_string_format(const godot_string *p_self, const godot_variant *p_values) {
	const String *self = (const String *)p_self;
	const Variant *values = (const Variant *)p_values;
	godot_string result;
	memnew_placement(&result, String(self->format(*values)));

	return result;
}

godot_string GDAPI godot_string_format_with_custom_placeholder(const godot_string *p_self, const godot_variant *p_values, const char *p_placeholder) {
	const String *self = (const String *)p_self;
	const Variant *values = (const Variant *)p_values;
	String placeholder = String(p_placeholder);
	godot_string result;
	memnew_placement(&result, String(self->format(*values, placeholder)));

	return result;
}

godot_string GDAPI godot_string_hex_encode_buffer(const uint8_t *p_buffer, godot_int p_len) {
	godot_string result;
	memnew_placement(&result, String(String::hex_encode_buffer(p_buffer, p_len)));

	return result;
}

godot_string GDAPI godot_string_insert(const godot_string *p_self, godot_int p_at_pos, const godot_string *p_string) {
	const String *self = (const String *)p_self;
	const String *content = (const String *)p_string;
	godot_string result;
	memnew_placement(&result, String(self->insert(p_at_pos, *content)));

	return result;
}

godot_bool GDAPI godot_string_is_numeric(const godot_string *p_self) {
	const String *self = (const String *)p_self;

	return self->is_numeric();
}

godot_bool GDAPI godot_string_is_subsequence_of(const godot_string *p_self, const godot_string *p_string) {
	const String *self = (const String *)p_self;
	const String *string = (const String *)p_string;

	return self->is_subsequence_of(*string);
}

godot_bool GDAPI godot_string_is_subsequence_ofi(const godot_string *p_self, const godot_string *p_string) {
	const String *self = (const String *)p_self;
	const String *string = (const String *)p_string;

	return self->is_subsequence_ofi(*string);
}

godot_string GDAPI godot_string_lpad(const godot_string *p_self, godot_int p_min_length) {
	const String *self = (const String *)p_self;
	godot_string result;
	memnew_placement(&result, String(self->lpad(p_min_length)));

	return result;
}

godot_string GDAPI godot_string_lpad_with_custom_character(const godot_string *p_self, godot_int p_min_length, const godot_string *p_character) {
	const String *self = (const String *)p_self;
	const String *character = (const String *)p_character;
	godot_string result;
	memnew_placement(&result, String(self->lpad(p_min_length, *character)));

	return result;
}

godot_bool GDAPI godot_string_match(const godot_string *p_self, const godot_string *p_wildcard) {
	const String *self = (const String *)p_self;
	const String *wildcard = (const String *)p_wildcard;

	return self->match(*wildcard);
}

godot_bool GDAPI godot_string_matchn(const godot_string *p_self, const godot_string *p_wildcard) {
	const String *self = (const String *)p_self;
	const String *wildcard = (const String *)p_wildcard;

	return self->matchn(*wildcard);
}

godot_string GDAPI godot_string_md5(const uint8_t *p_md5) {
	godot_string result;
	memnew_placement(&result, String(String::md5(p_md5)));

	return result;
}

godot_string GDAPI godot_string_num(double p_num) {
	godot_string result;
	memnew_placement(&result, String(String::num(p_num)));

	return result;
}

godot_string GDAPI godot_string_num_int64(int64_t p_num, godot_int p_base) {
	godot_string result;
	memnew_placement(&result, String(String::num_int64(p_num, p_base)));

	return result;
}

godot_string GDAPI godot_string_num_int64_capitalized(int64_t p_num, godot_int p_base, godot_bool p_capitalize_hex) {
	godot_string result;
	memnew_placement(&result, String(String::num_int64(p_num, p_base, true)));

	return result;
}

godot_string GDAPI godot_string_num_real(double p_num) {
	godot_string result;
	memnew_placement(&result, String(String::num_real(p_num)));

	return result;
}

godot_string GDAPI godot_string_num_scientific(double p_num) {
	godot_string result;
	memnew_placement(&result, String(String::num_scientific(p_num)));

	return result;
}

godot_string GDAPI godot_string_num_with_decimals(double p_num, godot_int p_decimals) {
	godot_string result;
	memnew_placement(&result, String(String::num(p_num, p_decimals)));

	return result;
}

godot_string GDAPI godot_string_pad_decimals(const godot_string *p_self, godot_int p_digits) {
	const String *self = (const String *)p_self;
	godot_string result;
	memnew_placement(&result, String(self->pad_decimals(p_digits)));

	return result;
}

godot_string GDAPI godot_string_pad_zeros(const godot_string *p_self, godot_int p_digits) {
	const String *self = (const String *)p_self;
	godot_string result;
	memnew_placement(&result, String(self->pad_zeros(p_digits)));

	return result;
}

godot_string GDAPI godot_string_replace(const godot_string *p_self, const godot_string *p_key, const godot_string *p_with) {
	const String *self = (const String *)p_self;
	const String *key = (const String *)p_key;
	const String *with = (const String *)p_with;
	godot_string result;
	memnew_placement(&result, String(self->replace(*key, *with)));

	return result;
}

godot_string GDAPI godot_string_replacen(const godot_string *p_self, const godot_string *p_key, const godot_string *p_with) {
	const String *self = (const String *)p_self;
	const String *key = (const String *)p_key;
	const String *with = (const String *)p_with;
	godot_string result;
	memnew_placement(&result, String(self->replacen(*key, *with)));

	return result;
}

godot_int GDAPI godot_string_rfind(const godot_string *p_self, const godot_string *p_what) {
	const String *self = (const String *)p_self;
	const String *what = (const String *)p_what;

	return self->rfind(*what);
}

godot_int GDAPI godot_string_rfindn(const godot_string *p_self, const godot_string *p_what) {
	const String *self = (const String *)p_self;
	const String *what = (const String *)p_what;

	return self->rfindn(*what);
}

godot_int GDAPI godot_string_rfind_from(const godot_string *p_self, const godot_string *p_what, godot_int p_from) {
	const String *self = (const String *)p_self;
	const String *what = (const String *)p_what;

	return self->rfind(*what, p_from);
}

godot_int GDAPI godot_string_rfindn_from(const godot_string *p_self, const godot_string *p_what, godot_int p_from) {
	const String *self = (const String *)p_self;
	const String *what = (const String *)p_what;

	return self->rfindn(*what, p_from);
}

godot_string GDAPI godot_string_replace_first(const godot_string *p_self, const godot_string *p_key, const godot_string *p_with) {
	const String *self = (const String *)p_self;
	const String *key = (const String *)p_key;
	const String *with = (const String *)p_with;
	godot_string result;
	memnew_placement(&result, String(self->replace_first(*key, *with)));

	return result;
}

godot_string GDAPI godot_string_rpad(const godot_string *p_self, godot_int p_min_length) {
	const String *self = (const String *)p_self;
	godot_string result;
	memnew_placement(&result, String(self->rpad(p_min_length)));

	return result;
}

godot_string GDAPI godot_string_rpad_with_custom_character(const godot_string *p_self, godot_int p_min_length, const godot_string *p_character) {
	const String *self = (const String *)p_self;
	const String *character = (const String *)p_character;
	godot_string result;
	memnew_placement(&result, String(self->rpad(p_min_length, *character)));

	return result;
}

godot_real GDAPI godot_string_similarity(const godot_string *p_self, const godot_string *p_string) {
	const String *self = (const String *)p_self;
	const String *string = (const String *)p_string;

	return self->similarity(*string);
}

godot_string GDAPI godot_string_sprintf(const godot_string *p_self, const godot_array *p_values, godot_bool *p_error) {
	const String *self = (const String *)p_self;
	const Array *values = (const Array *)p_values;

	godot_string result;
	String return_value = self->sprintf(*values, p_error);
	memnew_placement(&result, String(return_value));

	return result;
}

godot_string GDAPI godot_string_substr(const godot_string *p_self, godot_int p_from, godot_int p_chars) {
	const String *self = (const String *)p_self;
	godot_string result;
	memnew_placement(&result, String(self->substr(p_from, p_chars)));

	return result;
}

godot_int GDAPI godot_string_to_int(const godot_string *p_self) {
	const String *self = (const String *)p_self;

	return self->to_int();
}

double GDAPI godot_string_to_float(const godot_string *p_self) {
	const String *self = (const String *)p_self;

	return self->to_float();
}

godot_string GDAPI godot_string_capitalize(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	memnew_placement(&result, String(self->capitalize()));

	return result;
}

godot_string GDAPI godot_string_camelcase_to_underscore(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	memnew_placement(&result, String(self->camelcase_to_underscore(false)));

	return result;
}

godot_string GDAPI godot_string_camelcase_to_underscore_lowercased(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	memnew_placement(&result, String(self->camelcase_to_underscore()));

	return result;
}

double GDAPI godot_string_char_to_float(const char *p_what) {
	return String::to_float(p_what);
}

double GDAPI godot_string_wchar_to_float(const wchar_t *p_str, const wchar_t **r_end) {
	return String::to_float(p_str, r_end);
}

godot_int GDAPI godot_string_char_to_int(const char *p_what) {
	return String::to_int(p_what);
}

godot_int GDAPI godot_string_wchar_to_int(const wchar_t *p_str) {
	return String::to_int(p_str);
}

godot_int GDAPI godot_string_char_to_int_with_len(const char *p_what, godot_int p_len) {
	return String::to_int(p_what, p_len);
}

godot_int GDAPI godot_string_wchar_to_int_with_len(const wchar_t *p_str, int p_len) {
	return String::to_int(p_str, p_len);
}

godot_int GDAPI godot_string_hex_to_int(const godot_string *p_self) {
	const String *self = (const String *)p_self;

	return self->hex_to_int(false);
}

godot_int GDAPI godot_string_hex_to_int_with_prefix(const godot_string *p_self) {
	const String *self = (const String *)p_self;

	return self->hex_to_int();
}

godot_string GDAPI godot_string_get_slice(const godot_string *p_self, const godot_string *p_splitter, godot_int p_slice) {
	const String *self = (const String *)p_self;
	const String *splitter = (const String *)p_splitter;
	godot_string result;
	memnew_placement(&result, String(self->get_slice(*splitter, p_slice)));

	return result;
}

godot_string GDAPI godot_string_get_slicec(const godot_string *p_self, godot_char_type p_splitter, godot_int p_slice) {
	const String *self = (const String *)p_self;
	godot_string result;
	memnew_placement(&result, String(self->get_slicec(p_splitter, p_slice)));

	return result;
}

godot_packed_string_array GDAPI godot_string_split(const godot_string *p_self, const godot_string *p_splitter) {
	const String *self = (const String *)p_self;
	const String *splitter = (const String *)p_splitter;
	godot_packed_string_array ret;
	memnew_placement(&ret, Vector<String>(self->split(*splitter, false)));
	return ret;
}

godot_packed_string_array GDAPI godot_string_split_allow_empty(const godot_string *p_self, const godot_string *p_splitter) {
	const String *self = (const String *)p_self;
	const String *splitter = (const String *)p_splitter;
	godot_packed_string_array ret;
	memnew_placement(&ret, Vector<String>(self->split(*splitter, true)));
	return ret;
}

godot_packed_string_array GDAPI godot_string_split_with_maxsplit(const godot_string *p_self, const godot_string *p_splitter, const godot_bool p_allow_empty, const godot_int p_maxsplit) {
	const String *self = (const String *)p_self;
	const String *splitter = (const String *)p_splitter;
	godot_packed_string_array ret;
	memnew_placement(&ret, Vector<String>(self->split(*splitter, p_allow_empty, p_maxsplit)));
	return ret;
}

godot_packed_string_array GDAPI godot_string_rsplit(const godot_string *p_self, const godot_string *p_splitter) {
	const String *self = (const String *)p_self;
	const String *splitter = (const String *)p_splitter;

	godot_packed_string_array ret;
	memnew_placement(&ret, Vector<String>(self->rsplit(*splitter, false)));
	return ret;
}

godot_packed_string_array GDAPI godot_string_rsplit_allow_empty(const godot_string *p_self, const godot_string *p_splitter) {
	const String *self = (const String *)p_self;
	const String *splitter = (const String *)p_splitter;

	godot_packed_string_array ret;
	memnew_placement(&ret, Vector<String>(self->rsplit(*splitter, true)));
	return ret;
}

godot_packed_string_array GDAPI godot_string_rsplit_with_maxsplit(const godot_string *p_self, const godot_string *p_splitter, const godot_bool p_allow_empty, const godot_int p_maxsplit) {
	const String *self = (const String *)p_self;
	const String *splitter = (const String *)p_splitter;

	godot_packed_string_array ret;
	memnew_placement(&ret, Vector<String>(self->rsplit(*splitter, p_allow_empty, p_maxsplit)));
	return ret;
}

godot_packed_float32_array GDAPI godot_string_split_floats(const godot_string *p_self, const godot_string *p_splitter) {
	const String *self = (const String *)p_self;
	const String *splitter = (const String *)p_splitter;

	godot_packed_float32_array ret;
	memnew_placement(&ret, Vector<float>(self->split_floats(*splitter, false)));
	return ret;
}

godot_packed_float32_array GDAPI godot_string_split_floats_allow_empty(const godot_string *p_self, const godot_string *p_splitter) {
	const String *self = (const String *)p_self;
	const String *splitter = (const String *)p_splitter;

	godot_packed_float32_array ret;
	memnew_placement(&ret, Vector<float>(self->split_floats(*splitter, true)));
	return ret;
}

godot_packed_float32_array GDAPI godot_string_split_floats_mk(const godot_string *p_self, const godot_packed_string_array *p_splitters) {
	const String *self = (const String *)p_self;
	const Vector<String> *splitters = (const Vector<String> *)p_splitters;

	godot_packed_float32_array ret;
	memnew_placement(&ret, Vector<float>(self->split_floats_mk(*splitters, false)));
	return ret;
}

godot_packed_float32_array GDAPI godot_string_split_floats_mk_allow_empty(const godot_string *p_self, const godot_packed_string_array *p_splitters) {
	const String *self = (const String *)p_self;
	const Vector<String> *splitters = (const Vector<String> *)p_splitters;

	godot_packed_float32_array ret;
	memnew_placement(&ret, Vector<float>(self->split_floats_mk(*splitters, true)));
	return ret;
}

godot_packed_int32_array GDAPI godot_string_split_ints(const godot_string *p_self, const godot_string *p_splitter) {
	const String *self = (const String *)p_self;
	const String *splitter = (const String *)p_splitter;

	godot_packed_int32_array ret;
	memnew_placement(&ret, Vector<int>(self->split_ints(*splitter, false)));
	return ret;
}

godot_packed_int32_array GDAPI godot_string_split_ints_allow_empty(const godot_string *p_self, const godot_string *p_splitter) {
	const String *self = (const String *)p_self;
	const String *splitter = (const String *)p_splitter;

	godot_packed_int32_array ret;
	memnew_placement(&ret, Vector<int>(self->split_ints(*splitter, true)));
	return ret;
}

godot_packed_int32_array GDAPI godot_string_split_ints_mk(const godot_string *p_self, const godot_packed_string_array *p_splitters) {
	const String *self = (const String *)p_self;
	const Vector<String> *splitters = (const Vector<String> *)p_splitters;

	godot_packed_int32_array ret;
	memnew_placement(&ret, Vector<int>(self->split_ints_mk(*splitters, false)));
	return ret;
}

godot_packed_int32_array GDAPI godot_string_split_ints_mk_allow_empty(const godot_string *p_self, const godot_packed_string_array *p_splitters) {
	const String *self = (const String *)p_self;
	const Vector<String> *splitters = (const Vector<String> *)p_splitters;

	godot_packed_int32_array ret;
	memnew_placement(&ret, Vector<int>(self->split_ints_mk(*splitters, true)));
	return ret;
}

godot_packed_string_array GDAPI godot_string_split_spaces(const godot_string *p_self) {
	const String *self = (const String *)p_self;

	godot_packed_string_array ret;
	memnew_placement(&ret, Vector<String>(self->split_spaces()));
	return ret;
}

godot_int GDAPI godot_string_get_slice_count(const godot_string *p_self, const godot_string *p_splitter) {
	const String *self = (const String *)p_self;
	const String *splitter = (const String *)p_splitter;

	return self->get_slice_count(*splitter);
}

godot_char_type GDAPI godot_string_char_lowercase(godot_char_type p_char) {
	return String::char_lowercase(p_char);
}

godot_char_type GDAPI godot_string_char_uppercase(godot_char_type p_char) {
	return String::char_uppercase(p_char);
}

godot_string GDAPI godot_string_to_lower(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	memnew_placement(&result, String(self->to_lower()));

	return result;
}

godot_string GDAPI godot_string_to_upper(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	memnew_placement(&result, String(self->to_upper()));

	return result;
}

godot_string GDAPI godot_string_get_basename(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	memnew_placement(&result, String(self->get_basename()));

	return result;
}

godot_string GDAPI godot_string_get_extension(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	memnew_placement(&result, String(self->get_extension()));

	return result;
}

godot_string GDAPI godot_string_left(const godot_string *p_self, godot_int p_pos) {
	const String *self = (const String *)p_self;
	godot_string result;
	memnew_placement(&result, String(self->left(p_pos)));

	return result;
}

godot_char_type GDAPI godot_string_ord_at(const godot_string *p_self, godot_int p_idx) {
	const String *self = (const String *)p_self;

	return self->ord_at(p_idx);
}

godot_string GDAPI godot_string_plus_file(const godot_string *p_self, const godot_string *p_file) {
	const String *self = (const String *)p_self;
	const String *file = (const String *)p_file;
	godot_string result;
	memnew_placement(&result, String(self->plus_file(*file)));

	return result;
}

godot_string GDAPI godot_string_right(const godot_string *p_self, godot_int p_pos) {
	const String *self = (const String *)p_self;
	godot_string result;
	memnew_placement(&result, String(self->right(p_pos)));

	return result;
}

godot_string GDAPI godot_string_repeat(const godot_string *p_self, godot_int p_count) {
	const String *self = (const String *)p_self;
	godot_string result;
	memnew_placement(&result, String(self->repeat(p_count)));

	return result;
}

godot_string GDAPI godot_string_strip_edges(const godot_string *p_self, godot_bool p_left, godot_bool p_right) {
	const String *self = (const String *)p_self;
	godot_string result;
	memnew_placement(&result, String(self->strip_edges(p_left, p_right)));

	return result;
}

godot_string GDAPI godot_string_strip_escapes(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	memnew_placement(&result, String(self->strip_escapes()));

	return result;
}

void GDAPI godot_string_erase(godot_string *p_self, godot_int p_pos, godot_int p_chars) {
	String *self = (String *)p_self;

	return self->erase(p_pos, p_chars);
}

godot_char_string GDAPI godot_string_ascii(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_char_string result;

	memnew_placement(&result, CharString(self->ascii()));

	return result;
}

godot_char_string GDAPI godot_string_latin1(const godot_string *p_self) {
	const String *self = (const String *)p_self;

	godot_char_string result;

	memnew_placement(&result, CharString(self->ascii(true)));

	return result;
}

godot_char_string GDAPI godot_string_utf8(const godot_string *p_self) {
	const String *self = (const String *)p_self;

	godot_char_string result;

	memnew_placement(&result, CharString(self->utf8()));

	return result;
}

godot_bool GDAPI godot_string_parse_utf8(godot_string *p_self, const char *p_utf8) {
	String *self = (String *)p_self;

	return self->parse_utf8(p_utf8);
}

godot_bool GDAPI godot_string_parse_utf8_with_len(godot_string *p_self, const char *p_utf8, godot_int p_len) {
	String *self = (String *)p_self;

	return self->parse_utf8(p_utf8, p_len);
}

godot_string GDAPI godot_string_chars_to_utf8(const char *p_utf8) {
	godot_string result;
	memnew_placement(&result, String(String::utf8(p_utf8)));

	return result;
}

godot_string GDAPI godot_string_chars_to_utf8_with_len(const char *p_utf8, godot_int p_len) {
	godot_string result;
	memnew_placement(&result, String(String::utf8(p_utf8, p_len)));

	return result;
}

godot_char16_string GDAPI godot_string_utf16(const godot_string *p_self) {
	const String *self = (const String *)p_self;

	godot_char16_string result;

	memnew_placement(&result, Char16String(self->utf16()));

	return result;
}

godot_bool GDAPI godot_string_parse_utf16(godot_string *p_self, const char16_t *p_utf16) {
	String *self = (String *)p_self;

	return self->parse_utf16(p_utf16);
}

godot_bool GDAPI godot_string_parse_utf16_with_len(godot_string *p_self, const char16_t *p_utf16, godot_int p_len) {
	String *self = (String *)p_self;

	return self->parse_utf16(p_utf16, p_len);
}

godot_string GDAPI godot_string_chars_to_utf16(const char16_t *p_utf16) {
	godot_string result;
	memnew_placement(&result, String(String::utf16(p_utf16)));

	return result;
}

godot_string GDAPI godot_string_chars_to_utf16_with_len(const char16_t *p_utf16, godot_int p_len) {
	godot_string result;
	memnew_placement(&result, String(String::utf16(p_utf16, p_len)));

	return result;
}

uint32_t GDAPI godot_string_hash(const godot_string *p_self) {
	const String *self = (const String *)p_self;

	return self->hash();
}

uint64_t GDAPI godot_string_hash64(const godot_string *p_self) {
	const String *self = (const String *)p_self;

	return self->hash64();
}

uint32_t GDAPI godot_string_hash_chars(const char *p_cstr) {
	return String::hash(p_cstr);
}

uint32_t GDAPI godot_string_hash_chars_with_len(const char *p_cstr, godot_int p_len) {
	return String::hash(p_cstr, p_len);
}

uint32_t GDAPI godot_string_hash_wide_chars(const wchar_t *p_str) {
	return String::hash(p_str);
}

uint32_t GDAPI godot_string_hash_wide_chars_with_len(const wchar_t *p_str, godot_int p_len) {
	return String::hash(p_str, p_len);
}

godot_packed_byte_array GDAPI godot_string_md5_buffer(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_packed_byte_array result;
	memnew_placement(&result, PackedByteArray(self->md5_buffer()));
	return result;
}

godot_string GDAPI godot_string_md5_text(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	memnew_placement(&result, String(self->md5_text()));

	return result;
}

godot_packed_byte_array GDAPI godot_string_sha1_buffer(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_packed_byte_array result;
	memnew_placement(&result, PackedByteArray(self->sha1_buffer()));
	return result;
}

godot_string GDAPI godot_string_sha1_text(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	memnew_placement(&result, String(self->sha1_text()));

	return result;
}

godot_packed_byte_array GDAPI godot_string_sha256_buffer(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_packed_byte_array result;
	memnew_placement(&result, PackedByteArray(self->sha256_buffer()));
	return result;
}

godot_string GDAPI godot_string_sha256_text(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	memnew_placement(&result, String(self->sha256_text()));

	return result;
}

godot_bool godot_string_empty(const godot_string *p_self) {
	const String *self = (const String *)p_self;

	return self->empty();
}

// path functions
godot_string GDAPI godot_string_get_base_dir(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	String return_value = self->get_base_dir();
	memnew_placement(&result, String(return_value));

	return result;
}

godot_string GDAPI godot_string_get_file(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	String return_value = self->get_file();
	memnew_placement(&result, String(return_value));

	return result;
}

godot_string GDAPI godot_string_humanize_size(size_t p_size) {
	godot_string result;
	String return_value = String::humanize_size(p_size);
	memnew_placement(&result, String(return_value));

	return result;
}

godot_bool GDAPI godot_string_is_abs_path(const godot_string *p_self) {
	const String *self = (const String *)p_self;

	return self->is_abs_path();
}

godot_bool GDAPI godot_string_is_rel_path(const godot_string *p_self) {
	const String *self = (const String *)p_self;

	return self->is_rel_path();
}

godot_bool GDAPI godot_string_is_resource_file(const godot_string *p_self) {
	const String *self = (const String *)p_self;

	return self->is_resource_file();
}

godot_string GDAPI godot_string_path_to(const godot_string *p_self, const godot_string *p_path) {
	const String *self = (const String *)p_self;
	String *path = (String *)p_path;
	godot_string result;
	String return_value = self->path_to(*path);
	memnew_placement(&result, String(return_value));

	return result;
}

godot_string GDAPI godot_string_path_to_file(const godot_string *p_self, const godot_string *p_path) {
	const String *self = (const String *)p_self;
	String *path = (String *)p_path;
	godot_string result;
	String return_value = self->path_to_file(*path);
	memnew_placement(&result, String(return_value));

	return result;
}

godot_string GDAPI godot_string_simplify_path(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	String return_value = self->simplify_path();
	memnew_placement(&result, String(return_value));

	return result;
}

godot_string GDAPI godot_string_c_escape(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	String return_value = self->c_escape();
	memnew_placement(&result, String(return_value));

	return result;
}

godot_string GDAPI godot_string_c_escape_multiline(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	String return_value = self->c_escape_multiline();
	memnew_placement(&result, String(return_value));

	return result;
}

godot_string GDAPI godot_string_c_unescape(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	String return_value = self->c_unescape();
	memnew_placement(&result, String(return_value));

	return result;
}

godot_string GDAPI godot_string_http_escape(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	String return_value = self->http_escape();
	memnew_placement(&result, String(return_value));

	return result;
}

godot_string GDAPI godot_string_http_unescape(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	String return_value = self->http_unescape();
	memnew_placement(&result, String(return_value));

	return result;
}

godot_string GDAPI godot_string_json_escape(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	String return_value = self->json_escape();
	memnew_placement(&result, String(return_value));

	return result;
}

godot_string GDAPI godot_string_xml_escape(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	String return_value = self->xml_escape();
	memnew_placement(&result, String(return_value));

	return result;
}

godot_string GDAPI godot_string_xml_escape_with_quotes(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	String return_value = self->xml_escape(true);
	memnew_placement(&result, String(return_value));

	return result;
}

godot_string GDAPI godot_string_xml_unescape(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	String return_value = self->xml_unescape();
	memnew_placement(&result, String(return_value));

	return result;
}

godot_string GDAPI godot_string_percent_decode(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	String return_value = self->percent_decode();
	memnew_placement(&result, String(return_value));

	return result;
}

godot_string GDAPI godot_string_percent_encode(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	String return_value = self->percent_encode();
	memnew_placement(&result, String(return_value));

	return result;
}

godot_string GDAPI godot_string_join(const godot_string *p_self, const godot_packed_string_array *p_parts) {
	const String *self = (const String *)p_self;
	const Vector<String> *parts = (const Vector<String> *)p_parts;
	godot_string result;
	String return_value = self->join(*parts);
	memnew_placement(&result, String(return_value));

	return result;
}

godot_bool GDAPI godot_string_is_valid_filename(const godot_string *p_self) {
	const String *self = (const String *)p_self;

	return self->is_valid_filename();
}

godot_bool GDAPI godot_string_is_valid_float(const godot_string *p_self) {
	const String *self = (const String *)p_self;

	return self->is_valid_float();
}

godot_bool GDAPI godot_string_is_valid_hex_number(const godot_string *p_self, godot_bool p_with_prefix) {
	const String *self = (const String *)p_self;

	return self->is_valid_hex_number(p_with_prefix);
}

godot_bool GDAPI godot_string_is_valid_html_color(const godot_string *p_self) {
	const String *self = (const String *)p_self;

	return self->is_valid_html_color();
}

godot_bool GDAPI godot_string_is_valid_identifier(const godot_string *p_self) {
	const String *self = (const String *)p_self;

	return self->is_valid_identifier();
}

godot_bool GDAPI godot_string_is_valid_integer(const godot_string *p_self) {
	const String *self = (const String *)p_self;

	return self->is_valid_integer();
}

godot_bool GDAPI godot_string_is_valid_ip_address(const godot_string *p_self) {
	const String *self = (const String *)p_self;

	return self->is_valid_ip_address();
}

godot_string GDAPI godot_string_dedent(const godot_string *p_self) {
	const String *self = (const String *)p_self;
	godot_string result;
	String return_value = self->dedent();
	memnew_placement(&result, String(return_value));

	return result;
}

godot_string GDAPI godot_string_trim_prefix(const godot_string *p_self, const godot_string *p_prefix) {
	const String *self = (const String *)p_self;
	String *prefix = (String *)p_prefix;
	godot_string result;
	String return_value = self->trim_prefix(*prefix);
	memnew_placement(&result, String(return_value));

	return result;
}

godot_string GDAPI godot_string_trim_suffix(const godot_string *p_self, const godot_string *p_suffix) {
	const String *self = (const String *)p_self;
	String *suffix = (String *)p_suffix;
	godot_string result;
	String return_value = self->trim_suffix(*suffix);
	memnew_placement(&result, String(return_value));

	return result;
}

godot_string GDAPI godot_string_lstrip(const godot_string *p_self, const godot_string *p_chars) {
	const String *self = (const String *)p_self;
	String *chars = (String *)p_chars;
	godot_string result;
	String return_value = self->lstrip(*chars);
	memnew_placement(&result, String(return_value));

	return result;
}

godot_string GDAPI godot_string_rstrip(const godot_string *p_self, const godot_string *p_chars) {
	const String *self = (const String *)p_self;
	String *chars = (String *)p_chars;
	godot_string result;
	String return_value = self->rstrip(*chars);
	memnew_placement(&result, String(return_value));

	return result;
}

#ifdef __cplusplus
}
#endif
