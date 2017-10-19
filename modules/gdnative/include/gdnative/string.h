/*************************************************************************/
/*  string.h                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef GODOT_STRING_H
#define GODOT_STRING_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <wchar.h>

#define GODOT_STRING_SIZE sizeof(void *)

#ifndef GODOT_CORE_API_GODOT_STRING_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_STRING_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_STRING_SIZE];
} godot_string;
#endif

// reduce extern "C" nesting for VS2013
#ifdef __cplusplus
}
#endif

#include <gdnative/gdnative.h>
#include <gdnative/variant.h>

#ifdef __cplusplus
extern "C" {
#endif

void GDAPI godot_string_new(godot_string *r_dest);
void GDAPI godot_string_new_copy(godot_string *r_dest, const godot_string *p_src);
void GDAPI godot_string_new_data(godot_string *r_dest, const char *p_contents, const int p_size);
void GDAPI godot_string_new_unicode_data(godot_string *r_dest, const wchar_t *p_contents, const int p_size);

void GDAPI godot_string_get_data(const godot_string *p_self, char *p_dest, int *p_size);

wchar_t GDAPI *godot_string_operator_index(godot_string *p_self, const godot_int p_idx);
const char GDAPI *godot_string_c_str(const godot_string *p_self);
const wchar_t GDAPI *godot_string_unicode_str(const godot_string *p_self);

godot_bool GDAPI godot_string_operator_equal(const godot_string *p_self, const godot_string *p_b);
godot_bool GDAPI godot_string_operator_less(const godot_string *p_self, const godot_string *p_b);
godot_string GDAPI godot_string_operator_plus(const godot_string *p_self, const godot_string *p_b);

/* Standard size stuff */

godot_int GDAPI godot_string_length(const godot_string *p_self);

/* Helpers */

godot_bool GDAPI godot_string_begins_with(const godot_string *p_self, const godot_string *p_string);
godot_bool GDAPI godot_string_begins_with_char_array(const godot_string *p_self, const char *p_char_array);
godot_array GDAPI godot_string_bigrams(const godot_string *p_self);
godot_string GDAPI godot_string_chr(wchar_t p_character);
godot_bool GDAPI godot_string_ends_with(const godot_string *p_self, const godot_string *p_string);
godot_int GDAPI godot_string_find(const godot_string *p_self, godot_string p_what);
godot_int GDAPI godot_string_find_from(const godot_string *p_self, godot_string p_what, godot_int p_from);
godot_int GDAPI godot_string_findmk(const godot_string *p_self, const godot_array *p_keys);
godot_int GDAPI godot_string_findmk_from(const godot_string *p_self, const godot_array *p_keys, godot_int p_from);
godot_int GDAPI godot_string_findmk_from_in_place(const godot_string *p_self, const godot_array *p_keys, godot_int p_from, godot_int *r_key);
godot_int GDAPI godot_string_findn(const godot_string *p_self, godot_string p_what);
godot_int GDAPI godot_string_findn_from(const godot_string *p_self, godot_string p_what, godot_int p_from);
godot_int GDAPI godot_string_find_last(const godot_string *p_self, godot_string p_what);
godot_string GDAPI godot_string_format(const godot_string *p_self, const godot_variant *p_values);
godot_string GDAPI godot_string_format_with_custom_placeholder(const godot_string *p_self, const godot_variant *p_values, const char *p_placeholder);
godot_string GDAPI godot_string_hex_encode_buffer(const uint8_t *p_buffer, godot_int p_len);
godot_int GDAPI godot_string_hex_to_int(const godot_string *p_self);
godot_int GDAPI godot_string_hex_to_int_without_prefix(const godot_string *p_self);
godot_string GDAPI godot_string_insert(const godot_string *p_self, godot_int p_at_pos, godot_string p_string);
godot_bool GDAPI godot_string_is_numeric(const godot_string *p_self);
godot_bool GDAPI godot_string_is_subsequence_of(const godot_string *p_self, const godot_string *p_string);
godot_bool GDAPI godot_string_is_subsequence_ofi(const godot_string *p_self, const godot_string *p_string);
godot_string GDAPI godot_string_lpad(const godot_string *p_self, godot_int p_min_length);
godot_string GDAPI godot_string_lpad_with_custom_character(const godot_string *p_self, godot_int p_min_length, const godot_string *p_character);
godot_bool GDAPI godot_string_match(const godot_string *p_self, const godot_string *p_wildcard);
godot_bool GDAPI godot_string_matchn(const godot_string *p_self, const godot_string *p_wildcard);
godot_string GDAPI godot_string_md5(const uint8_t *p_md5);
godot_string GDAPI godot_string_num(double p_num);
godot_string GDAPI godot_string_num_int64(int64_t p_num, godot_int p_base);
godot_string GDAPI godot_string_num_int64_capitalized(int64_t p_num, godot_int p_base, godot_bool p_capitalize_hex);
godot_string GDAPI godot_string_num_real(double p_num);
godot_string GDAPI godot_string_num_scientific(double p_num);
godot_string GDAPI godot_string_num_with_decimals(double p_num, godot_int p_decimals);
godot_string GDAPI godot_string_pad_decimals(const godot_string *p_self, godot_int p_digits);
godot_string GDAPI godot_string_pad_zeros(const godot_string *p_self, godot_int p_digits);
godot_string GDAPI godot_string_replace_first(const godot_string *p_self, godot_string p_key, godot_string p_with);
godot_string GDAPI godot_string_replace(const godot_string *p_self, godot_string p_key, godot_string p_with);
godot_string GDAPI godot_string_replacen(const godot_string *p_self, godot_string p_key, godot_string p_with);
godot_int GDAPI godot_string_rfind(const godot_string *p_self, godot_string p_what);
godot_int GDAPI godot_string_rfindn(const godot_string *p_self, godot_string p_what);
godot_int GDAPI godot_string_rfind_from(const godot_string *p_self, godot_string p_what, godot_int p_from);
godot_int GDAPI godot_string_rfindn_from(const godot_string *p_self, godot_string p_what, godot_int p_from);
godot_string GDAPI godot_string_rpad(const godot_string *p_self, godot_int p_min_length);
godot_string GDAPI godot_string_rpad_with_custom_character(const godot_string *p_self, godot_int p_min_length, const godot_string *p_character);
godot_real GDAPI godot_string_similarity(const godot_string *p_self, const godot_string *p_string);
godot_string GDAPI godot_string_sprintf(const godot_string *p_self, const godot_array *p_values, godot_bool *p_error);
godot_string GDAPI godot_string_substr(const godot_string *p_self, godot_int p_from, godot_int p_chars);
double GDAPI godot_string_to_double(const godot_string *p_self);
godot_real GDAPI godot_string_to_float(const godot_string *p_self);
godot_int GDAPI godot_string_to_int(const godot_string *p_self);

godot_string GDAPI godot_string_camelcase_to_underscore(const godot_string *p_self);
godot_string GDAPI godot_string_camelcase_to_underscore_lowercased(const godot_string *p_self);
godot_string GDAPI godot_string_capitalize(const godot_string *p_self);
double GDAPI godot_string_char_to_double(const char *p_what);
godot_int GDAPI godot_string_char_to_int(const char *p_what);
int64_t GDAPI godot_string_wchar_to_int(const wchar_t *p_str);
godot_int GDAPI godot_string_char_to_int_with_len(const char *p_what, godot_int p_len);
int64_t GDAPI godot_string_char_to_int64_with_len(const wchar_t *p_str, int p_len);
int64_t GDAPI godot_string_hex_to_int64(const godot_string *p_self);
int64_t GDAPI godot_string_hex_to_int64_with_prefix(const godot_string *p_self);
int64_t GDAPI godot_string_to_int64(const godot_string *p_self);
double GDAPI godot_string_unicode_char_to_double(const wchar_t *p_str, const wchar_t **r_end);

godot_int GDAPI godot_string_get_slice_count(const godot_string *p_self, godot_string p_splitter);
godot_string GDAPI godot_string_get_slice(const godot_string *p_self, godot_string p_splitter, godot_int p_slice);
godot_string GDAPI godot_string_get_slicec(const godot_string *p_self, wchar_t p_splitter, godot_int p_slice);

godot_array GDAPI godot_string_split(const godot_string *p_self, const godot_string *p_splitter);
godot_array GDAPI godot_string_split_allow_empty(const godot_string *p_self, const godot_string *p_splitter);
godot_array GDAPI godot_string_split_floats(const godot_string *p_self, const godot_string *p_splitter);
godot_array GDAPI godot_string_split_floats_allows_empty(const godot_string *p_self, const godot_string *p_splitter);
godot_array GDAPI godot_string_split_floats_mk(const godot_string *p_self, const godot_array *p_splitters);
godot_array GDAPI godot_string_split_floats_mk_allows_empty(const godot_string *p_self, const godot_array *p_splitters);
godot_array GDAPI godot_string_split_ints(const godot_string *p_self, const godot_string *p_splitter);
godot_array GDAPI godot_string_split_ints_allows_empty(const godot_string *p_self, const godot_string *p_splitter);
godot_array GDAPI godot_string_split_ints_mk(const godot_string *p_self, const godot_array *p_splitters);
godot_array GDAPI godot_string_split_ints_mk_allows_empty(const godot_string *p_self, const godot_array *p_splitters);
godot_array GDAPI godot_string_split_spaces(const godot_string *p_self);

wchar_t GDAPI godot_string_char_lowercase(wchar_t p_char);
wchar_t GDAPI godot_string_char_uppercase(wchar_t p_char);
godot_string GDAPI godot_string_to_lower(const godot_string *p_self);
godot_string GDAPI godot_string_to_upper(const godot_string *p_self);

godot_string GDAPI godot_string_get_basename(const godot_string *p_self);
godot_string GDAPI godot_string_get_extension(const godot_string *p_self);
godot_string GDAPI godot_string_left(const godot_string *p_self, godot_int p_pos);
wchar_t GDAPI godot_string_ord_at(const godot_string *p_self, godot_int p_idx);
godot_string GDAPI godot_string_plus_file(const godot_string *p_self, const godot_string *p_file);
godot_string GDAPI godot_string_right(const godot_string *p_self, godot_int p_pos);
godot_string GDAPI godot_string_strip_edges(const godot_string *p_self, godot_bool p_left, godot_bool p_right);
godot_string GDAPI godot_string_strip_escapes(const godot_string *p_self);

void GDAPI godot_string_erase(godot_string *p_self, godot_int p_pos, godot_int p_chars);

void GDAPI godot_string_ascii(godot_string *p_self, char *result);
void GDAPI godot_string_ascii_extended(godot_string *p_self, char *result);
void GDAPI godot_string_utf8(godot_string *p_self, char *result);
godot_bool GDAPI godot_string_parse_utf8(godot_string *p_self, const char *p_utf8);
godot_bool GDAPI godot_string_parse_utf8_with_len(godot_string *p_self, const char *p_utf8, godot_int p_len);
godot_string GDAPI godot_string_chars_to_utf8(const char *p_utf8);
godot_string GDAPI godot_string_chars_to_utf8_with_len(const char *p_utf8, godot_int p_len);

uint32_t GDAPI godot_string_hash(const godot_string *p_self);
uint64_t GDAPI godot_string_hash64(const godot_string *p_self);
uint32_t GDAPI godot_string_hash_chars(const char *p_cstr);
uint32_t GDAPI godot_string_hash_chars_with_len(const char *p_cstr, godot_int p_len);
uint32_t GDAPI godot_string_hash_utf8_chars(const wchar_t *p_str);
uint32_t GDAPI godot_string_hash_utf8_chars_with_len(const wchar_t *p_str, godot_int p_len);
godot_pool_byte_array GDAPI godot_string_md5_buffer(const godot_string *p_self);
godot_string GDAPI godot_string_md5_text(const godot_string *p_self);
godot_pool_byte_array GDAPI godot_string_sha256_buffer(const godot_string *p_self);
godot_string GDAPI godot_string_sha256_text(const godot_string *p_self);

godot_bool godot_string_empty(const godot_string *p_self);

// path functions
godot_string GDAPI godot_string_get_base_dir(const godot_string *p_self);
godot_string GDAPI godot_string_get_file(const godot_string *p_self);
godot_string GDAPI godot_string_humanize_size(size_t p_size);
godot_bool GDAPI godot_string_is_abs_path(const godot_string *p_self);
godot_bool GDAPI godot_string_is_rel_path(const godot_string *p_self);
godot_bool GDAPI godot_string_is_resource_file(const godot_string *p_self);
godot_string GDAPI godot_string_path_to(const godot_string *p_self, const godot_string *p_path);
godot_string GDAPI godot_string_path_to_file(const godot_string *p_self, const godot_string *p_path);
godot_string GDAPI godot_string_simplify_path(const godot_string *p_self);

godot_string GDAPI godot_string_c_escape(const godot_string *p_self);
godot_string GDAPI godot_string_c_escape_multiline(const godot_string *p_self);
godot_string GDAPI godot_string_c_unescape(const godot_string *p_self);
godot_string GDAPI godot_string_http_escape(const godot_string *p_self);
godot_string GDAPI godot_string_http_unescape(const godot_string *p_self);
godot_string GDAPI godot_string_json_escape(const godot_string *p_self);
godot_string GDAPI godot_string_word_wrap(const godot_string *p_self, godot_int p_chars_per_line);
godot_string GDAPI godot_string_xml_escape(const godot_string *p_self);
godot_string GDAPI godot_string_xml_escape_with_quotes(const godot_string *p_self);
godot_string GDAPI godot_string_xml_unescape(const godot_string *p_self);

godot_string GDAPI godot_string_percent_decode(const godot_string *p_self);
godot_string GDAPI godot_string_percent_encode(const godot_string *p_self);

godot_bool GDAPI godot_string_is_valid_float(const godot_string *p_self);
godot_bool GDAPI godot_string_is_valid_hex_number(const godot_string *p_self, godot_bool p_with_prefix);
godot_bool GDAPI godot_string_is_valid_html_color(const godot_string *p_self);
godot_bool GDAPI godot_string_is_valid_identifier(const godot_string *p_self);
godot_bool GDAPI godot_string_is_valid_integer(const godot_string *p_self);
godot_bool GDAPI godot_string_is_valid_ip_address(const godot_string *p_self);

void GDAPI godot_string_destroy(godot_string *p_self);

#ifdef __cplusplus
}
#endif

#endif // GODOT_STRING_H
