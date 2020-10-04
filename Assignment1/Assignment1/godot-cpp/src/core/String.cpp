#include "String.hpp"

#include "Array.hpp"
#include "GodotGlobal.hpp"
#include "NodePath.hpp"
#include "PoolArrays.hpp"
#include "Variant.hpp"

#include <gdnative/string.h>

#include <string.h>

namespace godot {

godot::CharString::~CharString() {
	godot::api->godot_char_string_destroy(&_char_string);
}

int godot::CharString::length() const {
	return godot::api->godot_char_string_length(&_char_string);
}

const char *godot::CharString::get_data() const {
	return godot::api->godot_char_string_get_data(&_char_string);
}

String String::num(double p_num, int p_decimals) {
	return String(godot::api->godot_string_num_with_decimals(p_num, p_decimals));
}

String String::num_scientific(double p_num) {
	return String(godot::api->godot_string_num_scientific(p_num));
}

String String::num_real(double p_num) {
	return String(godot::api->godot_string_num_real(p_num));
}

String String::num_int64(int64_t p_num, int base, bool capitalize_hex) {
	return String(godot::api->godot_string_num_int64_capitalized(p_num, base, capitalize_hex));
}

String String::chr(godot_char_type p_char) {
	return String(godot::api->godot_string_chr(p_char));
}

String String::md5(const uint8_t *p_md5) {
	return String(godot::api->godot_string_md5(p_md5));
}

String String::hex_encode_buffer(const uint8_t *p_buffer, int p_len) {
	return String(godot::api->godot_string_hex_encode_buffer(p_buffer, p_len));
}

godot::String::String() {
	godot::api->godot_string_new(&_godot_string);
}

String::String(const char *contents) {
	godot::api->godot_string_new(&_godot_string);
	godot::api->godot_string_parse_utf8(&_godot_string, contents);
}

String::String(const wchar_t *contents) {
	godot::api->godot_string_new_with_wide_string(&_godot_string, contents, wcslen(contents));
}

String::String(const wchar_t c) {
	godot::api->godot_string_new_with_wide_string(&_godot_string, &c, 1);
}

String::String(const String &other) {
	godot::api->godot_string_new_copy(&_godot_string, &other._godot_string);
}

String::~String() {
	godot::api->godot_string_destroy(&_godot_string);
}

wchar_t &String::operator[](const int idx) {
	return *const_cast<wchar_t *>(godot::api->godot_string_operator_index(&_godot_string, idx));
}

wchar_t String::operator[](const int idx) const {
	return *godot::api->godot_string_operator_index((godot_string *)&_godot_string, idx);
}

int String::length() const {
	return godot::api->godot_string_length(&_godot_string);
}

void String::operator=(const String &s) {
	godot::api->godot_string_destroy(&_godot_string);
	godot::api->godot_string_new_copy(&_godot_string, &s._godot_string);
}

bool String::operator==(const String &s) const {
	return godot::api->godot_string_operator_equal(&_godot_string, &s._godot_string);
}

bool String::operator!=(const String &s) const {
	return !(*this == s);
}

String String::operator+(const String &s) const {
	return String(godot::api->godot_string_operator_plus(&_godot_string, &s._godot_string));
}

void String::operator+=(const String &s) {
	*this = String(godot::api->godot_string_operator_plus(&_godot_string, &s._godot_string));
}

void String::operator+=(const wchar_t c) {
	String _to_be_added = String(c);
	*this = String(godot::api->godot_string_operator_plus(&_godot_string, &_to_be_added._godot_string));
}

bool String::operator<(const String &s) const {
	return godot::api->godot_string_operator_less(&_godot_string, &s._godot_string);
}

bool String::operator<=(const String &s) const {
	return godot::api->godot_string_operator_less(&_godot_string, &s._godot_string) ||
		   (*this == s);
}

bool String::operator>(const String &s) const {
	return !(*this <= s);
}

bool String::operator>=(const String &s) const {
	return !(*this < s);
}

String::operator NodePath() const {
	return NodePath(*this);
}

const wchar_t *String::unicode_str() const {
	return godot::api->godot_string_wide_str(&_godot_string);
}

char *String::alloc_c_string() const {

	godot_char_string contents = godot::api->godot_string_utf8(&_godot_string);

	int length = godot::api->godot_char_string_length(&contents);

	char *result = (char *)godot::api->godot_alloc(length + 1);

	if (result) {
		memcpy(result, godot::api->godot_char_string_get_data(&contents), length + 1);
	}

	godot::api->godot_char_string_destroy(&contents);

	return result;
}

CharString String::utf8() const {
	CharString ret;

	ret._char_string = godot::api->godot_string_utf8(&_godot_string);

	return ret;
}

CharString String::ascii(bool p_extended) const {

	CharString ret;

	if (p_extended)
		ret._char_string = godot::api->godot_string_ascii_extended(&_godot_string);
	else
		ret._char_string = godot::api->godot_string_ascii(&_godot_string);

	return ret;
}

String operator+(const char *a, const String &b) {
	return String(a) + b;
}

String operator+(const wchar_t *a, const String &b) {
	return String(a) + b;
}

bool String::begins_with(String &p_string) const {
	return godot::api->godot_string_begins_with(&_godot_string, &p_string._godot_string);
}

bool String::begins_with_char_array(const char *p_char_array) const {
	return godot::api->godot_string_begins_with_char_array(&_godot_string, p_char_array);
}

PoolStringArray String::bigrams() const {
	godot_array arr = godot::api->godot_string_bigrams(&_godot_string);

	return *(Array *)&arr;
}

String String::c_escape() const {
	return String(godot::api->godot_string_c_escape(&_godot_string));
}

String String::c_unescape() const {
	return String(godot::api->godot_string_c_unescape(&_godot_string));
}

String String::capitalize() const {
	return String(godot::api->godot_string_capitalize(&_godot_string));
}

bool String::empty() const {
	return godot::api->godot_string_empty(&_godot_string);
}

bool String::ends_with(String &p_string) const {
	return godot::api->godot_string_ends_with(&_godot_string, &p_string._godot_string);
}

void String::erase(int position, int chars) {
	godot::api->godot_string_erase(&_godot_string, position, chars);
}

int String::find(String p_what, int p_from) const {
	return godot::api->godot_string_find_from(&_godot_string, p_what._godot_string, p_from);
}

int String::find_last(String p_what) const {
	return godot::api->godot_string_find_last(&_godot_string, p_what._godot_string);
}

int String::findn(String p_what, int p_from) const {
	return godot::api->godot_string_findn_from(&_godot_string, p_what._godot_string, p_from);
}

String String::format(Variant values) const {
	return String(godot::api->godot_string_format(&_godot_string, (godot_variant *)&values));
}

String String::format(Variant values, String placeholder) const {
	godot_char_string contents = godot::api->godot_string_utf8(&placeholder._godot_string);
	String new_string(godot::api->godot_string_format_with_custom_placeholder(&_godot_string, (godot_variant *)&values, godot::api->godot_char_string_get_data(&contents)));
	godot::api->godot_char_string_destroy(&contents);

	return new_string;
}

String String::get_base_dir() const {
	return String(godot::api->godot_string_get_base_dir(&_godot_string));
}

String String::get_basename() const {
	return String(godot::api->godot_string_get_basename(&_godot_string));
}

String String::get_extension() const {
	return String(godot::api->godot_string_get_extension(&_godot_string));
}

String String::get_file() const {
	return String(godot::api->godot_string_get_file(&_godot_string));
}

int String::hash() const {
	return godot::api->godot_string_hash(&_godot_string);
}

int String::hex_to_int() const {
	return godot::api->godot_string_hex_to_int(&_godot_string);
}

String String::insert(int position, String what) const {
	return String(godot::api->godot_string_insert(&_godot_string, position, what._godot_string));
}

bool String::is_abs_path() const {
	return godot::api->godot_string_is_abs_path(&_godot_string);
}

bool String::is_rel_path() const {
	return godot::api->godot_string_is_rel_path(&_godot_string);
}

bool String::is_subsequence_of(String text) const {
	return godot::api->godot_string_is_subsequence_of(&_godot_string, &text._godot_string);
}

bool String::is_subsequence_ofi(String text) const {
	return godot::api->godot_string_is_subsequence_ofi(&_godot_string, &text._godot_string);
}

bool String::is_valid_float() const {
	return godot::api->godot_string_is_valid_float(&_godot_string);
}

bool String::is_valid_html_color() const {
	return godot::api->godot_string_is_valid_html_color(&_godot_string);
}

bool String::is_valid_identifier() const {
	return godot::api->godot_string_is_valid_identifier(&_godot_string);
}

bool String::is_valid_integer() const {
	return godot::api->godot_string_is_numeric(&_godot_string);
}

bool String::is_valid_ip_address() const {
	return godot::api->godot_string_is_valid_ip_address(&_godot_string);
}

String String::json_escape() const {
	return String(godot::api->godot_string_json_escape(&_godot_string));
}

String String::left(int position) const {
	return String(godot::api->godot_string_left(&_godot_string, position));
}

bool String::match(String expr) const {
	return godot::api->godot_string_match(&_godot_string, &expr._godot_string);
}

bool String::matchn(String expr) const {
	return godot::api->godot_string_match(&_godot_string, &expr._godot_string);
}

PoolByteArray String::md5_buffer() const {
	godot_pool_byte_array arr = godot::api->godot_string_md5_buffer(&_godot_string);
	return *(PoolByteArray *)&arr;
}

String String::md5_text() const {
	return String(godot::api->godot_string_md5_text(&_godot_string));
}

int String::ord_at(int at) const {
	return godot::api->godot_string_ord_at(&_godot_string, at);
}

String String::pad_decimals(int digits) const {
	return String(godot::api->godot_string_pad_decimals(&_godot_string, digits));
}

String String::pad_zeros(int digits) const {
	return String(godot::api->godot_string_pad_zeros(&_godot_string, digits));
}

String String::percent_decode() const {
	return String(godot::api->godot_string_percent_decode(&_godot_string));
}

String String::percent_encode() const {
	return String(godot::api->godot_string_percent_encode(&_godot_string));
}

String String::plus_file(String file) const {
	return String(godot::api->godot_string_plus_file(&_godot_string, &file._godot_string));
}

String String::replace(String p_key, String p_with) const {
	return String(godot::api->godot_string_replace(&_godot_string, p_key._godot_string, p_with._godot_string));
}

String String::replacen(String what, String forwhat) const {
	return String(godot::api->godot_string_replacen(&_godot_string, what._godot_string, forwhat._godot_string));
}

int String::rfind(String p_what, int p_from) const {
	return godot::api->godot_string_rfind_from(&_godot_string, p_what._godot_string, p_from);
}

int String::rfindn(String p_what, int p_from) const {
	return godot::api->godot_string_rfindn_from(&_godot_string, p_what._godot_string, p_from);
}

String String::right(int position) const {
	return String(godot::api->godot_string_right(&_godot_string, position));
}

PoolByteArray String::sha256_buffer() const {
	godot_pool_byte_array arr = godot::api->godot_string_sha256_buffer(&_godot_string);

	return *(PoolByteArray *)&arr;
}

String String::sha256_text() const {
	return String(godot::api->godot_string_sha256_text(&_godot_string));
}

float String::similarity(String text) const {
	return godot::api->godot_string_similarity(&_godot_string, &text._godot_string);
}

PoolStringArray String::split(String divisor, bool allow_empty) const {
	godot_array arr = godot::api->godot_string_split(&_godot_string, &divisor._godot_string);

	return *(Array *)&arr;
}

PoolIntArray String::split_ints(String divisor, bool allow_empty) const {
	godot_array arr = godot::api->godot_string_split_floats(&_godot_string, &divisor._godot_string);

	return *(Array *)&arr;
}

PoolRealArray String::split_floats(String divisor, bool allow_empty) const {
	godot_array arr = godot::api->godot_string_split_floats(&_godot_string, &divisor._godot_string);

	return *(Array *)&arr;
}

String String::strip_edges(bool left, bool right) const {
	return String(godot::api->godot_string_strip_edges(&_godot_string, left, right));
}

String String::substr(int from, int len) const {
	return String(godot::api->godot_string_substr(&_godot_string, from, len));
}

float String::to_float() const {
	return godot::api->godot_string_to_float(&_godot_string);
}

int64_t String::to_int() const {
	return godot::api->godot_string_to_int(&_godot_string);
}

String String::to_lower() const {
	return String(godot::api->godot_string_to_lower(&_godot_string));
}

String String::to_upper() const {
	return String(godot::api->godot_string_to_upper(&_godot_string));
}

String String::xml_escape() const {
	return String(godot::api->godot_string_xml_escape(&_godot_string));
}

String String::xml_unescape() const {
	return String(godot::api->godot_string_xml_unescape(&_godot_string));
}

signed char String::casecmp_to(String p_str) const {
	return godot::api->godot_string_casecmp_to(&_godot_string, &p_str._godot_string);
}

signed char String::nocasecmp_to(String p_str) const {
	return godot::api->godot_string_nocasecmp_to(&_godot_string, &p_str._godot_string);
}

signed char String::naturalnocasecmp_to(String p_str) const {
	return godot::api->godot_string_naturalnocasecmp_to(&_godot_string, &p_str._godot_string);
}

String String::dedent() const {
	String new_string;
	new_string._godot_string = godot::core_1_1_api->godot_string_dedent(&_godot_string);
	return new_string;
}

PoolStringArray String::rsplit(const String &divisor, const bool allow_empty,
		const int maxsplit) const {
	godot_pool_string_array arr = godot::core_1_1_api->godot_string_rsplit(&_godot_string, &divisor._godot_string, allow_empty, maxsplit);
	return *(PoolStringArray *)&arr;
}

String String::rstrip(const String &chars) const {
	String new_string;
	new_string._godot_string = godot::core_1_1_api->godot_string_rstrip(&_godot_string, &chars._godot_string);
	return new_string;
}

String String::trim_prefix(const String &prefix) const {
	String new_string;
	new_string._godot_string = godot::core_1_1_api->godot_string_trim_prefix(&_godot_string, &prefix._godot_string);
	return new_string;
}

String String::trim_suffix(const String &suffix) const {
	String new_string;
	new_string._godot_string = godot::core_1_1_api->godot_string_trim_suffix(&_godot_string, &suffix._godot_string);
	return new_string;
}

} // namespace godot
