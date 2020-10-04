#ifndef STRING_H
#define STRING_H

#include <gdnative/string.h>

namespace godot {

class NodePath;
class Variant;
class PoolByteArray;
class PoolIntArray;
class PoolRealArray;
class PoolStringArray;
class String;

class CharString {

	friend class String;

	godot_char_string _char_string;

public:
	~CharString();

	int length() const;
	const char *get_data() const;
};

class String {
	godot_string _godot_string;

	String(godot_string contents) :
			_godot_string(contents) {}

public:
	String();
	String(const char *contents);
	String(const wchar_t *contents);
	String(const wchar_t c);
	String(const String &other);

	~String();

	static String num(double p_num, int p_decimals = -1);
	static String num_scientific(double p_num);
	static String num_real(double p_num);
	static String num_int64(int64_t p_num, int base = 10, bool capitalize_hex = false);
	static String chr(godot_char_type p_char);
	static String md5(const uint8_t *p_md5);
	static String hex_encode_buffer(const uint8_t *p_buffer, int p_len);

	wchar_t &operator[](const int idx);
	wchar_t operator[](const int idx) const;

	void operator=(const String &s);
	bool operator==(const String &s) const;
	bool operator!=(const String &s) const;
	String operator+(const String &s) const;
	void operator+=(const String &s);
	void operator+=(const wchar_t c);
	bool operator<(const String &s) const;
	bool operator<=(const String &s) const;
	bool operator>(const String &s) const;
	bool operator>=(const String &s) const;

	operator NodePath() const;

	int length() const;
	const wchar_t *unicode_str() const;
	char *alloc_c_string() const;
	CharString utf8() const;
	CharString ascii(bool p_extended = false) const;

	bool begins_with(String &s) const;
	bool begins_with_char_array(const char *p_char_array) const;
	PoolStringArray bigrams() const;
	String c_escape() const;
	String c_unescape() const;
	String capitalize() const;
	bool empty() const;
	bool ends_with(String &text) const;
	void erase(int position, int chars);
	int find(String what, int from = 0) const;
	int find_last(String what) const;
	int findn(String what, int from = 0) const;
	String format(Variant values) const;
	String format(Variant values, String placeholder) const;
	String get_base_dir() const;
	String get_basename() const;
	String get_extension() const;
	String get_file() const;
	int hash() const;
	int hex_to_int() const;
	String insert(int position, String what) const;
	bool is_abs_path() const;
	bool is_rel_path() const;
	bool is_subsequence_of(String text) const;
	bool is_subsequence_ofi(String text) const;
	bool is_valid_float() const;
	bool is_valid_html_color() const;
	bool is_valid_identifier() const;
	bool is_valid_integer() const;
	bool is_valid_ip_address() const;
	String json_escape() const;
	String left(int position) const;
	bool match(String expr) const;
	bool matchn(String expr) const;
	PoolByteArray md5_buffer() const;
	String md5_text() const;
	int ord_at(int at) const;
	String pad_decimals(int digits) const;
	String pad_zeros(int digits) const;
	String percent_decode() const;
	String percent_encode() const;
	String plus_file(String file) const;
	String replace(String what, String forwhat) const;
	String replacen(String what, String forwhat) const;
	int rfind(String what, int from = -1) const;
	int rfindn(String what, int from = -1) const;
	String right(int position) const;
	PoolByteArray sha256_buffer() const;
	String sha256_text() const;
	float similarity(String text) const;
	PoolStringArray split(String divisor, bool allow_empty = true) const;
	PoolIntArray split_ints(String divisor, bool allow_empty = true) const;
	PoolRealArray split_floats(String divisor, bool allow_empty = true) const;
	String strip_edges(bool left = true, bool right = true) const;
	String substr(int from, int len) const;
	float to_float() const;
	int64_t to_int() const;
	String to_lower() const;
	String to_upper() const;
	String xml_escape() const;
	String xml_unescape() const;
	signed char casecmp_to(String p_str) const;
	signed char nocasecmp_to(String p_str) const;
	signed char naturalnocasecmp_to(String p_str) const;
	String dedent() const;
	PoolStringArray rsplit(const String &divisor, const bool allow_empty = true, const int maxsplit = 0) const;
	String rstrip(const String &chars) const;
	String trim_prefix(const String &prefix) const;
	String trim_suffix(const String &suffix) const;
};

String operator+(const char *a, const String &b);
String operator+(const wchar_t *a, const String &b);

} // namespace godot

#endif // STRING_H
