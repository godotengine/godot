#pragma once

#include "variant.hpp"
#include <string>

/**
 * @brief String wrapper for Godot String.
 * Implemented by referencing and mutating a host-side String Variant.
 */
union String {
	constexpr String() {} // DON'T TOUCH
	String(std::string_view value);
	template <size_t N>
	String(const char (&value)[N]);
	String(const std::string &value);

	String &operator =(std::string_view value);
	String &operator =(const String &value);

	// String operations
	void append(const String &value);
	void append(std::string_view value);
	void erase(int idx, int count = 1);
	void insert(int idx, const String &value);
	int find(const String &value) const;
	bool contains(std::string_view value) const { return find(value) != -1; }
	bool empty() const { return size() == 0; }

	// Call methods on the String
	template <typename... Args>
	Variant operator () (std::string_view method, Args&&... args);

	String &operator +=(const String &value) { append(value); return *this; }
	String &operator +=(std::string_view value) { append(value); return *this; }

	// String access
	String operator[](int idx) const;
	String at(int idx) const { return (*this)[idx]; }

	operator std::string() const { return utf8(); }
	operator std::u32string() const { return utf32(); }
	std::string utf8() const;
	std::u32string utf32() const;

	bool operator ==(const String &other) const;
	bool operator ==(const char *other) const;

	// String size
	int size() const;
	bool is_empty() const { return size() == 0; }

	METHOD(bool, begins_with);
	METHOD(PackedArray<std::string>, bigrams);
	METHOD(int64_t, bin_to_int);
	METHOD(String, c_escape);
	METHOD(String, c_unescape);
	METHOD(String, capitalize);
	METHOD(int64_t, casecmp_to);
	METHOD(String, chr);
	METHOD(bool, containsn);
	METHOD(int64_t, count);
	METHOD(int64_t, countn);
	METHOD(String, dedent);
	METHOD(bool, ends_with);
	METHOD(int64_t, filecasecmp_to);
	METHOD(int64_t, filenocasecmp_to);
	METHOD(int64_t, findn);
	METHOD(String, format);
	METHOD(String, get_base_dir);
	METHOD(String, get_basename);
	METHOD(String, get_extension);
	METHOD(String, get_file);
	METHOD(String, get_slice);
	METHOD(String, get_slice_count);
	METHOD(String, get_slicec);
	METHOD(int64_t, hash);
	METHOD(PackedArray<uint8_t>, hex_decode);
	METHOD(int64_t, hex_to_int);
	METHOD(String, humanize_size);
	METHOD(String, indent);
	METHOD(bool, is_absolute_path);
	METHOD(bool, is_relative_path);
	METHOD(bool, is_subsequence_of);
	METHOD(bool, is_subsequence_ofn);
	METHOD(bool, is_valid_filename);
	METHOD(bool, is_valid_float);
	METHOD(bool, is_valid_hex_number);
	METHOD(bool, is_valid_html_color);
	METHOD(bool, is_valid_identifier);
	METHOD(bool, is_valid_int);
	METHOD(bool, is_valid_ip_address);
	METHOD(String, join);
	METHOD(String, json_escape);
	METHOD(String, left);
	METHOD(int64_t, length);
	METHOD(String, lpad);
	METHOD(String, lstrip);
	METHOD(bool, match);
	METHOD(bool, matchn);
	METHOD(PackedArray<uint8_t>, md5_buffer);
	METHOD(String, md5_text);
	METHOD(int64_t, naturalcasecmp_to);
	METHOD(int64_t, naturalnocasecmp_to);
	METHOD(int64_t, nocasecmp_to);
	METHOD(String, num);
	METHOD(String, num_int64);
	METHOD(String, num_scientific);
	METHOD(String, num_uint64);
	METHOD(String, pad_decimals);
	METHOD(String, pad_zeros);
	METHOD(String, path_join);
	METHOD(String, repeat);
	METHOD(String, replace);
	METHOD(String, replacen);
	METHOD(String, reverse);
	METHOD(int64_t, rfind);
	METHOD(int64_t, rfindn);
	METHOD(String, right);
	METHOD(String, rpad);
	METHOD(PackedArray<std::string>, rsplit);
	METHOD(String, rstrip);
	METHOD(PackedArray<uint8_t>, sha1_buffer);
	METHOD(String, sha1_text);
	METHOD(PackedArray<uint8_t>, sha256_buffer);
	METHOD(String, sha256_text);
	METHOD(double, similarity);
	METHOD(String, simplify_path);
	METHOD(PackedArray<uint8_t>, split);
	METHOD(PackedArray<double>, split_floats);
	METHOD(String, strip_edges);
	METHOD(String, strip_escapes);
	METHOD(String, substr);
	METHOD(PackedArray<uint8_t>, to_ascii_buffer);
	METHOD(String, to_camel_case);
	METHOD(double, to_float);
	METHOD(int64_t, to_int);
	METHOD(String, to_lower);
	METHOD(String, to_pascal_case);
	METHOD(String, to_snake_case);
	METHOD(String, to_upper);
	METHOD(PackedArray<uint8_t>, to_utf8_buffer);
	METHOD(PackedArray<uint8_t>, to_utf16_buffer);
	METHOD(PackedArray<uint8_t>, to_utf32_buffer);
	METHOD(PackedArray<uint8_t>, to_wchar_buffer);
	METHOD(String, trim_prefix);
	METHOD(String, trim_suffix);
	METHOD(int64_t, unicode_at);
	METHOD(String, uri_decode);
	METHOD(String, uri_encode);
	METHOD(String, validate_filename);
	METHOD(String, validate_node_name);
	METHOD(String, xml_escape);
	METHOD(String, xml_unescape);

	static String from_variant_index(unsigned idx) { String a {}; a.m_idx = idx; return a; }
	unsigned get_variant_index() const noexcept { return m_idx; }
	bool is_permanent() const { return Variant::is_permanent_index(m_idx); }
	static unsigned Create(const char *data, size_t size);

private:
	unsigned m_idx = INT32_MIN;
};
using NodePath = String; // NodePath is compatible with String

inline Variant::Variant(const String &s) {
	m_type = Variant::STRING;
	v.i = s.get_variant_index();
}

inline Variant::operator String() const {
	return as_string();
}

inline String Variant::as_string() const {
	if (m_type != STRING && m_type != STRING_NAME && m_type != NODE_PATH) {
		api_throw("std::bad_cast", "Failed to cast Variant to String", this);
	}
	return String::from_variant_index(v.i);
}

inline String::String(std::string_view value)
	: m_idx(Create(value.data(), value.size())) {}
inline String &String::operator =(std::string_view value) {
	m_idx = Create(value.data(), value.size());
	return *this;
}
template <size_t N>
inline String::String(const char (&value)[N])
	: m_idx(Create(value, N - 1)) {}

inline String::String(const std::string &value)
	: m_idx(Create(value.data(), value.size())) {}

template <typename... Args>
inline Variant String::operator () (std::string_view method, Args&&... args) {
	return Variant(*this).method_call(method, std::forward<Args>(args)...);
}
