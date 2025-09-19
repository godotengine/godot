/**************************************************************************/
/*  ustring.h                                                             */
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

#pragma once

// Note: _GODOT suffix added to header guard to avoid conflict with ICU header.

#include <optional>

#include "core/string/char_utils.h" // IWYU pragma: export
#include "core/templates/cowdata.h"
#include "core/templates/vector.h"
#include "core/typedefs.h"
#include "core/variant/array.h"

class String;
struct FormatSpec;
template <typename T>
class CharStringT;

/*************************************************************************/
/*  Utility Functions                                                    */
/*************************************************************************/

// Not defined by std.
// strlen equivalent function for char16_t * arguments.
constexpr size_t strlen(const char16_t *p_str) {
	const char16_t *ptr = p_str;
	while (*ptr != 0) {
		++ptr;
	}
	return ptr - p_str;
}

// strlen equivalent function for char32_t * arguments.
constexpr size_t strlen(const char32_t *p_str) {
	const char32_t *ptr = p_str;
	while (*ptr != 0) {
		++ptr;
	}
	return ptr - p_str;
}

// strlen equivalent function for wchar_t * arguments; depends on the platform.
constexpr size_t strlen(const wchar_t *p_str) {
	// Use static_cast twice because reinterpret_cast is not allowed in constexpr
#ifdef WINDOWS_ENABLED
	// wchar_t is 16-bit
	return strlen(static_cast<const char16_t *>(static_cast<const void *>(p_str)));
#else
	// wchar_t is 32-bit
	return strlen(static_cast<const char32_t *>(static_cast<const void *>(p_str)));
#endif
}

// strnlen equivalent function for char16_t * arguments.
constexpr size_t strnlen(const char16_t *p_str, size_t p_clip_to_len) {
	size_t len = 0;
	while (len < p_clip_to_len && *(p_str++) != 0) {
		len++;
	}
	return len;
}

// strnlen equivalent function for char32_t * arguments.
constexpr size_t strnlen(const char32_t *p_str, size_t p_clip_to_len) {
	size_t len = 0;
	while (len < p_clip_to_len && *(p_str++) != 0) {
		len++;
	}
	return len;
}

// strnlen equivalent function for wchar_t * arguments; depends on the platform.
constexpr size_t strnlen(const wchar_t *p_str, size_t p_clip_to_len) {
	// Use static_cast twice because reinterpret_cast is not allowed in constexpr
#ifdef WINDOWS_ENABLED
	// wchar_t is 16-bit
	return strnlen(static_cast<const char16_t *>(static_cast<const void *>(p_str)), p_clip_to_len);
#else
	// wchar_t is 32-bit
	return strnlen(static_cast<const char32_t *>(static_cast<const void *>(p_str)), p_clip_to_len);
#endif
}

template <typename L, typename R>
constexpr int64_t str_compare(const L *l_ptr, const R *r_ptr) {
	while (true) {
		const char32_t l = *l_ptr;
		const char32_t r = *r_ptr;

		if (l == 0 || l != r) {
			return static_cast<int64_t>(l) - static_cast<int64_t>(r);
		}

		l_ptr++;
		r_ptr++;
	}
}

/*************************************************************************/
/*  CharProxy                                                            */
/*************************************************************************/

template <typename T>
class [[nodiscard]] CharProxy {
	friend String;
	friend CharStringT<T>;

	const int _index;
	CowData<T> &_cowdata;
	static constexpr T _null = 0;

	_FORCE_INLINE_ CharProxy(const int &p_index, CowData<T> &p_cowdata) :
			_index(p_index),
			_cowdata(p_cowdata) {}

public:
	_FORCE_INLINE_ CharProxy(const CharProxy<T> &p_other) :
			_index(p_other._index),
			_cowdata(p_other._cowdata) {}

	_FORCE_INLINE_ operator T() const {
		if (unlikely(_index == _cowdata.size())) {
			return _null;
		}

		return _cowdata.get(_index);
	}

	_FORCE_INLINE_ const T *operator&() const {
		return _cowdata.ptr() + _index;
	}

	_FORCE_INLINE_ void operator=(const T &p_other) const {
		_cowdata.set(_index, p_other);
	}

	_FORCE_INLINE_ void operator=(const CharProxy<T> &p_other) const {
		_cowdata.set(_index, p_other.operator T());
	}
};

/*************************************************************************/
/*  CharStringT                                                          */
/*************************************************************************/

template <typename T>
class [[nodiscard]] CharStringT {
	CowData<T> _cowdata;
	static constexpr T _null = 0;

public:
	_FORCE_INLINE_ T *ptrw() { return _cowdata.ptrw(); }
	_FORCE_INLINE_ const T *ptr() const { return _cowdata.ptr(); }
	_FORCE_INLINE_ const T *get_data() const { return ptr() ? ptr() : &_null; }

	_FORCE_INLINE_ int size() const { return _cowdata.size(); }
	_FORCE_INLINE_ int length() const { return ptr() ? size() - 1 : 0; }
	_FORCE_INLINE_ bool is_empty() const { return length() == 0; }

	_FORCE_INLINE_ operator Span<T>() const { return Span(ptr(), length()); }
	_FORCE_INLINE_ Span<T> span() const { return Span(ptr(), length()); }

	/// Resizes the string. The given size must include the null terminator.
	/// New characters are not initialized, and should be set by the caller.
	_FORCE_INLINE_ Error resize_uninitialized(int64_t p_size) { return _cowdata.template resize<false>(p_size); }

	_FORCE_INLINE_ T get(int p_index) const { return _cowdata.get(p_index); }
	_FORCE_INLINE_ void set(int p_index, const T &p_elem) { _cowdata.set(p_index, p_elem); }
	_FORCE_INLINE_ const T &operator[](int p_index) const {
		if (unlikely(p_index == _cowdata.size())) {
			return _null;
		}
		return _cowdata.get(p_index);
	}
	_FORCE_INLINE_ CharProxy<T> operator[](int p_index) { return CharProxy<T>(p_index, _cowdata); }

	_FORCE_INLINE_ CharStringT() = default;
	_FORCE_INLINE_ CharStringT(const CharStringT &p_str) = default;
	_FORCE_INLINE_ CharStringT(CharStringT &&p_str) = default;
	_FORCE_INLINE_ void operator=(const CharStringT &p_str) { _cowdata = p_str._cowdata; }
	_FORCE_INLINE_ void operator=(CharStringT &&p_str) { _cowdata = std::move(p_str._cowdata); }
	_FORCE_INLINE_ CharStringT(const T *p_cstr) { copy_from(p_cstr); }
	_FORCE_INLINE_ void operator=(const T *p_cstr) { copy_from(p_cstr); }

	_FORCE_INLINE_ bool operator==(const CharStringT<T> &p_other) const {
		if (length() != p_other.length()) {
			return false;
		}
		return memcmp(ptr(), p_other.ptr(), length() * sizeof(T)) == 0;
	}
	_FORCE_INLINE_ bool operator!=(const CharStringT<T> &p_other) const { return !(*this == p_other); }
	_FORCE_INLINE_ bool operator<(const CharStringT<T> &p_other) const {
		if (length() == 0) {
			return p_other.length() != 0;
		}
		return str_compare(get_data(), p_other.get_data()) < 0;
	}
	_FORCE_INLINE_ CharStringT<T> &operator+=(T p_char) {
		const int lhs_len = length();
		resize_uninitialized(lhs_len + 2);

		T *dst = ptrw();
		dst[lhs_len] = p_char;
		dst[lhs_len + 1] = _null;

		return *this;
	}

protected:
	void copy_from(const T *p_cstr) {
		if (!p_cstr) {
			resize_uninitialized(0);
			return;
		}

		size_t len = strlen(p_cstr);
		if (len == 0) {
			resize_uninitialized(0);
			return;
		}

		Error err = resize_uninitialized(++len); // include terminating null char.

		ERR_FAIL_COND_MSG(err != OK, "Failed to copy C-string.");

		memcpy(ptrw(), p_cstr, len * sizeof(T));
	}
};

template <typename T>
struct is_zero_constructible<CharStringT<T>> : std::true_type {};

using CharString = CharStringT<char>;
using Char16String = CharStringT<char16_t>;

/*************************************************************************/
/*  String                                                               */
/*************************************************************************/

class [[nodiscard]] String {
	CowData<char32_t> _cowdata;
	static constexpr char32_t _null = 0;
	static constexpr char32_t _replacement_char = 0xfffd;

	// Known-length copy.
	void copy_from_unchecked(const char32_t *p_char, int p_length);

	// NULL-terminated c string copy - automatically parse the string to find the length.
	void append_latin1(const char *p_cstr) {
		append_latin1(Span(p_cstr, p_cstr ? strlen(p_cstr) : 0));
	}
	void append_utf32(const char32_t *p_cstr) {
		append_utf32(Span(p_cstr, p_cstr ? strlen(p_cstr) : 0));
	}

	// wchar_t copy_from depends on the platform.
	void append_wstring(const Span<wchar_t> &p_cstr) {
#ifdef WINDOWS_ENABLED
		// wchar_t is 16-bit, parse as UTF-16
		append_utf16((const char16_t *)p_cstr.ptr(), p_cstr.size());
#else
		// wchar_t is 32-bit, copy directly
		append_utf32((Span<char32_t> &)p_cstr);
#endif
	}
	void append_wstring(const wchar_t *p_cstr) {
#ifdef WINDOWS_ENABLED
		// wchar_t is 16-bit, parse as UTF-16
		append_utf16((const char16_t *)p_cstr);
#else
		// wchar_t is 32-bit, copy directly
		append_utf32((const char32_t *)p_cstr);
#endif
	}

	bool _base_is_subsequence_of(const String &p_string, bool case_insensitive) const;
	int _count(const String &p_string, int p_from, int p_to, bool p_case_insensitive) const;
	int _count(const char *p_string, int p_from, int p_to, bool p_case_insensitive) const;
	String _separate_compound_words() const;
	Error _append_formatted_without_padding(const FormatSpec &p_format_spec, const Variant &p_value);

public:
	enum {
		npos = -1 ///<for "some" compatibility with std::string (npos is a huge value in std::string)
	};

	_FORCE_INLINE_ char32_t *ptrw() { return _cowdata.ptrw(); }
	_FORCE_INLINE_ const char32_t *ptr() const { return _cowdata.ptr(); }
	_FORCE_INLINE_ const char32_t *get_data() const { return ptr() ? ptr() : &_null; }

	_FORCE_INLINE_ int size() const { return _cowdata.size(); }
	_FORCE_INLINE_ int length() const { return ptr() ? size() - 1 : 0; }
	_FORCE_INLINE_ bool is_empty() const { return length() == 0; }

	_FORCE_INLINE_ operator Span<char32_t>() const { return Span(ptr(), length()); }
	_FORCE_INLINE_ Span<char32_t> span() const { return Span(ptr(), length()); }

	void remove_at(int p_index) { _cowdata.remove_at(p_index); }

	_FORCE_INLINE_ void clear() { resize_uninitialized(0); }

	_FORCE_INLINE_ char32_t get(int p_index) const { return _cowdata.get(p_index); }
	_FORCE_INLINE_ void set(int p_index, const char32_t &p_elem) { _cowdata.set(p_index, p_elem); }

	/// Resizes the string. The given size must include the null terminator.
	/// New characters are not initialized, and should be set by the caller.
	Error resize_uninitialized(int64_t p_size) { return _cowdata.resize<false>(p_size); }

	_FORCE_INLINE_ const char32_t &operator[](int p_index) const {
		if (unlikely(p_index == _cowdata.size())) {
			return _null;
		}

		return _cowdata.get(p_index);
	}
	_FORCE_INLINE_ CharProxy<char32_t> operator[](int p_index) { return CharProxy<char32_t>(p_index, _cowdata); }

	/* Compatibility Operators */

	bool operator==(const String &p_str) const;
	bool operator!=(const String &p_str) const;
	String operator+(const String &p_str) const;
	String operator+(const char *p_char) const;
	String operator+(const wchar_t *p_char) const;
	String operator+(const char32_t *p_char) const;
	String operator+(char32_t p_char) const;

	String &operator+=(const String &);
	String &operator+=(char32_t p_char);
	String &operator+=(const char *p_str);
	String &operator+=(const wchar_t *p_str);
	String &operator+=(const char32_t *p_str);

	bool operator==(const char *p_str) const;
	bool operator==(const wchar_t *p_str) const;
	bool operator==(const char32_t *p_str) const;
	bool operator==(const Span<char32_t> &p_str_range) const;

	bool operator!=(const char *p_str) const;
	bool operator!=(const wchar_t *p_str) const;
	bool operator!=(const char32_t *p_str) const;

	bool operator<(const char32_t *p_str) const;
	bool operator<(const char *p_str) const;
	bool operator<(const wchar_t *p_str) const;

	bool operator<(const String &p_str) const;
	bool operator<=(const String &p_str) const;
	bool operator>(const String &p_str) const;
	bool operator>=(const String &p_str) const;

	signed char casecmp_to(const String &p_str) const;
	signed char nocasecmp_to(const String &p_str) const;
	signed char naturalcasecmp_to(const String &p_str) const;
	signed char naturalnocasecmp_to(const String &p_str) const;
	// Special sorting for file names. Names starting with `_` are put before all others except those starting with `.`, otherwise natural comparison is used.
	signed char filecasecmp_to(const String &p_str) const;
	signed char filenocasecmp_to(const String &p_str) const;

	bool is_valid_string() const;

	/* debug, error messages */
	void print_unicode_error(const String &p_message, bool p_critical = false) const;

	/* complex helpers */
	String substr(int p_from, int p_chars = -1) const;
	int find(const String &p_str, int p_from = 0) const; ///< return <0 if failed
	int find(const char *p_str, int p_from = 0) const; ///< return <0 if failed
	int find_char(char32_t p_char, int p_from = 0) const; ///< return <0 if failed
	int findn(const String &p_str, int p_from = 0) const; ///< return <0 if failed, case insensitive
	int findn(const char *p_str, int p_from = 0) const; ///< return <0 if failed
	int rfind(const String &p_str, int p_from = -1) const; ///< return <0 if failed
	int rfind(const char *p_str, int p_from = -1) const; ///< return <0 if failed
	int rfind_char(char32_t p_char, int p_from = -1) const; ///< return <0 if failed
	int rfindn(const String &p_str, int p_from = -1) const; ///< return <0 if failed, case insensitive
	int rfindn(const char *p_str, int p_from = -1) const; ///< return <0 if failed
	int findmk(const Vector<String> &p_keys, int p_from = 0, int *r_key = nullptr) const; ///< return <0 if failed
	bool match(const String &p_wildcard) const;
	bool matchn(const String &p_wildcard) const;
	bool begins_with(const String &p_string) const;
	bool begins_with(const char *p_string) const;
	bool ends_with(const String &p_string) const;
	bool ends_with(const char *p_string) const;
	bool is_enclosed_in(const String &p_string) const;
	bool is_subsequence_of(const String &p_string) const;
	bool is_subsequence_ofn(const String &p_string) const;
	bool is_quoted() const;
	bool is_lowercase() const;
	Vector<String> bigrams() const;
	float similarity(const String &p_string) const;
	String format(const Variant &values, const String &placeholder = "{_}") const;
	String replace_first(const String &p_key, const String &p_with) const;
	String replace_first(const char *p_key, const char *p_with) const;
	String replace(const String &p_key, const String &p_with) const;
	String replace(const char *p_key, const char *p_with) const;
	String replace_char(char32_t p_key, char32_t p_with) const;
	String replace_chars(const String &p_keys, char32_t p_with) const;
	String replace_chars(const char *p_keys, char32_t p_with) const;
	String replacen(const String &p_key, const String &p_with) const;
	String replacen(const char *p_key, const char *p_with) const;
	String repeat(int p_count) const;
	String reverse() const;
	String insert(int p_at_pos, const String &p_string) const;
	String erase(int p_pos, int p_chars = 1) const;
	String remove_char(char32_t p_what) const;
	String remove_chars(const String &p_chars) const;
	String remove_chars(const char *p_chars) const;
	String pad_decimals(int p_digits) const;
	String pad_zeros(int p_digits) const;
	String trim_prefix(const String &p_prefix) const;
	String trim_prefix(const char *p_prefix) const;
	String trim_suffix(const String &p_suffix) const;
	String trim_suffix(const char *p_suffix) const;
	String lpad(int min_length, const String &character = " ") const;
	String rpad(int min_length, const String &character = " ") const;
	String sprintf(const Span<Variant> &values, bool *error) const;
	String quote(const String &quotechar = "\"") const;
	String unquote() const;
	static String num(double p_num, int p_decimals = -1, bool remove_extra_trailing_zeroes = true);
	static String num_scientific(double p_num);
	static String num_scientific(float p_num);
	static String num_real(double p_num, bool p_trailing = true);
	static String num_real(float p_num, bool p_trailing = true);
	static String num_int64(int64_t p_num, int base = 10, bool capitalize_hex = false);
	static String num_uint64(uint64_t p_num, int base = 10, bool capitalize_hex = false);
	static String chr(char32_t p_char) {
		String string;
		string.append_utf32(Span(&p_char, 1));
		return string;
	}
	static String md5(const uint8_t *p_md5);
	static String hex_encode_buffer(const uint8_t *p_buffer, int p_len);
	Vector<uint8_t> hex_decode() const;

	bool is_numeric() const;

	double to_float() const;
	int64_t hex_to_int() const;
	int64_t bin_to_int() const;
	int64_t to_int() const;

	static int64_t to_int(const char *p_str, int p_len = -1);
	static int64_t to_int(const wchar_t *p_str, int p_len = -1);
	static int64_t to_int(const char32_t *p_str, int p_len = -1, bool p_clamp = false);

	static double to_float(const char *p_str);
	static double to_float(const wchar_t *p_str, const wchar_t **r_end = nullptr);
	static double to_float(const char32_t *p_str, const char32_t **r_end = nullptr);
	static uint32_t num_characters(int64_t p_int);

	String capitalize() const;
	String to_camel_case() const;
	String to_pascal_case() const;
	String to_snake_case() const;
	String to_kebab_case() const;

	String get_with_code_lines() const;
	int get_slice_count(const String &p_splitter) const;
	int get_slice_count(const char *p_splitter) const;
	String get_slice(const String &p_splitter, int p_slice) const;
	String get_slice(const char *p_splitter, int p_slice) const;
	String get_slicec(char32_t p_splitter, int p_slice) const;

	Vector<String> split(const String &p_splitter = "", bool p_allow_empty = true, int p_maxsplit = 0) const;
	Vector<String> split(const char *p_splitter = "", bool p_allow_empty = true, int p_maxsplit = 0) const;
	Vector<String> rsplit(const String &p_splitter = "", bool p_allow_empty = true, int p_maxsplit = 0) const;
	Vector<String> rsplit(const char *p_splitter = "", bool p_allow_empty = true, int p_maxsplit = 0) const;
	Vector<String> split_spaces(int p_maxsplit = 0) const;
	Vector<double> split_floats(const String &p_splitter, bool p_allow_empty = true) const;
	Vector<float> split_floats_mk(const Vector<String> &p_splitters, bool p_allow_empty = true) const;
	Vector<int> split_ints(const String &p_splitter, bool p_allow_empty = true) const;
	Vector<int> split_ints_mk(const Vector<String> &p_splitters, bool p_allow_empty = true) const;

	String join(const Vector<String> &parts) const;

	static char32_t char_uppercase(char32_t p_char);
	static char32_t char_lowercase(char32_t p_char);
	String to_upper() const;
	String to_lower() const;

	int count(const String &p_string, int p_from = 0, int p_to = 0) const;
	int count(const char *p_string, int p_from = 0, int p_to = 0) const;
	int countn(const String &p_string, int p_from = 0, int p_to = 0) const;
	int countn(const char *p_string, int p_from = 0, int p_to = 0) const;

	String left(int p_len) const;
	String right(int p_len) const;
	String indent(const String &p_prefix) const;
	String dedent() const;
	String strip_edges(bool left = true, bool right = true) const;
	String strip_escapes() const;
	String lstrip(const String &p_chars) const;
	String rstrip(const String &p_chars) const;
	String get_extension() const;
	String get_basename() const;
	String path_join(const String &p_path) const;
	char32_t unicode_at(int p_idx) const;

	CharString ascii(bool p_allow_extended = false) const;
	// Parse an ascii string.
	// If any character is > 127, an error will be logged, and 0xfffd will be inserted.
	Error append_ascii(const Span<char> &p_range);
	static String ascii(const Span<char> &p_range) {
		String s;
		s.append_ascii(p_range);
		return s;
	}
	CharString latin1() const { return ascii(true); }
	void append_latin1(const Span<char> &p_cstr);
	static String latin1(const Span<char> &p_string) {
		String string;
		string.append_latin1(p_string);
		return string;
	}

	CharString utf8(Vector<uint8_t> *r_ch_length_map = nullptr) const;
	Error append_utf8(const char *p_utf8, int p_len = -1, bool p_skip_cr = false);
	Error append_utf8(const Span<char> &p_range, bool p_skip_cr = false) {
		return append_utf8(p_range.ptr(), p_range.size(), p_skip_cr);
	}
	static String utf8(const char *p_utf8, int p_len = -1) {
		String ret;
		ret.append_utf8(p_utf8, p_len);
		return ret;
	}
	static String utf8(const Span<char> &p_range) { return utf8(p_range.ptr(), p_range.size()); }

	Char16String utf16() const;
	Error append_utf16(const char16_t *p_utf16, int p_len = -1, bool p_default_little_endian = true);
	Error append_utf16(const Span<char16_t> p_range, bool p_skip_cr = false) {
		return append_utf16(p_range.ptr(), p_range.size(), p_skip_cr);
	}
	static String utf16(const char16_t *p_utf16, int p_len = -1) {
		String ret;
		ret.append_utf16(p_utf16, p_len);
		return ret;
	}
	static String utf16(const Span<char16_t> &p_range) { return utf16(p_range.ptr(), p_range.size()); }

	void append_utf32(const Span<char32_t> &p_cstr);
	static String utf32(const Span<char32_t> &p_span) {
		String string;
		string.append_utf32(p_span);
		return string;
	}

	// String-formats `p_value` according to `p_format_spec`.
	// If `p_value` cannot be formatted according to this format spec, return error.
	Error append_formatted(const FormatSpec &p_format_spec, const Variant &p_value);
	static String formatted(const FormatSpec &p_format_spec, const Variant &p_value) {
		String string;
		string.append_formatted(p_format_spec, p_value);
		return string;
	}

	static uint32_t hash(const char32_t *p_cstr, int p_len); /* hash the string */
	static uint32_t hash(const char32_t *p_cstr); /* hash the string */
	static uint32_t hash(const wchar_t *p_cstr, int p_len); /* hash the string */
	static uint32_t hash(const wchar_t *p_cstr); /* hash the string */
	static uint32_t hash(const char *p_cstr, int p_len); /* hash the string */
	static uint32_t hash(const char *p_cstr); /* hash the string */
	uint32_t hash() const; /* hash the string */
	uint64_t hash64() const; /* hash the string */
	String md5_text() const;
	String sha1_text() const;
	String sha256_text() const;
	Vector<uint8_t> md5_buffer() const;
	Vector<uint8_t> sha1_buffer() const;
	Vector<uint8_t> sha256_buffer() const;

	_FORCE_INLINE_ bool contains(const char *p_str) const { return find(p_str) != -1; }
	_FORCE_INLINE_ bool contains(const String &p_str) const { return find(p_str) != -1; }
	_FORCE_INLINE_ bool contains_char(char32_t p_chr) const { return find_char(p_chr) != -1; }
	_FORCE_INLINE_ bool containsn(const char *p_str) const { return findn(p_str) != -1; }
	_FORCE_INLINE_ bool containsn(const String &p_str) const { return findn(p_str) != -1; }

	// path functions
	bool is_absolute_path() const;
	bool is_relative_path() const;
	bool is_resource_file() const;
	String path_to(const String &p_path) const;
	String path_to_file(const String &p_path) const;
	String get_base_dir() const;
	String get_file() const;
	static String humanize_size(uint64_t p_size);
	String simplify_path() const;
	bool is_network_share_path() const;

	String xml_escape(bool p_escape_quotes = false) const;
	String xml_unescape() const;
	String uri_encode() const;
	String uri_decode() const;
	String uri_file_decode() const;
	String c_escape() const;
	String c_escape_multiline() const;
	String c_unescape() const;
	String json_escape() const;
	String regex_escape() const;
	Error parse_url(String &r_scheme, String &r_host, int &r_port, String &r_path, String &r_fragment) const;

	String property_name_encode() const;

	// node functions
	static String get_invalid_node_name_characters(bool p_allow_internal = false);
	String validate_node_name() const;
	String validate_ascii_identifier() const;
	String validate_unicode_identifier() const;
	String validate_filename() const;

	bool is_valid_ascii_identifier() const;
	bool is_valid_unicode_identifier() const;
	bool is_valid_int() const;
	bool is_valid_float() const;
	bool is_valid_hex_number(bool p_with_prefix) const;
	bool is_valid_html_color() const;
	bool is_valid_ip_address() const;
	bool is_valid_filename() const;

	// Use `is_valid_ascii_identifier()` instead. Kept for compatibility.
	bool is_valid_identifier() const { return is_valid_ascii_identifier(); }

	/**
	 * The constructors must not depend on other overloads
	 */

	_FORCE_INLINE_ String() {}
	_FORCE_INLINE_ String(const String &p_str) = default;
	_FORCE_INLINE_ String(String &&p_str) = default;
#ifdef SIZE_EXTRA
	_NO_INLINE_ ~String() {}
#endif
	_FORCE_INLINE_ void operator=(const String &p_str) { _cowdata = p_str._cowdata; }
	_FORCE_INLINE_ void operator=(String &&p_str) { _cowdata = std::move(p_str._cowdata); }

	Vector<uint8_t> to_ascii_buffer() const;
	Vector<uint8_t> to_utf8_buffer() const;
	Vector<uint8_t> to_utf16_buffer() const;
	Vector<uint8_t> to_utf32_buffer() const;
	Vector<uint8_t> to_wchar_buffer() const;
	Vector<uint8_t> to_multibyte_char_buffer(const String &p_encoding = String()) const;

	// Constructors for NULL terminated C strings.
	String(const char *p_cstr) {
		append_latin1(p_cstr);
	}
	String(const wchar_t *p_cstr) {
		append_wstring(p_cstr);
	}
	String(const char32_t *p_cstr) {
		append_utf32(p_cstr);
	}

	// Copy assignment for NULL terminated C strings.
	void operator=(const char *p_cstr) {
		clear();
		append_latin1(p_cstr);
	}
	void operator=(const wchar_t *p_cstr) {
		clear();
		append_wstring(p_cstr);
	}
	void operator=(const char32_t *p_cstr) {
		clear();
		append_utf32(p_cstr);
	}
};

// Zero-constructing String initializes _cowdata.ptr() to nullptr and thus empty.
template <>
struct is_zero_constructible<String> : std::true_type {};

bool operator==(const char *p_chr, const String &p_str);
bool operator==(const wchar_t *p_chr, const String &p_str);
bool operator!=(const char *p_chr, const String &p_str);
bool operator!=(const wchar_t *p_chr, const String &p_str);

String operator+(const char *p_chr, const String &p_str);
String operator+(const wchar_t *p_chr, const String &p_str);
String operator+(char32_t p_chr, const String &p_str);

String itos(int64_t p_val);
String uitos(uint64_t p_val);
String rtos(double p_val);
String rtoss(double p_val); //scientific version

struct NoCaseComparator {
	bool operator()(const String &p_a, const String &p_b) const {
		return p_a.nocasecmp_to(p_b) < 0;
	}
};

struct NaturalNoCaseComparator {
	bool operator()(const String &p_a, const String &p_b) const {
		return p_a.naturalnocasecmp_to(p_b) < 0;
	}
};

struct FileNoCaseComparator {
	bool operator()(const String &p_a, const String &p_b) const {
		return p_a.filenocasecmp_to(p_b) < 0;
	}
};

// Specifies how a given value should be string-formatted. See FormatSpec documentation for context
// and semantics.
struct ValueFormatOptions {
	// Minimum field width, including prefixes, separators, and other formatting characters.
	int min_width = 0;

	enum struct Alignment {
		DEFAULT, // Default per formatted value type. Right for numbers, left for other types.
		LEFT, // '<'
		RIGHT, // '>'
		CENTER, // '^'
		PAD_AFTER_SIGN, // '='. For numbers, pad after the sign and before the number.
	};
	Alignment align_type = Alignment::DEFAULT;
	char32_t align_pad = ' ';

	enum struct SignType {
		MINUS_ONLY, // '-'. Show sign only for negative numbers. Default.
		PLUS_MINUS, // '+'. Always show sign.
		SPACE_MINUS, // ' '. Show space for positive numbers, sign for negative numbers.
	};
	SignType sign_type = SignType::MINUS_ONLY;

	// Add '0b', '0o', '0x', '0X' prefixes for binary, octal, hexadecimal types.
	bool use_base_prefix = false; // '#'

	// Separate floats and decimals by groups of 3 digits, binary, octal, hex by 4 digits.
	// group_separator == nullopt means no grouping.
	std::optional<char32_t> group_separator = std::nullopt;

	// For Float type, precision is the number of digits to be displayed _after_ the decimal point. Default 6.
	// For FloatSigFig type, precision is the number of digits before _and_ after the decimal point.
	// For String type, precision is the maximum field size.
	// For int types, precision is not allowed.
	// Nullopt precision indicates precision is not specified (use defaults).
	std::optional<uint8_t> precision = std::nullopt;

	enum struct Type {
		// Types have '_T' ending because the Windows compiler defines some all cap types like CHAR.
		DEFAULT, // Default per argument type.
		STRING_T, // 's'
		CHAR_T, // 'c'
		DECIMAL_T, // 'd'
		BINARY_T, // 'b'
		OCTAL_T, // 'o'
		HEX_T, // 'x'
		HEX_UPPER_T, // 'X'
		FLOAT_T, // 'f'
		FLOAT_SIG_FIG_T, // 'h'  Float but with precision meaning significant figures.
		// TODO: Consider adding support for GENERAL, SCIENTIFIC when they have precision handling formatters in String.
		VECTOR_T, // 'v'
	};
	Type format_type = Type::DEFAULT;

	// Whether this format is for a Vector element.
	// This lets us print floats differently when Vector's real_t is float and not double.
	bool is_vector_element = false;
};

// FormatSpec specifies a configuration for string-formatting a value.
// This can be used to directly format values with `FormatSpec::format`, or within
// `String::format` substitution fields. This configuration can be generated via
// FormatSpec::parse(format_spec) using a configuration string of the following form:
//
// format_spec ::= [[fill]align][sign]["#"]["0"][width][grouping]["." precision][type]
// fill        ::= <any character>
// align       ::= "<" | ">" | "^" | "="
// sign        ::= "+" | "-" | " "
// width       ::= digit+
// grouping    ::= "_" | "," | "."
// precision   ::= digit+
// type        ::= "s" | "c" | "d" | "b" | "o" | "x" | "X" | "f" | "h" | "v" [velement]
// velement    ::= "[" <any character>+ "]"
//
// If a valid `align` value is specified, it can be preceded by a `fill` character that
// can be any character and defaults to a space if omitted.
//
// The meaning of the various alignment options are as follows:
// '<': Forces the field to be left-aligned within the available space (default for most values).
// '>': Forces the field to be right-aligned within the available space (default for numbers).
// '^': Forces the field to be centered within the available space.
// '=': Forces the padding to be placed after the sign (if any) but before the digits.
//
// Note that unless a minimum field width is defined, the field width will always be
// the same size as the data to fill it, so that the alignment option has no meaning in
// this case.
//
// The `sign` option is only valid for number types and can be one of the following:
// '+': Indicates that a sign should be used for both positive as well as negative numbers.
// '-': Indicates that a sign should be used only for negative numbers (default).
// ' ': Indicates that a leading space should be used on positive numbers, and a
//      minus sign on negative numbers.
//
// The '#' option causes the 'alternate form' to be used for the conversion.
// For integers in binary, octal, or hexadecimal output, this option adds the
// respective prefix "0b", "0o", "0x", "0X" to the output value.
//
// When no explicit alignment is given, preceding the width field by a '0' character
// enables sign-aware zero-padding for numbers. This is equivalent to a fill character
// of '0' with an alignment type of '='.
//
// `width` is a decimal integer defining the minimum total field width, including
// any prefixes, separators, and other formatting characters. If not specified, then
// the field width will be determined by the content.
//
// The `precision` is a decimal integer indicating how many digits should be displayed
// after the decimal point for presentation type 'f', or before and after the decimal
// point for presentation type 'h'. For string presentation types the field indicates
// the maximum field size - in other words, how many characters will be used from the
// start of the field content. `precision` is not supported for integer presentation
// types.
//
// The `type` determines how the data should be presented.
//
// The available string presentation types are:
// 's': String format. This is the default type for strings and may be omitted.
// None: the same as 's'.
//
// The available integer presentation types are:
// 'd': Decimal integer. Outputs the number in base 10.
// 'b': Binary format. Outputs the number in base 2.
// 'o': Octal format. Outputs the number in base 8.
// 'x': Hex format. Outputs the number in base 16, using lower-case letters for
//      digits above 9.
// 'X': Hex format. Outputs the number in base 16, using upper-case letters for
//      digits above 9. If '#' is specified, the prefix '0x' will be upper-cased
//      to '0X' as well.
// 'c': Character. Converts the integer to the corresponding unicode character
//      before printing.
// None: The same as 'd'.
//
// The available float presentation types are:
// 'f': Fixed-point notation. For a given precision `p`, formats the number as
//      a decimal number with at most p digits following the decimal point but
//      with trailing 0's removed except for a zero in the tenth's digit.
//      With no precision given, uses a precision of 6 digits.
// 'h': Fixed-point notation but with precision specifying significant figures.
//      For a given precision `p`, formats the number with `p` digits combined
//      before and after the decimal point. This precision is only used to
//      limit the number of digits following the decimal point, not to zero
//      out any digits preceding the decimal point. For example, 1234.5678
//      printed with a precision of 6 will print "1234.57", whereas printing
//      12345678.9 with a precision of 6 will print "12345679".
// None: The same as 'f'.
//
// Vector presentation types:
// 'v': Formats the vector as a set of values e.g. "(value1, value2)".
//      If `velement` is specified, it is interpreted as another format spec
//      that will be used to format each vector element value. For example,
//      'v[,]' applied to Vector2i(12345, 67890) will produce "(12,345, 67,890)".
//      Options applied to the vector and options applied to its elements are
//      independent. So for example, both the vector and its elements can have
//      a specified width. e.g. '20v[5x]'.
// None: The same as 'v'.
//
// Note that this configuration mostly follows the Python format specification
// mini-language (https://docs.python.org/3/library/string.html#formatspec).
struct FormatSpec : public ValueFormatOptions {
	// Implementation detail:
	// FormatSpec holds two sets of formatting values.
	// - The primary format values, held directly via the superclass ValueFormatOptions fields.
	// - The vector element format values, held in the vector_element_format field.
	//   This is used to format the individual elements when formatting a Vector-typed value.

	FormatSpec() = default;
	FormatSpec(const FormatSpec &other) = default;
	FormatSpec &operator=(const FormatSpec &other) = default;
	FormatSpec(const ValueFormatOptions &base);
	FormatSpec &operator=(const ValueFormatOptions &base);

	// Parses a format spec string into a FormatSpec struct. See the format string specification above.
	// If p_format_spec_text is invalid, r_error = true, and an error message is returned via r_error_msg.
	static FormatSpec parse(const String &p_format_spec_text, bool *r_error = nullptr, String *r_error_msg = nullptr);

	// For a VECTOR spec, this specifies the format of each vector element.
	ValueFormatOptions vector_element_format;
};

/* end of namespace */

// Tool translate (TTR and variants) for the editor UI,
// and doc translate for the class reference (DTR).
#ifdef TOOLS_ENABLED
// Gets parsed.
String TTR(const String &p_text, const String &p_context = "");
String TTRN(const String &p_text, const String &p_text_plural, int p_n, const String &p_context = "");
String DTR(const String &p_text, const String &p_context = "");
String DTRN(const String &p_text, const String &p_text_plural, int p_n, const String &p_context = "");
// Use for C strings.
#define TTRC(m_value) (m_value)
// Use to avoid parsing (for use later with C strings).
#define TTRGET(m_value) TTR(m_value)

#else
#define TTRC(m_value) (m_value)
#define TTRGET(m_value) (m_value)
#endif

// Use this to mark property names for editor translation.
// Often for dynamic properties defined in _get_property_list().
// Property names defined directly inside EDITOR_DEF, GLOBAL_DEF, and ADD_PROPERTY macros don't need this.
#define PNAME(m_value) (m_value)

// Similar to PNAME, but to mark groups, i.e. properties with PROPERTY_USAGE_GROUP.
// Groups defined directly inside ADD_GROUP macros don't need this.
// The arguments are the same as ADD_GROUP. m_prefix is only used for extraction.
#define GNAME(m_value, m_prefix) (m_value)

// Runtime translate for the public node API.
String RTR(const String &p_text, const String &p_context = "");
String RTRN(const String &p_text, const String &p_text_plural, int p_n, const String &p_context = "");

/**
 * "Extractable TRanslate". Used for strings that can appear inside an exported
 * project (such as the ones in nodes like `FileDialog`), which are made possible
 * to add in the POT generator. A translation context can optionally be specified
 * to disambiguate between identical source strings in translations.
 * When placeholders are desired, use vformat(ETR("Example: %s"), some_string)`.
 * If a string mentions a quantity (and may therefore need a dynamic plural form),
 * use `ETRN()` instead of `ETR()`.
 *
 * NOTE: This function is for string extraction only, and will just return the
 * string it was given. The translation itself should be done internally by nodes
 * with `atr()` instead.
 */
_FORCE_INLINE_ String ETR(const String &p_text, const String &p_context = "") {
	return p_text;
}

/**
 * "Extractable TRanslate for N items". Used for strings that can appear inside an
 * exported project (such as the ones in nodes like `FileDialog`), which are made
 * possible to add in the POT generator. A translation context can optionally be
 * specified to disambiguate between identical source strings in translations.
 * Use `ETR()` if the string doesn't need dynamic plural form. When placeholders
 * are desired, use `vformat(ETRN("%d item", "%d items", some_integer), some_integer)`.
 * The placeholder must be present in both strings to avoid run-time warnings in `vformat()`.
 *
 * NOTE: This function is for string extraction only, and will just return the
 * string it was given. The translation itself should be done internally by nodes
 * with `atr()` instead.
 */
_FORCE_INLINE_ String ETRN(const String &p_text, const String &p_text_plural, int p_n, const String &p_context = "") {
	if (p_n == 1) {
		return p_text;
	}
	return p_text_plural;
}

template <typename... P>
_FORCE_INLINE_ Vector<String> sarray(P... p_args) {
	return Vector<String>({ String(p_args)... });
}
