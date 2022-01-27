/*************************************************************************/
/*  ustring.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "ustring.h"

#include "core/crypto/crypto_core.h"
#include "core/math/color.h"
#include "core/math/math_funcs.h"
#include "core/os/memory.h"
#include "core/string/print_string.h"
#include "core/string/translation.h"
#include "core/string/ucaps.h"
#include "core/variant/variant.h"
#include "core/version_generated.gen.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstdint>

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS // to disable build-time warning which suggested to use strcpy_s instead strcpy
#endif

#if defined(MINGW_ENABLED) || defined(_MSC_VER)
#define snprintf _snprintf_s
#endif

static const int MAX_DECIMALS = 32;

static _FORCE_INLINE_ bool is_digit(char32_t c) {
	return (c >= '0' && c <= '9');
}

static _FORCE_INLINE_ bool is_hex_digit(char32_t c) {
	return (is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'));
}

static _FORCE_INLINE_ bool is_upper_case(char32_t c) {
	return (c >= 'A' && c <= 'Z');
}

static _FORCE_INLINE_ bool is_lower_case(char32_t c) {
	return (c >= 'a' && c <= 'z');
}

static _FORCE_INLINE_ char32_t lower_case(char32_t c) {
	return (is_upper_case(c) ? (c + ('a' - 'A')) : c);
}

const char CharString::_null = 0;
const char16_t Char16String::_null = 0;
const char32_t String::_null = 0;

bool is_symbol(char32_t c) {
	return c != '_' && ((c >= '!' && c <= '/') || (c >= ':' && c <= '@') || (c >= '[' && c <= '`') || (c >= '{' && c <= '~') || c == '\t' || c == ' ');
}

bool select_word(const String &p_s, int p_col, int &r_beg, int &r_end) {
	const String &s = p_s;
	int beg = CLAMP(p_col, 0, s.length());
	int end = beg;

	if (s[beg] > 32 || beg == s.length()) {
		bool symbol = beg < s.length() && is_symbol(s[beg]);

		while (beg > 0 && s[beg - 1] > 32 && (symbol == is_symbol(s[beg - 1]))) {
			beg--;
		}
		while (end < s.length() && s[end + 1] > 32 && (symbol == is_symbol(s[end + 1]))) {
			end++;
		}

		if (end < s.length()) {
			end += 1;
		}

		r_beg = beg;
		r_end = end;

		return true;
	} else {
		return false;
	}
}

/*************************************************************************/
/*  Char16String                                                         */
/*************************************************************************/

bool Char16String::operator<(const Char16String &p_right) const {
	if (length() == 0) {
		return p_right.length() != 0;
	}

	return is_str_less(get_data(), p_right.get_data());
}

Char16String &Char16String::operator+=(char16_t p_char) {
	resize(size() ? size() + 1 : 2);
	set(length(), 0);
	set(length() - 1, p_char);

	return *this;
}

void Char16String::operator=(const char16_t *p_cstr) {
	copy_from(p_cstr);
}

const char16_t *Char16String::get_data() const {
	if (size()) {
		return &operator[](0);
	} else {
		return u"";
	}
}

void Char16String::copy_from(const char16_t *p_cstr) {
	if (!p_cstr) {
		resize(0);
		return;
	}

	const char16_t *s = p_cstr;
	for (; *s; s++) {
	}
	size_t len = s - p_cstr;

	if (len == 0) {
		resize(0);
		return;
	}

	Error err = resize(++len); // include terminating null char

	ERR_FAIL_COND_MSG(err != OK, "Failed to copy char16_t string.");

	memcpy(ptrw(), p_cstr, len * sizeof(char16_t));
}

/*************************************************************************/
/*  CharString                                                           */
/*************************************************************************/

bool CharString::operator<(const CharString &p_right) const {
	if (length() == 0) {
		return p_right.length() != 0;
	}

	return is_str_less(get_data(), p_right.get_data());
}

CharString &CharString::operator+=(char p_char) {
	resize(size() ? size() + 1 : 2);
	set(length(), 0);
	set(length() - 1, p_char);

	return *this;
}

void CharString::operator=(const char *p_cstr) {
	copy_from(p_cstr);
}

const char *CharString::get_data() const {
	if (size()) {
		return &operator[](0);
	} else {
		return "";
	}
}

void CharString::copy_from(const char *p_cstr) {
	if (!p_cstr) {
		resize(0);
		return;
	}

	size_t len = strlen(p_cstr);

	if (len == 0) {
		resize(0);
		return;
	}

	Error err = resize(++len); // include terminating null char

	ERR_FAIL_COND_MSG(err != OK, "Failed to copy C-string.");

	memcpy(ptrw(), p_cstr, len);
}

/*************************************************************************/
/*  String                                                               */
/*************************************************************************/

//kind of poor should be rewritten properly
String String::word_wrap(int p_chars_per_line) const {
	int from = 0;
	int last_space = 0;
	String ret;
	for (int i = 0; i < length(); i++) {
		if (i - from >= p_chars_per_line) {
			if (last_space == -1) {
				ret += substr(from, i - from + 1) + "\n";
			} else {
				ret += substr(from, last_space - from) + "\n";
				i = last_space; //rewind
			}
			from = i + 1;
			last_space = -1;
		} else if (operator[](i) == ' ' || operator[](i) == '\t') {
			last_space = i;
		} else if (operator[](i) == '\n') {
			ret += substr(from, i - from) + "\n";
			from = i + 1;
			last_space = -1;
		}
	}

	if (from < length()) {
		ret += substr(from, length());
	}

	return ret;
}

Error String::parse_url(String &r_scheme, String &r_host, int &r_port, String &r_path) const {
	// Splits the URL into scheme, host, port, path. Strip credentials when present.
	String base = *this;
	r_scheme = "";
	r_host = "";
	r_port = 0;
	r_path = "";
	int pos = base.find("://");
	// Scheme
	if (pos != -1) {
		r_scheme = base.substr(0, pos + 3).to_lower();
		base = base.substr(pos + 3, base.length() - pos - 3);
	}
	pos = base.find("/");
	// Path
	if (pos != -1) {
		r_path = base.substr(pos, base.length() - pos);
		base = base.substr(0, pos);
	}
	// Host
	pos = base.find("@");
	if (pos != -1) {
		// Strip credentials
		base = base.substr(pos + 1, base.length() - pos - 1);
	}
	if (base.begins_with("[")) {
		// Literal IPv6
		pos = base.rfind("]");
		if (pos == -1) {
			return ERR_INVALID_PARAMETER;
		}
		r_host = base.substr(1, pos - 1);
		base = base.substr(pos + 1, base.length() - pos - 1);
	} else {
		// Anything else
		if (base.get_slice_count(":") > 2) {
			return ERR_INVALID_PARAMETER;
		}
		pos = base.rfind(":");
		if (pos == -1) {
			r_host = base;
			base = "";
		} else {
			r_host = base.substr(0, pos);
			base = base.substr(pos, base.length() - pos);
		}
	}
	if (r_host.is_empty()) {
		return ERR_INVALID_PARAMETER;
	}
	r_host = r_host.to_lower();
	// Port
	if (base.begins_with(":")) {
		base = base.substr(1, base.length() - 1);
		if (!base.is_valid_int()) {
			return ERR_INVALID_PARAMETER;
		}
		r_port = base.to_int();
		if (r_port < 1 || r_port > 65535) {
			return ERR_INVALID_PARAMETER;
		}
	}
	return OK;
}

void String::copy_from(const char *p_cstr) {
	// copy Latin-1 encoded c-string directly
	if (!p_cstr) {
		resize(0);
		return;
	}

	int len = 0;
	const char *ptr = p_cstr;
	while (*(ptr++) != 0) {
		len++;
	}

	if (len == 0) {
		resize(0);
		return;
	}

	resize(len + 1); // include 0

	char32_t *dst = this->ptrw();

	for (int i = 0; i < len + 1; i++) {
		dst[i] = p_cstr[i];
	}
}

void String::copy_from(const char *p_cstr, const int p_clip_to) {
	// copy Latin-1 encoded c-string directly
	if (!p_cstr) {
		resize(0);
		return;
	}

	int len = 0;
	const char *ptr = p_cstr;
	while ((p_clip_to < 0 || len < p_clip_to) && *(ptr++) != 0) {
		len++;
	}

	if (len == 0) {
		resize(0);
		return;
	}

	resize(len + 1); // include 0

	char32_t *dst = this->ptrw();

	for (int i = 0; i < len; i++) {
		dst[i] = p_cstr[i];
	}
	dst[len] = 0;
}

void String::copy_from(const wchar_t *p_cstr) {
#ifdef WINDOWS_ENABLED
	// wchar_t is 16-bit, parse as UTF-16
	parse_utf16((const char16_t *)p_cstr);
#else
	// wchar_t is 32-bit, copy directly
	copy_from((const char32_t *)p_cstr);
#endif
}

void String::copy_from(const wchar_t *p_cstr, const int p_clip_to) {
#ifdef WINDOWS_ENABLED
	// wchar_t is 16-bit, parse as UTF-16
	parse_utf16((const char16_t *)p_cstr, p_clip_to);
#else
	// wchar_t is 32-bit, copy directly
	copy_from((const char32_t *)p_cstr, p_clip_to);
#endif
}

void String::copy_from(const char32_t &p_char) {
	resize(2);
	if ((p_char >= 0xd800 && p_char <= 0xdfff) || (p_char > 0x10ffff)) {
		print_error("Unicode parsing error: Invalid unicode codepoint " + num_int64(p_char, 16) + ".");
		set(0, 0xfffd);
	} else {
		set(0, p_char);
	}
	set(1, 0);
}

void String::copy_from(const char32_t *p_cstr) {
	if (!p_cstr) {
		resize(0);
		return;
	}

	int len = 0;
	const char32_t *ptr = p_cstr;
	while (*(ptr++) != 0) {
		len++;
	}

	if (len == 0) {
		resize(0);
		return;
	}

	copy_from_unchecked(p_cstr, len);
}

void String::copy_from(const char32_t *p_cstr, const int p_clip_to) {
	if (!p_cstr) {
		resize(0);
		return;
	}

	int len = 0;
	const char32_t *ptr = p_cstr;
	while ((p_clip_to < 0 || len < p_clip_to) && *(ptr++) != 0) {
		len++;
	}

	if (len == 0) {
		resize(0);
		return;
	}

	copy_from_unchecked(p_cstr, len);
}

// assumes the following have already been validated:
// p_char != nullptr
// p_length > 0
// p_length <= p_char strlen
void String::copy_from_unchecked(const char32_t *p_char, const int p_length) {
	resize(p_length + 1);
	set(p_length, 0);

	char32_t *dst = ptrw();

	for (int i = 0; i < p_length; i++) {
		if ((p_char[i] >= 0xd800 && p_char[i] <= 0xdfff) || (p_char[i] > 0x10ffff)) {
			print_error("Unicode parsing error: Invalid unicode codepoint " + num_int64(p_char[i], 16) + ".");
			dst[i] = 0xfffd;
		} else {
			dst[i] = p_char[i];
		}
	}
}

void String::operator=(const char *p_str) {
	copy_from(p_str);
}

void String::operator=(const char32_t *p_str) {
	copy_from(p_str);
}

void String::operator=(const wchar_t *p_str) {
	copy_from(p_str);
}

String String::operator+(const String &p_str) const {
	String res = *this;
	res += p_str;
	return res;
}

String operator+(const char *p_chr, const String &p_str) {
	String tmp = p_chr;
	tmp += p_str;
	return tmp;
}

String operator+(const wchar_t *p_chr, const String &p_str) {
#ifdef WINDOWS_ENABLED
	// wchar_t is 16-bit
	String tmp = String::utf16((const char16_t *)p_chr);
#else
	// wchar_t is 32-bi
	String tmp = (const char32_t *)p_chr;
#endif
	tmp += p_str;
	return tmp;
}

String operator+(char32_t p_chr, const String &p_str) {
	return (String::chr(p_chr) + p_str);
}

String &String::operator+=(const String &p_str) {
	if (is_empty()) {
		*this = p_str;
		return *this;
	}

	if (p_str.is_empty()) {
		return *this;
	}

	int from = length();

	resize(length() + p_str.size());

	const char32_t *src = p_str.get_data();
	char32_t *dst = ptrw();

	set(length(), 0);

	for (int i = 0; i < p_str.length(); i++) {
		dst[from + i] = src[i];
	}

	return *this;
}

String &String::operator+=(const char *p_str) {
	if (!p_str || p_str[0] == 0) {
		return *this;
	}

	int src_len = 0;
	const char *ptr = p_str;
	while (*(ptr++) != 0) {
		src_len++;
	}

	int from = length();

	resize(from + src_len + 1);

	char32_t *dst = ptrw();

	set(length(), 0);

	for (int i = 0; i < src_len; i++) {
		dst[from + i] = p_str[i];
	}

	return *this;
}

String &String::operator+=(const wchar_t *p_str) {
#ifdef WINDOWS_ENABLED
	// wchar_t is 16-bit
	*this += String::utf16((const char16_t *)p_str);
#else
	// wchar_t is 32-bit
	*this += String((const char32_t *)p_str);
#endif
	return *this;
}

String &String::operator+=(const char32_t *p_str) {
	*this += String(p_str);
	return *this;
}

String &String::operator+=(char32_t p_char) {
	resize(size() ? size() + 1 : 2);
	set(length(), 0);
	if ((p_char >= 0xd800 && p_char <= 0xdfff) || (p_char > 0x10ffff)) {
		print_error("Unicode parsing error: Invalid unicode codepoint " + num_int64(p_char, 16) + ".");
		set(length() - 1, 0xfffd);
	} else {
		set(length() - 1, p_char);
	}

	return *this;
}

bool String::operator==(const char *p_str) const {
	// compare Latin-1 encoded c-string
	int len = 0;
	const char *aux = p_str;

	while (*(aux++) != 0) {
		len++;
	}

	if (length() != len) {
		return false;
	}
	if (is_empty()) {
		return true;
	}

	int l = length();

	const char32_t *dst = get_data();

	// Compare char by char
	for (int i = 0; i < l; i++) {
		if ((char32_t)p_str[i] != dst[i]) {
			return false;
		}
	}

	return true;
}

bool String::operator==(const wchar_t *p_str) const {
#ifdef WINDOWS_ENABLED
	// wchar_t is 16-bit, parse as UTF-16
	return *this == String::utf16((const char16_t *)p_str);
#else
	// wchar_t is 32-bit, compare char by char
	return *this == (const char32_t *)p_str;
#endif
}

bool String::operator==(const char32_t *p_str) const {
	int len = 0;
	const char32_t *aux = p_str;

	while (*(aux++) != 0) {
		len++;
	}

	if (length() != len) {
		return false;
	}
	if (is_empty()) {
		return true;
	}

	int l = length();

	const char32_t *dst = get_data();

	/* Compare char by char */
	for (int i = 0; i < l; i++) {
		if (p_str[i] != dst[i]) {
			return false;
		}
	}

	return true;
}

bool String::operator==(const String &p_str) const {
	if (length() != p_str.length()) {
		return false;
	}
	if (is_empty()) {
		return true;
	}

	int l = length();

	const char32_t *src = get_data();
	const char32_t *dst = p_str.get_data();

	/* Compare char by char */
	for (int i = 0; i < l; i++) {
		if (src[i] != dst[i]) {
			return false;
		}
	}

	return true;
}

bool String::operator==(const StrRange &p_str_range) const {
	int len = p_str_range.len;

	if (length() != len) {
		return false;
	}
	if (is_empty()) {
		return true;
	}

	const char32_t *c_str = p_str_range.c_str;
	const char32_t *dst = &operator[](0);

	/* Compare char by char */
	for (int i = 0; i < len; i++) {
		if (c_str[i] != dst[i]) {
			return false;
		}
	}

	return true;
}

bool operator==(const char *p_chr, const String &p_str) {
	return p_str == p_chr;
}

bool operator==(const wchar_t *p_chr, const String &p_str) {
#ifdef WINDOWS_ENABLED
	// wchar_t is 16-bit
	return p_str == String::utf16((const char16_t *)p_chr);
#else
	// wchar_t is 32-bi
	return p_str == String((const char32_t *)p_chr);
#endif
}

bool operator!=(const char *p_chr, const String &p_str) {
	return !(p_str == p_chr);
}

bool operator!=(const wchar_t *p_chr, const String &p_str) {
#ifdef WINDOWS_ENABLED
	// wchar_t is 16-bit
	return !(p_str == String::utf16((const char16_t *)p_chr));
#else
	// wchar_t is 32-bi
	return !(p_str == String((const char32_t *)p_chr));
#endif
}

bool String::operator!=(const char *p_str) const {
	return (!(*this == p_str));
}

bool String::operator!=(const wchar_t *p_str) const {
	return (!(*this == p_str));
}

bool String::operator!=(const char32_t *p_str) const {
	return (!(*this == p_str));
}

bool String::operator!=(const String &p_str) const {
	return !((*this == p_str));
}

bool String::operator<=(const String &p_str) const {
	return !(p_str < *this);
}

bool String::operator>(const String &p_str) const {
	return p_str < *this;
}

bool String::operator>=(const String &p_str) const {
	return !(*this < p_str);
}

bool String::operator<(const char *p_str) const {
	if (is_empty() && p_str[0] == 0) {
		return false;
	}
	if (is_empty()) {
		return true;
	}
	return is_str_less(get_data(), p_str);
}

bool String::operator<(const wchar_t *p_str) const {
	if (is_empty() && p_str[0] == 0) {
		return false;
	}
	if (is_empty()) {
		return true;
	}

#ifdef WINDOWS_ENABLED
	// wchar_t is 16-bit
	return is_str_less(get_data(), String::utf16((const char16_t *)p_str).get_data());
#else
	// wchar_t is 32-bit
	return is_str_less(get_data(), (const char32_t *)p_str);
#endif
}

bool String::operator<(const char32_t *p_str) const {
	if (is_empty() && p_str[0] == 0) {
		return false;
	}
	if (is_empty()) {
		return true;
	}

	return is_str_less(get_data(), p_str);
}

bool String::operator<(const String &p_str) const {
	return operator<(p_str.get_data());
}

signed char String::nocasecmp_to(const String &p_str) const {
	if (is_empty() && p_str.is_empty()) {
		return 0;
	}
	if (is_empty()) {
		return -1;
	}
	if (p_str.is_empty()) {
		return 1;
	}

	const char32_t *that_str = p_str.get_data();
	const char32_t *this_str = get_data();

	while (true) {
		if (*that_str == 0 && *this_str == 0) {
			return 0; //we're equal
		} else if (*this_str == 0) {
			return -1; //if this is empty, and the other one is not, then we're less.. I think?
		} else if (*that_str == 0) {
			return 1; //otherwise the other one is smaller..
		} else if (_find_upper(*this_str) < _find_upper(*that_str)) { //more than
			return -1;
		} else if (_find_upper(*this_str) > _find_upper(*that_str)) { //less than
			return 1;
		}

		this_str++;
		that_str++;
	}
}

signed char String::casecmp_to(const String &p_str) const {
	if (is_empty() && p_str.is_empty()) {
		return 0;
	}
	if (is_empty()) {
		return -1;
	}
	if (p_str.is_empty()) {
		return 1;
	}

	const char32_t *that_str = p_str.get_data();
	const char32_t *this_str = get_data();

	while (true) {
		if (*that_str == 0 && *this_str == 0) {
			return 0; //we're equal
		} else if (*this_str == 0) {
			return -1; //if this is empty, and the other one is not, then we're less.. I think?
		} else if (*that_str == 0) {
			return 1; //otherwise the other one is smaller..
		} else if (*this_str < *that_str) { //more than
			return -1;
		} else if (*this_str > *that_str) { //less than
			return 1;
		}

		this_str++;
		that_str++;
	}
}

signed char String::naturalnocasecmp_to(const String &p_str) const {
	const char32_t *this_str = get_data();
	const char32_t *that_str = p_str.get_data();

	if (this_str && that_str) {
		while (*this_str == '.' || *that_str == '.') {
			if (*this_str++ != '.') {
				return 1;
			}
			if (*that_str++ != '.') {
				return -1;
			}
			if (!*that_str) {
				return 1;
			}
			if (!*this_str) {
				return -1;
			}
		}

		while (*this_str) {
			if (!*that_str) {
				return 1;
			} else if (is_digit(*this_str)) {
				if (!is_digit(*that_str)) {
					return -1;
				}

				// Keep ptrs to start of numerical sequences
				const char32_t *this_substr = this_str;
				const char32_t *that_substr = that_str;

				// Compare lengths of both numerical sequences, ignoring leading zeros
				while (is_digit(*this_str)) {
					this_str++;
				}
				while (is_digit(*that_str)) {
					that_str++;
				}
				while (*this_substr == '0') {
					this_substr++;
				}
				while (*that_substr == '0') {
					that_substr++;
				}
				int this_len = this_str - this_substr;
				int that_len = that_str - that_substr;

				if (this_len < that_len) {
					return -1;
				} else if (this_len > that_len) {
					return 1;
				}

				// If lengths equal, compare lexicographically
				while (this_substr != this_str && that_substr != that_str) {
					if (*this_substr < *that_substr) {
						return -1;
					} else if (*this_substr > *that_substr) {
						return 1;
					}
					this_substr++;
					that_substr++;
				}
			} else if (is_digit(*that_str)) {
				return 1;
			} else {
				if (_find_upper(*this_str) < _find_upper(*that_str)) { //more than
					return -1;
				} else if (_find_upper(*this_str) > _find_upper(*that_str)) { //less than
					return 1;
				}

				this_str++;
				that_str++;
			}
		}
		if (*that_str) {
			return -1;
		}
	}

	return 0;
}

const char32_t *String::get_data() const {
	static const char32_t zero = 0;
	return size() ? &operator[](0) : &zero;
}

String String::capitalize() const {
	String aux = this->camelcase_to_underscore(true).replace("_", " ").strip_edges();
	String cap;
	for (int i = 0; i < aux.get_slice_count(" "); i++) {
		String slice = aux.get_slicec(' ', i);
		if (slice.length() > 0) {
			slice[0] = _find_upper(slice[0]);
			if (i > 0) {
				cap += " ";
			}
			cap += slice;
		}
	}

	return cap;
}

String String::camelcase_to_underscore(bool lowercase) const {
	const char32_t *cstr = get_data();
	String new_string;
	int start_index = 0;

	for (int i = 1; i < this->size(); i++) {
		bool is_upper = is_upper_case(cstr[i]);
		bool is_number = is_digit(cstr[i]);

		bool are_next_2_lower = false;
		bool is_next_lower = false;
		bool is_next_number = false;
		bool was_precedent_upper = is_upper_case(cstr[i - 1]);
		bool was_precedent_number = is_digit(cstr[i - 1]);

		if (i + 2 < this->size()) {
			are_next_2_lower = is_lower_case(cstr[i + 1]) && is_lower_case(cstr[i + 2]);
		}

		if (i + 1 < this->size()) {
			is_next_lower = is_lower_case(cstr[i + 1]);
			is_next_number = is_digit(cstr[i + 1]);
		}

		const bool cond_a = is_upper && !was_precedent_upper && !was_precedent_number;
		const bool cond_b = was_precedent_upper && is_upper && are_next_2_lower;
		const bool cond_c = is_number && !was_precedent_number;
		const bool can_break_number_letter = is_number && !was_precedent_number && is_next_lower;
		const bool can_break_letter_number = !is_number && was_precedent_number && (is_next_lower || is_next_number);

		bool should_split = cond_a || cond_b || cond_c || can_break_number_letter || can_break_letter_number;
		if (should_split) {
			new_string += this->substr(start_index, i - start_index) + "_";
			start_index = i;
		}
	}

	new_string += this->substr(start_index, this->size() - start_index);
	return lowercase ? new_string.to_lower() : new_string;
}

String String::get_with_code_lines() const {
	const Vector<String> lines = split("\n");
	String ret;
	for (int i = 0; i < lines.size(); i++) {
		if (i > 0) {
			ret += "\n";
		}
		ret += vformat("%4d | %s", i + 1, lines[i]);
	}
	return ret;
}

int String::get_slice_count(String p_splitter) const {
	if (is_empty()) {
		return 0;
	}
	if (p_splitter.is_empty()) {
		return 0;
	}

	int pos = 0;
	int slices = 1;

	while ((pos = find(p_splitter, pos)) >= 0) {
		slices++;
		pos += p_splitter.length();
	}

	return slices;
}

String String::get_slice(String p_splitter, int p_slice) const {
	if (is_empty() || p_splitter.is_empty()) {
		return "";
	}

	int pos = 0;
	int prev_pos = 0;
	//int slices=1;
	if (p_slice < 0) {
		return "";
	}
	if (find(p_splitter) == -1) {
		return *this;
	}

	int i = 0;
	while (true) {
		pos = find(p_splitter, pos);
		if (pos == -1) {
			pos = length(); //reached end
		}

		int from = prev_pos;
		//int to=pos;

		if (p_slice == i) {
			return substr(from, pos - from);
		}

		if (pos == length()) { //reached end and no find
			break;
		}
		pos += p_splitter.length();
		prev_pos = pos;
		i++;
	}

	return ""; //no find!
}

String String::get_slicec(char32_t p_splitter, int p_slice) const {
	if (is_empty()) {
		return String();
	}

	if (p_slice < 0) {
		return String();
	}

	const char32_t *c = this->ptr();
	int i = 0;
	int prev = 0;
	int count = 0;
	while (true) {
		if (c[i] == 0 || c[i] == p_splitter) {
			if (p_slice == count) {
				return substr(prev, i - prev);
			} else if (c[i] == 0) {
				return String();
			} else {
				count++;
				prev = i + 1;
			}
		}

		i++;
	}
}

Vector<String> String::split_spaces() const {
	Vector<String> ret;
	int from = 0;
	int i = 0;
	int len = length();
	if (len == 0) {
		return ret;
	}

	bool inside = false;

	while (true) {
		bool empty = operator[](i) < 33;

		if (i == 0) {
			inside = !empty;
		}

		if (!empty && !inside) {
			inside = true;
			from = i;
		}

		if (empty && inside) {
			ret.push_back(substr(from, i - from));
			inside = false;
		}

		if (i == len) {
			break;
		}
		i++;
	}

	return ret;
}

Vector<String> String::split(const String &p_splitter, bool p_allow_empty, int p_maxsplit) const {
	Vector<String> ret;
	int from = 0;
	int len = length();

	while (true) {
		int end = find(p_splitter, from);
		if (end < 0) {
			end = len;
		}
		if (p_allow_empty || (end > from)) {
			if (p_maxsplit <= 0) {
				ret.push_back(substr(from, end - from));
			} else {
				// Put rest of the string and leave cycle.
				if (p_maxsplit == ret.size()) {
					ret.push_back(substr(from, len));
					break;
				}

				// Otherwise, push items until positive limit is reached.
				ret.push_back(substr(from, end - from));
			}
		}

		if (end == len) {
			break;
		}

		from = end + p_splitter.length();
	}

	return ret;
}

Vector<String> String::rsplit(const String &p_splitter, bool p_allow_empty, int p_maxsplit) const {
	Vector<String> ret;
	const int len = length();
	int remaining_len = len;

	while (true) {
		if (remaining_len < p_splitter.length() || (p_maxsplit > 0 && p_maxsplit == ret.size())) {
			// no room for another splitter or hit max splits, push what's left and we're done
			if (p_allow_empty || remaining_len > 0) {
				ret.push_back(substr(0, remaining_len));
			}
			break;
		}

		int left_edge = rfind(p_splitter, remaining_len - p_splitter.length());

		if (left_edge < 0) {
			// no more splitters, we're done
			ret.push_back(substr(0, remaining_len));
			break;
		}

		int substr_start = left_edge + p_splitter.length();
		if (p_allow_empty || substr_start < remaining_len) {
			ret.push_back(substr(substr_start, remaining_len - substr_start));
		}

		remaining_len = left_edge;
	}

	ret.reverse();
	return ret;
}

Vector<float> String::split_floats(const String &p_splitter, bool p_allow_empty) const {
	Vector<float> ret;
	int from = 0;
	int len = length();

	while (true) {
		int end = find(p_splitter, from);
		if (end < 0) {
			end = len;
		}
		if (p_allow_empty || (end > from)) {
			ret.push_back(String::to_float(&get_data()[from]));
		}

		if (end == len) {
			break;
		}

		from = end + p_splitter.length();
	}

	return ret;
}

Vector<float> String::split_floats_mk(const Vector<String> &p_splitters, bool p_allow_empty) const {
	Vector<float> ret;
	int from = 0;
	int len = length();

	while (true) {
		int idx;
		int end = findmk(p_splitters, from, &idx);
		int spl_len = 1;
		if (end < 0) {
			end = len;
		} else {
			spl_len = p_splitters[idx].length();
		}

		if (p_allow_empty || (end > from)) {
			ret.push_back(String::to_float(&get_data()[from]));
		}

		if (end == len) {
			break;
		}

		from = end + spl_len;
	}

	return ret;
}

Vector<int> String::split_ints(const String &p_splitter, bool p_allow_empty) const {
	Vector<int> ret;
	int from = 0;
	int len = length();

	while (true) {
		int end = find(p_splitter, from);
		if (end < 0) {
			end = len;
		}
		if (p_allow_empty || (end > from)) {
			ret.push_back(String::to_int(&get_data()[from], end - from));
		}

		if (end == len) {
			break;
		}

		from = end + p_splitter.length();
	}

	return ret;
}

Vector<int> String::split_ints_mk(const Vector<String> &p_splitters, bool p_allow_empty) const {
	Vector<int> ret;
	int from = 0;
	int len = length();

	while (true) {
		int idx;
		int end = findmk(p_splitters, from, &idx);
		int spl_len = 1;
		if (end < 0) {
			end = len;
		} else {
			spl_len = p_splitters[idx].length();
		}

		if (p_allow_empty || (end > from)) {
			ret.push_back(String::to_int(&get_data()[from], end - from));
		}

		if (end == len) {
			break;
		}

		from = end + spl_len;
	}

	return ret;
}

String String::join(Vector<String> parts) const {
	String ret;
	for (int i = 0; i < parts.size(); ++i) {
		if (i > 0) {
			ret += *this;
		}
		ret += parts[i];
	}
	return ret;
}

char32_t String::char_uppercase(char32_t p_char) {
	return _find_upper(p_char);
}

char32_t String::char_lowercase(char32_t p_char) {
	return _find_lower(p_char);
}

String String::to_upper() const {
	String upper = *this;

	for (int i = 0; i < upper.size(); i++) {
		const char32_t s = upper[i];
		const char32_t t = _find_upper(s);
		if (s != t) { // avoid copy on write
			upper[i] = t;
		}
	}

	return upper;
}

String String::to_lower() const {
	String lower = *this;

	for (int i = 0; i < lower.size(); i++) {
		const char32_t s = lower[i];
		const char32_t t = _find_lower(s);
		if (s != t) { // avoid copy on write
			lower[i] = t;
		}
	}

	return lower;
}

String String::chr(char32_t p_char) {
	char32_t c[2] = { p_char, 0 };
	return String(c);
}

String String::num(double p_num, int p_decimals) {
	if (Math::is_nan(p_num)) {
		return "nan";
	}

	if (Math::is_inf(p_num)) {
		if (signbit(p_num)) {
			return "-inf";
		} else {
			return "inf";
		}
	}

	if (p_decimals < 0) {
		p_decimals = 14;
		const double abs_num = ABS(p_num);
		if (abs_num > 10) {
			// We want to align the digits to the above sane default, so we only
			// need to subtract log10 for numbers with a positive power of ten.
			p_decimals -= (int)floor(log10(abs_num));
		}
	}
	if (p_decimals > MAX_DECIMALS) {
		p_decimals = MAX_DECIMALS;
	}

	char fmt[7];
	fmt[0] = '%';
	fmt[1] = '.';

	if (p_decimals < 0) {
		fmt[1] = 'l';
		fmt[2] = 'f';
		fmt[3] = 0;
	} else if (p_decimals < 10) {
		fmt[2] = '0' + p_decimals;
		fmt[3] = 'l';
		fmt[4] = 'f';
		fmt[5] = 0;
	} else {
		fmt[2] = '0' + (p_decimals / 10);
		fmt[3] = '0' + (p_decimals % 10);
		fmt[4] = 'l';
		fmt[5] = 'f';
		fmt[6] = 0;
	}
	char buf[256];

#if defined(__GNUC__) || defined(_MSC_VER)
	snprintf(buf, 256, fmt, p_num);
#else
	sprintf(buf, fmt, p_num);
#endif

	buf[255] = 0;
	//destroy trailing zeroes
	{
		bool period = false;
		int z = 0;
		while (buf[z]) {
			if (buf[z] == '.') {
				period = true;
			}
			z++;
		}

		if (period) {
			z--;
			while (z > 0) {
				if (buf[z] == '0') {
					buf[z] = 0;
				} else if (buf[z] == '.') {
					buf[z] = 0;
					break;
				} else {
					break;
				}

				z--;
			}
		}
	}

	return buf;
}

String String::num_int64(int64_t p_num, int base, bool capitalize_hex) {
	bool sign = p_num < 0;

	int64_t n = p_num;

	int chars = 0;
	do {
		n /= base;
		chars++;
	} while (n);

	if (sign) {
		chars++;
	}
	String s;
	s.resize(chars + 1);
	char32_t *c = s.ptrw();
	c[chars] = 0;
	n = p_num;
	do {
		int mod = ABS(n % base);
		if (mod >= 10) {
			char a = (capitalize_hex ? 'A' : 'a');
			c[--chars] = a + (mod - 10);
		} else {
			c[--chars] = '0' + mod;
		}

		n /= base;
	} while (n);

	if (sign) {
		c[0] = '-';
	}

	return s;
}

String String::num_uint64(uint64_t p_num, int base, bool capitalize_hex) {
	uint64_t n = p_num;

	int chars = 0;
	do {
		n /= base;
		chars++;
	} while (n);

	String s;
	s.resize(chars + 1);
	char32_t *c = s.ptrw();
	c[chars] = 0;
	n = p_num;
	do {
		int mod = n % base;
		if (mod >= 10) {
			char a = (capitalize_hex ? 'A' : 'a');
			c[--chars] = a + (mod - 10);
		} else {
			c[--chars] = '0' + mod;
		}

		n /= base;
	} while (n);

	return s;
}

String String::num_real(double p_num, bool p_trailing) {
	if (p_num == (double)(int64_t)p_num) {
		if (p_trailing) {
			return num_int64((int64_t)p_num) + ".0";
		} else {
			return num_int64((int64_t)p_num);
		}
	}
#ifdef REAL_T_IS_DOUBLE
	int decimals = 14;
#else
	int decimals = 6;
#endif
	// We want to align the digits to the above sane default, so we only
	// need to subtract log10 for numbers with a positive power of ten.
	if (p_num > 10) {
		decimals -= (int)floor(log10(p_num));
	}
	return num(p_num, decimals);
}

String String::num_scientific(double p_num) {
	if (Math::is_nan(p_num)) {
		return "nan";
	}

	if (Math::is_inf(p_num)) {
		if (signbit(p_num)) {
			return "-inf";
		} else {
			return "inf";
		}
	}

	char buf[256];

#if defined(__GNUC__) || defined(_MSC_VER)

#if defined(__MINGW32__) && defined(_TWO_DIGIT_EXPONENT) && !defined(_UCRT)
	// MinGW requires _set_output_format() to conform to C99 output for printf
	unsigned int old_exponent_format = _set_output_format(_TWO_DIGIT_EXPONENT);
#endif
	snprintf(buf, 256, "%lg", p_num);

#if defined(__MINGW32__) && defined(_TWO_DIGIT_EXPONENT) && !defined(_UCRT)
	_set_output_format(old_exponent_format);
#endif

#else
	sprintf(buf, "%.16lg", p_num);
#endif

	buf[255] = 0;

	return buf;
}

String String::md5(const uint8_t *p_md5) {
	return String::hex_encode_buffer(p_md5, 16);
}

String String::hex_encode_buffer(const uint8_t *p_buffer, int p_len) {
	static const char hex[16] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' };

	String ret;
	char v[2] = { 0, 0 };

	for (int i = 0; i < p_len; i++) {
		v[0] = hex[p_buffer[i] >> 4];
		ret += v;
		v[0] = hex[p_buffer[i] & 0xF];
		ret += v;
	}

	return ret;
}

CharString String::ascii(bool p_allow_extended) const {
	if (!length()) {
		return CharString();
	}

	CharString cs;
	cs.resize(size());

	for (int i = 0; i < size(); i++) {
		char32_t c = operator[](i);
		if ((c <= 0x7f) || (c <= 0xff && p_allow_extended)) {
			cs[i] = c;
		} else {
			print_error("Unicode parsing error: Cannot represent " + num_int64(c, 16) + " as ASCII/Latin-1 character.");
			cs[i] = 0x20;
		}
	}

	return cs;
}

String String::utf8(const char *p_utf8, int p_len) {
	String ret;
	ret.parse_utf8(p_utf8, p_len);

	return ret;
}

bool String::parse_utf8(const char *p_utf8, int p_len) {
#define UNICERROR(m_err) print_error("Unicode parsing error: " + String(m_err) + ". Is the string valid UTF-8?");

	if (!p_utf8) {
		return true;
	}

	String aux;

	int cstr_size = 0;
	int str_size = 0;

	/* HANDLE BOM (Byte Order Mark) */
	if (p_len < 0 || p_len >= 3) {
		bool has_bom = uint8_t(p_utf8[0]) == 0xef && uint8_t(p_utf8[1]) == 0xbb && uint8_t(p_utf8[2]) == 0xbf;
		if (has_bom) {
			//8-bit encoding, byte order has no meaning in UTF-8, just skip it
			if (p_len >= 0) {
				p_len -= 3;
			}
			p_utf8 += 3;
		}
	}

	{
		const char *ptrtmp = p_utf8;
		const char *ptrtmp_limit = &p_utf8[p_len];
		int skip = 0;
		while (ptrtmp != ptrtmp_limit && *ptrtmp) {
			if (skip == 0) {
				uint8_t c = *ptrtmp >= 0 ? *ptrtmp : uint8_t(256 + *ptrtmp);

				/* Determine the number of characters in sequence */
				if ((c & 0x80) == 0) {
					skip = 0;
				} else if ((c & 0xe0) == 0xc0) {
					skip = 1;
				} else if ((c & 0xf0) == 0xe0) {
					skip = 2;
				} else if ((c & 0xf8) == 0xf0) {
					skip = 3;
				} else {
					UNICERROR("invalid skip at " + num_int64(cstr_size));
					return true; //invalid utf8
				}

				if (skip == 1 && (c & 0x1e) == 0) {
					UNICERROR("overlong rejected at " + num_int64(cstr_size));
					return true; //reject overlong
				}

				str_size++;

			} else {
				--skip;
			}

			cstr_size++;
			ptrtmp++;
		}

		if (skip) {
			UNICERROR("no space left");
			return true; //not enough space
		}
	}

	if (str_size == 0) {
		clear();
		return false;
	}

	resize(str_size + 1);
	char32_t *dst = ptrw();
	dst[str_size] = 0;

	while (cstr_size) {
		int len = 0;

		/* Determine the number of characters in sequence */
		if ((*p_utf8 & 0x80) == 0) {
			len = 1;
		} else if ((*p_utf8 & 0xe0) == 0xc0) {
			len = 2;
		} else if ((*p_utf8 & 0xf0) == 0xe0) {
			len = 3;
		} else if ((*p_utf8 & 0xf8) == 0xf0) {
			len = 4;
		} else {
			UNICERROR("invalid len");
			return true; //invalid UTF8
		}

		if (len > cstr_size) {
			UNICERROR("no space left");
			return true; //not enough space
		}

		if (len == 2 && (*p_utf8 & 0x1E) == 0) {
			UNICERROR("no space left");
			return true; //reject overlong
		}

		/* Convert the first character */

		uint32_t unichar = 0;

		if (len == 1) {
			unichar = *p_utf8;
		} else {
			unichar = (0xff >> (len + 1)) & *p_utf8;

			for (int i = 1; i < len; i++) {
				if ((p_utf8[i] & 0xc0) != 0x80) {
					UNICERROR("invalid utf8");
					return true; //invalid utf8
				}
				if (unichar == 0 && i == 2 && ((p_utf8[i] & 0x7f) >> (7 - len)) == 0) {
					UNICERROR("invalid utf8 overlong");
					return true; //no overlong
				}
				unichar = (unichar << 6) | (p_utf8[i] & 0x3f);
			}
		}
		if (unichar >= 0xd800 && unichar <= 0xdfff) {
			UNICERROR("invalid code point");
			return CharString();
		}

		*(dst++) = unichar;
		cstr_size -= len;
		p_utf8 += len;
	}

	return false;
#undef UNICERROR
}

CharString String::utf8() const {
	int l = length();
	if (!l) {
		return CharString();
	}

	const char32_t *d = &operator[](0);
	int fl = 0;
	for (int i = 0; i < l; i++) {
		uint32_t c = d[i];
		if (c <= 0x7f) { // 7 bits.
			fl += 1;
		} else if (c <= 0x7ff) { // 11 bits
			fl += 2;
		} else if (c <= 0xffff) { // 16 bits
			fl += 3;
		} else if (c <= 0x0010ffff) { // 21 bits
			fl += 4;
		} else {
			print_error("Unicode parsing error: Invalid unicode codepoint " + num_int64(c, 16) + ".");
			return CharString();
		}
		if (c >= 0xd800 && c <= 0xdfff) {
			print_error("Unicode parsing error: Invalid unicode codepoint " + num_int64(c, 16) + ".");
			return CharString();
		}
	}

	CharString utf8s;
	if (fl == 0) {
		return utf8s;
	}

	utf8s.resize(fl + 1);
	uint8_t *cdst = (uint8_t *)utf8s.get_data();

#define APPEND_CHAR(m_c) *(cdst++) = m_c

	for (int i = 0; i < l; i++) {
		uint32_t c = d[i];

		if (c <= 0x7f) { // 7 bits.
			APPEND_CHAR(c);
		} else if (c <= 0x7ff) { // 11 bits
			APPEND_CHAR(uint32_t(0xc0 | ((c >> 6) & 0x1f))); // Top 5 bits.
			APPEND_CHAR(uint32_t(0x80 | (c & 0x3f))); // Bottom 6 bits.
		} else if (c <= 0xffff) { // 16 bits
			APPEND_CHAR(uint32_t(0xe0 | ((c >> 12) & 0x0f))); // Top 4 bits.
			APPEND_CHAR(uint32_t(0x80 | ((c >> 6) & 0x3f))); // Middle 6 bits.
			APPEND_CHAR(uint32_t(0x80 | (c & 0x3f))); // Bottom 6 bits.
		} else { // 21 bits
			APPEND_CHAR(uint32_t(0xf0 | ((c >> 18) & 0x07))); // Top 3 bits.
			APPEND_CHAR(uint32_t(0x80 | ((c >> 12) & 0x3f))); // Upper middle 6 bits.
			APPEND_CHAR(uint32_t(0x80 | ((c >> 6) & 0x3f))); // Lower middle 6 bits.
			APPEND_CHAR(uint32_t(0x80 | (c & 0x3f))); // Bottom 6 bits.
		}
	}
#undef APPEND_CHAR
	*cdst = 0; //trailing zero

	return utf8s;
}

String String::utf16(const char16_t *p_utf16, int p_len) {
	String ret;
	ret.parse_utf16(p_utf16, p_len);

	return ret;
}

bool String::parse_utf16(const char16_t *p_utf16, int p_len) {
#define UNICERROR(m_err) print_error("Unicode parsing error: " + String(m_err) + ". Is the string valid UTF-16?");

	if (!p_utf16) {
		return true;
	}

	String aux;

	int cstr_size = 0;
	int str_size = 0;

	/* HANDLE BOM (Byte Order Mark) */
	bool byteswap = false; // assume correct endianness if no BOM found
	if (p_len < 0 || p_len >= 1) {
		bool has_bom = false;
		if (uint16_t(p_utf16[0]) == 0xfeff) { // correct BOM, read as is
			has_bom = true;
			byteswap = false;
		} else if (uint16_t(p_utf16[0]) == 0xfffe) { // backwards BOM, swap bytes
			has_bom = true;
			byteswap = true;
		}
		if (has_bom) {
			if (p_len >= 0) {
				p_len -= 1;
			}
			p_utf16 += 1;
		}
	}

	{
		const char16_t *ptrtmp = p_utf16;
		const char16_t *ptrtmp_limit = &p_utf16[p_len];
		int skip = 0;
		while (ptrtmp != ptrtmp_limit && *ptrtmp) {
			uint32_t c = (byteswap) ? BSWAP16(*ptrtmp) : *ptrtmp;
			if (skip == 0) {
				if ((c & 0xfffffc00) == 0xd800) {
					skip = 1; // lead surrogate
				} else if ((c & 0xfffffc00) == 0xdc00) {
					UNICERROR("invalid utf16 surrogate at " + num_int64(cstr_size));
					return true; // invalid UTF16
				} else {
					skip = 0;
				}
				str_size++;
			} else {
				if ((c & 0xfffffc00) == 0xdc00) { // trail surrogate
					--skip;
				} else {
					UNICERROR("invalid utf16 surrogate at " + num_int64(cstr_size));
					return true; // invalid UTF16
				}
			}

			cstr_size++;
			ptrtmp++;
		}

		if (skip) {
			UNICERROR("no space left");
			return true; // not enough space
		}
	}

	if (str_size == 0) {
		clear();
		return false;
	}

	resize(str_size + 1);
	char32_t *dst = ptrw();
	dst[str_size] = 0;

	while (cstr_size) {
		int len = 0;
		uint32_t c = (byteswap) ? BSWAP16(*p_utf16) : *p_utf16;

		if ((c & 0xfffffc00) == 0xd800) {
			len = 2;
		} else {
			len = 1;
		}

		if (len > cstr_size) {
			UNICERROR("no space left");
			return true; //not enough space
		}

		uint32_t unichar = 0;
		if (len == 1) {
			unichar = c;
		} else {
			uint32_t c2 = (byteswap) ? BSWAP16(p_utf16[1]) : p_utf16[1];
			unichar = (c << 10UL) + c2 - ((0xd800 << 10UL) + 0xdc00 - 0x10000);
		}

		*(dst++) = unichar;
		cstr_size -= len;
		p_utf16 += len;
	}

	return false;
#undef UNICERROR
}

Char16String String::utf16() const {
	int l = length();
	if (!l) {
		return Char16String();
	}

	const char32_t *d = &operator[](0);
	int fl = 0;
	for (int i = 0; i < l; i++) {
		uint32_t c = d[i];
		if (c <= 0xffff) { // 16 bits.
			fl += 1;
		} else if (c <= 0x10ffff) { // 32 bits.
			fl += 2;
		} else {
			print_error("Unicode parsing error: Invalid unicode codepoint " + num_int64(c, 16) + ".");
			return Char16String();
		}
		if (c >= 0xd800 && c <= 0xdfff) {
			print_error("Unicode parsing error: Invalid unicode codepoint " + num_int64(c, 16) + ".");
			return Char16String();
		}
	}

	Char16String utf16s;
	if (fl == 0) {
		return utf16s;
	}

	utf16s.resize(fl + 1);
	uint16_t *cdst = (uint16_t *)utf16s.get_data();

#define APPEND_CHAR(m_c) *(cdst++) = m_c

	for (int i = 0; i < l; i++) {
		uint32_t c = d[i];

		if (c <= 0xffff) { // 16 bits.
			APPEND_CHAR(c);
		} else { // 32 bits.
			APPEND_CHAR(uint32_t((c >> 10) + 0xd7c0)); // lead surrogate.
			APPEND_CHAR(uint32_t((c & 0x3ff) | 0xdc00)); // trail surrogate.
		}
	}
#undef APPEND_CHAR
	*cdst = 0; //trailing zero

	return utf16s;
}

String::String(const char *p_str) {
	copy_from(p_str);
}

String::String(const wchar_t *p_str) {
	copy_from(p_str);
}

String::String(const char32_t *p_str) {
	copy_from(p_str);
}

String::String(const char *p_str, int p_clip_to_len) {
	copy_from(p_str, p_clip_to_len);
}

String::String(const wchar_t *p_str, int p_clip_to_len) {
	copy_from(p_str, p_clip_to_len);
}

String::String(const char32_t *p_str, int p_clip_to_len) {
	copy_from(p_str, p_clip_to_len);
}

String::String(const StrRange &p_range) {
	if (!p_range.c_str) {
		return;
	}
	copy_from(p_range.c_str, p_range.len);
}

int64_t String::hex_to_int() const {
	int len = length();
	if (len == 0) {
		return 0;
	}

	const char32_t *s = ptr();

	int64_t sign = s[0] == '-' ? -1 : 1;

	if (sign < 0) {
		s++;
	}

	if (len > 2 && s[0] == '0' && lower_case(s[1]) == 'x') {
		s += 2;
	}

	int64_t hex = 0;

	while (*s) {
		char32_t c = lower_case(*s);
		int64_t n;
		if (is_digit(c)) {
			n = c - '0';
		} else if (c >= 'a' && c <= 'f') {
			n = (c - 'a') + 10;
		} else {
			ERR_FAIL_COND_V_MSG(true, 0, "Invalid hexadecimal notation character \"" + chr(*s) + "\" in string \"" + *this + "\".");
		}
		// Check for overflow/underflow, with special case to ensure INT64_MIN does not result in error
		bool overflow = ((hex > INT64_MAX / 16) && (sign == 1 || (sign == -1 && hex != (INT64_MAX >> 4) + 1))) || (sign == -1 && hex == (INT64_MAX >> 4) + 1 && c > '0');
		ERR_FAIL_COND_V_MSG(overflow, sign == 1 ? INT64_MAX : INT64_MIN, "Cannot represent " + *this + " as 64-bit integer, provided value is " + (sign == 1 ? "too big." : "too small."));
		hex *= 16;
		hex += n;
		s++;
	}

	return hex * sign;
}

int64_t String::bin_to_int() const {
	int len = length();
	if (len == 0) {
		return 0;
	}

	const char32_t *s = ptr();

	int64_t sign = s[0] == '-' ? -1 : 1;

	if (sign < 0) {
		s++;
	}

	if (len > 2 && s[0] == '0' && lower_case(s[1]) == 'b') {
		s += 2;
	}

	int64_t binary = 0;

	while (*s) {
		char32_t c = lower_case(*s);
		int64_t n;
		if (c == '0' || c == '1') {
			n = c - '0';
		} else {
			return 0;
		}
		// Check for overflow/underflow, with special case to ensure INT64_MIN does not result in error
		bool overflow = ((binary > INT64_MAX / 2) && (sign == 1 || (sign == -1 && binary != (INT64_MAX >> 1) + 1))) || (sign == -1 && binary == (INT64_MAX >> 1) + 1 && c > '0');
		ERR_FAIL_COND_V_MSG(overflow, sign == 1 ? INT64_MAX : INT64_MIN, "Cannot represent " + *this + " as 64-bit integer, provided value is " + (sign == 1 ? "too big." : "too small."));
		binary *= 2;
		binary += n;
		s++;
	}

	return binary * sign;
}

int64_t String::to_int() const {
	if (length() == 0) {
		return 0;
	}

	int to = (find(".") >= 0) ? find(".") : length();

	int64_t integer = 0;
	int64_t sign = 1;

	for (int i = 0; i < to; i++) {
		char32_t c = operator[](i);
		if (is_digit(c)) {
			bool overflow = (integer > INT64_MAX / 10) || (integer == INT64_MAX / 10 && ((sign == 1 && c > '7') || (sign == -1 && c > '8')));
			ERR_FAIL_COND_V_MSG(overflow, sign == 1 ? INT64_MAX : INT64_MIN, "Cannot represent " + *this + " as 64-bit integer, provided value is " + (sign == 1 ? "too big." : "too small."));
			integer *= 10;
			integer += c - '0';

		} else if (integer == 0 && c == '-') {
			sign = -sign;
		}
	}

	return integer * sign;
}

int64_t String::to_int(const char *p_str, int p_len) {
	int to = 0;
	if (p_len >= 0) {
		to = p_len;
	} else {
		while (p_str[to] != 0 && p_str[to] != '.') {
			to++;
		}
	}

	int64_t integer = 0;
	int64_t sign = 1;

	for (int i = 0; i < to; i++) {
		char c = p_str[i];
		if (is_digit(c)) {
			bool overflow = (integer > INT64_MAX / 10) || (integer == INT64_MAX / 10 && ((sign == 1 && c > '7') || (sign == -1 && c > '8')));
			ERR_FAIL_COND_V_MSG(overflow, sign == 1 ? INT64_MAX : INT64_MIN, "Cannot represent " + String(p_str).substr(0, to) + " as integer, provided value is " + (sign == 1 ? "too big." : "too small."));
			integer *= 10;
			integer += c - '0';

		} else if (c == '-' && integer == 0) {
			sign = -sign;
		} else if (c != ' ') {
			break;
		}
	}

	return integer * sign;
}

int64_t String::to_int(const wchar_t *p_str, int p_len) {
	int to = 0;
	if (p_len >= 0) {
		to = p_len;
	} else {
		while (p_str[to] != 0 && p_str[to] != '.') {
			to++;
		}
	}

	int64_t integer = 0;
	int64_t sign = 1;

	for (int i = 0; i < to; i++) {
		wchar_t c = p_str[i];
		if (is_digit(c)) {
			bool overflow = (integer > INT64_MAX / 10) || (integer == INT64_MAX / 10 && ((sign == 1 && c > '7') || (sign == -1 && c > '8')));
			ERR_FAIL_COND_V_MSG(overflow, sign == 1 ? INT64_MAX : INT64_MIN, "Cannot represent " + String(p_str).substr(0, to) + " as integer, provided value is " + (sign == 1 ? "too big." : "too small."));
			integer *= 10;
			integer += c - '0';

		} else if (c == '-' && integer == 0) {
			sign = -sign;
		} else if (c != ' ') {
			break;
		}
	}

	return integer * sign;
}

bool String::is_numeric() const {
	if (length() == 0) {
		return false;
	}

	int s = 0;
	if (operator[](0) == '-') {
		++s;
	}
	bool dot = false;
	for (int i = s; i < length(); i++) {
		char32_t c = operator[](i);
		if (c == '.') {
			if (dot) {
				return false;
			}
			dot = true;
		} else if (c < '0' || c > '9') {
			return false;
		}
	}

	return true; // TODO: Use the parser below for this instead
}

template <class C>
static double built_in_strtod(
		/* A decimal ASCII floating-point number,
		 * optionally preceded by white space. Must
		 * have form "-I.FE-X", where I is the integer
		 * part of the mantissa, F is the fractional
		 * part of the mantissa, and X is the
		 * exponent. Either of the signs may be "+",
		 * "-", or omitted. Either I or F may be
		 * omitted, or both. The decimal point isn't
		 * necessary unless F is present. The "E" may
		 * actually be an "e". E and X may both be
		 * omitted (but not just one). */
		const C *string,
		/* If non-nullptr, store terminating Cacter's
		 * address here. */
		C **endPtr = nullptr) {
	/* Largest possible base 10 exponent. Any
	 * exponent larger than this will already
	 * produce underflow or overflow, so there's
	 * no need to worry about additional digits. */
	static const int maxExponent = 511;
	/* Table giving binary powers of 10. Entry
	 * is 10^2^i. Used to convert decimal
	 * exponents into floating-point numbers. */
	static const double powersOf10[] = {
		10.,
		100.,
		1.0e4,
		1.0e8,
		1.0e16,
		1.0e32,
		1.0e64,
		1.0e128,
		1.0e256
	};

	bool sign, expSign = false;
	double fraction, dblExp;
	const double *d;
	const C *p;
	int c;
	/* Exponent read from "EX" field. */
	int exp = 0;
	/* Exponent that derives from the fractional
	 * part. Under normal circumstances, it is
	 * the negative of the number of digits in F.
	 * However, if I is very long, the last digits
	 * of I get dropped (otherwise a long I with a
	 * large negative exponent could cause an
	 * unnecessary overflow on I alone). In this
	 * case, fracExp is incremented one for each
	 * dropped digit. */
	int fracExp = 0;
	/* Number of digits in mantissa. */
	int mantSize;
	/* Number of mantissa digits BEFORE decimal point. */
	int decPt;
	/* Temporarily holds location of exponent in string. */
	const C *pExp;

	/*
	 * Strip off leading blanks and check for a sign.
	 */

	p = string;
	while (*p == ' ' || *p == '\t' || *p == '\n') {
		p += 1;
	}
	if (*p == '-') {
		sign = true;
		p += 1;
	} else {
		if (*p == '+') {
			p += 1;
		}
		sign = false;
	}

	/*
	 * Count the number of digits in the mantissa (including the decimal
	 * point), and also locate the decimal point.
	 */

	decPt = -1;
	for (mantSize = 0;; mantSize += 1) {
		c = *p;
		if (!is_digit(c)) {
			if ((c != '.') || (decPt >= 0)) {
				break;
			}
			decPt = mantSize;
		}
		p += 1;
	}

	/*
	 * Now suck up the digits in the mantissa. Use two integers to collect 9
	 * digits each (this is faster than using floating-point). If the mantissa
	 * has more than 18 digits, ignore the extras, since they can't affect the
	 * value anyway.
	 */

	pExp = p;
	p -= mantSize;
	if (decPt < 0) {
		decPt = mantSize;
	} else {
		mantSize -= 1; /* One of the digits was the point. */
	}
	if (mantSize > 18) {
		fracExp = decPt - 18;
		mantSize = 18;
	} else {
		fracExp = decPt - mantSize;
	}
	if (mantSize == 0) {
		fraction = 0.0;
		p = string;
		goto done;
	} else {
		int frac1, frac2;

		frac1 = 0;
		for (; mantSize > 9; mantSize -= 1) {
			c = *p;
			p += 1;
			if (c == '.') {
				c = *p;
				p += 1;
			}
			frac1 = 10 * frac1 + (c - '0');
		}
		frac2 = 0;
		for (; mantSize > 0; mantSize -= 1) {
			c = *p;
			p += 1;
			if (c == '.') {
				c = *p;
				p += 1;
			}
			frac2 = 10 * frac2 + (c - '0');
		}
		fraction = (1.0e9 * frac1) + frac2;
	}

	/*
	 * Skim off the exponent.
	 */

	p = pExp;
	if ((*p == 'E') || (*p == 'e')) {
		p += 1;
		if (*p == '-') {
			expSign = true;
			p += 1;
		} else {
			if (*p == '+') {
				p += 1;
			}
			expSign = false;
		}
		if (!is_digit(char32_t(*p))) {
			p = pExp;
			goto done;
		}
		while (is_digit(char32_t(*p))) {
			exp = exp * 10 + (*p - '0');
			p += 1;
		}
	}
	if (expSign) {
		exp = fracExp - exp;
	} else {
		exp = fracExp + exp;
	}

	/*
	 * Generate a floating-point number that represents the exponent. Do this
	 * by processing the exponent one bit at a time to combine many powers of
	 * 2 of 10. Then combine the exponent with the fraction.
	 */

	if (exp < 0) {
		expSign = true;
		exp = -exp;
	} else {
		expSign = false;
	}

	if (exp > maxExponent) {
		exp = maxExponent;
		WARN_PRINT("Exponent too high");
	}
	dblExp = 1.0;
	for (d = powersOf10; exp != 0; exp >>= 1, ++d) {
		if (exp & 01) {
			dblExp *= *d;
		}
	}
	if (expSign) {
		fraction /= dblExp;
	} else {
		fraction *= dblExp;
	}

done:
	if (endPtr != nullptr) {
		*endPtr = (C *)p;
	}

	if (sign) {
		return -fraction;
	}
	return fraction;
}

#define READING_SIGN 0
#define READING_INT 1
#define READING_DEC 2
#define READING_EXP 3
#define READING_DONE 4

double String::to_float(const char *p_str) {
	return built_in_strtod<char>(p_str);
}

double String::to_float(const char32_t *p_str, const char32_t **r_end) {
	return built_in_strtod<char32_t>(p_str, (char32_t **)r_end);
}

double String::to_float(const wchar_t *p_str, const wchar_t **r_end) {
	return built_in_strtod<wchar_t>(p_str, (wchar_t **)r_end);
}

int64_t String::to_int(const char32_t *p_str, int p_len, bool p_clamp) {
	if (p_len == 0 || !p_str[0]) {
		return 0;
	}
	///@todo make more exact so saving and loading does not lose precision

	int64_t integer = 0;
	int64_t sign = 1;
	int reading = READING_SIGN;

	const char32_t *str = p_str;
	const char32_t *limit = &p_str[p_len];

	while (*str && reading != READING_DONE && str != limit) {
		char32_t c = *(str++);
		switch (reading) {
			case READING_SIGN: {
				if (is_digit(c)) {
					reading = READING_INT;
					// let it fallthrough
				} else if (c == '-') {
					sign = -1;
					reading = READING_INT;
					break;
				} else if (c == '+') {
					sign = 1;
					reading = READING_INT;
					break;
				} else {
					break;
				}
				[[fallthrough]];
			}
			case READING_INT: {
				if (is_digit(c)) {
					if (integer > INT64_MAX / 10) {
						String number("");
						str = p_str;
						while (*str && str != limit) {
							number += *(str++);
						}
						if (p_clamp) {
							if (sign == 1) {
								return INT64_MAX;
							} else {
								return INT64_MIN;
							}
						} else {
							ERR_FAIL_V_MSG(sign == 1 ? INT64_MAX : INT64_MIN, "Cannot represent " + number + " as integer, provided value is " + (sign == 1 ? "too big." : "too small."));
						}
					}
					integer *= 10;
					integer += c - '0';
				} else {
					reading = READING_DONE;
				}

			} break;
		}
	}

	return sign * integer;
}

double String::to_float() const {
	if (is_empty()) {
		return 0;
	}
	return built_in_strtod<char32_t>(get_data());
}

uint32_t String::hash(const char *p_cstr) {
	uint32_t hashv = 5381;
	uint32_t c;

	while ((c = *p_cstr++)) {
		hashv = ((hashv << 5) + hashv) + c; /* hash * 33 + c */
	}

	return hashv;
}

uint32_t String::hash(const char *p_cstr, int p_len) {
	uint32_t hashv = 5381;
	for (int i = 0; i < p_len; i++) {
		hashv = ((hashv << 5) + hashv) + p_cstr[i]; /* hash * 33 + c */
	}

	return hashv;
}

uint32_t String::hash(const wchar_t *p_cstr, int p_len) {
	uint32_t hashv = 5381;
	for (int i = 0; i < p_len; i++) {
		hashv = ((hashv << 5) + hashv) + p_cstr[i]; /* hash * 33 + c */
	}

	return hashv;
}

uint32_t String::hash(const wchar_t *p_cstr) {
	uint32_t hashv = 5381;
	uint32_t c;

	while ((c = *p_cstr++)) {
		hashv = ((hashv << 5) + hashv) + c; /* hash * 33 + c */
	}

	return hashv;
}

uint32_t String::hash(const char32_t *p_cstr, int p_len) {
	uint32_t hashv = 5381;
	for (int i = 0; i < p_len; i++) {
		hashv = ((hashv << 5) + hashv) + p_cstr[i]; /* hash * 33 + c */
	}

	return hashv;
}

uint32_t String::hash(const char32_t *p_cstr) {
	uint32_t hashv = 5381;
	uint32_t c;

	while ((c = *p_cstr++)) {
		hashv = ((hashv << 5) + hashv) + c; /* hash * 33 + c */
	}

	return hashv;
}

uint32_t String::hash() const {
	/* simple djb2 hashing */

	const char32_t *chr = get_data();
	uint32_t hashv = 5381;
	uint32_t c;

	while ((c = *chr++)) {
		hashv = ((hashv << 5) + hashv) + c; /* hash * 33 + c */
	}

	return hashv;
}

uint64_t String::hash64() const {
	/* simple djb2 hashing */

	const char32_t *chr = get_data();
	uint64_t hashv = 5381;
	uint64_t c;

	while ((c = *chr++)) {
		hashv = ((hashv << 5) + hashv) + c; /* hash * 33 + c */
	}

	return hashv;
}

String String::md5_text() const {
	CharString cs = utf8();
	unsigned char hash[16];
	CryptoCore::md5((unsigned char *)cs.ptr(), cs.length(), hash);
	return String::hex_encode_buffer(hash, 16);
}

String String::sha1_text() const {
	CharString cs = utf8();
	unsigned char hash[20];
	CryptoCore::sha1((unsigned char *)cs.ptr(), cs.length(), hash);
	return String::hex_encode_buffer(hash, 20);
}

String String::sha256_text() const {
	CharString cs = utf8();
	unsigned char hash[32];
	CryptoCore::sha256((unsigned char *)cs.ptr(), cs.length(), hash);
	return String::hex_encode_buffer(hash, 32);
}

Vector<uint8_t> String::md5_buffer() const {
	CharString cs = utf8();
	unsigned char hash[16];
	CryptoCore::md5((unsigned char *)cs.ptr(), cs.length(), hash);

	Vector<uint8_t> ret;
	ret.resize(16);
	for (int i = 0; i < 16; i++) {
		ret.write[i] = hash[i];
	}
	return ret;
}

Vector<uint8_t> String::sha1_buffer() const {
	CharString cs = utf8();
	unsigned char hash[20];
	CryptoCore::sha1((unsigned char *)cs.ptr(), cs.length(), hash);

	Vector<uint8_t> ret;
	ret.resize(20);
	for (int i = 0; i < 20; i++) {
		ret.write[i] = hash[i];
	}

	return ret;
}

Vector<uint8_t> String::sha256_buffer() const {
	CharString cs = utf8();
	unsigned char hash[32];
	CryptoCore::sha256((unsigned char *)cs.ptr(), cs.length(), hash);

	Vector<uint8_t> ret;
	ret.resize(32);
	for (int i = 0; i < 32; i++) {
		ret.write[i] = hash[i];
	}
	return ret;
}

String String::insert(int p_at_pos, const String &p_string) const {
	if (p_at_pos < 0) {
		return *this;
	}

	if (p_at_pos > length()) {
		p_at_pos = length();
	}

	String pre;
	if (p_at_pos > 0) {
		pre = substr(0, p_at_pos);
	}

	String post;
	if (p_at_pos < length()) {
		post = substr(p_at_pos, length() - p_at_pos);
	}

	return pre + p_string + post;
}

String String::substr(int p_from, int p_chars) const {
	if (p_chars == -1) {
		p_chars = length() - p_from;
	}

	if (is_empty() || p_from < 0 || p_from >= length() || p_chars <= 0) {
		return "";
	}

	if ((p_from + p_chars) > length()) {
		p_chars = length() - p_from;
	}

	if (p_from == 0 && p_chars >= length()) {
		return String(*this);
	}

	String s = String();
	s.copy_from_unchecked(&get_data()[p_from], p_chars);
	return s;
}

int String::find(const String &p_str, int p_from) const {
	if (p_from < 0) {
		return -1;
	}

	const int src_len = p_str.length();

	const int len = length();

	if (src_len == 0 || len == 0) {
		return -1; // won't find anything!
	}

	const char32_t *src = get_data();
	const char32_t *str = p_str.get_data();

	for (int i = p_from; i <= (len - src_len); i++) {
		bool found = true;
		for (int j = 0; j < src_len; j++) {
			int read_pos = i + j;

			if (read_pos >= len) {
				ERR_PRINT("read_pos>=len");
				return -1;
			}

			if (src[read_pos] != str[j]) {
				found = false;
				break;
			}
		}

		if (found) {
			return i;
		}
	}

	return -1;
}

int String::find(const char *p_str, int p_from) const {
	if (p_from < 0) {
		return -1;
	}

	const int len = length();

	if (len == 0) {
		return -1; // won't find anything!
	}

	const char32_t *src = get_data();

	int src_len = 0;
	while (p_str[src_len] != '\0') {
		src_len++;
	}

	if (src_len == 1) {
		const char32_t needle = p_str[0];

		for (int i = p_from; i < len; i++) {
			if (src[i] == needle) {
				return i;
			}
		}

	} else {
		for (int i = p_from; i <= (len - src_len); i++) {
			bool found = true;
			for (int j = 0; j < src_len; j++) {
				int read_pos = i + j;

				if (read_pos >= len) {
					ERR_PRINT("read_pos>=len");
					return -1;
				}

				if (src[read_pos] != (char32_t)p_str[j]) {
					found = false;
					break;
				}
			}

			if (found) {
				return i;
			}
		}
	}

	return -1;
}

int String::find_char(const char32_t &p_char, int p_from) const {
	return _cowdata.find(p_char, p_from);
}

int String::findmk(const Vector<String> &p_keys, int p_from, int *r_key) const {
	if (p_from < 0) {
		return -1;
	}
	if (p_keys.size() == 0) {
		return -1;
	}

	//int src_len=p_str.length();
	const String *keys = &p_keys[0];
	int key_count = p_keys.size();
	int len = length();

	if (len == 0) {
		return -1; // won't find anything!
	}

	const char32_t *src = get_data();

	for (int i = p_from; i < len; i++) {
		bool found = true;
		for (int k = 0; k < key_count; k++) {
			found = true;
			if (r_key) {
				*r_key = k;
			}
			const char32_t *cmp = keys[k].get_data();
			int l = keys[k].length();

			for (int j = 0; j < l; j++) {
				int read_pos = i + j;

				if (read_pos >= len) {
					found = false;
					break;
				}

				if (src[read_pos] != cmp[j]) {
					found = false;
					break;
				}
			}
			if (found) {
				break;
			}
		}

		if (found) {
			return i;
		}
	}

	return -1;
}

int String::findn(const String &p_str, int p_from) const {
	if (p_from < 0) {
		return -1;
	}

	int src_len = p_str.length();

	if (src_len == 0 || length() == 0) {
		return -1; // won't find anything!
	}

	const char32_t *srcd = get_data();

	for (int i = p_from; i <= (length() - src_len); i++) {
		bool found = true;
		for (int j = 0; j < src_len; j++) {
			int read_pos = i + j;

			if (read_pos >= length()) {
				ERR_PRINT("read_pos>=length()");
				return -1;
			}

			char32_t src = _find_lower(srcd[read_pos]);
			char32_t dst = _find_lower(p_str[j]);

			if (src != dst) {
				found = false;
				break;
			}
		}

		if (found) {
			return i;
		}
	}

	return -1;
}

int String::rfind(const String &p_str, int p_from) const {
	// establish a limit
	int limit = length() - p_str.length();
	if (limit < 0) {
		return -1;
	}

	// establish a starting point
	if (p_from < 0) {
		p_from = limit;
	} else if (p_from > limit) {
		p_from = limit;
	}

	int src_len = p_str.length();
	int len = length();

	if (src_len == 0 || len == 0) {
		return -1; // won't find anything!
	}

	const char32_t *src = get_data();

	for (int i = p_from; i >= 0; i--) {
		bool found = true;
		for (int j = 0; j < src_len; j++) {
			int read_pos = i + j;

			if (read_pos >= len) {
				ERR_PRINT("read_pos>=len");
				return -1;
			}

			if (src[read_pos] != p_str[j]) {
				found = false;
				break;
			}
		}

		if (found) {
			return i;
		}
	}

	return -1;
}

int String::rfindn(const String &p_str, int p_from) const {
	// establish a limit
	int limit = length() - p_str.length();
	if (limit < 0) {
		return -1;
	}

	// establish a starting point
	if (p_from < 0) {
		p_from = limit;
	} else if (p_from > limit) {
		p_from = limit;
	}

	int src_len = p_str.length();
	int len = length();

	if (src_len == 0 || len == 0) {
		return -1; // won't find anything!
	}

	const char32_t *src = get_data();

	for (int i = p_from; i >= 0; i--) {
		bool found = true;
		for (int j = 0; j < src_len; j++) {
			int read_pos = i + j;

			if (read_pos >= len) {
				ERR_PRINT("read_pos>=len");
				return -1;
			}

			char32_t srcc = _find_lower(src[read_pos]);
			char32_t dstc = _find_lower(p_str[j]);

			if (srcc != dstc) {
				found = false;
				break;
			}
		}

		if (found) {
			return i;
		}
	}

	return -1;
}

bool String::ends_with(const String &p_string) const {
	int l = p_string.length();
	if (l > length()) {
		return false;
	}

	if (l == 0) {
		return true;
	}

	const char32_t *p = &p_string[0];
	const char32_t *s = &operator[](length() - l);

	for (int i = 0; i < l; i++) {
		if (p[i] != s[i]) {
			return false;
		}
	}

	return true;
}

bool String::begins_with(const String &p_string) const {
	int l = p_string.length();
	if (l > length()) {
		return false;
	}

	if (l == 0) {
		return true;
	}

	const char32_t *p = &p_string[0];
	const char32_t *s = &operator[](0);

	for (int i = 0; i < l; i++) {
		if (p[i] != s[i]) {
			return false;
		}
	}

	return true;
}

bool String::begins_with(const char *p_string) const {
	int l = length();
	if (l == 0 || !p_string) {
		return false;
	}

	const char32_t *str = &operator[](0);
	int i = 0;

	while (*p_string && i < l) {
		if ((char32_t)*p_string != str[i]) {
			return false;
		}
		i++;
		p_string++;
	}

	return *p_string == 0;
}

bool String::is_enclosed_in(const String &p_string) const {
	return begins_with(p_string) && ends_with(p_string);
}

bool String::is_subsequence_of(const String &p_string) const {
	return _base_is_subsequence_of(p_string, false);
}

bool String::is_subsequence_ofn(const String &p_string) const {
	return _base_is_subsequence_of(p_string, true);
}

bool String::is_quoted() const {
	return is_enclosed_in("\"") || is_enclosed_in("'");
}

int String::_count(const String &p_string, int p_from, int p_to, bool p_case_insensitive) const {
	if (p_string.is_empty()) {
		return 0;
	}
	int len = length();
	int slen = p_string.length();
	if (len < slen) {
		return 0;
	}
	String str;
	if (p_from >= 0 && p_to >= 0) {
		if (p_to == 0) {
			p_to = len;
		} else if (p_from >= p_to) {
			return 0;
		}
		if (p_from == 0 && p_to == len) {
			str = String();
			str.copy_from_unchecked(&get_data()[0], len);
		} else {
			str = substr(p_from, p_to - p_from);
		}
	} else {
		return 0;
	}
	int c = 0;
	int idx = -1;
	do {
		idx = p_case_insensitive ? str.findn(p_string) : str.find(p_string);
		if (idx != -1) {
			str = str.substr(idx + slen, str.length() - slen);
			++c;
		}
	} while (idx != -1);
	return c;
}

int String::count(const String &p_string, int p_from, int p_to) const {
	return _count(p_string, p_from, p_to, false);
}

int String::countn(const String &p_string, int p_from, int p_to) const {
	return _count(p_string, p_from, p_to, true);
}

bool String::_base_is_subsequence_of(const String &p_string, bool case_insensitive) const {
	int len = length();
	if (len == 0) {
		// Technically an empty string is subsequence of any string
		return true;
	}

	if (len > p_string.length()) {
		return false;
	}

	const char32_t *src = &operator[](0);
	const char32_t *tgt = &p_string[0];

	for (; *src && *tgt; tgt++) {
		bool match = false;
		if (case_insensitive) {
			char32_t srcc = _find_lower(*src);
			char32_t tgtc = _find_lower(*tgt);
			match = srcc == tgtc;
		} else {
			match = *src == *tgt;
		}
		if (match) {
			src++;
			if (!*src) {
				return true;
			}
		}
	}

	return false;
}

Vector<String> String::bigrams() const {
	int n_pairs = length() - 1;
	Vector<String> b;
	if (n_pairs <= 0) {
		return b;
	}
	b.resize(n_pairs);
	for (int i = 0; i < n_pairs; i++) {
		b.write[i] = substr(i, 2);
	}
	return b;
}

// Similarity according to Sorensen-Dice coefficient
float String::similarity(const String &p_string) const {
	if (operator==(p_string)) {
		// Equal strings are totally similar
		return 1.0f;
	}
	if (length() < 2 || p_string.length() < 2) {
		// No way to calculate similarity without a single bigram
		return 0.0f;
	}

	Vector<String> src_bigrams = bigrams();
	Vector<String> tgt_bigrams = p_string.bigrams();

	int src_size = src_bigrams.size();
	int tgt_size = tgt_bigrams.size();

	int sum = src_size + tgt_size;
	int inter = 0;
	for (int i = 0; i < src_size; i++) {
		for (int j = 0; j < tgt_size; j++) {
			if (src_bigrams[i] == tgt_bigrams[j]) {
				inter++;
				break;
			}
		}
	}

	return (2.0f * inter) / sum;
}

static bool _wildcard_match(const char32_t *p_pattern, const char32_t *p_string, bool p_case_sensitive) {
	switch (*p_pattern) {
		case '\0':
			return !*p_string;
		case '*':
			return _wildcard_match(p_pattern + 1, p_string, p_case_sensitive) || (*p_string && _wildcard_match(p_pattern, p_string + 1, p_case_sensitive));
		case '?':
			return *p_string && (*p_string != '.') && _wildcard_match(p_pattern + 1, p_string + 1, p_case_sensitive);
		default:

			return (p_case_sensitive ? (*p_string == *p_pattern) : (_find_upper(*p_string) == _find_upper(*p_pattern))) && _wildcard_match(p_pattern + 1, p_string + 1, p_case_sensitive);
	}
}

bool String::match(const String &p_wildcard) const {
	if (!p_wildcard.length() || !length()) {
		return false;
	}

	return _wildcard_match(p_wildcard.get_data(), get_data(), true);
}

bool String::matchn(const String &p_wildcard) const {
	if (!p_wildcard.length() || !length()) {
		return false;
	}
	return _wildcard_match(p_wildcard.get_data(), get_data(), false);
}

String String::format(const Variant &values, String placeholder) const {
	String new_string = String(this->ptr());

	if (values.get_type() == Variant::ARRAY) {
		Array values_arr = values;

		for (int i = 0; i < values_arr.size(); i++) {
			String i_as_str = String::num_int64(i);

			if (values_arr[i].get_type() == Variant::ARRAY) { //Array in Array structure [["name","RobotGuy"],[0,"godot"],["strength",9000.91]]
				Array value_arr = values_arr[i];

				if (value_arr.size() == 2) {
					Variant v_key = value_arr[0];
					String key = v_key;

					Variant v_val = value_arr[1];
					String val = v_val;

					new_string = new_string.replace(placeholder.replace("_", key), val);
				} else {
					ERR_PRINT(String("STRING.format Inner Array size != 2 ").ascii().get_data());
				}
			} else { //Array structure ["RobotGuy","Logis","rookie"]
				Variant v_val = values_arr[i];
				String val = v_val;

				if (placeholder.find("_") > -1) {
					new_string = new_string.replace(placeholder.replace("_", i_as_str), val);
				} else {
					new_string = new_string.replace_first(placeholder, val);
				}
			}
		}
	} else if (values.get_type() == Variant::DICTIONARY) {
		Dictionary d = values;
		List<Variant> keys;
		d.get_key_list(&keys);

		for (const Variant &key : keys) {
			new_string = new_string.replace(placeholder.replace("_", key), d[key]);
		}
	} else {
		ERR_PRINT(String("Invalid type: use Array or Dictionary.").ascii().get_data());
	}

	return new_string;
}

String String::replace(const String &p_key, const String &p_with) const {
	String new_string;
	int search_from = 0;
	int result = 0;

	while ((result = find(p_key, search_from)) >= 0) {
		new_string += substr(search_from, result - search_from);
		new_string += p_with;
		search_from = result + p_key.length();
	}

	if (search_from == 0) {
		return *this;
	}

	new_string += substr(search_from, length() - search_from);

	return new_string;
}

String String::replace(const char *p_key, const char *p_with) const {
	String new_string;
	int search_from = 0;
	int result = 0;

	while ((result = find(p_key, search_from)) >= 0) {
		new_string += substr(search_from, result - search_from);
		new_string += p_with;
		int k = 0;
		while (p_key[k] != '\0') {
			k++;
		}
		search_from = result + k;
	}

	if (search_from == 0) {
		return *this;
	}

	new_string += substr(search_from, length() - search_from);

	return new_string;
}

String String::replace_first(const String &p_key, const String &p_with) const {
	int pos = find(p_key);
	if (pos >= 0) {
		return substr(0, pos) + p_with + substr(pos + p_key.length(), length());
	}

	return *this;
}

String String::replacen(const String &p_key, const String &p_with) const {
	String new_string;
	int search_from = 0;
	int result = 0;

	while ((result = findn(p_key, search_from)) >= 0) {
		new_string += substr(search_from, result - search_from);
		new_string += p_with;
		search_from = result + p_key.length();
	}

	if (search_from == 0) {
		return *this;
	}

	new_string += substr(search_from, length() - search_from);
	return new_string;
}

String String::repeat(int p_count) const {
	ERR_FAIL_COND_V_MSG(p_count < 0, "", "Parameter count should be a positive number.");

	String new_string;
	const char32_t *src = this->get_data();

	new_string.resize(length() * p_count + 1);
	new_string[length() * p_count] = 0;

	for (int i = 0; i < p_count; i++) {
		for (int j = 0; j < length(); j++) {
			new_string[i * length() + j] = src[j];
		}
	}

	return new_string;
}

String String::left(int p_pos) const {
	if (p_pos < 0) {
		p_pos = length() + p_pos;
	}

	if (p_pos <= 0) {
		return "";
	}

	if (p_pos >= length()) {
		return *this;
	}

	return substr(0, p_pos);
}

String String::right(int p_pos) const {
	if (p_pos < 0) {
		p_pos = length() + p_pos;
	}

	if (p_pos <= 0) {
		return "";
	}

	if (p_pos >= length()) {
		return *this;
	}

	return substr(length() - p_pos);
}

char32_t String::unicode_at(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, length(), 0);
	return operator[](p_idx);
}

String String::indent(const String &p_prefix) const {
	String new_string;
	int line_start = 0;

	for (int i = 0; i < length(); i++) {
		const char32_t c = operator[](i);
		if (c == '\n') {
			if (i == line_start) {
				new_string += c; // Leave empty lines empty.
			} else {
				new_string += p_prefix + substr(line_start, i - line_start + 1);
			}
			line_start = i + 1;
		}
	}
	if (line_start != length()) {
		new_string += p_prefix + substr(line_start);
	}
	return new_string;
}

String String::dedent() const {
	String new_string;
	String indent;
	bool has_indent = false;
	bool has_text = false;
	int line_start = 0;
	int indent_stop = -1;

	for (int i = 0; i < length(); i++) {
		char32_t c = operator[](i);
		if (c == '\n') {
			if (has_text) {
				new_string += substr(indent_stop, i - indent_stop);
			}
			new_string += "\n";
			has_text = false;
			line_start = i + 1;
			indent_stop = -1;
		} else if (!has_text) {
			if (c > 32) {
				has_text = true;
				if (!has_indent) {
					has_indent = true;
					indent = substr(line_start, i - line_start);
					indent_stop = i;
				}
			}
			if (has_indent && indent_stop < 0) {
				int j = i - line_start;
				if (j >= indent.length() || c != indent[j]) {
					indent_stop = i;
				}
			}
		}
	}

	if (has_text) {
		new_string += substr(indent_stop, length() - indent_stop);
	}

	return new_string;
}

String String::strip_edges(bool left, bool right) const {
	int len = length();
	int beg = 0, end = len;

	if (left) {
		for (int i = 0; i < len; i++) {
			if (operator[](i) <= 32) {
				beg++;
			} else {
				break;
			}
		}
	}

	if (right) {
		for (int i = len - 1; i >= 0; i--) {
			if (operator[](i) <= 32) {
				end--;
			} else {
				break;
			}
		}
	}

	if (beg == 0 && end == len) {
		return *this;
	}

	return substr(beg, end - beg);
}

String String::strip_escapes() const {
	String new_string;
	for (int i = 0; i < length(); i++) {
		// Escape characters on first page of the ASCII table, before 32 (Space).
		if (operator[](i) < 32) {
			continue;
		}
		new_string += operator[](i);
	}

	return new_string;
}

String String::lstrip(const String &p_chars) const {
	int len = length();
	int beg;

	for (beg = 0; beg < len; beg++) {
		if (p_chars.find_char(get(beg)) == -1) {
			break;
		}
	}

	if (beg == 0) {
		return *this;
	}

	return substr(beg, len - beg);
}

String String::rstrip(const String &p_chars) const {
	int len = length();
	int end;

	for (end = len - 1; end >= 0; end--) {
		if (p_chars.find_char(get(end)) == -1) {
			break;
		}
	}

	if (end == len - 1) {
		return *this;
	}

	return substr(0, end + 1);
}

String String::simplify_path() const {
	String s = *this;
	String drive;
	if (s.begins_with("local://")) {
		drive = "local://";
		s = s.substr(8, s.length());
	} else if (s.begins_with("res://")) {
		drive = "res://";
		s = s.substr(6, s.length());
	} else if (s.begins_with("user://")) {
		drive = "user://";
		s = s.substr(7, s.length());
	} else if (s.begins_with("/") || s.begins_with("\\")) {
		drive = s.substr(0, 1);
		s = s.substr(1, s.length() - 1);
	} else {
		int p = s.find(":/");
		if (p == -1) {
			p = s.find(":\\");
		}
		if (p != -1 && p < s.find("/")) {
			drive = s.substr(0, p + 2);
			s = s.substr(p + 2, s.length());
		}
	}

	s = s.replace("\\", "/");
	while (true) { // in case of using 2 or more slash
		String compare = s.replace("//", "/");
		if (s == compare) {
			break;
		} else {
			s = compare;
		}
	}
	Vector<String> dirs = s.split("/", false);

	for (int i = 0; i < dirs.size(); i++) {
		String d = dirs[i];
		if (d == ".") {
			dirs.remove_at(i);
			i--;
		} else if (d == "..") {
			if (i == 0) {
				dirs.remove_at(i);
				i--;
			} else {
				dirs.remove_at(i);
				dirs.remove_at(i - 1);
				i -= 2;
			}
		}
	}

	s = "";

	for (int i = 0; i < dirs.size(); i++) {
		if (i > 0) {
			s += "/";
		}
		s += dirs[i];
	}

	return drive + s;
}

static int _humanize_digits(int p_num) {
	if (p_num < 100) {
		return 2;
	} else if (p_num < 1024) {
		return 1;
	} else {
		return 0;
	}
}

String String::humanize_size(uint64_t p_size) {
	uint64_t _div = 1;
	Vector<String> prefixes;
	prefixes.push_back(RTR("B"));
	prefixes.push_back(RTR("KiB"));
	prefixes.push_back(RTR("MiB"));
	prefixes.push_back(RTR("GiB"));
	prefixes.push_back(RTR("TiB"));
	prefixes.push_back(RTR("PiB"));
	prefixes.push_back(RTR("EiB"));

	int prefix_idx = 0;

	while (prefix_idx < prefixes.size() - 1 && p_size > (_div * 1024)) {
		_div *= 1024;
		prefix_idx++;
	}

	const int digits = prefix_idx > 0 ? _humanize_digits(p_size / _div) : 0;
	const double divisor = prefix_idx > 0 ? _div : 1;

	return String::num(p_size / divisor).pad_decimals(digits) + " " + prefixes[prefix_idx];
}

bool String::is_absolute_path() const {
	if (length() > 1) {
		return (operator[](0) == '/' || operator[](0) == '\\' || find(":/") != -1 || find(":\\") != -1);
	} else if ((length()) == 1) {
		return (operator[](0) == '/' || operator[](0) == '\\');
	} else {
		return false;
	}
}

bool String::is_valid_identifier() const {
	int len = length();

	if (len == 0) {
		return false;
	}

	const char32_t *str = &operator[](0);

	for (int i = 0; i < len; i++) {
		if (i == 0) {
			if (is_digit(str[0])) {
				return false; // no start with number plz
			}
		}

		bool valid_char = is_digit(str[i]) || is_lower_case(str[i]) || is_upper_case(str[i]) || str[i] == '_';

		if (!valid_char) {
			return false;
		}
	}

	return true;
}

bool String::is_valid_string() const {
	int l = length();
	const char32_t *src = get_data();
	bool valid = true;
	for (int i = 0; i < l; i++) {
		valid = valid && (src[i] < 0xd800 || (src[i] > 0xdfff && src[i] <= 0x10ffff));
	}
	return valid;
}

String String::uri_encode() const {
	const CharString temp = utf8();
	String res;
	for (int i = 0; i < temp.length(); ++i) {
		char ord = temp[i];
		if (ord == '.' || ord == '-' || ord == '_' || ord == '~' || is_lower_case(ord) || is_upper_case(ord) || is_digit(ord)) {
			res += ord;
		} else {
			char h_Val[3];
#if defined(__GNUC__) || defined(_MSC_VER)
			snprintf(h_Val, 3, "%02hhX", ord);
#else
			sprintf(h_Val, "%02hhX", ord);
#endif
			res += "%";
			res += h_Val;
		}
	}
	return res;
}

String String::uri_decode() const {
	CharString src = utf8();
	CharString res;
	for (int i = 0; i < src.length(); ++i) {
		if (src[i] == '%' && i + 2 < src.length()) {
			char ord1 = src[i + 1];
			if (is_digit(ord1) || is_upper_case(ord1)) {
				char ord2 = src[i + 2];
				if (is_digit(ord2) || is_upper_case(ord2)) {
					char bytes[3] = { (char)ord1, (char)ord2, 0 };
					res += (char)strtol(bytes, nullptr, 16);
					i += 2;
				}
			} else {
				res += src[i];
			}
		} else if (src[i] == '+') {
			res += ' ';
		} else {
			res += src[i];
		}
	}
	return String::utf8(res);
}

String String::c_unescape() const {
	String escaped = *this;
	escaped = escaped.replace("\\a", "\a");
	escaped = escaped.replace("\\b", "\b");
	escaped = escaped.replace("\\f", "\f");
	escaped = escaped.replace("\\n", "\n");
	escaped = escaped.replace("\\r", "\r");
	escaped = escaped.replace("\\t", "\t");
	escaped = escaped.replace("\\v", "\v");
	escaped = escaped.replace("\\'", "\'");
	escaped = escaped.replace("\\\"", "\"");
	escaped = escaped.replace("\\?", "\?");
	escaped = escaped.replace("\\\\", "\\");

	return escaped;
}

String String::c_escape() const {
	String escaped = *this;
	escaped = escaped.replace("\\", "\\\\");
	escaped = escaped.replace("\a", "\\a");
	escaped = escaped.replace("\b", "\\b");
	escaped = escaped.replace("\f", "\\f");
	escaped = escaped.replace("\n", "\\n");
	escaped = escaped.replace("\r", "\\r");
	escaped = escaped.replace("\t", "\\t");
	escaped = escaped.replace("\v", "\\v");
	escaped = escaped.replace("\'", "\\'");
	escaped = escaped.replace("\?", "\\?");
	escaped = escaped.replace("\"", "\\\"");

	return escaped;
}

String String::c_escape_multiline() const {
	String escaped = *this;
	escaped = escaped.replace("\\", "\\\\");
	escaped = escaped.replace("\"", "\\\"");

	return escaped;
}

String String::json_escape() const {
	String escaped = *this;
	escaped = escaped.replace("\\", "\\\\");
	escaped = escaped.replace("\b", "\\b");
	escaped = escaped.replace("\f", "\\f");
	escaped = escaped.replace("\n", "\\n");
	escaped = escaped.replace("\r", "\\r");
	escaped = escaped.replace("\t", "\\t");
	escaped = escaped.replace("\v", "\\v");
	escaped = escaped.replace("\"", "\\\"");

	return escaped;
}

String String::xml_escape(bool p_escape_quotes) const {
	String str = *this;
	str = str.replace("&", "&amp;");
	str = str.replace("<", "&lt;");
	str = str.replace(">", "&gt;");
	if (p_escape_quotes) {
		str = str.replace("'", "&apos;");
		str = str.replace("\"", "&quot;");
	}
	/*
for (int i=1;i<32;i++) {
	char chr[2]={i,0};
	str=str.replace(chr,"&#"+String::num(i)+";");
}*/
	return str;
}

static _FORCE_INLINE_ int _xml_unescape(const char32_t *p_src, int p_src_len, char32_t *p_dst) {
	int len = 0;
	while (p_src_len) {
		if (*p_src == '&') {
			int eat = 0;

			if (p_src_len >= 4 && p_src[1] == '#') {
				char32_t c = 0;
				bool overflow = false;
				if (p_src[2] == 'x') {
					// Hex entity &#x<num>;
					for (int i = 3; i < p_src_len; i++) {
						eat = i + 1;
						char32_t ct = p_src[i];
						if (ct == ';') {
							break;
						} else if (is_digit(ct)) {
							ct = ct - '0';
						} else if (ct >= 'a' && ct <= 'f') {
							ct = (ct - 'a') + 10;
						} else if (ct >= 'A' && ct <= 'F') {
							ct = (ct - 'A') + 10;
						} else {
							break;
						}
						if (c > (UINT32_MAX >> 4)) {
							overflow = true;
							break;
						}
						c <<= 4;
						c |= ct;
					}
				} else {
					// Decimal entity &#<num>;
					for (int i = 2; i < p_src_len; i++) {
						eat = i + 1;
						char32_t ct = p_src[i];
						if (ct == ';' || ct < '0' || ct > '9') {
							break;
						}
					}
					if (p_src[eat - 1] == ';') {
						int64_t val = String::to_int(p_src + 2, eat - 3);
						if (val > 0 && val <= UINT32_MAX) {
							c = (char32_t)val;
						} else {
							overflow = true;
						}
					}
				}

				// Value must be non-zero, in the range of char32_t,
				// actually end with ';'. If invalid, leave the entity as-is
				if (c == '\0' || overflow || p_src[eat - 1] != ';') {
					eat = 1;
					c = *p_src;
				}
				if (p_dst) {
					*p_dst = c;
				}

			} else if (p_src_len >= 4 && p_src[1] == 'g' && p_src[2] == 't' && p_src[3] == ';') {
				if (p_dst) {
					*p_dst = '>';
				}
				eat = 4;
			} else if (p_src_len >= 4 && p_src[1] == 'l' && p_src[2] == 't' && p_src[3] == ';') {
				if (p_dst) {
					*p_dst = '<';
				}
				eat = 4;
			} else if (p_src_len >= 5 && p_src[1] == 'a' && p_src[2] == 'm' && p_src[3] == 'p' && p_src[4] == ';') {
				if (p_dst) {
					*p_dst = '&';
				}
				eat = 5;
			} else if (p_src_len >= 6 && p_src[1] == 'q' && p_src[2] == 'u' && p_src[3] == 'o' && p_src[4] == 't' && p_src[5] == ';') {
				if (p_dst) {
					*p_dst = '"';
				}
				eat = 6;
			} else if (p_src_len >= 6 && p_src[1] == 'a' && p_src[2] == 'p' && p_src[3] == 'o' && p_src[4] == 's' && p_src[5] == ';') {
				if (p_dst) {
					*p_dst = '\'';
				}
				eat = 6;
			} else {
				if (p_dst) {
					*p_dst = *p_src;
				}
				eat = 1;
			}

			if (p_dst) {
				p_dst++;
			}

			len++;
			p_src += eat;
			p_src_len -= eat;
		} else {
			if (p_dst) {
				*p_dst = *p_src;
				p_dst++;
			}
			len++;
			p_src++;
			p_src_len--;
		}
	}

	return len;
}

String String::xml_unescape() const {
	String str;
	int l = length();
	int len = _xml_unescape(get_data(), l, nullptr);
	if (len == 0) {
		return String();
	}
	str.resize(len + 1);
	_xml_unescape(get_data(), l, str.ptrw());
	str[len] = 0;
	return str;
}

String String::pad_decimals(int p_digits) const {
	String s = *this;
	int c = s.find(".");

	if (c == -1) {
		if (p_digits <= 0) {
			return s;
		}
		s += ".";
		c = s.length() - 1;
	} else {
		if (p_digits <= 0) {
			return s.substr(0, c);
		}
	}

	if (s.length() - (c + 1) > p_digits) {
		s = s.substr(0, c + p_digits + 1);
	} else {
		while (s.length() - (c + 1) < p_digits) {
			s += "0";
		}
	}
	return s;
}

String String::pad_zeros(int p_digits) const {
	String s = *this;
	int end = s.find(".");

	if (end == -1) {
		end = s.length();
	}

	if (end == 0) {
		return s;
	}

	int begin = 0;

	while (begin < end && (s[begin] < '0' || s[begin] > '9')) {
		begin++;
	}

	if (begin >= end) {
		return s;
	}

	while (end - begin < p_digits) {
		s = s.insert(begin, "0");
		end++;
	}

	return s;
}

String String::trim_prefix(const String &p_prefix) const {
	String s = *this;
	if (s.begins_with(p_prefix)) {
		return s.substr(p_prefix.length(), s.length() - p_prefix.length());
	}
	return s;
}

String String::trim_suffix(const String &p_suffix) const {
	String s = *this;
	if (s.ends_with(p_suffix)) {
		return s.substr(0, s.length() - p_suffix.length());
	}
	return s;
}

bool String::is_valid_int() const {
	int len = length();

	if (len == 0) {
		return false;
	}

	int from = 0;
	if (len != 1 && (operator[](0) == '+' || operator[](0) == '-')) {
		from++;
	}

	for (int i = from; i < len; i++) {
		if (operator[](i) < '0' || operator[](i) > '9') {
			return false; // no start with number plz
		}
	}

	return true;
}

bool String::is_valid_hex_number(bool p_with_prefix) const {
	int len = length();

	if (len == 0) {
		return false;
	}

	int from = 0;
	if (len != 1 && (operator[](0) == '+' || operator[](0) == '-')) {
		from++;
	}

	if (p_with_prefix) {
		if (len < 3) {
			return false;
		}
		if (operator[](from) != '0' || operator[](from + 1) != 'x') {
			return false;
		}
		from += 2;
	}

	for (int i = from; i < len; i++) {
		char32_t c = operator[](i);
		if (is_hex_digit(c)) {
			continue;
		}
		return false;
	}

	return true;
}

bool String::is_valid_float() const {
	int len = length();

	if (len == 0) {
		return false;
	}

	int from = 0;
	if (operator[](0) == '+' || operator[](0) == '-') {
		from++;
	}

	bool exponent_found = false;
	bool period_found = false;
	bool sign_found = false;
	bool exponent_values_found = false;
	bool numbers_found = false;

	for (int i = from; i < len; i++) {
		if (is_digit(operator[](i))) {
			if (exponent_found) {
				exponent_values_found = true;
			} else {
				numbers_found = true;
			}
		} else if (numbers_found && !exponent_found && operator[](i) == 'e') {
			exponent_found = true;
		} else if (!period_found && !exponent_found && operator[](i) == '.') {
			period_found = true;
		} else if ((operator[](i) == '-' || operator[](i) == '+') && exponent_found && !exponent_values_found && !sign_found) {
			sign_found = true;
		} else {
			return false; // no start with number plz
		}
	}

	return numbers_found;
}

String String::path_to_file(const String &p_path) const {
	// Don't get base dir for src, this is expected to be a dir already.
	String src = this->replace("\\", "/");
	String dst = p_path.replace("\\", "/").get_base_dir();
	String rel = src.path_to(dst);
	if (rel == dst) { // failed
		return p_path;
	} else {
		return rel + p_path.get_file();
	}
}

String String::path_to(const String &p_path) const {
	String src = this->replace("\\", "/");
	String dst = p_path.replace("\\", "/");
	if (!src.ends_with("/")) {
		src += "/";
	}
	if (!dst.ends_with("/")) {
		dst += "/";
	}

	String base;

	if (src.begins_with("res://") && dst.begins_with("res://")) {
		base = "res:/";
		src = src.replace("res://", "/");
		dst = dst.replace("res://", "/");

	} else if (src.begins_with("user://") && dst.begins_with("user://")) {
		base = "user:/";
		src = src.replace("user://", "/");
		dst = dst.replace("user://", "/");

	} else if (src.begins_with("/") && dst.begins_with("/")) {
		//nothing
	} else {
		//dos style
		String src_begin = src.get_slicec('/', 0);
		String dst_begin = dst.get_slicec('/', 0);

		if (src_begin != dst_begin) {
			return p_path; //impossible to do this
		}

		base = src_begin;
		src = src.substr(src_begin.length(), src.length());
		dst = dst.substr(dst_begin.length(), dst.length());
	}

	//remove leading and trailing slash and split
	Vector<String> src_dirs = src.substr(1, src.length() - 2).split("/");
	Vector<String> dst_dirs = dst.substr(1, dst.length() - 2).split("/");

	//find common parent
	int common_parent = 0;

	while (true) {
		if (src_dirs.size() == common_parent) {
			break;
		}
		if (dst_dirs.size() == common_parent) {
			break;
		}
		if (src_dirs[common_parent] != dst_dirs[common_parent]) {
			break;
		}
		common_parent++;
	}

	common_parent--;

	String dir;

	for (int i = src_dirs.size() - 1; i > common_parent; i--) {
		dir += "../";
	}

	for (int i = common_parent + 1; i < dst_dirs.size(); i++) {
		dir += dst_dirs[i] + "/";
	}

	if (dir.length() == 0) {
		dir = "./";
	}
	return dir;
}

bool String::is_valid_html_color() const {
	return Color::html_is_valid(*this);
}

bool String::is_valid_filename() const {
	String stripped = strip_edges();
	if (*this != stripped) {
		return false;
	}

	if (stripped.is_empty()) {
		return false;
	}

	return !(find(":") != -1 || find("/") != -1 || find("\\") != -1 || find("?") != -1 || find("*") != -1 || find("\"") != -1 || find("|") != -1 || find("%") != -1 || find("<") != -1 || find(">") != -1);
}

bool String::is_valid_ip_address() const {
	if (find(":") >= 0) {
		Vector<String> ip = split(":");
		for (int i = 0; i < ip.size(); i++) {
			String n = ip[i];
			if (n.is_empty()) {
				continue;
			}
			if (n.is_valid_hex_number(false)) {
				int64_t nint = n.hex_to_int();
				if (nint < 0 || nint > 0xffff) {
					return false;
				}
				continue;
			}
			if (!n.is_valid_ip_address()) {
				return false;
			}
		}

	} else {
		Vector<String> ip = split(".");
		if (ip.size() != 4) {
			return false;
		}
		for (int i = 0; i < ip.size(); i++) {
			String n = ip[i];
			if (!n.is_valid_int()) {
				return false;
			}
			int val = n.to_int();
			if (val < 0 || val > 255) {
				return false;
			}
		}
	}

	return true;
}

bool String::is_resource_file() const {
	return begins_with("res://") && find("::") == -1;
}

bool String::is_relative_path() const {
	return !is_absolute_path();
}

String String::get_base_dir() const {
	int end = 0;

	// url scheme style base
	int basepos = find("://");
	if (basepos != -1) {
		end = basepos + 3;
	}

	// windows top level directory base
	if (end == 0) {
		basepos = find(":/");
		if (basepos == -1) {
			basepos = find(":\\");
		}
		if (basepos != -1) {
			end = basepos + 2;
		}
	}

	// unix root directory base
	if (end == 0) {
		if (begins_with("/")) {
			end = 1;
		}
	}

	String rs;
	String base;
	if (end != 0) {
		rs = substr(end, length());
		base = substr(0, end);
	} else {
		rs = *this;
	}

	int sep = MAX(rs.rfind("/"), rs.rfind("\\"));
	if (sep == -1) {
		return base;
	}

	return base + rs.substr(0, sep);
}

String String::get_file() const {
	int sep = MAX(rfind("/"), rfind("\\"));
	if (sep == -1) {
		return *this;
	}

	return substr(sep + 1, length());
}

String String::get_extension() const {
	int pos = rfind(".");
	if (pos < 0 || pos < MAX(rfind("/"), rfind("\\"))) {
		return "";
	}

	return substr(pos + 1, length());
}

String String::plus_file(const String &p_file) const {
	if (is_empty()) {
		return p_file;
	}
	if (operator[](length() - 1) == '/' || (p_file.size() > 0 && p_file.operator[](0) == '/')) {
		return *this + p_file;
	}
	return *this + "/" + p_file;
}

String String::property_name_encode() const {
	// Escape and quote strings with extended ASCII or further Unicode characters
	// as well as '"', '=' or ' ' (32)
	const char32_t *cstr = get_data();
	for (int i = 0; cstr[i]; i++) {
		if (cstr[i] == '=' || cstr[i] == '"' || cstr[i] == ';' || cstr[i] == '[' || cstr[i] == ']' || cstr[i] < 33 || cstr[i] > 126) {
			return "\"" + c_escape_multiline() + "\"";
		}
	}
	// Keep as is
	return *this;
}

// Changes made to the set of invalid characters must also be reflected in the String documentation.
const String String::invalid_node_name_characters = ". : @ / \"";

String String::validate_node_name() const {
	Vector<String> chars = String::invalid_node_name_characters.split(" ");
	String name = this->replace(chars[0], "");
	for (int i = 1; i < chars.size(); i++) {
		name = name.replace(chars[i], "");
	}
	return name;
}

String String::get_basename() const {
	int pos = rfind(".");
	if (pos < 0 || pos < MAX(rfind("/"), rfind("\\"))) {
		return *this;
	}

	return substr(0, pos);
}

String itos(int64_t p_val) {
	return String::num_int64(p_val);
}

String uitos(uint64_t p_val) {
	return String::num_uint64(p_val);
}

String rtos(double p_val) {
	return String::num(p_val);
}

String rtoss(double p_val) {
	return String::num_scientific(p_val);
}

// Right-pad with a character.
String String::rpad(int min_length, const String &character) const {
	String s = *this;
	int padding = min_length - s.length();
	if (padding > 0) {
		for (int i = 0; i < padding; i++) {
			s = s + character;
		}
	}

	return s;
}

// Left-pad with a character.
String String::lpad(int min_length, const String &character) const {
	String s = *this;
	int padding = min_length - s.length();
	if (padding > 0) {
		for (int i = 0; i < padding; i++) {
			s = character + s;
		}
	}

	return s;
}

// sprintf is implemented in GDScript via:
//   "fish %s pie" % "frog"
//   "fish %s %d pie" % ["frog", 12]
// In case of an error, the string returned is the error description and "error" is true.
String String::sprintf(const Array &values, bool *error) const {
	String formatted;
	char32_t *self = (char32_t *)get_data();
	bool in_format = false;
	int value_index = 0;
	int min_chars = 0;
	int min_decimals = 0;
	bool in_decimals = false;
	bool pad_with_zeroes = false;
	bool left_justified = false;
	bool show_sign = false;

	if (error) {
		*error = true;
	}

	for (; *self; self++) {
		const char32_t c = *self;

		if (in_format) { // We have % - let's see what else we get.
			switch (c) {
				case '%': { // Replace %% with %
					formatted += chr(c);
					in_format = false;
					break;
				}
				case 'd': // Integer (signed)
				case 'o': // Octal
				case 'x': // Hexadecimal (lowercase)
				case 'X': { // Hexadecimal (uppercase)
					if (value_index >= values.size()) {
						return "not enough arguments for format string";
					}

					if (!values[value_index].is_num()) {
						return "a number is required";
					}

					int64_t value = values[value_index];
					int base = 16;
					bool capitalize = false;
					switch (c) {
						case 'd':
							base = 10;
							break;
						case 'o':
							base = 8;
							break;
						case 'x':
							break;
						case 'X':
							base = 16;
							capitalize = true;
							break;
					}
					// Get basic number.
					String str = String::num_int64(ABS(value), base, capitalize);
					int number_len = str.length();

					// Padding.
					int pad_chars_count = (value < 0 || show_sign) ? min_chars - 1 : min_chars;
					String pad_char = pad_with_zeroes ? String("0") : String(" ");
					if (left_justified) {
						str = str.rpad(pad_chars_count, pad_char);
					} else {
						str = str.lpad(pad_chars_count, pad_char);
					}

					// Sign.
					if (show_sign && value >= 0) {
						str = str.insert(pad_with_zeroes ? 0 : str.length() - number_len, "+");
					} else if (value < 0) {
						str = str.insert(pad_with_zeroes ? 0 : str.length() - number_len, "-");
					}

					formatted += str;
					++value_index;
					in_format = false;

					break;
				}
				case 'f': { // Float
					if (value_index >= values.size()) {
						return "not enough arguments for format string";
					}

					if (!values[value_index].is_num()) {
						return "a number is required";
					}

					double value = values[value_index];
					bool is_negative = (value < 0);
					String str = String::num(ABS(value), min_decimals);

					// Pad decimals out.
					str = str.pad_decimals(min_decimals);

					int initial_len = str.length();

					// Padding. Leave room for sign later if required.
					int pad_chars_count = (is_negative || show_sign) ? min_chars - 1 : min_chars;
					String pad_char = pad_with_zeroes ? String("0") : String(" ");
					if (left_justified) {
						if (pad_with_zeroes) {
							return "left justification cannot be used with zeros as the padding";
						} else {
							str = str.rpad(pad_chars_count, pad_char);
						}
					} else {
						str = str.lpad(pad_chars_count, pad_char);
					}

					// Add sign if needed.
					if (show_sign || is_negative) {
						String sign_char = is_negative ? "-" : "+";
						if (left_justified) {
							str = str.insert(0, sign_char);
						} else {
							str = str.insert(pad_with_zeroes ? 0 : str.length() - initial_len, sign_char);
						}
					}

					formatted += str;
					++value_index;
					in_format = false;
					break;
				}
				case 's': { // String
					if (value_index >= values.size()) {
						return "not enough arguments for format string";
					}

					String str = values[value_index];
					// Padding.
					if (left_justified) {
						str = str.rpad(min_chars);
					} else {
						str = str.lpad(min_chars);
					}

					formatted += str;
					++value_index;
					in_format = false;
					break;
				}
				case 'c': {
					if (value_index >= values.size()) {
						return "not enough arguments for format string";
					}

					// Convert to character.
					String str;
					if (values[value_index].is_num()) {
						int value = values[value_index];
						if (value < 0) {
							return "unsigned integer is lower than minimum";
						} else if (value >= 0xd800 && value <= 0xdfff) {
							return "unsigned integer is invalid Unicode character";
						} else if (value > 0x10ffff) {
							return "unsigned integer is greater than maximum";
						}
						str = chr(values[value_index]);
					} else if (values[value_index].get_type() == Variant::STRING) {
						str = values[value_index];
						if (str.length() != 1) {
							return "%c requires number or single-character string";
						}
					} else {
						return "%c requires number or single-character string";
					}

					// Padding.
					if (left_justified) {
						str = str.rpad(min_chars);
					} else {
						str = str.lpad(min_chars);
					}

					formatted += str;
					++value_index;
					in_format = false;
					break;
				}
				case '-': { // Left justify
					left_justified = true;
					break;
				}
				case '+': { // Show + if positive.
					show_sign = true;
					break;
				}
				case '0':
				case '1':
				case '2':
				case '3':
				case '4':
				case '5':
				case '6':
				case '7':
				case '8':
				case '9': {
					int n = c - '0';
					if (in_decimals) {
						min_decimals *= 10;
						min_decimals += n;
					} else {
						if (c == '0' && min_chars == 0) {
							pad_with_zeroes = true;
						} else {
							min_chars *= 10;
							min_chars += n;
						}
					}
					break;
				}
				case '.': { // Float separator.
					if (in_decimals) {
						return "too many decimal points in format";
					}
					in_decimals = true;
					min_decimals = 0; // We want to add the value manually.
					break;
				}

				case '*': { // Dynamic width, based on value.
					if (value_index >= values.size()) {
						return "not enough arguments for format string";
					}

					if (!values[value_index].is_num()) {
						return "* wants number";
					}

					int size = values[value_index];

					if (in_decimals) {
						min_decimals = size;
					} else {
						min_chars = size;
					}

					++value_index;
					break;
				}

				default: {
					return "unsupported format character";
				}
			}
		} else { // Not in format string.
			switch (c) {
				case '%':
					in_format = true;
					// Back to defaults:
					min_chars = 0;
					min_decimals = 6;
					pad_with_zeroes = false;
					left_justified = false;
					show_sign = false;
					in_decimals = false;
					break;
				default:
					formatted += chr(c);
			}
		}
	}

	if (in_format) {
		return "incomplete format";
	}

	if (value_index != values.size()) {
		return "not all arguments converted during string formatting";
	}

	if (error) {
		*error = false;
	}
	return formatted;
}

String String::quote(String quotechar) const {
	return quotechar + *this + quotechar;
}

String String::unquote() const {
	if (!is_quoted()) {
		return *this;
	}

	return substr(1, length() - 2);
}

Vector<uint8_t> String::to_ascii_buffer() const {
	const String *s = this;
	if (s->is_empty()) {
		return Vector<uint8_t>();
	}
	CharString charstr = s->ascii();

	Vector<uint8_t> retval;
	size_t len = charstr.length();
	retval.resize(len);
	uint8_t *w = retval.ptrw();
	memcpy(w, charstr.ptr(), len);

	return retval;
}

Vector<uint8_t> String::to_utf8_buffer() const {
	const String *s = this;
	if (s->is_empty()) {
		return Vector<uint8_t>();
	}
	CharString charstr = s->utf8();

	Vector<uint8_t> retval;
	size_t len = charstr.length();
	retval.resize(len);
	uint8_t *w = retval.ptrw();
	memcpy(w, charstr.ptr(), len);

	return retval;
}

Vector<uint8_t> String::to_utf16_buffer() const {
	const String *s = this;
	if (s->is_empty()) {
		return Vector<uint8_t>();
	}
	Char16String charstr = s->utf16();

	Vector<uint8_t> retval;
	size_t len = charstr.length() * sizeof(char16_t);
	retval.resize(len);
	uint8_t *w = retval.ptrw();
	memcpy(w, (const void *)charstr.ptr(), len);

	return retval;
}

Vector<uint8_t> String::to_utf32_buffer() const {
	const String *s = this;
	if (s->is_empty()) {
		return Vector<uint8_t>();
	}

	Vector<uint8_t> retval;
	size_t len = s->length() * sizeof(char32_t);
	retval.resize(len);
	uint8_t *w = retval.ptrw();
	memcpy(w, (const void *)s->ptr(), len);

	return retval;
}

#ifdef TOOLS_ENABLED
String TTR(const String &p_text, const String &p_context) {
	if (TranslationServer::get_singleton()) {
		return TranslationServer::get_singleton()->tool_translate(p_text, p_context);
	}

	return p_text;
}

String TTRN(const String &p_text, const String &p_text_plural, int p_n, const String &p_context) {
	if (TranslationServer::get_singleton()) {
		return TranslationServer::get_singleton()->tool_translate_plural(p_text, p_text_plural, p_n, p_context);
	}

	// Return message based on English plural rule if translation is not possible.
	if (p_n == 1) {
		return p_text;
	}
	return p_text_plural;
}

/* DTR and DTRN are used for the documentation, handling descriptions extracted
 * from the XML.
 * They also replace `$DOCS_URL` with the actual URL to the documentation's branch,
 * to allow dehardcoding it in the XML and doing proper substitutions everywhere.
 */
String DTR(const String &p_text, const String &p_context) {
	// Comes straight from the XML, so remove indentation and any trailing whitespace.
	const String text = p_text.dedent().strip_edges();

	if (TranslationServer::get_singleton()) {
		return String(TranslationServer::get_singleton()->doc_translate(text, p_context)).replace("$DOCS_URL", VERSION_DOCS_URL);
	}

	return text.replace("$DOCS_URL", VERSION_DOCS_URL);
}

String DTRN(const String &p_text, const String &p_text_plural, int p_n, const String &p_context) {
	const String text = p_text.dedent().strip_edges();
	const String text_plural = p_text_plural.dedent().strip_edges();

	if (TranslationServer::get_singleton()) {
		return String(TranslationServer::get_singleton()->doc_translate_plural(text, text_plural, p_n, p_context)).replace("$DOCS_URL", VERSION_DOCS_URL);
	}

	// Return message based on English plural rule if translation is not possible.
	if (p_n == 1) {
		return text.replace("$DOCS_URL", VERSION_DOCS_URL);
	}
	return text_plural.replace("$DOCS_URL", VERSION_DOCS_URL);
}
#endif

String RTR(const String &p_text, const String &p_context) {
	if (TranslationServer::get_singleton()) {
		String rtr = TranslationServer::get_singleton()->tool_translate(p_text, p_context);
		if (rtr.is_empty() || rtr == p_text) {
			return TranslationServer::get_singleton()->translate(p_text, p_context);
		} else {
			return rtr;
		}
	}

	return p_text;
}

String RTRN(const String &p_text, const String &p_text_plural, int p_n, const String &p_context) {
	if (TranslationServer::get_singleton()) {
		String rtr = TranslationServer::get_singleton()->tool_translate_plural(p_text, p_text_plural, p_n, p_context);
		if (rtr.is_empty() || rtr == p_text || rtr == p_text_plural) {
			return TranslationServer::get_singleton()->translate_plural(p_text, p_text_plural, p_n, p_context);
		} else {
			return rtr;
		}
	}

	// Return message based on English plural rule if translation is not possible.
	if (p_n == 1) {
		return p_text;
	}
	return p_text_plural;
}
