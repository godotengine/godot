/**************************************************************************/
/*  char_string.cpp                                                       */
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

#include <godot_cpp/variant/char_string.hpp>

#include <godot_cpp/core/memory.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/string_name.hpp>

#include <godot_cpp/godot.hpp>

#include <cmath>
#include <string>

namespace godot {

template <typename L, typename R>
_FORCE_INLINE_ bool is_str_less(const L *l_ptr, const R *r_ptr) {
	while (true) {
		const char32_t l = *l_ptr;
		const char32_t r = *r_ptr;

		if (l == 0 && r == 0) {
			return false;
		} else if (l == 0) {
			return true;
		} else if (r == 0) {
			return false;
		} else if (l < r) {
			return true;
		} else if (l > r) {
			return false;
		}

		l_ptr++;
		r_ptr++;
	}
}

template <typename T>
bool CharStringT<T>::operator<(const CharStringT<T> &p_right) const {
	if (length() == 0) {
		return p_right.length() != 0;
	}

	return is_str_less(get_data(), p_right.get_data());
}

template <typename T>
CharStringT<T> &CharStringT<T>::operator+=(T p_char) {
	const int64_t lhs_len = length();
	resize(lhs_len + 2);

	T *dst = ptrw();
	dst[lhs_len] = p_char;
	dst[lhs_len + 1] = 0;

	return *this;
}

template <typename T>
void CharStringT<T>::operator=(const T *p_cstr) {
	copy_from(p_cstr);
}

template <>
const char *CharStringT<char>::get_data() const {
	if (size()) {
		return &operator[](0);
	} else {
		return "";
	}
}

template <>
const char16_t *CharStringT<char16_t>::get_data() const {
	if (size()) {
		return &operator[](0);
	} else {
		return u"";
	}
}

template <>
const char32_t *CharStringT<char32_t>::get_data() const {
	if (size()) {
		return &operator[](0);
	} else {
		return U"";
	}
}

template <>
const wchar_t *CharStringT<wchar_t>::get_data() const {
	if (size()) {
		return &operator[](0);
	} else {
		return L"";
	}
}

template <typename T>
void CharStringT<T>::copy_from(const T *p_cstr) {
	if (!p_cstr) {
		resize(0);
		return;
	}

	size_t len = std::char_traits<T>::length(p_cstr);

	if (len == 0) {
		resize(0);
		return;
	}

	Error err = resize(++len); // include terminating null char

	ERR_FAIL_COND_MSG(err != OK, "Failed to copy C-string.");

	memcpy(ptrw(), p_cstr, len);
}

template class CharStringT<char>;
template class CharStringT<char16_t>;
template class CharStringT<char32_t>;
template class CharStringT<wchar_t>;

// Custom String functions that are not part of bound API.
// It's easier to have them written in C++ directly than in a Python script that generates them.

String::String(const char *from) {
	internal::gdextension_interface_string_new_with_latin1_chars(_native_ptr(), from);
}

String::String(const wchar_t *from) {
	internal::gdextension_interface_string_new_with_wide_chars(_native_ptr(), from);
}

String::String(const char16_t *from) {
	internal::gdextension_interface_string_new_with_utf16_chars(_native_ptr(), from);
}

String::String(const char32_t *from) {
	internal::gdextension_interface_string_new_with_utf32_chars(_native_ptr(), from);
}

String String::utf8(const char *from, int64_t len) {
	String ret;
	ret.parse_utf8(from, len);
	return ret;
}

Error String::parse_utf8(const char *from, int64_t len) {
	return (Error)internal::gdextension_interface_string_new_with_utf8_chars_and_len2(_native_ptr(), from, len);
}

String String::utf16(const char16_t *from, int64_t len) {
	String ret;
	ret.parse_utf16(from, len);
	return ret;
}

Error String::parse_utf16(const char16_t *from, int64_t len, bool default_little_endian) {
	return (Error)internal::gdextension_interface_string_new_with_utf16_chars_and_len2(_native_ptr(), from, len, default_little_endian);
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

CharString String::utf8() const {
	int64_t length = internal::gdextension_interface_string_to_utf8_chars(_native_ptr(), nullptr, 0);
	int64_t size = length + 1;
	CharString str;
	str.resize(size);
	internal::gdextension_interface_string_to_utf8_chars(_native_ptr(), str.ptrw(), length);

	str[length] = '\0';

	return str;
}

CharString String::ascii() const {
	int64_t length = internal::gdextension_interface_string_to_latin1_chars(_native_ptr(), nullptr, 0);
	int64_t size = length + 1;
	CharString str;
	str.resize(size);
	internal::gdextension_interface_string_to_latin1_chars(_native_ptr(), str.ptrw(), length);

	str[length] = '\0';

	return str;
}

Char16String String::utf16() const {
	int64_t length = internal::gdextension_interface_string_to_utf16_chars(_native_ptr(), nullptr, 0);
	int64_t size = length + 1;
	Char16String str;
	str.resize(size);
	internal::gdextension_interface_string_to_utf16_chars(_native_ptr(), str.ptrw(), length);

	str[length] = '\0';

	return str;
}

Char32String String::utf32() const {
	int64_t length = internal::gdextension_interface_string_to_utf32_chars(_native_ptr(), nullptr, 0);
	int64_t size = length + 1;
	Char32String str;
	str.resize(size);
	internal::gdextension_interface_string_to_utf32_chars(_native_ptr(), str.ptrw(), length);

	str[length] = '\0';

	return str;
}

CharWideString String::wide_string() const {
	int64_t length = internal::gdextension_interface_string_to_wide_chars(_native_ptr(), nullptr, 0);
	int64_t size = length + 1;
	CharWideString str;
	str.resize(size);
	internal::gdextension_interface_string_to_wide_chars(_native_ptr(), str.ptrw(), length);

	str[length] = '\0';

	return str;
}

Error String::resize(int64_t p_size) {
	return (Error)internal::gdextension_interface_string_resize(_native_ptr(), p_size);
}

String &String::operator=(const char *p_str) {
	*this = String(p_str);
	return *this;
}

String &String::operator=(const wchar_t *p_str) {
	*this = String(p_str);
	return *this;
}

String &String::operator=(const char16_t *p_str) {
	*this = String(p_str);
	return *this;
}

String &String::operator=(const char32_t *p_str) {
	*this = String(p_str);
	return *this;
}

bool String::operator==(const char *p_str) const {
	return *this == String(p_str);
}

bool String::operator==(const wchar_t *p_str) const {
	return *this == String(p_str);
}

bool String::operator==(const char16_t *p_str) const {
	return *this == String(p_str);
}

bool String::operator==(const char32_t *p_str) const {
	return *this == String(p_str);
}

bool String::operator!=(const char *p_str) const {
	return *this != String(p_str);
}

bool String::operator!=(const wchar_t *p_str) const {
	return *this != String(p_str);
}

bool String::operator!=(const char16_t *p_str) const {
	return *this != String(p_str);
}

bool String::operator!=(const char32_t *p_str) const {
	return *this != String(p_str);
}

String String::operator+(const char *p_str) {
	return *this + String(p_str);
}

String String::operator+(const wchar_t *p_str) {
	return *this + String(p_str);
}

String String::operator+(const char16_t *p_str) {
	return *this + String(p_str);
}

String String::operator+(const char32_t *p_str) {
	return *this + String(p_str);
}

String String::operator+(const char32_t p_char) {
	return *this + String::chr(p_char);
}

String &String::operator+=(const String &p_str) {
	internal::gdextension_interface_string_operator_plus_eq_string((GDExtensionStringPtr)this, (GDExtensionConstStringPtr)&p_str);
	return *this;
}

String &String::operator+=(char32_t p_char) {
	internal::gdextension_interface_string_operator_plus_eq_char((GDExtensionStringPtr)this, p_char);
	return *this;
}

String &String::operator+=(const char *p_str) {
	internal::gdextension_interface_string_operator_plus_eq_cstr((GDExtensionStringPtr)this, p_str);
	return *this;
}

String &String::operator+=(const wchar_t *p_str) {
	internal::gdextension_interface_string_operator_plus_eq_wcstr((GDExtensionStringPtr)this, p_str);
	return *this;
}

String &String::operator+=(const char32_t *p_str) {
	internal::gdextension_interface_string_operator_plus_eq_c32str((GDExtensionStringPtr)this, p_str);
	return *this;
}

const char32_t &String::operator[](int64_t p_index) const {
	return *internal::gdextension_interface_string_operator_index_const((GDExtensionStringPtr)this, p_index);
}

char32_t &String::operator[](int64_t p_index) {
	return *internal::gdextension_interface_string_operator_index((GDExtensionStringPtr)this, p_index);
}

const char32_t *String::ptr() const {
	return internal::gdextension_interface_string_operator_index_const((GDExtensionStringPtr)this, 0);
}

char32_t *String::ptrw() {
	return internal::gdextension_interface_string_operator_index((GDExtensionStringPtr)this, 0);
}

bool operator==(const char *p_chr, const String &p_str) {
	return p_str == String(p_chr);
}

bool operator==(const wchar_t *p_chr, const String &p_str) {
	return p_str == String(p_chr);
}

bool operator==(const char16_t *p_chr, const String &p_str) {
	return p_str == String(p_chr);
}

bool operator==(const char32_t *p_chr, const String &p_str) {
	return p_str == String(p_chr);
}

bool operator!=(const char *p_chr, const String &p_str) {
	return !(p_str == p_chr);
}

bool operator!=(const wchar_t *p_chr, const String &p_str) {
	return !(p_str == p_chr);
}

bool operator!=(const char16_t *p_chr, const String &p_str) {
	return !(p_str == p_chr);
}

bool operator!=(const char32_t *p_chr, const String &p_str) {
	return !(p_str == p_chr);
}

String operator+(const char *p_chr, const String &p_str) {
	return String(p_chr) + p_str;
}

String operator+(const wchar_t *p_chr, const String &p_str) {
	return String(p_chr) + p_str;
}

String operator+(const char16_t *p_chr, const String &p_str) {
	return String(p_chr) + p_str;
}

String operator+(const char32_t *p_chr, const String &p_str) {
	return String(p_chr) + p_str;
}

String operator+(char32_t p_char, const String &p_str) {
	return String::chr(p_char) + p_str;
}

StringName::StringName(const char *from, bool p_static) {
	internal::gdextension_interface_string_name_new_with_latin1_chars(&opaque, from, p_static);
}

StringName::StringName(const wchar_t *from) :
		StringName(String(from)) {}

StringName::StringName(const char16_t *from) :
		StringName(String(from)) {}

StringName::StringName(const char32_t *from) :
		StringName(String(from)) {}

NodePath::NodePath(const char *from) :
		NodePath(String(from)) {}

NodePath::NodePath(const wchar_t *from) :
		NodePath(String(from)) {}

NodePath::NodePath(const char16_t *from) :
		NodePath(String(from)) {}

NodePath::NodePath(const char32_t *from) :
		NodePath(String(from)) {}

} // namespace godot
