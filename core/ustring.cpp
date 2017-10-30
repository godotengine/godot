/*************************************************************************/
/*  ustring.cpp                                                          */
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
#include "ustring.h"

#include "color.h"
#include "math_funcs.h"
#include "os/memory.h"
#include "print_string.h"
#include "ucaps.h"
#include "variant.h"

#include "thirdparty/misc/md5.h"
#include "thirdparty/misc/sha256.h"

#include <wchar.h>

#ifndef NO_USE_STDLIB
#include <stdio.h>
#include <stdlib.h>
#endif

#if defined(MINGW_ENABLED) || defined(_MSC_VER)
#define snprintf _snprintf
#endif

#define MAX_DIGITS 6
#define UPPERCASE(m_c) (((m_c) >= 'a' && (m_c) <= 'z') ? ((m_c) - ('a' - 'A')) : (m_c))
#define LOWERCASE(m_c) (((m_c) >= 'A' && (m_c) <= 'Z') ? ((m_c) + ('a' - 'A')) : (m_c))
#define IS_DIGIT(m_d) ((m_d) >= '0' && (m_d) <= '9')
#define IS_HEX_DIGIT(m_d) (((m_d) >= '0' && (m_d) <= '9') || ((m_d) >= 'a' && (m_d) <= 'f') || ((m_d) >= 'A' && (m_d) <= 'F'))

/** STRING **/

bool CharString::operator<(const CharString &p_right) const {

	if (length() == 0) {
		return p_right.length() != 0;
	}

	const char *this_str = get_data();
	const char *that_str = p_right.get_data();
	while (true) {

		if (*that_str == 0 && *this_str == 0)
			return false; //this can't be equal, sadly
		else if (*this_str == 0)
			return true; //if this is empty, and the other one is not, then we're less.. I think?
		else if (*that_str == 0)
			return false; //otherwise the other one is smaller..
		else if (*this_str < *that_str) //more than
			return true;
		else if (*this_str > *that_str) //less than
			return false;

		this_str++;
		that_str++;
	}

	return false; //should never reach here anyway
}

const char *CharString::get_data() const {

	if (size())
		return &operator[](0);
	else
		return "";
}

void String::copy_from(const char *p_cstr) {

	if (!p_cstr) {

		resize(0);
		return;
	}

	int len = 0;
	const char *ptr = p_cstr;
	while (*(ptr++) != 0)
		len++;

	if (len == 0) {

		resize(0);
		return;
	}

	resize(len + 1); // include 0

	CharType *dst = this->ptr();

	for (int i = 0; i < len + 1; i++) {

		dst[i] = p_cstr[i];
	}
}

void String::copy_from(const CharType *p_cstr, int p_clip_to) {

	if (!p_cstr) {

		resize(0);
		return;
	}

	int len = 0;
	const CharType *ptr = p_cstr;
	while (*(ptr++) != 0)
		len++;

	if (p_clip_to >= 0 && len > p_clip_to)
		len = p_clip_to;

	if (len == 0) {

		resize(0);
		return;
	}

	resize(len + 1);
	set(len, 0);

	CharType *dst = &operator[](0);

	for (int i = 0; i < len; i++) {

		dst[i] = p_cstr[i];
	}
}

void String::copy_from(const CharType &p_char) {

	resize(2);
	set(0, p_char);
	set(1, 0);
}

bool String::operator==(const String &p_str) const {

	if (length() != p_str.length())
		return false;
	if (empty())
		return true;

	int l = length();

	const CharType *src = c_str();
	const CharType *dst = p_str.c_str();

	/* Compare char by char */
	for (int i = 0; i < l; i++) {

		if (src[i] != dst[i])
			return false;
	}

	return true;
}

bool String::operator!=(const String &p_str) const {

	return !(*this == p_str);
}

String String::operator+(const String &p_str) const {

	String res = *this;
	res += p_str;
	return res;
}

/*
String String::operator+(CharType p_chr)  const {

	String res=*this;
	res+=p_chr;
	return res;
}
*/
String &String::operator+=(const String &p_str) {

	if (empty()) {
		*this = p_str;
		return *this;
	}

	if (p_str.empty())
		return *this;

	int from = length();

	resize(length() + p_str.size());

	const CharType *src = p_str.c_str();
	CharType *dst = &operator[](0);

	set(length(), 0);

	for (int i = 0; i < p_str.length(); i++)
		dst[from + i] = src[i];

	return *this;
}

String &String::operator+=(const CharType *p_str) {

	*this += String(p_str);
	return *this;
}

String &String::operator+=(CharType p_char) {

	resize(size() ? size() + 1 : 2);
	set(length(), 0);
	set(length() - 1, p_char);

	return *this;
}

String &String::operator+=(const char *p_str) {

	if (!p_str || p_str[0] == 0)
		return *this;

	int src_len = 0;
	const char *ptr = p_str;
	while (*(ptr++) != 0)
		src_len++;

	int from = length();

	resize(from + src_len + 1);

	CharType *dst = &operator[](0);

	set(length(), 0);

	for (int i = 0; i < src_len; i++)
		dst[from + i] = p_str[i];

	return *this;
}

void String::operator=(const char *p_str) {

	copy_from(p_str);
}

void String::operator=(const CharType *p_str) {

	copy_from(p_str);
}

bool String::operator==(const StrRange &p_str_range) const {

	int len = p_str_range.len;

	if (length() != len)
		return false;
	if (empty())
		return true;

	const CharType *c_str = p_str_range.c_str;
	const CharType *dst = &operator[](0);

	/* Compare char by char */
	for (int i = 0; i < len; i++) {

		if (c_str[i] != dst[i])
			return false;
	}

	return true;
}

bool String::operator==(const char *p_str) const {

	int len = 0;
	const char *aux = p_str;

	while (*(aux++) != 0)
		len++;

	if (length() != len)
		return false;
	if (empty())
		return true;

	int l = length();

	const CharType *dst = c_str();

	/* Compare char by char */
	for (int i = 0; i < l; i++) {

		if (p_str[i] != dst[i])
			return false;
	}

	return true;
}

bool String::operator==(const CharType *p_str) const {

	int len = 0;
	const CharType *aux = p_str;

	while (*(aux++) != 0)
		len++;

	if (length() != len)
		return false;
	if (empty())
		return true;

	int l = length();

	const CharType *dst = c_str();

	/* Compare char by char */
	for (int i = 0; i < l; i++) {

		if (p_str[i] != dst[i])
			return false;
	}

	return true;
}

bool String::operator!=(const char *p_str) const {

	return (!(*this == p_str));
}

bool String::operator!=(const CharType *p_str) const {

	return (!(*this == p_str));
}

bool String::operator<(const CharType *p_str) const {

	if (empty() && p_str[0] == 0)
		return false;
	if (empty())
		return true;

	const CharType *this_str = c_str();
	while (true) {

		if (*p_str == 0 && *this_str == 0)
			return false; //this can't be equal, sadly
		else if (*this_str == 0)
			return true; //if this is empty, and the other one is not, then we're less.. I think?
		else if (*p_str == 0)
			return false; //otherwise the other one is smaller..
		else if (*this_str < *p_str) //more than
			return true;
		else if (*this_str > *p_str) //less than
			return false;

		this_str++;
		p_str++;
	}

	return false; //should never reach here anyway
}

bool String::operator<=(String p_str) const {

	return (*this < p_str) || (*this == p_str);
}

bool String::operator<(const char *p_str) const {

	if (empty() && p_str[0] == 0)
		return false;
	if (empty())
		return true;

	const CharType *this_str = c_str();
	while (true) {

		if (*p_str == 0 && *this_str == 0)
			return false; //this can't be equal, sadly
		else if (*this_str == 0)
			return true; //if this is empty, and the other one is not, then we're less.. I think?
		else if (*p_str == 0)
			return false; //otherwise the other one is smaller..
		else if (*this_str < *p_str) //more than
			return true;
		else if (*this_str > *p_str) //less than
			return false;

		this_str++;
		p_str++;
	}

	return false; //should never reach here anyway
}

bool String::operator<(String p_str) const {

	return operator<(p_str.c_str());
}

signed char String::nocasecmp_to(const String &p_str) const {

	if (empty() && p_str.empty())
		return 0;
	if (empty())
		return -1;
	if (p_str.empty())
		return 1;

	const CharType *that_str = p_str.c_str();
	const CharType *this_str = c_str();

	while (true) {

		if (*that_str == 0 && *this_str == 0)
			return 0; //we're equal
		else if (*this_str == 0)
			return -1; //if this is empty, and the other one is not, then we're less.. I think?
		else if (*that_str == 0)
			return 1; //otherwise the other one is smaller..
		else if (_find_upper(*this_str) < _find_upper(*that_str)) //more than
			return -1;
		else if (_find_upper(*this_str) > _find_upper(*that_str)) //less than
			return 1;

		this_str++;
		that_str++;
	}

	return 0; //should never reach anyway
}

signed char String::casecmp_to(const String &p_str) const {

	if (empty() && p_str.empty())
		return 0;
	if (empty())
		return -1;
	if (p_str.empty())
		return 1;

	const CharType *that_str = p_str.c_str();
	const CharType *this_str = c_str();

	while (true) {

		if (*that_str == 0 && *this_str == 0)
			return 0; //we're equal
		else if (*this_str == 0)
			return -1; //if this is empty, and the other one is not, then we're less.. I think?
		else if (*that_str == 0)
			return 1; //otherwise the other one is smaller..
		else if (*this_str < *that_str) //more than
			return -1;
		else if (*this_str > *that_str) //less than
			return 1;

		this_str++;
		that_str++;
	}

	return 0; //should never reach anyway
}

signed char String::naturalnocasecmp_to(const String &p_str) const {

	const CharType *this_str = c_str();
	const CharType *that_str = p_str.c_str();

	if (this_str && that_str) {

		while (*this_str == '.' || *that_str == '.') {
			if (*this_str++ != '.')
				return 1;
			if (*that_str++ != '.')
				return -1;
			if (!*that_str)
				return 1;
			if (!*this_str)
				return -1;
		}

		while (*this_str) {

			if (!*that_str)
				return 1;
			else if (IS_DIGIT(*this_str)) {

				int64_t this_int, that_int;

				if (!IS_DIGIT(*that_str))
					return -1;

				/* Compare the numbers */
				this_int = to_int(this_str);
				that_int = to_int(that_str);

				if (this_int < that_int)
					return -1;
				else if (this_int > that_int)
					return 1;

				/* Skip */
				while (IS_DIGIT(*this_str))
					this_str++;
				while (IS_DIGIT(*that_str))
					that_str++;
			} else if (IS_DIGIT(*that_str))
				return 1;
			else {
				if (_find_upper(*this_str) < _find_upper(*that_str)) //more than
					return -1;
				else if (_find_upper(*this_str) > _find_upper(*that_str)) //less than
					return 1;

				this_str++;
				that_str++;
			}
		}
		if (*that_str)
			return -1;
	}

	return 0;
}

void String::erase(int p_pos, int p_chars) {

	*this = left(p_pos) + substr(p_pos + p_chars, length() - ((p_pos + p_chars)));
}

String String::capitalize() const {

	String aux = this->replace("_", " ").to_lower();
	String cap;
	for (int i = 0; i < aux.get_slice_count(" "); i++) {

		String slice = aux.get_slicec(' ', i);
		if (slice.length() > 0) {

			slice[0] = _find_upper(slice[0]);
			if (i > 0)
				cap += " ";
			cap += slice;
		}
	}

	return cap;
}

String String::camelcase_to_underscore(bool lowercase) const {
	const CharType *cstr = c_str();
	String new_string;
	const char A = 'A', Z = 'Z';
	const char a = 'a', z = 'z';
	int start_index = 0;

	for (int i = 1; i < this->size(); i++) {
		bool is_upper = cstr[i] >= A && cstr[i] <= Z;
		bool is_number = cstr[i] >= '0' && cstr[i] <= '9';
		bool are_next_2_lower = false;
		bool was_precedent_upper = cstr[i - 1] >= A && cstr[i - 1] <= Z;
		bool was_precedent_number = cstr[i - 1] >= '0' && cstr[i - 1] <= '9';

		if (i + 2 < this->size()) {
			are_next_2_lower = cstr[i + 1] >= a && cstr[i + 1] <= z && cstr[i + 2] >= a && cstr[i + 2] <= z;
		}

		bool should_split = ((is_upper && !was_precedent_upper && !was_precedent_number) || (was_precedent_upper && is_upper && are_next_2_lower) || (is_number && !was_precedent_number));
		if (should_split) {
			new_string += this->substr(start_index, i - start_index) + "_";
			start_index = i;
		}
	}

	new_string += this->substr(start_index, this->size() - start_index);
	return lowercase ? new_string.to_lower() : new_string;
}

int String::get_slice_count(String p_splitter) const {

	if (empty())
		return 0;
	if (p_splitter.empty())
		return 0;

	int pos = 0;
	int slices = 1;

	while ((pos = find(p_splitter, pos)) >= 0) {

		slices++;
		pos += p_splitter.length();
	}

	return slices;
}

String String::get_slice(String p_splitter, int p_slice) const {

	if (empty() || p_splitter.empty())
		return "";

	int pos = 0;
	int prev_pos = 0;
	//int slices=1;
	if (p_slice < 0)
		return "";
	if (find(p_splitter) == -1)
		return *this;

	int i = 0;
	while (true) {

		pos = find(p_splitter, pos);
		if (pos == -1)
			pos = length(); //reached end

		int from = prev_pos;
		//int to=pos;

		if (p_slice == i) {

			return substr(from, pos - from);
		}

		if (pos == length()) //reached end and no find
			break;
		pos += p_splitter.length();
		prev_pos = pos;
		i++;
	}

	return ""; //no find!
}

String String::get_slicec(CharType p_splitter, int p_slice) const {

	if (empty())
		return String();

	if (p_slice < 0)
		return String();

	const CharType *c = this->ptr();
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

	return String(); //no find!
}

Vector<String> String::split_spaces() const {

	Vector<String> ret;
	int from = 0;
	int i = 0;
	int len = length();
	bool inside = false;

	while (true) {

		bool empty = operator[](i) < 33;

		if (i == 0)
			inside = !empty;

		if (!empty && !inside) {
			inside = true;
			from = i;
		}

		if (empty && inside) {

			ret.push_back(substr(from, i - from));
			inside = false;
		}

		if (i == len)
			break;
		i++;
	}

	return ret;
}

Vector<String> String::split(const String &p_splitter, bool p_allow_empty) const {

	Vector<String> ret;
	int from = 0;
	int len = length();

	while (true) {

		int end = find(p_splitter, from);
		if (end < 0)
			end = len;
		if (p_allow_empty || (end > from))
			ret.push_back(substr(from, end - from));

		if (end == len)
			break;

		from = end + p_splitter.length();
	}

	return ret;
}

Vector<float> String::split_floats(const String &p_splitter, bool p_allow_empty) const {

	Vector<float> ret;
	int from = 0;
	int len = length();

	while (true) {

		int end = find(p_splitter, from);
		if (end < 0)
			end = len;
		if (p_allow_empty || (end > from))
			ret.push_back(String::to_double(&c_str()[from]));

		if (end == len)
			break;

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
			ret.push_back(String::to_double(&c_str()[from]));
		}

		if (end == len)
			break;

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
		if (end < 0)
			end = len;
		if (p_allow_empty || (end > from))
			ret.push_back(String::to_int(&c_str()[from], end - from));

		if (end == len)
			break;

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

		if (p_allow_empty || (end > from))
			ret.push_back(String::to_int(&c_str()[from], end - from));

		if (end == len)
			break;

		from = end + spl_len;
	}

	return ret;
}

CharType String::char_uppercase(CharType p_char) {

	return _find_upper(p_char);
}

CharType String::char_lowercase(CharType p_char) {

	return _find_lower(p_char);
}

String String::to_upper() const {

	String upper = *this;

	for (int i = 0; i < upper.size(); i++) {

		upper[i] = _find_upper(upper[i]);
	}

	return upper;
}

String String::to_lower() const {

	String upper = *this;

	for (int i = 0; i < upper.size(); i++) {

		upper[i] = _find_lower(upper[i]);
	}

	return upper;
}

int String::length() const {

	int s = size();
	return s ? (s - 1) : 0; // length does not include zero
}

const CharType *String::c_str() const {

	static const CharType zero = 0;

	return size() ? &operator[](0) : &zero;
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

String String::chr(CharType p_char) {

	CharType c[2] = { p_char, 0 };
	return String(c);
}
String String::num(double p_num, int p_decimals) {

#ifndef NO_USE_STDLIB

	if (p_decimals > 12)
		p_decimals = 12;

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
			if (buf[z] == '.')
				period = true;
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
#else

	String s;
	String sd;
	/* integer part */

	bool neg = p_num < 0;
	p_num = ABS(p_num);
	int intn = (int)p_num;

	/* decimal part */

	if (p_decimals > 0 || (p_decimals == -1 && (int)p_num != p_num)) {

		double dec = p_num - (float)((int)p_num);

		int digit = 0;
		if (p_decimals > MAX_DIGITS)
			p_decimals = MAX_DIGITS;

		int dec_int = 0;
		int dec_max = 0;

		while (true) {

			dec *= 10.0;
			dec_int = dec_int * 10 + (int)dec % 10;
			dec_max = dec_max * 10 + 9;
			digit++;

			if (p_decimals == -1) {

				if (digit == MAX_DIGITS) //no point in going to infinite
					break;

				if ((dec - (float)((int)dec)) < 1e-6)
					break;
			}

			if (digit == p_decimals)
				break;
		}
		dec *= 10;
		int last = (int)dec % 10;

		if (last > 5) {
			if (dec_int == dec_max) {

				dec_int = 0;
				intn++;
			} else {

				dec_int++;
			}
		}

		String decimal;
		for (int i = 0; i < digit; i++) {

			char num[2] = { 0, 0 };
			num[0] = '0' + dec_int % 10;
			decimal = num + decimal;
			dec_int /= 10;
		}
		sd = '.' + decimal;
	}

	if (intn == 0)

		s = "0";
	else {
		while (intn) {

			CharType num = '0' + (intn % 10);
			intn /= 10;
			s = num + s;
		}
	}

	s = s + sd;
	if (neg)
		s = "-" + s;
	return s;
#endif
}

String String::num_int64(int64_t p_num, int base, bool capitalize_hex) {

	bool sign = p_num < 0;
	int64_t num = ABS(p_num);

	int64_t n = num;

	int chars = 0;
	do {
		n /= base;
		chars++;
	} while (n);

	if (sign)
		chars++;
	String s;
	s.resize(chars + 1);
	CharType *c = s.ptr();
	c[chars] = 0;
	n = num;
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

	if (sign)
		c[0] = '-';

	return s;
}

String String::num_real(double p_num) {

	String s;
	String sd;
	/* integer part */

	bool neg = p_num < 0;
	p_num = ABS(p_num);
	int intn = (int)p_num;

	/* decimal part */

	if ((int)p_num != p_num) {

		double dec = p_num - (float)((int)p_num);

		int digit = 0;
		int decimals = MAX_DIGITS;

		int dec_int = 0;
		int dec_max = 0;

		while (true) {

			dec *= 10.0;
			dec_int = dec_int * 10 + (int)dec % 10;
			dec_max = dec_max * 10 + 9;
			digit++;

			if ((dec - (float)((int)dec)) < 1e-6)
				break;

			if (digit == decimals)
				break;
		}

		dec *= 10;
		int last = (int)dec % 10;

		if (last > 5) {
			if (dec_int == dec_max) {

				dec_int = 0;
				intn++;
			} else {

				dec_int++;
			}
		}

		String decimal;
		for (int i = 0; i < digit; i++) {

			char num[2] = { 0, 0 };
			num[0] = '0' + dec_int % 10;
			decimal = num + decimal;
			dec_int /= 10;
		}
		sd = '.' + decimal;
	} else {
		sd = ".0";
	}

	if (intn == 0)

		s = "0";
	else {
		while (intn) {

			CharType num = '0' + (intn % 10);
			intn /= 10;
			s = num + s;
		}
	}

	s = s + sd;
	if (neg)
		s = "-" + s;
	return s;
}

String String::num_scientific(double p_num) {

#ifndef NO_USE_STDLIB

	char buf[256];

#if defined(__GNUC__) || defined(_MSC_VER)
	snprintf(buf, 256, "%lg", p_num);
#else
	sprintf(buf, "%.16lg", p_num);
#endif

	buf[255] = 0;

	return buf;
#else

	return String::num(p_num);
#endif
}

CharString String::ascii(bool p_allow_extended) const {

	if (!length())
		return CharString();

	CharString cs;
	cs.resize(size());

	for (int i = 0; i < size(); i++)
		cs[i] = operator[](i);

	return cs;
}

String String::utf8(const char *p_utf8, int p_len) {

	String ret;
	ret.parse_utf8(p_utf8, p_len);

	return ret;
};

bool String::parse_utf8(const char *p_utf8, int p_len) {

#define _UNICERROR(m_err) print_line("unicode error: " + String(m_err));

	String aux;

	int cstr_size = 0;
	int str_size = 0;

	/* HANDLE BOM (Byte Order Mark) */
	if (p_len < 0 || p_len >= 3) {

		bool has_bom = uint8_t(p_utf8[0]) == 0xEF && uint8_t(p_utf8[1]) == 0xBB && uint8_t(p_utf8[2]) == 0xBF;
		if (has_bom) {

			//just skip it
			if (p_len >= 0)
				p_len -= 3;
			p_utf8 += 3;
		}
	}

	{
		const char *ptrtmp = p_utf8;
		const char *ptrtmp_limit = &p_utf8[p_len];
		int skip = 0;
		while (ptrtmp != ptrtmp_limit && *ptrtmp) {

			if (skip == 0) {

				uint8_t c = *ptrtmp;

				/* Determine the number of characters in sequence */
				if ((c & 0x80) == 0)
					skip = 0;
				else if ((c & 0xE0) == 0xC0)
					skip = 1;
				else if ((c & 0xF0) == 0xE0)
					skip = 2;
				else if ((c & 0xF8) == 0xF0)
					skip = 3;
				else if ((c & 0xFC) == 0xF8)
					skip = 4;
				else if ((c & 0xFE) == 0xFC)
					skip = 5;
				else {
					_UNICERROR("invalid skip");
					return true; //invalid utf8
				}

				if (skip == 1 && (c & 0x1E) == 0) {
					//printf("overlong rejected\n");
					_UNICERROR("overlong rejected");
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
			_UNICERROR("no space left");
			return true; //not enough spac
		}
	}

	if (str_size == 0) {
		clear();
		return false;
	}

	resize(str_size + 1);
	CharType *dst = &operator[](0);
	dst[str_size] = 0;

	while (cstr_size) {

		int len = 0;

		/* Determine the number of characters in sequence */
		if ((*p_utf8 & 0x80) == 0)
			len = 1;
		else if ((*p_utf8 & 0xE0) == 0xC0)
			len = 2;
		else if ((*p_utf8 & 0xF0) == 0xE0)
			len = 3;
		else if ((*p_utf8 & 0xF8) == 0xF0)
			len = 4;
		else if ((*p_utf8 & 0xFC) == 0xF8)
			len = 5;
		else if ((*p_utf8 & 0xFE) == 0xFC)
			len = 6;
		else {
			_UNICERROR("invalid len");

			return true; //invalid UTF8
		}

		if (len > cstr_size) {
			_UNICERROR("no space left");
			return true; //not enough space
		}

		if (len == 2 && (*p_utf8 & 0x1E) == 0) {
			//printf("overlong rejected\n");
			_UNICERROR("no space left");
			return true; //reject overlong
		}

		/* Convert the first character */

		uint32_t unichar = 0;

		if (len == 1)
			unichar = *p_utf8;
		else {

			unichar = (0xFF >> (len + 1)) & *p_utf8;

			for (int i = 1; i < len; i++) {

				if ((p_utf8[i] & 0xC0) != 0x80) {
					_UNICERROR("invalid utf8");
					return true; //invalid utf8
				}
				if (unichar == 0 && i == 2 && ((p_utf8[i] & 0x7F) >> (7 - len)) == 0) {
					_UNICERROR("invalid utf8 overlong");
					return true; //no overlong
				}
				unichar = (unichar << 6) | (p_utf8[i] & 0x3F);
			}
		}

		//printf("char %i, len %i\n",unichar,len);
		if (sizeof(wchar_t) == 2 && unichar > 0xFFFF) {
			unichar = ' '; //too long for windows
		}

		*(dst++) = unichar;
		cstr_size -= len;
		p_utf8 += len;
	}

	return false;
}

CharString String::utf8() const {

	int l = length();
	if (!l)
		return CharString();

	const CharType *d = &operator[](0);
	int fl = 0;
	for (int i = 0; i < l; i++) {

		uint32_t c = d[i];
		if (c <= 0x7f) // 7 bits.
			fl += 1;
		else if (c <= 0x7ff) { // 11 bits
			fl += 2;
		} else if (c <= 0xffff) { // 16 bits
			fl += 3;
		} else if (c <= 0x001fffff) { // 21 bits
			fl += 4;

		} else if (c <= 0x03ffffff) { // 26 bits
			fl += 5;
		} else if (c <= 0x7fffffff) { // 31 bits
			fl += 6;
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

		if (c <= 0x7f) // 7 bits.
			APPEND_CHAR(c);
		else if (c <= 0x7ff) { // 11 bits

			APPEND_CHAR(uint32_t(0xc0 | ((c >> 6) & 0x1f))); // Top 5 bits.
			APPEND_CHAR(uint32_t(0x80 | (c & 0x3f))); // Bottom 6 bits.
		} else if (c <= 0xffff) { // 16 bits

			APPEND_CHAR(uint32_t(0xe0 | ((c >> 12) & 0x0f))); // Top 4 bits.
			APPEND_CHAR(uint32_t(0x80 | ((c >> 6) & 0x3f))); // Middle 6 bits.
			APPEND_CHAR(uint32_t(0x80 | (c & 0x3f))); // Bottom 6 bits.
		} else if (c <= 0x001fffff) { // 21 bits

			APPEND_CHAR(uint32_t(0xf0 | ((c >> 18) & 0x07))); // Top 3 bits.
			APPEND_CHAR(uint32_t(0x80 | ((c >> 12) & 0x3f))); // Upper middle 6 bits.
			APPEND_CHAR(uint32_t(0x80 | ((c >> 6) & 0x3f))); // Lower middle 6 bits.
			APPEND_CHAR(uint32_t(0x80 | (c & 0x3f))); // Bottom 6 bits.
		} else if (c <= 0x03ffffff) { // 26 bits

			APPEND_CHAR(uint32_t(0xf8 | ((c >> 24) & 0x03))); // Top 2 bits.
			APPEND_CHAR(uint32_t(0x80 | ((c >> 18) & 0x3f))); // Upper middle 6 bits.
			APPEND_CHAR(uint32_t(0x80 | ((c >> 12) & 0x3f))); // middle 6 bits.
			APPEND_CHAR(uint32_t(0x80 | ((c >> 6) & 0x3f))); // Lower middle 6 bits.
			APPEND_CHAR(uint32_t(0x80 | (c & 0x3f))); // Bottom 6 bits.
		} else if (c <= 0x7fffffff) { // 31 bits

			APPEND_CHAR(uint32_t(0xfc | ((c >> 30) & 0x01))); // Top 1 bit.
			APPEND_CHAR(uint32_t(0x80 | ((c >> 24) & 0x3f))); // Upper upper middle 6 bits.
			APPEND_CHAR(uint32_t(0x80 | ((c >> 18) & 0x3f))); // Lower upper middle 6 bits.
			APPEND_CHAR(uint32_t(0x80 | ((c >> 12) & 0x3f))); // Upper lower middle 6 bits.
			APPEND_CHAR(uint32_t(0x80 | ((c >> 6) & 0x3f))); // Lower lower middle 6 bits.
			APPEND_CHAR(uint32_t(0x80 | (c & 0x3f))); // Bottom 6 bits.
		}
	}
#undef APPEND_CHAR
	*cdst = 0; //trailing zero

	return utf8s;
}

/*
String::String(CharType p_char) {

	shared=NULL;
	copy_from(p_char);
}
*/

String::String(const char *p_str) {

	copy_from(p_str);
}
String::String(const CharType *p_str, int p_clip_to_len) {

	copy_from(p_str, p_clip_to_len);
}

String::String(const StrRange &p_range) {

	if (!p_range.c_str)
		return;

	copy_from(p_range.c_str, p_range.len);
}

int String::hex_to_int(bool p_with_prefix) const {

	int l = length();
	if (p_with_prefix && l < 3)
		return 0;

	const CharType *s = ptr();

	int sign = s[0] == '-' ? -1 : 1;

	if (sign < 0) {
		s++;
		l--;
		if (p_with_prefix && l < 2)
			return 0;
	}

	if (p_with_prefix) {
		if (s[0] != '0' || s[1] != 'x')
			return 0;
		s += 2;
		l -= 2;
	};

	int hex = 0;

	while (*s) {

		CharType c = LOWERCASE(*s);
		int n;
		if (c >= '0' && c <= '9') {
			n = c - '0';
		} else if (c >= 'a' && c <= 'f') {
			n = (c - 'a') + 10;
		} else {
			return 0;
		}

		hex *= 16;
		hex += n;
		s++;
	}

	return hex * sign;
}

int64_t String::hex_to_int64(bool p_with_prefix) const {

	int l = length();
	if (p_with_prefix && l < 3)
		return 0;

	const CharType *s = ptr();

	int64_t sign = s[0] == '-' ? -1 : 1;

	if (sign < 0) {
		s++;
		l--;
		if (p_with_prefix && l < 2)
			return 0;
	}

	if (p_with_prefix) {
		if (s[0] != '0' || s[1] != 'x')
			return 0;
		s += 2;
		l -= 2;
	};

	int64_t hex = 0;

	while (*s) {

		CharType c = LOWERCASE(*s);
		int64_t n;
		if (c >= '0' && c <= '9') {
			n = c - '0';
		} else if (c >= 'a' && c <= 'f') {
			n = (c - 'a') + 10;
		} else {
			return 0;
		}

		hex *= 16;
		hex += n;
		s++;
	}

	return hex * sign;
}

int String::to_int() const {

	if (length() == 0)
		return 0;

	int to = (find(".") >= 0) ? find(".") : length();

	int integer = 0;
	int sign = 1;

	for (int i = 0; i < to; i++) {

		CharType c = operator[](i);
		if (c >= '0' && c <= '9') {

			integer *= 10;
			integer += c - '0';

		} else if (integer == 0 && c == '-') {

			sign = -sign;
		}
	}

	return integer * sign;
}

int64_t String::to_int64() const {

	if (length() == 0)
		return 0;

	int to = (find(".") >= 0) ? find(".") : length();

	int64_t integer = 0;
	int64_t sign = 1;

	for (int i = 0; i < to; i++) {

		CharType c = operator[](i);
		if (c >= '0' && c <= '9') {

			integer *= 10;
			integer += c - '0';

		} else if (integer == 0 && c == '-') {

			sign = -sign;
		}
	}

	return integer * sign;
}

int String::to_int(const char *p_str, int p_len) {

	int to = 0;
	if (p_len >= 0)
		to = p_len;
	else {
		while (p_str[to] != 0 && p_str[to] != '.')
			to++;
	}

	int integer = 0;
	int sign = 1;

	for (int i = 0; i < to; i++) {

		char c = p_str[i];
		if (c >= '0' && c <= '9') {

			integer *= 10;
			integer += c - '0';

		} else if (c == '-' && integer == 0) {

			sign = -sign;
		} else if (c != ' ')
			break;
	}

	return integer * sign;
}

bool String::is_numeric() const {

	if (length() == 0) {
		return false;
	};

	int s = 0;
	if (operator[](0) == '-') ++s;
	bool dot = false;
	for (int i = s; i < length(); i++) {

		CharType c = operator[](i);
		if (c == '.') {
			if (dot) {
				return false;
			};
			dot = true;
		}
		if (c < '0' || c > '9') {
			return false;
		};
	};

	return true; // TODO: Use the parser below for this instead
};

template <class C>
static double built_in_strtod(const C *string, /* A decimal ASCII floating-point number,
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
		C **endPtr = NULL) /* If non-NULL, store terminating Cacter's
				 * address here. */
{

	static const int maxExponent = 511; /* Largest possible base 10 exponent.  Any
					 * exponent larger than this will already
					 * produce underflow or overflow, so there's
					 * no need to worry about additional digits.
					 */
	static const double powersOf10[] = { /* Table giving binary powers of 10.  Entry */
		10., /* is 10^2^i.  Used to convert decimal */
		100., /* exponents into floating-point numbers. */
		1.0e4,
		1.0e8,
		1.0e16,
		1.0e32,
		1.0e64,
		1.0e128,
		1.0e256
	};

	int sign, expSign = false;
	double fraction, dblExp;
	const double *d;
	register const C *p;
	register int c;
	int exp = 0; /* Exponent read from "EX" field. */
	int fracExp = 0; /* Exponent that derives from the fractional
				 * part. Under normal circumstances, it is
				 * the negative of the number of digits in F.
				 * However, if I is very long, the last digits
				 * of I get dropped (otherwise a long I with a
				 * large negative exponent could cause an
				 * unnecessary overflow on I alone). In this
				 * case, fracExp is incremented one for each
				 * dropped digit. */
	int mantSize; /* Number of digits in mantissa. */
	int decPt; /* Number of mantissa digits BEFORE decimal
				 * point. */
	const C *pExp; /* Temporarily holds location of exponent in
				 * string. */

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
		if (!IS_DIGIT(c)) {
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
		if (!IS_DIGIT(CharType(*p))) {
			p = pExp;
			goto done;
		}
		while (IS_DIGIT(CharType(*p))) {
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
	if (endPtr != NULL) {
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

double String::to_double(const char *p_str) {

#ifndef NO_USE_STDLIB
	return built_in_strtod<char>(p_str);
//return atof(p_str); DOES NOT WORK ON ANDROID(??)
#else
	return built_in_strtod<char>(p_str);
#endif
}

float String::to_float() const {

	return to_double();
}

double String::to_double(const CharType *p_str, const CharType **r_end) {

	return built_in_strtod<CharType>(p_str, (CharType **)r_end);
}

int64_t String::to_int(const CharType *p_str, int p_len) {

	if (p_len == 0 || !p_str[0])
		return 0;
	///@todo make more exact so saving and loading does not lose precision

	int64_t integer = 0;
	int64_t sign = 1;
	int reading = READING_SIGN;

	const CharType *str = p_str;
	const CharType *limit = &p_str[p_len];

	while (*str && reading != READING_DONE && str != limit) {

		CharType c = *(str++);
		switch (reading) {
			case READING_SIGN: {
				if (c >= '0' && c <= '9') {
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
			}
			case READING_INT: {

				if (c >= '0' && c <= '9') {

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

double String::to_double() const {

	if (empty())
		return 0;
#ifndef NO_USE_STDLIB
	return built_in_strtod<CharType>(c_str());
//return wcstod(c_str(),NULL); DOES NOT WORK ON ANDROID :(
#else
	return built_in_strtod<CharType>(c_str());
#endif
}

bool operator==(const char *p_chr, const String &p_str) {

	return p_str == p_chr;
}

String operator+(const char *p_chr, const String &p_str) {

	String tmp = p_chr;
	tmp += p_str;
	return tmp;
}
String operator+(CharType p_chr, const String &p_str) {

	return (String::chr(p_chr) + p_str);
}

uint32_t String::hash(const char *p_cstr) {

	uint32_t hashv = 5381;
	uint32_t c;

	while ((c = *p_cstr++))
		hashv = ((hashv << 5) + hashv) + c; /* hash * 33 + c */

	return hashv;
}

uint32_t String::hash(const char *p_cstr, int p_len) {

	uint32_t hashv = 5381;
	for (int i = 0; i < p_len; i++)
		hashv = ((hashv << 5) + hashv) + p_cstr[i]; /* hash * 33 + c */

	return hashv;
}

uint32_t String::hash(const CharType *p_cstr, int p_len) {

	uint32_t hashv = 5381;
	for (int i = 0; i < p_len; i++)
		hashv = ((hashv << 5) + hashv) + p_cstr[i]; /* hash * 33 + c */

	return hashv;
}

uint32_t String::hash(const CharType *p_cstr) {

	uint32_t hashv = 5381;
	uint32_t c;

	while ((c = *p_cstr++))
		hashv = ((hashv << 5) + hashv) + c; /* hash * 33 + c */

	return hashv;
}

uint32_t String::hash() const {

	/* simple djb2 hashing */

	const CharType *chr = c_str();
	uint32_t hashv = 5381;
	uint32_t c;

	while ((c = *chr++))
		hashv = ((hashv << 5) + hashv) + c; /* hash * 33 + c */

	return hashv;
}

uint64_t String::hash64() const {

	/* simple djb2 hashing */

	const CharType *chr = c_str();
	uint64_t hashv = 5381;
	uint64_t c;

	while ((c = *chr++))
		hashv = ((hashv << 5) + hashv) + c; /* hash * 33 + c */

	return hashv;
}

String String::md5_text() const {

	CharString cs = utf8();
	MD5_CTX ctx;
	MD5Init(&ctx);
	MD5Update(&ctx, (unsigned char *)cs.ptr(), cs.length());
	MD5Final(&ctx);
	return String::md5(ctx.digest);
}

String String::sha256_text() const {
	CharString cs = utf8();
	unsigned char hash[32];
	sha256_context ctx;
	sha256_init(&ctx);
	sha256_hash(&ctx, (unsigned char *)cs.ptr(), cs.length());
	sha256_done(&ctx, hash);
	return String::hex_encode_buffer(hash, 32);
}

Vector<uint8_t> String::md5_buffer() const {

	CharString cs = utf8();
	MD5_CTX ctx;
	MD5Init(&ctx);
	MD5Update(&ctx, (unsigned char *)cs.ptr(), cs.length());
	MD5Final(&ctx);

	Vector<uint8_t> ret;
	ret.resize(16);
	for (int i = 0; i < 16; i++) {
		ret[i] = ctx.digest[i];
	};

	return ret;
};

Vector<uint8_t> String::sha256_buffer() const {
	CharString cs = utf8();
	unsigned char hash[32];
	sha256_context ctx;
	sha256_init(&ctx);
	sha256_hash(&ctx, (unsigned char *)cs.ptr(), cs.length());
	sha256_done(&ctx, hash);

	Vector<uint8_t> ret;
	ret.resize(32);
	for (int i = 0; i < 32; i++) {
		ret[i] = hash[i];
	}

	return ret;
}

String String::insert(int p_at_pos, String p_string) const {

	if (p_at_pos < 0)
		return *this;

	if (p_at_pos > length())
		p_at_pos = length();

	String pre;
	if (p_at_pos > 0)
		pre = substr(0, p_at_pos);

	String post;
	if (p_at_pos < length())
		post = substr(p_at_pos, length() - p_at_pos);

	return pre + p_string + post;
}
String String::substr(int p_from, int p_chars) const {

	if (empty() || p_from < 0 || p_from >= length() || p_chars <= 0)
		return "";

	if ((p_from + p_chars) > length()) {

		p_chars = length() - p_from;
	}

	return String(&c_str()[p_from], p_chars);
}

int String::find_last(String p_str) const {

	int pos = -1;
	int findfrom = 0;
	int findres = -1;
	while ((findres = find(p_str, findfrom)) != -1) {

		pos = findres;
		findfrom = pos + 1;
	}

	return pos;
}
int String::find(String p_str, int p_from) const {

	if (p_from < 0)
		return -1;

	int src_len = p_str.length();

	int len = length();

	if (src_len == 0 || len == 0)
		return -1; //wont find anything!

	const CharType *src = c_str();

	for (int i = p_from; i <= (len - src_len); i++) {

		bool found = true;
		for (int j = 0; j < src_len; j++) {

			int read_pos = i + j;

			if (read_pos >= len) {

				ERR_PRINT("read_pos>=len");
				return -1;
			};

			if (src[read_pos] != p_str[j]) {
				found = false;
				break;
			}
		}

		if (found)
			return i;
	}

	return -1;
}

int String::findmk(const Vector<String> &p_keys, int p_from, int *r_key) const {

	if (p_from < 0)
		return -1;
	if (p_keys.size() == 0)
		return -1;

	//int src_len=p_str.length();
	const String *keys = &p_keys[0];
	int key_count = p_keys.size();
	int len = length();

	if (len == 0)
		return -1; //wont find anything!

	const CharType *src = c_str();

	for (int i = p_from; i < len; i++) {

		bool found = true;
		for (int k = 0; k < key_count; k++) {

			found = true;
			if (r_key)
				*r_key = k;
			const CharType *cmp = keys[k].c_str();
			int l = keys[k].length();

			for (int j = 0; j < l; j++) {

				int read_pos = i + j;

				if (read_pos >= len) {

					found = false;
					break;
				};

				if (src[read_pos] != cmp[j]) {
					found = false;
					break;
				}
			}
			if (found)
				break;
		}

		if (found)
			return i;
	}

	return -1;
}

int String::findn(String p_str, int p_from) const {

	if (p_from < 0)
		return -1;

	int src_len = p_str.length();

	if (src_len == 0 || length() == 0)
		return -1; //wont find anything!

	const CharType *srcd = c_str();

	for (int i = p_from; i <= (length() - src_len); i++) {

		bool found = true;
		for (int j = 0; j < src_len; j++) {

			int read_pos = i + j;

			if (read_pos >= length()) {

				ERR_PRINT("read_pos>=length()");
				return -1;
			};

			CharType src = _find_lower(srcd[read_pos]);
			CharType dst = _find_lower(p_str[j]);

			if (src != dst) {
				found = false;
				break;
			}
		}

		if (found)
			return i;
	}

	return -1;
}

int String::rfind(String p_str, int p_from) const {

	// establish a limit
	int limit = length() - p_str.length();
	if (limit < 0)
		return -1;

	// establish a starting point
	if (p_from < 0)
		p_from = limit;
	else if (p_from > limit)
		p_from = limit;

	int src_len = p_str.length();
	int len = length();

	if (src_len == 0 || len == 0)
		return -1; // won't find anything!

	const CharType *src = c_str();

	for (int i = p_from; i >= 0; i--) {

		bool found = true;
		for (int j = 0; j < src_len; j++) {

			int read_pos = i + j;

			if (read_pos >= len) {

				ERR_PRINT("read_pos>=len");
				return -1;
			};

			if (src[read_pos] != p_str[j]) {
				found = false;
				break;
			}
		}

		if (found)
			return i;
	}

	return -1;
}
int String::rfindn(String p_str, int p_from) const {

	// establish a limit
	int limit = length() - p_str.length();
	if (limit < 0)
		return -1;

	// establish a starting point
	if (p_from < 0)
		p_from = limit;
	else if (p_from > limit)
		p_from = limit;

	int src_len = p_str.length();
	int len = length();

	if (src_len == 0 || len == 0)
		return -1; //wont find anything!

	const CharType *src = c_str();

	for (int i = p_from; i >= 0; i--) {

		bool found = true;
		for (int j = 0; j < src_len; j++) {

			int read_pos = i + j;

			if (read_pos >= len) {

				ERR_PRINT("read_pos>=len");
				return -1;
			};

			CharType srcc = _find_lower(src[read_pos]);
			CharType dstc = _find_lower(p_str[j]);

			if (srcc != dstc) {
				found = false;
				break;
			}
		}

		if (found)
			return i;
	}

	return -1;
}

bool String::ends_with(const String &p_string) const {

	int pos = find_last(p_string);
	if (pos == -1)
		return false;
	return pos + p_string.length() == length();
}

bool String::begins_with(const String &p_string) const {

	if (p_string.length() > length())
		return false;

	int l = p_string.length();
	if (l == 0)
		return true;

	const CharType *src = &p_string[0];
	const CharType *str = &operator[](0);

	int i = 0;
	for (; i < l; i++) {

		if (src[i] != str[i])
			return false;
	}

	// only if i == l the p_string matches the beginning
	return i == l;
}
bool String::begins_with(const char *p_string) const {

	int l = length();
	if (l == 0 || !p_string)
		return false;

	const CharType *str = &operator[](0);
	int i = 0;

	while (*p_string && i < l) {

		if (*p_string != str[i])
			return false;
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

bool String::is_subsequence_ofi(const String &p_string) const {

	return _base_is_subsequence_of(p_string, true);
}

bool String::is_quoted() const {

	return is_enclosed_in("\"") || is_enclosed_in("'");
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

	const CharType *src = &operator[](0);
	const CharType *tgt = &p_string[0];

	for (; *src && *tgt; tgt++) {
		bool match = false;
		if (case_insensitive) {
			CharType srcc = _find_lower(*src);
			CharType tgtc = _find_lower(*tgt);
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
		b[i] = substr(i, 2);
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

	float sum = src_size + tgt_size;
	float inter = 0;
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

static bool _wildcard_match(const CharType *p_pattern, const CharType *p_string, bool p_case_sensitive) {
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

	if (!p_wildcard.length() || !length())
		return false;

	return _wildcard_match(p_wildcard.c_str(), c_str(), true);
}

bool String::matchn(const String &p_wildcard) const {

	if (!p_wildcard.length() || !length())
		return false;
	return _wildcard_match(p_wildcard.c_str(), c_str(), false);
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
					String key;

					key = v_key.get_construct_string();
					if (key.left(1) == "\"" && key.right(key.length() - 1) == "\"") {
						key = key.substr(1, key.length() - 2);
					}

					Variant v_val = value_arr[1];
					String val;
					val = v_val.get_construct_string();

					if (val.left(1) == "\"" && val.right(val.length() - 1) == "\"") {
						val = val.substr(1, val.length() - 2);
					}

					new_string = new_string.replacen(placeholder.replace("_", key), val);
				} else {
					ERR_PRINT(String("STRING.format Inner Array size != 2 ").ascii().get_data());
				}
			} else { //Array structure ["RobotGuy","Logis","rookie"]
				Variant v_val = values_arr[i];
				String val;
				val = v_val.get_construct_string();

				if (val.left(1) == "\"" && val.right(val.length() - 1) == "\"") {
					val = val.substr(1, val.length() - 2);
				}

				new_string = new_string.replacen(placeholder.replace("_", i_as_str), val);
			}
		}
	} else if (values.get_type() == Variant::DICTIONARY) {
		Dictionary d = values;
		List<Variant> keys;
		d.get_key_list(&keys);

		for (List<Variant>::Element *E = keys.front(); E; E = E->next()) {
			String key = E->get().get_construct_string();
			String val = d[E->get()].get_construct_string();

			if (key.left(1) == "\"" && key.right(key.length() - 1) == "\"") {
				key = key.substr(1, key.length() - 2);
			}

			if (val.left(1) == "\"" && val.right(val.length() - 1) == "\"") {
				val = val.substr(1, val.length() - 2);
			}

			new_string = new_string.replacen(placeholder.replace("_", key), val);
		}
	} else {
		ERR_PRINT(String("Invalid type: use Array or Dictionary.").ascii().get_data());
	}

	return new_string;
}

String String::replace(String p_key, String p_with) const {

	String new_string;
	int search_from = 0;
	int result = 0;

	while ((result = find(p_key, search_from)) >= 0) {

		new_string += substr(search_from, result - search_from);
		new_string += p_with;
		search_from = result + p_key.length();
	}

	new_string += substr(search_from, length() - search_from);

	return new_string;
}

String String::replace_first(String p_key, String p_with) const {

	String new_string;
	int search_from = 0;
	int result = 0;

	while ((result = find(p_key, search_from)) >= 0) {

		new_string += substr(search_from, result - search_from);
		new_string += p_with;
		search_from = result + p_key.length();
		break;
	}

	new_string += substr(search_from, length() - search_from);

	return new_string;
}
String String::replacen(String p_key, String p_with) const {

	String new_string;
	int search_from = 0;
	int result = 0;

	while ((result = findn(p_key, search_from)) >= 0) {

		new_string += substr(search_from, result - search_from);
		new_string += p_with;
		search_from = result + p_key.length();
	}

	new_string += substr(search_from, length() - search_from);
	return new_string;
}

String String::left(int p_pos) const {

	if (p_pos <= 0)
		return "";

	if (p_pos >= length())
		return *this;

	return substr(0, p_pos);
}

String String::right(int p_pos) const {

	if (p_pos >= size())
		return *this;

	if (p_pos < 0)
		return "";

	return substr(p_pos, (length() - p_pos));
}

CharType String::ord_at(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, length(), 0);
	return operator[](p_idx);
}

String String::strip_edges(bool left, bool right) const {

	int len = length();
	int beg = 0, end = len;

	if (left) {
		for (int i = 0; i < len; i++) {

			if (operator[](i) <= 32)
				beg++;
			else
				break;
		}
	}

	if (right) {
		for (int i = (int)(len - 1); i >= 0; i--) {

			if (operator[](i) <= 32)
				end--;
			else
				break;
		}
	}

	if (beg == 0 && end == len)
		return *this;

	return substr(beg, end - beg);
}

String String::strip_escapes() const {

	int len = length();
	int beg = 0, end = len;

	for (int i = 0; i < length(); i++) {

		if (operator[](i) <= 31)
			beg++;
		else
			break;
	}

	for (int i = (int)(length() - 1); i >= 0; i--) {

		if (operator[](i) <= 31)
			end--;
		else
			break;
	}

	if (beg == 0 && end == len)
		return *this;

	return substr(beg, end - beg);
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
		s = s.substr(6, s.length());
	} else if (s.begins_with("/") || s.begins_with("\\")) {

		drive = s.substr(0, 1);
		s = s.substr(1, s.length() - 1);
	} else {

		int p = s.find(":/");
		if (p == -1)
			p = s.find(":\\");
		if (p != -1 && p < s.find("/")) {

			drive = s.substr(0, p + 2);
			s = s.substr(p + 2, s.length());
		}
	}

	s = s.replace("\\", "/");
	while (true) { // in case of using 2 or more slash
		String compare = s.replace("//", "/");
		if (s == compare)
			break;
		else
			s = compare;
	}
	Vector<String> dirs = s.split("/", false);

	for (int i = 0; i < dirs.size(); i++) {

		String d = dirs[i];
		if (d == ".") {
			dirs.remove(i);
			i--;
		} else if (d == "..") {

			if (i == 0) {
				dirs.remove(i);
				i--;
			} else {
				dirs.remove(i);
				dirs.remove(i - 1);
				i -= 2;
			}
		}
	}

	s = "";

	for (int i = 0; i < dirs.size(); i++) {

		if (i > 0)
			s += "/";
		s += dirs[i];
	}

	return drive + s;
}

static int _humanize_digits(int p_num) {

	if (p_num < 10)
		return 2;
	else if (p_num < 100)
		return 2;
	else if (p_num < 1024)
		return 1;
	else
		return 0;
}

String String::humanize_size(size_t p_size) {

	uint64_t _div = 1;
	static const char *prefix[] = { " Bytes", " KB", " MB", " GB", "TB", " PB", "HB", "" };
	int prefix_idx = 0;

	while (p_size > (_div * 1024) && prefix[prefix_idx][0]) {
		_div *= 1024;
		prefix_idx++;
	}

	int digits = prefix_idx > 0 ? _humanize_digits(p_size / _div) : 0;
	double divisor = prefix_idx > 0 ? _div : 1;

	return String::num(p_size / divisor, digits) + prefix[prefix_idx];
}
bool String::is_abs_path() const {

	if (length() > 1)
		return (operator[](0) == '/' || operator[](0) == '\\' || find(":/") != -1 || find(":\\") != -1);
	else if ((length()) == 1)
		return (operator[](0) == '/' || operator[](0) == '\\');
	else
		return false;
}

bool String::is_valid_identifier() const {

	int len = length();

	if (len == 0)
		return false;

	const wchar_t *str = &operator[](0);

	for (int i = 0; i < len; i++) {

		if (i == 0) {
			if (str[0] >= '0' && str[0] <= '9')
				return false; // no start with number plz
		}

		bool valid_char = (str[i] >= '0' && str[i] <= '9') || (str[i] >= 'a' && str[i] <= 'z') || (str[i] >= 'A' && str[i] <= 'Z') || str[i] == '_';

		if (!valid_char)
			return false;
	}

	return true;
}

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

String String::http_escape() const {
	const CharString temp = utf8();
	String res;
	for (int i = 0; i < length(); ++i) {
		CharType ord = temp[i];
		if (ord == '.' || ord == '-' || ord == '_' || ord == '~' ||
				(ord >= 'a' && ord <= 'z') ||
				(ord >= 'A' && ord <= 'Z') ||
				(ord >= '0' && ord <= '9')) {
			res += ord;
		} else {
			char h_Val[3];
#if defined(__GNUC__) || defined(_MSC_VER)
			snprintf(h_Val, 3, "%.2X", ord);
#else
			sprintf(h_Val, "%.2X", ord);
#endif
			res += "%";
			res += h_Val;
		}
	}
	return res;
}

String String::http_unescape() const {
	String res;
	for (int i = 0; i < length(); ++i) {
		if (ord_at(i) == '%' && i + 2 < length()) {
			CharType ord1 = ord_at(i + 1);
			if ((ord1 >= '0' && ord1 <= '9') || (ord1 >= 'A' && ord1 <= 'Z')) {
				CharType ord2 = ord_at(i + 2);
				if ((ord2 >= '0' && ord2 <= '9') || (ord2 >= 'A' && ord2 <= 'Z')) {
					char bytes[2] = { ord1, ord2 };
					res += (char)strtol(bytes, NULL, 16);
					i += 2;
				}
			} else {
				res += ord_at(i);
			}
		} else {
			res += ord_at(i);
		}
	}
	return String::utf8(res.ascii());
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

static _FORCE_INLINE_ int _xml_unescape(const CharType *p_src, int p_src_len, CharType *p_dst) {

	int len = 0;
	while (p_src_len) {

		if (*p_src == '&') {

			int eat = 0;

			if (p_src_len >= 4 && p_src[1] == '#') {

				CharType c = 0;

				for (int i = 2; i < p_src_len; i++) {

					eat = i + 1;
					CharType ct = p_src[i];
					if (ct == ';') {
						break;
					} else if (ct >= '0' && ct <= '9') {
						ct = ct - '0';
					} else if (ct >= 'a' && ct <= 'f') {
						ct = (ct - 'a') + 10;
					} else if (ct >= 'A' && ct <= 'F') {
						ct = (ct - 'A') + 10;
					} else {
						continue;
					}
					c <<= 4;
					c |= ct;
				}

				if (p_dst)
					*p_dst = c;

			} else if (p_src_len >= 4 && p_src[1] == 'g' && p_src[2] == 't' && p_src[3] == ';') {

				if (p_dst)
					*p_dst = '>';
				eat = 4;
			} else if (p_src_len >= 4 && p_src[1] == 'l' && p_src[2] == 't' && p_src[3] == ';') {

				if (p_dst)
					*p_dst = '<';
				eat = 4;
			} else if (p_src_len >= 5 && p_src[1] == 'a' && p_src[2] == 'm' && p_src[3] == 'p' && p_src[4] == ';') {

				if (p_dst)
					*p_dst = '&';
				eat = 5;
			} else if (p_src_len >= 6 && p_src[1] == 'q' && p_src[2] == 'u' && p_src[3] == 'o' && p_src[4] == 't' && p_src[5] == ';') {

				if (p_dst)
					*p_dst = '"';
				eat = 6;
			} else if (p_src_len >= 6 && p_src[1] == 'a' && p_src[2] == 'p' && p_src[3] == 'o' && p_src[4] == 's' && p_src[5] == ';') {

				if (p_dst)
					*p_dst = '\'';
				eat = 6;
			} else {

				if (p_dst)
					*p_dst = *p_src;
				eat = 1;
			}

			if (p_dst)
				p_dst++;

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
	int len = _xml_unescape(c_str(), l, NULL);
	if (len == 0)
		return String();
	str.resize(len + 1);
	_xml_unescape(c_str(), l, &str[0]);
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

	if (end == 0)
		return s;

	int begin = 0;

	while (begin < end && (s[begin] < '0' || s[begin] > '9')) {
		begin++;
	}

	if (begin >= end)
		return s;

	while (end - begin < p_digits) {

		s = s.insert(begin, "0");
		end++;
	}

	return s;
}

bool String::is_valid_integer() const {

	int len = length();

	if (len == 0)
		return false;

	int from = 0;
	if (len != 1 && (operator[](0) == '+' || operator[](0) == '-'))
		from++;

	for (int i = from; i < len; i++) {

		if (operator[](i) < '0' || operator[](i) > '9')
			return false; // no start with number plz
	}

	return true;
}

bool String::is_valid_hex_number(bool p_with_prefix) const {

	int from = 0;
	int len = length();

	if (len != 1 && (operator[](0) == '+' || operator[](0) == '-'))
		from++;

	if (p_with_prefix) {

		if (len < 2)
			return false;
		if (operator[](from) != '0' || operator[](from + 1) != 'x') {
			return false;
		};
		from += 2;
	};

	for (int i = from; i < len; i++) {

		CharType c = operator[](i);
		if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))
			continue;
		return false;
	};

	return true;
};

bool String::is_valid_float() const {

	int len = length();

	if (len == 0)
		return false;

	int from = 0;
	if (operator[](0) == '+' || operator[](0) == '-') {
		from++;
	}

	//this was pulled out of my ass, i wonder if it's correct...

	bool exponent_found = false;
	bool period_found = false;
	bool sign_found = false;
	bool exponent_values_found = false;
	bool numbers_found = false;

	for (int i = from; i < len; i++) {

		if (operator[](i) >= '0' && operator[](i) <= '9') {

			if (exponent_found)
				exponent_values_found = true;
			else
				numbers_found = true;
		} else if (numbers_found && !exponent_found && operator[](i) == 'e') {
			exponent_found = true;
		} else if (!period_found && !exponent_found && operator[](i) == '.') {
			period_found = true;
		} else if ((operator[](i) == '-' || operator[](i) == '+') && exponent_found && !exponent_values_found && !sign_found) {
			sign_found = true;
		} else
			return false; // no start with number plz
	}

	return numbers_found;
}

String String::path_to_file(const String &p_path) const {

	String src = this->replace("\\", "/").get_base_dir();
	String dst = p_path.replace("\\", "/").get_base_dir();
	String rel = src.path_to(dst);
	if (rel == dst) // failed
		return p_path;
	else
		return rel + p_path.get_file();
}

String String::path_to(const String &p_path) const {

	String src = this->replace("\\", "/");
	String dst = p_path.replace("\\", "/");
	if (!src.ends_with("/"))
		src += "/";
	if (!dst.ends_with("/"))
		dst += "/";

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

		if (src_begin != dst_begin)
			return p_path; //impossible to do this

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
		if (src_dirs.size() == common_parent)
			break;
		if (dst_dirs.size() == common_parent)
			break;
		if (src_dirs[common_parent] != dst_dirs[common_parent])
			break;
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

	if (dir.length() == 0)
		dir = "./";
	return dir;
}

bool String::is_valid_html_color() const {

	return Color::html_is_valid(*this);
}

bool String::is_valid_ip_address() const {

	if (find(":") >= 0) {

		Vector<String> ip = split(":");
		for (int i = 0; i < ip.size(); i++) {

			String n = ip[i];
			if (n.empty())
				continue;
			if (n.is_valid_hex_number(false)) {
				int nint = n.hex_to_int(false);
				if (nint < 0 || nint > 0xffff)
					return false;
				continue;
			};
			if (!n.is_valid_ip_address())
				return false;
		};

	} else {
		Vector<String> ip = split(".");
		if (ip.size() != 4)
			return false;
		for (int i = 0; i < ip.size(); i++) {

			String n = ip[i];
			if (!n.is_valid_integer())
				return false;
			int val = n.to_int();
			if (val < 0 || val > 255)
				return false;
		}
	};

	return true;
}

bool String::is_resource_file() const {

	return begins_with("res://") && find("::") == -1;
}

bool String::is_rel_path() const {

	return !is_abs_path();
}

String String::get_base_dir() const {

	int basepos = find("://");
	String rs;
	String base;
	if (basepos != -1) {
		int end = basepos + 3;
		rs = substr(end, length());
		base = substr(0, end);
	} else {
		if (begins_with("/")) {
			rs = substr(1, length());
			base = "/";
		} else {

			rs = *this;
		}
	}

	int sep = MAX(rs.find_last("/"), rs.find_last("\\"));
	if (sep == -1)
		return base;

	return base + rs.substr(0, sep);
}

String String::get_file() const {

	int sep = MAX(find_last("/"), find_last("\\"));
	if (sep == -1)
		return *this;

	return substr(sep + 1, length());
}

String String::get_extension() const {

	int pos = find_last(".");
	if (pos < 0)
		return *this;

	return substr(pos + 1, length());
}

String String::plus_file(const String &p_file) const {
	if (empty())
		return p_file;
	if (operator[](length() - 1) == '/' || (p_file.size() > 0 && p_file.operator[](0) == '/'))
		return *this + p_file;
	return *this + "/" + p_file;
}

String String::percent_encode() const {

	CharString cs = utf8();
	String encoded;
	for (int i = 0; i < cs.length(); i++) {
		uint8_t c = cs[i];
		if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '~' || c == '.') {

			char p[2] = { (char)c, 0 };
			encoded += p;
		} else {
			char p[4] = { '%', 0, 0, 0 };
			static const char hex[16] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' };

			p[1] = hex[c >> 4];
			p[2] = hex[c & 0xF];
			encoded += p;
		}
	}

	return encoded;
}
String String::percent_decode() const {

	CharString pe;

	CharString cs = utf8();
	for (int i = 0; i < cs.length(); i++) {

		uint8_t c = cs[i];
		if (c == '%' && i < length() - 2) {

			uint8_t a = LOWERCASE(cs[i + 1]);
			uint8_t b = LOWERCASE(cs[i + 2]);

			c = 0;
			if (a >= '0' && a <= '9')
				c = (a - '0') << 4;
			else if (a >= 'a' && a <= 'f')
				c = (a - 'a' + 10) << 4;
			else
				continue;

			uint8_t d = 0;

			if (b >= '0' && b <= '9')
				d = (b - '0');
			else if (b >= 'a' && b <= 'f')
				d = (b - 'a' + 10);
			else
				continue;
			c += d;
			i += 2;
		}
		pe.push_back(c);
	}

	pe.push_back(0);

	return String::utf8(pe.ptr());
}

String String::get_basename() const {

	int pos = find_last(".");
	if (pos < 0)
		return *this;

	return substr(0, pos);
}

String itos(int64_t p_val) {

	return String::num_int64(p_val);
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
		for (int i = 0; i < padding; i++)
			s = s + character;
	}

	return s;
}
// Left-pad with a character.
String String::lpad(int min_length, const String &character) const {
	String s = *this;
	int padding = min_length - s.length();
	if (padding > 0) {
		for (int i = 0; i < padding; i++)
			s = character + s;
	}

	return s;
}

// sprintf is implemented in GDScript via:
//   "fish %s pie" % "frog"
//   "fish %s %d pie" % ["frog", 12]
// In case of an error, the string returned is the error description and "error" is true.
String String::sprintf(const Array &values, bool *error) const {
	String formatted;
	CharType *self = (CharType *)c_str();
	bool in_format = false;
	int value_index = 0;
	int min_chars = 0;
	int min_decimals = 0;
	bool in_decimals = false;
	bool pad_with_zeroes = false;
	bool left_justified = false;
	bool show_sign = false;

	*error = true;

	for (; *self; self++) {
		const CharType c = *self;

		if (in_format) { // We have % - lets see what else we get.
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
						case 'd': base = 10; break;
						case 'o': base = 8; break;
						case 'x': break;
						case 'X':
							base = 16;
							capitalize = true;
							break;
					}
					// Get basic number.
					String str = String::num_int64(ABS(value), base, capitalize);
					int number_len = str.length();

					// Padding.
					String pad_char = pad_with_zeroes ? String("0") : String(" ");
					if (left_justified) {
						str = str.rpad(min_chars, pad_char);
					} else {
						str = str.lpad(min_chars, pad_char);
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
					String str = String::num(value, min_decimals);

					// Pad decimals out.
					str = str.pad_decimals(min_decimals);

					// Show sign
					if (show_sign && value >= 0) {
						str = str.insert(0, "+");
					}

					// Padding
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
							return "unsigned byte integer is lower than maximum";
						} else if (value > 255) {
							return "unsigned byte integer is greater than maximum";
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

	*error = false;
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

#include "translation.h"

#ifdef TOOLS_ENABLED
String TTR(const String &p_text) {

	if (TranslationServer::get_singleton()) {
		return TranslationServer::get_singleton()->tool_translate(p_text);
	}

	return p_text;
}

#endif

String RTR(const String &p_text) {

	if (TranslationServer::get_singleton()) {
		String rtr = TranslationServer::get_singleton()->tool_translate(p_text);
		if (rtr == String() || rtr == p_text) {
			return TranslationServer::get_singleton()->translate(p_text);
		} else {
			return rtr;
		}
	}

	return p_text;
}
