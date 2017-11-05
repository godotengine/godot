/*************************************************************************/
/*  text_edit.h                                                          */
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
#ifndef TEXT_UTILS_H
#define TEXT_UTILS_H

class TextUtils {
protected:
	static bool _is_text_char(CharType c) {

		return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
	}

	static bool _is_symbol(CharType c) {

		return c != '_' && ((c >= '!' && c <= '/') || (c >= ':' && c <= '@') || (c >= '[' && c <= '`') || (c >= '{' && c <= '~') || c == '\t' || c == ' ');
	}

	static bool _is_whitespace(CharType c) {
		return c == '\t' || c == ' ';
	}

	static bool _is_char(CharType c) {

		return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
	}

	static bool _is_number(CharType c) {
		return (c >= '0' && c <= '9');
	}

	static bool _is_hex_symbol(CharType c) {
		return ((c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'));
	}

	static bool _is_pair_right_symbol(CharType c) {
		return c == '"' ||
			   c == '\'' ||
			   c == ')' ||
			   c == ']' ||
			   c == '}';
	}

	static bool _is_pair_left_symbol(CharType c) {
		return c == '"' ||
			   c == '\'' ||
			   c == '(' ||
			   c == '[' ||
			   c == '{';
	}

	static bool _is_pair_symbol(CharType c) {
		return _is_pair_left_symbol(c) || _is_pair_right_symbol(c);
	}

	static CharType _get_right_pair_symbol(CharType c) {
		if (c == '"')
			return '"';
		if (c == '\'')
			return '\'';
		if (c == '(')
			return ')';
		if (c == '[')
			return ']';
		if (c == '{')
			return '}';
		return 0;
	}

	static bool _is_completable(CharType c) {

		return !_is_symbol(c) || c == '"' || c == '\'';
	}

	static bool _select_word(const String &s, int &beg, int &end) {
		if (s[beg] > 32 || beg == s.length()) {

			bool symbol = beg < s.length() && _is_symbol(s[beg]); //not sure if right but most editors behave like this

			while (beg > 0 && s[beg - 1] > 32 && (symbol == _is_symbol(s[beg - 1]))) {
				beg--;
			}
			while (end < s.length() && s[end + 1] > 32 && (symbol == _is_symbol(s[end + 1]))) {
				end++;
			}

			if (end < s.length())
				end += 1;

			return true;
		} else {

			return false;
		}
	}
};

#endif // TEXT_UTILS_H
