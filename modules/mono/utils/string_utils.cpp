/*************************************************************************/
/*  string_utils.cpp                                                     */
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
#include "string_utils.h"

namespace {

int sfind(const String &p_text, int p_from) {
	if (p_from < 0)
		return -1;

	int src_len = 2;
	int len = p_text.length();

	if (src_len == 0 || len == 0)
		return -1;

	const CharType *src = p_text.c_str();

	for (int i = p_from; i <= (len - src_len); i++) {
		bool found = true;

		for (int j = 0; j < src_len; j++) {
			int read_pos = i + j;

			if (read_pos >= len) {
				ERR_PRINT("read_pos >= len");
				return -1;
			};

			switch (j) {
				case 0:
					found = src[read_pos] == '%';
					break;
				case 1: {
					CharType c = src[read_pos];
					found = src[read_pos] == 's' || (c >= '0' || c <= '4');
					break;
				}
				default:
					found = false;
			}

			if (!found) {
				break;
			}
		}

		if (found)
			return i;
	}

	return -1;
}
} // namespace

String sformat(const String &p_text, const Variant &p1, const Variant &p2, const Variant &p3, const Variant &p4, const Variant &p5) {
	if (p_text.length() < 2)
		return p_text;

	Array args;

	if (p1.get_type() != Variant::NIL) {
		args.push_back(p1);

		if (p2.get_type() != Variant::NIL) {
			args.push_back(p2);

			if (p3.get_type() != Variant::NIL) {
				args.push_back(p3);

				if (p4.get_type() != Variant::NIL) {
					args.push_back(p4);

					if (p5.get_type() != Variant::NIL) {
						args.push_back(p5);
					}
				}
			}
		}
	}

	String new_string;

	int findex = 0;
	int search_from = 0;
	int result = 0;

	while ((result = sfind(p_text, search_from)) >= 0) {
		CharType c = p_text[result + 1];

		int req_index = (c == 's' ? findex++ : c - '0');

		new_string += p_text.substr(search_from, result - search_from);
		new_string += args[req_index].operator String();
		search_from = result + 2;
	}

	new_string += p_text.substr(search_from, p_text.length() - search_from);

	return new_string;
}

bool is_csharp_keyword(const String &p_name) {

	// Reserved keywords

	return p_name == "abstract" || p_name == "as" || p_name == "base" || p_name == "bool" ||
		   p_name == "break" || p_name == "byte" || p_name == "case" || p_name == "catch" ||
		   p_name == "char" || p_name == "checked" || p_name == "class" || p_name == "const" ||
		   p_name == "continue" || p_name == "decimal" || p_name == "default" || p_name == "delegate" ||
		   p_name == "do" || p_name == "double" || p_name == "else" || p_name == "enum" ||
		   p_name == "event" || p_name == "explicit" || p_name == "extern" || p_name == "false" ||
		   p_name == "finally" || p_name == "fixed" || p_name == "float" || p_name == "for" ||
		   p_name == "forech" || p_name == "goto" || p_name == "if" || p_name == "implicit" ||
		   p_name == "in" || p_name == "int" || p_name == "interface" || p_name == "internal" ||
		   p_name == "is" || p_name == "lock" || p_name == "long" || p_name == "namespace" ||
		   p_name == "new" || p_name == "null" || p_name == "object" || p_name == "operator" ||
		   p_name == "out" || p_name == "override" || p_name == "params" || p_name == "private" ||
		   p_name == "protected" || p_name == "public" || p_name == "readonly" || p_name == "ref" ||
		   p_name == "return" || p_name == "sbyte" || p_name == "sealed" || p_name == "short" ||
		   p_name == "sizeof" || p_name == "stackalloc" || p_name == "static" || p_name == "string" ||
		   p_name == "struct" || p_name == "switch" || p_name == "this" || p_name == "throw" ||
		   p_name == "true" || p_name == "try" || p_name == "typeof" || p_name == "uint" || p_name == "ulong" ||
		   p_name == "unchecked" || p_name == "unsafe" || p_name == "ushort" || p_name == "using" ||
		   p_name == "virtual" || p_name == "volatile" || p_name == "void" || p_name == "while";
}

String escape_csharp_keyword(const String &p_name) {
	return is_csharp_keyword(p_name) ? "@" + p_name : p_name;
}
