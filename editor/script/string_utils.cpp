/**************************************************************************/
/*  string_utils.cpp                                                      */
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

#include "core/object/editor_language.h"
#include "core/string/string_builder.h"
#include "core/string/ustring.h"

String EditorStringUtils::insert(const Vector<String> &p_str, EditorLanguage::Position p_position, const String &p_what) {
	StringBuilder builder;

	for (uint_fast32_t i = 0; i < static_cast<uint_fast32_t>(p_str.size()); i++) {
		if (i == p_position.line) {
			builder += p_str[i].insert(p_position.column, p_what);
		} else {
			builder += p_str[i];
		}
	}

	return builder.as_string();
}

String EditorStringUtils::insert(const String &p_str, EditorLanguage::Position p_position, const String &p_what) {
	return EditorStringUtils::insert(p_str.split("\n"), p_position, p_what);
}

EditorLanguage::Position EditorStringUtils::find_char(const String &p_str, char32_t p_chr) {
	uint_fast32_t line = 0;
	uint_fast32_t column = 0;
	for (int i = 0; i < p_str.length(); i++) {
		if (p_str[i] == p_chr) {
			return EditorLanguage::Position(line, column);
		} else if (p_str[i] == '\n') {
			column = 0;
			line += 1;
		} else {
			column += 1;
		}
	}
	ERR_FAIL_V(EditorLanguage::Position(0, 0));
}
