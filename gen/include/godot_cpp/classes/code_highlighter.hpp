/**************************************************************************/
/*  code_highlighter.hpp                                                  */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/syntax_highlighter.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/dictionary.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class String;

class CodeHighlighter : public SyntaxHighlighter {
	GDEXTENSION_CLASS(CodeHighlighter, SyntaxHighlighter)

public:
	void add_keyword_color(const String &p_keyword, const Color &p_color);
	void remove_keyword_color(const String &p_keyword);
	bool has_keyword_color(const String &p_keyword) const;
	Color get_keyword_color(const String &p_keyword) const;
	void set_keyword_colors(const Dictionary &p_keywords);
	void clear_keyword_colors();
	Dictionary get_keyword_colors() const;
	void add_member_keyword_color(const String &p_member_keyword, const Color &p_color);
	void remove_member_keyword_color(const String &p_member_keyword);
	bool has_member_keyword_color(const String &p_member_keyword) const;
	Color get_member_keyword_color(const String &p_member_keyword) const;
	void set_member_keyword_colors(const Dictionary &p_member_keyword);
	void clear_member_keyword_colors();
	Dictionary get_member_keyword_colors() const;
	void add_color_region(const String &p_start_key, const String &p_end_key, const Color &p_color, bool p_line_only = false);
	void remove_color_region(const String &p_start_key);
	bool has_color_region(const String &p_start_key) const;
	void set_color_regions(const Dictionary &p_color_regions);
	void clear_color_regions();
	Dictionary get_color_regions() const;
	void set_function_color(const Color &p_color);
	Color get_function_color() const;
	void set_number_color(const Color &p_color);
	Color get_number_color() const;
	void set_symbol_color(const Color &p_color);
	Color get_symbol_color() const;
	void set_member_variable_color(const Color &p_color);
	Color get_member_variable_color() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		SyntaxHighlighter::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

