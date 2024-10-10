/**************************************************************************/
/*  syntax_highlighter.h                                                  */
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

#ifndef SYNTAX_HIGHLIGHTER_H
#define SYNTAX_HIGHLIGHTER_H

#include "core/io/resource.h"
#include "core/object/gdvirtual.gen.inc"

class TextEdit;

class SyntaxHighlighter : public Resource {
	GDCLASS(SyntaxHighlighter, Resource)
public:
	enum SyntaxFontStyle {
		SYNTAX_STYLE_REGULAR,
		SYNTAX_STYLE_BOLD,
		SYNTAX_STYLE_ITALIC,
		SYNTAX_STYLE_BOLD_ITALIC,
	};

private:
	RBMap<int, Dictionary> highlighting_cache;
	void _lines_edited_from(int p_from_line, int p_to_line);

protected:
	ObjectID text_edit_instance_id; // For validity check
	TextEdit *text_edit = nullptr;

	static void _bind_methods();

	GDVIRTUAL1RC(Dictionary, _get_line_syntax_highlighting, int)
	GDVIRTUAL0(_clear_highlighting_cache)
	GDVIRTUAL0(_update_cache)
public:
	Dictionary get_line_syntax_highlighting(int p_line);
	virtual Dictionary _get_line_syntax_highlighting_impl(int p_line) { return Dictionary(); }

	void clear_line_highlighting_cache(int p_line);

	void clear_highlighting_cache();
	virtual void _clear_highlighting_cache() {}

	void update_cache();
	virtual void _update_cache() {}

	void set_text_edit(TextEdit *p_text_edit);
	TextEdit *get_text_edit() const;

	SyntaxHighlighter() {}
	virtual ~SyntaxHighlighter() {}
};

VARIANT_ENUM_CAST(SyntaxHighlighter::SyntaxFontStyle);

///////////////////////////////////////////////////////////////////////////////

class CodeHighlighter : public SyntaxHighlighter {
	GDCLASS(CodeHighlighter, SyntaxHighlighter)

private:
	struct ColorRegion {
		Color color;
		SyntaxHighlighter::SyntaxFontStyle style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR;
		bool text_segment = false;
		String start_key;
		String end_key;
		bool line_only = false;
	};
	Vector<ColorRegion> color_regions;
	HashMap<int, int> color_region_cache;

	struct ColorRec {
		Color color;
		SyntaxHighlighter::SyntaxFontStyle style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR;
	};
	HashMap<String, ColorRec> keywords;
	HashMap<String, ColorRec> member_keywords;

	Color font_color;
	SyntaxHighlighter::SyntaxFontStyle font_style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR;
	Color member_color;
	SyntaxHighlighter::SyntaxFontStyle member_style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR;
	Color function_color;
	SyntaxHighlighter::SyntaxFontStyle function_style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR;
	Color symbol_color;
	SyntaxHighlighter::SyntaxFontStyle symbol_style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR;
	Color number_color;
	SyntaxHighlighter::SyntaxFontStyle number_style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR;

	bool uint_suffix_enabled = false;

protected:
	static void _bind_methods();

public:
	virtual Dictionary _get_line_syntax_highlighting_impl(int p_line) override;

	virtual void _clear_highlighting_cache() override;
	virtual void _update_cache() override;

#ifndef DISABLE_DEPRECATED
	void add_keyword_color(const String &p_keyword, const Color &p_color);
#endif
	void add_keyword(const String &p_keyword, const Color &p_color, SyntaxHighlighter::SyntaxFontStyle p_style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR);
	void remove_keyword(const String &p_keyword);
	bool has_keyword(const String &p_keyword) const;

	Color get_keyword_color(const String &p_keyword) const;
	SyntaxHighlighter::SyntaxFontStyle get_keyword_style(const String &p_member_keyword) const;

#ifndef DISABLE_DEPRECATED
	void set_keyword_colors(const Dictionary &p_keywords);
	Dictionary get_keyword_colors() const;
#endif
	void set_keywords(const Dictionary &p_keywords);
	void clear_keywords();
	Dictionary get_keywords() const;

#ifndef DISABLE_DEPRECATED
	void add_member_keyword_color(const String &p_member_keyword, const Color &p_color);
#endif
	void add_member_keyword(const String &p_keyword, const Color &p_color, SyntaxHighlighter::SyntaxFontStyle p_style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR);
	void remove_member_keyword(const String &p_keyword);
	bool has_member_keyword(const String &p_keyword) const;

	Color get_member_keyword_color(const String &p_member_keyword) const;
	SyntaxHighlighter::SyntaxFontStyle get_member_keyword_style(const String &p_member_keyword) const;

#ifndef DISABLE_DEPRECATED
	void set_member_keyword_colors(const Dictionary &p_color_regions);
	Dictionary get_member_keyword_colors() const;
#endif
	void set_member_keywords(const Dictionary &p_color_regions);
	void clear_member_keywords();
	Dictionary get_member_keywords() const;

#ifndef DISABLE_DEPRECATED
	void add_color_region(const String &p_start_key, const String &p_end_key, const Color &p_color, bool p_line_only = false);

	void set_color_regions(const Dictionary &p_member_keyword);
	Dictionary get_color_regions() const;
#endif
	void add_region(const String &p_start_key, const String &p_end_key, const Color &p_color, SyntaxHighlighter::SyntaxFontStyle p_style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR, bool p_line_only = false, bool p_is_text_segment = false);
	void remove_region(const String &p_start_key);
	bool has_region(const String &p_start_key) const;

	void set_regions(const Dictionary &p_regions);
	void clear_regions();
	Dictionary get_regions() const;

	void set_number_color(Color p_color);
	Color get_number_color() const;
	void set_number_style(SyntaxHighlighter::SyntaxFontStyle p_style);
	SyntaxHighlighter::SyntaxFontStyle get_number_style() const;

	void set_symbol_color(Color p_color);
	Color get_symbol_color() const;
	void set_symbol_style(SyntaxHighlighter::SyntaxFontStyle p_style);
	SyntaxHighlighter::SyntaxFontStyle get_symbol_style() const;

	void set_function_color(Color p_color);
	Color get_function_color() const;
	void set_function_style(SyntaxHighlighter::SyntaxFontStyle p_style);
	SyntaxHighlighter::SyntaxFontStyle get_function_style() const;

	void set_member_variable_color(Color p_color);
	Color get_member_variable_color() const;
	void set_member_variable_style(SyntaxHighlighter::SyntaxFontStyle p_style);
	SyntaxHighlighter::SyntaxFontStyle get_member_variable_style() const;

	void set_uint_suffix_enabled(bool p_enabled);
};

#endif // SYNTAX_HIGHLIGHTER_H
