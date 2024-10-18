/**************************************************************************/
/*  gdscript_highlighter.h                                                */
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

#ifndef GDSCRIPT_HIGHLIGHTER_H
#define GDSCRIPT_HIGHLIGHTER_H

#include "editor/plugins/script_editor_plugin.h"
#include "scene/gui/text_edit.h"

class GDScriptSyntaxHighlighter : public EditorSyntaxHighlighter {
	GDCLASS(GDScriptSyntaxHighlighter, EditorSyntaxHighlighter)

private:
	struct ColorRegion {
		enum Type {
			TYPE_NONE,
			TYPE_STRING, // `"` and `'`, optional prefix `&`, `^`, or `r`.
			TYPE_MULTILINE_STRING, // `"""` and `'''`, optional prefix `r`.
			TYPE_COMMENT, // `#` and `##`.
			TYPE_CODE_REGION, // `#region` and `#endregion`.
		};

		Type type = TYPE_NONE;
		Color color;
		SyntaxHighlighter::SyntaxFontStyle style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR;
		String start_key;
		String end_key;
		bool line_only = false;
		bool r_prefix = false;
		bool is_string = false; // `TYPE_STRING` or `TYPE_MULTILINE_STRING`.
		bool is_comment = false; // `TYPE_COMMENT` or `TYPE_CODE_REGION`.
	};
	Vector<ColorRegion> color_regions;
	HashMap<int, int> color_region_cache;

	struct ColorRec {
		Color color;
		SyntaxHighlighter::SyntaxFontStyle style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR;
	};
	HashMap<StringName, ColorRec> class_names;
	HashMap<StringName, ColorRec> reserved_keywords;
	HashMap<StringName, ColorRec> member_keywords;
	HashSet<StringName> global_functions;

	enum Type {
		NONE,
		REGION,
		NODE_PATH,
		NODE_REF,
		ANNOTATION,
		STRING_NAME,
		SYMBOL,
		NUMBER,
		FUNCTION,
		SIGNAL,
		KEYWORD,
		MEMBER,
		IDENTIFIER,
		TYPE,
	};

	// Colors.
	Color font_color;
	SyntaxHighlighter::SyntaxFontStyle font_style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR;
	Color symbol_color;
	SyntaxHighlighter::SyntaxFontStyle symbol_style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR;
	Color function_color;
	SyntaxHighlighter::SyntaxFontStyle function_style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR;
	Color global_function_color;
	SyntaxHighlighter::SyntaxFontStyle global_function_style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR;
	Color function_definition_color;
	SyntaxHighlighter::SyntaxFontStyle function_definition_style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR;
	Color built_in_type_color;
	SyntaxHighlighter::SyntaxFontStyle built_in_type_style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR;
	Color number_color;
	SyntaxHighlighter::SyntaxFontStyle number_style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR;
	Color member_color;
	SyntaxHighlighter::SyntaxFontStyle member_style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR;
	Color string_color;
	SyntaxHighlighter::SyntaxFontStyle string_style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR;
	Color node_path_color;
	SyntaxHighlighter::SyntaxFontStyle node_path_style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR;
	Color node_ref_color;
	SyntaxHighlighter::SyntaxFontStyle node_ref_style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR;
	Color annotation_color;
	SyntaxHighlighter::SyntaxFontStyle annotation_style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR;
	Color string_name_color;
	SyntaxHighlighter::SyntaxFontStyle string_name_style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR;
	Color type_color;
	SyntaxHighlighter::SyntaxFontStyle type_style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR;

	enum CommentMarkerLevel {
		COMMENT_MARKER_CRITICAL,
		COMMENT_MARKER_WARNING,
		COMMENT_MARKER_NOTICE,
		COMMENT_MARKER_MAX,
	};
	ColorRec comment_marker_colors[COMMENT_MARKER_MAX];
	HashMap<String, CommentMarkerLevel> comment_markers;

	void add_color_region(ColorRegion::Type p_type, const String &p_start_key, const String &p_end_key, const Color &p_color, SyntaxHighlighter::SyntaxFontStyle p_style = SyntaxHighlighter::SYNTAX_STYLE_REGULAR, bool p_line_only = false, bool p_r_prefix = false);

public:
	virtual void _update_cache() override;
	virtual Dictionary _get_line_syntax_highlighting_impl(int p_line) override;

	virtual String _get_name() const override;
	virtual PackedStringArray _get_supported_languages() const override;

	virtual Ref<EditorSyntaxHighlighter> _create() const override;
};

#endif // GDSCRIPT_HIGHLIGHTER_H
