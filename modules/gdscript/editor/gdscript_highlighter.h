/*************************************************************************/
/*  gdscript_highlighter.h                                               */
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

#ifndef GDSCRIPT_HIGHLIGHTER_H
#define GDSCRIPT_HIGHLIGHTER_H

#include "editor/plugins/script_editor_plugin.h"
#include "scene/gui/text_edit.h"

class GDScriptSyntaxHighlighter : public EditorSyntaxHighlighter {
	GDCLASS(GDScriptSyntaxHighlighter, EditorSyntaxHighlighter)

private:
	struct ColorRegion {
		Color color;
		String start_key;
		String end_key;
		bool line_only = false;
	};
	Vector<ColorRegion> color_regions;
	Map<int, int> color_region_cache;

	HashMap<StringName, Color> keywords;
	HashMap<StringName, Color> member_keywords;

	enum Type {
		NONE,
		REGION,
		NODE_PATH,
		ANNOTATION,
		SYMBOL,
		NUMBER,
		FUNCTION,
		SIGNAL,
		KEYWORD,
		MEMBER,
		IDENTIFIER,
		TYPE,
	};

	// colours
	Color font_color;
	Color symbol_color;
	Color function_color;
	Color function_definition_color;
	Color built_in_type_color;
	Color number_color;
	Color member_color;
	Color node_path_color;
	Color annotation_color;
	Color type_color;

	void add_color_region(const String &p_start_key, const String &p_end_key, const Color &p_color, bool p_line_only = false);

public:
	virtual void _update_cache() override;
	virtual Dictionary _get_line_syntax_highlighting_impl(int p_line) override;

	virtual String _get_name() const override;
	virtual Array _get_supported_languages() const override;

	virtual Ref<EditorSyntaxHighlighter> _create() const override;
};

#endif // GDSCRIPT_HIGHLIGHTER_H
