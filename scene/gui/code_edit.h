/*************************************************************************/
/*  code_edit.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef CODEEDIT_H
#define CODEEDIT_H

#include "scene/gui/text_edit.h"

class CodeEdit : public TextEdit {
	GDCLASS(CodeEdit, TextEdit)

private:
	/* Main Gutter */
	enum MainGutterType {
		MAIN_GUTTER_BREAKPOINT = 0x01,
		MAIN_GUTTER_BOOKMARK = 0x02,
		MAIN_GUTTER_EXECUTING = 0x04
	};

	int main_gutter = -1;
	void _update_draw_main_gutter();
	void _main_gutter_draw_callback(int p_line, int p_gutter, const Rect2 &p_region);

	// breakpoints
	HashMap<int, bool> breakpointed_lines;
	bool draw_breakpoints = false;
	Color breakpoint_color = Color(1, 1, 1);
	Ref<Texture2D> breakpoint_icon = Ref<Texture2D>();

	// bookmarks
	bool draw_bookmarks = false;
	Color bookmark_color = Color(1, 1, 1);
	Ref<Texture2D> bookmark_icon = Ref<Texture2D>();

	// executing lines
	bool draw_executing_lines = false;
	Color executing_line_color = Color(1, 1, 1);
	Ref<Texture2D> executing_line_icon = Ref<Texture2D>();

	/* Line numbers */
	int line_number_gutter = -1;
	int line_number_digits = 0;
	String line_number_padding = " ";
	Color line_number_color = Color(1, 1, 1);
	void _line_number_draw_callback(int p_line, int p_gutter, const Rect2 &p_region);

	/* Fold Gutter */
	int fold_gutter = -1;
	bool draw_fold_gutter = false;
	Color folding_color = Color(1, 1, 1);
	Ref<Texture2D> can_fold_icon = Ref<Texture2D>();
	Ref<Texture2D> folded_icon = Ref<Texture2D>();
	void _fold_gutter_draw_callback(int p_line, int p_gutter, Rect2 p_region);

	void _gutter_clicked(int p_line, int p_gutter);
	void _lines_edited_from(int p_from_line, int p_to_line);

	void _update_gutter_indexes();

protected:
	void _notification(int p_what);

	static void _bind_methods();

public:
	/* Main Gutter */
	void set_draw_breakpoints_gutter(bool p_draw);
	bool is_drawing_breakpoints_gutter() const;

	void set_draw_bookmarks_gutter(bool p_draw);
	bool is_drawing_bookmarks_gutter() const;

	void set_draw_executing_lines_gutter(bool p_draw);
	bool is_drawing_executing_lines_gutter() const;

	// breakpoints
	void set_line_as_breakpoint(int p_line, bool p_breakpointed);
	bool is_line_breakpointed(int p_line) const;
	void clear_breakpointed_lines();
	Array get_breakpointed_lines() const;

	// bookmarks
	void set_line_as_bookmarked(int p_line, bool p_bookmarked);
	bool is_line_bookmarked(int p_line) const;
	void clear_bookmarked_lines();
	Array get_bookmarked_lines() const;

	// executing lines
	void set_line_as_executing(int p_line, bool p_executing);
	bool is_line_executing(int p_line) const;
	void clear_executing_lines();
	Array get_executing_lines() const;

	/* Line numbers */
	void set_draw_line_numbers(bool p_draw);
	bool is_draw_line_numbers_enabled() const;
	void set_line_numbers_zero_padded(bool p_zero_padded);
	bool is_line_numbers_zero_padded() const;

	/* Fold gutter */
	void set_draw_fold_gutter(bool p_draw);
	bool is_drawing_fold_gutter() const;

	CodeEdit();
	~CodeEdit();
};

#endif // CODEEDIT_H
