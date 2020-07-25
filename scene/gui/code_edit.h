/*************************************************************************/
/*  code_edit.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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
	int cached_line_count = 0;

	/* Line numbers */
	int line_number_gutter = -1;
	int line_number_digits = 0;
	String line_number_padding = " ";
	Color line_number_color = Color(1, 1, 1);
	void _line_number_draw_callback(int p_line, int p_gutter, const Rect2 &p_region);

	void _gutter_clicked(int p_line, int p_gutter);
	void _line_edited_from(int p_line);

	void _update_gutter_indexes();

protected:
	void _notification(int p_what);

	static void _bind_methods();

public:
	/* Line numbers */
	void set_draw_line_numbers(bool p_draw);
	bool is_draw_line_numbers_enabled() const;
	void set_line_numbers_zero_padded(bool p_zero_padded);
	bool is_line_numbers_zero_padded() const;

	CodeEdit();
	~CodeEdit();
};

#endif // CODEEDIT_H
