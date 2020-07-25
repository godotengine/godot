/*************************************************************************/
/*  code_edit.cpp                                                        */
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

#include "code_edit.h"

void CodeEdit::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED:
		case NOTIFICATION_ENTER_TREE: {
			set_gutter_width(line_number_gutter, (line_number_digits + 1) * cache.font->get_char_size('0').width);

			line_number_color = get_theme_color("line_number_color");
		} break;
		case NOTIFICATION_DRAW: {
		} break;
	}
}

/* Line numbers */
void CodeEdit::set_draw_line_numbers(bool p_draw) {
	set_gutter_draw(line_number_gutter, p_draw);
}

bool CodeEdit::is_draw_line_numbers_enabled() const {
	return is_gutter_drawn(line_number_gutter);
}

void CodeEdit::set_line_numbers_zero_padded(bool p_zero_padded) {
	p_zero_padded ? line_number_padding = "0" : line_number_padding = " ";
	update();
}

bool CodeEdit::is_line_numbers_zero_padded() const {
	return line_number_padding == "0";
}

void CodeEdit::_line_number_draw_callback(int p_line, int p_gutter, const Rect2 &p_region) {
	String fc = String::num(p_line + 1).lpad(line_number_digits, line_number_padding);

	int yofs = region.position.y + (cache.row_height - cache.font->get_height()) / 2;
	cache.font->draw(get_canvas_item(), Point2(region.position.x, yofs + cache.font->get_ascent()), fc, line_number_color);
}

void CodeEdit::_bind_methods() {
	/* Line numbers */
	ClassDB::bind_method(D_METHOD("_line_number_draw_callback"), &CodeEdit::_line_number_draw_callback);

	ClassDB::bind_method(D_METHOD("set_draw_line_numbers", "enable"), &CodeEdit::set_draw_line_numbers);
	ClassDB::bind_method(D_METHOD("is_draw_line_numbers_enabled"), &CodeEdit::is_draw_line_numbers_enabled);
	ClassDB::bind_method(D_METHOD("set_line_numbers_zero_padded", "enable"), &CodeEdit::set_line_numbers_zero_padded);
	ClassDB::bind_method(D_METHOD("is_line_numbers_zero_padded"), &CodeEdit::is_line_numbers_zero_padded);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_line_numbers"), "set_draw_line_numbers", "is_draw_line_numbers_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "zero_pad_line_numbers"), "set_line_numbers_zero_padded", "is_line_numbers_zero_padded");
}

void CodeEdit::_gutter_clicked(int p_line, int p_gutter) {
	if (p_gutter == line_number_gutter) {
		cursor_set_line(p_line);
		return;
	}
}

void CodeEdit::_line_edited_from(int p_line) {
	int line_count = get_line_count();
	if (line_count != cached_line_count) {
		int lc = line_count;
		line_number_digits = 1;
		while (lc /= 10) {
			line_number_digits++;
		}
		set_gutter_width(line_number_gutter, (line_number_digits + 1) * cache.font->get_char_size('0').width);

		cached_line_count = line_count;
	}
}

void CodeEdit::_update_gutter_indexes() {
	for (int i = 0; i < get_gutter_count(); i++) {
		if (get_gutter_name(i) == "line_numbers") {
			line_number_gutter = i;
			continue;
		}
	}
}

CodeEdit::CodeEdit() {
	/* Line numbers */
	add_gutter();
	set_gutter_name(0, "line_numbers");
	set_gutter_draw(0, false);
	set_gutter_type(0, GUTTER_TPYE_CUSTOM);
	set_gutter_custom_draw(0, this, "_line_number_draw_callback");

	connect("line_edited_from", callable_mp(this, &CodeEdit::_line_edited_from));
	connect("gutter_clicked", callable_mp(this, &CodeEdit::_gutter_clicked));

	connect("gutter_added", callable_mp(this, &CodeEdit::_update_gutter_indexes));
	connect("gutter_removed", callable_mp(this, &CodeEdit::_update_gutter_indexes));
	_update_gutter_indexes();
}

CodeEdit::~CodeEdit() {
}
