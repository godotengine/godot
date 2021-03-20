/*************************************************************************/
/*  code_edit.cpp                                                        */
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

#include "code_edit.h"

void CodeEdit::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED:
		case NOTIFICATION_ENTER_TREE: {
			set_gutter_width(main_gutter, get_row_height());
			set_gutter_width(line_number_gutter, (line_number_digits + 1) * cache.font->get_char_size('0', 0, cache.font_size).width);
			set_gutter_width(fold_gutter, get_row_height() / 1.2);

			breakpoint_color = get_theme_color("breakpoint_color");
			breakpoint_icon = get_theme_icon("breakpoint");

			bookmark_color = get_theme_color("bookmark_color");
			bookmark_icon = get_theme_icon("bookmark");

			executing_line_color = get_theme_color("executing_line_color");
			executing_line_icon = get_theme_icon("executing_line");

			line_number_color = get_theme_color("line_number_color");

			folding_color = get_theme_color("code_folding_color");
			can_fold_icon = get_theme_icon("can_fold");
			folded_icon = get_theme_icon("folded");
		} break;
		case NOTIFICATION_DRAW: {
		} break;
	}
}

/* Main Gutter */
void CodeEdit::_update_draw_main_gutter() {
	set_gutter_draw(main_gutter, draw_breakpoints || draw_bookmarks || draw_executing_lines);
}

void CodeEdit::set_draw_breakpoints_gutter(bool p_draw) {
	draw_breakpoints = p_draw;
	set_gutter_clickable(main_gutter, p_draw);
	_update_draw_main_gutter();
}

bool CodeEdit::is_drawing_breakpoints_gutter() const {
	return draw_breakpoints;
}

void CodeEdit::set_draw_bookmarks_gutter(bool p_draw) {
	draw_bookmarks = p_draw;
	_update_draw_main_gutter();
}

bool CodeEdit::is_drawing_bookmarks_gutter() const {
	return draw_bookmarks;
}

void CodeEdit::set_draw_executing_lines_gutter(bool p_draw) {
	draw_executing_lines = p_draw;
	_update_draw_main_gutter();
}

bool CodeEdit::is_drawing_executing_lines_gutter() const {
	return draw_executing_lines;
}

void CodeEdit::_main_gutter_draw_callback(int p_line, int p_gutter, const Rect2 &p_region) {
	if (draw_breakpoints && is_line_breakpointed(p_line)) {
		int padding = p_region.size.x / 6;

		Rect2 breakpoint_region = p_region;
		breakpoint_region.position += Point2(padding, padding);
		breakpoint_region.size -= Point2(padding, padding) * 2;
		breakpoint_icon->draw_rect(get_canvas_item(), breakpoint_region, false, breakpoint_color);
	}

	if (draw_bookmarks && is_line_bookmarked(p_line)) {
		int horizontal_padding = p_region.size.x / 2;
		int vertical_padding = p_region.size.y / 4;

		Rect2 bookmark_region = p_region;
		bookmark_region.position += Point2(horizontal_padding, 0);
		bookmark_region.size -= Point2(horizontal_padding * 1.1, vertical_padding);
		bookmark_icon->draw_rect(get_canvas_item(), bookmark_region, false, bookmark_color);
	}

	if (draw_executing_lines && is_line_executing(p_line)) {
		int horizontal_padding = p_region.size.x / 10;
		int vertical_padding = p_region.size.y / 4;

		Rect2 executing_line_region = p_region;
		executing_line_region.position += Point2(horizontal_padding, vertical_padding);
		executing_line_region.size -= Point2(horizontal_padding, vertical_padding) * 2;
		executing_line_icon->draw_rect(get_canvas_item(), executing_line_region, false, executing_line_color);
	}
}

// Breakpoints
void CodeEdit::set_line_as_breakpoint(int p_line, bool p_breakpointed) {
	int mask = get_line_gutter_metadata(p_line, main_gutter);
	set_line_gutter_metadata(p_line, main_gutter, p_breakpointed ? mask | MAIN_GUTTER_BREAKPOINT : mask & ~MAIN_GUTTER_BREAKPOINT);
	if (p_breakpointed) {
		breakpointed_lines[p_line] = true;
	} else if (breakpointed_lines.has(p_line)) {
		breakpointed_lines.erase(p_line);
	}
	emit_signal("breakpoint_toggled", p_line);
	update();
}

bool CodeEdit::is_line_breakpointed(int p_line) const {
	return (int)get_line_gutter_metadata(p_line, main_gutter) & MAIN_GUTTER_BREAKPOINT;
}

void CodeEdit::clear_breakpointed_lines() {
	for (int i = 0; i < get_line_count(); i++) {
		if (is_line_breakpointed(i)) {
			set_line_as_breakpoint(i, false);
		}
	}
}

Array CodeEdit::get_breakpointed_lines() const {
	Array ret;
	for (int i = 0; i < get_line_count(); i++) {
		if (is_line_breakpointed(i)) {
			ret.append(i);
		}
	}
	return ret;
}

// Bookmarks
void CodeEdit::set_line_as_bookmarked(int p_line, bool p_bookmarked) {
	int mask = get_line_gutter_metadata(p_line, main_gutter);
	set_line_gutter_metadata(p_line, main_gutter, p_bookmarked ? mask | MAIN_GUTTER_BOOKMARK : mask & ~MAIN_GUTTER_BOOKMARK);
	update();
}

bool CodeEdit::is_line_bookmarked(int p_line) const {
	return (int)get_line_gutter_metadata(p_line, main_gutter) & MAIN_GUTTER_BOOKMARK;
}

void CodeEdit::clear_bookmarked_lines() {
	for (int i = 0; i < get_line_count(); i++) {
		if (is_line_bookmarked(i)) {
			set_line_as_bookmarked(i, false);
		}
	}
}

Array CodeEdit::get_bookmarked_lines() const {
	Array ret;
	for (int i = 0; i < get_line_count(); i++) {
		if (is_line_bookmarked(i)) {
			ret.append(i);
		}
	}
	return ret;
}

// executing lines
void CodeEdit::set_line_as_executing(int p_line, bool p_executing) {
	int mask = get_line_gutter_metadata(p_line, main_gutter);
	set_line_gutter_metadata(p_line, main_gutter, p_executing ? mask | MAIN_GUTTER_EXECUTING : mask & ~MAIN_GUTTER_EXECUTING);
	update();
}

bool CodeEdit::is_line_executing(int p_line) const {
	return (int)get_line_gutter_metadata(p_line, main_gutter) & MAIN_GUTTER_EXECUTING;
}

void CodeEdit::clear_executing_lines() {
	for (int i = 0; i < get_line_count(); i++) {
		if (is_line_executing(i)) {
			set_line_as_executing(i, false);
		}
	}
}

Array CodeEdit::get_executing_lines() const {
	Array ret;
	for (int i = 0; i < get_line_count(); i++) {
		if (is_line_executing(i)) {
			ret.append(i);
		}
	}
	return ret;
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
	String fc = TS->format_number(String::num(p_line + 1).lpad(line_number_digits, line_number_padding));
	Ref<TextLine> tl;
	tl.instance();
	tl->add_string(fc, cache.font, cache.font_size);
	int yofs = p_region.position.y + (get_row_height() - tl->get_size().y) / 2;
	Color number_color = get_line_gutter_item_color(p_line, line_number_gutter);
	if (number_color == Color(1, 1, 1)) {
		number_color = line_number_color;
	}
	tl->draw(get_canvas_item(), Point2(p_region.position.x, yofs), number_color);
}

/* Fold Gutter */
void CodeEdit::set_draw_fold_gutter(bool p_draw) {
	set_gutter_draw(fold_gutter, p_draw);
}

bool CodeEdit::is_drawing_fold_gutter() const {
	return is_gutter_drawn(fold_gutter);
}

void CodeEdit::_fold_gutter_draw_callback(int p_line, int p_gutter, Rect2 p_region) {
	if (!can_fold(p_line) && !is_folded(p_line)) {
		set_line_gutter_clickable(p_line, fold_gutter, false);
		return;
	}
	set_line_gutter_clickable(p_line, fold_gutter, true);

	int horizontal_padding = p_region.size.x / 10;
	int vertical_padding = p_region.size.y / 6;

	p_region.position += Point2(horizontal_padding, vertical_padding);
	p_region.size -= Point2(horizontal_padding, vertical_padding) * 2;

	if (can_fold(p_line)) {
		can_fold_icon->draw_rect(get_canvas_item(), p_region, false, folding_color);
		return;
	}
	folded_icon->draw_rect(get_canvas_item(), p_region, false, folding_color);
}

void CodeEdit::_bind_methods() {
	/* Main Gutter */
	ClassDB::bind_method(D_METHOD("_main_gutter_draw_callback"), &CodeEdit::_main_gutter_draw_callback);

	ClassDB::bind_method(D_METHOD("set_draw_breakpoints_gutter", "enable"), &CodeEdit::set_draw_breakpoints_gutter);
	ClassDB::bind_method(D_METHOD("is_drawing_breakpoints_gutter"), &CodeEdit::is_drawing_breakpoints_gutter);

	ClassDB::bind_method(D_METHOD("set_draw_bookmarks_gutter", "enable"), &CodeEdit::set_draw_bookmarks_gutter);
	ClassDB::bind_method(D_METHOD("is_drawing_bookmarks_gutter"), &CodeEdit::is_drawing_bookmarks_gutter);

	ClassDB::bind_method(D_METHOD("set_draw_executing_lines_gutter", "enable"), &CodeEdit::set_draw_executing_lines_gutter);
	ClassDB::bind_method(D_METHOD("is_drawing_executing_lines_gutter"), &CodeEdit::is_drawing_executing_lines_gutter);

	// Breakpoints
	ClassDB::bind_method(D_METHOD("set_line_as_breakpoint", "line", "breakpointed"), &CodeEdit::set_line_as_breakpoint);
	ClassDB::bind_method(D_METHOD("is_line_breakpointed", "line"), &CodeEdit::is_line_breakpointed);
	ClassDB::bind_method(D_METHOD("clear_breakpointed_lines"), &CodeEdit::clear_breakpointed_lines);
	ClassDB::bind_method(D_METHOD("get_breakpointed_lines"), &CodeEdit::get_breakpointed_lines);

	// Bookmarks
	ClassDB::bind_method(D_METHOD("set_line_as_bookmarked", "line", "bookmarked"), &CodeEdit::set_line_as_bookmarked);
	ClassDB::bind_method(D_METHOD("is_line_bookmarked", "line"), &CodeEdit::is_line_bookmarked);
	ClassDB::bind_method(D_METHOD("clear_bookmarked_lines"), &CodeEdit::clear_bookmarked_lines);
	ClassDB::bind_method(D_METHOD("get_bookmarked_lines"), &CodeEdit::get_bookmarked_lines);

	// executing lines
	ClassDB::bind_method(D_METHOD("set_line_as_executing", "line", "executing"), &CodeEdit::set_line_as_executing);
	ClassDB::bind_method(D_METHOD("is_line_executing", "line"), &CodeEdit::is_line_executing);
	ClassDB::bind_method(D_METHOD("clear_executing_lines"), &CodeEdit::clear_executing_lines);
	ClassDB::bind_method(D_METHOD("get_executing_lines"), &CodeEdit::get_executing_lines);

	/* Line numbers */
	ClassDB::bind_method(D_METHOD("_line_number_draw_callback"), &CodeEdit::_line_number_draw_callback);

	ClassDB::bind_method(D_METHOD("set_draw_line_numbers", "enable"), &CodeEdit::set_draw_line_numbers);
	ClassDB::bind_method(D_METHOD("is_draw_line_numbers_enabled"), &CodeEdit::is_draw_line_numbers_enabled);
	ClassDB::bind_method(D_METHOD("set_line_numbers_zero_padded", "enable"), &CodeEdit::set_line_numbers_zero_padded);
	ClassDB::bind_method(D_METHOD("is_line_numbers_zero_padded"), &CodeEdit::is_line_numbers_zero_padded);

	/* Fold Gutter */
	ClassDB::bind_method(D_METHOD("_fold_gutter_draw_callback"), &CodeEdit::_fold_gutter_draw_callback);

	ClassDB::bind_method(D_METHOD("set_draw_fold_gutter", "enable"), &CodeEdit::set_draw_fold_gutter);
	ClassDB::bind_method(D_METHOD("is_drawing_fold_gutter"), &CodeEdit::is_drawing_fold_gutter);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_breakpoints_gutter"), "set_draw_breakpoints_gutter", "is_drawing_breakpoints_gutter");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_bookmarks"), "set_draw_bookmarks_gutter", "is_drawing_bookmarks_gutter");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_executing_lines"), "set_draw_executing_lines_gutter", "is_drawing_executing_lines_gutter");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_line_numbers"), "set_draw_line_numbers", "is_draw_line_numbers_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "zero_pad_line_numbers"), "set_line_numbers_zero_padded", "is_line_numbers_zero_padded");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_fold_gutter"), "set_draw_fold_gutter", "is_drawing_fold_gutter");

	ADD_SIGNAL(MethodInfo("breakpoint_toggled", PropertyInfo(Variant::INT, "line")));
}

void CodeEdit::_gutter_clicked(int p_line, int p_gutter) {
	if (p_gutter == main_gutter) {
		if (draw_breakpoints) {
			set_line_as_breakpoint(p_line, !is_line_breakpointed(p_line));
		}
		return;
	}

	if (p_gutter == line_number_gutter) {
		set_selection_mode(TextEdit::SelectionMode::SELECTION_MODE_LINE, p_line, 0);
		select(p_line, 0, p_line + 1, 0);
		cursor_set_line(p_line + 1);
		cursor_set_column(0);
		return;
	}

	if (p_gutter == fold_gutter) {
		if (is_folded(p_line)) {
			unfold_line(p_line);
		} else if (can_fold(p_line)) {
			fold_line(p_line);
		}
		return;
	}
}

void CodeEdit::_lines_edited_from(int p_from_line, int p_to_line) {
	if (p_from_line == p_to_line) {
		return;
	}

	int lc = get_line_count();
	line_number_digits = 1;
	while (lc /= 10) {
		line_number_digits++;
	}
	set_gutter_width(line_number_gutter, (line_number_digits + 1) * cache.font->get_char_size('0', 0, cache.font_size).width);

	int from_line = MIN(p_from_line, p_to_line);
	int line_count = (p_to_line - p_from_line);
	List<int> breakpoints;
	breakpointed_lines.get_key_list(&breakpoints);
	for (const List<int>::Element *E = breakpoints.front(); E; E = E->next()) {
		int line = E->get();
		if (line <= from_line) {
			continue;
		}
		breakpointed_lines.erase(line);

		emit_signal("breakpoint_toggled", line);
		if (line_count > 0 || line >= p_from_line) {
			emit_signal("breakpoint_toggled", line + line_count);
			breakpointed_lines[line + line_count] = true;
			continue;
		}
	}
}

void CodeEdit::_update_gutter_indexes() {
	for (int i = 0; i < get_gutter_count(); i++) {
		if (get_gutter_name(i) == "main_gutter") {
			main_gutter = i;
			continue;
		}

		if (get_gutter_name(i) == "line_numbers") {
			line_number_gutter = i;
			continue;
		}

		if (get_gutter_name(i) == "fold_gutter") {
			fold_gutter = i;
			continue;
		}
	}
}

CodeEdit::CodeEdit() {
	/* Text Direction */
	set_layout_direction(LAYOUT_DIRECTION_LTR);
	set_text_direction(TEXT_DIRECTION_LTR);

	/* Gutters */
	int gutter_idx = 0;

	/* Main Gutter */
	add_gutter();
	set_gutter_name(gutter_idx, "main_gutter");
	set_gutter_draw(gutter_idx, false);
	set_gutter_overwritable(gutter_idx, true);
	set_gutter_type(gutter_idx, GUTTER_TPYE_CUSTOM);
	set_gutter_custom_draw(gutter_idx, this, "_main_gutter_draw_callback");
	gutter_idx++;

	/* Line numbers */
	add_gutter();
	set_gutter_name(gutter_idx, "line_numbers");
	set_gutter_draw(gutter_idx, false);
	set_gutter_type(gutter_idx, GUTTER_TPYE_CUSTOM);
	set_gutter_custom_draw(gutter_idx, this, "_line_number_draw_callback");
	gutter_idx++;

	/* Fold Gutter */
	add_gutter();
	set_gutter_name(gutter_idx, "fold_gutter");
	set_gutter_draw(gutter_idx, false);
	set_gutter_type(gutter_idx, GUTTER_TPYE_CUSTOM);
	set_gutter_custom_draw(gutter_idx, this, "_fold_gutter_draw_callback");
	gutter_idx++;

	connect("lines_edited_from", callable_mp(this, &CodeEdit::_lines_edited_from));
	connect("gutter_clicked", callable_mp(this, &CodeEdit::_gutter_clicked));

	connect("gutter_added", callable_mp(this, &CodeEdit::_update_gutter_indexes));
	connect("gutter_removed", callable_mp(this, &CodeEdit::_update_gutter_indexes));
	_update_gutter_indexes();
}

CodeEdit::~CodeEdit() {
}
