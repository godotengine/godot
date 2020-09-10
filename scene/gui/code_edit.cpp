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

#include "core/string/ustring.h"

static bool _is_whitespace(char32_t c) {
	return c == '\t' || c == ' ';
}

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

/* Delimiters */
// Strings
void CodeEdit::add_string_delimiter(const String &p_start_key, const String &p_end_key, bool p_line_only) {
	_add_delimiter(p_start_key, p_end_key, p_line_only, TYPE_STRING);
}

void CodeEdit::remove_string_delimiter(const String &p_start_key) {
	_remove_delimiter(p_start_key, TYPE_STRING);
}

bool CodeEdit::has_string_delimiter(const String &p_start_key) const {
	return _has_delimiter(p_start_key, TYPE_STRING);
}

void CodeEdit::set_string_delimiters(const TypedArray<String> &p_string_delimiters) {
	_set_delimiters(p_string_delimiters, TYPE_STRING);
}

void CodeEdit::clear_string_delimiters() {
	_clear_delimiters(TYPE_STRING);
}

TypedArray<String> CodeEdit::get_string_delimiters() const {
	return _get_delimiters(TYPE_STRING);
}

int CodeEdit::is_in_string(int p_line, int p_column) const {
	return _is_in_delimiter(p_line, p_column, TYPE_STRING);
}

// Comments
void CodeEdit::add_comment_delimiter(const String &p_start_key, const String &p_end_key, bool p_line_only) {
	_add_delimiter(p_start_key, p_end_key, p_line_only, TYPE_COMMENT);
}

void CodeEdit::remove_comment_delimiter(const String &p_start_key) {
	_remove_delimiter(p_start_key, TYPE_COMMENT);
}

bool CodeEdit::has_comment_delimiter(const String &p_start_key) const {
	return _has_delimiter(p_start_key, TYPE_COMMENT);
}

void CodeEdit::set_comment_delimiters(const TypedArray<String> &p_comment_delimiters) {
	_set_delimiters(p_comment_delimiters, TYPE_COMMENT);
}

void CodeEdit::clear_comment_delimiters() {
	_clear_delimiters(TYPE_COMMENT);
}

TypedArray<String> CodeEdit::get_comment_delimiters() const {
	return _get_delimiters(TYPE_COMMENT);
}

int CodeEdit::is_in_comment(int p_line, int p_column) const {
	return _is_in_delimiter(p_line, p_column, TYPE_COMMENT);
}

String CodeEdit::get_delimiter_start_key(int p_delimiter_idx) const {
	ERR_FAIL_INDEX_V(p_delimiter_idx, delimiters.size(), "");
	return delimiters[p_delimiter_idx].start_key;
}

String CodeEdit::get_delimiter_end_key(int p_delimiter_idx) const {
	ERR_FAIL_INDEX_V(p_delimiter_idx, delimiters.size(), "");
	return delimiters[p_delimiter_idx].end_key;
}

Point2 CodeEdit::get_delimiter_start_position(int p_line, int p_column) const {
	if (delimiters.size() == 0) {
		return Point2(-1, -1);
	}
	ERR_FAIL_INDEX_V(p_line, get_line_count(), Point2(-1, -1));
	ERR_FAIL_COND_V(p_column - 1 > get_line(p_line).size(), Point2(-1, -1));

	Point2 start_position;
	start_position.y = -1;
	start_position.x = -1;

	bool in_region = ((p_line <= 0 || delimiter_cache[p_line - 1].size() < 1) ? -1 : delimiter_cache[p_line - 1].back()->value()) != -1;

	/* Check the keys for this line. */
	for (Map<int, int>::Element *E = delimiter_cache[p_line].front(); E; E = E->next()) {
		if (E->key() > p_column) {
			break;
		}
		in_region = E->value() != -1;
		start_position.x = in_region ? E->key() : -1;
	}

	/* Region was found on this line and is not a multiline continuation. */
	if (start_position.x != -1 && start_position.x != get_line(p_line).length() + 1) {
		start_position.y = p_line;
		return start_position;
	}

	/* Not in a region */
	if (!in_region) {
		return start_position;
	}

	/* Region starts on a previous line */
	for (int i = p_line - 1; i >= 0; i--) {
		if (delimiter_cache[i].size() < 1) {
			continue;
		}
		start_position.y = i;
		start_position.x = delimiter_cache[i].back()->key();

		/* Make sure it's not a multiline continuation. */
		if (start_position.x != get_line(i).length() + 1) {
			break;
		}
	}
	return start_position;
}

Point2 CodeEdit::get_delimiter_end_position(int p_line, int p_column) const {
	if (delimiters.size() == 0) {
		return Point2(-1, -1);
	}
	ERR_FAIL_INDEX_V(p_line, get_line_count(), Point2(-1, -1));
	ERR_FAIL_COND_V(p_column - 1 > get_line(p_line).size(), Point2(-1, -1));

	Point2 end_position;
	end_position.y = -1;
	end_position.x = -1;

	int region = (p_line <= 0 || delimiter_cache[p_line - 1].size() < 1) ? -1 : delimiter_cache[p_line - 1].back()->value();

	/* Check the keys for this line. */
	for (Map<int, int>::Element *E = delimiter_cache[p_line].front(); E; E = E->next()) {
		end_position.x = (E->value() == -1) ? E->key() : -1;
		if (E->key() > p_column) {
			break;
		}
		region = E->value();
	}

	/* Region was found on this line and is not a multiline continuation. */
	if (region != -1 && end_position.x != -1 && (delimiters[region].line_only || end_position.x != get_line(p_line).length() + 1)) {
		end_position.y = p_line;
		return end_position;
	}

	/* Not in a region */
	if (region == -1) {
		end_position.x = -1;
		return end_position;
	}

	/* Region ends on a later line */
	for (int i = p_line + 1; i < get_line_count(); i++) {
		if (delimiter_cache[i].size() < 1 || delimiter_cache[i].front()->value() != -1) {
			continue;
		}
		end_position.x = delimiter_cache[i].front()->key();

		/* Make sure it's not a multiline continuation. */
		if (get_line(i).length() > 0 && end_position.x != get_line(i).length() + 1) {
			end_position.y = i;
			break;
		}
		end_position.x = -1;
	}
	return end_position;
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

	/* Delimiters */
	// Strings
	ClassDB::bind_method(D_METHOD("add_string_delimiter", "start_key", "end_key", "line_only"), &CodeEdit::add_string_delimiter, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("remove_string_delimiter", "start_key"), &CodeEdit::remove_string_delimiter);
	ClassDB::bind_method(D_METHOD("has_string_delimiter", "start_key"), &CodeEdit::has_string_delimiter);

	ClassDB::bind_method(D_METHOD("set_string_delimiters", "string_delimiters"), &CodeEdit::set_string_delimiters);
	ClassDB::bind_method(D_METHOD("clear_string_delimiters"), &CodeEdit::clear_string_delimiters);
	ClassDB::bind_method(D_METHOD("get_string_delimiters"), &CodeEdit::get_string_delimiters);

	ClassDB::bind_method(D_METHOD("is_in_string", "line", "column"), &CodeEdit::is_in_string, DEFVAL(-1));

	// Comments
	ClassDB::bind_method(D_METHOD("add_comment_delimiter", "start_key", "end_key", "line_only"), &CodeEdit::add_comment_delimiter, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("remove_comment_delimiter", "start_key"), &CodeEdit::remove_comment_delimiter);
	ClassDB::bind_method(D_METHOD("has_comment_delimiter", "start_key"), &CodeEdit::has_comment_delimiter);

	ClassDB::bind_method(D_METHOD("set_comment_delimiters", "comment_delimiters"), &CodeEdit::set_comment_delimiters);
	ClassDB::bind_method(D_METHOD("clear_comment_delimiters"), &CodeEdit::clear_comment_delimiters);
	ClassDB::bind_method(D_METHOD("get_comment_delimiters"), &CodeEdit::get_comment_delimiters);

	ClassDB::bind_method(D_METHOD("is_in_comment", "line", "column"), &CodeEdit::is_in_comment, DEFVAL(-1));

	// Util
	ClassDB::bind_method(D_METHOD("get_delimiter_start_key", "delimiter_index"), &CodeEdit::get_delimiter_start_key);
	ClassDB::bind_method(D_METHOD("get_delimiter_end_key", "delimiter_index"), &CodeEdit::get_delimiter_end_key);

	ClassDB::bind_method(D_METHOD("get_delimiter_start_postion", "line", "column"), &CodeEdit::get_delimiter_start_position);
	ClassDB::bind_method(D_METHOD("get_delimiter_end_postion", "line", "column"), &CodeEdit::get_delimiter_end_position);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_breakpoints_gutter"), "set_draw_breakpoints_gutter", "is_drawing_breakpoints_gutter");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_bookmarks"), "set_draw_bookmarks_gutter", "is_drawing_bookmarks_gutter");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_executing_lines"), "set_draw_executing_lines_gutter", "is_drawing_executing_lines_gutter");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_line_numbers"), "set_draw_line_numbers", "is_draw_line_numbers_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "zero_pad_line_numbers"), "set_line_numbers_zero_padded", "is_line_numbers_zero_padded");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_fold_gutter"), "set_draw_fold_gutter", "is_drawing_fold_gutter");

	ADD_GROUP("Delimiters", "delimiter_");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "delimiter_strings"), "set_string_delimiters", "get_string_delimiters");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "delimiter_comments"), "set_comment_delimiters", "get_comment_delimiters");

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

/* Delimiters */
void CodeEdit::_update_delimiter_cache(int p_from_line, int p_to_line) {
	if (delimiters.size() == 0) {
		return;
	}

	int line_count = get_line_count();
	if (p_to_line == -1) {
		p_to_line = line_count;
	}

	int start_line = MIN(p_from_line, p_to_line);
	int end_line = MAX(p_from_line, p_to_line);

	/* Make sure delimiter_cache has all the lines. */
	if (start_line != end_line) {
		if (p_to_line < p_from_line) {
			for (int i = end_line; i > start_line; i--) {
				delimiter_cache.remove(i);
			}
		} else {
			for (int i = start_line; i < end_line; i++) {
				delimiter_cache.insert(i, Map<int, int>());
			}
		}
	}

	int in_region = -1;
	for (int i = start_line; i < MIN(end_line + 1, line_count); i++) {
		int current_end_region = (i <= 0 || delimiter_cache[i].size() < 1) ? -1 : delimiter_cache[i].back()->value();
		in_region = (i <= 0 || delimiter_cache[i - 1].size() < 1) ? -1 : delimiter_cache[i - 1].back()->value();

		const String &str = get_line(i);
		const int line_length = str.length();
		delimiter_cache.write[i].clear();

		if (str.length() == 0) {
			if (in_region != -1) {
				delimiter_cache.write[i][0] = in_region;
			}
			if (i == end_line && current_end_region != in_region) {
				end_line = MIN(end_line++, line_count);
			}
			continue;
		}

		int end_region = -1;
		for (int j = 0; j < line_length; j++) {
			int from = j;
			for (; from < line_length; from++) {
				if (str[from] == '\\') {
					from++;
					continue;
				}
				break;
			}

			/* check if we are in entering a region */
			bool same_line = false;
			if (in_region == -1) {
				for (int d = 0; d < delimiters.size(); d++) {
					/* check there is enough room */
					int chars_left = line_length - from;
					int start_key_length = delimiters[d].start_key.length();
					int end_key_length = delimiters[d].end_key.length();
					if (chars_left < start_key_length) {
						continue;
					}

					/* search the line */
					bool match = true;
					const char32_t *start_key = delimiters[d].start_key.get_data();
					for (int k = 0; k < start_key_length; k++) {
						if (start_key[k] != str[from + k]) {
							match = false;
							break;
						}
					}
					if (!match) {
						continue;
					}
					same_line = true;
					in_region = d;
					delimiter_cache.write[i][from + 1] = d;
					from += start_key_length;

					/* check if it's the whole line */
					if (end_key_length == 0 || delimiters[d].line_only || from + end_key_length > line_length) {
						j = line_length;
						if (delimiters[d].line_only) {
							delimiter_cache.write[i][line_length + 1] = -1;
						} else {
							end_region = in_region;
						}
					}
					break;
				}

				if (j == line_length || in_region == -1) {
					continue;
				}
			}

			/* if we are in one find the end key */
			/* search the line */
			int region_end_index = -1;
			int end_key_length = delimiters[in_region].end_key.length();
			const char32_t *end_key = delimiters[in_region].end_key.get_data();
			for (; from < line_length; from++) {
				if (line_length - from < end_key_length) {
					break;
				}

				if (!is_symbol(str[from])) {
					continue;
				}

				if (str[from] == '\\') {
					from++;
					continue;
				}

				region_end_index = from;
				for (int k = 0; k < end_key_length; k++) {
					if (end_key[k] != str[from + k]) {
						region_end_index = -1;
						break;
					}
				}

				if (region_end_index != -1) {
					break;
				}
			}

			j = from + (end_key_length - 1);
			end_region = (region_end_index == -1) ? in_region : -1;
			if (!same_line || region_end_index != -1) {
				delimiter_cache.write[i][j + 1] = end_region;
			}
			in_region = -1;
		}

		if (i == end_line && current_end_region != end_region) {
			end_line = MIN(end_line++, line_count);
		}
	}
}

int CodeEdit::_is_in_delimiter(int p_line, int p_column, DelimiterType p_type) const {
	if (delimiters.size() == 0) {
		return -1;
	}
	ERR_FAIL_INDEX_V(p_line, get_line_count(), 0);

	int region = (p_line <= 0 || delimiter_cache[p_line - 1].size() < 1) ? -1 : delimiter_cache[p_line - 1].back()->value();
	bool in_region = region != -1 && delimiters[region].type == p_type;
	for (Map<int, int>::Element *E = delimiter_cache[p_line].front(); E; E = E->next()) {
		/* If column is specified, loop untill the key is larger then the column. */
		if (p_column != -1) {
			if (E->key() > p_column) {
				break;
			}
			in_region = E->value() != -1 && delimiters[E->value()].type == p_type;
			region = in_region ? E->value() : -1;
			continue;
		}

		/* If no column, calulate if the entire line is a region       */
		/* excluding whitespace.                                       */
		const String line = get_line(p_line);
		if (!in_region) {
			if (E->value() == -1 || delimiters[E->value()].type != p_type) {
				break;
			}

			region = E->value();
			in_region = true;
			for (int i = E->key() - 2; i >= 0; i--) {
				if (!_is_whitespace(line[i])) {
					return -1;
				}
			}
		}

		if (delimiters[region].line_only) {
			return region;
		}

		int end_col = E->key();
		if (E->value() != -1) {
			if (!E->next()) {
				return region;
			}
			end_col = E->next()->key();
		}

		for (int i = end_col; i < line.length(); i++) {
			if (!_is_whitespace(line[i])) {
				return -1;
			}
		}
		return region;
	}
	return in_region ? region : -1;
}

void CodeEdit::_add_delimiter(const String &p_start_key, const String &p_end_key, bool p_line_only, DelimiterType p_type) {
	if (p_start_key.length() > 0) {
		for (int i = 0; i < p_start_key.length(); i++) {
			ERR_FAIL_COND_MSG(!is_symbol(p_start_key[i]), "delimiter must start with a symbol");
		}
	}

	if (p_end_key.length() > 0) {
		for (int i = 0; i < p_end_key.length(); i++) {
			ERR_FAIL_COND_MSG(!is_symbol(p_end_key[i]), "delimiter must end with a symbol");
		}
	}

	int at = 0;
	for (int i = 0; i < delimiters.size(); i++) {
		ERR_FAIL_COND_MSG(delimiters[i].start_key == p_start_key, "delimiter with start key '" + p_start_key + "' already exists.");
		if (p_start_key.length() < delimiters[i].start_key.length()) {
			at++;
		}
	}

	Delimiter delimiter;
	delimiter.type = p_type;
	delimiter.start_key = p_start_key;
	delimiter.end_key = p_end_key;
	delimiter.line_only = p_line_only || p_end_key == "";
	delimiters.insert(at, delimiter);
	if (!setting_delimiters) {
		delimiter_cache.clear();
		_update_delimiter_cache();
	}
}

void CodeEdit::_remove_delimiter(const String &p_start_key, DelimiterType p_type) {
	for (int i = 0; i < delimiters.size(); i++) {
		if (delimiters[i].start_key != p_start_key) {
			continue;
		}

		if (delimiters[i].type != p_type) {
			break;
		}

		delimiters.remove(i);
		if (!setting_delimiters) {
			delimiter_cache.clear();
			_update_delimiter_cache();
		}
		break;
	}
}

bool CodeEdit::_has_delimiter(const String &p_start_key, DelimiterType p_type) const {
	for (int i = 0; i < delimiters.size(); i++) {
		if (delimiters[i].start_key == p_start_key) {
			return delimiters[i].type == p_type;
		}
	}
	return false;
}

void CodeEdit::_set_delimiters(const TypedArray<String> &p_delimiters, DelimiterType p_type) {
	setting_delimiters = true;
	_clear_delimiters(p_type);

	for (int i = 0; i < p_delimiters.size(); i++) {
		String key = p_delimiters[i].is_null() ? "" : p_delimiters[i];

		const String start_key = key.get_slice(" ", 0);
		const String end_key = key.get_slice_count(" ") > 1 ? key.get_slice(" ", 1) : String();

		_add_delimiter(start_key, end_key, end_key == "", p_type);
	}
	setting_delimiters = false;
	_update_delimiter_cache();
}

void CodeEdit::_clear_delimiters(DelimiterType p_type) {
	for (int i = delimiters.size() - 1; i >= 0; i--) {
		if (delimiters[i].type == p_type) {
			delimiters.remove(i);
		}
	}
	delimiter_cache.clear();
}

TypedArray<String> CodeEdit::_get_delimiters(DelimiterType p_type) const {
	TypedArray<String> r_delimiters;
	for (int i = 0; i < delimiters.size(); i++) {
		if (delimiters[i].type != p_type) {
			continue;
		}
		r_delimiters.push_back(delimiters[i].start_key + (delimiters[i].end_key.empty() ? "" : " " + delimiters[i].end_key));
	}
	return r_delimiters;
}

void CodeEdit::_lines_edited_from(int p_from_line, int p_to_line) {
	_update_delimiter_cache(p_from_line, p_to_line);

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
