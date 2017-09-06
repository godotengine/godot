/*************************************************************************/
/*  code_editor.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "code_editor.h"

#include "editor/editor_scale.h"
#include "editor_node.h"
#include "editor_settings.h"
#include "os/keyboard.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/separator.h"
#include "scene/resources/dynamic_font.h"

void GotoLineDialog::popup_find_line(TextEdit *p_edit) {

	text_editor = p_edit;

	line->set_text(itos(text_editor->cursor_get_line()));
	line->select_all();
	popup_centered(Size2(180, 80));
	line->grab_focus();
}

int GotoLineDialog::get_line() const {

	return line->get_text().to_int();
}

void GotoLineDialog::ok_pressed() {

	if (get_line() < 1 || get_line() > text_editor->get_line_count())
		return;
	text_editor->cursor_set_line(get_line() - 1);
	hide();
}

GotoLineDialog::GotoLineDialog() {

	set_title(TTR("Go to Line"));
	Label *l = memnew(Label);
	l->set_text(TTR("Line Number:"));
	l->set_position(Point2(5, 5));
	add_child(l);

	line = memnew(LineEdit);
	line->set_anchor(MARGIN_RIGHT, ANCHOR_END);
	line->set_begin(Point2(15, 22));
	line->set_end(Point2(-15, 35));
	add_child(line);
	register_text_enter(line);
	text_editor = NULL;

	set_hide_on_ok(false);
}

void FindReplaceBar::_notification(int p_what) {

	if (p_what == NOTIFICATION_READY) {

		find_prev->set_icon(get_icon("MoveUp", "EditorIcons"));
		find_next->set_icon(get_icon("MoveDown", "EditorIcons"));
		hide_button->set_normal_texture(get_icon("Close", "EditorIcons"));
		hide_button->set_hover_texture(get_icon("CloseHover", "EditorIcons"));
		hide_button->set_pressed_texture(get_icon("Close", "EditorIcons"));

	} else if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {

		set_process_unhandled_input(is_visible_in_tree());
	} else if (p_what == EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED) {

		find_prev->set_icon(get_icon("MoveUp", "EditorIcons"));
		find_next->set_icon(get_icon("MoveDown", "EditorIcons"));
		hide_button->set_normal_texture(get_icon("Close", "EditorIcons"));
		hide_button->set_hover_texture(get_icon("CloseHover", "EditorIcons"));
		hide_button->set_pressed_texture(get_icon("Close", "EditorIcons"));
	}
}

void FindReplaceBar::_unhandled_input(const Ref<InputEvent> &p_event) {

	Ref<InputEventKey> k = p_event;
	if (k.is_valid()) {

		if (k->is_pressed() && (text_edit->has_focus() || text_vbc->is_a_parent_of(get_focus_owner()))) {

			bool accepted = true;

			switch (k->get_scancode()) {

				case KEY_ESCAPE: {

					_hide_bar();
				} break;
				default: {

					accepted = false;
				} break;
			}

			if (accepted) {
				accept_event();
			}
		}
	}
}

bool FindReplaceBar::_search(uint32_t p_flags, int p_from_line, int p_from_col) {

	int line, col;
	String text = get_search_text();

	bool found = text_edit->search(text, p_flags, p_from_line, p_from_col, line, col);

	if (found) {
		if (!preserve_cursor) {
			text_edit->cursor_set_line(line, false);
			text_edit->cursor_set_column(col + text.length(), false);
			text_edit->center_viewport_to_cursor();
		}

		text_edit->set_search_text(text);
		text_edit->set_search_flags(p_flags);
		text_edit->set_current_search_result(line, col);

		result_line = line;
		result_col = col;

		set_error("");
	} else {
		result_line = -1;
		result_col = -1;
		text_edit->set_search_text("");
		set_error(text.empty() ? "" : TTR("No Matches"));
	}

	return found;
}

void FindReplaceBar::_replace() {

	if (result_line != -1 && result_col != -1) {
		text_edit->begin_complex_operation();

		text_edit->select(result_line, result_col, result_line, result_col + get_search_text().length());
		text_edit->insert_text_at_cursor(get_replace_text());

		text_edit->end_complex_operation();
	}

	search_current();
}

void FindReplaceBar::_replace_all() {

	// line as x so it gets priority in comparison, column as y
	Point2i orig_cursor(text_edit->cursor_get_line(), text_edit->cursor_get_column());
	Point2i prev_match = Point2(-1, -1);

	bool selection_enabled = text_edit->is_selection_active();
	Point2i selection_begin, selection_end;
	if (selection_enabled) {
		selection_begin = Point2i(text_edit->get_selection_from_line(), text_edit->get_selection_from_column());
		selection_end = Point2i(text_edit->get_selection_to_line(), text_edit->get_selection_to_column());
	}

	int vsval = text_edit->get_v_scroll();

	text_edit->cursor_set_line(0);
	text_edit->cursor_set_column(0);

	String replace_text = get_replace_text();
	int search_text_len = get_search_text().length();

	int rc = 0;

	replace_all_mode = true;

	text_edit->begin_complex_operation();

	while (search_next()) {

		// replace area
		Point2i match_from(result_line, result_col);
		Point2i match_to(result_line, result_col + search_text_len);

		if (match_from < prev_match)
			break; // done

		prev_match = Point2i(result_line, result_col + replace_text.length());

		text_edit->select(result_line, result_col, result_line, match_to.y);

		if (selection_enabled && is_selection_only()) {

			if (match_from < selection_begin || match_to > selection_end)
				continue;

			// replace but adjust selection bounds
			text_edit->insert_text_at_cursor(replace_text);
			if (match_to.x == selection_end.x)
				selection_end.y += replace_text.length() - search_text_len;
		} else {
			// just replace
			text_edit->insert_text_at_cursor(replace_text);
		}

		rc++;
	}

	text_edit->end_complex_operation();

	replace_all_mode = false;

	// restore editor state (selection, cursor, scroll)
	text_edit->cursor_set_line(orig_cursor.x);
	text_edit->cursor_set_column(orig_cursor.y);

	if (selection_enabled && is_selection_only()) {
		// reselect
		text_edit->select(selection_begin.x, selection_begin.y, selection_end.x, selection_end.y);
	} else {
		text_edit->deselect();
	}

	text_edit->set_v_scroll(vsval);
	set_error(vformat(TTR("Replaced %d occurrence(s)."), rc));
}

void FindReplaceBar::_get_search_from(int &r_line, int &r_col) {

	r_line = text_edit->cursor_get_line();
	r_col = text_edit->cursor_get_column();

	if (text_edit->is_selection_active() && !replace_all_mode) {

		int selection_line = text_edit->get_selection_from_line();

		if (text_edit->get_selection_text() == get_search_text() && r_line == selection_line) {

			int selection_from_col = text_edit->get_selection_from_column();

			if (r_col >= selection_from_col && r_col <= text_edit->get_selection_to_column()) {
				r_col = selection_from_col;
			}
		}
	}

	if (r_line == result_line && r_col >= result_col && r_col <= result_col + get_search_text().length()) {
		r_col = result_col;
	}
}

bool FindReplaceBar::search_current() {

	uint32_t flags = 0;

	if (is_whole_words())
		flags |= TextEdit::SEARCH_WHOLE_WORDS;
	if (is_case_sensitive())
		flags |= TextEdit::SEARCH_MATCH_CASE;

	int line, col;
	_get_search_from(line, col);

	return _search(flags, line, col);
}

bool FindReplaceBar::search_prev() {

	uint32_t flags = 0;
	String text = get_search_text();

	if (is_whole_words())
		flags |= TextEdit::SEARCH_WHOLE_WORDS;
	if (is_case_sensitive())
		flags |= TextEdit::SEARCH_MATCH_CASE;

	flags |= TextEdit::SEARCH_BACKWARDS;

	int line, col;
	_get_search_from(line, col);

	col -= text.length();
	if (col < 0) {
		line -= 1;
		if (line < 0)
			line = text_edit->get_line_count() - 1;
		col = text_edit->get_line(line).length();
	}

	return _search(flags, line, col);
}

bool FindReplaceBar::search_next() {

	uint32_t flags = 0;
	String text = get_search_text();

	if (is_whole_words())
		flags |= TextEdit::SEARCH_WHOLE_WORDS;
	if (is_case_sensitive())
		flags |= TextEdit::SEARCH_MATCH_CASE;

	int line, col;
	_get_search_from(line, col);

	if (line == result_line && col == result_col) {
		col += text.length();
		if (col > text_edit->get_line(line).length()) {
			line += 1;
			if (line >= text_edit->get_line_count())
				line = 0;
			col = 0;
		}
	}

	return _search(flags, line, col);
}

void FindReplaceBar::_hide_bar() {

	if (replace_text->has_focus() || search_text->has_focus())
		text_edit->grab_focus();

	text_edit->set_search_text("");
	result_line = -1;
	result_col = -1;
	replace_hbc->hide();
	replace_options_hbc->hide();
	hide();
}

void FindReplaceBar::_show_search() {

	show();
	search_text->grab_focus();

	if (text_edit->is_selection_active() && !selection_only->is_pressed()) {
		search_text->set_text(text_edit->get_selection_text());
	}

	if (!get_search_text().empty()) {
		search_text->select_all();
		search_text->set_cursor_pos(search_text->get_text().length());
		search_current();
	}
}

void FindReplaceBar::popup_search() {

	replace_hbc->hide();
	replace_options_hbc->hide();
	_show_search();
}

void FindReplaceBar::popup_replace() {

	if (!replace_hbc->is_visible_in_tree() || !replace_options_hbc->is_visible_in_tree()) {
		replace_text->clear();
		replace_hbc->show();
		replace_options_hbc->show();
	}

	selection_only->set_pressed((text_edit->is_selection_active() && text_edit->get_selection_from_line() < text_edit->get_selection_to_line()));

	_show_search();
}

void FindReplaceBar::_search_options_changed(bool p_pressed) {

	search_current();
}

void FindReplaceBar::_editor_text_changed() {

	if (is_visible_in_tree()) {
		preserve_cursor = true;
		search_current();
		preserve_cursor = false;
	}
}

void FindReplaceBar::_search_text_changed(const String &p_text) {

	search_current();
}

void FindReplaceBar::_search_text_entered(const String &p_text) {

	search_next();
}

void FindReplaceBar::_replace_text_entered(const String &p_text) {

	if (selection_only->is_pressed() && text_edit->is_selection_active()) {
		_replace_all();
		_hide_bar();
	}
}

String FindReplaceBar::get_search_text() const {

	return search_text->get_text();
}

String FindReplaceBar::get_replace_text() const {

	return replace_text->get_text();
}

bool FindReplaceBar::is_case_sensitive() const {

	return case_sensitive->is_pressed();
}

bool FindReplaceBar::is_whole_words() const {

	return whole_words->is_pressed();
}

bool FindReplaceBar::is_selection_only() const {

	return selection_only->is_pressed();
}

void FindReplaceBar::set_error(const String &p_label) {

	error_label->set_text(p_label);
}

void FindReplaceBar::set_text_edit(TextEdit *p_text_edit) {

	text_edit = p_text_edit;
	text_edit->connect("text_changed", this, "_editor_text_changed");
}

void FindReplaceBar::_bind_methods() {

	ClassDB::bind_method("_unhandled_input", &FindReplaceBar::_unhandled_input);

	ClassDB::bind_method("_editor_text_changed", &FindReplaceBar::_editor_text_changed);
	ClassDB::bind_method("_search_text_changed", &FindReplaceBar::_search_text_changed);
	ClassDB::bind_method("_search_text_entered", &FindReplaceBar::_search_text_entered);
	ClassDB::bind_method("_replace_text_entered", &FindReplaceBar::_replace_text_entered);
	ClassDB::bind_method("_search_current", &FindReplaceBar::search_current);
	ClassDB::bind_method("_search_next", &FindReplaceBar::search_next);
	ClassDB::bind_method("_search_prev", &FindReplaceBar::search_prev);
	ClassDB::bind_method("_replace_pressed", &FindReplaceBar::_replace);
	ClassDB::bind_method("_replace_all_pressed", &FindReplaceBar::_replace_all);
	ClassDB::bind_method("_search_options_changed", &FindReplaceBar::_search_options_changed);
	ClassDB::bind_method("_hide_pressed", &FindReplaceBar::_hide_bar);

	ADD_SIGNAL(MethodInfo("search"));
}

FindReplaceBar::FindReplaceBar() {

	replace_all_mode = false;
	preserve_cursor = false;

	text_vbc = memnew(VBoxContainer);
	add_child(text_vbc);

	HBoxContainer *search_hbc = memnew(HBoxContainer);
	text_vbc->add_child(search_hbc);

	search_text = memnew(LineEdit);
	search_hbc->add_child(search_text);
	search_text->set_custom_minimum_size(Size2(200, 0));
	search_text->connect("text_changed", this, "_search_text_changed");
	search_text->connect("text_entered", this, "_search_text_entered");

	find_prev = memnew(ToolButton);
	search_hbc->add_child(find_prev);
	find_prev->set_focus_mode(FOCUS_NONE);
	find_prev->connect("pressed", this, "_search_prev");

	find_next = memnew(ToolButton);
	search_hbc->add_child(find_next);
	find_next->set_focus_mode(FOCUS_NONE);
	find_next->connect("pressed", this, "_search_next");

	replace_hbc = memnew(HBoxContainer);
	text_vbc->add_child(replace_hbc);
	replace_hbc->hide();

	replace_text = memnew(LineEdit);
	replace_hbc->add_child(replace_text);
	replace_text->set_custom_minimum_size(Size2(200, 0));
	replace_text->connect("text_entered", this, "_replace_text_entered");

	replace = memnew(Button);
	replace_hbc->add_child(replace);
	replace->set_text(TTR("Replace"));
	//replace->set_focus_mode(FOCUS_NONE);
	replace->connect("pressed", this, "_replace_pressed");

	replace_all = memnew(Button);
	replace_hbc->add_child(replace_all);
	replace_all->set_text(TTR("Replace All"));
	//replace_all->set_focus_mode(FOCUS_NONE);
	replace_all->connect("pressed", this, "_replace_all_pressed");

	Control *spacer_split = memnew(Control);
	spacer_split->set_custom_minimum_size(Size2(0, 1));
	text_vbc->add_child(spacer_split);

	VBoxContainer *options_vbc = memnew(VBoxContainer);
	add_child(options_vbc);
	options_vbc->set_h_size_flags(SIZE_EXPAND_FILL);

	HBoxContainer *search_options = memnew(HBoxContainer);
	options_vbc->add_child(search_options);

	case_sensitive = memnew(CheckBox);
	search_options->add_child(case_sensitive);
	case_sensitive->set_text(TTR("Match Case"));
	case_sensitive->set_focus_mode(FOCUS_NONE);
	case_sensitive->connect("toggled", this, "_search_options_changed");

	whole_words = memnew(CheckBox);
	search_options->add_child(whole_words);
	whole_words->set_text(TTR("Whole Words"));
	whole_words->set_focus_mode(FOCUS_NONE);
	whole_words->connect("toggled", this, "_search_options_changed");

	error_label = memnew(Label);
	search_options->add_child(error_label);
	error_label->add_color_override("font_color", EditorNode::get_singleton()->get_gui_base()->get_color("error_color", "Editor"));

	search_options->add_spacer();

	hide_button = memnew(TextureButton);
	search_options->add_child(hide_button);
	hide_button->set_focus_mode(FOCUS_NONE);
	hide_button->connect("pressed", this, "_hide_pressed");

	replace_options_hbc = memnew(HBoxContainer);
	options_vbc->add_child(replace_options_hbc);
	replace_options_hbc->hide();

	selection_only = memnew(CheckBox);
	replace_options_hbc->add_child(selection_only);
	selection_only->set_text(TTR("Selection Only"));
	selection_only->set_focus_mode(FOCUS_NONE);
	selection_only->connect("toggled", this, "_search_options_changed");
}

void FindReplaceDialog::popup_search() {

	set_title(TTR("Search"));
	replace_mc->hide();
	replace_label->hide();
	replace_vb->hide();
	skip->hide();
	popup_centered(Point2(300, 190));
	get_ok()->set_text(TTR("Find"));
	search_text->grab_focus();
	if (text_edit->is_selection_active() && (text_edit->get_selection_from_line() == text_edit->get_selection_to_line())) {

		search_text->set_text(text_edit->get_selection_text());
	}
	search_text->select_all();

	error_label->set_text("");
}

void FindReplaceDialog::popup_replace() {

	set_title(TTR("Replace"));
	bool do_selection = (text_edit->is_selection_active() && text_edit->get_selection_from_line() < text_edit->get_selection_to_line());

	set_replace_selection_only(do_selection);

	if (!do_selection && text_edit->is_selection_active()) {
		search_text->set_text(text_edit->get_selection_text());
	}

	replace_mc->show();
	replace_label->show();
	replace_vb->show();
	popup_centered(Point2(300, 300));
	if (search_text->get_text() != "" && replace_text->get_text() == "") {
		search_text->select(0, 0);
		replace_text->grab_focus();
	} else {
		search_text->grab_focus();
		search_text->select_all();
	}
	error_label->set_text("");

	if (prompt->is_pressed()) {
		skip->show();
		get_ok()->set_text(TTR("Next"));
		selection_only->set_disabled(true);

	} else {
		skip->hide();
		get_ok()->set_text(TTR("Replace"));
		selection_only->set_disabled(false);
	}
}

void FindReplaceDialog::_search_callback() {

	if (is_replace_mode())
		_replace();
	else
		_search();
}

void FindReplaceDialog::_replace_skip_callback() {

	_search();
}

void FindReplaceDialog::_replace() {

	text_edit->begin_complex_operation();
	if (is_replace_all_mode()) {

		//line as x so it gets priority in comparison, column as y
		Point2i orig_cursor(text_edit->cursor_get_line(), text_edit->cursor_get_column());
		Point2i prev_match = Point2(-1, -1);

		bool selection_enabled = text_edit->is_selection_active();
		Point2i selection_begin, selection_end;
		if (selection_enabled) {
			selection_begin = Point2i(text_edit->get_selection_from_line(), text_edit->get_selection_from_column());
			selection_end = Point2i(text_edit->get_selection_to_line(), text_edit->get_selection_to_column());
		}
		int vsval = text_edit->get_v_scroll();
		//int hsval = text_edit->get_h_scroll();

		text_edit->cursor_set_line(0);
		text_edit->cursor_set_column(0);

		int rc = 0;

		while (_search()) {

			if (!text_edit->is_selection_active()) {
				//search selects
				break;
			}

			//replace area
			Point2i match_from(text_edit->get_selection_from_line(), text_edit->get_selection_from_column());
			Point2i match_to(text_edit->get_selection_to_line(), text_edit->get_selection_to_column());

			if (match_from < prev_match)
				break; //done

			prev_match = match_to;

			if (selection_enabled && is_replace_selection_only()) {

				if (match_from < selection_begin || match_to > selection_end)
					continue;

				//replace but adjust selection bounds

				text_edit->insert_text_at_cursor(get_replace_text());
				if (match_to.x == selection_end.x)
					selection_end.y += get_replace_text().length() - get_search_text().length();
			} else {
				//just replace
				text_edit->insert_text_at_cursor(get_replace_text());
			}
			rc++;
		}
		//restore editor state (selection, cursor, scroll)
		text_edit->cursor_set_line(orig_cursor.x);
		text_edit->cursor_set_column(orig_cursor.y);

		if (selection_enabled && is_replace_selection_only()) {
			//reselect
			text_edit->select(selection_begin.x, selection_begin.y, selection_end.x, selection_end.y);
		} else {
			text_edit->deselect();
		}

		text_edit->set_v_scroll(vsval);
		//text_edit->set_h_scroll(hsval);
		error_label->set_text(vformat(TTR("Replaced %d occurrence(s)."), rc));

		//hide();
	} else {

		if (text_edit->get_selection_text() == get_search_text()) {

			text_edit->insert_text_at_cursor(get_replace_text());
		}

		_search();
	}
	text_edit->end_complex_operation();
}

bool FindReplaceDialog::_search() {

	String text = get_search_text();
	uint32_t flags = 0;

	if (is_whole_words())
		flags |= TextEdit::SEARCH_WHOLE_WORDS;
	if (is_case_sensitive())
		flags |= TextEdit::SEARCH_MATCH_CASE;
	if (is_backwards())
		flags |= TextEdit::SEARCH_BACKWARDS;

	int line = text_edit->cursor_get_line(), col = text_edit->cursor_get_column();

	if (is_backwards()) {
		col -= 1;
		if (col < 0) {
			line -= 1;
			if (line < 0) {
				line = text_edit->get_line_count() - 1;
			}
			col = text_edit->get_line(line).length();
		}
	}
	bool found = text_edit->search(text, flags, line, col, line, col);

	if (found) {
		// print_line("found");
		text_edit->cursor_set_line(line);
		if (is_backwards())
			text_edit->cursor_set_column(col);
		else
			text_edit->cursor_set_column(col + text.length());
		text_edit->select(line, col, line, col + text.length());
		set_error("");
		return true;
	} else {

		set_error(TTR("Not found!"));
		return false;
	}
}

void FindReplaceDialog::_prompt_changed() {

	if (prompt->is_pressed()) {
		skip->show();
		get_ok()->set_text(TTR("Next"));
		selection_only->set_disabled(true);

	} else {
		skip->hide();
		get_ok()->set_text(TTR("Replace"));
		selection_only->set_disabled(false);
	}
}

void FindReplaceDialog::_skip_pressed() {

	_replace_skip_callback();
}

bool FindReplaceDialog::is_replace_mode() const {

	return replace_text->is_visible_in_tree();
}

bool FindReplaceDialog::is_replace_all_mode() const {

	return !prompt->is_pressed();
}

bool FindReplaceDialog::is_replace_selection_only() const {

	return selection_only->is_pressed();
}
void FindReplaceDialog::set_replace_selection_only(bool p_enable) {

	selection_only->set_pressed(p_enable);
}

void FindReplaceDialog::ok_pressed() {

	_search_callback();
}

void FindReplaceDialog::_search_text_entered(const String &p_text) {

	if (replace_text->is_visible_in_tree())
		return;
	emit_signal("search");
	_search();
}

void FindReplaceDialog::_replace_text_entered(const String &p_text) {

	if (!replace_text->is_visible_in_tree())
		return;

	emit_signal("search");
	_replace();
}

String FindReplaceDialog::get_search_text() const {

	return search_text->get_text();
}
String FindReplaceDialog::get_replace_text() const {

	return replace_text->get_text();
}
bool FindReplaceDialog::is_whole_words() const {

	return whole_words->is_pressed();
}
bool FindReplaceDialog::is_case_sensitive() const {

	return case_sensitive->is_pressed();
}
bool FindReplaceDialog::is_backwards() const {

	return backwards->is_pressed();
}

void FindReplaceDialog::set_error(const String &p_error) {

	error_label->set_text(p_error);
}

void FindReplaceDialog::set_text_edit(TextEdit *p_text_edit) {

	text_edit = p_text_edit;
}

void FindReplaceDialog::search_next() {
	_search();
}

void FindReplaceDialog::_bind_methods() {

	ClassDB::bind_method("_search_text_entered", &FindReplaceDialog::_search_text_entered);
	ClassDB::bind_method("_replace_text_entered", &FindReplaceDialog::_replace_text_entered);
	ClassDB::bind_method("_prompt_changed", &FindReplaceDialog::_prompt_changed);
	ClassDB::bind_method("_skip_pressed", &FindReplaceDialog::_skip_pressed);
	ADD_SIGNAL(MethodInfo("search"));
	ADD_SIGNAL(MethodInfo("skip"));
}

FindReplaceDialog::FindReplaceDialog() {

	set_self_modulate(Color(1, 1, 1, 0.8));

	VBoxContainer *vb = memnew(VBoxContainer);
	add_child(vb);

	search_text = memnew(LineEdit);
	vb->add_margin_child(TTR("Search"), search_text);
	search_text->connect("text_entered", this, "_search_text_entered");

	replace_label = memnew(Label);
	replace_label->set_text(TTR("Replace By"));
	vb->add_child(replace_label);
	replace_mc = memnew(MarginContainer);
	vb->add_child(replace_mc);

	replace_text = memnew(LineEdit);
	replace_text->set_anchor(MARGIN_RIGHT, ANCHOR_END);
	replace_text->set_begin(Point2(15, 132));
	replace_text->set_end(Point2(-15, 135));

	replace_mc->add_child(replace_text);

	replace_text->connect("text_entered", this, "_replace_text_entered");

	MarginContainer *opt_mg = memnew(MarginContainer);
	vb->add_child(opt_mg);
	VBoxContainer *svb = memnew(VBoxContainer);
	opt_mg->add_child(svb);

	svb->add_child(memnew(Label));

	whole_words = memnew(CheckButton);
	whole_words->set_text(TTR("Whole Words"));
	svb->add_child(whole_words);

	case_sensitive = memnew(CheckButton);
	case_sensitive->set_text(TTR("Case Sensitive"));
	svb->add_child(case_sensitive);

	backwards = memnew(CheckButton);
	backwards->set_text(TTR("Backwards"));
	svb->add_child(backwards);

	opt_mg = memnew(MarginContainer);
	vb->add_child(opt_mg);
	VBoxContainer *rvb = memnew(VBoxContainer);
	opt_mg->add_child(rvb);
	replace_vb = rvb;
	//rvb ->add_child(memnew(HSeparator));
	rvb->add_child(memnew(Label));

	prompt = memnew(CheckButton);
	prompt->set_text(TTR("Prompt On Replace"));
	rvb->add_child(prompt);
	prompt->connect("pressed", this, "_prompt_changed");

	selection_only = memnew(CheckButton);
	selection_only->set_text(TTR("Selection Only"));
	rvb->add_child(selection_only);

	int margin = get_constant("margin", "Dialogs");
	int button_margin = get_constant("button_margin", "Dialogs");

	skip = memnew(Button);
	skip->set_anchor(MARGIN_LEFT, ANCHOR_END);
	skip->set_anchor(MARGIN_TOP, ANCHOR_END);
	skip->set_anchor(MARGIN_RIGHT, ANCHOR_END);
	skip->set_anchor(MARGIN_BOTTOM, ANCHOR_END);
	skip->set_begin(Point2(-70, -button_margin));
	skip->set_end(Point2(-10, -margin));
	skip->set_text(TTR("Skip"));
	add_child(skip);
	skip->connect("pressed", this, "_skip_pressed");

	error_label = memnew(Label);
	error_label->set_align(Label::ALIGN_CENTER);
	error_label->add_color_override("font_color", EditorNode::get_singleton()->get_gui_base()->get_color("error_color", "Editor"));

	vb->add_child(error_label);

	set_hide_on_ok(false);
}

/*** CODE EDITOR ****/

void CodeTextEditor::_text_editor_gui_input(const Ref<InputEvent> &p_event) {

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {

		if (mb->is_pressed() && mb->get_command()) {

			if (mb->get_button_index() == BUTTON_WHEEL_UP) {
				_zoom_in();
			} else if (mb->get_button_index() == BUTTON_WHEEL_DOWN) {
				_zoom_out();
			}
		}
	}

	Ref<InputEventKey> k = p_event;

	if (k.is_valid()) {

		if (k->is_pressed()) {
			if (ED_IS_SHORTCUT("script_editor/zoom_in", p_event)) {
				_zoom_in();
			}
			if (ED_IS_SHORTCUT("script_editor/zoom_out", p_event)) {
				_zoom_out();
			}
			if (ED_IS_SHORTCUT("script_editor/reset_zoom", p_event)) {
				_reset_zoom();
			}
		}
	}
}

void CodeTextEditor::_zoom_in() {
	font_resize_val += 1;

	if (font_resize_timer->get_time_left() == 0)
		font_resize_timer->start();
}

void CodeTextEditor::_zoom_out() {
	font_resize_val -= 1;

	if (font_resize_timer->get_time_left() == 0)
		font_resize_timer->start();
}

void CodeTextEditor::_reset_zoom() {
	Ref<DynamicFont> font = text_editor->get_font("font"); // reset source font size to default

	if (font.is_valid()) {
		EditorSettings::get_singleton()->set("interface/source_font_size", 14);
		font->set_size(14);
	}
}

void CodeTextEditor::_line_col_changed() {

	line_nb->set_text(itos(text_editor->cursor_get_line() + 1));
	col_nb->set_text(itos(text_editor->cursor_get_column() + 1));
}

void CodeTextEditor::_text_changed() {

	if (text_editor->is_insert_text_operation()) {
		code_complete_timer->start();
		idle->start();
	}
}

void CodeTextEditor::_code_complete_timer_timeout() {
	if (!is_visible_in_tree())
		return;
	if (enable_complete_timer)
		text_editor->query_code_comple();
}

void CodeTextEditor::_complete_request() {

	List<String> entries;
	String ctext = text_editor->get_text_for_completion();
	_code_complete_script(ctext, &entries);
	bool forced = false;
	if (code_complete_func) {
		code_complete_func(code_complete_ud, ctext, &entries, forced);
	}
	// print_line("COMPLETE: "+p_request);
	if (entries.size() == 0)
		return;
	Vector<String> strs;
	strs.resize(entries.size());
	int i = 0;
	for (List<String>::Element *E = entries.front(); E; E = E->next()) {

		strs[i++] = E->get();
	}

	text_editor->code_complete(strs, forced);
}

void CodeTextEditor::_font_resize_timeout() {

	Ref<DynamicFont> font = text_editor->get_font("font");

	if (font.is_valid()) {
		int size = font->get_size() + font_resize_val;

		if (size >= 8 && size <= 96) {
			EditorSettings::get_singleton()->set("interface/source_font_size", size);
			font->set_size(size);
		}

		font_resize_val = 0;
	}
}

void CodeTextEditor::update_editor_settings() {

	text_editor->set_auto_brace_completion(EditorSettings::get_singleton()->get("text_editor/completion/auto_brace_complete"));
	text_editor->set_scroll_pass_end_of_file(EditorSettings::get_singleton()->get("text_editor/cursor/scroll_past_end_of_file"));
	text_editor->set_indent_using_spaces(EditorSettings::get_singleton()->get("text_editor/indent/type"));
	text_editor->set_indent_size(EditorSettings::get_singleton()->get("text_editor/indent/size"));
	text_editor->set_auto_indent(EditorSettings::get_singleton()->get("text_editor/indent/auto_indent"));
	text_editor->set_draw_tabs(EditorSettings::get_singleton()->get("text_editor/indent/draw_tabs"));
	text_editor->set_show_line_numbers(EditorSettings::get_singleton()->get("text_editor/line_numbers/show_line_numbers"));
	text_editor->set_line_numbers_zero_padded(EditorSettings::get_singleton()->get("text_editor/line_numbers/line_numbers_zero_padded"));
	text_editor->set_show_line_length_guideline(EditorSettings::get_singleton()->get("text_editor/line_numbers/show_line_length_guideline"));
	text_editor->set_line_length_guideline_column(EditorSettings::get_singleton()->get("text_editor/line_numbers/line_length_guideline_column"));
	text_editor->set_syntax_coloring(EditorSettings::get_singleton()->get("text_editor/highlighting/syntax_highlighting"));
	text_editor->set_highlight_all_occurrences(EditorSettings::get_singleton()->get("text_editor/highlighting/highlight_all_occurrences"));
	text_editor->cursor_set_blink_enabled(EditorSettings::get_singleton()->get("text_editor/cursor/caret_blink"));
	text_editor->cursor_set_blink_speed(EditorSettings::get_singleton()->get("text_editor/cursor/caret_blink_speed"));
	text_editor->set_draw_breakpoint_gutter(EditorSettings::get_singleton()->get("text_editor/line_numbers/show_breakpoint_gutter"));
	text_editor->cursor_set_block_mode(EditorSettings::get_singleton()->get("text_editor/cursor/block_caret"));
	text_editor->set_smooth_scroll_enabled(EditorSettings::get_singleton()->get("text_editor/open_scripts/smooth_scrolling"));
	text_editor->set_v_scroll_speed(EditorSettings::get_singleton()->get("text_editor/open_scripts/v_scroll_speed"));
}

void CodeTextEditor::set_error(const String &p_error) {

	error->set_text(p_error);
}

void CodeTextEditor::_update_font() {

	// FONTS
	String editor_font = EDITOR_DEF("text_editor/theme/font", "");
	bool font_overridden = false;
	if (editor_font != "") {
		Ref<Font> fnt = ResourceLoader::load(editor_font);
		if (fnt.is_valid()) {
			text_editor->add_font_override("font", fnt);
			font_overridden = true;
		}
	}
	if (!font_overridden) {

		text_editor->add_font_override("font", get_font("source", "EditorFonts"));
	}
}

void CodeTextEditor::_on_settings_change() {

	_update_font();

	// AUTO BRACE COMPLETION
	text_editor->set_auto_brace_completion(
			EDITOR_DEF("text_editor/completion/auto_brace_complete", true));

	code_complete_timer->set_wait_time(
			EDITOR_DEF("text_editor/completion/code_complete_delay", .3f));

	enable_complete_timer = EDITOR_DEF("text_editor/completion/enable_code_completion_delay", true);

	// call hint settings
	text_editor->set_callhint_settings(
			EDITOR_DEF("text_editor/completion/put_callhint_tooltip_below_current_line", true),
			EDITOR_DEF("text_editor/completion/callhint_tooltip_offset", Vector2()));
}

void CodeTextEditor::_text_changed_idle_timeout() {

	_validate_script();
	emit_signal("validate_script");
}

void CodeTextEditor::_notification(int p_what) {

	if (p_what == EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED) {
		_load_theme_settings();
		emit_signal("load_theme_settings");
	}
	if (p_what == NOTIFICATION_THEME_CHANGED) {
		_update_font();
	}
}

void CodeTextEditor::_bind_methods() {

	ClassDB::bind_method("_text_editor_gui_input", &CodeTextEditor::_text_editor_gui_input);
	ClassDB::bind_method("_line_col_changed", &CodeTextEditor::_line_col_changed);
	ClassDB::bind_method("_text_changed", &CodeTextEditor::_text_changed);
	ClassDB::bind_method("_on_settings_change", &CodeTextEditor::_on_settings_change);
	ClassDB::bind_method("_text_changed_idle_timeout", &CodeTextEditor::_text_changed_idle_timeout);
	ClassDB::bind_method("_code_complete_timer_timeout", &CodeTextEditor::_code_complete_timer_timeout);
	ClassDB::bind_method("_complete_request", &CodeTextEditor::_complete_request);
	ClassDB::bind_method("_font_resize_timeout", &CodeTextEditor::_font_resize_timeout);

	ADD_SIGNAL(MethodInfo("validate_script"));
	ADD_SIGNAL(MethodInfo("load_theme_settings"));
}

void CodeTextEditor::set_code_complete_func(CodeTextEditorCodeCompleteFunc p_code_complete_func, void *p_ud) {
	code_complete_func = p_code_complete_func;
	code_complete_ud = p_ud;
}

CodeTextEditor::CodeTextEditor() {

	code_complete_func = NULL;
	ED_SHORTCUT("script_editor/zoom_in", TTR("Zoom In"), KEY_MASK_CMD | KEY_EQUAL);
	ED_SHORTCUT("script_editor/zoom_out", TTR("Zoom Out"), KEY_MASK_CMD | KEY_MINUS);
	ED_SHORTCUT("script_editor/reset_zoom", TTR("Reset Zoom"), KEY_MASK_CMD | KEY_0);

	find_replace_bar = memnew(FindReplaceBar);
	add_child(find_replace_bar);
	find_replace_bar->set_h_size_flags(SIZE_EXPAND_FILL);
	find_replace_bar->hide();

	text_editor = memnew(TextEdit);
	add_child(text_editor);
	text_editor->set_v_size_flags(SIZE_EXPAND_FILL);

	find_replace_bar->set_text_edit(text_editor);

	text_editor->set_show_line_numbers(true);
	text_editor->set_brace_matching(true);
	text_editor->set_auto_indent(true);

	MarginContainer *status_mc = memnew(MarginContainer);
	add_child(status_mc);
	status_mc->set("custom_constants/margin_left", 2);
	status_mc->set("custom_constants/margin_top", 5);
	status_mc->set("custom_constants/margin_right", 2);
	status_mc->set("custom_constants/margin_bottom", 1);

	HBoxContainer *status_bar = memnew(HBoxContainer);
	status_mc->add_child(status_bar);
	status_bar->set_h_size_flags(SIZE_EXPAND_FILL);
	status_bar->add_child(memnew(Label)); //to keep the height if the other labels are not visible

	idle = memnew(Timer);
	add_child(idle);
	idle->set_one_shot(true);
	idle->set_wait_time(EDITOR_DEF("text_editor/completion/idle_parse_delay", 2));

	code_complete_timer = memnew(Timer);
	add_child(code_complete_timer);
	code_complete_timer->set_one_shot(true);
	enable_complete_timer = EDITOR_DEF("text_editor/completion/enable_code_completion_delay", true);

	code_complete_timer->set_wait_time(EDITOR_DEF("text_editor/completion/code_complete_delay", .3f));

	error = memnew(Label);
	status_bar->add_child(error);
	error->set_clip_text(true); //do not change, or else very long errors can push the whole container to the right
	error->set_valign(Label::VALIGN_CENTER);
	error->add_color_override("font_color", EditorNode::get_singleton()->get_gui_base()->get_color("error_color", "Editor"));
	error->set_h_size_flags(SIZE_EXPAND_FILL); //required for it to display, given now it's clipping contents, do not touch

	Label *line_txt = memnew(Label);
	status_bar->add_child(line_txt);
	line_txt->set_align(Label::ALIGN_RIGHT);
	line_txt->set_valign(Label::VALIGN_CENTER);
	line_txt->set_v_size_flags(SIZE_FILL);
	line_txt->set_text(TTR("Line:"));

	line_nb = memnew(Label);
	status_bar->add_child(line_nb);
	line_nb->set_valign(Label::VALIGN_CENTER);
	line_nb->set_v_size_flags(SIZE_FILL);
	line_nb->set_autowrap(true); // workaround to prevent resizing the label on each change, do not touch
	line_nb->set_clip_text(true); // workaround to prevent resizing the label on each change, do not touch
	line_nb->set_custom_minimum_size(Size2(40, 1) * EDSCALE);

	Label *col_txt = memnew(Label);
	status_bar->add_child(col_txt);
	col_txt->set_align(Label::ALIGN_RIGHT);
	col_txt->set_valign(Label::VALIGN_CENTER);
	col_txt->set_v_size_flags(SIZE_FILL);
	col_txt->set_text(TTR("Col:"));

	col_nb = memnew(Label);
	status_bar->add_child(col_nb);
	col_nb->set_valign(Label::VALIGN_CENTER);
	col_nb->set_v_size_flags(SIZE_FILL);
	col_nb->set_autowrap(true); // workaround to prevent resizing the label on each change, do not touch
	col_nb->set_clip_text(true); // workaround to prevent resizing the label on each change, do not touch
	col_nb->set_custom_minimum_size(Size2(40, 1) * EDSCALE);

	text_editor->connect("gui_input", this, "_text_editor_gui_input");
	text_editor->connect("cursor_changed", this, "_line_col_changed");
	text_editor->connect("text_changed", this, "_text_changed");
	text_editor->connect("request_completion", this, "_complete_request");
	Vector<String> cs;
	cs.push_back(".");
	cs.push_back(",");
	cs.push_back("(");
	cs.push_back("=");
	cs.push_back("$");
	text_editor->set_completion(true, cs);
	idle->connect("timeout", this, "_text_changed_idle_timeout");

	code_complete_timer->connect("timeout", this, "_code_complete_timer_timeout");

	font_resize_val = 0;
	font_resize_timer = memnew(Timer);
	add_child(font_resize_timer);
	font_resize_timer->set_one_shot(true);
	font_resize_timer->set_wait_time(0.07);
	font_resize_timer->connect("timeout", this, "_font_resize_timeout");

	EditorSettings::get_singleton()->connect("settings_changed", this, "_on_settings_change");
}
