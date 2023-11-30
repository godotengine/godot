/**************************************************************************/
/*  code_editor.cpp                                                       */
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

#include "code_editor.h"

#include "core/input/input.h"
#include "core/os/keyboard.h"
#include "core/string/string_builder.h"
#include "core/templates/pair.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/plugins/script_editor_plugin.h"
#include "scene/resources/font.h"

void GotoLineDialog::popup_find_line(CodeEdit *p_edit) {
	text_editor = p_edit;

	// Add 1 because text_editor->get_caret_line() starts from 0, but the editor user interface starts from 1.
	line->set_text(itos(text_editor->get_caret_line() + 1));
	line->select_all();
	popup_centered(Size2(180, 80) * EDSCALE);
	line->grab_focus();
}

int GotoLineDialog::get_line() const {
	return line->get_text().to_int();
}

void GotoLineDialog::ok_pressed() {
	// Subtract 1 because the editor user interface starts from 1, but text_editor->set_caret_line(n) starts from 0.
	const int line_number = get_line() - 1;
	if (line_number < 0 || line_number >= text_editor->get_line_count()) {
		return;
	}
	text_editor->remove_secondary_carets();
	text_editor->unfold_line(line_number);
	text_editor->set_caret_line(line_number);
	hide();
}

GotoLineDialog::GotoLineDialog() {
	set_title(TTR("Go to Line"));

	VBoxContainer *vbc = memnew(VBoxContainer);
	vbc->set_anchor_and_offset(SIDE_LEFT, Control::ANCHOR_BEGIN, 8 * EDSCALE);
	vbc->set_anchor_and_offset(SIDE_TOP, Control::ANCHOR_BEGIN, 8 * EDSCALE);
	vbc->set_anchor_and_offset(SIDE_RIGHT, Control::ANCHOR_END, -8 * EDSCALE);
	vbc->set_anchor_and_offset(SIDE_BOTTOM, Control::ANCHOR_END, -8 * EDSCALE);
	add_child(vbc);

	Label *l = memnew(Label);
	l->set_text(TTR("Line Number:"));
	vbc->add_child(l);

	line = memnew(LineEdit);
	vbc->add_child(line);
	register_text_enter(line);
	text_editor = nullptr;

	line_label = nullptr;

	set_hide_on_ok(false);
}

void FindReplaceBar::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY:
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			find_prev->set_icon(get_editor_theme_icon(SNAME("MoveUp")));
			find_next->set_icon(get_editor_theme_icon(SNAME("MoveDown")));
			hide_button->set_texture_normal(get_editor_theme_icon(SNAME("Close")));
			hide_button->set_texture_hover(get_editor_theme_icon(SNAME("Close")));
			hide_button->set_texture_pressed(get_editor_theme_icon(SNAME("Close")));
			hide_button->set_custom_minimum_size(hide_button->get_texture_normal()->get_size());
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			set_process_unhandled_input(is_visible_in_tree());
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			matches_label->add_theme_color_override("font_color", results_count > 0 ? get_theme_color(SNAME("font_color"), SNAME("Label")) : get_theme_color(SNAME("error_color"), EditorStringName(Editor)));
		} break;

		case NOTIFICATION_PREDELETE: {
			if (base_text_editor) {
				base_text_editor->remove_find_replace_bar();
				base_text_editor = nullptr;
			}
		} break;
	}
}

void FindReplaceBar::unhandled_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventKey> k = p_event;

	if (k.is_valid() && k->is_action_pressed(SNAME("ui_cancel"), false, true)) {
		Control *focus_owner = get_viewport()->gui_get_focus_owner();

		if (text_editor->has_focus() || (focus_owner && is_ancestor_of(focus_owner))) {
			_hide_bar();
			accept_event();
		}
	}
}

void FindReplaceBar::_focus_lost() {
	if (Input::get_singleton()->is_action_pressed(SNAME("ui_cancel"))) {
		// Unfocused after pressing Escape, so hide the bar.
		_hide_bar(true);
	}
}

void FindReplaceBar::_update_flags(bool p_direction_backwards) {
	flags = 0;

	if (is_whole_words()) {
		flags |= TextEdit::SEARCH_WHOLE_WORDS;
	}
	if (is_case_sensitive()) {
		flags |= TextEdit::SEARCH_MATCH_CASE;
	}
	if (p_direction_backwards) {
		flags |= TextEdit::SEARCH_BACKWARDS;
	}
}

bool FindReplaceBar::_search(uint32_t p_flags, int p_from_line, int p_from_col) {
	if (!preserve_cursor) {
		text_editor->remove_secondary_carets();
	}
	String text = get_search_text();
	Point2i pos = text_editor->search(text, p_flags, p_from_line, p_from_col);

	if (pos.x != -1) {
		if (!preserve_cursor && !is_selection_only()) {
			text_editor->unfold_line(pos.y);
			text_editor->set_caret_line(pos.y, false);
			text_editor->set_caret_column(pos.x + text.length(), false);
			text_editor->center_viewport_to_caret(0);
			text_editor->select(pos.y, pos.x, pos.y, pos.x + text.length());

			line_col_changed_for_result = true;
		}

		text_editor->set_search_text(text);
		text_editor->set_search_flags(p_flags);

		result_line = pos.y;
		result_col = pos.x;

		_update_results_count();
	} else {
		results_count = 0;
		result_line = -1;
		result_col = -1;
		text_editor->set_search_text("");
		text_editor->set_search_flags(p_flags);
	}

	_update_matches_label();

	return pos.x != -1;
}

void FindReplaceBar::_replace() {
	text_editor->remove_secondary_carets();
	bool selection_enabled = text_editor->has_selection(0);
	Point2i selection_begin, selection_end;
	if (selection_enabled) {
		selection_begin = Point2i(text_editor->get_selection_from_line(0), text_editor->get_selection_from_column(0));
		selection_end = Point2i(text_editor->get_selection_to_line(0), text_editor->get_selection_to_column(0));
	}

	String repl_text = get_replace_text();
	int search_text_len = get_search_text().length();

	text_editor->begin_complex_operation();
	if (selection_enabled && is_selection_only()) {
		// Restrict search_current() to selected region.
		text_editor->set_caret_line(selection_begin.width, false, true, 0, 0);
		text_editor->set_caret_column(selection_begin.height, true, 0);
	}

	if (search_current()) {
		text_editor->unfold_line(result_line);
		text_editor->select(result_line, result_col, result_line, result_col + search_text_len, 0);

		if (selection_enabled && is_selection_only()) {
			Point2i match_from(result_line, result_col);
			Point2i match_to(result_line, result_col + search_text_len);
			if (!(match_from < selection_begin || match_to > selection_end)) {
				text_editor->insert_text_at_caret(repl_text, 0);
				if (match_to.x == selection_end.x) {
					// Adjust selection bounds if necessary.
					selection_end.y += repl_text.length() - search_text_len;
				}
			}
		} else {
			text_editor->insert_text_at_caret(repl_text, 0);
		}
	}
	text_editor->end_complex_operation();
	results_count = -1;
	results_count_to_current = -1;
	needs_to_count_results = true;

	if (selection_enabled && is_selection_only()) {
		// Reselect in order to keep 'Replace' restricted to selection.
		text_editor->select(selection_begin.x, selection_begin.y, selection_end.x, selection_end.y, 0);
	} else {
		text_editor->deselect(0);
	}
}

void FindReplaceBar::_replace_all() {
	text_editor->remove_secondary_carets();
	text_editor->disconnect("text_changed", callable_mp(this, &FindReplaceBar::_editor_text_changed));
	// Line as x so it gets priority in comparison, column as y.
	Point2i orig_cursor(text_editor->get_caret_line(0), text_editor->get_caret_column(0));
	Point2i prev_match = Point2(-1, -1);

	bool selection_enabled = text_editor->has_selection(0);
	if (!is_selection_only()) {
		text_editor->deselect();
		selection_enabled = false;
	} else {
		result_line = -1;
		result_col = -1;
	}

	Point2i selection_begin, selection_end;
	if (selection_enabled) {
		selection_begin = Point2i(text_editor->get_selection_from_line(0), text_editor->get_selection_from_column(0));
		selection_end = Point2i(text_editor->get_selection_to_line(0), text_editor->get_selection_to_column(0));
	}

	int vsval = text_editor->get_v_scroll();

	String repl_text = get_replace_text();
	int search_text_len = get_search_text().length();

	int rc = 0;

	replace_all_mode = true;

	text_editor->begin_complex_operation();

	if (selection_enabled && is_selection_only()) {
		text_editor->set_caret_line(selection_begin.width, false, true, 0, 0);
		text_editor->set_caret_column(selection_begin.height, true, 0);
	} else {
		text_editor->set_caret_line(0, false, true, 0, 0);
		text_editor->set_caret_column(0, true, 0);
	}

	if (search_current()) {
		do {
			// Replace area.
			Point2i match_from(result_line, result_col);
			Point2i match_to(result_line, result_col + search_text_len);

			if (match_from < prev_match) {
				break; // Done.
			}

			prev_match = Point2i(result_line, result_col + repl_text.length());

			text_editor->unfold_line(result_line);
			text_editor->select(result_line, result_col, result_line, match_to.y, 0);

			if (selection_enabled) {
				if (match_from < selection_begin || match_to > selection_end) {
					break; // Done.
				}

				// Replace but adjust selection bounds.
				text_editor->insert_text_at_caret(repl_text, 0);
				if (match_to.x == selection_end.x) {
					selection_end.y += repl_text.length() - search_text_len;
				}

			} else {
				// Just replace.
				text_editor->insert_text_at_caret(repl_text, 0);
			}

			rc++;
		} while (search_next());
	}

	text_editor->end_complex_operation();

	replace_all_mode = false;

	// Restore editor state (selection, cursor, scroll).
	text_editor->set_caret_line(orig_cursor.x, false, true, 0, 0);
	text_editor->set_caret_column(orig_cursor.y, true, 0);

	if (selection_enabled) {
		// Reselect.
		text_editor->select(selection_begin.x, selection_begin.y, selection_end.x, selection_end.y, 0);
	}

	text_editor->set_v_scroll(vsval);
	matches_label->add_theme_color_override("font_color", rc > 0 ? get_theme_color(SNAME("font_color"), SNAME("Label")) : get_theme_color(SNAME("error_color"), EditorStringName(Editor)));
	matches_label->set_text(vformat(TTR("%d replaced."), rc));

	text_editor->call_deferred(SNAME("connect"), "text_changed", callable_mp(this, &FindReplaceBar::_editor_text_changed));
	results_count = -1;
	results_count_to_current = -1;
	needs_to_count_results = true;
}

void FindReplaceBar::_get_search_from(int &r_line, int &r_col, bool p_is_searching_next) {
	if (!text_editor->has_selection(0) || is_selection_only()) {
		r_line = text_editor->get_caret_line(0);
		r_col = text_editor->get_caret_column(0);

		if (!p_is_searching_next && r_line == result_line && r_col >= result_col && r_col <= result_col + get_search_text().length()) {
			r_col = result_col;
		}
		return;
	}

	if (p_is_searching_next) {
		r_line = text_editor->get_selection_to_line();
		r_col = text_editor->get_selection_to_column();
	} else {
		r_line = text_editor->get_selection_from_line();
		r_col = text_editor->get_selection_from_column();
	}
}

void FindReplaceBar::_update_results_count() {
	if (!needs_to_count_results && (result_line != -1) && results_count_to_current > 0) {
		results_count_to_current += (flags & TextEdit::SEARCH_BACKWARDS) ? -1 : 1;

		if (results_count_to_current > results_count) {
			results_count_to_current = results_count_to_current - results_count;
		} else if (results_count_to_current <= 0) {
			results_count_to_current = results_count;
		}

		return;
	}

	String searched = get_search_text();
	if (searched.is_empty()) {
		return;
	}

	needs_to_count_results = false;

	results_count = 0;

	for (int i = 0; i < text_editor->get_line_count(); i++) {
		String line_text = text_editor->get_line(i);

		int col_pos = 0;

		bool searched_start_is_symbol = is_symbol(searched[0]);
		bool searched_end_is_symbol = is_symbol(searched[searched.length() - 1]);

		while (true) {
			col_pos = is_case_sensitive() ? line_text.find(searched, col_pos) : line_text.findn(searched, col_pos);

			if (col_pos == -1) {
				break;
			}

			if (is_whole_words()) {
				if (!searched_start_is_symbol && col_pos > 0 && !is_symbol(line_text[col_pos - 1])) {
					col_pos += searched.length();
					continue;
				}
				if (!searched_end_is_symbol && col_pos + searched.length() < line_text.length() && !is_symbol(line_text[col_pos + searched.length()])) {
					col_pos += searched.length();
					continue;
				}
			}

			results_count++;

			if (i == result_line) {
				if (col_pos == result_col) {
					results_count_to_current = results_count;
				} else if (col_pos < result_col && col_pos + searched.length() > result_col) {
					col_pos = result_col;
					results_count_to_current = results_count;
				}
			}

			col_pos += searched.length();
		}
	}
}

void FindReplaceBar::_update_matches_label() {
	if (search_text->get_text().is_empty() || results_count == -1) {
		matches_label->hide();
	} else {
		matches_label->show();

		matches_label->add_theme_color_override("font_color", results_count > 0 ? get_theme_color(SNAME("font_color"), SNAME("Label")) : get_theme_color(SNAME("error_color"), EditorStringName(Editor)));

		if (results_count == 0) {
			matches_label->set_text(TTR("No match"));
		} else if (results_count_to_current == -1) {
			matches_label->set_text(vformat(TTRN("%d match", "%d matches", results_count), results_count));
		} else {
			matches_label->set_text(vformat(TTRN("%d of %d match", "%d of %d matches", results_count), results_count_to_current, results_count));
		}
	}
}

bool FindReplaceBar::search_current() {
	_update_flags(false);

	int line, col;
	_get_search_from(line, col);

	return _search(flags, line, col);
}

bool FindReplaceBar::search_prev() {
	if (is_selection_only() && !replace_all_mode) {
		return false;
	}

	if (!is_visible()) {
		popup_search(true);
	}

	String text = get_search_text();

	_update_flags(true);

	int line, col;
	_get_search_from(line, col);

	col -= text.length();
	if (col < 0) {
		line -= 1;
		if (line < 0) {
			line = text_editor->get_line_count() - 1;
		}
		col = text_editor->get_line(line).length();
	}

	return _search(flags, line, col);
}

bool FindReplaceBar::search_next() {
	if (is_selection_only() && !replace_all_mode) {
		return false;
	}

	if (!is_visible()) {
		popup_search(true);
	}

	_update_flags(false);

	int line, col;
	_get_search_from(line, col, true);

	return _search(flags, line, col);
}

void FindReplaceBar::_hide_bar(bool p_force_focus) {
	if (replace_text->has_focus() || search_text->has_focus() || p_force_focus) {
		text_editor->grab_focus();
	}

	text_editor->set_search_text("");
	result_line = -1;
	result_col = -1;
	hide();
}

void FindReplaceBar::_show_search(bool p_focus_replace, bool p_show_only) {
	show();
	if (p_show_only) {
		return;
	}

	if (p_focus_replace) {
		search_text->deselect();
		replace_text->call_deferred(SNAME("grab_focus"));
	} else {
		replace_text->deselect();
		search_text->call_deferred(SNAME("grab_focus"));
	}

	if (text_editor->has_selection(0) && !is_selection_only()) {
		search_text->set_text(text_editor->get_selected_text(0));
		result_line = text_editor->get_selection_from_line();
		result_col = text_editor->get_selection_from_column();
	}

	if (!get_search_text().is_empty()) {
		if (p_focus_replace) {
			replace_text->select_all();
			replace_text->set_caret_column(replace_text->get_text().length());
		} else {
			search_text->select_all();
			search_text->set_caret_column(search_text->get_text().length());
		}

		preserve_cursor = true;
		_search_text_changed(get_search_text());
		preserve_cursor = false;
	}
}

void FindReplaceBar::popup_search(bool p_show_only) {
	replace_text->hide();
	hbc_button_replace->hide();
	hbc_option_replace->hide();
	selection_only->set_pressed(false);

	_show_search(false, p_show_only);
}

void FindReplaceBar::popup_replace() {
	if (!replace_text->is_visible_in_tree()) {
		replace_text->show();
		hbc_button_replace->show();
		hbc_option_replace->show();
	}

	selection_only->set_pressed((text_editor->has_selection(0) && text_editor->get_selection_from_line(0) < text_editor->get_selection_to_line(0)));

	_show_search(is_visible() || text_editor->has_selection(0));
}

void FindReplaceBar::_search_options_changed(bool p_pressed) {
	results_count = -1;
	results_count_to_current = -1;
	needs_to_count_results = true;
	search_current();
}

void FindReplaceBar::_editor_text_changed() {
	results_count = -1;
	results_count_to_current = -1;
	needs_to_count_results = true;
	if (is_visible_in_tree()) {
		preserve_cursor = true;
		search_current();
		preserve_cursor = false;
	}
}

void FindReplaceBar::_search_text_changed(const String &p_text) {
	results_count = -1;
	results_count_to_current = -1;
	needs_to_count_results = true;
	search_current();
}

void FindReplaceBar::_search_text_submitted(const String &p_text) {
	if (Input::get_singleton()->is_key_pressed(Key::SHIFT)) {
		search_prev();
	} else {
		search_next();
	}
}

void FindReplaceBar::_replace_text_submitted(const String &p_text) {
	if (selection_only->is_pressed() && text_editor->has_selection(0)) {
		_replace_all();
		_hide_bar();
	} else if (Input::get_singleton()->is_key_pressed(Key::SHIFT)) {
		_replace();
		search_prev();
	} else {
		_replace();
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
	emit_signal(SNAME("error"), p_label);
}

void FindReplaceBar::set_text_edit(CodeTextEditor *p_text_editor) {
	if (p_text_editor == base_text_editor) {
		return;
	}

	if (base_text_editor) {
		base_text_editor->remove_find_replace_bar();
		base_text_editor = nullptr;
		text_editor->disconnect("text_changed", callable_mp(this, &FindReplaceBar::_editor_text_changed));
		text_editor = nullptr;
	}

	if (!p_text_editor) {
		return;
	}

	results_count = -1;
	results_count_to_current = -1;
	needs_to_count_results = true;
	base_text_editor = p_text_editor;
	text_editor = base_text_editor->get_text_editor();
	text_editor->connect("text_changed", callable_mp(this, &FindReplaceBar::_editor_text_changed));

	_update_results_count();
	_update_matches_label();
}

void FindReplaceBar::_bind_methods() {
	ClassDB::bind_method("_search_current", &FindReplaceBar::search_current);

	ADD_SIGNAL(MethodInfo("error"));
}

FindReplaceBar::FindReplaceBar() {
	vbc_lineedit = memnew(VBoxContainer);
	add_child(vbc_lineedit);
	vbc_lineedit->set_alignment(BoxContainer::ALIGNMENT_CENTER);
	vbc_lineedit->set_h_size_flags(SIZE_EXPAND_FILL);
	VBoxContainer *vbc_button = memnew(VBoxContainer);
	add_child(vbc_button);
	VBoxContainer *vbc_option = memnew(VBoxContainer);
	add_child(vbc_option);

	HBoxContainer *hbc_button_search = memnew(HBoxContainer);
	vbc_button->add_child(hbc_button_search);
	hbc_button_search->set_alignment(BoxContainer::ALIGNMENT_END);
	hbc_button_replace = memnew(HBoxContainer);
	vbc_button->add_child(hbc_button_replace);
	hbc_button_replace->set_alignment(BoxContainer::ALIGNMENT_END);

	HBoxContainer *hbc_option_search = memnew(HBoxContainer);
	vbc_option->add_child(hbc_option_search);
	hbc_option_replace = memnew(HBoxContainer);
	vbc_option->add_child(hbc_option_replace);

	// Search toolbar
	search_text = memnew(LineEdit);
	vbc_lineedit->add_child(search_text);
	search_text->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
	search_text->connect("text_changed", callable_mp(this, &FindReplaceBar::_search_text_changed));
	search_text->connect("text_submitted", callable_mp(this, &FindReplaceBar::_search_text_submitted));
	search_text->connect("focus_exited", callable_mp(this, &FindReplaceBar::_focus_lost));

	matches_label = memnew(Label);
	hbc_button_search->add_child(matches_label);
	matches_label->hide();

	find_prev = memnew(Button);
	find_prev->set_flat(true);
	hbc_button_search->add_child(find_prev);
	find_prev->set_focus_mode(FOCUS_NONE);
	find_prev->connect("pressed", callable_mp(this, &FindReplaceBar::search_prev));

	find_next = memnew(Button);
	find_next->set_flat(true);
	hbc_button_search->add_child(find_next);
	find_next->set_focus_mode(FOCUS_NONE);
	find_next->connect("pressed", callable_mp(this, &FindReplaceBar::search_next));

	case_sensitive = memnew(CheckBox);
	hbc_option_search->add_child(case_sensitive);
	case_sensitive->set_text(TTR("Match Case"));
	case_sensitive->set_focus_mode(FOCUS_NONE);
	case_sensitive->connect("toggled", callable_mp(this, &FindReplaceBar::_search_options_changed));

	whole_words = memnew(CheckBox);
	hbc_option_search->add_child(whole_words);
	whole_words->set_text(TTR("Whole Words"));
	whole_words->set_focus_mode(FOCUS_NONE);
	whole_words->connect("toggled", callable_mp(this, &FindReplaceBar::_search_options_changed));

	// Replace toolbar
	replace_text = memnew(LineEdit);
	vbc_lineedit->add_child(replace_text);
	replace_text->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
	replace_text->connect("text_submitted", callable_mp(this, &FindReplaceBar::_replace_text_submitted));
	replace_text->connect("focus_exited", callable_mp(this, &FindReplaceBar::_focus_lost));

	replace = memnew(Button);
	hbc_button_replace->add_child(replace);
	replace->set_text(TTR("Replace"));
	replace->connect("pressed", callable_mp(this, &FindReplaceBar::_replace));

	replace_all = memnew(Button);
	hbc_button_replace->add_child(replace_all);
	replace_all->set_text(TTR("Replace All"));
	replace_all->connect("pressed", callable_mp(this, &FindReplaceBar::_replace_all));

	selection_only = memnew(CheckBox);
	hbc_option_replace->add_child(selection_only);
	selection_only->set_text(TTR("Selection Only"));
	selection_only->set_focus_mode(FOCUS_NONE);
	selection_only->connect("toggled", callable_mp(this, &FindReplaceBar::_search_options_changed));

	hide_button = memnew(TextureButton);
	add_child(hide_button);
	hide_button->set_focus_mode(FOCUS_NONE);
	hide_button->connect("pressed", callable_mp(this, &FindReplaceBar::_hide_bar).bind(false));
	hide_button->set_v_size_flags(SIZE_SHRINK_CENTER);
}

/*** CODE EDITOR ****/

// This function should be used to handle shortcuts that could otherwise
// be handled too late if they weren't handled here.
void CodeTextEditor::input(const Ref<InputEvent> &event) {
	ERR_FAIL_COND(event.is_null());

	const Ref<InputEventKey> key_event = event;

	if (!key_event.is_valid()) {
		return;
	}
	if (!key_event->is_pressed()) {
		return;
	}

	if (!text_editor->has_focus()) {
		if ((find_replace_bar != nullptr && find_replace_bar->is_visible()) && (find_replace_bar->has_focus() || (get_viewport()->gui_get_focus_owner() && find_replace_bar->is_ancestor_of(get_viewport()->gui_get_focus_owner())))) {
			if (ED_IS_SHORTCUT("script_text_editor/find_next", key_event)) {
				find_replace_bar->search_next();
				accept_event();
				return;
			}
			if (ED_IS_SHORTCUT("script_text_editor/find_previous", key_event)) {
				find_replace_bar->search_prev();
				accept_event();
				return;
			}
		}
		return;
	}

	if (ED_IS_SHORTCUT("script_text_editor/move_up", key_event)) {
		move_lines_up();
		accept_event();
		return;
	}
	if (ED_IS_SHORTCUT("script_text_editor/move_down", key_event)) {
		move_lines_down();
		accept_event();
		return;
	}
	if (ED_IS_SHORTCUT("script_text_editor/delete_line", key_event)) {
		delete_lines();
		accept_event();
		return;
	}
	if (ED_IS_SHORTCUT("script_text_editor/duplicate_selection", key_event)) {
		duplicate_selection();
		accept_event();
		return;
	}
	if (ED_IS_SHORTCUT("script_text_editor/duplicate_lines", key_event)) {
		text_editor->duplicate_lines();
		accept_event();
		return;
	}
}

void CodeTextEditor::_text_editor_gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		if (mb->is_pressed() && mb->is_command_or_control_pressed()) {
			if (mb->get_button_index() == MouseButton::WHEEL_UP) {
				_zoom_in();
			} else if (mb->get_button_index() == MouseButton::WHEEL_DOWN) {
				_zoom_out();
			}
		}
	}

	Ref<InputEventMagnifyGesture> magnify_gesture = p_event;
	if (magnify_gesture.is_valid()) {
		font_size = text_editor->get_theme_font_size(SNAME("font_size"));
		font_size *= powf(magnify_gesture->get_factor(), 0.25);

		_add_font_size((int)font_size - text_editor->get_theme_font_size(SNAME("font_size")));
		return;
	}

	Ref<InputEventKey> k = p_event;

	if (k.is_valid()) {
		if (k->is_pressed()) {
			if (ED_IS_SHORTCUT("script_editor/zoom_in", p_event)) {
				_zoom_in();
				accept_event();
			}
			if (ED_IS_SHORTCUT("script_editor/zoom_out", p_event)) {
				_zoom_out();
				accept_event();
			}
			if (ED_IS_SHORTCUT("script_editor/reset_zoom", p_event)) {
				_reset_zoom();
				accept_event();
			}
		}
	}
}

void CodeTextEditor::_zoom_in() {
	font_resize_val += MAX(EDSCALE, 1.0f);
	_zoom_changed();
}

void CodeTextEditor::_zoom_out() {
	font_resize_val -= MAX(EDSCALE, 1.0f);
	_zoom_changed();
}

void CodeTextEditor::_zoom_changed() {
	if (font_resize_timer->get_time_left() == 0) {
		font_resize_timer->start();
	}
}

void CodeTextEditor::_reset_zoom() {
	EditorSettings::get_singleton()->set("interface/editor/code_font_size", 14);
	text_editor->add_theme_font_size_override("font_size", 14 * EDSCALE);
}

void CodeTextEditor::_line_col_changed() {
	if (!code_complete_timer->is_stopped() && code_complete_timer_line != text_editor->get_caret_line()) {
		code_complete_timer->stop();
	}

	String line = text_editor->get_line(text_editor->get_caret_line());

	int positional_column = 0;
	for (int i = 0; i < text_editor->get_caret_column(); i++) {
		if (line[i] == '\t') {
			positional_column += text_editor->get_indent_size(); // Tab size
		} else {
			positional_column += 1;
		}
	}

	StringBuilder sb;
	sb.append(itos(text_editor->get_caret_line() + 1).lpad(4));
	sb.append(" : ");
	sb.append(itos(positional_column + 1).lpad(3));

	sb.append(" | ");
	sb.append(text_editor->is_indent_using_spaces() ? TTR("Spaces", "Indentation") : TTR("Tabs", "Indentation"));

	line_and_col_txt->set_text(sb.as_string());

	if (find_replace_bar) {
		if (!find_replace_bar->line_col_changed_for_result) {
			find_replace_bar->needs_to_count_results = true;
		}

		find_replace_bar->line_col_changed_for_result = false;
	}
}

void CodeTextEditor::_text_changed() {
	if (code_complete_enabled && text_editor->is_insert_text_operation()) {
		code_complete_timer_line = text_editor->get_caret_line();
		code_complete_timer->start();
	}

	idle->start();

	if (find_replace_bar) {
		find_replace_bar->needs_to_count_results = true;
	}
}

void CodeTextEditor::_code_complete_timer_timeout() {
	if (!is_visible_in_tree()) {
		return;
	}
	text_editor->request_code_completion();
}

void CodeTextEditor::_complete_request() {
	List<ScriptLanguage::CodeCompletionOption> entries;
	String ctext = text_editor->get_text_for_code_completion();
	_code_complete_script(ctext, &entries);
	bool forced = false;
	if (code_complete_func) {
		code_complete_func(code_complete_ud, ctext, &entries, forced);
	}
	if (entries.size() == 0) {
		return;
	}

	for (const ScriptLanguage::CodeCompletionOption &e : entries) {
		Color font_color = completion_font_color;
		if (!e.theme_color_name.is_empty() && EDITOR_GET("text_editor/completion/colorize_suggestions")) {
			font_color = get_theme_color(e.theme_color_name, SNAME("Editor"));
		} else if (e.insert_text.begins_with("\"") || e.insert_text.begins_with("\'")) {
			font_color = completion_string_color;
		} else if (e.insert_text.begins_with("##") || e.insert_text.begins_with("///")) {
			font_color = completion_doc_comment_color;
		} else if (e.insert_text.begins_with("#") || e.insert_text.begins_with("//")) {
			font_color = completion_comment_color;
		}
		text_editor->add_code_completion_option((CodeEdit::CodeCompletionKind)e.kind, e.display, e.insert_text, font_color, _get_completion_icon(e), e.default_value, e.location);
	}
	text_editor->update_code_completion_options(forced);
}

Ref<Texture2D> CodeTextEditor::_get_completion_icon(const ScriptLanguage::CodeCompletionOption &p_option) {
	Ref<Texture2D> tex;
	switch (p_option.kind) {
		case ScriptLanguage::CODE_COMPLETION_KIND_CLASS: {
			if (has_theme_icon(p_option.display, EditorStringName(EditorIcons))) {
				tex = get_editor_theme_icon(p_option.display);
			} else {
				tex = get_editor_theme_icon(SNAME("Object"));
			}
		} break;
		case ScriptLanguage::CODE_COMPLETION_KIND_ENUM:
			tex = get_editor_theme_icon(SNAME("Enum"));
			break;
		case ScriptLanguage::CODE_COMPLETION_KIND_FILE_PATH:
			tex = get_editor_theme_icon(SNAME("File"));
			break;
		case ScriptLanguage::CODE_COMPLETION_KIND_NODE_PATH:
			tex = get_editor_theme_icon(SNAME("NodePath"));
			break;
		case ScriptLanguage::CODE_COMPLETION_KIND_VARIABLE:
			tex = get_editor_theme_icon(SNAME("Variant"));
			break;
		case ScriptLanguage::CODE_COMPLETION_KIND_CONSTANT:
			tex = get_editor_theme_icon(SNAME("MemberConstant"));
			break;
		case ScriptLanguage::CODE_COMPLETION_KIND_MEMBER:
			tex = get_editor_theme_icon(SNAME("MemberProperty"));
			break;
		case ScriptLanguage::CODE_COMPLETION_KIND_SIGNAL:
			tex = get_editor_theme_icon(SNAME("MemberSignal"));
			break;
		case ScriptLanguage::CODE_COMPLETION_KIND_FUNCTION:
			tex = get_editor_theme_icon(SNAME("MemberMethod"));
			break;
		case ScriptLanguage::CODE_COMPLETION_KIND_PLAIN_TEXT:
			tex = get_editor_theme_icon(SNAME("BoxMesh"));
			break;
		default:
			tex = get_editor_theme_icon(SNAME("String"));
			break;
	}
	return tex;
}

void CodeTextEditor::_font_resize_timeout() {
	if (_add_font_size(font_resize_val)) {
		font_resize_val = 0;
	}
}

bool CodeTextEditor::_add_font_size(int p_delta) {
	int old_size = text_editor->get_theme_font_size(SNAME("font_size"));
	int new_size = CLAMP(old_size + p_delta, 8 * EDSCALE, 96 * EDSCALE);

	if (new_size != old_size) {
		EditorSettings::get_singleton()->set("interface/editor/code_font_size", new_size / EDSCALE);
		text_editor->add_theme_font_size_override("font_size", new_size);
	}

	return true;
}

void CodeTextEditor::update_editor_settings() {
	// Theme: Highlighting
	completion_font_color = EDITOR_GET("text_editor/theme/highlighting/completion_font_color");
	completion_string_color = EDITOR_GET("text_editor/theme/highlighting/string_color");
	completion_comment_color = EDITOR_GET("text_editor/theme/highlighting/comment_color");
	completion_doc_comment_color = EDITOR_GET("text_editor/theme/highlighting/doc_comment_color");

	// Appearance: Caret
	text_editor->set_caret_type((TextEdit::CaretType)EDITOR_GET("text_editor/appearance/caret/type").operator int());
	text_editor->set_caret_blink_enabled(EDITOR_GET("text_editor/appearance/caret/caret_blink"));
	text_editor->set_caret_blink_interval(EDITOR_GET("text_editor/appearance/caret/caret_blink_interval"));
	text_editor->set_highlight_current_line(EDITOR_GET("text_editor/appearance/caret/highlight_current_line"));
	text_editor->set_highlight_all_occurrences(EDITOR_GET("text_editor/appearance/caret/highlight_all_occurrences"));

	// Appearance: Gutters
	text_editor->set_draw_line_numbers(EDITOR_GET("text_editor/appearance/gutters/show_line_numbers"));
	text_editor->set_line_numbers_zero_padded(EDITOR_GET("text_editor/appearance/gutters/line_numbers_zero_padded"));

	// Appearance: Minimap
	text_editor->set_draw_minimap(EDITOR_GET("text_editor/appearance/minimap/show_minimap"));
	text_editor->set_minimap_width((int)EDITOR_GET("text_editor/appearance/minimap/minimap_width") * EDSCALE);

	// Appearance: Lines
	text_editor->set_line_folding_enabled(EDITOR_GET("text_editor/appearance/lines/code_folding"));
	text_editor->set_draw_fold_gutter(EDITOR_GET("text_editor/appearance/lines/code_folding"));
	text_editor->set_line_wrapping_mode((TextEdit::LineWrappingMode)EDITOR_GET("text_editor/appearance/lines/word_wrap").operator int());
	text_editor->set_autowrap_mode((TextServer::AutowrapMode)EDITOR_GET("text_editor/appearance/lines/autowrap_mode").operator int());

	// Appearance: Whitespace
	text_editor->set_draw_tabs(EDITOR_GET("text_editor/appearance/whitespace/draw_tabs"));
	text_editor->set_draw_spaces(EDITOR_GET("text_editor/appearance/whitespace/draw_spaces"));
	text_editor->add_theme_constant_override("line_spacing", EDITOR_GET("text_editor/appearance/whitespace/line_spacing"));

	// Behavior: Navigation
	text_editor->set_scroll_past_end_of_file_enabled(EDITOR_GET("text_editor/behavior/navigation/scroll_past_end_of_file"));
	text_editor->set_smooth_scroll_enabled(EDITOR_GET("text_editor/behavior/navigation/smooth_scrolling"));
	text_editor->set_v_scroll_speed(EDITOR_GET("text_editor/behavior/navigation/v_scroll_speed"));
	text_editor->set_drag_and_drop_selection_enabled(EDITOR_GET("text_editor/behavior/navigation/drag_and_drop_selection"));

	// Behavior: indent
	text_editor->set_indent_using_spaces(EDITOR_GET("text_editor/behavior/indent/type"));
	text_editor->set_indent_size(EDITOR_GET("text_editor/behavior/indent/size"));
	text_editor->set_auto_indent_enabled(EDITOR_GET("text_editor/behavior/indent/auto_indent"));

	// Completion
	text_editor->set_auto_brace_completion_enabled(EDITOR_GET("text_editor/completion/auto_brace_complete"));

	// Appearance: Guidelines
	if (EDITOR_GET("text_editor/appearance/guidelines/show_line_length_guidelines")) {
		TypedArray<int> guideline_cols;
		guideline_cols.append(EDITOR_GET("text_editor/appearance/guidelines/line_length_guideline_hard_column"));
		if (EDITOR_GET("text_editor/appearance/guidelines/line_length_guideline_soft_column") != guideline_cols[0]) {
			guideline_cols.append(EDITOR_GET("text_editor/appearance/guidelines/line_length_guideline_soft_column"));
		}
		text_editor->set_line_length_guidelines(guideline_cols);
	} else {
		text_editor->set_line_length_guidelines(TypedArray<int>());
	}
}

void CodeTextEditor::set_find_replace_bar(FindReplaceBar *p_bar) {
	if (find_replace_bar) {
		return;
	}

	find_replace_bar = p_bar;
	find_replace_bar->set_text_edit(this);
	find_replace_bar->connect("error", callable_mp(error, &Label::set_text));
}

void CodeTextEditor::remove_find_replace_bar() {
	if (!find_replace_bar) {
		return;
	}

	find_replace_bar->disconnect("error", callable_mp(error, &Label::set_text));
	find_replace_bar = nullptr;
}

void CodeTextEditor::trim_trailing_whitespace() {
	bool trimmed_whitespace = false;
	for (int i = 0; i < text_editor->get_line_count(); i++) {
		String line = text_editor->get_line(i);
		if (line.ends_with(" ") || line.ends_with("\t")) {
			if (!trimmed_whitespace) {
				text_editor->begin_complex_operation();
				trimmed_whitespace = true;
			}

			int end = 0;
			for (int j = line.length() - 1; j > -1; j--) {
				if (line[j] != ' ' && line[j] != '\t') {
					end = j + 1;
					break;
				}
			}
			text_editor->set_line(i, line.substr(0, end));
		}
	}

	if (trimmed_whitespace) {
		text_editor->merge_overlapping_carets();
		text_editor->end_complex_operation();
		text_editor->queue_redraw();
	}
}

void CodeTextEditor::insert_final_newline() {
	int final_line = text_editor->get_line_count() - 1;

	String line = text_editor->get_line(final_line);

	// Length 0 means it's already an empty line, no need to add a newline.
	if (line.length() > 0 && !line.ends_with("\n")) {
		text_editor->begin_complex_operation();

		line += "\n";
		text_editor->set_line(final_line, line);

		text_editor->end_complex_operation();
		text_editor->queue_redraw();
	}
}

void CodeTextEditor::convert_case(CaseStyle p_case) {
	if (!text_editor->has_selection()) {
		return;
	}
	text_editor->begin_complex_operation();

	Vector<int> caret_edit_order = text_editor->get_caret_index_edit_order();
	for (const int &c : caret_edit_order) {
		if (!text_editor->has_selection(c)) {
			continue;
		}

		int begin = text_editor->get_selection_from_line(c);
		int end = text_editor->get_selection_to_line(c);
		int begin_col = text_editor->get_selection_from_column(c);
		int end_col = text_editor->get_selection_to_column(c);

		for (int i = begin; i <= end; i++) {
			int len = text_editor->get_line(i).length();
			if (i == end) {
				len = end_col;
			}
			if (i == begin) {
				len -= begin_col;
			}
			String new_line = text_editor->get_line(i).substr(i == begin ? begin_col : 0, len);

			switch (p_case) {
				case UPPER: {
					new_line = new_line.to_upper();
				} break;
				case LOWER: {
					new_line = new_line.to_lower();
				} break;
				case CAPITALIZE: {
					new_line = new_line.capitalize();
				} break;
			}

			if (i == begin) {
				new_line = text_editor->get_line(i).left(begin_col) + new_line;
			}
			if (i == end) {
				new_line = new_line + text_editor->get_line(i).substr(end_col);
			}
			text_editor->set_line(i, new_line);
		}
	}
	text_editor->end_complex_operation();
}

void CodeTextEditor::move_lines_up() {
	text_editor->begin_complex_operation();

	Vector<int> caret_edit_order = text_editor->get_caret_index_edit_order();

	// Lists of carets representing each group.
	Vector<Vector<int>> caret_groups;
	Vector<Pair<int, int>> group_borders;

	// Search for groups of carets and their selections residing on the same lines.
	for (int i = 0; i < caret_edit_order.size(); i++) {
		int c = caret_edit_order[i];

		Vector<int> new_group{ c };
		Pair<int, int> group_border;
		group_border.first = _get_affected_lines_from(c);
		group_border.second = _get_affected_lines_to(c);

		for (int j = i; j < caret_edit_order.size() - 1; j++) {
			int c_current = caret_edit_order[j];
			int c_next = caret_edit_order[j + 1];

			int next_start_pos = _get_affected_lines_from(c_next);
			int next_end_pos = _get_affected_lines_to(c_next);

			int current_start_pos = text_editor->has_selection(c_current) ? text_editor->get_selection_from_line(c_current) : text_editor->get_caret_line(c_current);

			i = j;
			if (next_end_pos != current_start_pos && next_end_pos + 1 != current_start_pos) {
				break;
			}
			group_border.first = next_start_pos;
			new_group.push_back(c_next);
			// If the last caret is added to the current group there is no need to process it again.
			if (j + 1 == caret_edit_order.size() - 1) {
				i++;
			}
		}
		group_borders.push_back(group_border);
		caret_groups.push_back(new_group);
	}

	for (int i = group_borders.size() - 1; i >= 0; i--) {
		if (group_borders[i].first - 1 < 0) {
			continue;
		}

		// If the group starts overlapping with the upper group don't move it.
		if (i < group_borders.size() - 1 && group_borders[i].first - 1 <= group_borders[i + 1].second) {
			continue;
		}

		// We have to remember caret positions and selections prior to line swapping.
		Vector<Vector<int>> caret_group_parameters;

		for (int j = 0; j < caret_groups[i].size(); j++) {
			int c = caret_groups[i][j];
			int cursor_line = text_editor->get_caret_line(c);
			int cursor_column = text_editor->get_caret_column(c);

			if (!text_editor->has_selection(c)) {
				caret_group_parameters.push_back(Vector<int>{ -1, -1, -1, -1, cursor_line, cursor_column });
				continue;
			}
			int from_line = text_editor->get_selection_from_line(c);
			int from_col = text_editor->get_selection_from_column(c);
			int to_line = text_editor->get_selection_to_line(c);
			int to_column = text_editor->get_selection_to_column(c);
			caret_group_parameters.push_back(Vector<int>{ from_line, from_col, to_line, to_column, cursor_line, cursor_column });
		}

		for (int line_id = group_borders[i].first; line_id <= group_borders[i].second; line_id++) {
			text_editor->unfold_line(line_id);
			text_editor->unfold_line(line_id - 1);

			text_editor->swap_lines(line_id - 1, line_id);
		}

		for (int j = 0; j < caret_groups[i].size(); j++) {
			int c = caret_groups[i][j];
			Vector<int> caret_parameters = caret_group_parameters[j];
			text_editor->set_caret_line(caret_parameters[4] - 1, c == 0, true, 0, c);
			text_editor->set_caret_column(caret_parameters[5], c == 0, c);

			if (caret_parameters[0] >= 0) {
				text_editor->select(caret_parameters[0] - 1, caret_parameters[1], caret_parameters[2] - 1, caret_parameters[3], c);
			}
		}
	}

	text_editor->end_complex_operation();
	text_editor->merge_overlapping_carets();
	text_editor->queue_redraw();
}

void CodeTextEditor::move_lines_down() {
	text_editor->begin_complex_operation();

	Vector<int> caret_edit_order = text_editor->get_caret_index_edit_order();

	// Lists of carets representing each group.
	Vector<Vector<int>> caret_groups;
	Vector<Pair<int, int>> group_borders;
	Vector<int> group_border_ends;
	// Search for groups of carets and their selections residing on the same lines.
	for (int i = 0; i < caret_edit_order.size(); i++) {
		int c = caret_edit_order[i];

		Vector<int> new_group{ c };
		Pair<int, int> group_border;
		group_border.first = _get_affected_lines_from(c);
		group_border.second = _get_affected_lines_to(c);

		for (int j = i; j < caret_edit_order.size() - 1; j++) {
			int c_current = caret_edit_order[j];
			int c_next = caret_edit_order[j + 1];

			int next_start_pos = _get_affected_lines_from(c_next);
			int next_end_pos = _get_affected_lines_to(c_next);

			int current_start_pos = text_editor->has_selection(c_current) ? text_editor->get_selection_from_line(c_current) : text_editor->get_caret_line(c_current);

			i = j;
			if (next_end_pos == current_start_pos || next_end_pos + 1 == current_start_pos) {
				group_border.first = next_start_pos;
				new_group.push_back(c_next);
				// If the last caret is added to the current group there is no need to process it again.
				if (j + 1 == caret_edit_order.size() - 1) {
					i++;
				}
			} else {
				break;
			}
		}
		group_borders.push_back(group_border);
		group_border_ends.push_back(text_editor->has_selection(c) ? text_editor->get_selection_to_line(c) : text_editor->get_caret_line(c));
		caret_groups.push_back(new_group);
	}

	for (int i = 0; i < group_borders.size(); i++) {
		if (group_border_ends[i] + 1 > text_editor->get_line_count() - 1) {
			continue;
		}

		// If the group starts overlapping with the upper group don't move it.
		if (i > 0 && group_border_ends[i] + 1 >= group_borders[i - 1].first) {
			continue;
		}

		// We have to remember caret positions and selections prior to line swapping.
		Vector<Vector<int>> caret_group_parameters;

		for (int j = 0; j < caret_groups[i].size(); j++) {
			int c = caret_groups[i][j];
			int cursor_line = text_editor->get_caret_line(c);
			int cursor_column = text_editor->get_caret_column(c);

			if (!text_editor->has_selection(c)) {
				caret_group_parameters.push_back(Vector<int>{ -1, -1, -1, -1, cursor_line, cursor_column });
				continue;
			}
			int from_line = text_editor->get_selection_from_line(c);
			int from_col = text_editor->get_selection_from_column(c);
			int to_line = text_editor->get_selection_to_line(c);
			int to_column = text_editor->get_selection_to_column(c);
			caret_group_parameters.push_back(Vector<int>{ from_line, from_col, to_line, to_column, cursor_line, cursor_column });
		}

		for (int line_id = group_borders[i].second; line_id >= group_borders[i].first; line_id--) {
			text_editor->unfold_line(line_id);
			text_editor->unfold_line(line_id + 1);

			text_editor->swap_lines(line_id + 1, line_id);
		}

		for (int j = 0; j < caret_groups[i].size(); j++) {
			int c = caret_groups[i][j];
			Vector<int> caret_parameters = caret_group_parameters[j];
			text_editor->set_caret_line(caret_parameters[4] + 1, c == 0, true, 0, c);
			text_editor->set_caret_column(caret_parameters[5], c == 0, c);

			if (caret_parameters[0] >= 0) {
				text_editor->select(caret_parameters[0] + 1, caret_parameters[1], caret_parameters[2] + 1, caret_parameters[3], c);
			}
		}
	}

	text_editor->merge_overlapping_carets();
	text_editor->end_complex_operation();
	text_editor->queue_redraw();
}

void CodeTextEditor::delete_lines() {
	text_editor->begin_complex_operation();

	Vector<int> caret_edit_order = text_editor->get_caret_index_edit_order();
	Vector<int> lines;
	int last_line = INT_MAX;
	for (const int &c : caret_edit_order) {
		for (int line = _get_affected_lines_to(c); line >= _get_affected_lines_from(c); line--) {
			if (line >= last_line) {
				continue;
			}
			last_line = line;
			lines.append(line);
		}
	}

	for (const int &line : lines) {
		if (line != text_editor->get_line_count() - 1) {
			text_editor->remove_text(line, 0, line + 1, 0);
		} else {
			text_editor->remove_text(line - 1, text_editor->get_line(line - 1).length(), line, text_editor->get_line(line).length());
		}
		// Readjust carets.
		int new_line = MIN(line, text_editor->get_line_count() - 1);
		text_editor->unfold_line(new_line);
		for (const int &c : caret_edit_order) {
			if (text_editor->get_caret_line(c) == line || (text_editor->get_caret_line(c) == line + 1 && text_editor->get_caret_column(c) == 0)) {
				text_editor->deselect(c);
				text_editor->set_caret_line(new_line, c == 0, true, 0, c);
				continue;
			}
			if (text_editor->get_caret_line(c) > line) {
				text_editor->set_caret_line(text_editor->get_caret_line(c) - 1, c == 0, true, 0, c);
				continue;
			}
			break;
		}
	}
	text_editor->merge_overlapping_carets();
	text_editor->end_complex_operation();
}

void CodeTextEditor::duplicate_selection() {
	text_editor->begin_complex_operation();

	Vector<int> caret_edit_order = text_editor->get_caret_index_edit_order();
	for (const int &c : caret_edit_order) {
		const int cursor_column = text_editor->get_caret_column(c);
		int from_line = text_editor->get_caret_line(c);
		int to_line = text_editor->get_caret_line(c);
		int from_column = 0;
		int to_column = 0;
		int cursor_new_line = to_line + 1;
		int cursor_new_column = text_editor->get_caret_column(c);
		String new_text = "\n" + text_editor->get_line(from_line);
		bool selection_active = false;

		text_editor->set_caret_column(text_editor->get_line(from_line).length(), c == 0, c);
		if (text_editor->has_selection(c)) {
			from_column = text_editor->get_selection_from_column(c);
			to_column = text_editor->get_selection_to_column(c);

			from_line = text_editor->get_selection_from_line(c);
			to_line = text_editor->get_selection_to_line(c);
			cursor_new_line = to_line + text_editor->get_caret_line(c) - from_line;
			cursor_new_column = to_column == cursor_column ? 2 * to_column - from_column : to_column;
			new_text = text_editor->get_selected_text(c);
			selection_active = true;

			text_editor->set_caret_line(to_line, c == 0, true, 0, c);
			text_editor->set_caret_column(to_column, c == 0, c);
		}

		for (int i = from_line; i <= to_line; i++) {
			text_editor->unfold_line(i);
		}
		text_editor->deselect(c);
		text_editor->insert_text_at_caret(new_text, c);
		text_editor->set_caret_line(cursor_new_line, c == 0, true, 0, c);
		text_editor->set_caret_column(cursor_new_column, c == 0, c);
		if (selection_active) {
			text_editor->select(to_line, to_column, 2 * to_line - from_line, to_line == from_line ? 2 * to_column - from_column : to_column, c);
		}
	}
	text_editor->merge_overlapping_carets();
	text_editor->end_complex_operation();
	text_editor->queue_redraw();
}

void CodeTextEditor::toggle_inline_comment(const String &delimiter) {
	text_editor->begin_complex_operation();

	Vector<int> caret_edit_order = text_editor->get_caret_index_edit_order();
	caret_edit_order.reverse();
	int last_line = -1;
	int folded_to = 0;
	for (const int &c1 : caret_edit_order) {
		int from = _get_affected_lines_from(c1);
		from += from == last_line ? 1 + folded_to : 0;
		int to = _get_affected_lines_to(c1);
		last_line = to;
		// If last line is folded, extends to the end of the folded section
		if (text_editor->is_line_folded(to)) {
			folded_to = text_editor->get_next_visible_line_offset_from(to + 1, 1) - 1;
			to += folded_to;
		}
		// Check first if there's any uncommented lines in selection.
		bool is_commented = true;
		bool is_all_empty = true;
		for (int line = from; line <= to; line++) {
			// `+ delimiter.length()` here because comment delimiter is not actually `in comment` so we check first character after it
			int delimiter_idx = text_editor->is_in_comment(line, text_editor->get_first_non_whitespace_column(line) + delimiter.length());
			// Empty lines should not be counted.
			bool is_empty = text_editor->get_line(line).strip_edges().is_empty();
			is_all_empty = is_all_empty && is_empty;
			// get_delimiter_start_key will return `##` instead of `#` when there is multiple comment delimiter in a line.
			if (!is_empty && (delimiter_idx == -1 || !text_editor->get_delimiter_start_key(delimiter_idx).begins_with(delimiter))) {
				is_commented = false;
				break;
			}
		}

		// Special case for commenting empty lines, treat it/them as uncommented lines.
		is_commented = is_commented && !is_all_empty;

		// Caret positions need to be saved since they could be moved at the eol.
		Vector<int> caret_cols;
		Vector<int> selection_to_cols;
		for (const int &c2 : caret_edit_order) {
			if (text_editor->get_caret_line(c2) >= from && text_editor->get_caret_line(c2) <= to) {
				caret_cols.append(text_editor->get_caret_column(c2));
			}
			if (text_editor->has_selection(c2) && text_editor->get_selection_to_line(c2) >= from && text_editor->get_selection_to_line(c2) <= to) {
				selection_to_cols.append(text_editor->get_selection_to_column(c2));
			}
		}

		// Comment/uncomment.
		for (int line = from; line <= to; line++) {
			String line_text = text_editor->get_line(line);
			if (is_all_empty) {
				text_editor->set_line(line, delimiter);
				continue;
			}

			if (is_commented) {
				text_editor->set_line(line, line_text.replace_first(delimiter, ""));
			} else {
				text_editor->set_line(line, line_text.insert(text_editor->get_first_non_whitespace_column(line), delimiter));
			}
		}

		// Readjust carets and selections.
		int caret_i = 0;
		int selection_i = 0;
		int offset = (is_commented ? -1 : 1) * delimiter.length();
		for (const int &c2 : caret_edit_order) {
			bool is_line_selection = text_editor->has_selection(c2) && text_editor->get_selection_from_line(c2) < text_editor->get_selection_to_line(c2);
			if (text_editor->get_caret_line(c2) >= from && text_editor->get_caret_line(c2) <= to) {
				int caret_col = caret_cols[caret_i++];
				caret_col += (is_line_selection && caret_col == 0) ? 0 : offset;
				text_editor->set_caret_column(caret_col, c2 == 0, c2);
			}
			if (text_editor->has_selection(c2) && text_editor->get_selection_to_line(c2) >= from && text_editor->get_selection_to_line(c2) <= to) {
				int from_col = text_editor->get_selection_from_column(c2);
				from_col += (is_line_selection && from_col == 0) ? 0 : offset;
				int to_col = selection_to_cols[selection_i++];
				to_col += (to_col == 0) ? 0 : offset;
				text_editor->select(
						text_editor->get_selection_from_line(c2), from_col,
						text_editor->get_selection_to_line(c2), to_col, c2);
			}
		}
	}
	text_editor->merge_overlapping_carets();
	text_editor->end_complex_operation();
	text_editor->queue_redraw();
}

void CodeTextEditor::goto_line(int p_line) {
	text_editor->remove_secondary_carets();
	text_editor->deselect();
	text_editor->unfold_line(p_line);
	text_editor->call_deferred(SNAME("set_caret_line"), p_line);
}

void CodeTextEditor::goto_line_selection(int p_line, int p_begin, int p_end) {
	text_editor->remove_secondary_carets();
	text_editor->unfold_line(p_line);
	text_editor->call_deferred(SNAME("set_caret_line"), p_line);
	text_editor->call_deferred(SNAME("set_caret_column"), p_begin);
	text_editor->select(p_line, p_begin, p_line, p_end);
}

void CodeTextEditor::goto_line_centered(int p_line) {
	goto_line(p_line);
	text_editor->call_deferred(SNAME("center_viewport_to_caret"));
}

void CodeTextEditor::set_executing_line(int p_line) {
	text_editor->set_line_as_executing(p_line, true);
}

void CodeTextEditor::clear_executing_line() {
	text_editor->clear_executing_lines();
}

Variant CodeTextEditor::get_edit_state() {
	Dictionary state;
	state.merge(get_navigation_state());

	state["folded_lines"] = text_editor->get_folded_lines();
	state["breakpoints"] = text_editor->get_breakpointed_lines();
	state["bookmarks"] = text_editor->get_bookmarked_lines();

	Ref<EditorSyntaxHighlighter> syntax_highlighter = text_editor->get_syntax_highlighter();
	state["syntax_highlighter"] = syntax_highlighter->_get_name();

	return state;
}

void CodeTextEditor::set_edit_state(const Variant &p_state) {
	Dictionary state = p_state;

	/* update the row first as it sets the column to 0 */
	text_editor->set_caret_line(state["row"]);
	text_editor->set_caret_column(state["column"]);
	text_editor->set_v_scroll(state["scroll_position"]);
	text_editor->set_h_scroll(state["h_scroll_position"]);

	if (state.get("selection", false)) {
		text_editor->select(state["selection_from_line"], state["selection_from_column"], state["selection_to_line"], state["selection_to_column"]);
	} else {
		text_editor->deselect();
	}

	if (state.has("folded_lines")) {
		Vector<int> folded_lines = state["folded_lines"];
		for (int i = 0; i < folded_lines.size(); i++) {
			text_editor->fold_line(folded_lines[i]);
		}
	}

	if (state.has("breakpoints")) {
		Array breakpoints = state["breakpoints"];
		for (int i = 0; i < breakpoints.size(); i++) {
			text_editor->set_line_as_breakpoint(breakpoints[i], true);
		}
	}

	if (state.has("bookmarks")) {
		Array bookmarks = state["bookmarks"];
		for (int i = 0; i < bookmarks.size(); i++) {
			text_editor->set_line_as_bookmarked(bookmarks[i], true);
		}
	}
}

Variant CodeTextEditor::get_navigation_state() {
	Dictionary state;

	state["scroll_position"] = text_editor->get_v_scroll();
	state["h_scroll_position"] = text_editor->get_h_scroll();
	state["column"] = text_editor->get_caret_column();
	state["row"] = text_editor->get_caret_line();

	state["selection"] = get_text_editor()->has_selection();
	if (get_text_editor()->has_selection()) {
		state["selection_from_line"] = text_editor->get_selection_from_line();
		state["selection_from_column"] = text_editor->get_selection_from_column();
		state["selection_to_line"] = text_editor->get_selection_to_line();
		state["selection_to_column"] = text_editor->get_selection_to_column();
	}

	return state;
}

void CodeTextEditor::set_error(const String &p_error) {
	error->set_text(p_error);
	if (!p_error.is_empty()) {
		error->set_default_cursor_shape(CURSOR_POINTING_HAND);
	} else {
		error->set_default_cursor_shape(CURSOR_ARROW);
	}
}

void CodeTextEditor::set_error_pos(int p_line, int p_column) {
	error_line = p_line;
	error_column = p_column;
}

Point2i CodeTextEditor::get_error_pos() const {
	return Point2i(error_line, error_column);
}

void CodeTextEditor::goto_error() {
	if (!error->get_text().is_empty()) {
		if (text_editor->get_line_count() != error_line) {
			text_editor->unfold_line(error_line);
		}
		text_editor->remove_secondary_carets();
		text_editor->set_caret_line(error_line);
		text_editor->set_caret_column(error_column);
		text_editor->center_viewport_to_caret();
	}
}

void CodeTextEditor::_update_text_editor_theme() {
	emit_signal(SNAME("load_theme_settings"));

	error->begin_bulk_theme_override();
	error->add_theme_font_override(SNAME("font"), get_theme_font(SNAME("status_source"), EditorStringName(EditorFonts)));
	error->add_theme_font_size_override(SNAME("font_size"), get_theme_font_size(SNAME("status_source_size"), EditorStringName(EditorFonts)));
	error->add_theme_color_override(SNAME("font_color"), get_theme_color(SNAME("error_color"), EditorStringName(Editor)));

	Ref<Font> status_bar_font = get_theme_font(SNAME("status_source"), EditorStringName(EditorFonts));
	int status_bar_font_size = get_theme_font_size(SNAME("status_source_size"), EditorStringName(EditorFonts));
	error->add_theme_font_override("font", status_bar_font);
	error->add_theme_font_size_override("font_size", status_bar_font_size);
	error->end_bulk_theme_override();

	int count = status_bar->get_child_count();
	for (int i = 0; i < count; i++) {
		Control *n = Object::cast_to<Control>(status_bar->get_child(i));
		if (n) {
			n->add_theme_font_override("font", status_bar_font);
			n->add_theme_font_size_override("font_size", status_bar_font_size);
		}
	}
}

void CodeTextEditor::_on_settings_change() {
	_apply_settings_change();
}

void CodeTextEditor::_apply_settings_change() {
	_update_text_editor_theme();

	font_size = EDITOR_GET("interface/editor/code_font_size");
	int ot_mode = EDITOR_GET("interface/editor/code_font_contextual_ligatures");

	Ref<FontVariation> fc = text_editor->get_theme_font(SNAME("font"));
	if (fc.is_valid()) {
		switch (ot_mode) {
			case 1: { // Disable ligatures.
				Dictionary ftrs;
				ftrs[TS->name_to_tag("calt")] = 0;
				fc->set_opentype_features(ftrs);
			} break;
			case 2: { // Custom.
				Vector<String> subtag = String(EDITOR_GET("interface/editor/code_font_custom_opentype_features")).split(",");
				Dictionary ftrs;
				for (int i = 0; i < subtag.size(); i++) {
					Vector<String> subtag_a = subtag[i].split("=");
					if (subtag_a.size() == 2) {
						ftrs[TS->name_to_tag(subtag_a[0])] = subtag_a[1].to_int();
					} else if (subtag_a.size() == 1) {
						ftrs[TS->name_to_tag(subtag_a[0])] = 1;
					}
				}
				fc->set_opentype_features(ftrs);
			} break;
			default: { // Enabled.
				Dictionary ftrs;
				ftrs[TS->name_to_tag("calt")] = 1;
				fc->set_opentype_features(ftrs);
			} break;
		}
	}

	text_editor->set_code_hint_draw_below(EDITOR_GET("text_editor/completion/put_callhint_tooltip_below_current_line"));

	code_complete_enabled = EDITOR_GET("text_editor/completion/code_complete_enabled");
	code_complete_timer->set_wait_time(EDITOR_GET("text_editor/completion/code_complete_delay"));
	idle->set_wait_time(EDITOR_GET("text_editor/completion/idle_parse_delay"));
}

void CodeTextEditor::_text_changed_idle_timeout() {
	_validate_script();
	emit_signal(SNAME("validate_script"));
}

void CodeTextEditor::validate_script() {
	idle->start();
}

void CodeTextEditor::_error_button_pressed() {
	_set_show_errors_panel(!is_errors_panel_opened);
	_set_show_warnings_panel(false);
}

void CodeTextEditor::_warning_button_pressed() {
	_set_show_warnings_panel(!is_warnings_panel_opened);
	_set_show_errors_panel(false);
}

void CodeTextEditor::_set_show_errors_panel(bool p_show) {
	is_errors_panel_opened = p_show;
	emit_signal(SNAME("show_errors_panel"), p_show);
}

void CodeTextEditor::_set_show_warnings_panel(bool p_show) {
	is_warnings_panel_opened = p_show;
	emit_signal(SNAME("show_warnings_panel"), p_show);
}

void CodeTextEditor::_toggle_scripts_pressed() {
	ScriptEditor::get_singleton()->toggle_scripts_panel();
	update_toggle_scripts_button();
}

int CodeTextEditor::_get_affected_lines_from(int p_caret) {
	return text_editor->has_selection(p_caret) ? text_editor->get_selection_from_line(p_caret) : text_editor->get_caret_line(p_caret);
}

int CodeTextEditor::_get_affected_lines_to(int p_caret) {
	if (!text_editor->has_selection(p_caret)) {
		return text_editor->get_caret_line(p_caret);
	}
	int line = text_editor->get_selection_to_line(p_caret);
	// Don't affect a line with no selected characters.
	if (text_editor->get_selection_to_column(p_caret) == 0) {
		line--;
	}
	return line;
}

void CodeTextEditor::_error_pressed(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		goto_error();
	}
}

void CodeTextEditor::_update_status_bar_theme() {
	error_button->set_icon(get_editor_theme_icon(SNAME("StatusError")));
	warning_button->set_icon(get_editor_theme_icon(SNAME("NodeWarning")));

	error_button->begin_bulk_theme_override();
	error_button->add_theme_color_override("font_color", get_theme_color(SNAME("error_color"), EditorStringName(Editor)));
	error_button->add_theme_font_override("font", get_theme_font(SNAME("status_source"), EditorStringName(EditorFonts)));
	error_button->add_theme_font_size_override("font_size", get_theme_font_size(SNAME("status_source_size"), EditorStringName(EditorFonts)));
	error_button->end_bulk_theme_override();

	warning_button->begin_bulk_theme_override();
	warning_button->add_theme_color_override("font_color", get_theme_color(SNAME("warning_color"), EditorStringName(Editor)));
	warning_button->add_theme_font_override("font", get_theme_font(SNAME("status_source"), EditorStringName(EditorFonts)));
	warning_button->add_theme_font_size_override("font_size", get_theme_font_size(SNAME("status_source_size"), EditorStringName(EditorFonts)));
	warning_button->end_bulk_theme_override();

	line_and_col_txt->begin_bulk_theme_override();
	line_and_col_txt->add_theme_font_override("font", get_theme_font(SNAME("status_source"), EditorStringName(EditorFonts)));
	line_and_col_txt->add_theme_font_size_override("font_size", get_theme_font_size(SNAME("status_source_size"), EditorStringName(EditorFonts)));
	line_and_col_txt->end_bulk_theme_override();
}

void CodeTextEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			_update_status_bar_theme();
			if (toggle_scripts_button->is_visible()) {
				update_toggle_scripts_button();
			}
			_update_text_editor_theme();
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (toggle_scripts_button->is_visible()) {
				update_toggle_scripts_button();
			}
			set_process_input(is_visible_in_tree());
		} break;

		case NOTIFICATION_PREDELETE: {
			if (find_replace_bar) {
				find_replace_bar->set_text_edit(nullptr);
			}
		} break;
	}
}

void CodeTextEditor::set_error_count(int p_error_count) {
	error_button->set_text(itos(p_error_count));
	error_button->set_visible(p_error_count > 0);
	if (!p_error_count) {
		_set_show_errors_panel(false);
	}
}

void CodeTextEditor::set_warning_count(int p_warning_count) {
	warning_button->set_text(itos(p_warning_count));
	warning_button->set_visible(p_warning_count > 0);
	if (!p_warning_count) {
		_set_show_warnings_panel(false);
	}
}

void CodeTextEditor::toggle_bookmark() {
	Vector<int> caret_edit_order = text_editor->get_caret_index_edit_order();
	caret_edit_order.reverse();
	int last_line = -1;
	for (const int &c : caret_edit_order) {
		int from = text_editor->has_selection(c) ? text_editor->get_selection_from_line(c) : text_editor->get_caret_line(c);
		from += from == last_line ? 1 : 0;
		int to = text_editor->has_selection(c) ? text_editor->get_selection_to_line(c) : text_editor->get_caret_line(c);
		if (to < from) {
			continue;
		}
		// Check first if there's any bookmarked lines in the selection.
		bool selection_has_bookmarks = false;
		for (int line = from; line <= to; line++) {
			if (text_editor->is_line_bookmarked(line)) {
				selection_has_bookmarks = true;
				break;
			}
		}

		// Set bookmark on caret or remove all bookmarks from the selection.
		if (!selection_has_bookmarks) {
			if (text_editor->get_caret_line(c) != last_line) {
				text_editor->set_line_as_bookmarked(text_editor->get_caret_line(c), true);
			}
		} else {
			for (int line = from; line <= to; line++) {
				text_editor->set_line_as_bookmarked(line, false);
			}
		}
		last_line = to;
	}
}

void CodeTextEditor::goto_next_bookmark() {
	PackedInt32Array bmarks = text_editor->get_bookmarked_lines();
	if (bmarks.size() <= 0) {
		return;
	}

	int current_line = text_editor->get_caret_line();
	int bmark_idx = 0;
	if (current_line < (int)bmarks[bmarks.size() - 1]) {
		while (bmark_idx < bmarks.size() && bmarks[bmark_idx] <= current_line) {
			bmark_idx++;
		}
	}
	goto_line_centered(bmarks[bmark_idx]);
}

void CodeTextEditor::goto_prev_bookmark() {
	PackedInt32Array bmarks = text_editor->get_bookmarked_lines();
	if (bmarks.size() <= 0) {
		return;
	}

	int current_line = text_editor->get_caret_line();
	int bmark_idx = bmarks.size() - 1;
	if (current_line > (int)bmarks[0]) {
		while (bmark_idx >= 0 && bmarks[bmark_idx] >= current_line) {
			bmark_idx--;
		}
	}
	goto_line_centered(bmarks[bmark_idx]);
}

void CodeTextEditor::remove_all_bookmarks() {
	text_editor->clear_bookmarked_lines();
}

void CodeTextEditor::_bind_methods() {
	ADD_SIGNAL(MethodInfo("validate_script"));
	ADD_SIGNAL(MethodInfo("load_theme_settings"));
	ADD_SIGNAL(MethodInfo("show_errors_panel"));
	ADD_SIGNAL(MethodInfo("show_warnings_panel"));
}

void CodeTextEditor::set_code_complete_func(CodeTextEditorCodeCompleteFunc p_code_complete_func, void *p_ud) {
	code_complete_func = p_code_complete_func;
	code_complete_ud = p_ud;
}

void CodeTextEditor::show_toggle_scripts_button() {
	toggle_scripts_button->show();
}

void CodeTextEditor::update_toggle_scripts_button() {
	if (is_layout_rtl()) {
		toggle_scripts_button->set_icon(get_editor_theme_icon(ScriptEditor::get_singleton()->is_scripts_panel_toggled() ? SNAME("Forward") : SNAME("Back")));
	} else {
		toggle_scripts_button->set_icon(get_editor_theme_icon(ScriptEditor::get_singleton()->is_scripts_panel_toggled() ? SNAME("Back") : SNAME("Forward")));
	}
	toggle_scripts_button->set_tooltip_text(vformat("%s (%s)", TTR("Toggle Scripts Panel"), ED_GET_SHORTCUT("script_editor/toggle_scripts_panel")->get_as_text()));
}

CodeTextEditor::CodeTextEditor() {
	code_complete_func = nullptr;
	ED_SHORTCUT("script_editor/zoom_in", TTR("Zoom In"), KeyModifierMask::CMD_OR_CTRL | Key::EQUAL);
	ED_SHORTCUT("script_editor/zoom_out", TTR("Zoom Out"), KeyModifierMask::CMD_OR_CTRL | Key::MINUS);
	ED_SHORTCUT_ARRAY("script_editor/reset_zoom", TTR("Reset Zoom"),
			{ int32_t(KeyModifierMask::CMD_OR_CTRL | Key::KEY_0), int32_t(KeyModifierMask::CMD_OR_CTRL | Key::KP_0) });

	text_editor = memnew(CodeEdit);
	add_child(text_editor);
	text_editor->set_v_size_flags(SIZE_EXPAND_FILL);
	text_editor->set_structured_text_bidi_override(TextServer::STRUCTURED_TEXT_GDSCRIPT);
	text_editor->set_draw_bookmarks_gutter(true);

	int ot_mode = EDITOR_GET("interface/editor/code_font_contextual_ligatures");
	Ref<FontVariation> fc = text_editor->get_theme_font(SNAME("font"));
	if (fc.is_valid()) {
		switch (ot_mode) {
			case 1: { // Disable ligatures.
				Dictionary ftrs;
				ftrs[TS->name_to_tag("calt")] = 0;
				fc->set_opentype_features(ftrs);
			} break;
			case 2: { // Custom.
				Vector<String> subtag = String(EDITOR_GET("interface/editor/code_font_custom_opentype_features")).split(",");
				Dictionary ftrs;
				for (int i = 0; i < subtag.size(); i++) {
					Vector<String> subtag_a = subtag[i].split("=");
					if (subtag_a.size() == 2) {
						ftrs[TS->name_to_tag(subtag_a[0])] = subtag_a[1].to_int();
					} else if (subtag_a.size() == 1) {
						ftrs[TS->name_to_tag(subtag_a[0])] = 1;
					}
				}
				fc->set_opentype_features(ftrs);
			} break;
			default: { // Enabled.
				Dictionary ftrs;
				ftrs[TS->name_to_tag("calt")] = 1;
				fc->set_opentype_features(ftrs);
			} break;
		}
	}

	text_editor->set_draw_line_numbers(true);
	text_editor->set_highlight_matching_braces_enabled(true);
	text_editor->set_auto_indent_enabled(true);
	text_editor->set_deselect_on_focus_loss_enabled(false);

	status_bar = memnew(HBoxContainer);
	add_child(status_bar);
	status_bar->set_h_size_flags(SIZE_EXPAND_FILL);
	status_bar->set_custom_minimum_size(Size2(0, 24 * EDSCALE)); // Adjust for the height of the warning icon.

	idle = memnew(Timer);
	add_child(idle);
	idle->set_one_shot(true);
	idle->set_wait_time(EDITOR_GET("text_editor/completion/idle_parse_delay"));

	code_complete_enabled = EDITOR_GET("text_editor/completion/code_complete_enabled");
	code_complete_timer = memnew(Timer);
	add_child(code_complete_timer);
	code_complete_timer->set_one_shot(true);
	code_complete_timer->set_wait_time(EDITOR_GET("text_editor/completion/code_complete_delay"));

	error_line = 0;
	error_column = 0;

	toggle_scripts_button = memnew(Button);
	toggle_scripts_button->set_flat(true);
	toggle_scripts_button->connect("pressed", callable_mp(this, &CodeTextEditor::_toggle_scripts_pressed));
	status_bar->add_child(toggle_scripts_button);
	toggle_scripts_button->hide();

	// Error
	ScrollContainer *scroll = memnew(ScrollContainer);
	scroll->set_h_size_flags(SIZE_EXPAND_FILL);
	scroll->set_v_size_flags(SIZE_EXPAND_FILL);
	scroll->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	status_bar->add_child(scroll);

	error = memnew(Label);
	scroll->add_child(error);
	error->set_v_size_flags(SIZE_EXPAND | SIZE_SHRINK_CENTER);
	error->set_mouse_filter(MOUSE_FILTER_STOP);
	error->connect("gui_input", callable_mp(this, &CodeTextEditor::_error_pressed));

	// Errors
	error_button = memnew(Button);
	error_button->set_flat(true);
	status_bar->add_child(error_button);
	error_button->set_v_size_flags(SIZE_EXPAND | SIZE_SHRINK_CENTER);
	error_button->set_default_cursor_shape(CURSOR_POINTING_HAND);
	error_button->connect("pressed", callable_mp(this, &CodeTextEditor::_error_button_pressed));
	error_button->set_tooltip_text(TTR("Errors"));
	set_error_count(0);

	// Warnings
	warning_button = memnew(Button);
	warning_button->set_flat(true);
	status_bar->add_child(warning_button);
	warning_button->set_v_size_flags(SIZE_EXPAND | SIZE_SHRINK_CENTER);
	warning_button->set_default_cursor_shape(CURSOR_POINTING_HAND);
	warning_button->connect("pressed", callable_mp(this, &CodeTextEditor::_warning_button_pressed));
	warning_button->set_tooltip_text(TTR("Warnings"));
	set_warning_count(0);

	// Line and column
	line_and_col_txt = memnew(Label);
	status_bar->add_child(line_and_col_txt);
	line_and_col_txt->set_v_size_flags(SIZE_EXPAND | SIZE_SHRINK_CENTER);
	line_and_col_txt->set_tooltip_text(TTR("Line and column numbers."));
	line_and_col_txt->set_mouse_filter(MOUSE_FILTER_STOP);

	text_editor->connect("gui_input", callable_mp(this, &CodeTextEditor::_text_editor_gui_input));
	text_editor->connect("caret_changed", callable_mp(this, &CodeTextEditor::_line_col_changed));
	text_editor->connect("text_changed", callable_mp(this, &CodeTextEditor::_text_changed));
	text_editor->connect("code_completion_requested", callable_mp(this, &CodeTextEditor::_complete_request));
	TypedArray<String> cs;
	cs.push_back(".");
	cs.push_back(",");
	cs.push_back("(");
	cs.push_back("=");
	cs.push_back("$");
	cs.push_back("@");
	cs.push_back("\"");
	cs.push_back("\'");
	text_editor->set_code_completion_prefixes(cs);
	idle->connect("timeout", callable_mp(this, &CodeTextEditor::_text_changed_idle_timeout));

	code_complete_timer->connect("timeout", callable_mp(this, &CodeTextEditor::_code_complete_timer_timeout));

	font_resize_val = 0;
	font_size = EDITOR_GET("interface/editor/code_font_size");
	font_resize_timer = memnew(Timer);
	add_child(font_resize_timer);
	font_resize_timer->set_one_shot(true);
	font_resize_timer->set_wait_time(0.07);
	font_resize_timer->connect("timeout", callable_mp(this, &CodeTextEditor::_font_resize_timeout));

	EditorSettings::get_singleton()->connect("settings_changed", callable_mp(this, &CodeTextEditor::_on_settings_change));
	add_theme_constant_override("separation", 4 * EDSCALE);
}
