/*************************************************************************/
/*  code_editor.cpp                                                      */
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

#include "code_editor.h"

#include "core/input/input.h"
#include "core/object/message_queue.h"
#include "core/os/keyboard.h"
#include "core/string/string_builder.h"
#include "editor/editor_scale.h"
#include "editor_node.h"
#include "editor_settings.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/separator.h"
#include "scene/resources/font.h"

void GotoLineDialog::popup_find_line(CodeEdit *p_edit) {
	text_editor = p_edit;

	line->set_text(itos(text_editor->get_caret_line()));
	line->select_all();
	popup_centered(Size2(180, 80) * EDSCALE);
	line->grab_focus();
}

int GotoLineDialog::get_line() const {
	return line->get_text().to_int();
}

void GotoLineDialog::ok_pressed() {
	if (get_line() < 1 || get_line() > text_editor->get_line_count()) {
		return;
	}
	text_editor->unfold_line(get_line() - 1);
	text_editor->set_caret_line(get_line() - 1);
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
	if (p_what == NOTIFICATION_READY) {
		find_prev->set_icon(get_theme_icon(SNAME("MoveUp"), SNAME("EditorIcons")));
		find_next->set_icon(get_theme_icon(SNAME("MoveDown"), SNAME("EditorIcons")));
		hide_button->set_normal_texture(get_theme_icon(SNAME("Close"), SNAME("EditorIcons")));
		hide_button->set_hover_texture(get_theme_icon(SNAME("Close"), SNAME("EditorIcons")));
		hide_button->set_pressed_texture(get_theme_icon(SNAME("Close"), SNAME("EditorIcons")));
		hide_button->set_custom_minimum_size(hide_button->get_normal_texture()->get_size());
	} else if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		set_process_unhandled_input(is_visible_in_tree());
	} else if (p_what == EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED) {
		find_prev->set_icon(get_theme_icon(SNAME("MoveUp"), SNAME("EditorIcons")));
		find_next->set_icon(get_theme_icon(SNAME("MoveDown"), SNAME("EditorIcons")));
		hide_button->set_normal_texture(get_theme_icon(SNAME("Close"), SNAME("EditorIcons")));
		hide_button->set_hover_texture(get_theme_icon(SNAME("Close"), SNAME("EditorIcons")));
		hide_button->set_pressed_texture(get_theme_icon(SNAME("Close"), SNAME("EditorIcons")));
		hide_button->set_custom_minimum_size(hide_button->get_normal_texture()->get_size());
	} else if (p_what == NOTIFICATION_THEME_CHANGED) {
		matches_label->add_theme_color_override("font_color", results_count > 0 ? get_theme_color(SNAME("font_color"), SNAME("Label")) : get_theme_color(SNAME("error_color"), SNAME("Editor")));
	} else if (p_what == NOTIFICATION_PREDELETE) {
		if (base_text_editor) {
			base_text_editor->remove_find_replace_bar();
			base_text_editor = nullptr;
		}
	}
}

void FindReplaceBar::unhandled_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventKey> k = p_event;
	if (!k.is_valid() || !k->is_pressed()) {
		return;
	}

	Control *focus_owner = get_focus_owner();
	if (text_editor->has_focus() || (focus_owner && vbc_lineedit->is_ancestor_of(focus_owner))) {
		bool accepted = true;

		switch (k->get_keycode()) {
			case Key::ESCAPE: {
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

bool FindReplaceBar::_search(uint32_t p_flags, int p_from_line, int p_from_col) {
	String text = get_search_text();
	Point2i pos = text_editor->search(text, p_flags, p_from_line, p_from_col);

	if (pos.x != -1) {
		if (!preserve_cursor && !is_selection_only()) {
			text_editor->unfold_line(pos.y);
			text_editor->set_caret_line(pos.y, false);
			text_editor->set_caret_column(pos.x + text.length(), false);
			text_editor->center_viewport_to_caret();
			text_editor->select(pos.y, pos.x, pos.y, pos.x + text.length());
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
	bool selection_enabled = text_editor->has_selection();
	Point2i selection_begin, selection_end;
	if (selection_enabled) {
		selection_begin = Point2i(text_editor->get_selection_from_line(), text_editor->get_selection_from_column());
		selection_end = Point2i(text_editor->get_selection_to_line(), text_editor->get_selection_to_column());
	}

	String replace_text = get_replace_text();
	int search_text_len = get_search_text().length();

	text_editor->begin_complex_operation();
	if (selection_enabled && is_selection_only()) { // To restrict search_current() to selected region
		text_editor->set_caret_line(selection_begin.width);
		text_editor->set_caret_column(selection_begin.height);
	}

	if (search_current()) {
		text_editor->unfold_line(result_line);
		text_editor->select(result_line, result_col, result_line, result_col + search_text_len);

		if (selection_enabled && is_selection_only()) {
			Point2i match_from(result_line, result_col);
			Point2i match_to(result_line, result_col + search_text_len);
			if (!(match_from < selection_begin || match_to > selection_end)) {
				text_editor->insert_text_at_caret(replace_text);
				if (match_to.x == selection_end.x) { // Adjust selection bounds if necessary
					selection_end.y += replace_text.length() - search_text_len;
				}
			}
		} else {
			text_editor->insert_text_at_caret(replace_text);
		}
	}
	text_editor->end_complex_operation();
	results_count = -1;

	if (selection_enabled && is_selection_only()) {
		// Reselect in order to keep 'Replace' restricted to selection
		text_editor->select(selection_begin.x, selection_begin.y, selection_end.x, selection_end.y);
	} else {
		text_editor->deselect();
	}
}

void FindReplaceBar::_replace_all() {
	text_editor->disconnect("text_changed", callable_mp(this, &FindReplaceBar::_editor_text_changed));
	// Line as x so it gets priority in comparison, column as y.
	Point2i orig_cursor(text_editor->get_caret_line(), text_editor->get_caret_column());
	Point2i prev_match = Point2(-1, -1);

	bool selection_enabled = text_editor->has_selection();
	Point2i selection_begin, selection_end;
	if (selection_enabled) {
		selection_begin = Point2i(text_editor->get_selection_from_line(), text_editor->get_selection_from_column());
		selection_end = Point2i(text_editor->get_selection_to_line(), text_editor->get_selection_to_column());
	}

	int vsval = text_editor->get_v_scroll();

	text_editor->set_caret_line(0);
	text_editor->set_caret_column(0);

	String replace_text = get_replace_text();
	int search_text_len = get_search_text().length();

	int rc = 0;

	replace_all_mode = true;

	text_editor->begin_complex_operation();

	if (selection_enabled && is_selection_only()) {
		text_editor->set_caret_line(selection_begin.width);
		text_editor->set_caret_column(selection_begin.height);
	}
	if (search_current()) {
		do {
			// replace area
			Point2i match_from(result_line, result_col);
			Point2i match_to(result_line, result_col + search_text_len);

			if (match_from < prev_match) {
				break; // Done.
			}

			prev_match = Point2i(result_line, result_col + replace_text.length());

			text_editor->unfold_line(result_line);
			text_editor->select(result_line, result_col, result_line, match_to.y);

			if (selection_enabled && is_selection_only()) {
				if (match_from < selection_begin || match_to > selection_end) {
					break; // Done.
				}

				// Replace but adjust selection bounds.
				text_editor->insert_text_at_caret(replace_text);
				if (match_to.x == selection_end.x) {
					selection_end.y += replace_text.length() - search_text_len;
				}

			} else {
				// Just replace.
				text_editor->insert_text_at_caret(replace_text);
			}

			rc++;
		} while (search_next());
	}

	text_editor->end_complex_operation();

	replace_all_mode = false;

	// Restore editor state (selection, cursor, scroll).
	text_editor->set_caret_line(orig_cursor.x);
	text_editor->set_caret_column(orig_cursor.y);

	if (selection_enabled && is_selection_only()) {
		// Reselect.
		text_editor->select(selection_begin.x, selection_begin.y, selection_end.x, selection_end.y);
	} else {
		text_editor->deselect();
	}

	text_editor->set_v_scroll(vsval);
	matches_label->add_theme_color_override("font_color", rc > 0 ? get_theme_color(SNAME("font_color"), SNAME("Label")) : get_theme_color(SNAME("error_color"), SNAME("Editor")));
	matches_label->set_text(vformat(TTR("%d replaced."), rc));

	text_editor->call_deferred(SNAME("connect"), "text_changed", callable_mp(this, &FindReplaceBar::_editor_text_changed));
	results_count = -1;
}

void FindReplaceBar::_get_search_from(int &r_line, int &r_col) {
	r_line = text_editor->get_caret_line();
	r_col = text_editor->get_caret_column();

	if (text_editor->has_selection() && is_selection_only()) {
		return;
	}

	if (r_line == result_line && r_col >= result_col && r_col <= result_col + get_search_text().length()) {
		r_col = result_col;
	}
}

void FindReplaceBar::_update_results_count() {
	if (results_count != -1) {
		return;
	}

	results_count = 0;

	String searched = get_search_text();
	if (searched.is_empty()) {
		return;
	}

	String full_text = text_editor->get_text();

	int from_pos = 0;

	while (true) {
		int pos = is_case_sensitive() ? full_text.find(searched, from_pos) : full_text.findn(searched, from_pos);
		if (pos == -1) {
			break;
		}

		int pos_subsequent = pos + searched.length();
		if (is_whole_words()) {
			from_pos = pos + 1; // Making sure we won't hit the same match next time, if we get out via a continue.
			if (pos > 0 && !(is_symbol(full_text[pos - 1]) || full_text[pos - 1] == '\n')) {
				continue;
			}
			if (pos_subsequent < full_text.length() && !(is_symbol(full_text[pos_subsequent]) || full_text[pos_subsequent] == '\n')) {
				continue;
			}
		}

		results_count++;
		from_pos = pos_subsequent;
	}
}

void FindReplaceBar::_update_matches_label() {
	if (search_text->get_text().is_empty() || results_count == -1) {
		matches_label->hide();
	} else {
		matches_label->show();

		matches_label->add_theme_color_override("font_color", results_count > 0 ? get_theme_color(SNAME("font_color"), SNAME("Label")) : get_theme_color(SNAME("error_color"), SNAME("Editor")));
		matches_label->set_text(vformat(results_count == 1 ? TTR("%d match.") : TTR("%d matches."), results_count));
	}
}

bool FindReplaceBar::search_current() {
	uint32_t flags = 0;

	if (is_whole_words()) {
		flags |= TextEdit::SEARCH_WHOLE_WORDS;
	}
	if (is_case_sensitive()) {
		flags |= TextEdit::SEARCH_MATCH_CASE;
	}

	int line, col;
	_get_search_from(line, col);

	return _search(flags, line, col);
}

bool FindReplaceBar::search_prev() {
	if (!is_visible()) {
		popup_search(true);
	}

	uint32_t flags = 0;
	String text = get_search_text();

	if (is_whole_words()) {
		flags |= TextEdit::SEARCH_WHOLE_WORDS;
	}
	if (is_case_sensitive()) {
		flags |= TextEdit::SEARCH_MATCH_CASE;
	}

	flags |= TextEdit::SEARCH_BACKWARDS;

	int line, col;
	_get_search_from(line, col);
	if (text_editor->has_selection()) {
		col--; // Skip currently selected word.
	}

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
	if (!is_visible()) {
		popup_search(true);
	}

	uint32_t flags = 0;
	String text;
	if (replace_all_mode) {
		text = get_replace_text();
	} else {
		text = get_search_text();
	}

	if (is_whole_words()) {
		flags |= TextEdit::SEARCH_WHOLE_WORDS;
	}
	if (is_case_sensitive()) {
		flags |= TextEdit::SEARCH_MATCH_CASE;
	}

	int line, col;
	_get_search_from(line, col);

	if (line == result_line && col == result_col) {
		col += text.length();
		if (col > text_editor->get_line(line).length()) {
			line += 1;
			if (line >= text_editor->get_line_count()) {
				line = 0;
			}
			col = 0;
		}
	}

	return _search(flags, line, col);
}

void FindReplaceBar::_hide_bar() {
	if (replace_text->has_focus() || search_text->has_focus()) {
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

	if (text_editor->has_selection() && !selection_only->is_pressed()) {
		search_text->set_text(text_editor->get_selected_text());
	}

	if (!get_search_text().is_empty()) {
		if (p_focus_replace) {
			replace_text->select_all();
			replace_text->set_caret_column(replace_text->get_text().length());
		} else {
			search_text->select_all();
			search_text->set_caret_column(search_text->get_text().length());
		}

		results_count = -1;
		_update_results_count();
		_update_matches_label();
	}
}

void FindReplaceBar::popup_search(bool p_show_only) {
	replace_text->hide();
	hbc_button_replace->hide();
	hbc_option_replace->hide();

	_show_search(false, p_show_only);
}

void FindReplaceBar::popup_replace() {
	if (!replace_text->is_visible_in_tree()) {
		replace_text->show();
		hbc_button_replace->show();
		hbc_option_replace->show();
	}

	selection_only->set_pressed((text_editor->has_selection() && text_editor->get_selection_from_line() < text_editor->get_selection_to_line()));

	_show_search(is_visible() || text_editor->has_selection());
}

void FindReplaceBar::_search_options_changed(bool p_pressed) {
	results_count = -1;
	search_current();
}

void FindReplaceBar::_editor_text_changed() {
	results_count = -1;
	if (is_visible_in_tree()) {
		preserve_cursor = true;
		search_current();
		preserve_cursor = false;
	}
}

void FindReplaceBar::_search_text_changed(const String &p_text) {
	results_count = -1;
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
	if (selection_only->is_pressed() && text_editor->has_selection()) {
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
	base_text_editor = p_text_editor;
	text_editor = base_text_editor->get_text_editor();
	text_editor->connect("text_changed", callable_mp(this, &FindReplaceBar::_editor_text_changed));

	_update_results_count();
	_update_matches_label();
}

void FindReplaceBar::_bind_methods() {
	ClassDB::bind_method("_search_current", &FindReplaceBar::search_current);

	ADD_SIGNAL(MethodInfo("search"));
	ADD_SIGNAL(MethodInfo("error"));
}

FindReplaceBar::FindReplaceBar() {
	results_count = -1;
	replace_all_mode = false;
	preserve_cursor = false;

	vbc_lineedit = memnew(VBoxContainer);
	add_child(vbc_lineedit);
	vbc_lineedit->set_alignment(ALIGN_CENTER);
	vbc_lineedit->set_h_size_flags(SIZE_EXPAND_FILL);
	VBoxContainer *vbc_button = memnew(VBoxContainer);
	add_child(vbc_button);
	VBoxContainer *vbc_option = memnew(VBoxContainer);
	add_child(vbc_option);

	HBoxContainer *hbc_button_search = memnew(HBoxContainer);
	vbc_button->add_child(hbc_button_search);
	hbc_button_search->set_alignment(ALIGN_END);
	hbc_button_replace = memnew(HBoxContainer);
	vbc_button->add_child(hbc_button_replace);
	hbc_button_replace->set_alignment(ALIGN_END);

	HBoxContainer *hbc_option_search = memnew(HBoxContainer);
	vbc_option->add_child(hbc_option_search);
	hbc_option_replace = memnew(HBoxContainer);
	vbc_option->add_child(hbc_option_replace);

	// search toolbar
	search_text = memnew(LineEdit);
	vbc_lineedit->add_child(search_text);
	search_text->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
	search_text->connect("text_changed", callable_mp(this, &FindReplaceBar::_search_text_changed));
	search_text->connect("text_submitted", callable_mp(this, &FindReplaceBar::_search_text_submitted));

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

	// replace toolbar
	replace_text = memnew(LineEdit);
	vbc_lineedit->add_child(replace_text);
	replace_text->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
	replace_text->connect("text_submitted", callable_mp(this, &FindReplaceBar::_replace_text_submitted));

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
	hide_button->connect("pressed", callable_mp(this, &FindReplaceBar::_hide_bar));
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
		if ((find_replace_bar != nullptr && find_replace_bar->is_visible()) && (find_replace_bar->has_focus() || find_replace_bar->is_ancestor_of(get_focus_owner()))) {
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
}

void CodeTextEditor::_text_editor_gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		if (mb->is_pressed() && mb->is_command_pressed()) {
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
	String line = text_editor->get_line(text_editor->get_caret_line());

	int positional_column = 0;
	for (int i = 0; i < text_editor->get_caret_column(); i++) {
		if (line[i] == '\t') {
			positional_column += text_editor->get_indent_size(); //tab size
		} else {
			positional_column += 1;
		}
	}

	StringBuilder sb;
	sb.append(itos(text_editor->get_caret_line() + 1).lpad(4));
	sb.append(" : ");
	sb.append(itos(positional_column + 1).lpad(3));

	line_and_col_txt->set_text(sb.as_string());
}

void CodeTextEditor::_text_changed() {
	if (text_editor->is_insert_text_operation()) {
		code_complete_timer->start();
	}

	idle->start();
}

void CodeTextEditor::_code_complete_timer_timeout() {
	if (!is_visible_in_tree()) {
		return;
	}
	text_editor->request_code_completion();
}

void CodeTextEditor::_complete_request() {
	List<ScriptCodeCompletionOption> entries;
	String ctext = text_editor->get_text_for_code_completion();
	_code_complete_script(ctext, &entries);
	bool forced = false;
	if (code_complete_func) {
		code_complete_func(code_complete_ud, ctext, &entries, forced);
	}
	if (entries.size() == 0) {
		return;
	}

	for (const ScriptCodeCompletionOption &e : entries) {
		Color font_color = completion_font_color;
		if (e.insert_text.begins_with("\"") || e.insert_text.begins_with("\'")) {
			font_color = completion_string_color;
		} else if (e.insert_text.begins_with("#") || e.insert_text.begins_with("//")) {
			font_color = completion_comment_color;
		}
		text_editor->add_code_completion_option((CodeEdit::CodeCompletionKind)e.kind, e.display, e.insert_text, font_color, _get_completion_icon(e), e.default_value);
	}
	text_editor->update_code_completion_options(forced);
}

Ref<Texture2D> CodeTextEditor::_get_completion_icon(const ScriptCodeCompletionOption &p_option) {
	Ref<Texture2D> tex;
	switch (p_option.kind) {
		case ScriptCodeCompletionOption::KIND_CLASS: {
			if (has_theme_icon(p_option.display, "EditorIcons")) {
				tex = get_theme_icon(p_option.display, "EditorIcons");
			} else {
				tex = get_theme_icon(SNAME("Object"), SNAME("EditorIcons"));
			}
		} break;
		case ScriptCodeCompletionOption::KIND_ENUM:
			tex = get_theme_icon(SNAME("Enum"), SNAME("EditorIcons"));
			break;
		case ScriptCodeCompletionOption::KIND_FILE_PATH:
			tex = get_theme_icon(SNAME("File"), SNAME("EditorIcons"));
			break;
		case ScriptCodeCompletionOption::KIND_NODE_PATH:
			tex = get_theme_icon(SNAME("NodePath"), SNAME("EditorIcons"));
			break;
		case ScriptCodeCompletionOption::KIND_VARIABLE:
			tex = get_theme_icon(SNAME("Variant"), SNAME("EditorIcons"));
			break;
		case ScriptCodeCompletionOption::KIND_CONSTANT:
			tex = get_theme_icon(SNAME("MemberConstant"), SNAME("EditorIcons"));
			break;
		case ScriptCodeCompletionOption::KIND_MEMBER:
			tex = get_theme_icon(SNAME("MemberProperty"), SNAME("EditorIcons"));
			break;
		case ScriptCodeCompletionOption::KIND_SIGNAL:
			tex = get_theme_icon(SNAME("MemberSignal"), SNAME("EditorIcons"));
			break;
		case ScriptCodeCompletionOption::KIND_FUNCTION:
			tex = get_theme_icon(SNAME("MemberMethod"), SNAME("EditorIcons"));
			break;
		case ScriptCodeCompletionOption::KIND_PLAIN_TEXT:
			tex = get_theme_icon(SNAME("BoxMesh"), SNAME("EditorIcons"));
			break;
		default:
			tex = get_theme_icon(SNAME("String"), SNAME("EditorIcons"));
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

	// Appearance: Caret
	text_editor->set_caret_type((TextEdit::CaretType)EditorSettings::get_singleton()->get("text_editor/appearance/caret/type").operator int());
	text_editor->set_caret_blink_enabled(EditorSettings::get_singleton()->get("text_editor/appearance/caret/caret_blink"));
	text_editor->set_caret_blink_speed(EditorSettings::get_singleton()->get("text_editor/appearance/caret/caret_blink_speed"));
	text_editor->set_highlight_current_line(EditorSettings::get_singleton()->get("text_editor/appearance/caret/highlight_current_line"));
	text_editor->set_highlight_all_occurrences(EditorSettings::get_singleton()->get("text_editor/appearance/caret/highlight_all_occurrences"));

	// Appearance: Gutters
	text_editor->set_draw_line_numbers(EditorSettings::get_singleton()->get("text_editor/appearance/gutters/show_line_numbers"));
	text_editor->set_line_numbers_zero_padded(EditorSettings::get_singleton()->get("text_editor/appearance/gutters/line_numbers_zero_padded"));
	text_editor->set_draw_bookmarks_gutter(EditorSettings::get_singleton()->get("text_editor/appearance/gutters/show_bookmark_gutter"));

	// Appearance: Minimap
	text_editor->set_draw_minimap(EditorSettings::get_singleton()->get("text_editor/appearance/minimap/show_minimap"));
	text_editor->set_minimap_width((int)EditorSettings::get_singleton()->get("text_editor/appearance/minimap/minimap_width") * EDSCALE);

	// Appearance: Lines
	text_editor->set_line_folding_enabled(EditorSettings::get_singleton()->get("text_editor/appearance/lines/code_folding"));
	text_editor->set_draw_fold_gutter(EditorSettings::get_singleton()->get("text_editor/appearance/lines/code_folding"));
	text_editor->set_line_wrapping_mode((TextEdit::LineWrappingMode)EditorSettings::get_singleton()->get("text_editor/appearance/lines/word_wrap").operator int());

	// Appearance: Whitespace
	text_editor->set_draw_tabs(EditorSettings::get_singleton()->get("text_editor/appearance/whitespace/draw_tabs"));
	text_editor->set_draw_spaces(EditorSettings::get_singleton()->get("text_editor/appearance/whitespace/draw_spaces"));

	// Behavior: Navigation
	text_editor->set_scroll_past_end_of_file_enabled(EditorSettings::get_singleton()->get("text_editor/behavior/navigation/scroll_past_end_of_file"));
	text_editor->set_smooth_scroll_enabled(EditorSettings::get_singleton()->get("text_editor/behavior/navigation/smooth_scrolling"));
	text_editor->set_v_scroll_speed(EditorSettings::get_singleton()->get("text_editor/behavior/navigation/v_scroll_speed"));

	// Behavior: indent
	text_editor->set_indent_using_spaces(EditorSettings::get_singleton()->get("text_editor/behavior/indent/type"));
	text_editor->set_indent_size(EditorSettings::get_singleton()->get("text_editor/behavior/indent/size"));
	text_editor->set_auto_indent_enabled(EditorSettings::get_singleton()->get("text_editor/behavior/indent/auto_indent"));

	// Completion
	text_editor->set_auto_brace_completion_enabled(EditorSettings::get_singleton()->get("text_editor/completion/auto_brace_complete"));

	// Appearance: Guidelines
	if (EditorSettings::get_singleton()->get("text_editor/appearance/guidelines/show_line_length_guidelines")) {
		TypedArray<int> guideline_cols;
		guideline_cols.append(EditorSettings::get_singleton()->get("text_editor/appearance/guidelines/line_length_guideline_hard_column"));
		if (EditorSettings::get_singleton()->get("text_editor/appearance/guidelines/line_length_guideline_soft_column") != guideline_cols[0]) {
			guideline_cols.append(EditorSettings::get_singleton()->get("text_editor/appearance/guidelines/line_length_guideline_soft_column"));
		}
		text_editor->set_line_length_guidelines(guideline_cols);
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
	bool trimed_whitespace = false;
	for (int i = 0; i < text_editor->get_line_count(); i++) {
		String line = text_editor->get_line(i);
		if (line.ends_with(" ") || line.ends_with("\t")) {
			if (!trimed_whitespace) {
				text_editor->begin_complex_operation();
				trimed_whitespace = true;
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

	if (trimed_whitespace) {
		text_editor->end_complex_operation();
		text_editor->update();
	}
}

void CodeTextEditor::insert_final_newline() {
	int final_line = text_editor->get_line_count() - 1;

	String line = text_editor->get_line(final_line);

	//length 0 means it's already an empty line,
	//no need to add a newline
	if (line.length() > 0 && !line.ends_with("\n")) {
		text_editor->begin_complex_operation();

		line += "\n";
		text_editor->set_line(final_line, line);

		text_editor->end_complex_operation();
		text_editor->update();
	}
}

void CodeTextEditor::convert_indent_to_spaces() {
	int indent_size = EditorSettings::get_singleton()->get("text_editor/behavior/indent/size");
	String indent = "";

	for (int i = 0; i < indent_size; i++) {
		indent += " ";
	}

	int cursor_line = text_editor->get_caret_line();
	int cursor_column = text_editor->get_caret_column();

	bool changed_indentation = false;
	for (int i = 0; i < text_editor->get_line_count(); i++) {
		String line = text_editor->get_line(i);

		if (line.length() <= 0) {
			continue;
		}

		int j = 0;
		while (j < line.length() && (line[j] == ' ' || line[j] == '\t')) {
			if (line[j] == '\t') {
				if (!changed_indentation) {
					text_editor->begin_complex_operation();
					changed_indentation = true;
				}
				if (cursor_line == i && cursor_column > j) {
					cursor_column += indent_size - 1;
				}
				line = line.left(j) + indent + line.substr(j + 1);
			}
			j++;
		}
		if (changed_indentation) {
			text_editor->set_line(i, line);
		}
	}
	if (changed_indentation) {
		text_editor->set_caret_column(cursor_column);
		text_editor->end_complex_operation();
		text_editor->update();
	}
}

void CodeTextEditor::convert_indent_to_tabs() {
	int indent_size = EditorSettings::get_singleton()->get("text_editor/behavior/indent/size");
	indent_size -= 1;

	int cursor_line = text_editor->get_caret_line();
	int cursor_column = text_editor->get_caret_column();

	bool changed_indentation = false;
	for (int i = 0; i < text_editor->get_line_count(); i++) {
		String line = text_editor->get_line(i);

		if (line.length() <= 0) {
			continue;
		}

		int j = 0;
		int space_count = -1;
		while (j < line.length() && (line[j] == ' ' || line[j] == '\t')) {
			if (line[j] != '\t') {
				space_count++;

				if (space_count == indent_size) {
					if (!changed_indentation) {
						text_editor->begin_complex_operation();
						changed_indentation = true;
					}
					if (cursor_line == i && cursor_column > j) {
						cursor_column -= indent_size;
					}
					line = line.left(j - indent_size) + "\t" + line.substr(j + 1);
					j = 0;
					space_count = -1;
				}
			} else {
				space_count = -1;
			}
			j++;
		}
		if (changed_indentation) {
			text_editor->set_line(i, line);
		}
	}
	if (changed_indentation) {
		text_editor->set_caret_column(cursor_column);
		text_editor->end_complex_operation();
		text_editor->update();
	}
}

void CodeTextEditor::convert_case(CaseStyle p_case) {
	if (!text_editor->has_selection()) {
		return;
	}

	text_editor->begin_complex_operation();

	int begin = text_editor->get_selection_from_line();
	int end = text_editor->get_selection_to_line();
	int begin_col = text_editor->get_selection_from_column();
	int end_col = text_editor->get_selection_to_column();

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
	text_editor->end_complex_operation();
}

void CodeTextEditor::move_lines_up() {
	text_editor->begin_complex_operation();
	if (text_editor->has_selection()) {
		int from_line = text_editor->get_selection_from_line();
		int from_col = text_editor->get_selection_from_column();
		int to_line = text_editor->get_selection_to_line();
		int to_column = text_editor->get_selection_to_column();
		int cursor_line = text_editor->get_caret_line();

		for (int i = from_line; i <= to_line; i++) {
			int line_id = i;
			int next_id = i - 1;

			if (line_id == 0 || next_id < 0) {
				return;
			}

			text_editor->unfold_line(line_id);
			text_editor->unfold_line(next_id);

			text_editor->swap_lines(line_id, next_id);
			text_editor->set_caret_line(next_id);
		}
		int from_line_up = from_line > 0 ? from_line - 1 : from_line;
		int to_line_up = to_line > 0 ? to_line - 1 : to_line;
		int cursor_line_up = cursor_line > 0 ? cursor_line - 1 : cursor_line;
		text_editor->select(from_line_up, from_col, to_line_up, to_column);
		text_editor->set_caret_line(cursor_line_up);
	} else {
		int line_id = text_editor->get_caret_line();
		int next_id = line_id - 1;

		if (line_id == 0 || next_id < 0) {
			return;
		}

		text_editor->unfold_line(line_id);
		text_editor->unfold_line(next_id);

		text_editor->swap_lines(line_id, next_id);
		text_editor->set_caret_line(next_id);
	}
	text_editor->end_complex_operation();
	text_editor->update();
}

void CodeTextEditor::move_lines_down() {
	text_editor->begin_complex_operation();
	if (text_editor->has_selection()) {
		int from_line = text_editor->get_selection_from_line();
		int from_col = text_editor->get_selection_from_column();
		int to_line = text_editor->get_selection_to_line();
		int to_column = text_editor->get_selection_to_column();
		int cursor_line = text_editor->get_caret_line();

		for (int i = to_line; i >= from_line; i--) {
			int line_id = i;
			int next_id = i + 1;

			if (line_id == text_editor->get_line_count() - 1 || next_id > text_editor->get_line_count()) {
				return;
			}

			text_editor->unfold_line(line_id);
			text_editor->unfold_line(next_id);

			text_editor->swap_lines(line_id, next_id);
			text_editor->set_caret_line(next_id);
		}
		int from_line_down = from_line < text_editor->get_line_count() ? from_line + 1 : from_line;
		int to_line_down = to_line < text_editor->get_line_count() ? to_line + 1 : to_line;
		int cursor_line_down = cursor_line < text_editor->get_line_count() ? cursor_line + 1 : cursor_line;
		text_editor->select(from_line_down, from_col, to_line_down, to_column);
		text_editor->set_caret_line(cursor_line_down);
	} else {
		int line_id = text_editor->get_caret_line();
		int next_id = line_id + 1;

		if (line_id == text_editor->get_line_count() - 1 || next_id > text_editor->get_line_count()) {
			return;
		}

		text_editor->unfold_line(line_id);
		text_editor->unfold_line(next_id);

		text_editor->swap_lines(line_id, next_id);
		text_editor->set_caret_line(next_id);
	}
	text_editor->end_complex_operation();
	text_editor->update();
}

void CodeTextEditor::_delete_line(int p_line) {
	// this is currently intended to be called within delete_lines()
	// so `begin_complex_operation` is omitted here
	text_editor->set_line(p_line, "");
	if (p_line == 0 && text_editor->get_line_count() > 1) {
		text_editor->set_caret_line(1);
		text_editor->set_caret_column(0);
	}
	text_editor->backspace();
	if (p_line < text_editor->get_line_count()) {
		text_editor->unfold_line(p_line);
	}
	text_editor->set_caret_line(p_line);
}

void CodeTextEditor::delete_lines() {
	text_editor->begin_complex_operation();
	if (text_editor->has_selection()) {
		int to_line = text_editor->get_selection_to_line();
		int from_line = text_editor->get_selection_from_line();
		int count = Math::abs(to_line - from_line) + 1;

		text_editor->set_caret_line(from_line, false);
		for (int i = 0; i < count; i++) {
			_delete_line(from_line);
		}
		text_editor->deselect();
	} else {
		_delete_line(text_editor->get_caret_line());
	}
	text_editor->end_complex_operation();
}

void CodeTextEditor::duplicate_selection() {
	const int cursor_column = text_editor->get_caret_column();
	int from_line = text_editor->get_caret_line();
	int to_line = text_editor->get_caret_line();
	int from_column = 0;
	int to_column = 0;
	int cursor_new_line = to_line + 1;
	int cursor_new_column = text_editor->get_caret_column();
	String new_text = "\n" + text_editor->get_line(from_line);
	bool selection_active = false;

	text_editor->set_caret_column(text_editor->get_line(from_line).length());
	if (text_editor->has_selection()) {
		from_column = text_editor->get_selection_from_column();
		to_column = text_editor->get_selection_to_column();

		from_line = text_editor->get_selection_from_line();
		to_line = text_editor->get_selection_to_line();
		cursor_new_line = to_line + text_editor->get_caret_line() - from_line;
		cursor_new_column = to_column == cursor_column ? 2 * to_column - from_column : to_column;
		new_text = text_editor->get_selected_text();
		selection_active = true;

		text_editor->set_caret_line(to_line);
		text_editor->set_caret_column(to_column);
	}

	text_editor->begin_complex_operation();

	for (int i = from_line; i <= to_line; i++) {
		text_editor->unfold_line(i);
	}
	text_editor->deselect();
	text_editor->insert_text_at_caret(new_text);
	text_editor->set_caret_line(cursor_new_line);
	text_editor->set_caret_column(cursor_new_column);
	if (selection_active) {
		text_editor->select(to_line, to_column, 2 * to_line - from_line, to_line == from_line ? 2 * to_column - from_column : to_column);
	}

	text_editor->end_complex_operation();
	text_editor->update();
}

void CodeTextEditor::toggle_inline_comment(const String &delimiter) {
	text_editor->begin_complex_operation();
	if (text_editor->has_selection()) {
		int begin = text_editor->get_selection_from_line();
		int end = text_editor->get_selection_to_line();

		// End of selection ends on the first column of the last line, ignore it.
		if (text_editor->get_selection_to_column() == 0) {
			end -= 1;
		}

		int col_to = text_editor->get_selection_to_column();
		int cursor_pos = text_editor->get_caret_column();

		// Check if all lines in the selected block are commented.
		bool is_commented = true;
		for (int i = begin; i <= end; i++) {
			if (!text_editor->get_line(i).begins_with(delimiter)) {
				is_commented = false;
				break;
			}
		}
		for (int i = begin; i <= end; i++) {
			String line_text = text_editor->get_line(i);

			if (line_text.strip_edges().is_empty()) {
				line_text = delimiter;
			} else {
				if (is_commented) {
					line_text = line_text.substr(delimiter.length(), line_text.length());
				} else {
					line_text = delimiter + line_text;
				}
			}
			text_editor->set_line(i, line_text);
		}

		// Adjust selection & cursor position.
		int offset = (is_commented ? -1 : 1) * delimiter.length();
		int col_from = text_editor->get_selection_from_column() > 0 ? text_editor->get_selection_from_column() + offset : 0;

		if (is_commented && text_editor->get_caret_column() == text_editor->get_line(text_editor->get_caret_line()).length() + 1) {
			cursor_pos += 1;
		}

		if (text_editor->get_selection_to_column() != 0 && col_to != text_editor->get_line(text_editor->get_selection_to_line()).length() + 1) {
			col_to += offset;
		}

		if (text_editor->get_caret_column() != 0) {
			cursor_pos += offset;
		}

		text_editor->select(begin, col_from, text_editor->get_selection_to_line(), col_to);
		text_editor->set_caret_column(cursor_pos);

	} else {
		int begin = text_editor->get_caret_line();
		String line_text = text_editor->get_line(begin);
		int delimiter_length = delimiter.length();

		int col = text_editor->get_caret_column();
		if (line_text.begins_with(delimiter)) {
			line_text = line_text.substr(delimiter_length, line_text.length());
			col -= delimiter_length;
		} else {
			line_text = delimiter + line_text;
			col += delimiter_length;
		}

		text_editor->set_line(begin, line_text);
		text_editor->set_caret_column(col);
	}
	text_editor->end_complex_operation();
	text_editor->update();
}

void CodeTextEditor::goto_line(int p_line) {
	text_editor->deselect();
	text_editor->unfold_line(p_line);
	text_editor->call_deferred(SNAME("set_caret_line"), p_line);
}

void CodeTextEditor::goto_line_selection(int p_line, int p_begin, int p_end) {
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

	if (state.has("selection")) {
		text_editor->select(state["selection_from_line"], state["selection_from_column"], state["selection_to_line"], state["selection_to_column"]);
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

void CodeTextEditor::set_error(const String &p_error) {
	error->set_text(p_error);
	if (p_error != "") {
		error->set_default_cursor_shape(CURSOR_POINTING_HAND);
	} else {
		error->set_default_cursor_shape(CURSOR_ARROW);
	}
}

void CodeTextEditor::set_error_pos(int p_line, int p_column) {
	error_line = p_line;
	error_column = p_column;
}

void CodeTextEditor::goto_error() {
	if (error->get_text() != "") {
		text_editor->unfold_line(error_line);
		text_editor->set_caret_line(error_line);
		text_editor->set_caret_column(error_column);
		text_editor->center_viewport_to_caret();
	}
}

void CodeTextEditor::_update_text_editor_theme() {
	emit_signal(SNAME("load_theme_settings"));

	error->begin_bulk_theme_override();
	error->add_theme_font_override(SNAME("font"), get_theme_font(SNAME("status_source"), SNAME("EditorFonts")));
	error->add_theme_font_size_override(SNAME("font_size"), get_theme_font_size(SNAME("status_source_size"), SNAME("EditorFonts")));
	error->add_theme_color_override(SNAME("font_color"), get_theme_color(SNAME("error_color"), SNAME("Editor")));

	Ref<Font> status_bar_font = get_theme_font(SNAME("status_source"), SNAME("EditorFonts"));
	int status_bar_font_size = get_theme_font_size(SNAME("status_source_size"), SNAME("EditorFonts"));
	error->add_theme_font_override("font", status_bar_font);
	error->add_theme_font_size_override("font_size", status_bar_font_size);
	int count = status_bar->get_child_count();
	for (int i = 0; i < count; i++) {
		Control *n = Object::cast_to<Control>(status_bar->get_child(i));
		if (n) {
			n->add_theme_font_override("font", status_bar_font);
			n->add_theme_font_size_override("font_size", status_bar_font_size);
		}
	}
	error->end_bulk_theme_override();
}

void CodeTextEditor::_on_settings_change() {
	_apply_settings_change();
}

void CodeTextEditor::_apply_settings_change() {
	_update_text_editor_theme();

	font_size = EditorSettings::get_singleton()->get("interface/editor/code_font_size");

	int ot_mode = EditorSettings::get_singleton()->get("interface/editor/code_font_contextual_ligatures");
	switch (ot_mode) {
		case 1: { // Disable ligatures.
			text_editor->clear_opentype_features();
			text_editor->set_opentype_feature("calt", 0);
		} break;
		case 2: { // Custom.
			text_editor->clear_opentype_features();
			Vector<String> subtag = String(EditorSettings::get_singleton()->get("interface/editor/code_font_custom_opentype_features")).split(",");
			Dictionary ftrs;
			for (int i = 0; i < subtag.size(); i++) {
				Vector<String> subtag_a = subtag[i].split("=");
				if (subtag_a.size() == 2) {
					text_editor->set_opentype_feature(subtag_a[0], subtag_a[1].to_int());
				} else if (subtag_a.size() == 1) {
					text_editor->set_opentype_feature(subtag_a[0], 1);
				}
			}
		} break;
		default: { // Default.
			text_editor->clear_opentype_features();
			text_editor->set_opentype_feature("calt", 1);
		} break;
	}

	text_editor->set_code_hint_draw_below(EDITOR_GET("text_editor/completion/put_callhint_tooltip_below_current_line"));

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

void CodeTextEditor::_error_pressed(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		goto_error();
	}
}

void CodeTextEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			if (toggle_scripts_button->is_visible()) {
				update_toggle_scripts_button();
			}
			_update_text_editor_theme();
		} break;
		case NOTIFICATION_ENTER_TREE: {
			error_button->set_icon(get_theme_icon(SNAME("StatusError"), SNAME("EditorIcons")));
			warning_button->set_icon(get_theme_icon(SNAME("NodeWarning"), SNAME("EditorIcons")));
			add_theme_constant_override("separation", 4 * EDSCALE);
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
		default:
			break;
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
	int line = text_editor->get_caret_line();
	text_editor->set_line_as_bookmarked(line, !text_editor->is_line_bookmarked(line));
}

void CodeTextEditor::goto_next_bookmark() {
	Array bmarks = text_editor->get_bookmarked_lines();
	if (bmarks.size() <= 0) {
		return;
	}

	int line = text_editor->get_caret_line();
	if (line >= (int)bmarks[bmarks.size() - 1]) {
		text_editor->unfold_line(bmarks[0]);
		text_editor->set_caret_line(bmarks[0]);
		text_editor->center_viewport_to_caret();
	} else {
		for (int i = 0; i < bmarks.size(); i++) {
			int bmark_line = bmarks[i];
			if (bmark_line > line) {
				text_editor->unfold_line(bmark_line);
				text_editor->set_caret_line(bmark_line);
				text_editor->center_viewport_to_caret();
				return;
			}
		}
	}
}

void CodeTextEditor::goto_prev_bookmark() {
	Array bmarks = text_editor->get_bookmarked_lines();
	if (bmarks.size() <= 0) {
		return;
	}

	int line = text_editor->get_caret_line();
	if (line <= (int)bmarks[0]) {
		text_editor->unfold_line(bmarks[bmarks.size() - 1]);
		text_editor->set_caret_line(bmarks[bmarks.size() - 1]);
		text_editor->center_viewport_to_caret();
	} else {
		for (int i = bmarks.size() - 1; i >= 0; i--) {
			int bmark_line = bmarks[i];
			if (bmark_line < line) {
				text_editor->unfold_line(bmark_line);
				text_editor->set_caret_line(bmark_line);
				text_editor->center_viewport_to_caret();
				return;
			}
		}
	}
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
		toggle_scripts_button->set_icon(get_theme_icon(ScriptEditor::get_singleton()->is_scripts_panel_toggled() ? SNAME("Forward") : SNAME("Back"), SNAME("EditorIcons")));
	} else {
		toggle_scripts_button->set_icon(get_theme_icon(ScriptEditor::get_singleton()->is_scripts_panel_toggled() ? SNAME("Back") : SNAME("Forward"), SNAME("EditorIcons")));
	}
	toggle_scripts_button->set_tooltip(vformat("%s (%s)", TTR("Toggle Scripts Panel"), ED_GET_SHORTCUT("script_editor/toggle_scripts_panel")->get_as_text()));
}

CodeTextEditor::CodeTextEditor() {
	code_complete_func = nullptr;
	ED_SHORTCUT("script_editor/zoom_in", TTR("Zoom In"), KeyModifierMask::CMD | Key::EQUAL);
	ED_SHORTCUT("script_editor/zoom_out", TTR("Zoom Out"), KeyModifierMask::CMD | Key::MINUS);
	ED_SHORTCUT("script_editor/reset_zoom", TTR("Reset Zoom"), KeyModifierMask::CMD | Key::KEY_0);

	text_editor = memnew(CodeEdit);
	add_child(text_editor);
	text_editor->set_v_size_flags(SIZE_EXPAND_FILL);

	int ot_mode = EditorSettings::get_singleton()->get("interface/editor/code_font_contextual_ligatures");
	switch (ot_mode) {
		case 1: { // Disable ligatures.
			text_editor->clear_opentype_features();
			text_editor->set_opentype_feature("calt", 0);
		} break;
		case 2: { // Custom.
			text_editor->clear_opentype_features();
			Vector<String> subtag = String(EditorSettings::get_singleton()->get("interface/editor/code_font_custom_opentype_features")).split(",");
			Dictionary ftrs;
			for (int i = 0; i < subtag.size(); i++) {
				Vector<String> subtag_a = subtag[i].split("=");
				if (subtag_a.size() == 2) {
					text_editor->set_opentype_feature(subtag_a[0], subtag_a[1].to_int());
				} else if (subtag_a.size() == 1) {
					text_editor->set_opentype_feature(subtag_a[0], 1);
				}
			}
		} break;
		default: { // Default.
			text_editor->clear_opentype_features();
			text_editor->set_opentype_feature("calt", 1);
		} break;
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
	scroll->set_enable_v_scroll(false);
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
	error_button->set_tooltip(TTR("Errors"));

	error_button->add_theme_color_override("font_color", EditorNode::get_singleton()->get_gui_base()->get_theme_color(SNAME("error_color"), SNAME("Editor")));
	error_button->add_theme_font_override("font", EditorNode::get_singleton()->get_gui_base()->get_theme_font(SNAME("status_source"), SNAME("EditorFonts")));
	error_button->add_theme_font_size_override("font_size", EditorNode::get_singleton()->get_gui_base()->get_theme_font_size(SNAME("status_source_size"), SNAME("EditorFonts")));

	is_errors_panel_opened = false;
	set_error_count(0);

	// Warnings
	warning_button = memnew(Button);
	warning_button->set_flat(true);
	status_bar->add_child(warning_button);
	warning_button->set_v_size_flags(SIZE_EXPAND | SIZE_SHRINK_CENTER);
	warning_button->set_default_cursor_shape(CURSOR_POINTING_HAND);
	warning_button->connect("pressed", callable_mp(this, &CodeTextEditor::_warning_button_pressed));
	warning_button->set_tooltip(TTR("Warnings"));

	warning_button->add_theme_color_override("font_color", EditorNode::get_singleton()->get_gui_base()->get_theme_color(SNAME("warning_color"), SNAME("Editor")));
	warning_button->add_theme_font_override("font", EditorNode::get_singleton()->get_gui_base()->get_theme_font(SNAME("status_source"), SNAME("EditorFonts")));
	warning_button->add_theme_font_size_override("font_size", EditorNode::get_singleton()->get_gui_base()->get_theme_font_size(SNAME("status_source_size"), SNAME("EditorFonts")));

	is_warnings_panel_opened = false;
	set_warning_count(0);

	// Line and column
	line_and_col_txt = memnew(Label);
	status_bar->add_child(line_and_col_txt);
	line_and_col_txt->set_v_size_flags(SIZE_EXPAND | SIZE_SHRINK_CENTER);
	line_and_col_txt->add_theme_font_override("font", EditorNode::get_singleton()->get_gui_base()->get_theme_font(SNAME("status_source"), SNAME("EditorFonts")));
	line_and_col_txt->add_theme_font_size_override("font_size", EditorNode::get_singleton()->get_gui_base()->get_theme_font_size(SNAME("status_source_size"), SNAME("EditorFonts")));
	line_and_col_txt->set_tooltip(TTR("Line and column numbers."));
	line_and_col_txt->set_mouse_filter(MOUSE_FILTER_STOP);

	text_editor->connect("gui_input", callable_mp(this, &CodeTextEditor::_text_editor_gui_input));
	text_editor->connect("caret_changed", callable_mp(this, &CodeTextEditor::_line_col_changed));
	text_editor->connect("text_changed", callable_mp(this, &CodeTextEditor::_text_changed));
	text_editor->connect("request_code_completion", callable_mp(this, &CodeTextEditor::_complete_request));
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
	font_size = EditorSettings::get_singleton()->get("interface/editor/code_font_size");
	font_resize_timer = memnew(Timer);
	add_child(font_resize_timer);
	font_resize_timer->set_one_shot(true);
	font_resize_timer->set_wait_time(0.07);
	font_resize_timer->connect("timeout", callable_mp(this, &CodeTextEditor::_font_resize_timeout));

	EditorSettings::get_singleton()->connect("settings_changed", callable_mp(this, &CodeTextEditor::_on_settings_change));
}
