/*************************************************************************/
/*  code_editor.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "core/os/input.h"
#include "core/os/keyboard.h"
#include "core/string_builder.h"
#include "editor/editor_scale.h"
#include "editor_node.h"
#include "editor_settings.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/separator.h"
#include "scene/resources/dynamic_font.h"

void GotoLineDialog::popup_find_line(TextEdit *p_edit) {

	text_editor = p_edit;

	line->set_text(itos(text_editor->cursor_get_line()));
	line->select_all();
	popup_centered(Size2(180, 80) * EDSCALE);
	line->grab_focus();
}

int GotoLineDialog::get_line() const {

	return line->get_text().to_int();
}

void GotoLineDialog::ok_pressed() {

	if (get_line() < 1 || get_line() > text_editor->get_line_count())
		return;
	text_editor->unfold_line(get_line() - 1);
	text_editor->cursor_set_line(get_line() - 1);
	hide();
}

GotoLineDialog::GotoLineDialog() {

	set_title(TTR("Go to Line"));

	VBoxContainer *vbc = memnew(VBoxContainer);
	vbc->set_anchor_and_margin(MARGIN_LEFT, ANCHOR_BEGIN, 8 * EDSCALE);
	vbc->set_anchor_and_margin(MARGIN_TOP, ANCHOR_BEGIN, 8 * EDSCALE);
	vbc->set_anchor_and_margin(MARGIN_RIGHT, ANCHOR_END, -8 * EDSCALE);
	vbc->set_anchor_and_margin(MARGIN_BOTTOM, ANCHOR_END, -8 * EDSCALE);
	add_child(vbc);

	Label *l = memnew(Label);
	l->set_text(TTR("Line Number:"));
	vbc->add_child(l);

	line = memnew(LineEdit);
	vbc->add_child(line);
	register_text_enter(line);
	text_editor = NULL;

	set_hide_on_ok(false);
}

void FindReplaceBar::_notification(int p_what) {

	if (p_what == NOTIFICATION_READY) {

		find_prev->set_icon(get_icon("MoveUp", "EditorIcons"));
		find_next->set_icon(get_icon("MoveDown", "EditorIcons"));
		hide_button->set_normal_texture(get_icon("Close", "EditorIcons"));
		hide_button->set_hover_texture(get_icon("Close", "EditorIcons"));
		hide_button->set_pressed_texture(get_icon("Close", "EditorIcons"));
		hide_button->set_custom_minimum_size(hide_button->get_normal_texture()->get_size());
	} else if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {

		set_process_unhandled_input(is_visible_in_tree());
	} else if (p_what == EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED) {

		find_prev->set_icon(get_icon("MoveUp", "EditorIcons"));
		find_next->set_icon(get_icon("MoveDown", "EditorIcons"));
		hide_button->set_normal_texture(get_icon("Close", "EditorIcons"));
		hide_button->set_hover_texture(get_icon("Close", "EditorIcons"));
		hide_button->set_pressed_texture(get_icon("Close", "EditorIcons"));
		hide_button->set_custom_minimum_size(hide_button->get_normal_texture()->get_size());
	}
}

void FindReplaceBar::_unhandled_input(const Ref<InputEvent> &p_event) {

	Ref<InputEventKey> k = p_event;
	if (k.is_valid()) {

		if (k->is_pressed() && (text_edit->has_focus() || vbc_lineedit->is_a_parent_of(get_focus_owner()))) {

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
			text_edit->unfold_line(line);
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

		text_edit->unfold_line(result_line);
		text_edit->select(result_line, result_col, result_line, result_col + get_search_text().length());
		text_edit->insert_text_at_cursor(get_replace_text());

		text_edit->end_complex_operation();
	}

	search_current();
}

void FindReplaceBar::_replace_all() {

	text_edit->disconnect("text_changed", this, "_editor_text_changed");
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

	if (search_current()) {
		do {
			// replace area
			Point2i match_from(result_line, result_col);
			Point2i match_to(result_line, result_col + search_text_len);

			if (match_from < prev_match) {
				break; // done
			}

			prev_match = Point2i(result_line, result_col + replace_text.length());

			text_edit->unfold_line(result_line);
			text_edit->select(result_line, result_col, result_line, match_to.y);

			if (selection_enabled && is_selection_only()) {
				if (match_from < selection_begin || match_to > selection_end) {
					continue;
				}

				// replace but adjust selection bounds
				text_edit->insert_text_at_cursor(replace_text);
				if (match_to.x == selection_end.x) {
					selection_end.y += replace_text.length() - search_text_len;
				}

			} else {
				// just replace
				text_edit->insert_text_at_cursor(replace_text);
			}

			rc++;
		} while (search_next());
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

	text_edit->call_deferred("connect", "text_changed", this, "_editor_text_changed");
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
	set_error("");
	hide();
}

void FindReplaceBar::_show_search() {

	show();
	search_text->call_deferred("grab_focus");

	if (text_edit->is_selection_active() && !selection_only->is_pressed()) {
		search_text->set_text(text_edit->get_selection_text());
	}

	if (!get_search_text().empty()) {
		search_text->select_all();
		search_text->set_cursor_position(search_text->get_text().length());
		search_current();
	}
}

void FindReplaceBar::popup_search() {

	replace_text->hide();
	hbc_button_replace->hide();
	hbc_option_replace->hide();
	_show_search();
}

void FindReplaceBar::popup_replace() {

	if (!replace_text->is_visible_in_tree()) {
		replace_text->clear();
		replace_text->show();
		hbc_button_replace->show();
		hbc_option_replace->show();
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

	if (Input::get_singleton()->is_key_pressed(KEY_SHIFT)) {
		search_prev();
	} else {
		search_next();
	}
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

	emit_signal("error", p_label);
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
	ADD_SIGNAL(MethodInfo("error"));
}

FindReplaceBar::FindReplaceBar() {

	replace_all_mode = false;
	preserve_cursor = false;

	vbc_lineedit = memnew(VBoxContainer);
	add_child(vbc_lineedit);
	vbc_lineedit->set_h_size_flags(SIZE_EXPAND_FILL);
	VBoxContainer *vbc_button = memnew(VBoxContainer);
	add_child(vbc_button);
	VBoxContainer *vbc_option = memnew(VBoxContainer);
	add_child(vbc_option);

	HBoxContainer *hbc_button_search = memnew(HBoxContainer);
	vbc_button->add_child(hbc_button_search);
	hbc_button_replace = memnew(HBoxContainer);
	vbc_button->add_child(hbc_button_replace);

	HBoxContainer *hbc_option_search = memnew(HBoxContainer);
	vbc_option->add_child(hbc_option_search);
	hbc_option_replace = memnew(HBoxContainer);
	vbc_option->add_child(hbc_option_replace);

	// search toolbar
	search_text = memnew(LineEdit);
	vbc_lineedit->add_child(search_text);
	search_text->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
	search_text->connect("text_changed", this, "_search_text_changed");
	search_text->connect("text_entered", this, "_search_text_entered");

	find_prev = memnew(ToolButton);
	hbc_button_search->add_child(find_prev);
	find_prev->set_focus_mode(FOCUS_NONE);
	find_prev->connect("pressed", this, "_search_prev");

	find_next = memnew(ToolButton);
	hbc_button_search->add_child(find_next);
	find_next->set_focus_mode(FOCUS_NONE);
	find_next->connect("pressed", this, "_search_next");

	case_sensitive = memnew(CheckBox);
	hbc_option_search->add_child(case_sensitive);
	case_sensitive->set_text(TTR("Match Case"));
	case_sensitive->set_focus_mode(FOCUS_NONE);
	case_sensitive->connect("toggled", this, "_search_options_changed");

	whole_words = memnew(CheckBox);
	hbc_option_search->add_child(whole_words);
	whole_words->set_text(TTR("Whole Words"));
	whole_words->set_focus_mode(FOCUS_NONE);
	whole_words->connect("toggled", this, "_search_options_changed");

	// replace toolbar
	replace_text = memnew(LineEdit);
	vbc_lineedit->add_child(replace_text);
	replace_text->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
	replace_text->connect("text_entered", this, "_replace_text_entered");

	replace = memnew(Button);
	hbc_button_replace->add_child(replace);
	replace->set_text(TTR("Replace"));
	replace->connect("pressed", this, "_replace_pressed");

	replace_all = memnew(Button);
	hbc_button_replace->add_child(replace_all);
	replace_all->set_text(TTR("Replace All"));
	replace_all->connect("pressed", this, "_replace_all_pressed");

	selection_only = memnew(CheckBox);
	hbc_option_replace->add_child(selection_only);
	selection_only->set_text(TTR("Selection Only"));
	selection_only->set_focus_mode(FOCUS_NONE);
	selection_only->connect("toggled", this, "_search_options_changed");

	hide_button = memnew(TextureButton);
	add_child(hide_button);
	hide_button->set_focus_mode(FOCUS_NONE);
	hide_button->connect("pressed", this, "_hide_pressed");
	hide_button->set_v_size_flags(SIZE_SHRINK_CENTER);
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

	Ref<InputEventMagnifyGesture> magnify_gesture = p_event;
	if (magnify_gesture.is_valid()) {

		Ref<DynamicFont> font = text_editor->get_font("font");

		if (font.is_valid()) {
			if (font->get_size() != (int)font_size) {
				font_size = font->get_size();
			}

			font_size *= powf(magnify_gesture->get_factor(), 0.25);

			_add_font_size((int)font_size - font->get_size());
		}
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
	if (font_resize_timer->get_time_left() == 0)
		font_resize_timer->start();
}

void CodeTextEditor::_reset_zoom() {
	Ref<DynamicFont> font = text_editor->get_font("font"); // reset source font size to default

	if (font.is_valid()) {
		EditorSettings::get_singleton()->set("interface/editor/code_font_size", 14);
		font->set_size(14);
	}
}

void CodeTextEditor::_line_col_changed() {

	String line = text_editor->get_line(text_editor->cursor_get_line());

	int positional_column = 0;
	for (int i = 0; i < text_editor->cursor_get_column(); i++) {
		if (line[i] == '\t') {
			positional_column += text_editor->get_indent_size(); //tab size
		} else {
			positional_column += 1;
		}
	}

	StringBuilder *sb = memnew(StringBuilder);
	sb->append("(");
	sb->append(itos(text_editor->cursor_get_line() + 1).lpad(3));
	sb->append(",");
	sb->append(itos(positional_column + 1).lpad(3));
	sb->append(")");

	line_and_col_txt->set_text(sb->as_string());
}

void CodeTextEditor::_text_changed() {

	if (text_editor->is_insert_text_operation()) {
		code_complete_timer->start();
	}

	idle->start();
}

void CodeTextEditor::_code_complete_timer_timeout() {
	if (!is_visible_in_tree())
		return;
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
	if (entries.size() == 0)
		return;
	Vector<String> strs;
	strs.resize(entries.size());
	int i = 0;
	for (List<String>::Element *E = entries.front(); E; E = E->next()) {

		strs.write[i++] = E->get();
	}

	text_editor->code_complete(strs, forced);
}

void CodeTextEditor::_font_resize_timeout() {

	if (_add_font_size(font_resize_val)) {
		font_resize_val = 0;
	}
}

bool CodeTextEditor::_add_font_size(int p_delta) {

	Ref<DynamicFont> font = text_editor->get_font("font");

	if (font.is_valid()) {
		int new_size = CLAMP(font->get_size() + p_delta, 8 * EDSCALE, 96 * EDSCALE);

		if (new_size != font->get_size()) {
			EditorSettings::get_singleton()->set("interface/editor/code_font_size", new_size / EDSCALE);
			font->set_size(new_size);
		}

		return true;
	} else {
		return false;
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
	text_editor->set_highlight_current_line(EditorSettings::get_singleton()->get("text_editor/highlighting/highlight_current_line"));
	text_editor->cursor_set_blink_enabled(EditorSettings::get_singleton()->get("text_editor/cursor/caret_blink"));
	text_editor->cursor_set_blink_speed(EditorSettings::get_singleton()->get("text_editor/cursor/caret_blink_speed"));
	text_editor->set_breakpoint_gutter_enabled(EditorSettings::get_singleton()->get("text_editor/line_numbers/show_breakpoint_gutter"));
	text_editor->set_hiding_enabled(EditorSettings::get_singleton()->get("text_editor/line_numbers/code_folding"));
	text_editor->set_draw_fold_gutter(EditorSettings::get_singleton()->get("text_editor/line_numbers/code_folding"));
	text_editor->set_wrap_enabled(EditorSettings::get_singleton()->get("text_editor/line_numbers/word_wrap"));
	text_editor->cursor_set_block_mode(EditorSettings::get_singleton()->get("text_editor/cursor/block_caret"));
	text_editor->set_smooth_scroll_enabled(EditorSettings::get_singleton()->get("text_editor/open_scripts/smooth_scrolling"));
	text_editor->set_v_scroll_speed(EditorSettings::get_singleton()->get("text_editor/open_scripts/v_scroll_speed"));
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

void CodeTextEditor::convert_indent_to_spaces() {
	int indent_size = EditorSettings::get_singleton()->get("text_editor/indent/size");
	String indent = "";

	for (int i = 0; i < indent_size; i++) {
		indent += " ";
	}

	int cursor_line = text_editor->cursor_get_line();
	int cursor_column = text_editor->cursor_get_column();

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
				line = line.left(j) + indent + line.right(j + 1);
			}
			j++;
		}
		if (changed_indentation) {
			text_editor->set_line(i, line);
		}
	}
	if (changed_indentation) {
		text_editor->cursor_set_column(cursor_column);
		text_editor->end_complex_operation();
		text_editor->update();
	}
}

void CodeTextEditor::convert_indent_to_tabs() {
	int indent_size = EditorSettings::get_singleton()->get("text_editor/indent/size");
	indent_size -= 1;

	int cursor_line = text_editor->cursor_get_line();
	int cursor_column = text_editor->cursor_get_column();

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
					line = line.left(j - indent_size) + "\t" + line.right(j + 1);
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
		text_editor->cursor_set_column(cursor_column);
		text_editor->end_complex_operation();
		text_editor->update();
	}
}

void CodeTextEditor::convert_case(CaseStyle p_case) {
	if (!text_editor->is_selection_active()) {
		return;
	}

	text_editor->begin_complex_operation();

	int begin = text_editor->get_selection_from_line();
	int end = text_editor->get_selection_to_line();
	int begin_col = text_editor->get_selection_from_column();
	int end_col = text_editor->get_selection_to_column();

	for (int i = begin; i <= end; i++) {
		int len = text_editor->get_line(i).length();
		if (i == end)
			len -= len - end_col;
		if (i == begin)
			len -= begin_col;
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
			new_line = new_line + text_editor->get_line(i).right(end_col);
		}
		text_editor->set_line(i, new_line);
	}
	text_editor->end_complex_operation();
}

void CodeTextEditor::move_lines_up() {
	text_editor->begin_complex_operation();
	if (text_editor->is_selection_active()) {
		int from_line = text_editor->get_selection_from_line();
		int from_col = text_editor->get_selection_from_column();
		int to_line = text_editor->get_selection_to_line();
		int to_column = text_editor->get_selection_to_column();

		for (int i = from_line; i <= to_line; i++) {
			int line_id = i;
			int next_id = i - 1;

			if (line_id == 0 || next_id < 0)
				return;

			text_editor->unfold_line(line_id);
			text_editor->unfold_line(next_id);

			text_editor->swap_lines(line_id, next_id);
			text_editor->cursor_set_line(next_id);
		}
		int from_line_up = from_line > 0 ? from_line - 1 : from_line;
		int to_line_up = to_line > 0 ? to_line - 1 : to_line;
		text_editor->select(from_line_up, from_col, to_line_up, to_column);
	} else {
		int line_id = text_editor->cursor_get_line();
		int next_id = line_id - 1;

		if (line_id == 0 || next_id < 0)
			return;

		text_editor->unfold_line(line_id);
		text_editor->unfold_line(next_id);

		text_editor->swap_lines(line_id, next_id);
		text_editor->cursor_set_line(next_id);
	}
	text_editor->end_complex_operation();
	text_editor->update();
}

void CodeTextEditor::move_lines_down() {
	text_editor->begin_complex_operation();
	if (text_editor->is_selection_active()) {
		int from_line = text_editor->get_selection_from_line();
		int from_col = text_editor->get_selection_from_column();
		int to_line = text_editor->get_selection_to_line();
		int to_column = text_editor->get_selection_to_column();

		for (int i = to_line; i >= from_line; i--) {
			int line_id = i;
			int next_id = i + 1;

			if (line_id == text_editor->get_line_count() - 1 || next_id > text_editor->get_line_count())
				return;

			text_editor->unfold_line(line_id);
			text_editor->unfold_line(next_id);

			text_editor->swap_lines(line_id, next_id);
			text_editor->cursor_set_line(next_id);
		}
		int from_line_down = from_line < text_editor->get_line_count() ? from_line + 1 : from_line;
		int to_line_down = to_line < text_editor->get_line_count() ? to_line + 1 : to_line;
		text_editor->select(from_line_down, from_col, to_line_down, to_column);
	} else {
		int line_id = text_editor->cursor_get_line();
		int next_id = line_id + 1;

		if (line_id == text_editor->get_line_count() - 1 || next_id > text_editor->get_line_count())
			return;

		text_editor->unfold_line(line_id);
		text_editor->unfold_line(next_id);

		text_editor->swap_lines(line_id, next_id);
		text_editor->cursor_set_line(next_id);
	}
	text_editor->end_complex_operation();
	text_editor->update();
}

void CodeTextEditor::delete_lines() {
	text_editor->begin_complex_operation();
	if (text_editor->is_selection_active()) {
		int to_line = text_editor->get_selection_to_line();
		int from_line = text_editor->get_selection_from_line();
		int count = Math::abs(to_line - from_line) + 1;

		text_editor->cursor_set_line(to_line, false);
		while (count) {
			text_editor->set_line(text_editor->cursor_get_line(), "");
			text_editor->backspace_at_cursor();
			count--;
			if (count)
				text_editor->unfold_line(from_line);
		}
		text_editor->cursor_set_line(from_line - 1);
		text_editor->deselect();
	} else {
		int line = text_editor->cursor_get_line();
		text_editor->set_line(text_editor->cursor_get_line(), "");
		text_editor->backspace_at_cursor();
		text_editor->unfold_line(line);
		text_editor->cursor_set_line(line);
	}
	text_editor->end_complex_operation();
}

void CodeTextEditor::clone_lines_down() {
	const int cursor_column = text_editor->cursor_get_column();
	int from_line = text_editor->cursor_get_line();
	int to_line = text_editor->cursor_get_line();
	int from_column = 0;
	int to_column = 0;
	int cursor_new_line = to_line + 1;
	int cursor_new_column = text_editor->cursor_get_column();
	String new_text = "\n" + text_editor->get_line(from_line);
	bool selection_active = false;

	text_editor->cursor_set_column(text_editor->get_line(from_line).length());
	if (text_editor->is_selection_active()) {
		from_column = text_editor->get_selection_from_column();
		to_column = text_editor->get_selection_to_column();

		from_line = text_editor->get_selection_from_line();
		to_line = text_editor->get_selection_to_line();
		cursor_new_line = to_line + text_editor->cursor_get_line() - from_line;
		cursor_new_column = to_column == cursor_column ? 2 * to_column - from_column : to_column;
		new_text = text_editor->get_selection_text();
		selection_active = true;

		text_editor->cursor_set_line(to_line);
		text_editor->cursor_set_column(to_column);
	}

	text_editor->begin_complex_operation();

	for (int i = from_line; i <= to_line; i++) {
		text_editor->unfold_line(i);
	}
	text_editor->deselect();
	text_editor->insert_text_at_cursor(new_text);
	text_editor->cursor_set_line(cursor_new_line);
	text_editor->cursor_set_column(cursor_new_column);
	if (selection_active) {
		text_editor->select(to_line, to_column, 2 * to_line - from_line, 2 * to_column - from_column);
	}

	text_editor->end_complex_operation();
	text_editor->update();
}

void CodeTextEditor::toggle_inline_comment(const String &delimiter) {
	text_editor->begin_complex_operation();
	if (text_editor->is_selection_active()) {
		int begin = text_editor->get_selection_from_line();
		int end = text_editor->get_selection_to_line();

		// End of selection ends on the first column of the last line, ignore it.
		if (text_editor->get_selection_to_column() == 0)
			end -= 1;

		int col_to = text_editor->get_selection_to_column();
		int cursor_pos = text_editor->cursor_get_column();

		// Check if all lines in the selected block are commented
		bool is_commented = true;
		for (int i = begin; i <= end; i++) {
			if (!text_editor->get_line(i).begins_with(delimiter)) {
				is_commented = false;
				break;
			}
		}
		for (int i = begin; i <= end; i++) {
			String line_text = text_editor->get_line(i);

			if (line_text.strip_edges().empty()) {
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

		if (is_commented && text_editor->cursor_get_column() == text_editor->get_line(text_editor->cursor_get_line()).length() + 1)
			cursor_pos += 1;

		if (text_editor->get_selection_to_column() != 0 && col_to != text_editor->get_line(text_editor->get_selection_to_line()).length() + 1)
			col_to += offset;

		if (text_editor->cursor_get_column() != 0)
			cursor_pos += offset;

		text_editor->select(begin, col_from, text_editor->get_selection_to_line(), col_to);
		text_editor->cursor_set_column(cursor_pos);

	} else {
		int begin = text_editor->cursor_get_line();
		String line_text = text_editor->get_line(begin);
		int delimiter_length = delimiter.length();

		int col = text_editor->cursor_get_column();
		if (line_text.begins_with(delimiter)) {
			line_text = line_text.substr(delimiter_length, line_text.length());
			col -= delimiter_length;
		} else {
			line_text = delimiter + line_text;
			col += delimiter_length;
		}

		text_editor->set_line(begin, line_text);
		text_editor->cursor_set_column(col);
	}
	text_editor->end_complex_operation();
	text_editor->update();
}

void CodeTextEditor::goto_line(int p_line) {
	text_editor->deselect();
	text_editor->unfold_line(p_line);
	text_editor->call_deferred("cursor_set_line", p_line);
}

void CodeTextEditor::goto_line_selection(int p_line, int p_begin, int p_end) {
	text_editor->unfold_line(p_line);
	text_editor->call_deferred("cursor_set_line", p_line);
	text_editor->call_deferred("cursor_set_column", p_begin);
	text_editor->select(p_line, p_begin, p_line, p_end);
}

Variant CodeTextEditor::get_edit_state() {
	Dictionary state;

	state["scroll_position"] = text_editor->get_v_scroll();
	state["column"] = text_editor->cursor_get_column();
	state["row"] = text_editor->cursor_get_line();

	return state;
}

void CodeTextEditor::set_edit_state(const Variant &p_state) {
	Dictionary state = p_state;
	text_editor->cursor_set_column(state["column"]);
	text_editor->cursor_set_line(state["row"]);
	text_editor->set_v_scroll(state["scroll_position"]);
	text_editor->grab_focus();
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
		text_editor->cursor_set_line(error_line);
		text_editor->cursor_set_column(error_column);
		text_editor->center_viewport_to_cursor();
	}
}

void CodeTextEditor::_update_font() {

	text_editor->add_font_override("font", get_font("source", "EditorFonts"));

	Ref<Font> status_bar_font = get_font("status_source", "EditorFonts");
	error->add_font_override("font", status_bar_font);
	int count = status_bar->get_child_count();
	for (int i = 0; i < count; i++) {
		Control *n = Object::cast_to<Control>(status_bar->get_child(i));
		if (n)
			n->add_font_override("font", status_bar_font);
	}
}

void CodeTextEditor::_on_settings_change() {

	_update_font();

	font_size = EditorSettings::get_singleton()->get("interface/editor/code_font_size");

	// AUTO BRACE COMPLETION
	text_editor->set_auto_brace_completion(
			EDITOR_DEF("text_editor/completion/auto_brace_complete", true));

	code_complete_timer->set_wait_time(
			EDITOR_DEF("text_editor/completion/code_complete_delay", .3f));

	// call hint settings
	text_editor->set_callhint_settings(
			EDITOR_DEF("text_editor/completion/put_callhint_tooltip_below_current_line", true),
			EDITOR_DEF("text_editor/completion/callhint_tooltip_offset", Vector2()));
}

void CodeTextEditor::_text_changed_idle_timeout() {

	_validate_script();
	emit_signal("validate_script");
}

void CodeTextEditor::_warning_label_gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {
		_warning_button_pressed();
	}
}

void CodeTextEditor::_warning_button_pressed() {
	_set_show_warnings_panel(!is_warnings_panel_opened);
}

void CodeTextEditor::_set_show_warnings_panel(bool p_show) {
	is_warnings_panel_opened = p_show;
	emit_signal("show_warnings_panel", p_show);
}

void CodeTextEditor::_error_pressed(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {
		emit_signal("error_pressed");
	}
}

void CodeTextEditor::_notification(int p_what) {

	switch (p_what) {
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			_load_theme_settings();
			emit_signal("load_theme_settings");
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			_update_font();
		} break;
		case NOTIFICATION_ENTER_TREE: {
			warning_button->set_icon(get_icon("NodeWarning", "EditorIcons"));
			add_constant_override("separation", 4 * EDSCALE);
		} break;
		default:
			break;
	}
}

void CodeTextEditor::set_warning_nb(int p_warning_nb) {
	warning_count_label->set_text(itos(p_warning_nb));
	warning_count_label->set_visible(p_warning_nb > 0);
	warning_button->set_visible(p_warning_nb > 0);
	if (!p_warning_nb)
		_set_show_warnings_panel(false);
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
	ClassDB::bind_method("_error_pressed", &CodeTextEditor::_error_pressed);
	ClassDB::bind_method("_warning_button_pressed", &CodeTextEditor::_warning_button_pressed);
	ClassDB::bind_method("_warning_label_gui_input", &CodeTextEditor::_warning_label_gui_input);

	ADD_SIGNAL(MethodInfo("validate_script"));
	ADD_SIGNAL(MethodInfo("load_theme_settings"));
	ADD_SIGNAL(MethodInfo("show_warnings_panel"));
	ADD_SIGNAL(MethodInfo("error_pressed"));
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

	text_editor = memnew(TextEdit);
	add_child(text_editor);
	text_editor->set_v_size_flags(SIZE_EXPAND_FILL);

	// Added second so it opens at the bottom, so it won't shift the entire text editor when opening.
	find_replace_bar = memnew(FindReplaceBar);
	add_child(find_replace_bar);
	find_replace_bar->set_h_size_flags(SIZE_EXPAND_FILL);
	find_replace_bar->hide();

	find_replace_bar->set_text_edit(text_editor);

	text_editor->set_show_line_numbers(true);
	text_editor->set_brace_matching(true);
	text_editor->set_auto_indent(true);

	status_bar = memnew(HBoxContainer);
	add_child(status_bar);
	status_bar->set_h_size_flags(SIZE_EXPAND_FILL);
	status_bar->set_custom_minimum_size(Size2(0, 24 * EDSCALE)); // Adjust for the height of the warning icon.

	idle = memnew(Timer);
	add_child(idle);
	idle->set_one_shot(true);
	idle->set_wait_time(EDITOR_DEF("text_editor/completion/idle_parse_delay", 2));

	code_complete_timer = memnew(Timer);
	add_child(code_complete_timer);
	code_complete_timer->set_one_shot(true);
	code_complete_timer->set_wait_time(EDITOR_DEF("text_editor/completion/code_complete_delay", .3f));

	error_line = 0;
	error_column = 0;

	// Error
	ScrollContainer *scroll = memnew(ScrollContainer);
	scroll->set_h_size_flags(SIZE_EXPAND_FILL);
	scroll->set_v_size_flags(SIZE_EXPAND_FILL);
	scroll->set_enable_v_scroll(false);
	status_bar->add_child(scroll);

	error = memnew(Label);
	scroll->add_child(error);
	error->set_v_size_flags(SIZE_EXPAND | SIZE_SHRINK_CENTER);
	error->add_color_override("font_color", EditorNode::get_singleton()->get_gui_base()->get_color("error_color", "Editor"));
	error->add_font_override("font", EditorNode::get_singleton()->get_gui_base()->get_font("status_source", "EditorFonts"));
	error->set_mouse_filter(MOUSE_FILTER_STOP);
	error->connect("gui_input", this, "_error_pressed");
	find_replace_bar->connect("error", error, "set_text");

	// Warnings
	warning_button = memnew(ToolButton);
	status_bar->add_child(warning_button);
	warning_button->set_v_size_flags(SIZE_EXPAND | SIZE_SHRINK_CENTER);
	warning_button->set_default_cursor_shape(CURSOR_POINTING_HAND);
	warning_button->connect("pressed", this, "_warning_button_pressed");
	warning_button->set_tooltip(TTR("Warnings"));

	warning_count_label = memnew(Label);
	status_bar->add_child(warning_count_label);
	warning_count_label->set_v_size_flags(SIZE_EXPAND | SIZE_SHRINK_CENTER);
	warning_count_label->set_align(Label::ALIGN_RIGHT);
	warning_count_label->set_default_cursor_shape(CURSOR_POINTING_HAND);
	warning_count_label->set_mouse_filter(MOUSE_FILTER_STOP);
	warning_count_label->set_tooltip(TTR("Warnings"));
	warning_count_label->add_color_override("font_color", EditorNode::get_singleton()->get_gui_base()->get_color("warning_color", "Editor"));
	warning_count_label->add_font_override("font", EditorNode::get_singleton()->get_gui_base()->get_font("status_source", "EditorFonts"));
	warning_count_label->connect("gui_input", this, "_warning_label_gui_input");

	is_warnings_panel_opened = false;
	set_warning_nb(0);

	// Line and column
	line_and_col_txt = memnew(Label);
	status_bar->add_child(line_and_col_txt);
	line_and_col_txt->set_v_size_flags(SIZE_EXPAND | SIZE_SHRINK_CENTER);
	line_and_col_txt->add_font_override("font", EditorNode::get_singleton()->get_gui_base()->get_font("status_source", "EditorFonts"));
	line_and_col_txt->set_tooltip(TTR("Line and column numbers."));
	line_and_col_txt->set_mouse_filter(MOUSE_FILTER_STOP);

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
	font_size = EditorSettings::get_singleton()->get("interface/editor/code_font_size");
	font_resize_timer = memnew(Timer);
	add_child(font_resize_timer);
	font_resize_timer->set_one_shot(true);
	font_resize_timer->set_wait_time(0.07);
	font_resize_timer->connect("timeout", this, "_font_resize_timeout");

	EditorSettings::get_singleton()->connect("settings_changed", this, "_on_settings_change");
}
