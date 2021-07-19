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

#include "core/os/keyboard.h"
#include "core/string/string_builder.h"
#include "core/string/ustring.h"

static bool _is_whitespace(char32_t c) {
	return c == '\t' || c == ' ';
}

static bool _is_char(char32_t c) {
	return !is_symbol(c);
}

void CodeEdit::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED:
		case NOTIFICATION_ENTER_TREE: {
			set_gutter_width(main_gutter, get_row_height());
			set_gutter_width(line_number_gutter, (line_number_digits + 1) * cache.font->get_char_size('0', 0, cache.font_size).width);
			set_gutter_width(fold_gutter, get_row_height() / 1.2);

			breakpoint_color = get_theme_color(SNAME("breakpoint_color"));
			breakpoint_icon = get_theme_icon(SNAME("breakpoint"));

			bookmark_color = get_theme_color(SNAME("bookmark_color"));
			bookmark_icon = get_theme_icon(SNAME("bookmark"));

			executing_line_color = get_theme_color(SNAME("executing_line_color"));
			executing_line_icon = get_theme_icon(SNAME("executing_line"));

			line_number_color = get_theme_color(SNAME("line_number_color"));

			folding_color = get_theme_color(SNAME("code_folding_color"));
			can_fold_icon = get_theme_icon(SNAME("can_fold"));
			folded_icon = get_theme_icon(SNAME("folded"));

			code_completion_max_width = get_theme_constant(SNAME("completion_max_width")) * cache.font->get_char_size('x').x;
			code_completion_max_lines = get_theme_constant(SNAME("completion_lines"));
			code_completion_scroll_width = get_theme_constant(SNAME("completion_scroll_width"));
			code_completion_scroll_color = get_theme_color(SNAME("completion_scroll_color"));
			code_completion_background_color = get_theme_color(SNAME("completion_background_color"));
			code_completion_selected_color = get_theme_color(SNAME("completion_selected_color"));
			code_completion_existing_color = get_theme_color(SNAME("completion_existing_color"));
		} break;
		case NOTIFICATION_DRAW: {
			RID ci = get_canvas_item();
			const bool caret_visible = is_caret_visible();
			const bool rtl = is_layout_rtl();
			const int row_height = get_row_height();

			bool code_completion_below = false;
			if (caret_visible && code_completion_active && code_completion_options.size() > 0) {
				Ref<StyleBox> csb = get_theme_stylebox(SNAME("completion"));

				const int code_completion_options_count = code_completion_options.size();
				const int lines = MIN(code_completion_options_count, code_completion_max_lines);
				const int icon_hsep = get_theme_constant(SNAME("hseparation"), SNAME("ItemList"));
				const Size2 icon_area_size(row_height, row_height);

				code_completion_rect.size.width = code_completion_longest_line + icon_hsep + icon_area_size.width + 2;
				code_completion_rect.size.height = lines * row_height;

				const Point2 caret_pos = get_caret_draw_pos();
				const int total_height = csb->get_minimum_size().y + code_completion_rect.size.height;
				if (caret_pos.y + row_height + total_height > get_size().height) {
					code_completion_rect.position.y = (caret_pos.y - total_height - row_height) + cache.line_spacing;
				} else {
					code_completion_rect.position.y = caret_pos.y + (cache.line_spacing / 2.0f);
					code_completion_below = true;
				}

				const int scroll_width = code_completion_options_count > code_completion_max_lines ? code_completion_scroll_width : 0;
				const int code_completion_base_width = cache.font->get_string_size(code_completion_base).width;
				if (caret_pos.x - code_completion_base_width + code_completion_rect.size.width + scroll_width > get_size().width) {
					code_completion_rect.position.x = get_size().width - code_completion_rect.size.width - scroll_width;
				} else {
					code_completion_rect.position.x = caret_pos.x - code_completion_base_width;
				}

				draw_style_box(csb, Rect2(code_completion_rect.position - csb->get_offset(), code_completion_rect.size + csb->get_minimum_size() + Size2(scroll_width, 0)));
				if (code_completion_background_color.a > 0.01) {
					RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(code_completion_rect.position, code_completion_rect.size + Size2(scroll_width, 0)), code_completion_background_color);
				}

				code_completion_line_ofs = CLAMP(code_completion_current_selected - lines / 2, 0, code_completion_options_count - lines);
				RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(Point2(code_completion_rect.position.x, code_completion_rect.position.y + (code_completion_current_selected - code_completion_line_ofs) * row_height), Size2(code_completion_rect.size.width, row_height)), code_completion_selected_color);
				draw_rect(Rect2(code_completion_rect.position + Vector2(icon_area_size.x + icon_hsep, 0), Size2(MIN(code_completion_base_width, code_completion_rect.size.width - (icon_area_size.x + icon_hsep)), code_completion_rect.size.height)), code_completion_existing_color);

				for (int i = 0; i < lines; i++) {
					int l = code_completion_line_ofs + i;
					ERR_CONTINUE(l < 0 || l >= code_completion_options_count);

					Ref<TextLine> tl;
					tl.instantiate();
					tl->add_string(code_completion_options[l].display, cache.font, cache.font_size);

					int yofs = (row_height - tl->get_size().y) / 2;
					Point2 title_pos(code_completion_rect.position.x, code_completion_rect.position.y + i * row_height + yofs);

					/* Draw completion icon if it is valid. */
					const Ref<Texture2D> &icon = code_completion_options[l].icon;
					Rect2 icon_area(code_completion_rect.position.x, code_completion_rect.position.y + i * row_height, icon_area_size.width, icon_area_size.height);
					if (icon.is_valid()) {
						Size2 icon_size = icon_area.size * 0.7;
						icon->draw_rect(ci, Rect2(icon_area.position + (icon_area.size - icon_size) / 2, icon_size));
					}
					title_pos.x = icon_area.position.x + icon_area.size.width + icon_hsep;

					tl->set_width(code_completion_rect.size.width - (icon_area_size.x + icon_hsep));
					if (rtl) {
						if (code_completion_options[l].default_value.get_type() == Variant::COLOR) {
							draw_rect(Rect2(Point2(code_completion_rect.position.x, icon_area.position.y), icon_area_size), (Color)code_completion_options[l].default_value);
						}
						tl->set_align(HALIGN_RIGHT);
					} else {
						if (code_completion_options[l].default_value.get_type() == Variant::COLOR) {
							draw_rect(Rect2(Point2(code_completion_rect.position.x + code_completion_rect.size.width - icon_area_size.x, icon_area.position.y), icon_area_size), (Color)code_completion_options[l].default_value);
						}
						tl->set_align(HALIGN_LEFT);
					}
					tl->draw(ci, title_pos, code_completion_options[l].font_color);
				}

				/* Draw a small scroll rectangle to show a position in the options. */
				if (scroll_width) {
					float r = (float)code_completion_max_lines / code_completion_options_count;
					float o = (float)code_completion_line_ofs / code_completion_options_count;
					draw_rect(Rect2(code_completion_rect.position.x + code_completion_rect.size.width, code_completion_rect.position.y + o * code_completion_rect.size.y, scroll_width, code_completion_rect.size.y * r), code_completion_scroll_color);
				}
			}

			/* Code hint */
			if (caret_visible && code_hint != "" && (!code_completion_active || (code_completion_below != code_hint_draw_below))) {
				const Ref<Font> font = cache.font;
				const int font_height = font->get_height(cache.font_size);
				Ref<StyleBox> sb = get_theme_stylebox(SNAME("panel"), SNAME("TooltipPanel"));
				Color font_color = get_theme_color(SNAME("font_color"), SNAME("TooltipLabel"));

				Vector<String> code_hint_lines = code_hint.split("\n");
				int line_count = code_hint_lines.size();

				int max_width = 0;
				for (int i = 0; i < line_count; i++) {
					max_width = MAX(max_width, font->get_string_size(code_hint_lines[i], cache.font_size).x);
				}
				Size2 minsize = sb->get_minimum_size() + Size2(max_width, line_count * font_height + (cache.line_spacing * line_count - 1));

				int offset = font->get_string_size(code_hint_lines[0].substr(0, code_hint_lines[0].find(String::chr(0xFFFF))), cache.font_size).x;
				if (code_hint_xpos == -0xFFFF) {
					code_hint_xpos = get_caret_draw_pos().x - offset;
				}
				Point2 hint_ofs = Vector2(code_hint_xpos, get_caret_draw_pos().y);
				if (code_hint_draw_below) {
					hint_ofs.y += cache.line_spacing / 2.0f;
				} else {
					hint_ofs.y -= (minsize.y + row_height) - cache.line_spacing;
				}

				draw_style_box(sb, Rect2(hint_ofs, minsize));

				int line_spacing = 0;
				for (int i = 0; i < line_count; i++) {
					const String &line = code_hint_lines[i];

					int begin = 0;
					int end = 0;
					if (line.find(String::chr(0xFFFF)) != -1) {
						begin = font->get_string_size(line.substr(0, line.find(String::chr(0xFFFF))), cache.font_size).x;
						end = font->get_string_size(line.substr(0, line.rfind(String::chr(0xFFFF))), cache.font_size).x;
					}

					Point2 round_ofs = hint_ofs + sb->get_offset() + Vector2(0, font->get_ascent() + font_height * i + line_spacing);
					round_ofs = round_ofs.round();
					draw_string(font, round_ofs, line.replace(String::chr(0xFFFF), ""), HALIGN_LEFT, -1, cache.font_size, font_color);
					if (end > 0) {
						Vector2 b = hint_ofs + sb->get_offset() + Vector2(begin, font_height + font_height * i + line_spacing - 1);
						draw_line(b, b + Vector2(end - begin, 0), font_color);
					}
					line_spacing += cache.line_spacing;
				}
			}
		} break;
	}
}

void CodeEdit::_gui_input(const Ref<InputEvent> &p_gui_input) {
	Ref<InputEventMouseButton> mb = p_gui_input;

	if (mb.is_valid()) {
		/* Ignore mouse clicks in IME input mode. */
		if (has_ime_text()) {
			return;
		}

		if (code_completion_active && code_completion_rect.has_point(mb->get_position())) {
			if (!mb->is_pressed()) {
				return;
			}

			switch (mb->get_button_index()) {
				case MOUSE_BUTTON_WHEEL_UP: {
					if (code_completion_current_selected > 0) {
						code_completion_current_selected--;
						update();
					}
				} break;
				case MOUSE_BUTTON_WHEEL_DOWN: {
					if (code_completion_current_selected < code_completion_options.size() - 1) {
						code_completion_current_selected++;
						update();
					}
				} break;
				case MOUSE_BUTTON_LEFT: {
					code_completion_current_selected = CLAMP(code_completion_line_ofs + (mb->get_position().y - code_completion_rect.position.y) / get_row_height(), 0, code_completion_options.size() - 1);
					if (mb->is_double_click()) {
						confirm_code_completion();
					}
					update();
				} break;
				default:
					break;
			}
			return;
		}
		cancel_code_completion();
		set_code_hint("");

		if (mb->is_pressed()) {
			Vector2i mpos = mb->get_position();
			if (is_layout_rtl()) {
				mpos.x = get_size().x - mpos.x;
			}

			int line, col;
			_get_mouse_pos(Point2i(mpos.x, mpos.y), line, col);

			if (mb->get_button_index() == MOUSE_BUTTON_LEFT) {
				if (is_line_folded(line)) {
					int wrap_index = get_line_wrap_index_at_col(line, col);
					if (wrap_index == times_line_wraps(line)) {
						int eol_icon_width = cache.folded_eol_icon->get_width();
						int left_margin = get_total_gutter_width() + eol_icon_width + get_line_width(line, wrap_index) - get_h_scroll();
						if (mpos.x > left_margin && mpos.x <= left_margin + eol_icon_width + 3) {
							unfold_line(line);
							return;
						}
					}
				}
			}
		}
	}

	Ref<InputEventKey> k = p_gui_input;
	bool update_code_completion = false;
	if (!k.is_valid()) {
		TextEdit::_gui_input(p_gui_input);
		return;
	}

	/* If a modifier has been pressed, and nothing else, return. */
	if (!k->is_pressed() || k->get_keycode() == KEY_CTRL || k->get_keycode() == KEY_ALT || k->get_keycode() == KEY_SHIFT || k->get_keycode() == KEY_META) {
		return;
	}

	/* Allow unicode handling if:              */
	/* No Modifiers are pressed (except shift) */
	bool allow_unicode_handling = !(k->is_command_pressed() || k->is_ctrl_pressed() || k->is_alt_pressed() || k->is_meta_pressed());

	/* AUTO-COMPLETE */
	if (code_completion_enabled && k->is_action("ui_text_completion_query", true)) {
		request_code_completion(true);
		accept_event();
		return;
	}

	if (code_completion_active) {
		if (k->is_action("ui_up", true)) {
			if (code_completion_current_selected > 0) {
				code_completion_current_selected--;
			} else {
				code_completion_current_selected = code_completion_options.size() - 1;
			}
			update();
			accept_event();
			return;
		}
		if (k->is_action("ui_down", true)) {
			if (code_completion_current_selected < code_completion_options.size() - 1) {
				code_completion_current_selected++;
			} else {
				code_completion_current_selected = 0;
			}
			update();
			accept_event();
			return;
		}
		if (k->is_action("ui_page_up", true)) {
			code_completion_current_selected = MAX(0, code_completion_current_selected - code_completion_max_lines);
			update();
			accept_event();
			return;
		}
		if (k->is_action("ui_page_down", true)) {
			code_completion_current_selected = MIN(code_completion_options.size() - 1, code_completion_current_selected + code_completion_max_lines);
			update();
			accept_event();
			return;
		}
		if (k->is_action("ui_home", true)) {
			code_completion_current_selected = 0;
			update();
			accept_event();
			return;
		}
		if (k->is_action("ui_end", true)) {
			code_completion_current_selected = MIN(code_completion_options.size() - 1, code_completion_current_selected + code_completion_max_lines);
			update();
			accept_event();
			return;
		}
		if (k->is_action("ui_text_completion_replace", true) || k->is_action("ui_text_completion_accept", true)) {
			confirm_code_completion(k->is_action("ui_text_completion_replace", true));
			accept_event();
			return;
		}
		if (k->is_action("ui_cancel", true)) {
			cancel_code_completion();
			accept_event();
			return;
		}
		if (k->is_action("ui_text_backspace", true)) {
			backspace();
			_filter_code_completion_candidates();
			accept_event();
			return;
		}

		if (k->is_action("ui_left", true) || k->is_action("ui_right", true)) {
			update_code_completion = true;
		} else {
			update_code_completion = (allow_unicode_handling && k->get_unicode() >= 32);
		}

		if (!update_code_completion) {
			cancel_code_completion();
		}
	}

	/* MISC */
	if (k->is_action("ui_cancel", true)) {
		set_code_hint("");
		accept_event();
		return;
	}
	if (allow_unicode_handling && k->get_unicode() == ')') {
		set_code_hint("");
	}

	/* Indentation */
	if (k->is_action("ui_text_indent", true)) {
		do_indent();
		accept_event();
		return;
	}

	if (k->is_action("ui_text_dedent", true)) {
		do_unindent();
		accept_event();
		return;
	}

	// Override new line actions, for auto indent
	if (k->is_action("ui_text_newline_above", true)) {
		_new_line(false, true);
		accept_event();
		return;
	}
	if (k->is_action("ui_text_newline_blank", true)) {
		_new_line(false);
		accept_event();
		return;
	}
	if (k->is_action("ui_text_newline", true)) {
		_new_line();
		accept_event();
		return;
	}

	/* Remove shift otherwise actions will not match. */
	k = k->duplicate();
	k->set_shift_pressed(false);

	if (k->is_action("ui_text_caret_up", true) ||
			k->is_action("ui_text_caret_down", true) ||
			k->is_action("ui_text_caret_line_start", true) ||
			k->is_action("ui_text_caret_line_end", true) ||
			k->is_action("ui_text_caret_page_up", true) ||
			k->is_action("ui_text_caret_page_down", true)) {
		set_code_hint("");
	}

	TextEdit::_gui_input(p_gui_input);

	if (update_code_completion) {
		_filter_code_completion_candidates();
	}
}

Control::CursorShape CodeEdit::get_cursor_shape(const Point2 &p_pos) const {
	if ((code_completion_active && code_completion_rect.has_point(p_pos)) || (is_readonly() && (!is_selecting_enabled() || get_line_count() == 0))) {
		return CURSOR_ARROW;
	}

	int line, col;
	_get_mouse_pos(p_pos, line, col);

	if (is_line_folded(line)) {
		int wrap_index = get_line_wrap_index_at_col(line, col);
		if (wrap_index == times_line_wraps(line)) {
			int eol_icon_width = cache.folded_eol_icon->get_width();
			int left_margin = get_total_gutter_width() + eol_icon_width + get_line_width(line, wrap_index) - get_h_scroll();
			if (p_pos.x > left_margin && p_pos.x <= left_margin + eol_icon_width + 3) {
				return CURSOR_POINTING_HAND;
			}
		}
	}

	return TextEdit::get_cursor_shape(p_pos);
}

/* Indent management */
void CodeEdit::set_indent_size(const int p_size) {
	ERR_FAIL_COND_MSG(p_size <= 0, "Indend size must be greater than 0.");
	if (indent_size == p_size) {
		return;
	}

	indent_size = p_size;
	if (indent_using_spaces) {
		indent_text = String(" ").repeat(p_size);
	} else {
		indent_text = "\t";
	}
	set_tab_size(p_size);
}

int CodeEdit::get_indent_size() const {
	return indent_size;
}

void CodeEdit::set_indent_using_spaces(const bool p_use_spaces) {
	indent_using_spaces = p_use_spaces;
	if (indent_using_spaces) {
		indent_text = String(" ").repeat(indent_size);
	} else {
		indent_text = "\t";
	}
}

bool CodeEdit::is_indent_using_spaces() const {
	return indent_using_spaces;
}

void CodeEdit::set_auto_indent_enabled(bool p_enabled) {
	auto_indent = p_enabled;
}

bool CodeEdit::is_auto_indent_enabled() const {
	return auto_indent;
}

void CodeEdit::set_auto_indent_prefixes(const TypedArray<String> &p_prefixes) {
	auto_indent_prefixes.clear();
	for (int i = 0; i < p_prefixes.size(); i++) {
		const String prefix = p_prefixes[i];
		auto_indent_prefixes.insert(prefix[0]);
	}
}

TypedArray<String> CodeEdit::get_auto_indent_prefixes() const {
	TypedArray<String> prefixes;
	for (const Set<char32_t>::Element *E = auto_indent_prefixes.front(); E; E = E->next()) {
		prefixes.push_back(String::chr(E->get()));
	}
	return prefixes;
}

void CodeEdit::do_indent() {
	if (is_readonly()) {
		return;
	}

	if (is_selection_active()) {
		indent_lines();
		return;
	}

	if (!indent_using_spaces) {
		_insert_text_at_cursor("\t");
		return;
	}

	int spaces_to_add = _calculate_spaces_till_next_right_indent(cursor_get_column());
	if (spaces_to_add > 0) {
		_insert_text_at_cursor(String(" ").repeat(spaces_to_add));
	}
}

void CodeEdit::indent_lines() {
	if (is_readonly()) {
		return;
	}

	begin_complex_operation();

	/* This value informs us by how much we changed selection position by indenting right. */
	/* Default is 1 for tab indentation.                                                   */
	int selection_offset = 1;

	int start_line = cursor_get_line();
	int end_line = start_line;
	if (is_selection_active()) {
		start_line = get_selection_from_line();
		end_line = get_selection_to_line();

		/* Ignore the last line if the selection is not past the first column. */
		if (get_selection_to_column() == 0) {
			selection_offset = 0;
			end_line--;
		}
	}

	for (int i = start_line; i <= end_line; i++) {
		const String line_text = get_line(i);
		if (line_text.size() == 0 && is_selection_active()) {
			continue;
		}

		if (!indent_using_spaces) {
			set_line(i, '\t' + line_text);
			continue;
		}

		/* We don't really care where selection is - we just need to know indentation level at the beginning of the line. */
		/* Since we will add this many spaces, we want to move the whole selection and caret by this much.                */
		int spaces_to_add = _calculate_spaces_till_next_right_indent(get_first_non_whitespace_column(i));
		set_line(i, String(" ").repeat(spaces_to_add) + line_text);
		selection_offset = spaces_to_add;
	}

	/* Fix selection and caret being off after shifting selection right.*/
	if (is_selection_active()) {
		select(start_line, get_selection_from_column() + selection_offset, get_selection_to_line(), get_selection_to_column() + selection_offset);
	}
	cursor_set_column(cursor_get_column() + selection_offset, false);

	end_complex_operation();
}

void CodeEdit::do_unindent() {
	if (is_readonly()) {
		return;
	}

	int cc = cursor_get_column();

	if (is_selection_active() || cc <= 0) {
		unindent_lines();
		return;
	}

	int cl = cursor_get_line();
	const String &line = get_line(cl);

	if (line[cc - 1] == '\t') {
		_remove_text(cl, cc - 1, cl, cc);
		cursor_set_column(MAX(0, cc - 1));
		return;
	}

	if (line[cc - 1] != ' ') {
		return;
	}

	int spaces_to_remove = _calculate_spaces_till_next_left_indent(cc);
	if (spaces_to_remove > 0) {
		for (int i = 1; i <= spaces_to_remove; i++) {
			if (line[cc - i] != ' ') {
				spaces_to_remove = i - 1;
				break;
			}
		}
		_remove_text(cl, cc - spaces_to_remove, cl, cc);
		cursor_set_column(MAX(0, cc - spaces_to_remove));
	}
}

void CodeEdit::unindent_lines() {
	if (is_readonly()) {
		return;
	}

	begin_complex_operation();

	/* Moving caret and selection after unindenting can get tricky because                                                      */
	/* changing content of line can move caret and selection on its own (if new line ends before previous position of either),  */
	/* therefore we just remember initial values and at the end of the operation offset them by number of removed characters.   */
	int removed_characters = 0;
	int initial_selection_end_column = 0;
	int initial_cursor_column = cursor_get_column();

	int start_line = cursor_get_line();
	int end_line = start_line;
	if (is_selection_active()) {
		start_line = get_selection_from_line();
		end_line = get_selection_to_line();

		/* Ignore the last line if the selection is not past the first column. */
		initial_selection_end_column = get_selection_to_column();
		if (initial_selection_end_column == 0) {
			end_line--;
		}
	}

	bool first_line_edited = false;
	bool last_line_edited = false;

	for (int i = start_line; i <= end_line; i++) {
		String line_text = get_line(i);

		if (line_text.begins_with("\t")) {
			line_text = line_text.substr(1, line_text.length());

			set_line(i, line_text);
			removed_characters = 1;

			first_line_edited = (i == start_line) ? true : first_line_edited;
			last_line_edited = (i == end_line) ? true : last_line_edited;
			continue;
		}

		if (line_text.begins_with(" ")) {
			/* When unindenting we aim to remove spaces before line that has selection no matter what is selected,         */
			/* Here we remove only enough spaces to align text to nearest full multiple of indentation_size.               */
			/* In case where selection begins at the start of indentation_size multiple we remove whole indentation level. */
			int spaces_to_remove = _calculate_spaces_till_next_left_indent(get_first_non_whitespace_column(i));
			line_text = line_text.substr(spaces_to_remove, line_text.length());

			set_line(i, line_text);
			removed_characters = spaces_to_remove;

			first_line_edited = (i == start_line) ? true : first_line_edited;
			last_line_edited = (i == end_line) ? true : last_line_edited;
		}
	}

	if (is_selection_active()) {
		/* Fix selection being off by one on the first line. */
		if (first_line_edited) {
			select(get_selection_from_line(), get_selection_from_column() - removed_characters, get_selection_to_line(), initial_selection_end_column);
		}

		/* Fix selection being off by one on the last line. */
		if (last_line_edited) {
			select(get_selection_from_line(), get_selection_from_column(), get_selection_to_line(), initial_selection_end_column - removed_characters);
		}
	}
	cursor_set_column(initial_cursor_column - removed_characters, false);

	end_complex_operation();
}

int CodeEdit::_calculate_spaces_till_next_left_indent(int p_column) const {
	int spaces_till_indent = p_column % indent_size;
	if (spaces_till_indent == 0) {
		spaces_till_indent = indent_size;
	}
	return spaces_till_indent;
}

int CodeEdit::_calculate_spaces_till_next_right_indent(int p_column) const {
	return indent_size - p_column % indent_size;
}

/* TODO: remove once brace completion is refactored. */
static char32_t _get_right_pair_symbol(char32_t c) {
	if (c == '"') {
		return '"';
	}
	if (c == '\'') {
		return '\'';
	}
	if (c == '(') {
		return ')';
	}
	if (c == '[') {
		return ']';
	}
	if (c == '{') {
		return '}';
	}
	return 0;
}

static bool _is_pair_left_symbol(char32_t c) {
	return c == '"' ||
		   c == '\'' ||
		   c == '(' ||
		   c == '[' ||
		   c == '{';
}

void CodeEdit::_new_line(bool p_split_current_line, bool p_above) {
	if (is_readonly()) {
		return;
	}

	const int cc = cursor_get_column();
	const int cl = cursor_get_line();
	const String line = get_line(cl);

	String ins = "\n";

	/* Append current indentation. */
	int space_count = 0;
	int line_col = 0;
	for (; line_col < cc; line_col++) {
		if (line[line_col] == '\t') {
			ins += indent_text;
			space_count = 0;
			continue;
		}

		if (line[line_col] == ' ') {
			space_count++;

			if (space_count == indent_size) {
				ins += indent_text;
				space_count = 0;
			}
			continue;
		}
		break;
	}

	if (is_line_folded(cl)) {
		unfold_line(cl);
	}

	/* Indent once again if the previous line needs it, ie ':'.          */
	/* Then add an addition new line for any closing pairs aka '()'.     */
	/* Skip this in comments or if we are going above.                   */
	bool brace_indent = false;
	if (auto_indent && !p_above && cc > 0 && is_in_comment(cl) == -1) {
		bool should_indent = false;
		char32_t indent_char = ' ';

		for (; line_col < cc; line_col++) {
			char32_t c = line[line_col];
			if (auto_indent_prefixes.has(c)) {
				should_indent = true;
				indent_char = c;
				continue;
			}

			/* Make sure this is the last char, trailing whitespace or comments are okay. */
			if (should_indent && (!_is_whitespace(c) && is_in_comment(cl, cc) == -1)) {
				should_indent = false;
			}
		}

		if (should_indent) {
			ins += indent_text;

			/* TODO: Change when brace completion is refactored. */
			char32_t closing_char = _get_right_pair_symbol(indent_char);
			if (closing_char != 0 && closing_char == line[cc]) {
				/* No need to move the brace below if we are not taking the text with us. */
				if (p_split_current_line) {
					brace_indent = true;
					ins += "\n" + ins.substr(1, ins.length() - 2);
				} else {
					brace_indent = false;
					ins = "\n" + ins.substr(1, ins.length() - 2);
				}
			}
		}
	}

	begin_complex_operation();

	bool first_line = false;
	if (!p_split_current_line) {
		if (p_above) {
			if (cl > 0) {
				cursor_set_line(cl - 1, false);
				cursor_set_column(get_line(cursor_get_line()).length());
			} else {
				cursor_set_column(0);
				first_line = true;
			}
		} else {
			cursor_set_column(line.length());
		}
	}

	insert_text_at_cursor(ins);

	if (first_line) {
		cursor_set_line(0);
	} else if (brace_indent) {
		cursor_set_line(cursor_get_line() - 1, false);
		cursor_set_column(get_line(cursor_get_line()).length());
	}

	end_complex_operation();
}

void CodeEdit::backspace() {
	if (is_readonly()) {
		return;
	}

	int cc = cursor_get_column();
	int cl = cursor_get_line();

	if (cc == 0 && cl == 0) {
		return;
	}

	if (is_selection_active()) {
		delete_selection();
		return;
	}

	if (cl > 0 && is_line_hidden(cl - 1)) {
		unfold_line(cursor_get_line() - 1);
	}

	int prev_line = cc ? cl : cl - 1;
	int prev_column = cc ? (cc - 1) : (get_line(cl - 1).length());

	merge_gutters(cl, prev_line);

	/* TODO: Change when brace completion is refactored. */
	if (auto_brace_completion_enabled && cc > 0 && _is_pair_left_symbol(get_line(cl)[cc - 1])) {
		_consume_backspace_for_pair_symbol(prev_line, prev_column);
		cursor_set_line(prev_line, false, true);
		cursor_set_column(prev_column);
		return;
	}

	/* For space indentation we need to do a simple unindent if there are no chars to the left, acting in the */
	/* same way as tabs.                                                                                      */
	if (indent_using_spaces && cc != 0) {
		if (get_first_non_whitespace_column(cl) > cc) {
			prev_column = cc - _calculate_spaces_till_next_left_indent(cc);
			prev_line = cl;
		}
	}

	_remove_text(prev_line, prev_column, cl, cc);

	cursor_set_line(prev_line, false, true);
	cursor_set_column(prev_column);
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
	emit_signal(SNAME("breakpoint_toggled"), p_line);
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
	tl.instantiate();
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
	if (!can_fold_line(p_line) && !is_line_folded(p_line)) {
		set_line_gutter_clickable(p_line, fold_gutter, false);
		return;
	}
	set_line_gutter_clickable(p_line, fold_gutter, true);

	int horizontal_padding = p_region.size.x / 10;
	int vertical_padding = p_region.size.y / 6;

	p_region.position += Point2(horizontal_padding, vertical_padding);
	p_region.size -= Point2(horizontal_padding, vertical_padding) * 2;

	if (can_fold_line(p_line)) {
		can_fold_icon->draw_rect(get_canvas_item(), p_region, false, folding_color);
		return;
	}
	folded_icon->draw_rect(get_canvas_item(), p_region, false, folding_color);
}

/* Line Folding */
void CodeEdit::set_line_folding_enabled(bool p_enabled) {
	line_folding_enabled = p_enabled;
	set_hiding_enabled(p_enabled);
}

bool CodeEdit::is_line_folding_enabled() const {
	return line_folding_enabled;
}

bool CodeEdit::can_fold_line(int p_line) const {
	ERR_FAIL_INDEX_V(p_line, get_line_count(), false);
	if (!line_folding_enabled) {
		return false;
	}

	if (p_line + 1 >= get_line_count() || get_line(p_line).strip_edges().size() == 0) {
		return false;
	}

	if (is_line_hidden(p_line) || is_line_folded(p_line)) {
		return false;
	}

	/* Check for full multiline line or block strings / comments. */
	int in_comment = is_in_comment(p_line);
	int in_string = (in_comment == -1) ? is_in_string(p_line) : -1;
	if (in_string != -1 || in_comment != -1) {
		if (get_delimiter_start_position(p_line, get_line(p_line).size() - 1).y != p_line) {
			return false;
		}

		int delimter_end_line = get_delimiter_end_position(p_line, get_line(p_line).size() - 1).y;
		/* No end line, therefore we have a multiline region over the rest of the file. */
		if (delimter_end_line == -1) {
			return true;
		}
		/* End line is the same therefore we have a block. */
		if (delimter_end_line == p_line) {
			/* Check we are the start of the block. */
			if (p_line - 1 >= 0) {
				if ((in_string != -1 && is_in_string(p_line - 1) != -1) || (in_comment != -1 && is_in_comment(p_line - 1) != -1)) {
					return false;
				}
			}
			/* Check it continues for at least one line. */
			return ((in_string != -1 && is_in_string(p_line + 1) != -1) || (in_comment != -1 && is_in_comment(p_line + 1) != -1));
		}
		return ((in_string != -1 && is_in_string(delimter_end_line) != -1) || (in_comment != -1 && is_in_comment(delimter_end_line) != -1));
	}

	/* Otherwise check indent levels. */
	int start_indent = get_indent_level(p_line);
	for (int i = p_line + 1; i < get_line_count(); i++) {
		if (is_in_string(i) != -1 || is_in_comment(i) != -1 || get_line(i).strip_edges().size() == 0) {
			continue;
		}
		return (get_indent_level(i) > start_indent);
	}
	return false;
}

void CodeEdit::fold_line(int p_line) {
	ERR_FAIL_INDEX(p_line, get_line_count());
	if (!is_line_folding_enabled() || !can_fold_line(p_line)) {
		return;
	}

	/* Find the last line to be hidden. */
	int end_line = get_line_count();

	int in_comment = is_in_comment(p_line);
	int in_string = (in_comment == -1) ? is_in_string(p_line) : -1;
	if (in_string != -1 || in_comment != -1) {
		end_line = get_delimiter_end_position(p_line, get_line(p_line).size() - 1).y;
		/* End line is the same therefore we have a block. */
		if (end_line == p_line) {
			for (int i = p_line + 1; i < get_line_count(); i++) {
				if ((in_string != -1 && is_in_string(i) == -1) || (in_comment != -1 && is_in_comment(i) == -1)) {
					end_line = i - 1;
					break;
				}
			}
		}
	} else {
		int start_indent = get_indent_level(p_line);
		for (int i = p_line + 1; i < get_line_count(); i++) {
			if (get_line(p_line).strip_edges().size() == 0 || is_in_string(i) != -1 || is_in_comment(i) != -1) {
				end_line = i;
				continue;
			}

			if (get_indent_level(i) <= start_indent && get_line(i).strip_edges().size() != 0) {
				end_line = i - 1;
				break;
			}
		}
	}

	for (int i = p_line + 1; i <= end_line; i++) {
		set_line_as_hidden(i, true);
	}

	/* Fix selection. */
	if (is_selection_active()) {
		if (is_line_hidden(get_selection_from_line()) && is_line_hidden(get_selection_to_line())) {
			deselect();
		} else if (is_line_hidden(get_selection_from_line())) {
			select(p_line, 9999, get_selection_to_line(), get_selection_to_column());
		} else if (is_line_hidden(get_selection_to_line())) {
			select(get_selection_from_line(), get_selection_from_column(), p_line, 9999);
		}
	}

	/* Reset caret. */
	if (is_line_hidden(cursor_get_line())) {
		cursor_set_line(p_line, false, false);
		cursor_set_column(get_line(p_line).length(), false);
	}
	update();
}

void CodeEdit::unfold_line(int p_line) {
	ERR_FAIL_INDEX(p_line, get_line_count());
	if (!is_line_folded(p_line) && !is_line_hidden(p_line)) {
		return;
	}

	int fold_start = p_line;
	for (; fold_start > 0; fold_start--) {
		if (is_line_folded(fold_start)) {
			break;
		}
	}
	fold_start = is_line_folded(fold_start) ? fold_start : p_line;

	for (int i = fold_start + 1; i < get_line_count(); i++) {
		if (!is_line_hidden(i)) {
			break;
		}
		set_line_as_hidden(i, false);
	}
	update();
}

void CodeEdit::fold_all_lines() {
	for (int i = 0; i < get_line_count(); i++) {
		fold_line(i);
	}
	update();
}

void CodeEdit::unfold_all_lines() {
	unhide_all_lines();
}

void CodeEdit::toggle_foldable_line(int p_line) {
	ERR_FAIL_INDEX(p_line, get_line_count());
	if (is_line_folded(p_line)) {
		unfold_line(p_line);
		return;
	}
	fold_line(p_line);
}

bool CodeEdit::is_line_folded(int p_line) const {
	ERR_FAIL_INDEX_V(p_line, get_line_count(), false);
	return p_line + 1 < get_line_count() && !is_line_hidden(p_line) && is_line_hidden(p_line + 1);
}

TypedArray<int> CodeEdit::get_folded_lines() const {
	TypedArray<int> folded_lines;
	for (int i = 0; i < get_line_count(); i++) {
		if (is_line_folded(i)) {
			folded_lines.push_back(i);
		}
	}
	return folded_lines;
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

/* Code hint */
void CodeEdit::set_code_hint(const String &p_hint) {
	code_hint = p_hint;
	code_hint_xpos = -0xFFFF;
	update();
}

void CodeEdit::set_code_hint_draw_below(bool p_below) {
	code_hint_draw_below = p_below;
	update();
}

/* Code Completion */
void CodeEdit::set_code_completion_enabled(bool p_enable) {
	code_completion_enabled = p_enable;
}

bool CodeEdit::is_code_completion_enabled() const {
	return code_completion_enabled;
}

void CodeEdit::set_code_completion_prefixes(const TypedArray<String> &p_prefixes) {
	code_completion_prefixes.clear();
	for (int i = 0; i < p_prefixes.size(); i++) {
		code_completion_prefixes.insert(p_prefixes[i]);
	}
}

TypedArray<String> CodeEdit::get_code_completion_prefixes() const {
	TypedArray<String> prefixes;
	for (Set<String>::Element *E = code_completion_prefixes.front(); E; E = E->next()) {
		prefixes.push_back(E->get());
	}
	return prefixes;
}

String CodeEdit::get_text_for_code_completion() const {
	StringBuilder completion_text;
	const int text_size = get_line_count();
	for (int i = 0; i < text_size; i++) {
		String line = get_line(i);

		if (i == cursor_get_line()) {
			completion_text += line.substr(0, cursor_get_column());
			/* Not unicode, represents the caret. */
			completion_text += String::chr(0xFFFF);
			completion_text += line.substr(cursor_get_column(), line.size());
		} else {
			completion_text += line;
		}

		if (i != text_size - 1) {
			completion_text += "\n";
		}
	}
	return completion_text.as_string();
}

void CodeEdit::request_code_completion(bool p_force) {
	ScriptInstance *si = get_script_instance();
	if (si && si->has_method("_request_code_completion")) {
		si->call("_request_code_completion", p_force);
		return;
	}

	/* Don't re-query if all existing options are quoted types, eg path, signal. */
	bool ignored = code_completion_active && !code_completion_options.is_empty();
	if (ignored) {
		ScriptCodeCompletionOption::Kind kind = ScriptCodeCompletionOption::KIND_PLAIN_TEXT;
		const ScriptCodeCompletionOption *previous_option = nullptr;
		for (int i = 0; i < code_completion_options.size(); i++) {
			const ScriptCodeCompletionOption &current_option = code_completion_options[i];
			if (!previous_option) {
				previous_option = &current_option;
				kind = current_option.kind;
			}
			if (previous_option->kind != current_option.kind) {
				ignored = false;
				break;
			}
		}
		ignored = ignored && (kind == ScriptCodeCompletionOption::KIND_FILE_PATH || kind == ScriptCodeCompletionOption::KIND_NODE_PATH || kind == ScriptCodeCompletionOption::KIND_SIGNAL);
	}

	if (ignored) {
		return;
	}

	if (p_force) {
		emit_signal(SNAME("request_code_completion"));
		return;
	}

	String line = get_line(cursor_get_line());
	int ofs = CLAMP(cursor_get_column(), 0, line.length());

	if (ofs > 0 && (is_in_string(cursor_get_line(), ofs) != -1 || _is_char(line[ofs - 1]) || code_completion_prefixes.has(String::chr(line[ofs - 1])))) {
		emit_signal(SNAME("request_code_completion"));
	} else if (ofs > 1 && line[ofs - 1] == ' ' && code_completion_prefixes.has(String::chr(line[ofs - 2]))) {
		emit_signal(SNAME("request_code_completion"));
	}
}

void CodeEdit::add_code_completion_option(CodeCompletionKind p_type, const String &p_display_text, const String &p_insert_text, const Color &p_text_color, const RES &p_icon, const Variant &p_value) {
	ScriptCodeCompletionOption completion_option;
	completion_option.kind = (ScriptCodeCompletionOption::Kind)p_type;
	completion_option.display = p_display_text;
	completion_option.insert_text = p_insert_text;
	completion_option.font_color = p_text_color;
	completion_option.icon = p_icon;
	completion_option.default_value = p_value;
	code_completion_option_submitted.push_back(completion_option);
}

void CodeEdit::update_code_completion_options(bool p_forced) {
	code_completion_forced = p_forced;
	code_completion_option_sources = code_completion_option_submitted;
	code_completion_option_submitted.clear();
	_filter_code_completion_candidates();
}

TypedArray<Dictionary> CodeEdit::get_code_completion_options() const {
	if (!code_completion_active) {
		return TypedArray<Dictionary>();
	}

	TypedArray<Dictionary> completion_options;
	completion_options.resize(code_completion_options.size());
	for (int i = 0; i < code_completion_options.size(); i++) {
		Dictionary option;
		option["kind"] = code_completion_options[i].kind;
		option["display_text"] = code_completion_options[i].display;
		option["insert_text"] = code_completion_options[i].insert_text;
		option["font_color"] = code_completion_options[i].font_color;
		option["icon"] = code_completion_options[i].icon;
		option["default_value"] = code_completion_options[i].default_value;
		completion_options[i] = option;
	}
	return completion_options;
}

Dictionary CodeEdit::get_code_completion_option(int p_index) const {
	if (!code_completion_active) {
		return Dictionary();
	}
	ERR_FAIL_INDEX_V(p_index, code_completion_options.size(), Dictionary());

	Dictionary option;
	option["kind"] = code_completion_options[p_index].kind;
	option["display_text"] = code_completion_options[p_index].display;
	option["insert_text"] = code_completion_options[p_index].insert_text;
	option["font_color"] = code_completion_options[p_index].font_color;
	option["icon"] = code_completion_options[p_index].icon;
	option["default_value"] = code_completion_options[p_index].default_value;
	return option;
}

int CodeEdit::get_code_completion_selected_index() const {
	return (code_completion_active) ? code_completion_current_selected : -1;
}

void CodeEdit::set_code_completion_selected_index(int p_index) {
	if (!code_completion_active) {
		return;
	}
	ERR_FAIL_INDEX(p_index, code_completion_options.size());
	code_completion_current_selected = p_index;
	update();
}

void CodeEdit::confirm_code_completion(bool p_replace) {
	if (is_readonly() || !code_completion_active) {
		return;
	}

	ScriptInstance *si = get_script_instance();
	if (si && si->has_method("_confirm_code_completion")) {
		si->call("_confirm_code_completion", p_replace);
		return;
	}
	begin_complex_operation();

	int caret_line = cursor_get_line();

	const String &insert_text = code_completion_options[code_completion_current_selected].insert_text;
	const String &display_text = code_completion_options[code_completion_current_selected].display;

	if (p_replace) {
		/* Find end of current section */
		const String line = get_line(caret_line);
		int caret_col = cursor_get_column();
		int caret_remove_line = caret_line;

		bool merge_text = true;
		int in_string = is_in_string(caret_line, caret_col);
		if (in_string != -1) {
			Point2 string_end = get_delimiter_end_position(caret_line, caret_col);
			if (string_end.x != -1) {
				merge_text = false;
				caret_remove_line = string_end.y;
				caret_col = string_end.x - 1;
			}
		}

		if (merge_text) {
			for (; caret_col < line.length(); caret_col++) {
				if (!_is_char(line[caret_col])) {
					break;
				}
			}
		}

		/* Replace. */
		_remove_text(caret_line, cursor_get_column() - code_completion_base.length(), caret_remove_line, caret_col);
		cursor_set_column(cursor_get_column() - code_completion_base.length(), false);
		insert_text_at_cursor(insert_text);
	} else {
		/* Get first non-matching char. */
		const String line = get_line(caret_line);
		int caret_col = cursor_get_column();
		int matching_chars = code_completion_base.length();
		for (; matching_chars <= insert_text.length(); matching_chars++) {
			if (caret_col >= line.length() || line[caret_col] != insert_text[matching_chars]) {
				break;
			}
			caret_col++;
		}

		/* Remove base completion text. */
		_remove_text(caret_line, cursor_get_column() - code_completion_base.length(), caret_line, cursor_get_column());
		cursor_set_column(cursor_get_column() - code_completion_base.length(), false);

		/* Merge with text. */
		insert_text_at_cursor(insert_text.substr(0, code_completion_base.length()));
		cursor_set_column(caret_col, false);
		insert_text_at_cursor(insert_text.substr(matching_chars));
	}

	/* TODO: merge with autobrace completion, when in CodeEdit. */
	/* Handle merging of symbols eg strings, brackets. */
	const String line = get_line(caret_line);
	char32_t next_char = line[cursor_get_column()];
	char32_t last_completion_char = insert_text[insert_text.length() - 1];
	char32_t last_completion_char_display = display_text[display_text.length() - 1];

	if ((last_completion_char == '"' || last_completion_char == '\'') && (last_completion_char == next_char || last_completion_char_display == next_char)) {
		_remove_text(caret_line, cursor_get_column(), caret_line, cursor_get_column() + 1);
	}

	if (last_completion_char == '(') {
		if (next_char == last_completion_char) {
			_remove_text(caret_line, cursor_get_column() - 1, caret_line, cursor_get_column());
		} else if (auto_brace_completion_enabled) {
			insert_text_at_cursor(")");
			cursor_set_column(cursor_get_column() - 1);
		}
	} else if (last_completion_char == ')' && next_char == '(') {
		_remove_text(caret_line, cursor_get_column() - 2, caret_line, cursor_get_column());
		if (line[cursor_get_column() + 1] != ')') {
			cursor_set_column(cursor_get_column() - 1);
		}
	}

	end_complex_operation();

	cancel_code_completion();
	if (last_completion_char == '(') {
		request_code_completion();
	}
}

void CodeEdit::cancel_code_completion() {
	if (!code_completion_active) {
		return;
	}
	code_completion_forced = false;
	code_completion_active = false;
	update();
}

void CodeEdit::_bind_methods() {
	/* Indent management */
	ClassDB::bind_method(D_METHOD("set_indent_size", "size"), &CodeEdit::set_indent_size);
	ClassDB::bind_method(D_METHOD("get_indent_size"), &CodeEdit::get_indent_size);

	ClassDB::bind_method(D_METHOD("set_indent_using_spaces", "use_spaces"), &CodeEdit::set_indent_using_spaces);
	ClassDB::bind_method(D_METHOD("is_indent_using_spaces"), &CodeEdit::is_indent_using_spaces);

	ClassDB::bind_method(D_METHOD("set_auto_indent_enabled", "enable"), &CodeEdit::set_auto_indent_enabled);
	ClassDB::bind_method(D_METHOD("is_auto_indent_enabled"), &CodeEdit::is_auto_indent_enabled);

	ClassDB::bind_method(D_METHOD("set_auto_indent_prefixes", "prefixes"), &CodeEdit::set_auto_indent_prefixes);
	ClassDB::bind_method(D_METHOD("get_auto_indent_prefixes"), &CodeEdit::get_auto_indent_prefixes);

	ClassDB::bind_method(D_METHOD("do_indent"), &CodeEdit::do_indent);
	ClassDB::bind_method(D_METHOD("do_unindent"), &CodeEdit::do_unindent);

	ClassDB::bind_method(D_METHOD("indent_lines"), &CodeEdit::indent_lines);
	ClassDB::bind_method(D_METHOD("unindent_lines"), &CodeEdit::unindent_lines);

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

	/* Line folding */
	ClassDB::bind_method(D_METHOD("set_line_folding_enabled", "enabled"), &CodeEdit::set_line_folding_enabled);
	ClassDB::bind_method(D_METHOD("is_line_folding_enabled"), &CodeEdit::is_line_folding_enabled);

	ClassDB::bind_method(D_METHOD("can_fold_line", "line"), &CodeEdit::can_fold_line);

	ClassDB::bind_method(D_METHOD("fold_line", "line"), &CodeEdit::fold_line);
	ClassDB::bind_method(D_METHOD("unfold_line", "line"), &CodeEdit::unfold_line);
	ClassDB::bind_method(D_METHOD("fold_all_lines"), &CodeEdit::fold_all_lines);
	ClassDB::bind_method(D_METHOD("unfold_all_lines"), &CodeEdit::unfold_all_lines);
	ClassDB::bind_method(D_METHOD("toggle_foldable_line", "line"), &CodeEdit::toggle_foldable_line);

	ClassDB::bind_method(D_METHOD("is_line_folded", "line"), &CodeEdit::is_line_folded);
	ClassDB::bind_method(D_METHOD("get_folded_lines"), &CodeEdit::get_folded_lines);

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

	ClassDB::bind_method(D_METHOD("get_delimiter_start_position", "line", "column"), &CodeEdit::get_delimiter_start_position);
	ClassDB::bind_method(D_METHOD("get_delimiter_end_position", "line", "column"), &CodeEdit::get_delimiter_end_position);

	/* Code hint */
	ClassDB::bind_method(D_METHOD("set_code_hint", "code_hint"), &CodeEdit::set_code_hint);
	ClassDB::bind_method(D_METHOD("set_code_hint_draw_below", "draw_below"), &CodeEdit::set_code_hint_draw_below);

	/* Code Completion */
	BIND_ENUM_CONSTANT(KIND_CLASS);
	BIND_ENUM_CONSTANT(KIND_FUNCTION);
	BIND_ENUM_CONSTANT(KIND_SIGNAL);
	BIND_ENUM_CONSTANT(KIND_VARIABLE);
	BIND_ENUM_CONSTANT(KIND_MEMBER);
	BIND_ENUM_CONSTANT(KIND_ENUM);
	BIND_ENUM_CONSTANT(KIND_CONSTANT);
	BIND_ENUM_CONSTANT(KIND_NODE_PATH);
	BIND_ENUM_CONSTANT(KIND_FILE_PATH);
	BIND_ENUM_CONSTANT(KIND_PLAIN_TEXT);

	ClassDB::bind_method(D_METHOD("get_text_for_code_completion"), &CodeEdit::get_text_for_code_completion);
	ClassDB::bind_method(D_METHOD("request_code_completion", "force"), &CodeEdit::request_code_completion, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("add_code_completion_option", "type", "display_text", "insert_text", "text_color", "icon", "value"), &CodeEdit::add_code_completion_option, DEFVAL(Color(1, 1, 1)), DEFVAL(RES()), DEFVAL(Variant::NIL));
	ClassDB::bind_method(D_METHOD("update_code_completion_options", "force"), &CodeEdit::update_code_completion_options);
	ClassDB::bind_method(D_METHOD("get_code_completion_options"), &CodeEdit::get_code_completion_options);
	ClassDB::bind_method(D_METHOD("get_code_completion_option", "index"), &CodeEdit::get_code_completion_option);
	ClassDB::bind_method(D_METHOD("get_code_completion_selected_index"), &CodeEdit::get_code_completion_selected_index);
	ClassDB::bind_method(D_METHOD("set_code_completion_selected_index", "index"), &CodeEdit::set_code_completion_selected_index);

	ClassDB::bind_method(D_METHOD("confirm_code_completion", "replace"), &CodeEdit::confirm_code_completion, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("cancel_code_completion"), &CodeEdit::cancel_code_completion);

	ClassDB::bind_method(D_METHOD("set_code_completion_enabled", "enable"), &CodeEdit::set_code_completion_enabled);
	ClassDB::bind_method(D_METHOD("is_code_completion_enabled"), &CodeEdit::is_code_completion_enabled);

	ClassDB::bind_method(D_METHOD("set_code_completion_prefixes", "prefixes"), &CodeEdit::set_code_completion_prefixes);
	ClassDB::bind_method(D_METHOD("get_code_comletion_prefixes"), &CodeEdit::get_code_completion_prefixes);

	// Overridable
	BIND_VMETHOD(MethodInfo("_confirm_code_completion", PropertyInfo(Variant::BOOL, "replace")));
	BIND_VMETHOD(MethodInfo("_request_code_completion", PropertyInfo(Variant::BOOL, "force")));
	BIND_VMETHOD(MethodInfo(Variant::ARRAY, "_filter_code_completion_candidates", PropertyInfo(Variant::ARRAY, "candidates")));

	/* Inspector */
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_breakpoints_gutter"), "set_draw_breakpoints_gutter", "is_drawing_breakpoints_gutter");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_bookmarks"), "set_draw_bookmarks_gutter", "is_drawing_bookmarks_gutter");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_executing_lines"), "set_draw_executing_lines_gutter", "is_drawing_executing_lines_gutter");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_line_numbers"), "set_draw_line_numbers", "is_draw_line_numbers_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "zero_pad_line_numbers"), "set_line_numbers_zero_padded", "is_line_numbers_zero_padded");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_fold_gutter"), "set_draw_fold_gutter", "is_drawing_fold_gutter");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "line_folding"), "set_line_folding_enabled", "is_line_folding_enabled");

	ADD_GROUP("Delimiters", "delimiter_");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "delimiter_strings"), "set_string_delimiters", "get_string_delimiters");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "delimiter_comments"), "set_comment_delimiters", "get_comment_delimiters");

	ADD_GROUP("Code Completion", "code_completion_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "code_completion_enabled"), "set_code_completion_enabled", "is_code_completion_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "code_completion_prefixes"), "set_code_completion_prefixes", "get_code_comletion_prefixes");

	ADD_GROUP("Indentation", "indent_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "indent_size"), "set_indent_size", "get_indent_size");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "indent_use_spaces"), "set_indent_using_spaces", "is_indent_using_spaces");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "indent_automatic"), "set_auto_indent_enabled", "is_auto_indent_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "indent_automatic_prefixes"), "set_auto_indent_prefixes", "get_auto_indent_prefixes");

	/* Signals */
	ADD_SIGNAL(MethodInfo("breakpoint_toggled", PropertyInfo(Variant::INT, "line")));
	ADD_SIGNAL(MethodInfo("request_code_completion"));
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
		if (is_line_folded(p_line)) {
			unfold_line(p_line);
		} else if (can_fold_line(p_line)) {
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
				end_line++;
				end_line = MIN(end_line, line_count);
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
			end_line++;
			end_line = MIN(end_line, line_count);
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
	if (!setting_delimiters) {
		_update_delimiter_cache();
	}
}

TypedArray<String> CodeEdit::_get_delimiters(DelimiterType p_type) const {
	TypedArray<String> r_delimiters;
	for (int i = 0; i < delimiters.size(); i++) {
		if (delimiters[i].type != p_type) {
			continue;
		}
		r_delimiters.push_back(delimiters[i].start_key + (delimiters[i].end_key.is_empty() ? "" : " " + delimiters[i].end_key));
	}
	return r_delimiters;
}

/* Code Completion */
void CodeEdit::_filter_code_completion_candidates() {
	ScriptInstance *si = get_script_instance();
	if (si && si->has_method("_filter_code_completion_candidates")) {
		code_completion_options.clear();
		code_completion_base = "";

		/* Build options argument. */
		TypedArray<Dictionary> completion_options_sources;
		completion_options_sources.resize(code_completion_option_sources.size());
		int i = 0;
		for (List<ScriptCodeCompletionOption>::Element *E = code_completion_option_sources.front(); E; E = E->next()) {
			Dictionary option;
			option["kind"] = E->get().kind;
			option["display_text"] = E->get().display;
			option["insert_text"] = E->get().insert_text;
			option["font_color"] = E->get().font_color;
			option["icon"] = E->get().icon;
			option["default_value"] = E->get().default_value;
			completion_options_sources[i] = option;
			i++;
		}

		TypedArray<Dictionary> completion_options = si->call("_filter_code_completion_candidates", completion_options_sources);

		/* No options to complete, cancel. */
		if (completion_options.size() == 0) {
			cancel_code_completion();
			return;
		}

		/* Convert back into options. */
		int max_width = 0;
		for (i = 0; i < completion_options.size(); i++) {
			ScriptCodeCompletionOption option;
			option.kind = (ScriptCodeCompletionOption::Kind)(int)completion_options[i].get("kind");
			option.display = completion_options[i].get("display_text");
			option.insert_text = completion_options[i].get("insert_text");
			option.font_color = completion_options[i].get("font_color");
			option.icon = completion_options[i].get("icon");
			option.default_value = completion_options[i].get("default_value");

			max_width = MAX(max_width, cache.font->get_string_size(option.display).width);
			code_completion_options.push_back(option);
		}

		code_completion_longest_line = MIN(max_width, code_completion_max_width);
		code_completion_current_selected = 0;
		code_completion_active = true;
		update();
		return;
	}

	const int caret_line = cursor_get_line();
	const int caret_column = cursor_get_column();
	const String line = get_line(caret_line);

	if (caret_column > 0 && line[caret_column - 1] == '(' && !code_completion_forced) {
		cancel_code_completion();
		return;
	}

	/* Get string status, are we in one or at the close. */
	int in_string = is_in_string(caret_line, caret_column);
	int first_quote_col = -1;
	if (in_string != -1) {
		Point2 string_start_pos = get_delimiter_start_position(caret_line, caret_column);
		first_quote_col = (string_start_pos.y == caret_line) ? string_start_pos.x : -1;
	} else if (caret_column > 0) {
		if (is_in_string(caret_line, caret_column - 1) != -1) {
			first_quote_col = caret_column - 1;
		}
	}

	int cofs = caret_column;
	String string_to_complete;
	bool prev_is_word = false;

	/* Cancel if we are at the close of a string. */
	if (in_string == -1 && first_quote_col == cofs - 1) {
		cancel_code_completion();
		return;
		/* In a string, therefore we are trying to complete the string text. */
	} else if (in_string != -1 && first_quote_col != -1) {
		int key_length = delimiters[in_string].start_key.length();
		string_to_complete = line.substr(first_quote_col - key_length, (cofs - first_quote_col) + key_length);
		/* If we have a space, previous word might be a keyword. eg "func |". */
	} else if (cofs > 0 && line[cofs - 1] == ' ') {
		int ofs = cofs - 1;
		while (ofs >= 0 && line[ofs] == ' ') {
			ofs--;
		}
		prev_is_word = _is_char(line[ofs]);
		/* Otherwise get current word and set cofs to the start. */
	} else {
		int start_cofs = cofs;
		while (cofs > 0 && line[cofs - 1] > 32 && (line[cofs - 1] == '/' || _is_char(line[cofs - 1]))) {
			cofs--;
		}
		string_to_complete = line.substr(cofs, start_cofs - cofs);
	}

	/* If all else fails, check for a prefix.         */
	/* Single space between caret and prefix is okay. */
	bool prev_is_prefix = false;
	if (cofs > 0 && code_completion_prefixes.has(String::chr(line[cofs - 1]))) {
		prev_is_prefix = true;
	} else if (cofs > 1 && line[cofs - 1] == ' ' && code_completion_prefixes.has(String::chr(line[cofs - 2]))) {
		prev_is_prefix = true;
	}

	if (!prev_is_word && string_to_complete.is_empty() && (cofs == 0 || !prev_is_prefix)) {
		cancel_code_completion();
		return;
	}

	/* Filter Options. */
	/* For now handle only tradional quoted strings. */
	bool single_quote = in_string != -1 && first_quote_col > 0 && delimiters[in_string].start_key == "'";

	code_completion_options.clear();
	code_completion_base = string_to_complete;

	Vector<ScriptCodeCompletionOption> completion_options_casei;
	Vector<ScriptCodeCompletionOption> completion_options_subseq;
	Vector<ScriptCodeCompletionOption> completion_options_subseq_casei;

	int max_width = 0;
	String string_to_complete_lower = string_to_complete.to_lower();
	for (List<ScriptCodeCompletionOption>::Element *E = code_completion_option_sources.front(); E; E = E->next()) {
		ScriptCodeCompletionOption &option = E->get();

		if (single_quote && option.display.is_quoted()) {
			option.display = option.display.unquote().quote("'");
		}

		if (in_string != -1) {
			String quote = single_quote ? "'" : "\"";
			option.display = option.display.unquote().quote(quote);
			option.insert_text = option.insert_text.unquote().quote(quote);
		}

		if (option.display.length() == 0) {
			continue;
		}

		if (string_to_complete.length() == 0) {
			code_completion_options.push_back(option);
			max_width = MAX(max_width, cache.font->get_string_size(option.display).width);
			continue;
		}

		/* This code works the same as:

		if (option.display.begins_with(s)) {
			completion_options.push_back(option);
		} else if (option.display.to_lower().begins_with(s.to_lower())) {
			completion_options_casei.push_back(option);
		} else if (s.is_subsequence_of(option.display)) {
			completion_options_subseq.push_back(option);
		} else if (s.is_subsequence_ofi(option.display)) {
			completion_options_subseq_casei.push_back(option);
		}

		But is more performant due to being inlined and looping over the characters only once
		*/

		String display_lower = option.display.to_lower();

		const char32_t *ssq = &string_to_complete[0];
		const char32_t *ssq_lower = &string_to_complete_lower[0];

		const char32_t *tgt = &option.display[0];
		const char32_t *tgt_lower = &display_lower[0];

		const char32_t *ssq_last_tgt = nullptr;
		const char32_t *ssq_lower_last_tgt = nullptr;

		for (; *tgt; tgt++, tgt_lower++) {
			if (*ssq == *tgt) {
				ssq++;
				ssq_last_tgt = tgt;
			}
			if (*ssq_lower == *tgt_lower) {
				ssq_lower++;
				ssq_lower_last_tgt = tgt;
			}
		}

		/* Matched the whole subsequence in s. */
		if (!*ssq) {
			/* Finished matching in the first s.length() characters. */
			if (ssq_last_tgt == &option.display[string_to_complete.length() - 1]) {
				code_completion_options.push_back(option);
			} else {
				completion_options_subseq.push_back(option);
			}
			max_width = MAX(max_width, cache.font->get_string_size(option.display).width);
			/* Matched the whole subsequence in s_lower. */
		} else if (!*ssq_lower) {
			/* Finished matching in the first s.length() characters. */
			if (ssq_lower_last_tgt == &option.display[string_to_complete.length() - 1]) {
				completion_options_casei.push_back(option);
			} else {
				completion_options_subseq_casei.push_back(option);
			}
			max_width = MAX(max_width, cache.font->get_string_size(option.display).width);
		}
	}

	code_completion_options.append_array(completion_options_casei);
	code_completion_options.append_array(completion_options_subseq);
	code_completion_options.append_array(completion_options_subseq_casei);

	/* No options to complete, cancel. */
	if (code_completion_options.size() == 0) {
		cancel_code_completion();
		return;
	}

	/* A perfect match, stop completion. */
	if (code_completion_options.size() == 1 && string_to_complete == code_completion_options[0].display) {
		cancel_code_completion();
		return;
	}

	code_completion_longest_line = MIN(max_width, code_completion_max_width);
	code_completion_current_selected = 0;
	code_completion_active = true;
	update();
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

		emit_signal(SNAME("breakpoint_toggled"), line);
		if (line_count > 0 || line >= p_from_line) {
			emit_signal(SNAME("breakpoint_toggled"), line + line_count);
			breakpointed_lines[line + line_count] = true;
			continue;
		}
	}
}

CodeEdit::CodeEdit() {
	/* Indent management */
	auto_indent_prefixes.insert(':');
	auto_indent_prefixes.insert('{');
	auto_indent_prefixes.insert('[');
	auto_indent_prefixes.insert('(');

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
	set_gutter_type(gutter_idx, GUTTER_TYPE_CUSTOM);
	set_gutter_custom_draw(gutter_idx, this, "_main_gutter_draw_callback");
	gutter_idx++;

	/* Line numbers */
	add_gutter();
	set_gutter_name(gutter_idx, "line_numbers");
	set_gutter_draw(gutter_idx, false);
	set_gutter_type(gutter_idx, GUTTER_TYPE_CUSTOM);
	set_gutter_custom_draw(gutter_idx, this, "_line_number_draw_callback");
	gutter_idx++;

	/* Fold Gutter */
	add_gutter();
	set_gutter_name(gutter_idx, "fold_gutter");
	set_gutter_draw(gutter_idx, false);
	set_gutter_type(gutter_idx, GUTTER_TYPE_CUSTOM);
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
