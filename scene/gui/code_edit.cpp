/*************************************************************************/
/*  code_edit.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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
			style_normal = get_theme_stylebox(SNAME("normal"));

			font = get_theme_font(SNAME("font"));
			font_size = get_theme_font_size(SNAME("font_size"));

			line_spacing = get_theme_constant(SNAME("line_spacing"));

			set_gutter_width(main_gutter, get_line_height());
			set_gutter_width(line_number_gutter, (line_number_digits + 1) * font->get_char_size('0', 0, font_size).width);
			set_gutter_width(fold_gutter, get_line_height() / 1.2);

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

			code_completion_max_width = get_theme_constant(SNAME("completion_max_width"));
			code_completion_max_lines = get_theme_constant(SNAME("completion_lines"));
			code_completion_scroll_width = get_theme_constant(SNAME("completion_scroll_width"));
			code_completion_scroll_color = get_theme_color(SNAME("completion_scroll_color"));
			code_completion_background_color = get_theme_color(SNAME("completion_background_color"));
			code_completion_selected_color = get_theme_color(SNAME("completion_selected_color"));
			code_completion_existing_color = get_theme_color(SNAME("completion_existing_color"));

			line_length_guideline_color = get_theme_color(SNAME("line_length_guideline_color"));
		} break;
		case NOTIFICATION_DRAW: {
			RID ci = get_canvas_item();
			const Size2 size = get_size();
			const bool caret_visible = is_caret_visible();
			const bool rtl = is_layout_rtl();
			const int row_height = get_line_height();

			if (line_length_guideline_columns.size() > 0) {
				const int xmargin_beg = style_normal->get_margin(SIDE_LEFT) + get_total_gutter_width();
				const int xmargin_end = size.width - style_normal->get_margin(SIDE_RIGHT) - (is_drawing_minimap() ? get_minimap_width() : 0);
				const int char_size = Math::round(font->get_char_size('0', 0, font_size).width);

				for (int i = 0; i < line_length_guideline_columns.size(); i++) {
					const int xoffset = xmargin_beg + char_size * (int)line_length_guideline_columns[i] - get_h_scroll();
					if (xoffset > xmargin_beg && xoffset < xmargin_end) {
						Color guideline_color = (i == 0) ? line_length_guideline_color : line_length_guideline_color * Color(1, 1, 1, 0.5);
						if (rtl) {
							RenderingServer::get_singleton()->canvas_item_add_line(ci, Point2(size.width - xoffset, 0), Point2(size.width - xoffset, size.height), guideline_color);
							continue;
						}
						RenderingServer::get_singleton()->canvas_item_add_line(ci, Point2(xoffset, 0), Point2(xoffset, size.height), guideline_color);
					}
				}
			}

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
					code_completion_rect.position.y = (caret_pos.y - total_height - row_height) + line_spacing;
				} else {
					code_completion_rect.position.y = caret_pos.y + (line_spacing / 2.0f);
					code_completion_below = true;
				}

				const int scroll_width = code_completion_options_count > code_completion_max_lines ? code_completion_scroll_width : 0;
				const int code_completion_base_width = font->get_string_size(code_completion_base, font_size).width;
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

				for (int i = 0; i < lines; i++) {
					int l = code_completion_line_ofs + i;
					ERR_CONTINUE(l < 0 || l >= code_completion_options_count);

					Ref<TextLine> tl;
					tl.instantiate();
					tl->add_string(code_completion_options[l].display, font, font_size);

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
						tl->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_RIGHT);
					} else {
						if (code_completion_options[l].default_value.get_type() == Variant::COLOR) {
							draw_rect(Rect2(Point2(code_completion_rect.position.x + code_completion_rect.size.width - icon_area_size.x, icon_area.position.y), icon_area_size), (Color)code_completion_options[l].default_value);
						}
						tl->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_LEFT);
					}

					Point2 match_pos = Point2(code_completion_rect.position.x + icon_area_size.x + icon_hsep, code_completion_rect.position.y + i * row_height);

					for (int j = 0; j < code_completion_options[l].matches.size(); j++) {
						Pair<int, int> match = code_completion_options[l].matches[j];
						int match_offset = font->get_string_size(code_completion_options[l].display.substr(0, match.first), font_size).width;
						int match_len = font->get_string_size(code_completion_options[l].display.substr(match.first, match.second), font_size).width;

						draw_rect(Rect2(match_pos + Point2(match_offset, 0), Size2(match_len, row_height)), code_completion_existing_color);
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
			if (caret_visible && !code_hint.is_empty() && (!code_completion_active || (code_completion_below != code_hint_draw_below))) {
				const int font_height = font->get_height(font_size);
				Ref<StyleBox> sb = get_theme_stylebox(SNAME("panel"), SNAME("TooltipPanel"));
				Color font_color = get_theme_color(SNAME("font_color"), SNAME("TooltipLabel"));

				Vector<String> code_hint_lines = code_hint.split("\n");
				int line_count = code_hint_lines.size();

				int max_width = 0;
				for (int i = 0; i < line_count; i++) {
					max_width = MAX(max_width, font->get_string_size(code_hint_lines[i], font_size).x);
				}
				Size2 minsize = sb->get_minimum_size() + Size2(max_width, line_count * font_height + (line_spacing * line_count - 1));

				int offset = font->get_string_size(code_hint_lines[0].substr(0, code_hint_lines[0].find(String::chr(0xFFFF))), font_size).x;
				if (code_hint_xpos == -0xFFFF) {
					code_hint_xpos = get_caret_draw_pos().x - offset;
				}
				Point2 hint_ofs = Vector2(code_hint_xpos, get_caret_draw_pos().y);
				if (code_hint_draw_below) {
					hint_ofs.y += line_spacing / 2.0f;
				} else {
					hint_ofs.y -= (minsize.y + row_height) - line_spacing;
				}

				draw_style_box(sb, Rect2(hint_ofs, minsize));

				int yofs = 0;
				for (int i = 0; i < line_count; i++) {
					const String &line = code_hint_lines[i];

					int begin = 0;
					int end = 0;
					if (line.find(String::chr(0xFFFF)) != -1) {
						begin = font->get_string_size(line.substr(0, line.find(String::chr(0xFFFF))), font_size).x;
						end = font->get_string_size(line.substr(0, line.rfind(String::chr(0xFFFF))), font_size).x;
					}

					Point2 round_ofs = hint_ofs + sb->get_offset() + Vector2(0, font->get_ascent(font_size) + font_height * i + yofs);
					round_ofs = round_ofs.round();
					draw_string(font, round_ofs, line.replace(String::chr(0xFFFF), ""), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, font_color);
					if (end > 0) {
						// Draw an underline for the currently edited function parameter.
						const Vector2 b = hint_ofs + sb->get_offset() + Vector2(begin, font_height + font_height * i + yofs);
						draw_line(b, b + Vector2(end - begin, 0), font_color, 2);

						// Draw a translucent text highlight as well.
						const Rect2 highlight_rect = Rect2(
								b - Vector2(0, font_height),
								Vector2(end - begin, font_height));
						draw_rect(highlight_rect, font_color * Color(1, 1, 1, 0.2));
					}
					yofs += line_spacing;
				}
			}
		} break;
	}
}

void CodeEdit::gui_input(const Ref<InputEvent> &p_gui_input) {
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
				case MouseButton::WHEEL_UP: {
					if (code_completion_current_selected > 0) {
						code_completion_current_selected--;
						update();
					}
				} break;
				case MouseButton::WHEEL_DOWN: {
					if (code_completion_current_selected < code_completion_options.size() - 1) {
						code_completion_current_selected++;
						update();
					}
				} break;
				case MouseButton::LEFT: {
					code_completion_current_selected = CLAMP(code_completion_line_ofs + (mb->get_position().y - code_completion_rect.position.y) / get_line_height(), 0, code_completion_options.size() - 1);
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

			Point2i pos = get_line_column_at_pos(mpos, false);
			int line = pos.y;
			int col = pos.x;

			if (line != -1 && mb->get_button_index() == MouseButton::LEFT) {
				if (is_line_folded(line)) {
					int wrap_index = get_line_wrap_index_at_column(line, col);
					if (wrap_index == get_line_wrap_count(line)) {
						int eol_icon_width = folded_eol_icon->get_width();
						int left_margin = get_total_gutter_width() + eol_icon_width + get_line_width(line, wrap_index) - get_h_scroll();
						if (mpos.x > left_margin && mpos.x <= left_margin + eol_icon_width + 3) {
							unfold_line(line);
							return;
						}
					}
				}
			}
		} else {
			if (mb->get_button_index() == MouseButton::LEFT) {
				if (mb->is_command_pressed() && !symbol_lookup_word.is_empty()) {
					Vector2i mpos = mb->get_position();
					if (is_layout_rtl()) {
						mpos.x = get_size().x - mpos.x;
					}

					Point2i pos = get_line_column_at_pos(mpos, false);
					int line = pos.y;
					int col = pos.x;

					if (line != -1) {
						emit_signal(SNAME("symbol_lookup"), symbol_lookup_word, line, col);
					}
					return;
				}
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_gui_input;
	if (mm.is_valid()) {
		Vector2i mpos = mm->get_position();
		if (is_layout_rtl()) {
			mpos.x = get_size().x - mpos.x;
		}

		if (symbol_lookup_on_click_enabled) {
			if (mm->is_command_pressed() && mm->get_button_mask() == MouseButton::NONE && !is_dragging_cursor()) {
				symbol_lookup_new_word = get_word_at_pos(mpos);
				if (symbol_lookup_new_word != symbol_lookup_word) {
					emit_signal(SNAME("symbol_validate"), symbol_lookup_new_word);
				}
			} else {
				set_symbol_lookup_word_as_valid(false);
			}
		}
	}

	Ref<InputEventKey> k = p_gui_input;
	bool update_code_completion = false;
	if (!k.is_valid()) {
		TextEdit::gui_input(p_gui_input);
		return;
	}

	/* Ctrl + Hover symbols */
#ifdef OSX_ENABLED
	if (k->get_keycode() == Key::META) {
#else
	if (k->get_keycode() == Key::CTRL) {
#endif
		if (symbol_lookup_on_click_enabled) {
			if (k->is_pressed() && !is_dragging_cursor()) {
				symbol_lookup_new_word = get_word_at_pos(get_local_mouse_pos());
				if (symbol_lookup_new_word != symbol_lookup_word) {
					emit_signal(SNAME("symbol_validate"), symbol_lookup_new_word);
				}
			} else {
				set_symbol_lookup_word_as_valid(false);
			}
		}
		return;
	}

	/* If a modifier has been pressed, and nothing else, return. */
	if (!k->is_pressed() || k->get_keycode() == Key::CTRL || k->get_keycode() == Key::ALT || k->get_keycode() == Key::SHIFT || k->get_keycode() == Key::META) {
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
			code_completion_current_selected = code_completion_options.size() - 1;
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
			_filter_code_completion_candidates_impl();
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
	if (!code_hint.is_empty() && k->is_action("ui_cancel", true)) {
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

	TextEdit::gui_input(p_gui_input);

	if (update_code_completion) {
		_filter_code_completion_candidates_impl();
	}
}

/* General overrides */
Control::CursorShape CodeEdit::get_cursor_shape(const Point2 &p_pos) const {
	if (!symbol_lookup_word.is_empty()) {
		return CURSOR_POINTING_HAND;
	}

	if ((code_completion_active && code_completion_rect.has_point(p_pos)) || (!is_editable() && (!is_selecting_enabled() || get_line_count() == 0))) {
		return CURSOR_ARROW;
	}

	Point2i pos = get_line_column_at_pos(p_pos, false);
	int line = pos.y;
	int col = pos.x;

	if (line != -1 && is_line_folded(line)) {
		int wrap_index = get_line_wrap_index_at_column(line, col);
		if (wrap_index == get_line_wrap_count(line)) {
			int eol_icon_width = folded_eol_icon->get_width();
			int left_margin = get_total_gutter_width() + eol_icon_width + get_line_width(line, wrap_index) - get_h_scroll();
			if (p_pos.x > left_margin && p_pos.x <= left_margin + eol_icon_width + 3) {
				return CURSOR_POINTING_HAND;
			}
		}
	}

	return TextEdit::get_cursor_shape(p_pos);
}

/* Text manipulation */

// Overridable actions
void CodeEdit::_handle_unicode_input_internal(const uint32_t p_unicode) {
	bool had_selection = has_selection();
	if (had_selection) {
		begin_complex_operation();
		delete_selection();
	}

	// Remove the old character if in overtype mode and no selection.
	if (is_overtype_mode_enabled() && !had_selection) {
		begin_complex_operation();

		/* Make sure we don't try and remove empty space. */
		if (get_caret_column() < get_line(get_caret_line()).length()) {
			remove_text(get_caret_line(), get_caret_column(), get_caret_line(), get_caret_column() + 1);
		}
	}

	const char32_t chr[2] = { (char32_t)p_unicode, 0 };

	if (auto_brace_completion_enabled) {
		int cl = get_caret_line();
		int cc = get_caret_column();
		int caret_move_offset = 1;

		int post_brace_pair = cc < get_line(cl).length() ? _get_auto_brace_pair_close_at_pos(cl, cc) : -1;

		if (has_string_delimiter(chr) && cc > 0 && _is_char(get_line(cl)[cc - 1]) && post_brace_pair == -1) {
			insert_text_at_caret(chr);
		} else if (cc < get_line(cl).length() && _is_char(get_line(cl)[cc])) {
			insert_text_at_caret(chr);
		} else if (post_brace_pair != -1 && auto_brace_completion_pairs[post_brace_pair].close_key[0] == chr[0]) {
			caret_move_offset = auto_brace_completion_pairs[post_brace_pair].close_key.length();
		} else if (is_in_comment(cl, cc) != -1 || (is_in_string(cl, cc) != -1 && has_string_delimiter(chr))) {
			insert_text_at_caret(chr);
		} else {
			insert_text_at_caret(chr);

			int pre_brace_pair = _get_auto_brace_pair_open_at_pos(cl, cc + 1);
			if (pre_brace_pair != -1) {
				insert_text_at_caret(auto_brace_completion_pairs[pre_brace_pair].close_key);
			}
		}
		set_caret_column(cc + caret_move_offset);
	} else {
		insert_text_at_caret(chr);
	}

	if ((is_overtype_mode_enabled() && !had_selection) || (had_selection)) {
		end_complex_operation();
	}
}

void CodeEdit::_backspace_internal() {
	if (!is_editable()) {
		return;
	}

	if (has_selection()) {
		delete_selection();
		return;
	}

	int cc = get_caret_column();
	int cl = get_caret_line();

	if (cc == 0 && cl == 0) {
		return;
	}

	if (cl > 0 && _is_line_hidden(cl - 1)) {
		unfold_line(get_caret_line() - 1);
	}

	int prev_line = cc ? cl : cl - 1;
	int prev_column = cc ? (cc - 1) : (get_line(cl - 1).length());

	merge_gutters(prev_line, cl);

	if (auto_brace_completion_enabled && cc > 0) {
		int idx = _get_auto_brace_pair_open_at_pos(cl, cc);
		if (idx != -1) {
			prev_column = cc - auto_brace_completion_pairs[idx].open_key.length();

			if (_get_auto_brace_pair_close_at_pos(cl, cc) == idx) {
				remove_text(prev_line, prev_column, cl, cc + auto_brace_completion_pairs[idx].close_key.length());
			} else {
				remove_text(prev_line, prev_column, cl, cc);
			}
			set_caret_line(prev_line, false, true);
			set_caret_column(prev_column);
			return;
		}
	}

	// For space indentation we need to do a simple unindent if there are no chars to the left, acting in the
	// same way as tabs.
	if (indent_using_spaces && cc != 0) {
		if (get_first_non_whitespace_column(cl) >= cc) {
			prev_column = cc - _calculate_spaces_till_next_left_indent(cc);
			prev_line = cl;
		}
	}

	remove_text(prev_line, prev_column, cl, cc);

	set_caret_line(prev_line, false, true);
	set_caret_column(prev_column);
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
	if (!is_editable()) {
		return;
	}

	if (has_selection()) {
		indent_lines();
		return;
	}

	if (!indent_using_spaces) {
		insert_text_at_caret("\t");
		return;
	}

	int spaces_to_add = _calculate_spaces_till_next_right_indent(get_caret_column());
	if (spaces_to_add > 0) {
		insert_text_at_caret(String(" ").repeat(spaces_to_add));
	}
}

void CodeEdit::indent_lines() {
	if (!is_editable()) {
		return;
	}

	begin_complex_operation();

	/* This value informs us by how much we changed selection position by indenting right. */
	/* Default is 1 for tab indentation.                                                   */
	int selection_offset = 1;

	int start_line = get_caret_line();
	int end_line = start_line;
	if (has_selection()) {
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
		if (line_text.size() == 0 && has_selection()) {
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
	if (has_selection()) {
		select(start_line, get_selection_from_column() + selection_offset, get_selection_to_line(), get_selection_to_column() + selection_offset);
	}
	set_caret_column(get_caret_column() + selection_offset, false);

	end_complex_operation();
}

void CodeEdit::do_unindent() {
	if (!is_editable()) {
		return;
	}

	int cc = get_caret_column();

	if (has_selection() || cc <= 0) {
		unindent_lines();
		return;
	}

	int cl = get_caret_line();
	const String &line = get_line(cl);

	if (line[cc - 1] == '\t') {
		remove_text(cl, cc - 1, cl, cc);
		set_caret_column(MAX(0, cc - 1));
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
		remove_text(cl, cc - spaces_to_remove, cl, cc);
		set_caret_column(MAX(0, cc - spaces_to_remove));
	}
}

void CodeEdit::unindent_lines() {
	if (!is_editable()) {
		return;
	}

	begin_complex_operation();

	/* Moving caret and selection after unindenting can get tricky because                                                      */
	/* changing content of line can move caret and selection on its own (if new line ends before previous position of either),  */
	/* therefore we just remember initial values and at the end of the operation offset them by number of removed characters.   */
	int removed_characters = 0;
	int initial_selection_end_column = 0;
	int initial_cursor_column = get_caret_column();

	int start_line = get_caret_line();
	int end_line = start_line;
	if (has_selection()) {
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

	if (has_selection()) {
		/* Fix selection being off by one on the first line. */
		if (first_line_edited) {
			select(get_selection_from_line(), get_selection_from_column() - removed_characters, get_selection_to_line(), initial_selection_end_column);
		}

		/* Fix selection being off by one on the last line. */
		if (last_line_edited) {
			select(get_selection_from_line(), get_selection_from_column(), get_selection_to_line(), initial_selection_end_column - removed_characters);
		}
	}
	set_caret_column(initial_cursor_column - removed_characters, false);

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

void CodeEdit::_new_line(bool p_split_current_line, bool p_above) {
	if (!is_editable()) {
		return;
	}

	/* When not splitting the line, we need to factor in indentation from the end of the current line. */
	const int cc = p_split_current_line ? get_caret_column() : get_line(get_caret_line()).length();
	const int cl = get_caret_line();

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

			String closing_pair = get_auto_brace_completion_close_key(String::chr(indent_char));
			if (!closing_pair.is_empty() && line.find(closing_pair, cc) == cc) {
				/* No need to move the brace below if we are not taking the text with us. */
				if (p_split_current_line) {
					brace_indent = true;
					ins += "\n" + ins.substr(indent_text.size(), ins.length() - 2);
				} else {
					brace_indent = false;
					ins = "\n" + ins.substr(indent_text.size(), ins.length() - 2);
				}
			}
		}
	}

	begin_complex_operation();

	bool first_line = false;
	if (!p_split_current_line) {
		deselect();

		if (p_above) {
			if (cl > 0) {
				set_caret_line(cl - 1, false);
				set_caret_column(get_line(get_caret_line()).length());
			} else {
				set_caret_column(0);
				first_line = true;
			}
		} else {
			set_caret_column(line.length());
		}
	}

	insert_text_at_caret(ins);

	if (first_line) {
		set_caret_line(0);
	} else if (brace_indent) {
		set_caret_line(get_caret_line() - 1, false);
		set_caret_column(get_line(get_caret_line()).length());
	}

	end_complex_operation();
}

/* Auto brace completion */
void CodeEdit::set_auto_brace_completion_enabled(bool p_enabled) {
	auto_brace_completion_enabled = p_enabled;
}

bool CodeEdit::is_auto_brace_completion_enabled() const {
	return auto_brace_completion_enabled;
}

void CodeEdit::set_highlight_matching_braces_enabled(bool p_enabled) {
	highlight_matching_braces_enabled = p_enabled;
	update();
}

bool CodeEdit::is_highlight_matching_braces_enabled() const {
	return highlight_matching_braces_enabled;
}

void CodeEdit::add_auto_brace_completion_pair(const String &p_open_key, const String &p_close_key) {
	ERR_FAIL_COND_MSG(p_open_key.is_empty(), "auto brace completion open key cannot be empty");
	ERR_FAIL_COND_MSG(p_close_key.is_empty(), "auto brace completion close key cannot be empty");

	for (int i = 0; i < p_open_key.length(); i++) {
		ERR_FAIL_COND_MSG(!is_symbol(p_open_key[i]), "auto brace completion open key must be a symbol");
	}
	for (int i = 0; i < p_close_key.length(); i++) {
		ERR_FAIL_COND_MSG(!is_symbol(p_close_key[i]), "auto brace completion close key must be a symbol");
	}

	int at = 0;
	for (int i = 0; i < auto_brace_completion_pairs.size(); i++) {
		ERR_FAIL_COND_MSG(auto_brace_completion_pairs[i].open_key == p_open_key, "auto brace completion open key '" + p_open_key + "' already exists.");
		if (p_open_key.length() < auto_brace_completion_pairs[i].open_key.length()) {
			at++;
		}
	}

	BracePair brace_pair;
	brace_pair.open_key = p_open_key;
	brace_pair.close_key = p_close_key;
	auto_brace_completion_pairs.insert(at, brace_pair);
}

void CodeEdit::set_auto_brace_completion_pairs(const Dictionary &p_auto_brace_completion_pairs) {
	auto_brace_completion_pairs.clear();

	Array keys = p_auto_brace_completion_pairs.keys();
	for (int i = 0; i < keys.size(); i++) {
		add_auto_brace_completion_pair(keys[i], p_auto_brace_completion_pairs[keys[i]]);
	}
}

Dictionary CodeEdit::get_auto_brace_completion_pairs() const {
	Dictionary brace_pairs;
	for (int i = 0; i < auto_brace_completion_pairs.size(); i++) {
		brace_pairs[auto_brace_completion_pairs[i].open_key] = auto_brace_completion_pairs[i].close_key;
	}
	return brace_pairs;
}

bool CodeEdit::has_auto_brace_completion_open_key(const String &p_open_key) const {
	for (int i = 0; i < auto_brace_completion_pairs.size(); i++) {
		if (auto_brace_completion_pairs[i].open_key == p_open_key) {
			return true;
		}
	}
	return false;
}

bool CodeEdit::has_auto_brace_completion_close_key(const String &p_close_key) const {
	for (int i = 0; i < auto_brace_completion_pairs.size(); i++) {
		if (auto_brace_completion_pairs[i].close_key == p_close_key) {
			return true;
		}
	}
	return false;
}

String CodeEdit::get_auto_brace_completion_close_key(const String &p_open_key) const {
	for (int i = 0; i < auto_brace_completion_pairs.size(); i++) {
		if (auto_brace_completion_pairs[i].open_key == p_open_key) {
			return auto_brace_completion_pairs[i].close_key;
		}
	}
	return String();
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
	if (draw_breakpoints && breakpoint_icon.is_valid()) {
		bool hovering = p_region.has_point(get_local_mouse_pos());
		bool breakpointed = is_line_breakpointed(p_line);

		if (breakpointed || (hovering && !is_dragging_cursor())) {
			int padding = p_region.size.x / 6;
			Rect2 icon_region = p_region;
			icon_region.position += Point2(padding, padding);
			icon_region.size -= Point2(padding, padding) * 2;

			// Darken icon when hovering & not yet breakpointed.
			Color use_color = hovering && !breakpointed ? breakpoint_color.darkened(0.4) : breakpoint_color;
			breakpoint_icon->draw_rect(get_canvas_item(), icon_region, false, use_color);
		}
	}

	if (draw_bookmarks && is_line_bookmarked(p_line) && bookmark_icon.is_valid()) {
		int horizontal_padding = p_region.size.x / 2;
		int vertical_padding = p_region.size.y / 4;

		Rect2 bookmark_region = p_region;
		bookmark_region.position += Point2(horizontal_padding, 0);
		bookmark_region.size -= Point2(horizontal_padding * 1.1, vertical_padding);
		bookmark_icon->draw_rect(get_canvas_item(), bookmark_region, false, bookmark_color);
	}

	if (draw_executing_lines && is_line_executing(p_line) && executing_line_icon.is_valid()) {
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
	ERR_FAIL_INDEX(p_line, get_line_count());

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
	tl->add_string(fc, font, font_size);
	int yofs = p_region.position.y + (get_line_height() - tl->get_size().y) / 2;
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
	_set_hiding_enabled(p_enabled);
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

	if (_is_line_hidden(p_line) || is_line_folded(p_line)) {
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
	const int line_count = get_line_count() - 1;
	int end_line = line_count;

	int in_comment = is_in_comment(p_line);
	int in_string = (in_comment == -1) ? is_in_string(p_line) : -1;
	if (in_string != -1 || in_comment != -1) {
		end_line = get_delimiter_end_position(p_line, get_line(p_line).size() - 1).y;
		/* End line is the same therefore we have a block of single line delimiters. */
		if (end_line == p_line) {
			for (int i = p_line + 1; i <= line_count; i++) {
				if ((in_string != -1 && is_in_string(i) == -1) || (in_comment != -1 && is_in_comment(i) == -1)) {
					break;
				}
				end_line = i;
			}
		}
	} else {
		int start_indent = get_indent_level(p_line);
		for (int i = p_line + 1; i <= line_count; i++) {
			if (get_line(i).strip_edges().size() == 0) {
				continue;
			}
			if (get_indent_level(i) > start_indent) {
				end_line = i;
				continue;
			}
			if (is_in_string(i) == -1 && is_in_comment(i) == -1) {
				break;
			}
		}
	}

	for (int i = p_line + 1; i <= end_line; i++) {
		_set_line_as_hidden(i, true);
	}

	/* Fix selection. */
	if (has_selection()) {
		if (_is_line_hidden(get_selection_from_line()) && _is_line_hidden(get_selection_to_line())) {
			deselect();
		} else if (_is_line_hidden(get_selection_from_line())) {
			select(p_line, 9999, get_selection_to_line(), get_selection_to_column());
		} else if (_is_line_hidden(get_selection_to_line())) {
			select(get_selection_from_line(), get_selection_from_column(), p_line, 9999);
		}
	}

	/* Reset caret. */
	if (_is_line_hidden(get_caret_line())) {
		set_caret_line(p_line, false, false);
		set_caret_column(get_line(p_line).length(), false);
	}
	update();
}

void CodeEdit::unfold_line(int p_line) {
	ERR_FAIL_INDEX(p_line, get_line_count());
	if (!is_line_folded(p_line) && !_is_line_hidden(p_line)) {
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
		if (!_is_line_hidden(i)) {
			break;
		}
		_set_line_as_hidden(i, false);
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
	_unhide_all_lines();
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
	return p_line + 1 < get_line_count() && !_is_line_hidden(p_line) && _is_line_hidden(p_line + 1);
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
	for (const KeyValue<int, int> &E : delimiter_cache[p_line]) {
		if (E.key > p_column) {
			break;
		}
		in_region = E.value != -1;
		start_position.x = in_region ? E.key : -1;
	}

	/* Region was found on this line and is not a multiline continuation. */
	int line_length = get_line(p_line).length();
	if (start_position.x != -1 && line_length > 0 && start_position.x != line_length + 1) {
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
		line_length = get_line(i).length();
		if (line_length > 0 && start_position.x != line_length + 1) {
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
	for (const KeyValue<int, int> &E : delimiter_cache[p_line]) {
		end_position.x = (E.value == -1) ? E.key : -1;
		if (E.key > p_column) {
			break;
		}
		region = E.value;
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
		const String prefix = p_prefixes[i];

		ERR_CONTINUE_MSG(prefix.is_empty(), "Code completion prefix cannot be empty.");
		code_completion_prefixes.insert(prefix[0]);
	}
}

TypedArray<String> CodeEdit::get_code_completion_prefixes() const {
	TypedArray<String> prefixes;
	for (const Set<char32_t>::Element *E = code_completion_prefixes.front(); E; E = E->next()) {
		prefixes.push_back(String::chr(E->get()));
	}
	return prefixes;
}

String CodeEdit::get_text_for_code_completion() const {
	StringBuilder completion_text;
	const int text_size = get_line_count();
	for (int i = 0; i < text_size; i++) {
		String line = get_line(i);

		if (i == get_caret_line()) {
			completion_text += line.substr(0, get_caret_column());
			/* Not unicode, represents the caret. */
			completion_text += String::chr(0xFFFF);
			completion_text += line.substr(get_caret_column(), line.size());
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
	if (GDVIRTUAL_CALL(_request_code_completion, p_force)) {
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
		emit_signal(SNAME("code_completion_requested"));
		return;
	}

	String line = get_line(get_caret_line());
	int ofs = CLAMP(get_caret_column(), 0, line.length());

	if (ofs > 0 && (is_in_string(get_caret_line(), ofs) != -1 || _is_char(line[ofs - 1]) || code_completion_prefixes.has(line[ofs - 1]))) {
		emit_signal(SNAME("code_completion_requested"));
	} else if (ofs > 1 && line[ofs - 1] == ' ' && code_completion_prefixes.has(line[ofs - 2])) {
		emit_signal(SNAME("code_completion_requested"));
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
	_filter_code_completion_candidates_impl();
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
	if (!is_editable() || !code_completion_active) {
		return;
	}

	if (GDVIRTUAL_CALL(_confirm_code_completion, p_replace)) {
		return;
	}

	begin_complex_operation();

	int caret_line = get_caret_line();

	const String &insert_text = code_completion_options[code_completion_current_selected].insert_text;
	const String &display_text = code_completion_options[code_completion_current_selected].display;

	if (p_replace) {
		/* Find end of current section */
		const String line = get_line(caret_line);
		int caret_col = get_caret_column();
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
		remove_text(caret_line, get_caret_column() - code_completion_base.length(), caret_remove_line, caret_col);
		set_caret_column(get_caret_column() - code_completion_base.length(), false);
		insert_text_at_caret(insert_text);
	} else {
		/* Get first non-matching char. */
		const String line = get_line(caret_line);
		int caret_col = get_caret_column();
		int matching_chars = code_completion_base.length();
		for (; matching_chars <= insert_text.length(); matching_chars++) {
			if (caret_col >= line.length() || line[caret_col] != insert_text[matching_chars]) {
				break;
			}
			caret_col++;
		}

		/* Remove base completion text. */
		remove_text(caret_line, get_caret_column() - code_completion_base.length(), caret_line, get_caret_column());
		set_caret_column(get_caret_column() - code_completion_base.length(), false);

		/* Merge with text. */
		insert_text_at_caret(insert_text.substr(0, code_completion_base.length()));
		set_caret_column(caret_col, false);
		insert_text_at_caret(insert_text.substr(matching_chars));
	}

	/* Handle merging of symbols eg strings, brackets. */
	const String line = get_line(caret_line);
	char32_t next_char = line[get_caret_column()];
	char32_t last_completion_char = insert_text[insert_text.length() - 1];
	char32_t last_completion_char_display = display_text[display_text.length() - 1];

	int pre_brace_pair = get_caret_column() > 0 ? _get_auto_brace_pair_open_at_pos(caret_line, get_caret_column()) : -1;
	int post_brace_pair = get_caret_column() < get_line(caret_line).length() ? _get_auto_brace_pair_close_at_pos(caret_line, get_caret_column()) : -1;

	if (post_brace_pair != -1 && (last_completion_char == next_char || last_completion_char_display == next_char)) {
		remove_text(caret_line, get_caret_column(), caret_line, get_caret_column() + 1);
	}

	if (pre_brace_pair != -1 && pre_brace_pair != post_brace_pair && (last_completion_char == next_char || last_completion_char_display == next_char)) {
		remove_text(caret_line, get_caret_column(), caret_line, get_caret_column() + 1);
	} else if (auto_brace_completion_enabled && pre_brace_pair != -1 && post_brace_pair == -1) {
		insert_text_at_caret(auto_brace_completion_pairs[pre_brace_pair].close_key);
		set_caret_column(get_caret_column() - auto_brace_completion_pairs[pre_brace_pair].close_key.length());
	}

	if (pre_brace_pair == -1 && post_brace_pair == -1 && get_caret_column() > 0 && get_caret_column() < get_line(caret_line).length()) {
		pre_brace_pair = _get_auto_brace_pair_open_at_pos(caret_line, get_caret_column() + 1);
		if (pre_brace_pair != -1 && pre_brace_pair == _get_auto_brace_pair_close_at_pos(caret_line, get_caret_column() - 1)) {
			remove_text(caret_line, get_caret_column() - 2, caret_line, get_caret_column());
			if (_get_auto_brace_pair_close_at_pos(caret_line, get_caret_column() - 1) != pre_brace_pair) {
				set_caret_column(get_caret_column() - 1);
			}
		}
	}

	end_complex_operation();

	cancel_code_completion();
	if (code_completion_prefixes.has(last_completion_char)) {
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

/* Line length guidelines */
void CodeEdit::set_line_length_guidelines(TypedArray<int> p_guideline_columns) {
	line_length_guideline_columns = p_guideline_columns;
	update();
}

TypedArray<int> CodeEdit::get_line_length_guidelines() const {
	return line_length_guideline_columns;
}

/* Symbol lookup */
void CodeEdit::set_symbol_lookup_on_click_enabled(bool p_enabled) {
	symbol_lookup_on_click_enabled = p_enabled;
	set_symbol_lookup_word_as_valid(false);
}

bool CodeEdit::is_symbol_lookup_on_click_enabled() const {
	return symbol_lookup_on_click_enabled;
}

String CodeEdit::get_text_for_symbol_lookup() {
	Point2i mp = get_local_mouse_pos();

	Point2i pos = get_line_column_at_pos(mp, false);
	int line = pos.y;
	int col = pos.x;

	if (line == -1) {
		return String();
	}

	StringBuilder lookup_text;
	const int text_size = get_line_count();
	for (int i = 0; i < text_size; i++) {
		String text = get_line(i);

		if (i == line) {
			lookup_text += text.substr(0, col);
			/* Not unicode, represents the cursor. */
			lookup_text += String::chr(0xFFFF);
			lookup_text += text.substr(col, text.size());
		} else {
			lookup_text += text;
		}

		if (i != text_size - 1) {
			lookup_text += "\n";
		}
	}
	return lookup_text.as_string();
}

void CodeEdit::set_symbol_lookup_word_as_valid(bool p_valid) {
	symbol_lookup_word = p_valid ? symbol_lookup_new_word : "";
	symbol_lookup_new_word = "";
	if (lookup_symbol_word != symbol_lookup_word) {
		_set_symbol_lookup_word(symbol_lookup_word);
	}
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

	/* Auto brace completion */
	ClassDB::bind_method(D_METHOD("set_auto_brace_completion_enabled", "enable"), &CodeEdit::set_auto_brace_completion_enabled);
	ClassDB::bind_method(D_METHOD("is_auto_brace_completion_enabled"), &CodeEdit::is_auto_brace_completion_enabled);

	ClassDB::bind_method(D_METHOD("set_highlight_matching_braces_enabled", "enable"), &CodeEdit::set_highlight_matching_braces_enabled);
	ClassDB::bind_method(D_METHOD("is_highlight_matching_braces_enabled"), &CodeEdit::is_highlight_matching_braces_enabled);

	ClassDB::bind_method(D_METHOD("add_auto_brace_completion_pair", "start_key", "end_key"), &CodeEdit::add_auto_brace_completion_pair);
	ClassDB::bind_method(D_METHOD("set_auto_brace_completion_pairs", "pairs"), &CodeEdit::set_auto_brace_completion_pairs);
	ClassDB::bind_method(D_METHOD("get_auto_brace_completion_pairs"), &CodeEdit::get_auto_brace_completion_pairs);

	ClassDB::bind_method(D_METHOD("has_auto_brace_completion_open_key", "open_key"), &CodeEdit::has_auto_brace_completion_open_key);
	ClassDB::bind_method(D_METHOD("has_auto_brace_completion_close_key", "close_key"), &CodeEdit::has_auto_brace_completion_close_key);

	ClassDB::bind_method(D_METHOD("get_auto_brace_completion_close_key", "open_key"), &CodeEdit::get_auto_brace_completion_close_key);

	/* Main Gutter */
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
	ClassDB::bind_method(D_METHOD("set_draw_line_numbers", "enable"), &CodeEdit::set_draw_line_numbers);
	ClassDB::bind_method(D_METHOD("is_draw_line_numbers_enabled"), &CodeEdit::is_draw_line_numbers_enabled);
	ClassDB::bind_method(D_METHOD("set_line_numbers_zero_padded", "enable"), &CodeEdit::set_line_numbers_zero_padded);
	ClassDB::bind_method(D_METHOD("is_line_numbers_zero_padded"), &CodeEdit::is_line_numbers_zero_padded);

	/* Fold Gutter */
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

	GDVIRTUAL_BIND(_confirm_code_completion, "replace")
	GDVIRTUAL_BIND(_request_code_completion, "force")
	GDVIRTUAL_BIND(_filter_code_completion_candidates, "candidates")

	/* Line length guidelines */
	ClassDB::bind_method(D_METHOD("set_line_length_guidelines", "guideline_columns"), &CodeEdit::set_line_length_guidelines);
	ClassDB::bind_method(D_METHOD("get_line_length_guidelines"), &CodeEdit::get_line_length_guidelines);

	/* Symbol lookup */
	ClassDB::bind_method(D_METHOD("set_symbol_lookup_on_click_enabled", "enable"), &CodeEdit::set_symbol_lookup_on_click_enabled);
	ClassDB::bind_method(D_METHOD("is_symbol_lookup_on_click_enabled"), &CodeEdit::is_symbol_lookup_on_click_enabled);

	ClassDB::bind_method(D_METHOD("get_text_for_symbol_lookup"), &CodeEdit::get_text_for_symbol_lookup);

	ClassDB::bind_method(D_METHOD("set_symbol_lookup_word_as_valid", "valid"), &CodeEdit::set_symbol_lookup_word_as_valid);

	/* Inspector */
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "symbol_lookup_on_click"), "set_symbol_lookup_on_click_enabled", "is_symbol_lookup_on_click_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "line_folding"), "set_line_folding_enabled", "is_line_folding_enabled");

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "line_length_guidelines"), "set_line_length_guidelines", "get_line_length_guidelines");

	ADD_GROUP("Gutters", "gutters_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gutters_draw_breakpoints_gutter"), "set_draw_breakpoints_gutter", "is_drawing_breakpoints_gutter");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gutters_draw_bookmarks"), "set_draw_bookmarks_gutter", "is_drawing_bookmarks_gutter");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gutters_draw_executing_lines"), "set_draw_executing_lines_gutter", "is_drawing_executing_lines_gutter");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gutters_draw_line_numbers"), "set_draw_line_numbers", "is_draw_line_numbers_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gutters_zero_pad_line_numbers"), "set_line_numbers_zero_padded", "is_line_numbers_zero_padded");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gutters_draw_fold_gutter"), "set_draw_fold_gutter", "is_drawing_fold_gutter");

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

	ADD_GROUP("Auto brace completion", "auto_brace_completion_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_brace_completion_enabled"), "set_auto_brace_completion_enabled", "is_auto_brace_completion_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_brace_completion_highlight_matching"), "set_highlight_matching_braces_enabled", "is_highlight_matching_braces_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "auto_brace_completion_pairs"), "set_auto_brace_completion_pairs", "get_auto_brace_completion_pairs");

	/* Signals */
	/* Gutters */
	ADD_SIGNAL(MethodInfo("breakpoint_toggled", PropertyInfo(Variant::INT, "line")));

	/* Code Completion */
	ADD_SIGNAL(MethodInfo("code_completion_requested"));

	/* Symbol lookup */
	ADD_SIGNAL(MethodInfo("symbol_lookup", PropertyInfo(Variant::STRING, "symbol"), PropertyInfo(Variant::INT, "line"), PropertyInfo(Variant::INT, "column")));
	ADD_SIGNAL(MethodInfo("symbol_validate", PropertyInfo(Variant::STRING, "symbol")));
}

/* Auto brace completion */
int CodeEdit::_get_auto_brace_pair_open_at_pos(int p_line, int p_col) {
	const String &line = get_line(p_line);

	/* Should be fast enough, expecting low amount of pairs... */
	for (int i = 0; i < auto_brace_completion_pairs.size(); i++) {
		const String &open_key = auto_brace_completion_pairs[i].open_key;
		if (p_col - open_key.length() < 0) {
			continue;
		}

		bool is_match = true;
		for (int j = 0; j < open_key.length(); j++) {
			if (line[(p_col - 1) - j] != open_key[(open_key.length() - 1) - j]) {
				is_match = false;
				break;
			}
		}

		if (is_match) {
			return i;
		}
	}
	return -1;
}

int CodeEdit::_get_auto_brace_pair_close_at_pos(int p_line, int p_col) {
	const String &line = get_line(p_line);

	/* Should be fast enough, expecting low amount of pairs... */
	for (int i = 0; i < auto_brace_completion_pairs.size(); i++) {
		if (p_col + auto_brace_completion_pairs[i].close_key.length() > line.length()) {
			continue;
		}

		bool is_match = true;
		for (int j = 0; j < auto_brace_completion_pairs[i].close_key.length(); j++) {
			if (line[p_col + j] != auto_brace_completion_pairs[i].close_key[j]) {
				is_match = false;
				break;
			}
		}

		if (is_match) {
			return i;
		}
	}
	return -1;
}

/* Gutters */
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
		set_caret_line(p_line + 1);
		set_caret_column(0);
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
				delimiter_cache.remove_at(i);
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
		/* If column is specified, loop until the key is larger then the column. */
		if (p_column != -1) {
			if (E->key() > p_column) {
				break;
			}
			in_region = E->value() != -1 && delimiters[E->value()].type == p_type;
			region = in_region ? E->value() : -1;
			continue;
		}

		/* If no column, calculate if the entire line is a region       */
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
	// If we are the editor allow "null" as a valid start key, otherwise users cannot add delimiters via the inspector.
	if (!(Engine::get_singleton()->is_editor_hint() && p_start_key == "null")) {
		ERR_FAIL_COND_MSG(p_start_key.is_empty(), "delimiter start key cannot be empty");

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
	delimiter.line_only = p_line_only || p_end_key.is_empty();
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

		delimiters.remove_at(i);
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
		String key = p_delimiters[i];

		if (key.is_empty()) {
			continue;
		}

		const String start_key = key.get_slice(" ", 0);
		const String end_key = key.get_slice_count(" ") > 1 ? key.get_slice(" ", 1) : String();

		_add_delimiter(start_key, end_key, end_key.is_empty(), p_type);
	}
	setting_delimiters = false;
	_update_delimiter_cache();
}

void CodeEdit::_clear_delimiters(DelimiterType p_type) {
	for (int i = delimiters.size() - 1; i >= 0; i--) {
		if (delimiters[i].type == p_type) {
			delimiters.remove_at(i);
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
void CodeEdit::_filter_code_completion_candidates_impl() {
	int line_height = get_line_height();

	if (GDVIRTUAL_IS_OVERRIDDEN(_filter_code_completion_candidates)) {
		code_completion_options.clear();
		code_completion_base = "";

		/* Build options argument. */
		TypedArray<Dictionary> completion_options_sources;
		completion_options_sources.resize(code_completion_option_sources.size());
		int i = 0;
		for (const ScriptCodeCompletionOption &E : code_completion_option_sources) {
			Dictionary option;
			option["kind"] = E.kind;
			option["display_text"] = E.display;
			option["insert_text"] = E.insert_text;
			option["font_color"] = E.font_color;
			option["icon"] = E.icon;
			option["default_value"] = E.default_value;
			completion_options_sources[i] = option;
			i++;
		}

		Array completion_options;

		GDVIRTUAL_CALL(_filter_code_completion_candidates, completion_options_sources, completion_options);

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

			int offset = 0;
			if (option.default_value.get_type() == Variant::COLOR) {
				offset = line_height;
			}

			max_width = MAX(max_width, font->get_string_size(option.display, font_size).width + offset);
			code_completion_options.push_back(option);
		}

		code_completion_longest_line = MIN(max_width, code_completion_max_width * font_size);
		code_completion_current_selected = 0;
		code_completion_active = true;
		update();
		return;
	}

	const int caret_line = get_caret_line();
	const int caret_column = get_caret_column();
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
	if (caret_column > 0 && in_string == -1 && first_quote_col == cofs - 1) {
		cancel_code_completion();
		return;
		/* In a string, therefore we are trying to complete the string text. */
	} else if (in_string != -1 && first_quote_col != -1) {
		int key_length = delimiters[in_string].start_key.length();
		string_to_complete = line.substr(first_quote_col - key_length, (cofs - first_quote_col) + key_length);
		/* If we have a space, previous word might be a keyword. eg "func |". */
	} else if (cofs > 0 && line[cofs - 1] == ' ') {
		int ofs = cofs - 1;
		while (ofs > 0 && line[ofs] == ' ') {
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
	if (cofs > 0 && code_completion_prefixes.has(line[cofs - 1])) {
		prev_is_prefix = true;
	} else if (cofs > 1 && line[cofs - 1] == ' ' && code_completion_prefixes.has(line[cofs - 2])) {
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
	Vector<ScriptCodeCompletionOption> completion_options_substr;
	Vector<ScriptCodeCompletionOption> completion_options_substr_casei;
	Vector<ScriptCodeCompletionOption> completion_options_subseq;
	Vector<ScriptCodeCompletionOption> completion_options_subseq_casei;

	int max_width = 0;
	String string_to_complete_lower = string_to_complete.to_lower();
	for (ScriptCodeCompletionOption &option : code_completion_option_sources) {
		if (single_quote && option.display.is_quoted()) {
			option.display = option.display.unquote().quote("'");
		}

		int offset = 0;
		if (option.default_value.get_type() == Variant::COLOR) {
			offset = line_height;
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
			max_width = MAX(max_width, font->get_string_size(option.display, font_size).width + offset);
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

		const char32_t *sst = &string_to_complete[0];
		const char32_t *sst_lower = &display_lower[0];

		Vector<Pair<int, int>> ssq_matches;
		int ssq_match_start = 0;
		int ssq_match_len = 0;

		Vector<Pair<int, int>> ssq_lower_matches;
		int ssq_lower_match_start = 0;
		int ssq_lower_match_len = 0;

		int sst_start = -1;
		int sst_lower_start = -1;

		for (int i = 0; *tgt; tgt++, tgt_lower++, i++) {
			// Check substring.
			if (*sst == *tgt) {
				sst++;
				if (sst_start == -1) {
					sst_start = i;
				}
			} else if (sst_start != -1 && *sst) {
				sst = &string_to_complete[0];
				sst_start = -1;
			}

			// Check subsequence.
			if (*ssq == *tgt) {
				ssq++;
				if (ssq_match_len == 0) {
					ssq_match_start = i;
				}
				ssq_match_len++;
			} else if (ssq_match_len > 0) {
				ssq_matches.push_back(Pair<int, int>(ssq_match_start, ssq_match_len));
				ssq_match_len = 0;
			}

			// Check lower substring.
			if (*sst_lower == *tgt) {
				sst_lower++;
				if (sst_lower_start == -1) {
					sst_lower_start = i;
				}
			} else if (sst_lower_start != -1 && *sst_lower) {
				sst_lower = &string_to_complete[0];
				sst_lower_start = -1;
			}

			// Check lower subsequence.
			if (*ssq_lower == *tgt_lower) {
				ssq_lower++;
				if (ssq_lower_match_len == 0) {
					ssq_lower_match_start = i;
				}
				ssq_lower_match_len++;
			} else if (ssq_lower_match_len > 0) {
				ssq_lower_matches.push_back(Pair<int, int>(ssq_lower_match_start, ssq_lower_match_len));
				ssq_lower_match_len = 0;
			}
		}

		/* Matched the whole subsequence in s. */
		if (!*ssq) { // Matched the whole subsequence in s.
			option.matches.clear();

			if (sst_start == 0) { // Matched substring in beginning of s.
				option.matches.push_back(Pair<int, int>(sst_start, string_to_complete.length()));
				code_completion_options.push_back(option);
			} else if (sst_start > 0) { // Matched substring in s.
				option.matches.push_back(Pair<int, int>(sst_start, string_to_complete.length()));
				completion_options_substr.push_back(option);
			} else {
				if (ssq_match_len > 0) {
					ssq_matches.push_back(Pair<int, int>(ssq_match_start, ssq_match_len));
				}
				option.matches.append_array(ssq_matches);
				completion_options_subseq.push_back(option);
			}
			max_width = MAX(max_width, font->get_string_size(option.display, font_size).width + offset);
		} else if (!*ssq_lower) { // Matched the whole subsequence in s_lower.
			option.matches.clear();

			if (sst_lower_start == 0) { // Matched substring in beginning of s_lower.
				option.matches.push_back(Pair<int, int>(sst_lower_start, string_to_complete.length()));
				completion_options_casei.push_back(option);
			} else if (sst_lower_start > 0) { // Matched substring in s_lower.
				option.matches.push_back(Pair<int, int>(sst_lower_start, string_to_complete.length()));
				completion_options_substr_casei.push_back(option);
			} else {
				if (ssq_lower_match_len > 0) {
					ssq_lower_matches.push_back(Pair<int, int>(ssq_lower_match_start, ssq_lower_match_len));
				}
				option.matches.append_array(ssq_lower_matches);
				completion_options_subseq_casei.push_back(option);
			}
			max_width = MAX(max_width, font->get_string_size(option.display, font_size).width + offset);
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

	code_completion_longest_line = MIN(max_width, code_completion_max_width * font_size);
	code_completion_current_selected = 0;
	code_completion_active = true;
	update();
}

void CodeEdit::_lines_edited_from(int p_from_line, int p_to_line) {
	_update_delimiter_cache(p_from_line, p_to_line);

	if (p_from_line == p_to_line) {
		return;
	}

	lines_edited_changed += p_to_line - p_from_line;
	lines_edited_from = (lines_edited_from == -1) ? MIN(p_from_line, p_to_line) : MIN(lines_edited_from, MIN(p_from_line, p_to_line));
	lines_edited_to = (lines_edited_to == -1) ? MAX(p_from_line, p_to_line) : MAX(lines_edited_from, MAX(p_from_line, p_to_line));
}

void CodeEdit::_text_set() {
	lines_edited_from = 0;
	lines_edited_to = 9999;
	_text_changed();
}

void CodeEdit::_text_changed() {
	if (lines_edited_from == -1) {
		return;
	}

	int lc = get_line_count();
	line_number_digits = 1;
	while (lc /= 10) {
		line_number_digits++;
	}

	if (font.is_valid()) {
		set_gutter_width(line_number_gutter, (line_number_digits + 1) * font->get_char_size('0', 0, font_size).width);
	}

	lc = get_line_count();
	List<int> breakpoints;
	breakpointed_lines.get_key_list(&breakpoints);
	for (const int &line : breakpoints) {
		if (line < lines_edited_from || (line < lc && is_line_breakpointed(line))) {
			continue;
		}

		breakpointed_lines.erase(line);
		emit_signal(SNAME("breakpoint_toggled"), line);

		int next_line = line + lines_edited_changed;
		if (next_line > -1 && next_line < lc && is_line_breakpointed(next_line)) {
			emit_signal(SNAME("breakpoint_toggled"), next_line);
			breakpointed_lines[next_line] = true;
			continue;
		}
	}

	lines_edited_from = -1;
	lines_edited_to = -1;
	lines_edited_changed = 0;
}

CodeEdit::CodeEdit() {
	/* Indent management */
	auto_indent_prefixes.insert(':');
	auto_indent_prefixes.insert('{');
	auto_indent_prefixes.insert('[');
	auto_indent_prefixes.insert('(');

	/* Auto brace completion */
	add_auto_brace_completion_pair("(", ")");
	add_auto_brace_completion_pair("{", "}");
	add_auto_brace_completion_pair("[", "]");
	add_auto_brace_completion_pair("\"", "\"");
	add_auto_brace_completion_pair("\'", "\'");

	/* Delimiter tracking */
	add_string_delimiter("\"", "\"", false);
	add_string_delimiter("\'", "\'", false);

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
	set_gutter_custom_draw(gutter_idx, callable_mp(this, &CodeEdit::_main_gutter_draw_callback));
	gutter_idx++;

	/* Line numbers */
	add_gutter();
	set_gutter_name(gutter_idx, "line_numbers");
	set_gutter_draw(gutter_idx, false);
	set_gutter_type(gutter_idx, GUTTER_TYPE_CUSTOM);
	set_gutter_custom_draw(gutter_idx, callable_mp(this, &CodeEdit::_line_number_draw_callback));
	gutter_idx++;

	/* Fold Gutter */
	add_gutter();
	set_gutter_name(gutter_idx, "fold_gutter");
	set_gutter_draw(gutter_idx, false);
	set_gutter_type(gutter_idx, GUTTER_TYPE_CUSTOM);
	set_gutter_custom_draw(gutter_idx, callable_mp(this, &CodeEdit::_fold_gutter_draw_callback));
	gutter_idx++;

	connect("lines_edited_from", callable_mp(this, &CodeEdit::_lines_edited_from));
	connect("text_set", callable_mp(this, &CodeEdit::_text_set));
	connect("text_changed", callable_mp(this, &CodeEdit::_text_changed));

	connect("gutter_clicked", callable_mp(this, &CodeEdit::_gutter_clicked));
	connect("gutter_added", callable_mp(this, &CodeEdit::_update_gutter_indexes));
	connect("gutter_removed", callable_mp(this, &CodeEdit::_update_gutter_indexes));
	_update_gutter_indexes();
}

CodeEdit::~CodeEdit() {
}
