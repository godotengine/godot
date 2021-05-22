/*************************************************************************/
/*  text_edit.cpp                                                        */
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

#include "text_edit.h"

#include "core/config/project_settings.h"
#include "core/input/input.h"
#include "core/input/input_map.h"
#include "core/object/message_queue.h"
#include "core/object/script_language.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "core/string/translation.h"

#include "scene/main/window.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_scale.h"
#endif

#define TAB_PIXELS

inline bool _is_symbol(char32_t c) {
	return is_symbol(c);
}

static bool _is_text_char(char32_t c) {
	return !is_symbol(c);
}

static bool _is_whitespace(char32_t c) {
	return c == '\t' || c == ' ';
}

static bool _is_char(char32_t c) {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static bool _is_pair_right_symbol(char32_t c) {
	return c == '"' ||
		   c == '\'' ||
		   c == ')' ||
		   c == ']' ||
		   c == '}';
}

static bool _is_pair_left_symbol(char32_t c) {
	return c == '"' ||
		   c == '\'' ||
		   c == '(' ||
		   c == '[' ||
		   c == '{';
}

static bool _is_pair_symbol(char32_t c) {
	return _is_pair_left_symbol(c) || _is_pair_right_symbol(c);
}

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

static int _find_first_non_whitespace_column_of_line(const String &line) {
	int left = 0;
	while (left < line.length() && _is_whitespace(line[left])) {
		left++;
	}
	return left;
}

///////////////////////////////////////////////////////////////////////////////

void TextEdit::Text::set_font(const Ref<Font> &p_font) {
	font = p_font;
}

void TextEdit::Text::set_font_size(int p_font_size) {
	font_size = p_font_size;
}

void TextEdit::Text::set_indent_size(int p_indent_size) {
	indent_size = p_indent_size;
}

void TextEdit::Text::set_font_features(const Dictionary &p_features) {
	opentype_features = p_features;
}

void TextEdit::Text::set_direction_and_language(TextServer::Direction p_direction, String p_language) {
	direction = p_direction;
	language = p_language;
}

void TextEdit::Text::set_draw_control_chars(bool p_draw_control_chars) {
	draw_control_chars = p_draw_control_chars;
}

int TextEdit::Text::get_line_width(int p_line) const {
	ERR_FAIL_INDEX_V(p_line, text.size(), 0);
	return text[p_line].data_buf->get_size().x;
}

int TextEdit::Text::get_line_height(int p_line, int p_wrap_index) const {
	ERR_FAIL_INDEX_V(p_line, text.size(), 0);

	return text[p_line].data_buf->get_line_size(p_wrap_index).y;
}

void TextEdit::Text::set_width(float p_width) {
	width = p_width;
}

int TextEdit::Text::get_line_wrap_amount(int p_line) const {
	ERR_FAIL_INDEX_V(p_line, text.size(), 0);

	return text[p_line].data_buf->get_line_count() - 1;
}

Vector<Vector2i> TextEdit::Text::get_line_wrap_ranges(int p_line) const {
	Vector<Vector2i> ret;
	ERR_FAIL_INDEX_V(p_line, text.size(), ret);

	for (int i = 0; i < text[p_line].data_buf->get_line_count(); i++) {
		ret.push_back(text[p_line].data_buf->get_line_range(i));
	}
	return ret;
}

const Ref<TextParagraph> TextEdit::Text::get_line_data(int p_line) const {
	ERR_FAIL_INDEX_V(p_line, text.size(), Ref<TextParagraph>());
	return text[p_line].data_buf;
}

_FORCE_INLINE_ const String &TextEdit::Text::operator[](int p_line) const {
	return text[p_line].data;
}

void TextEdit::Text::invalidate_cache(int p_line, int p_column, const String &p_ime_text, const Vector<Vector2i> &p_bidi_override) {
	ERR_FAIL_INDEX(p_line, text.size());

	if (font.is_null() || font_size <= 0) {
		return; // Not in tree?
	}

	text.write[p_line].data_buf->clear();
	text.write[p_line].data_buf->set_width(width);
	text.write[p_line].data_buf->set_direction((TextServer::Direction)direction);
	text.write[p_line].data_buf->set_preserve_control(draw_control_chars);
	if (p_ime_text.length() > 0) {
		text.write[p_line].data_buf->add_string(p_ime_text, font, font_size, opentype_features, language);
		if (!p_bidi_override.is_empty()) {
			TS->shaped_text_set_bidi_override(text.write[p_line].data_buf->get_rid(), p_bidi_override);
		}
	} else {
		text.write[p_line].data_buf->add_string(text[p_line].data, font, font_size, opentype_features, language);
		if (!text[p_line].bidi_override.is_empty()) {
			TS->shaped_text_set_bidi_override(text.write[p_line].data_buf->get_rid(), text[p_line].bidi_override);
		}
	}

	// Apply tab align.
	if (indent_size > 0) {
		Vector<float> tabs;
		tabs.push_back(font->get_char_size(' ', 0, font_size).width * indent_size);
		text.write[p_line].data_buf->tab_align(tabs);
	}
}

void TextEdit::Text::invalidate_all_lines() {
	for (int i = 0; i < text.size(); i++) {
		text.write[i].data_buf->set_width(width);
		if (indent_size > 0) {
			Vector<float> tabs;
			tabs.push_back(font->get_char_size(' ', 0, font_size).width * indent_size);
			text.write[i].data_buf->tab_align(tabs);
		}
	}
}

void TextEdit::Text::invalidate_all() {
	for (int i = 0; i < text.size(); i++) {
		invalidate_cache(i);
	}
}

void TextEdit::Text::clear() {
	text.clear();
	insert(0, "", Vector<Vector2i>());
}

int TextEdit::Text::get_max_width(bool p_exclude_hidden) const {
	// Quite some work, but should be fast enough.

	int max = 0;
	for (int i = 0; i < text.size(); i++) {
		if (!p_exclude_hidden || !is_hidden(i)) {
			max = MAX(max, get_line_width(i));
		}
	}
	return max;
}

void TextEdit::Text::set(int p_line, const String &p_text, const Vector<Vector2i> &p_bidi_override) {
	ERR_FAIL_INDEX(p_line, text.size());

	text.write[p_line].data = p_text;
	text.write[p_line].bidi_override = p_bidi_override;
	invalidate_cache(p_line);
}

void TextEdit::Text::insert(int p_at, const String &p_text, const Vector<Vector2i> &p_bidi_override) {
	Line line;
	line.gutters.resize(gutter_count);
	line.hidden = false;
	line.data = p_text;
	line.bidi_override = p_bidi_override;
	text.insert(p_at, line);

	invalidate_cache(p_at);
}

void TextEdit::Text::remove(int p_at) {
	text.remove(p_at);
}

void TextEdit::Text::add_gutter(int p_at) {
	for (int i = 0; i < text.size(); i++) {
		if (p_at < 0 || p_at > gutter_count) {
			text.write[i].gutters.push_back(Gutter());
		} else {
			text.write[i].gutters.insert(p_at, Gutter());
		}
	}
	gutter_count++;
}

void TextEdit::Text::remove_gutter(int p_gutter) {
	for (int i = 0; i < text.size(); i++) {
		text.write[i].gutters.remove(p_gutter);
	}
	gutter_count--;
}

void TextEdit::Text::move_gutters(int p_from_line, int p_to_line) {
	text.write[p_to_line].gutters = text[p_from_line].gutters;
	text.write[p_from_line].gutters.clear();
	text.write[p_from_line].gutters.resize(gutter_count);
}

////////////////////////////////////////////////////////////////////////////////

void TextEdit::_update_scrollbars() {
	Size2 size = get_size();
	Size2 hmin = h_scroll->get_combined_minimum_size();
	Size2 vmin = v_scroll->get_combined_minimum_size();

	v_scroll->set_begin(Point2(size.width - vmin.width, cache.style_normal->get_margin(SIDE_TOP)));
	v_scroll->set_end(Point2(size.width, size.height - cache.style_normal->get_margin(SIDE_TOP) - cache.style_normal->get_margin(SIDE_BOTTOM)));

	h_scroll->set_begin(Point2(0, size.height - hmin.height));
	h_scroll->set_end(Point2(size.width - vmin.width, size.height));

	int visible_rows = get_visible_rows();
	int total_rows = get_total_visible_rows();
	if (scroll_past_end_of_file_enabled) {
		total_rows += visible_rows - 1;
	}

	int visible_width = size.width - cache.style_normal->get_minimum_size().width;
	int total_width = text.get_max_width(true) + vmin.x + gutters_width + gutter_padding;

	if (draw_minimap) {
		total_width += cache.minimap_width;
	}

	updating_scrolls = true;

	if (total_rows > visible_rows) {
		v_scroll->show();
		v_scroll->set_max(total_rows + get_visible_rows_offset());
		v_scroll->set_page(visible_rows + get_visible_rows_offset());
		if (smooth_scroll_enabled) {
			v_scroll->set_step(0.25);
		} else {
			v_scroll->set_step(1);
		}
		set_v_scroll(get_v_scroll());

	} else {
		cursor.line_ofs = 0;
		cursor.wrap_ofs = 0;
		v_scroll->set_value(0);
		v_scroll->hide();
	}

	if (total_width > visible_width && !is_wrap_enabled()) {
		h_scroll->show();
		h_scroll->set_max(total_width);
		h_scroll->set_page(visible_width);
		if (cursor.x_ofs > (total_width - visible_width)) {
			cursor.x_ofs = (total_width - visible_width);
		}
		if (fabs(h_scroll->get_value() - (double)cursor.x_ofs) >= 1) {
			h_scroll->set_value(cursor.x_ofs);
		}

	} else {
		cursor.x_ofs = 0;
		h_scroll->set_value(0);
		h_scroll->hide();
	}

	updating_scrolls = false;
}

void TextEdit::_click_selection_held() {
	// Warning: is_mouse_button_pressed(MOUSE_BUTTON_LEFT) returns false for double+ clicks, so this doesn't work for MODE_WORD
	// and MODE_LINE. However, moving the mouse triggers _gui_input, which calls these functions too, so that's not a huge problem.
	// I'm unsure if there's an actual fix that doesn't have a ton of side effects.
	if (Input::get_singleton()->is_mouse_button_pressed(MOUSE_BUTTON_LEFT) && selection.selecting_mode != SelectionMode::SELECTION_MODE_NONE) {
		switch (selection.selecting_mode) {
			case SelectionMode::SELECTION_MODE_POINTER: {
				_update_selection_mode_pointer();
			} break;
			case SelectionMode::SELECTION_MODE_WORD: {
				_update_selection_mode_word();
			} break;
			case SelectionMode::SELECTION_MODE_LINE: {
				_update_selection_mode_line();
			} break;
			default: {
				break;
			}
		}
	} else {
		click_select_held->stop();
	}
}

Point2 TextEdit::_get_local_mouse_pos() const {
	Point2 mp = get_local_mouse_position();
	if (is_layout_rtl()) {
		mp.x = get_size().width - mp.x;
	}
	return mp;
}

void TextEdit::_update_selection_mode_pointer() {
	dragging_selection = true;
	Point2 mp = _get_local_mouse_pos();

	int row, col;
	_get_mouse_pos(Point2i(mp.x, mp.y), row, col);

	select(selection.selecting_line, selection.selecting_column, row, col);

	cursor_set_line(row, false);
	cursor_set_column(col);
	update();

	click_select_held->start();
}

void TextEdit::_update_selection_mode_word() {
	dragging_selection = true;
	Point2 mp = _get_local_mouse_pos();

	int row, col;
	_get_mouse_pos(Point2i(mp.x, mp.y), row, col);

	String line = text[row];
	int cursor_pos = CLAMP(col, 0, line.length());
	int beg = cursor_pos;
	int end = beg;
	Vector<Vector2i> words = TS->shaped_text_get_word_breaks(text.get_line_data(row)->get_rid());
	for (int i = 0; i < words.size(); i++) {
		if (words[i].x < cursor_pos && words[i].y > cursor_pos) {
			beg = words[i].x;
			end = words[i].y;
			break;
		}
	}

	// Initial selection.
	if (!selection.active) {
		select(row, beg, row, end);
		selection.selecting_column = beg;
		selection.selected_word_beg = beg;
		selection.selected_word_end = end;
		selection.selected_word_origin = beg;
		cursor_set_line(selection.to_line, false);
		cursor_set_column(selection.to_column);
	} else {
		if ((col <= selection.selected_word_origin && row == selection.selecting_line) || row < selection.selecting_line) {
			selection.selecting_column = selection.selected_word_end;
			select(row, beg, selection.selecting_line, selection.selected_word_end);
			cursor_set_line(selection.from_line, false);
			cursor_set_column(selection.from_column);
		} else {
			selection.selecting_column = selection.selected_word_beg;
			select(selection.selecting_line, selection.selected_word_beg, row, end);
			cursor_set_line(selection.to_line, false);
			cursor_set_column(selection.to_column);
		}
	}

	update();

	click_select_held->start();
}

void TextEdit::_update_selection_mode_line() {
	dragging_selection = true;
	Point2 mp = _get_local_mouse_pos();

	int row, col;
	_get_mouse_pos(Point2i(mp.x, mp.y), row, col);

	col = 0;
	if (row < selection.selecting_line) {
		// Cursor is above us.
		cursor_set_line(row - 1, false);
		selection.selecting_column = text[selection.selecting_line].length();
	} else {
		// Cursor is below us.
		cursor_set_line(row + 1, false);
		selection.selecting_column = 0;
		col = text[row].length();
	}
	cursor_set_column(0);

	select(selection.selecting_line, selection.selecting_column, row, col);
	update();

	click_select_held->start();
}

void TextEdit::_update_minimap_click() {
	Point2 mp = _get_local_mouse_pos();

	int xmargin_end = get_size().width - cache.style_normal->get_margin(SIDE_RIGHT);
	if (!dragging_minimap && (mp.x < xmargin_end - minimap_width || mp.y > xmargin_end)) {
		minimap_clicked = false;
		return;
	}
	minimap_clicked = true;
	dragging_minimap = true;

	int row;
	_get_minimap_mouse_row(Point2i(mp.x, mp.y), row);

	if (row >= get_first_visible_line() && (row < get_last_full_visible_line() || row >= (text.size() - 1))) {
		minimap_scroll_ratio = v_scroll->get_as_ratio();
		minimap_scroll_click_pos = mp.y;
		can_drag_minimap = true;
		return;
	}

	int wi;
	int first_line = row - num_lines_from_rows(row, 0, -get_visible_rows() / 2, wi) + 1;
	double delta = get_scroll_pos_for_line(first_line, wi) - get_v_scroll();
	if (delta < 0) {
		_scroll_up(-delta);
	} else {
		_scroll_down(delta);
	}
}

void TextEdit::_update_minimap_drag() {
	if (!can_drag_minimap) {
		return;
	}

	int control_height = _get_control_height();
	int scroll_height = v_scroll->get_max() * (minimap_char_size.y + minimap_line_spacing);
	if (control_height > scroll_height) {
		control_height = scroll_height;
	}

	Point2 mp = _get_local_mouse_pos();

	double diff = (mp.y - minimap_scroll_click_pos) / control_height;
	v_scroll->set_as_ratio(minimap_scroll_ratio + diff);
}

void TextEdit::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_update_caches();
			if (cursor_changed_dirty) {
				MessageQueue::get_singleton()->push_call(this, "_cursor_changed_emit");
			}
			if (text_changed_dirty) {
				MessageQueue::get_singleton()->push_call(this, "_text_changed_emit");
			}
			_update_wrap_at(true);
		} break;
		case NOTIFICATION_RESIZED: {
			_update_scrollbars();
			_update_wrap_at();
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (is_visible()) {
				call_deferred("_update_scrollbars");
				call_deferred("_update_wrap_at");
			}
		} break;
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_THEME_CHANGED: {
			_update_caches();
			_update_wrap_at(true);
		} break;
		case NOTIFICATION_WM_WINDOW_FOCUS_IN: {
			window_has_focus = true;
			draw_caret = true;
			update();
		} break;
		case NOTIFICATION_WM_WINDOW_FOCUS_OUT: {
			window_has_focus = false;
			draw_caret = false;
			update();
		} break;
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (scrolling && get_v_scroll() != target_v_scroll) {
				double target_y = target_v_scroll - get_v_scroll();
				double dist = sqrt(target_y * target_y);
				// To ensure minimap is responsive override the speed setting.
				double vel = ((target_y / dist) * ((minimap_clicked) ? 3000 : v_scroll_speed)) * get_physics_process_delta_time();

				if (Math::abs(vel) >= dist) {
					set_v_scroll(target_v_scroll);
					scrolling = false;
					minimap_clicked = false;
					set_physics_process_internal(false);
				} else {
					set_v_scroll(get_v_scroll() + vel);
				}
			} else {
				scrolling = false;
				minimap_clicked = false;
				set_physics_process_internal(false);
			}
		} break;
		case NOTIFICATION_DRAW: {
			if (first_draw) {
				// Size may not be the final one, so attempts to ensure cursor was visible may have failed.
				adjust_viewport_to_cursor();
				first_draw = false;
			}

			/* Prevent the resource getting lost between the editor and game. */
			if (Engine::get_singleton()->is_editor_hint()) {
				if (syntax_highlighter.is_valid() && syntax_highlighter->get_text_edit() != this) {
					syntax_highlighter->set_text_edit(this);
				}
			}

			Size2 size = get_size();
			bool rtl = is_layout_rtl();
			if ((!has_focus() && !menu->has_focus()) || !window_has_focus) {
				draw_caret = false;
			}

			cache.minimap_width = 0;
			if (draw_minimap) {
				cache.minimap_width = minimap_width;
			}

			_update_scrollbars();

			RID ci = get_canvas_item();
			RenderingServer::get_singleton()->canvas_item_set_clip(get_canvas_item(), true);
			int xmargin_beg = cache.style_normal->get_margin(SIDE_LEFT) + gutters_width + gutter_padding;

			int xmargin_end = size.width - cache.style_normal->get_margin(SIDE_RIGHT) - cache.minimap_width;
			// Let's do it easy for now.
			cache.style_normal->draw(ci, Rect2(Point2(), size));
			if (readonly) {
				cache.style_readonly->draw(ci, Rect2(Point2(), size));
				draw_caret = false;
			}
			if (has_focus()) {
				cache.style_focus->draw(ci, Rect2(Point2(), size));
			}

			int visible_rows = get_visible_rows() + 1;

			Color color = readonly ? cache.font_readonly_color : cache.font_color;

			if (cache.background_color.a > 0.01) {
				RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(Point2i(), get_size()), cache.background_color);
			}

			if (line_length_guidelines) {
				const int hard_x = xmargin_beg + (int)cache.font->get_char_size('0', 0, cache.font_size).width * line_length_guideline_hard_col - cursor.x_ofs;
				if (hard_x > xmargin_beg && hard_x < xmargin_end) {
					if (rtl) {
						RenderingServer::get_singleton()->canvas_item_add_line(ci, Point2(size.width - hard_x, 0), Point2(size.width - hard_x, size.height), cache.line_length_guideline_color);
					} else {
						RenderingServer::get_singleton()->canvas_item_add_line(ci, Point2(hard_x, 0), Point2(hard_x, size.height), cache.line_length_guideline_color);
					}
				}

				// Draw a "Soft" line length guideline, less visible than the hard line length guideline.
				// It's usually set to a lower column compared to the hard line length guideline.
				// Only drawn if its column differs from the hard line length guideline.
				const int soft_x = xmargin_beg + (int)cache.font->get_char_size('0', 0, cache.font_size).width * line_length_guideline_soft_col - cursor.x_ofs;
				if (hard_x != soft_x && soft_x > xmargin_beg && soft_x < xmargin_end) {
					if (rtl) {
						RenderingServer::get_singleton()->canvas_item_add_line(ci, Point2(size.width - soft_x, 0), Point2(size.width - soft_x, size.height), cache.line_length_guideline_color * Color(1, 1, 1, 0.5));
					} else {
						RenderingServer::get_singleton()->canvas_item_add_line(ci, Point2(soft_x, 0), Point2(soft_x, size.height), cache.line_length_guideline_color * Color(1, 1, 1, 0.5));
					}
				}
			}

			int brace_open_match_line = -1;
			int brace_open_match_column = -1;
			bool brace_open_matching = false;
			bool brace_open_mismatch = false;
			int brace_close_match_line = -1;
			int brace_close_match_column = -1;
			bool brace_close_matching = false;
			bool brace_close_mismatch = false;

			if (brace_matching_enabled && cursor.line >= 0 && cursor.line < text.size() && cursor.column >= 0) {
				if (cursor.column < text[cursor.line].length()) {
					// Check for open.
					char32_t c = text[cursor.line][cursor.column];
					char32_t closec = 0;

					if (c == '[') {
						closec = ']';
					} else if (c == '{') {
						closec = '}';
					} else if (c == '(') {
						closec = ')';
					}

					if (closec != 0) {
						int stack = 1;

						for (int i = cursor.line; i < text.size(); i++) {
							int from = i == cursor.line ? cursor.column + 1 : 0;
							for (int j = from; j < text[i].length(); j++) {
								char32_t cc = text[i][j];
								// Ignore any brackets inside a string.
								if (cc == '"' || cc == '\'') {
									char32_t quotation = cc;
									do {
										j++;
										if (!(j < text[i].length())) {
											break;
										}
										cc = text[i][j];
										// Skip over escaped quotation marks inside strings.
										if (cc == '\\') {
											bool escaped = true;
											while (j + 1 < text[i].length() && text[i][j + 1] == '\\') {
												escaped = !escaped;
												j++;
											}
											if (escaped) {
												j++;
												continue;
											}
										}
									} while (cc != quotation);
								} else if (cc == c) {
									stack++;
								} else if (cc == closec) {
									stack--;
								}

								if (stack == 0) {
									brace_open_match_line = i;
									brace_open_match_column = j;
									brace_open_matching = true;

									break;
								}
							}
							if (brace_open_match_line != -1) {
								break;
							}
						}

						if (!brace_open_matching) {
							brace_open_mismatch = true;
						}
					}
				}

				if (cursor.column > 0) {
					char32_t c = text[cursor.line][cursor.column - 1];
					char32_t closec = 0;

					if (c == ']') {
						closec = '[';
					} else if (c == '}') {
						closec = '{';
					} else if (c == ')') {
						closec = '(';
					}

					if (closec != 0) {
						int stack = 1;

						for (int i = cursor.line; i >= 0; i--) {
							int from = i == cursor.line ? cursor.column - 2 : text[i].length() - 1;
							for (int j = from; j >= 0; j--) {
								char32_t cc = text[i][j];
								// Ignore any brackets inside a string.
								if (cc == '"' || cc == '\'') {
									char32_t quotation = cc;
									do {
										j--;
										if (!(j >= 0)) {
											break;
										}
										cc = text[i][j];
										// Skip over escaped quotation marks inside strings.
										if (cc == quotation) {
											bool escaped = false;
											while (j - 1 >= 0 && text[i][j - 1] == '\\') {
												escaped = !escaped;
												j--;
											}
											if (escaped) {
												cc = '\\';
												continue;
											}
										}
									} while (cc != quotation);
								} else if (cc == c) {
									stack++;
								} else if (cc == closec) {
									stack--;
								}

								if (stack == 0) {
									brace_close_match_line = i;
									brace_close_match_column = j;
									brace_close_matching = true;

									break;
								}
							}
							if (brace_close_match_line != -1) {
								break;
							}
						}

						if (!brace_close_matching) {
							brace_close_mismatch = true;
						}
					}
				}
			}

			bool is_cursor_line_visible = false;
			Point2 cursor_pos;

			// Get the highlighted words.
			String highlighted_text = get_selection_text();

			// Check if highlighted words contain only whitespaces (tabs or spaces).
			bool only_whitespaces_highlighted = highlighted_text.strip_edges() == String();

			int cursor_wrap_index = get_cursor_wrap_index();

			//FontDrawer drawer(cache.font, Color(1, 1, 1));

			int first_visible_line = get_first_visible_line() - 1;
			int draw_amount = visible_rows + (smooth_scroll_enabled ? 1 : 0);
			draw_amount += times_line_wraps(first_visible_line + 1);

			// minimap
			if (draw_minimap) {
				int minimap_visible_lines = _get_minimap_visible_rows();
				int minimap_line_height = (minimap_char_size.y + minimap_line_spacing);
				int minimap_tab_size = minimap_char_size.x * indent_size;

				// calculate viewport size and y offset
				int viewport_height = (draw_amount - 1) * minimap_line_height;
				int control_height = _get_control_height() - viewport_height;
				int viewport_offset_y = round(get_scroll_pos_for_line(first_visible_line + 1) * control_height) / ((v_scroll->get_max() <= minimap_visible_lines) ? (minimap_visible_lines - draw_amount) : (v_scroll->get_max() - draw_amount));

				// calculate the first line.
				int num_lines_before = round((viewport_offset_y) / minimap_line_height);
				int wi;
				int minimap_line = (v_scroll->get_max() <= minimap_visible_lines) ? -1 : first_visible_line;
				if (minimap_line >= 0) {
					minimap_line -= num_lines_from_rows(first_visible_line, 0, -num_lines_before, wi);
					minimap_line -= (minimap_line > 0 && smooth_scroll_enabled ? 1 : 0);
				}
				int minimap_draw_amount = minimap_visible_lines + times_line_wraps(minimap_line + 1);

				// draw the minimap
				Color viewport_color = (cache.background_color.get_v() < 0.5) ? Color(1, 1, 1, 0.1) : Color(0, 0, 0, 0.1);
				if (rtl) {
					RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(size.width - (xmargin_end + 2) - cache.minimap_width, viewport_offset_y, cache.minimap_width, viewport_height), viewport_color);
				} else {
					RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2((xmargin_end + 2), viewport_offset_y, cache.minimap_width, viewport_height), viewport_color);
				}
				for (int i = 0; i < minimap_draw_amount; i++) {
					minimap_line++;

					if (minimap_line < 0 || minimap_line >= (int)text.size()) {
						break;
					}

					while (is_line_hidden(minimap_line)) {
						minimap_line++;
						if (minimap_line < 0 || minimap_line >= (int)text.size()) {
							break;
						}
					}

					if (minimap_line < 0 || minimap_line >= (int)text.size()) {
						break;
					}

					Dictionary color_map = _get_line_syntax_highlighting(minimap_line);

					Color line_background_color = text.get_line_background_color(minimap_line);
					line_background_color.a *= 0.6;
					Color current_color = cache.font_color;
					if (readonly) {
						current_color = cache.font_readonly_color;
					}

					Vector<String> wrap_rows = get_wrap_rows_text(minimap_line);
					int line_wrap_amount = times_line_wraps(minimap_line);
					int last_wrap_column = 0;

					for (int line_wrap_index = 0; line_wrap_index < line_wrap_amount + 1; line_wrap_index++) {
						if (line_wrap_index != 0) {
							i++;
							if (i >= minimap_draw_amount) {
								break;
							}
						}

						const String &str = wrap_rows[line_wrap_index];
						int indent_px = line_wrap_index != 0 ? get_indent_level(minimap_line) : 0;
						if (indent_px >= wrap_at) {
							indent_px = 0;
						}
						indent_px = minimap_char_size.x * indent_px;

						if (line_wrap_index > 0) {
							last_wrap_column += wrap_rows[line_wrap_index - 1].length();
						}

						if (minimap_line == cursor.line && cursor_wrap_index == line_wrap_index && highlight_current_line) {
							if (rtl) {
								RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(size.width - (xmargin_end + 2) - cache.minimap_width, i * 3, cache.minimap_width, 2), cache.current_line_color);
							} else {
								RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2((xmargin_end + 2), i * 3, cache.minimap_width, 2), cache.current_line_color);
							}
						} else if (line_background_color != Color(0, 0, 0, 0)) {
							if (rtl) {
								RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(size.width - (xmargin_end + 2) - cache.minimap_width, i * 3, cache.minimap_width, 2), line_background_color);
							} else {
								RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2((xmargin_end + 2), i * 3, cache.minimap_width, 2), line_background_color);
							}
						}

						Color previous_color;
						int characters = 0;
						int tabs = 0;
						for (int j = 0; j < str.length(); j++) {
							if (color_map.has(last_wrap_column + j)) {
								current_color = color_map[last_wrap_column + j].get("color");
								if (readonly) {
									current_color.a = cache.font_readonly_color.a;
								}
							}
							color = current_color;

							if (j == 0) {
								previous_color = color;
							}

							int xpos = indent_px + ((xmargin_end + minimap_char_size.x) + (minimap_char_size.x * j)) + tabs;
							bool out_of_bounds = (xpos >= xmargin_end + cache.minimap_width);

							bool is_whitespace = _is_whitespace(str[j]);
							if (!is_whitespace) {
								characters++;

								if (j < str.length() - 1 && color == previous_color && !out_of_bounds) {
									continue;
								}

								// If we've changed colour we are at the start of a new section, therefore we need to go back to the end
								// of the previous section to draw it, we'll also add the character back on.
								if (color != previous_color) {
									characters--;
									j--;

									if (str[j] == '\t') {
										tabs -= minimap_tab_size;
									}
								}
							}

							if (characters > 0) {
								previous_color.a *= 0.6;
								// take one for zero indexing, and if we hit whitespace / the end of a word.
								int chars = MAX(0, (j - (characters - 1)) - (is_whitespace ? 1 : 0)) + 1;
								int char_x_ofs = indent_px + ((xmargin_end + minimap_char_size.x) + (minimap_char_size.x * chars)) + tabs;
								if (rtl) {
									RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(Point2(size.width - char_x_ofs - minimap_char_size.x * characters, minimap_line_height * i), Point2(minimap_char_size.x * characters, minimap_char_size.y)), previous_color);
								} else {
									RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(Point2(char_x_ofs, minimap_line_height * i), Point2(minimap_char_size.x * characters, minimap_char_size.y)), previous_color);
								}
							}

							if (out_of_bounds) {
								break;
							}

							if (str[j] == '\t') {
								tabs += minimap_tab_size;
							}

							previous_color = color;
							characters = 0;
						}
					}
				}
			}

			int top_limit_y = 0;
			int bottom_limit_y = get_size().height;
			if (readonly) {
				top_limit_y += cache.style_readonly->get_margin(SIDE_TOP);
				bottom_limit_y -= cache.style_readonly->get_margin(SIDE_BOTTOM);
			} else {
				top_limit_y += cache.style_normal->get_margin(SIDE_TOP);
				bottom_limit_y -= cache.style_normal->get_margin(SIDE_BOTTOM);
			}

			// draw main text
			int row_height = get_row_height();
			int line = first_visible_line;
			for (int i = 0; i < draw_amount; i++) {
				line++;

				if (line < 0 || line >= (int)text.size()) {
					continue;
				}

				while (is_line_hidden(line)) {
					line++;
					if (line < 0 || line >= (int)text.size()) {
						break;
					}
				}

				if (line < 0 || line >= (int)text.size()) {
					continue;
				}

				Dictionary color_map = _get_line_syntax_highlighting(line);

				// Ensure we at least use the font color.
				Color current_color = readonly ? cache.font_readonly_color : cache.font_color;

				const Ref<TextParagraph> ldata = text.get_line_data(line);

				Vector<String> wrap_rows = get_wrap_rows_text(line);
				int line_wrap_amount = times_line_wraps(line);

				for (int line_wrap_index = 0; line_wrap_index <= line_wrap_amount; line_wrap_index++) {
					if (line_wrap_index != 0) {
						i++;
						if (i >= draw_amount) {
							break;
						}
					}

					const String &str = wrap_rows[line_wrap_index];
					int char_margin = xmargin_beg - cursor.x_ofs;

					int ofs_x = 0;
					int ofs_y = 0;
					if (readonly) {
						ofs_x = cache.style_readonly->get_offset().x / 2;
						ofs_x -= cache.style_normal->get_offset().x / 2;
						ofs_y = cache.style_readonly->get_offset().y / 2;
					} else {
						ofs_y = cache.style_normal->get_offset().y / 2;
					}

					ofs_y += i * row_height + cache.line_spacing / 2;
					ofs_y -= cursor.wrap_ofs * row_height;
					ofs_y -= get_v_scroll_offset() * row_height;

					bool clipped = false;
					if (ofs_y + row_height < top_limit_y) {
						// Line is outside the top margin, clip current line.
						// Still need to go through the process to prepare color changes for next lines.
						clipped = true;
					}

					if (ofs_y > bottom_limit_y) {
						// Line is outside the bottom margin, clip any remaining text.
						i = draw_amount;
						break;
					}

					if (text.get_line_background_color(line) != Color(0, 0, 0, 0)) {
						if (rtl) {
							RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(size.width - ofs_x - xmargin_end, ofs_y, xmargin_end - xmargin_beg, row_height), text.get_line_background_color(line));
						} else {
							RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(xmargin_beg + ofs_x, ofs_y, xmargin_end - xmargin_beg, row_height), text.get_line_background_color(line));
						}
					}

					if (str.length() == 0) {
						// Draw line background if empty as we won't loop at all.
						if (line == cursor.line && cursor_wrap_index == line_wrap_index && highlight_current_line) {
							if (rtl) {
								RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(size.width - ofs_x - xmargin_end, ofs_y, xmargin_end, row_height), cache.current_line_color);
							} else {
								RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(ofs_x, ofs_y, xmargin_end, row_height), cache.current_line_color);
							}
						}

						// Give visual indication of empty selected line.
						if (selection.active && line >= selection.from_line && line <= selection.to_line && char_margin >= xmargin_beg) {
							int char_w = cache.font->get_char_size(' ', 0, cache.font_size).width;
							if (rtl) {
								RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(size.width - xmargin_beg - ofs_x - char_w, ofs_y, char_w, row_height), cache.selection_color);
							} else {
								RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(xmargin_beg + ofs_x, ofs_y, char_w, row_height), cache.selection_color);
							}
						}
					} else {
						// If it has text, then draw current line marker in the margin, as line number etc will draw over it, draw the rest of line marker later.
						if (line == cursor.line && cursor_wrap_index == line_wrap_index && highlight_current_line) {
							if (rtl) {
								RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(size.width - ofs_x - xmargin_end, ofs_y, xmargin_end, row_height), cache.current_line_color);
							} else {
								RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(ofs_x, ofs_y, xmargin_end, row_height), cache.current_line_color);
							}
						}
					}

					if (line_wrap_index == 0) {
						// Only do these if we are on the first wrapped part of a line.

						int gutter_offset = cache.style_normal->get_margin(SIDE_LEFT);
						for (int g = 0; g < gutters.size(); g++) {
							const GutterInfo gutter = gutters[g];

							if (!gutter.draw || gutter.width <= 0) {
								continue;
							}

							switch (gutter.type) {
								case GUTTER_TYPE_STRING: {
									const String &text = get_line_gutter_text(line, g);
									if (text == "") {
										break;
									}

									Ref<TextLine> tl;
									tl.instance();
									tl->add_string(text, cache.font, cache.font_size);

									int yofs = ofs_y + (row_height - tl->get_size().y) / 2;
									if (cache.outline_size > 0 && cache.outline_color.a > 0) {
										tl->draw_outline(ci, Point2(gutter_offset + ofs_x, yofs), cache.outline_size, cache.outline_color);
									}
									tl->draw(ci, Point2(gutter_offset + ofs_x, yofs), get_line_gutter_item_color(line, g));
								} break;
								case GUTTER_TPYE_ICON: {
									const Ref<Texture2D> icon = get_line_gutter_icon(line, g);
									if (icon.is_null()) {
										break;
									}

									Rect2 gutter_rect = Rect2(Point2i(gutter_offset, ofs_y), Size2i(gutter.width, row_height));

									int horizontal_padding = gutter_rect.size.x / 6;
									int vertical_padding = gutter_rect.size.y / 6;

									gutter_rect.position += Point2(horizontal_padding, vertical_padding);
									gutter_rect.size -= Point2(horizontal_padding, vertical_padding) * 2;

									// Correct icon aspect ratio.
									float icon_ratio = icon->get_width() / icon->get_height();
									float gutter_ratio = gutter_rect.size.x / gutter_rect.size.y;
									if (gutter_ratio > icon_ratio) {
										gutter_rect.size.x = floor(icon->get_width() * (gutter_rect.size.y / icon->get_height()));
									} else {
										gutter_rect.size.y = floor(icon->get_height() * (gutter_rect.size.x / icon->get_width()));
									}
									if (rtl) {
										gutter_rect.position.x = size.width - gutter_rect.position.x - gutter_rect.size.x;
									}

									icon->draw_rect(ci, gutter_rect, false, get_line_gutter_item_color(line, g));
								} break;
								case GUTTER_TPYE_CUSTOM: {
									if (gutter.custom_draw_obj.is_valid()) {
										Object *cdo = ObjectDB::get_instance(gutter.custom_draw_obj);
										if (cdo) {
											Rect2i gutter_rect = Rect2i(Point2i(gutter_offset, ofs_y), Size2i(gutter.width, row_height));
											if (rtl) {
												gutter_rect.position.x = size.width - gutter_rect.position.x - gutter_rect.size.x;
											}
											cdo->call(gutter.custom_draw_callback, line, g, Rect2(gutter_rect));
										}
									}
								} break;
							}

							gutter_offset += gutter.width;
						}
					}

					// Draw line.
					RID rid = ldata->get_line_rid(line_wrap_index);
					float text_height = TS->shaped_text_get_size(rid).y + cache.font->get_spacing(Font::SPACING_TOP) + cache.font->get_spacing(Font::SPACING_BOTTOM);

					if (rtl) {
						char_margin = size.width - char_margin - TS->shaped_text_get_size(rid).x;
					}

					if (!clipped && selection.active && line >= selection.from_line && line <= selection.to_line) { // Selection
						int sel_from = (line > selection.from_line) ? TS->shaped_text_get_range(rid).x : selection.from_column;
						int sel_to = (line < selection.to_line) ? TS->shaped_text_get_range(rid).y : selection.to_column;
						Vector<Vector2> sel = TS->shaped_text_get_selection(rid, sel_from, sel_to);
						for (int j = 0; j < sel.size(); j++) {
							Rect2 rect = Rect2(sel[j].x + char_margin + ofs_x, ofs_y, sel[j].y - sel[j].x, row_height);
							if (rect.position.x + rect.size.x <= xmargin_beg || rect.position.x > xmargin_end) {
								continue;
							}
							if (rect.position.x < xmargin_beg) {
								rect.size.x -= (xmargin_beg - rect.position.x);
								rect.position.x = xmargin_beg;
							} else if (rect.position.x + rect.size.x > xmargin_end) {
								rect.size.x = xmargin_end - rect.position.x;
							}
							draw_rect(rect, cache.selection_color, true);
						}
					}

					int start = TS->shaped_text_get_range(rid).x;
					if (!clipped && !search_text.is_empty()) { // Search highhlight
						int search_text_col = _get_column_pos_of_word(search_text, str, search_flags, 0);
						while (search_text_col != -1) {
							Vector<Vector2> sel = TS->shaped_text_get_selection(rid, search_text_col + start, search_text_col + search_text.length() + start);
							for (int j = 0; j < sel.size(); j++) {
								Rect2 rect = Rect2(sel[j].x + char_margin + ofs_x, ofs_y, sel[j].y - sel[j].x, row_height);
								if (rect.position.x + rect.size.x <= xmargin_beg || rect.position.x > xmargin_end) {
									continue;
								}
								if (rect.position.x < xmargin_beg) {
									rect.size.x -= (xmargin_beg - rect.position.x);
									rect.position.x = xmargin_beg;
								} else if (rect.position.x + rect.size.x > xmargin_end) {
									rect.size.x = xmargin_end - rect.position.x;
								}
								draw_rect(rect, cache.search_result_color, true);
								draw_rect(rect, cache.search_result_border_color, false);
							}

							search_text_col = _get_column_pos_of_word(search_text, str, search_flags, search_text_col + 1);
						}
					}

					if (!clipped && highlight_all_occurrences && !only_whitespaces_highlighted && !highlighted_text.is_empty()) { // Highlight
						int highlighted_text_col = _get_column_pos_of_word(highlighted_text, str, SEARCH_MATCH_CASE | SEARCH_WHOLE_WORDS, 0);
						while (highlighted_text_col != -1) {
							Vector<Vector2> sel = TS->shaped_text_get_selection(rid, highlighted_text_col + start, highlighted_text_col + highlighted_text.length() + start);
							for (int j = 0; j < sel.size(); j++) {
								Rect2 rect = Rect2(sel[j].x + char_margin + ofs_x, ofs_y, sel[j].y - sel[j].x, row_height);
								if (rect.position.x + rect.size.x <= xmargin_beg || rect.position.x > xmargin_end) {
									continue;
								}
								if (rect.position.x < xmargin_beg) {
									rect.size.x -= (xmargin_beg - rect.position.x);
									rect.position.x = xmargin_beg;
								} else if (rect.position.x + rect.size.x > xmargin_end) {
									rect.size.x = xmargin_end - rect.position.x;
								}
								draw_rect(rect, cache.word_highlighted_color);
							}

							highlighted_text_col = _get_column_pos_of_word(highlighted_text, str, SEARCH_MATCH_CASE | SEARCH_WHOLE_WORDS, highlighted_text_col + 1);
						}
					}

					if (!clipped && select_identifiers_enabled && highlighted_word.length() != 0) { // Highlight word
						if (_is_char(highlighted_word[0]) || highlighted_word[0] == '.') {
							int highlighted_word_col = _get_column_pos_of_word(highlighted_word, str, SEARCH_MATCH_CASE | SEARCH_WHOLE_WORDS, 0);
							while (highlighted_word_col != -1) {
								Vector<Vector2> sel = TS->shaped_text_get_selection(rid, highlighted_word_col + start, highlighted_word_col + highlighted_word.length() + start);
								for (int j = 0; j < sel.size(); j++) {
									Rect2 rect = Rect2(sel[j].x + char_margin + ofs_x, ofs_y, sel[j].y - sel[j].x, row_height);
									if (rect.position.x + rect.size.x <= xmargin_beg || rect.position.x > xmargin_end) {
										continue;
									}
									if (rect.position.x < xmargin_beg) {
										rect.size.x -= (xmargin_beg - rect.position.x);
										rect.position.x = xmargin_beg;
									} else if (rect.position.x + rect.size.x > xmargin_end) {
										rect.size.x = xmargin_end - rect.position.x;
									}
									rect.position.y = TS->shaped_text_get_ascent(rid) + cache.font->get_underline_position(cache.font_size);
									rect.size.y = cache.font->get_underline_thickness(cache.font_size);
									draw_rect(rect, cache.font_selected_color);
								}

								highlighted_word_col = _get_column_pos_of_word(highlighted_word, str, SEARCH_MATCH_CASE | SEARCH_WHOLE_WORDS, highlighted_word_col + 1);
							}
						}
					}

					const int line_top_offset_y = ofs_y;
					ofs_y += (row_height - text_height) / 2;

					const Vector<TextServer::Glyph> visual = TS->shaped_text_get_glyphs(rid);
					const TextServer::Glyph *glyphs = visual.ptr();
					int gl_size = visual.size();

					ofs_y += ldata->get_line_ascent(line_wrap_index);
					int char_ofs = 0;
					if (cache.outline_size > 0 && cache.outline_color.a > 0) {
						for (int j = 0; j < gl_size; j++) {
							for (int k = 0; k < glyphs[j].repeat; k++) {
								if ((char_ofs + char_margin) >= xmargin_beg && (char_ofs + glyphs[j].advance + char_margin) <= xmargin_end) {
									if (glyphs[j].font_rid != RID()) {
										TS->font_draw_glyph_outline(glyphs[j].font_rid, ci, glyphs[j].font_size, cache.outline_size, Vector2(char_margin + char_ofs + ofs_x + glyphs[j].x_off, ofs_y + glyphs[j].y_off), glyphs[j].index, cache.outline_color);
									}
								}
								char_ofs += glyphs[j].advance;
							}
							if ((char_ofs + char_margin) >= xmargin_end) {
								break;
							}
						}
						char_ofs = 0;
					}
					for (int j = 0; j < gl_size; j++) {
						if (color_map.has(glyphs[j].start)) {
							current_color = color_map[glyphs[j].start].get("color");
							if (readonly && current_color.a > cache.font_readonly_color.a) {
								current_color.a = cache.font_readonly_color.a;
							}
						}

						if (selection.active && line >= selection.from_line && line <= selection.to_line) { // Selection
							int sel_from = (line > selection.from_line) ? TS->shaped_text_get_range(rid).x : selection.from_column;
							int sel_to = (line < selection.to_line) ? TS->shaped_text_get_range(rid).y : selection.to_column;

							if (glyphs[j].start >= sel_from && glyphs[j].end <= sel_to && override_selected_font_color) {
								current_color = cache.font_selected_color;
							}
						}

						int char_pos = char_ofs + char_margin + ofs_x;
						if (char_pos >= xmargin_beg) {
							if (brace_matching_enabled) {
								if ((brace_open_match_line == line && brace_open_match_column == glyphs[j].start) ||
										(cursor.column == glyphs[j].start && cursor.line == line && cursor_wrap_index == line_wrap_index && (brace_open_matching || brace_open_mismatch))) {
									if (brace_open_mismatch) {
										current_color = cache.brace_mismatch_color;
									}
									Rect2 rect = Rect2(char_pos, ofs_y + cache.font->get_underline_position(cache.font_size), glyphs[j].advance * glyphs[j].repeat, cache.font->get_underline_thickness(cache.font_size));
									draw_rect(rect, current_color);
								}

								if ((brace_close_match_line == line && brace_close_match_column == glyphs[j].start) ||
										(cursor.column == glyphs[j].start + 1 && cursor.line == line && cursor_wrap_index == line_wrap_index && (brace_close_matching || brace_close_mismatch))) {
									if (brace_close_mismatch) {
										current_color = cache.brace_mismatch_color;
									}
									Rect2 rect = Rect2(char_pos, ofs_y + cache.font->get_underline_position(cache.font_size), glyphs[j].advance * glyphs[j].repeat, cache.font->get_underline_thickness(cache.font_size));
									draw_rect(rect, current_color);
								}
							}

							if (draw_tabs && ((glyphs[j].flags & TextServer::GRAPHEME_IS_TAB) == TextServer::GRAPHEME_IS_TAB)) {
								int yofs = (text_height - cache.tab_icon->get_height()) / 2 - ldata->get_line_ascent(line_wrap_index);
								cache.tab_icon->draw(ci, Point2(char_pos, ofs_y + yofs), current_color);
							} else if (draw_spaces && ((glyphs[j].flags & TextServer::GRAPHEME_IS_SPACE) == TextServer::GRAPHEME_IS_SPACE)) {
								int yofs = (text_height - cache.space_icon->get_height()) / 2 - ldata->get_line_ascent(line_wrap_index);
								int xofs = (glyphs[j].advance * glyphs[j].repeat - cache.space_icon->get_width()) / 2;
								cache.space_icon->draw(ci, Point2(char_pos + xofs, ofs_y + yofs), current_color);
							}
						}

						for (int k = 0; k < glyphs[j].repeat; k++) {
							if (!clipped && (char_ofs + char_margin) >= xmargin_beg && (char_ofs + glyphs[j].advance + char_margin) <= xmargin_end) {
								if (glyphs[j].font_rid != RID()) {
									TS->font_draw_glyph(glyphs[j].font_rid, ci, glyphs[j].font_size, Vector2(char_margin + char_ofs + ofs_x + glyphs[j].x_off, ofs_y + glyphs[j].y_off), glyphs[j].index, current_color);
								} else if ((glyphs[j].flags & TextServer::GRAPHEME_IS_VIRTUAL) != TextServer::GRAPHEME_IS_VIRTUAL) {
									TS->draw_hex_code_box(ci, glyphs[j].font_size, Vector2(char_margin + char_ofs + ofs_x + glyphs[j].x_off, ofs_y + glyphs[j].y_off), glyphs[j].index, current_color);
								}
							}
							char_ofs += glyphs[j].advance;
						}
						if ((char_ofs + char_margin) >= xmargin_end) {
							break;
						}
					}

					if (line_wrap_index == line_wrap_amount && is_folded(line)) {
						int xofs = char_ofs + char_margin + ofs_x + (cache.folded_eol_icon->get_width() / 2);
						if (xofs >= xmargin_beg && xofs < xmargin_end) {
							int yofs = (text_height - cache.folded_eol_icon->get_height()) / 2 - ldata->get_line_ascent(line_wrap_index);
							Color eol_color = cache.code_folding_color;
							eol_color.a = 1;
							cache.folded_eol_icon->draw(ci, Point2(xofs, ofs_y + yofs), eol_color);
						}
					}

					// Carets
#ifdef TOOLS_ENABLED
					int caret_width = Math::round(EDSCALE);
#else
					int caret_width = 1;
#endif
					if (!clipped && cursor.line == line && ((line_wrap_index == line_wrap_amount) || (cursor.column != TS->shaped_text_get_range(rid).y))) {
						is_cursor_line_visible = true;
						cursor_pos.y = line_top_offset_y;

						if (ime_text.length() == 0) {
							Rect2 l_caret, t_caret;
							TextServer::Direction l_dir, t_dir;
							if (str.length() != 0) {
								// Get carets.
								TS->shaped_text_get_carets(rid, cursor.column, l_caret, l_dir, t_caret, t_dir);
							} else {
								// No carets, add one at the start.
								int h = cache.font->get_height(cache.font_size);
								if (rtl) {
									l_dir = TextServer::DIRECTION_RTL;
									l_caret = Rect2(Vector2(xmargin_end - char_margin + ofs_x, -h / 2), Size2(caret_width * 4, h));
								} else {
									l_dir = TextServer::DIRECTION_LTR;
									l_caret = Rect2(Vector2(char_ofs, -h / 2), Size2(caret_width * 4, h));
								}
							}

							if ((l_caret != Rect2() && (l_dir == TextServer::DIRECTION_AUTO || l_dir == (TextServer::Direction)input_direction)) || (t_caret == Rect2())) {
								cursor_pos.x = char_margin + ofs_x + l_caret.position.x;
							} else {
								cursor_pos.x = char_margin + ofs_x + t_caret.position.x;
							}

							if (draw_caret && cursor_pos.x >= xmargin_beg && cursor_pos.x < xmargin_end) {
								if (block_caret || insert_mode) {
									//Block or underline caret, draw trailing carets at full height.
									int h = cache.font->get_height(cache.font_size);

									if (t_caret != Rect2()) {
										if (insert_mode) {
											t_caret.position.y = TS->shaped_text_get_descent(rid);
											t_caret.size.y = caret_width;
										} else {
											t_caret.position.y = -TS->shaped_text_get_ascent(rid);
											t_caret.size.y = h;
										}
										t_caret.position += Vector2(char_margin + ofs_x, ofs_y);

										draw_rect(t_caret, cache.caret_color, false);
									} else { // End of the line.
										if (insert_mode) {
											l_caret.position.y = TS->shaped_text_get_descent(rid);
											l_caret.size.y = caret_width;
										} else {
											l_caret.position.y = -TS->shaped_text_get_ascent(rid);
											l_caret.size.y = h;
										}
										l_caret.position += Vector2(char_margin + ofs_x, ofs_y);
										l_caret.size.x = cache.font->get_char_size('M', 0, cache.font_size).x;

										draw_rect(l_caret, cache.caret_color, false);
									}
								} else {
									// Normal caret.
									if (l_caret != Rect2() && l_dir == TextServer::DIRECTION_AUTO) {
										// Draw extra marker on top of mid caret.
										Rect2 trect = Rect2(l_caret.position.x - 3 * caret_width, l_caret.position.y, 6 * caret_width, caret_width);
										trect.position += Vector2(char_margin + ofs_x, ofs_y);
										RenderingServer::get_singleton()->canvas_item_add_rect(ci, trect, cache.caret_color);
									}
									l_caret.position += Vector2(char_margin + ofs_x, ofs_y);
									l_caret.size.x = caret_width;

									draw_rect(l_caret, cache.caret_color);

									t_caret.position += Vector2(char_margin + ofs_x, ofs_y);
									t_caret.size.x = caret_width;

									draw_rect(t_caret, cache.caret_color);
								}
							}
						} else {
							{
								// IME Intermediate text range.
								Vector<Vector2> sel = TS->shaped_text_get_selection(rid, cursor.column, cursor.column + ime_text.length());
								for (int j = 0; j < sel.size(); j++) {
									Rect2 rect = Rect2(sel[j].x + char_margin + ofs_x, ofs_y, sel[j].y - sel[j].x, text_height);
									if (rect.position.x + rect.size.x <= xmargin_beg || rect.position.x > xmargin_end) {
										continue;
									}
									if (rect.position.x < xmargin_beg) {
										rect.size.x -= (xmargin_beg - rect.position.x);
										rect.position.x = xmargin_beg;
									} else if (rect.position.x + rect.size.x > xmargin_end) {
										rect.size.x = xmargin_end - rect.position.x;
									}
									rect.size.y = caret_width;
									draw_rect(rect, cache.caret_color);
									cursor_pos.x = rect.position.x;
								}
							}
							{
								// IME caret.
								Vector<Vector2> sel = TS->shaped_text_get_selection(rid, cursor.column + ime_selection.x, cursor.column + ime_selection.x + ime_selection.y);
								for (int j = 0; j < sel.size(); j++) {
									Rect2 rect = Rect2(sel[j].x + char_margin + ofs_x, ofs_y, sel[j].y - sel[j].x, text_height);
									if (rect.position.x + rect.size.x <= xmargin_beg || rect.position.x > xmargin_end) {
										continue;
									}
									if (rect.position.x < xmargin_beg) {
										rect.size.x -= (xmargin_beg - rect.position.x);
										rect.position.x = xmargin_beg;
									} else if (rect.position.x + rect.size.x > xmargin_end) {
										rect.size.x = xmargin_end - rect.position.x;
									}
									rect.size.y = caret_width * 3;
									draw_rect(rect, cache.caret_color);
									cursor_pos.x = rect.position.x;
								}
							}
						}
					}
				}
			}

			bool completion_below = false;
			if (completion_active && is_cursor_line_visible && completion_options.size() > 0) {
				// Completion panel

				const Ref<StyleBox> csb = get_theme_stylebox("completion");
				const int maxlines = get_theme_constant("completion_lines");
				const int cmax_width = get_theme_constant("completion_max_width") * cache.font->get_char_size('x', 0, cache.font_size).x;
				const Color scrollc = get_theme_color("completion_scroll_color");

				const int completion_options_size = completion_options.size();
				const int row_count = MIN(completion_options_size, maxlines);
				const int completion_rows_height = row_count * row_height;
				const int completion_base_width = cache.font->get_string_size(completion_base, cache.font_size).width;

				int scroll_rectangle_width = get_theme_constant("completion_scroll_width");
				int width = 0;

				// Compute max width of the panel based on the longest completion option
				if (completion_options_size < 50) {
					for (int i = 0; i < completion_options_size; i++) {
						int line_width = MIN(cache.font->get_string_size(completion_options[i].display, cache.font_size).x, cmax_width);
						if (line_width > width) {
							width = line_width;
						}
					}
				} else {
					width = cmax_width;
				}

				// Add space for completion icons.
				const int icon_hsep = get_theme_constant("hseparation", "ItemList");
				const Size2 icon_area_size(row_height, row_height);
				const int icon_area_width = icon_area_size.width + icon_hsep;
				width += icon_area_width;

				const int line_from = CLAMP(completion_index - row_count / 2, 0, completion_options_size - row_count);

				for (int i = 0; i < row_count; i++) {
					int l = line_from + i;
					ERR_CONTINUE(l < 0 || l >= completion_options_size);
					if (completion_options[l].default_value.get_type() == Variant::COLOR) {
						width += icon_area_size.width;
						break;
					}
				}

				// Position completion panel
				completion_rect.size.width = width + 2;
				completion_rect.size.height = completion_rows_height;

				if (completion_options_size <= maxlines) {
					scroll_rectangle_width = 0;
				}

				const Point2 csb_offset = csb->get_offset();

				const int total_width = completion_rect.size.width + csb->get_minimum_size().x + scroll_rectangle_width;
				const int total_height = completion_rect.size.height + csb->get_minimum_size().y;

				const int rect_left_border_x = cursor_pos.x - completion_base_width - icon_area_width - csb_offset.x;
				const int rect_right_border_x = rect_left_border_x + total_width;

				if (rect_left_border_x < 0) {
					// Anchor the completion panel to the left
					completion_rect.position.x = 0;
				} else if (rect_right_border_x > get_size().width) {
					// Anchor the completion panel to the right
					completion_rect.position.x = get_size().width - total_width;
				} else {
					// Let the completion panel float with the cursor
					completion_rect.position.x = rect_left_border_x;
				}

				if (cursor_pos.y + row_height + total_height > get_size().height && cursor_pos.y > total_height) {
					// Completion panel above the cursor line
					completion_rect.position.y = cursor_pos.y - total_height;
				} else {
					// Completion panel below the cursor line
					completion_rect.position.y = cursor_pos.y + row_height;
					completion_below = true;
				}

				draw_style_box(csb, Rect2(completion_rect.position - csb_offset, completion_rect.size + csb->get_minimum_size() + Size2(scroll_rectangle_width, 0)));

				if (cache.completion_background_color.a > 0.01) {
					RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(completion_rect.position, completion_rect.size + Size2(scroll_rectangle_width, 0)), cache.completion_background_color);
				}
				RenderingServer::get_singleton()->canvas_item_add_rect(ci, Rect2(Point2(completion_rect.position.x, completion_rect.position.y + (completion_index - line_from) * get_row_height()), Size2(completion_rect.size.width, get_row_height())), cache.completion_selected_color);

				draw_rect(Rect2(completion_rect.position + Vector2(icon_area_size.x + icon_hsep, 0), Size2(MIN(completion_base_width, completion_rect.size.width - (icon_area_size.x + icon_hsep)), completion_rect.size.height)), cache.completion_existing_color);

				for (int i = 0; i < row_count; i++) {
					int l = line_from + i;
					ERR_CONTINUE(l < 0 || l >= completion_options_size);

					Ref<TextLine> tl;
					tl.instance();
					tl->add_string(completion_options[l].display, cache.font, cache.font_size);

					int yofs = (row_height - tl->get_size().y) / 2;
					Point2 title_pos(completion_rect.position.x, completion_rect.position.y + i * row_height + yofs);

					// Draw completion icon if it is valid.
					Ref<Texture2D> icon = completion_options[l].icon;
					Rect2 icon_area(completion_rect.position.x, completion_rect.position.y + i * row_height, icon_area_size.width, icon_area_size.height);
					if (icon.is_valid()) {
						const real_t max_scale = 0.7f;
						const real_t side = max_scale * icon_area.size.width;
						real_t scale = MIN(side / icon->get_width(), side / icon->get_height());
						Size2 icon_size = icon->get_size() * scale;
						draw_texture_rect(icon, Rect2(icon_area.position + (icon_area.size - icon_size) / 2, icon_size));
					}

					title_pos.x = icon_area.position.x + icon_area.size.width + icon_hsep;

					tl->set_width(completion_rect.size.width - (icon_area_size.x + icon_hsep));

					if (rtl) {
						if (completion_options[l].default_value.get_type() == Variant::COLOR) {
							draw_rect(Rect2(Point2(completion_rect.position.x, icon_area.position.y), icon_area_size), (Color)completion_options[l].default_value);
						}
						tl->set_align(HALIGN_RIGHT);
					} else {
						if (completion_options[l].default_value.get_type() == Variant::COLOR) {
							draw_rect(Rect2(Point2(completion_rect.position.x + completion_rect.size.width - icon_area_size.x, icon_area.position.y), icon_area_size), (Color)completion_options[l].default_value);
						}
						tl->set_align(HALIGN_LEFT);
					}
					if (cache.outline_size > 0 && cache.outline_color.a > 0) {
						tl->draw_outline(ci, title_pos, cache.outline_size, cache.outline_color);
					}
					tl->draw(ci, title_pos, completion_options[l].font_color);
				}

				if (scroll_rectangle_width) {
					// Draw a small scroll rectangle to show a position in the options.
					float r = (float)maxlines / completion_options_size;
					float o = (float)line_from / completion_options_size;
					draw_rect(Rect2(completion_rect.position.x + completion_rect.size.width, completion_rect.position.y + o * completion_rect.size.y, scroll_rectangle_width, completion_rect.size.y * r), scrollc);
				}

				completion_line_ofs = line_from;
			}

			// Check to see if the hint should be drawn.
			bool show_hint = false;
			if (is_cursor_line_visible && completion_hint != "") {
				if (completion_active) {
					if (completion_below && !callhint_below) {
						show_hint = true;
					} else if (!completion_below && callhint_below) {
						show_hint = true;
					}
				} else {
					show_hint = true;
				}
			}

			if (show_hint) {
				Ref<StyleBox> sb = get_theme_stylebox("panel", "TooltipPanel");
				Ref<Font> font = cache.font;
				Color font_color = get_theme_color("font_color", "TooltipLabel");

				int max_w = 0;
				int sc = completion_hint.get_slice_count("\n");
				int offset = 0;
				int spacing = 0;
				for (int i = 0; i < sc; i++) {
					String l = completion_hint.get_slice("\n", i);
					int len = font->get_string_size(l, cache.font_size).x;
					max_w = MAX(len, max_w);
					if (i == 0) {
						offset = font->get_string_size(l.substr(0, l.find(String::chr(0xFFFF))), cache.font_size).x;
					} else {
						spacing += cache.line_spacing;
					}
				}

				Size2 size2 = Size2(max_w, sc * font->get_height(cache.font_size) + spacing);
				Size2 minsize = size2 + sb->get_minimum_size();

				if (completion_hint_offset == -0xFFFF) {
					completion_hint_offset = cursor_pos.x - offset;
				}

				Point2 hint_ofs = Vector2(completion_hint_offset, cursor_pos.y) + callhint_offset;

				if (callhint_below) {
					hint_ofs.y += row_height + sb->get_offset().y;
				} else {
					hint_ofs.y -= minsize.y + sb->get_offset().y;
				}

				draw_style_box(sb, Rect2(hint_ofs, minsize));

				spacing = 0;
				for (int i = 0; i < sc; i++) {
					int begin = 0;
					int end = 0;
					String l = completion_hint.get_slice("\n", i);

					if (l.find(String::chr(0xFFFF)) != -1) {
						begin = font->get_string_size(l.substr(0, l.find(String::chr(0xFFFF))), cache.font_size).x;
						end = font->get_string_size(l.substr(0, l.rfind(String::chr(0xFFFF))), cache.font_size).x;
					}

					Point2 round_ofs = hint_ofs + sb->get_offset() + Vector2(0, font->get_ascent(cache.font_size) + font->get_height(cache.font_size) * i + spacing);
					round_ofs = round_ofs.round();
					draw_string(font, round_ofs, l.replace(String::chr(0xFFFF), ""), HALIGN_LEFT, -1, cache.font_size, font_color);
					if (end > 0) {
						Vector2 b = hint_ofs + sb->get_offset() + Vector2(begin, font->get_height(cache.font_size) + font->get_height(cache.font_size) * i + spacing - 1);
						draw_line(b, b + Vector2(end - begin, 0), font_color);
					}
					spacing += cache.line_spacing;
				}
			}

			if (has_focus()) {
				if (get_viewport()->get_window_id() != DisplayServer::INVALID_WINDOW_ID && DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_IME)) {
					DisplayServer::get_singleton()->window_set_ime_active(true, get_viewport()->get_window_id());
					DisplayServer::get_singleton()->window_set_ime_position(get_global_position() + cursor_pos, get_viewport()->get_window_id());
				}
			}
		} break;
		case NOTIFICATION_FOCUS_ENTER: {
			if (caret_blink_enabled) {
				caret_blink_timer->start();
			} else {
				draw_caret = true;
			}

			if (get_viewport()->get_window_id() != DisplayServer::INVALID_WINDOW_ID && DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_IME)) {
				DisplayServer::get_singleton()->window_set_ime_active(true, get_viewport()->get_window_id());
				DisplayServer::get_singleton()->window_set_ime_position(get_global_position() + _get_cursor_pixel_pos(false), get_viewport()->get_window_id());
			}

			if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_VIRTUAL_KEYBOARD) && virtual_keyboard_enabled) {
				int cursor_start = -1;
				int cursor_end = -1;

				if (!selection.active) {
					String full_text = _base_get_text(0, 0, cursor.line, cursor.column);

					cursor_start = full_text.length();
				} else {
					String pre_text = _base_get_text(0, 0, selection.from_line, selection.from_column);
					String post_text = _base_get_text(selection.from_line, selection.from_column, selection.to_line, selection.to_column);

					cursor_start = pre_text.length();
					cursor_end = cursor_start + post_text.length();
				}

				DisplayServer::get_singleton()->virtual_keyboard_show(get_text(), get_global_rect(), true, -1, cursor_start, cursor_end);
			}
		} break;
		case NOTIFICATION_FOCUS_EXIT: {
			if (caret_blink_enabled) {
				caret_blink_timer->stop();
			}

			if (get_viewport()->get_window_id() != DisplayServer::INVALID_WINDOW_ID && DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_IME)) {
				DisplayServer::get_singleton()->window_set_ime_position(Point2(), get_viewport()->get_window_id());
				DisplayServer::get_singleton()->window_set_ime_active(false, get_viewport()->get_window_id());
			}
			ime_text = "";
			ime_selection = Point2();
			text.invalidate_cache(cursor.line, cursor.column, ime_text);

			if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_VIRTUAL_KEYBOARD) && virtual_keyboard_enabled) {
				DisplayServer::get_singleton()->virtual_keyboard_hide();
			}
		} break;
		case MainLoop::NOTIFICATION_OS_IME_UPDATE: {
			if (has_focus()) {
				ime_text = DisplayServer::get_singleton()->ime_get_text();
				ime_selection = DisplayServer::get_singleton()->ime_get_selection();

				String t;
				if (cursor.column >= 0) {
					t = text[cursor.line].substr(0, cursor.column) + ime_text + text[cursor.line].substr(cursor.column, text[cursor.line].length());
				} else {
					t = ime_text;
				}

				text.invalidate_cache(cursor.line, cursor.column, t, structured_text_parser(st_parser, st_args, t));
				update();
			}
		} break;
	}
}

void TextEdit::_consume_pair_symbol(char32_t ch) {
	int cursor_position_to_move = cursor_get_column() + 1;

	char32_t ch_single[2] = { ch, 0 };
	char32_t ch_single_pair[2] = { _get_right_pair_symbol(ch), 0 };
	char32_t ch_pair[3] = { ch, _get_right_pair_symbol(ch), 0 };

	if (is_selection_active()) {
		int new_column, new_line;

		begin_complex_operation();
		_insert_text(get_selection_from_line(), get_selection_from_column(),
				ch_single,
				&new_line, &new_column);

		int to_col_offset = 0;
		if (get_selection_from_line() == get_selection_to_line()) {
			to_col_offset = 1;
		}

		_insert_text(get_selection_to_line(),
				get_selection_to_column() + to_col_offset,
				ch_single_pair,
				&new_line, &new_column);
		end_complex_operation();

		cursor_set_line(get_selection_to_line());
		cursor_set_column(get_selection_to_column() + to_col_offset);

		deselect();
		update();
		return;
	}

	if ((ch == '\'' || ch == '"') &&
			cursor_get_column() > 0 && _is_text_char(text[cursor.line][cursor_get_column() - 1]) && !_is_pair_right_symbol(text[cursor.line][cursor_get_column()])) {
		insert_text_at_cursor(ch_single);
		cursor_set_column(cursor_position_to_move);
		return;
	}

	if (cursor_get_column() < text[cursor.line].length()) {
		if (_is_text_char(text[cursor.line][cursor_get_column()])) {
			insert_text_at_cursor(ch_single);
			cursor_set_column(cursor_position_to_move);
			return;
		}
		if (_is_pair_right_symbol(ch) &&
				text[cursor.line][cursor_get_column()] == ch) {
			cursor_set_column(cursor_position_to_move);
			return;
		}
	}

	String line = text[cursor.line];

	bool in_single_quote = false;
	bool in_double_quote = false;
	bool found_comment = false;

	int c = 0;
	while (c < line.length()) {
		if (line[c] == '\\') {
			c++; // Skip quoted anything.

			if (cursor.column == c) {
				break;
			}
		} else if (!in_single_quote && !in_double_quote && line[c] == '#') {
			found_comment = true;
			break;
		} else {
			if (line[c] == '\'' && !in_double_quote) {
				in_single_quote = !in_single_quote;
			} else if (line[c] == '"' && !in_single_quote) {
				in_double_quote = !in_double_quote;
			}
		}

		c++;

		if (cursor.column == c) {
			break;
		}
	}

	// Do not need to duplicate quotes while in comments
	if (found_comment) {
		insert_text_at_cursor(ch_single);
		cursor_set_column(cursor_position_to_move);

		return;
	}

	// Disallow inserting duplicated quotes while already in string
	if ((in_single_quote || in_double_quote) && (ch == '"' || ch == '\'')) {
		insert_text_at_cursor(ch_single);
		cursor_set_column(cursor_position_to_move);

		return;
	}

	insert_text_at_cursor(ch_pair);
	cursor_set_column(cursor_position_to_move);
}

void TextEdit::_consume_backspace_for_pair_symbol(int prev_line, int prev_column) {
	bool remove_right_symbol = false;

	if (cursor.column < text[cursor.line].length() && cursor.column > 0) {
		char32_t left_char = text[cursor.line][cursor.column - 1];
		char32_t right_char = text[cursor.line][cursor.column];

		if (right_char == _get_right_pair_symbol(left_char)) {
			remove_right_symbol = true;
		}
	}
	if (remove_right_symbol) {
		_remove_text(prev_line, prev_column, cursor.line, cursor.column + 1);
	} else {
		_remove_text(prev_line, prev_column, cursor.line, cursor.column);
	}
}

void TextEdit::backspace_at_cursor() {
	if (readonly) {
		return;
	}

	if (cursor.column == 0 && cursor.line == 0) {
		return;
	}

	int prev_line = cursor.column ? cursor.line : cursor.line - 1;
	int prev_column = cursor.column ? (cursor.column - 1) : (text[cursor.line - 1].length());

	if (cursor.line != prev_line) {
		for (int i = 0; i < gutters.size(); i++) {
			if (!gutters[i].overwritable) {
				continue;
			}

			if (text.get_line_gutter_text(cursor.line, i) != "") {
				text.set_line_gutter_text(prev_line, i, text.get_line_gutter_text(cursor.line, i));
				text.set_line_gutter_item_color(prev_line, i, text.get_line_gutter_item_color(cursor.line, i));
			}

			if (text.get_line_gutter_icon(cursor.line, i).is_valid()) {
				text.set_line_gutter_icon(prev_line, i, text.get_line_gutter_icon(cursor.line, i));
				text.set_line_gutter_item_color(prev_line, i, text.get_line_gutter_item_color(cursor.line, i));
			}

			if (text.get_line_gutter_metadata(cursor.line, i) != "") {
				text.set_line_gutter_metadata(prev_line, i, text.get_line_gutter_metadata(cursor.line, i));
			}

			if (text.is_line_gutter_clickable(cursor.line, i)) {
				text.set_line_gutter_clickable(prev_line, i, true);
			}
		}
	}

	if (is_line_hidden(cursor.line)) {
		set_line_as_hidden(prev_line, true);
	}

	if (auto_brace_completion_enabled &&
			cursor.column > 0 &&
			_is_pair_left_symbol(text[cursor.line][cursor.column - 1])) {
		_consume_backspace_for_pair_symbol(prev_line, prev_column);
	} else {
		// Handle space indentation.
		if (cursor.column != 0 && indent_using_spaces) {
			// Check if there are no other chars before cursor, just indentation.
			bool unindent = true;
			int i = 0;
			while (i < cursor.column && i < text[cursor.line].length()) {
				if (!_is_whitespace(text[cursor.line][i])) {
					unindent = false;
					break;
				}
				i++;
			}

			// Then we can remove all spaces as a single character.
			if (unindent) {
				// We want to remove spaces up to closest indent, or whole indent if cursor is pointing at it.
				int spaces_to_delete = _calculate_spaces_till_next_left_indent(cursor.column);
				prev_column = cursor.column - spaces_to_delete;
				_remove_text(cursor.line, prev_column, cursor.line, cursor.column);
			} else {
				_remove_text(prev_line, prev_column, cursor.line, cursor.column);
			}
		} else {
			_remove_text(prev_line, prev_column, cursor.line, cursor.column);
		}
	}

	cursor_set_line(prev_line, false, true);
	cursor_set_column(prev_column);
}

void TextEdit::indent_selected_lines_right() {
	int start_line;
	int end_line;

	// This value informs us by how much we changed selection position by indenting right.
	// Default is 1 for tab indentation.
	int selection_offset = 1;
	begin_complex_operation();

	if (is_selection_active()) {
		start_line = get_selection_from_line();
		end_line = get_selection_to_line();
	} else {
		start_line = cursor.line;
		end_line = start_line;
	}

	// Ignore if the cursor is not past the first column.
	if (is_selection_active() && get_selection_to_column() == 0) {
		selection_offset = 0;
		end_line--;
	}

	for (int i = start_line; i <= end_line; i++) {
		String line_text = get_line(i);
		if (line_text.size() == 0 && is_selection_active()) {
			continue;
		}
		if (indent_using_spaces) {
			// We don't really care where selection is - we just need to know indentation level at the beginning of the line.
			int left = _find_first_non_whitespace_column_of_line(line_text);
			int spaces_to_add = _calculate_spaces_till_next_right_indent(left);
			// Since we will add these many spaces, we want to move the whole selection and cursor by this much.
			selection_offset = spaces_to_add;
			for (int j = 0; j < spaces_to_add; j++) {
				line_text = ' ' + line_text;
			}
		} else {
			line_text = '\t' + line_text;
		}
		set_line(i, line_text);
	}

	// Fix selection and cursor being off after shifting selection right.
	if (is_selection_active()) {
		select(selection.from_line, selection.from_column + selection_offset, selection.to_line, selection.to_column + selection_offset);
	}
	cursor_set_column(cursor.column + selection_offset, false);
	end_complex_operation();
	update();
}

void TextEdit::indent_selected_lines_left() {
	int start_line;
	int end_line;

	// Moving cursor and selection after unindenting can get tricky because
	// changing content of line can move cursor and selection on its own (if new line ends before previous position of either),
	// therefore we just remember initial values and at the end of the operation offset them by number of removed characters.
	int removed_characters = 0;
	int initial_selection_end_column = selection.to_column;
	int initial_cursor_column = cursor.column;

	begin_complex_operation();

	if (is_selection_active()) {
		start_line = get_selection_from_line();
		end_line = get_selection_to_line();
	} else {
		start_line = cursor.line;
		end_line = start_line;
	}

	// Ignore if the cursor is not past the first column.
	if (is_selection_active() && get_selection_to_column() == 0) {
		end_line--;
	}
	String first_line_text = get_line(start_line);
	String last_line_text = get_line(end_line);

	for (int i = start_line; i <= end_line; i++) {
		String line_text = get_line(i);

		if (line_text.begins_with("\t")) {
			line_text = line_text.substr(1, line_text.length());
			set_line(i, line_text);
			removed_characters = 1;
		} else if (line_text.begins_with(" ")) {
			// When unindenting we aim to remove spaces before line that has selection no matter what is selected,
			// so we start of by finding first non whitespace character of line
			int left = _find_first_non_whitespace_column_of_line(line_text);

			// Here we remove only enough spaces to align text to nearest full multiple of indentation_size.
			// In case where selection begins at the start of indentation_size multiple we remove whole indentation level.
			int spaces_to_remove = _calculate_spaces_till_next_left_indent(left);

			line_text = line_text.substr(spaces_to_remove, line_text.length());
			set_line(i, line_text);
			removed_characters = spaces_to_remove;
		}
	}

	if (is_selection_active()) {
		// Fix selection being off by one on the first line.
		if (first_line_text != get_line(start_line)) {
			select(selection.from_line, selection.from_column - removed_characters,
					selection.to_line, initial_selection_end_column);
		}
		// Fix selection being off by one on the last line.
		if (last_line_text != get_line(end_line)) {
			select(selection.from_line, selection.from_column,
					selection.to_line, initial_selection_end_column - removed_characters);
		}
	}
	cursor_set_column(initial_cursor_column - removed_characters, false);
	end_complex_operation();
	update();
}

int TextEdit::_calculate_spaces_till_next_left_indent(int column) {
	int spaces_till_indent = column % indent_size;
	if (spaces_till_indent == 0) {
		spaces_till_indent = indent_size;
	}
	return spaces_till_indent;
}

int TextEdit::_calculate_spaces_till_next_right_indent(int column) {
	return indent_size - column % indent_size;
}

void TextEdit::_swap_current_input_direction() {
	if (input_direction == TEXT_DIRECTION_LTR) {
		input_direction = TEXT_DIRECTION_RTL;
	} else {
		input_direction = TEXT_DIRECTION_LTR;
	}
	cursor_set_column(cursor.column);
	update();
}

void TextEdit::_new_line(bool p_split_current_line, bool p_above) {
	if (readonly) {
		return;
	}

	String ins = "\n";

	// Keep indentation.
	int space_count = 0;
	for (int i = 0; i < cursor.column; i++) {
		if (text[cursor.line][i] == '\t') {
			if (indent_using_spaces) {
				ins += space_indent;
			} else {
				ins += "\t";
			}
			space_count = 0;
		} else if (text[cursor.line][i] == ' ') {
			space_count++;

			if (space_count == indent_size) {
				if (indent_using_spaces) {
					ins += space_indent;
				} else {
					ins += "\t";
				}
				space_count = 0;
			}
		} else {
			break;
		}
	}

	if (is_folded(cursor.line)) {
		unfold_line(cursor.line);
	}

	bool brace_indent = false;

	// No need to indent if we are going upwards.
	if (auto_indent && !p_above) {
		// Indent once again if previous line will end with ':','{','[','(' and the line is not a comment
		// (i.e. colon/brace precedes current cursor position).
		if (cursor.column > 0) {
			bool indent_char_found = false;
			bool should_indent = false;
			char indent_char = ':';
			char c = text[cursor.line][cursor.column];

			for (int i = 0; i < cursor.column; i++) {
				c = text[cursor.line][i];
				switch (c) {
					case ':':
					case '{':
					case '[':
					case '(':
						indent_char_found = true;
						should_indent = true;
						indent_char = c;
						continue;
				}

				if (indent_char_found && is_line_comment(cursor.line)) {
					should_indent = true;
					break;
				} else if (indent_char_found && !_is_whitespace(c)) {
					should_indent = false;
					indent_char_found = false;
				}
			}

			if (!is_line_comment(cursor.line) && should_indent) {
				if (indent_using_spaces) {
					ins += space_indent;
				} else {
					ins += "\t";
				}

				// No need to move the brace below if we are not taking the text with us.
				char32_t closing_char = _get_right_pair_symbol(indent_char);
				if ((closing_char != 0) && (closing_char == text[cursor.line][cursor.column])) {
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
	}
	begin_complex_operation();
	bool first_line = false;
	if (!p_split_current_line) {
		if (p_above) {
			if (cursor.line > 0) {
				cursor_set_line(cursor.line - 1, false);
				cursor_set_column(text[cursor.line].length());
			} else {
				cursor_set_column(0);
				first_line = true;
			}
		} else {
			cursor_set_column(text[cursor.line].length());
		}
	}

	insert_text_at_cursor(ins);

	if (first_line) {
		cursor_set_line(0);
	} else if (brace_indent) {
		cursor_set_line(cursor.line - 1, false);
		cursor_set_column(text[cursor.line].length());
	}
	end_complex_operation();
}

void TextEdit::_indent_right() {
	if (readonly) {
		return;
	}

	if (is_selection_active()) {
		indent_selected_lines_right();
	} else {
		// Simple indent.
		if (indent_using_spaces) {
			// Insert only as much spaces as needed till next indentation level.
			int spaces_to_add = _calculate_spaces_till_next_right_indent(cursor.column);
			String indent_to_insert = String();
			for (int i = 0; i < spaces_to_add; i++) {
				indent_to_insert = ' ' + indent_to_insert;
			}
			_insert_text_at_cursor(indent_to_insert);
		} else {
			_insert_text_at_cursor("\t");
		}
	}
}

void TextEdit::_indent_left() {
	if (readonly) {
		return;
	}

	if (is_selection_active()) {
		indent_selected_lines_left();
	} else {
		// Simple unindent.
		int cc = cursor.column;
		const String &line = text[cursor.line];

		int left = _find_first_non_whitespace_column_of_line(line);
		cc = MIN(cc, left);

		while (cc < indent_size && cc < left && line[cc] == ' ') {
			cc++;
		}

		if (cc > 0 && cc <= text[cursor.line].length()) {
			if (text[cursor.line][cc - 1] == '\t') {
				// Tabs unindentation.
				_remove_text(cursor.line, cc - 1, cursor.line, cc);
				if (cursor.column >= left) {
					cursor_set_column(MAX(0, cursor.column - 1));
				}
				update();
			} else {
				// Spaces unindentation.
				int spaces_to_remove = _calculate_spaces_till_next_left_indent(cc);
				if (spaces_to_remove > 0) {
					_remove_text(cursor.line, cc - spaces_to_remove, cursor.line, cc);
					if (cursor.column > left - spaces_to_remove) { // Inside text?
						cursor_set_column(MAX(0, cursor.column - spaces_to_remove));
					}
					update();
				}
			}
		} else if (cc == 0 && line.length() > 0 && line[0] == '\t') {
			_remove_text(cursor.line, 0, cursor.line, 1);
			update();
		}
	}
}

void TextEdit::_move_cursor_left(bool p_select, bool p_move_by_word) {
	// Handle selection
	if (p_select) {
		_pre_shift_selection();
	} else {
		deselect();
	}

	if (p_move_by_word) {
		int cc = cursor.column;

		if (cc == 0 && cursor.line > 0) {
			cursor_set_line(cursor.line - 1);
			cursor_set_column(text[cursor.line].length());
		} else {
			Vector<Vector2i> words = TS->shaped_text_get_word_breaks(text.get_line_data(cursor.line)->get_rid());
			for (int i = words.size() - 1; i >= 0; i--) {
				if (words[i].x < cc) {
					cc = words[i].x;
					break;
				}
			}
			cursor_set_column(cc);
		}
	} else {
		// If the cursor is at the start of the line, and not on the first line, move it up to the end of the previous line.
		if (cursor.column == 0) {
			if (cursor.line > 0) {
				cursor_set_line(cursor.line - num_lines_from(CLAMP(cursor.line - 1, 0, text.size() - 1), -1));
				cursor_set_column(text[cursor.line].length());
			}
		} else {
			if (mid_grapheme_caret_enabled) {
				cursor_set_column(cursor_get_column() - 1);
			} else {
				cursor_set_column(TS->shaped_text_prev_grapheme_pos(text.get_line_data(cursor.line)->get_rid(), cursor_get_column()));
			}
		}
	}

	if (p_select) {
		_post_shift_selection();
	}
}

void TextEdit::_move_cursor_right(bool p_select, bool p_move_by_word) {
	// Handle selection
	if (p_select) {
		_pre_shift_selection();
	} else {
		deselect();
	}

	if (p_move_by_word) {
		int cc = cursor.column;

		if (cc == text[cursor.line].length() && cursor.line < text.size() - 1) {
			cursor_set_line(cursor.line + 1);
			cursor_set_column(0);
		} else {
			Vector<Vector2i> words = TS->shaped_text_get_word_breaks(text.get_line_data(cursor.line)->get_rid());
			for (int i = 0; i < words.size(); i++) {
				if (words[i].y > cc) {
					cc = words[i].y;
					break;
				}
			}
			cursor_set_column(cc);
		}
	} else {
		// If we are at the end of the line, move the caret to the next line down.
		if (cursor.column == text[cursor.line].length()) {
			if (cursor.line < text.size() - 1) {
				cursor_set_line(cursor_get_line() + num_lines_from(CLAMP(cursor.line + 1, 0, text.size() - 1), 1), true, false);
				cursor_set_column(0);
			}
		} else {
			if (mid_grapheme_caret_enabled) {
				cursor_set_column(cursor_get_column() + 1);
			} else {
				cursor_set_column(TS->shaped_text_next_grapheme_pos(text.get_line_data(cursor.line)->get_rid(), cursor_get_column()));
			}
		}
	}

	if (p_select) {
		_post_shift_selection();
	}
}

void TextEdit::_move_cursor_up(bool p_select) {
	if (p_select) {
		_pre_shift_selection();
	} else {
		deselect();
	}

	int cur_wrap_index = get_cursor_wrap_index();
	if (cur_wrap_index > 0) {
		cursor_set_line(cursor.line, true, false, cur_wrap_index - 1);
	} else if (cursor.line == 0) {
		cursor_set_column(0);
	} else {
		int new_line = cursor.line - num_lines_from(cursor.line - 1, -1);
		if (line_wraps(new_line)) {
			cursor_set_line(new_line, true, false, times_line_wraps(new_line));
		} else {
			cursor_set_line(new_line, true, false);
		}
	}

	if (p_select) {
		_post_shift_selection();
	}

	_cancel_code_hint();
}

void TextEdit::_move_cursor_down(bool p_select) {
	if (p_select) {
		_pre_shift_selection();
	} else {
		deselect();
	}

	int cur_wrap_index = get_cursor_wrap_index();
	if (cur_wrap_index < times_line_wraps(cursor.line)) {
		cursor_set_line(cursor.line, true, false, cur_wrap_index + 1);
	} else if (cursor.line == get_last_unhidden_line()) {
		cursor_set_column(text[cursor.line].length());
	} else {
		int new_line = cursor.line + num_lines_from(CLAMP(cursor.line + 1, 0, text.size() - 1), 1);
		cursor_set_line(new_line, true, false, 0);
	}

	if (p_select) {
		_post_shift_selection();
	}

	_cancel_code_hint();
}

void TextEdit::_move_cursor_to_line_start(bool p_select) {
	if (p_select) {
		_pre_shift_selection();
	} else {
		deselect();
	}

	// Move cursor column to start of wrapped row and then to start of text.
	Vector<String> rows = get_wrap_rows_text(cursor.line);
	int wi = get_cursor_wrap_index();
	int row_start_col = 0;
	for (int i = 0; i < wi; i++) {
		row_start_col += rows[i].length();
	}
	if (cursor.column == row_start_col || wi == 0) {
		// Compute whitespace symbols sequence length.
		int current_line_whitespace_len = 0;
		while (current_line_whitespace_len < text[cursor.line].length()) {
			char32_t c = text[cursor.line][current_line_whitespace_len];
			if (c != '\t' && c != ' ') {
				break;
			}
			current_line_whitespace_len++;
		}

		if (cursor_get_column() == current_line_whitespace_len) {
			cursor_set_column(0);
		} else {
			cursor_set_column(current_line_whitespace_len);
		}
	} else {
		cursor_set_column(row_start_col);
	}

	if (p_select) {
		_post_shift_selection();
	}

	_cancel_completion();
	completion_hint = "";
}

void TextEdit::_move_cursor_to_line_end(bool p_select) {
	if (p_select) {
		_pre_shift_selection();
	} else {
		deselect();
	}

	// Move cursor column to end of wrapped row and then to end of text.
	Vector<String> rows = get_wrap_rows_text(cursor.line);
	int wi = get_cursor_wrap_index();
	int row_end_col = -1;
	for (int i = 0; i < wi + 1; i++) {
		row_end_col += rows[i].length();
	}
	if (wi == rows.size() - 1 || cursor.column == row_end_col) {
		cursor_set_column(text[cursor.line].length());
	} else {
		cursor_set_column(row_end_col);
	}

	if (p_select) {
		_post_shift_selection();
	}
	_cancel_completion();
	completion_hint = "";
}

void TextEdit::_move_cursor_page_up(bool p_select) {
	if (p_select) {
		_pre_shift_selection();
	} else {
		deselect();
	}

	int wi;
	int n_line = cursor.line - num_lines_from_rows(cursor.line, get_cursor_wrap_index(), -get_visible_rows(), wi) + 1;
	cursor_set_line(n_line, true, false, wi);

	if (p_select) {
		_post_shift_selection();
	}

	_cancel_completion();
	completion_hint = "";
}

void TextEdit::_move_cursor_page_down(bool p_select) {
	if (p_select) {
		_pre_shift_selection();
	} else {
		deselect();
	}

	int wi;
	int n_line = cursor.line + num_lines_from_rows(cursor.line, get_cursor_wrap_index(), get_visible_rows(), wi) - 1;
	cursor_set_line(n_line, true, false, wi);

	if (p_select) {
		_post_shift_selection();
	}

	_cancel_completion();
	completion_hint = "";
}

void TextEdit::_backspace(bool p_word, bool p_all_to_left) {
	if (readonly) {
		return;
	}

	if (is_selection_active()) {
		_delete_selection();
		return;
	}
	if (p_all_to_left) {
		int cursor_current_column = cursor.column;
		cursor.column = 0;
		_remove_text(cursor.line, 0, cursor.line, cursor_current_column);
	} else if (p_word) {
		int line = cursor.line;
		int column = cursor.column;

		Vector<Vector2i> words = TS->shaped_text_get_word_breaks(text.get_line_data(line)->get_rid());
		for (int i = words.size() - 1; i >= 0; i--) {
			if (words[i].x < column) {
				column = words[i].x;
				break;
			}
		}

		_remove_text(line, column, cursor.line, cursor.column);

		cursor_set_line(line, false);
		cursor_set_column(column);
	} else {
		// One character.
		if (cursor.line > 0 && is_line_hidden(cursor.line - 1)) {
			unfold_line(cursor.line - 1);
		}
		backspace_at_cursor();
	}
}

void TextEdit::_delete(bool p_word, bool p_all_to_right) {
	if (readonly) {
		return;
	}

	if (is_selection_active()) {
		_delete_selection();
		return;
	}
	int curline_len = text[cursor.line].length();

	if (cursor.line == text.size() - 1 && cursor.column == curline_len) {
		return; // Last line, last column: Nothing to do.
	}

	int next_line = cursor.column < curline_len ? cursor.line : cursor.line + 1;
	int next_column;

	if (p_all_to_right) {
		// Delete everything to right of cursor
		next_column = curline_len;
		next_line = cursor.line;
	} else if (p_word && cursor.column < curline_len - 1) {
		// Delete next word to right of cursor
		int line = cursor.line;
		int column = cursor.column;

		Vector<Vector2i> words = TS->shaped_text_get_word_breaks(text.get_line_data(line)->get_rid());
		for (int i = 0; i < words.size(); i++) {
			if (words[i].y > column) {
				column = words[i].y;
				break;
			}
		}

		next_line = line;
		next_column = column;
	} else {
		// Delete one character
		next_column = cursor.column < curline_len ? (cursor.column + 1) : 0;
		if (mid_grapheme_caret_enabled) {
			next_column = cursor.column < curline_len ? (cursor.column + 1) : 0;
		} else {
			next_column = cursor.column < curline_len ? TS->shaped_text_next_grapheme_pos(text.get_line_data(cursor.line)->get_rid(), (cursor.column)) : 0;
		}
	}

	_remove_text(cursor.line, cursor.column, next_line, next_column);
	update();
}

void TextEdit::_delete_selection() {
	if (is_selection_active()) {
		selection.active = false;
		update();
		_remove_text(selection.from_line, selection.from_column, selection.to_line, selection.to_column);
		cursor_set_line(selection.from_line, false, false);
		cursor_set_column(selection.from_column);
		update();
	}
}

void TextEdit::_move_cursor_document_start(bool p_select) {
	if (p_select) {
		_pre_shift_selection();
	} else {
		deselect();
	}

	cursor_set_line(0);
	cursor_set_column(0);

	if (p_select) {
		_post_shift_selection();
	}
}

void TextEdit::_move_cursor_document_end(bool p_select) {
	if (p_select) {
		_pre_shift_selection();
	} else {
		deselect();
	}

	cursor_set_line(get_last_unhidden_line(), true, false, 9999);
	cursor_set_column(text[cursor.line].length());

	if (p_select) {
		_post_shift_selection();
	}
}

void TextEdit::_handle_unicode_character(uint32_t unicode, bool p_had_selection, bool p_update_auto_complete) {
	if (p_update_auto_complete) {
		_reset_caret_blink_timer();
	}

	if (p_had_selection) {
		_delete_selection();
	}

	// Remove the old character if in insert mode and no selection.
	if (insert_mode && !p_had_selection) {
		begin_complex_operation();

		// Make sure we don't try and remove empty space.
		if (cursor.column < get_line(cursor.line).length()) {
			_remove_text(cursor.line, cursor.column, cursor.line, cursor.column + 1);
		}
	}

	const char32_t chr[2] = { (char32_t)unicode, 0 };

	// Clear completion hint when function closed
	if (completion_hint != "" && unicode == ')') {
		completion_hint = "";
	}

	if (auto_brace_completion_enabled && _is_pair_symbol(chr[0])) {
		_consume_pair_symbol(chr[0]);
	} else {
		_insert_text_at_cursor(chr);
	}

	if ((insert_mode && !p_had_selection) || (selection.active != p_had_selection)) {
		end_complex_operation();
	}

	if (p_update_auto_complete) {
		_update_completion_candidates();
	}
}

void TextEdit::_get_mouse_pos(const Point2i &p_mouse, int &r_row, int &r_col) const {
	float rows = p_mouse.y;
	rows -= cache.style_normal->get_margin(SIDE_TOP);
	rows /= get_row_height();
	rows += get_v_scroll_offset();
	int first_vis_line = get_first_visible_line();
	int row = first_vis_line + Math::floor(rows);
	int wrap_index = 0;

	if (is_wrap_enabled() || is_hiding_enabled()) {
		int f_ofs = num_lines_from_rows(first_vis_line, cursor.wrap_ofs, rows + (1 * SGN(rows)), wrap_index) - 1;
		if (rows < 0) {
			row = first_vis_line - f_ofs;
		} else {
			row = first_vis_line + f_ofs;
		}
	}

	if (row < 0) {
		row = 0;
	}

	int col = 0;

	if (row >= text.size()) {
		row = text.size() - 1;
		col = text[row].size();
	} else {
		int colx = p_mouse.x - (cache.style_normal->get_margin(SIDE_LEFT) + gutters_width + gutter_padding);
		colx += cursor.x_ofs;
		col = get_char_pos_for_line(colx, row, wrap_index);
		if (is_wrap_enabled() && wrap_index < times_line_wraps(row)) {
			// Move back one if we are at the end of the row.
			Vector<String> rows2 = get_wrap_rows_text(row);
			int row_end_col = 0;
			for (int i = 0; i < wrap_index + 1; i++) {
				row_end_col += rows2[i].length();
			}
			if (col >= row_end_col) {
				col -= 1;
			}
		}

		RID text_rid = text.get_line_data(row)->get_line_rid(wrap_index);
		if (is_layout_rtl()) {
			colx = TS->shaped_text_get_size(text_rid).x - colx;
		}
		col = TS->shaped_text_hit_test_position(text_rid, colx);
	}

	r_row = row;
	r_col = col;
}

Vector2i TextEdit::_get_cursor_pixel_pos(bool p_adjust_viewport) {
	if (p_adjust_viewport) {
		adjust_viewport_to_cursor();
	}
	int row = 1;
	for (int i = get_first_visible_line(); i < cursor.line; i++) {
		if (!is_line_hidden(i)) {
			row += times_line_wraps(i) + 1;
		}
	}
	row += cursor.wrap_ofs;

	// Calculate final pixel position
	int y = (row - get_v_scroll_offset()) * get_row_height();
	int x = cache.style_normal->get_margin(SIDE_LEFT) + gutters_width + gutter_padding - cursor.x_ofs;

	Rect2 l_caret, t_caret;
	TextServer::Direction l_dir, t_dir;
	RID text_rid = text.get_line_data(cursor.line)->get_line_rid(cursor.wrap_ofs);
	TS->shaped_text_get_carets(text_rid, cursor.column, l_caret, l_dir, t_caret, t_dir);
	if ((l_caret != Rect2() && (l_dir == TextServer::DIRECTION_AUTO || l_dir == (TextServer::Direction)input_direction)) || (t_caret == Rect2())) {
		x += l_caret.position.x;
	} else {
		x += t_caret.position.x;
	}

	return Vector2i(x, y);
}

void TextEdit::_get_minimap_mouse_row(const Point2i &p_mouse, int &r_row) const {
	float rows = p_mouse.y;
	rows -= cache.style_normal->get_margin(SIDE_TOP);
	rows /= (minimap_char_size.y + minimap_line_spacing);
	rows += get_v_scroll_offset();

	// calculate visible lines
	int minimap_visible_lines = _get_minimap_visible_rows();
	int visible_rows = get_visible_rows() + 1;
	int first_visible_line = get_first_visible_line() - 1;
	int draw_amount = visible_rows + (smooth_scroll_enabled ? 1 : 0);
	draw_amount += times_line_wraps(first_visible_line + 1);
	int minimap_line_height = (minimap_char_size.y + minimap_line_spacing);

	// calculate viewport size and y offset
	int viewport_height = (draw_amount - 1) * minimap_line_height;
	int control_height = _get_control_height() - viewport_height;
	int viewport_offset_y = round(get_scroll_pos_for_line(first_visible_line) * control_height) / ((v_scroll->get_max() <= minimap_visible_lines) ? (minimap_visible_lines - draw_amount) : (v_scroll->get_max() - draw_amount));

	// calculate the first line.
	int num_lines_before = round((viewport_offset_y) / minimap_line_height);
	int wi;
	int minimap_line = (v_scroll->get_max() <= minimap_visible_lines) ? -1 : first_visible_line;
	if (first_visible_line > 0 && minimap_line >= 0) {
		minimap_line -= num_lines_from_rows(first_visible_line, 0, -num_lines_before, wi);
		minimap_line -= (minimap_line > 0 && smooth_scroll_enabled ? 1 : 0);
	} else {
		minimap_line = 0;
	}

	int row = minimap_line + Math::floor(rows);
	int wrap_index = 0;

	if (is_wrap_enabled() || is_hiding_enabled()) {
		int f_ofs = num_lines_from_rows(minimap_line, cursor.wrap_ofs, rows + (1 * SGN(rows)), wrap_index) - 1;
		if (rows < 0) {
			row = minimap_line - f_ofs;
		} else {
			row = minimap_line + f_ofs;
		}
	}

	if (row < 0) {
		row = 0;
	}

	if (row >= text.size()) {
		row = text.size() - 1;
	}

	r_row = row;
}

void TextEdit::_gui_input(const Ref<InputEvent> &p_gui_input) {
	ERR_FAIL_COND(p_gui_input.is_null());

	double prev_v_scroll = v_scroll->get_value();
	double prev_h_scroll = h_scroll->get_value();

	Ref<InputEventMouseButton> mb = p_gui_input;

	if (mb.is_valid()) {
		Vector2i mpos = mb->get_position();
		if (is_layout_rtl()) {
			mpos.x = get_size().x - mpos.x;
		}
		if (ime_text.length() != 0) {
			// Ignore mouse clicks in IME input mode.
			return;
		}
		if (completion_active && completion_rect.has_point(mpos)) {
			if (!mb->is_pressed()) {
				return;
			}

			if (mb->get_button_index() == MOUSE_BUTTON_WHEEL_UP) {
				if (completion_index > 0) {
					completion_index--;
					completion_current = completion_options[completion_index];
					update();
				}
			}
			if (mb->get_button_index() == MOUSE_BUTTON_WHEEL_DOWN) {
				if (completion_index < completion_options.size() - 1) {
					completion_index++;
					completion_current = completion_options[completion_index];
					update();
				}
			}

			if (mb->get_button_index() == MOUSE_BUTTON_LEFT) {
				completion_index = CLAMP(completion_line_ofs + (mpos.y - completion_rect.position.y) / get_row_height(), 0, completion_options.size() - 1);

				completion_current = completion_options[completion_index];
				update();
				if (mb->is_double_click()) {
					_confirm_completion();
				}
			}
			return;
		} else {
			_cancel_completion();
			_cancel_code_hint();
		}

		if (mb->is_pressed()) {
			if (mb->get_button_index() == MOUSE_BUTTON_WHEEL_UP && !mb->is_command_pressed()) {
				if (mb->is_shift_pressed()) {
					h_scroll->set_value(h_scroll->get_value() - (100 * mb->get_factor()));
				} else if (mb->is_alt_pressed()) {
					// Scroll 5 times as fast as normal (like in Visual Studio Code).
					_scroll_up(15 * mb->get_factor());
				} else if (v_scroll->is_visible()) {
					// Scroll 3 lines.
					_scroll_up(3 * mb->get_factor());
				}
			}
			if (mb->get_button_index() == MOUSE_BUTTON_WHEEL_DOWN && !mb->is_command_pressed()) {
				if (mb->is_shift_pressed()) {
					h_scroll->set_value(h_scroll->get_value() + (100 * mb->get_factor()));
				} else if (mb->is_alt_pressed()) {
					// Scroll 5 times as fast as normal (like in Visual Studio Code).
					_scroll_down(15 * mb->get_factor());
				} else if (v_scroll->is_visible()) {
					// Scroll 3 lines.
					_scroll_down(3 * mb->get_factor());
				}
			}
			if (mb->get_button_index() == MOUSE_BUTTON_WHEEL_LEFT) {
				h_scroll->set_value(h_scroll->get_value() - (100 * mb->get_factor()));
			}
			if (mb->get_button_index() == MOUSE_BUTTON_WHEEL_RIGHT) {
				h_scroll->set_value(h_scroll->get_value() + (100 * mb->get_factor()));
			}
			if (mb->get_button_index() == MOUSE_BUTTON_LEFT) {
				_reset_caret_blink_timer();

				int row, col;
				_get_mouse_pos(Point2i(mpos.x, mpos.y), row, col);

				int left_margin = cache.style_normal->get_margin(SIDE_LEFT);
				for (int i = 0; i < gutters.size(); i++) {
					if (!gutters[i].draw || gutters[i].width <= 0) {
						continue;
					}

					if (mpos.x > left_margin && mpos.x <= (left_margin + gutters[i].width) - 3) {
						emit_signal("gutter_clicked", row, i);
						return;
					}

					left_margin += gutters[i].width;
				}

				// Unfold on folded icon click.
				if (is_folded(row)) {
					left_margin += gutter_padding + text.get_line_width(row) - cursor.x_ofs;
					if (mpos.x > left_margin && mpos.x <= left_margin + cache.folded_eol_icon->get_width() + 3) {
						unfold_line(row);
						return;
					}
				}

				// minimap
				if (draw_minimap) {
					_update_minimap_click();
					if (dragging_minimap) {
						return;
					}
				}

				int prev_col = cursor.column;
				int prev_line = cursor.line;

				cursor_set_line(row, false, false);
				cursor_set_column(col);

				if (mb->is_shift_pressed() && (cursor.column != prev_col || cursor.line != prev_line)) {
					if (!selection.active) {
						selection.active = true;
						selection.selecting_mode = SelectionMode::SELECTION_MODE_POINTER;
						selection.from_column = prev_col;
						selection.from_line = prev_line;
						selection.to_column = cursor.column;
						selection.to_line = cursor.line;

						if (selection.from_line > selection.to_line || (selection.from_line == selection.to_line && selection.from_column > selection.to_column)) {
							SWAP(selection.from_column, selection.to_column);
							SWAP(selection.from_line, selection.to_line);
							selection.shiftclick_left = false;
						} else {
							selection.shiftclick_left = true;
						}
						selection.selecting_line = prev_line;
						selection.selecting_column = prev_col;
						update();
					} else {
						if (cursor.line < selection.selecting_line || (cursor.line == selection.selecting_line && cursor.column < selection.selecting_column)) {
							if (selection.shiftclick_left) {
								selection.shiftclick_left = !selection.shiftclick_left;
							}
							selection.from_column = cursor.column;
							selection.from_line = cursor.line;

						} else if (cursor.line > selection.selecting_line || (cursor.line == selection.selecting_line && cursor.column > selection.selecting_column)) {
							if (!selection.shiftclick_left) {
								SWAP(selection.from_column, selection.to_column);
								SWAP(selection.from_line, selection.to_line);
								selection.shiftclick_left = !selection.shiftclick_left;
							}
							selection.to_column = cursor.column;
							selection.to_line = cursor.line;

						} else {
							selection.active = false;
						}

						update();
					}
				} else {
					selection.active = false;
					selection.selecting_mode = SelectionMode::SELECTION_MODE_POINTER;
					selection.selecting_line = row;
					selection.selecting_column = col;
				}

				if (!mb->is_double_click() && (OS::get_singleton()->get_ticks_msec() - last_dblclk) < 600 && cursor.line == prev_line) {
					// Triple-click select line.
					selection.selecting_mode = SelectionMode::SELECTION_MODE_LINE;
					_update_selection_mode_line();
					last_dblclk = 0;
				} else if (mb->is_double_click() && text[cursor.line].length()) {
					// Double-click select word.
					selection.selecting_mode = SelectionMode::SELECTION_MODE_WORD;
					_update_selection_mode_word();
					last_dblclk = OS::get_singleton()->get_ticks_msec();
				}

				update();
			}

			if (mb->get_button_index() == MOUSE_BUTTON_RIGHT && context_menu_enabled) {
				_reset_caret_blink_timer();

				int row, col;
				_get_mouse_pos(Point2i(mpos.x, mpos.y), row, col);

				if (is_right_click_moving_caret()) {
					if (is_selection_active()) {
						int from_line = get_selection_from_line();
						int to_line = get_selection_to_line();
						int from_column = get_selection_from_column();
						int to_column = get_selection_to_column();

						if (row < from_line || row > to_line || (row == from_line && col < from_column) || (row == to_line && col > to_column)) {
							// Right click is outside the selected text.
							deselect();
						}
					}
					if (!is_selection_active()) {
						cursor_set_line(row, true, false);
						cursor_set_column(col);
					}
				}

				menu->set_position(get_screen_transform().xform(mpos));
				menu->set_size(Vector2(1, 1));
				_generate_context_menu();
				menu->popup();
				grab_focus();
			}
		} else {
			if (mb->get_button_index() == MOUSE_BUTTON_LEFT) {
				if (mb->is_command_pressed() && highlighted_word != String()) {
					int row, col;
					_get_mouse_pos(Point2i(mpos.x, mpos.y), row, col);

					emit_signal("symbol_lookup", highlighted_word, row, col);
					return;
				}

				dragging_minimap = false;
				dragging_selection = false;
				can_drag_minimap = false;
				click_select_held->stop();
			}

			// Notify to show soft keyboard.
			notification(NOTIFICATION_FOCUS_ENTER);
		}
	}

	const Ref<InputEventPanGesture> pan_gesture = p_gui_input;
	if (pan_gesture.is_valid()) {
		const real_t delta = pan_gesture->get_delta().y;
		if (delta < 0) {
			_scroll_up(-delta);
		} else {
			_scroll_down(delta);
		}
		h_scroll->set_value(h_scroll->get_value() + pan_gesture->get_delta().x * 100);
		if (v_scroll->get_value() != prev_v_scroll || h_scroll->get_value() != prev_h_scroll) {
			accept_event(); // Accept event if scroll changed.
		}

		return;
	}

	Ref<InputEventMouseMotion> mm = p_gui_input;

	if (mm.is_valid()) {
		Vector2i mpos = mm->get_position();
		if (is_layout_rtl()) {
			mpos.x = get_size().x - mpos.x;
		}
		if (select_identifiers_enabled) {
			if (!dragging_minimap && !dragging_selection && mm->is_command_pressed() && mm->get_button_mask() == 0) {
				String new_word = get_word_at_pos(mpos);
				if (new_word != highlighted_word) {
					emit_signal("symbol_validate", new_word);
				}
			} else {
				if (highlighted_word != String()) {
					set_highlighted_word(String());
				}
			}
		}

		if (mm->get_button_mask() & MOUSE_BUTTON_MASK_LEFT && get_viewport()->gui_get_drag_data() == Variant()) { // Ignore if dragging.
			_reset_caret_blink_timer();

			if (draw_minimap && !dragging_selection) {
				_update_minimap_drag();
			}

			if (!dragging_minimap) {
				switch (selection.selecting_mode) {
					case SelectionMode::SELECTION_MODE_POINTER: {
						_update_selection_mode_pointer();
					} break;
					case SelectionMode::SELECTION_MODE_WORD: {
						_update_selection_mode_word();
					} break;
					case SelectionMode::SELECTION_MODE_LINE: {
						_update_selection_mode_line();
					} break;
					default: {
						break;
					}
				}
			}
		}
	}

	if (v_scroll->get_value() != prev_v_scroll || h_scroll->get_value() != prev_h_scroll) {
		accept_event(); // Accept event if scroll changed.
	}

	Ref<InputEventKey> k = p_gui_input;

	if (k.is_valid()) {
		// Ctrl + Hover symbols
#ifdef OSX_ENABLED
		if (k->get_keycode() == KEY_META) {
#else
		if (k->get_keycode() == KEY_CTRL) {
#endif
			if (select_identifiers_enabled) {
				if (k->is_pressed() && !dragging_minimap && !dragging_selection) {
					Point2 mp = _get_local_mouse_pos();
					emit_signal("symbol_validate", get_word_at_pos(mp));
				} else {
					set_highlighted_word(String());
				}
			}
			return;
		}

		if (!k->is_pressed()) {
			return;
		}

		// If a modifier has been pressed, and nothing else, return.
		if (k->get_keycode() == KEY_CTRL || k->get_keycode() == KEY_ALT || k->get_keycode() == KEY_SHIFT || k->get_keycode() == KEY_META) {
			return;
		}

		_reset_caret_blink_timer();

		// Allow unicode handling if:
		// * No Modifiers are pressed (except shift)
		bool allow_unicode_handling = !(k->is_command_pressed() || k->is_ctrl_pressed() || k->is_alt_pressed() || k->is_meta_pressed());

		// Save here for insert mode, just in case it is cleared in the following section.
		bool had_selection = selection.active;

		selection.selecting_text = false;

		// Check and handle all built in shortcuts.

		// AUTO-COMPLETE

		if (k->is_action("ui_text_completion_query", true)) {
			query_code_comple();
			accept_event();
			return;
		}

		if (completion_active) {
			if (k->is_action("ui_up", true)) {
				if (completion_index > 0) {
					completion_index--;
				} else {
					completion_index = completion_options.size() - 1;
				}
				completion_current = completion_options[completion_index];
				update();
				accept_event();
				return;
			}
			if (k->is_action("ui_down", true)) {
				if (completion_index < completion_options.size() - 1) {
					completion_index++;
				} else {
					completion_index = 0;
				}
				completion_current = completion_options[completion_index];
				update();
				accept_event();
				return;
			}
			if (k->is_action("ui_page_up", true)) {
				completion_index -= get_theme_constant("completion_lines");
				if (completion_index < 0) {
					completion_index = 0;
				}
				completion_current = completion_options[completion_index];
				update();
				accept_event();
				return;
			}
			if (k->is_action("ui_page_down", true)) {
				completion_index += get_theme_constant("completion_lines");
				if (completion_index >= completion_options.size()) {
					completion_index = completion_options.size() - 1;
				}
				completion_current = completion_options[completion_index];
				update();
				accept_event();
				return;
			}
			if (k->is_action("ui_home", true)) {
				if (completion_index > 0) {
					completion_index = 0;
					completion_current = completion_options[completion_index];
					update();
				}
				accept_event();
				return;
			}
			if (k->is_action("ui_end", true)) {
				if (completion_index < completion_options.size() - 1) {
					completion_index = completion_options.size() - 1;
					completion_current = completion_options[completion_index];
					update();
				}
				accept_event();
				return;
			}
			if (k->is_action("ui_text_completion_accept", true)) {
				_confirm_completion();
				accept_event();
				return;
			}
			if (k->is_action("ui_cancel", true)) {
				_cancel_completion();
				accept_event();
				return;
			}

			// Handle Unicode here (if no modifiers active) and update autocomplete.
			if (k->get_unicode() >= 32) {
				if (allow_unicode_handling && !readonly) {
					_handle_unicode_character(k->get_unicode(), had_selection, true);
					accept_event();
					return;
				}
			}
		}

		// NEWLINES.
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

		// INDENTATION.
		if (k->is_action("ui_text_dedent", true)) {
			_indent_left();
			accept_event();
			return;
		}
		if (k->is_action("ui_text_indent", true)) {
			_indent_right();
			accept_event();
			return;
		}

		// BACKSPACE AND DELETE.
		if (k->is_action("ui_text_backspace_all_to_left", true)) {
			_backspace(false, true);
			accept_event();
			return;
		}
		if (k->is_action("ui_text_backspace_word", true)) {
			_backspace(true);
			accept_event();
			return;
		}
		if (k->is_action("ui_text_backspace", true)) {
			_backspace();
			if (completion_active) {
				_update_completion_candidates();
			}
			accept_event();
			return;
		}
		if (k->is_action("ui_text_delete_all_to_right", true)) {
			_delete(false, true);
			accept_event();
			return;
		}
		if (k->is_action("ui_text_delete_word", true)) {
			_delete(true);
			accept_event();
			return;
		}
		if (k->is_action("ui_text_delete", true)) {
			_delete();
			accept_event();
			return;
		}

		// SCROLLING.
		if (k->is_action("ui_text_scroll_up", true)) {
			_scroll_lines_up();
			accept_event();
			return;
		}
		if (k->is_action("ui_text_scroll_down", true)) {
			_scroll_lines_down();
			accept_event();
			return;
		}

		// SELECT ALL, SELECT WORD UNDER CARET, CUT, COPY, PASTE.

		if (k->is_action("ui_text_select_all", true)) {
			select_all();
			accept_event();
			return;
		}
		if (k->is_action("ui_text_select_word_under_caret", true)) {
			select_word_under_caret();
			accept_event();
			return;
		}
		if (k->is_action("ui_cut", true)) {
			cut();
			accept_event();
			return;
		}
		if (k->is_action("ui_copy", true)) {
			copy();
			accept_event();
			return;
		}
		if (k->is_action("ui_paste", true)) {
			paste();
			accept_event();
			return;
		}

		// UNDO/REDO.
		if (k->is_action("ui_undo", true)) {
			undo();
			accept_event();
			return;
		}
		if (k->is_action("ui_redo", true)) {
			redo();
			accept_event();
			return;
		}

		// MISC.

		if (k->is_action("ui_menu", true)) {
			if (context_menu_enabled) {
				menu->set_position(get_screen_transform().xform(_get_cursor_pixel_pos()));
				menu->set_size(Vector2(1, 1));
				_generate_context_menu();
				menu->popup();
				menu->grab_focus();
			}
			accept_event();
			return;
		}
		if (k->is_action("ui_text_toggle_insert_mode", true)) {
			set_insert_mode(!insert_mode);
			accept_event();
			return;
		}
		if (k->is_action("ui_cancel", true)) {
			if (completion_hint != "") {
				completion_hint = "";
				update();
			}
			accept_event();
			return;
		}
		if (k->is_action("ui_swap_input_direction", true)) {
			_swap_current_input_direction();
			accept_event();
			return;
		}

		// CURSOR MOVEMENT

		k = k->duplicate();
		bool shift_pressed = k->is_shift_pressed();
		// Remove shift or else actions will not match. Use above variable for selection.
		k->set_shift_pressed(false);

		// CURSOR MOVEMENT - LEFT, RIGHT.
		if (k->is_action("ui_text_caret_word_left", true)) {
			_move_cursor_left(shift_pressed, true);
			accept_event();
			return;
		}
		if (k->is_action("ui_text_caret_left", true)) {
			_move_cursor_left(shift_pressed, false);
			accept_event();
			return;
		}
		if (k->is_action("ui_text_caret_word_right", true)) {
			_move_cursor_right(shift_pressed, true);
			accept_event();
			return;
		}
		if (k->is_action("ui_text_caret_right", true)) {
			_move_cursor_right(shift_pressed, false);
			accept_event();
			return;
		}

		// CURSOR MOVEMENT - UP, DOWN.
		if (k->is_action("ui_text_caret_up", true)) {
			_move_cursor_up(shift_pressed);
			accept_event();
			return;
		}
		if (k->is_action("ui_text_caret_down", true)) {
			_move_cursor_down(shift_pressed);
			accept_event();
			return;
		}

		// CURSOR MOVEMENT - DOCUMENT START/END.
		if (k->is_action("ui_text_caret_document_start", true)) { // && shift_pressed) {
			_move_cursor_document_start(shift_pressed);
			accept_event();
			return;
		}
		if (k->is_action("ui_text_caret_document_end", true)) { // && shift_pressed) {
			_move_cursor_document_end(shift_pressed);
			accept_event();
			return;
		}

		// CURSOR MOVEMENT - LINE START/END.
		if (k->is_action("ui_text_caret_line_start", true)) {
			_move_cursor_to_line_start(shift_pressed);
			accept_event();
			return;
		}
		if (k->is_action("ui_text_caret_line_end", true)) {
			_move_cursor_to_line_end(shift_pressed);
			accept_event();
			return;
		}

		// CURSOR MOVEMENT - PAGE UP/DOWN.
		if (k->is_action("ui_text_caret_page_up", true)) {
			_move_cursor_page_up(shift_pressed);
			accept_event();
			return;
		}
		if (k->is_action("ui_text_caret_page_down", true)) {
			_move_cursor_page_down(shift_pressed);
			accept_event();
			return;
		}

		if (allow_unicode_handling && !readonly && k->get_unicode() >= 32) {
			// Handle Unicode (if no modifiers active).
			_handle_unicode_character(k->get_unicode(), had_selection, false);
			accept_event();
			return;
		}
	}
}

void TextEdit::_scroll_up(real_t p_delta) {
	if (scrolling && smooth_scroll_enabled && SGN(target_v_scroll - v_scroll->get_value()) != SGN(-p_delta)) {
		scrolling = false;
		minimap_clicked = false;
	}

	if (scrolling) {
		target_v_scroll = (target_v_scroll - p_delta);
	} else {
		target_v_scroll = (get_v_scroll() - p_delta);
	}

	if (smooth_scroll_enabled) {
		if (target_v_scroll <= 0) {
			target_v_scroll = 0;
		}
		if (Math::abs(target_v_scroll - v_scroll->get_value()) < 1.0) {
			v_scroll->set_value(target_v_scroll);
		} else {
			scrolling = true;
			set_physics_process_internal(true);
		}
	} else {
		set_v_scroll(target_v_scroll);
	}
}

void TextEdit::_scroll_down(real_t p_delta) {
	if (scrolling && smooth_scroll_enabled && SGN(target_v_scroll - v_scroll->get_value()) != SGN(p_delta)) {
		scrolling = false;
		minimap_clicked = false;
	}

	if (scrolling) {
		target_v_scroll = (target_v_scroll + p_delta);
	} else {
		target_v_scroll = (get_v_scroll() + p_delta);
	}

	if (smooth_scroll_enabled) {
		int max_v_scroll = round(v_scroll->get_max() - v_scroll->get_page());
		if (target_v_scroll > max_v_scroll) {
			target_v_scroll = max_v_scroll;
		}
		if (Math::abs(target_v_scroll - v_scroll->get_value()) < 1.0) {
			v_scroll->set_value(target_v_scroll);
		} else {
			scrolling = true;
			set_physics_process_internal(true);
		}
	} else {
		set_v_scroll(target_v_scroll);
	}
}

void TextEdit::_pre_shift_selection() {
	if (!selection.active || selection.selecting_mode == SelectionMode::SELECTION_MODE_NONE) {
		selection.selecting_line = cursor.line;
		selection.selecting_column = cursor.column;
		selection.active = true;
	}

	selection.selecting_mode = SelectionMode::SELECTION_MODE_SHIFT;
}

void TextEdit::_post_shift_selection() {
	if (selection.active && selection.selecting_mode == SelectionMode::SELECTION_MODE_SHIFT) {
		select(selection.selecting_line, selection.selecting_column, cursor.line, cursor.column);
		update();
	}

	selection.selecting_text = true;
}

void TextEdit::_scroll_lines_up() {
	scrolling = false;
	minimap_clicked = false;

	// Adjust the vertical scroll.
	set_v_scroll(get_v_scroll() - 1);

	// Adjust the cursor to viewport.
	if (!selection.active) {
		int cur_line = cursor.line;
		int cur_wrap = get_cursor_wrap_index();
		int last_vis_line = get_last_full_visible_line();
		int last_vis_wrap = get_last_full_visible_line_wrap_index();

		if (cur_line > last_vis_line || (cur_line == last_vis_line && cur_wrap > last_vis_wrap)) {
			cursor_set_line(last_vis_line, false, false, last_vis_wrap);
		}
	}
}

void TextEdit::_scroll_lines_down() {
	scrolling = false;
	minimap_clicked = false;

	// Adjust the vertical scroll.
	set_v_scroll(get_v_scroll() + 1);

	// Adjust the cursor to viewport.
	if (!selection.active) {
		int cur_line = cursor.line;
		int cur_wrap = get_cursor_wrap_index();
		int first_vis_line = get_first_visible_line();
		int first_vis_wrap = cursor.wrap_ofs;

		if (cur_line < first_vis_line || (cur_line == first_vis_line && cur_wrap < first_vis_wrap)) {
			cursor_set_line(first_vis_line, false, false, first_vis_wrap);
		}
	}
}

/**** TEXT EDIT CORE API ****/

void TextEdit::_base_insert_text(int p_line, int p_char, const String &p_text, int &r_end_line, int &r_end_column) {
	// Save for undo.
	ERR_FAIL_INDEX(p_line, text.size());
	ERR_FAIL_COND(p_char < 0);

	/* STEP 1: Remove \r from source text and separate in substrings. */

	Vector<String> substrings = p_text.replace("\r", "").split("\n");

	// Is this just a new empty line?
	bool shift_first_line = p_char == 0 && p_text.replace("\r", "") == "\n";

	/* STEP 2: Add spaces if the char is greater than the end of the line. */
	while (p_char > text[p_line].length()) {
		text.set(p_line, text[p_line] + String::chr(' '), structured_text_parser(st_parser, st_args, text[p_line] + String::chr(' ')));
	}

	/* STEP 3: Separate dest string in pre and post text. */

	String preinsert_text = text[p_line].substr(0, p_char);
	String postinsert_text = text[p_line].substr(p_char, text[p_line].size());

	for (int j = 0; j < substrings.size(); j++) {
		// Insert the substrings.

		if (j == 0) {
			text.set(p_line, preinsert_text + substrings[j], structured_text_parser(st_parser, st_args, preinsert_text + substrings[j]));
		} else {
			text.insert(p_line + j, substrings[j], structured_text_parser(st_parser, st_args, substrings[j]));
		}

		if (j == substrings.size() - 1) {
			text.set(p_line + j, text[p_line + j] + postinsert_text, structured_text_parser(st_parser, st_args, text[p_line + j] + postinsert_text));
		}
	}

	if (shift_first_line) {
		text.move_gutters(p_line, p_line + 1);
		text.set_hidden(p_line + 1, text.is_hidden(p_line));

		text.set_hidden(p_line, false);
	}

	text.invalidate_cache(p_line);

	r_end_line = p_line + substrings.size() - 1;
	r_end_column = text[r_end_line].length() - postinsert_text.length();

	TextServer::Direction dir = TS->shaped_text_get_dominant_direciton_in_range(text.get_line_data(r_end_line)->get_rid(), (r_end_line == p_line) ? cursor.column : 0, r_end_column);
	if (dir != TextServer::DIRECTION_AUTO) {
		input_direction = (TextDirection)dir;
	}

	if (!text_changed_dirty && !setting_text) {
		if (is_inside_tree()) {
			MessageQueue::get_singleton()->push_call(this, "_text_changed_emit");
		}
		text_changed_dirty = true;
	}
	emit_signal("lines_edited_from", p_line, r_end_line);
}

String TextEdit::_base_get_text(int p_from_line, int p_from_column, int p_to_line, int p_to_column) const {
	ERR_FAIL_INDEX_V(p_from_line, text.size(), String());
	ERR_FAIL_INDEX_V(p_from_column, text[p_from_line].length() + 1, String());
	ERR_FAIL_INDEX_V(p_to_line, text.size(), String());
	ERR_FAIL_INDEX_V(p_to_column, text[p_to_line].length() + 1, String());
	ERR_FAIL_COND_V(p_to_line < p_from_line, String()); // 'from > to'.
	ERR_FAIL_COND_V(p_to_line == p_from_line && p_to_column < p_from_column, String()); // 'from > to'.

	String ret;

	for (int i = p_from_line; i <= p_to_line; i++) {
		int begin = (i == p_from_line) ? p_from_column : 0;
		int end = (i == p_to_line) ? p_to_column : text[i].length();

		if (i > p_from_line) {
			ret += "\n";
		}
		ret += text[i].substr(begin, end - begin);
	}

	return ret;
}

void TextEdit::_base_remove_text(int p_from_line, int p_from_column, int p_to_line, int p_to_column) {
	ERR_FAIL_INDEX(p_from_line, text.size());
	ERR_FAIL_INDEX(p_from_column, text[p_from_line].length() + 1);
	ERR_FAIL_INDEX(p_to_line, text.size());
	ERR_FAIL_INDEX(p_to_column, text[p_to_line].length() + 1);
	ERR_FAIL_COND(p_to_line < p_from_line); // 'from > to'.
	ERR_FAIL_COND(p_to_line == p_from_line && p_to_column < p_from_column); // 'from > to'.

	String pre_text = text[p_from_line].substr(0, p_from_column);
	String post_text = text[p_to_line].substr(p_to_column, text[p_to_line].length());

	for (int i = p_from_line; i < p_to_line; i++) {
		text.remove(p_from_line + 1);
	}
	text.set(p_from_line, pre_text + post_text, structured_text_parser(st_parser, st_args, pre_text + post_text));

	//text.set_line_wrap_amount(p_from_line, -1);
	text.invalidate_cache(p_from_line);

	if (!text_changed_dirty && !setting_text) {
		if (is_inside_tree()) {
			MessageQueue::get_singleton()->push_call(this, "_text_changed_emit");
		}
		text_changed_dirty = true;
	}
	emit_signal("lines_edited_from", p_to_line, p_from_line);
}

void TextEdit::_insert_text(int p_line, int p_char, const String &p_text, int *r_end_line, int *r_end_char) {
	if (!setting_text && idle_detect->is_inside_tree()) {
		idle_detect->start();
	}

	if (undo_enabled) {
		_clear_redo();
	}

	int retline, retchar;
	_base_insert_text(p_line, p_char, p_text, retline, retchar);
	if (r_end_line) {
		*r_end_line = retline;
	}
	if (r_end_char) {
		*r_end_char = retchar;
	}

	if (!undo_enabled) {
		return;
	}

	/* UNDO!! */
	TextOperation op;
	op.type = TextOperation::TYPE_INSERT;
	op.from_line = p_line;
	op.from_column = p_char;
	op.to_line = retline;
	op.to_column = retchar;
	op.text = p_text;
	op.version = ++version;
	op.chain_forward = false;
	op.chain_backward = false;

	// See if it should just be set as current op.
	if (current_op.type != op.type) {
		op.prev_version = get_version();
		_push_current_op();
		current_op = op;

		return; // Set as current op, return.
	}
	// See if it can be merged.
	if (current_op.to_line != p_line || current_op.to_column != p_char) {
		op.prev_version = get_version();
		_push_current_op();
		current_op = op;
		return; // Set as current op, return.
	}
	// Merge current op.

	current_op.text += p_text;
	current_op.to_column = retchar;
	current_op.to_line = retline;
	current_op.version = op.version;
}

void TextEdit::_remove_text(int p_from_line, int p_from_column, int p_to_line, int p_to_column) {
	if (!setting_text && idle_detect->is_inside_tree()) {
		idle_detect->start();
	}

	String text;
	if (undo_enabled) {
		_clear_redo();
		text = _base_get_text(p_from_line, p_from_column, p_to_line, p_to_column);
	}

	_base_remove_text(p_from_line, p_from_column, p_to_line, p_to_column);

	if (!undo_enabled) {
		return;
	}

	/* UNDO! */
	TextOperation op;
	op.type = TextOperation::TYPE_REMOVE;
	op.from_line = p_from_line;
	op.from_column = p_from_column;
	op.to_line = p_to_line;
	op.to_column = p_to_column;
	op.text = text;
	op.version = ++version;
	op.chain_forward = false;
	op.chain_backward = false;

	// See if it should just be set as current op.
	if (current_op.type != op.type) {
		op.prev_version = get_version();
		_push_current_op();
		current_op = op;
		return; // Set as current op, return.
	}
	// See if it can be merged.
	if (current_op.from_line == p_to_line && current_op.from_column == p_to_column) {
		// Backspace or similar.
		current_op.text = text + current_op.text;
		current_op.from_line = p_from_line;
		current_op.from_column = p_from_column;
		return; // Update current op.
	}

	op.prev_version = get_version();
	_push_current_op();
	current_op = op;
}

void TextEdit::_insert_text_at_cursor(const String &p_text) {
	int new_column, new_line;
	_insert_text(cursor.line, cursor.column, p_text, &new_line, &new_column);
	_update_scrollbars();
	cursor_set_line(new_line, false);
	cursor_set_column(new_column);

	update();
}

int TextEdit::get_char_count() {
	int totalsize = 0;

	for (int i = 0; i < text.size(); i++) {
		if (i > 0) {
			totalsize++; // Include \n.
		}
		totalsize += text[i].length();
	}

	return totalsize; // Omit last \n.
}

Size2 TextEdit::get_minimum_size() const {
	return cache.style_normal->get_minimum_size();
}

int TextEdit::_get_control_height() const {
	int control_height = get_size().height;
	control_height -= cache.style_normal->get_minimum_size().height;
	if (h_scroll->is_visible_in_tree()) {
		control_height -= h_scroll->get_size().height;
	}
	return control_height;
}

int TextEdit::_get_menu_action_accelerator(const String &p_action) {
	const List<Ref<InputEvent>> *events = InputMap::get_singleton()->action_get_events(p_action);
	if (!events) {
		return 0;
	}

	// Use first event in the list for the accelerator.
	const List<Ref<InputEvent>>::Element *first_event = events->front();
	if (!first_event) {
		return 0;
	}

	const Ref<InputEventKey> event = first_event->get();
	if (event.is_null()) {
		return 0;
	}

	// Use physical keycode if non-zero
	if (event->get_physical_keycode() != 0) {
		return event->get_physical_keycode_with_modifiers();
	} else {
		return event->get_keycode_with_modifiers();
	}
}

void TextEdit::_generate_context_menu() {
	// Reorganize context menu.
	menu->clear();
	if (!readonly) {
		menu->add_item(RTR("Cut"), MENU_CUT, is_shortcut_keys_enabled() ? _get_menu_action_accelerator("ui_cut") : 0);
	}
	menu->add_item(RTR("Copy"), MENU_COPY, is_shortcut_keys_enabled() ? _get_menu_action_accelerator("ui_copy") : 0);
	if (!readonly) {
		menu->add_item(RTR("Paste"), MENU_PASTE, is_shortcut_keys_enabled() ? _get_menu_action_accelerator("ui_paste") : 0);
	}
	menu->add_separator();
	if (is_selecting_enabled()) {
		menu->add_item(RTR("Select All"), MENU_SELECT_ALL, is_shortcut_keys_enabled() ? _get_menu_action_accelerator("ui_text_select_all") : 0);
	}
	if (!readonly) {
		menu->add_item(RTR("Clear"), MENU_CLEAR);
		menu->add_separator();
		menu->add_item(RTR("Undo"), MENU_UNDO, is_shortcut_keys_enabled() ? _get_menu_action_accelerator("ui_undo") : 0);
		menu->add_item(RTR("Redo"), MENU_REDO, is_shortcut_keys_enabled() ? _get_menu_action_accelerator("ui_redo") : 0);
	}
	menu->add_separator();
	menu->add_submenu_item(RTR("Text writing direction"), "DirMenu");
	menu->add_separator();
	menu->add_check_item(RTR("Display control characters"), MENU_DISPLAY_UCC);
	if (!readonly) {
		menu->add_submenu_item(RTR("Insert control character"), "CTLMenu");
	}
}

int TextEdit::get_visible_rows() const {
	return _get_control_height() / get_row_height();
}

int TextEdit::_get_minimap_visible_rows() const {
	return _get_control_height() / (minimap_char_size.y + minimap_line_spacing);
}

int TextEdit::get_total_visible_rows() const {
	// Returns the total amount of rows we need in the editor.
	// This skips hidden lines and counts each wrapping of a line.
	if (!is_hiding_enabled() && !is_wrap_enabled()) {
		return text.size();
	}

	int total_rows = 0;
	for (int i = 0; i < text.size(); i++) {
		if (!text.is_hidden(i)) {
			total_rows++;
			total_rows += times_line_wraps(i);
		}
	}
	return total_rows;
}

void TextEdit::_update_wrap_at(bool p_force) {
	int new_wrap_at = get_size().width - cache.style_normal->get_minimum_size().width - gutters_width - gutter_padding;
	if (draw_minimap) {
		new_wrap_at -= minimap_width;
	}
	if (v_scroll->is_visible_in_tree()) {
		new_wrap_at -= v_scroll->get_combined_minimum_size().width;
	}
	new_wrap_at -= wrap_right_offset; // Give it a little more space.

	if ((wrap_at != new_wrap_at) || p_force) {
		wrap_at = new_wrap_at;
		if (wrap_enabled) {
			text.set_width(wrap_at);
		} else {
			text.set_width(-1);
		}
		text.invalidate_all_lines();
	}

	update_cursor_wrap_offset();
}

void TextEdit::adjust_viewport_to_cursor() {
	// Make sure cursor is visible on the screen.
	scrolling = false;
	minimap_clicked = false;

	int cur_line = cursor.line;
	int cur_wrap = get_cursor_wrap_index();

	int first_vis_line = get_first_visible_line();
	int first_vis_wrap = cursor.wrap_ofs;
	int last_vis_line = get_last_full_visible_line();
	int last_vis_wrap = get_last_full_visible_line_wrap_index();

	if (cur_line < first_vis_line || (cur_line == first_vis_line && cur_wrap < first_vis_wrap)) {
		// Cursor is above screen.
		set_line_as_first_visible(cur_line, cur_wrap);
	} else if (cur_line > last_vis_line || (cur_line == last_vis_line && cur_wrap > last_vis_wrap)) {
		// Cursor is below screen.
		set_line_as_last_visible(cur_line, cur_wrap);
	}

	int visible_width = get_size().width - cache.style_normal->get_minimum_size().width - gutters_width - gutter_padding - cache.minimap_width;
	if (v_scroll->is_visible_in_tree()) {
		visible_width -= v_scroll->get_combined_minimum_size().width;
	}
	visible_width -= 20; // Give it a little more space.

	if (!is_wrap_enabled()) {
		// Adjust x offset.
		Vector2i cursor_pos;

		// Get position of the start of caret.
		if (ime_text.length() != 0 && ime_selection.x != 0) {
			cursor_pos.x = get_column_x_offset_for_line(cursor.column + ime_selection.x, cursor.line);
		} else {
			cursor_pos.x = get_column_x_offset_for_line(cursor.column, cursor.line);
		}

		// Get position of the end of caret.
		if (ime_text.length() != 0) {
			if (ime_selection.y != 0) {
				cursor_pos.y = get_column_x_offset_for_line(cursor.column + ime_selection.x + ime_selection.y, cursor.line);
			} else {
				cursor_pos.y = get_column_x_offset_for_line(cursor.column + ime_text.size(), cursor.line);
			}
		} else {
			cursor_pos.y = cursor_pos.x;
		}

		if (MAX(cursor_pos.x, cursor_pos.y) > (cursor.x_ofs + visible_width)) {
			cursor.x_ofs = MAX(cursor_pos.x, cursor_pos.y) - visible_width + 1;
		}

		if (MIN(cursor_pos.x, cursor_pos.y) < cursor.x_ofs) {
			cursor.x_ofs = MIN(cursor_pos.x, cursor_pos.y);
		}
	} else {
		cursor.x_ofs = 0;
	}
	h_scroll->set_value(cursor.x_ofs);

	update();
}

void TextEdit::center_viewport_to_cursor() {
	// Move viewport so the cursor is in the center of the screen.
	scrolling = false;
	minimap_clicked = false;

	if (is_line_hidden(cursor.line)) {
		unfold_line(cursor.line);
	}

	set_line_as_center_visible(cursor.line, get_cursor_wrap_index());
	int visible_width = get_size().width - cache.style_normal->get_minimum_size().width - gutters_width - gutter_padding - cache.minimap_width;
	if (v_scroll->is_visible_in_tree()) {
		visible_width -= v_scroll->get_combined_minimum_size().width;
	}
	visible_width -= 20; // Give it a little more space.

	if (is_wrap_enabled()) {
		// Center x offset.

		Vector2i cursor_pos;

		// Get position of the start of caret.
		if (ime_text.length() != 0 && ime_selection.x != 0) {
			cursor_pos.x = get_column_x_offset_for_line(cursor.column + ime_selection.x, cursor.line);
		} else {
			cursor_pos.x = get_column_x_offset_for_line(cursor.column, cursor.line);
		}

		// Get position of the end of caret.
		if (ime_text.length() != 0) {
			if (ime_selection.y != 0) {
				cursor_pos.y = get_column_x_offset_for_line(cursor.column + ime_selection.x + ime_selection.y, cursor.line);
			} else {
				cursor_pos.y = get_column_x_offset_for_line(cursor.column + ime_text.size(), cursor.line);
			}
		} else {
			cursor_pos.y = cursor_pos.x;
		}

		if (MAX(cursor_pos.x, cursor_pos.y) > (cursor.x_ofs + visible_width)) {
			cursor.x_ofs = MAX(cursor_pos.x, cursor_pos.y) - visible_width + 1;
		}

		if (MIN(cursor_pos.x, cursor_pos.y) < cursor.x_ofs) {
			cursor.x_ofs = MIN(cursor_pos.x, cursor_pos.y);
		}
	} else {
		cursor.x_ofs = 0;
	}
	h_scroll->set_value(cursor.x_ofs);

	update();
}

void TextEdit::update_cursor_wrap_offset() {
	int first_vis_line = get_first_visible_line();
	if (line_wraps(first_vis_line)) {
		cursor.wrap_ofs = MIN(cursor.wrap_ofs, times_line_wraps(first_vis_line));
	} else {
		cursor.wrap_ofs = 0;
	}
	set_line_as_first_visible(cursor.line_ofs, cursor.wrap_ofs);
}

bool TextEdit::line_wraps(int line) const {
	ERR_FAIL_INDEX_V(line, text.size(), 0);
	if (!is_wrap_enabled()) {
		return false;
	}
	return text.get_line_wrap_amount(line) > 0;
}

int TextEdit::times_line_wraps(int line) const {
	ERR_FAIL_INDEX_V(line, text.size(), 0);

	if (!line_wraps(line)) {
		return 0;
	}

	return text.get_line_wrap_amount(line);
}

Vector<String> TextEdit::get_wrap_rows_text(int p_line) const {
	ERR_FAIL_INDEX_V(p_line, text.size(), Vector<String>());

	Vector<String> lines;
	if (!line_wraps(p_line)) {
		lines.push_back(text[p_line]);
		return lines;
	}

	const String &line_text = text[p_line];
	Vector<Vector2i> line_ranges = text.get_line_wrap_ranges(p_line);
	for (int i = 0; i < line_ranges.size(); i++) {
		lines.push_back(line_text.substr(line_ranges[i].x, line_ranges[i].y - line_ranges[i].x));
	}

	return lines;
}

int TextEdit::get_cursor_wrap_index() const {
	return get_line_wrap_index_at_col(cursor.line, cursor.column);
}

int TextEdit::get_line_wrap_index_at_col(int p_line, int p_column) const {
	ERR_FAIL_INDEX_V(p_line, text.size(), 0);

	if (!line_wraps(p_line)) {
		return 0;
	}

	// Loop through wraps in the line text until we get to the column.
	int wrap_index = 0;
	int col = 0;
	Vector<String> rows = get_wrap_rows_text(p_line);
	for (int i = 0; i < rows.size(); i++) {
		wrap_index = i;
		String s = rows[wrap_index];
		col += s.length();
		if (col > p_column) {
			break;
		}
	}
	return wrap_index;
}

void TextEdit::set_mid_grapheme_caret_enabled(const bool p_enabled) {
	mid_grapheme_caret_enabled = p_enabled;
}

bool TextEdit::get_mid_grapheme_caret_enabled() const {
	return mid_grapheme_caret_enabled;
}

void TextEdit::cursor_set_column(int p_col, bool p_adjust_viewport) {
	if (p_col < 0) {
		p_col = 0;
	}

	cursor.column = p_col;
	if (cursor.column > get_line(cursor.line).length()) {
		cursor.column = get_line(cursor.line).length();
	}

	cursor.last_fit_x = get_column_x_offset_for_line(cursor.column, cursor.line);

	if (p_adjust_viewport) {
		adjust_viewport_to_cursor();
	}

	if (!cursor_changed_dirty) {
		if (is_inside_tree()) {
			MessageQueue::get_singleton()->push_call(this, "_cursor_changed_emit");
		}
		cursor_changed_dirty = true;
	}
}

void TextEdit::cursor_set_line(int p_row, bool p_adjust_viewport, bool p_can_be_hidden, int p_wrap_index) {
	if (setting_row) {
		return;
	}

	setting_row = true;
	if (p_row < 0) {
		p_row = 0;
	}

	if (p_row >= text.size()) {
		p_row = text.size() - 1;
	}

	if (!p_can_be_hidden) {
		if (is_line_hidden(CLAMP(p_row, 0, text.size() - 1))) {
			int move_down = num_lines_from(p_row, 1) - 1;
			if (p_row + move_down <= text.size() - 1 && !is_line_hidden(p_row + move_down)) {
				p_row += move_down;
			} else {
				int move_up = num_lines_from(p_row, -1) - 1;
				if (p_row - move_up > 0 && !is_line_hidden(p_row - move_up)) {
					p_row -= move_up;
				} else {
					WARN_PRINT(("Cursor set to hidden line " + itos(p_row) + " and there are no nonhidden lines."));
				}
			}
		}
	}
	cursor.line = p_row;

	int n_col = get_char_pos_for_line(cursor.last_fit_x, p_row, p_wrap_index);
	if (n_col != 0 && is_wrap_enabled() && p_wrap_index < times_line_wraps(p_row)) {
		Vector<String> rows = get_wrap_rows_text(p_row);
		int row_end_col = 0;
		for (int i = 0; i < p_wrap_index + 1; i++) {
			row_end_col += rows[i].length();
		}
		if (n_col >= row_end_col) {
			n_col -= 1;
		}
	}
	cursor.column = n_col;

	if (p_adjust_viewport) {
		adjust_viewport_to_cursor();
	}

	setting_row = false;

	if (!cursor_changed_dirty) {
		if (is_inside_tree()) {
			MessageQueue::get_singleton()->push_call(this, "_cursor_changed_emit");
		}
		cursor_changed_dirty = true;
	}
}

int TextEdit::cursor_get_column() const {
	return cursor.column;
}

int TextEdit::cursor_get_line() const {
	return cursor.line;
}

bool TextEdit::cursor_get_blink_enabled() const {
	return caret_blink_enabled;
}

void TextEdit::cursor_set_blink_enabled(const bool p_enabled) {
	caret_blink_enabled = p_enabled;

	if (has_focus()) {
		if (p_enabled) {
			caret_blink_timer->start();
		} else {
			caret_blink_timer->stop();
		}
	}

	draw_caret = true;
}

float TextEdit::cursor_get_blink_speed() const {
	return caret_blink_timer->get_wait_time();
}

void TextEdit::cursor_set_blink_speed(const float p_speed) {
	ERR_FAIL_COND(p_speed <= 0);
	caret_blink_timer->set_wait_time(p_speed);
}

void TextEdit::cursor_set_block_mode(const bool p_enable) {
	block_caret = p_enable;
	update();
}

bool TextEdit::cursor_is_block_mode() const {
	return block_caret;
}

void TextEdit::set_right_click_moves_caret(bool p_enable) {
	right_click_moves_caret = p_enable;
}

bool TextEdit::is_right_click_moving_caret() const {
	return right_click_moves_caret;
}

TextEdit::SelectionMode TextEdit::get_selection_mode() const {
	return selection.selecting_mode;
}

void TextEdit::set_selection_mode(SelectionMode p_mode, int p_line, int p_column) {
	selection.selecting_mode = p_mode;
	if (p_line >= 0) {
		ERR_FAIL_INDEX(p_line, text.size());
		selection.selecting_line = p_line;
	}
	if (p_column >= 0) {
		ERR_FAIL_INDEX(p_column, text[selection.selecting_line].length());
		selection.selecting_column = p_column;
	}
}

int TextEdit::get_selection_line() const {
	return selection.selecting_line;
};

int TextEdit::get_selection_column() const {
	return selection.selecting_column;
};

void TextEdit::_v_scroll_input() {
	scrolling = false;
	minimap_clicked = false;
}

void TextEdit::_scroll_moved(double p_to_val) {
	if (updating_scrolls) {
		return;
	}

	if (h_scroll->is_visible_in_tree()) {
		cursor.x_ofs = h_scroll->get_value();
	}
	if (v_scroll->is_visible_in_tree()) {
		// Set line ofs and wrap ofs.
		int v_scroll_i = floor(get_v_scroll());
		int sc = 0;
		int n_line;
		for (n_line = 0; n_line < text.size(); n_line++) {
			if (!is_line_hidden(n_line)) {
				sc++;
				sc += times_line_wraps(n_line);
				if (sc > v_scroll_i) {
					break;
				}
			}
		}
		n_line = MIN(n_line, text.size() - 1);
		int line_wrap_amount = times_line_wraps(n_line);
		int wi = line_wrap_amount - (sc - v_scroll_i - 1);
		wi = CLAMP(wi, 0, line_wrap_amount);

		cursor.line_ofs = n_line;
		cursor.wrap_ofs = wi;
	}
	update();
}

int TextEdit::get_row_height() const {
	int height = cache.font->get_height(cache.font_size);
	for (int i = 0; i < text.size(); i++) {
		for (int j = 0; j <= text.get_line_wrap_amount(i); j++) {
			height = MAX(height, text.get_line_height(i, j));
		}
	}
	return height + cache.line_spacing;
}

int TextEdit::get_char_pos_for_line(int p_px, int p_line, int p_wrap_index) const {
	ERR_FAIL_INDEX_V(p_line, text.size(), 0);
	p_wrap_index = MIN(p_wrap_index, text.get_line_data(p_line)->get_line_count() - 1);

	RID text_rid = text.get_line_data(p_line)->get_line_rid(p_wrap_index);
	if (is_layout_rtl()) {
		p_px = TS->shaped_text_get_size(text_rid).x - p_px;
	}
	return TS->shaped_text_hit_test_position(text_rid, p_px);
}

int TextEdit::get_column_x_offset_for_line(int p_char, int p_line) const {
	ERR_FAIL_INDEX_V(p_line, text.size(), 0);

	int row = 0;
	Vector<Vector2i> rows2 = text.get_line_wrap_ranges(p_line);
	for (int i = 0; i < rows2.size(); i++) {
		if ((p_char >= rows2[i].x) && (p_char < rows2[i].y)) {
			row = i;
			break;
		}
	}

	Rect2 l_caret, t_caret;
	TextServer::Direction l_dir, t_dir;
	RID text_rid = text.get_line_data(p_line)->get_line_rid(row);
	TS->shaped_text_get_carets(text_rid, cursor.column, l_caret, l_dir, t_caret, t_dir);
	if ((l_caret != Rect2() && (l_dir == TextServer::DIRECTION_AUTO || l_dir == (TextServer::Direction)input_direction)) || (t_caret == Rect2())) {
		return l_caret.position.x;
	} else {
		return t_caret.position.x;
	}
}

void TextEdit::insert_text_at_cursor(const String &p_text) {
	if (selection.active) {
		cursor_set_line(selection.from_line, false);
		cursor_set_column(selection.from_column);

		_remove_text(selection.from_line, selection.from_column, selection.to_line, selection.to_column);
		selection.active = false;
		selection.selecting_mode = SelectionMode::SELECTION_MODE_NONE;
	}

	_insert_text_at_cursor(p_text);
	update();
}

Control::CursorShape TextEdit::get_cursor_shape(const Point2 &p_pos) const {
	if (highlighted_word != String()) {
		return CURSOR_POINTING_HAND;
	}

	if ((completion_active && completion_rect.has_point(p_pos)) || (is_readonly() && (!is_selecting_enabled() || text.size() == 0))) {
		return CURSOR_ARROW;
	}

	int row, col;
	_get_mouse_pos(p_pos, row, col);

	int left_margin = cache.style_normal->get_margin(SIDE_LEFT);
	int gutter = left_margin + gutters_width;
	if (p_pos.x < gutter) {
		for (int i = 0; i < gutters.size(); i++) {
			if (!gutters[i].draw) {
				continue;
			}

			if (p_pos.x > left_margin && p_pos.x <= (left_margin + gutters[i].width) - 3) {
				if (gutters[i].clickable || is_line_gutter_clickable(row, i)) {
					return CURSOR_POINTING_HAND;
				}
			}
			left_margin += gutters[i].width;
		}
		return CURSOR_ARROW;
	}

	int xmargin_end = get_size().width - cache.style_normal->get_margin(SIDE_RIGHT);
	if (draw_minimap && p_pos.x > xmargin_end - minimap_width && p_pos.x <= xmargin_end) {
		return CURSOR_ARROW;
	}

	// EOL fold icon.
	if (is_folded(row)) {
		gutter += gutter_padding + text.get_line_width(row) - cursor.x_ofs;
		if (p_pos.x > gutter - 3 && p_pos.x <= gutter + cache.folded_eol_icon->get_width() + 3) {
			return CURSOR_POINTING_HAND;
		}
	}

	return get_default_cursor_shape();
}

void TextEdit::set_text(String p_text) {
	setting_text = true;
	if (!undo_enabled) {
		_clear();
		_insert_text_at_cursor(p_text);
	}

	if (undo_enabled) {
		cursor_set_line(0);
		cursor_set_column(0);

		begin_complex_operation();
		_remove_text(0, 0, MAX(0, get_line_count() - 1), MAX(get_line(MAX(get_line_count() - 1, 0)).size() - 1, 0));
		_insert_text_at_cursor(p_text);
		end_complex_operation();
		selection.active = false;
	}

	cursor_set_line(0);
	cursor_set_column(0);

	update();
	setting_text = false;
}

String TextEdit::get_text() {
	String longthing;
	int len = text.size();
	for (int i = 0; i < len; i++) {
		longthing += text[i];
		if (i != len - 1) {
			longthing += "\n";
		}
	}

	return longthing;
}

void TextEdit::set_structured_text_bidi_override(Control::StructuredTextParser p_parser) {
	if (st_parser != p_parser) {
		st_parser = p_parser;
		for (int i = 0; i < text.size(); i++) {
			text.set(i, text[i], structured_text_parser(st_parser, st_args, text[i]));
		}
		update();
	}
}

Control::StructuredTextParser TextEdit::get_structured_text_bidi_override() const {
	return st_parser;
}

void TextEdit::set_structured_text_bidi_override_options(Array p_args) {
	st_args = p_args;
	for (int i = 0; i < text.size(); i++) {
		text.set(i, text[i], structured_text_parser(st_parser, st_args, text[i]));
	}
	update();
}

Array TextEdit::get_structured_text_bidi_override_options() const {
	return st_args;
}

void TextEdit::set_text_direction(Control::TextDirection p_text_direction) {
	ERR_FAIL_COND((int)p_text_direction < -1 || (int)p_text_direction > 3);
	if (text_direction != p_text_direction) {
		text_direction = p_text_direction;
		if (text_direction != TEXT_DIRECTION_AUTO && text_direction != TEXT_DIRECTION_INHERITED) {
			input_direction = text_direction;
		}
		TextServer::Direction dir;
		if (text_direction == Control::TEXT_DIRECTION_INHERITED) {
			dir = is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR;
		} else {
			dir = (TextServer::Direction)text_direction;
		}
		text.set_direction_and_language(dir, (language != "") ? language : TranslationServer::get_singleton()->get_tool_locale());
		text.invalidate_all();

		menu_dir->set_item_checked(menu_dir->get_item_index(MENU_DIR_INHERITED), text_direction == TEXT_DIRECTION_INHERITED);
		menu_dir->set_item_checked(menu_dir->get_item_index(MENU_DIR_AUTO), text_direction == TEXT_DIRECTION_AUTO);
		menu_dir->set_item_checked(menu_dir->get_item_index(MENU_DIR_LTR), text_direction == TEXT_DIRECTION_LTR);
		menu_dir->set_item_checked(menu_dir->get_item_index(MENU_DIR_RTL), text_direction == TEXT_DIRECTION_RTL);
		update();
	}
}

Control::TextDirection TextEdit::get_text_direction() const {
	return text_direction;
}

void TextEdit::clear_opentype_features() {
	opentype_features.clear();
	text.set_font_features(opentype_features);
	text.invalidate_all();
	update();
}

void TextEdit::set_opentype_feature(const String &p_name, int p_value) {
	int32_t tag = TS->name_to_tag(p_name);
	if (!opentype_features.has(tag) || (int)opentype_features[tag] != p_value) {
		opentype_features[tag] = p_value;
		text.set_font_features(opentype_features);
		text.invalidate_all();
		update();
	}
}

int TextEdit::get_opentype_feature(const String &p_name) const {
	int32_t tag = TS->name_to_tag(p_name);
	if (!opentype_features.has(tag)) {
		return -1;
	}
	return opentype_features[tag];
}

void TextEdit::set_language(const String &p_language) {
	if (language != p_language) {
		language = p_language;
		TextServer::Direction dir;
		if (text_direction == Control::TEXT_DIRECTION_INHERITED) {
			dir = is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR;
		} else {
			dir = (TextServer::Direction)text_direction;
		}
		text.set_direction_and_language(dir, (language != "") ? language : TranslationServer::get_singleton()->get_tool_locale());
		text.invalidate_all();
		update();
	}
}

String TextEdit::get_language() const {
	return language;
}

void TextEdit::set_draw_control_chars(bool p_draw_control_chars) {
	if (draw_control_chars != p_draw_control_chars) {
		draw_control_chars = p_draw_control_chars;
		menu->set_item_checked(menu->get_item_index(MENU_DISPLAY_UCC), draw_control_chars);
		text.set_draw_control_chars(draw_control_chars);
		text.invalidate_all();
		update();
	}
}

bool TextEdit::get_draw_control_chars() const {
	return draw_control_chars;
}

String TextEdit::get_text_for_lookup_completion() {
	int row, col;
	Point2i mp = _get_local_mouse_pos();
	_get_mouse_pos(mp, row, col);

	String longthing;
	int len = text.size();
	for (int i = 0; i < len; i++) {
		if (i == row) {
			longthing += text[i].substr(0, col);
			longthing += String::chr(0xFFFF); // Not unicode, represents the cursor.
			longthing += text[i].substr(col, text[i].size());
		} else {
			longthing += text[i];
		}

		if (i != len - 1) {
			longthing += "\n";
		}
	}

	return longthing;
}

String TextEdit::get_text_for_completion() {
	String longthing;
	int len = text.size();
	for (int i = 0; i < len; i++) {
		if (i == cursor.line) {
			longthing += text[i].substr(0, cursor.column);
			longthing += String::chr(0xFFFF); // Not unicode, represents the cursor.
			longthing += text[i].substr(cursor.column, text[i].size());
		} else {
			longthing += text[i];
		}

		if (i != len - 1) {
			longthing += "\n";
		}
	}

	return longthing;
};

String TextEdit::get_line(int line) const {
	if (line < 0 || line >= text.size()) {
		return "";
	}

	return text[line];
};

void TextEdit::_clear() {
	clear_undo_history();
	text.clear();
	cursor.column = 0;
	cursor.line = 0;
	cursor.x_ofs = 0;
	cursor.line_ofs = 0;
	cursor.wrap_ofs = 0;
	cursor.last_fit_x = 0;
	selection.active = false;
}

void TextEdit::clear() {
	setting_text = true;
	_clear();
	setting_text = false;
};

void TextEdit::set_readonly(bool p_readonly) {
	if (readonly == p_readonly) {
		return;
	}

	readonly = p_readonly;
	_generate_context_menu();

	update();
}

bool TextEdit::is_readonly() const {
	return readonly;
}

void TextEdit::set_wrap_enabled(bool p_wrap_enabled) {
	if (wrap_enabled != p_wrap_enabled) {
		wrap_enabled = p_wrap_enabled;
		_update_wrap_at(true);
	}
}

bool TextEdit::is_wrap_enabled() const {
	return wrap_enabled;
}

void TextEdit::set_max_chars(int p_max_chars) {
	max_chars = p_max_chars;
}

int TextEdit::get_max_chars() const {
	return max_chars;
}

void TextEdit::_reset_caret_blink_timer() {
	if (caret_blink_enabled) {
		draw_caret = true;
		if (has_focus()) {
			caret_blink_timer->stop();
			caret_blink_timer->start();
			update();
		}
	}
}

void TextEdit::_toggle_draw_caret() {
	draw_caret = !draw_caret;
	if (is_visible_in_tree() && has_focus() && window_has_focus) {
		update();
	}
}

void TextEdit::_update_caches() {
	cache.style_normal = get_theme_stylebox("normal");
	cache.style_focus = get_theme_stylebox("focus");
	cache.style_readonly = get_theme_stylebox("read_only");
	cache.completion_background_color = get_theme_color("completion_background_color");
	cache.completion_selected_color = get_theme_color("completion_selected_color");
	cache.completion_existing_color = get_theme_color("completion_existing_color");
	cache.completion_font_color = get_theme_color("completion_font_color");
	cache.font = get_theme_font("font");
	cache.font_size = get_theme_font_size("font_size");
	cache.outline_color = get_theme_color("font_outline_color");
	cache.outline_size = get_theme_constant("outline_size");
	cache.caret_color = get_theme_color("caret_color");
	cache.caret_background_color = get_theme_color("caret_background_color");
	cache.font_color = get_theme_color("font_color");
	cache.font_selected_color = get_theme_color("font_selected_color");
	cache.font_readonly_color = get_theme_color("font_readonly_color");
	cache.selection_color = get_theme_color("selection_color");
	cache.current_line_color = get_theme_color("current_line_color");
	cache.line_length_guideline_color = get_theme_color("line_length_guideline_color");
	cache.code_folding_color = get_theme_color("code_folding_color");
	cache.brace_mismatch_color = get_theme_color("brace_mismatch_color");
	cache.word_highlighted_color = get_theme_color("word_highlighted_color");
	cache.search_result_color = get_theme_color("search_result_color");
	cache.search_result_border_color = get_theme_color("search_result_border_color");
	cache.background_color = get_theme_color("background_color");
#ifdef TOOLS_ENABLED
	cache.line_spacing = get_theme_constant("line_spacing") * EDSCALE;
#else
	cache.line_spacing = get_theme_constant("line_spacing");
#endif
	cache.tab_icon = get_theme_icon("tab");
	cache.space_icon = get_theme_icon("space");
	cache.folded_eol_icon = get_theme_icon("GuiEllipsis", "EditorIcons");

	TextServer::Direction dir;
	if (text_direction == Control::TEXT_DIRECTION_INHERITED) {
		dir = is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR;
	} else {
		dir = (TextServer::Direction)text_direction;
	}
	text.set_direction_and_language(dir, (language != "") ? language : TranslationServer::get_singleton()->get_tool_locale());
	text.set_font_features(opentype_features);
	text.set_draw_control_chars(draw_control_chars);
	text.set_font(cache.font);
	text.set_font_size(cache.font_size);
	text.invalidate_all();

	if (syntax_highlighter.is_valid()) {
		syntax_highlighter->set_text_edit(this);
	}
}

/* Syntax Highlighting. */
Ref<SyntaxHighlighter> TextEdit::get_syntax_highlighter() {
	return syntax_highlighter;
}

void TextEdit::set_syntax_highlighter(Ref<SyntaxHighlighter> p_syntax_highlighter) {
	syntax_highlighter = p_syntax_highlighter;
	if (syntax_highlighter.is_valid()) {
		syntax_highlighter->set_text_edit(this);
	}
	update();
}

/* Gutters. */
void TextEdit::_update_gutter_width() {
	gutters_width = 0;
	for (int i = 0; i < gutters.size(); i++) {
		if (gutters[i].draw) {
			gutters_width += gutters[i].width;
		}
	}
	if (gutters_width > 0) {
		gutter_padding = 2;
	}
	update();
}

void TextEdit::add_gutter(int p_at) {
	if (p_at < 0 || p_at > gutters.size()) {
		gutters.push_back(GutterInfo());
	} else {
		gutters.insert(p_at, GutterInfo());
	}

	for (int i = 0; i < text.size() + 1; i++) {
		text.add_gutter(p_at);
	}
	emit_signal("gutter_added");
	update();
}

void TextEdit::remove_gutter(int p_gutter) {
	ERR_FAIL_INDEX(p_gutter, gutters.size());

	gutters.remove(p_gutter);

	for (int i = 0; i < text.size() + 1; i++) {
		text.remove_gutter(p_gutter);
	}
	emit_signal("gutter_removed");
	update();
}

int TextEdit::get_gutter_count() const {
	return gutters.size();
}

void TextEdit::set_gutter_name(int p_gutter, const String &p_name) {
	ERR_FAIL_INDEX(p_gutter, gutters.size());
	gutters.write[p_gutter].name = p_name;
}

String TextEdit::get_gutter_name(int p_gutter) const {
	ERR_FAIL_INDEX_V(p_gutter, gutters.size(), "");
	return gutters[p_gutter].name;
}

void TextEdit::set_gutter_type(int p_gutter, GutterType p_type) {
	ERR_FAIL_INDEX(p_gutter, gutters.size());
	gutters.write[p_gutter].type = p_type;
	update();
}

TextEdit::GutterType TextEdit::get_gutter_type(int p_gutter) const {
	ERR_FAIL_INDEX_V(p_gutter, gutters.size(), GUTTER_TYPE_STRING);
	return gutters[p_gutter].type;
}

void TextEdit::set_gutter_width(int p_gutter, int p_width) {
	ERR_FAIL_INDEX(p_gutter, gutters.size());
	gutters.write[p_gutter].width = p_width;
	_update_gutter_width();
}

int TextEdit::get_gutter_width(int p_gutter) const {
	ERR_FAIL_INDEX_V(p_gutter, gutters.size(), -1);
	return gutters[p_gutter].width;
}

void TextEdit::set_gutter_draw(int p_gutter, bool p_draw) {
	ERR_FAIL_INDEX(p_gutter, gutters.size());
	gutters.write[p_gutter].draw = p_draw;
	_update_gutter_width();
}

bool TextEdit::is_gutter_drawn(int p_gutter) const {
	ERR_FAIL_INDEX_V(p_gutter, gutters.size(), false);
	return gutters[p_gutter].draw;
}

void TextEdit::set_gutter_clickable(int p_gutter, bool p_clickable) {
	ERR_FAIL_INDEX(p_gutter, gutters.size());
	gutters.write[p_gutter].clickable = p_clickable;
	update();
}

bool TextEdit::is_gutter_clickable(int p_gutter) const {
	ERR_FAIL_INDEX_V(p_gutter, gutters.size(), false);
	return gutters[p_gutter].clickable;
}

void TextEdit::set_gutter_overwritable(int p_gutter, bool p_overwritable) {
	ERR_FAIL_INDEX(p_gutter, gutters.size());
	gutters.write[p_gutter].overwritable = p_overwritable;
}

bool TextEdit::is_gutter_overwritable(int p_gutter) const {
	ERR_FAIL_INDEX_V(p_gutter, gutters.size(), false);
	return gutters[p_gutter].overwritable;
}

void TextEdit::set_gutter_custom_draw(int p_gutter, Object *p_object, const StringName &p_callback) {
	ERR_FAIL_INDEX(p_gutter, gutters.size());
	ERR_FAIL_NULL(p_object);

	gutters.write[p_gutter].custom_draw_obj = p_object->get_instance_id();
	gutters.write[p_gutter].custom_draw_callback = p_callback;
	update();
}

// Line gutters.
void TextEdit::set_line_gutter_metadata(int p_line, int p_gutter, const Variant &p_metadata) {
	ERR_FAIL_INDEX(p_line, text.size());
	ERR_FAIL_INDEX(p_gutter, gutters.size());
	text.set_line_gutter_metadata(p_line, p_gutter, p_metadata);
}

Variant TextEdit::get_line_gutter_metadata(int p_line, int p_gutter) const {
	ERR_FAIL_INDEX_V(p_line, text.size(), "");
	ERR_FAIL_INDEX_V(p_gutter, gutters.size(), "");
	return text.get_line_gutter_metadata(p_line, p_gutter);
}

void TextEdit::set_line_gutter_text(int p_line, int p_gutter, const String &p_text) {
	ERR_FAIL_INDEX(p_line, text.size());
	ERR_FAIL_INDEX(p_gutter, gutters.size());
	text.set_line_gutter_text(p_line, p_gutter, p_text);
	update();
}

String TextEdit::get_line_gutter_text(int p_line, int p_gutter) const {
	ERR_FAIL_INDEX_V(p_line, text.size(), "");
	ERR_FAIL_INDEX_V(p_gutter, gutters.size(), "");
	return text.get_line_gutter_text(p_line, p_gutter);
}

void TextEdit::set_line_gutter_icon(int p_line, int p_gutter, Ref<Texture2D> p_icon) {
	ERR_FAIL_INDEX(p_line, text.size());
	ERR_FAIL_INDEX(p_gutter, gutters.size());
	text.set_line_gutter_icon(p_line, p_gutter, p_icon);
	update();
}

Ref<Texture2D> TextEdit::get_line_gutter_icon(int p_line, int p_gutter) const {
	ERR_FAIL_INDEX_V(p_line, text.size(), Ref<Texture2D>());
	ERR_FAIL_INDEX_V(p_gutter, gutters.size(), Ref<Texture2D>());
	return text.get_line_gutter_icon(p_line, p_gutter);
}

void TextEdit::set_line_gutter_item_color(int p_line, int p_gutter, const Color &p_color) {
	ERR_FAIL_INDEX(p_line, text.size());
	ERR_FAIL_INDEX(p_gutter, gutters.size());
	text.set_line_gutter_item_color(p_line, p_gutter, p_color);
	update();
}

Color TextEdit::get_line_gutter_item_color(int p_line, int p_gutter) {
	ERR_FAIL_INDEX_V(p_line, text.size(), Color());
	ERR_FAIL_INDEX_V(p_gutter, gutters.size(), Color());
	return text.get_line_gutter_item_color(p_line, p_gutter);
}

void TextEdit::set_line_gutter_clickable(int p_line, int p_gutter, bool p_clickable) {
	ERR_FAIL_INDEX(p_line, text.size());
	ERR_FAIL_INDEX(p_gutter, gutters.size());
	text.set_line_gutter_clickable(p_line, p_gutter, p_clickable);
}

bool TextEdit::is_line_gutter_clickable(int p_line, int p_gutter) const {
	ERR_FAIL_INDEX_V(p_line, text.size(), false);
	ERR_FAIL_INDEX_V(p_gutter, gutters.size(), false);
	return text.is_line_gutter_clickable(p_line, p_gutter);
}

// Line style
void TextEdit::set_line_background_color(int p_line, const Color &p_color) {
	ERR_FAIL_INDEX(p_line, text.size());
	text.set_line_background_color(p_line, p_color);
	update();
}

Color TextEdit::get_line_background_color(int p_line) {
	ERR_FAIL_INDEX_V(p_line, text.size(), Color());
	return text.get_line_background_color(p_line);
}

void TextEdit::add_keyword(const String &p_keyword) {
	keywords.insert(p_keyword);
}

void TextEdit::clear_keywords() {
	keywords.clear();
}

void TextEdit::set_auto_indent(bool p_auto_indent) {
	auto_indent = p_auto_indent;
}

void TextEdit::cut() {
	if (readonly) {
		return;
	}

	if (!selection.active) {
		String clipboard = text[cursor.line];
		DisplayServer::get_singleton()->clipboard_set(clipboard);
		cursor_set_line(cursor.line);
		cursor_set_column(0);

		if (cursor.line == 0 && get_line_count() > 1) {
			_remove_text(cursor.line, 0, cursor.line + 1, 0);
		} else {
			_remove_text(cursor.line, 0, cursor.line, text[cursor.line].length());
			backspace_at_cursor();
			cursor_set_line(cursor.line + 1);
		}

		update();
		cut_copy_line = clipboard;

	} else {
		String clipboard = _base_get_text(selection.from_line, selection.from_column, selection.to_line, selection.to_column);
		DisplayServer::get_singleton()->clipboard_set(clipboard);

		_remove_text(selection.from_line, selection.from_column, selection.to_line, selection.to_column);
		cursor_set_line(selection.from_line, false); // Set afterwards else it causes the view to be offset.
		cursor_set_column(selection.from_column);

		selection.active = false;
		selection.selecting_mode = SelectionMode::SELECTION_MODE_NONE;
		update();
		cut_copy_line = "";
	}
}

void TextEdit::copy() {
	if (!selection.active) {
		if (text[cursor.line].length() != 0) {
			String clipboard = _base_get_text(cursor.line, 0, cursor.line, text[cursor.line].length());
			DisplayServer::get_singleton()->clipboard_set(clipboard);
			cut_copy_line = clipboard;
		}
	} else {
		String clipboard = _base_get_text(selection.from_line, selection.from_column, selection.to_line, selection.to_column);
		DisplayServer::get_singleton()->clipboard_set(clipboard);
		cut_copy_line = "";
	}
}

void TextEdit::paste() {
	if (readonly) {
		return;
	}

	String clipboard = DisplayServer::get_singleton()->clipboard_get();

	begin_complex_operation();
	if (selection.active) {
		selection.active = false;
		selection.selecting_mode = SelectionMode::SELECTION_MODE_NONE;
		_remove_text(selection.from_line, selection.from_column, selection.to_line, selection.to_column);
		cursor_set_line(selection.from_line, false);
		cursor_set_column(selection.from_column);

	} else if (!cut_copy_line.is_empty() && cut_copy_line == clipboard) {
		cursor_set_column(0);
		String ins = "\n";
		clipboard += ins;
	}

	_insert_text_at_cursor(clipboard);
	end_complex_operation();

	update();
}

void TextEdit::select_all() {
	if (!selecting_enabled) {
		return;
	}

	if (text.size() == 1 && text[0].length() == 0) {
		return;
	}
	selection.active = true;
	selection.from_line = 0;
	selection.from_column = 0;
	selection.selecting_line = 0;
	selection.selecting_column = 0;
	selection.to_line = text.size() - 1;
	selection.to_column = text[selection.to_line].length();
	selection.selecting_mode = SelectionMode::SELECTION_MODE_SHIFT;
	selection.shiftclick_left = true;
	cursor_set_line(selection.to_line, false);
	cursor_set_column(selection.to_column, false);
	update();
}

void TextEdit::select_word_under_caret() {
	if (!selecting_enabled) {
		return;
	}

	if (text.size() == 1 && text[0].length() == 0) {
		return;
	}

	if (selection.active) {
		// Allow toggling selection by pressing the shortcut a second time.
		// This is also usable as a general-purpose "deselect" shortcut after
		// selecting anything.
		deselect();
		return;
	}

	int begin = 0;
	int end = 0;
	const Vector<Vector2i> words = TS->shaped_text_get_word_breaks(text.get_line_data(cursor.line)->get_rid());
	for (int i = 0; i < words.size(); i++) {
		if (words[i].x <= cursor.column && words[i].y >= cursor.column) {
			begin = words[i].x;
			end = words[i].y;
			break;
		}
	}

	select(cursor.line, begin, cursor.line, end);
	// Move the cursor to the end of the word for easier editing.
	cursor_set_column(end, false);
}

void TextEdit::deselect() {
	selection.active = false;
	update();
}

void TextEdit::select(int p_from_line, int p_from_column, int p_to_line, int p_to_column) {
	if (!selecting_enabled) {
		return;
	}

	if (p_from_line < 0) {
		p_from_line = 0;
	} else if (p_from_line >= text.size()) {
		p_from_line = text.size() - 1;
	}
	if (p_from_column >= text[p_from_line].length()) {
		p_from_column = text[p_from_line].length();
	}
	if (p_from_column < 0) {
		p_from_column = 0;
	}

	if (p_to_line < 0) {
		p_to_line = 0;
	} else if (p_to_line >= text.size()) {
		p_to_line = text.size() - 1;
	}
	if (p_to_column >= text[p_to_line].length()) {
		p_to_column = text[p_to_line].length();
	}
	if (p_to_column < 0) {
		p_to_column = 0;
	}

	selection.from_line = p_from_line;
	selection.from_column = p_from_column;
	selection.to_line = p_to_line;
	selection.to_column = p_to_column;

	selection.active = true;

	if (selection.from_line == selection.to_line) {
		if (selection.from_column == selection.to_column) {
			selection.active = false;

		} else if (selection.from_column > selection.to_column) {
			selection.shiftclick_left = false;
			SWAP(selection.from_column, selection.to_column);
		} else {
			selection.shiftclick_left = true;
		}
	} else if (selection.from_line > selection.to_line) {
		selection.shiftclick_left = false;
		SWAP(selection.from_line, selection.to_line);
		SWAP(selection.from_column, selection.to_column);
	} else {
		selection.shiftclick_left = true;
	}

	update();
}

void TextEdit::swap_lines(int line1, int line2) {
	String tmp = get_line(line1);
	String tmp2 = get_line(line2);
	set_line(line2, tmp);
	set_line(line1, tmp2);
}

bool TextEdit::is_selection_active() const {
	return selection.active;
}

int TextEdit::get_selection_from_line() const {
	ERR_FAIL_COND_V(!selection.active, -1);
	return selection.from_line;
}

int TextEdit::get_selection_from_column() const {
	ERR_FAIL_COND_V(!selection.active, -1);
	return selection.from_column;
}

int TextEdit::get_selection_to_line() const {
	ERR_FAIL_COND_V(!selection.active, -1);
	return selection.to_line;
}

int TextEdit::get_selection_to_column() const {
	ERR_FAIL_COND_V(!selection.active, -1);
	return selection.to_column;
}

String TextEdit::get_selection_text() const {
	if (!selection.active) {
		return "";
	}

	return _base_get_text(selection.from_line, selection.from_column, selection.to_line, selection.to_column);
}

String TextEdit::get_word_under_cursor() const {
	Vector<Vector2i> words = TS->shaped_text_get_word_breaks(text.get_line_data(cursor.line)->get_rid());
	for (int i = 0; i < words.size(); i++) {
		if (words[i].x <= cursor.column && words[i].y > cursor.column) {
			return text[cursor.line].substr(words[i].x, words[i].y - words[i].x);
		}
	}
	return "";
}

void TextEdit::set_search_text(const String &p_search_text) {
	search_text = p_search_text;
}

void TextEdit::set_search_flags(uint32_t p_flags) {
	search_flags = p_flags;
}

void TextEdit::set_current_search_result(int line, int col) {
	search_result_line = line;
	search_result_col = col;
	update();
}

void TextEdit::set_highlight_all_occurrences(const bool p_enabled) {
	highlight_all_occurrences = p_enabled;
	update();
}

bool TextEdit::is_highlight_all_occurrences_enabled() const {
	return highlight_all_occurrences;
}

int TextEdit::_get_column_pos_of_word(const String &p_key, const String &p_search, uint32_t p_search_flags, int p_from_column) {
	int col = -1;

	if (p_key.length() > 0 && p_search.length() > 0) {
		if (p_from_column < 0 || p_from_column > p_search.length()) {
			p_from_column = 0;
		}

		while (col == -1 && p_from_column <= p_search.length()) {
			if (p_search_flags & SEARCH_MATCH_CASE) {
				col = p_search.find(p_key, p_from_column);
			} else {
				col = p_search.findn(p_key, p_from_column);
			}

			// Whole words only.
			if (col != -1 && p_search_flags & SEARCH_WHOLE_WORDS) {
				p_from_column = col;

				if (col > 0 && _is_text_char(p_search[col - 1])) {
					col = -1;
				} else if ((col + p_key.length()) < p_search.length() && _is_text_char(p_search[col + p_key.length()])) {
					col = -1;
				}
			}

			p_from_column += 1;
		}
	}
	return col;
}

Dictionary TextEdit::_search_bind(const String &p_key, uint32_t p_search_flags, int p_from_line, int p_from_column) const {
	int col, line;
	if (search(p_key, p_search_flags, p_from_line, p_from_column, line, col)) {
		Dictionary result;
		result["line"] = line;
		result["column"] = col;
		return result;

	} else {
		return Dictionary();
	}
}

bool TextEdit::search(const String &p_key, uint32_t p_search_flags, int p_from_line, int p_from_column, int &r_line, int &r_column) const {
	if (p_key.length() == 0) {
		return false;
	}
	ERR_FAIL_INDEX_V(p_from_line, text.size(), false);
	ERR_FAIL_INDEX_V(p_from_column, text[p_from_line].length() + 1, false);

	// Search through the whole document, but start by current line.

	int line = p_from_line;
	int pos = -1;

	for (int i = 0; i < text.size() + 1; i++) {
		if (line < 0) {
			line = text.size() - 1;
		}
		if (line == text.size()) {
			line = 0;
		}

		String text_line = text[line];
		int from_column = 0;
		if (line == p_from_line) {
			if (i == text.size()) {
				// Wrapped.

				if (p_search_flags & SEARCH_BACKWARDS) {
					from_column = text_line.length();
				} else {
					from_column = 0;
				}

			} else {
				from_column = p_from_column;
			}

		} else {
			if (p_search_flags & SEARCH_BACKWARDS) {
				from_column = text_line.length() - 1;
			} else {
				from_column = 0;
			}
		}

		pos = -1;

		int pos_from = (p_search_flags & SEARCH_BACKWARDS) ? text_line.length() : 0;
		int last_pos = -1;

		while (true) {
			if (p_search_flags & SEARCH_BACKWARDS) {
				while ((last_pos = (p_search_flags & SEARCH_MATCH_CASE) ? text_line.rfind(p_key, pos_from) : text_line.rfindn(p_key, pos_from)) != -1) {
					if (last_pos <= from_column) {
						pos = last_pos;
						break;
					}
					pos_from = last_pos - p_key.length();
					if (pos_from < 0) {
						break;
					}
				}
			} else {
				while ((last_pos = (p_search_flags & SEARCH_MATCH_CASE) ? text_line.find(p_key, pos_from) : text_line.findn(p_key, pos_from)) != -1) {
					if (last_pos >= from_column) {
						pos = last_pos;
						break;
					}
					pos_from = last_pos + p_key.length();
				}
			}

			bool is_match = true;

			if (pos != -1 && (p_search_flags & SEARCH_WHOLE_WORDS)) {
				// Validate for whole words.
				if (pos > 0 && _is_text_char(text_line[pos - 1])) {
					is_match = false;
				} else if (pos + p_key.length() < text_line.length() && _is_text_char(text_line[pos + p_key.length()])) {
					is_match = false;
				}
			}

			if (pos_from == -1) {
				pos = -1;
			}

			if (is_match || last_pos == -1 || pos == -1) {
				break;
			}

			pos_from = (p_search_flags & SEARCH_BACKWARDS) ? pos - 1 : pos + 1;
			pos = -1;
		}

		if (pos != -1) {
			break;
		}

		if (p_search_flags & SEARCH_BACKWARDS) {
			line--;
		} else {
			line++;
		}
	}

	if (pos == -1) {
		r_line = -1;
		r_column = -1;
		return false;
	}

	r_line = line;
	r_column = pos;

	return true;
}

void TextEdit::_cursor_changed_emit() {
	emit_signal("cursor_changed");
	cursor_changed_dirty = false;
}

void TextEdit::_text_changed_emit() {
	emit_signal("text_changed");
	text_changed_dirty = false;
}

void TextEdit::set_line_as_hidden(int p_line, bool p_hidden) {
	ERR_FAIL_INDEX(p_line, text.size());
	if (is_hiding_enabled() || !p_hidden) {
		text.set_hidden(p_line, p_hidden);
	}
	update();
}

bool TextEdit::is_line_hidden(int p_line) const {
	ERR_FAIL_INDEX_V(p_line, text.size(), false);
	return text.is_hidden(p_line);
}

void TextEdit::fold_all_lines() {
	for (int i = 0; i < text.size(); i++) {
		fold_line(i);
	}
	_update_scrollbars();
	update();
}

void TextEdit::unhide_all_lines() {
	for (int i = 0; i < text.size(); i++) {
		text.set_hidden(i, false);
	}
	_update_scrollbars();
	update();
}

int TextEdit::num_lines_from(int p_line_from, int visible_amount) const {
	// Returns the number of lines (hidden and unhidden) from p_line_from to (p_line_from + visible_amount of unhidden lines).
	ERR_FAIL_INDEX_V(p_line_from, text.size(), ABS(visible_amount));

	if (!is_hiding_enabled()) {
		return ABS(visible_amount);
	}

	int num_visible = 0;
	int num_total = 0;
	if (visible_amount >= 0) {
		for (int i = p_line_from; i < text.size(); i++) {
			num_total++;
			if (!is_line_hidden(i)) {
				num_visible++;
			}
			if (num_visible >= visible_amount) {
				break;
			}
		}
	} else {
		visible_amount = ABS(visible_amount);
		for (int i = p_line_from; i >= 0; i--) {
			num_total++;
			if (!is_line_hidden(i)) {
				num_visible++;
			}
			if (num_visible >= visible_amount) {
				break;
			}
		}
	}
	return num_total;
}

int TextEdit::num_lines_from_rows(int p_line_from, int p_wrap_index_from, int visible_amount, int &wrap_index) const {
	// Returns the number of lines (hidden and unhidden) from (p_line_from + p_wrap_index_from) row to (p_line_from + visible_amount of unhidden and wrapped rows).
	// Wrap index is set to the wrap index of the last line.
	wrap_index = 0;
	ERR_FAIL_INDEX_V(p_line_from, text.size(), ABS(visible_amount));

	if (!is_hiding_enabled() && !is_wrap_enabled()) {
		return ABS(visible_amount);
	}

	int num_visible = 0;
	int num_total = 0;
	if (visible_amount == 0) {
		num_total = 0;
		wrap_index = 0;
	} else if (visible_amount > 0) {
		int i;
		num_visible -= p_wrap_index_from;
		for (i = p_line_from; i < text.size(); i++) {
			num_total++;
			if (!is_line_hidden(i)) {
				num_visible++;
				num_visible += times_line_wraps(i);
			}
			if (num_visible >= visible_amount) {
				break;
			}
		}
		wrap_index = times_line_wraps(MIN(i, text.size() - 1)) - MAX(0, num_visible - visible_amount);
	} else {
		visible_amount = ABS(visible_amount);
		int i;
		num_visible -= times_line_wraps(p_line_from) - p_wrap_index_from;
		for (i = p_line_from; i >= 0; i--) {
			num_total++;
			if (!is_line_hidden(i)) {
				num_visible++;
				num_visible += times_line_wraps(i);
			}
			if (num_visible >= visible_amount) {
				break;
			}
		}
		wrap_index = MAX(0, num_visible - visible_amount);
	}
	wrap_index = MAX(wrap_index, 0);
	return num_total;
}

int TextEdit::get_last_unhidden_line() const {
	// Returns the last line in the text that is not hidden.
	if (!is_hiding_enabled()) {
		return text.size() - 1;
	}

	int last_line;
	for (last_line = text.size() - 1; last_line > 0; last_line--) {
		if (!is_line_hidden(last_line)) {
			break;
		}
	}
	return last_line;
}

int TextEdit::get_indent_level(int p_line) const {
	ERR_FAIL_INDEX_V(p_line, text.size(), 0);

	// Counts number of tabs and spaces before line starts.
	int tab_count = 0;
	int whitespace_count = 0;
	int line_length = text[p_line].size();
	for (int i = 0; i < line_length - 1; i++) {
		if (text[p_line][i] == '\t') {
			tab_count++;
		} else if (text[p_line][i] == ' ') {
			whitespace_count++;
		} else {
			break;
		}
	}
	return tab_count * indent_size + whitespace_count;
}

bool TextEdit::is_line_comment(int p_line) const {
	// Checks to see if this line is the start of a comment.
	ERR_FAIL_INDEX_V(p_line, text.size(), false);

	int line_length = text[p_line].size();
	for (int i = 0; i < line_length - 1; i++) {
		if (_is_whitespace(text[p_line][i])) {
			continue;
		}
		if (_is_symbol(text[p_line][i])) {
			if (text[p_line][i] == '\\') {
				i++; // Skip quoted anything.
				continue;
			}
			return text[p_line][i] == '#' || (i + 1 < line_length && text[p_line][i] == '/' && text[p_line][i + 1] == '/');
		}
		break;
	}
	return false;
}

bool TextEdit::can_fold(int p_line) const {
	ERR_FAIL_INDEX_V(p_line, text.size(), false);
	if (!is_hiding_enabled()) {
		return false;
	}
	if (p_line + 1 >= text.size()) {
		return false;
	}
	if (text[p_line].strip_edges().size() == 0) {
		return false;
	}
	if (is_folded(p_line)) {
		return false;
	}
	if (is_line_hidden(p_line)) {
		return false;
	}
	if (is_line_comment(p_line)) {
		return false;
	}

	int start_indent = get_indent_level(p_line);

	for (int i = p_line + 1; i < text.size(); i++) {
		if (text[i].strip_edges().size() == 0) {
			continue;
		}
		int next_indent = get_indent_level(i);
		if (is_line_comment(i)) {
			continue;
		} else if (next_indent > start_indent) {
			return true;
		} else {
			return false;
		}
	}

	return false;
}

bool TextEdit::is_folded(int p_line) const {
	ERR_FAIL_INDEX_V(p_line, text.size(), false);
	if (p_line + 1 >= text.size()) {
		return false;
	}
	return !is_line_hidden(p_line) && is_line_hidden(p_line + 1);
}

Vector<int> TextEdit::get_folded_lines() const {
	Vector<int> folded_lines;

	for (int i = 0; i < text.size(); i++) {
		if (is_folded(i)) {
			folded_lines.push_back(i);
		}
	}
	return folded_lines;
}

void TextEdit::fold_line(int p_line) {
	ERR_FAIL_INDEX(p_line, text.size());
	if (!is_hiding_enabled()) {
		return;
	}
	if (!can_fold(p_line)) {
		return;
	}

	// Hide lines below this one.
	int start_indent = get_indent_level(p_line);
	int last_line = start_indent;
	for (int i = p_line + 1; i < text.size(); i++) {
		if (text[i].strip_edges().size() != 0) {
			if (is_line_comment(i)) {
				continue;
			} else if (get_indent_level(i) > start_indent) {
				last_line = i;
			} else {
				break;
			}
		}
	}
	for (int i = p_line + 1; i <= last_line; i++) {
		set_line_as_hidden(i, true);
	}

	// Fix selection.
	if (is_selection_active()) {
		if (is_line_hidden(selection.from_line) && is_line_hidden(selection.to_line)) {
			deselect();
		} else if (is_line_hidden(selection.from_line)) {
			select(p_line, 9999, selection.to_line, selection.to_column);
		} else if (is_line_hidden(selection.to_line)) {
			select(selection.from_line, selection.from_column, p_line, 9999);
		}
	}

	// Reset cursor.
	if (is_line_hidden(cursor.line)) {
		cursor_set_line(p_line, false, false);
		cursor_set_column(get_line(p_line).length(), false);
	}
	_update_scrollbars();
	update();
}

void TextEdit::unfold_line(int p_line) {
	ERR_FAIL_INDEX(p_line, text.size());

	if (!is_folded(p_line) && !is_line_hidden(p_line)) {
		return;
	}
	int fold_start;
	for (fold_start = p_line; fold_start > 0; fold_start--) {
		if (is_folded(fold_start)) {
			break;
		}
	}
	fold_start = is_folded(fold_start) ? fold_start : p_line;

	for (int i = fold_start + 1; i < text.size(); i++) {
		if (is_line_hidden(i)) {
			set_line_as_hidden(i, false);
		} else {
			break;
		}
	}
	_update_scrollbars();
	update();
}

void TextEdit::toggle_fold_line(int p_line) {
	ERR_FAIL_INDEX(p_line, text.size());

	if (!is_folded(p_line)) {
		fold_line(p_line);
	} else {
		unfold_line(p_line);
	}
}

int TextEdit::get_line_count() const {
	return text.size();
}

void TextEdit::_do_text_op(const TextOperation &p_op, bool p_reverse) {
	ERR_FAIL_COND(p_op.type == TextOperation::TYPE_NONE);

	bool insert = p_op.type == TextOperation::TYPE_INSERT;
	if (p_reverse) {
		insert = !insert;
	}

	if (insert) {
		int check_line;
		int check_column;
		_base_insert_text(p_op.from_line, p_op.from_column, p_op.text, check_line, check_column);
		ERR_FAIL_COND(check_line != p_op.to_line); // BUG.
		ERR_FAIL_COND(check_column != p_op.to_column); // BUG.
	} else {
		_base_remove_text(p_op.from_line, p_op.from_column, p_op.to_line, p_op.to_column);
	}
}

void TextEdit::_clear_redo() {
	if (undo_stack_pos == nullptr) {
		return; // Nothing to clear.
	}

	_push_current_op();

	while (undo_stack_pos) {
		List<TextOperation>::Element *elem = undo_stack_pos;
		undo_stack_pos = undo_stack_pos->next();
		undo_stack.erase(elem);
	}
}

void TextEdit::undo() {
	if (readonly) {
		return;
	}

	_push_current_op();

	if (undo_stack_pos == nullptr) {
		if (!undo_stack.size()) {
			return; // Nothing to undo.
		}

		undo_stack_pos = undo_stack.back();

	} else if (undo_stack_pos == undo_stack.front()) {
		return; // At the bottom of the undo stack.
	} else {
		undo_stack_pos = undo_stack_pos->prev();
	}

	deselect();

	TextOperation op = undo_stack_pos->get();
	_do_text_op(op, true);
	if (op.type != TextOperation::TYPE_INSERT && (op.from_line != op.to_line || op.to_column != op.from_column + 1)) {
		select(op.from_line, op.from_column, op.to_line, op.to_column);
	}

	current_op.version = op.prev_version;
	if (undo_stack_pos->get().chain_backward) {
		while (true) {
			ERR_BREAK(!undo_stack_pos->prev());
			undo_stack_pos = undo_stack_pos->prev();
			op = undo_stack_pos->get();
			_do_text_op(op, true);
			current_op.version = op.prev_version;
			if (undo_stack_pos->get().chain_forward) {
				break;
			}
		}
	}

	_update_scrollbars();
	if (undo_stack_pos->get().type == TextOperation::TYPE_REMOVE) {
		cursor_set_line(undo_stack_pos->get().to_line, false);
		cursor_set_column(undo_stack_pos->get().to_column);
		_cancel_code_hint();
	} else {
		cursor_set_line(undo_stack_pos->get().from_line, false);
		cursor_set_column(undo_stack_pos->get().from_column);
	}
	update();
}

void TextEdit::redo() {
	if (readonly) {
		return;
	}
	_push_current_op();

	if (undo_stack_pos == nullptr) {
		return; // Nothing to do.
	}

	deselect();

	TextOperation op = undo_stack_pos->get();
	_do_text_op(op, false);
	current_op.version = op.version;
	if (undo_stack_pos->get().chain_forward) {
		while (true) {
			ERR_BREAK(!undo_stack_pos->next());
			undo_stack_pos = undo_stack_pos->next();
			op = undo_stack_pos->get();
			_do_text_op(op, false);
			current_op.version = op.version;
			if (undo_stack_pos->get().chain_backward) {
				break;
			}
		}
	}

	_update_scrollbars();
	cursor_set_line(undo_stack_pos->get().to_line, false);
	cursor_set_column(undo_stack_pos->get().to_column);
	undo_stack_pos = undo_stack_pos->next();
	update();
}

void TextEdit::clear_undo_history() {
	saved_version = 0;
	current_op.type = TextOperation::TYPE_NONE;
	undo_stack_pos = nullptr;
	undo_stack.clear();
}

void TextEdit::begin_complex_operation() {
	_push_current_op();
	next_operation_is_complex = true;
}

void TextEdit::end_complex_operation() {
	_push_current_op();
	ERR_FAIL_COND(undo_stack.size() == 0);

	if (undo_stack.back()->get().chain_forward) {
		undo_stack.back()->get().chain_forward = false;
		return;
	}

	undo_stack.back()->get().chain_backward = true;
}

void TextEdit::_push_current_op() {
	if (current_op.type == TextOperation::TYPE_NONE) {
		return; // Nothing to do.
	}

	if (next_operation_is_complex) {
		current_op.chain_forward = true;
		next_operation_is_complex = false;
	}

	undo_stack.push_back(current_op);
	current_op.type = TextOperation::TYPE_NONE;
	current_op.text = "";
	current_op.chain_forward = false;

	if (undo_stack.size() > undo_stack_max_size) {
		undo_stack.pop_front();
	}
}

void TextEdit::set_indent_using_spaces(const bool p_use_spaces) {
	indent_using_spaces = p_use_spaces;
}

bool TextEdit::is_indent_using_spaces() const {
	return indent_using_spaces;
}

void TextEdit::set_indent_size(const int p_size) {
	ERR_FAIL_COND_MSG(p_size <= 0, "Indend size must be greater than 0.");
	if (indent_size != p_size) {
		indent_size = p_size;
		text.set_indent_size(p_size);
		text.invalidate_all_lines();
	}

	space_indent = "";
	for (int i = 0; i < p_size; i++) {
		space_indent += " ";
	}

	update();
}

int TextEdit::get_indent_size() {
	return indent_size;
}

void TextEdit::set_draw_tabs(bool p_draw) {
	draw_tabs = p_draw;
	update();
}

bool TextEdit::is_drawing_tabs() const {
	return draw_tabs;
}

void TextEdit::set_draw_spaces(bool p_draw) {
	draw_spaces = p_draw;
	update();
}

bool TextEdit::is_drawing_spaces() const {
	return draw_spaces;
}

void TextEdit::set_override_selected_font_color(bool p_override_selected_font_color) {
	override_selected_font_color = p_override_selected_font_color;
}

bool TextEdit::is_overriding_selected_font_color() const {
	return override_selected_font_color;
}

void TextEdit::set_insert_mode(bool p_enabled) {
	insert_mode = p_enabled;
	update();
}

bool TextEdit::is_insert_mode() const {
	return insert_mode;
}

bool TextEdit::is_insert_text_operation() {
	return (current_op.type == TextOperation::TYPE_INSERT);
}

uint32_t TextEdit::get_version() const {
	return current_op.version;
}

uint32_t TextEdit::get_saved_version() const {
	return saved_version;
}

void TextEdit::tag_saved_version() {
	saved_version = get_version();
}

double TextEdit::get_scroll_pos_for_line(int p_line, int p_wrap_index) const {
	if (!is_wrap_enabled() && !is_hiding_enabled()) {
		return p_line;
	}

	// Count the number of visible lines up to this line.
	double new_line_scroll_pos = 0.0;
	int to = CLAMP(p_line, 0, text.size() - 1);
	for (int i = 0; i < to; i++) {
		if (!text.is_hidden(i)) {
			new_line_scroll_pos++;
			new_line_scroll_pos += times_line_wraps(i);
		}
	}
	new_line_scroll_pos += p_wrap_index;
	return new_line_scroll_pos;
}

void TextEdit::set_line_as_first_visible(int p_line, int p_wrap_index) {
	set_v_scroll(get_scroll_pos_for_line(p_line, p_wrap_index));
}

void TextEdit::set_line_as_center_visible(int p_line, int p_wrap_index) {
	int visible_rows = get_visible_rows();
	int wi;
	int first_line = p_line - num_lines_from_rows(p_line, p_wrap_index, -visible_rows / 2, wi) + 1;

	set_v_scroll(get_scroll_pos_for_line(first_line, wi));
}

void TextEdit::set_line_as_last_visible(int p_line, int p_wrap_index) {
	int wi;
	int first_line = p_line - num_lines_from_rows(p_line, p_wrap_index, -get_visible_rows() - 1, wi) + 1;

	set_v_scroll(get_scroll_pos_for_line(first_line, wi) + get_visible_rows_offset());
}

int TextEdit::get_first_visible_line() const {
	return CLAMP(cursor.line_ofs, 0, text.size() - 1);
}

int TextEdit::get_last_full_visible_line() const {
	int first_vis_line = get_first_visible_line();
	int last_vis_line = 0;
	int wi;
	last_vis_line = first_vis_line + num_lines_from_rows(first_vis_line, cursor.wrap_ofs, get_visible_rows(), wi) - 1;
	last_vis_line = CLAMP(last_vis_line, 0, text.size() - 1);
	return last_vis_line;
}

int TextEdit::get_last_full_visible_line_wrap_index() const {
	int first_vis_line = get_first_visible_line();
	int wi;
	num_lines_from_rows(first_vis_line, cursor.wrap_ofs, get_visible_rows(), wi);
	return wi;
}

double TextEdit::get_visible_rows_offset() const {
	double total = _get_control_height();
	total /= (double)get_row_height();
	total = total - floor(total);
	total = -CLAMP(total, 0.001, 1) + 1;
	return total;
}

double TextEdit::get_v_scroll_offset() const {
	double val = get_v_scroll() - floor(get_v_scroll());
	return CLAMP(val, 0, 1);
}

double TextEdit::get_v_scroll() const {
	return v_scroll->get_value();
}

void TextEdit::set_v_scroll(double p_scroll) {
	v_scroll->set_value(p_scroll);
	int max_v_scroll = v_scroll->get_max() - v_scroll->get_page();
	if (p_scroll >= max_v_scroll - 1.0) {
		_scroll_moved(v_scroll->get_value());
	}
}

int TextEdit::get_h_scroll() const {
	return h_scroll->get_value();
}

void TextEdit::set_h_scroll(int p_scroll) {
	if (p_scroll < 0) {
		p_scroll = 0;
	}
	h_scroll->set_value(p_scroll);
}

void TextEdit::set_smooth_scroll_enabled(bool p_enable) {
	v_scroll->set_smooth_scroll_enabled(p_enable);
	smooth_scroll_enabled = p_enable;
}

bool TextEdit::is_smooth_scroll_enabled() const {
	return smooth_scroll_enabled;
}

void TextEdit::set_v_scroll_speed(float p_speed) {
	v_scroll_speed = p_speed;
}

float TextEdit::get_v_scroll_speed() const {
	return v_scroll_speed;
}

void TextEdit::set_completion(bool p_enabled, const Vector<String> &p_prefixes) {
	completion_prefixes.clear();
	completion_enabled = p_enabled;
	for (int i = 0; i < p_prefixes.size(); i++) {
		completion_prefixes.insert(p_prefixes[i]);
	}
}

void TextEdit::_confirm_completion() {
	begin_complex_operation();

	_remove_text(cursor.line, cursor.column - completion_base.length(), cursor.line, cursor.column);
	cursor_set_column(cursor.column - completion_base.length(), false);
	insert_text_at_cursor(completion_current.insert_text);

	// When inserted into the middle of an existing string/method, don't add an unnecessary quote/bracket.
	String line = text[cursor.line];
	char32_t next_char = line[cursor.column];
	char32_t last_completion_char = completion_current.insert_text[completion_current.insert_text.length() - 1];
	char32_t last_completion_char_display = completion_current.display[completion_current.display.length() - 1];

	if ((last_completion_char == '"' || last_completion_char == '\'') && (last_completion_char == next_char || last_completion_char_display == next_char)) {
		_remove_text(cursor.line, cursor.column, cursor.line, cursor.column + 1);
	}

	if (last_completion_char == '(') {
		if (next_char == last_completion_char) {
			_base_remove_text(cursor.line, cursor.column - 1, cursor.line, cursor.column);
		} else if (auto_brace_completion_enabled) {
			insert_text_at_cursor(")");
			cursor.column--;
		}
	} else if (last_completion_char == ')' && next_char == '(') {
		_base_remove_text(cursor.line, cursor.column - 2, cursor.line, cursor.column);
		if (line[cursor.column + 1] != ')') {
			cursor.column--;
		}
	}

	end_complex_operation();

	_cancel_completion();

	if (last_completion_char == '(') {
		query_code_comple();
	}
}

void TextEdit::_cancel_code_hint() {
	completion_hint = "";
	update();
}

void TextEdit::_cancel_completion() {
	if (!completion_active) {
		return;
	}

	completion_active = false;
	completion_forced = false;
	update();
}

static bool _is_completable(char32_t c) {
	return !_is_symbol(c) || c == '"' || c == '\'';
}

void TextEdit::_update_completion_candidates() {
	String l = text[cursor.line];
	int cofs = CLAMP(cursor.column, 0, l.length());

	String s;

	// Look for keywords first.

	bool inquote = false;
	int first_quote = -1;
	int restore_quotes = -1;

	int c = cofs - 1;
	while (c >= 0) {
		if (l[c] == '"' || l[c] == '\'') {
			inquote = !inquote;
			if (first_quote == -1) {
				first_quote = c;
			}
			restore_quotes = 0;
		} else if (restore_quotes == 0 && l[c] == '$') {
			restore_quotes = 1;
		} else if (restore_quotes == 0 && !_is_whitespace(l[c])) {
			restore_quotes = -1;
		}
		c--;
	}

	bool pre_keyword = false;
	bool cancel = false;

	if (!inquote && first_quote == cofs - 1) {
		// No completion here.
		cancel = true;
	} else if (inquote && first_quote != -1) {
		s = l.substr(first_quote, cofs - first_quote);
	} else if (cofs > 0 && l[cofs - 1] == ' ') {
		int kofs = cofs - 1;
		String kw;
		while (kofs >= 0 && l[kofs] == ' ') {
			kofs--;
		}

		while (kofs >= 0 && l[kofs] > 32 && _is_completable(l[kofs])) {
			kw = String::chr(l[kofs]) + kw;
			kofs--;
		}

		pre_keyword = keywords.has(kw);

	} else {
		while (cofs > 0 && l[cofs - 1] > 32 && (l[cofs - 1] == '/' || _is_completable(l[cofs - 1]))) {
			s = String::chr(l[cofs - 1]) + s;
			if (l[cofs - 1] == '\'' || l[cofs - 1] == '"' || l[cofs - 1] == '$') {
				break;
			}

			cofs--;
		}
	}

	if (cursor.column > 0 && l[cursor.column - 1] == '(' && !pre_keyword && !completion_forced) {
		cancel = true;
	}

	update();

	bool prev_is_prefix = false;
	if (cofs > 0 && completion_prefixes.has(String::chr(l[cofs - 1]))) {
		prev_is_prefix = true;
	}
	// Check with one space before prefix, to allow indent.
	if (cofs > 1 && l[cofs - 1] == ' ' && completion_prefixes.has(String::chr(l[cofs - 2]))) {
		prev_is_prefix = true;
	}

	if (cancel || (!pre_keyword && s == "" && (cofs == 0 || !prev_is_prefix))) {
		// None to complete, cancel.
		_cancel_completion();
		return;
	}

	completion_options.clear();
	completion_index = 0;
	completion_base = s;
	Vector<float> sim_cache;
	bool single_quote = s.begins_with("'");
	Vector<ScriptCodeCompletionOption> completion_options_casei;
	Vector<ScriptCodeCompletionOption> completion_options_subseq;
	Vector<ScriptCodeCompletionOption> completion_options_subseq_casei;

	String s_lower = s.to_lower();

	for (List<ScriptCodeCompletionOption>::Element *E = completion_sources.front(); E; E = E->next()) {
		ScriptCodeCompletionOption &option = E->get();

		if (single_quote && option.display.is_quoted()) {
			option.display = option.display.unquote().quote("'");
		}

		if (inquote && restore_quotes == 1 && !option.display.is_quoted()) {
			String quote = single_quote ? "'" : "\"";
			option.display = option.display.quote(quote);
			option.insert_text = option.insert_text.quote(quote);
		}

		if (option.display.length() == 0) {
			continue;
		} else if (s.length() == 0) {
			completion_options.push_back(option);
		} else {
			// This code works the same as:
			/*
			if (option.display.begins_with(s)) {
				completion_options.push_back(option);
			} else if (option.display.to_lower().begins_with(s.to_lower())) {
				completion_options_casei.push_back(option);
			} else if (s.is_subsequence_of(option.display)) {
				completion_options_subseq.push_back(option);
			} else if (s.is_subsequence_ofi(option.display)) {
				completion_options_subseq_casei.push_back(option);
			}
			*/
			// But is more performant due to being inlined and looping over the characters only once

			String display_lower = option.display.to_lower();

			const char32_t *ssq = &s[0];
			const char32_t *ssq_lower = &s_lower[0];

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

			if (!*ssq) { // Matched the whole subsequence in s
				if (ssq_last_tgt == &option.display[s.length() - 1]) { // Finished matching in the first s.length() characters
					completion_options.push_back(option);
				} else {
					completion_options_subseq.push_back(option);
				}
			} else if (!*ssq_lower) { // Matched the whole subsequence in s_lower
				if (ssq_lower_last_tgt == &option.display[s.length() - 1]) { // Finished matching in the first s.length() characters
					completion_options_casei.push_back(option);
				} else {
					completion_options_subseq_casei.push_back(option);
				}
			}
		}
	}

	completion_options.append_array(completion_options_casei);
	completion_options.append_array(completion_options_subseq);
	completion_options.append_array(completion_options_subseq_casei);

	if (completion_options.size() == 0) {
		// No options to complete, cancel.
		_cancel_completion();
		return;
	}

	if (completion_options.size() == 1 && s == completion_options[0].display) {
		// A perfect match, stop completion.
		_cancel_completion();
		return;
	}

	// The top of the list is the best match.
	completion_current = completion_options[0];
	completion_enabled = true;
}

void TextEdit::query_code_comple() {
	String l = text[cursor.line];
	int ofs = CLAMP(cursor.column, 0, l.length());

	bool inquote = false;

	int c = ofs - 1;
	while (c >= 0) {
		if (l[c] == '"' || l[c] == '\'') {
			inquote = !inquote;
		}
		c--;
	}

	bool ignored = completion_active && !completion_options.is_empty();
	if (ignored) {
		ScriptCodeCompletionOption::Kind kind = ScriptCodeCompletionOption::KIND_PLAIN_TEXT;
		const ScriptCodeCompletionOption *previous_option = nullptr;
		for (int i = 0; i < completion_options.size(); i++) {
			const ScriptCodeCompletionOption &current_option = completion_options[i];
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

	if (!ignored) {
		if (ofs > 0 && (inquote || _is_completable(l[ofs - 1]) || completion_prefixes.has(String::chr(l[ofs - 1])))) {
			emit_signal("request_completion");
		} else if (ofs > 1 && l[ofs - 1] == ' ' && completion_prefixes.has(String::chr(l[ofs - 2]))) { // Make it work with a space too, it's good enough.
			emit_signal("request_completion");
		}
	}
}

void TextEdit::set_code_hint(const String &p_hint) {
	completion_hint = p_hint;
	completion_hint_offset = -0xFFFF;
	update();
}

void TextEdit::code_complete(const List<ScriptCodeCompletionOption> &p_strings, bool p_forced) {
	completion_sources = p_strings;
	completion_active = true;
	completion_forced = p_forced;
	completion_current = ScriptCodeCompletionOption();
	completion_index = 0;
	_update_completion_candidates();
}

String TextEdit::get_word_at_pos(const Vector2 &p_pos) const {
	int row, col;
	_get_mouse_pos(p_pos, row, col);

	String s = text[row];
	if (s.length() == 0) {
		return "";
	}
	int beg, end;
	if (select_word(s, col, beg, end)) {
		bool inside_quotes = false;
		char32_t selected_quote = '\0';
		int qbegin = 0, qend = 0;
		for (int i = 0; i < s.length(); i++) {
			if (s[i] == '"' || s[i] == '\'') {
				if (i == 0 || s[i - 1] != '\\') {
					if (inside_quotes && selected_quote == s[i]) {
						qend = i;
						inside_quotes = false;
						selected_quote = '\0';
						if (col >= qbegin && col <= qend) {
							return s.substr(qbegin, qend - qbegin);
						}
					} else if (!inside_quotes) {
						qbegin = i + 1;
						inside_quotes = true;
						selected_quote = s[i];
					}
				}
			}
		}

		return s.substr(beg, end - beg);
	}

	return String();
}

String TextEdit::get_tooltip(const Point2 &p_pos) const {
	if (!tooltip_obj) {
		return Control::get_tooltip(p_pos);
	}
	int row, col;
	_get_mouse_pos(p_pos, row, col);

	String s = text[row];
	if (s.length() == 0) {
		return Control::get_tooltip(p_pos);
	}
	int beg, end;
	if (select_word(s, col, beg, end)) {
		String tt = tooltip_obj->call(tooltip_func, s.substr(beg, end - beg), tooltip_ud);

		return tt;
	}

	return Control::get_tooltip(p_pos);
}

void TextEdit::set_tooltip_request_func(Object *p_obj, const StringName &p_function, const Variant &p_udata) {
	tooltip_obj = p_obj;
	tooltip_func = p_function;
	tooltip_ud = p_udata;
}

void TextEdit::set_line(int line, String new_text) {
	if (line < 0 || line >= text.size()) {
		return;
	}
	_remove_text(line, 0, line, text[line].length());
	_insert_text(line, 0, new_text);
	if (cursor.line == line) {
		cursor.column = MIN(cursor.column, new_text.length());
	}
	if (is_selection_active() && line == selection.to_line && selection.to_column > text[line].length()) {
		selection.to_column = text[line].length();
	}
}

void TextEdit::insert_at(const String &p_text, int at) {
	_insert_text(at, 0, p_text + "\n");
	if (cursor.line >= at) {
		// offset cursor when located after inserted line
		++cursor.line;
	}
	if (is_selection_active()) {
		if (selection.from_line >= at) {
			// offset selection when located after inserted line
			++selection.from_line;
			++selection.to_line;
		} else if (selection.to_line >= at) {
			// extend selection that includes inserted line
			++selection.to_line;
		}
	}
}

void TextEdit::set_show_line_length_guidelines(bool p_show) {
	line_length_guidelines = p_show;
	update();
}

void TextEdit::set_line_length_guideline_soft_column(int p_column) {
	line_length_guideline_soft_col = p_column;
	update();
}

void TextEdit::set_line_length_guideline_hard_column(int p_column) {
	line_length_guideline_hard_col = p_column;
	update();
}

void TextEdit::set_draw_minimap(bool p_draw) {
	if (draw_minimap != p_draw) {
		draw_minimap = p_draw;
		_update_wrap_at();
	}
	update();
}

bool TextEdit::is_drawing_minimap() const {
	return draw_minimap;
}

void TextEdit::set_minimap_width(int p_minimap_width) {
	if (minimap_width != p_minimap_width) {
		minimap_width = p_minimap_width;
		_update_wrap_at();
	}
	update();
}

int TextEdit::get_minimap_width() const {
	return minimap_width;
}

void TextEdit::set_hiding_enabled(bool p_enabled) {
	if (!p_enabled) {
		unhide_all_lines();
	}
	hiding_enabled = p_enabled;
	update();
}

bool TextEdit::is_hiding_enabled() const {
	return hiding_enabled;
}

void TextEdit::set_highlight_current_line(bool p_enabled) {
	highlight_current_line = p_enabled;
	update();
}

bool TextEdit::is_highlight_current_line_enabled() const {
	return highlight_current_line;
}

bool TextEdit::is_text_field() const {
	return true;
}

void TextEdit::menu_option(int p_option) {
	switch (p_option) {
		case MENU_CUT: {
			if (!readonly) {
				cut();
			}
		} break;
		case MENU_COPY: {
			copy();
		} break;
		case MENU_PASTE: {
			if (!readonly) {
				paste();
			}
		} break;
		case MENU_CLEAR: {
			if (!readonly) {
				clear();
			}
		} break;
		case MENU_SELECT_ALL: {
			select_all();
		} break;
		case MENU_UNDO: {
			undo();
		} break;
		case MENU_REDO: {
			redo();
		} break;
		case MENU_DIR_INHERITED: {
			set_text_direction(TEXT_DIRECTION_INHERITED);
		} break;
		case MENU_DIR_AUTO: {
			set_text_direction(TEXT_DIRECTION_AUTO);
		} break;
		case MENU_DIR_LTR: {
			set_text_direction(TEXT_DIRECTION_LTR);
		} break;
		case MENU_DIR_RTL: {
			set_text_direction(TEXT_DIRECTION_RTL);
		} break;
		case MENU_DISPLAY_UCC: {
			set_draw_control_chars(!get_draw_control_chars());
		} break;
		case MENU_INSERT_LRM: {
			if (!readonly) {
				insert_text_at_cursor(String::chr(0x200E));
			}
		} break;
		case MENU_INSERT_RLM: {
			if (!readonly) {
				insert_text_at_cursor(String::chr(0x200F));
			}
		} break;
		case MENU_INSERT_LRE: {
			if (!readonly) {
				insert_text_at_cursor(String::chr(0x202A));
			}
		} break;
		case MENU_INSERT_RLE: {
			if (!readonly) {
				insert_text_at_cursor(String::chr(0x202B));
			}
		} break;
		case MENU_INSERT_LRO: {
			if (!readonly) {
				insert_text_at_cursor(String::chr(0x202D));
			}
		} break;
		case MENU_INSERT_RLO: {
			if (!readonly) {
				insert_text_at_cursor(String::chr(0x202E));
			}
		} break;
		case MENU_INSERT_PDF: {
			if (!readonly) {
				insert_text_at_cursor(String::chr(0x202C));
			}
		} break;
		case MENU_INSERT_ALM: {
			if (!readonly) {
				insert_text_at_cursor(String::chr(0x061C));
			}
		} break;
		case MENU_INSERT_LRI: {
			if (!readonly) {
				insert_text_at_cursor(String::chr(0x2066));
			}
		} break;
		case MENU_INSERT_RLI: {
			if (!readonly) {
				insert_text_at_cursor(String::chr(0x2067));
			}
		} break;
		case MENU_INSERT_FSI: {
			if (!readonly) {
				insert_text_at_cursor(String::chr(0x2068));
			}
		} break;
		case MENU_INSERT_PDI: {
			if (!readonly) {
				insert_text_at_cursor(String::chr(0x2069));
			}
		} break;
		case MENU_INSERT_ZWJ: {
			if (!readonly) {
				insert_text_at_cursor(String::chr(0x200D));
			}
		} break;
		case MENU_INSERT_ZWNJ: {
			if (!readonly) {
				insert_text_at_cursor(String::chr(0x200C));
			}
		} break;
		case MENU_INSERT_WJ: {
			if (!readonly) {
				insert_text_at_cursor(String::chr(0x2060));
			}
		} break;
		case MENU_INSERT_SHY: {
			if (!readonly) {
				insert_text_at_cursor(String::chr(0x00AD));
			}
		}
	}
}

void TextEdit::set_highlighted_word(const String &new_word) {
	highlighted_word = new_word;
	update();
}

void TextEdit::set_select_identifiers_on_hover(bool p_enable) {
	select_identifiers_enabled = p_enable;
}

bool TextEdit::is_selecting_identifiers_on_hover_enabled() const {
	return select_identifiers_enabled;
}

void TextEdit::set_context_menu_enabled(bool p_enable) {
	context_menu_enabled = p_enable;
}

bool TextEdit::is_context_menu_enabled() {
	return context_menu_enabled;
}

void TextEdit::set_shortcut_keys_enabled(bool p_enabled) {
	shortcut_keys_enabled = p_enabled;

	_generate_context_menu();
}

void TextEdit::set_virtual_keyboard_enabled(bool p_enable) {
	virtual_keyboard_enabled = p_enable;
}

void TextEdit::set_selecting_enabled(bool p_enabled) {
	selecting_enabled = p_enabled;

	if (!selecting_enabled) {
		deselect();
	}

	_generate_context_menu();
}

bool TextEdit::is_selecting_enabled() const {
	return selecting_enabled;
}

bool TextEdit::is_shortcut_keys_enabled() const {
	return shortcut_keys_enabled;
}

bool TextEdit::is_virtual_keyboard_enabled() const {
	return virtual_keyboard_enabled;
}

PopupMenu *TextEdit::get_menu() const {
	return menu;
}

bool TextEdit::_set(const StringName &p_name, const Variant &p_value) {
	String str = p_name;
	if (str.begins_with("opentype_features/")) {
		String name = str.get_slicec('/', 1);
		int32_t tag = TS->name_to_tag(name);
		double value = p_value;
		if (value == -1) {
			if (opentype_features.has(tag)) {
				opentype_features.erase(tag);
				text.set_font_features(opentype_features);
				text.invalidate_all();
				update();
			}
		} else {
			if ((double)opentype_features[tag] != value) {
				opentype_features[tag] = value;
				text.set_font_features(opentype_features);
				text.invalidate_all();
				;
				update();
			}
		}
		notify_property_list_changed();
		return true;
	}

	return false;
}

bool TextEdit::_get(const StringName &p_name, Variant &r_ret) const {
	String str = p_name;
	if (str.begins_with("opentype_features/")) {
		String name = str.get_slicec('/', 1);
		int32_t tag = TS->name_to_tag(name);
		if (opentype_features.has(tag)) {
			r_ret = opentype_features[tag];
			return true;
		} else {
			r_ret = -1;
			return true;
		}
	}
	return false;
}

void TextEdit::_get_property_list(List<PropertyInfo> *p_list) const {
	for (const Variant *ftr = opentype_features.next(nullptr); ftr != nullptr; ftr = opentype_features.next(ftr)) {
		String name = TS->tag_to_name(*ftr);
		p_list->push_back(PropertyInfo(Variant::FLOAT, "opentype_features/" + name));
	}
	p_list->push_back(PropertyInfo(Variant::NIL, "opentype_features/_new", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
}

void TextEdit::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_gui_input"), &TextEdit::_gui_input);
	ClassDB::bind_method(D_METHOD("_cursor_changed_emit"), &TextEdit::_cursor_changed_emit);
	ClassDB::bind_method(D_METHOD("_text_changed_emit"), &TextEdit::_text_changed_emit);
	ClassDB::bind_method(D_METHOD("_update_wrap_at", "force"), &TextEdit::_update_wrap_at, DEFVAL(false));

	BIND_ENUM_CONSTANT(SEARCH_MATCH_CASE);
	BIND_ENUM_CONSTANT(SEARCH_WHOLE_WORDS);
	BIND_ENUM_CONSTANT(SEARCH_BACKWARDS);

	BIND_ENUM_CONSTANT(SELECTION_MODE_NONE);
	BIND_ENUM_CONSTANT(SELECTION_MODE_SHIFT);
	BIND_ENUM_CONSTANT(SELECTION_MODE_POINTER);
	BIND_ENUM_CONSTANT(SELECTION_MODE_WORD);
	BIND_ENUM_CONSTANT(SELECTION_MODE_LINE);

	/*
	ClassDB::bind_method(D_METHOD("delete_char"),&TextEdit::delete_char);
	ClassDB::bind_method(D_METHOD("delete_line"),&TextEdit::delete_line);
*/

	ClassDB::bind_method(D_METHOD("get_draw_control_chars"), &TextEdit::get_draw_control_chars);
	ClassDB::bind_method(D_METHOD("set_draw_control_chars", "enable"), &TextEdit::set_draw_control_chars);
	ClassDB::bind_method(D_METHOD("set_text_direction", "direction"), &TextEdit::set_text_direction);
	ClassDB::bind_method(D_METHOD("get_text_direction"), &TextEdit::get_text_direction);
	ClassDB::bind_method(D_METHOD("set_opentype_feature", "tag", "value"), &TextEdit::set_opentype_feature);
	ClassDB::bind_method(D_METHOD("get_opentype_feature", "tag"), &TextEdit::get_opentype_feature);
	ClassDB::bind_method(D_METHOD("clear_opentype_features"), &TextEdit::clear_opentype_features);
	ClassDB::bind_method(D_METHOD("set_language", "language"), &TextEdit::set_language);
	ClassDB::bind_method(D_METHOD("get_language"), &TextEdit::get_language);

	ClassDB::bind_method(D_METHOD("set_text", "text"), &TextEdit::set_text);
	ClassDB::bind_method(D_METHOD("insert_text_at_cursor", "text"), &TextEdit::insert_text_at_cursor);

	ClassDB::bind_method(D_METHOD("get_line_count"), &TextEdit::get_line_count);
	ClassDB::bind_method(D_METHOD("get_text"), &TextEdit::get_text);
	ClassDB::bind_method(D_METHOD("get_line", "line"), &TextEdit::get_line);
	ClassDB::bind_method(D_METHOD("get_visible_line_count"), &TextEdit::get_total_visible_rows);
	ClassDB::bind_method(D_METHOD("set_line", "line", "new_text"), &TextEdit::set_line);

	ClassDB::bind_method(D_METHOD("set_structured_text_bidi_override", "parser"), &TextEdit::set_structured_text_bidi_override);
	ClassDB::bind_method(D_METHOD("get_structured_text_bidi_override"), &TextEdit::get_structured_text_bidi_override);
	ClassDB::bind_method(D_METHOD("set_structured_text_bidi_override_options", "args"), &TextEdit::set_structured_text_bidi_override_options);
	ClassDB::bind_method(D_METHOD("get_structured_text_bidi_override_options"), &TextEdit::get_structured_text_bidi_override_options);

	ClassDB::bind_method(D_METHOD("center_viewport_to_cursor"), &TextEdit::center_viewport_to_cursor);
	ClassDB::bind_method(D_METHOD("cursor_set_column", "column", "adjust_viewport"), &TextEdit::cursor_set_column, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("cursor_set_line", "line", "adjust_viewport", "can_be_hidden", "wrap_index"), &TextEdit::cursor_set_line, DEFVAL(true), DEFVAL(true), DEFVAL(0));

	ClassDB::bind_method(D_METHOD("cursor_get_column"), &TextEdit::cursor_get_column);
	ClassDB::bind_method(D_METHOD("cursor_get_line"), &TextEdit::cursor_get_line);
	ClassDB::bind_method(D_METHOD("cursor_set_blink_enabled", "enable"), &TextEdit::cursor_set_blink_enabled);
	ClassDB::bind_method(D_METHOD("cursor_get_blink_enabled"), &TextEdit::cursor_get_blink_enabled);
	ClassDB::bind_method(D_METHOD("cursor_set_blink_speed", "blink_speed"), &TextEdit::cursor_set_blink_speed);
	ClassDB::bind_method(D_METHOD("cursor_get_blink_speed"), &TextEdit::cursor_get_blink_speed);
	ClassDB::bind_method(D_METHOD("cursor_set_block_mode", "enable"), &TextEdit::cursor_set_block_mode);
	ClassDB::bind_method(D_METHOD("cursor_is_block_mode"), &TextEdit::cursor_is_block_mode);

	ClassDB::bind_method(D_METHOD("set_mid_grapheme_caret_enabled", "enabled"), &TextEdit::set_mid_grapheme_caret_enabled);
	ClassDB::bind_method(D_METHOD("get_mid_grapheme_caret_enabled"), &TextEdit::get_mid_grapheme_caret_enabled);

	ClassDB::bind_method(D_METHOD("set_right_click_moves_caret", "enable"), &TextEdit::set_right_click_moves_caret);
	ClassDB::bind_method(D_METHOD("is_right_click_moving_caret"), &TextEdit::is_right_click_moving_caret);

	ClassDB::bind_method(D_METHOD("get_selection_mode"), &TextEdit::get_selection_mode);
	ClassDB::bind_method(D_METHOD("set_selection_mode", "mode", "line", "column"), &TextEdit::set_selection_mode, DEFVAL(-1), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("get_selection_line"), &TextEdit::get_selection_line);
	ClassDB::bind_method(D_METHOD("get_selection_column"), &TextEdit::get_selection_column);

	ClassDB::bind_method(D_METHOD("set_readonly", "enable"), &TextEdit::set_readonly);
	ClassDB::bind_method(D_METHOD("is_readonly"), &TextEdit::is_readonly);

	ClassDB::bind_method(D_METHOD("set_wrap_enabled", "enable"), &TextEdit::set_wrap_enabled);
	ClassDB::bind_method(D_METHOD("is_wrap_enabled"), &TextEdit::is_wrap_enabled);
	ClassDB::bind_method(D_METHOD("set_context_menu_enabled", "enable"), &TextEdit::set_context_menu_enabled);
	ClassDB::bind_method(D_METHOD("is_context_menu_enabled"), &TextEdit::is_context_menu_enabled);
	ClassDB::bind_method(D_METHOD("set_shortcut_keys_enabled", "enable"), &TextEdit::set_shortcut_keys_enabled);
	ClassDB::bind_method(D_METHOD("is_shortcut_keys_enabled"), &TextEdit::is_shortcut_keys_enabled);
	ClassDB::bind_method(D_METHOD("set_virtual_keyboard_enabled", "enable"), &TextEdit::set_virtual_keyboard_enabled);
	ClassDB::bind_method(D_METHOD("is_virtual_keyboard_enabled"), &TextEdit::is_virtual_keyboard_enabled);
	ClassDB::bind_method(D_METHOD("set_selecting_enabled", "enable"), &TextEdit::set_selecting_enabled);
	ClassDB::bind_method(D_METHOD("is_selecting_enabled"), &TextEdit::is_selecting_enabled);

	ClassDB::bind_method(D_METHOD("cut"), &TextEdit::cut);
	ClassDB::bind_method(D_METHOD("copy"), &TextEdit::copy);
	ClassDB::bind_method(D_METHOD("paste"), &TextEdit::paste);

	ClassDB::bind_method(D_METHOD("select", "from_line", "from_column", "to_line", "to_column"), &TextEdit::select);
	ClassDB::bind_method(D_METHOD("select_all"), &TextEdit::select_all);
	ClassDB::bind_method(D_METHOD("deselect"), &TextEdit::deselect);

	ClassDB::bind_method(D_METHOD("is_selection_active"), &TextEdit::is_selection_active);
	ClassDB::bind_method(D_METHOD("get_selection_from_line"), &TextEdit::get_selection_from_line);
	ClassDB::bind_method(D_METHOD("get_selection_from_column"), &TextEdit::get_selection_from_column);
	ClassDB::bind_method(D_METHOD("get_selection_to_line"), &TextEdit::get_selection_to_line);
	ClassDB::bind_method(D_METHOD("get_selection_to_column"), &TextEdit::get_selection_to_column);
	ClassDB::bind_method(D_METHOD("get_selection_text"), &TextEdit::get_selection_text);
	ClassDB::bind_method(D_METHOD("get_word_under_cursor"), &TextEdit::get_word_under_cursor);
	ClassDB::bind_method(D_METHOD("search", "key", "flags", "from_line", "from_column"), &TextEdit::_search_bind);

	ClassDB::bind_method(D_METHOD("undo"), &TextEdit::undo);
	ClassDB::bind_method(D_METHOD("redo"), &TextEdit::redo);
	ClassDB::bind_method(D_METHOD("clear_undo_history"), &TextEdit::clear_undo_history);

	ClassDB::bind_method(D_METHOD("set_draw_tabs"), &TextEdit::set_draw_tabs);
	ClassDB::bind_method(D_METHOD("is_drawing_tabs"), &TextEdit::is_drawing_tabs);
	ClassDB::bind_method(D_METHOD("set_draw_spaces"), &TextEdit::set_draw_spaces);
	ClassDB::bind_method(D_METHOD("is_drawing_spaces"), &TextEdit::is_drawing_spaces);

	ClassDB::bind_method(D_METHOD("set_hiding_enabled", "enable"), &TextEdit::set_hiding_enabled);
	ClassDB::bind_method(D_METHOD("is_hiding_enabled"), &TextEdit::is_hiding_enabled);
	ClassDB::bind_method(D_METHOD("set_line_as_hidden", "line", "enable"), &TextEdit::set_line_as_hidden);
	ClassDB::bind_method(D_METHOD("is_line_hidden", "line"), &TextEdit::is_line_hidden);
	ClassDB::bind_method(D_METHOD("fold_all_lines"), &TextEdit::fold_all_lines);
	ClassDB::bind_method(D_METHOD("unhide_all_lines"), &TextEdit::unhide_all_lines);
	ClassDB::bind_method(D_METHOD("fold_line", "line"), &TextEdit::fold_line);
	ClassDB::bind_method(D_METHOD("unfold_line", "line"), &TextEdit::unfold_line);
	ClassDB::bind_method(D_METHOD("toggle_fold_line", "line"), &TextEdit::toggle_fold_line);
	ClassDB::bind_method(D_METHOD("can_fold", "line"), &TextEdit::can_fold);
	ClassDB::bind_method(D_METHOD("is_folded", "line"), &TextEdit::is_folded);

	ClassDB::bind_method(D_METHOD("set_highlight_all_occurrences", "enable"), &TextEdit::set_highlight_all_occurrences);
	ClassDB::bind_method(D_METHOD("is_highlight_all_occurrences_enabled"), &TextEdit::is_highlight_all_occurrences_enabled);

	ClassDB::bind_method(D_METHOD("set_override_selected_font_color", "override"), &TextEdit::set_override_selected_font_color);
	ClassDB::bind_method(D_METHOD("is_overriding_selected_font_color"), &TextEdit::is_overriding_selected_font_color);

	ClassDB::bind_method(D_METHOD("set_syntax_highlighter", "syntax_highlighter"), &TextEdit::set_syntax_highlighter);
	ClassDB::bind_method(D_METHOD("get_syntax_highlighter"), &TextEdit::get_syntax_highlighter);

	/* Gutters. */
	BIND_ENUM_CONSTANT(GUTTER_TYPE_STRING);
	BIND_ENUM_CONSTANT(GUTTER_TPYE_ICON);
	BIND_ENUM_CONSTANT(GUTTER_TPYE_CUSTOM);

	ClassDB::bind_method(D_METHOD("add_gutter", "at"), &TextEdit::add_gutter, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("remove_gutter", "gutter"), &TextEdit::remove_gutter);
	ClassDB::bind_method(D_METHOD("get_gutter_count"), &TextEdit::get_gutter_count);
	ClassDB::bind_method(D_METHOD("set_gutter_name", "gutter", "name"), &TextEdit::set_gutter_name);
	ClassDB::bind_method(D_METHOD("get_gutter_name", "gutter"), &TextEdit::get_gutter_name);
	ClassDB::bind_method(D_METHOD("set_gutter_type", "gutter", "type"), &TextEdit::set_gutter_type);
	ClassDB::bind_method(D_METHOD("get_gutter_type", "gutter"), &TextEdit::get_gutter_type);
	ClassDB::bind_method(D_METHOD("set_gutter_width", "gutter", "width"), &TextEdit::set_gutter_width);
	ClassDB::bind_method(D_METHOD("get_gutter_width", "gutter"), &TextEdit::get_gutter_width);
	ClassDB::bind_method(D_METHOD("set_gutter_draw", "gutter", "draw"), &TextEdit::set_gutter_draw);
	ClassDB::bind_method(D_METHOD("is_gutter_drawn", "gutter"), &TextEdit::is_gutter_drawn);
	ClassDB::bind_method(D_METHOD("set_gutter_clickable", "gutter", "clickable"), &TextEdit::set_gutter_clickable);
	ClassDB::bind_method(D_METHOD("is_gutter_clickable", "gutter"), &TextEdit::is_gutter_clickable);
	ClassDB::bind_method(D_METHOD("set_gutter_overwritable", "gutter", "overwritable"), &TextEdit::set_gutter_overwritable);
	ClassDB::bind_method(D_METHOD("is_gutter_overwritable", "gutter"), &TextEdit::is_gutter_overwritable);
	ClassDB::bind_method(D_METHOD("set_gutter_custom_draw", "column", "object", "callback"), &TextEdit::set_gutter_custom_draw);

	// Line gutters.
	ClassDB::bind_method(D_METHOD("set_line_gutter_metadata", "line", "gutter", "metadata"), &TextEdit::set_line_gutter_metadata);
	ClassDB::bind_method(D_METHOD("get_line_gutter_metadata", "line", "gutter"), &TextEdit::get_line_gutter_metadata);
	ClassDB::bind_method(D_METHOD("set_line_gutter_text", "line", "gutter", "text"), &TextEdit::set_line_gutter_text);
	ClassDB::bind_method(D_METHOD("get_line_gutter_text", "line", "gutter"), &TextEdit::get_line_gutter_text);
	ClassDB::bind_method(D_METHOD("set_line_gutter_icon", "line", "gutter", "icon"), &TextEdit::set_line_gutter_icon);
	ClassDB::bind_method(D_METHOD("get_line_gutter_icon", "line", "gutter"), &TextEdit::get_line_gutter_icon);
	ClassDB::bind_method(D_METHOD("set_line_gutter_item_color", "line", "gutter", "color"), &TextEdit::set_line_gutter_item_color);
	ClassDB::bind_method(D_METHOD("get_line_gutter_item_color", "line", "gutter"), &TextEdit::get_line_gutter_item_color);
	ClassDB::bind_method(D_METHOD("set_line_gutter_clickable", "line", "gutter", "clickable"), &TextEdit::set_line_gutter_clickable);
	ClassDB::bind_method(D_METHOD("is_line_gutter_clickable", "line", "gutter"), &TextEdit::is_line_gutter_clickable);

	// Line style
	ClassDB::bind_method(D_METHOD("set_line_background_color", "line", "color"), &TextEdit::set_line_background_color);
	ClassDB::bind_method(D_METHOD("get_line_background_color", "line"), &TextEdit::get_line_background_color);

	ClassDB::bind_method(D_METHOD("set_highlight_current_line", "enabled"), &TextEdit::set_highlight_current_line);
	ClassDB::bind_method(D_METHOD("is_highlight_current_line_enabled"), &TextEdit::is_highlight_current_line_enabled);

	ClassDB::bind_method(D_METHOD("set_smooth_scroll_enable", "enable"), &TextEdit::set_smooth_scroll_enabled);
	ClassDB::bind_method(D_METHOD("is_smooth_scroll_enabled"), &TextEdit::is_smooth_scroll_enabled);
	ClassDB::bind_method(D_METHOD("set_v_scroll_speed", "speed"), &TextEdit::set_v_scroll_speed);
	ClassDB::bind_method(D_METHOD("get_v_scroll_speed"), &TextEdit::get_v_scroll_speed);
	ClassDB::bind_method(D_METHOD("set_v_scroll", "value"), &TextEdit::set_v_scroll);
	ClassDB::bind_method(D_METHOD("get_v_scroll"), &TextEdit::get_v_scroll);
	ClassDB::bind_method(D_METHOD("set_h_scroll", "value"), &TextEdit::set_h_scroll);
	ClassDB::bind_method(D_METHOD("get_h_scroll"), &TextEdit::get_h_scroll);

	ClassDB::bind_method(D_METHOD("menu_option", "option"), &TextEdit::menu_option);
	ClassDB::bind_method(D_METHOD("get_menu"), &TextEdit::get_menu);

	ClassDB::bind_method(D_METHOD("draw_minimap", "draw"), &TextEdit::set_draw_minimap);
	ClassDB::bind_method(D_METHOD("is_drawing_minimap"), &TextEdit::is_drawing_minimap);
	ClassDB::bind_method(D_METHOD("set_minimap_width", "width"), &TextEdit::set_minimap_width);
	ClassDB::bind_method(D_METHOD("get_minimap_width"), &TextEdit::get_minimap_width);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "text", PROPERTY_HINT_MULTILINE_TEXT), "set_text", "get_text");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_direction", PROPERTY_HINT_ENUM, "Auto,Left-to-Right,Right-to-Left,Inherited"), "set_text_direction", "get_text_direction");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "language"), "set_language", "get_language");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_control_chars"), "set_draw_control_chars", "get_draw_control_chars");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "readonly"), "set_readonly", "is_readonly");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "highlight_current_line"), "set_highlight_current_line", "is_highlight_current_line_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_tabs"), "set_draw_tabs", "is_drawing_tabs");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_spaces"), "set_draw_spaces", "is_drawing_spaces");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "highlight_all_occurrences"), "set_highlight_all_occurrences", "is_highlight_all_occurrences_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "override_selected_font_color"), "set_override_selected_font_color", "is_overriding_selected_font_color");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "context_menu_enabled"), "set_context_menu_enabled", "is_context_menu_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "shortcut_keys_enabled"), "set_shortcut_keys_enabled", "is_shortcut_keys_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "virtual_keyboard_enabled"), "set_virtual_keyboard_enabled", "is_virtual_keyboard_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "selecting_enabled"), "set_selecting_enabled", "is_selecting_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "smooth_scrolling"), "set_smooth_scroll_enable", "is_smooth_scroll_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "v_scroll_speed"), "set_v_scroll_speed", "get_v_scroll_speed");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "hiding_enabled"), "set_hiding_enabled", "is_hiding_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "wrap_enabled"), "set_wrap_enabled", "is_wrap_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "scroll_vertical"), "set_v_scroll", "get_v_scroll");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "scroll_horizontal"), "set_h_scroll", "get_h_scroll");

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "syntax_highlighter", PROPERTY_HINT_RESOURCE_TYPE, "SyntaxHighlighter", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_DO_NOT_SHARE_ON_DUPLICATE), "set_syntax_highlighter", "get_syntax_highlighter");

	ADD_GROUP("Minimap", "minimap_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "minimap_draw"), "draw_minimap", "is_drawing_minimap");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "minimap_width"), "set_minimap_width", "get_minimap_width");

	ADD_GROUP("Caret", "caret_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "caret_block_mode"), "cursor_set_block_mode", "cursor_is_block_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "caret_blink"), "cursor_set_blink_enabled", "cursor_get_blink_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "caret_blink_speed", PROPERTY_HINT_RANGE, "0.1,10,0.01"), "cursor_set_blink_speed", "cursor_get_blink_speed");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "caret_moving_by_right_click"), "set_right_click_moves_caret", "is_right_click_moving_caret");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "caret_mid_grapheme"), "set_mid_grapheme_caret_enabled", "get_mid_grapheme_caret_enabled");

	ADD_GROUP("Structured Text", "structured_text_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "structured_text_bidi_override", PROPERTY_HINT_ENUM, "Default,URI,File,Email,List,None,Custom"), "set_structured_text_bidi_override", "get_structured_text_bidi_override");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "structured_text_bidi_override_options"), "set_structured_text_bidi_override_options", "get_structured_text_bidi_override_options");

	ADD_SIGNAL(MethodInfo("cursor_changed"));
	ADD_SIGNAL(MethodInfo("text_changed"));
	ADD_SIGNAL(MethodInfo("lines_edited_from", PropertyInfo(Variant::INT, "from_line"), PropertyInfo(Variant::INT, "to_line")));
	ADD_SIGNAL(MethodInfo("request_completion"));
	ADD_SIGNAL(MethodInfo("gutter_clicked", PropertyInfo(Variant::INT, "line"), PropertyInfo(Variant::INT, "gutter")));
	ADD_SIGNAL(MethodInfo("gutter_added"));
	ADD_SIGNAL(MethodInfo("gutter_removed"));
	ADD_SIGNAL(MethodInfo("symbol_lookup", PropertyInfo(Variant::STRING, "symbol"), PropertyInfo(Variant::INT, "row"), PropertyInfo(Variant::INT, "column")));
	ADD_SIGNAL(MethodInfo("symbol_validate", PropertyInfo(Variant::STRING, "symbol")));

	BIND_ENUM_CONSTANT(MENU_CUT);
	BIND_ENUM_CONSTANT(MENU_COPY);
	BIND_ENUM_CONSTANT(MENU_PASTE);
	BIND_ENUM_CONSTANT(MENU_CLEAR);
	BIND_ENUM_CONSTANT(MENU_SELECT_ALL);
	BIND_ENUM_CONSTANT(MENU_UNDO);
	BIND_ENUM_CONSTANT(MENU_REDO);
	BIND_ENUM_CONSTANT(MENU_DIR_INHERITED);
	BIND_ENUM_CONSTANT(MENU_DIR_AUTO);
	BIND_ENUM_CONSTANT(MENU_DIR_LTR);
	BIND_ENUM_CONSTANT(MENU_DIR_RTL);
	BIND_ENUM_CONSTANT(MENU_DISPLAY_UCC);
	BIND_ENUM_CONSTANT(MENU_INSERT_LRM);
	BIND_ENUM_CONSTANT(MENU_INSERT_RLM);
	BIND_ENUM_CONSTANT(MENU_INSERT_LRE);
	BIND_ENUM_CONSTANT(MENU_INSERT_RLE);
	BIND_ENUM_CONSTANT(MENU_INSERT_LRO);
	BIND_ENUM_CONSTANT(MENU_INSERT_RLO);
	BIND_ENUM_CONSTANT(MENU_INSERT_PDF);
	BIND_ENUM_CONSTANT(MENU_INSERT_ALM);
	BIND_ENUM_CONSTANT(MENU_INSERT_LRI);
	BIND_ENUM_CONSTANT(MENU_INSERT_RLI);
	BIND_ENUM_CONSTANT(MENU_INSERT_FSI);
	BIND_ENUM_CONSTANT(MENU_INSERT_PDI);
	BIND_ENUM_CONSTANT(MENU_INSERT_ZWJ);
	BIND_ENUM_CONSTANT(MENU_INSERT_ZWNJ);
	BIND_ENUM_CONSTANT(MENU_INSERT_WJ);
	BIND_ENUM_CONSTANT(MENU_INSERT_SHY);
	BIND_ENUM_CONSTANT(MENU_MAX);

	GLOBAL_DEF("gui/timers/text_edit_idle_detect_sec", 3);
	ProjectSettings::get_singleton()->set_custom_property_info("gui/timers/text_edit_idle_detect_sec", PropertyInfo(Variant::FLOAT, "gui/timers/text_edit_idle_detect_sec", PROPERTY_HINT_RANGE, "0,10,0.01,or_greater")); // No negative numbers.
	GLOBAL_DEF("gui/common/text_edit_undo_stack_max_size", 1024);
	ProjectSettings::get_singleton()->set_custom_property_info("gui/common/text_edit_undo_stack_max_size", PropertyInfo(Variant::INT, "gui/common/text_edit_undo_stack_max_size", PROPERTY_HINT_RANGE, "0,10000,1,or_greater")); // No negative numbers.
}

TextEdit::TextEdit() {
	clear();
	set_focus_mode(FOCUS_ALL);
	_update_caches();
	set_default_cursor_shape(CURSOR_IBEAM);

	text.set_indent_size(indent_size);
	text.clear();

	h_scroll = memnew(HScrollBar);
	v_scroll = memnew(VScrollBar);

	add_child(h_scroll);
	add_child(v_scroll);

	h_scroll->connect("value_changed", callable_mp(this, &TextEdit::_scroll_moved));
	v_scroll->connect("value_changed", callable_mp(this, &TextEdit::_scroll_moved));

	v_scroll->connect("scrolling", callable_mp(this, &TextEdit::_v_scroll_input));

	caret_blink_timer = memnew(Timer);
	add_child(caret_blink_timer);
	caret_blink_timer->set_wait_time(0.65);
	caret_blink_timer->connect("timeout", callable_mp(this, &TextEdit::_toggle_draw_caret));
	cursor_set_blink_enabled(false);

	idle_detect = memnew(Timer);
	add_child(idle_detect);
	idle_detect->set_one_shot(true);
	idle_detect->set_wait_time(GLOBAL_GET("gui/timers/text_edit_idle_detect_sec"));
	idle_detect->connect("timeout", callable_mp(this, &TextEdit::_push_current_op));

	click_select_held = memnew(Timer);
	add_child(click_select_held);
	click_select_held->set_wait_time(0.05);
	click_select_held->connect("timeout", callable_mp(this, &TextEdit::_click_selection_held));

	undo_stack_max_size = GLOBAL_GET("gui/common/text_edit_undo_stack_max_size");

	menu = memnew(PopupMenu);
	add_child(menu);

	menu_dir = memnew(PopupMenu);
	menu_dir->set_name("DirMenu");
	menu_dir->add_radio_check_item(RTR("Same as layout direction"), MENU_DIR_INHERITED);
	menu_dir->add_radio_check_item(RTR("Auto-detect direction"), MENU_DIR_AUTO);
	menu_dir->add_radio_check_item(RTR("Left-to-right"), MENU_DIR_LTR);
	menu_dir->add_radio_check_item(RTR("Right-to-left"), MENU_DIR_RTL);
	menu_dir->set_item_checked(menu_dir->get_item_index(MENU_DIR_INHERITED), true);
	menu->add_child(menu_dir);

	menu_ctl = memnew(PopupMenu);
	menu_ctl->set_name("CTLMenu");
	menu_ctl->add_item(RTR("Left-to-right mark (LRM)"), MENU_INSERT_LRM);
	menu_ctl->add_item(RTR("Right-to-left mark (RLM)"), MENU_INSERT_RLM);
	menu_ctl->add_item(RTR("Start of left-to-right embedding (LRE)"), MENU_INSERT_LRE);
	menu_ctl->add_item(RTR("Start of right-to-left embedding (RLE)"), MENU_INSERT_RLE);
	menu_ctl->add_item(RTR("Start of left-to-right override (LRO)"), MENU_INSERT_LRO);
	menu_ctl->add_item(RTR("Start of right-to-left override (RLO)"), MENU_INSERT_RLO);
	menu_ctl->add_item(RTR("Pop direction formatting (PDF)"), MENU_INSERT_PDF);
	menu_ctl->add_separator();
	menu_ctl->add_item(RTR("Arabic letter mark (ALM)"), MENU_INSERT_ALM);
	menu_ctl->add_item(RTR("Left-to-right isolate (LRI)"), MENU_INSERT_LRI);
	menu_ctl->add_item(RTR("Right-to-left isolate (RLI)"), MENU_INSERT_RLI);
	menu_ctl->add_item(RTR("First strong isolate (FSI)"), MENU_INSERT_FSI);
	menu_ctl->add_item(RTR("Pop direction isolate (PDI)"), MENU_INSERT_PDI);
	menu_ctl->add_separator();
	menu_ctl->add_item(RTR("Zero width joiner (ZWJ)"), MENU_INSERT_ZWJ);
	menu_ctl->add_item(RTR("Zero width non-joiner (ZWNJ)"), MENU_INSERT_ZWNJ);
	menu_ctl->add_item(RTR("Word joiner (WJ)"), MENU_INSERT_WJ);
	menu_ctl->add_item(RTR("Soft hyphen (SHY)"), MENU_INSERT_SHY);
	menu->add_child(menu_ctl);

	set_readonly(false);
	menu->connect("id_pressed", callable_mp(this, &TextEdit::menu_option));
	menu_dir->connect("id_pressed", callable_mp(this, &TextEdit::menu_option));
	menu_ctl->connect("id_pressed", callable_mp(this, &TextEdit::menu_option));
}

TextEdit::~TextEdit() {
}

///////////////////////////////////////////////////////////////////////////////

Dictionary TextEdit::_get_line_syntax_highlighting(int p_line) {
	return syntax_highlighter.is_null() && !setting_text ? Dictionary() : syntax_highlighter->get_line_syntax_highlighting(p_line);
}
