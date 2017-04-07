/*************************************************************************/
/*  line_edit.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "line_edit.h"
#include "label.h"
#include "os/keyboard.h"
#include "os/os.h"
#include "print_string.h"
#include "translation.h"
#ifdef TOOLS_ENABLED
#include "editor/editor_settings.h"
#endif

static bool _is_text_char(CharType c) {

	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}

void LineEdit::_gui_input(InputEvent p_event) {

	switch (p_event.type) {

		case InputEvent::MOUSE_BUTTON: {

			const InputEventMouseButton &b = p_event.mouse_button;

			if (b.pressed && b.button_index == BUTTON_RIGHT) {
				menu->set_pos(get_global_transform().xform(get_local_mouse_pos()));
				menu->set_size(Vector2(1, 1));
				menu->popup();
				grab_focus();
				return;
			}

			if (b.button_index != BUTTON_LEFT)
				break;

			_reset_caret_blink_timer();
			if (b.pressed) {

				shift_selection_check_pre(b.mod.shift);

				set_cursor_at_pixel_pos(b.x);

				if (b.mod.shift) {

					selection_fill_at_cursor();
					selection.creating = true;

				} else {

					if (b.doubleclick) {

						selection.enabled = true;
						selection.begin = 0;
						selection.end = text.length();
						selection.doubleclick = true;
					}

					selection.drag_attempt = false;

					if ((cursor_pos < selection.begin) || (cursor_pos > selection.end) || !selection.enabled) {

						selection_clear();
						selection.cursor_start = cursor_pos;
						selection.creating = true;
					} else if (selection.enabled) {

						selection.drag_attempt = true;
					}
				}

				update();

			} else {

				if ((!selection.creating) && (!selection.doubleclick)) {
					selection_clear();
				}
				selection.creating = false;
				selection.doubleclick = false;

				if (OS::get_singleton()->has_virtual_keyboard())
					OS::get_singleton()->show_virtual_keyboard(text, get_global_rect());
			}

			update();
		} break;
		case InputEvent::MOUSE_MOTION: {

			const InputEventMouseMotion &m = p_event.mouse_motion;

			if (m.button_mask & BUTTON_LEFT) {

				if (selection.creating) {
					set_cursor_at_pixel_pos(m.x);
					selection_fill_at_cursor();
				}
			}

		} break;
		case InputEvent::KEY: {

			const InputEventKey &k = p_event.key;

			if (!k.pressed)
				return;
			unsigned int code = k.scancode;

			if (k.mod.command) {

				bool handled = true;

				switch (code) {

					case (KEY_X): { // CUT

						if (editable) {
							cut_text();
						}

					} break;

					case (KEY_C): { // COPY

						copy_text();

					} break;

					case (KEY_V): { // PASTE

						if (editable) {

							paste_text();
						}

					} break;

					case (KEY_Z): { // Simple One level undo

						if (editable) {

							undo();
						}

					} break;

					case (KEY_U): { // Delete from start to cursor

						if (editable) {

							selection_clear();
							undo_text = text;
							text = text.substr(cursor_pos, text.length() - cursor_pos);

							Ref<Font> font = get_font("font");

							cached_width = 0;
							if (font != NULL) {
								for (int i = 0; i < text.length(); i++)
									cached_width += font->get_char_size(text[i]).width;
							}

							set_cursor_pos(0);
							_text_changed();
						}

					} break;

					case (KEY_Y): { // PASTE (Yank for unix users)

						if (editable) {

							paste_text();
						}

					} break;
					case (KEY_K): { // Delete from cursor_pos to end

						if (editable) {

							selection_clear();
							undo_text = text;
							text = text.substr(0, cursor_pos);
							_text_changed();
						}

					} break;
					case (KEY_A): { //Select All
						select();
					} break;
					default: { handled = false; }
				}

				if (handled) {
					accept_event();
					return;
				}
			}

			_reset_caret_blink_timer();
			if (!k.mod.meta) {

				bool handled = true;
				switch (code) {

					case KEY_ENTER:
					case KEY_RETURN: {

						emit_signal("text_entered", text);
						if (OS::get_singleton()->has_virtual_keyboard())
							OS::get_singleton()->hide_virtual_keyboard();

						return;
					} break;

					case KEY_BACKSPACE: {

						if (!editable)
							break;

						if (selection.enabled) {
							undo_text = text;
							selection_delete();
							break;
						}

#ifdef APPLE_STYLE_KEYS
						if (k.mod.alt) {
#else
						if (k.mod.alt) {
							handled = false;
							break;
						} else if (k.mod.command) {
#endif
							int cc = cursor_pos;
							bool prev_char = false;

							while (cc > 0) {
								bool ischar = _is_text_char(text[cc - 1]);

								if (prev_char && !ischar)
									break;

								prev_char = ischar;
								cc--;
							}

							delete_text(cc, cursor_pos);

							set_cursor_pos(cc);

						} else {
							undo_text = text;
							delete_char();
						}

					} break;
					case KEY_KP_4: {
						if (k.unicode != 0) {
							handled = false;
							break;
						}
						// numlock disabled. fallthrough to key_left
					}
					case KEY_LEFT: {

#ifndef APPLE_STYLE_KEYS
						if (!k.mod.alt)
#endif
							shift_selection_check_pre(k.mod.shift);

#ifdef APPLE_STYLE_KEYS
						if (k.mod.command) {
							set_cursor_pos(0);
						} else if (k.mod.alt) {

#else
						if (k.mod.alt) {
							handled = false;
							break;
						} else if (k.mod.command) {
#endif
							bool prev_char = false;
							int cc = cursor_pos;

							while (cc > 0) {
								bool ischar = _is_text_char(text[cc - 1]);

								if (prev_char && !ischar)
									break;

								prev_char = ischar;
								cc--;
							}

							set_cursor_pos(cc);

						} else {
							set_cursor_pos(get_cursor_pos() - 1);
						}

						shift_selection_check_post(k.mod.shift);

					} break;
					case KEY_KP_6: {
						if (k.unicode != 0) {
							handled = false;
							break;
						}
						// numlock disabled. fallthrough to key_right
					}
					case KEY_RIGHT: {

						shift_selection_check_pre(k.mod.shift);

#ifdef APPLE_STYLE_KEYS
						if (k.mod.command) {
							set_cursor_pos(text.length());
						} else if (k.mod.alt) {
#else
						if (k.mod.alt) {
							handled = false;
							break;
						} else if (k.mod.command) {
#endif
							bool prev_char = false;
							int cc = cursor_pos;

							while (cc < text.length()) {
								bool ischar = _is_text_char(text[cc]);

								if (prev_char && !ischar)
									break;

								prev_char = ischar;
								cc++;
							}

							set_cursor_pos(cc);

						} else {
							set_cursor_pos(get_cursor_pos() + 1);
						}

						shift_selection_check_post(k.mod.shift);

					} break;
					case KEY_DELETE: {

						if (!editable)
							break;

						if (k.mod.shift && !k.mod.command && !k.mod.alt) {
							cut_text();
							break;
						}

						if (selection.enabled) {
							undo_text = text;
							selection_delete();
							break;
						}

						int text_len = text.length();

						if (cursor_pos == text_len)
							break; // nothing to do

#ifdef APPLE_STYLE_KEYS
						if (k.mod.alt) {
#else
						if (k.mod.alt) {
							handled = false;
							break;
						} else if (k.mod.command) {
#endif
							int cc = cursor_pos;

							bool prev_char = false;

							while (cc < text.length()) {

								bool ischar = _is_text_char(text[cc]);

								if (prev_char && !ischar)
									break;
								prev_char = ischar;
								cc++;
							}

							delete_text(cursor_pos, cc);

						} else {
							undo_text = text;
							set_cursor_pos(cursor_pos + 1);
							delete_char();
						}

					} break;
					case KEY_KP_7: {
						if (k.unicode != 0) {
							handled = false;
							break;
						}
						// numlock disabled. fallthrough to key_home
					}
					case KEY_HOME: {

						shift_selection_check_pre(k.mod.shift);
						set_cursor_pos(0);
						shift_selection_check_post(k.mod.shift);
					} break;
					case KEY_KP_1: {
						if (k.unicode != 0) {
							handled = false;
							break;
						}
						// numlock disabled. fallthrough to key_end
					}
					case KEY_END: {

						shift_selection_check_pre(k.mod.shift);
						set_cursor_pos(text.length());
						shift_selection_check_post(k.mod.shift);
					} break;

					default: {

						handled = false;
					} break;
				}

				if (handled) {
					accept_event();
				} else if (!k.mod.alt && !k.mod.command) {
					if (k.unicode >= 32 && k.scancode != KEY_DELETE) {

						if (editable) {
							selection_delete();
							CharType ucodestr[2] = { (CharType)k.unicode, 0 };
							append_at_cursor(ucodestr);
							_text_changed();
							accept_event();
						}

					} else {
						return;
					}
				}

				update();
			}

			return;

		} break;
	}
}

void LineEdit::set_align(Align p_align) {

	ERR_FAIL_INDEX(p_align, 4);
	align = p_align;
	update();
}

LineEdit::Align LineEdit::get_align() const {

	return align;
}

Variant LineEdit::get_drag_data(const Point2 &p_point) {

	if (selection.drag_attempt && selection.enabled) {
		String t = text.substr(selection.begin, selection.end - selection.begin);
		Label *l = memnew(Label);
		l->set_text(t);
		set_drag_preview(l);
		return t;
	}

	return Variant();
}
bool LineEdit::can_drop_data(const Point2 &p_point, const Variant &p_data) const {

	return p_data.get_type() == Variant::STRING;
}
void LineEdit::drop_data(const Point2 &p_point, const Variant &p_data) {

	if (p_data.get_type() == Variant::STRING) {
		set_cursor_at_pixel_pos(p_point.x);
		int selected = selection.end - selection.begin;

		Ref<Font> font = get_font("font");
		if (font != NULL) {
			for (int i = selection.begin; i < selection.end; i++)
				cached_width -= font->get_char_size(text[i]).width;
		}

		text.erase(selection.begin, selected);

		append_at_cursor(p_data);
		selection.begin = cursor_pos - selected;
		selection.end = cursor_pos;
	}
}

void LineEdit::_notification(int p_what) {

	switch (p_what) {
#ifdef TOOLS_ENABLED
		case NOTIFICATION_ENTER_TREE: {
			if (get_tree()->is_editor_hint()) {
				cursor_set_blink_enabled(EDITOR_DEF("text_editor/cursor/caret_blink", false));
				cursor_set_blink_speed(EDITOR_DEF("text_editor/cursor/caret_blink_speed", 0.65));

				if (!EditorSettings::get_singleton()->is_connected("settings_changed", this, "_editor_settings_changed")) {
					EditorSettings::get_singleton()->connect("settings_changed", this, "_editor_settings_changed");
				}
			}
		} break;
#endif
		case NOTIFICATION_RESIZED: {

			set_cursor_pos(get_cursor_pos());

		} break;
		case MainLoop::NOTIFICATION_WM_FOCUS_IN: {
			window_has_focus = true;
			draw_caret = true;
			update();
		} break;
		case MainLoop::NOTIFICATION_WM_FOCUS_OUT: {
			window_has_focus = false;
			draw_caret = false;
			update();
		} break;
		case NOTIFICATION_DRAW: {

			if ((!has_focus() && !menu->has_focus()) || !window_has_focus) {
				draw_caret = false;
			}

			int width, height;

			Size2 size = get_size();
			width = size.width;
			height = size.height;

			RID ci = get_canvas_item();

			Ref<StyleBox> style = get_stylebox("normal");
			if (!is_editable())
				style = get_stylebox("read_only");

			Ref<Font> font = get_font("font");

			style->draw(ci, Rect2(Point2(), size));

			if (has_focus()) {

				get_stylebox("focus")->draw(ci, Rect2(Point2(), size));
			}

			int x_ofs = 0;

			switch (align) {

				case ALIGN_FILL:
				case ALIGN_LEFT: {

					x_ofs = style->get_offset().x;
				} break;
				case ALIGN_CENTER: {

					x_ofs = int(size.width - (cached_width)) / 2;
				} break;
				case ALIGN_RIGHT: {

					x_ofs = int(size.width - style->get_offset().x - (cached_width));
				} break;
			}

			int ofs_max = width - style->get_minimum_size().width;
			int char_ofs = window_pos;

			int y_area = height - style->get_minimum_size().height;
			int y_ofs = style->get_offset().y;

			int font_ascent = font->get_ascent();

			Color selection_color = get_color("selection_color");
			Color font_color = get_color("font_color");
			Color font_color_selected = get_color("font_color_selected");
			Color cursor_color = get_color("cursor_color");

			const String &t = text.empty() ? placeholder : text;
			// draw placeholder color
			if (text.empty())
				font_color.a *= placeholder_alpha;

			int caret_height = font->get_height() > y_area ? y_area : font->get_height();
			while (true) {

				//end of string, break!
				if (char_ofs >= t.length())
					break;

				CharType cchar = pass ? '*' : t[char_ofs];
				CharType next = pass ? '*' : t[char_ofs + 1];
				int char_width = font->get_char_size(cchar, next).width;

				// end of widget, break!
				if ((x_ofs + char_width) > ofs_max)
					break;

				bool selected = selection.enabled && char_ofs >= selection.begin && char_ofs < selection.end;

				if (selected)
					VisualServer::get_singleton()->canvas_item_add_rect(ci, Rect2(Point2(x_ofs, y_ofs), Size2(char_width, caret_height)), selection_color);

				font->draw_char(ci, Point2(x_ofs, y_ofs + font_ascent), cchar, next, selected ? font_color_selected : font_color);

				if (char_ofs == cursor_pos && draw_caret) {
					VisualServer::get_singleton()->canvas_item_add_rect(ci, Rect2(
																					Point2(x_ofs, y_ofs), Size2(1, caret_height)),
							cursor_color);
				}

				x_ofs += char_width;
				char_ofs++;
			}

			if (char_ofs == cursor_pos && draw_caret) { //may be at the end
				VisualServer::get_singleton()->canvas_item_add_rect(ci, Rect2(
																				Point2(x_ofs, y_ofs), Size2(1, caret_height)),
						cursor_color);
			}
		} break;
		case NOTIFICATION_FOCUS_ENTER: {

			if (!caret_blink_enabled) {
				draw_caret = true;
			}

			if (OS::get_singleton()->has_virtual_keyboard())
				OS::get_singleton()->show_virtual_keyboard(text, get_global_rect());

		} break;
		case NOTIFICATION_FOCUS_EXIT: {

			if (OS::get_singleton()->has_virtual_keyboard())
				OS::get_singleton()->hide_virtual_keyboard();

		} break;
	}
}

void LineEdit::copy_text() {

	if (selection.enabled) {

		OS::get_singleton()->set_clipboard(text.substr(selection.begin, selection.end - selection.begin));
	}
}

void LineEdit::cut_text() {

	if (selection.enabled) {
		undo_text = text;
		OS::get_singleton()->set_clipboard(text.substr(selection.begin, selection.end - selection.begin));
		selection_delete();
	}
}

void LineEdit::paste_text() {

	String paste_buffer = OS::get_singleton()->get_clipboard();

	if (paste_buffer != "") {

		if (selection.enabled) selection_delete();
		append_at_cursor(paste_buffer);

		_text_changed();
	}
}

void LineEdit::undo() {

	int old_cursor_pos = cursor_pos;
	text = undo_text;

	Ref<Font> font = get_font("font");

	cached_width = 0;
	for (int i = 0; i < text.length(); i++)
		cached_width += font->get_char_size(text[i]).width;

	if (old_cursor_pos > text.length()) {
		set_cursor_pos(text.length());
	} else {
		set_cursor_pos(old_cursor_pos);
	}

	_text_changed();
}

void LineEdit::shift_selection_check_pre(bool p_shift) {

	if (!selection.enabled && p_shift) {
		selection.cursor_start = cursor_pos;
	}
	if (!p_shift)
		selection_clear();
}

void LineEdit::shift_selection_check_post(bool p_shift) {

	if (p_shift)
		selection_fill_at_cursor();
}

void LineEdit::set_cursor_at_pixel_pos(int p_x) {

	Ref<Font> font = get_font("font");
	int ofs = window_pos;
	Ref<StyleBox> style = get_stylebox("normal");
	int pixel_ofs = 0;
	Size2 size = get_size();

	switch (align) {

		case ALIGN_FILL:
		case ALIGN_LEFT: {

			pixel_ofs = int(style->get_offset().x);
		} break;
		case ALIGN_CENTER: {

			pixel_ofs = int(size.width - (cached_width)) / 2;
		} break;
		case ALIGN_RIGHT: {

			pixel_ofs = int(size.width - style->get_offset().x - (cached_width));
		} break;
	}

	while (ofs < text.length()) {

		int char_w = 0;
		if (font != NULL) {
			char_w = font->get_char_size(text[ofs]).width;
		}
		pixel_ofs += char_w;

		if (pixel_ofs > p_x) { //found what we look for
			break;
		}

		ofs++;
	}

	set_cursor_pos(ofs);

	/*
	int new_cursor_pos=p_x;
	int charwidth=draw_area->get_font_char_width(' ',0);
	new_cursor_pos=( ( (new_cursor_pos-2)+ (charwidth/2) ) /charwidth );
	if (new_cursor_pos>(int)text.length()) new_cursor_pos=text.length();
	set_cursor_pos(window_pos+new_cursor_pos); */
}

bool LineEdit::cursor_get_blink_enabled() const {
	return caret_blink_enabled;
}

void LineEdit::cursor_set_blink_enabled(const bool p_enabled) {
	caret_blink_enabled = p_enabled;
	if (p_enabled) {
		caret_blink_timer->start();
	} else {
		caret_blink_timer->stop();
	}
	draw_caret = true;
}

float LineEdit::cursor_get_blink_speed() const {
	return caret_blink_timer->get_wait_time();
}

void LineEdit::cursor_set_blink_speed(const float p_speed) {
	ERR_FAIL_COND(p_speed <= 0);
	caret_blink_timer->set_wait_time(p_speed);
}

void LineEdit::_reset_caret_blink_timer() {
	if (caret_blink_enabled) {
		caret_blink_timer->stop();
		caret_blink_timer->start();
		draw_caret = true;
		update();
	}
}

void LineEdit::_toggle_draw_caret() {
	draw_caret = !draw_caret;
	if (is_visible_in_tree() && has_focus() && window_has_focus) {
		update();
	}
}

void LineEdit::delete_char() {

	if ((text.length() <= 0) || (cursor_pos == 0)) return;

	Ref<Font> font = get_font("font");
	if (font != NULL) {
		cached_width -= font->get_char_size(text[cursor_pos - 1]).width;
	}

	text.erase(cursor_pos - 1, 1);

	set_cursor_pos(get_cursor_pos() - 1);

	if (cursor_pos == window_pos) {

		//set_window_pos(cursor_pos-get_window_length());
	}

	_text_changed();
}

void LineEdit::delete_text(int p_from_column, int p_to_column) {

	undo_text = text;

	if (text.size() > 0) {
		Ref<Font> font = get_font("font");
		if (font != NULL) {
			for (int i = p_from_column; i < p_to_column; i++)
				cached_width -= font->get_char_size(text[i]).width;
		}
	} else {
		cached_width = 0;
	}

	text.erase(p_from_column, p_to_column - p_from_column);
	cursor_pos -= CLAMP(cursor_pos - p_from_column, 0, p_to_column - p_from_column);

	if (cursor_pos >= text.length()) {

		cursor_pos = text.length();
	}
	if (window_pos > cursor_pos) {

		window_pos = cursor_pos;
	}

	_text_changed();
}

void LineEdit::set_text(String p_text) {

	clear_internal();
	append_at_cursor(p_text);
	update();
	cursor_pos = 0;
	window_pos = 0;
	_text_changed();
}

void LineEdit::clear() {

	clear_internal();
	_text_changed();
}

String LineEdit::get_text() const {

	return text;
}

void LineEdit::set_placeholder(String p_text) {

	placeholder = XL_MESSAGE(p_text);
	update();
}

String LineEdit::get_placeholder() const {

	return placeholder;
}

void LineEdit::set_placeholder_alpha(float p_alpha) {

	placeholder_alpha = p_alpha;
	update();
}

float LineEdit::get_placeholder_alpha() const {

	return placeholder_alpha;
}

void LineEdit::set_cursor_pos(int p_pos) {

	if (p_pos > (int)text.length())
		p_pos = text.length();

	if (p_pos < 0)
		p_pos = 0;

	cursor_pos = p_pos;

	if (!is_inside_tree()) {

		window_pos = cursor_pos;
		return;
	}

	Ref<StyleBox> style = get_stylebox("normal");
	Ref<Font> font = get_font("font");

	if (cursor_pos < window_pos) {
		/* Adjust window if cursor goes too much to the left */
		set_window_pos(cursor_pos);
	} else if (cursor_pos > window_pos) {
		/* Adjust window if cursor goes too much to the right */
		int window_width = get_size().width - style->get_minimum_size().width;

		if (window_width < 0)
			return;
		int wp = window_pos;

		if (font.is_valid()) {

			int accum_width = 0;

			for (int i = cursor_pos; i >= window_pos; i--) {

				if (i >= text.length()) {
					accum_width = font->get_char_size(' ').width; //anything should do
				} else {
					accum_width += font->get_char_size(text[i], i + 1 < text.length() ? text[i + 1] : 0).width; //anything should do
				}
				if (accum_width >= window_width)
					break;

				wp = i;
			}
		}

		if (wp != window_pos)
			set_window_pos(wp);
	}
	update();
}

int LineEdit::get_cursor_pos() const {

	return cursor_pos;
}

void LineEdit::set_window_pos(int p_pos) {

	window_pos = p_pos;
	if (window_pos < 0) window_pos = 0;
}

void LineEdit::append_at_cursor(String p_text) {

	if ((max_length <= 0) || (text.length() + p_text.length() <= max_length)) {

		undo_text = text;

		Ref<Font> font = get_font("font");
		if (font != NULL) {
			for (int i = 0; i < p_text.length(); i++)
				cached_width += font->get_char_size(p_text[i]).width;
		} else {
			cached_width = 0;
		}

		String pre = text.substr(0, cursor_pos);
		String post = text.substr(cursor_pos, text.length() - cursor_pos);
		text = pre + p_text + post;
		set_cursor_pos(cursor_pos + p_text.length());
	}
}

void LineEdit::clear_internal() {

	cached_width = 0;
	cursor_pos = 0;
	window_pos = 0;
	undo_text = "";
	text = "";
	update();
}

Size2 LineEdit::get_minimum_size() const {

	Ref<StyleBox> style = get_stylebox("normal");
	Ref<Font> font = get_font("font");

	Size2 min = style->get_minimum_size();
	min.height += font->get_height();

	//minimum size of text
	int space_size = font->get_char_size(' ').x;
	int mstext = get_constant("minimum_spaces") * space_size;

	if (expand_to_text_length) {
		mstext = MAX(mstext, font->get_string_size(text).x + space_size); //add a spce because some fonts are too exact
	}

	min.width += mstext;

	return min;
}

/* selection */

void LineEdit::selection_clear() {

	selection.begin = 0;
	selection.end = 0;
	selection.cursor_start = 0;
	selection.enabled = false;
	selection.creating = false;
	selection.doubleclick = false;
	update();
}

void LineEdit::selection_delete() {

	if (selection.enabled)
		delete_text(selection.begin, selection.end);

	selection_clear();
}

void LineEdit::set_max_length(int p_max_length) {

	ERR_FAIL_COND(p_max_length < 0);
	max_length = p_max_length;
	set_text(text);
}

int LineEdit::get_max_length() const {

	return max_length;
}

void LineEdit::selection_fill_at_cursor() {

	int aux;

	selection.begin = cursor_pos;
	selection.end = selection.cursor_start;

	if (selection.end < selection.begin) {
		aux = selection.end;
		selection.end = selection.begin;
		selection.begin = aux;
	}

	selection.enabled = (selection.begin != selection.end);
}

void LineEdit::select_all() {

	if (!text.length())
		return;

	selection.begin = 0;
	selection.end = text.length();
	selection.enabled = true;
	update();
}
void LineEdit::set_editable(bool p_editable) {

	editable = p_editable;
	update();
}

bool LineEdit::is_editable() const {

	return editable;
}

void LineEdit::set_secret(bool p_secret) {

	pass = p_secret;
	update();
}
bool LineEdit::is_secret() const {

	return pass;
}

void LineEdit::select(int p_from, int p_to) {

	if (p_from == 0 && p_to == 0) {
		selection_clear();
		return;
	}

	int len = text.length();
	if (p_from < 0)
		p_from = 0;
	if (p_from > len)
		p_from = len;
	if (p_to < 0 || p_to > len)
		p_to = len;

	if (p_from >= p_to)
		return;

	selection.enabled = true;
	selection.begin = p_from;
	selection.end = p_to;
	selection.creating = false;
	selection.doubleclick = false;
	update();
}

bool LineEdit::is_text_field() const {

	return true;
}

void LineEdit::menu_option(int p_option) {

	switch (p_option) {
		case MENU_CUT: {
			if (editable) {
				cut_text();
			}
		} break;
		case MENU_COPY: {

			copy_text();
		} break;
		case MENU_PASTE: {
			if (editable) {
				paste_text();
			}
		} break;
		case MENU_CLEAR: {
			if (editable) {
				clear();
			}
		} break;
		case MENU_SELECT_ALL: {
			select_all();
		} break;
		case MENU_UNDO: {
			undo();
		} break;
	}
}

PopupMenu *LineEdit::get_menu() const {
	return menu;
}

#ifdef TOOLS_ENABLED
void LineEdit::_editor_settings_changed() {
	cursor_set_blink_enabled(EDITOR_DEF("text_editor/cursor/caret_blink", false));
	cursor_set_blink_speed(EDITOR_DEF("text_editor/cursor/caret_blink_speed", 0.65));
}
#endif

void LineEdit::set_expand_to_text_length(bool p_enabled) {

	expand_to_text_length = p_enabled;
	minimum_size_changed();
}

bool LineEdit::get_expand_to_text_length() const {

	return expand_to_text_length;
}

void LineEdit::_text_changed() {

	if (expand_to_text_length)
		minimum_size_changed();

	emit_signal("text_changed", text);
	_change_notify("text");
}

void LineEdit::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_toggle_draw_caret"), &LineEdit::_toggle_draw_caret);

#ifdef TOOLS_ENABLED
	ClassDB::bind_method("_editor_settings_changed", &LineEdit::_editor_settings_changed);
#endif

	ClassDB::bind_method(D_METHOD("set_align", "align"), &LineEdit::set_align);
	ClassDB::bind_method(D_METHOD("get_align"), &LineEdit::get_align);

	ClassDB::bind_method(D_METHOD("_gui_input"), &LineEdit::_gui_input);
	ClassDB::bind_method(D_METHOD("clear"), &LineEdit::clear);
	ClassDB::bind_method(D_METHOD("select_all"), &LineEdit::select_all);
	ClassDB::bind_method(D_METHOD("set_text", "text"), &LineEdit::set_text);
	ClassDB::bind_method(D_METHOD("get_text"), &LineEdit::get_text);
	ClassDB::bind_method(D_METHOD("set_placeholder", "text"), &LineEdit::set_placeholder);
	ClassDB::bind_method(D_METHOD("get_placeholder"), &LineEdit::get_placeholder);
	ClassDB::bind_method(D_METHOD("set_placeholder_alpha", "alpha"), &LineEdit::set_placeholder_alpha);
	ClassDB::bind_method(D_METHOD("get_placeholder_alpha"), &LineEdit::get_placeholder_alpha);
	ClassDB::bind_method(D_METHOD("set_cursor_pos", "pos"), &LineEdit::set_cursor_pos);
	ClassDB::bind_method(D_METHOD("get_cursor_pos"), &LineEdit::get_cursor_pos);
	ClassDB::bind_method(D_METHOD("set_expand_to_text_length", "enabled"), &LineEdit::set_expand_to_text_length);
	ClassDB::bind_method(D_METHOD("get_expand_to_text_length"), &LineEdit::get_expand_to_text_length);
	ClassDB::bind_method(D_METHOD("cursor_set_blink_enabled", "enabled"), &LineEdit::cursor_set_blink_enabled);
	ClassDB::bind_method(D_METHOD("cursor_get_blink_enabled"), &LineEdit::cursor_get_blink_enabled);
	ClassDB::bind_method(D_METHOD("cursor_set_blink_speed", "blink_speed"), &LineEdit::cursor_set_blink_speed);
	ClassDB::bind_method(D_METHOD("cursor_get_blink_speed"), &LineEdit::cursor_get_blink_speed);
	ClassDB::bind_method(D_METHOD("set_max_length", "chars"), &LineEdit::set_max_length);
	ClassDB::bind_method(D_METHOD("get_max_length"), &LineEdit::get_max_length);
	ClassDB::bind_method(D_METHOD("append_at_cursor", "text"), &LineEdit::append_at_cursor);
	ClassDB::bind_method(D_METHOD("set_editable", "enabled"), &LineEdit::set_editable);
	ClassDB::bind_method(D_METHOD("is_editable"), &LineEdit::is_editable);
	ClassDB::bind_method(D_METHOD("set_secret", "enabled"), &LineEdit::set_secret);
	ClassDB::bind_method(D_METHOD("is_secret"), &LineEdit::is_secret);
	ClassDB::bind_method(D_METHOD("select", "from", "to"), &LineEdit::select, DEFVAL(0), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("menu_option", "option"), &LineEdit::menu_option);
	ClassDB::bind_method(D_METHOD("get_menu:PopupMenu"), &LineEdit::get_menu);

	ADD_SIGNAL(MethodInfo("text_changed", PropertyInfo(Variant::STRING, "text")));
	ADD_SIGNAL(MethodInfo("text_entered", PropertyInfo(Variant::STRING, "text")));

	BIND_CONSTANT(ALIGN_LEFT);
	BIND_CONSTANT(ALIGN_CENTER);
	BIND_CONSTANT(ALIGN_RIGHT);
	BIND_CONSTANT(ALIGN_FILL);

	BIND_CONSTANT(MENU_CUT);
	BIND_CONSTANT(MENU_COPY);
	BIND_CONSTANT(MENU_PASTE);
	BIND_CONSTANT(MENU_CLEAR);
	BIND_CONSTANT(MENU_SELECT_ALL);
	BIND_CONSTANT(MENU_UNDO);
	BIND_CONSTANT(MENU_MAX);

	ADD_PROPERTYNZ(PropertyInfo(Variant::STRING, "text"), "set_text", "get_text");
	ADD_PROPERTYNZ(PropertyInfo(Variant::INT, "align", PROPERTY_HINT_ENUM, "Left,Center,Right,Fill"), "set_align", "get_align");
	ADD_PROPERTYNZ(PropertyInfo(Variant::INT, "max_length"), "set_max_length", "get_max_length");
	ADD_PROPERTYNO(PropertyInfo(Variant::BOOL, "editable"), "set_editable", "is_editable");
	ADD_PROPERTYNZ(PropertyInfo(Variant::BOOL, "secret"), "set_secret", "is_secret");
	ADD_PROPERTYNO(PropertyInfo(Variant::BOOL, "expand_to_len"), "set_expand_to_text_length", "get_expand_to_text_length");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "focus_mode", PROPERTY_HINT_ENUM, "None,Click,All"), "set_focus_mode", "get_focus_mode");
	ADD_GROUP("Placeholder", "placeholder_");
	ADD_PROPERTYNZ(PropertyInfo(Variant::STRING, "placeholder_text"), "set_placeholder", "get_placeholder");
	ADD_PROPERTYNZ(PropertyInfo(Variant::REAL, "placeholder_alpha", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_placeholder_alpha", "get_placeholder_alpha");
	ADD_GROUP("Caret", "caret_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "caret_blink"), "cursor_set_blink_enabled", "cursor_get_blink_enabled");
	ADD_PROPERTYNZ(PropertyInfo(Variant::REAL, "caret_blink_speed", PROPERTY_HINT_RANGE, "0.1,10,0.1"), "cursor_set_blink_speed", "cursor_get_blink_speed");
}

LineEdit::LineEdit() {

	align = ALIGN_LEFT;
	cached_width = 0;
	cursor_pos = 0;
	window_pos = 0;
	window_has_focus = true;
	max_length = 0;
	pass = false;
	placeholder_alpha = 0.6;

	selection_clear();
	set_focus_mode(FOCUS_ALL);
	editable = true;
	set_default_cursor_shape(CURSOR_IBEAM);
	set_mouse_filter(MOUSE_FILTER_STOP);

	draw_caret = true;
	caret_blink_enabled = false;
	caret_blink_timer = memnew(Timer);
	add_child(caret_blink_timer);
	caret_blink_timer->set_wait_time(0.65);
	caret_blink_timer->connect("timeout", this, "_toggle_draw_caret");
	cursor_set_blink_enabled(false);

	menu = memnew(PopupMenu);
	add_child(menu);
	menu->add_item(TTR("Cut"), MENU_CUT, KEY_MASK_CMD | KEY_X);
	menu->add_item(TTR("Copy"), MENU_COPY, KEY_MASK_CMD | KEY_C);
	menu->add_item(TTR("Paste"), MENU_PASTE, KEY_MASK_CMD | KEY_V);
	menu->add_separator();
	menu->add_item(TTR("Select All"), MENU_SELECT_ALL, KEY_MASK_CMD | KEY_A);
	menu->add_item(TTR("Clear"), MENU_CLEAR);
	menu->add_separator();
	menu->add_item(TTR("Undo"), MENU_UNDO, KEY_MASK_CMD | KEY_Z);
	menu->connect("id_pressed", this, "menu_option");
	expand_to_text_length = false;
}

LineEdit::~LineEdit() {
}
