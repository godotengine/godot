/*************************************************************************/
/*  line_edit.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/object/message_queue.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/string/translation.h"
#include "label.h"
#include "servers/display_server.h"
#include "servers/text_server.h"
#ifdef TOOLS_ENABLED
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#endif
#include "scene/main/window.h"

void LineEdit::_gui_input(Ref<InputEvent> p_event) {
	Ref<InputEventMouseButton> b = p_event;

	if (b.is_valid()) {
		if (ime_text.length() != 0) {
			// Ignore mouse clicks in IME input mode.
			return;
		}
		if (b->is_pressed() && b->get_button_index() == BUTTON_RIGHT && context_menu_enabled) {
			menu->set_position(get_screen_transform().xform(get_local_mouse_position()));
			menu->set_size(Vector2(1, 1));
			//menu->set_scale(get_global_transform().get_scale());
			menu->popup();
			grab_focus();
			accept_event();
			return;
		}

		if (b->get_button_index() != BUTTON_LEFT) {
			return;
		}

		_reset_caret_blink_timer();
		if (b->is_pressed()) {
			accept_event(); //don't pass event further when clicked on text field
			if (!text.empty() && is_editable() && _is_over_clear_button(b->get_position())) {
				clear_button_status.press_attempt = true;
				clear_button_status.pressing_inside = true;
				update();
				return;
			}

			shift_selection_check_pre(b->get_shift());

			set_cursor_at_pixel_pos(b->get_position().x);

			if (b->get_shift()) {
				selection_fill_at_cursor();
				selection.creating = true;

			} else {
				if (b->is_doubleclick() && selecting_enabled) {
					selection.enabled = true;
					selection.begin = 0;
					selection.end = text.length();
					selection.doubleclick = true;
				}

				selection.drag_attempt = false;

				if ((cursor_pos < selection.begin) || (cursor_pos > selection.end) || !selection.enabled) {
					deselect();
					selection.cursor_start = cursor_pos;
					selection.creating = true;
				} else if (selection.enabled) {
					selection.drag_attempt = true;
				}
			}

			update();

		} else {
			if (!text.empty() && is_editable() && clear_button_enabled) {
				bool press_attempt = clear_button_status.press_attempt;
				clear_button_status.press_attempt = false;
				if (press_attempt && clear_button_status.pressing_inside && _is_over_clear_button(b->get_position())) {
					clear();
					return;
				}
			}

			if ((!selection.creating) && (!selection.doubleclick)) {
				deselect();
			}
			selection.creating = false;
			selection.doubleclick = false;

			if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_VIRTUAL_KEYBOARD) && virtual_keyboard_enabled) {
				if (selection.enabled) {
					DisplayServer::get_singleton()->virtual_keyboard_show(text, get_global_rect(), false, max_length, selection.begin, selection.end);
				} else {
					DisplayServer::get_singleton()->virtual_keyboard_show(text, get_global_rect(), false, max_length, cursor_pos);
				}
			}
		}

		update();
	}

	Ref<InputEventMouseMotion> m = p_event;

	if (m.is_valid()) {
		if (!text.empty() && is_editable() && clear_button_enabled) {
			bool last_press_inside = clear_button_status.pressing_inside;
			clear_button_status.pressing_inside = clear_button_status.press_attempt && _is_over_clear_button(m->get_position());
			if (last_press_inside != clear_button_status.pressing_inside) {
				update();
			}
		}

		if (m->get_button_mask() & BUTTON_LEFT) {
			if (selection.creating) {
				set_cursor_at_pixel_pos(m->get_position().x);
				selection_fill_at_cursor();
			}
		}
	}

	Ref<InputEventKey> k = p_event;

	if (k.is_valid()) {
		if (!k->is_pressed()) {
			return;
		}

#ifdef APPLE_STYLE_KEYS
		if (k->get_control() && !k->get_shift() && !k->get_alt() && !k->get_command()) {
			uint32_t remap_key = KEY_UNKNOWN;
			switch (k->get_keycode()) {
				case KEY_F: {
					remap_key = KEY_RIGHT;
				} break;
				case KEY_B: {
					remap_key = KEY_LEFT;
				} break;
				case KEY_P: {
					remap_key = KEY_UP;
				} break;
				case KEY_N: {
					remap_key = KEY_DOWN;
				} break;
				case KEY_D: {
					remap_key = KEY_DELETE;
				} break;
				case KEY_H: {
					remap_key = KEY_BACKSPACE;
				} break;
				case KEY_A: {
					remap_key = KEY_HOME;
				} break;
				case KEY_E: {
					remap_key = KEY_END;
				} break;
			}

			if (remap_key != KEY_UNKNOWN) {
				k->set_keycode(remap_key);
				k->set_control(false);
			}
		}
#endif

		unsigned int code = k->get_keycode();

		if (k->get_command() && is_shortcut_keys_enabled()) {
			bool handled = true;

			switch (code) {
				case (KEY_QUOTELEFT): { // Swap current input direction (primary cursor)

					if (input_direction == TEXT_DIRECTION_LTR) {
						input_direction = TEXT_DIRECTION_RTL;
					} else {
						input_direction = TEXT_DIRECTION_LTR;
					}
					set_cursor_position(get_cursor_position());
					update();

				} break;

				case (KEY_X): { // CUT.

					if (editable) {
						cut_text();
					}

				} break;

				case (KEY_C): { // COPY.

					copy_text();

				} break;

				case (KEY_V): { // PASTE.

					if (editable) {
						paste_text();
					}

				} break;

				case (KEY_Z): { // Undo/redo.
					if (editable) {
						if (k->get_shift()) {
							redo();
						} else {
							undo();
						}
					}
				} break;

				case (KEY_U): { // Delete from start to cursor.

					if (editable) {
						deselect();
						text = text.substr(cursor_pos, text.length() - cursor_pos);
						_shape();
						set_cursor_position(0);
						_text_changed();
					}

				} break;

				case (KEY_Y): { // PASTE (Yank for unix users).

					if (editable) {
						paste_text();
					}

				} break;
				case (KEY_K): { // Delete from cursor_pos to end.

					if (editable) {
						deselect();
						text = text.substr(0, cursor_pos);
						_text_changed();
					}

				} break;
				case (KEY_A): { // Select all.
					select();

				} break;
#ifdef APPLE_STYLE_KEYS
				case (KEY_LEFT): { // Go to start of text - like HOME key.
					shift_selection_check_pre(k->get_shift());
					set_cursor_position(0);
					shift_selection_check_post(k->get_shift());
				} break;
				case (KEY_RIGHT): { // Go to end of text - like END key.
					shift_selection_check_pre(k->get_shift());
					set_cursor_position(text.length());
					shift_selection_check_post(k->get_shift());
				} break;
				case (KEY_BACKSPACE): {
					if (!editable)
						break;

					// If selected, delete the selection
					if (selection.enabled) {
						selection_delete();
						break;
					}

					// Otherwise delete from cursor to beginning of text edit
					int current_pos = get_cursor_position();
					if (current_pos != 0) {
						delete_text(0, current_pos);
					}
				} break;
#endif
				default: {
					handled = false;
				}
			}

			if (handled) {
				accept_event();
				return;
			}
		}

		_reset_caret_blink_timer();
		if (!k->get_metakey()) {
			bool handled = true;
			switch (code) {
				case KEY_KP_ENTER:
				case KEY_ENTER: {
					emit_signal("text_entered", text);
					if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_VIRTUAL_KEYBOARD) && virtual_keyboard_enabled) {
						DisplayServer::get_singleton()->virtual_keyboard_hide();
					}

				} break;

				case KEY_BACKSPACE: {
					if (!editable) {
						break;
					}

					if (selection.enabled) {
						selection_delete();
						break;
					}

#ifdef APPLE_STYLE_KEYS
					if (k->get_alt()) {
#else
					if (k->get_alt()) {
						handled = false;
						break;
					} else if (k->get_command()) {
#endif
						int cc = cursor_pos;

						Vector<Vector2i> words = TS->shaped_text_get_word_breaks(text_rid);
						for (int i = words.size() - 1; i >= 0; i--) {
							if (words[i].x < cc) {
								cc = words[i].x;
								break;
							}
						}

						delete_text(cc, cursor_pos);

						set_cursor_position(cc);

					} else {
						delete_char();
					}

				} break;
				case KEY_KP_4: {
					if (k->get_unicode() != 0) {
						handled = false;
						break;
					}
					[[fallthrough]];
				}
				case KEY_LEFT: {
#ifndef APPLE_STYLE_KEYS
					if (!k->get_alt()) {
#endif
						if (selection.enabled && !k->get_shift()) {
							set_cursor_position(selection.begin);
							deselect();
							handled = true;
							break;
						}

						shift_selection_check_pre(k->get_shift());
#ifndef APPLE_STYLE_KEYS
					}
#endif

#ifdef APPLE_STYLE_KEYS
					if (k->get_command()) {
						set_cursor_position(0);
					} else if (k->get_alt()) {
#else
					if (k->get_alt()) {
						handled = false;
						break;
					} else if (k->get_command()) {
#endif
						int cc = cursor_pos;

						Vector<Vector2i> words = TS->shaped_text_get_word_breaks(text_rid);
						for (int i = words.size() - 1; i >= 0; i--) {
							if (words[i].x < cc) {
								cc = words[i].x;
								break;
							}
						}

						set_cursor_position(cc);

					} else {
						if (mid_grapheme_caret_enabled) {
							set_cursor_position(get_cursor_position() - 1);
						} else {
							set_cursor_position(TS->shaped_text_prev_grapheme_pos(text_rid, get_cursor_position()));
						}
					}

					shift_selection_check_post(k->get_shift());

				} break;
				case KEY_KP_6: {
					if (k->get_unicode() != 0) {
						handled = false;
						break;
					}
					[[fallthrough]];
				}
				case KEY_RIGHT: {
#ifndef APPLE_STYLE_KEYS
					if (!k->get_alt()) {
#endif
						if (selection.enabled && !k->get_shift()) {
							set_cursor_position(selection.end);
							deselect();
							handled = true;
							break;
						}

						shift_selection_check_pre(k->get_shift());
#ifndef APPLE_STYLE_KEYS
					}
#endif

#ifdef APPLE_STYLE_KEYS
					if (k->get_command()) {
						set_cursor_position(text.length());
					} else if (k->get_alt()) {
#else
					if (k->get_alt()) {
						handled = false;
						break;
					} else if (k->get_command()) {
#endif
						int cc = cursor_pos;

						Vector<Vector2i> words = TS->shaped_text_get_word_breaks(text_rid);
						for (int i = 0; i < words.size(); i++) {
							if (words[i].y > cc) {
								cc = words[i].y;
								break;
							}
						}

						set_cursor_position(cc);

					} else {
						if (mid_grapheme_caret_enabled) {
							set_cursor_position(get_cursor_position() + 1);
						} else {
							set_cursor_position(TS->shaped_text_next_grapheme_pos(text_rid, get_cursor_position()));
						}
					}

					shift_selection_check_post(k->get_shift());

				} break;
				case KEY_UP: {
					shift_selection_check_pre(k->get_shift());
					if (get_cursor_position() == 0) {
						handled = false;
					}
					set_cursor_position(0);
					shift_selection_check_post(k->get_shift());
				} break;
				case KEY_DOWN: {
					shift_selection_check_pre(k->get_shift());
					if (get_cursor_position() == text.length()) {
						handled = false;
					}
					set_cursor_position(text.length());
					shift_selection_check_post(k->get_shift());
				} break;
				case KEY_DELETE: {
					if (!editable) {
						break;
					}

					if (k->get_shift() && !k->get_command() && !k->get_alt()) {
						cut_text();
						break;
					}

					if (selection.enabled) {
						selection_delete();
						break;
					}

					int text_len = text.length();

					if (cursor_pos == text_len) {
						break; // Nothing to do.
					}

#ifdef APPLE_STYLE_KEYS
					if (k->get_alt()) {
#else
					if (k->get_alt()) {
						handled = false;
						break;
					} else if (k->get_command()) {
#endif
						int cc = cursor_pos;

						Vector<Vector2i> words = TS->shaped_text_get_word_breaks(text_rid);
						for (int i = 0; i < words.size(); i++) {
							if (words[i].y > cc) {
								cc = words[i].y;
								break;
							}
						}

						delete_text(cursor_pos, cc);

					} else {
						if (mid_grapheme_caret_enabled) {
							set_cursor_position(cursor_pos + 1);
							delete_char();
						} else {
							int cc = cursor_pos;
							set_cursor_position(TS->shaped_text_next_grapheme_pos(text_rid, cursor_pos));
							delete_text(cc, cursor_pos);
						}
					}

				} break;
				case KEY_KP_7: {
					if (k->get_unicode() != 0) {
						handled = false;
						break;
					}
					[[fallthrough]];
				}
				case KEY_HOME: {
					shift_selection_check_pre(k->get_shift());
					set_cursor_position(0);
					shift_selection_check_post(k->get_shift());
				} break;
				case KEY_KP_1: {
					if (k->get_unicode() != 0) {
						handled = false;
						break;
					}
					[[fallthrough]];
				}
				case KEY_END: {
					shift_selection_check_pre(k->get_shift());
					set_cursor_position(text.length());
					shift_selection_check_post(k->get_shift());
				} break;
				case KEY_MENU: {
					if (context_menu_enabled) {
						Point2 pos = Point2(get_cursor_pixel_pos().x, (get_size().y + get_theme_font("font")->get_height(get_theme_font_size("font_size"))) / 2);
						menu->set_position(get_global_transform().xform(pos));
						menu->set_size(Vector2(1, 1));
						//menu->set_scale(get_global_transform().get_scale());
						menu->popup();
						menu->grab_focus();
					}
				} break;

				default: {
					handled = false;
				} break;
			}

			if (handled) {
				accept_event();
			} else if (!k->get_command() || (k->get_command() && k->get_alt())) {
				if (k->get_unicode() >= 32 && k->get_keycode() != KEY_DELETE) {
					if (editable) {
						selection_delete();
						char32_t ucodestr[2] = { (char32_t)k->get_unicode(), 0 };
						int prev_len = text.length();
						append_at_cursor(ucodestr);
						if (text.length() != prev_len) {
							_text_changed();
						}
						accept_event();
					}

				} else {
					return;
				}
			}

			update();
		}

		return;
	}
}

void LineEdit::set_align(Align p_align) {
	ERR_FAIL_INDEX((int)p_align, 4);
	if (align != p_align) {
		align = p_align;
		_shape();
	}
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

		text.erase(selection.begin, selected);
		_shape();

		append_at_cursor(p_data);
		selection.begin = cursor_pos - selected;
		selection.end = cursor_pos;
	}
}

Control::CursorShape LineEdit::get_cursor_shape(const Point2 &p_pos) const {
	if (!text.empty() && is_editable() && _is_over_clear_button(p_pos)) {
		return CURSOR_ARROW;
	}
	return Control::get_cursor_shape(p_pos);
}

bool LineEdit::_is_over_clear_button(const Point2 &p_pos) const {
	if (!clear_button_enabled || !has_point(p_pos)) {
		return false;
	}
	Ref<Texture2D> icon = Control::get_theme_icon("clear");
	int x_ofs = get_theme_stylebox("normal")->get_offset().x;
	return p_pos.x > get_size().width - icon->get_width() - x_ofs;
}

void LineEdit::_notification(int p_what) {
	switch (p_what) {
#ifdef TOOLS_ENABLED
		case NOTIFICATION_ENTER_TREE: {
			if (Engine::get_singleton()->is_editor_hint() && !get_tree()->is_node_being_edited(this)) {
				cursor_set_blink_enabled(EDITOR_DEF("text_editor/cursor/caret_blink", false));
				cursor_set_blink_speed(EDITOR_DEF("text_editor/cursor/caret_blink_speed", 0.65));

				if (!EditorSettings::get_singleton()->is_connected("settings_changed", callable_mp(this, &LineEdit::_editor_settings_changed))) {
					EditorSettings::get_singleton()->connect("settings_changed", callable_mp(this, &LineEdit::_editor_settings_changed));
				}
			}
		} break;
#endif
		case NOTIFICATION_RESIZED: {
			_fit_to_width();
			scroll_offset = 0;
			set_cursor_position(get_cursor_position());
		} break;
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case NOTIFICATION_THEME_CHANGED: {
			_shape();
			update();
		} break;
		case NOTIFICATION_TRANSLATION_CHANGED: {
			placeholder_translated = tr(placeholder);
			_shape();
			update();
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
		case NOTIFICATION_DRAW: {
			if ((!has_focus() && !menu->has_focus() && !caret_force_displayed) || !window_has_focus) {
				draw_caret = false;
			}

			int width, height;
			bool rtl = is_layout_rtl();

			Size2 size = get_size();
			width = size.width;
			height = size.height;

			RID ci = get_canvas_item();

			Ref<StyleBox> style = get_theme_stylebox("normal");
			if (!is_editable()) {
				style = get_theme_stylebox("read_only");
				draw_caret = false;
			}

			style->draw(ci, Rect2(Point2(), size));

			if (has_focus()) {
				get_theme_stylebox("focus")->draw(ci, Rect2(Point2(), size));
			}

			int x_ofs = 0;
			bool using_placeholder = text.empty() && ime_text.empty();
			float text_width = TS->shaped_text_get_size(text_rid).x;
			float text_height = TS->shaped_text_get_size(text_rid).y;

			switch (align) {
				case ALIGN_FILL:
				case ALIGN_LEFT: {
					if (rtl) {
						x_ofs = MAX(style->get_margin(MARGIN_LEFT), int(size.width - style->get_margin(MARGIN_RIGHT) - (text_width)));
					} else {
						x_ofs = style->get_offset().x;
					}
				} break;
				case ALIGN_CENTER: {
					if (scroll_offset != 0) {
						x_ofs = style->get_offset().x;
					} else {
						x_ofs = MAX(style->get_margin(MARGIN_LEFT), int(size.width - (text_width)) / 2);
					}
				} break;
				case ALIGN_RIGHT: {
					if (rtl) {
						x_ofs = style->get_offset().x;
					} else {
						x_ofs = MAX(style->get_margin(MARGIN_LEFT), int(size.width - style->get_margin(MARGIN_RIGHT) - (text_width)));
					}
				} break;
			}

			int ofs_max = width - style->get_margin(MARGIN_RIGHT);

			int y_area = height - style->get_minimum_size().height;
			int y_ofs = style->get_offset().y + (y_area - text_height) / 2;

			Color selection_color = get_theme_color("selection_color");
			Color font_color = is_editable() ? get_theme_color("font_color") : get_theme_color("font_color_uneditable");
			Color font_color_selected = get_theme_color("font_color_selected");
			Color cursor_color = get_theme_color("cursor_color");

			// Draw placeholder color.
			if (using_placeholder) {
				font_color.a *= placeholder_alpha;
			}

			bool display_clear_icon = !using_placeholder && is_editable() && clear_button_enabled;
			if (right_icon.is_valid() || display_clear_icon) {
				Ref<Texture2D> r_icon = display_clear_icon ? Control::get_theme_icon("clear") : right_icon;
				Color color_icon(1, 1, 1, !is_editable() ? .5 * .9 : .9);
				if (display_clear_icon) {
					if (clear_button_status.press_attempt && clear_button_status.pressing_inside) {
						color_icon = get_theme_color("clear_button_color_pressed");
					} else {
						color_icon = get_theme_color("clear_button_color");
					}
				}

				r_icon->draw(ci, Point2(width - r_icon->get_width() - style->get_margin(MARGIN_RIGHT), height / 2 - r_icon->get_height() / 2), color_icon);

				if (align == ALIGN_CENTER) {
					if (scroll_offset == 0) {
						x_ofs = MAX(style->get_margin(MARGIN_LEFT), int(size.width - text_width - r_icon->get_width() - style->get_margin(MARGIN_RIGHT) * 2) / 2);
					}
				} else {
					x_ofs = MAX(style->get_margin(MARGIN_LEFT), x_ofs - r_icon->get_width() - style->get_margin(MARGIN_RIGHT));
				}

				ofs_max -= r_icon->get_width();
			}

#ifdef TOOLS_ENABLED
			int caret_width = Math::round(EDSCALE);
#else
			int caret_width = 1;
#endif

			// Draw selections rects.
			Vector2 ofs = Point2(x_ofs + scroll_offset, y_ofs);
			if (selection.enabled) {
				Vector<Vector2> sel = TS->shaped_text_get_selection(text_rid, selection.begin, selection.end);
				for (int i = 0; i < sel.size(); i++) {
					Rect2 rect = Rect2(sel[i].x + ofs.x, ofs.y, sel[i].y - sel[i].x, text_height);
					if (rect.position.x + rect.size.x <= x_ofs || rect.position.x > ofs_max) {
						continue;
					}
					if (rect.position.x < x_ofs) {
						rect.size.x -= (x_ofs - rect.position.x);
						rect.position.x = x_ofs;
					} else if (rect.position.x + rect.size.x > ofs_max) {
						rect.size.x = ofs_max - rect.position.x;
					}
					RenderingServer::get_singleton()->canvas_item_add_rect(ci, rect, selection_color);
				}
			}
			const Vector<TextServer::Glyph> glyphs = TS->shaped_text_get_glyphs(text_rid);

			// Draw text.
			ofs.y += TS->shaped_text_get_ascent(text_rid);
			for (int i = 0; i < glyphs.size(); i++) {
				bool selected = selection.enabled && glyphs[i].start >= selection.begin && glyphs[i].end <= selection.end;
				for (int j = 0; j < glyphs[i].repeat; j++) {
					if (ceil(ofs.x) >= x_ofs && floor(ofs.x + glyphs[i].advance) <= ofs_max) {
						if (glyphs[i].font_rid != RID()) {
							TS->font_draw_glyph(glyphs[i].font_rid, ci, glyphs[i].font_size, ofs + Vector2(glyphs[i].x_off, glyphs[i].y_off), glyphs[i].index, selected ? font_color_selected : font_color);
						} else if ((glyphs[i].flags & TextServer::GRAPHEME_IS_VIRTUAL) != TextServer::GRAPHEME_IS_VIRTUAL) {
							TS->draw_hex_code_box(ci, glyphs[i].font_size, ofs + Vector2(glyphs[i].x_off, glyphs[i].y_off), glyphs[i].index, selected ? font_color_selected : font_color);
						}
					}
					ofs.x += glyphs[i].advance;
				}
				if (ofs.x >= ofs_max) {
					break;
				}
			}

			// Draw carets.
			ofs.x = x_ofs + scroll_offset;
			if (draw_caret) {
				if (ime_text.length() == 0) {
					// Normal caret.
					Rect2 l_caret, t_caret;
					TextServer::Direction l_dir, t_dir;
					TS->shaped_text_get_carets(text_rid, cursor_pos, l_caret, l_dir, t_caret, t_dir);

					if (l_caret == Rect2() && t_caret == Rect2()) {
						// No carets, add one at the start.
						int h = get_theme_font("font")->get_height(get_theme_font_size("font_size"));
						int y = style->get_offset().y + (y_area - h) / 2;
						if (rtl) {
							l_dir = TextServer::DIRECTION_RTL;
							l_caret = Rect2(Vector2(ofs_max, y), Size2(caret_width, h));
						} else {
							l_dir = TextServer::DIRECTION_LTR;
							l_caret = Rect2(Vector2(x_ofs, y), Size2(caret_width, h));
						}
						RenderingServer::get_singleton()->canvas_item_add_rect(ci, l_caret, cursor_color);
					} else {
						if (l_caret != Rect2() && l_dir == TextServer::DIRECTION_AUTO) {
							// Draw extra marker on top of mid caret.
							Rect2 trect = Rect2(l_caret.position.x - 3 * caret_width, l_caret.position.y, 6 * caret_width, caret_width);
							trect.position += ofs;
							RenderingServer::get_singleton()->canvas_item_add_rect(ci, trect, cursor_color);
						}

						l_caret.position += ofs;
						l_caret.size.x = caret_width;
						RenderingServer::get_singleton()->canvas_item_add_rect(ci, l_caret, cursor_color);

						t_caret.position += ofs;
						t_caret.size.x = caret_width;

						RenderingServer::get_singleton()->canvas_item_add_rect(ci, t_caret, cursor_color);
					}
				} else {
					{
						// IME intermidiet text range.
						Vector<Vector2> sel = TS->shaped_text_get_selection(text_rid, cursor_pos, cursor_pos + ime_text.length());
						for (int i = 0; i < sel.size(); i++) {
							Rect2 rect = Rect2(sel[i].x + ofs.x, ofs.y, sel[i].y - sel[i].x, text_height);
							if (rect.position.x + rect.size.x <= x_ofs || rect.position.x > ofs_max) {
								continue;
							}
							if (rect.position.x < x_ofs) {
								rect.size.x -= (x_ofs - rect.position.x);
								rect.position.x = x_ofs;
							} else if (rect.position.x + rect.size.x > ofs_max) {
								rect.size.x = ofs_max - rect.position.x;
							}
							rect.size.y = caret_width;
							RenderingServer::get_singleton()->canvas_item_add_rect(ci, rect, cursor_color);
						}
					}
					{
						// IME caret.
						Vector<Vector2> sel = TS->shaped_text_get_selection(text_rid, cursor_pos + ime_selection.x, cursor_pos + ime_selection.x + ime_selection.y);
						for (int i = 0; i < sel.size(); i++) {
							Rect2 rect = Rect2(sel[i].x + ofs.x, ofs.y, sel[i].y - sel[i].x, text_height);
							if (rect.position.x + rect.size.x <= x_ofs || rect.position.x > ofs_max) {
								continue;
							}
							if (rect.position.x < x_ofs) {
								rect.size.x -= (x_ofs - rect.position.x);
								rect.position.x = x_ofs;
							} else if (rect.position.x + rect.size.x > ofs_max) {
								rect.size.x = ofs_max - rect.position.x;
							}
							rect.size.y = caret_width * 3;
							RenderingServer::get_singleton()->canvas_item_add_rect(ci, rect, cursor_color);
						}
					}
				}
			}

			if (has_focus()) {
				if (get_viewport()->get_window_id() != DisplayServer::INVALID_WINDOW_ID) {
					DisplayServer::get_singleton()->window_set_ime_active(true, get_viewport()->get_window_id());
					DisplayServer::get_singleton()->window_set_ime_position(get_global_position() + Point2(using_placeholder ? 0 : x_ofs, y_ofs + TS->shaped_text_get_size(text_rid).y), get_viewport()->get_window_id());
				}
			}
		} break;
		case NOTIFICATION_FOCUS_ENTER: {
			if (!caret_force_displayed) {
				if (caret_blink_enabled) {
					if (caret_blink_timer->is_stopped()) {
						caret_blink_timer->start();
					}
				} else {
					draw_caret = true;
				}
			}

			if (get_viewport()->get_window_id() != DisplayServer::INVALID_WINDOW_ID) {
				DisplayServer::get_singleton()->window_set_ime_active(true, get_viewport()->get_window_id());
				Point2 cursor_pos = Point2(get_cursor_position(), 1) * get_minimum_size().height;
				DisplayServer::get_singleton()->window_set_ime_position(get_global_position() + cursor_pos, get_viewport()->get_window_id());
			}

			if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_VIRTUAL_KEYBOARD) && virtual_keyboard_enabled) {
				if (selection.enabled) {
					DisplayServer::get_singleton()->virtual_keyboard_show(text, get_global_rect(), false, max_length, selection.begin, selection.end);
				} else {
					DisplayServer::get_singleton()->virtual_keyboard_show(text, get_global_rect(), false, max_length, cursor_pos);
				}
			}

		} break;
		case NOTIFICATION_FOCUS_EXIT: {
			if (caret_blink_enabled && !caret_force_displayed) {
				caret_blink_timer->stop();
			}

			if (get_viewport()->get_window_id() != DisplayServer::INVALID_WINDOW_ID) {
				DisplayServer::get_singleton()->window_set_ime_position(Point2(), get_viewport()->get_window_id());
				DisplayServer::get_singleton()->window_set_ime_active(false, get_viewport()->get_window_id());
			}
			ime_text = "";
			ime_selection = Point2();
			_shape();
			set_cursor_position(cursor_pos); // Update scroll_offset

			if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_VIRTUAL_KEYBOARD) && virtual_keyboard_enabled) {
				DisplayServer::get_singleton()->virtual_keyboard_hide();
			}

		} break;
		case MainLoop::NOTIFICATION_OS_IME_UPDATE: {
			if (has_focus()) {
				ime_text = DisplayServer::get_singleton()->ime_get_text();
				ime_selection = DisplayServer::get_singleton()->ime_get_selection();
				_shape();
				set_cursor_position(cursor_pos); // Update scroll_offset

				update();
			}
		} break;
	}
}

void LineEdit::copy_text() {
	if (selection.enabled && !pass) {
		DisplayServer::get_singleton()->clipboard_set(text.substr(selection.begin, selection.end - selection.begin));
	}
}

void LineEdit::cut_text() {
	if (selection.enabled && !pass) {
		DisplayServer::get_singleton()->clipboard_set(text.substr(selection.begin, selection.end - selection.begin));
		selection_delete();
	}
}

void LineEdit::paste_text() {
	// Strip escape characters like \n and \t as they can't be displayed on LineEdit.
	String paste_buffer = DisplayServer::get_singleton()->clipboard_get().strip_escapes();

	if (paste_buffer != "") {
		int prev_len = text.length();
		if (selection.enabled) {
			selection_delete();
		}
		append_at_cursor(paste_buffer);

		if (!text_changed_dirty) {
			if (is_inside_tree() && text.length() != prev_len) {
				MessageQueue::get_singleton()->push_call(this, "_text_changed");
			}
			text_changed_dirty = true;
		}
	}
}

void LineEdit::undo() {
	if (undo_stack_pos == nullptr) {
		if (undo_stack.size() <= 1) {
			return;
		}
		undo_stack_pos = undo_stack.back();
	} else if (undo_stack_pos == undo_stack.front()) {
		return;
	}
	undo_stack_pos = undo_stack_pos->prev();
	TextOperation op = undo_stack_pos->get();
	text = op.text;
	scroll_offset = op.scroll_offset;
	set_cursor_position(op.cursor_pos);

	_shape();
	_emit_text_change();
}

void LineEdit::redo() {
	if (undo_stack_pos == nullptr) {
		return;
	}
	if (undo_stack_pos == undo_stack.back()) {
		return;
	}
	undo_stack_pos = undo_stack_pos->next();
	TextOperation op = undo_stack_pos->get();
	text = op.text;
	scroll_offset = op.scroll_offset;
	set_cursor_position(op.cursor_pos);

	_shape();
	_emit_text_change();
}

void LineEdit::shift_selection_check_pre(bool p_shift) {
	if (!selection.enabled && p_shift) {
		selection.cursor_start = cursor_pos;
	}
	if (!p_shift) {
		deselect();
	}
}

void LineEdit::shift_selection_check_post(bool p_shift) {
	if (p_shift) {
		selection_fill_at_cursor();
	}
}

void LineEdit::set_cursor_at_pixel_pos(int p_x) {
	Ref<StyleBox> style = get_theme_stylebox("normal");
	bool rtl = is_layout_rtl();

	int x_ofs = 0;
	float text_width = TS->shaped_text_get_size(text_rid).x;
	switch (align) {
		case ALIGN_FILL:
		case ALIGN_LEFT: {
			if (rtl) {
				x_ofs = MAX(style->get_margin(MARGIN_LEFT), int(get_size().width - style->get_margin(MARGIN_RIGHT) - (text_width)));
			} else {
				x_ofs = style->get_offset().x;
			}
		} break;
		case ALIGN_CENTER: {
			if (scroll_offset != 0) {
				x_ofs = style->get_offset().x;
			} else {
				x_ofs = MAX(style->get_margin(MARGIN_LEFT), int(get_size().width - (text_width)) / 2);
			}
		} break;
		case ALIGN_RIGHT: {
			if (rtl) {
				x_ofs = style->get_offset().x;
			} else {
				x_ofs = MAX(style->get_margin(MARGIN_LEFT), int(get_size().width - style->get_margin(MARGIN_RIGHT) - (text_width)));
			}
		} break;
	}

	bool using_placeholder = text.empty() && ime_text.empty();
	bool display_clear_icon = !using_placeholder && is_editable() && clear_button_enabled;
	if (right_icon.is_valid() || display_clear_icon) {
		Ref<Texture2D> r_icon = display_clear_icon ? Control::get_theme_icon("clear") : right_icon;
		if (align == ALIGN_CENTER) {
			if (scroll_offset == 0) {
				x_ofs = MAX(style->get_margin(MARGIN_LEFT), int(get_size().width - text_width - r_icon->get_width() - style->get_margin(MARGIN_RIGHT) * 2) / 2);
			}
		} else {
			x_ofs = MAX(style->get_margin(MARGIN_LEFT), x_ofs - r_icon->get_width() - style->get_margin(MARGIN_RIGHT));
		}
	}

	int ofs = TS->shaped_text_hit_test_position(text_rid, p_x - x_ofs - scroll_offset);
	set_cursor_position(ofs);
}

Vector2i LineEdit::get_cursor_pixel_pos() {
	Ref<StyleBox> style = get_theme_stylebox("normal");
	bool rtl = is_layout_rtl();

	int x_ofs = 0;
	float text_width = TS->shaped_text_get_size(text_rid).x;
	switch (align) {
		case ALIGN_FILL:
		case ALIGN_LEFT: {
			if (rtl) {
				x_ofs = MAX(style->get_margin(MARGIN_LEFT), int(get_size().width - style->get_margin(MARGIN_RIGHT) - (text_width)));
			} else {
				x_ofs = style->get_offset().x;
			}
		} break;
		case ALIGN_CENTER: {
			if (scroll_offset != 0) {
				x_ofs = style->get_offset().x;
			} else {
				x_ofs = MAX(style->get_margin(MARGIN_LEFT), int(get_size().width - (text_width)) / 2);
			}
		} break;
		case ALIGN_RIGHT: {
			if (rtl) {
				x_ofs = style->get_offset().x;
			} else {
				x_ofs = MAX(style->get_margin(MARGIN_LEFT), int(get_size().width - style->get_margin(MARGIN_RIGHT) - (text_width)));
			}
		} break;
	}

	bool using_placeholder = text.empty() && ime_text.empty();
	bool display_clear_icon = !using_placeholder && is_editable() && clear_button_enabled;
	if (right_icon.is_valid() || display_clear_icon) {
		Ref<Texture2D> r_icon = display_clear_icon ? Control::get_theme_icon("clear") : right_icon;
		if (align == ALIGN_CENTER) {
			if (scroll_offset == 0) {
				x_ofs = MAX(style->get_margin(MARGIN_LEFT), int(get_size().width - text_width - r_icon->get_width() - style->get_margin(MARGIN_RIGHT) * 2) / 2);
			}
		} else {
			x_ofs = MAX(style->get_margin(MARGIN_LEFT), x_ofs - r_icon->get_width() - style->get_margin(MARGIN_RIGHT));
		}
	}

	Vector2i ret;
	Rect2 l_caret, t_caret;
	TextServer::Direction l_dir, t_dir;
	// Get position of the start of caret.
	if (ime_text.length() != 0 && ime_selection.x != 0) {
		TS->shaped_text_get_carets(text_rid, cursor_pos + ime_selection.x, l_caret, l_dir, t_caret, t_dir);
	} else {
		TS->shaped_text_get_carets(text_rid, cursor_pos, l_caret, l_dir, t_caret, t_dir);
	}

	if ((l_caret != Rect2() && (l_dir == TextServer::DIRECTION_AUTO || l_dir == (TextServer::Direction)input_direction)) || (t_caret == Rect2())) {
		ret.x = x_ofs + l_caret.position.x + scroll_offset;
	} else {
		ret.x = x_ofs + t_caret.position.x + scroll_offset;
	}

	// Get position of the end of caret.
	if (ime_text.length() != 0) {
		if (ime_selection.y != 0) {
			TS->shaped_text_get_carets(text_rid, cursor_pos + ime_selection.x + ime_selection.y, l_caret, l_dir, t_caret, t_dir);
		} else {
			TS->shaped_text_get_carets(text_rid, cursor_pos + ime_text.size(), l_caret, l_dir, t_caret, t_dir);
		}
		if ((l_caret != Rect2() && (l_dir == TextServer::DIRECTION_AUTO || l_dir == (TextServer::Direction)input_direction)) || (t_caret == Rect2())) {
			ret.y = x_ofs + l_caret.position.x + scroll_offset;
		} else {
			ret.y = x_ofs + t_caret.position.x + scroll_offset;
		}
	} else {
		ret.y = ret.x;
	}

	return ret;
}

void LineEdit::set_mid_grapheme_caret_enabled(const bool p_enabled) {
	mid_grapheme_caret_enabled = p_enabled;
}

bool LineEdit::get_mid_grapheme_caret_enabled() const {
	return mid_grapheme_caret_enabled;
}

bool LineEdit::cursor_get_blink_enabled() const {
	return caret_blink_enabled;
}

void LineEdit::cursor_set_blink_enabled(const bool p_enabled) {
	caret_blink_enabled = p_enabled;

	if (has_focus() || caret_force_displayed) {
		if (p_enabled) {
			if (caret_blink_timer->is_stopped()) {
				caret_blink_timer->start();
			}
		} else {
			caret_blink_timer->stop();
		}
	}

	draw_caret = true;
}

bool LineEdit::cursor_get_force_displayed() const {
	return caret_force_displayed;
}

void LineEdit::cursor_set_force_displayed(const bool p_enabled) {
	caret_force_displayed = p_enabled;
	cursor_set_blink_enabled(caret_blink_enabled);
	update();
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
		draw_caret = true;
		if (has_focus()) {
			caret_blink_timer->stop();
			caret_blink_timer->start();
			update();
		}
	}
}

void LineEdit::_toggle_draw_caret() {
	draw_caret = !draw_caret;
	if (is_visible_in_tree() && ((has_focus() && window_has_focus) || caret_force_displayed)) {
		update();
	}
}

void LineEdit::delete_char() {
	if ((text.length() <= 0) || (cursor_pos == 0)) {
		return;
	}

	text.erase(cursor_pos - 1, 1);
	_shape();

	set_cursor_position(get_cursor_position() - 1);

	_text_changed();
}

void LineEdit::delete_text(int p_from_column, int p_to_column) {
	ERR_FAIL_COND_MSG(p_from_column < 0 || p_from_column > p_to_column || p_to_column > text.length(),
			vformat("Positional parameters (from: %d, to: %d) are inverted or outside the text length (%d).", p_from_column, p_to_column, text.length()));

	text.erase(p_from_column, p_to_column - p_from_column);
	_shape();

	cursor_pos -= CLAMP(cursor_pos - p_from_column, 0, p_to_column - p_from_column);

	if (cursor_pos >= text.length()) {
		cursor_pos = text.length();
	}

	if (!text_changed_dirty) {
		if (is_inside_tree()) {
			MessageQueue::get_singleton()->push_call(this, "_text_changed");
		}
		text_changed_dirty = true;
	}
}

void LineEdit::set_text(String p_text) {
	clear_internal();
	append_at_cursor(p_text);

	update();
	cursor_pos = 0;
	scroll_offset = 0;
}

void LineEdit::set_text_direction(Control::TextDirection p_text_direction) {
	ERR_FAIL_COND((int)p_text_direction < -1 || (int)p_text_direction > 3);
	if (text_direction != p_text_direction) {
		text_direction = p_text_direction;
		if (text_direction != TEXT_DIRECTION_AUTO && text_direction != TEXT_DIRECTION_INHERITED) {
			input_direction = text_direction;
		}
		_shape();

		menu_dir->set_item_checked(menu_dir->get_item_index(MENU_DIR_INHERITED), text_direction == TEXT_DIRECTION_INHERITED);
		menu_dir->set_item_checked(menu_dir->get_item_index(MENU_DIR_AUTO), text_direction == TEXT_DIRECTION_AUTO);
		menu_dir->set_item_checked(menu_dir->get_item_index(MENU_DIR_LTR), text_direction == TEXT_DIRECTION_LTR);
		menu_dir->set_item_checked(menu_dir->get_item_index(MENU_DIR_RTL), text_direction == TEXT_DIRECTION_RTL);
		update();
	}
}

Control::TextDirection LineEdit::get_text_direction() const {
	return text_direction;
}

void LineEdit::clear_opentype_features() {
	opentype_features.clear();
	_shape();
	update();
}

void LineEdit::set_opentype_feature(const String &p_name, int p_value) {
	int32_t tag = TS->name_to_tag(p_name);
	if (!opentype_features.has(tag) || (int)opentype_features[tag] != p_value) {
		opentype_features[tag] = p_value;
		_shape();
		update();
	}
}

int LineEdit::get_opentype_feature(const String &p_name) const {
	int32_t tag = TS->name_to_tag(p_name);
	if (!opentype_features.has(tag)) {
		return -1;
	}
	return opentype_features[tag];
}

void LineEdit::set_language(const String &p_language) {
	if (language != p_language) {
		language = p_language;
		_shape();
		update();
	}
}

String LineEdit::get_language() const {
	return language;
}

void LineEdit::set_draw_control_chars(bool p_draw_control_chars) {
	if (draw_control_chars != p_draw_control_chars) {
		draw_control_chars = p_draw_control_chars;
		menu->set_item_checked(menu->get_item_index(MENU_DISPLAY_UCC), draw_control_chars);
		_shape();
		update();
	}
}

bool LineEdit::get_draw_control_chars() const {
	return draw_control_chars;
}

void LineEdit::set_structured_text_bidi_override(Control::StructuredTextParser p_parser) {
	if (st_parser != p_parser) {
		st_parser = p_parser;
		_shape();
		update();
	}
}

Control::StructuredTextParser LineEdit::get_structured_text_bidi_override() const {
	return st_parser;
}

void LineEdit::set_structured_text_bidi_override_options(Array p_args) {
	st_args = p_args;
	_shape();
	update();
}

Array LineEdit::get_structured_text_bidi_override_options() const {
	return st_args;
}

void LineEdit::clear() {
	clear_internal();
	_text_changed();
}

String LineEdit::get_text() const {
	return text;
}

void LineEdit::set_placeholder(String p_text) {
	placeholder = p_text;
	placeholder_translated = tr(placeholder);
	_shape();
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

void LineEdit::set_cursor_position(int p_pos) {
	if (p_pos > (int)text.length()) {
		p_pos = text.length();
	}

	if (p_pos < 0) {
		p_pos = 0;
	}

	cursor_pos = p_pos;

	// Fit to window.

	if (!is_inside_tree()) {
		scroll_offset = 0;
		return;
	}

	Ref<StyleBox> style = get_theme_stylebox("normal");
	bool rtl = is_layout_rtl();

	int x_ofs = 0;
	float text_width = TS->shaped_text_get_size(text_rid).x;
	switch (align) {
		case ALIGN_FILL:
		case ALIGN_LEFT: {
			if (rtl) {
				x_ofs = MAX(style->get_margin(MARGIN_LEFT), int(get_size().width - style->get_margin(MARGIN_RIGHT) - (text_width)));
			} else {
				x_ofs = style->get_offset().x;
			}
		} break;
		case ALIGN_CENTER: {
			if (scroll_offset != 0) {
				x_ofs = style->get_offset().x;
			} else {
				x_ofs = MAX(style->get_margin(MARGIN_LEFT), int(get_size().width - (text_width)) / 2);
			}
		} break;
		case ALIGN_RIGHT: {
			if (rtl) {
				x_ofs = style->get_offset().x;
			} else {
				x_ofs = MAX(style->get_margin(MARGIN_LEFT), int(get_size().width - style->get_margin(MARGIN_RIGHT) - (text_width)));
			}
		} break;
	}

	int ofs_max = get_size().width - style->get_margin(MARGIN_RIGHT);
	bool using_placeholder = text.empty() && ime_text.empty();
	bool display_clear_icon = !using_placeholder && is_editable() && clear_button_enabled;
	if (right_icon.is_valid() || display_clear_icon) {
		Ref<Texture2D> r_icon = display_clear_icon ? Control::get_theme_icon("clear") : right_icon;
		if (align == ALIGN_CENTER) {
			if (scroll_offset == 0) {
				x_ofs = MAX(style->get_margin(MARGIN_LEFT), int(get_size().width - text_width - r_icon->get_width() - style->get_margin(MARGIN_RIGHT) * 2) / 2);
			}
		} else {
			x_ofs = MAX(style->get_margin(MARGIN_LEFT), x_ofs - r_icon->get_width() - style->get_margin(MARGIN_RIGHT));
		}
		ofs_max -= r_icon->get_width();
	}

	// Note: Use too coordinates to fit IME input range.
	Vector2i primary_catret_offset = get_cursor_pixel_pos();

	if (MIN(primary_catret_offset.x, primary_catret_offset.y) <= x_ofs) {
		scroll_offset += (x_ofs - MIN(primary_catret_offset.x, primary_catret_offset.y));
	} else if (MAX(primary_catret_offset.x, primary_catret_offset.y) >= ofs_max) {
		scroll_offset += (ofs_max - MAX(primary_catret_offset.x, primary_catret_offset.y));
	}
	scroll_offset = MIN(0, scroll_offset);

	update();
}

int LineEdit::get_cursor_position() const {
	return cursor_pos;
}

void LineEdit::set_scroll_offset(int p_pos) {
	scroll_offset = p_pos;
	if (scroll_offset < 0) {
		scroll_offset = 0;
	}
}

int LineEdit::get_scroll_offset() const {
	return scroll_offset;
}

void LineEdit::append_at_cursor(String p_text) {
	if ((max_length <= 0) || (text.length() + p_text.length() <= max_length)) {
		String pre = text.substr(0, cursor_pos);
		String post = text.substr(cursor_pos, text.length() - cursor_pos);
		text = pre + p_text + post;
		_shape();
		TextServer::Direction dir = TS->shaped_text_get_dominant_direciton_in_range(text_rid, cursor_pos, cursor_pos + p_text.length());
		if (dir != TextServer::DIRECTION_AUTO) {
			input_direction = (TextDirection)dir;
		}
		set_cursor_position(cursor_pos + p_text.length());
	} else {
		emit_signal("text_change_rejected");
	}
}

void LineEdit::clear_internal() {
	deselect();
	_clear_undo_stack();
	cursor_pos = 0;
	scroll_offset = 0;
	undo_text = "";
	text = "";
	_shape();
	update();
}

Size2 LineEdit::get_minimum_size() const {
	Ref<StyleBox> style = get_theme_stylebox("normal");
	Ref<Font> font = get_theme_font("font");
	int font_size = get_theme_font_size("font_size");

	Size2 min_size;

	// Minimum size of text.
	int space_size = font->get_char_size('m', 0, font_size).x;
	min_size.width = get_theme_constant("minimum_spaces") * space_size;

	if (expand_to_text_length) {
		// Add a space because some fonts are too exact, and because cursor needs a bit more when at the end.
		min_size.width = MAX(min_size.width, full_width + space_size);
	}

	min_size.height = MAX(TS->shaped_text_get_size(text_rid).y, font->get_height(font_size));

	// Take icons into account.
	bool using_placeholder = text.empty() && ime_text.empty();
	bool display_clear_icon = !using_placeholder && is_editable() && clear_button_enabled;
	if (right_icon.is_valid() || display_clear_icon) {
		Ref<Texture2D> r_icon = display_clear_icon ? Control::get_theme_icon("clear") : right_icon;
		min_size.width += r_icon->get_width();
		min_size.height = MAX(min_size.height, r_icon->get_height());
	}

	return style->get_minimum_size() + min_size;
}

void LineEdit::deselect() {
	selection.begin = 0;
	selection.end = 0;
	selection.cursor_start = 0;
	selection.enabled = false;
	selection.creating = false;
	selection.doubleclick = false;
	update();
}

void LineEdit::selection_delete() {
	if (selection.enabled) {
		delete_text(selection.begin, selection.end);
	}

	deselect();
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
	if (!selecting_enabled) {
		return;
	}

	selection.begin = cursor_pos;
	selection.end = selection.cursor_start;

	if (selection.end < selection.begin) {
		int aux = selection.end;
		selection.end = selection.begin;
		selection.begin = aux;
	}

	selection.enabled = (selection.begin != selection.end);
}

void LineEdit::select_all() {
	if (!selecting_enabled) {
		return;
	}

	if (!text.length()) {
		return;
	}

	selection.begin = 0;
	selection.end = text.length();
	selection.enabled = true;
	update();
}

void LineEdit::set_editable(bool p_editable) {
	if (editable == p_editable) {
		return;
	}

	editable = p_editable;
	_generate_context_menu();

	minimum_size_changed();
	update();
}

bool LineEdit::is_editable() const {
	return editable;
}

void LineEdit::set_secret(bool p_secret) {
	if (pass != p_secret) {
		pass = p_secret;
		_shape();
	}
	update();
}

bool LineEdit::is_secret() const {
	return pass;
}

void LineEdit::set_secret_character(const String &p_string) {
	// An empty string as the secret character would crash the engine.
	// It also wouldn't make sense to use multiple characters as the secret character.
	ERR_FAIL_COND_MSG(p_string.length() != 1, "Secret character must be exactly one character long (" + itos(p_string.length()) + " characters given).");

	if (secret_character != p_string) {
		secret_character = p_string;
		_shape();
	}
	update();
}

String LineEdit::get_secret_character() const {
	return secret_character;
}

void LineEdit::select(int p_from, int p_to) {
	if (!selecting_enabled) {
		return;
	}

	if (p_from == 0 && p_to == 0) {
		deselect();
		return;
	}

	int len = text.length();
	if (p_from < 0) {
		p_from = 0;
	}
	if (p_from > len) {
		p_from = len;
	}
	if (p_to < 0 || p_to > len) {
		p_to = len;
	}

	if (p_from >= p_to) {
		return;
	}

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
			if (editable) {
				undo();
			}
		} break;
		case MENU_REDO: {
			if (editable) {
				redo();
			}
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
			if (editable) {
				append_at_cursor(String::chr(0x200E));
			}
		} break;
		case MENU_INSERT_RLM: {
			if (editable) {
				append_at_cursor(String::chr(0x200F));
			}
		} break;
		case MENU_INSERT_LRE: {
			if (editable) {
				append_at_cursor(String::chr(0x202A));
			}
		} break;
		case MENU_INSERT_RLE: {
			if (editable) {
				append_at_cursor(String::chr(0x202B));
			}
		} break;
		case MENU_INSERT_LRO: {
			if (editable) {
				append_at_cursor(String::chr(0x202D));
			}
		} break;
		case MENU_INSERT_RLO: {
			if (editable) {
				append_at_cursor(String::chr(0x202E));
			}
		} break;
		case MENU_INSERT_PDF: {
			if (editable) {
				append_at_cursor(String::chr(0x202C));
			}
		} break;
		case MENU_INSERT_ALM: {
			if (editable) {
				append_at_cursor(String::chr(0x061C));
			}
		} break;
		case MENU_INSERT_LRI: {
			if (editable) {
				append_at_cursor(String::chr(0x2066));
			}
		} break;
		case MENU_INSERT_RLI: {
			if (editable) {
				append_at_cursor(String::chr(0x2067));
			}
		} break;
		case MENU_INSERT_FSI: {
			if (editable) {
				append_at_cursor(String::chr(0x2068));
			}
		} break;
		case MENU_INSERT_PDI: {
			if (editable) {
				append_at_cursor(String::chr(0x2069));
			}
		} break;
		case MENU_INSERT_ZWJ: {
			if (editable) {
				append_at_cursor(String::chr(0x200D));
			}
		} break;
		case MENU_INSERT_ZWNJ: {
			if (editable) {
				append_at_cursor(String::chr(0x200C));
			}
		} break;
		case MENU_INSERT_WJ: {
			if (editable) {
				append_at_cursor(String::chr(0x2060));
			}
		} break;
		case MENU_INSERT_SHY: {
			if (editable) {
				append_at_cursor(String::chr(0x00AD));
			}
		}
	}
}

void LineEdit::set_context_menu_enabled(bool p_enable) {
	context_menu_enabled = p_enable;
}

bool LineEdit::is_context_menu_enabled() {
	return context_menu_enabled;
}

PopupMenu *LineEdit::get_menu() const {
	return menu;
}

void LineEdit::_editor_settings_changed() {
#ifdef TOOLS_ENABLED
	cursor_set_blink_enabled(EDITOR_DEF("text_editor/cursor/caret_blink", false));
	cursor_set_blink_speed(EDITOR_DEF("text_editor/cursor/caret_blink_speed", 0.65));
#endif
}

void LineEdit::set_expand_to_text_length(bool p_enabled) {
	expand_to_text_length = p_enabled;
	minimum_size_changed();
	set_cursor_position(cursor_pos);
}

bool LineEdit::get_expand_to_text_length() const {
	return expand_to_text_length;
}

void LineEdit::set_clear_button_enabled(bool p_enabled) {
	if (clear_button_enabled == p_enabled) {
		return;
	}
	clear_button_enabled = p_enabled;
	_fit_to_width();
	minimum_size_changed();
	update();
}

bool LineEdit::is_clear_button_enabled() const {
	return clear_button_enabled;
}

void LineEdit::set_shortcut_keys_enabled(bool p_enabled) {
	shortcut_keys_enabled = p_enabled;

	_generate_context_menu();
}

bool LineEdit::is_shortcut_keys_enabled() const {
	return shortcut_keys_enabled;
}

void LineEdit::set_virtual_keyboard_enabled(bool p_enable) {
	virtual_keyboard_enabled = p_enable;
}

bool LineEdit::is_virtual_keyboard_enabled() const {
	return virtual_keyboard_enabled;
}

void LineEdit::set_selecting_enabled(bool p_enabled) {
	selecting_enabled = p_enabled;

	if (!selecting_enabled) {
		deselect();
	}

	_generate_context_menu();
}

bool LineEdit::is_selecting_enabled() const {
	return selecting_enabled;
}

void LineEdit::set_right_icon(const Ref<Texture2D> &p_icon) {
	if (right_icon == p_icon) {
		return;
	}
	right_icon = p_icon;
	_fit_to_width();
	minimum_size_changed();
	update();
}

Ref<Texture2D> LineEdit::get_right_icon() {
	return right_icon;
}

void LineEdit::_text_changed() {
	_emit_text_change();
	_clear_redo();
}

void LineEdit::_emit_text_change() {
	emit_signal("text_changed", text);
	_change_notify("text");
	text_changed_dirty = false;
}

void LineEdit::_shape() {
	Size2 old_size = TS->shaped_text_get_size(text_rid);
	TS->shaped_text_clear(text_rid);

	String t;
	if (text.length() == 0) {
		t = placeholder_translated;
	} else if (pass) {
		t = secret_character.repeat(text.length() + ime_text.length());
	} else {
		if (ime_text.length() > 0) {
			t = text.substr(0, cursor_pos) + ime_text + text.substr(cursor_pos, text.length());
		} else {
			t = text;
		}
	}
	if (text_direction == Control::TEXT_DIRECTION_INHERITED) {
		TS->shaped_text_set_direction(text_rid, is_layout_rtl() ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR);
	} else {
		TS->shaped_text_set_direction(text_rid, (TextServer::Direction)text_direction);
	}
	TS->shaped_text_set_preserve_control(text_rid, draw_control_chars);

	const Ref<Font> &font = get_theme_font("font");
	int font_size = get_theme_font_size("font_size");
	TS->shaped_text_add_string(text_rid, t, font->get_rids(), font_size, opentype_features, (language != "") ? language : TranslationServer::get_singleton()->get_tool_locale());
	TS->shaped_text_set_bidi_override(text_rid, structured_text_parser(st_parser, st_args, t));

	full_width = TS->shaped_text_get_size(text_rid).x;
	_fit_to_width();

	Size2 size = TS->shaped_text_get_size(text_rid);

	if ((expand_to_text_length && old_size.x != size.x) || (old_size.y != size.y)) {
		minimum_size_changed();
	}
}

void LineEdit::_fit_to_width() {
	if (align == ALIGN_FILL) {
		Ref<StyleBox> style = get_theme_stylebox("normal");
		int t_width = get_size().width - style->get_margin(MARGIN_RIGHT) - style->get_margin(MARGIN_LEFT);
		bool using_placeholder = text.empty() && ime_text.empty();
		bool display_clear_icon = !using_placeholder && is_editable() && clear_button_enabled;
		if (right_icon.is_valid() || display_clear_icon) {
			Ref<Texture2D> r_icon = display_clear_icon ? Control::get_theme_icon("clear") : right_icon;
			t_width -= r_icon->get_width();
		}
		TS->shaped_text_fit_to_width(text_rid, MAX(t_width, full_width));
	}
}

void LineEdit::_clear_redo() {
	_create_undo_state();
	if (undo_stack_pos == nullptr) {
		return;
	}

	undo_stack_pos = undo_stack_pos->next();
	while (undo_stack_pos) {
		List<TextOperation>::Element *elem = undo_stack_pos;
		undo_stack_pos = undo_stack_pos->next();
		undo_stack.erase(elem);
	}
	_create_undo_state();
}

void LineEdit::_clear_undo_stack() {
	undo_stack.clear();
	undo_stack_pos = nullptr;
	_create_undo_state();
}

void LineEdit::_create_undo_state() {
	TextOperation op;
	op.text = text;
	op.cursor_pos = cursor_pos;
	op.scroll_offset = scroll_offset;
	undo_stack.push_back(op);
}

void LineEdit::_generate_context_menu() {
	// Reorganize context menu.
	menu->clear();
	if (editable) {
		menu->add_item(RTR("Cut"), MENU_CUT, is_shortcut_keys_enabled() ? KEY_MASK_CMD | KEY_X : 0);
	}
	menu->add_item(RTR("Copy"), MENU_COPY, is_shortcut_keys_enabled() ? KEY_MASK_CMD | KEY_C : 0);
	if (editable) {
		menu->add_item(RTR("Paste"), MENU_PASTE, is_shortcut_keys_enabled() ? KEY_MASK_CMD | KEY_V : 0);
	}
	menu->add_separator();
	if (is_selecting_enabled()) {
		menu->add_item(RTR("Select All"), MENU_SELECT_ALL, is_shortcut_keys_enabled() ? KEY_MASK_CMD | KEY_A : 0);
	}
	if (editable) {
		menu->add_item(RTR("Clear"), MENU_CLEAR);
		menu->add_separator();
		menu->add_item(RTR("Undo"), MENU_UNDO, is_shortcut_keys_enabled() ? KEY_MASK_CMD | KEY_Z : 0);
		menu->add_item(RTR("Redo"), MENU_REDO, is_shortcut_keys_enabled() ? KEY_MASK_CMD | KEY_MASK_SHIFT | KEY_Z : 0);
	}
	menu->add_separator();
	menu->add_submenu_item(RTR("Text writing direction"), "DirMenu");
	menu->add_separator();
	menu->add_check_item(RTR("Display control characters"), MENU_DISPLAY_UCC);
	if (editable) {
		menu->add_submenu_item(RTR("Insert control character"), "CTLMenu");
	}
}

bool LineEdit::_set(const StringName &p_name, const Variant &p_value) {
	String str = p_name;
	if (str.begins_with("opentype_features/")) {
		String name = str.get_slicec('/', 1);
		int32_t tag = TS->name_to_tag(name);
		double value = p_value;
		if (value == -1) {
			if (opentype_features.has(tag)) {
				opentype_features.erase(tag);
				_shape();
				update();
			}
		} else {
			if ((double)opentype_features[tag] != value) {
				opentype_features[tag] = value;
				_shape();
				update();
			}
		}
		_change_notify();
		return true;
	}

	return false;
}

bool LineEdit::_get(const StringName &p_name, Variant &r_ret) const {
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

void LineEdit::_get_property_list(List<PropertyInfo> *p_list) const {
	for (const Variant *ftr = opentype_features.next(nullptr); ftr != nullptr; ftr = opentype_features.next(ftr)) {
		String name = TS->tag_to_name(*ftr);
		p_list->push_back(PropertyInfo(Variant::FLOAT, "opentype_features/" + name));
	}
	p_list->push_back(PropertyInfo(Variant::NIL, "opentype_features/_new", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
}

void LineEdit::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_text_changed"), &LineEdit::_text_changed);

	ClassDB::bind_method(D_METHOD("set_align", "align"), &LineEdit::set_align);
	ClassDB::bind_method(D_METHOD("get_align"), &LineEdit::get_align);

	ClassDB::bind_method(D_METHOD("_gui_input"), &LineEdit::_gui_input);
	ClassDB::bind_method(D_METHOD("clear"), &LineEdit::clear);
	ClassDB::bind_method(D_METHOD("select", "from", "to"), &LineEdit::select, DEFVAL(0), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("select_all"), &LineEdit::select_all);
	ClassDB::bind_method(D_METHOD("deselect"), &LineEdit::deselect);
	ClassDB::bind_method(D_METHOD("set_text", "text"), &LineEdit::set_text);
	ClassDB::bind_method(D_METHOD("get_text"), &LineEdit::get_text);
	ClassDB::bind_method(D_METHOD("get_draw_control_chars"), &LineEdit::get_draw_control_chars);
	ClassDB::bind_method(D_METHOD("set_draw_control_chars", "enable"), &LineEdit::set_draw_control_chars);
	ClassDB::bind_method(D_METHOD("set_text_direction", "direction"), &LineEdit::set_text_direction);
	ClassDB::bind_method(D_METHOD("get_text_direction"), &LineEdit::get_text_direction);
	ClassDB::bind_method(D_METHOD("set_opentype_feature", "tag", "value"), &LineEdit::set_opentype_feature);
	ClassDB::bind_method(D_METHOD("get_opentype_feature", "tag"), &LineEdit::get_opentype_feature);
	ClassDB::bind_method(D_METHOD("clear_opentype_features"), &LineEdit::clear_opentype_features);
	ClassDB::bind_method(D_METHOD("set_language", "language"), &LineEdit::set_language);
	ClassDB::bind_method(D_METHOD("get_language"), &LineEdit::get_language);
	ClassDB::bind_method(D_METHOD("set_structured_text_bidi_override", "parser"), &LineEdit::set_structured_text_bidi_override);
	ClassDB::bind_method(D_METHOD("get_structured_text_bidi_override"), &LineEdit::get_structured_text_bidi_override);
	ClassDB::bind_method(D_METHOD("set_structured_text_bidi_override_options", "args"), &LineEdit::set_structured_text_bidi_override_options);
	ClassDB::bind_method(D_METHOD("get_structured_text_bidi_override_options"), &LineEdit::get_structured_text_bidi_override_options);
	ClassDB::bind_method(D_METHOD("set_placeholder", "text"), &LineEdit::set_placeholder);
	ClassDB::bind_method(D_METHOD("get_placeholder"), &LineEdit::get_placeholder);
	ClassDB::bind_method(D_METHOD("set_placeholder_alpha", "alpha"), &LineEdit::set_placeholder_alpha);
	ClassDB::bind_method(D_METHOD("get_placeholder_alpha"), &LineEdit::get_placeholder_alpha);
	ClassDB::bind_method(D_METHOD("set_cursor_position", "position"), &LineEdit::set_cursor_position);
	ClassDB::bind_method(D_METHOD("get_cursor_position"), &LineEdit::get_cursor_position);
	ClassDB::bind_method(D_METHOD("get_scroll_offset"), &LineEdit::get_scroll_offset);
	ClassDB::bind_method(D_METHOD("set_expand_to_text_length", "enabled"), &LineEdit::set_expand_to_text_length);
	ClassDB::bind_method(D_METHOD("get_expand_to_text_length"), &LineEdit::get_expand_to_text_length);
	ClassDB::bind_method(D_METHOD("cursor_set_blink_enabled", "enabled"), &LineEdit::cursor_set_blink_enabled);
	ClassDB::bind_method(D_METHOD("cursor_get_blink_enabled"), &LineEdit::cursor_get_blink_enabled);
	ClassDB::bind_method(D_METHOD("set_mid_grapheme_caret_enabled", "enabled"), &LineEdit::set_mid_grapheme_caret_enabled);
	ClassDB::bind_method(D_METHOD("get_mid_grapheme_caret_enabled"), &LineEdit::get_mid_grapheme_caret_enabled);
	ClassDB::bind_method(D_METHOD("cursor_set_force_displayed", "enabled"), &LineEdit::cursor_set_force_displayed);
	ClassDB::bind_method(D_METHOD("cursor_get_force_displayed"), &LineEdit::cursor_get_force_displayed);
	ClassDB::bind_method(D_METHOD("cursor_set_blink_speed", "blink_speed"), &LineEdit::cursor_set_blink_speed);
	ClassDB::bind_method(D_METHOD("cursor_get_blink_speed"), &LineEdit::cursor_get_blink_speed);
	ClassDB::bind_method(D_METHOD("set_max_length", "chars"), &LineEdit::set_max_length);
	ClassDB::bind_method(D_METHOD("get_max_length"), &LineEdit::get_max_length);
	ClassDB::bind_method(D_METHOD("append_at_cursor", "text"), &LineEdit::append_at_cursor);
	ClassDB::bind_method(D_METHOD("delete_char_at_cursor"), &LineEdit::delete_char);
	ClassDB::bind_method(D_METHOD("delete_text", "from_column", "to_column"), &LineEdit::delete_text);
	ClassDB::bind_method(D_METHOD("set_editable", "enabled"), &LineEdit::set_editable);
	ClassDB::bind_method(D_METHOD("is_editable"), &LineEdit::is_editable);
	ClassDB::bind_method(D_METHOD("set_secret", "enabled"), &LineEdit::set_secret);
	ClassDB::bind_method(D_METHOD("is_secret"), &LineEdit::is_secret);
	ClassDB::bind_method(D_METHOD("set_secret_character", "character"), &LineEdit::set_secret_character);
	ClassDB::bind_method(D_METHOD("get_secret_character"), &LineEdit::get_secret_character);
	ClassDB::bind_method(D_METHOD("menu_option", "option"), &LineEdit::menu_option);
	ClassDB::bind_method(D_METHOD("get_menu"), &LineEdit::get_menu);
	ClassDB::bind_method(D_METHOD("set_context_menu_enabled", "enable"), &LineEdit::set_context_menu_enabled);
	ClassDB::bind_method(D_METHOD("is_context_menu_enabled"), &LineEdit::is_context_menu_enabled);
	ClassDB::bind_method(D_METHOD("set_virtual_keyboard_enabled", "enable"), &LineEdit::set_virtual_keyboard_enabled);
	ClassDB::bind_method(D_METHOD("is_virtual_keyboard_enabled"), &LineEdit::is_virtual_keyboard_enabled);
	ClassDB::bind_method(D_METHOD("set_clear_button_enabled", "enable"), &LineEdit::set_clear_button_enabled);
	ClassDB::bind_method(D_METHOD("is_clear_button_enabled"), &LineEdit::is_clear_button_enabled);
	ClassDB::bind_method(D_METHOD("set_shortcut_keys_enabled", "enable"), &LineEdit::set_shortcut_keys_enabled);
	ClassDB::bind_method(D_METHOD("is_shortcut_keys_enabled"), &LineEdit::is_shortcut_keys_enabled);
	ClassDB::bind_method(D_METHOD("set_selecting_enabled", "enable"), &LineEdit::set_selecting_enabled);
	ClassDB::bind_method(D_METHOD("is_selecting_enabled"), &LineEdit::is_selecting_enabled);
	ClassDB::bind_method(D_METHOD("set_right_icon", "icon"), &LineEdit::set_right_icon);
	ClassDB::bind_method(D_METHOD("get_right_icon"), &LineEdit::get_right_icon);

	ADD_SIGNAL(MethodInfo("text_changed", PropertyInfo(Variant::STRING, "new_text")));
	ADD_SIGNAL(MethodInfo("text_change_rejected"));
	ADD_SIGNAL(MethodInfo("text_entered", PropertyInfo(Variant::STRING, "new_text")));

	BIND_ENUM_CONSTANT(ALIGN_LEFT);
	BIND_ENUM_CONSTANT(ALIGN_CENTER);
	BIND_ENUM_CONSTANT(ALIGN_RIGHT);
	BIND_ENUM_CONSTANT(ALIGN_FILL);

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

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "text"), "set_text", "get_text");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "align", PROPERTY_HINT_ENUM, "Left,Center,Right,Fill"), "set_align", "get_align");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_length"), "set_max_length", "get_max_length");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editable"), "set_editable", "is_editable");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "secret"), "set_secret", "is_secret");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "secret_character"), "set_secret_character", "get_secret_character");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "expand_to_text_length"), "set_expand_to_text_length", "get_expand_to_text_length");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "context_menu_enabled"), "set_context_menu_enabled", "is_context_menu_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "virtual_keyboard_enabled"), "set_virtual_keyboard_enabled", "is_virtual_keyboard_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "clear_button_enabled"), "set_clear_button_enabled", "is_clear_button_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "shortcut_keys_enabled"), "set_shortcut_keys_enabled", "is_shortcut_keys_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "selecting_enabled"), "set_selecting_enabled", "is_selecting_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "right_icon", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_right_icon", "get_right_icon");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_direction", PROPERTY_HINT_ENUM, "Auto,LTR,RTL,Inherited"), "set_text_direction", "get_text_direction");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "language"), "set_language", "get_language");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_control_chars"), "set_draw_control_chars", "get_draw_control_chars");
	ADD_GROUP("Structured Text", "structured_text_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "structured_text_bidi_override", PROPERTY_HINT_ENUM, "Default,URI,File,Email,List,None,Custom"), "set_structured_text_bidi_override", "get_structured_text_bidi_override");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "structured_text_bidi_override_options"), "set_structured_text_bidi_override_options", "get_structured_text_bidi_override_options");
	ADD_GROUP("Placeholder", "placeholder_");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "placeholder_text"), "set_placeholder", "get_placeholder");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "placeholder_alpha", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_placeholder_alpha", "get_placeholder_alpha");
	ADD_GROUP("Caret", "caret_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "caret_blink"), "cursor_set_blink_enabled", "cursor_get_blink_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "caret_blink_speed", PROPERTY_HINT_RANGE, "0.1,10,0.01"), "cursor_set_blink_speed", "cursor_get_blink_speed");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "caret_position"), "set_cursor_position", "get_cursor_position");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "caret_force_displayed"), "cursor_set_force_displayed", "cursor_get_force_displayed");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "caret_mid_grapheme"), "set_mid_grapheme_caret_enabled", "get_mid_grapheme_caret_enabled");
}

LineEdit::LineEdit() {
	text_rid = TS->create_shaped_text();
	_create_undo_state();

	clear_button_status.press_attempt = false;
	clear_button_status.pressing_inside = false;

	deselect();
	set_focus_mode(FOCUS_ALL);
	set_default_cursor_shape(CURSOR_IBEAM);
	set_mouse_filter(MOUSE_FILTER_STOP);

	caret_blink_timer = memnew(Timer);
	add_child(caret_blink_timer);
	caret_blink_timer->set_wait_time(0.65);
	caret_blink_timer->connect("timeout", callable_mp(this, &LineEdit::_toggle_draw_caret));
	cursor_set_blink_enabled(false);

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

	set_editable(true); // Initialise to opposite first, so we get past the early-out in set_editable.
	menu->connect("id_pressed", callable_mp(this, &LineEdit::menu_option));
	menu_dir->connect("id_pressed", callable_mp(this, &LineEdit::menu_option));
	menu_ctl->connect("id_pressed", callable_mp(this, &LineEdit::menu_option));
}

LineEdit::~LineEdit() {
	TS->free(text_rid);
}
