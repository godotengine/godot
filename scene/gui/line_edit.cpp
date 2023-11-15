/**************************************************************************/
/*  line_edit.cpp                                                         */
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

#include "line_edit.h"

#include "core/input/input_map.h"
#include "core/object/message_queue.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/string/translation.h"
#include "scene/gui/label.h"
#include "scene/main/window.h"
#include "scene/theme/theme_db.h"
#include "servers/display_server.h"
#include "servers/text_server.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_settings.h"
#endif

void LineEdit::_swap_current_input_direction() {
	if (input_direction == TEXT_DIRECTION_LTR) {
		input_direction = TEXT_DIRECTION_RTL;
	} else {
		input_direction = TEXT_DIRECTION_LTR;
	}
	set_caret_column(get_caret_column());
	queue_redraw();
}

void LineEdit::_move_caret_left(bool p_select, bool p_move_by_word) {
	if (selection.enabled && !p_select) {
		set_caret_column(selection.begin);
		deselect();
		return;
	}

	shift_selection_check_pre(p_select);

	if (p_move_by_word) {
		int cc = caret_column;

		PackedInt32Array words = TS->shaped_text_get_word_breaks(text_rid);
		for (int i = words.size() - 2; i >= 0; i = i - 2) {
			if (words[i] < cc) {
				cc = words[i];
				break;
			}
		}

		set_caret_column(cc);
	} else {
		if (caret_mid_grapheme_enabled) {
			set_caret_column(get_caret_column() - 1);
		} else {
			set_caret_column(TS->shaped_text_prev_character_pos(text_rid, get_caret_column()));
		}
	}

	shift_selection_check_post(p_select);
	_reset_caret_blink_timer();
}

void LineEdit::_move_caret_right(bool p_select, bool p_move_by_word) {
	if (selection.enabled && !p_select) {
		set_caret_column(selection.end);
		deselect();
		return;
	}

	shift_selection_check_pre(p_select);

	if (p_move_by_word) {
		int cc = caret_column;

		PackedInt32Array words = TS->shaped_text_get_word_breaks(text_rid);
		for (int i = 1; i < words.size(); i = i + 2) {
			if (words[i] > cc) {
				cc = words[i];
				break;
			}
		}

		set_caret_column(cc);
	} else {
		if (caret_mid_grapheme_enabled) {
			set_caret_column(get_caret_column() + 1);
		} else {
			set_caret_column(TS->shaped_text_next_character_pos(text_rid, get_caret_column()));
		}
	}

	shift_selection_check_post(p_select);
	_reset_caret_blink_timer();
}

void LineEdit::_move_caret_start(bool p_select) {
	shift_selection_check_pre(p_select);
	set_caret_column(0);
	shift_selection_check_post(p_select);
}

void LineEdit::_move_caret_end(bool p_select) {
	shift_selection_check_pre(p_select);
	set_caret_column(text.length());
	shift_selection_check_post(p_select);
}

void LineEdit::_backspace(bool p_word, bool p_all_to_left) {
	if (!editable) {
		return;
	}

	if (p_all_to_left) {
		deselect();
		text = text.substr(0, caret_column);
		_text_changed();
		return;
	}

	if (selection.enabled) {
		selection_delete();
		return;
	}

	if (p_word) {
		int cc = caret_column;

		PackedInt32Array words = TS->shaped_text_get_word_breaks(text_rid);
		for (int i = words.size() - 2; i >= 0; i = i - 2) {
			if (words[i] < cc) {
				cc = words[i];
				break;
			}
		}

		delete_text(cc, caret_column);

		set_caret_column(cc);
	} else {
		delete_char();
	}
}

void LineEdit::_delete(bool p_word, bool p_all_to_right) {
	if (!editable) {
		return;
	}

	if (p_all_to_right) {
		deselect();
		text = text.substr(caret_column, text.length() - caret_column);
		_shape();
		set_caret_column(0);
		_text_changed();
		return;
	}

	if (selection.enabled) {
		selection_delete();
		return;
	}

	int text_len = text.length();

	if (caret_column == text_len) {
		return; // Nothing to do.
	}

	if (p_word) {
		int cc = caret_column;
		PackedInt32Array words = TS->shaped_text_get_word_breaks(text_rid);
		for (int i = 1; i < words.size(); i = i + 2) {
			if (words[i] > cc) {
				cc = words[i];
				break;
			}
		}

		delete_text(caret_column, cc);
		set_caret_column(caret_column);
	} else {
		if (caret_mid_grapheme_enabled) {
			set_caret_column(caret_column + 1);
			delete_char();
		} else {
			int cc = caret_column;
			set_caret_column(TS->shaped_text_next_character_pos(text_rid, caret_column));
			delete_text(cc, caret_column);
		}
	}
}

void LineEdit::unhandled_key_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventKey> k = p_event;

	if (k.is_valid()) {
		if (!k->is_pressed()) {
			return;
		}
		// Handle Unicode (with modifiers active, process after shortcuts).
		if (has_focus() && editable && (k->get_unicode() >= 32)) {
			selection_delete();
			char32_t ucodestr[2] = { (char32_t)k->get_unicode(), 0 };
			int prev_len = text.length();
			insert_text_at_caret(ucodestr);
			if (text.length() != prev_len) {
				_text_changed();
			}
			accept_event();
		}
	}
}

void LineEdit::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventMouseButton> b = p_event;

	if (b.is_valid()) {
		if (ime_text.length() != 0) {
			// Ignore mouse clicks in IME input mode.
			return;
		}
		if (b->is_pressed() && b->get_button_index() == MouseButton::RIGHT && context_menu_enabled) {
			_update_context_menu();
			menu->set_position(get_screen_position() + get_local_mouse_position());
			menu->reset_size();
			menu->popup();
			grab_focus();
			accept_event();
			return;
		}

		if (is_middle_mouse_paste_enabled() && b->is_pressed() && b->get_button_index() == MouseButton::MIDDLE && is_editable() && DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_CLIPBOARD_PRIMARY)) {
			String paste_buffer = DisplayServer::get_singleton()->clipboard_get_primary().strip_escapes();

			deselect();
			set_caret_at_pixel_pos(b->get_position().x);
			if (!paste_buffer.is_empty()) {
				insert_text_at_caret(paste_buffer);

				if (!text_changed_dirty) {
					if (is_inside_tree()) {
						MessageQueue::get_singleton()->push_call(this, "_text_changed");
					}
					text_changed_dirty = true;
				}
			}
			grab_focus();
			accept_event();
			return;
		}

		if (b->get_button_index() != MouseButton::LEFT) {
			return;
		}

		_reset_caret_blink_timer();
		if (b->is_pressed()) {
			accept_event(); // Don't pass event further when clicked on text field.
			if (!text.is_empty() && is_editable() && _is_over_clear_button(b->get_position())) {
				clear_button_status.press_attempt = true;
				clear_button_status.pressing_inside = true;
				queue_redraw();
				return;
			}

			if (b->is_shift_pressed()) {
				shift_selection_check_pre(true);
			}

			set_caret_at_pixel_pos(b->get_position().x);

			if (b->is_shift_pressed()) {
				selection_fill_at_caret();
				selection.creating = true;

			} else {
				if (selecting_enabled) {
					const int triple_click_timeout = 600;
					const int triple_click_tolerance = 5;
					const bool is_triple_click = !b->is_double_click() && (OS::get_singleton()->get_ticks_msec() - last_dblclk) < triple_click_timeout && b->get_position().distance_to(last_dblclk_pos) < triple_click_tolerance;

					if (is_triple_click && text.length()) {
						// Triple-click select all.
						selection.enabled = true;
						selection.begin = 0;
						selection.end = text.length();
						selection.double_click = true;
						last_dblclk = 0;
						set_caret_column(selection.begin);
						if (!pass && DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_CLIPBOARD_PRIMARY)) {
							DisplayServer::get_singleton()->clipboard_set_primary(text);
						}
					} else if (b->is_double_click()) {
						// Double-click select word.
						last_dblclk = OS::get_singleton()->get_ticks_msec();
						last_dblclk_pos = b->get_position();
						PackedInt32Array words = TS->shaped_text_get_word_breaks(text_rid);
						for (int i = 0; i < words.size(); i = i + 2) {
							if ((words[i] < caret_column && words[i + 1] > caret_column) || (i == words.size() - 2 && caret_column == words[i + 1])) {
								selection.enabled = true;
								selection.begin = words[i];
								selection.end = words[i + 1];
								selection.double_click = true;
								set_caret_column(selection.end);
								break;
							}
						}
						if (!pass && DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_CLIPBOARD_PRIMARY)) {
							DisplayServer::get_singleton()->clipboard_set_primary(get_selected_text());
						}
					}
				}

				selection.drag_attempt = false;
				if (!selection.double_click) {
					bool is_inside_sel = selection.enabled && caret_column >= selection.begin && caret_column <= selection.end;
					if (drag_and_drop_selection_enabled && is_inside_sel) {
						selection.drag_attempt = true;
					} else {
						deselect();
						selection.start_column = caret_column;
						selection.creating = true;
					}
				}
			}

			queue_redraw();

		} else {
			if (selection.enabled && !pass && b->get_button_index() == MouseButton::LEFT && DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_CLIPBOARD_PRIMARY)) {
				DisplayServer::get_singleton()->clipboard_set_primary(get_selected_text());
			}
			if (!text.is_empty() && is_editable() && clear_button_enabled) {
				bool press_attempt = clear_button_status.press_attempt;
				clear_button_status.press_attempt = false;
				if (press_attempt && clear_button_status.pressing_inside && _is_over_clear_button(b->get_position())) {
					clear();
					return;
				}
			}

			if ((!selection.creating) && (!selection.double_click)) {
				deselect();
			}
			selection.creating = false;
			selection.double_click = false;
			if (!drag_action) {
				selection.drag_attempt = false;
			}

			if (pending_select_all_on_focus) {
				select_all();
				pending_select_all_on_focus = false;
			}

			show_virtual_keyboard();
		}

		queue_redraw();
		return;
	}

	Ref<InputEventMouseMotion> m = p_event;

	if (m.is_valid()) {
		if (!text.is_empty() && is_editable() && clear_button_enabled) {
			bool last_press_inside = clear_button_status.pressing_inside;
			clear_button_status.pressing_inside = clear_button_status.press_attempt && _is_over_clear_button(m->get_position());
			if (last_press_inside != clear_button_status.pressing_inside) {
				queue_redraw();
			}
		}

		if (m->get_button_mask().has_flag(MouseButtonMask::LEFT)) {
			if (selection.creating) {
				set_caret_at_pixel_pos(m->get_position().x);
				selection_fill_at_caret();
			}
		}

		if (drag_action && can_drop_data(m->get_position(), get_viewport()->gui_get_drag_data())) {
			drag_caret_force_displayed = true;
			set_caret_at_pixel_pos(m->get_position().x);
		}

		return;
	}

	Ref<InputEventKey> k = p_event;

	if (k.is_valid()) {
		if (!k->is_pressed()) {
			if (alt_start && k->get_keycode() == Key::ALT) {
				alt_start = false;
				if ((alt_code > 0x31 && alt_code < 0xd800) || (alt_code > 0xdfff && alt_code <= 0x10ffff)) {
					char32_t ucodestr[2] = { (char32_t)alt_code, 0 };
					insert_text_at_caret(ucodestr);
				}
				accept_event();
				return;
			}
			return;
		}

		// Alt + Unicode input:
		if (k->is_alt_pressed()) {
			if (!alt_start) {
				if (k->get_keycode() == Key::KP_ADD) {
					alt_start = true;
					alt_code = 0;
					accept_event();
					return;
				}
			} else {
				if (k->get_keycode() >= Key::KEY_0 && k->get_keycode() <= Key::KEY_9) {
					alt_code = alt_code << 4;
					alt_code += (uint32_t)(k->get_keycode() - Key::KEY_0);
				}
				if (k->get_keycode() >= Key::KP_0 && k->get_keycode() <= Key::KP_9) {
					alt_code = alt_code << 4;
					alt_code += (uint32_t)(k->get_keycode() - Key::KP_0);
				}
				if (k->get_keycode() >= Key::A && k->get_keycode() <= Key::F) {
					alt_code = alt_code << 4;
					alt_code += (uint32_t)(k->get_keycode() - Key::A) + 10;
				}
				accept_event();
				return;
			}
		}

		if (context_menu_enabled) {
			if (k->is_action("ui_menu", true)) {
				_update_context_menu();
				Point2 pos = Point2(get_caret_pixel_pos().x, (get_size().y + theme_cache.font->get_height(theme_cache.font_size)) / 2);
				menu->set_position(get_screen_position() + pos);
				menu->reset_size();
				menu->popup();
				menu->grab_focus();

				accept_event();
				return;
			}
		}

		// Default is ENTER and KP_ENTER. Cannot use ui_accept as default includes SPACE.
		if (k->is_action("ui_text_submit", false)) {
			emit_signal(SNAME("text_submitted"), text);
			if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_VIRTUAL_KEYBOARD) && virtual_keyboard_enabled) {
				DisplayServer::get_singleton()->virtual_keyboard_hide();
			}
			accept_event();
			return;
		}

		if (k->is_action("ui_cancel")) {
			callable_mp((Control *)this, &Control::release_focus).call_deferred();
			accept_event();
			return;
		}

		if (is_shortcut_keys_enabled()) {
			if (k->is_action("ui_copy", true)) {
				copy_text();
				accept_event();
				return;
			}

			if (k->is_action("ui_text_select_all", true)) {
				select();
				accept_event();
				return;
			}

			// Cut / Paste
			if (k->is_action("ui_cut", true)) {
				cut_text();
				accept_event();
				return;
			}

			if (k->is_action("ui_paste", true)) {
				paste_text();
				accept_event();
				return;
			}

			// Undo / Redo
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
		}

		// BACKSPACE
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
			accept_event();
			return;
		}

		// DELETE
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

		// Cursor Movement

		k = k->duplicate();
		bool shift_pressed = k->is_shift_pressed();
		// Remove shift or else actions will not match. Use above variable for selection.
		k->set_shift_pressed(false);

		if (k->is_action("ui_text_caret_word_left", true)) {
			_move_caret_left(shift_pressed, true);
			accept_event();
			return;
		}
		if (k->is_action("ui_text_caret_left", true)) {
			_move_caret_left(shift_pressed);
			accept_event();
			return;
		}
		if (k->is_action("ui_text_caret_word_right", true)) {
			_move_caret_right(shift_pressed, true);
			accept_event();
			return;
		}
		if (k->is_action("ui_text_caret_right", true)) {
			_move_caret_right(shift_pressed, false);
			accept_event();
			return;
		}

		// Up = Home, Down = End
		if (k->is_action("ui_text_caret_up", true) || k->is_action("ui_text_caret_line_start", true) || k->is_action("ui_text_caret_page_up", true)) {
			_move_caret_start(shift_pressed);
			accept_event();
			return;
		}
		if (k->is_action("ui_text_caret_down", true) || k->is_action("ui_text_caret_line_end", true) || k->is_action("ui_text_caret_page_down", true)) {
			_move_caret_end(shift_pressed);
			accept_event();
			return;
		}

		// Misc
		if (k->is_action("ui_swap_input_direction", true)) {
			_swap_current_input_direction();
			accept_event();
			return;
		}

		_reset_caret_blink_timer();

		// Allow unicode handling if:
		// * No Modifiers are pressed (except shift)
		bool allow_unicode_handling = !(k->is_ctrl_pressed() || k->is_alt_pressed() || k->is_meta_pressed());

		if (allow_unicode_handling && editable && k->get_unicode() >= 32) {
			// Handle Unicode if no modifiers are active.
			selection_delete();
			char32_t ucodestr[2] = { (char32_t)k->get_unicode(), 0 };
			int prev_len = text.length();
			insert_text_at_caret(ucodestr);
			if (text.length() != prev_len) {
				_text_changed();
			}
			accept_event();
			return;
		}
	}
}

void LineEdit::set_horizontal_alignment(HorizontalAlignment p_alignment) {
	ERR_FAIL_INDEX((int)p_alignment, 4);
	if (alignment == p_alignment) {
		return;
	}

	alignment = p_alignment;
	_shape();
	queue_redraw();
}

HorizontalAlignment LineEdit::get_horizontal_alignment() const {
	return alignment;
}

Variant LineEdit::get_drag_data(const Point2 &p_point) {
	Variant ret = Control::get_drag_data(p_point);
	if (ret != Variant()) {
		return ret;
	}

	if (selection.drag_attempt && selection.enabled) {
		String t = get_selected_text();
		Label *l = memnew(Label);
		l->set_text(t);
		set_drag_preview(l);
		return t;
	}

	return Variant();
}

bool LineEdit::can_drop_data(const Point2 &p_point, const Variant &p_data) const {
	bool drop_override = Control::can_drop_data(p_point, p_data); // In case user wants to drop custom data.
	if (drop_override) {
		return drop_override;
	}

	return is_editable() && p_data.get_type() == Variant::STRING;
}

void LineEdit::drop_data(const Point2 &p_point, const Variant &p_data) {
	Control::drop_data(p_point, p_data);

	if (p_data.get_type() == Variant::STRING && is_editable()) {
		set_caret_at_pixel_pos(p_point.x);
		int caret_column_tmp = caret_column;
		bool is_inside_sel = selection.enabled && caret_column >= selection.begin && caret_column <= selection.end;
		if (Input::get_singleton()->is_key_pressed(Key::CMD_OR_CTRL)) {
			is_inside_sel = selection.enabled && caret_column > selection.begin && caret_column < selection.end;
		}
		if (selection.drag_attempt) {
			selection.drag_attempt = false;
			if (!is_inside_sel) {
				if (!Input::get_singleton()->is_key_pressed(Key::CMD_OR_CTRL)) {
					if (caret_column_tmp > selection.end) {
						caret_column_tmp = caret_column_tmp - (selection.end - selection.begin);
					}
					selection_delete();
				}

				set_caret_column(caret_column_tmp);
				insert_text_at_caret(p_data);
			}
		} else if (selection.enabled && caret_column >= selection.begin && caret_column <= selection.end) {
			caret_column_tmp = selection.begin;
			selection_delete();
			set_caret_column(caret_column_tmp);
			insert_text_at_caret(p_data);
			grab_focus();
		} else {
			insert_text_at_caret(p_data);
			grab_focus();
		}
		select(caret_column_tmp, caret_column);
		if (!text_changed_dirty) {
			if (is_inside_tree()) {
				MessageQueue::get_singleton()->push_call(this, "_text_changed");
			}
			text_changed_dirty = true;
		}
		queue_redraw();
	}
}

Control::CursorShape LineEdit::get_cursor_shape(const Point2 &p_pos) const {
	if ((!text.is_empty() && is_editable() && _is_over_clear_button(p_pos)) || (!is_editable() && (!is_selecting_enabled() || text.is_empty()))) {
		return CURSOR_ARROW;
	}
	return Control::get_cursor_shape(p_pos);
}

bool LineEdit::_is_over_clear_button(const Point2 &p_pos) const {
	if (!clear_button_enabled || !has_point(p_pos)) {
		return false;
	}
	Ref<Texture2D> icon = theme_cache.clear_icon;
	int x_ofs = theme_cache.normal->get_margin(SIDE_RIGHT);
	return p_pos.x > get_size().width - icon->get_width() - x_ofs;
}

void LineEdit::_update_theme_item_cache() {
	Control::_update_theme_item_cache();

	theme_cache.base_scale = get_theme_default_base_scale();
}

void LineEdit::_notification(int p_what) {
	switch (p_what) {
#ifdef TOOLS_ENABLED
		case NOTIFICATION_ENTER_TREE: {
			if (Engine::get_singleton()->is_editor_hint() && !get_tree()->is_node_being_edited(this)) {
				set_caret_blink_enabled(EDITOR_GET("text_editor/appearance/caret/caret_blink"));
				set_caret_blink_interval(EDITOR_GET("text_editor/appearance/caret/caret_blink_interval"));

				if (!EditorSettings::get_singleton()->is_connected("settings_changed", callable_mp(this, &LineEdit::_editor_settings_changed))) {
					EditorSettings::get_singleton()->connect("settings_changed", callable_mp(this, &LineEdit::_editor_settings_changed));
				}
			}
		} break;
#endif

		case NOTIFICATION_RESIZED: {
			_fit_to_width();
			scroll_offset = 0.0;
			set_caret_column(get_caret_column());
		} break;

		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case NOTIFICATION_THEME_CHANGED: {
			_shape();
			queue_redraw();
		} break;

		case NOTIFICATION_TRANSLATION_CHANGED: {
			placeholder_translated = atr(placeholder);
			_shape();
			queue_redraw();
		} break;

		case NOTIFICATION_WM_WINDOW_FOCUS_IN: {
			window_has_focus = true;
			_validate_caret_can_draw();
			queue_redraw();
		} break;

		case NOTIFICATION_WM_WINDOW_FOCUS_OUT: {
			window_has_focus = false;
			_validate_caret_can_draw();
			queue_redraw();
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {
			if (caret_blink_enabled && caret_can_draw) {
				caret_blink_timer += get_process_delta_time();

				if (caret_blink_timer >= caret_blink_interval) {
					caret_blink_timer = 0.0;
					_toggle_draw_caret();
				}
			}
		} break;

		case NOTIFICATION_DRAW: {
			int width, height;
			bool rtl = is_layout_rtl();

			Size2 size = get_size();
			width = size.width;
			height = size.height;

			RID ci = get_canvas_item();

			Ref<StyleBox> style = theme_cache.normal;
			if (!is_editable()) {
				style = theme_cache.read_only;
			}
			Ref<Font> font = theme_cache.font;

			if (!flat) {
				style->draw(ci, Rect2(Point2(), size));
			}

			if (has_focus()) {
				theme_cache.focus->draw(ci, Rect2(Point2(), size));
			}

			int x_ofs = 0;
			bool using_placeholder = text.is_empty() && ime_text.is_empty();
			float text_width = TS->shaped_text_get_size(text_rid).x;
			float text_height = TS->shaped_text_get_size(text_rid).y;

			switch (alignment) {
				case HORIZONTAL_ALIGNMENT_FILL:
				case HORIZONTAL_ALIGNMENT_LEFT: {
					if (rtl) {
						x_ofs = MAX(style->get_margin(SIDE_LEFT), int(size.width - Math::ceil(style->get_margin(SIDE_RIGHT) + (text_width))));
					} else {
						x_ofs = style->get_offset().x;
					}
				} break;
				case HORIZONTAL_ALIGNMENT_CENTER: {
					if (!Math::is_zero_approx(scroll_offset)) {
						x_ofs = style->get_offset().x;
					} else {
						x_ofs = MAX(style->get_margin(SIDE_LEFT), int(size.width - (text_width)) / 2);
					}
				} break;
				case HORIZONTAL_ALIGNMENT_RIGHT: {
					if (rtl) {
						x_ofs = style->get_offset().x;
					} else {
						x_ofs = MAX(style->get_margin(SIDE_LEFT), int(size.width - Math::ceil(style->get_margin(SIDE_RIGHT) + (text_width))));
					}
				} break;
			}

			int ofs_max = width - style->get_margin(SIDE_RIGHT);

			int y_area = height - style->get_minimum_size().height;
			int y_ofs = style->get_offset().y + (y_area - text_height) / 2;

			Color selection_color = theme_cache.selection_color;
			Color font_color;
			if (is_editable()) {
				font_color = theme_cache.font_color;
			} else {
				font_color = theme_cache.font_uneditable_color;
			}
			Color font_selected_color = theme_cache.font_selected_color;
			Color caret_color = theme_cache.caret_color;

			// Draw placeholder color.
			if (using_placeholder) {
				font_color = theme_cache.font_placeholder_color;
			}

			bool display_clear_icon = !using_placeholder && is_editable() && clear_button_enabled;
			if (right_icon.is_valid() || display_clear_icon) {
				Ref<Texture2D> r_icon = display_clear_icon ? theme_cache.clear_icon : right_icon;
				Color color_icon(1, 1, 1, !is_editable() ? .5 * .9 : .9);
				if (display_clear_icon) {
					if (clear_button_status.press_attempt && clear_button_status.pressing_inside) {
						color_icon = theme_cache.clear_button_color_pressed;
					} else {
						color_icon = theme_cache.clear_button_color;
					}
				}

				r_icon->draw(ci, Point2(width - r_icon->get_width() - style->get_margin(SIDE_RIGHT), height / 2 - r_icon->get_height() / 2), color_icon);

				if (alignment == HORIZONTAL_ALIGNMENT_CENTER) {
					if (Math::is_zero_approx(scroll_offset)) {
						x_ofs = MAX(style->get_margin(SIDE_LEFT), int(size.width - text_width - r_icon->get_width() - style->get_margin(SIDE_RIGHT) * 2) / 2);
					}
				} else {
					x_ofs = MAX(style->get_margin(SIDE_LEFT), x_ofs - r_icon->get_width() - style->get_margin(SIDE_RIGHT));
				}

				ofs_max -= r_icon->get_width();
			}

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
			const Glyph *glyphs = TS->shaped_text_get_glyphs(text_rid);
			int gl_size = TS->shaped_text_get_glyph_count(text_rid);

			// Draw text.
			ofs.y += TS->shaped_text_get_ascent(text_rid);
			Color font_outline_color = theme_cache.font_outline_color;
			int outline_size = theme_cache.font_outline_size;
			if (outline_size > 0 && font_outline_color.a > 0) {
				Vector2 oofs = ofs;
				for (int i = 0; i < gl_size; i++) {
					for (int j = 0; j < glyphs[i].repeat; j++) {
						if (ceil(oofs.x) >= x_ofs && (oofs.x + glyphs[i].advance) <= ofs_max) {
							if (glyphs[i].font_rid != RID()) {
								TS->font_draw_glyph_outline(glyphs[i].font_rid, ci, glyphs[i].font_size, outline_size, oofs + Vector2(glyphs[i].x_off, glyphs[i].y_off), glyphs[i].index, font_outline_color);
							}
						}
						oofs.x += glyphs[i].advance;
					}
					if (oofs.x >= ofs_max) {
						break;
					}
				}
			}
			for (int i = 0; i < gl_size; i++) {
				bool selected = selection.enabled && glyphs[i].start >= selection.begin && glyphs[i].end <= selection.end;
				for (int j = 0; j < glyphs[i].repeat; j++) {
					if (ceil(ofs.x) >= x_ofs && (ofs.x + glyphs[i].advance) <= ofs_max) {
						if (glyphs[i].font_rid != RID()) {
							TS->font_draw_glyph(glyphs[i].font_rid, ci, glyphs[i].font_size, ofs + Vector2(glyphs[i].x_off, glyphs[i].y_off), glyphs[i].index, selected ? font_selected_color : font_color);
						} else if (((glyphs[i].flags & TextServer::GRAPHEME_IS_VIRTUAL) != TextServer::GRAPHEME_IS_VIRTUAL) && ((glyphs[i].flags & TextServer::GRAPHEME_IS_EMBEDDED_OBJECT) != TextServer::GRAPHEME_IS_EMBEDDED_OBJECT)) {
							TS->draw_hex_code_box(ci, glyphs[i].font_size, ofs + Vector2(glyphs[i].x_off, glyphs[i].y_off), glyphs[i].index, selected ? font_selected_color : font_color);
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
			if ((caret_can_draw && draw_caret) || drag_caret_force_displayed) {
				// Prevent carets from disappearing at theme scales below 1.0 (if the caret width is 1).
				const int caret_width = theme_cache.caret_width * MAX(1, theme_cache.base_scale);

				if (ime_text.is_empty() || ime_selection.y == 0) {
					// Normal caret.
					CaretInfo caret = TS->shaped_text_get_carets(text_rid, ime_text.is_empty() ? caret_column : caret_column + ime_selection.x);
					if (using_placeholder || (caret.l_caret == Rect2() && caret.t_caret == Rect2())) {
						// No carets, add one at the start.
						int h = theme_cache.font->get_height(theme_cache.font_size);
						int y = style->get_offset().y + (y_area - h) / 2;
						caret.l_dir = (rtl) ? TextServer::DIRECTION_RTL : TextServer::DIRECTION_LTR;
						switch (alignment) {
							case HORIZONTAL_ALIGNMENT_FILL:
							case HORIZONTAL_ALIGNMENT_LEFT: {
								if (rtl) {
									caret.l_caret = Rect2(Vector2(ofs_max, y), Size2(caret_width, h));
								} else {
									caret.l_caret = Rect2(Vector2(style->get_offset().x, y), Size2(caret_width, h));
								}
							} break;
							case HORIZONTAL_ALIGNMENT_CENTER: {
								caret.l_caret = Rect2(Vector2(size.x / 2, y), Size2(caret_width, h));
							} break;
							case HORIZONTAL_ALIGNMENT_RIGHT: {
								if (rtl) {
									caret.l_caret = Rect2(Vector2(style->get_offset().x, y), Size2(caret_width, h));
								} else {
									caret.l_caret = Rect2(Vector2(ofs_max, y), Size2(caret_width, h));
								}
							} break;
						}

						RenderingServer::get_singleton()->canvas_item_add_rect(ci, caret.l_caret, caret_color);
					} else {
						if (caret.l_caret != Rect2() && caret.l_dir == TextServer::DIRECTION_AUTO) {
							// Draw extra marker on top of mid caret.
							Rect2 trect = Rect2(caret.l_caret.position.x - 2.5 * caret_width, caret.l_caret.position.y, 6 * caret_width, caret_width);
							trect.position += ofs;
							RenderingServer::get_singleton()->canvas_item_add_rect(ci, trect, caret_color);
						} else if (caret.l_caret != Rect2() && caret.t_caret != Rect2() && caret.l_dir != caret.t_dir) {
							// Draw extra direction marker on top of split caret.
							float d = (caret.l_dir == TextServer::DIRECTION_LTR) ? 0.5 : -3;
							Rect2 trect = Rect2(caret.l_caret.position.x + d * caret_width, caret.l_caret.position.y + caret.l_caret.size.y - caret_width, 3 * caret_width, caret_width);
							trect.position += ofs;
							RenderingServer::get_singleton()->canvas_item_add_rect(ci, trect, caret_color);

							d = (caret.t_dir == TextServer::DIRECTION_LTR) ? 0.5 : -3;
							trect = Rect2(caret.t_caret.position.x + d * caret_width, caret.t_caret.position.y, 3 * caret_width, caret_width);
							trect.position += ofs;
							RenderingServer::get_singleton()->canvas_item_add_rect(ci, trect, caret_color);
						}

						caret.l_caret.position += ofs;
						caret.l_caret.size.x = caret_width;
						RenderingServer::get_singleton()->canvas_item_add_rect(ci, caret.l_caret, caret_color);

						caret.t_caret.position += ofs;
						caret.t_caret.size.x = caret_width;

						RenderingServer::get_singleton()->canvas_item_add_rect(ci, caret.t_caret, caret_color);
					}
				}
				if (!ime_text.is_empty()) {
					{
						// IME intermediate text range.
						Vector<Vector2> sel = TS->shaped_text_get_selection(text_rid, caret_column, caret_column + ime_text.length());
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
							RenderingServer::get_singleton()->canvas_item_add_rect(ci, rect, caret_color);
						}
					}
					{
						// IME caret.
						if (ime_selection.y > 0) {
							Vector<Vector2> sel = TS->shaped_text_get_selection(text_rid, caret_column + ime_selection.x, caret_column + ime_selection.x + ime_selection.y);
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
								RenderingServer::get_singleton()->canvas_item_add_rect(ci, rect, caret_color);
							}
						}
					}
				}
			}

			if (has_focus()) {
				if (get_viewport()->get_window_id() != DisplayServer::INVALID_WINDOW_ID && DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_IME)) {
					DisplayServer::get_singleton()->window_set_ime_active(true, get_viewport()->get_window_id());
					Point2 pos = Point2(get_caret_pixel_pos().x, (get_size().y + theme_cache.font->get_height(theme_cache.font_size)) / 2) + get_global_position();
					if (get_window()->get_embedder()) {
						pos += get_viewport()->get_popup_base_transform().get_origin();
					}
					DisplayServer::get_singleton()->window_set_ime_position(pos, get_viewport()->get_window_id());
				}
			}
		} break;

		case NOTIFICATION_FOCUS_ENTER: {
			_validate_caret_can_draw();

			if (select_all_on_focus) {
				if (Input::get_singleton()->is_mouse_button_pressed(MouseButton::LEFT)) {
					// Select all when the mouse button is up.
					pending_select_all_on_focus = true;
				} else {
					select_all();
				}
			}

			if (get_viewport()->get_window_id() != DisplayServer::INVALID_WINDOW_ID && DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_IME)) {
				DisplayServer::get_singleton()->window_set_ime_active(true, get_viewport()->get_window_id());
				Point2 pos = Point2(get_caret_pixel_pos().x, (get_size().y + theme_cache.font->get_height(theme_cache.font_size)) / 2) + get_global_position();
				if (get_window()->get_embedder()) {
					pos += get_viewport()->get_popup_base_transform().get_origin();
				}
				DisplayServer::get_singleton()->window_set_ime_position(pos, get_viewport()->get_window_id());
			}

			show_virtual_keyboard();
		} break;

		case NOTIFICATION_FOCUS_EXIT: {
			_validate_caret_can_draw();

			if (get_viewport()->get_window_id() != DisplayServer::INVALID_WINDOW_ID && DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_IME)) {
				DisplayServer::get_singleton()->window_set_ime_position(Point2(), get_viewport()->get_window_id());
				DisplayServer::get_singleton()->window_set_ime_active(false, get_viewport()->get_window_id());
			}
			ime_text = "";
			ime_selection = Point2();
			_shape();
			set_caret_column(caret_column); // Update scroll_offset.

			if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_VIRTUAL_KEYBOARD) && virtual_keyboard_enabled) {
				DisplayServer::get_singleton()->virtual_keyboard_hide();
			}

			if (deselect_on_focus_loss_enabled && !selection.drag_attempt) {
				deselect();
			}
		} break;

		case MainLoop::NOTIFICATION_OS_IME_UPDATE: {
			if (has_focus()) {
				ime_text = DisplayServer::get_singleton()->ime_get_text();
				ime_selection = DisplayServer::get_singleton()->ime_get_selection();

				if (!ime_text.is_empty()) {
					selection_delete();
				}

				_shape();
				set_caret_column(caret_column); // Update scroll_offset.

				queue_redraw();
			}
		} break;

		case NOTIFICATION_DRAG_BEGIN: {
			drag_action = true;
		} break;

		case NOTIFICATION_DRAG_END: {
			if (is_drag_successful()) {
				if (selection.drag_attempt) {
					selection.drag_attempt = false;
					if (is_editable() && !Input::get_singleton()->is_key_pressed(Key::CMD_OR_CTRL)) {
						selection_delete();
					} else if (deselect_on_focus_loss_enabled) {
						deselect();
					}
				}
			} else {
				selection.drag_attempt = false;
			}
			drag_action = false;
			drag_caret_force_displayed = false;
		} break;
	}
}

void LineEdit::copy_text() {
	if (selection.enabled && !pass) {
		DisplayServer::get_singleton()->clipboard_set(get_selected_text());
	}
}

void LineEdit::cut_text() {
	if (editable && selection.enabled && !pass) {
		DisplayServer::get_singleton()->clipboard_set(get_selected_text());
		selection_delete();
	}
}

void LineEdit::paste_text() {
	if (!editable) {
		return;
	}

	// Strip escape characters like \n and \t as they can't be displayed on LineEdit.
	String paste_buffer = DisplayServer::get_singleton()->clipboard_get().strip_escapes();

	if (!paste_buffer.is_empty()) {
		int prev_len = text.length();
		if (selection.enabled) {
			selection_delete();
		}
		insert_text_at_caret(paste_buffer);

		if (!text_changed_dirty) {
			if (is_inside_tree() && text.length() != prev_len) {
				MessageQueue::get_singleton()->push_call(this, "_text_changed");
			}
			text_changed_dirty = true;
		}
	}
}

bool LineEdit::has_undo() const {
	if (undo_stack_pos == nullptr) {
		return undo_stack.size() > 1;
	}
	return undo_stack_pos != undo_stack.front();
}

bool LineEdit::has_redo() const {
	return undo_stack_pos != nullptr && undo_stack_pos != undo_stack.back();
}

void LineEdit::undo() {
	if (!editable) {
		return;
	}

	if (undo_stack_pos == nullptr) {
		if (undo_stack.size() <= 1) {
			return;
		}
		undo_stack_pos = undo_stack.back();
	} else if (undo_stack_pos == undo_stack.front()) {
		return;
	}

	deselect();

	undo_stack_pos = undo_stack_pos->prev();
	TextOperation op = undo_stack_pos->get();
	text = op.text;
	scroll_offset = op.scroll_offset;
	set_caret_column(op.caret_column);

	_shape();
	_emit_text_change();
}

void LineEdit::redo() {
	if (!editable) {
		return;
	}

	if (undo_stack_pos == nullptr) {
		return;
	}
	if (undo_stack_pos == undo_stack.back()) {
		return;
	}

	deselect();

	undo_stack_pos = undo_stack_pos->next();
	TextOperation op = undo_stack_pos->get();
	text = op.text;
	scroll_offset = op.scroll_offset;
	set_caret_column(op.caret_column);

	_shape();
	_emit_text_change();
}

void LineEdit::shift_selection_check_pre(bool p_shift) {
	if (!selection.enabled && p_shift) {
		selection.start_column = caret_column;
	}
	if (!p_shift) {
		deselect();
	}
}

void LineEdit::shift_selection_check_post(bool p_shift) {
	if (p_shift) {
		selection_fill_at_caret();
	}
}

void LineEdit::set_caret_at_pixel_pos(int p_x) {
	Ref<StyleBox> style = theme_cache.normal;
	bool rtl = is_layout_rtl();

	int x_ofs = 0;
	float text_width = TS->shaped_text_get_size(text_rid).x;
	switch (alignment) {
		case HORIZONTAL_ALIGNMENT_FILL:
		case HORIZONTAL_ALIGNMENT_LEFT: {
			if (rtl) {
				x_ofs = MAX(style->get_margin(SIDE_LEFT), int(get_size().width - style->get_margin(SIDE_RIGHT) - (text_width)));
			} else {
				x_ofs = style->get_offset().x;
			}
		} break;
		case HORIZONTAL_ALIGNMENT_CENTER: {
			if (!Math::is_zero_approx(scroll_offset)) {
				x_ofs = style->get_offset().x;
			} else {
				x_ofs = MAX(style->get_margin(SIDE_LEFT), int(get_size().width - (text_width)) / 2);
			}
		} break;
		case HORIZONTAL_ALIGNMENT_RIGHT: {
			if (rtl) {
				x_ofs = style->get_offset().x;
			} else {
				x_ofs = MAX(style->get_margin(SIDE_LEFT), int(get_size().width - style->get_margin(SIDE_RIGHT) - (text_width)));
			}
		} break;
	}

	bool using_placeholder = text.is_empty() && ime_text.is_empty();
	bool display_clear_icon = !using_placeholder && is_editable() && clear_button_enabled;
	if (right_icon.is_valid() || display_clear_icon) {
		Ref<Texture2D> r_icon = display_clear_icon ? theme_cache.clear_icon : right_icon;
		if (alignment == HORIZONTAL_ALIGNMENT_CENTER) {
			if (Math::is_zero_approx(scroll_offset)) {
				x_ofs = MAX(style->get_margin(SIDE_LEFT), int(get_size().width - text_width - r_icon->get_width() - style->get_margin(SIDE_RIGHT) * 2) / 2);
			}
		} else {
			x_ofs = MAX(style->get_margin(SIDE_LEFT), x_ofs - r_icon->get_width() - style->get_margin(SIDE_RIGHT));
		}
	}

	int ofs = ceil(TS->shaped_text_hit_test_position(text_rid, p_x - x_ofs - scroll_offset));
	if (!caret_mid_grapheme_enabled) {
		ofs = TS->shaped_text_closest_character_pos(text_rid, ofs);
	}
	set_caret_column(ofs);
}

Vector2 LineEdit::get_caret_pixel_pos() {
	Ref<StyleBox> style = theme_cache.normal;
	bool rtl = is_layout_rtl();

	int x_ofs = 0;
	float text_width = TS->shaped_text_get_size(text_rid).x;
	switch (alignment) {
		case HORIZONTAL_ALIGNMENT_FILL:
		case HORIZONTAL_ALIGNMENT_LEFT: {
			if (rtl) {
				x_ofs = MAX(style->get_margin(SIDE_LEFT), int(get_size().width - style->get_margin(SIDE_RIGHT) - (text_width)));
			} else {
				x_ofs = style->get_offset().x;
			}
		} break;
		case HORIZONTAL_ALIGNMENT_CENTER: {
			if (!Math::is_zero_approx(scroll_offset)) {
				x_ofs = style->get_offset().x;
			} else {
				x_ofs = MAX(style->get_margin(SIDE_LEFT), int(get_size().width - (text_width)) / 2);
			}
		} break;
		case HORIZONTAL_ALIGNMENT_RIGHT: {
			if (rtl) {
				x_ofs = style->get_offset().x;
			} else {
				x_ofs = MAX(style->get_margin(SIDE_LEFT), int(get_size().width - style->get_margin(SIDE_RIGHT) - (text_width)));
			}
		} break;
	}

	bool using_placeholder = text.is_empty() && ime_text.is_empty();
	bool display_clear_icon = !using_placeholder && is_editable() && clear_button_enabled;
	if (right_icon.is_valid() || display_clear_icon) {
		Ref<Texture2D> r_icon = display_clear_icon ? theme_cache.clear_icon : right_icon;
		if (alignment == HORIZONTAL_ALIGNMENT_CENTER) {
			if (Math::is_zero_approx(scroll_offset)) {
				x_ofs = MAX(style->get_margin(SIDE_LEFT), int(get_size().width - text_width - r_icon->get_width() - style->get_margin(SIDE_RIGHT) * 2) / 2);
			}
		} else {
			x_ofs = MAX(style->get_margin(SIDE_LEFT), x_ofs - r_icon->get_width() - style->get_margin(SIDE_RIGHT));
		}
	}

	Vector2 ret;
	CaretInfo caret;
	// Get position of the start of caret.
	if (ime_text.length() != 0 && ime_selection.x != 0) {
		caret = TS->shaped_text_get_carets(text_rid, caret_column + ime_selection.x);
	} else {
		caret = TS->shaped_text_get_carets(text_rid, caret_column);
	}

	if ((caret.l_caret != Rect2() && (caret.l_dir == TextServer::DIRECTION_AUTO || caret.l_dir == (TextServer::Direction)input_direction)) || (caret.t_caret == Rect2())) {
		ret.x = x_ofs + caret.l_caret.position.x + scroll_offset;
	} else {
		ret.x = x_ofs + caret.t_caret.position.x + scroll_offset;
	}

	// Get position of the end of caret.
	if (ime_text.length() != 0) {
		if (ime_selection.y != 0) {
			caret = TS->shaped_text_get_carets(text_rid, caret_column + ime_selection.x + ime_selection.y);
		} else {
			caret = TS->shaped_text_get_carets(text_rid, caret_column + ime_text.size());
		}
		if ((caret.l_caret != Rect2() && (caret.l_dir == TextServer::DIRECTION_AUTO || caret.l_dir == (TextServer::Direction)input_direction)) || (caret.t_caret == Rect2())) {
			ret.y = x_ofs + caret.l_caret.position.x + scroll_offset;
		} else {
			ret.y = x_ofs + caret.t_caret.position.x + scroll_offset;
		}
	} else {
		ret.y = ret.x;
	}

	return ret;
}

void LineEdit::set_caret_mid_grapheme_enabled(const bool p_enabled) {
	caret_mid_grapheme_enabled = p_enabled;
}

bool LineEdit::is_caret_mid_grapheme_enabled() const {
	return caret_mid_grapheme_enabled;
}

bool LineEdit::is_caret_blink_enabled() const {
	return caret_blink_enabled;
}

void LineEdit::set_caret_blink_enabled(const bool p_enabled) {
	if (caret_blink_enabled == p_enabled) {
		return;
	}

	caret_blink_enabled = p_enabled;
	set_process_internal(p_enabled);

	draw_caret = !caret_blink_enabled;
	if (caret_blink_enabled) {
		caret_blink_timer = 0.0;
	}
	queue_redraw();

	notify_property_list_changed();
}

bool LineEdit::is_caret_force_displayed() const {
	return caret_force_displayed;
}

void LineEdit::set_caret_force_displayed(const bool p_enabled) {
	if (caret_force_displayed == p_enabled) {
		return;
	}

	caret_force_displayed = p_enabled;
	_validate_caret_can_draw();

	queue_redraw();
}

float LineEdit::get_caret_blink_interval() const {
	return caret_blink_interval;
}

void LineEdit::set_caret_blink_interval(const float p_interval) {
	ERR_FAIL_COND(p_interval <= 0);
	caret_blink_interval = p_interval;
}

void LineEdit::_reset_caret_blink_timer() {
	if (caret_blink_enabled) {
		draw_caret = true;
		if (caret_can_draw) {
			caret_blink_timer = 0.0;
			queue_redraw();
		}
	}
}

void LineEdit::_toggle_draw_caret() {
	draw_caret = !draw_caret;
	if (is_visible_in_tree() && caret_can_draw) {
		queue_redraw();
	}
}

void LineEdit::_validate_caret_can_draw() {
	if (caret_blink_enabled) {
		draw_caret = true;
		caret_blink_timer = 0.0;
	}
	caret_can_draw = editable && (window_has_focus || (menu && menu->has_focus())) && (has_focus() || caret_force_displayed);
}

void LineEdit::delete_char() {
	if ((text.length() <= 0) || (caret_column == 0)) {
		return;
	}

	text = text.left(caret_column - 1) + text.substr(caret_column);
	_shape();

	set_caret_column(get_caret_column() - 1);

	_text_changed();
}

void LineEdit::delete_text(int p_from_column, int p_to_column) {
	ERR_FAIL_COND_MSG(p_from_column < 0 || p_from_column > p_to_column || p_to_column > text.length(),
			vformat("Positional parameters (from: %d, to: %d) are inverted or outside the text length (%d).", p_from_column, p_to_column, text.length()));

	text = text.left(p_from_column) + text.substr(p_to_column);
	_shape();

	set_caret_column(caret_column - CLAMP(caret_column - p_from_column, 0, p_to_column - p_from_column));

	if (!text_changed_dirty) {
		if (is_inside_tree()) {
			MessageQueue::get_singleton()->push_call(this, "_text_changed");
		}
		text_changed_dirty = true;
	}
}

void LineEdit::set_text(String p_text) {
	clear_internal();
	insert_text_at_caret(p_text);
	_create_undo_state();

	queue_redraw();
	caret_column = 0;
	scroll_offset = 0.0;
}

void LineEdit::set_text_with_selection(const String &p_text) {
	Selection selection_copy = selection;

	clear_internal();
	insert_text_at_caret(p_text);
	_create_undo_state();

	int tlen = text.length();
	selection = selection_copy;
	selection.begin = MIN(selection.begin, tlen);
	selection.end = MIN(selection.end, tlen);
	selection.start_column = MIN(selection.start_column, tlen);

	queue_redraw();
}

void LineEdit::set_text_direction(Control::TextDirection p_text_direction) {
	ERR_FAIL_COND((int)p_text_direction < -1 || (int)p_text_direction > 3);
	if (text_direction != p_text_direction) {
		text_direction = p_text_direction;
		if (text_direction != TEXT_DIRECTION_AUTO && text_direction != TEXT_DIRECTION_INHERITED) {
			input_direction = text_direction;
		}
		_shape();

		if (menu_dir) {
			menu_dir->set_item_checked(menu_dir->get_item_index(MENU_DIR_INHERITED), text_direction == TEXT_DIRECTION_INHERITED);
			menu_dir->set_item_checked(menu_dir->get_item_index(MENU_DIR_AUTO), text_direction == TEXT_DIRECTION_AUTO);
			menu_dir->set_item_checked(menu_dir->get_item_index(MENU_DIR_LTR), text_direction == TEXT_DIRECTION_LTR);
			menu_dir->set_item_checked(menu_dir->get_item_index(MENU_DIR_RTL), text_direction == TEXT_DIRECTION_RTL);
		}
		queue_redraw();
	}
}

Control::TextDirection LineEdit::get_text_direction() const {
	return text_direction;
}

void LineEdit::set_language(const String &p_language) {
	if (language != p_language) {
		language = p_language;
		_shape();
		queue_redraw();
	}
}

String LineEdit::get_language() const {
	return language;
}

void LineEdit::set_draw_control_chars(bool p_draw_control_chars) {
	if (draw_control_chars != p_draw_control_chars) {
		draw_control_chars = p_draw_control_chars;
		if (menu && menu->get_item_index(MENU_DISPLAY_UCC) >= 0) {
			menu->set_item_checked(menu->get_item_index(MENU_DISPLAY_UCC), draw_control_chars);
		}
		_shape();
		queue_redraw();
	}
}

bool LineEdit::get_draw_control_chars() const {
	return draw_control_chars;
}

void LineEdit::set_structured_text_bidi_override(TextServer::StructuredTextParser p_parser) {
	if (st_parser != p_parser) {
		st_parser = p_parser;
		_shape();
		queue_redraw();
	}
}

TextServer::StructuredTextParser LineEdit::get_structured_text_bidi_override() const {
	return st_parser;
}

void LineEdit::set_structured_text_bidi_override_options(Array p_args) {
	st_args = p_args;
	_shape();
	queue_redraw();
}

Array LineEdit::get_structured_text_bidi_override_options() const {
	return st_args;
}

void LineEdit::clear() {
	clear_internal();
	_text_changed();

	// This should reset virtual keyboard state if needed.
	if (has_focus()) {
		show_virtual_keyboard();
	}
}

void LineEdit::show_virtual_keyboard() {
	if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_VIRTUAL_KEYBOARD) && virtual_keyboard_enabled) {
		if (selection.enabled) {
			DisplayServer::get_singleton()->virtual_keyboard_show(text, get_global_rect(), DisplayServer::VirtualKeyboardType(virtual_keyboard_type), max_length, selection.begin, selection.end);
		} else {
			DisplayServer::get_singleton()->virtual_keyboard_show(text, get_global_rect(), DisplayServer::VirtualKeyboardType(virtual_keyboard_type), max_length, caret_column);
		}
	}
}

String LineEdit::get_text() const {
	return text;
}

void LineEdit::set_placeholder(String p_text) {
	if (placeholder == p_text) {
		return;
	}

	placeholder = p_text;
	placeholder_translated = atr(placeholder);
	_shape();
	queue_redraw();
}

String LineEdit::get_placeholder() const {
	return placeholder;
}

void LineEdit::set_caret_column(int p_column) {
	if (p_column > (int)text.length()) {
		p_column = text.length();
	}

	if (p_column < 0) {
		p_column = 0;
	}

	caret_column = p_column;

	// Fit to window.

	if (!is_inside_tree()) {
		scroll_offset = 0.0;
		return;
	}

	Ref<StyleBox> style = theme_cache.normal;
	bool rtl = is_layout_rtl();

	int x_ofs = 0;
	float text_width = TS->shaped_text_get_size(text_rid).x;
	switch (alignment) {
		case HORIZONTAL_ALIGNMENT_FILL:
		case HORIZONTAL_ALIGNMENT_LEFT: {
			if (rtl) {
				x_ofs = MAX(style->get_margin(SIDE_LEFT), int(get_size().width - style->get_margin(SIDE_RIGHT) - (text_width)));
			} else {
				x_ofs = style->get_offset().x;
			}
		} break;
		case HORIZONTAL_ALIGNMENT_CENTER: {
			if (!Math::is_zero_approx(scroll_offset)) {
				x_ofs = style->get_offset().x;
			} else {
				x_ofs = MAX(style->get_margin(SIDE_LEFT), int(get_size().width - (text_width)) / 2);
			}
		} break;
		case HORIZONTAL_ALIGNMENT_RIGHT: {
			if (rtl) {
				x_ofs = style->get_offset().x;
			} else {
				x_ofs = MAX(style->get_margin(SIDE_LEFT), int(get_size().width - style->get_margin(SIDE_RIGHT) - (text_width)));
			}
		} break;
	}

	int ofs_max = get_size().width - style->get_margin(SIDE_RIGHT);
	bool using_placeholder = text.is_empty() && ime_text.is_empty();
	bool display_clear_icon = !using_placeholder && is_editable() && clear_button_enabled;
	if (right_icon.is_valid() || display_clear_icon) {
		Ref<Texture2D> r_icon = display_clear_icon ? theme_cache.clear_icon : right_icon;
		if (alignment == HORIZONTAL_ALIGNMENT_CENTER) {
			if (Math::is_zero_approx(scroll_offset)) {
				x_ofs = MAX(style->get_margin(SIDE_LEFT), int(get_size().width - text_width - r_icon->get_width() - style->get_margin(SIDE_RIGHT) * 2) / 2);
			}
		} else {
			x_ofs = MAX(style->get_margin(SIDE_LEFT), x_ofs - r_icon->get_width() - style->get_margin(SIDE_RIGHT));
		}
		ofs_max -= r_icon->get_width();
	}

	// Note: Use two coordinates to fit IME input range.
	Vector2 primary_caret_offset = get_caret_pixel_pos();

	if (MIN(primary_caret_offset.x, primary_caret_offset.y) <= x_ofs) {
		scroll_offset += x_ofs - MIN(primary_caret_offset.x, primary_caret_offset.y);
	} else if (MAX(primary_caret_offset.x, primary_caret_offset.y) >= ofs_max) {
		scroll_offset += ofs_max - MAX(primary_caret_offset.x, primary_caret_offset.y);
	}
	scroll_offset = MIN(0, scroll_offset);

	queue_redraw();
}

int LineEdit::get_caret_column() const {
	return caret_column;
}

void LineEdit::set_scroll_offset(float p_pos) {
	scroll_offset = p_pos;
	if (scroll_offset < 0.0) {
		scroll_offset = 0.0;
	}
}

float LineEdit::get_scroll_offset() const {
	return scroll_offset;
}

void LineEdit::insert_text_at_caret(String p_text) {
	if (max_length > 0) {
		// Truncate text to append to fit in max_length, if needed.
		int available_chars = max_length - text.length();
		if (p_text.length() > available_chars) {
			emit_signal(SNAME("text_change_rejected"), p_text.substr(available_chars));
			p_text = p_text.substr(0, available_chars);
		}
	}
	String pre = text.substr(0, caret_column);
	String post = text.substr(caret_column, text.length() - caret_column);
	text = pre + p_text + post;
	_shape();
	TextServer::Direction dir = TS->shaped_text_get_dominant_direction_in_range(text_rid, caret_column, caret_column + p_text.length());
	if (dir != TextServer::DIRECTION_AUTO) {
		input_direction = (TextDirection)dir;
	}
	set_caret_column(caret_column + p_text.length());

	if (!ime_text.is_empty()) {
		_shape();
	}
}

void LineEdit::clear_internal() {
	deselect();
	_clear_undo_stack();
	caret_column = 0;
	scroll_offset = 0.0;
	undo_text = "";
	text = "";
	_shape();
	queue_redraw();
}

Size2 LineEdit::get_minimum_size() const {
	Ref<StyleBox> style = theme_cache.normal;
	Ref<Font> font = theme_cache.font;
	int font_size = theme_cache.font_size;

	Size2 min_size;

	// Minimum size of text.
	float em_space_size = font->get_char_size('M', font_size).x;
	min_size.width = theme_cache.minimum_character_width * em_space_size;

	if (expand_to_text_length) {
		// Ensure some space for the caret when placed at the end.
		min_size.width = MAX(min_size.width, full_width + theme_cache.caret_width);
	}

	min_size.height = MAX(TS->shaped_text_get_size(text_rid).y, font->get_height(font_size));

	// Take icons into account.
	int icon_max_width = 0;
	if (right_icon.is_valid()) {
		min_size.height = MAX(min_size.height, right_icon->get_height());
		icon_max_width = right_icon->get_width();
	}
	if (clear_button_enabled) {
		min_size.height = MAX(min_size.height, theme_cache.clear_icon->get_height());
		icon_max_width = MAX(icon_max_width, theme_cache.clear_icon->get_width());
	}
	min_size.width += icon_max_width;

	return style->get_minimum_size() + min_size;
}

void LineEdit::deselect() {
	selection.begin = 0;
	selection.end = 0;
	selection.start_column = 0;
	selection.enabled = false;
	selection.creating = false;
	selection.double_click = false;
	queue_redraw();
}

bool LineEdit::has_selection() const {
	return selection.enabled;
}

String LineEdit::get_selected_text() {
	if (selection.enabled) {
		return text.substr(selection.begin, selection.end - selection.begin);
	} else {
		return String();
	}
}

int LineEdit::get_selection_from_column() const {
	ERR_FAIL_COND_V(!selection.enabled, -1);
	return selection.begin;
}

int LineEdit::get_selection_to_column() const {
	ERR_FAIL_COND_V(!selection.enabled, -1);
	return selection.end;
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

void LineEdit::selection_fill_at_caret() {
	if (!selecting_enabled) {
		return;
	}

	selection.begin = caret_column;
	selection.end = selection.start_column;

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
	queue_redraw();
}

void LineEdit::set_editable(bool p_editable) {
	if (editable == p_editable) {
		return;
	}

	editable = p_editable;
	_validate_caret_can_draw();

	update_minimum_size();
	queue_redraw();
}

bool LineEdit::is_editable() const {
	return editable;
}

void LineEdit::set_secret(bool p_secret) {
	if (pass == p_secret) {
		return;
	}

	pass = p_secret;
	_shape();
	queue_redraw();
}

bool LineEdit::is_secret() const {
	return pass;
}

void LineEdit::set_secret_character(const String &p_string) {
	if (secret_character == p_string) {
		return;
	}

	secret_character = p_string;
	update_configuration_warnings();
	_shape();
	queue_redraw();
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
	selection.double_click = false;
	queue_redraw();
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
				insert_text_at_caret(String::chr(0x200E));
			}
		} break;
		case MENU_INSERT_RLM: {
			if (editable) {
				insert_text_at_caret(String::chr(0x200F));
			}
		} break;
		case MENU_INSERT_LRE: {
			if (editable) {
				insert_text_at_caret(String::chr(0x202A));
			}
		} break;
		case MENU_INSERT_RLE: {
			if (editable) {
				insert_text_at_caret(String::chr(0x202B));
			}
		} break;
		case MENU_INSERT_LRO: {
			if (editable) {
				insert_text_at_caret(String::chr(0x202D));
			}
		} break;
		case MENU_INSERT_RLO: {
			if (editable) {
				insert_text_at_caret(String::chr(0x202E));
			}
		} break;
		case MENU_INSERT_PDF: {
			if (editable) {
				insert_text_at_caret(String::chr(0x202C));
			}
		} break;
		case MENU_INSERT_ALM: {
			if (editable) {
				insert_text_at_caret(String::chr(0x061C));
			}
		} break;
		case MENU_INSERT_LRI: {
			if (editable) {
				insert_text_at_caret(String::chr(0x2066));
			}
		} break;
		case MENU_INSERT_RLI: {
			if (editable) {
				insert_text_at_caret(String::chr(0x2067));
			}
		} break;
		case MENU_INSERT_FSI: {
			if (editable) {
				insert_text_at_caret(String::chr(0x2068));
			}
		} break;
		case MENU_INSERT_PDI: {
			if (editable) {
				insert_text_at_caret(String::chr(0x2069));
			}
		} break;
		case MENU_INSERT_ZWJ: {
			if (editable) {
				insert_text_at_caret(String::chr(0x200D));
			}
		} break;
		case MENU_INSERT_ZWNJ: {
			if (editable) {
				insert_text_at_caret(String::chr(0x200C));
			}
		} break;
		case MENU_INSERT_WJ: {
			if (editable) {
				insert_text_at_caret(String::chr(0x2060));
			}
		} break;
		case MENU_INSERT_SHY: {
			if (editable) {
				insert_text_at_caret(String::chr(0x00AD));
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

bool LineEdit::is_menu_visible() const {
	return menu && menu->is_visible();
}

PopupMenu *LineEdit::get_menu() const {
	if (!menu) {
		const_cast<LineEdit *>(this)->_generate_context_menu();
	}
	return menu;
}

void LineEdit::_editor_settings_changed() {
#ifdef TOOLS_ENABLED
	set_caret_blink_enabled(EDITOR_GET("text_editor/appearance/caret/caret_blink"));
	set_caret_blink_interval(EDITOR_GET("text_editor/appearance/caret/caret_blink_interval"));
#endif
}

void LineEdit::set_expand_to_text_length_enabled(bool p_enabled) {
	expand_to_text_length = p_enabled;
	update_minimum_size();
	set_caret_column(caret_column);
}

bool LineEdit::is_expand_to_text_length_enabled() const {
	return expand_to_text_length;
}

void LineEdit::set_clear_button_enabled(bool p_enabled) {
	if (clear_button_enabled == p_enabled) {
		return;
	}
	clear_button_enabled = p_enabled;
	_fit_to_width();
	update_minimum_size();
	queue_redraw();
}

bool LineEdit::is_clear_button_enabled() const {
	return clear_button_enabled;
}

void LineEdit::set_shortcut_keys_enabled(bool p_enabled) {
	shortcut_keys_enabled = p_enabled;
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

void LineEdit::set_virtual_keyboard_type(VirtualKeyboardType p_type) {
	virtual_keyboard_type = p_type;
}

LineEdit::VirtualKeyboardType LineEdit::get_virtual_keyboard_type() const {
	return virtual_keyboard_type;
}

void LineEdit::set_middle_mouse_paste_enabled(bool p_enabled) {
	middle_mouse_paste_enabled = p_enabled;
}

bool LineEdit::is_middle_mouse_paste_enabled() const {
	return middle_mouse_paste_enabled;
}

void LineEdit::set_selecting_enabled(bool p_enabled) {
	if (selecting_enabled == p_enabled) {
		return;
	}

	selecting_enabled = p_enabled;

	if (!selecting_enabled) {
		deselect();
	}
}

bool LineEdit::is_selecting_enabled() const {
	return selecting_enabled;
}

void LineEdit::set_deselect_on_focus_loss_enabled(const bool p_enabled) {
	if (deselect_on_focus_loss_enabled == p_enabled) {
		return;
	}

	deselect_on_focus_loss_enabled = p_enabled;
	if (p_enabled && selection.enabled && !has_focus()) {
		deselect();
	}
}

bool LineEdit::is_deselect_on_focus_loss_enabled() const {
	return deselect_on_focus_loss_enabled;
}

void LineEdit::set_drag_and_drop_selection_enabled(const bool p_enabled) {
	drag_and_drop_selection_enabled = p_enabled;
}

bool LineEdit::is_drag_and_drop_selection_enabled() const {
	return drag_and_drop_selection_enabled;
}

void LineEdit::set_right_icon(const Ref<Texture2D> &p_icon) {
	if (right_icon == p_icon) {
		return;
	}
	right_icon = p_icon;
	_fit_to_width();
	update_minimum_size();
	queue_redraw();
}

Ref<Texture2D> LineEdit::get_right_icon() {
	return right_icon;
}

void LineEdit::set_flat(bool p_enabled) {
	if (flat != p_enabled) {
		flat = p_enabled;
		queue_redraw();
	}
}

bool LineEdit::is_flat() const {
	return flat;
}

void LineEdit::set_select_all_on_focus(bool p_enabled) {
	select_all_on_focus = p_enabled;
}

bool LineEdit::is_select_all_on_focus() const {
	return select_all_on_focus;
}

void LineEdit::clear_pending_select_all_on_focus() {
	pending_select_all_on_focus = false;
}

void LineEdit::_text_changed() {
	_emit_text_change();
	_clear_redo();
}

void LineEdit::_emit_text_change() {
	emit_signal(SNAME("text_changed"), text);
	text_changed_dirty = false;
}
PackedStringArray LineEdit::get_configuration_warnings() const {
	PackedStringArray warnings = Control::get_configuration_warnings();
	if (secret_character.length() > 1) {
		warnings.push_back("Secret Character property supports only one character. Extra characters will be ignored.");
	}
	return warnings;
}

void LineEdit::_shape() {
	const Ref<Font> &font = theme_cache.font;
	int font_size = theme_cache.font_size;
	if (font.is_null()) {
		return;
	}

	Size2 old_size = TS->shaped_text_get_size(text_rid);
	TS->shaped_text_clear(text_rid);

	String t;
	if (text.length() == 0 && ime_text.length() == 0) {
		t = placeholder_translated;
	} else if (pass) {
		// TODO: Integrate with text server to add support for non-latin scripts.
		// Allow secret_character as empty strings, act like if a space was used as a secret character.
		String secret = " ";
		// Allow values longer than 1 character in the property, but trim characters after the first one.
		if (!secret_character.is_empty()) {
			secret = secret_character.left(1);
		}
		t = secret.repeat(text.length() + ime_text.length());
	} else {
		if (ime_text.length() > 0) {
			t = text.substr(0, caret_column) + ime_text + text.substr(caret_column, text.length());
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

	TS->shaped_text_add_string(text_rid, t, font->get_rids(), font_size, font->get_opentype_features(), language);
	TS->shaped_text_set_bidi_override(text_rid, structured_text_parser(st_parser, st_args, t));

	full_width = TS->shaped_text_get_size(text_rid).x;
	_fit_to_width();

	Size2 size = TS->shaped_text_get_size(text_rid);

	if ((expand_to_text_length && old_size.x != size.x) || (old_size.y != size.y)) {
		update_minimum_size();
	}
}

void LineEdit::_fit_to_width() {
	if (alignment == HORIZONTAL_ALIGNMENT_FILL) {
		Ref<StyleBox> style = theme_cache.normal;
		int t_width = get_size().width - style->get_margin(SIDE_RIGHT) - style->get_margin(SIDE_LEFT);
		bool using_placeholder = text.is_empty() && ime_text.is_empty();
		bool display_clear_icon = !using_placeholder && is_editable() && clear_button_enabled;
		if (right_icon.is_valid() || display_clear_icon) {
			Ref<Texture2D> r_icon = display_clear_icon ? theme_cache.clear_icon : right_icon;
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
	op.caret_column = caret_column;
	op.scroll_offset = scroll_offset;
	undo_stack.push_back(op);
}

Key LineEdit::_get_menu_action_accelerator(const String &p_action) {
	const List<Ref<InputEvent>> *events = InputMap::get_singleton()->action_get_events(p_action);
	if (!events) {
		return Key::NONE;
	}

	// Use first event in the list for the accelerator.
	const List<Ref<InputEvent>>::Element *first_event = events->front();
	if (!first_event) {
		return Key::NONE;
	}

	const Ref<InputEventKey> event = first_event->get();
	if (event.is_null()) {
		return Key::NONE;
	}

	// Use physical keycode if non-zero.
	if (event->get_physical_keycode() != Key::NONE) {
		return event->get_physical_keycode_with_modifiers();
	} else {
		return event->get_keycode_with_modifiers();
	}
}

void LineEdit::_generate_context_menu() {
	menu = memnew(PopupMenu);
	add_child(menu, false, INTERNAL_MODE_FRONT);

	menu_dir = memnew(PopupMenu);
	menu_dir->set_name("DirMenu");
	menu_dir->add_radio_check_item(RTR("Same as Layout Direction"), MENU_DIR_INHERITED);
	menu_dir->add_radio_check_item(RTR("Auto-Detect Direction"), MENU_DIR_AUTO);
	menu_dir->add_radio_check_item(RTR("Left-to-Right"), MENU_DIR_LTR);
	menu_dir->add_radio_check_item(RTR("Right-to-Left"), MENU_DIR_RTL);
	menu->add_child(menu_dir, false, INTERNAL_MODE_FRONT);

	menu_ctl = memnew(PopupMenu);
	menu_ctl->set_name("CTLMenu");
	menu_ctl->add_item(RTR("Left-to-Right Mark (LRM)"), MENU_INSERT_LRM);
	menu_ctl->add_item(RTR("Right-to-Left Mark (RLM)"), MENU_INSERT_RLM);
	menu_ctl->add_item(RTR("Start of Left-to-Right Embedding (LRE)"), MENU_INSERT_LRE);
	menu_ctl->add_item(RTR("Start of Right-to-Left Embedding (RLE)"), MENU_INSERT_RLE);
	menu_ctl->add_item(RTR("Start of Left-to-Right Override (LRO)"), MENU_INSERT_LRO);
	menu_ctl->add_item(RTR("Start of Right-to-Left Override (RLO)"), MENU_INSERT_RLO);
	menu_ctl->add_item(RTR("Pop Direction Formatting (PDF)"), MENU_INSERT_PDF);
	menu_ctl->add_separator();
	menu_ctl->add_item(RTR("Arabic Letter Mark (ALM)"), MENU_INSERT_ALM);
	menu_ctl->add_item(RTR("Left-to-Right Isolate (LRI)"), MENU_INSERT_LRI);
	menu_ctl->add_item(RTR("Right-to-Left Isolate (RLI)"), MENU_INSERT_RLI);
	menu_ctl->add_item(RTR("First Strong Isolate (FSI)"), MENU_INSERT_FSI);
	menu_ctl->add_item(RTR("Pop Direction Isolate (PDI)"), MENU_INSERT_PDI);
	menu_ctl->add_separator();
	menu_ctl->add_item(RTR("Zero-Width Joiner (ZWJ)"), MENU_INSERT_ZWJ);
	menu_ctl->add_item(RTR("Zero-Width Non-Joiner (ZWNJ)"), MENU_INSERT_ZWNJ);
	menu_ctl->add_item(RTR("Word Joiner (WJ)"), MENU_INSERT_WJ);
	menu_ctl->add_item(RTR("Soft Hyphen (SHY)"), MENU_INSERT_SHY);
	menu->add_child(menu_ctl, false, INTERNAL_MODE_FRONT);

	menu->add_item(RTR("Cut"), MENU_CUT);
	menu->add_item(RTR("Copy"), MENU_COPY);
	menu->add_item(RTR("Paste"), MENU_PASTE);
	menu->add_separator();
	menu->add_item(RTR("Select All"), MENU_SELECT_ALL);
	menu->add_item(RTR("Clear"), MENU_CLEAR);
	menu->add_separator();
	menu->add_item(RTR("Undo"), MENU_UNDO);
	menu->add_item(RTR("Redo"), MENU_REDO);
	menu->add_separator();
	menu->add_submenu_item(RTR("Text Writing Direction"), "DirMenu", MENU_SUBMENU_TEXT_DIR);
	menu->add_separator();
	menu->add_check_item(RTR("Display Control Characters"), MENU_DISPLAY_UCC);
	menu->add_submenu_item(RTR("Insert Control Character"), "CTLMenu", MENU_SUBMENU_INSERT_UCC);

	menu->connect("id_pressed", callable_mp(this, &LineEdit::menu_option));
	menu_dir->connect("id_pressed", callable_mp(this, &LineEdit::menu_option));
	menu_ctl->connect("id_pressed", callable_mp(this, &LineEdit::menu_option));

	menu->connect(SNAME("focus_entered"), callable_mp(this, &LineEdit::_validate_caret_can_draw));
	menu->connect(SNAME("focus_exited"), callable_mp(this, &LineEdit::_validate_caret_can_draw));
}

void LineEdit::_update_context_menu() {
	if (!menu) {
		_generate_context_menu();
	}

	int idx = -1;

#define MENU_ITEM_ACTION_DISABLED(m_menu, m_id, m_action, m_disabled)                                                  \
	idx = m_menu->get_item_index(m_id);                                                                                \
	if (idx >= 0) {                                                                                                    \
		m_menu->set_item_accelerator(idx, shortcut_keys_enabled ? _get_menu_action_accelerator(m_action) : Key::NONE); \
		m_menu->set_item_disabled(idx, m_disabled);                                                                    \
	}

#define MENU_ITEM_ACTION(m_menu, m_id, m_action)                                                                       \
	idx = m_menu->get_item_index(m_id);                                                                                \
	if (idx >= 0) {                                                                                                    \
		m_menu->set_item_accelerator(idx, shortcut_keys_enabled ? _get_menu_action_accelerator(m_action) : Key::NONE); \
	}

#define MENU_ITEM_DISABLED(m_menu, m_id, m_disabled) \
	idx = m_menu->get_item_index(m_id);              \
	if (idx >= 0) {                                  \
		m_menu->set_item_disabled(idx, m_disabled);  \
	}

#define MENU_ITEM_CHECKED(m_menu, m_id, m_checked) \
	idx = m_menu->get_item_index(m_id);            \
	if (idx >= 0) {                                \
		m_menu->set_item_checked(idx, m_checked);  \
	}

	MENU_ITEM_ACTION_DISABLED(menu, MENU_CUT, "ui_cut", !editable)
	MENU_ITEM_ACTION(menu, MENU_COPY, "ui_copy")
	MENU_ITEM_ACTION_DISABLED(menu, MENU_PASTE, "ui_paste", !editable)
	MENU_ITEM_ACTION_DISABLED(menu, MENU_SELECT_ALL, "ui_text_select_all", !selecting_enabled)
	MENU_ITEM_DISABLED(menu, MENU_CLEAR, !editable)
	MENU_ITEM_ACTION_DISABLED(menu, MENU_UNDO, "ui_undo", !editable || !has_undo())
	MENU_ITEM_ACTION_DISABLED(menu, MENU_REDO, "ui_redo", !editable || !has_redo())
	MENU_ITEM_CHECKED(menu_dir, MENU_DIR_INHERITED, text_direction == TEXT_DIRECTION_INHERITED)
	MENU_ITEM_CHECKED(menu_dir, MENU_DIR_AUTO, text_direction == TEXT_DIRECTION_AUTO)
	MENU_ITEM_CHECKED(menu_dir, MENU_DIR_LTR, text_direction == TEXT_DIRECTION_LTR)
	MENU_ITEM_CHECKED(menu_dir, MENU_DIR_RTL, text_direction == TEXT_DIRECTION_RTL)
	MENU_ITEM_CHECKED(menu, MENU_DISPLAY_UCC, draw_control_chars)
	MENU_ITEM_DISABLED(menu, MENU_SUBMENU_INSERT_UCC, !editable)

#undef MENU_ITEM_ACTION_DISABLED
#undef MENU_ITEM_ACTION
#undef MENU_ITEM_DISABLED
#undef MENU_ITEM_CHECKED
}

void LineEdit::_validate_property(PropertyInfo &p_property) const {
	if (!caret_blink_enabled && p_property.name == "caret_blink_interval") {
		p_property.usage = PROPERTY_USAGE_NO_EDITOR;
	}
}

void LineEdit::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_text_changed"), &LineEdit::_text_changed);

	ClassDB::bind_method(D_METHOD("set_horizontal_alignment", "alignment"), &LineEdit::set_horizontal_alignment);
	ClassDB::bind_method(D_METHOD("get_horizontal_alignment"), &LineEdit::get_horizontal_alignment);

	ClassDB::bind_method(D_METHOD("clear"), &LineEdit::clear);
	ClassDB::bind_method(D_METHOD("select", "from", "to"), &LineEdit::select, DEFVAL(0), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("select_all"), &LineEdit::select_all);
	ClassDB::bind_method(D_METHOD("deselect"), &LineEdit::deselect);
	ClassDB::bind_method(D_METHOD("has_selection"), &LineEdit::has_selection);
	ClassDB::bind_method(D_METHOD("get_selected_text"), &LineEdit::get_selected_text);
	ClassDB::bind_method(D_METHOD("get_selection_from_column"), &LineEdit::get_selection_from_column);
	ClassDB::bind_method(D_METHOD("get_selection_to_column"), &LineEdit::get_selection_to_column);
	ClassDB::bind_method(D_METHOD("set_text", "text"), &LineEdit::set_text);
	ClassDB::bind_method(D_METHOD("get_text"), &LineEdit::get_text);
	ClassDB::bind_method(D_METHOD("get_draw_control_chars"), &LineEdit::get_draw_control_chars);
	ClassDB::bind_method(D_METHOD("set_draw_control_chars", "enable"), &LineEdit::set_draw_control_chars);
	ClassDB::bind_method(D_METHOD("set_text_direction", "direction"), &LineEdit::set_text_direction);
	ClassDB::bind_method(D_METHOD("get_text_direction"), &LineEdit::get_text_direction);
	ClassDB::bind_method(D_METHOD("set_language", "language"), &LineEdit::set_language);
	ClassDB::bind_method(D_METHOD("get_language"), &LineEdit::get_language);
	ClassDB::bind_method(D_METHOD("set_structured_text_bidi_override", "parser"), &LineEdit::set_structured_text_bidi_override);
	ClassDB::bind_method(D_METHOD("get_structured_text_bidi_override"), &LineEdit::get_structured_text_bidi_override);
	ClassDB::bind_method(D_METHOD("set_structured_text_bidi_override_options", "args"), &LineEdit::set_structured_text_bidi_override_options);
	ClassDB::bind_method(D_METHOD("get_structured_text_bidi_override_options"), &LineEdit::get_structured_text_bidi_override_options);
	ClassDB::bind_method(D_METHOD("set_placeholder", "text"), &LineEdit::set_placeholder);
	ClassDB::bind_method(D_METHOD("get_placeholder"), &LineEdit::get_placeholder);
	ClassDB::bind_method(D_METHOD("set_caret_column", "position"), &LineEdit::set_caret_column);
	ClassDB::bind_method(D_METHOD("get_caret_column"), &LineEdit::get_caret_column);
	ClassDB::bind_method(D_METHOD("get_scroll_offset"), &LineEdit::get_scroll_offset);
	ClassDB::bind_method(D_METHOD("set_expand_to_text_length_enabled", "enabled"), &LineEdit::set_expand_to_text_length_enabled);
	ClassDB::bind_method(D_METHOD("is_expand_to_text_length_enabled"), &LineEdit::is_expand_to_text_length_enabled);
	ClassDB::bind_method(D_METHOD("set_caret_blink_enabled", "enabled"), &LineEdit::set_caret_blink_enabled);
	ClassDB::bind_method(D_METHOD("is_caret_blink_enabled"), &LineEdit::is_caret_blink_enabled);
	ClassDB::bind_method(D_METHOD("set_caret_mid_grapheme_enabled", "enabled"), &LineEdit::set_caret_mid_grapheme_enabled);
	ClassDB::bind_method(D_METHOD("is_caret_mid_grapheme_enabled"), &LineEdit::is_caret_mid_grapheme_enabled);
	ClassDB::bind_method(D_METHOD("set_caret_force_displayed", "enabled"), &LineEdit::set_caret_force_displayed);
	ClassDB::bind_method(D_METHOD("is_caret_force_displayed"), &LineEdit::is_caret_force_displayed);
	ClassDB::bind_method(D_METHOD("set_caret_blink_interval", "interval"), &LineEdit::set_caret_blink_interval);
	ClassDB::bind_method(D_METHOD("get_caret_blink_interval"), &LineEdit::get_caret_blink_interval);
	ClassDB::bind_method(D_METHOD("set_max_length", "chars"), &LineEdit::set_max_length);
	ClassDB::bind_method(D_METHOD("get_max_length"), &LineEdit::get_max_length);
	ClassDB::bind_method(D_METHOD("insert_text_at_caret", "text"), &LineEdit::insert_text_at_caret);
	ClassDB::bind_method(D_METHOD("delete_char_at_caret"), &LineEdit::delete_char);
	ClassDB::bind_method(D_METHOD("delete_text", "from_column", "to_column"), &LineEdit::delete_text);
	ClassDB::bind_method(D_METHOD("set_editable", "enabled"), &LineEdit::set_editable);
	ClassDB::bind_method(D_METHOD("is_editable"), &LineEdit::is_editable);
	ClassDB::bind_method(D_METHOD("set_secret", "enabled"), &LineEdit::set_secret);
	ClassDB::bind_method(D_METHOD("is_secret"), &LineEdit::is_secret);
	ClassDB::bind_method(D_METHOD("set_secret_character", "character"), &LineEdit::set_secret_character);
	ClassDB::bind_method(D_METHOD("get_secret_character"), &LineEdit::get_secret_character);
	ClassDB::bind_method(D_METHOD("menu_option", "option"), &LineEdit::menu_option);
	ClassDB::bind_method(D_METHOD("get_menu"), &LineEdit::get_menu);
	ClassDB::bind_method(D_METHOD("is_menu_visible"), &LineEdit::is_menu_visible);
	ClassDB::bind_method(D_METHOD("set_context_menu_enabled", "enable"), &LineEdit::set_context_menu_enabled);
	ClassDB::bind_method(D_METHOD("is_context_menu_enabled"), &LineEdit::is_context_menu_enabled);
	ClassDB::bind_method(D_METHOD("set_virtual_keyboard_enabled", "enable"), &LineEdit::set_virtual_keyboard_enabled);
	ClassDB::bind_method(D_METHOD("is_virtual_keyboard_enabled"), &LineEdit::is_virtual_keyboard_enabled);
	ClassDB::bind_method(D_METHOD("set_virtual_keyboard_type", "type"), &LineEdit::set_virtual_keyboard_type);
	ClassDB::bind_method(D_METHOD("get_virtual_keyboard_type"), &LineEdit::get_virtual_keyboard_type);
	ClassDB::bind_method(D_METHOD("set_clear_button_enabled", "enable"), &LineEdit::set_clear_button_enabled);
	ClassDB::bind_method(D_METHOD("is_clear_button_enabled"), &LineEdit::is_clear_button_enabled);
	ClassDB::bind_method(D_METHOD("set_shortcut_keys_enabled", "enable"), &LineEdit::set_shortcut_keys_enabled);
	ClassDB::bind_method(D_METHOD("is_shortcut_keys_enabled"), &LineEdit::is_shortcut_keys_enabled);
	ClassDB::bind_method(D_METHOD("set_middle_mouse_paste_enabled", "enable"), &LineEdit::set_middle_mouse_paste_enabled);
	ClassDB::bind_method(D_METHOD("is_middle_mouse_paste_enabled"), &LineEdit::is_middle_mouse_paste_enabled);
	ClassDB::bind_method(D_METHOD("set_selecting_enabled", "enable"), &LineEdit::set_selecting_enabled);
	ClassDB::bind_method(D_METHOD("is_selecting_enabled"), &LineEdit::is_selecting_enabled);
	ClassDB::bind_method(D_METHOD("set_deselect_on_focus_loss_enabled", "enable"), &LineEdit::set_deselect_on_focus_loss_enabled);
	ClassDB::bind_method(D_METHOD("is_deselect_on_focus_loss_enabled"), &LineEdit::is_deselect_on_focus_loss_enabled);
	ClassDB::bind_method(D_METHOD("set_drag_and_drop_selection_enabled", "enable"), &LineEdit::set_drag_and_drop_selection_enabled);
	ClassDB::bind_method(D_METHOD("is_drag_and_drop_selection_enabled"), &LineEdit::is_drag_and_drop_selection_enabled);
	ClassDB::bind_method(D_METHOD("set_right_icon", "icon"), &LineEdit::set_right_icon);
	ClassDB::bind_method(D_METHOD("get_right_icon"), &LineEdit::get_right_icon);
	ClassDB::bind_method(D_METHOD("set_flat", "enabled"), &LineEdit::set_flat);
	ClassDB::bind_method(D_METHOD("is_flat"), &LineEdit::is_flat);
	ClassDB::bind_method(D_METHOD("set_select_all_on_focus", "enabled"), &LineEdit::set_select_all_on_focus);
	ClassDB::bind_method(D_METHOD("is_select_all_on_focus"), &LineEdit::is_select_all_on_focus);

	ADD_SIGNAL(MethodInfo("text_changed", PropertyInfo(Variant::STRING, "new_text")));
	ADD_SIGNAL(MethodInfo("text_change_rejected", PropertyInfo(Variant::STRING, "rejected_substring")));
	ADD_SIGNAL(MethodInfo("text_submitted", PropertyInfo(Variant::STRING, "new_text")));

	BIND_ENUM_CONSTANT(MENU_CUT);
	BIND_ENUM_CONSTANT(MENU_COPY);
	BIND_ENUM_CONSTANT(MENU_PASTE);
	BIND_ENUM_CONSTANT(MENU_CLEAR);
	BIND_ENUM_CONSTANT(MENU_SELECT_ALL);
	BIND_ENUM_CONSTANT(MENU_UNDO);
	BIND_ENUM_CONSTANT(MENU_REDO);
	BIND_ENUM_CONSTANT(MENU_SUBMENU_TEXT_DIR);
	BIND_ENUM_CONSTANT(MENU_DIR_INHERITED);
	BIND_ENUM_CONSTANT(MENU_DIR_AUTO);
	BIND_ENUM_CONSTANT(MENU_DIR_LTR);
	BIND_ENUM_CONSTANT(MENU_DIR_RTL);
	BIND_ENUM_CONSTANT(MENU_DISPLAY_UCC);
	BIND_ENUM_CONSTANT(MENU_SUBMENU_INSERT_UCC);
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

	BIND_ENUM_CONSTANT(KEYBOARD_TYPE_DEFAULT);
	BIND_ENUM_CONSTANT(KEYBOARD_TYPE_MULTILINE);
	BIND_ENUM_CONSTANT(KEYBOARD_TYPE_NUMBER);
	BIND_ENUM_CONSTANT(KEYBOARD_TYPE_NUMBER_DECIMAL);
	BIND_ENUM_CONSTANT(KEYBOARD_TYPE_PHONE);
	BIND_ENUM_CONSTANT(KEYBOARD_TYPE_EMAIL_ADDRESS);
	BIND_ENUM_CONSTANT(KEYBOARD_TYPE_PASSWORD);
	BIND_ENUM_CONSTANT(KEYBOARD_TYPE_URL);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "text"), "set_text", "get_text");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "placeholder_text"), "set_placeholder", "get_placeholder");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "alignment", PROPERTY_HINT_ENUM, "Left,Center,Right,Fill"), "set_horizontal_alignment", "get_horizontal_alignment");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_length", PROPERTY_HINT_RANGE, "0,1000,1,or_greater"), "set_max_length", "get_max_length");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editable"), "set_editable", "is_editable");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "expand_to_text_length"), "set_expand_to_text_length_enabled", "is_expand_to_text_length_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "context_menu_enabled"), "set_context_menu_enabled", "is_context_menu_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "virtual_keyboard_enabled"), "set_virtual_keyboard_enabled", "is_virtual_keyboard_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "virtual_keyboard_type", PROPERTY_HINT_ENUM, "Default,Multiline,Number,Decimal,Phone,Email,Password,URL"), "set_virtual_keyboard_type", "get_virtual_keyboard_type");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "clear_button_enabled"), "set_clear_button_enabled", "is_clear_button_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "shortcut_keys_enabled"), "set_shortcut_keys_enabled", "is_shortcut_keys_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "middle_mouse_paste_enabled"), "set_middle_mouse_paste_enabled", "is_middle_mouse_paste_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "selecting_enabled"), "set_selecting_enabled", "is_selecting_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "deselect_on_focus_loss_enabled"), "set_deselect_on_focus_loss_enabled", "is_deselect_on_focus_loss_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "drag_and_drop_selection_enabled"), "set_drag_and_drop_selection_enabled", "is_drag_and_drop_selection_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "right_icon", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_right_icon", "get_right_icon");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flat"), "set_flat", "is_flat");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_control_chars"), "set_draw_control_chars", "get_draw_control_chars");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "select_all_on_focus"), "set_select_all_on_focus", "is_select_all_on_focus");

	ADD_GROUP("Caret", "caret_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "caret_blink"), "set_caret_blink_enabled", "is_caret_blink_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "caret_blink_interval", PROPERTY_HINT_RANGE, "0.1,10,0.01"), "set_caret_blink_interval", "get_caret_blink_interval");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "caret_column", PROPERTY_HINT_RANGE, "0,1000,1,or_greater"), "set_caret_column", "get_caret_column");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "caret_force_displayed"), "set_caret_force_displayed", "is_caret_force_displayed");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "caret_mid_grapheme"), "set_caret_mid_grapheme_enabled", "is_caret_mid_grapheme_enabled");

	ADD_GROUP("Secret", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "secret"), "set_secret", "is_secret");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "secret_character"), "set_secret_character", "get_secret_character");

	ADD_GROUP("BiDi", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "text_direction", PROPERTY_HINT_ENUM, "Auto,Left-to-Right,Right-to-Left,Inherited"), "set_text_direction", "get_text_direction");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "language", PROPERTY_HINT_LOCALE_ID, ""), "set_language", "get_language");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "structured_text_bidi_override", PROPERTY_HINT_ENUM, "Default,URI,File,Email,List,None,Custom"), "set_structured_text_bidi_override", "get_structured_text_bidi_override");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "structured_text_bidi_override_options"), "set_structured_text_bidi_override_options", "get_structured_text_bidi_override_options");

	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, LineEdit, normal);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, LineEdit, read_only);
	BIND_THEME_ITEM(Theme::DATA_TYPE_STYLEBOX, LineEdit, focus);

	BIND_THEME_ITEM(Theme::DATA_TYPE_FONT, LineEdit, font);
	BIND_THEME_ITEM(Theme::DATA_TYPE_FONT_SIZE, LineEdit, font_size);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, LineEdit, font_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, LineEdit, font_uneditable_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, LineEdit, font_selected_color);
	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_CONSTANT, LineEdit, font_outline_size, "outline_size");
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, LineEdit, font_outline_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, LineEdit, font_placeholder_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, LineEdit, caret_width);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, LineEdit, caret_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_CONSTANT, LineEdit, minimum_character_width);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, LineEdit, selection_color);

	BIND_THEME_ITEM_CUSTOM(Theme::DATA_TYPE_ICON, LineEdit, clear_icon, "clear");
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, LineEdit, clear_button_color);
	BIND_THEME_ITEM(Theme::DATA_TYPE_COLOR, LineEdit, clear_button_color_pressed);
}

LineEdit::LineEdit(const String &p_placeholder) {
	text_rid = TS->create_shaped_text();
	_create_undo_state();

	deselect();
	set_focus_mode(FOCUS_ALL);
	set_default_cursor_shape(CURSOR_IBEAM);
	set_mouse_filter(MOUSE_FILTER_STOP);
	set_process_unhandled_key_input(true);

	set_caret_blink_enabled(false);

	set_placeholder(p_placeholder);

	set_editable(true); // Initialize to opposite first, so we get past the early-out in set_editable.
}

LineEdit::~LineEdit() {
	TS->free_rid(text_rid);
}
